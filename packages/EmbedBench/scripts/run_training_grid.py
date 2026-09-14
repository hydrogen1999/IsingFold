#!/usr/bin/env python3
"""Run one registered training cell with resumable, test-locked artifacts.

The same entry point is used directly on Apollo and from a Slurm array on Goose.
Indices are zero based within a named stage.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

from quality_v2_paper_contract import strict_json_loads


class GridCell(NamedTuple):
    stage: str
    index: int
    cell_id: str
    params: dict[str, object]
    extra_args: tuple[str, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_provenance(inputs: Sequence[Path], split_manifest: Path) -> dict[str, object]:
    """Hash the exact corpus and split manifest consumed by one grid cell."""

    names = [path.name for path in inputs]
    if len(names) != len(set(names)):
        raise ValueError("input corpus basenames must be unique")
    split_document = strict_json_loads(
        split_manifest.read_bytes(),
        location=f"split manifest {split_manifest}",
    )
    if not isinstance(split_document, Mapping):
        raise ValueError("split manifest must be a JSON object")
    schema = split_document.get("schema")
    schema_version = split_document.get("schema_version")
    if schema is not None or schema_version is not None:
        if schema != "embedbench.split-manifest" or schema_version != 2:
            raise ValueError(
                f"unsupported split manifest schema {schema!r} version {schema_version!r}"
            )
        provenance = split_document.get("provenance")
        split_tables = split_document.get("splits")
        if not isinstance(provenance, Mapping) or not isinstance(split_tables, Mapping):
            raise ValueError("schema-v2 split manifest lacks provenance or split tables")
        declared_inputs = provenance.get("inputs")
        if not isinstance(declared_inputs, list):
            raise ValueError("schema-v2 provenance inputs must be a list")
        expected_hashes: dict[str, str] = {}
        for index, entry in enumerate(declared_inputs):
            if not isinstance(entry, Mapping):
                raise ValueError(f"schema-v2 provenance input {index} must be an object")
            filename = entry.get("file")
            expected_hash = entry.get("sha256")
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"schema-v2 provenance input {index} has no file name")
            if filename in expected_hashes:
                raise ValueError(f"duplicate schema-v2 provenance input {filename!r}")
            if (
                not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or any(character not in "0123456789abcdef" for character in expected_hash)
            ):
                raise ValueError(f"invalid SHA-256 for schema-v2 input {filename!r}")
            expected_hashes[filename] = expected_hash
        if set(expected_hashes) != set(split_tables):
            raise ValueError(
                "schema-v2 provenance inputs and split tables must cover the same files"
            )
        if set(names) != set(expected_hashes):
            missing = sorted(set(expected_hashes) - set(names))
            unexpected = sorted(set(names) - set(expected_hashes))
            raise ValueError(
                "schema-v2 manifest requires the exact corpus input set; "
                f"missing={missing}, unexpected={unexpected}"
            )
        for path in inputs:
            actual = _sha256(path)
            expected = expected_hashes[path.name]
            if actual != expected:
                raise ValueError(
                    f"SHA-256 mismatch for {path.name}: manifest records {expected}, "
                    f"but the grid corpus is {actual}"
                )
    return {
        "corpus_inputs": [{"file": path.name, "sha256": _sha256(path)} for path in inputs],
        "split_manifest": {
            "file": split_manifest.name,
            "sha256": _sha256(split_manifest),
            "schema": split_document.get("schema"),
            "schema_version": split_document.get("schema_version"),
        },
    }


def _source_sha256(root: Path) -> str:
    """Bind receipts to the staged trainer and EmbedBench Python implementation."""
    paths = sorted(
        {
            *root.glob("src/embedbench/**/*.py"),
            *root.glob("scripts/train_*.py"),
            *root.glob("scripts/training_*.py"),
            root / "scripts" / "run_training_grid.py",
        },
        key=lambda path: path.relative_to(root).as_posix(),
    )
    if not paths or any(not path.is_file() for path in paths):
        raise ValueError("staged training source tree is incomplete")
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _stage(document: Mapping[str, object], stage_name: str) -> Mapping[str, object]:
    if document.get("schema") != "embedbench.training-grid" or document.get("schema_version") != 1:
        raise ValueError("unsupported training-grid schema")
    stages = document.get("stages")
    if not isinstance(stages, Mapping) or stage_name not in stages:
        raise ValueError(f"unknown grid stage {stage_name!r}")
    stage = stages[stage_name]
    if not isinstance(stage, Mapping):
        raise ValueError(f"grid stage {stage_name!r} must be an object")
    return stage


def expand_stage(document: Mapping[str, object], stage_name: str) -> list[GridCell]:
    """Expand a stage's explicitly ordered Cartesian product."""
    stage = _stage(document, stage_name)
    axes = stage.get("axes")
    order = stage.get("axis_order")
    extra_args = stage.get("extra_args", [])
    if not isinstance(axes, Mapping) or not isinstance(order, list):
        raise ValueError(f"grid stage {stage_name!r} needs axes and axis_order")
    if set(order) != set(axes) or len(order) != len(set(order)):
        raise ValueError(f"grid stage {stage_name!r} has inconsistent axis_order")
    if not isinstance(extra_args, list) or not all(isinstance(arg, str) for arg in extra_args):
        raise ValueError(f"grid stage {stage_name!r} extra_args must be strings")
    if "--evaluate-test" in extra_args:
        raise ValueError("architecture screens must not evaluate the fixed test split")

    values: list[list[object]] = []
    for axis in order:
        axis_values = axes[axis]
        if not isinstance(axis_values, list) or not axis_values:
            raise ValueError(f"axis {axis!r} must be a non-empty list")
        values.append(axis_values)

    cells = []
    for index, combination in enumerate(itertools.product(*values)):
        params = dict(zip(order, combination, strict=True))
        suffix = "__".join(f"{axis}-{params[axis]}" for axis in order)
        cell_id = f"{stage_name}__{suffix}"
        cells.append(GridCell(stage_name, index, cell_id, params, tuple(extra_args)))
    return cells


def expected_execution_site(stage: Mapping[str, object], cell: GridCell) -> str | None:
    """Return a registered site assignment, if the stage defines one."""

    execution = stage.get("execution")
    if execution is None:
        return None
    if not isinstance(execution, Mapping):
        raise ValueError("stage execution contract must be an object")
    if execution.get("assignment") != "cell_index_modulo":
        raise ValueError("unsupported execution-site assignment")
    modulus = execution.get("modulus")
    mapping = execution.get("remainder_to_site")
    if (
        isinstance(modulus, bool)
        or not isinstance(modulus, int)
        or modulus < 1
        or not isinstance(mapping, Mapping)
        or set(mapping) != {str(index) for index in range(modulus)}
        or any(not isinstance(site, str) or not site for site in mapping.values())
    ):
        raise ValueError("invalid cell-index execution-site mapping")
    return str(mapping[str(cell.index % modulus)])


def _required_path(stage: Mapping[str, object], key: str, root: Path) -> Path:
    value = stage.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"stage field {key!r} must be a path string")
    path = root / value
    if not path.is_file():
        raise ValueError(f"required file does not exist: {path}")
    return path


def _stage_inputs(stage: Mapping[str, object], root: Path) -> tuple[tuple[Path, ...], Path]:
    split_manifest = _required_path(stage, "splits", root)
    input_pattern = stage.get("inputs")
    if not isinstance(input_pattern, str) or not input_pattern:
        raise ValueError("stage field 'inputs' must be a glob string")
    inputs = tuple(sorted(root.glob(input_pattern)))
    if not inputs:
        raise ValueError(f"input glob matched no files: {root / input_pattern}")
    return inputs, split_manifest


def _expected_checkpoint(checkpoint_prefix: Path, cell: GridCell) -> Path:
    architecture = cell.params.get("arch")
    seed = cell.params.get("seed")
    if not isinstance(architecture, str) or not architecture:
        raise ValueError("each training cell requires a non-empty architecture")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("each training cell requires one integer seed")
    return Path(f"{checkpoint_prefix}_{architecture}_s{seed}.pt")


def build_command(
    document: Mapping[str, object],
    cell: GridCell,
    *,
    root: Path,
    python: str,
    device: str,
) -> tuple[list[str], Path, Path]:
    stage = _stage(document, cell.stage)
    trainer = _required_path(stage, "trainer", root)
    inputs, split_manifest = _stage_inputs(stage, root)

    epochs = stage.get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("stage epochs must be a positive integer")
    raw_output_dir = stage.get("output_dir")
    raw_checkpoint_dir = stage.get("checkpoint_dir")
    if not isinstance(raw_output_dir, str) or not raw_output_dir:
        raise ValueError("stage output_dir must be a non-empty path string")
    if not isinstance(raw_checkpoint_dir, str) or not raw_checkpoint_dir:
        raise ValueError("stage checkpoint_dir must be a non-empty path string")
    output_dir = root / raw_output_dir
    checkpoint_dir = root / raw_checkpoint_dir
    result_path = output_dir / f"{cell.cell_id}.json"
    checkpoint_prefix = checkpoint_dir / cell.cell_id

    command = [python, str(trainer), *(str(path) for path in inputs)]
    command.extend(["--splits", str(split_manifest)])
    for axis, value in cell.params.items():
        flag = "--seeds" if axis == "seed" else f"--{axis.replace('_', '-')}"
        command.extend([flag, str(value)])
    command.extend(["--epochs", str(epochs), "--device", device])
    command.extend(["--save", str(checkpoint_prefix)])
    command.extend(cell.extra_args)
    command.extend(["--out", str(result_path)])
    return command, result_path, _expected_checkpoint(checkpoint_prefix, cell)


def _atomic_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _completed(
    receipt_path: Path,
    result_path: Path,
    checkpoint_path: Path,
    *,
    grid_sha256: str,
    source_sha256: str,
    data_provenance: Mapping[str, object],
    execution_site: str | None = None,
) -> bool:
    if not all(path.is_file() for path in (receipt_path, result_path, checkpoint_path)):
        return False
    try:
        receipt = strict_json_loads(
            receipt_path.read_bytes(),
            location=f"training receipt {receipt_path}",
        )
        if not isinstance(receipt, Mapping):
            return False
        return (
            receipt.get("schema") == "embedbench.training-receipt"
            and receipt.get("schema_version") == 1
            and receipt.get("grid_sha256") == grid_sha256
            and receipt.get("source_sha256") == source_sha256
            and receipt.get("data_provenance") == data_provenance
            and receipt.get("result_sha256") == _sha256(result_path)
            and receipt.get("checkpoint") == str(checkpoint_path)
            and receipt.get("checkpoint_sha256") == _sha256(checkpoint_path)
            and receipt.get("test_locked") is True
            and receipt.get("execution_site") == execution_site
        )
    except (OSError, ValueError):
        return False


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--index", type=int)
    parser.add_argument("--root", default=".")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--execution-site", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.root).resolve()
    grid_path = Path(args.grid).resolve()
    document = strict_json_loads(
        grid_path.read_bytes(),
        location=f"training grid {grid_path}",
    )
    cells = expand_stage(document, args.stage)
    if args.list:
        print(json.dumps([cell._asdict() for cell in cells], indent=2, sort_keys=True))
        return 0
    if args.index is None or not 0 <= args.index < len(cells):
        raise SystemExit(f"--index must be between 0 and {len(cells) - 1}")

    cell = cells[args.index]
    stage = _stage(document, cell.stage)
    expected_site = expected_execution_site(stage, cell)
    execution_site = args.execution_site or os.environ.get("ISINGFOLD_EXECUTION_SITE")
    if expected_site is not None and execution_site != expected_site:
        raise SystemExit(
            f"grid cell {cell.index} is assigned to {expected_site!r}, not {execution_site!r}"
        )
    command, result_path, checkpoint_path = build_command(
        document,
        cell,
        root=root,
        python=args.python,
        device=args.device,
    )
    inputs, split_manifest = _stage_inputs(stage, root)
    data_provenance = _data_provenance(inputs, split_manifest)
    grid_sha256 = _sha256(grid_path)
    source_sha256 = _source_sha256(root)
    receipt_path = result_path.with_suffix(".receipt.json")
    if not args.force and _completed(
        receipt_path,
        result_path,
        checkpoint_path,
        grid_sha256=grid_sha256,
        source_sha256=source_sha256,
        data_provenance=data_provenance,
        execution_site=execution_site,
    ):
        print(f"skip completed cell {cell.cell_id}")
        return 0
    print(json.dumps({"cell": cell._asdict(), "command": command}, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    result_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    subprocess.run(command, cwd=root, check=True)
    if _data_provenance(inputs, split_manifest) != data_provenance:
        raise RuntimeError("training corpus or split manifest changed while the cell was running")
    result = strict_json_loads(
        result_path.read_bytes(),
        location=f"training result {result_path}",
    )
    if not isinstance(result, Mapping):
        raise RuntimeError("a registered cell must emit a JSON object")
    rows = result.get("results")
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError("a registered cell must emit exactly one seed result")
    if not isinstance(rows[0], Mapping):
        raise RuntimeError("a registered cell seed result must be a JSON object")
    if rows[0].get("test_evaluated") is not False or rows[0].get("test") is not None:
        raise RuntimeError("architecture-screen cell unexpectedly evaluated the test split")
    if result.get("data_provenance") != data_provenance:
        raise RuntimeError("trainer result does not match the registered data provenance")
    if not checkpoint_path.is_file():
        raise RuntimeError(f"trainer did not emit the expected checkpoint: {checkpoint_path}")
    checkpoint_sha256 = _sha256(checkpoint_path)

    receipt = {
        "schema": "embedbench.training-receipt",
        "schema_version": 1,
        "grid_id": document.get("grid_id"),
        "grid_sha256": grid_sha256,
        "source_sha256": source_sha256,
        "data_provenance": data_provenance,
        "cell": cell._asdict(),
        "command": command,
        "hostname": socket.gethostname(),
        "execution_site": execution_site,
        "slurm": {
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "started_unix": started,
        "finished_unix": time.time(),
        "result": str(result_path),
        "result_sha256": _sha256(result_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "test_locked": True,
    }
    _atomic_json(receipt_path, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
