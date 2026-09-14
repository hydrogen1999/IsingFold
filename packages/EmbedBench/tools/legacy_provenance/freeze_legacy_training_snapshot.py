#!/usr/bin/env python3
"""Reconstruct the immutable 80-cell Quality V2 legacy training snapshot.

This verifier never decodes corpus records, evaluates models, or opens a test-label artifact.
It captures checkpoint and model-source bytes by digest, then loads only those captured bytes
in an isolated semantic validator. It is independent of mutable selection and paper-audit code.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

REGISTERED_LEGACY_SOURCE_SHA256 = "294caa56d0bb04f716163b143a995ee176c8e72dd209c5f5f77caef92d72a264"
REGISTERED_GRID_SHA256 = "10be13fe54669e3577c9c6a32f972d1cd94640f091bb262b03f302bd59bdc270"
REGISTERED_QUALITY_INPUTS: dict[str, dict[str, object]] = {
    "quality_chimera5_app.jsonl": {
        "bytes": 347753,
        "sha256": "8b11660c068b626e4eae55c6d8d3ed31fdd335a48ebd682bd06727cc99b5410e",
    },
    "quality_chimera5_inkdrop.jsonl": {
        "bytes": 1825147,
        "sha256": "15cc622b3d8d4cb4fd66a378a83d2d15a886f076344817515d2c250bedaa2cce",
    },
    "quality_chimera5_random.jsonl": {
        "bytes": 167045,
        "sha256": "1a4403535fa4d3ecf15c4c9d89d60456b1bba2d0bf85f2012af9b9090d196368",
    },
    "quality_pegasus3_app.jsonl": {
        "bytes": 1146306,
        "sha256": "f1d911990fed7773f624dcf1badf2c806c7a3d42a0fb1c33aeb246ff56a9e9ed",
    },
    "quality_pegasus3_inkdrop.jsonl": {
        "bytes": 4237876,
        "sha256": "d2ffc17be0ab3770d24436d1e28335ed91711971303c35200183783bc4b9dfb3",
    },
    "quality_pegasus3_random.jsonl": {
        "bytes": 472656,
        "sha256": "07871cdae0d8fb2858bc5c21af14dbee1c7e59807f724c54df52131d9dd1cd32",
    },
    "quality_zephyr2_app.jsonl": {
        "bytes": 1165152,
        "sha256": "3bd996cef358e4b924bdc3d051e5815af34ec168b55af8abdd4db5fb4280e48d",
    },
    "quality_zephyr2_inkdrop.jsonl": {
        "bytes": 4755319,
        "sha256": "b1b62bcf1a8637eb9bbb2ed2aefddbffb387fb451dc6145e37fdc1331e91cd7a",
    },
    "quality_zephyr2_random.jsonl": {
        "bytes": 393094,
        "sha256": "3b82118599bfb10fe3300db11504d0dd15577574c12d08724b6d667c37278aca",
    },
}
REGISTERED_QUALITY_SPLIT_SHA256 = "cf3b7b44a9d3e95e1dc82b11870a9056492d4d3d7029f181e7737d07906a78f0"
REGISTERED_GRID_RELATIVE_PATH = Path("configs/training_grid_quality_v2.json")
REGISTERED_STAGE = "quality_value_v2_screen"
REGISTERED_ARCHITECTURES = ("mpnn", "gin", "gatv2", "gps", "hetero")
REGISTERED_OBJECTIVES = ("p_only", "p_connectivity", "p_robustness", "full")
REGISTERED_SEEDS = (0, 1, 2, 3)
EXPECTED_CELL_COUNT = 80
SNAPSHOT_SCHEMA = "embedbench.legacy-training-snapshot"
SNAPSHOT_SCHEMA_VERSION = 1
SOURCE_DIGEST_ALGORITHM = "legacy-length-framed-path-and-payload-sha256-v1"
SNAPSHOT_MANIFEST_ALGORITHM = "canonical-relative-path-file-sha256-inventory-v1"
DEPENDENCY_LOCK_CANDIDATES = (
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
    "requirements.lock",
    "conda-lock.yml",
)
REGISTERED_SITES = ("apollo", "goose")
REGISTERED_GOOSE_ARRAY_TASK_COUNT = 2
REGISTERED_GOOSE_FIRST_INDEX = 1
REGISTERED_GOOSE_INDEX_STEP = 2
REGISTERED_LAUNCH_WRAPPER = Path("scripts/goose_training_grid.sbatch")
REGISTERED_GOOSE_LAUNCH_WRAPPER_SHA256 = (
    "da45ee57952ecc2d6ffce15dda079c4d991863530cbcf6e893c232eeba9b6850"
)
REGISTERED_APOLLO_DIRECT_LAUNCHER = Path("scripts/run_training_grid.py")
REGISTERED_BUDGET_RATIOS = (1.0, 1.1, 1.25, 1.5, None)
REGISTERED_PRIMARY_METRIC = "mean_finite_budget_regret"
REGISTERED_SELECTION_CONTRACT = "quality_mean_or_lcb_subject_to_exact_feasibility_and_budget-v1"
REGISTERED_FORWARD_INPUTS = (
    "x",
    "adjacency",
    "candidate_masks",
    "candidate_features",
    "hamiltonian_context",
)
REGISTERED_ROBUSTNESS_NAMES = (
    "minimum_logical_contact_count",
    "mean_logical_contact_count",
    "total_logical_contact_count",
    "candidate_chain_cycle_rank",
)
REGISTERED_RUNTIME_PACKAGES = (
    "embedbench",
    "numpy",
    "scipy",
    "networkx",
    "dimod",
    "dwave-samplers",
    "dwave-networkx",
    "minorminer",
)
TRAINER_ARG_DEFAULTS: dict[str, object] = {
    "hidden": 64,
    "layers": 3,
    "heads": 4,
    "lr": 1e-3,
    "tol": 0.05,
    "rank_margin": 0.05,
    "stage1_rank_margin": 0.10,
    "lcb_z": 1.0,
    "stage1_weight": 0.25,
    "lambda_connectivity": 0.1,
    "lambda_robustness": 0.1,
    "limit": None,
    "neighbour_feats": False,
    "deploy_view": False,
    "deploy_max_free": 8,
    "evaluation_support": "full",
}
TRAINER_VALUE_OPTIONS: dict[str, tuple[str, type]] = {
    "--hidden": ("hidden", int),
    "--layers": ("layers", int),
    "--heads": ("heads", int),
    "--lr": ("lr", float),
    "--tol": ("tol", float),
    "--rank-margin": ("rank_margin", float),
    "--stage1-rank-margin": ("stage1_rank_margin", float),
    "--lcb-z": ("lcb_z", float),
    "--stage1-weight": ("stage1_weight", float),
    "--lambda-connectivity": ("lambda_connectivity", float),
    "--lambda-robustness": ("lambda_robustness", float),
    "--limit": ("limit", int),
    "--deploy-max-free": ("deploy_max_free", int),
    "--evaluation-support": ("evaluation_support", str),
}
TRAINER_BOOLEAN_OPTIONS = {
    "--neighbour-feats": "neighbour_feats",
    "--deploy-view": "deploy_view",
}
RECEIPT_REQUIRED_FIELDS = {
    "schema",
    "schema_version",
    "grid_id",
    "grid_sha256",
    "source_sha256",
    "data_provenance",
    "cell",
    "command",
    "hostname",
    "execution_site",
    "slurm",
    "started_unix",
    "finished_unix",
    "result",
    "result_sha256",
    "checkpoint",
    "checkpoint_sha256",
    "test_locked",
}
RECEIPT_OPTIONAL_ENVIRONMENT_FIELDS = {
    "dependency_lock_sha256",
    "environment_sha256",
    "launch_wrapper_sha256",
}
RESULT_FIELDS = {
    "artifact_schema",
    "artifact_schema_version",
    "args",
    "device",
    "split_sha256",
    "data_provenance",
    "runtime_provenance",
    "preprocessing",
    "deployment_host_contract",
    "training_scope",
    "test_locked",
    "test_partition_encoded",
    "results",
}
RESULT_ROW_FIELDS = {
    "arch",
    "objective_variant",
    "seed",
    "training_run_id",
    "device",
    "hidden",
    "layers",
    "heads",
    "params",
    "best_epoch",
    "n_train",
    "n_val",
    "n_test",
    "preprocessing",
    "deployment_host_contract",
    "training_scope",
    "evaluation_support",
    "registered_budget_ratios",
    "primary_validation_metric",
    "lcb_z",
    "lcb_is_secondary_until_calibrated",
    "ranking_thresholds",
    "selection_contract",
    "auxiliary_heads_used_for_selection",
    "validation",
    "test_evaluated",
    "test_partition_encoded",
    "test",
    "seconds",
    "loss_weights",
    "best_epoch_training_losses",
}
TRAINING_LOSS_FIELDS = {
    "loss",
    "quality_probability",
    "within_state_rank",
    "future_capacity",
    "robustness",
    "terminal_qubits",
}


class GridCell(NamedTuple):
    stage: str
    index: int
    cell_id: str
    params: dict[str, object]
    extra_args: tuple[str, ...]


def _sha256(path: Path) -> str:
    return str(_file_evidence(path, allow_symlink=True)["file_sha256"])


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant {value!r}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number {value!r}")
    return parsed


def _parse_exact_json_int(value: str) -> int:
    return int(value, 10)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    document: dict[str, object] = {}
    for key, value in pairs:
        if key in document:
            raise ValueError(f"duplicate JSON key {key!r}")
        document[key] = value
    return document


def _strict_json(payload: bytes, *, kind: str, path: Path) -> dict[str, object]:
    try:
        document = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
            parse_float=_parse_finite_json_float,
            parse_int=_parse_exact_json_int,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        message = str(error)
        if any(
            marker in message
            for marker in (
                "duplicate JSON key",
                "non-finite JSON constant",
                "non-finite JSON number",
            )
        ):
            raise ValueError(message) from error
        raise ValueError(f"invalid {kind}: {path}") from error
    if not isinstance(document, dict):
        raise ValueError(f"{kind} must be a JSON object: {path}")
    return document


def _file_evidence(
    path: Path,
    *,
    allow_symlink: bool,
    root: Path | None = None,
) -> dict[str, object]:
    """Hash one stable regular-file view and reject unsafe path substitution."""

    if not allow_symlink:
        if root is not None and _traverses_symlink(path, root):
            raise ValueError(f"unsafe symlink substitute: {path}")
        if root is None and path.is_symlink():
            raise ValueError(f"unsafe symlink substitute: {path}")
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"expected regular file: {path}")
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
            after = os.fstat(stream.fileno())
    except OSError as error:
        raise ValueError(f"cannot read stable file: {path}") from error
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ValueError(f"file changed while being hashed: {path}")
    if size != after.st_size:
        raise ValueError(f"file size changed while being hashed: {path}")
    if not allow_symlink:
        if root is not None and _traverses_symlink(path, root):
            raise ValueError(f"unsafe symlink substitute: {path}")
        if root is None and path.is_symlink():
            raise ValueError(f"unsafe symlink substitute: {path}")
    return {"file_sha256": digest.hexdigest(), "bytes": size}


def _capture_verified_file(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    root: Path,
) -> None:
    """Copy and verify one regular file through the same open descriptor."""

    if _traverses_symlink(source, root):
        raise ValueError(f"cannot capture a symlink substitute: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    try:
        with source.open("rb") as reader, destination.open("xb") as writer:
            before = os.fstat(reader.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"expected regular file while capturing: {source}")
            for chunk in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
            after = os.fstat(reader.fileno())
    except OSError as error:
        raise ValueError(f"cannot capture stable file: {source}") from error
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise ValueError(f"file changed while being captured: {source}")
    if _traverses_symlink(source, root):
        raise ValueError(f"captured path became a symlink substitute: {source}")
    if size != expected_bytes or digest.hexdigest() != expected_sha256:
        raise ValueError(f"captured file disagrees with its inventory bytes: {source}")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_object(path: Path, kind: str) -> dict[str, object]:
    evidence = _file_evidence(path, allow_symlink=False)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"invalid {kind}: {path}") from error
    if (
        len(payload) != evidence["bytes"]
        or hashlib.sha256(payload).hexdigest() != evidence["file_sha256"]
    ):
        raise ValueError(f"{kind} changed while being read: {path}")
    return _strict_json(payload, kind=kind, path=path)


def _read_json_evidence(
    path: Path,
    *,
    kind: str,
    relative_path: str,
    role: str,
    root: Path | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    file_evidence = _file_evidence(path, allow_symlink=False, root=root)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"invalid {kind}: {path}") from error
    if (
        len(payload) != file_evidence["bytes"]
        or hashlib.sha256(payload).hexdigest() != file_evidence["file_sha256"]
    ):
        raise ValueError(f"{kind} changed while being read: {path}")
    document = _strict_json(payload, kind=kind, path=path)
    evidence = {
        "relative_path": relative_path,
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "role": role,
    }
    return document, evidence


def _relative_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field} must be a canonical relative path")
    return path


def _inventory_entry(
    path: Path,
    relative_path: str,
    role: str,
    *,
    allow_symlink: bool = True,
    root: Path | None = None,
) -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"missing staged {role}: {relative_path}")
    evidence = _file_evidence(path, allow_symlink=allow_symlink, root=root)
    return {
        "relative_path": relative_path,
        "file_sha256": evidence["file_sha256"],
        "bytes": evidence["bytes"],
        "role": role,
    }


def _traverses_symlink(path: Path, root: Path) -> bool:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return True
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            return True
    return False


def inventory_sha256(inventory: object) -> str:
    """Hash only the registered sorted path/hash pairs, not the legacy framing."""

    if not isinstance(inventory, Sequence) or isinstance(inventory, (str, bytes)):
        raise ValueError("file inventory must be a sequence")
    pairs: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, entry in enumerate(inventory):
        if not isinstance(entry, Mapping):
            raise ValueError(f"file inventory entry {index} must be an object")
        relative_path = entry.get("relative_path")
        file_sha256 = entry.get("file_sha256")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"file inventory entry {index} has no relative path")
        _relative_path(relative_path, field=f"file inventory entry {index}")
        if relative_path in seen:
            raise ValueError(f"duplicate file inventory path: {relative_path}")
        if not _is_sha256(file_sha256):
            raise ValueError(f"file inventory entry {index} has invalid SHA-256")
        seen.add(relative_path)
        pairs.append({"relative_path": relative_path, "file_sha256": str(file_sha256)})
    pairs.sort(key=lambda entry: entry["relative_path"])
    return hashlib.sha256(_canonical_bytes(pairs)).hexdigest()


def _source_paths(root: Path) -> list[Path]:
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
        raise ValueError(f"staged training source tree is incomplete: {root}")
    symlinks = [
        path.relative_to(root).as_posix() for path in paths if _traverses_symlink(path, root)
    ]
    if symlinks:
        raise ValueError(f"legacy source files must not be symlink substitutes: {symlinks}")
    return paths


def legacy_source_inventory(root: str | Path) -> tuple[str, list[dict[str, object]]]:
    """Reproduce ``run_training_grid._source_sha256`` byte for byte."""

    staged_root = Path(root).resolve()
    digest = hashlib.sha256()
    inventory: list[dict[str, object]] = []
    for path in _source_paths(staged_root):
        relative = path.relative_to(staged_root).as_posix()
        relative_bytes = relative.encode("utf-8")
        evidence = _file_evidence(path, allow_symlink=False, root=staged_root)
        payload = path.read_bytes()
        if (
            len(payload) != evidence["bytes"]
            or hashlib.sha256(payload).hexdigest() != evidence["file_sha256"]
        ):
            raise ValueError(f"legacy source changed while being read: {relative}")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
        inventory.append(
            {
                "relative_path": relative,
                "file_sha256": evidence["file_sha256"],
                "bytes": evidence["bytes"],
                "role": "legacy_training_source",
            }
        )
    return digest.hexdigest(), inventory


def _stage(document: Mapping[str, object]) -> Mapping[str, object]:
    if not _same_typed_value(document.get("schema"), "embedbench.training-grid") or not (
        _same_typed_value(document.get("schema_version"), 1)
    ):
        raise ValueError("unsupported registered training-grid schema")
    stages = document.get("stages")
    stage = stages.get(REGISTERED_STAGE) if isinstance(stages, Mapping) else None
    if not isinstance(stage, Mapping):
        raise ValueError(f"registered grid has no {REGISTERED_STAGE!r} stage")
    return stage


def _expand_registered_grid(document: Mapping[str, object]) -> list[GridCell]:
    stage = _stage(document)
    axes = stage.get("axes")
    order = stage.get("axis_order")
    extra_args = stage.get("extra_args")
    if not isinstance(axes, Mapping) or order != ["objective_variant", "arch", "seed"]:
        raise ValueError("registered grid axis order changed")
    expected_axes = {
        "objective_variant": list(REGISTERED_OBJECTIVES),
        "arch": list(REGISTERED_ARCHITECTURES),
        "seed": list(REGISTERED_SEEDS),
    }
    if dict(axes) != expected_axes:
        raise ValueError("registered grid architecture, objective, or seed axes changed")
    if not isinstance(extra_args, list) or not all(
        isinstance(argument, str) for argument in extra_args
    ):
        raise ValueError("registered grid extra_args must be strings")
    if "--evaluate-test" in extra_args:
        raise ValueError("legacy architecture screen must keep test evaluation disabled")

    cells: list[GridCell] = []
    values = [axes[axis] for axis in order]
    for index, combination in enumerate(itertools.product(*values)):
        params = dict(zip(order, combination, strict=True))
        suffix = "__".join(f"{axis}-{params[axis]}" for axis in order)
        cell_id = f"{REGISTERED_STAGE}__{suffix}"
        cells.append(GridCell(REGISTERED_STAGE, index, cell_id, params, tuple(extra_args)))
    if len(cells) != EXPECTED_CELL_COUNT or len({cell.cell_id for cell in cells}) != len(cells):
        raise ValueError("registered grid must expand to exactly 80 unique cells")
    selection = document.get("selection")
    if not isinstance(selection, Mapping) or selection.get("seeds") != list(REGISTERED_SEEDS):
        raise ValueError("registered selection seeds changed")
    return cells


def _expected_site(stage: Mapping[str, object], cell: GridCell) -> str:
    execution = stage.get("execution")
    if not isinstance(execution, Mapping):
        raise ValueError("registered grid has no execution contract")
    if execution.get("assignment") != "cell_index_modulo":
        raise ValueError("registered execution assignment changed")
    modulus = execution.get("modulus")
    mapping = execution.get("remainder_to_site")
    if (
        isinstance(modulus, bool)
        or not isinstance(modulus, int)
        or modulus < 1
        or not isinstance(mapping, Mapping)
        or set(mapping) != {str(index) for index in range(modulus)}
    ):
        raise ValueError("registered execution-site mapping is invalid")
    site = mapping.get(str(cell.index % modulus))
    if not isinstance(site, str) or not site:
        raise ValueError("registered execution site must be a non-empty string")
    return site


def _grid_path(stage_root: Path) -> Path:
    return stage_root / REGISTERED_GRID_RELATIVE_PATH


def _data_provenance(
    root: Path, stage: Mapping[str, object]
) -> tuple[dict[str, object], list[dict[str, object]], tuple[Path, ...], Path]:
    input_pattern = stage.get("inputs")
    if not isinstance(input_pattern, str) or not input_pattern:
        raise ValueError("registered stage inputs must be a glob")
    _relative_path(input_pattern, field="stage inputs")
    inputs = tuple(sorted(root.glob(input_pattern)))
    if not inputs or any(not path.is_file() for path in inputs):
        raise ValueError(f"registered corpus input set is incomplete in {root}")
    names = [path.name for path in inputs]
    if len(names) != len(set(names)):
        raise ValueError("registered corpus basenames must be unique")

    split_relative = _relative_path(stage.get("splits"), field="stage splits")
    split_path = root / split_relative
    if _traverses_symlink(split_path, root):
        raise ValueError("split manifest must not traverse a symlink substitute")
    split_document, split_entry = _read_json_evidence(
        split_path,
        kind="split manifest",
        relative_path=split_relative.as_posix(),
        role="legacy_split_manifest",
        root=root,
    )
    if not _same_typed_value(
        split_document.get("schema"), "embedbench.split-manifest"
    ) or not _same_typed_value(split_document.get("schema_version"), 2):
        raise ValueError("legacy snapshot requires split-manifest schema version 2")

    input_entries = [
        _inventory_entry(path, path.relative_to(root).as_posix(), "legacy_corpus_input")
        for path in inputs
    ]
    corpus = [
        {"file": path.name, "sha256": entry["file_sha256"]}
        for path, entry in zip(inputs, input_entries, strict=True)
    ]
    provenance = split_document.get("provenance")
    split_tables = split_document.get("splits")
    declared = provenance.get("inputs") if isinstance(provenance, Mapping) else None
    if not isinstance(declared, list) or not isinstance(split_tables, Mapping):
        raise ValueError("split manifest lacks provenance or split tables")
    expected: dict[str, str] = {}
    for entry in declared:
        if not isinstance(entry, Mapping):
            raise ValueError("split provenance input must be an object")
        filename = entry.get("file")
        file_sha256 = entry.get("sha256")
        if not isinstance(filename, str) or filename in expected or not _is_sha256(file_sha256):
            raise ValueError("split provenance input is duplicate or invalid")
        expected[filename] = str(file_sha256)
    actual = {str(entry["file"]): str(entry["sha256"]) for entry in corpus}
    if set(expected) != set(split_tables) or actual != expected:
        raise ValueError("staged corpus bytes disagree with split data provenance")
    registered_hashes = {
        filename: str(binding["sha256"]) for filename, binding in REGISTERED_QUALITY_INPUTS.items()
    }
    registered_sizes = {
        filename: int(binding["bytes"]) for filename, binding in REGISTERED_QUALITY_INPUTS.items()
    }
    actual_sizes = {
        path.name: int(entry["bytes"]) for path, entry in zip(inputs, input_entries, strict=True)
    }
    if actual != registered_hashes or actual_sizes != registered_sizes:
        raise ValueError("staged bytes do not match the registered immutable corpus")
    split_sha256 = str(split_entry["file_sha256"])
    if split_sha256 != REGISTERED_QUALITY_SPLIT_SHA256:
        raise ValueError("staged split does not match the registered immutable split")

    document = {
        "corpus_inputs": corpus,
        "split_manifest": {
            "file": split_path.name,
            "sha256": split_sha256,
            "schema": split_document.get("schema"),
            "schema_version": split_document.get("schema_version"),
        },
    }
    inventory = input_entries
    inventory.append(split_entry)
    return document, inventory, inputs, split_path


def _dependency_provenance(
    root: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    pyproject = root / "pyproject.toml"
    if not pyproject.is_file() or _traverses_symlink(pyproject, root):
        raise ValueError("staged pyproject.toml is missing or is a symlink substitute")
    inventory = [
        _inventory_entry(
            pyproject,
            "pyproject.toml",
            "dependency_specification",
            allow_symlink=False,
            root=root,
        )
    ]
    locks: list[dict[str, object]] = []
    for relative in DEPENDENCY_LOCK_CANDIDATES:
        path = root / relative
        if path.exists():
            if not path.is_file() or _traverses_symlink(path, root):
                raise ValueError(f"dependency lock is not a regular staged file: {relative}")
            entry = _inventory_entry(
                path,
                relative,
                "dependency_lock",
                allow_symlink=False,
                root=root,
            )
            inventory.append(entry)
            locks.append(
                {
                    "relative_path": relative,
                    "file_sha256": entry["file_sha256"],
                    "bytes": entry["bytes"],
                }
            )
    if len(locks) > 1:
        raise ValueError("staged root has multiple ambiguous dependency lock files")
    specification = inventory[0]
    lock = locks[0] if locks else None
    return (
        {
            "staged_lock_status": "present" if lock is not None else "absent",
            "staged_locks": locks,
            "dependency_lock_relative_path": (lock["relative_path"] if lock is not None else None),
            "dependency_lock_sha256": lock["file_sha256"] if lock is not None else None,
            "pyproject": {
                "relative_path": "pyproject.toml",
                "file_sha256": specification["file_sha256"],
                "bytes": specification["bytes"],
            },
        },
        inventory,
    )


def _launch_provenance(
    root: Path, *, site: str
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if site == "apollo":
        launcher = root / REGISTERED_APOLLO_DIRECT_LAUNCHER
        if not launcher.is_file() or _traverses_symlink(launcher, root):
            raise ValueError("staged Apollo direct launcher is missing or substituted")
        evidence = _file_evidence(launcher, allow_symlink=False, root=root)
        return (
            {
                "mode": "direct_python",
                "launcher_relative_path": REGISTERED_APOLLO_DIRECT_LAUNCHER.as_posix(),
                "launcher_sha256": evidence["file_sha256"],
                "launcher_bound_by_legacy_source_digest": True,
                "shell_launch_command_attested": False,
                "resource_allocation_attested": False,
            },
            [],
        )
    if site != "goose":
        raise ValueError(f"unsupported registered launch site {site!r}")
    wrapper = root / REGISTERED_LAUNCH_WRAPPER
    if not wrapper.is_file() or _traverses_symlink(wrapper, root):
        raise ValueError(f"staged Goose launch wrapper is missing or substituted: {wrapper}")
    entry = _inventory_entry(
        wrapper,
        REGISTERED_LAUNCH_WRAPPER.as_posix(),
        "legacy_goose_launch_wrapper",
        allow_symlink=False,
        root=root,
    )
    if entry["file_sha256"] != REGISTERED_GOOSE_LAUNCH_WRAPPER_SHA256:
        raise ValueError("staged wrapper does not match the registered Goose wrapper SHA-256")
    return (
        {
            "mode": "slurm_wrapper",
            "wrapper_relative_path": REGISTERED_LAUNCH_WRAPPER.as_posix(),
            "wrapper_sha256": entry["file_sha256"],
            "array_task_count_expected_by_wrapper": REGISTERED_GOOSE_ARRAY_TASK_COUNT,
            "first_global_cell_index_expected_by_protocol": REGISTERED_GOOSE_FIRST_INDEX,
            "global_cell_step_expected_by_protocol": REGISTERED_GOOSE_INDEX_STEP,
            "slurm_submission_command_attested": False,
            "slurm_exported_environment_attested": False,
            "slurm_array_shape_attested": False,
            "slurm_resource_request_attested": False,
        },
        [entry],
    )


def _paths_for_cell(
    root: Path, stage: Mapping[str, object], cell: GridCell
) -> tuple[Path, Path, Path, Path]:
    output_relative = _relative_path(stage.get("output_dir"), field="stage output_dir")
    checkpoint_relative = _relative_path(stage.get("checkpoint_dir"), field="stage checkpoint_dir")
    result = root / output_relative / f"{cell.cell_id}.json"
    receipt = result.with_suffix(".receipt.json")
    architecture = cell.params["arch"]
    seed = cell.params["seed"]
    prefix = root / checkpoint_relative / cell.cell_id
    checkpoint = Path(f"{prefix}_{architecture}_s{seed}.pt")
    return result, receipt, checkpoint, prefix


def _canonical_absolute_path(value: object, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value)
    if (
        not path.is_absolute()
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{field} must be a canonical absolute path")
    return path


def _historical_root_from_path(value: object, relative: Path, *, field: str) -> Path:
    actual = _canonical_absolute_path(value, field=field)
    relative_parts = relative.parts
    if (
        not relative_parts
        or len(actual.parts) <= len(relative_parts)
        or tuple(actual.parts[-len(relative_parts) :]) != relative_parts
    ):
        raise ValueError(f"{field} does not identify the registered historical root")
    historical_root = Path(*actual.parts[: -len(relative_parts)])
    if not historical_root.is_absolute():
        raise ValueError(f"{field} has no canonical historical root")
    return historical_root


def _require_historical_path(
    value: object, *, historical_root: Path, relative: Path, field: str
) -> None:
    actual = _canonical_absolute_path(value, field=field)
    expected = historical_root / relative
    if actual != expected:
        raise ValueError(f"{field} changed historical root or logical path")


def _registered_trainer_args(stage: Mapping[str, object]) -> dict[str, object]:
    raw = stage.get("extra_args")
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        raise ValueError("registered grid extra_args must be strings")
    parsed = dict(TRAINER_ARG_DEFAULTS)
    seen: set[str] = set()
    index = 0
    while index < len(raw):
        option = raw[index]
        if option in seen:
            raise ValueError(f"registered trainer repeats option {option}")
        seen.add(option)
        destination = TRAINER_BOOLEAN_OPTIONS.get(option)
        if destination is not None:
            parsed[destination] = True
            index += 1
            continue
        specification = TRAINER_VALUE_OPTIONS.get(option)
        if specification is None or index + 1 >= len(raw):
            raise ValueError(f"unsupported or incomplete registered trainer option {option}")
        destination, converter = specification
        try:
            parsed[destination] = converter(raw[index + 1])
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid value for registered trainer option {option}") from error
        index += 2
    for name, value in parsed.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"registered trainer argument {name} must be finite")
    if parsed["limit"] is not None:
        raise ValueError("registered legacy screen must not use a corpus limit")
    return parsed


def _runtime_contract(
    value: object, *, required_device: str, cell_id: str
) -> tuple[dict[str, object], str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"cell {cell_id} has no runtime provenance")
    python = value.get("python")
    torch = value.get("torch")
    device = value.get("device")
    packages = value.get("packages")
    if (
        not _same_typed_value(value.get("schema"), "embedbench.runtime-environment")
        or not _same_typed_value(value.get("schema_version"), 1)
        or set(value) != {"schema", "schema_version", "python", "torch", "device", "packages"}
        or not isinstance(python, dict)
        or set(python) != {"implementation", "version", "executable"}
        or not isinstance(torch, dict)
        or set(torch) != {"version", "cuda_available", "cuda_runtime_version", "cudnn_version"}
        or not isinstance(device, dict)
        or set(device) != {"selected", "type", "index", "gpu"}
        or not isinstance(packages, dict)
        or set(packages) != set(REGISTERED_RUNTIME_PACKAGES)
    ):
        raise ValueError(f"cell {cell_id} has invalid runtime provenance")
    implementation = python.get("implementation")
    version = python.get("version")
    executable = python.get("executable")
    if not all(isinstance(item, str) and item for item in (implementation, version, executable)):
        raise ValueError(f"cell {cell_id} has incomplete Python runtime provenance")
    _canonical_absolute_path(executable, field=f"cell {cell_id} runtime interpreter")
    if not _same_typed_value(device.get("type"), required_device) or not _same_typed_value(
        device.get("selected"), required_device
    ):
        raise ValueError(f"cell {cell_id} did not use the registered device type")
    if required_device == "cuda" and torch.get("cuda_available") is not True:
        raise ValueError(f"cell {cell_id} runtime did not expose CUDA")
    device_index = device.get("index")
    gpu = device.get("gpu")
    if (
        isinstance(device_index, bool)
        or not isinstance(device_index, int)
        or device_index < 0
        or not isinstance(gpu, dict)
        or set(gpu) != {"name", "compute_capability", "total_memory_bytes"}
        or not isinstance(gpu.get("name"), str)
        or not gpu["name"]
    ):
        raise ValueError(f"cell {cell_id} has invalid CUDA device provenance")
    capability = gpu.get("compute_capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in capability
        )
        or isinstance(gpu.get("total_memory_bytes"), bool)
        or not isinstance(gpu.get("total_memory_bytes"), int)
        or gpu["total_memory_bytes"] < 1
    ):
        raise ValueError(f"cell {cell_id} has invalid CUDA hardware provenance")
    if (
        not isinstance(torch.get("version"), str)
        or not torch["version"]
        or not isinstance(torch.get("cuda_runtime_version"), str)
        or not torch["cuda_runtime_version"]
        or isinstance(torch.get("cudnn_version"), bool)
        or not isinstance(torch.get("cudnn_version"), int)
        or torch["cudnn_version"] < 1
    ):
        raise ValueError(f"cell {cell_id} has incomplete CUDA runtime provenance")
    package_versions: dict[str, str] = {}
    for name, package_version in packages.items():
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(package_version, str)
            or not package_version
        ):
            raise ValueError(f"cell {cell_id} has invalid installed-package provenance")
        package_versions[name] = package_version
    signature = {
        "python_implementation": implementation,
        "python_version": version,
        "python_executable": executable,
        "torch_version": torch.get("version"),
        "cuda_runtime_version": torch.get("cuda_runtime_version"),
        "cudnn_version": torch.get("cudnn_version"),
        "device_type": device.get("type"),
        "packages": dict(sorted(package_versions.items())),
    }
    environment_sha256 = hashlib.sha256(_canonical_bytes(value)).hexdigest()
    return signature, environment_sha256, str(executable)


def _validate_command(
    command: object,
    *,
    root: Path,
    stage: Mapping[str, object],
    cell: GridCell,
    inputs: Sequence[Path],
    split_path: Path,
    result_path: Path,
    checkpoint_prefix: Path,
    python_executable: str,
) -> tuple[list[str], Path]:
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(token, str) and token for token in command)
    ):
        raise ValueError(f"cell {cell.cell_id} has an invalid launch command")
    if command[0] != python_executable:
        raise ValueError(f"cell {cell.cell_id} command interpreter changed")
    trainer_relative = _relative_path(stage.get("trainer"), field="stage trainer")
    historical_root = _historical_root_from_path(
        command[1], trainer_relative, field=f"cell {cell.cell_id} trainer"
    )
    execution = stage.get("execution")
    required_device = (
        execution.get("required_device_type") if isinstance(execution, Mapping) else None
    )
    if required_device not in {"cpu", "cuda"}:
        raise ValueError("registered grid has no required device type")
    epochs = stage.get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("registered grid epochs must be positive")

    expected: list[tuple[str, Path | None]] = [(python_executable, None), ("", trainer_relative)]
    expected.extend(("", path.relative_to(root)) for path in inputs)
    expected.extend((("--splits", None), ("", split_path.relative_to(root))))
    order = stage.get("axis_order")
    if not isinstance(order, list):
        raise ValueError("registered grid axis order is invalid")
    for axis in order:
        flag = "--seeds" if axis == "seed" else f"--{axis.replace('_', '-')}"
        expected.extend(((flag, None), (str(cell.params[axis]), None)))
    expected.extend(
        (
            ("--epochs", None),
            (str(epochs), None),
            ("--device", None),
            (str(required_device), None),
            ("--save", None),
            ("", checkpoint_prefix.relative_to(root)),
        )
    )
    expected.extend((argument, None) for argument in cell.extra_args)
    expected.extend((("--out", None), ("", result_path.relative_to(root))))
    if len(command) != len(expected):
        raise ValueError(f"cell {cell.cell_id} launch command length changed")
    for index, (actual, (registered, path)) in enumerate(zip(command, expected, strict=True)):
        if path is not None:
            try:
                _require_historical_path(
                    actual,
                    historical_root=historical_root,
                    relative=path,
                    field=f"cell {cell.cell_id} command argv {index}",
                )
            except ValueError as error:
                raise ValueError(
                    f"cell {cell.cell_id} launch command changed at semantic argv {index}: {error}"
                ) from error
        elif actual != registered:
            raise ValueError(f"cell {cell.cell_id} launch command changed at semantic argv {index}")
    return list(command), historical_root


def _validate_slurm(
    receipt: Mapping[str, object],
    *,
    stage: Mapping[str, object],
    cell: GridCell,
    site: str,
) -> dict[str, object]:
    execution = stage.get("execution")
    slurm = receipt.get("slurm")
    if not isinstance(execution, Mapping) or not isinstance(slurm, Mapping):
        raise ValueError(f"cell {cell.cell_id} has incomplete execution provenance")
    fields = ("array_job_id", "array_task_id", "job_id")
    required = execution.get("slurm_required_sites", [])
    forbidden = execution.get("slurm_forbidden_sites", [])
    if not isinstance(required, list) or not isinstance(forbidden, list):
        raise ValueError("registered Slurm site policy is invalid")
    if site in required and any(
        not isinstance(slurm.get(field), str) or not slurm[field] for field in fields
    ):
        raise ValueError(f"cell {cell.cell_id} is missing required Slurm provenance")
    if site in forbidden and any(slurm.get(field) is not None for field in fields):
        raise ValueError(f"cell {cell.cell_id} unexpectedly ran under Slurm")
    if set(slurm) != set(fields):
        raise ValueError(f"cell {cell.cell_id} Slurm receipt fields changed")
    if site in required:
        for field in ("array_job_id", "array_task_id", "job_id"):
            value = slurm.get(field)
            if not isinstance(value, str) or not value.isascii() or not value.isdigit():
                label = "Slurm array task" if field == "array_task_id" else f"Slurm {field}"
                raise ValueError(f"cell {cell.cell_id} has invalid numeric {label}")
        task_id = int(str(slurm["array_task_id"]))
        if (
            cell.index < REGISTERED_GOOSE_FIRST_INDEX
            or (cell.index - REGISTERED_GOOSE_FIRST_INDEX) % REGISTERED_GOOSE_INDEX_STEP
        ):
            raise ValueError(f"cell {cell.cell_id} is outside the registered Goose shard")
        site_ordinal = (cell.index - REGISTERED_GOOSE_FIRST_INDEX) // REGISTERED_GOOSE_INDEX_STEP
        expected_task_id = site_ordinal % REGISTERED_GOOSE_ARRAY_TASK_COUNT
        if task_id != expected_task_id:
            raise ValueError(
                f"cell {cell.cell_id} Slurm array task {task_id} does not match "
                f"registered task {expected_task_id}"
            )
    return dict(slurm)


def _same_typed_value(actual: object, expected: object) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, float):
        return math.isfinite(actual) and math.isfinite(expected) and actual == expected
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _same_typed_value(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_typed_value(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    if isinstance(expected, tuple):
        return len(actual) == len(expected) and all(
            _same_typed_value(actual_item, expected_item)
            for actual_item, expected_item in zip(actual, expected, strict=True)
        )
    return actual == expected


def _validate_result_args(
    value: object,
    *,
    root: Path,
    historical_root: Path,
    stage: Mapping[str, object],
    cell: GridCell,
    inputs: Sequence[Path],
    split_path: Path,
    result_path: Path,
    checkpoint_prefix: Path,
    required_device: str,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"cell {cell.cell_id} result args are missing")
    trainer_args = _registered_trainer_args(stage)
    expected: dict[str, object] = {
        "files": [str(historical_root / path.relative_to(root)) for path in inputs],
        "arch": cell.params["arch"],
        "objective_variant": cell.params["objective_variant"],
        "seeds": str(cell.params["seed"]),
        "epochs": stage.get("epochs"),
        **trainer_args,
        "device": required_device,
        "splits": str(historical_root / split_path.relative_to(root)),
        "save": str(historical_root / checkpoint_prefix.relative_to(root)),
        "out": str(historical_root / result_path.relative_to(root)),
    }
    if set(value) != set(expected):
        raise ValueError(f"cell {cell.cell_id} result args fields changed")
    mismatches = sorted(
        name
        for name, expected_value in expected.items()
        if not _same_typed_value(value.get(name), expected_value)
    )
    if mismatches:
        raise ValueError(f"cell {cell.cell_id} result args changed: {mismatches}")
    return trainer_args


def _integer(value: object, *, minimum: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _finite_number(value: object, *, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must be finite")
    return float(value)


def _validate_result(
    result: Mapping[str, object],
    *,
    root: Path,
    historical_root: Path,
    stage: Mapping[str, object],
    cell: GridCell,
    data_provenance: Mapping[str, object],
    inputs: Sequence[Path],
    split_path: Path,
    result_path: Path,
    checkpoint_prefix: Path,
    required_device: str,
) -> dict[str, object]:
    if set(result) != RESULT_FIELDS:
        raise ValueError(f"cell {cell.cell_id} result fields changed")
    if not _same_typed_value(
        result.get("artifact_schema"), "embedbench.quality-v2-training-results"
    ) or not _same_typed_value(result.get("artifact_schema_version"), 1):
        raise ValueError(f"cell {cell.cell_id} has an invalid Quality V2 result schema")
    if not _same_typed_value(result.get("data_provenance"), dict(data_provenance)):
        raise ValueError(f"cell {cell.cell_id} result data_provenance changed")
    split = data_provenance.get("split_manifest")
    expected_split_sha256 = split.get("sha256") if isinstance(split, Mapping) else None
    if not _same_typed_value(result.get("split_sha256"), expected_split_sha256):
        raise ValueError(f"cell {cell.cell_id} result split_sha256 changed")
    if result.get("test_locked") is not True or result.get("test_partition_encoded") is not False:
        raise ValueError(f"cell {cell.cell_id} result does not preserve the test lock")
    trainer_args = _validate_result_args(
        result.get("args"),
        root=root,
        historical_root=historical_root,
        stage=stage,
        cell=cell,
        inputs=inputs,
        split_path=split_path,
        result_path=result_path,
        checkpoint_prefix=checkpoint_prefix,
        required_device=required_device,
    )
    if result.get("device") != required_device:
        raise ValueError(f"cell {cell.cell_id} result device changed")
    training_scope = result.get("training_scope")
    if not _same_typed_value(training_scope, {"limit": None, "full_fixed_train_validation": True}):
        raise ValueError(f"cell {cell.cell_id} did not use the full fixed training scope")
    preprocessing = result.get("preprocessing")
    host_contract = result.get("deployment_host_contract")
    if not isinstance(preprocessing, Mapping) or not isinstance(host_contract, Mapping):
        raise ValueError(f"cell {cell.cell_id} lacks preprocessing or host provenance")
    expected_encoder = "heterogeneous" if cell.params["arch"] == "hetero" else "chain"
    preprocessing_checks = {
        "deploy_view": trainer_args["deploy_view"],
        "deploy_max_free": trainer_args["deploy_max_free"],
        "neighbour_feats": trainer_args["neighbour_feats"],
        "encoder": expected_encoder,
    }
    if any(
        not _same_typed_value(preprocessing.get(name), expected)
        for name, expected in preprocessing_checks.items()
    ):
        raise ValueError(f"cell {cell.cell_id} preprocessing changed")
    rows = result.get("results")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ValueError(f"cell {cell.cell_id} result must contain exactly one seed")
    row = rows[0]
    if set(row) != RESULT_ROW_FIELDS:
        raise ValueError(f"cell {cell.cell_id} seed-result fields changed")
    if any(not _same_typed_value(row.get(axis), value) for axis, value in cell.params.items()):
        raise ValueError(f"cell {cell.cell_id} result axes changed")
    if (
        row.get("test_evaluated") is not False
        or row.get("test_partition_encoded") is not False
        or row.get("test") is not None
    ):
        raise ValueError(f"cell {cell.cell_id} result evaluated or encoded the test split")
    if (
        not _same_typed_value(row.get("training_scope"), training_scope)
        or not _same_typed_value(row.get("preprocessing"), preprocessing)
        or not _same_typed_value(row.get("deployment_host_contract"), host_contract)
        or not _same_typed_value(row.get("device"), required_device)
    ):
        raise ValueError(f"cell {cell.cell_id} seed provenance disagrees with its result")
    for name in ("hidden", "layers", "heads"):
        if not _same_typed_value(row.get(name), trainer_args[name]):
            raise ValueError(f"cell {cell.cell_id} seed result changed registered {name}")
    expected_protocol = {
        "evaluation_support": "full",
        "registered_budget_ratios": list(REGISTERED_BUDGET_RATIOS),
        "primary_validation_metric": REGISTERED_PRIMARY_METRIC,
        "lcb_z": trainer_args["lcb_z"],
        "lcb_is_secondary_until_calibrated": True,
        "ranking_thresholds": {
            "stage2_pair": trainer_args["rank_margin"],
            "stage1_involved_pair": trainer_args["stage1_rank_margin"],
        },
        "selection_contract": REGISTERED_SELECTION_CONTRACT,
        "auxiliary_heads_used_for_selection": False,
    }
    protocol_mismatches = sorted(
        name
        for name, expected in expected_protocol.items()
        if not _same_typed_value(row.get(name), expected)
    )
    if protocol_mismatches:
        raise ValueError(
            f"cell {cell.cell_id} seed result changed registered protocol fields: "
            f"{protocol_mismatches}"
        )
    counts = {
        "n_train": _integer(row.get("n_train"), minimum=1, field=f"cell {cell.cell_id} n_train"),
        "n_val": _integer(row.get("n_val"), minimum=1, field=f"cell {cell.cell_id} n_val"),
        "n_test": _integer(row.get("n_test"), minimum=0, field=f"cell {cell.cell_id} n_test"),
    }
    parameter_count = _integer(row.get("params"), minimum=1, field=f"cell {cell.cell_id} params")
    best_epoch = _integer(row.get("best_epoch"), minimum=1, field=f"cell {cell.cell_id} best_epoch")
    if best_epoch > int(stage["epochs"]):
        raise ValueError(f"cell {cell.cell_id} best_epoch exceeds the registered run")
    seconds = row.get("seconds")
    if type(seconds) is not float or not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError(f"cell {cell.cell_id} seconds must be a finite non-negative float")
    training_run_id = row.get("training_run_id")
    if (
        not isinstance(training_run_id, str)
        or len(training_run_id) != 32
        or not all(character in "0123456789abcdef" for character in training_run_id)
    ):
        raise ValueError(f"cell {cell.cell_id} has an invalid training_run_id")
    validation = row.get("validation")
    budget_sweep = validation.get("budget_sweep") if isinstance(validation, Mapping) else None
    if not isinstance(budget_sweep, Mapping):
        raise ValueError(f"cell {cell.cell_id} has no validation budget sweep")
    primary_metric = _finite_number(
        budget_sweep.get("primary_metric"),
        field=f"cell {cell.cell_id} validation primary metric",
    )
    if primary_metric < 0.0:
        raise ValueError(f"cell {cell.cell_id} validation primary metric must be non-negative")
    objective = cell.params["objective_variant"]
    expected_loss_weights = {
        "quality_probability": 1.0,
        "within_state_rank": 0.5,
        "residual_connectivity_proxy": (
            trainer_args["lambda_connectivity"] if objective in {"p_connectivity", "full"} else 0.0
        ),
        "robustness": (
            trainer_args["lambda_robustness"] if objective in {"p_robustness", "full"} else 0.0
        ),
        "terminal_qubits": 0.0,
    }
    if not _same_typed_value(row.get("loss_weights"), expected_loss_weights):
        raise ValueError(f"cell {cell.cell_id} has invalid registered loss weights")
    training_losses = row.get("best_epoch_training_losses")
    if not isinstance(training_losses, dict) or set(training_losses) != TRAINING_LOSS_FIELDS:
        raise ValueError(f"cell {cell.cell_id} has invalid best-epoch training losses")
    if any(
        type(value) is not float or not math.isfinite(value) for value in training_losses.values()
    ):
        raise ValueError(f"cell {cell.cell_id} has non-finite best-epoch training losses")
    return {
        "trainer_args": trainer_args,
        "counts": counts,
        "parameter_count": parameter_count,
        "training_run_id": training_run_id,
        "best_epoch": best_epoch,
        "primary_metric": primary_metric,
        "training_scope": dict(training_scope),
        "preprocessing": dict(preprocessing),
        "deployment_host_contract": dict(host_contract),
        "loss_weights": dict(row["loss_weights"]),
    }


_CHECKPOINT_VALIDATOR = r"""
import json
import math
import sys
from pathlib import Path

request = json.load(sys.stdin)
source_root = Path(request["source_root"])
sys.path.insert(0, str(source_root / "src"))

import torch
from embedbench.models_quality_v2 import build_quality_v2_model

def same_exact(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, float):
        return math.isfinite(actual) and math.isfinite(expected) and actual == expected
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            same_exact(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            same_exact(left, right) for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected

validated = []
for item in request["checkpoints"]:
    cell_id = item["cell_id"]
    checkpoint = torch.load(item["path"], map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or set(checkpoint) != {"state", "meta"}:
        raise ValueError(f"{cell_id}: checkpoint envelope changed")
    metadata = checkpoint["meta"]
    state = checkpoint["state"]
    if not isinstance(metadata, dict) or not isinstance(state, dict) or not state:
        raise ValueError(f"{cell_id}: checkpoint metadata/state is missing")
    expected_metadata = item["expected_metadata"]
    if set(metadata) != set(expected_metadata) | {"model_config"}:
        raise ValueError(f"{cell_id}: checkpoint metadata fields changed")
    for field, expected in item["expected_metadata"].items():
        if not same_exact(metadata.get(field), expected):
            raise ValueError(f"{cell_id}: checkpoint metadata changed at {field}")
    model_config = metadata.get("model_config")
    if not same_exact(model_config, item["expected_model_config"]):
        raise ValueError(f"{cell_id}: checkpoint model_config changed")
    model = build_quality_v2_model(**model_config)
    model.load_state_dict(state, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    for name, tensor in state.items():
        if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{cell_id}: invalid state entry")
        if tensor.numel() == 0:
            raise ValueError(f"{cell_id}: empty state tensor {name}")
        if (tensor.is_floating_point() or tensor.is_complex()) and not bool(
            torch.isfinite(tensor).all()
        ):
            raise ValueError(f"{cell_id}: non-finite state tensor {name}")
    validated.append({"cell_id": cell_id, "parameter_count": parameter_count})
print(json.dumps(validated, sort_keys=True, allow_nan=False))
"""


def _validate_checkpoints_isolated(
    source_root: Path,
    source_inventory: Sequence[Mapping[str, object]],
    requests: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    if not requests:
        raise ValueError("checkpoint validation request is empty")
    source_root = source_root.resolve()
    with tempfile.TemporaryDirectory(prefix="embedbench-legacy-semantic-capture-") as directory:
        capture_root = Path(directory)
        captured_source_count = 0
        for entry in source_inventory:
            relative_value = entry.get("relative_path")
            file_sha256 = entry.get("file_sha256")
            byte_count = entry.get("bytes")
            if not isinstance(relative_value, str):
                raise ValueError("legacy source inventory has an invalid path")
            relative = _relative_path(relative_value, field="legacy source capture path")
            if relative.parts[:2] != ("src", "embedbench"):
                continue
            if (
                not _is_sha256(file_sha256)
                or isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count < 0
            ):
                raise ValueError("legacy source inventory has invalid capture evidence")
            _capture_verified_file(
                source_root / relative,
                capture_root / relative,
                expected_sha256=str(file_sha256),
                expected_bytes=byte_count,
                root=source_root,
            )
            captured_source_count += 1
        if captured_source_count == 0:
            raise ValueError("legacy semantic capture contains no EmbedBench model source")

        captured_requests: list[dict[str, object]] = []
        reported_parameter_counts: dict[str, int] = {}
        for index, raw_item in enumerate(requests):
            item = dict(raw_item)
            cell_id = item.get("cell_id")
            path = item.get("path")
            staged_root = item.get("staged_root")
            file_sha256 = item.get("file_sha256")
            byte_count = item.get("bytes")
            reported_parameter_count = item.get("reported_parameter_count")
            if (
                not isinstance(cell_id, str)
                or not cell_id
                or not isinstance(path, str)
                or not isinstance(staged_root, str)
                or not _is_sha256(file_sha256)
                or isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count < 1
                or isinstance(reported_parameter_count, bool)
                or not isinstance(reported_parameter_count, int)
                or reported_parameter_count < 1
            ):
                raise ValueError("checkpoint capture request is invalid")
            staged_root_path = Path(staged_root).resolve()
            captured_checkpoint = capture_root / "checkpoints" / f"{index:03d}.pt"
            _capture_verified_file(
                Path(path),
                captured_checkpoint,
                expected_sha256=str(file_sha256),
                expected_bytes=byte_count,
                root=staged_root_path,
            )
            item["path"] = str(captured_checkpoint)
            captured_requests.append(item)
            reported_parameter_counts[cell_id] = reported_parameter_count

        payload = _canonical_bytes(
            {"source_root": str(capture_root), "checkpoints": captured_requests}
        )
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP"}
        }
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        try:
            completed = subprocess.run(
                [sys.executable, "-E", "-c", _CHECKPOINT_VALIDATOR],
                input=payload,
                capture_output=True,
                cwd=capture_root,
                env=environment,
                check=False,
                timeout=180,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise ValueError("isolated checkpoint validation could not complete") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip().splitlines()
        message = detail[-1] if detail else "unknown checkpoint validation failure"
        raise ValueError(f"isolated checkpoint validation failed: {message}")
    try:
        rows = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("isolated checkpoint validator returned invalid JSON") from error
    if not isinstance(rows, list) or len(rows) != len(requests):
        raise ValueError("isolated checkpoint validator returned an incomplete result")
    validated: dict[str, dict[str, object]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("isolated checkpoint validator returned an invalid row")
        cell_id = row.get("cell_id")
        parameter_count = row.get("parameter_count")
        if (
            not isinstance(cell_id, str)
            or cell_id in validated
            or isinstance(parameter_count, bool)
            or not isinstance(parameter_count, int)
            or parameter_count < 1
        ):
            raise ValueError("isolated checkpoint validator returned invalid semantics")
        if parameter_count != reported_parameter_counts.get(cell_id):
            raise ValueError(
                f"cell {cell_id} reported a parameter count that disagrees with its checkpoint"
            )
        validated[cell_id] = row
    return validated


def _validate_receipt(
    *,
    root: Path,
    document: Mapping[str, object],
    stage: Mapping[str, object],
    cell: GridCell,
    grid_sha256: str,
    source_sha256: str,
    data_provenance: Mapping[str, object],
    inputs: Sequence[Path],
    split_path: Path,
    dependency_lock_sha256: str | None,
    approved_goose_wrapper_sha256: str,
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    result_path, receipt_path, checkpoint_path, checkpoint_prefix = _paths_for_cell(
        root, stage, cell
    )
    missing = [
        path.relative_to(root).as_posix()
        for path in (result_path, receipt_path, checkpoint_path)
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"cell {cell.cell_id} has missing artifacts: {missing}")
    substitutes = [
        path.relative_to(root).as_posix()
        for path in (result_path, receipt_path, checkpoint_path)
        if _traverses_symlink(path, root)
    ]
    if substitutes:
        raise ValueError(f"cell {cell.cell_id} uses a symlink substitute artifact: {substitutes}")
    receipt, receipt_entry = _read_json_evidence(
        receipt_path,
        kind="training receipt",
        relative_path=receipt_path.relative_to(root).as_posix(),
        role="legacy_training_receipt",
        root=root,
    )
    result, result_entry = _read_json_evidence(
        result_path,
        kind="training result",
        relative_path=result_path.relative_to(root).as_posix(),
        role="legacy_training_result",
        root=root,
    )
    receipt_fields = set(receipt)
    if not RECEIPT_REQUIRED_FIELDS.issubset(receipt_fields) or not receipt_fields.issubset(
        RECEIPT_REQUIRED_FIELDS | RECEIPT_OPTIONAL_ENVIRONMENT_FIELDS
    ):
        raise ValueError(f"cell {cell.cell_id} receipt fields changed")
    if not isinstance(receipt.get("hostname"), str) or not receipt["hostname"]:
        raise ValueError(f"cell {cell.cell_id} receipt hostname is missing")
    checkpoint_entry = _inventory_entry(
        checkpoint_path,
        checkpoint_path.relative_to(root).as_posix(),
        "legacy_training_checkpoint",
        allow_symlink=False,
        root=root,
    )
    if checkpoint_entry["bytes"] == 0:
        raise ValueError(f"cell {cell.cell_id} has an empty checkpoint")
    expected_cell = {
        "stage": cell.stage,
        "index": cell.index,
        "cell_id": cell.cell_id,
        "params": cell.params,
        "extra_args": list(cell.extra_args),
    }
    checks = {
        "schema": _same_typed_value(receipt.get("schema"), "embedbench.training-receipt"),
        "schema_version": _same_typed_value(receipt.get("schema_version"), 1),
        "grid_id": _same_typed_value(receipt.get("grid_id"), document.get("grid_id")),
        "grid_sha256": _same_typed_value(receipt.get("grid_sha256"), grid_sha256),
        "source_sha256": _same_typed_value(receipt.get("source_sha256"), source_sha256),
        "data_provenance": _same_typed_value(receipt.get("data_provenance"), dict(data_provenance)),
        "cell": _same_typed_value(receipt.get("cell"), expected_cell),
        "result_sha256": _same_typed_value(
            receipt.get("result_sha256"), result_entry["file_sha256"]
        ),
        "checkpoint_sha256": _same_typed_value(
            receipt.get("checkpoint_sha256"), checkpoint_entry["file_sha256"]
        ),
        "test_locked": receipt.get("test_locked") is True,
    }
    failed = sorted(field for field, valid in checks.items() if not valid)
    if failed:
        raise ValueError(f"cell {cell.cell_id} has invalid receipt fields: {failed}")
    site = _expected_site(stage, cell)
    if not _same_typed_value(receipt.get("execution_site"), site):
        raise ValueError(f"cell {cell.cell_id} has invalid execution_site")
    slurm = _validate_slurm(receipt, stage=stage, cell=cell, site=site)
    execution = stage.get("execution")
    required_device = (
        execution.get("required_device_type") if isinstance(execution, Mapping) else None
    )
    if required_device not in {"cpu", "cuda"}:
        raise ValueError("registered grid has no required device type")
    runtime_signature, environment_sha256, python_executable = _runtime_contract(
        result.get("runtime_provenance"),
        required_device=str(required_device),
        cell_id=cell.cell_id,
    )
    recorded_lock = receipt.get("dependency_lock_sha256")
    recorded_environment = receipt.get("environment_sha256")
    recorded_wrapper = receipt.get("launch_wrapper_sha256")
    if recorded_lock is not None and recorded_lock != dependency_lock_sha256:
        raise ValueError(f"cell {cell.cell_id} recorded the wrong dependency lock")
    if recorded_environment is not None and recorded_environment != environment_sha256:
        raise ValueError(f"cell {cell.cell_id} recorded the wrong runtime environment")
    if (
        site == "goose"
        and recorded_wrapper is not None
        and recorded_wrapper != approved_goose_wrapper_sha256
    ):
        raise ValueError(f"cell {cell.cell_id} recorded the wrong launch wrapper")
    dependency_bound = (
        dependency_lock_sha256 is not None and recorded_lock == dependency_lock_sha256
    )
    environment_bound = recorded_environment == environment_sha256
    wrapper_bound = site != "goose" or recorded_wrapper == approved_goose_wrapper_sha256
    posthoc_environment_fields_consistent = dependency_bound and environment_bound and wrapper_bound
    command, historical_root = _validate_command(
        receipt.get("command"),
        root=root,
        stage=stage,
        cell=cell,
        inputs=inputs,
        split_path=split_path,
        result_path=result_path,
        checkpoint_prefix=checkpoint_prefix,
        python_executable=python_executable,
    )
    _require_historical_path(
        receipt.get("result"),
        historical_root=historical_root,
        relative=result_path.relative_to(root),
        field=f"cell {cell.cell_id} receipt result",
    )
    _require_historical_path(
        receipt.get("checkpoint"),
        historical_root=historical_root,
        relative=checkpoint_path.relative_to(root),
        field=f"cell {cell.cell_id} receipt checkpoint",
    )
    started = receipt.get("started_unix")
    finished = receipt.get("finished_unix")
    if (
        isinstance(started, bool)
        or isinstance(finished, bool)
        or not isinstance(started, (int, float))
        or not isinstance(finished, (int, float))
        or not math.isfinite(float(started))
        or not math.isfinite(float(finished))
        or float(finished) < float(started)
    ):
        raise ValueError(f"cell {cell.cell_id} has invalid receipt timestamps")
    result_contract = _validate_result(
        result,
        root=root,
        historical_root=historical_root,
        stage=stage,
        cell=cell,
        data_provenance=data_provenance,
        inputs=inputs,
        split_path=split_path,
        result_path=result_path,
        checkpoint_prefix=checkpoint_prefix,
        required_device=str(required_device),
    )

    artifact_entries = [result_entry, receipt_entry, checkpoint_entry]
    cell_document = {
        "cell_id": cell.cell_id,
        "index": cell.index,
        "params": dict(cell.params),
        "objective_variant": cell.params["objective_variant"],
        "architecture": cell.params["arch"],
        "seed": cell.params["seed"],
        "execution_site": site,
        "hostname": receipt.get("hostname"),
        "slurm": slurm,
        "slurm_receipt_claim_only": site == "goose",
        "recorded_source_sha256": receipt.get("source_sha256"),
        "recorded_grid_sha256": receipt.get("grid_sha256"),
        "recorded_data_provenance": receipt.get("data_provenance"),
        "runtime_compatibility_signature": runtime_signature,
        "runtime_provenance": dict(result["runtime_provenance"]),
        "environment_sha256": environment_sha256,
        "posthoc_environment_fields_consistent": posthoc_environment_fields_consistent,
        "receipt_dependency_lock_sha256": recorded_lock,
        "receipt_launch_wrapper_sha256": recorded_wrapper,
        "started_unix": started,
        "finished_unix": finished,
        "command": command,
        "command_sha256": hashlib.sha256(_canonical_bytes(command)).hexdigest(),
        "historical_staged_root": str(historical_root),
        "validation_primary_metric": result_contract["primary_metric"],
        "partition_record_counts": result_contract["counts"],
        "result": {
            "relative_path": result_entry["relative_path"],
            "file_sha256": result_entry["file_sha256"],
            "bytes": result_entry["bytes"],
        },
        "receipt": {
            "relative_path": receipt_entry["relative_path"],
            "file_sha256": receipt_entry["file_sha256"],
            "bytes": receipt_entry["bytes"],
        },
        "checkpoint": {
            "relative_path": checkpoint_entry["relative_path"],
            "file_sha256": checkpoint_entry["file_sha256"],
            "bytes": checkpoint_entry["bytes"],
        },
    }
    trainer_args = result_contract["trainer_args"]
    checkpoint_request = {
        "cell_id": cell.cell_id,
        "path": str(checkpoint_path),
        "staged_root": str(root),
        "file_sha256": checkpoint_entry["file_sha256"],
        "bytes": checkpoint_entry["bytes"],
        "reported_parameter_count": result_contract["parameter_count"],
        "expected_model_config": {
            "arch": cell.params["arch"],
            "hidden": trainer_args["hidden"],
            "layers": trainer_args["layers"],
            "heads": trainer_args["heads"],
            "neighbour_feats": trainer_args["neighbour_feats"],
            "predict_terminal_qubits": False,
            "robustness_dim": 4,
        },
        "expected_metadata": {
            "artifact_schema": "embedbench.quality-value-v2",
            "artifact_schema_version": 2,
            "forward_inputs": list(REGISTERED_FORWARD_INPUTS),
            "robustness_names": list(REGISTERED_ROBUSTNESS_NAMES),
            "objective_variant": cell.params["objective_variant"],
            "seed": cell.params["seed"],
            "training_run_id": result_contract["training_run_id"],
            "best_epoch": result_contract["best_epoch"],
            "selected_validation_metric": result_contract["primary_metric"],
            "split_sha256": data_provenance["split_manifest"]["sha256"],
            "data_provenance": data_provenance,
            "runtime_provenance": result.get("runtime_provenance"),
            "device": required_device,
            "splits": str(historical_root / split_path.relative_to(root)),
            "evaluation_support": "full",
            "registered_budget_ratios": list(REGISTERED_BUDGET_RATIOS),
            "primary_validation_metric": REGISTERED_PRIMARY_METRIC,
            "lcb_z": trainer_args["lcb_z"],
            "lcb_is_secondary_until_calibrated": True,
            "ranking_thresholds": {
                "stage2_pair": trainer_args["rank_margin"],
                "stage1_involved_pair": trainer_args["stage1_rank_margin"],
            },
            "selection_contract": REGISTERED_SELECTION_CONTRACT,
            "auxiliary_heads_used_for_selection": False,
            "test_evaluated": False,
            "test_partition_encoded": False,
            "training_scope": result_contract["training_scope"],
            "preprocessing": result_contract["preprocessing"],
            "deployment_host_contract": result_contract["deployment_host_contract"],
            "loss_weights": result_contract["loss_weights"],
            "neighbour_feats": trainer_args["neighbour_feats"],
        },
    }
    return cell_document, artifact_entries, checkpoint_request


def _merge_common_inventory(
    destination: dict[str, dict[str, object]], entries: Sequence[Mapping[str, object]]
) -> None:
    for raw_entry in entries:
        entry = dict(raw_entry)
        relative = entry.get("relative_path")
        if not isinstance(relative, str):
            raise ValueError("inventory entry has no relative path")
        previous = destination.get(relative)
        if previous is not None and previous != entry:
            raise ValueError(f"staged roots disagree on file bytes: {relative}")
        destination[relative] = entry


def _artifact_presence(
    root: Path, stage: Mapping[str, object], cells: Sequence[GridCell]
) -> set[str]:
    expected_by_id = {cell.cell_id: cell for cell in cells}
    output_dir = root / _relative_path(stage.get("output_dir"), field="stage output_dir")
    checkpoint_dir = root / _relative_path(
        stage.get("checkpoint_dir"), field="stage checkpoint_dir"
    )
    receipts = set()
    if output_dir.exists():
        for path in output_dir.glob("*.receipt.json"):
            cell_id = path.name.removesuffix(".receipt.json")
            if cell_id not in expected_by_id:
                raise ValueError(f"unregistered receipt in staged grid: {path.name}")
            receipts.add(cell_id)

        expected_results = {f"{cell.cell_id}.json" for cell in cells}
        for path in output_dir.glob("*.json"):
            if path.name.endswith(".receipt.json"):
                continue
            if path.name not in expected_results:
                raise ValueError(f"unregistered result in staged grid: {path.name}")
            cell_id = path.name.removesuffix(".json")
            if cell_id not in receipts:
                raise ValueError(f"orphan result without receipt: {path.name}")

    if checkpoint_dir.exists():
        expected_checkpoints = {
            _paths_for_cell(root, stage, cell)[2].name: cell.cell_id for cell in cells
        }
        for path in checkpoint_dir.glob("*.pt"):
            cell_id = expected_checkpoints.get(path.name)
            if cell_id is None:
                raise ValueError(f"unregistered checkpoint in staged grid: {path.name}")
            if cell_id not in receipts:
                raise ValueError(f"orphan checkpoint without receipt: {path.name}")
    return receipts


def _qualified_roots(
    staged_roots: Sequence[str | Path] | Mapping[str, str | Path],
) -> dict[str, Path]:
    raw: list[tuple[str, str | Path]] = []
    if isinstance(staged_roots, Mapping):
        raw = [(str(site), path) for site, path in staged_roots.items()]
    else:
        for value in staged_roots:
            if isinstance(value, str) and "=" in value:
                site, raw_path = value.split("=", 1)
                raw.append((site, raw_path))
            elif isinstance(value, str):
                raise ValueError("command-line staged roots must be site-qualified as SITE=PATH")
            else:
                path = Path(value)
                raw.append((path.name, path))
    roots: dict[str, Path] = {}
    for site, raw_path in raw:
        if site not in REGISTERED_SITES:
            raise ValueError("staged roots must be site-qualified as apollo=PATH and goose=PATH")
        if site in roots:
            raise ValueError(f"duplicate staged root for site {site}")
        roots[site] = Path(raw_path).resolve()
    if set(roots) != set(REGISTERED_SITES):
        raise ValueError("both site-qualified Apollo and Goose staged roots are required")
    if len(set(roots.values())) != len(roots):
        raise ValueError("Apollo and Goose staged roots must be distinct")
    if any(not root.is_dir() for root in roots.values()):
        raise ValueError("every staged root must be an existing directory")
    return dict(sorted(roots.items()))


def build_snapshot(
    *,
    grid_path: str | Path,
    staged_roots: Sequence[str | Path] | Mapping[str, str | Path],
    expected_legacy_source_sha256: str,
    allow_unlocked_legacy: bool = False,
) -> dict[str, object]:
    """Validate all frozen roots and return a deterministic snapshot document."""

    if expected_legacy_source_sha256 != REGISTERED_LEGACY_SOURCE_SHA256:
        raise ValueError(
            "expected legacy source SHA-256 must equal the registered frozen digest "
            f"{REGISTERED_LEGACY_SOURCE_SHA256}"
        )
    roots_by_site = _qualified_roots(staged_roots)

    registered_grid = Path(grid_path).resolve()
    if not registered_grid.is_file() or registered_grid.is_symlink():
        raise ValueError("registered grid must be a regular file")
    grid_sha256 = _sha256(registered_grid)
    if grid_sha256 != REGISTERED_GRID_SHA256:
        raise ValueError(
            f"registered grid SHA-256 mismatch: expected {REGISTERED_GRID_SHA256}, "
            f"got {grid_sha256}"
        )
    grid_payload = registered_grid.read_bytes()
    document = _read_object(registered_grid, "registered training grid")
    cells = _expand_registered_grid(document)
    stage = _stage(document)

    common_inventory: dict[str, dict[str, object]] = {}
    source_inventory_reference: list[dict[str, object]] | None = None
    data_provenance_reference: dict[str, object] | None = None
    dependency_reference: dict[str, object] | None = None
    launch_by_site: dict[str, dict[str, object]] = {}
    root_contexts: dict[Path, tuple[tuple[Path, ...], Path]] = {}
    owners: dict[str, tuple[str, Path]] = {}

    for staged_site, root in roots_by_site.items():
        source_sha256, source_inventory = legacy_source_inventory(root)
        if source_sha256 != expected_legacy_source_sha256:
            raise ValueError(
                f"legacy source SHA-256 mismatch in staged root: expected "
                f"{expected_legacy_source_sha256}, got {source_sha256}"
            )
        if source_inventory_reference is None:
            source_inventory_reference = source_inventory
        elif source_inventory != source_inventory_reference:
            raise ValueError("staged roots disagree on the legacy source inventory")
        _merge_common_inventory(common_inventory, source_inventory)

        staged_grid = _grid_path(root)
        if (
            not staged_grid.is_file()
            or _traverses_symlink(staged_grid, root)
            or staged_grid.read_bytes() != grid_payload
            or _sha256(staged_grid) != grid_sha256
        ):
            raise ValueError(f"staged root has dirty or substituted registered grid bytes: {root}")
        _merge_common_inventory(
            common_inventory,
            [
                _inventory_entry(
                    staged_grid,
                    REGISTERED_GRID_RELATIVE_PATH.as_posix(),
                    "legacy_training_grid",
                    allow_symlink=False,
                    root=root,
                )
            ],
        )

        data_provenance, data_inventory, inputs, split_path = _data_provenance(root, stage)
        if data_provenance_reference is None:
            data_provenance_reference = data_provenance
        elif data_provenance != data_provenance_reference:
            raise ValueError("staged roots disagree on corpus or split data_provenance")
        _merge_common_inventory(common_inventory, data_inventory)

        dependencies, dependency_inventory = _dependency_provenance(root)
        if dependency_reference is None:
            dependency_reference = dependencies
        elif dependencies != dependency_reference:
            raise ValueError("staged roots disagree on dependency provenance")
        _merge_common_inventory(common_inventory, dependency_inventory)

        launch, launch_inventory = _launch_provenance(root, site=staged_site)
        launch_by_site[staged_site] = launch
        _merge_common_inventory(common_inventory, launch_inventory)

        present = _artifact_presence(root, stage, cells)
        if not present:
            raise ValueError(f"staged root contributes no registered receipts: {root}")
        for cell_id in present:
            if cell_id in owners:
                raise ValueError(f"duplicate registered cell across staged roots: {cell_id}")
            cell = next(candidate for candidate in cells if candidate.cell_id == cell_id)
            expected_site = _expected_site(stage, cell)
            if staged_site != expected_site:
                raise ValueError(
                    f"cell {cell_id} is owned by staged site {staged_site}, "
                    f"not registered site {expected_site}"
                )
            owners[cell_id] = (staged_site, root)
        root_contexts[root] = (inputs, split_path)

    expected_ids = {cell.cell_id for cell in cells}
    missing_ids = sorted(expected_ids - set(owners))
    if missing_ids:
        raise ValueError(f"missing registered cells: {missing_ids}")
    unexpected_ids = sorted(set(owners) - expected_ids)
    if unexpected_ids:
        raise ValueError(f"unregistered cells: {unexpected_ids}")

    if (
        source_inventory_reference is None
        or data_provenance_reference is None
        or dependency_reference is None
        or set(launch_by_site) != set(REGISTERED_SITES)
    ):
        raise RuntimeError("snapshot validation produced no staged-root provenance")
    artifact_inventory: list[dict[str, object]] = []
    cell_documents: list[dict[str, object]] = []
    checkpoint_requests: list[dict[str, object]] = []
    for cell in cells:
        _, root = owners[cell.cell_id]
        inputs, split_path = root_contexts[root]
        cell_document, entries, checkpoint_request = _validate_receipt(
            root=root,
            document=document,
            stage=stage,
            cell=cell,
            grid_sha256=grid_sha256,
            source_sha256=expected_legacy_source_sha256,
            data_provenance=data_provenance_reference,
            inputs=inputs,
            split_path=split_path,
            dependency_lock_sha256=(
                str(dependency_reference["dependency_lock_sha256"])
                if dependency_reference["dependency_lock_sha256"] is not None
                else None
            ),
            approved_goose_wrapper_sha256=REGISTERED_GOOSE_LAUNCH_WRAPPER_SHA256,
        )
        cell_documents.append(cell_document)
        artifact_inventory.extend(entries)
        checkpoint_requests.append(checkpoint_request)

    historical_roots_by_site: dict[str, str] = {}
    for site in REGISTERED_SITES:
        historical_roots = {
            str(cell["historical_staged_root"])
            for cell in cell_documents
            if cell["execution_site"] == site
        }
        if len(historical_roots) != 1:
            raise ValueError(
                f"legacy cells at {site} disagree on their exact historical staged root: "
                f"{sorted(historical_roots)}"
            )
        historical_roots_by_site[site] = historical_roots.pop()

    runtime_signatures = {
        _canonical_bytes(cell["runtime_compatibility_signature"]) for cell in cell_documents
    }
    if len(runtime_signatures) != 1:
        raise ValueError("legacy grid cells used incompatible runtime environments")
    count_signatures = {
        _canonical_bytes(cell["partition_record_counts"]) for cell in cell_documents
    }
    if len(count_signatures) != 1:
        raise ValueError("legacy grid cells disagree on train/validation/test record counts")
    checkpoint_semantics = _validate_checkpoints_isolated(
        next(iter(roots_by_site.values())), source_inventory_reference, checkpoint_requests
    )
    for cell in cell_documents:
        cell["checkpoint_semantics"] = checkpoint_semantics[cell["cell_id"]]

    complete_inventory = [*common_inventory.values(), *artifact_inventory]
    complete_inventory.sort(key=lambda entry: str(entry["relative_path"]))
    if len({entry["relative_path"] for entry in complete_inventory}) != len(complete_inventory):
        raise ValueError("snapshot file inventory contains duplicate logical paths")
    snapshot_manifest_sha256 = inventory_sha256(complete_inventory)
    site_counts: dict[str, int] = {}
    for cell in cell_documents:
        site = str(cell["execution_site"])
        site_counts[site] = site_counts.get(site, 0) + 1

    posthoc_environment_fields_consistent = all(
        cell["posthoc_environment_fields_consistent"] is True for cell in cell_documents
    )
    if not allow_unlocked_legacy:
        raise ValueError(
            "the registered historical grid requires --allow-unlocked-legacy and is usable "
            "only as architecture-selection evidence"
        )
    dependency_document = dict(dependency_reference)
    dependency_document["posthoc_environment_fields_consistent"] = (
        posthoc_environment_fields_consistent
    )
    dependency_document["historical_receipts_authenticated"] = False
    dependency_document["legacy_override_used"] = True
    limitations = [
        (
            "Legacy receipts are unsigned and do not bind a staged-root attestation; "
            "internally consistent regenerated artifacts cannot be authenticated as original "
            "HPC output."
        ),
        (
            "Legacy training parsed raw test records to construct the fixed split, but did not "
            "encode the test partition or compute a test metric."
        ),
    ]
    if posthoc_environment_fields_consistent:
        limitations.append(
            "Any dependency, runtime-environment, or wrapper fields added to a historical "
            "receipt are post-hoc declarations because the frozen launcher did not emit them."
        )
    else:
        limitations.append(
            "Historical receipts do not contain a complete dependency-lock, runtime, and "
            "Goose-wrapper binding."
        )

    return {
        "schema": SNAPSHOT_SCHEMA,
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "legacy_source_digest_algorithm": SOURCE_DIGEST_ALGORITHM,
        "legacy_source_sha256": expected_legacy_source_sha256,
        "legacy_source_inventory": source_inventory_reference,
        "grid_id": document.get("grid_id"),
        "grid_file": REGISTERED_GRID_RELATIVE_PATH.name,
        "grid_relative_path": REGISTERED_GRID_RELATIVE_PATH.as_posix(),
        "grid_sha256": grid_sha256,
        "stage": REGISTERED_STAGE,
        "registered_cell_count": EXPECTED_CELL_COUNT,
        "registered_architectures": list(REGISTERED_ARCHITECTURES),
        "registered_objective_variants": list(REGISTERED_OBJECTIVES),
        "registered_seeds": list(REGISTERED_SEEDS),
        "registered_sites": list(REGISTERED_SITES),
        "logical_site_assignment_verified": True,
        "physical_execution_site_attested": False,
        "historical_staged_roots_by_site": dict(sorted(historical_roots_by_site.items())),
        "execution_site_cell_counts": dict(sorted(site_counts.items())),
        "data_provenance": data_provenance_reference,
        "dependency_provenance": dependency_document,
        "launch_provenance": dict(sorted(launch_by_site.items())),
        "runtime_compatibility_signature": cell_documents[0]["runtime_compatibility_signature"],
        "partition_record_counts": cell_documents[0]["partition_record_counts"],
        "cells": cell_documents,
        "file_inventory": complete_inventory,
        "snapshot_manifest_hash_algorithm": SNAPSHOT_MANIFEST_ALGORITHM,
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
        "evidence_scope": "legacy_architecture_selection_only",
        "historical_receipts_authenticated": False,
        "paper_reproducibility_eligible": False,
        "final_paper_evaluation_eligible": False,
        "legacy_environment_override_used": True,
        "test_access": {
            "corpus_bytes_hashed_by_snapshot": True,
            "corpus_records_parsed_by_snapshot": False,
            "legacy_split_loader_parsed_test_records": True,
            "test_partition_encoded_by_training": False,
            "test_metrics_evaluated_by_training": False,
        },
        "checkpoints_loaded_only_in_isolated_subprocess": True,
        "limitations": limitations,
    }


def _serialized(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(document, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_new_or_verify(path: Path, document: Mapping[str, object]) -> None:
    payload = _serialized(document)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ValueError(f"existing snapshot is stale or tampered: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-legacy-source-sha256", required=True)
    parser.add_argument("--grid", required=True)
    parser.add_argument(
        "--staged-root",
        action="append",
        dest="staged_roots",
        required=True,
        metavar="SITE=PATH",
        help="Site-qualified staged root; pass exactly apollo=PATH and goose=PATH.",
    )
    parser.add_argument(
        "--allow-unlocked-legacy",
        action="store_true",
        help=(
            "Explicitly accept the unsigned historical receipts only as non-paper-eligible "
            "architecture-selection evidence; this can never upgrade their eligibility."
        ),
    )
    parser.add_argument("--out", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    snapshot = build_snapshot(
        grid_path=args.grid,
        staged_roots=args.staged_roots,
        expected_legacy_source_sha256=args.expected_legacy_source_sha256,
        allow_unlocked_legacy=args.allow_unlocked_legacy,
    )
    _write_new_or_verify(Path(args.out), snapshot)
    print(
        json.dumps(
            {
                "cells": snapshot["registered_cell_count"],
                "snapshot_manifest_sha256": snapshot["snapshot_manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
