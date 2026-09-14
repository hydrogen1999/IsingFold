"""Load release records according to a checked ``splits.json`` manifest."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

_SPLIT_NAMES = ("train", "val", "test")
_SPLIT_SCHEMA = "embedbench.split-manifest"
_SPLIT_SCHEMA_VERSION = 2


class SplitRecords(NamedTuple):
    train: list[dict]
    val: list[dict]
    test: list[dict]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_v2_input_hashes(
    split_document: Mapping[str, object], splits_path: str | Path
) -> dict[str, str] | None:
    """Return declared corpus hashes, or ``None`` for a legacy manifest."""
    schema = split_document.get("schema")
    schema_version = split_document.get("schema_version")
    if schema is None and schema_version is None:
        return None
    if schema != _SPLIT_SCHEMA or schema_version != _SPLIT_SCHEMA_VERSION:
        raise ValueError(
            f"{splits_path}: unsupported split manifest schema "
            f"{schema!r} version {schema_version!r}"
        )

    provenance = split_document.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{splits_path}: schema-v2 manifest has no provenance object")
    inputs = provenance.get("inputs")
    if not isinstance(inputs, list):
        raise ValueError(f"{splits_path}: schema-v2 provenance inputs must be a list")

    input_hashes: dict[str, str] = {}
    for index, entry in enumerate(inputs):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{splits_path}: provenance input {index} must be an object")
        filename = entry.get("file")
        expected = entry.get("sha256")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"{splits_path}: provenance input {index} has no file name")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError(
                f"{splits_path}: provenance input {filename!r} has an invalid SHA-256"
            )
        if filename in input_hashes:
            raise ValueError(f"{splits_path}: duplicate provenance input {filename!r}")
        input_hashes[filename] = expected
    return input_hashes


def _require_exact_schema_v2_inputs(
    names: Sequence[str],
    expected_input_hashes: Mapping[str, str] | None,
    splits_path: str | Path,
) -> None:
    """Reject partial or extra corpora for a provenance-bound split manifest."""
    if expected_input_hashes is None:
        return
    provided = set(names)
    expected = set(expected_input_hashes)
    if provided != expected:
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        raise ValueError(
            f"{splits_path}: schema-v2 manifest requires the exact corpus input set; "
            f"missing={missing}, unexpected={unexpected}"
        )


def data_provenance(
    files: Sequence[str | Path],
    splits_path: str | Path | None,
) -> dict[str, object]:
    """Return portable hashes identifying the corpus and optional split manifest.

    Schema-v2 manifests already commit to every release input.  This helper copies
    those verified commitments into each result and checkpoint artifact.  Legacy or
    seed-based runs still record hashes computed from the files actually consumed.
    Paths are represented by unique basenames so artifacts can move between hosts.
    """
    paths = [Path(file) for file in files]
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("input corpus basenames must be unique")

    split_manifest = None
    expected_input_hashes = None
    if splits_path is not None:
        split_document = json.loads(Path(splits_path).read_text(encoding="utf-8"))
        if not isinstance(split_document, Mapping):
            raise ValueError(f"{splits_path}: split manifest must be an object")
        expected_input_hashes = _schema_v2_input_hashes(split_document, splits_path)
        split_manifest = {
            "file": Path(splits_path).name,
            "sha256": file_sha256(splits_path),
            "schema": split_document.get("schema"),
            "schema_version": split_document.get("schema_version"),
        }

    _require_exact_schema_v2_inputs(names, expected_input_hashes, splits_path or "<none>")

    corpus_inputs = []
    for path in paths:
        actual = file_sha256(path)
        if expected_input_hashes is not None:
            expected = expected_input_hashes.get(path.name)
            if expected is None:
                raise ValueError(f"{splits_path}: no provenance hash for {path.name}")
            if actual != expected:
                raise ValueError(
                    f"SHA-256 mismatch for {path.name}: manifest records {expected}, "
                    f"but the training corpus is {actual}"
                )
        corpus_inputs.append({"file": path.name, "sha256": actual})

    return {
        "split_manifest": split_manifest,
        "corpus_inputs": corpus_inputs,
    }


def quality_problem_id(instance_id: str) -> str:
    """Return the ID shared by every state derived from one planted problem.

    Witness/minorminer states share the suffix-free ID. Application and random
    problems are also reused across host topologies by the release generator, so
    their topology prefix is excluded as well.
    """
    base, separator, source = instance_id.rpartition("-")
    if separator and source in {"w", "m"}:
        instance_id = base
    topology, separator, remainder = instance_id.partition("-")
    is_host_tag = any(
        topology.startswith(family) and topology.removeprefix(family).isdigit()
        for family in ("chimera", "pegasus", "zephyr")
    )
    if separator and is_host_tag and remainder.startswith(("app-", "random-")):
        return remainder
    return instance_id


def _canonical_float(value: object, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"problem {field} must be numeric, not bool")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"problem {field} must be finite")
    return 0.0 if number == 0.0 else number


def _canonical_node(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("problem node IDs must be integers, not bool")
    node = int(value)
    if isinstance(value, float) and not value.is_integer():
        raise ValueError(f"problem node ID {value!r} is not an integer")
    return node


def quality_problem_digest(record: Mapping[str, object]) -> str:
    """Hash the semantic Ising payload independently of JSON and edge ordering."""
    problem = record.get("problem")
    if not isinstance(problem, Mapping):
        instance_id = record.get("instance_id", "<unknown>")
        raise ValueError(f"quality record {instance_id!r} has no full problem payload")
    raw_h = problem.get("h")
    raw_j = problem.get("J")
    if not isinstance(raw_h, Mapping) or not isinstance(raw_j, list) or "e0" not in problem:
        raise ValueError("problem payload must contain h, J, and e0")

    h = sorted(
        (_canonical_node(node), _canonical_float(bias, f"h[{node!r}]"))
        for node, bias in raw_h.items()
    )
    couplers = []
    for index, edge in enumerate(raw_j):
        if not isinstance(edge, (list, tuple)) or len(edge) != 3:
            raise ValueError(f"problem J[{index}] must be [u, v, bias]")
        u, v = sorted((_canonical_node(edge[0]), _canonical_node(edge[1])))
        couplers.append((u, v, _canonical_float(edge[2], f"J[{index}]")))
    canonical = {
        "h": h,
        "J": sorted(couplers),
        "e0": _canonical_float(problem["e0"], "e0"),
    }
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_split_records(
    files: Sequence[str | Path],
    splits_path: str | Path,
    *,
    group_key: Callable[[str], str] = str,
    record_group_key: Callable[[Mapping[str, object]], str] | None = None,
    limit: int | None = None,
) -> SplitRecords:
    """Load records into fixed partitions and reject incomplete or leaking manifests.

    ``group_key`` identifies the unit that must remain in one partition. Structural
    records use their exact instance ID. Quality records pass ``quality_problem_id``
    because witness and minorminer states are derived from the same planted problem.
    """
    split_document = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    if not isinstance(split_document, Mapping):
        raise ValueError(f"{splits_path}: split manifest must be an object")
    split_tables = split_document.get("splits")
    if not isinstance(split_tables, dict):
        raise ValueError(f"{splits_path}: expected an object field named 'splits'")
    expected_input_hashes = _schema_v2_input_hashes(split_document, splits_path)

    paths = [Path(file) for file in files]
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("input corpus basenames must be unique")
    if expected_input_hashes is not None:
        if set(expected_input_hashes) != set(split_tables):
            raise ValueError(
                f"{splits_path}: schema-v2 provenance inputs and split tables "
                "must cover the same files"
            )
        _require_exact_schema_v2_inputs(names, expected_input_hashes, splits_path)
        for path in paths:
            expected = expected_input_hashes.get(path.name)
            if expected is None:
                raise ValueError(f"{splits_path}: no provenance hash for {path.name}")
            actual = file_sha256(path)
            if actual != expected:
                raise ValueError(
                    f"SHA-256 mismatch for {path.name}: manifest records {expected}, "
                    f"but the training corpus is {actual}"
                )

    group_assignments: dict[str, str] = {}
    selected_tables: dict[str, dict[str, str]] = {}
    for name in names:
        table = split_tables.get(name)
        if not isinstance(table, dict):
            raise ValueError(f"{splits_path}: no split table for {name}")
        selected_tables[name] = table
        for raw_instance_id, split in table.items():
            instance_id = str(raw_instance_id)
            if split not in _SPLIT_NAMES:
                raise ValueError(
                    f"{splits_path}: invalid split {split!r} for {instance_id!r} in {name}"
                )
            group_id = group_key(instance_id)
            previous = group_assignments.setdefault(group_id, split)
            if previous != split:
                raise ValueError(
                    f"split leakage: group {group_id!r} is assigned to both "
                    f"{previous!r} and {split!r}"
                )

    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    partitions: dict[str, list[dict]] = {name: [] for name in _SPLIT_NAMES}
    record_group_assignments: dict[str, str] = {}
    loaded = 0
    for path in paths:
        table = selected_tables[path.name]
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                record = json.loads(line)
                if "instance_id" not in record:
                    raise ValueError(f"{path}:{line_number}: record has no instance_id")
                instance_id = str(record["instance_id"])
                split = table.get(instance_id)
                if split is None:
                    raise ValueError(
                        f"instance {instance_id!r} in {path.name} is unmapped by {splits_path}"
                    )
                if record_group_key is not None:
                    record_group = record_group_key(record)
                    previous = record_group_assignments.setdefault(record_group, split)
                    if previous != split:
                        raise ValueError(
                            "split leakage: problem payload "
                            f"{record_group!r} is assigned to both {previous!r} and {split!r}"
                        )
                if limit is None or loaded < limit:
                    partitions[split].append(record)
                    loaded += 1

    return SplitRecords(*(partitions[name] for name in _SPLIT_NAMES))
