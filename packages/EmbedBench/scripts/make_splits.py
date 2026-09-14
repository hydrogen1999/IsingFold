#!/usr/bin/env python3
"""Build deterministic release split manifests.

The default ``instance-id`` mode preserves the v1 assignment rule. For v1.1 quality
corpora, use ``--quality-group problem-digest`` so every record carrying the same
canonical full Ising problem stays in one partition, including records from different
sources or host topologies. Structural records continue to be grouped by instance ID.

Example::

    python3 scripts/make_splits.py runs/release_v1_1/*.jsonl \
      --quality-group problem-digest \
      --out runs/release_v1_1/splits_problem_v2.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path

_SCRIPT_DIRECTORY = str(Path(__file__).resolve().parent)
if _SCRIPT_DIRECTORY not in sys.path:
    sys.path.insert(0, _SCRIPT_DIRECTORY)

_training_splits = import_module("training_splits")
file_sha256 = _training_splits.file_sha256
quality_problem_digest = _training_splits.quality_problem_digest


_SPLIT_NAMES = ("train", "val", "test")
_LEGACY_RULE = "sha256(instance_id)[:8]/2^32 < 0.7 train, < 0.8 val, else test"
_PROBLEM_RULE = (
    "structural: sha256(instance_id)[:8]/2^32; quality: "
    "sha256(canonical_problem_sha256)[:8]/2^32; < 0.7 train, "
    "< 0.8 val, else test"
)


def _assign_split(group_key: str) -> str:
    value = int(hashlib.sha256(group_key.encode()).hexdigest()[:8], 16) / 2**32
    if value < 0.7:
        return "train"
    if value < 0.8:
        return "val"
    return "test"


def _is_quality_record(filename: str, record: Mapping[str, object]) -> bool:
    return filename.startswith("quality_") or "problem" in record


def validate_group_disjoint(manifest: Mapping[str, object]) -> None:
    """Reject manifests that assign one canonical group to multiple partitions."""
    splits = manifest.get("splits")
    groups = manifest.get("groups")
    if not isinstance(splits, Mapping) or not isinstance(groups, Mapping):
        raise ValueError("manifest must contain object fields 'splits' and 'groups'")
    if set(splits) != set(groups):
        raise ValueError("split and group tables must cover the same files")

    assignments: dict[str, str] = {}
    for filename in sorted(splits):
        split_table = splits[filename]
        group_table = groups[filename]
        if not isinstance(split_table, Mapping) or not isinstance(group_table, Mapping):
            raise ValueError(f"split and group tables for {filename!r} must be objects")
        if set(split_table) != set(group_table):
            raise ValueError(
                f"split and group tables for {filename!r} must cover the same instances"
            )
        for instance_id in sorted(split_table):
            split = split_table[instance_id]
            group_id = group_table[instance_id]
            if split not in _SPLIT_NAMES:
                raise ValueError(f"invalid split {split!r} for {instance_id!r} in {filename!r}")
            if not isinstance(group_id, str) or not group_id:
                raise ValueError(f"invalid canonical group for {instance_id!r} in {filename!r}")
            previous = assignments.setdefault(group_id, split)
            if previous != split:
                raise ValueError(
                    f"split leakage: group {group_id!r} is assigned to both "
                    f"{previous!r} and {split!r}"
                )


def build_manifest(
    files: Sequence[str | Path],
    *,
    quality_group: str = "instance-id",
) -> dict[str, object]:
    """Build a manifest without writing it, allowing validation before publication."""
    if quality_group not in {"instance-id", "problem-digest"}:
        raise ValueError(f"unsupported quality grouping {quality_group!r}")

    paths = sorted((Path(file) for file in files), key=lambda path: path.name)
    if not paths:
        raise ValueError("at least one corpus file is required")
    names = [path.name for path in paths]
    if len(names) != len(set(names)):
        raise ValueError("input corpus basenames must be unique")

    split_tables: dict[str, dict[str, str]] = {}
    group_tables: dict[str, dict[str, str]] = {}
    record_counts: dict[str, dict[str, int]] = {}
    group_assignments: dict[str, str] = {}
    group_members: dict[str, set[str]] = {split: set() for split in _SPLIT_NAMES}
    inputs: list[dict[str, str]] = []

    for path in paths:
        filename = path.name
        file_splits: dict[str, str] = {}
        file_groups: dict[str, str] = {}
        counts = {split: 0 for split in _SPLIT_NAMES}
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
                if not isinstance(record, Mapping):
                    raise ValueError(f"{path}:{line_number}: record must be an object")
                if "instance_id" not in record:
                    raise ValueError(f"{path}:{line_number}: record has no instance_id")
                instance_id = str(record["instance_id"])
                if not instance_id:
                    raise ValueError(f"{path}:{line_number}: instance_id must not be empty")

                use_problem = quality_group == "problem-digest" and _is_quality_record(
                    filename, record
                )
                if use_problem:
                    try:
                        digest = quality_problem_digest(record)
                    except ValueError as error:
                        raise ValueError(f"{path}:{line_number}: {error}") from error
                    group_id = f"problem-digest:{digest}"
                    partition_key = digest
                else:
                    group_id = f"instance-id:{instance_id}"
                    partition_key = instance_id
                split = _assign_split(partition_key)

                previous_group = file_groups.setdefault(instance_id, group_id)
                if previous_group != group_id:
                    raise ValueError(
                        f"{path}:{line_number}: instance {instance_id!r} contains "
                        "multiple canonical problems"
                    )
                previous_split = file_splits.setdefault(instance_id, split)
                if previous_split != split:
                    raise ValueError(
                        f"{path}:{line_number}: instance {instance_id!r} maps to "
                        "multiple partitions"
                    )
                previous_assignment = group_assignments.setdefault(group_id, split)
                if previous_assignment != split:
                    raise ValueError(
                        f"split leakage: group {group_id!r} is assigned to both "
                        f"{previous_assignment!r} and {split!r}"
                    )
                counts[split] += 1
                group_members[split].add(group_id)

        split_tables[filename] = file_splits
        group_tables[filename] = file_groups
        record_counts[filename] = counts
        inputs.append({"file": filename, "sha256": file_sha256(path)})

    manifest: dict[str, object] = {
        "schema": "embedbench.split-manifest",
        "schema_version": 2,
        "provenance": {
            "generator": "EmbedBench/scripts/make_splits.py",
            "quality_group": quality_group,
            "inputs": inputs,
            "assignment": {
                "hash": "sha256",
                "prefix_hex_chars": 8,
                "denominator": 2**32,
                "thresholds": {"train": 0.7, "val": 0.8, "test": 1.0},
            },
        },
        "rule": _PROBLEM_RULE if quality_group == "problem-digest" else _LEGACY_RULE,
        "splits": split_tables,
        "groups": group_tables,
        "record_counts": record_counts,
        "group_counts": {split: len(group_members[split]) for split in _SPLIT_NAMES},
    }
    validate_group_disjoint(manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--quality-group",
        choices=("instance-id", "problem-digest"),
        default="instance-id",
        help=("quality grouping unit; problem-digest is required for leakage-safe v1.1 training"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    output = Path(args.out)
    if output.resolve() in {Path(file).resolve() for file in args.files}:
        parser.error("--out must not overwrite an input corpus")
    try:
        manifest = build_manifest(args.files, quality_group=args.quality_group)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest["record_counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
