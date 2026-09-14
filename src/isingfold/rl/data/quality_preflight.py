"""Immutable artifacts for distributed exact replay of quality row-v7 shards.

The final quality-preflight v3 receipt is deliberately outside this module: that receipt must be
produced by one fresh direct replay.  These artifacts authorize the earlier quality-label merge to
reuse a complete, externally pinned replay census without executing the same continuations a
second time.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from isingfold.rl.data.exact_conformance import (
    ExactConformanceError,
    publish_new_file,
    read_regular_file,
)
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest


QUALITY_PREFLIGHT_PLAN_SCHEMA = "isingfold.quality-preflight-plan"
QUALITY_PREFLIGHT_PLAN_VERSION = 1
QUALITY_PREFLIGHT_ROW_SCHEMA = "isingfold.quality-preflight-row-evidence"
QUALITY_PREFLIGHT_ROW_VERSION = 1
QUALITY_PREFLIGHT_SHARD_SCHEMA = "isingfold.quality-preflight-shard"
QUALITY_PREFLIGHT_SHARD_VERSION = 1
QUALITY_PREFLIGHT_REPLAY_BUNDLE_SCHEMA = "isingfold.quality-preflight-replay-bundle"
QUALITY_PREFLIGHT_REPLAY_BUNDLE_VERSION = 1

_HEX = frozenset("0123456789abcdef")
_ROW_FIELDS = {
    "lineage",
    "source_row_record_digest",
    "task_id",
    "trajectory_count",
}
_SOURCE_FIELDS = {
    "manifest_record_digest",
    "manifest_sha256",
    "records_sha256",
    "rows",
    "selected_lineages",
    "selected_task_ids",
    "shard_index",
}
_EVIDENCE_INPUT_FIELDS = {
    "expected_continuation_root_digest",
    "lineage",
    "match_count",
    "pass",
    "replayed_continuation_root_digest",
    "source_row_record_digest",
    "task_id",
    "trajectory_count",
}
_EVIDENCE_FIELDS = _EVIDENCE_INPUT_FIELDS | {
    "record_digest",
    "schema",
    "schema_version",
}
_PLAN_FIELDS = {
    "assignments",
    "identity",
    "record_digest",
    "row_count",
    "row_set_digest",
    "schema",
    "schema_version",
    "shard_count",
    "source_shards",
    "trajectory_count",
}
_ASSIGNMENT_FIELDS = {
    "lineages",
    "row_count",
    "rows",
    "shard_index",
    "source_quality_shard",
    "task_ids",
    "trajectory_count",
}
_SOURCE_IDENTITY_FIELDS = {
    "manifest_record_digest",
    "manifest_sha256",
    "record_count",
    "records_sha256",
    "row_set_digest",
    "selected_lineages",
    "selected_task_ids",
    "shard_index",
    "trajectory_count",
}
_SHARD_FIELDS = {
    "assigned",
    "evidence_record_set_digest",
    "evidence_sha256",
    "identity",
    "matching_count",
    "mismatch_source_row_record_digests",
    "pass",
    "plan",
    "record_digest",
    "replayed_count",
    "runtime",
    "schema",
    "schema_version",
    "source_quality_shard",
}
_BUNDLE_FIELDS = {
    "identity",
    "matching_count",
    "mismatch_source_row_record_digests",
    "pass",
    "plan",
    "record_digest",
    "replay_shard_merkle_root",
    "replay_shards",
    "replayed_count",
    "row_count",
    "row_evidence",
    "row_set_digest",
    "runtime_set_digest",
    "schema",
    "schema_version",
    "source_shards",
    "trajectory_count",
}
_REPLAY_SHARD_IDENTITY_FIELDS = {
    "evidence_record_set_digest",
    "evidence_sha256",
    "manifest_record_digest",
    "manifest_sha256",
    "matching_count",
    "replayed_count",
    "runtime_digest",
    "shard_index",
}


class QualityPreflightError(ValueError):
    """An exact replay artifact is malformed, incomplete, or not externally pinned."""


def _digest(value: object, label: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in _HEX for char in value):
        raise QualityPreflightError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _nonnegative(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise QualityPreflightError(f"{label} must be a nonnegative integer")
    return value


def _positive(value: object, label: str) -> int:
    result = _nonnegative(value, label)
    if result == 0:
        raise QualityPreflightError(f"{label} must be a positive integer")
    return result


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise QualityPreflightError(f"{label} must be nonempty text")
    return value


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in sorted(value.items())})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_thaw(item) for item in value]
    return value


def _with_digest(payload: Mapping[str, object]) -> dict[str, object]:
    return {**payload, "record_digest": content_digest(payload)}


def _verify_record(record: Mapping[str, object], label: str) -> None:
    observed = _digest(record.get("record_digest"), f"{label} record digest")
    payload = {key: value for key, value in record.items() if key != "record_digest"}
    if not hmac.compare_digest(observed, content_digest(payload)):
        raise QualityPreflightError(f"{label} record digest mismatch")


def _strict_json(raw: bytes, label: str) -> dict[str, object]:
    def reject_constant(token: str) -> None:
        raise QualityPreflightError(f"{label} contains non-finite number {token}")

    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise QualityPreflightError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as error:
        raise QualityPreflightError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise QualityPreflightError(f"{label} must contain one JSON object")
    canonical_json_bytes(value)
    return value


def _read_pinned_json(
    path: str | Path, expected_sha256: str, label: str
) -> tuple[dict[str, object], str]:
    expected = _digest(expected_sha256, f"expected {label} SHA-256")
    try:
        _resolved, raw = read_regular_file(path, label)
    except (ExactConformanceError, OSError) as error:
        raise QualityPreflightError(str(error)) from error
    observed = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(observed, expected):
        raise QualityPreflightError(f"{label} differs from its externally pinned SHA-256")
    return _strict_json(raw, label), observed


def _publish_file(path: Path, record: Mapping[str, object]) -> str:
    raw = canonical_json_bytes(record) + b"\n"
    try:
        publish_new_file(path, raw)
    except ExactConformanceError as error:
        raise QualityPreflightError(str(error)) from error
    return hashlib.sha256(raw).hexdigest()


def _validated_row(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or set(value) != _ROW_FIELDS:
        raise QualityPreflightError(f"{label} fields differ from the exact schema")
    row = dict(value)
    _text(row["lineage"], f"{label} lineage")
    _text(row["task_id"], f"{label} task id")
    _digest(row["source_row_record_digest"], f"{label} source row")
    _positive(row["trajectory_count"], f"{label} trajectory count")
    return row


def _validated_sources(
    source_shards: Sequence[Mapping[str, object]], *, shard_count: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    _positive(shard_count, "quality-preflight shard count")
    if len(source_shards) != shard_count:
        raise QualityPreflightError("source quality shard count differs from replay shard count")
    sources: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    seen_rows: set[str] = set()
    seen_lineages: set[str] = set()
    for expected_index, value in enumerate(source_shards):
        if not isinstance(value, Mapping) or set(value) != _SOURCE_FIELDS:
            raise QualityPreflightError("source quality shard fields differ from the exact schema")
        source = dict(value)
        if source["shard_index"] != expected_index:
            raise QualityPreflightError("source quality shards are missing or out of order")
        for key in ("manifest_record_digest", "manifest_sha256", "records_sha256"):
            _digest(source[key], f"source quality shard {expected_index} {key}")
        rows_raw = source["rows"]
        lineages = source["selected_lineages"]
        tasks = source["selected_task_ids"]
        if (
            not isinstance(rows_raw, Sequence)
            or isinstance(rows_raw, (str, bytes, bytearray))
            or not rows_raw
            or not isinstance(lineages, Sequence)
            or isinstance(lineages, (str, bytes, bytearray))
            or not lineages
            or not isinstance(tasks, Sequence)
            or isinstance(tasks, (str, bytes, bytearray))
            or not tasks
        ):
            raise QualityPreflightError("source quality shard has an empty census")
        rows = [
            _validated_row(row, f"source quality shard {expected_index} row {position}")
            for position, row in enumerate(rows_raw)
        ]
        source_lineages = [_text(item, "source quality lineage") for item in lineages]
        source_tasks = [_text(item, "source quality task") for item in tasks]
        if source_lineages != sorted(set(source_lineages)):
            raise QualityPreflightError("source quality lineages are repeated or unsorted")
        if source_tasks != sorted(set(source_tasks)):
            raise QualityPreflightError("source quality tasks are repeated or unsorted")
        if {str(row["lineage"]) for row in rows} - set(source_lineages):
            raise QualityPreflightError("source quality row lies outside its lineage census")
        if {str(row["task_id"]) for row in rows} - set(source_tasks):
            raise QualityPreflightError("source quality row lies outside its task census")
        if seen_lineages & set(source_lineages):
            raise QualityPreflightError("source quality shards duplicate a lineage")
        seen_lineages.update(source_lineages)
        for row in rows:
            row_digest = str(row["source_row_record_digest"])
            if row_digest in seen_rows:
                raise QualityPreflightError("source row coverage contains a duplicate")
            seen_rows.add(row_digest)
        trajectory_count = sum(int(row["trajectory_count"]) for row in rows)
        row_set_digest = content_digest(
            sorted(str(row["source_row_record_digest"]) for row in rows)
        )
        source_identity = {
            "manifest_record_digest": source["manifest_record_digest"],
            "manifest_sha256": source["manifest_sha256"],
            "record_count": len(rows),
            "records_sha256": source["records_sha256"],
            "row_set_digest": row_set_digest,
            "selected_lineages": source_lineages,
            "selected_task_ids": source_tasks,
            "shard_index": expected_index,
            "trajectory_count": trajectory_count,
        }
        sources.append(source_identity)
        assignments.append(
            {
                "lineages": source_lineages,
                "row_count": len(rows),
                "rows": rows,
                "shard_index": expected_index,
                "source_quality_shard": source_identity,
                "task_ids": source_tasks,
                "trajectory_count": trajectory_count,
            }
        )
    return sources, assignments


@dataclasses.dataclass(frozen=True, slots=True)
class QualityPreflightPlan:
    """Read-only target-free replay plan loaded through a raw-file pin."""

    _record: Mapping[str, object]
    raw_sha256: str

    def as_dict(self) -> dict[str, object]:
        return _thaw(self._record)  # type: ignore[return-value]


def _validate_plan(record: Mapping[str, object]) -> None:
    if set(record) != _PLAN_FIELDS:
        raise QualityPreflightError("quality-preflight plan fields differ from the exact schema")
    _verify_record(record, "quality-preflight plan")
    if (
        record.get("schema") != QUALITY_PREFLIGHT_PLAN_SCHEMA
        or record.get("schema_version") != QUALITY_PREFLIGHT_PLAN_VERSION
    ):
        raise QualityPreflightError("unsupported quality-preflight plan schema")
    shard_count = _positive(record.get("shard_count"), "quality-preflight shard count")
    sources_raw = record.get("source_shards")
    assignments_raw = record.get("assignments")
    if not isinstance(sources_raw, list) or not isinstance(assignments_raw, list):
        raise QualityPreflightError("quality-preflight plan censuses must be arrays")
    synthetic_sources: list[dict[str, object]] = []
    for assignment in assignments_raw:
        if not isinstance(assignment, Mapping) or set(assignment) != _ASSIGNMENT_FIELDS:
            raise QualityPreflightError("quality-preflight assignment fields differ")
        source = assignment["source_quality_shard"]
        if not isinstance(source, Mapping) or set(source) != _SOURCE_IDENTITY_FIELDS:
            raise QualityPreflightError("quality-preflight source identity fields differ")
        synthetic_sources.append(
            {
                "manifest_record_digest": source["manifest_record_digest"],
                "manifest_sha256": source["manifest_sha256"],
                "records_sha256": source["records_sha256"],
                "rows": assignment["rows"],
                "selected_lineages": assignment["lineages"],
                "selected_task_ids": assignment["task_ids"],
                "shard_index": assignment["shard_index"],
            }
        )
    expected_sources, expected_assignments = _validated_sources(
        synthetic_sources, shard_count=shard_count
    )
    if sources_raw != expected_sources or assignments_raw != expected_assignments:
        raise QualityPreflightError("quality-preflight plan censuses cannot be reconstructed")
    row_digests = [
        str(row["source_row_record_digest"])
        for assignment in expected_assignments
        for row in assignment["rows"]
    ]
    trajectory_count = sum(int(item["trajectory_count"]) for item in expected_assignments)
    identity = record.get("identity")
    if not isinstance(identity, Mapping) or not identity:
        raise QualityPreflightError("quality-preflight plan identity is missing")
    canonical_json_bytes(identity)
    if (
        record.get("row_count") != len(row_digests)
        or record.get("trajectory_count") != trajectory_count
        or record.get("row_set_digest") != content_digest(sorted(row_digests))
    ):
        raise QualityPreflightError("quality-preflight plan totals cannot be reconstructed")


def build_quality_preflight_plan(
    *,
    source_shards: Sequence[Mapping[str, object]],
    shard_count: int,
    identity: Mapping[str, object],
    output_path: str | Path,
) -> str:
    """Publish one target-free whole-source-shard replay plan."""

    if not isinstance(identity, Mapping) or not identity:
        raise QualityPreflightError("quality-preflight identity must be nonempty")
    canonical_json_bytes(identity)
    sources, assignments = _validated_sources(source_shards, shard_count=shard_count)
    row_digests = [
        str(row["source_row_record_digest"])
        for assignment in assignments
        for row in assignment["rows"]
    ]
    payload: dict[str, object] = {
        "schema": QUALITY_PREFLIGHT_PLAN_SCHEMA,
        "schema_version": QUALITY_PREFLIGHT_PLAN_VERSION,
        "identity": dict(identity),
        "source_shards": sources,
        "assignments": assignments,
        "shard_count": shard_count,
        "row_count": len(row_digests),
        "trajectory_count": sum(int(item["trajectory_count"]) for item in assignments),
        "row_set_digest": content_digest(sorted(row_digests)),
    }
    record = _with_digest(payload)
    _validate_plan(record)
    return _publish_file(Path(output_path), record)


def load_quality_preflight_plan(path: str | Path, *, expected_sha256: str) -> QualityPreflightPlan:
    record, observed = _read_pinned_json(path, expected_sha256, "quality-preflight plan")
    _validate_plan(record)
    return QualityPreflightPlan(_freeze(record), observed)  # type: ignore[arg-type]


def _plan_identity(
    plan: QualityPreflightPlan, expected_plan_sha256: str
) -> tuple[dict[str, object], str]:
    if not isinstance(plan, QualityPreflightPlan):
        raise TypeError("plan must be a QualityPreflightPlan")
    expected = _digest(expected_plan_sha256, "expected quality-preflight plan SHA-256")
    if not hmac.compare_digest(plan.raw_sha256, expected):
        raise QualityPreflightError("quality-preflight plan differs from its external pin")
    record = plan.as_dict()
    _validate_plan(record)
    return record, expected


def _validated_evidence(
    value: Mapping[str, object], expected_row: Mapping[str, object]
) -> dict[str, object]:
    if set(value) == _EVIDENCE_FIELDS:
        _verify_record(value, "quality-preflight row evidence")
        payload = {
            key: item
            for key, item in value.items()
            if key not in {"schema", "schema_version", "record_digest"}
        }
    elif set(value) == _EVIDENCE_INPUT_FIELDS:
        payload = dict(value)
    else:
        raise QualityPreflightError("quality-preflight row evidence fields differ")
    for key in ("expected_continuation_root_digest", "replayed_continuation_root_digest"):
        _digest(payload[key], f"quality-preflight {key}")
    for key in ("lineage", "task_id"):
        _text(payload[key], f"quality-preflight evidence {key}")
    trajectory_count = _positive(
        payload["trajectory_count"], "quality-preflight evidence trajectory count"
    )
    match_count = _nonnegative(payload["match_count"], "quality-preflight evidence match count")
    if (
        payload["source_row_record_digest"] != expected_row["source_row_record_digest"]
        or payload["lineage"] != expected_row["lineage"]
        or payload["task_id"] != expected_row["task_id"]
        or trajectory_count != expected_row["trajectory_count"]
    ):
        raise QualityPreflightError("quality-preflight evidence belongs to another source row")
    passed = payload["pass"]
    expected_pass = (
        match_count == trajectory_count
        and payload["expected_continuation_root_digest"]
        == payload["replayed_continuation_root_digest"]
    )
    if type(passed) is not bool or passed is not expected_pass:
        raise QualityPreflightError("quality-preflight row pass cannot be reconstructed")
    body = {
        "schema": QUALITY_PREFLIGHT_ROW_SCHEMA,
        "schema_version": QUALITY_PREFLIGHT_ROW_VERSION,
        **payload,
    }
    record = _with_digest(body)
    if set(value) == _EVIDENCE_FIELDS and dict(value) != record:
        raise QualityPreflightError("quality-preflight evidence canonical form differs")
    return record


def publish_quality_preflight_shard(
    plan: QualityPreflightPlan,
    *,
    expected_plan_sha256: str,
    shard_index: int,
    evidence: Sequence[Mapping[str, object]],
    runtime_identity: Mapping[str, object],
    output_directory: str | Path,
) -> str:
    """Publish one immutable exact replay result for its assigned source shard."""

    plan_record, plan_sha = _plan_identity(plan, expected_plan_sha256)
    index = _nonnegative(shard_index, "quality-preflight shard index")
    assignments = plan_record["assignments"]
    if index >= len(assignments):
        raise QualityPreflightError("quality-preflight shard index is outside the plan")
    assignment = assignments[index]
    if not isinstance(runtime_identity, Mapping) or not runtime_identity:
        raise QualityPreflightError("quality-preflight runtime identity must be nonempty")
    canonical_json_bytes(runtime_identity)
    expected_rows = assignment["rows"]
    if len(evidence) != len(expected_rows):
        raise QualityPreflightError("quality-preflight evidence row count differs from assignment")
    records = [
        _validated_evidence(value, expected_row)
        for value, expected_row in zip(evidence, expected_rows, strict=True)
    ]
    evidence_raw = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    mismatches = [
        str(record["source_row_record_digest"]) for record in records if record["pass"] is not True
    ]
    replayed = sum(int(record["trajectory_count"]) for record in records)
    matching = sum(int(record["match_count"]) for record in records)
    payload: dict[str, object] = {
        "schema": QUALITY_PREFLIGHT_SHARD_SCHEMA,
        "schema_version": QUALITY_PREFLIGHT_SHARD_VERSION,
        "plan": {"raw_sha256": plan_sha, "record_digest": plan_record["record_digest"]},
        "identity": plan_record["identity"],
        "source_quality_shard": assignment["source_quality_shard"],
        "assigned": {
            "lineages": assignment["lineages"],
            "row_count": assignment["row_count"],
            "shard_count": plan_record["shard_count"],
            "shard_index": index,
            "task_ids": assignment["task_ids"],
            "trajectory_count": assignment["trajectory_count"],
        },
        "runtime": dict(runtime_identity),
        "evidence_sha256": hashlib.sha256(evidence_raw).hexdigest(),
        "evidence_record_set_digest": content_digest(
            [record["record_digest"] for record in records]
        ),
        "replayed_count": replayed,
        "matching_count": matching,
        "mismatch_source_row_record_digests": mismatches,
        "pass": not mismatches and replayed == matching == assignment["trajectory_count"],
    }
    manifest = _with_digest(payload)
    destination = Path(output_directory)
    if destination.exists():
        raise FileExistsError(f"quality-preflight shard already exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"quality-preflight shard parent is missing: {destination.parent}")
    destination.mkdir()
    try:
        publish_new_file(destination / "evidence.jsonl", evidence_raw)
        publish_new_file(destination / "manifest.json", canonical_json_bytes(manifest) + b"\n")
    except ExactConformanceError as error:
        raise QualityPreflightError(str(error)) from error
    return hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest()


def _load_replay_shard(
    root: str | Path,
    expected_manifest_sha256: str,
    *,
    plan_record: Mapping[str, object],
    plan_sha256: str,
) -> tuple[dict[str, object], list[dict[str, object]], str]:
    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        raise QualityPreflightError("quality-preflight shard root must be a regular directory")
    if {item.name for item in directory.iterdir()} != {"evidence.jsonl", "manifest.json"}:
        raise QualityPreflightError("quality-preflight shard file set differs")
    manifest, observed = _read_pinned_json(
        directory / "manifest.json",
        expected_manifest_sha256,
        "quality-preflight shard manifest",
    )
    if set(manifest) != _SHARD_FIELDS:
        raise QualityPreflightError("quality-preflight shard manifest fields differ")
    _verify_record(manifest, "quality-preflight shard manifest")
    if (
        manifest.get("schema") != QUALITY_PREFLIGHT_SHARD_SCHEMA
        or manifest.get("schema_version") != QUALITY_PREFLIGHT_SHARD_VERSION
        or manifest.get("plan")
        != {"raw_sha256": plan_sha256, "record_digest": plan_record["record_digest"]}
        or manifest.get("identity") != plan_record["identity"]
    ):
        raise QualityPreflightError("quality-preflight shard belongs to another plan")
    assigned = manifest.get("assigned")
    if not isinstance(assigned, Mapping):
        raise QualityPreflightError("quality-preflight shard assignment is missing")
    index = _nonnegative(assigned.get("shard_index"), "quality-preflight shard index")
    if index >= plan_record["shard_count"]:
        raise QualityPreflightError("quality-preflight shard index is outside its plan")
    assignment = plan_record["assignments"][index]
    expected_assigned = {
        "lineages": assignment["lineages"],
        "row_count": assignment["row_count"],
        "shard_count": plan_record["shard_count"],
        "shard_index": index,
        "task_ids": assignment["task_ids"],
        "trajectory_count": assignment["trajectory_count"],
    }
    if (
        dict(assigned) != expected_assigned
        or manifest.get("source_quality_shard") != assignment["source_quality_shard"]
    ):
        raise QualityPreflightError("quality-preflight shard assignment differs from its plan")
    try:
        _resolved, raw = read_regular_file(
            directory / "evidence.jsonl", "quality-preflight shard evidence"
        )
    except (ExactConformanceError, OSError) as error:
        raise QualityPreflightError(str(error)) from error
    if hashlib.sha256(raw).hexdigest() != manifest.get("evidence_sha256"):
        raise QualityPreflightError("quality-preflight shard evidence checksum mismatch")
    lines = raw.splitlines()
    if len(lines) != len(assignment["rows"]) or any(not line for line in lines):
        raise QualityPreflightError("quality-preflight shard evidence count differs")
    records = [
        _validated_evidence(
            _strict_json(line, f"quality-preflight evidence line {position}"), expected_row
        )
        for position, (line, expected_row) in enumerate(
            zip(lines, assignment["rows"], strict=True), start=1
        )
    ]
    mismatches = [
        str(record["source_row_record_digest"]) for record in records if record["pass"] is not True
    ]
    replayed = sum(int(record["trajectory_count"]) for record in records)
    matching = sum(int(record["match_count"]) for record in records)
    if (
        manifest.get("evidence_record_set_digest")
        != content_digest([record["record_digest"] for record in records])
        or manifest.get("replayed_count") != replayed
        or manifest.get("matching_count") != matching
        or manifest.get("mismatch_source_row_record_digests") != mismatches
        or manifest.get("pass")
        is not (not mismatches and replayed == matching == assignment["trajectory_count"])
    ):
        raise QualityPreflightError("quality-preflight shard totals cannot be reconstructed")
    runtime = manifest.get("runtime")
    if not isinstance(runtime, Mapping) or not runtime:
        raise QualityPreflightError("quality-preflight shard runtime identity is missing")
    canonical_json_bytes(runtime)
    return manifest, records, observed


def merge_quality_preflight_shards(
    plan: QualityPreflightPlan,
    *,
    expected_plan_sha256: str,
    replay_shards: Sequence[tuple[str | Path, str]],
    output_path: str | Path,
) -> str:
    """Authenticate one complete pinned replay census and publish its trusted bundle."""

    plan_record, plan_sha = _plan_identity(plan, expected_plan_sha256)
    loaded: dict[int, tuple[dict[str, object], list[dict[str, object]], str]] = {}
    for root, expected_manifest_sha256 in replay_shards:
        manifest, evidence, observed = _load_replay_shard(
            root,
            expected_manifest_sha256,
            plan_record=plan_record,
            plan_sha256=plan_sha,
        )
        index = int(manifest["assigned"]["shard_index"])
        if index in loaded:
            raise QualityPreflightError(f"duplicate replay shard index {index}")
        loaded[index] = manifest, evidence, observed
    missing = sorted(set(range(int(plan_record["shard_count"]))) - set(loaded))
    if missing:
        raise QualityPreflightError(f"missing replay shard indices {missing}")
    rows: list[dict[str, object]] = []
    shard_receipts: list[dict[str, object]] = []
    for index in range(int(plan_record["shard_count"])):
        manifest, evidence, observed = loaded[index]
        if manifest["pass"] is not True:
            raise QualityPreflightError(f"quality-preflight replay shard {index} did not pass")
        rows.extend(evidence)
        shard_receipts.append(
            {
                "shard_index": index,
                "manifest_sha256": observed,
                "manifest_record_digest": manifest["record_digest"],
                "evidence_sha256": manifest["evidence_sha256"],
                "evidence_record_set_digest": manifest["evidence_record_set_digest"],
                "replayed_count": manifest["replayed_count"],
                "matching_count": manifest["matching_count"],
                "runtime_digest": content_digest(manifest["runtime"]),
            }
        )
    row_digests = [str(record["source_row_record_digest"]) for record in rows]
    if len(set(row_digests)) != len(row_digests):
        raise QualityPreflightError("replay shards duplicate source row coverage")
    expected_row_digests = [
        str(row["source_row_record_digest"])
        for assignment in plan_record["assignments"]
        for row in assignment["rows"]
    ]
    if row_digests != expected_row_digests:
        raise QualityPreflightError("replay shard row coverage differs from the plan")
    trajectory_count = sum(int(record["trajectory_count"]) for record in rows)
    matching_count = sum(int(record["match_count"]) for record in rows)
    payload: dict[str, object] = {
        "schema": QUALITY_PREFLIGHT_REPLAY_BUNDLE_SCHEMA,
        "schema_version": QUALITY_PREFLIGHT_REPLAY_BUNDLE_VERSION,
        "plan": {"raw_sha256": plan_sha, "record_digest": plan_record["record_digest"]},
        "identity": plan_record["identity"],
        "source_shards": plan_record["source_shards"],
        "replay_shards": shard_receipts,
        "row_evidence": rows,
        "row_count": len(rows),
        "trajectory_count": trajectory_count,
        "replayed_count": trajectory_count,
        "matching_count": matching_count,
        "mismatch_source_row_record_digests": [],
        "row_set_digest": content_digest(sorted(row_digests)),
        "runtime_set_digest": content_digest(
            [receipt["runtime_digest"] for receipt in shard_receipts]
        ),
        "replay_shard_merkle_root": content_digest(shard_receipts),
        "pass": matching_count == trajectory_count == plan_record["trajectory_count"],
    }
    record = _with_digest(payload)
    if record["pass"] is not True:
        raise QualityPreflightError("quality-preflight replay bundle is incomplete")
    return _publish_file(Path(output_path), record)


def load_trusted_quality_preflight_replay_bundle(
    path: str | Path,
    *,
    expected_sha256: str,
    source_shards: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return row-scoped replay capabilities for the exact externally pinned shard set."""

    record, observed = _read_pinned_json(path, expected_sha256, "quality-preflight replay bundle")
    if set(record) != _BUNDLE_FIELDS:
        raise QualityPreflightError("quality-preflight replay bundle fields differ")
    _verify_record(record, "quality-preflight replay bundle")
    if (
        record.get("schema") != QUALITY_PREFLIGHT_REPLAY_BUNDLE_SCHEMA
        or record.get("schema_version") != QUALITY_PREFLIGHT_REPLAY_BUNDLE_VERSION
        or record.get("pass") is not True
    ):
        raise QualityPreflightError("quality-preflight replay bundle is not passing v1")
    plan_identity = record.get("plan")
    identity = record.get("identity")
    if (
        not isinstance(plan_identity, Mapping)
        or set(plan_identity) != {"raw_sha256", "record_digest"}
        or not isinstance(identity, Mapping)
        or not identity
    ):
        raise QualityPreflightError("replay bundle plan identity is malformed")
    _digest(plan_identity["raw_sha256"], "replay bundle plan SHA-256")
    _digest(plan_identity["record_digest"], "replay bundle plan record digest")
    canonical_json_bytes(identity)
    expected_sources, assignments = _validated_sources(
        source_shards, shard_count=len(source_shards)
    )
    if record.get("source_shards") != expected_sources:
        raise QualityPreflightError("replay bundle belongs to another quality shard set")
    replay_shards = record.get("replay_shards")
    rows = record.get("row_evidence")
    if not isinstance(replay_shards, list) or any(
        not isinstance(item, Mapping) or set(item) != _REPLAY_SHARD_IDENTITY_FIELDS
        for item in replay_shards
    ):
        raise QualityPreflightError("replay bundle shard census differs")
    if len(replay_shards) != len(expected_sources):
        raise QualityPreflightError("replay bundle shard count differs from source quality")
    for expected_index, replay_shard in enumerate(replay_shards):
        if replay_shard["shard_index"] != expected_index:
            raise QualityPreflightError("replay bundle shards are missing or out of order")
        for key in (
            "evidence_record_set_digest",
            "evidence_sha256",
            "manifest_record_digest",
            "manifest_sha256",
            "runtime_digest",
        ):
            _digest(replay_shard[key], f"replay bundle shard {expected_index} {key}")
        replayed = _positive(
            replay_shard["replayed_count"],
            f"replay bundle shard {expected_index} replayed count",
        )
        matching = _positive(
            replay_shard["matching_count"],
            f"replay bundle shard {expected_index} matching count",
        )
        if replayed != matching or replayed != expected_sources[expected_index]["trajectory_count"]:
            raise QualityPreflightError("replay bundle shard trajectory count differs")
    if not isinstance(rows, list):
        raise QualityPreflightError("replay bundle row evidence is missing")
    expected_rows = [row for assignment in assignments for row in assignment["rows"]]
    if len(rows) != len(expected_rows):
        raise QualityPreflightError("replay bundle row count differs from source quality")
    validated_rows = [
        _validated_evidence(value, expected_row)
        for value, expected_row in zip(rows, expected_rows, strict=True)
    ]
    row_digests = [str(item["source_row_record_digest"]) for item in validated_rows]
    trajectory_count = sum(int(item["trajectory_count"]) for item in validated_rows)
    matching_count = sum(int(item["match_count"]) for item in validated_rows)
    if (
        record.get("row_count") != len(validated_rows)
        or record.get("trajectory_count") != trajectory_count
        or record.get("replayed_count") != trajectory_count
        or record.get("matching_count") != matching_count
        or record.get("mismatch_source_row_record_digests") != []
        or record.get("row_set_digest") != content_digest(sorted(row_digests))
        or record.get("replay_shard_merkle_root") != content_digest(replay_shards)
        or record.get("runtime_set_digest")
        != content_digest([item["runtime_digest"] for item in replay_shards])
        or sum(int(item["replayed_count"]) for item in replay_shards) != trajectory_count
        or sum(int(item["matching_count"]) for item in replay_shards) != matching_count
        or matching_count != trajectory_count
    ):
        raise QualityPreflightError("replay bundle totals cannot be reconstructed")
    return {
        "bundle_record_digest": record["record_digest"],
        "bundle_sha256": observed,
        "matching_count": matching_count,
        "pass": True,
        "trajectory_count": trajectory_count,
        "trusted_rows": {str(item["source_row_record_digest"]): item for item in validated_rows},
    }


__all__ = [
    "QUALITY_PREFLIGHT_PLAN_SCHEMA",
    "QUALITY_PREFLIGHT_PLAN_VERSION",
    "QUALITY_PREFLIGHT_REPLAY_BUNDLE_SCHEMA",
    "QUALITY_PREFLIGHT_REPLAY_BUNDLE_VERSION",
    "QUALITY_PREFLIGHT_ROW_SCHEMA",
    "QUALITY_PREFLIGHT_ROW_VERSION",
    "QUALITY_PREFLIGHT_SHARD_SCHEMA",
    "QUALITY_PREFLIGHT_SHARD_VERSION",
    "QualityPreflightError",
    "QualityPreflightPlan",
    "build_quality_preflight_plan",
    "load_quality_preflight_plan",
    "load_trusted_quality_preflight_replay_bundle",
    "merge_quality_preflight_shards",
    "publish_quality_preflight_shard",
]
