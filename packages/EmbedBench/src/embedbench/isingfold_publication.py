"""Fail-closed publication bridge for IsingFold prepared-v4 source inputs.

This module is an assembler, not a ground-energy authority.  It accepts an
externally pinned CandidateBank manifest, publication index, and reference
index.  The publication index in turn binds the already authored corpus design
and its six evidence artifacts.  The resulting manifest reports integrity; it
does not replace an out-of-band design pin or publisher attestation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any

from embedbench.candidate_bank import (
    BankManifest,
    CandidateGroup,
    InstanceRecord,
    canonical_json_bytes,
    content_digest,
    read_bank,
    write_bank,
)
from embedbench.ground_certificate import IsingProblem

PUBLICATION_INDEX_SCHEMA = "embedbench.isingfold-publication-index"
REFERENCE_SCHEMA = "embedbench.isingfold-reference"
PUBLICATION_SCHEMA = "embedbench.isingfold-source-publication"
SCHEMA_VERSION = 1
PROVENANCE_SCHEMA = "embedbench.isingfold-task-provenance"
PROVENANCE_SCHEMA_VERSION = 2
TARGET_SCHEMA = "embedbench.evaluator-target"

STRATUM_ARTIFACTS = ("authority", "budget", "evidence", "origin", "panel", "protocol")
STRATUM_SCHEMAS = {
    "authority": "embedbench.stratum-publisher-authority",
    "budget": "embedbench.difficulty-budget",
    "evidence": "embedbench.difficulty-evidence",
    "origin": "embedbench.origin-provenance",
    "panel": "embedbench.difficulty-panel",
    "protocol": "embedbench.difficulty-protocol",
}
CERTIFIED_REFERENCE_STATUSES = frozenset(
    {"planted_proof", "exact_enumeration", "certified_optimal"}
)
TRANSFORM_KINDS = frozenset(
    {"identity", "gauge", "relabel", "topology", "fault", "mechanism", "composed"}
)
LEARNING_PARTITIONS = frozenset({"train", "val", "test"})
_HEX = frozenset("0123456789abcdef")
_MAX_CONTROL_BYTES = 256 * 1024 * 1024
_MAX_CERTIFICATE_BYTES = 64 * 1024 * 1024
_MAX_BANK_BYTES = 4 * 1024 * 1024 * 1024

_INDEX_FIELDS = {
    "corpus_design",
    "record_digest",
    "schema",
    "schema_version",
    "source_release_id",
    "source_release_manifest_sha256",
    "split_manifest_sha256",
    "strata",
    "tasks",
}
_TASK_FIELDS = {
    "active_topology_identity",
    "base_parent_lineage",
    "calibration_identity",
    "descendant_transform_identity",
    "distribution",
    "fault_identity",
    "group_id",
    "nominal_topology_identity",
}
_REFERENCE_FIELDS = {
    "certificate",
    "evaluator_protocol_digest",
    "problem_sha256",
    "record_digest",
    "reference_energy",
    "reference_status",
    "schema",
    "schema_version",
}
_DESIGN_FIELDS = {
    "axis_values",
    "corpus_design_version",
    "difficulty_calibration",
    "independent_unit",
    "lineage_registry",
    "minimum_partition_base_lineages",
    "partition_quotas",
    "power_targets",
    "precision_targets",
    "record_digest",
    "schema",
    "schema_version",
    "stratum_quotas",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be nonempty text")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{name} contains a Unicode surrogate")
    return value


def _require_schema_version(value: object, expected: int, name: str) -> int:
    if type(value) is not int or value != expected:
        raise ValueError(f"{name} requires schema_version {expected}")
    return value


def _require_exact_keys(value: object, expected: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise TypeError(f"{name} must be an object")
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} schema fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return value


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {token}")


def _closed_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _parse_json(payload: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload,
            object_pairs_hook=_closed_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{name} must contain one JSON object")
    canonical_json_bytes(value)
    return value


def _read_regular(path: Path, name: str, *, limit: int) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise ValueError(f"{name} is missing or unreadable") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be a regular file, not a symlink")
    if metadata.st_size > limit:
        raise ValueError(f"{name} exceeds its byte limit")
    payload = path.read_bytes()
    after = path.lstat()
    if (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"{name} changed while it was being read")
    return payload


def _read_canonical_json(
    path: Path,
    name: str,
    *,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular(path, name, limit=_MAX_CONTROL_BYTES)
    if expected_sha256 is not None and _sha256_bytes(payload) != _require_sha256(
        expected_sha256, f"expected {name} SHA-256"
    ):
        raise ValueError(f"{name} differs from its external SHA-256 commitment")
    value = _parse_json(payload, name)
    if payload != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{name} must be canonical JSON followed by one line feed")
    return value, payload


def _read_canonical_jsonl(
    path: Path,
    name: str,
    *,
    expected_sha256: str,
) -> tuple[list[dict[str, Any]], bytes]:
    payload = _read_regular(path, name, limit=_MAX_CONTROL_BYTES)
    if _sha256_bytes(payload) != _require_sha256(expected_sha256, f"expected {name} SHA-256"):
        raise ValueError(f"{name} differs from its external SHA-256 commitment")
    if not payload or not payload.endswith(b"\n"):
        raise ValueError(f"{name} must be nonempty and end with one line feed")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line:
            raise ValueError(f"{name} line {line_number} is blank")
        row = _parse_json(line, f"{name} line {line_number}")
        if line != canonical_json_bytes(row):
            raise ValueError(f"{name} line {line_number} is not canonical JSON")
        rows.append(row)
    return rows, payload


def _verify_record(record: Mapping[str, Any], name: str) -> str:
    digest = _require_sha256(record.get("record_digest"), f"{name} record digest")
    payload = {key: value for key, value in record.items() if key != "record_digest"}
    if content_digest(payload) != digest:
        raise ValueError(f"{name} record digest mismatch")
    return digest


def _safe_relative_path(value: object, name: str) -> PurePosixPath:
    text = _require_text(value, name)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or "\\" in text
        or "\x00" in text
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{name} must be a safe normalized relative POSIX path")
    return path


def _bound_regular_file(
    root: Path,
    relative: object,
    name: str,
    *,
    limit: int,
) -> tuple[Path, bytes]:
    path = _safe_relative_path(relative, f"{name} path")
    resolved_root = root.resolve(strict=True)
    current = resolved_root
    for part in path.parts:
        current = current / part
        try:
            metadata = current.lstat()
        except OSError as error:
            raise ValueError(f"{name} is missing or unreadable") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{name} path must not traverse a symlink")
    if not current.is_relative_to(resolved_root):
        raise ValueError(f"{name} escapes its authenticated root")
    return current, _read_regular(current, name, limit=limit)


def _descriptor(
    value: object,
    *,
    root: Path,
    name: str,
    limit: int = _MAX_CONTROL_BYTES,
) -> tuple[dict[str, str], Path, bytes]:
    raw = _require_exact_keys(value, {"path", "sha256"}, f"{name} descriptor")
    expected = _require_sha256(raw["sha256"], f"{name} SHA-256")
    path, payload = _bound_regular_file(root, raw["path"], name, limit=limit)
    if _sha256_bytes(payload) != expected:
        raise ValueError(f"{name} differs from its bound SHA-256")
    return {"path": str(raw["path"]), "sha256": expected}, path, payload


def _load_bank(
    bank_path: Path,
    manifest_path: Path,
    *,
    expected_manifest_sha256: str,
) -> tuple[tuple[InstanceRecord, ...], tuple[CandidateGroup, ...]]:
    raw_manifest, _ = _read_canonical_json(
        manifest_path,
        "CandidateBank manifest",
        expected_sha256=expected_manifest_sha256,
    )
    _require_exact_keys(
        raw_manifest,
        {"group_count", "instance_count", "jsonl_sha256", "record_digest", "schema_version"},
        "CandidateBank manifest",
    )
    _require_schema_version(
        raw_manifest["schema_version"], SCHEMA_VERSION, "CandidateBank manifest"
    )
    _require_sha256(raw_manifest["jsonl_sha256"], "CandidateBank JSONL SHA-256")
    _require_sha256(raw_manifest["record_digest"], "CandidateBank manifest record digest")
    for field in ("instance_count", "group_count"):
        if type(raw_manifest[field]) is not int or raw_manifest[field] < 0:
            raise ValueError(f"CandidateBank manifest {field} must be a non-negative integer")
    _verify_record(raw_manifest, "CandidateBank manifest")
    manifest = BankManifest(
        jsonl_sha256=raw_manifest["jsonl_sha256"],
        instance_count=raw_manifest["instance_count"],
        group_count=raw_manifest["group_count"],
        schema_version=raw_manifest["schema_version"],
        record_digest=raw_manifest["record_digest"],
    )
    before = _read_regular(bank_path, "CandidateBank", limit=_MAX_BANK_BYTES)
    if _sha256_bytes(before) != manifest.jsonl_sha256:
        raise ValueError("CandidateBank differs from its authenticated manifest")
    instances, groups = read_bank(bank_path, manifest)
    after = _read_regular(bank_path, "CandidateBank", limit=_MAX_BANK_BYTES)
    if after != before:
        raise ValueError("CandidateBank changed while it was being validated")
    return instances, groups


def _load_publication_index(
    path: Path,
    *,
    expected_sha256: str,
) -> tuple[dict[str, Any], bytes]:
    index, raw = _read_canonical_json(
        path,
        "IsingFold publication index",
        expected_sha256=expected_sha256,
    )
    _require_exact_keys(index, _INDEX_FIELDS, "IsingFold publication index")
    if index["schema"] != PUBLICATION_INDEX_SCHEMA:
        raise ValueError("unsupported IsingFold publication-index schema")
    _require_schema_version(index["schema_version"], SCHEMA_VERSION, "IsingFold publication index")
    _verify_record(index, "IsingFold publication index")
    _require_text(index["source_release_id"], "source release ID")
    _require_sha256(
        index["source_release_manifest_sha256"],
        "source release manifest SHA-256",
    )
    _require_sha256(index["split_manifest_sha256"], "split manifest SHA-256")
    if type(index["tasks"]) is not list or not index["tasks"]:
        raise ValueError("publication index tasks must be a nonempty list")
    if type(index["strata"]) is not dict:
        raise ValueError("publication index strata must be an object")
    return index, raw


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_task(
    raw: object,
    *,
    group: CandidateGroup,
    instance: InstanceRecord,
    source_release_id: str,
    source_release_manifest_sha256: str,
    split_manifest_sha256: str,
) -> dict[str, Any]:
    task = _require_exact_keys(raw, _TASK_FIELDS, "publication task")
    if task["group_id"] != group.group_id:
        raise ValueError("publication task references the wrong CandidateBank group")
    base_parent_lineage = _require_text(task["base_parent_lineage"], "base parent lineage")

    transform = _require_exact_keys(
        task["descendant_transform_identity"],
        {"kinds", "transform_sha256"},
        "descendant transform identity",
    )
    kinds = transform["kinds"]
    if (
        type(kinds) is not list
        or not kinds
        or kinds != sorted(set(kinds))
        or any(type(kind) is not str or kind not in TRANSFORM_KINDS for kind in kinds)
        or ("identity" in kinds and kinds != ["identity"])
    ):
        raise ValueError("descendant transform kinds must be a sorted registered set")
    _require_sha256(transform["transform_sha256"], "descendant transform SHA-256")

    nominal = _require_exact_keys(
        task["nominal_topology_identity"],
        {"pristine_host_sha256", "size", "topology"},
        "nominal topology identity",
    )
    active = _require_exact_keys(
        task["active_topology_identity"],
        {"host_artifact_sha256", "host_sha256", "topology"},
        "active topology identity",
    )
    if nominal["topology"] != instance.topology or active["topology"] != instance.topology:
        raise ValueError("publication task topology differs from its CandidateBank instance")
    _positive_int(nominal["size"], "nominal topology size")
    _require_sha256(nominal["pristine_host_sha256"], "pristine host SHA-256")
    host_artifact_sha256 = _require_sha256(
        active["host_artifact_sha256"], "active host artifact SHA-256"
    )
    expected_host_sha256 = content_digest(
        {
            "edges": instance.host_edges,
            "nodes": instance.host_nodes,
            "schema": "embedbench.host-graph",
            "schema_version": 1,
        }
    )
    if active["host_sha256"] != expected_host_sha256:
        raise ValueError("active host identity differs from the CandidateBank host graph")

    fault = _require_exact_keys(
        task["fault_identity"], {"fault_mask_sha256", "status"}, "fault identity"
    )
    if fault["status"] not in {"none", "faulted"}:
        raise ValueError("fault status must be explicit")
    if fault["fault_mask_sha256"] != host_artifact_sha256:
        raise ValueError("fault identity differs from the active host artifact")
    if (fault["status"] == "faulted") != ("fault" in kinds):
        raise ValueError("fault status and descendant transform disagree")

    calibration = _require_exact_keys(
        task["calibration_identity"],
        {"calibration_sha256", "status"},
        "calibration identity",
    )
    if calibration["status"] == "not_applicable":
        if calibration["calibration_sha256"] is not None:
            raise ValueError("not-applicable calibration cannot carry a digest")
    elif calibration["status"] == "recorded":
        _require_sha256(calibration["calibration_sha256"], "calibration SHA-256")
    else:
        raise ValueError("calibration status must be explicit")

    distribution = _require_exact_keys(
        task["distribution"],
        {"learning_partition", "regime", "source_partition", "stratum"},
        "distribution identity",
    )
    if distribution["learning_partition"] not in LEARNING_PARTITIONS:
        raise ValueError("unknown learning partition")
    if distribution["regime"] not in {"iid", "ood"}:
        raise ValueError("distribution regime must be iid or ood")
    if distribution["regime"] == "ood" and distribution["learning_partition"] != "test":
        raise ValueError("OOD tasks must remain in the sealed test partition")
    _require_text(distribution["source_partition"], "source partition")
    _require_text(distribution["stratum"], "distribution stratum")

    payload = {
        "active_topology_identity": active,
        "base_parent_lineage": base_parent_lineage,
        "calibration_identity": calibration,
        "descendant_transform_identity": transform,
        "distribution": distribution,
        "fault_identity": fault,
        "group_id": group.group_id,
        "group_record_digest": group.record_digest,
        "instance_id": instance.instance_id,
        "instance_record_digest": instance.record_digest,
        "nominal_topology_identity": nominal,
        "schema": PROVENANCE_SCHEMA,
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "source_release_id": source_release_id,
        "source_release_manifest_sha256": source_release_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
    }
    return {**payload, "record_digest": content_digest(payload)}


def _build_provenance(
    index: Mapping[str, Any],
    *,
    instances: Sequence[InstanceRecord],
    groups: Sequence[CandidateGroup],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    groups_by_id = {group.group_id: group for group in groups}
    instances_by_id = {instance.instance_id: instance for instance in instances}
    raw_tasks = index["tasks"]
    task_ids = [task.get("group_id") if type(task) is dict else None for task in raw_tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("publication index repeats a CandidateBank group")
    if set(task_ids) != set(groups_by_id):
        raise ValueError("publication task coverage differs from the CandidateBank groups")
    if task_ids != sorted(task_ids):
        raise ValueError("publication tasks must be sorted by group_id")

    records: list[dict[str, Any]] = []
    lineage_partitions: dict[str, str] = {}
    source_lineage_parents: dict[str, str] = {}
    transform_parents: dict[str, str] = {}
    instance_partitions: dict[str, str] = {}
    for raw in raw_tasks:
        group = groups_by_id[raw["group_id"]]
        instance = instances_by_id[group.instance_id]
        record = _validate_task(
            raw,
            group=group,
            instance=instance,
            source_release_id=index["source_release_id"],
            source_release_manifest_sha256=index["source_release_manifest_sha256"],
            split_manifest_sha256=index["split_manifest_sha256"],
        )
        partition = record["distribution"]["learning_partition"]
        lineage = record["base_parent_lineage"]
        if lineage_partitions.setdefault(lineage, partition) != partition:
            raise ValueError("one base lineage appears in multiple learning partitions")
        if source_lineage_parents.setdefault(group.split_unit_id, lineage) != lineage:
            raise ValueError("one source logical lineage maps to multiple base parents")
        transform_sha256 = record["descendant_transform_identity"]["transform_sha256"]
        if transform_parents.setdefault(transform_sha256, lineage) != lineage:
            raise ValueError("one descendant transform identity maps to multiple base parents")
        if instance_partitions.setdefault(instance.instance_id, partition) != partition:
            raise ValueError("one CandidateBank instance appears in multiple learning partitions")
        records.append(record)
    return records, instance_partitions


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _problem_identity(instance: InstanceRecord) -> str:
    return IsingProblem(
        variables=instance.logical_nodes,
        linear=instance.h,
        quadratic=instance.j,
    ).problem_sha256


def _certificate_problem_binding(payload: bytes, name: str) -> str | None:
    """Return an optional embedded problem identity from an opaque certificate.

    Certificates are allowed to use proof-system-specific encodings.  JSON
    certificates that expose a ``problem_sha256`` field are nevertheless checked
    against the authenticated reference row, preventing a valid file checksum from
    being accidentally attached to another problem.
    """

    try:
        value = json.loads(
            payload,
            object_pairs_hook=_closed_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping) or "problem_sha256" not in value:
        return None
    return _require_sha256(value["problem_sha256"], f"{name} problem SHA-256")


def _load_references(
    path: Path,
    *,
    expected_sha256: str,
    instances: Sequence[InstanceRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], dict[str, bytes]]:
    rows, _ = _read_canonical_jsonl(
        path,
        "IsingFold reference index",
        expected_sha256=expected_sha256,
    )
    references: dict[str, dict[str, Any]] = {}
    certificates: dict[str, bytes] = {}
    certificate_problems: dict[str, str] = {}
    order: list[str] = []
    for line_number, raw in enumerate(rows, start=1):
        reference = _require_exact_keys(
            raw,
            _REFERENCE_FIELDS,
            f"reference index line {line_number}",
        )
        if reference["schema"] != REFERENCE_SCHEMA:
            raise ValueError("unsupported IsingFold reference schema")
        _require_schema_version(
            reference["schema_version"],
            SCHEMA_VERSION,
            f"reference index line {line_number}",
        )
        _verify_record(reference, f"reference index line {line_number}")
        problem_sha256 = _require_sha256(
            reference["problem_sha256"],
            f"reference index line {line_number} problem SHA-256",
        )
        if problem_sha256 in references:
            raise ValueError("reference index repeats a logical problem")
        evaluator_protocol_digest = _require_sha256(
            reference["evaluator_protocol_digest"],
            f"reference index line {line_number} evaluator protocol digest",
        )
        reference_status = reference["reference_status"]
        if reference_status not in CERTIFIED_REFERENCE_STATUSES:
            raise ValueError("reference index contains a non-certified reference status")
        reference_energy = _finite(
            reference["reference_energy"],
            f"reference index line {line_number} reference energy",
        )
        descriptor, _, certificate = _descriptor(
            reference["certificate"],
            root=path.parent,
            name=f"certificate for problem {problem_sha256}",
            limit=_MAX_CERTIFICATE_BYTES,
        )
        certificate_sha256 = descriptor["sha256"]
        embedded_problem = _certificate_problem_binding(
            certificate,
            f"certificate for problem {problem_sha256}",
        )
        if embedded_problem is not None and embedded_problem != problem_sha256:
            raise ValueError("certificate problem identity differs from its reference row")
        previous_problem = certificate_problems.setdefault(certificate_sha256, problem_sha256)
        if previous_problem != problem_sha256:
            raise ValueError("one certificate artifact is bound to multiple logical problems")
        if certificates.setdefault(certificate_sha256, certificate) != certificate:
            raise RuntimeError("equal certificate digests unexpectedly carry different bytes")
        references[problem_sha256] = {
            "certificate_digest": certificate_sha256,
            "evaluator_protocol_digest": evaluator_protocol_digest,
            "reference_energy": reference_energy,
            "reference_status": reference_status,
        }
        order.append(problem_sha256)
    if order != sorted(order):
        raise ValueError("reference index must be sorted by problem_sha256")

    instances_by_id = {instance.instance_id: instance for instance in instances}
    if len(instances_by_id) != len(instances):
        raise ValueError("CandidateBank repeats an instance identity")
    required_problems = {_problem_identity(instance) for instance in instances}
    if set(references) != required_problems:
        missing = sorted(required_problems - set(references))
        unknown = sorted(set(references) - required_problems)
        raise ValueError(
            f"reference problem coverage mismatch: missing={missing}, unknown={unknown}"
        )

    targets: list[dict[str, Any]] = []
    for instance in sorted(instances, key=lambda item: item.instance_id):
        reference = references[_problem_identity(instance)]
        targets.append(
            {
                **reference,
                "instance_id": instance.instance_id,
                "instance_record_digest": instance.record_digest,
                "schema": TARGET_SCHEMA,
                "schema_version": SCHEMA_VERSION,
            }
        )
    certificate_mapping = [
        {
            "path": f"certificates/{certificate_sha256}",
            "problem_sha256": problem_sha256,
            "sha256": certificate_sha256,
        }
        for certificate_sha256, problem_sha256 in sorted(
            certificate_problems.items(), key=lambda item: item[1]
        )
    ]
    return targets, certificate_mapping, certificates


def _load_design(
    index: Mapping[str, Any],
    *,
    index_path: Path,
) -> tuple[bytes, dict[str, bytes]]:
    _, _, design_raw = _descriptor(
        index["corpus_design"],
        root=index_path.parent,
        name="corpus design",
    )
    design = _parse_json(design_raw, "corpus design")
    if design_raw != canonical_json_bytes(design) + b"\n":
        raise ValueError("corpus design must be canonical JSON followed by one line feed")
    _require_exact_keys(design, _DESIGN_FIELDS, "corpus design")
    if (
        design["schema"] != "isingfold.corpus-design"
        or design["corpus_design_version"] != "if-core-v2"
    ):
        raise ValueError("publication requires the registered IsingFold corpus design v2")
    _require_schema_version(design["schema_version"], 2, "corpus design")
    if design["independent_unit"] != "immutable-base-lineage":
        raise ValueError("corpus design must use immutable base lineages")
    _verify_record(design, "corpus design")

    index_strata = _require_exact_keys(
        index["strata"], set(STRATUM_ARTIFACTS), "publication-index strata"
    )
    calibration = _require_exact_keys(
        design["difficulty_calibration"],
        {*STRATUM_ARTIFACTS, "outcome_blind", "publisher_id"},
        "corpus-design difficulty calibration",
    )
    if calibration["outcome_blind"] is not True:
        raise ValueError("corpus-design difficulty calibration must be outcome-blind")
    publisher_id = _require_text(calibration["publisher_id"], "stratum publisher ID")

    strata: dict[str, bytes] = {}
    stratum_records: dict[str, dict[str, Any]] = {}
    for name in STRATUM_ARTIFACTS:
        index_descriptor, _, raw = _descriptor(
            index_strata[name],
            root=index_path.parent,
            name=f"stratum {name}",
        )
        record = _parse_json(raw, f"stratum {name}")
        if raw != canonical_json_bytes(record) + b"\n":
            raise ValueError(f"stratum {name} must be canonical JSON followed by one line feed")
        if record.get("schema") != STRATUM_SCHEMAS[name]:
            raise ValueError(f"unsupported stratum {name} schema")
        _require_schema_version(record.get("schema_version"), 1, f"stratum {name}")
        _verify_record(record, f"stratum {name}")
        design_descriptor = _require_exact_keys(
            calibration[name], {"path", "sha256"}, f"corpus-design stratum {name} descriptor"
        )
        expected_output_path = f"strata/{name}.json"
        if (
            design_descriptor["path"] != expected_output_path
            or design_descriptor["sha256"] != index_descriptor["sha256"]
        ):
            raise ValueError(
                f"corpus-design stratum {name} descriptor differs from the publication index"
            )
        strata[name] = raw
        stratum_records[name] = record

    authority = stratum_records["authority"]
    if (
        authority.get("publisher_id") != publisher_id
        or authority.get("source_release_id") != index["source_release_id"]
        or authority.get("source_release_manifest_sha256")
        != index["source_release_manifest_sha256"]
    ):
        raise ValueError("stratum authority differs from the publication identity")
    origin = stratum_records["origin"]
    if (
        origin.get("publisher_id") != publisher_id
        or origin.get("source_release_id") != index["source_release_id"]
        or origin.get("source_release_manifest_sha256") != index["source_release_manifest_sha256"]
    ):
        raise ValueError("origin provenance differs from the publication identity")
    return design_raw, strata


def _jsonl_bytes(records: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _file_inventory(files: Mapping[str, bytes]) -> list[dict[str, Any]]:
    return [
        {
            "byte_count": len(files[path]),
            "path": path,
            "sha256": _sha256_bytes(files[path]),
        }
        for path in sorted(files)
    ]


def _stage_publication(
    stage: Path,
    *,
    instances: Sequence[InstanceRecord],
    groups: Sequence[CandidateGroup],
    provenance: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    design_raw: bytes,
    strata: Mapping[str, bytes],
    certificates: Mapping[str, bytes],
    certificate_mapping: Sequence[Mapping[str, str]],
    index: Mapping[str, Any],
    source_inputs: Mapping[str, str],
) -> dict[str, Any]:
    bank_path = stage / "candidate_bank_v2.jsonl"
    output_bank_manifest = write_bank(bank_path, instances=instances, groups=groups)
    files: dict[str, bytes] = {
        "candidate_bank_v2.jsonl": _read_regular(
            bank_path, "staged CandidateBank", limit=_MAX_BANK_BYTES
        ),
        "candidate_bank_v2.manifest.json": canonical_json_bytes(asdict(output_bank_manifest))
        + b"\n",
        "corpus_design_v2.json": design_raw,
        "evaluator_targets.jsonl": _jsonl_bytes(targets),
        "isingfold_task_provenance.jsonl": _jsonl_bytes(provenance),
    }
    for name in STRATUM_ARTIFACTS:
        files[f"strata/{name}.json"] = strata[name]
    for certificate_sha256, payload in sorted(certificates.items()):
        files[f"certificates/{certificate_sha256}"] = payload

    # ``write_bank`` has already created this file.  Every other payload is opened
    # exclusively so an internal path collision cannot silently replace bytes.
    for relative, payload in sorted(files.items()):
        if relative != "candidate_bank_v2.jsonl":
            _write_exclusive(stage / relative, payload)

    manifest_payload = {
        "candidate_bank": {
            "manifest_path": "candidate_bank_v2.manifest.json",
            "manifest_record_digest": output_bank_manifest.record_digest,
            "manifest_sha256": _sha256_bytes(files["candidate_bank_v2.manifest.json"]),
            "path": "candidate_bank_v2.jsonl",
            "sha256": output_bank_manifest.jsonl_sha256,
        },
        "certificates": list(certificate_mapping),
        "corpus_design": {
            "path": "corpus_design_v2.json",
            "sha256": _sha256_bytes(design_raw),
        },
        "counts": {
            "candidate_groups": len(groups),
            "certificates": len(certificates),
            "evaluator_targets": len(targets),
            "instances": len(instances),
            "provenance_records": len(provenance),
        },
        "files": _file_inventory(files),
        "schema": PUBLICATION_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "source_inputs": dict(sorted(source_inputs.items())),
        "source_release_id": index["source_release_id"],
        "source_release_manifest_sha256": index["source_release_manifest_sha256"],
        "split_manifest_sha256": index["split_manifest_sha256"],
    }
    manifest = {**manifest_payload, "record_digest": content_digest(manifest_payload)}
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    _write_exclusive(stage / "publication_manifest.json", manifest_raw)
    checksum_files = {**files, "publication_manifest.json": manifest_raw}
    checksum_raw = "".join(
        f"{_sha256_bytes(checksum_files[path])}  {path}\n" for path in sorted(checksum_files)
    ).encode("utf-8")
    _write_exclusive(stage / "SHA256SUMS", checksum_raw)
    return manifest


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _publish_staged(stage: Path, destination: Path) -> None:
    if _path_exists(destination):
        raise FileExistsError(f"publication output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir()
    except FileExistsError:
        raise FileExistsError(f"publication output already exists: {destination}") from None
    created = destination.lstat()
    try:
        paths = sorted(
            path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()
        )
        # The self-digested publication manifest is the protocol-level commit marker.
        # It is linked only after every payload and checksum table is durable.
        paths.remove("publication_manifest.json")
        for relative in paths:
            target = destination.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(stage / relative, target)
        for directory in sorted(
            (path for path in destination.rglob("*") if path.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            _fsync_directory(directory)
        _fsync_directory(destination)
        os.link(
            stage / "publication_manifest.json",
            destination / "publication_manifest.json",
        )
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
    except BaseException:
        try:
            current = destination.lstat()
        except FileNotFoundError:
            current = None
        if current is not None and (current.st_dev, current.st_ino) == (
            created.st_dev,
            created.st_ino,
        ):
            shutil.rmtree(destination)
        raise


def publish_isingfold_source(
    *,
    bank_path: str | os.PathLike[str],
    bank_manifest_path: str | os.PathLike[str],
    publication_index_path: str | os.PathLike[str],
    reference_index_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    expected_bank_manifest_sha256: str,
    expected_publication_index_sha256: str,
    expected_reference_index_sha256: str,
) -> dict[str, Any]:
    """Validate and publish the complete source bundle consumed by IsingFold v4.

    The three expected SHA-256 values are mandatory external commitments.  A
    self-digest inside any supplied file establishes integrity only; it cannot make
    that file authoritative.  ``output_dir`` is immutable and must not already exist.
    """

    bank = Path(bank_path)
    bank_manifest = Path(bank_manifest_path)
    publication_index = Path(publication_index_path)
    reference_index = Path(reference_index_path)
    destination = Path(output_dir)
    if _path_exists(destination):
        raise FileExistsError(f"publication output already exists: {destination}")
    instances, groups = _load_bank(
        bank,
        bank_manifest,
        expected_manifest_sha256=expected_bank_manifest_sha256,
    )
    index, _ = _load_publication_index(
        publication_index,
        expected_sha256=expected_publication_index_sha256,
    )
    provenance, _ = _build_provenance(index, instances=instances, groups=groups)
    targets, certificate_mapping, certificates = _load_references(
        reference_index,
        expected_sha256=expected_reference_index_sha256,
        instances=instances,
    )
    design_raw, strata = _load_design(index, index_path=publication_index)

    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.publish-", dir=destination.parent))
    try:
        manifest = _stage_publication(
            stage,
            instances=instances,
            groups=groups,
            provenance=provenance,
            targets=targets,
            design_raw=design_raw,
            strata=strata,
            certificates=certificates,
            certificate_mapping=certificate_mapping,
            index=index,
            source_inputs={
                "candidate_bank_manifest_sha256": _require_sha256(
                    expected_bank_manifest_sha256,
                    "expected CandidateBank manifest SHA-256",
                ),
                "publication_index_sha256": _require_sha256(
                    expected_publication_index_sha256,
                    "expected publication-index SHA-256",
                ),
                "reference_index_sha256": _require_sha256(
                    expected_reference_index_sha256,
                    "expected reference-index SHA-256",
                ),
            },
        )
        _publish_staged(stage, destination)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


__all__ = [
    "PUBLICATION_INDEX_SCHEMA",
    "PUBLICATION_SCHEMA",
    "REFERENCE_SCHEMA",
    "publish_isingfold_source",
]
