"""Publish and independently verify IsingFold quality-attestation v2 bundles.

This module is a post-prepare publisher boundary.  It consumes only the public,
partition-sealed prepared-v4 wire format and certificate bytes.  In particular, it
does not import the IsingFold training runtime.  Record digests use EmbedBench's
canonical JSON convention; file checksums cover the exact published bytes.
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
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from embedbench.candidate_bank import canonical_json_bytes, content_digest

PARTITIONS = ("train", "val", "test")
PREPARED_SCHEMA = "isingfold.prepared-candidate-bank"
PREPARED_SCHEMA_VERSION = 4
PREPARED_SCOPE = "production-designed-v4"
TARGET_SCHEMA = "isingfold.evaluator-target"
TARGET_SCHEMA_VERSION = 2
TARGET_AUTHORITY_SCHEMA = "isingfold.partitioned-target-authority"
TARGET_AUTHORITY_VERSION = 1
PROVENANCE_SCHEMA = "isingfold.task-provenance"
PROVENANCE_VERSION = 2
PUBLISHER_ATTESTATION_SCHEMA = "isingfold.quality-publisher-attestation"
PUBLISHER_ATTESTATION_VERSION = 2
PUBLISHER_ATTESTATION_STATEMENT = "publisher-attests-ground-energy-evidence"
QUALITY_EVIDENCE_SCHEMA = "isingfold.quality-ground-evidence"
QUALITY_EVIDENCE_VERSION = 2
EVIDENCE_KIND_BY_REFERENCE_STATUS = {
    "planted_proof": "planted-witness",
    "exact_enumeration": "enumeration-transcript",
    "certified_optimal": "optimality-certificate",
}
_HEX = frozenset("0123456789abcdef")
_PREPARED_OUTPUTS = frozenset(
    {
        "initializers.jsonl",
        "policy_instances.jsonl",
        "provenance.jsonl",
        "splits.json",
        "targets/train.jsonl",
        "targets/val.jsonl",
        "targets/test.jsonl",
    }
)
_PREPARED_MANIFEST_FIELDS = {
    "corpus_design",
    "corpus_scope",
    "counts",
    "outputs",
    "policy_model_feature_allowlist",
    "provenance_record_schema",
    "qubit_cap",
    "record_digest",
    "schema",
    "schema_version",
    "source_bank_manifest_record_digest",
    "source_sha256",
    "target_authority",
}
_TARGET_FIELDS = {
    "certificate_digest",
    "evaluator_protocol_digest",
    "instance_id",
    "instance_record_digest",
    "learning_partition",
    "record_digest",
    "reference_energy",
    "reference_status",
    "schema",
    "schema_version",
}
_PROVENANCE_FIELDS = {
    "active_topology_identity",
    "base_parent_lineage",
    "calibration_identity",
    "descendant_transform_identity",
    "distribution",
    "fault_identity",
    "group_id",
    "group_record_digest",
    "instance_id",
    "instance_record_digest",
    "nominal_topology_identity",
    "record_digest",
    "schema",
    "schema_version",
    "source_logical_lineage",
    "source_record_digest",
    "source_release_id",
    "source_release_manifest_sha256",
    "split_manifest_sha256",
    "task_id",
}


class PublisherAttestationError(ValueError):
    """A prepared corpus or publisher-attestation bundle is invalid."""


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise PublisherAttestationError(f"{name} must be non-empty text")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise PublisherAttestationError(f"{name} contains a Unicode surrogate")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in _HEX for char in value):
        raise PublisherAttestationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive " if positive else "non-negative "
        raise PublisherAttestationError(f"{name} must be a {qualifier}integer")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PublisherAttestationError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise PublisherAttestationError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _exact_fields(value: object, expected: set[str], name: str) -> dict[str, Any]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise PublisherAttestationError(f"{name} must be an exact JSON object")
    actual = set(value)
    if actual != expected:
        raise PublisherAttestationError(
            f"{name} schema fields differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return value


def _strict_object(raw: bytes, name: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise PublisherAttestationError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise PublisherAttestationError(f"{name} contains non-finite number {token}")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        canonical_json_bytes(value)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        if isinstance(error, PublisherAttestationError):
            raise
        raise PublisherAttestationError(f"{name} is invalid JSON") from error
    if type(value) is not dict:
        raise PublisherAttestationError(f"{name} must be a JSON object")
    return value


def _read_regular(path: Path, name: str) -> bytes:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise PublisherAttestationError(f"{name} is missing or not a regular file")
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise PublisherAttestationError(f"cannot read {name}: {error}") from error
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or len(raw) != before.st_size:
        raise PublisherAttestationError(f"{name} changed while it was being read")
    return raw


def _canonical_object(path: Path, name: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, name)
    value = _strict_object(raw, name)
    if raw != canonical_json_bytes(value) + b"\n":
        raise PublisherAttestationError(f"{name} is not canonical JSON")
    return value, raw


def _canonical_jsonl(path: Path, name: str) -> tuple[list[dict[str, Any]], bytes]:
    raw = _read_regular(path, name)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise PublisherAttestationError(f"{name} line {line_number} is blank")
        rows.append(_strict_object(line, f"{name} line {line_number}"))
    if not rows:
        raise PublisherAttestationError(f"{name} must not be empty")
    expected = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    if raw != expected:
        raise PublisherAttestationError(f"{name} is not canonical JSONL")
    return rows, raw


def _verify_record(record: Mapping[str, object], name: str) -> str:
    recorded = _sha256(record.get("record_digest"), f"{name} record digest")
    payload = {key: value for key, value in record.items() if key != "record_digest"}
    try:
        observed = content_digest(payload)
    except (TypeError, ValueError) as error:
        raise PublisherAttestationError(f"{name} contains non-canonical data") from error
    if observed != recorded:
        raise PublisherAttestationError(f"{name} record digest mismatch")
    return recorded


def _safe_relative_path(value: object, name: str) -> str:
    text = _text(value, name)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or "\\" in text
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise PublisherAttestationError(f"{name} must be a safe relative path")
    return text


def _directory(path: str | os.PathLike[str], name: str) -> Path:
    candidate = Path(path)
    try:
        if candidate.is_symlink() or not candidate.is_dir():
            raise PublisherAttestationError(f"{name} is missing or not a directory")
        return candidate.resolve(strict=True)
    except OSError as error:
        raise PublisherAttestationError(f"cannot open {name}: {error}") from error


def _bound_file(root: Path, relative: object, name: str) -> tuple[Path, str]:
    text = _safe_relative_path(relative, f"{name} path")
    candidate = root.joinpath(*PurePosixPath(text).parts)
    cursor = root
    for part in PurePosixPath(text).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise PublisherAttestationError(f"{name} path traverses a symbolic link")
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise PublisherAttestationError(f"{name} is missing") from error
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise PublisherAttestationError(f"{name} escapes its publication directory")
    return resolved, text


def _file_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _target_set_digest(targets: Sequence[Mapping[str, object]], partition: str) -> str:
    return content_digest(
        {
            "domain": "isingfold-partition-target-set-v1",
            "partition": partition,
            "targets": [
                {
                    "instance_id": target["instance_id"],
                    "target_record_digest": target["record_digest"],
                }
                for target in sorted(targets, key=lambda row: str(row["instance_id"]))
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class VerifierIdentity:
    """Publisher-pinned identity of the certificate verifier implementation."""

    name: str
    version: str
    implementation_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _text(self.name, "verifier name"))
        object.__setattr__(self, "version", _text(self.version, "verifier version"))
        object.__setattr__(
            self,
            "implementation_sha256",
            _sha256(self.implementation_sha256, "verifier implementation SHA-256"),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "implementation_sha256": self.implementation_sha256,
            "name": self.name,
            "version": self.version,
        }


@dataclass(frozen=True, slots=True)
class PublisherAttestationReceipt:
    """Stable trust anchors and counts for one verified publication."""

    output_directory: Path
    attestation_path: Path
    attestation_record_digest: str
    attestation_sha256: str
    publisher_id: str
    publication_id: str
    prepared_manifest_sha256: str
    target_count: int
    certificate_count: int
    verifier: VerifierIdentity


@dataclass(frozen=True, slots=True)
class _PreparedV4:
    root: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    target_authority_record_digest: str
    targets: Mapping[str, tuple[Mapping[str, object], ...]]
    target_raw_sha256: Mapping[str, str]
    target_set_digests: Mapping[str, str]
    source_release_id: str
    source_release_manifest_sha256: str


def _validate_output_receipt(
    value: object,
    *,
    expected_records: int,
    observed_raw: bytes,
    name: str,
) -> None:
    descriptor = _exact_fields(value, {"records", "sha256"}, name)
    if _integer(descriptor["records"], f"{name} records", positive=True) != expected_records:
        raise PublisherAttestationError(f"{name} record count mismatch")
    expected_sha = _sha256(descriptor["sha256"], f"{name} SHA-256")
    if expected_sha != _file_sha256(observed_raw):
        raise PublisherAttestationError(f"{name} SHA-256 mismatch")


def _validate_prepared_v4(path: str | os.PathLike[str]) -> _PreparedV4:
    root = _directory(path, "prepared-v4 directory")
    manifest, manifest_raw = _canonical_object(root / "manifest.json", "prepared manifest")
    _exact_fields(manifest, _PREPARED_MANIFEST_FIELDS, "prepared manifest")
    if (
        manifest["schema"] != PREPARED_SCHEMA
        or manifest["schema_version"] != PREPARED_SCHEMA_VERSION
        or manifest["corpus_scope"] != PREPARED_SCOPE
    ):
        raise PublisherAttestationError("prepared manifest is not production prepared-v4")
    _verify_record(manifest, "prepared manifest")
    _integer(manifest["qubit_cap"], "prepared qubit cap", positive=True)
    _sha256(
        manifest["source_bank_manifest_record_digest"],
        "source CandidateBank manifest record digest",
    )
    if manifest["provenance_record_schema"] != PROVENANCE_SCHEMA:
        raise PublisherAttestationError("prepared provenance schema is unsupported")
    expected_allowlist = [
        "family",
        "h",
        "host_edges",
        "host_nodes",
        "j",
        "logical_edges",
        "logical_nodes",
        "topology",
    ]
    if manifest["policy_model_feature_allowlist"] != expected_allowlist:
        raise PublisherAttestationError("prepared policy feature allowlist differs from v4")
    source_sha = _exact_fields(
        manifest["source_sha256"],
        {
            "candidate_bank_jsonl",
            "candidate_bank_manifest",
            "corpus_design_manifest",
            "evaluator_targets",
            "task_provenance",
        },
        "prepared source checksums",
    )
    for name, digest in source_sha.items():
        _sha256(digest, f"prepared source {name} SHA-256")
    design = manifest["corpus_design"]
    if not isinstance(design, Mapping) or (
        design.get("schema") != "isingfold.prepared-corpus-design-receipt"
        or design.get("schema_version") != 2
        or design.get("corpus_design_version") != "if-core-v2"
    ):
        raise PublisherAttestationError("prepared corpus-design receipt is not IF-Core v2")

    counts = _exact_fields(
        manifest["counts"],
        {
            "evaluator_targets",
            "evaluator_targets_by_partition",
            "initializers",
            "policy_instances",
            "provenance_records",
        },
        "prepared counts",
    )
    total_targets = _integer(counts["evaluator_targets"], "evaluator target count", positive=True)
    initializer_count = _integer(counts["initializers"], "initializer count", positive=True)
    policy_count = _integer(counts["policy_instances"], "policy instance count", positive=True)
    provenance_count = _integer(
        counts["provenance_records"], "provenance record count", positive=True
    )
    partition_counts = _exact_fields(
        counts["evaluator_targets_by_partition"],
        set(PARTITIONS),
        "prepared partition target counts",
    )
    for partition in PARTITIONS:
        _integer(partition_counts[partition], f"{partition} target count", positive=True)
    if sum(int(partition_counts[partition]) for partition in PARTITIONS) != total_targets:
        raise PublisherAttestationError("prepared partition target counts do not sum to total")

    outputs = _exact_fields(manifest["outputs"], set(_PREPARED_OUTPUTS), "prepared outputs")
    ordinary_specs = {
        "initializers.jsonl": (initializer_count, "prepared initializers"),
        "policy_instances.jsonl": (policy_count, "prepared policy instances"),
        "provenance.jsonl": (provenance_count, "prepared provenance"),
    }
    ordinary_rows: dict[str, list[dict[str, Any]]] = {}
    for relative, (expected_count, label) in ordinary_specs.items():
        file_path, _ = _bound_file(root, relative, label)
        rows, raw = _canonical_jsonl(file_path, label)
        if len(rows) != expected_count:
            raise PublisherAttestationError(f"{label} record count mismatch")
        for index, row in enumerate(rows):
            _verify_record(row, f"{label} row {index}")
        _validate_output_receipt(
            outputs[relative],
            expected_records=len(rows),
            observed_raw=raw,
            name=f"{label} output receipt",
        )
        ordinary_rows[relative] = rows

    for row in ordinary_rows["provenance.jsonl"]:
        _exact_fields(row, _PROVENANCE_FIELDS, "prepared provenance row")
        if row["schema"] != PROVENANCE_SCHEMA or row["schema_version"] != PROVENANCE_VERSION:
            raise PublisherAttestationError("prepared provenance row has an unsupported schema")
        _text(row["source_release_id"], "provenance source release ID")
        _sha256(
            row["source_release_manifest_sha256"],
            "provenance source release manifest SHA-256",
        )
    release_identities = {
        (row["source_release_id"], row["source_release_manifest_sha256"])
        for row in ordinary_rows["provenance.jsonl"]
    }
    if len(release_identities) != 1:
        raise PublisherAttestationError("prepared provenance mixes source release identities")
    source_release_id, source_release_sha = next(iter(release_identities))

    splits_path, _ = _bound_file(root, "splits.json", "prepared splits")
    splits, splits_raw = _canonical_object(splits_path, "prepared splits")
    _verify_record(splits, "prepared splits")
    if splits.get("schema") != "isingfold.lineage-splits" or splits.get("schema_version") != 4:
        raise PublisherAttestationError("prepared splits has an unsupported schema")
    _validate_output_receipt(
        outputs["splits.json"],
        expected_records=1,
        observed_raw=splits_raw,
        name="prepared splits output receipt",
    )

    authority = _exact_fields(
        manifest["target_authority"],
        {"partitions", "record_digest", "schema", "schema_version", "total_targets"},
        "prepared target authority",
    )
    if (
        authority["schema"] != TARGET_AUTHORITY_SCHEMA
        or authority["schema_version"] != TARGET_AUTHORITY_VERSION
    ):
        raise PublisherAttestationError("prepared target authority has an unsupported schema")
    authority_digest = _verify_record(authority, "prepared target authority")
    if (
        _integer(authority["total_targets"], "target-authority total", positive=True)
        != total_targets
    ):
        raise PublisherAttestationError("prepared target-authority total mismatch")
    authority_partitions = _exact_fields(
        authority["partitions"], set(PARTITIONS), "prepared target-authority partitions"
    )

    policy_instances = {
        _text(row.get("instance_id"), "policy instance ID")
        for row in ordinary_rows["policy_instances.jsonl"]
    }
    if len(policy_instances) != policy_count:
        raise PublisherAttestationError("prepared policy instances repeat an instance ID")
    all_target_instances: set[str] = set()
    targets_by_partition: dict[str, tuple[Mapping[str, object], ...]] = {}
    target_raw_sha: dict[str, str] = {}
    target_set_digests: dict[str, str] = {}
    for partition in PARTITIONS:
        descriptor = _exact_fields(
            authority_partitions[partition],
            {"path", "records", "sha256", "target_set_digest"},
            f"prepared {partition} target descriptor",
        )
        expected_relative = f"targets/{partition}.jsonl"
        if _safe_relative_path(descriptor["path"], f"{partition} target path") != expected_relative:
            raise PublisherAttestationError(f"prepared {partition} target path is not canonical")
        target_path, _ = _bound_file(root, expected_relative, f"prepared {partition} targets")
        rows, raw = _canonical_jsonl(target_path, f"prepared {partition} targets")
        expected_count = _integer(
            partition_counts[partition], f"prepared {partition} target count", positive=True
        )
        descriptor_count = _integer(
            descriptor["records"],
            f"prepared {partition} target descriptor count",
            positive=True,
        )
        if len(rows) != expected_count or descriptor_count != expected_count:
            raise PublisherAttestationError(f"prepared {partition} target count mismatch")
        raw_sha = _file_sha256(raw)
        if _sha256(descriptor["sha256"], f"prepared {partition} target SHA-256") != raw_sha:
            raise PublisherAttestationError(f"prepared {partition} target SHA-256 mismatch")
        _validate_output_receipt(
            outputs[expected_relative],
            expected_records=expected_count,
            observed_raw=raw,
            name=f"prepared {partition} target output receipt",
        )
        instance_order: list[str] = []
        for index, row in enumerate(rows):
            _exact_fields(row, _TARGET_FIELDS, f"prepared {partition} target row {index}")
            if row["schema"] != TARGET_SCHEMA or row["schema_version"] != TARGET_SCHEMA_VERSION:
                raise PublisherAttestationError(
                    "prepared evaluator target has an unsupported schema"
                )
            if row["learning_partition"] != partition:
                raise PublisherAttestationError("prepared evaluator target crosses partitions")
            _verify_record(row, f"prepared {partition} target row {index}")
            instance_id = _text(row["instance_id"], "target instance ID")
            _sha256(row["instance_record_digest"], "target instance record digest")
            _sha256(row["certificate_digest"], "target certificate digest")
            _sha256(row["evaluator_protocol_digest"], "target evaluator protocol digest")
            _finite(row["reference_energy"], "target reference energy")
            if row["reference_status"] not in EVIDENCE_KIND_BY_REFERENCE_STATUS:
                raise PublisherAttestationError("target reference status has no evidence kind")
            if instance_id in all_target_instances:
                raise PublisherAttestationError("prepared targets repeat an instance ID")
            all_target_instances.add(instance_id)
            instance_order.append(instance_id)
        if instance_order != sorted(instance_order):
            raise PublisherAttestationError(
                f"prepared {partition} targets are not sorted by instance ID"
            )
        digest = _target_set_digest(rows, partition)
        if _sha256(
            descriptor["target_set_digest"], f"prepared {partition} target-set digest"
        ) != digest:
            raise PublisherAttestationError(f"prepared {partition} target-set digest mismatch")
        targets_by_partition[partition] = tuple(rows)
        target_raw_sha[partition] = raw_sha
        target_set_digests[partition] = digest
    if all_target_instances != policy_instances:
        raise PublisherAttestationError(
            "prepared evaluator targets do not exactly cover policy instances"
        )

    return _PreparedV4(
        root=root,
        manifest=manifest,
        manifest_sha256=_file_sha256(manifest_raw),
        target_authority_record_digest=authority_digest,
        targets=targets_by_partition,
        target_raw_sha256=target_raw_sha,
        target_set_digests=target_set_digests,
        source_release_id=str(source_release_id),
        source_release_manifest_sha256=str(source_release_sha),
    )


def _write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise PublisherAttestationError(f"cannot publish {path.name}: {error}") from error


def _publication_receipt(
    root: Path,
    attestation: Mapping[str, object],
    attestation_raw: bytes,
    *,
    target_count: int,
    certificate_count: int,
    verifier: VerifierIdentity,
) -> PublisherAttestationReceipt:
    return PublisherAttestationReceipt(
        output_directory=root,
        attestation_path=root / "publisher-attestation.json",
        attestation_record_digest=str(attestation["record_digest"]),
        attestation_sha256=_file_sha256(attestation_raw),
        publisher_id=str(attestation["publisher_id"]),
        publication_id=str(attestation["publication_id"]),
        prepared_manifest_sha256=str(attestation["prepared_manifest_sha256"]),
        target_count=target_count,
        certificate_count=certificate_count,
        verifier=verifier,
    )


def _verify_publication(
    prepared: _PreparedV4,
    publication_root: Path,
    *,
    expected_attestation_record_digest: str,
    expected_publisher_id: str,
    expected_verifier: VerifierIdentity,
) -> PublisherAttestationReceipt:
    expected_digest = _sha256(
        expected_attestation_record_digest, "expected publisher-attestation record digest"
    )
    expected_publisher = _text(expected_publisher_id, "expected publisher ID")
    if not isinstance(expected_verifier, VerifierIdentity):
        raise TypeError("expected_verifier must be a VerifierIdentity")

    attestation_path, _ = _bound_file(
        publication_root,
        "publisher-attestation.json",
        "publisher attestation",
    )
    attestation, attestation_raw = _canonical_object(
        attestation_path, "publisher attestation"
    )
    _exact_fields(
        attestation,
        {
            "evidence_manifests",
            "prepared_manifest_sha256",
            "publication_id",
            "publisher_id",
            "record_digest",
            "schema",
            "schema_version",
            "source_release_id",
            "source_release_manifest_sha256",
            "statement",
            "target_authority_record_digest",
        },
        "publisher attestation",
    )
    if (
        attestation["schema"] != PUBLISHER_ATTESTATION_SCHEMA
        or attestation["schema_version"] != PUBLISHER_ATTESTATION_VERSION
        or attestation["statement"] != PUBLISHER_ATTESTATION_STATEMENT
    ):
        raise PublisherAttestationError("publisher attestation has an unsupported schema")
    observed_digest = _verify_record(attestation, "publisher attestation")
    if observed_digest != expected_digest:
        raise PublisherAttestationError("publisher attestation differs from its pinned digest")
    if _text(attestation["publisher_id"], "publisher ID") != expected_publisher:
        raise PublisherAttestationError("publisher attestation differs from its pinned publisher")
    _text(attestation["publication_id"], "publication ID")
    if (
        _sha256(attestation["prepared_manifest_sha256"], "prepared-manifest SHA-256")
        != prepared.manifest_sha256
    ):
        raise PublisherAttestationError("publisher attestation binds another prepared manifest")
    if (
        _sha256(
            attestation["target_authority_record_digest"],
            "target-authority record digest",
        )
        != prepared.target_authority_record_digest
    ):
        raise PublisherAttestationError("publisher attestation binds another target authority")
    if (
        attestation["source_release_id"] != prepared.source_release_id
        or _sha256(
            attestation["source_release_manifest_sha256"],
            "source release manifest SHA-256",
        )
        != prepared.source_release_manifest_sha256
    ):
        raise PublisherAttestationError("publisher attestation binds another source release")

    descriptors = _exact_fields(
        attestation["evidence_manifests"],
        set(PARTITIONS),
        "partition evidence manifests",
    )
    evidence_files: set[Path] = set()
    certificate_files: set[Path] = set()
    certificate_digests: set[str] = set()
    certificate_path_by_digest: dict[str, Path] = {}
    target_count = 0
    for partition in PARTITIONS:
        descriptor = _exact_fields(
            descriptors[partition],
            {"path", "sha256", "target_count", "target_set_digest"},
            f"{partition} evidence descriptor",
        )
        evidence_path, _ = _bound_file(
            publication_root,
            descriptor["path"],
            f"{partition} quality evidence manifest",
        )
        if evidence_path in evidence_files:
            raise PublisherAttestationError("partition evidence paths must be unique")
        evidence_files.add(evidence_path)
        evidence, evidence_raw = _canonical_object(
            evidence_path, f"{partition} quality evidence manifest"
        )
        if _sha256(descriptor["sha256"], f"{partition} evidence SHA-256") != _file_sha256(
            evidence_raw
        ):
            raise PublisherAttestationError(f"{partition} quality evidence SHA-256 mismatch")
        _exact_fields(
            evidence,
            {
                "evaluator_targets_sha256",
                "evidence",
                "partition",
                "record_digest",
                "schema",
                "schema_version",
                "target_count",
                "target_set_digest",
            },
            f"{partition} quality evidence manifest",
        )
        if (
            evidence["schema"] != QUALITY_EVIDENCE_SCHEMA
            or evidence["schema_version"] != QUALITY_EVIDENCE_VERSION
            or evidence["partition"] != partition
        ):
            raise PublisherAttestationError("quality evidence has an unsupported schema")
        _verify_record(evidence, f"{partition} quality evidence manifest")
        if (
            _sha256(
                evidence["evaluator_targets_sha256"],
                f"{partition} evaluator-target SHA-256",
            )
            != prepared.target_raw_sha256[partition]
        ):
            raise PublisherAttestationError("quality evidence binds another target file")
        targets = prepared.targets[partition]
        count = _integer(evidence["target_count"], f"{partition} target count", positive=True)
        descriptor_count = _integer(
            descriptor["target_count"], f"{partition} descriptor target count", positive=True
        )
        if count != len(targets) or descriptor_count != count:
            raise PublisherAttestationError("quality evidence target count mismatch")
        target_set_digest = _sha256(
            evidence["target_set_digest"], f"{partition} target-set digest"
        )
        descriptor_set_digest = _sha256(
            descriptor["target_set_digest"],
            f"{partition} descriptor target-set digest",
        )
        if (
            target_set_digest != prepared.target_set_digests[partition]
            or descriptor_set_digest != target_set_digest
        ):
            raise PublisherAttestationError("quality evidence target-set digest mismatch")

        raw_rows = evidence["evidence"]
        if type(raw_rows) is not list:
            raise PublisherAttestationError("quality evidence rows must be a list")
        targets_by_instance = {str(row["instance_id"]): row for row in targets}
        observed_instances: set[str] = set()
        instance_order: list[str] = []
        for index, raw_row in enumerate(raw_rows):
            row = _exact_fields(
                raw_row,
                {
                    "artifact_path",
                    "artifact_sha256",
                    "certificate_digest",
                    "evidence_kind",
                    "instance_id",
                    "target_record_digest",
                    "verifier",
                },
                f"{partition} evidence row {index}",
            )
            instance_id = _text(row["instance_id"], "evidence instance ID")
            if instance_id in observed_instances:
                raise PublisherAttestationError("quality evidence repeats an instance ID")
            observed_instances.add(instance_id)
            instance_order.append(instance_id)
            target = targets_by_instance.get(instance_id)
            if target is None:
                raise PublisherAttestationError("quality evidence references an unknown target")
            if _sha256(row["target_record_digest"], "evidence target record digest") != target[
                "record_digest"
            ]:
                raise PublisherAttestationError("quality evidence references the wrong target")
            certificate_digest = _sha256(
                row["certificate_digest"], "evidence certificate digest"
            )
            if certificate_digest != target["certificate_digest"]:
                raise PublisherAttestationError("quality evidence references the wrong certificate")
            if _sha256(row["artifact_sha256"], "evidence artifact SHA-256") != certificate_digest:
                raise PublisherAttestationError("certificate digest does not identify its artifact")
            if row["evidence_kind"] != EVIDENCE_KIND_BY_REFERENCE_STATUS[
                str(target["reference_status"])
            ]:
                raise PublisherAttestationError("evidence kind differs from reference status")
            verifier = _exact_fields(
                row["verifier"],
                {"implementation_sha256", "name", "version"},
                "quality evidence verifier identity",
            )
            if verifier != expected_verifier.as_dict():
                raise PublisherAttestationError("quality evidence verifier identity differs")
            artifact, _ = _bound_file(
                evidence_path.parent,
                row["artifact_path"],
                "certificate artifact",
            )
            artifact_raw = _read_regular(artifact, "certificate artifact")
            if _file_sha256(artifact_raw) != certificate_digest:
                raise PublisherAttestationError("certificate artifact SHA-256 mismatch")
            previous_artifact = certificate_path_by_digest.setdefault(
                certificate_digest, artifact
            )
            if previous_artifact != artifact:
                raise PublisherAttestationError(
                    "one certificate digest is published at multiple artifact paths"
                )
            certificate_files.add(artifact)
            certificate_digests.add(certificate_digest)
        if instance_order != sorted(instance_order):
            raise PublisherAttestationError("quality evidence rows are not sorted by instance ID")
        if observed_instances != set(targets_by_instance):
            raise PublisherAttestationError("quality evidence does not exactly cover targets")
        target_count += count

    expected_files = {
        attestation_path.resolve(),
        *evidence_files,
        *certificate_files,
    }
    observed_files: set[Path] = set()
    for path in publication_root.rglob("*"):
        if path.is_symlink():
            raise PublisherAttestationError("publication contains a symbolic link")
        if path.is_file():
            observed_files.add(path.resolve())
    if observed_files != expected_files:
        raise PublisherAttestationError("published certificate artifact set is not exact")

    return _publication_receipt(
        publication_root,
        attestation,
        attestation_raw,
        target_count=target_count,
        certificate_count=len(certificate_digests),
        verifier=expected_verifier,
    )


def build_publisher_attestation_v2(
    prepared_directory: str | os.PathLike[str],
    certificate_artifacts: Mapping[str, str | os.PathLike[str]],
    output_directory: str | os.PathLike[str],
    *,
    publisher_id: str,
    publication_id: str,
    verifier: VerifierIdentity,
) -> PublisherAttestationReceipt:
    """Build, self-verify, and atomically publish a partitioned v2 attestation.

    ``certificate_artifacts`` is an exact digest-to-source-path registry.  The distinct
    key set must equal the certificate digests referenced by all prepared targets.
    Certificate bytes are copied into the publication, so the result is immutable and
    self-contained rather than dependent on mutable source paths.
    """

    prepared = _validate_prepared_v4(prepared_directory)
    publisher = _text(publisher_id, "publisher ID")
    publication = _text(publication_id, "publication ID")
    if not isinstance(verifier, VerifierIdentity):
        raise TypeError("verifier must be a VerifierIdentity")
    if not isinstance(certificate_artifacts, Mapping):
        raise TypeError("certificate_artifacts must be a digest-to-path mapping")

    expected_digests = {
        str(target["certificate_digest"])
        for partition in PARTITIONS
        for target in prepared.targets[partition]
    }
    supplied_digests = {
        _sha256(digest, "certificate registry digest") for digest in certificate_artifacts
    }
    if supplied_digests != expected_digests:
        missing = sorted(expected_digests - supplied_digests)
        extra = sorted(supplied_digests - expected_digests)
        raise PublisherAttestationError(
            f"certificate coverage differs: missing={missing}, extra={extra}"
        )
    certificate_bytes: dict[str, bytes] = {}
    for digest in sorted(expected_digests):
        source = Path(certificate_artifacts[digest])
        raw = _read_regular(source, f"certificate {digest}")
        if _file_sha256(raw) != digest:
            raise PublisherAttestationError(
                f"certificate SHA-256 mismatch for {digest}"
            )
        certificate_bytes[digest] = raw

    destination = Path(output_directory)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"publisher-attestation output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.publish-", dir=destination.parent)
    ).resolve()
    try:
        for digest in sorted(certificate_bytes):
            _write_new(
                temporary / "quality-evidence" / "certificates" / digest,
                certificate_bytes[digest],
            )

        evidence_descriptors: dict[str, dict[str, object]] = {}
        for partition in PARTITIONS:
            evidence_rows = []
            for target in prepared.targets[partition]:
                certificate_digest = str(target["certificate_digest"])
                evidence_rows.append(
                    {
                        "artifact_path": f"certificates/{certificate_digest}",
                        "artifact_sha256": certificate_digest,
                        "certificate_digest": certificate_digest,
                        "evidence_kind": EVIDENCE_KIND_BY_REFERENCE_STATUS[
                            str(target["reference_status"])
                        ],
                        "instance_id": target["instance_id"],
                        "target_record_digest": target["record_digest"],
                        "verifier": verifier.as_dict(),
                    }
                )
            evidence_rows.sort(key=lambda row: str(row["instance_id"]))
            evidence_payload: dict[str, object] = {
                "evaluator_targets_sha256": prepared.target_raw_sha256[partition],
                "evidence": evidence_rows,
                "partition": partition,
                "schema": QUALITY_EVIDENCE_SCHEMA,
                "schema_version": QUALITY_EVIDENCE_VERSION,
                "target_count": len(evidence_rows),
                "target_set_digest": prepared.target_set_digests[partition],
            }
            evidence = {**evidence_payload, "record_digest": content_digest(evidence_payload)}
            evidence_raw = canonical_json_bytes(evidence) + b"\n"
            relative = f"quality-evidence/{partition}.json"
            _write_new(temporary / relative, evidence_raw)
            evidence_descriptors[partition] = {
                "path": relative,
                "sha256": _file_sha256(evidence_raw),
                "target_count": len(evidence_rows),
                "target_set_digest": prepared.target_set_digests[partition],
            }

        attestation_payload: dict[str, object] = {
            "evidence_manifests": evidence_descriptors,
            "prepared_manifest_sha256": prepared.manifest_sha256,
            "publication_id": publication,
            "publisher_id": publisher,
            "schema": PUBLISHER_ATTESTATION_SCHEMA,
            "schema_version": PUBLISHER_ATTESTATION_VERSION,
            "source_release_id": prepared.source_release_id,
            "source_release_manifest_sha256": prepared.source_release_manifest_sha256,
            "statement": PUBLISHER_ATTESTATION_STATEMENT,
            "target_authority_record_digest": prepared.target_authority_record_digest,
        }
        attestation = {
            **attestation_payload,
            "record_digest": content_digest(attestation_payload),
        }
        attestation_raw = canonical_json_bytes(attestation) + b"\n"
        # The public authority root is deliberately the final file written.
        _write_new(temporary / "publisher-attestation.json", attestation_raw)

        revalidated_prepared = _validate_prepared_v4(prepared.root)
        if revalidated_prepared != prepared:
            raise PublisherAttestationError(
                "prepared-v4 corpus changed while the attestation was being built"
            )
        _verify_publication(
            revalidated_prepared,
            temporary,
            expected_attestation_record_digest=str(attestation["record_digest"]),
            expected_publisher_id=publisher,
            expected_verifier=verifier,
        )
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"publisher-attestation output already exists: {destination}"
            )
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    final_root = destination.resolve(strict=True)
    return _publication_receipt(
        final_root,
        attestation,
        attestation_raw,
        target_count=sum(len(prepared.targets[partition]) for partition in PARTITIONS),
        certificate_count=len(certificate_bytes),
        verifier=verifier,
    )


def verify_publisher_attestation_v2(
    prepared_directory: str | os.PathLike[str],
    publication_directory: str | os.PathLike[str],
    *,
    expected_attestation_record_digest: str,
    expected_publisher_id: str,
    expected_verifier: VerifierIdentity,
) -> PublisherAttestationReceipt:
    """Independently verify a published v2 bundle against caller-owned pins."""

    prepared = _validate_prepared_v4(prepared_directory)
    publication = _directory(publication_directory, "publisher-attestation directory")
    return _verify_publication(
        prepared,
        publication,
        expected_attestation_record_digest=expected_attestation_record_digest,
        expected_publisher_id=expected_publisher_id,
        expected_verifier=expected_verifier,
    )


__all__ = [
    "EVIDENCE_KIND_BY_REFERENCE_STATUS",
    "PUBLISHER_ATTESTATION_SCHEMA",
    "PUBLISHER_ATTESTATION_STATEMENT",
    "PUBLISHER_ATTESTATION_VERSION",
    "PublisherAttestationError",
    "PublisherAttestationReceipt",
    "QUALITY_EVIDENCE_SCHEMA",
    "QUALITY_EVIDENCE_VERSION",
    "VerifierIdentity",
    "build_publisher_attestation_v2",
    "verify_publisher_attestation_v2",
]
