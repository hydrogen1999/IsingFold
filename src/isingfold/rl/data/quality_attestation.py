"""Pinned publisher attestation for trusted ground-energy targets.

Record digests and file checksums provide integrity, not authority.  This module adds the
missing authority boundary: callers must obtain an attestation digest and publisher ID from
an out-of-band trusted release channel.  The public attestation binds the prepared corpus
and source provenance.  Its separately hashed evidence manifest is opened only by trusted
training/evaluation code and binds every target to a real certificate artifact and verifier.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from isingfold.rl.data.import_embedbench import content_digest

PUBLISHER_ATTESTATION_SCHEMA = "isingfold.quality-publisher-attestation"
PUBLISHER_ATTESTATION_VERSION = 1
PUBLISHER_ATTESTATION_VERSION_V2 = 2
PUBLISHER_ATTESTATION_STATEMENT = "publisher-attests-ground-energy-evidence"
QUALITY_EVIDENCE_SCHEMA = "isingfold.quality-ground-evidence"
QUALITY_EVIDENCE_VERSION = 1
QUALITY_EVIDENCE_VERSION_V2 = 2
EVIDENCE_KIND_BY_REFERENCE_STATUS: dict[str, str] = {
    "planted_proof": "planted-witness",
    "exact_enumeration": "enumeration-transcript",
    "certified_optimal": "optimality-certificate",
}
_HEX = frozenset("0123456789abcdef")


class QualityAttestationError(ValueError):
    """A quality authority, evidence manifest, or certificate binding is invalid."""


@dataclass(frozen=True)
class QualityAttestationPin:
    """Trust anchor supplied independently of the corpus being authenticated.

    ``expected_digest`` must come from a trusted release registry, paper artifact, or
    operator configuration.  Reading it from the attestation itself defeats the boundary.
    """

    path: str | Path
    expected_digest: str
    expected_publisher_id: str


@dataclass(frozen=True)
class EvidenceManifestIdentity:
    """One partition-scoped evidence-manifest commitment in the public authority root."""

    partition: str
    path: str
    sha256: str
    target_set_digest: str
    target_count: int


@dataclass(frozen=True)
class PublisherAttestation:
    """Validated public publisher statement; contains no ground-energy scalar."""

    path: Path
    digest: str
    publisher_id: str
    publication_id: str
    prepared_manifest_sha256: str
    source_release_id: str
    source_release_manifest_sha256: str
    evidence_manifest_path: str | None
    evidence_manifest_sha256: str | None
    schema_version: int
    evidence_manifests: Mapping[str, EvidenceManifestIdentity]
    target_authority_record_digest: str | None


@dataclass(frozen=True)
class QualityEvidenceReceipt:
    """Identity of the fully validated target-to-certificate evidence chain."""

    publisher_attestation_digest: str
    evidence_manifest_digest: str
    evidence_manifest_sha256: str
    target_set_digest: str
    target_count: int
    partition: str | None
    opened_files: tuple[tuple[str, str], ...]


def _strict_object(raw: bytes, name: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise QualityAttestationError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise QualityAttestationError(f"{name} contains non-finite number {token}")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QualityAttestationError(f"{name} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise QualityAttestationError(f"{name} must be a JSON object")
    _check_finite(value, name)
    return value


def _check_finite(value: object, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise QualityAttestationError(f"{name} contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _check_finite(item, name)
    elif isinstance(value, list):
        for item in value:
            _check_finite(item, name)


def _read(path: Path, name: str) -> bytes:
    try:
        if not path.is_file():
            raise QualityAttestationError(f"{name} is missing or not a regular file")
        return path.read_bytes()
    except OSError as exc:
        raise QualityAttestationError(f"cannot read {name}: {exc}") from exc


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise QualityAttestationError(
            f"{name} schema differs: missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise QualityAttestationError(f"{name} must be nonempty text")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise QualityAttestationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QualityAttestationError(f"{name} must be a positive integer")
    return value


def _schema_version(value: object, expected: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise QualityAttestationError(f"unsupported {name} schema version")


def _verify_record(record: Mapping[str, object], name: str) -> str:
    digest = _sha256(record.get("record_digest"), f"{name} record digest")
    payload = {key: value for key, value in record.items() if key != "record_digest"}
    try:
        observed_digest = content_digest(payload)
    except (TypeError, ValueError) as exc:
        raise QualityAttestationError(f"{name} contains non-canonical data") from exc
    if not hmac.compare_digest(digest, observed_digest):
        raise QualityAttestationError(f"{name} record digest mismatch")
    return digest


def _safe_relative_path(value: object, name: str) -> str:
    text = _text(value, name)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or "\\" in text
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise QualityAttestationError(f"{name} must be a safe relative path")
    return text


def _bound_path(root: Path, relative_path: str, name: str) -> Path:
    root = root.resolve()
    candidate = root.joinpath(*PurePosixPath(relative_path).parts)
    try:
        resolved = candidate.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise QualityAttestationError(f"{name} is missing") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise QualityAttestationError(f"{name} must remain inside the attestation directory")
    return resolved


def _file_sha256(path: Path, name: str) -> str:
    return hashlib.sha256(_read(path, name)).hexdigest()


def load_publisher_attestation(
    pin: QualityAttestationPin,
    *,
    prepared_manifest_path: str | Path,
) -> PublisherAttestation:
    """Authenticate the public publisher statement against an out-of-band pin."""

    if not isinstance(pin, QualityAttestationPin):
        raise TypeError("quality attestation pin must be QualityAttestationPin")
    expected_digest = _sha256(pin.expected_digest, "pinned publisher-attestation digest")
    expected_publisher = _text(pin.expected_publisher_id, "pinned publisher ID")
    path = Path(pin.path)
    raw = _read(path, "publisher attestation")
    record = _strict_object(raw, "publisher attestation")
    common_fields = {
        "prepared_manifest_sha256",
        "publication_id",
        "publisher_id",
        "record_digest",
        "schema",
        "schema_version",
        "source_release_id",
        "source_release_manifest_sha256",
        "statement",
    }
    if record["schema"] != PUBLISHER_ATTESTATION_SCHEMA:
        raise QualityAttestationError("unsupported publisher-attestation schema")
    schema_version = record["schema_version"]
    if schema_version == PUBLISHER_ATTESTATION_VERSION:
        _exact_keys(record, common_fields | {"evidence_manifest"}, "publisher attestation")
    elif schema_version == PUBLISHER_ATTESTATION_VERSION_V2:
        _exact_keys(
            record,
            common_fields | {"evidence_manifests", "target_authority_record_digest"},
            "publisher attestation",
        )
    else:
        raise QualityAttestationError("unsupported publisher-attestation schema version")
    if record["statement"] != PUBLISHER_ATTESTATION_STATEMENT:
        raise QualityAttestationError("publisher attestation has an unsupported statement")
    digest = _verify_record(record, "publisher attestation")
    if not hmac.compare_digest(digest, expected_digest):
        raise QualityAttestationError("publisher attestation differs from the pinned digest")
    publisher_id = _text(record["publisher_id"], "publisher ID")
    if publisher_id != expected_publisher:
        raise QualityAttestationError("publisher attestation differs from the pinned publisher")

    prepared_sha256 = _sha256(
        record["prepared_manifest_sha256"], "attested prepared-manifest digest"
    )
    observed_prepared = _file_sha256(Path(prepared_manifest_path), "prepared manifest")
    if not hmac.compare_digest(prepared_sha256, observed_prepared):
        raise QualityAttestationError("prepared manifest differs from publisher attestation")

    evidence_path: str | None = None
    evidence_sha256: str | None = None
    evidence_manifests: dict[str, EvidenceManifestIdentity] = {}
    target_authority_digest: str | None = None
    if schema_version == PUBLISHER_ATTESTATION_VERSION:
        evidence = record["evidence_manifest"]
        if not isinstance(evidence, Mapping):
            raise QualityAttestationError("evidence-manifest identity must be an object")
        _exact_keys(evidence, {"path", "sha256"}, "evidence-manifest identity")
        evidence_path = _safe_relative_path(evidence["path"], "evidence-manifest path")
        evidence_sha256 = _sha256(evidence["sha256"], "evidence-manifest SHA-256")
    else:
        prepared_record = _strict_object(
            _read(Path(prepared_manifest_path), "prepared manifest"), "prepared manifest"
        )
        target_authority = prepared_record.get("target_authority")
        if not isinstance(target_authority, Mapping):
            raise QualityAttestationError(
                "partitioned publisher attestation requires prepared target authority"
            )
        target_authority_digest = _sha256(
            record["target_authority_record_digest"], "target-authority record digest"
        )
        if target_authority.get("record_digest") != target_authority_digest:
            raise QualityAttestationError(
                "publisher attestation binds another prepared target authority"
            )
        raw_manifests = record["evidence_manifests"]
        if not isinstance(raw_manifests, Mapping) or set(raw_manifests) != {
            "train",
            "val",
            "test",
        }:
            raise QualityAttestationError(
                "partitioned evidence identities must cover train, val, and test"
            )
        for partition in ("train", "val", "test"):
            descriptor = raw_manifests[partition]
            if not isinstance(descriptor, Mapping):
                raise QualityAttestationError("partition evidence identity must be an object")
            _exact_keys(
                descriptor,
                {"path", "sha256", "target_count", "target_set_digest"},
                "partition evidence identity",
            )
            evidence_manifests[partition] = EvidenceManifestIdentity(
                partition=partition,
                path=_safe_relative_path(
                    descriptor["path"], f"{partition} evidence-manifest path"
                ),
                sha256=_sha256(
                    descriptor["sha256"], f"{partition} evidence-manifest SHA-256"
                ),
                target_set_digest=_sha256(
                    descriptor["target_set_digest"],
                    f"{partition} target-set digest",
                ),
                target_count=_positive_int(
                    descriptor["target_count"], f"{partition} target count"
                ),
            )
    return PublisherAttestation(
        path=path.resolve(),
        digest=digest,
        publisher_id=publisher_id,
        publication_id=_text(record["publication_id"], "publication ID"),
        prepared_manifest_sha256=prepared_sha256,
        source_release_id=_text(record["source_release_id"], "source release ID"),
        source_release_manifest_sha256=_sha256(
            record["source_release_manifest_sha256"],
            "source release manifest digest",
        ),
        evidence_manifest_path=evidence_path,
        evidence_manifest_sha256=evidence_sha256,
        schema_version=schema_version,
        evidence_manifests=evidence_manifests,
        target_authority_record_digest=target_authority_digest,
    )


def quality_target_set_digest(
    targets: Sequence[Mapping[str, object]],
    *,
    partition: str | None = None,
) -> str:
    """Canonical semantic identity of an exactly covered evaluator-target set."""

    identities: list[dict[str, str]] = []
    seen: set[str] = set()
    for target in targets:
        target_digest = _verify_record(target, "evaluator target")
        instance_id = _text(target.get("instance_id"), "target instance ID")
        if instance_id in seen:
            raise QualityAttestationError("target set repeats an instance ID")
        seen.add(instance_id)
        identities.append(
            {
                "instance_id": instance_id,
                "target_record_digest": target_digest,
            }
        )
    if partition is None:
        payload: dict[str, object] = {
            "domain": "isingfold-quality-target-set-v1",
            "targets": sorted(identities, key=lambda item: item["instance_id"]),
        }
    else:
        if partition not in {"train", "val", "test"}:
            raise QualityAttestationError("quality target partition is invalid")
        payload = {
            "domain": "isingfold-partition-target-set-v1",
            "partition": partition,
            "targets": sorted(identities, key=lambda item: item["instance_id"]),
        }
    return content_digest(payload)


def validate_attested_provenance(
    attestation: PublisherAttestation,
    provenance_rows: Sequence[Mapping[str, object]],
) -> None:
    """Bind self-digested task provenance to the publisher's pinned source release."""

    identities: set[tuple[str, str]] = set()
    for row in provenance_rows:
        identities.add(
            (
                _text(row.get("source_release_id"), "provenance source release ID"),
                _sha256(
                    row.get("source_release_manifest_sha256"),
                    "provenance source release manifest digest",
                ),
            )
        )
    if identities and identities != {
        (attestation.source_release_id, attestation.source_release_manifest_sha256)
    }:
        raise QualityAttestationError(
            "prepared provenance differs from the attested source release"
        )


def validate_quality_evidence(
    attestation: PublisherAttestation,
    *,
    evaluator_targets_path: str | Path,
    targets: Sequence[Mapping[str, object]],
    partition: str | None = None,
) -> QualityEvidenceReceipt:
    """Validate exact target coverage and every target-to-certificate artifact chain."""

    if attestation.schema_version == PUBLISHER_ATTESTATION_VERSION_V2:
        if partition not in {"train", "val", "test"}:
            raise QualityAttestationError(
                "partitioned quality evidence requires an explicit partition"
            )
        evidence_identity = attestation.evidence_manifests[partition]
        evidence_relative = evidence_identity.path
        expected_evidence_sha256 = evidence_identity.sha256
    else:
        if partition is not None:
            raise QualityAttestationError(
                "legacy quality evidence does not carry a partition identity"
            )
        if (
            attestation.evidence_manifest_path is None
            or attestation.evidence_manifest_sha256 is None
        ):
            raise RuntimeError("legacy publisher attestation lost its evidence identity")
        evidence_identity = None
        evidence_relative = attestation.evidence_manifest_path
        expected_evidence_sha256 = attestation.evidence_manifest_sha256

    target_by_instance: dict[str, Mapping[str, object]] = {}
    for target in targets:
        _verify_record(target, "evaluator target")
        instance_id = _text(target.get("instance_id"), "target instance ID")
        if instance_id in target_by_instance:
            raise QualityAttestationError("evaluator targets repeat an instance ID")
        _sha256(target.get("record_digest"), "target record digest")
        _sha256(target.get("certificate_digest"), "target certificate digest")
        status = _text(target.get("reference_status"), "target reference status")
        if status not in EVIDENCE_KIND_BY_REFERENCE_STATUS:
            raise QualityAttestationError("target reference status has no evidence contract")
        if partition is not None and target.get("learning_partition") != partition:
            raise QualityAttestationError("evaluator target crosses its evidence partition")
        target_by_instance[instance_id] = target
    if not target_by_instance:
        raise QualityAttestationError("quality evidence cannot attest an empty target set")

    evidence_path = _bound_path(
        attestation.path.parent,
        evidence_relative,
        "quality evidence manifest",
    )
    evidence_raw = _read(evidence_path, "quality evidence manifest")
    observed_evidence_sha256 = hashlib.sha256(evidence_raw).hexdigest()
    if not hmac.compare_digest(
        observed_evidence_sha256, expected_evidence_sha256
    ):
        raise QualityAttestationError("quality evidence manifest differs from attestation")
    manifest = _strict_object(evidence_raw, "quality evidence manifest")
    expected_fields = {
        "evaluator_targets_sha256",
        "evidence",
        "record_digest",
        "schema",
        "schema_version",
        "target_count",
        "target_set_digest",
    }
    expected_evidence_version = QUALITY_EVIDENCE_VERSION
    if partition is not None:
        expected_fields.add("partition")
        expected_evidence_version = QUALITY_EVIDENCE_VERSION_V2
    _exact_keys(manifest, expected_fields, "quality evidence manifest")
    if manifest["schema"] != QUALITY_EVIDENCE_SCHEMA:
        raise QualityAttestationError("unsupported quality-evidence schema")
    _schema_version(manifest["schema_version"], expected_evidence_version, "quality-evidence")
    if partition is not None and manifest["partition"] != partition:
        raise QualityAttestationError("quality evidence manifest crosses partitions")
    evidence_manifest_digest = _verify_record(manifest, "quality evidence manifest")

    targets_sha256 = _sha256(
        manifest["evaluator_targets_sha256"], "attested evaluator-target file digest"
    )
    observed_targets_sha256 = _file_sha256(
        Path(evaluator_targets_path), "evaluator-target file"
    )
    if not hmac.compare_digest(targets_sha256, observed_targets_sha256):
        raise QualityAttestationError("evaluator-target file differs from quality evidence")
    target_count = _positive_int(manifest["target_count"], "quality target count")
    if target_count != len(target_by_instance):
        raise QualityAttestationError("quality evidence target count mismatch")
    expected_set_digest = quality_target_set_digest(
        list(target_by_instance.values()), partition=partition
    )
    recorded_set_digest = _sha256(manifest["target_set_digest"], "quality target-set digest")
    if not hmac.compare_digest(recorded_set_digest, expected_set_digest):
        raise QualityAttestationError("quality target-set digest mismatch")
    if evidence_identity is not None and (
        target_count != evidence_identity.target_count
        or expected_set_digest != evidence_identity.target_set_digest
    ):
        raise QualityAttestationError(
            "quality evidence differs from its partition authority commitment"
        )

    evidence_rows = manifest["evidence"]
    if not isinstance(evidence_rows, list):
        raise QualityAttestationError("quality evidence rows must be a list")
    evidence_instances: set[str] = set()
    row_fields = {
        "artifact_path",
        "artifact_sha256",
        "certificate_digest",
        "evidence_kind",
        "instance_id",
        "target_record_digest",
        "verifier",
    }
    opened_files: list[tuple[str, str]] = [
        (evidence_relative, observed_evidence_sha256)
    ]
    for row in evidence_rows:
        if not isinstance(row, Mapping):
            raise QualityAttestationError("quality evidence row must be an object")
        _exact_keys(row, row_fields, "quality evidence row")
        instance_id = _text(row["instance_id"], "evidence instance ID")
        if instance_id in evidence_instances:
            raise QualityAttestationError("quality evidence repeats an instance ID")
        evidence_instances.add(instance_id)
        target = target_by_instance.get(instance_id)
        if target is None:
            raise QualityAttestationError("quality evidence references an unknown target")
        target_digest = _sha256(row["target_record_digest"], "evidence target digest")
        if not hmac.compare_digest(target_digest, str(target["record_digest"])):
            raise QualityAttestationError("quality evidence references the wrong target record")
        certificate_digest = _sha256(row["certificate_digest"], "evidence certificate digest")
        if not hmac.compare_digest(certificate_digest, str(target["certificate_digest"])):
            raise QualityAttestationError("quality evidence has the wrong certificate digest")
        artifact_sha256 = _sha256(row["artifact_sha256"], "evidence artifact digest")
        if not hmac.compare_digest(artifact_sha256, certificate_digest):
            raise QualityAttestationError(
                "certificate digest does not identify the evidence artifact"
            )
        expected_kind = EVIDENCE_KIND_BY_REFERENCE_STATUS[str(target["reference_status"])]
        if row["evidence_kind"] != expected_kind:
            raise QualityAttestationError("evidence kind differs from target reference status")

        verifier = row["verifier"]
        if not isinstance(verifier, Mapping):
            raise QualityAttestationError("quality evidence verifier must be an object")
        _exact_keys(
            verifier,
            {"implementation_sha256", "name", "version"},
            "quality evidence verifier",
        )
        _text(verifier["name"], "quality evidence verifier name")
        _text(verifier["version"], "quality evidence verifier version")
        _sha256(
            verifier["implementation_sha256"],
            "quality evidence verifier implementation digest",
        )

        artifact_path = _safe_relative_path(row["artifact_path"], "evidence artifact path")
        artifact = _bound_path(evidence_path.parent, artifact_path, "evidence artifact")
        observed_artifact_sha256 = _file_sha256(artifact, "evidence artifact")
        if not hmac.compare_digest(observed_artifact_sha256, artifact_sha256):
            raise QualityAttestationError("evidence artifact checksum mismatch")
        opened_files.append((artifact_path, observed_artifact_sha256))
    if evidence_instances != set(target_by_instance):
        raise QualityAttestationError("quality evidence does not exactly cover evaluator targets")

    return QualityEvidenceReceipt(
        publisher_attestation_digest=attestation.digest,
        evidence_manifest_digest=evidence_manifest_digest,
        evidence_manifest_sha256=observed_evidence_sha256,
        target_set_digest=expected_set_digest,
        target_count=target_count,
        partition=partition,
        opened_files=tuple(opened_files),
    )


__all__ = [
    "EvidenceManifestIdentity",
    "EVIDENCE_KIND_BY_REFERENCE_STATUS",
    "PUBLISHER_ATTESTATION_SCHEMA",
    "PUBLISHER_ATTESTATION_STATEMENT",
    "PUBLISHER_ATTESTATION_VERSION_V2",
    "QUALITY_EVIDENCE_SCHEMA",
    "QUALITY_EVIDENCE_VERSION_V2",
    "PublisherAttestation",
    "QualityAttestationError",
    "QualityAttestationPin",
    "QualityEvidenceReceipt",
    "load_publisher_attestation",
    "quality_target_set_digest",
    "validate_attested_provenance",
    "validate_quality_evidence",
]
