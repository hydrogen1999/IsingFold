from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.quality_attestation import (
    EVIDENCE_KIND_BY_REFERENCE_STATUS,
    PUBLISHER_ATTESTATION_SCHEMA,
    PUBLISHER_ATTESTATION_STATEMENT,
    QUALITY_EVIDENCE_SCHEMA,
    QualityAttestationError,
    QualityAttestationPin,
    load_publisher_attestation,
    quality_target_set_digest,
    validate_attested_provenance,
    validate_quality_evidence,
)


def _with_digest(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "record_digest": content_digest(payload)}


def _target(certificate_digest: str) -> dict[str, object]:
    return _with_digest(
        {
            "certificate_digest": certificate_digest,
            "evaluator_protocol_digest": "b" * 64,
            "instance_id": "instance-fixture",
            "instance_record_digest": "c" * 64,
            "reference_energy": -3.5,
            "reference_status": "exact_enumeration",
            "schema": "isingfold.evaluator-target",
            "schema_version": 1,
        }
    )


def _write_bundle(
    root: Path,
    *,
    target_certificate: str | None = None,
    artifact_bytes: bytes = b'{"enumerated_states":16,"minimum":-3.5}\n',
    artifact_path: str = "evidence/exact.json",
) -> tuple[QualityAttestationPin, Path, Path, dict[str, object]]:
    root.mkdir()
    prepared_manifest = root / "prepared-manifest.json"
    prepared_manifest.write_bytes(b'{"fixture":"prepared"}\n')
    artifact = root / artifact_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(artifact_bytes)
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    target = _target(artifact_sha256 if target_certificate is None else target_certificate)
    evaluator_targets = root / "evaluator-targets.jsonl"
    evaluator_targets.write_bytes(canonical_json_bytes(target) + b"\n")

    evidence_payload = {
        "evaluator_targets_sha256": hashlib.sha256(evaluator_targets.read_bytes()).hexdigest(),
        "evidence": [
            {
                "artifact_path": artifact_path,
                "artifact_sha256": artifact_sha256,
                "certificate_digest": target["certificate_digest"],
                "evidence_kind": EVIDENCE_KIND_BY_REFERENCE_STATUS[
                    target["reference_status"]
                ],
                "instance_id": target["instance_id"],
                "target_record_digest": target["record_digest"],
                "verifier": {
                    "implementation_sha256": "d" * 64,
                    "name": "fixture-exact-verifier",
                    "version": "1.0",
                },
            }
        ],
        "schema": QUALITY_EVIDENCE_SCHEMA,
        "schema_version": 1,
        "target_count": 1,
        "target_set_digest": quality_target_set_digest([target]),
    }
    evidence = _with_digest(evidence_payload)
    evidence_path = root / "quality-evidence.json"
    evidence_path.write_bytes(canonical_json_bytes(evidence) + b"\n")

    attestation_payload = {
        "evidence_manifest": {
            "path": evidence_path.name,
            "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        },
        "prepared_manifest_sha256": hashlib.sha256(prepared_manifest.read_bytes()).hexdigest(),
        "publication_id": "fixture-publication-v1",
        "publisher_id": "isingfold-release-authority",
        "schema": PUBLISHER_ATTESTATION_SCHEMA,
        "schema_version": 1,
        "source_release_id": "embedbench-release-v2",
        "source_release_manifest_sha256": "e" * 64,
        "statement": PUBLISHER_ATTESTATION_STATEMENT,
    }
    attestation = _with_digest(attestation_payload)
    attestation_path = root / "publisher-attestation.json"
    attestation_path.write_bytes(canonical_json_bytes(attestation) + b"\n")
    pin = QualityAttestationPin(
        path=attestation_path,
        expected_digest=attestation["record_digest"],
        expected_publisher_id="isingfold-release-authority",
    )
    return pin, prepared_manifest, evaluator_targets, target


def _repin_changed_evidence(
    pin: QualityAttestationPin,
    mutate,
) -> QualityAttestationPin:
    attestation_path = Path(pin.path)
    evidence_path = attestation_path.parent / "quality-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    mutate(evidence)
    evidence_payload = {
        key: value for key, value in evidence.items() if key != "record_digest"
    }
    evidence["record_digest"] = content_digest(evidence_payload)
    evidence_path.write_bytes(canonical_json_bytes(evidence) + b"\n")
    attestation = json.loads(attestation_path.read_text())
    attestation["evidence_manifest"]["sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    attestation_payload = {
        key: value for key, value in attestation.items() if key != "record_digest"
    }
    attestation["record_digest"] = content_digest(attestation_payload)
    attestation_path.write_bytes(canonical_json_bytes(attestation) + b"\n")
    return QualityAttestationPin(
        attestation_path,
        attestation["record_digest"],
        pin.expected_publisher_id,
    )


def test_publisher_attestation_requires_an_out_of_band_digest_pin(tmp_path: Path) -> None:
    pin, prepared_manifest, _, _ = _write_bundle(tmp_path / "trust")
    raw = json.loads(Path(pin.path).read_text())
    raw["publication_id"] = "forged-but-self-digested"
    payload = {key: value for key, value in raw.items() if key != "record_digest"}
    raw["record_digest"] = content_digest(payload)
    Path(pin.path).write_bytes(canonical_json_bytes(raw) + b"\n")

    with pytest.raises(QualityAttestationError, match="pinned digest"):
        load_publisher_attestation(pin, prepared_manifest_path=prepared_manifest)


def test_publisher_identity_is_part_of_the_out_of_band_pin(tmp_path: Path) -> None:
    pin, prepared_manifest, _, _ = _write_bundle(tmp_path / "trust")
    wrong_publisher = QualityAttestationPin(
        pin.path,
        pin.expected_digest,
        "untrusted-publisher",
    )

    with pytest.raises(QualityAttestationError, match="pinned publisher"):
        load_publisher_attestation(
            wrong_publisher,
            prepared_manifest_path=prepared_manifest,
        )


def test_quality_target_requires_hash_verified_evidence_not_an_arbitrary_hex(
    tmp_path: Path,
) -> None:
    pin, prepared_manifest, targets_path, target = _write_bundle(
        tmp_path / "trust",
        target_certificate="a" * 64,
    )
    attestation = load_publisher_attestation(pin, prepared_manifest_path=prepared_manifest)

    with pytest.raises(QualityAttestationError, match="certificate digest"):
        validate_quality_evidence(
            attestation,
            evaluator_targets_path=targets_path,
            targets=[target],
        )


def test_quality_target_attestation_validates_complete_evidence_chain(tmp_path: Path) -> None:
    pin, prepared_manifest, targets_path, target = _write_bundle(tmp_path / "trust")

    attestation = load_publisher_attestation(pin, prepared_manifest_path=prepared_manifest)
    receipt = validate_quality_evidence(
        attestation,
        evaluator_targets_path=targets_path,
        targets=[target],
    )

    assert receipt.publisher_attestation_digest == pin.expected_digest
    assert receipt.target_set_digest == quality_target_set_digest([target])
    assert receipt.target_count == 1


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda evidence: evidence["evidence"].clear(), "exactly cover"),
        (
            lambda evidence: evidence["evidence"].append(
                {**evidence["evidence"][0], "instance_id": "unknown-instance"}
            ),
            "unknown target",
        ),
        (
            lambda evidence: evidence["evidence"].append(
                dict(evidence["evidence"][0])
            ),
            "repeats an instance",
        ),
    ],
)
def test_quality_evidence_requires_exact_one_to_one_target_coverage(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    pin, prepared_manifest, targets_path, target = _write_bundle(tmp_path / "trust")
    repinned = _repin_changed_evidence(pin, mutate)
    attestation = load_publisher_attestation(
        repinned,
        prepared_manifest_path=prepared_manifest,
    )

    with pytest.raises(QualityAttestationError, match=message):
        validate_quality_evidence(
            attestation,
            evaluator_targets_path=targets_path,
            targets=[target],
        )


def test_quality_evidence_kind_must_match_the_reference_status(tmp_path: Path) -> None:
    pin, prepared_manifest, targets_path, target = _write_bundle(tmp_path / "trust")
    repinned = _repin_changed_evidence(
        pin,
        lambda evidence: evidence["evidence"][0].__setitem__(
            "evidence_kind", "planted-witness"
        ),
    )
    attestation = load_publisher_attestation(
        repinned,
        prepared_manifest_path=prepared_manifest,
    )

    with pytest.raises(QualityAttestationError, match="reference status"):
        validate_quality_evidence(
            attestation,
            evaluator_targets_path=targets_path,
            targets=[target],
        )


def test_quality_evidence_recomputes_target_record_digests() -> None:
    artifact_sha256 = hashlib.sha256(b"certificate").hexdigest()
    target = _target(artifact_sha256)
    target["reference_energy"] = 1234.0

    with pytest.raises(QualityAttestationError, match="target record digest mismatch"):
        quality_target_set_digest([target])


def test_quality_evidence_detects_certificate_artifact_tampering(tmp_path: Path) -> None:
    pin, prepared_manifest, targets_path, target = _write_bundle(tmp_path / "trust")
    artifact = Path(pin.path).parent / "evidence" / "exact.json"
    artifact.write_bytes(b"tampered certificate\n")
    attestation = load_publisher_attestation(pin, prepared_manifest_path=prepared_manifest)

    with pytest.raises(QualityAttestationError, match="artifact checksum"):
        validate_quality_evidence(
            attestation,
            evaluator_targets_path=targets_path,
            targets=[target],
        )


def test_attestation_rejects_evidence_path_escape(tmp_path: Path) -> None:
    pin, prepared_manifest, targets_path, target = _write_bundle(tmp_path / "trust")
    evidence_path = Path(pin.path).parent / "quality-evidence.json"
    evidence = json.loads(evidence_path.read_text())
    evidence["evidence"][0]["artifact_path"] = "../outside.json"
    evidence_payload = {
        key: value for key, value in evidence.items() if key != "record_digest"
    }
    evidence["record_digest"] = content_digest(evidence_payload)
    evidence_path.write_bytes(canonical_json_bytes(evidence) + b"\n")
    attestation_raw = json.loads(Path(pin.path).read_text())
    attestation_raw["evidence_manifest"]["sha256"] = hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    attestation_payload = {
        key: value for key, value in attestation_raw.items() if key != "record_digest"
    }
    attestation_raw["record_digest"] = content_digest(attestation_payload)
    Path(pin.path).write_bytes(canonical_json_bytes(attestation_raw) + b"\n")
    repinned = QualityAttestationPin(
        pin.path,
        attestation_raw["record_digest"],
        pin.expected_publisher_id,
    )
    attestation = load_publisher_attestation(
        repinned,
        prepared_manifest_path=prepared_manifest,
    )

    with pytest.raises(QualityAttestationError, match="safe relative path"):
        validate_quality_evidence(
            attestation,
            evaluator_targets_path=targets_path,
            targets=[target],
        )


def test_attestation_binds_provenance_to_the_published_source_release(
    tmp_path: Path,
) -> None:
    pin, prepared_manifest, _, _ = _write_bundle(tmp_path / "trust")
    attestation = load_publisher_attestation(pin, prepared_manifest_path=prepared_manifest)

    with pytest.raises(QualityAttestationError, match="source release"):
        validate_attested_provenance(
            attestation,
            [
                {
                    "source_release_id": "a-different-release",
                    "source_release_manifest_sha256": "f" * 64,
                }
            ],
        )
