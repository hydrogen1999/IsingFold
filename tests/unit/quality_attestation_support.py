from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.quality_attestation import (
    EVIDENCE_KIND_BY_REFERENCE_STATUS,
    PUBLISHER_ATTESTATION_SCHEMA,
    PUBLISHER_ATTESTATION_STATEMENT,
    QUALITY_EVIDENCE_SCHEMA,
    QualityAttestationPin,
    quality_target_set_digest,
)

FIXTURE_CERTIFICATE_BYTES = b'{"kind":"fixture-proof","verified":true}\n'
FIXTURE_CERTIFICATE_DIGEST = hashlib.sha256(FIXTURE_CERTIFICATE_BYTES).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def attest_prepared(output: Path, trust_root: Path) -> QualityAttestationPin:
    """Publish a complete test-only trust bundle for one prepared fixture."""

    trust_root.mkdir(parents=True, exist_ok=True)
    targets = _read_jsonl(output / "evaluator_targets.jsonl")
    evidence_rows = []
    for target in targets:
        if target["certificate_digest"] != FIXTURE_CERTIFICATE_DIGEST:
            raise AssertionError("fixture attestation only supports its registered certificate")
        artifact_path = f"evidence/{target['certificate_digest']}"
        artifact = trust_root / artifact_path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(FIXTURE_CERTIFICATE_BYTES)
        evidence_rows.append(
            {
                "artifact_path": artifact_path,
                "artifact_sha256": target["certificate_digest"],
                "certificate_digest": target["certificate_digest"],
                "evidence_kind": EVIDENCE_KIND_BY_REFERENCE_STATUS[
                    target["reference_status"]
                ],
                "instance_id": target["instance_id"],
                "target_record_digest": target["record_digest"],
                "verifier": {
                    "implementation_sha256": "d" * 64,
                    "name": "fixture-proof-verifier",
                    "version": "1.0",
                },
            }
        )
    evidence_payload = {
        "evaluator_targets_sha256": hashlib.sha256(
            (output / "evaluator_targets.jsonl").read_bytes()
        ).hexdigest(),
        "evidence": evidence_rows,
        "schema": QUALITY_EVIDENCE_SCHEMA,
        "schema_version": 1,
        "target_count": len(targets),
        "target_set_digest": quality_target_set_digest(targets),
    }
    evidence = {**evidence_payload, "record_digest": content_digest(evidence_payload)}
    evidence_path = trust_root / "quality-evidence.json"
    evidence_path.write_bytes(canonical_json_bytes(evidence) + b"\n")
    provenance_path = output / "provenance.jsonl"
    if provenance_path.is_file():
        provenance = _read_jsonl(provenance_path)[0]
        source_release_id = provenance["source_release_id"]
        source_release_manifest_sha256 = provenance["source_release_manifest_sha256"]
    else:
        source_release_id = "embedbench-legacy-fixture"
        source_release_manifest_sha256 = "e" * 64
    attestation_payload = {
        "evidence_manifest": {
            "path": evidence_path.name,
            "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        },
        "prepared_manifest_sha256": hashlib.sha256(
            (output / "manifest.json").read_bytes()
        ).hexdigest(),
        "publication_id": "fixture-publication-v1",
        "publisher_id": "isingfold-test-authority",
        "schema": PUBLISHER_ATTESTATION_SCHEMA,
        "schema_version": 1,
        "source_release_id": source_release_id,
        "source_release_manifest_sha256": source_release_manifest_sha256,
        "statement": PUBLISHER_ATTESTATION_STATEMENT,
    }
    attestation = {
        **attestation_payload,
        "record_digest": content_digest(attestation_payload),
    }
    attestation_path = trust_root / "publisher-attestation.json"
    attestation_path.write_bytes(canonical_json_bytes(attestation) + b"\n")
    return QualityAttestationPin(
        path=attestation_path,
        expected_digest=attestation["record_digest"],
        expected_publisher_id=attestation["publisher_id"],
    )


def attest_prepared_v4(output: Path, trust_root: Path) -> QualityAttestationPin:
    """Publish partition-scoped quality evidence for a prepared-v4 fixture."""

    trust_root.mkdir(parents=True, exist_ok=True)
    artifact_path = f"evidence/{FIXTURE_CERTIFICATE_DIGEST}"
    artifact = trust_root / "quality-evidence" / artifact_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(FIXTURE_CERTIFICATE_BYTES)
    evidence_manifests: dict[str, dict[str, object]] = {}
    for partition in ("train", "val", "test"):
        target_path = output / "targets" / f"{partition}.jsonl"
        targets = _read_jsonl(target_path)
        evidence_rows = []
        for target in targets:
            if target["certificate_digest"] != FIXTURE_CERTIFICATE_DIGEST:
                raise AssertionError("fixture attestation has an unknown certificate")
            evidence_rows.append(
                {
                    "artifact_path": artifact_path,
                    "artifact_sha256": FIXTURE_CERTIFICATE_DIGEST,
                    "certificate_digest": FIXTURE_CERTIFICATE_DIGEST,
                    "evidence_kind": EVIDENCE_KIND_BY_REFERENCE_STATUS[
                        target["reference_status"]
                    ],
                    "instance_id": target["instance_id"],
                    "target_record_digest": target["record_digest"],
                    "verifier": {
                        "implementation_sha256": "d" * 64,
                        "name": "fixture-proof-verifier",
                        "version": "1.0",
                    },
                }
            )
        target_set_digest = quality_target_set_digest(targets, partition=partition)
        evidence_payload = {
            "evaluator_targets_sha256": hashlib.sha256(target_path.read_bytes()).hexdigest(),
            "evidence": evidence_rows,
            "partition": partition,
            "schema": QUALITY_EVIDENCE_SCHEMA,
            "schema_version": 2,
            "target_count": len(targets),
            "target_set_digest": target_set_digest,
        }
        evidence = {**evidence_payload, "record_digest": content_digest(evidence_payload)}
        relative = f"quality-evidence/{partition}.json"
        evidence_path = trust_root / relative
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_bytes(canonical_json_bytes(evidence) + b"\n")
        evidence_manifests[partition] = {
            "path": relative,
            "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            "target_count": len(targets),
            "target_set_digest": target_set_digest,
        }
    provenance = _read_jsonl(output / "provenance.jsonl")[0]
    prepared_manifest = json.loads((output / "manifest.json").read_text())
    attestation_payload = {
        "evidence_manifests": evidence_manifests,
        "prepared_manifest_sha256": hashlib.sha256(
            (output / "manifest.json").read_bytes()
        ).hexdigest(),
        "publication_id": "fixture-publication-v2",
        "publisher_id": "isingfold-test-authority",
        "schema": PUBLISHER_ATTESTATION_SCHEMA,
        "schema_version": 2,
        "source_release_id": provenance["source_release_id"],
        "source_release_manifest_sha256": provenance[
            "source_release_manifest_sha256"
        ],
        "statement": PUBLISHER_ATTESTATION_STATEMENT,
        "target_authority_record_digest": prepared_manifest["target_authority"][
            "record_digest"
        ],
    }
    attestation = {
        **attestation_payload,
        "record_digest": content_digest(attestation_payload),
    }
    attestation_path = trust_root / "publisher-attestation.json"
    attestation_path.write_bytes(canonical_json_bytes(attestation) + b"\n")
    return QualityAttestationPin(
        path=attestation_path,
        expected_digest=attestation["record_digest"],
        expected_publisher_id=attestation["publisher_id"],
    )


__all__ = ["FIXTURE_CERTIFICATE_DIGEST", "attest_prepared", "attest_prepared_v4"]
