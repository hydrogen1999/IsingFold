from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from embedbench.candidate_bank import canonical_json_bytes, content_digest
from embedbench.isingfold_attestation import (
    PublisherAttestationError,
    VerifierIdentity,
    build_publisher_attestation_v2,
    verify_publisher_attestation_v2,
)

PARTITIONS = ("train", "val", "test")


def _record(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "record_digest": content_digest(payload)}


def _write_canonical(path: Path, value: object) -> bytes:
    raw = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return raw


def _target(partition: str, certificate_digest: str) -> dict[str, object]:
    return _record(
        {
            "certificate_digest": certificate_digest,
            "evaluator_protocol_digest": "b" * 64,
            "instance_id": f"instance-{partition}",
            "instance_record_digest": "c" * 64,
            "learning_partition": partition,
            "reference_energy": -1.25,
            "reference_status": "exact_enumeration",
            "schema": "isingfold.evaluator-target",
            "schema_version": 2,
        }
    )


def _target_set_digest(rows: list[dict[str, object]], partition: str) -> str:
    return content_digest(
        {
            "domain": "isingfold-partition-target-set-v1",
            "partition": partition,
            "targets": [
                {
                    "instance_id": row["instance_id"],
                    "target_record_digest": row["record_digest"],
                }
                for row in sorted(rows, key=lambda row: str(row["instance_id"]))
            ],
        }
    )


def _prepared_v4(root: Path) -> tuple[Path, dict[str, Path]]:
    root.mkdir()
    certificate_paths: dict[str, Path] = {}
    targets: dict[str, list[dict[str, object]]] = {}
    for partition in PARTITIONS:
        certificate = root.parent / "source-certificates" / f"{partition}.json"
        certificate_raw = _write_canonical(
            certificate,
            {"partition": partition, "proof": "fixture"},
        )
        digest = hashlib.sha256(certificate_raw).hexdigest()
        certificate_paths[digest] = certificate
        targets[partition] = [_target(partition, digest)]

    policy_rows = [
        _record(
            {
                "instance_id": f"instance-{partition}",
                "schema": "isingfold.policy-instance",
                "schema_version": 1,
            }
        )
        for partition in PARTITIONS
    ]
    initializer_rows = [
        _record(
            {
                "instance_id": f"instance-{partition}",
                "schema": "isingfold.profile-i-initializer",
                "schema_version": 1,
                "task_id": f"task-{partition}",
            }
        )
        for partition in PARTITIONS
    ]
    provenance_rows = [
        _record(
            {
                "active_topology_identity": {},
                "base_parent_lineage": f"lineage-{partition}",
                "calibration_identity": {},
                "descendant_transform_identity": {},
                "distribution": {"learning_partition": partition},
                "fault_identity": {},
                "group_id": f"group-{partition}",
                "group_record_digest": "d" * 64,
                "instance_id": f"instance-{partition}",
                "instance_record_digest": "c" * 64,
                "nominal_topology_identity": {},
                "schema": "isingfold.task-provenance",
                "schema_version": 2,
                "source_logical_lineage": f"logical-{partition}",
                "source_record_digest": "e" * 64,
                "source_release_id": "embedbench-release-fixture",
                "source_release_manifest_sha256": "f" * 64,
                "split_manifest_sha256": "1" * 64,
                "task_id": f"task-{partition}",
            }
        )
        for partition in PARTITIONS
    ]
    splits = _record(
        {
            "base_parent_lineage_to_split": {
                f"lineage-{partition}": partition for partition in PARTITIONS
            },
            "schema": "isingfold.lineage-splits",
            "schema_version": 4,
            "test": ["task-test"],
            "train": ["task-train"],
            "val": ["task-val"],
        }
    )

    raw_files: dict[str, bytes] = {
        "policy_instances.jsonl": b"".join(
            canonical_json_bytes(row) + b"\n" for row in policy_rows
        ),
        "initializers.jsonl": b"".join(
            canonical_json_bytes(row) + b"\n" for row in initializer_rows
        ),
        "provenance.jsonl": b"".join(
            canonical_json_bytes(row) + b"\n" for row in provenance_rows
        ),
        "splits.json": canonical_json_bytes(splits) + b"\n",
    }
    for partition in PARTITIONS:
        raw_files[f"targets/{partition}.jsonl"] = b"".join(
            canonical_json_bytes(row) + b"\n" for row in targets[partition]
        )
    for relative, raw in raw_files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    target_descriptors = {
        partition: {
            "path": f"targets/{partition}.jsonl",
            "records": 1,
            "sha256": hashlib.sha256(raw_files[f"targets/{partition}.jsonl"]).hexdigest(),
            "target_set_digest": _target_set_digest(targets[partition], partition),
        }
        for partition in PARTITIONS
    }
    authority = _record(
        {
            "partitions": target_descriptors,
            "schema": "isingfold.partitioned-target-authority",
            "schema_version": 1,
            "total_targets": 3,
        }
    )
    output_receipts = {
        relative: {
            "records": 1 if relative == "splits.json" or relative.startswith("targets/") else 3,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        for relative, raw in raw_files.items()
    }
    manifest_payload: dict[str, object] = {
        "corpus_design": {
            "corpus_design_version": "if-core-v2",
            "schema": "isingfold.prepared-corpus-design-receipt",
            "schema_version": 2,
        },
        "corpus_scope": "production-designed-v4",
        "counts": {
            "evaluator_targets": 3,
            "evaluator_targets_by_partition": {
                "test": 1,
                "train": 1,
                "val": 1,
            },
            "initializers": 3,
            "policy_instances": 3,
            "provenance_records": 3,
        },
        "outputs": output_receipts,
        "policy_model_feature_allowlist": [
            "family",
            "h",
            "host_edges",
            "host_nodes",
            "j",
            "logical_edges",
            "logical_nodes",
            "topology",
        ],
        "provenance_record_schema": "isingfold.task-provenance",
        "qubit_cap": 4,
        "schema": "isingfold.prepared-candidate-bank",
        "schema_version": 4,
        "source_bank_manifest_record_digest": "2" * 64,
        "source_sha256": {
            "candidate_bank_jsonl": "3" * 64,
            "candidate_bank_manifest": "4" * 64,
            "corpus_design_manifest": "5" * 64,
            "evaluator_targets": "6" * 64,
            "task_provenance": "7" * 64,
        },
        "target_authority": authority,
    }
    _write_canonical(root / "manifest.json", _record(manifest_payload))
    return root, certificate_paths


def _verifier() -> VerifierIdentity:
    return VerifierIdentity(
        name="embedbench-ground-certificate-verifier",
        version="2.0.0",
        implementation_sha256="8" * 64,
    )


def test_builds_and_independently_verifies_partitioned_attestation_v2(
    tmp_path: Path,
) -> None:
    prepared, certificates = _prepared_v4(tmp_path / "prepared-v4")
    output = tmp_path / "publisher"

    built = build_publisher_attestation_v2(
        prepared,
        certificates,
        output,
        publisher_id="embedbench-release-authority",
        publication_id="embedbench-release-v2",
        verifier=_verifier(),
    )

    assert built.attestation_path == output / "publisher-attestation.json"
    assert built.target_count == 3
    assert built.certificate_count == 3
    assert built.attestation_path.read_bytes().endswith(b"\n")
    attestation = json.loads(built.attestation_path.read_text())
    assert attestation["schema"] == "isingfold.quality-publisher-attestation"
    assert attestation["schema_version"] == 2
    assert set(attestation["evidence_manifests"]) == set(PARTITIONS)

    for partition in PARTITIONS:
        evidence_path = output / "quality-evidence" / f"{partition}.json"
        raw = evidence_path.read_bytes()
        evidence = json.loads(raw)
        assert raw == canonical_json_bytes(evidence) + b"\n"
        assert evidence["schema"] == "isingfold.quality-ground-evidence"
        assert evidence["schema_version"] == 2
        assert evidence["partition"] == partition
        assert evidence["evidence"][0]["verifier"] == _verifier().as_dict()

    verified = verify_publisher_attestation_v2(
        prepared,
        output,
        expected_attestation_record_digest=built.attestation_record_digest,
        expected_publisher_id=built.publisher_id,
        expected_verifier=_verifier(),
    )
    assert verified == built


def test_builder_requires_exact_certificate_digest_coverage(tmp_path: Path) -> None:
    prepared, certificates = _prepared_v4(tmp_path / "prepared-v4")
    missing = dict(certificates)
    missing.pop(next(iter(missing)))

    with pytest.raises(PublisherAttestationError, match="certificate coverage"):
        build_publisher_attestation_v2(
            prepared,
            missing,
            tmp_path / "publisher-missing",
            publisher_id="publisher",
            publication_id="publication",
            verifier=_verifier(),
        )
    assert not (tmp_path / "publisher-missing").exists()

    extra_path = tmp_path / "extra-certificate"
    extra_path.write_bytes(b"extra\n")
    extra = {**certificates, hashlib.sha256(extra_path.read_bytes()).hexdigest(): extra_path}
    with pytest.raises(PublisherAttestationError, match="certificate coverage"):
        build_publisher_attestation_v2(
            prepared,
            extra,
            tmp_path / "publisher-extra",
            publisher_id="publisher",
            publication_id="publication",
            verifier=_verifier(),
        )


def test_builder_recomputes_certificate_sha_and_fails_atomically(tmp_path: Path) -> None:
    prepared, certificates = _prepared_v4(tmp_path / "prepared-v4")
    digest, artifact = next(iter(certificates.items()))
    artifact.write_bytes(b"tampered\n")
    output = tmp_path / "publisher"

    with pytest.raises(PublisherAttestationError, match="certificate SHA-256"):
        build_publisher_attestation_v2(
            prepared,
            certificates,
            output,
            publisher_id="publisher",
            publication_id="publication",
            verifier=_verifier(),
        )
    assert not output.exists()
    assert digest in certificates


def test_builder_rejects_existing_output_without_mutating_it(tmp_path: Path) -> None:
    prepared, certificates = _prepared_v4(tmp_path / "prepared-v4")
    output = tmp_path / "publisher"
    output.mkdir()
    marker = output / "owned-by-user"
    marker.write_text("preserve", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        build_publisher_attestation_v2(
            prepared,
            certificates,
            output,
            publisher_id="publisher",
            publication_id="publication",
            verifier=_verifier(),
        )
    assert marker.read_text(encoding="utf-8") == "preserve"


def test_builder_rejects_tampered_prepared_target_authority(tmp_path: Path) -> None:
    prepared, certificates = _prepared_v4(tmp_path / "prepared-v4")
    target = prepared / "targets" / "train.jsonl"
    row = json.loads(target.read_text())
    row["reference_energy"] = -2.0
    payload = {key: value for key, value in row.items() if key != "record_digest"}
    row["record_digest"] = content_digest(payload)
    target.write_bytes(canonical_json_bytes(row) + b"\n")

    with pytest.raises(PublisherAttestationError, match="target.*SHA-256"):
        build_publisher_attestation_v2(
            prepared,
            certificates,
            tmp_path / "publisher",
            publisher_id="publisher",
            publication_id="publication",
            verifier=_verifier(),
        )


def test_verifier_rejects_safe_path_escape_even_when_records_are_resigned(
    tmp_path: Path,
) -> None:
    prepared, certificates = _prepared_v4(tmp_path / "prepared-v4")
    output = tmp_path / "publisher"
    built = build_publisher_attestation_v2(
        prepared,
        certificates,
        output,
        publisher_id="publisher",
        publication_id="publication",
        verifier=_verifier(),
    )
    attestation = json.loads(built.attestation_path.read_text())
    attestation["evidence_manifests"]["train"]["path"] = "../outside.json"
    payload = {key: value for key, value in attestation.items() if key != "record_digest"}
    attestation["record_digest"] = content_digest(payload)
    built.attestation_path.write_bytes(canonical_json_bytes(attestation) + b"\n")

    with pytest.raises(PublisherAttestationError, match="safe relative path"):
        verify_publisher_attestation_v2(
            prepared,
            output,
            expected_attestation_record_digest=attestation["record_digest"],
            expected_publisher_id="publisher",
            expected_verifier=_verifier(),
        )


def test_verifier_identity_is_a_pinned_part_of_every_evidence_row(tmp_path: Path) -> None:
    prepared, certificates = _prepared_v4(tmp_path / "prepared-v4")
    output = tmp_path / "publisher"
    built = build_publisher_attestation_v2(
        prepared,
        certificates,
        output,
        publisher_id="publisher",
        publication_id="publication",
        verifier=_verifier(),
    )
    wrong = VerifierIdentity(
        name=_verifier().name,
        version="different",
        implementation_sha256=_verifier().implementation_sha256,
    )

    with pytest.raises(PublisherAttestationError, match="verifier identity"):
        verify_publisher_attestation_v2(
            prepared,
            output,
            expected_attestation_record_digest=built.attestation_record_digest,
            expected_publisher_id="publisher",
            expected_verifier=wrong,
        )


def test_verifier_rejects_unlisted_certificate_files(tmp_path: Path) -> None:
    prepared, certificates = _prepared_v4(tmp_path / "prepared-v4")
    output = tmp_path / "publisher"
    built = build_publisher_attestation_v2(
        prepared,
        certificates,
        output,
        publisher_id="publisher",
        publication_id="publication",
        verifier=_verifier(),
    )
    (output / "quality-evidence" / "certificates" / ("9" * 64)).write_bytes(b"extra")

    with pytest.raises(PublisherAttestationError, match="artifact set"):
        verify_publisher_attestation_v2(
            prepared,
            output,
            expected_attestation_record_digest=built.attestation_record_digest,
            expected_publisher_id="publisher",
            expected_verifier=_verifier(),
        )
