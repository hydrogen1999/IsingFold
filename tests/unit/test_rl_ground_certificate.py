from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import isingfold.rl.data.import_embedbench as import_embedbench_module
import isingfold.rl.data.prepared as prepared_module
from isingfold.rl import cli
from isingfold.rl.data.ground_certificate import (
    GroundCertificateError,
    GroundCertificateVerifierPin,
    load_ground_certificate_preflight,
    verify_ground_certificates,
)
from isingfold.rl.data.import_embedbench import (
    canonical_json_bytes,
    content_digest,
    prepare_candidate_bank_v3,
)
from isingfold.rl.data.prepared import load_prepared_tasks
from isingfold.rl.data.quality_attestation import (
    EVIDENCE_KIND_BY_REFERENCE_STATUS,
    PUBLISHER_ATTESTATION_SCHEMA,
    PUBLISHER_ATTESTATION_STATEMENT,
    QUALITY_EVIDENCE_SCHEMA,
    QualityAttestationPin,
    quality_target_set_digest,
)
from tests.unit.test_rl_import_embedbench import _write_design_inputs


@pytest.fixture(autouse=True)
def _small_fixture_partition_floors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep verifier-process fixtures small; production floors have dedicated tests."""

    monkeypatch.setattr(import_embedbench_module, "MINIMUM_TRAIN_BASE_LINEAGES", 1)
    monkeypatch.setattr(import_embedbench_module, "MINIMUM_VALIDATION_BASE_LINEAGES", 1)
    monkeypatch.setattr(import_embedbench_module, "MINIMUM_TEST_BASE_LINEAGES", 1)
    monkeypatch.setattr(
        import_embedbench_module,
        "MINIMUM_VALIDATION_TUNING_BASE_LINEAGES",
        1,
    )
    monkeypatch.setattr(
        import_embedbench_module,
        "VALIDATION_TUNING_PRECISION_CONFIDENCE_LEVEL",
        0.51,
    )
    monkeypatch.setattr(
        import_embedbench_module,
        "VALIDATION_TUNING_PRECISION_MAX_HALF_WIDTH",
        0.99,
    )
    monkeypatch.setattr(prepared_module, "MINIMUM_TRAIN_BASE_LINEAGES", 1)
    monkeypatch.setattr(prepared_module, "MINIMUM_VALIDATION_BASE_LINEAGES", 1)
    monkeypatch.setattr(prepared_module, "MINIMUM_TEST_BASE_LINEAGES", 1)
    monkeypatch.setattr(prepared_module, "MINIMUM_VALIDATION_TUNING_BASE_LINEAGES", 1)
    monkeypatch.setattr(
        prepared_module,
        "VALIDATION_TUNING_PRECISION_CONFIDENCE_LEVEL",
        0.51,
    )
    monkeypatch.setattr(
        prepared_module,
        "VALIDATION_TUNING_PRECISION_MAX_HALF_WIDTH",
        0.99,
    )


def _with_digest(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "record_digest": content_digest(payload)}


def _ground_energy(instance: dict[str, Any]) -> float:
    nodes = instance["logical_nodes"]
    best = float("inf")
    for values in itertools.product((-1, 1), repeat=len(nodes)):
        spins = dict(zip(nodes, values, strict=True))
        energy = sum(float(value) * spins[node] for node, value in instance["h"])
        energy += sum(
            float(value) * spins[first] * spins[second]
            for first, second, value in instance["j"]
        )
        best = min(best, energy)
    return best


def _write_verifier(path: Path, *, canonical_output: bool = True) -> bytes:
    executable = f"""#!{Path(sys.executable).resolve()}
import base64
import hashlib
import itertools
import json
import sys

def canonical(value):
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(\",\", \":\"), sort_keys=True).encode(\"utf-8\")

request = json.loads(sys.stdin.buffer.read())
instance = request[\"public_instance\"]
nodes = instance[\"logical_nodes\"]
energies = []
for values in itertools.product((-1, 1), repeat=len(nodes)):
    spins = dict(zip(nodes, values))
    energy = sum(float(value) * spins[node] for node, value in instance[\"h\"])
    energy += sum(float(value) * spins[first] * spins[second] for first, second, value in instance[\"j\"])
    energies.append(energy)
artifact = json.loads(base64.b64decode(request[\"certificate\"][\"artifact_base64\"], validate=True))
claimed = float(request[\"claimed_reference_energy\"])
accepted = min(energies) == claimed == float(artifact[\"minimum\"]) and int(artifact[\"state_count\"]) == len(energies)
payload = {{
    \"accepted\": accepted,
    \"claimed_reference_energy\": request[\"claimed_reference_energy\"],
    \"instance_id\": instance[\"instance_id\"],
    \"reason_code\": \"accepted\" if accepted else \"energy-mismatch\",
    \"request_digest\": request[\"record_digest\"],
    \"schema\": \"isingfold.ground-certificate-verifier-result\",
    \"schema_version\": 1,
    \"verifier_identity\": request[\"verifier_identity\"],
}}
result = {{**payload, \"record_digest\": hashlib.sha256(canonical(payload)).hexdigest()}}
raw = canonical(result)
if {canonical_output!r}:
    sys.stdout.buffer.write(raw + b\"\\n\")
else:
    sys.stdout.buffer.write(json.dumps(result, indent=2, sort_keys=True).encode() + b\"\\n\")
""".encode()
    path.write_bytes(executable)
    path.chmod(0o700)
    return executable


def _write_scientific_bundle(
    root: Path,
    *,
    reference_status: str = "exact_enumeration",
    false_artifact: bool = False,
    canonical_verifier_output: bool = True,
) -> tuple[Path, QualityAttestationPin, GroundCertificateVerifierPin]:
    inputs = root / "inputs"
    bank, bank_manifest, targets_path, provenance, design, design_digest = (
        _write_design_inputs(inputs)
    )
    instances = {
        row["record"]["instance_id"]: row["record"]
        for row in (json.loads(line) for line in bank.read_text().splitlines())
        if row["kind"] == "instance"
    }
    source_targets = [json.loads(line) for line in targets_path.read_text().splitlines()]
    artifact_bytes_by_instance: dict[str, bytes] = {}
    for index, target in enumerate(source_targets):
        energy = _ground_energy(instances[target["instance_id"]])
        artifact_energy = energy + 1.0 if false_artifact and index == 0 else energy
        artifact = canonical_json_bytes(
            {
                "minimum": artifact_energy,
                "state_count": 2 ** len(instances[target["instance_id"]]["logical_nodes"]),
            }
        ) + b"\n"
        artifact_bytes_by_instance[target["instance_id"]] = artifact
        target["certificate_digest"] = hashlib.sha256(artifact).hexdigest()
        target["reference_energy"] = energy
        target["reference_status"] = reference_status
    targets_path.write_bytes(
        b"".join(canonical_json_bytes(target) + b"\n" for target in source_targets)
    )

    corpus = root / "prepared-v3"
    prepare_candidate_bank_v3(
        bank,
        bank_manifest,
        targets_path,
        provenance,
        design,
        corpus,
        expected_corpus_design_sha256=design_digest,
        qubit_cap=4,
    )
    targets = [
        json.loads(line)
        for line in (corpus / "evaluator_targets.jsonl").read_text().splitlines()
    ]

    trust = root / "trust"
    trust.mkdir()
    verifier_path = trust / "standalone-verifier"
    verifier_bytes = _write_verifier(
        verifier_path,
        canonical_output=canonical_verifier_output,
    )
    verifier_sha256 = hashlib.sha256(verifier_bytes).hexdigest()
    environment_path = trust / "verifier-environment.lock"
    environment_path.write_bytes(b'{"fixture_environment":"stdlib-only"}\n')
    environment_sha256 = hashlib.sha256(environment_path.read_bytes()).hexdigest()

    evidence_rows = []
    for target in targets:
        artifact_path = f"evidence/{target['instance_id']}.json"
        artifact = trust / artifact_path
        artifact.parent.mkdir(exist_ok=True)
        artifact.write_bytes(artifact_bytes_by_instance[target["instance_id"]])
        evidence_rows.append(
            {
                "artifact_path": artifact_path,
                "artifact_sha256": target["certificate_digest"],
                "certificate_digest": target["certificate_digest"],
                "evidence_kind": EVIDENCE_KIND_BY_REFERENCE_STATUS[reference_status],
                "instance_id": target["instance_id"],
                "target_record_digest": target["record_digest"],
                "verifier": {
                    "implementation_sha256": verifier_sha256,
                    "name": "fixture-standalone-ground-verifier",
                    "version": "1.0",
                },
            }
        )
    evidence_payload = {
        "evaluator_targets_sha256": hashlib.sha256(
            (corpus / "evaluator_targets.jsonl").read_bytes()
        ).hexdigest(),
        "evidence": evidence_rows,
        "schema": QUALITY_EVIDENCE_SCHEMA,
        "schema_version": 1,
        "target_count": len(targets),
        "target_set_digest": quality_target_set_digest(targets),
    }
    evidence = _with_digest(evidence_payload)
    evidence_path = trust / "quality-evidence.json"
    evidence_path.write_bytes(canonical_json_bytes(evidence) + b"\n")
    provenance_row = json.loads((corpus / "provenance.jsonl").read_text().splitlines()[0])
    attestation_payload = {
        "evidence_manifest": {
            "path": evidence_path.name,
            "sha256": hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        },
        "prepared_manifest_sha256": hashlib.sha256(
            (corpus / "manifest.json").read_bytes()
        ).hexdigest(),
        "publication_id": "fixture-publication-v1",
        "publisher_id": "fixture-ground-authority",
        "schema": PUBLISHER_ATTESTATION_SCHEMA,
        "schema_version": 1,
        "source_release_id": provenance_row["source_release_id"],
        "source_release_manifest_sha256": provenance_row[
            "source_release_manifest_sha256"
        ],
        "statement": PUBLISHER_ATTESTATION_STATEMENT,
    }
    attestation = _with_digest(attestation_payload)
    attestation_path = trust / "publisher-attestation.json"
    attestation_path.write_bytes(canonical_json_bytes(attestation) + b"\n")
    quality_pin = QualityAttestationPin(
        path=attestation_path,
        expected_digest=attestation["record_digest"],
        expected_publisher_id=attestation["publisher_id"],
    )
    verifier_pin = GroundCertificateVerifierPin(
        executable_path=verifier_path,
        expected_executable_sha256=verifier_sha256,
        source_path=verifier_path,
        expected_source_sha256=verifier_sha256,
        environment_path=environment_path,
        expected_environment_sha256=environment_sha256,
        expected_name="fixture-standalone-ground-verifier",
        expected_version="1.0",
    )
    return corpus, quality_pin, verifier_pin


@pytest.mark.parametrize("reference_status", ["exact_enumeration", "planted_proof"])
def test_ground_certificate_preflight_executes_every_target_and_publishes_atomic_receipt(
    tmp_path: Path,
    reference_status: str,
) -> None:
    corpus, quality_pin, verifier_pin = _write_scientific_bundle(
        tmp_path,
        reference_status=reference_status,
    )
    output = tmp_path / "ground-certificate-receipt.json"

    receipt = verify_ground_certificates(
        corpus,
        quality_attestation_pin=quality_pin,
        verifier_pin=verifier_pin,
        output_path=output,
        timeout_seconds=5.0,
    )

    persisted = json.loads(output.read_text())
    assert persisted == receipt
    assert receipt["schema"] == "isingfold.ground-certificate-preflight"
    assert receipt["census"]["target_count"] == 3
    assert receipt["census"]["accepted_count"] == 3
    assert receipt["census"]["base_lineage_count"] == 3
    assert receipt["census"]["by_reference_status"] == {reference_status: 3}
    assert receipt["census"]["by_design_axis"]["problem_origin"] == {
        "application-derived": 1,
        "synthetic": 2,
    }
    assert len(receipt["targets"]) == 3
    assert all(row["status"] == "accepted" for row in receipt["targets"])
    assert all(len(row["request_sha256"]) == 64 for row in receipt["targets"])
    assert all(len(row["artifact_sha256"]) == 64 for row in receipt["targets"])
    assert all(len(row["result_sha256"]) == 64 for row in receipt["targets"])
    assert (
        receipt["verifier"]["executable_sha256"]
        == verifier_pin.expected_executable_sha256
    )
    assert receipt["verifier"]["source_sha256"] == verifier_pin.expected_source_sha256
    assert (
        receipt["verifier"]["environment_sha256"]
        == verifier_pin.expected_environment_sha256
    )
    assert receipt["record_digest"] == content_digest(
        {key: value for key, value in receipt.items() if key != "record_digest"}
    )
    assert output.read_bytes() == canonical_json_bytes(receipt) + b"\n"
    expected_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    assert (
        load_ground_certificate_preflight(
            output,
            expected_sha256=expected_sha256,
            corpus_directory=corpus,
            quality_attestation_pin=quality_pin,
        )
        == receipt
    )

    with pytest.raises(FileExistsError):
        verify_ground_certificates(
            corpus,
            quality_attestation_pin=quality_pin,
            verifier_pin=verifier_pin,
            output_path=output,
        )


def test_ground_certificate_preflight_loader_rejects_resigned_census_tamper(
    tmp_path: Path,
) -> None:
    corpus, quality_pin, verifier_pin = _write_scientific_bundle(tmp_path)
    output = tmp_path / "ground-certificate-receipt.json"
    receipt = verify_ground_certificates(
        corpus,
        quality_attestation_pin=quality_pin,
        verifier_pin=verifier_pin,
        output_path=output,
    )
    receipt["census"]["target_count"] -= 1
    payload = {key: value for key, value in receipt.items() if key != "record_digest"}
    receipt["record_digest"] = content_digest(payload)
    output.write_bytes(canonical_json_bytes(receipt) + b"\n")

    with pytest.raises(GroundCertificateError, match="full current target census"):
        load_ground_certificate_preflight(
            output,
            expected_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            corpus_directory=corpus,
            quality_attestation_pin=quality_pin,
        )


def test_ground_certificate_preflight_loader_rejects_wrong_corpus(tmp_path: Path) -> None:
    corpus, quality_pin, verifier_pin = _write_scientific_bundle(tmp_path / "first")
    output = tmp_path / "ground-certificate-receipt.json"
    verify_ground_certificates(
        corpus,
        quality_attestation_pin=quality_pin,
        verifier_pin=verifier_pin,
        output_path=output,
    )
    other_corpus, other_quality_pin, _other_verifier_pin = _write_scientific_bundle(
        tmp_path / "second",
        reference_status="planted_proof",
    )

    with pytest.raises(GroundCertificateError, match="another prepared corpus"):
        load_ground_certificate_preflight(
            output,
            expected_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            corpus_directory=other_corpus,
            quality_attestation_pin=other_quality_pin,
        )


def test_ground_certificate_preflight_loader_requires_out_of_band_file_pin(
    tmp_path: Path,
) -> None:
    corpus, quality_pin, verifier_pin = _write_scientific_bundle(tmp_path)
    output = tmp_path / "ground-certificate-receipt.json"
    verify_ground_certificates(
        corpus,
        quality_attestation_pin=quality_pin,
        verifier_pin=verifier_pin,
        output_path=output,
    )

    with pytest.raises(GroundCertificateError, match="out-of-band SHA-256 pin"):
        load_ground_certificate_preflight(
            output,
            expected_sha256="0" * 64,
            corpus_directory=corpus,
            quality_attestation_pin=quality_pin,
        )


def test_hash_valid_but_mathematically_false_certificate_is_rejected(tmp_path: Path) -> None:
    corpus, quality_pin, verifier_pin = _write_scientific_bundle(
        tmp_path,
        false_artifact=True,
    )
    output = tmp_path / "ground-certificate-receipt.json"

    # The existing hash/authority boundary accepts this internally consistent artifact bundle.
    assert load_prepared_tasks(
        corpus,
        include_evaluator=True,
        quality_attestation_pin=quality_pin,
    )
    with pytest.raises(GroundCertificateError, match="energy-mismatch"):
        verify_ground_certificates(
            corpus,
            quality_attestation_pin=quality_pin,
            verifier_pin=verifier_pin,
            output_path=output,
        )
    assert not output.exists()


def test_ground_certificate_preflight_rejects_unimplemented_optimality_checker(
    tmp_path: Path,
) -> None:
    corpus, quality_pin, verifier_pin = _write_scientific_bundle(
        tmp_path,
        reference_status="certified_optimal",
    )

    with pytest.raises(GroundCertificateError, match="certified_optimal.*not supported"):
        verify_ground_certificates(
            corpus,
            quality_attestation_pin=quality_pin,
            verifier_pin=verifier_pin,
            output_path=tmp_path / "receipt.json",
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("expected_executable_sha256", "executable differs"),
        ("expected_source_sha256", "source differs"),
        ("expected_environment_sha256", "environment differs"),
    ],
)
def test_ground_certificate_preflight_rejects_wrong_verifier_pin(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    corpus, quality_pin, verifier_pin = _write_scientific_bundle(tmp_path)
    bad_pin = replace(verifier_pin, **{field: "0" * 64})

    with pytest.raises(GroundCertificateError, match=message):
        verify_ground_certificates(
            corpus,
            quality_attestation_pin=quality_pin,
            verifier_pin=bad_pin,
            output_path=tmp_path / "receipt.json",
        )


def test_ground_certificate_preflight_requires_canonical_verifier_output(
    tmp_path: Path,
) -> None:
    corpus, quality_pin, verifier_pin = _write_scientific_bundle(
        tmp_path,
        canonical_verifier_output=False,
    )

    with pytest.raises(GroundCertificateError, match="canonical JSON"):
        verify_ground_certificates(
            corpus,
            quality_attestation_pin=quality_pin,
            verifier_pin=verifier_pin,
            output_path=tmp_path / "receipt.json",
        )


def test_cli_rejects_retired_monolithic_ground_preflight_authority(
    tmp_path: Path,
) -> None:
    corpus, publisher_pin, verifier_pin = _write_scientific_bundle(tmp_path)
    output = tmp_path / "ground-certificate-receipt.json"
    receipt = verify_ground_certificates(
        corpus,
        quality_attestation_pin=publisher_pin,
        verifier_pin=verifier_pin,
        output_path=output,
    )
    receipt_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    args = argparse.Namespace(
        corpus=str(corpus),
        quality_attestation=str(publisher_pin.path),
        expected_quality_attestation_digest=publisher_pin.expected_digest,
        expected_quality_publisher_id=publisher_pin.expected_publisher_id,
        ground_certificate_preflight=str(output),
        expected_ground_certificate_preflight_sha256=receipt_sha256,
    )

    assert receipt["record_digest"]
    with pytest.raises(ValueError, match="ground-certificate-root"):
        cli._quality_attestation_pin(args)


def test_ground_verifier_command_does_not_require_its_own_output_receipt() -> None:
    parsed = cli.build_parser().parse_args(
        [
            "verify-ground-certificates",
            "--corpus",
            "prepared",
            "--quality-attestation",
            "publisher.json",
            "--expected-quality-attestation-digest",
            "a" * 64,
            "--expected-quality-publisher-id",
            "publisher",
            "--verifier-executable",
            "verifier",
            "--expected-verifier-executable-sha256",
            "b" * 64,
            "--verifier-source",
            "source",
            "--expected-verifier-source-sha256",
            "c" * 64,
            "--verifier-environment",
            "environment",
            "--expected-verifier-environment-sha256",
            "d" * 64,
            "--expected-verifier-name",
            "independent-verifier",
            "--expected-verifier-version",
            "1.0",
            "--verifier-execution-mode",
            "static-elf",
            "--verifier-build-attestation",
            "verifier-build-attestation.json",
            "--expected-verifier-build-attestation-sha256",
            "e" * 64,
            "--out",
            "ground-certificates",
        ]
    )

    assert not hasattr(parsed, "ground_certificate_root")
