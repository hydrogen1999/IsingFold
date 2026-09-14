from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import isingfold.rl.data.ground_certificate as ground_module
import isingfold.rl.data.import_embedbench as import_module
import isingfold.rl.data.prepared as prepared_module
from isingfold.rl.data.ground_certificate import (
    GROUND_CERTIFICATE_PROTOCOL_V2,
    GroundCertificateError,
    GroundCertificateVerifierPin,
    load_ground_certificate_partition,
    load_ground_certificate_root,
    project_ground_partition_quality_authority,
    project_ground_root_quality_authority,
    verify_ground_certificate_partitions,
)
from isingfold.rl.data.import_embedbench import (
    canonical_json_bytes,
    content_digest,
    prepare_candidate_bank_v4,
)
from isingfold.rl.data.quality_attestation import QualityAttestationPin
from tests.unit.quality_attestation_support import attest_prepared_v4
from tests.unit.test_rl_ground_certificate import _write_verifier
from tests.unit.trust_v4_support import write_v4_inputs


@pytest.fixture(autouse=True)
def _small_production_floors(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (import_module, prepared_module):
        monkeypatch.setattr(module, "MINIMUM_TRAIN_BASE_LINEAGES", 1)
        monkeypatch.setattr(module, "MINIMUM_VALIDATION_BASE_LINEAGES", 1)
        monkeypatch.setattr(module, "MINIMUM_TEST_BASE_LINEAGES", 1)
        monkeypatch.setattr(module, "MINIMUM_VALIDATION_TUNING_BASE_LINEAGES", 1)
        monkeypatch.setattr(module, "VALIDATION_TUNING_PRECISION_CONFIDENCE_LEVEL", 0.51)
        monkeypatch.setattr(module, "VALIDATION_TUNING_PRECISION_MAX_HALF_WIDTH", 0.99)


def _resign(record: dict[str, Any]) -> None:
    payload = {key: value for key, value in record.items() if key != "record_digest"}
    record["record_digest"] = content_digest(payload)


def _publication_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, QualityAttestationPin, GroundCertificateVerifierPin]:
    bank, manifest, targets, provenance, design, design_sha = write_v4_inputs(
        tmp_path / "source"
    )
    corpus = tmp_path / "prepared-v4"
    prepare_candidate_bank_v4(
        bank,
        manifest,
        targets,
        provenance,
        design,
        corpus,
        expected_corpus_design_sha256=design_sha,
        qubit_cap=4,
    )
    trust = tmp_path / "trust"
    quality_pin = attest_prepared_v4(corpus, trust)
    verifier_path = trust / "test-only-verifier"
    executable = _write_verifier(verifier_path)
    executable_sha = hashlib.sha256(executable).hexdigest()
    environment_path = trust / "environment.lock"
    environment_path.write_bytes(b"test-only\n")
    environment_sha = hashlib.sha256(environment_path.read_bytes()).hexdigest()

    attestation_path = Path(quality_pin.path)
    attestation = json.loads(attestation_path.read_text())
    for descriptor in attestation["evidence_manifests"].values():
        evidence_path = trust / descriptor["path"]
        evidence = json.loads(evidence_path.read_text())
        for row in evidence["evidence"]:
            row["verifier"] = {
                "implementation_sha256": executable_sha,
                "name": "fixture-publication-verifier",
                "version": "2.0",
            }
        _resign(evidence)
        evidence_path.write_bytes(canonical_json_bytes(evidence) + b"\n")
        descriptor["sha256"] = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
    _resign(attestation)
    attestation_path.write_bytes(canonical_json_bytes(attestation) + b"\n")
    quality_pin = QualityAttestationPin(
        path=attestation_path,
        expected_digest=attestation["record_digest"],
        expected_publisher_id=attestation["publisher_id"],
    )
    verifier_pin = GroundCertificateVerifierPin(
        executable_path=verifier_path,
        expected_executable_sha256=executable_sha,
        source_path=verifier_path,
        expected_source_sha256=executable_sha,
        environment_path=environment_path,
        expected_environment_sha256=environment_sha,
        expected_name="fixture-publication-verifier",
        expected_version="2.0",
        execution_mode="test-only-host",
    )
    publication_identity: dict[str, object] = {
        "build_attestation_record_digest": "a" * 64,
        "build_attestation_sha256": "b" * 64,
        "environment_sha256": environment_sha,
        "execution_mode": "static-elf",
        "executable_sha256": executable_sha,
        "name": "fixture-publication-verifier",
        "protocol": GROUND_CERTIFICATE_PROTOCOL_V2,
        "runtime_sha256": None,
        "source_sha256": executable_sha,
        "version": "2.0",
    }

    def fixture_identity(
        pin: GroundCertificateVerifierPin,
        *,
        publication: bool = False,
    ) -> tuple[
        dict[str, object],
        bytes,
        tuple[tuple[Path, str], ...],
        ground_module._VerifierExecution,
    ]:
        assert pin is verifier_pin and publication
        return (
            publication_identity,
            executable,
            (
                (verifier_path, executable_sha),
                (environment_path, environment_sha),
            ),
            ground_module._VerifierExecution(
                mode="test-only-host",
                environment_path=environment_path,
                runtime_path=None,
            ),
        )

    monkeypatch.setattr(ground_module, "_verifier_identity", fixture_identity)

    def fixture_run(
        executable_bytes: bytes,
        request: dict[str, object],
        *,
        verifier_identity: dict[str, object],
        execution: ground_module._VerifierExecution,
        timeout_seconds: float,
    ) -> tuple[dict[str, Any], str]:
        del executable_bytes, execution, timeout_seconds
        instance = request["public_instance"]
        assert isinstance(instance, dict)
        payload = {
            "accepted": True,
            "claimed_reference_energy": request["claimed_reference_energy"],
            "instance_id": instance["instance_id"],
            "reason_code": "accepted",
            "request_digest": request["record_digest"],
            "schema": "isingfold.ground-certificate-verifier-result",
            "schema_version": 1,
            "verifier_identity": verifier_identity,
        }
        result = {**payload, "record_digest": content_digest(payload)}
        return result, hashlib.sha256(canonical_json_bytes(result) + b"\n").hexdigest()

    monkeypatch.setattr(ground_module, "_run_verifier", fixture_run)
    return corpus, quality_pin, verifier_pin


def test_v2_root_is_target_free_and_loads_without_opening_sealed_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, quality_pin, verifier_pin = _publication_bundle(tmp_path, monkeypatch)
    output = tmp_path / "ground-v2"
    root_record = verify_ground_certificate_partitions(
        corpus,
        quality_attestation_pin=quality_pin,
        verifier_pin=verifier_pin,
        output_directory=output,
        timeout_seconds=5.0,
    )
    root_path = output / "root.json"
    root_sha = hashlib.sha256(root_path.read_bytes()).hexdigest()
    serialized = root_path.read_text()
    assert "claimed_reference_energy" not in serialized
    assert '"reference_energy"' not in serialized
    assert "targets" not in root_record

    trust = Path(quality_pin.path).parent
    forbidden = {
        *(corpus / "targets" / f"{part}.jsonl" for part in ("train", "val", "test")),
        *(trust / "quality-evidence" / f"{part}.json" for part in ("train", "val", "test")),
        *(output / "partitions" / f"{part}.json" for part in ("train", "val", "test")),
    }
    forbidden = {path.resolve() for path in forbidden}
    original = Path.read_bytes

    def guarded_read(path: Path) -> bytes:
        if path.resolve() in forbidden:
            raise AssertionError(f"sealed artifact was opened: {path}")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    loaded = load_ground_certificate_root(
        root_path,
        expected_sha256=root_sha,
        corpus_directory=corpus,
        quality_attestation_pin=quality_pin,
    )
    assert loaded.record_digest == root_record["record_digest"]
    assert loaded.as_dict() == root_record
    global_authority, test_authority = project_ground_root_quality_authority(
        loaded,
        partition="test",
    )
    assert global_authority.as_dict()["target_authority_record_digest"] == (
        loaded.target_authority_record_digest
    )
    assert test_authority is not None
    assert test_authority.as_dict()["name"] == "test"


def test_v2_partition_loader_opens_only_selected_partition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus, quality_pin, verifier_pin = _publication_bundle(tmp_path, monkeypatch)
    output = tmp_path / "ground-v2"
    verify_ground_certificate_partitions(
        corpus,
        quality_attestation_pin=quality_pin,
        verifier_pin=verifier_pin,
        output_directory=output,
    )
    root_path = output / "root.json"
    root_sha = hashlib.sha256(root_path.read_bytes()).hexdigest()
    trust = Path(quality_pin.path).parent
    forbidden = {
        (corpus / "targets" / "train.jsonl").resolve(),
        (corpus / "targets" / "val.jsonl").resolve(),
        (trust / "quality-evidence" / "train.json").resolve(),
        (trust / "quality-evidence" / "val.json").resolve(),
        (output / "partitions" / "train.json").resolve(),
        (output / "partitions" / "val.json").resolve(),
    }
    original = Path.read_bytes

    def guarded_read(path: Path) -> bytes:
        if path.resolve() in forbidden:
            raise AssertionError(f"other partition was opened: {path}")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    loaded = load_ground_certificate_partition(
        root_path,
        expected_root_sha256=root_sha,
        partition="test",
        corpus_directory=corpus,
        quality_attestation_pin=quality_pin,
    )
    assert loaded.partition == "test"
    assert loaded.tasks and {task.partition for task in loaded.tasks} == {"test"}
    assert loaded.target_access.partition == "test"
    assert loaded.receipt["authority"] == loaded.target_access.as_dict()
    assert "claimed_reference_energy" in json.dumps(loaded.receipt)
    global_authority, partition_authority = project_ground_partition_quality_authority(
        loaded
    )
    assert partition_authority.as_dict()["target_access_record_digest"] == (
        loaded.target_access.record_digest
    )
    assert global_authority.target_authority_record_digest == (
        loaded.target_access.target_authority_record_digest
    )


def test_publication_protocol_rejects_test_only_shebang_verifier(tmp_path: Path) -> None:
    verifier = tmp_path / "verifier"
    raw = _write_verifier(verifier)
    environment = tmp_path / "environment.lock"
    environment.write_bytes(b"test\n")
    pin = GroundCertificateVerifierPin(
        executable_path=verifier,
        expected_executable_sha256=hashlib.sha256(raw).hexdigest(),
        source_path=verifier,
        expected_source_sha256=hashlib.sha256(raw).hexdigest(),
        environment_path=environment,
        expected_environment_sha256=hashlib.sha256(environment.read_bytes()).hexdigest(),
        expected_name="test-verifier",
        expected_version="1",
        execution_mode="test-only-host",
    )

    with pytest.raises(GroundCertificateError, match="cannot authorize a publication"):
        ground_module._verifier_identity(pin, publication=True)


def test_publication_verifier_rejects_resigned_build_attestation_tamper(
    tmp_path: Path,
) -> None:
    executable = tmp_path / "verifier"
    executable.write_bytes(b"\x7fELF" + b"\0" * 80)
    executable.chmod(0o700)
    source = tmp_path / "verifier.c"
    source.write_text("int main(void) { return 0; }\n")
    image = tmp_path / "runtime.sif"
    image.write_bytes(b"fixture-image")
    executable_sha = hashlib.sha256(executable.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    build_payload = {
        "build_recipe_sha256": "1" * 64,
        "executable_sha256": executable_sha,
        "independent_reproduction_count": 2,
        "reproduced_executable_sha256": executable_sha,
        "schema": "isingfold.verifier-build-attestation",
        "schema_version": 1,
        "source_sha256": source_sha,
        "toolchain_image_sha256": "2" * 64,
    }
    build = {**build_payload, "record_digest": content_digest(build_payload)}
    build["independent_reproduction_count"] = 3
    build_path = tmp_path / "build.json"
    build_path.write_bytes(canonical_json_bytes(build) + b"\n")
    runtime = tmp_path / "apptainer"
    runtime.write_bytes(b"\x7fELF" + b"\0" * 80)
    runtime.chmod(0o700)
    pin = GroundCertificateVerifierPin(
        executable_path=executable,
        expected_executable_sha256=executable_sha,
        source_path=source,
        expected_source_sha256=source_sha,
        environment_path=image,
        expected_environment_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
        expected_name="verifier",
        expected_version="1",
        execution_mode="apptainer",
        runtime_path=runtime,
        expected_runtime_sha256=hashlib.sha256(runtime.read_bytes()).hexdigest(),
        build_attestation_path=build_path,
        expected_build_attestation_sha256=hashlib.sha256(build_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(GroundCertificateError, match="record digest mismatch"):
        ground_module._verifier_identity(pin, publication=True)
