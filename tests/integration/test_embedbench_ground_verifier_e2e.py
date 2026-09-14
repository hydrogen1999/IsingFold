from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
from embedbench.candidate_bank import canonical_json_bytes, content_digest
from embedbench.ground_certificate import (
    CheckerRun,
    IsingProblem,
    SolverRun,
    build_exact_enumeration_certificate,
)
from embedbench.isingfold_attestation import (
    VerifierIdentity as PublisherVerifierIdentity,
)
from embedbench.isingfold_attestation import (
    build_publisher_attestation_v2,
    verify_publisher_attestation_v2,
)
from embedbench.isingfold_reference_adapter import adapt_reference_export_bundle
from embedbench.reference_export import (
    export_problem_references,
    verify_problem_reference_export,
)

import isingfold.rl.data.ground_certificate as ground_module
import isingfold.rl.data.import_embedbench as import_module
import isingfold.rl.data.prepared as prepared_module
from isingfold.rl.data.ground_certificate import (
    GROUND_CERTIFICATE_PROTOCOL_V2,
    GroundCertificateVerifierPin,
    load_ground_certificate_partition,
    load_ground_certificate_root,
    verify_ground_certificate_partitions,
)
from isingfold.rl.data.import_embedbench import prepare_candidate_bank_v4
from isingfold.rl.data.quality_attestation import QualityAttestationPin
from tests.unit.trust_v4_support import write_v4_inputs


ROOT = Path(__file__).resolve().parents[2]
VERIFIER_SOURCE = ROOT / "tools" / "ground_verifier" / "verifier.cpp"
VERIFIER_NAME = "isingfold-independent-ground-verifier"
VERIFIER_VERSION = "1.0.0"
PARTITIONS = ("train", "val", "test")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def _problem(instance: dict[str, Any]) -> IsingProblem:
    return IsingProblem(
        variables=tuple(instance["logical_nodes"]),
        linear=tuple((node, coefficient) for node, coefficient in instance["h"]),
        quadratic=tuple(
            (left, right, coefficient) for left, right, coefficient in instance["j"]
        ),
    )


def _exact_energy(problem: IsingProblem) -> float:
    state_count = 1 << len(problem.variables)
    bundle = build_exact_enumeration_certificate(
        problem,
        solver=SolverRun(
            name="embedbench-integration-enumerator",
            version="1",
            command=("registered-gray-code-enumeration",),
            seed=None,
            deterministic_work_limit=state_count,
            safety_timeout_seconds=None,
        ),
        checker=CheckerRun(
            source_sha256="1" * 64,
            environment_sha256="2" * 64,
            command=("registered-exhaustive-proof-check",),
            exit_code=0,
            log_sha256="3" * 64,
        ),
    )
    return bundle.certificate.energy.to_float()


def _export_real_references(
    root: Path,
    instances: dict[str, dict[str, Any]],
) -> Path:
    release = root / "release"
    release.mkdir(parents=True)
    corpus = release / "quality_integration_random.jsonl"
    rows = []
    for instance_id, instance in sorted(instances.items()):
        problem = _problem(instance)
        rows.append(
            {
                "instance_id": instance_id,
                "mode": "random",
                "problem": {
                    "J": [list(term) for term in problem.quadratic],
                    "e0": _exact_energy(problem),
                    "h": {str(node): coefficient for node, coefficient in problem.linear},
                },
            }
        )
    _write_jsonl(corpus, rows)
    corpus_sha256 = _sha256(corpus)
    manifest = release / f"{corpus.name}.manifest.json"
    manifest.write_bytes(
        canonical_json_bytes(
            {
                "file": f"runs/integration/{corpus.name}",
                "sha256": corpus_sha256,
            }
        )
        + b"\n"
    )
    checksums = release / "SHA256SUMS"
    checksums.write_text(
        f"{corpus_sha256}  {corpus.name}\n{_sha256(manifest)}  {manifest.name}\n",
        encoding="utf-8",
    )

    exported = root / "reference-export"
    export_problem_references(
        [corpus],
        checksums_path=checksums,
        output_dir=exported,
        max_exact_variables=2,
    )
    census = verify_problem_reference_export(
        [corpus],
        checksums_path=checksums,
        export_dir=exported,
    )
    assert census["certified_problems"] == len(instances)

    adapted = root / "isingfold-references"
    adapt_reference_export_bundle(
        exported,
        adapted,
        expected_manifest_sha256=_sha256(
            exported / "problem_reference_export_manifest.json"
        ),
        expected_protocol_sha256=_sha256(exported / "problem_reference_protocol.json"),
        expected_references_sha256=_sha256(exported / "problem_references.jsonl"),
        expected_certificates_sha256=_sha256(
            exported / "problem_reference_certificates.jsonl"
        ),
    )
    return adapted


def _bind_source_targets_to_references(
    targets_path: Path,
    instances: dict[str, dict[str, Any]],
    references_root: Path,
) -> dict[str, Path]:
    references = {
        row["problem_sha256"]: row
        for row in _jsonl(references_root / "isingfold_references.jsonl")
    }
    targets = _jsonl(targets_path)
    certificate_artifacts: dict[str, Path] = {}
    for target in targets:
        instance = instances[target["instance_id"]]
        reference = references[_problem(instance).problem_sha256]
        certificate = reference["certificate"]
        target.update(
            certificate_digest=certificate["sha256"],
            evaluator_protocol_digest=reference["evaluator_protocol_digest"],
            reference_energy=reference["reference_energy"],
            reference_status=reference["reference_status"],
        )
        artifact = references_root / certificate["path"]
        assert _sha256(artifact) == certificate["sha256"]
        assert json.loads(artifact.read_bytes())["schema"] == (
            "embedbench.problem-reference-certificate"
        )
        certificate_artifacts[certificate["sha256"]] = artifact
    _write_jsonl(targets_path, targets)
    return certificate_artifacts


@pytest.fixture(scope="module")
def compiled_verifier(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        pytest.skip("a C++17 compiler is required for the cross-runtime verifier test")
    output = tmp_path_factory.mktemp("embedbench-ground-verifier") / "verifier"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            str(VERIFIER_SOURCE),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
    )
    return output


def _allow_small_authentic_fixture(monkeypatch: pytest.MonkeyPatch) -> None:
    for module in (import_module, prepared_module):
        monkeypatch.setattr(module, "MINIMUM_TRAIN_BASE_LINEAGES", 1)
        monkeypatch.setattr(module, "MINIMUM_VALIDATION_BASE_LINEAGES", 1)
        monkeypatch.setattr(module, "MINIMUM_TEST_BASE_LINEAGES", 1)
        monkeypatch.setattr(module, "MINIMUM_VALIDATION_TUNING_BASE_LINEAGES", 1)
        monkeypatch.setattr(
            module,
            "VALIDATION_TUNING_PRECISION_CONFIDENCE_LEVEL",
            0.51,
        )
        monkeypatch.setattr(
            module,
            "VALIDATION_TUNING_PRECISION_MAX_HALF_WIDTH",
            0.99,
        )


def _host_compiled_v2_identity(
    monkeypatch: pytest.MonkeyPatch,
    verifier_pin: GroundCertificateVerifierPin,
    build_record_path: Path,
) -> None:
    """Adapt only the isolation identity; request/result logic remains production code.

    Publication mode normally requires a reproducible static Linux ELF or an Apptainer
    image.  This integration test also runs on macOS, so the already-tested pin resolver
    authenticates a host executable while this shim supplies the v2 publication identity.
    The real `_run_verifier` subprocess boundary is deliberately left untouched.
    """

    original = ground_module._verifier_identity

    def resolve(
        pin: GroundCertificateVerifierPin,
        *,
        publication: bool = False,
    ) -> tuple[
        dict[str, object],
        bytes,
        tuple[tuple[Path, str], ...],
        ground_module._VerifierExecution,
    ]:
        assert publication is True
        assert pin is verifier_pin
        host_identity, executable, pinned_paths, execution = original(
            pin,
            publication=False,
        )
        build = json.loads(build_record_path.read_text(encoding="utf-8"))
        identity = {
            "build_attestation_record_digest": build["record_digest"],
            "build_attestation_sha256": _sha256(build_record_path),
            "environment_sha256": host_identity["environment_sha256"],
            "execution_mode": "static-elf",
            "executable_sha256": host_identity["executable_sha256"],
            "name": host_identity["name"],
            "protocol": GROUND_CERTIFICATE_PROTOCOL_V2,
            "runtime_sha256": None,
            "source_sha256": host_identity["source_sha256"],
            "version": host_identity["version"],
        }
        return (
            identity,
            executable,
            (*pinned_paths, (build_record_path, _sha256(build_record_path))),
            execution,
        )

    monkeypatch.setattr(ground_module, "_verifier_identity", resolve)


def test_real_embedbench_certificate_crosses_cpp_v2_and_receipt_loaders(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compiled_verifier: Path,
) -> None:
    _allow_small_authentic_fixture(monkeypatch)
    bank, bank_manifest, targets, provenance, design, design_sha256 = write_v4_inputs(
        tmp_path / "source"
    )
    instances = {
        row["record"]["instance_id"]: row["record"]
        for row in _jsonl(bank)
        if row["kind"] == "instance"
    }
    references = _export_real_references(tmp_path / "references", instances)
    certificate_artifacts = _bind_source_targets_to_references(
        targets,
        instances,
        references,
    )

    prepared = tmp_path / "prepared-v4"
    prepare_candidate_bank_v4(
        bank,
        bank_manifest,
        targets,
        provenance,
        design,
        prepared,
        expected_corpus_design_sha256=design_sha256,
        qubit_cap=4,
    )

    source_sha256 = _sha256(VERIFIER_SOURCE)
    verifier_identity = PublisherVerifierIdentity(
        name=VERIFIER_NAME,
        version=VERIFIER_VERSION,
        implementation_sha256=source_sha256,
    )
    publication = tmp_path / "publisher-publication"
    attestation_receipt = build_publisher_attestation_v2(
        prepared,
        certificate_artifacts,
        publication,
        publisher_id="embedbench-integration-publisher",
        publication_id="embedbench-integration-publication-v2",
        verifier=verifier_identity,
    )
    verified_attestation = verify_publisher_attestation_v2(
        prepared,
        publication,
        expected_attestation_record_digest=attestation_receipt.attestation_record_digest,
        expected_publisher_id=attestation_receipt.publisher_id,
        expected_verifier=verifier_identity,
    )
    quality_pin = QualityAttestationPin(
        path=verified_attestation.attestation_path,
        expected_digest=verified_attestation.attestation_record_digest,
        expected_publisher_id=verified_attestation.publisher_id,
    )

    environment = tmp_path / "host-compiler-environment.lock"
    environment.write_bytes(b"cross-runtime-test-host\n")
    build_payload = {
        "compiler_executable": "clang++-or-g++",
        "flags": ["-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror"],
        "source_sha256": source_sha256,
    }
    build_record = {**build_payload, "record_digest": content_digest(build_payload)}
    build_record_path = tmp_path / "host-build-record.json"
    build_record_path.write_bytes(canonical_json_bytes(build_record) + b"\n")
    verifier_pin = GroundCertificateVerifierPin(
        executable_path=compiled_verifier,
        expected_executable_sha256=_sha256(compiled_verifier),
        source_path=VERIFIER_SOURCE,
        expected_source_sha256=source_sha256,
        environment_path=environment,
        expected_environment_sha256=_sha256(environment),
        expected_name=VERIFIER_NAME,
        expected_version=VERIFIER_VERSION,
        execution_mode="test-only-host",
    )
    _host_compiled_v2_identity(monkeypatch, verifier_pin, build_record_path)

    ground_output = tmp_path / "ground-v2"
    root_record = verify_ground_certificate_partitions(
        prepared,
        quality_attestation_pin=quality_pin,
        verifier_pin=verifier_pin,
        output_directory=ground_output,
        timeout_seconds=10.0,
    )
    root_path = ground_output / "root.json"
    loaded_root = load_ground_certificate_root(
        root_path,
        expected_sha256=_sha256(root_path),
        corpus_directory=prepared,
        quality_attestation_pin=quality_pin,
    )
    assert loaded_root.as_dict() == root_record
    assert loaded_root.record_digest == root_record["record_digest"]
    assert root_record["census"] == {
        "accepted_count": len(instances),
        "by_partition": {partition: 1 for partition in PARTITIONS},
        "target_count": len(instances),
    }

    for partition in PARTITIONS:
        loaded = load_ground_certificate_partition(
            root_path,
            expected_root_sha256=loaded_root.sha256,
            partition=partition,
            corpus_directory=prepared,
            quality_attestation_pin=quality_pin,
        )
        receipt = loaded.as_dict()
        assert loaded.partition == partition
        assert receipt["census"]["accepted_count"] == 1
        assert receipt["census"]["target_count"] == 1
        assert receipt["targets"][0]["status"] == "accepted"
        assert receipt["targets"][0]["verifier_result"]["accepted"] is True
        assert receipt["targets"][0]["artifact_sha256"] in certificate_artifacts
