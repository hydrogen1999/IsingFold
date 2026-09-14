from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from embedbench.candidate_bank import InstanceRecord, canonical_json_bytes, content_digest
from embedbench.ground_certificate import IsingProblem
from embedbench.isingfold_attestation import (
    VerifierIdentity,
    build_publisher_attestation_v2,
)
from embedbench.isingfold_publication import publish_isingfold_source

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ISINGFOLD_ROOT = next(
    root
    for root in (PACKAGE_ROOT.parents[1], PACKAGE_ROOT.parent / "IsingFold")
    if (root / "src" / "isingfold").is_dir()
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "record_digest": content_digest(payload)}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _write_canonical(path: Path, value: dict[str, Any]) -> None:
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _isingfold_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    original_path = list(sys.path)
    original_modules = set(sys.modules)
    try:
        for entry in (ISINGFOLD_ROOT, ISINGFOLD_ROOT / "src"):
            if str(entry) not in sys.path:
                sys.path.insert(0, str(entry))
        support_path = ISINGFOLD_ROOT / "tests" / "unit" / "trust_v4_support.py"
        specification = importlib.util.spec_from_file_location(
            "_isingfold_bridge_trust_v4_support", support_path
        )
        if specification is None or specification.loader is None:
            raise RuntimeError("cannot load the sibling IsingFold trust fixture")
        support = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(support)
        bank, bank_manifest, original_targets, provenance, design, _ = support.write_v4_inputs(
            root / "authority"
        )
    finally:
        sys.path[:] = original_path
        for module_name in set(sys.modules) - original_modules:
            if module_name == "tests" or module_name.startswith("tests."):
                sys.modules.pop(module_name, None)

    instances = [
        InstanceRecord.from_dict(row["record"])
        for row in _jsonl(bank)
        if row["kind"] == "instance"
    ]
    targets = {row["instance_id"]: row for row in _jsonl(original_targets)}
    certificate_root = root / "reference-certificates"
    references = []
    for instance in instances:
        problem = IsingProblem(
            variables=instance.logical_nodes,
            linear=instance.h,
            quadratic=instance.j,
        )
        certificate = canonical_json_bytes(
            {
                "problem_sha256": problem.problem_sha256,
                "schema": "embedbench.fixture-ground-certificate",
                "schema_version": 1,
            }
        )
        certificate_digest = hashlib.sha256(certificate).hexdigest()
        certificate_root.mkdir(parents=True, exist_ok=True)
        (certificate_root / certificate_digest).write_bytes(certificate)
        target = targets[instance.instance_id]
        references.append(
            _record(
                {
                    "certificate": {
                        "path": f"reference-certificates/{certificate_digest}",
                        "sha256": certificate_digest,
                    },
                    "evaluator_protocol_digest": target["evaluator_protocol_digest"],
                    "problem_sha256": problem.problem_sha256,
                    "reference_energy": target["reference_energy"],
                    "reference_status": target["reference_status"],
                    "schema": "embedbench.isingfold-reference",
                    "schema_version": 1,
                }
            )
        )
    reference_index = root / "references.jsonl"
    reference_index.write_bytes(
        b"".join(
            canonical_json_bytes(row) + b"\n"
            for row in sorted(references, key=lambda row: row["problem_sha256"])
        )
    )

    provenance_rows = _jsonl(provenance)
    task_fields = {
        "active_topology_identity",
        "base_parent_lineage",
        "calibration_identity",
        "descendant_transform_identity",
        "distribution",
        "fault_identity",
        "group_id",
        "nominal_topology_identity",
    }
    index_payload = {
        "corpus_design": {
            "path": design.relative_to(root).as_posix(),
            "sha256": _sha256(design),
        },
        "schema": "embedbench.isingfold-publication-index",
        "schema_version": 1,
        "source_release_id": provenance_rows[0]["source_release_id"],
        "source_release_manifest_sha256": provenance_rows[0][
            "source_release_manifest_sha256"
        ],
        "split_manifest_sha256": provenance_rows[0]["split_manifest_sha256"],
        "strata": {
            name: {
                "path": (design.parent / "strata" / f"{name}.json")
                .relative_to(root)
                .as_posix(),
                "sha256": _sha256(design.parent / "strata" / f"{name}.json"),
            }
            for name in ("authority", "budget", "evidence", "origin", "panel", "protocol")
        },
        "tasks": [
            {key: row[key] for key in sorted(task_fields)}
            for row in sorted(provenance_rows, key=lambda row: row["group_id"])
        ],
    }
    publication_index = root / "publication-index.json"
    _write_canonical(publication_index, _record(index_payload))
    return bank, bank_manifest, reference_index, publication_index


def test_complete_independent_publication_bridge_is_accepted_by_isingfold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank, bank_manifest, references, publication_index = _isingfold_fixture(tmp_path)
    source = tmp_path / "source-publication"
    publish_isingfold_source(
        bank_path=bank,
        bank_manifest_path=bank_manifest,
        publication_index_path=publication_index,
        reference_index_path=references,
        output_dir=source,
        expected_bank_manifest_sha256=_sha256(bank_manifest),
        expected_publication_index_sha256=_sha256(publication_index),
        expected_reference_index_sha256=_sha256(references),
    )

    for entry in (ISINGFOLD_ROOT, ISINGFOLD_ROOT / "src"):
        monkeypatch.syspath_prepend(str(entry))
    import isingfold.rl.data.import_embedbench as consumer

    monkeypatch.setattr(consumer, "MINIMUM_TRAIN_BASE_LINEAGES", 1)
    monkeypatch.setattr(consumer, "MINIMUM_VALIDATION_BASE_LINEAGES", 1)
    monkeypatch.setattr(consumer, "MINIMUM_TEST_BASE_LINEAGES", 1)
    monkeypatch.setattr(consumer, "MINIMUM_VALIDATION_TUNING_BASE_LINEAGES", 1)
    monkeypatch.setattr(consumer, "VALIDATION_TUNING_PRECISION_CONFIDENCE_LEVEL", 0.51)
    monkeypatch.setattr(consumer, "VALIDATION_TUNING_PRECISION_MAX_HALF_WIDTH", 0.99)
    prepared = tmp_path / "prepared-v4"
    consumer.prepare_candidate_bank_v4(
        source / "candidate_bank_v2.jsonl",
        source / "candidate_bank_v2.manifest.json",
        source / "evaluator_targets.jsonl",
        source / "isingfold_task_provenance.jsonl",
        source / "corpus_design_v2.json",
        prepared,
        expected_corpus_design_sha256=_sha256(source / "corpus_design_v2.json"),
        qubit_cap=4,
    )

    publication_manifest = json.loads((source / "publication_manifest.json").read_text())
    certificates = {
        row["sha256"]: source / row["path"]
        for row in publication_manifest["certificates"]
    }
    verifier = VerifierIdentity(
        name="embedbench-independent-ground-verifier",
        version="1.0.0",
        implementation_sha256=hashlib.sha256(b"fixture-verifier-source").hexdigest(),
    )
    attestation_root = tmp_path / "quality-attestation"
    attestation = build_publisher_attestation_v2(
        prepared,
        certificates,
        attestation_root,
        publisher_id="embedbench-release-authority",
        publication_id="embedbench-isingfold-smoke-v1",
        verifier=verifier,
    )

    import isingfold.rl.data.prepared as prepared_loader
    from isingfold.rl.data.prepared import load_prepared_partition
    from isingfold.rl.data.quality_attestation import QualityAttestationPin

    monkeypatch.setattr(prepared_loader, "MINIMUM_TRAIN_BASE_LINEAGES", 1)
    monkeypatch.setattr(prepared_loader, "MINIMUM_VALIDATION_BASE_LINEAGES", 1)
    monkeypatch.setattr(prepared_loader, "MINIMUM_TEST_BASE_LINEAGES", 1)
    monkeypatch.setattr(prepared_loader, "MINIMUM_VALIDATION_TUNING_BASE_LINEAGES", 1)
    monkeypatch.setattr(
        prepared_loader, "VALIDATION_TUNING_PRECISION_CONFIDENCE_LEVEL", 0.51
    )
    monkeypatch.setattr(prepared_loader, "VALIDATION_TUNING_PRECISION_MAX_HALF_WIDTH", 0.99)

    pin = QualityAttestationPin(
        path=attestation.attestation_path,
        expected_digest=attestation.attestation_record_digest,
        expected_publisher_id=attestation.publisher_id,
    )
    for partition in ("train", "val", "test"):
        public = load_prepared_partition(
            prepared,
            partition=partition,
            include_evaluator=False,
        )
        trusted = load_prepared_partition(
            prepared,
            partition=partition,
            include_evaluator=True,
            quality_attestation_pin=pin,
        )
        assert public.target_access is None
        assert len(public.tasks) == len(trusted.tasks) == 1
        assert trusted.target_access is not None
        assert trusted.target_access.target_count == 1
        assert trusted.tasks[0].quality_attestation_digest == (
            attestation.attestation_record_digest
        )
