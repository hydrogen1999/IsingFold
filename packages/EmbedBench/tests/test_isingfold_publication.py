from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
from embedbench.candidate_bank import (
    InstanceRecord,
    canonical_json_bytes,
    content_digest,
)
from embedbench.ground_certificate import IsingProblem
from embedbench.isingfold_publication import publish_isingfold_source

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
ISINGFOLD_ROOT = next(
    root
    for root in (PACKAGE_ROOT.parents[1], PACKAGE_ROOT.parent / "IsingFold")
    if (root / "src" / "isingfold").is_dir()
)
STRATUM_ARTIFACTS = ("authority", "budget", "evidence", "origin", "panel", "protocol")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_canonical(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _record(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "record_digest": content_digest(payload)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _isingfold_fixture(root: Path) -> tuple[Path, Path, Path, Path]:
    if not ISINGFOLD_ROOT.is_dir():
        pytest.skip("the sibling IsingFold consumer checkout is unavailable")
    original_path = list(sys.path)
    original_modules = set(sys.modules)
    try:
        for entry in (ISINGFOLD_ROOT, ISINGFOLD_ROOT / "src"):
            if str(entry) not in sys.path:
                sys.path.insert(0, str(entry))
        support_path = ISINGFOLD_ROOT / "tests" / "unit" / "trust_v4_support.py"
        specification = importlib.util.spec_from_file_location(
            "_isingfold_publication_trust_v4_support", support_path
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
    bank_rows = _read_jsonl(bank)
    instances = [
        InstanceRecord.from_dict(row["record"]) for row in bank_rows if row["kind"] == "instance"
    ]
    original_by_instance = {row["instance_id"]: row for row in _read_jsonl(original_targets)}

    reference_rows: list[dict[str, Any]] = []
    certificate_root = root / "reference-certificates"
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
        certificate_sha256 = hashlib.sha256(certificate).hexdigest()
        certificate_path = certificate_root / certificate_sha256
        certificate_path.parent.mkdir(parents=True, exist_ok=True)
        certificate_path.write_bytes(certificate)
        target = original_by_instance[instance.instance_id]
        reference_rows.append(
            _record(
                {
                    "certificate": {
                        "path": f"reference-certificates/{certificate_sha256}",
                        "sha256": certificate_sha256,
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
    reference_rows.sort(key=lambda row: row["problem_sha256"])
    references = root / "references.jsonl"
    references.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in reference_rows))

    provenance_rows = _read_jsonl(provenance)
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
    tasks = [
        {key: row[key] for key in sorted(task_fields)}
        for row in sorted(provenance_rows, key=lambda row: row["group_id"])
    ]
    design_relative = design.relative_to(root).as_posix()
    index_payload = {
        "corpus_design": {"path": design_relative, "sha256": _sha256(design)},
        "schema": "embedbench.isingfold-publication-index",
        "schema_version": 1,
        "source_release_id": provenance_rows[0]["source_release_id"],
        "source_release_manifest_sha256": provenance_rows[0]["source_release_manifest_sha256"],
        "split_manifest_sha256": provenance_rows[0]["split_manifest_sha256"],
        "strata": {
            name: {
                "path": (design.parent / "strata" / f"{name}.json").relative_to(root).as_posix(),
                "sha256": _sha256(design.parent / "strata" / f"{name}.json"),
            }
            for name in STRATUM_ARTIFACTS
        },
        "tasks": tasks,
    }
    publication_index = root / "publication-index.json"
    _write_canonical(publication_index, _record(index_payload))
    return bank, bank_manifest, references, publication_index


def _publish(root: Path) -> tuple[Path, dict[str, Any]]:
    bank, bank_manifest, references, publication_index = _isingfold_fixture(root)
    output = root / "published"
    receipt = _invoke_publish(
        bank,
        bank_manifest,
        references,
        publication_index,
        output,
    )
    return output, receipt


def _invoke_publish(
    bank: Path,
    bank_manifest: Path,
    references: Path,
    publication_index: Path,
    output: Path,
) -> dict[str, Any]:
    return publish_isingfold_source(
        bank_path=bank,
        bank_manifest_path=bank_manifest,
        publication_index_path=publication_index,
        reference_index_path=references,
        output_dir=output,
        expected_bank_manifest_sha256=_sha256(bank_manifest),
        expected_publication_index_sha256=_sha256(publication_index),
        expected_reference_index_sha256=_sha256(references),
    )


def _rewrite_record(path: Path, value: dict[str, Any]) -> None:
    payload = {key: item for key, item in value.items() if key != "record_digest"}
    _write_canonical(path, _record(payload))


def _write_records(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_bytes(b"".join(canonical_json_bytes(row) + b"\n" for row in rows))


def test_source_publication_is_accepted_by_real_prepared_v4_importer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    published, receipt = _publish(tmp_path)

    assert receipt["schema"] == "embedbench.isingfold-source-publication"
    assert {path.name for path in published.iterdir()} == {
        "SHA256SUMS",
        "candidate_bank_v2.jsonl",
        "candidate_bank_v2.manifest.json",
        "certificates",
        "corpus_design_v2.json",
        "evaluator_targets.jsonl",
        "isingfold_task_provenance.jsonl",
        "publication_manifest.json",
        "strata",
    }
    assert {path.name for path in (published / "strata").iterdir()} == {
        f"{name}.json" for name in STRATUM_ARTIFACTS
    }

    import isingfold.rl.data.import_embedbench as consumer

    monkeypatch.setattr(consumer, "MINIMUM_TRAIN_BASE_LINEAGES", 1)
    monkeypatch.setattr(consumer, "MINIMUM_VALIDATION_BASE_LINEAGES", 1)
    monkeypatch.setattr(consumer, "MINIMUM_TEST_BASE_LINEAGES", 1)
    monkeypatch.setattr(consumer, "MINIMUM_VALIDATION_TUNING_BASE_LINEAGES", 1)
    monkeypatch.setattr(consumer, "VALIDATION_TUNING_PRECISION_CONFIDENCE_LEVEL", 0.51)
    monkeypatch.setattr(consumer, "VALIDATION_TUNING_PRECISION_MAX_HALF_WIDTH", 0.99)
    prepared = tmp_path / "prepared-v4"
    manifest = consumer.prepare_candidate_bank_v4(
        published / "candidate_bank_v2.jsonl",
        published / "candidate_bank_v2.manifest.json",
        published / "evaluator_targets.jsonl",
        published / "isingfold_task_provenance.jsonl",
        published / "corpus_design_v2.json",
        prepared,
        expected_corpus_design_sha256=_sha256(published / "corpus_design_v2.json"),
        qubit_cap=4,
    )

    assert manifest["schema_version"] == 4
    assert {path.name for path in (prepared / "targets").iterdir()} == {
        "test.jsonl",
        "train.jsonl",
        "val.jsonl",
    }
    public_bytes = (prepared / "policy_instances.jsonl").read_bytes()
    assert b"reference_energy" not in public_bytes
    assert b"certificate_digest" not in public_bytes
    targets = _read_jsonl(published / "evaluator_targets.jsonl")
    assert len(targets) == len({row["instance_id"] for row in targets}) == 3


def test_certificate_tamper_is_rejected_without_publishing(tmp_path: Path) -> None:
    bank, bank_manifest, references, publication_index = _isingfold_fixture(tmp_path)
    reference = _read_jsonl(references)[0]
    certificate = tmp_path / reference["certificate"]["path"]
    certificate.write_bytes(certificate.read_bytes() + b"tampered")
    output = tmp_path / "published"

    with pytest.raises(ValueError, match="certificate.*SHA-256"):
        _invoke_publish(bank, bank_manifest, references, publication_index, output)

    assert not output.exists()


@pytest.mark.parametrize("coverage", ["missing", "duplicate"])
def test_task_coverage_is_exact_and_unique(tmp_path: Path, coverage: str) -> None:
    bank, bank_manifest, references, publication_index = _isingfold_fixture(tmp_path)
    index = json.loads(publication_index.read_text(encoding="utf-8"))
    if coverage == "missing":
        index["tasks"].pop()
    else:
        index["tasks"].append(dict(index["tasks"][-1]))
    _rewrite_record(publication_index, index)
    output = tmp_path / "published"

    with pytest.raises(ValueError, match="repeats|coverage"):
        _invoke_publish(bank, bank_manifest, references, publication_index, output)

    assert not output.exists()


def test_duplicate_reference_coverage_is_rejected(tmp_path: Path) -> None:
    bank, bank_manifest, references, publication_index = _isingfold_fixture(tmp_path)
    rows = _read_jsonl(references)
    rows.append(dict(rows[-1]))
    _write_records(references, rows)
    output = tmp_path / "published"

    with pytest.raises(ValueError, match="repeats a logical problem"):
        _invoke_publish(bank, bank_manifest, references, publication_index, output)

    assert not output.exists()


def test_unsafe_certificate_path_is_rejected(tmp_path: Path) -> None:
    bank, bank_manifest, references, publication_index = _isingfold_fixture(tmp_path)
    rows = _read_jsonl(references)
    first = rows[0]
    first["certificate"]["path"] = "../outside-certificate"
    payload = {key: value for key, value in first.items() if key != "record_digest"}
    rows[0] = _record(payload)
    _write_records(references, rows)
    output = tmp_path / "published"

    with pytest.raises(ValueError, match="safe normalized relative POSIX path"):
        _invoke_publish(bank, bank_manifest, references, publication_index, output)

    assert not output.exists()


def test_existing_output_is_never_modified(tmp_path: Path) -> None:
    bank, bank_manifest, references, publication_index = _isingfold_fixture(tmp_path)
    output = tmp_path / "published"
    output.mkdir()
    sentinel = output / "owned-by-caller"
    sentinel.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="already exists"):
        _invoke_publish(bank, bank_manifest, references, publication_index, output)

    assert sentinel.read_bytes() == b"keep"
    assert {path.name for path in output.iterdir()} == {"owned-by-caller"}


def test_externally_pinned_bank_manifest_rejects_byte_tamper(tmp_path: Path) -> None:
    bank, bank_manifest, references, publication_index = _isingfold_fixture(tmp_path)
    original_sha256 = _sha256(bank_manifest)
    bank_manifest.write_bytes(bank_manifest.read_bytes() + b" ")
    output = tmp_path / "published"

    with pytest.raises(ValueError, match="external SHA-256 commitment"):
        publish_isingfold_source(
            bank_path=bank,
            bank_manifest_path=bank_manifest,
            publication_index_path=publication_index,
            reference_index_path=references,
            output_dir=output,
            expected_bank_manifest_sha256=original_sha256,
            expected_publication_index_sha256=_sha256(publication_index),
            expected_reference_index_sha256=_sha256(references),
        )

    assert not output.exists()


def test_boolean_schema_version_cannot_impersonate_integer_one(tmp_path: Path) -> None:
    bank, bank_manifest, references, publication_index = _isingfold_fixture(tmp_path)
    index = json.loads(publication_index.read_text(encoding="utf-8"))
    index["schema_version"] = True
    _rewrite_record(publication_index, index)
    output = tmp_path / "published"

    with pytest.raises(ValueError, match="requires schema_version 1"):
        _invoke_publish(bank, bank_manifest, references, publication_index, output)

    assert not output.exists()
