from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

import isingfold.rl.data.import_embedbench as import_module
import isingfold.rl.data.prepared as prepared_module
from isingfold.rl.data.import_embedbench import (
    canonical_json_bytes,
    content_digest,
    prepare_candidate_bank_v4,
)
from isingfold.rl.data.prepared import (
    load_prepared_partition,
    load_prepared_partition_access,
)
from isingfold.rl.data.prepared import load_prepared_tasks
from tests.unit.quality_attestation_support import attest_prepared_v4
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


def _prepare_v4(tmp_path: Path) -> tuple[Path, tuple[Path, ...]]:
    bank, bank_manifest, targets, provenance, design, design_sha = write_v4_inputs(
        tmp_path / "source"
    )
    output = tmp_path / "prepared-v4"
    prepare_candidate_bank_v4(
        bank,
        bank_manifest,
        targets,
        provenance,
        design,
        output,
        expected_corpus_design_sha256=design_sha,
        qubit_cap=4,
    )
    sidecars = tuple(output / "targets" / f"{partition}.jsonl" for partition in ("train", "val", "test"))
    return output, sidecars


def test_v4_publishes_three_partition_target_sidecars_and_public_commitment(
    tmp_path: Path,
) -> None:
    output, sidecars = _prepare_v4(tmp_path)
    manifest = json.loads((output / "manifest.json").read_text())

    assert manifest["schema_version"] == 4
    assert manifest["corpus_scope"] == "production-designed-v4"
    assert set(manifest["target_authority"]["partitions"]) == {"train", "val", "test"}
    assert not (output / "evaluator_targets.jsonl").exists()
    for partition, sidecar in zip(("train", "val", "test"), sidecars, strict=True):
        assert sidecar.is_file()
        descriptor = manifest["target_authority"]["partitions"][partition]
        assert descriptor["path"] == f"targets/{partition}.jsonl"
        assert descriptor["sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
        rows = [json.loads(line) for line in sidecar.read_text().splitlines()]
        assert rows and {row["learning_partition"] for row in rows} == {partition}


def test_v4_rejects_relabelled_difficulty_even_when_design_is_resigned(tmp_path: Path) -> None:
    bank, bank_manifest, targets, provenance, design, _ = write_v4_inputs(tmp_path / "source")
    record = json.loads(design.read_text())
    record["lineage_registry"][0]["decision_difficulty"] = (
        "hard" if record["lineage_registry"][0]["decision_difficulty"] == "easy" else "easy"
    )
    payload = {key: value for key, value in record.items() if key != "record_digest"}
    record["record_digest"] = content_digest(payload)
    design.write_bytes(canonical_json_bytes(record) + b"\n")

    with pytest.raises(ValueError, match="derived difficulty"):
        prepare_candidate_bank_v4(
            bank,
            bank_manifest,
            targets,
            provenance,
            design,
            tmp_path / "out",
            expected_corpus_design_sha256=hashlib.sha256(design.read_bytes()).hexdigest(),
            qubit_cap=4,
        )


def test_v4_rejects_relabelled_origin_even_when_design_is_resigned(tmp_path: Path) -> None:
    bank, bank_manifest, targets, provenance, design, _ = write_v4_inputs(tmp_path / "source")
    record = json.loads(design.read_text())
    record["lineage_registry"][0]["problem_origin"] = "application-derived"
    payload = {key: value for key, value in record.items() if key != "record_digest"}
    record["record_digest"] = content_digest(payload)
    design.write_bytes(canonical_json_bytes(record) + b"\n")

    with pytest.raises(ValueError, match="derived problem origin"):
        prepare_candidate_bank_v4(
            bank,
            bank_manifest,
            targets,
            provenance,
            design,
            tmp_path / "out",
            expected_corpus_design_sha256=hashlib.sha256(design.read_bytes()).hexdigest(),
            qubit_cap=4,
        )


def test_v4_evaluator_loader_requires_an_explicit_partition(tmp_path: Path) -> None:
    output, _ = _prepare_v4(tmp_path)

    with pytest.raises(ValueError, match="explicit partition"):
        load_prepared_partition(output, partition=None, include_evaluator=True)


def test_train_target_access_never_opens_val_or_test_sidecars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output, _ = _prepare_v4(tmp_path)
    pin = attest_prepared_v4(output, tmp_path / "trust")
    forbidden = {
        (output / "targets" / "val.jsonl").resolve(),
        (output / "targets" / "test.jsonl").resolve(),
        (tmp_path / "trust" / "quality-evidence" / "val.json").resolve(),
        (tmp_path / "trust" / "quality-evidence" / "test.json").resolve(),
    }
    original = Path.read_bytes

    def guarded_read(path: Path) -> bytes:
        if path.resolve() in forbidden:
            raise AssertionError(f"sealed sidecar was opened: {path}")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded_read)
    loaded = load_prepared_partition(
        output,
        partition="train",
        include_evaluator=True,
        quality_attestation_pin=pin,
    )

    assert loaded.tasks and {task.partition for task in loaded.tasks} == {"train"}
    assert loaded.target_access is not None
    assert loaded.target_access.partition == "train"
    assert loaded.target_access.target_path == "targets/train.jsonl"
    assert all("/test." not in row.relative_path for row in loaded.target_access.opened_files)
    assert all("/val." not in row.relative_path for row in loaded.target_access.opened_files)
    assert all(task.task.ground_energy == -1.25 for task in loaded.tasks)

    access = loaded.target_access.as_dict()
    assert access["record_digest"] == loaded.target_access.recompute_record_digest()
    assert access["partition"] == "train"


def test_cli_facing_partition_access_returns_validated_canonical_record(
    tmp_path: Path,
) -> None:
    output, _ = _prepare_v4(tmp_path)
    pin = attest_prepared_v4(output, tmp_path / "trust")

    tasks, access = load_prepared_partition_access(
        output,
        partition="test",
        quality_attestation_pin=pin,
    )

    assert tasks and {task.partition for task in tasks} == {"test"}
    assert access["partition"] == "test"
    assert access["target_path"] == "targets/test.jsonl"
    assert access["record_digest"] == content_digest(
        {key: value for key, value in access.items() if key != "record_digest"}
    )


def test_target_access_as_dict_rejects_forged_dataclass(tmp_path: Path) -> None:
    output, _ = _prepare_v4(tmp_path)
    pin = attest_prepared_v4(output, tmp_path / "trust")
    loaded = load_prepared_partition(
        output,
        partition="train",
        include_evaluator=True,
        quality_attestation_pin=pin,
    )
    assert loaded.target_access is not None

    forged = replace(loaded.target_access, target_count=loaded.target_access.target_count + 1)
    with pytest.raises(ValueError, match="record digest mismatch"):
        forged.as_dict()


def test_legacy_loader_cannot_discard_a_v4_target_access_receipt(tmp_path: Path) -> None:
    output, _ = _prepare_v4(tmp_path)
    pin = attest_prepared_v4(output, tmp_path / "trust")

    with pytest.raises(ValueError, match="access receipt cannot be discarded"):
        load_prepared_tasks(
            output,
            partition="train",
            include_evaluator=True,
            quality_attestation_pin=pin,
        )


def test_partition_field_prevents_rehashed_cross_partition_target_splice(
    tmp_path: Path,
) -> None:
    output, _ = _prepare_v4(tmp_path)
    target_path = output / "targets" / "train.jsonl"
    rows = [json.loads(line) for line in target_path.read_text().splitlines()]
    rows[0]["learning_partition"] = "test"
    payload = {key: value for key, value in rows[0].items() if key != "record_digest"}
    rows[0]["record_digest"] = content_digest(payload)
    raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    target_path.write_bytes(raw)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    descriptor = manifest["target_authority"]["partitions"]["train"]
    descriptor["sha256"] = hashlib.sha256(raw).hexdigest()
    descriptor["target_set_digest"] = content_digest(
        {
            "domain": "isingfold-partition-target-set-v1",
            "partition": "train",
            "targets": [
                {
                    "instance_id": row["instance_id"],
                    "target_record_digest": row["record_digest"],
                }
                for row in rows
            ],
        }
    )
    manifest["outputs"]["targets/train.jsonl"]["sha256"] = descriptor["sha256"]
    authority_payload = {
        key: value
        for key, value in manifest["target_authority"].items()
        if key != "record_digest"
    }
    manifest["target_authority"]["record_digest"] = content_digest(authority_payload)
    manifest_payload = {key: value for key, value in manifest.items() if key != "record_digest"}
    manifest["record_digest"] = content_digest(manifest_payload)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    pin = attest_prepared_v4(output, tmp_path / "trust")

    with pytest.raises(ValueError, match="crosses its committed partition"):
        load_prepared_partition(
            output,
            partition="train",
            include_evaluator=True,
            quality_attestation_pin=pin,
        )
