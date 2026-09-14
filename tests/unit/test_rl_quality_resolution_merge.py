from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import isingfold.rl.data.quality_resolution_plan as plan_module
import isingfold.rl.data.quality_resolution_delta as delta_module
import isingfold.rl.data.quality as quality_module
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.quality import continuation_seed
from isingfold.rl.data.quality_resolution import QualityResolutionError
from isingfold.rl.data.quality_resolution_delta import (
    QUALITY_RESOLUTION_DELTA_ROW_SCHEMA,
    QUALITY_RESOLUTION_DELTA_ROW_VERSION,
    QUALITY_RESOLUTION_VERIFICATION_SCHEMA,
    QUALITY_RESOLUTION_VERIFICATION_VERSION,
)
from isingfold.rl.data.quality_resolution_merge import (
    QUALITY_RESOLUTION_STUDY_SCHEMA,
    QualityResolutionShardPin,
    load_quality_resolution_study_receipt,
    merge_quality_resolution,
    publish_quality_resolution_shard_pin_registry,
)
from isingfold.rl.data.quality_resolution_plan import (
    PlannedResolutionAction,
    QualityResolutionPlan,
    ResolutionProductionPlan,
    build_quality_resolution_plan,
)
from tests.unit.test_rl_quality_resolution_delta import _plan_authority
from tests.unit.test_rl_quality_resolution_plan import _planning_inputs, _production


QUALITY_CONTRACT = {"contract": "quality-v7-fixture"}
QUALITY_CONTRACT_DIGEST = content_digest(QUALITY_CONTRACT)


def _record(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "record_digest": content_digest(payload)}


TARGET_ACCESS = _record(
    {
        "partition": "train",
        "schema": "fixture-target-access",
        "schema_version": 1,
    }
)
GROUND_PARTITION = _record(
    {
        "partition": "train",
        "schema": "fixture-ground-partition",
        "schema_version": 1,
    }
)
QUALITY_AUTHORITY = _record(
    {
        "name": "fixture-quality-authority",
        "schema": "fixture-quality-authority",
        "schema_version": 1,
    }
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_sha(module: object) -> str:
    return hashlib.sha256(
        Path(str(module.__file__)).resolve().read_bytes()  # type: ignore[union-attr]
    ).hexdigest()


def _plan_sha(plan: QualityResolutionPlan) -> str:
    return hashlib.sha256(canonical_json_bytes(plan.as_dict()) + b"\n").hexdigest()


def _two_action_plan() -> QualityResolutionPlan:
    production = _production()
    rows = []
    for row in production.rows:
        second = PlannedResolutionAction(
            action_index=1,
            payload_key=f"{row.task_id}-second",
            opcode="STOP",
            selected_payload_digest=hashlib.sha256(f"payload:{row.task_id}:1".encode()).hexdigest(),
            applied_action_record_digest=None,
            continuation_seeds=tuple(
                continuation_seed(
                    row.environment_seed,
                    row.task_id,
                    row.state_fingerprint,
                    1,
                    offset,
                )
                for offset in range(128)
            ),
        )
        rows.append(replace(row, actions=(*row.actions, second)))
    implementation = {
        "planner": "quality-resolution-plan-v1",
        "quality_implementation_contract_digest": QUALITY_CONTRACT_DIGEST,
        "quality_module_sha256": _module_sha(quality_module),
        "quality_resolution_delta_module_sha256": _module_sha(delta_module),
    }
    authority = replace(
        _plan_authority(),
        implementation=implementation,
        implementation_digest=content_digest(implementation),
    )
    return build_quality_resolution_plan(
        _planning_inputs(),
        authority,
        ResolutionProductionPlan(
            lineages=production.lineages,
            rows=tuple(rows),
            source_census=production.source_census,
            initializer_bank_contract=production.initializer_bank_contract,
        ),
    )


class _ManifestAuthority:
    def __init__(self, plan: QualityResolutionPlan) -> None:
        plan_record = plan.as_dict()
        self.execution_identity = {
            "delta_module_sha256": _module_sha(delta_module),
            "device": plan_record["selector"]["device"],
            "quality_implementation_contract_digest": QUALITY_CONTRACT_DIGEST,
            "quality_module_sha256": _module_sha(quality_module),
            "runtime_sha256": "2" * 64,
            "selector_digest": plan_record["selector"]["selector_digest"],
            "threads": 1,
        }
        self.execution_identity_digest = content_digest(self.execution_identity)

    def identity_fields(self) -> dict[str, object]:
        return {
            "execution_identity": self.execution_identity,
            "execution_identity_digest": self.execution_identity_digest,
            "ground_partition_receipt_record_digest": GROUND_PARTITION["record_digest"],
            "ground_partition_receipt_sha256": "4" * 64,
            "quality_authority_record_digest": QUALITY_AUTHORITY["record_digest"],
            "target_access_record_digest": TARGET_ACCESS["record_digest"],
        }


def _write_stage_zero_sources(
    root: Path, plan: QualityResolutionPlan
) -> tuple[QualityResolutionShardPin, ...]:
    # These are sealed phase-2-format artifacts.  Constructing them directly keeps this
    # target-free phase-3 test independent from target-bearing continuation execution.
    import isingfold.rl.data.quality_resolution_delta as delta_module

    plan_record = plan.as_dict()
    plan_sha = _plan_sha(plan)
    authority = _ManifestAuthority(plan)
    pins: list[QualityResolutionShardPin] = []
    for shard_index in range(64):
        stage, shard, work = delta_module._delta_work(plan_record, 0, shard_index)
        records: list[dict[str, object]] = []
        for item in work:
            payload: dict[str, object] = {
                "action_index": item.action["action_index"],
                "action_payload_digest": item.action["selected_payload_digest"],
                "base_lineage": item.row["base_lineage"],
                "compiled_program_digests": [
                    hashlib.sha256(f"program:{item.key}:{index}".encode()).hexdigest()
                    for index in range(4)
                ],
                "continuation_index": item.continuation_index,
                "continuation_receipt_digest": hashlib.sha256(
                    f"receipt:{item.key}".encode()
                ).hexdigest(),
                "continuation_seed": item.continuation_seed,
                "execution_identity_digest": authority.execution_identity_digest,
                "instance_id": item.row["instance_id"],
                "initializer_binding_record_digest": hashlib.sha256(
                    f"initializer-binding:{item.key}".encode()
                ).hexdigest(),
                "plan_record_digest": plan_record["record_digest"],
                "plan_sha256": plan_sha,
                "returned_valid": True,
                "reward": 1.0 if item.action["action_index"] == 0 else 0.0,
                "row_id": item.row["row_id"],
                "schema": QUALITY_RESOLUTION_DELTA_ROW_SCHEMA,
                "schema_version": QUALITY_RESOLUTION_DELTA_ROW_VERSION,
                "selected_embedding_digest": hashlib.sha256(
                    f"embedding:{item.key}".encode()
                ).hexdigest(),
                "selected_strength_index": 0,
                "shard_index": shard_index,
                "stage_index": 0,
                "state_fingerprint": item.row["state_fingerprint"],
                "task_id": item.row["task_id"],
            }
            records.append(_record(payload))
        records_raw = b"".join(canonical_json_bytes(row) + b"\n" for row in records)
        manifest = delta_module._manifest(
            records,
            records_raw,
            plan_sha256=plan_sha,
            plan_record_digest=plan_record["record_digest"],
            stage=stage,
            shard=shard,
            context_digest=plan_record["context"]["digest"],
            authority=authority,
            initializer_bank_contract=plan_record["production_plan"]["initializer_bank"],
        )
        delta_root = root / f"delta-{shard_index:02d}"
        delta_root.mkdir()
        (delta_root / "records.jsonl").write_bytes(records_raw)
        (delta_root / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")

        verifier_identity = _record(
            {
                "attestor_id": "fixture-independent-verifier",
                "continuation_runner": "isingfold.rl.data.quality.run_continuation",
                "quality_implementation_contract_digest": QUALITY_CONTRACT_DIGEST,
                "quality_module_sha256": _module_sha(quality_module),
                "schema": delta_module.QUALITY_RESOLUTION_VERIFIER_IDENTITY_SCHEMA,
                "schema_version": delta_module.QUALITY_RESOLUTION_VERIFIER_IDENTITY_VERSION,
                "verification_module_sha256": _module_sha(delta_module),
                "verification_runtime_sha256": "7" * 64,
            }
        )
        verification = _record(
            {
                "delta_manifest_record_digest": manifest["record_digest"],
                "delta_manifest_sha256": _sha(delta_root / "manifest.json"),
                "execution_identity_digest": authority.execution_identity_digest,
                "matching_count": len(records),
                "mismatches": [],
                "pass": True,
                "plan_record_digest": plan_record["record_digest"],
                "plan_sha256": plan_sha,
                "replayed_count": len(records),
                "schema": QUALITY_RESOLUTION_VERIFICATION_SCHEMA,
                "schema_version": QUALITY_RESOLUTION_VERIFICATION_VERSION,
                "shard_index": shard_index,
                "stage_index": 0,
                "verifier_identity": verifier_identity,
                "verifier_identity_digest": verifier_identity["record_digest"],
                "verifier_identity_sha256": hashlib.sha256(
                    canonical_json_bytes(verifier_identity) + b"\n"
                ).hexdigest(),
            }
        )
        verification_path = root / f"verification-{shard_index:02d}.json"
        verification_path.write_bytes(canonical_json_bytes(verification) + b"\n")
        pins.append(
            QualityResolutionShardPin(
                stage_index=0,
                shard_index=shard_index,
                delta_root=delta_root.name,
                delta_manifest_sha256=_sha(delta_root / "manifest.json"),
                verification_path=verification_path.name,
                verification_sha256=_sha(verification_path),
            )
        )
    return tuple(pins)


@pytest.fixture(autouse=True)
def _small_population(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plan_module, "MINIMUM_PRODUCTION_LINEAGES", 128)


@pytest.fixture()
def merged_fixture(tmp_path: Path) -> dict[str, Any]:
    plan = _two_action_plan()
    pins = _write_stage_zero_sources(tmp_path, plan)
    registry_path = tmp_path / "pins.json"
    registry_sha = publish_quality_resolution_shard_pin_registry(
        registry_path,
        plan=plan,
        expected_plan_sha256=_plan_sha(plan),
        completed_stage_index=0,
        pins=pins,
    )
    receipt_path = tmp_path / "study.json"
    receipt_sha = merge_quality_resolution(
        plan,
        expected_plan_sha256=_plan_sha(plan),
        pin_registry_path=registry_path,
        expected_pin_registry_sha256=registry_sha,
        output_path=receipt_path,
    )
    return {
        "plan": plan,
        "pins": pins,
        "registry_path": registry_path,
        "registry_sha": registry_sha,
        "receipt_path": receipt_path,
        "receipt_sha": receipt_sha,
    }


def test_merge_authenticates_complete_cumulative_prefix_and_selects_smallest_count(
    merged_fixture: dict[str, Any],
) -> None:
    plan = merged_fixture["plan"]
    receipt = load_quality_resolution_study_receipt(
        merged_fixture["receipt_path"],
        expected_receipt_sha256=merged_fixture["receipt_sha"],
        plan=plan,
        expected_plan_sha256=_plan_sha(plan),
    ).as_dict()

    assert receipt["schema"] == QUALITY_RESOLUTION_STUDY_SCHEMA
    assert receipt["completed_continuation_range"] == [0, 12]
    assert len(receipt["source_shards"]) == 64
    assert receipt["results"] == [
        {
            "continuations": 12,
            "lower_population_bound": 128,
            "passes": True,
            "sampled_resolved_lineages": 128,
            "sampled_resolved_rows": 128,
            "sampled_unresolved_lineages": 0,
        }
    ]
    assert receipt["selected_continuations"] == 12
    assert receipt["terminal"] is True
    assert receipt["advance"] is True
    assert receipt["production_protocol"]["requested_lineages"] == 0
    assert receipt["production_protocol"]["continuations"] == 12
    serialized = canonical_json_bytes(receipt).decode()
    assert "ground_energy" not in serialized
    assert "continuation_rewards" not in serialized


def test_registry_and_study_are_immutable_and_require_external_raw_pins(
    merged_fixture: dict[str, Any],
) -> None:
    plan = merged_fixture["plan"]
    with pytest.raises(FileExistsError):
        publish_quality_resolution_shard_pin_registry(
            merged_fixture["registry_path"],
            plan=plan,
            expected_plan_sha256=_plan_sha(plan),
            completed_stage_index=0,
            pins=merged_fixture["pins"],
        )
    with pytest.raises(FileExistsError):
        merge_quality_resolution(
            plan,
            expected_plan_sha256=_plan_sha(plan),
            pin_registry_path=merged_fixture["registry_path"],
            expected_pin_registry_sha256=merged_fixture["registry_sha"],
            output_path=merged_fixture["receipt_path"],
        )
    with pytest.raises(QualityResolutionError, match="out-of-band pin"):
        load_quality_resolution_study_receipt(
            merged_fixture["receipt_path"],
            expected_receipt_sha256="f" * 64,
            plan=plan,
            expected_plan_sha256=_plan_sha(plan),
        )


def test_registry_rejects_missing_shard_and_merge_rejects_unpinned_verification(
    tmp_path: Path,
) -> None:
    plan = _two_action_plan()
    pins = _write_stage_zero_sources(tmp_path, plan)
    with pytest.raises(QualityResolutionError, match="complete stage/shard census"):
        publish_quality_resolution_shard_pin_registry(
            tmp_path / "missing.json",
            plan=plan,
            expected_plan_sha256=_plan_sha(plan),
            completed_stage_index=0,
            pins=pins[:-1],
        )

    registry_path = tmp_path / "pins.json"
    bad = replace(pins[0], verification_sha256="f" * 64)
    registry_sha = publish_quality_resolution_shard_pin_registry(
        registry_path,
        plan=plan,
        expected_plan_sha256=_plan_sha(plan),
        completed_stage_index=0,
        pins=(bad, *pins[1:]),
    )
    with pytest.raises(QualityResolutionError, match="verification.*out-of-band pin"):
        merge_quality_resolution(
            plan,
            expected_plan_sha256=_plan_sha(plan),
            pin_registry_path=registry_path,
            expected_pin_registry_sha256=registry_sha,
            output_path=tmp_path / "must-not-exist.json",
        )
    assert not (tmp_path / "must-not-exist.json").exists()
