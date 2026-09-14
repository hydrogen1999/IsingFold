from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import isingfold.rl.data.quality as quality_module
import isingfold.rl.data.quality_capacity as capacity_module
import isingfold.rl.data.quality_resolution_delta as delta_module
import isingfold.rl.data.quality_resolution_plan as plan_module
from isingfold.rl.contracts import Context
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.prepared import PreparedProvenance, PreparedTask
from isingfold.rl.data.quality_attestation import quality_target_set_digest
from isingfold.rl.data.quality_resolution import QualityResolutionError
from isingfold.rl.data.quality_resolution_delta import QualityResolutionExecutionAuthority
from isingfold.rl.data.quality_resolution_merge import QualityResolutionStudyReceipt
from isingfold.rl.data.quality_resolution_plan import (
    QualityResolutionPlan,
    materialize_resolution_production_plan,
)
from isingfold.rl.data.quality_capacity import (
    QUALITY_CAPACITY_CANARY_SCHEMA,
    QUALITY_CAPACITY_CANARY_SELECTION_SCHEMA,
    QUALITY_CAPACITY_MEASUREMENT_PROTOCOL,
    QualityCapacityBudget,
    build_quality_capacity_canary_selection,
    compute_guarded_capacity_projection,
    load_quality_capacity_canary,
    load_quality_capacity_canary_selection,
    load_quality_capacity_budget,
    publish_quality_capacity_budget,
    require_passing_quality_capacity_canary,
    run_quality_capacity_canary,
)
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.initializer_bank import (
    build_initializer_bank_plan,
    generate_initializer_bank,
    load_initializer_bank,
    seal_initializer_bank,
)
from tests.unit.test_rl_quality_resolution_delta import (
    _execution_authority,
    _plan_authority,
)
from tests.unit.test_rl_quality_resolution_merge import (
    QUALITY_CONTRACT_DIGEST,
    _plan_sha,
    _two_action_plan,
)
from tests.unit.test_rl_quality_resolution_plan import (
    _planning_inputs,
    _public_prepared,
)
from tests.unit.test_rl_initializer_bank import (
    _FakeLACInitializer,
    _config as _initializer_config,
    _result as _initializer_result,
)


def _record(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "record_digest": content_digest(payload)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _module_sha(module: Any) -> str:
    return _sha(Path(module.__file__).resolve())


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value  # type: ignore[union-attr]
    return value


def _provenance(item: PreparedTask, index: int) -> PreparedProvenance:
    def digest(label: str) -> str:
        return content_digest({"index": index, "label": label})

    return PreparedProvenance(
        base_parent_lineage=str(item.task.lineage),
        source_logical_lineage=f"source-{index:04d}",
        descendant_transform_kinds=("identity",),
        descendant_transform_sha256=digest("transform"),
        nominal_topology="path",
        nominal_size=item.task.host.number_of_nodes(),
        pristine_host_sha256=digest("pristine-host"),
        active_topology="path",
        active_host_sha256=digest("active-host"),
        host_artifact_sha256=digest("host-artifact"),
        fault_status="none",
        fault_mask_sha256=digest("fault-mask"),
        calibration_status="not_applicable",
        calibration_sha256=None,
        distribution_regime="iid",
        distribution_stratum="capacity-fixture",
        source_partition="train",
        source_release_id="capacity-fixture-release",
        source_release_manifest_sha256=digest("release"),
        split_manifest_sha256=digest("split"),
        group_id=f"group-{index:04d}",
        group_record_digest=digest("group"),
        instance_record_digest=digest("instance"),
        source_record_digest=digest("source"),
        record_digest=digest("provenance"),
    )


def _real_projection_plan(tmp_path: Path):
    context = Context(qubit_cap=4, n_est_reads=256, num_sweeps=1)
    public = [
        replace(item, provenance=_provenance(item, index))
        for index in range(16)
        for item in [_public_prepared(index)]
    ]
    config = replace(
        _planning_inputs().config,
        evaluated_actions=2,
        states_per_lineage_cap=1,
    )
    loader_backend = _FakeLACInitializer([])
    bank_plan = build_initializer_bank_plan(
        public,
        prepared_manifest_sha256="2" * 64,
        training_seed=1103,
        episode_schedule_start=0,
        episode_count=len(public),
        max_draws_per_conditional_episode=1,
        initializer=loader_backend,
        runtime_implementation_manifest=loader_backend.runtime_implementation_manifest,
        config=_initializer_config(),
        context=context,
        allow_test_backend=True,
    )
    generation_backend = _FakeLACInitializer(
        [
            _initializer_result(
                {node: frozenset(chain) for node, chain in public[0].task.initial_embedding.items()}
            )
            for _ in range(6 * len(public))
        ]
    )
    bank_root = tmp_path / "capacity-initializer-bank"
    generate_initializer_bank(
        bank_root,
        bank_plan,
        public,
        initializer=generation_backend,
        runtime_implementation_manifest=generation_backend.runtime_implementation_manifest,
        config=_initializer_config(),
        context=context,
        allow_test_backend=True,
    )
    bank_sha256 = seal_initializer_bank(bank_root, bank_plan)
    bank = load_initializer_bank(
        bank_root,
        expected_manifest_sha256=bank_sha256,
        prepared_tasks=public,
        prepared_manifest_sha256="2" * 64,
        initializer=loader_backend,
        runtime_implementation_manifest=loader_backend.runtime_implementation_manifest,
        config=_initializer_config(),
        context=context,
        allow_test_backend=True,
    )
    production = materialize_resolution_production_plan(
        public,
        context=context,
        selector=fixed_strength_selector(),
        config=config,
        source_corpus_manifest_sha256="2" * 64,
        initializer_bank=bank,
        expected_initializer_bank_manifest_sha256=bank_sha256,
        allow_test_initializer_bank=True,
    )
    old_minimum = plan_module.MINIMUM_PRODUCTION_LINEAGES
    plan_module.MINIMUM_PRODUCTION_LINEAGES = 128
    try:
        template = _two_action_plan().as_dict()
    finally:
        plan_module.MINIMUM_PRODUCTION_LINEAGES = old_minimum
    template["production_plan"] = production.as_dict()
    template["production_plan_digest"] = production.record_digest
    template["prepared_corpus"]["manifest_sha256"] = "2" * 64
    context_snapshot = _jsonable(context)
    assert isinstance(context_snapshot, dict)
    template["context"] = {
        "digest": content_digest(context_snapshot),
        "snapshot": context_snapshot,
    }
    template["selector"] = _plan_authority().sections()["selector"]
    template["config"]["registered"]["evaluated_actions"] = 2
    template["config"]["registered"]["states_per_lineage_cap"] = 1
    template["implementation"]["registry"]["quality_module_sha256"] = _module_sha(quality_module)
    template["implementation"]["digest"] = content_digest(template["implementation"]["registry"])
    body = {key: value for key, value in template.items() if key != "record_digest"}
    template["record_digest"] = content_digest(body)
    return QualityResolutionPlan(template), public, context, bank, bank_sha256


def _capacity_authority(
    plan: QualityResolutionPlan,
    target_rows: list[dict[str, object]],
    public: list[PreparedTask],
) -> QualityResolutionExecutionAuthority:
    target_by_instance = {str(row["instance_id"]): row for row in target_rows}
    authority_tasks = []
    for item in public:
        target = target_by_instance[item.instance_id]
        authority_tasks.append(
            replace(
                item,
                task=replace(
                    item.task,
                    ground_energy=float(target["reference_energy"]),
                ),
                reference_status=str(target["reference_status"]),
                certificate_digest=str(target["certificate_digest"]),
                evaluator_protocol_digest=str(target["evaluator_protocol_digest"]),
            )
        )
    base = _execution_authority(
        plan,
        tasks=authority_tasks,
        target_rows=target_rows,
    )
    target = json.loads(canonical_json_bytes(base.target_access))
    target.pop("record_digest")
    target["target_count"] = len(target_rows)
    target["target_set_digest"] = quality_target_set_digest(target_rows, partition="train")
    target_raw = b"".join(canonical_json_bytes(row) + b"\n" for row in target_rows)
    target["target_sha256"] = hashlib.sha256(target_raw).hexdigest()
    for opened in target["opened_files"]:
        if opened["authority_root"] == "prepared":
            opened["sha256"] = target["target_sha256"]
    target_access = _record(target)

    ground = json.loads(canonical_json_bytes(base.ground_partition_receipt))
    ground.pop("record_digest")
    ground["authority"] = target_access
    ground["census"]["accepted_count"] = len(target_rows)
    ground["census"]["target_count"] = len(target_rows)
    ground_partition = _record(ground)
    ground_sha = hashlib.sha256(canonical_json_bytes(ground_partition) + b"\n").hexdigest()

    quality = json.loads(canonical_json_bytes(base.quality_authority))
    quality.pop("record_digest")
    partition = quality["training_partition"]
    partition.pop("record_digest")
    partition["target_access_record_digest"] = target_access["record_digest"]
    partition["target_count"] = len(target_rows)
    partition["target_set_digest"] = target_access["target_set_digest"]
    partition["ground_partition"] = {
        **partition["ground_partition"],
        "accepted_count": len(target_rows),
        "receipt_record_digest": ground_partition["record_digest"],
        "receipt_sha256": ground_sha,
    }
    quality["training_partition"] = _record(partition)
    quality_authority = _record(quality)

    execution = {
        "delta_module_sha256": _module_sha(delta_module),
        "device": plan.as_dict()["selector"]["device"],
        "quality_implementation_contract_digest": QUALITY_CONTRACT_DIGEST,
        "quality_module_sha256": _module_sha(quality_module),
        "runtime_sha256": "2" * 64,
        "selector_digest": plan.as_dict()["selector"]["selector_digest"],
        "threads": 1,
    }
    return QualityResolutionExecutionAuthority(
        quality_authority=quality_authority,
        target_access=target_access,
        ground_partition_receipt=ground_partition,
        execution_identity=execution,
        execution_identity_digest=content_digest(execution),
    )


def _study(plan: QualityResolutionPlan, authority: QualityResolutionExecutionAuthority):
    record = plan.as_dict()
    registered = record["config"]["registered"]
    protocol = {
        "continuations": 12,
        "evaluated_actions": registered["evaluated_actions"],
        "partition": "train",
        "quality_implementation_contract_digest": QUALITY_CONTRACT_DIGEST,
        "requested_lineages": 0,
        "reward_reads": 256,
        "seed": 907,
        "selector_device": record["selector"]["device"],
        "states_per_lineage_cap": registered["states_per_lineage_cap"],
        "tasks_per_lineage_cap": registered["tasks_per_lineage_cap"],
    }
    quality_authority = {
        "ground_partition_receipt_record_digest": authority.ground_partition_receipt[
            "record_digest"
        ],
        "ground_partition_receipt_sha256": authority.quality_authority["training_partition"][
            "ground_partition"
        ]["receipt_sha256"],
        "quality_authority_record_digest": authority.quality_authority["record_digest"],
        "target_access_record_digest": authority.target_access["record_digest"],
    }
    payload: dict[str, object] = {
        "advance": True,
        "plan": {
            "raw_sha256": _plan_sha(plan),
            "record_digest": record["record_digest"],
        },
        "production_protocol": protocol,
        "quality_authority": quality_authority,
        "schema": "isingfold.quality-continuation-resolution-study",
        "schema_version": 1,
        "selected_continuations": 12,
        "terminal": True,
    }
    receipt = _record(payload)
    raw = canonical_json_bytes(receipt) + b"\n"
    return QualityResolutionStudyReceipt(receipt, hashlib.sha256(raw).hexdigest())


def _target_rows(public: list[PreparedTask]) -> list[dict[str, object]]:
    result = []
    for index, item in enumerate(public):
        assert item.provenance is not None
        payload: dict[str, object] = {
            "certificate_digest": hashlib.sha256(
                f"certificate:{item.task_id}".encode()
            ).hexdigest(),
            "evaluator_protocol_digest": hashlib.sha256(
                f"evaluator:{item.task_id}".encode()
            ).hexdigest(),
            "instance_id": item.instance_id,
            "instance_record_digest": item.provenance.instance_record_digest,
            "learning_partition": "train",
            "reference_energy": -1.25,
            "reference_status": "exact_enumeration",
            "schema": "isingfold.evaluator-target",
            "schema_version": 2,
        }
        result.append(_record(payload))
    return sorted(result, key=lambda row: str(row["instance_id"]))


def _target_tasks(
    public: list[PreparedTask],
    target_rows: list[dict[str, object]],
    authority: QualityResolutionExecutionAuthority,
) -> list[PreparedTask]:
    by_instance = {row["instance_id"]: row for row in target_rows}
    result = []
    for item in public:
        target = by_instance[item.instance_id]
        result.append(
            replace(
                item,
                task=replace(item.task, ground_energy=float(target["reference_energy"])),
                reference_status=str(target["reference_status"]),
                certificate_digest=str(target["certificate_digest"]),
                evaluator_protocol_digest=str(target["evaluator_protocol_digest"]),
                quality_attestation_digest=str(
                    authority.target_access["publisher_attestation_digest"]
                ),
                quality_evidence_manifest_digest=str(
                    authority.target_access["evidence_manifest_record_digest"]
                ),
                quality_evidence_manifest_sha256=str(
                    authority.target_access["evidence_manifest_sha256"]
                ),
                quality_target_set_digest=str(authority.target_access["target_set_digest"]),
                quality_target_count=int(authority.target_access["target_count"]),
            )
        )
    return result


def _runner(task, context, **kwargs):
    del task, context, kwargs
    raise AssertionError("an unregistered continuation runner must never execute")


def _runtime_identity() -> dict[str, object]:
    return {
        "capacity_module_sha256": _module_sha(capacity_module),
        "continuation_runner": "isingfold.rl.data.quality.run_continuation",
        "execution_runtime_sha256": "2" * 64,
        "host_class": "unit-fixture",
        "measurement_clock": QUALITY_CAPACITY_MEASUREMENT_PROTOCOL,
        "quality_implementation_contract_digest": QUALITY_CONTRACT_DIGEST,
        "quality_module_sha256": _module_sha(quality_module),
    }


@pytest.fixture()
def capacity_inputs(tmp_path: Path) -> dict[str, Any]:
    plan, public, context, bank, bank_sha256 = _real_projection_plan(tmp_path)
    target_rows = _target_rows(public)
    authority = _capacity_authority(plan, target_rows, public)
    targets = _target_tasks(public, target_rows, authority)
    study = _study(plan, authority)
    budget = QualityCapacityBudget(
        budget_id="paper-capacity-budget-fixture",
        maximum_artifact_bytes=10**12,
        maximum_cpu_seconds=10**9,
        maximum_elapsed_seconds=10**9,
        available_workers=16,
        minimum_scratch_free_bytes=1,
    )
    budget_path = tmp_path / "budget.json"
    budget_sha = publish_quality_capacity_budget(budget_path, budget)
    selection_path = tmp_path / "selection.json"
    selection_sha = build_quality_capacity_canary_selection(
        plan,
        expected_plan_sha256=_plan_sha(plan),
        resolution_receipt=study,
        public_prepared=public,
        context=context,
        selector=fixed_strength_selector(),
        output_path=selection_path,
        initializer_bank=bank,
        expected_initializer_bank_manifest_sha256=bank_sha256,
        allow_test_initializer_bank=True,
    )
    return {
        "authority": authority,
        "budget": budget,
        "budget_path": budget_path,
        "budget_sha": budget_sha,
        "bank": bank,
        "bank_sha256": bank_sha256,
        "context": context,
        "plan": plan,
        "public": public,
        "selection_path": selection_path,
        "selection_sha": selection_sha,
        "study": study,
        "target_rows": target_rows,
        "targets": targets,
        "tmp_path": tmp_path,
    }


def test_budget_and_canary_selection_are_pinned_outcome_blind_and_use_real_work(
    capacity_inputs: dict[str, Any],
) -> None:
    item = capacity_inputs
    loaded_budget = load_quality_capacity_budget(
        item["budget_path"], expected_budget_sha256=item["budget_sha"]
    )
    assert loaded_budget == item["budget"]
    with pytest.raises(FileExistsError):
        publish_quality_capacity_budget(item["budget_path"], item["budget"])

    selection = load_quality_capacity_canary_selection(
        item["selection_path"],
        expected_selection_sha256=item["selection_sha"],
        plan=item["plan"],
        expected_plan_sha256=_plan_sha(item["plan"]),
        resolution_receipt=item["study"],
    ).as_dict()
    assert selection["schema"] == QUALITY_CAPACITY_CANARY_SELECTION_SCHEMA
    assert len(selection["selection"]["selected_lineages"]) == 16
    assert selection["selection"]["coverage_complete"] is True
    assert selection["target_accessed"] is False
    assert selection["rows"]
    assert all(row["public_envelope_projection_digest"] for row in selection["rows"])
    assert any(
        any(value > 0 for value in row["work_coordinates"]["candidate_max"].values())
        for row in selection["rows"]
    )
    production = item["plan"].as_dict()["production_plan"]
    expected_actions = sum(len(row["actions"]) for row in production["rows"])
    assert selection["upper_census"] == {
        "actions": expected_actions,
        "lineages": 16,
        "rows": len(production["rows"]),
        "trajectories": expected_actions * 12,
    }


def test_capacity_projection_uses_ceilings_and_fails_every_underestimated_budget() -> None:
    permissive = QualityCapacityBudget(
        budget_id="permissive",
        maximum_artifact_bytes=606,
        maximum_cpu_seconds=4,
        maximum_elapsed_seconds=4,
        available_workers=2,
        minimum_scratch_free_bytes=1,
    )
    result = compute_guarded_capacity_projection(
        effective_max_bytes_per_trajectory=101,
        max_cpu_time_ns_per_trajectory=600_000_000,
        max_wall_time_ns_per_trajectory=900_000_000,
        upper_trajectory_census=3,
        scratch_free_bytes=1_818,
        budget=permissive,
    )
    assert result["projected_artifact_bytes"] == 606
    assert result["projected_cpu_seconds"] == 4
    assert result["projected_elapsed_seconds"] == 4
    assert result["required_free_bytes"] == 1_818
    assert result["pass"] is True

    for field in (
        "maximum_artifact_bytes",
        "maximum_cpu_seconds",
        "maximum_elapsed_seconds",
        "minimum_scratch_free_bytes",
    ):
        values = {
            "budget_id": f"tight-{field}",
            "maximum_artifact_bytes": 606,
            "maximum_cpu_seconds": 4,
            "maximum_elapsed_seconds": 4,
            "available_workers": 2,
            "minimum_scratch_free_bytes": 1,
        }
        if field == "minimum_scratch_free_bytes":
            values[field] = 1_819
        else:
            values[field] -= 1
        failed = compute_guarded_capacity_projection(
            effective_max_bytes_per_trajectory=101,
            max_cpu_time_ns_per_trajectory=600_000_000,
            max_wall_time_ns_per_trajectory=900_000_000,
            upper_trajectory_census=3,
            scratch_free_bytes=1_818,
            budget=QualityCapacityBudget(**values),
        )
        assert failed["pass"] is False

    guarded_scratch_failure = compute_guarded_capacity_projection(
        effective_max_bytes_per_trajectory=101,
        max_cpu_time_ns_per_trajectory=600_000_000,
        max_wall_time_ns_per_trajectory=900_000_000,
        upper_trajectory_census=3,
        scratch_free_bytes=1_817,
        budget=replace(permissive, minimum_scratch_free_bytes=1),
    )
    assert guarded_scratch_failure["pass"] is False
    assert guarded_scratch_failure["comparisons"]["scratch_above_guarded_requirement"] is False


def test_real_capacity_canary_serializes_v7_rows_and_publishes_a_launch_gate(
    capacity_inputs: dict[str, Any],
) -> None:
    item = capacity_inputs
    output = item["tmp_path"] / "capacity-canary.json"
    runtime = _runtime_identity()
    canary_sha = run_quality_capacity_canary(
        item["plan"],
        expected_plan_sha256=_plan_sha(item["plan"]),
        resolution_receipt=item["study"],
        selection_path=item["selection_path"],
        expected_selection_sha256=item["selection_sha"],
        budget_path=item["budget_path"],
        expected_budget_sha256=item["budget_sha"],
        prepared=item["targets"],
        target_records=item["target_rows"],
        authority=item["authority"],
        context=item["context"],
        selector=fixed_strength_selector(),
        runtime_identity=runtime,
        runtime_identity_digest=content_digest(runtime),
        scratch_directory=item["tmp_path"],
        output_path=output,
        initializer_bank=item["bank"],
        expected_initializer_bank_manifest_sha256=item["bank_sha256"],
        allow_test_initializer_bank=True,
    )
    canary = load_quality_capacity_canary(
        output,
        expected_canary_sha256=canary_sha,
        plan=item["plan"],
        expected_plan_sha256=_plan_sha(item["plan"]),
        resolution_receipt=item["study"],
        selection_path=item["selection_path"],
        expected_selection_sha256=item["selection_sha"],
        budget_path=item["budget_path"],
        expected_budget_sha256=item["budget_sha"],
        allow_test_initializer_bank=True,
    ).as_dict()
    assert canary["schema"] == QUALITY_CAPACITY_CANARY_SCHEMA
    assert canary["pass"] is True
    assert canary["actual"]["lineages"] == 16
    assert canary["actual"]["rows"] == len(
        load_quality_capacity_canary_selection(
            item["selection_path"],
            expected_selection_sha256=item["selection_sha"],
            plan=item["plan"],
            expected_plan_sha256=_plan_sha(item["plan"]),
            resolution_receipt=item["study"],
        ).as_dict()["rows"]
    )
    assert canary["actual"]["trajectories"] == (canary["actual"]["actions"] * 12)
    assert (
        canary["actual"]["valid_trajectories"] + canary["actual"]["invalid_trajectories"]
        == canary["actual"]["trajectories"]
    )
    assert canary["measurements"]["row_bytes"]["maximum"] > 0
    assert canary["measurements"]["continuation_receipt_bytes"]["maximum"] > 0
    assert canary["projections"]["projected_artifact_bytes"] == (
        2
        * canary["measurements"]["effective_max_bytes_per_trajectory"]
        * canary["upper_census"]["trajectories"]
    )
    with pytest.raises(QualityResolutionError, match="publication-eligible"):
        require_passing_quality_capacity_canary(
            output,
            expected_canary_sha256=canary_sha,
            plan=item["plan"],
            expected_plan_sha256=_plan_sha(item["plan"]),
            resolution_receipt=item["study"],
            selection_path=item["selection_path"],
            expected_selection_sha256=item["selection_sha"],
            budget_path=item["budget_path"],
            expected_budget_sha256=item["budget_sha"],
        )


def test_capacity_canary_rejects_target_value_not_joined_to_authenticated_set(
    capacity_inputs: dict[str, Any],
) -> None:
    item = capacity_inputs
    changed = list(item["targets"])
    changed[0] = replace(
        changed[0],
        task=replace(changed[0].task, ground_energy=-999.0),
    )
    runtime = _runtime_identity()
    with pytest.raises(QualityResolutionError, match="target row"):
        run_quality_capacity_canary(
            item["plan"],
            expected_plan_sha256=_plan_sha(item["plan"]),
            resolution_receipt=item["study"],
            selection_path=item["selection_path"],
            expected_selection_sha256=item["selection_sha"],
            budget_path=item["budget_path"],
            expected_budget_sha256=item["budget_sha"],
            prepared=changed,
            target_records=item["target_rows"],
            authority=item["authority"],
            context=item["context"],
            selector=fixed_strength_selector(),
            runtime_identity=runtime,
            runtime_identity_digest=content_digest(runtime),
            scratch_directory=item["tmp_path"],
            output_path=item["tmp_path"] / "target-tamper-canary.json",
            initializer_bank=item["bank"],
            expected_initializer_bank_manifest_sha256=item["bank_sha256"],
            allow_test_initializer_bank=True,
        )


def test_capacity_canary_rejects_self_declared_fake_continuation_runner(
    capacity_inputs: dict[str, Any],
) -> None:
    item = capacity_inputs
    runtime = _runtime_identity()
    with pytest.raises(QualityResolutionError, match="registered run_continuation"):
        run_quality_capacity_canary(
            item["plan"],
            expected_plan_sha256=_plan_sha(item["plan"]),
            resolution_receipt=item["study"],
            selection_path=item["selection_path"],
            expected_selection_sha256=item["selection_sha"],
            budget_path=item["budget_path"],
            expected_budget_sha256=item["budget_sha"],
            prepared=item["targets"],
            target_records=item["target_rows"],
            authority=item["authority"],
            context=item["context"],
            selector=fixed_strength_selector(),
            runtime_identity=runtime,
            runtime_identity_digest=content_digest(runtime),
            scratch_directory=item["tmp_path"],
            output_path=item["tmp_path"] / "fake-runner-canary.json",
            initializer_bank=item["bank"],
            expected_initializer_bank_manifest_sha256=item["bank_sha256"],
            allow_test_initializer_bank=True,
            continuation_runner=_runner,
        )


def test_capacity_canary_rejects_unknown_target_row_field(
    capacity_inputs: dict[str, Any],
) -> None:
    item = capacity_inputs
    changed = [dict(row) for row in item["target_rows"]]
    changed[0]["unregistered_target_value"] = -999.0
    runtime = _runtime_identity()
    with pytest.raises(QualityResolutionError, match="target row schema"):
        run_quality_capacity_canary(
            item["plan"],
            expected_plan_sha256=_plan_sha(item["plan"]),
            resolution_receipt=item["study"],
            selection_path=item["selection_path"],
            expected_selection_sha256=item["selection_sha"],
            budget_path=item["budget_path"],
            expected_budget_sha256=item["budget_sha"],
            prepared=item["targets"],
            target_records=changed,
            authority=item["authority"],
            context=item["context"],
            selector=fixed_strength_selector(),
            runtime_identity=runtime,
            runtime_identity_digest=content_digest(runtime),
            scratch_directory=item["tmp_path"],
            output_path=item["tmp_path"] / "unknown-target-field-canary.json",
            initializer_bank=item["bank"],
            expected_initializer_bank_manifest_sha256=item["bank_sha256"],
            allow_test_initializer_bank=True,
        )
