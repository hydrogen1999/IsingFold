"""Target-free, shared validation bootstrap bank for RL-value selection."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import networkx as nx
import pytest

from isingfold.embedding import LogicalProblem
from isingfold.rl.complete_system import (
    COMPLETE_POLICY_RESTART_MODE,
    LAC_INITIALIZER_METHOD_ID,
    CompleteSystemConfig,
    InitializerAttemptReceipt,
    PartialWorkVector,
)
from isingfold.rl.contracts import Context, WorkVector, stable_digest
from isingfold.rl.data.prepared import (
    PreparedDesignCondition,
    PreparedProvenance,
    PreparedTask,
)
from isingfold.rl.env import EmbeddingTask
from isingfold.rl.external import BackendIdentity
from isingfold.rl.initializer_bank import EpisodeBootstrapOutcome, InitializerSnapshot
from isingfold.rl.validation_bootstrap_bank import (
    REPRESENTATION_EVALUATION_SEED,
    REPRESENTATION_VALIDATION_BOOTSTRAP_PRESET,
    RL_VALUE_EVALUATION_SEED,
    RL_VALUE_FINAL_TEST_SEED,
    RL_VALUE_REGISTERED_REPETITIONS,
    bootstrap_context_digest,
    bootstrap_execution_record_digest,
    bootstrap_same_support_contract_digest,
    build_bootstrap_plan,
    execute_bootstrap_row,
    load_validation_bootstrap_bank,
    load_bootstrap_plan,
    publish_validation_bootstrap_record,
    seal_validation_bootstrap_bank,
    validation_bootstrap_record_from_outcome,
    write_validation_bootstrap_plan,
)


def _runtime_manifest() -> dict[str, object]:
    body: dict[str, object] = {
        "schema": "lac-minorminer.runtime-implementation",
        "schema_version": 1,
        "native_extension_sha256": stable_digest({"native": "test-double"}),
        "python_source_sha256": {"api.py": stable_digest({"source": "api"})},
        "backend_info": {
            "backend": "lac_minorminer_cpp",
            "package_version": "0.1.0",
        },
    }
    return {**body, "manifest_sha256": stable_digest(body)}


class _FakeLAC:
    def __init__(self) -> None:
        runtime = _runtime_manifest()
        self.identity = BackendIdentity(
            method_id=LAC_INITIALIZER_METHOD_ID,
            distribution="lac-minorminer",
            version="0.1.0",
            entrypoint="lac_minorminer.find_embedding",
            implementation=(
                "lac_minorminer_cpp:runtime-sha256:" + str(runtime["manifest_sha256"])
            ),
        )
        self.runtime_implementation_manifest = runtime


def _config() -> CompleteSystemConfig:
    return CompleteSystemConfig(
        online_wallclock_seconds=30.0,
        max_initializer_attempts=2,
        audit_reads=4096,
        selection_rule="resource-lexicographic",
        online_evaluator_feedback=False,
        initializer_backend="lac-minorminer",
        initializer_method_id=LAC_INITIALIZER_METHOD_ID,
        expected_initializer_version="0.1.0",
        policy_restart_mode=COMPLETE_POLICY_RESTART_MODE,
        initializer_parameters={"tries": 1, "max_transitions": 31, "max_candidates": 8},
    )


def _context() -> Context:
    return Context(qubit_cap=9)


def _provenance(lineage: str) -> PreparedProvenance:
    digest = stable_digest({"lineage": lineage})
    return PreparedProvenance(
        base_parent_lineage=lineage,
        source_logical_lineage=f"source-{lineage}",
        descendant_transform_kinds=("identity",),
        descendant_transform_sha256=stable_digest({"transform": lineage}),
        nominal_topology="grid",
        nominal_size=3,
        pristine_host_sha256=stable_digest({"host": lineage, "kind": "pristine"}),
        active_topology="grid",
        active_host_sha256=stable_digest({"host": lineage, "kind": "active"}),
        host_artifact_sha256=stable_digest({"host": lineage, "kind": "artifact"}),
        fault_status="none",
        fault_mask_sha256=stable_digest({"fault": lineage}),
        calibration_status="not_applicable",
        calibration_sha256=None,
        distribution_regime="id",
        distribution_stratum="unit-test",
        # This is the immutable source-corpus partition, not the learning split.
        source_partition="isingfold-corpus-v2",
        source_release_id="unit-test-release",
        source_release_manifest_sha256=stable_digest({"release": 1}),
        split_manifest_sha256=stable_digest({"split": 1}),
        group_id=f"group-{lineage}",
        group_record_digest=stable_digest({"group": lineage}),
        instance_record_digest=stable_digest({"instance": lineage}),
        source_record_digest=stable_digest({"source": lineage}),
        record_digest=digest,
    )


def _condition(lineage: str, partition: str = "val") -> PreparedDesignCondition:
    return PreparedDesignCondition(
        base_lineage_key=lineage,
        learning_partition=partition,
        application_family="planted-ising",
        problem_origin="generated",
        host_family="grid",
        fault_status="none",
        distribution_regime="id",
        calibration_status="not_applicable",
        calibration_sha256=None,
        embedding_difficulty="hard",
        sampling_difficulty="hard",
        decision_difficulty="hard",
        registry_row_digest=stable_digest({"condition": lineage}),
    )


def _prepared(
    task_id: str, instance_id: str, lineage: str, partition: str = "val"
) -> PreparedTask:
    logical = nx.path_graph(3)
    host = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))
    problem = LogicalProblem.from_dicts(
        {0: -1.0, 1: 0.5, 2: -0.25},
        {(0, 1): -1.0, (1, 2): 0.75},
    )
    stale = {0: frozenset({6}), 1: frozenset({7}), 2: frozenset({8})}
    return PreparedTask(
        task=EmbeddingTask(
            name=task_id,
            logical=logical,
            host=host,
            problem=problem,
            ground_energy=None,
            lineage=lineage,
            initial_embedding=stale,
        ),
        task_id=task_id,
        instance_id=instance_id,
        partition=partition,
        initializer_record_digest=stable_digest({"initializer": task_id}),
        public_instance_record_digest=stable_digest({"policy-instance": instance_id}),
        reference_status=None,
        certificate_digest=None,
        evaluator_protocol_digest=None,
        prepared_schema_version=4,
        corpus_scope="production-designed-v4",
        provenance=_provenance(lineage),
        design_condition=_condition(lineage, partition),
    )


def _prepared_census_authority(
    tasks: list[PreparedTask], *, partition: str
) -> tuple[dict[str, object], dict[str, object], str]:
    split_body: dict[str, object] = {
        "base_parent_lineage_to_split": {
            task.task.lineage: task.partition for task in tasks
        },
        "schema": "isingfold.lineage-splits",
        "schema_version": 4,
        "train": sorted(task.task_id for task in tasks if task.partition == "train"),
        "val": sorted(task.task_id for task in tasks if task.partition == "val"),
        "test": sorted(task.task_id for task in tasks if task.partition == "test"),
    }
    splits = {**split_body, "record_digest": stable_digest(split_body)}
    split_raw = (
        json.dumps(splits, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()
    instance_count = len({task.instance_id for task in tasks if task.partition == partition})
    descriptors = {
        name: {
            "path": f"targets/{name}.jsonl",
            "records": instance_count if name == partition else 0,
            "sha256": stable_digest({"targets": name}),
            "target_set_digest": stable_digest({"target-set": name}),
        }
        for name in ("train", "val", "test")
    }
    authority_body: dict[str, object] = {
        "partitions": descriptors,
        "schema": "isingfold.evaluator-target-authority",
        "schema_version": 1,
        "total_targets": instance_count,
    }
    authority = {**authority_body, "record_digest": stable_digest(authority_body)}
    outputs: dict[str, object] = {
        "splits.json": {
            "records": 1,
            "sha256": hashlib.sha256(split_raw).hexdigest(),
        }
    }
    for name, descriptor in descriptors.items():
        outputs[f"targets/{name}.jsonl"] = {
            "records": descriptor["records"],
            "sha256": descriptor["sha256"],
        }
    manifest_body: dict[str, object] = {
        "counts": {
            "evaluator_targets_by_partition": {
                name: descriptor["records"] for name, descriptor in descriptors.items()
            }
        },
        "corpus_scope": "production-designed-v4",
        "outputs": outputs,
        "schema": "isingfold.prepared-candidate-bank",
        "schema_version": 4,
        "target_authority": authority,
    }
    manifest = {**manifest_body, "record_digest": stable_digest(manifest_body)}
    manifest_raw = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode()
    return manifest, splits, hashlib.sha256(manifest_raw).hexdigest()


def _plan(
    tasks: list[PreparedTask],
    *,
    partition: str = "val",
    evaluation_seed: int = RL_VALUE_EVALUATION_SEED,
):
    backend = _FakeLAC()
    config = _config()
    context = _context()
    context_digest = bootstrap_context_digest(context)
    manifest, splits, manifest_sha = _prepared_census_authority(
        tasks, partition=partition
    )
    return build_bootstrap_plan(
        tasks,
        prepared_manifest=manifest,
        prepared_split_registry=splits,
        prepared_manifest_sha256=manifest_sha,
        protocol_registry_sha256=stable_digest({"grid": "bytes"}),
        protocol_record_digest=stable_digest({"grid": "rl-value-evaluation"}),
        same_support_contract_digest=bootstrap_same_support_contract_digest(
            partition=partition,
            evaluation_seed=evaluation_seed,
            repetitions=RL_VALUE_REGISTERED_REPETITIONS,
            config_digest=config.digest,
            context_digest=context_digest,
        ),
        partition=partition,
        evaluation_seed=evaluation_seed,
        repetitions=RL_VALUE_REGISTERED_REPETITIONS,
        initializer=backend,
        runtime_implementation_manifest=backend.runtime_implementation_manifest,
        config=config,
        context=context,
        allow_test_backend=True,
    )


def _attempt(seed: int, index: int = 0) -> InitializerAttemptReceipt:
    work = WorkVector(route_expansions=2, materializations=1)
    return InitializerAttemptReceipt(
        attempt_index=index,
        seed=seed,
        status="VALID_CANDIDATE",
        reported_seconds=0.01,
        observed_seconds=0.01,
        backend_work=PartialWorkVector.known(work),
        runner_work=WorkVector(restart_work=1, validator_calls=1),
        embedding_digest=stable_digest({"embedding": seed}),
        validation_digest=stable_digest({"validation": seed}),
        qubits=3,
        max_chain=1,
        backend_diagnostics={"runtime_implementation_manifest": _runtime_manifest()},
    )


def _snapshot(
    plan,
    row,
    *,
    kind: str,
    slot: int,
    work_before: WorkVector,
    time_before: float,
    success: bool = True,
) -> InitializerSnapshot:
    system_seed = row.system_seed if kind == "initial" else row.cache_system_seeds[slot]
    attempt_seed = (
        row.initial_attempt_seeds[0]
        if kind == "initial"
        else row.cache_attempt_seeds[slot][0]
    )
    attempt = _attempt(attempt_seed)
    work = attempt.total_work.to_work_vector()
    assert work is not None
    attempts = (attempt,)
    if not success:
        failed = dataclasses.replace(
            attempt,
            status="NO_EMBEDDING",
            runner_work=WorkVector(restart_work=1),
            embedding_digest=None,
            validation_digest=None,
            qubits=None,
            max_chain=None,
        )
        attempts = (failed,)
        work = failed.total_work.to_work_vector()
        assert work is not None
    embedding = ((0, (0,)), (1, (1,)), (2, (2,))) if success else None
    draw_key = stable_digest(
        {
            "schema": "isingfold.rl-value-bootstrap-draw",
            "schema_version": 1,
            "plan_record_digest": plan.record_digest,
            "row_key": row.row_key,
            "census_index": row.census_index,
            "slot_kind": kind,
            "slot_index": slot,
            "system_seed": system_seed,
            "attempt_seeds": list(
                row.initial_attempt_seeds
                if kind == "initial"
                else row.cache_attempt_seeds[slot]
            ),
        }
    )
    return InitializerSnapshot(
        plan_record_digest=plan.record_digest,
        conditional_episode_index=row.census_index,
        draw_index=slot,
        slot_kind=kind,
        slot_index=slot,
        draw_key=draw_key,
        instance_id=row.instance_id,
        base_lineage=row.base_lineage,
        system_seed=system_seed,
        attempts=attempts,
        selected_attempt=0 if success else None,
        selected_embedding=embedding,
        selected_embedding_digest=(attempt.embedding_digest if success else None),
        initializer_work=work,
        environment_budget_debit=work if success or kind == "restart_cache" else WorkVector(),
        work_before=work_before,
        work_after=work_before + work,
        time_before_seconds=time_before,
        time_after_seconds=time_before + 0.01,
        execution_status="SUCCESS" if success else "FAILED",
        success=success,
        failure_reason=None if success else "no valid initializer candidate",
        initialization_seconds=0.01,
    )


def _outcome(plan, row, *, initial_success: bool) -> EpisodeBootstrapOutcome:
    initial = _snapshot(
        plan,
        row,
        kind="initial",
        slot=0,
        work_before=WorkVector(),
        time_before=0.0,
        success=initial_success,
    )
    cache: list[InitializerSnapshot] = []
    work = initial.work_after
    elapsed = initial.time_after_seconds
    if initial_success:
        for slot in range(2):
            item = _snapshot(
                plan,
                row,
                kind="restart_cache",
                slot=slot,
                work_before=work,
                time_before=elapsed,
            )
            cache.append(item)
            work = item.work_after
            elapsed = item.time_after_seconds
    cache_work = WorkVector()
    for item in cache:
        cache_work = cache_work + item.initializer_work
    cache_tuple = tuple(cache)
    return EpisodeBootstrapOutcome(
        episode_schedule_index=row.census_index,
        instance_id=row.instance_id,
        base_lineage=row.base_lineage,
        public_task_digest=next(
            task.public_task_digest
            for task in plan.tasks
            if task.instance_id == row.instance_id
        ),
        initial_snapshot=initial,
        restart_cache_snapshots=cache_tuple,
        restart_cache_slot_count=2,
        initial_generation_debit=initial.initializer_work,
        restart_cache_fill_debit=cache_work,
        total_pre_policy_debit=initial.initializer_work + cache_work,
        precomputed_online_seconds=elapsed,
        plan_record_digest=initial.plan_record_digest,
        manifest_record_digest=bootstrap_execution_record_digest(
            plan, row.row_key, initial, cache_tuple
        ),
        config_digest=plan.config_digest,
        context_digest=plan.context_digest,
    )


class _RawOutcome:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def as_dict(self) -> dict[str, object]:
        return self.payload


@pytest.mark.parametrize("initial_success, expected_calls", [(True, 3), (False, 1)])
def test_row_executor_runs_one_initial_then_causal_k2_or_short_circuits_failure(
    monkeypatch: pytest.MonkeyPatch,
    initial_success: bool,
    expected_calls: int,
) -> None:
    prepared = _prepared("candidate-a", "instance-a", "lineage-0")
    plan = _plan([prepared])
    row = plan.census[0]
    calls: list[tuple[str, int, WorkVector, float]] = []

    def fake_execute(draw, **kwargs):
        calls.append(
            (
                draw.slot_kind,
                draw.slot_index,
                kwargs.get("work_before", WorkVector()),
                kwargs.get("time_before_seconds", 0.0),
            )
        )
        success = initial_success if draw.slot_kind == "initial" else True
        return _snapshot(
            plan,
            row,
            kind=draw.slot_kind,
            slot=draw.slot_index,
            work_before=calls[-1][2],
            time_before=calls[-1][3],
            success=success,
        )

    monkeypatch.setattr(
        "isingfold.rl.validation_bootstrap_bank.execute_initializer_draw", fake_execute
    )
    public_task = dataclasses.replace(
        prepared.task, name=prepared.instance_id, initial_embedding=None
    )
    outcome = execute_bootstrap_row(
        plan,
        row.row_key,
        public_task,
        initializer=_FakeLAC(),
        config=_config(),
        context=_context(),
        allow_test_backend=True,
    )

    assert len(calls) == expected_calls
    assert len(outcome.restart_cache_snapshots) == (2 if initial_success else 0)
    assert calls[0][2] == WorkVector()
    assert calls[0][3] == 0.0
    if initial_success:
        assert calls[1][2] == outcome.initial_snapshot.work_after
        assert calls[2][2] == outcome.restart_cache_snapshots[0].work_after


def test_plan_enumerates_the_complete_validation_census_canonically() -> None:
    tasks = [
        _prepared("candidate-b", "instance-b", "lineage-0"),
        _prepared("candidate-c", "instance-c", "lineage-1"),
        _prepared("candidate-a", "instance-a", "lineage-0"),
    ]
    plan = _plan(tasks)

    assert plan.partition == "val"
    assert plan.evaluation_seed == 44021
    assert plan.repetitions == 4
    assert plan.restart_cache_slots == 2
    assert len(plan.census) == 12
    assert [
        (row.base_lineage, row.instance_id, row.repetition)
        for row in plan.census
    ] == [
        (lineage, instance, repetition)
        for lineage, instance in (
            ("lineage-0", "instance-a"),
            ("lineage-0", "instance-b"),
            ("lineage-1", "instance-c"),
        )
        for repetition in range(4)
    ]
    assert len({row.row_key for row in plan.census}) == 12
    assert all(len(row.cache_system_seeds) == 2 for row in plan.census)
    assert all(len(row.initial_attempt_seeds) == 2 for row in plan.census)


def test_plan_rejects_a_legacy_no_restart_complete_system_config() -> None:
    task = _prepared("candidate-a", "instance-a", "lineage-0")
    manifest, splits, prepared_manifest_sha = _prepared_census_authority(
        [task], partition="val"
    )
    config = dataclasses.replace(
        _config(), policy_restart_mode="disabled-no-native-replay-v1"
    )
    backend = _FakeLAC()
    with pytest.raises(ValueError, match="persistent K=2"):
        build_bootstrap_plan(
            [task],
            prepared_manifest=manifest,
            prepared_split_registry=splits,
            prepared_manifest_sha256=prepared_manifest_sha,
            protocol_registry_sha256=stable_digest({"grid": "bytes"}),
            protocol_record_digest=stable_digest({"grid": "rl-value-evaluation"}),
            same_support_contract_digest=bootstrap_same_support_contract_digest(
                partition="val",
                evaluation_seed=RL_VALUE_EVALUATION_SEED,
                repetitions=RL_VALUE_REGISTERED_REPETITIONS,
                config_digest=config.digest,
                context_digest=bootstrap_context_digest(_context()),
            ),
            partition="val",
            evaluation_seed=RL_VALUE_EVALUATION_SEED,
            repetitions=RL_VALUE_REGISTERED_REPETITIONS,
            initializer=backend,
            runtime_implementation_manifest=backend.runtime_implementation_manifest,
            config=config,
            context=_context(),
            allow_test_backend=True,
        )


def test_plan_rejects_a_subset_of_the_authenticated_partition_census() -> None:
    tasks = [
        _prepared("candidate-a", "instance-a", "lineage-0"),
        _prepared("candidate-b", "instance-b", "lineage-1"),
        _prepared("candidate-c", "instance-c", "lineage-2"),
    ]
    manifest, splits, manifest_sha = _prepared_census_authority(
        tasks, partition="val"
    )
    backend = _FakeLAC()
    config = _config()
    context = _context()
    with pytest.raises(ValueError, match="full authenticated partition census"):
        build_bootstrap_plan(
            tasks[:2],
            prepared_manifest=manifest,
            prepared_split_registry=splits,
            prepared_manifest_sha256=manifest_sha,
            protocol_registry_sha256=stable_digest({"grid": "bytes"}),
            protocol_record_digest=stable_digest({"grid": "rl-value-evaluation"}),
            same_support_contract_digest=bootstrap_same_support_contract_digest(
                partition="val",
                evaluation_seed=RL_VALUE_EVALUATION_SEED,
                repetitions=RL_VALUE_REGISTERED_REPETITIONS,
                config_digest=config.digest,
                context_digest=bootstrap_context_digest(context),
            ),
            partition="val",
            evaluation_seed=RL_VALUE_EVALUATION_SEED,
            repetitions=RL_VALUE_REGISTERED_REPETITIONS,
            initializer=backend,
            runtime_implementation_manifest=backend.runtime_implementation_manifest,
            config=config,
            context=context,
            allow_test_backend=True,
        )


def test_representation_rl_value_and_final_test_have_distinct_registered_presets() -> None:
    representation = _plan(
        [_prepared("candidate-a", "instance-a", "lineage-0")],
        evaluation_seed=REPRESENTATION_EVALUATION_SEED,
    )
    assert representation.protocol_preset == REPRESENTATION_VALIDATION_BOOTSTRAP_PRESET
    assert representation.partition == "val"
    assert representation.evaluation_seed == 33049

    plan = _plan(
        [_prepared("candidate-a", "instance-a", "lineage-0", "test")],
        partition="test",
        evaluation_seed=RL_VALUE_FINAL_TEST_SEED,
    )
    assert plan.protocol_preset == "final-test"
    assert plan.partition == "test"
    assert plan.evaluation_seed == 55079
    assert len(plan.census) == 4
    validation = _plan([_prepared("candidate-a", "instance-a", "lineage-0")])
    assert representation.same_support_contract_digest != validation.same_support_contract_digest
    assert representation.record_digest != validation.record_digest
    assert plan.same_support_contract_digest != validation.same_support_contract_digest
    assert plan.record_digest != validation.record_digest

    with pytest.raises(ValueError, match="not a registered"):
        _plan(
            [_prepared("candidate-a", "instance-a", "lineage-0", "test")],
            partition="test",
            evaluation_seed=RL_VALUE_EVALUATION_SEED,
        )


@pytest.mark.parametrize("partition", ["train", "test"])
def test_plan_rejects_nonvalidation_and_any_evaluator_target(partition: str) -> None:
    task = _prepared("candidate-a", "instance-a", "lineage-0")
    task = dataclasses.replace(
        task,
        partition=partition,
        provenance=dataclasses.replace(task.provenance, source_partition=partition),
        design_condition=dataclasses.replace(task.design_condition, learning_partition=partition),
    )
    with pytest.raises(ValueError, match="registered partition"):
        _plan([task])

    targeted = dataclasses.replace(
        _prepared("candidate-a", "instance-a", "lineage-0"),
        task=dataclasses.replace(
            _prepared("candidate-a", "instance-a", "lineage-0").task,
            ground_energy=-2.5,
        ),
    )
    with pytest.raises(ValueError, match="evaluator target"):
        _plan([targeted])


def test_failed_initial_is_denominator_zero_and_forbids_actor_and_cache() -> None:
    plan = _plan([_prepared("candidate-a", "instance-a", "lineage-0")])
    row = plan.census[0]
    outcome = _outcome(plan, row, initial_success=False)
    record = validation_bootstrap_record_from_outcome(plan, row.row_key, outcome)

    assert record.denominator_eligible is True
    assert record.fixed_utility == 0.0
    assert record.actor_invocation_permitted is False
    assert record.cache_invocation_permitted is False
    assert record.initial_status == "FAILED"
    assert record.cache_slot_count == 0

    illegal_payload = outcome.as_dict()
    illegal_payload["restart_cache_snapshots"] = [outcome.initial_snapshot.as_dict()]
    body = {key: value for key, value in illegal_payload.items() if key != "record_digest"}
    illegal_payload["record_digest"] = stable_digest(body)
    with pytest.raises(ValueError, match="failed initial"):
        validation_bootstrap_record_from_outcome(
            plan, row.row_key, _RawOutcome(illegal_payload)
        )

    target_bearing = outcome.as_dict()
    target_bearing["ground_energy"] = -2.5
    with pytest.raises(ValueError, match="forbidden evaluator target"):
        validation_bootstrap_record_from_outcome(
            plan, row.row_key, _RawOutcome(target_bearing)
        )

    nested_target = outcome.as_dict()
    nested_target["initial_snapshot"]["attempts"][0]["backend_diagnostics"][
        "reference_energy"
    ] = -2.5
    with pytest.raises(ValueError, match="forbidden evaluator target"):
        validation_bootstrap_record_from_outcome(
            plan, row.row_key, _RawOutcome(nested_target)
        )


def test_success_requires_exactly_two_causal_cache_slots_and_registered_seeds() -> None:
    plan = _plan([_prepared("candidate-a", "instance-a", "lineage-0")])
    row = plan.census[0]
    outcome = _outcome(plan, row, initial_success=True)
    record = validation_bootstrap_record_from_outcome(plan, row.row_key, outcome)

    assert record.actor_invocation_permitted is True
    assert record.fixed_utility is None
    assert record.cache_slot_count == 2
    assert record.total_pre_policy_work == outcome.restart_cache_snapshots[-1].work_after

    shortened_payload = outcome.as_dict()
    shortened_payload["restart_cache_snapshots"] = [
        outcome.restart_cache_snapshots[0].as_dict()
    ]
    body = {key: value for key, value in shortened_payload.items() if key != "record_digest"}
    shortened_payload["record_digest"] = stable_digest(body)
    with pytest.raises(ValueError, match="restart-cache slot"):
        validation_bootstrap_record_from_outcome(
            plan, row.row_key, _RawOutcome(shortened_payload)
        )

    changed = dataclasses.replace(
        outcome.restart_cache_snapshots[0],
        system_seed=(outcome.restart_cache_snapshots[0].system_seed + 1) % (2**31),
    )
    wrong_seed = dataclasses.replace(
        outcome,
        restart_cache_snapshots=(changed, outcome.restart_cache_snapshots[1]),
    )
    with pytest.raises(ValueError, match="seed schedule"):
        validation_bootstrap_record_from_outcome(plan, row.row_key, wrong_seed)


def test_authenticated_bank_clones_one_bootstrap_identically_across_arms(tmp_path: Path) -> None:
    task = _prepared("candidate-a", "instance-a", "lineage-0")
    manifest, splits, prepared_manifest_sha = _prepared_census_authority(
        [task], partition="val"
    )
    plan = _plan([task])
    plan_sha = write_validation_bootstrap_plan(tmp_path / "plan.json", plan)
    for row in plan.census:
        record = validation_bootstrap_record_from_outcome(
            plan,
            row.row_key,
            _outcome(plan, row, initial_success=True),
        )
        publish_validation_bootstrap_record(tmp_path, plan, record)
    manifest_sha = seal_validation_bootstrap_bank(tmp_path, plan)
    with pytest.raises(FileExistsError, match="immutable"):
        seal_validation_bootstrap_bank(tmp_path, plan)
    loaded_plan = load_bootstrap_plan(
        tmp_path / "plan.json",
        expected_plan_sha256=plan_sha,
        prepared_tasks=[task],
        prepared_manifest=manifest,
        prepared_split_registry=splits,
        prepared_manifest_sha256=prepared_manifest_sha,
        protocol_registry_sha256=stable_digest({"grid": "bytes"}),
        protocol_record_digest=stable_digest({"grid": "rl-value-evaluation"}),
        same_support_contract_digest=plan.same_support_contract_digest,
        partition="val",
        evaluation_seed=RL_VALUE_EVALUATION_SEED,
        repetitions=RL_VALUE_REGISTERED_REPETITIONS,
        initializer=_FakeLAC(),
        runtime_implementation_manifest=_runtime_manifest(),
        config=_config(),
        context=_context(),
        allow_test_backend=True,
    )
    bank = load_validation_bootstrap_bank(
        tmp_path,
        plan=loaded_plan,
        expected_plan_sha256=plan_sha,
        expected_manifest_sha256=manifest_sha,
    )
    first = bank.clone_for_consumer(
        plan.census[0].row_key,
        consumer_id="rl-003:selected-policy",
        expected_same_support_contract_digest=plan.same_support_contract_digest,
    )
    second = bank.clone_for_consumer(
        plan.census[0].row_key,
        consumer_id="rl-003:return-initial",
        expected_same_support_contract_digest=plan.same_support_contract_digest,
    )

    assert first.bootstrap_record_digest == second.bootstrap_record_digest
    assert first.bootstrap_payload_sha256 == second.bootstrap_payload_sha256
    assert first.bootstrap_payload_bytes == second.bootstrap_payload_bytes
    assert first.bootstrap_record.as_dict() == second.bootstrap_record.as_dict()
    forbidden = {"ground_energy", "utility", "evaluator_hits"}

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | {item for nested in value.values() for item in keys(nested)}
        if isinstance(value, list):
            return {item for nested in value for item in keys(nested)}
        return set()

    assert not forbidden & keys(json.loads(first.bootstrap_payload_bytes))
    assert bank.access_receipt.denominator_count == len(plan.census)
    with pytest.raises(ValueError, match="same-support contract"):
        bank.clone_for_consumer(
            plan.census[0].row_key,
            consumer_id="rl-003:wrong-support",
            expected_same_support_contract_digest="0" * 64,
        )


def test_incomplete_census_cannot_be_sealed(tmp_path: Path) -> None:
    plan = _plan([_prepared("candidate-a", "instance-a", "lineage-0")])
    write_validation_bootstrap_plan(tmp_path / "plan.json", plan)
    for row in plan.census[:-1]:
        publish_validation_bootstrap_record(
            tmp_path,
            plan,
            validation_bootstrap_record_from_outcome(
                plan, row.row_key, _outcome(plan, row, initial_success=True)
            ),
        )

    with pytest.raises(ValueError, match="complete census"):
        seal_validation_bootstrap_bank(tmp_path, plan)
    assert not (tmp_path / "manifest.json").exists()


def test_published_record_is_no_replace(tmp_path: Path) -> None:
    plan = _plan([_prepared("candidate-a", "instance-a", "lineage-0")])
    write_validation_bootstrap_plan(tmp_path / "plan.json", plan)
    row = plan.census[0]
    record = validation_bootstrap_record_from_outcome(
        plan, row.row_key, _outcome(plan, row, initial_success=True)
    )
    publish_validation_bootstrap_record(tmp_path, plan, record)

    with pytest.raises(FileExistsError, match="immutable"):
        publish_validation_bootstrap_record(tmp_path, plan, record)


def test_bank_rejects_an_unregistered_top_level_artifact(tmp_path: Path) -> None:
    plan = _plan([_prepared("candidate-a", "instance-a", "lineage-0")])
    write_validation_bootstrap_plan(tmp_path / "plan.json", plan)
    for row in plan.census:
        publish_validation_bootstrap_record(
            tmp_path,
            plan,
            validation_bootstrap_record_from_outcome(
                plan, row.row_key, _outcome(plan, row, initial_success=True)
            ),
        )
    (tmp_path / "unregistered-targets.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unknown top-level artifact"):
        seal_validation_bootstrap_bank(tmp_path, plan)


def test_failed_rows_remain_sealed_denominator_rows_without_actor_or_cache(
    tmp_path: Path,
) -> None:
    task = _prepared("candidate-a", "instance-a", "lineage-0")
    plan = _plan([task])
    plan_sha = write_validation_bootstrap_plan(tmp_path / "plan.json", plan)
    for row in plan.census:
        publish_validation_bootstrap_record(
            tmp_path,
            plan,
            validation_bootstrap_record_from_outcome(
                plan,
                row.row_key,
                _outcome(plan, row, initial_success=row.repetition != 0),
            ),
        )
    manifest_sha = seal_validation_bootstrap_bank(tmp_path, plan)
    bank = load_validation_bootstrap_bank(
        tmp_path,
        plan=plan,
        expected_plan_sha256=plan_sha,
        expected_manifest_sha256=manifest_sha,
    )
    failed_key = bank.row_key("lineage-0", "instance-a", 0)
    failed = bank.clone_for_consumer(
        failed_key,
        consumer_id="rl-003:selected-policy",
        expected_same_support_contract_digest=plan.same_support_contract_digest,
    )

    assert bank.access_receipt.denominator_count == 4
    assert bank.access_receipt.initial_success_count == 3
    assert bank.access_receipt.initial_failure_count == 1
    assert failed.bootstrap_record.denominator_eligible is True
    assert failed.bootstrap_record.fixed_utility == 0.0
    assert failed.bootstrap_record.actor_invocation_permitted is False
    assert failed.bootstrap_record.cache_invocation_permitted is False


def test_tamper_and_manifest_pin_fail_closed(tmp_path: Path) -> None:
    task = _prepared("candidate-a", "instance-a", "lineage-0")
    plan = _plan([task])
    plan_sha = write_validation_bootstrap_plan(tmp_path / "plan.json", plan)
    for row in plan.census:
        publish_validation_bootstrap_record(
            tmp_path,
            plan,
            validation_bootstrap_record_from_outcome(
                plan, row.row_key, _outcome(plan, row, initial_success=True)
            ),
        )
    manifest_sha = seal_validation_bootstrap_bank(tmp_path, plan)

    with pytest.raises(ValueError, match="manifest SHA-256 pin"):
        load_validation_bootstrap_bank(
            tmp_path,
            plan=plan,
            expected_plan_sha256=plan_sha,
            expected_manifest_sha256="0" * 64,
        )

    record_path = next((tmp_path / "records").iterdir())
    original = record_path.read_bytes()
    record_path.chmod(0o644)
    record_path.write_bytes(original + b" \n")
    with pytest.raises(ValueError, match="canonical|manifest differs"):
        load_validation_bootstrap_bank(
            tmp_path,
            plan=plan,
            expected_plan_sha256=plan_sha,
            expected_manifest_sha256=manifest_sha,
        )


def test_manifest_rejects_boolean_disguised_as_zero_count(tmp_path: Path) -> None:
    plan = _plan([_prepared("candidate-a", "instance-a", "lineage-0")])
    plan_sha = write_validation_bootstrap_plan(tmp_path / "plan.json", plan)
    for row in plan.census:
        publish_validation_bootstrap_record(
            tmp_path,
            plan,
            validation_bootstrap_record_from_outcome(
                plan, row.row_key, _outcome(plan, row, initial_success=True)
            ),
        )
    seal_validation_bootstrap_bank(tmp_path, plan)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["initial_failure_count"] = False
    body = {key: value for key, value in manifest.items() if key != "record_digest"}
    manifest["record_digest"] = stable_digest(body)
    content = (
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("utf-8")
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(content)

    with pytest.raises(ValueError, match="integer counts"):
        load_validation_bootstrap_bank(
            tmp_path,
            plan=plan,
            expected_plan_sha256=plan_sha,
            expected_manifest_sha256=hashlib.sha256(content).hexdigest(),
        )


def test_external_pins_and_no_replace_are_mandatory(tmp_path: Path) -> None:
    task = _prepared("candidate-a", "instance-a", "lineage-0")
    manifest, splits, manifest_sha = _prepared_census_authority(
        [task], partition="val"
    )
    plan = _plan([task])
    path = tmp_path / "plan.json"
    plan_sha = write_validation_bootstrap_plan(path, plan)
    with pytest.raises(FileExistsError, match="immutable"):
        write_validation_bootstrap_plan(path, plan)
    with pytest.raises(ValueError, match="SHA-256"):
        load_bootstrap_plan(
            path,
            expected_plan_sha256="0" * 64,
            prepared_tasks=[task],
            prepared_manifest=manifest,
            prepared_split_registry=splits,
            prepared_manifest_sha256=manifest_sha,
            protocol_registry_sha256=stable_digest({"grid": "bytes"}),
            protocol_record_digest=stable_digest({"grid": "rl-value-evaluation"}),
            same_support_contract_digest=plan.same_support_contract_digest,
            partition="val",
            evaluation_seed=RL_VALUE_EVALUATION_SEED,
            repetitions=RL_VALUE_REGISTERED_REPETITIONS,
            initializer=_FakeLAC(),
            runtime_implementation_manifest=_runtime_manifest(),
            config=_config(),
            context=_context(),
            allow_test_backend=True,
        )
    assert hashlib.sha256(path.read_bytes()).hexdigest() == plan_sha
