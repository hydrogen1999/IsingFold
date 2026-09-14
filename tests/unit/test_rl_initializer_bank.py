"""Authenticated deployment-initializer snapshots for complete-system PPO training."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import networkx as nx
import pytest

import isingfold.rl.initializer_bank as initializer_bank_module
from isingfold.embedding import LogicalProblem
from isingfold.rl.complete_system import (
    COMPLETE_POLICY_RESTART_MODE,
    LAC_INITIALIZER_METHOD_ID,
    CompleteInitializerResult,
    CompleteSystemConfig,
    LACMinorminerInitializerBackend,
    PartialWorkVector,
)
from isingfold.rl.contracts import Context, DecisionState, WorkVector, stable_digest
from isingfold.rl.data.prepared import (
    PreparedDesignCondition,
    PreparedProvenance,
    PreparedTask,
)
from isingfold.rl.env import EmbeddingEnv, EmbeddingTask, fixed_strength_selector
from isingfold.rl.external import BackendIdentity, SearchStatus
from isingfold.rl.initializer_bank import (
    InitializerSnapshotUnavailable,
    build_initializer_bank_plan,
    environment_from_bootstrap_outcome,
    episode_bootstrap_outcome_from_payload,
    generate_initializer_bank,
    lac_runtime_implementation_manifest,
    load_initializer_bank,
    load_initializer_bank_plan,
    seal_initializer_bank,
    write_initializer_bank_plan,
)


def _runtime_manifest() -> dict[str, object]:
    body: dict[str, object] = {
        "schema": "lac-minorminer.runtime-implementation",
        "schema_version": 1,
        "native_extension_sha256": stable_digest({"artifact": "native-test-double"}),
        "python_source_sha256": {
            "__init__.py": stable_digest({"source": "init"}),
            "api.py": stable_digest({"source": "api"}),
        },
        "backend_info": {
            "backend": "lac_minorminer_cpp",
            "package_version": "0.1.0",
            "work_counter_schema": "lac-minorminer.native-work",
            "work_counter_version": 3,
        },
    }
    return {**body, "manifest_sha256": stable_digest(body)}


class _FakeLACInitializer:
    def __init__(
        self,
        results: list[CompleteInitializerResult],
        *,
        mutate_identity: bool = False,
    ) -> None:
        runtime = _runtime_manifest()
        self.runtime_implementation_manifest = runtime
        self.identity = BackendIdentity(
            method_id=LAC_INITIALIZER_METHOD_ID,
            distribution="lac-minorminer",
            version="0.1.0",
            entrypoint="lac_minorminer.find_embedding",
            implementation=(
                "lac_minorminer_cpp:runtime-sha256:" + str(runtime["manifest_sha256"])
            ),
        )
        self.results = list(results)
        self.calls: list[dict[str, object]] = []
        self.mutate_identity = mutate_identity

    def search(
        self,
        logical,
        host,
        *,
        seed: int,
        timeout_seconds: float,
        parameters,
        work_cap: WorkVector,
    ) -> CompleteInitializerResult:
        del logical, host
        self.calls.append(
            {
                "seed": seed,
                "timeout_seconds": timeout_seconds,
                "parameters": dict(parameters),
                "work_cap": work_cap,
            }
        )
        result = self.results.pop(0)
        result = dataclasses.replace(
            result,
            diagnostics={
                **dict(result.diagnostics),
                "runtime_implementation_manifest": self.runtime_implementation_manifest,
            },
        )
        if self.mutate_identity:
            self.identity = dataclasses.replace(self.identity, version="changed")
        return result


def _exact_work(**overrides: int) -> PartialWorkVector:
    values = WorkVector(route_expansions=2, materializations=1).as_dict()
    values.update(overrides)
    return PartialWorkVector.known(WorkVector(**values))


def _result(
    embedding: dict[int, frozenset[int]] | None,
    *,
    status: SearchStatus | None = None,
    work: PartialWorkVector | None = None,
) -> CompleteInitializerResult:
    resolved = status or (
        SearchStatus.EMBEDDING if embedding is not None else SearchStatus.NO_EMBEDDING
    )
    return CompleteInitializerResult(
        status=resolved,
        embedding=embedding,
        elapsed_seconds=0.0,
        work=work or _exact_work(),
        detail=resolved.value,
    )


def _config(*, attempts: int = 2) -> CompleteSystemConfig:
    return CompleteSystemConfig(
        online_wallclock_seconds=30.0,
        max_initializer_attempts=attempts,
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
    base = Context(qubit_cap=9)
    return dataclasses.replace(base, quotas={**dict(base.quotas), "restart": 0})


def _public_task(*, name: str, lineage: str) -> EmbeddingTask:
    logical = nx.path_graph(3)
    host = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))
    problem = LogicalProblem.from_dicts(
        {0: -1.0, 1: 0.5, 2: -0.25},
        {(0, 1): -1.0, (1, 2): 0.75},
    )
    stale = {0: frozenset({6}), 1: frozenset({7}), 2: frozenset({8})}
    return EmbeddingTask(
        name=name,
        logical=logical,
        host=host,
        problem=problem,
        ground_energy=None,
        lineage=lineage,
        initial_embedding=stale,
    )


def _provenance(*, lineage: str, partition: str = "train") -> PreparedProvenance:
    digest = stable_digest({"lineage": lineage, "partition": partition})
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
        fault_mask_sha256=stable_digest({"host": lineage, "kind": "artifact"}),
        calibration_status="not_applicable",
        calibration_sha256=None,
        distribution_regime="id",
        distribution_stratum="unit-test",
        source_partition=partition,
        source_release_id="unit-test-release",
        source_release_manifest_sha256=stable_digest({"release": 1}),
        split_manifest_sha256=stable_digest({"split": 1}),
        group_id=f"group-{lineage}",
        group_record_digest=stable_digest({"group": lineage}),
        instance_record_digest=stable_digest({"instance": lineage}),
        source_record_digest=stable_digest({"source": lineage}),
        record_digest=digest,
    )


def _condition(*, lineage: str, partition: str = "train") -> PreparedDesignCondition:
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
    *,
    task_id: str = "candidate-0",
    instance_id: str = "instance-0",
    lineage: str = "base-0",
    partition: str = "train",
) -> PreparedTask:
    return PreparedTask(
        task=_public_task(name=task_id, lineage=lineage),
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
        provenance=_provenance(lineage=lineage, partition=partition),
        design_condition=_condition(lineage=lineage, partition=partition),
    )


def _build_plan(
    prepared: list[PreparedTask],
    backend: _FakeLACInitializer,
    *,
    episode_count: int = 3,
    max_draws_per_episode: int = 2,
):
    return build_initializer_bank_plan(
        prepared,
        prepared_manifest_sha256=stable_digest({"prepared": "manifest"}),
        training_seed=1103,
        episode_schedule_start=0,
        episode_count=episode_count,
        max_draws_per_conditional_episode=max_draws_per_episode,
        initializer=backend,
        runtime_implementation_manifest=backend.runtime_implementation_manifest,
        config=_config(),
        context=_context(),
        allow_test_backend=True,
    )


def _bound_training_target(
    public: PreparedTask, *, prepared_manifest_sha256: str
) -> tuple[PreparedTask, dict[str, object], dict[str, object]]:
    certificate_digest = stable_digest({"certificate": public.instance_id})
    evaluator_protocol_digest = stable_digest({"protocol": "train-only"})
    attestation_digest = stable_digest({"publisher": "attestation"})
    evidence_digest = stable_digest({"evidence": "manifest-record"})
    evidence_sha = stable_digest({"evidence": "manifest-file"})
    claimed_energy = -2.75
    assert public.provenance is not None and public.design_condition is not None
    target_payload = {
        "certificate_digest": certificate_digest,
        "evaluator_protocol_digest": evaluator_protocol_digest,
        "instance_id": public.instance_id,
        "instance_record_digest": public.provenance.instance_record_digest,
        "learning_partition": "train",
        "reference_energy": claimed_energy,
        "reference_status": "exact",
        "schema": "isingfold.evaluator-target",
        "schema_version": 2,
    }
    target_digest = stable_digest(target_payload)
    target_set_digest = stable_digest(
        {
            "domain": "isingfold-partition-target-set-v1",
            "partition": "train",
            "targets": [
                {
                    "instance_id": public.instance_id,
                    "target_record_digest": target_digest,
                }
            ],
        }
    )
    target_sha = stable_digest({"targets": "train-file"})
    access_body: dict[str, object] = {
        "evidence_manifest_record_digest": evidence_digest,
        "evidence_manifest_sha256": evidence_sha,
        "opened_files": [
            {
                "authority_root": "prepared",
                "role": "evaluator-targets",
                "relative_path": "targets/train.jsonl",
                "sha256": target_sha,
            },
            {
                "authority_root": "publisher",
                "role": "quality-evidence-manifest",
                "relative_path": "evidence/manifest.json",
                "sha256": evidence_sha,
            },
        ],
        "partition": "train",
        "prepared_manifest_record_digest": stable_digest(
            {"prepared": "manifest-record"}
        ),
        "prepared_manifest_sha256": prepared_manifest_sha256,
        "publisher_attestation_digest": attestation_digest,
        "publisher_id": "unit-test-publisher",
        "target_authority_record_digest": stable_digest(
            {"target": "authority-record"}
        ),
        "target_count": 1,
        "target_path": "targets/train.jsonl",
        "target_set_digest": target_set_digest,
        "target_sha256": target_sha,
    }
    target_access = {**access_body, "record_digest": stable_digest(access_body)}
    result_body = {
        "accepted": True,
        "claimed_reference_energy": claimed_energy,
        "instance_id": public.instance_id,
        "reason_code": "accepted",
    }
    verifier_result = {**result_body, "record_digest": stable_digest(result_body)}
    target_row = {
        "artifact_path": "certificates/train/instance-0.json",
        "artifact_sha256": certificate_digest,
        "artifact_size_bytes": 1,
        "base_lineage_key": public.design_condition.base_lineage_key,
        "claimed_reference_energy": claimed_energy,
        "design_condition": dataclasses.asdict(public.design_condition),
        "evidence_kind": "exact",
        "instance_id": public.instance_id,
        "learning_partition": "train",
        "public_instance_record_digest": public.public_instance_record_digest,
        "reference_status": "exact",
        "request_record_digest": stable_digest({"request": public.instance_id}),
        "request_sha256": stable_digest({"request-file": public.instance_id}),
        "result_record_digest": verifier_result["record_digest"],
        "result_sha256": stable_digest({"result-file": public.instance_id}),
        "status": "accepted",
        "target_record_digest": target_digest,
        "verifier_result": verifier_result,
    }
    receipt_body: dict[str, object] = {
        "authority": target_access,
        "census": {
            "accepted_count": 1,
            "instance_set_digest": stable_digest([public.instance_id]),
            "target_count": 1,
        },
        "partition": "train",
        "protocol": "isingfold-ground-certificate-isolated-runtime-v2",
        "schema": "isingfold.ground-certificate-partition",
        "schema_version": 1,
        "targets": [target_row],
        "verifier": {"name": "unit-test-verifier"},
    }
    ground_receipt = {
        **receipt_body,
        "record_digest": stable_digest(receipt_body),
    }
    opened = dataclasses.replace(
        public,
        task=dataclasses.replace(public.task, ground_energy=claimed_energy),
        reference_status="exact",
        certificate_digest=certificate_digest,
        evaluator_protocol_digest=evaluator_protocol_digest,
        quality_attestation_digest=attestation_digest,
        quality_evidence_manifest_digest=evidence_digest,
        quality_evidence_manifest_sha256=evidence_sha,
        quality_target_set_digest=target_set_digest,
        quality_target_count=1,
    )
    return opened, target_access, ground_receipt


def test_plan_is_train_only_and_seals_the_exact_initializer_distribution() -> None:
    backend = _FakeLACInitializer([])
    plan = _build_plan([_prepared()], backend)

    assert plan.partition == "train"
    assert plan.training_seed == 1103
    assert plan.publication_eligible is False
    assert len(plan.episodes) == 3
    assert [row.episode_schedule_index for row in plan.episodes] == [0, 1, 2]
    assert all(row.instance_id == "instance-0" for row in plan.episodes)
    assert all(row.base_lineage == "base-0" for row in plan.episodes)
    assert all(len(row.draws) == 2 for row in plan.episodes)
    assert all(len(draw.attempt_seeds) == 2 for row in plan.episodes for draw in row.draws)
    assert plan.initializer_identity == backend.identity
    assert plan.runtime_implementation_digest == (
        backend.runtime_implementation_manifest["manifest_sha256"]
    )
    assert plan.config_digest == _config().digest
    assert "ground_energy" not in json.dumps(plan.as_dict())


def test_plan_v2_seals_two_upfront_restart_cache_slots() -> None:
    backend = _FakeLACInitializer([])
    context = dataclasses.replace(
        _context(),
        caps=dataclasses.replace(_context().caps, restart_work=5),
        quotas={**dict(_context().quotas), "restart": 1},
    )

    plan = build_initializer_bank_plan(
        [_prepared()],
        prepared_manifest_sha256=stable_digest({"prepared": "manifest"}),
        training_seed=1103,
        episode_schedule_start=0,
        episode_count=2,
        max_draws_per_conditional_episode=2,
        initializer=backend,
        runtime_implementation_manifest=backend.runtime_implementation_manifest,
        config=_config(),
        context=context,
        allow_test_backend=True,
    )

    assert plan.as_dict()["schema_version"] == 2
    assert plan.restart_cache_slots_per_episode == 2
    assert all(len(episode.restart_draws) == 2 for episode in plan.episodes)
    assert all(
        draw.slot_kind == "restart_cache" and draw.slot_index == slot_index
        for episode in plan.episodes
        for slot_index, draw in enumerate(episode.restart_draws)
    )
    assert len(
        {
            draw.system_seed
            for episode in plan.episodes
            for draw in (*episode.draws, *episode.restart_draws)
        }
    ) == 8


@pytest.mark.parametrize("partition", ["val", "test"])
def test_plan_rejects_validation_and_test_rows(partition: str) -> None:
    backend = _FakeLACInitializer([])

    with pytest.raises(ValueError, match="train partition"):
        _build_plan([_prepared(partition=partition)], backend)


def test_plan_rejects_any_open_evaluator_target() -> None:
    backend = _FakeLACInitializer([])
    row = _prepared()
    opened = dataclasses.replace(
        row,
        task=dataclasses.replace(row.task, ground_energy=-2.0),
        reference_status="exact",
        certificate_digest=stable_digest({"certificate": 1}),
        evaluator_protocol_digest=stable_digest({"protocol": 1}),
    )

    with pytest.raises(ValueError, match="evaluator target"):
        _build_plan([opened], backend)


def test_plan_rejects_missing_authenticated_public_policy_identity() -> None:
    backend = _FakeLACInitializer([])
    malformed = dataclasses.replace(
        _prepared(),
        public_instance_record_digest="not-a-sha256",
    )

    with pytest.raises(ValueError, match="public policy-instance record"):
        _build_plan([malformed], backend)


def test_plan_rejects_obsolete_prepared_v3_even_when_marked_production() -> None:
    backend = _FakeLACInitializer([])
    obsolete = dataclasses.replace(
        _prepared(),
        prepared_schema_version=3,
        corpus_scope="production-designed-v3",
    )

    with pytest.raises(ValueError, match="prepared schema v4"):
        _build_plan([obsolete], backend)


def test_plan_rejects_runtime_manifest_mutation_and_identity_drift() -> None:
    backend = _FakeLACInitializer([])
    mutated = dict(backend.runtime_implementation_manifest)
    mutated["native_extension_sha256"] = stable_digest({"different": "native"})

    with pytest.raises(ValueError, match="runtime implementation manifest digest"):
        build_initializer_bank_plan(
            [_prepared()],
            prepared_manifest_sha256=stable_digest({"prepared": "manifest"}),
            training_seed=1103,
            episode_schedule_start=0,
            episode_count=1,
            max_draws_per_conditional_episode=2,
            initializer=backend,
            runtime_implementation_manifest=mutated,
            config=_config(),
            context=_context(),
            allow_test_backend=True,
        )

    changed = dataclasses.replace(_config(), expected_initializer_version="9.9.9")
    with pytest.raises(ValueError, match="initializer version"):
        build_initializer_bank_plan(
            [_prepared()],
            prepared_manifest_sha256=stable_digest({"prepared": "manifest"}),
            training_seed=1103,
            episode_schedule_start=0,
            episode_count=1,
            max_draws_per_conditional_episode=2,
            initializer=backend,
            runtime_implementation_manifest=backend.runtime_implementation_manifest,
            config=changed,
            context=_context(),
            allow_test_backend=True,
        )


def test_schedule_is_lineage_equal_and_deterministic() -> None:
    backend = _FakeLACInitializer([])
    rows = [
        _prepared(task_id="a-0", instance_id="ia-0", lineage="base-a"),
        _prepared(task_id="a-1", instance_id="ia-1", lineage="base-a"),
        _prepared(task_id="b-0", instance_id="ib-0", lineage="base-b"),
    ]
    first = _build_plan(rows, backend, episode_count=8)
    second = _build_plan(list(reversed(rows)), backend, episode_count=8)

    assert first.record_digest == second.record_digest
    assert [row.base_lineage for row in first.episodes] == [
        "base-a",
        "base-b",
        "base-a",
        "base-b",
        "base-a",
        "base-b",
        "base-a",
        "base-b",
    ]
    assert {row.instance_id for row in first.episodes[::2]} == {"ia-0", "ia-1"}


def test_plan_record_digest_hashes_the_full_immutable_plan_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_stable_digest = initializer_bank_module.stable_digest
    plan_digest_inputs: list[object] = []

    def counting_stable_digest(value: object) -> str:
        if (
            isinstance(value, dict)
            and value.get("schema")
            == initializer_bank_module.INITIALIZER_BANK_PLAN_SCHEMA
        ):
            plan_digest_inputs.append(value)
        return real_stable_digest(value)

    monkeypatch.setattr(
        initializer_bank_module,
        "stable_digest",
        counting_stable_digest,
    )
    plan = _build_plan([_prepared()], _FakeLACInitializer([]))

    first = plan.record_digest
    second = plan.record_digest

    assert first == second
    assert len(plan_digest_inputs) == 1


def test_plan_record_digest_hashes_once_under_concurrent_python_312_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_stable_digest = initializer_bank_module.stable_digest
    simultaneous_digest_calls = threading.Barrier(2)
    call_count_lock = threading.Lock()
    plan_digest_call_count = 0

    def blocking_stable_digest(value: object) -> str:
        nonlocal plan_digest_call_count
        if (
            isinstance(value, dict)
            and value.get("schema")
            == initializer_bank_module.INITIALIZER_BANK_PLAN_SCHEMA
        ):
            with call_count_lock:
                plan_digest_call_count += 1
            try:
                simultaneous_digest_calls.wait(timeout=1.0)
            except threading.BrokenBarrierError:
                pass
        return real_stable_digest(value)

    monkeypatch.setattr(
        initializer_bank_module,
        "stable_digest",
        blocking_stable_digest,
    )
    plan = _build_plan([_prepared()], _FakeLACInitializer([]))

    with ThreadPoolExecutor(max_workers=2) as executor:
        digests = tuple(executor.map(lambda _: plan.record_digest, range(2)))

    assert len(set(digests)) == 1
    assert plan_digest_call_count == 1


def test_plan_freezes_directly_replaced_transitive_identity_mappings() -> None:
    plan = _build_plan([_prepared()], _FakeLACInitializer([]))
    mutable_scheduler: dict[str, object] = {"nested": ["original"]}
    replaced = dataclasses.replace(plan, scheduler=mutable_scheduler)

    digest_before_mutation = replaced.record_digest
    nested = mutable_scheduler["nested"]
    assert isinstance(nested, list)
    nested.append("mutated")
    mutable_scheduler["extra"] = True

    assert replaced.record_digest == digest_before_mutation
    payload = replaced.as_dict()
    assert replaced.record_digest == payload["record_digest"]
    assert replaced.record_digest == stable_digest(
        {key: value for key, value in payload.items() if key != "record_digest"}
    )
    assert dict(replaced.scheduler) == {"nested": ("original",)}


def test_plan_freezes_directly_replaced_transitive_identity_sequences() -> None:
    plan = _build_plan([_prepared()], _FakeLACInitializer([]))
    source_task_ids = list(plan.tasks[0].source_task_ids)
    attempt_seeds = list(plan.episodes[0].draws[0].attempt_seeds)
    replaced_task = dataclasses.replace(
        plan.tasks[0],
        source_task_ids=source_task_ids,
    )
    replaced_draw = dataclasses.replace(
        plan.episodes[0].draws[0],
        attempt_seeds=attempt_seeds,
    )
    episode_draws = [replaced_draw, *plan.episodes[0].draws[1:]]
    replaced_episode = dataclasses.replace(plan.episodes[0], draws=episode_draws)
    tasks = [replaced_task, *plan.tasks[1:]]
    episodes = [replaced_episode, *plan.episodes[1:]]
    replaced = dataclasses.replace(plan, tasks=tasks, episodes=episodes)

    digest_before_mutation = replaced.record_digest
    source_task_ids.clear()
    attempt_seeds.clear()
    episode_draws.clear()
    tasks.clear()
    episodes.clear()

    assert replaced.record_digest == digest_before_mutation
    payload = replaced.as_dict()
    assert replaced.record_digest == payload["record_digest"]
    assert replaced.record_digest == stable_digest(
        {key: value for key, value in payload.items() if key != "record_digest"}
    )
    assert isinstance(replaced.tasks, tuple)
    assert isinstance(replaced.tasks[0].source_task_ids, tuple)
    assert isinstance(replaced.episodes, tuple)
    assert isinstance(replaced.episodes[0].draws, tuple)
    assert isinstance(replaced.episodes[0].draws[0].attempt_seeds, tuple)


def test_plan_rejects_backend_identity_subclass_with_mutable_as_dict() -> None:
    class MutableBackendIdentity(BackendIdentity):
        mutable_identity_extension: list[str]

        def __init__(self, **kwargs: str) -> None:
            super().__init__(**kwargs)
            self.mutable_identity_extension = ["original"]

        def as_dict(self) -> dict[str, str | list[str]]:
            return {
                **super().as_dict(),
                "mutable_identity_extension": list(self.mutable_identity_extension),
            }

    backend = _FakeLACInitializer([])
    backend.identity = MutableBackendIdentity(**backend.identity.as_dict())

    with pytest.raises(TypeError, match="BackendIdentity"):
        _build_plan([_prepared()], backend)


def test_plan_has_an_externally_pinned_round_trip_for_resumable_workers(
    tmp_path: Path,
) -> None:
    prepared = [_prepared()]
    backend = _FakeLACInitializer([])
    plan = _build_plan(prepared, backend)
    path = tmp_path / "sealed-plan.json"

    plan_sha256 = write_initializer_bank_plan(path, plan)
    loaded = load_initializer_bank_plan(
        path,
        expected_plan_sha256=plan_sha256,
        prepared_tasks=prepared,
        prepared_manifest_sha256=stable_digest({"prepared": "manifest"}),
        initializer=backend,
        runtime_implementation_manifest=backend.runtime_implementation_manifest,
        config=_config(),
        context=_context(),
        allow_test_backend=True,
    )

    assert loaded.as_dict() == plan.as_dict()
    assert path.read_bytes().endswith(b"\n")
    with pytest.raises(ValueError, match="plan SHA-256 pin"):
        load_initializer_bank_plan(
            path,
            expected_plan_sha256="0" * 64,
            prepared_tasks=prepared,
            prepared_manifest_sha256=stable_digest({"prepared": "manifest"}),
            initializer=backend,
            runtime_implementation_manifest=backend.runtime_implementation_manifest,
            config=_config(),
            context=_context(),
            allow_test_backend=True,
        )


def _valid_embedding() -> dict[int, frozenset[int]]:
    return {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})}


def _larger_embedding() -> dict[int, frozenset[int]]:
    return {0: frozenset({0, 3}), 1: frozenset({4}), 2: frozenset({5})}


def _successful_cache_attempts() -> list[CompleteInitializerResult]:
    return [
        _result(_valid_embedding()),
        _result(None),
        _result(_larger_embedding()),
        _result(None),
    ]


def _generate_two_episode_bank(root: Path):
    prepared = [_prepared()]
    plan_backend = _FakeLACInitializer([])
    plan = _build_plan(prepared, plan_backend, episode_count=2)
    run_backend = _FakeLACInitializer(
        [
            _result(_larger_embedding()),
            _result(_valid_embedding()),
            *_successful_cache_attempts(),
            _result(None),
            _result(None),
            _result(_valid_embedding()),
            _result(None),
            *_successful_cache_attempts(),
        ]
    )
    generated = generate_initializer_bank(
        root,
        plan,
        prepared,
        initializer=run_backend,
        runtime_implementation_manifest=run_backend.runtime_implementation_manifest,
        config=_config(),
        context=_context(),
        episode_indices=[0],
        allow_test_backend=True,
    )
    assert [row.conditional_episode_index for row in generated] == [0, 0, 0]
    assert len(run_backend.calls) == 6
    generated = generate_initializer_bank(
        root,
        plan,
        prepared,
        initializer=run_backend,
        runtime_implementation_manifest=run_backend.runtime_implementation_manifest,
        config=_config(),
        context=_context(),
        episode_indices=[0, 1],
        allow_test_backend=True,
    )
    assert [row.conditional_episode_index for row in generated] == [
        0,
        0,
        0,
        1,
        1,
        1,
        1,
    ]
    assert len(run_backend.calls) == 14  # Episode zero was resumed, not rerun.
    manifest_sha256 = seal_initializer_bank(root, plan)
    return prepared, plan, plan_backend, manifest_sha256


def test_generation_seals_success_and_failure_for_every_restart_slot(
    tmp_path: Path,
) -> None:
    prepared = [_prepared()]
    base = _context()
    context = dataclasses.replace(
        base,
        caps=dataclasses.replace(base.caps, restart_work=4),
        quotas={**dict(base.quotas), "restart": 1},
    )
    config = _config(attempts=1)
    plan_backend = _FakeLACInitializer([])
    plan = build_initializer_bank_plan(
        prepared,
        prepared_manifest_sha256=stable_digest({"prepared": "manifest"}),
        training_seed=1103,
        episode_schedule_start=0,
        episode_count=1,
        max_draws_per_conditional_episode=2,
        initializer=plan_backend,
        runtime_implementation_manifest=plan_backend.runtime_implementation_manifest,
        config=config,
        context=context,
        allow_test_backend=True,
    )
    run_backend = _FakeLACInitializer(
        [
            _result(_valid_embedding()),
            _result(None),
            _result(_valid_embedding()),
            _result(None),
            _result(_valid_embedding()),
        ]
    )

    snapshots = generate_initializer_bank(
        tmp_path,
        plan,
        prepared,
        initializer=run_backend,
        runtime_implementation_manifest=run_backend.runtime_implementation_manifest,
        config=config,
        context=context,
        allow_test_backend=True,
    )

    assert [(row.slot_kind, row.slot_index) for row in snapshots] == [
        ("initial", 0),
        ("restart_cache", 0),
        ("restart_cache", 1),
    ]
    assert [row.success for row in snapshots] == [True, False, True]
    assert all(
        row.environment_budget_debit == row.initializer_work
        for row in snapshots[1:]
    )
    manifest_sha256 = seal_initializer_bank(tmp_path, plan)
    bank = load_initializer_bank(
        tmp_path,
        expected_manifest_sha256=manifest_sha256,
        prepared_tasks=prepared,
        prepared_manifest_sha256=stable_digest({"prepared": "manifest"}),
        initializer=plan_backend,
        runtime_implementation_manifest=plan_backend.runtime_implementation_manifest,
        config=config,
        context=context,
        allow_test_backend=True,
    )
    env = bank.environment(
        0,
        context=context,
        selector=fixed_strength_selector(),
    )
    decision = env.reset()
    assert isinstance(decision, DecisionState)
    assert env.state is not None
    assert env.state.budget_debit == sum(
        (row.initializer_work for row in snapshots),
        start=WorkVector(),
    )
    assert [slot.status for slot in env.state.restart_cache] == ["FAILED", "SUCCESS"]
    decoded = episode_bootstrap_outcome_from_payload(
        bank.bootstrap_outcome(0).as_dict()
    )
    assert decoded == bank.bootstrap_outcome(0)


def test_cache_fill_is_sequential_and_retains_budget_not_invoked_slot(
    tmp_path: Path,
) -> None:
    prepared = [_prepared()]
    base = _context()
    context = dataclasses.replace(
        base,
        caps=dataclasses.replace(base.caps, restart_work=2),
        quotas={**dict(base.quotas), "restart": 2},
    )
    config = _config(attempts=1)
    plan_backend = _FakeLACInitializer([])
    plan = build_initializer_bank_plan(
        prepared,
        prepared_manifest_sha256=stable_digest({"prepared": "manifest"}),
        training_seed=1103,
        episode_schedule_start=0,
        episode_count=1,
        max_draws_per_conditional_episode=1,
        initializer=plan_backend,
        runtime_implementation_manifest=plan_backend.runtime_implementation_manifest,
        config=config,
        context=context,
        allow_test_backend=True,
    )
    backend = _FakeLACInitializer(
        [_result(_valid_embedding()), _result(None)]
    )

    snapshots = generate_initializer_bank(
        tmp_path,
        plan,
        prepared,
        initializer=backend,
        runtime_implementation_manifest=backend.runtime_implementation_manifest,
        config=config,
        context=context,
        allow_test_backend=True,
    )

    assert len(backend.calls) == 2
    assert [snapshot.execution_status for snapshot in snapshots] == [
        "SUCCESS",
        "FAILED",
        "BUDGET_NOT_INVOKED",
    ]
    assert snapshots[2].attempts == ()
    assert snapshots[2].initializer_work == WorkVector()
    assert snapshots[1].work_before == snapshots[0].work_after
    assert snapshots[2].work_before == snapshots[1].work_after


def _load_bank(
    root: Path,
    prepared: list[PreparedTask],
    backend: _FakeLACInitializer,
    manifest_sha256: str,
):
    return load_initializer_bank(
        root,
        expected_manifest_sha256=manifest_sha256,
        prepared_tasks=prepared,
        prepared_manifest_sha256=stable_digest({"prepared": "manifest"}),
        initializer=backend,
        runtime_implementation_manifest=backend.runtime_implementation_manifest,
        config=_config(),
        context=_context(),
        allow_test_backend=True,
    )


def test_generation_is_resumable_and_records_exact_attempts_and_debit(tmp_path: Path) -> None:
    prepared, plan, backend, manifest_sha256 = _generate_two_episode_bank(tmp_path)
    bank = _load_bank(tmp_path, prepared, backend, manifest_sha256)

    first = bank.snapshot(0)
    assert first.success is True
    assert first.selected_attempt == 1
    assert len(first.attempts) == 2
    assert tuple(attempt.seed for attempt in first.attempts) == (
        plan.episodes[0].draws[0].attempt_seeds
    )
    assert first.initializer_work == WorkVector(
        route_expansions=4,
        materializations=2,
        validator_calls=2,
        restart_work=2,
    )
    assert first.environment_budget_debit == first.initializer_work
    assert first.selected_embedding_digest == first.attempts[1].embedding_digest

    second = bank.snapshot(1)
    assert second.success is True
    assert second.draw_index == 1
    failed_draw, successful_draw = bank.draws(1)
    assert failed_draw.success is False
    assert failed_draw.failure_reason == "INITIALIZER_NO_VALID_EMBEDDING"
    assert failed_draw.environment_budget_debit == WorkVector()
    assert failed_draw.initializer_work.restart_work == 2
    assert successful_draw == second

    assert bank.access_receipt.partition == "train"
    assert bank.access_receipt.opened_evaluator_targets is False
    assert bank.access_receipt.prepared_manifest_sha256 == plan.prepared_manifest_sha256
    assert bank.access_receipt.executed_draw_count == 7
    assert bank.access_receipt.failed_draw_count == 1
    assert bank.access_receipt.failure_record_digests == (failed_draw.record_digest,)
    assert bank.access_receipt.failed_draw_work == failed_draw.initializer_work
    assert bank.access_receipt.selected_initial_debit == (
        first.environment_budget_debit + second.environment_budget_debit
    )
    assert bank.access_receipt.generation_work == (
        bank.access_receipt.failed_draw_work
        + bank.access_receipt.selected_initial_debit
        + bank.access_receipt.restart_cache_fill_debit
    )


def test_bank_environment_matches_direct_precomputed_deployment_state(tmp_path: Path) -> None:
    prepared = [_prepared()]
    plan_backend = _FakeLACInitializer([])
    plan = _build_plan(prepared, plan_backend, episode_count=1)
    run_backend = _FakeLACInitializer(
        [
            _result(_larger_embedding()),
            _result(_valid_embedding()),
            *_successful_cache_attempts(),
        ]
    )
    generate_initializer_bank(
        tmp_path,
        plan,
        prepared,
        initializer=run_backend,
        runtime_implementation_manifest=run_backend.runtime_implementation_manifest,
        config=_config(),
        context=_context(),
        allow_test_backend=True,
    )
    manifest_sha256 = seal_initializer_bank(tmp_path, plan)
    bank = _load_bank(tmp_path, prepared, plan_backend, manifest_sha256)
    snapshot = bank.snapshot(0)
    bootstrap = bank.bootstrap_outcome(0)

    from_bank = bank.environment(
        0,
        context=_context(),
        selector=fixed_strength_selector(),
    )
    direct_task = dataclasses.replace(
        prepared[0].task,
        name="instance-0",
        ground_energy=None,
        initial_embedding=None,
    )
    direct = EmbeddingEnv(
        direct_task,
        from_bank.ctx,
        initializer=lambda logical, host, seed: _valid_embedding(),
        selector=fixed_strength_selector(),
        budget_debit=bootstrap.total_pre_policy_debit,
        initializer_precomputed=True,
        restart_cache=from_bank._restart_cache_template,
        restart_cache_manifest_digest=bank.access_receipt.manifest_record_digest,
        seed=snapshot.system_seed,
    )

    bank_state = from_bank.reset()
    direct_state = direct.reset()
    assert isinstance(bank_state, DecisionState)
    assert isinstance(direct_state, DecisionState)
    assert bank_state.state_fingerprint == direct_state.state_fingerprint
    assert from_bank.state is not None and direct.state is not None
    assert from_bank.state.chains == direct.state.chains == _valid_embedding()
    assert from_bank.state.remaining == direct.state.remaining
    assert from_bank.state.spent == direct.state.spent
    assert from_bank.state.budget_debit == bootstrap.total_pre_policy_debit
    with pytest.raises(Exception, match="more than once"):
        from_bank.reset()


def test_bootstrap_outcome_adapter_reconstructs_cached_deployment_state(
    tmp_path: Path,
) -> None:
    prepared = [_prepared()]
    plan_backend = _FakeLACInitializer([])
    plan = _build_plan(prepared, plan_backend, episode_count=1)
    run_backend = _FakeLACInitializer(
        [_result(_valid_embedding()), _result(None), *_successful_cache_attempts()]
    )
    generate_initializer_bank(
        tmp_path,
        plan,
        prepared,
        initializer=run_backend,
        runtime_implementation_manifest=run_backend.runtime_implementation_manifest,
        config=_config(),
        context=_context(),
        allow_test_backend=True,
    )
    manifest_sha256 = seal_initializer_bank(tmp_path, plan)
    bank = _load_bank(tmp_path, prepared, plan_backend, manifest_sha256)
    bootstrap = bank.bootstrap_outcome(0)
    task = dataclasses.replace(
        prepared[0].task,
        name=bootstrap.instance_id,
        initial_embedding=None,
    )

    adapted = environment_from_bootstrap_outcome(
        bootstrap,
        task=task,
        context=_context(),
        selector=fixed_strength_selector(),
    )
    reference = bank.environment(
        0,
        context=_context(),
        selector=fixed_strength_selector(),
    )

    adapted_decision = adapted.reset()
    reference_decision = reference.reset()
    assert isinstance(adapted_decision, DecisionState)
    assert isinstance(reference_decision, DecisionState)
    assert adapted_decision.state_fingerprint == reference_decision.state_fingerprint
    assert adapted.state is not None
    assert adapted.state.budget_debit == bootstrap.total_pre_policy_debit
    assert len(adapted.state.restart_cache) == 2


def test_generation_fails_closed_on_backend_identity_mutation(tmp_path: Path) -> None:
    prepared = [_prepared()]
    plan_backend = _FakeLACInitializer([])
    plan = _build_plan(prepared, plan_backend, episode_count=1)
    run_backend = _FakeLACInitializer(
        [_result(_valid_embedding()), _result(None)], mutate_identity=True
    )

    with pytest.raises(Exception, match="identity changed"):
        generate_initializer_bank(
            tmp_path,
            plan,
            prepared,
            initializer=run_backend,
            runtime_implementation_manifest=run_backend.runtime_implementation_manifest,
            config=_config(),
            context=_context(),
            allow_test_backend=True,
        )


def test_seal_rejects_missing_or_duplicate_episode_keys(tmp_path: Path) -> None:
    prepared = [_prepared()]
    backend = _FakeLACInitializer([])
    plan = _build_plan(prepared, backend, episode_count=2)
    run_backend = _FakeLACInitializer(
        [_result(_valid_embedding()), _result(None), *_successful_cache_attempts()]
    )
    generate_initializer_bank(
        tmp_path,
        plan,
        prepared,
        initializer=run_backend,
        runtime_implementation_manifest=run_backend.runtime_implementation_manifest,
        config=_config(),
        context=_context(),
        episode_indices=[0],
        allow_test_backend=True,
    )
    with pytest.raises(ValueError, match="missing initializer snapshots"):
        seal_initializer_bank(tmp_path, plan)

    snapshot_file = next((tmp_path / "snapshots").glob("*.json"))
    duplicate = snapshot_file.with_name(snapshot_file.stem + "-duplicate.json")
    shutil.copyfile(snapshot_file, duplicate)
    with pytest.raises(ValueError, match="duplicate initializer snapshot"):
        seal_initializer_bank(tmp_path, plan)


def test_seal_rejects_a_conditional_episode_when_all_draws_fail(tmp_path: Path) -> None:
    prepared = [_prepared()]
    backend = _FakeLACInitializer([])
    plan = _build_plan(prepared, backend, episode_count=1, max_draws_per_episode=2)
    run_backend = _FakeLACInitializer(
        [_result(None), _result(None), _result(None), _result(None)]
    )
    generated = generate_initializer_bank(
        tmp_path,
        plan,
        prepared,
        initializer=run_backend,
        runtime_implementation_manifest=run_backend.runtime_implementation_manifest,
        config=_config(),
        context=_context(),
        allow_test_backend=True,
    )

    assert [row.draw_index for row in generated] == [0, 1]
    assert all(not row.success for row in generated)
    with pytest.raises(InitializerSnapshotUnavailable, match="draw cap"):
        seal_initializer_bank(tmp_path, plan)


def test_loader_rejects_snapshot_mutation_missing_files_and_wrong_pin(tmp_path: Path) -> None:
    prepared, _, backend, manifest_sha256 = _generate_two_episode_bank(tmp_path)
    with pytest.raises(ValueError, match="manifest SHA-256 pin"):
        _load_bank(tmp_path, prepared, backend, "0" * 64)

    snapshot_file = sorted((tmp_path / "snapshots").glob("*.json"))[0]
    original = snapshot_file.read_bytes()
    payload = json.loads(original)
    payload["environment_budget_debit"]["restart_work"] += 1
    snapshot_file.chmod(0o644)
    snapshot_file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot SHA-256"):
        _load_bank(tmp_path, prepared, backend, manifest_sha256)

    snapshot_file.write_bytes(original)
    snapshot_file.chmod(0o444)
    victim = sorted((tmp_path / "snapshots").glob("*.json"))[1]
    victim.unlink()
    with pytest.raises(ValueError, match="missing initializer snapshot"):
        _load_bank(tmp_path, prepared, backend, manifest_sha256)


def test_loader_rejects_task_and_runtime_identity_drift(tmp_path: Path) -> None:
    prepared, _, backend, manifest_sha256 = _generate_two_episode_bank(tmp_path)
    changed_task = dataclasses.replace(
        prepared[0],
        task=dataclasses.replace(
            prepared[0].task,
            host=nx.path_graph(9),
        ),
    )
    with pytest.raises(ValueError, match="plan differs"):
        _load_bank(tmp_path, [changed_task], backend, manifest_sha256)

    drifted = _FakeLACInitializer([])
    drifted.runtime_implementation_manifest = {
        **_runtime_manifest(),
        "native_extension_sha256": stable_digest({"artifact": "other"}),
    }
    with pytest.raises(ValueError, match="runtime implementation manifest digest"):
        _load_bank(tmp_path, prepared, drifted, manifest_sha256)


def test_loader_reads_only_sealed_bank_files_and_never_evaluator_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared, _, backend, manifest_sha256 = _generate_two_episode_bank(tmp_path)
    observed: list[Path] = []
    original_read_bytes = Path.read_bytes

    def recording_read_bytes(path: Path) -> bytes:
        observed.append(path)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", recording_read_bytes)
    bank = _load_bank(tmp_path, prepared, backend, manifest_sha256)

    assert len(bank) == 2
    assert observed
    assert all(path.is_relative_to(tmp_path) for path in observed)
    assert all("evaluator_targets" not in path.name for path in observed)


def test_loader_materializes_one_snapshot_lookup_for_all_episode_resolutions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, plan, backend, manifest_sha256 = _generate_two_episode_bank(tmp_path)
    observed_lookups: list[object] = []
    real_episode_resolution = initializer_bank_module._episode_resolution
    real_restart_resolution = initializer_bank_module._restart_resolution

    def recording_episode_resolution(episode, files, *, plan):
        observed_lookups.append(files)
        return real_episode_resolution(episode, files, plan=plan)

    def recording_restart_resolution(
        episode,
        files,
        *,
        plan,
        initial_snapshot,
    ):
        observed_lookups.append(files)
        return real_restart_resolution(
            episode,
            files,
            plan=plan,
            initial_snapshot=initial_snapshot,
        )

    monkeypatch.setattr(
        initializer_bank_module,
        "_episode_resolution",
        recording_episode_resolution,
    )
    monkeypatch.setattr(
        initializer_bank_module,
        "_restart_resolution",
        recording_restart_resolution,
    )

    bank = _load_bank(tmp_path, prepared, backend, manifest_sha256)

    assert len(bank) == len(plan.episodes)
    assert len(observed_lookups) == 2 * len(plan.episodes)
    assert all(lookup is observed_lookups[0] for lookup in observed_lookups)


def test_loader_rejects_a_redigested_unregistered_snapshot_row_and_file(
    tmp_path: Path,
) -> None:
    prepared, _, backend, _ = _generate_two_episode_bank(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    source_row = next(row for row in manifest["snapshot_files"] if row["success"])
    source_snapshot_path = tmp_path / source_row["filename"]
    unknown_snapshot = json.loads(source_snapshot_path.read_bytes())
    unknown_snapshot["draw_key"] = "f" * 64
    unknown_snapshot_body = {
        key: value
        for key, value in unknown_snapshot.items()
        if key != "record_digest"
    }
    unknown_snapshot["record_digest"] = stable_digest(unknown_snapshot_body)
    unknown_filename = (
        f"snapshots/{unknown_snapshot['conditional_episode_index']:012d}-"
        f"{unknown_snapshot['slot_kind']}-{unknown_snapshot['slot_index']:04d}-"
        f"{unknown_snapshot['draw_key']}-{unknown_snapshot['record_digest']}.json"
    )
    unknown_snapshot_raw = (
        json.dumps(
            unknown_snapshot,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    (tmp_path / unknown_filename).write_bytes(unknown_snapshot_raw)

    unknown_row = {
        **source_row,
        "draw_key": unknown_snapshot["draw_key"],
        "filename": unknown_filename,
        "record_digest": unknown_snapshot["record_digest"],
        "sha256": hashlib.sha256(unknown_snapshot_raw).hexdigest(),
    }
    manifest["snapshot_files"].append(unknown_row)
    manifest["snapshot_files"].sort(
        key=lambda row: (
            row["conditional_episode_index"],
            row["slot_kind"] != "initial",
            row["slot_index"],
        )
    )
    manifest["snapshot_root_digest"] = stable_digest(manifest["snapshot_files"])
    manifest_body = {
        key: value for key, value in manifest.items() if key != "record_digest"
    }
    manifest["record_digest"] = stable_digest(manifest_body)
    manifest_raw = (
        json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    manifest_path.chmod(0o644)
    manifest_path.write_bytes(manifest_raw)

    with pytest.raises(ValueError, match="unregistered draw"):
        _load_bank(
            tmp_path,
            prepared,
            backend,
            hashlib.sha256(manifest_raw).hexdigest(),
        )


def test_failed_draw_is_resumed_without_rerun_and_does_not_change_the_task(
    tmp_path: Path,
) -> None:
    prepared = [_prepared()]
    plan_backend = _FakeLACInitializer([])
    plan = _build_plan(prepared, plan_backend, episode_count=1)
    interrupted = _FakeLACInitializer([_result(None), _result(None)])
    with pytest.raises(Exception, match="initializer raised IndexError"):
        generate_initializer_bank(
            tmp_path,
            plan,
            prepared,
            initializer=interrupted,
            runtime_implementation_manifest=interrupted.runtime_implementation_manifest,
            config=_config(),
            context=_context(),
            allow_test_backend=True,
        )
    persisted = sorted((tmp_path / "snapshots").glob("*.json"))
    assert len(persisted) == 1
    failed_digest = json.loads(persisted[0].read_bytes())["record_digest"]

    resumed = _FakeLACInitializer(
        [_result(_valid_embedding()), _result(None), *_successful_cache_attempts()]
    )
    rows = generate_initializer_bank(
        tmp_path,
        plan,
        prepared,
        initializer=resumed,
        runtime_implementation_manifest=resumed.runtime_implementation_manifest,
        config=_config(),
        context=_context(),
        allow_test_backend=True,
    )
    assert [
        (row.draw_index, row.success)
        for row in rows
        if row.slot_kind == "initial"
    ] == [(0, False), (1, True)]
    assert rows[0].record_digest == failed_digest
    assert all(row.instance_id == "instance-0" for row in rows)
    assert len(resumed.calls) == 6


def test_conditional_failure_skipping_preserves_lineage_equal_episode_weight(
    tmp_path: Path,
) -> None:
    prepared = [
        _prepared(task_id="a", instance_id="ia", lineage="base-a"),
        _prepared(task_id="b", instance_id="ib", lineage="base-b"),
    ]
    plan_backend = _FakeLACInitializer([])
    plan = _build_plan(prepared, plan_backend, episode_count=4)
    run_backend = _FakeLACInitializer(
        [
            _result(None),
            _result(None),
            _result(_valid_embedding()),
            _result(None),
            *_successful_cache_attempts(),
            _result(_valid_embedding()),
            _result(None),
            *_successful_cache_attempts(),
            _result(_valid_embedding()),
            _result(None),
            *_successful_cache_attempts(),
            _result(_valid_embedding()),
            _result(None),
            *_successful_cache_attempts(),
        ]
    )
    generate_initializer_bank(
        tmp_path,
        plan,
        prepared,
        initializer=run_backend,
        runtime_implementation_manifest=run_backend.runtime_implementation_manifest,
        config=_config(),
        context=_context(),
        allow_test_backend=True,
    )
    manifest_sha256 = seal_initializer_bank(tmp_path, plan)
    bank = _load_bank(tmp_path, prepared, plan_backend, manifest_sha256)

    assert [bank.snapshot(index).base_lineage for index in range(4)] == [
        "base-a",
        "base-b",
        "base-a",
        "base-b",
    ]
    assert [bank.snapshot(index).draw_index for index in range(4)] == [1, 0, 0, 0]
    assert bank.access_receipt.failed_draw_count == 1


def test_loader_recomputes_resolution_mapping_even_if_mutation_is_repinned(
    tmp_path: Path,
) -> None:
    prepared, _, backend, _ = _generate_two_episode_bank(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.chmod(0o644)
    payload = json.loads(manifest_path.read_bytes())
    payload["resolutions"][1]["failed_draw_count"] = 0
    payload["resolution_digest"] = stable_digest(payload["resolutions"])
    body = {key: value for key, value in payload.items() if key != "record_digest"}
    payload["record_digest"] = stable_digest(body)
    raw = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()
    manifest_path.write_bytes(raw)
    manifest_path.chmod(0o444)

    with pytest.raises(ValueError, match="failure accounting"):
        _load_bank(
            tmp_path,
            prepared,
            backend,
            hashlib.sha256(raw).hexdigest(),
        )


def test_production_runtime_accessor_is_read_only_and_makes_plan_eligible() -> None:
    backend = LACMinorminerInitializerBackend()
    runtime = lac_runtime_implementation_manifest(backend)
    config = dataclasses.replace(
        _config(),
        expected_initializer_version=backend.identity.version,
    )
    plan = build_initializer_bank_plan(
        [_prepared()],
        prepared_manifest_sha256=stable_digest({"prepared": "manifest"}),
        training_seed=1103,
        episode_schedule_start=0,
        episode_count=1,
        max_draws_per_conditional_episode=2,
        initializer=backend,
        runtime_implementation_manifest=runtime,
        config=config,
        context=_context(),
    )

    assert plan.publication_eligible is True
    assert runtime["native_extension_sha256"]
    with pytest.raises(TypeError):
        runtime["native_extension_sha256"] = "0" * 64  # type: ignore[index]


def test_environment_accepts_only_a_bound_train_target_after_bank_authentication(
    tmp_path: Path,
) -> None:
    prepared = [_prepared()]
    plan_backend = _FakeLACInitializer([])
    plan = _build_plan(prepared, plan_backend, episode_count=1)
    run_backend = _FakeLACInitializer(
        [_result(_valid_embedding()), _result(None), *_successful_cache_attempts()]
    )
    generate_initializer_bank(
        tmp_path,
        plan,
        prepared,
        initializer=run_backend,
        runtime_implementation_manifest=run_backend.runtime_implementation_manifest,
        config=_config(),
        context=_context(),
        allow_test_backend=True,
    )
    manifest_sha256 = seal_initializer_bank(tmp_path, plan)
    bank = _load_bank(tmp_path, prepared, plan_backend, manifest_sha256)
    opened_train, target_access, ground_receipt = _bound_training_target(
        prepared[0],
        prepared_manifest_sha256=plan.prepared_manifest_sha256,
    )

    with pytest.raises(ValueError, match="requires authenticated target-access"):
        bank.environment(
            0,
            context=_context(),
            selector=fixed_strength_selector(),
            training_task=opened_train,
            ground_partition_receipt=ground_receipt,
        )
    with pytest.raises(ValueError, match="requires authenticated target-access"):
        bank.environment(
            0,
            context=_context(),
            selector=fixed_strength_selector(),
            training_task=opened_train,
            target_access=target_access,
        )

    env = bank.environment(
        0,
        context=_context(),
        selector=fixed_strength_selector(),
        training_task=opened_train,
        target_access=target_access,
        ground_partition_receipt=ground_receipt,
    )
    assert env.task.ground_energy == -2.75
    assert "-2.75" not in json.dumps(bank.access_receipt.as_dict())

    validation_row = _prepared(partition="val")
    validation_row = dataclasses.replace(
        validation_row,
        task=dataclasses.replace(validation_row.task, ground_energy=-2.75),
    )
    with pytest.raises(ValueError, match="partition or public identity"):
        bank.environment(
            0,
            context=_context(),
            selector=fixed_strength_selector(),
            training_task=validation_row,
            target_access=target_access,
            ground_partition_receipt=ground_receipt,
        )

    mutations = (
        dataclasses.replace(
            opened_train,
            task=dataclasses.replace(opened_train.task, ground_energy=-9.5),
        ),
        dataclasses.replace(
            opened_train,
            certificate_digest=stable_digest({"certificate": "mutated"}),
        ),
        dataclasses.replace(
            opened_train,
            evaluator_protocol_digest=stable_digest({"protocol": "mutated"}),
        ),
        dataclasses.replace(
            opened_train,
            public_instance_record_digest=stable_digest(
                {"policy-instance": "mutated"}
            ),
        ),
    )
    for mutation in mutations:
        with pytest.raises(ValueError, match="target|quality|ground energy"):
            bank.environment(
                0,
                context=_context(),
                selector=fixed_strength_selector(),
                training_task=mutation,
                target_access=target_access,
                ground_partition_receipt=ground_receipt,
            )
