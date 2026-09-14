"""Honest complete-system evaluation for the learned Profile-I arm."""

from __future__ import annotations

import dataclasses
import itertools
import json
import random
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest

from tests.unit.evaluation_strata_support import evaluation_contract
from isingfold.embedding import LogicalProblem
from isingfold.rl.complete_system import (
    COMPLETE_POPULATION_TASK_DIGEST_SCHEMA,
    CompleteInitializerResult,
    CompletePopulationIdentity,
    CompleteSystemConfig,
    FrozenComponentIdentity,
    LACMinorminerInitializerBackend,
    PartialWorkVector,
    complete_policy_context,
    complete_system_metrics,
    complete_system_method_metadata,
    pair_with_stock_minorminer,
    read_complete_system_evidence,
    read_complete_system_outcomes,
    read_complete_system_receipts,
    run_complete_system,
    task_population_digest,
    verify_terminal_evidence,
    verify_complete_system_receipts,
    write_complete_system_evidence,
    write_complete_system_outcomes,
    write_complete_system_receipts,
)
from isingfold.rl.complete_system import (
    _context_digest,
    _lac_profile_detail_consistent,
    _repair_target_satisfied,
)
from isingfold.rl.contracts import Context, Opcode, WorkVector, stable_digest
from isingfold.rl.env import EmbeddingTask, fixed_strength_selector
from isingfold.rl.evaluator import ReadBlock
from isingfold.rl.external import (
    BackendIdentity,
    BackendSearchResult,
    ExternalBaselineConfig,
    SearchStatus,
    run_external_system,
)
from isingfold.rl.validation_bootstrap_bank import (
    RL_VALUE_EVALUATION_SEED,
    bootstrap_execution_record_digest,
    load_validation_bootstrap_bank,
    publish_validation_bootstrap_record,
    seal_validation_bootstrap_bank,
    validation_bootstrap_record_from_outcome,
    write_validation_bootstrap_plan,
)
from tests.unit.test_rl_validation_bootstrap_bank import (
    _FakeLAC as ValidationFakeLAC,
    _config as validation_config,
    _context as validation_context,
    _outcome as validation_outcome,
    _plan as validation_plan,
    _prepared as validation_prepared,
)


def _tiny_task(*, prepared_initial: bool = True) -> EmbeddingTask:
    host = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))
    logical = nx.path_graph(3)
    rng = random.Random(7)
    h = {node: rng.choice((-1.0, 1.0)) for node in logical}
    j = {edge: rng.choice((-1.0, 1.0)) for edge in logical.edges()}
    problem = LogicalProblem.from_dicts(h, j)
    nodes = tuple(logical)
    ground = min(
        sum(h[node] * spins[node] for node in nodes)
        + sum(weight * spins[left] * spins[right] for (left, right), weight in j.items())
        for values in itertools.product((-1, 1), repeat=len(nodes))
        for spins in ({node: value for node, value in zip(nodes, values, strict=True)},)
    )
    # This is deliberately invalid.  A complete-system runner must never replay it.
    stale = {0: frozenset({8}), 1: frozenset({8}), 2: frozenset({8})}
    return EmbeddingTask(
        "complete-tiny",
        logical,
        host,
        problem,
        ground,
        lineage="base-0",
        initial_embedding=stale if prepared_initial else None,
    )


def _partial_native_work(*, route_expansions: int | None = None) -> PartialWorkVector:
    return PartialWorkVector(
        decisions=0,
        route_expansions=route_expansions,
        materializations=0,
        compiler_calls=0,
        validator_calls=0,
        cut_edge_visits=0,
        restart_work=0,
        evaluator_reads=0,
        feature_work=0,
    )


class FakeInitializer:
    def __init__(self, results: list[CompleteInitializerResult]) -> None:
        self.results = list(results)
        self.calls: list[tuple[int, float]] = []
        self.identity = BackendIdentity(
            method_id="named-greedy-initializer-v2",
            distribution="isingfold",
            version="2.1.0",
            entrypoint="tests.FakeInitializer.search",
            implementation="injected-test-double",
        )

    def search(self, logical, host, *, seed: int, timeout_seconds: float, parameters):
        del logical, host, parameters
        self.calls.append((seed, timeout_seconds))
        return self.results.pop(0)


class FakeExternal:
    def __init__(self, embedding) -> None:
        self.embedding = embedding
        self.identity = BackendIdentity(
            method_id="stock-minorminer-restarts-v1",
            distribution="minorminer",
            version="0.2.22",
            entrypoint="minorminer.find_embedding",
            implementation="injected-test-double",
        )

    def search(self, logical, host, *, seed: int, timeout_seconds: float, parameters):
        del logical, host, seed, timeout_seconds, parameters
        return BackendSearchResult(SearchStatus.EMBEDDING, self.embedding, 0.001)


def _component(name: str) -> FrozenComponentIdentity:
    return FrozenComponentIdentity(
        component_id=name,
        version="checkpoint-v4",
        implementation=f"tests.{name}",
        artifact_sha256=stable_digest({"component": name, "version": 4}),
    )


def _population(
    task: EmbeddingTask, *, repetitions: int = 1, evaluation_seed: int = 0
) -> CompletePopulationIdentity:
    identities = ((task.lineage or task.name, task.name),)
    strata, design = evaluation_contract(identities)
    return CompletePopulationIdentity(
        population_id="sealed-test-population",
        source_manifest_sha256=stable_digest({"manifest": "all-policy-instances"}),
        task_payload_sha256=task_population_digest([task]),
        expected_instances=identities,
        expected_repetitions=repetitions,
        evaluation_seed=evaluation_seed,
        evaluation_strata=strata,
        confirmatory_design=design,
    )


def test_population_digest_excludes_sealed_evaluator_target() -> None:
    """The pre-outcome population commitment must use public problem content only."""

    task = _tiny_task()
    hidden_target = dataclasses.replace(task, ground_energy=None)

    assert task_population_digest([task]) == task_population_digest([hidden_target])
    assert COMPLETE_POPULATION_TASK_DIGEST_SCHEMA.endswith("-v2")


def _config(*, attempts: int = 1) -> CompleteSystemConfig:
    return CompleteSystemConfig(
        online_wallclock_seconds=30.0,
        max_initializer_attempts=attempts,
        audit_reads=4096,
        selection_rule="resource-lexicographic",
        online_evaluator_feedback=False,
        initializer_backend="isingfold",
        initializer_method_id="named-greedy-initializer-v2",
        expected_initializer_version="2.1.0",
        policy_restart_mode="disabled-no-native-replay-v1",
        initializer_parameters={"tries": 1},
    )


def _external_config(*, attempts: int = 1) -> ExternalBaselineConfig:
    return ExternalBaselineConfig.from_mapping(
        {
            "schema": "isingfold.external-baseline-config",
            "schema_version": 1,
            "backend": "stock-minorminer",
            "expected_backend_version": "0.2.22",
            "embedding_wallclock_seconds": 30.0,
            "max_restarts": attempts,
            "audit_reads": 4096,
            "selection_rule": "resource-lexicographic",
            "online_evaluator_feedback": False,
            "minorminer_parameters": {
                "tries": 1,
                "threads": 1,
                "max_no_improvement": 10,
                "chainlength_patience": 10,
            },
        }
    )


def test_complete_system_config_is_strict_and_round_trips() -> None:
    config = _config()

    assert CompleteSystemConfig.from_mapping(config.as_dict()) == config
    incomplete = config.as_dict()
    incomplete.pop("policy_restart_mode")
    with pytest.raises(ValueError, match="schema differs"):
        CompleteSystemConfig.from_mapping(incomplete)


def test_lac_profile_detail_rejects_unknown_or_activation_inconsistent_values() -> None:
    common = {
        "success": False,
        "termination": "tries_exhausted",
        "search_profile": "hybrid_chimera_clique_v1",
        "structural_fallback_invoked": True,
    }

    assert not _lac_profile_detail_consistent(
        profile_detail="v0_failure_chimera_clique_inapplicable_forged",
        **common,
    )
    assert not _lac_profile_detail_consistent(
        success=True,
        termination="success",
        search_profile="hybrid_chimera_clique_v1",
        profile_detail="v0_success",
        structural_fallback_invoked=True,
    )


def _commit_initial(decision, rng) -> int:
    del rng
    for index, (candidate, legal) in enumerate(
        zip(decision.candidates, decision.legal_mask, strict=True)
    ):
        if legal and candidate.opcode is Opcode.COMMIT and candidate.archive_ref == 0:
            return index
    raise AssertionError("protected initializer was not offered for COMMIT")


def _sealed_validation_bank(tmp_path: Path, *, initial_success: bool):
    prepared = validation_prepared("candidate-a", "instance-a", "lineage-0")
    plan = validation_plan([prepared])
    plan_sha = write_validation_bootstrap_plan(tmp_path / "plan.json", plan)
    for row in plan.census:
        outcome = validation_outcome(plan, row, initial_success=initial_success)

        def enrich(snapshot):
            attempts = tuple(
                dataclasses.replace(
                    attempt,
                    backend_diagnostics={
                        "backend": "lac_minorminer_cpp",
                        "package_version": "0.1.0",
                        "random_seed": attempt.seed,
                        "elapsed_seconds": attempt.reported_seconds,
                        "success": attempt.status == "VALID_CANDIDATE",
                        "transitions": attempt.backend_work.decisions,
                        "trace": [],
                        "incumbents": (
                            [{"generation": 0}]
                            if attempt.status == "VALID_CANDIDATE"
                            else []
                        ),
                        "termination_reason": (
                            "success"
                            if attempt.status == "VALID_CANDIDATE"
                            else "tries_exhausted"
                        ),
                        "work_budget_exhausted_coordinate": None,
                        "work_counter_schema": "lac-minorminer.native-work",
                        "work_counter_version": 3,
                        "search_profile": "hybrid_chimera_clique_v1",
                        "profile_detail": (
                            "v0_success"
                            if attempt.status == "VALID_CANDIDATE"
                            else (
                                "v0_failure_chimera_clique_inapplicable_"
                                "source_not_complete"
                            )
                        ),
                        "structural_fallback_invoked": (
                            attempt.status != "VALID_CANDIDATE"
                        ),
                        "work": attempt.backend_work.as_dict(),
                        "work_cap": validation_context().caps.as_dict(),
                        "runtime_implementation_manifest": (
                            ValidationFakeLAC().runtime_implementation_manifest
                        ),
                    },
                )
                for attempt in snapshot.attempts
            )
            return dataclasses.replace(snapshot, attempts=attempts)

        enriched_initial = enrich(outcome.initial_snapshot)
        enriched_cache = tuple(enrich(item) for item in outcome.restart_cache_snapshots)
        outcome = dataclasses.replace(
            outcome,
            initial_snapshot=enriched_initial,
            restart_cache_snapshots=enriched_cache,
            manifest_record_digest=bootstrap_execution_record_digest(
                plan,
                row.row_key,
                enriched_initial,
                enriched_cache,
            ),
        )
        record = validation_bootstrap_record_from_outcome(
            plan,
            row.row_key,
            outcome,
        )
        publish_validation_bootstrap_record(tmp_path, plan, record)
    manifest_sha = seal_validation_bootstrap_bank(tmp_path, plan)
    bank = load_validation_bootstrap_bank(
        tmp_path,
        plan=plan,
        expected_plan_sha256=plan_sha,
        expected_manifest_sha256=manifest_sha,
    )
    task = dataclasses.replace(
        prepared.task,
        name=prepared.instance_id,
        ground_energy=-2.0,
        initial_embedding=None,
    )
    return task, plan, bank


def test_k2_complete_system_consumes_validation_cache_without_native_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, plan, bank = _sealed_validation_bank(tmp_path, initial_success=True)
    backend = ValidationFakeLAC()  # Deliberately has no search method.
    calls = 0

    def controller(decision, rng):
        nonlocal calls
        calls += 1
        return _commit_initial(decision, rng)

    monkeypatch.setattr(
        "isingfold.rl.complete_system.sample_program",
        lambda program, *args, num_reads, **kwargs: ReadBlock(
            3072, num_reads, 0.0, 0.0, program.strength_index
        ),
    )
    ctx = validation_context()
    outcomes, receipts = run_complete_system(
        [task],
        ctx,
        backend,
        validation_config(),
        controller=controller,
        selector=fixed_strength_selector(),
        controller_identity=_component("cached-controller"),
        selector_identity=_component("selector"),
        population=_population(
            task,
            repetitions=4,
            evaluation_seed=RL_VALUE_EVALUATION_SEED,
        ),
        seed=RL_VALUE_EVALUATION_SEED,
        repetitions=4,
        validation_bootstrap_bank=bank,
        same_support_contract_digest=plan.same_support_contract_digest,
        bootstrap_consumer_id="if-core:selected-policy:seed-0",
    )

    assert calls == 4
    assert all(outcome.returned_valid for outcome in outcomes)
    assert all(receipt.bootstrap_binding is not None for receipt in receipts)
    assert all(receipt.policy_context_digest == _context_digest(ctx) for receipt in receipts)
    assert all(
        receipt.environment_budget_debit
        == receipt.initializer_work.to_work_vector()
        for receipt in receipts
    )
    assert all(receipt.initializer_work.restart_work == 3 for receipt in receipts)


def test_k2_initial_failure_is_zero_and_never_calls_actor_or_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task, plan, bank = _sealed_validation_bank(tmp_path, initial_success=False)
    backend = ValidationFakeLAC()

    def forbidden(*args, **kwargs):
        raise AssertionError("initial failure must not invoke actor or evaluator")

    monkeypatch.setattr("isingfold.rl.complete_system.sample_program", forbidden)
    outcomes, receipts = run_complete_system(
        [task],
        validation_context(),
        backend,
        validation_config(),
        controller=forbidden,
        selector=fixed_strength_selector(),
        controller_identity=_component("cached-controller"),
        selector_identity=_component("selector"),
        population=_population(
            task,
            repetitions=4,
            evaluation_seed=RL_VALUE_EVALUATION_SEED,
        ),
        seed=RL_VALUE_EVALUATION_SEED,
        repetitions=4,
        validation_bootstrap_bank=bank,
        same_support_contract_digest=plan.same_support_contract_digest,
        bootstrap_consumer_id="if-core:selected-policy:seed-0",
    )

    assert all(outcome.population_eligible for outcome in outcomes)
    assert all(outcome.utility == 0.0 and outcome.controller_calls == 0 for outcome in outcomes)
    assert all(receipt.environment_budget_debit == WorkVector() for receipt in receipts)
    assert all(receipt.bootstrap_binding is not None for receipt in receipts)


def test_complete_system_runs_named_initializer_and_never_replays_prepared_state(
    monkeypatch,
) -> None:
    task = _tiny_task(prepared_initial=True)
    actual = {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})}
    backend = FakeInitializer(
        [
            CompleteInitializerResult(
                SearchStatus.EMBEDDING,
                actual,
                elapsed_seconds=0.01,
                work=_partial_native_work(route_expansions=11),
            )
            for _ in range(2)
        ]
    )
    sampled: list[dict[int, frozenset[int]]] = []

    def fake_sample(program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps):
        del program, problem, ground_energy, seed, num_sweeps
        sampled.append(dict(chains))
        return ReadBlock(3072, num_reads, 0.125, 0.25, 1)

    monkeypatch.setattr("isingfold.rl.complete_system.sample_program", fake_sample)
    ctx = Context(qubit_cap=9)
    outcomes, receipts = run_complete_system(
        [task],
        ctx,
        backend,
        _config(),
        controller=_commit_initial,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=_population(task, repetitions=2, evaluation_seed=91),
        seed=91,
        repetitions=2,
    )

    assert len(backend.calls) == len(outcomes) == len(receipts) == 2
    assert sampled == [actual, actual]
    assert all(outcome.utility == pytest.approx(0.75) for outcome in outcomes)
    assert all(outcome.population_eligible for outcome in outcomes)
    assert receipts[0].initializer.version == "2.1.0"
    assert receipts[0].attempts[0].status == "VALID_CANDIDATE"
    assert receipts[0].attempts[0].seed == backend.calls[0][0]
    # The external initializer debit is exact, authenticated, and is not double-counted as
    # environment work.  The fixed registered work-cap denominator remains unchanged.
    assert receipts[0].initializer_work.restart_work == 1
    assert receipts[0].environment_budget_debit == receipts[0].initializer_work.to_work_vector()
    assert receipts[0].work_cap == ctx.caps
    assert receipts[0].context_digest == _context_digest(ctx)
    assert receipts[0].policy_context_digest == _context_digest(complete_policy_context(ctx))
    assert receipts[0].environment_work.restart_work == 0
    assert receipts[0].total_work.restart_work == 1
    assert receipts[0].initializer_work.route_expansions == 11
    assert receipts[0].total_work.route_expansions >= 11
    assert receipts[0].cap_compliance["route_expansions"] is True
    assert outcomes[0].work == receipts[0].total_work.to_work_vector()


def test_valid_complete_receipt_cannot_claim_a_wallclock_overrun() -> None:
    task = _tiny_task()
    backend = FakeInitializer(
        [
            CompleteInitializerResult(
                SearchStatus.EMBEDDING,
                {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})},
                elapsed_seconds=0.01,
                work=_partial_native_work(route_expansions=0),
            )
        ]
    )
    _, receipts = run_complete_system(
        [task],
        Context(qubit_cap=9),
        backend,
        _config(),
        controller=_commit_initial,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=_population(task),
    )
    receipt = receipts[0]
    overrun = receipt.wallclock_cap_seconds + 1.0
    outcome = dataclasses.replace(receipt.outcome, online_seconds=overrun)

    with pytest.raises(ValueError, match="valid.*wall-clock"):
        dataclasses.replace(
            receipt,
            outcome=outcome,
            online_seconds=overrun,
            post_initialization_seconds=overrun - receipt.initialization_seconds,
            wallclock_compliant=False,
        )


def test_terminal_boundary_overrun_becomes_zero_utility_before_evaluator(monkeypatch) -> None:
    task = _tiny_task()
    backend = FakeInitializer(
        [
            CompleteInitializerResult(
                SearchStatus.EMBEDDING,
                {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})},
                elapsed_seconds=0.01,
                work=_partial_native_work(route_expansions=0),
            )
        ]
    )

    class BoundaryClock:
        controller_returned = False
        calls_after_controller = 0

        def __call__(self) -> float:
            if not self.controller_returned:
                return 0.0
            self.calls_after_controller += 1
            return 31.0 if self.calls_after_controller >= 3 else 0.0

    clock = BoundaryClock()

    def controller(decision, rng) -> int:
        choice = _commit_initial(decision, rng)
        clock.controller_returned = True
        return choice

    monkeypatch.setattr("isingfold.rl.complete_system.time.perf_counter", clock)
    monkeypatch.setattr(
        "isingfold.rl.complete_system.sample_program",
        lambda *args, **kwargs: pytest.fail("wall-clock overrun must not open the evaluator"),
    )

    outcomes, receipts = run_complete_system(
        [task],
        Context(qubit_cap=9),
        backend,
        _config(),
        controller=controller,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=_population(task),
    )

    assert outcomes[0].returned_valid is False
    assert outcomes[0].utility == 0.0
    assert outcomes[0].reason == "COMPLETE_SYSTEM_WALLCLOCK_EXHAUSTED"
    assert receipts[0].evaluator_seed is None
    assert receipts[0].terminal_evidence is None
    assert receipts[0].wallclock_compliant is False


def test_unknown_initializer_work_fails_closed_before_policy_or_evaluator(monkeypatch) -> None:
    task = _tiny_task()
    actual = {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})}
    backend = FakeInitializer(
        [
            CompleteInitializerResult(
                SearchStatus.EMBEDDING,
                actual,
                elapsed_seconds=0.01,
                work=_partial_native_work(route_expansions=None),
            )
        ]
    )
    monkeypatch.setattr(
        "isingfold.rl.complete_system.EmbeddingEnv",
        lambda *args, **kwargs: pytest.fail("unknown work must not enter the policy environment"),
    )
    monkeypatch.setattr(
        "isingfold.rl.complete_system.sample_program",
        lambda *args, **kwargs: pytest.fail("unknown work must not open the evaluator"),
    )

    outcomes, receipts = run_complete_system(
        [task],
        Context(qubit_cap=9),
        backend,
        _config(),
        controller=_commit_initial,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=_population(task),
    )

    assert outcomes[0].utility == 0.0
    assert outcomes[0].reason == "INITIALIZER_WORK_INCOMPLETE"
    assert outcomes[0].work is None
    assert receipts[0].initializer_work.route_expansions is None
    assert receipts[0].environment_budget_debit == WorkVector()
    assert receipts[0].environment_work == WorkVector()
    assert receipts[0].cap_compliance["route_expansions"] is None


def test_initializer_failure_is_an_unconditional_zero_without_evaluator(monkeypatch) -> None:
    task = _tiny_task()
    backend = FakeInitializer(
        [
            CompleteInitializerResult(
                SearchStatus.NO_EMBEDDING,
                None,
                elapsed_seconds=0.01,
                work=_partial_native_work(route_expansions=17),
                detail="exhausted",
            )
        ]
    )
    monkeypatch.setattr(
        "isingfold.rl.complete_system.sample_program",
        lambda *args, **kwargs: pytest.fail("failed initialization must not open evaluator"),
    )

    outcomes, receipts = run_complete_system(
        [task],
        Context(qubit_cap=9),
        backend,
        _config(),
        controller=_commit_initial,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=_population(task),
    )

    assert outcomes[0].utility == 0.0
    assert outcomes[0].returned_valid is False
    assert outcomes[0].population_eligible is True
    assert outcomes[0].outcome_kind == "TASK_TERMINAL"
    assert receipts[0].selected_attempt is None
    assert receipts[0].attempts[0].status == SearchStatus.NO_EMBEDDING.value
    assert receipts[0].initializer_work.route_expansions == 17
    metrics = complete_system_metrics(receipts)
    assert metrics["initialization_failures"] == 1
    assert metrics["initialization_failure_rate"] == 1.0
    assert metrics["complete_failure_taxonomy"] == {
        "denominator": 1,
        "valid_returns": 0,
        "pre_policy_initializer_failures": 1,
        "no_valid_initializer_candidate": 1,
        "initializer_work_incomplete": 0,
        "policy_environment_bootstrap_failures": 0,
        "post_bootstrap_failures": 0,
        "initializer_attempt_status_counts": {"NO_EMBEDDING": 1},
        "terminal_reason_counts": {"INITIALIZER_NO_VALID_EMBEDDING": 1},
    }


def test_all_initializer_attempts_are_retained_and_selected_without_evaluator_feedback(
    monkeypatch,
) -> None:
    task = _tiny_task()
    larger = {0: frozenset({0, 3}), 1: frozenset({1, 4}), 2: frozenset({2, 5})}
    smaller = {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})}
    backend = FakeInitializer(
        [
            CompleteInitializerResult(
                SearchStatus.EMBEDDING,
                larger,
                0.01,
                _partial_native_work(route_expansions=20),
            ),
            CompleteInitializerResult(
                SearchStatus.EMBEDDING,
                smaller,
                0.01,
                _partial_native_work(route_expansions=10),
            ),
        ]
    )
    sampled: list[dict[int, frozenset[int]]] = []

    def fake_sample(program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps):
        del program, problem, ground_energy, seed, num_sweeps
        sampled.append(dict(chains))
        return ReadBlock(num_reads, num_reads, 0.0, 0.0, 1)

    monkeypatch.setattr("isingfold.rl.complete_system.sample_program", fake_sample)
    _, receipts = run_complete_system(
        [task],
        Context(qubit_cap=9),
        backend,
        _config(attempts=2),
        controller=_commit_initial,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=_population(task),
    )

    assert [item.status for item in receipts[0].attempts] == [
        "VALID_CANDIDATE",
        "VALID_CANDIDATE",
    ]
    assert receipts[0].selected_attempt == 1
    assert receipts[0].initializer_work.restart_work == 2
    assert receipts[0].initializer_work.route_expansions == 30
    assert sampled == [smaller]


def test_known_initializer_work_over_cap_is_zero_and_never_opens_evaluator(monkeypatch) -> None:
    task = _tiny_task()
    embedding = {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})}
    ctx = Context(qubit_cap=9)
    backend = FakeInitializer(
        [
            CompleteInitializerResult(
                SearchStatus.EMBEDDING,
                embedding,
                0.01,
                _partial_native_work(route_expansions=ctx.caps.route_expansions + 1),
            )
        ]
    )
    monkeypatch.setattr(
        "isingfold.rl.complete_system.sample_program",
        lambda *args, **kwargs: pytest.fail("over-cap system must not open evaluator"),
    )

    outcomes, receipts = run_complete_system(
        [task],
        ctx,
        backend,
        _config(),
        controller=_commit_initial,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=_population(task),
    )

    assert outcomes[0].utility == 0.0
    assert outcomes[0].reason == "INITIALIZER_EXCEEDED_WORK_CAP"
    assert receipts[0].cap_compliance["route_expansions"] is False


def test_unknown_backend_counter_keeps_known_validator_reservation(monkeypatch) -> None:
    task = _tiny_task()
    ctx = Context(qubit_cap=9)
    embedding = {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})}
    backend_work = PartialWorkVector(
        decisions=0,
        route_expansions=None,
        materializations=0,
        compiler_calls=0,
        validator_calls=None,
        cut_edge_visits=0,
        restart_work=0,
        evaluator_reads=0,
        feature_work=0,
        known_lower_bound=WorkVector(validator_calls=ctx.caps.validator_calls),
    )
    backend = FakeInitializer(
        [CompleteInitializerResult(SearchStatus.EMBEDDING, embedding, 0.01, backend_work)]
    )
    monkeypatch.setattr(
        "isingfold.rl.complete_system.p_embed",
        lambda *args, **kwargs: pytest.fail(
            "runner must not exceed the validator cap merely because backend work is unknown"
        ),
    )

    outcomes, receipts = run_complete_system(
        [task],
        ctx,
        backend,
        _config(),
        controller=_commit_initial,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=_population(task),
    )

    assert outcomes[0].reason == "INITIALIZER_WORK_CAP_BEFORE_VALIDATION"
    assert receipts[0].attempts[0].status == "WORK_CAP_EXHAUSTED"
    assert receipts[0].initializer_work.validator_calls is None
    assert (
        receipts[0].initializer_work.known_lower_bound.validator_calls == ctx.caps.validator_calls
    )


def test_success_only_population_view_fails_closed_before_initializer() -> None:
    task = _tiny_task()
    identities = (("base-0", "complete-tiny"), ("base-1", "missing-failure"))
    strata, design = evaluation_contract(identities)
    missing = CompletePopulationIdentity(
        population_id="full-test-population",
        source_manifest_sha256=stable_digest({"manifest": "full"}),
        task_payload_sha256=task_population_digest([task]),
        expected_instances=identities,
        expected_repetitions=1,
        evaluation_strata=strata,
        confirmatory_design=design,
    )
    backend = FakeInitializer([])

    with pytest.raises(Exception, match="sealed denominator"):
        run_complete_system(
            [task],
            Context(qubit_cap=9),
            backend,
            _config(),
            controller=_commit_initial,
            selector=fixed_strength_selector(),
            controller_identity=_component("controller"),
            selector_identity=_component("selector"),
            population=missing,
        )

    assert backend.calls == []


def test_population_rejects_missing_or_extra_evaluation_strata() -> None:
    task = _tiny_task()
    identity = ((task.lineage or task.name, task.name),)
    _, design = evaluation_contract(identity)

    with pytest.raises(ValueError, match="evaluation-stratum coverage"):
        CompletePopulationIdentity(
            population_id="missing-stratum",
            source_manifest_sha256=stable_digest({"manifest": "full"}),
            task_payload_sha256=task_population_digest([task]),
            expected_instances=identity,
            expected_repetitions=1,
            evaluation_strata=(),
            confirmatory_design=design,
        )


def test_global_confirmatory_targets_must_cover_the_full_sealed_population() -> None:
    task = _tiny_task()
    identities = (
        (task.lineage or task.name, task.name),
        ("base-other", "other-instance"),
    )
    strata, design = evaluation_contract(identities)
    narrowed = (
        strata[0],
        dataclasses.replace(strata[1], host_family="path-b"),
    )

    with pytest.raises(ValueError, match="full sealed.*population"):
        CompletePopulationIdentity(
            population_id="narrow-global-target",
            source_manifest_sha256=stable_digest({"manifest": "full"}),
            task_payload_sha256=stable_digest({"tasks": "irrelevant-to-constructor"}),
            expected_instances=identities,
            expected_repetitions=1,
            evaluation_strata=narrowed,
            confirmatory_design=design,
        )


def test_complete_receipts_are_atomic_authenticated_and_fail_closed(tmp_path: Path) -> None:
    task = _tiny_task()
    backend = FakeInitializer(
        [
            CompleteInitializerResult(
                SearchStatus.NO_EMBEDDING,
                None,
                elapsed_seconds=0.01,
                work=_partial_native_work(route_expansions=None),
            )
        ]
    )
    _, receipts = run_complete_system(
        [task],
        Context(qubit_cap=9),
        backend,
        _config(),
        controller=_commit_initial,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=_population(task),
    )
    destination = tmp_path / "complete.jsonl"

    file_digest = write_complete_system_receipts(destination, receipts)
    outcome_destination = tmp_path / "outcomes.jsonl"
    outcome_digest = write_complete_system_outcomes(outcome_destination, receipts)
    verified = verify_complete_system_receipts(destination, expected_sha256=file_digest)
    typed = read_complete_system_receipts(destination, expected_sha256=file_digest)
    projected_rows = read_complete_system_outcomes(
        outcome_destination,
        receipts=typed,
        expected_sha256=outcome_digest,
    )

    assert verified[0]["record_digest"] == stable_digest(
        {key: value for key, value in verified[0].items() if key != "record_digest"}
    )
    assert typed == tuple(receipts)
    assert len(outcome_digest) == 64
    projected = json.loads(outcome_destination.read_text())
    assert projected["complete_receipt_digest"] == verified[0]["record_digest"]
    assert projected["outcome"] == receipts[0].outcome.as_dict()
    assert projected_rows[0]["outcome"] == receipts[0].outcome.as_dict()

    rehashed_debit = tmp_path / "rehashed-debit.jsonl"
    debit_payload = json.loads(destination.read_text())
    debit_payload["environment_budget_debit"]["restart_work"] = 1
    debit_payload["record_digest"] = stable_digest(
        {key: value for key, value in debit_payload.items() if key != "record_digest"}
    )
    rehashed_debit.write_text(
        json.dumps(debit_payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    with pytest.raises(Exception, match="semantically|policy execution|debit"):
        read_complete_system_receipts(rehashed_debit)

    with pytest.raises(FileExistsError):
        write_complete_system_receipts(destination, receipts)
    destination.write_text(destination.read_text().replace("complete-tiny", "tampered-tiny"))
    with pytest.raises(Exception, match="digest"):
        verify_complete_system_receipts(destination)


def test_rehashed_outcome_projection_cannot_cross_the_complete_receipt_link(
    tmp_path: Path,
) -> None:
    task = _tiny_task()
    backend = FakeInitializer(
        [
            CompleteInitializerResult(
                SearchStatus.NO_EMBEDDING,
                None,
                elapsed_seconds=0.01,
                work=_partial_native_work(route_expansions=3),
            )
        ]
    )
    _, receipts = run_complete_system(
        [task],
        Context(qubit_cap=9),
        backend,
        _config(),
        controller=_commit_initial,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=_population(task),
    )
    destination = tmp_path / "tampered-outcomes.jsonl"
    write_complete_system_outcomes(destination, receipts)
    payload = json.loads(destination.read_text())
    payload["outcome"]["utility"] = 1.0
    payload["record_digest"] = stable_digest(
        {key: value for key, value in payload.items() if key != "record_digest"}
    )
    destination.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

    with pytest.raises(Exception, match="receipt link|projected outcome"):
        read_complete_system_outcomes(destination, receipts=receipts)


def test_valid_complete_return_emits_recompilable_four_program_evidence(
    monkeypatch, tmp_path: Path
) -> None:
    task = _tiny_task()
    ctx = Context(qubit_cap=9)
    embedding = {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})}
    backend = FakeInitializer(
        [
            CompleteInitializerResult(
                SearchStatus.EMBEDDING,
                embedding,
                elapsed_seconds=0.01,
                work=_partial_native_work(route_expansions=3),
            )
        ]
    )

    def fake_sample(program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps):
        del program, chains, problem, ground_energy, seed, num_sweeps
        return ReadBlock(1234, num_reads, 0.125, 0.25, 2)

    monkeypatch.setattr("isingfold.rl.complete_system.sample_program", fake_sample)
    _, receipts = run_complete_system(
        [task],
        ctx,
        backend,
        _config(),
        controller=_commit_initial,
        selector=fixed_strength_selector(2),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=_population(task),
    )

    receipt = receipts[0]
    evidence = receipt.terminal_evidence
    assert evidence is not None
    assert receipt.terminal_evidence_digest == evidence.digest
    assert len(evidence.programs) == 4
    assert tuple(program["strength_index"] for program in evidence.programs) == (0, 1, 2, 3)
    assert evidence.selected_index == 2
    assert evidence.evaluator_counts == {"hits": 1234, "reads": 4096}
    verify_terminal_evidence(evidence, task=task, context=ctx, outcome=receipt.outcome)

    raw_destination = tmp_path / "complete-system.jsonl"
    raw_digest = write_complete_system_receipts(raw_destination, receipts)
    recovered_receipts = read_complete_system_receipts(raw_destination, expected_sha256=raw_digest)
    assert recovered_receipts[0].terminal_evidence is None
    assert recovered_receipts[0].terminal_evidence_digest == evidence.digest

    destination = tmp_path / "terminal-evidence.jsonl"
    file_digest = write_complete_system_evidence(destination, receipts)
    recovered = read_complete_system_evidence(
        destination,
        receipts=recovered_receipts,
        tasks=[task],
        context=ctx,
        expected_sha256=file_digest,
    )
    assert recovered[0].terminal_evidence == evidence
    assert recovered[0].complete_receipt_digest == receipt.as_dict()["record_digest"]


def test_evidence_sidecar_authenticates_absence_for_initializer_failure(tmp_path: Path) -> None:
    task = _tiny_task()
    backend = FakeInitializer(
        [
            CompleteInitializerResult(
                SearchStatus.NO_EMBEDDING,
                None,
                elapsed_seconds=0.01,
                work=_partial_native_work(route_expansions=3),
            )
        ]
    )
    _, receipts = run_complete_system(
        [task],
        Context(qubit_cap=9),
        backend,
        _config(),
        controller=_commit_initial,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=_population(task),
    )

    assert receipts[0].terminal_evidence is None
    assert receipts[0].terminal_evidence_digest is None
    assert receipts[0].evaluator_seed is None
    destination = tmp_path / "failure-evidence.jsonl"
    digest = write_complete_system_evidence(destination, receipts)
    recovered = read_complete_system_evidence(
        destination,
        receipts=receipts,
        expected_sha256=digest,
    )
    assert recovered[0].terminal_evidence is None


def test_ordinary_invalid_complete_return_has_no_valid_only_evidence() -> None:
    base = _tiny_task()
    mismatched_problem = LogicalProblem.from_dicts(
        {0: 1.0, 1: -1.0, 2: 1.0}, {(0, 1): -1.0}
    )
    task = EmbeddingTask(
        base.name,
        base.logical,
        base.host,
        mismatched_problem,
        -2.0,
        lineage=base.lineage,
    )
    embedding = {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})}
    backend = FakeInitializer(
        [
            CompleteInitializerResult(
                SearchStatus.EMBEDDING,
                embedding,
                elapsed_seconds=0.01,
                work=_partial_native_work(route_expansions=3),
            )
        ]
    )

    outcomes, receipts = run_complete_system(
        [task],
        Context(qubit_cap=9),
        backend,
        _config(),
        controller=lambda decision, rng: pytest.fail(
            "a rejected selected initializer must not invoke the controller"
        ),
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=_population(task),
    )

    outcome, receipt = outcomes[0], receipts[0]
    assert outcome.returned_valid is False
    assert outcome.qubits is outcome.max_chain is outcome.evaluator_seed is None
    assert outcome.program_digest is outcome.selected_strength is None
    assert outcome.evaluator_hits is outcome.evaluator_reads is None
    assert receipt.selected_attempt == 0
    assert receipt.evaluator_seed is None
    assert receipt.terminal_evidence_digest is None
    assert receipt.terminal_evidence is None


def test_rehashed_evidence_tamper_cannot_cross_the_complete_receipt_link(
    monkeypatch, tmp_path: Path
) -> None:
    task = _tiny_task()
    ctx = Context(qubit_cap=9)
    embedding = {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})}
    backend = FakeInitializer(
        [
            CompleteInitializerResult(
                SearchStatus.EMBEDDING,
                embedding,
                elapsed_seconds=0.01,
                work=_partial_native_work(route_expansions=3),
            )
        ]
    )
    monkeypatch.setattr(
        "isingfold.rl.complete_system.sample_program",
        lambda program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps: ReadBlock(
            num_reads, num_reads, 0.0, 0.0, 1
        ),
    )
    _, receipts = run_complete_system(
        [task],
        ctx,
        backend,
        _config(),
        controller=_commit_initial,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=_population(task),
    )
    destination = tmp_path / "tampered-evidence.jsonl"
    write_complete_system_evidence(destination, receipts)
    payload = json.loads(destination.read_text())
    payload["terminal_evidence"]["programs"][0]["scale"] = 0.75
    payload["terminal_evidence"]["evidence_digest"] = stable_digest(
        {
            key: value
            for key, value in payload["terminal_evidence"].items()
            if key != "evidence_digest"
        }
    )
    payload["record_digest"] = stable_digest(
        {key: value for key, value in payload.items() if key != "record_digest"}
    )
    destination.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

    with pytest.raises(Exception, match="receipt link|evidence digest"):
        read_complete_system_evidence(destination, receipts=receipts, tasks=[task], context=ctx)


def test_semantic_receipt_tampering_fails_even_after_attacker_rehashes(tmp_path: Path) -> None:
    task = _tiny_task()
    backend = FakeInitializer(
        [
            CompleteInitializerResult(
                SearchStatus.NO_EMBEDDING,
                None,
                elapsed_seconds=0.01,
                work=_partial_native_work(route_expansions=None),
            )
        ]
    )
    _, receipts = run_complete_system(
        [task],
        Context(qubit_cap=9),
        backend,
        _config(),
        controller=_commit_initial,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=_population(task),
    )
    destination = tmp_path / "semantic-tamper.jsonl"
    write_complete_system_receipts(destination, receipts)
    payload = json.loads(destination.read_text())
    payload["total_work_known_lower_bound"]["restart_work"] += 1
    payload["record_digest"] = stable_digest(
        {key: value for key, value in payload.items() if key != "record_digest"}
    )
    destination.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")

    with pytest.raises(Exception, match="semantically|total work"):
        read_complete_system_receipts(destination)


def test_pairing_with_stock_fails_closed_until_external_receipts_bind_full_envelope(
    monkeypatch,
) -> None:
    task = _tiny_task()
    embedding = {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})}
    ctx = Context(qubit_cap=9)
    learned_backend = FakeInitializer(
        [
            CompleteInitializerResult(
                SearchStatus.EMBEDDING,
                embedding,
                elapsed_seconds=0.01,
                work=_partial_native_work(route_expansions=None),
            )
        ]
    )

    def fake_sample(program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps):
        del program, chains, problem, ground_energy, seed, num_sweeps
        return ReadBlock(2048, num_reads, 0.1, 0.2, 1)

    monkeypatch.setattr("isingfold.rl.complete_system.sample_program", fake_sample)
    monkeypatch.setattr("isingfold.rl.external.sample_program", fake_sample)
    learned_outcomes, learned_receipts = run_complete_system(
        [task],
        ctx,
        learned_backend,
        _config(),
        controller=_commit_initial,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=_population(task, evaluation_seed=1201),
        seed=1201,
    )
    external_outcomes, external_receipts = run_external_system(
        [task],
        ctx,
        FakeExternal(embedding),
        _external_config(),
        selector=fixed_strength_selector(),
        seed=1201,
    )

    assert learned_outcomes and external_outcomes
    with pytest.raises(Exception, match="cannot support an honest whole-system pairing"):
        pair_with_stock_minorminer(
            learned_receipts,
            external_receipts,
            learned_config=_config(),
            stock_config=_external_config(),
            context=ctx,
            population=_population(task, evaluation_seed=1201),
        )

    incompatible = CompleteSystemConfig(
        online_wallclock_seconds=29.0,
        max_initializer_attempts=1,
        audit_reads=4096,
        selection_rule="resource-lexicographic",
        online_evaluator_feedback=False,
        initializer_backend="isingfold",
        initializer_method_id="named-greedy-initializer-v2",
        expected_initializer_version="2.1.0",
        policy_restart_mode="disabled-no-native-replay-v1",
        initializer_parameters={"tries": 1},
    )
    with pytest.raises(ValueError, match="wall-clock"):
        pair_with_stock_minorminer(
            learned_receipts,
            external_receipts,
            learned_config=incompatible,
            stock_config=_external_config(),
            context=ctx,
            population=_population(task),
        )


def test_serialized_partial_work_keeps_unavailable_native_counters_null() -> None:
    payload = _partial_native_work(route_expansions=None).as_dict()

    assert set(payload) == set(WorkVector().__dataclass_fields__)
    assert payload["route_expansions"] is None
    assert json.dumps(payload, allow_nan=False)


def test_unknown_work_preserves_known_runner_lower_bounds_and_cap_violation() -> None:
    native = PartialWorkVector(
        decisions=0,
        route_expansions=None,
        materializations=0,
        compiler_calls=0,
        validator_calls=None,
        cut_edge_visits=0,
        restart_work=0,
        evaluator_reads=0,
        feature_work=0,
        known_lower_bound=WorkVector(validator_calls=7),
    )

    total = native + WorkVector(validator_calls=1, restart_work=1)

    assert total.validator_calls is None
    assert total.known_lower_bound.validator_calls == 8
    assert total.known_lower_bound.restart_work == 1
    assert (
        total.cap_compliance(WorkVector(validator_calls=7, restart_work=1))["validator_calls"]
        is False
    )


def test_complete_policy_support_disables_replay_only_restart_actions(monkeypatch) -> None:
    task = _tiny_task()
    embedding = {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})}
    backend = FakeInitializer(
        [
            CompleteInitializerResult(
                SearchStatus.EMBEDDING,
                embedding,
                elapsed_seconds=0.01,
                work=_partial_native_work(route_expansions=1),
            )
        ]
    )

    def controller(decision, rng) -> int:
        assert all(
            candidate.opcode is not Opcode.RESTART
            for candidate, legal in zip(decision.candidates, decision.legal_mask, strict=True)
            if legal
        )
        return _commit_initial(decision, rng)

    monkeypatch.setattr(
        "isingfold.rl.complete_system.sample_program",
        lambda program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps: ReadBlock(
            num_reads, num_reads, 0.0, 0.0, 1
        ),
    )
    _, receipts = run_complete_system(
        [task],
        Context(qubit_cap=9),
        backend,
        _config(),
        controller=controller,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=_population(task),
    )

    assert receipts[0].environment_work.restart_work == 0


def test_lac_minorminer_production_adapter_reports_native_diagnostics() -> None:
    backend = LACMinorminerInitializerBackend()
    result = backend.search(
        nx.path_graph(2),
        nx.path_graph(2),
        seed=17,
        timeout_seconds=2.0,
        parameters={"tries": 1, "max_transitions": 100, "max_candidates": 4},
        work_cap=Context(qubit_cap=100).caps,
    )

    assert backend.identity.distribution == "lac-minorminer"
    assert backend.identity.method_id == (
        "lac-minorminer-hybrid-chimera-clique-v1-initializer-v4"
    )
    assert result.status is SearchStatus.EMBEDDING
    assert result.work.complete
    assert result.work.evaluator_reads == 0
    assert result.work.route_expansions == 0
    assert result.diagnostics["random_seed"] == 17
    assert result.diagnostics["package_version"] == backend.identity.version


def test_initializer_identity_mutation_fails_closed() -> None:
    task = _tiny_task()
    result = CompleteInitializerResult(
        SearchStatus.NO_EMBEDDING,
        None,
        elapsed_seconds=0.01,
        work=_partial_native_work(route_expansions=1),
    )

    class MutatingInitializer(FakeInitializer):
        def search(self, logical, host, *, seed, timeout_seconds, parameters):
            returned = super().search(
                logical,
                host,
                seed=seed,
                timeout_seconds=timeout_seconds,
                parameters=parameters,
            )
            self.identity = BackendIdentity(
                method_id="changed-mid-run",
                distribution="isingfold",
                version="2.1.0",
                entrypoint="tests.changed",
                implementation="invalid-mutation",
            )
            return returned

    with pytest.raises(Exception, match="identity changed"):
        run_complete_system(
            [task],
            Context(qubit_cap=9),
            MutatingInitializer([result]),
            _config(),
            controller=_commit_initial,
            selector=fixed_strength_selector(),
            controller_identity=_component("controller"),
            selector_identity=_component("selector"),
            population=_population(task),
        )


def test_method_metadata_separates_complete_and_post_initialization_estimands() -> None:
    task = _tiny_task()
    backend = FakeInitializer([])

    metadata = complete_system_method_metadata(
        _config(),
        backend.identity,
        _component("controller"),
        _component("selector"),
        _population(task),
    )

    assert metadata["prepared_initial_embedding_replay"] is False
    assert metadata["initializer_failure_utility"] == 0.0
    assert metadata["post_initialization_api"].endswith("run_controller")
    assert metadata["complete_system_api"].endswith("run_complete_system")
    assert metadata["work_accounting"]["unknown_coordinate_rule"].startswith("null")


def test_repair_success_counts_newly_satisfied_demand_not_only_resolved_conflict() -> None:
    host = nx.Graph([(10, 11)])
    env = SimpleNamespace(
        state=SimpleNamespace(chains={"left": frozenset({10}), "right": frozenset({11})}),
        task=SimpleNamespace(host=host),
    )
    demand_repair = SimpleNamespace(target_conflict=None, target_demand=("left", "right"))

    assert _repair_target_satisfied(demand_repair, env) is True
