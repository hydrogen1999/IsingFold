from __future__ import annotations

import dataclasses
import hashlib
import importlib
import json
from pathlib import Path

import networkx as nx
import pytest

from tests.unit.evaluation_strata_support import evaluation_contract
from isingfold.embedding import LogicalProblem
from isingfold.rl.complete_system import (
    CompletePopulationIdentity,
    CompleteSystemConfig,
    FrozenComponentIdentity,
    LACMinorminerInitializerBackend,
    run_complete_system,
    task_population_digest,
)
from isingfold.rl.contracts import Context, Opcode, WorkVector, stable_digest
from isingfold.rl.env import EmbeddingTask, fixed_strength_selector
from isingfold.rl.evaluator import ReadBlock
from isingfold.rl.external import SearchStatus
from lac_minorminer import (
    SearchDiagnostics,
    SearchSession,
    SearchWorkCounters,
    TerminationReason,
    WORK_COUNTER_SCHEMA,
    WORK_COUNTER_VERSION,
    WorkBudgetExceeded,
    backend_info,
    find_embedding,
)


def _run_nontrivial_search() -> tuple[dict, SearchDiagnostics]:
    return find_embedding(
        nx.complete_graph(3),
        nx.cycle_graph(4),
        random_seed=7,
        tries=1,
        max_transitions=100,
        max_candidates=4,
        search_profile="v0",
        return_diagnostics=True,
    )


def test_native_work_receipt_is_exact_versioned_and_deterministic() -> None:
    first_embedding, first = _run_nontrivial_search()
    second_embedding, second = _run_nontrivial_search()

    expected = SearchWorkCounters(
        decisions=1,
        route_expansions=8,
        materializations=2,
        compiler_calls=0,
        validator_calls=5,
        cut_edge_visits=0,
        restart_work=0,
        evaluator_reads=0,
        feature_work=82,
    )
    assert first_embedding == second_embedding
    assert first.work == second.work == expected
    assert first.work.decisions == first.transitions
    assert first.work_counter_schema == WORK_COUNTER_SCHEMA
    assert first.work_counter_version == WORK_COUNTER_VERSION
    assert backend_info()["work_counter_schema"] == WORK_COUNTER_SCHEMA
    assert backend_info()["work_counter_version"] == WORK_COUNTER_VERSION
    assert first.structural_dict() == second.structural_dict()


def test_work_schema_round_trip_rejects_missing_or_fabricated_coordinates() -> None:
    _, diagnostics = _run_nontrivial_search()
    payload = json.loads(json.dumps(diagnostics.to_dict(), allow_nan=False))
    assert SearchDiagnostics.from_dict(payload) == diagnostics

    missing = json.loads(json.dumps(payload))
    missing["work"].pop("feature_work")
    with pytest.raises(ValueError, match="missing or unknown"):
        SearchDiagnostics.from_dict(missing)

    fabricated = json.loads(json.dumps(payload))
    fabricated["work"]["cut_edge_visits"] = True
    with pytest.raises(ValueError, match="nonnegative integer"):
        SearchDiagnostics.from_dict(fabricated)

    forbidden = json.loads(json.dumps(payload))
    forbidden["work"]["evaluator_reads"] = 1
    with pytest.raises(ValueError, match="cannot claim"):
        SearchDiagnostics.from_dict(forbidden)

    inconsistent = json.loads(json.dumps(payload))
    inconsistent["work"]["decisions"] += 1
    with pytest.raises(ValueError, match="decision work"):
        SearchDiagnostics.from_dict(inconsistent)


def test_python_search_stops_before_native_work_cap_boundary() -> None:
    work_cap = SearchWorkCounters(
        decisions=0,
        route_expansions=100,
        materializations=100,
        compiler_calls=0,
        validator_calls=100,
        cut_edge_visits=0,
        restart_work=0,
        evaluator_reads=0,
        feature_work=10_000,
    )
    embedding, diagnostics = find_embedding(
        nx.complete_graph(3),
        nx.cycle_graph(4),
        random_seed=7,
        tries=1,
        max_transitions=100,
        max_candidates=4,
        search_profile="v0",
        work_cap=work_cap,
        return_diagnostics=True,
    )

    assert embedding == {}
    assert diagnostics.termination_reason is TerminationReason.WORK_BUDGET_EXHAUSTED
    assert diagnostics.work_budget_exhausted_coordinate == "decisions"
    assert diagnostics.work.decisions == 0
    assert diagnostics.work.route_expansions == 0
    assert diagnostics.work.materializations == 0
    assert diagnostics.work_cap == work_cap


def test_python_session_snapshot_charges_multiple_coordinates_atomically() -> None:
    session = SearchSession(
        nx.empty_graph(1),
        nx.empty_graph(1),
        random_seed=5,
        max_candidates=2,
        work_cap=SearchWorkCounters(validator_calls=0, feature_work=10_000),
    )
    with pytest.raises(WorkBudgetExceeded, match="validator_calls"):
        session.snapshot()
    assert session.work_counters().validator_calls == 0
    assert session.work_counters().feature_work == 0


def test_complete_initializer_adapter_propagates_all_nine_exact_coordinates() -> None:
    backend = LACMinorminerInitializerBackend()
    result = backend.search(
        nx.complete_graph(3),
        nx.cycle_graph(4),
        seed=7,
        timeout_seconds=2.0,
        parameters={"tries": 1, "max_transitions": 100, "max_candidates": 4},
        work_cap=Context(qubit_cap=100).caps,
    )

    assert backend.identity.method_id == (
        "lac-minorminer-hybrid-chimera-clique-v1-initializer-v4"
    )
    assert result.status is SearchStatus.EMBEDDING
    assert result.work.complete
    assert result.work.to_work_vector() == WorkVector(
        decisions=1,
        route_expansions=8,
        materializations=2,
        compiler_calls=0,
        validator_calls=5,
        cut_edge_visits=0,
        restart_work=0,
        evaluator_reads=0,
        feature_work=82,
    )
    assert result.diagnostics["work"] == result.work.as_dict()

    manifest = result.diagnostics["runtime_implementation_manifest"]
    native_path = Path(importlib.import_module("lac_minorminer._core").__file__).resolve()
    assert manifest["native_extension_sha256"] == hashlib.sha256(
        native_path.read_bytes()
    ).hexdigest()
    assert manifest["python_source_sha256"]
    assert backend.identity.implementation == (
        f"lac_minorminer_cpp:runtime-sha256:{manifest['manifest_sha256']}"
    )


def _component(name: str) -> FrozenComponentIdentity:
    return FrozenComponentIdentity(
        component_id=name,
        version="test-v1",
        implementation=f"tests.{name}",
        artifact_sha256=stable_digest({"component": name}),
    )


def test_real_lac_initializer_reaches_the_policy_in_complete_system(monkeypatch) -> None:
    logical = nx.empty_graph(1)
    host = nx.empty_graph(1)
    task = EmbeddingTask(
        "native-one-node",
        logical,
        host,
        LogicalProblem.from_dicts({0: 1.0}, {}),
        -1.0,
        lineage="native-lineage",
    )
    identities = (("native-lineage", "native-one-node"),)
    strata, design = evaluation_contract(identities)
    population = CompletePopulationIdentity(
        population_id="native-work-integration",
        source_manifest_sha256=stable_digest({"manifest": "native-one-node"}),
        task_payload_sha256=task_population_digest([task]),
        expected_instances=identities,
        expected_repetitions=1,
        evaluation_seed=11,
        evaluation_strata=strata,
        confirmatory_design=design,
    )
    config = CompleteSystemConfig(
        online_wallclock_seconds=10.0,
        max_initializer_attempts=1,
        audit_reads=4096,
        selection_rule="resource-lexicographic",
        online_evaluator_feedback=False,
        initializer_backend="lac-minorminer",
        initializer_method_id="lac-minorminer-hybrid-chimera-clique-v1-initializer-v4",
        expected_initializer_version="0.1.0",
        policy_restart_mode="disabled-no-native-replay-v1",
        initializer_parameters={"tries": 1, "max_transitions": 4, "max_candidates": 2},
    )
    policy_calls = 0

    def commit(decision, rng) -> int:
        nonlocal policy_calls
        del rng
        policy_calls += 1
        return next(
            index
            for index, (candidate, legal) in enumerate(
                zip(decision.candidates, decision.legal_mask, strict=True)
            )
            if legal and candidate.opcode is Opcode.COMMIT
        )

    monkeypatch.setattr(
        "isingfold.rl.complete_system.sample_program",
        lambda program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps: ReadBlock(
            num_reads, num_reads, 0.0, 0.0, program.strength_index
        ),
    )
    outcomes, receipts = run_complete_system(
        [task],
        Context(qubit_cap=1),
        LACMinorminerInitializerBackend(),
        config,
        controller=commit,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=population,
        seed=11,
    )

    assert policy_calls == 1
    assert outcomes[0].returned_valid
    assert receipts[0].attempts[0].status == "VALID_CANDIDATE"
    assert receipts[0].initializer_work.complete
    assert receipts[0].environment_budget_debit != WorkVector()
    assert outcomes[0].reason != "INITIALIZER_WORK_INCOMPLETE"

    raw_diagnostics = dict(receipts[0].attempts[0].backend_diagnostics)
    changed_work = dict(raw_diagnostics["work"])
    changed_work["decisions"] += 1
    raw_diagnostics["work"] = changed_work
    changed_attempt = dataclasses.replace(
        receipts[0].attempts[0],
        backend_diagnostics=raw_diagnostics,
    )
    with pytest.raises(ValueError, match="diagnostics disagree"):
        dataclasses.replace(receipts[0], attempts=(changed_attempt,))


def test_complete_system_records_native_budget_exhaustion_before_policy() -> None:
    logical = nx.complete_graph(3)
    host = nx.cycle_graph(4)
    task = EmbeddingTask(
        "native-budget-stop",
        logical,
        host,
        LogicalProblem.from_dicts(
            {0: 0.0, 1: 0.0, 2: 0.0},
            {(0, 1): -1.0, (0, 2): -1.0, (1, 2): -1.0},
        ),
        -3.0,
        lineage="native-budget-lineage",
    )
    base_context = Context(qubit_cap=4)
    context = dataclasses.replace(
        base_context,
        caps=dataclasses.replace(base_context.caps, decisions=1),
    )
    identities = (("native-budget-lineage", "native-budget-stop"),)
    strata, design = evaluation_contract(identities)
    population = CompletePopulationIdentity(
        population_id="native-budget-population",
        source_manifest_sha256=stable_digest({"manifest": "native-budget-stop"}),
        task_payload_sha256=task_population_digest([task]),
        expected_instances=identities,
        expected_repetitions=1,
        evaluation_seed=13,
        evaluation_strata=strata,
        confirmatory_design=design,
    )
    config = CompleteSystemConfig(
        online_wallclock_seconds=10.0,
        max_initializer_attempts=1,
        audit_reads=4096,
        selection_rule="resource-lexicographic",
        online_evaluator_feedback=False,
        initializer_backend="lac-minorminer",
        initializer_method_id="lac-minorminer-hybrid-chimera-clique-v1-initializer-v4",
        expected_initializer_version="0.1.0",
        policy_restart_mode="disabled-no-native-replay-v1",
        initializer_parameters={"tries": 1, "max_transitions": 31, "max_candidates": 2},
    )
    policy_calls = 0

    def should_not_run(decision, rng) -> int:
        nonlocal policy_calls
        del decision, rng
        policy_calls += 1
        return 0

    outcomes, receipts = run_complete_system(
        [task],
        context,
        LACMinorminerInitializerBackend(),
        config,
        controller=should_not_run,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=population,
        seed=13,
    )

    assert policy_calls == 0
    assert outcomes[0].reason == "INITIALIZER_NATIVE_WORK_BUDGET_EXHAUSTED"
    assert receipts[0].attempts[0].status == "WORK_BUDGET_EXHAUSTED"
    assert (
        receipts[0].attempts[0].backend_diagnostics[
            "work_budget_exhausted_coordinate"
        ]
        == "decisions"
    )
    assert receipts[0].attempts[0].backend_work.decisions == 0
