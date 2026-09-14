from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

import isingfold.rl.gates as gates_module
from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import (
    Candidate,
    Context,
    DecisionState,
    Mode,
    Opcode,
    TerminalReason,
    TerminalRecord,
    WorkVector,
)
from isingfold.rl.env import EmbeddingTask
from isingfold.rl.evaluator import ReadBlock
from isingfold.rl.gates import (
    EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES,
    EXACT_CONFORMANCE_MAX_TASKS,
    gate_conformance,
    gate_authenticated_k2_support_coverage,
    gate_profile_c_readiness,
    gate_profile_i_signal,
    gate_selector_discrimination,
    gate_support_headroom,
    gate_valid_returns,
    support_headroom_inference,
)
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.quality_resolution_plan import (
    QUALITY_RESOLUTION_PLAN_SCHEMA,
    QUALITY_RESOLUTION_PLAN_VERSION,
    QualityResolutionPlan,
    materialize_resolution_production_plan,
)
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.strength import StrengthGraphInput, StrengthRecord
from tests.unit.test_rl_initializer_bank import _context as _bank_context
from tests.unit.test_rl_quality_initializer_bank import _sealed_test_bank
from tests.unit.test_rl_quality_resolution_plan import _planning_inputs


def _packed(rows: int, slots: int, value: float = 0.0) -> np.ndarray:
    result = np.zeros((rows, 2 * slots), dtype=np.float32)
    result[:, :slots] = value
    result[:, slots:] = 1.0
    return result


def _graph(index: int, normalizer_digest: str) -> StrengthGraphInput:
    return StrengthGraphInput(
        logical=_packed(2, 20, 0.1),
        hardware=_packed(3, 18, 0.2),
        logical_edges=_packed(2, 6, 0.3),
        hardware_edges=_packed(4, 10, 0.4),
        claims=_packed(3, 6, 0.5),
        globals_=_packed(1, 32, 0.6)[0],
        index_logical_edges=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        index_hardware_edges=np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
        index_claims=np.asarray([[0, 1, 1], [0, 1, 2]], dtype=np.int64),
        strength_index=index,
        strength=(0.5, 1.0, 2.0, 4.0)[index],
        scale=0.9,
        normalizer_digest=normalizer_digest,
    ).validate()


class _GraphSelector:
    deployment_ready = True
    normalizer_digest = "a" * 64
    coefficient_transform_scale = 1.0

    def __init__(self) -> None:
        self.calls: list[tuple[StrengthGraphInput, ...]] = []

    def select_embedding(self, inputs) -> int:
        frozen = tuple(inputs)
        self.calls.append(frozen)
        return 3


def _task() -> EmbeddingTask:
    logical = nx.Graph([(0, 1)])
    host = nx.cycle_graph(5)
    return EmbeddingTask(
        name="task",
        logical=logical,
        host=host,
        problem=LogicalProblem.from_dicts({0: 0.25, 1: -0.5}, {(0, 1): -1.0}),
        ground_energy=-1.75,
        lineage="lineage",
        initial_embedding={0: frozenset((0,)), 1: frozenset((1,))},
    )


def test_support_gate_calls_the_deployment_graph_selector(
    monkeypatch,
) -> None:
    selector = _GraphSelector()

    def sampled(program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps):
        del chains, problem, ground_energy, seed, num_sweeps
        return ReadBlock(
            hits=program.strength_index,
            reads=num_reads,
            broken_fraction=0.0,
            mean_residual=0.0,
            strength_index=program.strength_index,
        )

    monkeypatch.setattr("isingfold.rl.gates.sample_program", sampled)
    report = gate_support_headroom(
        [_task()],
        Context(qubit_cap=5, n_est_reads=8),
        selector=selector,
        reads=8,
    )

    assert selector.calls
    assert all(
        tuple(graph.strength_index for graph in call) == (0, 1, 2, 3) for call in selector.calls
    )
    assert all(
        graph.normalizer_digest == selector.normalizer_digest
        for call in selector.calls
        for graph in call
    )
    assert report["signal_name"] == "frozen-broad-reference-support-audit"
    assert report["broad_reference_batches_per_instance"] == 8
    assert "mean_feasibility_support_recall" in report
    assert "mean_broad_pool_quality_advantage" in report


def test_support_gate_confirms_selected_candidate_on_an_independent_read_block(
    monkeypatch,
) -> None:
    initial = {0: frozenset((0,)), 1: frozenset((1,))}
    candidate_calls: dict[tuple[tuple[int, tuple[int, ...]], ...], int] = {}

    def sampled(program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps):
        del program, problem, ground_energy, seed, num_sweeps
        frozen = tuple(sorted((node, tuple(sorted(chain))) for node, chain in chains.items()))
        if dict(chains) == initial:
            hits = 6
        else:
            prior = candidate_calls.get(frozen, 0)
            candidate_calls[frozen] = prior + 1
            hits = 8 if prior == 0 else 0
        return ReadBlock(
            hits=hits,
            reads=num_reads,
            broken_fraction=0.0,
            mean_residual=0.0,
            strength_index=0,
        )

    monkeypatch.setattr("isingfold.rl.gates.sample_program", sampled)
    report = gate_support_headroom(
        [_task()],
        Context(qubit_cap=5, n_est_reads=8),
        reads=8,
    )

    assert any(count == 2 for count in candidate_calls.values())
    assert report["mean_supported_headroom_over_initial"] < 0.0


def test_support_gate_audits_every_legal_candidate_without_a_positional_cutoff(
    monkeypatch,
) -> None:
    initial = {0: frozenset((0,)), 1: frozenset((1,))}
    candidates = tuple(
        Candidate(
            opcode=(Opcode.REWRITE_ONE if index < 13 else Opcode.REPAIR_GROUP),
            affected=(0,),
            old_chains={0: initial[0]},
            new_chains={0: frozenset((index + 2,))},
            work=WorkVector(materializations=1),
            payload_key=f"candidate-{index:02d}",
            provenance=("single" if index < 13 else "repair"),
        )
        for index in range(14)
    )
    decision = DecisionState(
        observation=None,
        candidates=candidates,
        legal_mask=(True,) * len(candidates),
        state_fingerprint="state",
        context_version="test",
        charged_work_receipt=WorkVector(),
        support_fingerprint="support",
    )

    class FakeEnv:
        def __init__(self, *_args, **_kwargs):
            self.state = SimpleNamespace(
                archive=[SimpleNamespace(chains=initial)],
                chains=initial,
            )

        def reset(self, _seed):
            return decision

    def sampled(program, chains, problem, ground_energy, *, num_reads, seed, num_sweeps):
        del program, chains, problem, ground_energy, seed, num_sweeps
        return ReadBlock(
            hits=num_reads // 2,
            reads=num_reads,
            broken_fraction=0.0,
            mean_residual=0.0,
            strength_index=0,
        )

    logical = nx.Graph([(0, 1)])
    host = nx.complete_graph(20)
    task = EmbeddingTask(
        name="full-support-task",
        logical=logical,
        host=host,
        problem=LogicalProblem.from_dicts(
            {0: 0.25, 1: -0.5},
            {(0, 1): -1.0},
        ),
        ground_energy=-1.25,
        lineage="full-support-lineage",
        initial_embedding=initial,
    )
    monkeypatch.setattr("isingfold.rl.gates.EmbeddingEnv", FakeEnv)
    monkeypatch.setattr("isingfold.rl.gates.sample_program", sampled)

    report = gate_support_headroom(
        [task],
        Context(qubit_cap=20, n_est_reads=8),
        reads=8,
        broad_reference_batches=2,
    )

    assert report["candidate_selection"] == (
        "all-legal-materialized-workspace-candidates-no-truncation"
    )
    assert report["offline_candidate_attempts"] == 28
    assert report["legal_workspace_opcode_counts"]["REPAIR_GROUP"] == 2


def _one_lineage_k2_support_plan(tmp_path: Path):
    public, bank, bank_sha256 = _sealed_test_bank(tmp_path / "initializer-bank")
    context = _bank_context()
    production = materialize_resolution_production_plan(
        [public],
        context=context,
        selector=fixed_strength_selector(),
        config=_planning_inputs().config,
        source_corpus_manifest_sha256=bank.plan.prepared_manifest_sha256,
        initializer_bank=bank,
        expected_initializer_bank_manifest_sha256=bank_sha256,
        allow_test_initializer_bank=True,
    )
    production_record = production.as_dict()
    row_ids = [row["row_id"] for row in production_record["rows"]]
    payload: dict[str, object] = {
        "prepared_corpus": {
            "manifest_sha256": bank.plan.prepared_manifest_sha256,
        },
        "production_plan": production_record,
        "sample": {
            "selected_lineages": [public.task.lineage],
            "selected_row_ids": row_ids,
        },
        "schema": QUALITY_RESOLUTION_PLAN_SCHEMA,
        "schema_version": QUALITY_RESOLUTION_PLAN_VERSION,
    }
    plan = QualityResolutionPlan({**payload, "record_digest": content_digest(payload)})
    plan_sha256 = hashlib.sha256(canonical_json_bytes(plan.as_dict()) + b"\n").hexdigest()
    return public, context, bank, bank_sha256, plan, plan_sha256


def test_authenticated_k2_support_audit_replays_complete_target_free_support(
    tmp_path: Path,
    monkeypatch,
) -> None:
    public, context, bank, bank_sha256, plan, plan_sha256 = _one_lineage_k2_support_plan(tmp_path)
    monkeypatch.setattr(gates_module, "SUPPORT_FULL_AUDIT_MINIMUM_LINEAGES", 1)

    report = gate_authenticated_k2_support_coverage(
        plan,
        expected_plan_sha256=plan_sha256,
        public_prepared=[public],
        context=context,
        selector=fixed_strength_selector(),
        initializer_bank=bank,
        expected_initializer_bank_manifest_sha256=bank_sha256,
        allow_test_initializer_bank=True,
    )

    assert report["protocol"] == "authenticated-k2-full-legal-support-v1"
    assert report["candidate_selection"] == ("all-legal-materialized-candidates-no-truncation")
    assert report["target_accessed"] is False
    assert report["sampled_independent_lineages"] == 1
    assert report["audited_candidate_count"] == report["legal_candidate_count"]
    assert report["initializer_bank"]["manifest_sha256"] == bank_sha256
    assert report["rows"]
    assert all(row["audited_indices"] == row["legal_indices"] for row in report["rows"])

    with pytest.raises(ValueError, match="pinned persistent K=2"):
        gate_authenticated_k2_support_coverage(
            plan,
            expected_plan_sha256=plan_sha256,
            public_prepared=[public],
            context=context,
            selector=fixed_strength_selector(),
            initializer_bank=None,
            expected_initializer_bank_manifest_sha256=None,
            allow_test_initializer_bank=True,
        )


def test_authenticated_k2_support_audit_requires_128_independent_lineages(
    tmp_path: Path,
) -> None:
    public, context, bank, bank_sha256, plan, plan_sha256 = _one_lineage_k2_support_plan(tmp_path)

    with pytest.raises(ValueError, match="at least 128 independent"):
        gate_authenticated_k2_support_coverage(
            plan,
            expected_plan_sha256=plan_sha256,
            public_prepared=[public],
            context=context,
            selector=fixed_strength_selector(),
            initializer_bank=bank,
            expected_initializer_bank_manifest_sha256=bank_sha256,
            allow_test_initializer_bank=True,
        )


def test_support_headroom_inference_rejects_an_all_worse_rewrite_pool() -> None:
    report = support_headroom_inference(
        (-0.60, -0.20, -0.10, -0.30),
        ("easy", "easy", "hard", "hard"),
    )

    assert report["mean_supported_headroom_over_initial"] < 0.0
    assert report["all_strata_meet_minimum_coverage"] is False
    assert report["pass"] is False


def test_support_headroom_inference_requires_the_registered_lower_bound() -> None:
    report = support_headroom_inference(
        (0.90, -0.10, 0.10, 0.10),
        ("mixed", "mixed", "mixed", "mixed"),
    )

    assert report["mean_supported_headroom_over_initial"] > 0.02
    assert report["headroom_one_sided_lower_confidence_bound"] < 0.02
    assert report["all_strata_meet_minimum_coverage"] is True
    assert report["pass"] is False


def test_support_headroom_lower_bound_includes_finite_evaluator_uncertainty() -> None:
    report = support_headroom_inference(
        (0.03, 0.03, 0.03, 0.03),
        ("mixed", "mixed", "mixed", "mixed"),
        evaluator_reads=256,
    )

    assert report["between_lineage_standard_error"] == 0.0
    assert report["evaluator_standard_error_upper_bound"] > 0.0
    assert report["headroom_one_sided_lower_confidence_bound"] < 0.03
    assert report["pass"] is False


def test_support_headroom_inference_requires_coverage_in_every_sampled_stratum() -> None:
    report = support_headroom_inference(
        (*([0.10] * 10), -0.01),
        (*(["easy"] * 10), "hard"),
    )

    assert report["headroom_one_sided_lower_confidence_bound"] > 0.02
    assert report["per_stratum_headroom"]["hard"]["coverage"] == 0.0
    assert report["all_strata_meet_minimum_coverage"] is False
    assert report["pass"] is False


def test_support_headroom_inference_passes_broad_meaningful_headroom() -> None:
    report = support_headroom_inference(
        (0.05, 0.06, 0.07, 0.08),
        ("easy", "easy", "hard", "hard"),
    )

    assert report["headroom_one_sided_lower_confidence_bound"] >= 0.02
    assert report["all_strata_meet_minimum_coverage"] is True
    assert report["pass"] is True


def test_conformance_unknown_exact_label_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        "isingfold.rl.gates.exact_feasibility",
        lambda *args, **kwargs: SimpleNamespace(feasible=None),
    )

    report = gate_conformance([_task()], Context(qubit_cap=5, n_est_reads=8))

    assert report["eligible_instances"] == 1
    assert report["structural_checked"] == 0
    assert report["structural_unknown"] == 1
    assert report["pass"] is False


def test_conformance_checks_negative_and_relaxed_overlap_semantics() -> None:
    report = gate_conformance([_task()], Context(qubit_cap=5, n_est_reads=8))

    assert report["positive_structural_checked"] == 1
    assert report["positive_structural_agreement"] == 1.0
    assert report["negative_structural_checked"] == 1
    assert report["negative_structural_unknown"] == 0
    assert report["negative_structural_agreement"] == 1.0
    assert report["overlap_search_checked"] == 1
    assert report["overlap_search_admissible_rate"] == 1.0
    assert report["overlap_return_rejection_rate"] == 1.0
    assert report["pass"] is False


def test_conformance_pass_requires_exactly_eight_unique_tasks_and_lineages() -> None:
    base = _task()
    tasks = [
        replace(base, name=f"task-{index}", lineage=f"lineage-{index}")
        for index in range(EXACT_CONFORMANCE_MAX_TASKS)
    ]

    report = gate_conformance(tasks, Context(qubit_cap=5, n_est_reads=8))

    assert report["registered_population"] == {
        "exact_task_count": True,
        "task_count": EXACT_CONFORMANCE_MAX_TASKS,
        "unique_task_id_count": EXACT_CONFORMANCE_MAX_TASKS,
        "unique_base_lineage_count": EXACT_CONFORMANCE_MAX_TASKS,
    }
    assert report["pass"] is True

    repeated_task = gate_conformance(
        [replace(task, name="repeated-task") for task in tasks],
        Context(qubit_cap=5, n_est_reads=8),
    )
    repeated_lineage = gate_conformance(
        [replace(task, lineage="repeated-lineage") for task in tasks],
        Context(qubit_cap=5, n_est_reads=8),
    )
    assert repeated_task["pass"] is False
    assert repeated_lineage["pass"] is False


def test_conformance_negative_motif_denominator_does_not_scale_with_task_count() -> None:
    base = _task()
    tasks = [replace(base, name=f"task-{index}", lineage=f"lineage-{index}") for index in range(3)]

    report = gate_conformance(tasks, Context(qubit_cap=5, n_est_reads=8))

    assert report["eligible_instances"] == 3
    assert report["negative_structural_checked"] == 1
    assert report["negative_structural_unknown"] == 0


def test_exact_conformance_is_order_invariant_and_fails_closed_above_bounds() -> None:
    base = _task()
    first = replace(base, name="task-a", lineage="lineage-a")
    second = replace(base, name="task-b", lineage="lineage-b")
    context = Context(qubit_cap=20, n_est_reads=8)

    assert gate_conformance([first, second], context) == gate_conformance([second, first], context)
    with pytest.raises(ValueError, match="bounded task cap"):
        gate_conformance([base] * (EXACT_CONFORMANCE_MAX_TASKS + 1), context)

    logical = nx.path_graph(EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES + 1)
    host = nx.path_graph(EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES + 1)
    oversized = EmbeddingTask(
        name="oversized-exact-task",
        logical=logical,
        host=host,
        problem=LogicalProblem.from_dicts(
            {node: 0.25 for node in logical},
            {edge: -1.0 for edge in logical.edges()},
        ),
        ground_energy=-1.0,
        lineage="oversized-lineage",
        initial_embedding={node: frozenset((node,)) for node in logical},
    )
    with pytest.raises(ValueError, match="logical-size cap"):
        gate_conformance([oversized], context)


def test_conformance_rejects_structural_solver_disagreement_with_independent_oracle(
    monkeypatch,
) -> None:
    """A self-consistent primary solver cannot certify its own release gate."""

    monkeypatch.setattr(
        "isingfold.rl.gates.exact_feasibility",
        lambda *args, **kwargs: SimpleNamespace(feasible=False),
    )

    report = gate_conformance([_task()], Context(qubit_cap=5, n_est_reads=8))

    assert report["independent_oracle"]["version"] == "structural-bruteforce-v1"
    assert report["positive_structural_agreement"] == 0.0
    assert report["pass"] is False


def test_conformance_rejects_an_invalid_authenticated_witness_without_crashing() -> None:
    task = _task()
    invalid = EmbeddingTask(
        name=task.name,
        logical=task.logical,
        host=task.host,
        problem=task.problem,
        ground_energy=task.ground_energy,
        lineage=task.lineage,
        initial_embedding={0: frozenset((0,)), 1: frozenset((0,))},
    )

    report = gate_conformance([invalid], Context(qubit_cap=5, n_est_reads=8))

    assert report["witness_valid_rate"] == 0.0
    assert report["program_faithful_rate"] == 0.0
    assert report["pass"] is False


def test_valid_return_gate_separates_profile_i_and_empty_start_profile_c(
    monkeypatch,
) -> None:
    calls: list[tuple[Mode, object]] = []

    class FakeEnv:
        def __init__(self, task, ctx, *, mode, initializer, **kwargs):
            del task, ctx, kwargs
            calls.append((mode, initializer))
            self.mode = mode

        def reset(self, seed):
            del seed
            return TerminalRecord(
                returned_valid=self.mode is Mode.IMPROVEMENT,
                terminal_reason=(
                    TerminalReason.COMMIT
                    if self.mode is Mode.IMPROVEMENT
                    else TerminalReason.STOP_NO_VALID
                ),
                embedding=None,
                selected_program=None,
                selected_strength=None,
                selected_index=None,
                training_reward=0.5 if self.mode is Mode.IMPROVEMENT else 0.0,
                cumulative_work=WorkVector(),
                validation_receipt={},
            )

    monkeypatch.setattr("isingfold.rl.gates.EmbeddingEnv", FakeEnv)
    profile_i = gate_profile_i_signal(
        [_task()],
        Context(qubit_cap=5, n_est_reads=8),
        episodes=2,
        min_continuation_share=0.0,
        min_nonzero_utility_share=0.0,
    )
    profile_c = gate_profile_c_readiness([_task()], Context(qubit_cap=5, n_est_reads=8), episodes=2)

    assert profile_i["attempted_episodes"] == 2
    assert profile_i["valid_return_rate"] == 1.0
    assert profile_i["pass"] is True
    assert profile_c["attempted_episodes"] == 2
    assert profile_c["valid_return_rate"] == 0.0
    assert profile_c["empty_start"] is True
    assert profile_c["pass"] is False
    assert [mode for mode, _ in calls] == [
        Mode.IMPROVEMENT,
        Mode.IMPROVEMENT,
        Mode.CONSTRUCTION,
        Mode.CONSTRUCTION,
    ]
    assert all(initializer is None for mode, initializer in calls if mode is Mode.CONSTRUCTION)


def test_profile_c_failure_does_not_block_the_profile_i_compatibility_gate(
    monkeypatch,
) -> None:
    class FakeEnv:
        def __init__(self, task, ctx, *, mode, **kwargs):
            del task, ctx, kwargs
            self.mode = mode

        def reset(self, seed):
            del seed
            return TerminalRecord(
                returned_valid=self.mode is Mode.IMPROVEMENT,
                terminal_reason=(
                    TerminalReason.COMMIT
                    if self.mode is Mode.IMPROVEMENT
                    else TerminalReason.STOP_NO_VALID
                ),
                embedding=None,
                selected_program=None,
                selected_strength=None,
                selected_index=None,
                training_reward=0.5 if self.mode is Mode.IMPROVEMENT else 0.0,
                cumulative_work=WorkVector(),
                validation_receipt={},
            )

    monkeypatch.setattr("isingfold.rl.gates.EmbeddingEnv", FakeEnv)
    report = gate_valid_returns(
        [_task()],
        Context(qubit_cap=5, n_est_reads=8),
        episodes=1,
        min_profile_i_continuation_share=0.0,
        min_profile_i_nonzero_utility_share=0.0,
    )

    assert report["profile_i"]["pass"] is True
    assert report["profile_c"]["pass"] is False
    assert report["pass"] is True
    assert report["construction_ready"] is False


def test_selector_gate_uses_authenticated_calibration_graph_records(tmp_path, monkeypatch) -> None:
    selector = _GraphSelector()
    graphs = tuple(_graph(index, selector.normalizer_digest) for index in range(4))
    records = (
        StrengthRecord(
            features=tuple(
                {
                    "strength": graph.strength,
                    "scale": graph.scale,
                    "qubits": 4.0,
                    "max_chain": 2.0,
                    "mean_chain": 2.0,
                    "single_qubit_fraction": 0.0,
                    "max_field": 0.5,
                    "max_coupling": 1.0,
                    "mean_contacts": 1.0,
                    "single_contact_fraction": 1.0,
                    "strength_over_jmax": graph.strength,
                    "chain_edges": 2.0,
                }
                for graph in graphs
            ),
            hits=(0, 2, 4, 8),
            reads=(8, 8, 8, 8),
            lineage="calibration-lineage",
            graph_inputs=graphs,
        ),
    )
    metadata = SimpleNamespace(
        source_prepared_manifest_sha256="b" * 64,
        context_digest="c" * 64,
        normalizer_digest=selector.normalizer_digest,
        coefficient_scale=1.0,
        manifest_sha256="d" * 64,
    )
    monkeypatch.setattr(
        "isingfold.rl.gates.load_selector_metadata", lambda path: metadata, raising=False
    )
    monkeypatch.setattr(
        "isingfold.rl.gates.load_selector_records",
        lambda path, *, partition: records,
        raising=False,
    )

    report = gate_selector_discrimination(
        tmp_path,
        selector,
        expected_source_manifest_sha256=metadata.source_prepared_manifest_sha256,
        expected_context_digest=metadata.context_digest,
    )

    assert report["records"] == 1
    assert report["partition"] == "calibration"
    assert report["selected_mean"] == 1.0
    assert report["fixed_f2_mean"] == 0.25
    assert report["random_mean"] == (0.0 + 0.25 + 0.5 + 1.0) / 4.0
    assert report["random_estimator"] == "exact-uniform-four-strength-expectation"
    assert report["selector_label_manifest_sha256"] == metadata.manifest_sha256
    assert report["pass"] is True
