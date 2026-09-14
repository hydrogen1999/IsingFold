"""Focused tests for the same-support classical evaluation controls."""

from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import numpy as np

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import Candidate, Context, Opcode, WorkVector
from isingfold.rl.env import EmbeddingTask, fixed_strength_selector
from isingfold.rl.evaluate import (
    baseline_method_metadata,
    first_commit_controller,
    quality_aware_controller,
    resource_first_controller,
    run_controller,
)
from isingfold.rl.evaluator import ReadBlock
from isingfold.rl.tensorize import N_ACTION, N_ARCHIVE, N_FACTOR, N_LEDGE, pack, t_count, t_signed


def _candidate(opcode: Opcode, key: str, *, archive_ref: int | None = None) -> Candidate:
    changes = opcode not in (Opcode.COMMIT, Opcode.STOP)
    return Candidate(
        opcode=opcode,
        affected=("x",) if changes else (),
        old_chains={"x": frozenset({0})} if changes else {},
        new_chains={"x": frozenset({1})} if changes else {},
        work=WorkVector(decisions=1),
        payload_key=key,
        archive_ref=archive_ref,
    )


def _action_rows(rows: list[dict[int, float]]) -> np.ndarray:
    dense: list[list[float]] = []
    for sparse in rows:
        row = [0.0] * N_ACTION
        for index, value in sparse.items():
            row[index] = value
        dense.append(row)
    return pack(dense, N_ACTION)


def _minimal_observation(
    action_rows: list[dict[int, float]],
    *,
    archive_rows: list[list[float]] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        actions=_action_rows(action_rows),
        archive=pack(archive_rows or [[0.0] * N_ARCHIVE], N_ARCHIVE),
        factors=pack([], N_FACTOR),
        factor_roles=np.zeros(0, dtype=np.int64),
        index_factor_action=np.zeros((2, 0), dtype=np.int64),
        index_factor_archive=np.zeros((2, 0), dtype=np.int64),
        index_edge_use_factor=np.zeros(0, dtype=np.int64),
        edge_use_roles=np.zeros(0, dtype=np.int64),
        edge_use_logical=pack([], N_LEDGE),
    )


def test_resource_first_reduces_resources_and_randomises_only_exact_ties() -> None:
    candidates = (
        _candidate(Opcode.REWRITE_ONE, "rewrite-a"),
        _candidate(Opcode.REWRITE_ONE, "rewrite-b"),
        _candidate(Opcode.REWRITE_ONE, "rewrite-worse"),
        _candidate(Opcode.COMMIT, "commit", archive_ref=0),
    )
    observation = _minimal_observation(
        [
            {5: t_signed(-1), 14: t_count(1)},
            {5: t_signed(-1), 14: t_count(1)},
            {5: t_signed(1), 14: t_count(1)},
            {14: t_count(1)},
        ],
        archive_rows=[[0.0, 1.0, 1.0, t_count(5), t_count(2)]],
    )
    decision = SimpleNamespace(
        candidates=candidates,
        legal_mask=(True, True, True, True),
        observation=observation,
    )

    first = resource_first_controller(decision, np.random.default_rng(7))
    second = resource_first_controller(decision, np.random.default_rng(7))
    choices = {
        resource_first_controller(decision, np.random.default_rng(seed))
        for seed in range(32)
    }

    assert first == second
    assert choices == {0, 1}


def test_resource_first_commits_smallest_archive_when_no_move_improves() -> None:
    candidates = (
        _candidate(Opcode.REWRITE_ONE, "rewrite"),
        _candidate(Opcode.COMMIT, "large", archive_ref=0),
        _candidate(Opcode.COMMIT, "small", archive_ref=1),
    )
    observation = _minimal_observation(
        [{5: t_signed(1)}, {}, {}],
        archive_rows=[
            [0.0, 1.0, 1.0, t_count(9), t_count(3)],
            [0.0, 1.0, 1.0, t_count(6), t_count(4)],
        ],
    )
    decision = SimpleNamespace(
        candidates=candidates,
        legal_mask=(True, True, True),
        observation=observation,
    )

    assert resource_first_controller(decision, np.random.default_rng(0)) == 2


def test_quality_aware_prefers_weighted_contact_gain_without_evaluator_data() -> None:
    candidates = (
        _candidate(Opcode.REWRITE_ONE, "contact-gain"),
        _candidate(Opcode.REWRITE_ONE, "resource-only"),
        _candidate(Opcode.COMMIT, "commit", archive_ref=0),
    )
    observation = _minimal_observation(
        [
            {5: t_signed(1), 14: t_count(1)},
            {5: t_signed(-1), 14: t_count(1)},
            {},
        ],
        archive_rows=[[0.0, 1.0, 1.0, t_count(4), t_count(2)]],
    )
    factor_rows = [
        [0.0, t_count(2), t_count(1), t_count(1), 0.0, 0.0, 1.0, 1.0],
        [1.0, t_count(3), t_count(2), t_count(2), 0.0, 0.0, 1.0, 1.0],
        [0.0, t_count(2), t_count(1), t_count(1), 0.0, 0.0, 1.0, 1.0],
        [1.0, t_count(1), 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
    ]
    observation.factors = pack(factor_rows, N_FACTOR)
    observation.factor_roles = np.asarray([0, 1, 0, 1], dtype=np.int64)
    observation.index_factor_action = np.asarray(
        [[0, 1, 2, 3], [0, 0, 1, 1]], dtype=np.int64
    )
    # Action 0 grows from one to two contacts. Action 1 keeps one contact.
    observation.index_edge_use_factor = np.asarray([0, 1, 1, 2, 3], dtype=np.int64)
    observation.edge_use_roles = np.ones(5, dtype=np.int64)
    edge_rows = []
    for _ in range(5):
        row = [0.0] * N_LEDGE
        row[1] = t_count(1)
        edge_rows.append(row)
    observation.edge_use_logical = pack(edge_rows, N_LEDGE)
    decision = SimpleNamespace(
        candidates=candidates,
        legal_mask=(True, True, True),
        observation=observation,
    )

    assert quality_aware_controller(decision, np.random.default_rng(123)) == 0


def test_baseline_metadata_declares_no_oracle_and_excludes_stock_minorminer() -> None:
    metadata = baseline_method_metadata()

    assert set(metadata) == {
        "return_initial",
        "random_masked",
        "classical_resource_first",
        "classical_quality_aware",
    }
    assert all(item["online_evaluator_feedback"] is False for item in metadata.values())
    assert all(item["evaluator_oracle"] is False for item in metadata.values())
    assert not any("minorminer" in item["method_id"] for item in metadata.values())


def test_final_audit_block_can_exceed_training_reserve_and_is_charged_once(monkeypatch) -> None:
    logical = nx.path_graph(3)
    host = nx.path_graph(3)
    problem = LogicalProblem.from_dicts(
        {node: 0.0 for node in logical},
        {edge: -1.0 for edge in logical.edges()},
    )
    task = EmbeddingTask(
        "audit",
        logical,
        host,
        problem,
        -2.0,
        lineage="audit-lineage",
        initial_embedding={node: frozenset({node}) for node in logical},
    )
    calls: list[int] = []

    def fake_sample(
        program,
        chains,
        logical_problem,
        ground_energy,
        *,
        num_reads,
        seed,
        num_sweeps,
    ) -> ReadBlock:
        del program, chains, logical_problem, ground_energy, seed, num_sweeps
        calls.append(num_reads)
        return ReadBlock(num_reads // 2, num_reads, 0.25, 0.0, 1)

    monkeypatch.setattr("isingfold.rl.evaluate.sample_program", fake_sample)
    context = Context(qubit_cap=3)
    outcomes = run_controller(
        [task],
        context,
        first_commit_controller,
        initializer=lambda logical, host, seed: None,
        selector=fixed_strength_selector(),
        seed=11,
    )

    assert calls == [context.audit_reads]
    assert outcomes[0].evaluator_reads == context.audit_reads
    assert outcomes[0].work is not None
    assert outcomes[0].work.evaluator_reads == context.audit_reads
    assert context.audit_reads > context.reserve.evaluator_reads
