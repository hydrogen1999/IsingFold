from __future__ import annotations

from dataclasses import replace

import networkx as nx
import pytest

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import (
    Candidate,
    Context,
    DecisionState,
    Mode,
    Opcode,
    OverlapProfile,
    WorkVector,
)
from isingfold.rl.env import EmbeddingEnv, EmbeddingTask, IntegrityError
from isingfold.rl.proposal import (
    LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
    ProposalBatch,
    bound_successor_key,
)
from isingfold.rl.validate import p_search


def _environment(*, overlap: OverlapProfile | None = None) -> EmbeddingEnv:
    logical = nx.Graph([(0, 1)])
    host = nx.path_graph(4)
    problem = LogicalProblem.from_dicts({0: 0.25, 1: -0.5}, {(0, 1): -1.25})
    initializer = lambda _logical, _host, _seed: {  # noqa: E731
        0: frozenset({0}),
        1: frozenset({1}),
    }
    context = Context(qubit_cap=4, overlap=overlap or OverlapProfile())
    return EmbeddingEnv(
        EmbeddingTask("tiny", logical, host, problem, ground_energy=-1.5, lineage="tiny-0"),
        context,
        mode=Mode.IMPROVEMENT,
        initializer=initializer,
        reward_reads=4,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=7,
    )


def _reset(env: EmbeddingEnv) -> DecisionState:
    decision = env.reset()
    assert isinstance(decision, DecisionState)
    return decision


def _candidate_index(decision: DecisionState, opcode: Opcode) -> int:
    return next(i for i, candidate in enumerate(decision.candidates) if candidate.opcode is opcode)


def test_one_selected_state_change_consumes_exactly_one_decision() -> None:
    env = _environment()
    decision = _reset(env)
    index = next(
        i for i, candidate in enumerate(decision.candidates) if candidate.changes_workspace
    )
    before = env.state.spent.decisions  # type: ignore[union-attr]

    env.step(decision, index, evaluate_training_reward=False)

    after = env.state.spent.decisions  # type: ignore[union-attr]
    assert after - before == 1


def test_group_swap_is_checked_and_applied_atomically_under_o0() -> None:
    overlap = OverlapProfile(name="O0", max_occupancy=1, excess_fraction_of_qubit_cap=0.0)
    env = _environment(overlap=overlap)
    old = {0: frozenset({0}), 1: frozenset({1})}
    new = {0: frozenset({1}), 1: frozenset({0})}
    successor = dict(old)
    successor.update(new)
    work = WorkVector(decisions=1, compiler_calls=4, validator_calls=1)
    proposal_work = WorkVector(materializations=1)
    candidate = Candidate(
        opcode=Opcode.REWRITE_GROUP,
        affected=(0, 1),
        old_chains=old,
        new_chains=new,
        work=work,
        payload_key=bound_successor_key(successor, work, restart=False),
        proposal_work=proposal_work,
        provenance="atomic-swap-test",
    )

    class SwapGenerator:
        def generate(
            self,
            _chains,
            _rng,
            *,
            restarts_left=0,
            archive=(),
            restart_cache=(),
            allowance=None,
        ):
            del restarts_left, archive, restart_cache, allowance
            return ProposalBatch((candidate,), proposal_work, {"group2": 1}, 0)

    env.generator = SwapGenerator()  # type: ignore[assignment]
    prepared = _reset(env)

    sequential_intermediate = {0: new[0], 1: old[1]}
    assert not p_search(
        sequential_intermediate,
        env.task.logical,
        env.task.host,
        env.ctx.qubit_cap,
        overlap,
    ).valid
    assert env._is_legal(candidate)

    selected = next(
        i for i, item in enumerate(prepared.candidates) if item.opcode is Opcode.REWRITE_GROUP
    )
    env.step(prepared, selected, evaluate_training_reward=False)

    assert env.state is not None
    assert env.state.chains == successor


def test_exact_state_fingerprint_detects_budget_mutation() -> None:
    env = _environment()
    decision = _reset(env)
    assert env.state is not None
    env.state.remaining = env.state.remaining - WorkVector(feature_work=1)
    commit = _candidate_index(decision, Opcode.COMMIT)

    with pytest.raises(IntegrityError, match="ledger|stale"):
        env.step(decision, commit, evaluate_training_reward=False)


def test_exact_state_fingerprint_binds_future_proposal_randomness() -> None:
    env = _environment()
    decision = _reset(env)
    commit = _candidate_index(decision, Opcode.COMMIT)

    # Proposal randomness is part of the transition state even though it is never exposed
    # as a neural feature.  Mutating it must invalidate an already prepared action support.
    env.rng.random()

    with pytest.raises(IntegrityError, match="stale"):
        env.step(decision, commit, evaluate_training_reward=False)


def test_improvement_rejects_caps_that_cannot_pay_initialization_and_terminal_reserve() -> None:
    env = _environment()
    too_small = replace(
        env.ctx.caps,
        compiler_calls=env.ctx.reserve.compiler_calls,
        validator_calls=env.ctx.reserve.validator_calls,
        restart_work=0,
    )

    with pytest.raises(ValueError, match="initialization.*reserve"):
        EmbeddingEnv(
            env.task,
            replace(env.ctx, caps=too_small),
            mode=Mode.IMPROVEMENT,
            initializer=env.initializer,
            reward_reads=4,
            improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        )


def test_terminal_reserve_keeps_commit_legal_with_a_full_archive() -> None:
    env = _environment()
    _reset(env)
    assert env.state is not None
    assignments = (
        {0: frozenset({0}), 1: frozenset({1})},
        {0: frozenset({1}), 1: frozenset({0})},
        {0: frozenset({1}), 1: frozenset({2})},
        {0: frozenset({2}), 1: frozenset({1})},
        {0: frozenset({2}), 1: frozenset({3})},
        {0: frozenset({3}), 1: frozenset({2})},
        {0: frozenset({0, 1}), 1: frozenset({2})},
        {0: frozenset({0}), 1: frozenset({1, 2})},
    )
    for chains in assignments:
        receipt, _ = env._validated_return(chains, fresh=True)
        assert receipt.valid
    env.state.archive = [
        env._entry(chains, protected=index == 0)
        for index, chains in enumerate(assignments)
    ]
    env.state.remaining = env.ctx.reserve
    env.state.spent = env.ctx.caps - env.ctx.reserve
    env.state.decisions_used = env.state.spent.decisions
    env._refresh_integrity_seal()

    decision = env._prepare(training_labels=False)

    assert isinstance(decision, DecisionState)
    commits = [
        legal
        for candidate, legal in zip(
            decision.candidates,
            decision.legal_mask,
            strict=True,
        )
        if candidate.opcode is Opcode.COMMIT
    ]
    assert len(commits) == 8
    assert all(commits)


def test_bound_payload_tampering_is_an_integrity_error() -> None:
    env = _environment()
    decision = _reset(env)
    commit = _candidate_index(decision, Opcode.COMMIT)
    candidates = list(decision.candidates)
    candidates[commit] = replace(candidates[commit], payload_key="forged-payload")
    forged = replace(decision, candidates=tuple(candidates))

    with pytest.raises(IntegrityError, match="payload"):
        env.step(forged, commit, evaluate_training_reward=False)


def test_out_of_range_strength_selector_is_not_silently_clamped() -> None:
    env = _environment()
    env.selector = lambda _programs, _features: 99
    decision = _reset(env)
    commit = _candidate_index(decision, Opcode.COMMIT)

    with pytest.raises(IntegrityError, match="selector"):
        env.step(decision, commit, evaluate_training_reward=False)
