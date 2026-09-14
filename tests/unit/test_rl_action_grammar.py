"""Opcode preconditions and Profile-C construction semantics."""

from __future__ import annotations

from dataclasses import replace

import networkx as nx

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import (
    Candidate,
    Context,
    DecisionState,
    Mode,
    Opcode,
    RestartCacheSlot,
    WorkVector,
    stable_digest,
)
from isingfold.rl.env import EmbeddingEnv, EmbeddingTask
from isingfold.rl.proposal import (
    LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
    bound_successor_key,
)


def _task() -> EmbeddingTask:
    logical = nx.Graph([(0, 1)])
    host = nx.cycle_graph(6)
    problem = LogicalProblem.from_dicts({0: 0.0, 1: 0.0}, {(0, 1): -1.0})
    return EmbeddingTask("construct", logical, host, problem, -1.0, "lineage-c")


def _state_actions(decision: DecisionState) -> list[Candidate]:
    return [candidate for candidate in decision.candidates if candidate.changes_workspace]


def test_empty_constructor_exposes_place_and_never_rewrite() -> None:
    decision = EmbeddingEnv(_task(), Context(qubit_cap=6), mode=Mode.CONSTRUCTION, seed=3).reset()
    assert isinstance(decision, DecisionState)
    opcodes = {candidate.opcode for candidate in _state_actions(decision)}
    assert opcodes == {Opcode.PLACE}
    assert all(len(candidate.affected) == 1 for candidate in _state_actions(decision))


def test_construction_routes_only_after_both_demand_endpoints_are_placed() -> None:
    env = EmbeddingEnv(_task(), Context(qubit_cap=6), mode=Mode.CONSTRUCTION, seed=5)
    decision = env.reset()
    assert isinstance(decision, DecisionState)
    first = next(i for i, candidate in enumerate(decision.candidates) if candidate.opcode is Opcode.PLACE)
    decision = env.step(decision, first, evaluate_training_reward=False).next_decision_or_terminal
    assert isinstance(decision, DecisionState)
    assert Opcode.ROUTE not in {candidate.opcode for candidate in _state_actions(decision)}

    other = next(
        i
        for i, candidate in enumerate(decision.candidates)
        if candidate.opcode is Opcode.PLACE and candidate.affected != (0,)
    )
    decision = env.step(decision, other, evaluate_training_reward=False).next_decision_or_terminal
    assert isinstance(decision, DecisionState)
    state_actions = _state_actions(decision)
    if any(candidate.opcode is Opcode.ROUTE for candidate in state_actions):
        assert all(
            candidate.target_demand == (0, 1)
            for candidate in state_actions
            if candidate.opcode is Opcode.ROUTE
        )


def _bound_route_candidate(
    env: EmbeddingEnv,
    *,
    routes: tuple[tuple[tuple[int, ...], int], ...],
) -> Candidate:
    assert env.state is not None
    old = {0: frozenset({0})}
    new = {0: frozenset({0, 1, 2})}
    successor = dict(env.state.chains)
    successor.update(new)
    work = WorkVector(decisions=1, compiler_calls=4, validator_calls=1)
    return Candidate(
        opcode=Opcode.ROUTE,
        affected=(0,),
        old_chains=old,
        new_chains=new,
        work=work,
        payload_key=bound_successor_key(successor, work, restart=False),
        proposal_work=WorkVector(materializations=1),
        routes=routes,
        target_demand=(0, 1),
        provenance="route-payload-contract-test",
    )


def test_route_payload_exactly_accounts_for_added_vertices_and_attachment() -> None:
    env = EmbeddingEnv(_task(), Context(qubit_cap=6), mode=Mode.CONSTRUCTION, seed=19)
    env.reset()
    assert env.state is not None
    env.state.chains = {0: frozenset({0}), 1: frozenset({3})}

    complete = _bound_route_candidate(env, routes=(((0, 1, 2), 0),))
    omitted_vertices = _bound_route_candidate(env, routes=(((0,), 0),))
    unattached_segment = _bound_route_candidate(env, routes=(((1, 2), 0),))

    assert env._is_legal(complete)
    assert not env._is_legal(omitted_vertices)
    assert not env._is_legal(unattached_segment)


def test_construction_restart_is_bound_to_empty_workspace() -> None:
    env = EmbeddingEnv(_task(), Context(qubit_cap=6), mode=Mode.CONSTRUCTION, seed=7)
    decision = env.reset()
    assert isinstance(decision, DecisionState)
    place = next(i for i, candidate in enumerate(decision.candidates) if candidate.opcode is Opcode.PLACE)
    decision = env.step(decision, place, evaluate_training_reward=False).next_decision_or_terminal
    assert isinstance(decision, DecisionState)
    restart = next(candidate for candidate in decision.candidates if candidate.opcode is Opcode.RESTART)
    assert set(restart.new_chains) == set(_task().logical.nodes())
    assert all(not chain for chain in restart.new_chains.values())


def test_improvement_restart_resets_workspace_clocks_but_ages_archive() -> None:
    task = _task()
    initial = {0: frozenset({0}), 1: frozenset({1})}
    env = EmbeddingEnv(
        task,
        Context(qubit_cap=6),
        mode=Mode.IMPROVEMENT,
        initializer=lambda *_: initial,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=13,
    )
    decision = env.reset()
    assert isinstance(decision, DecisionState)
    restart = next(
        index
        for index, candidate in enumerate(decision.candidates)
        if candidate.opcode is Opcode.RESTART
    )

    result = env.step(
        decision,
        restart,
        evaluate_training_reward=False,
    ).next_decision_or_terminal

    assert isinstance(result, DecisionState)
    assert env.state is not None
    ages = env.state.ages
    assert all(age == 0 for mapping in vars(ages).values() for age in mapping.values())
    assert env.state.archive[0].age == 1


def test_selected_cached_restart_consumes_exact_slot_but_unselected_cache_persists() -> None:
    task = _task()
    initial = {0: frozenset({0}), 1: frozenset({1})}
    cached = {0: frozenset({2}), 1: frozenset({3})}
    manifest_digest = stable_digest({"bank": "restart-cache-test"})
    slots = (
        RestartCacheSlot(
            slot_index=0,
            status="SUCCESS",
            chains=cached,
            snapshot_record_digest=stable_digest({"slot": 0}),
            attempt_receipt_root=stable_digest({"attempts": 0}),
        ),
        RestartCacheSlot(
            slot_index=1,
            status="FAILED",
            chains=None,
            snapshot_record_digest=stable_digest({"slot": 1}),
            attempt_receipt_root=stable_digest({"attempts": 1}),
        ),
    )
    context = Context(qubit_cap=6, quotas={"restart": 2})
    env = EmbeddingEnv(
        task,
        context,
        mode=Mode.IMPROVEMENT,
        initializer=lambda *_: initial,
        restart_cache=slots,
        restart_cache_manifest_digest=manifest_digest,
        budget_debit=WorkVector(restart_work=3),
        initializer_precomputed=True,
        seed=13,
    )

    decision = env.reset()
    assert isinstance(decision, DecisionState)
    restart_rows = [
        (index, candidate)
        for index, candidate in enumerate(decision.candidates)
        if candidate.opcode is Opcode.RESTART
    ]
    assert len(restart_rows) == 1
    restart_index, restart = restart_rows[0]
    assert restart.restart_cache_slot == 0
    assert env.state is not None
    assert [slot.consumed for slot in env.state.restart_cache] == [False, False]

    result = env.step(
        decision,
        restart_index,
        evaluate_training_reward=False,
    ).next_decision_or_terminal

    assert isinstance(result, DecisionState)
    assert env.state is not None
    assert [slot.consumed for slot in env.state.restart_cache] == [True, False]
    assert env.state.restarts_left == 1
    assert all(candidate.opcode is not Opcode.RESTART for candidate in result.candidates)


def test_opcode_specific_legality_rejects_a_forged_rewrite_payload() -> None:
    task = _task()
    initial = {0: frozenset({0}), 1: frozenset({1})}
    env = EmbeddingEnv(
        task,
        Context(qubit_cap=6),
        mode=Mode.IMPROVEMENT,
        initializer=lambda *_: initial,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=2,
    )
    decision = env.reset()
    assert isinstance(decision, DecisionState)
    forged = Candidate(
        opcode=Opcode.REWRITE_ONE,
        affected=(0, 1),
        old_chains={0: initial[0]},
        new_chains={0: frozenset({5})},
        work=WorkVector(decisions=1),
        payload_key="forged",
    )
    assert not env._is_legal(forged)


def test_newly_unrealized_demand_age_starts_at_zero() -> None:
    task = _task()
    initial = {0: frozenset({0}), 1: frozenset({1})}
    env = EmbeddingEnv(
        task,
        Context(qubit_cap=6),
        initializer=lambda *_: initial,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=11,
    )
    decision = env.reset()
    assert isinstance(decision, DecisionState)
    assert env.state is not None
    previous = dict(env.state.chains)
    env.state.chains = {0: frozenset({0}), 1: frozenset({3})}
    env._advance_ages(previous)
    assert env.state.ages.demand[(0, 1)] == 0


def test_removed_claim_loses_its_age_record() -> None:
    task = _task()
    initial = {0: frozenset({0, 5}), 1: frozenset({1})}
    env = EmbeddingEnv(
        task,
        Context(qubit_cap=6),
        initializer=lambda *_: initial,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=17,
    )
    decision = env.reset()
    assert isinstance(decision, DecisionState)
    assert env.state is not None
    env.state.ages.claim[(0, 5)] = 4
    previous = dict(env.state.chains)
    env.state.chains = {0: frozenset({0}), 1: frozenset({1})}

    env._advance_ages(previous)

    assert (0, 5) not in env.state.ages.claim


def test_restore_legality_is_bound_to_the_referenced_archive_payload() -> None:
    task = _task()
    archived = {0: frozenset({0}), 1: frozenset({1})}
    current = {0: frozenset({4}), 1: frozenset({5})}
    env = EmbeddingEnv(
        task,
        Context(qubit_cap=6, quotas={"restart": 4}),
        initializer=lambda *_: archived,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=23,
    )
    decision = env.reset()
    assert isinstance(decision, DecisionState)
    assert env.state is not None
    env.state.chains = dict(current)
    receipt, programs = env._validated_return(current, fresh=True)
    assert receipt.valid
    env.state.workspace_valid = receipt.valid
    env.state.reference_program = programs[0]
    env.state.ages = env._zero_workspace_ages(current)
    env._refresh_integrity_seal()

    prepared = env._prepare(training_labels=False)
    assert isinstance(prepared, DecisionState)
    restore_index = next(
        index
        for index, candidate in enumerate(prepared.candidates)
        if candidate.opcode is Opcode.REWRITE_GROUP and candidate.archive_ref is not None
    )
    restore = prepared.candidates[restore_index]
    assert prepared.legal_mask[restore_index]
    assert all(restore.new_chains[node] == archived[node] for node in restore.affected)

    forged_new = dict(restore.new_chains)
    forged_new[restore.affected[0]] = current[restore.affected[0]]
    forged_successor = dict(current)
    forged_successor.update(forged_new)
    forged = replace(
        restore,
        new_chains=forged_new,
        payload_key=bound_successor_key(forged_successor, restore.work, restart=False),
    )
    assert not env._is_legal(forged)

    archive_before = [(entry.key, dict(entry.chains)) for entry in env.state.archive]
    restarts_before = env.state.restarts_left
    result = env.step(prepared, restore_index, evaluate_training_reward=False)

    assert result.work_receipt == restore.work
    assert result.work_receipt.restart_work == 0
    assert env.state.restarts_left == restarts_before
    assert all(env.state.chains[node] == archived[node] for node in restore.affected)
    assert [(entry.key, dict(entry.chains)) for entry in env.state.archive] == archive_before


def test_selected_restart_does_not_recharge_initializer_proposal_work() -> None:
    task = _task()
    initial = {0: frozenset({0}), 1: frozenset({1})}
    fresh = {0: frozenset({2}), 1: frozenset({3})}
    calls = 0

    def initializer(_logical, _host, _seed):
        nonlocal calls
        calls += 1
        return initial if calls == 1 else fresh

    env = EmbeddingEnv(
        task,
        Context(qubit_cap=6, quotas={"restart": 4}),
        initializer=initializer,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=29,
    )
    decision = env.reset()
    assert isinstance(decision, DecisionState)
    restart_index = next(
        index
        for index, candidate in enumerate(decision.candidates)
        if candidate.opcode is Opcode.RESTART
    )
    restart = decision.candidates[restart_index]
    assert restart.proposal_work.restart_work == 1
    assert restart.work.restart_work == 0
    assert decision.charged_work_receipt.restart_work >= 1

    result = env.step(decision, restart_index, evaluate_training_reward=False)

    assert result.work_receipt.restart_work == 0
    assert env.state is not None
    assert env.state.restarts_left == 1
