"""Adversarial checks for the whole exact ``P_search`` state contract."""

from __future__ import annotations

from dataclasses import replace

import networkx as nx
import pytest

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import Context, DecisionState, Mode, WorkVector
from isingfold.rl.env import EmbeddingEnv, EmbeddingTask, IntegrityError
from isingfold.rl.proposal import LEGACY_ONLINE_INITIALIZER_RESTARTS_V1


def _task() -> EmbeddingTask:
    logical = nx.Graph([(0, 1)])
    host = nx.cycle_graph(6)
    problem = LogicalProblem.from_dicts({0: 0.25, 1: -0.5}, {(0, 1): -1.0})
    return EmbeddingTask("integrity", logical, host, problem, -1.25, "integrity-0")


def _improvement_env(*, debit: WorkVector = WorkVector()) -> EmbeddingEnv:
    initial = {0: frozenset({0}), 1: frozenset({1})}
    env = EmbeddingEnv(
        _task(),
        Context(qubit_cap=6),
        initializer=lambda *_: initial,
        budget_debit=debit,
        reward_reads=4,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=17,
    )
    decision = env.reset()
    assert isinstance(decision, DecisionState)
    assert env.state is not None
    return env


def test_reset_initializes_every_age_map_on_its_exact_domain() -> None:
    env = _improvement_env()
    assert env.state is not None
    state = env.state
    assert state.ages.chain == {0: 0, 1: 0}
    assert state.ages.claim == {(0, 0): 0, (1, 1): 0}
    assert state.ages.conflict == {}
    assert state.ages.demand == {(0, 1): 0}
    assert state.ages.occupancy == {0: 0, 1: 0}

    constructor = EmbeddingEnv(
        _task(), Context(qubit_cap=6), mode=Mode.CONSTRUCTION, seed=19
    )
    decision = constructor.reset()
    assert isinstance(decision, DecisionState)
    assert constructor.state is not None
    assert constructor.state.ages.chain == {0: 0, 1: 0}
    assert constructor.state.ages.claim == {}
    assert constructor.state.ages.conflict == {}
    assert constructor.state.ages.demand == {(0, 1): 0}
    assert constructor.state.ages.occupancy == {}


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda env: env.state.chains.pop(1), "logical domain"),
        (lambda env: env.state.ages.claim.pop((0, 0)), "claim ages"),
        (
            lambda env: env.state.ages.demand.__setitem__((1, 0), 0),
            "demand ages",
        ),
        (lambda env: env.state.ages.chain.__setitem__(0, -1), "chain ages"),
        (lambda env: setattr(env.state, "restarts_left", -1), "restart"),
        (lambda env: setattr(env.state, "decisions_used", 1), "decision count"),
        (lambda env: setattr(env.state, "workspace_valid", False), "workspace-valid"),
        (lambda env: setattr(env.state, "reference_program", None), "reference program"),
    ],
)
def test_each_exact_state_family_fails_closed(mutate, match: str) -> None:
    env = _improvement_env()
    mutate(env)
    with pytest.raises(IntegrityError, match=match):
        env._assert_search_state()


def test_invalid_archive_cannot_be_blessed_by_self_consistent_metadata() -> None:
    env = _improvement_env()
    assert env.state is not None
    overlapping = {0: frozenset({0}), 1: frozenset({0})}
    env.state.archive[0] = env._entry(overlapping, protected=True)

    with pytest.raises(IntegrityError, match="P_return"):
        env._assert_search_state()


def test_archive_protection_capacity_and_dedup_are_integrity_conditions() -> None:
    env = _improvement_env()
    assert env.state is not None
    entry = env.state.archive[0]
    env.state.archive[0] = replace(entry, protected=False)
    with pytest.raises(IntegrityError, match="protected"):
        env._assert_search_state()

    env = _improvement_env()
    assert env.state is not None
    env.state.archive.append(replace(env.state.archive[0], protected=False))
    with pytest.raises(IntegrityError, match="duplicate"):
        env._assert_search_state()


def test_work_identity_includes_spent_remaining_and_pre_environment_debit() -> None:
    debit = WorkVector(route_expansions=3, materializations=1)
    env = _improvement_env(debit=debit)
    assert env.state is not None
    env.state.remaining = env.state.remaining - WorkVector(feature_work=1)
    with pytest.raises(IntegrityError, match="ledger"):
        env._assert_search_state()

    env = _improvement_env(debit=debit)
    assert env.state is not None
    env.state.budget_debit = WorkVector()
    with pytest.raises(IntegrityError, match="budget debit"):
        env._assert_search_state()


def test_integrity_seal_rejects_a_plausible_but_unregistered_counter_change() -> None:
    env = _improvement_env()
    assert env.state is not None
    env.state.restarts_left -= 1
    # The value remains locally in range, but no selected RESTART caused the decrement.
    with pytest.raises(IntegrityError, match="outside a registered transition"):
        env._assert_search_state()


def test_reference_program_must_be_the_exact_current_f1_compilation() -> None:
    env = _improvement_env()
    assert env.state is not None
    assert env.state.reference_program is not None
    env.state.reference_program = replace(
        env.state.reference_program,
        strength_index=env.state.reference_program.strength_index + 1,
    )
    with pytest.raises(IntegrityError, match="reference program"):
        env._assert_search_state()


def test_occupancy_age_resets_when_owner_changes_at_equal_count() -> None:
    env = _improvement_env()
    assert env.state is not None
    previous = dict(env.state.chains)
    # Qubit zero remains singly occupied, but ownership changes from logical 0 to logical 1.
    env.state.chains = {0: frozenset({2}), 1: frozenset({0})}
    env._advance_ages(previous)
    assert env.state.ages.occupancy[0] == 0


def test_public_preparation_boundary_rejects_corruption_instead_of_emitting_reward() -> None:
    env = _improvement_env()
    assert env.state is not None
    env.state.ages.occupancy.clear()
    with pytest.raises(IntegrityError, match="occupancy ages"):
        env._prepare(training_labels=False)
