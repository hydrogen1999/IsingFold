from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from lac_minorminer import SearchSession
from lac_minorminer.components import (
    ControlContext,
    ControlDecision,
    DefaultVertexSelector,
    GreedyAcceptancePolicy,
    NativeCandidateProvider,
    NoRouteCostProvider,
    StagnationControlPolicy,
    ViolationNeighborhoodSelector,
    WideCandidateProvider,
)
from lac_minorminer.scoring import ResourceScorer


def test_public_session_normalizes_labels_but_policies_use_compact_ids() -> None:
    session = SearchSession(
        [("left", "right")],
        [("x", "middle"), ("middle", "y")],
        random_seed=3,
        max_candidates=2,
    )

    assert session.source_labels == ("left", "right")
    assert session.target_labels == ("x", "middle", "y")
    assert len(session.snapshot().chains) == 2


def test_classical_components_are_deterministic_and_resource_ordered() -> None:
    snapshot = SimpleNamespace(conflicts=[2, 2, 0], missing_incident_edges=[0, 1, 5])
    selector = DefaultVertexSelector()
    assert selector.select(snapshot, [0, 1], random.Random(9)) == 1

    first_rng = random.Random(21)
    second_rng = random.Random(21)
    tied = SimpleNamespace(conflicts=[1, 1], missing_incident_edges=[0, 0])
    assert selector.select(tied, [0, 1], first_rng) == selector.select(tied, [0, 1], second_rng)

    session = SearchSession([], [(0, 1), (1, 2)], random_seed=4, max_candidates=2)
    assert NoRouteCostProvider().costs(session.snapshot(), 0, 3) is None

    isolated = SearchSession(
        GraphLike(["u"], []), [(0, 1), (1, 2)], random_seed=4, max_candidates=2
    )
    batch = NativeCandidateProvider().propose(isolated, 0, [4.0, 1.0, 2.0])
    scores = ResourceScorer().score(isolated.snapshot(), batch)
    assert scores == [0.0, 1.0]
    assert GreedyAcceptancePolicy().choose(scores, batch) == 0
    isolated.discard(batch)


def test_default_control_restarts_only_after_a_repeated_state() -> None:
    policy = StagnationControlPolicy(max_repeated_visits=3)
    base = dict(attempt=1, transitions=2, accepted=True, has_candidates=True)

    assert (
        policy.decide(ControlContext(repeated_state_visits=2, **base)) is ControlDecision.CONTINUE
    )
    assert policy.decide(ControlContext(repeated_state_visits=3, **base)) is ControlDecision.RESTART


def test_violation_neighborhood_selector_is_seeded_bounded_and_pressure_first() -> None:
    snapshot = SimpleNamespace(
        chains=[[0]] * 8,
        conflicts=[0, 2, 0, 1, 0, 2, 0, 0],
        missing_incident_edges=[9, 0, 0, 5, 1, 1, 0, 0],
    )
    selector = ViolationNeighborhoodSelector(max_size=4)

    first = selector.select(snapshot, random.Random(17))
    second = selector.select(snapshot, random.Random(17))

    assert first == second
    assert len(first) == 4
    assert set(first[:3]) == {1, 3, 5}
    assert len(set(first)) == len(first)

    seven_logicals = SimpleNamespace(
        chains=[[0]] * 7,
        conflicts=[0] * 7,
        missing_incident_edges=[0] * 7,
    )
    assert len(selector.select(seven_logicals, random.Random(17))) == 3


def test_public_session_validates_an_explicit_route_cost_vector() -> None:
    session = SearchSession(GraphLike(["u"], []), [(0, 1), (1, 2)], random_seed=4, max_candidates=2)

    with pytest.raises(ValueError, match="one finite non-negative cost"):
        session.propose(0, [])
    with pytest.raises(ValueError, match="one finite non-negative cost"):
        session.propose(0, [1.0, float("nan"), 1.0])

    batch = session.propose(0)
    session.discard(batch)


def test_wide_candidate_provider_returns_an_applicable_bounded_batch() -> None:
    session = SearchSession(
        GraphLike(["logical"], []),
        GraphLike(["left", "middle", "right"], [("left", "middle"), ("middle", "right")]),
        random_seed=17,
        max_candidates=1,
    )

    batch = WideCandidateProvider(scoring_candidates=3).propose(session, 0, [4.0, 1.0, 2.0])

    assert [candidate.candidate_id for candidate in batch.candidates] == [0, 1, 2]
    session.apply(batch, 2)
    assert session.snapshot().chains == [[0]]


@pytest.mark.parametrize("bound", [True, 0, -1, 1.5])
def test_wide_candidate_provider_rejects_an_invalid_bound(bound) -> None:
    with pytest.raises(ValueError, match="scoring_candidates"):
        WideCandidateProvider(scoring_candidates=bound)


def test_wide_candidate_provider_cannot_narrow_the_session_default() -> None:
    session = SearchSession(
        GraphLike(["logical"], []),
        [(0, 1), (1, 2)],
        random_seed=4,
        max_candidates=2,
    )

    with pytest.raises(ValueError, match="scoring_candidates"):
        WideCandidateProvider(scoring_candidates=1).propose(session, 0, None)


class GraphLike:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def is_directed(self) -> bool:
        return False
