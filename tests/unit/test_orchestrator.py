from __future__ import annotations

import random

import pytest

from lac_minorminer import SearchSession
from lac_minorminer.orchestrator import SearchOrchestrator


class GraphLike:
    def __init__(self, nodes, edges):
        self.nodes = nodes
        self.edges = edges

    def is_directed(self) -> bool:
        return False


class FixedSelector:
    def __init__(self, logical_id: int):
        self.logical_id = logical_id
        self.calls = 0

    def select(self, snapshot, eligible, rng) -> int:
        self.calls += 1
        return self.logical_id


class ReverseScorer:
    def __init__(self):
        self.calls = 0

    def score(self, snapshot, candidates):
        self.calls += 1
        return [-float(index) for index in range(len(candidates.candidates))]


def colliding_session(seed: int) -> SearchSession:
    source = GraphLike(["a", "b"], [])
    target = [(0, 1), (1, 2)]
    return SearchSession(source, target, random_seed=seed, max_candidates=3)


def find_colliding_seed() -> int:
    for seed in range(100):
        if colliding_session(seed).snapshot().total_excess_occupancy == 1:
            return seed
    raise AssertionError("expected to find a deterministic colliding initialization")


def test_injected_scorer_changes_only_the_selected_candidate() -> None:
    seed = find_colliding_seed()
    default_session = colliding_session(seed)
    reverse_session = colliding_session(seed)
    logical = default_session.eligible_variables()[0]
    default_selector = FixedSelector(logical)
    reverse_selector = FixedSelector(logical)
    reverse_scorer = ReverseScorer()

    default_outcome = SearchOrchestrator(selector=default_selector).transition(
        default_session, random.Random(seed)
    )
    reverse_outcome = SearchOrchestrator(
        selector=reverse_selector, scorer=reverse_scorer
    ).transition(reverse_session, random.Random(seed))

    assert default_outcome.candidate_chains == reverse_outcome.candidate_chains
    assert default_outcome.candidate_index == 0
    assert reverse_outcome.candidate_index == len(reverse_outcome.candidate_chains) - 1
    assert default_session.snapshot().chains != reverse_session.snapshot().chains
    assert default_selector.calls == reverse_selector.calls == reverse_scorer.calls == 1


def test_invalid_scores_are_rejected_and_outstanding_batch_is_discarded() -> None:
    seed = find_colliding_seed()
    session = colliding_session(seed)
    logical = session.eligible_variables()[0]
    generation = session.snapshot().generation

    class BadScorer:
        def score(self, snapshot, candidates):
            return [0.0]

    with pytest.raises(ValueError, match="one finite score"):
        SearchOrchestrator(selector=FixedSelector(logical), scorer=BadScorer()).transition(
            session, random.Random(seed)
        )

    assert session.snapshot().generation == generation
    batch = session.propose(logical)
    session.discard(batch)


def test_callback_interrupt_still_discards_the_outstanding_batch() -> None:
    seed = find_colliding_seed()
    session = colliding_session(seed)
    logical = session.eligible_variables()[0]

    class InterruptingScorer:
        def score(self, snapshot, candidates):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        SearchOrchestrator(selector=FixedSelector(logical), scorer=InterruptingScorer()).transition(
            session, random.Random(seed)
        )

    batch = session.propose(logical)
    session.discard(batch)


def test_valid_state_can_only_be_rerouted_through_explicit_refinement_mode() -> None:
    session = SearchSession(
        GraphLike(["logical"], []),
        GraphLike([0, 1, 2], [(0, 1), (1, 2)]),
        random_seed=0,
        max_candidates=3,
    )
    orchestrator = SearchOrchestrator(selector=FixedSelector(0))

    assert session.snapshot().valid
    with pytest.raises(ValueError, match="eligible"):
        orchestrator.transition(session, random.Random(0))

    outcome = orchestrator.transition(
        session,
        random.Random(0),
        allow_valid_refinement=True,
    )

    assert outcome.before.valid
    assert outcome.after.valid
    assert outcome.accepted


@pytest.mark.parametrize("selection", [(), (0, 0), (99,), (True,)])
def test_invalid_local_repair_neighborhood_is_rejected_before_mutation(selection) -> None:
    class BadRepairSelector:
        def select(self, snapshot, rng):
            return selection

    session = colliding_session(find_colliding_seed())
    before = session.snapshot()
    orchestrator = SearchOrchestrator(repair_selector=BadRepairSelector(), max_local_repairs=1)

    with pytest.raises(ValueError, match="unique in-range"):
        orchestrator.repair(session, before, random.Random(0))

    assert session.snapshot().chains == before.chains
    assert session.snapshot().generation == before.generation


def test_local_repair_configuration_must_be_coupled_and_bounded() -> None:
    class RepairSelector:
        def select(self, snapshot, rng):
            return (0,)

    with pytest.raises(ValueError, match="repair_selector"):
        SearchOrchestrator(max_local_repairs=1)
    with pytest.raises(ValueError, match="max_local_repairs"):
        SearchOrchestrator(repair_selector=RepairSelector())
    with pytest.raises(ValueError, match="max_local_repairs"):
        SearchOrchestrator(repair_selector=RepairSelector(), max_local_repairs=True)
