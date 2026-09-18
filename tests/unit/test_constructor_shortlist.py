"""Bounded shortlist diversity, validity, and wall-clock accounting contracts."""
from types import SimpleNamespace

import networkx as nx
import pytest

import constructor_shortlist as shortlist


def task(logical=None, host=None):
    return SimpleNamespace(name="shortlist-toy",
                           logical=nx.empty_graph(1) if logical is None else logical,
                           host=nx.path_graph(8) if host is None else host)


def terminal(chains, index=2):
    return SimpleNamespace(returned_valid=True,
                           embedding={v: frozenset(c) for v, c in chains.items()},
                           selected_index=index,
                           selected_program=SimpleNamespace(strength_index=index))


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def stream(candidates, timer, seconds=1.0):
    iterator = iter(candidates)
    calls = []

    def propose(seed, left):
        calls.append((seed, left))
        timer.now += seconds
        return next(iterator)

    return propose, calls


def test_collector_keeps_searching_after_full_and_charges_duplicates_and_failures():
    clock = Clock()
    a, b = terminal({0: {0}}), terminal({0: {1}})
    propose, calls = stream([a, a, None, b, b], clock)
    kept, receipt = shortlist.collect_shortlist(task(), propose, deadline=5, capacity=2,
                                                clock=clock, seed=3)
    assert kept == (a, b)
    assert receipt["attempts"] == 5
    assert receipt["constructed_attempts"] == 4
    assert receipt["duplicate_attempts"] == 2
    assert receipt["invalid_attempts"] == 1
    assert receipt["selection_reads"] == receipt["assessment_reads"] == 0
    assert receipt["proposal_seconds"] == 5
    assert [left for _, left in calls] == [5, 4, 3, 2, 1]
    assert len({seed for seed, _ in calls}) == 5


def test_diversity_replacement_can_prefer_more_qubits_and_ties_keep_earlier():
    clock = Clock()
    a = terminal({0: {0}})
    b = terminal({0: {0, 1}})
    larger = terminal({0: {3, 4, 5}})
    tie = terminal({0: {6, 7}})
    propose, _ = stream([a, b, larger, tie], clock)
    kept, receipt = shortlist.collect_shortlist(task(), propose, deadline=4, capacity=2,
                                                clock=clock)
    assert kept == (a, larger)
    assert receipt["shortlist_replacements"] == 1
    assert receipt["shortlist_rejections"] == 1
    assert [r["proposal_attempt"] for r in receipt["retained_metadata"]] == [1, 3]


def test_first_mode_and_capacity_one_keep_initial_program_but_count_every_attempt():
    candidates = [terminal({0: {i}}) for i in range(4)]
    for mode, capacity in [("first", 2), ("diverse", 1)]:
        clock = Clock()
        propose, _ = stream(candidates, clock)
        kept, receipt = shortlist.collect_shortlist(task(), propose, deadline=4,
                                                    capacity=capacity, clock=clock, mode=mode)
        assert kept == tuple(candidates[:capacity])
        assert receipt["attempts"] == 4
        assert receipt["shortlist_rejections"] == 4 - capacity


def test_same_assignment_distinct_strengths_are_distinct_programs():
    clock = Clock()
    a, b = terminal({0: {0}}, 1), terminal({0: {0}}, 2)
    propose, _ = stream([a, b], clock)
    kept, receipt = shortlist.collect_shortlist(task(), propose, deadline=2, capacity=2,
                                                clock=clock)
    assert kept == (a, b)
    assert receipt["duplicate_attempts"] == 0
    assert shortlist.chain_assignment_distance(a.embedding, b.embedding) == 0


def test_exhausted_finite_proposer_is_not_counted_as_an_additional_attempt():
    clock = Clock()
    a = terminal({0: {0}})
    iterator = iter([a])

    def propose(seed, left):
        return next(iterator)

    kept, receipt = shortlist.collect_shortlist(task(), propose, deadline=1, clock=clock)
    assert kept == (a,)
    assert receipt["proposer_exhausted"] and receipt["attempts"] == 1


def test_late_answer_consumes_attempt_and_time_but_cannot_enter_shortlist():
    clock = Clock()
    a = terminal({0: {0}})
    propose, _ = stream([a], clock, seconds=2)
    kept, receipt = shortlist.collect_shortlist(task(), propose, deadline=1, clock=clock)
    assert kept == ()
    assert receipt["attempts"] == receipt["late_attempts"] == 1
    assert receipt["constructed_attempts"] == 0
    assert receipt["deadline_overrun_seconds"] == 1


def test_late_shortlist_computation_preserves_earlier_members(monkeypatch):
    clock = Clock()
    a, b, c = terminal({0: {0}}), terminal({0: {0, 1}}), terminal({0: {3}})
    propose, _ = stream([a, b, c], clock)
    original = shortlist._minimum_distance

    def slow(items):
        clock.now += 1
        return original(items)

    monkeypatch.setattr(shortlist, "_minimum_distance", slow)
    kept, receipt = shortlist.collect_shortlist(task(), propose, deadline=4, capacity=2,
                                                clock=clock)
    assert kept == (a, b)
    assert receipt["attempts"] == 3 and receipt["late_attempts"] == 1


@pytest.mark.parametrize("chains,reason", [
    ({0: {0}}, "incomplete_assignment"),
    ({0: set(), 1: {1}}, "empty_chain"),
    ({0: {99}, 1: {1}}, "unknown_qubit"),
    ({0: {0, 1}, 1: {1}}, "overlapping_chains"),
    ({0: {0, 2}, 1: {1}}, "disconnected_chain"),
    ({0: {0}, 1: {3}}, "unrealized_logical_edge"),
])
def test_proposer_valid_flag_does_not_bypass_graph_validation(chains, reason):
    clock = Clock()
    propose, _ = stream([terminal(chains)], clock)
    kept, receipt = shortlist.collect_shortlist(task(logical=nx.path_graph(2)), propose,
                                                deadline=1, clock=clock)
    assert kept == ()
    assert receipt["invalid_reasons"] == {reason: 1}


def test_missing_or_mismatched_program_is_rejected():
    for program in [None, SimpleNamespace(strength_index=99)]:
        clock = Clock()
        candidate = terminal({0: {0}})
        candidate.selected_program = program
        propose, _ = stream([candidate], clock)
        kept, receipt = shortlist.collect_shortlist(task(), propose, deadline=1, clock=clock)
        assert not kept
        assert receipt["invalid_reasons"] == {"missing_or_mismatched_program": 1}


def test_distance_is_invariant_to_iteration_and_consistent_node_relabeling():
    a, b = {0: {0, 1}, 1: {2}}, {1: {2, 3}, 0: {0}}
    transform = lambda x: {str(v): {q + 10 for q in c} for v, c in reversed(x.items())}
    assert shortlist.chain_assignment_distance(a, b) == .5
    assert shortlist.chain_assignment_distance(transform(a), transform(b)) == .5


def test_collector_does_not_consult_quality_reward_or_ground_truth():
    class BlindTerminal:
        returned_valid = True
        embedding = {0: frozenset({0})}
        selected_index = 2
        selected_program = SimpleNamespace(strength_index=2)

        @property
        def training_reward(self):
            pytest.fail("shortlist cannot inspect measured reward")

    clock = Clock()
    candidate = BlindTerminal()
    propose, _ = stream([candidate], clock)
    kept, receipt = shortlist.collect_shortlist(task(), propose, deadline=1, clock=clock)
    assert kept == (candidate,) and receipt["quality_outcomes_used"] is False


@pytest.mark.parametrize("kwargs", [
    {"deadline": 0}, {"deadline": float("nan")}, {"deadline": True},
    {"capacity": 0}, {"capacity": True}, {"capacity": 1.5}, {"mode": "resource"},
])
def test_bad_budget_and_selection_settings_fail_before_proposal(kwargs):
    with pytest.raises(ValueError):
        shortlist.collect_shortlist(task(), lambda *_: pytest.fail("no proposal"),
                                    **({"deadline": 1} | kwargs))
