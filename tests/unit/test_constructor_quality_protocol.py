"""Paired quality measurement, failure inclusion and deployment receipts."""
from types import SimpleNamespace

import pytest

import constructor_protocol as protocol
from constructor_objective import quality_utility


@pytest.fixture
def clock(monkeypatch):
    current = [0.0]
    monkeypatch.setattr(protocol.time, "monotonic", lambda: current[0])
    return current


def task():
    return SimpleNamespace(name="quality-task", problem=SimpleNamespace(h={0: 1.}, j={(0, 1): -3.}),
                           ground_energy=-4.)


def terminal(size=1):
    return SimpleNamespace(returned_valid=True, embedding={0: frozenset(range(size))}, selected_index=2)


def paired(item, reads, residual=.25, p_solve=.5):
    return {"residual": residual, "p_solve": p_solve, "reads": reads,
            "strength_index": item.selected_index}


def test_equal_read_budgets_do_not_confuse_selection_with_assessment(clock):
    candidates = [terminal(1), terminal(4)]
    calls = []

    def propose(seed, left):
        clock[0] += .1
        return candidates[len(calls)]

    def measure(t, item, seed, reads):
        calls.append((item, seed, reads))
        # Candidate 0 has more hits but worse residual. Selection must use only
        # the prespecified residual, and assessment cannot select candidate 0.
        if len(calls) == 1:
            return paired(item, reads, .5, .9)
        if len(calls) == 2:
            return paired(item, reads, .1, .1)
        return paired(item, reads, .8, .25)

    result = protocol.evaluate_search(task(), propose, deadline=2., select_cap=2,
                                      selection_reads=4, assessment_reads=4, measure=measure,
                                      utility=quality_utility)
    assert calls[-1][0] is candidates[1]
    assert len(calls) == 3  # Not a second sample call for p_solve.
    assert result["residual"] == .8 and result["p_solve"] == .25
    assert result["quality_utility"] == pytest.approx(.8)
    assert result["selection_residual"] == .1
    assert result["assessment_ok"] and result["valid"]
    assert result["chosen_candidate"]["candidate_id"] == 1
    assert result["chosen_candidate"]["proposal_attempt"] == 2
    assert result["chosen_qubits"] == result["chosen_longest_chain"] == 4
    assert result["candidate_qubits_mean"] == 2.5  # Each candidate counted once.
    assert result["selection_calls"] == 2 and result["assessment_calls"] == 1
    assert result["selection_reads_verified"] == 8
    assert result["assessment_reads_verified"] == 4 and result["total_reads"] == 12
    assert result["assessment_seed"] not in [r["seed"] for r in result["selection_receipts"]]
    assert result["budget"] == {"deadline_seconds": 2., "select_cap": 2,
                                "selection_reads_per_call": 4, "assessment_reads_per_call": 4}


def test_failed_assessment_keeps_construction_coverage_but_zeroes_utility(clock):
    calls = []

    def propose(seed, left):
        clock[0] += .1
        return terminal()

    def measure(t, item, seed, reads):
        calls.append(seed)
        return paired(item, reads) if len(calls) == 1 else None

    result = protocol.evaluate_search(task(), propose, deadline=2., select_cap=1,
                                      selection_reads=4, assessment_reads=7, measure=measure,
                                      utility=quality_utility)
    assert result["valid"] and result["construction_valid"]
    assert not result["assessment_ok"] and result["quality_utility"] == 0.
    assert result["residual"] is result["p_solve"] is None
    assert result["assessment_failures"] == 1 and result["selection_failures"] == 0
    assert result["assessment_reads"] == 7 and result["assessment_reads_verified"] == 0
    assert result["total_reads"] == 11
    assert result["chosen_qubits"] == 1  # Does not depend on a successful sample.


def test_no_output_is_not_omitted_or_assigned_a_good_quality_score(clock):
    def propose(seed, left):
        clock[0] += .6
        return None

    result = protocol.evaluate_search(task(), propose, deadline=1., utility=quality_utility)
    assert not result["valid"] and not result["construction_valid"]
    assert result["quality_utility"] == 0. and not result["assessment_ok"]
    assert result["chosen_candidate"] is result["chosen_qubits"] is None
    assert result["candidate_qubits_mean"] is None
    assert result["selection_calls"] == result["assessment_calls"] == result["total_reads"] == 0


def test_failed_selection_retains_attempt_and_read_accounting(clock):
    count = [0]

    def propose(seed, left):
        clock[0] += .1
        count[0] += 1
        return terminal(count[0])

    result = protocol.evaluate_search(task(), propose, deadline=1., select_cap=2,
                                      selection_reads=3, measure=lambda *args: None,
                                      utility=quality_utility)
    assert not result["valid"] and result["construction_valid"]
    assert result["quality_utility"] == 0. and result["total_reads"] == 6
    assert result["selection_failures"] == 2 and result["assessment_calls"] == 0
    assert len(result["selection_receipts"]) == result["constructed_attempts"] == 2
    assert result["chosen_qubits"] is None


def test_late_pilot_cannot_set_chosen_metadata_or_leak_to_assessment(clock):
    count = [0]

    def propose(seed, left):
        clock[0] += .1
        count[0] += 1
        return terminal(count[0])

    def measure(t, item, seed, reads):
        clock[0] += 2.
        return paired(item, reads, 0., 1.)

    result = protocol.evaluate_search(task(), propose, deadline=1., select_cap=2,
                                      selection_reads=3, assessment_reads=3, measure=measure,
                                      utility=quality_utility)
    assert not result["valid"] and result["quality_utility"] == 0.
    assert result["chosen_candidate"] is None and result["assessment_seed"] is None
    assert result["total_reads"] == result["selection_reads_verified"] == 3
    assert not result["selection_receipts"][0]["on_time"]


def test_seed_collision_resolution_keeps_all_phases_disjoint(clock, monkeypatch):
    monkeypatch.setattr(protocol, "experiment_seed", lambda *args: 0)
    count = [0]
    seeds = []

    def propose(seed, left):
        clock[0] += .1
        count[0] += 1
        seeds.append(seed)
        return terminal(count[0])

    def measure(t, item, seed, reads):
        seeds.append(seed)
        return paired(item, reads)

    result = protocol.evaluate_search(task(), propose, deadline=1., select_cap=2, measure=measure)
    assert len(seeds) == len(set(seeds)) == 5
    assert result["assessment_seed"] == seeds[-1]


@pytest.mark.parametrize("replacement", [
    {"reads": 3}, {"strength_index": 99}, {"p_solve": -0.1}, {"p_solve": 1.1},
    {"p_solve": float("nan")}, {"residual": -1.}, {"residual": float("inf")},
    {"p_solve": None}, {"residual": None},
])
def test_paired_metrics_require_valid_receipts_and_finite_metric_ranges(clock, replacement):
    def propose(seed, left):
        clock[0] += .1
        return terminal()

    def measure(t, item, seed, reads):
        return {**paired(item, reads), **replacement}

    with pytest.raises(ValueError):
        protocol.evaluate_search(task(), propose, deadline=1., selection_reads=4,
                                  measure=measure, utility=quality_utility)


def test_scalar_callbacks_remain_supported_without_fabricating_p_solve(clock):
    def propose(seed, left):
        clock[0] += .1
        return terminal()

    result = protocol.evaluate_search(task(), propose, deadline=1., select_cap=1,
                                      measure=lambda *args: .5, utility=quality_utility)
    assert result["residual"] == .5 and result["p_solve"] is None
    assert result["quality_utility"] == .875 and result["assessment_ok"]
    assert result["selection_reads_verified"] == result["assessment_reads_verified"] == 0
    assert result["total_reads"] == 512


def anytime_task():
    import networkx as nx
    return SimpleNamespace(name="quality-task", logical=nx.path_graph(2), host=nx.path_graph(6),
                           problem=SimpleNamespace(h={}, j={(0, 1): -1.}), ground_energy=-1.)


def adjacent_terminal(root):
    return SimpleNamespace(returned_valid=True,
                           embedding={0: frozenset({root}), 1: frozenset({root + 1})},
                           selected_index=2, selected_program=SimpleNamespace(strength_index=2))


def test_anytime_spends_proposal_allocation_beyond_read_cap_and_keeps_receipts(clock):
    proposed, measured = [], []

    def propose(seed, left):
        proposed.append((seed, left))
        clock[0] += .11
        return adjacent_terminal(len(proposed) - 1)

    def measure(t, item, seed, reads):
        measured.append((seed, reads))
        clock[0] += .1
        return paired(item, reads, .5, .25)

    result = protocol.evaluate_anytime_search(anytime_task(), propose, deadline=1., select_cap=2,
                                              selection_reads=3, assessment_reads=7, measure=measure,
                                              utility=quality_utility, shortlist_mode="first")
    assert len(proposed) == result["attempts"] == 5  # Not capped at two pilot blocks.
    assert result["constructed_attempts"] == 4
    assert result["proposal_collection"]["late_attempts"] == 1
    assert result["retained_candidates"] == result["shortlist_retrieval_attempts"] == 2
    assert result["selection_reads"] == 6 and result["assessment_reads"] == 7
    assert result["total_reads"] == 13 and len(measured) == 3
    assert result["deployment_seconds"] == pytest.approx(.75)
    assert result["total_seconds"] == pytest.approx(.85)
    assert result["budget"]["deadline_seconds"] == 1.
    assert result["budget"]["selection_seconds_available"] == pytest.approx(.45)
    assert result["proposal_collection"]["quality_outcomes_used"] is False
    assert result["chosen_candidate"]["proposal_seed"] == proposed[0][0]
    assert result["chosen_candidate"]["proposal_attempt"] == 1
    assert result["quality_utility"] == .875 and result["p_solve"] == .25
    assert len({seed for seed, _ in measured}) == 3


def test_anytime_empty_shortlist_finishes_without_busy_wait_or_sampling(clock):
    def propose(seed, left):
        clock[0] += .2
        return None

    def forbidden(*args):
        pytest.fail("empty pools cannot spend assessment or pilot reads")

    result = protocol.evaluate_anytime_search(anytime_task(), propose, deadline=1., measure=forbidden,
                                              utility=quality_utility)
    assert result["attempts"] == 3 and result["retained_candidates"] == 0
    assert result["deployment_seconds"] == pytest.approx(.6)
    assert result["total_reads"] == 0 and result["quality_utility"] == 0.
    assert not result["valid"] and result["shortlist_retrieval_attempts"] == 0


def test_anytime_global_deadline_overrun_cannot_start_pilot(clock):
    def propose(seed, left):
        clock[0] += 1.2
        return adjacent_terminal(0)

    def forbidden(*args):
        pytest.fail("late construction cannot start selection")

    result = protocol.evaluate_anytime_search(anytime_task(), propose, deadline=1., measure=forbidden)
    assert not result["valid"] and result["total_reads"] == 0
    assert result["deadline_overrun_seconds"] == pytest.approx(.2)
    assert result["budget"]["selection_seconds_available"] == 0.


def test_finite_pool_exhaustion_is_not_a_proposal_attempt(clock):
    def empty(seed, left):
        raise StopIteration

    result = protocol.evaluate_search(task(), empty, deadline=1., utility=quality_utility)
    assert result["attempts"] == 0 and result["proposal_seeds"] == []
    assert not result["valid"] and result["total_seconds"] == 0.


@pytest.mark.parametrize("fraction", [0., 1., -1., float("inf"), float("nan")])
def test_anytime_rejects_invalid_allocations_before_proposals(fraction):
    def forbidden(*args):
        pytest.fail("invalid budget must fail before construction")

    with pytest.raises(ValueError, match="proposal_fraction"):
        protocol.evaluate_anytime_search(anytime_task(), forbidden, deadline=1.,
                                         proposal_fraction=fraction)


@pytest.mark.parametrize("anytime", [False, True])
def test_public_selection_never_uses_ground_reference_or_assessment_outcomes(clock, anytime):
    source = anytime_task()

    class PublicTask:
        name = source.name
        problem = source.problem
        logical = source.logical
        host = source.host

        @property
        def ground_energy(self):
            pytest.fail("selection must not access a hidden ground-energy reference")

    t = PublicTask()
    candidates = [adjacent_terminal(0), adjacent_terminal(2)]
    iterator = iter(candidates)
    selection_calls, assessment_calls = [], []

    def propose(seed, left):
        result = next(iterator)
        clock[0] += .1
        return result

    def select_measure(task, item, seed, reads):
        assert task.problem is source.problem
        selection_calls.append((item, seed))
        return .8 if item is candidates[0] else .2

    def assess(task, item, seed, reads):
        # A trusted evaluator may have a private reference independently, but
        # neither the candidate policy nor the selection callback receives it.
        assert len(selection_calls) == 2
        assert item is candidates[1]
        assessment_calls.append((item, seed))
        return paired(item, reads, residual=1.2, p_solve=.125)

    evaluate = protocol.evaluate_anytime_search if anytime else protocol.evaluate_search
    result = evaluate(t, propose, deadline=1., select_cap=2, selection_reads=3,
                      assessment_reads=7, select_measure=select_measure, measure=assess)
    assert len(selection_calls) == 2 and len(assessment_calls) == 1
    assert result["selection_score"] == .2
    assert result["selection_score_semantics"] == "callback_lower_is_better"
    assert result["selection_residual"] is None  # Do not mislabel a public energy score.
    assert result["residual"] == 1.2 and result["p_solve"] == .125
    assert result["selection_reads"] == 6 and result["assessment_reads"] == 7
    assert all(r["residual"] is None for r in result["selection_receipts"])
    assert [r["score"] for r in result["selection_receipts"]] == [.8, .2]
    assert assessment_calls[0][1] not in {seed for _, seed in selection_calls}
