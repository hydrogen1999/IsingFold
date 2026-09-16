"""Constructor selection, exact-program labels and wall-clock accounting contracts."""
from types import SimpleNamespace

import pytest

import constructor_objective as objective
import constructor_protocol as protocol


def terminal(root=0):
    return SimpleNamespace(returned_valid=True, embedding={0: frozenset({root})},
                           selected_program=SimpleNamespace(strength_index=2), selected_index=2)


@pytest.fixture
def clock(monkeypatch):
    current = [0.]
    monkeypatch.setattr(protocol.time, "monotonic", lambda: current[0])
    return current


def test_selection_and_assessment_are_disjoint_and_assessment_cannot_reselect(clock):
    task = SimpleNamespace(name="toy")
    proposed, measured, budgets = [], [], []
    candidates = [terminal(0), terminal(1)]
    def propose(seed, left):
        proposed.append(seed)
        budgets.append(left)
        clock[0] += .2  # Feature extraction, action selection and compilation.
        return candidates[len(proposed) - 1]
    def measure(task, item, seed, reads):
        measured.append((item, seed, reads))
        clock[0] += .1
        return (1. if item is candidates[0] else .1) if reads == 3 else 8.
    result = protocol.evaluate_search(task, propose, deadline=1., select_cap=2,
                                      selection_reads=3, assessment_reads=7, measure=measure, seed=6)
    assert measured[-1][0] is candidates[1]
    assert result["valid"] and result["residual"] == 8.
    assert result["selection_reads"] == 6 and result["assessment_reads"] == 7
    assert result["deployment_seconds"] == pytest.approx(.6)
    assert result["total_seconds"] == pytest.approx(.7)
    assert budgets == pytest.approx([1., .7])
    all_seeds = proposed + [s for _, s, _ in measured]
    assert len(set(all_seeds)) == len(all_seeds)


def test_late_feature_proposal_never_counts_as_valid_or_spends_reads(clock):
    def propose(seed, left):
        assert left == 1.
        clock[0] += 1.1
        return terminal()
    def forbidden(*args):
        pytest.fail("late candidates must not be measured")
    result = protocol.evaluate_search(SimpleNamespace(name="toy"), propose, deadline=1., measure=forbidden)
    assert not result["valid"] and result["constructed_attempts"] == 0
    assert result["selection_reads"] == result["assessment_reads"] == 0
    assert result["deadline_overrun_seconds"] == pytest.approx(.1)


def test_late_measurement_is_charged_but_cannot_replace_on_time_candidate(clock):
    first, late = terminal(0), terminal(1)
    proposed, measured = [], []
    def propose(seed, left):
        clock[0] += .1
        proposed.append(seed)
        return first if len(proposed) == 1 else late
    def measure(task, item, seed, reads):
        measured.append((item, reads))
        clock[0] += .2 if len(measured) != 2 else 1.
        return .5 if item is first else .01
    result = protocol.evaluate_search(SimpleNamespace(name="toy"), propose, deadline=1.,
                                      selection_reads=3, assessment_reads=7, measure=measure)
    assert result["selection_reads"] == 6 and result["assessment_reads"] == 7
    assert measured[-1] == (first, 7)
    assert result["valid"] and result["residual"] == .5
    assert result["deadline_overrun_seconds"] > 0


def test_all_failed_proposals_remain_in_coverage_and_use_no_reads(clock):
    def propose(seed, left):
        clock[0] += .4
        return None
    result = protocol.evaluate_search(SimpleNamespace(name="toy"), propose, deadline=1.)
    assert not result["valid"] and result["attempts"] == 3
    assert result["unique_candidates"] == result["selection_reads"] == 0
    key = protocol.checkpoint_key([result, {"valid": True, "residual": .2}], "quality")
    assert key == (.5, .5, -.2)


def test_failed_measurements_still_consume_declared_read_budget(clock):
    counter = [0]
    def propose(seed, left):
        counter[0] += 1
        clock[0] += .1
        return terminal(counter[0])
    def missing(*args):
        clock[0] += .1
        return None
    result = protocol.evaluate_search(SimpleNamespace(name="toy"), propose, deadline=2.,
                                      select_cap=2, selection_reads=5, measure=missing)
    assert not result["valid"] and result["selection_reads"] == 10
    assert result["measurement_failures"] == 2 and result["assessment_reads"] == 0
    assert result["constructed_attempts"] == 2


def test_feasibility_does_not_call_the_quality_backend(clock):
    def propose(seed, left):
        clock[0] += .1
        return terminal()
    def forbidden(*args):
        pytest.fail("feasibility curriculum must not call quality backend")
    result = protocol.evaluate_search(SimpleNamespace(name="toy"), propose, deadline=1.,
                                      objective="feasibility", measure=forbidden)
    assert result["valid"] and result["residual"] is None
    assert result["selection_reads"] == result["assessment_reads"] == 0


def test_duplicate_embedding_is_not_measured_again(clock):
    calls = [0]
    same = terminal()
    def propose(seed, left):
        clock[0] += .2
        return same
    def measure(*args):
        calls[0] += 1
        return .3
    result = protocol.evaluate_search(SimpleNamespace(name="toy"), propose, deadline=.7,
                                      selection_reads=4, assessment_reads=9, measure=measure)
    assert result["unique_candidates"] == 1 and result["selection_reads"] == 4
    assert calls[0] == 2  # One selection block and one fresh reporting block.


def test_distinct_strength_programs_of_same_embedding_are_not_deduplicated(clock):
    first, second = terminal(), terminal()
    second.selected_index = second.selected_program.strength_index = 3
    proposed = []
    def propose(seed, left):
        proposed.append(seed)
        clock[0] += .1
        return first if len(proposed) == 1 else second
    result = protocol.evaluate_search(SimpleNamespace(name="toy"), propose, deadline=1.,
                                      select_cap=2, selection_reads=4, assessment_reads=9,
                                      measure=lambda task, item, seed, reads: .5)
    assert result["unique_candidates"] == 2 and result["selection_reads"] == 8


@pytest.mark.parametrize("name", ["selection_reads", "assessment_reads", "select_cap"])
@pytest.mark.parametrize("value", [True, 0, 1.5, float("inf")])
def test_read_and_selection_caps_must_be_positive_integers(name, value):
    with pytest.raises(ValueError):
        protocol.evaluate_search(SimpleNamespace(name="toy"), lambda *_: None,
                                  deadline=1., **{name: value})


def test_split_keeps_siblings_out_of_validation():
    tasks = [SimpleNamespace(name=f"task-{i}-{j}", lineage=f"family-{i}")
             for i in range(4) for j in range(3)]
    train, validation = protocol.split_by_lineage(tasks, .25, 1)
    assert {t.lineage for t in train}.isdisjoint(t.lineage for t in validation)
    assert len(train) == 9 and len(validation) == 3


def test_quality_measurement_uses_exact_selected_program_and_registered_schedule(monkeypatch):
    from _context import REGISTERED_BETA_RANGE
    import isingfold.rl.evaluator as backend
    import isingfold.rl.env as env
    import seeded_minorminer
    t = SimpleNamespace(problem=object(), ground_energy=-4.)
    item = terminal()
    captured = {}
    def sample(program, embedding, problem, ground, **kwargs):
        assert program is item.selected_program and embedding is item.embedding
        assert problem is t.problem and ground == -4.
        captured.update(kwargs)
        return SimpleNamespace(reads=11, strength_index=2, mean_residual=.4)
    def forbidden(*args, **kwargs):
        pytest.fail("quality measurement must not construct or complete an embedding")
    monkeypatch.setattr(backend, "sample_program", sample)
    monkeypatch.setattr(env, "EmbeddingEnv", forbidden)
    monkeypatch.setattr(seeded_minorminer, "attempt", forbidden)
    assert objective.measure_terminal(t, item, 9, reads=11) == .4
    assert captured == {"num_reads": 11, "seed": 9, "num_sweeps": 200,
                        "beta_range": REGISTERED_BETA_RANGE}


@pytest.mark.parametrize("reads,strength", [(10, 2), (11, 1)])
def test_quality_measurement_checks_read_and_strength_receipts(monkeypatch, reads, strength):
    import isingfold.rl.evaluator as backend
    monkeypatch.setattr(backend, "sample_program", lambda *a, **kw:
                        SimpleNamespace(reads=reads, strength_index=strength, mean_residual=.4))
    with pytest.raises(ValueError, match="read/strength"):
        objective.measure_terminal(SimpleNamespace(problem=object(), ground_energy=-4.), terminal(), 0, 11)


def test_quality_utility_is_affine_and_has_no_resource_penalty():
    task = SimpleNamespace(problem=SimpleNamespace(h={0: 1.}, j={(0, 1): -3.}), ground_energy=-4.)
    assert [objective.quality_utility(task, x) for x in (0., .5, 1., 1.5, 2.)] == [1., .875, .75, .625, .5]
    assert objective.quality_utility(task, None) == 0.
    with pytest.raises(ValueError, match="coefficient bound"):
        objective.quality_utility(task, 2.1)
    task.problem = SimpleNamespace(h={}, j={})
    task.ground_energy = 0.
    assert objective.quality_utility(task, 0.) == 1.
    with pytest.raises(ValueError, match="constant-zero"):
        objective.quality_utility(task, .1)
