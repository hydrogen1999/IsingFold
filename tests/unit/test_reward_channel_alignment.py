"""Calibration cannot reuse final-test instances or conflate two quality objectives."""
import json
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

import reward_channel as rc
from isingfold.embedding import LogicalProblem


def test_residual_self_ranking_win_does_not_imply_solve_probability_win():
    # Candidate A lowers expected energy but finds exact solutions less often.
    residual = np.array([[.1] * 8, [.3] * 8])
    solve = np.array([[.2] * 8, [.8] * 8])
    same, _ = rc.ranking_gain(residual, [0], list(range(1, 8)), False)
    cross = rc.ranking_assessment(residual, solve, [0], list(range(1, 8)), False, True)
    assert same == pytest.approx(.1)
    assert cross["gain"] == pytest.approx(-.3)
    reverse = rc.ranking_assessment(solve, residual, [0], list(range(1, 8)), True, False)
    assert reverse["gain"] == pytest.approx(-.1)


def test_assessment_values_cannot_change_candidate_selected_on_ranking_block():
    rank = np.array([[.8, .0, .0], [.2, 1., 1.]])
    assessment = np.array([[0., .9, .9], [1., .1, .1]])
    before = rc.ranking_assessment(rank, assessment, [0], [1, 2], True, True)
    assessment[:, 1:] = 1 - assessment[:, 1:]
    after = rc.ranking_assessment(rank, assessment, [0], [1, 2], True, True)
    assert before["selected_index"] == after["selected_index"] == 0
    assert before["gain"] == pytest.approx(-after["gain"])


def test_ties_are_seeded_uniform_draws_not_pool_first_candidate():
    values = np.ones((3, 8))
    picks = [rc.ranking_assessment(values, values, [0], list(range(1, 8)), True, True, seed)
             for seed in range(30)]
    assert {p["selected_index"] for p in picks} == {0, 1, 2}
    assert all(p["tie_count"] == 3 for p in picks)
    assert picks[0] == rc.ranking_assessment(values, values, [0], list(range(1, 8)), True, True, 0)


@pytest.mark.parametrize("rank_cols,assess_cols", [([0], [0, 1]), ([], [1]), ([0], []),
                                                    ([0, 0], [1]), ([0], [8]), ([False], [1])])
def test_rank_assessment_blocks_must_be_nonempty_valid_and_disjoint(rank_cols, assess_cols):
    with pytest.raises(ValueError):
        rc.ranking_assessment(np.ones((2, 8)), np.ones((2, 8)), rank_cols, assess_cols, True, True)


def test_bad_shapes_and_repeat_counts_fail():
    with pytest.raises(ValueError, match="identical"):
        rc.ranking_assessment(np.ones((2, 8)), np.ones((3, 8)), [0], [1], True, True)
    with pytest.raises(ValueError, match="finite"):
        rc.stats([[0., float("nan")], [1., 1.]], 2)
    with pytest.raises(ValueError, match="repeats"):
        rc.stats(np.ones((2, 8)), 4)


def test_zero_noise_ratio_is_explicitly_undefined_and_json_safe():
    within, between, ratio = rc.stats(np.array([[.1] * 8, [.4] * 8]), 8)
    assert within == pytest.approx(0.)
    # Binary-exact constants guarantee exact zero sampling variance.
    within, between, ratio = rc.stats(np.array([[.25] * 8, [.5] * 8]), 8)
    assert within == 0 and between > 0 and ratio is None
    assert json.loads(json.dumps({"ratio": ratio}, allow_nan=False)) == {"ratio": None}


@pytest.mark.parametrize("split", ["train", "validation"])
@pytest.mark.parametrize("allow_unregistered", [False, True])
def test_registered_locked_test_never_enters_calibration(tmp_path, monkeypatch, split, allow_unregistered):
    import isingfold.rl.data.generate as data
    logical = nx.path_graph(2)
    problem = LogicalProblem.from_dicts({0: 0., 1: 0.}, {(0, 1): -1.})
    tasks = [SimpleNamespace(name=role, lineage=role, logical=logical, host=nx.path_graph(4),
                              problem=problem, ground_energy=-1., witness=None)
             for role in ("train", "validation", "test")]
    monkeypatch.setattr(data, "load_instances", lambda path: tasks)
    (tmp_path / "splits.json").write_text(json.dumps({r: [r] for r in ("train", "validation", "test")}))
    selected, registered = rc.calibration_tasks(tmp_path, [], 1, 0, split, allow_unregistered)
    assert registered and [t.name for t in selected] == [split]


def test_no_manifest_requires_explicit_exploratory_opt_in(tmp_path, monkeypatch):
    captured = []
    def build(*args, **kwargs):
        captured.append(kwargs)
        return ["exploratory"], []
    monkeypatch.setattr(rc.cc, "build_corpus_sets", build)
    with pytest.raises(ValueError, match="registered split"):
        rc.calibration_tasks(tmp_path, [], 1, 0)
    assert not captured
    selected, registered = rc.calibration_tasks(tmp_path, [], 1, 0, allow_unregistered=True)
    assert selected == ["exploratory"] and not registered
    assert captured == [{"use_manifest_split": False, "heldout_role": "validation"}]
    with pytest.raises(ValueError, match="never test"):
        rc.calibration_tasks(tmp_path, [], 1, 0, split="test", allow_unregistered=True)


def test_calibration_pool_deduplicates_unchanged_growth(monkeypatch):
    task = SimpleNamespace(logical=nx.path_graph(2), host=nx.path_graph(2))
    monkeypatch.setattr(rc, "minorminer_initializer", lambda tries: lambda *args: {0: [0], 1: [1]})
    found = rc.pool(task, 8, 2, np.random.default_rng(0), 0)
    assert len(found) == 1


def test_calibration_pool_refuses_invalid_embeddings(monkeypatch):
    task = SimpleNamespace(logical=nx.path_graph(2), host=nx.path_graph(2))
    monkeypatch.setattr(rc, "minorminer_initializer", lambda tries: lambda *args: {0: [0], 1: [0]})
    with pytest.raises(ValueError, match="disjoint"):
        rc.pool(task, 3, 2, np.random.default_rng(0), 0)
