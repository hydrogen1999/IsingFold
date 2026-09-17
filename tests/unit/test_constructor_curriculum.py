import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

_p = Path(__file__).resolve().parents[2] / "probes" / "constructor_curriculum.py"
spec = importlib.util.spec_from_file_location("constructor_curriculum", _p)
cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)


@pytest.mark.parametrize("n", [2, 3, 4, 6, 8])
def test_logical_graphs_are_connected_with_the_requested_size(n):
    for seed in range(6):
        g, family = cc.logical_graph(np.random.default_rng(seed), n)
        assert g.number_of_nodes() == n and nx.is_connected(g)
        assert sorted(g.nodes()) == list(range(n))
        assert family in ("path", "cycle", "star", "complete", "random", "triangle_tail")


@pytest.mark.parametrize("stage", ["a", "b"])
def test_hosts_are_connected_and_carry_a_dead_end(stage):
    for seed in range(6):
        h = cc.host_graph(np.random.default_rng(seed), stage)
        assert nx.is_connected(h)
        assert any(d == 1 for _, d in h.degree()), "a pendant dead end is required"
        assert sorted(h.nodes()) == list(range(h.number_of_nodes()))


def test_unknown_stage_is_rejected():
    with pytest.raises(ValueError):
        cc.host_graph(np.random.default_rng(0), "q")


def test_sets_are_certified_disjoint_and_guarded():
    train, heldout = cc.build_sets("a", 3, 3, seed=1)
    assert len(train) == 3 and len(heldout) == 3
    names = [t.name for t in train + heldout]
    assert len(set(names)) == len(names)
    for t in train + heldout:
        assert cc.certified_embeddable(t.logical, t.host, 0)
        with pytest.raises(AssertionError):
            t.witness
        with pytest.raises(AssertionError):
            t.ground_energy
        with pytest.raises(AssertionError):
            t.initial_embedding
    for a in train:
        for b in heldout:
            assert not cc.isomorphic_pair(a, b)


def test_minorminer_is_forbidden_inside_the_guard():
    train, _ = cc.build_sets("a", 1, 1, seed=2)
    with cc.no_completion_solver():
        with pytest.raises(AssertionError):
            cc.certified_embeddable(train[0].logical, train[0].host, 0)


def test_one_update_runs_end_to_end_and_reports_both_sets(tmp_path):
    args = cc.parse(["--stage", "a", "--train", "2", "--heldout", "1", "--episodes", "2",
                     "--iterations", "1", "--eval-episodes", "2", "--eval-every", "5",
                     "--seed", "3", "--max-steps", "16", "--out", str(tmp_path / "g.pt")])
    train, heldout = cc.build_sets("a", 2, 1, seed=3)
    with cc.no_completion_solver():
        summary = cc.run(args, train, heldout)
    for name in ("train", "heldout"):
        assert 0. <= summary[name]["init"] <= 1. and 0. <= summary[name]["final"] <= 1.
        assert summary[name]["instances"] == (2 if name == "train" else 1)
        lo, hi = summary[name]["gain_ci"]
        assert lo <= summary[name]["gain"] <= hi
    assert (tmp_path / "g.pt").exists()


def test_mlp_actor_has_more_parameters_than_linear():
    linear = sum(p.numel() for p in cc.make_actor("linear", 32).parameters())
    mlp = sum(p.numel() for p in cc.make_actor("mlp", 32).parameters())
    assert linear == cc.FEATURE_WIDTH and mlp > linear
    with pytest.raises(ValueError):
        cc.make_actor("gnn", 32)


@pytest.mark.parametrize("stage", ["p", "z"])
def test_hardware_fragments_are_connected_and_sized(stage):
    pytest.importorskip("dwave_networkx")
    for seed in range(4):
        h = cc.host_graph(np.random.default_rng(seed), stage)
        assert nx.is_connected(h)
        assert cc.FRAGMENT[0] <= h.number_of_nodes() <= cc.FRAGMENT[1]
        assert sorted(h.nodes()) == list(range(h.number_of_nodes()))


def test_hardware_stage_sets_build_and_stay_disjoint():
    pytest.importorskip("dwave_networkx")
    train, heldout = cc.build_sets("p", 2, 2, seed=5)
    assert len(train) == 2 and len(heldout) == 2
    for a in train:
        for b in heldout:
            assert not cc.isomorphic_pair(a, b)
    assert all(4 <= t.logical.number_of_nodes() <= 8 for t in train + heldout)
