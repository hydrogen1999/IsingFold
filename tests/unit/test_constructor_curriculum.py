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
        assert np.isfinite(t.ground_energy)  # reachable only by the quality backends
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
    contextual = cc.make_actor("contextual", 16, cc.FEATURE_WIDTHS["construction"])
    assert hasattr(contextual, "distribution_value")
    with pytest.raises(ValueError):
        cc.make_features("dense", None)


def test_contextual_actor_with_construction_features_runs_one_update(tmp_path):
    args = cc.parse(["--stage", "a", "--train", "2", "--heldout", "1", "--episodes", "2",
                     "--iterations", "1", "--eval-episodes", "2", "--eval-every", "5",
                     "--seed", "4", "--max-steps", "12", "--actor", "contextual",
                     "--features", "construction", "--baseline", "value", "--width", "8"])
    train, heldout = cc.build_sets("a", 2, 1, seed=4)
    with cc.no_completion_solver():
        summary = cc.run(args, train, heldout)
    assert summary["features"] == "construction" and summary["baseline"] == "value"
    assert 0. <= summary["heldout"]["final"] <= 1.


def test_value_baseline_requires_the_contextual_actor():
    with pytest.raises(SystemExit):
        cc.parse(["--actor", "linear", "--baseline", "value"])


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


@pytest.mark.parametrize("stage", ["P", "Z", "F", "G"])
def test_larger_hardware_fragments_match_their_declared_range(stage):
    pytest.importorskip("dwave_networkx")
    family, size, (lo, hi) = cc.HARDWARE[stage]
    for seed in range(2):
        h = cc.host_graph(np.random.default_rng(seed), stage)
        assert nx.is_connected(h) and lo <= h.number_of_nodes() <= hi
    assert cc.VARIABLES[stage] == ((8, 14) if stage in "PZ" else (12, 20))


class _Fake:
    def __init__(self, name, lineage):
        self.name, self.lineage = name, lineage
        self.logical = nx.path_graph(3); self.host = nx.cycle_graph(6)
        self.problem = None
        self.witness = {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})}


def test_corpus_sets_split_by_lineage_filter_by_cell_and_guard_the_witness():
    tasks = [_Fake("h-fill80-a2.0-%d" % i, "L%d" % (i // 2)) for i in range(8)]
    tasks += [_Fake("h-fill95-a3.0-%d" % i, "M%d" % i) for i in range(3)]
    train, heldout = cc.corpus_sets_from_tasks(tasks, ["fill80"], 3, 1, seed=0)
    assert len(train) == 3 and len(heldout) == 1
    assert {t.lineage for t in train}.isdisjoint({t.lineage for t in heldout})
    assert all("fill80" in t.name for t in train + heldout)
    with pytest.raises(AssertionError):
        train[0].witness
    with pytest.raises(ValueError):
        cc.corpus_sets_from_tasks(tasks, ["fill95"], 3, 1, seed=0)


def test_init_checkpoint_must_match_actor_and_features(tmp_path):
    import torch
    actor = cc.make_actor("linear", 8, cc.FEATURE_WIDTHS["tiny"])
    torch.save({"state": actor.state_dict(), "actor": "linear", "features": "tiny", "summary": {"x": 1}}, tmp_path / "a.pt")
    assert cc.load_init(str(tmp_path / "a.pt"), cc.make_actor("linear", 8, cc.FEATURE_WIDTHS["tiny"]), "linear", "tiny") == {"x": 1}
    with pytest.raises(ValueError):
        cc.load_init(str(tmp_path / "a.pt"), cc.make_actor("linear", 8, cc.FEATURE_WIDTHS["construction"]), "linear", "construction")
    with pytest.raises(SystemExit):
        cc.parse(["--stage", "corpus"])


def test_exact_ground_energy_on_small_problems():
    from isingfold.embedding import LogicalProblem
    tri = LogicalProblem.from_dicts({0: 0., 1: 0., 2: 0.}, {(0, 1): -1., (1, 2): -1., (0, 2): -1.})
    assert cc.exact_ground_energy(tri) == -3.
    frustrated = LogicalProblem.from_dicts({0: 0., 1: 0., 2: 0.}, {(0, 1): 1., (1, 2): 1., (0, 2): 1.})
    assert cc.exact_ground_energy(frustrated) == -1.
    field = LogicalProblem.from_dicts({0: 1., 1: -2.}, {(0, 1): 1.})
    assert cc.exact_ground_energy(field) == -4.
    with pytest.raises(ValueError):
        cc.exact_ground_energy(LogicalProblem.from_dicts({i: 0. for i in range(15)}, {}), limit=14)


def test_generated_tasks_carry_a_ground_energy_and_still_guard_the_witness():
    train, _ = cc.build_sets("a", 1, 1, seed=6)
    assert np.isfinite(train[0].ground_energy)
    with pytest.raises(AssertionError):
        train[0].witness


def test_quality_objective_runs_the_deadline_protocol_with_the_comparison_arm(tmp_path):
    args = cc.parse(["--stage", "a", "--train", "2", "--heldout", "1", "--episodes", "2",
                     "--iterations", "1", "--eval-episodes", "2", "--eval-every", "5", "--seed", "8",
                     "--max-steps", "16", "--objective", "quality", "--reward-reads", "16",
                     "--selection-reads", "16", "--assessment-reads", "16", "--select-cap", "2",
                     "--deadline", "3", "--comparison", "minorminer"])
    train, heldout = cc.build_sets("a", 2, 1, seed=8)
    with cc.no_completion_solver():
        summary = cc.run(args, train, heldout)
    assert summary["objective"] == "quality"
    for name in ("train", "heldout"):
        assert set(summary[name]["final_valid"]) == {"policy", "minorminer"}
        assert 0. <= summary[name]["final_valid"]["minorminer"] <= 1.
    # the guard is back in force after the comparison arm ran
    with pytest.raises(AssertionError):
        cc.certified_embeddable(train[0].logical, train[0].host, 0) if cc.minorminer.find_embedding is cc.forbidden_solver else (_ for _ in ()).throw(AssertionError())
