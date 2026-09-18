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
        assert cc.minorminer.find_embedding is not cc._ORIGINAL_FIND_EMBEDDING
    assert summary["objective"] == "quality"
    for name in ("train", "heldout"):
        assert set(summary[name]["final_valid"]) == {"policy", "minorminer"}
        assert 0. <= summary[name]["final_valid"]["minorminer"] <= 1.
    assert set(summary["heldout"]["final_shape"]) == {"policy", "minorminer"}


def test_init_accepts_legacy_checkpoints_that_stored_the_feature_width(tmp_path):
    import torch
    actor = cc.make_actor("linear", 8, cc.FEATURE_WIDTHS["tiny"])
    torch.save({"state": actor.state_dict(), "actor": "linear", "features": cc.FEATURE_WIDTHS["tiny"]}, tmp_path / "old.pt")
    assert cc.load_init(str(tmp_path / "old.pt"), cc.make_actor("linear", 8, cc.FEATURE_WIDTHS["tiny"]), "linear", "tiny") is None
    with pytest.raises(ValueError):
        cc.load_init(str(tmp_path / "old.pt"), cc.make_actor("linear", 8, cc.FEATURE_WIDTHS["construction"]), "linear", "construction")


def test_zero_iterations_evaluates_only(tmp_path):
    args = cc.parse(["--stage", "a", "--train", "1", "--heldout", "1", "--episodes", "2",
                     "--iterations", "0", "--eval-episodes", "2", "--seed", "9", "--max-steps", "12"])
    train, heldout = cc.build_sets("a", 1, 1, seed=9)
    with cc.no_completion_solver():
        summary = cc.run(args, train, heldout)
    assert summary["heldout"]["init"] == summary["heldout"]["final"]


def test_grow_chains_adds_adjacent_free_qubits_and_keeps_chains_disjoint():
    host = nx.path_graph(6)
    chains = {0: frozenset({0}), 1: frozenset({3})}
    grown = cc.grow_chains(chains, host, 2, np.random.default_rng(0))
    assert sum(len(c) for c in grown.values()) == 4
    assert set(grown[0]).isdisjoint(grown[1])
    for c in grown.values():
        assert nx.is_connected(host.subgraph(c))
    full = cc.grow_chains({0: frozenset({0, 1, 2, 3, 4, 5})}, host, 3, np.random.default_rng(0))
    assert len(full[0]) == 6


def test_grown_minorminer_arm_runs_in_the_quality_protocol():
    args = cc.parse(["--stage", "a", "--train", "1", "--heldout", "2", "--episodes", "2",
                     "--iterations", "0", "--eval-episodes", "2", "--seed", "10", "--max-steps", "16",
                     "--objective", "quality", "--reward-reads", "16", "--selection-reads", "16",
                     "--assessment-reads", "16", "--select-cap", "2", "--deadline", "3",
                     "--comparison", "minorminer_grown", "--grow-extra", "1"])
    train, heldout = cc.build_sets("a", 1, 2, seed=10)
    with cc.no_completion_solver():
        summary = cc.run(args, train, heldout)
        # inside the guard the solver stays forbidden after the comparison arms ran
        assert cc.minorminer.find_embedding is not cc._ORIGINAL_FIND_EMBEDDING
    assert set(summary["heldout"]["final_valid"]) == {"policy", "minorminer", "minorminer_grown"}


def test_prefix_sources_exist_only_on_train_tasks_and_initializers_are_partial():
    train, heldout = cc.build_sets("b", 2, 2, seed=13)
    assert all(t.prefix_source for t in train) and all(t.prefix_source is None for t in heldout)
    t = train[0]
    init = cc.prefix_initializer(t, 0.5, seed=1)
    partial = init(t.logical, t.host, 0)
    assert 0 < len(partial) < len(t.logical) + 1 and set(partial) <= set(t.logical)
    used = set()
    for v, chain in partial.items():
        assert chain == frozenset(t.prefix_source[v]) and not (used & chain)
        used |= chain
    assert cc.prefix_initializer(heldout[0], 0.5, seed=1) is None
    full = cc.prefix_initializer(t, 1.0, seed=2)
    assert full is None or len(full(t.logical, t.host, 0)) == len(t.logical) - 1
    assert cc.prefix_initializer(t, 0.0, seed=1) is None
    assert cc.prefix_fraction("0.8:0.2", 0, 5) == 0.8 and abs(cc.prefix_fraction("0.8:0.2", 4, 5) - 0.2) < 1e-12
    assert cc.prefix_fraction("", 0, 5) is None


def test_prefix_curriculum_trains_from_partial_starts_and_evaluates_from_empty():
    args = cc.parse(["--stage", "a", "--train", "2", "--heldout", "1", "--episodes", "2",
                     "--iterations", "2", "--eval-episodes", "2", "--eval-every", "5", "--seed", "14",
                     "--max-steps", "16", "--prefix-fraction", "0.9:0.5"])
    train, heldout = cc.build_sets("a", 2, 1, seed=14)
    with cc.no_completion_solver():
        summary = cc.run(args, train, heldout)
    assert 0. <= summary["heldout"]["final"] <= 1.
    with pytest.raises(SystemExit):
        cc.parse(["--prefix-fraction", "1.5:0"])


def test_manifest_split_keeps_the_test_list_untouched():
    tasks = [_Fake("h-dense24-%d" % i, "L%d" % i) for i in range(10)]
    split = {"train": ["h-dense24-%d" % i for i in range(6)], "validation": ["L6", "L7"], "test": ["h-dense24-8", "L9"]}
    train, heldout = cc.manifest_split_sets(tasks, split, ["dense24"], 4, 2, seed=0)
    names = {t.name for t in train + heldout}
    assert len(train) == 4 and len(heldout) == 2
    assert "h-dense24-8" not in names and "h-dense24-9" not in names
    assert {t.lineage for t in heldout} <= {"L6", "L7"}
    assert all(t.prefix_source for t in train) and all(t.prefix_source is None for t in heldout)
    with pytest.raises(ValueError):
        cc.manifest_split_sets(tasks, split, ["dense24"], 7, 2, seed=0)


def test_test_role_uses_the_test_list_and_is_evaluation_only():
    tasks = [_Fake("h-dense24-%d" % i, "L%d" % i) for i in range(10)]
    split = {"train": ["L%d" % i for i in range(6)], "validation": ["L6", "L7"], "test": ["L8", "L9"]}
    train, heldout = cc.manifest_split_sets(tasks, split, [], 2, 2, seed=0, heldout_role="test")
    assert {t.lineage for t in heldout} == {"L8", "L9"} and all(t.lineage in split["train"] for t in train)
    with pytest.raises(ValueError):
        cc.manifest_split_sets(tasks, split, [], 2, 2, seed=0, heldout_role="ood")
    with pytest.raises(SystemExit):
        cc.parse(["--stage", "corpus", "--corpus", "x", "--manifest-split", "--heldout-role", "test", "--iterations", "5"])
    with pytest.raises(SystemExit):
        cc.parse(["--stage", "corpus", "--corpus", "x", "--heldout-role", "test", "--iterations", "0", "--init", "a.pt"])


def test_prefix_by_qubits_reaches_the_target_occupancy_and_is_shared_per_group():
    train, _ = cc.build_sets("b", 1, 1, seed=41)
    t = train[0]
    total = sum(len(c) for c in t.prefix_source.values())
    init = cc.prefix_initializer(t, 0.6, seed=7, unit="qubits")
    partial = init(t.logical, t.host, 0)
    covered = sum(len(c) for c in partial.values())
    assert covered >= 0.6 * total and len(partial) < len(t.logical)
    again = cc.prefix_initializer(t, 0.6, seed=7, unit="qubits")(t.logical, t.host, 0)
    assert again == partial
    other = cc.prefix_initializer(t, 0.6, seed=8, unit="qubits")(t.logical, t.host, 0)
    assert other != partial or len(t.logical) <= 3
    with pytest.raises(ValueError):
        cc.prefix_initializer(t, 0.6, seed=7, unit="edges")


def test_terminal_bias_moves_only_the_stop_and_restart_logits():
    import torch
    for features in ("tiny", "construction"):
        actor = cc.make_actor("linear", 8, cc.FEATURE_WIDTHS[features])
        before = actor.m.weight.detach().clone()
        assert cc.apply_terminal_bias(actor, features, -6.0)
        diff = (actor.m.weight.detach() - before)[0]
        moved = {i for i in range(len(diff)) if abs(float(diff[i])) > 0}
        assert moved == {cc.opcode_channel(features, "STOP"), cc.opcode_channel(features, "RESTART")}
        assert all(float(diff[i]) == -6.0 for i in moved)
    assert not cc.apply_terminal_bias(cc.make_actor("mlp", 8), "tiny", -6.0)
    assert not cc.apply_terminal_bias(cc.make_actor("linear", 8), "tiny", 0.0)


def test_mastery_schedule_and_options_parse_and_run():
    args = cc.parse(["--stage", "a", "--train", "2", "--heldout", "1", "--episodes", "2",
                     "--iterations", "2", "--eval-episodes", "1", "--eval-every", "9", "--seed", "42",
                     "--max-steps", "16", "--prefix-fraction", "0.9:0.0", "--prefix-schedule", "mastery",
                     "--mastery-threshold", "0.1", "--prefix-empty-mix", "0.5", "--stop-bias", "-6"])
    train, heldout = cc.build_sets("a", 2, 1, seed=42)
    with cc.no_completion_solver():
        summary = cc.run(args, train, heldout)
    assert 0. <= summary["heldout"]["final"] <= 1.
    with pytest.raises(SystemExit):
        cc.parse(["--prefix-empty-mix", "1.5"])


def test_heldout_only_evaluation_and_a_separate_training_deadline():
    args = cc.parse(["--stage", "a", "--train", "2", "--heldout", "1", "--episodes", "2", "--iterations", "1",
                     "--eval-episodes", "1", "--seed", "43", "--max-steps", "12", "--eval-sets", "heldout",
                     "--episode-seconds", "20", "--train-episode-seconds", "5"])
    train, heldout = cc.build_sets("a", 2, 1, seed=43)
    with cc.no_completion_solver():
        summary = cc.run(args, train, heldout)
    assert "heldout" in summary and "train" not in summary
    with pytest.raises(SystemExit):
        cc.parse(["--train-episode-seconds", "-1"])


def test_heuristic_growth_extends_the_most_strongly_coupled_chain_with_room():
    import networkx as nx
    from isingfold.embedding import LogicalProblem
    host = nx.path_graph(8)
    logical = nx.path_graph(3)
    problem = LogicalProblem.from_dicts({0: 0., 1: 0., 2: 0.}, {(0, 1): -0.2, (1, 2): -3.0})
    chains = {0: frozenset({0}), 1: frozenset({2}), 2: frozenset({4})}
    grown = cc.heuristic_growth(chains, host, logical, problem, 1)
    assert len(grown[1]) == 2 and len(grown[0]) == 1 and len(grown[2]) == 1   # variable 1 has mass 3.2
    more = cc.heuristic_growth(chains, host, logical, problem, 3)
    assert sum(len(c) for c in more.values()) == 6
    used = set()
    for c in more.values():
        assert not (used & c); used |= c


def test_the_all_comparison_reports_every_arm():
    args = cc.parse(["--stage", "a", "--train", "1", "--heldout", "2", "--episodes", "2", "--iterations", "0",
                     "--eval-episodes", "2", "--seed", "44", "--max-steps", "16", "--objective", "quality",
                     "--reward-reads", "16", "--selection-reads", "16", "--assessment-reads", "16",
                     "--select-cap", "2", "--deadline", "4", "--comparison", "all"])
    train, heldout = cc.build_sets("a", 1, 2, seed=44)
    with cc.no_completion_solver():
        summary = cc.run(args, train, heldout)
    arms = set(summary["heldout"]["final_valid"])
    assert arms == {"policy", "minorminer", "minorminer_grown", "minorminer_grown_selected",
                    "minorminer_grown_heuristic"}


def test_mastery_advances_on_assisted_episodes_only(monkeypatch):
    """A failed empty-start group must not hold the curriculum back."""
    args = cc.parse(["--stage", "a", "--train", "2", "--heldout", "1", "--episodes", "2",
                     "--iterations", "3", "--eval-episodes", "1", "--eval-every", "9", "--seed", "45",
                     "--max-steps", "12", "--prefix-fraction", "0.9:0.0", "--prefix-schedule", "mastery",
                     "--mastery-threshold", "1.0", "--mastery-step", "0.3", "--prefix-empty-mix", "0.5",
                     "--instances-per-iteration", "2"])
    train, heldout = cc.build_sets("a", 2, 1, seed=45)
    real_episode = cc.episode
    seen = {"assisted": 0, "empty": 0}

    def fake_episode(task, model, fc, temperature, max_steps, rng, deadline, **kw):
        record = real_episode(task, model, fc, temperature, 4, rng, deadline, **kw)
        assisted = kw.get("initializer") is not None
        seen["assisted" if assisted else "empty"] += 1
        record["valid"] = assisted            # assisted always succeeds, empty never does
        return record
    monkeypatch.setattr(cc, "episode", fake_episode)
    with cc.no_completion_solver():
        cc.run(args, train, heldout)
    assert seen["assisted"] and seen["empty"]
