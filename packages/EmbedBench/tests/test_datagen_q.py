"""Planted Ising layer and downstream-quality decision samples."""
from __future__ import annotations

import itertools
import json
import random

import dwave_networkx as dnx
import networkx as nx
import pytest

from embedbench.quality_step import QConfig, QStats, generate_q, q_samples_from_witness
from embedbench.exact import SearchAborted, all_completions, best_completion
from embedbench.inkdrop import ink_drop
from embedbench.objective import outcome_of
from embedbench.planted_ising import PlantingError, frustrated_loops


@pytest.mark.parametrize("seed", range(4))
def test_planted_state_is_a_ground_state_by_brute_force(seed):
    pe = ink_drop(dnx.chimera_graph(2), 10, 2, mode="compact", seed=seed)
    pl = frustrated_loops(pe.logical, alpha=0.5, seed=seed)
    nodes = sorted(pl.problem.graph.nodes())
    assert abs(pl.energy_of_planted() - pl.ground_energy) < 1e-9
    best = min(
        sum(w * s[a] * s[b] for (a, b), w in pl.problem.j.items())
        for s in ({v: x for v, x in zip(nodes, cfg)} for cfg in itertools.product((-1, 1), repeat=len(nodes)))
    )
    assert abs(best - pl.ground_energy) < 1e-9
    assert max(abs(v) for v in pl.problem.j.values()) <= 1.0 + 1e-12


def test_planting_fails_loudly_on_a_tree():
    with pytest.raises(PlantingError):
        frustrated_loops(nx.path_graph(6), alpha=0.5, seed=0)


def test_all_completions_agree_with_best_completion():
    host = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))
    logical = nx.Graph([(0, 1), (1, 2)])
    cores = {0: frozenset([4])}
    comps = all_completions(host, logical, cores, l_cap=2)
    assert comps
    best_direct, _ = best_completion(host, logical, cores, l_cap=2)
    assert max(outcome_of(c, logical, host) for c in comps) == best_direct
    for c in comps:
        assert outcome_of(c, logical, host)[0] == 1
    with pytest.raises(SearchAborted):
        all_completions(host, logical, cores, l_cap=2, max_count=2)


def test_quality_samples_are_reproducible_and_self_consistent(tmp_path):
    cfg = QConfig(size=3, n_vars=20, chain_size=3, modes=("elongated",), samples_per_instance=3,
                  alpha=0.5, max_completions=40, num_reads=50, refine_reads=100)
    st1 = generate_q(cfg, 2, 5, tmp_path / "a.jsonl", jobs=1)
    st2 = generate_q(cfg, 2, 5, tmp_path / "b.jsonl", jobs=2)
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()
    assert st1.samples_attempted == st2.samples_attempted > 0
    for ln in (tmp_path / "a.jsonl").read_text().splitlines():
        r = json.loads(ln)
        best = r["values"][r["actions"].index(r["best_action"])]
        assert best[0] == max(v[0] for v in r["values"])
        assert r["margin"] >= cfg.min_margin
        assert len(r["frozen"]) + len(r["in_play"]) == cfg.n_vars
        assert r["witness_p_solve"] >= 0.0


def test_defective_host_is_connected_and_smaller():
    from embedbench.quality_step import defective_host
    h = dnx.chimera_graph(3)
    d = defective_host(h, 0.05, 0.03, seed=1)
    assert nx.is_connected(d)
    assert d.number_of_nodes() < h.number_of_nodes()
    assert d.number_of_edges() < h.number_of_edges()
    assert defective_host(h, 0.05, 0.03, seed=1).number_of_edges() == d.number_of_edges()


def test_hard_config_generates_with_fill_and_defects(tmp_path):
    cfg = QConfig(size=3, chain_size=3, modes=("near_capacity",), samples_per_instance=2, alpha=0.8,
                  max_completions=40, num_reads=50, refine_reads=100, defect_qubits=0.05, defect_couplers=0.03,
                  fill=0.7, difficulty="hard-test", k_in_play=3, l_cap=3)
    st = generate_q(cfg, 2, 3, tmp_path / "h.jsonl", jobs=1)
    assert st.instances >= 1
    for ln in (tmp_path / "h.jsonl").read_text().splitlines():
        r = json.loads(ln)
        assert r["difficulty"] == "hard-test" and r["host_nodes"] < 72 and r["n_vars"] >= 4


def test_completions_realise_edges_to_frozen_chains():
    """Regression for the 2026-09-08 label bug: a completion that leaves a logical edge to a
    frozen variable unrealised must not count as feasible."""
    from embedbench.structural import frozen_requirements
    host = nx.path_graph(7)             # 0-1-2-3-4-5-6
    logical = nx.Graph([(0, 1), (1, 2)])
    frozen = {2: frozenset([6])}        # variable 2 sits at the far end
    in_play = [0, 1]
    window = {0, 1, 2, 3, 4, 5}
    must_hit, edges = frozen_requirements(host, logical, in_play, frozen, window)
    assert edges == [[1, 2]] and must_hit[1] == [frozenset([5])]
    wh = host.subgraph(window)
    wl = logical.subgraph(in_play)
    comps = all_completions(wh, wl, {0: frozenset([0])}, l_cap=4, must_hit=must_hit)
    assert comps and all(5 in c[1] for c in comps)
    loose = all_completions(wh, wl, {0: frozenset([0])}, l_cap=4)
    assert len(loose) > len(comps)
    out, _ = best_completion(wh, wl, {0: frozenset([0])}, l_cap=2, must_hit=must_hit)
    assert out[0] == 0   # with chains of at most 2, variable 1 cannot touch 0's chain and qubit 5
    out3, _ = best_completion(wh, wl, {0: frozenset([0])}, l_cap=3, must_hit=must_hit)
    assert out3 == (1, -6, -3)   # 0 -> {0,1,2}, 1 -> {3,4,5}
