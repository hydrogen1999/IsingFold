"""The decision-sample generator: every kept sample is certified and self-consistent."""
from __future__ import annotations

import json
import random

import dwave_networkx as dnx
import networkx as nx

from embedbench.structural import (
    GenConfig, GenStats, decision_samples_from_witness, generate, greedy_local_choice,
)
from embedbench.exact import best_completion
from embedbench.inkdrop import ink_drop
from embedbench.objective import outcome_of


def _records(cfg, seed=0, n=2, mode="compact"):
    host = dnx.chimera_graph(cfg.size)
    rng = random.Random(seed)
    stats = GenStats()
    recs = []
    for i in range(n):
        pe = ink_drop(host, cfg.n_vars, cfg.chain_size, mode=mode, seed=seed + i)
        recs += list(decision_samples_from_witness(pe, cfg, rng, stats, f"t{i}"))
    return recs, stats


def test_kept_samples_have_positive_margin_and_a_feasible_best_action():
    cfg = GenConfig(size=3, n_vars=10, chain_size=3, k_in_play=3, radius=1, samples_per_instance=12)
    recs, stats = _records(cfg, n=4)
    assert recs, stats
    for r in recs:
        assert r.margin > 0
        best = r.values[r.actions.index(r.best_action)]
        assert best[0] == 1
        assert tuple(best) == max(map(tuple, r.values))
        assert r.witness_outcome[0] == 1  # the witness completes inside the window


def test_values_recompute_from_the_record_alone():
    """A consumer must be able to re-certify a sample from the JSON fields."""
    cfg = GenConfig(size=3, n_vars=10, chain_size=3, k_in_play=3, radius=1, samples_per_instance=12)
    recs, _ = _records(cfg, n=4)
    r = recs[0]
    wh = nx.Graph(); wh.add_nodes_from(r.window_nodes); wh.add_edges_from(map(tuple, r.window_edges))
    wl = nx.Graph(); wl.add_nodes_from(r.in_play); wl.add_edges_from(map(tuple, r.logical_edges))
    for (v, q), val in zip(r.actions, r.values):
        cores = {int(u): frozenset(c) for u, c in r.cores.items()}
        cores[v] = cores.get(v, frozenset()) | {q}
        out, chains = best_completion(wh, wl, cores, l_cap=r.l_cap)
        assert list(out) == val
        if out[0] == 1:
            assert outcome_of(chains, wl, wh) == out


def test_capacity_pressure_produces_feasibility_margins():
    cfg = GenConfig(size=3, n_vars=10, chain_size=3, k_in_play=3, radius=1,
                    samples_per_instance=10, q_cap_slack=0)
    recs, stats = _records(cfg, n=6, mode="near_capacity")
    kinds = {r.margin_kind for r in recs}
    assert "feasibility" in kinds, (kinds, stats)


def test_greedy_rule_is_deterministic_and_picks_a_listed_action():
    h = nx.path_graph(5)
    g = nx.Graph([(0, 1)])
    cores = {0: frozenset([0]), 1: frozenset()}
    acts = [(1, 1), (1, 3)]
    assert greedy_local_choice(h, g, cores, 1, acts) == (1, 1)


def test_generate_writes_jsonl_and_manifest(tmp_path):
    cfg = GenConfig(size=2, n_vars=6, chain_size=3, k_in_play=3, radius=1, samples_per_instance=3,
                    modes=("compact", "elongated"))
    out = tmp_path / "d.jsonl"
    stats = generate(cfg, 1, 0, out)
    lines = out.read_text().splitlines()
    assert len(lines) == stats.samples_kept
    m = json.loads((tmp_path / "d.jsonl.manifest.json").read_text())
    assert m["stats"]["samples_kept"] == stats.samples_kept and len(m["sha256"]) == 64
    for ln in lines:
        json.loads(ln)


def test_pegasus_windows_are_truncated_not_dropped():
    import dwave_networkx as dnx
    cfg = GenConfig(topology="pegasus", size=3, n_vars=12, chain_size=3, k_in_play=3, radius=1,
                    samples_per_instance=10, max_window_free=24)
    host = dnx.pegasus_graph(3)
    rng = random.Random(0)
    stats = GenStats()
    pe = ink_drop(host, cfg.n_vars, cfg.chain_size, mode="compact", seed=5)
    recs = list(decision_samples_from_witness(pe, cfg, rng, stats, "p"))
    assert stats.dropped_window_too_large == 0
    assert stats.samples_kept + stats.dropped_no_margin + stats.dropped_no_actions + stats.dropped_aborted == stats.samples_attempted
    for r in recs:
        assert len(set(r.window_nodes) - {q for c in r.cores.values() for q in c}) <= cfg.max_window_free
        assert r.witness_outcome[0] == 1


def test_generation_is_deterministic_for_a_seed(tmp_path):
    cfg = GenConfig(size=2, n_vars=6, chain_size=3, k_in_play=3, radius=1, samples_per_instance=4,
                    modes=("compact", "near_capacity"))
    generate(cfg, 2, 7, tmp_path / "a.jsonl")
    generate(cfg, 2, 7, tmp_path / "b.jsonl")
    assert (tmp_path / "a.jsonl").read_bytes() == (tmp_path / "b.jsonl").read_bytes()


def test_parallel_generation_matches_serial(tmp_path):
    cfg = GenConfig(size=3, n_vars=10, chain_size=3, k_in_play=3, radius=1, samples_per_instance=4,
                    modes=("compact", "elongated"))
    generate(cfg, 3, 11, tmp_path / "s.jsonl", jobs=1)
    generate(cfg, 3, 11, tmp_path / "p.jsonl", jobs=3)
    assert (tmp_path / "s.jsonl").read_bytes() == (tmp_path / "p.jsonl").read_bytes()
    assert (tmp_path / "s.jsonl").stat().st_size > 0
