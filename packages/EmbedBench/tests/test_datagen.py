"""Tests for the controlled data generation of the 2026-09-08 meeting note."""
from __future__ import annotations

import itertools
import random

import dwave_networkx as dnx
import networkx as nx
import pytest

from embedbench import (
    INFEASIBLE, MODES, MOTIFS, best_completion, build_motif, certify_motif, ink_drop,
    outcome_of, value_of_action,
)
from embedbench.exact import connected_supersets
from embedbench.inkdrop import PlantedGenerationError, ink_drop_with_retries


def _brute_force(host, logical, cores, l_cap):
    """Reference: enumerate every assignment of connected chains directly."""
    free = set(host.nodes) - set().union(*cores.values()) if cores else set(host.nodes)
    per_var = []
    for v in logical.nodes():
        core = cores.get(v, frozenset())
        opts = list(connected_supersets(host, core, free, l_cap))
        per_var.append(opts)
    best = INFEASIBLE
    for combo in itertools.product(*per_var):
        chains = dict(zip(logical.nodes(), combo))
        best = max(best, outcome_of(chains, logical, host))
    return best


def test_connected_supersets_are_connected_and_contain_core():
    h = nx.grid_2d_graph(3, 3)
    h = nx.convert_node_labels_to_integers(h)
    core = frozenset([4])
    free = set(h.nodes) - core
    sets = list(connected_supersets(h, core, free, 3))
    assert len(sets) == len(set(sets))
    for s in sets:
        assert core <= s and len(s) <= 3 and nx.is_connected(h.subgraph(s))
    # 1 (core) + 4 (size 2) + size-3 sets: 4 neighbours choose 2 = 6, plus 4 * 2 corner extensions = 8
    assert len(sets) == 1 + 4 + 6 + 8


@pytest.mark.parametrize("seed", range(6))
def test_best_completion_matches_brute_force(seed):
    rng = random.Random(seed)
    host = nx.convert_node_labels_to_integers(nx.grid_2d_graph(3, 3))
    n = 3
    logical = nx.Graph()
    logical.add_nodes_from(range(n))
    for u, v in itertools.combinations(range(n), 2):
        if rng.random() < 0.7:
            logical.add_edge(u, v)
    cores = {}
    if rng.random() < 0.5:
        cores[0] = frozenset([rng.choice(list(host.nodes))])
    exact, chains = best_completion(host, logical, cores, l_cap=2)
    assert exact == _brute_force(host, logical, cores, 2)
    if exact[0] == 1:
        assert outcome_of(chains, logical, host) == exact
        for v, c in cores.items():
            assert c <= chains[v]


def test_value_of_action_rejects_illegal_actions():
    h = nx.path_graph(4)
    g = nx.Graph([(0, 1)])
    with pytest.raises(ValueError):
        value_of_action(h, g, {0: [0]}, (1, 0))
    with pytest.raises(ValueError):
        value_of_action(h, g, {0: [0]}, (0, 3))


@pytest.mark.parametrize("name", sorted(MOTIFS))
def test_every_motif_is_certified_with_a_positive_margin(name):
    c = certify_motif(build_motif(name))
    assert c.certified_winner == c.hypothesis_winner, (name, c.sample.values)
    assert c.margin[1] > 0, (name, c.margin)


def test_motif_margins_cover_both_feasibility_and_quality():
    kinds = {certify_motif(build_motif(n)).margin[0] for n in MOTIFS}
    assert "feasibility" in kinds and "qubits" in kinds


@pytest.mark.parametrize("mode", sorted(MODES))
@pytest.mark.parametrize("host_name", ["chimera2", "pegasus2"])
def test_ink_drop_free_growth_is_a_valid_witness(mode, host_name):
    host = dnx.chimera_graph(2) if host_name == "chimera2" else dnx.pegasus_graph(2)
    pe = ink_drop(host, 6, 3, mode=mode, seed=1)
    assert pe.validate() == []
    assert set(pe.chains) == set(range(6))
    assert all(1 <= len(c) <= 3 for c in pe.chains.values())
    assert set(pe.logical.edges) == set(pe.contact_graph.edges)


def test_elongated_mode_makes_longer_thinner_chains_than_compact():
    host = dnx.chimera_graph(3)
    comp = ink_drop(host, 8, 6, mode="compact", seed=3)
    elon = ink_drop(host, 8, 6, mode="elongated", seed=3)
    def mean_internal_degree(pe):
        return sum(pe.host.subgraph(c).number_of_edges() / len(c) for c in pe.chains.values()) / len(pe.chains)
    assert mean_internal_degree(elon) <= mean_internal_degree(comp)


def test_target_guided_growth_realises_the_required_graph_or_fails_loudly():
    host = dnx.chimera_graph(2)
    required = nx.cycle_graph(4)
    pe, failures = ink_drop_with_retries(host, 4, 4, required=required, mode="cut_congested", tries=30, seed=0)
    assert pe.validate() == []
    assert set(pe.logical.edges) == set(required.edges)
    assert failures >= 0
    # an impossible requirement: K5 on a 2x2 grid with chains of one qubit
    grid = nx.convert_node_labels_to_integers(nx.grid_2d_graph(2, 2))
    with pytest.raises(PlantedGenerationError):
        ink_drop_with_retries(grid, 4, 1, required=nx.complete_graph(4), tries=5, seed=0)


def test_planted_record_round_trips_to_json():
    import json
    pe = ink_drop(dnx.chimera_graph(1), 3, 2, seed=0)
    d = json.loads(json.dumps(pe.to_dict()))
    assert d["Q"] == pe.qubits_used and len(d["chains"]) == 3
