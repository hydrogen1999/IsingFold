"""The two embedding transforms the frontier probes rely on must return valid embeddings.

`witness_prune.prune` produces the number the record calls the certified fill, and
`solvability_frontier.grow` produces the degraded arm of the discrimination table. If either
returns something that is not an embedding of the same logical graph, both tables are wrong in
a way no downstream check would catch, because the sampler happily samples an invalid program.
"""
import networkx as nx
import numpy as np
import pytest

from solvability_frontier import grow
from witness_prune import prune


def valid(host, chains, graph):
    """Non-empty, connected, disjoint chains realising every logical edge."""
    seen = set()
    for node, chain in chains.items():
        assert chain, f"chain for {node} is empty"
        assert nx.is_connected(host.subgraph(chain)), f"chain for {node} is disconnected"
        assert not (seen & set(chain)), f"chain for {node} overlaps another"
        seen |= set(chain)
    for u, v in graph.edges():
        assert any(host.has_edge(a, b) for a in chains[u] for b in chains[v]), \
            f"logical edge {u}-{v} has no realised contact"
    return sum(len(c) for c in chains.values())


@pytest.fixture
def line():
    """Three chains on a six-qubit path, with one removable qubit at each end."""
    host = nx.path_graph(6)
    graph = nx.Graph([(0, 1), (1, 2)])
    chains = {0: frozenset({0, 1}), 1: frozenset({2, 3}), 2: frozenset({4, 5})}
    return host, graph, chains


def test_prune_drops_exactly_the_redundant_qubits(line):
    host, graph, chains = line
    assert valid(host, chains, graph) == 6
    tight = prune(host, chains, graph)
    assert valid(host, tight, graph) == 4
    # The middle chain carries both contacts and cannot shrink; the ends each lose their tip.
    assert tight[0] == frozenset({1})
    assert tight[1] == frozenset({2, 3})
    assert tight[2] == frozenset({4})


def test_prune_is_idempotent(line):
    host, graph, chains = line
    once = prune(host, chains, graph)
    assert prune(host, once, graph) == once


def test_prune_never_grows_the_embedding():
    # Columns of a grid: neighbours touch, so the logical graph is the path, not the triangle.
    host = nx.grid_2d_graph(4, 4)
    graph = nx.Graph([(0, 1), (1, 2)])
    chains = {0: frozenset({(0, 0), (0, 1), (0, 2)}),
              1: frozenset({(1, 0), (1, 1), (1, 2)}),
              2: frozenset({(2, 0), (2, 1), (2, 2)})}
    before = valid(host, chains, graph)
    tight = prune(host, chains, graph)
    after = valid(host, tight, graph)
    assert after <= before


def test_grow_adds_at_most_the_requested_qubits_and_stays_valid(line):
    host, graph, chains = line
    grown, added = grow(chains, host, 0, np.random.default_rng(0))
    assert added == 0
    assert valid(host, grown, graph) == 6

    tight = prune(host, chains, graph)
    grown, added = grow(tight, host, 2, np.random.default_rng(0))
    assert added == 2
    assert valid(host, grown, graph) == 6


def test_grow_stops_when_the_host_is_full(line):
    host, graph, chains = line
    grown, added = grow(chains, host, 10, np.random.default_rng(0))
    assert added == 0, "a fully occupied host has nothing left to absorb"
    assert valid(host, grown, graph) == 6
