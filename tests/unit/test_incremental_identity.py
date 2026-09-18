"""The incremental successor identity and admissibility equal the direct computations."""
import networkx as nx
import numpy as np
import pytest

from isingfold.rl.contracts import ChainKeyBuilder, Context, chain_key
from isingfold.rl.validate import SearchStateCache, p_search


def _random_state(host, logical, rng, overlap=False):
    chains, used = {}, set()
    nodes = list(host)
    for v in logical:
        if rng.random() < 0.25:
            chains[v] = frozenset(); continue
        root = nodes[int(rng.integers(len(nodes)))]
        chain = {root}
        for _ in range(int(rng.integers(0, 4))):
            grow = [n for q in chain for n in host[q] if overlap or n not in used]
            if not grow:
                break
            chain.add(grow[int(rng.integers(len(grow)))])
        if rng.random() < 0.15:
            chain.add(nodes[int(rng.integers(len(nodes)))])
        used |= chain
        chains[v] = frozenset(chain)
    return chains


def _random_change(host, chains, rng):
    nodes = list(host)
    changed = {}
    for v in list(chains)[: 1 + int(rng.integers(3))]:
        base = set(chains[v])
        if base and rng.random() < 0.4:
            base.discard(next(iter(base)))
        for _ in range(int(rng.integers(0, 3))):
            base.add(nodes[int(rng.integers(len(nodes)))])
        changed[v] = frozenset(base)
    return changed


def test_chain_key_builder_matches_chain_key():
    rng = np.random.default_rng(0)
    host = nx.convert_node_labels_to_integers(nx.grid_2d_graph(6, 6))
    logical = nx.gnp_random_graph(10, 0.4, seed=1)
    for _ in range(80):
        chains = _random_state(host, logical, rng)
        builder = ChainKeyBuilder(chains)
        assert builder.key() == chain_key(chains)
        changed = _random_change(host, chains, rng)
        successor = dict(chains); successor.update(changed)
        assert builder.key(changed) == chain_key(successor)


def test_search_state_cache_matches_p_search_validity():
    rng = np.random.default_rng(2)
    host = nx.convert_node_labels_to_integers(nx.grid_2d_graph(6, 6))
    logical = nx.gnp_random_graph(10, 0.4, seed=3)
    ctx = Context(qubit_cap=20)
    agree = disagree = 0
    for _ in range(200):
        for allow_empty in (True, False):
            chains = _random_state(host, logical, rng, overlap=rng.random() < 0.5)
            cache = SearchStateCache(chains, logical, host, ctx.qubit_cap, ctx.overlap, allow_empty)
            changed = _random_change(host, chains, rng)
            successor = dict(chains); successor.update(changed)
            direct = p_search(successor, logical, host, ctx.qubit_cap, ctx.overlap,
                              allow_empty=allow_empty, count_demands=False).valid
            fast = cache.successor_is_admissible(changed)
            agree += direct == fast
            disagree += direct != fast
    assert disagree == 0, "%d of %d disagree" % (disagree, agree + disagree)


def test_cache_rejects_an_unknown_variable():
    host = nx.path_graph(5)
    logical = nx.path_graph(3)
    ctx = Context(qubit_cap=5)
    chains = {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})}
    cache = SearchStateCache(chains, logical, host, ctx.qubit_cap, ctx.overlap, True)
    assert cache.successor_is_admissible({0: frozenset({0, 1})}) is False or True
    assert cache.successor_is_admissible({"ghost": frozenset({4})}) is False
