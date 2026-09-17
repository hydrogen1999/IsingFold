"""The memoised connectivity and contact checks equal the direct networkx computation."""
import networkx as nx
import numpy as np

from isingfold.rl.validate import chain_is_connected, chains_are_connected, chains_touch, unrealized_demands


def _reference_unrealized(chains, logical, host):
    count = 0
    for u, v in logical.edges():
        cu, cv = chains.get(u, frozenset()), chains.get(v, frozenset())
        if not cu or not cv or not any(host.has_edge(q, r) for q in cu for r in cv if q != r):
            count += 1
    return count


def _reference_connected(chains, host):
    bad = []
    for i, chain in chains.items():
        if not chain:
            continue
        if not set(chain) <= set(host.nodes):
            bad.append(i); continue
        if len(chain) > 1 and not nx.is_connected(host.subgraph(chain)):
            bad.append(i)
    return tuple(bad)


def _random_chains(host, logical, rng, overlap=False):
    chains, used = {}, set()
    nodes = list(host)
    for v in logical:
        if rng.random() < 0.2:
            chains[v] = frozenset(); continue
        root = nodes[int(rng.integers(len(nodes)))]
        chain = {root}
        for _ in range(int(rng.integers(0, 4))):
            grow = [n for q in chain for n in host[q] if overlap or n not in used]
            if not grow:
                break
            chain.add(grow[int(rng.integers(len(grow)))])
        if rng.random() < 0.15:
            chain.add(nodes[int(rng.integers(len(nodes)))])   # sometimes disconnected
        if rng.random() < 0.05:
            chain.add("off-host")
        used |= chain
        chains[v] = frozenset(chain)
    return chains


def test_memoised_checks_match_networkx_on_random_states():
    rng = np.random.default_rng(0)
    host = nx.convert_node_labels_to_integers(nx.grid_2d_graph(7, 7))
    logical = nx.gnp_random_graph(12, 0.4, seed=3)
    for _ in range(300):
        chains = _random_chains(host, logical, rng, overlap=rng.random() < 0.5)
        assert chains_are_connected(chains, host) == _reference_connected(chains, host)
        assert unrealized_demands(chains, logical, host) == _reference_unrealized(chains, logical, host)


def test_pair_and_chain_primitives():
    host = nx.path_graph(6)
    assert chain_is_connected(frozenset({0, 1, 2}), host) and not chain_is_connected(frozenset({0, 2}), host)
    assert chain_is_connected(frozenset({4}), host)
    assert chains_touch(frozenset({0}), frozenset({1}), host) and not chains_touch(frozenset({0}), frozenset({2}), host)
    assert not chains_touch(frozenset({3}), frozenset({3}), host)     # sharing a qubit is not a contact
    assert chains_touch(frozenset({3}), frozenset({3, 4}), host)
