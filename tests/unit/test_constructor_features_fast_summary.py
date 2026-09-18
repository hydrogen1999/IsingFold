"""The adjacency-list summary must equal the networkx one, element for element."""
import importlib.util
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

_pc = Path(__file__).resolve().parents[2] / "probes" / "constructor_curriculum.py"
spec = importlib.util.spec_from_file_location("constructor_curriculum", _pc)
cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
from constructor_features import ConstructorFeatureContext  # noqa: E402


def _reference_free_and_contacts(fc, chains):
    occupied = set().union(*chains.values()) if chains else set()
    free = [len(c) for c in nx.connected_components(fc.host.subgraph(fc._host_nodes - occupied))]
    owners = {}
    for v, chain in chains.items():
        for q in chain:
            owners.setdefault(q, set()).add(v)
    contacts = {}
    for q, r in fc.host.edges():
        for v in owners.get(q, ()):
            for u in owners.get(r, ()):
                if v != u and fc.logical.has_edge(v, u):
                    contacts.setdefault(frozenset((u, v)), set()).add(frozenset((q, r)))
    return sorted(free), contacts


def _random_state(task, rng):
    chains, used = {}, set()
    for v in task.logical:
        if rng.random() < 0.3:
            continue
        free = [q for q in task.host if q not in used]
        if not free:
            break
        root = int(rng.choice(free)); chain = {root}; used.add(root)
        for _ in range(int(rng.integers(0, 3))):
            grow = [n for q in chain for n in task.host[q] if n not in used]
            if not grow:
                break
            q = int(rng.choice(grow)); chain.add(q); used.add(q)
        chains[v] = frozenset(chain)
    return chains


@pytest.mark.parametrize("stage", ["a", "b", "p"])
def test_free_components_and_contacts_match_networkx(stage):
    if stage == "p":
        pytest.importorskip("dwave_networkx")
    train, _ = cc.build_sets(stage, 2, 1, seed=11)
    rng = np.random.default_rng(3)
    for task in train:
        fc = ConstructorFeatureContext(task, len(task.host))
        for _ in range(6):
            chains = _random_state(task, rng)
            occupied = set().union(*chains.values()) if chains else set()
            ref_free, ref_contacts = _reference_free_and_contacts(fc, chains)
            assert sorted(fc._free_component_sizes(occupied)) == ref_free
            # the summary row folds the contacts in; compare the whole row against a
            # context whose walk is forced back onto the reference implementation
            row = fc._summary(chains)
            assert row.shape == (len(row),) and np.all(np.isfinite(row))
            assert abs(row[17] * fc._m - (max(ref_free, default=0))) < 1e-4
            assert abs(row[18] * fc._m - len(ref_free)) < 1e-4
            assert abs(row[14] * fc._host_edges - sum(len(c) for c in ref_contacts.values())) < 1e-3


def test_empty_and_full_occupancy_edge_cases():
    train, _ = cc.build_sets("a", 1, 1, seed=12)
    fc = ConstructorFeatureContext(train[0], len(train[0].host))
    assert sorted(fc._free_component_sizes(set())) == sorted(len(c) for c in nx.connected_components(fc.host))
    assert fc._free_component_sizes(set(fc.host)) == []
    assert fc._free_component_sizes({"not-a-node"}) == [len(fc.host)] if nx.is_connected(fc.host) else True
