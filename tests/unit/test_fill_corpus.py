"""The fill-planted generator reports what the witness uses and the bound no embedding beats."""
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

PROBES = Path(__file__).resolve().parents[2] / "probes"
SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture(autouse=True)
def _paths(monkeypatch):
    monkeypatch.setenv("ISINGFOLD_SRC", str(SRC))
    monkeypatch.syspath_prepend(str(PROBES))
    monkeypatch.syspath_prepend(str(SRC))


def test_planted_partition_is_a_valid_embedding_of_its_quotient_at_the_requested_fill():
    from gen_fill_corpus import lower_bound_fill, plant_partition, quotient, witness_occupancy

    host = nx.grid_2d_graph(6, 6)
    rng = np.random.default_rng(0)
    chains = plant_partition(host, 0.75, 3.0, 1, 4, rng)
    logical = quotient(host, chains)
    witness = {v: frozenset(chains[v]) for v in logical.nodes()}
    used = witness_occupancy(witness)
    assert used == round(0.75 * host.number_of_nodes())
    seen = set()
    for c in witness.values():
        assert host.subgraph(c).number_of_nodes() == len(c)
        assert nx.is_connected(host.subgraph(c))
        assert not (seen & set(c))
        seen |= set(c)
    for u, v in logical.edges():
        assert any(host.has_edge(a, b) for a in witness[u] for b in witness[v])
    assert lower_bound_fill(logical.number_of_nodes(), host.number_of_nodes()) <= 0.75
