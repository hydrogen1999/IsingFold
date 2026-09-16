"""A contact-growth episode adds only qubits that touch a coupled chain, keeps chains
connected and disjoint, and returns one log-probability per move."""
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


def test_contact_episode_adds_touching_qubits_only():
    import torch
    from candidate_features import FeatureContext
    from train_contact_policy import episode, random_episode
    from train_prioritiser import Prioritiser
    from . import test_witness_replay as tw

    # A grid of degree four rarely has a free qubit touching two chains; Pegasus does.
    from gen_fill_corpus import plant_partition, quotient
    from isingfold.embedding import LogicalProblem
    from isingfold.rl.data.generate import host_graph
    from isingfold.rl.env import EmbeddingTask

    host = host_graph("pegasus", 2)
    rng = np.random.default_rng(0)
    chains = plant_partition(host, 0.5, 3.0, 1, 2, rng)
    logical = quotient(host, chains)
    comp = sorted(max(nx.connected_components(logical), key=len))
    relabel = {v: i for i, v in enumerate(comp)}
    logical = nx.relabel_nodes(logical.subgraph(comp).copy(), relabel)
    witness = {relabel[v]: frozenset(chains[v]) for v in comp}
    j = {(u, v): 1.0 for u, v in logical.edges()}
    problem = LogicalProblem.from_dicts({}, j)
    task = EmbeddingTask(name="pegasus2-toy", logical=problem.graph, host=host, problem=problem,
                         ground_energy=None, lineage="toy-l0", witness=witness)
    start = {v: frozenset(c) for v, c in witness.items() if v in problem.graph}
    torch.manual_seed(0)
    from candidate_features import FRONTIER_WIDTH, WIDTH
    model = Prioritiser(16, in_dim=WIDTH + FRONTIER_WIDTH)
    grown, logps = episode(task, start, model, FeatureContext(task), 3, 1.0, np.random.default_rng(0))
    added = sum(len(grown[v]) - len(start[v]) for v in start)
    assert 1 <= added <= 3 and len(logps) == added
    seen = set()
    for v, c in grown.items():
        assert start[v] <= c and nx.is_connected(task.host.subgraph(c))
        assert not (seen & set(c)); seen |= set(c)
    rnd = random_episode(task, start, 3, np.random.default_rng(1))
    assert sum(len(rnd[v]) - len(start[v]) for v in start) >= 1
