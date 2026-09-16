"""Task 12: a planted witness must be reachable through the construction action API.

Imitation needs the teacher's trajectory to be expressible as a sequence of the environment's
own candidates. This drives the construction environment on a toy host, choosing at every
decision a candidate consistent with the witness (a PLACE whose root lies in the witness chain
of that variable, a ROUTE whose owned segment stays inside the witness chain), and asks for
the witness back. Where no consistent candidate is offered, the failure message says which
variable and how many decisions were left, which is the defect to fix.
"""
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


def _planted_task(side=4, fill=0.75, seed=0, lmax=3):
    from gen_fill_corpus import plant_partition, quotient
    from isingfold.embedding import LogicalProblem
    from isingfold.rl.env import EmbeddingTask

    host = nx.convert_node_labels_to_integers(nx.grid_2d_graph(side, side))
    rng = np.random.default_rng(seed)
    chains = plant_partition(host, fill, 3.0, 1, lmax, rng)
    logical = quotient(host, chains)
    comp = max(nx.connected_components(logical), key=len)
    keep = sorted(comp)
    relabel = {v: i for i, v in enumerate(keep)}
    logical = nx.relabel_nodes(logical.subgraph(keep).copy(), relabel)
    witness = {relabel[v]: frozenset(chains[v]) for v in keep}
    j = {(u, v): 1.0 if (u + v) % 2 else -1.0 for u, v in logical.edges()}
    problem = LogicalProblem.from_dicts({}, j)
    task = EmbeddingTask(name="toy", logical=problem.graph, host=host, problem=problem,
                         ground_energy=None, lineage="toy-l0", witness=witness)
    return task, witness


def _consistent(cand, witness, current):
    from isingfold.rl.contracts import Opcode

    if cand.opcode is Opcode.PLACE:
        (v, chain), = cand.new_chains.items()
        return chain <= witness[v]
    if cand.opcode is Opcode.ROUTE:
        return all(chain <= witness[v] for v, chain in cand.new_chains.items())
    if cand.opcode is Opcode.COMMIT:
        # Every chain placed and inside its witness chain; the environment only offers
        # COMMIT when the embedding is valid.
        return all(current.get(v) and current[v] <= c for v, c in witness.items())
    return False


def replay(task, witness, max_steps=400):
    from _context import construction_context
    from isingfold.rl.contracts import DecisionState, Opcode
    from isingfold.rl.env import EmbeddingEnv, Mode, fixed_strength_selector

    ctx = construction_context(task.host.number_of_nodes(), task.logical.number_of_nodes(),
                               task.logical.number_of_edges())
    env = EmbeddingEnv(task, ctx, mode=Mode.CONSTRUCTION, initializer=None,
                       selector=fixed_strength_selector(), reward_reads=8)
    dec = env.reset(0)
    steps = 0
    while isinstance(dec, DecisionState) and steps < max_steps:
        current = env.state.chains
        choices = [i for i, (c, ok) in enumerate(zip(dec.candidates, dec.legal_mask))
                   if ok and _consistent(c, witness, current)]
        if not choices:
            placed = sum(1 for v in witness if current.get(v))
            offered = sorted({c.opcode.value for c, ok in zip(dec.candidates, dec.legal_mask) if ok})
            return False, ("no witness-consistent candidate at step %d: %d/%d variables placed, "
                           "%d decisions left, offered %s"
                           % (steps, placed, len(witness), env.state.remaining.decisions, offered))
        # Prefer COMMIT once the witness is complete, else the first consistent candidate.
        commit = [i for i in choices if dec.candidates[i].opcode is Opcode.COMMIT]
        pick = commit[0] if commit else choices[0]
        dec = env.step(dec, pick, evaluate_training_reward=False).next_decision_or_terminal
        steps += 1
    if isinstance(dec, DecisionState):
        return False, "no terminal after %d steps" % steps
    returned = getattr(dec, "embedding", None)
    if returned is None or not getattr(dec, "returned_valid", False):
        return False, "terminal without a valid embedding: %s" % (getattr(dec, "terminal_reason", dec),)
    # Every chain the environment returns lies inside the witness chain: the router may drop
    # a qubit the witness did not need, and that is still the witness's embedding.
    inside = all(returned.get(v, frozenset()) and returned[v] <= c for v, c in witness.items())
    return inside, "returned valid but outside the witness"


def test_a_planted_witness_replays_through_the_construction_api_on_a_small_host():
    task, witness = _planted_task()
    ok, why = replay(task, witness)
    assert ok, why


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_planted_witness_replays_on_a_host_of_a_hundred_qubits(seed):
    # Before Task 12 this failed at step 0: PLACE offered the 24 lexicographically first
    # roots for the first empty variable, the horizon was 32 decisions, and ROUTE offered one
    # router path per demand. Placement now follows placed neighbours, the horizon scales
    # with the instance, and one-qubit bridges are offered beside the router path.
    task, witness = _planted_task(side=10, fill=0.8, seed=seed, lmax=2)
    assert len(witness) >= 40
    ok, why = replay(task, witness, max_steps=2000)
    assert ok, why
