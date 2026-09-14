"""Shared helpers for the hand-tests: tiny, explicit, no cleverness."""
from __future__ import annotations

import itertools
import sys

import networkx as nx


class HandTestFailure(AssertionError):
    """Raised with a sentence a person can act on."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise HandTestFailure(message)


def explain_and_exit_if_asked(docstring: str) -> None:
    if "--explain" in sys.argv:
        print(docstring.strip())
        print()


def brute_force_ground_energy(problem) -> tuple[float, dict]:
    """Minimum energy of a logical Ising model by enumerating every spin assignment.

    Only for problems small enough to enumerate; the caller is responsible for that.
    """
    nodes = sorted(problem.h)
    best_energy, best_state = None, None
    for bits in itertools.product((-1, 1), repeat=len(nodes)):
        state = dict(zip(nodes, bits))
        energy = sum(problem.h[v] * state[v] for v in nodes)
        energy += sum(w * state[u] * state[v] for (u, v), w in problem.j.items())
        if best_energy is None or energy < best_energy:
            best_energy, best_state = energy, state
    return float(best_energy), best_state


def chains_are_valid_embedding(chains, logical: nx.Graph, host: nx.Graph) -> list[str]:
    """Every reason the chains fail to be a minor embedding, as sentences. Empty means valid."""
    problems: list[str] = []
    seen: dict = {}
    for variable, chain in chains.items():
        if not chain:
            problems.append(f"variable {variable} has an empty chain")
            continue
        missing = [q for q in chain if q not in host]
        if missing:
            problems.append(f"variable {variable} uses qubits absent from the host: {missing[:4]}")
            continue
        if not nx.is_connected(host.subgraph(chain)):
            problems.append(f"the chain of variable {variable} is not connected in the host")
        for q in chain:
            if q in seen:
                problems.append(f"qubit {q} is used by both {seen[q]} and {variable}")
            seen[q] = variable
    for u, v in logical.edges():
        cu, cv = chains.get(u, frozenset()), chains.get(v, frozenset())
        if not any(host.has_edge(a, b) for a in cu for b in cv):
            problems.append(f"logical edge ({u}, {v}) has no coupler between its chains")
    for v in logical.nodes():
        if v not in chains:
            problems.append(f"variable {v} has no chain at all")
    return problems
