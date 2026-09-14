"""The decision objective, section 4 of the 2026-09-08 meeting note.

A complete embedding is scored lexicographically:

    feasible completion  >  final resource quality (Q, then L_max)

"Future capacity/connectivity", the middle term of the meeting's priority, is not a property
of a finished embedding; it is what V* measures on a partial one. So the terminal outcome is
(feasible, -Q, -L_max) and V*(S_t, a) is the best terminal outcome reachable after `a`. An
action that keeps a feasible completion beats one that does not, whatever it saves in qubits;
among feasible continuations fewer qubits win, then shorter longest chain.

Outcomes are tuples so that `max` and `>` do the right thing.
"""
from __future__ import annotations

from collections.abc import Mapping

import networkx as nx

from embedbench.embedding import Node, Qubit

Outcome = tuple[int, int, int]
"""(feasible, -Q, -L_max). Larger is better."""

INFEASIBLE: Outcome = (0, 0, 0)


def outcome_of(chains: Mapping[Node, frozenset[Qubit]], logical: nx.Graph, host: nx.Graph) -> Outcome:
    """Score a complete assignment. Infeasible if any chain is empty or disconnected, chains
    overlap, or a logical edge has no coupler between its two chains."""
    seen: set[Qubit] = set()
    for i, c in chains.items():
        if not c or not nx.is_connected(host.subgraph(c)):
            return INFEASIBLE
        if seen & c:
            return INFEASIBLE
        seen |= c
    for u, v in logical.edges():
        if u not in chains or v not in chains:
            return INFEASIBLE
        if not any(host.has_edge(a, b) for a in chains[u] for b in chains[v]):
            return INFEASIBLE
    q = sum(len(c) for c in chains.values())
    lmax = max(len(c) for c in chains.values())
    return (1, -q, -lmax)


def compare(better: Outcome, worse: Outcome) -> tuple[str, int]:
    """The decision margin between two outcomes: which component separates them and by how
    much. ('feasibility', 1) when one completes and the other cannot; ('qubits', dQ) or
    ('max_chain', dL) otherwise; ('none', 0) when equal."""
    if better[0] != worse[0]:
        return "feasibility", better[0] - worse[0]
    if better[1] != worse[1]:
        return "qubits", better[1] - worse[1]
    if better[2] != worse[2]:
        return "max_chain", better[2] - worse[2]
    return "none", 0
