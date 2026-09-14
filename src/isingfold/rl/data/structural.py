"""Exact structural labels inside a registered continuation domain.

Spec: Rev2 section "Exact structural labels and their continuation domain". The label
``y_A(s, a)`` says whether *some* completion in a finite registered domain is complete, valid
and in budget. A timeout is ``None`` (unknown), never a negative: heuristic failure cannot
certify infeasibility. Minimum qubits and minimum longest chain are reported inside the same
feasible set; they are local structural descriptors, not quality labels.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Hashable, Mapping

import networkx as nx

Node = Hashable
Qubit = Hashable


@dataclass(frozen=True)
class ContinuationDomain:
    """Frozen exterior, movable variables, allowed window, caps and horizon."""

    movable: tuple[Node, ...]
    window: frozenset[Qubit]
    max_chain: int = 3
    qubit_cap: int | None = None
    timeout_ms: int = 2_000


@dataclass(frozen=True)
class StructuralLabel:
    feasible: bool | None
    witness: Mapping[Node, frozenset[Qubit]] | None
    min_qubits: int | None
    min_max_chain: int | None
    exhausted: bool
    nodes_explored: int
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "feasible": self.feasible,
            "min_qubits": self.min_qubits,
            "min_max_chain": self.min_max_chain,
            "exhausted": self.exhausted,
            "nodes_explored": self.nodes_explored,
            "reason": self.reason,
        }


def connected_subsets(
    host: nx.Graph, window: frozenset[Qubit], max_size: int, blocked: frozenset[Qubit]
) -> list[frozenset[Qubit]]:
    """Every connected subset of the window up to ``max_size``, avoiding blocked qubits."""

    allowed = [q for q in sorted(window, key=str) if q not in blocked]
    found: set[frozenset[Qubit]] = set()
    for start in allowed:
        stack = [frozenset({start})]
        while stack:
            current = stack.pop()
            if current in found:
                continue
            found.add(current)
            if len(current) >= max_size:
                continue
            for q in current:
                for r in host.neighbors(q):
                    if r in window and r not in blocked and r not in current:
                        stack.append(current | {r})
    return sorted(found, key=lambda s: (len(s), sorted(map(str, s))))


def exact_feasibility(
    chains: Mapping[Node, frozenset[Qubit]],
    logical: nx.Graph,
    host: nx.Graph,
    domain: ContinuationDomain,
) -> StructuralLabel:
    """Exhaustively decide completion inside the domain; report unknown on timeout."""

    logical_nodes = set(logical.nodes())
    movable_set = set(domain.movable)
    if len(movable_set) != len(domain.movable) or not movable_set <= logical_nodes:
        return StructuralLabel(False, None, None, None, True, 0, "invalid movable-variable domain")
    if domain.max_chain <= 0 or domain.timeout_ms <= 0:
        return StructuralLabel(False, None, None, None, True, 0, "invalid continuation horizon")
    if domain.qubit_cap is not None and domain.qubit_cap <= 0:
        return StructuralLabel(False, None, None, None, True, 0, "invalid qubit cap")
    if not set(domain.window) <= set(host.nodes()):
        return StructuralLabel(False, None, None, None, True, 0, "window leaves the active host")
    if not set(chains) <= logical_nodes:
        return StructuralLabel(False, None, None, None, True, 0, "unknown logical node in state")

    frozen = {i: chains.get(i, frozenset()) for i in logical_nodes - movable_set}
    frozen_seen: set[Qubit] = set()
    for node, chain in frozen.items():
        if not chain or not set(chain) <= set(host.nodes()):
            return StructuralLabel(False, None, None, None, True, 0, "invalid frozen exterior")
        if len(chain) > 1 and not nx.is_connected(host.subgraph(chain)):
            return StructuralLabel(False, None, None, None, True, 0, "invalid frozen exterior")
        if frozen_seen.intersection(chain):
            return StructuralLabel(False, None, None, None, True, 0, "invalid frozen exterior overlap")
        frozen_seen.update(chain)
    blocked = frozenset(q for c in frozen.values() for q in c)
    options = {
        i: connected_subsets(host, domain.window, domain.max_chain, blocked)
        for i in domain.movable
    }
    if any(not choices for choices in options.values()):
        return StructuralLabel(False, None, None, None, True, 0, "no connected subset in the window")

    deadline = time.monotonic() + domain.timeout_ms / 1000.0
    best: tuple[int, int, dict[Node, frozenset[Qubit]]] | None = None
    explored = 0
    order = list(domain.movable)

    def demands_ok(assignment: Mapping[Node, frozenset[Qubit]], partial: bool) -> bool:
        merged = dict(frozen)
        merged.update(assignment)
        for u, v in logical.edges():
            cu, cv = merged.get(u), merged.get(v)
            if cu is None or cv is None:
                if partial:
                    continue
                return False
            if not any(a != b and host.has_edge(a, b) for a in cu for b in cv):
                return False
        return True

    def recurse(depth: int, assignment: dict[Node, frozenset[Qubit]], used: frozenset[Qubit]) -> bool:
        nonlocal best, explored
        if time.monotonic() > deadline:
            return True
        if depth == len(order):
            explored += 1
            if not demands_ok(assignment, partial=False):
                return False
            total = len(used) + len(blocked)
            if domain.qubit_cap is not None and total > domain.qubit_cap:
                return False
            longest = max((len(c) for c in {**frozen, **assignment}.values()), default=0)
            key = (total, longest)
            if best is None or key < (best[0], best[1]):
                best = (total, longest, dict(assignment))
            return False
        variable = order[depth]
        for choice in options[variable]:
            if choice & used:
                continue
            assignment[variable] = choice
            if demands_ok(assignment, partial=True):
                if recurse(depth + 1, assignment, used | choice):
                    return True
            del assignment[variable]
        return False

    timed_out = recurse(0, {}, frozenset())
    if best is not None:
        complete = dict(frozen)
        complete.update(best[2])
        # A positive witness remains valid on timeout, but provisional resource
        # minima are not exact until the registered continuation domain is exhausted.
        return StructuralLabel(
            True,
            complete,
            best[0] if not timed_out else None,
            best[1] if not timed_out else None,
            not timed_out,
            explored,
            "" if not timed_out else "positive witness found; resource minima unknown after timeout",
        )
    if timed_out:
        return StructuralLabel(None, None, None, None, False, explored, "timeout: unknown, not infeasible")
    return StructuralLabel(False, None, None, None, True, explored, "domain exhausted")


def tiny_conformance_cases(host: nx.Graph, seed: int = 0) -> list[dict[str, object]]:
    """Positive and negative cases small enough to check by hand (Rev2 slice 1)."""

    nodes = sorted(host.nodes(), key=str)[:6]
    sub = host.subgraph(nodes)
    logical = nx.Graph()
    logical.add_edges_from([(0, 1), (1, 2)])
    window = frozenset(nodes)
    cases = []
    for movable in ([0], [0, 2]):
        cases.append(
            {
                "logical": logical,
                "host": sub,
                "domain": ContinuationDomain(tuple(movable), window, max_chain=2, timeout_ms=1_000),
            }
        )
    return cases
