"""Independent tiny brute-force oracle for structural conformance gates.

This module intentionally does not import the production branch-and-bound labeler or the
runtime return validator.  It enumerates the registered finite domain directly so that a
shared defect in those implementations cannot certify its own release gate.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Mapping, Protocol

import networkx as nx

Node = Hashable
Qubit = Hashable
STRUCTURAL_ORACLE_VERSION = "structural-bruteforce-v1"


class DomainLike(Protocol):
    movable: tuple[Node, ...]
    window: frozenset[Qubit]
    max_chain: int
    qubit_cap: int | None


@dataclass(frozen=True)
class BruteForceResult:
    feasible: bool
    assignments_checked: int


def implementation_identity() -> dict[str, object]:
    source = Path(__file__).read_bytes()
    return {
        "implementation": "isingfold.rl.data.structural_oracle.brute_force_feasibility",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "version": STRUCTURAL_ORACLE_VERSION,
    }


def _connected_nonempty_subsets(
    host: nx.Graph,
    window: frozenset[Qubit],
    blocked: frozenset[Qubit],
    max_chain: int,
) -> tuple[frozenset[Qubit], ...]:
    available = tuple(q for q in sorted(window, key=str) if q not in blocked)
    choices: list[frozenset[Qubit]] = []
    for size in range(1, min(max_chain, len(available)) + 1):
        for raw in itertools.combinations(available, size):
            candidate = frozenset(raw)
            if size == 1 or nx.is_connected(host.subgraph(candidate)):
                choices.append(candidate)
    return tuple(choices)


def brute_force_feasibility(
    chains: Mapping[Node, frozenset[Qubit]],
    logical: nx.Graph,
    host: nx.Graph,
    domain: DomainLike,
) -> BruteForceResult:
    """Enumerate the finite continuation domain without production-labeler helpers."""

    logical_nodes = set(logical.nodes())
    movable = tuple(domain.movable)
    movable_set = set(movable)
    host_nodes = set(host.nodes())
    if (
        len(movable_set) != len(movable)
        or not movable_set <= logical_nodes
        or domain.max_chain <= 0
        or not set(domain.window) <= host_nodes
        or (domain.qubit_cap is not None and domain.qubit_cap <= 0)
    ):
        return BruteForceResult(False, 0)

    frozen = {node: chains.get(node, frozenset()) for node in logical_nodes - movable_set}
    used: set[Qubit] = set()
    for chain in frozen.values():
        if (
            not chain
            or not set(chain) <= host_nodes
            or (len(chain) > 1 and not nx.is_connected(host.subgraph(chain)))
            or bool(used.intersection(chain))
        ):
            return BruteForceResult(False, 0)
        used.update(chain)
    blocked = frozenset(used)
    options = tuple(
        _connected_nonempty_subsets(host, domain.window, blocked, domain.max_chain)
        for _node in movable
    )
    if any(not choices for choices in options):
        return BruteForceResult(False, 0)

    checked = 0
    for selected in itertools.product(*options):
        checked += 1
        occupied = set(blocked)
        overlaps = False
        for chain in selected:
            if occupied.intersection(chain):
                overlaps = True
                break
            occupied.update(chain)
        if overlaps:
            continue
        assignment = dict(frozen)
        assignment.update(zip(movable, selected, strict=True))
        if domain.qubit_cap is not None and len(occupied) > domain.qubit_cap:
            continue
        if all(
            any(q != r and host.has_edge(q, r) for q in assignment[left] for r in assignment[right])
            for left, right in logical.edges()
        ):
            return BruteForceResult(True, checked)
    return BruteForceResult(False, checked)


__all__ = [
    "BruteForceResult",
    "STRUCTURAL_ORACLE_VERSION",
    "brute_force_feasibility",
    "implementation_identity",
]
