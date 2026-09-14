"""Ink-drop generation with an explicit, independently validated witness.

Spec: Rev2 section "Ink-drop generation with an explicit witness". Faults are applied first,
disjoint connected branch sets are grown from distinct roots under a registered mixture of
growth modes, and the quotient contact graph becomes the logical support. The witness proves
feasibility on that exact active host; it is not a quality-optimal target and never reaches
the actor.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Hashable, Mapping

import networkx as nx

Qubit = Hashable
Node = int

GROWTH_MODES: tuple[str, ...] = ("compact", "elongated", "bottleneck", "contact_seeking")


class InkDropError(RuntimeError):
    """The requested cell is impossible; never silently changed into another cell."""


@dataclass(frozen=True)
class InkDropResult:
    logical: nx.Graph
    witness: Mapping[Node, frozenset[Qubit]]
    quotient: nx.Graph
    host: nx.Graph
    modes: tuple[str, ...]
    budget: int
    receipt: Mapping[str, object] = field(default_factory=dict)

    @property
    def qubits(self) -> int:
        return sum(len(c) for c in self.witness.values())


def apply_faults(host: nx.Graph, fault_rate: float, seed: int) -> nx.Graph:
    """Faults precede growth, so the witness is certified on the realised active graph."""

    if not 0.0 <= fault_rate < 1.0:
        raise InkDropError("fault_rate must lie in [0, 1)")
    if fault_rate <= 0:
        return host.copy()
    rng = random.Random(seed)
    active = host.copy()
    dead = [q for q in host.nodes() if rng.random() < fault_rate]
    active.remove_nodes_from(dead)
    if active.number_of_nodes() == 0:
        raise InkDropError("fault realization removed every active qubit")
    largest = max(nx.connected_components(active), key=len)
    return active.subgraph(largest).copy()


def ink_drop(
    host: nx.Graph,
    n_variables: int,
    chain_size: int = 3,
    *,
    seed: int = 0,
    mode: str = "compact",
    density: float = 1.0,
    budget: int | None = None,
) -> InkDropResult:
    """Grow ``n_variables`` disjoint connected branch sets and take their quotient support."""

    if mode not in GROWTH_MODES:
        raise InkDropError(f"unregistered growth mode {mode!r}")
    if isinstance(n_variables, bool) or not isinstance(n_variables, int) or n_variables <= 0:
        raise InkDropError("n_variables must be a positive integer")
    if isinstance(chain_size, bool) or not isinstance(chain_size, int) or chain_size <= 0:
        raise InkDropError("chain_size must be a positive integer")
    if not 0.0 <= density <= 1.0:
        raise InkDropError("density must lie in [0, 1]")
    if budget is not None and (
        isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0
    ):
        raise InkDropError("budget must be a positive integer")
    rng = random.Random(seed)
    nodes = sorted(host.nodes(), key=str)
    if len(nodes) < n_variables * chain_size:
        raise InkDropError("active host is too small for the requested cell")

    roots = rng.sample(nodes, n_variables)
    chains: dict[Node, set[Qubit]] = {i: {r} for i, r in enumerate(roots)}
    claimed: set[Qubit] = set(roots)
    target = {i: chain_size for i in chains}
    if mode == "elongated":
        target = {i: chain_size + rng.randrange(0, 3) for i in chains}
    if mode == "bottleneck":
        target = {i: (1 if i % 3 == 0 else chain_size + 1) for i in chains}

    cap = budget if budget is not None else sum(target.values())
    if cap < n_variables:
        raise InkDropError("budget is smaller than the number of distinct roots")
    if cap < sum(target.values()):
        raise InkDropError("budget is smaller than the requested branch-set targets")
    if len(nodes) < sum(target.values()):
        raise InkDropError("active host is too small for the mode-specific target cell")
    growing = [i for i in chains if len(chains[i]) < target[i]]
    guard = 0
    while growing and len(claimed) < cap and guard < 10_000:
        guard += 1
        i = rng.choice(growing)
        frontier = [
            r
            for q in chains[i]
            for r in host.neighbors(q)
            if r not in claimed
        ]
        if not frontier:
            growing.remove(i)
            continue
        if mode == "contact_seeking":
            frontier.sort(
                key=lambda r: -sum(1 for s in host.neighbors(r) if s in claimed and s not in chains[i])
            )
            pick = frontier[0]
        elif mode == "compact":
            frontier.sort(key=lambda r: -sum(1 for s in host.neighbors(r) if s in chains[i]))
            pick = frontier[0]
        else:
            pick = rng.choice(frontier)
        chains[i].add(pick)
        claimed.add(pick)
        if len(chains[i]) >= target[i]:
            growing.remove(i)

    if growing or any(len(chains[i]) != target[i] for i in chains):
        raise InkDropError("growth could not realize every requested branch-set target")

    witness = {i: frozenset(c) for i, c in chains.items()}
    quotient = nx.Graph()
    quotient.add_nodes_from(witness)
    for i in witness:
        for j in witness:
            if i >= j:
                continue
            if any(host.has_edge(q, r) for q in witness[i] for r in witness[j]):
                quotient.add_edge(i, j)

    logical = nx.Graph()
    logical.add_nodes_from(quotient.nodes())
    edges = sorted(quotient.edges())
    rng.shuffle(edges)
    keep_count = int(round(density * len(edges)))
    if density > 0.0 and edges:
        keep_count = max(1, keep_count)
    keep = edges[:keep_count]
    logical.add_edges_from(keep)

    receipt = validate_witness(witness, logical, host)
    if not receipt["valid"]:
        raise InkDropError(f"generator produced an invalid witness: {receipt}")
    return InkDropResult(
        logical=logical,
        witness=witness,
        quotient=quotient,
        host=host,
        modes=(mode,),
        budget=cap,
        receipt=receipt,
    )


def validate_witness(
    witness: Mapping[Node, frozenset[Qubit]], logical: nx.Graph, host: nx.Graph
) -> dict[str, object]:
    """Independent check of every branch set, contact and the disjointness of the witness."""

    reasons: list[str] = []
    if set(witness) != set(logical.nodes()):
        reasons.append("witness keys do not exactly cover the logical graph")
    seen: set[Qubit] = set()
    for i, chain in witness.items():
        if not chain:
            reasons.append(f"empty branch set {i}")
            continue
        if not set(chain) <= set(host.nodes()):
            reasons.append(f"branch set {i} leaves the active host")
        if len(chain) > 1 and not nx.is_connected(host.subgraph(chain)):
            reasons.append(f"branch set {i} is disconnected")
        if seen & set(chain):
            reasons.append(f"branch set {i} overlaps another")
        seen |= set(chain)
    for u, v in logical.edges():
        if not any(host.has_edge(q, r) for q in witness[u] for r in witness[v]):
            reasons.append(f"demand {(u, v)} has no physical contact")
    return {
        "valid": not reasons,
        "reasons": reasons,
        "qubits": len(seen),
        "max_chain": max((len(c) for c in witness.values()), default=0),
    }
