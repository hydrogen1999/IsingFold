"""The occupancy-weighted router that materialises branch sets.

This is the classical inner loop the environment owns: a Dijkstra expansion from each
neighbour's branch set with a cost that grows with how many chains already claim a qubit,
then the cheapest root. Temporary overlap is what makes repair possible, so occupied qubits
are expensive rather than forbidden (Rev2 section 3.3). Randomised edge-cost perturbations
and root diversity give proposal diversity; none of it reads an outcome.
"""

from __future__ import annotations

import heapq
import random
from dataclasses import dataclass
from collections import deque
from typing import Hashable, Iterable, Mapping, Sequence

import networkx as nx

Qubit = Hashable
Node = Hashable


@dataclass(frozen=True)
class RouteResult:
    chain: frozenset[Qubit]
    paths: tuple[tuple[Qubit, ...], ...]
    cost: float
    expansions: int


@dataclass(frozen=True)
class RouteAttempt:
    """One completed routing attempt, including work consumed before failure."""

    result: RouteResult | None
    expansions: int


@dataclass(frozen=True)
class RebuildAttempt:
    """One group-rebuild attempt with an exact aggregate search charge."""

    placed: Mapping[Node, frozenset[Qubit]] | None
    expansions: int


@dataclass(frozen=True)
class PathAttempt:
    """A metered shortest-path attempt between two endpoint sets."""

    path: tuple[Qubit, ...] | None
    expansions: int


def qubit_cost(occ: int, overfill: float) -> float:
    """A free qubit costs one; each additional claimant multiplies the price."""

    return float(overfill ** max(0, occ))


def _dijkstra(
    host: nx.Graph,
    sources: Iterable[Qubit],
    occupancy: Mapping[Qubit, int],
    overfill: float,
    jitter: Mapping[Qubit, float] | None,
    budget: int,
    blocked: frozenset[Qubit] = frozenset(),
    node_costs: Mapping[Qubit, float] | None = None,
) -> tuple[dict[Qubit, float], dict[Qubit, Qubit | None], int]:
    costs = (
        node_costs
        if node_costs is not None
        else {
            q: qubit_cost(occupancy.get(q, 0), overfill) * (jitter.get(q, 1.0) if jitter else 1.0)
            for q in host.nodes
        }
    )
    dist: dict[Qubit, float] = {}
    tentative: dict[Qubit, float] = {}
    prev: dict[Qubit, Qubit | None] = {}
    heap: list[tuple[float, int, Qubit]] = []
    tie = 0
    for s in sources:
        if s not in host:
            continue
        c = costs[s]
        if c >= tentative.get(s, float("inf")):
            continue
        tentative[s] = c
        heapq.heappush(heap, (c, tie, s))
        tie += 1
    expansions = 0
    while heap and expansions < budget:
        d, _, q = heapq.heappop(heap)
        if q in dist and d >= dist[q]:
            continue
        dist[q] = d
        expansions += 1
        for r in host.neighbors(q):
            if r in blocked or r in dist:
                continue
            nd = d + costs[r]
            if nd < tentative.get(r, float("inf")):
                tentative[r] = nd
                prev[r] = q
                tie += 1
                heapq.heappush(heap, (nd, tie, r))
    return dist, prev, expansions


def _path_to(
    prev: Mapping[Qubit, Qubit | None], target: Qubit, sources: set[Qubit]
) -> tuple[Qubit, ...]:
    path = [target]
    while path[-1] not in sources:
        nxt = prev.get(path[-1])
        if nxt is None:
            break
        path.append(nxt)
    return tuple(reversed(path))


def route_variable(
    host: nx.Graph,
    neighbour_chains: Sequence[frozenset[Qubit]],
    occupancy: Mapping[Qubit, int],
    *,
    overfill: float = 4.0,
    rng: random.Random | None = None,
    jitter_scale: float = 0.0,
    root_rank: int = 0,
    forbidden: frozenset[Qubit] = frozenset(),
    exclusive: bool = False,
    expansion_budget: int = 20_000,
) -> RouteAttempt:
    """Connect one variable to every already-placed neighbour; return its branch set.

    ``root_rank`` picks the k-th cheapest root, which is how the generator produces genuinely
    different routes instead of many copies of one shortest path.
    """

    jitter = None
    if jitter_scale > 0 and rng is not None:
        jitter = {q: 1.0 + jitter_scale * rng.random() for q in host.nodes}
    occ = dict(occupancy)
    # A routed chain never claims a qubit of the very chains it connects to: those belong to
    # other variables. Overlap with third-party chains stays legal but priced.
    owned_elsewhere = frozenset().union(*neighbour_chains) if neighbour_chains else frozenset()
    blocked = frozenset(forbidden) | owned_elsewhere
    if exclusive:
        blocked = blocked | frozenset(q for q, o in occ.items() if o > 0)

    if not neighbour_chains:
        free = [q for q in host.nodes if occ.get(q, 0) == 0 and q not in forbidden]
        if not free:
            return RouteAttempt(None, 0)
        pick = (rng or random).choice(free) if rng else free[0]
        result = RouteResult(frozenset({pick}), ((pick,),), qubit_cost(0, overfill), 1)
        return RouteAttempt(result, result.expansions)

    node_costs = {
        q: qubit_cost(occ.get(q, 0), overfill) * (jitter.get(q, 1.0) if jitter else 1.0)
        for q in host.nodes
    }

    totals: dict[Qubit, float] = {}
    prevs: list[dict[Qubit, Qubit | None]] = []
    dists: list[dict[Qubit, float]] = []
    expansions = 0
    for chain in neighbour_chains:
        # ``expansion_budget`` is a total across the neighbour loop, not a per-call bound:
        # otherwise a high-degree variable silently spends degree times the allowance.
        remaining_budget = expansion_budget - expansions
        if remaining_budget <= 0:
            return RouteAttempt(None, expansions)
        dist, prev, used = _dijkstra(
            host,
            chain,
            occ,
            overfill,
            jitter,
            remaining_budget,
            blocked - frozenset(chain),
            node_costs,
        )
        expansions += used
        dists.append(dist)
        prevs.append(prev)
        for q, d in dist.items():
            totals[q] = totals.get(q, 0.0) + d
    reachable = [q for q in totals if all(q in d for d in dists) and q not in blocked]
    if not reachable:
        return RouteAttempt(None, expansions)
    reachable.sort(key=lambda q: (totals[q], str(q)))
    root = reachable[min(root_rank, len(reachable) - 1)]

    chain: set[Qubit] = {root}
    paths: list[tuple[Qubit, ...]] = []
    for prev, source in zip(prevs, neighbour_chains, strict=True):
        path = _path_to(prev, root, set(source))
        paths.append(path)
        chain.update(path[1:] if path and path[0] in source else path)
    if not chain:
        return RouteAttempt(None, expansions)
    sub = host.subgraph(chain)
    if not nx.is_connected(sub):
        return RouteAttempt(None, expansions)
    result = RouteResult(frozenset(chain), tuple(paths), totals[root], expansions)
    return RouteAttempt(result, expansions)


def route_between_sets(
    host: nx.Graph,
    sources: Iterable[Qubit],
    targets: Iterable[Qubit],
    *,
    expansion_budget: int,
) -> PathAttempt:
    """Find one deterministic unweighted route and count every popped vertex.

    This replaces unmetered all-pairs ``networkx.shortest_path`` calls in the ROUTE family.
    The first target reached by a multi-source BFS is a shortest endpoint-to-endpoint route;
    lexical source/neighbor order makes exact replay independent of graph insertion order.
    """

    if isinstance(expansion_budget, bool) or not isinstance(expansion_budget, int):
        raise TypeError("expansion_budget must be an integer")
    if expansion_budget < 0:
        raise ValueError("expansion_budget must be nonnegative")
    target_set = {target for target in targets if target in host}
    source_list = sorted({source for source in sources if source in host}, key=str)
    if not target_set or not source_list or expansion_budget == 0:
        return PathAttempt(None, 0)
    queue: deque[Qubit] = deque(source_list)
    previous: dict[Qubit, Qubit | None] = {source: None for source in source_list}
    expansions = 0
    while queue and expansions < expansion_budget:
        current = queue.popleft()
        expansions += 1
        if current in target_set:
            path = [current]
            while previous[path[-1]] is not None:
                path.append(previous[path[-1]])
            return PathAttempt(tuple(reversed(path)), expansions)
        for neighbour in sorted(host.neighbors(current), key=str):
            if neighbour not in previous:
                previous[neighbour] = current
                queue.append(neighbour)
    return PathAttempt(None, expansions)


def rebuild_group(
    host: nx.Graph,
    chains: Mapping[Node, frozenset[Qubit]],
    logical: nx.Graph,
    group: Sequence[Node],
    *,
    rng: random.Random,
    overfill: float = 4.0,
    jitter_scale: float = 0.15,
    root_rank: int = 0,
    order: Sequence[Node] | None = None,
    exclusive: bool = False,
    expansion_budget: int = 20_000,
) -> RebuildAttempt:
    """Erase a group of chains and route them back one by one against the rest.

    Returns the complete replacement for the group, which may claim qubits still held by
    other chains: that temporary overlap is legal inside the envelope and is precisely what
    a later repair resolves.
    """

    kept = {i: c for i, c in chains.items() if i not in set(group)}
    occ: dict[Qubit, int] = {}
    for c in kept.values():
        for q in c:
            occ[q] = occ.get(q, 0) + 1

    sequence = list(order) if order is not None else list(group)
    if order is None:
        rng.shuffle(sequence)
    placed: dict[Node, frozenset[Qubit]] = {}
    expansions = 0
    budget_left = expansion_budget
    for i in sequence:
        if budget_left <= 0:
            return RebuildAttempt(None, expansions)
        neighbours = [
            placed.get(u, kept.get(u, frozenset()))
            for u in logical.neighbors(i)
            if placed.get(u) or kept.get(u)
        ]
        attempt = route_variable(
            host,
            neighbours,
            occ,
            overfill=overfill,
            rng=rng,
            jitter_scale=jitter_scale,
            root_rank=root_rank,
            exclusive=exclusive,
            expansion_budget=budget_left,
        )
        expansions += attempt.expansions
        budget_left -= attempt.expansions
        result = attempt.result
        if result is None:
            return RebuildAttempt(None, expansions)
        placed[i] = result.chain
        for q in result.chain:
            occ[q] = occ.get(q, 0) + 1
    return RebuildAttempt(placed, expansions)
