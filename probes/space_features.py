"""Where is there room to grow, and in which direction?

Sections 7.2 to 7.4 of the meeting design. The search representation already carries occupancy, a
free-neighbour ratio, a nearest-free distance and a two-hop count. What it does not carry is the
thing a placement decision actually needs: how much free space is reachable if you step this way
rather than that way, on the graph as it is after occupancy and faults, not on the pristine host.

Three definitions, kept separate because they answer different questions.

`residual_graph` is the graph a given operation is allowed to move on: free qubits, plus the
qubits of the chains it is permitted to release, with every other chain left standing as an
obstacle and faulty qubits absent. Reachability computed on the pristine host and then corrected
for occupancy is not the same quantity and is usually optimistic.

`free_volume` is the number of free qubits within k hops of an anchor on that residual graph,
for k = 1, 2, 3, together with the size of the free component the anchor touches. Components are
unioned by node identity, so a component reachable through three different neighbours is counted
once rather than three times.

`directional_capacity` is the design's first-edge definition. For a directed first step q -> r
with r admissible and free,

    A_k(q -> r) = |{ v free : d(r, v) <= k - 1 on the residual graph with q removed }|,

the free qubits reachable if you commit to leaving through r and do not turn back. Different
directions overlap, so these are never summed: a candidate feature takes the minimum, the mean,
the maximum and the count of dead directions, which is what a choice between directions needs.

Every traversal is bounded, and a truncated search returns a flag rather than a zero, because a
bounded walk that ran out of budget has not shown that the rest of the space is empty.
"""
from __future__ import annotations

from collections import deque


def residual_graph(host, chains, releasable=(), faults=()):
    """The nodes an operation may occupy or pass through, and the obstacles it may not."""
    occupied = {q for node, chain in chains.items() if node not in set(releasable)
                for q in chain}
    blocked = occupied | set(faults)
    return {q for q in host.nodes() if q not in blocked}


def _bounded_bfs(host, allowed, start, radius, cap):
    """Free nodes within `radius` hops, plus a flag saying whether the cap cut the search."""
    seen = {start}
    frontier = deque([(start, 0)])
    reached, visited = set(), 0
    truncated = False
    while frontier:
        q, d = frontier.popleft()
        visited += 1
        if visited > cap:
            truncated = True
            break
        if d >= radius:
            continue
        for r in host.neighbors(q):
            if r in seen or r not in allowed:
                continue
            seen.add(r)
            reached.add(r)
            frontier.append((r, d + 1))
    return reached, truncated


def free_volume(host, allowed, anchor, radii=(1, 2, 3), cap=4096):
    """Reachable free volume at several radii, and the component the anchor touches."""
    out, truncated = [], False
    for k in radii:
        reached, tr = _bounded_bfs(host, allowed, anchor, k, cap)
        out.append(float(len(reached)))
        truncated = truncated or tr
    component, tr = _bounded_bfs(host, allowed, anchor, 10 ** 6, cap)
    out.append(float(len(component)))
    out.append(1.0 if (truncated or tr) else 0.0)
    return out


def directional_capacity(host, allowed, anchor, radius=3, cap=2048):
    """One number per direction out of the anchor, then the summary a choice needs.

    The per-direction values are never added together: two directions out of one qubit reach
    overlapping space, and summing them would report capacity the hardware does not have.
    """
    exits = [r for r in host.neighbors(anchor) if r in allowed]
    if not exits:
        return [0.0, 0.0, 0.0, 0.0, float(len(list(host.neighbors(anchor))))]
    reduced = allowed - {anchor}
    volumes = []
    for r in exits:
        reached, _ = _bounded_bfs(host, reduced, r, radius - 1, cap)
        volumes.append(float(len(reached) + 1))
    dead = sum(1 for v in volumes if v <= 1.0)
    return [min(volumes), sum(volumes) / len(volumes), max(volumes), float(dead),
            float(len(list(host.neighbors(anchor))) - len(exits))]


def chain_space_features(host, chains, node, faults=(), radius=3):
    """The space features for one chain, aggregated over its qubits.

    A chain is grown from whichever of its qubits has room, so the maximum over its members is the
    quantity that decides whether it can grow at all, while the minimum says how boxed in its
    worst attachment point is. Both are reported rather than an average, which would hide both.
    """
    allowed = residual_graph(host, chains, releasable=(node,), faults=faults)
    members = sorted(chains[node], key=str)
    vols, dirs = [], []
    for q in members:
        vols.append(free_volume(host, allowed | {q}, q, radii=(1, 2, radius)))
        dirs.append(directional_capacity(host, allowed | {q}, q, radius=radius))
    n = len(members)
    best = [max(v[i] for v in vols) for i in range(3)]
    worst_component = min(v[3] for v in vols)
    truncated = max(v[4] for v in vols)
    return [
        best[0], best[1], best[2], worst_component, truncated,
        max(d[2] for d in dirs),                    # best single direction out of this chain
        min(d[0] for d in dirs),                    # the most boxed-in attachment point
        sum(d[3] for d in dirs) / n,                # dead directions per qubit
        sum(d[4] for d in dirs) / n,                # blocked neighbours per qubit
    ]


SPACE_WIDTH = 9
"""Channels chain_space_features returns."""
