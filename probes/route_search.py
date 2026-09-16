"""Our completion search: roots fixed, connect every demand, backtrack when stuck.

The learned part of the embedder places one root per variable; this finishes the embedding
without minorminer. It is a constraint search over the demands, most constrained first:
for a demand between chains A and B the options, in order of cost, are a free qubit adjacent
to both (a bridge), two adjacent free qubits one touching each (a meet), and the interior
of a shortest path on the residual host, assigned to the owner with the shorter chain.
Depth-first with chronological backtracking and a bounded number of options per demand,
under the qubit budget and a deadline. Everything runs on plain chain dictionaries; a BFS
on a few hundred qubits costs milliseconds, so thousands of backtracks fit in a second.
"""
import time
from collections import deque


def _neighbors_free(host, chain, occupied):
    out = set()
    for q in chain:
        for r in host.neighbors(q):
            if r not in occupied:
                out.add(r)
    return out


def _touch(host, a, b):
    return any(host.has_edge(x, y) for x in a for y in b)


def _shortest_interior(host, src, dst, occupied, limit=6):
    """Interior qubits of a shortest path from a qubit of src to a qubit of dst through free
    qubits only, or None. Multi-source BFS, bounded depth."""
    prev = {q: None for q in src}
    frontier = deque(src)
    depth = {q: 0 for q in src}
    while frontier:
        q = frontier.popleft()
        if depth[q] >= limit:
            continue
        for r in host.neighbors(q):
            if r in prev:
                continue
            if r in dst:
                path = []
                x = q
                while x is not None and x not in src:
                    path.append(x)
                    x = prev[x]
                return list(reversed(path))
            if r in occupied:
                continue
            prev[r] = q
            depth[r] = depth[q] + 1
            frontier.append(r)
    return None


def options(host, chains, occupied, u, v, max_options):
    """Candidate additions realising demand (u, v): list of dicts {variable: added qubits}."""
    cu, cv = chains[u], chains[v]
    fu = _neighbors_free(host, cu, occupied)
    fv = _neighbors_free(host, cv, occupied)
    out, seen = [], set()

    def add(option):
        key = tuple(sorted((str(var), tuple(sorted(qs, key=str))) for var, qs in option.items()))
        if key not in seen:
            seen.add(key)
            out.append(option)
        return len(out) >= max_options

    # A bridge may go to either chain: which one keeps it decides what the other chain can
    # still reach for its remaining demands, so both are options, shorter owner first.
    first, second = (u, v) if len(cu) <= len(cv) else (v, u)
    for r in sorted(fu & fv, key=str):
        if add({first: {r}}) or add({second: {r}}):
            return out
    for x in sorted(fu, key=str):
        for y in sorted(host.neighbors(x), key=str):
            if y in fv and y != x and y not in cu and y not in cv:
                if add({u: {x}, v: {y}}):
                    return out
    interior = _shortest_interior(host, cu, cv, occupied)
    if interior:
        add({first: set(interior)})
    return out


def complete(host, logical, roots, budget, deadline=10.0, max_options=4, max_backtracks=20000):
    """Return chains realising every demand from the given roots, or None."""
    chains = {v: set(c) for v, c in roots.items()}
    for v in logical.nodes():
        chains.setdefault(v, set())
    if any(not c for c in chains.values()):
        return None
    occupied = {q for c in chains.values() for q in c}
    if len(occupied) != sum(len(c) for c in chains.values()):
        return None
    t0 = time.time()
    demands = [(u, v) for u, v in logical.edges() if not _touch(host, chains[u], chains[v])]
    used = [len(occupied)]
    backtracks = [0]

    def order(ds):
        # most constrained first: fewest free neighbours around the two chains
        return sorted(ds, key=lambda d: (len(_neighbors_free(host, chains[d[0]], occupied))
                                         + len(_neighbors_free(host, chains[d[1]], occupied)), str(d)))

    def apply(add):
        for var, qs in add.items():
            chains[var] |= qs
            occupied.update(qs)
        used[0] += sum(len(qs) for qs in add.values())

    def undo(add):
        for var, qs in add.items():
            chains[var] -= qs
            occupied.difference_update(qs)
        used[0] -= sum(len(qs) for qs in add.values())

    def solve(remaining):
        if time.time() - t0 > deadline or backtracks[0] > max_backtracks:
            return False
        remaining = [d for d in remaining if not _touch(host, chains[d[0]], chains[d[1]])]
        if not remaining:
            return True
        u, v = order(remaining)[0]
        rest = [d for d in remaining if d != (u, v)]
        for add in options(host, chains, occupied, u, v, max_options):
            cost = sum(len(qs) for qs in add.values())
            if used[0] + cost > budget:
                continue
            apply(add)
            if solve(rest):
                return True
            undo(add)
            backtracks[0] += 1
            if time.time() - t0 > deadline or backtracks[0] > max_backtracks:
                return False
        return False

    import sys
    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    ok = solve(demands)
    complete.last = {"backtracks": backtracks[0], "secs": time.time() - t0, "demands": len(demands)}
    return {v: frozenset(c) for v, c in chains.items()} if ok else None
