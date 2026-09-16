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


# ---------------------------------------------------------------------------------------
# Negotiated-congestion completion (rip up and reroute), the completion the embedder uses.
# ---------------------------------------------------------------------------------------
import heapq


def _dijkstra_route(host, sources, targets, cost, limit_cost=1e9):
    """Cheapest path from any source to any target; returns the interior qubits."""
    dist = {q: 0.0 for q in sources}
    prev = {q: None for q in sources}
    heap = [(0.0, str(q), q) for q in sources]
    heapq.heapify(heap)
    while heap:
        d, _, q = heapq.heappop(heap)
        if d > dist.get(q, 1e18):
            continue
        if q in targets and q not in sources:
            path = []
            x = prev[q]
            while x is not None and x not in sources:
                path.append(x)
                x = prev[x]
            return list(reversed(path)), d
        if d > limit_cost:
            return None, d
        for r in host.neighbors(q):
            c = 0.0 if r in targets else cost(r)
            nd = d + c
            if nd < dist.get(r, 1e18):
                dist[r] = nd
                prev[r] = q
                heapq.heappush(heap, (nd, str(r), r))
    return None, float("inf")


def negotiate(host, logical, roots, budget, deadline=10.0, max_rounds=200, pressure=2.0,
              history_step=1.0, rng=None):
    """Complete an embedding from roots by negotiated congestion.

    Every unmet demand is routed by the cheapest path on the host where a free qubit costs
    one and a qubit already held by another chain costs more, rising each round it stays
    contested. Overlaps are allowed while routing and removed by rerouting the demands that
    caused them, until no qubit is shared, the qubit budget is respected, or the deadline
    passes. Chains are the root plus every interior segment routed for them. This is the
    scheme behind minorminer and PathFinder, written for the case where the roots are
    given, which is what the learned part supplies.
    """
    t0 = time.time()
    root_of = {v: next(iter(c)) for v, c in roots.items()}
    for v in logical.nodes():
        if v not in root_of:
            return None
    demands = [(u, v) for u, v in logical.edges()]
    jitter = {}
    if rng is not None:
        # random demand order and a small random cost per qubit break the ties that make
        # two contested routes trade places round after round
        rng.shuffle(demands)
        jitter = {q: float(rng.random()) * 0.3 for q in host.nodes()}
    seg = {}                      # demand -> (owner, [qubits])
    history = {}                  # qubit -> accumulated congestion history

    def chains_now():
        ch = {v: {root_of[v]} for v in logical.nodes()}
        for key, (owner, qs) in seg.items():
            ch[owner].update(qs)
        return ch

    def owners_of(ch):
        own = {}
        for v, qs in ch.items():
            for q in qs:
                own.setdefault(q, set()).add(v)
        return own

    def route_one(u, v, ch, own):
        """Route demand (u, v) against the current occupancy; interior split between them."""
        cu, cv = ch[u], ch[v]

        def cost(r):
            holders = own.get(r, set()) - {u, v}
            return 1.0 + jitter.get(r, 0.0) + pressure * len(holders) + history.get(r, 0.0)

        path, _ = _dijkstra_route(host, cu, cv, cost)
        if path is None:
            return False
        k = (len(path) + 1) // 2
        seg[(u, v)] = (u, path[:k])
        if path[k:]:
            seg[(u, v, "tail")] = (v, path[k:])
        else:
            seg.pop((u, v, "tail"), None)
        for q in path[:k]:
            own.setdefault(q, set()).add(u); ch[u].add(q)
        for q in path[k:]:
            own.setdefault(q, set()).add(v); ch[v].add(q)
        return True

    def drop(u, v, ch, own):
        for key in ((u, v), (u, v, "tail")):
            if key in seg:
                owner, qs = seg.pop(key)
                for q in qs:
                    ch[owner].discard(q)
                    s = own.get(q)
                    if s:
                        s.discard(owner)
                        if not s:
                            del own[q]

    def cascade(ch, own):
        """Drop every segment no longer connected to its owner's root. A segment routed
        from another segment's qubit loses its footing when that one is ripped up, and a
        chain must stay connected at all times."""
        changed = True
        while changed:
            changed = False
            for var in list(ch):
                comp = set()
                frontier = [root_of[var]]
                comp.add(root_of[var])
                while frontier:
                    q = frontier.pop()
                    for r in host.neighbors(q):
                        if r in ch[var] and r not in comp:
                            comp.add(r); frontier.append(r)
                loose = ch[var] - comp
                if loose:
                    for key, (owner, qs) in list(seg.items()):
                        if owner == var and any(q in loose for q in qs):
                            drop(key[0], key[1], ch, own)
                            changed = True

    rounds = 0
    ch = chains_now()
    own = owners_of(ch)
    while time.time() - t0 < deadline and rounds < max_rounds:
        # route every unmet demand against the current occupancy, overlaps allowed
        unmet = [(u, v) for (u, v) in demands if not _touch(host, ch[u], ch[v])]
        for (u, v) in unmet:
            if _touch(host, ch[u], ch[v]):
                continue
            if not route_one(u, v, ch, own):
                negotiate.last = {"rounds": rounds, "secs": time.time() - t0, "why": "unroutable"}
                return None
        shared = {q for q, s in own.items() if len(s) > 1}
        if not shared:
            total = sum(len(c) for c in ch.values())
            if total <= budget:
                negotiate.last = {"rounds": rounds, "secs": time.time() - t0, "why": "done"}
                return {v: frozenset(c) for v, c in ch.items()}
            longest = sorted(((k, o, qs) for k, (o, qs) in seg.items()), key=lambda t: -len(t[2]))
            for key, _, _ in longest[: max(1, len(longest) // 8)]:
                drop(key[0], key[1], ch, own)
            cascade(ch, own)
            pressure *= 1.5
            rounds += 1
            continue
        for q in shared:
            history[q] = history.get(q, 0.0) + history_step
        conflicted = []
        for key, (owner, qs) in list(seg.items()):
            if any(q in shared for q in qs) and key[:2] not in conflicted:
                conflicted.append(key[:2])
        for q in shared:
            for var in list(own.get(q, ())):
                if root_of[var] == q:
                    for key, (owner, qs) in list(seg.items()):
                        if q in qs and key[:2] not in conflicted:
                            conflicted.append(key[:2])
        for (u, v) in conflicted:
            drop(u, v, ch, own)
        cascade(ch, own)
        rounds += 1
    negotiate.last = {"rounds": rounds, "secs": time.time() - t0}
    return None


def negotiate_restarts(host, logical, roots, budget, deadline=10.0, seed=0):
    """negotiate() with randomised restarts until the deadline; the first success wins."""
    import random
    t0 = time.time()
    rng = random.Random(seed)
    attempt = 0
    while time.time() - t0 < deadline:
        attempt += 1
        left = deadline - (time.time() - t0)
        res = negotiate(host, logical, roots, budget, deadline=left, max_rounds=400,
                        pressure=2.0 + rng.random(), history_step=0.5 + rng.random(), rng=rng)
        if res is not None:
            negotiate_restarts.last = {"attempts": attempt, "secs": time.time() - t0}
            return res
    negotiate_restarts.last = {"attempts": attempt, "secs": time.time() - t0}
    return None


def matched_completion(host, logical, roots, budget, deadline=10.0, seed=0):
    """Bridge demands by maximum matching first, then negotiate the rest.

    At high fill with short chains most unmet demands can be met by one free qubit adjacent
    to both roots, and the free qubits are scarce, so which demand gets which bridge is an
    assignment problem. A maximum bipartite matching (demand to bridge qubit) settles it
    exactly in milliseconds; the negotiated router then only handles the demands that no
    single bridge can serve, with the matched bridges in place.
    """
    import networkx as nx
    from networkx.algorithms import bipartite

    t0 = time.time()
    root_of = {v: next(iter(c)) for v, c in roots.items()}
    if any(v not in root_of for v in logical.nodes()):
        return None
    chains = {v: {root_of[v]} for v in logical.nodes()}
    occupied = set(root_of.values())
    unmet = [(u, v) for u, v in logical.edges() if not _touch(host, chains[u], chains[v])]
    # candidate bridges per demand
    g = nx.Graph()
    demand_nodes = []
    for (u, v) in unmet:
        fu = _neighbors_free(host, chains[u], occupied)
        fv = _neighbors_free(host, chains[v], occupied)
        node = ("d", u, v)
        g.add_node(node, bipartite=0)
        demand_nodes.append(node)
        for r in fu & fv:
            g.add_node(("q", r), bipartite=1)
            g.add_edge(node, ("q", r))
    matching = bipartite.hopcroft_karp_matching(g, top_nodes=demand_nodes) if g.number_of_edges() else {}
    seeded = {v: set(c) for v, c in chains.items()}
    for node in demand_nodes:
        q = matching.get(node)
        if q is None:
            continue
        _, u, v = node
        r = q[1]
        owner = u if len(seeded[u]) <= len(seeded[v]) else v
        seeded[owner].add(r)
        occupied.add(r)
    # every chain is still root plus adjacent bridges: connected by construction
    rest = [(u, v) for u, v in logical.edges() if not _touch(host, seeded[u], seeded[v])]
    if not rest:
        total = sum(len(c) for c in seeded.values())
        matched_completion.last = {"matched": len(matching) // 2, "rest": 0, "secs": time.time() - t0}
        return {v: frozenset(c) for v, c in seeded.items()} if total <= budget else None
    # negotiate the remaining demands with the bridged chains as the starting chains:
    # negotiate() takes roots, so pass the seeded chains as multi-qubit "roots" by giving
    # it a logical view where each seeded chain is contracted to its root and the extra
    # qubits are pre-occupied. Simplest faithful way: run negotiate on the original roots
    # but with the matched bridges fixed through a wrapper host cost. Here we fall back to
    # negotiate on the seeded chains directly by treating each chain as fixed occupancy.
    res = negotiate_from_chains(host, logical, seeded, budget, deadline - (time.time() - t0), seed)
    matched_completion.last = {"matched": len(matching) // 2, "rest": len(rest), "secs": time.time() - t0}
    return res


def negotiate_from_chains(host, logical, chains, budget, deadline=10.0, seed=0):
    """negotiate_restarts() generalised to start from connected multi-qubit chains."""
    import random
    t0 = time.time()
    rng = random.Random(seed)
    attempt = 0
    while time.time() - t0 < deadline:
        attempt += 1
        left = deadline - (time.time() - t0)
        res = _negotiate_chains(host, logical, chains, budget, left, rng)
        if res is not None:
            return res
    return None


def _negotiate_chains(host, logical, start, budget, deadline, rng, max_rounds=400):
    t0 = time.time()
    pressure = 2.0 + rng.random()
    history_step = 0.5 + rng.random()
    base = {v: set(c) for v, c in start.items()}
    demands = [(u, v) for u, v in logical.edges()]
    rng.shuffle(demands)
    jitter = {q: rng.random() * 0.3 for q in host.nodes()}
    seg = {}
    history = {}
    ch = {v: set(c) for v, c in base.items()}
    own = {}
    for v, qs in ch.items():
        for q in qs:
            own.setdefault(q, set()).add(v)

    def route_one(u, v):
        def cost(r):
            holders = own.get(r, set()) - {u, v}
            return 1.0 + jitter.get(r, 0.0) + pressure * len(holders) + history.get(r, 0.0)
        path, _ = _dijkstra_route(host, ch[u], ch[v], cost)
        if path is None:
            return False
        k = (len(path) + 1) // 2
        seg[(u, v)] = (u, path[:k]); seg[(u, v, "tail")] = (v, path[k:])
        for q in path[:k]:
            own.setdefault(q, set()).add(u); ch[u].add(q)
        for q in path[k:]:
            own.setdefault(q, set()).add(v); ch[v].add(q)
        return True

    def drop(u, v):
        for key in ((u, v), (u, v, "tail")):
            if key in seg:
                owner, qs = seg.pop(key)
                for q in qs:
                    ch[owner].discard(q)
                    s = own.get(q)
                    if s:
                        s.discard(owner)
                        if not s:
                            del own[q]

    def cascade():
        changed = True
        while changed:
            changed = False
            for var in list(ch):
                comp, frontier = set(base[var]), list(base[var])
                while frontier:
                    q = frontier.pop()
                    for r in host.neighbors(q):
                        if r in ch[var] and r not in comp:
                            comp.add(r); frontier.append(r)
                loose = ch[var] - comp
                if loose:
                    for key, (owner, qs) in list(seg.items()):
                        if owner == var and any(q in loose for q in qs):
                            drop(key[0], key[1]); changed = True

    rounds = 0
    while time.time() - t0 < deadline and rounds < max_rounds:
        for (u, v) in demands:
            if not _touch(host, ch[u], ch[v]) and not route_one(u, v):
                return None
        shared = {q for q, s in own.items() if len(s) > 1}
        if not shared:
            total = sum(len(c) for c in ch.values())
            if total <= budget:
                return {v: frozenset(c) for v, c in ch.items()}
            longest = sorted(((k, o, qs) for k, (o, qs) in seg.items()), key=lambda t: -len(t[2]))
            for key, _, _ in longest[: max(1, len(longest) // 8)]:
                drop(key[0], key[1])
            cascade(); pressure *= 1.5; rounds += 1
            continue
        for q in shared:
            history[q] = history.get(q, 0.0) + history_step
        conflicted = []
        for key, (owner, qs) in list(seg.items()):
            if any(q in shared for q in qs) and key[:2] not in conflicted:
                conflicted.append(key[:2])
        for q in shared:
            for var in list(own.get(q, ())):
                if q in base[var]:
                    for key, (owner, qs) in list(seg.items()):
                        if q in qs and key[:2] not in conflicted:
                            conflicted.append(key[:2])
        for (u, v) in conflicted:
            drop(u, v)
        cascade(); rounds += 1
    return None
