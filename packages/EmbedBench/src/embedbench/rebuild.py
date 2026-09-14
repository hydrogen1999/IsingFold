"""Rebuild seam: a learned (or rule) chooser rebuilds one chain at a time inside a valid
minorminer embedding. This is the end-to-end test that matches the training distribution:
frozen context = the other chains, in-play = the focus variable and one logical neighbour,
window = their old qubits plus the nearest free ones, candidates = single-qubit extensions
that keep a feasible completion (checked exactly on the window, with the edges to frozen
chains enforced), chooser = model / greedy / random.

Two protocols:
  blind      every rebuild is accepted; measures whether the chooser's local decisions add
             up to a better final embedding without any evaluation in the loop;
  evaluated  a rebuild is kept only if the surrogate p_solve does not drop (same number of
             evaluations for every arm), i.e. LNS with a learned proposal.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import networkx as nx

from embedbench.structural import frozen_adjacency, frozen_requirements, greedy_local_choice
from embedbench.exact import best_completion
from embedbench.objective import outcome_of
from embedbench.embedding import Node, Qubit


@dataclass
class RebuildTrace:
    rounds: int = 0
    rebuilt: int = 0
    accepted: int = 0
    skipped: int = 0
    decisions: int = 0
    p_history: list[float] = field(default_factory=list)


def _window(host, chains, in_play, rng, max_free):
    frozen = {v: c for v, c in chains.items() if v not in in_play}
    blocked = {q for c in frozen.values() for q in c}
    seed_set = set().union(*(chains[v] for v in in_play))
    ranked = []
    for q in seed_set:
        for n in host.neighbors(q):
            if n not in blocked and n not in seed_set:
                ranked.append((rng.random(), n))
    extra = [n for _, n in sorted(set(ranked))[:max_free]]
    return frozen, seed_set | set(extra)


def rebuild_once(host, logical, chains: dict, focus: Node, chooser, rng: random.Random, *,
                 l_cap: int = 3, max_free: int = 10, keep_neighbour: bool = True, trace: RebuildTrace | None = None,
                 feasibility_filter: bool = True, stats: dict | None = None):
    """Rebuild `focus` (and re-complete one neighbour) inside a window; returns new chains or
    None when no feasible rebuild exists in this window. With `feasibility_filter` every
    candidate qubit is pre-checked by exact completion, so the chooser only orders feasible
    actions; without it the chooser decides among all adjacent free qubits and a dead end is
    a failed rebuild (the setting where a learned feasibility judgement has value). `stats`
    accumulates exact search nodes."""
    nbrs = [u for u in logical.neighbors(focus)]
    if not nbrs:
        return None
    u = rng.choice(nbrs)
    in_play = [focus, u]
    frozen, window = _window(host, chains, in_play, rng, max_free)
    wh = host.subgraph(window)
    wl = logical.subgraph(in_play)
    must_hit, frozen_edges = frozen_requirements(host, logical, in_play, frozen, window)
    cores = {focus: frozenset(), u: chains[u] if keep_neighbour else frozenset()}
    free = set(window) - cores[u]
    core: set = set()
    # feasibility of the empty state
    st = {} if stats is None else stats
    if feasibility_filter and best_completion(wh, wl, cores, l_cap=l_cap, must_hit=must_hit, stats=st)[0][0] != 1:
        return None
    fa = frozen_adjacency(host, frozen, window)
    while True:
        # candidates: free window qubits adjacent to the core (or, for an empty core, adjacent
        # to u's core or to any required frozen chain, else any free window qubit), filtered
        # by exact feasibility of the resulting state
        if core:
            cand = sorted({n for q in core for n in wh.neighbors(q) if n in free})
        else:
            near = {n for q in cores[u] for n in wh.neighbors(q) if n in free}
            near |= {q for q in free if q in fa}
            cand = sorted(near) if near else sorted(free)
        feasible = []
        if feasibility_filter:
            for q in cand:
                c2 = dict(cores); c2[focus] = frozenset(core | {q})
                if best_completion(wh, wl, c2, l_cap=l_cap, must_hit=must_hit, stats=st)[0][0] == 1:
                    feasible.append(q)
        else:
            feasible = list(cand)
        if not feasible:
            return None
        rec = {
            "focus": focus, "in_play": in_play, "window_nodes": sorted(window),
            "window_edges": sorted([min(a, b), max(a, b)] for a, b in wh.edges()),
            "logical_edges": sorted([min(a, b), max(a, b)] for a, b in wl.edges()),
            "cores": {v: sorted(c) for v, c in cores.items()},
            "frozen": {v: sorted(c) for v, c in frozen.items() if any(q in fa for q in window)},
            "frozen_edges": frozen_edges, "frozen_adjacency": fa,
            "actions": [[focus, q] for q in feasible],
            "values": [[0.0, 0, 0] for _ in feasible], "best_action": [focus, feasible[0]],
            "greedy_action": [focus, feasible[0]],
            "witness_outcome": [1, -(sum(len(c) for c in cores.values()) + l_cap * 2), -l_cap],
            "l_cap": l_cap, "instance_id": "rebuild", "topology": "rebuild",
        }
        q = chooser(rec, wh, wl, cores, [(focus, x) for x in feasible])
        if trace: trace.decisions += 1
        core.add(q); free.discard(q)
        cores[focus] = frozenset(core)
        # done when focus touches every logical neighbour it has to (in-play u and frozen)
        touches_u = any(host.has_edge(a, b) for a in core for b in cores[u]) or not logical.has_edge(focus, u)
        touches_frozen = all(core & h for h in must_hit.get(focus, []))
        if (touches_u and touches_frozen) or len(core) >= l_cap:
            break
    # complete u (it may need to grow to touch the new focus chain); exact minimum-Q completion
    out, comp = best_completion(wh, wl, cores, l_cap=l_cap, must_hit=must_hit, stats=st)
    if out[0] != 1:
        return None
    new = dict(chains); new.update(comp)
    assert outcome_of(new, logical, host)[0] == 1
    return new


def model_rebuild_chooser(scorer):
    def choose(rec, wh, wl, cores, actions):
        s = scorer(rec)
        return actions[max(range(len(actions)), key=lambda i: s[i])][1]
    return choose


def greedy_rebuild_chooser():
    def choose(rec, wh, wl, cores, actions):
        return greedy_local_choice(wh, wl, cores, rec["focus"], actions)[1]
    return choose


def random_rebuild_chooser(seed=0):
    rng = random.Random(seed)
    return lambda rec, wh, wl, cores, actions: rng.choice(actions)[1]


def lns(host, logical, chains, chooser, evaluate, *, rounds: int, seed: int = 0, l_cap: int = 3,
        max_free: int = 10, mode: str = "evaluated", budget: int | None = None,
        feasibility_filter: bool = True, focus_chooser=None, stats: dict | None = None) -> tuple[dict, RebuildTrace]:
    """Rounds of single-chain rebuild. `evaluate(chains) -> p_solve` is called once per
    accepted-or-rejected candidate in 'evaluated' mode and only at the end in 'blind' mode.
    `feasibility_filter=False` lets the chooser decide without the exact pre-check (a dead
    end is a skipped round); `focus_chooser(chains, round, rng) -> variable` overrides the
    round-robin destroy choice; `stats` accumulates exact search nodes."""
    rng = random.Random(seed)
    tr = RebuildTrace()
    cur = dict(chains)
    p_cur = evaluate(cur) if mode == "evaluated" else None
    if p_cur is not None:
        tr.p_history.append(p_cur)
    order = list(logical.nodes())
    evals = 0
    for r in range(rounds):
        if budget is not None and mode == "evaluated" and evals >= budget:
            break
        tr.rounds += 1
        if focus_chooser is not None:
            focus = focus_chooser(cur, r, rng)
        else:
            focus = order[r % len(order)] if r < len(order) else rng.choice(order)
        new = rebuild_once(host, logical, cur, focus, chooser, rng, l_cap=l_cap, max_free=max_free, trace=tr,
                           feasibility_filter=feasibility_filter, stats=stats)
        if new is None or new == cur:
            tr.skipped += 1
            continue
        tr.rebuilt += 1
        if mode == "blind":
            cur = new; tr.accepted += 1
        elif mode == "structural":
            key = lambda ch: (max(len(c) for c in ch.values()), sum(len(c) for c in ch.values()))
            if key(new) <= key(cur):
                cur = new; tr.accepted += 1
        else:
            p_new = evaluate(new); evals += 1
            if p_new >= p_cur:
                cur, p_cur = new, p_new; tr.accepted += 1
            tr.p_history.append(p_cur)
    return cur, tr
