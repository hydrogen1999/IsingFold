"""Exhaustive completion of a partial embedding, for hand-tests small enough to enumerate.

The meeting's requirement (section 6): a hand-test is only a certified counterfactual if the
value of every candidate action is computed by exact enumeration, not by running a heuristic
and recording what it did. This module does that enumeration.

State: a host graph, a logical graph, and a `cores` map giving each variable the qubits it
already holds (possibly none). A completion assigns every variable a connected chain that is
a superset of its core, chains are disjoint, and every logical edge is realised by a coupler.
Search is depth-first over variables with connected-subset enumeration per variable and
branch-and-bound on the lexicographic outcome. Sizes are capped by `l_cap` per chain and
optionally `q_cap` in total.

This is exponential and meant for motifs of up to roughly eight variables on thirty qubits.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import networkx as nx

from embedbench.embedding import Node, Qubit
from embedbench.objective import INFEASIBLE, Outcome, compare

Chains = dict[Node, frozenset[Qubit]]


class SearchAborted(RuntimeError):
    """The node budget ran out before the enumeration finished. No certificate."""


def connected_supersets(
    host: nx.Graph, core: frozenset[Qubit], free: set[Qubit], max_size: int
) -> Iterable[frozenset[Qubit]]:
    """Every connected qubit set S with core <= S <= core | free and |S| <= max_size.
    Enumerated level by level with de-duplication; fine at the sizes hand-tests use."""
    if core:
        if not nx.is_connected(host.subgraph(core)):
            return
        level = {core}
    else:
        level = {frozenset([q]) for q in free}
    seen: set[frozenset[Qubit]] = set()
    while level:
        nxt: set[frozenset[Qubit]] = set()
        for s in level:
            if s in seen:
                continue
            seen.add(s)
            yield s
            if len(s) >= max_size:
                continue
            for q in s:
                for nb in host.neighbors(q):
                    if nb in free and nb not in s:
                        nxt.add(s | {nb})
        level = nxt


MustHit = Mapping[Node, Sequence[frozenset[Qubit]]]
"""Per variable, qubit sets each of which its chain must intersect: the window qubits
adjacent to the chain of every *frozen* logical neighbour. Without this a completion could
leave a logical edge to the frozen context unrealised and still count as feasible."""


def _hits(s: frozenset[Qubit], reqs: Sequence[frozenset[Qubit]] | None) -> bool:
    return not reqs or all(s & r for r in reqs)


@dataclass
class _Search:
    host: nx.Graph
    logical: nx.Graph
    order: list[Node]
    cores: Chains
    l_cap: int
    q_cap: int | None
    must_hit: MustHit | None = None
    best: Outcome = INFEASIBLE
    best_chains: Chains | None = None
    nodes_visited: int = 0
    max_nodes: int | None = None
    aborted: bool = False
    assigned: Chains = field(default_factory=dict)

    def run(self) -> None:
        free = (
            set(self.host.nodes) - set().union(*self.cores.values())
            if self.cores
            else set(self.host.nodes)
        )
        self._dfs(0, free, 0, 0)

    def _lower_bound_q(self, idx: int, q_now: int) -> int:
        return q_now + sum(max(1, len(self.cores.get(v, ()))) for v in self.order[idx:])

    def _dfs(self, idx: int, free: set[Qubit], q_now: int, lmax_now: int) -> None:
        if self.aborted:
            return
        if self.max_nodes is not None and self.nodes_visited >= self.max_nodes:
            self.aborted = True
            return
        self.nodes_visited += 1
        if self.best[0] == 1:
            lb = self._lower_bound_q(idx, q_now)
            if -lb < self.best[1]:
                return
            if -lb == self.best[1] and -lmax_now <= self.best[2]:
                return
        if self.q_cap is not None and self._lower_bound_q(idx, q_now) > self.q_cap:
            return
        if idx == len(self.order):
            out = (1, -q_now, -lmax_now)
            if out > self.best:
                self.best, self.best_chains = out, dict(self.assigned)
            return
        v = self.order[idx]
        core = self.cores.get(v, frozenset())
        placed_nb = [u for u in self.logical.neighbors(v) if u in self.assigned]
        reqs = self.must_hit.get(v) if self.must_hit else None
        for s in connected_supersets(self.host, core, free, self.l_cap):
            if not _hits(s, reqs):
                continue
            ok = True
            for u in placed_nb:
                cu = self.assigned[u]
                if not any(self.host.has_edge(a, b) for a in s for b in cu):
                    ok = False
                    break
            if not ok:
                continue
            self.assigned[v] = s
            self._dfs(idx + 1, free - s, q_now + len(s), max(lmax_now, len(s)))
            del self.assigned[v]


def best_completion(
    host: nx.Graph,
    logical: nx.Graph,
    cores: Mapping[Node, Iterable[Qubit]] | None = None,
    *,
    l_cap: int = 4,
    q_cap: int | None = None,
    max_nodes: int | None = None,
    stats: dict | None = None,
    must_hit: MustHit | None = None,
) -> tuple[Outcome, Chains | None]:
    """V* of a partial state: the best lexicographic outcome over all completions, and one
    completion that attains it (None when no completion exists).

    `max_nodes` bounds DFS-state entries. Before state `max_nodes + 1` is entered,
    `SearchAborted` is raised with exactly `max_nodes` visited states, because a truncated
    enumeration is not a certificate and must not be mistaken for one."""
    if max_nodes is not None and (type(max_nodes) is not int or max_nodes <= 0):
        raise ValueError("max_nodes must be a positive integer or None")
    cores_f: Chains = {v: frozenset(qs) for v, qs in (cores or {}).items() if qs}
    for v, c in cores_f.items():
        if not c <= set(host.nodes):
            raise ValueError(f"core of {v} leaves the host")
    used = [q for c in cores_f.values() for q in c]
    if len(used) != len(set(used)):
        raise ValueError("cores overlap")
    # constrained variables first: those with a core, then by logical degree
    order = sorted(
        logical.nodes(),
        key=lambda v: (0 if v in cores_f else 1, -logical.degree(v), v),
    )
    s = _Search(host, logical, order, cores_f, l_cap, q_cap, max_nodes=max_nodes, must_hit=must_hit)
    s.run()
    if stats is not None:
        stats["nodes"] = stats.get("nodes", 0) + s.nodes_visited
    if s.aborted:
        raise SearchAborted(f"exact completion exceeded {max_nodes} nodes")
    return s.best, s.best_chains


@dataclass(frozen=True)
class DecisionSample:
    """One training sample (S_t, A_t, y_t) of section 6: the partial state, the legal
    candidate actions, and the certified value of each."""

    cores: Chains
    actions: tuple[tuple[Node, Qubit], ...]
    values: tuple[Outcome, ...]
    completions: tuple[Chains | None, ...]

    def ranking(self) -> list[int]:
        return sorted(range(len(self.actions)), key=lambda k: self.values[k], reverse=True)

    def margin(self) -> tuple[str, int]:
        """Between the best and the second-best action."""
        r = self.ranking()
        if len(r) < 2:
            return "none", 0
        return compare(self.values[r[0]], self.values[r[1]])


def value_of_action(
    host: nx.Graph,
    logical: nx.Graph,
    cores: Mapping[Node, Iterable[Qubit]],
    action: tuple[Node, Qubit],
    **kw,
) -> tuple[Outcome, Chains | None]:
    """V*(S_t, a) for the action 'add qubit q to the chain of variable v' (placing v if it has
    no chain yet). The qubit must be free and, if v already has a chain, adjacent to it."""
    v, q = action
    cores_f: Chains = {u: frozenset(qs) for u, qs in cores.items() if qs}
    if any(q in c for c in cores_f.values()):
        raise ValueError(f"qubit {q} is occupied")
    core = cores_f.get(v, frozenset())
    if core and not any(host.has_edge(q, a) for a in core):
        raise ValueError(f"qubit {q} is not adjacent to the chain of {v}")
    cores_f[v] = core | {q}
    return best_completion(host, logical, cores_f, **kw)


def sample_at(
    host: nx.Graph,
    logical: nx.Graph,
    cores: Mapping[Node, Iterable[Qubit]],
    actions: Iterable[tuple[Node, Qubit]],
    **kw,
) -> DecisionSample:
    acts = tuple(actions)
    vals, comps = [], []
    for a in acts:
        o, c = value_of_action(host, logical, cores, a, **kw)
        vals.append(o)
        comps.append(c)
    return DecisionSample(
        cores={u: frozenset(qs) for u, qs in cores.items() if qs},
        actions=acts, values=tuple(vals), completions=tuple(comps),
    )


def all_completions(
    host: nx.Graph,
    logical: nx.Graph,
    cores: Mapping[Node, Iterable[Qubit]] | None = None,
    *,
    l_cap: int = 4,
    q_cap: int | None = None,
    max_count: int | None = None,
    must_hit: MustHit | None = None,
) -> list[Chains]:
    """Every feasible completion of a partial state (no branch-and-bound), for objectives
    that must be evaluated on complete embeddings rather than read off (Q, L_max). Raises
    `SearchAborted` when more than `max_count` completions exist."""
    cores_f: Chains = {v: frozenset(qs) for v, qs in (cores or {}).items() if qs}
    order = sorted(logical.nodes(), key=lambda v: (0 if v in cores_f else 1, -logical.degree(v), v))
    free0 = set(host.nodes) - {q for c in cores_f.values() for q in c}
    out: list[Chains] = []
    assigned: Chains = {}

    def dfs(idx, free, q_now):
        minimum_total = q_now + sum(
            max(1, len(cores_f.get(v, ()))) for v in order[idx:]
        )
        if q_cap is not None and minimum_total > q_cap:
            return
        if idx == len(order):
            out.append(dict(assigned))
            if max_count is not None and len(out) > max_count:
                raise SearchAborted(f"more than {max_count} completions")
            return
        v = order[idx]
        placed_nb = [u for u in logical.neighbors(v) if u in assigned]
        reqs = must_hit.get(v) if must_hit else None
        for s in connected_supersets(host, cores_f.get(v, frozenset()), free, l_cap):
            if not _hits(s, reqs):
                continue
            if all(any(host.has_edge(a, b) for a in s for b in assigned[u]) for u in placed_nb):
                assigned[v] = s
                dfs(idx + 1, free - s, q_now + len(s))
                del assigned[v]

    dfs(0, free0, 0)
    return out
