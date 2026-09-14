"""The Track B seam, finally used: a learned scorer inside the modular minorminer search.

`lac_minorminer`'s orchestrator asks `scorer.score(snapshot, batch)` for every candidate
batch and applies the candidate with the lowest score. The default `ResourceScorer` ranks
by (max occupancy, excess occupancy, used target nodes, route cost). The scorers here keep
the first two keys, which are the feasibility part of that order and what minorminer's
repair relies on, and replace the rest:

  LearnedTieBreakScorer  among the candidates tied on (max occupancy, excess occupancy)
                         and free of overlaps, the chain-level scorer of L-79 orders them
                         (higher model score = better); other candidates keep the resource
                         order behind them;
  RandomTieBreakScorer   the same, with a random order in the tied group (control);
  ResourceScorer         the harness default (baseline).

The chain scorer sees the state as the chain-seam records did: the candidate chains as the
window, every other placed chain as frozen context, the logical neighbours that already have
chains, and the problem coefficients on the focus's edges.
"""
from __future__ import annotations

import random
from typing import Any

import networkx as nx

from embedbench.embedding import LogicalProblem


class _Base:
    def __init__(self, problem: LogicalProblem, logical: nx.Graph, host: nx.Graph, session, l_cap: int = 4, max_free: int = 8):
        self.problem = problem
        self.max_free = max_free
        self.logical = logical
        """The graph being embedded. `problem.graph` drops zero-coupling edges, so it is not
        the embedding constraint; `logical` is."""
        self.host = host
        self.src = session.normalized_source
        self.tgt = session.normalized_target
        self.l_cap = l_cap
        self.calls = 0
        self.groups = 0
        self.decided = 0
        self.tabu: set = set()
        """(logical label, chain) pairs already evaluated and rejected at the current state;
        `score` pushes them behind every other candidate so a deterministic order moves to
        its next-ranked candidate instead of re-evaluating the same one (codex ML audit)."""

    @staticmethod
    def _base_key(c):
        r = c.rank
        return (r.max_occupancy, r.total_excess_occupancy)

    @staticmethod
    def _resource_tail(c):
        r = c.rank
        return (r.used_target_nodes, r.route_cost)

    def _order_group(self, snapshot, batch, group: list[int]) -> list[float]:
        raise NotImplementedError

    def score(self, snapshot: Any, batch: Any) -> list[float]:
        self.calls += 1
        cands = list(batch.candidates)
        keys = [self._base_key(c) for c in cands]
        best = min(keys)
        group = [i for i, k in enumerate(keys) if k == best]
        scores = [0.0] * len(cands)
        # non-group candidates: resource order behind the group
        base_rank = {k: j for j, k in enumerate(sorted(set(keys)))}
        for i, c in enumerate(cands):
            scores[i] = base_rank[keys[i]] * 1e6 + self._resource_tail(c)[0] * 1e2 + self._resource_tail(c)[1]
        if len(group) > 1 and best[0] <= 1:
            self.groups += 1
            inner = self._order_group(snapshot, batch, group)
            for j, i in enumerate(group):
                scores[i] = inner[j]  # in [0, 1e6): ahead of every other base rank
            self.decided += 1
        if self.tabu:
            focus = self.src.labels[batch.logical]
            for i, c in enumerate(cands):
                if (focus, frozenset(self.tgt.labels[t] for t in c.chain)) in self.tabu:
                    scores[i] += 1e9
        return scores


class RandomTieBreakScorer(_Base):
    def __init__(self, problem, logical, host, session, seed: int = 0, l_cap: int = 4):
        super().__init__(problem, logical, host, session, l_cap)
        self.rng = random.Random(seed)

    def _order_group(self, snapshot, batch, group):
        order = list(range(len(group)))
        self.rng.shuffle(order)
        return [float(o) for o in order]


class LearnedTieBreakScorer(_Base):
    def __init__(self, problem, logical, host, session, chain_scorer, l_cap: int = 4):
        super().__init__(problem, logical, host, session, l_cap)
        self.chain_scorer = chain_scorer

    def _record(self, snapshot, batch, group):
        logical_idx = batch.logical
        focus = self.src.labels[logical_idx]
        cands = [batch.candidates[i] for i in group]
        chains_t = [[self.tgt.labels[t] for t in c.chain] for c in cands]
        # frozen: every other variable's current chain (target labels)
        frozen = {}
        for li, ch in enumerate(snapshot.chains):
            if li != logical_idx and len(ch):
                frozen[self.src.labels[li]] = sorted(self.tgt.labels[t] for t in ch)
        nbrs = [u for u in self.logical.neighbors(focus) if u in frozen]
        # window as in training (chainseam.py): the focus's current chain and the candidate
        # chains, plus up to `max_free` free qubits within two hops (deterministic order)
        current = {self.tgt.labels[t] for t in snapshot.chains[logical_idx]}
        seed_set = {q for ch in chains_t for q in ch} | current
        blocked = {q for c in frozen.values() for q in c}
        first = sorted({n for q in seed_set for n in self.host.neighbors(q) if n not in blocked and n not in seed_set})
        second = sorted({n for q in first for n in self.host.neighbors(q) if n not in blocked and n not in seed_set and n not in first})
        extra = (first + second)[: self.max_free]
        window = sorted(seed_set | set(extra))
        wset = set(window)
        fq = {q for c in frozen.values() for q in c}
        fa = {int(q): sorted(int(n) for n in self.host.neighbors(q) if n in fq) for q in window
              if any(n in fq for n in self.host.neighbors(q))}
        rec = {
            "window_nodes": [int(q) for q in window],
            "window_edges": sorted([min(a, b), max(a, b)] for a, b in self.host.subgraph(wset).edges()),
            "frozen": {int(v): [int(q) for q in c] for v, c in frozen.items()
                       if any(n in wset for q in c for n in self.host.neighbors(q))},
            "frozen_adjacency": fa,
            "neighbours": [int(u) for u in nbrs],
            "candidates": [sorted(int(q) for q in ch) for ch in chains_t],
            "p_solve": [0.0] * len(cands), "best_index": 0, "resource_index": 0, "original_index": -1,
            "source": "seam", "instance_id": "seam", "topology": "seam", "l_cap": self.l_cap,
            "focus_h": float(self.problem.h.get(focus, 0.0)),
            "edge_J": [float(self.problem.j.get((min(focus, u), max(focus, u)), 0.0)) for u in nbrs],
            "neighbour_degree": [int(self.logical.degree(u)) for u in nbrs],
            "neighbour_chain_size": [len(frozen[u]) for u in nbrs],
            "j_scale": 1.0,
        }
        return rec

    def _order_group(self, snapshot, batch, group):
        rec = self._record(snapshot, batch, group)
        s = self.chain_scorer(rec)
        order = sorted(range(len(group)), key=lambda j: -s[j])
        out = [0.0] * len(group)
        for rank, j in enumerate(order):
            out[j] = float(rank)
        return out

    def score_with_current(self, snapshot, batch, indices, current_chain):
        """Model scores of the given candidates and of the current chain (appended as an
        extra candidate in the same record), so a caller can gate on the margin."""
        rec = self._record(snapshot, batch, indices)
        rec["candidates"].append(sorted(int(q) for q in current_chain))
        rec["p_solve"].append(0.0)
        window = set(rec["window_nodes"]) | set(current_chain)
        rec["window_nodes"] = sorted(int(q) for q in window)
        rec["window_edges"] = sorted([min(a, b), max(a, b)] for a, b in self.host.subgraph(window).edges())
        fq = {q for c in rec["frozen"].values() for q in c}
        rec["frozen_adjacency"] = {int(q): sorted(int(n) for n in self.host.neighbors(q) if n in fq) for q in window
                                   if any(n in fq for n in self.host.neighbors(q))}
        s = self.chain_scorer(rec)
        return s[:-1], s[-1]


class QualityRouteCost:
    """The harness's second seam: a non-negative cost per target qubit before the router
    proposes chains for a variable. The default (None) is plain route length, which biases
    every proposal toward the shortest chain. This provider pulls routes toward the chains of
    the focus's logical neighbours in proportion to |J| on that edge (a contact where the
    coupling is strong matters most for the decoded energy), and pushes them away from
    qubits crowded by other chains. Costs stay in [floor, 1 + lam]; no learning here, the
    scorer still decides among the proposals.
    """

    def __init__(self, problem: LogicalProblem, logical: nx.Graph, host: nx.Graph, session,
                 mu: float = 0.8, lam: float = 0.5, floor: float = 0.2):
        self.problem, self.logical, self.host = problem, logical, host
        self.src, self.tgt = session.normalized_source, session.normalized_target
        self.mu, self.lam, self.floor = mu, lam, floor
        self.jmax = max((abs(v) for v in problem.j.values()), default=1.0) or 1.0
        self.calls = 0

    def costs(self, snapshot: Any, logical_id: int, target_size: int):
        self.calls += 1
        focus = self.src.labels[logical_id]
        owner: dict[int, int] = {}
        for li, ch in enumerate(snapshot.chains):
            if li == logical_id:
                continue
            for t in ch:
                owner[t] = li
        pull = [0.0] * target_size
        crowd = [0] * target_size
        for t in range(target_size):
            q = self.tgt.labels[t]
            for n in self.host.neighbors(q):
                ti = self.tgt.index(n)
                li = owner.get(ti)
                if li is None:
                    continue
                u = self.src.labels[li]
                if self.logical.has_edge(focus, u):
                    j = abs(self.problem.j.get((min(focus, u), max(focus, u)), 0.0)) / self.jmax
                    pull[t] = max(pull[t], j)
                else:
                    crowd[t] += 1
        return [max(self.floor, 1.0 - self.mu * pull[t] + self.lam * min(crowd[t], 3) / 3.0) for t in range(target_size)]
