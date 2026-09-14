"""Frustrated-loop planting on an arbitrary logical graph (Hen et al. 2015 construction).

Pick a planted spin configuration s*. Repeatedly draw a simple cycle C of the logical graph
by a self-avoiding walk; on every edge of C add a coupling that s* satisfies
(J_ij -= s*_i s*_j), then flip one edge of C so exactly one bond of the loop is frustrated.
s* satisfies |C| - 1 of the |C| bonds, which is the most any configuration can, so s*
minimises every loop term and therefore their sum. The ground energy is the sum of the loop
minima and needs no solver. Degeneracy is not controlled: s* is a ground state, not
necessarily the only one, which is fine for a solve probability defined by energy.

Hardness is set by alpha (loops per variable) and the loop length range; the returned
record keeps both so a corpus can state them.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import networkx as nx

from embedbench.embedding import LogicalProblem, Node


class PlantingError(RuntimeError):
    """Not enough cycles could be drawn on this graph at this alpha."""


@dataclass(frozen=True)
class PlantedIsing:
    problem: LogicalProblem
    spins: dict[Node, int]
    ground_energy: float
    n_loops: int
    alpha: float
    loop_lengths: tuple[int, ...]

    def energy_of_planted(self) -> float:
        e = 0.0
        for (u, v), j in self.problem.j.items():
            e += j * self.spins[u] * self.spins[v]
        return e


def _random_cycle(g: nx.Graph, rng: random.Random, min_len: int, max_len: int) -> list[Node] | None:
    """A simple cycle of length in [min_len, max_len] by a self-avoiding walk that returns to
    its start; None if the walk dies."""
    nodes = list(g.nodes())
    start = rng.choice(nodes)
    path = [start]
    seen = {start}
    while len(path) <= max_len:
        cur = path[-1]
        nbs = list(g.neighbors(cur))
        rng.shuffle(nbs)
        if len(path) >= min_len and start in nbs:
            return path
        nxt = [n for n in nbs if n not in seen]
        if not nxt:
            return None
        n = nxt[0]
        path.append(n)
        seen.add(n)
    return None


def frustrated_loops(
    graph: nx.Graph, *, alpha: float = 0.3, min_len: int = 3, max_len: int = 8,
    seed: int = 0, spins: dict[Node, int] | None = None, attempts_per_loop: int = 200,
) -> PlantedIsing:
    rng = random.Random(seed)
    nodes = sorted(graph.nodes())
    s = spins or {v: rng.choice((-1, 1)) for v in nodes}
    n_loops = max(1, round(alpha * len(nodes)))
    j: dict[tuple[Node, Node], float] = {}
    lens: list[int] = []

    def key(a, b):
        return (a, b) if a < b else (b, a)

    for _ in range(n_loops):
        cyc = None
        for _a in range(attempts_per_loop):
            cyc = _random_cycle(graph, rng, min_len, max_len)
            if cyc is not None:
                break
        if cyc is None:
            raise PlantingError(f"could not draw loop {len(lens)+1} of {n_loops} on this graph")
        edges = [key(cyc[i], cyc[(i + 1) % len(cyc)]) for i in range(len(cyc))]
        for (a, b) in edges:
            j[(a, b)] = j.get((a, b), 0.0) - s[a] * s[b]
        a, b = rng.choice(edges)
        j[(a, b)] += 2.0 * s[a] * s[b]
        lens.append(len(cyc))
    scale = max(abs(v) for v in j.values())
    j = {k: v / scale for k, v in j.items() if abs(v) > 1e-12}
    h = {v: 0.0 for v in nodes}
    problem = LogicalProblem.from_dicts(h, j)
    # ground energy: each loop contributes -(|C| - 2) before scaling; recompute from s*
    e0 = sum(w * s[a] * s[b] for (a, b), w in j.items())
    return PlantedIsing(problem=problem, spins=dict(s), ground_energy=float(e0), n_loops=n_loops,
                        alpha=alpha, loop_lengths=tuple(lens))
