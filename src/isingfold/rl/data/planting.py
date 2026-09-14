"""Frustrated-loop planting composed onto an ink-drop support.

Spec: Rev2 section "Compose embedding planting with logical-energy planting". Each clause is
a simple cycle with one antiferromagnetic edge in the all-plus gauge, then gauged by a private
planted spin vector. Every clause is bounded below independently and the planted assignment
attains all bounds simultaneously, so ``E_0 = -sum_l w_l (L_l - 2) - sum_i b_i`` is a proof,
not a consistency check. Coefficient limits are imposed by *rejecting whole clauses*: clipping
a summed coupler would destroy the decomposition and invalidate the bound.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Hashable, Mapping, Sequence

import networkx as nx

from isingfold.embedding import LogicalProblem

Node = Hashable


class PlantingError(RuntimeError):
    """The requested clause budget is not satisfiable on this support."""


@dataclass(frozen=True)
class Clause:
    cycle: tuple[Node, ...]
    weight: float
    unsatisfied_edge: tuple[Node, Node]

    @property
    def bound(self) -> float:
        return -self.weight * (len(self.cycle) - 2)


@dataclass(frozen=True)
class PlantedInstance:
    problem: LogicalProblem
    ground_energy: float
    planted_spins: Mapping[Node, int] = field(repr=False)
    clauses: tuple[Clause, ...] = field(repr=False, default=())
    field_clauses: Mapping[Node, float] = field(repr=False, default_factory=dict)
    support: nx.Graph = field(repr=False, default_factory=nx.Graph)

    def verify(self) -> dict[str, object]:
        """Check that the planted assignment attains the clause-sum bound exactly."""

        energy = sum(
            self.problem.h.get(i, 0.0) * self.planted_spins[i] for i in self.planted_spins
        )
        for (u, v), w in self.problem.j.items():
            energy += w * self.planted_spins[u] * self.planted_spins[v]
        bound = sum(c.bound for c in self.clauses) - sum(self.field_clauses.values())
        return {
            "planted_energy": energy,
            "clause_bound": bound,
            "declared_ground": self.ground_energy,
            "attains_bound": abs(energy - bound) < 1e-9,
            "matches_declared": abs(energy - self.ground_energy) < 1e-9,
        }


def _simple_cycle(graph: nx.Graph, rng: random.Random, max_length: int) -> list[Node] | None:
    nodes = list(graph.nodes())
    if len(nodes) < 3:
        return None
    for _ in range(64):
        start = rng.choice(nodes)
        walk = [start]
        used = {start}
        while len(walk) < max_length:
            options = [n for n in graph.neighbors(walk[-1]) if n not in used]
            closing = [n for n in graph.neighbors(walk[-1]) if n == start and len(walk) >= 3]
            if closing and (not options or rng.random() < 0.4):
                return walk
            if not options:
                break
            nxt = rng.choice(options)
            walk.append(nxt)
            used.add(nxt)
        if len(walk) >= 3 and graph.has_edge(walk[-1], start):
            return walk
    return None


def frustrated_loops(
    support: nx.Graph,
    *,
    alpha: float = 0.3,
    seed: int = 0,
    max_length: int = 6,
    weight_choices: Sequence[float] = (1.0,),
    coupler_limit: float = 4.0,
    field_clause_rate: float = 0.0,
) -> PlantedInstance:
    """Plant a known logical optimum on ``support`` and return the certified instance."""

    rng = random.Random(seed)
    spins = {v: (1 if rng.random() < 0.5 else -1) for v in support.nodes()}
    target = max(1, int(round(alpha * support.number_of_nodes())))
    couplings: dict[tuple[Node, Node], float] = {}
    clauses: list[Clause] = []

    attempts = 0
    while len(clauses) < target and attempts < 200 * target:
        attempts += 1
        cycle = _simple_cycle(support, rng, max_length)
        if cycle is None:
            break
        weight = rng.choice(list(weight_choices))
        edges = [(cycle[k], cycle[(k + 1) % len(cycle)]) for k in range(len(cycle))]
        unsat = edges[rng.randrange(len(edges))]
        trial = dict(couplings)
        for u, v in edges:
            key = (u, v) if str(u) <= str(v) else (v, u)
            sign = 1.0 if (u, v) == unsat or (v, u) == unsat else -1.0
            trial[key] = trial.get(key, 0.0) + sign * weight * spins[u] * spins[v]
        if any(abs(w) > coupler_limit for w in trial.values()):
            continue  # reject the whole clause; never clip a summed coupler
        couplings = trial
        clauses.append(Clause(tuple(cycle), weight, unsat))

    if not clauses:
        raise PlantingError("no frustrated clause fits this support")

    fields: dict[Node, float] = {}
    field_clauses: dict[Node, float] = {}
    if field_clause_rate > 0:
        for v in support.nodes():
            if rng.random() < field_clause_rate:
                b = rng.choice(list(weight_choices))
                fields[v] = -b * spins[v]
                field_clauses[v] = b

    active = {k: w for k, w in couplings.items() if abs(w) > 1e-12}
    graph_nodes = set(support.nodes())
    problem = LogicalProblem.from_dicts(
        {v: fields.get(v, 0.0) for v in graph_nodes}, active
    )
    ground = sum(c.bound for c in clauses) - sum(field_clauses.values())
    instance = PlantedInstance(
        problem=problem,
        ground_energy=ground,
        planted_spins=spins,
        clauses=tuple(clauses),
        field_clauses=field_clauses,
        support=support,
    )
    report = instance.verify()
    if not report["attains_bound"]:
        raise PlantingError(f"clause decomposition broken: {report}")
    return instance
