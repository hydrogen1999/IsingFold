"""Ink-drop planted-embedding generator, section 5 of the 2026-09-08 meeting note.

Each logical variable starts as a droplet on a seed qubit and expands to unused neighbouring
qubits, so every planted chain is connected by construction. Growth is not uniform: a
candidate qubit q for droplet v is scored

    S(q | C_v) = alpha A(q; v) + beta K_res(q) + gamma B(q) - lambda R(q) - mu L(v)

and sampled with softmax temperature tau. The terms:

    A(q; v)   attraction: pull toward the regions of the logical neighbours v still has to
              reach (only when a required logical graph is given);
    K_res(q)  residual connectivity: free neighbours of q;
    B(q)      required contacts that adding q completes;
    R(q)      congestion: occupied neighbours of q, plus a penalty when q is an articulation
              point of the free subgraph (a bridge some other droplet may need);
    L(v)      chain-growth discouragement: |C_v| relative to the target size.

Output is (H, G_L, Pi*): the host, the logical graph, and the witness embedding. Two uses:

* free growth (`required=None`): G_L is the contact quotient of the regions; any G_L produced
  this way is embeddable in H by construction;
* target-guided growth (`required=G`): the droplets are pulled toward the logical neighbours
  they must touch; the result is a planted witness for G if every required edge is realised,
  otherwise generation fails and the caller counts the attrition.

Modes preset the weights and the frontier rule for the instance families of section 8.
"""
from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

import networkx as nx

from embedbench.embedding import Node, Qubit


class PlantedGenerationError(RuntimeError):
    """A seed that cannot realise its required logical graph. Expected attrition."""


@dataclass(frozen=True)
class InkDropConfig:
    alpha: float = 1.0
    beta: float = 0.3
    gamma: float = 1.0
    lam: float = 0.5
    mu: float = 0.5
    tau: float = 0.5
    frontier: str = "region"
    """'region': any free neighbour of the region; 'walk': free neighbours of the last-added
    qubit (elongated, random-walk chains), falling back to the region frontier."""
    articulation_penalty: float = 1.0
    seed_spacing: int = 2
    """Minimum host distance between seed qubits."""


MODES: dict[str, InkDropConfig] = {
    "compact": InkDropConfig(beta=0.6, lam=1.0, mu=0.8, tau=0.3, frontier="region"),
    "elongated": InkDropConfig(beta=0.0, lam=0.1, mu=0.1, tau=1.0, frontier="walk", seed_spacing=3),
    "cut_congested": InkDropConfig(alpha=2.0, beta=0.2, lam=0.0, mu=0.3, tau=0.5, articulation_penalty=0.0),
    "near_capacity": InkDropConfig(beta=0.3, lam=0.3, mu=0.0, tau=0.7, seed_spacing=1),
}


@dataclass(frozen=True)
class PlantedEmbedding:
    host: nx.Graph = field(repr=False)
    logical: nx.Graph = field(repr=False)
    chains: Mapping[Node, frozenset[Qubit]]
    contact_graph: nx.Graph = field(repr=False)
    """The full quotient of the regions; equals `logical` under free growth."""
    mode: str
    seed: int
    config: InkDropConfig

    @property
    def qubits_used(self) -> int:
        return sum(len(c) for c in self.chains.values())

    @property
    def max_chain_length(self) -> int:
        return max(len(c) for c in self.chains.values())

    def validate(self) -> list[str]:
        """Independent check of the witness. Empty list means Pi* embeds G_L in H."""
        errs = []
        seen: set[Qubit] = set()
        for v, c in self.chains.items():
            if not c:
                errs.append(f"chain {v} empty")
                continue
            if not c <= set(self.host.nodes):
                errs.append(f"chain {v} leaves the host")
            if not nx.is_connected(self.host.subgraph(c)):
                errs.append(f"chain {v} disconnected")
            if seen & c:
                errs.append(f"chain {v} overlaps another")
            seen |= c
        for u, v in self.logical.edges():
            if not any(self.host.has_edge(a, b) for a in self.chains[u] for b in self.chains[v]):
                errs.append(f"logical edge {(u, v)} has no coupler")
        return errs

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "seed": self.seed,
            "host_nodes": sorted(self.host.nodes),
            "host_edges": sorted(tuple(sorted(e)) for e in self.host.edges),
            "logical_edges": sorted(tuple(sorted(e)) for e in self.logical.edges),
            "chains": {int(v): sorted(c) for v, c in self.chains.items()},
            "Q": self.qubits_used,
            "L_max": self.max_chain_length,
        }


def _quotient(host: nx.Graph, chains: Mapping[Node, frozenset[Qubit]]) -> nx.Graph:
    owner = {q: v for v, c in chains.items() for q in c}
    g = nx.Graph()
    g.add_nodes_from(chains)
    for a, b in host.edges():
        if a in owner and b in owner and owner[a] != owner[b]:
            g.add_edge(owner[a], owner[b])
    return g


def _pick_seeds(host: nx.Graph, n: int, spacing: int, rng: random.Random, free: set[Qubit]) -> list[Qubit]:
    order = sorted(free)
    rng.shuffle(order)
    chosen: list[Qubit] = []
    for q in order:
        if len(chosen) == n:
            break
        if spacing <= 1 or all(
            nx.shortest_path_length(host, q, c) >= spacing if nx.has_path(host, q, c) else True
            for c in chosen
        ):
            chosen.append(q)
    if len(chosen) < n:
        # relax spacing rather than fail: capacity is the caller's knob
        for q in order:
            if len(chosen) == n:
                break
            if q not in chosen:
                chosen.append(q)
    if len(chosen) < n:
        raise PlantedGenerationError(f"host has {len(free)} free qubits, {n} seeds requested")
    return chosen


def ink_drop(
    host: nx.Graph,
    n_vars: int,
    sizes: Sequence[int] | int,
    *,
    required: nx.Graph | None = None,
    mode: str = "compact",
    config: InkDropConfig | None = None,
    seed: int = 0,
    blocked: set[Qubit] | None = None,
) -> PlantedEmbedding:
    """Grow `n_vars` droplets on `host` to the target `sizes` (one int for all)."""
    cfg = config or MODES[mode]
    rng = random.Random(seed)
    targets = [sizes] * n_vars if isinstance(sizes, int) else list(sizes)
    if len(targets) != n_vars:
        raise ValueError("one target size per variable")
    if required is not None and set(required.nodes) - set(range(n_vars)):
        raise ValueError("required graph must be on variables 0..n_vars-1")

    free: set[Qubit] = set(host.nodes) - (blocked or set())
    seeds = _pick_seeds(host, n_vars, cfg.seed_spacing, rng, free)
    regions: dict[Node, set[Qubit]] = {v: {s} for v, s in enumerate(seeds)}
    last: dict[Node, Qubit] = {v: s for v, s in enumerate(seeds)}
    for s in seeds:
        free.discard(s)
    owner: dict[Qubit, Node] = {s: v for v, s in enumerate(seeds)}
    mean_target = sum(targets) / n_vars

    def contacts(v: Node) -> set[Node]:
        out = set()
        for q in regions[v]:
            for nb in host.neighbors(q):
                o = owner.get(nb)
                if o is not None and o != v:
                    out.add(o)
        return out

    def dist_to_region(q: Qubit, u: Node) -> float:
        # BFS from q through free qubits and u's region, bounded
        target = regions[u]
        if q in target:
            return 0.0
        frontier, seen, d = [q], {q}, 0
        while frontier and d < 12:
            d += 1
            nxt = []
            for x in frontier:
                for nb in host.neighbors(x):
                    if nb in target:
                        return float(d)
                    if nb in free and nb not in seen:
                        seen.add(nb)
                        nxt.append(nb)
            frontier = nxt
        return float("inf")

    def score(q: Qubit, v: Node, articulation: set[Qubit]) -> float:
        a_term = b_term = 0.0
        if required is not None:
            missing = set(required.neighbors(v)) - contacts(v)
            for u in missing:
                d = dist_to_region(q, u)
                if d == 0.0 or any(owner.get(nb) == u for nb in host.neighbors(q)):
                    b_term += 1.0
                a_term += 1.0 / (1.0 + d) if math.isfinite(d) else 0.0
        nbs = list(host.neighbors(q))
        k_res = sum(1 for nb in nbs if nb in free)
        occupied = sum(1 for nb in nbs if nb in owner and owner[nb] != v)
        r_term = occupied / max(1, len(nbs)) + (cfg.articulation_penalty if q in articulation else 0.0)
        l_term = len(regions[v]) / mean_target
        return cfg.alpha * a_term + cfg.beta * k_res + cfg.gamma * b_term - cfg.lam * r_term - cfg.mu * l_term

    active = [v for v in range(n_vars) if len(regions[v]) < targets[v]]
    stalled: set[Node] = set()
    while active:
        articulation: set[Qubit] = set()
        if cfg.lam > 0 and cfg.articulation_penalty > 0 and len(free) <= 4000:
            fg = host.subgraph(free)
            articulation = set(nx.articulation_points(fg)) if fg.number_of_nodes() else set()
        for v in list(active):
            if cfg.frontier == "walk":
                cand = [nb for nb in host.neighbors(last[v]) if nb in free]
                if not cand:
                    cand = [nb for q in regions[v] for nb in host.neighbors(q) if nb in free]
            else:
                cand = [nb for q in regions[v] for nb in host.neighbors(q) if nb in free]
            cand = sorted(set(cand))
            if not cand:
                stalled.add(v)
                continue
            scores = [score(q, v, articulation) for q in cand]
            m = max(scores)
            weights = [math.exp((s - m) / max(cfg.tau, 1e-6)) for s in scores]
            q = rng.choices(cand, weights=weights, k=1)[0]
            regions[v].add(q)
            owner[q] = v
            last[v] = q
            free.discard(q)
        active = [v for v in range(n_vars) if len(regions[v]) < targets[v] and v not in stalled]

    chains = {v: frozenset(r) for v, r in regions.items()}
    contact = _quotient(host, chains)
    if required is None:
        logical = contact
    else:
        missing = [e for e in required.edges() if not contact.has_edge(*e)]
        if missing:
            raise PlantedGenerationError(f"seed {seed}: {len(missing)} required edges unrealised: {missing[:5]}")
        logical = nx.Graph()
        logical.add_nodes_from(range(n_vars))
        logical.add_edges_from(required.edges())
    pe = PlantedEmbedding(host=host, logical=logical, chains=chains, contact_graph=contact,
                          mode=mode, seed=seed, config=cfg)
    errs = pe.validate()
    if errs:
        raise RuntimeError(f"generator produced an invalid witness: {errs}")
    return pe


def ink_drop_with_retries(
    host: nx.Graph, n_vars: int, sizes, *, required: nx.Graph, tries: int = 20, seed: int = 0, **kw
) -> tuple[PlantedEmbedding, int]:
    """Target-guided growth with seed retries; returns the witness and the number of failed
    seeds, which is the attrition the meeting asks to be counted rather than hidden."""
    failures = 0
    for k in range(tries):
        try:
            return ink_drop(host, n_vars, sizes, required=required, seed=seed + k, **kw), failures
        except PlantedGenerationError:
            failures += 1
    raise PlantedGenerationError(f"no witness in {tries} seeds")
