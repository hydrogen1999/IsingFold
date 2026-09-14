"""Chain-seam quality samples: which valid chain for one variable gives the best solve
probability, everything else held fixed.

Why this seam (2026-09-08, late): with the frozen-edge constraint enforced, a single-qubit
extension inside a small window rarely changes downstream quality beyond sampling noise, and
windows large enough to matter make the completion space explode. Quality differences live at
the level of whole chains (L-46's two-chain refinement moved p_solve by +0.16), and the Track
B harness's `propose` seam offers whole candidate chains. So:

    state      a valid embedding with the chain of one variable (the focus) removed;
    window     the removed chain's qubits plus the nearest free qubits (cap);
    actions    every connected qubit set in the window, of size at most l_cap, that touches
               the chain of each logical neighbour of the focus (exact enumeration, capped);
    label      p_solve of the full embedding with that candidate (T1 surrogate, best over the
               strength grid), screening reads for all and refinement reads for the top few;
    baselines  the resource-best candidate (fewest qubits, then fewest internal couplers as
               a tie-break) and, when the state came from minorminer, minorminer's own chain.

States come from two sources: planted witnesses (as before) and stock minorminer embeddings
of the same logical graphs, which is the deployment distribution of a rebuild seam.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx

from embedbench.structural import connected_random_graph, GenStats, host_graph, _instance_seed
from embedbench.exact import SearchAborted, connected_supersets
from embedbench.inkdrop import MODES, PlantedGenerationError, ink_drop
from embedbench.planted_ising import PlantedIsing, PlantingError, frustrated_loops
from embedbench.embedding import Embedding, Node, Qubit
from embedbench.surrogate import default_strength_grid, solve_probability_at

_SEED_MOD = 2**32 - 1


@dataclass(frozen=True)
class ChainConfig:
    topology: str = "chimera"
    size: int = 5
    n_vars: int = 22
    chain_size: int = 3
    modes: tuple[str, ...] = tuple(MODES)
    source: str = "both"
    """'witness', 'minorminer', or 'both': where the starting embedding comes from."""
    graph: str = "inkdrop"
    """'inkdrop' quotient graphs, 'random' G(n, m) with `degree`, or 'app' (portfolio 16,
    graph-cut 5x5 and 4x6 with their real coefficients; ground energy exact when the
    problem has at most 22 spins, else the tabu reference)."""
    degree: float = 3.0
    l_cap: int = 4
    max_window_free: int = 14
    max_candidates: int = 60
    max_enum: int = 4000
    samples_per_instance: int = 4
    alpha: float = 0.6
    loop_min: int = 3
    loop_max: int = 8
    num_reads: int = 100
    num_sweeps: int = 200
    n_strengths: int = 4
    refine_reads: int = 400
    refine_top: int = 3
    min_spread: float = 0.05
    """Keep a sample when the best candidate beats the worst by at least this and the
    best beats the second-best by at least two pooled standard errors or the sample has a
    structurally distinct best (so pairwise training still has signal)."""
    difficulty: str = "base"
    defect_qubits: float = 0.0
    defect_couplers: float = 0.0


@dataclass
class ChainStats(GenStats):
    planting_failures: int = 0
    mm_failures: int = 0
    dropped_too_many_candidates: int = 0
    dropped_single_candidate: int = 0
    dropped_noise: int = 0
    resource_disagreements: int = 0
    mm_disagreements: int = 0
    mm_states: int = 0
    sa_calls: int = 0


@dataclass(frozen=True)
class ChainRecord:
    instance_id: str
    source: str
    mode: str
    topology: str
    size: int
    focus: int
    window_nodes: list[int]
    window_edges: list[list[int]]
    frozen: dict[int, list[int]]
    frozen_adjacency: dict[int, list[int]]
    neighbours: list[int]
    candidates: list[list[int]]
    p_solve: list[float]
    best_F: list[float]
    Q: list[int]
    best_index: int
    resource_index: int
    resource_agrees: bool
    original_index: int
    original_agrees: bool
    spread: float
    margin: float
    l_cap: int
    ground_energy: float
    sa_seed: int
    difficulty: str
    n_vars: int
    n_enumerated: int
    focus_h: float = 0.0
    edge_J: list[float] = field(default_factory=list)
    """Coupling of the logical edge to each required neighbour, in `neighbours` order."""
    neighbour_degree: list[int] = field(default_factory=list)
    neighbour_chain_size: list[int] = field(default_factory=list)
    j_scale: float = 1.0
    """max |J| of the problem (couplings are already normalised to it)."""
    stage: list[int] = field(default_factory=list)
    problem: dict | None = None
    """v1.1: the full programmed problem {h, J, e0} and every other chain, so a record can be rescored from its own fields."""
    all_chains: dict | None = None
    """1 = screening estimate (num_reads, seed A); 2 = fresh-seed estimate (refine_reads per
    strength, seed B). Only stage-2 values are compared for best / resource / original."""

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def _s(x):
    return int(x) % _SEED_MOD


def _score(problem, host, chains, e0, grid, reads, sweeps, seed, stats):
    emb = Embedding.from_chains(chains, host, problem)
    if any(not c for c in emb.contacts.values()):
        raise RuntimeError("unrealised logical edge")
    best = (-1.0, 0.0)
    for k, f in enumerate(grid):
        r = solve_probability_at(emb, problem, e0, f, num_reads=reads, num_sweeps=sweeps, seed=_s(seed + 7919 * k))
        stats.sa_calls += 1
        if r.p_solve > best[0]:
            best = (r.p_solve, f)
    return best


def enumerate_chains(host, window: set, must_hit: list[frozenset], l_cap: int, max_enum: int) -> list[frozenset]:
    """All connected qubit sets inside the window (size <= l_cap) that intersect every
    required set. Raises SearchAborted past max_enum enumerated sets."""
    wh = host.subgraph(window)
    out = []
    n = 0
    for s in connected_supersets(wh, frozenset(), set(window), l_cap):
        n += 1
        if n > max_enum:
            raise SearchAborted("too many chains")
        if all(s & r for r in must_hit):
            out.append(s)
    return out


def chain_samples(host, logical, chains: dict, planted, cfg: ChainConfig, rng: random.Random,
                  stats: ChainStats, iid: str, source: str, mode: str):
    problem, e0 = planted.problem, planted.ground_energy
    grid = default_strength_grid(problem, cfg.n_strengths)
    variables = sorted(chains)
    picks = rng.sample(variables, min(cfg.samples_per_instance, len(variables)))
    for focus in picks:
        stats.samples_attempted += 1
        nbrs = sorted(logical.neighbors(focus))
        if not nbrs:
            stats.dropped_no_actions += 1
            continue
        frozen = {v: c for v, c in chains.items() if v != focus}
        blocked = {q for c in frozen.values() for q in c}
        old = set(chains[focus])
        ranked = []
        for q in old:
            for n in host.neighbors(q):
                if n not in blocked and n not in old:
                    ranked.append((1, rng.random(), n))
        # second hop so the window offers alternatives
        first = {n for _, _, n in ranked}
        for q in first:
            for n in host.neighbors(q):
                if n not in blocked and n not in old and n not in first:
                    ranked.append((2, rng.random(), n))
        extra = [n for _, _, n in sorted(set(ranked))[: cfg.max_window_free]]
        window = old | set(extra)
        must_hit = []
        for u in nbrs:
            hit = frozenset(q for q in window if any(host.has_edge(q, t) for t in frozen[u]))
            must_hit.append(hit)
        if any(not h for h in must_hit):
            stats.dropped_no_actions += 1
            continue
        try:
            cands = enumerate_chains(host, window, must_hit, cfg.l_cap, cfg.max_enum)
        except SearchAborted:
            stats.dropped_too_many_candidates += 1
            continue
        if len(cands) < 2:
            stats.dropped_single_candidate += 1
            continue
        # keep the original chain in the set; subsample the rest if needed
        cands = sorted(cands, key=lambda s: (len(s), tuple(sorted(s))))
        orig = frozenset(old)
        if orig not in cands and len(orig) <= cfg.l_cap and all(orig & r for r in must_hit):
            cands.append(orig)
        n_enum = len(cands)
        if len(cands) > cfg.max_candidates:
            keep = set(rng.sample(range(len(cands)), cfg.max_candidates))
            if orig in cands:
                keep.add(cands.index(orig))
            cands = [c for i, c in enumerate(cands) if i in keep]
        sa_seed = int.from_bytes(hashlib.sha256(f"{iid}:{focus}".encode()).digest()[:4], "big") % (2**30)
        scored = []
        for ci, c in enumerate(cands):
            full = dict(frozen); full[focus] = c
            p, f = _score(problem, host, full, e0, grid, cfg.num_reads, cfg.num_sweeps, sa_seed + 1000 * ci, stats)
            scored.append([p, f, ci])
        # stage 2 (fresh seeds, more reads): the screening top, the resource-best and the
        # original chain. Comparisons between those use stage-2 estimates only, so the
        # selection on noisy stage-1 values does not inflate the winner (winner's curse);
        # the residual bias is a max over `refine_top` unbiased estimates.
        Qs = [len(c) for c in cands]
        internal = [host.subgraph(c).number_of_edges() for c in cands]
        r_idx = min(range(len(cands)), key=lambda i: (Qs[i], internal[i], i))
        o_idx = cands.index(orig) if orig in cands else -1
        order1 = sorted(range(len(scored)), key=lambda i: -scored[i][0])
        stage2 = set(order1[: cfg.refine_top]) | {r_idx} | ({o_idx} if o_idx >= 0 else set())
        stage = [1] * len(cands)
        for i in stage2:
            p, f, ci = scored[i]
            full = dict(frozen); full[focus] = cands[ci]
            best2 = 0.0
            for k, f2 in enumerate(grid):
                emb = Embedding.from_chains(full, host, problem)
                r = solve_probability_at(emb, problem, e0, f2, num_reads=cfg.refine_reads, num_sweeps=cfg.num_sweeps,
                                         seed=_s(sa_seed + 1000 * ci + 555 + 97 * k))
                stats.sa_calls += 1
                if r.p_solve > best2:
                    best2, scored[i][1] = r.p_solve, f2
            scored[i][0] = best2
            stage[i] = 2
        ps = [x[0] for x in scored]
        s2 = sorted(stage2, key=lambda i: -ps[i])
        best_i, second_i = s2[0], s2[1]
        best, second = ps[best_i], ps[second_i]
        spread = best - min(ps)
        n_eff = cfg.refine_reads
        se = math.sqrt(max(best * (1 - best), 1e-4) / n_eff + max(second * (1 - second), 1e-4) / n_eff)
        if spread < cfg.min_spread:
            stats.dropped_no_margin += 1
            continue
        margin = best - second
        if margin < 2 * se and spread < 4 * se:
            stats.dropped_noise += 1
            continue
        order = [best_i] + [i for i in sorted(range(len(ps)), key=lambda i: -ps[i]) if i != best_i]
        r_ok = ps[r_idx] >= best - 1e-12
        o_ok = (ps[o_idx] >= best - 1e-12) if o_idx >= 0 else True
        if not r_ok: stats.resource_disagreements += 1
        if source == "minorminer":
            stats.mm_states += 1
            if not o_ok: stats.mm_disagreements += 1
        stats.samples_kept += 1
        stats.margin_kinds["p_solve"] = stats.margin_kinds.get("p_solve", 0) + 1
        fq = {q for c in frozen.values() for q in c}
        yield ChainRecord(
            instance_id=iid, source=source, mode=mode, topology=cfg.topology, size=cfg.size, focus=int(focus),
            window_nodes=sorted(int(q) for q in window),
            window_edges=sorted([min(a, b), max(a, b)] for a, b in host.subgraph(window).edges()),
            frozen={int(v): sorted(int(q) for q in c) for v, c in frozen.items()
                    if any(host.has_edge(q, t) for q in window for t in c)},
            frozen_adjacency={int(q): sorted(int(n) for n in host.neighbors(q) if n in fq)
                              for q in window if any(n in fq for n in host.neighbors(q))},
            neighbours=[int(u) for u in nbrs],
            candidates=[sorted(int(q) for q in c) for c in cands],
            p_solve=[float(p) for p in ps], best_F=[float(x[1]) for x in scored], Q=Qs,
            best_index=int(order[0]), resource_index=int(r_idx), resource_agrees=bool(r_ok),
            original_index=int(o_idx), original_agrees=bool(o_ok),
            spread=float(spread), margin=float(margin), l_cap=cfg.l_cap, ground_energy=float(e0),
            sa_seed=int(sa_seed), difficulty=cfg.difficulty, n_vars=len(variables), n_enumerated=n_enum,
            focus_h=float(problem.h.get(focus, 0.0)),
            edge_J=[float(problem.j.get((min(focus, u), max(focus, u)), 0.0)) for u in nbrs],
            neighbour_degree=[int(logical.degree(u)) for u in nbrs],
            neighbour_chain_size=[len(frozen[u]) for u in nbrs],
            j_scale=float(max(abs(v) for v in problem.j.values()) or 1.0),
            problem={"h": {str(int(k)): float(v) for k, v in problem.h.items()}, "J": [[int(u), int(v), float(w)] for (u, v), w in problem.j.items()], "e0": float(e0)},
            all_chains={str(int(v)): sorted(int(q) for q in c) for v, c in frozen.items()},
            stage=stage,
        )


def _work(args):
    cfg, mode, i, seed = args
    from embedbench.quality_step import defective_host
    host = defective_host(host_graph(cfg.topology, cfg.size), cfg.defect_qubits, cfg.defect_couplers, _instance_seed(seed, mode, i))
    stats = ChainStats()
    gseed = _instance_seed(seed, mode, i)
    rng = random.Random(gseed ^ 0x5DEECE66D)
    try:
        if cfg.graph == "inkdrop":
            pe = ink_drop(host, cfg.n_vars, cfg.chain_size, mode=mode, seed=gseed)
            logical, witness = pe.logical, dict(pe.chains)
            planted = frustrated_loops(logical, alpha=cfg.alpha, min_len=cfg.loop_min, max_len=cfg.loop_max, seed=gseed)
        elif cfg.graph == "app":
            from embedbench.apps import portfolio, graphcut_mrf
            from embedbench.surrogate import logical_ground_state
            if mode == "portfolio16": inst = portfolio(16, seed=i)
            elif mode == "graphcut5x5": inst = graphcut_mrf(5, 5, seed=i)
            else: inst = graphcut_mrf(4, 6, seed=i)
            problem = inst.problem; logical = problem.graph; witness = None
            gs = logical_ground_state(problem, restarts=20, timeout_ms=2000)
            planted = PlantedIsing(problem=problem, spins={}, ground_energy=gs.energy, n_loops=0, alpha=0.0, loop_lengths=())
        else:
            logical = connected_random_graph(cfg.n_vars, int(round(cfg.degree * cfg.n_vars / 2)), gseed)
            witness = None
            planted = frustrated_loops(logical, alpha=cfg.alpha, min_len=cfg.loop_min, max_len=cfg.loop_max, seed=gseed)
    except PlantedGenerationError:
        stats.generation_failures += 1
        return stats, []
    except PlantingError:
        stats.planting_failures += 1
        return stats, []
    stats.instances += 1
    iid = f"{cfg.topology}{cfg.size}-{cfg.graph}-{mode}-{i}-s{gseed}"
    lines = []
    if cfg.source in ("witness", "both") and witness is not None:
        lines += [r.to_json() for r in chain_samples(host, logical, witness, planted, cfg, rng, stats, iid + "-w", "witness", mode)]
    if cfg.source in ("minorminer", "both"):
        import minorminer
        isolated = [v for v in logical.nodes() if logical.degree(v) == 0]
        emb = minorminer.find_embedding(list(logical.edges()), list(host.edges()), random_seed=gseed % (2**31), tries=10)
        if emb and set(emb) == set(logical.nodes()) - set(isolated):
            ch = {v: frozenset(c) for v, c in emb.items()}
            used = {q for c in ch.values() for q in c}
            fr = iter(sorted(set(host.nodes) - used))
            for v in isolated:
                ch[v] = frozenset([next(fr)])
            lines += [r.to_json() for r in chain_samples(host, logical, ch, planted, cfg, rng, stats, iid + "-m", "minorminer", mode)]
        else:
            stats.mm_failures += 1
    return stats, lines


def _merge(into, part):
    for k, v in asdict(part).items():
        if k == "margin_kinds":
            for kk, vv in v.items():
                into.margin_kinds[kk] = into.margin_kinds.get(kk, 0) + vv
        else:
            setattr(into, k, getattr(into, k) + v)


def generate_chain(cfg: ChainConfig, n_instances: int, seed: int, out: Path | None, jobs: int = 1) -> ChainStats:
    modes = cfg.modes if cfg.graph == "inkdrop" else (("portfolio16", "graphcut5x5", "graphcut4x6") if cfg.graph == "app" else ("random",))
    units = [(cfg, mode, i, seed) for mode in modes for i in range(n_instances)]
    stats = ChainStats()
    fh = open(out, "w") if out else None
    try:
        if jobs > 1:
            from multiprocessing import Pool
            with Pool(jobs) as pool:
                for part, lines in pool.imap(_work, units, chunksize=1):
                    _merge(stats, part)
                    if fh: fh.write("".join(l + "\n" for l in lines)); fh.flush()
        else:
            for u in units:
                part, lines = _work(u)
                _merge(stats, part)
                if fh: fh.write("".join(l + "\n" for l in lines)); fh.flush()
    finally:
        if fh: fh.close()
    if out:
        digest = hashlib.sha256(Path(out).read_bytes()).hexdigest()
        Path(str(out) + ".manifest.json").write_text(json.dumps(
            {"config": asdict(cfg), "n_instances": n_instances, "seed": seed, "jobs": jobs,
             "objective": "chain seam: max_F p_solve (T1) per candidate chain", "stats": asdict(stats),
             "sha256": digest, "file": str(out)}, indent=1))
    return stats
