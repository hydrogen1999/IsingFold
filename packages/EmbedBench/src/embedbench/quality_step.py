"""Decision samples labelled by downstream solution quality (decision of 2026-09-08 evening).

Same construction as `dataset.py` (planted witness, frozen context, window, focus variable,
single-qubit actions) with two changes:

1. the witness's logical graph carries a frustrated-loop planted Ising problem with a known
   ground energy (`planted_ising.py`), so solve probability is exact-referenced;
2. an action's value is the best solve probability over *all* feasible completions of the
   window, each completion joined with the frozen chains into a full embedding and scored by
   the fixed-schedule SA surrogate with majority decoding, best over a chain-strength grid
   (`surrogate.solve_probability_at`, the T1 objective of L-33 onward).

The label is therefore V*(S_t, a) = max over completions of max over F of p_solve, estimated
from `num_reads` anneals per (completion, F) with seeds fixed by the sample, so it is
reproducible to the bit. Sampling noise is handled by the margin rule: a sample is kept only
when the best action's p_solve exceeds the second-best by at least `min_margin` and by at
least two pooled binomial standard errors.

Each record also stores what the *resource* objective would choose (the action whose best
structural completion has fewest qubits, then shortest chain), so the corpus reports how
often resource-first and quality-first disagree; that number is the point of the redo.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx

from embedbench.structural import GenStats, _chain_prefix, frozen_adjacency, frozen_requirements, greedy_local_choice, host_graph, _instance_seed
from embedbench.exact import SearchAborted, all_completions, best_completion
from embedbench.inkdrop import MODES, PlantedGenerationError, ink_drop
from embedbench.planted_ising import PlantingError, frustrated_loops
from embedbench.embedding import Embedding, Node, Qubit
from embedbench.surrogate import default_strength_grid, solve_probability_at


@dataclass(frozen=True)
class QConfig:
    topology: str = "chimera"
    size: int = 4
    n_vars: int = 16
    chain_size: int = 3
    modes: tuple[str, ...] = tuple(MODES)
    k_in_play: int = 2
    radius: int = 1
    l_cap: int = 3
    max_window_free: int = 10
    max_completions: int = 24
    samples_per_instance: int = 4
    prefix_keep_prob: float = 0.5
    max_actions: int = 8
    alpha: float = 0.3
    loop_min: int = 3
    loop_max: int = 8
    num_reads: int = 100
    num_sweeps: int = 200
    n_strengths: int = 4
    refine_reads: int = 400
    min_margin: float = 0.05
    debug: bool = False
    top_m: int = 0
    """When > 0, only the `top_m` structurally best completions (fewest qubits, then shortest
    longest chain) of each action are scored; V*(a) is the max over those. This is the value
    of the action under a resource-optimal classical finisher, which is how the model is
    deployed (it chooses the step, exact completion finishes). 0 scores every completion,
    which is exact but explodes with window size."""
    max_enum: int = 3000
    """Structural enumeration cap when top_m > 0 (cheap; no SA involved)."""
    # hardness axes (2026-09-08, "add hard cases")
    defect_qubits: float = 0.0
    """Fraction of host qubits removed by a deterministic mask (L-52 style)."""
    defect_couplers: float = 0.0
    fill: float = 0.0
    """If > 0, n_vars is set so the witness occupies about this fraction of the host's
    qubits (capacity pressure: routes must squeeze past other chains)."""
    difficulty: str = "base"
    """Free-text tag written into every record for stratified evaluation."""


@dataclass
class QStats(GenStats):
    planting_failures: int = 0
    dropped_too_many_completions: int = 0
    dropped_noise: int = 0
    resource_disagreements: int = 0
    sa_calls: int = 0


@dataclass(frozen=True)
class QRecord:
    instance_id: str
    mode: str
    topology: str
    size: int
    focus: int
    in_play: list[int]
    window_nodes: list[int]
    window_edges: list[list[int]]
    logical_edges: list[list[int]]
    cores: dict[int, list[int]]
    frozen: dict[int, list[int]]
    actions: list[list[int]]
    values: list[list[float]]
    """[p_solve, -Q, -L_max] of the best completion per action (p_solve decides)."""
    best_action: list[int]
    margin_kind: str
    margin: float
    greedy_action: list[int]
    greedy_agrees: bool
    resource_action: list[int]
    resource_agrees: bool
    radius: int
    l_cap: int
    exact_nodes: int
    witness_outcome: list[int]
    witness_p_solve: float
    n_completions: list[int]
    best_F: list[float]
    ground_energy: float
    sa_seed: int
    difficulty: str = "base"
    alpha: float = 0.0
    n_vars: int = 0
    host_nodes: int = 0
    frozen_edges: list[list[int]] = field(default_factory=list)
    frozen_adjacency: dict[int, list[int]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


_SEED_MOD = 2**32 - 1


def _s(x: int) -> int:
    return int(x) % _SEED_MOD


def _score_full(problem, host, full_chains, e0, grid, reads, sweeps, seed, stats):
    emb = Embedding.from_chains(full_chains, host, problem)
    if any(not c for c in emb.contacts.values()):
        raise RuntimeError("scoring an embedding with an unrealised logical edge")
    best = (-1.0, 0.0)
    for k, f in enumerate(grid):
        r = solve_probability_at(emb, problem, e0, f, num_reads=reads, num_sweeps=sweeps, seed=_s(seed + 7919 * k))
        stats.sa_calls += 1
        if r.p_solve > best[0]:
            best = (r.p_solve, f)
    return best


def q_samples_from_witness(pe, planted, cfg: QConfig, rng: random.Random, stats: QStats, iid: str):
    host, logical, chains = pe.host, pe.logical, dict(pe.chains)
    problem, e0 = planted.problem, planted.ground_energy
    grid = default_strength_grid(problem, cfg.n_strengths)
    variables = sorted(chains)
    for _ in range(cfg.samples_per_instance):
        stats.samples_attempted += 1
        focus = rng.choice(variables)
        nbrs = list(logical.neighbors(focus))
        rng.shuffle(nbrs)
        in_play = [focus] + nbrs[: cfg.k_in_play - 1]
        others = [v for v in variables if v not in in_play]
        rng.shuffle(others)
        while len(in_play) < cfg.k_in_play and others:
            in_play.append(others.pop())
        frozen = {v: chains[v] for v in variables if v not in in_play}
        blocked = {q for c in frozen.values() for q in c}
        cores: dict[Node, frozenset[Qubit]] = {}
        cores[focus] = _chain_prefix(host, chains[focus], rng.randrange(0, len(chains[focus])), rng)
        for v in in_play[1:]:
            cores[v] = chains[v] if rng.random() < cfg.prefix_keep_prob else _chain_prefix(
                host, chains[v], rng.randrange(0, len(chains[v]) + 1), rng)
        seed_set = set().union(*(chains[v] for v in in_play))
        window, frontier, ranked = set(seed_set), set(seed_set), []
        for hop in range(1, cfg.radius + 1):
            nxt = set()
            for q in frontier:
                for n in host.neighbors(q):
                    if n not in blocked and n not in window and n not in nxt:
                        nxt.add(n)
            for n in nxt:
                ranked.append((hop, rng.random(), n))
            window |= nxt
            frontier = nxt
        core_qubits = {q for c in cores.values() for q in c}
        witness_free = seed_set - core_qubits
        if len(witness_free) > cfg.max_window_free:
            stats.dropped_window_too_large += 1
            continue
        extra = [n for _, _, n in sorted(ranked)[: cfg.max_window_free - len(witness_free)]]
        window = seed_set | set(extra)
        free_in_window = window - core_qubits
        wh = host.subgraph(window).copy()
        wl = logical.subgraph(in_play).copy()
        must_hit, frozen_edges = frozen_requirements(host, logical, in_play, frozen, window)
        core = cores[focus]
        if core:
            cand = sorted({n for q in core for n in wh.neighbors(q) if n in free_in_window})
        else:
            placed_nb = [u for u in wl.neighbors(focus) if cores.get(u)]
            near = {n for u in placed_nb for q in cores[u] for n in wh.neighbors(q) if n in free_in_window}
            cand = sorted(near) if placed_nb else sorted(free_in_window)
        if len(cand) < 2:
            stats.dropped_no_actions += 1
            continue
        if len(cand) > cfg.max_actions:
            cand = rng.sample(cand, cfg.max_actions)
        actions = [(focus, q) for q in cand]
        sa_seed = int.from_bytes(hashlib.sha256(f"{iid}:{stats.samples_attempted}".encode()).digest()[:4], "big") % (2**30)

        values, n_comps, best_fs, struct_best = [], [], [], []
        try:
            for ai, (v, q) in enumerate(actions):
                c2 = dict(cores); c2[v] = cores[v] | {q}
                if cfg.top_m > 0:
                    comps = all_completions(wh, wl, c2, l_cap=cfg.l_cap, max_count=cfg.max_enum, must_hit=must_hit)
                    comps.sort(key=lambda c: (sum(len(x) for x in c.values()), max(len(x) for x in c.values()),
                                              tuple(sorted((k, tuple(sorted(x))) for k, x in c.items()))))
                    comps = comps[: cfg.top_m]
                else:
                    comps = all_completions(wh, wl, c2, l_cap=cfg.l_cap, max_count=cfg.max_completions, must_hit=must_hit)
                if not comps:
                    values.append([0.0, 0, 0]); n_comps.append(0); best_fs.append(0.0)
                    struct_best.append((0, 0, 0)); continue
                # screening pass
                scored = []
                for ci, comp in enumerate(comps):
                    full = dict(frozen); full.update(comp)
                    p, f = _score_full(problem, host, full, e0, grid, cfg.num_reads, cfg.num_sweeps,
                                       sa_seed + 100_000 * ai + 1000 * ci, stats)
                    scored.append((p, f, comp))
                scored.sort(key=lambda t: -t[0])
                # refine the top two completions with more reads
                refined = []
                for p, f, comp in scored[:2]:
                    full = dict(frozen); full.update(comp)
                    emb = Embedding.from_chains(full, host, problem)
                    r = solve_probability_at(emb, problem, e0, f, num_reads=cfg.refine_reads, num_sweeps=cfg.num_sweeps,
                                             seed=_s(sa_seed + 100_000 * ai + 555))
                    stats.sa_calls += 1
                    p2 = (p * cfg.num_reads + r.p_solve * cfg.refine_reads) / (cfg.num_reads + cfg.refine_reads)
                    refined.append((p2, f, comp))
                p, f, comp = max(refined, key=lambda t: t[0])
                qn = sum(len(c) for c in comp.values()); lm = max(len(c) for c in comp.values())
                values.append([float(p), -qn, -lm]); n_comps.append(len(comps)); best_fs.append(float(f))
                sb, _ = best_completion(wh, wl, c2, l_cap=cfg.l_cap, must_hit=must_hit)
                struct_best.append(sb)
        except SearchAborted:
            stats.dropped_too_many_completions += 1
            continue
        order = sorted(range(len(actions)), key=lambda k: values[k], reverse=True)
        p1, p2 = values[order[0]][0], values[order[1]][0]
        if cfg.debug:
            import sys
            print(f"DEBUG {iid} focus={focus} p={[round(v[0],3) for v in values]} Q={[v[1] for v in values]} ncomp={n_comps}", file=sys.stderr)
        if values[order[0]][1] == 0 and values[order[0]][2] == 0 and p1 == 0.0:
            stats.dropped_no_margin += 1
            continue
        margin = p1 - p2
        n_eff = cfg.num_reads + cfg.refine_reads
        se = math.sqrt(max(p1 * (1 - p1), 1e-4) / n_eff + max(p2 * (1 - p2), 1e-4) / n_eff)
        if margin < cfg.min_margin:
            stats.dropped_no_margin += 1
            continue
        if margin < 2 * se:
            stats.dropped_noise += 1
            continue
        greedy = greedy_local_choice(wh, wl, cores, focus, actions)
        g_ok = values[actions.index(greedy)][0] >= p1 - 1e-12
        r_idx = max(range(len(actions)), key=lambda k: struct_best[k])
        r_ok = values[r_idx][0] >= p1 - 1e-12
        if not g_ok: stats.greedy_disagreements += 1
        if not r_ok: stats.resource_disagreements += 1
        stats.samples_kept += 1
        stats.margin_kinds["p_solve"] = stats.margin_kinds.get("p_solve", 0) + 1
        wfull = dict(frozen); wfull.update({v: chains[v] for v in in_play})
        wp, _ = _score_full(problem, host, wfull, e0, grid, cfg.num_reads, cfg.num_sweeps, sa_seed + 99, stats)
        yield QRecord(
            instance_id=iid, mode=pe.mode, topology=cfg.topology, size=cfg.size, focus=int(focus),
            in_play=[int(v) for v in in_play], window_nodes=sorted(int(q) for q in window),
            window_edges=sorted([int(a), int(b)] if a < b else [int(b), int(a)] for a, b in wh.edges()),
            logical_edges=sorted([int(a), int(b)] if a < b else [int(b), int(a)] for a, b in wl.edges()),
            cores={int(v): sorted(int(q) for q in c) for v, c in cores.items()},
            frozen={int(v): sorted(int(q) for q in c) for v, c in frozen.items()},
            actions=[[int(v), int(q)] for v, q in actions], values=values,
            best_action=[int(actions[order[0]][0]), int(actions[order[0]][1])],
            margin_kind="p_solve", margin=float(margin),
            greedy_action=[int(greedy[0]), int(greedy[1])], greedy_agrees=bool(g_ok),
            resource_action=[int(actions[r_idx][0]), int(actions[r_idx][1])], resource_agrees=bool(r_ok),
            radius=cfg.radius, l_cap=cfg.l_cap, exact_nodes=0,
            witness_outcome=[1, -sum(len(c) for c in wfull.values()), -max(len(c) for c in wfull.values())],
            witness_p_solve=float(wp), n_completions=n_comps, best_F=best_fs,
            ground_energy=float(e0), sa_seed=int(sa_seed),
            difficulty=cfg.difficulty, alpha=cfg.alpha, n_vars=len(variables), host_nodes=host.number_of_nodes(),
            frozen_edges=frozen_edges, frozen_adjacency=frozen_adjacency(host, frozen, window),
        )


def defective_host(host: nx.Graph, q_frac: float, c_frac: float, seed: int) -> nx.Graph:
    """Remove a deterministic random fraction of qubits and couplers; keep the largest
    connected component so every remaining qubit is reachable."""
    if q_frac <= 0 and c_frac <= 0:
        return host
    rng = random.Random(seed)
    h = host.copy()
    qs = sorted(h.nodes)
    rng.shuffle(qs)
    h.remove_nodes_from(qs[: int(q_frac * len(qs))])
    es = sorted(tuple(sorted(e)) for e in h.edges)
    rng.shuffle(es)
    h.remove_edges_from(es[: int(c_frac * len(es))])
    if h.number_of_nodes() and not nx.is_connected(h):
        h = h.subgraph(max(nx.connected_components(h), key=len)).copy()
    return h


def _work(args):
    cfg, mode, i, seed = args
    host = host_graph(cfg.topology, cfg.size)
    stats = QStats()
    gseed = _instance_seed(seed, mode, i)
    host = defective_host(host, cfg.defect_qubits, cfg.defect_couplers, gseed)
    n_vars = cfg.n_vars if cfg.fill <= 0 else max(4, int(cfg.fill * host.number_of_nodes() / cfg.chain_size))
    try:
        pe = ink_drop(host, n_vars, cfg.chain_size, mode=mode, seed=gseed)
    except PlantedGenerationError:
        stats.generation_failures += 1
        return stats, []
    try:
        planted = frustrated_loops(pe.logical, alpha=cfg.alpha, min_len=cfg.loop_min, max_len=cfg.loop_max, seed=gseed)
    except PlantingError:
        stats.planting_failures += 1
        return stats, []
    stats.instances += 1
    iid = f"{cfg.topology}{cfg.size}-{mode}-{i}-s{gseed}"
    rng = random.Random(gseed ^ 0x5DEECE66D)
    lines = [r.to_json() for r in q_samples_from_witness(pe, planted, cfg, rng, stats, iid)]
    return stats, lines


def _merge(into: QStats, part: QStats):
    for k, v in asdict(part).items():
        if k == "margin_kinds":
            for kk, vv in v.items():
                into.margin_kinds[kk] = into.margin_kinds.get(kk, 0) + vv
        else:
            setattr(into, k, getattr(into, k) + v)


def generate_q(cfg: QConfig, n_instances: int, seed: int, out: Path | None, jobs: int = 1) -> QStats:
    units = [(cfg, mode, i, seed) for mode in cfg.modes for i in range(n_instances)]
    stats = QStats()
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
            {"config": asdict(cfg), "n_instances_per_mode": n_instances, "seed": seed, "jobs": jobs,
             "objective": "max_F p_solve, fixed-schedule SA, majority decode (T1)" + (f"; top-{cfg.top_m} structural completions" if cfg.top_m else "; all completions"),
             "stats": asdict(stats), "sha256": digest, "file": str(out)}, indent=1))
    return stats
