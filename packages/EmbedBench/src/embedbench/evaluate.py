"""End-to-end evaluation of an embedder under the benchmark's objective.

An embedder is any callable ``embed(logical: nx.Graph, host: nx.Graph, seed: int) -> dict``
mapping each logical node to an iterable of host qubits (or an empty dict on failure).
`evaluate_embedder` runs it on a suite of instances (planted-Ising random graphs, planted
quotient graphs, application instances) and reports feasibility, qubits, longest chain and
the surrogate solve probability (max over a chain-strength grid, fixed-schedule simulated
annealing, majority-vote decoding), paired against stock minorminer on the same instances.

`decision_top1` scores a decision scorer (``score(record) -> list[float]`` over the record's
actions) on a structural or chain-seam dataset by top-1 / regret against the stored labels.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Callable

import networkx as nx

from embedbench.embedding import Embedding, LogicalProblem
from embedbench.inkdrop import MODES, PlantedGenerationError, ink_drop
from embedbench.objective import outcome_of
from embedbench.planted_ising import PlantingError, frustrated_loops
from embedbench.structural import connected_random_graph, _instance_seed, host_graph
from embedbench.surrogate import default_strength_grid, solve_probability_at

Embedder = Callable[[nx.Graph, nx.Graph, int], dict]


@dataclass(frozen=True)
class EvalConfig:
    topology: str = "chimera"
    size: int = 5
    source: str = "random"
    """'random' G(n, m); 'inkdrop' planted quotient graphs; 'app' portfolio / graph-cut."""
    n_vars: int = 22
    degree: float = 3.0
    chain_size: int = 3
    alpha: float = 0.6
    n: int = 30
    reads: int = 800
    n_strengths: int = 5
    seed: int = 12345
    ice: float = 0.0
    """> 0: also score every embedding under control-error noise of this relative size
    (Gaussian on every programmed h and J, one programming per call), field p_solve_ice."""
    window_multiple: float = 4.0
    """Ink-drop planting is confined to a connected window this many times the size of the
    qubits it plants, so a big host does not scatter the droplets into a problem with no edges.
    0 plants on the whole host; a host smaller than the window is used whole."""
    modes: str = ""
    """Comma list restricting which instance families of `source` are generated, so a long run
    can be sharded one family to a process. Empty runs them all."""
    sweeps: int = 200
    """Annealing sweeps per read in the scoring surrogate. A claim about which embedding
    anneals better has to survive a change of this and of `sampler`, or it is a claim about one
    solver's dynamics rather than about the embedding."""
    sampler: str = "sa"
    """'sa' simulated annealing, 'tabu' tabu search, 'sd' steepest descent from random
    states. All three take independent reads, so p_solve means the same thing in each."""
    objective: str = "psolve"
    """'psolve' (fraction of reads at the ground state, best over the strength grid) or
    'residual' (minus the mean relative residual energy of the decoded reads, best over the
    grid; the scale tier's objective, where p_solve is at the floor). The row field is
    always named p_solve so downstream tooling is unchanged; with 'residual' it holds
    -mean_residual (higher is better)."""


def p_solve(problem: LogicalProblem, host: nx.Graph, chains: dict, e0: float, reads: int, seed: int, n_f: int = 5, objective: str = "psolve", sweeps: int = 200, sampler: str = "sa") -> tuple[float, float]:
    emb = Embedding.from_chains(chains, host, problem)
    best = (0.0, 0.0) if objective == "psolve" else (-float("inf"), 0.0)
    for k, f in enumerate(default_strength_grid(problem, n_f)):
        r = solve_probability_at(emb, problem, e0, f, num_reads=reads, num_sweeps=sweeps, seed=(seed + 31 * k) % (2**31), sampler=sampler)
        p = r.p_solve if objective == "psolve" else -r.mean_residual
        if p > best[0]:
            best = (p, f)
    return best


def stock_minorminer(logical: nx.Graph, host: nx.Graph, seed: int) -> dict:
    import minorminer
    isolated = [v for v in logical.nodes() if logical.degree(v) == 0]
    emb = minorminer.find_embedding(list(logical.edges()), list(host.edges()), random_seed=seed % (2**31), tries=10)
    if not emb or set(emb) != set(logical.nodes()) - set(isolated):
        return {}
    ch = {v: frozenset(c) for v, c in emb.items()}
    used = {q for c in ch.values() for q in c}
    fr = iter(sorted(set(host.nodes) - used))
    for v in isolated:
        ch[v] = frozenset([next(fr)])
    return ch


def _host_window(host: nx.Graph, size: int, seed: int) -> set:
    """A connected patch of `size` qubits, grown breadth-first from one of them."""
    import random
    rng = random.Random(seed)
    start = rng.choice(sorted(host.nodes()))
    seen = {start}
    frontier = [start]
    while len(seen) < size and frontier:
        following = []
        for q in frontier:
            for n in host.neighbors(q):
                if n not in seen:
                    seen.add(n)
                    following.append(n)
                    if len(seen) >= size:
                        return seen
        frontier = following
    return seen


def instances(cfg: EvalConfig):
    host = host_graph(cfg.topology, cfg.size)
    if cfg.source == "inkdrop": modes = list(MODES)
    elif cfg.source == "app": modes = ["portfolio16", "graphcut5x5", "graphcut4x6"]
    elif cfg.source == "app_large": modes = ["portfolio48", "graphcut10x10", "graphcut14x14"]
    elif cfg.source == "app_large2": modes = ["portfolio24", "portfolio32", "graphcut10x10hard", "graphcut14x14hard"]
    else: modes = ["random"]
    if cfg.modes:
        wanted = [m.strip() for m in cfg.modes.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in modes]
        if unknown:
            raise ValueError(f"{cfg.source} has no families {unknown}; it has {modes}")
        modes = wanted
    for mode in modes:
        for i in range(cfg.n):
            gseed = _instance_seed(cfg.seed, mode, i)
            try:
                if cfg.source == "inkdrop":
                    # Free growth lets the droplets settle wherever they like, and on a host
                    # the size of real hardware they settle far apart and never touch: on
                    # Pegasus 16, sixteen droplets of chain size four produce a logical graph
                    # with two edges, which is not a problem at all. The planting is therefore
                    # confined to a connected window a few times the size of what it plants,
                    # which is how a real problem occupies a real processor, while the embedder
                    # still sees the whole host. On a host no larger than the window this is
                    # exactly the old behaviour.
                    blocked = None
                    if cfg.window_multiple > 0:
                        room = int(cfg.window_multiple * cfg.n_vars * cfg.chain_size)
                        if room < host.number_of_nodes():
                            blocked = set(host.nodes()) - _host_window(host, room, gseed)
                    logical = ink_drop(host, cfg.n_vars, cfg.chain_size, mode=mode, seed=gseed,
                                       blocked=blocked).logical
                    pl = frustrated_loops(logical, alpha=cfg.alpha, seed=gseed); problem, e0 = pl.problem, pl.ground_energy
                elif cfg.source in ("app", "app_large", "app_large2"):
                    from embedbench.apps import graphcut_mrf, portfolio
                    from embedbench.surrogate import logical_ground_state
                    gens = {"portfolio16": lambda: portfolio(16, seed=i), "graphcut5x5": lambda: graphcut_mrf(5, 5, seed=i), "graphcut4x6": lambda: graphcut_mrf(4, 6, seed=i),
                            "portfolio48": lambda: portfolio(48, seed=i), "graphcut10x10": lambda: graphcut_mrf(10, 10, seed=i), "graphcut14x14": lambda: graphcut_mrf(14, 14, seed=i),
                            "portfolio24": lambda: portfolio(24, seed=i), "portfolio32": lambda: portfolio(32, seed=i),
                            "graphcut10x10hard": lambda: graphcut_mrf(10, 10, seed=i, noise=2.0), "graphcut14x14hard": lambda: graphcut_mrf(14, 14, seed=i, noise=2.0)}
                    inst = gens[mode]()
                    problem = inst.problem; logical = problem.graph
                    e0 = logical_ground_state(problem, restarts=20, timeout_ms=(2000 if cfg.source == "app" else 20000)).energy
                else:
                    logical = connected_random_graph(cfg.n_vars, int(round(cfg.degree * cfg.n_vars / 2)), gseed)
                    pl = frustrated_loops(logical, alpha=cfg.alpha, seed=gseed); problem, e0 = pl.problem, pl.ground_energy
            except (PlantedGenerationError, PlantingError):
                continue
            yield f"{cfg.topology}{cfg.size}-{mode}-{i}", host, logical, problem, e0, gseed


def evaluate_embedder(embed, cfg: EvalConfig, reference: Embedder = stock_minorminer, out=None, problem_aware: bool = False) -> dict:
    """`embed` is a callable (logical, host, seed) -> chains, or, with `problem_aware=True`, a
    factory (problem, e0) -> such a callable (the objective-guided embedder needs the
    problem's fields and couplings, which is the point of the benchmark)."""
    rows = []
    fh = open(out, "w") if out else None
    for iid, host, logical, problem, e0, gseed in instances(cfg):
        row = {"instance": iid, "n_vars": logical.number_of_nodes(), "n_edges": logical.number_of_edges()}
        emb_fn = embed(problem, e0) if problem_aware else embed
        for name, fn in (("embedder", emb_fn), ("reference", reference)):
            import time as _time
            _t0 = _time.perf_counter()
            ch = fn(logical, host, gseed)
            _secs = _time.perf_counter() - _t0
            ch = {k: frozenset(v) for k, v in ch.items()} if ch else {}
            ok = bool(ch) and set(ch) == set(logical.nodes()) and outcome_of(ch, logical, host)[0] == 1
            d = {"feasible": ok, "seconds": round(_secs, 3)}
            if ok:
                p, f = p_solve(problem, host, ch, e0, cfg.reads, gseed + 1, cfg.n_strengths, cfg.objective, cfg.sweeps, cfg.sampler)
                d.update(Q=sum(len(c) for c in ch.values()), L=max(len(c) for c in ch.values()), p_solve=p, F=f)
                if cfg.ice > 0:
                    emb = Embedding.from_chains(ch, host, problem)
                    d["p_solve_ice"] = max(solve_probability_at(emb, problem, e0, F, num_reads=cfg.reads, num_sweeps=200, seed=(gseed + 5 + 31 * k) % (2**31), ice=cfg.ice).p_solve
                                           for k, F in enumerate(default_strength_grid(problem, cfg.n_strengths)))
            row[name] = d
        rows.append(row)
        if fh: fh.write(json.dumps(row) + "\n"); fh.flush()
    if fh: fh.close()
    return summarise(rows)


def summarise(rows: list[dict]) -> dict:
    out = {"n": len(rows)}
    for name in ("embedder", "reference"):
        ok = [r[name] for r in rows if r[name]["feasible"]]
        out[name] = {"feasible": len(ok) / max(1, len(rows)),
                     "p_solve": statistics.mean(x["p_solve"] for x in ok) if ok else None,
                     "Q": statistics.mean(x["Q"] for x in ok) if ok else None,
                     "L": statistics.mean(x["L"] for x in ok) if ok else None}
    d = [r["embedder"]["p_solve"] - r["reference"]["p_solve"] for r in rows if r["embedder"]["feasible"] and r["reference"]["feasible"]]
    out["paired_dp"] = {"n": len(d), "mean": statistics.mean(d) if d else None, "wins": sum(1 for x in d if x > 0), "losses": sum(1 for x in d if x < 0)}
    return out


def decision_top1(score: Callable[[dict], list[float]], path: str, kind: str = "structural") -> dict:
    """Top-1 and regret of a scorer on a stored dataset. structural: values are lexicographic
    tuples, top-1 means the arg-max is a certified best action; chain: values are p_solve,
    regret against the best of the reliable (stage-2) candidates."""
    n = top = 0; regret = []
    for ln in open(path):
        r = json.loads(ln); s = score(r); n += 1
        if kind == "structural":
            vals = [tuple(v) for v in r["values"]]; best = max(vals)
            top += vals[max(range(len(s)), key=lambda i: s[i])] == best
        else:
            rel = [i for i in range(len(r["p_solve"])) if r.get("stage", [1] * len(s))[i] == 2] or list(range(len(s)))
            best = max(r["p_solve"][i] for i in rel); k = max(rel, key=lambda i: s[i])
            regret.append(best - r["p_solve"][k]); top += r["p_solve"][k] >= best - 0.02
    return {"n": n, "top1": top / max(1, n), "regret": statistics.mean(regret) if regret else None}


def evaluate_objective_embedder(cfg: EvalConfig, scorer_paths: list[str], policy: str = "ens_ucb", budget: int = 100, out=None, **kw) -> dict:
    """Convenience: the packaged ObjectiveEmbedder against stock minorminer on `cfg`."""
    from embedbench.models_hetero import load_scorer
    from embedbench.objective_embedder import ObjectiveEmbedder
    scorers = [load_scorer(p) for p in scorer_paths]
    return evaluate_embedder(lambda problem, e0: ObjectiveEmbedder(problem, e0, budget=budget, scorers=scorers, policy=policy, **kw),
                             cfg, out=out, problem_aware=True)


def _inflate(chains: dict, host: nx.Graph, mult: float, seed: int) -> dict:
    """Control for the qubit price: lengthen stock chains with random adjacent free qubits
    until the total reaches `mult` times the stock count. Chains stay connected and
    disjoint, so the embedding stays valid; only the qubit count changes."""
    import random
    rng = random.Random(seed)
    ch = {v: set(c) for v, c in chains.items()}
    used = {q for c in ch.values() for q in c}
    target = int(round(mult * len(used)))
    vars_ = list(ch)
    stalled = 0
    while len(used) < target and stalled < 50 * len(vars_):
        v = rng.choice(vars_)
        free = [n for q in ch[v] for n in host.neighbors(q) if n not in used]
        if not free:
            stalled += 1; continue
        q = rng.choice(free); ch[v].add(q); used.add(q); stalled = 0
    return {v: frozenset(c) for v, c in ch.items()}


def stock_inflate_110(logical, host, seed): ch = stock_minorminer(logical, host, seed); return _inflate(ch, host, 1.10, seed) if ch else ch
def stock_inflate_125(logical, host, seed): ch = stock_minorminer(logical, host, seed); return _inflate(ch, host, 1.25, seed) if ch else ch
def stock_inflate_150(logical, host, seed): ch = stock_minorminer(logical, host, seed); return _inflate(ch, host, 1.50, seed) if ch else ch


# ---- reference embedders for the matched-compute comparison (L-144) ----

def _finish(emb, logical, host):
    isolated = [v for v in logical.nodes() if logical.degree(v) == 0]
    if not emb or set(emb) != set(logical.nodes()) - set(isolated):
        return {}
    ch = {v: frozenset(c) for v, c in emb.items()}
    used = {q for c in ch.values() for q in c}; fr = iter(sorted(set(host.nodes) - used))
    for v in isolated: ch[v] = frozenset([next(fr)])
    return ch


def mm_tries100(logical, host, seed):
    """minorminer with tries=100 (ten times the stock call): the same heuristic given more time."""
    import minorminer
    return _finish(minorminer.find_embedding(list(logical.edges()), list(host.edges()), random_seed=seed % (2**31), tries=100), logical, host)


def mm_patient(logical, host, seed):
    """minorminer with tries=50 and chain-length patience raised (its own chain-shortening
    phase given more room)."""
    import minorminer
    return _finish(minorminer.find_embedding(list(logical.edges()), list(host.edges()), random_seed=seed % (2**31), tries=50, chainlength_patience=100, max_no_improvement=100), logical, host)


def _mm_best_of_n_resource(logical, host, seed, n: int):
    """`n` stock calls, keep the one with the shortest longest chain, then fewest qubits.

    This is the baseline that decides whether an objective-guided search is needed at all: it
    is the field's own selection rule given as much wall clock as the search spends, and it
    spends no reads doing it. A stock call on these instances costs about 13 milliseconds, so
    n = 1000 is roughly a quarter of the search's budget and n = 8000 matches it.
    """
    import minorminer
    best = None
    for k in range(n):
        ch = _finish(minorminer.find_embedding(list(logical.edges()), list(host.edges()), random_seed=(seed + 17 * k) % (2**31), tries=10), logical, host)
        if not ch: continue
        key = (max(len(c) for c in ch.values()), sum(len(c) for c in ch.values()))
        if best is None or key < best[0]: best = (key, ch)
    return best[1] if best else {}


def mm_best_of_100_resource(logical, host, seed): return _mm_best_of_n_resource(logical, host, seed, 100)
def mm_best_of_1000_resource(logical, host, seed): return _mm_best_of_n_resource(logical, host, seed, 1000)
def mm_best_of_8000_resource(logical, host, seed): return _mm_best_of_n_resource(logical, host, seed, 8000)


def mm_best_of_10_resource(logical, host, seed):
    """Ten stock calls, keep the one with the shortest longest chain, then fewest qubits
    (no evaluation, the free selection rule)."""
    import minorminer
    best = None
    for k in range(10):
        ch = _finish(minorminer.find_embedding(list(logical.edges()), list(host.edges()), random_seed=(seed + 17 * k) % (2**31), tries=10), logical, host)
        if not ch: continue
        key = (max(len(c) for c in ch.values()), sum(len(c) for c in ch.values()))
        if best is None or key < best[0]: best = (key, ch)
    return best[1] if best else {}


def mm_layout(logical, host, seed):
    """Layout-aware minorminer (minorminer.layout.find_embedding: place by a spring layout,
    then route)."""
    try:
        from minorminer.layout import find_embedding as lfe
        emb = lfe(logical, host, random_seed=seed % (2**31), tries=10)
    except Exception:
        return {}
    return _finish(emb, logical, host)


def clique_embedding(logical, host, seed):
    """The clique embedder of dwave.embedding for the host family (Chimera / Pegasus / Zephyr):
    every variable gets a chain of the K_n clique embedding, whatever the instance's density."""
    try:
        import dwave_networkx as dnx
        fam = host.graph.get("family")
        n = logical.number_of_nodes()
        if fam == "chimera":
            from dwave.embedding.chimera import find_clique_embedding as fce
            emb = fce(n, target_graph=host)
        elif fam == "pegasus":
            from dwave.embedding.pegasus import find_clique_embedding as fce
            emb = fce(n, target_graph=host)
        elif fam == "zephyr":
            from dwave.embedding.zephyr import find_clique_embedding as fce
            emb = fce(n, target_graph=host)
        else:
            return {}
        nodes = sorted(logical.nodes()); keys = sorted(emb)
        return _finish({v: list(emb[k]) for v, k in zip(nodes, keys)}, logical, host)
    except Exception:
        return {}
