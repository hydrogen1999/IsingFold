"""Training-data generation: certified decision samples from planted embeddings.

Pipeline (sections 5, 6 and 10 of the 2026-09-08 meeting note):

1. Grow a planted witness Pi* on a real host with the ink-drop generator, in one of its
   diversity modes.
2. Choose a decision point: k variables in play, the rest of Pi* frozen as blockers. The
   variable in focus keeps a prefix of its witness chain as its core (possibly nothing); the
   other in-play variables keep their full witness chains or a prefix, at random.
3. Cut a window: the witness chains of the in-play variables plus every free qubit within
   `radius` hops. Everything outside the window is unavailable. Because the witness chains lie
   inside the window, at least one completion exists, so the sample always has a feasible
   action and a certified V* for every candidate.
4. Candidate actions A_t: extend the focus variable's core by an adjacent free qubit in the
   window, or, if it has no core, place it on any free window qubit within `radius` of a
   logical neighbour's chain (or anywhere in the window when it has no placed neighbour).
5. Label every action by exact completion inside the window (`exact.best_completion`), with a
   node budget; a sample whose enumeration exceeds the budget is dropped and counted.
6. Keep the sample only if the best and second-best actions differ (margin > 0), as the note
   requires for a usable hand-test. Also record what a local greedy rule would have chosen,
   which is the hard-negative statistic of section 7.

Certification is relative to the window: V* is exact for the problem "complete the in-play
variables using window qubits". A window action that is infeasible could be feasible with
qubits outside the window; the record keeps `radius` and the window so that claim is scoped.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import networkx as nx

from embedbench.exact import SearchAborted, best_completion
from embedbench.inkdrop import MODES, PlantedEmbedding, PlantedGenerationError, ink_drop
from embedbench.objective import Outcome, compare
from embedbench.embedding import Node, Qubit


@dataclass(frozen=True)
class GenConfig:
    topology: str = "chimera"
    size: int = 4
    n_vars: int = 12
    chain_size: int = 3
    modes: tuple[str, ...] = tuple(MODES)
    k_in_play: int = 3
    radius: int = 1
    l_cap: int = 4
    max_window_free: int = 30
    max_nodes: int = 200_000
    samples_per_instance: int = 6
    prefix_keep_prob: float = 0.5
    """Probability that a non-focus in-play variable keeps its full witness chain."""
    max_actions: int = 12
    q_cap_slack: int | None = None
    """When set, completions inside the window may use at most (witness qubits of the
    in-play variables + slack). This is the capacity-pressure lever: with slack 0 any action
    that overspends the witness's budget has no feasible completion."""


@dataclass
class GenStats:
    instances: int = 0
    generation_failures: int = 0
    samples_attempted: int = 0
    samples_kept: int = 0
    dropped_no_margin: int = 0
    dropped_window_too_large: int = 0
    dropped_aborted: int = 0
    dropped_no_actions: int = 0
    greedy_disagreements: int = 0
    margin_kinds: dict[str, int] = field(default_factory=dict)
    exact_nodes_total: int = 0


def connected_random_graph(n: int, m: int, seed: int, draws: int = 50) -> nx.Graph:
    """A connected random graph with n nodes and m edges: G(n, m) if a draw is connected
    within `draws` attempts (this keeps every corpus generated before v1.1 reproducible), else
    a random tree plus random extra edges up to m."""
    for k in range(draws):
        g = nx.gnm_random_graph(n, m, seed=(seed + k) % (2**31))
        if nx.is_connected(g):
            return g
    rng = random.Random(seed); order = list(range(n)); rng.shuffle(order)
    g = nx.Graph(); g.add_nodes_from(order)
    for k in range(1, n):
        g.add_edge(order[k], order[rng.randrange(k)])
    while g.number_of_edges() < m:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            g.add_edge(u, v)
    return g


def host_graph(topology: str, size: int) -> nx.Graph:
    import dwave_networkx as dnx
    if topology == "chimera":
        return dnx.chimera_graph(size)
    if topology == "pegasus":
        return dnx.pegasus_graph(size)
    if topology == "zephyr":
        return dnx.zephyr_graph(size)
    raise ValueError(topology)


def _chain_prefix(host: nx.Graph, chain: frozenset[Qubit], keep: int, rng: random.Random) -> frozenset[Qubit]:
    """A connected prefix of `keep` qubits of a witness chain, grown by BFS from a random
    member so it stays connected."""
    if keep <= 0:
        return frozenset()
    if keep >= len(chain):
        return chain
    start = rng.choice(sorted(chain))
    out, frontier = {start}, [start]
    while frontier and len(out) < keep:
        x = frontier.pop(0)
        nbs = [n for n in host.neighbors(x) if n in chain and n not in out]
        rng.shuffle(nbs)
        for n in nbs:
            if len(out) < keep:
                out.add(n)
                frontier.append(n)
    return frozenset(out)


def frozen_adjacency(host, frozen: dict, window: set) -> dict:
    """window qubit -> sorted frozen qubits adjacent to it in the full host (the encoder
    cannot see outside the window otherwise)."""
    fq = {q for c in frozen.values() for q in c}
    out = {}
    for q in window:
        t = sorted(int(n) for n in host.neighbors(q) if n in fq)
        if t:
            out[int(q)] = t
    return out


def frozen_requirements(host, logical, in_play, frozen: dict, window: set):
    """For each in-play variable, the window qubits adjacent to the chain of each frozen
    logical neighbour (one set per neighbour), plus the list of those neighbours."""
    must_hit: dict = {}
    frozen_edges: list = []
    for v in in_play:
        reqs = []
        for u in logical.neighbors(v):
            if u in frozen:
                hit = frozenset(q for q in window if any(host.has_edge(q, t) for t in frozen[u]))
                reqs.append(hit)
                frozen_edges.append([int(v), int(u)])
        if reqs:
            must_hit[v] = reqs
    return must_hit, frozen_edges


def greedy_local_choice(
    host: nx.Graph, logical: nx.Graph, cores: dict[Node, frozenset[Qubit]], focus: Node,
    actions: Sequence[tuple[Node, Qubit]],
) -> tuple[Node, Qubit]:
    """The local surrogate the meeting warns about: prefer the qubit that completes the most
    logical contacts now, then the one closest (BFS through free qubits) to the nearest
    unreached logical neighbour, then the lowest id. No lookahead."""
    occupied = {q for c in cores.values() for q in c}
    owner = {q: v for v, c in cores.items() for q in c}
    need = [u for u in logical.neighbors(focus) if u in cores and cores[u]]
    core = cores.get(focus, frozenset())
    touched = {owner[n] for q in core for n in host.neighbors(q) if n in owner and owner[n] != focus}

    def dist_to(q: Qubit, u: Node) -> int:
        target = cores[u]
        seen, frontier, d = {q}, [q], 0
        while frontier and d < 20:
            d += 1
            nxt = []
            for x in frontier:
                for n in host.neighbors(x):
                    if n in target:
                        return d
                    if n not in occupied and n not in seen:
                        seen.add(n)
                        nxt.append(n)
            frontier = nxt
        return 99

    def key(a):
        _, q = a
        completes = sum(1 for n in host.neighbors(q) if n in owner and owner[n] in need and owner[n] not in touched)
        unreached = [u for u in need if u not in touched]
        d = min((dist_to(q, u) for u in unreached), default=0)
        return (-completes, d, q)

    return min(actions, key=key)


@dataclass(frozen=True)
class DecisionRecord:
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
    values: list[list[int]]
    best_action: list[int]
    margin_kind: str
    margin: int
    greedy_action: list[int]
    greedy_agrees: bool
    radius: int
    l_cap: int
    exact_nodes: int
    witness_outcome: list[int]
    frozen_edges: list[list[int]] = field(default_factory=list)
    """Logical edges from in-play variables to frozen ones; every completion realises them."""
    frozen_adjacency: dict[int, list[int]] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))


def decision_samples_from_witness(
    pe: PlantedEmbedding, cfg: GenConfig, rng: random.Random, stats: GenStats, instance_id: str,
) -> Iterable[DecisionRecord]:
    host, logical, chains = pe.host, pe.logical, dict(pe.chains)
    variables = sorted(chains)
    for _ in range(cfg.samples_per_instance):
        stats.samples_attempted += 1
        focus = rng.choice(variables)
        # in-play set: focus plus logical neighbours first, then random fill
        nbrs = [u for u in logical.neighbors(focus)]
        rng.shuffle(nbrs)
        in_play = [focus] + nbrs[: cfg.k_in_play - 1]
        others = [v for v in variables if v not in in_play]
        rng.shuffle(others)
        while len(in_play) < cfg.k_in_play and others:
            in_play.append(others.pop())
        frozen = {v: chains[v] for v in variables if v not in in_play}
        blocked = {q for c in frozen.values() for q in c}

        # cores: focus keeps a prefix of length 0..|chain|-1; others keep full or a prefix
        cores: dict[Node, frozenset[Qubit]] = {}
        fk = rng.randrange(0, len(chains[focus]))
        cores[focus] = _chain_prefix(host, chains[focus], fk, rng)
        for v in in_play[1:]:
            if rng.random() < cfg.prefix_keep_prob:
                cores[v] = chains[v]
            else:
                cores[v] = _chain_prefix(host, chains[v], rng.randrange(0, len(chains[v]) + 1), rng)

        # window: witness chains of in-play variables plus free qubits within radius, the
        # free qubits ranked by hop distance from the witness (ties broken by the rng) and
        # truncated to max_window_free. Truncation keeps the witness, so a completion still
        # exists; on Pegasus and Zephyr a one-hop neighbourhood alone exceeds the cap.
        seed_set = set().union(*(chains[v] for v in in_play))
        window = set(seed_set)
        frontier = set(seed_set)
        ranked: list[tuple[int, float, Qubit]] = []
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
        budget = cfg.max_window_free - len(witness_free)
        extra = [n for _, _, n in sorted(ranked)[:budget]]
        window = seed_set | set(extra)
        free_in_window = window - core_qubits
        wh = host.subgraph(window).copy()
        wl = logical.subgraph(in_play).copy()
        must_hit, frozen_edges = frozen_requirements(host, logical, in_play, frozen, window)

        # actions for the focus variable
        core = cores[focus]
        if core:
            cand = sorted({n for q in core for n in wh.neighbors(q) if n in free_in_window})
        else:
            placed_nb = [u for u in wl.neighbors(focus) if cores.get(u)]
            if placed_nb:
                near = set()
                for u in placed_nb:
                    for q in cores[u]:
                        for n in wh.neighbors(q):
                            if n in free_in_window:
                                near.add(n)
                cand = sorted(near)
            else:
                cand = sorted(free_in_window)
        if len(cand) < 2:
            stats.dropped_no_actions += 1
            continue
        if len(cand) > cfg.max_actions:
            cand = rng.sample(cand, cfg.max_actions)
        actions = [(focus, q) for q in cand]

        values: list[Outcome] = []
        st: dict = {}
        q_cap = None
        if cfg.q_cap_slack is not None:
            q_cap = sum(len(chains[v]) for v in in_play) + cfg.q_cap_slack
        try:
            for (v, q) in actions:
                c2 = dict(cores)
                c2[v] = cores[v] | {q}
                out, _ = best_completion(wh, wl, c2, l_cap=cfg.l_cap, q_cap=q_cap,
                                         max_nodes=cfg.max_nodes, stats=st, must_hit=must_hit)
                values.append(out)
            witness_out, _ = best_completion(
                wh, wl, {v: chains[v] for v in in_play}, l_cap=cfg.l_cap, q_cap=q_cap,
                max_nodes=cfg.max_nodes, stats=st, must_hit=must_hit,
            )
        except SearchAborted:
            stats.dropped_aborted += 1
            continue
        if witness_out[0] != 1:
            raise RuntimeError("witness must complete inside its own window")
        nodes = st.get("nodes", 0)
        stats.exact_nodes_total += nodes
        order = sorted(range(len(actions)), key=lambda k: values[k], reverse=True)
        kind, margin = compare(values[order[0]], values[order[1]])
        if margin <= 0:
            stats.dropped_no_margin += 1
            continue
        greedy = greedy_local_choice(wh, wl, cores, focus, actions)
        agrees = values[actions.index(greedy)] == values[order[0]]
        if not agrees:
            stats.greedy_disagreements += 1
        stats.samples_kept += 1
        stats.margin_kinds[kind] = stats.margin_kinds.get(kind, 0) + 1
        yield DecisionRecord(
            instance_id=instance_id, mode=pe.mode, topology=cfg.topology, size=cfg.size,
            focus=int(focus), in_play=[int(v) for v in in_play],
            window_nodes=sorted(int(q) for q in window),
            window_edges=sorted([int(a), int(b)] if a < b else [int(b), int(a)] for a, b in wh.edges()),
            logical_edges=sorted([int(a), int(b)] if a < b else [int(b), int(a)] for a, b in wl.edges()),
            cores={int(v): sorted(int(q) for q in c) for v, c in cores.items()},
            frozen={int(v): sorted(int(q) for q in c) for v, c in frozen.items() if c & set(
                n for q in window for n in host.neighbors(q))},
            actions=[[int(v), int(q)] for v, q in actions],
            values=[list(o) for o in values],
            best_action=[int(actions[order[0]][0]), int(actions[order[0]][1])],
            margin_kind=kind, margin=int(margin),
            greedy_action=[int(greedy[0]), int(greedy[1])], greedy_agrees=agrees,
            radius=cfg.radius, l_cap=cfg.l_cap, exact_nodes=int(nodes),
            witness_outcome=list(witness_out), frozen_edges=frozen_edges,
            frozen_adjacency=frozen_adjacency(host, frozen, window),
        )


def _instance_seed(seed: int, mode: str, i: int) -> int:
    """Deterministic per-instance seed (Python's hash() of a str is salted per process, so it
    must not be used here)."""
    h = hashlib.sha256(f"{seed}:{mode}:{i}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def _merge_stats(into: GenStats, part: GenStats) -> None:
    for k, v in asdict(part).items():
        if k == "margin_kinds":
            for kk, vv in v.items():
                into.margin_kinds[kk] = into.margin_kinds.get(kk, 0) + vv
        else:
            setattr(into, k, getattr(into, k) + v)


def _work(args) -> tuple[GenStats, list[str]]:
    """One (mode, instance) unit: grow the witness and emit its samples. Self-contained so it
    can run in a worker process; the host is rebuilt per call (cheap next to enumeration)."""
    cfg, mode, i, seed = args
    host = host_graph(cfg.topology, cfg.size)
    stats = GenStats()
    gseed = _instance_seed(seed, mode, i)
    try:
        pe = ink_drop(host, cfg.n_vars, cfg.chain_size, mode=mode, seed=gseed)
    except PlantedGenerationError:
        stats.generation_failures += 1
        return stats, []
    stats.instances += 1
    iid = f"{cfg.topology}{cfg.size}-{mode}-{i}-s{gseed}"
    rng = random.Random(gseed ^ 0x5DEECE66D)
    lines = [rec.to_json() for rec in decision_samples_from_witness(pe, cfg, rng, stats, iid)]
    return stats, lines


def generate(cfg: GenConfig, n_instances: int, seed: int, out: Path | None = None, jobs: int = 1) -> GenStats:
    """Generate `n_instances` planted witnesses per mode and their certified decision
    samples; write JSONL to `out` and a manifest next to it. `jobs` > 1 runs instances in
    worker processes; output order and content are identical for any `jobs`."""
    units = [(cfg, mode, i, seed) for mode in cfg.modes for i in range(n_instances)]
    stats = GenStats()
    fh = open(out, "w") if out else None
    try:
        if jobs > 1:
            from multiprocessing import Pool
            with Pool(jobs) as pool:
                results = pool.imap(_work, units, chunksize=4)
                for part, lines in results:
                    _merge_stats(stats, part)
                    if fh:
                        fh.write("".join(l + "\n" for l in lines))
        else:
            for u in units:
                part, lines = _work(u)
                _merge_stats(stats, part)
                if fh:
                    fh.write("".join(l + "\n" for l in lines))
    finally:
        if fh:
            fh.close()
    if out:
        digest = hashlib.sha256(Path(out).read_bytes()).hexdigest()
        manifest = {"config": asdict(cfg), "n_instances_per_mode": n_instances, "seed": seed, "jobs": jobs,
                    "stats": asdict(stats), "sha256": digest, "file": str(out)}
        Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=1))
    return stats
