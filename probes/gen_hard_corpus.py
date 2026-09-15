"""Build a corpus where the embedding problem is actually hard.

The corpus everything so far was measured on is nominally sixteen variables and is, on average,
4.48 independent problems whose largest piece holds eleven of them, with three spins coupled to
nothing. minorminer embeds it in two milliseconds and never fails. On instances like that the
chains are one or two qubits long, chain breaks barely happen, and the choice of embedding cannot
matter much: measured over its candidates, spending more qubits correlates with *worse* quality
at -0.203, because more qubits there just means a longer chain on a problem that did not need
one.

Nothing about optimising the right objective can show itself on data like that.

This generator inverts the construction. Instead of dropping chains onto the hardware and taking
whatever contact graph falls out, it starts from a dense logical graph, plants a frustrated-loop
Ising on it for a certified ground energy, and then asks minorminer whether the thing can be
embedded at all. Instances it cannot embed within the cap are dropped and counted; the ones that
survive need real chains, so chain integrity is on the critical path and the embedding actually
has to be chosen well.

The witness is minorminer's own embedding. It proves feasibility and nothing else: it is not a
quality label, and no learned method is scored against it.
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import networkx as nx
import numpy as np
from isingfold.rl.contracts import Context
from isingfold.rl.data.generate import (GeneratedInstance, host_graph, write_corpus)
from isingfold.rl.data.lineage import Lineage
from isingfold.rl.data.planting import PlantingError, frustrated_loops
from isingfold.rl.env import EmbeddingTask

from _initializers import minorminer_initializer


FAMILIES = ("clique", "dense", "sparse", "bipartite", "lattice", "modular", "scalefree")
"""Structural axes, not 600 draws of one shape.

A corpus of one family tests transfer between coefficient draws and calls it generalisation. It
also cannot show a resource-quality relation that only appears between shapes: a clique forces
long chains everywhere, so within it every embedding costs about the same and the question never
arises. These seven differ in the thing that decides embedding difficulty, which is how demand
for connectivity is distributed: uniformly and high, uniformly and low, concentrated in hubs,
split between blocks, or laid out in a plane.
"""


def _connect(g, rng):
    """Join components rather than returning several problems in a trenchcoat, which is how the
    corpus this replaces ended up with 4.48 independent pieces per instance."""
    comps = [sorted(c) for c in nx.connected_components(g)]
    for a, b in zip(comps, comps[1:]):
        g.add_edge(a[0], b[0])
    return g


def logical_graph(n, family, rng):
    seed = int(rng.integers(0, 2 ** 31))
    if family == "clique":
        return nx.complete_graph(n)
    if family == "dense":
        return _connect(nx.gnm_random_graph(n, int(round(3.5 * n)), seed=seed), rng)
    if family == "sparse":
        return _connect(nx.gnm_random_graph(n, int(round(1.6 * n)), seed=seed), rng)
    if family == "bipartite":
        a = n // 2
        return nx.complete_bipartite_graph(a, n - a)
    if family == "lattice":
        rows = max(2, int(round(n ** 0.5)))
        cols = max(2, (n + rows - 1) // rows)
        g = nx.convert_node_labels_to_integers(nx.grid_2d_graph(rows, cols))
        return nx.convert_node_labels_to_integers(g.subgraph(list(g.nodes())[:n]).copy())
    if family == "modular":
        blocks = 3 if n >= 12 else 2
        sizes = [n // blocks] * blocks
        for i in range(n - sum(sizes)):
            sizes[i] += 1
        g = nx.random_partition_graph(sizes, 0.85, 0.06, seed=seed)
        return _connect(nx.Graph(g), rng)
    if family == "scalefree":
        return _connect(nx.barabasi_albert_graph(n, 3, seed=seed), rng)
    raise ValueError("unregistered family %r" % family)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="chimera")
    ap.add_argument("--host-size", type=int, default=4)
    ap.add_argument("--variables", default="14,16,18",
                    help="comma list of logical sizes; every family is generated at every size")
    ap.add_argument("--families", default=",".join(FAMILIES),
                    help="comma list from %s" % ", ".join(FAMILIES))
    ap.add_argument("--instances", type=int, default=600)
    ap.add_argument("--alpha", type=float, default=0.9)
    ap.add_argument("--clause-length", type=int, default=5)
    ap.add_argument("--weights", default="0.4,1.0,2.5")
    ap.add_argument("--qubit-cap", type=int, default=120)
    ap.add_argument("--mm-tries", type=int, default=20)
    ap.add_argument("--strength-reads", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260915)
    a = ap.parse_args()

    host = host_graph(a.host, a.host_size)
    mm = minorminer_initializer(a.mm_tries)
    rng = np.random.default_rng(a.seed)
    weights = tuple(float(x) for x in a.weights.split(","))
    ctx = Context(qubit_cap=a.qubit_cap)

    families = [f.strip() for f in a.families.split(",") if f.strip()]
    sizes = [int(v) for v in a.variables.split(",") if v]
    cells = [(f, v) for f in families for v in sizes]
    per_cell = max(1, a.instances // len(cells))
    print(json.dumps({"families": families, "sizes": sizes, "cells": len(cells),
                      "target_per_cell": per_cell}), flush=True)

    kept, tried, no_embedding, over_cap, no_planting = [], 0, 0, 0, 0
    chain_stats, mm_secs = [], []
    by_cell: dict[tuple, int] = {c: 0 for c in cells}
    started = time.time()
    while len(kept) < a.instances and tried < a.instances * 20:
        tried += 1
        local = a.seed * 1000 + tried
        # Round-robin over the cells that still want instances, so a family that is easy to
        # generate cannot crowd out one that is not.
        hungry = [c for c in cells if by_cell[c] < per_cell] or cells
        family, n_vars = hungry[tried % len(hungry)]
        logical = logical_graph(n_vars, family, rng)
        try:
            planted = frustrated_loops(logical, alpha=a.alpha, seed=local,
                                       max_length=a.clause_length, weight_choices=weights)
        except PlantingError:
            no_planting += 1
            continue
        graph = planted.problem.graph
        t0 = time.time()
        chains = mm(graph, host, local)
        mm_secs.append(time.time() - t0)
        if chains is None:
            no_embedding += 1
            continue
        used = sum(len(c) for c in chains.values())
        if used > a.qubit_cap:
            over_cap += 1
            continue
        chain_stats.append((used, max(len(c) for c in chains.values()),
                            used / max(1, graph.number_of_nodes())))
        name = "%s%d-%s%d-%d" % (a.host, a.host_size, family, n_vars,
                                 by_cell[(family, n_vars)])
        lineage = Lineage(lineage_id="%s-l%d" % (name, local),
                          family="%s%d" % (family, n_vars),
                          host="%s%d" % (a.host, a.host_size), size=n_vars,
                          generator_version="hard-diverse-frustrated-1")
        task = EmbeddingTask(name=name, logical=graph, host=host, problem=planted.problem,
                             ground_energy=planted.ground_energy, lineage=lineage.lineage_id,
                             witness={v: frozenset(c) for v, c in chains.items()})
        kept.append(GeneratedInstance(task=task, lineage=lineage,
                                      witness_receipt={"source": "minorminer",
                                                       "qubits": used,
                                                       "max_chain": chain_stats[-1][1]},
                                      clause_report=planted.verify()))
        by_cell[(family, n_vars)] += 1

    stats = np.array(chain_stats) if chain_stats else np.zeros((1, 3))
    print(json.dumps({
        "kept": len(kept), "attempted": tried, "no_planting": no_planting,
        "minorminer_found_nothing": no_embedding, "over_qubit_cap": over_cap,
        "mean_qubits": float(stats[:, 0].mean()), "mean_max_chain": float(stats[:, 1].mean()),
        "mean_qubits_per_variable": float(stats[:, 2].mean()),
        "mean_minorminer_seconds": float(np.mean(mm_secs)) if mm_secs else None,
        "max_minorminer_seconds": float(np.max(mm_secs)) if mm_secs else None,
        "seconds": round(time.time() - started),
        "per_cell": {"%s/%d" % (f, v): c for (f, v), c in sorted(by_cell.items())}}),
        flush=True)
    if not kept:
        print("nothing survived; loosen the settings"); return 1

    manifest = write_corpus(kept, a.out, ctx=ctx, strength_reads=a.strength_reads, seed=a.seed)
    print(json.dumps(manifest, indent=1), flush=True)
    print("\nHARD CORPUS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
