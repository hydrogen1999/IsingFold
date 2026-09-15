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


def dense_logical(n, degree, rng, kind):
    """A logical graph that needs chains, rather than one that falls out of a placement."""
    if kind == "clique":
        return nx.complete_graph(n)
    target = int(round(degree * n / 2))
    for _ in range(200):
        g = nx.gnm_random_graph(n, target, seed=int(rng.integers(0, 2 ** 31)))
        if nx.is_connected(g):
            return g
    g = nx.gnm_random_graph(n, target, seed=int(rng.integers(0, 2 ** 31)))
    # Connect what the draw left apart rather than returning several problems in a trenchcoat,
    # which is the failure mode of the corpus this replaces.
    comps = list(nx.connected_components(g))
    for a, b in zip(comps, comps[1:]):
        g.add_edge(sorted(a)[0], sorted(b)[0])
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="chimera")
    ap.add_argument("--host-size", type=int, default=4)
    ap.add_argument("--variables", type=int, default=14)
    ap.add_argument("--degree", type=float, default=7.0)
    ap.add_argument("--kind", default="dense", choices=["dense", "clique"])
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

    kept, tried, no_embedding, over_cap, no_planting = [], 0, 0, 0, 0
    chain_stats, mm_secs = [], []
    started = time.time()
    while len(kept) < a.instances and tried < a.instances * 12:
        tried += 1
        local = a.seed * 1000 + tried
        logical = dense_logical(a.variables, a.degree, rng, a.kind)
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
        name = "%s%d-%s-%d" % (a.host, a.host_size, a.kind, len(kept))
        lineage = Lineage(lineage_id="%s-l%d" % (name, local), family=a.kind,
                          host="%s%d" % (a.host, a.host_size), size=a.variables,
                          generator_version="hard-dense-frustrated-1")
        task = EmbeddingTask(name=name, logical=graph, host=host, problem=planted.problem,
                             ground_energy=planted.ground_energy, lineage=lineage.lineage_id,
                             witness={v: frozenset(c) for v, c in chains.items()})
        kept.append(GeneratedInstance(task=task, lineage=lineage,
                                      witness_receipt={"source": "minorminer",
                                                       "qubits": used,
                                                       "max_chain": chain_stats[-1][1]},
                                      clause_report=planted.verify()))

    stats = np.array(chain_stats) if chain_stats else np.zeros((1, 3))
    print(json.dumps({
        "kept": len(kept), "attempted": tried, "no_planting": no_planting,
        "minorminer_found_nothing": no_embedding, "over_qubit_cap": over_cap,
        "mean_qubits": float(stats[:, 0].mean()), "mean_max_chain": float(stats[:, 1].mean()),
        "mean_qubits_per_variable": float(stats[:, 2].mean()),
        "mean_minorminer_seconds": float(np.mean(mm_secs)) if mm_secs else None,
        "max_minorminer_seconds": float(np.max(mm_secs)) if mm_secs else None,
        "seconds": round(time.time() - started)}), flush=True)
    if not kept:
        print("nothing survived; loosen the settings"); return 1

    manifest = write_corpus(kept, a.out, ctx=ctx, strength_reads=a.strength_reads, seed=a.seed)
    print(json.dumps(manifest, indent=1), flush=True)
    print("\nHARD CORPUS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
