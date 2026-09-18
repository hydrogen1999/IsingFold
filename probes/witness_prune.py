"""How much of the planted witness is actually required, and therefore what fill is certified.

The fill corpora are named by the fraction of the host the planted partition occupies, and the
paper has been reading that number as the instance's congestion. It is only an upper bound. The
witness is one embedding, not a minimum one, and the generator's own lower bound is variables
divided by host qubits (`gen_fill_corpus.lower_bound_fill`). For a 28-variable instance on a
40-qubit host the two bounds are 0.70 and 0.90, so "ninety percent congestion" is unearned until
the gap is measured.

This prunes the witness greedily: a qubit may leave a chain when the chain stays non-empty and
connected and every logical edge still has a realised host contact. What survives is a certified
sufficient occupancy. If a fill-0.90 witness prunes to 0.72, the instance was never a 0.90
instance and minorminer's failure on it needs a different explanation.
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
from isingfold.rl.data.generate import host_graph
from isingfold.rl.data.planting import PlantingError, frustrated_loops

from _initializers import minorminer_initializer
from gen_fill_corpus import plant_partition, quotient, witness_occupancy


def contacts_ok(host, chains, u, v):
    return any(host.has_edge(a, b) for a in chains[u] for b in chains[v])


def prune(host, chains, graph):
    """Drop every qubit whose absence keeps the embedding valid, largest chains first."""
    work = {v: set(c) for v, c in chains.items()}
    changed = True
    while changed:
        changed = False
        for v in sorted(work, key=lambda x: -len(work[x])):
            if len(work[v]) <= 1:
                continue
            for q in sorted(work[v], key=str):
                if len(work[v]) <= 1:
                    break
                trial = set(work[v]) - {q}
                if not nx.is_connected(host.subgraph(trial)):
                    continue
                keep = dict(work)
                keep[v] = trial
                if all(contacts_ok(host, keep, a, b) for a, b in graph.edges(v)):
                    work[v] = trial
                    changed = True
    return {v: frozenset(c) for v, c in work.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="pegasus")
    ap.add_argument("--sizes", default="2,3")
    ap.add_argument("--fills", default="0.80,0.90,0.95")
    ap.add_argument("--alpha", type=float, default=3.0)
    ap.add_argument("--alpha-clause", type=float, default=0.9)
    ap.add_argument("--clause-length", type=int, default=5)
    ap.add_argument("--weights", default="0.4,1.0,2.5")
    ap.add_argument("--instances", type=int, default=12)
    ap.add_argument("--lmin", type=int, default=1)
    ap.add_argument("--lmax", type=int, default=6)
    ap.add_argument("--mm-tries", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260918)
    a = ap.parse_args()

    weights = tuple(float(x) for x in a.weights.split(","))
    mm = minorminer_initializer(a.mm_tries)
    print(json.dumps({"probe": "witness_prune", "host": a.host,
                      "sizes": [int(x) for x in a.sizes.split(",")],
                      "fills": [float(x) for x in a.fills.split(",")],
                      "instances": a.instances, "mm_tries": a.mm_tries}), flush=True)
    print("  %-9s %5s %6s %8s %8s %8s %8s %8s %7s %8s"
          % ("host", "vars", "named", "witness", "pruned", "prunedfl", "lowbound", "mmqubits",
             "mm_ok", "mmfill"), flush=True)

    local = 0
    for size in [int(x) for x in a.sizes.split(",")]:
        host = host_graph(a.host, size)
        cap = host.number_of_nodes()
        for fill in [float(x) for x in a.fills.split(",")]:
            rng = np.random.default_rng(a.seed + size * 100 + int(fill * 100))
            rows = []
            for _ in range(a.instances):
                local += 1
                chains = plant_partition(host, fill, a.alpha, a.lmin, a.lmax, rng)
                logical = quotient(host, chains)
                if not nx.is_connected(logical):
                    comp = max(nx.connected_components(logical), key=len)
                    chains = [chains[k] for k in sorted(comp)]
                    logical = quotient(host, chains)
                try:
                    planted = frustrated_loops(logical, alpha=a.alpha_clause,
                                               seed=a.seed + local, max_length=a.clause_length,
                                               weight_choices=weights)
                except PlantingError:
                    continue
                graph = planted.problem.graph
                witness = {v: frozenset(chains[v]) for v in graph.nodes()}
                used = witness_occupancy(witness)
                tight = prune(host, witness, graph)
                kept = witness_occupancy(tight)
                found = mm(graph, host, a.seed + local)
                rows.append({"vars": graph.number_of_nodes(), "witness": used, "pruned": kept,
                             "mm": (sum(len(c) for c in found.values())
                                    if found is not None else None)})
                print(json.dumps({"host": "%s%d" % (a.host, size), "named_fill": fill,
                                  "qubits": cap, **rows[-1],
                                  "pruned_fill": kept / float(cap),
                                  "lower_bound_fill": rows[-1]["vars"] / float(cap),
                                  "instance": local}), flush=True)
            if not rows:
                continue
            m = lambda f: sum(f(r) for r in rows) / len(rows)
            mmq = [r["mm"] for r in rows if r["mm"] is not None]
            print("  %-9s %5.0f %6.2f %8.1f %8.1f %8.2f %8.2f %8s %7.2f %8s"
                  % ("%s%d" % (a.host, size), m(lambda r: r["vars"]), fill,
                     m(lambda r: r["witness"]), m(lambda r: r["pruned"]),
                     m(lambda r: r["pruned"]) / cap, m(lambda r: r["vars"]) / cap,
                     ("%.1f" % (sum(mmq) / len(mmq))) if mmq else "-",
                     len(mmq) / len(rows),
                     ("%.2f" % (sum(mmq) / len(mmq) / cap)) if mmq else "-"), flush=True)
    print("WITNESS PRUNE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
