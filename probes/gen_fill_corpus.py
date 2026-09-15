"""Instances planted at a chosen fill of the host, so feasibility is known in advance.

The advisor's scale for difficulty is the fraction of the host an embedding needs: ninety
percent is slightly hard, ninety-five is hard, a hundred is infeasible. minorminer breaks at
seventy percent on cliques and eighty-eight on sparse graphs, so the whole scale above ninety
is empty for the standard tool, and no instance found by asking minorminer can live there.

This plants the embedding first. The host is partitioned into connected chains whose lengths
follow a truncated power law, until the chosen fraction of qubits is used; the logical graph is
the quotient of that partition, one node per chain and an edge wherever two chains touch. The
partition is then a valid embedding of the logical graph at exactly the chosen fill, kept as the
witness, and a frustrated-loop problem is planted on the logical graph as in the other corpora.

For every instance minorminer is also asked, at its deployed budget, so the same run reports
its validity rate at each fill.
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
import networkx as nx
import numpy as np
from isingfold.rl.data.generate import GeneratedInstance, host_graph, write_corpus
from isingfold.rl.data.lineage import Lineage
from isingfold.rl.data.planting import PlantingError, frustrated_loops
from isingfold.rl.env import EmbeddingTask

from _context import host_context
from _initializers import minorminer_initializer


def draw_length(rng, alpha, lmin, lmax):
    ls = np.arange(lmin, lmax + 1)
    p = ls.astype(float) ** (-alpha)
    return int(rng.choice(ls, p=p / p.sum()))


def plant_partition(host, fill, alpha, lmin, lmax, rng):
    """Connected chains covering `fill` of the host; returns {chain_id: [qubits]}."""
    nodes = list(host.nodes())
    order = rng.permutation(len(nodes))
    used = set()
    chains = []
    target = int(round(fill * len(nodes)))
    for idx in order:
        if len(used) >= target:
            break
        start = nodes[idx]
        if start in used:
            continue
        want = draw_length(rng, alpha, lmin, lmax)
        chain = [start]
        used.add(start)
        while len(chain) < want and len(used) < target:
            frontier = [t for q in chain for t in host.neighbors(q) if t not in used]
            if not frontier:
                break
            nxt = frontier[rng.integers(0, len(frontier))]
            chain.append(nxt)
            used.add(nxt)
        chains.append(chain)
    return chains


def quotient(host, chains):
    owner = {q: i for i, c in enumerate(chains) for q in c}
    g = nx.Graph()
    g.add_nodes_from(range(len(chains)))
    for u, v in host.edges():
        a, b = owner.get(u), owner.get(v)
        if a is not None and b is not None and a != b:
            g.add_edge(a, b)
    return g


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="pegasus")
    ap.add_argument("--host-size", type=int, default=6)
    ap.add_argument("--fills", default="0.80,0.85,0.90,0.95")
    ap.add_argument("--alphas", default="3.0,2.0")
    ap.add_argument("--lmin", type=int, default=1)
    ap.add_argument("--lmax", type=int, default=6)
    ap.add_argument("--per-cell", type=int, default=6)
    ap.add_argument("--mm-tries", type=int, default=10)
    ap.add_argument("--alpha-clause", type=float, default=0.9)
    ap.add_argument("--clause-length", type=int, default=5)
    ap.add_argument("--weights", default="0.4,1.0,2.5")
    ap.add_argument("--strength-reads", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260915)
    a = ap.parse_args()

    host = host_graph(a.host, a.host_size)
    cap = host.number_of_nodes()
    ctx = host_context(cap)
    mm = minorminer_initializer(a.mm_tries)
    rng = np.random.default_rng(a.seed)
    weights = tuple(float(x) for x in a.weights.split(","))
    fills = [float(x) for x in a.fills.split(",")]
    alphas = [float(x) for x in a.alphas.split(",")]
    print(json.dumps({"host": "%s%d" % (a.host, a.host_size), "qubits": cap, "fills": fills,
                      "alphas": alphas, "per_cell": a.per_cell}), flush=True)
    print("  %-5s %-5s %6s %8s %8s %9s %8s %9s" % ("fill", "alpha", "vars", "witness",
                                                     "mmvalid", "mmqubits", "mmsecs", "mmfill"))
    kept = []
    local = 0
    for fill in fills:
        for alpha in alphas:
            n_ok, n_vars, mm_ok, mm_q, mm_s = 0, [], 0, [], []
            for i in range(a.per_cell):
                local += 1
                chains = plant_partition(host, fill, alpha, a.lmin, a.lmax, rng)
                logical = quotient(host, chains)
                if not nx.is_connected(logical):
                    comp = max(nx.connected_components(logical), key=len)
                    chains = [chains[i] for i in sorted(comp)]
                    logical = quotient(host, chains)
                try:
                    planted = frustrated_loops(logical, alpha=a.alpha_clause, seed=a.seed + local,
                                               max_length=a.clause_length, weight_choices=weights)
                except PlantingError:
                    continue
                graph = planted.problem.graph
                witness = {v: frozenset(chains[v]) for v in graph.nodes()}
                used = sum(len(c) for c in witness.values())
                t0 = time.time()
                found = mm(graph, host, a.seed + local)
                mm_s.append(time.time() - t0)
                if found is not None:
                    mm_ok += 1
                    mm_q.append(sum(len(c) for c in found.values()))
                n_ok += 1
                n_vars.append(graph.number_of_nodes())
                name = "%s%d-fill%02d-a%.1f-%d" % (a.host, a.host_size, round(fill * 100), alpha, i)
                lineage = Lineage(lineage_id="%s-l%d" % (name, local),
                                  family="fill%02d-a%.1f" % (round(fill * 100), alpha),
                                  host="%s%d" % (a.host, a.host_size),
                                  size=graph.number_of_nodes(),
                                  generator_version="planted-fill-1")
                task = EmbeddingTask(name=name, logical=graph, host=host, problem=planted.problem,
                                     ground_energy=planted.ground_energy,
                                     lineage=lineage.lineage_id, witness=witness)
                kept.append(GeneratedInstance(task=task, lineage=lineage,
                                              witness_receipt={"source": "planted-partition",
                                                               "qubits": used, "fill": fill,
                                                               "max_chain": max(len(c) for c in witness.values())},
                                              clause_report=planted.verify()))
            print("  %-5.2f %-5.1f %6.0f %8.0f %8.2f %9s %8.1f %9s"
                  % (fill, alpha, np.mean(n_vars) if n_vars else 0,
                     fill * cap, mm_ok / max(1, n_ok),
                     ("%.0f" % np.mean(mm_q)) if mm_q else "-",
                     np.mean(mm_s) if mm_s else 0,
                     ("%.2f" % (np.mean(mm_q) / cap)) if mm_q else "-"), flush=True)
    if not kept:
        print("nothing planted"); return 1
    manifest = write_corpus(kept, a.out, ctx=ctx, strength_reads=a.strength_reads, seed=a.seed)
    print(json.dumps({k: manifest[k] for k in list(manifest)[:6]}), flush=True)
    print("FILL CORPUS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
