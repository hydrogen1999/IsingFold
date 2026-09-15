"""What is actually in a corpus, beyond the number of variables its name claims.

Ink-drop planting guarantees a witness and frustrated loops guarantee the optimum, but neither
guarantees that the problem is hard, connected, or even one problem. After planting, couplings
that no clause covers are dropped and the reduced graph becomes the logical graph, so a corpus
advertised as sixteen variables can be a handful of small independent problems with several
spins that nothing couples to at all. An external audit measured this on thirty-two instances
and found most of them disconnected; this runs the same measurement on a whole corpus so the
number is a property of the thing being used rather than of a sample of it.

Reported per corpus:

    variables / edges        nominal size and how much of it survived
    isolated                 spins with no coupling, which are free to any solver
    components               how many independent problems one instance really is
    largest component        the effective size, which is what difficulty tracks
    degeneracy               ground states, where small enough to count exactly
    minorminer seconds       how hard the embedding problem is for the standard tool
    headroom                 spread of utility over independent embeddings of the same instance
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
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.evaluate import first_commit_controller, run_controller

from _initializers import minorminer_initializer


def degeneracy(problem, limit=18):
    """Exact count of ground states, or None when the instance is too big to enumerate."""
    nodes = sorted(problem.graph.nodes(), key=str)
    if len(nodes) > limit:
        return None, None
    index = {v: i for i, v in enumerate(nodes)}
    h = np.zeros(len(nodes))
    for v, val in problem.h.items():
        h[index[v]] = val
    js = [(index[u], index[v], w) for (u, v), w in problem.j.items()]
    best, count = None, 0
    for mask in range(1 << len(nodes)):
        spins = np.array([1 if (mask >> i) & 1 else -1 for i in range(len(nodes))])
        e = float(h @ spins) + sum(w * spins[i] * spins[k] for i, k, w in js)
        if best is None or e < best - 1e-12:
            best, count = e, 1
        elif abs(e - best) <= 1e-12:
            count += 1
    return best, count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--lineages", type=int, default=64)
    ap.add_argument("--headroom-draws", type=int, default=6)
    ap.add_argument("--reads", type=int, default=256)
    ap.add_argument("--qubit-cap", type=int, default=120)
    ap.add_argument("--degeneracy", action="store_true",
                    help="enumerate ground states exactly, which is 2^n work per instance")
    a = ap.parse_args()

    tasks = load_instances(a.corpus)[: a.lineages]
    ctx = Context(qubit_cap=a.qubit_cap)
    mm = minorminer_initializer(10)
    man = json.loads((Path(a.corpus) / "manifest.json").read_text())
    print(json.dumps({"corpus": a.corpus, "digest": man.get("digest"),
                      "generator": man.get("generator_version"), "examined": len(tasks)}),
          flush=True)

    rows = []
    for t in tasks:
        g = t.logical
        comps = sorted((len(c) for c in nx.connected_components(g)), reverse=True)
        iso = sum(1 for v in g.nodes() if g.degree(v) == 0)
        started = time.time()
        chains = mm(t.logical, t.host, 5000)
        mm_secs = time.time() - started
        utils = []
        if chains is not None:
            for j in range(a.headroom_draws):
                ch = mm(t.logical, t.host, 5000 + 97 * j)
                if ch is None:
                    continue

                def fixed(l, h, s, c=ch):
                    return c

                try:
                    out = run_controller([t], ctx, first_commit_controller, initializer=fixed,
                                         selector=fixed_strength_selector(),
                                         reward_reads=a.reads, repetitions=1, seed=4242 + j)
                except Exception:
                    continue
                o = out[0]
                if o.returned_valid and o.utility is not None:
                    utils.append(float(o.utility))
        e0, deg = (None, None)
        if a.degeneracy:
            e0, deg = degeneracy(t.problem)
        rows.append({"n": g.number_of_nodes(), "m": g.number_of_edges(), "iso": iso,
                     "comps": len(comps), "largest": comps[0] if comps else 0,
                     "mm": mm_secs, "utils": utils, "deg": deg})

    def col(key):
        return np.array([r[key] for r in rows], dtype=float)

    n = len(rows)
    print("\n  instances examined            %d" % n)
    print("  nominal variables             %.1f" % col("n").mean())
    print("  logical edges                 %.1f" % col("m").mean())
    print("  isolated spins                %.2f mean, present in %d of %d"
          % (col("iso").mean(), int((col("iso") > 0).sum()), n))
    print("  independent components        %.2f mean, more than one in %d of %d"
          % (col("comps").mean(), int((col("comps") > 1).sum()), n))
    print("  largest component             %.2f mean, smallest %d, at most 12 in %d of %d"
          % (col("largest").mean(), int(col("largest").min()),
             int((col("largest") <= 12).sum()), n))
    print("  minorminer seconds per draw   %.3f mean, %.3f max"
          % (col("mm").mean(), col("mm").max()))
    spreads = [max(r["utils"]) - min(r["utils"]) for r in rows if len(r["utils"]) >= 2]
    if spreads:
        print("  headroom over %d embeddings    %.4f mean spread, %.4f median"
              % (a.headroom_draws, float(np.mean(spreads)), float(np.median(spreads))))
        flat = sum(1 for s in spreads if s < 0.02)
        print("  instances with no headroom    %d of %d (spread below 0.02)" % (flat, len(spreads)))
    degs = [r["deg"] for r in rows if r["deg"]]
    if degs:
        print("  ground states                 %d min, %d max, %d instances counted"
              % (min(degs), max(degs), len(degs)))
    print("\nCORPUS REPORT DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
