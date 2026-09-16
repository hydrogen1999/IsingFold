"""If the learned part is the proposal distribution, is there a better pool to propose?

Every measurement here agrees: embedding quality is measurable and not predictable. Measured
selection over minorminer draws climbs from 0.68 to 0.86 as the pool grows; every model that
tries to predict quality transfers about +0.01; the one method that wins, Track A at +0.270,
measures four hundred times and predicts nothing. A policy that predicts and commits loses to
that structure and cannot be trained out of it, because there is nothing there to learn.

So the question for a learned embedder is not whether it can judge a candidate but whether it
can propose a better set of them, leaving the judging to the evaluator. This measures the
ceiling of three kinds of pool at the same size K:

    minorminer      K independent draws, the pool the deployed baseline already has
    grown           the same K draws, each grown toward its coupled chains, which is the one
                    way of spending qubits measured to beat the others at equal cost
    mixed           the first K/2 draws plus the grown version of each, which is what a policy
                    that can choose between the two spends would offer at the same budget

The number is the best measured candidate in the pool, re-measured on independent reads so a
maximum over noisy blocks does not carry its inflation into the comparison. Every pool holds
exactly K measured candidates, so pool size cannot explain a difference. The first version of
this probe built the mixed pool from the best half of each of the other two, which made it a
maximum over 2K measured candidates and read +0.011 and +0.021 for that reason alone.

    diverse         K of 3K draws, chosen for structural distance from each other before any
                    of them is measured, so the choice costs minorminer time and no reads

The corrected mixed pool reads -0.054 and -0.047: a grown variant of a draw is correlated with
its parent, so a pool of four draws and their four variants has four independent seeds where
the minorminer pool has eight. Pool diversity is worth more than any way of spending qubits.
The diverse pool asks whether that can be pushed the other way: if choosing draws for distance
raises the ceiling at equal K, proposing for diversity is a structural rule a policy can learn
without predicting quality.
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
from isingfold.rl.contracts import Context
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.evaluate import first_commit_controller, run_controller

from _context import host_context
from _initializers import minorminer_initializer
from powerlaw_embeddings import grown_embedding, realises

SELECT_BASE = 5_000
ASSESS_BASE = 90_000_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--lineages", type=int, default=60)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=3.0,
                    help="growth exponent; steep, so the grown pool spends few extra qubits")
    ap.add_argument("--reads", type=int, default=256)
    ap.add_argument("--assess-reads", type=int, default=512)
    ap.add_argument("--qubit-cap", type=int, default=248)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tasks = load_instances(a.corpus)[: a.lineages]
    ctx = host_context(a.qubit_cap)
    mm = minorminer_initializer(20)
    rng = np.random.default_rng(a.seed)
    print(json.dumps({"corpus": a.corpus, "lineages": len(tasks), "k": a.k, "alpha": a.alpha,
                      "reads": a.reads, "assess_reads": a.assess_reads}), flush=True)

    def measure(task, chains, seed, reads):
        def fixed(l, h, s):
            return chains
        try:
            out = run_controller([task], ctx, first_commit_controller, initializer=fixed,
                                 selector=fixed_strength_selector(), reward_reads=reads,
                                 repetitions=1, seed=seed)
        except Exception:
            return None
        o = out[0]
        return float(o.utility) if o.returned_valid and o.utility is not None else None

    pools = {"minorminer": {}, "grown": {}, "mixed": {}, "diverse": {}}
    cost = {"minorminer": [], "grown": [], "mixed": [], "diverse": []}

    def distance(c1, c2):
        # mean over variables of the Jaccard distance between the two chains' qubit sets
        ds = []
        for v in c1:
            a1, a2 = set(c1[v]), set(c2.get(v, ()))
            ds.append(1.0 - len(a1 & a2) / max(1, len(a1 | a2)))
        return float(np.mean(ds)) if ds else 0.0

    def farthest(cands, k):
        # greedy farthest-point selection on structure alone; nothing here has been measured
        chosen = [0]
        while len(chosen) < min(k, len(cands)):
            best, best_d = None, -1.0
            for i in range(len(cands)):
                if i in chosen:
                    continue
                d = min(distance(cands[i], cands[j]) for j in chosen)
                if d > best_d:
                    best, best_d = i, d
            chosen.append(best)
        return [cands[i] for i in chosen]
    started = time.time()
    for k, task in enumerate(tasks):
        draws, grown, pairs = [], [], []
        for j in range(a.k):
            ch = mm(task.logical, task.host, SELECT_BASE + 97 * j + 1000 * k)
            if ch is None:
                continue
            u = measure(task, ch, SELECT_BASE + 97 * j + 1000 * k, a.reads)
            if u is None:
                continue
            draws.append((u, ch, sum(len(c) for c in ch.values())))
            g, _ = grown_embedding(task.host, ch, a.alpha, 1, 8, rng, mode="contact",
                                   logical=task.logical)
            if realises(g, task.logical, task.host) and sum(len(c) for c in g.values()) <= a.qubit_cap:
                ug = measure(task, g, SELECT_BASE + 97 * j + 1000 * k + 50, a.reads)
                if ug is not None:
                    grown.append((ug, g, sum(len(c) for c in g.values())))
                    pairs.append((draws[-1], grown[-1]))
        if len(draws) < 3 or len(grown) < 3 or len(pairs) < 2:
            continue
        # Three K unmeasured draws, K kept for mutual distance, then measured like the others.
        unmeasured = []
        for j in range(3 * a.k):
            ch = mm(task.logical, task.host, SELECT_BASE + 7919 * j + 1000 * k + 500)
            if ch is not None:
                unmeasured.append(ch)
        diverse = []
        for j, ch in enumerate(farthest(unmeasured, a.k)):
            u = measure(task, ch, SELECT_BASE + 131 * j + 1000 * k + 900, a.reads)
            if u is not None:
                diverse.append((u, ch, sum(len(c) for c in ch.values())))
        if len(diverse) < 3:
            continue
        # The mixed pool takes the first K/2 draws by index, never by score, and the grown
        # version of each. K measured candidates, like the other two pools.
        half = a.k // 2
        mixed = [d for d, g in pairs[:half]] + [g for d, g in pairs[:half]]
        for name, pool in (("minorminer", draws), ("grown", grown), ("mixed", mixed),
                           ("diverse", diverse)):
            best = max(pool, key=lambda t: t[0])
            fresh = measure(task, best[1], ASSESS_BASE + 13 * (k * 7 + hash(name) % 5), a.assess_reads)
            if fresh is not None:
                pools[name][task.name] = fresh
                cost[name].append(float(np.mean([t[2] for t in pool])))

    print("  %d lineages in %.0fs" % (len(pools["minorminer"]), time.time() - started), flush=True)
    print("\n  %-12s %-10s %-9s %s" % ("pool", "ceiling", "qubits", "against the minorminer pool"))
    ref = pools["minorminer"]
    for name in ("minorminer", "grown", "mixed", "diverse"):
        vals = pools[name]
        keys = sorted(set(vals) & set(ref))
        if len(keys) < 5:
            continue
        d = np.array([vals[t] - ref[t] for t in keys])
        rng2 = np.random.default_rng(0)
        bs = [d[rng2.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        print("  %-12s %-10.4f %-9.1f %+.4f [%+.4f, %+.4f] over %d"
              % (name, float(np.mean([vals[t] for t in keys])), float(np.mean(cost[name])),
                 d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(keys)))
    print("\n  Each ceiling is the pool's best candidate by its selection block, re-measured on")
    print("  independent reads. All pools are size K, so the selection inflation is matched.")
    print("\nPOOL CEILING DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
