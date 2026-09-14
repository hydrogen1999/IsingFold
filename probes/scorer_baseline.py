"""The control that decides whether the contribution is sequential control or just selection.

The review's sharpest point: if a plain supervised scorer, fed the same candidate embeddings,
picks as well as the RL policy improves, then what the project has found is that learned
selection works, not that a learned sequential controller is necessary. That control has never
been run here, so the RL claim has nothing holding it up.

This probe runs it. On the training lineages it draws minorminer embeddings, measures each one,
and fits a gradient-boosted regressor from cheap structural features to measured utility. On the
held-out lineages it then picks among fresh draws three ways:

    resource        fewest qubits, then shortest chain      0 evaluator blocks
    scorer          the fitted model's prediction           0 evaluator blocks
    measured        the largest measured block              N evaluator blocks

and every pick is assessed on independent reads that took no part in choosing it. The scorer
column is the interesting one: it spends no annealer time at all, which is the resource the
application actually pays for.
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

from _initializers import minorminer_initializer

SELECT_BASE = 5_000
ASSESS_BASE = 90_000_000


def fixed_initializer(chains):
    def _init(logical, host, seed):
        return chains
    return _init


def measure(task, ctx, chains, seed, reads):
    try:
        out = run_controller([task], ctx, first_commit_controller,
                             initializer=fixed_initializer(chains),
                             selector=fixed_strength_selector(), reward_reads=reads,
                             repetitions=1, seed=seed)
    except Exception:
        return None
    o = out[0]
    if not o.returned_valid or o.utility is None:
        return None
    return float(o.utility)


def features(chains, task):
    """Cheap structural description of one embedding. Nothing here touches the ground energy.

    The first five are what a resource-first embedder looks at. The rest describe how the
    couplings are realised, which is where an objective-first view expects the signal to be:
    a logical edge carried by one coupler is a bottleneck no matter how short the chains are.
    """
    host, logical, problem = task.host, task.logical, task.problem
    lens = np.array([len(c) for c in chains.values()], dtype=float)
    owner = {q: v for v, c in chains.items() for q in c}
    contacts, weighted = [], []
    for u, v in logical.edges():
        k = sum(1 for q in chains[u] for r in host.neighbors(q) if owner.get(r) == v)
        contacts.append(k)
        j = abs(problem.j.get((u, v), problem.j.get((v, u), 0.0)))
        weighted.append(j / k if k else 0.0)
    contacts = np.array(contacts, dtype=float) if contacts else np.zeros(1)
    weighted = np.array(weighted, dtype=float) if weighted else np.zeros(1)
    internal = np.array([host.subgraph(c).number_of_edges() for c in chains.values()], dtype=float)
    return [
        float(lens.sum()), float(lens.max()), float(lens.mean()), float(lens.std()),
        float((lens > lens.mean() + lens.std()).sum()),
        float(contacts.min()), float(contacts.mean()), float((contacts == 1).mean()),
        float(weighted.max()), float(weighted.mean()),
        float((internal / np.maximum(lens - 1, 1)).mean()),
        float(len(chains)), float(logical.number_of_edges()),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="runs/corpus_c4x10")
    ap.add_argument("--train-lineages", type=int, default=120)
    ap.add_argument("--test-lineages", type=int, default=60)
    ap.add_argument("--train-draws", type=int, default=6)
    ap.add_argument("--ladder", default="2,4,8,16")
    ap.add_argument("--qubit-cap", type=int, default=120)
    ap.add_argument("--reads", type=int, default=128)
    ap.add_argument("--assess-reads", type=int, default=512)
    ap.add_argument("--mm-tries", type=int, default=10)
    a = ap.parse_args()

    tasks = load_instances(a.corpus)
    split = json.loads((Path(a.corpus) / "splits.json").read_text())
    train_roots, test_roots = set(split["train"]), set(split["test"])
    train = [t for t in tasks if t.lineage in train_roots][: a.train_lineages]
    test = [t for t in tasks if t.lineage in test_roots][: a.test_lineages]
    ctx = Context(qubit_cap=a.qubit_cap)
    mm = minorminer_initializer(a.mm_tries)
    ladder = [int(x) for x in a.ladder.split(",") if x]
    print(json.dumps({"train": len(train), "test": len(test), "train_draws": a.train_draws,
                      "ladder": ladder}), flush=True)

    started = time.time()
    X, y, groups = [], [], []
    for t in train:
        for j in range(a.train_draws):
            chains = mm(t.logical, t.host, SELECT_BASE + 97 * j)
            if chains is None:
                continue
            u = measure(t, ctx, chains, SELECT_BASE + 97 * j, a.reads)
            if u is None:
                continue
            X.append(features(chains, t)); y.append(u); groups.append(t.lineage)
    print("  training rows %d over %d lineages, %.1fs"
          % (len(X), len(set(groups)), time.time() - started), flush=True)

    from sklearn.ensemble import GradientBoostingRegressor
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                      subsample=0.9, random_state=0)
    model.fit(np.array(X), np.array(y))

    # Within-lineage rank quality on training draws, which is the only thing selection needs.
    by_lineage = {}
    for row, target, g in zip(X, y, groups):
        by_lineage.setdefault(g, []).append((model.predict([row])[0], target))
    hits, total = 0, 0
    for g, rows in by_lineage.items():
        if len(rows) < 2:
            continue
        pick = max(rows, key=lambda r: r[0])[1]
        total += 1
        hits += 1 if pick >= max(r[1] for r in rows) - 1e-12 else 0
    print("  on its own training lineages the scorer picks the best draw %d/%d times"
          % (hits, total), flush=True)

    print("\n== held-out lineages, every pick assessed on %d fresh reads" % a.assess_reads)
    results = {}
    pool = {}
    biggest = max(ladder)
    started = time.time()
    for t in test:
        draws = []
        for j in range(biggest):
            chains = mm(t.logical, t.host, SELECT_BASE + 97 * j)
            if chains is None:
                continue
            u = measure(t, ctx, chains, SELECT_BASE + 97 * j, a.reads)
            if u is None:
                continue
            draws.append({"chains": chains, "select": u, "j": j,
                          "f": features(chains, t)})
        pool[t.name] = draws
    print("  drew and scored %d per lineage in %.1fs" % (biggest, time.time() - started),
          flush=True)

    def assess(t, chains, tag):
        return measure(t, ctx, chains, ASSESS_BASE + 13 * tag, a.assess_reads)

    for n in ladder:
        for name, key, blocks in (
            ("resource", lambda d: (-d["f"][0], -d["f"][1]), 0),
            ("scorer", lambda d: model.predict([d["f"]])[0], 0),
            ("measured", lambda d: d["select"], n),
        ):
            vals = {}
            for t in test:
                draws = pool[t.name][:n]
                if not draws:
                    vals[t.name] = None; continue
                win = max(draws, key=key)
                vals[t.name] = assess(t, win["chains"], hash((name, n, win["j"])) % 100000)
            ok = [v for v in vals.values() if v is not None]
            results[(name, n)] = vals
            print("  best-of-%-2d by %-9s assessed %.4f over %d lineages, %d blocks spent"
                  % (n, name, float(np.mean(ok)) if ok else float("nan"), len(ok), blocks),
                  flush=True)

    def paired(label, arm, ref, note):
        keys = sorted(set(arm) & set(ref))
        d = np.array([(0.0 if arm[k] is None else arm[k]) - (0.0 if ref[k] is None else ref[k])
                      for k in keys])
        if len(d) < 5:
            print("  %-50s (too few)" % label); return
        rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        print("  %-50s n %3d  %+.4f [%+.4f, %+.4f]  (%s)"
              % (label, len(d), d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), note),
              flush=True)

    print("\n== what the scorer is worth, and what measuring adds on top of it")
    for n in ladder:
        paired("best-of-%d: scorer minus resource" % n, results[("scorer", n)],
               results[("resource", n)], "both spend zero annealer blocks")
        paired("best-of-%d: measured minus scorer" % n, results[("measured", n)],
               results[("scorer", n)], "measuring costs %d blocks, the scorer costs none" % n)

    print("\nSCORER DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
