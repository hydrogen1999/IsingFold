#!/usr/bin/env python3
"""How much is there for a policy to win in this environment at all?

Before asking which architecture learns best, ask whether the environment rewards any policy.
For each development lineage this rolls out many random masked controllers, records the terminal
utility of each, and compares the best of them with the mean and with returning the protected
initializer. The gap between the best rollout and the mean rollout is what a perfect policy
could capture with this action set and this decision budget; the gap between the mean and the
initializer is what undirected search is worth. If the first gap is small, no model will show
anything here, and the environment is the thing to change.

    python3 probes/headroom.py --corpus runs/corpus_c4x10 --rollouts 64 --instances 40
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

_pin = os.environ.get("ISINGFOLD_SRC")
if _pin:
    sys.path.insert(0, _pin)
    sys.meta_path[:] = [f for f in sys.meta_path
                        if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                                and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]

from isingfold.rl.contracts import Context, Mode
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import LEGACY_ONLINE_INITIALIZER_RESTARTS_V1, EmbeddingEnv, fixed_strength_selector
from isingfold.rl.evaluate import (first_commit_controller, quality_aware_controller,
                                   random_masked_controller, run_controller, secondary_metrics)
from isingfold.rl.proposal import router_initializer

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--rollouts", type=int, default=64)
ap.add_argument("--instances", type=int, default=40)
ap.add_argument("--reward-reads", type=int, default=128)
ap.add_argument("--qubit-cap", type=int, default=120)
ap.add_argument("--out", default=None)
a = ap.parse_args()

tasks = load_instances(a.corpus)
split = json.loads((Path(a.corpus) / "splits.json").read_text())
dev = set(split["validation"]) | set(split["test"])
dev_tasks = [t for t in tasks if t.lineage in dev][: a.instances]
ctx = Context(qubit_cap=a.qubit_cap)
init = router_initializer()
sel = fixed_strength_selector()
common = dict(initializer=init, selector=sel, reward_reads=a.reward_reads, repetitions=1)

print(json.dumps({"dev_tasks": len(dev_tasks), "rollouts": a.rollouts}), flush=True)
rows = []
t0 = time.time()
for i, task in enumerate(dev_tasks):
    try:
        base = run_controller([task], ctx, first_commit_controller, seed=17, **common)
        heur = run_controller([task], ctx, quality_aware_controller, seed=17, **common)
    except Exception:
        continue
    b = secondary_metrics(base); h = secondary_metrics(heur)
    if not b["valid_returns"]:
        continue
    us = []
    for k in range(a.rollouts):
        try:
            out = run_controller([task], ctx, random_masked_controller, seed=1000 + k, **common)
        except Exception:
            continue
        m = secondary_metrics(out)
        if m["valid_returns"]:
            us.append(m["utility_mean"])
    if len(us) < 8:
        continue
    us = np.array(us)
    rows.append(dict(instance=task.name, initial=b["utility_mean"], heuristic=h["utility_mean"],
                     mean=float(us.mean()), best=float(us.max()), worst=float(us.min()),
                     sd=float(us.std()), n=len(us)))
    if (i + 1) % 5 == 0:
        r = rows[-1]
        print("%3d/%d %-28s initial %.3f  random mean %.3f  best %.3f  sd %.3f  (%.0fs)"
              % (i + 1, len(dev_tasks), r["instance"], r["initial"], r["mean"], r["best"], r["sd"],
                 time.time() - t0), flush=True)

if rows:
    A = {k: np.array([r[k] for r in rows]) for k in ("initial", "heuristic", "mean", "best", "worst", "sd")}
    print("\n%d lineages, %d rollouts each" % (len(rows), a.rollouts))
    print("  returning the initializer      %.4f" % A["initial"].mean())
    print("  hand-written quality heuristic %.4f  (%+.4f)" % (A["heuristic"].mean(), (A["heuristic"] - A["initial"]).mean()))
    print("  random policy, mean            %.4f  (%+.4f)" % (A["mean"].mean(), (A["mean"] - A["initial"]).mean()))
    print("  random policy, best rollout    %.4f  (%+.4f)" % (A["best"].mean(), (A["best"] - A["initial"]).mean()))
    print("  random policy, worst rollout   %.4f" % A["worst"].mean())
    print("\n  HEADROOM, best minus mean of the rollouts: %+.4f" % (A["best"] - A["mean"]).mean())
    print("  spread within an instance, sd:             %.4f" % A["sd"].mean())
    print("  share of lineages where best > initial:    %.0f%%" % (100 * np.mean(A["best"] > A["initial"] + 1e-9)))
    if a.out:
        json.dump(rows, open(a.out, "w"))
