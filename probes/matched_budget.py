#!/usr/bin/env python3
"""The bar Track B has to clear: beat best-of-K random rollouts at the same evaluation budget.

Returning the protected initializer is too low a bar, because the headroom probe shows random
rollouts already scatter widely around it and their best is far above it. A learned policy that
merely beats the initializer has not shown it learned anything a sampler could not stumble into.
The comparison here gives both arms the same number of episodes per instance: the policy is
rolled out K times and its best kept, and a random masked controller is rolled out K times and
its best kept, and the two bests are paired.

    python3 probes/matched_budget.py --corpus runs/corpus_c4x10 --model runs/ladder6_if-core_s0/if-core_s0.pt \
        --family if-core --k 8 --instances 40
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

_pin = os.environ.get("ISINGFOLD_SRC")
if _pin:
    sys.path.insert(0, _pin)
    sys.meta_path[:] = [f for f in sys.meta_path
                        if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                                and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]

import torch
from isingfold.rl.contracts import Context
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.evaluate import (first_commit_controller, quality_aware_controller,
                                   random_masked_controller, run_controller, secondary_metrics,
                                   torch_controller)
from isingfold.rl.model import build_model
from isingfold.rl.proposal import router_initializer

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--family", required=True)
ap.add_argument("--k", type=int, default=8, help="rollouts per instance for each arm")
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
common = dict(initializer=router_initializer(), selector=fixed_strength_selector(),
              reward_reads=a.reward_reads, repetitions=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(a.family, improvement_mode=True)
model.load_state_dict(torch.load(a.model, map_location=device))
model.to(device).eval()
policy = torch_controller(model, device)

def best_of(task, controller, k, tag):
    best = None
    for j in range(k):
        try:
            out = run_controller([task], ctx, controller, seed=5000 + 97 * j, **common)
        except Exception:
            continue
        m = secondary_metrics(out)
        if m["valid_returns"] and (best is None or m["utility_mean"] > best):
            best = m["utility_mean"]
    return best

rows = []
for task in dev_tasks:
    try:
        base = secondary_metrics(run_controller([task], ctx, first_commit_controller, seed=17, **common))
    except Exception:
        continue
    if not base["valid_returns"]:
        continue
    p = best_of(task, policy, a.k, "policy")
    r = best_of(task, random_masked_controller, a.k, "random")
    try:
        h = secondary_metrics(run_controller([task], ctx, quality_aware_controller, seed=17, **common))["utility_mean"]
    except Exception:
        h = None
    if p is None or r is None:
        continue
    rows.append(dict(instance=task.name, initial=base["utility_mean"], policy_best=p,
                     random_best=r, heuristic=h))

if rows:
    P = np.array([r["policy_best"] for r in rows]); R = np.array([r["random_best"] for r in rows])
    I = np.array([r["initial"] for r in rows])
    H = np.array([r["heuristic"] for r in rows if r["heuristic"] is not None])
    rng = np.random.default_rng(0)
    def ci(d):
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        return "%+.4f [%+.4f, %+.4f]" % (d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5))
    print("\n%d lineages, %d rollouts per arm per instance" % (len(rows), a.k))
    print("  policy best-of-%d          %.4f" % (a.k, P.mean()))
    print("  random best-of-%d          %.4f" % (a.k, R.mean()))
    print("  returning the initializer %.4f" % I.mean())
    if len(H): print("  hand-written heuristic    %.4f" % H.mean())
    print("\n  policy minus random, both best-of-%d: %s   <- the bar" % (a.k, ci(P - R)))
    print("  policy minus initializer:            %s" % ci(P - I))
    if a.out: json.dump(rows, open(a.out, "w"))
