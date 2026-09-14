"""Paired comparison of every deployment rule available to a trained policy.

The controller can take its mode, sample once, or sample K times and keep the best. Those are
three different methods with three different costs, and the project has only ever reported the
middle one. Each is paired on the same lineages against returning the protected initializer and
against a random masked controller given the same number of episodes.
"""
import json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import torch
from isingfold.rl.contracts import Context
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.evaluate import (first_commit_controller, random_masked_controller,
                                   run_controller, secondary_metrics, torch_controller)
from isingfold.rl.model import build_model
from isingfold.rl.proposal import router_initializer

K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
tasks = load_instances("runs/corpus_c4x10")
split = json.loads(Path("runs/corpus_c4x10/splits.json").read_text())
dev = set(split["validation"]) | set(split["test"])
dev_tasks = [t for t in tasks if t.lineage in dev][:N]
ctx = Context(qubit_cap=120)
common = dict(initializer=router_initializer(), selector=fixed_strength_selector(),
              reward_reads=128, repetitions=1)

def per_instance(controller, seed):
    out = {}
    for t in dev_tasks:
        try:
            m = secondary_metrics(run_controller([t], ctx, controller, seed=seed, **common))
        except Exception:
            continue
        if m["valid_returns"]:
            out[t.name] = m["utility_mean"]
    return out

def best_of(controller_factory, k):
    acc = {}
    for j in range(k):
        d = per_instance(controller_factory(j), 5000 + 97 * j)
        for name, v in d.items():
            acc[name] = max(acc.get(name, -1e9), v)
    return acc

def report(label, arm, ref):
    common_keys = sorted(set(arm) & set(ref))
    if len(common_keys) < 5:
        print("  %-34s (too few paired instances)" % label); return
    d = np.array([arm[i] - ref[i] for i in common_keys])
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    print("  %-34s n %3d  %+.4f [%+.4f, %+.4f]" % (label, len(d), d.mean(),
          np.percentile(bs, 2.5), np.percentile(bs, 97.5)))

initial = per_instance(first_commit_controller, 17)
rand1 = per_instance(random_masked_controller, 17)
randk = best_of(lambda j: random_masked_controller, K)
print("reference utilities: initializer %.4f | random once %.4f | random best-of-%d %.4f"
      % (np.mean(list(initial.values())), np.mean(list(rand1.values())), K, np.mean(list(randk.values()))), flush=True)

for fam, ckpt in (("if-core", "runs/bestof2_if-core_s0/policy.pt"),
                  ("if-dual", "runs/bestof2_if-dual_s0/policy_r25.pt"),
                  ("if-mlp", "runs/bestof2_if-mlp_s0/policy.pt")):
    if not os.path.exists(ckpt):
        continue
    m = build_model(fam, improvement_mode=True)
    m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.eval()
    print("\n%s" % fam, flush=True)
    greedy = per_instance(torch_controller(m, None, greedy=True), 17)
    once = per_instance(torch_controller(m, None, greedy=False), 17)
    bk = best_of(lambda j, mm=m: torch_controller(mm, None, greedy=False), K)
    report("mode, against the initializer", greedy, initial)
    report("one sample, against the initializer", once, initial)
    report("best-of-%d, against the initializer" % K, bk, initial)
    report("best-of-%d, against random best-of-%d" % (K, K), bk, randk)
