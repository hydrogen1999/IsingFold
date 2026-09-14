"""If the value is in the tail, the sampling temperature is a first-class knob.

The mode of every trained policy here is COMMIT, so argmax reproduces the initializer exactly and
the policy's worth lies in the actions it takes occasionally. That makes deployment a best-of-K
draw, and it makes the temperature that shapes those draws a parameter to choose rather than a
constant to inherit. This sweeps it on held-out lineages, at a fixed episode budget, against a
random controller given the same budget.
"""
import json, math, os, sys
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
from isingfold.rl.evaluate import (EvaluationProtocolError, first_commit_controller,
                                   random_masked_controller, run_controller, secondary_metrics)
from isingfold.rl.model import build_model
from isingfold.rl.proposal import router_initializer

K = int(sys.argv[1]) if len(sys.argv) > 1 else 8
N = int(sys.argv[2]) if len(sys.argv) > 2 else 60

def tempered(model, temperature):
    def _controller(decision, rng):
        with torch.no_grad():
            out = model.forward_single(decision.observation, None)
        logp = out.masked_log_probs.detach().cpu().numpy()
        finite = np.isfinite(logp)
        if not finite.any():
            raise EvaluationProtocolError("model produced an all-false support")
        scaled = np.full_like(logp, -np.inf)
        scaled[finite] = logp[finite] / max(temperature, 1e-6)
        scaled[finite] -= scaled[finite].max()
        p = np.zeros_like(logp)
        p[finite] = np.exp(scaled[finite])
        p /= p.sum()
        return int(rng.choice(len(p), p=p))
    return _controller

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

def best_of(make, k):
    acc = {}
    for j in range(k):
        for name, v in per_instance(make(j), 5000 + 97 * j).items():
            acc[name] = max(acc.get(name, -1e9), v)
    return acc

def paired(arm, ref, label):
    keys = sorted(set(arm) & set(ref))
    if len(keys) < 5:
        print("  %-28s too few" % label); return None
    d = np.array([arm[i] - ref[i] for i in keys]); rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    print("  %-28s n %3d  %+.4f [%+.4f, %+.4f]" % (label, len(d), d.mean(),
          np.percentile(bs, 2.5), np.percentile(bs, 97.5)), flush=True)
    return d.mean()

initial = per_instance(first_commit_controller, 17)
randk = best_of(lambda j: random_masked_controller, K)
print("initializer %.4f | random best-of-%d %.4f\n"
      % (np.mean(list(initial.values())), K, np.mean(list(randk.values()))), flush=True)

for fam, ckpt in (("if-core", "runs/ladder3_if-core/if-core_s2.pt"),
                  ("if-dual", "runs/ladder3_if-dual/if-dual_s1.pt"),
                  ("if-mlp", "runs/ladder3_if-mlp/if-mlp_s0.pt")):
    if not os.path.exists(ckpt):
        continue
    m = build_model(fam, improvement_mode=True)
    m.load_state_dict(torch.load(ckpt, map_location="cpu")); m.eval()
    print(fam, flush=True)
    for T in (0.5, 0.75, 1.0, 1.5, 2.0):
        bk = best_of(lambda j, mm=m, t=T: tempered(mm, t), K)
        paired(bk, randk, "T=%.2f, best-of-%d vs random" % (T, K))
