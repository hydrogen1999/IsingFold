"""Let the actor propose and the critic choose.

Best-of-K only pays if something can tell the K apart, and paying the sampler to score all K is
the expensive way. The specified network already carries a utility critic whose job is exactly
this: predict the terminal reward from a state. This measures whether it can rank K trajectories
the actor proposed, against picking one of them at random and against paying to evaluate all K.
The same structure is worth +0.047 in the other track of this project, where a small model ranks
twenty repaired states without spending a read.

    python3 probes/propose_rank.py --ckpt runs/ladder3_if-core/if-core_s2.pt --family if-core --k 8
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import torch
from isingfold.rl.contracts import Context, InitFailureRecord, Mode, TerminalRecord
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import (LEGACY_ONLINE_INITIALIZER_RESTARTS_V1, EmbeddingEnv,
                              fixed_strength_selector)
from isingfold.rl.model import build_model
from isingfold.rl.proposal import router_initializer

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--family", required=True)
ap.add_argument("--k", type=int, default=8)
ap.add_argument("--instances", type=int, default=50)
ap.add_argument("--reward-reads", type=int, default=128)
ap.add_argument("--qubit-cap", type=int, default=120)
ap.add_argument("--corpus", default="runs/corpus_c4x10")
a = ap.parse_args()

tasks = load_instances(a.corpus)
split = json.loads((Path(a.corpus) / "splits.json").read_text())
dev = set(split["validation"]) | set(split["test"])
dev_tasks = [t for t in tasks if t.lineage in dev][: a.instances]
ctx = Context(qubit_cap=a.qubit_cap)
initializer, selector = router_initializer(), fixed_strength_selector()

model = build_model(a.family, improvement_mode=True)
model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
model.eval()

def rollout(task, seed, rng):
    """One complete episode under the policy; returns (terminal reward, critic value at the last
    decision before termination)."""
    env = EmbeddingEnv(task, ctx, mode=Mode.IMPROVEMENT, initializer=initializer,
                       selector=selector, reward_reads=a.reward_reads, seed=seed,
                       improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1)
    state = env.reset()
    if isinstance(state, InitFailureRecord):
        return None
    last_value = None
    for _ in range(64):
        if isinstance(state, TerminalRecord):
            break
        with torch.no_grad():
            out = model.forward_single(state.observation, None)
        logp = out.masked_log_probs.detach().cpu().numpy()
        finite = np.isfinite(logp)
        if not finite.any():
            return None
        p = np.zeros_like(logp); p[finite] = np.exp(logp[finite]); p /= p.sum()
        last_value = float(out.utility_value.detach().cpu().reshape(-1)[0])
        step = env.step(state, int(rng.choice(len(p), p=p)))
        state = step.next_decision_or_terminal
    if not isinstance(state, TerminalRecord) or state.training_reward is None:
        return None
    return float(state.training_reward), last_value

rows = []
rng = np.random.default_rng(0)
for task in dev_tasks:
    draws = [rollout(task, 5000 + 97 * j, rng) for j in range(a.k)]
    draws = [d for d in draws if d is not None]
    if len(draws) < max(3, a.k // 2):
        continue
    rewards = np.array([d[0] for d in draws]); values = np.array([d[1] for d in draws])
    rows.append(dict(instance=task.name, n=len(draws),
                     critic=float(rewards[int(np.argmax(values))]),
                     random=float(rng.choice(rewards)),
                     oracle=float(rewards.max()), mean=float(rewards.mean()),
                     corr=float(np.corrcoef(values, rewards)[0, 1]) if values.std() > 1e-9 else float("nan")))

if rows:
    C = np.array([r["critic"] for r in rows]); R = np.array([r["random"] for r in rows])
    O = np.array([r["oracle"] for r in rows]); M = np.array([r["mean"] for r in rows])
    corr = np.array([r["corr"] for r in rows]); corr = corr[np.isfinite(corr)]
    rg = np.random.default_rng(0)
    def ci(d):
        bs = [d[rg.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        return "%+.4f [%+.4f, %+.4f]" % (d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5))
    print("%s, %d lineages, K=%d proposed per lineage" % (a.family, len(rows), a.k))
    print("  mean of the K            %.4f" % M.mean())
    print("  random pick of the K     %.4f" % R.mean())
    print("  critic pick of the K     %.4f" % C.mean())
    print("  oracle pick of the K     %.4f" % O.mean())
    print("\n  critic minus random      %s   <- what the critic is worth" % ci(C - R))
    print("  oracle minus random      %s   <- what any ranker could be worth" % ci(O - R))
    print("  median correlation between the critic's value and the realised reward: %+.3f"
          % (np.median(corr) if len(corr) else float("nan")))
