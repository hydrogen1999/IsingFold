#!/usr/bin/env python3
"""Teach the specified critic what it is supposed to know, by supervision rather than by PPO.

The propose-and-rank probe measured three things about a trained IF-Core: ranking the K
trajectories its actor proposes is worth +0.184 over picking one at random, the critic that is
supposed to do that ranking picks 0.031 *worse* than random, and its value estimate correlates
with the realised reward at -0.19. A critic pointing the wrong way is also the baseline PPO
subtracts to form its advantages, so it is a suspect for the training failures as well as a
missed opportunity at deployment.

This trains the utility critic on its own target directly: roll out episodes, record the state
each one was in before it terminated together with the reward it went on to receive, and fit by
mean squared error. Nothing about the environment, the architecture or the action space changes.

    python3 probes/train_value.py --corpus runs/corpus_c4x10 --family if-core \
        --rollouts 1200 --out runs/value_if-core
"""
import argparse, json, os, sys, time
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
ap.add_argument("--corpus", required=True)
ap.add_argument("--family", default="if-core")
ap.add_argument("--out", required=True)
ap.add_argument("--rollouts", type=int, default=1200)
ap.add_argument("--epochs", type=int, default=8)
ap.add_argument("--minibatch", type=int, default=32)
ap.add_argument("--learning-rate", type=float, default=3e-4)
ap.add_argument("--reward-reads", type=int, default=128)
ap.add_argument("--qubit-cap", type=int, default=120)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--centre-per-lineage", action="store_true",
                help="fit the reward minus its lineage's own mean. The deployed decision ranks K "
                     "trajectories of one problem, and a pointwise fit on absolute reward learns "
                     "which problem is easy instead: measured global correlation +0.36, "
                     "within-instance +0.12, and picking by it loses to chance")
ap.add_argument("--cache", default=None,
                help="pickle of gathered (state, reward) rows; written if absent, reused if "
                     "present, so the three families are judged on identical data")
a = ap.parse_args()

out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
tasks = load_instances(a.corpus)
split = json.loads((Path(a.corpus) / "splits.json").read_text())
train_roots, dev_roots = set(split["train"]), set(split["validation"]) | set(split["test"])
ctx = Context(qubit_cap=a.qubit_cap)
initializer, selector = router_initializer(), fixed_strength_selector()

def make_env(task, seed):
    return EmbeddingEnv(task, ctx, mode=Mode.IMPROVEMENT, initializer=initializer,
                        selector=selector, reward_reads=a.reward_reads, seed=seed,
                        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1)

train = [t for t in tasks if t.lineage in train_roots]
train = [t for t in train
         if all(not isinstance(make_env(t, 1000 + k).reset(), InitFailureRecord) for k in range(3))]
dev = [t for t in tasks if t.lineage in dev_roots][:60]
print(json.dumps({"train": len(train), "dev": len(dev)}), flush=True)

rng = np.random.default_rng(a.seed)

def gather(pool, n):
    """Random rollouts; keep the state each episode was in before it terminated, and its reward."""
    rows, t0 = [], time.time()
    while len(rows) < n:
        task = pool[int(rng.integers(0, len(pool)))]
        env = make_env(task, int(rng.integers(0, 2**31)))
        state = env.reset()
        if isinstance(state, InitFailureRecord):
            continue
        last_obs = None
        for _ in range(64):
            if isinstance(state, TerminalRecord):
                break
            legal = np.flatnonzero(state.legal_mask)
            if not len(legal):
                break
            last_obs = state.observation
            state = env.step(state, int(rng.choice(legal))).next_decision_or_terminal
        if isinstance(state, TerminalRecord) and state.training_reward is not None and last_obs is not None:
            rows.append((last_obs, float(state.training_reward), task.name))
        if len(rows) % 200 == 0 and rows:
            print("  gathered %d/%d (%.0fs)" % (len(rows), n, time.time() - t0), flush=True)
    return rows

import pickle
if a.cache and os.path.exists(a.cache):
    with open(a.cache, "rb") as fh:
        train_rows, dev_rows = pickle.load(fh)
    print("reused %d train and %d dev rows from the cache" % (len(train_rows), len(dev_rows)), flush=True)
else:
    train_rows = gather(train, a.rollouts)
    dev_rows = gather(dev, max(200, a.rollouts // 5))
    if a.cache:
        with open(a.cache, "wb") as fh:
            pickle.dump((train_rows, dev_rows), fh)
        print("wrote the cache to %s" % a.cache, flush=True)
if a.centre_per_lineage:
    def centre(rows_):
        if not rows_ or len(rows_[0]) < 3:
            return rows_
        import collections as _c
        groups = _c.defaultdict(list)
        for r in rows_:
            groups[r[2]].append(r[1])
        means = {k: float(np.mean(v)) for k, v in groups.items()}
        return [(r[0], r[1] - means[r[2]], r[2]) for r in rows_]
    train_rows = centre(train_rows); dev_rows = centre(dev_rows)
y_tr = np.array([r[1] for r in train_rows]); y_de = np.array([r[1] for r in dev_rows])
print("targets: train mean %.4f sd %.4f | dev mean %.4f sd %.4f"
      % (y_tr.mean(), y_tr.std(), y_de.mean(), y_de.std()), flush=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(a.seed)
model = build_model(a.family, improvement_mode=True).to(device)
opt = torch.optim.Adam(model.parameters(), lr=a.learning_rate)

def evaluate(rows):
    model.eval(); preds = []
    with torch.no_grad():
        for start in range(0, len(rows), a.minibatch):
            block = rows[start : start + a.minibatch]
            out_ = model([r[0] for r in block], device=device)
            preds.extend(out_.utility_value.detach().cpu().reshape(-1).tolist())
    p = np.array(preds); y = np.array([r[1] for r in rows])
    ss = 1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return ss, float(np.corrcoef(p, y)[0, 1]) if p.std() > 1e-9 else float("nan")

history = []
for epoch in range(a.epochs):
    model.train()
    order = rng.permutation(len(train_rows)); losses = []
    for start in range(0, len(order), a.minibatch):
        block = [train_rows[int(i)] for i in order[start : start + a.minibatch]]
        out_ = model([r[0] for r in block], device=device)
        target = torch.as_tensor([r[1] for r in block], dtype=torch.float32, device=device)
        loss = torch.nn.functional.mse_loss(out_.utility_value.reshape(-1), target)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        losses.append(float(loss.detach()))
    r2, corr = evaluate(dev_rows)
    rec = dict(epoch=epoch, loss=float(np.mean(losses)), dev_r2=r2, dev_corr=corr)
    history.append(rec); print(json.dumps(rec), flush=True)
    torch.save(model.state_dict(), out / "value.pt")
    (out / "history.json").write_text(json.dumps(history, indent=1, default=float))
