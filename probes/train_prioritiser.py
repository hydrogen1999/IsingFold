"""Train the shortlist prioritiser on imitation records from hinted witness replays.

Each record is one decision: the chains, every offered candidate, the legal mask and the
teacher's pick. The model scores each candidate from its local features and is trained with
a softmax cross-entropy over the legal candidates of the decision. Held-out is by instance.
The reported numbers are teacher agreement (top-1) and the rank of the teacher's pick, on
held-out decisions. The model is then used as the generator's preference in witness_replay.py
--scorer, where the number that matters is measured: unhinted validity at corpus scale.
"""
import argparse, json, os, pickle, sys, time
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
import torch
import torch.nn as nn
from isingfold.rl.data.generate import load_instances

from candidate_features import WIDTH, FeatureContext


class Prioritiser(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(WIDTH, width), nn.SiLU(), nn.Linear(width, width),
                                 nn.SiLU(), nn.Linear(width, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def featurise(records, tasks_by_name, limit_per_task=0):
    ctxs = {}
    out = []
    counts = {}
    for r in records:
        name = r["task"]
        if limit_per_task and counts.get(name, 0) >= limit_per_task:
            continue
        counts[name] = counts.get(name, 0) + 1
        if name not in ctxs:
            ctxs[name] = FeatureContext(tasks_by_name[name])
        fc = ctxs[name]
        chains = {v: frozenset(c) for v, c in r["chains"].items()}
        feats = np.stack([fc.candidate(c, chains) for c in r["candidates"]])
        legal = np.asarray(r["legal"], dtype=bool)
        out.append((name, feats, legal, int(r["teacher"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, nargs="+")
    ap.add_argument("--corpus", required=True, nargs="+")
    ap.add_argument("--holdout-fraction", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--limit-per-task", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tasks_by_name = {}
    for c in a.corpus:
        for t in load_instances(c):
            tasks_by_name[t.name] = t
    records = []
    for d in a.dump:
        with open(d, "rb") as fh:
            records += pickle.load(fh)["records"]
    names = sorted({r["task"] for r in records})
    rng = np.random.default_rng(a.seed)
    held = set(rng.choice(names, size=max(1, int(len(names) * a.holdout_fraction)), replace=False))
    t0 = time.time()
    data = featurise(records, tasks_by_name, a.limit_per_task)
    train = [d for d in data if d[0] not in held]
    test = [d for d in data if d[0] in held]
    print(json.dumps({"records": len(records), "instances": len(names), "held_out": len(held),
                      "train_decisions": len(train), "test_decisions": len(test),
                      "featurise_seconds": round(time.time() - t0)}), flush=True)
    torch.manual_seed(a.seed)
    model = Prioritiser(a.width)
    opt = torch.optim.Adam(model.parameters(), lr=a.learning_rate)

    def evaluate(rows):
        model.eval()
        top1, ranks = [], []
        with torch.no_grad():
            for _, feats, legal, teacher in rows:
                s = model(torch.as_tensor(feats)).numpy()
                s[~legal] = -1e9
                order = np.argsort(-s)
                rank = int(np.where(order == teacher)[0][0])
                top1.append(rank == 0)
                ranks.append(rank)
        model.train()
        return float(np.mean(top1)), float(np.mean(ranks)), float(np.median(ranks))

    for epoch in range(a.epochs):
        rng.shuffle(train)
        total = 0.0
        for _, feats, legal, teacher in train:
            s = model(torch.as_tensor(feats))
            s = s.masked_fill(~torch.as_tensor(legal), -1e9)
            loss = nn.functional.cross_entropy(s.unsqueeze(0), torch.tensor([teacher]))
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss)
        if epoch % 5 == 0 or epoch == a.epochs - 1:
            tr = evaluate(train[:2000]); te = evaluate(test)
            print("  epoch %3d  loss %.4f  train top1 %.3f rank %.1f | held-out top1 %.3f rank %.1f median %.0f"
                  % (epoch, total / max(1, len(train)), tr[0], tr[1], te[0], te[1], te[2]), flush=True)
    torch.save({"state": model.state_dict(), "width": a.width}, a.out)
    print("  wrote %s" % a.out)
    print("\nPRIORITISER DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
