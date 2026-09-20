"""Is the branch-point signal learnable, and does the existing observation carry it?

The branch measurement says a single construction decision moves the final residual by 0.057
while the policy picks the better action 0.298 of the time against a chance rate of 0.250. So the
quality is there and the policy's ranking does not see it. Before any of it is worth a week of
reinforcement learning, one question has to be answered with the records already collected: can
any model, fitted to the measured orderings, put the better action on top on branch points it was
not fitted to?

A pass says the observation carries the decision and the learning rule was the problem. A failure
says the rows the environment offers do not distinguish a good branch from a bad one, and no
reward schedule will fix that. No sampler is called here; the residuals were measured when the
branches were run.
"""
import argparse, json, math, statistics, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


def fit(model, records, epochs, lr, seed):
    """Rank the measured-best action of each branch point above the others."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    order = np.random.default_rng(seed)
    for _ in range(epochs):
        for i in order.permutation(len(records)):
            r = records[int(i)]
            rows = torch.as_tensor(np.asarray(r["rows"], dtype=np.float32))
            measured = {int(k): v for k, v in r["residual_by_action"].items()}
            best = min(measured, key=measured.get)
            scored = torch.as_tensor(sorted(measured), dtype=torch.long)
            s = model(rows).reshape(-1)[scored]
            target = torch.as_tensor([sorted(measured).index(best)], dtype=torch.long)
            loss = torch.nn.functional.cross_entropy(s[None, :], target)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
    return model


def score(model, records):
    """How often the model's top choice among the compared actions is the measured best."""
    hits, total, chance = 0, 0, 0.0
    with torch.no_grad():
        for r in records:
            rows = torch.as_tensor(np.asarray(r["rows"], dtype=np.float32))
            measured = {int(k): v for k, v in r["residual_by_action"].items()}
            keys = sorted(measured)
            # The fit raises the score of the measured-best action, so the model's choice is
            # its highest score, not its lowest. Negating here reversed it, and the giveaway was
            # an accuracy of 0.005 on the very records the model had just been fitted to.
            s = model(rows).reshape(-1)[torch.as_tensor(keys, dtype=torch.long)]
            hits += int(keys[int(torch.argmax(s))] == min(measured, key=measured.get))
            total += 1
            chance += 1.0 / len(keys)
    return hits / total, chance / total, total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261004)
    a = ap.parse_args()

    records = [json.loads(l) for l in Path(a.records).read_text().splitlines() if l.strip()]
    records = [r for r in records if len(r.get("residual_by_action", {})) >= 2]
    tasks = sorted({r["task"] for r in records})
    print(json.dumps({"probe": "branch_learnable", "records": len(records),
                      "instances": len(tasks), "folds": a.folds}), flush=True)
    if len(tasks) < a.folds:
        a.folds = max(2, len(tasks))

    import constructor_curriculum as cc
    print("  %-10s %12s %12s %12s" % ("model", "held-out", "chance", "fitted"), flush=True)
    out = {}
    for kind in ("linear", "mlp"):
        held, fitted, chances = [], [], []
        rng = np.random.default_rng(a.seed)
        shuffled = [tasks[i] for i in rng.permutation(len(tasks))]
        # Folds are grouped by instance, so a branch point is never scored by a model that saw
        # another branch point of the same construction.
        for f in range(a.folds):
            test_tasks = set(shuffled[f::a.folds])
            tr = [r for r in records if r["task"] not in test_tasks]
            te = [r for r in records if r["task"] in test_tasks]
            if not tr or not te:
                continue
            torch.manual_seed(a.seed + f)
            model = cc.make_actor(kind, a.width, a.channels)
            fit(model, tr, a.epochs, a.learning_rate, a.seed + f)
            h, c, _ = score(model, te)
            g, _, _ = score(model, tr)
            held.append(h); chances.append(c); fitted.append(g)
        n = len(held)
        m = sum(held) / n
        sd = statistics.stdev(held) if n > 1 else 0.0
        print("  %-10s %12.3f %12.3f %12.3f" % (kind, m, sum(chances) / n, sum(fitted) / n),
              flush=True)
        out[kind] = {"heldout": m, "heldout_sd": sd, "chance": sum(chances) / n,
                     "fitted": sum(fitted) / n, "folds": n}
    print(json.dumps({"summary": "branch_learnable", "models": out,
                      "policy_pick_rate_for_reference": 0.298,
                      "reading": ("held-out well above chance says the offered rows carry the "
                                  "branch decision and the learning rule was what could not see "
                                  "it; held-out at chance while fitted is high says the rows do "
                                  "not distinguish a good branch and no reward schedule fixes "
                                  "that")}), flush=True)
    print("BRANCH LEARNABLE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
