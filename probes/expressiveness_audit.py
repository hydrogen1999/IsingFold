"""Can a linear scorer put a good move on top, at every decision the witness makes?

The witness path is reachable: under the wide support an unhinted replay reaches a valid COMMIT
on every instance tried. The policy does not reach it. Three explanations fit, and they call for
three different weeks of work. The observation may not distinguish a good move from a bad one at
all. One weight vector may be unable to order them even when it does. Or the ordering may exist
and be learnable and the sampling may simply never find it.

This separates the first two from the third without any reinforcement learning, any annealing and
any reward. It caches the exact candidate rows the environment offered at each witness decision,
marks which were consistent with the witness, and then asks the only question that matters for a
scorer: fit a model to put a consistent candidate first, and count how often it does. Top-one
accuracy is reported rather than a loss, because a loss of 2.2 says nothing about whether the
argmax is right, and the whole-path figure is reported with it, because a per-step accuracy of
0.9 over a hundred decisions still completes almost nothing.
"""
import argparse, json, math, os, sys
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
import torch

import constructor_curriculum as cc
from constructor_clone import teacher_steps
from constructor_tiny_gate import no_completion_solver


def top1(model, steps):
    """Fraction of decisions whose highest-scored candidate is witness-consistent.

    The teacher records the consistent candidates as a list of row indices, not as a mask over
    the rows; reading it as a mask indexes past the end whenever fewer rows are consistent than
    offered, which is every decision.
    """
    hits, total, offered, good = 0, 0, 0.0, 0.0
    with torch.no_grad():
        for rows, ok in steps:
            scores = model(torch.as_tensor(rows, dtype=torch.float32)).reshape(-1)
            hits += int(int(torch.argmax(scores)) in set(ok))
            total += 1
            offered += len(rows)
            good += len(ok)
    return hits / total if total else 0.0, total, offered / max(1, total), good / max(1, total)


def fit(model, steps, epochs, lr, seed):
    """Rank loss: push the consistent set above the rest. Same objective for every model class."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    order = np.random.default_rng(seed)
    for _ in range(epochs):
        for i in order.permutation(len(steps)):
            rows, ok = steps[int(i)]
            x = torch.as_tensor(rows, dtype=torch.float32)
            s = model(x).reshape(-1)
            good = torch.as_tensor(list(ok), dtype=torch.long)
            if len(good) == 0 or len(good) == len(s):
                continue
            loss = torch.logsumexp(s, 0) - torch.logsumexp(s[good], 0)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            opt.step()
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cells", default="")
    ap.add_argument("--features", default="physics")
    ap.add_argument("--support", default="wide")
    ap.add_argument("--train", type=int, default=12)
    ap.add_argument("--heldout", type=int, default=6)
    ap.add_argument("--max-steps", type=int, default=900)
    ap.add_argument("--teacher-seconds", type=float, default=300.)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--learning-rate", type=float, default=0.01)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261002)
    a = ap.parse_args()

    cells = [c for c in a.cells.split(",") if c]
    # Only training instances carry a witness, by design: a held-out instance with a witness
    # would leak the answer into every evaluation. So the split here is over the training
    # instances themselves, fitting on the decisions of some and scoring on the decisions of
    # others, which is the same question asked of instances the scorer has not seen.
    train, _ = cc.build_corpus_sets(a.corpus, cells, a.train + a.heldout, 1, a.seed)
    train, heldout = train[:a.train], train[a.train:a.train + a.heldout]
    wide = a.support == "wide"
    width = cc.FEATURE_WIDTHS[a.features]
    print(json.dumps({"probe": "expressiveness_audit", "corpus": a.corpus, "cells": cells,
                      "features": a.features, "channels": width, "support": a.support,
                      "train": len(train), "heldout": len(heldout)}), flush=True)

    def collect(tasks, label):
        out = []
        with no_completion_solver():
            for t in tasks:
                witness = {v: frozenset(c) for v, c in (t.prefix_source or {}).items()}
                if not witness:
                    continue
                fc = cc.make_features(a.features, t)
                steps, reason, valid, progress, secs = teacher_steps(
                    t, witness, fc, a.max_steps, a.teacher_seconds, wide)
                print(json.dumps({"set": label, "task": t.name, "steps": len(steps),
                                  "reason": reason, "valid": valid, "progress": progress}),
                      flush=True)
                if valid and steps:
                    out.append(steps)
        return [s for traj in out for s in traj], len(out)

    tr, tr_paths = collect(train, "train")
    te, te_paths = collect(heldout, "heldout")
    if not tr or not te:
        print(json.dumps({"summary": "expressiveness_audit",
                          "skipped": "no complete teacher path in one of the sets"}), flush=True)
        print("EXPRESSIVENESS AUDIT DONE", flush=True)
        return 0

    print(json.dumps({"train_decisions": len(tr), "train_paths": tr_paths,
                      "heldout_decisions": len(te), "heldout_paths": te_paths}), flush=True)
    print()
    print("  %-10s %10s %10s %12s %14s"
          % ("model", "train top1", "held top1", "chance top1", "whole path"), flush=True)

    rows = {}
    for kind in ("linear", "mlp"):
        torch.manual_seed(a.seed)
        model = cc.make_actor(kind, a.width, width)
        fit(model, tr, a.epochs, a.learning_rate, a.seed)
        tr_acc, _, _, _ = top1(model, tr)
        te_acc, n, mean_offered, mean_good = top1(model, te)
        chance = mean_good / mean_offered if mean_offered else 0.0
        per_path = n / max(1, te_paths)
        whole = te_acc ** per_path
        print("  %-10s %10.3f %10.3f %12.3f %14.2e"
              % (kind, tr_acc, te_acc, chance, whole), flush=True)
        rows[kind] = {"train_top1": tr_acc, "heldout_top1": te_acc, "chance_top1": chance,
                      "decisions_per_path": per_path, "whole_path_probability": whole}

    print(json.dumps({"summary": "expressiveness_audit", "models": rows,
                      "reading": ("a linear model near chance while the small network is far "
                                  "above it implicates capacity; both near chance implicates the "
                                  "observation; both high while reinforcement learning fails "
                                  "implicates exploration and credit assignment")}), flush=True)
    print("EXPRESSIVENESS AUDIT DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
