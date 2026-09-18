"""Is the constructor's extra spend removable, and does removing it recover the quality?

The learned constructor commits 84.8 qubits where minorminer commits 31.3, and its samples are
worse by a factor of two. Two explanations fit that equally well and they call for different
fixes. The growth may be unnecessary, in which case the embedding contains qubits that can be
deleted while every logical contact survives, and deleting them should recover quality. Or the
growth may be compensating for placements that leave variables far apart, in which case almost
nothing is deletable and the fix belongs earlier in the episode.

This runs the policy, prunes each committed embedding greedily while keeping every chain
connected and every logical edge realised, and measures both the original and the pruned version
on independent read blocks. minorminer is measured on the same instances as the reference. It is
a diagnostic of the learned output, not a method: nothing here is offered as a way to deploy.
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
from isingfold.rl.evaluator import sample_program
from isingfold.rl.program import compile_program, strength_registry

import constructor_curriculum as cc
from constructor_rollout import episode
from _context import REGISTERED_BETA_RANGE, host_context
from _initializers import minorminer_initializer
from witness_prune import prune


def measure(task, chains, ctx, index, reads, seed):
    strengths = strength_registry(task.problem, ctx.strength_ratios, ctx.epsilon_strength)
    program = compile_program(chains, task.host, task.problem, strengths[index], index)
    block = sample_program(program, chains, task.problem, task.ground_energy, num_reads=reads,
                           seed=int(seed), num_sweeps=200, beta_range=REGISTERED_BETA_RANGE)
    return block.mean_residual, block.rate


def shape(chains):
    sizes = [len(c) for c in chains.values()]
    return sum(sizes), max(sizes), sum(sizes) / len(sizes)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cells", default="")
    ap.add_argument("--init", required=True)
    ap.add_argument("--features", default="local")
    ap.add_argument("--actor", default="linear")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--support", default="wide")
    ap.add_argument("--n-tasks", type=int, default=8)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--episode-seconds", type=float, default=90.)
    ap.add_argument("--reads", type=int, default=4096)
    ap.add_argument("--strength-index", type=int, default=1)
    ap.add_argument("--mm-tries", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260927)
    a = ap.parse_args()

    cells = [c for c in a.cells.split(",") if c]
    tasks, _ = cc.build_corpus_sets(a.corpus, cells, a.n_tasks, 1, a.seed)
    actor = cc.make_actor(a.actor, a.width, cc.FEATURE_WIDTHS[a.features])
    cc.load_init(a.init, actor, a.actor, a.features)
    mm = minorminer_initializer(a.mm_tries)
    wide = a.support == "wide"

    print(json.dumps({"probe": "policy_excess", "corpus": a.corpus, "cells": cells,
                      "init": a.init, "features": a.features, "support": a.support,
                      "tasks": len(tasks), "episodes": a.episodes, "reads": a.reads}), flush=True)
    print("  %-28s %6s %6s %6s %7s %8s %8s %8s"
          % ("task", "vars", "policy", "pruned", "minmin", "resid pol", "resid prn", "resid mm"),
          flush=True)

    rows = []
    for k, t in enumerate(tasks):
        ctx = host_context(t.host.number_of_nodes())
        fc = cc.make_features(a.features, t)
        found = None
        for e in range(a.episodes):
            with torch.no_grad():
                rec = episode(t, actor, fc, 1., a.max_steps,
                              np.random.default_rng(a.seed + 100 * k + e), a.episode_seconds,
                              train=True, objective="quality", evaluate_reward=False, wide=wide)
            if rec["valid"]:
                found = rec["terminal"].embedding
                break
        if found is None:
            print(json.dumps({"task": t.name, "skipped": "no valid COMMIT in %d episodes" % a.episodes}),
                  flush=True)
            continue
        chains = {v: frozenset(c) for v, c in found.items()}
        tight = prune(t.host, chains, t.logical)
        draw = mm(t.logical, t.host, a.seed + k)
        mm_chains = {v: frozenset(draw[v]) for v in t.logical.nodes()} if draw else None

        seed = a.seed + 10000 * k
        rp, pp = measure(t, chains, ctx, a.strength_index, a.reads, seed)
        rt, pt = measure(t, tight, ctx, a.strength_index, a.reads, seed + 1)
        rm, pm = (measure(t, mm_chains, ctx, a.strength_index, a.reads, seed + 2)
                  if mm_chains else (None, None))
        qp, lp, _ = shape(chains)
        qt, lt, _ = shape(tight)
        qm = sum(len(c) for c in mm_chains.values()) if mm_chains else None

        row = {"task": t.name, "variables": t.logical.number_of_nodes(),
               "policy": {"qubits": qp, "longest": lp, "residual": rp, "p_solve": pp},
               "pruned": {"qubits": qt, "longest": lt, "residual": rt, "p_solve": pt},
               "minorminer": ({"qubits": qm, "residual": rm, "p_solve": pm} if mm_chains else None),
               "removable_qubits": qp - qt,
               "removable_fraction": (qp - qt) / qp if qp else 0.0}
        rows.append(row)
        print("  %-28s %6d %6d %6d %7s %8.4f %8.4f %8s"
              % (t.name[:28], row["variables"], qp, qt, qm if qm else "-", rp, rt,
                 ("%.4f" % rm) if rm is not None else "-"), flush=True)
        print(json.dumps(row), flush=True)

    if rows:
        m = lambda f: sum(f(r) for r in rows) / len(rows)
        with_mm = [r for r in rows if r["minorminer"]]
        summary = {
            "summary": "policy_excess", "tasks": len(rows),
            "mean_policy_qubits": m(lambda r: r["policy"]["qubits"]),
            "mean_pruned_qubits": m(lambda r: r["pruned"]["qubits"]),
            "mean_minorminer_qubits": (sum(r["minorminer"]["qubits"] for r in with_mm) / len(with_mm)
                                       if with_mm else None),
            "mean_removable_fraction": m(lambda r: r["removable_fraction"]),
            "mean_residual_policy": m(lambda r: r["policy"]["residual"]),
            "mean_residual_pruned": m(lambda r: r["pruned"]["residual"]),
            "mean_residual_minorminer": (sum(r["minorminer"]["residual"] for r in with_mm) / len(with_mm)
                                         if with_mm else None),
            "recovered_by_pruning": m(lambda r: r["policy"]["residual"] - r["pruned"]["residual"]),
            "reading": ("a large removable fraction with a positive recovery implicates redundant "
                        "growth; little removable, or removal that does not recover, implicates "
                        "the placement that the growth was compensating for")}
        print(json.dumps(summary), flush=True)
    print("POLICY EXCESS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
