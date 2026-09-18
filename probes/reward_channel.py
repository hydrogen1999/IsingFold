"""Which measured channel is a usable training reward at this cell, and at what read budget.

Training rewards a constructed embedding with a measurement, and a measurement of a few hundred
reads is noisy. A channel is only usable as supervision when the variation it shows between
candidate embeddings of one instance is larger than the variation it shows between repeated
measurements of one candidate. That is a within-instance question and it has to be answered at
the cell training will actually run on, because the answer moves with instance size: solve
probability was the stronger discriminator where it is large and separates nothing where it is
small.

For each instance this builds a pool of valid embeddings, measures each one with several
independent blocks, and reports for both channels the between-candidate variation with the
measurement variance removed, the measurement variation, their ratio, and the thing that actually
matters: the independently assessed gain from ranking the pool with one block and then scoring
the winner on disjoint blocks. A channel that cannot rank is not a reward, whatever its ratio.
"""
import argparse, json, math, os, sys
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import networkx as nx
import numpy as np
from isingfold.rl.evaluator import sample_program
from isingfold.rl.program import compile_program, strength_registry

import constructor_curriculum as cc
from _context import REGISTERED_BETA_RANGE, host_context
from _initializers import minorminer_initializer

CHANNELS = ("solve probability", "residual")


def block(task, chains, ctx, strength_index, reads, seed):
    """One independent read block of a fixed embedding at a fixed registered strength."""
    strengths = strength_registry(task.problem, ctx.strength_ratios, ctx.epsilon_strength)
    program = compile_program(chains, task.host, task.problem,
                              strengths[strength_index], strength_index)
    return sample_program(program, chains, task.problem, task.ground_energy,
                          num_reads=reads, seed=int(seed), num_sweeps=200,
                          beta_range=REGISTERED_BETA_RANGE)


def grow(chains, host, extra, rng):
    """Absorb free qubits into random chains, the record's growth control."""
    used = {q for c in chains.values() for q in c}
    out = {v: set(c) for v, c in chains.items()}
    order = list(out)
    added = 0
    for _ in range(40 * max(1, extra)):
        if added >= extra:
            break
        v = order[rng.integers(0, len(order))]
        frontier = [t for q in out[v] for t in host.neighbors(q) if t not in used]
        if not frontier:
            continue
        q = frontier[rng.integers(0, len(frontier))]
        out[v].add(q)
        used.add(q)
        added += 1
    return {v: frozenset(c) for v, c in out.items()}


def pool(task, size, tries, rng, seed):
    """Valid embeddings of one instance: router draws, and growth variants of them.

    This is the pool a deployment actually chooses among, so the noise a selector faces on it is
    the noise a reward would face. minorminer is a comparison arm here and never a completion
    step of anything learned.
    """
    mm = minorminer_initializer(tries)
    out = []
    for k in range(size * 3):
        if len(out) >= size:
            break
        found = mm(task.logical, task.host, seed + k)
        if found is None:
            continue
        chains = {v: frozenset(found[v]) for v in task.logical.nodes()}
        out.append(chains)
        if len(out) < size:
            out.append(grow(chains, task.host, int(rng.integers(2, 10)), rng))
    return out[:size]


def stats(values, repeats):
    """Between-candidate variance with the measurement variance removed, and the ratio."""
    within = values.var(axis=1, ddof=1).mean()
    between_obs = values.mean(axis=1).var(ddof=1)
    between_true = max(0.0, between_obs - within / repeats)
    ratio = between_true / within if within > 0 else float("inf")
    return math.sqrt(within), math.sqrt(between_true), ratio


def ranking_gain(values, rank_cols, assess_cols, better_is_high):
    """Assessed value of the candidate a disjoint block picked, minus the pool mean."""
    rank_by = values[:, rank_cols].mean(axis=1)
    assessed = values[:, assess_cols].mean(axis=1)
    pick = int(np.argmax(rank_by) if better_is_high else np.argmin(rank_by))
    gain = assessed[pick] - assessed.mean()
    return (gain if better_is_high else -gain), assessed.max() - assessed.mean() if better_is_high \
        else assessed.mean() - assessed.min()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cells", default="")
    ap.add_argument("--n-tasks", type=int, default=12)
    ap.add_argument("--embeddings", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--reads", type=int, default=256)
    ap.add_argument("--strength-index", type=int, default=1)
    ap.add_argument("--mm-tries", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260924)
    a = ap.parse_args()

    cells = [c for c in a.cells.split(",") if c]
    tasks, _ = cc.build_corpus_sets(a.corpus, cells, a.n_tasks, 1, a.seed)
    print(json.dumps({"probe": "reward_channel", "corpus": a.corpus, "cells": cells,
                      "tasks": len(tasks), "embeddings": a.embeddings, "repeats": a.repeats,
                      "reads": a.reads, "strength_index": a.strength_index}), flush=True)
    print("  %-30s %5s %-18s %9s %9s %7s %9s %9s"
          % ("task", "pool", "channel", "within", "between", "ratio", "gain 1 blk", "gain 4 blk"),
          flush=True)

    rows = []
    for k, t in enumerate(tasks):
        ctx = host_context(t.host.number_of_nodes())
        rng = np.random.default_rng(a.seed + k)
        chains_pool = pool(t, a.embeddings, a.mm_tries, rng, a.seed + 1000 * k)
        if len(chains_pool) < 3:
            print(json.dumps({"task": t.name, "skipped": "pool of %d" % len(chains_pool)}), flush=True)
            continue
        rates = np.zeros((len(chains_pool), a.repeats))
        resid = np.zeros((len(chains_pool), a.repeats))
        for i, chains in enumerate(chains_pool):
            for r in range(a.repeats):
                b = block(t, chains, ctx, a.strength_index, a.reads,
                          a.seed + 100000 * k + 1000 * i + r)
                rates[i, r] = b.rate
                resid[i, r] = b.mean_residual
        row = {"task": t.name, "pool": len(chains_pool),
               "qubits": [sum(len(c) for c in ch.values()) for ch in chains_pool]}
        for name, values, high in (("solve probability", rates, True), ("residual", resid, False)):
            w, b, ratio = stats(values, a.repeats)
            g1, ceil1 = ranking_gain(values, [0], list(range(1, a.repeats)), high)
            g4, _ = ranking_gain(values, [0, 1, 2, 3], list(range(4, a.repeats)), high)
            row[name] = {"within_std": w, "between_std_true": b, "ratio": ratio,
                         "gain_one_block": float(g1), "gain_four_blocks": float(g4),
                         "oracle_gain": float(ceil1)}
            print("  %-30s %5d %-18s %9.4f %9.4f %7.2f %9.4f %9.4f"
                  % (t.name[:30], len(chains_pool), name, w, b, ratio, g1, g4), flush=True)
        rows.append(row)
        print(json.dumps(row), flush=True)

    if rows:
        summary = {"summary": "reward_channel", "tasks": len(rows)}
        for name in CHANNELS:
            summary[name] = {
                "median_ratio": float(np.median([r[name]["ratio"] for r in rows])),
                "median_gain_one_block": float(np.median([r[name]["gain_one_block"] for r in rows])),
                "median_gain_four_blocks": float(np.median([r[name]["gain_four_blocks"] for r in rows])),
                "mean_gain_one_block": float(np.mean([r[name]["gain_one_block"] for r in rows])),
                "usable": bool(np.median([r[name]["ratio"] for r in rows]) > 1.0
                               and np.mean([r[name]["gain_one_block"] for r in rows]) > 0)}
        print(json.dumps(summary), flush=True)
    print("REWARD CHANNEL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
