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
the winner on disjoint blocks. Cross-channel gains additionally test whether a residual ranker
improves solve probability; a residual-only ranking win cannot establish that claim. These are
development-set diagnostics on a heuristic/growth pool, not learned-policy performance evidence.
"""
import argparse, json, math, os, sys
from pathlib import Path

sys.path.insert(0, os.environ.get("ISINGFOLD_SRC", str(Path(__file__).resolve().parents[1] / "src")))
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
UNITS = {"solve probability": "probability (0 to 1)", "residual": "normalized energy residual"}


def calibration_tasks(path, cells, n_tasks, seed, split="train", allow_unregistered=False):
    """Calibration never consumes a registered test split, even with the escape flag."""
    if split not in {"train", "validation"}:
        raise ValueError("calibration split must be train or validation, never test")
    root = Path(path)
    if (root / "splits.json").exists():
        with (root / "splits.json").open() as stream:
            registered = json.load(stream)
    elif (root / "manifest.json").exists():
        with (root / "manifest.json").open() as stream:
            registered = json.load(stream).get("split") or {}
    else:
        registered = {}
    has_split = any(k in registered for k in ("train", "validation", "test"))
    if not has_split and not allow_unregistered:
        raise ValueError("calibration requires a registered split; use --allow-unregistered-corpus "
                         "only for a new exploratory corpus without a locked test split")
    train, validation = cc.build_corpus_sets(
        path, cells, n_tasks if split == "train" else 0,
        n_tasks if split == "validation" else 0, seed,
        use_manifest_split=has_split, heldout_role="validation")
    return (train if split == "train" else validation), has_split


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

    This off-policy diagnostic need not match embeddings produced by a learned constructor.
    Repeat calibration on policy-created embeddings before transferring its conclusions.
    minorminer is a comparison arm here and never a completion step of anything learned.
    """
    mm = minorminer_initializer(tries)
    out, seen = [], set()
    def add(chains):
        if set(chains) != set(task.logical):
            raise ValueError("calibration embedding must cover every logical variable")
        used = set()
        for chain in chains.values():
            if not chain or not set(chain) <= set(task.host) or not nx.is_connected(task.host.subgraph(chain)):
                raise ValueError("calibration chains must be nonempty connected host subsets")
            if used.intersection(chain):
                raise ValueError("calibration chains must be disjoint")
            used.update(chain)
        for u, v in task.logical.edges():
            if not any(task.host.has_edge(a, b) for a in chains[u] for b in chains[v]):
                raise ValueError("calibration embedding misses a logical interaction")
        fingerprint = frozenset((v, frozenset(chain)) for v, chain in chains.items())
        if fingerprint not in seen:
            seen.add(fingerprint)
            out.append(chains)
    for k in range(size * 3):
        if len(out) >= size:
            break
        found = mm(task.logical, task.host, seed + k)
        if found is None:
            continue
        chains = {v: frozenset(found[v]) for v in task.logical.nodes()}
        add(chains)
        if len(out) < size:
            add(grow(chains, task.host, int(rng.integers(2, 10)), rng))
    return out[:size]


def _values(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2 or not np.isfinite(values).all():
        raise ValueError("values must be a finite candidate-by-block matrix with at least 2 of each")
    return values


def stats(values, repeats):
    """Between-candidate variance with the measurement variance removed, and the ratio."""
    values = _values(values)
    if isinstance(repeats, bool) or not isinstance(repeats, (int, np.integer)) or repeats != values.shape[1]:
        raise ValueError("repeats must equal the number of measured blocks")
    within = values.var(axis=1, ddof=1).mean()
    between_obs = values.mean(axis=1).var(ddof=1)
    between_true = max(0.0, between_obs - within / repeats)
    ratio = float(between_true / within) if within > 0 else None
    if ratio is not None and not math.isfinite(ratio):
        ratio = None
    return math.sqrt(within), math.sqrt(between_true), ratio


def ranking_assessment(rank_values, assess_values, rank_cols, assess_cols,
                       rank_high, assess_high, seed=0):
    """Rank on one channel; score a selected candidate on disjoint blocks of another.

    Gains are in assessment-channel units, with positive always meaning improvement.
    A uniform seeded draw among exact ties avoids favoring pool insertion order.
    The assessment-pool best is optimistically chosen on assessment data; it is a
    descriptive ceiling, not an independently evaluated oracle baseline.
    """
    rank_values, assess_values = _values(rank_values), _values(assess_values)
    if rank_values.shape != assess_values.shape:
        raise ValueError("rank and assessment channels must describe identical candidate/block axes")
    cols = [list(rank_cols), list(assess_cols)]
    for group in cols:
        if (not group or any(isinstance(c, bool) or not isinstance(c, (int, np.integer))
                             or c < 0 or c >= rank_values.shape[1] for c in group)
                or len(set(group)) != len(group)):
            raise ValueError("ranking and assessment columns must be nonempty unique valid indices")
    if set(cols[0]).intersection(cols[1]):
        raise ValueError("ranking and assessment blocks must be disjoint")
    rank_by = rank_values[:, cols[0]].mean(axis=1)
    assessed = assess_values[:, cols[1]].mean(axis=1)
    best = rank_by.max() if rank_high else rank_by.min()
    tied = np.flatnonzero(rank_by == best)
    pick = int(np.random.default_rng(seed).choice(tied))
    gain = assessed[pick] - assessed.mean()
    return {"gain": float(gain if assess_high else -gain), "selected_index": pick,
            "tie_count": len(tied), "tie_break_seed": int(seed),
            "selected_assessed_value": float(assessed[pick]), "pool_mean": float(assessed.mean()),
            "assessment_pool_best_gain": float(assessed.max() - assessed.mean() if assess_high
                                                else assessed.mean() - assessed.min())}


def ranking_gain(values, rank_cols, assess_cols, better_is_high, seed=0):
    """Backward-compatible same-channel pair; see ranking_assessment for full receipts."""
    row = ranking_assessment(values, values, rank_cols, assess_cols, better_is_high, better_is_high, seed)
    return row["gain"], row["assessment_pool_best_gain"]


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
    ap.add_argument("--split", choices=("train", "validation"), default="train")
    ap.add_argument("--allow-unregistered-corpus", action="store_true",
                    help="explicitly exploratory corpus with no registered split; cannot override a test split")
    a = ap.parse_args()
    if min(a.n_tasks, a.reads, a.mm_tries) < 1 or a.embeddings < 2 or a.repeats < 5:
        ap.error("positive tasks/reads/tries, at least 2 embeddings and 5 repeats are required")
    if a.strength_index < 0 or a.seed < 0:
        ap.error("strength-index and seed must be nonnegative")
    if a.seed + a.n_tasks * a.embeddings * a.repeats >= 2 ** 31:
        ap.error("measurement seed range must fit in [0, 2**31)")

    cells = [c for c in a.cells.split(",") if c]
    tasks, manifest = calibration_tasks(a.corpus, cells, a.n_tasks, a.seed, a.split,
                                        a.allow_unregistered_corpus)
    print(json.dumps({"probe": "reward_channel", "corpus": a.corpus, "cells": cells,
                      "tasks": len(tasks), "embeddings": a.embeddings, "repeats": a.repeats,
                      "reads": a.reads, "strength_index": a.strength_index,
                      "split": a.split, "manifest_split": manifest, "locked_test_used": False,
                      "scope": "development calibration on minorminer/growth pool; no learned-policy claim",
                      "units": UNITS, "tie_break": "seeded uniform exact ties",
                      "independent_experimental_unit": "instance lineage"}, allow_nan=False), flush=True)
    print("  %-30s %5s %-18s %9s %9s %7s %9s %9s"
          % ("task", "pool", "channel", "within", "between", "ratio", "gain 1 blk", "gain 4 blk"),
          flush=True)

    rows = []
    for k, t in enumerate(tasks):
        ctx = host_context(t.host.number_of_nodes())
        rng = np.random.default_rng(a.seed + k)
        chains_pool = pool(t, a.embeddings, a.mm_tries, rng, a.seed + 1000 * k)
        if len(chains_pool) < 2:
            print(json.dumps({"task": t.name, "skipped": "pool of %d" % len(chains_pool)}), flush=True)
            continue
        rates = np.zeros((len(chains_pool), a.repeats))
        resid = np.zeros((len(chains_pool), a.repeats))
        for i, chains in enumerate(chains_pool):
            for r in range(a.repeats):
                b = block(t, chains, ctx, a.strength_index, a.reads,
                          a.seed + (k * a.embeddings + i) * a.repeats + r)
                rates[i, r] = b.rate
                resid[i, r] = b.mean_residual
                if (b.reads != a.reads or b.strength_index != a.strength_index
                        or not 0 <= rates[i, r] <= 1 or not np.isfinite(resid[i, r]) or resid[i, r] < 0):
                    raise ValueError("calibration sampler returned an invalid read/strength/quality receipt")
        row = {"task": t.name, "pool": len(chains_pool),
               "qubits": [sum(len(c) for c in ch.values()) for ch in chains_pool],
               "measurement_reads": len(chains_pool) * a.repeats * a.reads,
               "independent_blocks": {"solve probability": rates.tolist(), "residual": resid.tolist()}}
        for name, values, high in (("solve probability", rates, True), ("residual", resid, False)):
            w, b, ratio = stats(values, a.repeats)
            g1, ceil1 = ranking_gain(values, [0], list(range(1, a.repeats)), high, a.seed + k)
            g4, _ = ranking_gain(values, [0, 1, 2, 3], list(range(4, a.repeats)), high, a.seed + k)
            row[name] = {"within_std": w, "between_std_true": b, "ratio": ratio,
                         "gain_one_block": float(g1), "gain_four_blocks": float(g4),
                         "assessment_pool_best_gain": float(ceil1), "unit": UNITS[name]}
            print("  %-30s %5d %-18s %9.4f %9.4f %7s %9.4f %9.4f"
                  % (t.name[:30], len(chains_pool), name, w, b,
                     "n/a" if ratio is None else "%.2f" % ratio, g1, g4), flush=True)
        row["cross_channel"] = {}
        for name, rank_values, assess_values, rank_high, assess_high, unit in (
            ("residual_to_p_solve", resid, rates, False, True, UNITS["solve probability"]),
            ("p_solve_to_residual", rates, resid, True, False, UNITS["residual"]),
        ):
            row["cross_channel"][name] = {"unit": unit}
            for count, label in ((1, "one_block"), (4, "four_blocks")):
                row["cross_channel"][name][label] = ranking_assessment(
                    rank_values, assess_values, list(range(count)), list(range(count, a.repeats)),
                    rank_high, assess_high, a.seed + k)
        rows.append(row)
        print(json.dumps(row, allow_nan=False), flush=True)

    if rows:
        summary = {"summary": "reward_channel", "tasks": len(rows)}
        for name in CHANNELS:
            ratios = [r[name]["ratio"] for r in rows if r[name]["ratio"] is not None]
            summary[name] = {
                "median_ratio": float(np.median(ratios)) if ratios else None,
                "undefined_ratio_tasks": len(rows) - len(ratios),
                "median_gain_one_block": float(np.median([r[name]["gain_one_block"] for r in rows])),
                "median_gain_four_blocks": float(np.median([r[name]["gain_four_blocks"] for r in rows])),
                "mean_gain_one_block": float(np.mean([r[name]["gain_one_block"] for r in rows])),
                "unit": UNITS[name]}
        summary["cross_channel"] = {}
        for name in ("residual_to_p_solve", "p_solve_to_residual"):
            summary["cross_channel"][name] = {"unit": rows[0]["cross_channel"][name]["unit"]}
            for label in ("one_block", "four_blocks"):
                gains = [r["cross_channel"][name][label]["gain"] for r in rows]
                summary["cross_channel"][name][label] = {
                    "mean_gain": float(np.mean(gains)), "median_gain": float(np.median(gains)),
                    "positive_tasks": sum(g > 0 for g in gains), "per_task_gain": gains}
        summary["interpretation"] = "descriptive development calibration; no significance or policy-superiority claim"
        print(json.dumps(summary, allow_nan=False), flush=True)
    print("REWARD CHANNEL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
