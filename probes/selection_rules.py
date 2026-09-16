"""Selection rules over K independent router draws, assessed on a fresh block.

The claim under test is the paper's: the resource count of an embedding (total qubits, longest
chain) is not a signal of its solution quality, and selecting by measurement on the objective
is. Every rule sees the same K draws of the same router on the same instance, so the only
difference between rules is how the draw is chosen:

  single     the first valid draw (what a caller of the router gets)
  random     a uniformly random valid draw
  qubits     the valid draw with the fewest total qubits (resource-first)
  chain      the valid draw with the shortest longest chain, ties by fewest qubits
  measured   the valid draw with the best score on a selection block of `--reads` reads
  oracle     the valid draw with the best score on the assessment block (reference only:
             it reads the answer, so it bounds what any rule could reach on this block)

Every valid draw is assessed once on a fresh block of `--assess-reads` reads, so all rules are
compared on the same per-draw assessments; failed draws are counted, never dropped. Within
each instance the rank correlation between resource and assessed quality is reported too.
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

RULES = ("single", "random", "qubits", "chain", "measured", "oracle")


def total_qubits(chains):
    return sum(len(c) for c in chains.values())


def longest_chain(chains):
    return max((len(c) for c in chains.values()), default=0)


def pick_fewest_qubits(draws, valid):
    """Index of the valid draw with the fewest qubits; ties by the earliest draw."""
    return min(valid, key=lambda k: (total_qubits(draws[k]), k))


def pick_shortest_chain(draws, valid):
    """Index of the valid draw with the shortest longest chain; ties by fewest qubits, then order."""
    return min(valid, key=lambda k: (longest_chain(draws[k]), total_qubits(draws[k]), k))


def pick_by_score(scores, valid):
    """Index of the valid draw with the highest finite score; None when nothing was measurable."""
    scored = [k for k in valid if scores.get(k) is not None and np.isfinite(scores[k])]
    if not scored:
        return None
    return max(scored, key=lambda k: (scores[k], -k))


def spearman(x, y):
    """Spearman rank correlation with average ranks; None when either side is constant."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None

    def ranks(v):
        order = np.argsort(v, kind="stable"); r = np.empty(len(v)); r[order] = np.arange(len(v))
        # average the ranks of ties
        for value in np.unique(v):
            mask = v == value
            r[mask] = r[mask].mean()
        return r

    rx, ry = ranks(x), ranks(y)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx * rx).sum() * (ry * ry).sum()))


def paired_boot(values, seed=0, draws=2000):
    """Mean and a 95 percent bootstrap interval of paired differences."""
    d = np.asarray(values, dtype=float)
    if len(d) == 0:
        return None
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(draws)]
    return float(d.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def choose(draws, valid, select_scores, assess_scores, rng):
    """One pick per rule over the same draws; None where a rule has nothing to pick."""
    if not valid:
        return {r: None for r in RULES}
    return {
        "single": valid[0],
        "random": int(rng.choice(valid)),
        "qubits": pick_fewest_qubits(draws, valid),
        "chain": pick_shortest_chain(draws, valid),
        "measured": pick_by_score(select_scores, valid),
        "oracle": pick_by_score(assess_scores, valid),
    }


def main() -> int:
    from isingfold.rl.data.generate import load_instances
    from isingfold.rl.env import fixed_strength_selector
    from isingfold.rl.evaluate import first_commit_controller, run_controller
    from _context import host_context
    from _initializers import minorminer_initializer
    from train_contact_policy import block_seed, split_representatives

    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--qubit-cap", type=int, default=248)
    ap.add_argument("--draws", type=int, default=8, help="independent router draws per instance")
    ap.add_argument("--tries", type=int, default=20, help="router tries inside one draw")
    ap.add_argument("--reads", type=int, default=256, help="selection block per draw")
    ap.add_argument("--assess-reads", type=int, default=512, help="assessment block per draw")
    ap.add_argument("--train-lineages", type=int, default=60,
                    help="skipped, so the validation instances match the contact runs' split")
    ap.add_argument("--eval-lineages", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--objective", default="utility", choices=("utility", "residual"))
    a = ap.parse_args()
    for key in ("qubit_cap", "draws", "tries", "reads", "assess_reads", "eval_lineages"):
        if getattr(a, key) <= 0:
            ap.error("--" + key.replace("_", "-") + " must be positive")

    tasks = load_instances(a.corpus)
    _, eval_tasks = split_representatives(tasks, a.train_lineages, a.eval_lineages, a.seed)
    ctx = host_context(a.qubit_cap)
    mm = minorminer_initializer(a.tries)
    print(json.dumps({"corpus": a.corpus, "validation": len(eval_tasks), "draws": a.draws,
                      "tries": a.tries, "selection_reads_per_draw": a.reads,
                      "assessment_reads_per_draw": a.assess_reads, "objective": a.objective,
                      "rules": list(RULES), "beta_range": list(ctx.beta_range),
                      "validation_lineages": [t.lineage for t in eval_tasks]}), flush=True)

    def measure(task, chains, seed, reads):
        def fixed(l, h, s):
            return chains
        try:
            out = run_controller([task], ctx, first_commit_controller, initializer=fixed,
                                 selector=fixed_strength_selector(), reward_reads=reads,
                                 repetitions=1, seed=seed)
        except Exception as exc:
            print(json.dumps({"measurement_failure": type(exc).__name__, "task": task.name,
                              "seed": seed, "reads": reads}), flush=True)
            return None
        o = out[0]
        if not o.returned_valid:
            return None
        if a.objective == "residual":
            return -float(o.mean_energy_residual) if o.mean_energy_residual is not None else None
        return float(o.utility) if o.utility is not None else None

    rows = []
    rng = np.random.default_rng(block_seed(a.seed, "random-rule"))
    for t in eval_tasks:
        draws = [mm(t.logical, t.host, int(block_seed(a.seed, "draw", t.name, k)) % (2 ** 31))
                 for k in range(a.draws)]
        valid = [k for k, d in enumerate(draws) if d is not None
                 and total_qubits(d) <= min(a.qubit_cap, len(t.host))]
        select_scores = {k: measure(t, draws[k], int(block_seed(a.seed, "select", t.name, k)) % (2 ** 31), a.reads)
                         for k in valid}
        assess_scores = {k: measure(t, draws[k], int(block_seed(a.seed, "assess", t.name, k)) % (2 ** 31), a.assess_reads)
                         for k in valid}
        assessed_valid = [k for k in valid if assess_scores[k] is not None]
        picks = choose(draws, assessed_valid, select_scores, assess_scores, rng)
        row = {"task": t.name, "draws": a.draws, "valid": len(valid), "assessed": len(assessed_valid),
               "qubits": [total_qubits(draws[k]) for k in assessed_valid],
               "chain": [longest_chain(draws[k]) for k in assessed_valid],
               "assess": [assess_scores[k] for k in assessed_valid],
               "select": [select_scores[k] for k in assessed_valid],
               "picks": picks,
               "value": {r: (assess_scores[k] if k is not None else None) for r, k in picks.items()}}
        row["rho_qubits"] = spearman(row["qubits"], row["assess"])
        row["rho_chain"] = spearman(row["chain"], row["assess"])
        rows.append(row)
        print(json.dumps(row), flush=True)

    usable = [r for r in rows if r["value"]["single"] is not None]
    print("SELECTION RULES over %d instances, %d with a valid draw, mean valid draws %.2f of %d"
          % (len(rows), len(usable), np.mean([r["valid"] for r in rows]) if rows else 0.0, a.draws), flush=True)
    print("  rule       mean value   minus single (95 pct bootstrap)")
    for r in RULES:
        vals = [row["value"][r] for row in usable if row["value"][r] is not None]
        diff = paired_boot([row["value"][r] - row["value"]["single"] for row in usable
                            if row["value"][r] is not None])
        if diff is None:
            print("  %-9s  (nothing to pick)" % r); continue
        print("  %-9s  %+.4f      %+.4f [%+.4f, %+.4f]" % (r, float(np.mean(vals)), *diff), flush=True)
    rq = [r["rho_qubits"] for r in rows if r["rho_qubits"] is not None]
    rc = [r["rho_chain"] for r in rows if r["rho_chain"] is not None]
    print("  within-instance Spearman of assessed value against total qubits: %s over %d; against longest chain: %s over %d"
          % ("%+.3f" % np.mean(rq) if rq else "n/a", len(rq), "%+.3f" % np.mean(rc) if rc else "n/a", len(rc)), flush=True)
    top = [row["value"]["qubits"] >= np.median(row["assess"]) for row in usable if len(row["assess"]) >= 2]
    print("  fewest-qubit draw at or above the instance median: %.2f over %d" % (np.mean(top) if top else 0.0, len(top)), flush=True)
    print("SELECTION RULES DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
