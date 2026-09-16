"""How reliable is the within-state ranking that the scorer is asked to learn?

Codex's specification, executed as written. From a label cache: freeze and deduplicate the
candidates of each state; measure every candidate on two independent 512-read blocks, A and B;
report the tie-aware rank correlation between A and B, top-choice agreement, and the regret of
selecting with A and assessing with B and the reverse, against the uniform reference (the mean
of all candidates' assessment values). Per-read outcomes are not retained by the evaluator
here, so the 256-read prefix analysis is done with a separate 256-read block C.

Bootstrap is over lineages. The number that matters for the surrogate is the select-A-assess-B
gain: it is what a perfect predictor of block A would earn on fresh reads, and it bounds what
any model trained on these labels can show held-out.
"""
import argparse, json, os, pickle, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.evaluate import first_commit_controller, run_controller

from _context import host_context
from _provenance import check_provenance
from train_quality import rank_corr

BLOCK_A, BLOCK_B, BLOCK_C = 71_000_000, 72_000_000, 73_000_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--lineages", type=int, default=32)
    ap.add_argument("--reads", type=int, default=512)
    ap.add_argument("--short-reads", type=int, default=256)
    ap.add_argument("--qubit-cap", type=int, default=248)
    ap.add_argument("--split", default="eval", choices=("eval", "train"))
    a = ap.parse_args()
    with open(a.cache, "rb") as fh:
        blob = pickle.load(fh)
    ctx = host_context(a.qubit_cap)
    check_provenance(blob, blob["key"][0], ctx)
    states = blob[a.split]
    roots = sorted({st["lineage"] for st in states})[: a.lineages]
    states = [st for st in states if st["lineage"] in set(roots)]
    print(json.dumps({"cache": a.cache, "split": a.split, "lineages": len(roots),
                      "states": len(states), "reads": a.reads, "beta_range": ctx.beta_range}),
          flush=True)

    def measure(task, chains, seed, reads):
        def fixed(l, h, s):
            return {v: frozenset(c) for v, c in chains.items()}
        try:
            out = run_controller([task], ctx, first_commit_controller, initializer=fixed,
                                 selector=fixed_strength_selector(), reward_reads=reads,
                                 repetitions=1, seed=seed)
        except Exception:
            return None
        o = out[0]
        return float(o.utility) if o.returned_valid and o.utility is not None else None

    per_state = []
    started = time.time()
    for k, st in enumerate(states):
        rows = st["rows"]
        A, B, C = [], [], []
        for j, r in enumerate(rows):
            seed = 1000 * k + 7 * j
            ua = measure(st["task"], r["succ"], BLOCK_A + seed, a.reads)
            ub = measure(st["task"], r["succ"], BLOCK_B + seed, a.reads)
            uc = measure(st["task"], r["succ"], BLOCK_C + seed, a.short_reads)
            if None in (ua, ub, uc):
                continue
            A.append(ua); B.append(ub); C.append(uc)
        if len(A) < 3:
            continue
        A, B, C = map(np.array, (A, B, C))
        per_state.append({
            "lineage": st["lineage"], "n": len(A),
            "corr_AB": rank_corr(A, B), "corr_CB": rank_corr(C, B),
            "top_agree_AB": float(np.argmax(A) == np.argmax(B)),
            "gain_A_on_B": float(B[np.argmax(A)] - B.mean()),
            "gain_B_on_A": float(A[np.argmax(B)] - A.mean()),
            "gain_C_on_B": float(B[np.argmax(C)] - B.mean()),
            "oracle_B_on_B": float(B.max() - B.mean()),
            "spread_B": float(B.max() - B.min()),
        })
        if (k + 1) % 10 == 0:
            print("  %d/%d states, %.0fs" % (k + 1, len(states), time.time() - started), flush=True)

    def boot(key):
        by = defaultdict(list)
        for s in per_state:
            by[s["lineage"]].append(s[key])
        means = np.array([np.mean(v) for v in by.values()])
        rng = np.random.default_rng(0)
        bs = [means[rng.integers(0, len(means), len(means))].mean() for _ in range(4000)]
        return means.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    print("\n  %d states in %d lineages" % (len(per_state), len({s['lineage'] for s in per_state})))
    print("  %-22s %9s  %s" % ("quantity", "mean", "95% bootstrap over lineages"))
    for key, label in (("corr_AB", "rank corr A vs B, 512"), ("corr_CB", "rank corr C(256) vs B"),
                       ("top_agree_AB", "top choice agrees"),
                       ("gain_A_on_B", "select A, assess B"), ("gain_B_on_A", "select B, assess A"),
                       ("gain_C_on_B", "select C(256), assess B"),
                       ("oracle_B_on_B", "oracle on B (inflated)"), ("spread_B", "spread of B")):
        m, lo, hi = boot(key)
        print("  %-22s %+9.4f  [%+.4f, %+.4f]" % (label, m, lo, hi))
    print("\n  The select-A-assess-B gain is what a perfect predictor of block A earns on fresh")
    print("  reads; nothing trained on A can show more held-out. The oracle-on-B row carries the")
    print("  winner's curse and is printed only to show its size.")
    print("\nLABEL RELIABILITY DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
