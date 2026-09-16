"""Does a larger embedding of the same problem solve it better?

Every earlier attempt at this compared the candidates a state offers, which sit about five qubits
apart and are local perturbations of one embedding. That is not a budget sweep, and the claim
built on it, that the resource-quality premise fails, was withdrawn.

Each mode/exponent independently grows the same minorminer start. The logical problem stays
fixed, but both qubit count and chain/contact geometry can change. Each candidate contains the
start; candidates from different rows need not contain each other or have monotone costs.

When several growth draws compete, select on one sampling block and report quality on a fresh
block. The start receives the same number of assessment reads. This estimates the quality of a
growth-and-selection procedure, not an isolated causal effect of resource count, and is not a
matched-cost comparison against a baseline with its own search budget. It tests how extra qubits
are spent rather than optimizing minimum qubit use.
"""
import argparse, json, os, sys, time
from itertools import count
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.evaluate import first_commit_controller, run_controller

from _context import host_context
from _initializers import minorminer_initializer
from powerlaw_embeddings import grown_embedding, realises

MEASURE_BASE = 20_000_000


def select_and_assess(candidates, measure, *, selection_reads, assessment_reads):
    """Select by pilot quality, then assess the winner on fresh reads.

    ``measure(chains, reads)`` must allocate a distinct sampling seed/block on every call.
    A single candidate needs no pilot selection. Assessment outcomes never choose a candidate.
    """
    if not candidates:
        return None
    selected = candidates[0]
    if len(candidates) > 1:
        best_quality = None
        selected = None
        for candidate in candidates:
            quality = measure(candidate, selection_reads)
            if quality is not None and (best_quality is None or quality > best_quality):
                selected, best_quality = candidate, quality
        if selected is None:
            return None
    quality = measure(selected, assessment_reads)
    return None if quality is None else (selected, quality)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--lineages", type=int, default=80)
    ap.add_argument("--alphas", default="4.0,3.0,2.2,1.6,1.2",
                    help="steep to shallow; a steep exponent keeps chains short")
    ap.add_argument("--modes", default="length,redundant,contact",
                    help="growth preferences: free space, internal attachments, or contacts "
                         "with logically adjacent chains; quality must be measured for each")
    ap.add_argument("--lmin", type=int, default=1)
    ap.add_argument("--lmax", type=int, default=8)
    ap.add_argument("--reads", type=int, default=512, help="independent assessment reads per arm")
    ap.add_argument("--selection-reads", type=int, default=128,
                    help="pilot reads per growth draw when repeats compete; separate from assessment")
    ap.add_argument("--repeats", type=int, default=2,
                    help="growth draws per exponent, so one unlucky draw is not the arm")
    ap.add_argument("--qubit-cap", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if min(a.reads, a.selection_reads, a.repeats) < 1:
        ap.error("reads, selection-reads and repeats must be positive")

    tasks = load_instances(a.corpus)[: a.lineages]
    ctx = host_context(a.qubit_cap)
    mm = minorminer_initializer(20)
    alphas = [float(x) for x in a.alphas.split(",")]
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    rng = np.random.default_rng(a.seed)
    measure_seeds = count(MEASURE_BASE)
    print(json.dumps({"corpus": a.corpus, "lineages": len(tasks), "alphas": alphas,
                      "modes": modes, "lmin": a.lmin, "lmax": a.lmax, "reads": a.reads,
                      "repeats": a.repeats, "selection_reads": a.selection_reads,
                      "assessment_reads": a.reads, "seed": a.seed,
                      "measurement_seed_start": MEASURE_BASE, "beta_range": ctx.beta_range,
                      "qubit_cap": a.qubit_cap,
                      "protocol": "pilot_selection_then_independent_assessment_v2"}), flush=True)

    def measure(task, chains, reads):
        def fixed(l, h, s):
            return chains
        try:
            out = run_controller([task], ctx, first_commit_controller, initializer=fixed,
                                 selector=fixed_strength_selector(), reward_reads=reads,
                                 repetitions=1, seed=next(measure_seeds))
        except Exception:
            return None
        o = out[0]
        return float(o.utility) if o.returned_valid and o.utility is not None else None

    rows, invalid, over_cap, short = [], 0, 0, 0
    started = time.time()
    for k, task in enumerate(tasks):
        base = mm(task.logical, task.host, 5000 + k)
        if base is None:
            continue
        base_q = sum(len(c) for c in base.values())
        base_u = measure(task, base, a.reads)
        if base_u is None:
            continue
        per_alpha = {}
        for mode, alpha in ((m, al) for m in modes for al in alphas):
            candidates = []
            for _ in range(a.repeats):
                grown, missing = grown_embedding(task.host, base, alpha, a.lmin, a.lmax, rng,
                                                 mode=mode, logical=task.logical)
                q = sum(len(c) for c in grown.values())
                if q > a.qubit_cap:
                    over_cap += 1
                    continue
                if not realises(grown, task.logical, task.host):
                    invalid += 1
                    continue
                short += missing
                candidates.append(grown)
            assessed = select_and_assess(
                candidates, lambda chains, reads: measure(task, chains, reads),
                selection_reads=a.selection_reads, assessment_reads=a.reads)
            if assessed is not None:
                grown, u = assessed
                lengths = [len(c) for c in grown.values()]
                per_alpha[(mode, alpha)] = (sum(lengths), u, float(np.mean(lengths)),
                                           int(max(lengths)))
        if per_alpha:
            rows.append({"base_q": base_q, "base_u": base_u, "per_alpha": per_alpha,
                         "base_mean_chain": float(np.mean([len(c) for c in base.values()])),
                         "base_max_chain": max(len(c) for c in base.values())})

    print("  instances swept %d, invalid growths %d, over cap %d, qubits short of target %d, %.0fs"
          % (len(rows), invalid, over_cap, short, time.time() - started), flush=True)
    if not rows:
        print("  nothing to report"); return 1

    print("\n  %-11s %-7s %-9s %-11s %-11s %-9s %s"
          % ("mode", "alpha", "qubits", "mean chain", "max chain", "quality",
             "vs the minorminer start (paired assessment)"))
    base_q = float(np.mean([r["base_q"] for r in rows]))
    base_u = float(np.mean([r["base_u"] for r in rows]))
    print("  %-11s %-7s %-9.1f %-11.2f %-11.2f %-9.4f %s"
          % ("start", "-", base_q, np.mean([r["base_mean_chain"] for r in rows]),
             np.mean([r["base_max_chain"] for r in rows]), base_u, "-"))
    for key in [(m, al) for m in modes for al in alphas]:
        have = [r for r in rows if key in r["per_alpha"]]
        if len(have) < 5:
            continue
        q = np.array([r["per_alpha"][key][0] for r in have], dtype=float)
        u = np.array([r["per_alpha"][key][1] for r in have], dtype=float)
        mc = np.array([r["per_alpha"][key][2] for r in have], dtype=float)
        mx = np.array([r["per_alpha"][key][3] for r in have], dtype=float)
        d = u - np.array([r["base_u"] for r in have], dtype=float)
        rng2 = np.random.default_rng(0)
        bs = [d[rng2.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        print("  %-11s %-7.1f %-9.1f %-11.2f %-11.2f %-9.4f %+.4f [%+.4f, %+.4f] over %d"
              % (key[0], key[1], q.mean(), mc.mean(), mx.mean(), u.mean(), d.mean(),
                 np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(have)))

    print("\n  Every candidate grows the same start independently; rows are not nested.")
    print("  The logical problem is fixed, but resource count AND chain/contact geometry change.")
    print("  Quality and paired differences use fresh assessment reads, not winning pilot scores.")
    print("  Growth arms spend extra selection reads; this is not a matched-total-cost comparison.")
    print("  Results describe these growth rules and do not establish that all extra-qubit spends")
    print("  help or hurt, nor isolate resource count from the geometry of the added qubits.")
    print("\nBUDGET SWEEP DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
