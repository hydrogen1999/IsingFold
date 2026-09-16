"""Does adaptive allocation of reads pick as well as uniform allocation with fewer reads?

Measured selection is the one result here that stands: the best of K candidates by measured
solve probability, re-assessed on fresh reads, beats a random pick by +0.11. The measurement is
spent uniformly, K blocks of the same size. This asks whether spending it adaptively, as a
best-arm problem, reaches the same selected quality with fewer reads.

Arms, all on the same frozen candidate pools from a provenance-checked label cache, all
assessed on one independent 512-read block per chosen candidate:

    uniform-R        every candidate gets R reads, pick the best estimate            (K*R reads)
    halving-R        successive halving from K candidates with a total of K*R/2 reads,
                     rounds of equal spend, the worse half dropped each round
    ucb-R            round by round, the candidate with the highest upper confidence
                     bound gets the next block, total K*R/2 reads
    random           a uniform random candidate, no reads

Every read is a real evaluator call, never a simulation from the cached estimate. Seeds for
selection blocks and assessment blocks are disjoint.
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

SELECT_BASE, ASSESS_BASE = 61_000_000, 62_000_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--lineages", type=int, default=32)
    ap.add_argument("--reads", type=int, default=256, help="R, the uniform per-candidate reads")
    ap.add_argument("--block", type=int, default=32, help="smallest read block for adaptive arms")
    ap.add_argument("--assess-reads", type=int, default=512)
    ap.add_argument("--qubit-cap", type=int, default=248)
    ap.add_argument("--split", default="eval", choices=("eval", "train"))
    ap.add_argument("--prior", default="",
                    help="a trained successor scorer (train_successor.py checkpoint); adds the "
                         "arms prior (its argmax, no reads) and halving+prior (its bottom half "
                         "dropped unread, halving on the rest at the same total reads)")
    ap.add_argument("--width", type=int, default=64)
    a = ap.parse_args()
    with open(a.cache, "rb") as fh:
        blob = pickle.load(fh)
    ctx = host_context(a.qubit_cap)
    check_provenance(blob, blob["key"][0], ctx)
    states = blob[a.split]
    roots = sorted({st["lineage"] for st in states})[: a.lineages]
    states = [st for st in states if st["lineage"] in set(roots)]
    print(json.dumps({"cache": a.cache, "split": a.split, "lineages": len(roots),
                      "states": len(states), "reads": a.reads, "block": a.block,
                      "beta_range": list(ctx.beta_range), "prior": a.prior or None}), flush=True)
    prior_scores = {}
    if a.prior:
        import torch
        from successor_scorer import SuccessorScorer
        from train_successor import build
        device = torch.device("cpu")
        model = SuccessorScorer(width=a.width, node_dim=4, chain_dim=4)
        model.load_state_dict(torch.load(a.prior, map_location="cpu"))
        model.eval()
        built = build(states, ctx, device, "prior states")
        with torch.no_grad():
            for st in built:
                prior_scores[st["state_id"]] = [float(model(r["graph"])) for r in st["rows"]]

    calls = {"n": 0}

    def hits(task, chains, seed, reads):
        """Hits and reads from one evaluator block, a real sampler call."""
        calls["n"] += 1
        def fixed(l, h, s):
            return {v: frozenset(c) for v, c in chains.items()}
        try:
            out = run_controller([task], ctx, first_commit_controller, initializer=fixed,
                                 selector=fixed_strength_selector(), reward_reads=reads,
                                 repetitions=1, seed=seed)
        except Exception:
            return None
        o = out[0]
        if not o.returned_valid or o.utility is None:
            return None
        h = o.evaluator_hits if o.evaluator_hits is not None else round(float(o.utility) * reads)
        return int(h), int(o.evaluator_reads or reads)

    def assess(task, chains, k, j):
        r = hits(task, chains, ASSESS_BASE + 1000 * k + 7 * j, a.assess_reads)
        return None if r is None else r[0] / r[1]

    per_state = []
    started = time.time()
    for k, st in enumerate(states):
        rows = st["rows"]
        K = len(rows)
        if K < 4:
            continue
        task = st["task"]
        spent = {}
        # uniform: R reads each
        est = []
        for j, r in enumerate(rows):
            h = hits(task, r["succ"], SELECT_BASE + 1000 * k + 7 * j, a.reads)
            est.append(-1.0 if h is None else h[0] / h[1])
        pick_uniform = int(np.argmax(est)); spent["uniform"] = K * a.reads
        # successive halving with a budget of K*R/2 total reads
        budget = K * a.reads // 2
        alive = list(range(K)); acc = {j: [0, 0] for j in alive}
        rounds = int(np.ceil(np.log2(K)))
        per_round = budget // rounds
        seed_ctr = 10_000
        for _ in range(rounds):
            if len(alive) <= 1:
                break
            each = max(a.block, per_round // len(alive))
            for j in alive:
                seed_ctr += 1
                h = hits(task, rows[j]["succ"], SELECT_BASE + 1000 * k + seed_ctr, each)
                if h is not None:
                    acc[j][0] += h[0]; acc[j][1] += h[1]
            alive.sort(key=lambda j: -(acc[j][0] / max(1, acc[j][1])))
            alive = alive[: max(1, len(alive) // 2)]
        pick_halving = alive[0]; spent["halving"] = sum(v[1] for v in acc.values())
        # UCB with the same total budget, blocks of `block` reads
        acc = {j: [0, 0] for j in range(K)}; used = 0; seed_ctr = 20_000
        for j in range(K):
            seed_ctr += 1
            h = hits(task, rows[j]["succ"], SELECT_BASE + 1000 * k + seed_ctr, a.block)
            if h is not None:
                acc[j][0] += h[0]; acc[j][1] += h[1]; used += h[1]
        while used + a.block <= budget:
            t = sum(v[1] for v in acc.values())
            ucb = [acc[j][0] / max(1, acc[j][1]) + np.sqrt(2 * np.log(max(2, t)) / max(1, acc[j][1]))
                   for j in range(K)]
            j = int(np.argmax(ucb)); seed_ctr += 1
            h = hits(task, rows[j]["succ"], SELECT_BASE + 1000 * k + seed_ctr, a.block)
            if h is None:
                break
            acc[j][0] += h[0]; acc[j][1] += h[1]; used += h[1]
        pick_ucb = int(np.argmax([acc[j][0] / max(1, acc[j][1]) for j in range(K)]))
        spent["ucb"] = used
        rng = np.random.default_rng(k)
        pick_random = int(rng.integers(0, K))
        picks = {"uniform": pick_uniform, "halving": pick_halving, "ucb": pick_ucb,
                 "random": pick_random}
        if prior_scores.get(st.get("state_id")):
            pr = np.asarray(prior_scores[st["state_id"]])
            if len(pr) == K:
                picks["prior"] = int(np.argmax(pr)); spent["prior"] = 0
                # halving from the prior's top half, same total reads as plain halving
                alive = list(np.argsort(-pr)[: max(2, K // 2)])
                acc = {j: [0, 0] for j in alive}
                rounds = int(np.ceil(np.log2(len(alive))))
                per_round = budget // max(1, rounds)
                seed_ctr = 30_000
                for _ in range(rounds):
                    if len(alive) <= 1:
                        break
                    each = max(a.block, per_round // len(alive))
                    for j in alive:
                        seed_ctr += 1
                        h = hits(task, rows[j]["succ"], SELECT_BASE + 1000 * k + seed_ctr, each)
                        if h is not None:
                            acc[j][0] += h[0]; acc[j][1] += h[1]
                    alive.sort(key=lambda j: -(acc[j][0] / max(1, acc[j][1])))
                    alive = alive[: max(1, len(alive) // 2)]
                picks["halving+prior"] = int(alive[0]); spent["halving+prior"] = sum(v[1] for v in acc.values())
        # one assessment block per distinct chosen candidate, shared across arms that agree
        assessed = {}
        for name, j in picks.items():
            if j not in assessed:
                assessed[j] = assess(task, rows[j]["succ"], k, j)
        if any(assessed[j] is None for j in picks.values()):
            continue
        per_state.append({"lineage": st["lineage"], "spent": spent,
                          **{name: assessed[j] for name, j in picks.items()}})
        if (k + 1) % 10 == 0:
            print("  %d/%d states, %d evaluator calls, %.0fs"
                  % (k + 1, len(states), calls["n"], time.time() - started), flush=True)

    def boot(fn):
        by = defaultdict(list)
        for s in per_state:
            by[s["lineage"]].append(fn(s))
        means = np.array([np.mean(v) for v in by.values()])
        rng = np.random.default_rng(0)
        bs = [means[rng.integers(0, len(means), len(means))].mean() for _ in range(4000)]
        return means.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    print("\n  %d states in %d lineages, assessed on %d independent reads"
          % (len(per_state), len({s['lineage'] for s in per_state}), a.assess_reads))
    print("  %-14s %10s %10s  %s" % ("arm", "reads", "quality", "differences, 95% over lineages"))
    arms = ["uniform", "halving", "ucb", "random"] + (["prior", "halving+prior"] if a.prior else [])
    for name in arms:
        rows_ok = [s for s in per_state if name in s]
        if not rows_ok:
            continue
        reads = np.mean([s["spent"].get(name, 0) for s in rows_ok])
        q = float(np.mean([s[name] for s in rows_ok]))
        d, lo, hi = boot(lambda s: s[name] - s["uniform"]) if len(rows_ok) == len(per_state) else (float("nan"),) * 3
        dh = float(np.mean([s[name] - s["halving"] for s in rows_ok]))
        print("  %-14s %10.0f %10.4f  vs uniform %+.4f [%+.4f, %+.4f]  vs halving %+.4f"
              % (name, reads, q, d, lo, hi, dh))
    print("\nADAPTIVE ALLOCATION DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
