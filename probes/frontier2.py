"""The cost-utility frontier, measured without the selection bias the first probe had.

What was wrong with `frontier.py`
---------------------------------
Every arm reported the largest of the same 128-read scores it had used to choose its winner. A
maximum over N noisy measurements climbs on noise alone, so an arm that selects over more
measurements looks better even when every candidate has identical true quality. The comparison
of objective ranking against resource ranking was the worst case: the objective arm reported a
maximum over N scores and the resource arm reported a single score, so the objective arm could
not lose. Its reported +0.16 was partly guaranteed by the procedure.

What this probe does instead
----------------------------
Selection and assessment are separated. Each arm spends its registered budget of 128-read blocks
to choose one embedding, exactly as a deployed method would. The chosen embedding is then frozen
and measured again on independent reads that took no part in the choice, and that fresh number is
what gets reported. Assessment reads are the instrument, not the method's budget, so they are not
charged; the budget columns count only the blocks an arm actually needed to make its choice.

Three further corrections
-------------------------
  * Failures are counted. An arm that does not return an embedding on a lineage has failed on
    that lineage, and both the solved-only mean and the mean with failures scored zero are
    reported, with the denominator visible.
  * Iterated improvement reports the final state on its own, not the better of the final state
    and where it started, so degradation is visible instead of being clipped away.
  * A noise control assesses one unchanged embedding repeatedly, which puts a number on how much
    of any difference this instrument can manufacture on its own.
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
import torch
from isingfold.rl.contracts import Context
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.evaluate import first_commit_controller, run_controller, torch_controller
from isingfold.rl.model import build_model

from _initializers import minorminer_initializer

SELECT_BASE = 5_000
IMPROVE_BASE = 700_000
ASSESS_BASE = 90_000_000
"""Assessment seeds live in their own range so no read can serve both selection and assessment."""


def fixed_initializer(chains):
    def _init(logical, host, seed):
        return chains
    return _init


def measure(task, ctx, chains, seed, reads):
    """One evaluator block on a fixed embedding. Returns (utility, qubits, max_chain) or None."""
    try:
        out = run_controller([task], ctx, first_commit_controller,
                             initializer=fixed_initializer(chains),
                             selector=fixed_strength_selector(), reward_reads=reads,
                             repetitions=1, seed=seed)
    except Exception:
        return None
    o = out[0]
    if not o.returned_valid or o.utility is None:
        return None
    return float(o.utility), o.qubits, o.max_chain


def improve(task, ctx, controller, chains, seed, rounds, reward_reads=128):
    """Run the policy for `rounds` episodes, each starting from the last one's returned
    embedding, through the library's own `run_controller`.

    An earlier version of this function stepped the environment by hand so that rounds could be
    chained without spending an evaluator block each time. It produced embeddings far worse than
    the same policy run through `run_controller` from the same state with the same seed: -0.11
    against +0.0015 on the same twenty-four lineages. The hand-written loop was wrong and every
    degradation number it produced has been withdrawn. Chaining through the library costs one
    block per round, and that is what the budget columns now charge.
    """
    current = chains
    for r in range(rounds):
        try:
            out = run_controller([task], ctx, controller, initializer=fixed_initializer(current),
                                 selector=fixed_strength_selector(), reward_reads=reward_reads,
                                 repetitions=1, seed=seed + 7919 * r)
        except Exception:
            return None if r == 0 else current
        o = out[0]
        if not o.returned_valid or o.returned_embedding is None:
            return None if r == 0 else current
        current = {n: frozenset(c) for n, c in o.returned_embedding.items()}
    return current


def summarise(label, assessed, n_lineages, blocks, secs, extra=""):
    solved = [v for v in assessed.values() if v is not None]
    zeros = [0.0 if v is None else v for v in assessed.values()]
    print("  %-44s solved %2d/%2d  assessed %.4f  with failures as zero %.4f  "
          "%d blocks  %.1fs%s"
          % (label, len(solved), n_lineages,
             float(np.mean(solved)) if solved else float("nan"),
             float(np.mean(zeros)) if zeros else float("nan"), blocks, secs, extra), flush=True)
    return assessed


def paired(label, arm, ref, note):
    keys = [k for k in sorted(set(arm) & set(ref))]
    a = np.array([0.0 if arm[k] is None else arm[k] for k in keys])
    b = np.array([0.0 if ref[k] is None else ref[k] for k in keys])
    if len(keys) < 5:
        print("  %-52s (too few lineages: %d)" % (label, len(keys))); return
    d = a - b
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    print("  %-52s n %3d  %+.4f [%+.4f, %+.4f]  (%s)"
          % (label, len(d), d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), note),
          flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="runs/corpus_c4x10")
    ap.add_argument("--lineages", type=int, default=60)
    ap.add_argument("--split", default="test", choices=["test", "validation", "both"],
                    help="which held-out lineages to use. 'test' alone is the locked set")
    ap.add_argument("--ladder", default="2,4,8,12,16,24,40")
    ap.add_argument("--rounds", default="1,2,4,8,16",
                    help="policy improvement rounds applied to the selected embedding")
    ap.add_argument("--draws", type=int, default=8,
                    help="minorminer draws the composed arms select from")
    ap.add_argument("--checkpoints", default="")
    ap.add_argument("--mm-tries", type=int, default=10)
    ap.add_argument("--improve-target", default="best", choices=["best", "random", "worst"],
                    help="which draw the policy is asked to improve. The trainer starts every "
                         "episode from a fresh minorminer draw, so 'random' is the distribution "
                         "the policy was trained on and 'best' is not: asking it to improve the "
                         "best of eight puts it on states it has never seen. Reporting only "
                         "'best' would blame the operator for a deployment choice")
    ap.add_argument("--noop-control", action="store_true",
                    help="replace the policy with a controller that commits the embedding it "
                         "was given. The reported delta against its own starting point must "
                         "then be zero up to read noise; anything else is a comparator bug, "
                         "which is how the external audit found the last one")
    ap.add_argument("--greedy", action="store_true",
                    help="take the policy mode instead of sampling at temperature one. The "
                         "deployment rule in the specification samples, and every measurement "
                         "in this project has sampled, but an improvement operator applied to "
                         "an embedding that is already good is exactly where the two differ")
    ap.add_argument("--qubit-cap", type=int, default=120)
    ap.add_argument("--reads", type=int, default=128)
    ap.add_argument("--assess-reads", type=int, default=512,
                    help="independent reads used only to estimate the returned embedding's "
                         "quality, four times the selection block so the estimate is tighter "
                         "than any single selection measurement")
    a = ap.parse_args()

    tasks = load_instances(a.corpus)
    split = json.loads((Path(a.corpus) / "splits.json").read_text())
    held = (set(split["test"]) if a.split == "test"
            else set(split["validation"]) if a.split == "validation"
            else set(split["validation"]) | set(split["test"]))
    dev = [t for t in tasks if t.lineage in held][: a.lineages]
    ctx = Context(qubit_cap=a.qubit_cap)
    mm = minorminer_initializer(a.mm_tries)
    ladder = [int(x) for x in a.ladder.split(",") if x]
    rounds_list = [int(x) for x in a.rounds.split(",") if x]
    ckpts = dict(kv.split("=", 1) for kv in a.checkpoints.split(",") if kv)
    print(json.dumps({"lineages": len(dev), "split": a.split, "ladder": ladder,
                      "rounds": rounds_list, "draws": a.draws, "reads": a.reads,
                      "assess_reads": a.assess_reads}), flush=True)

    def assess(task, chains, tag):
        got = measure(task, ctx, chains, ASSESS_BASE + 13 * tag, a.assess_reads)
        return None if got is None else got[0]

    # One shared pool per lineage: the same draws, selected over by every arm.
    # Every attempt is kept, including the ones that produced nothing. best-of-n means the first
    # n attempts, not the first n successes: dropping a failed draw and reaching for another one
    # spends budget the arm was not charged for, and hides that the generator ever failed.
    biggest = max(ladder + [a.draws])
    pool, started, failures = {}, time.time(), 0
    for t in dev:
        attempts = []
        for j in range(biggest):
            chains = mm(t.logical, t.host, SELECT_BASE + 97 * j)
            got = None if chains is None else measure(t, ctx, chains, SELECT_BASE + 97 * j,
                                                      a.reads)
            if got is None:
                attempts.append(None)
                failures += 1
            else:
                attempts.append({"select": got[0], "qubits": got[1], "max_chain": got[2],
                                 "j": j, "chains": chains})
        pool[t.name] = attempts
    pool_secs = time.time() - started
    per_draw = pool_secs / max(1, len(dev) * biggest)
    print("  attempted %d draws per lineage in %.1fs (%.3fs each), %d attempts returned nothing"
          % (biggest, pool_secs, per_draw, failures), flush=True)

    def successes(name, n):
        """The successful draws among the first n attempts, which is what a budget of n buys."""
        return [d for d in pool.get(name, [])[:n] if d is not None]

    print("\n== noise control: the same unchanged embedding, assessed five times")
    spreads = []
    for t in dev[:12]:
        first = successes(t.name, 1)
        if not first:
            continue
        ch = first[0]["chains"]
        vals = [measure(t, ctx, ch, ASSESS_BASE + 7717 * k, a.assess_reads) for k in range(5)]
        vals = [v[0] for v in vals if v is not None]
        if len(vals) >= 2:
            spreads.append(max(vals) - min(vals))
    if spreads:
        print("  range over five assessments of one fixed embedding: mean %.4f, max %.4f "
              "over %d lineages" % (float(np.mean(spreads)), float(max(spreads)), len(spreads)),
              flush=True)
        print("  a difference smaller than this is inside what the instrument can invent",
              flush=True)

    print("\n== minorminer, selected on %d-read blocks, assessed on %d fresh reads"
          % (a.reads, a.assess_reads))
    frontier, frontier_secs = {}, {}
    for n in ladder:
        vals = {}
        for t in dev:
            draws = successes(t.name, n)
            if not draws:
                vals[t.name] = None
                continue
            win = max(draws, key=lambda d: d["select"])
            vals[t.name] = assess(t, win["chains"], n * 1000 + win["j"])
        secs = per_draw * len(dev) * n
        frontier[n] = vals
        frontier_secs[n] = secs
        summarise("minorminer best-of-%d" % n, vals, len(dev), n, secs)

    print("\n== the same draws, selected by qubit count, assessed the same way")
    resource = {}
    for n in ladder:
        vals = {}
        for t in dev:
            draws = successes(t.name, n)
            if not draws:
                vals[t.name] = None
                continue
            win = min(draws, key=lambda d: (d["qubits"], d["max_chain"]))
            vals[t.name] = assess(t, win["chains"], 500 + n * 1000 + win["j"])
        resource[n] = vals
        summarise("resource-picked best-of-%d" % n, vals, len(dev), 0, 0.0,
                  extra="   (no selection blocks needed)")

    print("\n== objective selection against resource selection, identical draws, fresh assessment")
    for n in ladder:
        paired("best-of-%d: objective minus resource" % n, frontier[n], resource[n],
               "objective spends %d blocks to choose, resource spends none" % n)

    for fam, path in ckpts.items():
        if not Path(path).exists():
            print("\n== %s: no checkpoint at %s" % (fam, path)); continue
        model = build_model(fam, improvement_mode=True)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        controller = (first_commit_controller if a.noop_control
                      else torch_controller(model, None, greedy=a.greedy))
        if a.noop_control:
            print("  NO-OP CONTROL: the policy is replaced by commit-what-you-were-given",
                  flush=True)
        print("\n== %s applied to the %s of %d minorminer draws (%s), %s"
              % (fam, a.improve_target, a.draws, path,
                 "mode" if a.greedy else "sampled at temperature one"))
        for rounds in rounds_list:
            started = time.time()
            search_secs = 0.0
            final_only, best_of_both, initial_assessed = {}, {}, {}
            for t in dev:
                draws = successes(t.name, a.draws)
                if not draws:
                    final_only[t.name] = best_of_both[t.name] = None
                    initial_assessed[t.name] = None
                    continue
                if a.improve_target == "best":
                    win = max(draws, key=lambda d: d["select"])
                elif a.improve_target == "worst":
                    win = min(draws, key=lambda d: d["select"])
                else:
                    win = draws[0]
                # The starting point has to be assessed the same way the result is, or the
                # difference is between two different instruments rather than two states.
                base = (frontier[a.draws].get(t.name) if a.improve_target == "best"
                        else assess(t, win["chains"], 31000 + win["j"]))
                move_started = time.time()
                out = improve(t, ctx, controller, win["chains"],
                              IMPROVE_BASE + 97 * win["j"], rounds, reward_reads=a.reads)
                search_secs += time.time() - move_started
                initial_assessed[t.name] = base
                if out is None:
                    final_only[t.name] = None
                    best_of_both[t.name] = base
                    continue
                got = assess(t, out, 7000 + rounds * 1000 + win["j"])
                final_only[t.name] = got
                # A deployed method would keep the better of the two, and pay one more block to
                # know which. That block is charged; the assessment that reports it is not.
                if got is None or base is None:
                    best_of_both[t.name] = base if got is None else got
                else:
                    pick = measure(t, ctx, out, SELECT_BASE + 555 + win["j"], a.reads)
                    keep_new = pick is not None and pick[0] >= win["select"]
                    best_of_both[t.name] = got if keep_new else base
            # The budget is what the arm spends to choose what it returns: the draws it was
            # given plus the policy's own episodes. The assessment that estimates the quality of
            # that choice is the instrument, not the method, and is excluded from both sides.
            secs = search_secs + per_draw * len(dev) * a.draws
            wall_with_assessment = time.time() - started + per_draw * len(dev) * a.draws
            summarise("%s: %d rounds, final state only" % (fam, rounds), final_only, len(dev),
                      a.draws + rounds, secs,
                      extra="   (assessment excluded; with it %.1fs)" % wall_with_assessment)
            summarise("%s: %d rounds, keep the better of the two" % (fam, rounds), best_of_both,
                      len(dev), a.draws + rounds + 1, secs)
            # Against the embedding actually handed to the policy, which is only the
            # best-of-draws winner when --improve-target is "best". Comparing a policy that was
            # given a random draw against the best-of-eight assessment made a no-op look like a
            # loss of a tenth of a unit, and the label on the line claimed the two were the same
            # embedding. External audit, section 8.
            paired("%d rounds, final state, against its own starting point" % rounds,
                   final_only, initial_assessed,
                   "the %s draw, assessed before and after" % a.improve_target)
            spent_blocks = a.draws + rounds + 1
            m = min(ladder, key=lambda q: abs(q - spent_blocks))
            paired("%d rounds, keep better, against minorminer best-of-%d" % (rounds, m),
                   best_of_both, frontier[m],
                   "%d blocks vs %d blocks" % (spent_blocks, m))

    print("\nFRONTIER2 DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
