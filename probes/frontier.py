"""The cost-utility frontier: what any budget buys, from minorminer alone and from composing.

`runs/external_baseline.log` settled that racing the policy against minorminer attempt for
attempt is a tie, and that minorminer wins on time because its attempts cost a quarter as much.
That is an argument about price, so the honest way to present it is a price curve rather than a
single matched point: for a ladder of budgets, what utility does each method reach?

Two families of arms, both on the same held-out lineages and the same evaluator:

  * `minorminer best-of-N` for a ladder of N. This is the frontier everything else must clear.
  * `compose(N, c)`: draw N minorminer embeddings, evaluate them, then spend one policy episode
    improving each of the best `c` and evaluate those. The policy is used where its improvement
    operator is worth its price, on an already-good embedding, instead of being asked to produce
    embeddings from nothing in competition with a cheaper generator.

Every arm reports its own wall clock, so a reader can drop a vertical line at any budget and see
which method is higher. A compose point above the minorminer curve at the same time is the claim;
a compose point below it is the refutation, and either is reported.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import torch
from isingfold.rl.contracts import Context, Opcode, TerminalRecord
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import (LEGACY_ONLINE_INITIALIZER_RESTARTS_V1, EmbeddingEnv, Mode,
                              fixed_strength_selector)
from isingfold.rl.evaluate import (first_commit_controller, run_controller, secondary_metrics,
                                   torch_controller)
from isingfold.rl.model import build_model

from _initializers import minorminer_initializer


def fixed_initializer(chains):
    """Protect the episode with one specific embedding, whatever seed the environment passes."""
    def _init(logical, host, seed):
        return chains
    return _init


def blind_improve(task, ctx, controller, chains, seed, rounds, max_steps=64, stride=0,
                  on_checkpoint=None):
    """Run the policy for `rounds` episodes, feeding each returned embedding into the next, and
    never call the evaluator in between.

    This is how the Track A embedder spends its budget: hundreds of destroy-and-repair moves,
    one measurement at the end. Every composed arm so far paid one evaluator block per policy
    pass, which is what made iteration expensive; it is expensive only because the probe was
    measuring after every step rather than because the method needs to. What this cannot do is
    pick the best round, since picking needs a measurement, so it returns the last state and
    lets the caller pay for exactly one.
    """
    current = chains
    for r in range(rounds):
        env = EmbeddingEnv(task, ctx, mode=Mode.IMPROVEMENT,
                           initializer=lambda l, h, sd, c=current: c,
                           selector=fixed_strength_selector(), seed=seed + 7919 * r,
                           improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1)
        rng = np.random.default_rng(seed + 7919 * r)
        result = env.reset(seed + 7919 * r)
        if not hasattr(result, "candidates"):
            return current if r else None
        steps = 0
        while not isinstance(result, TerminalRecord) and steps < max_steps:
            chosen = int(controller(result, rng))
            result = env.step(result, chosen, evaluate_training_reward=False).next_decision_or_terminal
            steps += 1
        if not isinstance(result, TerminalRecord) or not result.returned_valid:
            return current if r else None
        if result.embedding is None:
            return current if r else None
        current = {n: frozenset(c) for n, c in result.embedding.items()}
        # Iterating blind is a random walk once the policy has made the moves it is confident
        # about, so the run can be given a measurement every `stride` rounds to keep the best
        # state instead of whatever the last round happened to produce. That is what the Track A
        # embedder buys with its learned screen, bought here with evaluator reads and counted.
        if stride and on_checkpoint is not None and (r + 1) % stride == 0:
            on_checkpoint(current)
    return current


def resource_key(outcome):
    """What a resource-first embedder would rank by: fewest qubits, then shortest chain.

    This is the criterion minorminer itself optimises. Ranking the same draws by it instead of by
    the objective is the paper's thesis stated as an experiment: if the objective-ranked curve
    sits above the resource-ranked one, qubit count is not a quality signal.
    """
    return (outcome.qubits, outcome.max_chain)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="runs/corpus_c4x10")
    ap.add_argument("--lineages", type=int, default=60)
    ap.add_argument("--ladder", default="4,8,16,24,40,80",
                    help="the minorminer budgets that form the frontier")
    ap.add_argument("--compose", default="8:1,8:4,16:4",
                    help="comma list of N:c or N:c:T. N minorminer draws, the best c of them "
                         "improved by the policy, and each improvement iterated T times by "
                         "feeding the returned embedding back in as the next episode\'s floor. "
                         "T > 1 asks whether improvement compounds or saturates after one pass")
    ap.add_argument("--blind", default="",
                    help="comma list of N:T or N:T:k. N minorminer draws, then T policy rounds "
                         "on the best of them. With no k the rounds are blind and one block is "
                         "spent at the end, which is the Track A budget shape: many moves, one "
                         "measurement. With k the run is measured every k rounds and the best "
                         "state is kept, and every one of those measurements is charged")
    ap.add_argument("--checkpoints", default="",
                    help="comma list of family=path; defaults to the bestof3 run")
    ap.add_argument("--mm-tries", type=int, default=10)
    ap.add_argument("--qubit-cap", type=int, default=120)
    ap.add_argument("--reward-reads", type=int, default=128)
    a = ap.parse_args()

    tasks = load_instances(a.corpus)
    split = json.loads((Path(a.corpus) / "splits.json").read_text())
    held = set(split["validation"]) | set(split["test"])
    dev = [t for t in tasks if t.lineage in held][: a.lineages]
    ctx = Context(qubit_cap=a.qubit_cap)
    mm = minorminer_initializer(a.mm_tries)
    ladder = [int(x) for x in a.ladder.split(",") if x]
    compose = []
    for spec in a.compose.split(","):
        if not spec:
            continue
        parts = [int(y) for y in spec.split(":")]
        compose.append(tuple(parts) if len(parts) == 3 else (parts[0], parts[1], 1))
    ckpts = dict(kv.split("=", 1) for kv in a.checkpoints.split(",") if kv) or {
        f: "runs/bestof3_%s_s0/policy.pt" % f for f in ("if-core", "if-dual", "if-mlp")}
    blind = []
    for spec in a.blind.split(","):
        if spec:
            parts = [int(x) for x in spec.split(":")]
            blind.append(tuple(parts) if len(parts) == 3 else (parts[0], parts[1], 0))
    print(json.dumps({"lineages": len(dev), "ladder": ladder, "compose": compose,
                      "blind": blind, "mm_tries": a.mm_tries}), flush=True)

    def episode(task, controller, initializer, seed):
        """One attempt: returns (utility, chains) or None when the attempt did not return."""
        try:
            out = run_controller([task], ctx, controller, initializer=initializer,
                                 selector=fixed_strength_selector(),
                                 reward_reads=a.reward_reads, repetitions=1, seed=seed)
        except Exception:
            return None
        m = secondary_metrics(out)
        if not m["valid_returns"]:
            return None
        return float(m["utility_mean"]), out[0]


    # One shared pool of minorminer draws per lineage, so every arm that uses N of them uses the
    # same N. Arms then differ only in what they do with the pool, never in the luck of the draw.
    biggest = max(ladder + [n for n, _, _ in compose])
    pool = {}
    started = time.time()
    for t in dev:
        draws = []
        for j in range(biggest):
            chains = mm(t.logical, t.host, 5000 + 97 * j)
            if chains is None:
                continue
            r = episode(t, first_commit_controller, fixed_initializer(chains), 5000 + 97 * j)
            if r is not None:
                draws.append((r[0], j, chains, resource_key(r[1])))
        # Kept in draw order. best-of-N means the first N draws, so the ladder measures what a
        # budget actually buys; sorting here would have made every ladder point equal to the
        # maximum over the whole pool and the frontier flat.
        pool[t.name] = draws
    pool_secs = time.time() - started
    per_draw = pool_secs / max(1, len(dev) * biggest)
    print("  drew %d minorminer embeddings per lineage in %.1fs (%.3fs each)"
          % (biggest, pool_secs, per_draw), flush=True)

    def interpolate(curve, x):
        """Read the frontier at an arbitrary budget instead of rounding down to a ladder point.

        Rounding down was a real bias in the first version of this probe: an arm costing 132.6s
        was compared against the 83s ladder point and collected 49 seconds of free compute.
        """
        pts = sorted(curve)
        if x <= pts[0][0]:
            return pts[0][1]
        if x >= pts[-1][0]:
            return pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return pts[-1][1]

    print("\n== the minorminer frontier, draws ranked by the objective")
    frontier = {}
    for n in ladder:
        vals = {name: max(d[0] for d in draws[:n]) for name, draws in pool.items() if draws}
        # The pool is already paid for; the cost of this point is n draws at the measured price.
        secs = per_draw * len(dev) * n
        frontier[n] = (vals, secs)
        print("  minorminer best-of-%-3d utility %.4f over %d lineages, %.1fs, %d evaluator blocks"
              % (n, float(np.mean(list(vals.values()))), len(vals), secs, n), flush=True)

    print("\n== the same draws, ranked by qubit count instead (resource-first selection)")
    resource = {}
    for n in ladder:
        vals = {}
        for name, draws in pool.items():
            if not draws:
                continue
            pick = min(draws[:n], key=lambda d: d[3])
            vals[name] = pick[0]
        resource[n] = vals
        print("  resource-picked best-of-%-3d utility %.4f over %d lineages"
              % (n, float(np.mean(list(vals.values()))), len(vals)), flush=True)

    def report(label, arm, ref, ref_label):
        keys = sorted(set(arm) & set(ref))
        if len(keys) < 5:
            print("  %-52s (too few paired lineages: %d)" % (label, len(keys))); return
        d = np.array([arm[i] - ref[i] for i in keys])
        rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        print("  %-52s n %3d  %+.4f [%+.4f, %+.4f]  (vs %s)"
              % (label, len(d), d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5),
                 ref_label), flush=True)

    for fam, path in ckpts.items():
        if not Path(path).exists():
            print("\n== %s: no checkpoint at %s" % (fam, path)); continue
        model = build_model(fam, improvement_mode=True)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        controller = torch_controller(model, None, greedy=False)
        print("\n== compose with %s (%s)" % (fam, path))
        for n, c, iters in compose:
            started = time.time()
            vals = {}
            for t in dev:
                draws = pool.get(t.name, [])[:n]
                if not draws:
                    continue
                best = max(d[0] for d in draws)
                # The policy is spent on the best c of the N draws, which is the choice a user
                # with a budget would make: the evaluator has already scored all N.
                for _, j, chains, _res in sorted(draws, key=lambda d: -d[0])[:c]:
                    current = chains
                    for it in range(iters):
                        r = episode(t, controller, fixed_initializer(current),
                                    7000 + 97 * j + 13 * it)
                        if r is None:
                            break
                        best = max(best, r[0])
                        nxt = getattr(r[1], "returned_embedding", None)
                        if nxt is None:
                            break
                        current = nxt
                vals[t.name] = best
            secs = time.time() - started + per_draw * len(dev) * n
            blocks = n + c * iters
            mean = float(np.mean(list(vals.values()))) if vals else float("nan")
            print("  compose(%d draws, improve %d, iterate %d) utility %.4f over %d lineages, "
                  "%.1fs, %d evaluator blocks"
                  % (n, c, iters, mean, len(vals), secs, blocks), flush=True)
            # Two budgets, because the two say different things and only reporting the kinder one
            # would be a choice made for the answer it gives.
            #
            #   wall clock  - what this implementation costs on this machine. minorminer is C++,
            #                 the environment around the policy is Python, and the profile says
            #                 the gap is that overhead rather than the method.
            #   evaluator blocks - what the application pays. Each block is one programming and
            #                 one read budget on the annealer, which is the scarce resource in
            #                 deployment; classical embedding time is preprocessing beside it.
            time_curve = [(frontier[m][1], float(np.mean(list(frontier[m][0].values()))))
                          for m in ladder]
            block_curve = [(float(m), float(np.mean(list(frontier[m][0].values())))) for m in ladder]
            print("      at %.1fs the minorminer curve reads %.4f, so composing is %+.4f on time"
                  % (secs, interpolate(time_curve, secs), mean - interpolate(time_curve, secs)),
                  flush=True)
            print("      at %d blocks it reads %.4f, so composing is %+.4f on evaluator budget"
                  % (blocks, interpolate(block_curve, blocks),
                     mean - interpolate(block_curve, blocks)), flush=True)
            # The paired interval still needs a concrete reference arm, so use the nearest ladder
            # point in blocks, and say how far off it is rather than hiding the mismatch.
            m = min(ladder, key=lambda q: abs(q - blocks))
            report("compose(%d,%d,%d) against minorminer best-of-%d" % (n, c, iters, m),
                   vals, frontier[m][0],
                   "%d blocks vs %d blocks" % (blocks, m))

        for n, rounds, stride in blind:
            started = time.time()
            vals, spent = {}, []
            for t in dev:
                draws = pool.get(t.name, [])[:n]
                if not draws:
                    continue
                best = max(d[0] for d in draws)
                top = sorted(draws, key=lambda d: -d[0])[0]
                charged = [0]

                def measure(state, t=t, top=top, charged=charged):
                    charged[0] += 1
                    got = episode(t, first_commit_controller, fixed_initializer(state),
                                  9100 + top[1] + 31 * charged[0])
                    if got is not None:
                        measure.best = max(getattr(measure, "best", -1.0), got[0])
                measure.best = -1.0

                final = blind_improve(t, ctx, controller, top[2], 9000 + top[1], rounds,
                                      stride=stride, on_checkpoint=measure if stride else None)
                if not stride and final is not None:
                    charged[0] += 1
                    got = episode(t, first_commit_controller, fixed_initializer(final),
                                  9100 + top[1])
                    if got is not None:
                        best = max(best, got[0])
                best = max(best, measure.best)
                spent.append(charged[0])
                vals[t.name] = best
            secs = time.time() - started + per_draw * len(dev) * n
            blocks = n + (int(round(sum(spent) / max(1, len(spent)))) if spent else 1)
            mean = float(np.mean(list(vals.values()))) if vals else float("nan")
            time_curve = [(frontier[m][1], float(np.mean(list(frontier[m][0].values()))))
                          for m in ladder]
            block_curve = [(float(m), float(np.mean(list(frontier[m][0].values())))) for m in ladder]
            print("  blind(%d draws, %d rounds, stride %d) utility %.4f over %d lineages, "
                  "%.1fs, %d evaluator blocks"
                  % (n, rounds, stride, mean, len(vals), secs, blocks), flush=True)
            print("      at %.1fs the minorminer curve reads %.4f, so blind iteration is %+.4f on time"
                  % (secs, interpolate(time_curve, secs), mean - interpolate(time_curve, secs)),
                  flush=True)
            print("      at %d blocks it reads %.4f, so blind iteration is %+.4f on evaluator budget"
                  % (blocks, interpolate(block_curve, blocks),
                     mean - interpolate(block_curve, blocks)), flush=True)
            m = min(ladder, key=lambda q: abs(q - blocks))
            report("blind(%d,%d,%d) against minorminer best-of-%d" % (n, rounds, stride, m),
                   vals, frontier[m][0], "%d blocks vs %d blocks" % (blocks, m))

    print("\n== objective ranking against resource ranking, on identical draws")
    for n in ladder:
        report("objective-picked best-of-%d against resource-picked best-of-%d" % (n, n),
               frontier[n][0], resource[n], "same %d draws" % n)

    print("\nFRONTIER DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
