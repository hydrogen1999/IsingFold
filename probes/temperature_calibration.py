"""At what sampling temperature does a training episode stay valid, and does quality still vary?

The bounded utility gives an invalid episode zero and a valid one between 0.5 and 1, so a single
invalid construction in a batch costs about 0.99 while the entire residual gap the policy is
being asked to learn is worth 0.0079. At a training validity of 0.875 the standard deviation the
validity coin alone puts into a leave-one-out advantage is 0.33, forty-one times the signal. No
amount of training fixes that; the batch has to stop containing invalid episodes.

Sampling temperature is the lever, and it cuts both ways: at a low enough temperature the policy
is nearly deterministic, every episode is valid and the advantage is clean, but the episodes stop
differing from each other and there is nothing left to learn from. This measures both sides at
once, so a temperature can be chosen on evidence instead of taste: the fraction of episodes that
reach a valid COMMIT, and the spread of measured residual among the ones that do.
"""
import argparse, json, math, os, statistics, sys
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
import torch

import constructor_curriculum as cc
from constructor_objective import measure_terminal
from constructor_rollout import episode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cells", default="")
    ap.add_argument("--init", required=True)
    ap.add_argument("--features", default="local")
    ap.add_argument("--actor", default="linear")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--support", default="wide")
    ap.add_argument("--temperatures", default="1.0,0.7,0.5,0.3,0.15")
    ap.add_argument("--n-tasks", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--episode-seconds", type=float, default=90.)
    ap.add_argument("--reward-reads", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20260929)
    a = ap.parse_args()

    cells = [c for c in a.cells.split(",") if c]
    tasks, _ = cc.build_corpus_sets(a.corpus, cells, a.n_tasks, 1, a.seed)
    actor = cc.make_actor(a.actor, a.width, cc.FEATURE_WIDTHS[a.features])
    cc.load_init(a.init, actor, a.actor, a.features)
    wide = a.support == "wide"
    temps = [float(x) for x in a.temperatures.split(",")]

    print(json.dumps({"probe": "temperature_calibration", "corpus": a.corpus, "cells": cells,
                      "init": a.init, "temperatures": temps, "tasks": len(tasks),
                      "episodes_per_task": a.episodes}), flush=True)
    print("  %6s %8s %10s %12s %12s %10s"
          % ("temp", "valid", "episodes", "quality sd", "validity sd", "ratio"), flush=True)

    from constructor_objective import quality_utility

    for temp in temps:
        # Everything is computed within an instance and then averaged over instances. Pooling
        # residuals across instances measures how much instances differ, which is not what a
        # leave-one-out advantage inside one instance ever sees.
        per_instance, valid_total, total = [], 0, 0
        for k, t in enumerate(tasks):
            fc = cc.make_features(a.features, t)
            utilities, valid_here, here = [], 0, 0
            for e in range(a.episodes):
                here += 1
                with torch.no_grad():
                    rec = episode(t, actor, fc, temp, a.max_steps,
                                  np.random.default_rng(a.seed + 1000 * k + e),
                                  a.episode_seconds, train=True, objective="quality",
                                  evaluate_reward=False, wide=wide)
                if not rec["valid"]:
                    continue
                valid_here += 1
                r = measure_terminal(t, rec["terminal"], a.seed + 50000 + 100 * k + e,
                                     a.reward_reads)
                # The instance's own divisor, so this is in the units the advantage works in.
                utilities.append(quality_utility(t, r))
            total += here
            valid_total += valid_here
            rate = valid_here / here if here else 0.0
            spread = statistics.stdev(utilities) if len(utilities) > 1 else 0.0
            per_instance.append({"task": t.name, "valid_rate": rate, "valid": valid_here,
                                 "episodes": here, "quality_sd_in_utility": spread,
                                 "mean_utility": (sum(utilities) / len(utilities))
                                                 if utilities else 0.0})
        rate = valid_total / total if total else 0.0
        qsd = statistics.median([r["quality_sd_in_utility"] for r in per_instance])
        # The validity term is the spread a Bernoulli at this rate puts into the same advantage,
        # within an instance. At zero observed failures it is not zero but unmeasured: a rate of
        # one over n episodes is consistent with a true failure rate up to about 3/n.
        rates = [r["valid_rate"] for r in per_instance]
        vsd = statistics.median([0.99 * math.sqrt(x * (1 - x)) for x in rates])
        unresolved = sum(1 for x in rates if x in (0.0, 1.0))
        ratio = (qsd / vsd) if vsd > 0 else None
        print("  %6.2f %8.2f %10d %12.4f %12s %10s"
              % (temp, rate, total, qsd, "%.4f" % vsd,
                 ("%.2f" % ratio) if ratio is not None else "unmeasured"), flush=True)
        print(json.dumps({"temperature": temp, "valid_rate": rate, "episodes": total,
                          "median_quality_sd_in_utility": qsd,
                          "median_validity_sd_in_utility": vsd,
                          "quality_over_validity": ratio,
                          "instances_with_no_failure_observed": unresolved,
                          "instances": per_instance,
                          "reading": ("both columns are within-instance and in utility units. A "
                                      "ratio above one means the spread among an instance's own "
                                      "valid episodes is larger than the spread its failures "
                                      "inject. Where no failure was observed the validity term "
                                      "is unmeasured rather than zero, and the instance count "
                                      "for that is reported.")}), flush=True)
    print("TEMPERATURE CALIBRATION DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
