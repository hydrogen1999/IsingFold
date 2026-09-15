"""The frontier question asked correctly: within one instance, does a larger budget buy more?

An earlier probe correlated qubit count with quality across the candidates at a state and
concluded that the resource-quality premise fails. That conclusion was too strong, and the design
document says why: the monotonicity it claims is a property of the optimum at each budget,
because feasible sets are nested, not a property of sampled outputs. Candidates at a state are
local perturbations of one embedding, so that measurement answers whether a local repair that
spends a qubit helps. It does not answer whether the best embedding available under a larger
budget beats the best available under a smaller one.

This asks the second question. Within each instance, candidates are bucketed by the qubits they
use and the best measured quality in each bucket is taken. Then, per instance, the best below a
threshold is compared with the best at or above it. That comparison is paired inside an instance,
so it is not confounded by some instances being harder than others, which is what pooling across
instances does.

Two things it still cannot do. The best of a handful of sampled candidates is not the optimum at
that budget, so a flat result is evidence of a weak effect over the reachable candidates rather
than a statement about the frontier of optima. And the maximum over noisy measurements is
inflated, more so in the bucket with more candidates in it, so the counts per bucket are printed
beside the numbers.
"""
import argparse, json, pickle, sys
from collections import defaultdict

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="both", choices=["train", "eval", "both"])
    a = ap.parse_args()

    with open(a.cache, "rb") as fh:
        blob = pickle.load(fh)
    states = (blob["train"] if a.split == "train" else blob["eval"] if a.split == "eval"
              else blob["train"] + blob["eval"])

    # Candidates of one instance, gathered across the states sampled from it.
    per_instance = defaultdict(list)
    for st in states:
        for r in st["rows"]:
            per_instance[st["lineage"]].append((int(r["qubits"]), float(r["rate"])))
    print(json.dumps({"cache": a.cache, "instances": len(per_instance),
                      "candidates": sum(len(v) for v in per_instance.values())}), flush=True)

    rows, spans = [], []
    for lineage, cands in per_instance.items():
        qs = sorted({q for q, _ in cands})
        if len(qs) < 2:
            continue
        cut = qs[len(qs) // 2]
        low = [p for q, p in cands if q < cut]
        high = [p for q, p in cands if q >= cut]
        if not low or not high:
            continue
        rows.append((max(low), max(high), len(low), len(high),
                     float(np.mean([q for q, _ in cands if q < cut])),
                     float(np.mean([q for q, _ in cands if q >= cut]))))
        spans.append(qs[-1] - qs[0])

    if not rows:
        print("  no instance offered two different resource levels"); return 1
    lo = np.array([r[0] for r in rows])
    hi = np.array([r[1] for r in rows])
    d = hi - lo
    rng = np.random.default_rng(0)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
    print("\n  instances with two resource levels: %d" % len(rows))
    print("  qubits spanned within an instance:   %.1f mean, %d max"
          % (float(np.mean(spans)), int(max(spans))))
    print("  candidates in the cheaper half:      %.1f mean" % float(np.mean([r[2] for r in rows])))
    print("  candidates in the dearer half:       %.1f mean" % float(np.mean([r[3] for r in rows])))
    print("  mean qubits, cheaper half:           %.1f" % float(np.mean([r[4] for r in rows])))
    print("  mean qubits, dearer half:            %.1f" % float(np.mean([r[5] for r in rows])))
    print("\n  best in the cheaper half:            %.4f" % lo.mean())
    print("  best in the dearer half:             %.4f" % hi.mean())
    print("  dearer minus cheaper:                %+.4f [%+.4f, %+.4f]"
          % (d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
    print("  instances where the dearer half wins: %d of %d" % (int((d > 0).sum()), len(d)))
    print("  instances where it loses:             %d" % int((d < 0).sum()))
    print("  ties:                                 %d" % int((d == 0).sum()))
    print("\n  The dearer half holds more candidates on average, and a maximum over more noisy")
    print("  measurements is larger even when nothing improves, so a small positive difference")
    print("  here is not yet evidence that budget buys quality.")
    print("\nWITHIN FRONTIER DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
