"""Is there a resource-quality trade-off here at all?

The budget-conditioned design rests on a premise: that spending more qubits buys better solve
probability, so that sweeping a qubit budget traces a frontier worth tracing. That premise has
never been measured on this data. If quality is flat in resource, conditioning a policy on a
qubit budget conditions it on something that does not matter, and the frontier is a line.

Everything here is computed from labels that already exist, so it costs no reads. Three
questions, in order:

  1. Within one state, does the candidate using more qubits tend to score better?
  2. How often do the two orderings actually disagree? A proxy being imperfect is a weaker claim
     than a proxy being wrong, and the design asks for concrete counterexamples rather than the
     word "imperfect".
  3. What does the empirical frontier look like: best measured quality at each resource level,
     and how much of the spread in quality is explained by resource at all?
"""
import argparse, json, pickle, sys
from collections import defaultdict

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="train", choices=["train", "eval", "both"])
    a = ap.parse_args()

    with open(a.cache, "rb") as fh:
        blob = pickle.load(fh)
    states = (blob["train"] if a.split == "train" else blob["eval"] if a.split == "eval"
              else blob["train"] + blob["eval"])
    print(json.dumps({"cache": a.cache, "split": a.split, "states": len(states),
                      "candidates": sum(len(s["rows"]) for s in states)}), flush=True)

    within, pairs, inversions, ties = [], 0, 0, 0
    flat_states = 0
    per_state_spread = []
    for st in states:
        q = np.array([float(r["qubits"]) for r in st["rows"]])
        p = np.array([float(r["rate"]) for r in st["rows"]])
        per_state_spread.append(float(p.max() - p.min()))
        if len(set(q.tolist())) < 2:
            flat_states += 1
        elif len(set(p.tolist())) >= 2:
            c = np.corrcoef(q, p)[0, 1]
            if np.isfinite(c):
                within.append(float(c))
        for i in range(len(q)):
            for j in range(i + 1, len(q)):
                if q[i] == q[j]:
                    continue
                pairs += 1
                cheaper, dearer = (i, j) if q[i] < q[j] else (j, i)
                if p[cheaper] > p[dearer]:
                    inversions += 1
                elif p[cheaper] == p[dearer]:
                    ties += 1

    print("\n1. Within a state, does spending more qubits score better?")
    if within:
        print("   correlation of qubit count with measured quality: median %+.3f, mean %+.3f"
              % (float(np.median(within)), float(np.mean(within))))
        print("   states where it is positive: %d of %d"
              % (sum(1 for c in within if c > 0), len(within)))
    print("   states where every candidate uses the same qubit count: %d of %d"
          % (flat_states, len(states)))

    print("\n2. How often do the two orderings disagree?")
    if pairs:
        print("   comparable pairs %d" % pairs)
        print("   the cheaper embedding scored strictly better: %d (%.1f%%)"
              % (inversions, 100.0 * inversions / pairs))
        print("   the two tied on quality:                     %d (%.1f%%)"
              % (ties, 100.0 * ties / pairs))
        print("   so resource ordering predicts quality ordering %.1f%% of the time"
              % (100.0 * (pairs - inversions - ties) / pairs))

    print("\n3. What does the frontier look like?")
    by_q = defaultdict(list)
    for st in states:
        for r in st["rows"]:
            by_q[int(r["qubits"])].append(float(r["rate"]))
    rows = sorted(by_q)
    lo, hi = rows[0], rows[-1]
    width = max(1, (hi - lo) // 8)
    buckets = defaultdict(list)
    for qq, vals in by_q.items():
        buckets[lo + ((qq - lo) // width) * width] += vals
    print("   %-14s %-8s %-8s %-8s %s" % ("qubits", "n", "mean", "best", "spread"))
    for b in sorted(buckets):
        v = np.array(buckets[b])
        print("   %-14s %-8d %-8.4f %-8.4f %.4f"
              % ("%d-%d" % (b, b + width - 1), len(v), v.mean(), v.max(), v.max() - v.min()))

    allq = np.array([float(r["qubits"]) for st in states for r in st["rows"]])
    allp = np.array([float(r["rate"]) for st in states for r in st["rows"]])
    c = np.corrcoef(allq, allp)[0, 1]
    print("\n   pooled correlation of qubits with quality: %+.3f" % c)
    print("   variance of quality explained by qubit count: %.1f%%" % (100.0 * c * c))
    print("   mean spread of quality within one state: %.4f" % float(np.mean(per_state_spread)))
    print("\n   A trade-off worth conditioning on needs the frontier to rise and the pooled")
    print("   correlation to carry real variance. A within-state spread much larger than what")
    print("   resource explains means the budget is not the variable that decides quality.")
    print("\nFRONTIER PREMISE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
