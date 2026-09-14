#!/usr/bin/env python3
"""Hand-test 06: the splits are disjoint, and a case cannot appear on both sides of one.

What a person would do on paper
-------------------------------
1. Write out a few thousand split unit ids.
2. Ask the generator for each one's split. Every id gets exactly one of train, val, test.
3. Take the same id and ask again, and ask from a *fresh process*. The answer must not change.
   Python salts the hash of a string per process, so a splitter written with the builtin `hash`
   gives a different split on every run, and a corpus generated in two sessions leaks.
4. Give one logical case several instance ids that share its split unit. They must all land in
   the same split: that is what the split unit is for.
5. Count the three buckets. They should be near the declared 70 / 10 / 20.

What a failure means
--------------------
Development leaks into confirmation. Every number measured on the held-out side is then partly a
number about data the method was tuned on, and no amount of later care repairs it.
"""
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import check, explain_and_exit_if_asked
from embedbench.candidate_bank import assign_split

UNITS = [f"chimera5-compact-{i}" for i in range(2000)] + \
        [f"pegasus16-cut_congested-{i}" for i in range(2000)]


def main() -> int:
    explain_and_exit_if_asked(__doc__)
    first = {u: assign_split(u) for u in UNITS}
    check(set(first.values()) <= {"train", "val", "test"},
          f"a unit was assigned something that is not a split: {set(first.values())}")

    again = {u: assign_split(u) for u in UNITS}
    check(first == again, "assign_split is not a function: the same id gave two answers")

    buckets = {s: {u for u, v in first.items() if v == s} for s in ("train", "val", "test")}
    for a in buckets:
        for b in buckets:
            if a < b:
                shared = buckets[a] & buckets[b]
                check(not shared, f"{len(shared)} ids are in both {a} and {b}")
    check(set().union(*buckets.values()) == set(UNITS), "some ids fell out of every split")
    print("  ok    every id is in exactly one split, and the three are disjoint")

    # A logical case that produces many instances must not be split across them.
    for case in ("chimera5-compact-7", "pegasus16-cut_congested-123"):
        splits = {assign_split(case) for _ in range(5)}
        check(len(splits) == 1, f"case {case} landed in {splits}")
    print("  ok    a split unit keeps all of its instances on one side")

    probe = UNITS[:8]
    code = ("import sys;sys.path.insert(0,%r);"
            "from embedbench.candidate_bank import assign_split;"
            "print(','.join(assign_split(u) for u in %r))"
            % (str(Path(__file__).resolve().parents[1] / "src"), probe))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "random"})
    check(out.returncode == 0, f"the fresh-process probe failed: {out.stderr.strip()[:200]}")
    fresh = out.stdout.strip().split(",")
    check(fresh == [first[u] for u in probe],
          f"a fresh process assigns different splits: {fresh} vs {[first[u] for u in probe]}")
    print("  ok    a fresh process with a different hash seed assigns the same splits")

    counts = Counter(first.values())
    total = sum(counts.values())
    shares = {s: counts[s] / total for s in ("train", "val", "test")}
    print(f"  shares train {shares['train']:.3f}, val {shares['val']:.3f}, "
          f"test {shares['test']:.3f} over {total} ids")
    for name, want in (("train", 0.70), ("val", 0.10), ("test", 0.20)):
        check(abs(shares[name] - want) < 0.03,
              f"{name} is {shares[name]:.3f}, declared {want:.2f}")
    print("  ok    the buckets match the declared 70 / 10 / 20")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
