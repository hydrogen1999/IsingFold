"""minorminer as an anytime system: validity against a wall-time deadline.

The fair baseline for a learned constructor is not minorminer at ten tries but minorminer
given the same wall time, restarted with fresh seeds until the deadline, every attempt counted
including the failures. For each instance this records the time to the first valid embedding
within the largest deadline; validity at any smaller deadline follows from that. Tries per call
is a tuning knob and is reported with the result.
"""
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
from isingfold.rl.data.generate import load_instances
from _initializers import minorminer_initializer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--deadlines", default="30,60,120,300")
    ap.add_argument("--tries", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    deadlines = [float(x) for x in a.deadlines.split(",")]
    horizon = max(deadlines)
    tasks = load_instances(a.corpus)
    cap = tasks[0].host.number_of_nodes()
    mm = minorminer_initializer(a.tries)
    print(json.dumps({"corpus": a.corpus, "instances": len(tasks), "qubits": cap,
                      "tries_per_call": a.tries, "deadlines": deadlines}), flush=True)
    records = defaultdict(list)
    for k, task in enumerate(tasks):
        family = task.lineage.rsplit("-", 1)[0].split("-", 1)[1].rsplit("-", 1)[0]
        t0 = time.time()
        attempts, first_valid, fill = 0, None, None
        while time.time() - t0 < horizon:
            attempts += 1
            ch = mm(task.logical, task.host, 40_000 + 1000 * k + attempts + a.seed)
            if ch is not None:
                first_valid = time.time() - t0
                fill = sum(len(c) for c in ch.values()) / cap
                break
        records[family].append((first_valid, attempts, fill, time.time() - t0))
        print("  %3d/%d %-24s %s attempts %d" % (k + 1, len(tasks), task.name,
              ("valid at %.0fs" % first_valid) if first_valid is not None
              else "none within %.0fs" % horizon, attempts), flush=True)
    print("\n  %-16s %3s " % ("cell", "n") + " ".join("%7s" % ("<=%.0fs" % d) for d in deadlines)
          + " %9s %8s" % ("attempts", "mmfill"))
    for family in sorted(records):
        rows = records[family]
        cols = " ".join("%7.2f" % np.mean([r[0] is not None and r[0] <= d for r in rows])
                        for d in deadlines)
        fills = [r[2] for r in rows if r[2] is not None]
        print("  %-16s %3d %s %9.1f %8s" % (family, len(rows), cols,
              np.mean([r[1] for r in rows]), ("%.2f" % np.mean(fills)) if fills else "-"))
    print("\n  Validity at each deadline is the fraction of instances whose first valid embedding")
    print("  arrived within it; every attempt, including failures, is counted in wall time.")
    print("\nANYTIME BASELINE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
