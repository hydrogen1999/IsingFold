"""minorminer's validity on the fill-planted corpus as its budget grows.

The regime above eighty percent fill is where the standard tool fails at its deployed budget.
Before anything is built for that regime the baseline gets more time, because the earlier
threshold sweep moved by a variable at five times the tries and a learned embedder measured
against a starved baseline would be the fair-budget mistake this project has already made once.
"""
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from isingfold.rl.data.generate import load_instances
from _initializers import minorminer_initializer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--tries", default="50,200")
    a = ap.parse_args()
    tasks = load_instances(a.corpus)
    cap = tasks[0].host.number_of_nodes()
    print(json.dumps({"corpus": a.corpus, "instances": len(tasks), "qubits": cap}), flush=True)
    for tries in (int(t) for t in a.tries.split(",")):
        mm = minorminer_initializer(tries)
        cells = defaultdict(lambda: [0, 0, [], []])
        for k, task in enumerate(tasks):
            family = task.lineage.rsplit("-", 1)[0].split("-", 1)[1].rsplit("-", 1)[0]
            c = cells[family]
            t0 = time.time()
            ch = mm(task.logical, task.host, 77_000 + k)
            c[3].append(time.time() - t0)
            c[1] += 1
            if ch is not None:
                c[0] += 1
                c[2].append(sum(len(x) for x in ch.values()) / cap)
        print("\n  tries %d" % tries)
        print("  %-16s %3s %8s %8s %8s" % ("cell", "n", "valid", "fill", "secs"))
        for family in sorted(cells):
            v, n, fills, secs = cells[family]
            print("  %-16s %3d %8.2f %8s %8.1f"
                  % (family, n, v / max(1, n), ("%.2f" % np.mean(fills)) if fills else "-",
                     np.mean(secs)), flush=True)
    print("\nFILL BUDGET DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
