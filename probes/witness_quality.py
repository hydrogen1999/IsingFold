"""Is a planted embedding at high fill worth anything to the annealer?

A validity win in the regime where minorminer fails only matters if the valid embeddings there
solve something. For each instance of a fill-planted corpus this measures the witness
embedding's utility on independent reads, and alongside it minorminer's best of K draws where
minorminer finds any, so the two can be read on one line per cell.
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
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.evaluate import first_commit_controller, run_controller

from _context import host_context
from _initializers import minorminer_initializer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--reads", type=int, default=256)
    ap.add_argument("--assess-reads", type=int, default=512)
    a = ap.parse_args()

    tasks = load_instances(a.corpus)
    cap = tasks[0].host.number_of_nodes()
    ctx = host_context(cap)
    mm = minorminer_initializer(10)
    print(json.dumps({"corpus": a.corpus, "instances": len(tasks), "qubits": cap}), flush=True)

    def measure(task, chains, seed, reads):
        chains = {v: frozenset(c) for v, c in chains.items()}
        def fixed(l, h, s):
            return chains
        try:
            out = run_controller([task], ctx, first_commit_controller, initializer=fixed,
                                 selector=fixed_strength_selector(), reward_reads=reads,
                                 repetitions=1, seed=seed)
        except Exception as e:
            print("  measure failed on %s: %s" % (task.name, e), flush=True)
            return None
        o = out[0]
        return float(o.utility) if o.returned_valid and o.utility is not None else None

    cells = defaultdict(lambda: {"witness": [], "mm_best": [], "mm_valid": 0, "n": 0})
    for k, task in enumerate(tasks):
        family = task.lineage.rsplit("-", 1)[0].split("-", 1)[1].rsplit("-", 1)[0]
        cell = cells[family]
        cell["n"] += 1
        if task.witness:
            u = measure(task, task.witness, 90_000_000 + k, a.assess_reads)
            if u is not None:
                cell["witness"].append(u)
        draws = []
        for j in range(a.k):
            ch = mm(task.logical, task.host, 5_000 + 97 * j + 1000 * k)
            if ch is None:
                continue
            u = measure(task, ch, 5_000 + 97 * j + 1000 * k, a.reads)
            if u is not None:
                draws.append((u, ch))
        if draws:
            cell["mm_valid"] += 1
            best = max(draws, key=lambda t: t[0])[1]
            u = measure(task, best, 90_000_000 + 13 * k + 7, a.assess_reads)
            if u is not None:
                cell["mm_best"].append(u)
        print("  %3d/%d %s" % (k + 1, len(tasks), task.name), flush=True)

    print("\n  %-16s %3s %10s %9s %10s" % ("cell", "n", "witness", "mm valid", "mm best"))
    for family in sorted(cells):
        c = cells[family]
        print("  %-16s %3d %10s %9.2f %10s"
              % (family, c["n"],
                 ("%.4f" % np.mean(c["witness"])) if c["witness"] else "-",
                 c["mm_valid"] / max(1, c["n"]),
                 ("%.4f" % np.mean(c["mm_best"])) if c["mm_best"] else "-"))
    print("\nWITNESS QUALITY DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
