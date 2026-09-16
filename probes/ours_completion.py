"""Our completion search against minorminer's, both from the same roots.

Arms, each from witness roots and from a random half of them: our backtracking completion
(route_search.complete) under the instance's qubit budget and a deadline, and minorminer
seeded with the same roots under the same deadline. If ours matches minorminer from the
roots, it is the completion the learned embedder trains against and deploys with, and
minorminer returns to being a baseline.
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

from _context import qubit_budget
from placement_completion import witness_roots
from route_search import complete
from seeded_minorminer import attempt, valid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--deadline", type=float, default=30.0)
    a = ap.parse_args()
    tasks = load_instances(a.corpus)
    print(json.dumps({"corpus": a.corpus, "instances": len(tasks), "deadline": a.deadline}), flush=True)
    cells = defaultdict(lambda: defaultdict(list))
    for k, task in enumerate(tasks):
        family = task.lineage.rsplit("-", 1)[0].split("-", 1)[1].rsplit("-", 1)[0]
        witness = {v: frozenset(c) for v, c in task.witness.items()}
        roots = witness_roots(task, witness)
        rng = np.random.default_rng(k)
        variables = sorted(witness, key=str)
        half = set(rng.choice(variables, size=len(variables) // 2, replace=False))
        budget = qubit_budget(witness)
        row = []
        for tag, hint in (("witness", roots), ("half", {v: c for v, c in roots.items() if v in half})):
            # ours: unhinted variables get a root from the free qubits adjacent to a placed
            # neighbour, most-touching first, before the search; a placement rule, not learned
            chains = dict(hint)
            occupied = {q for c in chains.values() for q in c}
            for v in sorted((v for v in variables if v not in chains),
                            key=lambda v: (-sum(1 for u in task.logical.neighbors(v) if u in chains), str(v))):
                best, score = None, -1
                for u in task.logical.neighbors(v):
                    for q in chains.get(u, ()):
                        for r in task.host.neighbors(q):
                            if r in occupied:
                                continue
                            t = sum(1 for w in task.logical.neighbors(v) for x in chains.get(w, ()) if task.host.has_edge(r, x))
                            if t > score:
                                best, score = r, t
                if best is None:
                    best = next((q for q in sorted(task.host.nodes(), key=str) if q not in occupied), None)
                if best is None:
                    break
                chains[v] = frozenset({best}); occupied.add(best)
            t0 = time.time()
            ours = complete(task.host, task.logical, chains, budget, deadline=a.deadline) if len(chains) == len(variables) else None
            ours_ok = ours is not None and valid(ours, task.logical, task.host)
            ours_secs = time.time() - t0
            t0, mm, n = time.time(), None, 0
            while mm is None and time.time() - t0 < a.deadline:
                n += 1
                mm = attempt(task, hint, 61_000 + 1000 * k + n, 10)
            mm_secs = time.time() - t0
            cells[family][tag].append((ours_ok, ours_secs, mm is not None, mm_secs))
            row.append("%s: ours %s %5.1fs mm %s %5.1fs" % (tag, "ok" if ours_ok else "no", ours_secs, "ok" if mm else "no", mm_secs))
        print("  %3d/%d %-26s %s" % (k + 1, len(tasks), task.name, " | ".join(row)), flush=True)
    print("\n  %-16s %3s %10s %10s %10s %10s" % ("cell", "n", "ours/wit", "mm/wit", "ours/half", "mm/half"))
    for family in sorted(cells):
        c = cells[family]
        print("  %-16s %3d %10.2f %10.2f %10.2f %10.2f"
              % (family, len(c["witness"]), np.mean([r[0] for r in c["witness"]]), np.mean([r[2] for r in c["witness"]]),
                 np.mean([r[0] for r in c["half"]]), np.mean([r[2] for r in c["half"]])))
    print("\nOURS COMPLETION DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
