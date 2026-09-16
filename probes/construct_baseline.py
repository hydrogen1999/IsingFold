"""The in-tree constructive heuristic on the fill-planted corpus.

router_initializer builds an embedding from nothing: variables in degree order, each placed and
routed by the same router the environment's PLACE and ROUTE proposals use. It is the floor a
learned constructor has to clear, and running it here answers a prior question: whether the
construction path works at all on a 680-qubit host at 80 to 95 percent fill, before any policy
is trained on it. minorminer's rate on the same instances is in the corpus generation log.
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
from isingfold.rl.proposal import router_initializer


def valid(chains, logical, host):
    if chains is None or any(not c for c in chains.values()):
        return False
    seen = set()
    for c in chains.values():
        if seen & set(c):
            return False
        seen |= set(c)
    for u, v in logical.edges():
        if not any(host.has_edge(a, b) for a in chains[u] for b in chains[v]):
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--attempts", type=int, default=8)
    a = ap.parse_args()
    tasks = load_instances(a.corpus)
    cap = tasks[0].host.number_of_nodes()
    init = router_initializer(attempts=a.attempts)
    print(json.dumps({"corpus": a.corpus, "instances": len(tasks), "qubits": cap,
                      "draws": a.draws, "attempts": a.attempts}), flush=True)
    cells = defaultdict(lambda: [0, 0, [], []])
    for k, task in enumerate(tasks):
        family = task.lineage.rsplit("-", 1)[0].split("-", 1)[1].rsplit("-", 1)[0]
        c = cells[family]
        for j in range(a.draws):
            t0 = time.time()
            try:
                ch = init(task.logical, task.host, 31_000 + 17 * j + 1000 * k)
            except Exception as e:
                ch = None
            c[3].append(time.time() - t0)
            c[1] += 1
            if valid(ch, task.logical, task.host):
                c[0] += 1
                c[2].append(sum(len(x) for x in ch.values()) / cap)
    print("\n  %-16s %3s %8s %8s %8s" % ("cell", "n", "valid", "fill", "secs"))
    for family in sorted(cells):
        v, n, fills, secs = cells[family]
        print("  %-16s %3d %8.2f %8s %8.1f"
              % (family, n, v / max(1, n), ("%.2f" % np.mean(fills)) if fills else "-",
                 np.mean(secs)), flush=True)
    print("\nCONSTRUCT BASELINE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
