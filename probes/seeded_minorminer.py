"""Does a placement hint make minorminer succeed where it fails from scratch?

The constructive policy places well and cannot finish; minorminer finishes well and cannot
find a placement at high fill. minorminer accepts initial chains. This seeds it with one
qubit per variable and measures validity within a deadline, restarting with fresh seeds:

    none       no hint, the plain anytime baseline
    witness    one qubit per variable from the witness chain (a perfect placement prior)
    random     one free qubit per variable at random (a control for the act of seeding)

If the witness arm is high where the plain arm is zero, a learned placement prior with
minorminer's search as the completion is a deployable hybrid, and the learning target is
the roots, one decision per variable.
"""
import argparse, json, os, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import minorminer
import numpy as np
from isingfold.rl.data.generate import load_instances

from placement_completion import witness_roots


def valid(chains, logical, host):
    if not chains or any(not c for c in chains.values()) or set(chains) != set(logical.nodes()):
        return False
    seen = set()
    for c in chains.values():
        if seen & set(c):
            return False
        seen |= set(c)
    return all(any(host.has_edge(a, b) for a in chains[u] for b in chains[v]) for u, v in logical.edges())


def attempt(task, hint, seed, tries):
    """minorminer takes an edge list, so isolated logical nodes get no chain from it; they
    are placed afterwards on any free qubit, and the hint is restricted to nodes it knows."""
    edges = list(task.logical.edges())
    in_edges = {u for e in edges for u in e}
    kw = {"tries": tries, "random_seed": seed % (2 ** 31)}
    if hint:
        kw["initial_chains"] = {v: list(c) for v, c in hint.items() if v in in_edges}
    emb = minorminer.find_embedding(edges, list(task.host.edges()), **kw)
    if not emb:
        return None
    emb = {v: frozenset(c) for v, c in emb.items()}
    used = {q for c in emb.values() for q in c}
    free = iter(sorted((q for q in task.host.nodes() if q not in used), key=str))
    for v in task.logical.nodes():
        if v not in emb:
            try:
                emb[v] = frozenset({next(free)})
            except StopIteration:
                return None
    return emb if valid(emb, task.logical, task.host) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--deadline", type=float, default=120.0)
    ap.add_argument("--tries", type=int, default=10)
    a = ap.parse_args()
    tasks = load_instances(a.corpus)
    cap = tasks[0].host.number_of_nodes()
    print(json.dumps({"corpus": a.corpus, "instances": len(tasks), "deadline": a.deadline,
                      "tries": a.tries}), flush=True)
    cells = defaultdict(lambda: defaultdict(list))
    for k, task in enumerate(tasks):
        family = task.lineage.rsplit("-", 1)[0].split("-", 1)[1].rsplit("-", 1)[0]
        witness = {v: frozenset(c) for v, c in task.witness.items()}
        rng = np.random.default_rng(k)
        free = list(task.host.nodes())
        hints = {"none": None, "witness": witness_roots(task, witness),
                 "random": {v: frozenset({free[i]}) for v, i in
                            zip(witness, rng.choice(len(free), size=len(witness), replace=False))}}
        row = []
        for arm, hint in hints.items():
            t0, ok, n = time.time(), None, 0
            while ok is None and time.time() - t0 < a.deadline:
                n += 1
                ok = attempt(task, hint, 50_000 + 1000 * k + n, a.tries)
            secs = time.time() - t0
            cells[family][arm].append((ok is not None, secs, n,
                                       (sum(len(c) for c in ok.values()) / cap) if ok else None))
            row.append("%s %s %4.0fs" % (arm, "valid" if ok else "no   ", secs))
        print("  %3d/%d %-26s %s" % (k + 1, len(tasks), task.name, " | ".join(row)), flush=True)
    print("\n  %-16s %3s %8s %8s %8s %10s %10s %10s" % ("cell", "n", "none", "witness", "random",
                                                         "secs none", "secs wit", "fill wit"))
    for family in sorted(cells):
        c = cells[family]
        n = len(c["none"])
        fills = [r[3] for r in c["witness"] if r[3] is not None]
        print("  %-16s %3d %8.2f %8.2f %8.2f %10.0f %10.0f %10s"
              % (family, n, np.mean([r[0] for r in c["none"]]), np.mean([r[0] for r in c["witness"]]),
                 np.mean([r[0] for r in c["random"]]), np.mean([r[1] for r in c["none"]]),
                 np.mean([r[1] for r in c["witness"]]), ("%.2f" % np.mean(fills)) if fills else "-"))
    print("\nSEEDED MINORMINER DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
