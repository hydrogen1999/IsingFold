"""Where the objective stops discriminating: p_solve and energy residual against instance size.

The paper's objective is solve probability under the registered schedule. At the fill scale
(280 to 440 variables) every valid embedding reads 0.0000, so the record measures the mean
energy residual there instead. A reviewer needs the crossover: for planted instances of
increasing size at a fixed fill, measure the witness's p_solve and residual, and minorminer's
where it exists, under the registered schedule, on a fresh block each time. No learning here;
this is a property of the benchmark.
"""
# ruff: noqa: E402
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.environ.get("ISINGFOLD_SRC", str(Path(__file__).resolve().parents[1] / "src")))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]

import numpy as np

from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.evaluate import first_commit_controller, run_controller

from _context import host_context, qubit_budget
from _initializers import minorminer_initializer


def measure(task, chains, ctx, seed, reads):
    """p_solve and mean energy residual of one embedding under the registered schedule."""
    def fixed(l, h, s):
        return chains
    out = run_controller([task], ctx, first_commit_controller, initializer=fixed,
                         selector=fixed_strength_selector(), reward_reads=reads,
                         repetitions=1, seed=seed)
    o = out[0]
    if not o.returned_valid:
        return None
    return {"p_solve": float(o.utility) if o.utility is not None else None,
            "residual": float(o.mean_energy_residual) if o.mean_energy_residual is not None else None,
            "qubits": o.qubits, "max_chain": o.max_chain}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpora", required=True, help="comma-separated corpus directories, small to large")
    ap.add_argument("--cells", default="", help="comma-separated instance-name filters")
    ap.add_argument("--per-corpus", type=int, default=4)
    ap.add_argument("--reads", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minorminer-tries", type=int, default=20)
    a = ap.parse_args()
    if a.per_corpus < 1 or a.reads < 1:
        ap.error("per-corpus and reads must be positive")
    cells = [c for c in a.cells.split(",") if c]
    print(json.dumps({"probe": "objective-scale", "corpora": a.corpora.split(","), "cells": cells or None,
                      "per_corpus": a.per_corpus, "reads": a.reads}), flush=True)
    mm = minorminer_initializer(a.minorminer_tries)
    rows = []
    for path in a.corpora.split(","):
        tasks = [t for t in load_instances(path) if not cells or any(c in t.name for c in cells)]
        tasks = sorted(tasks, key=lambda t: t.name)[: a.per_corpus]
        for t in tasks:
            witness = {v: frozenset(c) for v, c in t.witness.items()} if t.witness else None
            if witness is None:
                continue
            ctx = host_context(max(qubit_budget(witness), len(t.host)))
            row = {"corpus": path, "instance": t.name, "variables": t.logical.number_of_nodes(),
                   "edges": t.logical.number_of_edges(), "host": len(t.host),
                   "fill": sum(len(c) for c in witness.values()) / len(t.host)}
            w = measure(t, witness, ctx, a.seed, a.reads)
            row["witness"] = w
            started = time.monotonic()
            chains = mm(t.logical, t.host, a.seed)
            row["minorminer_seconds"] = time.monotonic() - started
            row["minorminer"] = measure(t, chains, ctx, a.seed + 1, a.reads) if chains else None
            rows.append(row)
            print(json.dumps(row), flush=True)
    by_size = {}
    for r in rows:
        key = (r["corpus"], r["variables"] // 25 * 25)
        by_size.setdefault(key, []).append(r)
    print("  corpus                          vars  fill   witness p_solve  witness residual  mm p_solve  mm valid")
    for (corpus, size), group in sorted(by_size.items(), key=lambda kv: kv[0][1]):
        w = [g["witness"] for g in group if g["witness"]]
        m = [g["minorminer"] for g in group if g["minorminer"]]
        print("  %-30s %4d  %.2f   %14s  %16s  %10s  %d/%d"
              % (corpus.split("/")[-2][:30], size, float(np.mean([g["fill"] for g in group])),
                 ("%.4f" % np.mean([x["p_solve"] for x in w if x["p_solve"] is not None])) if w else "-",
                 ("%.4f" % np.mean([x["residual"] for x in w if x["residual"] is not None])) if w else "-",
                 ("%.4f" % np.mean([x["p_solve"] for x in m if x["p_solve"] is not None])) if m else "-",
                 len(m), len(group)), flush=True)
    print("OBJECTIVE SCALE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
