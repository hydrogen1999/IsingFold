#!/usr/bin/env python3
"""Dump the evaluation instances of an EvalConfig (logical graph, problem, host name) so an
external embedder can embed them; score the results with scripts/score_embeddings.py.
    python3 scripts/dump_instances.py --topology chimera --size 5 --source app --n 20 --seed 4243 --out runs/ember/c5_instances.json"""
import argparse, json
from embedbench.evaluate import EvalConfig, instances
ap = argparse.ArgumentParser(); ap.add_argument("--topology", required=True); ap.add_argument("--size", type=int, required=True); ap.add_argument("--source", default="app"); ap.add_argument("--n", type=int, default=20); ap.add_argument("--seed", type=int, required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
cfg = EvalConfig(topology=a.topology, size=a.size, source=a.source, n=a.n, seed=a.seed); out = []
for name, host, logical, problem, e0, gseed in instances(cfg):
    out.append({"instance": name, "topology": a.topology, "size": a.size, "seed": gseed, "nodes": [int(v) for v in logical.nodes()], "edges": [[int(u), int(v)] for u, v in logical.edges()],
                "problem": {"h": {str(k): float(v) for k, v in problem.h.items()}, "J": [[int(u), int(v), float(w)] for (u, v), w in problem.j.items()], "e0": float(e0)}})
json.dump(out, open(a.out, "w")); print(len(out), "instances")
