#!/usr/bin/env python3
"""Dump one instance's stock minorminer embedding and the released ObjectiveEmbedder's
result side by side (chains, problem, p_solve at the evaluation read count), for figures.
    python3 scripts/embed_pair.py --topology chimera --size 5 --seed 4243 --instance chimera5-portfolio16-14 --out pair.json
"""
import argparse, json
from embedbench.evaluate import EvalConfig, instances, stock_minorminer, p_solve
from embedbench.objective_embedder import ObjectiveEmbedder
ap = argparse.ArgumentParser(); ap.add_argument("--topology", required=True); ap.add_argument("--size", type=int, required=True); ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--instance", required=True); ap.add_argument("--budget", type=int, default=100); ap.add_argument("--reads", type=int, default=800); ap.add_argument("--out", required=True); a = ap.parse_args()
cfg = EvalConfig(topology=a.topology, size=a.size, source="app", n=20, reads=a.reads, seed=a.seed)
for name, host, logical, problem, e0, gseed in instances(cfg):
    if name != a.instance: continue
    stock = stock_minorminer(logical, host, gseed)
    emb = ObjectiveEmbedder(problem, e0, budget=a.budget)(logical, host, gseed)
    out = {"instance": name, "topology": a.topology, "size": a.size, "n_vars": logical.number_of_nodes(),
           "problem": {"h": {str(k): float(v) for k, v in problem.h.items()}, "J": [[int(u), int(v), float(w)] for (u, v), w in problem.j.items()], "e0": float(e0)}}
    for label, ch in (("stock", stock), ("embedder", emb)):
        p, f = p_solve(problem, host, ch, e0, a.reads, gseed + 1, cfg.n_strengths)
        out[label] = {"chains": {str(k): sorted(int(q) for q in c) for k, c in ch.items()}, "p_solve": float(p), "F": float(f), "Q": sum(len(c) for c in ch.values()), "L": max(len(c) for c in ch.values())}
        print(label, "p", round(p, 4), "Q", out[label]["Q"], "L", out[label]["L"], flush=True)
    json.dump(out, open(a.out, "w")); print("wrote", a.out); break
