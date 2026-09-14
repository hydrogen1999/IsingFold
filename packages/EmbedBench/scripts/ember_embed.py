#!/usr/bin/env python3
"""Embed dumped instances with every algorithm in Ember's registry (runs in a venv with
ember-qc installed; no EmbedBench import). Writes one record per (instance, algorithm)
with the embedding, wall seconds and the algorithm's version string.
    python3 scripts/ember_embed.py --instances runs/ember/c5_instances.json --out runs/ember/c5_embeddings.jsonl [--algorithms minorminer,clique,pssa] [--timeout 120]"""
import argparse, json, time
import networkx as nx, dwave_networkx as dnx
import ember_qc.algorithms
from ember_qc.registry import ALGORITHM_REGISTRY
ap = argparse.ArgumentParser(); ap.add_argument("--instances", required=True); ap.add_argument("--out", required=True); ap.add_argument("--algorithms", default=None); ap.add_argument("--timeout", type=float, default=120.0); a = ap.parse_args()
insts = json.load(open(a.instances)); names = a.algorithms.split(",") if a.algorithms else list(ALGORITHM_REGISTRY)
hosts = {}
fh = open(a.out, "w")
for r in insts:
    key = (r["topology"], r["size"])
    if key not in hosts: hosts[key] = {"chimera": dnx.chimera_graph, "pegasus": dnx.pegasus_graph, "zephyr": dnx.zephyr_graph}[r["topology"]](r["size"])
    host = hosts[key]; g = nx.Graph(); g.add_nodes_from(r["nodes"]); g.add_edges_from(r["edges"])
    for name in names:
        alg = ALGORITHM_REGISTRY[name]; t0 = time.perf_counter(); err = None
        try:
            res = alg.embed(g.copy(), host, seed=r["seed"], timeout=a.timeout) or {}
            emb = res.get("embedding") or {}
        except Exception as e:
            emb = {}; err = f"{type(e).__name__}: {str(e)[:120]}"
        secs = time.perf_counter() - t0
        emb = {str(k): [int(q) for q in v] for k, v in emb.items()}
        fh.write(json.dumps({"instance": r["instance"], "algorithm": name, "version": (alg.version() if callable(getattr(alg, "version", None)) else None), "seconds": round(secs, 3), "embedding": emb, "error": err}) + "\n"); fh.flush()
        print(r["instance"], name, "ok" if emb else "FAIL", round(secs, 2), err or "", flush=True)
