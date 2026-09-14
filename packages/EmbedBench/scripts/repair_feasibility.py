#!/usr/bin/env python3
"""Does the structural model make an exact-quality feasibility judgement affordable?

Rebuild one chain (and re-complete a neighbour) in a window after a destroy, with four
choosers: the exact filter (every candidate qubit pre-checked by branch-and-bound, greedy
among the feasible ones: the oracle), and three unfiltered choosers that decide among all
adjacent free qubits and fail on a dead end: the structural model, the greedy rule, random.
Reports success rate, qubits used and exact search nodes per rebuild.

    python3 scripts/repair_feasibility.py --model /path/to/structural_model.pt --topology chimera --size 5 --n-vars 30 --degree 4 --n 30 --destroys 20 --out runs/repair/c5.jsonl
"""
import argparse, json, random, time
import networkx as nx, minorminer
from embedbench.structural import host_graph
from embedbench.objective import outcome_of
from embedbench.rebuild import rebuild_once, model_rebuild_chooser, greedy_rebuild_chooser, random_rebuild_chooser
from embedbench.models_structural import load_model, Scorer

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--topology", default="chimera"); ap.add_argument("--size", type=int, default=5)
ap.add_argument("--n-vars", type=int, default=30); ap.add_argument("--degree", type=float, default=4.0); ap.add_argument("--n", type=int, default=30)
ap.add_argument("--destroys", type=int, default=20); ap.add_argument("--l-cap", type=int, default=4); ap.add_argument("--max-free", type=int, default=16)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--out", required=True); ap.add_argument("--source", default="random")
a = ap.parse_args()
host = host_graph(a.topology, a.size)
scorer = Scorer(load_model(a.model))
rows = []
with open(a.out, "w") as f:
    for i in range(a.n):
        seed = a.seed * 1000 + i
        if a.source == "app":
            from embedbench.apps import graphcut_mrf, portfolio
            g = (portfolio(16, seed=i) if i % 3 == 0 else graphcut_mrf(5, 5, seed=i) if i % 3 == 1 else graphcut_mrf(4, 6, seed=i)).problem.graph
        else:
            g = nx.gnm_random_graph(a.n_vars, int(a.n_vars * a.degree / 2), seed=seed)
            if not nx.is_connected(g): continue
        emb = minorminer.find_embedding(list(g.edges()), list(host.edges()), random_seed=seed, tries=10)
        if not emb: continue
        chains = {k: frozenset(v) for k, v in emb.items()}
        q0 = sum(len(c) for c in chains.values())
        for d in range(a.destroys):
            focus = random.Random(seed + d).choice(list(g.nodes()))
            row = {"instance": i, "seed": seed, "n_vars": a.n_vars, "q0": q0, "destroy": d, "focus": focus, "arms": {}}
            for name, chooser, filt in (("exact_greedy", greedy_rebuild_chooser(), True), ("model", model_rebuild_chooser(scorer), False),
                                        ("model_fallback", model_rebuild_chooser(scorer), False),
                                        ("greedy", greedy_rebuild_chooser(), False), ("random", random_rebuild_chooser(seed + d), False)):
                st = {}; t = time.time()
                try:
                    new = rebuild_once(host, g, chains, focus, chooser, random.Random(seed + d), l_cap=a.l_cap, max_free=a.max_free,
                                       feasibility_filter=filt, stats=st)
                    if name == "model_fallback" and new is None:  # model first, exact filter only on a dead end: matched final success
                        st["fallback"] = 1
                        new = rebuild_once(host, g, chains, focus, greedy_rebuild_chooser(), random.Random(seed + d), l_cap=a.l_cap, max_free=a.max_free,
                                           feasibility_filter=True, stats=st)
                except Exception as e:  # noqa
                    new = None; st["error"] = str(e)[:80]
                ok = new is not None and outcome_of(new, g, host)[0] == 1
                row["arms"][name] = {"ok": bool(ok), "q": (sum(len(c) for c in new.values()) if ok else None), "nodes": st.get("nodes", 0), "sec": round(time.time() - t, 4), "fallback": st.get("fallback", 0)}
            f.write(json.dumps(row) + "\n"); f.flush(); rows.append(row)
        print(f"instance {i} done, {len(rows)} rows", flush=True)
arms = ["exact_greedy", "model", "model_fallback", "greedy", "random"]
print(f"{len(rows)} rebuilds")
for k in arms:
    r = [x["arms"][k] for x in rows]
    ok = [y["ok"] for y in r]; q = [y["q"] - x["q0"] for x, y in zip(rows, r) if y["ok"]]
    print(f"{k:14s} success {sum(ok)/len(ok):.3f}  dQ {sum(q)/max(1,len(q)):+.2f}  nodes/rebuild {sum(y['nodes'] for y in r)/len(r):.0f}  sec/rebuild {sum(y['sec'] for y in r)/len(r):.4f}  fallbacks {sum(y.get('fallback',0) for y in r)}")
