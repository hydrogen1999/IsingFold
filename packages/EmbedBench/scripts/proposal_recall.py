#!/usr/bin/env python3
"""Proposal recall of the search's candidate generator: for quality records with the full
embedding (v1.1) and high-read labels, rebuild the state, ask the search for k proposals for
the focus variable (resource-ranked prefix, k = 8, 16, 32, 64), and report how often the
high-read best candidate, and any candidate within 0.02 of it, is among the proposals; also
the best achievable p_solve within the proposal set against the best over all enumerated
candidates (the recall gap).
    python3 scripts/proposal_recall.py runs/release_v1_1/quality_pegasus3_app.jsonl --hiread runs/release_v1_1/hiread_test_labels.jsonl --k 8,16,32,64
"""
import argparse, json, os
import numpy as np, networkx as nx
from embedbench.embedding import LogicalProblem
from embedbench.structural import host_graph
from lac_minorminer.session import SearchSession

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--hiread", required=True); ap.add_argument("--k", default="8,16,32,64"); ap.add_argument("--limit", type=int, default=60); a = ap.parse_args()
ks = [int(x) for x in a.k.split(",")]
hi = {}
for ln in open(a.hiread):
    h = json.loads(ln); hi[(os.path.basename(h["file"]), str(h["instance_id"]), int(h["focus"]))] = h["high_read_scores"]
for f in a.files:
    name = os.path.basename(f); stats = {k: {"n": 0, "hit_best": 0, "hit_within": 0, "gap": [], "n_prop": [], "len_best": [], "len_prop": []} for k in ks}
    for ln in open(f):
        r = json.loads(ln); key = (name, str(r["instance_id"]), int(r["focus"]))
        if key not in hi or "all_chains" not in r: continue
        scores = hi[key]; cands = [frozenset(c) for c in r["candidates"]]; best = int(np.argmax(scores)); pbest = scores[best]
        host = host_graph(r["topology"], r["size"]); pr = r["problem"]
        problem = LogicalProblem.from_dicts({int(k): v for k, v in pr["h"].items()}, {(u, v): w for u, v, w in pr["J"]})
        logical = nx.Graph(); logical.add_nodes_from(int(v) for v in r["all_chains"]); logical.add_node(int(r["focus"])); logical.add_edges_from(problem.graph.edges())
        focus = int(r["focus"])
        if logical.degree(focus) == 0: continue
        chains = {int(v): frozenset(c) for v, c in r["all_chains"].items()}
        chains[focus] = cands[r["original_index"]] if r.get("original_index", -1) >= 0 else cands[0]
        if set(chains) != set(logical.nodes()): continue
        for k in ks:
            probe = SearchSession(list(logical.edges()), list(host.edges()), random_seed=1, max_candidates=k)
            src, tgt = probe.normalized_source, probe.normalized_target
            try:
                sess = SearchSession.from_chains(list(logical.edges()), list(host.edges()), [[tgt.index(q) for q in chains[l]] for l in src.labels], random_seed=1, max_candidates=k)
                b = sess.fork(random_seed=1); batch = b.propose(src.index(focus))
                props = [frozenset(tgt.labels[t] for t in c.chain) for c in batch.candidates if c.rank.max_occupancy <= 1]
                b.discard(batch)
            except Exception:
                continue
            st = stats[k]; st["n"] += 1; st["n_prop"].append(len(props))
            idx = [i for i, c in enumerate(cands) if c in set(props)]
            st["hit_best"] += (best in idx); st["hit_within"] += any(scores[i] >= pbest - 0.02 for i in idx)
            st["gap"].append(pbest - (max(scores[i] for i in idx) if idx else min(scores)))
            st["len_best"].append(len(cands[best])); st["len_prop"].append(np.mean([len(c) for c in props]) if props else 0)
    for k in ks:
        st = stats[k]
        if st["n"]: print(f"{name:30s} k={k:3d} n={st['n']:3d} proposals {np.mean(st['n_prop']):.1f} | best in proposals {st['hit_best']/st['n']:.2f} | within 0.02 {st['hit_within']/st['n']:.2f} | recall gap (p_best - best in proposals) {np.mean(st['gap']):.4f} | mean len best {np.mean(st['len_best']):.2f} vs proposals {np.mean(st['len_prop']):.2f}")
