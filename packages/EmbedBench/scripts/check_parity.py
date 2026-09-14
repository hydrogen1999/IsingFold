#!/usr/bin/env python3
"""Train/deploy parity for the chain scorer: for corpus records that carry the full problem and
embedding (v1.1), rebuild the record the deployment seam would build for the SAME state and
the SAME candidates, and compare: window size, frozen set, neighbour fields, and the scorer's
ranking of the candidates on the two records (Spearman, top-1 agreement).
    python3 scripts/check_parity.py runs/release_v1_1/quality_pegasus3_app.jsonl --model /path/to/chain_model.pt --limit 50
"""
import argparse, json
import numpy as np
from scipy.stats import spearmanr
from embedbench.embedding import LogicalProblem
from embedbench.structural import host_graph
from embedbench.models_chain import load_chain_model, ChainScorer
from embedbench.seam import LearnedTieBreakScorer
from lac_minorminer.session import SearchSession

class _Cand:
    def __init__(self, chain): self.chain = chain
class _Batch:
    def __init__(self, logical, cands): self.logical = logical; self.candidates = cands

ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--model", required=True); ap.add_argument("--limit", type=int, default=50); a = ap.parse_args()
scorer = ChainScorer(load_chain_model(a.model)); rows = []
for f in a.files:
    n = 0
    for ln in open(f):
        r = json.loads(ln)
        if "problem" not in r or "all_chains" not in r: continue
        host = host_graph(r["topology"], r["size"]); pr = r["problem"]
        problem = LogicalProblem.from_dicts({int(k): v for k, v in pr["h"].items()}, {(u, v): w for u, v, w in pr["J"]})
        import networkx as nx
        logical = nx.Graph(); logical.add_nodes_from(int(v) for v in r["all_chains"]); logical.add_node(int(r["focus"])); logical.add_edges_from(problem.graph.edges()); focus = int(r["focus"])
        if logical.degree(focus) == 0: continue
        chains = {int(v): frozenset(c) for v, c in r["all_chains"].items()}; chains[focus] = frozenset(r["candidates"][r["original_index"]] if r.get("original_index", -1) >= 0 else r["candidates"][0])
        if set(chains) != set(logical.nodes()): continue
        probe = SearchSession(list(logical.edges()), list(host.edges()), random_seed=1, max_candidates=8)
        src, tgt = probe.normalized_source, probe.normalized_target
        sess = SearchSession.from_chains(list(logical.edges()), list(host.edges()), [[tgt.index(q) for q in chains[l]] for l in src.labels], random_seed=1, max_candidates=8)
        seam = LearnedTieBreakScorer(problem, logical, host, sess, scorer)
        snap = sess.snapshot(); li = src.index(focus)
        batch = _Batch(li, [_Cand([tgt.index(q) for q in c]) for c in r["candidates"]])
        dep = seam._record(snap, batch, list(range(len(r["candidates"]))))
        s_corpus = np.array(scorer(r)); s_dep = np.array(scorer(dep))
        rho = spearmanr(s_corpus, s_dep).correlation if len(s_corpus) > 2 else float("nan")
        rows.append({"file": f, "n_cands": len(r["candidates"]), "win_corpus": len(r["window_nodes"]), "win_dep": len(dep["window_nodes"]),
                     "frozen_corpus": len(r["frozen"]), "frozen_dep": len(dep["frozen"]), "nbrs_equal": r["neighbours"] == dep["neighbours"],
                     "edgeJ_equal": np.allclose(r["edge_J"], dep["edge_J"]) if r["neighbours"] == dep["neighbours"] else False,
                     "rho": rho, "top1_same": int(np.argmax(s_corpus) == np.argmax(s_dep)),
                     "corpus_top_is_best": int(np.argmax(s_corpus) == r["best_index"]), "dep_top_is_best": int(np.argmax(s_dep) == r["best_index"])})
        n += 1
        if n >= a.limit: break
print(len(rows), "records")
if rows:
    print("window size corpus vs deploy: mean", round(np.mean([x["win_corpus"] for x in rows]), 1), "vs", round(np.mean([x["win_dep"] for x in rows]), 1))
    print("frozen entries corpus vs deploy:", round(np.mean([x["frozen_corpus"] for x in rows]), 1), "vs", round(np.mean([x["frozen_dep"] for x in rows]), 1))
    print("neighbours equal:", np.mean([x["nbrs_equal"] for x in rows]), "edge_J equal:", np.mean([x["edgeJ_equal"] for x in rows]))
    print("score rank correlation corpus vs deploy record: median", round(float(np.nanmedian([x["rho"] for x in rows])), 3), "top-1 same:", round(np.mean([x["top1_same"] for x in rows]), 3))
    print("scorer top-1 = corpus best: corpus record", round(np.mean([x["corpus_top_is_best"] for x in rows]), 3), "deploy record", round(np.mean([x["dep_top_is_best"] for x in rows]), 3))
