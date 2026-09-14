#!/usr/bin/env python3
"""Score externally produced embeddings on the dumped instances with the benchmark's own
scorer (validity, qubits, longest chain, p_solve at --reads with a fresh seed).
    python3 scripts/score_embeddings.py --instances runs/ember/c5_instances.json --embeddings runs/ember/c5_embeddings.jsonl --reads 800 --out runs/ember/c5_scored.jsonl"""
import argparse, json
import networkx as nx
from embedbench.evaluate import p_solve, host_graph
from embedbench.embedding import LogicalProblem
from embedbench.objective import outcome_of
ap = argparse.ArgumentParser(); ap.add_argument("--instances", required=True); ap.add_argument("--embeddings", required=True); ap.add_argument("--reads", type=int, default=800); ap.add_argument("--objective", default="psolve"); ap.add_argument("--out", required=True); a = ap.parse_args()
insts = {r["instance"]: r for r in json.load(open(a.instances))}; hosts = {}; fh = open(a.out, "w")
for l in open(a.embeddings):
    e = json.loads(l); r = insts[e["instance"]]; key = (r["topology"], r["size"])
    if key not in hosts: hosts[key] = host_graph(*key)
    host = hosts[key]; g = nx.Graph(); g.add_nodes_from(r["nodes"]); g.add_edges_from(r["edges"])
    problem = LogicalProblem.from_dicts({int(k): v for k, v in r["problem"]["h"].items()}, {(u, v): w for u, v, w in r["problem"]["J"]})
    ch = {int(k): frozenset(v) for k, v in e["embedding"].items()}
    ok = bool(ch) and set(ch) == set(g.nodes()) and all(c and all(q in host for q in c) for c in ch.values()) and outcome_of(ch, g, host)[0] == 1
    d = {"instance": e["instance"], "algorithm": e["algorithm"], "feasible": ok, "seconds": e["seconds"], "error": e.get("error")}
    if ok:
        p, f = p_solve(problem, host, ch, r["problem"]["e0"], a.reads, r["seed"] + 1, 5, a.objective)
        d.update(Q=sum(len(c) for c in ch.values()), L=max(len(c) for c in ch.values()), p_solve=p, F=f)
    fh.write(json.dumps(d) + "\n"); fh.flush(); print(d["instance"], d["algorithm"], "feasible" if ok else "INFEASIBLE", d.get("p_solve"), d.get("Q"), flush=True)
