#!/usr/bin/env python3
"""Independent re-certification of structural decision samples (the benchmark's label audit).

Three checks per record, none of which reuses the branch-and-bound search that produced the
label:

1. brute force: enumerate connected supersets per variable independently and take the
   product (the reference used in the unit tests), whenever the product is below a bound;
   the outcome must equal the stored value for every action;
2. witness and best completions: the stored best action must have a feasible value and the
   witness chains must complete in the window (feasibility of the sample);
3. one-sided minorminer check: for every action labelled infeasible, run stock minorminer on
   the window with the cores plus the action fixed; if it returns chains within l_cap (and
   the qubit cap, when one applies) that is a contradiction of the exact label.

    python3 probes/recertify_decisions.py runs/datagen/*.jsonl --max-product 2e6 --tries 5
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time

import networkx as nx

from embedbench.exact import connected_supersets
from embedbench.objective import INFEASIBLE, outcome_of


def brute(host, logical, cores, l_cap, max_product, must_hit=None):
    free = set(host.nodes) - {q for c in cores.values() for q in c}
    per = []
    total = 1
    for v in logical.nodes():
        reqs = (must_hit or {}).get(v, [])
        opts = [s for s in connected_supersets(host, cores.get(v, frozenset()), free, l_cap) if all(s & r for r in reqs)]
        per.append(opts)
        total *= max(1, len(opts))
        if total > max_product:
            return None
    best = INFEASIBLE
    for combo in itertools.product(*per):
        best = max(best, outcome_of(dict(zip(logical.nodes(), combo)), logical, host))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--max-product", type=float, default=2e6)
    ap.add_argument("--tries", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="records per file")
    a = ap.parse_args()
    try:
        import minorminer
    except ImportError:
        minorminer = None
        print("minorminer not importable; check 3 skipped", file=sys.stderr)
    grand = dict(records=0, mm_invalid_output=0, brute_checked=0, brute_mismatch=0, feas_bad=0, infeasible_actions=0,
                 mm_attempted=0, mm_contradictions=0, mm_found_over_cap=0)
    t0 = time.time()
    for f in a.files:
        st = dict(grand)
        for k in st: st[k] = 0
        for ln in itertools.islice(open(f), a.limit):
            r = json.loads(ln)
            st["records"] += 1
            wh = nx.Graph(); wh.add_nodes_from(r["window_nodes"]); wh.add_edges_from(map(tuple, r["window_edges"]))
            wl = nx.Graph(); wl.add_nodes_from(r["in_play"]); wl.add_edges_from(map(tuple, r["logical_edges"]))
            cores = {int(u): frozenset(c) for u, c in r["cores"].items()}
            l_cap = r["l_cap"]
            fa = {int(q): set(t) for q, t in r.get("frozen_adjacency", {}).items()}
            frozen = {int(u): set(c) for u, c in r["frozen"].items()}
            must_hit = {}
            for v, u in r.get("frozen_edges", []):
                must_hit.setdefault(v, []).append(frozenset(q for q, t in fa.items() if t & frozen[u]))
            best_val = r["values"][r["actions"].index(r["best_action"])]
            if best_val[0] != 1 or r["witness_outcome"][0] != 1 or tuple(best_val) != max(map(tuple, r["values"])):
                st["feas_bad"] += 1
            # capacity cap, recovered from the witness outcome when the file used one
            q_cap = None
            if "slack0" in f:
                q_cap = -r["witness_outcome"][1]
            for (v, q), val in zip(r["actions"], r["values"]):
                c2 = dict(cores); c2[v] = cores.get(v, frozenset()) | {q}
                b = brute(wh, wl, c2, l_cap, a.max_product, must_hit)
                if b is not None:
                    st["brute_checked"] += 1
                    if q_cap is not None and b[0] == 1 and -b[1] > q_cap:
                        b = INFEASIBLE  # brute ignores the cap; apply it
                    # with a cap, brute's best-under-cap needs a filtered enumeration; only
                    # compare when brute agrees or the uncapped best already fits the cap
                    if q_cap is None or b == INFEASIBLE or -b[1] <= q_cap:
                        if q_cap is not None and b == INFEASIBLE and val[0] == 1:
                            pass  # uncapped best exceeded cap but a capped completion may exist; skip
                        elif list(b) != val:
                            st["brute_mismatch"] += 1
                if val[0] == 0:
                    st["infeasible_actions"] += 1
                    if minorminer is not None:
                        st["mm_attempted"] += 1
                        # minorminer must solve the same problem as the label: the in-play
                        # variables plus every frozen neighbour as a fixed-chain variable, on
                        # the window plus the frozen chains' qubits (adjacency from the record)
                        fixed = {int(u): list(c) for u, c in c2.items() if c}
                        mm_edges = list(wl.edges())
                        host_edges = list(wh.edges())
                        for q_, ts in fa.items():
                            host_edges += [(q_, t) for t in ts]
                        present = {x for e_ in host_edges for x in e_}
                        for v_, u_ in r.get("frozen_edges", []):
                            if u_ in frozen:
                                fq_ = [q for q in sorted(frozen[u_]) if q in present or True]
                                # keep the fixed chain connected and every one of its qubits present
                                host_edges += [(fq_[i], fq_[i + 1]) for i in range(len(fq_) - 1)]
                                if len(fq_) == 1 and fq_[0] not in present:
                                    continue  # a frozen chain with no adjacency to the window cannot be reached; the label already says so
                                mm_edges.append((v_, u_))
                                fixed[int(u_)] = fq_
                        present = {x for e_ in host_edges for x in e_}
                        src_nodes = {x for e_ in mm_edges for x in e_}
                        fixed = {k_: c for k_, c in fixed.items() if k_ in src_nodes}
                        if any(q not in present for c in fixed.values() for q in c):
                            st["mm_error"] = st.get("mm_error", 0) + 1
                            continue
                        try:
                            emb = minorminer.find_embedding(
                                mm_edges, host_edges, fixed_chains=fixed, tries=a.tries,
                                random_seed=1, verbose=0,
                            )
                        except (RuntimeError, ValueError):
                            # minorminer refuses some fixed-chain inputs at initialisation;
                            # that is not evidence either way
                            st["mm_error"] = st.get("mm_error", 0) + 1
                            emb = {}
                        # minorminer returns fixed chains without checking that they realise
                        # the logical edges, so its output must be validated independently
                        chains = {int(k): frozenset(c) for k, c in emb.items() if int(k) in set(wl.nodes)} if emb else {}
                        valid = bool(chains) and set(chains) == set(wl.nodes) and outcome_of(chains, wl, wh)[0] == 1 \
                            and all(chains[v_] & mh for v_, reqs in must_hit.items() for mh in reqs)
                        if valid and all(len(c) <= l_cap for c in chains.values()) and (
                            q_cap is None or sum(len(c) for c in chains.values()) <= q_cap
                        ):
                            st["mm_contradictions"] += 1
                        elif valid:
                            st["mm_found_over_cap"] += 1
                        elif emb:
                            st["mm_invalid_output"] += 1
        print(f"{f.split('/')[-1]:26s} " + " ".join(f"{k}={v}" for k, v in st.items()))
        for k in st:
            grand[k] = grand.get(k, 0) + st[k]
    print("TOTAL " + " ".join(f"{k}={v}" for k, v in grand.items()) + f" seconds={time.time()-t0:.0f}")


if __name__ == "__main__":
    main()
