"""Can a chain's break rate be predicted from its structure, before any sample is drawn?

Chain breaking is the strongest predictor of solution quality measured here, at a within-instance
rank correlation of 0.73 against 0.44 for qubit count, and it is not explained by chain length:
pruning forty percent of a policy's qubits cuts breaking by twelve percent. So something else
about a chain decides whether it holds, and the method question is whether the constructor can
see it at the moment it acts.

This separates two very different diagnoses. If a simple model predicts a chain's break rate from
structure the policy could compute while building, then the observation carries the signal and
the difficulty is credit assignment: the information is there and the learning rule cannot reach
it. If no such model works, the observation is missing the quantity that matters and no amount of
reward engineering will help.

The features are the ones the physics observation claims to supply, written out explicitly:
length, internal redundancy, how much logical coupling mass the chain carries, how that mass
concentrates on the contacts that realise it, and whether any single qubit is a cut vertex the
whole chain depends on. Breaking is decoded here rather than read from the evaluator, so nothing
on the registered path changes.
"""
import argparse, json, math, os, statistics, sys
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import networkx as nx
import numpy as np
import torch
from isingfold.rl.evaluator import sample_program
from isingfold.rl.program import compile_program, strength_registry

import constructor_curriculum as cc
from constructor_rollout import episode
from _context import REGISTERED_BETA_RANGE, host_context
from _initializers import minorminer_initializer

FEATURES = ("length", "internal_edges", "redundancy", "cut_vertices", "logical_degree",
            "coupling_mass", "contacts", "mass_per_contact", "max_contact_load",
            "load_concentration", "field")


def chain_features(task, chains, v):
    """What the policy could compute about one chain while it is building it."""
    chain = chains[v]
    sub = task.host.subgraph(chain)
    n = len(chain)
    internal = sub.number_of_edges()
    # A chain of n qubits needs n-1 edges to be connected; anything above that is redundancy.
    redundancy = internal - (n - 1)
    cuts = 0 if n <= 2 else len(list(nx.articulation_points(sub)))
    j = task.problem.j if hasattr(task.problem, "j") else {}
    mass, contacts, loads = 0.0, 0, []
    for u in task.logical.neighbors(v):
        w = abs(j.get((min(str(u), str(v)), max(str(u), str(v))), 0.0)) if isinstance(j, dict) else 0.0
        if not w:
            data = task.logical.get_edge_data(u, v) or {}
            w = abs(data.get("weight", 1.0))
        realised = sum(1 for a in chain for b in chains[u] if task.host.has_edge(a, b))
        if realised:
            contacts += realised
            mass += w
            loads.append(w / realised)
    return {"length": float(n), "internal_edges": float(internal), "redundancy": float(redundancy),
            "cut_vertices": float(cuts), "logical_degree": float(task.logical.degree(v)),
            "coupling_mass": mass, "contacts": float(contacts),
            "mass_per_contact": mass / contacts if contacts else 0.0,
            "max_contact_load": max(loads) if loads else 0.0,
            "load_concentration": (max(loads) / (sum(loads) / len(loads))) if loads else 0.0,
            "field": abs(float(getattr(task.problem, "h", {}).get(v, 0.0)))}


def break_rates(task, chains, ctx, index, reads, seed):
    """Fraction of reads in which each chain is not unanimous, decoded here."""
    strengths = strength_registry(task.problem, ctx.strength_ratios, ctx.epsilon_strength)
    program = compile_program(chains, task.host, task.problem, strengths[index], index)
    import dimod
    from dwave.samplers import SimulatedAnnealingSampler
    bqm = dimod.BinaryQuadraticModel({q: float(x) for q, x in program.h_phys.items()},
                                     {(a, b): float(w) for (a, b), w in program.j_phys.items()},
                                     0.0, dimod.SPIN)
    for node in sorted(chains, key=str):
        for q in sorted(chains[node], key=str):
            if q not in bqm.variables:
                bqm.add_variable(q, 0.0)
    ss = SimulatedAnnealingSampler().sample(bqm, num_reads=reads, seed=int(seed), num_sweeps=200,
                                            beta_range=list(REGISTERED_BETA_RANGE))
    order = {q: i for i, q in enumerate(ss.variables)}
    rows = np.asarray(ss.record.sample)
    out = {}
    for v, chain in chains.items():
        idx = [order[q] for q in chain if q in order]
        if len(idx) < 2:
            out[v] = 0.0
            continue
        block = rows[:, idx]
        out[v] = float(np.mean(~np.all(block == block[:, :1], axis=1)))
    return out


def ridge(X, y, lam=1.0):
    X = np.asarray(X, float); y = np.asarray(y, float)
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)
    return lambda A: (np.hstack([(np.asarray(A, float) - mu) / sd,
                                 np.ones((len(A), 1))]) @ w), w, mu, sd


def spearman(a, b):
    n = len(a)
    if n < 3:
        return None
    def rank(xs):
        o = sorted(range(n), key=lambda i: xs[i]); r = [0.] * n; i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[o[j + 1]] == xs[o[i]]:
                j += 1
            for k in range(i, j + 1):
                r[o[k]] = (i + j) / 2. + 1
            i = j + 1
        return r
    ra, rb = rank(a), rank(b); ma, mb = sum(ra) / n, sum(rb) / n
    den = math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
    return None if den == 0 else sum((x - ma) * (y - mb) for x, y in zip(ra, rb)) / den


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cells", default="")
    ap.add_argument("--init", required=True)
    ap.add_argument("--features", default="local")
    ap.add_argument("--actor", default="linear")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--support", default="wide")
    ap.add_argument("--n-tasks", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--episode-seconds", type=float, default=90.)
    ap.add_argument("--reads", type=int, default=2048)
    ap.add_argument("--strength-index", type=int, default=1)
    ap.add_argument("--mm-tries", type=int, default=20)
    ap.add_argument("--holdout", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20260930)
    a = ap.parse_args()

    cells = [c for c in a.cells.split(",") if c]
    tasks, _ = cc.build_corpus_sets(a.corpus, cells, a.n_tasks, 1, a.seed)
    actor = cc.make_actor(a.actor, a.width, cc.FEATURE_WIDTHS[a.features])
    cc.load_init(a.init, actor, a.actor, a.features)
    mm = minorminer_initializer(a.mm_tries)
    wide = a.support == "wide"

    print(json.dumps({"probe": "break_predictors", "corpus": a.corpus, "cells": cells,
                      "init": a.init, "tasks": len(tasks), "reads": a.reads,
                      "features": list(FEATURES)}), flush=True)

    samples = []  # (task index, feature row, measured break rate, source)
    for k, t in enumerate(tasks):
        ctx = host_context(t.host.number_of_nodes())
        fc = cc.make_features(a.features, t)
        sources = {}
        for e in range(a.episodes):
            with torch.no_grad():
                rec = episode(t, actor, fc, 1., a.max_steps,
                              np.random.default_rng(a.seed + 100 * k + e), a.episode_seconds,
                              train=True, objective="quality", evaluate_reward=False, wide=wide)
            if rec["valid"]:
                sources["policy%d" % e] = {v: frozenset(c) for v, c in rec["terminal"].embedding.items()}
        draw = mm(t.logical, t.host, a.seed + k)
        if draw:
            sources["minorminer"] = {v: frozenset(draw[v]) for v in t.logical.nodes()}
        for name, chains in sources.items():
            rates = break_rates(t, chains, ctx, a.strength_index, a.reads, a.seed + 7 * k)
            for v in chains:
                row = chain_features(t, chains, v)
                samples.append((k, [row[f] for f in FEATURES], rates[v], name))
        print(json.dumps({"task": t.name, "sources": sorted(sources), "chains": len(samples)}),
              flush=True)

    if len(samples) < 50:
        print(json.dumps({"summary": "break_predictors", "skipped": "only %d chains" % len(samples)}),
              flush=True)
        print("BREAK PREDICTORS DONE", flush=True)
        return 0

    # Every chain is logged so the analysis can be redone without drawing a single new sample.
    for k, row, rate, src in samples:
        print(json.dumps({"chain": {"task_index": k, "source": src, "break_rate": rate,
                                    **dict(zip(FEATURES, row))}}), flush=True)

    # Singleton chains are excluded from the score. A chain of one qubit cannot break, so its
    # rate is exactly zero by construction, and at this cell most chains are singletons: leaving
    # them in makes any feature that separates singletons from the rest look like a predictor of
    # breaking, which is how length alone appeared to beat the full feature set.
    length_at = FEATURES.index("length")
    multi = [x for x in samples if x[1][length_at] >= 2]
    print(json.dumps({"chains_total": len(samples), "chains_multi_qubit": len(multi),
                      "singleton_fraction": 1 - len(multi) / len(samples)}), flush=True)
    samples = multi
    if len(samples) < 50:
        print(json.dumps({"summary": "break_predictors",
                          "skipped": "only %d multi-qubit chains" % len(samples)}), flush=True)
        print("BREAK PREDICTORS DONE", flush=True)
        return 0

    holdout = set(range(len(tasks) - a.holdout, len(tasks)))
    tr = [s for s in samples if s[0] not in holdout]
    te = [s for s in samples if s[0] in holdout]
    predict, w, _, _ = ridge([s[1] for s in tr], [s[2] for s in tr])
    pred = predict([s[1] for s in te])
    truth = [s[2] for s in te]
    rho = spearman(list(pred), truth)
    base = spearman([s[1][FEATURES.index("length")] for s in te], truth)
    print()
    print("  chains: %d train on %d instances, %d held out on %d instances"
          % (len(tr), len(tasks) - a.holdout, len(te), a.holdout), flush=True)
    print("  %-34s %s" % ("held-out rank correlation, structure", "%+.3f" % rho if rho else "-"),
          flush=True)
    print("  %-34s %s" % ("held-out rank correlation, length alone",
                          "%+.3f" % base if base else "-"), flush=True)
    print()
    print("  %-22s %8s" % ("feature", "weight"), flush=True)
    for f, wi in sorted(zip(FEATURES, w[:-1]), key=lambda x: -abs(x[1])):
        print("  %-22s %8.4f" % (f, wi), flush=True)
    print(json.dumps({"summary": "break_predictors", "chains_train": len(tr),
                      "chains_heldout": len(te), "rho_structure": rho, "rho_length_alone": base,
                      "weights": dict(zip(FEATURES, [float(x) for x in w[:-1]])),
                      "scored_on": "multi-qubit chains only; singletons cannot break",
                      "reading": ("a held-out rank correlation well above the length-alone "
                                  "baseline says the observation carries what decides breaking "
                                  "and the difficulty is credit assignment; near or below it "
                                  "says the observation is missing the quantity")}), flush=True)
    print("BREAK PREDICTORS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
