#!/usr/bin/env python3
"""Destroy-and-repair on top of stock minorminer with the surrogate in the loop, in the
benchmark's own rebuild loop (no search harness): which repair chooser and which destroy
chooser give the most solve probability at an equal evaluation budget?

Arms (all accept a rebuilt embedding when the surrogate does not drop):
  exact_repair     every candidate qubit pre-checked by exact completion, greedy among them
  model_repair     the structural model decides without the exact filter (dead end = skipped)
  greedy_repair    the greedy rule without the filter
  random_repair    random qubit without the filter
  learned_destroy  model repair + a fitted predictor chooses which variable to destroy (--destroy-model)

    python3 scripts/lns_bench.py --model /path/to/structural_model.pt --topology chimera --size 5 --source app --n 20 --budget 40 --out runs/lns/c5_app_b40.jsonl
"""
import argparse, json, random, time
import numpy as np
from embedbench.evaluate import EvalConfig, instances, stock_minorminer, p_solve
from embedbench.rebuild import lns, model_rebuild_chooser, greedy_rebuild_chooser, random_rebuild_chooser
from embedbench.models_structural import load_model, Scorer
from embedbench.objective import outcome_of


def var_features(logical, problem, chains, host, v):
    """Same nine per-variable features as the harness-side neighbourhood predictor."""
    ch = chains[v]; nb = list(logical.neighbors(v))
    used = {q for c in chains.values() for q in c}
    js = [abs(problem.j.get((min(v, u), max(v, u)), 0.0)) for u in nb]
    contacts = [sum(1 for a in ch for b in chains[u] if host.has_edge(a, b)) for u in nb]
    free_share = sum(1 for q in ch if any(n not in used for n in host.neighbors(q))) / max(1, len(ch))
    return [len(ch), len(nb), abs(problem.h.get(v, 0.0)), sum(js), max(js, default=0.0), min(contacts, default=0),
            sum(contacts) / max(1, len(contacts)), free_share, sum(len(chains[u]) for u in nb) / max(1, len(nb))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--destroy-model", default=None)
    ap.add_argument("--topology", default="chimera"); ap.add_argument("--size", type=int, default=5); ap.add_argument("--source", default="app")
    ap.add_argument("--n-vars", type=int, default=22); ap.add_argument("--degree", type=float, default=3.0); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--budget", type=int, default=40); ap.add_argument("--rounds", type=int, default=None); ap.add_argument("--l-cap", type=int, default=4); ap.add_argument("--max-free", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0); ap.add_argument("--arms", type=lambda v: v.split(","), default=None)
    ap.add_argument("--reads", type=int, default=100); ap.add_argument("--final-reads", type=int, default=400); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    scorer = Scorer(load_model(a.model)); rounds = a.rounds or a.budget * 10
    dmodel = None
    if a.destroy_model:
        import joblib; dmodel = joblib.load(a.destroy_model)
    cfg = EvalConfig(topology=a.topology, size=a.size, source=a.source, n_vars=a.n_vars, degree=a.degree, n=a.n, seed=a.seed)
    fh = open(a.out, "w"); t0 = time.time()
    for name, host, logical, problem, e0, gseed in instances(cfg):
        start = stock_minorminer(logical, host, gseed)
        if not start: continue
        start = {v: frozenset(c) for v, c in start.items() if v in logical}
        ev = lambda ch, s=gseed: p_solve(problem, host, ch, e0, a.reads, s + 11)[0]
        fin = lambda ch, s=gseed: p_solve(problem, host, ch, e0, a.final_reads, s + 1)[0]
        row = {"instance": name, "n_vars": logical.number_of_nodes(), "start": {"p_solve": fin(start), "Q": sum(len(c) for c in start.values())}, "arms": {}}
        def sampled_chooser(T=a.temperature, seed=gseed):
            rng_ = random.Random(seed)
            def choose(rec, wh, wl, cores, actions):
                sc = np.array(scorer(rec)); w = np.exp((sc - sc.max()) / T); w /= w.sum()
                return actions[rng_.choices(range(len(actions)), weights=w)[0]][1]
            return choose
        arms = [("exact_repair", greedy_rebuild_chooser(), True, None), ("model_repair", model_rebuild_chooser(scorer), False, None),
                ("model_sampled", sampled_chooser(), False, None),
                ("greedy_repair", greedy_rebuild_chooser(), False, None), ("random_repair", random_rebuild_chooser(gseed), False, None)]
        if a.arms: arms = [x for x in arms if x[0] in a.arms]
        if dmodel is not None:
            tried = set()
            def learned_destroy(chains, r, rng, tried=tried):
                allv = list(chains); pool = [v for v in allv if v not in tried] or allv
                if not pool or len(tried) >= len(allv): tried.clear(); pool = allv
                if rng.random() < 0.1: v = rng.choice(pool)
                else:
                    X = np.array([var_features(logical, problem, chains, host, v) for v in pool]); v = pool[int(np.argmax(dmodel.predict(X)))]
                tried.add(v); return v
            arms.append(("learned_destroy", model_rebuild_chooser(scorer), False, learned_destroy))
        for arm, chooser, filt, fc in arms:
            rng_focus = (lambda ch, r, rng: rng.choice(list(ch))) if fc is None else fc
            st = {}; t = time.time()
            ch, tr = lns(host, logical, start, chooser, ev, rounds=rounds, seed=gseed, l_cap=a.l_cap, max_free=a.max_free,
                         mode="evaluated", budget=a.budget, feasibility_filter=filt, focus_chooser=rng_focus, stats=st)
            assert outcome_of(ch, logical, host)[0] == 1
            row["arms"][arm] = {"p_solve": fin(ch), "Q": sum(len(c) for c in ch.values()), "evals": len(tr.p_history) - 1, "accepted": tr.accepted,
                                "rebuilt": tr.rebuilt, "skipped": tr.skipped, "rounds": tr.rounds, "nodes": st.get("nodes", 0), "seconds": round(time.time() - t, 1)}
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print(name, round(row["start"]["p_solve"], 3), {k: (round(v["p_solve"], 3), v["Q"], v["evals"], v["skipped"]) for k, v in row["arms"].items()}, flush=True)
    print("seconds", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
