#!/usr/bin/env python3
"""Reference baselines on the release's test split. Structural files: top-1 on certified-best
actions for the local greedy rule, the resource rule, random, and the released structural
model(s). Quality files: regret (against the best reliable candidate) and top-1 within 0.02
for the fewest-qubit chain, the original chain, random, and the released chain scorer(s).
    python3 scripts/release_baselines.py --splits runs/release_v1_1/splits.json --dir runs/release_v1_1 --struct-models a.pt,b.pt --chain-models c.pt,d.pt --out runs/release_v1_1/baselines.json
"""
import argparse, json, os, random, statistics
import numpy as np

def structural_eval(recs, scorers):
    out = {}
    def top1(pick):
        return sum(1 for r in recs if tuple(r["values"][pick(r)]) == max(tuple(v) for v in r["values"])) / len(recs)
    out["local_greedy"] = top1(lambda r: r["actions"].index(r["greedy_action"]))
    if all("resource_action" in r for r in recs): out["resource"] = top1(lambda r: r["actions"].index(r["resource_action"]))
    rng = random.Random(0); out["random"] = top1(lambda r: rng.randrange(len(r["actions"])))
    for name, sc in scorers.items():
        out[name] = top1(lambda r, sc=sc: int(np.argmax(sc(r))))
    return out

def quality_eval(recs, scorers, hiread=None):
    """With `hiread` ({(file, instance_id, focus): high-read scores}), regret is measured against the
    high-read labels (the picker still sees only the corpus record); records without high-read
    labels are skipped."""
    out = {}
    if hiread is not None:
        def key(r):
            return (r["_file"], str(r["instance_id"]), int(r["focus"]))
        recs = [r for r in recs if key(r) in hiread and len(hiread[key(r)]) == len(r["p_solve"])]
        if not recs: return {"n_hiread": 0}
        out["n_hiread"] = len(recs)
    def truth(r):
        return hiread[key(r)] if hiread is not None else r["p_solve"]
    def stats(pick):
        reg = []; top = 0
        for r in recs:
            rel = [i for i in range(len(r["p_solve"])) if r.get("stage", [1] * len(r["p_solve"]))[i] == 2] or list(range(len(r["p_solve"])))
            t = truth(r); best = max(t); k = pick(r, rel)
            reg.append(best - t[k]); top += t[k] >= best - 0.02
        return {"regret": statistics.mean(reg), "top1_within_0.02": top / len(recs)}
    if hiread is not None:
        out["corpus_best"] = stats(lambda r, rel: r["best_index"])
    out["resource"] = stats(lambda r, rel: r["resource_index"])
    orig = [r for r in recs if r.get("original_index", -1) >= 0]
    if orig: out["original_chain"] = stats(lambda r, rel: r["original_index"] if r.get("original_index", -1) >= 0 else r["resource_index"])
    rng = random.Random(0); out["random"] = stats(lambda r, rel: rng.choice(rel))
    for name, sc in scorers.items():
        out[name] = stats(lambda r, rel, sc=sc: max(rel, key=lambda i, s=sc(r): s[i]))
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--splits", required=True); ap.add_argument("--dir", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--struct-models", default=""); ap.add_argument("--chain-models", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--hiread", default=None, help="jsonl of high-read rescored records (rescore_quality.py); quality regret is then measured against these labels"); a = ap.parse_args()
    hiread = None
    if a.hiread:
        hiread = {}
        for ln in open(a.hiread):
            h = json.loads(ln)
            if "focus" not in h: continue  # the first audit sample predates the focus field; records are matched exactly
            hiread[(os.path.basename(h["file"]), str(h["instance_id"]), int(h["focus"]))] = h["high_read_scores"]
    sp = json.load(open(a.splits))["splits"]
    s_sc, c_sc = {}, {}
    if a.struct_models:
        from embedbench.models_structural import load_model, Scorer
        for k, p in enumerate(a.struct_models.split(",")): s_sc[f"structural_model_s{k}"] = Scorer(load_model(p))
    if a.chain_models:
        from embedbench.models_chain import load_chain_model, ChainScorer
        for k, p in enumerate(a.chain_models.split(",")): c_sc[f"chain_scorer_s{k}"] = ChainScorer(load_chain_model(p))
    res = {}
    for name, ids in sp.items():
        path = os.path.join(a.dir, name)
        if not os.path.exists(path): continue
        recs = [json.loads(l) for l in open(path)]; recs = [r for r in recs if ids.get(str(r["instance_id"])) == a.split]
        for r in recs: r["_file"] = name
        if not recs: continue
        kind = "structural" if name.startswith("structural") else "quality"
        res[name] = {"n_test": len(recs), **(structural_eval(recs, s_sc) if kind == "structural" else quality_eval(recs, c_sc, hiread))}
        print(name, json.dumps(res[name]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
