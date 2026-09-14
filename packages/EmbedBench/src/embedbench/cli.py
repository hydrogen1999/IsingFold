"""embedbench command line: generate-structural, generate-chain, certify, evaluate, motifs."""
from __future__ import annotations
import argparse, json, sys
from dataclasses import asdict, fields
from pathlib import Path


def _add_dataclass_args(ap, cls, skip=()):
    d = cls()
    for f in fields(cls):
        if f.name in skip: continue
        v = getattr(d, f.name)
        if isinstance(v, tuple): ap.add_argument("--" + f.name.replace("_", "-"), default=",".join(v), help="comma list")
        elif isinstance(v, bool): ap.add_argument("--" + f.name.replace("_", "-"), action="store_true", default=v)
        elif v is None: ap.add_argument("--" + f.name.replace("_", "-"), type=int, default=None)
        else: ap.add_argument("--" + f.name.replace("_", "-"), type=type(v), default=v)


def _build(cls, a, skip=()):
    kw = {}
    for f in fields(cls):
        if f.name in skip: continue
        v = getattr(a, f.name)
        if isinstance(getattr(cls(), f.name), tuple) and isinstance(v, str): v = tuple(x for x in v.split(",") if x)
        kw[f.name] = v
    return cls(**kw)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="embedbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    from embedbench.structural import GenConfig
    from embedbench.quality_chain import ChainConfig
    from embedbench.evaluate import EvalConfig
    p = sub.add_parser("generate-structural"); _add_dataclass_args(p, GenConfig); p.add_argument("--n-instances", type=int, default=10); p.add_argument("--seed", type=int, default=0); p.add_argument("--jobs", type=int, default=1); p.add_argument("--out", required=True)
    p = sub.add_parser("generate-chain"); _add_dataclass_args(p, ChainConfig); p.add_argument("--n-instances", type=int, default=10); p.add_argument("--seed", type=int, default=0); p.add_argument("--jobs", type=int, default=1); p.add_argument("--out", required=True)
    p = sub.add_parser("certify"); p.add_argument("files", nargs="+"); p.add_argument("--limit", type=int, default=None); p.add_argument("--tries", type=int, default=3); p.add_argument("--max-product", type=float, default=3e5)
    p = sub.add_parser("evaluate"); _add_dataclass_args(p, EvalConfig); p.add_argument("--embedder", default="stock", help="'stock', 'objective', or module:function"); p.add_argument("--out", default=None)
    p.add_argument("--scorers", default=None, help="comma list of scorer checkpoints for --embedder objective"); p.add_argument("--policy", default="ens_ucb"); p.add_argument("--budget", type=int, default=100); p.add_argument("--screen-model", default="default", help="joblib classifier on post-repair move features (the learned decision inside the destroy-and-repair move); 'default' = the packaged screen_v1, 'none' = the classical embedder"); p.add_argument("--screen-topk", type=int, default=20); p.add_argument("--q-cap", type=float, default=None, help="total-qubit cap as a multiple of the stock embedding (1.10, 1.25, 1.5); default uncapped"); p.add_argument("--max-candidates", type=int, default=8, help="proposals per step offered by the search"); p.add_argument("--loop-reads", type=int, default=None, help="reads per surrogate evaluation inside the search (default: the embedder's 200); the probe stays at --select-reads"); p.add_argument("--select-reads", type=int, default=None, help="reads per selection probe (default 100)"); p.add_argument("--select-pool", type=int, default=0, help="generate this many stock embeddings and let --selector pick the --select-k to probe (0 = probe exactly select-k)"); p.add_argument("--selector", default="default", help="'default' = packaged shortlist_v1 ranker, 'none' = first select-k of the pool, or a joblib path"); p.add_argument("--select-k", type=int, default=10, help="start from the best of K stock embeddings by a surrogate probe (counts against the budget)"); p.add_argument("--destroy-focus", default="random", choices=["random", "longest", "weighted"], help="which variable a destroy-and-repair move resets: uniformly at random (released), the longest chain, or proportional to chain length"); p.add_argument("--embedder-objective", default="auto", choices=["auto", "psolve", "residual"], help="what the refinement reads: solve probability, mean residual energy, or a probe of the constructed embedding deciding once"); p.add_argument("--move", default="perturb3", help="perturb3 (default, destroy-and-repair) | perturb2 | perturbmix (sizes 2 to 4) | chain (single-chain replacement ordered by --policy)")
    p = sub.add_parser("motifs")
    a = ap.parse_args(argv)
    if a.cmd == "generate-structural":
        from embedbench.structural import generate
        st = generate(_build(GenConfig, a), a.n_instances, a.seed, Path(a.out), jobs=a.jobs); print(json.dumps(asdict(st), indent=1))
    elif a.cmd == "generate-chain":
        from embedbench.quality_chain import generate_chain
        st = generate_chain(_build(ChainConfig, a), a.n_instances, a.seed, Path(a.out), jobs=a.jobs); print(json.dumps(asdict(st), indent=1))
    elif a.cmd == "certify":
        from embedbench import certify
        sys.argv = ["certify"] + a.files + (["--limit", str(a.limit)] if a.limit else []) + ["--tries", str(a.tries), "--max-product", str(a.max_product)]
        certify.main()
    elif a.cmd == "evaluate":
        from embedbench.evaluate import evaluate_embedder, evaluate_objective_embedder, stock_minorminer
        if a.embedder == "objective":
            print(json.dumps(evaluate_objective_embedder(_build(EvalConfig, a), a.scorers.split(",") if a.scorers else [], policy=a.policy, budget=a.budget, out=a.out, move=a.move, destroy_focus=a.destroy_focus, objective=a.embedder_objective, select_k=a.select_k, select_pool=a.select_pool, **({"reads": a.loop_reads} if a.loop_reads else {}), **({"select_reads": a.select_reads} if a.select_reads else {}), selector=("default" if a.selector == "default" else None if a.selector in ("none", "") else __import__("joblib").load(a.selector)), max_candidates=a.max_candidates, screen_model=("default" if a.screen_model == "default" else None if a.screen_model in ("none", "") else __import__("joblib").load(a.screen_model)), screen_topk=a.screen_topk, q_cap=a.q_cap), indent=1)); return
        if a.embedder == "stock": fn = stock_minorminer
        else:
            import importlib; mod, name = a.embedder.split(":"); fn = getattr(importlib.import_module(mod), name)
        print(json.dumps(evaluate_embedder(fn, _build(EvalConfig, a), out=a.out), indent=1))
    elif a.cmd == "motifs":
        from embedbench.handtests import certify_all
        for c in certify_all(): print(c.motif, "agrees" if c.agrees else "DISAGREES", c.certified_winner, c.margin)


if __name__ == "__main__":
    main()
