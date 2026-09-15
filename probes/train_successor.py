"""Train the successor scorer on the same labels the action head was trained on.

This is R2 of the fifth audit's plan, and it is a representation ablation: same labels, same
states, same references, same evaluation. The only thing that changes is what the model is shown.
The action head reads the environment's observation and reaches the oracle on states it was
fitted on while transferring nothing; if the successor encoder transfers, the difference is the
representation and not capacity, data or loss.

Labels come from the cache `train_quality.py` writes, so the two are compared on identical rows.
"""
import argparse, hashlib, json, os, pickle, sys, time
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
import torch
import torch.nn.functional as F
from isingfold.rl.contracts import Context
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.evaluate import first_commit_controller, run_controller
from isingfold.rl.program import compile_program

from space_features import SPACE_WIDTH
from successor_scorer import (COORD_WIDTH, PHYS_WIDTH, SuccessorScorer,
                              native_coordinates, parse_host_name, program_graph)
from train_quality import rank_corr

FRESH_BASE = 95_000_000


def compile_for(task, ctx, chains, index=1):
    """The program the evaluator would actually run for this embedding at the registered index."""
    ratios = ctx.strength_ratios
    idx = min(index, len(ratios) - 1)
    mags = [abs(v) for v in task.problem.j.values()] or [1.0]
    unit = float(np.mean(mags))
    try:
        return compile_program(chains, task.host, task.problem, unit * ratios[idx], idx,
                               field_limit=ctx.field_limit, coupler_limit=ctx.coupler_limit)
    except Exception:
        return None


def build(states, ctx, device, tag, use_coords=False, use_physics=False,
          use_space=False):
    """Compile every labelled successor once; the graphs are what the model trains on."""
    out, dropped, started = [], 0, time.time()
    coord_cache: dict[int, object] = {}
    missing = 0
    for st in states:
        coords = None
        if use_coords:
            fam, size = parse_host_name(st["task"].name.split("-")[0])
            key = (fam, size)
            if key not in coord_cache:
                coord_cache[key] = native_coordinates(st["task"].host, fam, size)
            coords = coord_cache[key]
            if coords is None:
                missing += 1
        rows = []
        for r in st["rows"]:
            prog = compile_for(st["task"], ctx, r["succ"])
            if prog is None:
                dropped += 1
                continue
            g = program_graph(prog, r["succ"], st["task"].problem, device=device,
                              coords=coords,
                              host=st["task"].host if use_physics else None,
                              space_host=st["task"].host if use_space else None)
            rows.append({**r, "graph": g})
        if len(rows) >= 3:
            out.append({**st, "rows": rows})
    print("  %s: %d states compiled, %d candidates dropped, %d without native coordinates, %.0fs"
          % (tag, len(out), dropped, missing, time.time() - started), flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--rank-weight", type=float, default=1.0)
    ap.add_argument("--rank-temperature", type=float, default=0.05)
    ap.add_argument("--inner-fraction", type=float, default=0.0)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--fresh-reads", type=int, default=512)
    ap.add_argument("--qubit-cap", type=int, default=120)
    ap.add_argument("--train-subset", type=int, default=0,
                    help="fit on this many training lineages instead of all of them, drawn in a "
                         "fixed order so the smaller sets are nested inside the larger ones. "
                         "The held-out lineages never change, so the curve measures data and "
                         "not a different test set")
    ap.add_argument("--space", action="store_true",
                    help="give each chain its room to grow on the residual graph: free volume at "
                         "radius one, two and three, the size of the free component it touches, "
                         "the best and most boxed-in direction out of it, dead directions and "
                         "blocked neighbours per qubit, and a flag when a bounded walk was cut")
    ap.add_argument("--physics", action="store_true",
                    help="give each chain the energy margin of its cheapest cut, from section "
                         "8.2 of the design: flipping a subset A of a chain costs at least "
                         "2*cut(A) - 2*L(A) in programmed units, so a chain whose cheapest cut "
                         "has a small margin is one the sampler can break in half")
    ap.add_argument("--coords", action="store_true",
                    help="give every qubit its topology coordinates from the generator\'s own "
                         "converter, Chimera (i,j,u,k), Pegasus (u,w,k,z), Zephyr (u,w,k,j,z), "
                         "with a flag marking a host family that has no native mapping")
    ap.add_argument("--clip", type=float, default=1.0,
                    help="gradient norm cap. The first run clipped at 1.0 while the raw norm "
                         "reached 4200, so the optimiser was taking a step of fixed tiny length "
                         "in a direction it was confident about; a model constrained that way "
                         "can look like it cannot learn when it is only learning slowly")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    with open(a.cache, "rb") as fh:
        blob = pickle.load(fh)
    ctx = Context(qubit_cap=a.qubit_cap)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(json.dumps({"cache": a.cache, "key": blob["key"], "device": str(device),
                      "width": a.width}), flush=True)

    train_states = build(blob["train"], ctx, device, "training lineages", a.coords,
                         a.physics, a.space)
    eval_states = build(blob["eval"], ctx, device, "held-out lineages", a.coords,
                        a.physics, a.space)
    if not train_states or not eval_states:
        print("  nothing to fit"); return 1

    if a.train_subset:
        roots = sorted({st["lineage"] for st in train_states})[: a.train_subset]
        keep = set(roots)
        train_states = [st for st in train_states if st["lineage"] in keep]
        print("  fitting on %d lineages, %d states" % (len(keep), len(train_states)), flush=True)

    inner_states = []
    if a.inner_fraction > 0:
        roots = sorted({st["lineage"] for st in train_states})
        keep = set(roots[: int(len(roots) * (1.0 - a.inner_fraction))])
        inner_states = [st for st in train_states if st["lineage"] not in keep]
        train_states = [st for st in train_states if st["lineage"] in keep]
        assert not ({s["lineage"] for s in train_states} & {s["lineage"] for s in inner_states})
        print("  fitting on %d states, stopping on %d from other training lineages"
              % (len(train_states), len(inner_states)), flush=True)

    torch.manual_seed(a.seed)
    node_dim = 4 + (COORD_WIDTH + 1 if a.coords else 0)
    chain_dim = 4 + (PHYS_WIDTH if a.physics else 0) + (SPACE_WIDTH if a.space else 0)
    model = SuccessorScorer(width=a.width, node_dim=node_dim, chain_dim=chain_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    print("  scorer parameters %d" % sum(p.numel() for p in model.parameters()), flush=True)

    def scores_for(st):
        return torch.stack([model(r["graph"]) for r in st["rows"]])

    def inner_rank():
        if not inner_states:
            return None
        cs = []
        with torch.no_grad():
            for st in inner_states:
                c = rank_corr(scores_for(st).cpu().numpy(),
                              np.array([r["rate"] for r in st["rows"]]))
                if c is not None:
                    cs.append(c)
        return float(np.median(cs)) if cs else None

    best_inner, best_state, stale, steps = -2.0, None, 0, 0
    for epoch in range(a.epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        nll_sum = rank_sum = regret_sum = 0.0
        for st in train_states:
            logit = scores_for(st)
            hits = torch.tensor([float(r["hits"]) for r in st["rows"]], device=device)
            reads = torch.tensor([float(r["reads"]) for r in st["rows"]], device=device)
            target = torch.tensor([r["rate"] for r in st["rows"]], device=device)
            nll = F.binary_cross_entropy_with_logits(logit, hits / reads, weight=reads,
                                                    reduction="sum") / reads.sum()
            rank = -(F.softmax(target / a.rank_temperature, dim=0)
                     * torch.log_softmax(torch.sigmoid(logit) / a.rank_temperature, dim=0)).sum()
            (nll + a.rank_weight * rank).backward()
            nll_sum += float(nll.detach()); rank_sum += float(rank.detach())
            with torch.no_grad():
                tv = target.cpu().numpy()
                regret_sum += float(tv.max()) - float(tv[int(np.argmax(logit.detach().cpu()))])
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip))
        opt.step(); steps += 1
        n = max(1, len(train_states))
        if inner_states and (epoch % 5 == 0 or epoch == a.epochs - 1):
            model.eval(); r = inner_rank(); model.train()
            if r is not None and r > best_inner + 1e-4:
                best_inner, stale = r, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 5
            if stale >= a.patience:
                print("  stopping at step %d; inner rank peaked at %.3f" % (steps, best_inner),
                      flush=True)
                break
        if epoch % 50 == 0 or epoch == a.epochs - 1:
            print("  step %3d  nll %.5f  rank %.5f  train regret %.4f  grad %.3f  inner %s"
                  % (steps, nll_sum / n, rank_sum / n, regret_sum / n, grad,
                     "%.3f" % best_inner if inner_states else "-"), flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    def evaluate(states, label):
        picks = {k: {} for k in ("head", "oracle", "random", "resource", "incumbent")}
        corr, lineage_of = [], {}
        rng = np.random.default_rng(0)
        tags = {"head": 1, "oracle": 2, "random": 3, "resource": 4}
        for n, st in enumerate(states):
            with torch.no_grad():
                scores = scores_for(st).cpu().numpy()
            truths = np.array([r["rate"] for r in st["rows"]])
            sid, task = st["state_id"], st["task"]
            lineage_of[sid] = st["lineage"]

            def fresh(chains, tag):
                def fixed(l, h, s):
                    return chains
                try:
                    out = run_controller([task], ctx, first_commit_controller, initializer=fixed,
                                         selector=fixed_strength_selector(),
                                         reward_reads=a.fresh_reads, repetitions=1,
                                         seed=FRESH_BASE + 13 * (n * 97 + tag))
                except Exception:
                    return None
                o = out[0]
                return float(o.utility) if o.returned_valid and o.utility is not None else None

            chosen = {"head": st["rows"][int(np.argmax(scores))],
                      "oracle": st["rows"][int(np.argmax(truths))],
                      "random": st["rows"][int(rng.integers(0, len(st["rows"])))],
                      "resource": min(st["rows"], key=lambda r: (r["qubits"], r["max_chain"]))}
            for key, row in chosen.items():
                v = fresh(row["succ"], tags[key])
                if v is not None:
                    picks[key][sid] = v
            inc = fresh(st["incumbent"], 5)
            if inc is not None:
                picks["incumbent"][sid] = inc
            c = rank_corr(scores, truths)
            if c is not None:
                corr.append(c)

        print("\n== %s, %d states, every pick re-measured on %d independent reads"
              % (label, len(states), a.fresh_reads), flush=True)
        for k in ("oracle", "head", "random", "resource", "incumbent"):
            if picks[k]:
                print("  picking by %-10s %.4f  over %d states"
                      % (k, float(np.mean(list(picks[k].values()))), len(picks[k])), flush=True)

        def delta(x, y, lab):
            shared = sorted(set(picks[x]) & set(picks[y]))
            if len(shared) < 5:
                print("  %-36s (too few: %d)" % (lab, len(shared))); return
            groups: dict[str, list[float]] = {}
            for sid2 in shared:
                groups.setdefault(lineage_of[sid2], []).append(picks[x][sid2] - picks[y][sid2])
            per = np.array([float(np.mean(groups[k])) for k in sorted(groups)])
            r2 = np.random.default_rng(0)
            bs = [per[r2.integers(0, len(per), len(per))].mean() for _ in range(4000)]
            print("  %-36s n %3d states in %3d lineages  %+.4f [%+.4f, %+.4f]  (paired by lineage)"
                  % (lab, len(shared), len(per), per.mean(),
                     np.percentile(bs, 2.5), np.percentile(bs, 97.5)), flush=True)

        delta("head", "random", "head minus random")
        delta("head", "incumbent", "head minus the protected incumbent")
        delta("oracle", "random", "oracle minus random, the ceiling")
        delta("head", "oracle", "head minus oracle, left on the table")
        if corr:
            print("  median within-state rank correlation %+.3f" % float(np.median(corr)),
                  flush=True)

    evaluate(train_states[: len(eval_states)], "TRAINING states, fitted on")
    evaluate(eval_states, "HELD-OUT lineages, never trained on")

    if a.out:
        src = Path(os.environ["ISINGFOLD_SRC"]) / "isingfold" / "rl"
        h = hashlib.sha256()
        for f in sorted(src.rglob("*.py")):
            h.update(f.read_bytes())
        out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out.with_suffix(".pt"))
        out.write_text(json.dumps({
            "model": "SuccessorScorer", "coords": bool(a.coords),
            "physics": bool(a.physics), "space": bool(a.space), "width": a.width,
            "parameters":
                sum(p.numel() for p in model.parameters()),
            "optimizer_steps": steps, "learning_rate": a.learning_rate,
            "weight_decay": a.weight_decay, "rank_weight": a.rank_weight,
            "inner_fraction": a.inner_fraction, "cache": a.cache, "cache_key": blob["key"],
            "source_sha256": h.hexdigest()[:16], "seed": a.seed}, indent=1))
    print("\nSUCCESSOR SCORER DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
