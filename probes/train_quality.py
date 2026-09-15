"""Does a directly supervised quality head generalise to lineages it never saw?

A third external audit established two things on small fixtures. Within one state the candidates
really do differ: choosing by labels from one block of reads and scoring on an independent block
gives 74.0 percent against 55.7 for a uniform pick, and the two blocks agree at correlation 0.98
to 0.99, so the difference is not read noise. And a freshly initialised network fits those labels
to a regret of zero on the states it was fitted on, which says the features and the gradient path
carry the signal. What it could not test is the question that decides whether any of this is
useful: does the fit transfer to states from lineages the model never trained on?

This probe answers that. Labels come from the real evaluator on real candidates. Training states
and held-out states come from disjoint lineages. The head is supervised directly, with a binomial
likelihood on the counts and a within-state ranking term, because picking the best candidate at a
state is a ranking problem and a regression that is right on average can still order wrongly.

The reported number is what an argmax over the head's scores is worth on fresh reads at held-out
states, against four references: the oracle over the same pool re-measured independently, a
uniform pick, the resource criterion, and the protected incumbent.
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
import torch
import torch.nn.functional as F
from isingfold.rl.contracts import Context, Opcode
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import (LEGACY_ONLINE_INITIALIZER_RESTARTS_V1, EmbeddingEnv, Mode,
                              fixed_strength_selector)
from isingfold.rl.evaluate import first_commit_controller, run_controller
from isingfold.rl.model import build_model

from _initializers import minorminer_initializer

LABEL_BASE = 40_000_000
FRESH_BASE = 95_000_000


def fixed(chains):
    def _i(l, h, s):
        return chains
    return _i


def block(task, ctx, chains, seed, reads):
    """One evaluator block on a fixed embedding: hits and reads, not just the rate."""
    try:
        out = run_controller([task], ctx, first_commit_controller, initializer=fixed(chains),
                             selector=fixed_strength_selector(), reward_reads=reads,
                             repetitions=1, seed=seed)
    except Exception:
        return None
    o = out[0]
    if not o.returned_valid or o.utility is None:
        return None
    hits = o.evaluator_hits if o.evaluator_hits is not None else round(float(o.utility) * reads)
    return float(o.utility), int(hits), int(o.evaluator_reads or reads)


def states_for(task, ctx, mm, rng, n_states, max_candidates, reads, seed0):
    """Collect candidate pools with labels at a few states of one lineage."""
    env = EmbeddingEnv(task, ctx, mode=Mode.IMPROVEMENT, initializer=mm,
                       selector=fixed_strength_selector(), reward_reads=reads, seed=seed0,
                       improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1)
    dec = env.reset(seed0)
    out = []
    for step in range(n_states):
        if not hasattr(dec, "candidates"):
            break
        legal = [i for i, ok in enumerate(dec.legal_mask) if ok
                 and dec.candidates[i].opcode is not Opcode.STOP]
        if len(legal) < 3:
            break
        chosen = list(rng.permutation(legal))[:max_candidates]
        base = {n: frozenset(c) for n, c in env.state.chains.items()}
        rows, seen = [], set()
        for slot, i in enumerate(chosen):
            cand = dec.candidates[i]
            succ = dict(base)
            for node, chain in cand.new_chains.items():
                succ[node] = frozenset(chain)
            key = frozenset((n, frozenset(c)) for n, c in succ.items())
            if key in seen:
                continue
            seen.add(key)
            lab = block(task, ctx, succ, LABEL_BASE + 13 * (seed0 + slot + 1000 * step), reads)
            if lab is None:
                continue
            rows.append({"i": i, "succ": succ, "rate": lab[0], "hits": lab[1], "reads": lab[2],
                         "qubits": sum(len(c) for c in succ.values()),
                         "max_chain": max(len(c) for c in succ.values())})
        if len(rows) >= 3:
            out.append({"task": task, "obs": dec.observation, "rows": rows,
                        "incumbent": base, "dec": dec})
        if step + 1 >= n_states:
            break
        pick = int(rng.integers(0, len(legal)))
        nxt = env.step(dec, legal[pick], evaluate_training_reward=False).next_decision_or_terminal
        if not hasattr(nxt, "candidates"):
            break
        dec = nxt
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="runs/v1/corpus")
    ap.add_argument("--family", default="if-dual")
    ap.add_argument("--train-lineages", type=int, default=150)
    ap.add_argument("--eval-lineages", type=int, default=50)
    ap.add_argument("--states", type=int, default=2)
    ap.add_argument("--max-candidates", type=int, default=8)
    ap.add_argument("--reads", type=int, default=256)
    ap.add_argument("--fresh-reads", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--inner-fraction", type=float, default=0.0,
                    help="hold this fraction of the TRAINING lineages out of the fit and stop on "
                         "their rank correlation. Without it a few hundred states and a few "
                         "million parameters memorise, and a failure to transfer says nothing "
                         "about whether the signal is transferable")
    ap.add_argument("--patience", type=int, default=20,
                    help="epochs of no improvement on the inner split before stopping")
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--rank-weight", type=float, default=1.0)
    ap.add_argument("--qubit-cap", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    tasks = load_instances(a.corpus)
    split = json.loads((Path(a.corpus) / "splits.json").read_text())
    train_tasks = [t for t in tasks if t.lineage in set(split["train"])][: a.train_lineages]
    eval_tasks = [t for t in tasks if t.lineage in set(split["validation"])][: a.eval_lineages]
    ctx = Context(qubit_cap=a.qubit_cap)
    mm = minorminer_initializer(10)
    rng = np.random.default_rng(a.seed)
    print(json.dumps({"corpus": a.corpus, "family": a.family, "train_lineages": len(train_tasks),
                      "eval_lineages": len(eval_tasks), "states": a.states,
                      "max_candidates": a.max_candidates, "reads": a.reads}), flush=True)

    started = time.time()
    train_states, eval_states = [], []
    for k, t in enumerate(train_tasks):
        train_states += states_for(t, ctx, mm, rng, a.states, a.max_candidates, a.reads, 5000 + k)
    for k, t in enumerate(eval_tasks):
        eval_states += states_for(t, ctx, mm, rng, a.states, a.max_candidates, a.reads, 9000 + k)
    labels = sum(len(s["rows"]) for s in train_states + eval_states)
    print("  %d training states, %d held-out states, %d labelled candidates, %.0fs"
          % (len(train_states), len(eval_states), labels, time.time() - started), flush=True)
    if not train_states or not eval_states:
        print("  not enough states to fit"); return 1

    torch.manual_seed(a.seed)
    model = build_model(a.family, improvement_mode=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.learning_rate,
                           weight_decay=a.weight_decay)

    # The inner split comes out of the training lineages, so the held-out lineages stay untouched
    # by the stopping rule as well as by the gradient.
    inner_states = []
    if a.inner_fraction > 0:
        cut = int(len(train_states) * (1.0 - a.inner_fraction))
        inner_states = train_states[cut:]
        train_states = train_states[:cut]
        print("  fitting on %d states, stopping on %d held out of the training lineages"
              % (len(train_states), len(inner_states)), flush=True)

    def inner_rank():
        if not inner_states:
            return None
        cs = []
        with torch.no_grad():
            for st in inner_states:
                head = model.forward_single(st["obs"], None).action_quality_logit
                sc = np.array([float(head[r["i"]]) for r in st["rows"]])
                tr_ = np.array([r["rate"] for r in st["rows"]])
                if len(set(sc.tolist())) > 1 and len(set(tr_.tolist())) > 1:
                    a_ = np.argsort(np.argsort(sc)).astype(float)
                    b_ = np.argsort(np.argsort(tr_)).astype(float)
                    c = np.corrcoef(a_, b_)[0, 1]
                    if np.isfinite(c):
                        cs.append(float(c))
        return float(np.median(cs)) if cs else None

    best_inner, best_state, stale = -2.0, None, 0
    model.train()
    for epoch in range(a.epochs):
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for st in train_states:
            out = model.forward_single(st["obs"], None)
            head = out.action_quality_logit
            if head is None:
                print("  this family exposes no action-quality logit"); return 1
            idx = torch.as_tensor([r["i"] for r in st["rows"]], dtype=torch.long)
            logit = head[idx]
            hits = torch.as_tensor([float(r["hits"]) for r in st["rows"]])
            reads = torch.as_tensor([float(r["reads"]) for r in st["rows"]])
            # Binomial likelihood on the counts: a rate of 7/8 and one of 700/800 are not the
            # same evidence and a plain BCE on rates would treat them as if they were.
            p = torch.sigmoid(logit).clamp(1e-6, 1 - 1e-6)
            nll = -(hits * torch.log(p) + (reads - hits) * torch.log(1 - p)).sum() / reads.sum()
            # Ranking inside the state, because choosing the best candidate is an ordering
            # problem and a fit that is right on average can still order wrongly.
            target = torch.as_tensor([r["rate"] for r in st["rows"]])
            lo = F.log_softmax(logit, dim=0)
            tgt = F.softmax(target / 0.05, dim=0)
            rank = -(tgt * lo).sum()
            loss = nll + a.rank_weight * rank
            loss.backward()
            total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if inner_states and (epoch % 5 == 0 or epoch == a.epochs - 1):
            model.eval()
            r = inner_rank()
            model.train()
            if r is not None and r > best_inner + 1e-4:
                best_inner, stale = r, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 5
            if epoch % 25 == 0:
                print("  epoch %3d  loss %.5f  inner rank %s  best %.3f"
                      % (epoch, total / max(1, len(train_states)),
                         "%.3f" % r if r is not None else "-", best_inner), flush=True)
            if stale >= a.patience:
                print("  stopping at epoch %d; the inner split stopped improving at %.3f"
                      % (epoch, best_inner), flush=True)
                break
        elif not inner_states and (epoch % 50 == 0 or epoch == a.epochs - 1):
            print("  epoch %3d  mean loss %.5f"
                  % (epoch, total / max(1, len(train_states))), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
        print("  restored the checkpoint with the best inner rank correlation %.3f" % best_inner,
              flush=True)
    model.eval()
    def evaluate(states, label):
        picks = {k: [] for k in ("head", "oracle", "random", "resource", "incumbent")}
        corr = []
        for n, st in enumerate(states):
            rows = st["rows"]
            with torch.no_grad():
                head = model.forward_single(st["obs"], None).action_quality_logit
            scores = np.array([float(head[r["i"]]) for r in rows])
            truths = np.array([r["rate"] for r in rows])

            def fresh(row, tag):
                got = block(st["task"], ctx, row["succ"], FRESH_BASE + 13 * (n * 97 + tag),
                            a.fresh_reads)
                return None if got is None else got[0]

            best_by_head = rows[int(np.argmax(scores))]
            best_by_label = rows[int(np.argmax(truths))]
            worst_res = min(rows, key=lambda r: (r["qubits"], r["max_chain"]))
            pick_random = rows[int(rng.integers(0, len(rows)))]
            inc = block(st["task"], ctx, st["incumbent"], FRESH_BASE + 13 * (n * 97 + 77),
                        a.fresh_reads)
            for key, row in (("head", best_by_head), ("oracle", best_by_label),
                             ("random", pick_random), ("resource", worst_res)):
                v = fresh(row, hash(key) % 50)
                if v is not None:
                    picks[key].append(v)
            if inc is not None:
                picks["incumbent"].append(inc[0])
            if len(set(scores.tolist())) > 1 and len(set(truths.tolist())) > 1:
                sr = np.argsort(np.argsort(scores)).astype(float)
                tr = np.argsort(np.argsort(truths)).astype(float)
                c = np.corrcoef(sr, tr)[0, 1]
                if np.isfinite(c):
                    corr.append(float(c))
        print("\n== %s, %d states, every pick re-measured on %d independent reads"
              % (label, len(states), a.fresh_reads), flush=True)
        for k in ("oracle", "head", "random", "resource", "incumbent"):
            if picks[k]:
                print("  picking by %-10s %.4f" % (k, float(np.mean(picks[k]))), flush=True)

        def delta(x, y, lab):
            n2 = min(len(picks[x]), len(picks[y]))
            if n2 < 5:
                return
            d = np.array(picks[x][:n2]) - np.array(picks[y][:n2])
            r2 = np.random.default_rng(0)
            bs = [d[r2.integers(0, n2, n2)].mean() for _ in range(4000)]
            print("  %-36s n %3d  %+.4f [%+.4f, %+.4f]"
                  % (lab, n2, d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)),
                  flush=True)
        delta("head", "random", "head minus random")
        delta("head", "incumbent", "head minus the protected incumbent")
        delta("oracle", "random", "oracle minus random, the ceiling")
        delta("head", "oracle", "head minus oracle, what is left on the table")
        if corr:
            print("  median within-state rank correlation %+.3f" % float(np.median(corr)),
                  flush=True)
        return picks

    evaluate(train_states[: len(eval_states)], "TRAINING states, fitted on")
    evaluate(eval_states, "HELD-OUT lineages, never trained on")
    if a.out:
        Path(a.out).write_text(json.dumps({"family": a.family, "epochs": a.epochs}, indent=1))
    print("\nTRAIN QUALITY DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
