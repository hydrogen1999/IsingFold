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
import argparse, hashlib, json, os, sys, time
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

from _context import host_context
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




def tie_aware_ranks(values):
    """Average ranks, the way Spearman defines them for ties.

    A double argsort breaks ties by position, which is arbitrary, and 256-read labels produce
    ties constantly. The stopping rule and the final evaluation used different definitions until
    an audit pointed it out, so they share one now.
    """
    order = np.argsort(values)
    out = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        out[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return out


def rank_corr(scores, truths):
    if len(set(scores.tolist())) < 2 or len(set(truths.tolist())) < 2:
        return None
    c = np.corrcoef(tie_aware_ranks(scores), tie_aware_ranks(truths))[0, 1]
    return float(c) if np.isfinite(c) else None

def resolve_candidate(state, candidate):
    """The embedding an action actually returns, which is not always workspace plus new_chains.

    COMMIT carries no new_chains at all: the environment returns `archive[archive_ref].chains`.
    Resolving it the workspace way labels a commit of an older embedding with whatever the
    workspace happens to hold, which is right at the root, where the two coincide, and wrong at
    every state after a change. A fourth external audit reproduced it on four fixtures out of
    four. Returning None says this opcode has no immediate-commit estimand rather than inventing
    one for it.
    """
    if candidate.opcode is Opcode.COMMIT:
        ref = candidate.archive_ref
        if ref is None or ref >= len(state.archive):
            return None
        return {n: frozenset(c) for n, c in state.archive[ref].chains.items()}
    if not candidate.changes_workspace:
        return None
    succ = {n: frozenset(c) for n, c in state.chains.items()}
    for node, chain in candidate.new_chains.items():
        succ[node] = frozenset(chain)
    return succ


def protected_entry(state):
    """The embedding the episode is guaranteed to be able to return, which is the control.

    The workspace after a rewrite is a different object and calling it the incumbent renames the
    baseline halfway through the experiment; the audit found eight of sixteen states where the
    two differ.
    """
    for entry in state.archive:
        if getattr(entry, "protected", False):
            return {n: frozenset(c) for n, c in entry.chains.items()}
    return None

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
        rows, seen, skipped_opcode = [], set(), 0
        protected = protected_entry(env.state)
        for slot, i in enumerate(chosen):
            cand = dec.candidates[i]
            succ = resolve_candidate(env.state, cand)
            if succ is None:
                skipped_opcode += 1
                continue
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
        if len(rows) >= 3 and protected is not None:
            out.append({"task": task, "obs": dec.observation, "rows": rows,
                        "incumbent": protected, "lineage": task.lineage,
                        "state_id": "%s#%d" % (task.name, step), "dec": dec,
                        "skipped_opcode": skipped_opcode})
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
    ap.add_argument("--rank-temperature", type=float, default=0.05)
    ap.add_argument("--loss", default="joint", choices=["bce", "joint", "consistent"],
                    help="'bce' fits calibrated probabilities only. 'joint' adds a softmax over "
                         "raw logits against a softmax over rates, which cannot be satisfied at "
                         "the same time as calibration. 'consistent' ranks the probabilities the "
                         "head predicts, so both terms describe one quantity")
    ap.add_argument("--cache", default="",
                    help="path to save or reuse the labelled dataset, so loss and model arms "
                         "differ in what is under test rather than in the data they saw")
    ap.add_argument("--qubit-cap", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    tasks = load_instances(a.corpus)
    split = json.loads((Path(a.corpus) / "splits.json").read_text())
    train_tasks = [t for t in tasks if t.lineage in set(split["train"])][: a.train_lineages]
    eval_tasks = [t for t in tasks if t.lineage in set(split["validation"])][: a.eval_lineages]
    ctx = host_context(a.qubit_cap)
    mm = minorminer_initializer(10)
    rng = np.random.default_rng(a.seed)
    print(json.dumps({"corpus": a.corpus, "family": a.family, "train_lineages": len(train_tasks),
                      "eval_lineages": len(eval_tasks), "states": a.states,
                      "max_candidates": a.max_candidates, "reads": a.reads}), flush=True)

    started = time.time()
    # Labels cost about thirteen minutes a run and are identical across loss and model arms, so
    # they are built once and reused. Sharing them is not a convenience: arms that regenerate
    # their own data differ in the data as well as in the thing under test.
    cache_path = Path(a.cache) if a.cache else None
    train_states = eval_states = None
    if cache_path is not None and cache_path.exists():
        import pickle
        with cache_path.open("rb") as fh:
            blob = pickle.load(fh)
        if blob.get("key") == [a.corpus, a.train_lineages, a.eval_lineages, a.states,
                              a.max_candidates, a.reads, a.seed]:
            train_states, eval_states = blob["train"], blob["eval"]
            print("  reusing labels from %s" % cache_path, flush=True)
        else:
            print("  cache at %s was built for a different configuration; rebuilding"
                  % cache_path, flush=True)
    if train_states is None:
        train_states, eval_states = [], []
        for k, t in enumerate(train_tasks):
            train_states += states_for(t, ctx, mm, rng, a.states, a.max_candidates, a.reads,
                                       5000 + k)
        for k, t in enumerate(eval_tasks):
            eval_states += states_for(t, ctx, mm, rng, a.states, a.max_candidates, a.reads,
                                      9000 + k)
        if cache_path is not None:
            import pickle
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as fh:
                pickle.dump({"key": [a.corpus, a.train_lineages, a.eval_lineages, a.states,
                                     a.max_candidates, a.reads, a.seed],
                             "train": train_states, "eval": eval_states}, fh)
            print("  wrote labels to %s" % cache_path, flush=True)
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
        # By lineage, not by position. Two states of one lineage landing on opposite sides of
        # the cut would let the stopping rule read a lineage the fit had already seen.
        roots = sorted({st["lineage"] for st in train_states})
        keep = set(roots[: int(len(roots) * (1.0 - a.inner_fraction))])
        inner_states = [st for st in train_states if st["lineage"] not in keep]
        train_states = [st for st in train_states if st["lineage"] in keep]
        assert not ({st["lineage"] for st in train_states}
                    & {st["lineage"] for st in inner_states}), "inner split shares a lineage"
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
                c = rank_corr(sc, tr_)
                if c is not None:
                    cs.append(c)
        return float(np.median(cs)) if cs else None

    best_inner, best_state, stale, steps_taken = -2.0, None, 0, 0
    model.train()
    for epoch in range(a.epochs):
        opt.zero_grad(set_to_none=True)
        total = epoch_nll = epoch_rank = epoch_regret = 0.0
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
            # binary_cross_entropy_with_logits rather than a clamped sigmoid: the clamp kills
            # the gradient wherever the logit is confident, which is where a miscalibrated head
            # most needs to move.
            target = torch.as_tensor([r["rate"] for r in st["rows"]])
            nll = F.binary_cross_entropy_with_logits(
                logit, (hits / reads), weight=reads, reduction="sum") / reads.sum()
            if a.loss == "bce":
                rank = torch.zeros((), dtype=nll.dtype)
            elif a.loss == "joint":
                # The arm as it was: a softmax over raw logits against a softmax over rates.
                # These two cannot both be satisfied. With labels 0.4 and 0.6 the joint optimum
                # puts sigmoid(z) at about 0.245 and 0.755, so asking for calibrated
                # probabilities and this ordering at once asks for two different heads.
                rank = -(F.softmax(target / a.rank_temperature, dim=0)
                         * F.log_softmax(logit, dim=0)).sum()
            else:
                # Consistent: rank the probabilities the head actually predicts, so the ordering
                # term and the calibration term describe the same quantity.
                rank = -(F.softmax(target / a.rank_temperature, dim=0)
                         * torch.log_softmax(torch.sigmoid(logit) / a.rank_temperature,
                                             dim=0)).sum()
            loss = nll + a.rank_weight * rank
            epoch_nll += float(nll.detach())
            epoch_rank += float(rank.detach()) if a.loss != "bce" else 0.0
            with torch.no_grad():
                sc = logit.detach().cpu().numpy()
                tv = target.cpu().numpy()
                best = float(tv.max())
                epoch_regret += best - float(tv[int(np.argmax(sc))])
            loss.backward()
            total += float(loss.detach())
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        steps_taken += 1
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
                print("  step %3d  nll %.5f  rank %.5f  train regret %.4f  grad %.3f  "
                      "inner rank %s  best %.3f"
                      % (steps_taken, epoch_nll / max(1, len(train_states)),
                         epoch_rank / max(1, len(train_states)),
                         epoch_regret / max(1, len(train_states)), grad_norm,
                         "%.3f" % r if r is not None else "-", best_inner), flush=True)
            if stale >= a.patience:
                print("  stopping at epoch %d; the inner split stopped improving at %.3f"
                      % (epoch, best_inner), flush=True)
                break
        elif not inner_states and (epoch % 50 == 0 or epoch == a.epochs - 1):
            print("  step %3d  nll %.5f  rank %.5f  train regret %.4f  grad %.3f"
                  % (steps_taken, epoch_nll / max(1, len(train_states)),
                     epoch_rank / max(1, len(train_states)),
                     epoch_regret / max(1, len(train_states)), grad_norm), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
        print("  restored the checkpoint with the best inner rank correlation %.3f" % best_inner,
              flush=True)
    model.eval()
    def evaluate(states, label):
        picks = {k: {} for k in ("head", "oracle", "random", "resource", "incumbent")}
        corr, lineage_of = [], {}
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
            # A fixed table, because Python hashes strings with a per-process salt and the
            # seeds would then differ between runs of the same experiment. This project wrote a
            # hand-test for exactly that trap and then reproduced it here.
            tags = {"head": 1, "oracle": 2, "random": 3, "resource": 4}
            sid = st["state_id"]
            lineage_of[sid] = st["lineage"]
            for key, row in (("head", best_by_head), ("oracle", best_by_label),
                             ("random", pick_random), ("resource", worst_res)):
                v = fresh(row, tags[key])
                if v is not None:
                    picks[key][sid] = v
            if inc is not None:
                picks["incumbent"][sid] = inc[0]
            if len(set(scores.tolist())) > 1 and len(set(truths.tolist())) > 1:
                # Average ranks, so the ties that 256-read labels produce are handled the way
                # Spearman defines rather than broken arbitrarily by argsort order.
                def ranks(v):
                    order = np.argsort(v)
                    out = np.empty(len(v), dtype=float)
                    i = 0
                    while i < len(v):
                        j = i
                        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                            j += 1
                        out[order[i:j + 1]] = 0.5 * (i + j) + 1.0
                        i = j + 1
                    return out
                c = np.corrcoef(ranks(scores), ranks(truths))[0, 1]
                if np.isfinite(c):
                    corr.append(float(c))
        print("\n== %s, %d states, every pick re-measured on %d independent reads"
              % (label, len(states), a.fresh_reads), flush=True)
        for k in ("oracle", "head", "random", "resource", "incumbent"):
            if picks[k]:
                print("  picking by %-10s %.4f  over %d states"
                      % (k, float(np.mean(list(picks[k].values()))), len(picks[k])), flush=True)

        def delta(x, y, lab):
            # Joined on the state each number belongs to, not zipped to a common length: two
            # arms that lost different states to evaluator failures would otherwise be paired
            # across different problems. Resampled by lineage, because two states of one lineage
            # are not two independent draws and treating them as such narrows every interval.
            shared = sorted(set(picks[x]) & set(picks[y]))
            if len(shared) < 5:
                print("  %-36s (too few paired states: %d)" % (lab, len(shared))); return
            groups: dict[str, list[float]] = {}
            for sid in shared:
                groups.setdefault(lineage_of[sid], []).append(picks[x][sid] - picks[y][sid])
            keys = sorted(groups)
            per = np.array([float(np.mean(groups[k])) for k in keys])
            r2 = np.random.default_rng(0)
            bs = [per[r2.integers(0, len(per), len(per))].mean() for _ in range(4000)]
            print("  %-36s n %3d states in %3d lineages  %+.4f [%+.4f, %+.4f]  (paired by lineage)"
                  % (lab, len(shared), len(keys), per.mean(),
                     np.percentile(bs, 2.5), np.percentile(bs, 97.5)), flush=True)
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
        # Enough to say what produced a checkpoint and to load it again. The previous version
        # wrote the family and the epoch count, which identifies nothing.
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        src = Path(os.environ["ISINGFOLD_SRC"]) / "isingfold" / "rl"
        h = hashlib.sha256()
        for f in sorted(src.rglob("*.py")):
            h.update(f.read_bytes())
        torch.save(model.state_dict(), out.with_suffix(".pt"))
        out.write_text(json.dumps({
            "family": a.family, "loss": a.loss, "rank_weight": a.rank_weight,
            "rank_temperature": a.rank_temperature, "epochs_requested": a.epochs,
            "optimizer_steps": steps_taken, "learning_rate": a.learning_rate,
            "weight_decay": a.weight_decay, "inner_fraction": a.inner_fraction,
            "best_inner_rank": best_inner if best_state is not None else None,
            "train_states": len(train_states), "inner_states": len(inner_states),
            "eval_states": len(eval_states), "corpus": a.corpus, "seed": a.seed,
            "reads": a.reads, "fresh_reads": a.fresh_reads,
            "source_sha256": h.hexdigest()[:16],
            "checkpoint": str(out.with_suffix(".pt")),
        }, indent=1))
        print("  wrote %s and %s" % (out, out.with_suffix(".pt")), flush=True)
    print("\nTRAIN QUALITY DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
