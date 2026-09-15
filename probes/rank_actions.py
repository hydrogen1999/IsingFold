"""Can the model rank the actions available at one state? Measured on Q(s,a), not on V(s).

The value probes this replaces asked a different question than the one they reported. They took
the state value before an action and used it to rank the embeddings different actions lead to.
If a state offers two commits worth 0.8 and 0.2, the state value under a policy that picks
evenly is 0.5, and it is 0.5 for both: the quantity does not distinguish them and was never
going to. Concluding from that that the network cannot judge embeddings is a statement about the
target, not about the network. The external audit made this point and it is right.

Ranking candidates at a state is a question about Q(s, a). The specified network has a per-action
quality head, so this probe reads that head, and measures the truth by committing each candidate
and spending an evaluator block on it. Both the estimate and the truth are then per action, at
one state, which is the decision that actually has to be made.

What the numbers mean:

    oracle       the best candidate at this state, if you could measure them all
    random       picking one uniformly, which costs nothing
    quality head the candidate the model's head ranks first
    resource     the candidate with fewest qubits then shortest chain, the classical criterion

A head worth having sits between random and oracle. A head below random is worse than useless,
because acting on it is worse than not looking.
"""
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
import torch
from isingfold.rl.contracts import Context, Opcode
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import (LEGACY_ONLINE_INITIALIZER_RESTARTS_V1, EmbeddingEnv, Mode,
                              fixed_strength_selector)
from isingfold.rl.evaluate import first_commit_controller, run_controller
from isingfold.rl.model import build_model

from _initializers import minorminer_initializer

ASSESS_BASE = 80_000_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="runs/v1/corpus")
    ap.add_argument("--split", default="validation", choices=["validation", "test"])
    ap.add_argument("--lineages", type=int, default=40)
    ap.add_argument("--max-candidates", type=int, default=12,
                    help="how many legal candidates to measure per state; measuring every one "
                         "of sixty-odd would cost more reads than the experiment is worth")
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--qubit-cap", type=int, default=120)
    ap.add_argument("--reads", type=int, default=256)
    a = ap.parse_args()

    tasks = load_instances(a.corpus)
    split = json.loads((Path(a.corpus) / "splits.json").read_text())
    dev = [t for t in tasks if t.lineage in set(split[a.split])][: a.lineages]
    ctx = Context(qubit_cap=a.qubit_cap)
    mm = minorminer_initializer(10)
    print(json.dumps({"corpus": a.corpus, "split": a.split, "lineages": len(dev),
                      "max_candidates": a.max_candidates, "reads": a.reads}), flush=True)

    def truth(task, chains, tag):
        def fixed(l, h, s):
            return chains
        try:
            out = run_controller([task], ctx, first_commit_controller, initializer=fixed,
                                 selector=fixed_strength_selector(), reward_reads=a.reads,
                                 repetitions=1, seed=ASSESS_BASE + 13 * tag)
        except Exception:
            return None
        o = out[0]
        return float(o.utility) if o.returned_valid and o.utility is not None else None

    for fam, path in (kv.split("=", 1) for kv in a.checkpoints.split(",") if kv):
        if not Path(path).exists():
            print("\n== %s: no checkpoint at %s" % (fam, path)); continue
        model = build_model(fam, improvement_mode=True)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        print("\n== %s (%s)" % (fam, path), flush=True)

        picks = {k: [] for k in ("oracle", "observed_max", "random", "quality",
                                 "resource", "mean")}
        spearman, states, measured = [], 0, 0
        rng = np.random.default_rng(0)
        for idx, t in enumerate(dev):
            env = EmbeddingEnv(t, ctx, mode=Mode.IMPROVEMENT, initializer=mm,
                               selector=fixed_strength_selector(), reward_reads=a.reads, seed=0,
                               improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1)
            dec = env.reset(0)
            if not hasattr(dec, "candidates"):
                continue
            legal = [i for i, ok in enumerate(dec.legal_mask) if ok
                     and dec.candidates[i].opcode is not Opcode.STOP]
            if len(legal) < 3:
                continue
            legal = list(rng.permutation(legal))[: a.max_candidates]
            base = {n: frozenset(c) for n, c in env.state.chains.items()}

            rows = []
            for slot, i in enumerate(legal):
                cand = dec.candidates[i]
                succ = dict(base)
                for node, chain in cand.new_chains.items():
                    succ[node] = frozenset(chain)
                u = truth(t, succ, idx * 1000 + slot)
                if u is None:
                    continue
                q = sum(len(c) for c in succ.values())
                m = max(len(c) for c in succ.values())
                rows.append({"i": i, "u": u, "qubits": q, "max_chain": m})
                measured += 1
            if len(rows) < 3:
                continue
            states += 1

            with torch.no_grad():
                out = model.forward_single(dec.observation, None)
            head = out.action_quality_value
            if head is None:
                head = out.action_quality_logit
            head = None if head is None else head.detach().cpu().numpy()

            truths = np.array([r["u"] for r in rows])
            # The maximum of the same blocks used as truth is not the best candidate's quality,
            # it is that quality plus whatever the luckiest block added. A control with twelve
            # identical candidates at 256 reads puts the inflation near +0.046. So the winner is
            # re-measured on independent reads, and the inflated figure is kept beside it under
            # a name that says what it is.
            top = rows[int(np.argmax(truths))]
            fresh = truth(t, {**base, **{n: frozenset(c)
                                         for n, c in dec.candidates[top["i"]].new_chains.items()}},
                          500000 + idx)
            picks["observed_max"].append(float(truths.max()))
            if fresh is not None:
                picks["oracle"].append(fresh)
            picks["mean"].append(float(truths.mean()))
            picks["random"].append(float(truths[rng.integers(0, len(truths))]))
            picks["resource"].append(
                float(min(rows, key=lambda r: (r["qubits"], r["max_chain"]))["u"]))
            if head is not None:
                scores = np.array([float(head[r["i"]]) for r in rows])
                picks["quality"].append(float(rows[int(np.argmax(scores))]["u"]))
                if len(set(scores.tolist())) > 1 and len(set(truths.tolist())) > 1:
                    sr = np.argsort(np.argsort(scores)).astype(float)
                    tr = np.argsort(np.argsort(truths)).astype(float)
                    c = np.corrcoef(sr, tr)[0, 1]
                    if np.isfinite(c):
                        spearman.append(float(c))

        print("  states %d, candidates measured %d" % (states, measured), flush=True)
        for k in ("observed_max", "oracle", "quality", "random", "resource", "mean"):
            if picks[k]:
                print("  picking by %-12s utility %.4f" % (k, float(np.mean(picks[k]))),
                      flush=True)

        def delta(x, y, label):
            n = min(len(picks[x]), len(picks[y]))
            if n < 5:
                return
            d = np.array(picks[x][:n]) - np.array(picks[y][:n])
            rng2 = np.random.default_rng(0)
            bs = [d[rng2.integers(0, n, n)].mean() for _ in range(4000)]
            print("  %-34s n %3d  %+.4f [%+.4f, %+.4f]"
                  % (label, n, d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)),
                  flush=True)

        delta("quality", "random", "quality head minus random")
        delta("oracle", "random", "oracle minus random, the ceiling")
        delta("observed_max", "oracle", "what the selection maximum invents")
        delta("resource", "random", "resource criterion minus random")
        if spearman:
            print("  median within-state rank correlation of the head with the truth: %+.3f"
                  % float(np.median(spearman)), flush=True)

    print("\nRANK ACTIONS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
