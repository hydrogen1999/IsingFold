"""Does the policy's preferred action actually lead to the better embedding?

Terminal policy gradient gives every decision in a trajectory the same advantage, so seeing a good
final embedding says nothing about which early choice produced it. This asks the question directly
and locally. From one construction prefix it takes several competing legal actions, finishes each
of them with the same policy under the same continuation randomness, and measures the terminal
residual of each completion. The comparison is then between actions that differ only in
themselves, with the prefix and the continuation held fixed.

Two things come out. A measurement: how often the action the policy already prefers is the one
that leads to the better outcome, which is a direct test of whether its ranking is wrong and there
is anything to learn. And a training signal: at each branch point, an ordering over real
alternatives supported by measured energy, which is what a terminal reward cannot supply.

Nothing here prefers fewer qubits or more cycles. The only thing that earns preference is the
measured residual of the embedding the action led to.
"""
import argparse, json, math, os, statistics, sys
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
import torch
from isingfold.rl.env import DecisionState, EmbeddingEnv, Mode, TerminalRecord, fixed_strength_selector

import constructor_curriculum as cc
from constructor_objective import measure_terminal
from constructor_rollout import _context
from constructor_tiny_gate import no_completion_solver


def walk(task, actor, fc, ctx, seed, prefix_actions, branch, continuation_seed, max_steps,
         temperature=1.0):
    """Replay a fixed action prefix, take ``branch``, then let the policy finish.

    The prefix is replayed rather than copied because the environment holds no cheap snapshot, and
    at this cell a prefix is tens of decisions. The continuation seed is passed in so two
    different branches are finished under the same randomness and the comparison is the branch.
    """
    env = EmbeddingEnv(task, ctx, mode=Mode.CONSTRUCTION, initializer=None,
                       selector=fixed_strength_selector(), reward_reads=8, build_observation=False)
    current = env.reset(seed)
    rng = np.random.default_rng(continuation_seed)
    taken, rows_at_branch, legal_at_branch = [], None, None
    steps = 0
    while isinstance(current, DecisionState) and steps < max_steps:
        legal = [i for i, ok in enumerate(current.legal_mask) if ok]
        if not legal:
            return None, taken, rows_at_branch, legal_at_branch
        rows = np.stack([fc.observe(current.candidates[i], env.state.chains, state=env.state,
                                    ctx=ctx, steps_left=max_steps - steps, max_steps=max_steps)
                         for i in legal])
        if steps < len(prefix_actions):
            pick = prefix_actions[steps]
            if pick not in legal:            # the prefix is not replayable here; give up cleanly
                return None, taken, rows_at_branch, legal_at_branch
        elif steps == len(prefix_actions) and branch is not None:
            rows_at_branch, legal_at_branch = rows, legal
            if branch not in legal:
                return None, taken, rows_at_branch, legal_at_branch
            pick = branch
        else:
            with torch.no_grad():
                logits = actor(torch.as_tensor(rows, dtype=torch.float32)).reshape(-1) / temperature
                probs = torch.softmax(logits, 0).numpy()
            pick = int(legal[int(rng.choice(len(legal), p=probs / probs.sum()))])
        taken.append(pick)
        current = env.step(current, pick, evaluate_training_reward=False).next_decision_or_terminal
        steps += 1
    terminal = current if isinstance(current, TerminalRecord) else None
    return (terminal if terminal is not None and terminal.returned_valid else None), \
        taken, rows_at_branch, legal_at_branch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--cells", default="")
    ap.add_argument("--init", required=True)
    ap.add_argument("--features", default="physics")
    ap.add_argument("--actor", default="linear")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--support", default="wide")
    ap.add_argument("--n-tasks", type=int, default=12)
    ap.add_argument("--prefixes", type=int, default=4)
    ap.add_argument("--actions", type=int, default=4)
    ap.add_argument("--continuations", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--reads", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261003)
    ap.add_argument("--dump", default="")
    a = ap.parse_args()

    cells = [c for c in a.cells.split(",") if c]
    tasks, _ = cc.build_corpus_sets(a.corpus, cells, a.n_tasks, 1, a.seed)
    actor = cc.make_actor(a.actor, a.width, cc.FEATURE_WIDTHS[a.features])
    cc.load_init(a.init, actor, a.actor, a.features,
                 expand=(a.features == "physics"))
    wide = a.support == "wide"
    print(json.dumps({"probe": "branch_compare", "corpus": a.corpus, "cells": cells,
                      "init": a.init, "features": a.features, "tasks": len(tasks),
                      "prefixes": a.prefixes, "actions": a.actions,
                      "continuations": a.continuations, "reads": a.reads}), flush=True)
    print("  %-26s %6s %7s %9s %9s %9s %s"
          % ("task", "depth", "actions", "best", "policy", "spread", "policy was best"),
          flush=True)

    records, wins, comparisons = [], 0, 0
    with no_completion_solver():
        for k, t in enumerate(tasks):
            ctx = _context(t, len(t.host), a.max_steps, 8, None, 2, wide=wide)
            fc = cc.make_features(a.features, t)
            spine_terminal, spine, _, _ = walk(t, actor, fc, ctx, a.seed + k, [], None,
                                               a.seed + 1000 + k, a.max_steps)
            if spine_terminal is None or len(spine) < 6:
                print(json.dumps({"task": t.name, "skipped": "no valid spine"}), flush=True)
                continue
            rng = np.random.default_rng(a.seed + 7 * k)
            depths = sorted(set(int(x) for x in rng.integers(2, len(spine) - 2, a.prefixes)))
            for depth in depths:
                prefix = spine[:depth]
                # Find the branch point's candidates once, by replaying with no branch taken.
                _, _, rows, legal = walk(t, actor, fc, ctx, a.seed + k, prefix, -1,
                                         a.seed + 2000 + k, a.max_steps)
                if rows is None or legal is None or len(legal) < a.actions:
                    continue
                with torch.no_grad():
                    scores = actor(torch.as_tensor(rows, dtype=torch.float32)).reshape(-1)
                order = list(np.argsort(-scores.numpy()))
                # The policy's own pick, then alternatives spread across its ranking, so the
                # comparison is not confined to candidates it already likes.
                picks = [order[0]] + [order[int(x)] for x in
                                      np.linspace(1, len(order) - 1, a.actions - 1).astype(int)]
                picks = list(dict.fromkeys(picks))[:a.actions]
                results = {}
                for j in picks:
                    vals = []
                    for c in range(a.continuations):
                        term, _, _, _ = walk(t, actor, fc, ctx, a.seed + k, prefix, legal[int(j)],
                                             a.seed + 3000 + 100 * depth + c, a.max_steps)
                        if term is not None:
                            vals.append(measure_terminal(t, term, a.seed + 40000 + 10 * depth + c,
                                                         a.reads))
                    if vals:
                        results[int(j)] = sum(vals) / len(vals)
                if len(results) < 2:
                    continue
                best = min(results, key=results.get)
                policy_pick = int(picks[0])
                comparisons += 1
                wins += int(best == policy_pick)
                spread = max(results.values()) - min(results.values())
                print("  %-26s %6d %7d %9.4f %9.4f %9.4f %s"
                      % (t.name[:26], depth, len(results), results[best],
                         results.get(policy_pick, float("nan")), spread,
                         "yes" if best == policy_pick else "no"), flush=True)
                records.append({"task": t.name, "depth": depth,
                                "rows": rows.tolist(), "legal": [int(x) for x in legal],
                                "residual_by_action": {str(j): v for j, v in results.items()},
                                "policy_pick": policy_pick, "best": int(best)})

    if comparisons:
        spreads = [max(r["residual_by_action"].values()) - min(r["residual_by_action"].values())
                   for r in records]
        n = len(spreads); m = sum(spreads) / n
        h = 1.96 * (statistics.stdev(spreads) if n > 1 else 0.0) / math.sqrt(n)
        summary = {"summary": "branch_compare", "branch_points": comparisons,
                   "policy_pick_was_best": wins, "rate": wins / comparisons,
                   "chance_rate": 1.0 / max(1, sum(len(r["residual_by_action"]) for r in records)
                                            / max(1, len(records))),
                   "mean_residual_spread": m, "spread_interval": [m - h, m + h],
                   "reading": ("a rate near chance with a spread well above the measurement noise "
                               "says the branch choice matters and the policy is not making it; a "
                               "spread near zero says the branch choice does not matter here and "
                               "this is the wrong place to teach")}
        print(json.dumps(summary), flush=True)
        if a.dump:
            Path(a.dump).parent.mkdir(parents=True, exist_ok=True)
            Path(a.dump).write_text("\n".join(json.dumps(r) for r in records))
            print(json.dumps({"dumped": a.dump, "records": len(records)}), flush=True)
    print("BRANCH COMPARE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
