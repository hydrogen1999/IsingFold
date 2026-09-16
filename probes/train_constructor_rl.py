"""Reinforcement learning on the constructive embedder, from the prioritiser up.

The policy is the prioritiser: one network scores every candidate the generator offers from
local features, serves as the generator's preference before the shortlist is cut, and gives
the action distribution, a softmax over the legal candidates of the decision. Episodes run
the construction environment on a fill-planted instance from an empty embedding, with no
witness anywhere in the loop; the witness is only used, optionally, to initialise the
network from imitation. The reward is terminal: 1 for a valid COMMIT, plus the fraction of
logical demands realised at the end so that early policies get a gradient at all.

Learning is REINFORCE with a self-competition baseline: K episodes per instance, each
episode's advantage is its return minus the mean of the K, which is the best-of-K spirit
the project has measured to matter. Wall time per episode is recorded, because the number
that goes on the board is validity at a deadline against the anytime baseline.
"""
import argparse, json, os, pickle, sys, time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
import torch
from isingfold.rl.contracts import DecisionState, Opcode
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import EmbeddingEnv, Mode, fixed_strength_selector

from _context import construction_context
from candidate_features import FeatureContext
from train_prioritiser import Prioritiser


def candidate_tuple(c):
    return (c.opcode.value, tuple(c.affected),
            tuple(sorted({q for v, ch in c.new_chains.items()
                          for q in ch - c.old_chains.get(v, frozenset())}, key=str)))


def demands_realised(task, chains):
    edges = list(task.logical.edges())
    if not edges:
        return 1.0
    met = 0
    for u, v in edges:
        cu, cv = chains.get(u), chains.get(v)
        if cu and cv and any(task.host.has_edge(a, b) for a in cu for b in cv):
            met += 1
    return met / len(edges)


def episode(task, model, fc, temperature, max_steps, rng, deadline, train=True):
    """One construction episode with the policy as preference and as actor."""
    ctx = construction_context(task.host.number_of_nodes(), task.logical.number_of_nodes(),
                               task.logical.number_of_edges())
    env = EmbeddingEnv(task, ctx, mode=Mode.CONSTRUCTION, initializer=None,
                       selector=fixed_strength_selector(), reward_reads=8)

    def prefer(v, q):
        with torch.no_grad():
            return float(model(torch.as_tensor(fc.pair(v, [q], env.state.chains, "PLACE"))))

    env.generator.prefer = prefer
    dec = env.reset(int(rng.integers(0, 2 ** 31)))
    logps, steps, t0 = [], 0, time.time()
    while isinstance(dec, DecisionState) and steps < max_steps and time.time() - t0 < deadline:
        chains = env.state.chains
        legal = np.asarray(dec.legal_mask, dtype=bool)
        if not legal.any():
            break
        feats = np.stack([fc.candidate(candidate_tuple(c), chains) for c in dec.candidates])
        scores = model(torch.as_tensor(feats)) / temperature
        scores = scores.masked_fill(~torch.as_tensor(legal), -1e9)
        # Once everything is placed and every demand is met, commit; otherwise sample.
        commit = [i for i, c in enumerate(dec.candidates) if c.opcode is Opcode.COMMIT and legal[i]]
        if commit and demands_realised(task, chains) >= 1.0:
            pick = commit[0]
        else:
            dist = torch.distributions.Categorical(logits=scores)
            pick = int(dist.sample()) if train else int(torch.argmax(scores))
            logps.append(dist.log_prob(torch.tensor(pick)))
        dec = env.step(dec, pick, evaluate_training_reward=False).next_decision_or_terminal
        steps += 1
    valid = bool(getattr(dec, "returned_valid", False)) and not isinstance(dec, DecisionState)
    frac = demands_realised(task, env.state.chains)
    return {"valid": valid, "frac": frac, "steps": steps, "secs": time.time() - t0,
            "logps": logps, "return": (1.0 if valid else 0.0) + frac}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--init", default="", help="a trained prioritiser to start from")
    ap.add_argument("--out", required=True)
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--instances-per-iteration", type=int, default=4)
    ap.add_argument("--episodes-per-instance", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--deadline", type=float, default=300.0)
    ap.add_argument("--holdout-fraction", type=float, default=0.25)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tasks = load_instances(a.corpus)
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    names = sorted(t.name for t in tasks)
    held = set(rng.choice(names, size=max(1, int(len(names) * a.holdout_fraction)), replace=False))
    train_tasks = [t for t in tasks if t.name not in held]
    eval_tasks = [t for t in tasks if t.name in held]
    model = Prioritiser(a.width)
    if a.init:
        blob = torch.load(a.init, map_location="cpu")
        model.load_state_dict(blob["state"])
    opt = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    fcs = {}
    print(json.dumps({"corpus": a.corpus, "train": len(train_tasks), "held_out": len(eval_tasks),
                      "init": a.init or None, "episodes_per_instance": a.episodes_per_instance,
                      "deadline": a.deadline}), flush=True)

    def fc_for(task):
        if task.name not in fcs:
            fcs[task.name] = FeatureContext(task)
        return fcs[task.name]

    def evaluate(tag):
        model.eval()
        rows = []
        for t in eval_tasks:
            r = episode(t, model, fc_for(t), a.temperature, a.max_steps, rng, a.deadline, train=False)
            rows.append(r)
        model.train()
        v = np.mean([r["valid"] for r in rows]); f = np.mean([r["frac"] for r in rows])
        s = np.mean([r["secs"] for r in rows])
        print("  %s held-out: valid %.2f  demands %.3f  secs/episode %.1f  over %d instances"
              % (tag, v, f, s, len(rows)), flush=True)
        return v, f

    best = -1.0
    evaluate("init")
    for it in range(a.iterations):
        batch = rng.choice(len(train_tasks), size=min(a.instances_per_iteration, len(train_tasks)),
                           replace=False)
        loss, n, stats = 0.0, 0, defaultdict(list)
        for idx in batch:
            t = train_tasks[idx]
            eps = [episode(t, model, fc_for(t), a.temperature, a.max_steps, rng, a.deadline)
                   for _ in range(a.episodes_per_instance)]
            base = np.mean([e["return"] for e in eps])
            for e in eps:
                adv = e["return"] - base
                if e["logps"] and abs(adv) > 1e-9:
                    loss = loss - adv * torch.stack(e["logps"]).sum() / len(e["logps"])
                    n += 1
                stats["valid"].append(e["valid"]); stats["frac"].append(e["frac"])
                stats["secs"].append(e["secs"])
        if n:
            opt.zero_grad(); (loss / n).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        print("  iter %4d  train valid %.2f  demands %.3f  secs/episode %.1f"
              % (it, np.mean(stats["valid"]), np.mean(stats["frac"]), np.mean(stats["secs"])),
              flush=True)
        if (it + 1) % a.eval_every == 0:
            v, f = evaluate("iter %d" % it)
            if v + f > best:
                best = v + f
                torch.save({"state": model.state_dict(), "width": a.width}, a.out)
    torch.save({"state": model.state_dict(), "width": a.width}, a.out + ".last")
    print("\nCONSTRUCTOR RL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
