"""Policy-guided limited discrepancy search for construction.

Greedy construction under a learned prioritiser reaches the last few percent of demands and
dead-ends: some earlier choice took a qubit the completion needed, and nothing undoes it.
The environment has no undo, but it is deterministic given the action sequence, so a
prefix can be replayed. Limited discrepancy search uses that: run the policy greedily; on a
dead end at decision D, rerun with the second-ranked candidate forced at one decision near
D, then further back, then with two discrepancies, until a valid COMMIT or the deadline.
The policy orders the search; the search supplies the backtracking a single pass lacks.

Validity within the deadline is the number, read against the anytime baseline on the same
corpus at the same deadline.
"""
import argparse, json, os, sys, time
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

from _context import construction_context, qubit_budget
from candidate_features import FeatureContext
from train_constructor_rl import candidate_tuple, demands_realised
from train_prioritiser import Prioritiser


class Scorer:
    """Deterministic candidate ranking from the prioritiser, or from the generator's own
    order when no model is given (the unlearned control)."""

    def __init__(self, model, fc):
        self.model, self.fc = model, fc

    def prefer(self, env):
        if self.model is None:
            return None
        def f(v, q):
            with torch.no_grad():
                return float(self.model(torch.as_tensor(self.fc.pair(v, [q], env.state.chains, "PLACE"))))
        return f

    def ranking(self, dec, chains):
        legal = np.asarray(dec.legal_mask, dtype=bool)
        keep = np.array([c.opcode not in (Opcode.STOP, Opcode.RESTART, Opcode.COMMIT) for c in dec.candidates])
        ok = legal & keep
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            return []
        if self.model is None:
            return list(idx)
        feats = np.stack([self.fc.candidate(candidate_tuple(dec.candidates[i]), chains) for i in idx])
        with torch.no_grad():
            s = self.model(torch.as_tensor(feats)).numpy()
        return [int(idx[j]) for j in np.argsort(-s)]


def run(task, scorer, ctx, forced, max_steps, deadline_at):
    """One deterministic pass; ``forced`` maps decision index to the rank to take there.
    Returns (valid, dead_end_step, steps, demands)."""
    env = EmbeddingEnv(task, ctx, mode=Mode.CONSTRUCTION, initializer=None,
                       selector=fixed_strength_selector(), reward_reads=8)
    env.generator.prefer = scorer.prefer(env)
    dec = env.reset(0)
    steps = 0
    while isinstance(dec, DecisionState) and steps < max_steps and time.time() < deadline_at:
        chains = env.state.chains
        legal = [i for i, ok in enumerate(dec.legal_mask) if ok]
        commit = [i for i in legal if dec.candidates[i].opcode is Opcode.COMMIT]
        if commit and demands_realised(task, chains) >= 1.0:
            dec = env.step(dec, commit[0], evaluate_training_reward=False).next_decision_or_terminal
            steps += 1
            break
        ranked = scorer.ranking(dec, chains)
        if not ranked:
            return False, steps, steps, demands_realised(task, chains)
        rank = forced.get(steps, 0)
        if rank >= len(ranked):
            return False, steps, steps, demands_realised(task, chains)
        dec = env.step(dec, ranked[rank], evaluate_training_reward=False).next_decision_or_terminal
        steps += 1
    valid = bool(getattr(dec, "returned_valid", False)) and not isinstance(dec, DecisionState)
    return valid, steps, steps, demands_realised(task, env.state.chains)


def search(task, scorer, deadline, max_steps, window=12, max_rank=3):
    witness = {v: frozenset(c) for v, c in task.witness.items()}
    ctx = construction_context(qubit_budget(witness), task.logical.number_of_nodes(),
                               task.logical.number_of_edges())
    t0 = time.time()
    deadline_at = t0 + deadline
    attempts = 0
    valid, dead, steps, frac = run(task, scorer, ctx, {}, max_steps, deadline_at)
    attempts += 1
    best_frac = frac
    tried = set()
    # One discrepancy near the dead end, walking back; then a second one on top of the best.
    frontier = [{}]
    while not valid and time.time() < deadline_at:
        base = frontier[0]
        made = False
        for rank in range(1, max_rank):
            for d in range(max(0, dead - 1), max(-1, dead - 1 - window), -1):
                if d in base:
                    continue
                forced = dict(base); forced[d] = rank
                key = tuple(sorted(forced.items()))
                if key in tried:
                    continue
                tried.add(key)
                valid, dead2, steps, frac = run(task, scorer, ctx, forced, max_steps, deadline_at)
                attempts += 1
                made = True
                if frac > best_frac:
                    best_frac = frac
                    frontier.insert(0, forced)   # deeper progress: search from here next
                    dead = dead2
                    break
                if valid or time.time() >= deadline_at:
                    break
            if valid or time.time() >= deadline_at or made:
                break
        if not made:
            if len(frontier) > 1:
                frontier.pop(0)
            else:
                break
    return {"valid": valid, "attempts": attempts, "secs": time.time() - t0, "frac": best_frac}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--scorer", default="", help="prioritiser weights; empty for the unlearned order")
    ap.add_argument("--deadline", type=float, default=60.0)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    tasks = load_instances(a.corpus)
    if a.limit:
        tasks = tasks[: a.limit]
    model = None
    if a.scorer:
        blob = torch.load(a.scorer, map_location="cpu")
        model = Prioritiser(blob["width"]); model.load_state_dict(blob["state"]); model.eval()
    print(json.dumps({"corpus": a.corpus, "instances": len(tasks), "scorer": a.scorer or None,
                      "deadline": a.deadline}), flush=True)
    cells = defaultdict(list)
    for k, task in enumerate(tasks):
        family = task.lineage.rsplit("-", 1)[0].split("-", 1)[1].rsplit("-", 1)[0]
        witness = {v: frozenset(c) for v, c in task.witness.items()}
        fc = FeatureContext(task, qubit_budget(witness))
        r = search(task, Scorer(model, fc), a.deadline, a.max_steps)
        cells[family].append(r)
        print("  %3d/%d %-26s %s  attempts %3d  %5.0fs  demands %.3f"
              % (k + 1, len(tasks), task.name, "valid" if r["valid"] else "no   ", r["attempts"],
                 r["secs"], r["frac"]), flush=True)
    print("\n  %-16s %3s %8s %9s %8s" % ("cell", "n", "valid", "attempts", "demands"))
    for family in sorted(cells):
        rs = cells[family]
        print("  %-16s %3d %8.2f %9.1f %8.3f" % (family, len(rs), np.mean([r["valid"] for r in rs]),
                                                np.mean([r["attempts"] for r in rs]),
                                                np.mean([r["frac"] for r in rs])))
    print("\nSEARCH CONSTRUCT DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
