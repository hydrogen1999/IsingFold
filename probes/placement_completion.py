"""Is placement the learnable part? Seed every variable at a witness root, complete greedily.

The imitation policy reaches about seventy percent of demands under budget and never a
valid embedding; what it cannot do is finish. This separates the two halves. Every variable
is placed at one qubit of its witness chain, the qubit that touches the most neighbouring
witness chains, as the environment's partial start. Then a fixed completion rule, no policy:
at each decision take the first legal ROUTE (bridges and meets before router paths), else the
first legal grow, else the first legal shrink; COMMIT as soon as every demand is met. Arms:

    greedy      the rule above, no witness in the loop after the roots
    hinted      the rule above with the witness as the generator's preference (upper bound)

If greedy completion from witness roots is high, a policy only has to learn roots, one
decision per variable. If it is low, completion is the hard part and the learned component
must be a search, not a scorer.
"""
import argparse, json, os, sys, time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
from isingfold.rl.contracts import DecisionState, Opcode
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import EmbeddingEnv, Mode, fixed_strength_selector

from _context import construction_context, qubit_budget
from train_constructor_rl import demands_realised

ORDER = {"bridge": 0, "meet": 1, "route": 2, "grow": 3, "shrink": 4}


def witness_roots(task, witness):
    roots = {}
    for v, chain in witness.items():
        def touches(q):
            return sum(1 for u in task.logical.neighbors(v)
                       for r in task.host.neighbors(q) if r in witness[u])
        roots[v] = frozenset({max(sorted(chain, key=str), key=touches)})
    return roots


def complete(task, witness, hint, max_steps, deadline):
    ctx = construction_context(qubit_budget(witness), task.logical.number_of_nodes(),
                               task.logical.number_of_edges())
    roots = witness_roots(task, witness)
    env = EmbeddingEnv(task, ctx, mode=Mode.CONSTRUCTION, initializer=lambda l, h, s: roots,
                       selector=fixed_strength_selector(), reward_reads=8)
    if hint:
        env.generator.prefer = lambda v, q: 1.0 if q in witness[v] else 0.0
    dec = env.reset(0)
    steps, t0 = 0, time.time()
    while isinstance(dec, DecisionState) and steps < max_steps and time.time() - t0 < deadline:
        chains = env.state.chains
        legal = [i for i, ok in enumerate(dec.legal_mask) if ok]
        commit = [i for i in legal if dec.candidates[i].opcode is Opcode.COMMIT]
        if commit and demands_realised(task, chains) >= 1.0:
            pick = commit[0]
        else:
            ranked = sorted(
                (i for i in legal if dec.candidates[i].opcode not in (Opcode.STOP, Opcode.RESTART, Opcode.COMMIT)),
                key=lambda i: (ORDER.get(str(getattr(dec.candidates[i], "provenance", "")).split(":")[0], 9), i),
            )
            if not ranked:
                break
            pick = ranked[0]
        dec = env.step(dec, pick, evaluate_training_reward=False).next_decision_or_terminal
        steps += 1
    valid = bool(getattr(dec, "returned_valid", False)) and not isinstance(dec, DecisionState)
    return {"valid": valid, "frac": demands_realised(task, env.state.chains), "steps": steps,
            "secs": time.time() - t0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--deadline", type=float, default=600.0)
    a = ap.parse_args()
    tasks = load_instances(a.corpus)
    print(json.dumps({"corpus": a.corpus, "instances": len(tasks)}), flush=True)
    cells = defaultdict(lambda: {"greedy": [], "hinted": []})
    for k, task in enumerate(tasks):
        family = task.lineage.rsplit("-", 1)[0].split("-", 1)[1].rsplit("-", 1)[0]
        witness = {v: frozenset(c) for v, c in task.witness.items()}
        g = complete(task, witness, False, a.max_steps, a.deadline)
        h = complete(task, witness, True, a.max_steps, a.deadline)
        cells[family]["greedy"].append(g); cells[family]["hinted"].append(h)
        print("  %3d/%d %-26s greedy %s %.2f %4d steps %5.0fs | hinted %s %.2f"
              % (k + 1, len(tasks), task.name, "valid" if g["valid"] else "no   ", g["frac"],
                 g["steps"], g["secs"], "valid" if h["valid"] else "no   ", h["frac"]), flush=True)
    print("\n  %-16s %3s %8s %8s %8s %8s" % ("cell", "n", "greedy", "demands", "hinted", "demands"))
    for family in sorted(cells):
        c = cells[family]
        print("  %-16s %3d %8.2f %8.3f %8.2f %8.3f"
              % (family, len(c["greedy"]), np.mean([r["valid"] for r in c["greedy"]]),
                 np.mean([r["frac"] for r in c["greedy"]]),
                 np.mean([r["valid"] for r in c["hinted"]]),
                 np.mean([r["frac"] for r in c["hinted"]])))
    print("\nPLACEMENT COMPLETION DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
