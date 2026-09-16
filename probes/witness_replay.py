"""Can the construction environment reach the planted witness on the real corpora?

The unit test shows it on 16 and 100 qubits. This asks it on Pegasus 6 and Zephyr 4 at 300 to
500 variables: for each instance, drive the construction environment choosing at every
decision a candidate consistent with the witness (a PLACE whose root lies in the witness
chain, a ROUTE whose additions stay inside the witness chains, COMMIT once everything placed
lies inside the witness), and record whether a valid COMMIT is reached, in how many decisions
and seconds, and where it stops when it does not. Reaching the witness is what imitation
needs; the rate here is the fraction of the corpus a teacher can be built from.
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


def consistent(cand, witness, current):
    if cand.opcode is Opcode.PLACE:
        (v, chain), = cand.new_chains.items()
        return chain <= witness[v]
    if cand.opcode in (Opcode.ROUTE, Opcode.REWRITE_ONE):
        return all(chain <= witness[v] for v, chain in cand.new_chains.items())
    if cand.opcode is Opcode.COMMIT:
        return all(current.get(v) and current[v] <= c for v, c in witness.items())
    return False


def replay(task, witness, max_steps, hint, dump=None, scorer=None):
    ctx = construction_context(qubit_budget(witness), task.logical.number_of_nodes(),
                               task.logical.number_of_edges())
    first = max(witness, key=lambda v: (task.logical.degree(v), str(v)))
    seed_root = frozenset(witness[first])

    def start(logical, host, seed):
        return {first: seed_root}

    env = EmbeddingEnv(task, ctx, mode=Mode.CONSTRUCTION, initializer=start,
                       selector=fixed_strength_selector(), reward_reads=8)
    if hint:
        env.generator.prefer = lambda v, q: 1.0 if q in witness[v] else 0.0
    elif scorer is not None:
        env.generator.prefer = scorer(task, env)
    dec = env.reset(0)
    steps = 0
    t0 = time.time()
    while isinstance(dec, DecisionState) and steps < max_steps:
        current = env.state.chains
        choices = [i for i, (c, ok) in enumerate(zip(dec.candidates, dec.legal_mask))
                   if ok and consistent(c, witness, current)]
        if not choices:
            placed = sum(1 for v in witness if current.get(v))
            return {"ok": False, "steps": steps, "secs": time.time() - t0,
                    "why": "stuck", "placed": placed / len(witness)}
        commit = [i for i in choices if dec.candidates[i].opcode is Opcode.COMMIT]
        pick = commit[0] if commit else choices[0]
        if dump is not None:
            # Compact imitation record: the state as chains, every offered candidate as
            # (opcode, affected variables, qubits added), the legal mask, the teacher's
            # pick. Features for a candidate are recomputed from these at training time;
            # the full observation is 1.3 MB a step and is not what a prioritiser needs.
            dump.append({
                "task": task.name, "step": steps,
                "chains": {v: sorted(c, key=str) for v, c in current.items() if c},
                "candidates": [
                    (c.opcode.value, tuple(c.affected),
                     tuple(sorted({q for v, ch in c.new_chains.items()
                                   for q in ch - c.old_chains.get(v, frozenset())}, key=str)))
                    for c in dec.candidates
                ],
                "legal": tuple(bool(x) for x in dec.legal_mask),
                "teacher": pick,
            })
        dec = env.step(dec, pick, evaluate_training_reward=False).next_decision_or_terminal
        steps += 1
    if isinstance(dec, DecisionState):
        return {"ok": False, "steps": steps, "secs": time.time() - t0, "why": "horizon",
                "placed": sum(1 for v in witness if env.state.chains.get(v)) / len(witness)}
    ok = bool(getattr(dec, "returned_valid", False))
    return {"ok": ok, "steps": steps, "secs": time.time() - t0,
            "why": "valid" if ok else str(getattr(dec, "terminal_reason", "")), "placed": 1.0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--dump", default="", help="pickle path for compact imitation records")
    ap.add_argument("--scorer", default="", help="a trained prioritiser (train_prioritiser.py) "
                    "used as the generator's preference when --hint is not given")
    ap.add_argument("--hint", action="store_true",
                    help="let the generator prefer the witness's qubits before truncating its "
                         "offer: measures whether the grammar can express the witness at all, "
                         "as opposed to whether the unhinted heuristic order covers it")
    a = ap.parse_args()
    tasks = load_instances(a.corpus)
    if a.limit:
        tasks = tasks[: a.limit]
    print(json.dumps({"corpus": a.corpus, "instances": len(tasks), "hint": a.hint}), flush=True)
    cells = defaultdict(list)
    records = [] if a.dump else None
    scorer = None
    if a.scorer:
        import torch
        from candidate_features import FeatureContext
        from train_prioritiser import Prioritiser
        blob = torch.load(a.scorer, map_location="cpu")
        model = Prioritiser(blob["width"]); model.load_state_dict(blob["state"]); model.eval()

        def scorer(task, env):
            fc = FeatureContext(task, qubit_budget({v: frozenset(c) for v, c in task.witness.items()}))

            def prefer(v, q):
                with torch.no_grad():
                    x = fc.pair(v, [q], env.state.chains, "PLACE")
                    return float(model(torch.as_tensor(x)))
            return prefer
    for k, task in enumerate(tasks):
        family = task.lineage.rsplit("-", 1)[0].split("-", 1)[1].rsplit("-", 1)[0]
        witness = {v: frozenset(c) for v, c in task.witness.items()}
        r = replay(task, witness, a.max_steps, a.hint, records, scorer)
        if records is not None:
            import pickle
            with open(a.dump, "wb") as fh:
                pickle.dump({"corpus": a.corpus, "hint": a.hint, "records": records}, fh)
        cells[family].append(r)
        print("  %3d/%d %-26s %-7s steps %5d  %6.1fs  placed %.2f"
              % (k + 1, len(tasks), task.name, r["why"], r["steps"], r["secs"], r["placed"]),
              flush=True)
    print("\n  %-16s %3s %8s %8s %8s %s" % ("cell", "n", "reached", "steps", "secs", "stops"))
    for family in sorted(cells):
        rs = cells[family]
        stops = Counter(r["why"] for r in rs if not r["ok"])
        print("  %-16s %3d %8.2f %8.0f %8.1f %s"
              % (family, len(rs), np.mean([r["ok"] for r in rs]),
                 np.mean([r["steps"] for r in rs]), np.mean([r["secs"] for r in rs]),
                 dict(stops)))
    print("\nWITNESS REPLAY DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
