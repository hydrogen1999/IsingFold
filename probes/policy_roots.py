"""The hybrid learned embedder: policy places roots, minorminer completes.

minorminer seeded with the witness's roots finishes at the fills where it fails from scratch
and random roots do nothing, so the information in the roots is what is missing. This runs
the construction environment with PLACE as the only action the policy takes, one root per
variable, ordered and scored by the imitation prioritiser, then hands the roots to
minorminer under a deadline. Arms: policy roots, no hint, witness roots. The number is
validity within the deadline against the no-hint baseline; the witness arm is the ceiling.
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

from _context import LEAN_QUOTAS, construction_context, qubit_budget
from candidate_features import FeatureContext
from placement_completion import witness_roots
from seeded_minorminer import attempt
from train_constructor_rl import candidate_tuple
from train_prioritiser import Prioritiser


def policy_roots(task, model, fc, use_prefer, max_steps=5000):
    """PLACE-only episode: every decision takes the best-scored legal PLACE."""
    witness = {v: frozenset(c) for v, c in task.witness.items()}
    quotas = {"place": 64, "route": 0, "grow": 0, "shrink": 0}
    ctx = construction_context(task.host.number_of_nodes(), task.logical.number_of_nodes(),
                               task.logical.number_of_edges(), quotas=quotas)
    env = EmbeddingEnv(task, ctx, mode=Mode.CONSTRUCTION, initializer=None,
                       selector=fixed_strength_selector(), reward_reads=8)
    if use_prefer:
        def prefer(v, q):
            with torch.no_grad():
                return float(model(torch.as_tensor(fc.pair(v, [q], env.state.chains, "PLACE"))))
        env.generator.prefer = prefer
    dec = env.reset(0)
    steps = 0
    while isinstance(dec, DecisionState) and steps < max_steps:
        chains = env.state.chains
        idx = [i for i, ok in enumerate(dec.legal_mask) if ok and dec.candidates[i].opcode is Opcode.PLACE]
        if not idx:
            break
        feats = np.stack([fc.candidate(candidate_tuple(dec.candidates[i]), chains) for i in idx])
        with torch.no_grad():
            s = model(torch.as_tensor(feats)).numpy()
        dec = env.step(dec, idx[int(np.argmax(s))], evaluate_training_reward=False).next_decision_or_terminal
        steps += 1
    chains = env.state.chains
    roots = {v: frozenset(c) for v, c in chains.items() if c}
    return roots, len(roots) / max(1, task.logical.number_of_nodes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--scorer", required=True)
    ap.add_argument("--deadline", type=float, default=60.0)
    ap.add_argument("--tries", type=int, default=10)
    ap.add_argument("--no-prefer", action="store_true")
    a = ap.parse_args()
    tasks = load_instances(a.corpus)
    blob = torch.load(a.scorer, map_location="cpu")
    model = Prioritiser(blob["width"]); model.load_state_dict(blob["state"]); model.eval()
    print(json.dumps({"corpus": a.corpus, "instances": len(tasks), "scorer": a.scorer,
                      "deadline": a.deadline, "prefer": not a.no_prefer}), flush=True)
    cells = defaultdict(lambda: defaultdict(list))
    for k, task in enumerate(tasks):
        family = task.lineage.rsplit("-", 1)[0].split("-", 1)[1].rsplit("-", 1)[0]
        witness = {v: frozenset(c) for v, c in task.witness.items()}
        fc = FeatureContext(task, qubit_budget(witness))
        t0 = time.time()
        roots, placed = policy_roots(task, model, fc, not a.no_prefer)
        place_secs = time.time() - t0
        agree = np.mean([next(iter(roots[v])) in witness[v] for v in roots]) if roots else 0.0
        hints = {"policy": roots, "none": None, "witness": witness_roots(task, witness)}
        row = []
        for arm, hint in hints.items():
            t0, ok, n = time.time(), None, 0
            budget = a.deadline - (place_secs if arm == "policy" else 0.0)
            while ok is None and time.time() - t0 < budget:
                n += 1
                ok = attempt(task, hint, 60_000 + 1000 * k + n, a.tries)
            cells[family][arm].append((ok is not None, time.time() - t0 + (place_secs if arm == "policy" else 0.0)))
            row.append("%s %s %4.0fs" % (arm, "valid" if ok else "no   ", time.time() - t0))
        print("  %3d/%d %-26s placed %.2f agree %.2f in %4.0fs | %s"
              % (k + 1, len(tasks), task.name, placed, agree, place_secs, " | ".join(row)), flush=True)
    print("\n  %-16s %3s %8s %8s %8s %9s %9s" % ("cell", "n", "policy", "none", "witness", "s:policy", "s:none"))
    for family in sorted(cells):
        c = cells[family]
        print("  %-16s %3d %8.2f %8.2f %8.2f %9.0f %9.0f"
              % (family, len(c["none"]), np.mean([r[0] for r in c["policy"]]),
                 np.mean([r[0] for r in c["none"]]), np.mean([r[0] for r in c["witness"]]),
                 np.mean([r[1] for r in c["policy"]]), np.mean([r[1] for r in c["none"]])))
    print("\nPOLICY ROOTS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
