"""How often does resolving COMMIT as workspace-plus-new_chains name the wrong embedding?

COMMIT carries no new_chains. The environment returns `archive[archive_ref].chains`. Resolving it
the workspace way therefore labels a commit with whatever the workspace happens to hold, and the
two coincide exactly at the root, where the protected entry is the initial embedding. So the
error is invisible at the first state anyone checks and common afterwards.

This counts it on the real corpus rather than on fixtures, because how much a bug matters is a
property of the data it ran on. It is kept so the claim in results/README.md has something behind
it that can be re-run.
"""
import argparse, os, sys
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
from isingfold.rl.contracts import Context, Opcode
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import (LEGACY_ONLINE_INITIALIZER_RESTARTS_V1, EmbeddingEnv, Mode,
                              fixed_strength_selector)

from _initializers import minorminer_initializer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="runs/v1/corpus")
    ap.add_argument("--lineages", type=int, default=60)
    ap.add_argument("--states", type=int, default=2)
    ap.add_argument("--qubit-cap", type=int, default=120)
    a = ap.parse_args()

    tasks = load_instances(a.corpus)[: a.lineages]
    ctx = Context(qubit_cap=a.qubit_cap)
    mm = minorminer_initializer(10)
    rng = np.random.default_rng(0)
    states = commits = wrong = root_rows = root_wrong = 0
    for k, t in enumerate(tasks):
        env = EmbeddingEnv(t, ctx, mode=Mode.IMPROVEMENT, initializer=mm,
                           selector=fixed_strength_selector(), reward_reads=128, seed=5000 + k,
                           improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1)
        dec = env.reset(5000 + k)
        for step in range(a.states):
            if not hasattr(dec, "candidates"):
                break
            states += 1
            ws = {n: frozenset(c) for n, c in env.state.chains.items()}
            for cand, ok in zip(dec.candidates, dec.legal_mask):
                if not ok or cand.opcode is not Opcode.COMMIT:
                    continue
                ref = cand.archive_ref
                if ref is None or ref >= len(env.state.archive):
                    continue
                commits += 1
                entry = {n: frozenset(c) for n, c in env.state.archive[ref].chains.items()}
                stale = dict(ws)
                for node, chain in cand.new_chains.items():
                    stale[node] = frozenset(chain)
                if step == 0:
                    root_rows += 1
                if stale != entry:
                    wrong += 1
                    if step == 0:
                        root_wrong += 1
            legal = [i for i, o in enumerate(dec.legal_mask) if o]
            if not legal:
                break
            nxt = env.step(dec, legal[int(rng.integers(0, len(legal)))],
                           evaluate_training_reward=False).next_decision_or_terminal
            if not hasattr(nxt, "candidates"):
                break
            dec = nxt

    print("states examined              %d" % states)
    print("legal COMMIT rows            %d" % commits)
    print("named the wrong embedding    %d  (%.1f%%)"
          % (wrong, 100.0 * wrong / max(1, commits)))
    print("  at the root state          %d of %d" % (root_wrong, root_rows))
    print("  after at least one step    %d of %d" % (wrong - root_wrong, commits - root_rows))
    print("\nCOMMIT LABEL CHECK DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
