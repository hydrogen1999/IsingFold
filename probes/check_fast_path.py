"""Prove the timing fast path changes nothing but the clock.

`ISINGFOLD_FAST_INTERNAL_ASSERTS=1` skips the environment's defensive invariant re-checks and
the defensive copy on cached return validations. Neither is supposed to affect what an episode
does, only how long it takes to do it. "Supposed to" is not evidence, so this prints a full
fingerprint of every episode and the two runs are diffed.

    python -u probes/check_fast_path.py > /tmp/checked.txt
    ISINGFOLD_FAST_INTERNAL_ASSERTS=1 python -u probes/check_fast_path.py > /tmp/fast.txt
    diff /tmp/checked.txt /tmp/fast.txt && echo identical

A difference of any kind means the flag is not a timing flag and no measurement may use it.
"""
import json, os, sys, time
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import torch
from isingfold.rl.contracts import Context
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.evaluate import (first_commit_controller, random_masked_controller,
                                   run_controller, torch_controller)
from isingfold.rl.model import build_model

from _initializers import minorminer_initializer

CORPUS = os.environ.get("CHECK_CORPUS", "runs/corpus_c4x10")
N = int(os.environ.get("CHECK_LINEAGES", "12"))
CKPT = os.environ.get("CHECK_CKPT", "runs/bestof3_if-core_s0/policy.pt")


def main() -> int:
    tasks = load_instances(CORPUS)
    split = json.loads((Path(CORPUS) / "splits.json").read_text())
    held = set(split["validation"]) | set(split["test"])
    dev = [t for t in tasks if t.lineage in held][:N]
    ctx = Context(qubit_cap=120)
    mm = minorminer_initializer(10)
    common = dict(initializer=mm, selector=fixed_strength_selector(), reward_reads=128,
                  repetitions=1)

    arms = [("first_commit", first_commit_controller), ("random", random_masked_controller)]
    if Path(CKPT).exists():
        model = build_model("if-core", improvement_mode=True)
        model.load_state_dict(torch.load(CKPT, map_location="cpu"))
        model.eval()
        torch.set_num_threads(2)
        arms.append(("policy", torch_controller(model, None, greedy=False)))

    print("fast path: %s" % os.environ.get("ISINGFOLD_FAST_INTERNAL_ASSERTS", "0"))
    for label, ctrl in arms:
        started = time.time()
        for t in dev:
            try:
                out = run_controller([t], ctx, ctrl, seed=11, **common)
            except Exception as exc:
                print("%s %s RAISED %s" % (label, t.name, type(exc).__name__)); continue
            for o in out:
                print("%s %s valid=%s utility=%.10f qubits=%d max_chain=%d decisions=%d "
                      "strength=%.6f reason=%s digest=%s"
                      % (label, o.instance, o.returned_valid, (o.utility if o.utility is not None
                         else float("nan")), o.qubits, o.max_chain, o.decisions,
                         o.selected_strength, o.reason, o.program_digest))
        print("# %s took %.2fs" % (label, time.time() - started))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
