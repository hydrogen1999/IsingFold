"""Does the deployment rule sample or take the best action, and does it matter?

`torch_controller` defaults to categorical sampling at temperature one, which is what every
evaluation in this project has used, including the two ladders. A policy that has learned a
preference but kept its entropy will score close to a random controller under sampling and far
above it under argmax. This measures both on the same checkpoints.
"""
import json, os, sys
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import torch
from isingfold.rl.contracts import Context
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.evaluate import (first_commit_controller, random_masked_controller,
                                   run_controller, secondary_metrics, torch_controller)
from isingfold.rl.model import build_model
from isingfold.rl.proposal import router_initializer

tasks = load_instances("runs/corpus_c4x10")
split = json.loads(Path("runs/corpus_c4x10/splits.json").read_text())
dev = set(split["validation"]) | set(split["test"])
dev_tasks = [t for t in tasks if t.lineage in dev][:60]
ctx = Context(qubit_cap=120)
common = dict(initializer=router_initializer(), selector=fixed_strength_selector(),
              reward_reads=128, seed=17, repetitions=1)

def u(controller):
    return secondary_metrics(run_controller(dev_tasks, ctx, controller, **common))["utility_mean"]

print("returning the initializer : %.4f" % u(first_commit_controller), flush=True)
print("random masked             : %.4f" % u(random_masked_controller), flush=True)
for fam, ckpt in (("if-core", "runs/ladder3_if-core/if-core_s2.pt"),
                  ("if-dual", "runs/ladder3_if-dual/if-dual_s1.pt"),
                  ("if-mlp", "runs/ladder3_if-mlp/if-mlp_s0.pt")):
    if not os.path.exists(ckpt):
        continue
    m = build_model(fam, improvement_mode=True)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    s = u(torch_controller(m, None, greedy=False))
    g = u(torch_controller(m, None, greedy=True))
    print("%-8s sampled %.4f   greedy %.4f   (greedy minus sampled %+.4f)" % (fam, s, g, g - s), flush=True)
