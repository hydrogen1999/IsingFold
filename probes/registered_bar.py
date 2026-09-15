"""The comparison amendment 8 registered, run on whatever corpus and checkpoints you name.

The bar is the policy against a random masked controller **given the same number of episodes**,
not against the protected initializer. That distinction is the whole point: the best of forty
eight random rollouts beats returning the initializer on every development lineage by +0.26,
so an arm that only beats the initializer has shown that best-of-K works, not that the policy
does.

Every arm here is protected by the same initializer, which must be the one the policy was
trained under. Comparing a policy trained from a minorminer floor against a random controller
protected by a different floor would be comparing floors.

Two statistics are printed for each arm.

  registered   the maximum over K episode utilities, which is what the pre-registration fixed
               and what L-164 reported, so the numbers are comparable to it. Both arms take the
               maximum over the same number of measurements, so the winner's-curse inflation is
               matched between them.
  assessed     the embedding that maximum selected, re-measured on independent reads. This is
               the unbiased estimate of what the arm actually returns, and it is the one to
               quote outside the pre-registered comparison.
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
from isingfold.rl.contracts import Context
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.evaluate import (first_commit_controller, random_masked_controller,
                                   run_controller, torch_controller)
from isingfold.rl.model import build_model

from _initializers import pick_initializer

ASSESS_BASE = 90_000_000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--split", default="test", choices=["test", "validation", "both"])
    ap.add_argument("--lineages", type=int, default=60)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--initializer", default="minorminer", choices=["router", "minorminer"])
    ap.add_argument("--checkpoints", required=True, help="comma list of family=path")
    ap.add_argument("--qubit-cap", type=int, default=120)
    ap.add_argument("--reads", type=int, default=128)
    ap.add_argument("--assess-reads", type=int, default=512)
    a = ap.parse_args()

    tasks = load_instances(a.corpus)
    split = json.loads((Path(a.corpus) / "splits.json").read_text())
    held = (set(split["test"]) if a.split == "test"
            else set(split["validation"]) if a.split == "validation"
            else set(split["validation"]) | set(split["test"]))
    dev = [t for t in tasks if t.lineage in held][: a.lineages]
    ctx = Context(qubit_cap=a.qubit_cap)
    init = pick_initializer(a.initializer)
    print(json.dumps({"corpus": a.corpus, "split": a.split, "lineages": len(dev), "k": a.k,
                      "initializer": a.initializer}), flush=True)

    def episode(task, controller, seed):
        try:
            out = run_controller([task], ctx, controller, initializer=init,
                                 selector=fixed_strength_selector(), reward_reads=a.reads,
                                 repetitions=1, seed=seed)
        except Exception:
            return None
        o = out[0]
        if not o.returned_valid or o.utility is None or o.returned_embedding is None:
            return None
        return float(o.utility), o.returned_embedding

    def assess(task, chains, tag):
        def fixed(l, h, s):
            return chains
        try:
            out = run_controller([task], ctx, first_commit_controller, initializer=fixed,
                                 selector=fixed_strength_selector(),
                                 reward_reads=a.assess_reads, repetitions=1,
                                 seed=ASSESS_BASE + 13 * tag)
        except Exception:
            return None
        o = out[0]
        return float(o.utility) if o.returned_valid and o.utility is not None else None

    def arm(controller_of, k, tag):
        """Best-of-k by the episode's own measurement, then assess what it selected."""
        registered, assessed = {}, {}
        for t in dev:
            best = None
            for j in range(k):
                got = episode(t, controller_of(j), 5000 + 97 * j)
                if got is not None and (best is None or got[0] > best[0]):
                    best = got
            if best is None:
                registered[t.name] = assessed[t.name] = None
                continue
            registered[t.name] = best[0]
            assessed[t.name] = assess(t, best[1], tag)
        return registered, assessed

    def show(label, reg, asd):
        r = [v for v in reg.values() if v is not None]
        s = [v for v in asd.values() if v is not None]
        print("  %-34s registered %.4f   assessed %.4f   returned %d/%d"
              % (label, float(np.mean(r)) if r else float("nan"),
                 float(np.mean(s)) if s else float("nan"), len(r), len(dev)), flush=True)

    def paired(label, x, y, note):
        keys = sorted(set(x) & set(y))
        d = np.array([(0.0 if x[k] is None else x[k]) - (0.0 if y[k] is None else y[k])
                      for k in keys])
        if len(d) < 5:
            print("  %-46s (too few)" % label); return
        rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        print("  %-46s n %3d  %+.4f [%+.4f, %+.4f]  (%s)"
              % (label, len(d), d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), note),
              flush=True)

    print("\n== references, every arm on the same initializer")
    init_reg, init_asd = arm(lambda j: first_commit_controller, 1, 1)
    show("initializer, returned unchanged", init_reg, init_asd)
    r1_reg, r1_asd = arm(lambda j: random_masked_controller, 1, 2)
    show("random controller, one episode", r1_reg, r1_asd)
    rk_reg, rk_asd = arm(lambda j: random_masked_controller, a.k, 3)
    show("random controller, best-of-%d" % a.k, rk_reg, rk_asd)

    for tag, (fam, path) in enumerate(
            (kv.split("=", 1) for kv in a.checkpoints.split(",") if kv), start=10):
        if not Path(path).exists():
            print("\n== %s: no checkpoint at %s" % (fam, path)); continue
        model = build_model(fam, improvement_mode=True)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        print("\n== %s (%s)" % (fam, path))
        g_reg, g_asd = arm(lambda j, m=model: torch_controller(m, None, greedy=True), 1, tag)
        show("mode, one episode", g_reg, g_asd)
        s_reg, s_asd = arm(lambda j, m=model: torch_controller(m, None, greedy=False), 1, tag + 1)
        show("sampled, one episode", s_reg, s_asd)
        k_reg, k_asd = arm(lambda j, m=model: torch_controller(m, None, greedy=False), a.k,
                           tag + 2)
        show("sampled, best-of-%d" % a.k, k_reg, k_asd)
        paired("best-of-%d against the initializer" % a.k, k_reg, init_reg, "registered")
        paired("best-of-%d against RANDOM best-of-%d" % (a.k, a.k), k_reg, rk_reg,
               "registered, the bar amendment 8 fixed")
        paired("best-of-%d against RANDOM best-of-%d" % (a.k, a.k), k_asd, rk_asd,
               "assessed on independent reads")

    print("\nREGISTERED BAR DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
