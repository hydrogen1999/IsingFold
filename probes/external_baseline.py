"""The comparison Track B has never run: the learned embedder against the standard tool.

Every Track B number so far compares a policy to another policy inside our own environment: to
the protected initializer, or to a random masked controller given the same episodes. Neither is
what a practitioner would use. The tool a practitioner would use is minorminer, and until the
learned embedder is measured against it on the same instances, through the same evaluator, at a
matched budget, nothing has been shown about whether the learning is worth anything at all.

The design keeps every difference except the one under test:

  * the same held-out lineages, in the same order, as `deploy_modes.py`;
  * the same Context, strength selector and evaluator read budget;
  * the same best-of-K rule, which takes the largest realised utility over K attempts. That rule
    spends K evaluator blocks, so every arm here spends K of them and the comparison is matched
    in evaluator reads by construction. Wall clock is measured and reported separately, because
    minorminer and a forward pass do not cost the same and the reader is entitled to both.

minorminer enters as an *initializer*: `first_commit_controller` returns whatever the initializer
placed, so its embedding is scored by exactly the code path that scores the policy's.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

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


def minorminer_initializer(tries: int):
    """Stock minorminer as the environment's initializer, isolated variables placed after.

    `find_embedding` only returns variables that carry an edge, so a variable of degree zero has
    to be given a free qubit or the environment sees an incomplete placement and rejects it. That
    is bookkeeping, not help: a degree-zero variable has no coupling to realise.
    """
    import minorminer

    def _init(logical, host, seed):
        isolated = [v for v in logical.nodes() if logical.degree(v) == 0]
        emb = minorminer.find_embedding(list(logical.edges()), list(host.edges()),
                                        random_seed=seed % (2 ** 31), tries=tries)
        if not emb or set(emb) != set(logical.nodes()) - set(isolated):
            return None
        chains = {v: frozenset(c) for v, c in emb.items()}
        used = {q for c in chains.values() for q in c}
        free = iter(sorted(set(host.nodes()) - used))
        for v in isolated:
            try:
                chains[v] = frozenset([next(free)])
            except StopIteration:
                return None
        return chains
    return _init


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="runs/corpus_c4x10")
    ap.add_argument("--k", type=int, default=8, help="attempts per lineage, every arm")
    ap.add_argument("--lineages", type=int, default=60)
    ap.add_argument("--mm-tries", type=int, default=10, help="minorminer's own internal tries")
    ap.add_argument("--mm-k-extra", type=int, default=0,
                    help="a second minorminer arm with this many attempts, for the wall-clock "
                         "matched comparison: minorminer costs a fraction of a forward pass "
                         "here, so equal time buys it several times more attempts")
    ap.add_argument("--qubit-cap", type=int, default=120)
    ap.add_argument("--reward-reads", type=int, default=128)
    a = ap.parse_args()

    tasks = load_instances(a.corpus)
    split = json.loads((Path(a.corpus) / "splits.json").read_text())
    held = set(split["validation"]) | set(split["test"])
    dev = [t for t in tasks if t.lineage in held][: a.lineages]
    ctx = Context(qubit_cap=a.qubit_cap)
    print(json.dumps({"lineages": len(dev), "k": a.k, "mm_tries": a.mm_tries,
                      "qubit_cap": a.qubit_cap, "reward_reads": a.reward_reads}), flush=True)

    def per_instance(controller, initializer, seed):
        util, qubits, failures = {}, {}, 0
        for t in dev:
            try:
                out = run_controller([t], ctx, controller, initializer=initializer,
                                     selector=fixed_strength_selector(),
                                     reward_reads=a.reward_reads, repetitions=1, seed=seed)
            except Exception:
                failures += 1
                continue
            m = secondary_metrics(out)
            if m["valid_returns"]:
                util[t.name] = m["utility_mean"]
                q = [o.qubits for o in out if o.returned_valid]
                if q:
                    qubits[t.name] = float(np.mean(q))
            else:
                failures += 1
        return util, qubits, failures

    def best_of(controller_of, initializer_of, label, k=None):
        k = a.k if k is None else k
        started = time.time()
        acc, qub, failures = {}, {}, 0
        for j in range(k):
            u, q, f = per_instance(controller_of(j), initializer_of(j), 5000 + 97 * j)
            failures += f
            for name, v in u.items():
                if v > acc.get(name, -1e9):
                    acc[name] = v
                    qub[name] = q.get(name, float("nan"))
        secs = time.time() - started
        print("  %-26s solved %3d lineages, utility %.4f, qubits %.1f, "
              "%d failed attempts, %.1fs (%.2fs per lineage per attempt)"
              % (label, len(acc), float(np.mean(list(acc.values()))) if acc else float("nan"),
                 float(np.nanmean(list(qub.values()))) if qub else float("nan"),
                 failures, secs, secs / max(1, len(dev) * k)), flush=True)
        return acc, secs

    def report(label, arm, ref):
        keys = sorted(set(arm) & set(ref))
        if len(keys) < 5:
            print("  %-46s (too few paired lineages: %d)" % (label, len(keys)))
            return
        d = np.array([arm[i] - ref[i] for i in keys])
        rng = np.random.default_rng(0)
        bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(4000)]
        print("  %-46s n %3d  %+.4f [%+.4f, %+.4f]" % (label, len(d), d.mean(),
              np.percentile(bs, 2.5), np.percentile(bs, 97.5)), flush=True)

    router = router_initializer()
    mm = minorminer_initializer(a.mm_tries)

    print("\n== arms, each spending %d attempts and %d evaluator blocks per lineage" % (a.k, a.k))
    mm_k, mm_secs = best_of(lambda j: first_commit_controller, lambda j: mm, "minorminer best-of-K")
    rand_k, rand_secs = best_of(lambda j: random_masked_controller, lambda j: router,
                                "random policy best-of-K")
    mm_x, mm_x_secs = None, 0.0
    if a.mm_k_extra:
        mm_x, mm_x_secs = best_of(lambda j: first_commit_controller, lambda j: mm,
                                  "minorminer best-of-%d" % a.mm_k_extra, k=a.mm_k_extra)

    policies = {}
    for fam, ckpt in (("if-core", "runs/bestof2_if-core_s0/policy.pt"),
                      ("if-dual", "runs/bestof2_if-dual_s0/policy_r25.pt"),
                      ("if-mlp", "runs/bestof2_if-mlp_s0/policy.pt")):
        if not Path(ckpt).exists():
            print("  %s: no checkpoint at %s" % (fam, ckpt))
            continue
        model = build_model(fam, improvement_mode=True)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        model.eval()
        arm, secs = best_of(lambda j, m=model: torch_controller(m, None, greedy=False),
                            lambda j: router, "%s best-of-K" % fam)
        policies[fam] = (arm, secs)

    print("\n== paired differences in utility")
    report("minorminer best-of-K, against random best-of-K", mm_k, rand_k)
    for fam, (arm, _) in policies.items():
        report("%s best-of-K, against random best-of-K" % fam, arm, rand_k)
        report("%s best-of-K, against MINORMINER best-of-K" % fam, arm, mm_k)
        if mm_x is not None:
            report("%s best-of-K, against MINORMINER best-of-%d" % (fam, a.mm_k_extra), arm, mm_x)

    print("\n== wall clock for the whole arm, %d lineages x %d attempts" % (len(dev), a.k))
    print("  minorminer %.1fs, minorminer-extra %.1fs, random %.1fs, %s"
          % (mm_secs, mm_x_secs, rand_secs,
             ", ".join("%s %.1fs" % (f, s) for f, (_, s) in policies.items())))
    print("\nEXTERNAL BASELINE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
