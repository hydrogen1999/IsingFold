"""Does multiplying the whole Hamiltonian by a constant change anything the evaluator sees?

Under the fixed-temperature model the paper's theory uses, it must: the common autoscale
compresses every coefficient toward the hardware limits and the Boltzmann weight of an excited
state rises. But the surrogate annealer, left to itself, derives its inverse-temperature range
from the programmed h and J, so scaling the Hamiltonian by c divides the schedule by c and the
product of beta and energy comes back unchanged. The theory's mechanism is cancelled by the
instrument before it can act.

This probe measures the size of that effect on the real evaluator, both ways, so the sampler
contract can be chosen with a number rather than an assumption. An external audit found the
cancellation with a two-qubit control; this runs it on corpus instances and real embeddings.
"""
import argparse, json, os, sys
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
from isingfold.rl.contracts import Context
from isingfold.rl.data.generate import load_instances
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.evaluate import first_commit_controller, run_controller

from _initializers import minorminer_initializer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="runs/v1/corpus")
    ap.add_argument("--lineages", type=int, default=12)
    ap.add_argument("--reads", type=int, default=512)
    ap.add_argument("--scales", default="1.0,0.25,0.0625")
    ap.add_argument("--beta", default="", help="registered range as lo,hi; empty means auto")
    a = ap.parse_args()

    beta = None
    if a.beta:
        lo, hi = (float(x) for x in a.beta.split(","))
        beta = (lo, hi)
    tasks = load_instances(a.corpus)[: a.lineages]
    mm = minorminer_initializer(10)
    scales = [float(x) for x in a.scales.split(",")]
    print(json.dumps({"corpus": a.corpus, "lineages": len(tasks), "scales": scales,
                      "beta_range": beta}), flush=True)

    rows = {s: [] for s in scales}
    for t in tasks:
        chains = mm(t.logical, t.host, 5000)
        if chains is None:
            continue

        def fixed(l, h, sd, c=chains):
            return c

        for s in scales:
            # Scaling the coupler and field limits scales the whole programmed Hamiltonian,
            # because the compiler autoscales to fit them.
            ctx = Context(qubit_cap=120, beta_range=beta,
                          field_limit=4.0 * s, coupler_limit=2.0 * s)
            try:
                out = run_controller([t], ctx, first_commit_controller, initializer=fixed,
                                     selector=fixed_strength_selector(), reward_reads=a.reads,
                                     repetitions=1, seed=4242)
            except Exception as exc:
                print("  %s at scale %g: %s" % (t.name, s, type(exc).__name__)); continue
            o = out[0]
            if o.returned_valid and o.utility is not None:
                rows[s].append(float(o.utility))

    base = scales[0]
    print("\n  %-10s %-10s %-12s %s" % ("scale", "utility", "n", "change from scale %g" % base))
    ref = float(np.mean(rows[base])) if rows[base] else float("nan")
    for s in scales:
        v = rows[s]
        m = float(np.mean(v)) if v else float("nan")
        print("  %-10g %-10.4f %-12d %+.4f" % (s, m, len(v), m - ref), flush=True)
    print("\n  A schedule that tracks the programmed coefficients makes these rows identical.")
    print("  A registered schedule makes the compressed rows worse, which is the effect the")
    print("  fixed-temperature theory predicts and the quantity a claim about it must report.")
    print("\nSAMPLER SCALE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
