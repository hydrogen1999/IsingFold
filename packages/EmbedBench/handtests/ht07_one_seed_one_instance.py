#!/usr/bin/env python3
"""Hand-test 07: a seed names exactly one instance, and different seeds name different ones.

What a person would do on paper
-------------------------------
1. Generate instance number 3 of a family. Write down its chains, its couplings, its fields and
   its ground energy.
2. Generate it again, in the same process and then in a *fresh* process with a different hash
   seed. Compare digit by digit. Anything that differs is not reproducible.
3. Generate instances 0 through 9. They must not be the same instance ten times: a seed that is
   mixed in badly, or dropped, produces a corpus with one instance in it.

Point 2 has a specific trap behind it. Python's `hash()` of a string is salted per process, so a
generator that seeds from it reproduces nothing across runs, and nothing announces the problem.
That is why the package derives per-instance seeds from sha256 instead.

What a failure means
--------------------
Either the results cannot be reproduced from the seeds the paper reports, or the corpus is
smaller than it claims because instances are silently duplicated.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import check, explain_and_exit_if_asked
from embedbench.evaluate import frustrated_loops, host_graph, ink_drop
from embedbench.structural import _instance_seed

SRC = str(Path(__file__).resolve().parents[1] / "src")


def fingerprint(i: int) -> str:
    """Everything a consumer of the corpus would see for instance `i`, hashed."""
    host = host_graph("chimera", 5)
    gseed = _instance_seed(12345, "near_capacity", i)
    planted = ink_drop(host, 14, 4, mode="near_capacity", seed=gseed)
    ising = frustrated_loops(planted.logical, alpha=0.6, seed=gseed)
    payload = {
        "seed": gseed,
        "chains": sorted((str(v), sorted(c)) for v, c in planted.chains.items()),
        "edges": sorted(map(str, planted.logical.edges())),
        "h": sorted((str(k), round(v, 12)) for k, v in ising.problem.h.items()),
        "j": sorted((str(k), round(v, 12)) for k, v in ising.problem.j.items()),
        "ground": round(ising.ground_energy, 12),
        "spins": sorted((str(k), v) for k, v in ising.spins.items()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def main() -> int:
    explain_and_exit_if_asked(__doc__)
    a, b = fingerprint(3), fingerprint(3)
    check(a == b, f"the same seed gave two instances in one process: {a} and {b}")
    print(f"  ok    instance 3 reproduces in-process: {a}")

    code = (f"import sys;sys.path.insert(0,{SRC!r});sys.path.insert(0,{str(Path(__file__).parent)!r});"
            "from ht07_one_seed_one_instance import fingerprint;print(fingerprint(3))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"})
    check(out.returncode == 0, f"the fresh-process probe failed: {out.stderr.strip()[:300]}")
    fresh = out.stdout.strip()
    check(fresh == a,
          f"a fresh process with PYTHONHASHSEED=0 built a different instance: {fresh} vs {a}")
    print(f"  ok    instance 3 reproduces in a fresh process under a different hash seed")

    prints = [fingerprint(i) for i in range(10)]
    check(len(set(prints)) == len(prints),
          f"only {len(set(prints))} distinct instances among 10 seeds")
    seeds = [_instance_seed(12345, "near_capacity", i) for i in range(10)]
    check(len(set(seeds)) == len(seeds), "two instance indices share a seed")
    print(f"  ok    10 indices give 10 distinct instances, 10 distinct seeds")

    other = [_instance_seed(12345, "elongated", i) for i in range(10)]
    check(not (set(other) & set(seeds)), "two families share an instance seed")
    print("  ok    two families of the same run do not share a seed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
