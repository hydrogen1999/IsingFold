#!/usr/bin/env python3
"""Hand-test 03: the ground energy the corpus states is the true minimum.

What a person would do on paper
-------------------------------
1. Take a planted instance small enough to enumerate, say 12 logical variables.
2. Write down every one of the 2^12 spin assignments.
3. Compute the energy of each: sum of h_v s_v plus sum of J_uv s_u s_v.
4. Take the smallest. It must equal the ground energy shipped with the instance, to the last
   digit that floating point allows.

This is the one number every result in the project is measured against: solve probability is the
fraction of reads that reach it. If it is wrong, every measurement is wrong by an unknown amount
and in an unknown direction.

What a failure means
--------------------
Either the planting is not achieving what it claims, or the shipped energy is stale. A shipped
energy *below* the true minimum makes the problem look unsolvable; *above* it makes excited
states count as solutions.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import brute_force_ground_energy, check, explain_and_exit_if_asked
from embedbench.evaluate import frustrated_loops, host_graph, ink_drop
from embedbench.planted_ising import PlantingError


def main() -> int:
    explain_and_exit_if_asked(__doc__)
    host = host_graph("chimera", 5)
    checked = failures = 0
    for seed in range(1, 12):
        try:
            planted = ink_drop(host, 12, 4, mode="compact", seed=seed)
            if planted.logical.number_of_edges() < 6:
                continue          # hand-test 05 covers edgeless plantings
            ising = frustrated_loops(planted.logical, alpha=0.6, seed=seed)
        except (PlantingError, Exception) as exc:  # noqa: BLE001 - reported, not swallowed
            if isinstance(exc, PlantingError):
                continue
            raise
        exact, state = brute_force_ground_energy(ising.problem)
        checked += 1
        gap = abs(exact - ising.ground_energy)
        if gap > 1e-9:
            failures += 1
            print(f"  FAIL  seed={seed}: shipped {ising.ground_energy:.6f}, "
                  f"enumeration says {exact:.6f}, gap {gap:.2e}")
        else:
            print(f"  ok    seed={seed}: {ising.problem.n} variables, "
                  f"{len(ising.problem.j)} couplings, ground {exact:.4f} confirmed by enumeration")
    check(checked >= 3, f"only {checked} instances were enumerable; the test proved little")
    check(failures == 0, f"{failures} instances state a ground energy enumeration disagrees with")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
