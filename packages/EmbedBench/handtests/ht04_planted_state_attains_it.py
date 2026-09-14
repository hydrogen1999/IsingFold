#!/usr/bin/env python3
"""Hand-test 04: the spin assignment the planting chose actually attains the stated energy.

What a person would do on paper
-------------------------------
1. Take the planted instance and the spin assignment it says it planted.
2. Evaluate the energy of that assignment by hand.
3. It must equal the shipped ground energy.

This is weaker than hand-test 03 and independent of it: 03 asks whether the stated energy is the
minimum over all assignments, 04 asks whether the assignment the generator believes it planted is
one that reaches it. A generator can get either right while getting the other wrong, and the two
failures need different fixes.

What a failure means
--------------------
The planting and its own witness disagree, so the instance has no certified solution even though
it is advertised as planted.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import check, explain_and_exit_if_asked
from embedbench.evaluate import frustrated_loops, host_graph, ink_drop
from embedbench.planted_ising import PlantingError


def main() -> int:
    explain_and_exit_if_asked(__doc__)
    host = host_graph("chimera", 5)
    checked = failures = 0
    for seed in range(1, 16):
        try:
            planted = ink_drop(host, 14, 4, mode="compact", seed=seed)
            if planted.logical.number_of_edges() < 6:
                continue
            ising = frustrated_loops(planted.logical, alpha=0.6, seed=seed)
        except PlantingError:
            continue
        energy = ising.energy_of_planted()
        checked += 1
        if abs(energy - ising.ground_energy) > 1e-9:
            failures += 1
            print(f"  FAIL  seed={seed}: the planted spins give {energy:.6f}, "
                  f"the instance claims {ising.ground_energy:.6f}")
        else:
            print(f"  ok    seed={seed}: planted spins attain {energy:.4f}, "
                  f"{ising.n_loops} loops of lengths {ising.loop_lengths[:6]}")
    check(checked >= 3, f"only {checked} instances were testable")
    check(failures == 0, f"{failures} instances disagree with their own planted state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
