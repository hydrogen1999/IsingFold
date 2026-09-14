#!/usr/bin/env python3
"""Hand-test 08: the chain-strength grid contains the optimum it is used to find.

What a person would do on paper
-------------------------------
Solve probability is reported at each embedding's *own best* chain strength, taken over a small
grid. That is only an honest number if the grid brackets the real optimum. If the best point sits
at an endpoint, the true optimum is outside the grid and the reported figure is a lower bound of
unknown slack, so comparing two embeddings can amount to comparing two grid edges.

So:
1. Take an instance and an embedding.
2. Sweep the shipped five-point grid and note where the best point is.
3. Sweep a grid four times finer and four times wider, with the same reads and the same seed.
4. Check the fine optimum is inside the coarse grid's range, and the coarse best is close to the
   fine best. Then the coarse grid is not costing anything real.
5. Check that the coarse optimum is interior for most instances, which is what
   `EmbeddingScore.strength_is_interior` is there to report.

What a failure means
--------------------
The headline metric measures the grid rather than the embedding, and the direction of the error
is not even constant across instances.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from _common import check, explain_and_exit_if_asked
from embedbench.embedding import Embedding
from embedbench.evaluate import frustrated_loops, host_graph, ink_drop
from embedbench.structural import _instance_seed
from embedbench.surrogate import default_strength_grid, solve_probability_at

READS = 400
SWEEPS = 200


def sweep(embedding, problem, e0, grid, seed):
    # The same per-point seed rule the library uses in `p_solve`, so the two sweeps are
    # comparable and the sampler's own seed range is respected.
    return [(f, solve_probability_at(embedding, problem, e0, f, num_reads=READS,
                                     num_sweeps=SWEEPS, seed=(seed + 31 * k) % (2 ** 31)).p_solve)
            for k, f in enumerate(grid)]


def main() -> int:
    explain_and_exit_if_asked(__doc__)
    host = host_graph("chimera", 5)
    boundary = 0
    checked = 0
    for i in range(6):
        gseed = _instance_seed(12345, "near_capacity", i)
        planted = ink_drop(host, 14, 4, mode="near_capacity", seed=gseed)
        ising = frustrated_loops(planted.logical, alpha=0.6, seed=gseed)
        problem, e0 = ising.problem, ising.ground_energy

        coarse = default_strength_grid(problem, 5)
        unit = coarse[0] / 0.25
        fine = [float(unit * m) for m in np.geomspace(0.0625, 32.0, 33)]

        embedding = Embedding.from_chains(planted.chains, host, problem)
        cs = sweep(embedding, problem, e0, coarse, gseed + 1)
        fs = sweep(embedding, problem, e0, fine, gseed + 1)
        best_c = max(cs, key=lambda t: t[1])
        best_f = max(fs, key=lambda t: t[1])
        interior = best_c[0] not in (coarse[0], coarse[-1])
        checked += 1
        boundary += 0 if interior else 1
        inside = coarse[0] <= best_f[0] <= coarse[-1]
        print(f"  i={i}  coarse best p={best_c[1]:.3f} at F={best_c[0]:.3f}"
              f" ({'interior' if interior else 'AT AN ENDPOINT'});"
              f"  fine best p={best_f[1]:.3f} at F={best_f[0]:.3f}"
              f" ({'inside' if inside else 'OUTSIDE'} the coarse range)")
        check(inside, f"instance {i}: the true optimum F={best_f[0]:.3f} lies outside the "
                      f"shipped grid [{coarse[0]:.3f}, {coarse[-1]:.3f}]")
        check(best_f[1] - best_c[1] <= 0.12,
              f"instance {i}: the coarse grid loses {best_f[1]-best_c[1]:.3f} of solve "
              f"probability, so the reported number is a grid artefact")
    check(boundary <= checked // 3,
          f"{boundary} of {checked} instances put their optimum on a grid endpoint")
    print(f"  ok    {checked - boundary} of {checked} optima are interior, and every fine-grid "
          f"optimum falls inside the shipped range")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
