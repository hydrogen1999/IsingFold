"""Data generation: feasible witnesses, certified energies and policy experience.

Spec: Rev2 section "Data generation". Five record types with different guarantees are kept
apart: an embedding witness proves existence, a logical optimum certificate proves an energy,
a structural counterfactual is an exact completion result in a registered domain, a quality
observation is a finite-read estimate, and an RL trajectory is exact search history. Actor
tensorisation uses an allowlist that excludes every one of the evaluator-only fields.
"""

from isingfold.rl.data.inkdrop import InkDropResult, ink_drop
from isingfold.rl.data.lineage import Lineage, Split, split_by_lineage
from isingfold.rl.data.planting import PlantedInstance, frustrated_loops

__all__ = [
    "InkDropResult",
    "Lineage",
    "PlantedInstance",
    "Split",
    "frustrated_loops",
    "ink_drop",
    "split_by_lineage",
]
