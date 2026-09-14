"""IsingFold RL: the IF-Core typed graph actor-critic and its exact environment.

Implements ``documents/ISINGFOLD_MODEL_SPEC.md`` (model, environment contract, masked PPO)
and ``documents/IsingFold_Architecture_Rev2`` (objective IF-Q3-S0, data generation,
evaluation protocol). The environment owns validity, candidate materialisation and work
accounting; the model only reorders legal actions.

Submodules are importable independently: ``contracts``, ``program``, ``validate``,
``proposal``, ``env``, ``tensorize`` and ``data`` need no deep-learning runtime; ``model``,
``strength``, ``rollout`` and ``ppo`` require torch.
"""

from isingfold.rl.contracts import (
    ArchiveEntry,
    Candidate,
    Context,
    DecisionState,
    InitFailureRecord,
    Mode,
    Opcode,
    StepResult,
    TerminalReason,
    TerminalRecord,
    WorkVector,
)

__all__ = [
    "ArchiveEntry",
    "Candidate",
    "Context",
    "DecisionState",
    "InitFailureRecord",
    "Mode",
    "Opcode",
    "StepResult",
    "TerminalReason",
    "TerminalRecord",
    "WorkVector",
]
