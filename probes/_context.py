"""A Context for a whole modern host, with the registered annealing schedule.

Two things the registry needs that its defaults do not give a modern-host experiment:

- The terminal reserve encodes a COMMIT-only support of at most 32 features per qubit of cap,
  sized for the 120-qubit Chimera experiments. Pegasus 6 has 680 qubits and Zephyr 4 has 576,
  so the reserve's feature work is raised in step with the cap.
- The annealing schedule. Left at None the surrogate annealer derives beta from the programmed
  coefficients and cancels the energy compression the objective is about (shrinking every
  coefficient to a sixteenth changes utility by +0.0000 under auto, by -0.4195 under the
  registered range; results/audit/). ADR-002 makes the registered range part of the objective,
  so it is the default here and has to be switched off by name.
"""
from dataclasses import replace

from isingfold.rl.contracts import RESERVE, Context

REGISTERED_BETA_RANGE: tuple[float, float] = (0.1, 2.0)
_UNSET = object()


def host_context(qubit_cap: int, beta_range=_UNSET, **kw) -> Context:
    if beta_range is _UNSET:
        beta_range = REGISTERED_BETA_RANGE
    work = max(RESERVE.feature_work, 32 * (qubit_cap + 64))
    return Context(qubit_cap=qubit_cap, reserve=replace(RESERVE, feature_work=work),
                   beta_range=beta_range, **kw)
