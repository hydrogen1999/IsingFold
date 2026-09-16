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

from isingfold.rl.contracts import DEFAULT_CAPS, RESERVE, Context

REGISTERED_BETA_RANGE: tuple[float, float] = (0.1, 2.0)
_UNSET = object()


def host_context(qubit_cap: int, beta_range=_UNSET, **kw) -> Context:
    if beta_range is _UNSET:
        beta_range = REGISTERED_BETA_RANGE
    work = max(RESERVE.feature_work, 32 * (qubit_cap + 64))
    return Context(qubit_cap=qubit_cap, reserve=replace(RESERVE, feature_work=work),
                   beta_range=beta_range, **kw)


def construction_context(qubit_cap: int, n_vars: int, n_edges: int, **kw) -> Context:
    """A Context whose horizon fits building an embedding of this instance from nothing.

    The registered default of 32 decisions was sized for improving a finished embedding. A
    construction needs one PLACE per variable, at most one ROUTE per logical edge, and a
    COMMIT, so the horizon scales with the instance; route expansions and materializations
    scale with it in the same proportion so the meter is not the binding constraint instead.
    """
    decisions = n_vars + n_edges + 8
    factor = max(1.0, decisions / DEFAULT_CAPS.decisions)
    caps = replace(DEFAULT_CAPS, decisions=decisions,
                   route_expansions=int(DEFAULT_CAPS.route_expansions * factor),
                   materializations=int(DEFAULT_CAPS.materializations * factor),
                   compiler_calls=int(DEFAULT_CAPS.compiler_calls * factor),
                   validator_calls=int(DEFAULT_CAPS.validator_calls * factor),
                   feature_work=int(DEFAULT_CAPS.feature_work * factor))
    quotas = {"place": 64, "route": 64}
    return host_context(qubit_cap, caps=caps, construction_quotas=quotas, **kw)
