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


LEAN_QUOTAS = {"place": 8, "route": 8, "grow": 4, "shrink": 2}


# A family whose quota is zero is never built (proposal.generate filters on quotas > 0), so
# the wide registration must name every family it wants, recovery included: without rewrite,
# repair and restart the policy can place and route but can never undo a mistake.
WIDE_QUOTAS = {"place": 288, "route": 128, "grow": 48, "shrink": 16,
               "rewrite": 16, "repair": 12, "restart": 4}


def scale_caps_for_steps(ctx: Context, n_vars: int, steps: int) -> Context:
    """Work caps that a construction of ``steps`` decisions on ``n_vars`` variables cannot
    exhaust. The registered caps were sized for improving a finished embedding of a small
    instance: at fill scale the per-step feature charge, 32 x (padded actions + variables),
    ran a 200k-per-32-decision budget out after a few hundred decisions, ending episodes
    by BUDGET before any 400-variable embedding could be finished."""
    from isingfold.rl.contracts import WorkVector
    per_step = {
        "feature_work": 32 * (ctx.padded_actions + n_vars + 8),
        "validator_calls": ctx.max_state_changing + 8,
        "compiler_calls": ctx.max_state_changing + ctx.max_commit + 16,
        "materializations": 4 * ctx.max_state_changing,
        "route_expansions": 64_000,
        "cut_edge_visits": 64_000,
    }
    caps = replace(ctx.caps, **{
        name: max(getattr(ctx.caps, name), (steps + 2) * per + getattr(ctx.reserve, name))
        for name, per in per_step.items()
    })
    return replace(ctx, caps=caps)


def construction_context(qubit_cap: int, n_vars: int, n_edges: int, quotas=None, wide: bool = False,
                         **kw) -> Context:
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
    if quotas is None:
        quotas = dict(WIDE_QUOTAS) if wide else {"place": 32, "route": 16, "grow": 12, "shrink": 4}
    ctx = host_context(qubit_cap, caps=caps, construction_quotas=dict(quotas), **kw)
    if wide:
        from isingfold.rl.contracts import WIDE_STATE_CHANGING, WIDE_SUFFIX
        ctx = replace(ctx, max_state_changing=WIDE_STATE_CHANGING,
                      padded_actions=WIDE_STATE_CHANGING + ctx.max_commit + 1,
                      context_version=ctx.context_version + WIDE_SUFFIX)
    return ctx


def qubit_budget(witness, slack: float = 1.10) -> int:
    """The qubit cap for building an instance whose witness is known: the witness's qubits
    with a margin for the router's detours. The environment refuses any action beyond it, so
    a policy cannot fill the host and dead-end, and the budget is a control variable as the
    advisor's section 5 asks."""
    import math
    return int(math.ceil(slack * sum(len(c) for c in witness.values())))
