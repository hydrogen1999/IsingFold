"""Policy-controlled construction from empty chains, including the final COMMIT.

The graph environment proposes and validates complete macro-action outcomes. The actor
scores the entire legal support once; it never supplies a hidden constructor or alters
the proposal shortlist. Qubits and work are constraints, not reward penalties.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

import numpy as np
import torch

from isingfold.rl.contracts import DecisionState, Mode, Opcode, TerminalReason, TerminalRecord
from isingfold.rl.env import EmbeddingEnv, fixed_strength_selector

from _context import construction_context, scale_caps_for_steps


CONSTRUCTION_QUOTAS = {
    "place": 16, "route": 16, "grow": 8, "shrink": 4,
    "rewrite": 8, "repair": 8, "restart": 4,
}


@dataclass(frozen=True)
class ConstructorDecision:
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor | None
    support_size: int
    opcode: str
    provenance: str
    action: int
    base_value: torch.Tensor | None = None


def candidate_tuple(candidate):
    """Compatibility adapter for the explicitly selected legacy feature ablation."""
    added = {q for v, ch in candidate.new_chains.items()
             for q in ch - candidate.old_chains.get(v, frozenset())}
    return candidate.opcode.value, tuple(candidate.affected), tuple(sorted(added, key=str))


def demands_realised(task, chains):
    edges = list(task.logical.edges())
    if not edges:
        return 1.0
    return sum(any(task.host.has_edge(a, b)
                   for a in chains.get(u, ()) for b in chains.get(v, ()))
               for u, v in edges) / len(edges)


def _progress(task, chains):
    placed = sum(bool(chains.get(v)) for v in task.logical) / max(1, len(task.logical))
    # With no demands, placement is the only construction progress. This convention
    # keeps the potential zero at the empty initial state, including isolated nodes.
    demand = demands_realised(task, chains) if task.logical.number_of_edges() else placed
    return 0.5 * (placed + demand)


def _context(task, cap, max_steps, reward_reads, quotas, restart_allowance, wide=False):
    ctx = construction_context(
        cap, task.logical.number_of_nodes(), task.logical.number_of_edges(),
        quotas=(None if wide else CONSTRUCTION_QUOTAS) if quotas is None else quotas,
        restart_allowance=restart_allowance, wide=wide,
    )
    # Leave one decision beyond the caller's horizon: reaching max_steps does not
    # artificially turn the last policy support into a forced COMMIT-only state.
    count = max(ctx.caps.decisions, max_steps + 1)
    factor = count / ctx.caps.decisions
    caps = replace(ctx.caps, **{
        name: math.ceil(value * factor)
        for name, value in ctx.caps.as_dict().items()
    })
    reserve = replace(ctx.reserve, evaluator_reads=max(ctx.reserve.evaluator_reads, reward_reads))
    caps = replace(caps, evaluator_reads=max(caps.evaluator_reads, reserve.evaluator_reads))
    ctx = replace(ctx, caps=caps, reserve=reserve, n_est_reads=reward_reads,
                  context_version=ctx.context_version + "-independent-constructor-v1")
    return scale_caps_for_steps(ctx, task.logical.number_of_nodes(), max_steps)


def episode(task, model, fc, temperature, max_steps, rng, deadline, train=True, *,
            qubit_cap=None, objective="quality", reward_reads=256, shaping_coef=0.0,
            measure=None, quotas=None, restart_allowance=2, evaluate_reward=None,
            build_observation=False, initializer=None, wide=False, state_value_model=None):
    """Construct one embedding; only an on-time, actor-selected COMMIT can succeed.

    ``train`` selects stochastic sampling versus greedy inference. ``evaluate_reward``
    defaults to ``train`` and separately controls terminal quality measurement; inference
    can therefore run without a certified energy or any evaluator-only information.
    Quality measurement occurs after construction and has its own time receipt. There
    is no unselected archive fallback, forced COMMIT, witness budget or external router.

    Potential shaping uses gamma=1 and Phi(terminal)=0 on *every* ending, including
    truncations. ``potentials`` contains pre-action Phi, one entry per decision, and
    ``return == sum(rewards) == base_return - Phi(initial)`` for nonempty episodes.
    Empty-start construction has Phi(initial)=0; a training prefix need not.
    A requested quality label for a valid COMMIT must exist: evaluator failure
    must not silently turn a successful trajectory into a zero-reward failure.
    An optional independent ``state_value_model`` predicts the unshaped terminal
    base return before action sampling, using public state context and legal rows.
    Keep its parameters fixed throughout collection of a loss batch.
    """
    started = time.monotonic()
    if objective not in {"quality", "feasibility"}:
        raise ValueError("objective must be quality or the explicit feasibility curriculum")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 0:
        raise ValueError("max_steps must be a nonnegative integer")
    if not math.isfinite(deadline) or deadline < 0:
        raise ValueError("deadline must be finite and nonnegative")
    if not math.isfinite(shaping_coef) or shaping_coef < 0:
        raise ValueError("shaping_coef must be finite and nonnegative")
    if isinstance(reward_reads, bool) or not isinstance(reward_reads, int) or reward_reads <= 0:
        raise ValueError("reward_reads must be a positive integer")
    if evaluate_reward is None:
        evaluate_reward = bool(train)
    if type(evaluate_reward) is not bool:
        raise ValueError("evaluate_reward must be Boolean")
    cap = task.host.number_of_nodes() if qubit_cap is None else qubit_cap
    if isinstance(cap, bool) or not isinstance(cap, int) or not 0 < cap <= len(task.host):
        raise ValueError("qubit_cap must be a positive integer within the active host")
    if hasattr(fc, "budget") and fc.budget != cap:
        raise ValueError("feature context and environment must use the same public qubit cap")
    ctx = _context(task, cap, max_steps, reward_reads, quotas, restart_allowance, wide=wide)
    # The constructor scores candidates from ``fc``; the environment's tensor observation
    # is never read here and dominates the step cost on large hosts, so it is off by default.
    # ``initializer`` is a training-time curriculum hook only: a partial embedding to build
    # from (some chains placed, the rest empty). Deployment and evaluation leave it unset.
    env = EmbeddingEnv(task, ctx, mode=Mode.CONSTRUCTION, initializer=initializer,
                       selector=fixed_strength_selector(), reward_reads=reward_reads,
                       build_observation=build_observation)
    # Set before reset binds the first support. The context version registers this
    # support extension; valid embeddings can still trade spare capacity for quality.
    env.generator.allow_satisfied_growth = True
    seed = int(rng.integers(0, 2 ** 31))
    current = env.reset(seed)
    decisions, potentials = [], []
    last_opcode = None
    reason = None
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else torch.device("cpu")
    dtype = parameter.dtype if parameter is not None else torch.float32

    while isinstance(current, DecisionState):
        if time.monotonic() - started >= deadline:
            reason = "DEADLINE"
            break
        if len(decisions) >= max_steps:
            reason = "HORIZON"
            break
        legal_indices = [i for i, allowed in enumerate(current.legal_mask) if allowed]
        if not legal_indices:
            reason = "NO_LEGAL_ACTION"
            break
        # Pool only legal rows, with identical state and horizon context for each row.
        # In particular COMMIT competes with refinements and STOP/RESTART stay available.
        chains = env.state.chains
        rows = []
        for index in legal_indices:
            candidate = current.candidates[index]
            if hasattr(fc, "observe"):
                row = fc.observe(candidate, chains, state=env.state, ctx=ctx,
                                 steps_left=max_steps - len(decisions), max_steps=max_steps)
            else:
                row = fc.candidate(candidate_tuple(candidate), chains)
            rows.append(row)
        feats = torch.as_tensor(np.stack(rows), device=device, dtype=dtype)
        if time.monotonic() - started >= deadline:
            reason = "DEADLINE"
            break
        with torch.set_grad_enabled(bool(train) and torch.is_grad_enabled()):
            base_value = None
            if state_value_model is not None:
                base_value = state_value_model(
                    feats.detach(), remaining_fraction=(max_steps - len(decisions)) / max_steps,
                    progress=_progress(task, chains))
            if hasattr(model, "distribution_value"):
                dist, value = model.distribution_value(feats, temperature)
            else:
                dist = torch.distributions.Categorical(logits=model(feats) / temperature)
                value = None
            if train:
                probabilities = dist.probs.detach().cpu().numpy().astype(np.float64)
                probabilities /= probabilities.sum()
                action = int(rng.choice(len(legal_indices), p=probabilities))
            else:
                action = int(dist.logits.argmax())
            chosen_index = legal_indices[action]
            candidate = current.candidates[chosen_index]
            choice = ConstructorDecision(
                log_prob=dist.log_prob(torch.tensor(action, device=device)),
                entropy=dist.entropy(), value=value, support_size=len(legal_indices),
                opcode=candidate.opcode.value, provenance=candidate.provenance,
                action=chosen_index,
                base_value=base_value,
            )
        if time.monotonic() - started >= deadline:
            reason = "DEADLINE"
            break
        potentials.append(shaping_coef * _progress(task, chains))
        decisions.append(choice)
        last_opcode = candidate.opcode
        current = env.step(current, chosen_index,
                           evaluate_training_reward=False).next_decision_or_terminal

    construction_secs = time.monotonic() - started
    # Includes reset, feature generation, model evaluation, and terminal compilation.
    if construction_secs >= deadline:
        reason = "DEADLINE"
    terminal = current if isinstance(current, TerminalRecord) else None
    if reason is None:
        reason = terminal.terminal_reason.value if terminal is not None else "NO_TERMINAL"
    valid = bool(reason == TerminalReason.COMMIT.value
                 and last_opcode is Opcode.COMMIT and terminal is not None
                 and terminal.returned_valid and terminal.embedding is not None)
    if not valid:
        terminal = None
    residual = None
    quality_measured = False
    quality_status = "invalid" if not valid else "not_requested"
    reward_started = time.monotonic()
    if objective == "feasibility":
        base_return = 1.0 if valid else 0.25 * _progress(task, env.state.chains)
    else:
        base_return = 0.0
        if valid and evaluate_reward:
            from constructor_objective import measure_terminal, quality_utility
            evaluator = measure_terminal if measure is None else measure
            residual = evaluator(task, terminal, int(rng.integers(0, 2 ** 31)), reward_reads)
            if residual is None:
                raise ValueError("valid quality training episode is missing its requested residual label")
            residual = float(residual)
            if not math.isfinite(residual):
                raise ValueError("quality evaluator must return a finite residual")
            base_return = float(quality_utility(task, residual))
            quality_measured = True
            quality_status = "measured"
    reward_secs = time.monotonic() - reward_started
    # The last successor is absorbing even after timeout/horizon. This subtracts
    # accumulated shaping progress instead of accidentally paying for a dead end.
    successors = potentials[1:] + [0.0] if potentials else []
    rewards = [after - before for before, after in zip(potentials, successors)]
    if rewards:
        rewards[-1] += base_return
    togo, accumulated = [], 0.0
    for reward in reversed(rewards):
        accumulated += reward
        togo.append(accumulated)
    togo.reverse()
    return {
        "valid": valid, "embedding": terminal.embedding if terminal else None,
        "terminal": terminal, "selected_program": terminal.selected_program if terminal else None,
        "reason": reason,
        "frac": demands_realised(task, terminal.embedding if terminal else env.state.chains),
        "steps": len(decisions), "secs": construction_secs,
        "construction_secs": construction_secs, "reward_secs": reward_secs,
        "total_secs": time.monotonic() - started, "decisions": decisions,
        "logps": [decision.log_prob for decision in decisions],
        "rewards": rewards, "togo": togo, "return": sum(rewards),
        "base_return": base_return, "potentials": potentials, "terminal_potential": 0.0,
        "objective": objective, "residual": residual, "quality_measured": quality_measured,
        "quality_coverage": float(quality_measured), "quality_status": quality_status,
        "unmeasured_valid": valid and objective == "quality" and not quality_measured,
        "qubit_cap": cap, "seed": seed, "allow_satisfied_growth": True,
        "context_version": ctx.context_version,
    }
