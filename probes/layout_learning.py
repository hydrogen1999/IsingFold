"""On-policy terminal-return learning and training-only root supervision.

This is Monte Carlo actor-critic (gamma=1), not PPO and not Q learning. A rollout
is used for exactly one optimizer update. A state value is an action-independent
baseline; the witness teacher teaches compatible root placements, not quality.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


def trajectory_loss(episodes, *, baseline="loo", entropy_coef=0.0, value_coef=0.5):
    """Return loss, diagnostics for [(terminal_return, decisions), ...] on ONE task.

    Actor gradients sum over decisions and average over episodes. Critic loss and
    normalized entropy average over decisions, so their scale is independent of n.
    Advantages are detached; returns are never normalized with the current action.
    """
    if baseline not in ("loo", "value"):
        raise ValueError("baseline must be loo or value")
    if len(episodes) < 2 and baseline == "loo":
        raise ValueError("leave-one-out needs at least two episodes")
    if not all(np.isfinite(x) and x >= 0 for x in (entropy_coef, value_coef)):
        raise ValueError("loss coefficients must be finite and nonnegative")
    returns = np.asarray([r for r, _ in episodes], dtype=float)
    if not len(returns) or not np.isfinite(returns).all():
        raise ValueError("finite terminal returns are required")
    losses, advantages, entropies, supports, values, targets = [], [], [], [], [], []
    for i, (ret, decisions) in enumerate(episodes):
        if not decisions:
            continue
        actor, critic, exploration = [], [], []
        loo = (returns.sum() - ret) / (len(returns) - 1) if len(returns) > 1 else 0.0
        for d in decisions:
            if baseline == "value":
                if d.value is None:
                    raise ValueError("value baseline requires a contextual actor-critic")
                target = d.value.new_tensor(float(ret))
                advantage = target - d.value.detach()
                critic.append(F.smooth_l1_loss(d.value, target))
                values.append(float(d.value.detach()))
                targets.append(float(ret))
            else:
                advantage = d.log_prob.new_tensor(float(ret - loo))
            actor.append(-advantage * d.log_prob)
            # Exactly zero for a deterministic one-action state. The coefficient
            # controls exploration and is not a per-qubit resource penalty.
            h = d.entropy / math.log(d.support_size) if d.support_size > 1 else d.entropy * 0
            exploration.append(h)
            advantages.append(float(advantage))
            entropies.append(float(h.detach()))
            supports.append(d.support_size)
        loss = torch.stack(actor).sum() - entropy_coef * torch.stack(exploration).mean()
        if critic:
            loss = loss + value_coef * torch.stack(critic).mean()
        losses.append(loss)
    var = float(np.var(targets)) if targets else 0.0
    metrics = {
        "episodes": len(episodes), "updated_episodes": len(losses),
        "reward_mean": float(returns.mean()), "reward_std": float(returns.std()),
        "advantage_std": float(np.std(advantages)) if advantages else 0.0,
        "normalized_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "support_mean": float(np.mean(supports)) if supports else 0.0,
        "value_explained_variance": 1 - float(np.var(np.asarray(targets) - values)) / var
        if var > 1e-12 else None,
    }
    return (torch.stack(losses).mean() if losses else None), metrics


def witness_prefix_loss(task, witness, model, fc, rng, *, support="all_free", temperature=1.0):
    """Feasibility warm start on explicitly supplied TRAINING witnesses only.

    Each prefix contains a random root in each previous variable's disjoint witness
    chain. All free roots are scored. The target is the SET of compatible witness
    roots, not a canonical root and not a claim that each root has equal quality.
    No teacher candidate is injected into the declared support. Unsupported steps
    are logged and still teacher-forced; this exposes legacy support exclusions.
    The caller must split lineages before passing tasks into this function.
    """
    from layout_policy import next_variable, root_support
    from seeded_minorminer import valid

    if not valid(witness, task.logical, task.host):
        raise ValueError("warm start requires a valid, disjoint training witness")
    if sum(len(c) for c in witness.values()) > fc.budget:
        raise ValueError("training witness exceeds the declared qubit budget")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    chains, losses, covered, total = {}, [], 0, 0
    while len(chains) < task.logical.number_of_nodes():
        v = next_variable(task, chains)
        roots = root_support(task, v, chains, strategy=support, rng=rng)
        positives = [i for i, q in enumerate(roots) if q in witness[v]]
        total += 1
        if positives:
            features = torch.as_tensor(np.stack([fc.pair(v, [q], chains) for q in roots]))
            parameter = next(model.parameters())
            features = features.to(device=parameter.device, dtype=parameter.dtype)
            logits = model(features) / temperature
            losses.append(torch.logsumexp(logits, 0) - torch.logsumexp(logits[positives], 0))
            covered += 1
        # Compatible teacher prefix, not the policy's potentially incompatible one.
        choices = sorted(witness[v], key=lambda q: (str(type(q)), str(q)))
        chains[v] = frozenset({choices[int(rng.integers(len(choices)))]})
    return (torch.stack(losses).mean() if losses else None), {
        "teacher_steps": total, "teacher_covered": covered,
        "teacher_coverage": covered / max(1, total),
    }
