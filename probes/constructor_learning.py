"""Monte Carlo actor--critic for empty-start constructive embedding episodes.

The environment uses gamma=1 potential shaping with terminal potential zero,
including timeouts. Thus G_t = base_return - Phi(s_t). A leave-one-out baseline
uses other episodes' BASE returns minus the current state's potential. It never
includes this episode's sampled return and never divides actor credit by length.
No teacher, resource objective, router or annealing-quality approximation is used
by this loss. It consumes rewards supplied by the declared rollout objective.
"""
from collections.abc import Mapping
import math

import numpy as np
import torch
import torch.nn.functional as F


def _field(record, name):
    return record[name] if isinstance(record, Mapping) else getattr(record, name)


def _scalar_tensor(value, name):
    if not isinstance(value, torch.Tensor) or value.ndim != 0 or not torch.isfinite(value):
        raise ValueError(f"{name} must be a finite scalar tensor")
    return value


def _episode_data(episode):
    decisions = episode["decisions"]
    arrays = {}
    for name in ("rewards", "togo", "potentials"):
        values = np.asarray(episode[name], dtype=float)
        if values.ndim != 1 or len(values) != len(decisions) or not np.isfinite(values).all():
            raise ValueError(f"{name} must contain one finite value per decision")
        arrays[name] = values
    for name in ("return", "base_return"):
        if not np.isscalar(episode[name]) or not np.isfinite(episode[name]):
            raise ValueError(f"{name} must be finite")
    if "terminal_potential" in episode and episode["terminal_potential"] != 0:
        raise ValueError("terminal potential must be zero, including timeouts")
    if len(decisions):
        expected = np.cumsum(arrays["rewards"][::-1])[::-1]
        if not np.allclose(arrays["togo"], expected, rtol=1e-5, atol=1e-6):
            raise ValueError("togo does not match the rewards' reverse cumulative sum")
        if not np.isclose(episode["return"], expected[0], rtol=1e-5, atol=1e-6):
            raise ValueError("episode return does not match its rewards")
        if not np.allclose(arrays["togo"], episode["base_return"] - arrays["potentials"],
                           rtol=1e-5, atol=1e-6):
            raise ValueError("gamma=1 potential shaping must telescope to base_return - Phi")
    return decisions, arrays


def constructor_loss(episodes, baseline="value", entropy_coef=.01, value_coef=.5):
    """Return (loss, metrics) for on-policy episodes from one unchanged task/policy.

    Actor terms are summed along a trajectory, then averaged over ALL episodes,
    including any empty ones. Value regression and normalized entropy use a step
    average. The scalar value is computed before the sampled action; its actor
    advantage is detached. Trajectories are consumed once, before an optimizer
    step. This is Monte Carlo actor--critic, not PPO or Q learning.
    """
    if baseline not in {"value", "loo"}:
        raise ValueError("baseline must be 'value' or 'loo'")
    if not episodes:
        raise ValueError("at least one episode is required")
    if baseline == "loo" and len(episodes) < 2:
        raise ValueError("leave-one-out requires at least two episodes")
    if not all(np.isfinite(x) and x >= 0 for x in (entropy_coef, value_coef)):
        raise ValueError("loss coefficients must be finite and nonnegative")
    data = [_episode_data(e) for e in episodes]
    base_returns = np.array([e["base_return"] for e in episodes], dtype=float)
    returns = np.array([e["return"] for e in episodes], dtype=float)
    actor, critics, exploration = [], [], []
    advantages, supports, values, targets = [], [], [], []
    lengths = [len(d) for d, _ in data]
    for i, (decisions, arrays) in enumerate(data):
        other_return = float(np.delete(base_returns, i).mean()) if baseline == "loo" else None
        for j, d in enumerate(decisions):
            logp = _scalar_tensor(_field(d, "log_prob"), "log_prob")
            entropy = _scalar_tensor(_field(d, "entropy"), "entropy")
            support_size = _field(d, "support_size")
            if (isinstance(support_size, bool) or not isinstance(support_size, (int, np.integer))
                    or support_size < 1):
                raise ValueError("support_size must be a positive integer")
            if baseline == "value":
                value = _scalar_tensor(_field(d, "value"), "value")
                target = value.new_tensor(float(arrays["togo"][j]))
                advantage = (target - value).detach()
                critics.append(F.smooth_l1_loss(value, target))
                values.append(float(value.detach()))
                targets.append(float(target))
            else:
                # Baseline = E[other base returns] - Phi(current state).
                shifted_baseline = other_return - float(arrays["potentials"][j])
                advantage = logp.new_tensor(float(arrays["togo"][j]) - shifted_baseline)
            actor.append(-advantage * logp)
            exploration.append(entropy / math.log(support_size) if support_size > 1
                               else entropy * 0)
            advantages.append(float(advantage))
            supports.append(int(support_size))
    loss = None
    actor_loss = critic_loss = entropy_mean = None
    if actor:
        actor_loss = torch.stack(actor).sum() / len(episodes)
        entropy_mean = torch.stack(exploration).mean()
        critic_loss = torch.stack(critics).mean() if critics else actor_loss.new_tensor(0.)
        loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_mean
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite constructor loss")
    target_variance = float(np.var(targets)) if targets else 0.
    metrics = {
        "episodes": len(episodes), "updated_episodes": sum(length > 0 for length in lengths),
        "decisions": sum(lengths), "reward_mean": float(returns.mean()),
        "reward_std": float(returns.std()), "base_return_mean": float(base_returns.mean()),
        "base_return_std": float(base_returns.std()),
        "episode_length_mean": float(np.mean(lengths)), "episode_length_max": max(lengths),
        "advantage_mean": float(np.mean(advantages)) if advantages else 0.,
        "advantage_std": float(np.std(advantages)) if advantages else 0.,
        "normalized_entropy": float(entropy_mean.detach()) if entropy_mean is not None else 0.,
        "support_mean": float(np.mean(supports)) if supports else 0.,
        "actor_loss": float(actor_loss.detach()) if actor_loss is not None else None,
        "value_loss": float(critic_loss.detach()) if critic_loss is not None else None,
        "value_explained_variance": 1 - float(np.var(np.asarray(targets) - values)) / target_variance
        if target_variance > 1e-12 else None,
    }
    return loss, metrics
