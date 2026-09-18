"""Monte Carlo actor--critic for empty-start constructive embedding episodes.

The environment uses gamma=1 potential shaping with terminal potential zero,
including timeouts. Thus G_t = base_return - Phi(s_t). A leave-one-out baseline
uses other episodes' BASE returns minus the current state's potential. It never
includes this episode's sampled return and never divides actor credit by length.
No teacher, resource objective, router or annealing-quality approximation is used
by this loss. It consumes rewards supplied by the declared rollout objective.
Potential shaping cancels exactly in the corrected LOO advantage; it is not an
extra source of dense policy credit. A fixed positive advantage scale changes
gradient units, never episode rankings or the validity/quality trade-off.
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


def _quality_diagnostics(episodes, base_returns):
    """Validate labels and expose credit amplitude, not an inferred gradient SNR.

    For the declared affine utility, R = I(valid) + I(valid)*(R-1).
    Both terms use the same other-episode LOO baseline, so their advantages
    sum exactly to the total LOO advantage. The components are diagnostics;
    neither is independently weighted or used to discard failed episodes.
    """
    if not any(e.get("objective") == "quality" for e in episodes):
        return {}
    if not all(e.get("objective") == "quality" for e in episodes):
        raise ValueError("a quality loss batch cannot mix rollout objectives")
    valid, residuals = [], []
    for episode in episodes:
        if type(episode.get("valid")) is not bool:
            raise ValueError("quality episodes must declare Boolean validity")
        valid.append(episode["valid"])
        if episode["valid"]:
            residual = episode.get("residual")
            if (episode.get("quality_measured") is not True or residual is None
                    or not np.isscalar(residual) or not np.isfinite(residual) or residual < 0):
                raise ValueError("valid quality training episodes require a measured finite residual")
            if not .5 - 1e-7 <= episode["base_return"] <= 1. + 1e-7:
                raise ValueError("valid quality utility must lie in [0.5, 1]")
            residuals.append(float(residual))
        elif episode["base_return"] != 0:
            raise ValueError("invalid quality episodes must have zero base return")
    valid = np.asarray(valid, dtype=bool)
    validity_component = valid.astype(float)
    quality_component = base_returns - validity_component
    metrics = {
        "quality_valid_episodes": int(valid.sum()),
        "quality_measured_fraction_among_valid": 1. if valid.any() else None,
        "quality_valid_return_std": float(base_returns[valid].std()) if valid.any() else None,
        "quality_valid_return_range": float(np.ptp(base_returns[valid])) if valid.any() else None,
        "quality_valid_residual_mean": float(np.mean(residuals)) if residuals else None,
        "quality_valid_residual_std": float(np.std(residuals)) if residuals else None,
        "validity_component_mean": float(validity_component.mean()),
        "quality_component_mean": float(quality_component.mean()),
    }
    if len(episodes) > 1:
        def loo_advantage(component):
            return component - (component.sum() - component) / (len(component) - 1)
        for name, component in (("validity", validity_component), ("quality", quality_component)):
            advantage = loo_advantage(component)
            metrics[name + "_loo_advantage_rms"] = float(np.sqrt(np.mean(advantage ** 2)))
        denominator = metrics["validity_loo_advantage_rms"]
        metrics["quality_to_validity_loo_rms"] = (
            metrics["quality_loo_advantage_rms"] / denominator if denominator > 0 else None)
    return metrics


def constructor_loss(episodes, baseline="value", entropy_coef=.01, value_coef=.5, *,
                     advantage_scale=1.):
    """Return (loss, metrics) for on-policy episodes from one unchanged task/policy.

    Actor terms are summed along a trajectory, then averaged over ALL episodes,
    including any empty ones. Value regression and normalized entropy use a step
    average. The scalar value is computed before the sampled action; its actor
    advantage is detached. Trajectories are consumed once, before an optimizer
    step. This is Monte Carlo actor--critic, not PPO or Q learning.

    ``advantage_scale`` is a fixed positive scalar declared before collection,
    not a normalization estimated from this batch. It scales the whole actor
    gradient, preserving the unregularized expected-return objective. Entropy
    and value coefficients remain in their original units; holding their
    *relative* weights fixed requires scaling those coefficients as well.
    It cannot improve measurement SNR or create a missing quality signal.

    ``loo_value`` uses a separate pre-action base-return critic B(s): its base
    baseline is B(s_it) + mean_{j != i}[R_j - B(s_j0)]. The shaped baseline then
    subtracts Phi(s_it). Critic parameters must stay fixed across collection; the
    current episode's return never enters its own residual baseline. Independent
    episodes therefore retain the expected raw terminal-return policy gradient.
    The critic regresses base returns, with all failures retained. Empty episodes
    use initial prediction zero and remain in the episode denominator. A zero
    critic is exactly ordinary LOO, including with potential shaping. Approximate
    state values can reduce variance, but this is not guaranteed by the loss.
    """
    if baseline not in {"value", "loo", "loo_value"}:
        raise ValueError("baseline must be 'value', 'loo' or 'loo_value'")
    if not episodes:
        raise ValueError("at least one episode is required")
    if baseline in {"loo", "loo_value"} and len(episodes) < 2:
        raise ValueError("leave-one-out requires at least two episodes")
    if not all(np.isfinite(x) and x >= 0 for x in (entropy_coef, value_coef)):
        raise ValueError("loss coefficients must be finite and nonnegative")
    if (isinstance(advantage_scale, (bool, np.bool_)) or not np.isscalar(advantage_scale)
            or not np.isfinite(advantage_scale) or advantage_scale <= 0):
        raise ValueError("advantage_scale must be a fixed finite positive scalar")
    data = [_episode_data(e) for e in episodes]
    base_returns = np.array([e["base_return"] for e in episodes], dtype=float)
    returns = np.array([e["return"] for e in episodes], dtype=float)
    quality_metrics = _quality_diagnostics(episodes, base_returns)
    initial_base_values = np.zeros(len(episodes), dtype=float)
    if baseline == "loo_value":
        for i, (decisions, _) in enumerate(data):
            if decisions:
                initial_base_values[i] = float(_scalar_tensor(
                    _field(decisions[0], "base_value"), "base_value").detach())
    residual_returns = base_returns - initial_base_values
    actor, critics, exploration = [], [], []
    advantages, supports, values, targets = [], [], [], []
    lengths = [len(d) for d, _ in data]
    for i, (decisions, arrays) in enumerate(data):
        other_return = (float(np.delete(residual_returns, i).mean())
                        if baseline in {"loo", "loo_value"} else None)
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
            elif baseline == "loo_value":
                value = _scalar_tensor(_field(d, "base_value"), "base_value")
                target = value.new_tensor(float(base_returns[i]))
                shifted_baseline = value.detach() + other_return - float(arrays["potentials"][j])
                advantage = (logp.new_tensor(float(arrays["togo"][j])) - shifted_baseline).detach()
                critics.append(F.smooth_l1_loss(value, target))
                values.append(float(value.detach()))
                targets.append(float(target))
            else:
                # Baseline = E[other base returns] - Phi(current state).
                shifted_baseline = other_return - float(arrays["potentials"][j])
                advantage = logp.new_tensor(float(arrays["togo"][j]) - shifted_baseline)
            actor.append(-float(advantage_scale) * advantage * logp)
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
        "advantage_scale": float(advantage_scale),
        "scaled_advantage_std": float(advantage_scale) * float(np.std(advantages)) if advantages else 0.,
        "normalized_entropy": float(entropy_mean.detach()) if entropy_mean is not None else 0.,
        "support_mean": float(np.mean(supports)) if supports else 0.,
        "actor_loss": float(actor_loss.detach()) if actor_loss is not None else None,
        "value_loss": float(critic_loss.detach()) if critic_loss is not None else None,
        "value_explained_variance": 1 - float(np.var(np.asarray(targets) - values)) / target_variance
        if target_variance > 1e-12 else None,
    }
    metrics.update(quality_metrics)
    if baseline == "loo_value":
        metrics.update({
            "initial_base_values": initial_base_values.tolist(),
            "initial_base_value_mean": float(initial_base_values.mean()),
            "residual_base_return_std": float(residual_returns.std()),
            "value_target": "base_return",
        })
    return loss, metrics
