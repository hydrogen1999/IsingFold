"""Regression tests for full-rollout utility-advantage centering."""

from __future__ import annotations

import math
import copy

import numpy as np
import pytest

from isingfold.rl.contracts import Mode
from isingfold.rl.model import ModelOutput
from isingfold.rl.ppo import PPOConfig, PPOTrainer
from isingfold.rl.rollout import (
    COLLECTION_SCHEDULE,
    COLLECTION_SCHEDULE_SCHEMA,
    COLLECTION_SCHEDULE_SCHEMA_VERSION,
    UTILITY_ADVANTAGE_TRANSFORM,
    UTILITY_ADVANTAGE_TRANSFORM_SCHEMA,
    UTILITY_ADVANTAGE_TRANSFORM_SCHEMA_VERSION,
    RolloutBuffer,
    Transition,
    center_full_rollout_utility_advantages,
)

torch = pytest.importorskip("torch")


class _TwoActionPolicy(torch.nn.Module):
    """Small differentiable policy used to expose PPO's common-offset pathology."""

    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(2))
        self.value = torch.nn.Parameter(torch.tensor(0.0))
        self.failure = torch.nn.Linear(1, 1)
        self.failure.requires_grad_(False)
        self.improvement_mode = True

    def forward_single(self, observation: object, device=None) -> ModelOutput:
        del observation
        target = device or self.logits.device
        logits = self.logits.to(target)
        log_probs = torch.log_softmax(logits, dim=0)
        value = self.value.to(target)
        zero = value.new_zeros(())
        return ModelOutput(
            masked_log_probs=log_probs,
            utility_value=value,
            failure_logit=zero,
            failure_value=zero,
            action_count=torch.as_tensor(2, device=target),
        )


def _synthetic_transition(*, episode: int, action: int, reward: float) -> Transition:
    return Transition(
        observation=None,
        legal_mask=np.asarray([True, True]),
        chosen_index=action,
        old_log_prob=-math.log(2.0),
        old_log_probs=np.asarray([-math.log(2.0), -math.log(2.0)]),
        old_utility=0.0,
        old_failure=0.0,
        reward=reward,
        cost=0.0,
        terminated=True,
        episode=episode,
    )


def _synthetic_rollout() -> RolloutBuffer:
    # Three lower-return samples chose action 0 and one higher-return sample chose action 1.
    # A large positive common offset makes the old raw-advantage estimator follow sample
    # frequency and prefer action 0.  Centering must retain the 0.10 utility-unit contrast.
    buffer = RolloutBuffer()
    for action, reward in ((0, 0.55), (0, 0.55), (0, 0.55), (1, 0.65)):
        episode = buffer.add_episode()
        buffer.add(_synthetic_transition(episode=episode.index, action=action, reward=reward))
        episode.terminal_reward = reward
        episode.returned_valid = True
    buffer.metadata.update(
        {
            "collection_schedule_schema": COLLECTION_SCHEDULE_SCHEMA,
            "collection_schedule_schema_version": COLLECTION_SCHEDULE_SCHEMA_VERSION,
            "collection_schedule": COLLECTION_SCHEDULE,
            "collection_training_seed": 0,
            "collection_update_index": 0,
            "episode_schedule_start": 0,
            "episode_schedule_stop_exclusive": 4,
            "episode_schedule_indices": tuple(range(4)),
        }
    )
    buffer.compute_targets(gae_lambda=0.95)
    return buffer


def _ratio_surrogate_gradient(advantages: np.ndarray) -> np.ndarray:
    logits = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    old = torch.log_softmax(logits.detach(), dim=0)
    actions = torch.as_tensor([0, 0, 0, 1])
    current = torch.log_softmax(logits, dim=0)[actions]
    ratio = torch.exp(current - old[actions])
    loss = -(ratio * torch.as_tensor(advantages)).sum()
    loss.backward()
    assert logits.grad is not None
    return logits.grad.detach().numpy().copy()


def test_full_rollout_centering_fixes_frequency_dominated_gradient_direction() -> None:
    """Reproduce the pilot pathology and prove the centered PPO signal discriminates."""

    raw = np.asarray([0.55, 0.55, 0.55, 0.65], dtype=np.float64)
    centered = center_full_rollout_utility_advantages(
        raw,
        weights=np.ones(4),
        policy_controllable=np.ones(4, dtype=bool),
    )

    raw_gradient = _ratio_surrogate_gradient(raw)
    centered_gradient = _ratio_surrogate_gradient(centered.values)

    # Gradient descent on the uncentered estimator increases action 0 merely because it was
    # sampled three times.  The versioned transform instead increases higher-return action 1.
    assert raw_gradient[0] < 0.0 < raw_gradient[1]
    assert centered_gradient[1] < 0.0 < centered_gradient[0]
    assert np.average(centered.values, weights=np.ones(4)) == pytest.approx(0.0, abs=1e-15)
    assert np.ptp(centered.values) == pytest.approx(np.ptp(raw))
    assert centered.diagnostics.raw_mean == pytest.approx(0.575)
    assert centered.diagnostics.policy_signal_rms > 0.0


def test_transform_is_once_per_full_rollout_weighted_and_never_whitens() -> None:
    raw = np.asarray([0.2, 0.4, 0.8, 0.9])
    weights = np.asarray([2.0, 1.0, 1.0, 0.0])
    result = center_full_rollout_utility_advantages(
        raw,
        weights=weights,
        policy_controllable=np.asarray([True, True, True, True]),
    )

    expected_center = np.average(raw[:3], weights=weights[:3])
    assert result.diagnostics.center == pytest.approx(expected_center)
    assert np.average(result.values[:3], weights=weights[:3]) == pytest.approx(0.0)
    assert np.std(result.values[:3]) == pytest.approx(np.std(raw[:3]))
    assert np.ptp(result.values) == pytest.approx(np.ptp(raw))

    # This differs from independently centering arbitrary minibatches.  PPO must reuse the
    # one detached full-rollout transform for every epoch and minibatch.
    per_minibatch = np.concatenate([raw[:2] - np.mean(raw[:2]), raw[2:] - np.mean(raw[2:])])
    assert not np.allclose(result.values, per_minibatch)


def test_target_computation_records_version_and_ignores_padding_and_singletons() -> None:
    buffer = RolloutBuffer()
    masks = (
        np.asarray([True, True]),
        np.asarray([True]),
        np.asarray([True, False, True, False]),
    )
    for index, mask in enumerate(masks):
        episode = buffer.add_episode()
        transition = _synthetic_transition(
            episode=episode.index,
            action=0,
            reward=0.2 + 0.1 * index,
        )
        transition.legal_mask = mask
        transition.old_log_probs = np.full(mask.shape, -math.log(int(mask.sum())))
        transition.old_log_probs[~mask] = -np.inf
        buffer.add(transition)
        episode.terminal_reward = transition.reward

    buffer.compute_targets()
    transformed = buffer.utility_advantages_for_actor()

    assert buffer.metadata["utility_advantage_transform_schema"] == (
        UTILITY_ADVANTAGE_TRANSFORM_SCHEMA
    )
    assert buffer.metadata["utility_advantage_transform_schema_version"] == (
        UTILITY_ADVANTAGE_TRANSFORM_SCHEMA_VERSION
    )
    assert buffer.metadata["utility_advantage_transform"] == UTILITY_ADVANTAGE_TRANSFORM
    assert transformed.diagnostics.transitions == 3
    assert transformed.diagnostics.policy_controllable_transitions == 2
    assert transformed.diagnostics.singleton_transitions == 1


def test_actor_rejects_an_old_or_tampered_advantage_transform_version() -> None:
    buffer = _synthetic_rollout()
    buffer.metadata["utility_advantage_transform_schema_version"] = 0

    with pytest.raises(RuntimeError, match="transform schema/version"):
        buffer.utility_advantages_for_actor()


def test_zero_variance_and_all_singleton_rollouts_are_explicitly_degenerate() -> None:
    constant = center_full_rollout_utility_advantages(
        np.asarray([0.56, 0.56, 0.56]),
        weights=np.ones(3),
        policy_controllable=np.ones(3, dtype=bool),
    )
    singleton = center_full_rollout_utility_advantages(
        np.asarray([0.2, 0.8]),
        weights=np.ones(2),
        policy_controllable=np.zeros(2, dtype=bool),
    )

    assert np.array_equal(constant.values, np.zeros(3))
    assert constant.diagnostics.is_degenerate(1e-8)
    assert constant.diagnostics.policy_signal_rms == 0.0
    assert singleton.diagnostics.is_degenerate(1e-8)
    assert singleton.diagnostics.policy_controllable_transitions == 0


def test_transform_rejects_nonfinite_aggregate_weight_before_emitting_diagnostics() -> None:
    with pytest.raises(ValueError, match="aggregate weight"):
        center_full_rollout_utility_advantages(
            np.asarray([0.2, 0.8]),
            weights=np.asarray([1e308, 1e308]),
            policy_controllable=np.asarray([True, True]),
        )


def test_improvement_update_fails_before_optimizer_on_degenerate_utility_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = _synthetic_rollout()
    assert buffer.gae_utility is not None
    buffer.gae_utility[:] = 0.56
    model = _TwoActionPolicy()
    trainer = PPOTrainer(
        model,
        PPOConfig(
            episodes_per_batch=4,
            epochs=1,
            minibatch=4,
            entropy_initial=1e-9,
            entropy_floor=1e-9,
            value_weight=0.0,
            reference_length=1,
        ),
        mode=Mode.IMPROVEMENT,
        total_updates=1,
    )
    optimizer_calls = 0

    def optimizer_step(*_args, **_kwargs) -> None:
        nonlocal optimizer_calls
        optimizer_calls += 1

    monkeypatch.setattr(buffer, "authenticate_replay", lambda **_kwargs: 4)
    monkeypatch.setattr(trainer.optimizer, "step", optimizer_step)

    with pytest.raises(
        RuntimeError, match="degenerate.*utility advantage|utility advantage.*degenerate"
    ):
        trainer.update(buffer)
    assert optimizer_calls == 0


def test_ppo_update_uses_centered_signal_and_reports_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = _synthetic_rollout()
    model = _TwoActionPolicy()
    trainer = PPOTrainer(
        model,
        PPOConfig(
            episodes_per_batch=4,
            epochs=1,
            minibatch=4,
            entropy_initial=1e-9,
            entropy_floor=1e-9,
            value_weight=0.0,
            learning_rate=0.05,
            grad_norm=10.0,
            reference_length=1,
            seed=0,
        ),
        mode=Mode.IMPROVEMENT,
        total_updates=1,
    )
    monkeypatch.setattr(buffer, "authenticate_replay", lambda **_kwargs: 4)

    assert trainer.replay_check(buffer) == pytest.approx(0.0, abs=1e-7)
    old_likelihoods = [row.old_log_probs.copy() for row in buffer.transitions]
    logs = trainer.update(buffer)

    assert model.logits[1] > model.logits[0]
    assert logs["utility_advantage_raw_mean"] == pytest.approx(0.575)
    assert logs["utility_advantage_centered_mean"] == pytest.approx(0.0, abs=1e-15)
    assert logs["utility_advantage_policy_signal_rms"] > 0.0
    assert logs["utility_advantage_degenerate"] == 0.0
    assert logs["utility_advantage_degeneracy_tolerance"] == pytest.approx(1e-8)
    assert logs["utility_advantage_transform_version"] == float(
        UTILITY_ADVANTAGE_TRANSFORM_SCHEMA_VERSION
    )
    assert all(
        np.array_equal(before, after.old_log_probs)
        for before, after in zip(old_likelihoods, buffer.transitions, strict=True)
    )


def test_hard_kl_violation_rolls_back_epoch_and_retries_at_lower_learning_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = _synthetic_rollout()
    initial_model = _TwoActionPolicy()
    backtracked_model = copy.deepcopy(initial_model)
    reference_model = copy.deepcopy(initial_model)
    config = PPOConfig(
        episodes_per_batch=4,
        epochs=1,
        minibatch=4,
        entropy_initial=1e-9,
        entropy_floor=1e-9,
        value_weight=0.0,
        learning_rate=0.05,
        grad_norm=10.0,
        reference_length=1,
        kl_target=0.01,
        kl_stop=0.02,
        kl_backtrack_factor=0.5,
        kl_max_backtracks=1,
        seed=0,
    )
    trainer = PPOTrainer(backtracked_model, config, total_updates=1)
    monkeypatch.setattr(buffer, "authenticate_replay", lambda **_kwargs: 4)
    observed_kl = iter((0.2, 0.005))
    monkeypatch.setattr(trainer, "full_buffer_kl", lambda _buffer: next(observed_kl))

    logs = trainer.update(buffer)

    reference = PPOTrainer(
        reference_model,
        PPOConfig(
            **{
                **config.__dict__,
                "learning_rate": 0.025,
                "kl_max_backtracks": 0,
            }
        ),
        total_updates=1,
    )
    monkeypatch.setattr(reference, "full_buffer_kl", lambda _buffer: 0.005)
    reference.update(buffer)

    torch.testing.assert_close(backtracked_model.logits, reference_model.logits)
    torch.testing.assert_close(backtracked_model.value, reference_model.value)
    assert logs["kl"] == pytest.approx(0.005)
    assert logs["kl_rejected_epochs"] == 1.0
    assert logs["kl_backtracks"] == 1.0
    assert logs["optimizer_steps_attempted"] == 2.0
    assert logs["optimizer_steps"] == 1.0
    assert logs["optimizer_steps_rolled_back"] == 1.0
    assert logs["effective_learning_rate"] == pytest.approx(0.025)
    assert logs["kl_hard_limit_satisfied"] == 1.0


def test_hard_kl_exhaustion_restores_last_safe_epoch_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = _synthetic_rollout()
    model = _TwoActionPolicy()
    initial = copy.deepcopy(model.state_dict())
    trainer = PPOTrainer(
        model,
        PPOConfig(
            episodes_per_batch=4,
            epochs=2,
            minibatch=4,
            entropy_initial=1e-9,
            entropy_floor=1e-9,
            value_weight=0.0,
            learning_rate=0.05,
            grad_norm=10.0,
            reference_length=1,
            kl_target=0.01,
            kl_stop=0.02,
            kl_backtrack_factor=0.5,
            kl_max_backtracks=2,
            seed=0,
        ),
        total_updates=1,
    )
    monkeypatch.setattr(buffer, "authenticate_replay", lambda **_kwargs: 4)
    monkeypatch.setattr(trainer, "full_buffer_kl", lambda _buffer: 0.2)

    logs = trainer.update(buffer)

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name], rtol=0.0, atol=0.0)
    assert logs["kl"] == pytest.approx(0.0)
    assert logs["epochs_accepted"] == 0.0
    assert logs["epochs_attempted"] == 3.0
    assert logs["kl_rejected_epochs"] == 3.0
    assert logs["kl_backtracks"] == 2.0
    assert logs["optimizer_steps"] == 0.0
    assert logs["optimizer_steps_rolled_back"] == 3.0
    assert logs["kl_early_stop_reason"] == "hard-limit-no-safe-step"
    assert logs["kl_hard_limit_satisfied"] == 1.0


def test_accepted_soft_kl_target_stops_remaining_epochs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buffer = _synthetic_rollout()
    trainer = PPOTrainer(
        _TwoActionPolicy(),
        PPOConfig(
            episodes_per_batch=4,
            epochs=4,
            minibatch=4,
            entropy_initial=1e-9,
            entropy_floor=1e-9,
            value_weight=0.0,
            reference_length=1,
            kl_target=0.01,
            kl_stop=0.02,
        ),
        total_updates=1,
    )
    monkeypatch.setattr(buffer, "authenticate_replay", lambda **_kwargs: 4)
    monkeypatch.setattr(trainer, "full_buffer_kl", lambda _buffer: 0.015)

    logs = trainer.update(buffer)

    assert logs["epochs_accepted"] == 1.0
    assert logs["epochs_attempted"] == 1.0
    assert logs["kl_early_stop"] == 1.0
    assert logs["kl_early_stop_reason"] == "soft-target"
    assert logs["kl"] == pytest.approx(0.015)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"utility_advantage_transform": "per-minibatch-whitening"},
        {"utility_advantage_degeneracy_tolerance": -1.0},
        {"utility_advantage_degeneracy_tolerance": float("nan")},
    ),
)
def test_ppo_registry_rejects_unversioned_or_invalid_advantage_transform(
    kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="advantage"):
        PPOConfig(**kwargs)
