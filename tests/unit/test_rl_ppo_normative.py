"""Normative loss/replay invariants for complete-episode PPO."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from isingfold.rl.contracts import Mode
from isingfold.rl.model import IFCore
from isingfold.rl.model import ModelOutput
from isingfold.rl.ppo import (
    PPOConfig,
    PPOTrainer,
    episode_weighted_reduction,
    normalized_legal_entropy,
)
from isingfold.rl.rollout import (
    COLLECTION_SCHEDULE,
    COLLECTION_SCHEDULE_SCHEMA,
    COLLECTION_SCHEDULE_SCHEMA_VERSION,
    ROLLOUT_REPLAY_SCHEMA,
    ROLLOUT_REPLAY_SCHEMA_VERSION,
    STOCHASTIC_COLLECTION_RULE,
    RolloutBuffer,
    Transition,
)


def _transition(*, episode: int, reward: float, terminated: bool, value: float) -> Transition:
    return Transition(
        observation=None,
        legal_mask=np.asarray([True]),
        chosen_index=0,
        old_log_prob=0.0,
        old_log_probs=np.asarray([0.0]),
        old_utility=value,
        old_failure=0.0,
        reward=reward,
        cost=0.0,
        terminated=terminated,
        episode=episode,
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"episodes_per_batch": 0},
        {"epochs": 0},
        {"minibatch": 0},
        {"clip": 1.0},
        {"learning_rate": 0.0},
        {"gae_lambda": 1.1},
        {"gamma": 0.99},
        {"entropy_floor_at": 0.0},
        {"entropy_floor": 0.0},
        {"entropy_initial": 0.0},
        {"entropy_initial": 0.001, "entropy_floor": 0.002},
        {"kl_target": 0.02, "kl_stop": 0.02},
        {"kl_backtrack_factor": 1.0},
        {"kl_max_backtracks": -1},
        {"reference_length": 0},
        {"init_attempt_cap": 0},
    ),
)
def test_ppo_registry_rejects_invalid_or_unregistered_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PPOConfig(**kwargs)


def test_failure_constraint_requires_an_explicit_registered_target() -> None:
    with pytest.raises(ValueError, match="failure_target"):
        PPOConfig(failure_constraint_enabled=True)

    configured = PPOConfig(
        failure_constraint_enabled=True,
        failure_target=0.02,
    )
    assert configured.failure_target == 0.02


def test_entropy_is_normalized_by_exact_legal_support_size() -> None:
    log_probs = torch.tensor(
        [
            [-math.log(2.0), -math.log(2.0), float("-inf"), float("-inf")],
            [-math.log(4.0)] * 4,
            [0.0, float("-inf"), float("-inf"), float("-inf")],
        ]
    )
    legal = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, True],
            [True, False, False, False],
        ]
    )

    normalized, raw = normalized_legal_entropy(log_probs, legal)

    torch.testing.assert_close(raw, torch.tensor([math.log(2.0), math.log(4.0), 0.0]))
    torch.testing.assert_close(normalized, torch.tensor([1.0, 1.0, 0.0]))


def test_entropy_schedule_reaches_a_strictly_positive_floor() -> None:
    trainer = PPOTrainer(
        IFCore(improvement_mode=True),
        PPOConfig(
            episodes_per_batch=1,
            entropy_initial=0.02,
            entropy_floor=0.002,
            entropy_floor_at=0.5,
        ),
        total_updates=10,
    )

    assert trainer.entropy_coefficient() == pytest.approx(0.02)
    trainer.updates_done = 2
    assert trainer.entropy_coefficient() == pytest.approx(0.0128)
    trainer.updates_done = 5
    assert trainer.entropy_coefficient() == pytest.approx(0.002)
    trainer.updates_done = 10
    assert trainer.entropy_coefficient() == pytest.approx(0.002)


def test_dual_episode_reduction_uses_one_over_episode_count_not_weight_sum() -> None:
    values = np.asarray([1.0, 0.0])
    weights = np.asarray([2.0, 0.5])

    assert episode_weighted_reduction(values, weights) == pytest.approx(1.0)


def test_zero_decision_construction_batch_still_updates_failure_dual() -> None:
    model = IFCore(improvement_mode=False)
    config = PPOConfig(
        episodes_per_batch=1,
        epochs=1,
        failure_constraint_enabled=True,
        failure_target=0.25,
        dual_step=0.2,
    )
    trainer = PPOTrainer(model, config, mode=Mode.CONSTRUCTION, total_updates=1)
    buffer = RolloutBuffer(
        metadata={
            "replay_receipt_schema": ROLLOUT_REPLAY_SCHEMA,
            "replay_receipt_schema_version": ROLLOUT_REPLAY_SCHEMA_VERSION,
            "collection_rule": STOCHASTIC_COLLECTION_RULE,
                "training_eligible": True,
                "mode": "construction",
                "collection_schedule_schema": COLLECTION_SCHEDULE_SCHEMA,
                "collection_schedule_schema_version": COLLECTION_SCHEDULE_SCHEMA_VERSION,
                "collection_schedule": COLLECTION_SCHEDULE,
                "collection_training_seed": config.seed,
                "collection_update_index": 0,
                "episode_schedule_start": 0,
                "episode_schedule_stop_exclusive": 1,
                "episode_schedule_indices": (0,),
            }
        )
    episode = buffer.add_episode()
    episode.terminal_cost = 1.0
    episode.returned_valid = False
    optimizer_calls = 0

    def optimizer_step(*_args, **_kwargs) -> None:
        nonlocal optimizer_calls
        optimizer_calls += 1

    trainer.optimizer.step = optimizer_step
    logs = trainer.update(buffer)

    assert optimizer_calls == 0
    assert trainer.updates_done == 1
    assert trainer.lambda_f == pytest.approx(0.15)
    assert logs["transitions"] == 0.0
    assert logs["episodes"] == 1.0
    assert logs["optimizer_steps"] == 0.0


def test_target_recurrence_uses_the_next_row_in_the_same_episode() -> None:
    """Interleaved storage must not bootstrap from another episode's transition."""

    buffer = RolloutBuffer()
    first = buffer.add_episode()
    second = buffer.add_episode()
    first_row = buffer.add(
        _transition(episode=first.index, reward=0.0, terminated=False, value=0.1)
    )
    buffer.add(_transition(episode=second.index, reward=1.0, terminated=True, value=0.9))
    final_row = buffer.add(_transition(episode=first.index, reward=0.5, terminated=True, value=0.2))
    first.terminal_reward = 0.5
    second.terminal_reward = 1.0

    buffer.compute_targets(gae_lambda=0.95)

    assert buffer.gae_utility is not None
    expected_final = 0.5 - 0.2
    expected_first = (0.2 - 0.1) + 0.95 * expected_final
    assert buffer.gae_utility[final_row] == pytest.approx(expected_final)
    assert buffer.gae_utility[first_row] == pytest.approx(expected_first)


def test_transition_weights_follow_registered_episode_weights() -> None:
    buffer = RolloutBuffer()
    first = buffer.add_episode()
    second = buffer.add_episode()
    first.weight = 0.5
    second.weight = 2.0
    first_index = buffer.add(
        _transition(episode=first.index, reward=1.0, terminated=True, value=0.0)
    )
    second_index = buffer.add(
        _transition(episode=second.index, reward=1.0, terminated=True, value=0.0)
    )

    weights = buffer.transition_weights()

    assert weights[first_index] == pytest.approx(0.5)
    assert weights[second_index] == pytest.approx(2.0)


def test_replay_check_validates_every_transition_and_complete_distribution() -> None:
    class ReplayModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.tensor(0.0))
            self.improvement_mode = True
            self.failure = torch.nn.Linear(1, 1)
            self.failure.requires_grad_(False)
            self.calls = 0

        def forward_single(self, observation, device=None):
            del observation, device
            self.calls += 1
            values = [-np.log(2.0), -np.log(2.0)]
            if self.calls == 17:
                # The selected action is unchanged, but another legal action differs.
                values = torch.log_softmax(torch.tensor([0.0, -0.25]), dim=0).tolist()
            return ModelOutput(
                masked_log_probs=torch.tensor(values),
                utility_value=torch.tensor(0.0),
                failure_logit=torch.tensor(0.0),
                failure_value=torch.tensor(0.0),
                action_count=torch.tensor(2),
            )

    buffer = RolloutBuffer()
    episode = buffer.add_episode()
    for index in range(17):
        buffer.add(
            Transition(
                observation=None,
                legal_mask=np.asarray([True, True]),
                chosen_index=0,
                old_log_prob=-np.log(2.0),
                old_log_probs=np.asarray([-np.log(2.0), -np.log(2.0)]),
                old_utility=0.0,
                old_failure=0.0,
                reward=0.0,
                cost=0.0,
                terminated=index == 16,
                episode=episode.index,
                )
            )
    buffer.metadata.update(
        {
            "collection_schedule_schema": COLLECTION_SCHEDULE_SCHEMA,
            "collection_schedule_schema_version": COLLECTION_SCHEDULE_SCHEMA_VERSION,
            "collection_schedule": COLLECTION_SCHEDULE,
            "collection_training_seed": 0,
            "collection_update_index": 0,
            "episode_schedule_start": 0,
            "episode_schedule_stop_exclusive": 1,
            "episode_schedule_indices": (0,),
        }
    )

    model = ReplayModel()
    trainer = PPOTrainer(model, PPOConfig(episodes_per_batch=1), total_updates=1)
    with pytest.raises(RuntimeError, match="replay mismatch"):
        trainer.replay_check(buffer)
    assert model.calls == 17
