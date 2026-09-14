"""Registered loss profiles for the post-selection Q-label control."""

from __future__ import annotations

from dataclasses import replace

import pytest

from isingfold.rl.model import IFCore
from isingfold.rl.ppo import (
    WARM_START_FULL_PROFILE_ID,
    WARM_START_RANK_VALUE_CONTROL_PROFILE_ID,
    WarmStartActionValueTarget,
    WarmStartCorpusDenominators,
    WarmStartLossWeights,
    warm_start_actor_critic_loss,
    warm_start_loss_profile,
)
from tests.unit.test_rl_ifcore_normative import _conflict_observation

torch = pytest.importorskip("torch")


def _target(q_mu: tuple[float, float]) -> WarmStartActionValueTarget:
    return WarmStartActionValueTarget(
        action_indices=(0, 1),
        q_mu=q_mu,
        continuation_counts=(16, 16),
        inclusion_probabilities=(1.0, 1.0),
        legal_action_count=2,
        commit_index=0,
    )


def _loss(model: IFCore, profile_id: str, target: WarmStartActionValueTarget):
    return warm_start_actor_critic_loss(
        model,
        [_conflict_observation()],
        [(1,)],
        [(0, 1)],
        [0.6],
        [32.0],
        action_value_targets=[target],
        corpus_denominators=WarmStartCorpusDenominators(
            actor_ranking_records=1,
            utility_effective_count=32.0,
            action_value_records=1,
            commit_delta_records=1,
        ),
        loss_profile=warm_start_loss_profile(profile_id),
    )


def _gradients(model: IFCore) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def test_registered_profiles_differ_only_in_direct_q_supervision() -> None:
    full = warm_start_loss_profile(WARM_START_FULL_PROFILE_ID)
    control = warm_start_loss_profile(WARM_START_RANK_VALUE_CONTROL_PROFILE_ID)

    assert full.weights == WarmStartLossWeights(
        rank=1.0,
        utility=0.5,
        action_value=1.0,
        commit_delta=0.5,
    )
    assert control.weights == replace(
        full.weights,
        action_value=0.0,
        commit_delta=0.0,
    )
    assert full.q_mu_label_supervision is True
    assert full.production_transfer_eligible is True
    assert control.q_mu_label_supervision is False
    assert control.production_transfer_eligible is False
    assert full.contract()["profile_id"] == "full-qmu-v4"
    assert control.contract()["profile_id"] == "rank-value-only-control-v1"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rank", True),
        ("rank", -1.0),
        ("utility", float("nan")),
        ("action_value", float("inf")),
        ("commit_delta", -0.1),
    ],
)
def test_loss_weights_reject_invalid_values(field: str, value: object) -> None:
    values: dict[str, object] = {
        "rank": 1.0,
        "utility": 0.5,
        "action_value": 1.0,
        "commit_delta": 0.5,
    }
    values[field] = value
    with pytest.raises(ValueError, match="finite and nonnegative"):
        WarmStartLossWeights(**values)


def test_loss_rejects_unregistered_or_forged_profile_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = IFCore(improvement_mode=True)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("unregistered profile reached the model")

    monkeypatch.setattr(model, "forward", forbidden)
    with pytest.raises(ValueError, match="registered warm-start loss profile"):
        warm_start_loss_profile("unregistered")


def test_control_is_invariant_to_q_labels_but_preserves_rank_and_value() -> None:
    torch.manual_seed(260919)
    template = IFCore(improvement_mode=True).eval()
    model_a = IFCore(improvement_mode=True).eval()
    model_b = IFCore(improvement_mode=True).eval()
    model_a.load_state_dict(template.state_dict())
    model_b.load_state_dict(template.state_dict())

    low_high = _loss(
        model_a,
        WARM_START_RANK_VALUE_CONTROL_PROFILE_ID,
        _target((0.1, 0.9)),
    )
    high_low = _loss(
        model_b,
        WARM_START_RANK_VALUE_CONTROL_PROFILE_ID,
        _target((0.9, 0.1)),
    )

    assert low_high.action_value.item() == 0.0
    assert low_high.commit_delta.item() == 0.0
    assert low_high.rank.item() > 0.0
    assert low_high.utility.item() > 0.0
    torch.testing.assert_close(low_high.total, high_low.total)
    low_high.total.backward()
    high_low.total.backward()
    gradients_a = _gradients(model_a)
    gradients_b = _gradients(model_b)
    assert gradients_a.keys() == gradients_b.keys()
    for name in gradients_a:
        torch.testing.assert_close(gradients_a[name], gradients_b[name])


def test_full_profile_uses_q_and_commit_delta_labels() -> None:
    torch.manual_seed(260920)
    model = IFCore(improvement_mode=True).eval()

    losses = _loss(model, WARM_START_FULL_PROFILE_ID, _target((0.1, 0.9)))

    assert losses.action_value.item() > 0.0
    assert losses.commit_delta.item() > 0.0
    assert losses.total.item() == pytest.approx(
        sum(
            component.item()
            for component in (
                losses.rank,
                losses.utility,
                losses.action_value,
                losses.commit_delta,
            )
        ),
        rel=1e-6,
    )
