"""Normative tests for bounded per-action quality supervision."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from isingfold.rl.model import (
    QUALITY_POLICY_PRIOR_DEFAULT,
    QUALITY_POLICY_PRIOR_MODES,
    IFCore,
    build_model,
)
from isingfold.rl.ppo import (
    WarmStartActionValueTarget,
    WarmStartCorpusDenominators,
    warm_start_actor_critic_loss,
)
from tests.unit.test_rl_ifcore_normative import _conflict_observation, _phase_observation

torch = pytest.importorskip("torch")


def test_quality_policy_prior_variants_are_bounded_or_explicitly_diagnostic() -> None:
    logits = torch.tensor([-20.0, -2.0, 0.0, 2.0, 20.0], requires_grad=True)

    auxiliary = IFCore(quality_prior_mode="auxiliary-only-v1").quality_policy_prior(logits)
    bounded = IFCore(quality_prior_mode="bounded-centered-v1").quality_policy_prior(logits)
    clipped = IFCore(quality_prior_mode="clipped-logit-v1").quality_policy_prior(logits)
    detached = IFCore(quality_prior_mode="bounded-centered-detached-v1").quality_policy_prior(
        logits
    )
    diagnostic = IFCore(quality_prior_mode="raw-logit-diagnostic-v1").quality_policy_prior(logits)

    torch.testing.assert_close(auxiliary, torch.zeros_like(logits))
    torch.testing.assert_close(bounded, 2.0 * torch.sigmoid(logits) - 1.0)
    torch.testing.assert_close(clipped, 0.5 * torch.clamp(logits, -2.0, 2.0))
    torch.testing.assert_close(detached, bounded)
    torch.testing.assert_close(diagnostic, logits)
    assert bool(torch.all(torch.abs(bounded) <= 1.0))
    assert bool(torch.all(torch.abs(clipped) <= 1.0))
    assert detached.requires_grad is False
    assert diagnostic.abs().max() > 1.0


def test_quality_policy_prior_registry_and_factory_fail_closed() -> None:
    assert QUALITY_POLICY_PRIOR_DEFAULT == "bounded-centered-v1"
    assert QUALITY_POLICY_PRIOR_MODES == (
        "auxiliary-only-v1",
        "bounded-centered-v1",
        "clipped-logit-v1",
        "bounded-centered-detached-v1",
        "raw-logit-diagnostic-v1",
    )
    model = build_model("if-core", quality_prior_mode="clipped-logit-v1")
    assert model.quality_prior_mode == "clipped-logit-v1"
    with pytest.raises(ValueError, match="quality-prior mode"):
        IFCore(quality_prior_mode="unregistered")


@pytest.mark.parametrize(
    ("mode", "head_receives_policy_gradient"),
    [
        ("auxiliary-only-v1", False),
        ("bounded-centered-v1", True),
        ("clipped-logit-v1", True),
        ("bounded-centered-detached-v1", False),
        ("raw-logit-diagnostic-v1", True),
    ],
)
def test_quality_policy_prior_gradient_contract(
    mode: str, head_receives_policy_gradient: bool
) -> None:
    torch.manual_seed(260916)
    model = IFCore(improvement_mode=True, quality_prior_mode=mode).eval()
    output = model.forward_single(_conflict_observation())

    model.zero_grad(set_to_none=True)
    (-output.masked_log_probs[1]).backward()
    observed = any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in model.action_quality.parameters()
    )

    assert observed is head_receives_policy_gradient


def test_ifcore_exposes_bounded_action_values_and_uses_them_as_policy_prior() -> None:
    torch.manual_seed(260913)
    model = IFCore(improvement_mode=True).eval()
    observations = [_phase_observation(), _conflict_observation()]

    single = model.forward_single(observations[1])
    batched = model(observations)

    assert single.action_quality_value is not None
    assert single.action_quality_logit is not None
    assert single.action_quality_value.shape == single.masked_log_probs.shape
    assert bool(
        torch.all((single.action_quality_value >= 0.0) & (single.action_quality_value <= 1.0))
    )
    assert batched.action_quality_value is not None
    assert batched.action_quality_logit is not None
    assert batched.action_quality_value.shape == batched.masked_log_probs.shape == (2, 73)
    assert torch.allclose(
        batched.action_quality_value[1, : single.action_count],
        single.action_quality_value,
        atol=2e-6,
        rtol=2e-6,
    )
    assert torch.isnan(batched.action_quality_value[0, batched.action_count[0] :]).all()

    model.zero_grad(set_to_none=True)
    (-single.masked_log_probs[1]).backward()
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in model.action_quality.parameters()
    )


def test_unresolved_row_trains_action_value_and_commit_delta_heads() -> None:
    torch.manual_seed(260914)
    model = IFCore(improvement_mode=True).eval()
    observation = _conflict_observation()
    supervision = WarmStartActionValueTarget(
        action_indices=(0, 1),
        q_mu=(0.20, 0.80),
        continuation_counts=(16, 16),
        inclusion_probabilities=(1.0, 1.0),
        legal_action_count=2,
        commit_index=0,
    )
    output = model([observation])
    assert output.action_quality_logit is not None
    assert output.action_quality_value is not None
    expected_action_value = torch.nn.functional.binary_cross_entropy_with_logits(
        output.action_quality_logit[0, :2],
        torch.tensor([0.20, 0.80]),
    )
    expected_commit_delta = (
        0.5
        * ((output.action_quality_value[0, 1] - output.action_quality_value[0, 0]) - 0.60).square()
    )

    losses = warm_start_actor_critic_loss(
        model,
        [observation],
        [(0, 1)],
        [(0, 1)],
        [0.5],
        [32.0],
        action_value_targets=[supervision],
        corpus_denominators=WarmStartCorpusDenominators(
            actor_ranking_records=0,
            utility_effective_count=32.0,
            action_value_records=1,
            commit_delta_records=1,
        ),
    )

    assert losses.rank.item() == pytest.approx(0.0)
    assert torch.allclose(losses.action_value, expected_action_value)
    assert torch.allclose(losses.commit_delta, expected_commit_delta)
    assert losses.action_effective_count == pytest.approx(32.0)
    assert losses.commit_delta_rows == 1
    losses.total.backward()
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in model.action_quality.parameters()
    )
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in model.utility.parameters()
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"action_indices": (0, 0)}, "strictly increasing"),
        ({"q_mu": (0.2, 1.1)}, "bounded"),
        ({"continuation_counts": (0, 8)}, "positive"),
        ({"inclusion_probabilities": (1.0, 0.0)}, "inclusion"),
        ({"legal_action_count": 1}, "legal support"),
        ({"commit_index": 2}, "commit"),
    ],
)
def test_action_value_target_fails_closed(kwargs: dict[str, object], message: str) -> None:
    values: dict[str, object] = {
        "action_indices": (0, 1),
        "q_mu": (0.2, 0.8),
        "continuation_counts": (8, 8),
        "inclusion_probabilities": (1.0, 1.0),
        "legal_action_count": 2,
        "commit_index": 0,
    }
    values.update(kwargs)
    with pytest.raises(ValueError, match=message):
        WarmStartActionValueTarget(**values)


def test_action_value_supervision_is_validated_before_model_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = IFCore(improvement_mode=True).eval()
    target = WarmStartActionValueTarget(
        action_indices=(0, 1),
        q_mu=(0.2, 0.8),
        continuation_counts=(8, 8),
        inclusion_probabilities=(1.0, 1.0),
        legal_action_count=2,
        commit_index=0,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("malformed action-value labels reached the model")

    monkeypatch.setattr(model, "forward", forbidden)
    with pytest.raises(ValueError, match="same length"):
        warm_start_actor_critic_loss(
            model,
            [_conflict_observation()],
            [(0, 1)],
            [(0, 1)],
            [0.5],
            [16.0],
            action_value_targets=[target, target],
            corpus_denominators=WarmStartCorpusDenominators(
                actor_ranking_records=0,
                utility_effective_count=16.0,
                action_value_records=1,
                commit_delta_records=1,
            ),
        )


def test_action_value_indices_use_tensor_support_not_legal_count() -> None:
    observation = _conflict_observation()
    sparse = replace(
        observation,
        legal_mask=np.asarray([False, True]),
        real_action_mask=np.asarray([True, True]),
    )
    target = WarmStartActionValueTarget(
        action_indices=(1,),
        q_mu=(0.7,),
        continuation_counts=(8,),
        inclusion_probabilities=(1.0,),
        legal_action_count=1,
        support_size=2,
    )

    losses = warm_start_actor_critic_loss(
        IFCore(improvement_mode=True),
        [sparse],
        [(1,)],
        [(1,)],
        [0.7],
        [8.0],
        action_value_targets=[target],
        corpus_denominators=WarmStartCorpusDenominators(
            actor_ranking_records=0,
            utility_effective_count=8.0,
            action_value_records=1,
            commit_delta_records=0,
        ),
    )

    assert torch.isfinite(losses.total)
