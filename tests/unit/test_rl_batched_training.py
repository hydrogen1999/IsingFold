"""Equivalence and isolation checks for the segmented training batch path."""

from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from isingfold.rl.model import IFCore
from isingfold.rl.ppo import (
    WarmStartCorpusDenominators,
    warm_start_actor_critic_loss,
    warm_start_rank_loss,
)
from tests.unit.test_rl_ifcore_normative import _conflict_observation, _phase_observation
from tests.unit.test_rl_ppo_contract_hardening import _trainer_and_buffer

torch = pytest.importorskip("torch")


def _assert_output_matches_single(model: IFCore, observations: list) -> None:
    singles = [model.forward_single(observation) for observation in observations]
    batched = model(observations)
    for row, single in enumerate(singles):
        count = int(single.masked_log_probs.shape[0])
        assert torch.allclose(
            batched.masked_log_probs[row, :count],
            single.masked_log_probs,
            atol=2e-6,
            rtol=2e-6,
        )
        assert torch.allclose(
            batched.utility_value[row], single.utility_value, atol=2e-6, rtol=2e-6
        )
        assert torch.allclose(
            batched.failure_logit[row], single.failure_logit, atol=2e-6, rtol=2e-6
        )
        assert torch.allclose(
            batched.failure_value[row], single.failure_value, atol=2e-6, rtol=2e-6
        )
        assert torch.equal(batched.action_count[row], single.action_count)
        assert torch.isneginf(batched.masked_log_probs[row, count:]).all()


def test_disjoint_batch_matches_single_for_variable_graph_action_and_relation_sizes() -> None:
    torch.manual_seed(19)
    model = IFCore(improvement_mode=False).eval()
    observations = [_phase_observation(), _conflict_observation(), _phase_observation()]

    _assert_output_matches_single(model, observations)


def test_disjoint_batch_has_no_cross_observation_leakage() -> None:
    torch.manual_seed(23)
    model = IFCore(improvement_mode=False).eval()
    first = _phase_observation()
    second = _conflict_observation()
    original = model([first, second])
    perturbed = replace(
        second,
        logical=second.logical + np.float32(1000.0),
        hardware=second.hardware - np.float32(1000.0),
        claims=second.claims + np.float32(500.0),
        conflicts=second.conflicts - np.float32(500.0),
        globals_=second.globals_ + np.float32(250.0),
        actions=second.actions - np.float32(250.0),
    )
    changed = model([first, perturbed])

    assert torch.equal(original.masked_log_probs[0], changed.masked_log_probs[0])
    assert torch.equal(original.utility_value[0], changed.utility_value[0])
    assert torch.equal(original.failure_logit[0], changed.failure_logit[0])
    assert not torch.equal(original.utility_value[1], changed.utility_value[1])


def test_batched_outputs_loss_and_gradients_match_scalar_reference() -> None:
    torch.manual_seed(29)
    scalar_model = IFCore(improvement_mode=False).eval()
    batched_model = copy.deepcopy(scalar_model).eval()
    observations = [_phase_observation(), _conflict_observation(), _phase_observation()]
    chosen = [0, 1, 0]

    scalar_loss = torch.zeros(())
    for observation, index in zip(observations, chosen, strict=True):
        output = scalar_model.forward_single(observation)
        scalar_loss = scalar_loss - output.masked_log_probs[index]
        scalar_loss = scalar_loss + 0.17 * output.utility_value.square()
        scalar_loss = scalar_loss + 0.13 * output.failure_logit.square()
    scalar_loss = scalar_loss / len(observations)
    scalar_loss.backward()

    output = batched_model(observations)
    row = torch.arange(len(observations))
    index = torch.as_tensor(chosen)
    batched_loss = (
        -output.masked_log_probs[row, index]
        + 0.17 * output.utility_value.square()
        + 0.13 * output.failure_logit.square()
    ).mean()
    batched_loss.backward()

    assert torch.allclose(batched_loss, scalar_loss, atol=1e-7, rtol=1e-7)
    for (name_scalar, parameter_scalar), (name_batch, parameter_batch) in zip(
        scalar_model.named_parameters(), batched_model.named_parameters(), strict=True
    ):
        assert name_scalar == name_batch
        if parameter_scalar.grad is None or parameter_batch.grad is None:
            assert parameter_scalar.grad is None and parameter_batch.grad is None
            continue
        assert torch.allclose(
            parameter_batch.grad,
            parameter_scalar.grad,
            atol=2e-6,
            rtol=2e-5,
        ), name_scalar


def test_batched_warm_start_loss_and_gradients_match_scalar_equation() -> None:
    torch.manual_seed(31)
    scalar_model = IFCore(improvement_mode=True).eval()
    batched_model = copy.deepcopy(scalar_model).eval()
    observations = [_phase_observation(), _conflict_observation()]
    best_sets = [[0], [1]]
    evaluated_sets = [[0], [0, 1]]

    terms = []
    for observation, best, evaluated in zip(
        observations, best_sets, evaluated_sets, strict=True
    ):
        if set(best) == set(evaluated):
            continue
        logits = scalar_model.forward_single(observation).masked_log_probs
        numerator = torch.logsumexp(logits[torch.as_tensor(best)], dim=0)
        denominator = torch.logsumexp(logits[torch.as_tensor(evaluated)], dim=0)
        terms.append(-(numerator - denominator))
    scalar_loss = torch.stack(terms).mean()
    scalar_loss.backward()

    batched_loss = warm_start_rank_loss(
        batched_model, observations, best_sets, evaluated_sets
    )
    batched_loss.backward()

    assert torch.allclose(batched_loss, scalar_loss, atol=1e-7, rtol=1e-7)
    for (name_scalar, parameter_scalar), (name_batch, parameter_batch) in zip(
        scalar_model.named_parameters(), batched_model.named_parameters(), strict=True
    ):
        assert name_scalar == name_batch
        if parameter_scalar.grad is None or parameter_batch.grad is None:
            assert parameter_scalar.grad is None and parameter_batch.grad is None
            continue
        assert torch.allclose(
            parameter_batch.grad,
            parameter_scalar.grad,
            atol=2e-6,
            rtol=2e-5,
        ), name_scalar


def test_actor_critic_warm_start_adds_count_weighted_state_value_calibration() -> None:
    """The warm start must fit V, not merely rank the sampled actions."""

    torch.manual_seed(37)
    model = IFCore(improvement_mode=True).eval()
    observations = [_phase_observation(), _conflict_observation()]
    best_sets = [[0], [1]]
    evaluated_sets = [[0], [0, 1]]
    targets = [0.2, 0.8]
    effective_counts = [1.0, 3.0]

    output = model(observations)
    rank_terms = []
    for row, (best, evaluated) in enumerate(
        zip(best_sets, evaluated_sets, strict=True)
    ):
        if set(best) == set(evaluated):
            continue
        logits = output.masked_log_probs[row]
        rank_terms.append(
            -(
                torch.logsumexp(logits[torch.as_tensor(best)], dim=0)
                - torch.logsumexp(logits[torch.as_tensor(evaluated)], dim=0)
            )
        )
    expected_rank = torch.stack(rank_terms).mean()
    target_tensor = torch.as_tensor(targets, dtype=output.utility_value.dtype)
    count_tensor = torch.as_tensor(effective_counts, dtype=output.utility_value.dtype)
    expected_utility = 0.5 * (
        count_tensor * (output.utility_value - target_tensor).square()
    ).sum() / count_tensor.sum()

    losses = warm_start_actor_critic_loss(
        model,
        observations,
        best_sets,
        evaluated_sets,
        targets,
        effective_counts,
        corpus_denominators=WarmStartCorpusDenominators(
            actor_ranking_records=1,
            utility_effective_count=4.0,
            action_value_records=0,
            commit_delta_records=0,
        ),
    )

    assert torch.allclose(losses.rank, expected_rank, atol=1e-7, rtol=1e-7)
    assert torch.allclose(losses.utility, expected_utility, atol=1e-7, rtol=1e-7)
    assert torch.allclose(losses.total, expected_rank + expected_utility, atol=1e-7, rtol=1e-7)

    losses.total.backward()
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in model.utility.parameters()
    )
    assert any(
        parameter.grad is not None and bool(torch.any(parameter.grad != 0))
        for parameter in model.actor.parameters()
    )


def test_actor_critic_warm_start_masks_only_unresolved_actor_ranking() -> None:
    """An unresolved authenticated row still calibrates V without diluting actor loss."""

    torch.manual_seed(41)
    model = IFCore(improvement_mode=True).eval()
    observations = [_phase_observation(), _conflict_observation()]
    best_sets = [[0], [1]]
    evaluated_sets = [[0], [0, 1]]
    targets = [0.2, 0.9]
    effective_counts = [1.0, 3.0]

    output = model(observations)
    resolved_logits = output.masked_log_probs[1]
    expected_rank = -(
        resolved_logits[1]
        - torch.logsumexp(resolved_logits[torch.as_tensor([0, 1])], dim=0)
    )
    target_tensor = torch.as_tensor(targets, dtype=output.utility_value.dtype)
    count_tensor = torch.as_tensor(effective_counts, dtype=output.utility_value.dtype)
    expected_utility = 0.5 * (
        count_tensor * (output.utility_value - target_tensor).square()
    ).sum() / count_tensor.sum()

    losses = warm_start_actor_critic_loss(
        model,
        observations,
        best_sets,
        evaluated_sets,
        targets,
        effective_counts,
        corpus_denominators=WarmStartCorpusDenominators(
            actor_ranking_records=1,
            utility_effective_count=4.0,
            action_value_records=0,
            commit_delta_records=0,
        ),
    )

    assert torch.allclose(losses.rank, expected_rank, atol=1e-7, rtol=1e-7)
    assert torch.allclose(losses.utility, expected_utility, atol=1e-7, rtol=1e-7)
    assert losses.effective_count == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("targets", "counts", "message"),
    [
        ([0.2], [1.0, 1.0], "same length"),
        ([1.1, 0.2], [1.0, 1.0], "bounded"),
        ([0.1, 0.2], [0.0, 1.0], "positive"),
    ],
)
def test_actor_critic_warm_start_rejects_invalid_critic_supervision_before_forward(
    targets: list[float],
    counts: list[float],
    message: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = IFCore(improvement_mode=True).eval()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid evaluator targets reached the policy forward pass")

    monkeypatch.setattr(model, "forward", forbidden)
    with pytest.raises(ValueError, match=message):
        warm_start_actor_critic_loss(
            model,
            [_phase_observation(), _conflict_observation()],
            [[0], [1]],
            [[0], [0, 1]],
            targets,
            counts,
            corpus_denominators=WarmStartCorpusDenominators(
                actor_ranking_records=1,
                utility_effective_count=2.0,
                action_value_records=0,
                commit_delta_records=0,
            ),
        )


@pytest.mark.parametrize("bad_index", (-1, 2))
def test_disjoint_batch_fails_closed_before_offsetting_invalid_local_indices(
    bad_index: int,
) -> None:
    observation = _phase_observation()
    relation = observation.index_logical_edges.copy()
    relation[0, 0] = bad_index
    malformed = replace(observation, index_logical_edges=relation)
    model = IFCore(improvement_mode=True).eval()

    with pytest.raises(ValueError, match="out-of-range local index"):
        model([observation, malformed])
    with pytest.raises(ValueError, match="out-of-range local index"):
        model.forward_single(malformed)


def test_ppo_replay_and_update_use_batched_path_with_scalar_collection_likelihoods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer, buffer = _trainer_and_buffer()

    def forbidden_scalar_forward(*_args, **_kwargs):
        raise AssertionError("PPO recomputation must use the segmented batch path")

    monkeypatch.setattr(trainer.model, "forward_single", forbidden_scalar_forward)
    assert trainer.replay_check(buffer) < 1e-4
    logs = trainer.update(buffer)
    assert logs["transitions"] == buffer.n_transitions
