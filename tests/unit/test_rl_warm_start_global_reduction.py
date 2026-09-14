"""RED contracts for corpus-normalized warm-start optimization.

The warm-start objective is registered as a corpus objective.  Memory minibatches must
therefore contribute additive pieces of that fixed objective; they must not silently
become independent ratio estimators with one optimizer step each.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import pytest

import isingfold.rl.ppo as ppo
from isingfold.rl.model import IFCore
from isingfold.rl.ppo import WarmStartActionValueTarget
from tests.unit.test_rl_ifcore_normative import _conflict_observation

torch = pytest.importorskip("torch")


def _supervision_rows():
    observation = _conflict_observation()
    return (
        (
            observation,
            (0,),
            (0, 1),
            0.15,
            1.0,
            WarmStartActionValueTarget(
                (0, 1),
                (0.05, 0.95),
                (1, 3),
                (1.0, 0.5),
                2,
                0,
            ),
        ),
        (
            observation,
            (1,),
            (0, 1),
            0.80,
            100.0,
            WarmStartActionValueTarget(
                (0, 1),
                (0.80, 0.20),
                (10, 2),
                (0.5, 1.0),
                2,
            ),
        ),
        (
            observation,
            (0, 1),
            (0, 1),
            0.40,
            4.0,
            WarmStartActionValueTarget(
                (0, 1),
                (0.35, 0.65),
                (2, 8),
                (0.25, 0.5),
                2,
                1,
            ),
        ),
    )


def _corpus_denominators(rows):
    denominator_type = getattr(ppo, "WarmStartCorpusDenominators", None)
    assert denominator_type is not None, (
        "warm-start must expose an authenticated corpus-denominator contract"
    )
    return denominator_type(
        actor_ranking_records=sum(set(row[1]) != set(row[2]) for row in rows),
        utility_effective_count=math.fsum(row[4] for row in rows),
        action_value_records=sum(row[5] is not None for row in rows),
        commit_delta_records=sum(
            row[5] is not None
            and row[5].commit_index is not None
            and len(row[5].action_indices) > 1
            for row in rows
        ),
    )


def _loss(model: IFCore, rows, indices: Sequence[int], denominators):
    selected = [rows[index] for index in indices]
    return ppo.warm_start_actor_critic_loss(
        model,
        [row[0] for row in selected],
        [row[1] for row in selected],
        [row[2] for row in selected],
        [row[3] for row in selected],
        [row[4] for row in selected],
        action_value_targets=[row[5] for row in selected],
        corpus_denominators=denominators,
    )


_LOSS_COMPONENTS = ("rank", "utility", "action_value", "commit_delta", "total")


def _assert_full_equals_partition_sum(full, partitions) -> None:
    for name in _LOSS_COMPONENTS:
        expected = getattr(full, name)
        actual = torch.stack([getattr(part, name) for part in partitions]).sum()
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)


def test_global_reduction_does_not_overweight_a_short_tail_minibatch() -> None:
    """A 2+1 split must reconstruct the same registered corpus objective."""

    torch.manual_seed(260917)
    model = IFCore(improvement_mode=True).eval()
    rows = _supervision_rows()
    denominators = _corpus_denominators(rows)

    full = _loss(model, rows, (0, 1, 2), denominators)
    head = _loss(model, rows, (0, 1), denominators)
    short_tail = _loss(model, rows, (2,), denominators)

    _assert_full_equals_partition_sum(full, (head, short_tail))


def test_global_reduction_is_invariant_to_shuffled_minibatch_partition() -> None:
    """Shuffling records across memory batches must not change loss or gradient."""

    torch.manual_seed(260918)
    model = IFCore(improvement_mode=True).train()
    rows = _supervision_rows()
    denominators = _corpus_denominators(rows)

    full = _loss(model, rows, (0, 1, 2), denominators)
    partition_a = (
        _loss(model, rows, (0, 1), denominators),
        _loss(model, rows, (2,), denominators),
    )
    partition_b = (
        _loss(model, rows, (2, 0), denominators),
        _loss(model, rows, (1,), denominators),
    )

    _assert_full_equals_partition_sum(full, partition_a)
    _assert_full_equals_partition_sum(full, partition_b)

    model.zero_grad(set_to_none=True)
    full.total.backward()
    full_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    model.zero_grad(set_to_none=True)
    for contribution in partition_b:
        contribution.total.backward()
    partition_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }

    assert partition_gradients.keys() == full_gradients.keys()
    for name in full_gradients:
        torch.testing.assert_close(
            partition_gradients[name],
            full_gradients[name],
            rtol=1e-5,
            atol=2e-6,
            msg=lambda message, *, name=name: f"{name}: {message}",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("actor_ranking_records", True, "nonnegative integer"),
        ("actor_ranking_records", -1, "nonnegative integer"),
        ("action_value_records", 1.5, "nonnegative integer"),
        ("commit_delta_records", -1, "nonnegative integer"),
        ("utility_effective_count", False, "finite and positive"),
        ("utility_effective_count", 0.0, "finite and positive"),
        ("utility_effective_count", float("nan"), "finite and positive"),
        ("utility_effective_count", float("inf"), "finite and positive"),
    ],
)
def test_corpus_denominators_fail_closed(field: str, value: object, message: str) -> None:
    values = {
        "actor_ranking_records": 1,
        "utility_effective_count": 2.0,
        "action_value_records": 1,
        "commit_delta_records": 1,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        ppo.WarmStartCorpusDenominators(**values)


def test_commit_delta_denominator_cannot_exceed_action_value_denominator() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        ppo.WarmStartCorpusDenominators(
            actor_ranking_records=0,
            utility_effective_count=1.0,
            action_value_records=0,
            commit_delta_records=1,
        )


def test_minibatch_cannot_exceed_frozen_corpus_denominators_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = IFCore(improvement_mode=True)
    rows = _supervision_rows()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("inconsistent denominators reached the model")

    monkeypatch.setattr(model, "forward", forbidden)
    with pytest.raises(ValueError, match="actor-ranking rows exceed"):
        _loss(
            model,
            rows,
            (0,),
            ppo.WarmStartCorpusDenominators(
                actor_ranking_records=0,
                utility_effective_count=1.0,
                action_value_records=1,
                commit_delta_records=1,
            ),
        )
    with pytest.raises(ValueError, match="utility mass exceeds"):
        _loss(
            model,
            rows,
            (2,),
            ppo.WarmStartCorpusDenominators(
                actor_ranking_records=0,
                utility_effective_count=1.0,
                action_value_records=1,
                commit_delta_records=1,
            ),
        )
    with pytest.raises(ValueError, match="action-value rows exceed"):
        _loss(
            model,
            rows,
            (0,),
            ppo.WarmStartCorpusDenominators(
                actor_ranking_records=1,
                utility_effective_count=1.0,
                action_value_records=0,
                commit_delta_records=0,
            ),
        )
    with pytest.raises(ValueError, match="commit-delta rows exceed"):
        _loss(
            model,
            rows,
            (0,),
            ppo.WarmStartCorpusDenominators(
                actor_ranking_records=1,
                utility_effective_count=1.0,
                action_value_records=1,
                commit_delta_records=0,
            ),
        )


def test_zero_sparse_denominators_return_zero_without_local_contributors() -> None:
    model = IFCore(improvement_mode=True).eval()
    observation = _conflict_observation()
    losses = ppo.warm_start_actor_critic_loss(
        model,
        [observation],
        [(0, 1)],
        [(0, 1)],
        [0.5],
        [1.0],
        corpus_denominators=ppo.WarmStartCorpusDenominators(
            actor_ranking_records=0,
            utility_effective_count=1.0,
            action_value_records=0,
            commit_delta_records=0,
        ),
    )

    assert losses.rank.item() == 0.0
    assert losses.action_value.item() == 0.0
    assert losses.commit_delta.item() == 0.0
