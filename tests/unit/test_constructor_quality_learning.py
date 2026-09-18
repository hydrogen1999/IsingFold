"""Quality credit and measurement contracts, not embedding performance evidence."""
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from constructor_learning import constructor_loss
from constructor_objective import measure_terminal, measure_terminal_metrics, measure_selection_energy


def quality_episode(reward, logp, *, valid=True, potential=0., value=None):
    decision = SimpleNamespace(log_prob=logp, entropy=logp.new_tensor(0.),
                               support_size=3, value=value)
    return {"objective": "quality", "valid": valid, "quality_measured": valid,
            "residual": 4 * (1 - reward) if valid else None,
            "base_return": reward, "return": reward - potential,
            "rewards": [reward - potential], "togo": [reward - potential],
            "potentials": [potential], "terminal_potential": 0., "decisions": [decision]}


@pytest.mark.parametrize("scale", [1., 100.])
def test_scaled_loo_exact_expected_gradient_matches_quality_objective(scale):
    """Enumerate all pairs: failures are included and the estimator stays unbiased."""
    logits = torch.tensor([.2, -.1, .4], dtype=torch.float64, requires_grad=True)
    rewards = torch.tensor([.98, .96, 0.], dtype=torch.float64)
    probabilities = logits.softmax(0)
    expected = torch.autograd.grad(-scale * (probabilities * rewards).sum(), logits)[0]
    measured = torch.zeros_like(logits)
    for a, b in product(range(3), repeat=2):
        dist = torch.distributions.Categorical(logits=logits)
        eps = [quality_episode(float(rewards[x]), dist.log_prob(torch.tensor(x)), valid=x < 2)
               for x in (a, b)]
        loss, _ = constructor_loss(eps, baseline="loo", entropy_coef=0., advantage_scale=scale)
        weight = float((probabilities[a] * probabilities[b]).detach())
        measured += weight * torch.autograd.grad(loss, logits)[0]
    torch.testing.assert_close(measured, expected)


def test_quality_diagnostics_separate_close_valid_labels_from_validity_credit():
    def make(valid):
        return [quality_episode(r, torch.tensor(-.5, requires_grad=True), valid=v)
                for r, v in valid]
    _, all_valid = constructor_loss(make([(.999, True), (.997, True)]), baseline="loo")
    assert all_valid["validity_loo_advantage_rms"] == 0
    assert all_valid["quality_loo_advantage_rms"] == pytest.approx(.002)
    assert all_valid["quality_to_validity_loo_rms"] is None
    assert all_valid["quality_valid_return_std"] == pytest.approx(.001)
    _, mixed = constructor_loss(make([(.999, True), (.997, True), (0., False)]), baseline="loo")
    assert mixed["quality_valid_episodes"] == 2
    assert mixed["quality_valid_return_std"] == pytest.approx(.001)
    assert mixed["validity_loo_advantage_rms"] > .7
    assert mixed["quality_to_validity_loo_rms"] < .003
    assert mixed["quality_measured_fraction_among_valid"] == 1.


@pytest.mark.parametrize("field,value", [
    ("quality_measured", False), ("residual", None), ("residual", float("nan")),
    ("residual", -.1), ("base_return", .4), ("valid", None),
])
def test_incomplete_or_inconsistent_valid_quality_labels_fail(field, value):
    ep = quality_episode(.99, torch.tensor(-.5, requires_grad=True))
    ep[field] = value
    if field == "base_return":
        ep["rewards"] = ep["togo"] = [value]
        ep["return"] = value
    with pytest.raises(ValueError):
        constructor_loss([ep, quality_episode(.98, torch.tensor(-.5))], baseline="loo")


def test_failed_quality_episode_cannot_receive_progress_reward():
    ep = quality_episode(.2, torch.tensor(-.5), valid=False)
    with pytest.raises(ValueError, match="zero base return"):
        constructor_loss([ep, quality_episode(.99, torch.tensor(-.5))], baseline="loo")


@pytest.mark.parametrize("scale", [0., -1., float("inf"), float("nan"), True, np.array([2.])])
def test_invalid_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="fixed finite positive"):
        constructor_loss([quality_episode(.99, torch.tensor(-.5))], baseline="value",
                         advantage_scale=scale)


def test_fixed_scale_does_not_scale_critic_target_or_change_shaping_invariance():
    logp = torch.tensor(-.4, requires_grad=True)
    value = torch.tensor(.2, requires_grad=True)
    ep = quality_episode(.9, logp, value=value, potential=.3)
    loss, _ = constructor_loss([ep], baseline="value", entropy_coef=0., value_coef=.5,
                              advantage_scale=10.)
    loss.backward()
    assert logp.grad == pytest.approx(-4.)
    assert value.grad == pytest.approx(-.2)

    gradients = []
    for potentials in ([0., 0.], [.3, .8]):
        leaves = [torch.tensor(-.4, requires_grad=True) for _ in range(2)]
        eps = [quality_episode(r, p, potential=phi)
               for r, p, phi in zip([.99, .97], leaves, potentials)]
        loss, _ = constructor_loss(eps, baseline="loo", entropy_coef=0., advantage_scale=20.)
        loss.backward()
        gradients.append([float(p.grad) for p in leaves])
    assert gradients[0] == pytest.approx(gradients[1])


def test_metrics_share_one_read_block_and_scalar_adapter_remains_compatible(monkeypatch):
    from isingfold.rl import evaluator
    calls = []
    block = evaluator.ReadBlock(hits=3, reads=8, broken_fraction=.125,
                                mean_residual=.2, strength_index=2)
    def sample(*args, **kwargs):
        calls.append(kwargs)
        return block
    monkeypatch.setattr(evaluator, "sample_program", sample)
    task = SimpleNamespace(ground_energy=-4., problem=object())
    terminal = SimpleNamespace(returned_valid=True, embedding={0: frozenset({1})},
                               selected_index=2, selected_program=SimpleNamespace(strength_index=2))
    measured = measure_terminal_metrics(task, terminal, 11, 8)
    assert measured == {"residual": .2, "p_solve": .375, "hits": 3, "reads": 8,
                        "strength_index": 2, "broken_fraction": .125}
    assert len(calls) == 1
    assert measure_terminal(task, terminal, 12, 8) == .2
    assert len(calls) == 2


def test_deployment_selection_uses_public_energy_and_discards_uncertified_hit_rate(monkeypatch):
    from isingfold.rl import evaluator
    class PublicTask:
        problem = SimpleNamespace(h={0: 1.}, j={(0, 1): -3.})
        @property
        def ground_energy(self):
            raise AssertionError("deployment selector accessed certified optimum")
    class EnergyOnlyBlock:
        reads, strength_index = 8, 2
        def __init__(self, score):
            self.mean_residual = score
        @property
        def hits(self):
            raise AssertionError("public lower-bound hits are not solution hits")
        @property
        def rate(self):
            raise AssertionError("public lower-bound rate is not p_solve")
    energies = iter([-3., -1.])
    references = []
    def sample(program, embedding, problem, reference, **kwargs):
        references.append(reference)
        assert kwargs["num_reads"] == 8
        return EnergyOnlyBlock((next(energies) - reference) / abs(reference))
    monkeypatch.setattr(evaluator, "sample_program", sample)
    terminal = SimpleNamespace(returned_valid=True, embedding={0: frozenset({1})},
                               selected_index=2, selected_program=SimpleNamespace(strength_index=2))
    scores = [measure_selection_energy(PublicTask(), terminal, seed, 8) for seed in (1, 2)]
    assert references == [-4., -4.]
    assert scores == [.25, .75]   # same candidate order as raw mean energy


def test_deployment_selection_checks_sampler_receipts(monkeypatch):
    from isingfold.rl import evaluator
    task = SimpleNamespace(problem=SimpleNamespace(h={0: 1.}, j={}))
    terminal = SimpleNamespace(returned_valid=True, embedding={0: frozenset({1})},
                               selected_index=2, selected_program=SimpleNamespace(strength_index=2))
    monkeypatch.setattr(evaluator, "sample_program", lambda *args, **kwargs:
                        SimpleNamespace(reads=7, strength_index=2, mean_residual=.1))
    with pytest.raises(ValueError, match="read/strength"):
        measure_selection_energy(task, terminal, 0, 8)
