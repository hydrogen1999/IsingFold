"""Exact delayed-credit checks, separate from embedding performance claims."""
from itertools import product
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch

from constructor_learning import constructor_loss
from constructor_rollout import episode as rollout
from constructor_state_value import ConstructorStateValue
from constructor_tiny_gate import Actor, Features
from isingfold.embedding import LogicalProblem


def decision(logp, base_value):
    return SimpleNamespace(log_prob=logp, entropy=logp.new_tensor(0.),
                           support_size=2, base_value=base_value)


def episode(reward, decisions, potentials=None):
    phi = np.array(potentials if potentials is not None else [0.] * len(decisions))
    togo = reward - phi
    rewards = togo - np.r_[togo[1:], 0.] if len(togo) else np.array([])
    return {"objective": "quality", "valid": reward > 0,
            "quality_measured": reward > 0,
            "residual": 4 * (1 - reward) if reward > 0 else None,
            "base_return": reward, "return": float(togo[0]) if len(togo) else 0.,
            "rewards": rewards.tolist(), "togo": togo.tolist(),
            "potentials": phi.tolist(), "terminal_potential": 0., "decisions": decisions}


def test_zero_base_critic_is_exact_loo_with_shaping_and_empty_episode():
    gradients = []
    for baseline in ("loo", "loo_value"):
        logps = [torch.tensor(-.5, dtype=torch.float64, requires_grad=True) for _ in range(3)]
        decisions = [decision(p, p.new_tensor(0., requires_grad=True)) for p in logps]
        episodes = [episode(.98, decisions[:2], [.2, .8]),
                    episode(0., decisions[2:], [.7]), episode(0., [])]
        loss, metrics = constructor_loss(episodes, baseline=baseline,
                                        value_coef=0., entropy_coef=0.)
        gradients.append(torch.stack(torch.autograd.grad(loss, logps)))
        if baseline == "loo_value":
            assert metrics["initial_base_values"] == [0., 0., 0.]
            assert metrics["value_target"] == "base_return"
    torch.testing.assert_close(*gradients, rtol=0, atol=0)


@pytest.mark.parametrize("critic_values", [(0., 0.), (.12, .83), (.485, .97)])
def test_two_step_expected_gradient_retains_failure_and_quality_objective(critic_values):
    """Enumerate independent trajectory pairs; all critic values are pre-action."""
    theta = torch.tensor([.2, -.3], dtype=torch.float64, requires_grad=True)
    survive, high = theta.sigmoid()
    probability = torch.stack((1 - survive, survive * (1 - high), survive * high))
    objective = survive * ((1 - high) * .96 + high * .98)
    expected = torch.autograd.grad(-objective, theta)[0]
    measured = torch.zeros_like(theta)
    for a, b in product(range(3), repeat=2):
        first = torch.distributions.Bernoulli(logits=theta[0])
        second = torch.distributions.Bernoulli(logits=theta[1])
        batch = []
        for action in (a, b):
            ds = [decision(first.log_prob(theta.new_tensor(float(action > 0))),
                           theta.new_tensor(critic_values[0]))]
            if action:
                ds.append(decision(second.log_prob(theta.new_tensor(float(action == 2))),
                                   theta.new_tensor(critic_values[1])))
            batch.append(episode([0., .96, .98][action], ds,
                                 [.15, .6] if action else [.15]))
        loss, _ = constructor_loss(batch, baseline="loo_value", entropy_coef=0., value_coef=0.)
        measured += (probability[a] * probability[b]).detach() * torch.autograd.grad(loss, theta)[0]
    torch.testing.assert_close(measured, expected, rtol=1e-12, atol=1e-12)


def test_exact_fixture_reduces_late_quality_gradient_variance_without_changing_mean():
    """A controlled counterexample to needing perfect validity for quality credit.

    Half the trajectories fail before the quality decision. At the surviving state,
    a known state value reduces gradient variance in this fixture only.
    """
    moments = {}
    for baseline in ("loo", "loo_value"):
        gradients, weights = [], []
        for a, b in product(range(3), repeat=2):
            theta = torch.zeros(2, dtype=torch.float64, requires_grad=True)
            first = torch.distributions.Bernoulli(logits=theta[0])
            second = torch.distributions.Bernoulli(logits=theta[1])
            batch = []
            for action in (a, b):
                ds = [decision(first.log_prob(theta.new_tensor(float(action > 0))),
                               theta.new_tensor(.485))]
                if action:
                    ds.append(decision(second.log_prob(theta.new_tensor(float(action == 2))),
                                       theta.new_tensor(.97)))
                batch.append(episode([0., .96, .98][action], ds))
            loss, _ = constructor_loss(batch, baseline=baseline, entropy_coef=0., value_coef=0.)
            gradients.append(torch.autograd.grad(loss, theta)[0].numpy())
            weights.append([.5, .25, .25][a] * [.5, .25, .25][b])
        gradients, weights = np.asarray(gradients), np.asarray(weights)
        mean = weights @ gradients
        variance = weights @ (gradients - mean) ** 2
        moments[baseline] = mean, variance
    np.testing.assert_allclose(moments["loo"][0], [-.2425, -.0025])
    np.testing.assert_allclose(moments["loo"][0], moments["loo_value"][0], atol=1e-15)
    assert moments["loo_value"][1][1] < .51 * moments["loo"][1][1]


def test_actor_detaches_critic_and_regression_targets_base_return_not_shaped_return():
    logp = torch.tensor(-.5, requires_grad=True)
    value = torch.tensor(.2, requires_grad=True)
    other = torch.tensor(.4, requires_grad=True)
    batch = [episode(.98, [decision(logp, value)], [.7]),
             episode(0., [decision(torch.tensor(-.3), other)])]
    loss, _ = constructor_loss(batch, baseline="loo_value", entropy_coef=0., value_coef=.5)
    loss.backward()
    assert value.grad == pytest.approx(.5 * (.2 - .98) / 2)
    assert other.grad == pytest.approx(.5 * .4 / 2)
    # Own base baseline = .2 + (0 - .4); no own return enters the baseline.
    assert logp.grad == pytest.approx(-(.98 - (.2 - .4)) / 2)


def test_critic_is_permutation_invariant_and_regression_cannot_train_actor():
    torch.manual_seed(7)
    actor = torch.nn.Linear(3, 3)
    critic = ConstructorStateValue(3, width=8)
    rows = actor(torch.tensor([[.2, .1, -.3], [.8, -.6, .4], [.4, .5, .7]]))
    assert critic(rows, remaining_fraction=.8, progress=.2).item() == 0.
    with torch.no_grad():
        critic.net[-1].weight.fill_(.1)
    value = critic(rows, remaining_fraction=.8, progress=.2)
    permuted = critic(rows[[2, 0, 1]], remaining_fraction=.8, progress=.2)
    torch.testing.assert_close(value, permuted)
    before = {name: p.detach().clone() for name, p in actor.named_parameters()}
    optimizer = torch.optim.SGD(critic.parameters(), lr=.1)
    optimizer.zero_grad()
    (value - .9).square().backward()
    optimizer.step()
    assert all(p.grad is None for p in actor.parameters())
    for name, p in actor.named_parameters():
        torch.testing.assert_close(before[name], p, rtol=0, atol=0)


def test_rollout_records_independent_critic_before_sampled_action():
    logical, host = nx.path_graph(2), nx.path_graph(5)
    task = SimpleNamespace(logical=logical, host=host, name="state-value-toy", lineage="toy",
                           problem=LogicalProblem.from_dicts({0: 0., 1: 0.}, {(0, 1): -1.}))
    actor = Actor(torch.nn.Linear(16, 1, bias=False))
    torch.nn.init.zeros_(actor.m.weight)
    critic = ConstructorStateValue(16, width=8)
    contexts = []
    original = critic.forward
    def record(rows, **kwargs):
        contexts.append(kwargs)
        assert not rows.requires_grad
        return original(rows, **kwargs)
    critic.forward = record
    result = rollout(task, actor, Features(task), 1., 4, np.random.default_rng(2), 30.,
                     objective="feasibility", state_value_model=critic)
    assert contexts and contexts[0] == {"remaining_fraction": 1., "progress": 0.}
    assert len(contexts) == len(result["decisions"])
    assert all(d.value is None and d.base_value.item() == 0. for d in result["decisions"])
    assert all(d.base_value.requires_grad for d in result["decisions"])


def test_residual_loo_rejects_missing_critic_and_single_episode():
    record = episode(.98, [decision(torch.tensor(-.5), None)])
    with pytest.raises(ValueError, match="base_value"):
        constructor_loss([record, episode(0., [])], baseline="loo_value")
    with pytest.raises(ValueError, match="two episodes"):
        constructor_loss([record], baseline="loo_value")
