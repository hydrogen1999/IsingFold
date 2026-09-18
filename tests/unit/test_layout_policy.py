"""Behavioral contracts for contextual root construction and its training-only teacher."""
import networkx as nx
import numpy as np
import pytest
import torch

from layout_policy import (LayoutActorCritic, next_variable, root_support,
                           sample_layout_v2, witness_set_loss)


class NoWitnessTask:
    def __init__(self, n=3, m=7):
        self.logical = nx.path_graph(n)
        self.host = nx.path_graph(m)

    @property
    def witness(self):
        raise AssertionError("deployment must not access witness")


class SimpleFeatures:
    def pair(self, v, roots, chains, opcode):
        return np.array([v, roots[0], len(chains), 1], dtype=np.float32)


def test_all_free_support_includes_nonadjacent_roots_and_preserves_state():
    task = NoWitnessTask(n=2)
    chains = {1: frozenset({6})}
    assert root_support(task, 0, chains, "legacy") == (5,)
    assert root_support(task, 0, chains) == tuple(range(6))
    assert chains == {1: frozenset({6})}
    assert root_support(task, 0, chains, rng=np.random.default_rng(99)) == tuple(range(6))


def test_variable_order_matches_legacy_rule_and_skips_empty_chain_keys():
    task = NoWitnessTask(n=4)
    assert next_variable(task, {}) == 2
    assert next_variable(task, {2: frozenset({5}), 1: frozenset()}) == 1
    assert next_variable(task, {v: {v} for v in range(4)}) is None


def test_actor_equivariance_and_action_independent_value():
    torch.manual_seed(4)
    model = LayoutActorCritic(width=12, in_dim=4)
    feats = torch.randn(7, 4)
    perm = torch.tensor([3, 1, 5, 0, 6, 4, 2])
    dist, value = model.distribution_value(feats)
    dist_p, value_p = model.distribution_value(feats[perm])
    torch.testing.assert_close(dist_p.logits, dist.logits[perm])
    torch.testing.assert_close(value_p, value)
    assert value.ndim == 0
    changed = feats.clone()
    changed[1:] += 5
    assert not torch.allclose(model(changed)[0], model(feats)[0])


def test_sampler_has_no_witness_dependency_and_retains_gradients():
    torch.manual_seed(1)
    model = LayoutActorCritic(width=12, in_dim=4)
    task = NoWitnessTask()
    roots, decisions = sample_layout_v2(task, model, SimpleFeatures(), 1.0,
                                        np.random.default_rng(5), capture_teacher=True)
    assert set(roots) == set(task.logical)
    assert len(set().union(*roots.values())) == len(roots)
    assert [d.support_size for d in decisions] == [7, 6, 5]
    assert all(d.features is not None and len(d.state_chains) == i
               for i, d in enumerate(decisions))
    loss = sum(-d.log_prob - .01 * d.entropy + (d.value - 1).square() for d in decisions)
    loss.backward()
    for block in (model.encoder, model.actor, model.critic):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in block.parameters())


def test_local_prioritiser_ablation_and_rng_reproducibility():
    from train_prioritiser import Prioritiser
    model = Prioritiser(width=8, in_dim=4)
    task = NoWitnessTask()
    state_before = torch.random.get_rng_state().clone()
    a, decisions = sample_layout_v2(task, model, SimpleFeatures(), 1.0,
                                    np.random.default_rng(4))
    b, _ = sample_layout_v2(task, model, SimpleFeatures(), 1.0,
                           np.random.default_rng(4))
    assert a == b
    assert all(d.value is None and d.features is None and d.state_chains is None
               for d in decisions)
    assert torch.equal(state_before, torch.random.get_rng_state())
    sum(d.log_prob for d in decisions).backward()
    assert any(p.grad is not None for p in model.parameters())


@pytest.mark.parametrize("n,m,expected", [(0, 3, 0), (3, 0, 0), (4, 2, 2)])
def test_empty_or_exhausted_host_returns_only_constructed_roots(n, m, expected):
    model = LayoutActorCritic(width=8, in_dim=4)
    roots, decisions = sample_layout_v2(NoWitnessTask(n, m), model, SimpleFeatures(), 1.,
                                        np.random.default_rng(0))
    assert len(roots) == len(decisions) == expected


def test_legacy_sampler_greedy_matches_existing_sampler():
    from fast_layout import sample_layout
    from train_prioritiser import Prioritiser
    model = Prioritiser(width=8, in_dim=4)
    task = NoWitnessTask(n=5, m=12)
    before, _ = sample_layout(task, model, SimpleFeatures(), 1.,
                              np.random.default_rng(4), train=False)
    after, _ = sample_layout_v2(task, model, SimpleFeatures(), 1.,
                                np.random.default_rng(4), train=False, support="legacy")
    assert before == after


def test_training_witness_set_loss_uses_mass_without_injecting_candidates():
    logits = torch.zeros(3, requires_grad=True)
    roots = (0, 1, 2)
    loss = witness_set_loss(logits, roots, {0, 2, 9}, training=True)
    torch.testing.assert_close(loss, torch.tensor(np.log(3 / 2), dtype=torch.float32))
    loss.backward()
    assert logits.grad[0] < 0 and logits.grad[2] < 0 < logits.grad[1]
    assert roots == (0, 1, 2)
    assert witness_set_loss(logits, roots, {9}, training=True) is None
    with pytest.raises(ValueError, match="training-only"):
        witness_set_loss(logits, roots, {0})


@pytest.mark.parametrize("temperature", [0., -1., float("nan"), float("inf")])
def test_invalid_temperature_rejected(temperature):
    with pytest.raises(ValueError, match="temperature"):
        sample_layout_v2(NoWitnessTask(), LayoutActorCritic(8, 4), SimpleFeatures(),
                          temperature, np.random.default_rng(0))


def test_invalid_support_and_model_input_rejected():
    task = NoWitnessTask()
    with pytest.raises(ValueError, match="support"):
        root_support(task, 0, {}, strategy="unknown")
    with pytest.raises(ValueError, match="variable"):
        root_support(task, 99, {})
    model = LayoutActorCritic(8, 4)
    with pytest.raises(ValueError, match="nonempty"):
        model(torch.empty(0, 4))
    with pytest.raises(ValueError, match="in_dim"):
        model(torch.empty(3, 2))


@pytest.mark.parametrize("support", ["all_free", "legacy"])
@pytest.mark.parametrize("budget,expected", [(0, 0), (1, 1), (2.9, 2), (4, 4), (100, 5)])
def test_sampler_never_exceeds_declared_budget_even_when_host_has_space(support, budget, expected):
    features = SimpleFeatures()
    features.budget = budget
    roots, decisions = sample_layout_v2(
        NoWitnessTask(n=5, m=9), LayoutActorCritic(8, 4), features, 1.0,
        np.random.default_rng(2), support=support)
    assert len(roots) == len(decisions) == expected
    assert (len(set().union(*roots.values())) if roots else 0) == expected
    assert len(roots) <= budget


@pytest.mark.parametrize("budget", [-1.0, float("nan"), float("inf"), -float("inf")])
def test_nonfinite_or_negative_qubit_budget_rejected(budget):
    features = SimpleFeatures()
    features.budget = budget
    with pytest.raises(ValueError, match="budget"):
        sample_layout_v2(NoWitnessTask(), LayoutActorCritic(8, 4), features, 1.0,
                         np.random.default_rng(0))


@pytest.mark.parametrize("width,in_dim", [(0, 4), (8, -1), (8.2, 4), (8, 4.1),
                                          (float("nan"), 4), (8, float("inf"))])
def test_nonfinite_or_noninteger_model_dimensions_rejected(width, in_dim):
    with pytest.raises(ValueError, match="positive integers"):
        LayoutActorCritic(width, in_dim)
