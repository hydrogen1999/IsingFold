"""Policy-gradient contracts and a tiny representation overfit gate.

The gate is a manufactured two-state classification task. It checks observable
information and optimization, not annealing quality or embedding performance.
"""
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch

from candidate_features import FeatureContext
from layout_features import LayoutFeatureContext, WIDTH
from layout_learning import trajectory_loss, witness_prefix_loss
from layout_policy import LayoutActorCritic, LayoutDecision, witness_set_loss


def decision(log_prob, value=None, entropy=None, support_size=2):
    return LayoutDecision(log_prob=log_prob, entropy=entropy if entropy is not None else
                          log_prob.new_tensor(0.), value=value, support_size=support_size,
                          roots=tuple(range(support_size)), action=0, variable=0)


def test_positive_advantage_increases_probability_of_sampled_action():
    logits = torch.zeros(2, requires_grad=True)
    dist = torch.distributions.Categorical(logits=logits)
    value = torch.tensor(0., requires_grad=True)
    d = decision(dist.log_prob(torch.tensor(0)), value, dist.entropy())
    loss, metrics = trajectory_loss([(1., [d])], baseline="value", value_coef=0.)
    loss.backward()
    assert logits.grad[0] < 0 < logits.grad[1]
    assert value.grad == 0  # Actor must not train its baseline through advantage.
    assert metrics["reward_mean"] == 1 and metrics["reward_std"] == 0
    assert metrics["value_explained_variance"] is None


def test_critic_receives_only_declared_regression_gradient():
    logp = torch.tensor(-.4, requires_grad=True)
    value = torch.tensor(2., requires_grad=True)
    loss, _ = trajectory_loss([(1., [decision(logp, value)])], baseline="value", value_coef=.5)
    loss.backward()
    assert logp.grad == pytest.approx(1.)  # Return minus detached value is -1.
    assert value.grad == pytest.approx(.5)  # .5 times the Huber derivative.


def test_leave_one_out_baseline_excludes_own_return():
    def gradient(first_return):
        logps = [torch.tensor(-.5, requires_grad=True) for _ in range(3)]
        returns = [first_return, 3., 5.]
        loss, _ = trajectory_loss([(r, [decision(p)]) for r, p in zip(returns, logps)])
        loss.backward()
        return float(logps[0].grad)
    # dL / dlogp = -(return - baseline) / three episodes.
    inferred_a = 1. + 3 * gradient(1.)
    inferred_b = 7. + 3 * gradient(7.)
    assert inferred_a == inferred_b == pytest.approx(4.)


def test_actor_sums_decisions_instead_of_normalizing_trajectory_length():
    a, b, c = (torch.tensor(-.5, requires_grad=True) for _ in range(3))
    loss, _ = trajectory_loss([(1., [decision(a), decision(b)]), (0., [decision(c)])])
    loss.backward()
    assert a.grad == b.grad == pytest.approx(-.5)
    assert c.grad == pytest.approx(.5)


def test_identical_returns_produce_no_spurious_loo_actor_update():
    a, b = (torch.tensor(-.5, requires_grad=True) for _ in range(2))
    loss, metrics = trajectory_loss([(1., [decision(a)]), (1., [decision(b)])])
    loss.backward()
    assert a.grad == b.grad == 0
    assert metrics["advantage_std"] == metrics["reward_std"] == 0


def test_one_action_entropy_is_zero_and_gradients_finite():
    logits = torch.tensor([2.], requires_grad=True)
    dist = torch.distributions.Categorical(logits=logits)
    d = decision(dist.log_prob(torch.tensor(0)), torch.tensor(.2, requires_grad=True),
                 dist.entropy(), support_size=1)
    loss, metrics = trajectory_loss([(1., [d])], baseline="value", entropy_coef=.1)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert metrics["normalized_entropy"] == 0


def test_missing_value_and_invalid_returns_fail_explicitly():
    with pytest.raises(ValueError, match="value baseline"):
        trajectory_loss([(1., [decision(torch.tensor(-.2))])], baseline="value")
    with pytest.raises(ValueError, match="two episodes"):
        trajectory_loss([(1., [])])
    with pytest.raises(ValueError, match="finite terminal"):
        trajectory_loss([(float("nan"), []), (0., [])])
    loss, metrics = trajectory_loss([(1., []), (0., [])])
    assert loss is None and metrics["updated_episodes"] == 0


@pytest.mark.parametrize("name", ["entropy_coef", "value_coef"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.])
def test_nonfinite_loss_coefficients_rejected(name, value):
    with pytest.raises(ValueError):
        trajectory_loss([(1., []), (0., [])], **{name: value})


class TeacherTask:
    def __init__(self):
        self.host, self.logical = nx.path_graph(6), nx.path_graph(2)

    @property
    def witness(self):
        raise AssertionError("the teacher must use only the explicitly supplied training witness")


class TeacherFeatures:
    def __init__(self, witness):
        self.witness = witness
        self.budget = sum(len(c) for c in witness.values())

    def pair(self, v, roots, chains):
        assert all(set(c) <= set(self.witness[u]) for u, c in chains.items())
        return np.array([v, roots[0], len(chains), 1.], dtype=np.float32)


def test_witness_teacher_uses_compatible_prefixes_and_reports_support_exclusion():
    task = TeacherTask()
    witness = {0: {0, 1, 2}, 1: {3, 4, 5}}
    model = LayoutActorCritic(8, 4)
    features = TeacherFeatures(witness)
    loss, metrics = witness_prefix_loss(task, witness, model, features,
                                        np.random.default_rng(0), support="all_free")
    assert metrics["teacher_steps"] == metrics["teacher_covered"] == 2
    assert metrics["teacher_coverage"] == 1.
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.actor.parameters())
    _, legacy = witness_prefix_loss(task, witness, model, features,
                                     np.random.default_rng(0), support="legacy")
    # Teacher first chooses root 5 for variable 1. Legacy offers only 4 for
    # variable 0, whose witness chain is {0,1,2}; it must not inject a target.
    assert legacy["teacher_coverage"] == .5


def test_teacher_dtype_matches_model_and_rejects_invalid_witness():
    task = TeacherTask()
    witness = {0: {0, 1, 2}, 1: {3, 4, 5}}
    model = LayoutActorCritic(8, 4).double()
    loss, _ = witness_prefix_loss(task, witness, model, TeacherFeatures(witness),
                                  np.random.default_rng(0))
    assert loss.dtype == torch.float64 and torch.isfinite(loss)
    with pytest.raises(ValueError, match="valid, disjoint"):
        witness_prefix_loss(task, {0: {0, 1}, 1: {1, 2}}, model, TeacherFeatures(witness),
                             np.random.default_rng(0))


def test_witness_above_declared_budget_is_not_a_feasible_teacher():
    witness = {0: {0, 1, 2}, 1: {3, 4, 5}}
    features = TeacherFeatures(witness)
    features.budget = 5
    with pytest.raises(ValueError, match="exceeds the declared qubit budget"):
        witness_prefix_loss(TeacherTask(), witness, LayoutActorCritic(8, 4), features,
                             np.random.default_rng(0))


@pytest.mark.parametrize("temperature", [float("nan"), float("inf"), 0.])
def test_nonfinite_teacher_temperature_rejected(temperature):
    witness = {0: {0, 1, 2}, 1: {3, 4, 5}}
    with pytest.raises(ValueError):
        witness_prefix_loss(TeacherTask(), witness, LayoutActorCritic(8, 4),
                             TeacherFeatures(witness), np.random.default_rng(0),
                             temperature=temperature)


def test_tiny_overfit_learns_distinction_unobservable_in_legacy_features():
    """Artificial label: place nearer the stronger neighbor; no sampler/quality claim."""
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(9)
        def task(a, b):
            return SimpleNamespace(name="toy", host=nx.path_graph(7),
                                   logical=nx.Graph([(0, 1), (0, 2)]),
                                   problem=SimpleNamespace(h={}, j={(0, 1): a, (0, 2): b}))
        tasks = [task(4., 1.), task(1., 4.)]
        chains, roots = {1: {0}, 2: {6}}, (1, 5)
        old_rows, rows = [], []
        for t in tasks:
            old, new = FeatureContext(t), LayoutFeatureContext(t)
            old_rows.append(np.stack([old.pair(0, [q], chains) for q in roots]))
            rows.append(torch.tensor(np.stack([new.pair(0, [q], chains) for q in roots])))
        np.testing.assert_array_equal(old_rows[0], old_rows[1])
        # Opposite target indices on identical old inputs force average target
        # probability to 1/2 for any deterministic legacy policy, trained or not.
        model = LayoutActorCritic(16, WIDTH)
        opt = torch.optim.Adam(model.parameters(), lr=.02)
        def target_probability():
            with torch.no_grad():
                return float(torch.stack([model(x).softmax(0)[i] for i, x in enumerate(rows)]).mean())
        initial = target_probability()
        for _ in range(100):
            loss = sum(witness_set_loss(model(x), roots, {roots[i]}, training=True)
                       for i, x in enumerate(rows)) / 2
            opt.zero_grad()
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            opt.step()
        final = target_probability()
        print(f"toy representation gate: initial={initial:.6f}, final={final:.6f}, legacy ceiling=0.5")
        assert initial < .6 and final > .99
    finally:
        torch.set_num_threads(threads)
