"""MC-gradient contracts; synthetic terminal-reward learning, without an annealer."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from constructor_learning import constructor_loss


def decision(log_prob, value=None, support=2, entropy=None, opcode="PLACE"):
    return SimpleNamespace(log_prob=log_prob, value=value, support_size=support,
                           entropy=log_prob.new_tensor(0.) if entropy is None else entropy,
                           opcode=opcode)


def episode(base_return, decisions, potentials=None):
    phi = np.array(potentials if potentials is not None else [0.] * len(decisions), dtype=float)
    togo = base_return - phi
    rewards = togo - np.r_[togo[1:], 0.] if len(togo) else np.array([])
    return {"decisions": decisions, "potentials": phi.tolist(), "togo": togo.tolist(),
            "rewards": rewards.tolist(), "base_return": base_return,
            "return": float(togo[0]) if len(togo) else base_return,
            "terminal_potential": 0.}


def test_potential_corrected_loo_equal_terminal_returns_have_zero_advantage():
    logps = [torch.tensor(-.5, requires_grad=True) for _ in range(4)]
    eps = [episode(1., [decision(logps[0]), decision(logps[1])], [0., .7]),
           episode(1., [decision(logps[2]), decision(logps[3])], [0., .2])]
    loss, metrics = constructor_loss(eps, baseline="loo", entropy_coef=0.)
    loss.backward()
    assert all(p.grad == 0 for p in logps)
    assert metrics["advantage_std"] == metrics["advantage_mean"] == 0


def test_loo_baseline_is_independent_of_own_terminal_return():
    def own_baseline(ret):
        logps = [torch.tensor(-.2, requires_grad=True) for _ in range(3)]
        eps = [episode(r, [decision(p)], [.3]) for r, p in zip([ret, 2., 4.], logps)]
        loss, _ = constructor_loss(eps, baseline="loo", entropy_coef=0.)
        loss.backward()
        return ret - .3 + 3 * float(logps[0].grad)
    assert [own_baseline(1.), own_baseline(5.)] == pytest.approx([2.7, 2.7])


def test_actor_sums_all_actions_including_sampled_commit_without_length_normalization():
    a, b, c = [torch.tensor(-.5, requires_grad=True) for _ in range(3)]
    eps = [episode(1., [decision(a), decision(b, opcode="COMMIT")]),
           episode(0., [decision(c)])]
    loss, _ = constructor_loss(eps, baseline="loo", entropy_coef=0.)
    loss.backward()
    assert a.grad == b.grad == pytest.approx(-.5)
    assert c.grad == pytest.approx(.5)


def test_empty_episodes_still_count_in_episode_average():
    p = torch.tensor(-.5, requires_grad=True)
    loss, metrics = constructor_loss([episode(1., [decision(p)]), episode(0., [])],
                                      baseline="loo", entropy_coef=0.)
    loss.backward()
    assert p.grad == pytest.approx(-.5)
    assert metrics["episodes"] == 2 and metrics["updated_episodes"] == 1
    loss, metrics = constructor_loss([episode(0., []), episode(0., [])], baseline="loo")
    assert loss is None and metrics["decisions"] == 0


def test_actor_detaches_value_and_critic_targets_shaped_return_to_go():
    logp = torch.tensor(-.4, requires_grad=True)
    value = torch.tensor(.2, requires_grad=True)
    loss, _ = constructor_loss([episode(1., [decision(logp, value)], [.4])],
                               baseline="value", entropy_coef=0., value_coef=.5)
    loss.backward()
    # target=.6, advantage=.4; regression contributes .5*(.2-.6).
    assert logp.grad == pytest.approx(-.4)
    assert value.grad == pytest.approx(-.2)


def test_one_action_entropy_and_gradients_are_finite():
    logits = torch.tensor([0.], requires_grad=True)
    dist = torch.distributions.Categorical(logits=logits)
    value = torch.tensor(.2, requires_grad=True)
    d = decision(dist.log_prob(torch.tensor(0)), value, 1, dist.entropy(), "COMMIT")
    loss, metrics = constructor_loss([episode(1., [d])])
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert metrics["normalized_entropy"] == 0


@pytest.mark.parametrize("key", ["entropy_coef", "value_coef"])
@pytest.mark.parametrize("value", [-1., float("nan"), float("inf")])
def test_invalid_coefficients_fail(key, value):
    with pytest.raises(ValueError, match="coefficients"):
        constructor_loss([episode(0., [])], **{key: value})


def test_invalid_baseline_or_missing_critic_fails():
    with pytest.raises(ValueError, match="baseline"):
        constructor_loss([episode(0., [])], baseline="bad")
    with pytest.raises(ValueError, match="two episodes"):
        constructor_loss([episode(0., [])], baseline="loo")
    with pytest.raises(ValueError, match="at least one"):
        constructor_loss([])
    with pytest.raises(ValueError, match="value must be"):
        constructor_loss([episode(1., [decision(torch.tensor(-.5))])])


@pytest.mark.parametrize("field", ["rewards", "togo", "potentials"])
def test_nonfinite_or_misaligned_rollout_is_rejected(field):
    e = episode(1., [decision(torch.tensor(-.5))])
    e[field] = [float("nan")]
    with pytest.raises(ValueError, match="one finite value per decision"):
        constructor_loss([e], baseline="value")
    e[field] = []
    with pytest.raises(ValueError, match="one finite value per decision"):
        constructor_loss([e], baseline="value")


def test_timeout_must_zero_terminal_potential_and_telescope():
    e = episode(0., [decision(torch.tensor(-.5))], [.7])
    e["terminal_potential"] = .2
    with pytest.raises(ValueError, match="terminal potential"):
        constructor_loss([e])
    e["terminal_potential"] = 0.
    e["potentials"] = [.2]
    with pytest.raises(ValueError, match="telescope"):
        constructor_loss([e])


def test_mapping_decisions_and_nonfinite_model_output_contract():
    p, v = torch.tensor(-.5, requires_grad=True), torch.tensor(0., requires_grad=True)
    d = {"log_prob": p, "value": v, "entropy": torch.tensor(.5), "support_size": 2}
    loss, _ = constructor_loss([episode(1., [d])])
    assert torch.isfinite(loss)
    d["log_prob"] = torch.tensor(float("nan"))
    with pytest.raises(ValueError, match="log_prob must be"):
        constructor_loss([episode(1., [d])])


def test_tiny_on_policy_mc_update_learns_rewarded_terminal_action():
    """Toy two-action terminal environment, not embedding/annealing evidence."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        logits = torch.nn.Parameter(torch.zeros(2))
        optimizer = torch.optim.SGD([logits], lr=.4)
        rng = np.random.default_rng(12)
        initial = float(logits.softmax(0)[1].detach())
        for _ in range(40):
            dist = torch.distributions.Categorical(logits=logits)
            probabilities = dist.probs.detach().numpy().astype(float)
            probabilities /= probabilities.sum()
            batch = []
            for a in rng.choice(2, size=16, p=probabilities):
                d = decision(dist.log_prob(torch.tensor(int(a))), entropy=dist.entropy(),
                             opcode="COMMIT" if a == 1 else "STOP")
                batch.append(episode(float(a), [d]))
            loss, _ = constructor_loss(batch, baseline="loo", entropy_coef=0.)
            optimizer.zero_grad()
            loss.backward()
            assert torch.isfinite(logits.grad).all()
            optimizer.step()
        final = float(logits.softmax(0)[1].detach())
        print(f"toy on-policy terminal gate: initial={initial:.6f}, final={final:.6f}")
        assert initial == .5 and final > .95
    finally:
        torch.set_num_threads(old_threads)
