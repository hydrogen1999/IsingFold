"""Small wiring contracts for the real-environment overfit diagnostic."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import constructor_tiny_gate as gate
from isingfold.rl.contracts import OPCODES, Opcode


def test_tiny_features_match_registered_formula_and_privileged_inputs_raise():
    task = gate.Task()
    features = gate.Features(task)
    before = {0: {0}, 1: {2}, 2: {4}}
    after = {0: {0, 1}, 1: {2, 3}, 2: {4}}
    candidate = SimpleNamespace(opcode=Opcode.COMMIT, archive_ref=0)
    row = features.observe(candidate, before, state=SimpleNamespace(
        archive=[SimpleNamespace(chains=after)]), ctx=None, steps_left=1, max_steps=32)
    assert row.dtype == np.float32 and len(row) == len(OPCODES) + 8 == 16
    np.testing.assert_array_equal(row[len(OPCODES):-3], features.summary(after) - features.summary(before))
    np.testing.assert_array_equal(row[-3:], [1., 0., 0.])
    for name in ("witness", "ground_energy", "initial_embedding"):
        with pytest.raises(AssertionError):
            getattr(task, name)


def test_completion_guard_restores_solver_and_cli_rejects_invalid_sizes():
    original = gate.minorminer.find_embedding
    with gate.no_completion_solver():
        with pytest.raises(AssertionError, match="minorminer"):
            gate.minorminer.find_embedding([], [])
    assert gate.minorminer.find_embedding is original
    for argv in (["--episodes", "1"], ["--iterations", "0"],
                 ["--eval-episodes", "0"], ["--seed", "-1"]):
        with pytest.raises(SystemExit) as exc:
            gate.main(argv)
        assert exc.value.code == 2


def test_one_update_wiring_keeps_original_rng_reward_and_loss_settings(monkeypatch, capsys):
    records, losses = [], []
    original_loss = gate.constructor_loss
    def fake_episode(task, actor, features, temperature, steps, rng, seconds, **kwargs):
        assert temperature == 1. and steps == 32 and seconds == 30.
        assert kwargs == {"train": True, "objective": "feasibility", "evaluate_reward": False}
        rows = torch.eye(16)[:2]
        dist = torch.distributions.Categorical(logits=actor(rows))
        action = int(rng.integers(2))
        records.append((torch.is_grad_enabled(), action))
        d = SimpleNamespace(log_prob=dist.log_prob(torch.tensor(action)), entropy=dist.entropy(),
                            value=None, support_size=2, opcode="STOP")
        ret = float(action) * .1
        return {"decisions": [d], "rewards": [ret], "togo": [ret], "potentials": [0.],
                "return": ret, "base_return": ret, "terminal_potential": 0.,
                "valid": False, "steps": 1, "secs": 0., "reason": "STOP_NO_VALID"}
    def check_loss(episodes, **kwargs):
        losses.append(kwargs)
        return original_loss(episodes, **kwargs)
    monkeypatch.setattr(gate, "episode", fake_episode)
    monkeypatch.setattr(gate, "constructor_loss", check_loss)
    old_threads = torch.get_num_threads()
    try:
        assert gate.main(["--iterations", "1", "--episodes", "2", "--eval-episodes", "2"]) == 0
    finally:
        torch.set_num_threads(old_threads)
    assert losses == [{"baseline": "loo", "entropy_coef": 0.}]
    assert [train for train, _ in records] == [False, False, True, True, False, False]
    expected_seeds = [100000, 100001, 0, 1, 100000, 100001]
    assert [a for _, a in records] == [int(np.random.default_rng(seed).integers(2)) for seed in expected_seeds]
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert output[0]["parameters"] == 16 and output[0]["completion_solver"] is None
    assert [r["evaluation"] for r in output if "evaluation" in r] == ["init", "final"]
