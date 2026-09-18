"""Skipping the environment's tensor observation must not change what the constructor sees."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

_p = Path(__file__).resolve().parents[2] / "probes" / "constructor_tiny_gate.py"
spec = importlib.util.spec_from_file_location("constructor_tiny_gate", _p)
tg = importlib.util.module_from_spec(spec); spec.loader.exec_module(tg)
from constructor_rollout import episode  # noqa: E402  (path set by the tiny gate import)


def _trace(build_observation, seed=7):
    torch.manual_seed(0)
    task = tg.Task(); features = tg.Features(task)
    linear = torch.nn.Linear(len(tg.OPCODES) + 8, 1, bias=False)
    torch.nn.init.zeros_(linear.weight)
    actor = tg.Actor(linear)
    with tg.no_completion_solver(), torch.no_grad():
        record = episode(task, actor, features, 1., 32, np.random.default_rng(seed), 30.,
                         train=True, objective="feasibility", evaluate_reward=False,
                         build_observation=build_observation)
    return record


def test_same_trajectory_with_and_without_the_tensor_observation():
    with_obs, without = _trace(True), _trace(False)
    assert [d.opcode for d in with_obs["decisions"]] == [d.opcode for d in without["decisions"]]
    assert with_obs["valid"] == without["valid"] and with_obs["reason"] == without["reason"]
    assert with_obs["embedding"] == without["embedding"]
    assert without["steps"] == with_obs["steps"] > 0


def test_the_flag_defaults_off_for_the_constructor_and_on_for_the_environment():
    import inspect
    from isingfold.rl.env import EmbeddingEnv
    assert inspect.signature(episode).parameters["build_observation"].default is False
    assert inspect.signature(EmbeddingEnv.__init__).parameters["build_observation"].default is True
