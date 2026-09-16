"""One policy episode runs end to end on a toy: the policy scores the generator's candidates,
serves as its preference, samples actions, and the episode reports validity and demand
fraction with log-probabilities for every sampled step."""
from pathlib import Path

import numpy as np
import pytest

PROBES = Path(__file__).resolve().parents[2] / "probes"
SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture(autouse=True)
def _paths(monkeypatch):
    monkeypatch.setenv("ISINGFOLD_SRC", str(SRC))
    monkeypatch.syspath_prepend(str(PROBES))
    monkeypatch.syspath_prepend(str(SRC))


def test_one_policy_episode_runs_and_reports():
    import torch
    from candidate_features import FeatureContext
    from train_constructor_rl import episode
    from train_prioritiser import Prioritiser
    from . import test_witness_replay as tw

    task, witness = tw._planted_task(side=5, fill=0.7, seed=0, lmax=2)
    torch.manual_seed(0)
    model = Prioritiser(16)
    r = episode(task, model, FeatureContext(task), 1.0, 300, np.random.default_rng(0), 60.0)
    assert 0.0 <= r["frac"] <= 1.0
    assert r["steps"] >= 1
    assert len(r["logps"]) >= 1
    assert r["return"] == (1.0 if r["valid"] else 0.0) + r["frac"]
    assert len(r["togo"]) == len(r["logps"])
    assert all(np.isfinite(r["togo"]))
