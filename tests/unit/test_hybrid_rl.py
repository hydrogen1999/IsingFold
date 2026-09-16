"""One hybrid episode on a toy: roots from a PLACE-only policy pass, then minorminer."""
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


def test_place_roots_then_minorminer_on_a_toy():
    import torch
    from _context import qubit_budget
    from candidate_features import FeatureContext
    from train_hybrid_rl import complete, place_roots
    from train_prioritiser import Prioritiser
    from . import test_witness_replay as tw

    task, witness = tw._planted_task(side=5, fill=0.6, seed=0, lmax=2)
    torch.manual_seed(0)
    model = Prioritiser(16)
    roots, logps = place_roots(task, model, FeatureContext(task, qubit_budget(witness)), 1.0,
                               np.random.default_rng(0), train=True)
    assert 0 < len(roots) <= task.logical.number_of_nodes()
    assert len(logps) == len(roots)
    ok, secs, n = complete(task, roots, 5.0, 5, 1)
    assert isinstance(ok, bool) and n >= 1
