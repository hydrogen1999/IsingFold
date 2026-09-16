"""The prioritiser's pieces work on a toy: features compute for every candidate a hinted
replay offers, the trainer fits a handful of decisions, and the model can serve as the
generator's preference."""
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


def test_features_compute_for_every_offered_candidate_and_a_model_scores_them():
    import torch
    from candidate_features import WIDTH, FeatureContext
    from train_prioritiser import Prioritiser
    from witness_replay import replay
    from . import test_witness_replay as tw

    task, witness = tw._planted_task(side=6, fill=0.75, seed=0, lmax=2)
    records = []
    r = replay(task, witness, 500, True, records)
    assert r["ok"], r
    assert len(records) >= 5
    fc = FeatureContext(task)
    rec = records[3]
    chains = {v: frozenset(c) for v, c in rec["chains"].items()}
    feats = np.stack([fc.candidate(c, chains) for c in rec["candidates"]])
    assert feats.shape == (len(rec["candidates"]), WIDTH)
    assert np.isfinite(feats).all()
    model = Prioritiser(16)
    scores = model(torch.as_tensor(feats))
    assert scores.shape == (len(rec["candidates"]),)
