"""Policy-guided discrepancy search runs on a toy and returns a well-formed record."""
from pathlib import Path

import pytest

PROBES = Path(__file__).resolve().parents[2] / "probes"
SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture(autouse=True)
def _paths(monkeypatch):
    monkeypatch.setenv("ISINGFOLD_SRC", str(SRC))
    monkeypatch.syspath_prepend(str(PROBES))
    monkeypatch.syspath_prepend(str(SRC))


def test_search_runs_with_the_unlearned_order_on_a_toy():
    from _context import qubit_budget
    from candidate_features import FeatureContext
    from search_construct import Scorer, search
    from . import test_witness_replay as tw

    task, witness = tw._planted_task(side=5, fill=0.7, seed=0, lmax=2)
    r = search(task, Scorer(None, FeatureContext(task, qubit_budget(witness))), deadline=20.0, max_steps=400)
    assert set(r) == {"valid", "attempts", "secs", "frac"}
    assert r["attempts"] >= 1 and 0.0 <= r["frac"] <= 1.0
