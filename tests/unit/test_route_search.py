"""Our completion search finishes a planted instance from its witness roots on a toy."""
from pathlib import Path

import pytest

PROBES = Path(__file__).resolve().parents[2] / "probes"
SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture(autouse=True)
def _paths(monkeypatch):
    monkeypatch.setenv("ISINGFOLD_SRC", str(SRC))
    monkeypatch.syspath_prepend(str(PROBES))
    monkeypatch.syspath_prepend(str(SRC))


def test_completion_from_witness_roots_is_valid_on_a_toy():
    from _context import qubit_budget
    from placement_completion import witness_roots
    from route_search import complete
    from seeded_minorminer import valid
    from . import test_witness_replay as tw

    for seed in (0, 1, 2):
        task, witness = tw._planted_task(side=8, fill=0.8, seed=seed, lmax=2)
        roots = witness_roots(task, witness)
        chains = complete(task.host, task.logical, roots, qubit_budget(witness), deadline=10.0)
        assert chains is not None, "seed %d: no completion" % seed
        assert valid(chains, task.logical, task.host)


def test_negotiated_completion_from_witness_roots_on_larger_toys():
    from _context import qubit_budget
    from placement_completion import witness_roots
    from route_search import negotiate
    from seeded_minorminer import valid
    from . import test_witness_replay as tw

    done = 0
    for seed in range(1, 7):
        task, witness = tw._planted_task(side=10, fill=0.85, seed=seed, lmax=2)
        chains = negotiate(task.host, task.logical, witness_roots(task, witness), qubit_budget(witness), deadline=10.0)
        done += chains is not None and valid(chains, task.logical, task.host)
    assert done >= 5
