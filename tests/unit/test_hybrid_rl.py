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


def test_fast_layout_places_every_variable_in_milliseconds():
    import time
    import torch
    from _context import qubit_budget
    from candidate_features import FeatureContext
    from fast_layout import sample_layout
    from train_prioritiser import Prioritiser
    from . import test_witness_replay as tw

    task, witness = tw._planted_task(side=10, fill=0.85, seed=1, lmax=2)
    torch.manual_seed(0)
    model = Prioritiser(16)
    t0 = time.time()
    chains, logps = sample_layout(task, model, FeatureContext(task, qubit_budget(witness)), 1.0,
                                  np.random.default_rng(0), train=True)
    assert set(chains) == set(task.logical.nodes())
    assert len({next(iter(c)) for c in chains.values()}) == len(chains)
    assert len(logps) == len(chains)
    assert time.time() - t0 < 5.0


def test_complete_rejects_late_router_answer_and_resets_last(monkeypatch):
    import train_hybrid_rl as hybrid
    now = [0.0]
    received = []
    monkeypatch.setattr(hybrid.time, "monotonic", lambda: now[0])
    def router(task, roots, seed, tries, budget=None, timeout=None):
        received.append(timeout)
        now[0] += 3.0
        return {0: frozenset({0})}
    monkeypatch.setattr(hybrid, "attempt", router)
    hybrid.complete.last = {0: frozenset({99})}
    ok, seconds, attempts = hybrid.complete(None, None, 2.0, 1, 0)
    assert not ok and hybrid.complete.last is None
    assert seconds == 3.0 and attempts == 1 and received == [2.0]


def test_seeded_validity_rejects_disconnected_and_outside_chains():
    import networkx as nx
    from seeded_minorminer import valid
    logical, host = nx.path_graph(2), nx.path_graph(4)
    assert valid({0: {0, 1}, 1: {2}}, logical, host)
    assert not valid({0: {0, 2}, 1: {1}}, logical, host)
    assert not valid({0: {0, 99}, 1: {1}}, logical, host)
    assert not valid({0: {0, 1}, 1: {1}}, logical, host)


def test_router_handles_isolated_only_graph_and_enforces_budget(monkeypatch):
    import networkx as nx
    import seeded_minorminer as seeded
    from types import SimpleNamespace
    task = SimpleNamespace(logical=nx.empty_graph(2), host=nx.path_graph(3))
    def forbidden(*args, **kwargs):
        pytest.fail("an edgeless logical graph needs no router call")
    monkeypatch.setattr(seeded.minorminer, "find_embedding", forbidden)
    result = seeded.attempt(task, None, 0, 1, budget=2)
    assert seeded.valid(result, task.logical, task.host)
    assert seeded.attempt(task, None, 0, 1, budget=1) is None


def test_router_passes_timeout_and_rejects_backend_lateness(monkeypatch):
    import networkx as nx
    import seeded_minorminer as seeded
    from types import SimpleNamespace
    task = SimpleNamespace(logical=nx.path_graph(2), host=nx.path_graph(3))
    now = [0.0]
    monkeypatch.setattr(seeded.time, "monotonic", lambda: now[0])
    def router(*args, **kwargs):
        assert kwargs["timeout"] == 0.25
        now[0] = 0.3
        return {0: [0], 1: [1]}
    monkeypatch.setattr(seeded.minorminer, "find_embedding", router)
    assert seeded.attempt(task, None, 0, 1, timeout=0.25) is None


def test_lineage_split_cannot_leak_sibling_instances():
    from types import SimpleNamespace
    from train_hybrid_rl import split_by_lineage, experiment_seed
    tasks = [SimpleNamespace(name=f"{group}-{i}", lineage=group)
             for group in ("a", "b", "c") for i in range(2)]
    train, validation = split_by_lineage(tasks, 0.5, 7)
    assert not {t.lineage for t in train} & {t.lineage for t in validation}
    assert len(train) + len(validation) == len(tasks)
    phases = ("reference", "reward", "selection", "assessment")
    assert len({experiment_seed(0, phase, "task", 1) for phase in phases}) == len(phases)
    with pytest.raises(ValueError, match="lineages"):
        split_by_lineage(tasks[:2], 0.5, 7)


def test_quality_reward_and_checkpoint_do_not_drop_failed_instances():
    from train_hybrid_rl import quality_reward, validation_checkpoint_key
    assert quality_reward(1.0, 0.0, 10) > 0.5
    assert quality_reward(None, 0.0, 10) == 0.6
    assert quality_reward(0.1, 0.2, 1) > quality_reward(0.2, 0.2, 1)
    key = validation_checkpoint_key([
        {"valid": True, "residual": 0.1}, {"valid": False, "residual": None}], "quality")
    assert key == (0.5, 0.5, -0.1)
    missing = validation_checkpoint_key([{"valid": True, "residual": None}], "quality")
    assert missing == (1.0, 0.0, -float("inf"))


def test_evaluation_counts_proposal_time_before_routing(monkeypatch):
    import train_hybrid_rl as hybrid
    from types import SimpleNamespace
    now = [0.0]
    monkeypatch.setattr(hybrid.time, "monotonic", lambda: now[0])
    def propose(_):
        now[0] = 3.0
        return {}
    def forbidden(*args, **kwargs):
        pytest.fail("routing after proposal exhausted the deadline")
    monkeypatch.setattr(hybrid, "complete", forbidden)
    result = hybrid.evaluate_arm(SimpleNamespace(name="toy"), propose, deadline=2,
                                 router_seconds=1, tries=1, budget=2,
                                 select_cap=2, objective="valid", seed=0)
    assert not result["valid"] and result["deadline_overrun_seconds"] == 1


def test_evaluation_selection_has_matched_cap_and_independent_assessment(monkeypatch):
    import train_hybrid_rl as hybrid
    from types import SimpleNamespace
    now, calls = [0.0], []
    monkeypatch.setattr(hybrid.time, "monotonic", lambda: now[0])
    def router(task, roots, deadline, tries, seed, budget=None):
        now[0] += 0.1
        router.last = {0: frozenset({len(calls)})}
        return True, 0.1, 1
    def measure(task, chain, seed, reads=256):
        calls.append((seed, reads))
        now[0] += 0.1
        return float(10 - len(calls))
    monkeypatch.setattr(hybrid, "complete", router)
    monkeypatch.setattr(hybrid, "measure_residual", measure)
    result = hybrid.evaluate_arm(SimpleNamespace(name="toy"), None, deadline=2,
                                 router_seconds=1, tries=1, budget=2,
                                 select_cap=2, objective="quality", seed=0,
                                 selection_reads=16, assessment_reads=32)
    assert result["valid"] and result["selection_reads"] == 32
    assert result["assessment_reads"] == 32 and result["attempts"] == 2
    assert len(calls) == 3 and len({seed for seed, _ in calls}) == 3
    assert result["total_seconds"] > result["deployment_seconds"]


def test_late_measurement_cannot_select_a_new_winner(monkeypatch):
    import train_hybrid_rl as hybrid
    from types import SimpleNamespace
    now = [0.0]
    monkeypatch.setattr(hybrid.time, "monotonic", lambda: now[0])
    def router(*args, **kwargs):
        now[0] += 0.5
        router.last = {0: frozenset({0})}
        return True, 0.5, 1
    def measure(*args, **kwargs):
        now[0] += 2.0
        return 0.0
    monkeypatch.setattr(hybrid, "complete", router)
    monkeypatch.setattr(hybrid, "measure_residual", measure)
    result = hybrid.evaluate_arm(SimpleNamespace(name="toy"), None, deadline=2,
                                 router_seconds=1, tries=1, budget=2,
                                 select_cap=2, objective="quality", seed=0)
    assert not result["valid"] and result["selection_reads"] == 256
    assert result["assessment_reads"] == 0 and result["deadline_overrun_seconds"] == 0.5


def test_initial_quality_checkpoint_saved_without_training(monkeypatch, tmp_path):
    import torch
    import train_hybrid_rl as hybrid
    from dataclasses import replace
    from . import test_witness_replay as tw
    task, witness = tw._planted_task(side=4, fill=0.6, seed=0, lmax=2)
    class ForbiddenWitness(dict):
        def items(self):
            pytest.fail("deployment accessed the training witness")
        def values(self):
            pytest.fail("deployment accessed the training witness")
        def __iter__(self):
            pytest.fail("deployment accessed the training witness")
    tasks = [replace(task, name=f"toy-fill80-a3.0-{i}",
                     lineage=f"toy-fill80-a3.0-{i}-l{i}",
                     witness=ForbiddenWitness()) for i in range(2)]
    monkeypatch.setattr(hybrid, "load_instances", lambda _: tasks)
    # The mock constructor returns its own output; deployment must not inspect the witness.
    monkeypatch.setattr(hybrid, "sample_layout", lambda t, *args, **kw: (witness, []))
    def complete(t, roots, deadline, tries, seed, budget=None):
        assert budget == t.host.number_of_nodes()  # no privileged witness budget by default
        complete.last = witness
        return True, 0.0, 1
    monkeypatch.setattr(hybrid, "complete", complete)
    monkeypatch.setattr(hybrid, "measure_residual", lambda *args, **kw: 0.2)
    output = tmp_path / "nested" / "model.pt"
    monkeypatch.setattr(hybrid.sys, "argv", ["train_hybrid_rl", "--corpus", "unused",
                                           "--out", str(output), "--iterations", "0",
                                           "--fast", "--objective", "quality", "--select-cap", "1"])
    assert hybrid.main() == 0
    checkpoint = torch.load(output, weights_only=False)
    assert checkpoint["validation_score"] == (1.0, 1.0, -0.2)
    assert checkpoint["config"]["qubit_cap"] == 0
    assert Path(str(output) + ".last").exists()
