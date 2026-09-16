"""Regression contracts for the quality objective and the contact-growth evaluator."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _paths(monkeypatch):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("ISINGFOLD_SRC", str(root / "src"))
    monkeypatch.syspath_prepend(str(root / "probes"))
    monkeypatch.syspath_prepend(str(root / "src"))


@pytest.mark.parametrize("fraction,cap,host,expected", [
    (0.5, 4, 10, 0),    # exhausted declared budget
    (0.5, 10, 4, 0),    # exhausted physical hardware
    (0.0, 10, 10, 0),   # a zero-spend control must really be a no-op
    (0.5, 5, 10, 1),    # budget limits added qubits
    (1.0, 20, 6, 2),    # remaining physical space limits growth
])
def test_spend_respects_zero_slack_and_physical_host(fraction, cap, host, expected):
    from train_contact_policy import spend_limit
    assert spend_limit({0: frozenset(range(4))}, fraction, cap, host) == expected


def test_lineage_split_chooses_one_representative_without_overlap():
    from train_contact_policy import split_representatives
    tasks = [SimpleNamespace(name=f"{lineage}-{variant}", lineage=lineage)
             for lineage in ("a", "b", "c", "d") for variant in range(3)]
    train, validation = split_representatives(tasks, 2, 2, 4)
    assert len({t.lineage for t in train}) == len(train) == 2
    assert len({t.lineage for t in validation}) == len(validation) == 2
    assert not ({t.lineage for t in train} & {t.lineage for t in validation})
    with pytest.raises(ValueError, match="fewer distinct"):
        split_representatives(tasks, 4, 1, 4)


def test_reinforce_is_leave_one_out_and_uses_full_trajectory_score():
    import torch
    from train_contact_policy import reinforce_loss
    a = torch.tensor(-0.2, requires_grad=True)
    b = torch.tensor(-0.3, requires_grad=True)
    c = torch.tensor(-0.4, requires_grad=True)
    loss = reinforce_loss([(3.0, [a, b]), (1.0, [c])])
    loss.backward()
    # Baselines are respectively 1 and 3; averaging two complete trajectories
    # gives the same coefficient to both decisions of the longer trajectory.
    assert a.grad.item() == pytest.approx(-1.0)
    assert b.grad.item() == pytest.approx(-1.0)
    assert c.grad.item() == pytest.approx(1.0)


def test_selection_uses_only_growth_candidates_and_fresh_assessment():
    from train_contact_policy import select_and_assess
    grown_a, grown_b = {0: frozenset([0, 1])}, {0: frozenset([0, 2])}
    calls = []

    def measure(chains, seed, assess):
        calls.append((chains, seed, assess))
        return 0.4 if chains == grown_a else 0.2

    score, failures, failed_arm = select_and_assess(
        [grown_a, grown_b], measure, lambda i: 10 + i, 100, 0.7, 0.05, 2)
    # Even when the separately assessed start is better, it is not an inferred candidate.
    assert score == 0.4 and failures == 0 and not failed_arm
    assert len([c for c in calls if not c[2]]) == 2
    assert calls[-1] == (grown_a, 100, True)
    assert not ({seed for _, seed, assess in calls if not assess} & {100})


@pytest.mark.parametrize("failure_mode", ["selection", "assessment", "nan", "over_cap"])
def test_failed_arm_is_penalized_not_silently_excluded(failure_mode):
    from train_contact_policy import select_and_assess
    ch = {0: frozenset([0, 1])}

    def measure(chains, seed, assess):
        if failure_mode == "nan":
            return float("nan")
        if failure_mode == "selection" or (failure_mode == "assessment" and assess):
            return None
        return 0.8

    score, _, failed_arm = select_and_assess(
        [ch], measure, lambda i: 10 + i, 100, 0.6, 0.05,
        1 if failure_mode == "over_cap" else 2)
    assert failed_arm
    assert score == pytest.approx(0.55)


def test_measurement_blocks_are_domain_separated_and_stable():
    from train_contact_policy import block_seed
    seeds = {block_seed(0, domain, iteration, "instance", episode)
             for domain in ("train-start", "train-growth", "validation-selection", "validation-assessment")
             for iteration in range(10) for episode in range(4)}
    assert len(seeds) == 4 * 10 * 4
    assert block_seed(7, "validation-selection", "instance", 0) == block_seed(7, "validation-selection", "instance", 0)


def test_sampling_uses_explicit_rng_without_touching_torch_global_rng():
    import networkx as nx
    import torch
    from candidate_features import FeatureContext, FRONTIER_WIDTH, WIDTH
    from train_contact_policy import episode
    from train_prioritiser import Prioritiser

    host = nx.Graph([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)])
    logical = nx.Graph([(0, 1)])
    task = SimpleNamespace(name="toy", host=host, logical=logical, problem=None)
    start = {0: frozenset([0]), 1: frozenset([1])}
    model = Prioritiser(8, in_dim=WIDTH + FRONTIER_WIDTH)
    before = torch.random.get_rng_state().clone()
    result_a, logps = episode(task, start, model, FeatureContext(task), 1, 1.0, np.random.default_rng(2))
    result_b, _ = episode(task, start, model, FeatureContext(task), 1, 1.0, np.random.default_rng(2))
    assert result_a == result_b
    assert torch.equal(before, torch.random.get_rng_state())
    assert len(logps) == 1 and logps[0].requires_grad
    random_result, random_logps = episode(
        task, start, None, FeatureContext(task), 1, 1.0, np.random.default_rng(2), train=False)
    assert sum(map(len, random_result.values())) == 3
    assert random_logps == []


def test_main_smoke_records_occupancy_and_validation_checkpoint(monkeypatch, tmp_path, capsys):
    import networkx as nx
    import torch
    import train_contact_policy as probe

    host = nx.Graph([(0, 1), (0, 2), (1, 2), (0, 3), (1, 3)])
    tasks = [SimpleNamespace(name=f"toy-{i}", lineage=f"lineage-{i}",
                             logical=nx.Graph([(0, 1)]), host=host, problem=None,
                             witness={0: frozenset([0]), 1: frozenset([1])}) for i in range(2)]
    monkeypatch.setattr(probe, "load_instances", lambda _: tasks)
    calls = []

    def controller(tasks, context, policy, **kwargs):
        chains = kwargs["initializer"](None, None, None)
        calls.append((kwargs["seed"], kwargs["reward_reads"], sum(map(len, chains.values()))))
        return [SimpleNamespace(returned_valid=True, utility=0.7,
                                mean_energy_residual=0.1)]

    monkeypatch.setattr(probe, "run_controller", controller)
    out = tmp_path / "nested" / "model.pt"
    monkeypatch.setattr("sys.argv", ["train_contact_policy", "--corpus", "unused",
                        "--out", str(out), "--start", "witness", "--train-lineages", "1",
                        "--eval-lineages", "1", "--iterations", "1", "--eval-every", "1",
                        "--eval-k", "2", "--episodes-per-instance", "2", "--width", "8",
                        "--spend", "0.5", "--qubit-cap", "4", "--reads", "2", "--assess-reads", "3"])
    assert probe.main() == 0
    checkpoint = torch.load(out, weights_only=True)
    assert checkpoint["checkpoint_selection_split"] == "validation"
    assert checkpoint["evaluation_scope"] == "continuation_diagnostic_not_end_to_end_embedding"
    assert "force_spend" not in checkpoint
    assert len(checkpoint["validation_lineages"]) == 1
    assert Path(str(out) + ".last").exists()
    assert '"mean_actual_start_occupancy": 0.5' in capsys.readouterr().out
    # A fresh assessment stream cannot accidentally reuse a selection stream.
    assert not ({seed for seed, reads, _ in calls if reads == 2} &
                {seed for seed, reads, _ in calls if reads == 3})
    # Each of two validation calls has K=2 grown candidates for each of two arms.
    # The supplied start is evaluated only as a separate assessment reference.
    for t in tasks:
        for candidate in range(2):
            seed = probe.block_seed(0, "validation-selection", t.name, candidate)
            matching = [used for observed, reads, used in calls if observed == seed]
            if matching:  # Only the held-out representative is evaluated.
                assert len(matching) == 4 and matching == [3, 3, 3, 3]
