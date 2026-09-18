"""One optimizer/checkpoint iteration with real constructor features and mocked outcomes.

These CLI smoke tests call no annealer and claim no embedding performance.
"""
import json
import sys
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch

from isingfold.rl.contracts import Candidate, Opcode, WorkVector


class PublicTask:
    def __init__(self, lineage, sibling):
        self.name, self.lineage = f"toy-{lineage}-{sibling}", lineage
        self.host, self.logical = nx.path_graph(6), nx.path_graph(2)
        self.problem = SimpleNamespace(h={0: 1., 1: -.5}, j={(0, 1): -2.})

    @property
    def witness(self):
        raise AssertionError("CLI construction/validation must not access witness")

    @property
    def initial_embedding(self):
        raise AssertionError("CLI must not require a supplied embedding")

    @property
    def ground_energy(self):
        raise AssertionError("mocked sampler is the only label source in this smoke test")


def test_cli_training_checkpoint_with_real_features_no_witness_or_baseline(tmp_path, monkeypatch, capsys):
    import train_constructor_rl as cli
    import constructor_protocol as protocol
    import constructor_baseline as baseline
    from constructor_features import ConstructorFeatureContext, WIDTH, FEATURE_VERSION
    tasks = [PublicTask(f"lineage-{i}", j) for i in range(2) for j in range(2)]
    monkeypatch.setattr(cli, "load_instances", lambda _: tasks)
    def forbidden(*args, **kwargs):
        pytest.fail("--comparison none must never call a completion/baseline solver")
    monkeypatch.setattr(baseline, "baseline_proposal", forbidden)
    import seeded_minorminer
    monkeypatch.setattr(seeded_minorminer, "attempt", forbidden)
    calls, feature_widths = [], []
    def fake_episode(task, model, fc, temperature, max_steps, rng, seconds, train=True, **kwargs):
        assert isinstance(fc, ConstructorFeatureContext)
        assert fc.budget == kwargs["qubit_cap"] == len(task.host)
        calls.append((task.lineage, kwargs["evaluate_reward"]))
        rows = []
        for q in (0, 3):
            candidate = Candidate(opcode=Opcode.PLACE, affected=(0,),
                                  old_chains={0: frozenset()}, new_chains={0: frozenset({q})},
                                  work=WorkVector(decisions=1), payload_key=f"place-{q}")
            rows.append(fc.observe(candidate, {}))
        features = torch.tensor(np.stack(rows))
        feature_widths.append(features.shape[1])
        dist, value = model.distribution_value(features, temperature)
        probabilities = dist.probs.detach().numpy().astype(float)
        probabilities /= probabilities.sum()
        action = int(rng.choice(2, p=probabilities))
        d = SimpleNamespace(log_prob=dist.log_prob(torch.tensor(action)), entropy=dist.entropy(),
                            value=value, support_size=2)
        base_return = .8 + .1 * action
        terminal = SimpleNamespace(returned_valid=True, selected_index=0,
                                   embedding={0: frozenset({0}), 1: frozenset({1})})
        return {"decisions": [d], "rewards": [base_return], "togo": [base_return],
                "potentials": [0.], "base_return": base_return, "return": base_return,
                "terminal_potential": 0., "terminal": terminal, "valid": True,
                "residual": .2 if kwargs["evaluate_reward"] else None, "frac": 1., "steps": 1}
    monkeypatch.setattr(cli, "episode", fake_episode)
    monkeypatch.setattr(protocol, "measure_terminal", lambda *args, **kwargs: .25)
    out = tmp_path / "constructor.pt"
    monkeypatch.setattr(sys, "argv", ["train_constructor_rl.py", "--corpus", "mock",
                                      "--out", str(out), "--comparison", "none",
                                      "--iterations", "1", "--eval-every", "1",
                                      "--instances-per-iteration", "1", "--episodes-per-instance", "2",
                                      "--width", "8", "--deadline", "1", "--select-cap", "1",
                                      "--selection-reads", "3", "--assessment-reads", "5"])
    assert cli.main() == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    manifest = lines[0]
    assert manifest["completion_solver"] is None and manifest["inference_initial_state"] == "empty"
    assert manifest["qubit_objective_penalty"] == 0 and manifest["comparison_only_solver"] == "none"
    assert set(manifest["train_lineages"]).isdisjoint(manifest["validation_lineages"])
    assert {lineage for lineage, training in calls if training} <= set(manifest["train_lineages"])
    assert {lineage for lineage, training in calls if not training} == set(manifest["validation_lineages"])
    assert set(feature_widths) == {WIDTH}
    evaluation = [line for line in lines if "evaluation" in line]
    assert [line["evaluation"] for line in evaluation] == ["init", "iter 0"]
    assert all(r["arm"] == "policy" and r["selection_reads"] == 3 and r["assessment_reads"] == 5
               for line in evaluation for r in line["arms"])
    training = next(line for line in lines if "training_iteration" in line)
    assert training["gradient_norm_before_clip"] > 0 and training["decisions"] > 0
    initial = torch.load(out, weights_only=True)
    last = torch.load(str(out) + ".last", weights_only=True)
    assert last["model_spec"]["feature_version"] == FEATURE_VERSION
    assert last["model_spec"]["in_dim"] == WIDTH
    assert last["protocol"] == "independent-constructor-v1"
    assert last["current_split_verified"] and last["lineage_provenance_verified"]
    assert last["training_lineages"] == manifest["train_lineages"]
    assert last["validation_lineages"] == manifest["validation_lineages"]
    assert any(not torch.equal(initial["state"][k], last["state"][k]) for k in initial["state"])


def test_cli_rejects_incompatible_checkpoint_schema(tmp_path, monkeypatch):
    import train_constructor_rl as cli
    monkeypatch.setattr(cli, "load_instances", lambda _: [PublicTask("a", 0), PublicTask("b", 0)])
    wrong = tmp_path / "old.pt"
    torch.save({"state": {}, "width": 64, "model_spec": {"actor": "local",
                "feature_version": "candidate-v1", "in_dim": 35, "width": 64}}, wrong)
    monkeypatch.setattr(sys, "argv", ["train_constructor_rl.py", "--corpus", "mock", "--out",
                                      str(tmp_path / "new.pt"), "--init", str(wrong)])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def _checkpoint_init_case(tmp_path, monkeypatch, *, provenance=None, allow=False):
    """Prepare a compatible checkpoint and a zero-update CLI provenance check."""
    import train_constructor_rl as cli
    from constructor_features import FEATURE_VERSION, WIDTH
    from constructor_protocol import split_by_lineage
    from layout_policy import LayoutActorCritic
    tasks = [PublicTask("a", 0), PublicTask("b", 0)]
    train, validation = split_by_lineage(tasks, .25, 0)
    spec = {"actor": "contextual", "feature_version": FEATURE_VERSION, "in_dim": WIDTH, "width": 8}
    blob = {"state": LayoutActorCritic(8, WIDTH).state_dict(), "width": 8, "model_spec": spec}
    if provenance is not None:
        blob.update(provenance(train[0].lineage, validation[0].lineage))
    initial, output = tmp_path / "initial.pt", tmp_path / "output.pt"
    torch.save(blob, initial)
    monkeypatch.setattr(cli, "load_instances", lambda _: tasks)
    monkeypatch.setattr(cli, "evaluate_search", lambda *a, **kw: {"valid": False, "residual": None})
    args = ["train_constructor_rl.py", "--corpus", "mock", "--out", str(output),
            "--init", str(initial), "--width", "8", "--iterations", "0", "--comparison", "none"]
    if allow:
        args.append("--allow-unverified-init")
    monkeypatch.setattr(sys, "argv", args)
    return cli, output, train[0].lineage, validation[0].lineage


@pytest.mark.parametrize("allow", [False, True])
def test_init_cannot_train_on_current_validation_even_with_legacy_override(tmp_path, monkeypatch, allow):
    cli, _, _, _ = _checkpoint_init_case(tmp_path, monkeypatch, allow=allow,
        provenance=lambda train, validation: {"training_lineages": [validation],
                                               "lineage_provenance_verified": True})
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_disjoint_init_preserves_cumulative_training_lineages(tmp_path, monkeypatch, capsys):
    cli, output, train, validation = _checkpoint_init_case(tmp_path, monkeypatch,
        provenance=lambda train, validation: {"training_lineages": [train, "ancestor-corpus"],
                                               "lineage_provenance_verified": True})
    assert cli.main() == 0
    checkpoint = torch.load(output, weights_only=True)
    assert checkpoint["train_lineages"] == [train]
    assert set(checkpoint["training_lineages"]) == {train, "ancestor-corpus"}
    assert checkpoint["validation_lineages"] == [validation]
    assert checkpoint["lineage_provenance_verified"]
    assert checkpoint["init_lineage_provenance_verified"]
    manifest = json.loads(capsys.readouterr().out.splitlines()[0])
    assert manifest["training_lineages"] == checkpoint["training_lineages"]


@pytest.mark.parametrize("provenance", [None,
    lambda train, validation: {"training_lineages": [train], "lineage_provenance_verified": False}])
def test_unknown_or_previously_unverified_init_requires_explicit_flag(tmp_path, monkeypatch, provenance):
    cli, _, _, _ = _checkpoint_init_case(tmp_path, monkeypatch, provenance=provenance)
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2


def test_explicit_legacy_init_marks_checkpoint_and_log_unverified(tmp_path, monkeypatch, capsys):
    cli, output, train, _ = _checkpoint_init_case(tmp_path, monkeypatch, allow=True,
        provenance=lambda train, validation: {"train_lineages": ["legacy-known"]})
    assert cli.main() == 0
    checkpoint = torch.load(output, weights_only=True)
    assert checkpoint["current_split_verified"]
    assert not checkpoint["lineage_provenance_verified"]
    assert not checkpoint["init_lineage_provenance_verified"]
    assert set(checkpoint["training_lineages"]) == {train, "legacy-known"}
    manifest = json.loads(capsys.readouterr().out.splitlines()[0])
    assert not manifest["lineage_provenance_verified"]
    assert manifest["current_split_verified"]
