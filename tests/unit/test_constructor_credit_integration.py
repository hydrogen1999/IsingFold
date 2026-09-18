"""A separate state baseline must not change deployment or lose checkpoint history."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import networkx as nx
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "probes"))
import constructor_curriculum as cc
from constructor_checkpoint import ValidationCheckpoint
from constructor_state_value import ConstructorStateValue


def task(name):
    return cc.Task(name, nx.path_graph(2), nx.cycle_graph(4), name, ground_energy=-1.)


def test_linear_actor_can_use_independent_state_baseline():
    args = cc.parse(["--actor", "linear", "--baseline", "loo_value"])
    assert args.baseline == "loo_value"
    for flags in (["--critic-width", "0"], ["--critic-learning-rate", "nan"],
                  ["--reset-state-value"]):
        with pytest.raises(SystemExit):
            cc.parse(flags)


def test_selected_checkpoint_keeps_critic_at_same_iteration(tmp_path):
    actor = cc.make_actor("linear", 8)
    critic = ConstructorStateValue(cc.FEATURE_WIDTH, 8)
    cp = ValidationCheckpoint(tmp_path / "model.pt", {}, "quality",
                              extra_state=lambda: {"state_value_state": critic.state_dict()})
    cp.consider(actor, {"heldout": {"deployment_utility": {"policy": .9}}}, "good")
    with torch.no_grad():
        critic.net[-1].bias.fill_(2.)
    cp.consider(actor, {"heldout": {"deployment_utility": {"policy": .1}}}, "bad")
    state = torch.load(tmp_path / "model.best.pt", weights_only=False)
    assert state["state_value_state"]["net.2.bias"].item() == 0


def test_initial_checkpoint_cannot_have_trained_on_an_unsampled_reserved_lineage(tmp_path):
    path = tmp_path / "init.pt"
    actor = cc.make_actor("linear", 8)
    torch.save({"actor": "linear", "features": "tiny", "state": actor.state_dict(),
                "training_provenance": {"training_lineages": ["test"], "complete": True}}, path)
    raw_tasks = [SimpleNamespace(name=name, lineage=name, logical=nx.path_graph(2),
                                host=nx.cycle_graph(4), witness=None)
                 for name in ("train", "validation", "test")]
    train, held = cc.manifest_split_sets(raw_tasks,
                 {"train": ["train"], "validation": ["validation"], "test": ["test"]}, [], 1, 1, 0)
    args = cc.parse(["--init", str(path), "--iterations", "0"])
    with pytest.raises(ValueError, match="overlap"):
        cc.run(args, train, held)


def test_quality_rl_state_baseline_smoke_and_provenance(tmp_path, capsys):
    args = cc.parse(["--stage", "a", "--train", "1", "--heldout", "1", "--iterations", "2",
                     "--episodes", "2", "--objective", "quality", "--comparison", "none",
                     "--features", "local", "--baseline", "loo_value", "--stop-bias", "-6",
                     "--max-steps", "12", "--reward-reads", "4", "--selection-reads", "4",
                     "--assessment-reads", "4", "--select-cap", "1", "--deadline", "1",
                     "--episode-seconds", "1", "--eval-sets", "heldout", "--out", str(tmp_path / "q.pt")])
    with cc.no_completion_solver():
        cc.run(args, [task("train")], [task("validation")])
    state = torch.load(tmp_path / "q.pt", weights_only=False)
    assert state["training_provenance"]["training_lineages"] == ["train"]
    assert state["training_provenance"]["complete"]
    assert state["contract"]["train_lineages"] == ["train"]
    assert "state_value_state" in state
    deployed = cc.make_actor("linear", 8, cc.FEATURE_WIDTHS["local"])
    deployed.load_state_dict(state["state"], strict=True)
    state["state_value_schema"] = "incompatible-return-definition"
    altered = tmp_path / "wrong_critic.pt"
    torch.save(state, altered)
    args.init = str(altered)
    args.iterations = 0
    with pytest.raises(ValueError, match="critic schema changed"):
        cc.run(args, [task("train")], [task("validation")])
    # Reusing this checkpoint on a known training lineage as held-out must fail.
    args.init = str(tmp_path / "q.pt")
    args.iterations = 0
    with pytest.raises(ValueError, match="overlap"):
        cc.run(args, [task("different")], [task("train")])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    assert rows[0]["state_value_training_only"] is True
