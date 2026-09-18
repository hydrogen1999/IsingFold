"""Contracts needed to attribute an embedding-quality gain to policy learning."""
import json
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "probes"))
import constructor_curriculum as cc
from constructor_checkpoint import ValidationCheckpoint


def task(name, lineage):
    return cc.Task(name, nx.path_graph(2), nx.cycle_graph(4), lineage, ground_energy=-1.)


def test_lineage_leak_is_rejected_even_if_instance_names_differ():
    tasks = [task("a", "shared"), task("b", "shared"), task("c", "test")]
    split = {"train": ["a"], "validation": ["b"], "test": ["c"]}
    with pytest.raises(ValueError, match="lineage"):
        cc.manifest_split_sets(tasks, split, [], 1, 1, 0)


def test_synthetic_stages_cannot_claim_a_manifest_test_split():
    with pytest.raises(SystemExit):
        cc.parse(["--stage", "a", "--manifest-split", "--heldout-role", "test",
                  "--iterations", "0", "--init", "model.pt"])


def test_independently_generated_pairs_have_distinct_lineages():
    tasks = cc.generate("a", 3, 1, "development")
    assert len({t.lineage for t in tasks}) == 3


def test_repeated_lineage_is_not_counted_as_two_independent_tasks():
    tasks = [task("a", "same"), task("b", "same"), task("c", "validation")]
    split = {"train": ["a", "b"], "validation": ["c"], "test": []}
    with pytest.raises(ValueError, match="requested"):
        cc.manifest_split_sets(tasks, split, [], 2, 1, 0)


def test_feature_expansion_preserves_loaded_linear_policy(tmp_path):
    old = cc.make_actor("linear", 8, cc.FEATURE_WIDTHS["local"])
    with torch.no_grad():
        next(old.parameters()).copy_(torch.arange(20.)[None] / 20)
    path = tmp_path / "local.pt"
    torch.save({"actor": "linear", "features": "local", "state": old.state_dict()}, path)
    new = cc.make_actor("linear", 8, cc.FEATURE_WIDTHS["physics"])
    with pytest.raises(ValueError):
        cc.load_init(path, new, "linear", "physics")
    cc.load_init(path, new, "linear", "physics", expand=True)
    rows = torch.randn(5, 32)
    assert torch.equal(old(rows[:, :20]), new(rows))


def test_best_checkpoint_retains_precollapse_policy_and_never_uses_test(tmp_path):
    actor = cc.make_actor("linear", 8)
    checkpoint = ValidationCheckpoint(tmp_path / "model.pt", {}, "quality")
    good = {"heldout": {"deployment_utility": {"policy": .9}}}
    bad = {"heldout": {"deployment_utility": {"policy": .1}}}
    assert checkpoint.consider(actor, good, "iter9")
    with torch.no_grad():
        next(actor.parameters()).fill_(4.)
    assert not checkpoint.consider(actor, bad, "final")
    saved = torch.load(tmp_path / "model.best.pt", weights_only=False)
    assert saved["selected_tag"] == "iter9"
    assert torch.count_nonzero(next(iter(saved["state"].values()))) == 0
    test = ValidationCheckpoint(tmp_path / "test.pt", {}, "quality", "test")
    assert not test.consider(actor, good, "final")
    assert not (tmp_path / "test.best.pt").exists()


def test_quality_eval_only_runs_once_and_honors_eval_sets(monkeypatch, capsys):
    args = cc.parse(["--iterations", "0", "--objective", "quality", "--eval-sets", "heldout",
                     "--comparison", "none", "--train", "1", "--heldout", "1"])
    calls = []
    def search(t, *a, **kw):
        calls.append(t.name)
        return {"valid": True, "residual": .1, "p_solve": .25, "quality_utility": .975,
                "assessment_ok": True, "chosen_qubits": 3, "chosen_longest_chain": 2}
    monkeypatch.setattr(cc, "evaluate_search", search)
    cc.run(args, [task("train", "t")], [task("validation", "v")])
    assert calls == ["validation"]
    rows = [json.loads(s) for s in capsys.readouterr().out.splitlines() if s.startswith("{")]
    evaluation = [r for r in rows if "evaluation" in r]
    assert len(evaluation) == 1 and evaluation[0]["evaluation"] == "final"
    assert evaluation[0]["deployment_p_solve"]["policy"] == .25


@pytest.mark.parametrize("flag,value", [("--advantage-scale", "0"), ("--entropy-coef", "nan"),
                                         ("--proposal-fraction", "1"), ("--data-seed", "-1")])
def test_invalid_training_controls_are_rejected(flag, value):
    with pytest.raises(SystemExit):
        cc.parse([flag, value])


def test_quality_training_smoke_from_empty(tmp_path):
    args = cc.parse(["--stage", "a", "--train", "1", "--heldout", "1", "--iterations", "1",
                     "--episodes", "2", "--objective", "quality", "--comparison", "none",
                     "--features", "physics", "--stop-bias", "-6", "--max-steps", "12",
                     "--reward-reads", "4", "--selection-reads", "4", "--assessment-reads", "4",
                     "--select-cap", "1", "--deadline", "1", "--episode-seconds", "1",
                     "--eval-sets", "heldout", "--out", str(tmp_path / "q.pt")])
    with cc.no_completion_solver():
        summary = cc.run(args, [task("train", "t")], [task("validation", "v")])
    assert summary["objective"] == "quality"
    assert (tmp_path / "q.best.pt").exists()
    assert np.isfinite(summary["selected_checkpoint"]["validation_score"])
