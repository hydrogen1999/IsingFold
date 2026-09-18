"""Regression gates for split leakage, missing failures, and diagnostic receipts."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import pytest
import torch

_path = Path(__file__).resolve().parents[2] / "probes" / "policy_excess.py"
_spec = importlib.util.spec_from_file_location("policy_excess", _path)
pe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pe)


def task(name):
    return pe.cc.Task(name, nx.path_graph(2), nx.path_graph(4), "lineage-" + name,
                      ground_energy=-1.)


def loader_args(path, **changes):
    values = dict(corpus=str(path), cells="", n_tasks=1, seed=4, role="validation",
                  exploratory_repartition=False)
    return SimpleNamespace(**(values | changes))


@pytest.mark.parametrize("role", ["validation", "test"])
def test_manifest_loader_only_returns_requested_role_without_witness(tmp_path, monkeypatch, role):
    tasks = [task(name) for name in ("train", "validation", "test")]
    (tmp_path / "splits.json").write_text(json.dumps({r: [r] for r in ("train", "validation", "test")}))
    monkeypatch.setattr("isingfold.rl.data.generate.load_instances", lambda _: tasks)
    selected = pe.audit_tasks(loader_args(tmp_path, role=role))
    assert [t.name for t in selected] == [role]
    assert selected[0].prefix_source is None
    with pytest.raises(AssertionError, match="witness accessed"):
        selected[0].witness


def test_missing_manifest_never_silently_repartitions(tmp_path, monkeypatch):
    monkeypatch.setattr("isingfold.rl.data.generate.load_instances", lambda _: [task("x")])
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="no train/validation/test"):
        pe.audit_tasks(loader_args(tmp_path))
    exploratory = pe.audit_tasks(loader_args(tmp_path, exploratory_repartition=True))
    assert len(exploratory) == 1 and exploratory[0].prefix_source is None
    with pytest.raises(ValueError, match="test evaluation requires"):
        pe.audit_tasks(loader_args(tmp_path, exploratory_repartition=True, role="test"))


def checkpoint(path, lineages=None):
    actor = pe.cc.make_actor("linear", 32, pe.cc.FEATURE_WIDTHS["local"])
    payload = {"state": actor.state_dict(), "actor": "linear", "features": "local"}
    if lineages is not None:
        payload["training_provenance"] = {"training_lineages": lineages, "complete": True}
    torch.save(payload, path)


def test_known_checkpoint_overlap_rejected_before_any_rollout(tmp_path, monkeypatch):
    path = tmp_path / "actor.pt"
    checkpoint(path, ["lineage-heldout"])
    monkeypatch.setattr(pe, "audit_tasks", lambda _: [task("heldout")])
    monkeypatch.setattr(pe, "episode", lambda *a, **k: pytest.fail("rollout before overlap guard"))
    with pytest.raises(ValueError, match="overlap held-out"):
        pe.main(["--corpus", str(tmp_path), "--init", str(path), "--n-tasks", "1"])


def test_failed_tasks_stay_in_denominator_and_receipts_record_actual_work(tmp_path, monkeypatch, capsys):
    path = tmp_path / "legacy.pt"
    checkpoint(path)  # Unknown ancestry must remain visible in the output.
    tasks = [task("valid"), task("failed")]
    monkeypatch.setattr(pe, "audit_tasks", lambda _: tasks)
    chains = {0: frozenset([0]), 1: frozenset([1])}
    calls = []

    def rollout(t, *args, **kwargs):
        assert t.prefix_source is None and kwargs["evaluate_reward"] is False
        calls.append(t.name)
        valid = t.name == "valid"
        return {"valid": valid, "reason": "COMMIT" if valid else "HORIZON", "steps": 2,
                "terminal": SimpleNamespace(embedding=chains) if valid else None}

    baseline_calls = []

    def baseline(logical, host, seed):
        baseline_calls.append(seed)
        return None if len(baseline_calls) == 1 else chains

    measured = []

    def measure(t, emb, ctx, strength, reads, seed):
        measured.append((t.name, seed, reads))
        return .2, .6, .1

    monkeypatch.setattr(pe, "episode", rollout)
    monkeypatch.setattr(pe, "minorminer_initializer", lambda _: baseline)
    monkeypatch.setattr(pe, "prune", lambda host, emb, logical: emb)
    monkeypatch.setattr(pe, "measure", measure)
    assert pe.main(["--corpus", str(tmp_path), "--init", str(path), "--n-tasks", "2",
                    "--episodes", "2", "--reads", "16", "--seed", "9"]) == 0
    output = [json.loads(s) for s in capsys.readouterr().out.splitlines() if s.startswith("{")]
    header, first, failed, summary = output
    assert header["evaluation_role"] == "validation"
    assert header["heldout_ancestry_verified"] is False
    assert header["training_provenance"]["complete"] is False
    assert header["config"]["max_steps"] == 250
    assert header["contract"]["initial_checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(header["contract"]["code_sha256"]) == 64
    assert len(header["contract"]["heldout_fingerprints"]) == 2
    assert header["tasks"][1]["lineage"] == "lineage-failed"
    assert calls == ["valid", "failed", "failed"]
    assert baseline_calls == [9, 10]  # A failed policy never suppresses the baseline.
    assert measured == [("valid", 9, 16), ("valid", 10, 16), ("failed", 10011, 16)]
    assert first["policy_attempts"][0]["seed"] == 9
    assert [r["seed"] for r in failed["policy_attempts"]] == [109, 110]
    assert failed["policy"]["valid"] is False
    assert failed["policy"]["residual"] is None
    assert failed["policy"]["quality_utility"] == failed["policy"]["p_solve"] == 0
    assert failed["minorminer"]["valid"] is True
    assert summary["tasks"] == 2
    assert summary["policy_minorminer_pairs"] == 0
    for arm in ("policy", "pruned", "minorminer"):
        result = summary["arms"][arm]
        assert result["valid_tasks"] == result["failed_tasks"] == 1
        assert result["coverage"] == .5
        assert result["mean_residual_conditional"] == .2
        assert result["mean_broken_conditional"] == .1
        assert result["mean_p_solve_all_instances"] == .3
        assert result["mean_quality_utility_all_instances"] == pytest.approx(.475)
        assert result["assessment_reads"] == 16


def test_residual_and_solve_probability_use_one_checked_read_block(monkeypatch):
    sampled = []
    monkeypatch.setattr(pe, "strength_registry", lambda *args: [1., 2.])
    monkeypatch.setattr(pe, "compile_program", lambda *args: "program")

    def sample(*args, **kwargs):
        sampled.append(kwargs)
        return SimpleNamespace(mean_residual=.3, rate=.25, broken_fraction=.125,
                               reads=16, strength_index=1)

    monkeypatch.setattr(pe, "sample_program", sample)
    ctx = SimpleNamespace(strength_ratios=(1., 2.), epsilon_strength=.01)
    assert pe.measure(task("one"), {0: [0], 1: [1]}, ctx, 1, 16, 42) == (.3, .25, .125)
    assert len(sampled) == 1 and sampled[0]["seed"] == 42
    with pytest.raises(ValueError, match="read/strength receipt"):
        pe.measure(task("one"), {0: [0], 1: [1]}, ctx, 1, 32, 42)


@pytest.mark.parametrize("extra", [["--n-tasks", "0"], ["--reads", "0"],
                                   ["--episode-seconds", "nan"],
                                   ["--role", "test", "--exploratory-repartition"]])
def test_invalid_cli_is_rejected_before_loading(extra):
    with pytest.raises(SystemExit):
        pe.main(["--corpus", "unused", "--init", "unused", *extra])
