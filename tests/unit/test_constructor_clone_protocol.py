"""Cloning respects real corpus partitions, records ancestry, and avoids teacher cycles."""
import hashlib
import json
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest
import torch

import constructor_clone as clone
import constructor_rollout as rollout
from isingfold.rl.contracts import DecisionState, Mode, Opcode
from isingfold.rl.env import EmbeddingEnv


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    identities = [("train-a", "train-root"), ("train-b", "train-root"),
                  ("train-other", "other-root"), ("validation-a", "validation-root"),
                  ("validation-other", "validation-other-root"), ("test-a", "test-root")]
    public, evaluator = [], []
    for name, lineage in identities:
        public.append({"instance": name, "lineage_root": lineage,
                       "active_nodes": [0, 1, 2], "active_edges": [[0, 1], [1, 2]],
                       "h": {"a": 0., "b": 0.}, "J": [["a", "b", -1.]]})
        evaluator.append({"instance": name, "ground_energy": -1.,
                          "witness": {"a": [0], "b": [1]}})
    for name, records in (("instances.jsonl", public), ("evaluator_only.jsonl", evaluator)):
        (root / name).write_text("".join(json.dumps(r) + "\n" for r in records))
    # Deliberately mix instance names and lineage IDs; two train variants share a root.
    split = {"train": ["train-root", "train-other"],
             "validation": ["validation-a", "validation-other-root"], "test": ["test-root"]}
    (root / "manifest.json").write_text(json.dumps({"split": split}))
    return root


def args(root, **overrides):
    values = dict(corpus=str(root), train=2, heldout=1, seed=0, exploratory_repartition=False)
    return SimpleNamespace(**(values | overrides))


def test_manifest_split_is_default_and_only_train_wrappers_receive_witnesses(corpus):
    for seed in range(5):
        train, heldout, metadata = clone.clone_corpus_sets(args(corpus, seed=seed), [])
        assert {t.lineage for t in train} == {"train-root", "other-root"}
        assert all(t.lineage.startswith("validation") for t in heldout)
        assert all(t.prefix_source for t in train)
        assert all(t.prefix_source is None for t in heldout)
        for task in heldout:
            with pytest.raises(AssertionError, match="witness accessed"):
                task.witness
        assert metadata["manifest_split"] and metadata["heldout_role"] == "validation"
        assert metadata["manifest_lineages"]["test"] == ["test-root"]
        assert metadata["corpus_files_sha256"]["instances.jsonl"] == hashlib.sha256(
            (corpus / "instances.jsonl").read_bytes()).hexdigest()


def test_explicit_split_file_takes_precedence_and_is_hashed(corpus):
    split = {"train": ["train-root"], "validation": ["validation-root"], "test": ["test-a"]}
    (corpus / "splits.json").write_text(json.dumps(split))
    train, heldout, metadata = clone.clone_corpus_sets(args(corpus, train=1), [])
    assert train[0].lineage == "train-root" and heldout[0].lineage == "validation-root"
    assert metadata["split_source"] == "splits.json"
    assert "splits.json" in metadata["corpus_files_sha256"]


def test_missing_manifest_split_needs_explicit_exploratory_mode(corpus):
    (corpus / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="requires manifest train/validation"):
        clone.clone_corpus_sets(args(corpus), [])
    train, heldout, metadata = clone.clone_corpus_sets(args(corpus, exploratory_repartition=True), [])
    assert len(train) == 2 and len(heldout) == 1
    assert not metadata["manifest_split"] and metadata["heldout_role"] == "diagnostic"


def test_instance_and_root_overlap_is_rejected_before_cell_filtering(corpus):
    (corpus / "splits.json").write_text(json.dumps({
        "train": ["train-root"], "validation": ["validation-a", "train-b"], "test": []}))
    with pytest.raises(ValueError, match="overlap at the lineage"):
        clone.clone_corpus_sets(args(corpus, train=1), ["validation", "train-other"])


def cli(root, out, *extra):
    return ["--corpus", str(root), "--train", "2", "--heldout", "1", "--epochs", "1",
            "--eval-episodes", "1", "--max-steps", "20", "--out", str(out), *extra]


def stub_rollouts(monkeypatch):
    teachers, evaluations = [], []

    def teach(task, witness, features, *unused, **options):
        assert task.prefix_source and witness
        assert options["teacher_mode"] == "monotone"
        teachers.append(task.lineage)
        rows = np.zeros((2, clone.cc.FEATURE_WIDTHS["tiny"]), dtype=np.float32)
        rows[1, 0] = 1.
        return [(rows, [0])], "valid", True, 1., 0.

    def episode(task, *unused, **options):
        assert task.prefix_source is None
        with pytest.raises(AssertionError):
            task.witness
        evaluations.append(task.lineage)
        return {"valid": True}

    monkeypatch.setattr(clone, "teacher_steps", teach)
    monkeypatch.setattr(rollout, "episode", episode)
    return teachers, evaluations


def test_cli_trains_only_manifest_train_and_persists_contract(corpus, tmp_path, monkeypatch, capsys):
    teachers, evaluations = stub_rollouts(monkeypatch)
    out = tmp_path / "clone.pt"
    assert clone.main(cli(corpus, out)) == 0
    assert set(teachers) == {"train-root", "other-root"}
    assert evaluations and all(lineage.startswith("validation") for lineage in evaluations)
    saved = torch.load(out, weights_only=False)
    assert saved["training_provenance"]["complete"] is True
    assert set(saved["training_provenance"]["training_lineages"]) == set(teachers)
    assert saved["contract"]["options"]["teacher_mode"] == "monotone"
    assert saved["contract"]["source_files_sha256"]["constructor_clone.py"]
    assert saved["contract"]["train_fingerprints"] and saved["contract"]["heldout_fingerprints"]
    assert not saved["diagnostic"]
    header = json.loads(capsys.readouterr().out.splitlines()[0])
    assert header["heldout_role"] == "validation" and header["contract"] == saved["contract"]


def test_unknown_initialization_stays_diagnostic_in_logs_and_saved_checkpoint(
        corpus, tmp_path, monkeypatch, capsys):
    stub_rollouts(monkeypatch)
    initial, out = tmp_path / "legacy.pt", tmp_path / "clone.pt"
    actor = clone.cc.make_actor("linear", 32, clone.cc.FEATURE_WIDTHS["tiny"])
    torch.save({"state": actor.state_dict(), "actor": "linear", "features": "tiny"}, initial)
    clone.main(cli(corpus, out, "--init", str(initial)))
    saved = torch.load(out, weights_only=False)
    assert saved["diagnostic"] and not saved["training_provenance"]["complete"]
    assert "unknown" in saved["diagnostic_reasons"][0]
    assert saved["training_provenance"]["initial_checkpoint_sha256"] == hashlib.sha256(
        initial.read_bytes()).hexdigest()
    rows = [json.loads(row) for row in capsys.readouterr().out.splitlines() if row.startswith("{")]
    assert all(row["diagnostic"] for row in rows if "evaluation" in row)


@pytest.mark.parametrize("lineage", ["validation-other-root", "test-root"])
def test_initialization_cannot_have_trained_on_any_reserved_manifest_lineage(
        corpus, tmp_path, monkeypatch, lineage):
    initial = tmp_path / "contaminated.pt"
    torch.save({"training_lineages": [lineage]}, initial)
    with pytest.raises(ValueError, match="initial checkpoint training lineages overlap"):
        clone.main(cli(corpus, tmp_path / "out.pt", "--init", str(initial)))


def small_task():
    return clone.cc.Task("teacher", nx.path_graph(2), nx.path_graph(5), "train-teacher",
                         ground_energy=-1.)


def step_matching(env, decision, predicate):
    pick = next(i for i, candidate in enumerate(decision.candidates)
                if decision.legal_mask[i] and predicate(candidate))
    return env.step(decision, pick, evaluate_training_reward=False).next_decision_or_terminal


def test_monotone_labels_exclude_real_grow_shrink_cycle_and_stop_at_commit():
    task = small_task()
    witness = {0: frozenset({0, 1}), 1: frozenset({2})}
    context = clone._context(task, len(task.host), 20, 8, None, 2, wide=True)
    env = EmbeddingEnv(task, context, mode=Mode.CONSTRUCTION, initializer=None,
                       build_observation=False)
    env.generator.allow_satisfied_growth = True
    decision = env.reset(0)
    decision = step_matching(env, decision, lambda c: c.opcode is Opcode.PLACE
                             and c.new_chains == {0: frozenset({0})})
    grow = next(i for i in clone.teacher_candidates(decision, witness, env.state.chains)
                if decision.candidates[i].new_chains == {0: frozenset({0, 1})})
    decision = env.step(decision, grow, evaluate_training_reward=False).next_decision_or_terminal
    shrink = next(i for i, c in enumerate(decision.candidates) if decision.legal_mask[i]
                  and c.new_chains == {0: frozenset({0})})
    assert shrink in clone.teacher_candidates(decision, witness, env.state.chains, "witness_set")
    assert shrink not in clone.teacher_candidates(decision, witness, env.state.chains)
    only_shrink = SimpleNamespace(candidates=[decision.candidates[shrink]], legal_mask=[True])
    assert clone.teacher_candidates(only_shrink, witness, env.state.chains) == []
    decision = step_matching(env, decision, lambda c: c.opcode is Opcode.PLACE
                             and c.new_chains == {1: frozenset({2})})
    assert isinstance(decision, DecisionState)
    good = clone.teacher_candidates(decision, witness, env.state.chains)
    assert good and all(decision.candidates[i].opcode is Opcode.COMMIT for i in good)
    assert any(decision.candidates[i].opcode is not Opcode.COMMIT for i in
               clone.teacher_candidates(decision, witness, env.state.chains, "witness_set"))


def test_teacher_starts_empty_and_only_offered_candidates_are_observed():
    task = small_task()
    witness = {0: frozenset({0, 1}), 1: frozenset({2})}
    base = clone.cc.make_features("tiny", task)
    observations = []

    class Features:
        def observe(self, candidate, chains, **kwargs):
            observations.append(dict(chains))
            return base.observe(candidate, chains, **kwargs)

    with clone.cc.no_completion_solver():
        steps, reason, valid, _, _ = clone.teacher_steps(task, witness, Features(), 20, 10.)
    assert observations and not any(observations[0].values())
    assert steps and valid and reason == "valid"
    assert len(steps) <= sum(map(len, witness.values())) + 1
