from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from embedbench.models_chain import ChainEncoded, evaluate

ROOT = Path(__file__).parents[1]


class _FixedScoreModel:
    def __init__(self, scores: list[float]):
        import torch

        self._scores = torch.tensor(scores, dtype=torch.float32)

    def eval(self) -> None:
        pass

    def __call__(self, *_args):
        return self._scores


def _encoded() -> ChainEncoded:
    return ChainEncoded(
        x=np.zeros((3, 10), dtype=np.float32),
        adj=np.eye(3, dtype=np.float32),
        cand_masks=np.eye(3, dtype=np.float32),
        cand_feats=np.zeros((3, 10), dtype=np.float32),
        p=[0.50, 0.30, 0.90],
        stage=[2, 2, 1],
        best_index=0,
        resource_index=0,
        original_index=1,
        source="test",
        instance_id="instance-0",
        topology="chimera",
        difficulty="test",
    )


def _example():
    return SimpleNamespace(
        cand_masks=np.zeros((3, 1), dtype=np.float32),
        p=[0.50, 0.30, 0.90],
        stage=[2, 2, 1],
        resource_index=0,
        original_index=1,
        instance_id="instance-0",
        source="test",
        topology="chimera",
        audit_context=(
            ("quality_test.jsonl", "instance-0", 0),
            "a" * 64,
            _candidate_signature({"candidates": [[0], [1], [2]]}),
        ),
    )


def _quality_record(index: int, split: str) -> dict:
    return {
        "instance_id": f"problem-{split}-{index}-m",
        "focus": 0,
        "window_nodes": [0, 1],
        "window_edges": [[0, 1]],
        "frozen": {"2": [2]},
        "frozen_adjacency": {"0": [2], "1": [2]},
        "neighbours": [2],
        "edge_J": [-1.0],
        "neighbour_degree": [1],
        "neighbour_chain_size": [1],
        "candidates": [[0], [1], [0, 1]],
        "p_solve": [0.50, 0.30, 0.90],
        "stage": [2, 2, 1],
        "best_index": 2,
        "resource_index": 0,
        "original_index": 1,
        "source": "minorminer",
        "topology": "chimera",
        "size": 1,
        "difficulty": "test",
        "l_cap": 4,
        "focus_h": float(index + 1),
        "problem": {
            "h": {"0": float(index + 1), "2": 0.0},
            "J": [[0, 2, -1.0]],
            "e0": float(-index - 1),
        },
    }


def _write_fixed_quality_corpus(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    records = [
        _quality_record(index, split) for index, split in enumerate(("train", "val", "test"))
    ]
    corpus = tmp_path / "quality_chimera.jsonl"
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = tmp_path / "splits.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "embedbench.split-manifest",
                "schema_version": 2,
                "provenance": {
                    "inputs": [
                        {
                            "file": corpus.name,
                            "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                        }
                    ]
                },
                "splits": {
                    corpus.name: {
                        record["instance_id"]: split
                        for record, split in zip(
                            records,
                            ("train", "val", "test"),
                            strict=True,
                        )
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return corpus, manifest, records


def _candidate_signature(record: dict) -> str:
    payload = json.dumps(
        [sorted(int(qubit) for qubit in candidate) for candidate in record["candidates"]],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _credible_audit_row(corpus: Path, record: dict, scores: list[float]) -> dict:
    return {
        "audit_schema": "embedbench.independent-quality-audit",
        "audit_schema_version": 2,
        "file": f"/remote/release/{corpus.name}",
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "instance_id": record["instance_id"],
        "focus": record["focus"],
        "candidate_signature": _candidate_signature(record),
        "high_read_scores": scores,
        "reads": 4000,
        "provenance": {
            "generator": "EmbedBench/scripts/rescore_quality.py",
            "reads_per_strength": 4000,
            "num_sweeps": 200,
            "base_seed": 0,
            "registered_strength_count": 4,
            "seed_schedule": ("(base_seed + 7919*candidate_index + 31*strength_index) % 2**31"),
            "strength_schedule": "default_strength_grid(problem, 4)",
            "aggregation": "max_p_solve_over_strength_schedule",
        },
    }


def _load_rescore_quality_script():
    path = ROOT / "scripts" / "rescore_quality.py"
    spec = importlib.util.spec_from_file_location("full_support_rescore_quality", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def _verified_audit_scores(scores: list[float | None]):
    from embedbench.models_chain import (
        IndependentAuditEvidence,
        verify_independent_audit_scores,
    )

    record = {
        "_file": "quality_test.jsonl",
        "instance_id": "instance-0",
        "focus": 0,
        "candidates": [[0], [1], [2]],
    }
    provenance = {
        "generator": "EmbedBench/scripts/rescore_quality.py",
        "reads_per_strength": 4000,
        "num_sweeps": 200,
        "base_seed": 0,
        "registered_strength_count": 4,
        "seed_schedule": ("(base_seed + 7919*candidate_index + 31*strength_index) % 2**31"),
        "strength_schedule": "default_strength_grid(problem, 4)",
        "aggregation": "max_p_solve_over_strength_schedule",
    }
    evidence = IndependentAuditEvidence(
        key=("quality_test.jsonl", "instance-0", 0),
        scores=tuple(scores),
        audit_schema="embedbench.independent-quality-audit",
        audit_schema_version=2,
        corpus_sha256="a" * 64,
        candidate_signature=_candidate_signature(record),
        reads=4000,
        provenance=provenance,
        source="test-audit.jsonl",
        line_number=1,
    )
    return verify_independent_audit_scores(record, evidence, "a" * 64)


def test_evaluate_defaults_to_full_support_and_keeps_stage1_model_choice() -> None:
    result = evaluate(_FixedScoreModel([0.0, -1.0, 4.0]), [_encoded()])

    # Candidate 2 is a stage-1 label and the full-support quality winner.  The
    # legacy evaluator silently excludes it and therefore reports zero resource
    # regret instead of the true 0.40 gap.
    assert result["resource_regret"] == pytest.approx(0.40)
    assert result["evaluation_support"] == "full"
    assert result["support_uses_labels"] is False
    assert result["selections"][0]["model_index"] == 2
    assert result["model_regret"] == pytest.approx(0.0)
    assert result["candidate_support"]["total_supported_candidates"] == 3


def test_generic_full_support_evaluator_uses_identical_support_for_all_controls() -> None:
    from embedbench.models_chain import evaluate_quality_support

    result = evaluate_quality_support(
        lambda _example: [0.0, -1.0, 4.0],
        [_example()],
        random_seed=7,
    )

    selection = result["selections"][0]
    assert selection["support_indices"] == [0, 1, 2]
    assert selection["model_index"] in selection["support_indices"]
    assert selection["resource_index"] in selection["support_indices"]
    assert selection["original_index"] in selection["support_indices"]
    assert selection["random_index"] in selection["support_indices"]
    assert result["label_fidelity"] == "mixed-stage-release"
    assert result["provisional"] is True


def test_legacy_reliable_evaluator_remains_an_explicit_named_diagnostic() -> None:
    from embedbench.models_chain import evaluate_quality_support

    result = evaluate_quality_support(
        lambda _example: [0.0, -1.0, 4.0],
        [_example()],
        support="legacy-reliable",
    )

    assert result["evaluation_support"] == "legacy-reliable"
    assert result["support_uses_labels"] is True
    assert result["selections"][0]["support_indices"] == [0, 1]
    assert result["selections"][0]["model_index"] == 0


@pytest.mark.parametrize("mode", ["audit", "paper"])
def test_claim_modes_reject_label_derived_legacy_support(mode: str) -> None:
    from embedbench.models_chain import evaluate_quality_support

    with pytest.raises(ValueError, match="full candidate support"):
        evaluate_quality_support(
            lambda _example: [4.0, -1.0, 0.0],
            [_example()],
            support="legacy-reliable",
            mode=mode,
            audit_scores=[_verified_audit_scores([0.90, 0.20, None])],
        )


@pytest.mark.parametrize("mode", ["audit", "paper"])
def test_audit_and_paper_modes_fail_closed_without_complete_independent_labels(
    mode: str,
) -> None:
    from embedbench.models_chain import evaluate_quality_support

    with pytest.raises(ValueError, match="independent audit labels.*complete support"):
        evaluate_quality_support(
            lambda _example: [0.0, -1.0, 4.0],
            [_example()],
            mode=mode,
            audit_scores=[_verified_audit_scores([0.10, None, 0.95])],
        )


def test_paper_mode_uses_complete_independent_labels_and_reports_coverage() -> None:
    from embedbench.models_chain import evaluate_quality_support

    result = evaluate_quality_support(
        lambda _example: [0.0, -1.0, 4.0],
        [_example()],
        mode="paper",
        audit_scores=[_verified_audit_scores([0.10, 0.20, 0.95])],
    )

    assert result["label_fidelity"] == "independent-audit"
    assert result["provisional"] is False
    assert result["model_regret"] == pytest.approx(0.0)
    assert result["resource_regret"] == pytest.approx(0.85)
    assert result["audit_coverage"] == {
        "records_with_any_label": 1,
        "records_with_complete_support": 1,
        "record_fraction_complete": 1.0,
        "labeled_candidates": 3,
        "supported_candidates": 3,
        "candidate_fraction": 1.0,
    }
    assert result["selections"][0]["model_has_audit_label"] is True
    assert result["audit_provenance"]["all_complete_support_verified"] is True


@pytest.mark.parametrize("mode", ["audit", "paper"])
def test_claim_modes_reject_raw_unbound_score_vectors(mode: str) -> None:
    from embedbench.models_chain import evaluate_quality_support

    with pytest.raises(ValueError, match="verified high-read provenance"):
        evaluate_quality_support(
            lambda _example: [0.0, -1.0, 4.0],
            [_example()],
            mode=mode,
            audit_scores=[[0.10, 0.20, 0.95]],
        )


@pytest.mark.parametrize("replay", ["different-record", "reordered-candidates"])
def test_claim_mode_rejects_verified_wrapper_replayed_on_another_context(
    replay: str,
) -> None:
    from embedbench.models_chain import evaluate_quality_support

    verified = _verified_audit_scores([0.10, 0.20, 0.95])
    example = _example()
    if replay == "different-record":
        key = (verified.key[0], "instance-other", verified.key[2])
        signature = verified.candidate_signature
    else:
        key = verified.key
        signature = "c" * 64
    example.audit_context = (key, verified.corpus_sha256, signature)

    with pytest.raises(ValueError, match="audit context does not match"):
        evaluate_quality_support(
            lambda _example: [0.0, -1.0, 4.0],
            [example],
            mode="paper",
            audit_scores=[verified],
        )


@pytest.mark.parametrize("mode", ["audit", "paper"])
def test_claim_modes_reject_empty_evaluation_sets(mode: str) -> None:
    from embedbench.models_chain import evaluate_quality_support

    with pytest.raises(ValueError, match="at least one example"):
        evaluate_quality_support(lambda _example: [], [], mode=mode)


def test_development_mode_reports_partial_audit_coverage_without_using_it_as_truth() -> None:
    from embedbench.models_chain import evaluate_quality_support

    result = evaluate_quality_support(
        lambda _example: [0.0, -1.0, 4.0],
        [_example()],
        audit_scores=[[0.10, None, 0.95]],
    )

    assert result["model_regret"] == pytest.approx(0.0)
    assert result["audit_metrics"] is None
    assert result["audit_coverage"]["record_fraction_complete"] == 0.0
    assert result["audit_coverage"]["candidate_fraction"] == pytest.approx(2 / 3)
    assert result["selections"][0]["model_has_audit_label"] is True


def test_full_support_evaluator_rejects_partial_model_score_vectors() -> None:
    from embedbench.models_chain import evaluate_quality_support

    with pytest.raises(ValueError, match="one score per candidate"):
        evaluate_quality_support(lambda _example: [1.0, 0.0], [_example()])


def test_candidate_signature_binds_outer_order_but_not_chain_list_order() -> None:
    from embedbench.models_chain import quality_candidate_signature

    record = {"candidates": [[7, 2], [9], [8, 3]]}
    same_chains = {"candidates": [[2, 7], [9], [3, 8]]}
    reordered_candidates = {"candidates": [[9], [7, 2], [8, 3]]}

    assert quality_candidate_signature(record) == quality_candidate_signature(same_chains)
    assert quality_candidate_signature(record) != quality_candidate_signature(reordered_candidates)


def test_rescore_quality_builds_schema_v2_bound_audit_rows(tmp_path: Path) -> None:
    rescore = _load_rescore_quality_script()
    corpus = tmp_path / "quality_chimera.jsonl"
    corpus.write_text("release bytes\n", encoding="utf-8")
    record = {
        "instance_id": "instance-0",
        "focus": 7,
        "candidates": [[7, 2], [9]],
        "best_index": 0,
        "resource_index": 1,
        "original_index": -1,
        "p_solve": [0.4, 0.5],
    }

    row = rescore.build_audit_record(
        corpus,
        record,
        scores=[0.6, 0.7],
        reads=4000,
        base_seed=13,
    )

    assert row["file"] == corpus.name
    assert row["corpus_sha256"] == hashlib.sha256(corpus.read_bytes()).hexdigest()
    assert row["candidate_signature"] == _candidate_signature(record)
    assert row["provenance"] == {
        "generator": "EmbedBench/scripts/rescore_quality.py",
        "reads_per_strength": 4000,
        "num_sweeps": 200,
        "base_seed": 13,
        "registered_strength_count": 4,
        "seed_schedule": ("(base_seed + 7919*candidate_index + 31*strength_index) % 2**31"),
        "strength_schedule": "default_strength_grid(problem, 4)",
        "aggregation": "max_p_solve_over_strength_schedule",
    }


def test_rescore_quality_paper_row_binds_selection_and_realized_schedule(
    tmp_path: Path,
) -> None:
    rescore = _load_rescore_quality_script()
    corpus = tmp_path / "quality_chimera.jsonl"
    corpus.write_text("release bytes\n", encoding="utf-8")
    record = {
        "instance_id": "instance-0",
        "focus": 7,
        "candidates": [[7, 2], [9]],
        "best_index": 0,
        "resource_index": 1,
        "original_index": -1,
        "p_solve": [0.4, 0.5],
    }
    paper_binding = {
        "audit_contract": {"sha256": "a" * 64},
        "selection": {"sha256": "b" * 64},
        "audit_source": {
            "algorithm": "sha256_length_prefixed_relative_paths_and_bytes_v1",
            "sha256": "c" * 64,
        },
        "runtime_provenance": {"device": {"selected": "cpu"}},
    }
    ground_reference = {
        "status": "certified_exact",
        "method": "exhaustive_enumeration",
        "energy": -1.0,
        "n_variables": 2,
    }

    row = rescore.build_audit_record(
        corpus,
        record,
        scores=[0.6, 0.7],
        reads=4000,
        base_seed=0,
        paper_binding=paper_binding,
        realized_strengths=[0.25, 0.5, 1.0, 2.0],
        realized_seeds=[[0, 31, 62, 93], [7919, 7950, 7981, 8012]],
        strength_p_solve=[[0.6] * 4, [0.7] * 4],
        strength_success_counts=[[2400] * 4, [2800] * 4],
        ground_reference=ground_reference,
    )

    assert row["paper_binding"] == {
        **paper_binding,
        "ground_reference": ground_reference,
        "realized_strengths": [0.25, 0.5, 1.0, 2.0],
        "realized_seeds": [[0, 31, 62, 93], [7919, 7950, 7981, 8012]],
    }


def test_independent_audit_loader_keys_complete_candidate_vectors(tmp_path: Path) -> None:
    from embedbench.models_chain import load_independent_audit_scores

    labels = tmp_path / "audit.jsonl"
    labels.write_text(
        json.dumps(
            {
                "audit_schema": "embedbench.independent-quality-audit",
                "audit_schema_version": 2,
                "file": "/remote/release/quality_chimera.jsonl",
                "corpus_sha256": "a" * 64,
                "instance_id": "instance-0",
                "focus": 7,
                "candidate_signature": "b" * 64,
                "high_read_scores": [0.1, 0.2, 0.9],
                "reads": 4000,
                "provenance": {
                    "generator": "EmbedBench/scripts/rescore_quality.py",
                    "reads_per_strength": 4000,
                    "num_sweeps": 200,
                    "base_seed": 0,
                    "registered_strength_count": 4,
                    "seed_schedule": (
                        "(base_seed + 7919*candidate_index + 31*strength_index) % 2**31"
                    ),
                    "strength_schedule": "default_strength_grid(problem, 4)",
                    "aggregation": "max_p_solve_over_strength_schedule",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_independent_audit_scores([labels])

    evidence = loaded[("quality_chimera.jsonl", "instance-0", 7)]
    assert evidence.scores == (0.1, 0.2, 0.9)
    assert evidence.corpus_sha256 == "a" * 64
    assert evidence.candidate_signature == "b" * 64
    assert evidence.provenance["reads_per_strength"] == 4000


def test_independent_audit_loader_keeps_equal_local_ids_from_distinct_corpora(
    tmp_path: Path,
) -> None:
    from embedbench.models_chain import load_independent_audit_scores

    labels = tmp_path / "audit.jsonl"
    labels.write_text(
        "\n".join(
            json.dumps(
                {
                    "file": filename,
                    "instance_id": "instance-0",
                    "focus": 7,
                    "high_read_scores": scores,
                }
            )
            for filename, scores in (
                ("/apollo/quality_chimera.jsonl", [0.1, 0.9]),
                ("/goose/quality_pegasus.jsonl", [0.8, 0.2]),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_independent_audit_scores([labels])

    assert loaded[("quality_chimera.jsonl", "instance-0", 7)].scores == (0.1, 0.9)
    assert loaded[("quality_pegasus.jsonl", "instance-0", 7)].scores == (0.8, 0.2)


def test_independent_audit_loader_rejects_rows_without_corpus_identity(
    tmp_path: Path,
) -> None:
    from embedbench.models_chain import load_independent_audit_scores

    labels = tmp_path / "audit.jsonl"
    labels.write_text(
        json.dumps(
            {
                "instance_id": "instance-0",
                "focus": 7,
                "high_read_scores": [0.1, 0.9],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="corpus file"):
        load_independent_audit_scores([labels])


def test_independent_audit_loader_rejects_conflicting_duplicate_keys(
    tmp_path: Path,
) -> None:
    from embedbench.models_chain import load_independent_audit_scores

    labels = tmp_path / "audit.jsonl"
    labels.write_text(
        "\n".join(
            json.dumps(
                {
                    "file": "quality_chimera.jsonl",
                    "instance_id": "instance-0",
                    "focus": 7,
                    "high_read_scores": scores,
                }
            )
            for scores in ([0.1, 0.9], [0.9, 0.1])
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate independent audit labels"):
        load_independent_audit_scores([labels])


def test_train_chain_cli_exposes_full_support_and_explicit_legacy_mode() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/train_chain.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--evaluation-support {full,legacy-reliable}" in result.stdout
    assert "--evaluation-mode {development,audit,paper}" in result.stdout
    assert "--audit-labels" in result.stdout


def test_train_chain_paper_mode_selects_on_dev_validation_and_claims_on_test_audit(
    tmp_path: Path,
) -> None:
    corpus, manifest, records = _write_fixed_quality_corpus(tmp_path)
    labels = tmp_path / "audit.jsonl"
    labels.write_text(
        json.dumps(_credible_audit_row(corpus, records[2], [0.10, 0.20, 0.95])) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "paper.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_chain.py",
            str(corpus),
            "--splits",
            str(manifest),
            "--epochs",
            "1",
            "--hidden",
            "4",
            "--layers",
            "1",
            "--heads",
            "1",
            "--device",
            "cpu",
            "--evaluation-mode",
            "paper",
            "--evaluate-test",
            "--audit-labels",
            str(labels),
            "--out",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    validation = result["results"][0]["validation"]
    test = result["results"][0]["test"]
    assert validation["evaluation_support"] == "full"
    assert validation["evaluation_mode"] == "development"
    assert validation["label_fidelity"] == "mixed-stage-release"
    assert validation["provisional"] is True
    assert validation["candidate_support"]["total_supported_candidates"] == 3
    assert validation["audit_coverage"]["candidate_fraction"] == 0.0
    assert test["evaluation_mode"] == "paper"
    assert test["label_fidelity"] == "independent-audit"
    assert test["provisional"] is False
    assert test["audit_coverage"]["candidate_fraction"] == 1.0


def test_train_chain_paper_mode_fails_closed_on_wrong_corpus_identity(
    tmp_path: Path,
) -> None:
    corpus, manifest, records = _write_fixed_quality_corpus(tmp_path)
    labels = tmp_path / "audit.jsonl"
    labels.write_text(
        "".join(
            json.dumps(
                {
                    "file": "quality_other_topology.jsonl",
                    "instance_id": record["instance_id"],
                    "focus": record["focus"],
                    "high_read_scores": [0.10, 0.20, 0.95],
                }
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_chain.py",
            str(corpus),
            "--splits",
            str(manifest),
            "--epochs",
            "1",
            "--hidden",
            "4",
            "--layers",
            "1",
            "--heads",
            "1",
            "--device",
            "cpu",
            "--evaluation-mode",
            "paper",
            "--evaluate-test",
            "--audit-labels",
            str(labels),
            "--out",
            str(tmp_path / "must-not-exist.json"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "independent audit labels for the complete support" in result.stderr


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("corpus", "corpus SHA-256"),
        ("candidates", "candidate signature"),
        ("provenance", "seed_schedule"),
    ],
)
def test_train_chain_paper_mode_rejects_stale_or_untraceable_audit_labels(
    tmp_path: Path,
    tamper: str,
    message: str,
) -> None:
    corpus, manifest, records = _write_fixed_quality_corpus(tmp_path)
    row = _credible_audit_row(corpus, records[2], [0.10, 0.20, 0.95])
    if tamper == "corpus":
        row["corpus_sha256"] = "0" * 64
    elif tamper == "candidates":
        reordered = {**records[2], "candidates": list(reversed(records[2]["candidates"]))}
        row["candidate_signature"] = _candidate_signature(reordered)
    else:
        del row["provenance"]["seed_schedule"]
    labels = tmp_path / "audit.jsonl"
    labels.write_text(json.dumps(row) + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_chain.py",
            str(corpus),
            "--splits",
            str(manifest),
            "--epochs",
            "1",
            "--hidden",
            "4",
            "--layers",
            "1",
            "--heads",
            "1",
            "--device",
            "cpu",
            "--evaluation-mode",
            "paper",
            "--evaluate-test",
            "--audit-labels",
            str(labels),
            "--out",
            str(tmp_path / "must-not-exist.json"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert message in result.stderr


def test_train_chain_requires_explicit_locked_test_open_for_paper_mode(
    tmp_path: Path,
) -> None:
    corpus, manifest, _records = _write_fixed_quality_corpus(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_chain.py",
            str(corpus),
            "--splits",
            str(manifest),
            "--evaluation-mode",
            "paper",
            "--out",
            str(tmp_path / "must-not-exist.json"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "requires --evaluate-test with a fixed split" in result.stderr


@pytest.mark.parametrize("mode", ["audit", "paper"])
def test_train_chain_claim_modes_reject_legacy_reliable_support(
    tmp_path: Path,
    mode: str,
) -> None:
    corpus, manifest, _records = _write_fixed_quality_corpus(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_chain.py",
            str(corpus),
            "--splits",
            str(manifest),
            "--evaluation-mode",
            mode,
            "--evaluation-support",
            "legacy-reliable",
            "--evaluate-test",
            "--out",
            str(tmp_path / "must-not-exist.json"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "full candidate support" in result.stderr


@pytest.mark.parametrize("mode", ["audit", "paper"])
def test_train_chain_claim_modes_require_a_fixed_split(
    tmp_path: Path,
    mode: str,
) -> None:
    corpus, _manifest, _records = _write_fixed_quality_corpus(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/train_chain.py",
            str(corpus),
            "--evaluation-mode",
            mode,
            "--evaluate-test",
            "--out",
            str(tmp_path / "must-not-exist.json"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "requires --splits and --evaluate-test" in result.stderr
