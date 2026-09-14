"""Cross-fitted, train-only calibration checks for the warm action-Q head."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest  # noqa: E402
from isingfold.rl.checkpoint import runtime_implementation_registry  # noqa: E402
from isingfold.rl.ppo import WarmStartActionValueTarget  # noqa: E402
from isingfold.rl.q_calibration_audit import (  # noqa: E402
    CalibrationAuditExample,
    CalibrationFoldPredictor,
    audit_action_quality_calibration,
    assign_lineage_fold,
    load_q_calibration_audit_config,
    parse_q_calibration_audit_config,
    publish_q_calibration_audit,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "q_calibration_audit_v1.json"


class _FixedActionQ:
    action_quality_head = "bounded-qmu-action-head-v2"
    quality_prior_mode = "bounded-centered-v1"

    def __init__(self, probabilities: dict[str, tuple[float, ...]]) -> None:
        self.probabilities = probabilities
        self.training = True
        self.grad_enabled: list[bool] = []

    def eval(self):
        self.training = False
        return self

    def __call__(self, observations, *, device=None):
        del device
        self.grad_enabled.append(torch.is_grad_enabled())
        rows = [self.probabilities[str(observation)] for observation in observations]
        width = max(len(row) for row in rows)
        logits = torch.full((len(rows), width), float("nan"), dtype=torch.float32)
        values = torch.full_like(logits, float("nan"))
        counts = []
        for row_index, probabilities in enumerate(rows):
            probability = torch.tensor(probabilities, dtype=torch.float32)
            logits[row_index, : len(probabilities)] = torch.logit(probability)
            values[row_index, : len(probabilities)] = probability
            counts.append(len(probabilities))
        return SimpleNamespace(
            action_quality_logit=logits,
            action_quality_value=values,
            action_count=torch.tensor(counts, dtype=torch.long),
        )


def _signed(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "record_digest": content_digest(payload)}


def _config(*, folds: int = 2, minimum_lineages_per_fold: int = 1):
    payload = {
        "schema": "isingfold.action-quality-calibration-audit-config",
        "schema_version": 1,
        "protocol_id": "if-warm-action-q-crossfit-calibration-v1",
        "partition": "train",
        "folds": folds,
        "minimum_lineages_per_fold": minimum_lineages_per_fold,
        "fold_assignment": {
            "domain": "isingfold-action-q-audit-fold-v1",
            "salt": "unit-test-sealed-before-fit",
            "hash": "sha256-canonical-json-first-eight-bytes-big-endian-mod-k",
        },
        "inference_batch_size": 8,
        "ece": {
            "edges": [0.0, 0.5, 1.0],
            "boundary": "left-closed-right-open-last-bin-closed",
        },
        "action_weighting": "equal-lineage-equal-state-count-over-propensity-within-state",
        "bootstrap": {
            "replicates": 1000,
            "seed": 26091331,
            "cluster_unit": "immutable-base-lineage",
            "confidence_level": 0.95,
        },
        "scope": {
            "diagnostic_only": True,
            "model_selection_authorized": False,
            "retraining_authorized": False,
            "deployment_change_authorized": False,
            "validation_or_test_access_authorized": False,
        },
    }
    return parse_q_calibration_audit_config(_signed(payload), file_sha256="f" * 64)


def _example(
    lineage: str,
    observation: str,
    q_mu: tuple[float, float],
) -> CalibrationAuditExample:
    suffix = observation.rsplit("-", 1)[-1]
    return CalibrationAuditExample(
        observation=observation,
        target=WarmStartActionValueTarget(
            action_indices=(0, 1),
            q_mu=q_mu,
            continuation_counts=(16, 16),
            inclusion_probabilities=(1.0, 1.0),
            legal_action_count=2,
            support_size=2,
        ),
        base_lineage=lineage,
        task_id=f"task-{suffix}",
        instance_id=f"instance-{suffix}",
        state_fingerprint=f"state-{suffix}",
        source_quality_record_digest=hashlib.sha256(observation.encode()).hexdigest(),
    )


def _fold_identity(index: int, fit_lineages: tuple[str, ...]) -> dict[str, object]:
    payload = {
        "schema": "isingfold.action-quality-calibration-fold-checkpoint",
        "schema_version": 1,
        "partition": "train",
        "fold_index": index,
        "training_lineages": list(fit_lineages),
        "training_lineages_digest": content_digest(list(fit_lineages)),
        "checkpoint_sha256": f"{index + 1:x}" * 64,
        "checkpoint_payload_digest": f"{index + 3:x}" * 64,
        "run_record_digest": f"{index + 5:x}" * 64,
        "complete": True,
        "post_fit_frozen": True,
    }
    return _signed(payload)


def _predictors(config, examples, predictions):
    all_lineages = {example.base_lineage for example in examples}
    result = []
    for index in range(config.folds):
        audit = {
            lineage
            for lineage in all_lineages
            if assign_lineage_fold(lineage, config) == index
        }
        fit = tuple(sorted(all_lineages - audit))
        result.append(
            CalibrationFoldPredictor(
                fold_index=index,
                model=_FixedActionQ(predictions),
                fit_lineages=fit,
                checkpoint_identity=_fold_identity(index, fit),
            )
        )
    return tuple(result)


def _audit_inputs():
    config = _config()
    examples = (
        _example("lineage-a", "obs-a", (0.0, 1.0)),
        _example("lineage-b", "obs-b", (0.2, 0.8)),
        _example("lineage-c", "obs-c", (0.4, 0.6)),
        _example("lineage-d", "obs-d", (0.9, 0.1)),
    )
    predictions = {
        "obs-a": (0.1, 0.9),
        "obs-b": (0.8, 0.2),
        "obs-c": (0.4, 0.6),
        "obs-d": (0.7, 0.3),
    }
    return config, examples, _predictors(config, examples, predictions)


def _provenance(config, predictors) -> dict[str, object]:
    authority = _signed({"role": "training_partition"})
    target_access = _signed({"partition": "train"})
    ground = _signed({"partition": "train"})
    runtime = runtime_implementation_registry()
    return {
        "partition": "train",
        "sealed_validation_or_test_opened": False,
        "source_corpus_manifest_sha256": "1" * 64,
        "source_quality_manifest_sha256": "2" * 64,
        "source_quality_manifest_record_digest": "3" * 64,
        "source_quality_records_sha256": "4" * 64,
        "quality_preflight_receipt_sha256": "5" * 64,
        "quality_preflight_record_digest": "6" * 64,
        "quality_authority": authority,
        "target_access": target_access,
        "ground_partition_receipt": ground,
        "config_file_sha256": config.file_sha256,
        "config_record_digest": config.record_digest,
        "fold_checkpoints": [dict(fold.checkpoint_identity) for fold in predictors],
        "runtime_implementation_registry": runtime,
        "runtime_implementation_digest": content_digest(runtime),
        "implementation_source_sha256": hashlib.sha256(
            (ROOT / "src" / "isingfold" / "rl" / "q_calibration_audit.py").read_bytes()
        ).hexdigest(),
    }


def test_checked_in_calibration_config_is_sealed_and_train_only() -> None:
    config = load_q_calibration_audit_config(CONFIG)

    assert config.partition == "train"
    assert config.folds == 5
    assert len(config.ece_edges) == 11
    assert config.ece_edges[0] == 0.0
    assert config.ece_edges[-1] == 1.0
    assert config.scope == {
        "diagnostic_only": True,
        "model_selection_authorized": False,
        "retraining_authorized": False,
        "deployment_change_authorized": False,
        "validation_or_test_access_authorized": False,
    }


def test_lineage_fold_assignment_is_deterministic_and_never_splits_a_lineage() -> None:
    config = _config(folds=3)

    first = [assign_lineage_fold("same-base-lineage", config) for _ in range(20)]
    another_config = _config(folds=3)

    assert len(set(first)) == 1
    assert first[0] == assign_lineage_fold("same-base-lineage", another_config)
    assert 0 <= first[0] < 3


def test_cross_fitted_audit_reports_brier_ece_logit_and_ordering_metrics() -> None:
    config, examples, predictors = _audit_inputs()

    rows, aggregate = audit_action_quality_calibration(
        examples,
        predictors,
        config,
        device=torch.device("cpu"),
    )

    assert len(rows) == 4
    assert aggregate["census"] == {
        "folds": 2,
        "base_lineages": 4,
        "states": 4,
        "evaluated_actions": 8,
        "pairwise_comparisons": 4,
        "top_action_states": 4,
    }
    assert aggregate["metrics"]["brier_score"] == pytest.approx(0.1025)
    assert aggregate["metrics"]["fixed_bin_ece"] == pytest.approx(0.075)
    assert aggregate["metrics"]["pairwise_sign_accuracy"] == pytest.approx(0.75)
    assert aggregate["metrics"]["sampled_top_action_regret"] == pytest.approx(0.15)
    assert aggregate["metrics"]["sampled_top_action_hit_rate"] == pytest.approx(0.75)
    assert len(aggregate["ece_bins"]) == 2
    assert sum(row["weight"] for row in aggregate["ece_bins"]) == pytest.approx(1.0)
    logits = aggregate["logit_distribution"]
    assert logits["actions"] == 8
    assert logits["minimum"] == pytest.approx(torch.logit(torch.tensor(0.1)).item())
    assert logits["maximum"] == pytest.approx(torch.logit(torch.tensor(0.9)).item())
    assert logits["rms"] > 0.0
    assert all(model.model.training is False for model in predictors)
    assert all(model.model.grad_enabled == [False] for model in predictors)
    assert aggregate["bootstrap"]["brier_score"]["supported"] is True
    assert aggregate["bootstrap"]["fixed_bin_ece"]["supported"] is True


def test_audit_rejects_training_overlap_or_incomplete_crossfit_complements() -> None:
    config, examples, predictors = _audit_inputs()
    first = predictors[0]
    audit_lineage = next(
        example.base_lineage
        for example in examples
        if assign_lineage_fold(example.base_lineage, config) == first.fold_index
    )
    overlapping_fit = tuple(sorted({*first.fit_lineages, audit_lineage}))
    overlapping = (
        CalibrationFoldPredictor(
            fold_index=first.fold_index,
            model=first.model,
            fit_lineages=overlapping_fit,
            checkpoint_identity=_fold_identity(first.fold_index, overlapping_fit),
        ),
        *predictors[1:],
    )
    with pytest.raises(ValueError, match="disjoint|complement"):
        audit_action_quality_calibration(examples, overlapping, config)

    missing_fit = first.fit_lineages[:-1]
    incomplete = (
        CalibrationFoldPredictor(
            fold_index=first.fold_index,
            model=first.model,
            fit_lineages=missing_fit,
            checkpoint_identity=_fold_identity(first.fold_index, missing_fit),
        ),
        *predictors[1:],
    )
    with pytest.raises(ValueError, match="complement"):
        audit_action_quality_calibration(examples, incomplete, config)


def test_config_rejects_digest_drift_self_rehash_and_non_train_scope(tmp_path: Path) -> None:
    raw = json.loads(CONFIG.read_text())
    raw["ece"]["edges"][1] = 0.05
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="record digest"):
        load_q_calibration_audit_config(changed)

    raw["record_digest"] = content_digest(
        {key: value for key, value in raw.items() if key != "record_digest"}
    )
    self_rehashed = tmp_path / "self-rehashed.json"
    self_rehashed.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="pinned v1"):
        load_q_calibration_audit_config(self_rehashed)

    train_violation = json.loads(CONFIG.read_text())
    train_violation["partition"] = "validation"
    train_violation["record_digest"] = content_digest(
        {key: value for key, value in train_violation.items() if key != "record_digest"}
    )
    with pytest.raises(ValueError, match="train-only"):
        parse_q_calibration_audit_config(train_violation, file_sha256="a" * 64)


def test_publication_is_atomic_authenticated_and_cannot_authorize_selection(
    tmp_path: Path,
) -> None:
    config, examples, predictors = _audit_inputs()
    rows, aggregate = audit_action_quality_calibration(examples, predictors, config)
    destination = tmp_path / "q-calibration"

    receipt = publish_q_calibration_audit(
        destination,
        rows=rows,
        aggregate=aggregate,
        config=config,
        provenance=_provenance(config, predictors),
    )

    assert receipt["scope"] == "DIAGNOSTIC_ONLY"
    assert receipt["partition"] == "train"
    assert receipt["sealed_validation_or_test_opened"] is False
    assert receipt["model_selection_authorized"] is False
    assert receipt["retraining_authorized"] is False
    assert receipt["deployment_change_authorized"] is False
    assert {path.name for path in destination.iterdir()} == {
        "predictions.jsonl",
        "receipt.json",
    }
    assert (destination / "predictions.jsonl").read_bytes() == b"".join(
        canonical_json_bytes(dict(row)) + b"\n" for row in rows
    )
    assert json.loads((destination / "receipt.json").read_text()) == receipt

    forged = _provenance(config, predictors)
    forged["partition"] = "validation"
    with pytest.raises(ValueError, match="train-only"):
        publish_q_calibration_audit(
            tmp_path / "forged",
            rows=rows,
            aggregate=aggregate,
            config=config,
            provenance=forged,
        )

    tampered = deepcopy(rows)
    tampered[0]["state_brier_score"] += 0.1
    with pytest.raises(ValueError, match="record digest"):
        publish_q_calibration_audit(
            tmp_path / "tampered",
            rows=tampered,
            aggregate=aggregate,
            config=config,
            provenance=_provenance(config, predictors),
        )
