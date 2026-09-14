"""Cross-fitted train-only calibration audit for the warm-start action-Q head.

The audit is deliberately downstream of fitting.  Every prediction for an immutable
base lineage must come from a checkpoint whose declared training-lineage census is the
exact complement of that audit fold.  Results cannot select, recalibrate, or retrain a
model and cannot authorize validation or test access.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.checkpoint import runtime_implementation_registry
from isingfold.rl.model import ACTION_QUALITY_HEAD, QUALITY_POLICY_PRIOR_MODES
from isingfold.rl.ppo import WarmStartActionValueTarget

CONFIG_SCHEMA = "isingfold.action-quality-calibration-audit-config"
CONFIG_SCHEMA_VERSION = 1
CONFIG_PROTOCOL_ID = "if-warm-action-q-crossfit-calibration-v1"
CONFIG_V1_RECORD_DIGEST = "ed1cac8ff6318799beae6861177e04c599806c05a5c50eb64204d2b756f60802"
AUDIT_SCHEMA = "isingfold.action-quality-calibration-audit"
AUDIT_ROW_SCHEMA = "isingfold.action-quality-calibration-audit-row"
AUDIT_SCHEMA_VERSION = 1
FOLD_CHECKPOINT_SCHEMA = "isingfold.action-quality-calibration-fold-checkpoint"
MINIMUM_BOOTSTRAP_REPLICATES = 1_000
_HEX = frozenset("0123456789abcdef")
_METRICS = (
    "brier_score",
    "fixed_bin_ece",
    "pairwise_sign_accuracy",
    "sampled_top_action_regret",
    "sampled_top_action_hit_rate",
)
_SCOPE = {
    "diagnostic_only": True,
    "model_selection_authorized": False,
    "retraining_authorized": False,
    "deployment_change_authorized": False,
    "validation_or_test_access_authorized": False,
}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _strict_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    def reject_duplicates(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Q-calibration config contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"Q-calibration config contains non-finite number {token}")

    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except OSError as error:
        raise ValueError(f"cannot read Q-calibration config: {error}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Q-calibration config is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("Q-calibration config must be an object")
    return raw, value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{label} fields differ: missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


@dataclass(frozen=True, slots=True)
class QCalibrationAuditConfig:
    """Authenticated, outcome-independent audit configuration."""

    file_sha256: str
    record_digest: str
    protocol_id: str
    partition: str
    folds: int
    minimum_lineages_per_fold: int
    fold_domain: str
    fold_salt: str
    fold_hash: str
    inference_batch_size: int
    ece_edges: tuple[float, ...]
    ece_boundary: str
    action_weighting: str
    bootstrap_replicates: int
    bootstrap_seed: int
    bootstrap_cluster_unit: str
    bootstrap_confidence_level: float
    scope: dict[str, bool]

    def contract(self) -> dict[str, object]:
        return {
            "file_sha256": self.file_sha256,
            "record_digest": self.record_digest,
            "protocol_id": self.protocol_id,
            "partition": self.partition,
            "folds": self.folds,
            "minimum_lineages_per_fold": self.minimum_lineages_per_fold,
            "fold_assignment": {
                "domain": self.fold_domain,
                "salt": self.fold_salt,
                "hash": self.fold_hash,
            },
            "inference_batch_size": self.inference_batch_size,
            "ece": {
                "edges": list(self.ece_edges),
                "boundary": self.ece_boundary,
            },
            "action_weighting": self.action_weighting,
            "bootstrap": {
                "replicates": self.bootstrap_replicates,
                "seed": self.bootstrap_seed,
                "cluster_unit": self.bootstrap_cluster_unit,
                "confidence_level": self.bootstrap_confidence_level,
            },
            "scope": dict(self.scope),
        }


@dataclass(frozen=True, slots=True)
class CalibrationAuditExample:
    """One authenticated train-state action-Q target and policy-visible observation."""

    observation: object
    target: WarmStartActionValueTarget
    base_lineage: str
    task_id: str
    instance_id: str
    state_fingerprint: str
    source_quality_record_digest: str


@dataclass(frozen=True, slots=True)
class CalibrationFoldPredictor:
    """Post-fit fold model and its authenticated complement-lineage identity."""

    fold_index: int
    model: object
    fit_lineages: tuple[str, ...]
    checkpoint_identity: Mapping[str, object]


def parse_q_calibration_audit_config(
    record: Mapping[str, object], *, file_sha256: str
) -> QCalibrationAuditConfig:
    """Validate a signed config without imposing the checked-in v1 digest pin."""

    fields = {
        "schema",
        "schema_version",
        "protocol_id",
        "partition",
        "folds",
        "minimum_lineages_per_fold",
        "fold_assignment",
        "inference_batch_size",
        "ece",
        "action_weighting",
        "bootstrap",
        "scope",
        "record_digest",
    }
    _exact_keys(record, fields, "Q-calibration config")
    digest = record.get("record_digest")
    if not _is_sha256(file_sha256) or not _is_sha256(digest):
        raise ValueError("Q-calibration config requires lowercase SHA-256 identities")
    unsigned = {key: value for key, value in record.items() if key != "record_digest"}
    if digest != content_digest(unsigned):
        raise ValueError("Q-calibration config record digest mismatch")
    if (
        record["schema"] != CONFIG_SCHEMA
        or record["schema_version"] != CONFIG_SCHEMA_VERSION
        or record["protocol_id"] != CONFIG_PROTOCOL_ID
    ):
        raise ValueError("Q-calibration config has an unsupported identity")
    if record["partition"] != "train":
        raise ValueError("Q-calibration audit is train-only")
    folds = _positive_integer(record["folds"], "Q-calibration fold count")
    if folds < 2:
        raise ValueError("Q-calibration audit requires at least two folds")
    minimum = _positive_integer(
        record["minimum_lineages_per_fold"],
        "minimum audit lineages per fold",
    )
    assignment = record["fold_assignment"]
    if not isinstance(assignment, Mapping):
        raise ValueError("Q-calibration fold assignment must be an object")
    _exact_keys(assignment, {"domain", "salt", "hash"}, "fold assignment")
    if (
        assignment["domain"] != "isingfold-action-q-audit-fold-v1"
        or not isinstance(assignment["salt"], str)
        or not assignment["salt"]
        or assignment["hash"]
        != "sha256-canonical-json-first-eight-bytes-big-endian-mod-k"
    ):
        raise ValueError("Q-calibration fold assignment contract differs")
    ece = record["ece"]
    if not isinstance(ece, Mapping):
        raise ValueError("Q-calibration ECE contract must be an object")
    _exact_keys(ece, {"edges", "boundary"}, "Q-calibration ECE contract")
    raw_edges = ece["edges"]
    if not isinstance(raw_edges, list) or len(raw_edges) < 3:
        raise ValueError("Q-calibration ECE needs at least two fixed bins")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in raw_edges
    ):
        raise ValueError("Q-calibration ECE edges must be finite numbers")
    edges = tuple(float(value) for value in raw_edges)
    if edges[0] != 0.0 or edges[-1] != 1.0 or any(
        right <= left for left, right in zip(edges[:-1], edges[1:], strict=True)
    ):
        raise ValueError("Q-calibration ECE edges must increase exactly from zero to one")
    if ece["boundary"] != "left-closed-right-open-last-bin-closed":
        raise ValueError("Q-calibration ECE boundary rule differs")
    if record["action_weighting"] != (
        "equal-lineage-equal-state-count-over-propensity-within-state"
    ):
        raise ValueError("Q-calibration action weighting differs")
    bootstrap = record["bootstrap"]
    if not isinstance(bootstrap, Mapping):
        raise ValueError("Q-calibration bootstrap contract must be an object")
    _exact_keys(
        bootstrap,
        {"replicates", "seed", "cluster_unit", "confidence_level"},
        "Q-calibration bootstrap",
    )
    replicates = _positive_integer(bootstrap["replicates"], "bootstrap replicates")
    if replicates < MINIMUM_BOOTSTRAP_REPLICATES:
        raise ValueError("Q-calibration bootstrap has too few replicates")
    confidence = bootstrap["confidence_level"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 < float(confidence) < 1.0
    ):
        raise ValueError("Q-calibration confidence level must lie in (0, 1)")
    if bootstrap["cluster_unit"] != "immutable-base-lineage":
        raise ValueError("Q-calibration bootstrap must cluster by immutable base lineage")
    scope = record["scope"]
    if not isinstance(scope, Mapping) or dict(scope) != _SCOPE:
        raise ValueError("Q-calibration diagnostic-only scope differs")
    return QCalibrationAuditConfig(
        file_sha256=file_sha256,
        record_digest=str(digest),
        protocol_id=CONFIG_PROTOCOL_ID,
        partition="train",
        folds=folds,
        minimum_lineages_per_fold=minimum,
        fold_domain=str(assignment["domain"]),
        fold_salt=str(assignment["salt"]),
        fold_hash=str(assignment["hash"]),
        inference_batch_size=_positive_integer(
            record["inference_batch_size"], "Q-calibration inference batch size"
        ),
        ece_edges=edges,
        ece_boundary=str(ece["boundary"]),
        action_weighting=str(record["action_weighting"]),
        bootstrap_replicates=replicates,
        bootstrap_seed=_nonnegative_integer(bootstrap["seed"], "bootstrap seed"),
        bootstrap_cluster_unit=str(bootstrap["cluster_unit"]),
        bootstrap_confidence_level=float(confidence),
        scope=dict(_SCOPE),
    )


def load_q_calibration_audit_config(
    path: str | os.PathLike[str], *, expected_file_sha256: str | None = None
) -> QCalibrationAuditConfig:
    """Load the immutable checked-in v1 audit config and reject self-rehashed drift."""

    raw, record = _strict_json(Path(path))
    observed_sha = hashlib.sha256(raw).hexdigest()
    if expected_file_sha256 is not None:
        if not _is_sha256(expected_file_sha256) or observed_sha != expected_file_sha256:
            raise ValueError("Q-calibration config differs from its external SHA-256 pin")
    config = parse_q_calibration_audit_config(record, file_sha256=observed_sha)
    if config.record_digest != CONFIG_V1_RECORD_DIGEST:
        raise ValueError("Q-calibration config differs from the pinned v1 record digest")
    return config


def assign_lineage_fold(base_lineage: str, config: QCalibrationAuditConfig) -> int:
    """Assign every descendant of one immutable base lineage to one fixed fold."""

    if not isinstance(base_lineage, str) or not base_lineage:
        raise ValueError("Q-calibration base lineage must be nonempty text")
    if not isinstance(config, QCalibrationAuditConfig):
        raise TypeError("Q-calibration config has an unsupported type")
    payload = canonical_json_bytes(
        {
            "domain": config.fold_domain,
            "salt": config.fold_salt,
            "base_lineage": base_lineage,
        }
    )
    prefix = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(prefix, byteorder="big", signed=False) % config.folds


def _with_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**payload, "record_digest": content_digest(payload)}


def _finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _validate_signed_record(value: object, *, name: str) -> None:
    if not isinstance(value, Mapping) or "record_digest" not in value:
        raise ValueError(f"Q-calibration provenance has no authenticated {name}")
    digest = value["record_digest"]
    unsigned = {key: item for key, item in value.items() if key != "record_digest"}
    if not _is_sha256(digest) or digest != content_digest(unsigned):
        raise ValueError(f"Q-calibration {name} record digest mismatch")


def _validate_checkpoint_identity(
    identity: Mapping[str, object],
    *,
    fold_index: int,
    fit_lineages: tuple[str, ...],
) -> dict[str, object]:
    fields = {
        "schema",
        "schema_version",
        "partition",
        "fold_index",
        "training_lineages",
        "training_lineages_digest",
        "checkpoint_sha256",
        "checkpoint_payload_digest",
        "run_record_digest",
        "complete",
        "post_fit_frozen",
        "record_digest",
    }
    _exact_keys(identity, fields, "Q-calibration fold checkpoint")
    _validate_signed_record(identity, name="fold checkpoint")
    if (
        identity["schema"] != FOLD_CHECKPOINT_SCHEMA
        or identity["schema_version"] != AUDIT_SCHEMA_VERSION
        or identity["partition"] != "train"
        or identity["fold_index"] != fold_index
        or identity["complete"] is not True
        or identity["post_fit_frozen"] is not True
    ):
        raise ValueError("Q-calibration fold checkpoint identity differs")
    raw_lineages = identity["training_lineages"]
    if (
        not isinstance(raw_lineages, list)
        or any(not isinstance(lineage, str) or not lineage for lineage in raw_lineages)
        or raw_lineages != sorted(set(raw_lineages))
    ):
        raise ValueError("Q-calibration checkpoint training lineages are not canonical")
    if tuple(raw_lineages) != fit_lineages:
        raise ValueError("Q-calibration checkpoint differs from the exact fit complement")
    if identity["training_lineages_digest"] != content_digest(raw_lineages):
        raise ValueError("Q-calibration checkpoint training-lineage digest mismatch")
    for field in (
        "training_lineages_digest",
        "checkpoint_sha256",
        "checkpoint_payload_digest",
        "run_record_digest",
        "record_digest",
    ):
        if not _is_sha256(identity[field]):
            raise ValueError("Q-calibration fold checkpoint contains an invalid digest")
    return dict(identity)


def _validate_examples(
    examples: Sequence[CalibrationAuditExample],
    config: QCalibrationAuditConfig,
) -> tuple[tuple[CalibrationAuditExample, ...], dict[int, tuple[str, ...]]]:
    if not isinstance(config, QCalibrationAuditConfig):
        raise TypeError("Q-calibration audit requires an authenticated config")
    if not examples:
        raise ValueError("Q-calibration audit requires nonempty train examples")
    seen_sources: set[str] = set()
    seen_states: set[tuple[str, str, str, str]] = set()
    lineages: set[str] = set()
    checked: list[CalibrationAuditExample] = []
    for example in examples:
        if not isinstance(example, CalibrationAuditExample):
            raise TypeError("Q-calibration audit received an unsupported example")
        for field in ("base_lineage", "task_id", "instance_id", "state_fingerprint"):
            value = getattr(example, field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"Q-calibration example {field} must be nonempty text")
        if not isinstance(example.target, WarmStartActionValueTarget):
            raise ValueError("Q-calibration example has no authenticated action-Q target")
        if not _is_sha256(example.source_quality_record_digest):
            raise ValueError("Q-calibration example has an invalid source quality digest")
        if example.source_quality_record_digest in seen_sources:
            raise ValueError("Q-calibration examples repeat a source quality record")
        state_identity = (
            example.base_lineage,
            example.instance_id,
            example.task_id,
            example.state_fingerprint,
        )
        if state_identity in seen_states:
            raise ValueError("Q-calibration examples repeat a decision state")
        seen_sources.add(example.source_quality_record_digest)
        seen_states.add(state_identity)
        lineages.add(example.base_lineage)
        checked.append(example)
    fold_lineages = {
        index: tuple(
            sorted(
                lineage
                for lineage in lineages
                if assign_lineage_fold(lineage, config) == index
            )
        )
        for index in range(config.folds)
    }
    undersized = {
        index: len(values)
        for index, values in fold_lineages.items()
        if len(values) < config.minimum_lineages_per_fold
    }
    if undersized:
        raise ValueError(
            "Q-calibration audit folds do not meet the sealed minimum lineage census: "
            f"{undersized}"
        )
    checked.sort(
        key=lambda item: (
            item.base_lineage,
            item.instance_id,
            item.task_id,
            item.state_fingerprint,
            item.source_quality_record_digest,
        )
    )
    return tuple(checked), fold_lineages


def _validate_predictors(
    predictors: Sequence[CalibrationFoldPredictor],
    *,
    all_lineages: tuple[str, ...],
    fold_lineages: Mapping[int, tuple[str, ...]],
    config: QCalibrationAuditConfig,
) -> dict[int, tuple[CalibrationFoldPredictor, dict[str, object]]]:
    if len(predictors) != config.folds:
        raise ValueError("Q-calibration requires exactly one predictor per audit fold")
    by_fold: dict[int, tuple[CalibrationFoldPredictor, dict[str, object]]] = {}
    all_set = set(all_lineages)
    for predictor in predictors:
        if not isinstance(predictor, CalibrationFoldPredictor):
            raise TypeError("Q-calibration received an unsupported fold predictor")
        index = predictor.fold_index
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < config.folds
            or index in by_fold
        ):
            raise ValueError("Q-calibration predictor fold indices are incomplete or repeated")
        if (
            not isinstance(predictor.fit_lineages, tuple)
            or predictor.fit_lineages != tuple(sorted(set(predictor.fit_lineages)))
            or any(not isinstance(value, str) or not value for value in predictor.fit_lineages)
        ):
            raise ValueError("Q-calibration fit lineages must be a canonical tuple")
        audit_set = set(fold_lineages[index])
        fit_set = set(predictor.fit_lineages)
        if audit_set & fit_set:
            raise ValueError("Q-calibration fit and audit lineages are not disjoint")
        if fit_set != all_set - audit_set:
            raise ValueError("Q-calibration fit lineages are not the exact audit complement")
        if getattr(predictor.model, "action_quality_head", None) != ACTION_QUALITY_HEAD:
            raise ValueError("Q-calibration predictor omits the registered bounded action-Q head")
        if getattr(predictor.model, "quality_prior_mode", None) not in QUALITY_POLICY_PRIOR_MODES:
            raise ValueError("Q-calibration predictor has an unregistered quality-prior mode")
        if not isinstance(predictor.checkpoint_identity, Mapping):
            raise ValueError("Q-calibration predictor omits its checkpoint identity")
        identity = _validate_checkpoint_identity(
            predictor.checkpoint_identity,
            fold_index=index,
            fit_lineages=predictor.fit_lineages,
        )
        by_fold[index] = (predictor, identity)
    if set(by_fold) != set(range(config.folds)):
        raise ValueError("Q-calibration predictor fold coverage is incomplete")
    return by_fold


def _pairwise_action_counts(
    predictions: Sequence[float], targets: Sequence[float]
) -> dict[str, int]:
    result = {
        "correct": 0,
        "incorrect": 0,
        "prediction_ties": 0,
        "target_ties": 0,
        "comparable": 0,
    }
    for left, right in combinations(range(len(predictions)), 2):
        prediction_delta = predictions[left] - predictions[right]
        target_delta = targets[left] - targets[right]
        if prediction_delta == 0.0:
            result["prediction_ties"] += 1
        if target_delta == 0.0:
            result["target_ties"] += 1
        if prediction_delta == 0.0 or target_delta == 0.0:
            continue
        result["comparable"] += 1
        if (prediction_delta > 0.0) == (target_delta > 0.0):
            result["correct"] += 1
        else:
            result["incorrect"] += 1
    return result


def _prediction_row(
    example: CalibrationAuditExample,
    *,
    fold_index: int,
    checkpoint_record_digest: str,
    logits: Sequence[float],
    probabilities: Sequence[float],
) -> dict[str, Any]:
    target = example.target
    raw_weights = tuple(float(value) for value in target.regression_weights)
    denominator = math.fsum(raw_weights)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("Q-calibration within-state action weights are invalid")
    within_weights = tuple(value / denominator for value in raw_weights)
    q_values = tuple(float(value) for value in target.q_mu)
    actions = [
        {
            "action_index": int(action_index),
            "target_q_mu": q_value,
            "continuation_count": int(count),
            "inclusion_probability": float(inclusion),
            "regression_weight": raw_weight,
            "within_state_weight": within_weight,
            "predicted_logit": float(logit),
            "predicted_probability": float(probability),
        }
        for action_index, q_value, count, inclusion, raw_weight, within_weight, logit, probability in zip(
            target.action_indices,
            q_values,
            target.continuation_counts,
            target.inclusion_probabilities,
            raw_weights,
            within_weights,
            logits,
            probabilities,
            strict=True,
        )
    ]
    brier = math.fsum(
        weight * (probability - q_value) ** 2
        for weight, probability, q_value in zip(
            within_weights, probabilities, q_values, strict=True
        )
    )
    pairwise = _pairwise_action_counts(probabilities, q_values)
    top_position = min(
        range(len(actions)),
        key=lambda position: (-probabilities[position], target.action_indices[position]),
    )
    oracle_value = max(q_values)
    empirical_best = tuple(
        int(target.action_indices[position])
        for position, value in enumerate(q_values)
        if value == oracle_value
    )
    selected_index = int(target.action_indices[top_position])
    regret = oracle_value - q_values[top_position]
    if not -1e-12 <= regret <= 1.0 + 1e-12:
        raise RuntimeError("Q-calibration sampled top-action regret left [0, 1]")
    payload = {
        "schema": AUDIT_ROW_SCHEMA,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "protocol_id": CONFIG_PROTOCOL_ID,
        "partition": "train",
        "config_record_digest": None,  # Bound by the caller before signing.
        "audit_fold": fold_index,
        "base_lineage": example.base_lineage,
        "task_id": example.task_id,
        "instance_id": example.instance_id,
        "state_fingerprint": example.state_fingerprint,
        "source_quality_record_digest": example.source_quality_record_digest,
        "checkpoint_record_digest": checkpoint_record_digest,
        "legal_action_count": target.legal_action_count,
        "support_size": target.support_size,
        "evaluated_action_count": len(actions),
        "actions": actions,
        "state_brier_score": brier,
        "pairwise_counts": pairwise,
        "pairwise_sign_accuracy": (
            pairwise["correct"] / pairwise["comparable"]
            if pairwise["comparable"]
            else None
        ),
        "sampled_top_action_index": selected_index,
        "sampled_empirical_best_indices": list(empirical_best),
        "sampled_top_action_regret": max(0.0, min(1.0, regret)),
        "sampled_top_action_hit": float(selected_index in empirical_best),
    }
    return payload


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise RuntimeError("Q-calibration distribution received invalid values")
    quantiles = np.quantile(array, (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99))
    return {
        "actions": len(values),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "mean_absolute": float(np.mean(np.abs(array))),
        "quantiles": {
            name: float(value)
            for name, value in zip(
                ("p01", "p05", "p25", "p50", "p75", "p95", "p99"),
                quantiles,
                strict=True,
            )
        },
        "absolute_ge_2_rate": float(np.mean(np.abs(array) >= 2.0)),
        "absolute_ge_4_rate": float(np.mean(np.abs(array) >= 4.0)),
        "absolute_ge_8_rate": float(np.mean(np.abs(array) >= 8.0)),
    }


def _lineage_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["base_lineage"]), []).append(row)
    return {key: tuple(value) for key, value in sorted(grouped.items())}


def _lineage_metric(
    state_rows: Sequence[Mapping[str, Any]], metric: str
) -> float | None:
    values = [float(row[metric]) for row in state_rows if row[metric] is not None]
    return float(np.mean(values)) if values else None


def _ece_sufficient_statistics(
    state_rows: Sequence[Mapping[str, Any]], edges: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bins = len(edges) - 1
    masses = np.zeros(bins, dtype=np.float64)
    predictions = np.zeros(bins, dtype=np.float64)
    targets = np.zeros(bins, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.int64)
    state_weight = 1.0 / len(state_rows)
    for row in state_rows:
        for action in row["actions"]:
            probability = float(action["predicted_probability"])
            index = int(np.searchsorted(edges, probability, side="right") - 1)
            index = min(max(index, 0), bins - 1)
            weight = state_weight * float(action["within_state_weight"])
            masses[index] += weight
            predictions[index] += weight * probability
            targets[index] += weight * float(action["target_q_mu"])
            counts[index] += 1
    return masses, predictions, targets, counts


def _ece_from_sufficient(
    masses: np.ndarray,
    prediction_sums: np.ndarray,
    target_sums: np.ndarray,
    counts: np.ndarray,
    edges: tuple[float, ...],
) -> tuple[float, list[dict[str, Any]]]:
    total = float(masses.sum())
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-10):
        raise RuntimeError("Q-calibration ECE weights do not sum to one")
    result: list[dict[str, Any]] = []
    ece = 0.0
    for index, mass in enumerate(masses):
        weight = float(mass)
        if weight > 0.0:
            predicted = float(prediction_sums[index] / mass)
            target = float(target_sums[index] / mass)
            gap = abs(predicted - target)
            ece += weight * gap
        else:
            predicted = target = gap = None
        result.append(
            {
                "index": index,
                "lower": edges[index],
                "upper": edges[index + 1],
                "upper_inclusive": index == len(edges) - 2,
                "actions": int(counts[index]),
                "weight": weight,
                "mean_prediction": predicted,
                "mean_target_q_mu": target,
                "absolute_calibration_gap": gap,
            }
        )
    return float(ece), result


def _point_metrics(
    rows: Sequence[Mapping[str, Any]], config: QCalibrationAuditConfig
) -> tuple[dict[str, float | None], dict[str, dict[str, int]], list[dict[str, Any]]]:
    grouped = _lineage_rows(rows)
    per_lineage: dict[str, dict[str, float]] = {lineage: {} for lineage in grouped}
    support: dict[str, dict[str, int]] = {}
    row_metrics = {
        "brier_score": "state_brier_score",
        "pairwise_sign_accuracy": "pairwise_sign_accuracy",
        "sampled_top_action_regret": "sampled_top_action_regret",
        "sampled_top_action_hit_rate": "sampled_top_action_hit",
    }
    metrics: dict[str, float | None] = {}
    for output_name, row_name in row_metrics.items():
        for lineage, state_rows in grouped.items():
            value = _lineage_metric(state_rows, row_name)
            if value is not None:
                per_lineage[lineage][output_name] = value
        values = [items[output_name] for items in per_lineage.values() if output_name in items]
        metrics[output_name] = float(np.mean(values)) if values else None
        support[output_name] = {
            "states": sum(row[row_name] is not None for row in rows),
            "lineages": len(values),
        }
    lineage_stats = [
        _ece_sufficient_statistics(state_rows, config.ece_edges)
        for state_rows in grouped.values()
    ]
    masses = np.mean(np.stack([item[0] for item in lineage_stats]), axis=0)
    prediction_sums = np.mean(np.stack([item[1] for item in lineage_stats]), axis=0)
    target_sums = np.mean(np.stack([item[2] for item in lineage_stats]), axis=0)
    counts = np.sum(np.stack([item[3] for item in lineage_stats]), axis=0)
    ece, bins = _ece_from_sufficient(
        masses, prediction_sums, target_sums, counts, config.ece_edges
    )
    metrics["fixed_bin_ece"] = ece
    support["fixed_bin_ece"] = {
        "states": len(rows),
        "lineages": len(grouped),
    }
    ordered_metrics = {name: metrics[name] for name in _METRICS}
    ordered_support = {name: support[name] for name in _METRICS}
    return ordered_metrics, ordered_support, bins


def _bootstrap_metrics(
    rows: Sequence[Mapping[str, Any]], config: QCalibrationAuditConfig
) -> dict[str, dict[str, Any]]:
    grouped = _lineage_rows(rows)
    lineages = tuple(grouped)
    n_lineages = len(lineages)
    row_metrics = {
        "brier_score": "state_brier_score",
        "pairwise_sign_accuracy": "pairwise_sign_accuracy",
        "sampled_top_action_regret": "sampled_top_action_regret",
        "sampled_top_action_hit_rate": "sampled_top_action_hit",
    }
    lineage_values: dict[str, np.ndarray] = {}
    supported: dict[str, bool] = {}
    eligible_counts: dict[str, int] = {}
    for output_name, row_name in row_metrics.items():
        values = [_lineage_metric(grouped[lineage], row_name) for lineage in lineages]
        eligible_counts[output_name] = sum(value is not None for value in values)
        supported[output_name] = all(value is not None for value in values)
        if supported[output_name]:
            lineage_values[output_name] = np.asarray(values, dtype=np.float64)
    ece_stats = [
        _ece_sufficient_statistics(grouped[lineage], config.ece_edges)
        for lineage in lineages
    ]
    ece_mass = np.stack([item[0] for item in ece_stats])
    ece_prediction = np.stack([item[1] for item in ece_stats])
    ece_target = np.stack([item[2] for item in ece_stats])
    distributions = {
        name: np.empty(config.bootstrap_replicates, dtype=np.float64)
        for name in _METRICS
        if name == "fixed_bin_ece" or supported.get(name, False)
    }
    rng = np.random.default_rng(config.bootstrap_seed)
    chunk_size = min(512, config.bootstrap_replicates)
    for start in range(0, config.bootstrap_replicates, chunk_size):
        stop = min(start + chunk_size, config.bootstrap_replicates)
        sample = rng.integers(0, n_lineages, size=(stop - start, n_lineages))
        for metric, values in lineage_values.items():
            distributions[metric][start:stop] = values[sample].mean(axis=1)
        mass = ece_mass[sample].mean(axis=1)
        prediction = ece_prediction[sample].mean(axis=1)
        target = ece_target[sample].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            gaps = np.where(mass > 0.0, np.abs(prediction / mass - target / mass), 0.0)
        distributions["fixed_bin_ece"][start:stop] = np.sum(mass * gaps, axis=1)
    alpha = (1.0 - config.bootstrap_confidence_level) / 2.0
    result: dict[str, dict[str, Any]] = {}
    for metric in _METRICS:
        if metric not in distributions:
            result[metric] = {
                "supported": False,
                "reason": "metric is not supported in every independent base lineage",
                "lineages": eligible_counts.get(metric, 0),
                "required_lineages": n_lineages,
            }
            continue
        distribution = distributions[metric]
        lower, upper = np.quantile(distribution, (alpha, 1.0 - alpha))
        result[metric] = {
            "supported": True,
            "estimate": float(distribution.mean()) if metric != "fixed_bin_ece" else None,
            "lower": float(lower),
            "upper": float(upper),
            "confidence_level": config.bootstrap_confidence_level,
            "replicates": config.bootstrap_replicates,
            "cluster_unit": config.bootstrap_cluster_unit,
            "lineages": n_lineages,
        }
    point, _, _ = _point_metrics(rows, config)
    for metric, entry in result.items():
        if entry["supported"]:
            entry["estimate"] = point[metric]
    return result


def _build_aggregate(
    rows: Sequence[Mapping[str, Any]],
    config: QCalibrationAuditConfig,
    checkpoint_identities: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    metrics, support, ece_bins = _point_metrics(rows, config)
    fold_summaries: list[dict[str, Any]] = []
    all_lineages = tuple(sorted({str(row["base_lineage"]) for row in rows}))
    identities = {int(value["fold_index"]): value for value in checkpoint_identities}
    for index in range(config.folds):
        fold_rows = tuple(row for row in rows if row["audit_fold"] == index)
        audit_lineages = tuple(sorted({str(row["base_lineage"]) for row in fold_rows}))
        fit_lineages = tuple(sorted(set(all_lineages) - set(audit_lineages)))
        identity = identities[index]
        fold_metrics, _, _ = _point_metrics(fold_rows, config)
        fold_summaries.append(
            {
                "fold_index": index,
                "audit_lineages": list(audit_lineages),
                "audit_lineages_digest": content_digest(list(audit_lineages)),
                "fit_lineages": list(fit_lineages),
                "fit_lineages_digest": content_digest(list(fit_lineages)),
                "checkpoint_record_digest": identity["record_digest"],
                "census": {
                    "base_lineages": len(audit_lineages),
                    "states": len(fold_rows),
                    "evaluated_actions": sum(int(row["evaluated_action_count"]) for row in fold_rows),
                },
                "metrics": fold_metrics,
            }
        )
    logits = [float(action["predicted_logit"]) for row in rows for action in row["actions"]]
    probabilities = [
        float(action["predicted_probability"]) for row in rows for action in row["actions"]
    ]
    pairwise_comparisons = sum(int(row["pairwise_counts"]["comparable"]) for row in rows)
    payload = {
        "schema": AUDIT_SCHEMA,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "protocol_id": config.protocol_id,
        "partition": "train",
        "config_record_digest": config.record_digest,
        "census": {
            "folds": config.folds,
            "base_lineages": len(all_lineages),
            "states": len(rows),
            "evaluated_actions": len(logits),
            "pairwise_comparisons": pairwise_comparisons,
            "top_action_states": sum(row["sampled_top_action_regret"] is not None for row in rows),
        },
        "weighting_contract": config.action_weighting,
        "metrics": metrics,
        "metric_support": support,
        "ece_bins": ece_bins,
        "logit_distribution": _distribution(logits),
        "probability_distribution": _distribution(probabilities),
        "bootstrap": _bootstrap_metrics(rows, config),
        "folds": fold_summaries,
        "rows_record_digest": content_digest([row["record_digest"] for row in rows]),
    }
    return _with_digest(payload)


def audit_action_quality_calibration(
    examples: Sequence[CalibrationAuditExample],
    predictors: Sequence[CalibrationFoldPredictor],
    config: QCalibrationAuditConfig,
    *,
    device: torch.device | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Audit frozen cross-fit action-Q predictions without opening held-out partitions."""

    checked, fold_lineages = _validate_examples(examples, config)
    all_lineages = tuple(sorted({item.base_lineage for item in checked}))
    by_fold = _validate_predictors(
        predictors,
        all_lineages=all_lineages,
        fold_lineages=fold_lineages,
        config=config,
    )
    inference_device = device or torch.device("cpu")
    rows: list[dict[str, Any]] = []
    identities: list[dict[str, object]] = []
    for index in range(config.folds):
        predictor, identity = by_fold[index]
        identities.append(identity)
        fold_examples = tuple(
            example
            for example in checked
            if assign_lineage_fold(example.base_lineage, config) == index
        )
        evaluated_model = predictor.model.eval()  # type: ignore[attr-defined]
        if evaluated_model is not predictor.model:
            raise ValueError("Q-calibration model eval() unexpectedly replaced the model")
        for start in range(0, len(fold_examples), config.inference_batch_size):
            batch = fold_examples[start : start + config.inference_batch_size]
            observations = [example.observation for example in batch]
            with torch.inference_mode():
                output = predictor.model(observations, device=inference_device)  # type: ignore[operator]
            logits_tensor = getattr(output, "action_quality_logit", None)
            values_tensor = getattr(output, "action_quality_value", None)
            counts_tensor = getattr(output, "action_count", None)
            if not all(
                isinstance(value, torch.Tensor)
                for value in (logits_tensor, values_tensor, counts_tensor)
            ):
                raise ValueError("Q-calibration model omitted tensor action-Q outputs")
            if (
                logits_tensor.ndim != 2
                or values_tensor.shape != logits_tensor.shape
                or counts_tensor.ndim != 1
                or counts_tensor.shape[0] != len(batch)
                or logits_tensor.shape[0] != len(batch)
            ):
                raise ValueError("Q-calibration model action-Q output shapes differ")
            logits_cpu = logits_tensor.detach().to(dtype=torch.float64, device="cpu")
            values_cpu = values_tensor.detach().to(dtype=torch.float64, device="cpu")
            counts_cpu = counts_tensor.detach().to(dtype=torch.long, device="cpu")
            for row_index, example in enumerate(batch):
                count = int(counts_cpu[row_index].item())
                if count != example.target.support_size or not 0 < count <= logits_cpu.shape[1]:
                    raise ValueError("Q-calibration model action support differs from the target")
                indices = tuple(int(value) for value in example.target.action_indices)
                selected_logits = tuple(float(logits_cpu[row_index, value]) for value in indices)
                selected_values = tuple(float(values_cpu[row_index, value]) for value in indices)
                if any(not math.isfinite(value) for value in (*selected_logits, *selected_values)):
                    raise ValueError("Q-calibration model produced non-finite evaluated predictions")
                expected_values = tuple(1.0 / (1.0 + math.exp(-value)) for value in selected_logits)
                if any(
                    not 0.0 <= value <= 1.0
                    or not math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-7)
                    for value, expected in zip(selected_values, expected_values, strict=True)
                ):
                    raise ValueError("Q-calibration probability output differs from sigmoid(logit)")
                payload = _prediction_row(
                    example,
                    fold_index=index,
                    checkpoint_record_digest=str(identity["record_digest"]),
                    logits=selected_logits,
                    probabilities=selected_values,
                )
                payload["config_record_digest"] = config.record_digest
                rows.append(_with_digest(payload))
    rows.sort(
        key=lambda row: (
            str(row["base_lineage"]),
            str(row["instance_id"]),
            str(row["task_id"]),
            str(row["state_fingerprint"]),
            str(row["source_quality_record_digest"]),
        )
    )
    aggregate = _build_aggregate(rows, config, identities)
    return tuple(rows), aggregate


def _validate_raw_rows(
    rows: Sequence[Mapping[str, Any]], config: QCalibrationAuditConfig
) -> None:
    if not rows:
        raise ValueError("Q-calibration audit cannot publish empty predictions")
    sources: set[str] = set()
    state_keys: set[tuple[str, str, str, str]] = set()
    sort_keys: list[tuple[str, str, str, str, str]] = []
    for row in rows:
        if (
            row.get("schema") != AUDIT_ROW_SCHEMA
            or row.get("schema_version") != AUDIT_SCHEMA_VERSION
            or row.get("protocol_id") != config.protocol_id
            or row.get("partition") != "train"
            or row.get("config_record_digest") != config.record_digest
        ):
            raise ValueError("Q-calibration raw row has an unsupported identity or partition")
        digest = row.get("record_digest")
        unsigned = {key: value for key, value in row.items() if key != "record_digest"}
        if not _is_sha256(digest) or digest != content_digest(unsigned):
            raise ValueError("Q-calibration raw row record digest mismatch")
        source = row.get("source_quality_record_digest")
        if not _is_sha256(source) or source in sources:
            raise ValueError("Q-calibration raw rows repeat or omit source quality identities")
        sources.add(str(source))
        state_key = tuple(
            str(row[field])
            for field in ("base_lineage", "instance_id", "task_id", "state_fingerprint")
        )
        if any(not value for value in state_key) or state_key in state_keys:
            raise ValueError("Q-calibration raw rows repeat or omit state identities")
        state_keys.add(state_key)
        sort_keys.append((*state_key, str(source)))
        actions = row.get("actions")
        if not isinstance(actions, list) or len(actions) != row.get("evaluated_action_count"):
            raise ValueError("Q-calibration raw row action census differs")
        weight_sum = math.fsum(float(action["within_state_weight"]) for action in actions)
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("Q-calibration raw row action weights do not sum to one")
    if sort_keys != sorted(sort_keys):
        raise ValueError("Q-calibration raw rows are not in canonical order")


def _validate_provenance(
    provenance: Mapping[str, Any],
    *,
    rows: Sequence[Mapping[str, Any]],
    config: QCalibrationAuditConfig,
) -> tuple[dict[str, object], ...]:
    required = {
        "partition",
        "sealed_validation_or_test_opened",
        "source_corpus_manifest_sha256",
        "source_quality_manifest_sha256",
        "source_quality_manifest_record_digest",
        "source_quality_records_sha256",
        "quality_preflight_receipt_sha256",
        "quality_preflight_record_digest",
        "quality_authority",
        "target_access",
        "ground_partition_receipt",
        "config_file_sha256",
        "config_record_digest",
        "fold_checkpoints",
        "runtime_implementation_registry",
        "runtime_implementation_digest",
        "implementation_source_sha256",
    }
    _exact_keys(provenance, required, "Q-calibration provenance")
    if provenance["partition"] != "train" or provenance["sealed_validation_or_test_opened"] is not False:
        raise ValueError("Q-calibration diagnostic is train-only and cannot open validation/test")
    digest_fields = {
        "source_corpus_manifest_sha256",
        "source_quality_manifest_sha256",
        "source_quality_manifest_record_digest",
        "source_quality_records_sha256",
        "quality_preflight_receipt_sha256",
        "quality_preflight_record_digest",
        "config_file_sha256",
        "config_record_digest",
        "runtime_implementation_digest",
        "implementation_source_sha256",
    }
    if any(not _is_sha256(provenance[field]) for field in digest_fields):
        raise ValueError("Q-calibration provenance contains an invalid artifact digest")
    if (
        provenance["config_file_sha256"] != config.file_sha256
        or provenance["config_record_digest"] != config.record_digest
    ):
        raise ValueError("Q-calibration provenance config identity mismatch")
    for field in ("quality_authority", "target_access", "ground_partition_receipt"):
        _validate_signed_record(provenance[field], name=field.replace("_", " "))
    for field in ("target_access", "ground_partition_receipt"):
        record = provenance[field]
        if not isinstance(record, Mapping) or record.get("partition") != "train":
            raise ValueError("Q-calibration provenance attempts non-train target access")
    runtime = provenance["runtime_implementation_registry"]
    if (
        not isinstance(runtime, Mapping)
        or dict(runtime) != runtime_implementation_registry()
        or provenance["runtime_implementation_digest"] != content_digest(runtime)
    ):
        raise ValueError("Q-calibration runtime implementation registry mismatch")
    module_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if provenance["implementation_source_sha256"] != module_sha:
        raise ValueError("Q-calibration implementation source hash mismatch")
    raw_checkpoints = provenance["fold_checkpoints"]
    if not isinstance(raw_checkpoints, list) or len(raw_checkpoints) != config.folds:
        raise ValueError("Q-calibration provenance fold checkpoint census differs")
    lineages = tuple(sorted({str(row["base_lineage"]) for row in rows}))
    identities: list[dict[str, object]] = []
    seen_folds: set[int] = set()
    for raw_identity in raw_checkpoints:
        if not isinstance(raw_identity, Mapping):
            raise ValueError("Q-calibration provenance fold checkpoint is malformed")
        index = raw_identity.get("fold_index")
        if isinstance(index, bool) or not isinstance(index, int) or index in seen_folds:
            raise ValueError("Q-calibration provenance fold checkpoint indices differ")
        if not 0 <= index < config.folds:
            raise ValueError("Q-calibration provenance fold checkpoint index is out of range")
        seen_folds.add(index)
        audit_lineages = {
            str(row["base_lineage"]) for row in rows if row["audit_fold"] == index
        }
        fit_lineages = tuple(sorted(set(lineages) - audit_lineages))
        identity = _validate_checkpoint_identity(
            raw_identity, fold_index=index, fit_lineages=fit_lineages
        )
        row_digests = {
            str(row["checkpoint_record_digest"])
            for row in rows
            if row["audit_fold"] == index
        }
        if row_digests != {str(identity["record_digest"])}:
            raise ValueError("Q-calibration rows differ from their fold checkpoint identity")
        identities.append(identity)
    if seen_folds != set(range(config.folds)):
        raise ValueError("Q-calibration provenance fold checkpoint coverage is incomplete")
    identities.sort(key=lambda item: int(item["fold_index"]))
    return tuple(identities)


def _write_sync(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def publish_q_calibration_audit(
    destination: str | os.PathLike[str],
    *,
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    config: QCalibrationAuditConfig,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish diagnostic predictions and a provenance-bound receipt."""

    target = Path(destination)
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Q-calibration output already exists: {target}")
    _validate_raw_rows(rows, config)
    identities = _validate_provenance(provenance, rows=rows, config=config)
    expected_aggregate = _build_aggregate(rows, config, identities)
    if dict(aggregate) != expected_aggregate:
        raise ValueError("Q-calibration aggregate differs from authenticated raw predictions")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.q-calibration-", dir=target.parent)
    )
    try:
        raw = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
        _write_sync(temporary / "predictions.jsonl", raw)
        receipt = _with_digest(
            {
                "schema": AUDIT_SCHEMA,
                "schema_version": AUDIT_SCHEMA_VERSION,
                "scope": "DIAGNOSTIC_ONLY",
                "partition": "train",
                "sealed_validation_or_test_opened": False,
                "model_selection_authorized": False,
                "retraining_authorized": False,
                "deployment_change_authorized": False,
                "calibration_target": {
                    "quantity": "bounded Q^mu(s,a)",
                    "interval": [0.0, 1.0],
                    "prediction_transform": "sigmoid(action_quality_logit)",
                    "evidence": "authenticated sampled continuation outcomes",
                },
                "weighting_contract": config.action_weighting,
                "cross_fit_contract": {
                    "folds": config.folds,
                    "cluster_unit": "immutable-base-lineage",
                    "fit_set": "exact complement of each audit fold",
                    "checkpoint_state": "complete-post-fit-frozen",
                },
                "limitations": [
                    "train-only calibration evidence cannot establish held-out superiority",
                    "sampled top-action regret is bounded but only covers evaluated actions",
                    "the audit cannot choose a model, recalibrate, retrain, or change deployment",
                ],
                "config": config.contract(),
                "raw_predictions": {
                    "path": "predictions.jsonl",
                    "schema": AUDIT_ROW_SCHEMA,
                    "count": len(rows),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "record_digest": content_digest([row["record_digest"] for row in rows]),
                },
                "aggregate": expected_aggregate,
                "provenance": dict(provenance),
            }
        )
        _write_sync(temporary / "receipt.json", canonical_json_bytes(receipt) + b"\n")
        os.replace(temporary, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        except (AttributeError, OSError):
            pass
        else:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return receipt


__all__ = [
    "AUDIT_ROW_SCHEMA",
    "AUDIT_SCHEMA",
    "AUDIT_SCHEMA_VERSION",
    "CONFIG_PROTOCOL_ID",
    "CalibrationAuditExample",
    "CalibrationFoldPredictor",
    "QCalibrationAuditConfig",
    "assign_lineage_fold",
    "audit_action_quality_calibration",
    "load_q_calibration_audit_config",
    "parse_q_calibration_audit_config",
    "publish_q_calibration_audit",
]
