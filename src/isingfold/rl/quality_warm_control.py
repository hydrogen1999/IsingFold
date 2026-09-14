"""Registered post-selection control for direct counterfactual-Q supervision.

The primary representation screen jointly trains ranking, state value, action value and
COMMIT-relative action differences.  This diagnostic keeps the selected implementation,
data, initialization seed and order schedule fixed while setting only the direct Q and
COMMIT-delta coefficients to zero.  It is deliberately outside the 9+18 selection grid.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from isingfold.rl.capacity_control import parameter_census
from isingfold.rl.data.import_embedbench import content_digest
from isingfold.rl.ppo import (
    WARM_START_ACTOR_CRITIC_LOSS,
    WARM_START_FULL_LOSS_PROFILE,
    WARM_START_RANK_VALUE_CONTROL_PROFILE,
    WARM_START_RANK_VALUE_CONTROL_PROFILE_ID,
    WARM_START_REDUCTION,
    WarmStartLossProfile,
    warm_start_loss_profile,
)

QUALITY_WARM_CONTROL_SCHEMA = "isingfold.quality-warm-control"
QUALITY_WARM_CONTROL_VERSION = 1
QUALITY_WARM_CONTROL_PROTOCOL_ID = "if-core-v2-hybrid-direct-qmu-warm-control-v1"
QUALITY_WARM_CONTROL_STAGE = "post-representation-selection-diagnostic"
QUALITY_WARM_CONTROL_SELECTION_EFFECT = "diagnostic-only-no-main-grid-reselection"
QUALITY_WARM_CONTROL_PROFILE_ID = WARM_START_RANK_VALUE_CONTROL_PROFILE_ID
QUALITY_WARM_CONTROL_MAIN_GRID_SHA256 = (
    "d3a7cd99c0c96c1c2f7fdd974d0856aee94ccbbe7a01f30b15d4724632712d09"
)
# Updated only by an explicit protocol revision; a self-rehashed file is not sufficient.
QUALITY_WARM_CONTROL_V1_RECORD_DIGEST = (
    "73f99e6f1b157d638d9348ec67a9cf0814dcb3518b3cdd63e56a38c5e6ad4fb0"
)
QUALITY_WARM_CONTROL_BINDING_SCHEMA = "isingfold.quality-warm-control-binding"
QUALITY_WARM_CONTROL_BINDING_VERSION = 1

_SEEDS = (1103, 2207, 3301)
_SOURCE_CELLS = (
    (1103, "rep-006-if-core-s1103"),
    (2207, "rep-007-if-core-s2207"),
    (3301, "rep-008-if-core-s3301"),
)
_MODEL_FAMILY = "if-core"
_QUALITY_PRIOR_MODE = "bounded-centered-v1"
_EVALUATION_PARTITION = "validation"
_AGGREGATION = "equal-training-seed-then-equal-immutable-base-lineage"
_PRIMARY_METRIC = "paired-unconditional-if-q3-s0-utility-difference"
_PERMITTED_CLAIM = (
    "post-selection diagnostic evidence for the incremental value of direct Q-mu and "
    "COMMIT-delta labels beyond resolved ranking and state-value supervision"
)
_PROHIBITED_CLAIMS = (
    "quality supervision versus structural supervision",
    "changing the nine-cell representation selection",
    "changing the eighteen-cell RL-value selection",
    "selecting a training seed or checkpoint from this diagnostic",
    "confirmatory evidence on the sealed test partition",
)
_REQUIRED_REPORTS = (
    "all three paired training seeds",
    "paired aggregate over immutable base lineages",
    "valid-return rate and unconditional utility",
    "direct action-Q calibration diagnostics",
    "negative, null and positive paired effects",
)
_PAIRING_FLAGS = (
    "same_parameter_schema",
    "same_initial_state_for_seed",
    "same_training_corpus",
    "same_quality_labels_and_preflight",
    "same_exact_corpus_denominators",
    "same_minibatch_partition",
    "same_training_data_order",
    "same_optimizer_and_stopping_rule",
    "same_runtime_implementation",
    "same_device_type",
    "reuse_authenticated_main_grid_treatment",
)
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class QualityWarmTrainingProtocol:
    loss: str
    reduction: str
    epochs: int
    minibatch_records: int
    learning_rate: float
    weight_decay: float
    minimum_resolved_rows: int
    minimum_resolved_lineages: int


@dataclass(frozen=True, slots=True)
class QualityWarmControlRegistry:
    path: Path
    file_sha256: str
    record_digest: str
    protocol_id: str
    main_grid_sha256: str
    selection_effect: str
    model_family: str
    quality_prior_mode: str
    model_signature_sha256: str
    trainable_parameters: int
    total_parameters: int
    treatment_profile: WarmStartLossProfile
    control_profile: WarmStartLossProfile
    training_seeds: tuple[int, ...]
    source_treatment_cells: tuple[tuple[int, str], ...]
    training_protocol: QualityWarmTrainingProtocol
    same_parameter_schema: bool
    same_initial_state_for_seed: bool
    same_training_data_order: bool
    evaluation_partition: str
    aggregation: str
    primary_metric: str
    seed_selection_forbidden: bool
    permitted_claim: str
    prohibited_claims: tuple[str, ...]
    required_reports: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QualityWarmControlCell:
    control_cell_id: str
    source_treatment_cell_id: str
    seed: int
    model_family: str
    model_signature_sha256: str
    profile_id: str


@dataclass(frozen=True, slots=True)
class QualityWarmControlPlan:
    registry_file_sha256: str
    registry_record_digest: str
    main_grid_sha256: str
    selected_simpler: str
    representation_selection_sha256: str
    representation_selection_record_digest: str
    corpus_manifest_sha256: str
    selector_digest: str
    normalizer_digest: str
    quality_manifest_sha256: str
    quality_preflight_sha256: str
    quality_preflight_record_digest: str
    cells: tuple[QualityWarmControlCell, ...]
    training_protocol: QualityWarmTrainingProtocol
    selection_effect: str
    evaluation_partition: str
    aggregation: str
    primary_metric: str
    seed_selection_forbidden: bool
    plan_digest: str


def _reject_duplicates(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"quality-warm-control JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise ValueError(f"quality-warm-control JSON contains non-finite number {token}")


def _check_finite(value: object) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("quality-warm-control JSON contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _check_finite(item)
    elif isinstance(value, list):
        for item in value:
            _check_finite(item)


def _strict_json(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except OSError as exc:
        raise ValueError(f"cannot read {label}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    _check_finite(value)
    return raw, value


def _exact_keys(value: Mapping[str, object], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ValueError(
            f"{label} schema differs: missing={sorted(keys - set(value))}, "
            f"unknown={sorted(set(value) - keys)}"
        )


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be nonempty text")
    return value


def _sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{label} must be finite and {qualifier}")
    return result


def _true(value: object, label: str) -> bool:
    if value is not True:
        raise ValueError(f"{label} must be true")
    return True


def _text_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a nonempty list")
    result = tuple(_text(item, label) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates")
    return result


def _verify_record(record: Mapping[str, object], label: str) -> str:
    digest = _sha256(record.get("record_digest"), f"{label} record digest")
    unsigned = {key: value for key, value in record.items() if key != "record_digest"}
    try:
        observed = content_digest(unsigned)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains non-canonical data") from exc
    if not hmac.compare_digest(digest, observed):
        raise ValueError(f"{label} record digest mismatch")
    return digest


def _load_main_grid(path: Path, registered_sha256: str) -> Mapping[str, object]:
    raw, grid = _strict_json(path, "main grid")
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(observed_sha256, registered_sha256):
        raise ValueError("main grid differs from the pinned main-grid digest")
    if (
        grid.get("schema") != "isingfold.staged-grid"
        or grid.get("schema_version") != 2
        or grid.get("name") != "if-core-v2-profile-i-hybrid-chimera-registered"
    ):
        raise ValueError("quality warm control requires the registered staged grid")
    stages = grid.get("stages")
    if not isinstance(stages, Mapping):
        raise ValueError("main grid has no stages")
    representation = stages.get("representation")
    rl_value = stages.get("rl_value")
    if not isinstance(representation, list) or len(representation) != 9:
        raise ValueError("main grid must contain nine representation cells")
    if not isinstance(rl_value, list) or len(rl_value) != 18:
        raise ValueError("main grid must contain eighteen RL-value cells")
    return grid


def _load_profile(raw: object, *, role: str) -> WarmStartLossProfile:
    if not isinstance(raw, Mapping):
        raise ValueError("quality warm-control profile must be an object")
    _exact_keys(
        raw,
        {
            "profile_id",
            "production_transfer_eligible",
            "q_mu_label_supervision",
            "role",
            "weights",
        },
        "quality warm-control profile",
    )
    if raw["role"] != role:
        raise ValueError("quality warm-control profile role differs from v1")
    profile = warm_start_loss_profile(_text(raw["profile_id"], "loss profile ID"))
    expected = {**profile.contract(), "role": role}
    if dict(raw) != expected:
        raise ValueError("quality warm-control profile differs from its registered contract")
    return profile


def _load_training_protocol(
    raw: object,
    *,
    grid: Mapping[str, object],
) -> QualityWarmTrainingProtocol:
    fields = {
        "epochs",
        "learning_rate",
        "loss",
        "minibatch_records",
        "minimum_resolved_lineages",
        "minimum_resolved_rows",
        "reduction",
        "weight_decay",
    }
    if not isinstance(raw, Mapping):
        raise ValueError("quality warm-control training protocol must be an object")
    _exact_keys(raw, fields, "quality warm-control training protocol")
    protocol = QualityWarmTrainingProtocol(
        loss=_text(raw["loss"], "quality warm-control loss"),
        reduction=_text(raw["reduction"], "quality warm-control reduction"),
        epochs=_integer(raw["epochs"], "quality warm-control epochs", minimum=1),
        minibatch_records=_integer(
            raw["minibatch_records"], "quality warm-control minibatch", minimum=1
        ),
        learning_rate=_number(
            raw["learning_rate"], "quality warm-control learning rate", positive=True
        ),
        weight_decay=_number(raw["weight_decay"], "quality warm-control weight decay"),
        minimum_resolved_rows=_integer(
            raw["minimum_resolved_rows"], "minimum resolved rows", minimum=1
        ),
        minimum_resolved_lineages=_integer(
            raw["minimum_resolved_lineages"], "minimum resolved lineages", minimum=1
        ),
    )
    if protocol.loss != WARM_START_ACTOR_CRITIC_LOSS or protocol.reduction != WARM_START_REDUCTION:
        raise ValueError("quality warm-control objective differs from the live warm objective")
    if (
        protocol.epochs,
        protocol.minibatch_records,
        protocol.learning_rate,
        protocol.weight_decay,
        protocol.minimum_resolved_rows,
        protocol.minimum_resolved_lineages,
    ) != (200, 32, 3e-4, 0.0, 128, 128):
        raise ValueError("quality warm-control training budget differs from v1")
    stages = grid["stages"]
    assert isinstance(stages, Mapping)
    representation = stages["representation"]
    assert isinstance(representation, list)
    by_cell = {
        row.get("cell_id"): row for row in representation if isinstance(row, Mapping)
    }
    for seed, cell_id in _SOURCE_CELLS:
        row = by_cell.get(cell_id)
        expected = {
            "cell_id": cell_id,
            "model_family": _MODEL_FAMILY,
            "seed": seed,
            "epochs": protocol.epochs,
            "minibatch": protocol.minibatch_records,
            "learning_rate": protocol.learning_rate,
            "weight_decay": protocol.weight_decay,
        }
        if row != expected:
            raise ValueError("source treatment differs from the registered warm protocol")
    resolution = grid.get("quality_resolution")
    if not isinstance(resolution, Mapping) or dict(resolution) != {
        "min_resolved_rows": protocol.minimum_resolved_rows,
        "min_resolved_lineages": protocol.minimum_resolved_lineages,
    }:
        raise ValueError("quality-resolution thresholds differ from the main grid")
    return protocol


def load_quality_warm_control_registry(
    path: str | Path,
    *,
    main_grid_path: str | Path,
) -> QualityWarmControlRegistry:
    """Authenticate the immutable diagnostic registry against grid and live model code."""

    registry_path = Path(path)
    raw, record = _strict_json(registry_path, "quality-warm-control registry")
    _exact_keys(
        record,
        {
            "claim_scope",
            "evaluation",
            "main_grid",
            "mode",
            "model",
            "pairing",
            "profiles",
            "protocol_id",
            "record_digest",
            "schema",
            "schema_version",
            "stage",
            "training_protocol",
            "training_seeds",
        },
        "quality-warm-control registry",
    )
    if (
        record["schema"] != QUALITY_WARM_CONTROL_SCHEMA
        or record["schema_version"] != QUALITY_WARM_CONTROL_VERSION
        or record["protocol_id"] != QUALITY_WARM_CONTROL_PROTOCOL_ID
        or record["stage"] != QUALITY_WARM_CONTROL_STAGE
        or record["mode"] != "improvement"
    ):
        raise ValueError("quality-warm-control registry has an incompatible identity")
    digest = _verify_record(record, "quality-warm-control registry")
    if not hmac.compare_digest(digest, QUALITY_WARM_CONTROL_V1_RECORD_DIGEST):
        raise ValueError("quality-warm-control registry differs from the pinned v1 record digest")

    main = record["main_grid"]
    if not isinstance(main, Mapping):
        raise ValueError("quality warm-control main-grid binding must be an object")
    _exact_keys(
        main,
        {"representation_source_cells", "selection_effect", "sha256"},
        "quality warm-control main-grid binding",
    )
    main_sha = _sha256(main["sha256"], "pinned main-grid digest")
    if main_sha != QUALITY_WARM_CONTROL_MAIN_GRID_SHA256:
        raise ValueError("quality warm-control names another main-grid digest")
    grid = _load_main_grid(Path(main_grid_path), main_sha)
    source_cells = main["representation_source_cells"]
    if not isinstance(source_cells, list):
        raise ValueError("representation source cells must be a list")
    parsed_source_cells: list[tuple[int, str]] = []
    for row in source_cells:
        if not isinstance(row, Mapping):
            raise ValueError("representation source cell must be an object")
        _exact_keys(row, {"cell_id", "seed"}, "representation source cell")
        parsed_source_cells.append(
            (
                _integer(row["seed"], "source treatment seed"),
                _text(row["cell_id"], "source treatment cell ID"),
            )
        )
    if tuple(parsed_source_cells) != _SOURCE_CELLS:
        raise ValueError("quality warm-control source-treatment census differs from v1")
    selection_effect = _text(main["selection_effect"], "selection effect")
    if selection_effect != QUALITY_WARM_CONTROL_SELECTION_EFFECT:
        raise ValueError("quality warm control is not isolated from main selection")

    model = record["model"]
    if not isinstance(model, Mapping):
        raise ValueError("quality warm-control model contract must be an object")
    _exact_keys(
        model,
        {
            "expected_parameter_signature_sha256",
            "expected_total_parameters",
            "expected_trainable_parameters",
            "model_family",
            "quality_prior_mode",
        },
        "quality warm-control model contract",
    )
    if model["model_family"] != _MODEL_FAMILY or model["quality_prior_mode"] != _QUALITY_PRIOR_MODE:
        raise ValueError("quality warm-control model or policy coupling differs from v1")
    observed = {row.model_id: row for row in parameter_census(improvement_mode=True)}[
        _MODEL_FAMILY
    ]
    registered_model = (
        _sha256(model["expected_parameter_signature_sha256"], "model signature"),
        _integer(model["expected_trainable_parameters"], "trainable parameters", minimum=1),
        _integer(model["expected_total_parameters"], "total parameters", minimum=1),
    )
    observed_model = (
        observed.parameter_signature_sha256,
        observed.trainable_parameters,
        observed.total_parameters,
    )
    if registered_model != observed_model:
        raise ValueError("quality warm-control live IF-Core parameter schema drifted")

    profiles = record["profiles"]
    if not isinstance(profiles, list) or len(profiles) != 2:
        raise ValueError("quality warm control requires exactly two loss profiles")
    treatment = _load_profile(profiles[0], role="authenticated-main-grid-treatment")
    control = _load_profile(profiles[1], role="direct-qmu-label-control")
    if treatment != WARM_START_FULL_LOSS_PROFILE or control != WARM_START_RANK_VALUE_CONTROL_PROFILE:
        raise ValueError("quality warm-control treatment/control identities differ from v1")

    seeds = record["training_seeds"]
    if not isinstance(seeds, list):
        raise ValueError("quality warm-control seeds must be a list")
    parsed_seeds = tuple(_integer(seed, "quality warm-control seed") for seed in seeds)
    if parsed_seeds != _SEEDS:
        raise ValueError("quality warm control must retain all three registered seeds")
    training = _load_training_protocol(record["training_protocol"], grid=grid)

    pairing = record["pairing"]
    if not isinstance(pairing, Mapping):
        raise ValueError("quality warm-control pairing contract must be an object")
    _exact_keys(pairing, set(_PAIRING_FLAGS), "quality warm-control pairing contract")
    for field in _PAIRING_FLAGS:
        _true(pairing[field], f"quality warm-control pairing field {field}")

    evaluation = record["evaluation"]
    if not isinstance(evaluation, Mapping):
        raise ValueError("quality warm-control evaluation contract must be an object")
    _exact_keys(
        evaluation,
        {"aggregation", "partition", "primary_metric", "seed_selection_forbidden"},
        "quality warm-control evaluation contract",
    )
    if (
        evaluation["partition"] != _EVALUATION_PARTITION
        or evaluation["aggregation"] != _AGGREGATION
        or evaluation["primary_metric"] != _PRIMARY_METRIC
    ):
        raise ValueError("quality warm-control evaluation protocol differs from v1")
    seed_selection_forbidden = _true(
        evaluation["seed_selection_forbidden"], "seed-selection prohibition"
    )

    claim = record["claim_scope"]
    if not isinstance(claim, Mapping):
        raise ValueError("quality warm-control claim scope must be an object")
    _exact_keys(claim, {"permitted", "prohibited", "required_reports"}, "claim scope")
    prohibited = _text_tuple(claim["prohibited"], "prohibited claims")
    reports = _text_tuple(claim["required_reports"], "required reports")
    if (
        claim["permitted"] != _PERMITTED_CLAIM
        or prohibited != _PROHIBITED_CLAIMS
        or reports != _REQUIRED_REPORTS
    ):
        raise ValueError("quality warm-control claim scope differs from v1")

    return QualityWarmControlRegistry(
        path=registry_path.resolve(),
        file_sha256=hashlib.sha256(raw).hexdigest(),
        record_digest=digest,
        protocol_id=QUALITY_WARM_CONTROL_PROTOCOL_ID,
        main_grid_sha256=main_sha,
        selection_effect=selection_effect,
        model_family=_MODEL_FAMILY,
        quality_prior_mode=_QUALITY_PRIOR_MODE,
        model_signature_sha256=observed.parameter_signature_sha256,
        trainable_parameters=observed.trainable_parameters,
        total_parameters=observed.total_parameters,
        treatment_profile=treatment,
        control_profile=control,
        training_seeds=parsed_seeds,
        source_treatment_cells=tuple(parsed_source_cells),
        training_protocol=training,
        same_parameter_schema=True,
        same_initial_state_for_seed=True,
        same_training_data_order=True,
        evaluation_partition=_EVALUATION_PARTITION,
        aggregation=_AGGREGATION,
        primary_metric=_PRIMARY_METRIC,
        seed_selection_forbidden=seed_selection_forbidden,
        permitted_claim=_PERMITTED_CLAIM,
        prohibited_claims=prohibited,
        required_reports=reports,
    )


def _plan_payload(plan: QualityWarmControlPlan) -> dict[str, object]:
    return {
        "domain": "isingfold-quality-warm-control-plan-v1",
        "registry_file_sha256": plan.registry_file_sha256,
        "registry_record_digest": plan.registry_record_digest,
        "main_grid_sha256": plan.main_grid_sha256,
        "selected_simpler": plan.selected_simpler,
        "representation_selection_sha256": plan.representation_selection_sha256,
        "representation_selection_record_digest": (
            plan.representation_selection_record_digest
        ),
        "corpus_manifest_sha256": plan.corpus_manifest_sha256,
        "selector_digest": plan.selector_digest,
        "normalizer_digest": plan.normalizer_digest,
        "quality_manifest_sha256": plan.quality_manifest_sha256,
        "quality_preflight_sha256": plan.quality_preflight_sha256,
        "quality_preflight_record_digest": plan.quality_preflight_record_digest,
        "cells": [
            {
                "control_cell_id": cell.control_cell_id,
                "source_treatment_cell_id": cell.source_treatment_cell_id,
                "seed": cell.seed,
                "model_family": cell.model_family,
                "model_signature_sha256": cell.model_signature_sha256,
                "profile_id": cell.profile_id,
            }
            for cell in plan.cells
        ],
        "training_protocol": {
            "loss": plan.training_protocol.loss,
            "reduction": plan.training_protocol.reduction,
            "epochs": plan.training_protocol.epochs,
            "minibatch_records": plan.training_protocol.minibatch_records,
            "learning_rate": plan.training_protocol.learning_rate,
            "weight_decay": plan.training_protocol.weight_decay,
            "minimum_resolved_rows": plan.training_protocol.minimum_resolved_rows,
            "minimum_resolved_lineages": (
                plan.training_protocol.minimum_resolved_lineages
            ),
        },
        "selection_effect": plan.selection_effect,
        "evaluation_partition": plan.evaluation_partition,
        "aggregation": plan.aggregation,
        "primary_metric": plan.primary_metric,
        "seed_selection_forbidden": plan.seed_selection_forbidden,
    }


def build_quality_warm_control_plan(
    registry: QualityWarmControlRegistry,
    *,
    selected_simpler: str,
    representation_selection_sha256: str,
    representation_selection_record_digest: str,
    corpus_manifest_sha256: str,
    selector_digest: str,
    normalizer_digest: str,
    quality_manifest_sha256: str,
    quality_preflight_sha256: str,
    quality_preflight_record_digest: str,
) -> QualityWarmControlPlan:
    """Bind three controls to a frozen representation decision and exact train inputs."""

    if not isinstance(registry, QualityWarmControlRegistry):
        raise TypeError("registry must be an authenticated QualityWarmControlRegistry")
    if selected_simpler not in {"if-mlp", "if-dual"}:
        raise ValueError("quality warm-control selected simpler family is invalid")
    digests = {
        "representation_selection_sha256": representation_selection_sha256,
        "representation_selection_record_digest": representation_selection_record_digest,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "selector_digest": selector_digest,
        "normalizer_digest": normalizer_digest,
        "quality_manifest_sha256": quality_manifest_sha256,
        "quality_preflight_sha256": quality_preflight_sha256,
        "quality_preflight_record_digest": quality_preflight_record_digest,
    }
    digests = {name: _sha256(value, name) for name, value in digests.items()}
    source = dict(registry.source_treatment_cells)
    cells = tuple(
        QualityWarmControlCell(
            control_cell_id=f"qwarm-control-if-core-s{seed}",
            source_treatment_cell_id=source[seed],
            seed=seed,
            model_family=registry.model_family,
            model_signature_sha256=registry.model_signature_sha256,
            profile_id=registry.control_profile.profile_id,
        )
        for seed in registry.training_seeds
    )
    provisional = QualityWarmControlPlan(
        registry_file_sha256=registry.file_sha256,
        registry_record_digest=registry.record_digest,
        main_grid_sha256=registry.main_grid_sha256,
        selected_simpler=selected_simpler,
        representation_selection_sha256=digests["representation_selection_sha256"],
        representation_selection_record_digest=digests[
            "representation_selection_record_digest"
        ],
        corpus_manifest_sha256=digests["corpus_manifest_sha256"],
        selector_digest=digests["selector_digest"],
        normalizer_digest=digests["normalizer_digest"],
        quality_manifest_sha256=digests["quality_manifest_sha256"],
        quality_preflight_sha256=digests["quality_preflight_sha256"],
        quality_preflight_record_digest=digests["quality_preflight_record_digest"],
        cells=cells,
        training_protocol=registry.training_protocol,
        selection_effect=registry.selection_effect,
        evaluation_partition=registry.evaluation_partition,
        aggregation=registry.aggregation,
        primary_metric=registry.primary_metric,
        seed_selection_forbidden=True,
        plan_digest="0" * 64,
    )
    return replace(provisional, plan_digest=content_digest(_plan_payload(provisional)))


def build_quality_warm_control_binding(
    registry: QualityWarmControlRegistry,
    plan: QualityWarmControlPlan,
    *,
    cell: QualityWarmControlCell,
    source_treatment_run_record_digest: str,
    source_treatment_checkpoint_payload_digest: str,
    runtime_implementation_digest: str,
) -> dict[str, object]:
    """Sign the exact source treatment consumed by one diagnostic control run."""

    if not isinstance(registry, QualityWarmControlRegistry):
        raise TypeError("registry must be an authenticated QualityWarmControlRegistry")
    if not isinstance(plan, QualityWarmControlPlan) or plan.registry_record_digest != registry.record_digest:
        raise ValueError("quality warm-control plan differs from its registry")
    if cell not in plan.cells:
        raise ValueError("quality warm-control cell is absent from its plan")
    payload: dict[str, object] = {
        "schema": QUALITY_WARM_CONTROL_BINDING_SCHEMA,
        "schema_version": QUALITY_WARM_CONTROL_BINDING_VERSION,
        "protocol_id": registry.protocol_id,
        "registry_file_sha256": registry.file_sha256,
        "registry_record_digest": registry.record_digest,
        "plan_digest": plan.plan_digest,
        "main_grid_sha256": registry.main_grid_sha256,
        "representation_selection_sha256": plan.representation_selection_sha256,
        "representation_selection_record_digest": (
            plan.representation_selection_record_digest
        ),
        "control_cell_id": cell.control_cell_id,
        "source_treatment_cell_id": cell.source_treatment_cell_id,
        "source_treatment_run_record_digest": _sha256(
            source_treatment_run_record_digest, "source treatment record digest"
        ),
        "source_treatment_checkpoint_payload_digest": _sha256(
            source_treatment_checkpoint_payload_digest,
            "source treatment checkpoint payload digest",
        ),
        "runtime_implementation_digest": _sha256(
            runtime_implementation_digest, "runtime implementation digest"
        ),
        "profile_id": cell.profile_id,
        "model_family": cell.model_family,
        "seed": cell.seed,
        "corpus_manifest_sha256": plan.corpus_manifest_sha256,
        "selector_digest": plan.selector_digest,
        "normalizer_digest": plan.normalizer_digest,
        "quality_manifest_sha256": plan.quality_manifest_sha256,
        "quality_preflight_sha256": plan.quality_preflight_sha256,
        "quality_preflight_record_digest": plan.quality_preflight_record_digest,
        "selection_effect": registry.selection_effect,
    }
    return {**payload, "record_digest": content_digest(payload)}


_BINDING_FIELDS = {
    "control_cell_id",
    "corpus_manifest_sha256",
    "main_grid_sha256",
    "model_family",
    "normalizer_digest",
    "plan_digest",
    "profile_id",
    "protocol_id",
    "quality_manifest_sha256",
    "quality_preflight_record_digest",
    "quality_preflight_sha256",
    "record_digest",
    "registry_file_sha256",
    "registry_record_digest",
    "representation_selection_record_digest",
    "representation_selection_sha256",
    "runtime_implementation_digest",
    "schema",
    "schema_version",
    "seed",
    "selection_effect",
    "selector_digest",
    "source_treatment_cell_id",
    "source_treatment_checkpoint_payload_digest",
    "source_treatment_run_record_digest",
}


def validate_quality_warm_control_binding(
    raw: object,
    *,
    expected_profile_id: str,
    expected_seed: int,
    expected_model_family: str,
) -> dict[str, object]:
    """Fail closed before a diagnostic profile can enter the trainer."""

    if not isinstance(raw, Mapping):
        raise ValueError("rank-value warm control requires an authenticated diagnostic binding")
    _exact_keys(raw, _BINDING_FIELDS, "quality warm-control binding")
    binding = dict(raw)
    _verify_record(binding, "quality warm-control binding")
    if (
        binding["schema"] != QUALITY_WARM_CONTROL_BINDING_SCHEMA
        or binding["schema_version"] != QUALITY_WARM_CONTROL_BINDING_VERSION
        or binding["protocol_id"] != QUALITY_WARM_CONTROL_PROTOCOL_ID
        or binding["profile_id"] != expected_profile_id
        or binding["seed"] != expected_seed
        or binding["model_family"] != expected_model_family
        or binding["selection_effect"] != QUALITY_WARM_CONTROL_SELECTION_EFFECT
    ):
        raise ValueError("quality warm-control binding has an incompatible identity")
    for field in _BINDING_FIELDS - {
        "control_cell_id",
        "model_family",
        "profile_id",
        "protocol_id",
        "schema",
        "schema_version",
        "seed",
        "selection_effect",
        "source_treatment_cell_id",
    }:
        _sha256(binding[field], f"quality warm-control binding {field}")
    return binding


__all__ = [
    "QUALITY_WARM_CONTROL_BINDING_SCHEMA",
    "QUALITY_WARM_CONTROL_BINDING_VERSION",
    "QUALITY_WARM_CONTROL_MAIN_GRID_SHA256",
    "QUALITY_WARM_CONTROL_PROFILE_ID",
    "QUALITY_WARM_CONTROL_PROTOCOL_ID",
    "QUALITY_WARM_CONTROL_SCHEMA",
    "QUALITY_WARM_CONTROL_SELECTION_EFFECT",
    "QUALITY_WARM_CONTROL_STAGE",
    "QUALITY_WARM_CONTROL_V1_RECORD_DIGEST",
    "QUALITY_WARM_CONTROL_VERSION",
    "QualityWarmControlCell",
    "QualityWarmControlPlan",
    "QualityWarmControlRegistry",
    "QualityWarmTrainingProtocol",
    "build_quality_warm_control_binding",
    "build_quality_warm_control_plan",
    "load_quality_warm_control_registry",
    "validate_quality_warm_control_binding",
]
