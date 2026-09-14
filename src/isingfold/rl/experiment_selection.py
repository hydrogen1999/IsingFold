"""Fail-closed validation freeze for the registered eighteen-cell RL-value grid.

This module deliberately has no sealed-test entry point.  It authenticates validation reports,
resolves the ``selected-simpler`` placeholder from the earlier representation receipt, and
selects one *training configuration* over all registered seeds.  It never selects a seed or a
single lucky checkpoint.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.evaluate import (
    EVALUATION_RECEIPT_VERSION,
    EpisodeOutcome,
    read_episode_receipts,
    secondary_metrics,
)


RL_VALUE_FREEZE_SCHEMA = "isingfold.rl-value-validation-freeze"
RL_VALUE_FREEZE_VERSION = 4
REPRESENTATION_SELECTION_SCHEMA = "isingfold.representation-selection"
REPRESENTATION_SELECTION_SCHEMA_VERSION = 4
REPRESENTATION_SELECTION_RULE = (
    "equal-seed/equal-base-lineage unconditional utility among families passing "
    "the preregistered one-sided feasibility constraint; lower online time then "
    "IF-MLP as deterministic ties"
)
RL_VALUE_SELECTION_RULE = (
    "maximize unconditional IF-Q3-S0 utility over complete equal-training-seed and "
    "equal-immutable-base-lineage aggregates whose one-sided feasibility lower bound "
    "versus return_initial strictly exceeds -0.02; then lower online seconds; then "
    "registered configuration order"
)

_MODEL_FAMILIES = ("if-mlp", "if-dual", "if-core")
_GRID_FAMILIES = ("selected-simpler", "if-core")
_METHODS = ("supervised-only", "ppo-warm-start", "ppo-from-scratch")
_SAME_SUPPORT_ARMS = (
    "return_initial",
    "random_masked",
    "classical_resource_first",
    "classical_quality_aware",
    "policy",
)
_PROTOCOL_FIELDS = {
    "population",
    "quality_authority",
    "target_access",
    "ground_partition_receipt",
    "budget",
    "selector",
    "proposal",
    "support",
    "policy_seed",
    "evaluator_reads",
    "initializer",
    "repetitions",
    "inference_device_type",
    "inference_device_name",
    "inference_threads",
    "runtime_platform",
    "baseline_controller_registry",
    "evaluation_receipt_schema",
    "evaluation_implementation_sha256",
    "quality_preflight_receipt_sha256",
    "quality_preflight_record_digest",
    "runtime_implementation_registry",
    "runtime_implementation_digest",
}
_RL_BOOTSTRAP_PROTOCOL_FIELDS = {
    "bootstrap_bank_access",
    "same_support_contract_digest",
}
_RUNTIME_PROTOCOL_FIELDS = {
    "inference_device_name",
    "inference_threads",
    "runtime_platform",
}
_SCIENTIFIC_PROTOCOL_FIELDS = _PROTOCOL_FIELDS - _RUNTIME_PROTOCOL_FIELDS
_RUNTIME_FIELDS = {
    "hostname",
    "system",
    "release",
    "machine",
    "processor",
    "logical_cpu_count",
    "slurm_partition",
}
_RL_PROTOCOL_FIELDS = {
    "partition",
    "evaluation_seed",
    "repetitions",
    "audit_reads",
    "margin",
    "deployment_rule",
    "feasibility_reference",
    "feasibility_alpha",
    "feasibility_bootstrap",
    "selection_bootstrap_seed",
    "aggregation",
    "primary_metric",
    "tie_break",
}
_AGGREGATION = "equal-training-seed-then-equal-immutable-base-lineage"
_PRIMARY_METRIC = "unconditional-if-q3-s0-utility"
_TIE_BREAK = "lower-online-seconds-then-registered-configuration-order"
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class _GridCell:
    cell_id: str
    grid_family: str
    model_family: str
    method: str
    seed: int
    registered_order: int


@dataclass(frozen=True)
class _LoadedCell:
    cell: _GridCell
    report: Mapping[str, Any]
    policy: tuple[EpisodeOutcome, ...]
    reference: tuple[EpisodeOutcome, ...]
    source: Mapping[str, object]


@dataclass(frozen=True)
class FrozenRLValueSelection:
    """Authenticated three-seed configuration selected without opening test data."""

    receipt_sha256: str
    record_digest: str
    grid_manifest_sha256: str
    model_family: str
    grid_model_family: str
    method: str
    training_seeds: tuple[int, ...]
    cell_ids: tuple[str, ...]
    checkpoint_payload_digests: tuple[str, ...]
    representation_selection_sha256: str
    representation_selection_record_digest: str
    runtime_implementation_registry: Mapping[str, object]
    runtime_implementation_digest: str
    quality_preflight_receipt_sha256: str
    quality_preflight_record_digest: str

    def __post_init__(self) -> None:
        for value in (
            self.receipt_sha256,
            self.record_digest,
            self.grid_manifest_sha256,
            *self.checkpoint_payload_digests,
            self.representation_selection_sha256,
            self.representation_selection_record_digest,
            self.runtime_implementation_digest,
            self.quality_preflight_receipt_sha256,
            self.quality_preflight_record_digest,
        ):
            if not _is_digest(value):
                raise ValueError("frozen RL-value selection contains an invalid digest")
        if self.model_family not in _MODEL_FAMILIES:
            raise ValueError("frozen RL-value selection names an unknown model family")
        if self.grid_model_family not in _GRID_FAMILIES or self.method not in _METHODS:
            raise ValueError("frozen RL-value selection names an unknown grid configuration")
        if len(self.training_seeds) != 3 or len(set(self.training_seeds)) != 3:
            raise ValueError("frozen RL-value selection must retain all three training seeds")
        if tuple(sorted(self.training_seeds)) != self.training_seeds:
            raise ValueError("frozen RL-value training seeds must be sorted")
        if len(self.cell_ids) != 3 or len(set(self.cell_ids)) != 3:
            raise ValueError("frozen RL-value selection must retain three distinct cells")
        if len(self.checkpoint_payload_digests) != 3:
            raise ValueError("frozen RL-value selection must bind all three source checkpoints")
        if (
            not isinstance(self.runtime_implementation_registry, Mapping)
            or content_digest(self.runtime_implementation_registry)
            != self.runtime_implementation_digest
        ):
            raise ValueError("frozen RL-value selection has an inconsistent runtime registry")

    def cell_for_index(self, index: int) -> tuple[str, int, str]:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < 3:
            raise ValueError("selected complete-system seed index must be in [0, 3)")
        return (
            self.cell_ids[index],
            self.training_seeds[index],
            self.checkpoint_payload_digests[index],
        )


def _no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"nonfinite JSON scalar {value!r}")


def _strict_json(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"cannot read {label}: {path}") from exc
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not UTF-8") from exc
    try:
        value = json.loads(
            decoded,
            object_pairs_hook=_no_duplicates,
            parse_constant=_reject_nonfinite,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _digest(value: object, label: str) -> str:
    if not _is_digest(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return str(value)


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _verify_record(record: Mapping[str, Any], label: str) -> str:
    digest = _digest(record.get("record_digest"), f"{label} record digest")
    unsigned = {key: value for key, value in record.items() if key != "record_digest"}
    expected = content_digest(unsigned)
    if not hmac.compare_digest(digest, expected):
        raise ValueError(f"{label} record digest mismatch")
    return digest


def _validate_rl_protocol(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict) or set(raw) != _RL_PROTOCOL_FIELDS:
        raise ValueError("grid has no exact RL-value validation protocol")
    protocol = dict(raw)
    if (
        protocol["partition"] != "validation"
        or protocol["deployment_rule"] != "categorical-temperature-one"
        or protocol["feasibility_reference"] != "return_initial"
        or protocol["aggregation"] != _AGGREGATION
        or protocol["primary_metric"] != _PRIMARY_METRIC
        or protocol["tie_break"] != _TIE_BREAK
    ):
        raise ValueError("RL-value validation protocol changes a registered scientific rule")
    _nonnegative_integer(protocol["evaluation_seed"], "RL-value evaluation seed")
    _positive_integer(protocol["repetitions"], "RL-value repetitions")
    _positive_integer(protocol["audit_reads"], "RL-value audit reads")
    margin = _finite(protocol["margin"], "RL-value feasibility margin")
    alpha = _finite(protocol["feasibility_alpha"], "RL-value feasibility alpha")
    if not math.isclose(margin, 0.02, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("RL-value feasibility margin must remain exactly 0.02")
    if not 0.0 < alpha < 0.5:
        raise ValueError("RL-value feasibility alpha must be in (0, 0.5)")
    if _positive_integer(protocol["feasibility_bootstrap"], "RL-value bootstrap count") < 1_000:
        raise ValueError("RL-value bootstrap count is too small for the registered freeze")
    _nonnegative_integer(protocol["selection_bootstrap_seed"], "RL-value bootstrap seed")
    return protocol


def _load_grid(path: Path) -> tuple[dict[str, Any], str, dict[str, object]]:
    grid = _strict_json(path, "staged-grid manifest")
    if (
        grid.get("schema") != "isingfold.staged-grid"
        or grid.get("schema_version") != 2
        or grid.get("name") != "if-core-v2-profile-i-hybrid-chimera-registered"
    ):
        raise ValueError("unsupported staged-grid manifest")
    environment = grid.get("environment")
    selection = grid.get("selection")
    stages = grid.get("stages")
    if (
        not isinstance(environment, dict)
        or environment.get("endpoint") != "IF-Q3-S0"
        or environment.get("mode") != "improvement"
        or not isinstance(selection, dict)
        or selection.get("seed_selection_forbidden") is not True
        or selection.get("validation_partition_only") is not True
        or selection.get("test_partition_locked") is not True
        or not isinstance(stages, dict)
    ):
        raise ValueError("staged grid is incompatible with the locked RL-value study")
    representation = stages.get("representation")
    rl_value = stages.get("rl_value")
    if not isinstance(representation, list) or len(representation) != 9:
        raise ValueError("representation grid must contain exactly nine cells")
    if not isinstance(rl_value, list) or len(rl_value) != 18:
        raise ValueError("RL-value grid must contain exactly eighteen cells")

    representation_census: list[tuple[str, str, int]] = []
    for raw in representation:
        if not isinstance(raw, dict):
            raise ValueError("representation grid cell must be an object")
        cell_id = raw.get("cell_id")
        family = raw.get("model_family")
        seed = raw.get("seed")
        if not isinstance(cell_id, str) or not cell_id or family not in _MODEL_FAMILIES:
            raise ValueError("representation grid has an invalid cell identity")
        representation_census.append(
            (cell_id, str(family), _nonnegative_integer(seed, "representation seed"))
        )
    if len({cell_id for cell_id, _, _ in representation_census}) != 9:
        raise ValueError("representation grid has duplicate cell IDs")
    representation_seeds: tuple[int, ...] | None = None
    for family in _MODEL_FAMILIES:
        seeds = tuple(sorted(seed for _, name, seed in representation_census if name == family))
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError(f"representation family {family} lacks three registered seeds")
        if representation_seeds is None:
            representation_seeds = seeds
        elif seeds != representation_seeds:
            raise ValueError("representation families do not share one seed registry")

    rl_census: list[tuple[str, str, str, int]] = []
    for raw in rl_value:
        if not isinstance(raw, dict):
            raise ValueError("RL-value grid cell must be an object")
        cell_id = raw.get("cell_id")
        family = raw.get("model_family")
        method = raw.get("method")
        seed = raw.get("seed")
        if (
            not isinstance(cell_id, str)
            or not cell_id
            or family not in _GRID_FAMILIES
            or method not in _METHODS
        ):
            raise ValueError("RL-value grid has an invalid family, method, or cell ID")
        rl_census.append(
            (
                cell_id,
                str(family),
                str(method),
                _nonnegative_integer(seed, "RL-value training seed"),
            )
        )
    if len({cell_id for cell_id, _, _, _ in rl_census}) != 18:
        raise ValueError("RL-value grid has duplicate cell IDs")
    seed_registry: tuple[int, ...] | None = None
    for family in _GRID_FAMILIES:
        for method in _METHODS:
            seeds = tuple(
                sorted(
                    seed
                    for _, observed_family, observed_method, seed in rl_census
                    if observed_family == family and observed_method == method
                )
            )
            if len(seeds) != 3 or len(set(seeds)) != 3:
                raise ValueError(
                    f"RL-value configuration {family}/{method} lacks three registered seeds"
                )
            if seed_registry is None:
                seed_registry = seeds
            elif seeds != seed_registry:
                raise ValueError("RL-value configurations do not share one seed registry")
    if seed_registry != representation_seeds:
        raise ValueError("representation and RL-value stages use different seed registries")
    protocol = _validate_rl_protocol(grid.get("rl_value_evaluation"))
    representation_protocol = grid.get("representation_evaluation")
    if not isinstance(representation_protocol, Mapping):
        raise ValueError("grid has no representation-evaluation protocol")
    representation_evaluation_seed = _nonnegative_integer(
        representation_protocol.get("evaluation_seed"),
        "representation evaluation seed",
    )
    if representation_evaluation_seed == protocol["evaluation_seed"]:
        raise ValueError(
            "representation screening and RL-value selection require disjoint "
            "evaluation seed domains"
        )
    return grid, _sha256_file(path), protocol


def _representation_constraint(
    grid: Mapping[str, Any], receipt: Mapping[str, Any]
) -> tuple[str, dict[str, Mapping[str, Any]]]:
    evaluation = grid.get("representation_evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("grid lacks the representation validation protocol")
    expected = {
        "reference": evaluation.get("feasibility_reference"),
        "margin": float(evaluation.get("margin")),
        "one_sided_alpha": float(evaluation.get("feasibility_alpha")),
        "cluster_unit": "immutable-base-lineage",
        "training_seed_treatment": "equal-weight-crossed-resampling-with-replacement",
        "lineage_treatment": "equal-weight-crossed-resampling-with-replacement",
        "bootstrap_replicates": evaluation.get("feasibility_bootstrap"),
        "bootstrap_seed": evaluation.get("selection_bootstrap_seed"),
        "pass_rule": "lower_bound_strictly_greater_than_negative_margin",
    }
    constraint = receipt.get("feasibility_constraint")
    if not isinstance(constraint, dict) or any(
        constraint.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("representation receipt changes its registered feasibility protocol")
    aggregates = receipt.get("family_aggregates")
    if not isinstance(aggregates, dict) or set(aggregates) != set(_MODEL_FAMILIES):
        raise ValueError("representation receipt has incomplete family aggregates")
    typed: dict[str, Mapping[str, Any]] = {}
    representation = grid["stages"]["representation"]
    for family in _MODEL_FAMILIES:
        aggregate = aggregates[family]
        if not isinstance(aggregate, dict):
            raise ValueError(f"representation aggregate {family} is malformed")
        seeds = sorted(
            int(cell["seed"]) for cell in representation if cell["model_family"] == family
        )
        if aggregate.get("seed_count") != 3 or aggregate.get("seeds") != seeds:
            raise ValueError(f"representation aggregate {family} omits a registered seed")
        utility = _finite(aggregate.get("utility_mean"), f"{family} utility")
        online = _finite(aggregate.get("online_seconds_mean"), f"{family} online cost")
        low = _finite(aggregate.get("feasibility_ci_low"), f"{family} feasibility bound")
        if online < 0.0 or not 0.0 <= utility <= 1.0:
            raise ValueError(f"representation aggregate {family} has an invalid metric")
        passed = low > -float(expected["margin"])
        if aggregate.get("feasibility_noninferior") is not passed:
            raise ValueError(f"representation aggregate {family} has an inconsistent gate")
        typed[family] = aggregate
    eligible = [
        family
        for family in ("if-mlp", "if-dual")
        if typed[family].get("feasibility_noninferior") is True
    ]
    if not eligible or constraint.get("eligible_simpler") != eligible:
        raise ValueError("representation feasible-family shortlist is inconsistent")
    if (
        constraint.get("if_core_pass") is not True
        or typed["if-core"].get("feasibility_noninferior") is not True
    ):
        raise ValueError("representation IF-Core feasibility gate did not pass")
    expected_selected = max(
        eligible,
        key=lambda family: (
            float(typed[family]["utility_mean"]),
            -float(typed[family]["online_seconds_mean"]),
            family == "if-mlp",
        ),
    )
    return expected_selected, typed


def _load_representation_receipt(
    path: Path, *, grid: Mapping[str, Any], grid_digest: str
) -> tuple[str, dict[str, object]]:
    receipt = _strict_json(path, "representation-selection receipt")
    record_digest = _verify_record(receipt, "representation-selection receipt")
    if (
        receipt.get("schema") != REPRESENTATION_SELECTION_SCHEMA
        or receipt.get("schema_version") != REPRESENTATION_SELECTION_SCHEMA_VERSION
        or receipt.get("stage") != "representation"
        or receipt.get("partition") != "validation"
        or receipt.get("seed_selection_forbidden") is not True
        or receipt.get("grid_manifest_sha256") != grid_digest
        or receipt.get("selection_rule") != REPRESENTATION_SELECTION_RULE
    ):
        raise ValueError("representation-selection receipt has an incompatible identity")
    selected = receipt.get("selected_simpler")
    if selected not in {"if-mlp", "if-dual"}:
        raise ValueError("representation receipt has no registered simpler family")
    if receipt.get("retained_families") != [selected, "if-core"]:
        raise ValueError("representation receipt has an invalid retained-family set")
    representation = grid["stages"]["representation"]
    expected_census = {
        (cell["cell_id"], cell["model_family"], cell["seed"]) for cell in representation
    }
    source_reports = receipt.get("source_reports")
    if not isinstance(source_reports, list):
        raise ValueError("representation receipt has no source-report census")
    observed_census = {
        (row.get("cell_id"), row.get("model_family"), row.get("seed"))
        for row in source_reports
        if isinstance(row, dict)
    }
    if len(source_reports) != 9 or observed_census != expected_census:
        raise ValueError("representation receipt does not cover its complete nine-cell grid")
    training_seeds = sorted({int(cell["seed"]) for cell in representation})
    protocol_identity = receipt.get("protocol_identity")
    if not isinstance(protocol_identity, dict):
        raise ValueError("representation receipt has no scientific protocol identity")
    scientific_protocol = _validate_scientific_protocol_identity(
        protocol_identity.get("matched_metadata")
    )
    pinned_runtime_registry = receipt.get("runtime_implementation_registry")
    pinned_runtime_digest = receipt.get("runtime_implementation_digest")
    if (
        not isinstance(pinned_runtime_registry, dict)
        or pinned_runtime_digest != content_digest(pinned_runtime_registry)
        or scientific_protocol.get("runtime_implementation_registry")
        != pinned_runtime_registry
        or scientific_protocol.get("runtime_implementation_digest") != pinned_runtime_digest
    ):
        raise ValueError("representation receipt has an inconsistent runtime registry")
    runtime_by_seed = _validate_runtime_identity_by_seed(
        receipt.get("runtime_identity_by_training_seed"),
        expected_seeds=training_seeds,
    )
    for row in source_reports:
        if not isinstance(row, dict):
            raise ValueError("representation source report is malformed")
        for field in (
            "report_sha256",
            "report_record_digest",
            "policy_receipt_sha256",
            "checkpoint_payload_digest",
        ):
            _digest(row.get(field), f"representation source {field}")
        seed = row.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("representation source training seed is malformed")
        if _validate_runtime_identity(row.get("runtime_identity")) != runtime_by_seed[str(seed)]:
            raise ValueError("representation source runtime differs within a training seed")
        if (
            row.get("runtime_implementation_registry") != pinned_runtime_registry
            or row.get("runtime_implementation_digest") != pinned_runtime_digest
        ):
            raise ValueError(
                "all 27 validation cells require byte-identical source and dependency versions"
            )
    expected_selected, _ = _representation_constraint(grid, receipt)
    if selected != expected_selected:
        raise ValueError("receipt violates the registered representation selection")
    return str(selected), {
        "path": str(path),
        "sha256": _sha256_file(path),
        "record_digest": record_digest,
        "selected_simpler": selected,
        "source_report_count": len(source_reports),
        "runtime_implementation_registry": pinned_runtime_registry,
        "runtime_implementation_digest": pinned_runtime_digest,
        "quality_preflight_receipt_sha256": scientific_protocol[
            "quality_preflight_receipt_sha256"
        ],
        "quality_preflight_record_digest": scientific_protocol[
            "quality_preflight_record_digest"
        ],
    }


def _resolved_cells(grid: Mapping[str, Any], selected: str) -> list[_GridCell]:
    cells: list[_GridCell] = []
    for index, raw in enumerate(grid["stages"]["rl_value"]):
        grid_family = str(raw["model_family"])
        cells.append(
            _GridCell(
                cell_id=str(raw["cell_id"]),
                grid_family=grid_family,
                model_family=selected if grid_family == "selected-simpler" else grid_family,
                method=str(raw["method"]),
                seed=int(raw["seed"]),
                registered_order=index,
            )
        )
    census = {(cell.model_family, cell.method, cell.seed) for cell in cells}
    if len(census) != 18:
        raise ValueError("selected-simpler resolution does not produce eighteen unique cells")
    return cells


def _validate_runtime(runtime: object) -> None:
    if not isinstance(runtime, dict) or set(runtime) != _RUNTIME_FIELDS:
        raise ValueError("evaluation runtime-platform identity is malformed")
    _positive_integer(runtime.get("logical_cpu_count"), "logical CPU count")
    if runtime.get("slurm_partition") is not None and not isinstance(
        runtime.get("slurm_partition"), str
    ):
        raise ValueError("Slurm partition must be a string or null")
    for field in ("system", "release", "machine", "processor"):
        if not isinstance(runtime.get(field), str):
            raise ValueError(f"runtime field {field} must be a string")


def _split_protocol_identity(
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Separate scientific controls from the registered seed-blocked runtime identity."""

    protocol_fields = set(protocol)
    allowed_fields = (
        _PROTOCOL_FIELDS,
        _PROTOCOL_FIELDS | _RL_BOOTSTRAP_PROTOCOL_FIELDS,
    )
    if protocol_fields not in allowed_fields:
        raise ValueError("evaluation has an incomplete matched protocol")
    device_name = protocol.get("inference_device_name")
    if not isinstance(device_name, str) or not device_name:
        raise ValueError("matched inference device name must be nonempty")
    inference_threads = _positive_integer(
        protocol.get("inference_threads"), "inference thread count"
    )
    _validate_runtime(protocol.get("runtime_platform"))
    scientific = {
        field: protocol[field]
        for field in sorted(protocol_fields - _RUNTIME_PROTOCOL_FIELDS)
    }
    runtime = {
        "inference_device_name": device_name,
        "inference_threads": inference_threads,
        "runtime_platform": protocol["runtime_platform"],
    }
    return scientific, runtime


def _validate_scientific_protocol_identity(protocol: object) -> dict[str, Any]:
    if not isinstance(protocol, dict) or set(protocol) not in (
        _SCIENTIFIC_PROTOCOL_FIELDS,
        _SCIENTIFIC_PROTOCOL_FIELDS | _RL_BOOTSTRAP_PROTOCOL_FIELDS,
    ):
        raise ValueError("freeze receipt has an incomplete scientific protocol identity")
    device_type = protocol.get("inference_device_type")
    if not isinstance(device_type, str) or not device_type:
        raise ValueError("scientific inference device type must be nonempty")
    return {field: protocol[field] for field in sorted(protocol)}


def _validate_runtime_identity(runtime: object) -> dict[str, Any]:
    if not isinstance(runtime, dict) or set(runtime) != _RUNTIME_PROTOCOL_FIELDS:
        raise ValueError("seed runtime identity is malformed")
    device_name = runtime.get("inference_device_name")
    if not isinstance(device_name, str) or not device_name:
        raise ValueError("seed inference device name must be nonempty")
    threads = _positive_integer(runtime.get("inference_threads"), "inference thread count")
    _validate_runtime(runtime.get("runtime_platform"))
    return {
        "inference_device_name": device_name,
        "inference_threads": threads,
        "runtime_platform": runtime["runtime_platform"],
    }


def _validate_runtime_identity_by_seed(
    raw: object, *, expected_seeds: Sequence[int]
) -> dict[str, dict[str, Any]]:
    expected_keys = {str(seed) for seed in expected_seeds}
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        raise ValueError("runtime identity map does not cover the registered training seeds")
    return {
        key: _validate_runtime_identity(raw[key]) for key in sorted(expected_keys, key=int)
    }


def _arm_path(report_path: Path, relative: object, arm: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{arm} receipt has no local path")
    parsed = Path(relative)
    if parsed.is_absolute() or len(parsed.parts) != 1 or parsed.name in {".", ".."}:
        raise ValueError(f"{arm} receipt path must be one local filename")
    return report_path.parent / parsed


def _population_signature(outcomes: Sequence[EpisodeOutcome]) -> tuple[tuple[object, ...], ...]:
    return tuple(
        sorted(
            (
                *outcome.pair_key,
                outcome.population_eligible,
                outcome.episode_seed,
            )
            for outcome in outcomes
        )
    )


def reference_outcome_signature(
    outcomes: Sequence[EpisodeOutcome],
) -> tuple[tuple[object, ...], ...]:
    """Canonical identity of a return-initial population and its evaluator evidence."""

    return tuple(
        sorted(
            (
                *outcome.pair_key,
                outcome.population_eligible,
                outcome.returned_valid,
                outcome.utility,
                outcome.qubits,
                outcome.max_chain,
                outcome.episode_seed,
                outcome.evaluator_seed,
                outcome.program_digest,
                outcome.selected_strength,
                outcome.selected_strength_index,
                outcome.evaluator_hits,
                outcome.evaluator_reads,
                outcome.validation_digest,
                outcome.reason,
                outcome.outcome_kind,
                None if outcome.work is None else content_digest(outcome.work.as_dict()),
                outcome.broken_chain_fraction,
                outcome.mean_energy_residual,
            )
            for outcome in outcomes
        )
    )


def _read_complete_receipts_for_freeze(
    path: Path, *, expected_sha256: str
) -> tuple[object, ...]:
    """Narrow testable boundary around the strict complete-receipt parser."""

    from isingfold.rl.complete_system import read_complete_system_receipts

    return tuple(read_complete_system_receipts(path, expected_sha256=expected_sha256))


def _load_evaluation(
    report_path: Path,
    *,
    cell: _GridCell,
    grid_digest: str,
    protocol: Mapping[str, object],
    representation_receipt_sha256: str,
    representation_record_digest: str,
) -> _LoadedCell:
    if not report_path.is_file():
        raise FileNotFoundError(f"missing validation evaluation for {cell.cell_id}: {report_path}")
    report = _strict_json(report_path, f"RL-value evaluation {cell.cell_id}")
    report_record_digest = _verify_record(report, f"RL-value evaluation {cell.cell_id}")
    is_rl_value_cell = cell.cell_id.startswith("rl-")
    expected_identity = {
        "schema": "isingfold.paired-evaluation",
        "schema_version": 4 if is_rl_value_cell else 3,
        "partition": "validation",
        "model_family": cell.model_family,
        "training_method": cell.method,
        "training_seed": cell.seed,
        "grid_cell": cell.cell_id,
        "grid_manifest_sha256": grid_digest,
        "evaluation_seed": protocol["evaluation_seed"],
        "deployment_rule": protocol["deployment_rule"],
    }
    observed_identity = {field: report.get(field) for field in expected_identity}
    if observed_identity != expected_identity:
        raise ValueError(
            f"validation evaluation identity differs for {cell.cell_id}: {observed_identity}"
        )
    if (
        report.get("selection_receipt_sha256") != representation_receipt_sha256
        or report.get("selection_record_digest") != representation_record_digest
        or report.get("selection_mode") != "validated-receipt"
    ):
        raise ValueError(
            f"evaluation {cell.cell_id} is not bound to the representation-selection receipt"
        )
    population = _positive_integer(report.get("population"), "evaluation population")
    repetitions = _positive_integer(report.get("repetitions"), "evaluation repetitions")
    if repetitions != protocol["repetitions"]:
        raise ValueError(f"evaluation repetitions differ for {cell.cell_id}")
    selector_digest = _digest(report.get("selector_digest"), "selector digest")
    corpus_digest = _digest(report.get("corpus_manifest_sha256"), "corpus manifest digest")
    quality_authority = report.get("quality_authority")
    target_access = report.get("target_access")
    ground_partition = report.get("ground_partition_receipt")
    checkpoint_digest = _digest(
        report.get("checkpoint_payload_digest"), "checkpoint payload digest"
    )

    matched = report.get("matched_metadata")
    methods = report.get("methods")
    arm_receipts = report.get("arm_receipts")
    required_arms = _SAME_SUPPORT_ARMS
    expected_protocol_fields = _PROTOCOL_FIELDS | (
        _RL_BOOTSTRAP_PROTOCOL_FIELDS if is_rl_value_cell else set()
    )
    if not isinstance(matched, dict) or set(matched) != expected_protocol_fields:
        raise ValueError(f"evaluation {cell.cell_id} has incomplete matched protocol")
    if not isinstance(methods, dict) or any(
        not isinstance(methods.get(arm), dict) for arm in required_arms
    ):
        raise ValueError(f"evaluation {cell.cell_id} has incomplete method identities")
    if not isinstance(arm_receipts, dict) or any(
        not isinstance(arm_receipts.get(arm), dict) for arm in required_arms
    ):
        raise ValueError(f"evaluation {cell.cell_id} lacks required raw arms")
    if (
        matched.get("population") != f"{corpus_digest}:validation"
        or not isinstance(quality_authority, dict)
        or matched.get("quality_authority") != quality_authority
        or matched.get("target_access") != target_access
        or matched.get("ground_partition_receipt") != ground_partition
        or matched.get("selector") != selector_digest
        or matched.get("policy_seed") != protocol["evaluation_seed"]
        or matched.get("evaluator_reads") != protocol["audit_reads"]
        or matched.get("repetitions") != repetitions
        or matched.get("evaluation_receipt_schema") != EVALUATION_RECEIPT_VERSION
        or matched.get("support") != "same-conditional-generator-and-exact-mask-contract"
        or matched.get("initializer")
        != (
            "authenticated-persistent-upfront-lac-cache-k2-v2"
            if is_rl_value_cell
            else "authenticated-profile-i"
        )
    ):
        raise ValueError(f"evaluation {cell.cell_id} changes population/search/evaluator protocol")
    if is_rl_value_cell:
        bootstrap_access = matched.get("bootstrap_bank_access")
        support_digest = matched.get("same_support_contract_digest")
        if (
            not isinstance(bootstrap_access, dict)
            or bootstrap_access.get("schema") != "isingfold.rl-value-bootstrap-access"
            or bootstrap_access.get("schema_version") != 1
            or bootstrap_access.get("protocol_preset") != "validation"
            or bootstrap_access.get("opened_evaluator_targets") is not False
            or bootstrap_access.get("denominator_count") != population * repetitions
            or bootstrap_access.get("same_support_contract_digest") != support_digest
            or report.get("bootstrap_bank_access") != bootstrap_access
            or report.get("same_support_contract_digest") != support_digest
        ):
            raise ValueError(
                f"evaluation {cell.cell_id} lacks its authenticated validation bootstrap bank"
            )
        _verify_record(bootstrap_access, "evaluation bootstrap-bank access")
        _digest(support_digest, "evaluation same-support contract")
    if not isinstance(target_access, dict) or not isinstance(ground_partition, dict):
        raise ValueError(f"evaluation {cell.cell_id} omits target-access receipts")
    _verify_record(target_access, "evaluation target access")
    _verify_record(ground_partition, "evaluation ground partition")
    for digest_field in (
        "budget",
        "selector",
        "baseline_controller_registry",
        "evaluation_implementation_sha256",
        "quality_preflight_receipt_sha256",
        "quality_preflight_record_digest",
    ):
        _digest(matched.get(digest_field), f"matched protocol {digest_field}")
    if not isinstance(matched.get("proposal"), str) or not matched["proposal"]:
        raise ValueError("matched proposal identity must be nonempty")
    if not isinstance(matched.get("support"), str) or not matched["support"]:
        raise ValueError("matched support identity must be nonempty")
    if not isinstance(matched.get("initializer"), str) or not matched["initializer"]:
        raise ValueError("matched initializer identity must be nonempty")
    _positive_integer(matched.get("inference_threads"), "inference thread count")
    for field in ("inference_device_type", "inference_device_name"):
        if not isinstance(matched.get(field), str) or not matched[field]:
            raise ValueError(f"matched {field} identity must be nonempty")
    _validate_runtime(matched.get("runtime_platform"))
    runtime_registry = report.get("runtime_implementation_registry")
    runtime_digest = _digest(
        report.get("runtime_implementation_digest"), "runtime implementation digest"
    )
    if (
        not isinstance(runtime_registry, dict)
        or content_digest(runtime_registry) != runtime_digest
        or matched.get("runtime_implementation_registry") != runtime_registry
        or matched.get("runtime_implementation_digest") != runtime_digest
    ):
        raise ValueError(f"evaluation {cell.cell_id} has an inconsistent runtime registry")
    if (
        report.get("inference_device_type") != matched.get("inference_device_type")
        or report.get("inference_device_name") != matched.get("inference_device_name")
        or report.get("runtime_platform") != matched.get("runtime_platform")
    ):
        raise ValueError("report and matched runtime identities disagree")

    policy_method = methods["policy"]
    reference_method = methods["return_initial"]
    if (
        policy_method.get("method_id") != f"{cell.model_family}:{cell.method}"
        or policy_method.get("checkpoint_payload_digest") != checkpoint_digest
        or policy_method.get("selection_rule") != protocol["deployment_rule"]
        or policy_method.get("support_contract") != "environment-materialised-masked-actions"
        or policy_method.get("online_evaluator_feedback") is not False
        or policy_method.get("evaluator_oracle") is not False
    ):
        raise ValueError(f"policy method identity differs for {cell.cell_id}")
    if (
        reference_method.get("method_id") != "return-authenticated-initial-v1"
        or reference_method.get("support_contract") != "environment-materialised-masked-actions"
        or reference_method.get("online_evaluator_feedback") is not False
        or reference_method.get("evaluator_oracle") is not False
    ):
        raise ValueError(f"return_initial method identity differs for {cell.cell_id}")

    expected_count = population * repetitions
    loaded: dict[str, tuple[EpisodeOutcome, ...]] = {}
    for arm in required_arms:
        receipt = arm_receipts[arm]
        if set(receipt) != {"path", "sha256", "count", "method", "protocol"}:
            raise ValueError(f"{arm} receipt has an invalid schema in {cell.cell_id}")
        if receipt.get("method") != methods[arm] or receipt.get("protocol") != matched:
            raise ValueError(f"{arm} method/protocol does not match its report")
        if receipt.get("count") != expected_count:
            raise ValueError(f"{arm} count differs from population times repetitions")
        receipt_sha = _digest(receipt.get("sha256"), f"{arm} receipt digest")
        outcomes = tuple(
            read_episode_receipts(
                _arm_path(report_path, receipt.get("path"), arm),
                expected_sha256=receipt_sha,
            )
        )
        if len(outcomes) != expected_count:
            raise ValueError(f"raw {arm} count differs for {cell.cell_id}")
        if report.get(arm) != secondary_metrics(outcomes):
            raise ValueError(f"{arm} summary differs from raw outcomes in {cell.cell_id}")
        identities = {(row.lineage, row.instance) for row in outcomes}
        if len(identities) != population:
            raise ValueError(f"{arm} population denominator differs for {cell.cell_id}")
        for identity in identities:
            observed_repetitions = sorted(
                row.repetition for row in outcomes if (row.lineage, row.instance) == identity
            )
            if observed_repetitions != list(range(repetitions)):
                raise ValueError(f"{arm} has an incomplete repetition census")
        loaded[arm] = outcomes

    if not is_rl_value_cell:  # pragma: no cover - this loader is RL-value only
        raise ValueError("complete receipt validation is restricted to RL-value cells")
    assert isinstance(bootstrap_access, dict)
    assert isinstance(support_digest, str)
    complete_registry = report.get("complete_system_receipts")
    if not isinstance(complete_registry, dict) or set(complete_registry) != set(
        required_arms
    ):
        raise ValueError(
            f"evaluation {cell.cell_id} lacks complete receipts for every same-support arm"
        )
    complete_by_arm: dict[str, dict[tuple[str, str, int], object]] = {}
    for arm in required_arms:
        metadata = complete_registry.get(arm)
        expected_path = f"{arm}.complete.jsonl"
        if (
            not isinstance(metadata, dict)
            or set(metadata) != {"path", "sha256", "count"}
            or metadata.get("path") != expected_path
            or metadata.get("count") != expected_count
        ):
            raise ValueError(f"evaluation {cell.cell_id} has invalid {arm} complete metadata")
        receipt_sha = _digest(
            metadata.get("sha256"), f"{arm} complete receipt digest"
        )
        complete = _read_complete_receipts_for_freeze(
            _arm_path(report_path, expected_path, arm),
            expected_sha256=receipt_sha,
        )
        complete_pairs = {
            getattr(receipt, "outcome").pair_key: receipt for receipt in complete
        }
        ordinary_pairs = {row.pair_key: row for row in loaded[arm]}
        if (
            len(complete) != expected_count
            or len(complete_pairs) != expected_count
            or set(complete_pairs) != set(ordinary_pairs)
        ):
            raise ValueError(f"evaluation {cell.cell_id} changes {arm} complete census")
        for pair_key, ordinary in ordinary_pairs.items():
            receipt = complete_pairs[pair_key]
            complete_outcome = getattr(receipt, "outcome", None)
            binding = getattr(receipt, "bootstrap_binding", None)
            clone = binding.get("clone") if isinstance(binding, Mapping) else None
            if (
                not isinstance(complete_outcome, EpisodeOutcome)
                or complete_outcome.as_dict() != ordinary.as_dict()
            ):
                raise ValueError(
                    f"evaluation {cell.cell_id} changes {arm} outcome across receipt layers"
                )
            if (
                not isinstance(clone, Mapping)
                or clone.get("consumer_id") != f"rl-value/{cell.cell_id}/{arm}"
                or clone.get("bank_access_record_digest")
                != bootstrap_access["record_digest"]
                or clone.get("same_support_contract_digest") != support_digest
            ):
                raise ValueError(
                    f"evaluation {cell.cell_id} has an invalid {arm} bootstrap clone"
                )
        complete_by_arm[arm] = complete_pairs

    bootstrap_identity_fields = (
        "bank_access_record_digest",
        "same_support_contract_digest",
        "bootstrap_record_digest",
        "bootstrap_outcome_record_digest",
        "bootstrap_payload_sha256",
        "row_key",
    )
    for pair_key in sorted(complete_by_arm["policy"]):
        identities = []
        for arm in required_arms:
            receipt = complete_by_arm[arm][pair_key]
            binding = getattr(receipt, "bootstrap_binding")
            clone = binding["clone"]
            identities.append(
                tuple(clone[field] for field in bootstrap_identity_fields)
            )
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError(
                f"evaluation {cell.cell_id} changes bootstrap support across arms "
                f"at {pair_key!r}"
            )

    policy_by_pair = {row.pair_key: row for row in loaded["policy"]}
    reference_by_pair = {row.pair_key: row for row in loaded["return_initial"]}
    if set(policy_by_pair) != set(reference_by_pair):
        raise ValueError(f"policy and return_initial do not share exact pairs in {cell.cell_id}")
    for key in sorted(policy_by_pair):
        policy_row = policy_by_pair[key]
        reference_row = reference_by_pair[key]
        if (
            policy_row.population_eligible != reference_row.population_eligible
            or policy_row.episode_seed != reference_row.episode_seed
        ):
            raise ValueError(f"population/search seed mismatch at {cell.cell_id}:{key!r}")
        if (
            not is_rl_value_cell
            and reference_row.population_eligible
            and not reference_row.returned_valid
        ):
            raise ValueError("return_initial violated protected Profile-I feasibility")
        if (
            policy_row.returned_valid
            and reference_row.returned_valid
            and policy_row.evaluator_seed != reference_row.evaluator_seed
        ):
            raise ValueError(f"evaluator seed mismatch at {cell.cell_id}:{key!r}")
    source = {
        "cell_id": cell.cell_id,
        "grid_model_family": cell.grid_family,
        "model_family": cell.model_family,
        "method": cell.method,
        "training_seed": cell.seed,
        "report_path": f"{cell.cell_id}/report.json",
        "report_sha256": _sha256_file(report_path),
        "report_record_digest": report_record_digest,
        "checkpoint": {"payload_digest": checkpoint_digest},
        "runtime_implementation_registry": runtime_registry,
        "runtime_implementation_digest": runtime_digest,
        "representation_selection_sha256": representation_receipt_sha256,
        "representation_selection_record_digest": representation_record_digest,
        "policy_receipt_sha256": arm_receipts["policy"]["sha256"],
        "return_initial_receipt_sha256": arm_receipts["return_initial"]["sha256"],
        "runtime_identity": _split_protocol_identity(matched)[1],
    }
    return _LoadedCell(
        cell=cell,
        report=report,
        policy=loaded["policy"],
        reference=loaded["return_initial"],
        source=source,
    )


def _unconditional_utility(row: EpisodeOutcome) -> float:
    if not row.population_eligible:
        return 0.0
    if row.utility is None:
        raise ValueError("eligible outcome is missing unconditional utility")
    return float(row.utility)


def _lineage_statistics(
    policy: Sequence[EpisodeOutcome], reference: Sequence[EpisodeOutcome]
) -> dict[str, tuple[float, float, float, float]]:
    policy_by_lineage: dict[str, list[EpisodeOutcome]] = {}
    reference_by_lineage: dict[str, list[EpisodeOutcome]] = {}
    for row in policy:
        policy_by_lineage.setdefault(row.lineage, []).append(row)
    for row in reference:
        reference_by_lineage.setdefault(row.lineage, []).append(row)
    if set(policy_by_lineage) != set(reference_by_lineage):
        raise ValueError("policy and return_initial do not share immutable base lineages")
    result: dict[str, tuple[float, float, float, float]] = {}
    for lineage in sorted(policy_by_lineage):
        policy_rows = policy_by_lineage[lineage]
        reference_rows = reference_by_lineage[lineage]
        online = [row.online_seconds for row in policy_rows]
        if any(value is None or not math.isfinite(value) or value < 0.0 for value in online):
            raise ValueError("RL-value selection requires complete nonnegative online timing")
        policy_feasibility = float(np.mean([float(row.returned_valid) for row in policy_rows]))
        reference_feasibility = float(
            np.mean([float(row.returned_valid) for row in reference_rows])
        )
        result[lineage] = (
            float(np.mean([_unconditional_utility(row) for row in policy_rows])),
            policy_feasibility,
            reference_feasibility,
            float(np.mean([float(value) for value in online])),
        )
    return result


def crossed_bootstrap_bounds(
    values: np.ndarray, *, replicates: int, seed: int, alpha: float
) -> tuple[float, float]:
    """Return percentile bounds after resampling fitted seeds and base lineages.

    The two axes are independent experimental units.  Treating the fitted policies as fixed
    would understate representation-selection uncertainty, especially with only three seeds.
    """

    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("bootstrap requires a training-seed by lineage matrix")
    rng = np.random.Generator(np.random.PCG64(seed))
    seed_count, lineage_count = values.shape
    draws = np.empty(replicates, dtype=float)
    # Bound memory at final benchmark scale while preserving one deterministic RNG stream.
    # Resampling both independent axes avoids treating three fitted policies as fixed.
    batch_size = 256
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        count = stop - start
        sampled_seeds = rng.integers(0, seed_count, size=(count, seed_count))
        sampled_lineages = rng.integers(0, lineage_count, size=(count, lineage_count))
        for offset in range(count):
            draws[start + offset] = float(
                np.mean(values[np.ix_(sampled_seeds[offset], sampled_lineages[offset])])
            )
    return (
        float(np.percentile(draws, 100.0 * alpha, method="linear")),
        float(np.percentile(draws, 100.0 * (1.0 - alpha), method="linear")),
    )


def _aggregate_configurations(
    loaded: Sequence[_LoadedCell], protocol: Mapping[str, object]
) -> tuple[dict[str, dict[str, object]], dict[str, int]]:
    grouped: dict[tuple[str, str], list[_LoadedCell]] = {}
    registered_order: dict[str, int] = {}
    for item in loaded:
        grouped.setdefault((item.cell.model_family, item.cell.method), []).append(item)
        key = f"{item.cell.model_family}/{item.cell.method}"
        registered_order[key] = min(
            registered_order.get(key, item.cell.registered_order), item.cell.registered_order
        )
    if len(grouped) != 6:
        raise ValueError("resolved RL-value census does not contain six configurations")
    margin = float(protocol["margin"])
    alpha = float(protocol["feasibility_alpha"])
    replicates = int(protocol["feasibility_bootstrap"])
    bootstrap_seed = int(protocol["selection_bootstrap_seed"])
    aggregates: dict[str, dict[str, object]] = {}
    common_lineages: tuple[str, ...] | None = None
    for (family, method), runs in grouped.items():
        ordered_runs = sorted(runs, key=lambda item: item.cell.seed)
        seeds = [item.cell.seed for item in ordered_runs]
        if len(ordered_runs) != 3 or len(set(seeds)) != 3:
            raise ValueError(f"configuration {family}/{method} lacks all three training seeds")
        per_seed_lineage = [
            _lineage_statistics(item.policy, item.reference) for item in ordered_runs
        ]
        lineages = tuple(sorted(per_seed_lineage[0]))
        if any(tuple(sorted(rows)) != lineages for rows in per_seed_lineage[1:]):
            raise ValueError(f"training seeds for {family}/{method} use different lineages")
        if common_lineages is None:
            common_lineages = lineages
        elif lineages != common_lineages:
            raise ValueError("RL-value configurations do not share immutable base lineages")
        seed_lineage_values = np.asarray(
            [[rows[lineage] for lineage in lineages] for rows in per_seed_lineage],
            dtype=float,
        )
        lineage_values = np.asarray(
            [np.mean([rows[lineage] for rows in per_seed_lineage], axis=0) for lineage in lineages],
            dtype=float,
        )
        feasibility_delta_by_seed_lineage = (
            seed_lineage_values[:, :, 1] - seed_lineage_values[:, :, 2]
        )
        feasibility_delta = lineage_values[:, 1] - lineage_values[:, 2]
        low, high = crossed_bootstrap_bounds(
            feasibility_delta_by_seed_lineage,
            replicates=replicates,
            seed=bootstrap_seed,
            alpha=alpha,
        )
        per_seed: list[dict[str, object]] = []
        for item, rows in zip(ordered_runs, per_seed_lineage, strict=True):
            values = np.asarray([rows[lineage] for lineage in lineages], dtype=float)
            per_seed.append(
                {
                    "cell_id": item.cell.cell_id,
                    "training_seed": item.cell.seed,
                    "checkpoint_payload_digest": item.report["checkpoint_payload_digest"],
                    "unconditional_utility_mean": float(np.mean(values[:, 0])),
                    "valid_return_rate": float(np.mean(values[:, 1])),
                    "return_initial_valid_rate": float(np.mean(values[:, 2])),
                    "feasibility_delta_vs_return_initial": float(
                        np.mean(values[:, 1] - values[:, 2])
                    ),
                    "online_seconds_mean": float(np.mean(values[:, 3])),
                }
            )
        key = f"{family}/{method}"
        aggregates[key] = {
            "model_family": family,
            "grid_model_family": ordered_runs[0].cell.grid_family,
            "method": method,
            "seed_count": 3,
            "training_seeds": seeds,
            "cell_ids": [item.cell.cell_id for item in ordered_runs],
            "independent_lineages": len(lineages),
            "pairs_per_seed": len(ordered_runs[0].policy),
            "unconditional_utility_mean": float(np.mean(lineage_values[:, 0])),
            "valid_return_rate": float(np.mean(lineage_values[:, 1])),
            "return_initial_valid_rate": float(np.mean(lineage_values[:, 2])),
            "feasibility_delta_vs_return_initial": float(np.mean(feasibility_delta)),
            "feasibility_ci_low": low,
            "feasibility_ci_high": high,
            "feasibility_noninferior": bool(low > -margin),
            "online_seconds_mean": float(np.mean(lineage_values[:, 3])),
            "per_seed": per_seed,
        }
    return aggregates, registered_order


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"RL-value freeze receipt already exists: {path}")
    content = canonical_json_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(f"RL-value freeze receipt already exists: {path}") from None
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def freeze_rl_value_experiment(
    *,
    grid_path: str | os.PathLike[str],
    representation_receipt_path: str | os.PathLike[str],
    evaluations_root: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
) -> dict[str, object]:
    """Authenticate validation evidence and atomically freeze one RL-value configuration.

    The returned configuration is a three-seed training procedure.  All eighteen checkpoints
    remain in the receipt; this function never chooses a checkpoint or reads a test artifact.
    """

    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(f"RL-value freeze receipt already exists: {destination}")
    grid_file = Path(grid_path)
    representation_file = Path(representation_receipt_path)
    evaluation_root = Path(evaluations_root)
    grid, grid_digest, protocol = _load_grid(grid_file)
    selected_simpler, representation_source = _load_representation_receipt(
        representation_file,
        grid=grid,
        grid_digest=grid_digest,
    )
    frozen_runtime_registry = representation_source["runtime_implementation_registry"]
    frozen_runtime_digest = representation_source["runtime_implementation_digest"]
    cells = _resolved_cells(grid, selected_simpler)
    loaded: list[_LoadedCell] = []
    protocol_identity: Mapping[str, Any] | None = None
    runtime_identity_by_seed: dict[str, dict[str, Any]] = {}
    population_identity: tuple[tuple[object, ...], ...] | None = None
    reference_identity: tuple[tuple[object, ...], ...] | None = None
    for cell in cells:
        item = _load_evaluation(
            evaluation_root / cell.cell_id / "report.json",
            cell=cell,
            grid_digest=grid_digest,
            protocol=protocol,
            representation_receipt_sha256=str(representation_source["sha256"]),
            representation_record_digest=str(representation_source["record_digest"]),
        )
        if (
            item.report.get("runtime_implementation_registry") != frozen_runtime_registry
            or item.report.get("runtime_implementation_digest") != frozen_runtime_digest
        ):
            raise ValueError(
                "all 27 validation cells must use byte-identical model/env/PPO sources "
                "and dependency versions"
            )
        matched = item.report["matched_metadata"]
        scientific_identity, runtime_identity = _split_protocol_identity(matched)
        if protocol_identity is None:
            protocol_identity = scientific_identity
        elif scientific_identity != protocol_identity:
            raise ValueError(
                f"evaluation {cell.cell_id} does not match the common scientific protocol"
            )
        seed_key = str(cell.seed)
        registered_runtime = runtime_identity_by_seed.get(seed_key)
        if registered_runtime is None:
            runtime_identity_by_seed[seed_key] = runtime_identity
        elif runtime_identity != registered_runtime:
            raise ValueError(
                f"evaluation {cell.cell_id} changes runtime identity within training seed "
                f"{cell.seed}"
            )
        current_population = _population_signature(item.policy)
        if population_identity is None:
            population_identity = current_population
        elif current_population != population_identity:
            raise ValueError(
                f"evaluation {cell.cell_id} does not share exact population/search pairs"
            )
        current_reference = reference_outcome_signature(item.reference)
        if reference_identity is None:
            reference_identity = current_reference
        elif current_reference != reference_identity:
            raise ValueError(
                f"evaluation {cell.cell_id} does not share the same return_initial outcomes"
            )
        loaded.append(item)
    if len(loaded) != 18 or protocol_identity is None or population_identity is None:
        raise ValueError("RL-value validation census is incomplete")

    aggregates, registered_order = _aggregate_configurations(loaded, protocol)
    eligible = sorted(
        (
            key
            for key, aggregate in aggregates.items()
            if aggregate["feasibility_noninferior"] is True
        ),
        key=registered_order.__getitem__,
    )
    if not eligible:
        raise ValueError("no RL-value configuration passes the feasibility constraint")
    selected_key = max(
        eligible,
        key=lambda key: (
            float(aggregates[key]["unconditional_utility_mean"]),
            -float(aggregates[key]["online_seconds_mean"]),
            -registered_order[key],
        ),
    )
    selected = aggregates[selected_key]
    source_reports = sorted(
        (dict(item.source) for item in loaded),
        key=lambda row: int(
            next(cell.registered_order for cell in cells if cell.cell_id == row["cell_id"])
        ),
    )
    bootstrap_protocol = {
        "purpose": "one-sided paired feasibility lower bound versus return_initial",
        "cluster_unit": "immutable-base-lineage",
        "training_seed_treatment": (
            "equal-weight point aggregate; independently resample all registered training "
            "seeds with replacement in every bootstrap replicate"
        ),
        "within_lineage_treatment": "mean all registered instances and repetitions",
        "sampling": (
            "equal-weight crossed nonparametric bootstrap of training seeds and immutable "
            "base lineages with replacement"
        ),
        "rng": "numpy.random.Generator(PCG64)",
        "numpy_version": np.__version__,
        "replicates": int(protocol["feasibility_bootstrap"]),
        "seed": int(protocol["selection_bootstrap_seed"]),
        "one_sided_alpha": float(protocol["feasibility_alpha"]),
        "quantile": "numpy.percentile-linear",
        "pass_rule": "lower_bound_strictly_greater_than_negative_margin",
    }
    payload: dict[str, object] = {
        "schema": RL_VALUE_FREEZE_SCHEMA,
        "schema_version": RL_VALUE_FREEZE_VERSION,
        "stage": "rl_value",
        "partition": "validation",
        "sealed_test_opened": False,
        "grid_manifest_path": str(grid_file),
        "grid_manifest_sha256": grid_digest,
        "evaluations_root": str(evaluation_root),
        "representation_selection": representation_source,
        "resolved_selected_simpler": selected_simpler,
        "resolved_model_families": [selected_simpler, "if-core"],
        "registered_methods": list(_METHODS),
        "registered_training_seeds": sorted({cell.seed for cell in cells}),
        "registered_cell_count": 18,
        "authenticated_cell_count": len(loaded),
        "seed_selection_forbidden": True,
        "selection_unit": "three-seed-training-configuration-not-checkpoint",
        "selection_rule": RL_VALUE_SELECTION_RULE,
        "primary_endpoint": _PRIMARY_METRIC,
        "aggregation": _AGGREGATION,
        "population_convention": {
            "ordinary_no_valid_return_utility": 0.0,
            "initializer_failure_system_utility": 0.0,
            "raw_initializer_failure_utility_remains_null": True,
            "feasibility_denominator": "all registered validation attempts",
        },
        "feasibility_constraint": {
            "reference": "return_initial",
            "margin": float(protocol["margin"]),
            "one_sided_alpha": float(protocol["feasibility_alpha"]),
            "eligible_configurations": eligible,
        },
        "online_cost_comparability": {
            "required": True,
            "blocking_unit": "training-seed",
            "inference_device_type": protocol_identity["inference_device_type"],
            "runtime_identity_by_training_seed": runtime_identity_by_seed,
        },
        "matched_validation_protocol": protocol_identity,
        "runtime_identity_by_training_seed": runtime_identity_by_seed,
        "runtime_implementation_registry": frozen_runtime_registry,
        "runtime_implementation_digest": frozen_runtime_digest,
        "quality_preflight_receipt_sha256": representation_source[
            "quality_preflight_receipt_sha256"
        ],
        "quality_preflight_record_digest": representation_source[
            "quality_preflight_record_digest"
        ],
        "population_census": {
            "pair_count": len(population_identity),
            "independent_lineages": len({str(row[0]) for row in population_identity}),
            "pair_census_digest": content_digest(list(population_identity)),
        },
        "bootstrap_protocol": bootstrap_protocol,
        "selected_configuration": {
            "model_family": selected["model_family"],
            "grid_model_family": selected["grid_model_family"],
            "method": selected["method"],
            "training_seeds": selected["training_seeds"],
            "cell_ids": selected["cell_ids"],
            "checkpoint_payload_digests": [
                row["checkpoint_payload_digest"] for row in selected["per_seed"]
            ],
            "runtime_implementation_digest": frozen_runtime_digest,
            "single_checkpoint_selected": False,
        },
        "configuration_aggregates": dict(
            sorted(aggregates.items(), key=lambda item: registered_order[item[0]])
        ),
        "source_reports": source_reports,
    }
    receipt = {**payload, "record_digest": content_digest(payload)}
    _atomic_json(destination, receipt)
    return receipt


def load_rl_value_freeze(
    *,
    receipt_path: str | os.PathLike[str],
    grid_path: str | os.PathLike[str],
    expected_sha256: str,
) -> FrozenRLValueSelection:
    """Load the validation freeze used for post-selection test confirmation.

    The expected file digest is mandatory.  A self-rehashed JSON object is not an external
    preregistration anchor and is therefore insufficient to open the sealed test path.
    """

    receipt_file = Path(receipt_path)
    asserted_sha = _digest(expected_sha256, "expected RL-value freeze SHA-256")
    observed_sha = _sha256_file(receipt_file)
    if not hmac.compare_digest(asserted_sha, observed_sha):
        raise ValueError("RL-value freeze file differs from its externally pinned SHA-256")
    grid, grid_digest, _ = _load_grid(Path(grid_path))
    receipt = _strict_json(receipt_file, "RL-value freeze receipt")
    record_digest = _verify_record(receipt, "RL-value freeze receipt")
    if (
        receipt.get("schema") != RL_VALUE_FREEZE_SCHEMA
        or receipt.get("schema_version") != RL_VALUE_FREEZE_VERSION
        or receipt.get("stage") != "rl_value"
        or receipt.get("partition") != "validation"
        or receipt.get("sealed_test_opened") is not False
        or receipt.get("grid_manifest_sha256") != grid_digest
        or receipt.get("registered_cell_count") != 18
        or receipt.get("authenticated_cell_count") != 18
        or receipt.get("seed_selection_forbidden") is not True
        or receipt.get("selection_unit") != "three-seed-training-configuration-not-checkpoint"
        or receipt.get("selection_rule") != RL_VALUE_SELECTION_RULE
        or receipt.get("aggregation") != _AGGREGATION
        or receipt.get("primary_endpoint") != _PRIMARY_METRIC
    ):
        raise ValueError("RL-value freeze has an incompatible scientific identity")

    selected_simpler = receipt.get("resolved_selected_simpler")
    if selected_simpler not in {"if-mlp", "if-dual"}:
        raise ValueError("RL-value freeze has no registered simpler-family resolution")
    if receipt.get("resolved_model_families") != [selected_simpler, "if-core"]:
        raise ValueError("RL-value freeze changes the resolved family registry")
    pinned_runtime_registry = receipt.get("runtime_implementation_registry")
    pinned_runtime_digest = receipt.get("runtime_implementation_digest")
    if (
        not isinstance(pinned_runtime_registry, dict)
        or pinned_runtime_digest != content_digest(pinned_runtime_registry)
    ):
        raise ValueError("RL-value freeze has an inconsistent runtime implementation registry")
    representation = receipt.get("representation_selection")
    if not isinstance(representation, dict):
        raise ValueError("RL-value freeze has no representation-selection binding")
    representation_sha = _digest(
        representation.get("sha256"), "representation-selection SHA-256"
    )
    representation_record = _digest(
        representation.get("record_digest"), "representation-selection record digest"
    )
    if representation.get("selected_simpler") != selected_simpler:
        raise ValueError("RL-value and representation freezes disagree on the simpler family")
    quality_preflight_sha = _digest(
        receipt.get("quality_preflight_receipt_sha256"),
        "quality-preflight receipt SHA-256",
    )
    quality_preflight_record = _digest(
        receipt.get("quality_preflight_record_digest"),
        "quality-preflight record digest",
    )
    if (
        representation.get("runtime_implementation_registry") != pinned_runtime_registry
        or representation.get("runtime_implementation_digest") != pinned_runtime_digest
        or representation.get("quality_preflight_receipt_sha256") != quality_preflight_sha
        or representation.get("quality_preflight_record_digest") != quality_preflight_record
    ):
        raise ValueError(
            "RL-value freeze and representation freeze use different runtime or quality pins"
        )

    resolved = _resolved_cells(grid, str(selected_simpler))
    scientific_protocol = _validate_scientific_protocol_identity(
        receipt.get("matched_validation_protocol")
    )
    if not _RL_BOOTSTRAP_PROTOCOL_FIELDS.issubset(scientific_protocol):
        raise ValueError("RL-value freeze omits its persistent K=2 bootstrap-bank contract")
    registered_seeds = sorted({cell.seed for cell in resolved})
    runtime_by_seed = _validate_runtime_identity_by_seed(
        receipt.get("runtime_identity_by_training_seed"),
        expected_seeds=registered_seeds,
    )
    expected_cost_comparability = {
        "required": True,
        "blocking_unit": "training-seed",
        "inference_device_type": scientific_protocol["inference_device_type"],
        "runtime_identity_by_training_seed": runtime_by_seed,
    }
    if receipt.get("online_cost_comparability") != expected_cost_comparability:
        raise ValueError("RL-value freeze has an invalid seed-blocked runtime protocol")
    source_reports = receipt.get("source_reports")
    if not isinstance(source_reports, list) or len(source_reports) != len(resolved):
        raise ValueError("RL-value freeze has an incomplete source-report census")
    expected_cells = {cell.cell_id: cell for cell in resolved}
    observed_cells: dict[str, Mapping[str, object]] = {}
    for source in source_reports:
        if not isinstance(source, dict):
            raise ValueError("RL-value freeze source report is malformed")
        cell_id = source.get("cell_id")
        if not isinstance(cell_id, str) or cell_id in observed_cells or cell_id not in expected_cells:
            raise ValueError("RL-value freeze source report has a duplicate or unknown cell")
        cell = expected_cells[cell_id]
        if (
            source.get("grid_model_family") != cell.grid_family
            or source.get("model_family") != cell.model_family
            or source.get("method") != cell.method
            or source.get("training_seed") != cell.seed
            or source.get("representation_selection_sha256") != representation_sha
            or source.get("representation_selection_record_digest") != representation_record
        ):
            raise ValueError(f"RL-value freeze source identity differs for {cell_id}")
        for field in (
            "report_sha256",
            "report_record_digest",
            "policy_receipt_sha256",
            "return_initial_receipt_sha256",
        ):
            _digest(source.get(field), f"RL-value source {field}")
        checkpoint = source.get("checkpoint")
        if not isinstance(checkpoint, dict) or set(checkpoint) != {"payload_digest"}:
            raise ValueError("RL-value source checkpoint identity is malformed")
        _digest(checkpoint.get("payload_digest"), "RL-value source checkpoint payload")
        if (
            source.get("runtime_implementation_registry") != pinned_runtime_registry
            or source.get("runtime_implementation_digest") != pinned_runtime_digest
        ):
            raise ValueError(
                "all 27 validation cells require byte-identical source and dependency versions"
            )
        if (
            _validate_runtime_identity(source.get("runtime_identity"))
            != runtime_by_seed[str(cell.seed)]
        ):
            raise ValueError(f"RL-value source runtime differs within seed {cell.seed}")
        observed_cells[cell_id] = source
    if set(observed_cells) != set(expected_cells):
        raise ValueError("RL-value freeze source census differs from the registered grid")

    aggregates = receipt.get("configuration_aggregates")
    expected_keys = {
        f"{family}/{method}"
        for family in (str(selected_simpler), "if-core")
        for method in _METHODS
    }
    if not isinstance(aggregates, dict) or set(aggregates) != expected_keys:
        raise ValueError("RL-value freeze has an incomplete configuration aggregate")
    margin = float(grid["rl_value_evaluation"]["margin"])
    eligible: list[str] = []
    registered_order = {
        f"{cell.model_family}/{cell.method}": min(
            candidate.registered_order
            for candidate in resolved
            if candidate.model_family == cell.model_family and candidate.method == cell.method
        )
        for cell in resolved
    }
    for key, aggregate in aggregates.items():
        if not isinstance(aggregate, dict):
            raise ValueError(f"RL-value aggregate {key} is malformed")
        family, method = key.split("/", 1)
        cells = sorted(
            (
                cell
                for cell in resolved
                if cell.model_family == family and cell.method == method
            ),
            key=lambda cell: cell.seed,
        )
        if (
            aggregate.get("model_family") != family
            or aggregate.get("method") != method
            or aggregate.get("seed_count") != 3
            or aggregate.get("training_seeds") != [cell.seed for cell in cells]
            or aggregate.get("cell_ids") != [cell.cell_id for cell in cells]
        ):
            raise ValueError(f"RL-value aggregate {key} changes its registered cells")
        utility = _finite(aggregate.get("unconditional_utility_mean"), f"{key} utility")
        online = _finite(aggregate.get("online_seconds_mean"), f"{key} online cost")
        lower = _finite(aggregate.get("feasibility_ci_low"), f"{key} feasibility bound")
        if not 0.0 <= utility <= 1.0 or online < 0.0:
            raise ValueError(f"RL-value aggregate {key} has an invalid metric")
        passed = lower > -margin
        if aggregate.get("feasibility_noninferior") is not passed:
            raise ValueError(f"RL-value aggregate {key} has an inconsistent feasibility gate")
        if passed:
            eligible.append(key)
    eligible.sort(key=registered_order.__getitem__)
    constraint = receipt.get("feasibility_constraint")
    if not isinstance(constraint, dict) or constraint.get("eligible_configurations") != eligible:
        raise ValueError("RL-value freeze has an inconsistent eligible configuration set")
    if not eligible:
        raise ValueError("RL-value freeze contains no eligible configuration")
    expected_selected_key = max(
        eligible,
        key=lambda key: (
            float(aggregates[key]["unconditional_utility_mean"]),
            -float(aggregates[key]["online_seconds_mean"]),
            -registered_order[key],
        ),
    )
    selected = receipt.get("selected_configuration")
    if not isinstance(selected, dict) or set(selected) != {
        "model_family",
        "grid_model_family",
        "method",
        "training_seeds",
        "cell_ids",
        "checkpoint_payload_digests",
        "runtime_implementation_digest",
        "single_checkpoint_selected",
    }:
        raise ValueError("RL-value freeze selected configuration is malformed")
    selected_aggregate = aggregates[expected_selected_key]
    expected_cells_ordered = list(selected_aggregate["cell_ids"])
    expected_checkpoint_digests = [
        observed_cells[cell_id]["checkpoint"]["payload_digest"]
        for cell_id in expected_cells_ordered
    ]
    if (
        selected.get("model_family") != selected_aggregate["model_family"]
        or selected.get("grid_model_family") != selected_aggregate["grid_model_family"]
        or selected.get("method") != selected_aggregate["method"]
        or selected.get("training_seeds") != selected_aggregate["training_seeds"]
        or selected.get("cell_ids") != expected_cells_ordered
        or selected.get("checkpoint_payload_digests") != expected_checkpoint_digests
        or selected.get("runtime_implementation_digest") != pinned_runtime_digest
        or selected.get("single_checkpoint_selected") is not False
    ):
        raise ValueError("RL-value freeze violates its registered configuration decision")

    return FrozenRLValueSelection(
        receipt_sha256=observed_sha,
        record_digest=record_digest,
        grid_manifest_sha256=grid_digest,
        model_family=str(selected["model_family"]),
        grid_model_family=str(selected["grid_model_family"]),
        method=str(selected["method"]),
        training_seeds=tuple(int(value) for value in selected["training_seeds"]),
        cell_ids=tuple(str(value) for value in selected["cell_ids"]),
        checkpoint_payload_digests=tuple(
            _digest(value, "selected checkpoint payload")
            for value in selected["checkpoint_payload_digests"]
        ),
        representation_selection_sha256=representation_sha,
        representation_selection_record_digest=representation_record,
        runtime_implementation_registry=pinned_runtime_registry,
        runtime_implementation_digest=_digest(
            pinned_runtime_digest, "selected runtime implementation digest"
        ),
        quality_preflight_receipt_sha256=_digest(
            receipt.get("quality_preflight_receipt_sha256"),
            "selected quality-preflight receipt SHA-256",
        ),
        quality_preflight_record_digest=_digest(
            receipt.get("quality_preflight_record_digest"),
            "selected quality-preflight record digest",
        ),
    )


__all__ = [
    "FrozenRLValueSelection",
    "RL_VALUE_FREEZE_SCHEMA",
    "RL_VALUE_FREEZE_VERSION",
    "REPRESENTATION_SELECTION_SCHEMA_VERSION",
    "crossed_bootstrap_bounds",
    "freeze_rl_value_experiment",
    "load_rl_value_freeze",
]
