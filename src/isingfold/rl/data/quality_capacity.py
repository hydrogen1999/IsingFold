"""Production-like quality-v7 capacity canary and full-generation gate.

The outcome-blind selection phase replays only public state/action structure and seals v2 public
projection digests.  The execution phase is the explicit train-target boundary: it binds separate
target authority, reproduces the public projections, runs the selected continuation denominator,
serializes real quality-v7 rows, and publishes conservative capacity projections.
"""

from __future__ import annotations

import dataclasses
import hashlib
import hmac
import math
import resource
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from isingfold.rl.contracts import Context, WORK_FIELDS
from isingfold.rl.data.action_certificate import apply_envelope_action
from isingfold.rl.data.import_embedbench import (
    CERTIFIED_REFERENCE_STATUSES,
    canonical_json_bytes,
    content_digest,
)
from isingfold.rl.data.prepared import PreparedTask
import isingfold.rl.data.quality as quality_module
from isingfold.rl.data.quality import (
    CONTINUATION_MAX_STEPS,
    CONTINUATION_POLICY_ID,
    ActionQuality,
    ContinuationResult,
    QualityRecord,
    context_digest as quality_context_digest,
    decision_snapshot,
    deterministic_action_sample,
    random_masked_policy,
    replay_decision_with_action_envelope,
    run_continuation,
    validate_continuation_receipt,
    validate_quality_initializer_bank_contract,
)
from isingfold.rl.data.quality_attestation import quality_target_set_digest
from isingfold.rl.data.quality_public_projection import (
    QUALITY_PUBLIC_PROJECTION_VERSION,
    public_applied_action_projection_digest,
    public_envelope_projection_digest,
)
from isingfold.rl.data.quality_resolution import (
    QualityResolutionError,
    REGISTERED_CANDIDATE_CONTINUATIONS,
)
from isingfold.rl.data.quality_resolution_delta import (
    QualityResolutionExecutionAuthority,
    _context_identity,
    _initializer_bank_identity,
    _plan_identity,
    _publish_file,
    _read_pinned_json,
    _thaw,
    _verify_record,
)
from isingfold.rl.data.quality_resolution_merge import QualityResolutionStudyReceipt
from isingfold.rl.data.quality_resolution_plan import QualityResolutionPlan
from isingfold.rl.env import StrengthSelector
from isingfold.rl.initializer_bank import InitializerSnapshotBank


QUALITY_CAPACITY_BUDGET_SCHEMA = "isingfold.quality-capacity-budget"
QUALITY_CAPACITY_BUDGET_VERSION = 1
QUALITY_CAPACITY_CANARY_SELECTION_SCHEMA = "isingfold.quality-capacity-canary-selection"
QUALITY_CAPACITY_CANARY_SELECTION_VERSION = 2
QUALITY_CAPACITY_CANARY_SCHEMA = "isingfold.quality-capacity-canary"
QUALITY_CAPACITY_CANARY_VERSION = 2
QUALITY_CAPACITY_CANARY_LINEAGES = 16
QUALITY_CAPACITY_SAFETY_FACTOR = 2
QUALITY_CAPACITY_FREE_SPACE_FACTOR = 3
QUALITY_CAPACITY_SELECTION_RULE = "public-max-dimensions-and-host-fault-origin-coverage-v1"
QUALITY_CAPACITY_MEASUREMENT_PROTOCOL = (
    "time.process_time_ns+time.monotonic_ns+resource.getrusage+shutil.disk_usage-v1"
)

_HEX = frozenset("0123456789abcdef")
_BUDGET_FIELDS = {
    "available_workers",
    "budget_id",
    "maximum_artifact_bytes",
    "maximum_cpu_seconds",
    "maximum_elapsed_seconds",
    "minimum_scratch_free_bytes",
    "record_digest",
    "schema",
    "schema_version",
}
_SELECTION_FIELDS = {
    "context",
    "corpus",
    "initializer_bank",
    "population",
    "record_digest",
    "resolution_receipt",
    "rows",
    "schema",
    "schema_version",
    "selection",
    "selector",
    "target_accessed",
    "upper_census",
}
_LINEAGE_FIELDS = {
    "base_lineage",
    "dimensions",
    "record_digest",
    "row_ids",
    "stratum",
    "task_ids",
}
_ROW_FIELDS = {
    "action_dimensions",
    "action_indices",
    "base_lineage",
    "environment_seed",
    "public_applied_action_projection_digests",
    "public_envelope_projection_digest",
    "record_digest",
    "row_id",
    "state_fingerprint",
    "task_id",
    "work_coordinates",
}
_TARGET_ROW_FIELDS = {
    "certificate_digest",
    "evaluator_protocol_digest",
    "instance_id",
    "instance_record_digest",
    "learning_partition",
    "record_digest",
    "reference_energy",
    "reference_status",
    "schema",
    "schema_version",
}
_RUNTIME_IDENTITY_FIELDS = {
    "capacity_module_sha256",
    "continuation_runner",
    "execution_runtime_sha256",
    "host_class",
    "measurement_clock",
    "quality_implementation_contract_digest",
    "quality_module_sha256",
}
_SCALE_DIMENSION_FIELDS = {
    "evaluated_actions",
    "host_edges",
    "host_nodes",
    "legal_actions",
    "logical_edges",
    "logical_nodes",
    "problem_linear_coefficients",
    "problem_quadratic_coefficients",
    "rows",
    "support_actions",
}
_CANARY_FIELDS = {
    "actual",
    "budget",
    "budget_comparisons",
    "capacity_selection",
    "context",
    "corpus",
    "device_parity",
    "implementation",
    "initializer_bank",
    "measurements",
    "pass",
    "projections",
    "record_digest",
    "resolution_receipt",
    "runtime",
    "schema",
    "schema_version",
    "selection",
    "selector",
    "train_authority",
    "upper_census",
}


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise QualityResolutionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _positive(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise QualityResolutionError(f"{label} must be a positive integer")
    return value


def _nonnegative(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise QualityResolutionError(f"{label} must be a nonnegative integer")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise QualityResolutionError(f"{label} must be nonempty text")
    return value


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in sorted(value.items())})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True)
class QualityCapacityBudget:
    """Operator-sealed capacity limits in base units."""

    budget_id: str
    maximum_artifact_bytes: int
    maximum_cpu_seconds: int
    maximum_elapsed_seconds: int
    available_workers: int
    minimum_scratch_free_bytes: int

    def __post_init__(self) -> None:
        _text(self.budget_id, "quality-capacity budget ID")
        for label, value in (
            ("maximum artifact bytes", self.maximum_artifact_bytes),
            ("maximum CPU seconds", self.maximum_cpu_seconds),
            ("maximum elapsed seconds", self.maximum_elapsed_seconds),
            ("available workers", self.available_workers),
            ("minimum scratch free bytes", self.minimum_scratch_free_bytes),
        ):
            _positive(value, f"quality-capacity {label}")

    def payload(self) -> dict[str, object]:
        return {
            "available_workers": self.available_workers,
            "budget_id": self.budget_id,
            "maximum_artifact_bytes": self.maximum_artifact_bytes,
            "maximum_cpu_seconds": self.maximum_cpu_seconds,
            "maximum_elapsed_seconds": self.maximum_elapsed_seconds,
            "minimum_scratch_free_bytes": self.minimum_scratch_free_bytes,
            "schema": QUALITY_CAPACITY_BUDGET_SCHEMA,
            "schema_version": QUALITY_CAPACITY_BUDGET_VERSION,
        }

    @property
    def record_digest(self) -> str:
        return content_digest(self.payload())

    def as_dict(self) -> dict[str, object]:
        return {**self.payload(), "record_digest": self.record_digest}


@dataclass(frozen=True, slots=True)
class QualityCapacityCanarySelection:
    record: Mapping[str, object]
    raw_sha256: str

    def __post_init__(self) -> None:
        _digest(self.raw_sha256, "quality-capacity selection SHA-256")
        object.__setattr__(self, "record", _freeze(self.record))

    def as_dict(self) -> dict[str, Any]:
        value = _thaw(self.record)
        if not isinstance(value, dict):  # pragma: no cover
            raise RuntimeError("quality-capacity selection lost its object shape")
        return value


@dataclass(frozen=True, slots=True)
class QualityCapacityCanary:
    record: Mapping[str, object]
    raw_sha256: str

    def __post_init__(self) -> None:
        _digest(self.raw_sha256, "quality-capacity canary SHA-256")
        object.__setattr__(self, "record", _freeze(self.record))

    def as_dict(self) -> dict[str, Any]:
        value = _thaw(self.record)
        if not isinstance(value, dict):  # pragma: no cover
            raise RuntimeError("quality-capacity canary lost its object shape")
        return value


def publish_quality_capacity_budget(path: str | Path, budget: QualityCapacityBudget) -> str:
    if not isinstance(budget, QualityCapacityBudget):
        raise TypeError("budget must be QualityCapacityBudget")
    return _publish_file(Path(path), budget.as_dict())


def load_quality_capacity_budget(
    path: str | Path, *, expected_budget_sha256: str
) -> QualityCapacityBudget:
    record, _raw, _observed = _read_pinned_json(
        path, expected_budget_sha256, "quality-capacity budget"
    )
    if set(record) != _BUDGET_FIELDS:
        raise QualityResolutionError("quality-capacity budget schema fields differ")
    _verify_record(record, "quality-capacity budget")
    if (
        record.get("schema") != QUALITY_CAPACITY_BUDGET_SCHEMA
        or record.get("schema_version") != QUALITY_CAPACITY_BUDGET_VERSION
    ):
        raise QualityResolutionError("unsupported quality-capacity budget")
    budget = QualityCapacityBudget(
        budget_id=record["budget_id"],  # type: ignore[arg-type]
        maximum_artifact_bytes=record["maximum_artifact_bytes"],  # type: ignore[arg-type]
        maximum_cpu_seconds=record["maximum_cpu_seconds"],  # type: ignore[arg-type]
        maximum_elapsed_seconds=record["maximum_elapsed_seconds"],  # type: ignore[arg-type]
        available_workers=record["available_workers"],  # type: ignore[arg-type]
        minimum_scratch_free_bytes=record["minimum_scratch_free_bytes"],  # type: ignore[arg-type]
    )
    if budget.as_dict() != record:
        raise QualityResolutionError("quality-capacity budget cannot be reconstructed")
    return budget


def _study_authorization(
    study: QualityResolutionStudyReceipt,
    *,
    plan_record: Mapping[str, object],
    plan_sha256: str,
) -> tuple[dict[str, Any], int]:
    if not isinstance(study, QualityResolutionStudyReceipt):
        raise TypeError("resolution_receipt must be QualityResolutionStudyReceipt")
    record = study.as_dict()
    observed_raw = hashlib.sha256(canonical_json_bytes(record) + b"\n").hexdigest()
    if observed_raw != study.raw_sha256:
        raise QualityResolutionError("resolution receipt raw identity cannot be reproduced")
    _verify_record(record, "quality-resolution study receipt")
    if (
        record.get("schema") != "isingfold.quality-continuation-resolution-study"
        or record.get("schema_version") != 1
        or record.get("terminal") is not True
        or record.get("advance") is not True
        or record.get("plan")
        != {"raw_sha256": plan_sha256, "record_digest": plan_record["record_digest"]}
    ):
        raise QualityResolutionError(
            "capacity canary requires the first passing terminal resolution receipt"
        )
    selected = record.get("selected_continuations")
    if type(selected) is not int or selected not in REGISTERED_CANDIDATE_CONTINUATIONS:
        raise QualityResolutionError("capacity canary has no registered continuation count")
    protocol = record.get("production_protocol")
    registered = plan_record.get("config")
    selector = plan_record.get("selector")
    if (
        not isinstance(protocol, Mapping)
        or not isinstance(registered, Mapping)
        or not isinstance(registered.get("registered"), Mapping)
        or not isinstance(selector, Mapping)
    ):
        raise QualityResolutionError("resolution production protocol is malformed")
    config = registered["registered"]
    expected = {
        "continuations": selected,
        "evaluated_actions": config["evaluated_actions"],
        "partition": "train",
        "quality_implementation_contract_digest": plan_record["implementation"]["registry"][
            "quality_implementation_contract_digest"
        ],
        "requested_lineages": 0,
        "reward_reads": config["reward_reads"],
        "seed": config["quality_seed"],
        "selector_device": selector["device"],
        "states_per_lineage_cap": config["states_per_lineage_cap"],
        "tasks_per_lineage_cap": config["tasks_per_lineage_cap"],
    }
    if dict(protocol) != expected:
        raise QualityResolutionError("resolution production protocol differs from its plan")
    return record, selected


def _public_task_map(
    prepared: Sequence[PreparedTask], plan_record: Mapping[str, object]
) -> dict[str, PreparedTask]:
    if isinstance(prepared, (str, bytes)) or not isinstance(prepared, Sequence):
        raise TypeError("public_prepared must be a sequence of PreparedTask values")
    production = plan_record.get("production_plan")
    if not isinstance(production, Mapping) or not isinstance(production.get("lineages"), list):
        raise QualityResolutionError("resolution production census is malformed")
    expected_tasks = {
        str(task_id)
        for lineage in production["lineages"]
        if isinstance(lineage, Mapping)
        for task_id in lineage.get("task_ids", [])
    }
    by_id: dict[str, PreparedTask] = {}
    for item in prepared:
        if not isinstance(item, PreparedTask):
            raise TypeError("public capacity inputs must be PreparedTask values")
        if item.task_id in by_id:
            raise QualityResolutionError("public capacity inputs repeat a task ID")
        if item.partition != "train" or item.task.name != item.task_id:
            raise QualityResolutionError("capacity selection is train-only")
        evaluator_values = (
            item.task.ground_energy,
            item.reference_status,
            item.certificate_digest,
            item.evaluator_protocol_digest,
            item.quality_attestation_digest,
            item.quality_evidence_manifest_digest,
            item.quality_evidence_manifest_sha256,
            item.quality_target_set_digest,
            item.quality_target_count,
        )
        if any(value is not None for value in evaluator_values):
            raise QualityResolutionError("capacity selection received evaluator-target information")
        by_id[item.task_id] = item
    if set(by_id) != expected_tasks:
        raise QualityResolutionError(
            "capacity selection requires the complete all-train public task census"
        )
    return {task_id: by_id[task_id] for task_id in sorted(by_id)}


def _max_work(values: Sequence[Mapping[str, int]]) -> dict[str, int]:
    if not values:
        return {field: 0 for field in WORK_FIELDS}
    return {field: max(int(value[field]) for value in values) for field in WORK_FIELDS}


def _planned_row_projection(
    row: Mapping[str, object],
    task: PreparedTask,
    *,
    context: Context,
    selector: StrengthSelector,
    evaluated_actions: int,
    initializer_bank: InitializerSnapshotBank,
    initializer_bank_manifest_sha256: str,
    allow_test_initializer_bank: bool,
) -> dict[str, object]:
    replayed = replay_decision_with_action_envelope(
        task.task,
        context,
        initializer=None,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=(initializer_bank_manifest_sha256),
        initializer_bank_episode_index=row["initializer_bank_episode_index"],
        allow_test_initializer_bank=allow_test_initializer_bank,
        selector=selector,
        prefix=row["prefix"],  # type: ignore[arg-type]
        seed=row["environment_seed"],  # type: ignore[arg-type]
        reward_reads=context.n_est_reads,
        provenance_fingerprint=row["action_provenance_fingerprint"],  # type: ignore[arg-type]
    )
    if replayed is None:
        raise QualityResolutionError("capacity public row is no longer replayable")
    decision, envelope = replayed
    if (
        decision.state_fingerprint != row.get("state_fingerprint")
        or decision.support_fingerprint != row.get("support_fingerprint")
        or envelope.record_digest != row.get("action_envelope_record_digest")
    ):
        raise QualityResolutionError("capacity public row differs from its resolution plan")
    planned_actions = row.get("actions")
    if not isinstance(planned_actions, list) or not planned_actions:
        raise QualityResolutionError("capacity public row has no planned actions")
    action_sample = deterministic_action_sample(
        decision,
        evaluated_actions=evaluated_actions,
        seed=int(row["environment_seed"]),
    )
    selected = action_sample.action_indices
    if selected != tuple(int(action["action_index"]) for action in planned_actions):
        raise QualityResolutionError("capacity public action subset differs from plan")
    applied: list[dict[str, object]] = []
    for planned in planned_actions:
        action_index = int(planned["action_index"])
        bound = envelope.candidates[action_index]
        successor_digest: str | None = None
        full_successor_digest: str | None = None
        if not bound.opcode.is_terminal:
            successor = apply_envelope_action(envelope, action_index)
            successor.verify_against(envelope)
            full_successor_digest = successor.record_digest
            successor_digest = public_applied_action_projection_digest(successor)
        if (
            bound.payload_key != planned["payload_key"]
            or bound.opcode.value != planned["opcode"]
            or bound.payload_digest != planned["selected_payload_digest"]
            or full_successor_digest != planned["applied_action_record_digest"]
        ):
            raise QualityResolutionError("capacity public action identity differs from plan")
        applied.append({"action_index": action_index, "projection_digest": successor_digest})
    payload: dict[str, object] = {
        "action_dimensions": {
            "legal_actions": sum(decision.legal_mask),
            "selected_actions": len(selected),
            "support_actions": len(envelope.candidates),
        },
        "action_indices": list(selected),
        "base_lineage": row["base_lineage"],
        "environment_seed": row["environment_seed"],
        "public_applied_action_projection_digests": applied,
        "public_envelope_projection_digest": public_envelope_projection_digest(envelope),
        "row_id": row["row_id"],
        "state_fingerprint": row["state_fingerprint"],
        "task_id": row["task_id"],
        "work_coordinates": {
            "candidate_max": _max_work(
                [candidate.work.as_dict() for candidate in decision.candidates]
            ),
            "charged": decision.charged_work_receipt.as_dict(),
            "proposal_max": _max_work(
                [candidate.proposal_work.as_dict() for candidate in decision.candidates]
            ),
        },
    }
    return {**payload, "record_digest": content_digest(payload)}


def _lineage_dimensions(
    lineage: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    tasks: Mapping[str, PreparedTask],
) -> dict[str, object]:
    task_ids = lineage["task_ids"]
    if not isinstance(task_ids, list) or not task_ids:
        raise QualityResolutionError("capacity lineage has no public task")
    selected_tasks = [tasks[str(task_id)] for task_id in task_ids]
    scale = {
        "evaluated_actions": sum(len(row["action_indices"]) for row in rows),
        "host_edges": max(item.task.host.number_of_edges() for item in selected_tasks),
        "host_nodes": max(item.task.host.number_of_nodes() for item in selected_tasks),
        "legal_actions": max(int(row["action_dimensions"]["legal_actions"]) for row in rows),
        "logical_edges": max(item.task.logical.number_of_edges() for item in selected_tasks),
        "logical_nodes": max(item.task.logical.number_of_nodes() for item in selected_tasks),
        "problem_linear_coefficients": max(len(item.task.problem.h) for item in selected_tasks),
        "problem_quadratic_coefficients": max(len(item.task.problem.j) for item in selected_tasks),
        "rows": len(rows),
        "support_actions": max(
            (int(row["action_dimensions"]["support_actions"]) for row in rows),
            default=0,
        ),
    }
    work_maxima: dict[str, dict[str, int]] = {}
    for category in ("charged", "candidate_max", "proposal_max"):
        work_maxima[category] = {
            field: max(
                (int(row["work_coordinates"][category][field]) for row in rows),
                default=0,
            )
            for field in WORK_FIELDS
        }
    return {"scale": scale, "work_coordinate_maxima": work_maxima}


def _flatten_dimensions(dimensions: Mapping[str, object]) -> dict[str, int]:
    scale = dimensions.get("scale")
    work = dimensions.get("work_coordinate_maxima")
    if (
        set(dimensions) != {"scale", "work_coordinate_maxima"}
        or not isinstance(scale, Mapping)
        or set(scale) != _SCALE_DIMENSION_FIELDS
        or not isinstance(work, Mapping)
        or set(work) != {"candidate_max", "charged", "proposal_max"}
    ):
        raise QualityResolutionError("capacity public dimensions are malformed")
    flattened: dict[str, int] = {}
    for key, value in scale.items():
        flattened[f"scale.{key}"] = _nonnegative(value, f"capacity scale dimension {key}")
    for category, values in work.items():
        if not isinstance(values, Mapping) or set(values) != set(WORK_FIELDS):
            raise QualityResolutionError("capacity work dimensions are malformed")
        for key, value in values.items():
            flattened[f"work.{category}.{key}"] = _nonnegative(
                value, f"capacity work dimension {category}.{key}"
            )
    return flattened


def _selection_tokens(
    population: Sequence[Mapping[str, object]],
) -> tuple[dict[str, set[str]], tuple[str, ...]]:
    axes = ("host_family", "fault_status", "problem_origin")
    maxima: dict[str, int] = {}
    for entry in population:
        for name, value in _flatten_dimensions(entry["dimensions"]).items():
            maxima[name] = max(maxima.get(name, value), value)
    token_map: dict[str, set[str]] = {}
    required: set[str] = set()
    for entry in population:
        lineage = str(entry["base_lineage"])
        stratum = entry["stratum"]
        if not isinstance(stratum, Mapping):
            raise QualityResolutionError("capacity lineage stratum is malformed")
        tokens = {f"level:{axis}:{stratum[axis]}" for axis in axes}
        flat = _flatten_dimensions(entry["dimensions"])
        tokens.update(
            f"maximum:{name}:{value}" for name, value in flat.items() if value == maxima[name]
        )
        token_map[lineage] = tokens
        required.update(tokens)
    return token_map, tuple(sorted(required))


def _select_capacity_lineages(
    population: Sequence[Mapping[str, object]],
    *,
    source_manifest_sha256: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if len(population) < QUALITY_CAPACITY_CANARY_LINEAGES:
        raise QualityResolutionError("capacity canary requires at least 16 public lineages")
    token_map, required = _selection_tokens(population)
    dimensions = {
        str(entry["base_lineage"]): _flatten_dimensions(entry["dimensions"]) for entry in population
    }
    dimension_names = sorted(next(iter(dimensions.values())))
    remaining = {str(entry["base_lineage"]): entry for entry in population}
    uncovered = set(required)
    chosen: list[str] = []
    while len(chosen) < QUALITY_CAPACITY_CANARY_LINEAGES:

        def key(lineage: str) -> tuple[object, ...]:
            return (
                -len(token_map[lineage] & uncovered),
                *(-dimensions[lineage][name] for name in dimension_names),
                content_digest(
                    {
                        "domain": QUALITY_CAPACITY_SELECTION_RULE,
                        "lineage": lineage,
                        "source_manifest_sha256": source_manifest_sha256,
                    }
                ),
                lineage,
            )

        selected = min(remaining, key=key)
        chosen.append(selected)
        uncovered -= token_map[selected]
        del remaining[selected]
    if uncovered:
        raise QualityResolutionError(
            "16-lineage capacity canary cannot cover every public maximum and required level"
        )
    return tuple(chosen), required


def build_quality_capacity_canary_selection(
    plan: QualityResolutionPlan,
    *,
    expected_plan_sha256: str,
    resolution_receipt: QualityResolutionStudyReceipt,
    public_prepared: Sequence[PreparedTask],
    context: Context,
    selector: StrengthSelector,
    output_path: str | Path,
    initializer_bank: InitializerSnapshotBank | None = None,
    expected_initializer_bank_manifest_sha256: str | None = None,
    allow_test_initializer_bank: bool = False,
) -> str:
    """Seal the exact outcome-blind 16-lineage selection before target access."""

    plan_record, plan_sha256 = _plan_identity(plan, expected_plan_sha256)
    initializer_bank_contract = _initializer_bank_identity(
        plan_record,
        initializer_bank,
        expected_initializer_bank_manifest_sha256,
        allow_test_bank=allow_test_initializer_bank,
    )
    assert isinstance(initializer_bank, InitializerSnapshotBank)
    study, continuations = _study_authorization(
        resolution_receipt, plan_record=plan_record, plan_sha256=plan_sha256
    )
    _snapshot, context_digest = _context_identity(context, plan_record)
    tasks = _public_task_map(public_prepared, plan_record)
    production = plan_record["production_plan"]
    rows_raw = production["rows"]
    lineages_raw = production["lineages"]
    if not isinstance(rows_raw, list) or not isinstance(lineages_raw, list):
        raise QualityResolutionError("capacity production plan census is malformed")
    evaluated_actions = int(plan_record["config"]["registered"]["evaluated_actions"])
    row_projections = [
        _planned_row_projection(
            row,
            tasks[str(row["task_id"])],
            context=context,
            selector=selector,
            evaluated_actions=evaluated_actions,
            initializer_bank=initializer_bank,
            initializer_bank_manifest_sha256=str(initializer_bank_contract["manifest_sha256"]),
            allow_test_initializer_bank=allow_test_initializer_bank,
        )
        for row in rows_raw
        if isinstance(row, Mapping)
    ]
    by_lineage: dict[str, list[dict[str, object]]] = {}
    for row in row_projections:
        by_lineage.setdefault(str(row["base_lineage"]), []).append(row)
    population: list[dict[str, object]] = []
    for lineage in lineages_raw:
        if not isinstance(lineage, Mapping):
            raise QualityResolutionError("capacity production lineage is malformed")
        lineage_id = str(lineage["base_lineage"])
        rows = sorted(by_lineage.get(lineage_id, []), key=lambda row: str(row["row_id"]))
        dimensions = _lineage_dimensions(lineage, rows, tasks)
        lineage_payload: dict[str, object] = {
            "base_lineage": lineage_id,
            "dimensions": dimensions,
            "row_ids": [row["row_id"] for row in rows],
            "stratum": lineage["stratum"],
            "task_ids": lineage["task_ids"],
        }
        population.append({**lineage_payload, "record_digest": content_digest(lineage_payload)})
    population.sort(key=lambda entry: str(entry["base_lineage"]))
    selected, required_tokens = _select_capacity_lineages(
        population,
        source_manifest_sha256=str(plan_record["prepared_corpus"]["manifest_sha256"]),
    )
    selected_set = set(selected)
    selected_rows = sorted(
        (row for row in row_projections if row["base_lineage"] in selected_set),
        key=lambda row: (
            selected.index(str(row["base_lineage"])),
            str(row["row_id"]),
        ),
    )
    upper_actions = sum(len(row["actions"]) for row in rows_raw)
    upper = {
        "actions": upper_actions,
        "lineages": int(production["lineage_count"]),
        "rows": int(production["row_count"]),
        "trajectories": upper_actions * continuations,
    }
    payload: dict[str, object] = {
        "context": plan_record["context"],
        "corpus": plan_record["prepared_corpus"],
        "initializer_bank": initializer_bank_contract,
        "population": population,
        "resolution_receipt": {
            "raw_sha256": resolution_receipt.raw_sha256,
            "record_digest": study["record_digest"],
            "selected_continuations": continuations,
        },
        "rows": selected_rows,
        "schema": QUALITY_CAPACITY_CANARY_SELECTION_SCHEMA,
        "schema_version": QUALITY_CAPACITY_CANARY_SELECTION_VERSION,
        "selection": {
            "count": QUALITY_CAPACITY_CANARY_LINEAGES,
            "coverage_complete": True,
            "required_coverage_tokens": list(required_tokens),
            "rule": QUALITY_CAPACITY_SELECTION_RULE,
            "selected_lineages": list(selected),
        },
        "selector": plan_record["selector"],
        "target_accessed": False,
        "upper_census": upper,
    }
    selection_record = {**payload, "record_digest": content_digest(payload)}
    return _publish_file(Path(output_path), selection_record)


def _validate_selection_rows(
    rows: object,
    *,
    selected: Sequence[str],
    plan_record: Mapping[str, object],
) -> tuple[int, int]:
    if not isinstance(rows, list):
        raise QualityResolutionError("capacity selection rows must be an array")
    production_rows = plan_record["production_plan"]["rows"]
    if not isinstance(production_rows, list):
        raise QualityResolutionError("capacity production rows are malformed")
    planned = {row["row_id"]: row for row in production_rows if isinstance(row, Mapping)}
    expected_ids = sorted(
        (
            str(row["row_id"])
            for row in production_rows
            if isinstance(row, Mapping) and row.get("base_lineage") in set(selected)
        ),
        key=lambda row_id: (selected.index(str(planned[row_id]["base_lineage"])), row_id),
    )
    observed_ids: list[str] = []
    action_count = 0
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _ROW_FIELDS:
            raise QualityResolutionError("capacity selection row schema differs")
        _verify_record(row, "capacity selection row")
        row_id = row.get("row_id")
        if row_id not in planned:
            raise QualityResolutionError("capacity selection refers to an unknown planned row")
        source = planned[row_id]
        actions = row.get("action_indices")
        action_dimensions = row.get("action_dimensions")
        applied = row.get("public_applied_action_projection_digests")
        work = row.get("work_coordinates")
        if (
            row.get("base_lineage") != source["base_lineage"]
            or row.get("task_id") != source["task_id"]
            or row.get("environment_seed") != source["environment_seed"]
            or row.get("state_fingerprint") != source["state_fingerprint"]
            or not isinstance(action_dimensions, Mapping)
            or set(action_dimensions) != {"legal_actions", "selected_actions", "support_actions"}
            or not isinstance(actions, list)
            or actions != [action["action_index"] for action in source["actions"]]
            or not isinstance(applied, list)
            or [entry.get("action_index") for entry in applied if isinstance(entry, Mapping)]
            != actions
            or not isinstance(work, Mapping)
            or set(work) != {"candidate_max", "charged", "proposal_max"}
        ):
            raise QualityResolutionError("capacity selection row differs from resolution plan")
        for field in ("legal_actions", "selected_actions", "support_actions"):
            _positive(action_dimensions[field], f"capacity row action dimension {field}")
        if action_dimensions["selected_actions"] != len(actions):
            raise QualityResolutionError("capacity selected-action dimension differs")
        _digest(
            row.get("public_envelope_projection_digest"),
            "capacity public-envelope projection digest",
        )
        for entry in applied:
            if not isinstance(entry, Mapping) or set(entry) != {
                "action_index",
                "projection_digest",
            }:
                raise QualityResolutionError("capacity applied-projection schema differs")
            digest = entry["projection_digest"]
            if digest is not None:
                _digest(digest, "capacity public applied-action projection digest")
        for category in ("candidate_max", "charged", "proposal_max"):
            values = work[category]
            if not isinstance(values, Mapping) or set(values) != set(WORK_FIELDS):
                raise QualityResolutionError("capacity row work-coordinate schema differs")
            for field in WORK_FIELDS:
                _nonnegative(values[field], f"capacity row {category} {field}")
        observed_ids.append(str(row_id))
        action_count += len(actions)
    if observed_ids != expected_ids:
        raise QualityResolutionError("capacity selection does not cover every selected planned row")
    return len(rows), action_count


def load_quality_capacity_canary_selection(
    path: str | Path,
    *,
    expected_selection_sha256: str,
    plan: QualityResolutionPlan,
    expected_plan_sha256: str,
    resolution_receipt: QualityResolutionStudyReceipt,
) -> QualityCapacityCanarySelection:
    plan_record, plan_sha256 = _plan_identity(plan, expected_plan_sha256)
    study, continuations = _study_authorization(
        resolution_receipt, plan_record=plan_record, plan_sha256=plan_sha256
    )
    record, _raw, observed_sha = _read_pinned_json(
        path, expected_selection_sha256, "quality-capacity canary selection"
    )
    if set(record) != _SELECTION_FIELDS:
        raise QualityResolutionError("quality-capacity selection schema fields differ")
    _verify_record(record, "quality-capacity canary selection")
    if (
        record.get("schema") != QUALITY_CAPACITY_CANARY_SELECTION_SCHEMA
        or record.get("schema_version") != QUALITY_CAPACITY_CANARY_SELECTION_VERSION
        or record.get("target_accessed") is not False
        or record.get("corpus") != plan_record["prepared_corpus"]
        or record.get("selector") != plan_record["selector"]
        or record.get("context") != plan_record["context"]
        or record.get("initializer_bank") != plan_record["production_plan"]["initializer_bank"]
        or record.get("resolution_receipt")
        != {
            "raw_sha256": resolution_receipt.raw_sha256,
            "record_digest": study["record_digest"],
            "selected_continuations": continuations,
        }
    ):
        raise QualityResolutionError("quality-capacity selection belongs to another study")
    population = record.get("population")
    if not isinstance(population, list):
        raise QualityResolutionError("quality-capacity population must be an array")
    production_lineages = plan_record["production_plan"]["lineages"]
    planned_lineages = {
        str(lineage["base_lineage"]): lineage
        for lineage in production_lineages
        if isinstance(lineage, Mapping)
    }
    expected_lineages = sorted(
        str(lineage["base_lineage"])
        for lineage in production_lineages
        if isinstance(lineage, Mapping)
    )
    observed_lineages: list[str] = []
    for entry in population:
        if not isinstance(entry, Mapping) or set(entry) != _LINEAGE_FIELDS:
            raise QualityResolutionError("quality-capacity population entry schema differs")
        _verify_record(entry, "quality-capacity population entry")
        _flatten_dimensions(entry["dimensions"])
        lineage_id = str(entry["base_lineage"])
        source = planned_lineages.get(lineage_id)
        if source is None:
            raise QualityResolutionError("quality-capacity population has an unknown lineage")
        expected_row_ids = sorted(
            str(row["row_id"])
            for row in plan_record["production_plan"]["rows"]
            if row["base_lineage"] == lineage_id
        )
        if (
            entry["stratum"] != source["stratum"]
            or entry["task_ids"] != source["task_ids"]
            or entry["row_ids"] != expected_row_ids
        ):
            raise QualityResolutionError(
                "quality-capacity population lineage differs from its exact plan"
            )
        observed_lineages.append(lineage_id)
    if observed_lineages != expected_lineages:
        raise QualityResolutionError("quality-capacity population differs from plan")
    selection = record.get("selection")
    if not isinstance(selection, Mapping) or set(selection) != {
        "count",
        "coverage_complete",
        "required_coverage_tokens",
        "rule",
        "selected_lineages",
    }:
        raise QualityResolutionError("quality-capacity selection decision schema differs")
    reproduced, tokens = _select_capacity_lineages(
        population,
        source_manifest_sha256=str(plan_record["prepared_corpus"]["manifest_sha256"]),
    )
    if selection != {
        "count": QUALITY_CAPACITY_CANARY_LINEAGES,
        "coverage_complete": True,
        "required_coverage_tokens": list(tokens),
        "rule": QUALITY_CAPACITY_SELECTION_RULE,
        "selected_lineages": list(reproduced),
    }:
        raise QualityResolutionError("quality-capacity selection cannot be reproduced")
    rows, actions = _validate_selection_rows(
        record.get("rows"), selected=reproduced, plan_record=plan_record
    )
    production_rows = plan_record["production_plan"]["rows"]
    upper_actions = sum(len(row["actions"]) for row in production_rows)
    expected_upper = {
        "actions": upper_actions,
        "lineages": plan_record["production_plan"]["lineage_count"],
        "rows": plan_record["production_plan"]["row_count"],
        "trajectories": upper_actions * continuations,
    }
    if record.get("upper_census") != expected_upper:
        raise QualityResolutionError("quality-capacity upper census differs from exact plan")
    if rows <= 0 or actions <= 0:
        raise QualityResolutionError("quality-capacity selected no replayable work")
    return QualityCapacityCanarySelection(record, observed_sha)


def compute_guarded_capacity_projection(
    *,
    effective_max_bytes_per_trajectory: int,
    max_cpu_time_ns_per_trajectory: int,
    max_wall_time_ns_per_trajectory: int,
    upper_trajectory_census: int,
    scratch_free_bytes: int,
    budget: QualityCapacityBudget,
) -> dict[str, object]:
    """Apply exact integer ceilings and the registered factor-2/factor-3 guards."""

    if not isinstance(budget, QualityCapacityBudget):
        raise TypeError("budget must be QualityCapacityBudget")
    byte_charge = _positive(
        effective_max_bytes_per_trajectory,
        "effective maximum bytes per trajectory",
    )
    cpu_ns = _positive(max_cpu_time_ns_per_trajectory, "maximum CPU nanoseconds")
    wall_ns = _positive(max_wall_time_ns_per_trajectory, "maximum wall nanoseconds")
    trajectories = _positive(upper_trajectory_census, "upper trajectory census")
    scratch = _nonnegative(scratch_free_bytes, "observed scratch free bytes")
    projected_bytes = QUALITY_CAPACITY_SAFETY_FACTOR * byte_charge * trajectories
    projected_cpu_ns = QUALITY_CAPACITY_SAFETY_FACTOR * cpu_ns * trajectories
    projected_cpu_seconds = math.ceil(projected_cpu_ns / 1_000_000_000)
    worker_waves = math.ceil(trajectories / budget.available_workers)
    projected_wall_ns = QUALITY_CAPACITY_SAFETY_FACTOR * wall_ns * worker_waves
    projected_wall_seconds = math.ceil(projected_wall_ns / 1_000_000_000)
    cpu_parallel_seconds = math.ceil(projected_cpu_seconds / budget.available_workers)
    projected_elapsed = max(cpu_parallel_seconds, projected_wall_seconds)
    required_free = QUALITY_CAPACITY_FREE_SPACE_FACTOR * projected_bytes
    comparisons = {
        "artifact_bytes_within_budget": projected_bytes <= budget.maximum_artifact_bytes,
        "cpu_seconds_within_budget": projected_cpu_seconds <= budget.maximum_cpu_seconds,
        "elapsed_seconds_within_budget": projected_elapsed <= budget.maximum_elapsed_seconds,
        "scratch_above_budget_minimum": scratch >= budget.minimum_scratch_free_bytes,
        "scratch_above_guarded_requirement": scratch >= required_free,
    }
    return {
        "comparisons": comparisons,
        "pass": all(comparisons.values()),
        "projected_artifact_bytes": projected_bytes,
        "projected_cpu_nanoseconds": projected_cpu_ns,
        "projected_cpu_seconds": projected_cpu_seconds,
        "projected_elapsed_seconds": projected_elapsed,
        "projected_wall_nanoseconds": projected_wall_ns,
        "projected_wall_seconds": projected_wall_seconds,
        "required_free_bytes": required_free,
        "safety_factor": QUALITY_CAPACITY_SAFETY_FACTOR,
        "scratch_free_bytes": scratch,
        "worker_waves": worker_waves,
    }


def _target_provenance(item: PreparedTask, source_manifest_sha256: str) -> str:
    return content_digest(
        {
            "schema": "isingfold.quality-action-provenance",
            "schema_version": 1,
            "source_corpus_manifest_sha256": source_manifest_sha256,
            "task_id": item.task_id,
            "instance_id": item.instance_id,
            "partition": item.partition,
            "initializer_record_digest": item.initializer_record_digest,
            "certificate_digest": item.certificate_digest,
            "evaluator_protocol_digest": item.evaluator_protocol_digest,
        }
    )


def _module_sha256(module_file: object, label: str) -> str:
    if type(module_file) is not str or not module_file:
        raise QualityResolutionError(f"{label} has no concrete source file")
    source = Path(module_file)
    if source.is_symlink():
        raise QualityResolutionError(f"{label} source may not be a symbolic link")
    path = source.resolve()
    if not path.is_file():
        raise QualityResolutionError(f"{label} source is not a regular file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_runtime_identity(
    runtime_identity: Mapping[str, object],
    runtime_identity_digest: str,
    *,
    plan_record: Mapping[str, object],
    continuation_runner: Callable[..., ContinuationResult | None],
    authority: QualityResolutionExecutionAuthority | None = None,
) -> tuple[dict[str, object], str]:
    if (
        not isinstance(runtime_identity, Mapping)
        or set(runtime_identity) != _RUNTIME_IDENTITY_FIELDS
    ):
        raise QualityResolutionError("quality-capacity runtime identity schema differs")
    runtime = dict(runtime_identity)
    expected_digest = _digest(runtime_identity_digest, "quality-capacity runtime identity digest")
    if not hmac.compare_digest(content_digest(runtime), expected_digest):
        raise QualityResolutionError("quality-capacity runtime identity digest mismatch")
    if continuation_runner is not run_continuation:
        raise QualityResolutionError(
            "quality-capacity publication requires the registered run_continuation callable"
        )
    expected_runner = "isingfold.rl.data.quality.run_continuation"
    if runtime.get("continuation_runner") != expected_runner:
        raise QualityResolutionError("quality-capacity continuation runner identity differs")
    _text(runtime.get("host_class"), "quality-capacity host class")
    if runtime.get("measurement_clock") != QUALITY_CAPACITY_MEASUREMENT_PROTOCOL:
        raise QualityResolutionError("quality-capacity measurement protocol differs")
    _digest(
        runtime.get("execution_runtime_sha256"),
        "quality-capacity execution runtime SHA-256",
    )
    quality_sha = _module_sha256(quality_module.__file__, "quality implementation")
    capacity_sha = _module_sha256(__file__, "quality-capacity implementation")
    implementation = plan_record.get("implementation")
    if not isinstance(implementation, Mapping):
        raise QualityResolutionError("quality-capacity plan implementation is malformed")
    registry = implementation.get("registry")
    if not isinstance(registry, Mapping):
        raise QualityResolutionError("quality-capacity plan implementation registry is malformed")
    contract = _digest(
        registry.get("quality_implementation_contract_digest"),
        "planned quality implementation contract digest",
    )
    planned_quality_sha = _digest(
        registry.get("quality_module_sha256"), "planned quality module SHA-256"
    )
    if (
        runtime.get("quality_module_sha256") != quality_sha
        or runtime.get("quality_module_sha256") != planned_quality_sha
        or runtime.get("capacity_module_sha256") != capacity_sha
        or runtime.get("quality_implementation_contract_digest") != contract
    ):
        raise QualityResolutionError(
            "quality-capacity loaded implementation differs from its declared identities"
        )
    if authority is not None and (
        authority.execution_identity.get("quality_module_sha256") != quality_sha
        or authority.execution_identity.get("quality_implementation_contract_digest") != contract
        or runtime.get("execution_runtime_sha256")
        != authority.execution_identity.get("runtime_sha256")
    ):
        raise QualityResolutionError(
            "quality-capacity loaded implementation differs from execution authority"
        )
    return runtime, expected_digest


def _authenticated_target_rows(
    target_records: Sequence[Mapping[str, object]],
    *,
    authority: QualityResolutionExecutionAuthority,
) -> tuple[dict[str, Mapping[str, object]], dict[str, object]]:
    if (
        isinstance(target_records, (str, bytes))
        or not isinstance(target_records, Sequence)
        or not target_records
    ):
        raise QualityResolutionError(
            "capacity execution requires the complete evaluator-target row census"
        )
    rows: list[Mapping[str, object]] = []
    by_instance: dict[str, Mapping[str, object]] = {}
    for row in target_records:
        if not isinstance(row, Mapping) or set(row) != _TARGET_ROW_FIELDS:
            raise QualityResolutionError("capacity evaluator-target row schema differs")
        _verify_record(row, "capacity evaluator-target row")
        if (
            row.get("schema") != "isingfold.evaluator-target"
            or row.get("schema_version") != 2
            or row.get("learning_partition") != "train"
            or row.get("reference_status") not in CERTIFIED_REFERENCE_STATUSES
        ):
            raise QualityResolutionError("capacity evaluator-target row is not certified train v2")
        instance_id = _text(row.get("instance_id"), "capacity target instance ID")
        if instance_id in by_instance:
            raise QualityResolutionError("capacity evaluator targets repeat an instance ID")
        _digest(row.get("instance_record_digest"), "capacity target instance record digest")
        _digest(row.get("certificate_digest"), "capacity target certificate digest")
        _digest(
            row.get("evaluator_protocol_digest"),
            "capacity target evaluator protocol digest",
        )
        energy = row.get("reference_energy")
        if (
            isinstance(energy, bool)
            or not isinstance(energy, (int, float))
            or not math.isfinite(float(energy))
        ):
            raise QualityResolutionError("capacity target reference energy is not finite")
        by_instance[instance_id] = row
        rows.append(row)
    if [str(row["instance_id"]) for row in rows] != sorted(by_instance):
        raise QualityResolutionError("capacity evaluator-target rows are not in canonical order")
    target = authority.target_access
    if len(rows) != target.get("target_count"):
        raise QualityResolutionError("capacity evaluator-target count differs from authority")
    observed_set = quality_target_set_digest(rows, partition="train")
    if not hmac.compare_digest(observed_set, str(target.get("target_set_digest"))):
        raise QualityResolutionError("capacity evaluator-target set differs from authority")
    raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    observed_sha = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(observed_sha, str(target.get("target_sha256"))):
        raise QualityResolutionError("capacity evaluator-target bytes differ from authority")
    identities = [
        {
            "instance_id": row["instance_id"],
            "target_record_digest": row["record_digest"],
        }
        for row in rows
    ]
    binding = {
        "target_count": len(rows),
        "target_record_set_digest": content_digest(identities),
        "target_set_digest": observed_set,
        "target_sha256": observed_sha,
    }
    return by_instance, binding


def _target_task_map(
    prepared: Sequence[PreparedTask],
    *,
    target_records: Sequence[Mapping[str, object]],
    plan_record: Mapping[str, object],
    authority: QualityResolutionExecutionAuthority,
) -> tuple[dict[str, PreparedTask], dict[str, object]]:
    if isinstance(prepared, (str, bytes)) or not isinstance(prepared, Sequence):
        raise TypeError("capacity target tasks must be a sequence")
    production = plan_record["production_plan"]
    expected = {
        str(task_id) for lineage in production["lineages"] for task_id in lineage["task_ids"]
    }
    planned_identity = {
        str(row["task_id"]): (
            str(row["instance_id"]),
            str(row["base_lineage"]),
        )
        for row in production["rows"]
    }
    targets, target_binding = _authenticated_target_rows(target_records, authority=authority)
    by_id: dict[str, PreparedTask] = {}
    for item in prepared:
        if not isinstance(item, PreparedTask):
            raise TypeError("capacity target tasks must be PreparedTask values")
        if item.task_id in by_id:
            raise QualityResolutionError("capacity target tasks repeat an identity")
        if (
            item.partition != "train"
            or item.task.name != item.task_id
            or item.task.ground_energy is None
            or item.reference_status is None
            or item.certificate_digest is None
            or item.evaluator_protocol_digest is None
            or item.provenance is None
        ):
            raise QualityResolutionError("capacity execution requires target-bearing train tasks")
        _digest(item.certificate_digest, "capacity task certificate digest")
        _digest(item.evaluator_protocol_digest, "capacity evaluator protocol digest")
        metadata = (
            item.quality_attestation_digest,
            item.quality_evidence_manifest_digest,
            item.quality_evidence_manifest_sha256,
            item.quality_target_set_digest,
            item.quality_target_count,
        )
        if any(value is None for value in metadata):
            raise QualityResolutionError(
                "capacity target-bearing task lacks authenticated target metadata"
            )
        target = authority.target_access
        if (
            item.quality_attestation_digest != target["publisher_attestation_digest"]
            or item.quality_evidence_manifest_digest != target["evidence_manifest_record_digest"]
            or item.quality_evidence_manifest_sha256 != target["evidence_manifest_sha256"]
            or item.quality_target_set_digest != target["target_set_digest"]
            or item.quality_target_count != target["target_count"]
        ):
            raise QualityResolutionError("capacity task target metadata differs from authority")
        planned = planned_identity.get(item.task_id)
        target_row = targets.get(item.instance_id)
        if planned != (item.instance_id, str(item.task.lineage)) or target_row is None:
            raise QualityResolutionError("capacity task crosses its planned target identity")
        if (
            item.provenance.instance_record_digest != target_row["instance_record_digest"]
            or float(item.task.ground_energy) != float(target_row["reference_energy"])
            or item.reference_status != target_row["reference_status"]
            or item.certificate_digest != target_row["certificate_digest"]
            or item.evaluator_protocol_digest != target_row["evaluator_protocol_digest"]
        ):
            raise QualityResolutionError(
                "capacity PreparedTask target values differ from authenticated target row"
            )
        by_id[item.task_id] = item
    if set(by_id) != expected:
        raise QualityResolutionError(
            "capacity execution requires the complete authenticated train task census"
        )
    if authority.target_access["target_count"] != len(by_id):
        raise QualityResolutionError("capacity target census differs from target authority")
    if set(targets) != {item.instance_id for item in by_id.values()}:
        raise QualityResolutionError(
            "capacity target rows do not exactly cover the planned train tasks"
        )
    return {task_id: by_id[task_id] for task_id in sorted(by_id)}, target_binding


def _quality_v7_payload(record: QualityRecord, item: PreparedTask) -> dict[str, object]:
    payload: dict[str, object] = {
        "action_envelope_record_digest": record.action_envelope_record_digest,
        "best_actions": list(record.best_actions()),
        "certificate_digest": item.certificate_digest,
        "charged_work_receipt": record.charged_work_receipt,
        "context_digest": record.context_digest,
        "context_version": record.context_version,
        "continuation_policy": record.continuation_policy,
        "environment_seed": record.environment_seed,
        "evaluated": [dataclasses.asdict(action) for action in record.evaluated],
        "evaluator_protocol_digest": item.evaluator_protocol_digest,
        "exact_state": record.exact_state,
        "initializer_record_digest": item.initializer_record_digest,
        "instance_id": item.instance_id,
        "label_version": record.label_version,
        "lineage": record.lineage,
        "observation": record.observation,
        "partition": item.partition,
        "prefix": list(record.prefix),
        "reference_status": item.reference_status,
        "schema": "isingfold.quality-counterfactual",
        "schema_version": 7,
        "state_fingerprint": record.state_fingerprint,
        "support": list(record.support),
        "support_fingerprint": record.support_fingerprint,
        "task_id": record.instance,
    }
    return {**payload, "record_digest": content_digest(payload)}


def _clock_value(clock: Callable[[], int], label: str) -> int:
    value = clock()
    if type(value) is not int or value < 0:
        raise QualityResolutionError(f"{label} must return nonnegative integer nanoseconds")
    return value


def _timed_continuation(
    runner: Callable[..., ContinuationResult | None],
    *,
    task: PreparedTask,
    context: Context,
    selector: StrengthSelector,
    prefix: tuple[int, ...],
    environment_seed: int,
    continuation_seed: int,
    initializer_bank: InitializerSnapshotBank,
    initializer_bank_manifest_sha256: str,
    initializer_bank_episode_index: int,
    authority: QualityResolutionExecutionAuthority,
    allow_test_initializer_bank: bool,
    cpu_time_ns: Callable[[], int],
    wall_time_ns: Callable[[], int],
) -> tuple[ContinuationResult, int, int]:
    cpu_start = _clock_value(cpu_time_ns, "CPU clock")
    wall_start = _clock_value(wall_time_ns, "wall clock")
    outcome = runner(
        task.task,
        context,
        initializer=None,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=(initializer_bank_manifest_sha256),
        initializer_bank_episode_index=initializer_bank_episode_index,
        prepared_task=task,
        target_access=authority.target_access,
        ground_partition_receipt=authority.ground_partition_receipt,
        allow_test_initializer_bank=allow_test_initializer_bank,
        selector=selector,
        prefix=prefix,
        policy=random_masked_policy,
        seed=environment_seed,
        continuation_seed=continuation_seed,
        reward_reads=context.n_est_reads,
        max_steps=CONTINUATION_MAX_STEPS,
    )
    cpu_end = _clock_value(cpu_time_ns, "CPU clock")
    wall_end = _clock_value(wall_time_ns, "wall clock")
    if cpu_end <= cpu_start or wall_end <= wall_start:
        raise QualityResolutionError("capacity timing clocks must advance for every continuation")
    if outcome is None or not isinstance(outcome, ContinuationResult):
        raise QualityResolutionError("capacity continuation failed exact replay")
    return outcome, cpu_end - cpu_start, wall_end - wall_start


def _execute_capacity_row(
    row: Mapping[str, object],
    planned: Mapping[str, object],
    task: PreparedTask,
    *,
    context: Context,
    selector: StrengthSelector,
    continuations: int,
    evaluated_actions: int,
    source_manifest_sha256: str,
    initializer_bank: InitializerSnapshotBank,
    initializer_bank_contract: Mapping[str, object],
    authority: QualityResolutionExecutionAuthority,
    allow_test_initializer_bank: bool,
    runner: Callable[..., ContinuationResult | None],
    cpu_time_ns: Callable[[], int],
    wall_time_ns: Callable[[], int],
) -> tuple[dict[str, object], list[int], list[int], list[int], int]:
    provenance = _target_provenance(task, source_manifest_sha256)
    replayed = replay_decision_with_action_envelope(
        task.task,
        context,
        initializer=None,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=initializer_bank_contract["manifest_sha256"],
        initializer_bank_episode_index=planned["initializer_bank_episode_index"],
        prepared_task=task,
        target_access=authority.target_access,
        ground_partition_receipt=authority.ground_partition_receipt,
        allow_test_initializer_bank=allow_test_initializer_bank,
        selector=selector,
        prefix=planned["prefix"],  # type: ignore[arg-type]
        seed=int(planned["environment_seed"]),
        reward_reads=context.n_est_reads,
        provenance_fingerprint=provenance,
    )
    if replayed is None:
        raise QualityResolutionError("capacity target-bound row is no longer replayable")
    decision, envelope = replayed
    if (
        decision.state_fingerprint != planned["state_fingerprint"]
        or decision.support_fingerprint != planned["support_fingerprint"]
        or public_envelope_projection_digest(envelope) != row["public_envelope_projection_digest"]
    ):
        raise QualityResolutionError(
            "target-bound capacity row differs from its public v2 projection"
        )
    action_sample = deterministic_action_sample(
        decision,
        evaluated_actions=evaluated_actions,
        seed=int(planned["environment_seed"]),
    )
    action_indices = action_sample.action_indices
    if list(action_indices) != row["action_indices"]:
        raise QualityResolutionError("capacity target-bound action subset differs")
    expected_applied = {
        int(entry["action_index"]): entry["projection_digest"]
        for entry in row["public_applied_action_projection_digests"]
    }
    support, exact_state, observation = decision_snapshot(decision)
    labels: list[ActionQuality] = []
    receipt_bytes: list[int] = []
    cpu_values: list[int] = []
    wall_values: list[int] = []
    valid_total = 0
    planned_actions = {int(action["action_index"]): action for action in planned["actions"]}
    for action_index in action_indices:
        bound = envelope.candidates[action_index]
        applied_digest: str | None = None
        public_applied: str | None = None
        if not bound.opcode.is_terminal:
            applied = apply_envelope_action(envelope, action_index)
            applied.verify_against(envelope)
            applied_digest = applied.record_digest
            public_applied = public_applied_action_projection_digest(applied)
        if public_applied != expected_applied[action_index]:
            raise QualityResolutionError(
                "capacity target-bound successor differs from its public v2 projection"
            )
        plan_action = planned_actions[action_index]
        if (
            bound.payload_key != plan_action["payload_key"]
            or bound.opcode.value != plan_action["opcode"]
            or bound.payload_digest != plan_action["selected_payload_digest"]
        ):
            raise QualityResolutionError("capacity target-bound action identity differs")
        rewards: list[float] = []
        valid: list[bool] = []
        seeds: list[int] = []
        receipts: list[Mapping[str, object]] = []
        for continuation_index in range(continuations):
            seed = int(plan_action["continuation_seeds"][continuation_index])
            outcome, cpu_ns, wall_ns = _timed_continuation(
                runner,
                task=task,
                context=context,
                selector=selector,
                prefix=(*tuple(planned["prefix"]), action_index),
                environment_seed=int(planned["environment_seed"]),
                continuation_seed=seed,
                initializer_bank=initializer_bank,
                initializer_bank_manifest_sha256=str(initializer_bank_contract["manifest_sha256"]),
                initializer_bank_episode_index=int(planned["initializer_bank_episode_index"]),
                authority=authority,
                allow_test_initializer_bank=allow_test_initializer_bank,
                cpu_time_ns=cpu_time_ns,
                wall_time_ns=wall_time_ns,
            )
            receipt = outcome.receipt
            if not isinstance(receipt, Mapping):
                raise QualityResolutionError("capacity continuation omitted its full receipt")
            validate_continuation_receipt(receipt)
            initializer_binding = receipt.get("initializer_binding")
            if (
                receipt.get("continuation_seed") != seed
                or receipt.get("returned_valid") is not outcome.returned_valid
                or not isinstance(initializer_binding, Mapping)
                or initializer_binding.get("bank_contract_record_digest")
                != initializer_bank_contract.get("record_digest")
                or initializer_binding.get("episode_schedule_index")
                != planned.get("initializer_bank_episode_index")
                or initializer_binding.get("bootstrap_record_digest")
                != planned.get("initializer_bootstrap_record_digest")
            ):
                raise QualityResolutionError("capacity continuation receipt identity differs")
            reward = float(outcome.reward if outcome.returned_valid else 0.0)
            seeds.append(seed)
            rewards.append(reward)
            valid.append(outcome.returned_valid)
            receipts.append(receipt)
            valid_total += int(outcome.returned_valid)
            receipt_bytes.append(len(canonical_json_bytes(receipt)))
            cpu_values.append(cpu_ns)
            wall_values.append(wall_ns)
        label = ActionQuality(
            action_index=action_index,
            payload_key=bound.payload_key,
            opcode=bound.opcode.value,
            q_mu=math.fsum(rewards) / continuations,
            continuations=continuations,
            valid_returns=sum(valid),
            inclusion_probability=action_sample.inclusion_probability(action_index),
            selected_payload_digest=bound.payload_digest,
            applied_action_record_digest=applied_digest,
            continuation_seeds=tuple(seeds),
            continuation_rewards=tuple(rewards),
            continuation_valid=tuple(valid),
            continuation_receipts=tuple(receipts),
        )
        label.validate()
        labels.append(label)
    quality_record = QualityRecord(
        instance=task.task_id,
        lineage=task.task.lineage,
        prefix=tuple(planned["prefix"]),
        support=support,
        evaluated=tuple(labels),
        continuation_policy=CONTINUATION_POLICY_ID,
        action_envelope_record_digest=envelope.record_digest,
        state_fingerprint=decision.state_fingerprint,
        support_fingerprint=decision.support_fingerprint,
        context_version=decision.context_version,
        charged_work_receipt=decision.charged_work_receipt.as_dict(),
        exact_state=exact_state,
        observation=observation,
        environment_seed=int(planned["environment_seed"]),
        context_digest=quality_context_digest(context),
    )
    return (
        _quality_v7_payload(quality_record, task),
        receipt_bytes,
        cpu_values,
        wall_values,
        valid_total,
    )


def _default_peak_memory_bytes() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return raw if sys.platform == "darwin" else raw * 1024


def _default_scratch_free_bytes(path: Path) -> int:
    return int(shutil.disk_usage(path).free)


def _distribution(values: Sequence[int], label: str) -> dict[str, int | float]:
    if not values:
        raise QualityResolutionError(f"quality-capacity {label} distribution is empty")
    checked = [_nonnegative(value, f"quality-capacity {label} value") for value in values]
    ordered = sorted(checked)
    count = len(ordered)
    total = sum(ordered)
    midpoint = count // 2
    median = (
        float(ordered[midpoint]) if count % 2 else (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    )
    return {
        "count": count,
        "maximum": ordered[-1],
        "mean": total / count,
        "median": median,
        "minimum": ordered[0],
        "p95": ordered[math.ceil(0.95 * count) - 1],
        "total": total,
    }


def _study_quality_authority(
    study_record: Mapping[str, object],
    authority: QualityResolutionExecutionAuthority,
) -> dict[str, object]:
    partition = authority.quality_authority["training_partition"]
    ground = partition["ground_partition"]
    expected = {
        "ground_partition_receipt_record_digest": authority.ground_partition_receipt[
            "record_digest"
        ],
        "ground_partition_receipt_sha256": ground["receipt_sha256"],
        "quality_authority_record_digest": authority.quality_authority["record_digest"],
        "target_access_record_digest": authority.target_access["record_digest"],
    }
    if study_record.get("quality_authority") != expected:
        raise QualityResolutionError(
            "quality-capacity execution authority differs from resolution study"
        )
    return expected


def run_quality_capacity_canary(
    plan: QualityResolutionPlan,
    *,
    expected_plan_sha256: str,
    resolution_receipt: QualityResolutionStudyReceipt,
    selection_path: str | Path,
    expected_selection_sha256: str,
    budget_path: str | Path,
    expected_budget_sha256: str,
    prepared: Sequence[PreparedTask],
    target_records: Sequence[Mapping[str, object]],
    authority: QualityResolutionExecutionAuthority,
    context: Context,
    selector: StrengthSelector,
    runtime_identity: Mapping[str, object],
    runtime_identity_digest: str,
    scratch_directory: str | Path,
    output_path: str | Path,
    initializer_bank: InitializerSnapshotBank | None = None,
    expected_initializer_bank_manifest_sha256: str | None = None,
    allow_test_initializer_bank: bool = False,
    continuation_runner: Callable[..., ContinuationResult | None] = run_continuation,
) -> str:
    """Run the exact target-bound canary and publish its immutable capacity receipt."""

    destination = Path(output_path)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"quality-capacity canary already exists: {destination}")
    plan_record, plan_sha256 = _plan_identity(plan, expected_plan_sha256)
    initializer_bank_contract = _initializer_bank_identity(
        plan_record,
        initializer_bank,
        expected_initializer_bank_manifest_sha256,
        allow_test_bank=allow_test_initializer_bank,
    )
    assert isinstance(initializer_bank, InitializerSnapshotBank)
    study, continuations = _study_authorization(
        resolution_receipt, plan_record=plan_record, plan_sha256=plan_sha256
    )
    selection_artifact = load_quality_capacity_canary_selection(
        selection_path,
        expected_selection_sha256=expected_selection_sha256,
        plan=plan,
        expected_plan_sha256=expected_plan_sha256,
        resolution_receipt=resolution_receipt,
    )
    selection_record = selection_artifact.as_dict()
    budget = load_quality_capacity_budget(
        budget_path, expected_budget_sha256=expected_budget_sha256
    )
    if not isinstance(authority, QualityResolutionExecutionAuthority):
        raise TypeError("authority must be QualityResolutionExecutionAuthority")
    authority.validate_against_plan(plan)
    study_authority = _study_quality_authority(study, authority)
    _snapshot, context_identity = _context_identity(context, plan_record)
    runtime, runtime_digest = _validate_runtime_identity(
        runtime_identity,
        runtime_identity_digest,
        plan_record=plan_record,
        continuation_runner=continuation_runner,
        authority=authority,
    )
    tasks, target_binding = _target_task_map(
        prepared,
        target_records=target_records,
        plan_record=plan_record,
        authority=authority,
    )

    scratch = Path(scratch_directory)
    if scratch.is_symlink() or not scratch.is_dir():
        raise QualityResolutionError(
            "quality-capacity scratch root must be an existing regular directory"
        )
    scratch = scratch.resolve()
    observed_scratch_free = _nonnegative(
        _default_scratch_free_bytes(scratch), "quality-capacity scratch free bytes"
    )
    memory_before = _nonnegative(
        _default_peak_memory_bytes(), "quality-capacity peak memory before canary"
    )
    planned_rows = {str(row["row_id"]): row for row in plan_record["production_plan"]["rows"]}
    selected_rows = selection_record["rows"]
    if not isinstance(selected_rows, list):  # Already checked by the loader.
        raise RuntimeError("quality-capacity selection lost its row array")
    evaluated_actions = int(plan_record["config"]["registered"]["evaluated_actions"])
    source_manifest_sha256 = str(plan_record["prepared_corpus"]["manifest_sha256"])
    row_bytes: list[int] = []
    row_overhead_bytes: list[int] = []
    receipt_bytes: list[int] = []
    cpu_values: list[int] = []
    wall_values: list[int] = []
    row_digests: list[str] = []
    valid_total = 0
    disk_high_water = 0
    with tempfile.TemporaryDirectory(prefix="quality-capacity-", dir=scratch) as temporary:
        temporary_root = Path(temporary)
        for position, row in enumerate(selected_rows):
            planned = planned_rows[str(row["row_id"])]
            payload, receipts, cpu, wall, valid = _execute_capacity_row(
                row,
                planned,
                tasks[str(row["task_id"])],
                context=context,
                selector=selector,
                continuations=continuations,
                evaluated_actions=evaluated_actions,
                source_manifest_sha256=source_manifest_sha256,
                initializer_bank=initializer_bank,
                initializer_bank_contract=initializer_bank_contract,
                authority=authority,
                allow_test_initializer_bank=allow_test_initializer_bank,
                runner=continuation_runner,
                cpu_time_ns=time.process_time_ns,
                wall_time_ns=time.monotonic_ns,
            )
            raw = canonical_json_bytes(payload) + b"\n"
            row_path = temporary_root / f"{position:08d}.json"
            row_path.write_bytes(raw)
            serialized = len(raw)
            receipt_total = sum(receipts)
            if serialized < receipt_total:
                raise QualityResolutionError(
                    "quality-capacity row is smaller than its continuation receipts"
                )
            row_bytes.append(serialized)
            row_overhead_bytes.append(serialized - receipt_total)
            receipt_bytes.extend(receipts)
            cpu_values.extend(cpu)
            wall_values.extend(wall)
            row_digests.append(str(payload["record_digest"]))
            valid_total += valid
            disk_high_water = max(
                disk_high_water,
                sum(path.stat().st_size for path in temporary_root.iterdir()),
            )
    memory_after = _nonnegative(
        _default_peak_memory_bytes(), "quality-capacity peak memory after canary"
    )
    if not receipt_bytes or not cpu_values or not wall_values or not row_bytes:
        raise QualityResolutionError("quality-capacity canary executed no trajectories")
    trajectories = len(receipt_bytes)
    actions = sum(len(row["action_indices"]) for row in selected_rows)
    if trajectories != actions * continuations:
        raise QualityResolutionError(
            "quality-capacity canary trajectory census differs from selected C"
        )
    effective_bytes = max(receipt_bytes) + max(row_overhead_bytes)
    projection = compute_guarded_capacity_projection(
        effective_max_bytes_per_trajectory=effective_bytes,
        max_cpu_time_ns_per_trajectory=max(cpu_values),
        max_wall_time_ns_per_trajectory=max(wall_values),
        upper_trajectory_census=int(selection_record["upper_census"]["trajectories"]),
        scratch_free_bytes=observed_scratch_free,
        budget=budget,
    )
    comparisons = projection.pop("comparisons")
    capacity_pass = projection.pop("pass")
    measurements = {
        "continuation_receipt_bytes": _distribution(receipt_bytes, "continuation receipt bytes"),
        "cpu_time_nanoseconds": _distribution(cpu_values, "CPU nanoseconds"),
        "effective_max_bytes_per_trajectory": effective_bytes,
        "peak_memory_bytes": {
            "after": memory_after,
            "before": memory_before,
            "maximum": max(memory_before, memory_after),
        },
        "row_bytes": _distribution(row_bytes, "row bytes"),
        "row_nonreceipt_overhead_bytes": _distribution(
            row_overhead_bytes, "row non-receipt overhead bytes"
        ),
        "scratch_free_bytes": observed_scratch_free,
        "wall_time_nanoseconds": _distribution(wall_values, "wall nanoseconds"),
    }
    actual = {
        "actions": actions,
        "disk_high_water_bytes": disk_high_water,
        "invalid_trajectories": trajectories - valid_total,
        "lineages": len(selection_record["selection"]["selected_lineages"]),
        "row_record_digests": row_digests,
        "row_set_digest": content_digest(row_digests),
        "rows": len(selected_rows),
        "serialized_artifact_bytes": sum(row_bytes),
        "trajectories": trajectories,
        "valid_trajectories": valid_total,
    }
    budget_record = budget.as_dict()
    train_authority = {
        **study_authority,
        "execution_identity": _thaw(authority.execution_identity),
        "execution_identity_digest": authority.execution_identity_digest,
        "target_count": target_binding["target_count"],
        "target_record_set_digest": target_binding["target_record_set_digest"],
        "target_set_digest": target_binding["target_set_digest"],
        "target_sha256": target_binding["target_sha256"],
    }
    registry = plan_record["implementation"]["registry"]
    implementation = {
        "plan_implementation_digest": plan_record["implementation"]["digest"],
        "public_projection_version": QUALITY_PUBLIC_PROJECTION_VERSION,
        "quality_implementation_contract_digest": registry[
            "quality_implementation_contract_digest"
        ],
        "quality_module_sha256": registry["quality_module_sha256"],
    }
    payload: dict[str, object] = {
        "actual": actual,
        "budget": {
            "raw_sha256": _digest(
                expected_budget_sha256, "expected quality-capacity budget SHA-256"
            ),
            "record": budget_record,
        },
        "budget_comparisons": comparisons,
        "capacity_selection": {
            "raw_sha256": selection_artifact.raw_sha256,
            "record_digest": selection_record["record_digest"],
        },
        "context": {
            "digest": context_identity,
            "snapshot": plan_record["context"]["snapshot"],
        },
        "corpus": plan_record["prepared_corpus"],
        "device_parity": {
            "record_digest": plan_record["selector"]["device_parity_record_digest"],
            "sha256": plan_record["selector"]["device_parity_sha256"],
        },
        "implementation": implementation,
        "initializer_bank": initializer_bank_contract,
        "measurements": measurements,
        "pass": capacity_pass,
        "projections": projection,
        "resolution_receipt": {
            "raw_sha256": resolution_receipt.raw_sha256,
            "record_digest": study["record_digest"],
            "selected_continuations": continuations,
        },
        "runtime": {"digest": runtime_digest, "identity": runtime},
        "schema": QUALITY_CAPACITY_CANARY_SCHEMA,
        "schema_version": QUALITY_CAPACITY_CANARY_VERSION,
        "selection": selection_record["selection"],
        "selector": plan_record["selector"],
        "train_authority": train_authority,
        "upper_census": selection_record["upper_census"],
    }
    record = {**payload, "record_digest": content_digest(payload)}
    return _publish_file(destination, record)


_DISTRIBUTION_FIELDS = {
    "count",
    "maximum",
    "mean",
    "median",
    "minimum",
    "p95",
    "total",
}


def _validate_distribution(
    value: object,
    *,
    label: str,
    expected_count: int,
    positive: bool,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != _DISTRIBUTION_FIELDS:
        raise QualityResolutionError(f"quality-capacity {label} distribution schema differs")
    count = _positive(value["count"], f"quality-capacity {label} count")
    if count != expected_count:
        raise QualityResolutionError(f"quality-capacity {label} count differs")
    lower = _nonnegative(value["minimum"], f"quality-capacity {label} minimum")
    upper = _nonnegative(value["maximum"], f"quality-capacity {label} maximum")
    p95 = _nonnegative(value["p95"], f"quality-capacity {label} p95")
    total = _nonnegative(value["total"], f"quality-capacity {label} total")
    if positive and lower <= 0:
        raise QualityResolutionError(f"quality-capacity {label} must be strictly positive")
    mean = value["mean"]
    median = value["median"]
    if (
        isinstance(mean, bool)
        or not isinstance(mean, (int, float))
        or not math.isfinite(float(mean))
        or isinstance(median, bool)
        or not isinstance(median, (int, float))
        or not math.isfinite(float(median))
        or float(mean) != total / count
        or not lower <= float(median) <= upper
        or not lower <= p95 <= upper
        or not lower <= upper
        or not lower * count <= total <= upper * count
    ):
        raise QualityResolutionError(f"quality-capacity {label} distribution is inconsistent")
    return value


def load_quality_capacity_canary(
    path: str | Path,
    *,
    expected_canary_sha256: str,
    plan: QualityResolutionPlan,
    expected_plan_sha256: str,
    resolution_receipt: QualityResolutionStudyReceipt,
    selection_path: str | Path,
    expected_selection_sha256: str,
    budget_path: str | Path,
    expected_budget_sha256: str,
    allow_test_initializer_bank: bool = False,
) -> QualityCapacityCanary:
    """Authenticate and independently reconstruct the capacity decision arithmetic."""

    plan_record, plan_sha256 = _plan_identity(plan, expected_plan_sha256)
    production = plan_record.get("production_plan")
    planned_bank = production.get("initializer_bank") if isinstance(production, Mapping) else None
    if not isinstance(planned_bank, Mapping):
        raise QualityResolutionError("quality-capacity plan omits its initializer-bank contract")
    try:
        validate_quality_initializer_bank_contract(
            planned_bank,
            require_publication=not allow_test_initializer_bank,
        )
    except (TypeError, ValueError) as error:
        raise QualityResolutionError(str(error)) from error
    study, continuations = _study_authorization(
        resolution_receipt, plan_record=plan_record, plan_sha256=plan_sha256
    )
    selection = load_quality_capacity_canary_selection(
        selection_path,
        expected_selection_sha256=expected_selection_sha256,
        plan=plan,
        expected_plan_sha256=expected_plan_sha256,
        resolution_receipt=resolution_receipt,
    )
    selection_record = selection.as_dict()
    budget = load_quality_capacity_budget(
        budget_path, expected_budget_sha256=expected_budget_sha256
    )
    record, _raw, observed_sha = _read_pinned_json(
        path, expected_canary_sha256, "quality-capacity canary"
    )
    if set(record) != _CANARY_FIELDS:
        raise QualityResolutionError("quality-capacity canary schema fields differ")
    _verify_record(record, "quality-capacity canary")
    if (
        record.get("schema") != QUALITY_CAPACITY_CANARY_SCHEMA
        or record.get("schema_version") != QUALITY_CAPACITY_CANARY_VERSION
        or record.get("corpus") != plan_record["prepared_corpus"]
        or record.get("selector") != plan_record["selector"]
        or record.get("context") != plan_record["context"]
        or record.get("initializer_bank") != plan_record["production_plan"]["initializer_bank"]
        or record.get("upper_census") != selection_record["upper_census"]
        or record.get("selection") != selection_record["selection"]
    ):
        raise QualityResolutionError("quality-capacity canary belongs to another plan")
    if record.get("resolution_receipt") != {
        "raw_sha256": resolution_receipt.raw_sha256,
        "record_digest": study["record_digest"],
        "selected_continuations": continuations,
    }:
        raise QualityResolutionError("quality-capacity canary belongs to another study")
    if record.get("capacity_selection") != {
        "raw_sha256": selection.raw_sha256,
        "record_digest": selection_record["record_digest"],
    }:
        raise QualityResolutionError("quality-capacity canary selection identity differs")
    if record.get("budget") != {
        "raw_sha256": _digest(expected_budget_sha256, "expected quality-capacity budget SHA-256"),
        "record": budget.as_dict(),
    }:
        raise QualityResolutionError("quality-capacity canary budget identity differs")

    registry = plan_record["implementation"]["registry"]
    expected_implementation = {
        "plan_implementation_digest": plan_record["implementation"]["digest"],
        "public_projection_version": QUALITY_PUBLIC_PROJECTION_VERSION,
        "quality_implementation_contract_digest": registry[
            "quality_implementation_contract_digest"
        ],
        "quality_module_sha256": registry["quality_module_sha256"],
    }
    if record.get("implementation") != expected_implementation:
        raise QualityResolutionError("quality-capacity implementation identity differs")
    runtime = record.get("runtime")
    if not isinstance(runtime, Mapping) or set(runtime) != {"digest", "identity"}:
        raise QualityResolutionError("quality-capacity runtime block is malformed")
    _validate_runtime_identity(
        runtime["identity"],  # type: ignore[arg-type]
        runtime["digest"],  # type: ignore[arg-type]
        plan_record=plan_record,
        continuation_runner=run_continuation,
    )

    train = record.get("train_authority")
    expected_train_fields = {
        "execution_identity",
        "execution_identity_digest",
        "ground_partition_receipt_record_digest",
        "ground_partition_receipt_sha256",
        "quality_authority_record_digest",
        "target_access_record_digest",
        "target_count",
        "target_record_set_digest",
        "target_set_digest",
        "target_sha256",
    }
    if not isinstance(train, Mapping) or set(train) != expected_train_fields:
        raise QualityResolutionError("quality-capacity train authority schema differs")
    compact = {
        key: train[key]
        for key in (
            "ground_partition_receipt_record_digest",
            "ground_partition_receipt_sha256",
            "quality_authority_record_digest",
            "target_access_record_digest",
        )
    }
    if compact != study["quality_authority"]:
        raise QualityResolutionError("quality-capacity train authority differs from study")
    for field in expected_train_fields - {"execution_identity", "target_count"}:
        _digest(train[field], f"quality-capacity train authority {field}")
    execution = train["execution_identity"]
    if (
        not isinstance(execution, Mapping)
        or content_digest(execution) != train["execution_identity_digest"]
        or execution.get("quality_module_sha256") != registry["quality_module_sha256"]
        or execution.get("quality_implementation_contract_digest")
        != registry["quality_implementation_contract_digest"]
        or execution.get("selector_digest") != plan_record["selector"]["selector_digest"]
        or execution.get("device") != plan_record["selector"]["device"]
        or runtime["identity"]["execution_runtime_sha256"] != execution.get("runtime_sha256")
    ):
        raise QualityResolutionError(
            "quality-capacity execution identity differs from its plan or runtime"
        )
    expected_target_count = len(
        {str(row["task_id"]) for row in plan_record["production_plan"]["rows"]}
    )
    if train["target_count"] != expected_target_count:
        raise QualityResolutionError("quality-capacity target census differs from plan")

    actual = record.get("actual")
    actual_fields = {
        "actions",
        "disk_high_water_bytes",
        "invalid_trajectories",
        "lineages",
        "row_record_digests",
        "row_set_digest",
        "rows",
        "serialized_artifact_bytes",
        "trajectories",
        "valid_trajectories",
    }
    if not isinstance(actual, Mapping) or set(actual) != actual_fields:
        raise QualityResolutionError("quality-capacity actual census schema differs")
    expected_rows = selection_record["rows"]
    expected_actions = sum(len(row["action_indices"]) for row in expected_rows)
    expected_trajectories = expected_actions * continuations
    for field in (
        "actions",
        "disk_high_water_bytes",
        "invalid_trajectories",
        "lineages",
        "rows",
        "serialized_artifact_bytes",
        "trajectories",
        "valid_trajectories",
    ):
        _nonnegative(actual[field], f"quality-capacity actual {field}")
    if (
        actual["actions"] != expected_actions
        or actual["lineages"] != QUALITY_CAPACITY_CANARY_LINEAGES
        or actual["rows"] != len(expected_rows)
        or actual["trajectories"] != expected_trajectories
        or actual["valid_trajectories"] + actual["invalid_trajectories"] != expected_trajectories
        or actual["disk_high_water_bytes"] < actual["serialized_artifact_bytes"]
    ):
        raise QualityResolutionError("quality-capacity actual census is inconsistent")
    row_digests = actual["row_record_digests"]
    if not isinstance(row_digests, list) or len(row_digests) != actual["rows"]:
        raise QualityResolutionError("quality-capacity row digest census differs")
    for digest in row_digests:
        _digest(digest, "quality-capacity serialized row digest")
    if actual["row_set_digest"] != content_digest(row_digests):
        raise QualityResolutionError("quality-capacity row-set digest is inconsistent")

    measurements = record.get("measurements")
    measurement_fields = {
        "continuation_receipt_bytes",
        "cpu_time_nanoseconds",
        "effective_max_bytes_per_trajectory",
        "peak_memory_bytes",
        "row_bytes",
        "row_nonreceipt_overhead_bytes",
        "scratch_free_bytes",
        "wall_time_nanoseconds",
    }
    if not isinstance(measurements, Mapping) or set(measurements) != measurement_fields:
        raise QualityResolutionError("quality-capacity measurement schema differs")
    receipts = _validate_distribution(
        measurements["continuation_receipt_bytes"],
        label="continuation receipt bytes",
        expected_count=expected_trajectories,
        positive=True,
    )
    cpu = _validate_distribution(
        measurements["cpu_time_nanoseconds"],
        label="CPU nanoseconds",
        expected_count=expected_trajectories,
        positive=True,
    )
    wall = _validate_distribution(
        measurements["wall_time_nanoseconds"],
        label="wall nanoseconds",
        expected_count=expected_trajectories,
        positive=True,
    )
    row_sizes = _validate_distribution(
        measurements["row_bytes"],
        label="row bytes",
        expected_count=len(expected_rows),
        positive=True,
    )
    overhead = _validate_distribution(
        measurements["row_nonreceipt_overhead_bytes"],
        label="row non-receipt overhead bytes",
        expected_count=len(expected_rows),
        positive=False,
    )
    effective = _positive(
        measurements["effective_max_bytes_per_trajectory"],
        "quality-capacity effective bytes per trajectory",
    )
    scratch_free = _nonnegative(
        measurements["scratch_free_bytes"], "quality-capacity scratch free bytes"
    )
    peak = measurements["peak_memory_bytes"]
    if not isinstance(peak, Mapping) or set(peak) != {"after", "before", "maximum"}:
        raise QualityResolutionError("quality-capacity peak-memory schema differs")
    before = _nonnegative(peak["before"], "quality-capacity peak memory before")
    after = _nonnegative(peak["after"], "quality-capacity peak memory after")
    if peak["maximum"] != max(before, after):
        raise QualityResolutionError("quality-capacity peak-memory maximum differs")
    if (
        effective != receipts["maximum"] + overhead["maximum"]
        or actual["serialized_artifact_bytes"] != row_sizes["total"]
    ):
        raise QualityResolutionError("quality-capacity byte measurements are inconsistent")
    recomputed = compute_guarded_capacity_projection(
        effective_max_bytes_per_trajectory=effective,
        max_cpu_time_ns_per_trajectory=int(cpu["maximum"]),
        max_wall_time_ns_per_trajectory=int(wall["maximum"]),
        upper_trajectory_census=int(selection_record["upper_census"]["trajectories"]),
        scratch_free_bytes=scratch_free,
        budget=budget,
    )
    comparisons = recomputed.pop("comparisons")
    capacity_pass = recomputed.pop("pass")
    if (
        record.get("projections") != recomputed
        or record.get("budget_comparisons") != comparisons
        or record.get("pass") is not capacity_pass
    ):
        raise QualityResolutionError("quality-capacity guarded projection cannot be reproduced")
    return QualityCapacityCanary(record, observed_sha)


def require_passing_quality_capacity_canary(
    path: str | Path,
    *,
    expected_canary_sha256: str,
    plan: QualityResolutionPlan,
    expected_plan_sha256: str,
    resolution_receipt: QualityResolutionStudyReceipt,
    selection_path: str | Path,
    expected_selection_sha256: str,
    budget_path: str | Path,
    expected_budget_sha256: str,
) -> dict[str, object]:
    """Return the narrow launch capability only for a pinned passing receipt."""

    canary = load_quality_capacity_canary(
        path,
        expected_canary_sha256=expected_canary_sha256,
        plan=plan,
        expected_plan_sha256=expected_plan_sha256,
        resolution_receipt=resolution_receipt,
        selection_path=selection_path,
        expected_selection_sha256=expected_selection_sha256,
        budget_path=budget_path,
        expected_budget_sha256=expected_budget_sha256,
    )
    record = canary.as_dict()
    if record["pass"] is not True:
        raise QualityResolutionError("full quality generation is blocked by the capacity canary")
    initializer_bank = record["initializer_bank"]
    if not isinstance(initializer_bank, Mapping):  # guarded by the strict loader
        raise RuntimeError("quality-capacity canary lost its initializer-bank identity")
    return {
        "capacity_budget_sha256": record["budget"]["raw_sha256"],
        "capacity_canary_record_digest": record["record_digest"],
        "capacity_canary_sha256": canary.raw_sha256,
        "capacity_selection_sha256": record["capacity_selection"]["raw_sha256"],
        "full_quality_launch_authorized": True,
        "initializer_bank_contract_record_digest": initializer_bank["record_digest"],
        "initializer_bank_manifest_sha256": initializer_bank["manifest_sha256"],
        "quality_implementation_contract_digest": record["implementation"][
            "quality_implementation_contract_digest"
        ],
        "selected_continuations": record["resolution_receipt"]["selected_continuations"],
    }


__all__ = [
    "QUALITY_CAPACITY_BUDGET_SCHEMA",
    "QUALITY_CAPACITY_BUDGET_VERSION",
    "QUALITY_CAPACITY_CANARY_LINEAGES",
    "QUALITY_CAPACITY_CANARY_SCHEMA",
    "QUALITY_CAPACITY_CANARY_SELECTION_SCHEMA",
    "QUALITY_CAPACITY_CANARY_SELECTION_VERSION",
    "QUALITY_CAPACITY_CANARY_VERSION",
    "QUALITY_CAPACITY_FREE_SPACE_FACTOR",
    "QUALITY_CAPACITY_MEASUREMENT_PROTOCOL",
    "QUALITY_CAPACITY_SAFETY_FACTOR",
    "QUALITY_CAPACITY_SELECTION_RULE",
    "QualityCapacityBudget",
    "QualityCapacityCanary",
    "QualityCapacityCanarySelection",
    "build_quality_capacity_canary_selection",
    "compute_guarded_capacity_projection",
    "load_quality_capacity_budget",
    "load_quality_capacity_canary",
    "load_quality_capacity_canary_selection",
    "publish_quality_capacity_budget",
    "require_passing_quality_capacity_canary",
    "run_quality_capacity_canary",
]
