"""Immutable, target-free planning for the registered quality-resolution study.

This module owns only phase 1 of the workflow.  It authenticates the registered config and
staged grid, validates a complete public all-train projection, selects the preregistered
lineage sample, and seals the stage-by-shard schedule.  Evaluator target paths and target
access capabilities are deliberately absent from every public API in this module.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from isingfold.rl.contracts import Context, DecisionState, Opcode, stable_digest
from isingfold.rl.data.action_certificate import apply_envelope_action
from isingfold.rl.data.exact_conformance import (
    ExactConformanceError,
    publish_new_file,
    read_regular_file,
)
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.prepared import PreparedDesignCondition, PreparedTask
from isingfold.rl.data.quality import (
    continuation_seed,
    context_digest as quality_context_digest,
    deterministic_action_sample,
    quality_initializer_bank_contract,
    quality_initializer_bank_episode_binding,
    replay_decision_with_action_envelope,
    validate_quality_initializer_bank_contract,
)
from isingfold.rl.data.quality_resolution import (
    QUALITY_RESOLUTION_CONFIG_SCHEMA,
    QUALITY_RESOLUTION_CONFIG_VERSION,
    QualityResolutionConfig,
    QualityResolutionError,
    select_resolution_lineages,
    validate_quality_resolution_config,
)
from isingfold.rl.env import StrengthSelector
from isingfold.rl.initializer_bank import InitializerSnapshotBank


QUALITY_RESOLUTION_PLAN_SCHEMA = "isingfold.quality-resolution-plan"
QUALITY_RESOLUTION_PLAN_VERSION = 2
QUALITY_RESOLUTION_PRODUCTION_PLAN_SCHEMA = "isingfold.quality-production-plan"
QUALITY_RESOLUTION_PRODUCTION_PLAN_VERSION = 2
QUALITY_RESOLUTION_ROW_SCHEMA = "isingfold.quality-resolution-plan-row"
QUALITY_RESOLUTION_ROW_VERSION = 2
QUALITY_RESOLUTION_SAMPLE_RULE = "sha256-hash-permutation-without-replacement-v1"
QUALITY_RESOLUTION_SHARD_RULE = "round-robin-whole-lineage-v1"
QUALITY_RESOLUTION_SHARD_COUNT = 64
MINIMUM_PRODUCTION_LINEAGES = 1_024
QUALITY_INITIALIZER_EPISODE_ASSIGNMENT_RULE = "sha256-task-modulo-sealed-episodes-v1"

_HEX = frozenset("0123456789abcdef")
_STRATUM_FIELDS = (
    "application_family",
    "problem_origin",
    "host_family",
    "fault_status",
    "distribution_regime",
    "calibration_status",
    "embedding_difficulty",
    "sampling_difficulty",
    "decision_difficulty",
)
_PLAN_FIELDS = {
    "candidate_continuations",
    "config",
    "context",
    "ground_root",
    "grid",
    "implementation",
    "prepared_corpus",
    "production_plan",
    "production_plan_digest",
    "publisher",
    "record_digest",
    "sample",
    "schema",
    "schema_version",
    "selector",
    "shards",
    "stages",
    "study_id",
    "work_schedule",
}
_FORBIDDEN_TARGET_PAYLOAD_KEYS = {
    "certificate_digest",
    "evaluator_protocol_digest",
    "evaluator_targets",
    "ground_energy",
    "quality_attestation_digest",
    "quality_evidence_manifest_digest",
    "quality_evidence_manifest_sha256",
    "quality_target_count",
    "quality_target_set_digest",
    "reference_energy",
    "reference_status",
    "target_access",
    "target_count",
    "target_path",
    "target_set_digest",
    "target_sha256",
    "targets",
}


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise QualityResolutionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise QualityResolutionError(f"{label} must be nonempty text")
    return value


def _nonnegative_int(value: object, label: str, *, maximum: int | None = None) -> int:
    if type(value) is not int or value < 0 or (maximum is not None and value >= maximum):
        suffix = "" if maximum is None else f" below {maximum}"
        raise QualityResolutionError(f"{label} must be a nonnegative integer{suffix}")
    return value


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise QualityResolutionError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise QualityResolutionError(f"{label} contains non-finite number {token}")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except QualityResolutionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualityResolutionError(f"{label} is invalid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise QualityResolutionError(f"{label} must contain a JSON object")
    try:
        canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise QualityResolutionError(f"{label} is not finite canonical JSON") from error
    return value


def _pinned_json(
    path: str | Path,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], bytes, str]:
    expected = _digest(expected_sha256, f"expected {label} SHA-256")
    try:
        _resolved, raw = read_regular_file(path, label)
    except (ExactConformanceError, OSError) as error:
        raise QualityResolutionError(str(error)) from error
    observed = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(observed, expected):
        raise QualityResolutionError(f"{label} differs from its out-of-band pin")
    return _strict_json(raw, label), raw, observed


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise QualityResolutionError("authenticated mappings require text keys")
        return MappingProxyType({key: _freeze(value[key]) for key in sorted(value)})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise QualityResolutionError("authenticated identities must be finite JSON")
        return value
    if value is None or type(value) in {str, bool, int}:
        return value
    raise QualityResolutionError(
        f"cannot freeze {type(value).__name__} as an authenticated JSON identity"
    )


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_thaw(item) for item in value]
    return value


def _assert_recursively_target_free(value: object, *, path: str = "plan") -> None:
    """Reject evaluator payloads even when hidden inside a nested public section."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise QualityResolutionError("target-free plan mappings require text keys")
            if key in _FORBIDDEN_TARGET_PAYLOAD_KEYS:
                raise QualityResolutionError(
                    f"target-free plan contains forbidden evaluator field at {path}.{key}"
                )
            _assert_recursively_target_free(item, path=f"{path}.{key}")
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_recursively_target_free(item, path=f"{path}[{index}]")
        return
    if isinstance(value, str) and (value.startswith("targets/") or "/targets/" in value):
        raise QualityResolutionError(
            f"target-free plan contains an evaluator target path at {path}"
        )


def _config_dict(config: QualityResolutionConfig) -> dict[str, object]:
    return {
        "schema": QUALITY_RESOLUTION_CONFIG_SCHEMA,
        "schema_version": QUALITY_RESOLUTION_CONFIG_VERSION,
        "study_id": config.study_id,
        "partition": config.partition,
        "production_lineages": config.production_lineages,
        "study_lineages": config.study_lineages,
        "tasks_per_lineage_cap": config.tasks_per_lineage_cap,
        "states_per_lineage_cap": config.states_per_lineage_cap,
        "evaluated_actions": config.evaluated_actions,
        "candidate_continuations": list(config.candidate_continuations),
        "reward_reads": config.reward_reads,
        "quality_seed": config.quality_seed,
        "sampling_seed": config.sampling_seed,
        "familywise_alpha": config.familywise_alpha,
        "minimum_resolved_rows": config.minimum_resolved_rows,
        "minimum_resolved_lineages": config.minimum_resolved_lineages,
        "lineages_per_shard": config.lineages_per_shard,
    }


@dataclass(frozen=True, slots=True)
class ResolutionPlanningInputs:
    """Externally pinned registered inputs that may be opened before targets."""

    config: QualityResolutionConfig
    config_sha256: str
    grid_sha256: str
    grid_resolution_minima: Mapping[str, int]

    def __post_init__(self) -> None:
        _digest(self.config_sha256, "resolution config SHA-256")
        _digest(self.grid_sha256, "staged grid SHA-256")
        expected = {
            "min_resolved_lineages": self.config.minimum_resolved_lineages,
            "min_resolved_rows": self.config.minimum_resolved_rows,
        }
        if dict(self.grid_resolution_minima) != expected:
            raise QualityResolutionError(
                "staged grid quality-resolution minima differ from the registered config"
            )
        object.__setattr__(self, "grid_resolution_minima", MappingProxyType(expected))


def load_quality_resolution_planning_inputs(
    config_path: str | Path,
    *,
    expected_config_sha256: str,
    grid_path: str | Path,
    expected_grid_sha256: str,
) -> ResolutionPlanningInputs:
    """Authenticate the config and staged grid before any production row is inspected."""

    raw_config, _config_bytes, config_sha256 = _pinned_json(
        config_path, expected_config_sha256, "quality-resolution config"
    )
    config = validate_quality_resolution_config(raw_config)
    grid, _grid_bytes, grid_sha256 = _pinned_json(grid_path, expected_grid_sha256, "staged grid")
    if grid.get("schema") != "isingfold.staged-grid" or grid.get("schema_version") != 2:
        raise QualityResolutionError("quality-resolution planning requires staged-grid v2")
    minima = grid.get("quality_resolution")
    if not isinstance(minima, dict) or set(minima) != {
        "min_resolved_lineages",
        "min_resolved_rows",
    }:
        raise QualityResolutionError("staged grid quality-resolution minima schema differs")
    if any(type(value) is not int or value <= 0 for value in minima.values()):
        raise QualityResolutionError("staged grid quality-resolution minima are invalid")
    return ResolutionPlanningInputs(
        config=config,
        config_sha256=config_sha256,
        grid_sha256=grid_sha256,
        grid_resolution_minima=minima,
    )


@dataclass(frozen=True, slots=True)
class PlannedResolutionAction:
    """One target-free action identity and its complete ordered seed tape."""

    action_index: int
    payload_key: str
    opcode: str
    selected_payload_digest: str
    applied_action_record_digest: str | None
    continuation_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        _nonnegative_int(self.action_index, "planned action index")
        _text(self.payload_key, "planned action payload key")
        try:
            opcode = Opcode(self.opcode)
        except (TypeError, ValueError) as error:
            raise QualityResolutionError("planned action opcode is not registered") from error
        _digest(self.selected_payload_digest, "planned selected-payload digest")
        if opcode.is_terminal:
            if self.applied_action_record_digest is not None:
                raise QualityResolutionError(
                    "terminal planned action cannot carry an applied-action digest"
                )
        else:
            _digest(
                self.applied_action_record_digest,
                "planned applied-action record digest",
            )
        if not self.continuation_seeds:
            raise QualityResolutionError("planned action seed registry must be nonempty")
        for position, seed in enumerate(self.continuation_seeds):
            _nonnegative_int(seed, f"planned continuation seed {position}", maximum=2**63)

    def as_dict(self) -> dict[str, object]:
        return {
            "action_index": self.action_index,
            "applied_action_record_digest": self.applied_action_record_digest,
            "continuation_seeds": list(self.continuation_seeds),
            "opcode": self.opcode,
            "payload_key": self.payload_key,
            "selected_payload_digest": self.selected_payload_digest,
        }


@dataclass(frozen=True, slots=True)
class PlannedResolutionRow:
    """One replayable public state projected from the all-train production plan."""

    base_lineage: str
    task_id: str
    instance_id: str
    lineage_schedule_index: int
    state_schedule_index: int
    environment_seed: int
    initializer_bank_episode_index: int
    initializer_bootstrap_record_digest: str
    prefix: tuple[int, ...]
    state_fingerprint: str
    support_fingerprint: str
    action_envelope_record_digest: str
    action_provenance_fingerprint: str
    actions: tuple[PlannedResolutionAction, ...]
    partition: str = "train"

    def __post_init__(self) -> None:
        _text(self.base_lineage, "planned row base lineage")
        _text(self.task_id, "planned row task ID")
        _text(self.instance_id, "planned row instance ID")
        _nonnegative_int(self.lineage_schedule_index, "planned lineage schedule index")
        _nonnegative_int(self.state_schedule_index, "planned state schedule index")
        if self.partition != "train":
            raise QualityResolutionError("quality-resolution rows are train-only")
        _nonnegative_int(self.environment_seed, "planned environment seed", maximum=2**63)
        _nonnegative_int(
            self.initializer_bank_episode_index,
            "planned initializer-bank episode index",
        )
        _digest(
            self.initializer_bootstrap_record_digest,
            "planned initializer bootstrap record digest",
        )
        for position, index in enumerate(self.prefix):
            _nonnegative_int(index, f"planned prefix index {position}")
        _digest(self.state_fingerprint, "planned state fingerprint")
        _digest(self.support_fingerprint, "planned support fingerprint")
        _digest(
            self.action_envelope_record_digest,
            "planned action-envelope record digest",
        )
        _digest(
            self.action_provenance_fingerprint,
            "planned action-provenance fingerprint",
        )
        if not self.actions:
            raise QualityResolutionError("planned row must contain at least one action")
        indices = tuple(action.action_index for action in self.actions)
        if indices != tuple(sorted(set(indices))):
            raise QualityResolutionError("planned row actions must have unique sorted indices")

    @classmethod
    def create(cls, **values: object) -> PlannedResolutionRow:
        return cls(**values)  # type: ignore[arg-type]

    def payload(self) -> dict[str, object]:
        return {
            "action_envelope_record_digest": self.action_envelope_record_digest,
            "action_provenance_fingerprint": self.action_provenance_fingerprint,
            "actions": [action.as_dict() for action in self.actions],
            "base_lineage": self.base_lineage,
            "environment_seed": self.environment_seed,
            "initializer_bank_episode_index": self.initializer_bank_episode_index,
            "initializer_bootstrap_record_digest": (self.initializer_bootstrap_record_digest),
            "instance_id": self.instance_id,
            "lineage_schedule_index": self.lineage_schedule_index,
            "partition": self.partition,
            "prefix": list(self.prefix),
            "schema": QUALITY_RESOLUTION_ROW_SCHEMA,
            "schema_version": QUALITY_RESOLUTION_ROW_VERSION,
            "state_fingerprint": self.state_fingerprint,
            "state_schedule_index": self.state_schedule_index,
            "support_fingerprint": self.support_fingerprint,
            "task_id": self.task_id,
        }

    @property
    def row_id(self) -> str:
        return content_digest(self.payload())

    def as_dict(self) -> dict[str, object]:
        return {**self.payload(), "row_id": self.row_id}


@dataclass(frozen=True, slots=True)
class PlannedResolutionLineage:
    """One independent all-train lineage, retained even if it has no replayable row."""

    base_lineage: str
    schedule_index: int
    task_ids: tuple[str, ...]
    stratum: tuple[tuple[str, str | bool], ...]

    def __post_init__(self) -> None:
        _text(self.base_lineage, "planned base lineage")
        _nonnegative_int(self.schedule_index, "planned lineage schedule index")
        if not self.task_ids or any(type(value) is not str or not value for value in self.task_ids):
            raise QualityResolutionError("planned lineage task IDs must be nonempty text")
        if self.task_ids != tuple(sorted(set(self.task_ids))):
            raise QualityResolutionError("planned lineage task IDs must be unique and sorted")
        if tuple(name for name, _value in self.stratum) != _STRATUM_FIELDS:
            raise QualityResolutionError("planned lineage stratum schema differs")
        if any(
            type(value) not in {str, bool} or (type(value) is str and not value)
            for _, value in self.stratum
        ):
            raise QualityResolutionError("planned lineage stratum contains an invalid level")

    def as_dict(self, *, row_ids: Sequence[str]) -> dict[str, object]:
        return {
            "base_lineage": self.base_lineage,
            "row_ids": list(row_ids),
            "schedule_index": self.schedule_index,
            "stratum": {name: value for name, value in self.stratum},
            "task_ids": list(self.task_ids),
        }


@dataclass(frozen=True, slots=True)
class ResolutionPreparedCensus:
    """Exact target-free all-train task registry committed by the prepared manifest."""

    prepared_manifest_sha256: str
    lineage_tasks: tuple[tuple[str, tuple[str, ...]], ...]

    def __post_init__(self) -> None:
        _digest(self.prepared_manifest_sha256, "prepared census manifest SHA-256")
        if not self.lineage_tasks:
            raise QualityResolutionError("prepared train census has no lineages")
        lineages = tuple(lineage for lineage, _tasks in self.lineage_tasks)
        if lineages != tuple(sorted(set(lineages))):
            raise QualityResolutionError("prepared train census lineages must be unique and sorted")
        seen: set[str] = set()
        for lineage, task_ids in self.lineage_tasks:
            _text(lineage, "prepared census base lineage")
            if not task_ids or task_ids != tuple(sorted(set(task_ids))):
                raise QualityResolutionError(
                    "prepared train census task IDs must be nonempty, unique, and sorted"
                )
            if seen.intersection(task_ids):
                raise QualityResolutionError("prepared train census repeats a task ID")
            seen.update(task_ids)

    def payload(self) -> dict[str, object]:
        entries = [
            {"base_lineage": lineage, "task_ids": list(task_ids)}
            for lineage, task_ids in self.lineage_tasks
        ]
        return {
            "lineage_count": len(entries),
            "lineages": entries,
            "partition": "train",
            "prepared_manifest_sha256": self.prepared_manifest_sha256,
            "schema": "isingfold.quality-resolution-prepared-train-census",
            "schema_version": 1,
            "task_count": sum(len(task_ids) for _lineage, task_ids in self.lineage_tasks),
        }

    @property
    def record_digest(self) -> str:
        return content_digest(self.payload())

    def as_dict(self) -> dict[str, object]:
        return {**self.payload(), "record_digest": self.record_digest}


@dataclass(frozen=True, slots=True)
class ResolutionProductionPlan:
    """Complete all-train projection from which the study sample is selected."""

    lineages: tuple[PlannedResolutionLineage, ...]
    rows: tuple[PlannedResolutionRow, ...]
    source_census: ResolutionPreparedCensus
    initializer_bank_contract: Mapping[str, object]

    def _canonical_parts(
        self,
    ) -> tuple[tuple[PlannedResolutionLineage, ...], tuple[PlannedResolutionRow, ...]]:
        if not self.lineages:
            raise QualityResolutionError("quality production plan has no lineages")
        try:
            validate_quality_initializer_bank_contract(
                self.initializer_bank_contract,
                require_publication=False,
            )
        except (TypeError, ValueError) as error:
            raise QualityResolutionError(str(error)) from error
        lineages = tuple(sorted(self.lineages, key=lambda item: item.schedule_index))
        rows = tuple(
            sorted(
                self.rows,
                key=lambda item: (
                    item.lineage_schedule_index,
                    item.state_schedule_index,
                    item.row_id,
                ),
            )
        )
        lineage_ids = [item.base_lineage for item in lineages]
        if len(set(lineage_ids)) != len(lineage_ids):
            raise QualityResolutionError("quality production plan has duplicate lineages")
        if tuple(item.schedule_index for item in lineages) != tuple(range(len(lineages))):
            raise QualityResolutionError(
                "quality production lineage schedule must be a complete zero-based registry"
            )
        task_owner: dict[str, str] = {}
        lineage_tasks = {item.base_lineage: set(item.task_ids) for item in lineages}
        for lineage in lineages:
            for task_id in lineage.task_ids:
                if task_id in task_owner:
                    raise QualityResolutionError("quality production plan has duplicate task IDs")
                task_owner[task_id] = lineage.base_lineage
        row_ids = [row.row_id for row in rows]
        if len(set(row_ids)) != len(row_ids):
            raise QualityResolutionError("quality production plan has duplicate rows")
        for row in rows:
            if row.base_lineage not in lineage_tasks:
                raise QualityResolutionError("planned row refers to an unknown base lineage")
            if row.task_id not in lineage_tasks[row.base_lineage]:
                raise QualityResolutionError("planned row refers to another lineage's task")
            lineage = lineages[row.lineage_schedule_index]
            if lineage.base_lineage != row.base_lineage:
                raise QualityResolutionError("planned row lineage schedule index is inconsistent")
            start = self.initializer_bank_contract["episode_schedule_start"]
            stop = self.initializer_bank_contract["episode_schedule_stop_exclusive"]
            if not start <= row.initializer_bank_episode_index < stop:
                raise QualityResolutionError(
                    "planned row initializer episode is outside the sealed bank"
                )
        for lineage in lineages:
            state_indices = sorted(
                row.state_schedule_index for row in rows if row.base_lineage == lineage.base_lineage
            )
            if state_indices != list(range(len(state_indices))):
                raise QualityResolutionError(
                    "quality production state schedule must be complete within each lineage"
                )
        return lineages, rows

    def payload(self) -> dict[str, object]:
        lineages, rows = self._canonical_parts()
        rows_by_lineage: dict[str, list[str]] = {item.base_lineage: [] for item in lineages}
        for row in rows:
            rows_by_lineage[row.base_lineage].append(row.row_id)
        return {
            "independent_unit": "immutable-base-lineage",
            "initializer_bank": dict(self.initializer_bank_contract),
            "initializer_episode_assignment_rule": (QUALITY_INITIALIZER_EPISODE_ASSIGNMENT_RULE),
            "lineage_count": len(lineages),
            "lineages": [
                lineage.as_dict(row_ids=sorted(rows_by_lineage[lineage.base_lineage]))
                for lineage in lineages
            ],
            "partition": "train",
            "row_count": len(rows),
            "rows": [row.as_dict() for row in rows],
            "schema": QUALITY_RESOLUTION_PRODUCTION_PLAN_SCHEMA,
            "schema_version": QUALITY_RESOLUTION_PRODUCTION_PLAN_VERSION,
            "source_census": self.source_census.as_dict(),
            "task_count": sum(len(item.task_ids) for item in lineages),
        }

    @property
    def record_digest(self) -> str:
        return content_digest(self.payload())

    def as_dict(self) -> dict[str, object]:
        return {**self.payload(), "record_digest": self.record_digest}


@dataclass(frozen=True, slots=True)
class QualityResolutionPlanAuthority:
    """Target-free authenticated identities sealed into a resolution plan."""

    prepared_manifest_sha256: str
    prepared_manifest_record_digest: str
    prepared_train_census_record_digest: str
    corpus_design_manifest_sha256: str
    publisher_id: str
    publisher_attestation_sha256: str
    publisher_attestation_record_digest: str
    target_authority_record_digest: str
    ground_root_sha256: str
    ground_root_record_digest: str
    verifier_identity_digest: str
    selector_digest: str
    selector_file_sha256: str
    selector_fit_receipt_sha256: str
    selector_fit_record_digest: str
    normalizer_digest: str
    selector_device: str
    selector_device_parity_sha256: str
    selector_device_parity_record_digest: str
    context: Mapping[str, object]
    context_digest: str
    implementation: Mapping[str, object]
    implementation_digest: str

    def __post_init__(self) -> None:
        for label, value in (
            ("prepared manifest SHA-256", self.prepared_manifest_sha256),
            ("prepared manifest record digest", self.prepared_manifest_record_digest),
            (
                "prepared train census record digest",
                self.prepared_train_census_record_digest,
            ),
            ("corpus-design manifest SHA-256", self.corpus_design_manifest_sha256),
            ("publisher attestation SHA-256", self.publisher_attestation_sha256),
            ("publisher attestation record digest", self.publisher_attestation_record_digest),
            ("target-authority record digest", self.target_authority_record_digest),
            ("ground-root SHA-256", self.ground_root_sha256),
            ("ground-root record digest", self.ground_root_record_digest),
            ("verifier identity digest", self.verifier_identity_digest),
            ("selector digest", self.selector_digest),
            ("selector file SHA-256", self.selector_file_sha256),
            ("selector-fit receipt SHA-256", self.selector_fit_receipt_sha256),
            ("selector-fit record digest", self.selector_fit_record_digest),
            ("normalizer digest", self.normalizer_digest),
            ("selector-device parity SHA-256", self.selector_device_parity_sha256),
            ("selector-device parity record digest", self.selector_device_parity_record_digest),
            ("context digest", self.context_digest),
            ("implementation digest", self.implementation_digest),
        ):
            _digest(value, label)
        _text(self.publisher_id, "publisher ID")
        if self.selector_device not in {"cpu", "cuda"}:
            raise QualityResolutionError("quality selector device must be cpu or cuda")
        frozen_context = _freeze(self.context)
        frozen_implementation = _freeze(self.implementation)
        if not isinstance(frozen_implementation, Mapping) or set(frozen_implementation) != {
            "planner",
            "quality_implementation_contract_digest",
            "quality_module_sha256",
            "quality_resolution_delta_module_sha256",
        }:
            raise QualityResolutionError(
                "quality-resolution implementation registry schema differs"
            )
        if frozen_implementation.get("planner") != "quality-resolution-plan-v1":
            raise QualityResolutionError("quality-resolution planner identity differs")
        for label, key in (
            (
                "quality implementation contract digest",
                "quality_implementation_contract_digest",
            ),
            ("quality module SHA-256", "quality_module_sha256"),
            (
                "quality-resolution delta module SHA-256",
                "quality_resolution_delta_module_sha256",
            ),
        ):
            _digest(frozen_implementation.get(key), label)
        if content_digest(_thaw(frozen_context)) != self.context_digest:
            raise QualityResolutionError("quality-resolution context digest mismatch")
        if content_digest(_thaw(frozen_implementation)) != self.implementation_digest:
            raise QualityResolutionError("quality-resolution implementation digest mismatch")
        object.__setattr__(self, "context", frozen_context)
        object.__setattr__(self, "implementation", frozen_implementation)

    def sections(self) -> dict[str, object]:
        return {
            "prepared_corpus": {
                "corpus_design_manifest_sha256": self.corpus_design_manifest_sha256,
                "manifest_record_digest": self.prepared_manifest_record_digest,
                "manifest_sha256": self.prepared_manifest_sha256,
                "schema_version": 4,
                "train_census_record_digest": self.prepared_train_census_record_digest,
            },
            "publisher": {
                "attestation_record_digest": self.publisher_attestation_record_digest,
                "attestation_sha256": self.publisher_attestation_sha256,
                "publisher_id": self.publisher_id,
                "target_authority_record_digest": self.target_authority_record_digest,
            },
            "ground_root": {
                "record_digest": self.ground_root_record_digest,
                "sha256": self.ground_root_sha256,
                "verifier_identity_digest": self.verifier_identity_digest,
            },
            "selector": {
                "device": self.selector_device,
                "device_parity_record_digest": self.selector_device_parity_record_digest,
                "device_parity_sha256": self.selector_device_parity_sha256,
                "fit_receipt_record_digest": self.selector_fit_record_digest,
                "fit_receipt_sha256": self.selector_fit_receipt_sha256,
                "normalizer_digest": self.normalizer_digest,
                "selector_digest": self.selector_digest,
                "selector_file_sha256": self.selector_file_sha256,
            },
            "context": {
                "digest": self.context_digest,
                "snapshot": _thaw(self.context),
            },
            "implementation": {
                "digest": self.implementation_digest,
                "registry": _thaw(self.implementation),
            },
        }


@dataclass(frozen=True, slots=True)
class QualityResolutionPlan:
    """Read-only canonical plan record."""

    record: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "record", _freeze(self.record))

    def as_dict(self) -> dict[str, Any]:
        result = _thaw(self.record)
        if not isinstance(result, dict):  # pragma: no cover - construction is mapping-only
            raise RuntimeError("quality-resolution plan lost its object shape")
        return result


def _quality_protocol_seed(root: int, domain: str, *parts: object) -> int:
    """Match the registered quality-v7 environment and sampling seed framing."""

    _nonnegative_int(root, "quality protocol seed", maximum=2**63)
    _text(domain, "quality protocol seed domain")
    raw = canonical_json_bytes({"domain": domain, "parts": [root, *parts], "schema_version": 1})
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & (2**31 - 1)


def _validate_public_prepared_task(item: PreparedTask) -> PreparedDesignCondition:
    if not isinstance(item, PreparedTask):
        raise TypeError("quality-resolution input must contain PreparedTask values")
    if item.partition != "train":
        raise QualityResolutionError("quality-resolution materialization is train-only")
    if item.prepared_schema_version != 4 or item.corpus_scope != "production-designed-v4":
        raise QualityResolutionError(
            "quality-resolution materialization requires production prepared-v4 tasks"
        )
    if item.task.name != item.task_id:
        raise QualityResolutionError("prepared task identity differs from its embedded task")
    evaluator_fields = (
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
    if any(value is not None for value in evaluator_fields):
        raise QualityResolutionError(
            "quality-resolution planning received an opened evaluator target"
        )
    lineage = item.task.lineage
    if type(lineage) is not str or not lineage:
        raise QualityResolutionError("prepared quality task has no base lineage")
    condition = item.design_condition
    if not isinstance(condition, PreparedDesignCondition):
        raise QualityResolutionError("prepared quality task has no design condition")
    if condition.base_lineage_key != lineage or condition.learning_partition != "train":
        raise QualityResolutionError("prepared design condition crosses its train lineage boundary")
    if (
        item.provenance is not None
        and item.provenance.base_parent_lineage != lineage
    ):
        raise QualityResolutionError("prepared provenance crosses its train lineage boundary")
    _digest(item.initializer_record_digest, "prepared initializer record digest")
    return condition


def _condition_stratum(
    condition: PreparedDesignCondition,
) -> tuple[tuple[str, str | bool], ...]:
    values = tuple((field, getattr(condition, field)) for field in _STRATUM_FIELDS)
    if any(
        type(value) not in {str, bool} or (type(value) is str and not value) for _, value in values
    ):
        raise QualityResolutionError("prepared design condition has an invalid stratum level")
    return values


def quality_action_provenance_fingerprint(
    item: PreparedTask,
    source_corpus_manifest_sha256: str,
) -> str:
    condition = item.design_condition
    provenance = item.provenance
    if not isinstance(condition, PreparedDesignCondition) or provenance is None:
        raise QualityResolutionError(
            "quality action provenance requires authenticated prepared provenance"
        )
    source_record_digest = stable_digest(
        {
            "task_id": item.task_id,
            "initializer_record_digest": item.initializer_record_digest,
            "provenance_record_digest": provenance.record_digest,
            "design_condition_digest": condition.registry_row_digest,
        }
    )
    return content_digest(
        {
            "schema": "isingfold.quality-action-provenance",
            "schema_version": 2,
            "source_corpus_manifest_sha256": source_corpus_manifest_sha256,
            "task_id": item.task_id,
            "instance_id": item.instance_id,
            "partition": item.partition,
            "initializer_record_digest": item.initializer_record_digest,
            "prepared_provenance_record_digest": provenance.record_digest,
            "design_condition_registry_row_digest": condition.registry_row_digest,
            "initializer_bank_source_record_digest": source_record_digest,
            "certificate_digest": None,
            "evaluator_protocol_digest": None,
        }
    )


def _initializer_episode_assignment(
    item: PreparedTask,
    *,
    quality_seed: int,
    scheduled_episodes: Sequence[int],
) -> int:
    """Assign one task to a sealed episode without always favoring the first episode."""

    ordered = tuple(sorted(scheduled_episodes))
    if not ordered:
        raise QualityResolutionError("planned quality task has no sealed initializer episode")
    position = _quality_protocol_seed(
        quality_seed,
        "quality-initializer-episode-assignment",
        QUALITY_INITIALIZER_EPISODE_ASSIGNMENT_RULE,
        item.task.lineage,
        item.instance_id,
        item.task_id,
    ) % len(ordered)
    return ordered[position]


def _quality_prefixes(
    item: PreparedTask,
    context: Context,
    selector: StrengthSelector,
    *,
    count: int,
    seed: int,
    initializer_bank: InitializerSnapshotBank,
    expected_initializer_bank_manifest_sha256: str,
    initializer_bank_episode_index: int,
    allow_test_initializer_bank: bool,
) -> tuple[tuple[int, ...], ...]:
    quality_initializer_bank_episode_binding(
        initializer_bank,
        expected_manifest_sha256=expected_initializer_bank_manifest_sha256,
        episode_schedule_index=initializer_bank_episode_index,
        task=item.task,
        ctx=context,
        allow_test_bank=allow_test_initializer_bank,
    )
    bootstrap = initializer_bank.bootstrap_outcome(initializer_bank_episode_index)
    if seed != bootstrap.initial_snapshot.system_seed:
        raise QualityResolutionError(
            "quality prefix seed differs from its initializer-bank episode"
        )
    environment = initializer_bank.environment(
        initializer_bank_episode_index,
        context=context,
        selector=selector,
        reward_reads=context.n_est_reads,
    )
    result = environment.reset(seed)
    prefix: list[int] = []
    prefixes: list[tuple[int, ...]] = []
    rng = np.random.default_rng(_quality_protocol_seed(seed, "quality-prefix-policy", item.task_id))
    while isinstance(result, DecisionState) and len(prefixes) < count:
        prefixes.append(tuple(prefix))
        continuing = [
            index
            for index, (candidate, legal) in enumerate(
                zip(result.candidates, result.legal_mask, strict=True)
            )
            if legal and not candidate.opcode.is_terminal
        ]
        if not continuing:
            break
        chosen = int(rng.choice(continuing))
        prefix.append(chosen)
        result = environment.step(
            result, chosen, evaluate_training_reward=False
        ).next_decision_or_terminal
    return tuple(prefixes)


def materialize_resolution_production_plan(
    prepared: Sequence[PreparedTask],
    *,
    context: Context,
    selector: StrengthSelector,
    config: QualityResolutionConfig,
    source_corpus_manifest_sha256: str,
    initializer_bank: InitializerSnapshotBank | None = None,
    expected_initializer_bank_manifest_sha256: str | None = None,
    allow_test_initializer_bank: bool = False,
) -> ResolutionProductionPlan:
    """Materialize all-train states from a pinned target-free deployment bank."""

    if isinstance(prepared, (str, bytes)) or not isinstance(prepared, Sequence) or not prepared:
        raise QualityResolutionError("quality-resolution materialization needs public tasks")
    if not isinstance(context, Context):
        raise TypeError("context must be Context")
    if not isinstance(config, QualityResolutionConfig):
        raise TypeError("config must be QualityResolutionConfig")
    source_sha256 = _digest(source_corpus_manifest_sha256, "source corpus manifest SHA-256")
    if context.n_est_reads != config.reward_reads:
        raise QualityResolutionError(
            "quality-resolution context reward reads differ from the registered config"
        )

    grouped: dict[str, list[tuple[PreparedTask, PreparedDesignCondition]]] = {}
    task_ids: set[str] = set()
    for item in prepared:
        condition = _validate_public_prepared_task(item)
        if item.task_id in task_ids:
            raise QualityResolutionError("public quality input has duplicate task IDs")
        task_ids.add(item.task_id)
        grouped.setdefault(condition.base_lineage_key, []).append((item, condition))

    if initializer_bank is None or expected_initializer_bank_manifest_sha256 is None:
        raise QualityResolutionError(
            "quality-resolution materialization requires a pinned initializer bank"
        )
    try:
        bank_contract = quality_initializer_bank_contract(
            initializer_bank,
            expected_manifest_sha256=expected_initializer_bank_manifest_sha256,
            allow_test_bank=allow_test_initializer_bank,
        )
    except (TypeError, ValueError) as error:
        raise QualityResolutionError(str(error)) from error
    if bank_contract["prepared_manifest_sha256"] != source_sha256:
        raise QualityResolutionError("quality initializer bank belongs to another prepared corpus")
    if bank_contract["context_digest"] != quality_context_digest(context):
        raise QualityResolutionError("quality initializer bank belongs to another Context")

    bank_tasks = {identity.instance_id: identity for identity in initializer_bank.plan.tasks}
    episodes_by_instance: dict[str, list[int]] = {}
    for episode in initializer_bank.plan.episodes:
        episodes_by_instance.setdefault(episode.instance_id, []).append(
            episode.episode_schedule_index
        )

    ordered_lineages = sorted(
        grouped,
        key=lambda lineage: (
            _quality_protocol_seed(config.quality_seed, "quality-lineage-sample", lineage),
            lineage,
        ),
    )
    lineages: list[PlannedResolutionLineage] = []
    rows: list[PlannedResolutionRow] = []
    for lineage_schedule_index, lineage in enumerate(ordered_lineages):
        ordered_tasks = sorted(
            grouped[lineage],
            key=lambda pair: (
                _quality_protocol_seed(
                    config.quality_seed,
                    "quality-task-sample",
                    lineage,
                    pair[0].task_id,
                ),
                pair[0].task_id,
            ),
        )
        active = ordered_tasks[: config.tasks_per_lineage_cap]
        if not active:
            raise RuntimeError("validated production lineage lost all public tasks")
        lineages.append(
            PlannedResolutionLineage(
                base_lineage=lineage,
                schedule_index=lineage_schedule_index,
                task_ids=tuple(sorted(item.task_id for item, _condition in active)),
                stratum=_condition_stratum(active[0][1]),
            )
        )

        state_schedule_index = 0
        base, extra = divmod(config.states_per_lineage_cap, len(active))
        for task_position, (item, _condition) in enumerate(active):
            state_cap = base + (1 if task_position < extra else 0)
            bank_identity = bank_tasks.get(item.instance_id)
            scheduled_episodes = episodes_by_instance.get(item.instance_id, ())
            if (
                bank_identity is None
                or item.task_id not in bank_identity.source_task_ids
                or bank_identity.base_lineage != lineage
                or not scheduled_episodes
            ):
                raise QualityResolutionError(
                    "pinned initializer bank does not cover a planned quality task"
                )
            if item.provenance is None:
                raise QualityResolutionError(
                    "planned quality task has no authenticated prepared provenance"
                )
            bank_sources = dict(
                zip(
                    bank_identity.source_task_ids,
                    bank_identity.source_record_digests,
                    strict=True,
                )
            )
            expected_source_digest = stable_digest(
                {
                    "task_id": item.task_id,
                    "initializer_record_digest": item.initializer_record_digest,
                    "provenance_record_digest": item.provenance.record_digest,
                    "design_condition_digest": item.design_condition.registry_row_digest,
                }
            )
            if bank_sources.get(item.task_id) != expected_source_digest:
                raise QualityResolutionError(
                    "prepared quality task differs from its authenticated bank source identity"
                )
            initializer_bank_episode_index = _initializer_episode_assignment(
                item,
                quality_seed=config.quality_seed,
                scheduled_episodes=scheduled_episodes,
            )
            try:
                initializer_binding = quality_initializer_bank_episode_binding(
                    initializer_bank,
                    expected_manifest_sha256=(expected_initializer_bank_manifest_sha256),
                    episode_schedule_index=initializer_bank_episode_index,
                    task=item.task,
                    ctx=context,
                    allow_test_bank=allow_test_initializer_bank,
                )
            except (TypeError, ValueError) as error:
                raise QualityResolutionError(str(error)) from error
            bootstrap = initializer_bank.bootstrap_outcome(initializer_bank_episode_index)
            environment_seed = bootstrap.initial_snapshot.system_seed
            provenance_fingerprint = quality_action_provenance_fingerprint(item, source_sha256)
            prefixes = _quality_prefixes(
                item,
                context,
                selector,
                count=state_cap,
                seed=environment_seed,
                initializer_bank=initializer_bank,
                expected_initializer_bank_manifest_sha256=(
                    expected_initializer_bank_manifest_sha256
                ),
                initializer_bank_episode_index=initializer_bank_episode_index,
                allow_test_initializer_bank=allow_test_initializer_bank,
            )
            for prefix in prefixes:
                replayed = replay_decision_with_action_envelope(
                    item.task,
                    context,
                    initializer=None,
                    initializer_bank=initializer_bank,
                    expected_initializer_bank_manifest_sha256=(
                        expected_initializer_bank_manifest_sha256
                    ),
                    initializer_bank_episode_index=initializer_bank_episode_index,
                    allow_test_initializer_bank=allow_test_initializer_bank,
                    selector=selector,
                    prefix=prefix,
                    seed=environment_seed,
                    reward_reads=config.reward_reads,
                    provenance_fingerprint=provenance_fingerprint,
                )
                if replayed is None:
                    continue
                decision, envelope = replayed
                action_sample = deterministic_action_sample(
                    decision,
                    evaluated_actions=config.evaluated_actions,
                    seed=environment_seed,
                )
                chosen = action_sample.action_indices
                actions: list[PlannedResolutionAction] = []
                for action_index in chosen:
                    bound_action = envelope.candidates[action_index]
                    applied_digest = None
                    if not bound_action.opcode.is_terminal:
                        applied = apply_envelope_action(envelope, action_index)
                        applied.verify_against(envelope)
                        applied_digest = applied.record_digest
                    actions.append(
                        PlannedResolutionAction(
                            action_index=action_index,
                            applied_action_record_digest=applied_digest,
                            continuation_seeds=tuple(
                                continuation_seed(
                                    environment_seed,
                                    item.task_id,
                                    decision.state_fingerprint,
                                    action_index,
                                    continuation_index,
                                )
                                for continuation_index in range(max(config.candidate_continuations))
                            ),
                            opcode=bound_action.opcode.value,
                            payload_key=bound_action.payload_key,
                            selected_payload_digest=bound_action.payload_digest,
                        )
                    )
                rows.append(
                    PlannedResolutionRow(
                        action_envelope_record_digest=envelope.record_digest,
                        action_provenance_fingerprint=provenance_fingerprint,
                        actions=tuple(actions),
                        base_lineage=lineage,
                        environment_seed=environment_seed,
                        initializer_bank_episode_index=(initializer_bank_episode_index),
                        initializer_bootstrap_record_digest=initializer_binding[
                            "bootstrap_record_digest"
                        ],
                        instance_id=item.instance_id,
                        lineage_schedule_index=lineage_schedule_index,
                        prefix=tuple(prefix),
                        state_fingerprint=decision.state_fingerprint,
                        state_schedule_index=state_schedule_index,
                        support_fingerprint=decision.support_fingerprint,
                        task_id=item.task_id,
                    )
                )
                state_schedule_index += 1
    source_census = ResolutionPreparedCensus(
        prepared_manifest_sha256=source_sha256,
        lineage_tasks=tuple(
            (
                lineage,
                tuple(sorted(item.task_id for item, _condition in grouped[lineage])),
            )
            for lineage in sorted(grouped)
        ),
    )
    return ResolutionProductionPlan(
        lineages=tuple(lineages),
        rows=tuple(rows),
        source_census=source_census,
        initializer_bank_contract=bank_contract,
    )


def _validated_production(
    production: ResolutionProductionPlan,
    config: QualityResolutionConfig,
    *,
    prepared_manifest_sha256: str,
    prepared_train_census_record_digest: str,
) -> ResolutionProductionPlan:
    if not isinstance(production, ResolutionProductionPlan):
        raise TypeError("production must be a ResolutionProductionPlan")
    lineages, rows = production._canonical_parts()
    try:
        validate_quality_initializer_bank_contract(
            production.initializer_bank_contract,
            require_publication=True,
        )
    except (TypeError, ValueError) as error:
        raise QualityResolutionError(str(error)) from error
    if production.initializer_bank_contract["prepared_manifest_sha256"] != prepared_manifest_sha256:
        raise QualityResolutionError("quality initializer bank belongs to another prepared corpus")
    if len(lineages) < MINIMUM_PRODUCTION_LINEAGES:
        raise QualityResolutionError(
            "quality-resolution production plan requires at least "
            f"{MINIMUM_PRODUCTION_LINEAGES} independent train lineages"
        )
    if len(lineages) < config.study_lineages:
        raise QualityResolutionError(
            "quality-resolution production population is smaller than its study sample"
        )
    census = production.source_census
    if (
        census.prepared_manifest_sha256 != prepared_manifest_sha256
        or census.record_digest != prepared_train_census_record_digest
    ):
        raise QualityResolutionError(
            "quality production source census differs from prepared authority"
        )
    expected_source = dict(census.lineage_tasks)
    expected_lineage_order = sorted(
        expected_source,
        key=lambda lineage: (
            _quality_protocol_seed(config.quality_seed, "quality-lineage-sample", lineage),
            lineage,
        ),
    )
    if [lineage.base_lineage for lineage in lineages] != expected_lineage_order:
        raise QualityResolutionError(
            "quality production plan is not the exact all-train lineage census"
        )
    for lineage in lineages:
        ordered_tasks = sorted(
            expected_source[lineage.base_lineage],
            key=lambda task_id: (
                _quality_protocol_seed(
                    config.quality_seed,
                    "quality-task-sample",
                    lineage.base_lineage,
                    task_id,
                ),
                task_id,
            ),
        )
        expected_active = tuple(sorted(ordered_tasks[: config.tasks_per_lineage_cap]))
        if lineage.task_ids != expected_active:
            raise QualityResolutionError(
                "quality production tasks are not the registered all-train projection"
            )
    row_counts: Counter[str] = Counter()
    all_seeds: set[int] = set()
    expected_continuations = max(config.candidate_continuations)
    for lineage in lineages:
        if len(lineage.task_ids) > config.tasks_per_lineage_cap:
            raise QualityResolutionError(
                "quality production projection exceeds tasks_per_lineage_cap"
            )
    for row in rows:
        if row.partition != config.partition:
            raise QualityResolutionError("quality-resolution rows are train-only")
        row_counts[row.base_lineage] += 1
        if len(row.actions) > config.evaluated_actions:
            raise QualityResolutionError("planned row exceeds the registered evaluated-action cap")
        for action in row.actions:
            if len(action.continuation_seeds) != expected_continuations:
                raise QualityResolutionError(
                    "planned action continuation seed registry has the wrong length"
                )
            expected_seeds = tuple(
                continuation_seed(
                    row.environment_seed,
                    row.task_id,
                    row.state_fingerprint,
                    action.action_index,
                    continuation_index,
                )
                for continuation_index in range(expected_continuations)
            )
            if action.continuation_seeds != expected_seeds:
                raise QualityResolutionError(
                    "planned action continuation seed registry differs from quality.py"
                )
            overlap = all_seeds.intersection(action.continuation_seeds)
            if overlap:
                raise QualityResolutionError(
                    "quality-resolution continuation seed registry contains a collision"
                )
            all_seeds.update(action.continuation_seeds)
    if any(count > config.states_per_lineage_cap for count in row_counts.values()):
        raise QualityResolutionError("quality production projection exceeds states_per_lineage_cap")
    return ResolutionProductionPlan(
        lineages=lineages,
        rows=rows,
        source_census=census,
        initializer_bank_contract=production.initializer_bank_contract,
    )


def _stratum_census(
    lineages: Sequence[PlannedResolutionLineage],
) -> dict[str, list[dict[str, object]]]:
    census: dict[str, list[dict[str, object]]] = {}
    for position, field in enumerate(_STRATUM_FIELDS):
        counts = Counter(lineage.stratum[position][1] for lineage in lineages)
        ordered = sorted(counts, key=canonical_json_bytes)
        census[field] = [{"base_lineage_count": counts[level], "level": level} for level in ordered]
    return census


def _schedule_sections(
    production: ResolutionProductionPlan,
    inputs: ResolutionPlanningInputs,
    prepared_manifest_sha256: str,
) -> tuple[
    dict[str, object], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]
]:
    config = inputs.config
    lineages, rows = production._canonical_parts()
    lineage_by_id = {lineage.base_lineage: lineage for lineage in lineages}
    rows_by_lineage: dict[str, list[str]] = {lineage.base_lineage: [] for lineage in lineages}
    for row in rows:
        rows_by_lineage[row.base_lineage].append(row.row_id)
    selected_ids = select_resolution_lineages(
        (lineage.base_lineage for lineage in lineages),
        sample_size=config.study_lineages,
        sampling_seed=config.sampling_seed,
        source_corpus_manifest_sha256=prepared_manifest_sha256,
        production_plan_digest=production.record_digest,
    )
    sample = {
        "descriptive_stratum_census": _stratum_census(
            tuple(lineage_by_id[lineage] for lineage in selected_ids)
        ),
        "population_lineages": len(lineages),
        "population_plan_digest": production.record_digest,
        "rule": QUALITY_RESOLUTION_SAMPLE_RULE,
        "sample_size": config.study_lineages,
        "sampling_seed": config.sampling_seed,
        "selected_lineages": list(selected_ids),
        "selected_row_ids": sorted(
            row_id for lineage in selected_ids for row_id in rows_by_lineage[lineage]
        ),
        "selected_task_ids": sorted(
            task_id for lineage in selected_ids for task_id in lineage_by_id[lineage].task_ids
        ),
    }
    sample["record_digest"] = content_digest(sample)

    shard_count, remainder = divmod(config.study_lineages, config.lineages_per_shard)
    if remainder or shard_count != QUALITY_RESOLUTION_SHARD_COUNT:
        raise QualityResolutionError(
            "registered resolution sample must form exactly 64 equal whole-lineage shards"
        )
    shards: list[dict[str, object]] = []
    for shard_index in range(shard_count):
        assigned = tuple(selected_ids[shard_index::shard_count])
        if len(assigned) != config.lineages_per_shard:
            raise QualityResolutionError("resolution shard does not have exactly two lineages")
        task_ids = sorted(
            task_id for lineage in assigned for task_id in lineage_by_id[lineage].task_ids
        )
        row_ids = sorted(row_id for lineage in assigned for row_id in rows_by_lineage[lineage])
        shard = {
            "assignment": QUALITY_RESOLUTION_SHARD_RULE,
            "lineage_set_digest": content_digest(sorted(assigned)),
            "lineages": list(assigned),
            "row_ids": row_ids,
            "row_set_digest": content_digest(row_ids),
            "shard_count": shard_count,
            "shard_index": shard_index,
            "task_ids": task_ids,
            "task_set_digest": content_digest(task_ids),
        }
        shard["record_digest"] = content_digest(shard)
        shards.append(shard)

    stages = [
        {
            "continuation_count": delta.count,
            "continuation_range": [delta.lower, delta.upper],
            "stage_index": delta.stage_index,
        }
        for delta in config.continuation_deltas
    ]
    work_schedule: list[dict[str, object]] = []
    for stage in stages:
        for shard in shards:
            work = {
                "continuation_range": list(stage["continuation_range"]),
                "lineages": list(shard["lineages"]),
                "row_ids": list(shard["row_ids"]),
                "shard_index": shard["shard_index"],
                "stage_index": stage["stage_index"],
                "work_id": (
                    f"stage-{int(stage['stage_index']):02d}-shard-{int(shard['shard_index']):04d}"
                ),
            }
            work["record_digest"] = content_digest(work)
            work_schedule.append(work)
    return sample, shards, stages, work_schedule


def build_quality_resolution_plan(
    inputs: ResolutionPlanningInputs,
    authority: QualityResolutionPlanAuthority,
    production: ResolutionProductionPlan,
) -> QualityResolutionPlan:
    """Seal the complete all-train projection before selecting the 128-lineage study."""

    if not isinstance(inputs, ResolutionPlanningInputs):
        raise TypeError("inputs must be ResolutionPlanningInputs")
    if not isinstance(authority, QualityResolutionPlanAuthority):
        raise TypeError("authority must be QualityResolutionPlanAuthority")
    _assert_recursively_target_free(authority.sections())
    canonical_production = _validated_production(
        production,
        inputs.config,
        prepared_manifest_sha256=authority.prepared_manifest_sha256,
        prepared_train_census_record_digest=(authority.prepared_train_census_record_digest),
    )
    if canonical_production.initializer_bank_contract["context_digest"] != authority.context_digest:
        raise QualityResolutionError(
            "quality initializer bank Context differs from the planning authority"
        )
    sample, shards, stages, work_schedule = _schedule_sections(
        canonical_production,
        inputs,
        authority.prepared_manifest_sha256,
    )
    sections = authority.sections()
    payload: dict[str, object] = {
        "candidate_continuations": list(inputs.config.candidate_continuations),
        "config": {
            "raw_sha256": inputs.config_sha256,
            "registered": _config_dict(inputs.config),
        },
        "context": sections["context"],
        "ground_root": sections["ground_root"],
        "grid": {
            "quality_resolution": dict(inputs.grid_resolution_minima),
            "raw_sha256": inputs.grid_sha256,
            "schema": "isingfold.staged-grid",
            "schema_version": 2,
        },
        "implementation": sections["implementation"],
        "prepared_corpus": sections["prepared_corpus"],
        "production_plan": canonical_production.as_dict(),
        "production_plan_digest": canonical_production.record_digest,
        "publisher": sections["publisher"],
        "sample": sample,
        "schema": QUALITY_RESOLUTION_PLAN_SCHEMA,
        "schema_version": QUALITY_RESOLUTION_PLAN_VERSION,
        "selector": sections["selector"],
        "shards": shards,
        "stages": stages,
        "study_id": inputs.config.study_id,
        "work_schedule": work_schedule,
    }
    record = {**payload, "record_digest": content_digest(payload)}
    _assert_recursively_target_free(record)
    return QualityResolutionPlan(record)


def _action_from_dict(raw: object) -> PlannedResolutionAction:
    if not isinstance(raw, Mapping) or set(raw) != {
        "action_index",
        "applied_action_record_digest",
        "continuation_seeds",
        "opcode",
        "payload_key",
        "selected_payload_digest",
    }:
        raise QualityResolutionError("planned action schema differs")
    seeds = raw["continuation_seeds"]
    if not isinstance(seeds, list):
        raise QualityResolutionError("planned continuation seeds must be a list")
    return PlannedResolutionAction(
        action_index=raw["action_index"],  # type: ignore[arg-type]
        applied_action_record_digest=raw["applied_action_record_digest"],  # type: ignore[arg-type]
        continuation_seeds=tuple(seeds),
        opcode=raw["opcode"],  # type: ignore[arg-type]
        payload_key=raw["payload_key"],  # type: ignore[arg-type]
        selected_payload_digest=raw["selected_payload_digest"],  # type: ignore[arg-type]
    )


def _row_from_dict(raw: object) -> PlannedResolutionRow:
    expected = {
        "action_envelope_record_digest",
        "action_provenance_fingerprint",
        "actions",
        "base_lineage",
        "environment_seed",
        "initializer_bank_episode_index",
        "initializer_bootstrap_record_digest",
        "instance_id",
        "lineage_schedule_index",
        "partition",
        "prefix",
        "row_id",
        "schema",
        "schema_version",
        "state_fingerprint",
        "state_schedule_index",
        "support_fingerprint",
        "task_id",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise QualityResolutionError("planned row schema differs")
    if (
        raw["schema"] != QUALITY_RESOLUTION_ROW_SCHEMA
        or raw["schema_version"] != QUALITY_RESOLUTION_ROW_VERSION
        or not isinstance(raw["actions"], list)
        or not isinstance(raw["prefix"], list)
    ):
        raise QualityResolutionError("unsupported planned row")
    row = PlannedResolutionRow(
        action_envelope_record_digest=raw["action_envelope_record_digest"],  # type: ignore[arg-type]
        action_provenance_fingerprint=raw["action_provenance_fingerprint"],  # type: ignore[arg-type]
        actions=tuple(_action_from_dict(action) for action in raw["actions"]),
        base_lineage=raw["base_lineage"],  # type: ignore[arg-type]
        environment_seed=raw["environment_seed"],  # type: ignore[arg-type]
        initializer_bank_episode_index=raw["initializer_bank_episode_index"],  # type: ignore[arg-type]
        initializer_bootstrap_record_digest=raw["initializer_bootstrap_record_digest"],  # type: ignore[arg-type]
        instance_id=raw["instance_id"],  # type: ignore[arg-type]
        lineage_schedule_index=raw["lineage_schedule_index"],  # type: ignore[arg-type]
        partition=raw["partition"],  # type: ignore[arg-type]
        prefix=tuple(raw["prefix"]),
        state_fingerprint=raw["state_fingerprint"],  # type: ignore[arg-type]
        state_schedule_index=raw["state_schedule_index"],  # type: ignore[arg-type]
        support_fingerprint=raw["support_fingerprint"],  # type: ignore[arg-type]
        task_id=raw["task_id"],  # type: ignore[arg-type]
    )
    if raw["row_id"] != row.row_id:
        raise QualityResolutionError("planned row identity digest mismatch")
    return row


def _production_from_dict(raw: object) -> ResolutionProductionPlan:
    expected = {
        "independent_unit",
        "initializer_bank",
        "initializer_episode_assignment_rule",
        "lineage_count",
        "lineages",
        "partition",
        "record_digest",
        "row_count",
        "rows",
        "schema",
        "schema_version",
        "source_census",
        "task_count",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected:
        raise QualityResolutionError("quality production-plan schema differs")
    if (
        raw["schema"] != QUALITY_RESOLUTION_PRODUCTION_PLAN_SCHEMA
        or raw["schema_version"] != QUALITY_RESOLUTION_PRODUCTION_PLAN_VERSION
        or raw["partition"] != "train"
        or raw["independent_unit"] != "immutable-base-lineage"
        or raw["initializer_episode_assignment_rule"] != QUALITY_INITIALIZER_EPISODE_ASSIGNMENT_RULE
        or not isinstance(raw["lineages"], list)
        or not isinstance(raw["rows"], list)
    ):
        raise QualityResolutionError("unsupported quality production plan")
    lineages: list[PlannedResolutionLineage] = []
    for lineage_raw in raw["lineages"]:
        if not isinstance(lineage_raw, Mapping) or set(lineage_raw) != {
            "base_lineage",
            "row_ids",
            "schedule_index",
            "stratum",
            "task_ids",
        }:
            raise QualityResolutionError("planned lineage schema differs")
        if not isinstance(lineage_raw["stratum"], Mapping):
            raise QualityResolutionError("planned lineage stratum must be an object")
        lineages.append(
            PlannedResolutionLineage(
                base_lineage=lineage_raw["base_lineage"],  # type: ignore[arg-type]
                schedule_index=lineage_raw["schedule_index"],  # type: ignore[arg-type]
                task_ids=tuple(lineage_raw["task_ids"]),  # type: ignore[arg-type]
                stratum=tuple(
                    (field, lineage_raw["stratum"].get(field)) for field in _STRATUM_FIELDS
                ),  # type: ignore[arg-type]
            )
        )
    census_raw = raw["source_census"]
    census_fields = {
        "lineage_count",
        "lineages",
        "partition",
        "prepared_manifest_sha256",
        "record_digest",
        "schema",
        "schema_version",
        "task_count",
    }
    if (
        not isinstance(census_raw, Mapping)
        or set(census_raw) != census_fields
        or census_raw.get("schema") != "isingfold.quality-resolution-prepared-train-census"
        or census_raw.get("schema_version") != 1
        or census_raw.get("partition") != "train"
        or not isinstance(census_raw.get("lineages"), list)
    ):
        raise QualityResolutionError("prepared train census schema differs")
    census_entries: list[tuple[str, tuple[str, ...]]] = []
    for entry in census_raw["lineages"]:
        if (
            not isinstance(entry, Mapping)
            or set(entry)
            != {
                "base_lineage",
                "task_ids",
            }
            or not isinstance(entry["task_ids"], list)
        ):
            raise QualityResolutionError("prepared train census lineage schema differs")
        census_entries.append((entry["base_lineage"], tuple(entry["task_ids"])))
    census = ResolutionPreparedCensus(
        prepared_manifest_sha256=census_raw["prepared_manifest_sha256"],  # type: ignore[arg-type]
        lineage_tasks=tuple(census_entries),
    )
    if census.as_dict() != dict(census_raw):
        raise QualityResolutionError("prepared train census is not canonical")
    production = ResolutionProductionPlan(
        lineages=tuple(lineages),
        rows=tuple(_row_from_dict(row) for row in raw["rows"]),
        source_census=census,
        initializer_bank_contract=raw["initializer_bank"],  # type: ignore[arg-type]
    )
    if production.as_dict() != dict(raw):
        raise QualityResolutionError("quality production plan is not its canonical projection")
    return production


def _authority_from_plan(raw: Mapping[str, object]) -> QualityResolutionPlanAuthority:
    prepared = raw["prepared_corpus"]
    publisher = raw["publisher"]
    ground = raw["ground_root"]
    selector = raw["selector"]
    context = raw["context"]
    implementation = raw["implementation"]
    if not all(
        isinstance(value, Mapping)
        for value in (prepared, publisher, ground, selector, context, implementation)
    ):
        raise QualityResolutionError("quality-resolution plan authority sections are invalid")
    if prepared.get("schema_version") != 4:
        raise QualityResolutionError("quality-resolution plan requires prepared schema v4")
    return QualityResolutionPlanAuthority(
        prepared_manifest_sha256=prepared.get("manifest_sha256"),  # type: ignore[arg-type]
        prepared_manifest_record_digest=prepared.get("manifest_record_digest"),  # type: ignore[arg-type]
        prepared_train_census_record_digest=prepared.get("train_census_record_digest"),  # type: ignore[arg-type]
        corpus_design_manifest_sha256=prepared.get("corpus_design_manifest_sha256"),  # type: ignore[arg-type]
        publisher_id=publisher.get("publisher_id"),  # type: ignore[arg-type]
        publisher_attestation_sha256=publisher.get("attestation_sha256"),  # type: ignore[arg-type]
        publisher_attestation_record_digest=publisher.get("attestation_record_digest"),  # type: ignore[arg-type]
        target_authority_record_digest=publisher.get("target_authority_record_digest"),  # type: ignore[arg-type]
        ground_root_sha256=ground.get("sha256"),  # type: ignore[arg-type]
        ground_root_record_digest=ground.get("record_digest"),  # type: ignore[arg-type]
        verifier_identity_digest=ground.get("verifier_identity_digest"),  # type: ignore[arg-type]
        selector_digest=selector.get("selector_digest"),  # type: ignore[arg-type]
        selector_file_sha256=selector.get("selector_file_sha256"),  # type: ignore[arg-type]
        selector_fit_receipt_sha256=selector.get("fit_receipt_sha256"),  # type: ignore[arg-type]
        selector_fit_record_digest=selector.get("fit_receipt_record_digest"),  # type: ignore[arg-type]
        normalizer_digest=selector.get("normalizer_digest"),  # type: ignore[arg-type]
        selector_device=selector.get("device"),  # type: ignore[arg-type]
        selector_device_parity_sha256=selector.get("device_parity_sha256"),  # type: ignore[arg-type]
        selector_device_parity_record_digest=selector.get("device_parity_record_digest"),  # type: ignore[arg-type]
        context=context.get("snapshot"),  # type: ignore[arg-type]
        context_digest=context.get("digest"),  # type: ignore[arg-type]
        implementation=implementation.get("registry"),  # type: ignore[arg-type]
        implementation_digest=implementation.get("digest"),  # type: ignore[arg-type]
    )


def publish_quality_resolution_plan(
    path: str | Path,
    plan: QualityResolutionPlan,
) -> str:
    """Atomically publish a canonical plan without replacing an existing artifact."""

    if not isinstance(plan, QualityResolutionPlan):
        raise TypeError("plan must be a QualityResolutionPlan")
    raw = canonical_json_bytes(plan.as_dict()) + b"\n"
    try:
        publish_new_file(path, raw)
    except ExactConformanceError as error:
        raise QualityResolutionError(str(error)) from error
    return hashlib.sha256(raw).hexdigest()


def load_quality_resolution_plan(
    path: str | Path,
    *,
    expected_plan_sha256: str,
    expected_config_sha256: str,
    expected_grid_sha256: str,
    expected_prepared_manifest_sha256: str,
    expected_prepared_train_census_record_digest: str,
    expected_publisher_attestation_sha256: str,
    expected_ground_root_sha256: str,
    expected_selector_file_sha256: str,
    expected_selector_device_parity_sha256: str,
) -> QualityResolutionPlan:
    """Load a plan only when its raw bytes and every upstream artifact remain pinned."""

    record, raw, _observed = _pinned_json(path, expected_plan_sha256, "quality-resolution plan")
    if raw != canonical_json_bytes(record) + b"\n":
        raise QualityResolutionError(
            "quality-resolution plan is not canonical JSON followed by one newline"
        )
    if set(record) != _PLAN_FIELDS:
        raise QualityResolutionError("quality-resolution plan schema fields differ")
    _assert_recursively_target_free(record)
    if (
        record["schema"] != QUALITY_RESOLUTION_PLAN_SCHEMA
        or record["schema_version"] != QUALITY_RESOLUTION_PLAN_VERSION
    ):
        raise QualityResolutionError("unsupported quality-resolution plan")
    recorded_digest = _digest(record["record_digest"], "quality-resolution plan record digest")
    body = {key: value for key, value in record.items() if key != "record_digest"}
    if not hmac.compare_digest(recorded_digest, content_digest(body)):
        raise QualityResolutionError("quality-resolution plan record digest mismatch")

    config_section = record["config"]
    grid_section = record["grid"]
    if not isinstance(config_section, Mapping) or set(config_section) != {
        "raw_sha256",
        "registered",
    }:
        raise QualityResolutionError("quality-resolution plan config section differs")
    if not isinstance(grid_section, Mapping) or set(grid_section) != {
        "quality_resolution",
        "raw_sha256",
        "schema",
        "schema_version",
    }:
        raise QualityResolutionError("quality-resolution plan grid section differs")
    external_checks = (
        (config_section["raw_sha256"], expected_config_sha256, "config pin"),
        (grid_section["raw_sha256"], expected_grid_sha256, "grid pin"),
        (
            record["prepared_corpus"].get("manifest_sha256"),  # type: ignore[union-attr]
            expected_prepared_manifest_sha256,
            "prepared manifest pin",
        ),
        (
            record["publisher"].get("attestation_sha256"),  # type: ignore[union-attr]
            expected_publisher_attestation_sha256,
            "publisher attestation pin",
        ),
        (
            record["prepared_corpus"].get("train_census_record_digest"),  # type: ignore[union-attr]
            expected_prepared_train_census_record_digest,
            "prepared train census pin",
        ),
        (
            record["ground_root"].get("sha256"),  # type: ignore[union-attr]
            expected_ground_root_sha256,
            "ground-root pin",
        ),
        (
            record["selector"].get("selector_file_sha256"),  # type: ignore[union-attr]
            expected_selector_file_sha256,
            "selector file pin",
        ),
        (
            record["selector"].get("device_parity_sha256"),  # type: ignore[union-attr]
            expected_selector_device_parity_sha256,
            "selector-device parity pin",
        ),
    )
    for observed, expected, label in external_checks:
        if not hmac.compare_digest(_digest(observed, label), _digest(expected, label)):
            raise QualityResolutionError(f"quality-resolution {label} differs")

    config = validate_quality_resolution_config(config_section["registered"])
    inputs = ResolutionPlanningInputs(
        config=config,
        config_sha256=config_section["raw_sha256"],  # type: ignore[arg-type]
        grid_sha256=grid_section["raw_sha256"],  # type: ignore[arg-type]
        grid_resolution_minima=grid_section["quality_resolution"],  # type: ignore[arg-type]
    )
    authority = _authority_from_plan(record)
    production = _production_from_dict(record["production_plan"])
    reproduced = build_quality_resolution_plan(inputs, authority, production).as_dict()
    if reproduced != record:
        raise QualityResolutionError(
            "quality-resolution plan differs from its deterministic reconstruction"
        )
    return QualityResolutionPlan(record)


__all__ = [
    "MINIMUM_PRODUCTION_LINEAGES",
    "QUALITY_RESOLUTION_PLAN_SCHEMA",
    "QUALITY_RESOLUTION_PLAN_VERSION",
    "PlannedResolutionAction",
    "PlannedResolutionLineage",
    "PlannedResolutionRow",
    "ResolutionPreparedCensus",
    "QualityResolutionPlan",
    "QualityResolutionPlanAuthority",
    "ResolutionPlanningInputs",
    "ResolutionProductionPlan",
    "build_quality_resolution_plan",
    "load_quality_resolution_plan",
    "load_quality_resolution_planning_inputs",
    "materialize_resolution_production_plan",
    "publish_quality_resolution_plan",
    "quality_action_provenance_fingerprint",
]
