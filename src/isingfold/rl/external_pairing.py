"""Authenticated stock-baseline execution and paired whole-system inference.

The historical :mod:`isingfold.rl.external` runner is useful for diagnostics, but its v1
receipt predates the sealed pre-initialization denominator.  This module is the publication
boundary.  It runs stock minorminer on exactly that denominator, applies the same total
online wall-clock envelope as the learned system, and binds every result to the population,
context, selector, quality authority and immutable configuration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Hashable

import numpy as np

from isingfold.rl.checkpoint import (
    RUNTIME_DEPENDENCIES,
    RUNTIME_IMPLEMENTATION_SCHEMA,
    RUNTIME_IMPLEMENTATION_VERSION,
    RUNTIME_MODULE_SOURCES,
)
from isingfold.rl.complete_system import (
    CompletePopulationIdentity,
    CompleteSystemConfig,
    FrozenComponentIdentity,
    PartialWorkVector,
    TerminalEvidence,
    _canonical_embedding,
    _canonical_program,
    _lineage_coherent_execution_tasks,
    _terminal_evidence_from_payload,
    _validate_evidence_outcome,
    verify_terminal_evidence,
)
from isingfold.rl.contracts import WORK_FIELDS, Context, WorkVector, stable_digest
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.env import EmbeddingTask, StrengthSelector
from isingfold.rl.evaluation_strata import (
    ConfirmatoryEvaluationDesign,
    EvaluationStratum,
    PowerTarget,
    PrecisionTarget,
    SIZE_BIN_BOUNDARY_CONVENTION,
    SIZE_BIN_PROTOCOL,
    preregistered_subgroups,
)
from isingfold.rl.evaluate import (
    TASK_TERMINAL,
    EpisodeOutcome,
    EvaluationProtocolError,
    program_digest,
    secondary_metrics,
)
from isingfold.rl.evaluator import sample_program
from isingfold.rl.experiment_selection import crossed_bootstrap_bounds
from isingfold.rl.external import (
    AUDIT_READS,
    BackendIdentity,
    ExternalBackendError,
    ExternalBackendStartupTimeout,
    ExternalEmbedderBackend,
    SearchStatus,
    STOCK_MINORMINER_METHOD,
    _embedding_digest,
    _normalise_embedding,
    _restart_seed,
    _seed,
    _select_strength,
)
from isingfold.rl.external_tuning import ExternalTuningExecutionBinding
from isingfold.rl.program import Program
from isingfold.rl.validate import ValidationReceipt, p_embed, p_return

EXTERNAL_COMPLETE_CONFIG_SCHEMA = "isingfold.external-complete-system-config"
EXTERNAL_COMPLETE_CONFIG_VERSION = 2
EXTERNAL_COMPLETE_RECEIPT_SCHEMA = "isingfold.external-complete-system-attempt"
EXTERNAL_COMPLETE_RECEIPT_VERSION = 3
EXTERNAL_COMPLETE_REPORT_SCHEMA = "isingfold.external-complete-system-evaluation"
EXTERNAL_COMPLETE_REPORT_VERSION = 4
PAIRED_COMPLETE_AGGREGATE_SCHEMA = "isingfold.learned-vs-stock-complete-system"
PAIRED_COMPLETE_AGGREGATE_VERSION = 4
EXTERNAL_COMPLETE_EVIDENCE_SCHEMA = "isingfold.external-complete-system-evidence"
EXTERNAL_COMPLETE_EVIDENCE_VERSION = 2

REGISTERED_TRAINING_SEEDS = (1103, 2207, 3301)
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 130363
UTILITY_ALPHA = 0.025
FEASIBILITY_ALPHA = 0.05
FEASIBILITY_MARGIN = 0.02
AGGREGATION = "equal-training-seed-then-equal-immutable-base-lineage"
_COMPLETE_REPORT_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        "partition",
        "sealed_test_opened",
        "no_fallback",
        "evaluation_protocol",
        "training_seed_index",
        "training_seed",
        "grid_manifest_sha256",
        "source_corpus_manifest_sha256",
        "quality_authority",
        "target_access",
        "ground_partition_receipt",
        "context",
        "context_digest",
        "work_cap",
        "population",
        "population_digest",
        "selector",
        "selector_digest",
        "external_tuning_execution",
        "external_tuning_execution_digest",
        "backend",
        "external_config",
        "external_config_digest",
        "external_config_file_sha256",
        "learned_config",
        "learned_config_digest",
        "learned_config_file_sha256",
        "runtime_platform",
        "inference_device_type",
        "inference_device_name",
        "inference_threads",
        "deterministic",
        "runtime_implementation_registry",
        "runtime_implementation_digest",
        "external_pairing_implementation_sha256",
        "method",
        "summary",
        "artifacts",
        "record_digest",
    }
)
_FROZEN_PROTOCOL = {
    "partition": "test",
    "evaluation_seed": 55079,
    "repetitions": 4,
    "audit_reads": 4096,
    "deployment_rule": "categorical-temperature-one",
    "population_scope": "all-policy-instances-before-initialization",
    "aggregation": AGGREGATION,
    "primary_metric": "unconditional-if-q3-s0-utility",
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "two_sided_alpha": 0.05,
    "feasibility_noninferiority_margin": FEASIBILITY_MARGIN,
}

_RUNTIME_IDENTITY_KEYS = frozenset(
    {
        "runtime_platform",
        "inference_device_type",
        "inference_device_name",
        "inference_threads",
        "deterministic",
    }
)


def runtime_identity(
    *,
    runtime_platform: Mapping[str, object],
    inference_device_type: str,
    inference_device_name: str,
    inference_threads: int,
    deterministic: bool,
) -> Mapping[str, object]:
    """Return the exact machine/runtime identity used for one paired arm."""

    payload = {
        "runtime_platform": dict(runtime_platform),
        "inference_device_type": inference_device_type,
        "inference_device_name": inference_device_name,
        "inference_threads": inference_threads,
        "deterministic": deterministic,
    }
    if (
        not runtime_platform
        or not isinstance(inference_device_type, str)
        or not inference_device_type
        or not isinstance(inference_device_name, str)
        or not inference_device_name
        or type(inference_threads) is not int
        or inference_threads <= 0
        or type(deterministic) is not bool
    ):
        raise ValueError("paired runtime identity is malformed")
    return _canonical_mapping(payload, label="paired runtime identity")


def _is_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _verify_record(value: Mapping[str, object], label: str) -> str:
    recorded = value.get("record_digest")
    payload = {key: item for key, item in value.items() if key != "record_digest"}
    if not _is_digest(recorded) or recorded != content_digest(payload):
        raise ValueError(f"{label} record digest mismatch")
    return recorded


def _canonical_mapping(value: Mapping[str, object], *, label: str) -> Mapping[str, object]:
    try:
        copied = json.loads(canonical_json_bytes(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not canonical finite JSON") from exc
    if not isinstance(copied, dict) or not copied:
        raise ValueError(f"{label} must be a nonempty JSON object")
    frozen = _freeze_json(copied)
    assert isinstance(frozen, Mapping)
    return frozen


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _plain_json(value: object) -> object:
    return json.loads(canonical_json_bytes(value))


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"cannot snapshot {type(value).__name__} as canonical JSON")


def context_snapshot(context: Context) -> dict[str, object]:
    snapshot = _jsonable(context)
    if not isinstance(snapshot, dict):  # pragma: no cover - Context is a dataclass
        raise TypeError("context snapshot must be an object")
    return snapshot


@dataclass(frozen=True)
class ExternalCompleteSystemConfig:
    """Registered stock-minorminer budget with a total online timing envelope."""

    backend: str
    expected_backend_version: str
    online_wallclock_seconds: float
    max_restarts: int
    audit_reads: int
    selection_rule: str
    online_evaluator_feedback: bool
    minorminer_parameters: Mapping[str, int]

    _KEYS = frozenset(
        {
            "schema",
            "schema_version",
            "backend",
            "expected_backend_version",
            "online_wallclock_seconds",
            "max_restarts",
            "audit_reads",
            "selection_rule",
            "online_evaluator_feedback",
            "minorminer_parameters",
        }
    )
    _PARAMETERS = frozenset(
        {"tries", "threads", "max_no_improvement", "chainlength_patience"}
    )

    def __post_init__(self) -> None:
        if self.backend != "stock-minorminer":
            raise ValueError("the publication external arm is fixed to stock-minorminer")
        if not isinstance(self.expected_backend_version, str) or not self.expected_backend_version:
            raise ValueError("expected backend version must be nonempty")
        if (
            isinstance(self.online_wallclock_seconds, bool)
            or not math.isfinite(self.online_wallclock_seconds)
            or self.online_wallclock_seconds <= 0.0
        ):
            raise ValueError("online wall-clock cap must be positive and finite")
        if type(self.max_restarts) is not int or self.max_restarts <= 0:
            raise ValueError("max_restarts must be a positive integer")
        if self.audit_reads != AUDIT_READS:
            raise ValueError("the publication external arm fixes 4096 final audit reads")
        if self.selection_rule != "resource-lexicographic":
            raise ValueError("the stock arm uses resource-lexicographic restart selection")
        if self.online_evaluator_feedback is not False:
            raise ValueError("stock search cannot observe evaluator feedback")
        parameters = dict(self.minorminer_parameters)
        if set(parameters) != self._PARAMETERS:
            raise ValueError("minorminer parameter registry has missing or unknown fields")
        if parameters.get("tries") != 1 or parameters.get("threads") != 1:
            raise ValueError("each registered restart fixes tries=1 and threads=1")
        if any(type(value) is not int or value <= 0 for value in parameters.values()):
            raise ValueError("minorminer parameters must be positive integers")
        object.__setattr__(self, "minorminer_parameters", MappingProxyType(parameters))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> ExternalCompleteSystemConfig:
        if set(payload) != cls._KEYS:
            raise ValueError("external complete-system config keys differ from the registry")
        if (
            payload["schema"] != EXTERNAL_COMPLETE_CONFIG_SCHEMA
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != EXTERNAL_COMPLETE_CONFIG_VERSION
        ):
            raise ValueError("unsupported external complete-system config schema")
        parameters = payload["minorminer_parameters"]
        if not isinstance(parameters, Mapping):
            raise ValueError("minorminer_parameters must be an object")
        return cls(
            backend=payload["backend"],  # type: ignore[arg-type]
            expected_backend_version=payload["expected_backend_version"],  # type: ignore[arg-type]
            online_wallclock_seconds=payload["online_wallclock_seconds"],  # type: ignore[arg-type]
            max_restarts=payload["max_restarts"],  # type: ignore[arg-type]
            audit_reads=payload["audit_reads"],  # type: ignore[arg-type]
            selection_rule=payload["selection_rule"],  # type: ignore[arg-type]
            online_evaluator_feedback=payload["online_evaluator_feedback"],  # type: ignore[arg-type]
            minorminer_parameters=dict(parameters),  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": EXTERNAL_COMPLETE_CONFIG_SCHEMA,
            "schema_version": EXTERNAL_COMPLETE_CONFIG_VERSION,
            "backend": self.backend,
            "expected_backend_version": self.expected_backend_version,
            "online_wallclock_seconds": self.online_wallclock_seconds,
            "max_restarts": self.max_restarts,
            "audit_reads": self.audit_reads,
            "selection_rule": self.selection_rule,
            "online_evaluator_feedback": self.online_evaluator_feedback,
            "minorminer_parameters": dict(self.minorminer_parameters),
        }

    @property
    def digest(self) -> str:
        return stable_digest(self.as_dict())

    def validate_symmetric_envelope(
        self, learned: CompleteSystemConfig, context: Context
    ) -> None:
        if not math.isclose(
            self.online_wallclock_seconds,
            learned.online_wallclock_seconds,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("learned and stock systems use different total online time caps")
        if self.audit_reads != learned.audit_reads or self.audit_reads != context.audit_reads:
            raise ValueError("learned and stock systems use different final audit-read blocks")
        if self.selection_rule != learned.selection_rule:
            raise ValueError("learned and stock systems use different restart selection rules")
        if learned.online_evaluator_feedback or self.online_evaluator_feedback:
            raise ValueError("online evaluator feedback invalidates the complete-system pairing")
        if self.max_restarts > context.caps.restart_work:
            raise ValueError("stock restart cap exceeds the registered context work cap")


@dataclass(frozen=True)
class ExternalRestartEvidence:
    restart_index: int
    seed: int
    status: str
    reported_seconds: float = field(compare=False)
    observed_seconds: float = field(compare=False)
    embedding_digest: str | None = None
    validation_digest: str | None = None
    qubits: int | None = None
    max_chain: int | None = None
    selector_max_p_solve: float | None = None
    selector_strength_index: int | None = None
    detail: str = ""

    def __post_init__(self) -> None:
        allowed = {
            "VALID_CANDIDATE",
            "INVALID_CANDIDATE",
            "WORK_CAP_EXHAUSTED",
            SearchStatus.NO_EMBEDDING.value,
            SearchStatus.TIMED_OUT.value,
            SearchStatus.ERROR.value,
        }
        if self.status not in allowed:
            raise ValueError("external restart has an unknown status")
        if type(self.restart_index) is not int or self.restart_index < 0:
            raise ValueError("external restart index must be nonnegative")
        if type(self.seed) is not int or not 0 <= self.seed < 2**31:
            raise ValueError("external restart seed is outside its domain")
        for value in (self.reported_seconds, self.observed_seconds):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError("external restart timing must be finite and nonnegative")
        if self.status in {"VALID_CANDIDATE", "INVALID_CANDIDATE"}:
            if not _is_digest(self.embedding_digest) or not _is_digest(self.validation_digest):
                raise ValueError("screened external candidate lacks evidence digests")
            if type(self.qubits) is not int or type(self.max_chain) is not int:
                raise ValueError("screened external candidate lacks resource counts")
        elif any(
            value is not None
            for value in (
                self.embedding_digest,
                self.validation_digest,
                self.qubits,
                self.max_chain,
                self.selector_max_p_solve,
                self.selector_strength_index,
            )
        ):
            raise ValueError("non-embedding restart cannot carry embedding evidence")
        if (self.selector_max_p_solve is None) != (self.selector_strength_index is None):
            raise ValueError("external restart selector score and strength index must be paired")
        if self.selector_max_p_solve is not None and (
            not math.isfinite(self.selector_max_p_solve)
            or not 0.0 <= self.selector_max_p_solve <= 1.0
            or type(self.selector_strength_index) is not int
            or not 0 <= self.selector_strength_index < 4
            or self.status != "VALID_CANDIDATE"
        ):
            raise ValueError("external restart has an invalid frozen-selector ranking score")
        if not isinstance(self.detail, str):
            raise ValueError("external restart detail must be a string")

    def as_dict(self) -> dict[str, object]:
        return {
            "restart_index": self.restart_index,
            "seed": self.seed,
            "status": self.status,
            "reported_seconds": self.reported_seconds,
            "observed_seconds": self.observed_seconds,
            "embedding_digest": self.embedding_digest,
            "validation_digest": self.validation_digest,
            "qubits": self.qubits,
            "max_chain": self.max_chain,
            "selector_max_p_solve": self.selector_max_p_solve,
            "selector_strength_index": self.selector_strength_index,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class _RankedExternalCandidate:
    restart_index: int
    chains: Mapping[Hashable, frozenset[Hashable]]
    qubits: int
    max_chain: int
    embedding_digest: str
    return_validation: ValidationReceipt | None = None
    programs: tuple[Program, ...] = ()
    selected_strength_index: int | None = None
    selector_max_p_solve: float | None = None


@dataclass(frozen=True)
class ExternalCompleteSystemReceipt:
    """One stock attempt bound to the same denominator and online envelope as learned."""

    instance: str
    lineage: str
    repetition: int
    training_seed_index: int
    training_seed: int
    system_seed: int
    evaluator_seed: int | None
    backend: BackendIdentity
    selector: FrozenComponentIdentity
    population: CompletePopulationIdentity
    config_digest: str
    learned_config_digest: str
    context_digest: str
    quality_authority_digest: str
    runtime_identity_digest: str
    work_cap: WorkVector
    restarts: tuple[ExternalRestartEvidence, ...]
    selected_restart: int | None
    outcome: EpisodeOutcome
    terminal_evidence_digest: str | None
    terminal_evidence: TerminalEvidence | None = field(repr=False, compare=False)
    observed_work: PartialWorkVector
    cap_compliance: Mapping[str, bool | None]
    adapter_startup_seconds: float = field(compare=False)
    search_seconds: float = field(compare=False)
    postprocessing_seconds: float = field(compare=False)
    online_seconds: float = field(compare=False)
    evaluator_seconds: float | None = field(default=None, compare=False)
    wallclock_cap_seconds: float = field(default=0.0, compare=False)
    wallclock_compliant: bool = True
    tuning_execution: ExternalTuningExecutionBinding | None = None

    def __post_init__(self) -> None:
        if not self.instance or not self.lineage:
            raise ValueError("external complete receipt needs instance and lineage")
        if type(self.repetition) is not int or self.repetition < 0:
            raise ValueError("external repetition must be nonnegative")
        if (
            type(self.training_seed_index) is not int
            or self.training_seed_index not in range(len(REGISTERED_TRAINING_SEEDS))
            or type(self.training_seed) is not int
            or self.training_seed != REGISTERED_TRAINING_SEEDS[self.training_seed_index]
        ):
            raise ValueError("external run is not aligned to a registered training seed")
        if type(self.system_seed) is not int or not 0 <= self.system_seed < 2**31:
            raise ValueError("external system seed is outside its domain")
        if not isinstance(self.backend, BackendIdentity):
            raise TypeError("external backend identity has the wrong type")
        if not isinstance(self.selector, FrozenComponentIdentity):
            raise TypeError("external selector identity has the wrong type")
        if not isinstance(self.population, CompletePopulationIdentity):
            raise TypeError("external population identity has the wrong type")
        if not isinstance(self.work_cap, WorkVector):
            raise TypeError("external work cap has the wrong type")
        if self.tuning_execution is not None and not isinstance(
            self.tuning_execution, ExternalTuningExecutionBinding
        ):
            raise TypeError("external tuning execution binding has the wrong type")
        if self.evaluator_seed is not None and (
            type(self.evaluator_seed) is not int or not 0 <= self.evaluator_seed < 2**31
        ):
            raise ValueError("external evaluator seed is outside its domain")
        if self.selected_restart is not None and type(self.selected_restart) is not int:
            raise TypeError("external selected restart must be an integer or null")
        if type(self.wallclock_compliant) is not bool:
            raise TypeError("external wall-clock compliance must be Boolean")
        if any(
            not _is_digest(value)
            for value in (
                self.config_digest,
                self.learned_config_digest,
                self.context_digest,
                self.quality_authority_digest,
                self.runtime_identity_digest,
            )
        ):
            raise ValueError("external complete receipt has an invalid authority digest")
        if (self.lineage, self.instance) not in self.population.expected_instances:
            raise ValueError("external receipt lies outside its sealed population")
        if self.repetition >= self.population.expected_repetitions:
            raise ValueError("external receipt repetition lies outside its sealed population")
        indices = tuple(item.restart_index for item in self.restarts)
        if indices != tuple(range(len(self.restarts))):
            raise ValueError("external restart indices must be contiguous")
        if len({item.seed for item in self.restarts}) != len(self.restarts):
            raise ValueError("external restart seeds must be unique")
        if any(
            item.seed
            != _restart_seed(self.system_seed, self.backend.method_id, item.restart_index)
            for item in self.restarts
        ):
            raise ValueError("external restart seed schedule differs from the registry")
        valid = {item.restart_index for item in self.restarts if item.status == "VALID_CANDIDATE"}
        if self.selected_restart is not None and self.selected_restart not in valid:
            raise ValueError("external selected restart is not a validated candidate")
        if self.selected_restart is not None and not valid:
            raise ValueError("external valid candidates and selected restart disagree")
        if valid and self.selected_restart is None and self.outcome.reason != (
            "EXTERNAL_KNOWN_WORK_CAP_EXHAUSTED"
        ):
            raise ValueError("external valid candidates and selected restart disagree")
        quality_ranking = (
            self.tuning_execution is not None
            and self.tuning_execution.candidate.family == "time-saturating-quality"
        )
        scored_valid = tuple(
            item
            for item in self.restarts
            if item.status == "VALID_CANDIDATE" and item.selector_max_p_solve is not None
        )
        if quality_ranking:
            if any(
                item.status == "VALID_CANDIDATE"
                and item.selector_max_p_solve is None
                and self.outcome.reason != "EXTERNAL_KNOWN_WORK_CAP_EXHAUSTED"
                for item in self.restarts
            ):
                raise ValueError("quality-ranked external restart lacks its selector score")
        elif scored_valid:
            raise ValueError("resource-ranked external restart carries a quality selector score")
        if self.tuning_execution is not None and len(self.restarts) > (
            self.tuning_execution.candidate.outer_restart_cap
        ):
            raise ValueError("external restarts exceed the tuned strategy cap")
        if self.selected_restart is not None:
            expected_selected = min(
                (
                    item
                    for item in self.restarts
                    if item.status == "VALID_CANDIDATE"
                    and (not quality_ranking or item.selector_max_p_solve is not None)
                ),
                key=lambda item: (
                    (
                        -float(item.selector_max_p_solve)
                        if quality_ranking
                        else 0.0
                    ),
                    item.qubits,
                    item.max_chain,
                    item.embedding_digest,
                    item.restart_index,
                ),
            ).restart_index
            if self.selected_restart != expected_selected:
                raise ValueError("external selected restart violates its tuned ranking order")
        if self.outcome.pair_key != self.pair_key or self.outcome.episode_seed != self.system_seed:
            raise ValueError("external outcome identity differs from its detailed receipt")
        self.outcome.validate_receipt(require_complete=False)
        if self.terminal_evidence is not None and not isinstance(
            self.terminal_evidence, TerminalEvidence
        ):
            raise TypeError("external terminal evidence has the wrong type")
        if (
            self.terminal_evidence is not None
            and self.terminal_evidence.digest != self.terminal_evidence_digest
        ):
            raise ValueError("external terminal evidence digest is inconsistent")
        if self.terminal_evidence is not None:
            _validate_evidence_outcome(self.terminal_evidence, self.outcome)
        if self.outcome.returned_valid != (self.terminal_evidence_digest is not None):
            raise ValueError("external terminal evidence presence disagrees with validity")
        if self.outcome.returned_valid:
            if self.outcome.reason != "EXTERNAL_VALID_RETURN":
                raise ValueError("valid external outcome has an unauthenticated reason")
            if self.evaluator_seed is None or self.outcome.evaluator_seed != self.evaluator_seed:
                raise ValueError("valid external outcome lacks its evaluator seed")
            if not self.wallclock_compliant:
                raise ValueError("an over-budget external result cannot be accepted as valid")
            selected = self.restarts[self.selected_restart]  # type: ignore[index]
            if (
                self.outcome.qubits != selected.qubits
                or self.outcome.max_chain != selected.max_chain
                or self.outcome.evaluator_reads != AUDIT_READS
            ):
                raise ValueError("external outcome differs from its selected candidate or audit")
            if quality_ranking and self.terminal_evidence is not None and (
                self.terminal_evidence.selected_index != selected.selector_strength_index
            ):
                raise ValueError("quality-ranked restart and terminal strength selection differ")
        elif self.outcome.reason not in {
            "EXTERNAL_NO_VALID_EMBEDDING",
            "EXTERNAL_COMPLETE_SYSTEM_WALLCLOCK_EXHAUSTED",
            "EXTERNAL_KNOWN_WORK_CAP_EXHAUSTED",
        }:
            raise ValueError("invalid external outcome has an unauthenticated reason")
        elif (
            self.evaluator_seed is not None
            or self.outcome.evaluator_seed is not None
            or self.outcome.utility != 0.0
        ):
            raise ValueError("invalid external outcome must have zero utility and null evaluator seed")
        if self.outcome.population_eligible is not True:
            raise ValueError("external whole-system outcomes must remain denominator-eligible")
        if not isinstance(self.observed_work, PartialWorkVector):
            raise TypeError("external observed work must retain unknown native coordinates")
        expected_compilers = 4 * len(scored_valid) if quality_ranking else (
            4 if self.selected_restart is not None else 0
        )
        expected_reads = AUDIT_READS if self.outcome.returned_valid else 0
        expected_validators = sum(
            item.status in {"VALID_CANDIDATE", "INVALID_CANDIDATE"}
            for item in self.restarts
        ) + (len(scored_valid) if quality_ranking else int(expected_compilers == 4))
        if (
            self.observed_work.restart_work != len(self.restarts)
            or self.observed_work.validator_calls != expected_validators
            or self.observed_work.compiler_calls != expected_compilers
            or self.observed_work.feature_work != expected_compilers
            or self.observed_work.evaluator_reads != expected_reads
        ):
            raise ValueError("external observed work omits a registered runner operation")
        if not isinstance(self.cap_compliance, Mapping):
            raise TypeError("external cap compliance must be an object")
        expected_compliance = dict(self.observed_work.cap_compliance(self.work_cap))
        if dict(self.cap_compliance) != expected_compliance:
            raise ValueError("external work-cap compliance differs from its observed ledger")
        if any(value is False for value in expected_compliance.values()):
            raise ValueError("external receipt contains a known work-cap violation")
        object.__setattr__(self, "cap_compliance", MappingProxyType(expected_compliance))
        for value in (
            self.search_seconds,
            self.adapter_startup_seconds,
            self.postprocessing_seconds,
            self.online_seconds,
            self.wallclock_cap_seconds,
        ):
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError("external complete timing must be finite and nonnegative")
        if not math.isclose(
            self.search_seconds + self.postprocessing_seconds,
            self.online_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("external online time must include search and postprocessing")
        if sum(item.observed_seconds for item in self.restarts) > self.search_seconds + 1e-9:
            raise ValueError("external search time is smaller than its measured restart calls")
        if self.adapter_startup_seconds > self.search_seconds + 1e-9:
            raise ValueError("external adapter startup exceeds its measured search phase")
        compliant = self.online_seconds <= self.wallclock_cap_seconds + max(
            1e-9, self.wallclock_cap_seconds * 1e-9
        )
        if self.wallclock_compliant != compliant:
            raise ValueError("external wall-clock compliance flag is inconsistent")
        if self.outcome.online_seconds != self.online_seconds:
            raise ValueError("external outcome and detailed receipt disagree on online time")
        if self.outcome.evaluator_seconds != self.evaluator_seconds:
            raise ValueError("external outcome and detailed receipt disagree on evaluator time")

    @property
    def pair_key(self) -> tuple[str, str, int]:
        return (self.lineage, self.instance, self.repetition)

    def as_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": EXTERNAL_COMPLETE_RECEIPT_SCHEMA,
            "schema_version": EXTERNAL_COMPLETE_RECEIPT_VERSION,
            "instance": self.instance,
            "lineage": self.lineage,
            "repetition": self.repetition,
            "training_seed_index": self.training_seed_index,
            "training_seed": self.training_seed,
            "system_seed": self.system_seed,
            "evaluator_seed": self.evaluator_seed,
            "backend": self.backend.as_dict(),
            "selector": self.selector.as_dict(),
            "external_tuning_execution": (
                None if self.tuning_execution is None else self.tuning_execution.as_dict()
            ),
            "external_tuning_execution_digest": (
                None if self.tuning_execution is None else self.tuning_execution.digest
            ),
            "population": self.population.as_dict(),
            "population_digest": self.population.digest,
            "config_digest": self.config_digest,
            "learned_config_digest": self.learned_config_digest,
            "context_digest": self.context_digest,
            "quality_authority_digest": self.quality_authority_digest,
            "runtime_identity_digest": self.runtime_identity_digest,
            "work_cap": self.work_cap.as_dict(),
            "restarts": [item.as_dict() for item in self.restarts],
            "selected_restart": self.selected_restart,
            "outcome": self.outcome.as_dict(),
            "terminal_evidence_digest": self.terminal_evidence_digest,
            "observed_work": self.observed_work.as_dict(),
            "observed_work_known_lower_bound": self.observed_work.known_lower_bound.as_dict(),
            "cap_compliance": dict(self.cap_compliance),
            "adapter_startup_seconds": self.adapter_startup_seconds,
            "search_seconds": self.search_seconds,
            "postprocessing_seconds": self.postprocessing_seconds,
            "online_seconds": self.online_seconds,
            "evaluator_seconds": self.evaluator_seconds,
            "wallclock_cap_seconds": self.wallclock_cap_seconds,
            "wallclock_compliant": self.wallclock_compliant,
        }
        if include_digest:
            payload["record_digest"] = stable_digest(payload)
        return payload


def _partial_work(*, restarts: int, validators: int, compilers: int, reads: int) -> PartialWorkVector:
    lower = WorkVector(
        compiler_calls=compilers,
        validator_calls=validators,
        restart_work=restarts,
        evaluator_reads=reads,
        feature_work=compilers,
    )
    return PartialWorkVector(
        decisions=None,
        route_expansions=None,
        materializations=None,
        compiler_calls=compilers,
        validator_calls=validators,
        cut_edge_visits=None,
        restart_work=restarts,
        evaluator_reads=reads,
        feature_work=compilers,
        known_lower_bound=lower,
    )


def _failure_outcome(
    task: EmbeddingTask,
    *,
    repetition: int,
    system_seed: int,
    reason: str,
    validation_digest: str,
    online_seconds: float,
) -> EpisodeOutcome:
    outcome = EpisodeOutcome(
        instance=task.name,
        lineage=task.lineage or task.name,
        returned_valid=False,
        utility=0.0,
        qubits=None,
        max_chain=None,
        decisions=0,
        selected_strength=None,
        reason=reason,
        repetition=repetition,
        episode_seed=system_seed,
        evaluator_seed=None,
        work=None,
        validation_digest=validation_digest,
        online_seconds=online_seconds,
        evaluator_seconds=None,
        outcome_kind=TASK_TERMINAL,
        population_eligible=True,
    )
    outcome.validate_receipt(require_complete=False)
    return outcome


def _quality_rank_strengths(
    selector: StrengthSelector,
    context: Context,
    task: EmbeddingTask,
    chains: Mapping[Hashable, frozenset[Hashable]],
    programs: Sequence[Program],
) -> tuple[int, float]:
    """Return frozen-selector argmax and max probability without evaluator access."""

    required = (
        "select_embedding",
        "deployment_ready",
        "normalizer_digest",
        "coefficient_transform_scale",
    )
    if not all(hasattr(selector, name) for name in required):
        raise ExternalBackendError(
            "quality-aware stock reranking requires the frozen deployment graph selector"
        )
    if selector.deployment_ready is not True:  # type: ignore[attr-defined]
        raise ExternalBackendError("quality-aware stock reranking selector is not deployment-ready")
    from isingfold.rl.strength_tensorize import build_strength_inputs

    inputs = build_strength_inputs(
        ctx=context,
        logical=task.logical,
        host=task.host,
        problem=task.problem,
        chains=chains,
        programs=programs,
        coef_scale=float(selector.coefficient_transform_scale),  # type: ignore[attr-defined]
        normalizer_digest=selector.normalizer_digest,  # type: ignore[attr-defined]
    )
    try:
        import torch

        with torch.no_grad():
            raw = selector(inputs)  # type: ignore[operator]
    except Exception as exc:
        raise ExternalBackendError("frozen selector probability inference failed") from exc
    if hasattr(raw, "detach"):
        raw = raw.detach().cpu().numpy()
    scores = np.asarray(raw, dtype=float)
    if scores.shape != (4,) or not np.isfinite(scores).all() or (
        (scores < 0.0).any() or (scores > 1.0).any()
    ):
        raise ExternalBackendError("frozen selector returned invalid success probabilities")
    selected_index = int(np.argmax(scores))
    return selected_index, float(scores[selected_index])


def run_external_complete_system(
    tasks: Sequence[EmbeddingTask],
    context: Context,
    backend: ExternalEmbedderBackend,
    config: ExternalCompleteSystemConfig,
    *,
    learned_config: CompleteSystemConfig,
    selector: StrengthSelector,
    selector_identity: FrozenComponentIdentity,
    population: CompletePopulationIdentity,
    quality_authority: Mapping[str, object],
    seed: int,
    repetitions: int,
    training_seed_index: int = 0,
    training_seed: int = REGISTERED_TRAINING_SEEDS[0],
    compute_identity: Mapping[str, object] | None = None,
    tuning_execution: ExternalTuningExecutionBinding | None = None,
    execution_lineages: Sequence[str] | None = None,
) -> tuple[list[EpisodeOutcome], list[ExternalCompleteSystemReceipt]]:
    """Run stock minorminer on the sealed paper census with failure-aware utility."""

    config.validate_symmetric_envelope(learned_config, context)
    if seed != population.evaluation_seed or repetitions != population.expected_repetitions:
        raise EvaluationProtocolError("external runtime differs from its sealed seed/repetition census")
    population.validate_tasks(tasks)
    execution_tasks = _lineage_coherent_execution_tasks(
        tasks,
        execution_lineages=execution_lineages,
    )
    if (
        training_seed_index not in range(len(REGISTERED_TRAINING_SEEDS))
        or training_seed != REGISTERED_TRAINING_SEEDS[training_seed_index]
    ):
        raise EvaluationProtocolError(
            "external run must be matched to one registered learned training seed"
        )
    if tuning_execution is not None and not isinstance(
        tuning_execution, ExternalTuningExecutionBinding
    ):
        raise TypeError("external tuning execution binding has the wrong type")
    if tuning_execution is None:
        effective_parameters = config.minorminer_parameters
        effective_restart_cap = config.max_restarts
        quality_ranking = False
        execution_seed = seed
    else:
        candidate = tuning_execution.candidate
        effective_parameters = candidate.native_parameters
        effective_restart_cap = candidate.outer_restart_cap
        quality_ranking = candidate.family == "time-saturating-quality"
        if effective_restart_cap > config.max_restarts:
            raise EvaluationProtocolError("tuned stock strategy exceeds the symmetric outer cap")
        population_partitions = {
            row.learning_partition for row in population.evaluation_strata
        }
        expected_partition = (
            "val" if tuning_execution.mode == "validation-candidate" else "test"
        )
        if population_partitions != {expected_partition}:
            raise EvaluationProtocolError(
                "external tuning execution mode and sealed population partition differ"
            )
        # Tuning seeds randomize validation searches without changing the immutable population
        # object. Frozen test deployment retains the exact paired evaluation-seed schedule.
        execution_seed = (
            training_seed
            if tuning_execution.mode == "validation-candidate"
            else seed
        )
    if compute_identity is None:
        compute_identity = runtime_identity(
            runtime_platform={"test_or_library_runtime": "unspecified"},
            inference_device_type="cpu",
            inference_device_name="unspecified-cpu",
            inference_threads=1,
            deterministic=True,
        )
    else:
        compute_identity = _canonical_mapping(
            compute_identity, label="external compute identity"
        )
        if set(compute_identity) != _RUNTIME_IDENTITY_KEYS:
            raise EvaluationProtocolError("external compute identity fields differ")
        runtime_identity(**dict(compute_identity))  # type: ignore[arg-type]
    runtime_digest = content_digest(compute_identity)
    if not isinstance(backend.identity, BackendIdentity):
        raise TypeError("external backend must expose a BackendIdentity")
    backend_identity = backend.identity
    if (
        backend_identity.method_id != STOCK_MINORMINER_METHOD
        or backend_identity.distribution != "minorminer"
    ):
        raise EvaluationProtocolError("external backend identity differs from its config")
    if backend_identity.version != config.expected_backend_version:
        raise EvaluationProtocolError("external backend version differs from its config")
    if any(task.ground_energy is None or not math.isfinite(task.ground_energy) for task in tasks):
        raise EvaluationProtocolError("external IF-Q3 evaluation requires finite targets")
    quality = _canonical_mapping(quality_authority, label="quality authority")
    quality_digest = content_digest(quality)
    context_digest = stable_digest(context_snapshot(context))
    outcomes: list[EpisodeOutcome] = []
    receipts: list[ExternalCompleteSystemReceipt] = []

    for repetition in range(repetitions):
        for task in execution_tasks:
            seed_domain = (
                "external-tuning-policy"
                if tuning_execution is not None
                and tuning_execution.mode == "validation-candidate"
                else "policy"
            )
            evaluator_domain = (
                "external-tuning-final-evaluator"
                if tuning_execution is not None
                and tuning_execution.mode == "validation-candidate"
                else "final-evaluator"
            )
            system_seed = _seed(execution_seed, seed_domain, task, repetition)
            expected_evaluator_seed = _seed(
                execution_seed, evaluator_domain, task, repetition
            )
            started = time.perf_counter()
            deadline = started + config.online_wallclock_seconds
            restarts: list[ExternalRestartEvidence] = []
            candidates: list[_RankedExternalCandidate] = []
            validators = 0
            compilers = 0
            work_cap_failure = False
            adapter_startup_seconds = 0.0
            startup_timeout = False
            session = None
            open_session = getattr(backend, "open_session", None)
            if callable(open_session):
                startup_budget = deadline - time.perf_counter()
                if startup_budget <= 0.0:
                    startup_timeout = True
                else:
                    try:
                        session = open_session(
                            task.logical,
                            task.host,
                            timeout_seconds=startup_budget,
                            parameters=effective_parameters,
                        )
                        adapter_startup_seconds = float(session.startup_seconds)
                    except ExternalBackendStartupTimeout:
                        startup_timeout = True
            for restart_index in range(effective_restart_cap):
                if startup_timeout:
                    break
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                if len(restarts) + 1 > context.caps.restart_work:
                    work_cap_failure = True
                    break
                if backend.identity != backend_identity:
                    raise ExternalBackendError(
                        "external backend identity changed after the contract was sealed"
                    )
                restart_seed = _restart_seed(system_seed, backend_identity.method_id, restart_index)
                call_started = time.perf_counter()
                try:
                    if session is None:
                        result = backend.search(
                            task.logical,
                            task.host,
                            seed=restart_seed,
                            timeout_seconds=remaining,
                            parameters=effective_parameters,
                        )
                    else:
                        result = session.search(
                            seed=restart_seed,
                            timeout_seconds=remaining,
                        )
                except ExternalBackendError:
                    if session is not None:
                        session.close()
                    raise
                except Exception as exc:
                    if session is not None:
                        session.close()
                    raise ExternalBackendError(
                        f"external backend raised {type(exc).__name__}: {exc}"
                    ) from exc
                if backend.identity != backend_identity:
                    raise ExternalBackendError(
                        "external backend identity changed during a native search call"
                    )
                observed = time.perf_counter() - call_started
                timed_out = observed > remaining + max(1e-9, remaining * 1e-9)
                if timed_out or result.status is SearchStatus.TIMED_OUT:
                    restarts.append(
                        ExternalRestartEvidence(
                            restart_index,
                            restart_seed,
                            SearchStatus.TIMED_OUT.value,
                            result.elapsed_seconds,
                            observed,
                            detail=result.detail or "registered native-call deadline reached",
                        )
                    )
                    continue
                if result.status in {SearchStatus.NO_EMBEDDING, SearchStatus.ERROR}:
                    restarts.append(
                        ExternalRestartEvidence(
                            restart_index,
                            restart_seed,
                            result.status.value,
                            result.elapsed_seconds,
                            observed,
                            detail=result.detail,
                        )
                    )
                    continue
                if result.embedding is None:
                    raise ExternalBackendError("external EMBEDDING result omitted its payload")
                if validators + 1 > context.caps.validator_calls:
                    restarts.append(
                        ExternalRestartEvidence(
                            restart_index,
                            restart_seed,
                            "WORK_CAP_EXHAUSTED",
                            result.elapsed_seconds,
                            observed,
                            detail="validator_calls cap would be exceeded",
                        )
                    )
                    work_cap_failure = True
                    break
                chains = _normalise_embedding(result.embedding)
                validation = p_embed(chains, task.logical, task.host, context.qubit_cap)
                validators += 1
                embedding_digest = _embedding_digest(chains)
                validation_digest = stable_digest(validation.as_dict())
                if not validation.valid:
                    restarts.append(
                        ExternalRestartEvidence(
                            restart_index,
                            restart_seed,
                            "INVALID_CANDIDATE",
                            result.elapsed_seconds,
                            observed,
                            embedding_digest,
                            validation_digest,
                            validation.qubits,
                            validation.max_chain,
                            detail="; ".join(validation.reasons),
                        )
                    )
                    continue

                if quality_ranking:
                    can_rank = (
                        validators + 1 <= context.caps.validator_calls
                        and compilers + 4 <= context.caps.compiler_calls
                        and compilers + 4 <= context.caps.feature_work
                    )
                    if not can_rank:
                        restarts.append(
                            ExternalRestartEvidence(
                                restart_index,
                                restart_seed,
                                "VALID_CANDIDATE",
                                result.elapsed_seconds,
                                observed,
                                embedding_digest,
                                validation_digest,
                                validation.qubits,
                                validation.max_chain,
                                detail=(
                                    "frozen-selector ranking would exceed a known work cap"
                                ),
                            )
                        )
                        work_cap_failure = True
                        break
                    return_validation, programs = p_return(
                        chains, task.logical, task.host, task.problem, context
                    )
                    validators += 1
                    compilers += len(programs)
                    if not return_validation.valid or len(programs) != 4 or any(
                        not isinstance(program, Program) for program in programs
                    ):
                        raise ExternalBackendError(
                            "valid stock candidate failed quality-ranking compilation"
                        )
                    selected_strength_index, selector_max_p_solve = (
                        _quality_rank_strengths(
                            selector, context, task, chains, programs
                        )
                    )
                    restarts.append(
                        ExternalRestartEvidence(
                            restart_index,
                            restart_seed,
                            "VALID_CANDIDATE",
                            result.elapsed_seconds,
                            observed,
                            embedding_digest,
                            validation_digest,
                            validation.qubits,
                            validation.max_chain,
                            selector_max_p_solve,
                            selected_strength_index,
                            "; ".join(validation.reasons),
                        )
                    )
                    candidates.append(
                        _RankedExternalCandidate(
                            restart_index=restart_index,
                            chains=chains,
                            qubits=validation.qubits,
                            max_chain=validation.max_chain,
                            embedding_digest=embedding_digest,
                            return_validation=return_validation,
                            programs=tuple(programs),
                            selected_strength_index=selected_strength_index,
                            selector_max_p_solve=selector_max_p_solve,
                        )
                    )
                else:
                    restarts.append(
                        ExternalRestartEvidence(
                            restart_index,
                            restart_seed,
                            "VALID_CANDIDATE",
                            result.elapsed_seconds,
                            observed,
                            embedding_digest,
                            validation_digest,
                            validation.qubits,
                            validation.max_chain,
                            detail="; ".join(validation.reasons),
                        )
                    )
                    candidates.append(
                        _RankedExternalCandidate(
                            restart_index=restart_index,
                            chains=chains,
                            qubits=validation.qubits,
                            max_chain=validation.max_chain,
                            embedding_digest=embedding_digest,
                        )
                    )

            if session is not None:
                session.close()

            search_seconds = time.perf_counter() - started
            selected_restart: int | None = None
            evaluator_reads = 0
            evaluator_seconds: float | None = None
            terminal_evidence: TerminalEvidence | None = None
            if work_cap_failure:
                online_seconds = time.perf_counter() - started
                validation_digest = stable_digest(
                    {
                        "kind": "EXTERNAL_KNOWN_WORK_CAP_EXHAUSTED",
                        "restarts": [item.as_dict() for item in restarts],
                    }
                )
                outcome = _failure_outcome(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    reason="EXTERNAL_KNOWN_WORK_CAP_EXHAUSTED",
                    validation_digest=validation_digest,
                    online_seconds=online_seconds,
                )
            elif not candidates:
                online_seconds = time.perf_counter() - started
                failure_reason = (
                    "EXTERNAL_COMPLETE_SYSTEM_WALLCLOCK_EXHAUSTED"
                    if startup_timeout
                    else "EXTERNAL_NO_VALID_EMBEDDING"
                )
                validation_digest = stable_digest(
                    {
                        "kind": failure_reason,
                        "restarts": [item.as_dict() for item in restarts],
                    }
                )
                outcome = _failure_outcome(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    reason=failure_reason,
                    validation_digest=validation_digest,
                    online_seconds=online_seconds,
                )
            else:
                selected_candidate = min(
                    candidates,
                    key=lambda item: (
                        (
                            -float(item.selector_max_p_solve)
                            if quality_ranking
                            and item.selector_max_p_solve is not None
                            else 0.0
                        ),
                        item.qubits,
                        item.max_chain,
                        item.embedding_digest,
                        item.restart_index,
                    ),
                )
                selected_restart = selected_candidate.restart_index
                chains = selected_candidate.chains
                can_compile = quality_ranking or (
                    validators + 1 <= context.caps.validator_calls
                    and compilers + 4 <= context.caps.compiler_calls
                    and compilers + 4 <= context.caps.feature_work
                )
                if not can_compile:
                    selected_restart = None
                    work_cap_failure = True
                    online_seconds = time.perf_counter() - started
                    validation_digest = stable_digest(
                        {
                            "kind": "EXTERNAL_KNOWN_WORK_CAP_EXHAUSTED",
                            "operation": "four-program-return-and-selection",
                            "restarts": [item.as_dict() for item in restarts],
                        }
                    )
                    outcome = _failure_outcome(
                        task,
                        repetition=repetition,
                        system_seed=system_seed,
                        reason="EXTERNAL_KNOWN_WORK_CAP_EXHAUSTED",
                        validation_digest=validation_digest,
                        online_seconds=online_seconds,
                    )
                else:
                    if quality_ranking:
                        validation = selected_candidate.return_validation
                        programs = selected_candidate.programs
                        selected_index = selected_candidate.selected_strength_index
                        if validation is None or selected_index is None:
                            raise ExternalBackendError(
                                "quality-ranked candidate omitted compiled selector evidence"
                            )
                    else:
                        validation, programs = p_return(
                            chains, task.logical, task.host, task.problem, context
                        )
                        validators += 1
                        compilers += len(programs)
                        if not validation.valid or len(programs) != 4 or any(
                            not isinstance(program, Program) for program in programs
                        ):
                            raise ExternalBackendError(
                                "resource-selected external embedding failed final validation"
                            )
                        selected_index = _select_strength(
                            selector, context, task, chains, programs
                        )
                    selected_program = programs[selected_index]
                    online_seconds = time.perf_counter() - started
                    validation_digest = stable_digest(validation.as_dict())
                    if online_seconds > config.online_wallclock_seconds + max(
                        1e-9, config.online_wallclock_seconds * 1e-9
                    ):
                        outcome = _failure_outcome(
                            task,
                            repetition=repetition,
                            system_seed=system_seed,
                            reason="EXTERNAL_COMPLETE_SYSTEM_WALLCLOCK_EXHAUSTED",
                            validation_digest=validation_digest,
                            online_seconds=online_seconds,
                        )
                    elif config.audit_reads > context.caps.evaluator_reads:
                        outcome = _failure_outcome(
                            task,
                            repetition=repetition,
                            system_seed=system_seed,
                            reason="EXTERNAL_KNOWN_WORK_CAP_EXHAUSTED",
                            validation_digest=validation_digest,
                            online_seconds=online_seconds,
                        )
                    else:
                        evaluator_started = time.perf_counter()
                        block = sample_program(
                            selected_program,
                            chains,
                            task.problem,
                            task.ground_energy,
                            num_reads=config.audit_reads,
                            seed=expected_evaluator_seed,
                            num_sweeps=context.num_sweeps,
                        )
                        evaluator_seconds = time.perf_counter() - evaluator_started
                        if (
                            block.strength_index != selected_index
                            or block.reads != config.audit_reads
                        ):
                            raise ExternalBackendError(
                                "external evaluator returned a different program/read block"
                            )
                        evaluator_reads = block.reads
                        outcome = EpisodeOutcome(
                            instance=task.name,
                            lineage=task.lineage or task.name,
                            returned_valid=True,
                            utility=block.rate,
                            qubits=validation.qubits,
                            max_chain=validation.max_chain,
                            decisions=0,
                            selected_strength=selected_program.strength,
                            reason="EXTERNAL_VALID_RETURN",
                            repetition=repetition,
                            episode_seed=system_seed,
                            evaluator_seed=expected_evaluator_seed,
                            program_digest=program_digest(selected_program, chains),
                            selected_strength_index=selected_index,
                            evaluator_hits=block.hits,
                            evaluator_reads=block.reads,
                            work=None,
                            validation_digest=validation_digest,
                            broken_chain_fraction=block.broken_fraction,
                            mean_energy_residual=block.mean_residual,
                            online_seconds=online_seconds,
                            evaluator_seconds=evaluator_seconds,
                            chain_sizes=tuple(
                                sorted(len(chain) for chain in chains.values())
                            ),
                        )
                        outcome.validate_receipt(require_complete=False)
                        terminal_evidence = TerminalEvidence(
                            embedding=_canonical_embedding(chains),
                            programs=tuple(
                                _canonical_program(program) for program in programs
                            ),
                            selected_index=selected_index,
                            selected_program_digest=program_digest(
                                selected_program, chains
                            ),
                            evaluator_seed=expected_evaluator_seed,
                            evaluator_hits=block.hits,
                            evaluator_reads=block.reads,
                            broken_chain_fraction=block.broken_fraction,
                            mean_energy_residual=block.mean_residual,
                            validation_receipt=validation.as_dict(),
                            validation_digest=validation_digest,
                        )

            postprocessing_seconds = max(0.0, online_seconds - search_seconds)
            observed_work = _partial_work(
                restarts=len(restarts),
                validators=validators,
                compilers=compilers,
                reads=evaluator_reads,
            )
            receipt = ExternalCompleteSystemReceipt(
                instance=task.name,
                lineage=task.lineage or task.name,
                repetition=repetition,
                training_seed_index=training_seed_index,
                training_seed=training_seed,
                system_seed=system_seed,
                evaluator_seed=(expected_evaluator_seed if outcome.returned_valid else None),
                backend=backend_identity,
                selector=selector_identity,
                population=population,
                config_digest=config.digest,
                learned_config_digest=learned_config.digest,
                context_digest=context_digest,
                quality_authority_digest=quality_digest,
                runtime_identity_digest=runtime_digest,
                work_cap=context.caps,
                restarts=tuple(restarts),
                selected_restart=selected_restart,
                outcome=outcome,
                terminal_evidence_digest=(
                    None if terminal_evidence is None else terminal_evidence.digest
                ),
                terminal_evidence=terminal_evidence,
                observed_work=observed_work,
                cap_compliance=observed_work.cap_compliance(context.caps),
                adapter_startup_seconds=adapter_startup_seconds,
                search_seconds=search_seconds,
                postprocessing_seconds=postprocessing_seconds,
                online_seconds=online_seconds,
                evaluator_seconds=evaluator_seconds,
                wallclock_cap_seconds=config.online_wallclock_seconds,
                wallclock_compliant=(
                    online_seconds
                    <= config.online_wallclock_seconds
                    + max(1e-9, config.online_wallclock_seconds * 1e-9)
                ),
                tuning_execution=tuning_execution,
            )
            outcomes.append(outcome)
            receipts.append(receipt)
    population.validate_tasks(tasks)
    if backend.identity != backend_identity:
        raise ExternalBackendError(
            "external backend identity changed before evaluation completed"
        )
    return outcomes, receipts


def _receipt_bytes(receipts: Sequence[ExternalCompleteSystemReceipt]) -> bytes:
    return b"".join(
        (
            json.dumps(
                receipt.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
        for receipt in sorted(receipts, key=lambda item: item.pair_key)
    )


def _outcome_bytes(outcomes: Sequence[EpisodeOutcome]) -> bytes:
    return b"".join(
        (
            json.dumps(
                outcome.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
        for outcome in sorted(outcomes, key=lambda item: item.pair_key)
    )


@dataclass(frozen=True)
class ExternalCompleteSystemEvidenceRecord:
    """Replayable stock terminal evidence linked to one authenticated receipt."""

    instance: str
    lineage: str
    repetition: int
    population_digest: str
    external_receipt_digest: str
    terminal_evidence: TerminalEvidence | None

    def __post_init__(self) -> None:
        if not self.instance or not self.lineage or type(self.repetition) is not int:
            raise ValueError("external evidence identity is malformed")
        if self.repetition < 0 or not _is_digest(self.population_digest) or not _is_digest(
            self.external_receipt_digest
        ):
            raise ValueError("external evidence links are malformed")
        if self.terminal_evidence is not None and not isinstance(
            self.terminal_evidence, TerminalEvidence
        ):
            raise TypeError("external terminal evidence has the wrong type")

    @property
    def pair_key(self) -> tuple[str, str, int]:
        return (self.lineage, self.instance, self.repetition)

    def as_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": EXTERNAL_COMPLETE_EVIDENCE_SCHEMA,
            "schema_version": EXTERNAL_COMPLETE_EVIDENCE_VERSION,
            "instance": self.instance,
            "lineage": self.lineage,
            "repetition": self.repetition,
            "population_digest": self.population_digest,
            "external_receipt_digest": self.external_receipt_digest,
            "terminal_evidence": (
                None if self.terminal_evidence is None else self.terminal_evidence.as_dict()
            ),
        }
        if include_digest:
            payload["record_digest"] = stable_digest(payload)
        return payload


def _evidence_bytes(records: Sequence[ExternalCompleteSystemEvidenceRecord]) -> bytes:
    return b"".join(
        (
            json.dumps(
                record.as_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
        for record in sorted(records, key=lambda item: item.pair_key)
    )


def write_external_complete_evidence(
    path: str | os.PathLike[str],
    receipts: Sequence[ExternalCompleteSystemReceipt],
    *,
    overwrite: bool = False,
) -> str:
    """Write one replayable evidence-or-absence row for every sealed stock attempt."""

    if not receipts:
        raise EvaluationProtocolError("cannot write an empty external evidence batch")
    records = tuple(
        ExternalCompleteSystemEvidenceRecord(
            instance=receipt.instance,
            lineage=receipt.lineage,
            repetition=receipt.repetition,
            population_digest=receipt.population.digest,
            external_receipt_digest=str(receipt.as_dict()["record_digest"]),
            terminal_evidence=receipt.terminal_evidence,
        )
        for receipt in receipts
    )
    keys = [record.pair_key for record in records]
    if len(keys) != len(set(keys)):
        raise EvaluationProtocolError("external evidence batch has duplicate pair keys")
    content = _evidence_bytes(records)
    digest = hashlib.sha256(content).hexdigest()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def read_external_complete_evidence(
    path: str | os.PathLike[str],
    *,
    receipts: Sequence[ExternalCompleteSystemReceipt],
    tasks: Sequence[EmbeddingTask],
    context: Context,
    expected_sha256: str,
) -> tuple[ExternalCompleteSystemEvidenceRecord, ...]:
    """Authenticate and independently recompile every valid stock terminal result."""

    content = Path(path).read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise EvaluationProtocolError("external terminal-evidence file digest mismatch")
    receipt_by_key = {receipt.pair_key: receipt for receipt in receipts}
    task_by_key = {(task.lineage or task.name, task.name): task for task in tasks}
    if len(receipt_by_key) != len(receipts) or len(task_by_key) != len(tasks):
        raise EvaluationProtocolError("external evidence authorities contain duplicate identities")
    records: list[ExternalCompleteSystemEvidenceRecord] = []
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise EvaluationProtocolError("external evidence artifact is not UTF-8") from exc
    if not lines or any(not line for line in lines):
        raise EvaluationProtocolError("external evidence artifact is empty or contains blanks")
    expected_fields = {
        "schema",
        "schema_version",
        "instance",
        "lineage",
        "repetition",
        "population_digest",
        "external_receipt_digest",
        "terminal_evidence",
        "record_digest",
    }
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != expected_fields:
                raise ValueError("evidence fields differ")
            if (
                row["schema"] != EXTERNAL_COMPLETE_EVIDENCE_SCHEMA
                or row["schema_version"] != EXTERNAL_COMPLETE_EVIDENCE_VERSION
                or row["record_digest"]
                != stable_digest(
                    {key: value for key, value in row.items() if key != "record_digest"}
                )
            ):
                raise ValueError("evidence schema or digest differs")
            raw_terminal = row["terminal_evidence"]
            terminal = (
                None
                if raw_terminal is None
                else _terminal_evidence_from_payload(raw_terminal)
            )
            record = ExternalCompleteSystemEvidenceRecord(
                instance=row["instance"],
                lineage=row["lineage"],
                repetition=row["repetition"],
                population_digest=row["population_digest"],
                external_receipt_digest=row["external_receipt_digest"],
                terminal_evidence=terminal,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationProtocolError(
                f"external evidence line {line_number} is invalid"
            ) from exc
        if record.as_dict() != row:
            raise EvaluationProtocolError("external evidence record is not canonical")
        records.append(record)
    if _evidence_bytes(records) != content:
        raise EvaluationProtocolError("external evidence artifact is not canonical")
    if {record.pair_key for record in records} != set(receipt_by_key):
        raise EvaluationProtocolError("external evidence and receipt pair sets differ")
    if {(lineage, instance) for lineage, instance, _ in receipt_by_key} != set(task_by_key):
        raise EvaluationProtocolError("external evidence task set differs from its population")
    for record in records:
        receipt = receipt_by_key[record.pair_key]
        terminal_digest = (
            None if record.terminal_evidence is None else record.terminal_evidence.digest
        )
        if (
            record.population_digest != receipt.population.digest
            or record.external_receipt_digest != receipt.as_dict()["record_digest"]
            or terminal_digest != receipt.terminal_evidence_digest
            or receipt.outcome.returned_valid != (record.terminal_evidence is not None)
        ):
            raise EvaluationProtocolError("external evidence does not match its receipt link")
        if record.terminal_evidence is not None:
            verify_terminal_evidence(
                record.terminal_evidence,
                task=task_by_key[(record.lineage, record.instance)],
                context=context,
                outcome=receipt.outcome,
            )
    return tuple(sorted(records, key=lambda item: item.pair_key))


def write_external_complete_outcomes(
    path: str | os.PathLike[str],
    outcomes: Sequence[EpisodeOutcome],
    *,
    overwrite: bool = False,
) -> str:
    """Write external outcomes without pretending unavailable native work is exact."""

    if not outcomes:
        raise EvaluationProtocolError("cannot write an empty external outcome batch")
    keys = [outcome.pair_key for outcome in outcomes]
    if len(keys) != len(set(keys)):
        raise EvaluationProtocolError("external outcome batch has duplicate pair keys")
    for outcome in outcomes:
        outcome.validate_receipt(require_complete=False)
    content = _outcome_bytes(outcomes)
    digest = hashlib.sha256(content).hexdigest()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def read_external_complete_outcomes(
    path: str | os.PathLike[str], *, expected_sha256: str
) -> list[EpisodeOutcome]:
    content = Path(path).read_bytes()
    observed = hashlib.sha256(content).hexdigest()
    if observed != expected_sha256:
        raise EvaluationProtocolError(
            f"external outcome SHA-256 mismatch: expected {expected_sha256}, observed {observed}"
        )
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise EvaluationProtocolError("external outcome artifact is not UTF-8") from exc
    if not lines or any(not line.strip() for line in lines):
        raise EvaluationProtocolError("external outcome artifact is empty or contains blanks")
    outcomes: list[EpisodeOutcome] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationProtocolError(
                f"invalid external outcome JSON on line {line_number}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise EvaluationProtocolError("external outcome row must be an object")
        outcomes.append(EpisodeOutcome.from_dict(payload, require_complete=False))
    if len({outcome.pair_key for outcome in outcomes}) != len(outcomes):
        raise EvaluationProtocolError("external outcome artifact has duplicate pair keys")
    if _outcome_bytes(outcomes) != content:
        raise EvaluationProtocolError("external outcome artifact is not canonical")
    return outcomes


def write_external_complete_receipts(
    path: str | os.PathLike[str],
    receipts: Sequence[ExternalCompleteSystemReceipt],
    *,
    overwrite: bool = False,
) -> str:
    if not receipts:
        raise EvaluationProtocolError("cannot write an empty external complete receipt batch")
    keys = [receipt.pair_key for receipt in receipts]
    if len(keys) != len(set(keys)):
        raise EvaluationProtocolError("external complete receipt batch has duplicate pair keys")
    content = _receipt_bytes(receipts)
    digest = hashlib.sha256(content).hexdigest()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def read_external_complete_receipts(
    path: str | os.PathLike[str],
    *,
    outcomes: Sequence[EpisodeOutcome],
    expected_backend: BackendIdentity,
    expected_selector: FrozenComponentIdentity,
    expected_population: CompletePopulationIdentity,
    expected_config_digest: str,
    expected_learned_config_digest: str,
    expected_context_digest: str,
    expected_quality_authority_digest: str,
    expected_work_cap: WorkVector,
    expected_sha256: str,
    expected_training_seed_index: int = 0,
    expected_training_seed: int = REGISTERED_TRAINING_SEEDS[0],
    expected_runtime_identity_digest: str | None = None,
    expected_tuning_execution: ExternalTuningExecutionBinding | None = None,
) -> list[ExternalCompleteSystemReceipt]:
    """Read canonical v3 receipts against authorities supplied outside the JSONL file."""

    if expected_runtime_identity_digest is None:
        expected_runtime_identity_digest = content_digest(
            runtime_identity(
                runtime_platform={"test_or_library_runtime": "unspecified"},
                inference_device_type="cpu",
                inference_device_name="unspecified-cpu",
                inference_threads=1,
                deterministic=True,
            )
        )

    content = Path(path).read_bytes()
    observed_sha = hashlib.sha256(content).hexdigest()
    if observed_sha != expected_sha256:
        raise EvaluationProtocolError(
            f"external complete receipt SHA-256 mismatch: expected {expected_sha256}, "
            f"observed {observed_sha}"
        )
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise EvaluationProtocolError("external complete receipts are not UTF-8") from exc
    if not lines or any(not line.strip() for line in lines):
        raise EvaluationProtocolError("external complete receipts are empty or contain blanks")
    outcome_by_pair = {outcome.pair_key: outcome for outcome in outcomes}
    if len(outcome_by_pair) != len(outcomes):
        raise EvaluationProtocolError("external outcome artifact has duplicate pair keys")
    receipts: list[ExternalCompleteSystemReceipt] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationProtocolError(
                f"invalid external complete JSON on line {line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise EvaluationProtocolError("external complete receipt row is not an object")
        key = (row.get("lineage"), row.get("instance"), row.get("repetition"))
        outcome = outcome_by_pair.get(key)  # type: ignore[arg-type]
        if outcome is None or row.get("outcome") != outcome.as_dict():
            raise EvaluationProtocolError(
                "external complete receipt does not bind its outcome artifact"
            )
        if (
            row.get("backend") != expected_backend.as_dict()
            or row.get("selector") != expected_selector.as_dict()
            or row.get("population") != expected_population.as_dict()
            or row.get("population_digest") != expected_population.digest
            or row.get("config_digest") != expected_config_digest
            or row.get("learned_config_digest") != expected_learned_config_digest
            or row.get("context_digest") != expected_context_digest
            or row.get("quality_authority_digest") != expected_quality_authority_digest
            or row.get("training_seed_index") != expected_training_seed_index
            or row.get("training_seed") != expected_training_seed
            or row.get("runtime_identity_digest") != expected_runtime_identity_digest
            or row.get("work_cap") != expected_work_cap.as_dict()
            or row.get("external_tuning_execution")
            != (
                None
                if expected_tuning_execution is None
                else expected_tuning_execution.as_dict()
            )
            or row.get("external_tuning_execution_digest")
            != (
                None
                if expected_tuning_execution is None
                else expected_tuning_execution.digest
            )
        ):
            raise EvaluationProtocolError(
                "external complete receipt differs from an externally supplied authority"
            )
        raw_restarts = row.get("restarts")
        if not isinstance(raw_restarts, list):
            raise EvaluationProtocolError("external restart evidence must be an array")
        try:
            restarts = tuple(
                ExternalRestartEvidence(
                    restart_index=item["restart_index"],
                    seed=item["seed"],
                    status=item["status"],
                    reported_seconds=item["reported_seconds"],
                    observed_seconds=item["observed_seconds"],
                    embedding_digest=item["embedding_digest"],
                    validation_digest=item["validation_digest"],
                    qubits=item["qubits"],
                    max_chain=item["max_chain"],
                    selector_max_p_solve=item["selector_max_p_solve"],
                    selector_strength_index=item["selector_strength_index"],
                    detail=item["detail"],
                )
                for item in raw_restarts
                if isinstance(item, dict)
                and set(item)
                == {
                    "restart_index",
                    "seed",
                    "status",
                    "reported_seconds",
                    "observed_seconds",
                    "embedding_digest",
                    "validation_digest",
                    "qubits",
                    "max_chain",
                    "selector_max_p_solve",
                    "selector_strength_index",
                    "detail",
                }
            )
            if len(restarts) != len(raw_restarts):
                raise ValueError("malformed restart row")
            observed_mapping = row["observed_work"]
            lower_mapping = row["observed_work_known_lower_bound"]
            if not isinstance(observed_mapping, Mapping) or not isinstance(
                lower_mapping, Mapping
            ):
                raise ValueError("malformed work mapping")
            observed_work = PartialWorkVector(
                **{name: observed_mapping[name] for name in WORK_FIELDS},
                known_lower_bound=WorkVector(
                    **{name: lower_mapping[name] for name in WORK_FIELDS}
                ),
            )
            receipt = ExternalCompleteSystemReceipt(
                instance=row["instance"],
                lineage=row["lineage"],
                repetition=row["repetition"],
                training_seed_index=row["training_seed_index"],
                training_seed=row["training_seed"],
                system_seed=row["system_seed"],
                evaluator_seed=row["evaluator_seed"],
                backend=expected_backend,
                selector=expected_selector,
                population=expected_population,
                config_digest=row["config_digest"],
                learned_config_digest=row["learned_config_digest"],
                context_digest=row["context_digest"],
                quality_authority_digest=row["quality_authority_digest"],
                runtime_identity_digest=row["runtime_identity_digest"],
                work_cap=expected_work_cap,
                restarts=restarts,
                selected_restart=row["selected_restart"],
                outcome=outcome,
                terminal_evidence_digest=row["terminal_evidence_digest"],
                terminal_evidence=None,
                observed_work=observed_work,
                cap_compliance=row["cap_compliance"],
                adapter_startup_seconds=row["adapter_startup_seconds"],
                search_seconds=row["search_seconds"],
                postprocessing_seconds=row["postprocessing_seconds"],
                online_seconds=row["online_seconds"],
                evaluator_seconds=row["evaluator_seconds"],
                wallclock_cap_seconds=row["wallclock_cap_seconds"],
                wallclock_compliant=row["wallclock_compliant"],
                tuning_execution=expected_tuning_execution,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationProtocolError(
                f"external complete receipt line {line_number} is invalid"
            ) from exc
        if receipt.as_dict() != row:
            raise EvaluationProtocolError(
                "external complete receipt has unknown fields or a digest mismatch"
            )
        receipts.append(receipt)
    if set(outcome_by_pair) != {receipt.pair_key for receipt in receipts}:
        raise EvaluationProtocolError("external receipt and outcome censuses differ")
    if _receipt_bytes(receipts) != content:
        raise EvaluationProtocolError("external complete receipt file is not canonical")
    return receipts


@dataclass(frozen=True)
class AuthenticatedExternalCompleteRun:
    """A canonical report plus the exact typed receipt and outcome artifacts it names."""

    report: Mapping[str, object]
    receipts: tuple[ExternalCompleteSystemReceipt, ...]
    evidence: tuple[ExternalCompleteSystemEvidenceRecord, ...]
    receipt_file_sha256: str
    evidence_file_sha256: str
    outcome_file_sha256: str

    def __post_init__(self) -> None:
        report = _canonical_mapping(self.report, label="external complete report")
        object.__setattr__(self, "report", report)
        if not self.receipts:
            raise ValueError("external complete run cannot be empty")
        if not self.evidence or any(
            not isinstance(item, ExternalCompleteSystemEvidenceRecord)
            for item in self.evidence
        ):
            raise TypeError("external complete run requires typed terminal evidence")
        if any(
            not _is_digest(value)
            for value in (
                self.receipt_file_sha256,
                self.evidence_file_sha256,
                self.outcome_file_sha256,
            )
        ):
            raise ValueError("external complete artifact hashes must be SHA-256 digests")
        if hashlib.sha256(_receipt_bytes(self.receipts)).hexdigest() != self.receipt_file_sha256:
            raise ValueError("external receipt file identity differs from typed receipts")
        if hashlib.sha256(_evidence_bytes(self.evidence)).hexdigest() != self.evidence_file_sha256:
            raise ValueError("external evidence file identity differs from typed evidence")
        if (
            hashlib.sha256(
                _outcome_bytes([receipt.outcome for receipt in self.receipts])
            ).hexdigest()
            != self.outcome_file_sha256
        ):
            raise ValueError("external outcome file identity differs from typed receipts")
        payload = dict(report)
        recorded = payload.pop("record_digest", None)
        if recorded != content_digest(payload):
            raise ValueError("external complete report digest mismatch")
        if (
            set(report) != _COMPLETE_REPORT_KEYS
            or
            report.get("schema") != EXTERNAL_COMPLETE_REPORT_SCHEMA
            or report.get("schema_version") != EXTERNAL_COMPLETE_REPORT_VERSION
            or report.get("status") != "complete"
            or report.get("sealed_test_opened") is not True
        ):
            raise ValueError("external complete report has an unsupported contract")
        if report.get("partition") != "test" or report.get("no_fallback") is not True:
            raise ValueError("external complete report is not the registered test arm")
        protocol = report.get("evaluation_protocol")
        config_pin_fields = {
            "learned_config_digest",
            "learned_config_file_sha256",
            "external_config_digest",
            "external_config_file_sha256",
        }
        if (
            not isinstance(protocol, Mapping)
            or set(protocol) != set(_FROZEN_PROTOCOL) | config_pin_fields
            or {key: protocol[key] for key in _FROZEN_PROTOCOL} != _FROZEN_PROTOCOL
            or any(not _is_digest(protocol[name]) for name in config_pin_fields)
        ):
            raise ValueError("external complete report changes the frozen test protocol")
        for name in (
            "grid_manifest_sha256",
            "source_corpus_manifest_sha256",
            "context_digest",
            "population_digest",
            "selector_digest",
            "external_tuning_execution_digest",
            "external_config_digest",
            "external_config_file_sha256",
            "learned_config_digest",
            "learned_config_file_sha256",
            "external_pairing_implementation_sha256",
        ):
            if not _is_digest(report.get(name)):
                raise ValueError(f"external complete report {name} is not SHA-256")
        reference = self.receipts[0]
        if (
            report.get("training_seed_index") != reference.training_seed_index
            or report.get("training_seed") != reference.training_seed
        ):
            raise ValueError("external report training-seed pairing differs from receipts")
        if (
            reference.backend.method_id != STOCK_MINORMINER_METHOD
            or reference.backend.distribution != "minorminer"
            or reference.backend.entrypoint != "minorminer.find_embedding"
        ):
            raise ValueError("external complete run is not the registered stock backend")
        expected_keys = {
            (lineage, instance, repetition)
            for lineage, instance in reference.population.expected_instances
            for repetition in range(reference.population.expected_repetitions)
        }
        keys = [receipt.pair_key for receipt in self.receipts]
        if len(keys) != len(set(keys)) or set(keys) != expected_keys:
            raise ValueError("external receipts do not cover the sealed population census")
        for receipt in self.receipts:
            if (
                receipt.population != reference.population
                or receipt.backend != reference.backend
                or receipt.selector != reference.selector
                or receipt.config_digest != reference.config_digest
                or receipt.learned_config_digest != reference.learned_config_digest
                or receipt.context_digest != reference.context_digest
                or receipt.quality_authority_digest != reference.quality_authority_digest
                or receipt.training_seed_index != reference.training_seed_index
                or receipt.training_seed != reference.training_seed
                or receipt.runtime_identity_digest != reference.runtime_identity_digest
                or receipt.tuning_execution != reference.tuning_execution
                or receipt.work_cap != reference.work_cap
                or receipt.wallclock_cap_seconds != reference.wallclock_cap_seconds
            ):
                raise ValueError("external receipt batch mixes authenticated contracts")
        artifacts = report.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "receipts",
            "terminal_evidence",
            "outcomes",
        }:
            raise ValueError("external complete report has an incomplete artifact registry")
        expected_artifacts = {
            "receipts": ("external_receipts.jsonl", self.receipt_file_sha256),
            "terminal_evidence": (
                "terminal_evidence.jsonl",
                self.evidence_file_sha256,
            ),
            "outcomes": ("outcomes.jsonl", self.outcome_file_sha256),
        }
        for name, (path, sha) in expected_artifacts.items():
            entry = artifacts[name]
            if not isinstance(entry, Mapping) or dict(entry) != {
                "path": path,
                "sha256": sha,
                "count": len(self.receipts),
            }:
                raise ValueError(f"external {name} artifact registry differs")
        evidence_by_pair = {item.pair_key: item for item in self.evidence}
        if len(evidence_by_pair) != len(self.evidence) or set(evidence_by_pair) != set(keys):
            raise ValueError("external terminal evidence does not cover the sealed census")
        for receipt in self.receipts:
            evidence = evidence_by_pair[receipt.pair_key]
            observed_digest = (
                None if evidence.terminal_evidence is None else evidence.terminal_evidence.digest
            )
            if (
                evidence.population_digest != receipt.population.digest
                or evidence.external_receipt_digest != receipt.as_dict()["record_digest"]
                or observed_digest != receipt.terminal_evidence_digest
                or receipt.outcome.returned_valid != (evidence.terminal_evidence is not None)
            ):
                raise ValueError("external terminal evidence link differs from receipts")
        quality = report.get("quality_authority")
        target_access = report.get("target_access")
        ground_partition = report.get("ground_partition_receipt")
        context = report.get("context")
        if not isinstance(quality, Mapping) or content_digest(quality) != reference.quality_authority_digest:
            raise ValueError("external report quality authority differs from receipts")
        if not isinstance(target_access, Mapping) or not isinstance(
            ground_partition, Mapping
        ):
            raise ValueError("external report omits target-access receipts")
        _verify_record(target_access, "external report target access")
        _verify_record(ground_partition, "external report ground partition")
        partition_authority = quality.get("evaluation_partition")
        ground_identity = (
            partition_authority.get("ground_partition")
            if isinstance(partition_authority, Mapping)
            else None
        )
        if (
            not isinstance(partition_authority, Mapping)
            or target_access.get("record_digest")
            != partition_authority.get("target_access_record_digest")
            or not isinstance(ground_identity, Mapping)
            or ground_partition.get("record_digest")
            != ground_identity.get("receipt_record_digest")
        ):
            raise ValueError("external report target access differs from authority")
        if not isinstance(context, Mapping) or stable_digest(
            _plain_json(context)
        ) != reference.context_digest:
            raise ValueError("external report context differs from receipts")
        if (
            _plain_json(report.get("population")) != reference.population.as_dict()
            or report.get("population_digest") != reference.population.digest
            or report.get("selector") != reference.selector.as_dict()
            or report.get("selector_digest")
            != content_digest(reference.selector.as_dict())
            or reference.tuning_execution is None
            or reference.tuning_execution.mode != "frozen-deployment"
            or report.get("external_tuning_execution")
            != reference.tuning_execution.as_dict()
            or report.get("external_tuning_execution_digest")
            != reference.tuning_execution.digest
            or report.get("backend") != reference.backend.as_dict()
            or report.get("external_config_digest") != reference.config_digest
            or report.get("learned_config_digest") != reference.learned_config_digest
            or report.get("work_cap") != reference.work_cap.as_dict()
        ):
            raise ValueError("external report and typed receipts bind different contracts")
        external_config = report.get("external_config")
        learned_config = report.get("learned_config")
        if not isinstance(external_config, Mapping) or not isinstance(
            learned_config, Mapping
        ):
            raise ValueError("external report omits one complete-system config payload")
        parsed_external = ExternalCompleteSystemConfig.from_mapping(external_config)
        parsed_learned = CompleteSystemConfig.from_mapping(learned_config)
        if (
            parsed_external.digest != reference.config_digest
            or parsed_learned.digest != reference.learned_config_digest
            or not math.isclose(
                parsed_external.online_wallclock_seconds,
                parsed_learned.online_wallclock_seconds,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or parsed_external.audit_reads != parsed_learned.audit_reads
            or parsed_external.selection_rule != parsed_learned.selection_rule
        ):
            raise ValueError("external report does not preserve the symmetric budget envelope")
        if (
            parsed_external.digest != protocol["external_config_digest"]
            or report["external_config_file_sha256"]
            != protocol["external_config_file_sha256"]
            or parsed_learned.digest != protocol["learned_config_digest"]
            or report["learned_config_file_sha256"]
            != protocol["learned_config_file_sha256"]
        ):
            raise ValueError("external report configs differ from preregistered pins")
        from isingfold.rl.external import STOCK_MINORMINER_IMPLEMENTATION_PREFIX

        implementation_prefix = (
            f"{STOCK_MINORMINER_IMPLEMENTATION_PREFIX}:artifact-record-sha256:"
        )
        artifact_digest = reference.backend.implementation.removeprefix(
            implementation_prefix
        )
        if (
            reference.backend.version != parsed_external.expected_backend_version
            or not reference.backend.implementation.startswith(implementation_prefix)
            or not _is_digest(artifact_digest)
        ):
            raise ValueError("external backend version or implementation differs from config")
        identity_payload = {
            name: report.get(name) for name in _RUNTIME_IDENTITY_KEYS
        }
        try:
            sealed_runtime = runtime_identity(**identity_payload)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError("external report runtime identity is malformed") from exc
        if content_digest(sealed_runtime) != reference.runtime_identity_digest:
            raise ValueError("external report runtime identity differs from receipts")
        registry = report.get("runtime_implementation_registry")
        if not isinstance(registry, Mapping) or not registry:
            raise ValueError("external report omits its runtime implementation registry")
        if content_digest(registry) != report.get("runtime_implementation_digest"):
            raise ValueError("external runtime implementation registry digest mismatch")
        modules = registry.get("modules")
        dependencies = registry.get("dependencies")
        expected_modules = dict(RUNTIME_MODULE_SOURCES)
        if (
            registry.get("schema") != RUNTIME_IMPLEMENTATION_SCHEMA
            or registry.get("schema_version") != RUNTIME_IMPLEMENTATION_VERSION
            or not isinstance(modules, Mapping)
            or not isinstance(dependencies, Mapping)
            or set(modules) != set(expected_modules)
            or set(dependencies) != {"python", *RUNTIME_DEPENDENCIES}
            or dependencies.get("minorminer") != parsed_external.expected_backend_version
        ):
            raise ValueError("external runtime registry omits required implementation identity")
        for name, path in expected_modules.items():
            entry = modules[name]
            if (
                not isinstance(entry, Mapping)
                or set(entry) != {"source_path", "sha256"}
                or entry.get("source_path") != path
                or not _is_digest(entry.get("sha256"))
            ):
                raise ValueError("external runtime source identity is malformed")
        pairing_entry = modules["isingfold.rl.external_pairing"]
        assert isinstance(pairing_entry, Mapping)
        if report["external_pairing_implementation_sha256"] != pairing_entry["sha256"]:
            raise ValueError("external pairing source identity differs from runtime registry")
        if any(not isinstance(value, str) or not value for value in dependencies.values()):
            raise ValueError("external runtime dependency identity is malformed")
        if reference.wallclock_cap_seconds != parsed_external.online_wallclock_seconds:
            raise ValueError("external receipts use another total online wall-clock cap")
        for receipt in self.receipts:
            assert receipt.tuning_execution is not None
            if len(receipt.restarts) > (
                receipt.tuning_execution.candidate.outer_restart_cap
            ):
                raise ValueError("external receipt exceeds the registered restart cap")
            expected_system = _expected_seed(
                reference.population.evaluation_seed, "policy", receipt.pair_key
            )
            expected_evaluator = _expected_seed(
                reference.population.evaluation_seed,
                "final-evaluator",
                receipt.pair_key,
            )
            if receipt.system_seed != expected_system:
                raise ValueError("external receipt uses an unregistered system seed")
            if receipt.outcome.returned_valid and receipt.evaluator_seed != expected_evaluator:
                raise ValueError("external receipt uses an unregistered evaluator seed")
        if reference.population.source_manifest_sha256 != report.get(
            "source_corpus_manifest_sha256"
        ):
            raise ValueError("external population refers to another corpus manifest")
        method = report.get("method")
        if (
            not isinstance(method, Mapping)
            or set(method)
            != {
                "method_id",
                "comparison_scope",
                "selection_rule",
                "online_evaluator_feedback",
                "total_online_wallclock_seconds",
                "final_evaluator",
                "native_internal_work",
                "adapter_process_model",
                "latency_scope",
            }
            or method.get("method_id") != STOCK_MINORMINER_METHOD
            or method.get("comparison_scope") != "paired-external-complete-system"
            or method.get("selection_rule")
            != reference.tuning_execution.candidate.candidate_ranking
            or method.get("online_evaluator_feedback") is not False
            or method.get("total_online_wallclock_seconds")
            != parsed_external.online_wallclock_seconds
            or method.get("final_evaluator")
            != "one-fresh-independent-4096-read-block-after-selection"
            or method.get("native_internal_work") != "unavailable-retained-as-null"
            or method.get("adapter_process_model")
            != "one-persistent-python-worker-per-task-repetition"
            or method.get("latency_scope")
            != "same-machine-rowwise-total-online-wallclock-includes-adapter-startup"
        ):
            raise ValueError("external complete method metadata differs from the registry")
        expected_summary = external_complete_summary(
            [receipt.outcome for receipt in self.receipts], self.receipts
        )
        if _plain_json(report.get("summary")) != expected_summary:
            raise ValueError("external complete report summary differs from raw receipts")

    @property
    def report_file_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.report) + b"\n").hexdigest()


def _backend_from_payload(value: object) -> BackendIdentity:
    if not isinstance(value, Mapping) or set(value) != {
        "method_id",
        "distribution",
        "version",
        "entrypoint",
        "implementation",
    }:
        raise EvaluationProtocolError("external report backend identity is malformed")
    try:
        return BackendIdentity(**dict(value))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise EvaluationProtocolError("external report backend identity is invalid") from exc


def _component_from_payload(value: object) -> FrozenComponentIdentity:
    if not isinstance(value, Mapping) or set(value) != {
        "component_id",
        "version",
        "implementation",
        "artifact_sha256",
    }:
        raise EvaluationProtocolError("external report selector identity is malformed")
    try:
        return FrozenComponentIdentity(**dict(value))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise EvaluationProtocolError("external report selector identity is invalid") from exc


def _population_from_payload(value: object) -> CompletePopulationIdentity:
    expected = {
        "population_id",
        "source_manifest_sha256",
        "task_payload_sha256",
        "expected_instances",
        "expected_repetitions",
        "evaluation_seed",
        "evaluation_strata",
        "evaluation_strata_digest",
        "size_bin_protocol",
        "size_bin_boundary_convention",
        "confirmatory_design",
        "confirmatory_design_digest",
        "denominator_scope",
        "includes_initializer_failures",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise EvaluationProtocolError("external report population identity is malformed")
    instances = value["expected_instances"]
    if (
        not isinstance(instances, Sequence)
        or isinstance(instances, (str, bytes, bytearray))
        or any(
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes, bytearray))
            or len(item) != 2
            for item in instances
        )
    ):
        raise EvaluationProtocolError("external report population census is malformed")
    raw_strata = value["evaluation_strata"]
    raw_design = value["confirmatory_design"]
    if not isinstance(raw_strata, list) or any(
        not isinstance(row, Mapping) for row in raw_strata
    ):
        raise EvaluationProtocolError("external report evaluation strata are malformed")
    if not isinstance(raw_design, Mapping):
        raise EvaluationProtocolError("external report confirmatory design is malformed")
    try:
        population = CompletePopulationIdentity(
            population_id=value["population_id"],  # type: ignore[arg-type]
            source_manifest_sha256=value["source_manifest_sha256"],  # type: ignore[arg-type]
            task_payload_sha256=value["task_payload_sha256"],  # type: ignore[arg-type]
            expected_instances=tuple((item[0], item[1]) for item in instances),  # type: ignore[misc]
            expected_repetitions=value["expected_repetitions"],  # type: ignore[arg-type]
            evaluation_seed=value["evaluation_seed"],  # type: ignore[arg-type]
            evaluation_strata=tuple(
                EvaluationStratum.from_mapping(row) for row in raw_strata
            ),
            confirmatory_design=ConfirmatoryEvaluationDesign.from_mapping(raw_design),
            denominator_scope=value["denominator_scope"],  # type: ignore[arg-type]
            includes_initializer_failures=value["includes_initializer_failures"],  # type: ignore[arg-type]
        )
        if (
            value["evaluation_strata_digest"] != population.evaluation_strata_digest
            or value["size_bin_protocol"] != SIZE_BIN_PROTOCOL
            or value["size_bin_boundary_convention"] != SIZE_BIN_BOUNDARY_CONVENTION
            or value["confirmatory_design_digest"]
            != population.confirmatory_design.record_digest
        ):
            raise ValueError("external population evaluation contract digests are invalid")
        return population
    except (TypeError, ValueError) as exc:
        raise EvaluationProtocolError("external report population identity is invalid") from exc


def load_authenticated_external_complete_run(
    directory: str | os.PathLike[str],
    *,
    expected_report_sha256: str,
    tasks: Sequence[EmbeddingTask],
    context: Context,
) -> AuthenticatedExternalCompleteRun:
    """Load a stock run only when its canonical report has an out-of-band file pin."""

    root = Path(directory)
    report_path = root / "report.json"
    content = report_path.read_bytes()
    observed = hashlib.sha256(content).hexdigest()
    if observed != expected_report_sha256:
        raise EvaluationProtocolError(
            f"external report SHA-256 mismatch: expected {expected_report_sha256}, "
            f"observed {observed}"
        )
    try:
        report = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvaluationProtocolError("external report is not valid UTF-8 JSON") from exc
    if not isinstance(report, dict) or canonical_json_bytes(report) + b"\n" != content:
        raise EvaluationProtocolError("external report is not canonical JSON")
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise EvaluationProtocolError("external report has no artifact registry")
    outcomes_entry = artifacts.get("outcomes")
    receipts_entry = artifacts.get("receipts")
    evidence_entry = artifacts.get("terminal_evidence")
    if (
        not isinstance(outcomes_entry, Mapping)
        or not isinstance(receipts_entry, Mapping)
        or not isinstance(evidence_entry, Mapping)
    ):
        raise EvaluationProtocolError("external report artifact entries are malformed")
    outcome_sha = outcomes_entry.get("sha256")
    receipt_sha = receipts_entry.get("sha256")
    evidence_sha = evidence_entry.get("sha256")
    if (
        not _is_digest(outcome_sha)
        or not _is_digest(receipt_sha)
        or not _is_digest(evidence_sha)
    ):
        raise EvaluationProtocolError("external report artifact SHA-256 is invalid")
    if (
        outcomes_entry.get("path") != "outcomes.jsonl"
        or receipts_entry.get("path") != "external_receipts.jsonl"
        or evidence_entry.get("path") != "terminal_evidence.jsonl"
    ):
        raise EvaluationProtocolError("external report changes registered artifact paths")
    backend = _backend_from_payload(report.get("backend"))
    selector = _component_from_payload(report.get("selector"))
    try:
        tuning_execution = ExternalTuningExecutionBinding.from_mapping(
            report.get("external_tuning_execution")
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationProtocolError(
            "external report has no authenticated tuned deployment binding"
        ) from exc
    if (
        tuning_execution.mode != "frozen-deployment"
        or report.get("external_tuning_execution_digest")
        != tuning_execution.digest
        or report.get("selector_digest") != content_digest(selector.as_dict())
    ):
        raise EvaluationProtocolError(
            "external report tuning or selector authority differs"
        )
    population = _population_from_payload(report.get("population"))
    population.validate_tasks(tasks)
    if (
        report.get("context_digest") != stable_digest(context_snapshot(context))
        or report.get("context") != context_snapshot(context)
        or report.get("work_cap") != context.caps.as_dict()
    ):
        raise EvaluationProtocolError(
            "external report context differs from the supplied replay authority"
        )
    work_payload = report.get("work_cap")
    if not isinstance(work_payload, Mapping) or set(work_payload) != set(WORK_FIELDS):
        raise EvaluationProtocolError("external report work cap is malformed")
    try:
        work_cap = WorkVector(**{name: work_payload[name] for name in WORK_FIELDS})
    except (TypeError, ValueError) as exc:
        raise EvaluationProtocolError("external report work cap is invalid") from exc
    quality = report.get("quality_authority")
    if not isinstance(quality, Mapping):
        raise EvaluationProtocolError("external report quality authority is malformed")
    outcomes = read_external_complete_outcomes(
        root / "outcomes.jsonl", expected_sha256=outcome_sha
    )
    receipts = read_external_complete_receipts(
        root / "external_receipts.jsonl",
        outcomes=outcomes,
        expected_backend=backend,
        expected_selector=selector,
        expected_population=population,
        expected_config_digest=report.get("external_config_digest"),  # type: ignore[arg-type]
        expected_learned_config_digest=report.get("learned_config_digest"),  # type: ignore[arg-type]
        expected_context_digest=report.get("context_digest"),  # type: ignore[arg-type]
        expected_quality_authority_digest=content_digest(quality),
        expected_work_cap=work_cap,
        expected_training_seed_index=report.get("training_seed_index"),  # type: ignore[arg-type]
        expected_training_seed=report.get("training_seed"),  # type: ignore[arg-type]
        expected_runtime_identity_digest=content_digest(
            {name: report.get(name) for name in _RUNTIME_IDENTITY_KEYS}
        ),
        expected_tuning_execution=tuning_execution,
        expected_sha256=receipt_sha,
    )
    evidence = read_external_complete_evidence(
        root / "terminal_evidence.jsonl",
        receipts=receipts,
        tasks=tasks,
        context=context,
        expected_sha256=evidence_sha,
    )
    return AuthenticatedExternalCompleteRun(
        report=report,
        receipts=tuple(receipts),
        evidence=evidence,
        receipt_file_sha256=receipt_sha,
        evidence_file_sha256=evidence_sha,
        outcome_file_sha256=outcome_sha,
    )


def _expected_seed(base_seed: int, domain: str, key: tuple[str, str, int]) -> int:
    lineage, instance, repetition = key
    payload = {
        "base_seed": base_seed,
        "domain": domain,
        "instance": instance,
        "lineage": lineage,
        "repetition": repetition,
    }
    return int.from_bytes(
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).digest()[:4],
        "big",
    ) % (2**31)


def external_complete_summary(
    outcomes: Sequence[EpisodeOutcome],
    receipts: Sequence[ExternalCompleteSystemReceipt],
) -> dict[str, object]:
    if not outcomes or len(outcomes) != len(receipts):
        raise ValueError("external summary needs aligned nonempty outcomes and receipts")
    if {outcome.pair_key for outcome in outcomes} != {receipt.pair_key for receipt in receipts}:
        raise ValueError("external summary outcome and receipt censuses differ")
    terminal_reasons = {
        reason: sum(receipt.outcome.reason == reason for receipt in receipts)
        for reason in (
            "EXTERNAL_VALID_RETURN",
            "EXTERNAL_NO_VALID_EMBEDDING",
            "EXTERNAL_COMPLETE_SYSTEM_WALLCLOCK_EXHAUSTED",
            "EXTERNAL_KNOWN_WORK_CAP_EXHAUSTED",
        )
    }
    if sum(terminal_reasons.values()) != len(receipts):
        raise ValueError("external receipts contain an unauthenticated terminal reason")
    restart_statuses = {
        status: sum(
            restart.status == status
            for receipt in receipts
            for restart in receipt.restarts
        )
        for status in (
            "VALID_CANDIDATE",
            "INVALID_CANDIDATE",
            "WORK_CAP_EXHAUSTED",
            SearchStatus.NO_EMBEDDING.value,
            SearchStatus.TIMED_OUT.value,
            SearchStatus.ERROR.value,
        )
    }
    return {
        **secondary_metrics(outcomes),
        "external_failure_taxonomy": {
            "terminal_reason_counts": terminal_reasons,
            "restart_status_counts": restart_statuses,
            "attempt_count": len(receipts),
            "valid_return_count": sum(receipt.outcome.returned_valid for receipt in receipts),
            "invalid_return_count": sum(
                not receipt.outcome.returned_valid for receipt in receipts
            ),
        },
        "search_seconds_mean": float(np.mean([receipt.search_seconds for receipt in receipts])),
        "adapter_startup_seconds_mean": float(
            np.mean([receipt.adapter_startup_seconds for receipt in receipts])
        ),
        "adapter_startup_seconds_total": float(
            sum(receipt.adapter_startup_seconds for receipt in receipts)
        ),
        "postprocessing_seconds_mean": float(
            np.mean([receipt.postprocessing_seconds for receipt in receipts])
        ),
        "online_seconds_mean": float(np.mean([receipt.online_seconds for receipt in receipts])),
        "online_seconds_total": float(sum(receipt.online_seconds for receipt in receipts)),
        "wallclock_noncompliant_attempts": sum(
            not receipt.wallclock_compliant for receipt in receipts
        ),
    }


@dataclass(frozen=True)
class _PairedMatrices:
    lineages: tuple[str, ...]
    attempts_per_training_seed: int
    unique_pair_count: int
    utility_difference: np.ndarray
    feasibility_difference: np.ndarray
    learned_utility: np.ndarray
    learned_feasibility: np.ndarray
    stock_utility: np.ndarray
    stock_feasibility: np.ndarray
    informative_discordant_lineages: tuple[str, ...]
    informative_discordant_pairs: tuple[tuple[str, str, int], ...]
    discordant_pair_occurrences: int
    total_pair_occurrences: int


def _paired_subset_matrices(
    learned_by_row: Sequence[Mapping[tuple[str, str, int], object]],
    stock_by_row: Sequence[Mapping[tuple[str, str, int], object]],
    identities: frozenset[tuple[str, str]],
) -> _PairedMatrices:
    if len(learned_by_row) != 3 or len(stock_by_row) != 3:
        raise ValueError("paired subgroup matrices require exactly three seed rows")
    utility_rows: list[dict[str, list[float]]] = []
    feasibility_rows: list[dict[str, list[float]]] = []
    learned_utility_rows: list[dict[str, list[float]]] = []
    learned_feasibility_rows: list[dict[str, list[float]]] = []
    stock_utility_rows: list[dict[str, list[float]]] = []
    stock_feasibility_rows: list[dict[str, list[float]]] = []
    discordant_lineages: set[str] = set()
    discordant_pairs: set[tuple[str, str, int]] = set()
    discordant_occurrences = 0
    total_occurrences = 0
    expected_keys: tuple[tuple[str, str, int], ...] | None = None
    for learned, stock in zip(learned_by_row, stock_by_row, strict=True):
        keys = tuple(sorted(key for key in learned if key[:2] in identities))
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError("training seeds disagree on registered subgroup pair census")
        if not keys or any(key not in stock for key in keys):
            raise ValueError("registered paired subgroup has no complete shared census")
        utility: dict[str, list[float]] = {}
        feasibility: dict[str, list[float]] = {}
        learned_utility: dict[str, list[float]] = {}
        learned_feasibility: dict[str, list[float]] = {}
        stock_utility: dict[str, list[float]] = {}
        stock_feasibility: dict[str, list[float]] = {}
        for key in keys:
            learned_outcome = learned[key].outcome  # type: ignore[attr-defined]
            stock_outcome = stock[key].outcome  # type: ignore[attr-defined]
            learned_value = float(learned_outcome.utility)
            stock_value = float(stock_outcome.utility)
            if not math.isfinite(learned_value) or not math.isfinite(stock_value):
                raise ValueError("paired subgroup contains a non-finite utility")
            learned_valid = float(learned_outcome.returned_valid)
            stock_valid = float(stock_outcome.returned_valid)
            lineage = key[0]
            utility.setdefault(lineage, []).append(learned_value - stock_value)
            feasibility.setdefault(lineage, []).append(learned_valid - stock_valid)
            learned_utility.setdefault(lineage, []).append(learned_value)
            learned_feasibility.setdefault(lineage, []).append(learned_valid)
            stock_utility.setdefault(lineage, []).append(stock_value)
            stock_feasibility.setdefault(lineage, []).append(stock_valid)
            total_occurrences += 1
            if learned_valid != stock_valid:
                discordant_occurrences += 1
                discordant_lineages.add(lineage)
                discordant_pairs.add(key)
        utility_rows.append(utility)
        feasibility_rows.append(feasibility)
        learned_utility_rows.append(learned_utility)
        learned_feasibility_rows.append(learned_feasibility)
        stock_utility_rows.append(stock_utility)
        stock_feasibility_rows.append(stock_feasibility)

    assert expected_keys is not None
    lineages = tuple(sorted(utility_rows[0]))
    all_rows = (
        utility_rows
        + feasibility_rows
        + learned_utility_rows
        + learned_feasibility_rows
        + stock_utility_rows
        + stock_feasibility_rows
    )
    if any(tuple(sorted(row)) != lineages for row in all_rows):
        raise ValueError("paired subgroup rows disagree on immutable lineage clusters")

    def matrix(rows: Sequence[Mapping[str, Sequence[float]]]) -> np.ndarray:
        return np.asarray(
            [
                [float(np.mean(row[lineage])) for lineage in lineages]
                for row in rows
            ],
            dtype=float,
        )

    return _PairedMatrices(
        lineages=lineages,
        attempts_per_training_seed=len(expected_keys),
        unique_pair_count=len(expected_keys),
        utility_difference=matrix(utility_rows),
        feasibility_difference=matrix(feasibility_rows),
        learned_utility=matrix(learned_utility_rows),
        learned_feasibility=matrix(learned_feasibility_rows),
        stock_utility=matrix(stock_utility_rows),
        stock_feasibility=matrix(stock_feasibility_rows),
        informative_discordant_lineages=tuple(sorted(discordant_lineages)),
        informative_discordant_pairs=tuple(sorted(discordant_pairs)),
        discordant_pair_occurrences=discordant_occurrences,
        total_pair_occurrences=total_occurrences,
    )


def _paired_subgroup_results(
    learned_by_row: Sequence[Mapping[tuple[str, str, int], object]],
    stock_by_row: Sequence[Mapping[tuple[str, str, int], object]],
    population: CompletePopulationIdentity,
) -> tuple[dict[str, dict[str, dict[str, object]]], dict[str, object]]:
    registry = preregistered_subgroups(population.evaluation_strata)
    group_count = sum(len(groups) for groups in registry.values())
    comparisons = 2 * group_count
    if comparisons <= 0:
        raise ValueError("paired population has no preregistered subgroups")
    two_sided_alpha = 0.05 / comparisons
    tail_alpha = two_sided_alpha / 2.0
    results: dict[str, dict[str, dict[str, object]]] = {}
    family_index = 0
    for dimension, groups in registry.items():
        dimension_results: dict[str, dict[str, object]] = {}
        for value, identities in groups.items():
            matrices = _paired_subset_matrices(learned_by_row, stock_by_row, identities)
            utility_bounds = crossed_bootstrap_bounds(
                matrices.utility_difference,
                replicates=BOOTSTRAP_REPLICATES,
                seed=int(
                    stable_digest(
                        {
                            "base_seed": BOOTSTRAP_SEED,
                            "dimension": dimension,
                            "value": value,
                            "endpoint": "paired-utility",
                        }
                    )[:16],
                    16,
                ),
                alpha=tail_alpha,
            )
            feasibility_bounds = crossed_bootstrap_bounds(
                matrices.feasibility_difference,
                replicates=BOOTSTRAP_REPLICATES,
                seed=int(
                    stable_digest(
                        {
                            "base_seed": BOOTSTRAP_SEED,
                            "dimension": dimension,
                            "value": value,
                            "endpoint": "paired-feasibility",
                        }
                    )[:16],
                    16,
                ),
                alpha=tail_alpha,
            )
            dimension_results[value] = {
                "family_index": family_index,
                "identity_census_digest": stable_digest(
                    [list(identity) for identity in sorted(identities)]
                ),
                "instance_count": len(identities),
                "independent_lineages": len(matrices.lineages),
                "attempts_per_training_seed": matrices.attempts_per_training_seed,
                "unconditional_utility_difference": float(
                    np.mean(matrices.utility_difference)
                ),
                "unconditional_utility_difference_confidence_interval": {
                    "lower": utility_bounds[0],
                    "upper": utility_bounds[1],
                    "two_sided_alpha": two_sided_alpha,
                },
                "valid_return_rate_difference": float(
                    np.mean(matrices.feasibility_difference)
                ),
                "valid_return_rate_difference_confidence_interval": {
                    "lower": feasibility_bounds[0],
                    "upper": feasibility_bounds[1],
                    "two_sided_alpha": two_sided_alpha,
                },
                "arm_summaries": {
                    "learned": {
                        "unconditional_utility_mean": float(
                            np.mean(matrices.learned_utility)
                        ),
                        "valid_return_rate": float(
                            np.mean(matrices.learned_feasibility)
                        ),
                    },
                    "stock_minorminer": {
                        "unconditional_utility_mean": float(
                            np.mean(matrices.stock_utility)
                        ),
                        "valid_return_rate": float(
                            np.mean(matrices.stock_feasibility)
                        ),
                    },
                },
                "informative_discordant_lineages": len(
                    matrices.informative_discordant_lineages
                ),
                "informative_discordant_pairs": len(
                    matrices.informative_discordant_pairs
                ),
                "weighting": AGGREGATION,
                "confirmatory_claim": False,
            }
            family_index += 1
        results[dimension] = dimension_results
    inference = {
        "scope": "descriptive-not-powered",
        "registered_dimensions": list(registry),
        "independent_unit": "immutable-base-lineage",
        "weighting": AGGREGATION,
        "familywise_alpha": 0.05,
        "multiplicity_method": "bonferroni",
        "group_count": group_count,
        "endpoints_per_group": 2,
        "comparison_count": comparisons,
        "per_interval_two_sided_alpha": two_sided_alpha,
        "per_tail_alpha": tail_alpha,
        "bootstrap_replicates_per_interval": BOOTSTRAP_REPLICATES,
        "powered_subgroup_claims": False,
    }
    return results, inference


def _confirmatory_target_result(
    target: PowerTarget,
    design: ConfirmatoryEvaluationDesign,
    matrices: _PairedMatrices,
) -> dict[str, object]:
    lower, upper = crossed_bootstrap_bounds(
        matrices.feasibility_difference,
        replicates=BOOTSTRAP_REPLICATES,
        seed=int(
            stable_digest(
                {
                    "base_seed": BOOTSTRAP_SEED,
                    "target_id": target.target_id,
                    "endpoint": target.endpoint,
                }
            )[:16],
            16,
        ),
        alpha=target.alpha,
    )
    lineage_count = len(matrices.lineages)
    informative_lineages = len(matrices.informative_discordant_lineages)
    informative_pairs = len(matrices.informative_discordant_pairs)
    failure_reasons: list[str] = []
    if lineage_count < max(2, target.minimum_base_lineages):
        failure_reasons.append("insufficient-independent-lineages")
    if informative_lineages < design.minimum_informative_discordant_lineages:
        failure_reasons.append("insufficient-informative-discordant-lineages")
    if informative_pairs < design.minimum_informative_discordant_lineages:
        failure_reasons.append("insufficient-informative-discordant-pairs")
    if lower <= -design.noninferiority_margin:
        failure_reasons.append("noninferiority-bound-not-above-margin")
    return {
        **target.as_dict(),
        "target_record_digest": target.record_digest,
        "independent_lineages": lineage_count,
        "minimum_independent_lineages_guard": max(2, target.minimum_base_lineages),
        "informative_discordant_lineages": informative_lineages,
        "minimum_informative_discordant_lineages": (
            design.minimum_informative_discordant_lineages
        ),
        "informative_discordant_pairs": informative_pairs,
        "informative_discordant_pair_census_digest": stable_digest(
            [list(key) for key in matrices.informative_discordant_pairs]
        ),
        "observed_informative_lineage_rate": informative_lineages / lineage_count,
        "observed_pair_discordance_rate": (
            matrices.discordant_pair_occurrences / matrices.total_pair_occurrences
        ),
        "valid_return_rate_difference": float(np.mean(matrices.feasibility_difference)),
        "one_sided_interval": {
            "lower": lower,
            "upper_diagnostic": upper,
            "alpha": target.alpha,
        },
        "noninferiority_boundary": -design.noninferiority_margin,
        "sample_size_guard_pass": lineage_count >= max(2, target.minimum_base_lineages),
        "discordance_guard_pass": (
            informative_lineages >= design.minimum_informative_discordant_lineages
            and informative_pairs >= design.minimum_informative_discordant_lineages
        ),
        "bound_guard_pass": lower > -design.noninferiority_margin,
        "failure_reasons": failure_reasons,
        "passed": not failure_reasons,
    }


def _confirmatory_results(
    learned_by_row: Sequence[Mapping[tuple[str, str, int], object]],
    stock_by_row: Sequence[Mapping[tuple[str, str, int], object]],
    population: CompletePopulationIdentity,
) -> dict[str, object]:
    design = population.confirmatory_design
    strata = population.evaluation_strata
    rows = []
    for target in design.power_targets:
        identities = frozenset(
            stratum.identity for stratum in strata if target.design_filter.matches(stratum)
        )
        if not identities:
            raise ValueError(
                f"confirmatory target {target.target_id!r} has no sealed population members"
            )
        rows.append(
            _confirmatory_target_result(
                target,
                design,
                _paired_subset_matrices(learned_by_row, stock_by_row, identities),
            )
        )
    allocated_alpha = sum(target.alpha for target in design.power_targets)
    return {
        "design": design.as_dict(),
        "design_digest": design.record_digest,
        "endpoint_family": "valid-return-noninferiority",
        "noninferiority_margin": design.noninferiority_margin,
        "independent_unit": design.independent_unit,
        "multiplicity_method": design.multiplicity_method,
        "familywise_alpha": design.familywise_alpha,
        "allocated_alpha": allocated_alpha,
        "unallocated_alpha": design.familywise_alpha - allocated_alpha,
        "target_count": len(rows),
        "targets": rows,
        "all_targets_pass": all(bool(row["passed"]) for row in rows),
        "failure_is_noninferiority": True,
        "thin_or_uninformative_cells_fail_closed": True,
    }


def _primary_precision_target_result(
    target: PrecisionTarget,
    matrices: _PairedMatrices,
    *,
    superiority_alpha: float = 0.05,
) -> dict[str, object]:
    tail_alpha = (1.0 - target.confidence_level) / 2.0
    lower, upper = crossed_bootstrap_bounds(
        matrices.utility_difference,
        replicates=BOOTSTRAP_REPLICATES,
        seed=int(
            stable_digest(
                {
                    "base_seed": BOOTSTRAP_SEED,
                    "target_id": target.target_id,
                    "endpoint": target.endpoint,
                }
            )[:16],
            16,
        ),
        alpha=tail_alpha,
    )
    superiority_lower, superiority_upper = crossed_bootstrap_bounds(
        matrices.utility_difference,
        replicates=BOOTSTRAP_REPLICATES,
        seed=int(
            stable_digest(
                {
                    "base_seed": BOOTSTRAP_SEED,
                    "target_id": target.target_id,
                    "endpoint": target.endpoint,
                    "inference": "fixed-sequence-one-sided-superiority",
                }
            )[:16],
            16,
        ),
        alpha=superiority_alpha,
    )
    estimate = float(np.mean(matrices.utility_difference))
    observed_half_width = max(estimate - lower, upper - estimate)
    lineage_count = len(matrices.lineages)
    minimum_lineages = max(2, target.minimum_base_lineages)
    failure_reasons: list[str] = []
    if lineage_count < minimum_lineages:
        failure_reasons.append("insufficient-independent-lineages")
    if observed_half_width > target.half_width:
        failure_reasons.append("registered-half-width-not-achieved")
    return {
        **target.as_dict(),
        "target_record_digest": target.record_digest,
        "independent_lineages": lineage_count,
        "minimum_independent_lineages_guard": minimum_lineages,
        "unconditional_utility_difference": estimate,
        "confidence_interval": {
            "lower": lower,
            "upper": upper,
            "confidence_level": target.confidence_level,
            "tail_alpha": tail_alpha,
        },
        "superiority_one_sided_interval": {
            "lower": superiority_lower,
            "upper_diagnostic": superiority_upper,
            "alpha": superiority_alpha,
        },
        "observed_max_half_width": observed_half_width,
        "sample_size_guard_pass": lineage_count >= minimum_lineages,
        "achieved_half_width_guard_pass": observed_half_width <= target.half_width,
        "learned_superiority_ci_lower_above_zero": superiority_lower > 0.0,
        "failure_reasons": failure_reasons,
        "passed": not failure_reasons,
    }


def _primary_precision_results(
    learned_by_row: Sequence[Mapping[tuple[str, str, int], object]],
    stock_by_row: Sequence[Mapping[tuple[str, str, int], object]],
    population: CompletePopulationIdentity,
) -> dict[str, object]:
    design = population.confirmatory_design
    rows = []
    for target in design.precision_targets:
        identities = frozenset(
            stratum.identity
            for stratum in population.evaluation_strata
            if target.design_filter.matches(stratum)
        )
        if not identities:
            raise ValueError(
                f"primary precision target {target.target_id!r} has no population members"
            )
        rows.append(
            _primary_precision_target_result(
                target,
                _paired_subset_matrices(learned_by_row, stock_by_row, identities),
                superiority_alpha=design.familywise_alpha,
            )
        )
    return {
        "design_digest": design.record_digest,
        "endpoint_family": "paired-primary-utility-difference",
        "independent_unit": design.independent_unit,
        "multiplicity_method": "single-prespecified-primary-no-adjustment",
        "target_count": len(rows),
        "targets": rows,
        "all_targets_pass": all(bool(row["passed"]) for row in rows),
        "ungated_learned_superiority_criterion_pass": all(
            bool(row["passed"])
            and bool(row["learned_superiority_ci_lower_above_zero"])
            for row in rows
        ),
        "positive_result_guaranteed": False,
    }


def _fixed_sequence_gatekeeping(
    *,
    confirmatory: Mapping[str, object],
    primary_precision: Mapping[str, object],
    familywise_alpha: float,
) -> dict[str, object]:
    """Apply the preregistered two-claim fixed-sequence confirmatory family.

    Both tests may use the full familywise alpha because claim two is tested only after
    claim one rejects its null.  The utility decision additionally retains the registered
    sample-size and precision guards encoded by ``primary_precision``.
    """

    if familywise_alpha != 0.05:
        raise ValueError("fixed-sequence familywise alpha must be the registered 0.05")
    noninferiority_pass = confirmatory.get("all_targets_pass")
    ungated_superiority = primary_precision.get(
        "ungated_learned_superiority_criterion_pass"
    )
    if type(noninferiority_pass) is not bool or type(ungated_superiority) is not bool:
        raise ValueError("fixed-sequence inputs do not contain Boolean claim decisions")
    superiority_tested = bool(noninferiority_pass)
    superiority_pass = superiority_tested and bool(ungated_superiority)
    claims = [
        {
            "order": 1,
            "claim_id": "global-valid-return-noninferiority",
            "endpoint": "valid-return-noninferiority",
            "alpha": familywise_alpha,
            "tested": True,
            "criterion_pass": bool(noninferiority_pass),
            "passed": bool(noninferiority_pass),
            "gate": "always-open",
        },
        {
            "order": 2,
            "claim_id": "primary-learned-utility-superiority",
            "endpoint": "learned-minus-stock-unconditional-if-q3-s0",
            "alpha": familywise_alpha,
            "tested": superiority_tested,
            "criterion_pass": bool(ungated_superiority),
            "passed": superiority_pass,
            "gate": "claim-1-must-pass",
        },
    ]
    return {
        "method": "fixed-sequence-gatekeeping-v1",
        "familywise_alpha": familywise_alpha,
        "strong_familywise_error_control": True,
        "ordered_claim_count": len(claims),
        "claims": claims,
        "primary_learned_superiority": superiority_pass,
        "positive_result_guaranteed": False,
    }


def aggregate_learned_vs_stock(
    learned_runs: Sequence[object],
    stock_runs: Sequence[AuthenticatedExternalCompleteRun],
) -> dict[str, object]:
    """Authenticate and aggregate paired learned-minus-stock whole-system outcomes."""

    from isingfold.rl.complete_system_aggregate import (
        AuthenticatedCompleteSystemSeedRun,
        aggregate_complete_system_seeds,
    )

    materialized_stock = tuple(stock_runs)
    if len(materialized_stock) != 3 or any(
        not isinstance(run, AuthenticatedExternalCompleteRun)
        for run in materialized_stock
    ):
        raise TypeError("paired inference requires exactly three authenticated stock runs")
    if len(learned_runs) != 3 or any(
        not isinstance(run, AuthenticatedCompleteSystemSeedRun) for run in learned_runs
    ):
        raise ValueError("paired inference requires exactly three authenticated learned runs")
    typed_learned = tuple(learned_runs)
    learned_aggregate = aggregate_complete_system_seeds(typed_learned)
    ordered = tuple(sorted(typed_learned, key=lambda run: int(run.report["training_seed_index"])))
    ordered_stock = tuple(
        sorted(materialized_stock, key=lambda run: int(run.report["training_seed_index"]))
    )
    if tuple(int(run.report["training_seed"]) for run in ordered) != REGISTERED_TRAINING_SEEDS:
        raise ValueError("paired inference requires the three registered training seeds")
    if (
        tuple(int(run.report["training_seed_index"]) for run in ordered_stock) != (0, 1, 2)
        or tuple(int(run.report["training_seed"]) for run in ordered_stock)
        != REGISTERED_TRAINING_SEEDS
    ):
        raise ValueError("stock inference requires one run per registered training seed")
    learned_reference = ordered[0].receipts[0]
    stock_reference = ordered_stock[0].receipts[0]
    protocol = ordered[0].report["evaluation_protocol"]
    if (
        protocol.get("feasibility_noninferiority_margin")
        != learned_reference.population.confirmatory_design.noninferiority_margin
        or protocol.get("partition")
        != learned_reference.population.confirmatory_design.learning_partition
    ):
        raise ValueError("paired evaluation protocol differs from confirmatory design")
    for learned_run, stock_run in zip(ordered, ordered_stock, strict=True):
        learned_receipt = learned_run.receipts[0]
        stock_receipt = stock_run.receipts[0]
        learned_method = learned_run.report.get("method")
        selection_binding = learned_run.report.get("selection_binding")
        runtime_fields = (
            "runtime_platform",
            "inference_device_type",
            "inference_device_name",
            "inference_threads",
            "deterministic",
        )
        if (
            stock_receipt.population != learned_receipt.population
            or stock_receipt.selector != learned_receipt.selector
            or stock_receipt.context_digest != learned_receipt.context_digest
            or stock_receipt.work_cap != learned_receipt.work_cap
            or stock_receipt.learned_config_digest != learned_receipt.config_digest
            or stock_receipt.config_digest != stock_reference.config_digest
            or stock_receipt.backend != stock_reference.backend
            or stock_run.report["quality_authority"]
            != learned_run.report["quality_authority"]
            or stock_run.report["target_access"]
            != learned_run.report["target_access"]
            or stock_run.report["ground_partition_receipt"]
            != learned_run.report["ground_partition_receipt"]
            or stock_receipt.quality_authority_digest
            != content_digest(learned_run.report["quality_authority"])
            or stock_run.report.get("evaluation_protocol") != protocol
            or not isinstance(learned_method, Mapping)
            or stock_run.report.get("learned_config") != learned_method.get("config")
            or stock_run.report.get("learned_config_file_sha256")
            != learned_run.report.get("complete_system_config_sha256")
            or not isinstance(selection_binding, Mapping)
            or stock_run.report.get("grid_manifest_sha256")
            != selection_binding.get("grid_manifest_sha256")
            or stock_run.report.get("external_pairing_implementation_sha256")
            != ordered_stock[0].report.get("external_pairing_implementation_sha256")
            or stock_run.report.get("runtime_implementation_digest")
            != ordered_stock[0].report.get("runtime_implementation_digest")
            or stock_run.report.get("runtime_implementation_registry")
            != learned_run.report.get("runtime_implementation_registry")
            or any(
                stock_run.report.get(name) != learned_run.report.get(name)
                for name in runtime_fields
            )
        ):
            raise ValueError(
                "learned and stock row do not share authority and exact compute contracts"
            )
    base_seed = learned_reference.population.evaluation_seed
    stock_by_row = tuple(
        {receipt.pair_key: receipt for receipt in run.receipts} for run in ordered_stock
    )
    common_keys = tuple(sorted(stock_by_row[0]))
    expected_census = {
        (lineage, instance, repetition)
        for lineage, instance in learned_reference.population.expected_instances
        for repetition in range(learned_reference.population.expected_repetitions)
    }
    if any(set(row) != expected_census for row in stock_by_row):
        raise ValueError("one stock arm does not cover the sealed learned census")

    utility_rows: list[dict[str, list[float]]] = []
    feasibility_rows: list[dict[str, list[float]]] = []
    stock_utility_rows: list[dict[str, list[float]]] = []
    stock_feasibility_rows: list[dict[str, list[float]]] = []
    learned_by_row: list[dict[tuple[str, str, int], object]] = []
    for run, stock_by_pair in zip(ordered, stock_by_row, strict=True):
        learned_by_pair = {receipt.pair_key: receipt for receipt in run.receipts}
        if set(learned_by_pair) != expected_census:
            raise ValueError("one learned seed does not cover the stock pair census")
        learned_by_row.append(learned_by_pair)
        utility: dict[str, list[float]] = {}
        feasibility: dict[str, list[float]] = {}
        stock_utility: dict[str, list[float]] = {}
        stock_feasibility: dict[str, list[float]] = {}
        for key in common_keys:
            learned = learned_by_pair[key]
            stock = stock_by_pair[key]
            expected_system_seed = _expected_seed(base_seed, "policy", key)
            expected_evaluator_seed = _expected_seed(base_seed, "final-evaluator", key)
            if learned.system_seed != expected_system_seed or stock.system_seed != expected_system_seed:
                raise ValueError("paired arms use different or unregistered system seed schedules")
            for outcome, recorded_seed in (
                (learned.outcome, learned.evaluator_seed),
                (stock.outcome, stock.evaluator_seed),
            ):
                if outcome.returned_valid:
                    if recorded_seed != expected_evaluator_seed or outcome.evaluator_seed != expected_evaluator_seed:
                        raise ValueError("paired valid outcome uses the wrong evaluator seed")
                elif recorded_seed is not None or outcome.evaluator_seed is not None or outcome.utility != 0.0:
                    raise ValueError("paired invalid outcome must have null evaluator seed and zero utility")
            lineage = key[0]
            stock_utility.setdefault(lineage, []).append(float(stock.outcome.utility))
            stock_feasibility.setdefault(lineage, []).append(
                float(stock.outcome.returned_valid)
            )
            utility.setdefault(lineage, []).append(
                float(learned.outcome.utility) - float(stock.outcome.utility)
            )
            feasibility.setdefault(lineage, []).append(
                float(learned.outcome.returned_valid) - float(stock.outcome.returned_valid)
            )
        utility_rows.append(utility)
        feasibility_rows.append(feasibility)
        stock_utility_rows.append(stock_utility)
        stock_feasibility_rows.append(stock_feasibility)

    lineages = tuple(sorted(utility_rows[0]))
    if any(tuple(sorted(row)) != lineages for row in utility_rows + feasibility_rows):
        raise ValueError("paired arms do not share immutable lineage clusters")
    utility_matrix = np.asarray(
        [[float(np.mean(row[lineage])) for lineage in lineages] for row in utility_rows],
        dtype=float,
    )
    feasibility_matrix = np.asarray(
        [[float(np.mean(row[lineage])) for lineage in lineages] for row in feasibility_rows],
        dtype=float,
    )
    stock_utility_matrix = np.asarray(
        [
            [float(np.mean(row[lineage])) for lineage in lineages]
            for row in stock_utility_rows
        ],
        dtype=float,
    )
    stock_feasibility_matrix = np.asarray(
        [
            [float(np.mean(row[lineage])) for lineage in lineages]
            for row in stock_feasibility_rows
        ],
        dtype=float,
    )
    learned_utility_matrix = utility_matrix + stock_utility_matrix
    learned_feasibility_matrix = feasibility_matrix + stock_feasibility_matrix
    utility_low, utility_high = crossed_bootstrap_bounds(
        utility_matrix,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        alpha=UTILITY_ALPHA,
    )
    feasibility_low, feasibility_high = crossed_bootstrap_bounds(
        feasibility_matrix,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        alpha=FEASIBILITY_ALPHA,
    )
    subgroup_results, subgroup_inference = _paired_subgroup_results(
        learned_by_row, stock_by_row, learned_reference.population
    )
    confirmatory = _confirmatory_results(
        learned_by_row, stock_by_row, learned_reference.population
    )
    primary_precision = _primary_precision_results(
        learned_by_row, stock_by_row, learned_reference.population
    )
    fixed_sequence = _fixed_sequence_gatekeeping(
        confirmatory=confirmatory,
        primary_precision=primary_precision,
        familywise_alpha=learned_reference.population.confirmatory_design.familywise_alpha,
    )
    payload: dict[str, object] = {
        "schema": PAIRED_COMPLETE_AGGREGATE_SCHEMA,
        "schema_version": PAIRED_COMPLETE_AGGREGATE_VERSION,
        "partition": "test",
        "sealed_test_opened": True,
        "comparison": "learned-minus-stock-minorminer",
        "aggregation": AGGREGATION,
        "training_seeds": list(REGISTERED_TRAINING_SEEDS),
        "independent_lineages": len(lineages),
        "attempts_per_training_seed": len(common_keys),
        "pair_census_digest": content_digest([list(key) for key in common_keys]),
        "matched_contract": {
            "population": learned_reference.population.as_dict(),
            "population_digest": learned_reference.population.digest,
            "context_digest": learned_reference.context_digest,
            "work_cap": learned_reference.work_cap.as_dict(),
            "selector": learned_reference.selector.as_dict(),
            "quality_authority": ordered_stock[0].report["quality_authority"],
            "target_access": ordered_stock[0].report["target_access"],
            "ground_partition_receipt": ordered_stock[0].report[
                "ground_partition_receipt"
            ],
            "learned_config_digest": learned_reference.config_digest,
            "stock_config_digest": stock_reference.config_digest,
            "external_pairing_implementation_sha256": ordered_stock[0].report[
                "external_pairing_implementation_sha256"
            ],
            "evaluation_protocol": protocol,
            "online_wallclock_seconds": stock_reference.wallclock_cap_seconds,
            "system_seed_schedule": "common-domain-separated-by-pair",
            "evaluator_seed_schedule": "common-domain-separated-by-pair-null-on-invalid",
        },
        "unconditional_utility_difference": float(np.mean(utility_matrix)),
        "unconditional_utility_difference_confidence_interval": {
            "lower": utility_low,
            "upper": utility_high,
            "two_sided_alpha": 2.0 * UTILITY_ALPHA,
        },
        "valid_return_rate_difference": float(np.mean(feasibility_matrix)),
        "valid_return_rate_difference_one_sided_interval": {
            "lower": feasibility_low,
            "upper_diagnostic": feasibility_high,
            "alpha": FEASIBILITY_ALPHA,
        },
        "feasibility_noninferiority": confirmatory["all_targets_pass"],
        "feasibility_margin": learned_reference.population.confirmatory_design.noninferiority_margin,
        "overall_feasibility_interval_is_diagnostic": True,
        "confirmatory_feasibility_noninferiority": confirmatory,
        "primary_paired_utility_precision": primary_precision,
        "confirmatory_fixed_sequence": fixed_sequence,
        "primary_learned_superiority": fixed_sequence["primary_learned_superiority"],
        "subgroup_results": subgroup_results,
        "subgroup_inference": subgroup_inference,
        "arm_summaries": {
            "learned": {
                "unconditional_utility_mean": float(np.mean(learned_utility_matrix)),
                "valid_return_rate": float(np.mean(learned_feasibility_matrix)),
            },
            "stock_minorminer": {
                "unconditional_utility_mean": float(np.mean(stock_utility_matrix)),
                "valid_return_rate": float(np.mean(stock_feasibility_matrix)),
            },
        },
        "per_training_seed": [
            {
                "training_seed": seed,
                "learned_unconditional_utility_mean": float(
                    np.mean(learned_utility_matrix[index])
                ),
                "stock_unconditional_utility_mean": float(
                    np.mean(stock_utility_matrix[index])
                ),
                "utility_difference": float(np.mean(utility_matrix[index])),
                "learned_valid_return_rate": float(
                    np.mean(learned_feasibility_matrix[index])
                ),
                "stock_valid_return_rate": float(
                    np.mean(stock_feasibility_matrix[index])
                ),
                "learned_online_seconds_total": ordered[index].report["metrics"][
                    "online_seconds_total"
                ],
                "stock_online_seconds_total": ordered_stock[index].report["summary"][
                    "online_seconds_total"
                ],
                "online_seconds_total_difference": (
                    ordered[index].report["metrics"]["online_seconds_total"]
                    - ordered_stock[index].report["summary"]["online_seconds_total"]
                ),
                "valid_return_rate_difference": float(
                    np.mean(feasibility_matrix[index])
                ),
            }
            for index, seed in enumerate(REGISTERED_TRAINING_SEEDS)
        ],
        "bootstrap": {
            "sampling": "crossed-training-seed-and-immutable-base-lineage-with-replacement",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "rng": "numpy.random.Generator(PCG64)",
            "numpy_version": np.__version__,
        },
        "lineage_differences": [
            {
                "lineage": lineage,
                "unconditional_utility_difference": float(np.mean(utility_matrix[:, index])),
                "valid_return_rate_difference": float(np.mean(feasibility_matrix[:, index])),
            }
            for index, lineage in enumerate(lineages)
        ],
        "learned_aggregate_record_digest": learned_aggregate["record_digest"],
        "learned_report_file_sha256": [run.report_file_sha256 for run in ordered],
        "external_report_record_digests": [
            run.report["record_digest"] for run in ordered_stock
        ],
        "external_report_file_sha256": [
            run.report_file_sha256 for run in ordered_stock
        ],
        "runtime_comparability": {
            "total_online_envelope_symmetric": True,
            "evaluator_excluded_from_online_envelope": True,
            "native_internal_work_coordinates": "unavailable-and-retained-as-null",
            "same_machine_rowwise": True,
            "total_online_wallclock_reported_rowwise": True,
            "stock_adapter_process_model": (
                "one-persistent-python-worker-per-task-repetition"
            ),
            "equal_internal_work_claimed": False,
            "latency_difference_claimed": True,
        },
    }
    canonical = json.loads(canonical_json_bytes(payload))
    canonical["record_digest"] = content_digest(canonical)
    return canonical


__all__ = [
    "AuthenticatedExternalCompleteRun",
    "ExternalCompleteSystemConfig",
    "ExternalCompleteSystemEvidenceRecord",
    "ExternalCompleteSystemReceipt",
    "ExternalRestartEvidence",
    "aggregate_learned_vs_stock",
    "context_snapshot",
    "external_complete_summary",
    "load_authenticated_external_complete_run",
    "read_external_complete_evidence",
    "read_external_complete_receipts",
    "read_external_complete_outcomes",
    "run_external_complete_system",
    "runtime_identity",
    "write_external_complete_evidence",
    "write_external_complete_receipts",
    "write_external_complete_outcomes",
]
