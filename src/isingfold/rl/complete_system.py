"""Complete-system evaluation for the learned Profile-I pipeline.

This module is deliberately separate from :func:`isingfold.rl.evaluate.run_controller`.
That historical API evaluates a controller *after* a Profile-I incumbent exists and may
replay ``EmbeddingTask.initial_embedding``.  Publication-facing whole-system evaluation
must instead invoke a named, versioned initializer on every task/repetition, retain every
attempt, and count failure to initialize as unconditional utility zero.

External native backends may not expose operation counters.  ``PartialWorkVector`` represents
those coordinates as ``None`` and never converts them to convenient zeroes.  The in-repository
LAC v3 initializer is stricter: it supplies all nine exact semantic-operation coordinates.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Hashable, Iterable, Mapping, Protocol, Sequence

import networkx as nx
import numpy as np

from isingfold.rl.contracts import (
    WORK_FIELDS,
    Context,
    DecisionState,
    InitFailureRecord,
    Mode,
    Opcode,
    TerminalRecord,
    WorkVector,
    stable_digest,
)
from isingfold.rl.env import EmbeddingEnv, EmbeddingTask, StrengthSelector
from isingfold.rl.evaluation_strata import (
    ConfirmatoryEvaluationDesign,
    EvaluationStratum,
    SIZE_BIN_BOUNDARY_CONVENTION,
    SIZE_BIN_PROTOCOL,
)
from isingfold.rl.evaluate import (
    TASK_TERMINAL,
    Controller,
    EpisodeOutcome,
    EvaluationProtocolError,
    program_digest,
)
from isingfold.rl.evaluator import sample_program
from isingfold.rl.external import (
    BackendIdentity,
    ExternalAttemptReceipt,
    ExternalBaselineConfig,
    SearchStatus,
)
from isingfold.rl.program import Program
from isingfold.rl.validate import occupancy, p_embed, p_return

COMPLETE_SYSTEM_SCHEMA = "isingfold.learned-complete-system-attempt"
COMPLETE_SYSTEM_VERSION = 4
COMPLETE_SYSTEM_OUTCOME_SCHEMA = "isingfold.learned-complete-system-outcome"
COMPLETE_SYSTEM_OUTCOME_VERSION = 1
TERMINAL_EVIDENCE_SCHEMA = "isingfold.complete-system-terminal-evidence"
TERMINAL_EVIDENCE_VERSION = 1
COMPLETE_SYSTEM_EVIDENCE_SCHEMA = "isingfold.complete-system-evidence-record"
COMPLETE_SYSTEM_EVIDENCE_VERSION = 1
COMPLETE_SYSTEM_CONFIG_SCHEMA = "isingfold.learned-complete-system-config"
COMPLETE_SYSTEM_CONFIG_VERSION = 2
LEGACY_COMPLETE_POLICY_RESTART_MODE = "disabled-no-native-replay-v1"
COMPLETE_POLICY_RESTART_MODE = "persistent-upfront-lac-cache-k2-v2"
COMPLETE_BOOTSTRAP_BINDING_SCHEMA = "isingfold.complete-system-bootstrap-binding"
COMPLETE_BOOTSTRAP_BINDING_VERSION = 1
COMPLETE_POLICY_CONTEXT_SUFFIX = "+complete-no-native-replay-v1"
LAC_INITIALIZER_METHOD_ID = "lac-minorminer-hybrid-chimera-clique-v1-initializer-v4"
LAC_WORK_COUNTER_SCHEMA = "lac-minorminer.native-work"
LAC_WORK_COUNTER_VERSION = 3
COMPLETE_POPULATION_TASK_DIGEST_SCHEMA = "isingfold-complete-population-v2"


def _lac_profile_detail_consistent(
    *,
    success: object,
    termination: object,
    search_profile: object,
    profile_detail: object,
    structural_fallback_invoked: object,
) -> bool:
    if search_profile != "hybrid_chimera_clique_v1" or not isinstance(
        profile_detail, str
    ) or not isinstance(structural_fallback_invoked, bool):
        return False
    if profile_detail == "v0_success":
        return (
            success is True
            and termination == "success"
            and not structural_fallback_invoked
        )
    if profile_detail == "v0_failed_chimera_clique_success":
        return (
            success is True
            and termination == "success"
            and structural_fallback_invoked
        )
    if profile_detail == "v0_success_validator_budget_exhausted":
        return (
            success is False
            and termination == "work_budget_exhausted"
            and not structural_fallback_invoked
        )
    if profile_detail == "v0_failed_chimera_clique_success_validator_budget_exhausted":
        return (
            success is False
            and termination == "work_budget_exhausted"
            and structural_fallback_invoked
        )
    if profile_detail == "v0_work_budget_exhausted":
        return (
            success is False
            and termination == "work_budget_exhausted"
            and not structural_fallback_invoked
        )
    if profile_detail == "v0_failed_chimera_clique_work_budget_exhausted":
        return (
            success is False
            and termination == "work_budget_exhausted"
            and structural_fallback_invoked
        )
    if profile_detail == "v0_timeout":
        return (
            success is False
            and termination == "timeout"
            and not structural_fallback_invoked
        )
    if profile_detail == "v0_failed_chimera_clique_timeout":
        return (
            success is False
            and termination == "timeout"
            and structural_fallback_invoked
        )
    if profile_detail == "empty_target":
        return (
            success is False
            and termination == "empty_target"
            and not structural_fallback_invoked
        )
    if profile_detail in {
        "v0_failure_chimera_clique_inapplicable_source_not_complete",
        "v0_failure_chimera_clique_inapplicable_source_order_limit",
        "v0_failure_chimera_clique_inapplicable_target_not_chimera",
        "v0_failed_chimera_clique_clique_not_found",
        "v0_failed_chimera_clique_search_state_limit",
    }:
        return (
            success is False
            and structural_fallback_invoked
            and termination
            in {
                "transition_limit",
                "no_candidate",
                "tries_exhausted",
                "control_stop",
            }
        )
    return False


class CompleteSystemBackendError(RuntimeError):
    """A backend/protocol fault, which must not be relabelled as utility zero."""


@dataclass(frozen=True)
class PartialWorkVector:
    """Work coordinates whose unavailable native counters remain explicit ``None``.

    Backend results count operations *inside* one native call.  The runner separately adds
    one top-level ``restart_work`` unit per invocation and one validator call for every
    returned embedding it screens.
    """

    decisions: int | None = None
    route_expansions: int | None = None
    materializations: int | None = None
    compiler_calls: int | None = None
    validator_calls: int | None = None
    cut_edge_visits: int | None = None
    restart_work: int | None = None
    evaluator_reads: int | None = None
    feature_work: int | None = None
    known_lower_bound: WorkVector = field(default=WorkVector(), repr=False)

    def __post_init__(self) -> None:
        lower = self.known_lower_bound
        if not isinstance(lower, WorkVector) or any(
            type(getattr(lower, name)) is not int or getattr(lower, name) < 0
            for name in WORK_FIELDS
        ):
            raise ValueError("partial work known lower bound must be a nonnegative WorkVector")
        for name in WORK_FIELDS:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"partial work coordinate {name} must be nonnegative or null")
            stated_lower = getattr(lower, name)
            if value is not None and stated_lower not in (0, value):
                raise ValueError(f"exact work coordinate {name} disagrees with its lower bound")
        resolved_lower = WorkVector(
            **{
                name: (
                    getattr(self, name) if getattr(self, name) is not None else getattr(lower, name)
                )
                for name in WORK_FIELDS
            }
        )
        object.__setattr__(self, "known_lower_bound", resolved_lower)

    @classmethod
    def known(cls, work: WorkVector = WorkVector()) -> PartialWorkVector:
        return cls(**work.as_dict())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> PartialWorkVector:
        if set(payload) != set(WORK_FIELDS):
            raise ValueError("partial work mapping has missing or unknown coordinates")
        return cls(**{name: payload[name] for name in WORK_FIELDS})

    def __add__(self, other: PartialWorkVector | WorkVector) -> PartialWorkVector:
        rhs = other if isinstance(other, PartialWorkVector) else PartialWorkVector.known(other)
        values: dict[str, int | None] = {}
        for name in WORK_FIELDS:
            left_value, right_value = getattr(self, name), getattr(rhs, name)
            values[name] = (
                None if left_value is None or right_value is None else left_value + right_value
            )
        return PartialWorkVector(
            **values,
            known_lower_bound=self.known_lower_bound + rhs.known_lower_bound,
        )

    @property
    def complete(self) -> bool:
        return all(getattr(self, name) is not None for name in WORK_FIELDS)

    def to_work_vector(self) -> WorkVector | None:
        if not self.complete:
            return None
        return WorkVector(**{name: getattr(self, name) for name in WORK_FIELDS})

    def cap_compliance(self, cap: WorkVector) -> Mapping[str, bool | None]:
        return MappingProxyType(
            {
                name: (
                    False
                    if getattr(self.known_lower_bound, name) > getattr(cap, name)
                    else None
                    if getattr(self, name) is None
                    else True
                )
                for name in WORK_FIELDS
            }
        )

    def as_dict(self) -> dict[str, int | None]:
        return {name: getattr(self, name) for name in WORK_FIELDS}


@dataclass(frozen=True)
class CompleteInitializerResult:
    """One named initializer invocation before independent IsingFold validation."""

    status: SearchStatus
    embedding: Mapping[Hashable, Sequence[Hashable]] | None
    elapsed_seconds: float = field(compare=False)
    work: PartialWorkVector = field(compare=True)
    detail: str = ""
    diagnostics: Mapping[str, object] = field(default_factory=dict)
    budget_exhausted_coordinate: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", SearchStatus(self.status))
        if (
            isinstance(self.elapsed_seconds, bool)
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0.0
        ):
            raise ValueError("initializer elapsed time must be finite and nonnegative")
        if self.status is SearchStatus.EMBEDDING and self.embedding is None:
            raise ValueError("EMBEDDING initializer status requires an embedding")
        if self.status is not SearchStatus.EMBEDDING and self.embedding is not None:
            raise ValueError("non-embedding initializer status cannot carry an embedding")
        if not isinstance(self.work, PartialWorkVector):
            raise TypeError("initializer work must be a PartialWorkVector")
        if self.work.evaluator_reads != 0:
            raise ValueError(
                "an outcome-blind initializer must report exactly zero evaluator reads"
            )
        if not isinstance(self.detail, str):
            raise ValueError("initializer detail must be a string")
        if self.budget_exhausted_coordinate is not None:
            if (
                self.status is SearchStatus.EMBEDDING
                or self.budget_exhausted_coordinate not in WORK_FIELDS
            ):
                raise ValueError(
                    "initializer budget exhaustion requires a non-embedding status and known coordinate"
                )
        object.__setattr__(
            self,
            "diagnostics",
            _freeze_json_mapping(self.diagnostics, label="initializer diagnostics"),
        )


class CompleteInitializerBackend(Protocol):
    """Named and versioned initializer boundary used once per configured attempt."""

    identity: BackendIdentity

    def search(
        self,
        logical: nx.Graph,
        host: nx.Graph,
        *,
        seed: int,
        timeout_seconds: float,
        parameters: Mapping[str, int],
    ) -> CompleteInitializerResult: ...


def _sha256_runtime_file(path: Path, *, label: str) -> str:
    try:
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise OSError("not a regular file")
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    except OSError as exc:
        raise CompleteSystemBackendError(
            f"cannot authenticate lac_minorminer {label}: {type(exc).__name__}: {exc}"
        ) from exc


def _lac_runtime_implementation_manifest(module: object, info: Mapping[str, object]) -> dict[str, object]:
    package_file = getattr(module, "__file__", None)
    package_name = getattr(module, "__name__", None)
    if not isinstance(package_file, str) or not package_file or package_name != "lac_minorminer":
        raise CompleteSystemBackendError("cannot resolve the lac_minorminer Python package source")
    package_dir = Path(package_file).resolve().parent
    sources = sorted(package_dir.glob("*.py"), key=lambda item: item.name)
    required_sources = {
        "__init__.py",
        "_chimera_clique.py",
        "_search_loop.py",
        "api.py",
        "diagnostics.py",
        "orchestrator.py",
        "session.py",
    }
    if not required_sources <= {item.name for item in sources}:
        raise CompleteSystemBackendError(
            "lac_minorminer runtime source manifest is incomplete"
        )
    try:
        native_module = importlib.import_module("lac_minorminer._core")
    except Exception as exc:
        raise CompleteSystemBackendError(
            f"cannot resolve the lac_minorminer native extension: {type(exc).__name__}: {exc}"
        ) from exc
    native_file = getattr(native_module, "__file__", None)
    if not isinstance(native_file, str) or not native_file:
        raise CompleteSystemBackendError("lac_minorminer native extension has no file identity")
    body: dict[str, object] = {
        "schema": "lac-minorminer.runtime-implementation",
        "schema_version": 1,
        "native_extension_sha256": _sha256_runtime_file(
            Path(native_file), label="native extension"
        ),
        "python_source_sha256": {
            item.name: _sha256_runtime_file(item, label=f"Python source {item.name}")
            for item in sources
        },
        "backend_info": _jsonable(info),
    }
    return {**body, "manifest_sha256": stable_digest(body)}


class LACMinorminerInitializerBackend:
    """Strict production adapter for the in-repository C++ ``lac_minorminer`` backend."""

    _PARAMETER_KEYS = frozenset({"tries", "max_transitions", "max_candidates"})

    def __init__(self) -> None:
        try:
            module = importlib.import_module("lac_minorminer")
            info = module.backend_info()
            version = module.__version__
        except Exception as exc:  # pragma: no cover - installation faults vary by platform
            raise CompleteSystemBackendError(
                f"cannot load the required lac_minorminer backend: {type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(info, Mapping) or not isinstance(version, str) or not version:
            raise CompleteSystemBackendError("lac_minorminer returned malformed backend identity")
        if info.get("package_version") != version:
            raise CompleteSystemBackendError(
                "lac_minorminer Python and native package versions disagree"
            )
        native_backend = info.get("backend")
        if not isinstance(native_backend, str) or not native_backend:
            raise CompleteSystemBackendError("lac_minorminer omitted its native backend name")
        work_counter_schema = info.get("work_counter_schema")
        work_counter_version = info.get("work_counter_version")
        if (
            work_counter_schema != getattr(module, "WORK_COUNTER_SCHEMA", None)
            or work_counter_version != getattr(module, "WORK_COUNTER_VERSION", None)
            or work_counter_schema != LAC_WORK_COUNTER_SCHEMA
            or work_counter_version != LAC_WORK_COUNTER_VERSION
        ):
            raise CompleteSystemBackendError(
                "lac_minorminer Python and native work-counter schemas disagree"
            )
        implementation_manifest = _lac_runtime_implementation_manifest(module, info)
        implementation = implementation_manifest["manifest_sha256"]
        assert isinstance(implementation, str)
        self._module = module
        self._native_backend = native_backend
        self._work_counter_schema = work_counter_schema
        self._work_counter_version = work_counter_version
        self._implementation_manifest = _freeze_json_mapping(
            implementation_manifest,
            label="lac_minorminer runtime implementation manifest",
        )
        self.identity = BackendIdentity(
            method_id=LAC_INITIALIZER_METHOD_ID,
            distribution="lac-minorminer",
            version=version,
            entrypoint="lac_minorminer.find_embedding",
            implementation=f"{native_backend}:runtime-sha256:{implementation}",
        )

    def search(
        self,
        logical: nx.Graph,
        host: nx.Graph,
        *,
        seed: int,
        timeout_seconds: float,
        parameters: Mapping[str, int],
        work_cap: WorkVector,
    ) -> CompleteInitializerResult:
        if set(parameters) != self._PARAMETER_KEYS:
            raise CompleteSystemBackendError(
                "lac_minorminer initializer parameters differ: "
                f"missing={sorted(self._PARAMETER_KEYS - set(parameters))}, "
                f"unknown={sorted(set(parameters) - self._PARAMETER_KEYS)}"
            )
        if any(
            type(parameters[name]) is not int or parameters[name] <= 0
            for name in self._PARAMETER_KEYS
        ):
            raise CompleteSystemBackendError(
                "lac_minorminer initializer parameters must be positive integers"
            )
        if parameters["tries"] != 1:
            raise CompleteSystemBackendError(
                "lac_minorminer complete-system calls require tries=1; top-level attempts "
                "are controlled and receipted by CompleteSystemConfig"
            )
        if type(seed) is not int or not 0 <= seed < 2**31:
            raise CompleteSystemBackendError("lac_minorminer seed is outside its domain")
        if (
            isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0.0
        ):
            raise CompleteSystemBackendError("lac_minorminer timeout must be positive and finite")
        if not isinstance(work_cap, WorkVector) or not work_cap.is_nonnegative:
            raise CompleteSystemBackendError(
                "lac_minorminer requires one exact nonnegative remaining-work cap"
            )
        try:
            native_work_cap = self._module.SearchWorkCounters(**work_cap.as_dict())
        except Exception as exc:
            raise CompleteSystemBackendError(
                "cannot construct the sealed lac_minorminer native work cap"
            ) from exc
        try:
            embedding, diagnostics = self._module.find_embedding(
                logical,
                host,
                random_seed=seed,
                timeout=timeout_seconds,
                tries=parameters["tries"],
                max_transitions=parameters["max_transitions"],
                max_candidates=parameters["max_candidates"],
                search_profile="hybrid_chimera_clique_v1",
                work_cap=native_work_cap,
                return_diagnostics=True,
            )
        except Exception as exc:
            raise CompleteSystemBackendError(
                f"lac_minorminer search failed: {type(exc).__name__}: {exc}"
            ) from exc
        required = (
            "success",
            "termination_reason",
            "random_seed",
            "elapsed_seconds",
            "package_version",
            "backend",
            "work",
            "work_counter_schema",
            "work_counter_version",
            "work_cap",
            "work_budget_exhausted_coordinate",
            "search_profile",
            "profile_detail",
            "structural_fallback_invoked",
            "to_dict",
        )
        if any(not hasattr(diagnostics, name) for name in required):
            raise CompleteSystemBackendError("lac_minorminer returned malformed diagnostics")
        if diagnostics.random_seed != seed:
            raise CompleteSystemBackendError(
                "lac_minorminer diagnostics changed the requested seed"
            )
        if (
            diagnostics.search_profile != "hybrid_chimera_clique_v1"
            or not isinstance(diagnostics.profile_detail, str)
            or not diagnostics.profile_detail
        ):
            raise CompleteSystemBackendError(
                "lac_minorminer diagnostics changed the registered hybrid profile"
            )
        if (
            diagnostics.package_version != self.identity.version
            or diagnostics.backend != self._native_backend
        ):
            raise CompleteSystemBackendError(
                "lac_minorminer diagnostics disagree with the sealed backend identity"
            )
        raw_diagnostics = diagnostics.to_dict()
        if not isinstance(raw_diagnostics, Mapping):
            raise CompleteSystemBackendError("lac_minorminer diagnostics are not an object")
        raw_diagnostics = {
            **raw_diagnostics,
            "runtime_implementation_manifest": self._implementation_manifest,
        }
        success = diagnostics.success
        if type(success) is not bool:
            raise CompleteSystemBackendError("lac_minorminer success flag is not Boolean")
        termination = getattr(diagnostics.termination_reason, "value", None)
        if not isinstance(termination, str) or not termination:
            raise CompleteSystemBackendError("lac_minorminer termination reason is malformed")
        if not _lac_profile_detail_consistent(
            success=success,
            termination=termination,
            search_profile=diagnostics.search_profile,
            profile_detail=diagnostics.profile_detail,
            structural_fallback_invoked=diagnostics.structural_fallback_invoked,
        ):
            raise CompleteSystemBackendError(
                "lac_minorminer profile detail disagrees with its result"
            )
        status = (
            SearchStatus.EMBEDDING
            if success
            else SearchStatus.TIMED_OUT
            if termination == "timeout"
            else SearchStatus.NO_EMBEDDING
        )
        if success and not isinstance(embedding, Mapping):
            raise CompleteSystemBackendError("lac_minorminer success omitted an embedding")
        if (
            diagnostics.work_counter_schema != self._work_counter_schema
            or diagnostics.work_counter_version != self._work_counter_version
        ):
            raise CompleteSystemBackendError(
                "lac_minorminer diagnostics use an unsealed work-counter schema"
            )
        raw_cap = raw_diagnostics.get("work_cap")
        if raw_cap != work_cap.as_dict():
            raise CompleteSystemBackendError(
                "lac_minorminer diagnostics changed the prospective native work cap"
            )
        exhausted_coordinate = diagnostics.work_budget_exhausted_coordinate
        if termination == "work_budget_exhausted":
            if exhausted_coordinate not in WORK_FIELDS or success:
                raise CompleteSystemBackendError(
                    "lac_minorminer returned malformed work-budget termination"
                )
        elif exhausted_coordinate is not None:
            raise CompleteSystemBackendError(
                "lac_minorminer claimed budget exhaustion without budget termination"
            )
        try:
            work_payload = diagnostics.work.as_dict()
        except Exception as exc:
            raise CompleteSystemBackendError(
                "lac_minorminer diagnostics omitted typed work counters"
            ) from exc
        if not isinstance(work_payload, Mapping) or set(work_payload) != set(WORK_FIELDS):
            raise CompleteSystemBackendError(
                "lac_minorminer work counters have missing or unknown coordinates"
            )
        if any(
            isinstance(work_payload[name], bool)
            or not isinstance(work_payload[name], int)
            or work_payload[name] < 0
            for name in WORK_FIELDS
        ):
            raise CompleteSystemBackendError(
                "lac_minorminer work counters must be exact nonnegative integers"
            )
        raw_work = raw_diagnostics.get("work")
        if raw_work != dict(work_payload):
            raise CompleteSystemBackendError(
                "lac_minorminer serialized and typed work counters disagree"
            )
        if work_payload["decisions"] != diagnostics.transitions:
            raise CompleteSystemBackendError(
                "lac_minorminer decision counter disagrees with its transition trace"
            )
        if any(work_payload[name] != 0 for name in ("compiler_calls", "cut_edge_visits", "evaluator_reads")):
            raise CompleteSystemBackendError(
                "lac_minorminer v3 unexpectedly invoked a forbidden compiler, cut, or evaluator"
            )
        if work_payload["restart_work"] != 0:
            raise CompleteSystemBackendError(
                "lac_minorminer tries=1 hybrid invocation unexpectedly performed an internal restart"
            )
        work = PartialWorkVector.known(
            WorkVector(**{name: work_payload[name] for name in WORK_FIELDS})
        )
        return CompleteInitializerResult(
            status=status,
            embedding=embedding if success else None,
            elapsed_seconds=float(diagnostics.elapsed_seconds),
            work=work,
            detail=termination,
            diagnostics=raw_diagnostics,
            budget_exhausted_coordinate=exhausted_coordinate,
        )


@dataclass(frozen=True)
class FrozenComponentIdentity:
    """Cryptographic provenance for a frozen controller or selector artifact."""

    component_id: str
    version: str
    implementation: str
    artifact_sha256: str

    def __post_init__(self) -> None:
        for name in ("component_id", "version", "implementation"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"frozen component {name} must be a nonempty string")
        if not _is_digest(self.artifact_sha256):
            raise ValueError("frozen component artifact_sha256 must be a SHA-256 digest")

    def as_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True)
class CompletePopulationIdentity:
    """Authenticated denominator declared before any initializer is run.

    ``expected_instances`` enumerates the policy population, not only the subset with a
    prepared Profile-I incumbent.  A legacy prepared-v1 initializer table therefore cannot
    satisfy this contract by itself because it omits initialization failures.
    """

    population_id: str
    source_manifest_sha256: str
    task_payload_sha256: str
    expected_instances: tuple[tuple[str, str], ...]
    expected_repetitions: int
    evaluation_strata: tuple[EvaluationStratum, ...]
    confirmatory_design: ConfirmatoryEvaluationDesign
    evaluation_seed: int = 0
    denominator_scope: str = "all-policy-instances-before-initialization"
    includes_initializer_failures: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.population_id, str) or not self.population_id:
            raise ValueError("complete-system population_id must be nonempty")
        if not _is_digest(self.source_manifest_sha256):
            raise ValueError("population source manifest must have a SHA-256 digest")
        if not _is_digest(self.task_payload_sha256):
            raise ValueError("population task payload must have a SHA-256 digest")
        if self.denominator_scope != "all-policy-instances-before-initialization":
            raise ValueError("whole-system denominator must precede initializer filtering")
        if self.includes_initializer_failures is not True:
            raise ValueError("whole-system population must include initializer failures")
        if not self.expected_instances:
            raise ValueError("whole-system population cannot be empty")
        if (
            isinstance(self.expected_repetitions, bool)
            or not isinstance(self.expected_repetitions, int)
            or self.expected_repetitions <= 0
        ):
            raise ValueError("population expected_repetitions must be positive")
        if (
            isinstance(self.evaluation_seed, bool)
            or not isinstance(self.evaluation_seed, int)
            or self.evaluation_seed < 0
        ):
            raise ValueError("population evaluation_seed must be a nonnegative integer")
        for identity in self.expected_instances:
            if (
                not isinstance(identity, tuple)
                or len(identity) != 2
                or any(not isinstance(value, str) or not value for value in identity)
            ):
                raise ValueError("population identities must be (lineage, instance) strings")
        canonical = tuple(sorted(self.expected_instances))
        if canonical != self.expected_instances or len(canonical) != len(set(canonical)):
            raise ValueError("population identities must be unique and canonically sorted")
        if (
            not isinstance(self.evaluation_strata, tuple)
            or any(not isinstance(row, EvaluationStratum) for row in self.evaluation_strata)
        ):
            raise TypeError("population evaluation strata must be a tuple of typed rows")
        stratum_identities = tuple(row.identity for row in self.evaluation_strata)
        if (
            stratum_identities != tuple(sorted(set(stratum_identities)))
            or stratum_identities != self.expected_instances
        ):
            raise ValueError(
                "population evaluation-stratum coverage must exactly equal expected instances"
            )
        if not isinstance(self.confirmatory_design, ConfirmatoryEvaluationDesign):
            raise TypeError("population confirmatory design has the wrong type")
        if any(
            row.learning_partition != self.confirmatory_design.learning_partition
            for row in self.evaluation_strata
        ):
            raise ValueError("population strata differ from the confirmatory design partition")
        for target in (
            *self.confirmatory_design.power_targets,
            *self.confirmatory_design.precision_targets,
        ):
            matched = {
                row.identity for row in self.evaluation_strata if target.design_filter.matches(row)
            }
            if matched != set(self.expected_instances):
                raise ValueError(
                    f"global confirmatory target {target.target_id!r} does not cover the "
                    "full sealed preinitialization population"
                )
            matched_lineages = {lineage for lineage, _ in matched}
            if len(matched_lineages) < target.minimum_base_lineages:
                raise ValueError(
                    f"population does not meet confirmatory target {target.target_id!r} "
                    "minimum independent-lineage coverage"
                )

    def validate_tasks(self, tasks: Sequence[EmbeddingTask]) -> None:
        observed = tuple(sorted((task.lineage or task.name, task.name) for task in tasks))
        if observed != self.expected_instances:
            missing = sorted(set(self.expected_instances) - set(observed))
            extra = sorted(set(observed) - set(self.expected_instances))
            raise EvaluationProtocolError(
                "whole-system task population differs from its sealed denominator: "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )
        if task_population_digest(tasks) != self.task_payload_sha256:
            raise EvaluationProtocolError(
                "whole-system task payload differs from its sealed population digest"
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "population_id": self.population_id,
            "source_manifest_sha256": self.source_manifest_sha256,
            "task_payload_sha256": self.task_payload_sha256,
            "expected_instances": [list(item) for item in self.expected_instances],
            "expected_repetitions": self.expected_repetitions,
            "evaluation_seed": self.evaluation_seed,
            "evaluation_strata": [row.as_dict() for row in self.evaluation_strata],
            "evaluation_strata_digest": self.evaluation_strata_digest,
            "size_bin_protocol": SIZE_BIN_PROTOCOL,
            "size_bin_boundary_convention": SIZE_BIN_BOUNDARY_CONVENTION,
            "confirmatory_design": self.confirmatory_design.as_dict(),
            "confirmatory_design_digest": self.confirmatory_design.record_digest,
            "denominator_scope": self.denominator_scope,
            "includes_initializer_failures": self.includes_initializer_failures,
        }

    @property
    def evaluation_strata_digest(self) -> str:
        return stable_digest([row.as_dict() for row in self.evaluation_strata])

    @property
    def digest(self) -> str:
        return stable_digest(self.as_dict())


@dataclass(frozen=True)
class CompleteSystemConfig:
    """Preregistered whole-system budget and outcome-blind selection contract."""

    online_wallclock_seconds: float
    max_initializer_attempts: int
    audit_reads: int
    selection_rule: str
    online_evaluator_feedback: bool
    initializer_backend: str
    initializer_method_id: str
    expected_initializer_version: str
    policy_restart_mode: str
    initializer_parameters: Mapping[str, int]

    _TOP_LEVEL_KEYS = frozenset(
        {
            "schema",
            "schema_version",
            "online_wallclock_seconds",
            "max_initializer_attempts",
            "audit_reads",
            "selection_rule",
            "online_evaluator_feedback",
            "initializer_backend",
            "initializer_method_id",
            "expected_initializer_version",
            "policy_restart_mode",
            "initializer_parameters",
        }
    )

    def __post_init__(self) -> None:
        if (
            isinstance(self.online_wallclock_seconds, bool)
            or not math.isfinite(self.online_wallclock_seconds)
            or self.online_wallclock_seconds <= 0.0
        ):
            raise ValueError("online wall-clock cap must be positive and finite")
        if (
            isinstance(self.max_initializer_attempts, bool)
            or not isinstance(self.max_initializer_attempts, int)
            or self.max_initializer_attempts <= 0
        ):
            raise ValueError("max_initializer_attempts must be a positive integer")
        if (
            isinstance(self.audit_reads, bool)
            or not isinstance(self.audit_reads, int)
            or self.audit_reads <= 0
        ):
            raise ValueError("audit_reads must be a positive integer")
        if self.selection_rule != "resource-lexicographic":
            raise ValueError("complete-system initializer selection is resource-lexicographic")
        if self.online_evaluator_feedback is not False:
            raise ValueError("complete-system search cannot observe evaluator outcomes")
        for name in (
            "initializer_backend",
            "initializer_method_id",
            "expected_initializer_version",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"complete-system {name} must be a nonempty string")
        if self.policy_restart_mode not in {
            LEGACY_COMPLETE_POLICY_RESTART_MODE,
            COMPLETE_POLICY_RESTART_MODE,
        }:
            raise ValueError(
                "complete-system policy restart mode is not registered"
            )
        parameters = dict(self.initializer_parameters)
        if any(
            not isinstance(name, str)
            or not name
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            for name, value in parameters.items()
        ):
            raise ValueError("initializer parameters must map nonempty names to positive integers")
        object.__setattr__(self, "initializer_parameters", MappingProxyType(parameters))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, object]) -> CompleteSystemConfig:
        if set(payload) != cls._TOP_LEVEL_KEYS:
            raise ValueError(
                "complete-system config schema differs: "
                f"missing={sorted(cls._TOP_LEVEL_KEYS - set(payload))}, "
                f"unknown={sorted(set(payload) - cls._TOP_LEVEL_KEYS)}"
            )
        expected_version = (
            1
            if payload.get("policy_restart_mode")
            == LEGACY_COMPLETE_POLICY_RESTART_MODE
            else COMPLETE_SYSTEM_CONFIG_VERSION
        )
        if (
            payload["schema"] != COMPLETE_SYSTEM_CONFIG_SCHEMA
            or type(payload["schema_version"]) is not int
            or payload["schema_version"] != expected_version
        ):
            raise ValueError("unsupported complete-system config schema")
        parameters = payload["initializer_parameters"]
        if not isinstance(parameters, Mapping):
            raise ValueError("initializer_parameters must be an object")
        return cls(
            online_wallclock_seconds=payload["online_wallclock_seconds"],
            max_initializer_attempts=payload["max_initializer_attempts"],
            audit_reads=payload["audit_reads"],
            selection_rule=payload["selection_rule"],
            online_evaluator_feedback=payload["online_evaluator_feedback"],
            initializer_backend=payload["initializer_backend"],
            initializer_method_id=payload["initializer_method_id"],
            expected_initializer_version=payload["expected_initializer_version"],
            policy_restart_mode=payload["policy_restart_mode"],
            initializer_parameters=dict(parameters),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": COMPLETE_SYSTEM_CONFIG_SCHEMA,
            "schema_version": (
                1
                if self.policy_restart_mode == LEGACY_COMPLETE_POLICY_RESTART_MODE
                else COMPLETE_SYSTEM_CONFIG_VERSION
            ),
            "online_wallclock_seconds": self.online_wallclock_seconds,
            "max_initializer_attempts": self.max_initializer_attempts,
            "audit_reads": self.audit_reads,
            "selection_rule": self.selection_rule,
            "online_evaluator_feedback": self.online_evaluator_feedback,
            "initializer_backend": self.initializer_backend,
            "initializer_method_id": self.initializer_method_id,
            "expected_initializer_version": self.expected_initializer_version,
            "policy_restart_mode": self.policy_restart_mode,
            "initializer_parameters": dict(self.initializer_parameters),
        }

    @property
    def digest(self) -> str:
        return stable_digest(self.as_dict())


_PROGRAM_EVIDENCE_KEYS = frozenset(
    {
        "strength",
        "strength_index",
        "scale",
        "offset",
        "h_phys",
        "j_phys",
        "chain_edges",
        "contact_counts",
    }
)


def _finite_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def _canonical_embedding(
    chains: Mapping[Hashable, Sequence[Hashable]],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    rows = tuple(
        sorted(
            (
                _typed_identity(owner),
                tuple(sorted(_typed_identity(qubit) for qubit in chain)),
            )
            for owner, chain in chains.items()
        )
    )
    return rows


def _canonical_program(program: Program) -> Mapping[str, object]:
    def pair(left: object, right: object) -> tuple[str, str]:
        a, b = _typed_identity(left), _typed_identity(right)
        return (a, b) if a <= b else (b, a)

    payload = {
        "strength": program.strength,
        "strength_index": program.strength_index,
        "scale": program.scale,
        "offset": program.offset,
        "h_phys": sorted(
            (_typed_identity(qubit), value) for qubit, value in program.h_phys.items()
        ),
        "j_phys": sorted(
            (*pair(left, right), value) for (left, right), value in program.j_phys.items()
        ),
        "chain_edges": sorted(
            (
                _typed_identity(owner),
                sorted(pair(left, right) for left, right in edges),
            )
            for owner, edges in program.chain_edges.items()
        ),
        "contact_counts": sorted(
            (*pair(left, right), count) for (left, right), count in program.contact_counts.items()
        ),
    }
    return _freeze_json_mapping(payload, label="terminal program evidence")


def _program_evidence_digest(
    program: Mapping[str, object],
    embedding: tuple[tuple[str, tuple[str, ...]], ...],
) -> str:
    """Reproduce :func:`evaluate.program_digest` from JSON-safe evidence only."""

    return stable_digest(
        {
            **_jsonable(program),
            "decoder_chains": _jsonable(embedding),
        }
    )


def _validate_program_evidence(program: Mapping[str, object], *, expected_index: int) -> None:
    if set(program) != _PROGRAM_EVIDENCE_KEYS:
        raise ValueError("terminal program evidence has missing or unknown fields")
    if type(program["strength_index"]) is not int or program["strength_index"] != expected_index:
        raise ValueError("terminal program evidence indices must be exactly zero through three")
    strength = _finite_number(program["strength"], label="program strength")
    scale = _finite_number(program["scale"], label="program scale")
    _finite_number(program["offset"], label="program offset")
    if strength <= 0.0 or not 0.0 < scale <= 1.0:
        raise ValueError("terminal program strength/scale is outside its domain")

    def rows(name: str, width: int) -> tuple[object, ...]:
        value = program[name]
        if not isinstance(value, tuple) or any(
            not isinstance(item, tuple) or len(item) != width for item in value
        ):
            raise ValueError(f"terminal program {name} rows are malformed")
        return value

    fields = rows("h_phys", 2)
    if any(not isinstance(item[0], str) for item in fields):
        raise ValueError("terminal program field identities must be strings")
    if len({item[0] for item in fields}) != len(fields):
        raise ValueError("terminal program field support contains duplicates")
    if tuple(sorted(fields, key=lambda item: item[0])) != fields:
        raise ValueError("terminal program field support is not canonical")
    for item in fields:
        _finite_number(item[1], label="physical field")

    couplers = rows("j_phys", 3)
    chain_edges = rows("chain_edges", 2)
    contact_counts = rows("contact_counts", 3)
    for name, edge_rows, numeric in (
        ("j_phys", couplers, True),
        ("contact_counts", contact_counts, False),
    ):
        if any(
            not isinstance(item[0], str) or not isinstance(item[1], str) or item[0] > item[1]
            for item in edge_rows
        ):
            raise ValueError(f"terminal program {name} uses malformed edge identities")
        if len({(item[0], item[1]) for item in edge_rows}) != len(edge_rows):
            raise ValueError(f"terminal program {name} contains duplicate edges")
        if tuple(sorted(edge_rows, key=lambda item: (item[0], item[1]))) != edge_rows:
            raise ValueError(f"terminal program {name} is not canonically ordered")
        for item in edge_rows:
            if numeric:
                _finite_number(item[2], label="physical coupling")
            elif type(item[2]) is not int or item[2] < 0:
                raise ValueError("terminal program contact count must be nonnegative")
    if any(
        not isinstance(item[0], str)
        or not isinstance(item[1], tuple)
        or any(
            not isinstance(edge, tuple)
            or len(edge) != 2
            or not all(isinstance(endpoint, str) for endpoint in edge)
            or edge[0] > edge[1]
            for edge in item[1]
        )
        for item in chain_edges
    ):
        raise ValueError("terminal program chain-edge rows are malformed")
    if len({item[0] for item in chain_edges}) != len(chain_edges):
        raise ValueError("terminal program chain-edge owners contain duplicates")
    if tuple(sorted(chain_edges, key=lambda item: item[0])) != chain_edges or any(
        tuple(sorted(item[1])) != item[1] for item in chain_edges
    ):
        raise ValueError("terminal program chain edges are not canonically ordered")


@dataclass(frozen=True)
class TerminalEvidence:
    """JSON-safe terminal state sufficient for independent compilation and receipt audit."""

    embedding: tuple[tuple[str, tuple[str, ...]], ...]
    programs: tuple[Mapping[str, object], ...]
    selected_index: int
    selected_program_digest: str
    evaluator_seed: int
    evaluator_hits: int
    evaluator_reads: int
    broken_chain_fraction: float
    mean_energy_residual: float
    validation_receipt: Mapping[str, object]
    validation_digest: str

    def __post_init__(self) -> None:
        embedding = tuple((owner, tuple(qubits)) for owner, qubits in self.embedding)
        if (
            not embedding
            or any(
                not isinstance(owner, str)
                or not owner
                or not qubits
                or any(not isinstance(qubit, str) or not qubit for qubit in qubits)
                or tuple(sorted(qubits)) != qubits
                or len(set(qubits)) != len(qubits)
                for owner, qubits in embedding
            )
            or tuple(sorted(embedding)) != embedding
            or len({owner for owner, _ in embedding}) != len(embedding)
        ):
            raise ValueError("terminal evidence embedding is not canonical and complete")
        claimed = [qubit for _, qubits in embedding for qubit in qubits]
        if len(claimed) != len(set(claimed)):
            raise ValueError("terminal evidence embedding contains overlapping chains")
        object.__setattr__(self, "embedding", embedding)

        programs = tuple(
            _freeze_json_mapping(program, label="terminal program evidence")
            for program in self.programs
        )
        if len(programs) != 4:
            raise ValueError("terminal evidence must retain exactly four compiled programs")
        for index, program in enumerate(programs):
            _validate_program_evidence(program, expected_index=index)
        object.__setattr__(self, "programs", programs)
        if type(self.selected_index) is not int or not 0 <= self.selected_index < 4:
            raise ValueError(
                "terminal evidence selected index is outside the four-program registry"
            )
        if not _is_digest(self.selected_program_digest):
            raise ValueError("terminal evidence selected program digest is invalid")
        if self.selected_program_digest != _program_evidence_digest(
            programs[self.selected_index], embedding
        ):
            raise ValueError("terminal evidence selected program identity is inconsistent")
        if type(self.evaluator_seed) is not int or not 0 <= self.evaluator_seed < 2**31:
            raise ValueError("terminal evidence evaluator seed is outside its domain")
        if (
            type(self.evaluator_hits) is not int
            or type(self.evaluator_reads) is not int
            or self.evaluator_reads <= 0
            or not 0 <= self.evaluator_hits <= self.evaluator_reads
        ):
            raise ValueError("terminal evidence evaluator count block is invalid")
        broken = _finite_number(
            self.broken_chain_fraction, label="terminal evidence broken-chain fraction"
        )
        residual = _finite_number(
            self.mean_energy_residual, label="terminal evidence energy residual"
        )
        if not 0.0 <= broken <= 1.0 or residual < 0.0:
            raise ValueError("terminal evidence evaluator diagnostics are outside their domain")
        receipt = _freeze_json_mapping(self.validation_receipt, label="terminal validation receipt")
        if receipt.get("valid") is not True:
            raise ValueError("terminal evidence must retain a valid return receipt")
        object.__setattr__(self, "validation_receipt", receipt)
        if not _is_digest(self.validation_digest) or self.validation_digest != stable_digest(
            _jsonable(receipt)
        ):
            raise ValueError("terminal evidence validation receipt digest is inconsistent")

    @property
    def selected_strength(self) -> float:
        return float(self.programs[self.selected_index]["strength"])

    @property
    def evaluator_counts(self) -> Mapping[str, int]:
        return MappingProxyType({"hits": self.evaluator_hits, "reads": self.evaluator_reads})

    def as_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": TERMINAL_EVIDENCE_SCHEMA,
            "schema_version": TERMINAL_EVIDENCE_VERSION,
            "embedding": [
                {"logical": owner, "qubits": list(qubits)} for owner, qubits in self.embedding
            ],
            "programs": _jsonable(self.programs),
            "selected_program": {
                "index": self.selected_index,
                "strength": self.selected_strength,
                "program_digest": self.selected_program_digest,
            },
            "evaluator_count_block": {
                "seed": self.evaluator_seed,
                "hits": self.evaluator_hits,
                "reads": self.evaluator_reads,
                "broken_chain_fraction": self.broken_chain_fraction,
                "mean_energy_residual": self.mean_energy_residual,
            },
            "validation_receipt": _jsonable(self.validation_receipt),
            "validation_digest": self.validation_digest,
        }
        if include_digest:
            payload["evidence_digest"] = stable_digest(payload)
        return payload

    @property
    def digest(self) -> str:
        return stable_digest(self.as_dict(include_digest=False))


def _terminal_evidence_from_payload(payload: Mapping[str, object]) -> TerminalEvidence:
    expected = frozenset(
        {
            "schema",
            "schema_version",
            "embedding",
            "programs",
            "selected_program",
            "evaluator_count_block",
            "validation_receipt",
            "validation_digest",
            "evidence_digest",
        }
    )
    if set(payload) != expected:
        raise EvaluationProtocolError("terminal evidence has missing or unknown fields")
    if (
        payload["schema"] != TERMINAL_EVIDENCE_SCHEMA
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != TERMINAL_EVIDENCE_VERSION
    ):
        raise EvaluationProtocolError("unsupported terminal evidence schema")
    recorded = payload["evidence_digest"]
    without_digest = {key: value for key, value in payload.items() if key != "evidence_digest"}
    if not _is_digest(recorded) or recorded != stable_digest(without_digest):
        raise EvaluationProtocolError("terminal evidence digest mismatch")
    embedding_payload = payload["embedding"]
    if not isinstance(embedding_payload, list) or any(
        not isinstance(row, Mapping)
        or set(row) != {"logical", "qubits"}
        or not isinstance(row["logical"], str)
        or not isinstance(row["qubits"], list)
        or any(not isinstance(qubit, str) for qubit in row["qubits"])
        for row in embedding_payload
    ):
        raise EvaluationProtocolError("terminal evidence embedding rows are malformed")
    programs = payload["programs"]
    if not isinstance(programs, list) or any(not isinstance(item, Mapping) for item in programs):
        raise EvaluationProtocolError("terminal evidence programs are malformed")
    selected = payload["selected_program"]
    counts = payload["evaluator_count_block"]
    validation = payload["validation_receipt"]
    if not isinstance(selected, Mapping) or set(selected) != {
        "index",
        "strength",
        "program_digest",
    }:
        raise EvaluationProtocolError("terminal evidence selected-program block is malformed")
    if not isinstance(counts, Mapping) or set(counts) != {
        "seed",
        "hits",
        "reads",
        "broken_chain_fraction",
        "mean_energy_residual",
    }:
        raise EvaluationProtocolError("terminal evidence evaluator count block is malformed")
    if not isinstance(validation, Mapping):
        raise EvaluationProtocolError("terminal evidence validation receipt must be an object")
    if type(selected["index"]) is not int:
        raise EvaluationProtocolError("terminal evidence selected index must be an integer")
    if not isinstance(selected["program_digest"], str):
        raise EvaluationProtocolError("terminal evidence program digest must be a string")
    if not isinstance(payload["validation_digest"], str):
        raise EvaluationProtocolError("terminal evidence validation digest must be a string")
    for name in ("seed", "hits", "reads"):
        if type(counts[name]) is not int:
            raise EvaluationProtocolError(f"terminal evidence count field {name!r} must be integer")
    try:
        evidence = TerminalEvidence(
            embedding=tuple((row["logical"], tuple(row["qubits"])) for row in embedding_payload),
            programs=tuple(programs),
            selected_index=selected["index"],
            selected_program_digest=selected["program_digest"],
            evaluator_seed=counts["seed"],
            evaluator_hits=counts["hits"],
            evaluator_reads=counts["reads"],
            broken_chain_fraction=counts["broken_chain_fraction"],
            mean_energy_residual=counts["mean_energy_residual"],
            validation_receipt=validation,
            validation_digest=payload["validation_digest"],
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationProtocolError("terminal evidence is semantically invalid") from exc
    try:
        selected_strength = _finite_number(
            selected["strength"], label="terminal evidence selected strength"
        )
    except ValueError as exc:
        raise EvaluationProtocolError("terminal evidence selected strength is malformed") from exc
    if evidence.selected_strength != selected_strength:
        raise EvaluationProtocolError("terminal evidence selected strength is inconsistent")
    if evidence.as_dict() != dict(payload):
        raise EvaluationProtocolError("terminal evidence is not canonical")
    return evidence


def _validate_evidence_outcome(evidence: TerminalEvidence, outcome: EpisodeOutcome) -> None:
    if not outcome.returned_valid:
        raise ValueError("terminal evidence cannot be attached to an invalid outcome")
    expected_chain_sizes = tuple(sorted(len(qubits) for _, qubits in evidence.embedding))
    checks = (
        outcome.program_digest == evidence.selected_program_digest,
        outcome.selected_strength_index == evidence.selected_index,
        outcome.selected_strength == evidence.selected_strength,
        outcome.evaluator_seed == evidence.evaluator_seed,
        outcome.evaluator_hits == evidence.evaluator_hits,
        outcome.evaluator_reads == evidence.evaluator_reads,
        outcome.validation_digest == evidence.validation_digest,
        outcome.chain_sizes == expected_chain_sizes,
        outcome.qubits == sum(expected_chain_sizes),
        outcome.max_chain == max(expected_chain_sizes),
    )
    if not all(checks):
        raise ValueError("terminal evidence disagrees with its projected outcome")
    for outcome_value, evidence_value, label in (
        (
            outcome.broken_chain_fraction,
            evidence.broken_chain_fraction,
            "broken-chain fraction",
        ),
        (
            outcome.mean_energy_residual,
            evidence.mean_energy_residual,
            "mean energy residual",
        ),
    ):
        if outcome_value is None or not math.isclose(
            outcome_value, evidence_value, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(f"terminal evidence {label} disagrees with its outcome")
    if outcome.utility is None or not math.isclose(
        outcome.utility,
        evidence.evaluator_hits / evidence.evaluator_reads,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("terminal evidence evaluator counts disagree with utility")


@dataclass(frozen=True)
class InitializerAttemptReceipt:
    """Raw receipt for one actually invoked initializer attempt."""

    attempt_index: int
    seed: int
    status: str
    reported_seconds: float = field(compare=False)
    observed_seconds: float = field(compare=False)
    backend_work: PartialWorkVector = field(compare=True)
    runner_work: WorkVector = WorkVector()
    embedding_digest: str | None = None
    validation_digest: str | None = None
    qubits: int | None = None
    max_chain: int | None = None
    detail: str = ""
    backend_diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        allowed = {
            "VALID_CANDIDATE",
            "INVALID_CANDIDATE",
            "WORK_CAP_EXHAUSTED",
            "WORK_BUDGET_EXHAUSTED",
            SearchStatus.NO_EMBEDDING.value,
            SearchStatus.TIMED_OUT.value,
        }
        if self.status not in allowed:
            raise ValueError(f"unknown initializer attempt status {self.status!r}")
        if (
            type(self.attempt_index) is not int
            or self.attempt_index < 0
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**31
        ):
            raise ValueError("initializer attempt index/seed is outside its domain")
        if not isinstance(self.backend_work, PartialWorkVector) or not isinstance(
            self.runner_work, WorkVector
        ):
            raise TypeError("initializer attempt work ledgers have the wrong type")
        for name in ("reported_seconds", "observed_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"initializer {name} must be finite and nonnegative")
        if self.runner_work.restart_work != 1:
            raise ValueError("each invoked initializer attempt must charge one restart_work")
        has_candidate = self.status in {"VALID_CANDIDATE", "INVALID_CANDIDATE"}
        expected_runner_work = WorkVector(
            restart_work=1,
            validator_calls=1 if has_candidate else 0,
        )
        if self.runner_work != expected_runner_work:
            raise ValueError("initializer runner work contains an unperformed operation")
        if self.backend_work.evaluator_reads != 0:
            raise ValueError("initializer attempt cannot contain evaluator reads")
        if has_candidate:
            if (
                not _is_digest(self.embedding_digest)
                or not _is_digest(self.validation_digest)
                or type(self.qubits) is not int
                or self.qubits < 0
                or type(self.max_chain) is not int
                or self.max_chain < 0
                or self.runner_work.validator_calls != 1
            ):
                raise ValueError("initializer candidate lacks independent validation provenance")
        elif any(
            value is not None
            for value in (
                self.embedding_digest,
                self.validation_digest,
                self.qubits,
                self.max_chain,
            )
        ):
            raise ValueError("non-candidate attempt cannot claim validation provenance")
        if not isinstance(self.detail, str):
            raise ValueError("initializer attempt detail must be a string")
        object.__setattr__(
            self,
            "backend_diagnostics",
            _freeze_json_mapping(
                self.backend_diagnostics,
                label="initializer-attempt backend diagnostics",
            ),
        )

    @property
    def total_work(self) -> PartialWorkVector:
        return self.backend_work + self.runner_work

    def as_dict(self) -> dict[str, object]:
        return {
            "attempt_index": self.attempt_index,
            "seed": self.seed,
            "status": self.status,
            "reported_seconds": self.reported_seconds,
            "observed_seconds": self.observed_seconds,
            "backend_work": self.backend_work.as_dict(),
            "backend_work_known_lower_bound": self.backend_work.known_lower_bound.as_dict(),
            "runner_work": self.runner_work.as_dict(),
            "total_work": self.total_work.as_dict(),
            "total_work_known_lower_bound": self.total_work.known_lower_bound.as_dict(),
            "embedding_digest": self.embedding_digest,
            "validation_digest": self.validation_digest,
            "qubits": self.qubits,
            "max_chain": self.max_chain,
            "detail": self.detail,
            "backend_diagnostics": _jsonable(self.backend_diagnostics),
        }


_BOOTSTRAP_BINDING_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "clone",
        "initial_attempt_work",
        "restart_cache_fill_work",
        "total_pre_policy_work",
        "precomputed_online_seconds",
        "record_digest",
    }
)
_BOOTSTRAP_CLONE_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "consumer_id",
        "same_support_contract_digest",
        "bank_access_record_digest",
        "bootstrap_record_digest",
        "bootstrap_outcome_record_digest",
        "bootstrap_payload_sha256",
        "row_key",
        "denominator_eligible",
        "fixed_utility",
        "actor_invocation_permitted",
        "cache_invocation_permitted",
        "opened_evaluator_targets",
        "record_digest",
    }
)


def _exact_work_mapping(raw: object, *, label: str) -> WorkVector:
    if not isinstance(raw, Mapping) or set(raw) != set(WORK_FIELDS):
        raise ValueError(f"{label} has missing or unknown work coordinates")
    values: dict[str, int] = {}
    for name in WORK_FIELDS:
        value = raw[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"{label} work coordinates must be nonnegative integers")
        values[name] = value
    return WorkVector(**values)


def _validate_complete_bootstrap_binding(
    raw: Mapping[str, object],
) -> tuple[Mapping[str, object], WorkVector, WorkVector, WorkVector, float, bool]:
    if set(raw) != _BOOTSTRAP_BINDING_KEYS:
        raise ValueError("complete-system bootstrap binding schema differs")
    if (
        raw.get("schema") != COMPLETE_BOOTSTRAP_BINDING_SCHEMA
        or raw.get("schema_version") != COMPLETE_BOOTSTRAP_BINDING_VERSION
    ):
        raise ValueError("unsupported complete-system bootstrap binding")
    recorded = raw.get("record_digest")
    body = {key: value for key, value in raw.items() if key != "record_digest"}
    if not _is_digest(recorded) or stable_digest(_jsonable(body)) != recorded:
        raise ValueError("complete-system bootstrap binding digest mismatch")
    clone = raw.get("clone")
    if not isinstance(clone, Mapping) or set(clone) != _BOOTSTRAP_CLONE_KEYS:
        raise ValueError("complete-system bootstrap clone binding schema differs")
    clone_recorded = clone.get("record_digest")
    clone_body = {key: value for key, value in clone.items() if key != "record_digest"}
    if (
        clone.get("schema") != "isingfold.rl-value-bootstrap-clone"
        or clone.get("schema_version") != 1
        or clone.get("opened_evaluator_targets") is not False
        or clone.get("denominator_eligible") is not True
        or not _is_digest(clone_recorded)
        or stable_digest(_jsonable(clone_body)) != clone_recorded
        or any(
            not _is_digest(clone.get(name))
            for name in (
                "same_support_contract_digest",
                "bank_access_record_digest",
                "bootstrap_record_digest",
                "bootstrap_outcome_record_digest",
                "bootstrap_payload_sha256",
                "row_key",
            )
        )
    ):
        raise ValueError("complete-system bootstrap clone binding is invalid")
    actor_permitted = clone.get("actor_invocation_permitted")
    cache_permitted = clone.get("cache_invocation_permitted")
    fixed_utility = clone.get("fixed_utility")
    if type(actor_permitted) is not bool or type(cache_permitted) is not bool:
        raise ValueError("bootstrap clone invocation permissions must be Boolean")
    if actor_permitted != cache_permitted:
        raise ValueError("actor and cache permissions must agree for the K=2 contract")
    if actor_permitted:
        if fixed_utility is not None:
            raise ValueError("successful bootstrap cannot preassign utility")
    elif fixed_utility != 0.0:
        raise ValueError("failed bootstrap must retain denominator utility zero")

    initial = _exact_work_mapping(raw["initial_attempt_work"], label="initial bootstrap")
    cache = _exact_work_mapping(raw["restart_cache_fill_work"], label="restart-cache fill")
    total = _exact_work_mapping(raw["total_pre_policy_work"], label="total pre-policy")
    if total != initial + cache or (not actor_permitted and cache != WorkVector()):
        raise ValueError("complete-system bootstrap work split is inconsistent")
    elapsed = raw["precomputed_online_seconds"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        raise ValueError("bootstrap wallclock must be numeric")
    elapsed = float(elapsed)
    if not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("bootstrap wallclock must be finite and nonnegative")
    frozen = _freeze_json_mapping(raw, label="complete-system bootstrap binding")
    return frozen, initial, cache, total, elapsed, actor_permitted


@dataclass(frozen=True)
class CompleteSystemReceipt:
    """Authenticated whole-system receipt, including attempts and partial work semantics."""

    instance: str
    lineage: str
    repetition: int
    system_seed: int
    evaluator_seed: int | None
    initializer: BackendIdentity
    controller: FrozenComponentIdentity
    selector: FrozenComponentIdentity
    population: CompletePopulationIdentity
    config_digest: str
    context_digest: str
    policy_context_digest: str
    work_cap: WorkVector
    attempts: tuple[InitializerAttemptReceipt, ...]
    selected_attempt: int | None
    bootstrap_binding: Mapping[str, object] | None
    outcome: EpisodeOutcome
    terminal_evidence_digest: str | None
    terminal_evidence: TerminalEvidence | None = field(repr=False, compare=False)
    initializer_work: PartialWorkVector
    environment_budget_debit: WorkVector
    environment_work: WorkVector
    total_work: PartialWorkVector
    cap_compliance: Mapping[str, bool | None]
    initialization_seconds: float = field(compare=False)
    post_initialization_seconds: float = field(compare=False)
    online_seconds: float = field(compare=False)
    wallclock_cap_seconds: float = field(compare=False)
    wallclock_compliant: bool

    def __post_init__(self) -> None:
        if (
            not isinstance(self.instance, str)
            or not isinstance(self.lineage, str)
            or not self.instance
            or not self.lineage
        ):
            raise ValueError("complete-system receipt needs nonempty instance and lineage")
        if (
            type(self.repetition) is not int
            or self.repetition < 0
            or type(self.system_seed) is not int
            or not 0 <= self.system_seed < 2**31
        ):
            raise ValueError("complete-system repetition/system seed is outside its domain")
        if self.evaluator_seed is not None and (
            type(self.evaluator_seed) is not int or not 0 <= self.evaluator_seed < 2**31
        ):
            raise ValueError("complete-system evaluator seed is outside its domain")
        if not isinstance(self.initializer, BackendIdentity):
            raise TypeError("complete-system initializer identity has the wrong type")
        if not isinstance(self.controller, FrozenComponentIdentity) or not isinstance(
            self.selector, FrozenComponentIdentity
        ):
            raise TypeError("complete-system frozen component identity has the wrong type")
        if not isinstance(self.population, CompletePopulationIdentity):
            raise TypeError("complete-system population identity has the wrong type")
        if (
            not isinstance(self.work_cap, WorkVector)
            or not isinstance(self.environment_budget_debit, WorkVector)
            or not isinstance(self.environment_work, WorkVector)
        ):
            raise TypeError("complete-system exact work ledger has the wrong type")
        if any(
            type(getattr(work, name)) is not int or getattr(work, name) < 0
            for work in (
                self.work_cap,
                self.environment_budget_debit,
                self.environment_work,
            )
            for name in WORK_FIELDS
        ):
            raise ValueError("complete-system exact work ledgers must be nonnegative integers")
        if not isinstance(self.initializer_work, PartialWorkVector) or not isinstance(
            self.total_work, PartialWorkVector
        ):
            raise TypeError("complete-system partial work ledger has the wrong type")
        if not isinstance(self.attempts, tuple) or any(
            not isinstance(item, InitializerAttemptReceipt) for item in self.attempts
        ):
            raise TypeError("complete-system attempts must be a tuple of typed receipts")
        if not isinstance(self.outcome, EpisodeOutcome):
            raise TypeError("complete-system outcome has the wrong type")
        if self.terminal_evidence is not None and not isinstance(
            self.terminal_evidence, TerminalEvidence
        ):
            raise TypeError("complete-system terminal evidence has the wrong type")
        if self.selected_attempt is not None and type(self.selected_attempt) is not int:
            raise TypeError("complete-system selected attempt must be an integer or null")
        if not isinstance(self.cap_compliance, Mapping):
            raise TypeError("complete-system cap compliance must be an object")
        if type(self.wallclock_compliant) is not bool:
            raise TypeError("complete-system wall-clock compliance must be Boolean")
        if (
            not _is_digest(self.config_digest)
            or not _is_digest(self.context_digest)
            or not _is_digest(self.policy_context_digest)
        ):
            raise ValueError("complete-system config/context digest is invalid")
        if (self.lineage, self.instance) not in self.population.expected_instances:
            raise ValueError("complete-system receipt is outside its sealed population")
        indices = tuple(item.attempt_index for item in self.attempts)
        if indices != tuple(range(len(self.attempts))):
            raise ValueError("initializer attempt indices must be contiguous from zero")
        if len({item.seed for item in self.attempts}) != len(self.attempts):
            raise ValueError("initializer attempt seeds must be unique")
        if self.initializer.distribution == "lac-minorminer":
            if self.initializer.method_id != LAC_INITIALIZER_METHOD_ID:
                raise ValueError("unsupported lac_minorminer initializer work semantics")
            for item in self.attempts:
                diagnostics = item.backend_diagnostics
                diagnostic_work = diagnostics.get("work")
                diagnostic_cap = diagnostics.get("work_cap")
                runtime_manifest = diagnostics.get("runtime_implementation_manifest")
                manifest_body = (
                    {
                        key: value
                        for key, value in runtime_manifest.items()
                        if key != "manifest_sha256"
                    }
                    if isinstance(runtime_manifest, Mapping)
                    else {}
                )
                runtime_manifest_valid = (
                    isinstance(runtime_manifest, Mapping)
                    and set(runtime_manifest)
                    == {
                        "schema",
                        "schema_version",
                        "native_extension_sha256",
                        "python_source_sha256",
                        "backend_info",
                        "manifest_sha256",
                    }
                    and runtime_manifest.get("schema")
                    == "lac-minorminer.runtime-implementation"
                    and runtime_manifest.get("schema_version") == 1
                    and _is_digest(runtime_manifest.get("native_extension_sha256"))
                    and isinstance(runtime_manifest.get("python_source_sha256"), Mapping)
                    and bool(runtime_manifest.get("python_source_sha256"))
                    and all(
                        isinstance(name, str)
                        and name.endswith(".py")
                        and _is_digest(digest)
                        for name, digest in runtime_manifest.get(
                            "python_source_sha256", {}
                        ).items()
                    )
                    and runtime_manifest.get("manifest_sha256")
                    == stable_digest(_jsonable(manifest_body))
                )
                if (
                    diagnostics.get("random_seed") != item.seed
                    or diagnostics.get("package_version") != self.initializer.version
                    or not isinstance(diagnostics.get("backend"), str)
                    or not self.initializer.implementation.startswith(
                        f"{diagnostics.get('backend')}:runtime-sha256:"
                    )
                    or not runtime_manifest_valid
                    or self.initializer.implementation
                    != (
                        f"{diagnostics.get('backend')}:runtime-sha256:"
                        f"{runtime_manifest.get('manifest_sha256')}"
                    )
                    or not math.isclose(
                        float(diagnostics.get("elapsed_seconds", -1.0)),
                        item.reported_seconds,
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                    or diagnostics.get("work_counter_schema") != LAC_WORK_COUNTER_SCHEMA
                    or diagnostics.get("work_counter_version") != LAC_WORK_COUNTER_VERSION
                    or diagnostics.get("search_profile")
                    != "hybrid_chimera_clique_v1"
                    or not isinstance(diagnostics.get("profile_detail"), str)
                    or not _lac_profile_detail_consistent(
                        success=diagnostics.get("success"),
                        termination=diagnostics.get("termination_reason"),
                        search_profile=diagnostics.get("search_profile"),
                        profile_detail=diagnostics.get("profile_detail"),
                        structural_fallback_invoked=diagnostics.get(
                            "structural_fallback_invoked"
                        ),
                    )
                    or not isinstance(diagnostic_work, Mapping)
                    or set(diagnostic_work) != set(WORK_FIELDS)
                    or not isinstance(diagnostic_cap, Mapping)
                    or set(diagnostic_cap) != set(WORK_FIELDS)
                    or any(
                        type(diagnostic_work[name]) is not int or diagnostic_work[name] < 0
                        for name in WORK_FIELDS
                    )
                    or any(
                        type(diagnostic_cap[name]) is not int
                        or diagnostic_cap[name] < 0
                        or diagnostic_work[name] > diagnostic_cap[name]
                        for name in WORK_FIELDS
                    )
                    or not item.backend_work.complete
                    or dict(diagnostic_work) != item.backend_work.as_dict()
                    or diagnostic_work["decisions"] != diagnostics.get("transitions")
                    or any(
                        diagnostic_work[name] != 0
                        for name in ("compiler_calls", "cut_edge_visits", "evaluator_reads")
                    )
                    or diagnostic_work["restart_work"] != 0
                    or (
                        item.status == "WORK_BUDGET_EXHAUSTED"
                    )
                    != (
                        diagnostics.get("termination_reason")
                        == "work_budget_exhausted"
                    )
                    or (
                        diagnostics.get("work_budget_exhausted_coordinate")
                        in WORK_FIELDS
                    )
                    != (item.status == "WORK_BUDGET_EXHAUSTED")
                ):
                    raise ValueError(
                        "lac_minorminer attempt diagnostics disagree with its sealed receipt"
                    )
        valid_indices = {
            item.attempt_index for item in self.attempts if item.status == "VALID_CANDIDATE"
        }
        if bool(valid_indices) != (self.selected_attempt is not None):
            raise ValueError(
                "selected initializer attempt must exist exactly when valid candidates exist"
            )
        if self.selected_attempt is not None and self.selected_attempt not in valid_indices:
            raise ValueError("selected initializer attempt is not a valid candidate")
        if self.selected_attempt is not None:
            expected_selected = min(
                (item for item in self.attempts if item.status == "VALID_CANDIDATE"),
                key=lambda item: (
                    item.qubits,
                    item.max_chain,
                    item.embedding_digest,
                    item.attempt_index,
                ),
            ).attempt_index
            if self.selected_attempt != expected_selected:
                raise ValueError("selected initializer attempt violates the registered rule")
        if self.selected_attempt is None and self.outcome.returned_valid:
            raise ValueError("valid whole-system return lacks a successful initializer")
        if self.selected_attempt is None and self.environment_work != WorkVector():
            raise ValueError("initializer failure cannot claim post-initializer environment work")
        if self.outcome.pair_key != (self.lineage, self.instance, self.repetition):
            raise ValueError("complete-system outcome has a different pair key")
        if self.outcome.episode_seed != self.system_seed:
            raise ValueError("complete-system outcome has a different policy seed")
        if self.outcome.returned_valid:
            if self.evaluator_seed is None or self.outcome.evaluator_seed != self.evaluator_seed:
                raise ValueError("valid complete-system outcome has a different evaluator seed")
        elif self.evaluator_seed is not None:
            raise ValueError("invalid complete-system outcome cannot claim an evaluator seed")
        self.outcome.validate_receipt(require_complete=False)
        if self.outcome.returned_valid:
            if not _is_digest(self.terminal_evidence_digest):
                raise ValueError("valid complete-system outcome lacks terminal evidence identity")
            if self.terminal_evidence is not None:
                if self.terminal_evidence.digest != self.terminal_evidence_digest:
                    raise ValueError("terminal evidence differs from its complete-receipt identity")
                _validate_evidence_outcome(self.terminal_evidence, self.outcome)
        elif self.terminal_evidence_digest is not None or self.terminal_evidence is not None:
            raise ValueError("invalid complete-system outcome cannot carry terminal evidence")
        expected_initial_attempt_work = _sum_partial(
            item.total_work for item in self.attempts
        )
        if self.bootstrap_binding is None:
            if expected_initial_attempt_work != self.initializer_work:
                raise ValueError("initializer work does not equal the raw attempt ledger")
        else:
            if not isinstance(self.bootstrap_binding, Mapping):
                raise TypeError("complete-system bootstrap binding must be an object")
            (
                frozen_binding,
                bound_initial_work,
                _bound_cache_work,
                bound_total_work,
                bound_seconds,
                actor_permitted,
            ) = _validate_complete_bootstrap_binding(self.bootstrap_binding)
            exact_attempt_work = expected_initial_attempt_work.to_work_vector()
            exact_initializer_work = self.initializer_work.to_work_vector()
            if (
                exact_attempt_work != bound_initial_work
                or exact_initializer_work != bound_total_work
                or not math.isclose(
                    self.initialization_seconds,
                    bound_seconds,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError(
                    "complete-system receipt differs from its bootstrap work/time binding"
                )
            if not actor_permitted and (
                self.selected_attempt is not None
                or self.environment_budget_debit != WorkVector()
                or self.environment_work != WorkVector()
                or self.outcome.returned_valid
                or self.outcome.utility != 0.0
                or not self.outcome.population_eligible
                or self.outcome.controller_calls != 0
            ):
                raise ValueError(
                    "failed bootstrap invoked policy/cache or left the population denominator"
                )
            object.__setattr__(self, "bootstrap_binding", frozen_binding)
        if self.initializer_work + self.environment_work != self.total_work:
            raise ValueError("total work does not equal initializer plus environment work")
        exact_initializer_work = self.initializer_work.to_work_vector()
        policy_executed = self.environment_budget_debit != WorkVector()
        if policy_executed:
            if exact_initializer_work is None:
                raise ValueError("policy execution cannot debit incomplete initializer work")
            if self.environment_budget_debit != exact_initializer_work:
                raise ValueError(
                    "environment budget debit differs from exact initializer work"
                )
            if self.selected_attempt is None:
                raise ValueError("policy execution lacks a selected initializer")
            if any(
                value is not True
                for value in self.total_work.cap_compliance(self.work_cap).values()
            ):
                raise ValueError("executed policy total work is not proven within every cap")
        elif self.environment_work != WorkVector():
            raise ValueError("environment work cannot exist without an authenticated debit")
        if self.outcome.returned_valid and not policy_executed:
            raise ValueError("valid complete-system outcome lacks policy budget authentication")
        expected_cap = dict(self.total_work.cap_compliance(self.work_cap))
        if dict(self.cap_compliance) != expected_cap:
            raise ValueError("cap compliance disagrees with the partial total-work ledger")
        object.__setattr__(self, "cap_compliance", MappingProxyType(expected_cap))
        complete_work = self.total_work.to_work_vector()
        if self.outcome.work != complete_work:
            raise ValueError("outcome work must be exact when complete and null when unavailable")
        for name in (
            "initialization_seconds",
            "post_initialization_seconds",
            "online_seconds",
            "wallclock_cap_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"complete-system {name} must be finite and nonnegative")
        if not math.isclose(
            self.initialization_seconds + self.post_initialization_seconds,
            self.online_seconds,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("online time must equal initializer plus post-initializer time")
        observed_initializer_seconds = sum(item.observed_seconds for item in self.attempts)
        if self.initialization_seconds + 1e-9 < observed_initializer_seconds:
            raise ValueError("initializer phase time is smaller than its serialized native calls")
        if self.outcome.online_seconds != self.online_seconds:
            raise ValueError("outcome and detailed receipt disagree on online time")
        if self.wallclock_compliant != (
            self.online_seconds
            <= self.wallclock_cap_seconds + max(1e-9, self.wallclock_cap_seconds * 1e-9)
        ):
            raise ValueError("wall-clock compliance flag disagrees with observed online time")
        if self.outcome.returned_valid and not self.wallclock_compliant:
            raise ValueError("valid complete-system return exceeds its wall-clock envelope")

    @property
    def pair_key(self) -> tuple[str, str, int]:
        return (self.lineage, self.instance, self.repetition)

    def as_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": COMPLETE_SYSTEM_SCHEMA,
            "schema_version": COMPLETE_SYSTEM_VERSION,
            "instance": self.instance,
            "lineage": self.lineage,
            "repetition": self.repetition,
            "system_seed": self.system_seed,
            "evaluator_seed": self.evaluator_seed,
            "initializer": self.initializer.as_dict(),
            "controller": self.controller.as_dict(),
            "selector": self.selector.as_dict(),
            "population": self.population.as_dict(),
            "population_digest": self.population.digest,
            "config_digest": self.config_digest,
            "context_digest": self.context_digest,
            "policy_context_digest": self.policy_context_digest,
            "work_cap": self.work_cap.as_dict(),
            "attempts": [item.as_dict() for item in self.attempts],
            "selected_attempt": self.selected_attempt,
            "bootstrap_binding": (
                None if self.bootstrap_binding is None else _jsonable(self.bootstrap_binding)
            ),
            "outcome": self.outcome.as_dict(),
            "terminal_evidence_digest": self.terminal_evidence_digest,
            "initializer_work": self.initializer_work.as_dict(),
            "initializer_work_known_lower_bound": (
                self.initializer_work.known_lower_bound.as_dict()
            ),
            "environment_budget_debit": self.environment_budget_debit.as_dict(),
            "environment_work": self.environment_work.as_dict(),
            "total_work": self.total_work.as_dict(),
            "total_work_known_lower_bound": self.total_work.known_lower_bound.as_dict(),
            "cap_compliance": dict(self.cap_compliance),
            "initialization_seconds": self.initialization_seconds,
            "post_initialization_seconds": self.post_initialization_seconds,
            "online_seconds": self.online_seconds,
            "wallclock_cap_seconds": self.wallclock_cap_seconds,
            "wallclock_compliant": self.wallclock_compliant,
        }
        if include_digest:
            payload["record_digest"] = stable_digest(payload)
        return payload


@dataclass(frozen=True)
class CompleteSystemEvidenceRecord:
    """One sidecar row linked to exactly one complete-system raw receipt."""

    instance: str
    lineage: str
    repetition: int
    population_digest: str
    complete_receipt_digest: str
    terminal_evidence: TerminalEvidence | None

    def __post_init__(self) -> None:
        if any(not isinstance(value, str) or not value for value in (self.instance, self.lineage)):
            raise ValueError("complete-system evidence record needs an instance and lineage")
        if type(self.repetition) is not int or self.repetition < 0:
            raise ValueError("complete-system evidence repetition must be nonnegative")
        if not _is_digest(self.population_digest) or not _is_digest(self.complete_receipt_digest):
            raise ValueError("complete-system evidence links must be SHA-256 digests")
        if self.terminal_evidence is not None and not isinstance(
            self.terminal_evidence, TerminalEvidence
        ):
            raise TypeError("complete-system evidence payload has the wrong type")

    @property
    def pair_key(self) -> tuple[str, str, int]:
        return (self.lineage, self.instance, self.repetition)

    def as_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": COMPLETE_SYSTEM_EVIDENCE_SCHEMA,
            "schema_version": COMPLETE_SYSTEM_EVIDENCE_VERSION,
            "instance": self.instance,
            "lineage": self.lineage,
            "repetition": self.repetition,
            "population_digest": self.population_digest,
            "complete_receipt_digest": self.complete_receipt_digest,
            "terminal_evidence": (
                None if self.terminal_evidence is None else self.terminal_evidence.as_dict()
            ),
        }
        if include_digest:
            payload["record_digest"] = stable_digest(payload)
        return payload


@dataclass(frozen=True)
class CompleteSystemPairing:
    """Strictly paired endpoint inputs plus the limits of work-counter comparability."""

    pair_keys: tuple[tuple[str, str, int], ...]
    learned_outcomes: tuple[EpisodeOutcome, ...]
    stock_outcomes: tuple[EpisodeOutcome, ...]
    identical_system_seeds: bool
    identical_evaluator_seeds: bool
    identical_wallclock_cap: bool
    identical_work_cap: bool
    fresh_audit_reads: int
    work_counter_comparability: Mapping[str, bool]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "work_counter_comparability",
            MappingProxyType(dict(self.work_counter_comparability)),
        )


def _is_digest(value: str | None) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _freeze_json_value(value: object, *, label: str) -> object:
    if isinstance(value, Enum):
        return _freeze_json_value(value.value, label=label)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{label} object keys must be strings")
        return MappingProxyType(
            {key: _freeze_json_value(item, label=label) for key, item in sorted(value.items())}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json_value(item, label=label) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{label} cannot contain nonfinite floats")
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"{label} contains unsupported {type(value).__name__}")


def _freeze_json_mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    frozen = _freeze_json_value(value, label=label)
    assert isinstance(frozen, Mapping)
    return frozen


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"cannot encode {type(value).__name__} in complete-system identity")


def _context_digest(ctx: Context) -> str:
    return stable_digest(_jsonable(ctx))


def _identity_seed(
    base_seed: int,
    domain: str,
    *,
    instance: str,
    lineage: str,
    repetition: int,
) -> int:
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


def _seed(base_seed: int, domain: str, task: EmbeddingTask, repetition: int) -> int:
    return _identity_seed(
        base_seed,
        domain,
        instance=task.name,
        lineage=task.lineage or task.name,
        repetition=repetition,
    )


def _attempt_seed(system_seed: int, method_id: str, attempt_index: int) -> int:
    payload = f"{system_seed}:{method_id}:{attempt_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31)


def _typed_identity(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def task_population_digest(tasks: Sequence[EmbeddingTask]) -> str:
    """Bind public task content while deliberately excluding prepared initial embeddings."""

    def pair(left: object, right: object) -> tuple[str, str]:
        a, b = _typed_identity(left), _typed_identity(right)
        return (a, b) if a <= b else (b, a)

    records: list[dict[str, object]] = []
    for task in sorted(tasks, key=lambda item: (item.lineage or item.name, item.name)):
        records.append(
            {
                "instance": task.name,
                "lineage": task.lineage or task.name,
                "logical_nodes": sorted(_typed_identity(node) for node in task.logical),
                "logical_edges": sorted(pair(left, right) for left, right in task.logical.edges()),
                "host_nodes": sorted(_typed_identity(node) for node in task.host),
                "host_edges": sorted(pair(left, right) for left, right in task.host.edges()),
                "h": sorted(
                    (_typed_identity(node), float(value)) for node, value in task.problem.h.items()
                ),
                "j": sorted(
                    (*pair(left, right), float(value))
                    for (left, right), value in task.problem.j.items()
                ),
            }
        )
    return stable_digest(
        {"domain": COMPLETE_POPULATION_TASK_DIGEST_SCHEMA, "tasks": records}
    )


def verify_terminal_evidence(
    evidence: TerminalEvidence,
    *,
    task: EmbeddingTask,
    context: Context,
    outcome: EpisodeOutcome | None = None,
) -> None:
    """Recompile and verify one terminal evidence record without trusting the runner.

    Node identities are resolved against the supplied authenticated task.  The verifier then
    reconstructs return validity and every physical coefficient at all four strengths.
    """

    if not isinstance(evidence, TerminalEvidence):
        raise TypeError("evidence must be TerminalEvidence")
    if not isinstance(task, EmbeddingTask) or not isinstance(context, Context):
        raise TypeError("terminal evidence verification requires an EmbeddingTask and Context")

    def identity_index(values: Iterable[Hashable], *, label: str) -> dict[str, Hashable]:
        materialized = tuple(values)
        indexed = {_typed_identity(value): value for value in materialized}
        if len(indexed) != len(materialized):
            raise EvaluationProtocolError(f"{label} contains colliding typed identities")
        return indexed

    logical_nodes = tuple(task.logical.nodes())
    host_nodes = tuple(task.host.nodes())
    logical_by_identity = identity_index(logical_nodes, label="logical graph")
    host_by_identity = identity_index(host_nodes, label="host graph")
    if {owner for owner, _ in evidence.embedding} != set(logical_by_identity):
        raise EvaluationProtocolError(
            "terminal evidence embedding does not cover the authenticated logical domain"
        )
    try:
        chains = {
            logical_by_identity[owner]: frozenset(host_by_identity[qubit] for qubit in qubits)
            for owner, qubits in evidence.embedding
        }
    except KeyError as exc:
        raise EvaluationProtocolError(
            "terminal evidence embedding refers outside the authenticated host"
        ) from exc
    if _canonical_embedding(chains) != evidence.embedding:
        raise EvaluationProtocolError("terminal evidence embedding identity is inconsistent")
    validation, programs = p_return(
        chains,
        task.logical,
        task.host,
        task.problem,
        context,
    )
    if not validation.valid or len(programs) != 4:
        raise EvaluationProtocolError(
            "terminal evidence does not revalidate as a four-program return"
        )
    if tuple(_canonical_program(program) for program in programs) != evidence.programs:
        raise EvaluationProtocolError(
            "terminal evidence programs differ from independent recompilation"
        )
    if validation.as_dict() != _jsonable(evidence.validation_receipt):
        raise EvaluationProtocolError(
            "terminal evidence return-validation receipt differs from independent validation"
        )
    if program_digest(programs[evidence.selected_index], chains) != (
        evidence.selected_program_digest
    ):
        raise EvaluationProtocolError(
            "terminal evidence selected program differs from independent identity"
        )
    if outcome is not None:
        try:
            _validate_evidence_outcome(evidence, outcome)
        except ValueError as exc:
            raise EvaluationProtocolError(
                "terminal evidence differs from the projected outcome"
            ) from exc


def _embedding_digest(chains: Mapping[Hashable, Sequence[Hashable]]) -> str:
    rows = sorted(
        (
            _typed_identity(owner),
            sorted(_typed_identity(qubit) for qubit in chain),
        )
        for owner, chain in chains.items()
    )
    return stable_digest({"domain": "isingfold-complete-initializer-v1", "chains": rows})


def _normalise_embedding(
    raw: Mapping[Hashable, Sequence[Hashable]],
) -> dict[Hashable, frozenset[Hashable]]:
    if not isinstance(raw, Mapping):
        raise CompleteSystemBackendError("initializer embedding is not a mapping")
    try:
        return {owner: frozenset(chain) for owner, chain in raw.items()}
    except (TypeError, ValueError) as exc:
        raise CompleteSystemBackendError("initializer returned malformed branch sets") from exc


def _sum_partial(values: Iterable[PartialWorkVector]) -> PartialWorkVector:
    total = PartialWorkVector.known()
    for value in values:
        total = total + value
    return total


def complete_policy_context(ctx: Context) -> Context:
    """Return the registered policy context used after native initialization.

    Training and whole-system evaluation call the same helper.  This prevents a normal
    restart-enabled checkpoint from being presented as compatible with the no-native-replay
    complete-system variant.
    """

    version = ctx.context_version
    if not version.endswith(COMPLETE_POLICY_CONTEXT_SUFFIX):
        version = f"{version}{COMPLETE_POLICY_CONTEXT_SUFFIX}"
    return replace(
        ctx,
        quotas={**dict(ctx.quotas), "restart": 0},
        context_version=version,
    )


def _outcome_work(total: PartialWorkVector) -> WorkVector | None:
    return total.to_work_vector()


def _failure_outcome(
    task: EmbeddingTask,
    *,
    repetition: int,
    system_seed: int,
    reason: str,
    total_work: PartialWorkVector,
    online_seconds: float,
    decisions: int = 0,
    controller_calls: int = 0,
    controller_seconds: float = 0.0,
    telemetry: Mapping[str, int] | None = None,
) -> EpisodeOutcome:
    values = {
        "overlap_events": 0,
        "overlap_decisions": 0,
        "repair_attempts": 0,
        "repair_successes": 0,
        "candidate_states": 0,
        "legal_actions_total": 0,
        "legal_opcode_types_total": 0,
    }
    if telemetry is not None:
        values.update(telemetry)
    outcome = EpisodeOutcome(
        instance=task.name,
        lineage=task.lineage or task.name,
        returned_valid=False,
        utility=0.0,
        qubits=None,
        max_chain=None,
        decisions=decisions,
        selected_strength=None,
        reason=reason,
        repetition=repetition,
        episode_seed=system_seed,
        work=_outcome_work(total_work),
        validation_digest=stable_digest({"kind": reason, "system_seed": system_seed}),
        online_seconds=online_seconds,
        controller_calls=controller_calls,
        controller_seconds=controller_seconds,
        outcome_kind=TASK_TERMINAL,
        population_eligible=True,
        **values,
    )
    outcome.validate_receipt(require_complete=False)
    return outcome


def _receipt(
    task: EmbeddingTask,
    *,
    repetition: int,
    system_seed: int,
    evaluator_seed: int,
    backend_identity: BackendIdentity,
    controller_identity: FrozenComponentIdentity,
    selector_identity: FrozenComponentIdentity,
    population: CompletePopulationIdentity,
    config: CompleteSystemConfig,
    ctx: Context,
    attempts: Sequence[InitializerAttemptReceipt],
    selected_attempt: int | None,
    outcome: EpisodeOutcome,
    initializer_work: PartialWorkVector,
    environment_budget_debit: WorkVector,
    environment_work: WorkVector,
    initialization_seconds: float,
    online_seconds: float,
    terminal_evidence: TerminalEvidence | None = None,
    executed_policy_context: Context | None = None,
    bootstrap_binding: Mapping[str, object] | None = None,
) -> CompleteSystemReceipt:
    total_work = initializer_work + environment_work
    registered_policy_ctx = (
        ctx
        if config.policy_restart_mode == COMPLETE_POLICY_RESTART_MODE
        else complete_policy_context(ctx)
    )
    policy_ctx = registered_policy_ctx if executed_policy_context is None else executed_policy_context
    if policy_ctx != registered_policy_ctx:
        raise EvaluationProtocolError(
            "executed policy context differs from the fixed complete-policy context"
        )
    if policy_ctx.caps != ctx.caps:
        raise EvaluationProtocolError(
            "the fixed complete-policy context changed registered work-cap denominators"
        )
    return CompleteSystemReceipt(
        instance=task.name,
        lineage=task.lineage or task.name,
        repetition=repetition,
        system_seed=system_seed,
        evaluator_seed=evaluator_seed if outcome.returned_valid else None,
        initializer=backend_identity,
        controller=controller_identity,
        selector=selector_identity,
        population=population,
        config_digest=config.digest,
        context_digest=_context_digest(ctx),
        policy_context_digest=_context_digest(policy_ctx),
        work_cap=ctx.caps,
        attempts=tuple(attempts),
        selected_attempt=selected_attempt,
        bootstrap_binding=bootstrap_binding,
        outcome=outcome,
        terminal_evidence_digest=(None if terminal_evidence is None else terminal_evidence.digest),
        terminal_evidence=terminal_evidence,
        initializer_work=initializer_work,
        environment_budget_debit=environment_budget_debit,
        environment_work=environment_work,
        total_work=total_work,
        cap_compliance=total_work.cap_compliance(ctx.caps),
        initialization_seconds=initialization_seconds,
        post_initialization_seconds=max(0.0, online_seconds - initialization_seconds),
        online_seconds=online_seconds,
        wallclock_cap_seconds=config.online_wallclock_seconds,
        wallclock_compliant=(
            online_seconds
            <= config.online_wallclock_seconds + max(1e-9, config.online_wallclock_seconds * 1e-9)
        ),
    )


def _telemetry() -> dict[str, int]:
    return {
        "overlap_events": 0,
        "overlap_decisions": 0,
        "repair_attempts": 0,
        "repair_successes": 0,
        "candidate_states": 0,
        "legal_actions_total": 0,
        "legal_opcode_types_total": 0,
    }


def _live_environment_work(env: EmbeddingEnv) -> WorkVector:
    if env.state is None:
        return WorkVector()
    return env.state.spent


def _repair_target_satisfied(candidate: object, env: EmbeddingEnv) -> bool:
    """Count either registered REPAIR_GROUP success target after transition application."""

    if env.state is None:
        return False
    conflict = getattr(candidate, "target_conflict", None)
    conflict_resolved = conflict is not None and occupancy(env.state.chains).get(conflict, 0) <= 1
    demand = getattr(candidate, "target_demand", None)
    demand_realized = False
    if demand is not None:
        left, right = demand
        demand_realized = any(
            a != b and env.task.host.has_edge(a, b)
            for a in env.state.chains.get(left, frozenset())
            for b in env.state.chains.get(right, frozenset())
        )
    return conflict_resolved or demand_realized


def _lineage_coherent_execution_tasks(
    tasks: Sequence[EmbeddingTask],
    *,
    execution_lineages: Sequence[str] | None,
) -> tuple[EmbeddingTask, ...]:
    """Select a canonical, lineage-complete execution slice from a sealed population.

    The caller must still supply the complete population so its task-payload commitment is
    checked before execution.  A slice names base lineages, never individual descendants, so
    sharding cannot split a statistical unit or silently change the repetition denominator.
    """

    if execution_lineages is None:
        return tuple(tasks)
    if isinstance(execution_lineages, (str, bytes, bytearray)):
        raise EvaluationProtocolError("execution lineages must be a sequence of strings")
    lineages = tuple(execution_lineages)
    if (
        not lineages
        or any(not isinstance(lineage, str) or not lineage for lineage in lineages)
        or lineages != tuple(sorted(set(lineages)))
    ):
        raise EvaluationProtocolError(
            "execution lineages must be nonempty, unique, and canonically sorted"
        )
    available = {task.lineage or task.name for task in tasks}
    unknown = sorted(set(lineages) - available)
    if unknown:
        raise EvaluationProtocolError(
            f"execution slice names unknown base lineages: {unknown[:5]}"
        )
    selected = tuple(task for task in tasks if (task.lineage or task.name) in set(lineages))
    if {task.lineage or task.name for task in selected} != set(lineages):
        raise EvaluationProtocolError("execution slice omits a requested base lineage")
    return selected


def _complete_bootstrap_binding(clone: object) -> Mapping[str, object]:
    """Project one authenticated validation clone into a complete-system receipt link."""

    serializer = getattr(clone, "as_dict", None)
    record = getattr(clone, "bootstrap_record", None)
    outcome = getattr(record, "bootstrap_outcome", None)
    if not callable(serializer) or outcome is None:
        raise TypeError("complete-system validation requires a typed bootstrap clone")
    clone_payload = serializer()
    if not isinstance(clone_payload, Mapping):
        raise TypeError("bootstrap clone serializer returned a non-object payload")
    initial_attempt_work = _sum_partial(
        attempt.total_work for attempt in outcome.initial_snapshot.attempts
    ).to_work_vector()
    if initial_attempt_work is None:
        raise EvaluationProtocolError(
            "validation bootstrap initial attempt work is not exact"
        )
    body = {
        "schema": COMPLETE_BOOTSTRAP_BINDING_SCHEMA,
        "schema_version": COMPLETE_BOOTSTRAP_BINDING_VERSION,
        "clone": _jsonable(clone_payload),
        "initial_attempt_work": initial_attempt_work.as_dict(),
        "restart_cache_fill_work": outcome.restart_cache_fill_debit.as_dict(),
        "total_pre_policy_work": outcome.total_pre_policy_debit.as_dict(),
        "precomputed_online_seconds": outcome.precomputed_online_seconds,
    }
    payload = {**body, "record_digest": stable_digest(body)}
    frozen, *_ = _validate_complete_bootstrap_binding(payload)
    return frozen


def run_complete_system(
    tasks: Sequence[EmbeddingTask],
    ctx: Context,
    initializer: CompleteInitializerBackend,
    config: CompleteSystemConfig,
    *,
    controller: Controller,
    selector: StrengthSelector,
    controller_identity: FrozenComponentIdentity,
    selector_identity: FrozenComponentIdentity,
    population: CompletePopulationIdentity,
    seed: int = 0,
    repetitions: int = 1,
    max_steps: int = 64,
    execution_lineages: Sequence[str] | None = None,
    validation_bootstrap_bank: object | None = None,
    same_support_contract_digest: str | None = None,
    bootstrap_consumer_id: str | None = None,
) -> tuple[list[EpisodeOutcome], list[CompleteSystemReceipt]]:
    """Run initializer plus frozen controller/selector as one unconditional system arm.

    The final evaluator stream is opened exactly once, after a valid fixed program has been
    selected.  ``EmbeddingTask.initial_embedding`` is always cleared before environment
    construction, so this function cannot silently become a prepared-incumbent replay.
    """

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("repetitions must be a positive integer")
    if repetitions != population.expected_repetitions:
        raise EvaluationProtocolError(
            "runtime repetitions differ from the sealed population denominator"
        )
    if seed != population.evaluation_seed:
        raise EvaluationProtocolError(
            "runtime seed differs from the sealed population evaluation seed"
        )
    if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
        raise ValueError("max_steps must be a positive integer")
    if max_steps < ctx.caps.decisions:
        raise ValueError("max_steps cannot truncate the registered decision budget")
    if config.audit_reads != ctx.audit_reads or config.audit_reads > ctx.caps.evaluator_reads:
        raise ValueError("complete-system audit reads must equal the registered context audit")
    if config.max_initializer_attempts > ctx.caps.restart_work:
        raise ValueError(
            "initializer attempt cap exceeds the registered top-level restart-work cap"
        )
    cache_mode = config.policy_restart_mode == COMPLETE_POLICY_RESTART_MODE
    if cache_mode:
        from isingfold.rl.validation_bootstrap_bank import ValidationBootstrapBank

        if not isinstance(validation_bootstrap_bank, ValidationBootstrapBank):
            raise ValueError(
                "persistent K=2 complete-system evaluation requires a validation bootstrap bank"
            )
        if not _is_digest(same_support_contract_digest):
            raise ValueError("complete-system evaluation requires a same-support digest pin")
        if not isinstance(bootstrap_consumer_id, str) or not bootstrap_consumer_id:
            raise ValueError("complete-system evaluation requires a bootstrap consumer ID")
        plan = validation_bootstrap_bank.plan
        if (
            plan.config_digest != config.digest
            or plan.context_digest != _context_digest(ctx)
            or plan.evaluation_seed != seed
            or plan.repetitions != repetitions
            or plan.same_support_contract_digest != same_support_contract_digest
        ):
            raise EvaluationProtocolError(
                "validation bootstrap bank differs from the complete-system protocol"
            )
        policy_ctx = ctx
    else:
        if any(
            value is not None
            for value in (
                validation_bootstrap_bank,
                same_support_contract_digest,
                bootstrap_consumer_id,
            )
        ):
            raise ValueError("legacy no-restart evaluation cannot consume a bootstrap bank")
        policy_ctx = complete_policy_context(ctx)
    if policy_ctx.caps != ctx.caps:
        raise EvaluationProtocolError(
            "complete-policy context must preserve the registered work-cap denominators"
        )
    if not isinstance(initializer.identity, BackendIdentity):
        raise TypeError("initializer must expose a BackendIdentity")
    initializer_identity = initializer.identity
    if config.initializer_backend != initializer_identity.distribution:
        raise EvaluationProtocolError(
            "configured initializer backend differs from the injected backend identity"
        )
    if config.initializer_method_id != initializer_identity.method_id:
        raise EvaluationProtocolError(
            "configured initializer method differs from the injected backend identity"
        )
    if config.expected_initializer_version != initializer_identity.version:
        raise EvaluationProtocolError(
            "configured initializer version differs from the injected backend identity"
        )
    identities = [(task.lineage or task.name, task.name) for task in tasks]
    if len(identities) != len(set(identities)):
        raise EvaluationProtocolError("complete-system population has duplicate identities")
    population.validate_tasks(tasks)
    execution_tasks = _lineage_coherent_execution_tasks(
        tasks,
        execution_lineages=execution_lineages,
    )
    for task in tasks:
        if task.ground_energy is None or not math.isfinite(task.ground_energy):
            raise EvaluationProtocolError(
                f"complete IF-Q3 evaluation needs a finite target for {task.name!r}"
            )

    outcomes: list[EpisodeOutcome] = []
    detailed: list[CompleteSystemReceipt] = []
    environment_initialization_work = WorkVector(
        compiler_calls=4,
        validator_calls=1,
    )
    policy_entry_reserve = environment_initialization_work + policy_ctx.reserve
    for repetition in range(repetitions):
        for task in execution_tasks:
            system_seed = _seed(seed, "policy", task, repetition)
            evaluator_seed = _seed(seed, "final-evaluator", task, repetition)
            bootstrap_binding: Mapping[str, object] | None = None
            bootstrap_outcome = None
            precomputed_online_seconds = 0.0
            initializer_work_override: PartialWorkVector | None = None
            initialization_seconds_override: float | None = None
            attempts: list[InitializerAttemptReceipt]
            candidates: list[
                tuple[int, dict[Hashable, frozenset[Hashable]], int, int, str]
            ]
            initializer_attempt_limit = config.max_initializer_attempts
            if cache_mode:
                assert validation_bootstrap_bank is not None
                assert same_support_contract_digest is not None
                assert bootstrap_consumer_id is not None
                row_key = validation_bootstrap_bank.row_key(
                    task.lineage or task.name,
                    task.name,
                    repetition,
                )
                clone = validation_bootstrap_bank.clone_for_consumer(
                    row_key,
                    consumer_id=bootstrap_consumer_id,
                    expected_same_support_contract_digest=(
                        same_support_contract_digest
                    ),
                )
                record = clone.bootstrap_record
                bootstrap_outcome = record.bootstrap_outcome
                initial_snapshot = bootstrap_outcome.initial_snapshot
                if (
                    record.instance_id != task.name
                    or record.base_lineage != (task.lineage or task.name)
                    or record.repetition != repetition
                    or initial_snapshot.system_seed != system_seed
                    or record.actor_invocation_permitted
                    != initial_snapshot.success
                ):
                    raise EvaluationProtocolError(
                        "validation bootstrap clone differs from its policy episode"
                    )
                bootstrap_binding = _complete_bootstrap_binding(clone)
                precomputed_online_seconds = record.precomputed_online_seconds
                initialization_seconds_override = precomputed_online_seconds
                initializer_work_override = PartialWorkVector.known(
                    record.total_pre_policy_work
                )
                attempts = list(initial_snapshot.attempts)
                candidates = []
                initializer_attempt_limit = 0
                if initial_snapshot.success:
                    if initial_snapshot.selected_attempt is None:
                        raise EvaluationProtocolError(
                            "successful validation bootstrap omitted its selected attempt"
                        )
                    selected = attempts[initial_snapshot.selected_attempt]
                    if (
                        selected.status != "VALID_CANDIDATE"
                        or selected.qubits is None
                        or selected.max_chain is None
                        or selected.embedding_digest is None
                    ):
                        raise EvaluationProtocolError(
                            "validation bootstrap selected attempt is not a valid candidate"
                        )
                    candidates.append(
                        (
                            initial_snapshot.selected_attempt,
                            {},
                            selected.qubits,
                            selected.max_chain,
                            selected.embedding_digest,
                        )
                    )
            else:
                attempts = []
                candidates = []
            started = time.perf_counter()
            deadline = started + max(
                0.0,
                config.online_wallclock_seconds - precomputed_online_seconds,
            )

            def observed_online_seconds() -> float:
                return precomputed_online_seconds + (time.perf_counter() - started)

            pre_call_budget_blocked = False
            for attempt_index in range(initializer_attempt_limit):
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    break
                if initializer.identity != initializer_identity:
                    raise CompleteSystemBackendError(
                        "initializer identity changed after the evaluation contract was sealed"
                    )
                attempt_seed = _attempt_seed(
                    system_seed, initializer_identity.method_id, attempt_index
                )
                native_work_cap: WorkVector | None = None
                if initializer_identity.method_id == LAC_INITIALIZER_METHOD_ID:
                    exact_prior = _sum_partial(
                        item.total_work for item in attempts
                    ).to_work_vector()
                    if exact_prior is None:
                        raise CompleteSystemBackendError(
                            "lac_minorminer prior work became incomplete"
                        )
                    call_reserve = (
                        WorkVector(restart_work=1, validator_calls=1)
                        + policy_entry_reserve
                    )
                    native_work_cap = ctx.caps - exact_prior - call_reserve
                    if not native_work_cap.is_nonnegative:
                        pre_call_budget_blocked = True
                        break
                call_started = time.perf_counter()
                try:
                    if native_work_cap is None:
                        result = initializer.search(
                            task.logical,
                            task.host,
                            seed=attempt_seed,
                            timeout_seconds=remaining,
                            parameters=config.initializer_parameters,
                        )
                    else:
                        result = initializer.search(
                            task.logical,
                            task.host,
                            seed=attempt_seed,
                            timeout_seconds=remaining,
                            parameters=config.initializer_parameters,
                            work_cap=native_work_cap,
                        )
                except CompleteSystemBackendError:
                    raise
                except Exception as exc:
                    raise CompleteSystemBackendError(
                        f"initializer {initializer_identity.method_id!r} raised "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                if initializer.identity != initializer_identity:
                    raise CompleteSystemBackendError(
                        "initializer identity changed during a native search call"
                    )
                observed_seconds = time.perf_counter() - call_started
                if not isinstance(result, CompleteInitializerResult):
                    raise CompleteSystemBackendError(
                        "initializer returned a non-CompleteInitializerResult payload"
                    )
                if result.status is SearchStatus.ERROR:
                    raise CompleteSystemBackendError(result.detail or "initializer backend error")
                timed_out = observed_seconds > remaining + max(1e-9, remaining * 1e-9)
                runner_work = WorkVector(restart_work=1)
                if result.budget_exhausted_coordinate is not None:
                    attempts.append(
                        InitializerAttemptReceipt(
                            attempt_index=attempt_index,
                            seed=attempt_seed,
                            status="WORK_BUDGET_EXHAUSTED",
                            reported_seconds=result.elapsed_seconds,
                            observed_seconds=observed_seconds,
                            backend_work=result.work,
                            runner_work=runner_work,
                            detail=(
                                "native prospective work cap exhausted: "
                                f"{result.budget_exhausted_coordinate}"
                            ),
                            backend_diagnostics=result.diagnostics,
                        )
                    )
                    continue
                if timed_out or result.status is SearchStatus.TIMED_OUT:
                    attempts.append(
                        InitializerAttemptReceipt(
                            attempt_index=attempt_index,
                            seed=attempt_seed,
                            status=SearchStatus.TIMED_OUT.value,
                            reported_seconds=result.elapsed_seconds,
                            observed_seconds=observed_seconds,
                            backend_work=result.work,
                            runner_work=runner_work,
                            detail=(
                                result.detail or "initializer reached the registered call deadline"
                            ),
                            backend_diagnostics=result.diagnostics,
                        )
                    )
                    if any(
                        value is False
                        for value in _sum_partial(item.total_work for item in attempts)
                        .cap_compliance(ctx.caps)
                        .values()
                    ):
                        break
                    continue
                if result.status is SearchStatus.NO_EMBEDDING:
                    attempts.append(
                        InitializerAttemptReceipt(
                            attempt_index=attempt_index,
                            seed=attempt_seed,
                            status=result.status.value,
                            reported_seconds=result.elapsed_seconds,
                            observed_seconds=observed_seconds,
                            backend_work=result.work,
                            runner_work=runner_work,
                            detail=result.detail,
                            backend_diagnostics=result.diagnostics,
                        )
                    )
                    if any(
                        value is False
                        for value in _sum_partial(item.total_work for item in attempts)
                        .cap_compliance(ctx.caps)
                        .values()
                    ):
                        break
                    continue

                assert result.embedding is not None
                prospective_initializer = _sum_partial(
                    [*(item.total_work for item in attempts), result.work + runner_work]
                ) + WorkVector(validator_calls=1)
                if any(
                    value is False
                    for value in prospective_initializer.cap_compliance(ctx.caps).values()
                ):
                    attempts.append(
                        InitializerAttemptReceipt(
                            attempt_index=attempt_index,
                            seed=attempt_seed,
                            status="WORK_CAP_EXHAUSTED",
                            reported_seconds=result.elapsed_seconds,
                            observed_seconds=observed_seconds,
                            backend_work=result.work,
                            runner_work=runner_work,
                            detail=(
                                "initializer returned a candidate without enough registered "
                                "work for independent validation"
                            ),
                            backend_diagnostics=result.diagnostics,
                        )
                    )
                    break
                chains = _normalise_embedding(result.embedding)
                validation = p_embed(chains, task.logical, task.host, ctx.qubit_cap)
                runner_work = runner_work + WorkVector(validator_calls=1)
                embedding_digest = _embedding_digest(chains)
                validation_digest = stable_digest(validation.as_dict())
                status = "VALID_CANDIDATE" if validation.valid else "INVALID_CANDIDATE"
                attempts.append(
                    InitializerAttemptReceipt(
                        attempt_index=attempt_index,
                        seed=attempt_seed,
                        status=status,
                        reported_seconds=result.elapsed_seconds,
                        observed_seconds=observed_seconds,
                        backend_work=result.work,
                        runner_work=runner_work,
                        embedding_digest=embedding_digest,
                        validation_digest=validation_digest,
                        qubits=validation.qubits,
                        max_chain=validation.max_chain,
                        detail="; ".join(validation.reasons),
                        backend_diagnostics=result.diagnostics,
                    )
                )
                if validation.valid:
                    candidates.append(
                        (
                            attempt_index,
                            chains,
                            validation.qubits,
                            validation.max_chain,
                            embedding_digest,
                        )
                    )

            initialization_seconds = (
                time.perf_counter() - started
                if initialization_seconds_override is None
                else initialization_seconds_override
            )
            initializer_work = (
                _sum_partial([item.total_work for item in attempts])
                if initializer_work_override is None
                else initializer_work_override
            )
            if not candidates:
                online_seconds = observed_online_seconds()
                known_cap_exceeded = any(
                    value is False for value in initializer_work.cap_compliance(ctx.caps).values()
                )
                outcome = _failure_outcome(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    reason=(
                            "INITIALIZER_EXCEEDED_WORK_CAP"
                            if known_cap_exceeded
                            else "INITIALIZER_WORK_CAP_BEFORE_CALL"
                            if pre_call_budget_blocked
                            else "INITIALIZER_NATIVE_WORK_BUDGET_EXHAUSTED"
                            if any(
                                item.status == "WORK_BUDGET_EXHAUSTED"
                                for item in attempts
                            )
                            else "INITIALIZER_WORK_CAP_BEFORE_VALIDATION"
                        if any(item.status == "WORK_CAP_EXHAUSTED" for item in attempts)
                        else "INITIALIZER_NO_VALID_EMBEDDING"
                    ),
                    total_work=initializer_work,
                    online_seconds=online_seconds,
                )
                receipt = _receipt(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    evaluator_seed=evaluator_seed,
                    backend_identity=initializer_identity,
                    controller_identity=controller_identity,
                    selector_identity=selector_identity,
                    population=population,
                    config=config,
                    ctx=ctx,
                    attempts=attempts,
                    selected_attempt=None,
                    outcome=outcome,
                    initializer_work=initializer_work,
                    environment_budget_debit=WorkVector(),
                    environment_work=WorkVector(),
                    initialization_seconds=initialization_seconds,
                    online_seconds=online_seconds,
                    bootstrap_binding=bootstrap_binding,
                )
                outcomes.append(outcome)
                detailed.append(receipt)
                continue

            selected_attempt, selected_chains, _, _, _ = min(
                candidates,
                key=lambda item: (item[2], item[3], item[4], item[0]),
            )
            initializer_debit = initializer_work.to_work_vector()
            if initializer_debit is None:
                online_seconds = observed_online_seconds()
                outcome = _failure_outcome(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    reason="INITIALIZER_WORK_INCOMPLETE",
                    total_work=initializer_work,
                    online_seconds=online_seconds,
                )
                receipt = _receipt(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    evaluator_seed=evaluator_seed,
                    backend_identity=initializer_identity,
                    controller_identity=controller_identity,
                    selector_identity=selector_identity,
                    population=population,
                    config=config,
                    ctx=ctx,
                    attempts=attempts,
                    selected_attempt=selected_attempt,
                    outcome=outcome,
                    initializer_work=initializer_work,
                    environment_budget_debit=WorkVector(),
                    environment_work=WorkVector(),
                    initialization_seconds=initialization_seconds,
                    online_seconds=online_seconds,
                    bootstrap_binding=bootstrap_binding,
                )
                outcomes.append(outcome)
                detailed.append(receipt)
                continue

            if not (
                initializer_debit + environment_initialization_work + policy_ctx.reserve
            ).fits_in(policy_ctx.caps):
                online_seconds = observed_online_seconds()
                outcome = _failure_outcome(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    reason=(
                        "INITIALIZER_EXCEEDED_WORK_CAP"
                        if not initializer_debit.fits_in(policy_ctx.caps)
                        else "INITIALIZER_WORK_CAP_BEFORE_POLICY"
                    ),
                    total_work=initializer_work,
                    online_seconds=online_seconds,
                )
                receipt = _receipt(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    evaluator_seed=evaluator_seed,
                    backend_identity=initializer_identity,
                    controller_identity=controller_identity,
                    selector_identity=selector_identity,
                    population=population,
                    config=config,
                    ctx=ctx,
                    attempts=attempts,
                    selected_attempt=selected_attempt,
                    outcome=outcome,
                    initializer_work=initializer_work,
                    environment_budget_debit=WorkVector(),
                    environment_work=WorkVector(),
                    initialization_seconds=initialization_seconds,
                    online_seconds=online_seconds,
                    bootstrap_binding=bootstrap_binding,
                )
                outcomes.append(outcome)
                detailed.append(receipt)
                continue

            if time.perf_counter() >= deadline:
                online_seconds = observed_online_seconds()
                outcome = _failure_outcome(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    reason="COMPLETE_SYSTEM_WALLCLOCK_EXHAUSTED",
                    total_work=initializer_work,
                    online_seconds=online_seconds,
                )
                receipt = _receipt(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    evaluator_seed=evaluator_seed,
                    backend_identity=initializer_identity,
                    controller_identity=controller_identity,
                    selector_identity=selector_identity,
                    population=population,
                    config=config,
                    ctx=ctx,
                    attempts=attempts,
                    selected_attempt=selected_attempt,
                    outcome=outcome,
                    initializer_work=initializer_work,
                    environment_budget_debit=WorkVector(),
                    environment_work=WorkVector(),
                    initialization_seconds=initialization_seconds,
                    online_seconds=online_seconds,
                    bootstrap_binding=bootstrap_binding,
                )
                outcomes.append(outcome)
                detailed.append(receipt)
                continue

            clean_task = replace(task, initial_embedding=None)
            initializer_used = False
            if cache_mode:
                from isingfold.rl.initializer_bank import (
                    environment_from_bootstrap_outcome,
                )

                if bootstrap_outcome is None:
                    raise EvaluationProtocolError(
                        "cache-mode episode omitted its bootstrap outcome"
                    )
                env = environment_from_bootstrap_outcome(
                    bootstrap_outcome,
                    task=clean_task,
                    context=policy_ctx,
                    selector=selector,
                )
                initializer_used = True
            else:

                def selected_initializer_snapshot(
                    logical: nx.Graph,
                    host: nx.Graph,
                    local_seed: int,
                ):
                    nonlocal initializer_used
                    if logical is not clean_task.logical or host is not clean_task.host:
                        raise CompleteSystemBackendError(
                            "environment changed initializer graph identity"
                        )
                    if local_seed != system_seed:
                        raise CompleteSystemBackendError(
                            "environment changed the registered system seed"
                        )
                    if initializer_used:
                        raise CompleteSystemBackendError(
                            "complete-system bootstrap snapshot was requested more than once"
                        )
                    initializer_used = True
                    return dict(selected_chains)

                env = EmbeddingEnv(
                    clean_task,
                    policy_ctx,
                    mode=Mode.IMPROVEMENT,
                    initializer=selected_initializer_snapshot,
                    selector=selector,
                    reward_reads=policy_ctx.n_est_reads,
                    budget_debit=initializer_debit,
                    initializer_precomputed=True,
                    seed=system_seed,
                )
            rng = np.random.default_rng(system_seed)
            telemetry = _telemetry()
            overlap_active = False
            controller_calls = 0
            controller_seconds = 0.0
            result_or_decision = env.reset(system_seed)
            if not initializer_used:
                raise EvaluationProtocolError(
                    "environment did not consume the selected initializer"
                )
            if isinstance(result_or_decision, InitFailureRecord):
                environment_work = result_or_decision.work
                total_work = initializer_work + environment_work
                online_seconds = observed_online_seconds()
                outcome = _failure_outcome(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    reason=f"SELECTED_INITIALIZER_REJECTED:{result_or_decision.reason}",
                    total_work=total_work,
                    online_seconds=online_seconds,
                )
                receipt = _receipt(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    evaluator_seed=evaluator_seed,
                    backend_identity=initializer_identity,
                    controller_identity=controller_identity,
                    selector_identity=selector_identity,
                    population=population,
                    config=config,
                    ctx=ctx,
                    attempts=attempts,
                    selected_attempt=selected_attempt,
                    outcome=outcome,
                    initializer_work=initializer_work,
                    environment_budget_debit=env.budget_debit,
                    environment_work=environment_work,
                    initialization_seconds=initialization_seconds,
                    online_seconds=online_seconds,
                    bootstrap_binding=bootstrap_binding,
                    executed_policy_context=env.ctx,
                )
                outcomes.append(outcome)
                detailed.append(receipt)
                continue

            steps = 0
            timed_out = time.perf_counter() >= deadline
            while isinstance(result_or_decision, DecisionState) and not timed_out:
                if steps >= max_steps:
                    raise EvaluationProtocolError(
                        f"live task {task.name!r} reached max_steps; truncation is disabled"
                    )
                telemetry["candidate_states"] += 1
                legal_candidates = [
                    candidate
                    for candidate, legal in zip(
                        result_or_decision.candidates,
                        result_or_decision.legal_mask,
                        strict=True,
                    )
                    if legal
                ]
                telemetry["legal_actions_total"] += len(legal_candidates)
                telemetry["legal_opcode_types_total"] += len(
                    {candidate.opcode for candidate in legal_candidates}
                )
                current_overlap = bool(
                    env.state is not None
                    and any(value > 1 for value in occupancy(env.state.chains).values())
                )
                if current_overlap:
                    telemetry["overlap_decisions"] += 1
                    if not overlap_active:
                        telemetry["overlap_events"] += 1
                overlap_active = current_overlap
                controller_started = time.perf_counter()
                chosen = int(controller(result_or_decision, rng))
                controller_seconds += time.perf_counter() - controller_started
                controller_calls += 1
                selected = result_or_decision.candidates[chosen]
                step = env.step(result_or_decision, chosen, evaluate_training_reward=False)
                if selected.opcode is Opcode.REPAIR_GROUP:
                    telemetry["repair_attempts"] += 1
                    if _repair_target_satisfied(selected, env):
                        telemetry["repair_successes"] += 1
                result_or_decision = step.next_decision_or_terminal
                steps += 1
                timed_out = time.perf_counter() >= deadline

            if timed_out:
                environment_work = _live_environment_work(env)
                total_work = initializer_work + environment_work
                online_seconds = observed_online_seconds()
                outcome = _failure_outcome(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    reason="COMPLETE_SYSTEM_WALLCLOCK_EXHAUSTED",
                    total_work=total_work,
                    online_seconds=online_seconds,
                    decisions=steps,
                    controller_calls=controller_calls,
                    controller_seconds=controller_seconds,
                    telemetry=telemetry,
                )
                receipt = _receipt(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    evaluator_seed=evaluator_seed,
                    backend_identity=initializer_identity,
                    controller_identity=controller_identity,
                    selector_identity=selector_identity,
                    population=population,
                    config=config,
                    ctx=ctx,
                    attempts=attempts,
                    selected_attempt=selected_attempt,
                    outcome=outcome,
                    initializer_work=initializer_work,
                    environment_budget_debit=env.budget_debit,
                    environment_work=environment_work,
                    initialization_seconds=initialization_seconds,
                    online_seconds=online_seconds,
                    bootstrap_binding=bootstrap_binding,
                    executed_policy_context=env.ctx,
                )
                outcomes.append(outcome)
                detailed.append(receipt)
                continue
            if not isinstance(result_or_decision, TerminalRecord):
                raise EvaluationProtocolError("complete-system environment returned unknown state")

            terminal = result_or_decision
            base_environment_work = terminal.cumulative_work
            online_seconds = observed_online_seconds()
            wallclock_limit = config.online_wallclock_seconds + max(
                1e-9, config.online_wallclock_seconds * 1e-9
            )
            if online_seconds > wallclock_limit:
                outcome = _failure_outcome(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    reason="COMPLETE_SYSTEM_WALLCLOCK_EXHAUSTED",
                    total_work=initializer_work + base_environment_work,
                    online_seconds=online_seconds,
                    decisions=steps,
                    controller_calls=controller_calls,
                    controller_seconds=controller_seconds,
                    telemetry=telemetry,
                )
                receipt = _receipt(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    evaluator_seed=evaluator_seed,
                    backend_identity=initializer_identity,
                    controller_identity=controller_identity,
                    selector_identity=selector_identity,
                    population=population,
                    config=config,
                    ctx=ctx,
                    attempts=attempts,
                    selected_attempt=selected_attempt,
                    outcome=outcome,
                    initializer_work=initializer_work,
                    environment_budget_debit=env.budget_debit,
                    environment_work=base_environment_work,
                    initialization_seconds=initialization_seconds,
                    online_seconds=online_seconds,
                    bootstrap_binding=bootstrap_binding,
                    executed_policy_context=env.ctx,
                )
                outcomes.append(outcome)
                detailed.append(receipt)
                continue
            validation_digest = stable_digest(terminal.validation_receipt)
            if not terminal.returned_valid:
                total_work = initializer_work + base_environment_work
                outcome = EpisodeOutcome(
                    instance=task.name,
                    lineage=task.lineage or task.name,
                    returned_valid=False,
                    utility=0.0,
                    qubits=None,
                    max_chain=None,
                    decisions=steps,
                    selected_strength=None,
                    reason=terminal.terminal_reason.value,
                    repetition=repetition,
                    episode_seed=system_seed,
                    work=_outcome_work(total_work),
                    validation_digest=validation_digest,
                    online_seconds=online_seconds,
                    controller_calls=controller_calls,
                    controller_seconds=controller_seconds,
                    **telemetry,
                )
                outcome.validate_receipt(require_complete=False)
                receipt = _receipt(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    evaluator_seed=evaluator_seed,
                    backend_identity=initializer_identity,
                    controller_identity=controller_identity,
                    selector_identity=selector_identity,
                    population=population,
                    config=config,
                    ctx=ctx,
                    attempts=attempts,
                    selected_attempt=selected_attempt,
                    outcome=outcome,
                    initializer_work=initializer_work,
                    environment_budget_debit=env.budget_debit,
                    environment_work=base_environment_work,
                    initialization_seconds=initialization_seconds,
                    online_seconds=online_seconds,
                    bootstrap_binding=bootstrap_binding,
                    executed_policy_context=env.ctx,
                )
                outcomes.append(outcome)
                detailed.append(receipt)
                continue

            if not isinstance(terminal.selected_program, Program) or terminal.embedding is None:
                raise EvaluationProtocolError("valid terminal omitted its program or embedding")
            if (
                not isinstance(terminal.compiled_programs, tuple)
                or len(terminal.compiled_programs) != 4
                or any(not isinstance(program, Program) for program in terminal.compiled_programs)
            ):
                raise EvaluationProtocolError(
                    "valid terminal omitted its charged four-program compiler output"
                )
            if terminal.selected_index != terminal.selected_program.strength_index:
                raise EvaluationProtocolError("terminal selector/program indices disagree")
            if terminal.selected_index is None or not 0 <= terminal.selected_index < 4:
                raise EvaluationProtocolError("terminal selected index is outside the registry")
            if program_digest(terminal.selected_program, terminal.embedding) != program_digest(
                terminal.compiled_programs[terminal.selected_index], terminal.embedding
            ):
                raise EvaluationProtocolError(
                    "terminal selected program differs from its retained compiler registry"
                )
            environment_work = base_environment_work + WorkVector(
                evaluator_reads=config.audit_reads
            )
            prospective_total = initializer_work + environment_work
            if any(value is False for value in prospective_total.cap_compliance(ctx.caps).values()):
                outcome = _failure_outcome(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    reason="COMPLETE_SYSTEM_WORK_CAP_BEFORE_AUDIT",
                    total_work=initializer_work + base_environment_work,
                    online_seconds=online_seconds,
                    decisions=steps,
                    controller_calls=controller_calls,
                    controller_seconds=controller_seconds,
                    telemetry=telemetry,
                )
                receipt = _receipt(
                    task,
                    repetition=repetition,
                    system_seed=system_seed,
                    evaluator_seed=evaluator_seed,
                    backend_identity=initializer_identity,
                    controller_identity=controller_identity,
                    selector_identity=selector_identity,
                    population=population,
                    config=config,
                    ctx=ctx,
                    attempts=attempts,
                    selected_attempt=selected_attempt,
                    outcome=outcome,
                    initializer_work=initializer_work,
                    environment_budget_debit=env.budget_debit,
                    environment_work=base_environment_work,
                    initialization_seconds=initialization_seconds,
                    online_seconds=online_seconds,
                    bootstrap_binding=bootstrap_binding,
                    executed_policy_context=env.ctx,
                )
                outcomes.append(outcome)
                detailed.append(receipt)
                continue

            evaluator_started = time.perf_counter()
            block = sample_program(
                terminal.selected_program,
                terminal.embedding,
                task.problem,
                task.ground_energy,
                num_reads=config.audit_reads,
                seed=evaluator_seed,
                num_sweeps=ctx.num_sweeps,
            )
            evaluator_seconds = time.perf_counter() - evaluator_started
            if block.strength_index != terminal.selected_index or block.reads != config.audit_reads:
                raise EvaluationProtocolError(
                    "final evaluator returned the wrong program/read block"
                )
            chains = terminal.embedding
            outcome = EpisodeOutcome(
                instance=task.name,
                lineage=task.lineage or task.name,
                returned_valid=True,
                utility=block.rate,
                qubits=sum(len(chain) for chain in chains.values()),
                max_chain=max(len(chain) for chain in chains.values()),
                decisions=steps,
                selected_strength=terminal.selected_strength,
                reason=terminal.terminal_reason.value,
                repetition=repetition,
                episode_seed=system_seed,
                evaluator_seed=evaluator_seed,
                program_digest=program_digest(terminal.selected_program, chains),
                selected_strength_index=terminal.selected_index,
                evaluator_hits=block.hits,
                evaluator_reads=block.reads,
                work=_outcome_work(prospective_total),
                validation_digest=validation_digest,
                broken_chain_fraction=block.broken_fraction,
                mean_energy_residual=block.mean_residual,
                online_seconds=online_seconds,
                evaluator_seconds=evaluator_seconds,
                controller_calls=controller_calls,
                controller_seconds=controller_seconds,
                chain_sizes=tuple(sorted(len(chain) for chain in chains.values())),
                **telemetry,
            )
            outcome.validate_receipt(require_complete=False)
            terminal_evidence = TerminalEvidence(
                embedding=_canonical_embedding(chains),
                programs=tuple(
                    _canonical_program(program) for program in terminal.compiled_programs
                ),
                selected_index=terminal.selected_index,
                selected_program_digest=outcome.program_digest,
                evaluator_seed=evaluator_seed,
                evaluator_hits=block.hits,
                evaluator_reads=block.reads,
                broken_chain_fraction=block.broken_fraction,
                mean_energy_residual=block.mean_residual,
                validation_receipt=terminal.validation_receipt,
                validation_digest=validation_digest,
            )
            receipt = _receipt(
                task,
                repetition=repetition,
                system_seed=system_seed,
                evaluator_seed=evaluator_seed,
                backend_identity=initializer_identity,
                controller_identity=controller_identity,
                selector_identity=selector_identity,
                population=population,
                config=config,
                ctx=ctx,
                attempts=attempts,
                selected_attempt=selected_attempt,
                outcome=outcome,
                initializer_work=initializer_work,
                environment_budget_debit=env.budget_debit,
                environment_work=environment_work,
                initialization_seconds=initialization_seconds,
                online_seconds=online_seconds,
                terminal_evidence=terminal_evidence,
                bootstrap_binding=bootstrap_binding,
                executed_policy_context=env.ctx,
            )
            outcomes.append(outcome)
            detailed.append(receipt)
    if initializer.identity != initializer_identity:
        raise CompleteSystemBackendError(
            "initializer identity changed before complete-system evaluation finished"
        )
    population.validate_tasks(tasks)
    return outcomes, detailed


def _stock_outcome(receipt: ExternalAttemptReceipt) -> EpisodeOutcome:
    selected = (
        None if receipt.selected_restart is None else receipt.restarts[receipt.selected_restart]
    )
    utility = (
        0.0
        if receipt.evaluator_hits is None or receipt.evaluator_reads is None
        else receipt.evaluator_hits / receipt.evaluator_reads
    )
    outcome = EpisodeOutcome(
        instance=receipt.instance,
        lineage=receipt.lineage,
        returned_valid=receipt.returned_valid,
        utility=utility,
        qubits=None if selected is None else selected.qubits,
        max_chain=None if selected is None else selected.max_chain,
        decisions=0,
        selected_strength=receipt.selected_strength,
        reason=(
            "EXTERNAL_VALID_RETURN" if receipt.returned_valid else "EXTERNAL_NO_VALID_EMBEDDING"
        ),
        repetition=receipt.repetition,
        episode_seed=receipt.system_seed,
        evaluator_seed=receipt.evaluator_seed if receipt.returned_valid else None,
        program_digest=receipt.program_digest,
        selected_strength_index=receipt.selected_strength_index,
        evaluator_hits=receipt.evaluator_hits,
        evaluator_reads=receipt.evaluator_reads,
        # Native minorminer counters are not all observable; the scalar WorkVector schema
        # cannot encode that fact, so the paired endpoint projection deliberately omits it.
        work=None,
        validation_digest=receipt.validation_digest,
        online_seconds=receipt.online_seconds,
        evaluator_seconds=receipt.evaluator_seconds,
    )
    outcome.validate_receipt(require_complete=False)
    return outcome


def pair_with_stock_minorminer(
    learned: Sequence[CompleteSystemReceipt],
    stock: Sequence[ExternalAttemptReceipt],
    *,
    learned_config: CompleteSystemConfig,
    stock_config: ExternalBaselineConfig,
    context: Context,
    population: CompletePopulationIdentity,
) -> CompleteSystemPairing:
    """Fail closed unless learned and stock arms share the registered comparison envelope."""

    if not math.isclose(
        learned_config.online_wallclock_seconds,
        stock_config.embedding_wallclock_seconds,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("learned and stock systems use different wall-clock caps")
    if learned_config.max_initializer_attempts != stock_config.max_restarts:
        raise ValueError("learned and stock systems use different top-level attempt caps")
    if learned_config.audit_reads != stock_config.audit_reads or (
        learned_config.audit_reads != context.audit_reads
    ):
        raise ValueError("learned and stock systems use different fresh audit-read blocks")
    if learned_config.selection_rule != stock_config.selection_rule:
        raise ValueError("learned and stock systems use different outcome-blind selection rules")
    if learned_config.online_evaluator_feedback or stock_config.online_evaluator_feedback:
        raise ValueError("online evaluator feedback invalidates complete-system pairing")
    del learned, stock, context, population
    raise EvaluationProtocolError(
        "stock ExternalAttemptReceipt v1 cannot support an honest whole-system pairing: "
        "it does not authenticate the selector, sealed pre-initialization population, "
        "context/work-cap identity, or a wall-clock envelope covering post-initializer work; "
        "generate learned receipts only until the external receipt schema is upgraded"
    )


def complete_system_method_metadata(
    config: CompleteSystemConfig,
    initializer: BackendIdentity,
    controller: FrozenComponentIdentity,
    selector: FrozenComponentIdentity,
    population: CompletePopulationIdentity,
) -> dict[str, object]:
    """Paper-facing boundary between post-initialization and whole-system estimands."""

    return {
        "comparison_scope": "learned-complete-system-profile-i",
        "post_initialization_api": "isingfold.rl.evaluate.run_controller",
        "complete_system_api": "isingfold.rl.complete_system.run_complete_system",
        "prepared_initial_embedding_replay": False,
        "initializer": initializer.as_dict(),
        "controller": controller.as_dict(),
        "selector": selector.as_dict(),
        "population": population.as_dict(),
        "population_digest": population.digest,
        "config": config.as_dict(),
        "config_digest": config.digest,
        "initializer_failure_utility": 0.0,
        "initializer_failure_population_eligible": True,
        "selection_rule": config.selection_rule,
        "online_evaluator_feedback": False,
        "policy_restart_semantics": config.policy_restart_mode,
        "final_evaluator": f"one-fresh-independent-{config.audit_reads}-read-block",
        "terminal_evidence": {
            "schema": COMPLETE_SYSTEM_EVIDENCE_SCHEMA,
            "valid_return_contents": (
                "returned-embedding, four-compiled-programs, selected-program, "
                "validation-receipt, evaluator-count-block"
            ),
            "failure_row": "authenticated-null",
            "independent_verifier": "isingfold.rl.complete_system.verify_terminal_evidence",
        },
        "work_accounting": {
            "initializer_backend": "exact-when-exposed-null-when-unavailable",
            "initializer_invocations": "one-restart_work-per-actual-top-level-call",
            "environment": "exact-registered-ledger-with-authenticated-initializer-debit",
            "unknown_coordinate_rule": (
                "null-propagates-through-receipts-and-forbids-policy-execution"
            ),
        },
        "wallclock_supervision": (
            "cooperative checks at initializer and policy decision boundaries; "
            "backend adapters remain responsible for hard per-call termination"
        ),
        "stock_pairing_endpoint": (
            "isingfold.rl.external_pairing.aggregate_learned_vs_stock"
        ),
        "stock_pairing_receipt_contract": "isingfold.external-complete-system-attempt-v2",
    }


def complete_system_metrics(
    receipts: Sequence[CompleteSystemReceipt],
) -> dict[str, object]:
    """Return descriptive metrics with an authenticated complete-system failure taxonomy.

    Generic post-initialization metrics identify initializer failures through population
    exclusion.  Whole-system evaluation intentionally keeps every such row in the population,
    so its initializer-stage denominator must instead be recovered from the detailed receipts.
    """

    from isingfold.rl.evaluate import secondary_metrics

    if not receipts:
        raise ValueError("complete-system metrics require a nonempty receipt batch")
    outcomes = [receipt.outcome for receipt in receipts]
    for receipt in receipts:
        receipt.outcome.validate_receipt(require_complete=False)
        if receipt.outcome.population_eligible is not True:
            raise ValueError("complete-system metrics cannot exclude a population row")

    zero_work = WorkVector()
    initializer_stage = [
        receipt for receipt in receipts if receipt.environment_budget_debit == zero_work
    ]
    environment_bootstrap = [
        receipt
        for receipt in receipts
        if receipt.environment_budget_debit != zero_work
        and not receipt.outcome.returned_valid
        and receipt.outcome.reason.startswith("SELECTED_INITIALIZER_REJECTED:")
    ]
    post_bootstrap = [
        receipt
        for receipt in receipts
        if not receipt.outcome.returned_valid
        and receipt.environment_budget_debit != zero_work
        and not receipt.outcome.reason.startswith("SELECTED_INITIALIZER_REJECTED:")
    ]
    valid = [receipt for receipt in receipts if receipt.outcome.returned_valid]
    if len(initializer_stage) + len(environment_bootstrap) + len(post_bootstrap) + len(valid) != len(
        receipts
    ):
        raise RuntimeError("complete-system failure taxonomy does not partition its denominator")

    attempt_statuses = Counter(
        attempt.status for receipt in receipts for attempt in receipt.attempts
    )
    terminal_reasons = Counter(receipt.outcome.reason for receipt in receipts)
    result = secondary_metrics(outcomes)
    result["initialization_failures"] = len(initializer_stage)
    result["initialization_failure_rate"] = len(initializer_stage) / len(receipts)
    result["complete_failure_taxonomy"] = {
        "denominator": len(receipts),
        "valid_returns": len(valid),
        "pre_policy_initializer_failures": len(initializer_stage),
        "no_valid_initializer_candidate": sum(
            receipt.selected_attempt is None for receipt in initializer_stage
        ),
        "initializer_work_incomplete": sum(
            receipt.outcome.reason == "INITIALIZER_WORK_INCOMPLETE"
            for receipt in initializer_stage
        ),
        "policy_environment_bootstrap_failures": len(environment_bootstrap),
        "post_bootstrap_failures": len(post_bootstrap),
        "initializer_attempt_status_counts": dict(sorted(attempt_statuses.items())),
        "terminal_reason_counts": dict(sorted(terminal_reasons.items())),
    }
    return result


def write_complete_system_receipts(
    path: str | os.PathLike[str],
    receipts: Sequence[CompleteSystemReceipt],
    *,
    overwrite: bool = False,
) -> str:
    """Atomically write canonical raw JSONL and return its file SHA-256."""

    if not receipts:
        raise EvaluationProtocolError("cannot write an empty complete-system receipt batch")
    keys = [item.pair_key for item in receipts]
    if len(keys) != len(set(keys)):
        raise EvaluationProtocolError("complete-system receipt batch has duplicate pair keys")
    reference = receipts[0]
    population_digest = reference.population.digest
    expected_keys = {
        (lineage, instance, repetition)
        for lineage, instance in reference.population.expected_instances
        for repetition in range(reference.population.expected_repetitions)
    }
    if set(keys) != expected_keys:
        raise EvaluationProtocolError(
            "complete-system receipt batch does not cover its sealed population denominator"
        )
    for item in receipts:
        if item.population.digest != population_digest:
            raise EvaluationProtocolError("complete-system batch mixes population identities")
        if (
            item.config_digest != reference.config_digest
            or item.context_digest != reference.context_digest
            or item.policy_context_digest != reference.policy_context_digest
            or item.work_cap != reference.work_cap
            or item.initializer != reference.initializer
            or item.controller != reference.controller
            or item.selector != reference.selector
        ):
            raise EvaluationProtocolError("complete-system batch mixes method/budget contracts")
    rows = sorted(receipts, key=lambda item: item.pair_key)
    content = b"".join(
        (
            json.dumps(
                item.as_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for item in rows
    )
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
            try:
                os.link(temporary, destination)
            except FileExistsError:
                raise FileExistsError(destination) from None
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def write_complete_system_evidence(
    path: str | os.PathLike[str],
    receipts: Sequence[CompleteSystemReceipt],
    *,
    overwrite: bool = False,
) -> str:
    """Atomically write one evidence or authenticated absence row per sealed attempt."""

    if not receipts:
        raise EvaluationProtocolError("cannot write an empty complete-system evidence batch")
    reference = receipts[0]
    expected_keys = {
        (lineage, instance, repetition)
        for lineage, instance in reference.population.expected_instances
        for repetition in range(reference.population.expected_repetitions)
    }
    keys = [receipt.pair_key for receipt in receipts]
    if len(keys) != len(set(keys)) or set(keys) != expected_keys:
        raise EvaluationProtocolError(
            "complete-system evidence batch does not cover its sealed population denominator"
        )
    records: list[CompleteSystemEvidenceRecord] = []
    for receipt in receipts:
        if (
            receipt.population != reference.population
            or receipt.config_digest != reference.config_digest
            or receipt.context_digest != reference.context_digest
            or receipt.policy_context_digest != reference.policy_context_digest
            or receipt.initializer != reference.initializer
            or receipt.controller != reference.controller
            or receipt.selector != reference.selector
        ):
            raise EvaluationProtocolError("complete-system evidence batch mixes contracts")
        if receipt.outcome.returned_valid and receipt.terminal_evidence is None:
            raise EvaluationProtocolError(
                "valid complete-system receipt has no in-memory terminal evidence to write"
            )
        if receipt.terminal_evidence is not None:
            try:
                _validate_evidence_outcome(receipt.terminal_evidence, receipt.outcome)
            except ValueError as exc:
                raise EvaluationProtocolError(
                    "terminal evidence differs from its projected outcome"
                ) from exc
        complete_digest = receipt.as_dict()["record_digest"]
        assert isinstance(complete_digest, str)
        records.append(
            CompleteSystemEvidenceRecord(
                instance=receipt.instance,
                lineage=receipt.lineage,
                repetition=receipt.repetition,
                population_digest=receipt.population.digest,
                complete_receipt_digest=complete_digest,
                terminal_evidence=receipt.terminal_evidence,
            )
        )
    content = b"".join(
        (
            json.dumps(
                record.as_dict(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for record in sorted(records, key=lambda item: item.pair_key)
    )
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
            try:
                os.link(temporary, destination)
            except FileExistsError:
                raise FileExistsError(destination) from None
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def write_complete_system_outcomes(
    path: str | os.PathLike[str],
    receipts: Sequence[CompleteSystemReceipt],
    *,
    overwrite: bool = False,
) -> str:
    """Atomically write an authenticated outcome projection linked to raw receipts."""

    if not receipts:
        raise EvaluationProtocolError("cannot write an empty complete-system outcome batch")
    reference = receipts[0]
    expected_keys = {
        (lineage, instance, repetition)
        for lineage, instance in reference.population.expected_instances
        for repetition in range(reference.population.expected_repetitions)
    }
    if {item.pair_key for item in receipts} != expected_keys or len(receipts) != len(expected_keys):
        raise EvaluationProtocolError(
            "complete-system outcome batch does not cover its sealed population denominator"
        )
    rows: list[dict[str, object]] = []
    for receipt in sorted(receipts, key=lambda item: item.pair_key):
        if (
            receipt.population != reference.population
            or receipt.config_digest != reference.config_digest
            or receipt.context_digest != reference.context_digest
            or receipt.policy_context_digest != reference.policy_context_digest
            or receipt.initializer != reference.initializer
            or receipt.controller != reference.controller
            or receipt.selector != reference.selector
        ):
            raise EvaluationProtocolError("complete-system outcome batch mixes contracts")
        complete_payload = receipt.as_dict()
        payload: dict[str, object] = {
            "schema": COMPLETE_SYSTEM_OUTCOME_SCHEMA,
            "schema_version": COMPLETE_SYSTEM_OUTCOME_VERSION,
            "instance": receipt.instance,
            "lineage": receipt.lineage,
            "repetition": receipt.repetition,
            "population_digest": receipt.population.digest,
            "complete_receipt_digest": complete_payload["record_digest"],
            "outcome": receipt.outcome.as_dict(),
        }
        payload["record_digest"] = stable_digest(payload)
        rows.append(payload)
    content = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
            "utf-8"
        )
        for row in rows
    )
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
            try:
                os.link(temporary, destination)
            except FileExistsError:
                raise FileExistsError(destination) from None
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _strict_json(line: str, line_number: int) -> Mapping[str, object]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise EvaluationProtocolError(
                    f"duplicate JSON key {key!r} on complete receipt line {line_number}"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            line,
            object_pairs_hook=no_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                EvaluationProtocolError(
                    f"nonfinite JSON scalar {value!r} on complete receipt line {line_number}"
                )
            ),
        )
    except json.JSONDecodeError as exc:
        raise EvaluationProtocolError(
            f"invalid JSON on complete receipt line {line_number}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise EvaluationProtocolError(f"complete receipt line {line_number} is not an object")
    return payload


_COMPLETE_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "instance",
        "lineage",
        "repetition",
        "system_seed",
        "evaluator_seed",
        "initializer",
        "controller",
        "selector",
        "population",
        "population_digest",
        "config_digest",
        "context_digest",
        "policy_context_digest",
        "work_cap",
        "attempts",
        "selected_attempt",
        "bootstrap_binding",
        "outcome",
        "terminal_evidence_digest",
        "initializer_work",
        "initializer_work_known_lower_bound",
        "environment_budget_debit",
        "environment_work",
        "total_work",
        "total_work_known_lower_bound",
        "cap_compliance",
        "initialization_seconds",
        "post_initialization_seconds",
        "online_seconds",
        "wallclock_cap_seconds",
        "wallclock_compliant",
        "record_digest",
    }
)
_ATTEMPT_KEYS = frozenset(
    {
        "attempt_index",
        "seed",
        "status",
        "reported_seconds",
        "observed_seconds",
        "backend_work",
        "backend_work_known_lower_bound",
        "runner_work",
        "total_work",
        "total_work_known_lower_bound",
        "embedding_digest",
        "validation_digest",
        "qubits",
        "max_chain",
        "detail",
        "backend_diagnostics",
    }
)


def _checked_keys(payload: Mapping[str, object], expected: frozenset[str], *, label: str) -> None:
    if set(payload) != expected:
        raise EvaluationProtocolError(
            f"{label} keys differ: missing={sorted(expected - set(payload))}, "
            f"unknown={sorted(set(payload) - expected)}"
        )


def _required_mapping(payload: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = payload[name]
    if not isinstance(value, Mapping):
        raise EvaluationProtocolError(f"complete receipt field {name!r} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise EvaluationProtocolError(f"complete receipt field {name!r} has a non-string key")
    return value


def _required_string(payload: Mapping[str, object], name: str) -> str:
    value = payload[name]
    if not isinstance(value, str) or not value:
        raise EvaluationProtocolError(f"complete receipt field {name!r} must be a nonempty string")
    return value


def _required_int(payload: Mapping[str, object], name: str) -> int:
    value = payload[name]
    if type(value) is not int:
        raise EvaluationProtocolError(f"complete receipt field {name!r} must be an integer")
    return value


def _optional_int(payload: Mapping[str, object], name: str) -> int | None:
    value = payload[name]
    if value is not None and type(value) is not int:
        raise EvaluationProtocolError(f"complete receipt field {name!r} must be an integer or null")
    return value


def _required_float(payload: Mapping[str, object], name: str) -> float:
    value = payload[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationProtocolError(f"complete receipt field {name!r} must be numeric")
    converted = float(value)
    if not math.isfinite(converted):
        raise EvaluationProtocolError(f"complete receipt field {name!r} must be finite")
    return converted


def _work_from_payload(payload: Mapping[str, object], *, label: str) -> WorkVector:
    if set(payload) != set(WORK_FIELDS):
        raise EvaluationProtocolError(f"{label} has missing or unknown work coordinates")
    if any(type(payload[name]) is not int or payload[name] < 0 for name in WORK_FIELDS):
        raise EvaluationProtocolError(f"{label} work coordinates must be nonnegative integers")
    return WorkVector(**{name: payload[name] for name in WORK_FIELDS})


def _partial_from_payload(
    payload: Mapping[str, object],
    lower_payload: Mapping[str, object],
    *,
    label: str,
) -> PartialWorkVector:
    if set(payload) != set(WORK_FIELDS):
        raise EvaluationProtocolError(f"{label} has missing or unknown work coordinates")
    if any(
        value is not None and (type(value) is not int or value < 0) for value in payload.values()
    ):
        raise EvaluationProtocolError(
            f"{label} work coordinates must be nonnegative integers or null"
        )
    lower = _work_from_payload(lower_payload, label=f"{label} known lower bound")
    try:
        return PartialWorkVector(
            **{name: payload[name] for name in WORK_FIELDS},
            known_lower_bound=lower,
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationProtocolError(f"{label} is semantically inconsistent") from exc


def _backend_identity_from_payload(payload: Mapping[str, object]) -> BackendIdentity:
    expected = frozenset({"method_id", "distribution", "version", "entrypoint", "implementation"})
    _checked_keys(payload, expected, label="initializer identity")
    return BackendIdentity(**{name: _required_string(payload, name) for name in expected})


def _component_from_payload(
    payload: Mapping[str, object], *, label: str
) -> FrozenComponentIdentity:
    expected = frozenset({"component_id", "version", "implementation", "artifact_sha256"})
    _checked_keys(payload, expected, label=label)
    return FrozenComponentIdentity(**{name: _required_string(payload, name) for name in expected})


def _population_from_payload(payload: Mapping[str, object]) -> CompletePopulationIdentity:
    expected = frozenset(
        {
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
    )
    _checked_keys(payload, expected, label="population identity")
    identities = payload["expected_instances"]
    if not isinstance(identities, list) or any(
        not isinstance(item, list)
        or len(item) != 2
        or any(not isinstance(value, str) for value in item)
        for item in identities
    ):
        raise EvaluationProtocolError("population expected_instances is malformed")
    includes_failures = payload["includes_initializer_failures"]
    if type(includes_failures) is not bool:
        raise EvaluationProtocolError("population failure-inclusion flag must be Boolean")
    raw_strata = payload["evaluation_strata"]
    raw_design = payload["confirmatory_design"]
    if not isinstance(raw_strata, list) or any(
        not isinstance(row, Mapping) for row in raw_strata
    ):
        raise EvaluationProtocolError("population evaluation strata are malformed")
    if not isinstance(raw_design, Mapping):
        raise EvaluationProtocolError("population confirmatory design is malformed")
    try:
        strata = tuple(EvaluationStratum.from_mapping(row) for row in raw_strata)
        design = ConfirmatoryEvaluationDesign.from_mapping(raw_design)
        population = CompletePopulationIdentity(
            population_id=_required_string(payload, "population_id"),
            source_manifest_sha256=_required_string(payload, "source_manifest_sha256"),
            task_payload_sha256=_required_string(payload, "task_payload_sha256"),
            expected_instances=tuple(tuple(item) for item in identities),
            expected_repetitions=_required_int(payload, "expected_repetitions"),
            evaluation_seed=_required_int(payload, "evaluation_seed"),
            evaluation_strata=strata,
            confirmatory_design=design,
            denominator_scope=_required_string(payload, "denominator_scope"),
            includes_initializer_failures=includes_failures,
        )
        if (
            payload["evaluation_strata_digest"] != population.evaluation_strata_digest
            or payload["size_bin_protocol"] != SIZE_BIN_PROTOCOL
            or payload["size_bin_boundary_convention"] != SIZE_BIN_BOUNDARY_CONVENTION
            or payload["confirmatory_design_digest"]
            != population.confirmatory_design.record_digest
        ):
            raise ValueError("population evaluation contract digests are invalid")
        return population
    except (TypeError, ValueError) as exc:
        raise EvaluationProtocolError("population identity is semantically invalid") from exc


def _attempt_from_payload(payload: Mapping[str, object]) -> InitializerAttemptReceipt:
    _checked_keys(payload, _ATTEMPT_KEYS, label="initializer attempt")
    backend_work = _partial_from_payload(
        _required_mapping(payload, "backend_work"),
        _required_mapping(payload, "backend_work_known_lower_bound"),
        label="initializer backend work",
    )
    runner_work = _work_from_payload(
        _required_mapping(payload, "runner_work"), label="initializer runner work"
    )
    diagnostics = _required_mapping(payload, "backend_diagnostics")
    try:
        attempt = InitializerAttemptReceipt(
            attempt_index=_required_int(payload, "attempt_index"),
            seed=_required_int(payload, "seed"),
            status=_required_string(payload, "status"),
            reported_seconds=_required_float(payload, "reported_seconds"),
            observed_seconds=_required_float(payload, "observed_seconds"),
            backend_work=backend_work,
            runner_work=runner_work,
            embedding_digest=payload["embedding_digest"],
            validation_digest=payload["validation_digest"],
            qubits=_optional_int(payload, "qubits"),
            max_chain=_optional_int(payload, "max_chain"),
            detail=payload["detail"],
            backend_diagnostics=diagnostics,
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationProtocolError("initializer attempt is semantically invalid") from exc
    if payload["total_work"] != attempt.total_work.as_dict() or (
        payload["total_work_known_lower_bound"] != attempt.total_work.known_lower_bound.as_dict()
    ):
        raise EvaluationProtocolError("initializer attempt total work is inconsistent")
    return attempt


def _receipt_from_payload(payload: Mapping[str, object]) -> CompleteSystemReceipt:
    _checked_keys(payload, _COMPLETE_RECEIPT_KEYS, label="complete-system receipt")
    if (
        payload["schema"] != COMPLETE_SYSTEM_SCHEMA
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != COMPLETE_SYSTEM_VERSION
    ):
        raise EvaluationProtocolError("unsupported complete-system receipt schema")
    recorded = payload["record_digest"]
    without_digest = {key: value for key, value in payload.items() if key != "record_digest"}
    if not _is_digest(recorded) or recorded != stable_digest(without_digest):
        raise EvaluationProtocolError("complete-system record digest mismatch")
    population = _population_from_payload(_required_mapping(payload, "population"))
    if payload["population_digest"] != population.digest:
        raise EvaluationProtocolError("complete-system population digest mismatch")
    raw_attempts = payload["attempts"]
    if not isinstance(raw_attempts, list) or any(
        not isinstance(item, Mapping) for item in raw_attempts
    ):
        raise EvaluationProtocolError("complete-system attempts must be an object array")
    attempts = tuple(_attempt_from_payload(item) for item in raw_attempts)
    outcome_payload = _required_mapping(payload, "outcome")
    outcome = EpisodeOutcome.from_dict(outcome_payload, require_complete=False)
    initializer_work = _partial_from_payload(
        _required_mapping(payload, "initializer_work"),
        _required_mapping(payload, "initializer_work_known_lower_bound"),
        label="initializer aggregate work",
    )
    total_work = _partial_from_payload(
        _required_mapping(payload, "total_work"),
        _required_mapping(payload, "total_work_known_lower_bound"),
        label="complete-system total work",
    )
    environment_work = _work_from_payload(
        _required_mapping(payload, "environment_work"), label="environment work"
    )
    environment_budget_debit = _work_from_payload(
        _required_mapping(payload, "environment_budget_debit"),
        label="environment budget debit",
    )
    work_cap = _work_from_payload(_required_mapping(payload, "work_cap"), label="work cap")
    raw_compliance = _required_mapping(payload, "cap_compliance")
    if set(raw_compliance) != set(WORK_FIELDS) or any(
        value is not None and type(value) is not bool for value in raw_compliance.values()
    ):
        raise EvaluationProtocolError("cap compliance block is malformed")
    wallclock_compliant = payload["wallclock_compliant"]
    if type(wallclock_compliant) is not bool:
        raise EvaluationProtocolError("wallclock_compliant must be Boolean")
    try:
        bootstrap_binding = payload["bootstrap_binding"]
        if bootstrap_binding is not None and not isinstance(bootstrap_binding, Mapping):
            raise EvaluationProtocolError("bootstrap_binding must be an object or null")
        receipt = CompleteSystemReceipt(
            instance=_required_string(payload, "instance"),
            lineage=_required_string(payload, "lineage"),
            repetition=_required_int(payload, "repetition"),
            system_seed=_required_int(payload, "system_seed"),
            evaluator_seed=_optional_int(payload, "evaluator_seed"),
            initializer=_backend_identity_from_payload(_required_mapping(payload, "initializer")),
            controller=_component_from_payload(
                _required_mapping(payload, "controller"), label="controller identity"
            ),
            selector=_component_from_payload(
                _required_mapping(payload, "selector"), label="selector identity"
            ),
            population=population,
            config_digest=_required_string(payload, "config_digest"),
            context_digest=_required_string(payload, "context_digest"),
            policy_context_digest=_required_string(payload, "policy_context_digest"),
            work_cap=work_cap,
            attempts=attempts,
            selected_attempt=_optional_int(payload, "selected_attempt"),
            bootstrap_binding=bootstrap_binding,
            outcome=outcome,
            terminal_evidence_digest=payload["terminal_evidence_digest"],
            terminal_evidence=None,
            initializer_work=initializer_work,
            environment_budget_debit=environment_budget_debit,
            environment_work=environment_work,
            total_work=total_work,
            cap_compliance=raw_compliance,
            initialization_seconds=_required_float(payload, "initialization_seconds"),
            post_initialization_seconds=_required_float(payload, "post_initialization_seconds"),
            online_seconds=_required_float(payload, "online_seconds"),
            wallclock_cap_seconds=_required_float(payload, "wallclock_cap_seconds"),
            wallclock_compliant=wallclock_compliant,
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationProtocolError("complete-system receipt is semantically invalid") from exc
    expected_system_seed = _identity_seed(
        population.evaluation_seed,
        "policy",
        instance=receipt.instance,
        lineage=receipt.lineage,
        repetition=receipt.repetition,
    )
    expected_evaluator_seed = _identity_seed(
        population.evaluation_seed,
        "final-evaluator",
        instance=receipt.instance,
        lineage=receipt.lineage,
        repetition=receipt.repetition,
    )
    if receipt.system_seed != expected_system_seed or (
        receipt.outcome.returned_valid and receipt.evaluator_seed != expected_evaluator_seed
    ):
        raise EvaluationProtocolError("complete-system seed schedule is inconsistent")
    for attempt in receipt.attempts:
        if attempt.seed != _attempt_seed(
            receipt.system_seed, receipt.initializer.method_id, attempt.attempt_index
        ):
            raise EvaluationProtocolError("initializer attempt seed schedule is inconsistent")
    if receipt.as_dict() != dict(payload):
        raise EvaluationProtocolError("complete-system receipt is not canonical")
    return receipt


def read_complete_system_receipts(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> tuple[CompleteSystemReceipt, ...]:
    """Load, authenticate and reconstruct complete-system receipts fail closed."""

    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise EvaluationProtocolError("complete-system receipt file digest mismatch")
    receipts: list[CompleteSystemReceipt] = []
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvaluationProtocolError("complete-system receipt file is not valid UTF-8") from exc
    for line_number, raw in enumerate(decoded.splitlines(), start=1):
        if not raw:
            raise EvaluationProtocolError(f"blank complete receipt line {line_number}")
        try:
            receipts.append(_receipt_from_payload(_strict_json(raw, line_number)))
        except EvaluationProtocolError as exc:
            raise EvaluationProtocolError(
                f"invalid complete-system receipt on line {line_number}: {exc}"
            ) from exc
    if not receipts:
        raise EvaluationProtocolError("complete-system receipt file is empty")
    reference = receipts[0]
    keys = [item.pair_key for item in receipts]
    if len(keys) != len(set(keys)):
        raise EvaluationProtocolError("complete-system receipt file has duplicate pair keys")
    expected_keys = {
        (lineage, instance, repetition)
        for lineage, instance in reference.population.expected_instances
        for repetition in range(reference.population.expected_repetitions)
    }
    if set(keys) != expected_keys:
        raise EvaluationProtocolError(
            "complete-system file does not cover its sealed population denominator"
        )
    for item in receipts:
        if (
            item.population != reference.population
            or item.config_digest != reference.config_digest
            or item.context_digest != reference.context_digest
            or item.policy_context_digest != reference.policy_context_digest
            or item.work_cap != reference.work_cap
            or item.initializer != reference.initializer
            or item.controller != reference.controller
            or item.selector != reference.selector
            or item.wallclock_cap_seconds != reference.wallclock_cap_seconds
        ):
            raise EvaluationProtocolError("complete-system file mixes method/budget contracts")
    return tuple(sorted(receipts, key=lambda item: item.pair_key))


_COMPLETE_OUTCOME_RECORD_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "instance",
        "lineage",
        "repetition",
        "population_digest",
        "complete_receipt_digest",
        "outcome",
        "record_digest",
    }
)


def read_complete_system_outcomes(
    path: str | os.PathLike[str],
    *,
    receipts: Sequence[CompleteSystemReceipt],
    expected_sha256: str | None = None,
) -> tuple[Mapping[str, object], ...]:
    """Authenticate an outcome projection against the sealed complete receipts.

    Outcome rows are convenience projections, not independent evidence.  This reader therefore
    requires the authenticated receipts that authorize every projected value and denominator.
    """

    if not receipts:
        raise EvaluationProtocolError(
            "complete-system outcomes require a nonempty authenticated receipt batch"
        )
    receipt_by_key = {receipt.pair_key: receipt for receipt in receipts}
    if len(receipt_by_key) != len(receipts):
        raise EvaluationProtocolError("complete-system receipts contain duplicate pair keys")
    reference = receipts[0]
    expected_keys = {
        (lineage, instance, repetition)
        for lineage, instance in reference.population.expected_instances
        for repetition in range(reference.population.expected_repetitions)
    }
    if set(receipt_by_key) != expected_keys:
        raise EvaluationProtocolError(
            "complete-system receipts do not cover their sealed population denominator"
        )
    if any(receipt.population != reference.population for receipt in receipts):
        raise EvaluationProtocolError("complete-system receipts mix populations")

    content = Path(path).read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise EvaluationProtocolError("complete-system outcome file digest mismatch")
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvaluationProtocolError("complete-system outcome file is not valid UTF-8") from exc
    rows: list[Mapping[str, object]] = []
    observed_keys: list[tuple[str, str, int]] = []
    for line_number, raw in enumerate(decoded.splitlines(), start=1):
        if not raw:
            raise EvaluationProtocolError(f"blank complete-system outcome line {line_number}")
        try:
            payload = _strict_json(raw, line_number)
            _checked_keys(payload, _COMPLETE_OUTCOME_RECORD_KEYS, label="complete outcome record")
            if (
                payload["schema"] != COMPLETE_SYSTEM_OUTCOME_SCHEMA
                or type(payload["schema_version"]) is not int
                or payload["schema_version"] != COMPLETE_SYSTEM_OUTCOME_VERSION
            ):
                raise EvaluationProtocolError("unsupported complete-system outcome schema")
            recorded = payload["record_digest"]
            without_digest = {
                key: value for key, value in payload.items() if key != "record_digest"
            }
            if not _is_digest(recorded) or recorded != stable_digest(without_digest):
                raise EvaluationProtocolError("complete-system outcome record digest mismatch")
            key = (
                _required_string(payload, "lineage"),
                _required_string(payload, "instance"),
                _required_int(payload, "repetition"),
            )
            receipt = receipt_by_key.get(key)
            if receipt is None:
                raise EvaluationProtocolError(
                    "complete-system outcome has no authenticated receipt link"
                )
            receipt_payload = receipt.as_dict()
            expected: dict[str, object] = {
                "schema": COMPLETE_SYSTEM_OUTCOME_SCHEMA,
                "schema_version": COMPLETE_SYSTEM_OUTCOME_VERSION,
                "instance": receipt.instance,
                "lineage": receipt.lineage,
                "repetition": receipt.repetition,
                "population_digest": receipt.population.digest,
                "complete_receipt_digest": receipt_payload["record_digest"],
                "outcome": receipt.outcome.as_dict(),
            }
            expected["record_digest"] = stable_digest(expected)
            if dict(payload) != expected:
                raise EvaluationProtocolError(
                    "complete-system projected outcome differs from its authenticated receipt link"
                )
            canonical = json.dumps(
                expected,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            if raw != canonical:
                raise EvaluationProtocolError("complete-system outcome row is not canonical JSON")
            observed_keys.append(key)
            rows.append(MappingProxyType(expected))
        except EvaluationProtocolError as exc:
            raise EvaluationProtocolError(
                f"invalid complete-system outcome on line {line_number}: {exc}"
            ) from exc
    if not rows:
        raise EvaluationProtocolError("complete-system outcome file is empty")
    if len(observed_keys) != len(set(observed_keys)):
        raise EvaluationProtocolError("complete-system outcome file has duplicate pair keys")
    if set(observed_keys) != expected_keys:
        raise EvaluationProtocolError(
            "complete-system outcome file does not cover its sealed population denominator"
        )
    if observed_keys != sorted(observed_keys):
        raise EvaluationProtocolError("complete-system outcome rows are not canonically ordered")
    return tuple(rows)


_COMPLETE_EVIDENCE_RECORD_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "instance",
        "lineage",
        "repetition",
        "population_digest",
        "complete_receipt_digest",
        "terminal_evidence",
        "record_digest",
    }
)


def _evidence_record_from_payload(
    payload: Mapping[str, object],
) -> CompleteSystemEvidenceRecord:
    _checked_keys(payload, _COMPLETE_EVIDENCE_RECORD_KEYS, label="complete evidence record")
    if (
        payload["schema"] != COMPLETE_SYSTEM_EVIDENCE_SCHEMA
        or type(payload["schema_version"]) is not int
        or payload["schema_version"] != COMPLETE_SYSTEM_EVIDENCE_VERSION
    ):
        raise EvaluationProtocolError("unsupported complete-system evidence-record schema")
    recorded = payload["record_digest"]
    without_digest = {key: value for key, value in payload.items() if key != "record_digest"}
    if not _is_digest(recorded) or recorded != stable_digest(without_digest):
        raise EvaluationProtocolError("complete-system evidence record digest mismatch")
    raw_evidence = payload["terminal_evidence"]
    if raw_evidence is not None and not isinstance(raw_evidence, Mapping):
        raise EvaluationProtocolError("terminal evidence must be an object or null")
    evidence = None if raw_evidence is None else _terminal_evidence_from_payload(raw_evidence)
    try:
        result = CompleteSystemEvidenceRecord(
            instance=_required_string(payload, "instance"),
            lineage=_required_string(payload, "lineage"),
            repetition=_required_int(payload, "repetition"),
            population_digest=_required_string(payload, "population_digest"),
            complete_receipt_digest=_required_string(payload, "complete_receipt_digest"),
            terminal_evidence=evidence,
        )
    except (TypeError, ValueError) as exc:
        raise EvaluationProtocolError(
            "complete-system evidence record is semantically invalid"
        ) from exc
    if result.as_dict() != dict(payload):
        raise EvaluationProtocolError("complete-system evidence record is not canonical")
    return result


def read_complete_system_evidence(
    path: str | os.PathLike[str],
    *,
    receipts: Sequence[CompleteSystemReceipt] | None = None,
    tasks: Sequence[EmbeddingTask] | None = None,
    context: Context | None = None,
    expected_sha256: str | None = None,
) -> tuple[CompleteSystemEvidenceRecord, ...]:
    """Authenticate a sidecar and optionally recompile it against tasks and receipts."""

    source = Path(path)
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise EvaluationProtocolError("complete-system evidence file digest mismatch")
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvaluationProtocolError("complete-system evidence file is not valid UTF-8") from exc
    records: list[CompleteSystemEvidenceRecord] = []
    for line_number, raw in enumerate(decoded.splitlines(), start=1):
        if not raw:
            raise EvaluationProtocolError(f"blank complete-system evidence line {line_number}")
        try:
            records.append(_evidence_record_from_payload(_strict_json(raw, line_number)))
        except EvaluationProtocolError as exc:
            raise EvaluationProtocolError(
                f"invalid complete-system evidence on line {line_number}: {exc}"
            ) from exc
    if not records:
        raise EvaluationProtocolError("complete-system evidence file is empty")
    keys = [record.pair_key for record in records]
    if len(keys) != len(set(keys)):
        raise EvaluationProtocolError("complete-system evidence file has duplicate pair keys")
    if len({record.population_digest for record in records}) != 1:
        raise EvaluationProtocolError("complete-system evidence file mixes populations")

    receipt_by_key: dict[tuple[str, str, int], CompleteSystemReceipt] | None = None
    if receipts is not None:
        receipt_by_key = {receipt.pair_key: receipt for receipt in receipts}
        if len(receipt_by_key) != len(receipts) or set(receipt_by_key) != set(keys):
            raise EvaluationProtocolError("complete-system evidence and receipt pair sets differ")
        for record in records:
            receipt = receipt_by_key[record.pair_key]
            expected_complete_digest = receipt.as_dict()["record_digest"]
            observed_evidence_digest = (
                None if record.terminal_evidence is None else record.terminal_evidence.digest
            )
            if (
                record.population_digest != receipt.population.digest
                or record.complete_receipt_digest != expected_complete_digest
                or observed_evidence_digest != receipt.terminal_evidence_digest
            ):
                raise EvaluationProtocolError(
                    "complete-system evidence does not match its authenticated receipt link"
                )
            if receipt.outcome.returned_valid != (record.terminal_evidence is not None):
                raise EvaluationProtocolError(
                    "terminal-evidence presence disagrees with complete-system validity"
                )
            if record.terminal_evidence is not None:
                try:
                    _validate_evidence_outcome(record.terminal_evidence, receipt.outcome)
                except ValueError as exc:
                    raise EvaluationProtocolError(
                        "terminal evidence differs from its projected outcome"
                    ) from exc

    if (tasks is None) != (context is None):
        raise ValueError("tasks and context must be supplied together for recompilation")
    if tasks is not None and context is not None:
        if receipts is not None:
            reference_receipt = receipts[0]
            reference_receipt.population.validate_tasks(tasks)
            if any(
                receipt.context_digest != _context_digest(context)
                or receipt.policy_context_digest
                != _context_digest(
                    context
                    if receipt.bootstrap_binding is not None
                    else complete_policy_context(context)
                )
                for receipt in receipts
            ):
                raise EvaluationProtocolError(
                    "terminal-evidence context differs from its authenticated receipts"
                )
        task_by_key = {(task.lineage or task.name, task.name): task for task in tasks}
        if len(task_by_key) != len(tasks):
            raise EvaluationProtocolError("terminal-evidence task set has duplicate identities")
        expected_task_keys = {(lineage, instance) for lineage, instance, _ in keys}
        if set(task_by_key) != expected_task_keys:
            raise EvaluationProtocolError(
                "terminal-evidence task set differs from the evidence population"
            )
        for record in records:
            if record.terminal_evidence is None:
                continue
            outcome = None if receipt_by_key is None else receipt_by_key[record.pair_key].outcome
            verify_terminal_evidence(
                record.terminal_evidence,
                task=task_by_key[(record.lineage, record.instance)],
                context=context,
                outcome=outcome,
            )
    return tuple(sorted(records, key=lambda item: item.pair_key))


def verify_complete_system_receipts(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> tuple[Mapping[str, object], ...]:
    """Authenticate receipts and return immutable canonical payloads."""

    receipts = read_complete_system_receipts(path, expected_sha256=expected_sha256)
    return tuple(MappingProxyType(item.as_dict()) for item in receipts)


__all__ = [
    "COMPLETE_SYSTEM_SCHEMA",
    "COMPLETE_SYSTEM_EVIDENCE_SCHEMA",
    "COMPLETE_SYSTEM_OUTCOME_SCHEMA",
    "CompleteInitializerBackend",
    "CompleteInitializerResult",
    "CompletePopulationIdentity",
    "CompleteSystemBackendError",
    "CompleteSystemConfig",
    "CompleteSystemEvidenceRecord",
    "CompleteSystemPairing",
    "CompleteSystemReceipt",
    "FrozenComponentIdentity",
    "InitializerAttemptReceipt",
    "LACMinorminerInitializerBackend",
    "PartialWorkVector",
    "TerminalEvidence",
    "complete_system_method_metadata",
    "pair_with_stock_minorminer",
    "read_complete_system_evidence",
    "read_complete_system_outcomes",
    "read_complete_system_receipts",
    "run_complete_system",
    "task_population_digest",
    "verify_terminal_evidence",
    "verify_complete_system_receipts",
    "write_complete_system_outcomes",
    "write_complete_system_evidence",
    "write_complete_system_receipts",
]
