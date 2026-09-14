"""Validation-only tuning and preregistration for the stock-minorminer baseline.

This module owns the scientific boundary between validation-time baseline tuning and the
sealed test endpoint. It contains no test-data loader and treats every test marker as an
integrity failure. The selected strategy is useful at test time only after the canonical
selection receipt is authenticated with a caller-supplied, out-of-band file SHA-256.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from isingfold.rl.contracts import WORK_FIELDS, stable_digest
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.experiment_selection import crossed_bootstrap_bounds

EXTERNAL_TUNING_REGISTRY_SCHEMA = "isingfold.external-minorminer-tuning-registry"
EXTERNAL_TUNING_REGISTRY_VERSION = 1
EXTERNAL_TUNING_REGISTRY_RECORD_DIGEST = (
    "af9c25b8b9fb64d9c83a1189092e66ee72c98c62946fb63b91f385f4fad4a941"
)
EXTERNAL_TUNING_REGISTRY_FILE_SHA256 = (
    "0ddeaa14f4179f0e0abae03b58d3e9aa869501c104fb12ada413c7b5465b9e62"
)
EXTERNAL_TUNING_RUN_SCHEMA = "isingfold.external-tuning-validation-run"
EXTERNAL_TUNING_RUN_VERSION = 2
EXTERNAL_TUNING_SELECTION_SCHEMA = "isingfold.external-tuning-selection"
EXTERNAL_TUNING_SELECTION_VERSION = 2

TUNING_SEEDS = (1103, 2207, 3301)
REGISTERED_CANDIDATE_IDENTITIES = (
    ("stock-default-v1", "73d278aad634294db799e0236fb7ffecf38bc0f0b0b9354596b997349410c488"),
    ("time-resource-p5-v1", "e17f249a8e2f779d9fc85802763aa08b91c6df003ec412b361c26185b122897e"),
    ("time-resource-p10-v1", "6c4b5bcfed0d1ee08a27a7bce9b350c24709935c1ddc39bcd7ced2e3745070b4"),
    ("time-resource-p20-v1", "45abe6d95a06b5eb6bf0860fa8c3cebc2b1d4d90c38ea72dbabc0406d7f4c434"),
    ("time-quality-p5-v1", "46b0ae93b56e710d70ea7e33f5bae6ceb1f43d256a76eef8d63f412f77537a6b"),
    ("time-quality-p10-v1", "c1540bf4eae124eece17f5cd2ebf624a72c98455e62e9b07dca0d538051cb742"),
    ("time-quality-p20-v1", "9932ad7c0de8daf2fcc826f04572199a1f1f56c762046fa6aed3159e6a5c92a1"),
)
FAILURE_REASONS = (
    "EXTERNAL_VALID_RETURN",
    "EXTERNAL_NO_VALID_EMBEDDING",
    "EXTERNAL_COMPLETE_SYSTEM_WALLCLOCK_EXHAUSTED",
    "EXTERNAL_KNOWN_WORK_CAP_EXHAUSTED",
)
KNOWN_WORK_FIELDS = (
    "compiler_calls",
    "validator_calls",
    "restart_work",
    "evaluator_reads",
    "feature_work",
)
UNKNOWN_NATIVE_WORK_FIELDS = (
    "decisions",
    "route_expansions",
    "materializations",
    "cut_edge_visits",
)
RUNTIME_IDENTITY_FIELDS = (
    "runtime_platform",
    "inference_device_type",
    "inference_device_name",
    "inference_threads",
    "deterministic",
)

_HEX = frozenset("0123456789abcdef")
_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "family",
        "outer_restart_policy",
        "outer_restart_cap",
        "native_parameters",
        "candidate_ranking",
        "strength_rule",
        "online_evaluator_feedback",
    }
)
_NATIVE_PARAMETER_FIELDS = frozenset(
    {"tries", "threads", "max_no_improvement", "chainlength_patience"}
)
_REGISTRY_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "registry_id",
        "stage",
        "partition",
        "test_data_opened",
        "grid",
        "external_config",
        "backend",
        "tuning_seeds",
        "repetitions",
        "minimum_independent_lineages",
        "evaluation",
        "bootstrap",
        "selection",
        "failure_denominator",
        "work_disclosure",
        "candidates",
        "record_digest",
    }
)


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _exact_keys(
    value: Mapping[str, object], expected: set[str] | frozenset[str], label: str
) -> None:
    if set(value) != set(expected):
        raise ValueError(
            f"{label} fields differ: missing={sorted(set(expected) - set(value))}, "
            f"unknown={sorted(set(value) - set(expected))}"
        )


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _finite(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{label} must be finite and >= {minimum}")
    return result


def _strict_json_bytes(content: bytes, label: str) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> object:
        raise ValueError(f"nonfinite JSON scalar {value!r}")

    try:
        decoded = content.decode("utf-8")
        payload = json.loads(
            decoded,
            object_pairs_hook=no_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} is not strict finite UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _freeze(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in sorted(value.items())})
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: object) -> object:
    return json.loads(canonical_json_bytes(value))


def _frozen_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError(f"{label} must be a nonempty object")
    frozen = _freeze(_plain(value))
    assert isinstance(frozen, Mapping)
    return frozen


def _verify_record(payload: Mapping[str, object], label: str) -> str:
    digest = payload.get("record_digest")
    if not _is_digest(digest):
        raise ValueError(f"{label} has no valid record digest")
    observed = content_digest(
        {key: value for key, value in payload.items() if key != "record_digest"}
    )
    if not hmac.compare_digest(str(digest), observed):
        raise ValueError(f"{label} record digest mismatch")
    return str(digest)


@dataclass(frozen=True)
class ExternalTuningCandidate:
    """One finite, preregistered stock-minorminer deployment strategy."""

    candidate_id: str
    family: str
    outer_restart_policy: str
    outer_restart_cap: int
    native_parameters: Mapping[str, int]
    candidate_ranking: str
    strength_rule: str
    online_evaluator_feedback: bool

    def __post_init__(self) -> None:
        if not self.candidate_id or self.family not in {
            "stock-default",
            "time-saturating-resource",
            "time-saturating-quality",
        }:
            raise ValueError("external tuning candidate identity is invalid")
        if self.outer_restart_policy not in {
            "single-native-call",
            "until-wallclock-or-outer-cap",
        }:
            raise ValueError("external tuning candidate restart policy is invalid")
        _integer(self.outer_restart_cap, "outer restart cap", minimum=1)
        parameters = dict(self.native_parameters)
        _exact_keys(parameters, _NATIVE_PARAMETER_FIELDS, "native parameter registry")
        if any(type(value) is not int or value <= 0 for value in parameters.values()):
            raise ValueError("native tuning parameters must be positive integers")
        if parameters["threads"] != 1:
            raise ValueError("stock tuning fixes one native thread")
        if self.online_evaluator_feedback is not False:
            raise ValueError("deployment baseline tuning cannot use evaluator feedback")
        resource_ranking = "qubits-max_chain-embedding_digest-restart_index"
        quality_ranking = (
            "frozen-selector-max-p_solve-then-qubits-max_chain-embedding_digest-restart_index"
        )
        if self.family == "stock-default":
            expected = ("single-native-call", 1, 10, resource_ranking)
            observed = (
                self.outer_restart_policy,
                self.outer_restart_cap,
                parameters["tries"],
                self.candidate_ranking,
            )
            if observed != expected or self.strength_rule != (
                "frozen-selector-argmax-four-programs"
            ):
                raise ValueError("stock-default candidate changes documented defaults")
        else:
            if (
                self.outer_restart_policy != "until-wallclock-or-outer-cap"
                or self.outer_restart_cap != 64
                or parameters["tries"] != 1
            ):
                raise ValueError("time-saturating candidate changes its restart contract")
            expected_ranking = (
                quality_ranking if self.family == "time-saturating-quality" else resource_ranking
            )
            expected_strength = (
                "same-frozen-selector-argmax-four-programs"
                if self.family == "time-saturating-quality"
                else "frozen-selector-argmax-four-programs"
            )
            if (
                self.candidate_ranking != expected_ranking
                or self.strength_rule != expected_strength
            ):
                raise ValueError("time-saturating candidate changes its ranking contract")
        object.__setattr__(self, "native_parameters", MappingProxyType(parameters))

    @classmethod
    def from_mapping(cls, value: object) -> ExternalTuningCandidate:
        if not isinstance(value, Mapping):
            raise ValueError("external tuning candidate must be an object")
        _exact_keys(value, _CANDIDATE_FIELDS, "external tuning candidate")
        parameters = value["native_parameters"]
        if not isinstance(parameters, Mapping):
            raise ValueError("native_parameters must be an object")
        return cls(
            candidate_id=value["candidate_id"],  # type: ignore[arg-type]
            family=value["family"],  # type: ignore[arg-type]
            outer_restart_policy=value["outer_restart_policy"],  # type: ignore[arg-type]
            outer_restart_cap=value["outer_restart_cap"],  # type: ignore[arg-type]
            native_parameters=dict(parameters),  # type: ignore[arg-type]
            candidate_ranking=value["candidate_ranking"],  # type: ignore[arg-type]
            strength_rule=value["strength_rule"],  # type: ignore[arg-type]
            online_evaluator_feedback=value["online_evaluator_feedback"],  # type: ignore[arg-type]
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "outer_restart_policy": self.outer_restart_policy,
            "outer_restart_cap": self.outer_restart_cap,
            "native_parameters": dict(self.native_parameters),
            "candidate_ranking": self.candidate_ranking,
            "strength_rule": self.strength_rule,
            "online_evaluator_feedback": self.online_evaluator_feedback,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.as_dict())


@dataclass(frozen=True)
class ExternalTuningRegistry:
    """Authenticated finite validation design for selecting one external strategy."""

    path: Path
    file_sha256: str
    record_digest: str
    registry_id: str
    grid_file_sha256: str
    external_config_digest: str
    external_config_file_sha256: str
    tuning_seeds: tuple[int, ...]
    repetitions: int
    minimum_independent_lineages: int
    audit_reads: int
    online_wallclock_seconds: float
    candidates: tuple[ExternalTuningCandidate, ...]
    evaluation: Mapping[str, object]
    bootstrap: Mapping[str, object]
    selection: Mapping[str, object]
    failure_denominator: Mapping[str, object]
    work_disclosure: Mapping[str, object]

    def candidate_for_index(self, index: int) -> ExternalTuningCandidate:
        if type(index) is not int or not 0 <= index < len(self.candidates):
            raise ValueError("external tuning candidate index is outside the registry")
        return self.candidates[index]

    def seed_for_index(self, index: int) -> int:
        if type(index) is not int or not 0 <= index < len(self.tuning_seeds):
            raise ValueError("external tuning seed index is outside the registry")
        return self.tuning_seeds[index]


def _validate_registry_protocol(payload: Mapping[str, object]) -> None:
    expected_evaluation = {
        "audit_reads": 4096,
        "population_scope": "all-validation-policy-instances-before-initialization",
        "aggregation": "equal-tuning-seed-then-equal-immutable-base-lineage",
        "primary_metric": "unconditional-if-q3-s0-utility",
        "failure_utility": 0.0,
        "final_evaluator": "one-fresh-independent-4096-read-block-after-selection",
    }
    expected_bootstrap = {
        "replicates": 20_000,
        "seed": 15_485_863,
        "two_sided_alpha": 0.05,
        "sampling": "crossed-tuning-seed-and-immutable-base-lineage-with-replacement",
        "decision_use": "uncertainty-only-not-selection",
    }
    expected_selection = {
        "complete_census_required": True,
        "seed_selection_forbidden": True,
        "primary": "maximize-unconditional-if-q3-s0-utility",
        "tie_breaks": [
            "higher-valid-return-rate",
            "lower-mean-total-online-seconds",
            "lower-known-outer-restart-work",
            "earlier-registered-candidate",
        ],
    }
    expected_failure = {
        "ordinary_terminal_reasons": list(FAILURE_REASONS),
        "all_attempts_in_denominator": True,
        "integrity_errors_abort": True,
    }
    expected_work = {
        "exact_coordinates": list(KNOWN_WORK_FIELDS),
        "unavailable_native_coordinates": list(UNKNOWN_NATIVE_WORK_FIELDS),
        "equal_internal_work_claimed": False,
        "total_online_wallclock_comparable": True,
    }
    for name, expected in (
        ("evaluation", expected_evaluation),
        ("bootstrap", expected_bootstrap),
        ("selection", expected_selection),
        ("failure_denominator", expected_failure),
        ("work_disclosure", expected_work),
    ):
        if payload.get(name) != expected:
            raise ValueError(f"external tuning {name} differs from the fixed protocol")


def load_external_tuning_registry(
    path: str | os.PathLike[str],
    *,
    expected_file_sha256: str,
    grid_path: str | os.PathLike[str],
    external_config_path: str | os.PathLike[str],
) -> ExternalTuningRegistry:
    """Authenticate the immutable registry and its exact grid/config dependencies."""

    if not _is_digest(expected_file_sha256):
        raise ValueError("external tuning registry requires an out-of-band SHA-256")
    if expected_file_sha256 != EXTERNAL_TUNING_REGISTRY_FILE_SHA256:
        raise ValueError("external tuning registry differs from the fixed v1 registry")
    registry_path = Path(path)
    content = registry_path.read_bytes()
    observed_file_sha = hashlib.sha256(content).hexdigest()
    if not hmac.compare_digest(observed_file_sha, expected_file_sha256):
        raise ValueError("external tuning registry file differs from its out-of-band pin")
    payload = _strict_json_bytes(content, "external tuning registry")
    _exact_keys(payload, _REGISTRY_FIELDS, "external tuning registry")
    record_digest = _verify_record(payload, "external tuning registry")
    if record_digest != EXTERNAL_TUNING_REGISTRY_RECORD_DIGEST:
        raise ValueError("external tuning registry differs from the fixed v1 registry")
    if (
        payload["schema"] != EXTERNAL_TUNING_REGISTRY_SCHEMA
        or payload["schema_version"] != EXTERNAL_TUNING_REGISTRY_VERSION
        or payload["registry_id"]
        != "stock-minorminer-0.2.22-validation-tuning-hybrid-v1"
        or payload["stage"] != "validation-only-baseline-preregistration"
        or payload["partition"] != "validation"
        or payload["test_data_opened"] is not False
    ):
        raise ValueError("external tuning is validation-only and cannot open test data")

    grid_identity = payload["grid"]
    if not isinstance(grid_identity, Mapping):
        raise ValueError("external tuning grid identity must be an object")
    _exact_keys(
        grid_identity,
        {"schema", "schema_version", "name", "file_sha256"},
        "external tuning grid identity",
    )
    grid_content = Path(grid_path).read_bytes()
    grid_file_sha = hashlib.sha256(grid_content).hexdigest()
    grid = _strict_json_bytes(grid_content, "external tuning staged grid")
    if (
        grid_identity
        != {
            "schema": "isingfold.staged-grid",
            "schema_version": 2,
            "name": "if-core-v2-profile-i-hybrid-chimera-registered",
            "file_sha256": grid_file_sha,
        }
        or grid_file_sha != "d3a7cd99c0c96c1c2f7fdd974d0856aee94ccbbe7a01f30b15d4724632712d09"
        or grid.get("schema") != grid_identity["schema"]
        or grid.get("schema_version") != grid_identity["schema_version"]
        or grid.get("name") != grid_identity["name"]
    ):
        raise ValueError("external tuning staged-grid pin differs")

    config_identity = payload["external_config"]
    if not isinstance(config_identity, Mapping):
        raise ValueError("external tuning config identity must be an object")
    _exact_keys(
        config_identity,
        {"schema", "schema_version", "semantic_digest", "file_sha256"},
        "external tuning config identity",
    )
    config_content = Path(external_config_path).read_bytes()
    config_file_sha = hashlib.sha256(config_content).hexdigest()
    config = _strict_json_bytes(config_content, "external complete-system config")
    config_digest = stable_digest(config)
    if (
        config_identity.get("schema") != "isingfold.external-complete-system-config"
        or config_identity.get("schema_version") != 2
        or config.get("schema") != config_identity["schema"]
        or config.get("schema_version") != config_identity["schema_version"]
        or config_identity.get("semantic_digest") != config_digest
        or config_identity.get("file_sha256") != config_file_sha
    ):
        raise ValueError("external tuning complete-system config pin differs")
    if (
        config.get("expected_backend_version") != "0.2.22"
        or config.get("audit_reads") != 4096
        or config.get("max_restarts") != 64
        or config.get("online_evaluator_feedback") is not False
    ):
        raise ValueError("external tuning base config changes the fixed envelope")

    if payload.get("backend") != {
        "distribution": "minorminer",
        "version": "0.2.22",
        "entrypoint": "minorminer.find_embedding",
        "implementation": "native-package-persistent-subprocess-hard-deadline",
    }:
        raise ValueError("external tuning backend identity differs")
    seeds = payload["tuning_seeds"]
    if not isinstance(seeds, list) or tuple(seeds) != TUNING_SEEDS:
        raise ValueError("external tuning must use all three registered tuning seeds")
    repetitions = _integer(payload["repetitions"], "tuning repetitions", minimum=1)
    if repetitions != 4:
        raise ValueError("external tuning fixes four repetitions")
    minimum_lineages = _integer(
        payload["minimum_independent_lineages"],
        "minimum independent lineages",
        minimum=128,
    )
    _validate_registry_protocol(payload)

    raw_candidates = payload["candidates"]
    if not isinstance(raw_candidates, list):
        raise ValueError("external tuning candidates must be an array")
    candidates = tuple(ExternalTuningCandidate.from_mapping(item) for item in raw_candidates)
    if len(candidates) != 7 or len({item.candidate_id for item in candidates}) != 7:
        raise ValueError("external tuning requires the exact finite seven-candidate census")
    expected_ids = (
        "stock-default-v1",
        "time-resource-p5-v1",
        "time-resource-p10-v1",
        "time-resource-p20-v1",
        "time-quality-p5-v1",
        "time-quality-p10-v1",
        "time-quality-p20-v1",
    )
    if tuple(item.candidate_id for item in candidates) != expected_ids:
        raise ValueError("external tuning candidate order differs from the fixed v1 registry")
    resource = candidates[1:4]
    quality = candidates[4:]
    if tuple(item.native_parameters["max_no_improvement"] for item in resource) != (
        5,
        10,
        20,
    ) or tuple(item.native_parameters["chainlength_patience"] for item in resource) != (5, 10, 20):
        raise ValueError("external resource patience grid differs")
    if any(
        dict(left.native_parameters) != dict(right.native_parameters)
        for left, right in zip(resource, quality, strict=True)
    ):
        raise ValueError("resource and quality reranking do not share one search grid")

    return ExternalTuningRegistry(
        path=registry_path,
        file_sha256=observed_file_sha,
        record_digest=record_digest,
        registry_id=str(payload["registry_id"]),
        grid_file_sha256=grid_file_sha,
        external_config_digest=config_digest,
        external_config_file_sha256=config_file_sha,
        tuning_seeds=TUNING_SEEDS,
        repetitions=repetitions,
        minimum_independent_lineages=minimum_lineages,
        audit_reads=4096,
        online_wallclock_seconds=_finite(
            config["online_wallclock_seconds"],
            "external online wall-clock cap",
            minimum=1e-12,
        ),
        candidates=candidates,
        evaluation=_frozen_mapping(payload["evaluation"], "tuning evaluation"),
        bootstrap=_frozen_mapping(payload["bootstrap"], "tuning bootstrap"),
        selection=_frozen_mapping(payload["selection"], "tuning selection"),
        failure_denominator=_frozen_mapping(
            payload["failure_denominator"], "tuning failure denominator"
        ),
        work_disclosure=_frozen_mapping(payload["work_disclosure"], "tuning work disclosure"),
    )


_LINEAGE_FIELDS = frozenset(
    {
        "lineage",
        "attempts",
        "utility_sum",
        "valid_returns",
        "online_seconds_total",
        "failure_counts",
        "work_totals",
        "wallclock_noncompliant_attempts",
        "known_work_cap_violations",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "attempts",
        "independent_lineages",
        "utility_sum",
        "unconditional_utility_mean",
        "valid_returns",
        "valid_return_rate",
        "online_seconds_total",
        "mean_online_seconds",
        "failure_counts",
        "work_totals",
        "wallclock_noncompliant_attempts",
        "known_work_cap_violations",
    }
)
_ARTIFACT_FIELDS = frozenset({"path", "sha256", "count"})
_RUN_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "status",
        "partition",
        "test_data_opened",
        "registry_file_sha256",
        "registry_record_digest",
        "grid_file_sha256",
        "external_config_digest",
        "external_config_file_sha256",
        "candidate_index",
        "candidate_id",
        "candidate",
        "candidate_digest",
        "tuning_seed_index",
        "tuning_seed",
        "repetitions",
        "audit_reads",
        "wallclock_cap_seconds",
        *RUNTIME_IDENTITY_FIELDS,
        "runtime_implementation_registry",
        "runtime_implementation_digest",
        "selector",
        "selector_digest",
        "quality_authority",
        "quality_authority_digest",
        "target_access",
        "ground_partition_receipt",
        "population",
        "population_digest",
        "context",
        "context_digest",
        "task_census",
        "task_census_digest",
        "lineage_census_digest",
        "lineage_metrics",
        "artifacts",
        "summary",
        "record_digest",
    }
)


def _safe_relative_path(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or value != path.as_posix():
        raise ValueError(f"{label} must be a canonical relative path")
    return value


def _lineage_row(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("external tuning lineage metric must be an object")
    _exact_keys(value, _LINEAGE_FIELDS, "external tuning lineage metric")
    lineage = value["lineage"]
    if not isinstance(lineage, str) or not lineage:
        raise ValueError("external tuning lineage identity must be nonempty")
    attempts = _integer(value["attempts"], "lineage attempts", minimum=1)
    utility_sum = _finite(value["utility_sum"], "lineage utility sum", minimum=0.0)
    valid_returns = _integer(value["valid_returns"], "lineage valid returns")
    if valid_returns > attempts or utility_sum > valid_returns + 1e-12:
        raise ValueError("lineage utility or validity exceeds its all-attempt denominator")
    online = _finite(value["online_seconds_total"], "lineage online seconds", minimum=0.0)

    raw_failures = value["failure_counts"]
    if not isinstance(raw_failures, Mapping):
        raise ValueError("lineage failure counts must be an object")
    _exact_keys(raw_failures, set(FAILURE_REASONS), "lineage failure counts")
    failures = {
        reason: _integer(raw_failures[reason], f"failure count {reason}")
        for reason in FAILURE_REASONS
    }
    if sum(failures.values()) != attempts or failures["EXTERNAL_VALID_RETURN"] != valid_returns:
        raise ValueError("lineage failure counts omit or duplicate all-attempt outcomes")

    raw_work = value["work_totals"]
    if not isinstance(raw_work, Mapping):
        raise ValueError("lineage work totals must be an object")
    _exact_keys(raw_work, set(WORK_FIELDS), "lineage work totals")
    work: dict[str, int | None] = {}
    for coordinate in WORK_FIELDS:
        raw = raw_work[coordinate]
        if coordinate in UNKNOWN_NATIVE_WORK_FIELDS:
            if raw is not None:
                raise ValueError(
                    f"unavailable native work coordinate {coordinate!r} must remain null"
                )
            work[coordinate] = None
        else:
            work[coordinate] = _integer(raw, f"work total {coordinate}")
    if work["evaluator_reads"] != 4096 * valid_returns:
        raise ValueError("lineage evaluator reads differ from the fresh audit denominator")
    if work["feature_work"] != work["compiler_calls"]:
        raise ValueError("lineage selector feature work must equal its compiler-call count")
    if work["validator_calls"] < valid_returns:
        raise ValueError("lineage validator work omits a valid return")

    noncompliant = _integer(
        value["wallclock_noncompliant_attempts"],
        "wall-clock noncompliant attempts",
    )
    violations = _integer(value["known_work_cap_violations"], "known work-cap violations")
    if noncompliant > attempts or violations > attempts:
        raise ValueError("lineage compliance count exceeds its all-attempt denominator")
    return {
        "lineage": lineage,
        "attempts": attempts,
        "utility_sum": utility_sum,
        "valid_returns": valid_returns,
        "online_seconds_total": online,
        "failure_counts": failures,
        "work_totals": work,
        "wallclock_noncompliant_attempts": noncompliant,
        "known_work_cap_violations": violations,
    }


@dataclass(frozen=True)
class ExternalTuningRun:
    """One authenticated candidate-by-tuning-seed validation report."""

    report: Mapping[str, object]
    report_file_sha256: str
    report_path: str

    @staticmethod
    def summary_from_lineages(lineages: Sequence[object]) -> dict[str, object]:
        if (
            not isinstance(lineages, Sequence)
            or isinstance(lineages, (str, bytes, bytearray))
            or not lineages
        ):
            raise ValueError("external tuning needs nonempty lineage metrics")
        rows = tuple(_lineage_row(row) for row in lineages)
        identities = tuple(row["lineage"] for row in rows)
        if identities != tuple(sorted(identities)) or len(identities) != len(set(identities)):
            raise ValueError("external tuning lineage metrics must be unique and sorted")
        attempts = sum(int(row["attempts"]) for row in rows)
        utility = math.fsum(float(row["utility_sum"]) for row in rows)
        valid = sum(int(row["valid_returns"]) for row in rows)
        online = math.fsum(float(row["online_seconds_total"]) for row in rows)
        failures = {
            reason: sum(int(row["failure_counts"][reason]) for row in rows)  # type: ignore[index]
            for reason in FAILURE_REASONS
        }
        work: dict[str, int | None] = {}
        for coordinate in WORK_FIELDS:
            if coordinate in UNKNOWN_NATIVE_WORK_FIELDS:
                work[coordinate] = None
            else:
                work[coordinate] = sum(
                    int(row["work_totals"][coordinate])
                    for row in rows  # type: ignore[index]
                )
        return {
            "attempts": attempts,
            "independent_lineages": len(rows),
            "utility_sum": utility,
            "unconditional_utility_mean": utility / attempts,
            "valid_returns": valid,
            "valid_return_rate": valid / attempts,
            "online_seconds_total": online,
            "mean_online_seconds": online / attempts,
            "failure_counts": failures,
            "work_totals": work,
            "wallclock_noncompliant_attempts": sum(
                int(row["wallclock_noncompliant_attempts"]) for row in rows
            ),
            "known_work_cap_violations": sum(int(row["known_work_cap_violations"]) for row in rows),
        }

    @classmethod
    def from_mapping(
        cls,
        value: object,
        *,
        report_file_sha256: str,
        report_path: str,
    ) -> ExternalTuningRun:
        if not isinstance(value, Mapping):
            raise ValueError("external tuning report must be an object")
        _exact_keys(value, _RUN_FIELDS, "external tuning report")
        if (
            value["schema"] != EXTERNAL_TUNING_RUN_SCHEMA
            or value["schema_version"] != EXTERNAL_TUNING_RUN_VERSION
            or value["status"] != "complete"
            or value["partition"] != "validation"
            or value["test_data_opened"] is not False
        ):
            raise ValueError("external tuning report is validation-only and cannot open test data")
        _verify_record(value, "external tuning report")
        for label in (
            "registry_file_sha256",
            "registry_record_digest",
            "grid_file_sha256",
            "external_config_digest",
            "external_config_file_sha256",
            "candidate_digest",
            "runtime_implementation_digest",
            "selector_digest",
            "quality_authority_digest",
            "population_digest",
            "context_digest",
            "task_census_digest",
            "lineage_census_digest",
        ):
            if not _is_digest(value[label]):
                raise ValueError(f"external tuning report {label} is not a SHA-256 digest")
        candidate = ExternalTuningCandidate.from_mapping(value["candidate"])
        if value["candidate_id"] != candidate.candidate_id or value["candidate_digest"] != (
            candidate.digest
        ):
            raise ValueError("external tuning report candidate authority differs")
        _integer(value["candidate_index"], "candidate index")
        _integer(value["tuning_seed_index"], "tuning seed index")
        _integer(value["tuning_seed"], "tuning seed")
        repetitions = _integer(value["repetitions"], "tuning repetitions", minimum=1)
        audit_reads = _integer(value["audit_reads"], "audit reads", minimum=1)
        wallclock_cap = _finite(
            value["wallclock_cap_seconds"], "online wall-clock cap", minimum=1e-12
        )
        if audit_reads != 4096:
            raise ValueError("external tuning report changes the fixed audit-read block")

        runtime_platform = _frozen_mapping(value["runtime_platform"], "runtime platform")
        if (
            value["inference_device_type"] not in {"cpu", "cuda", "mps"}
            or not isinstance(value["inference_device_name"], str)
            or not value["inference_device_name"]
            or _integer(value["inference_threads"], "inference threads", minimum=1) != 1
            or value["deterministic"] is not True
        ):
            raise ValueError("external tuning runtime identity is not deterministic")
        runtime_registry = _frozen_mapping(
            value["runtime_implementation_registry"], "runtime implementation registry"
        )
        if content_digest(runtime_registry) != value["runtime_implementation_digest"]:
            raise ValueError("external tuning runtime implementation digest differs")
        selector = _frozen_mapping(value["selector"], "frozen selector")
        if content_digest(selector) != value["selector_digest"]:
            raise ValueError("external tuning selector digest differs")
        quality = _frozen_mapping(value["quality_authority"], "quality authority")
        if content_digest(quality) != value["quality_authority_digest"]:
            raise ValueError("external tuning quality authority digest differs")
        target_access = _frozen_mapping(value["target_access"], "target access")
        ground_partition = _frozen_mapping(
            value["ground_partition_receipt"], "ground partition receipt"
        )
        _verify_record(target_access, "external tuning target access")
        _verify_record(ground_partition, "external tuning ground partition")
        partition_authority = quality.get("evaluation_partition")
        ground_identity = (
            partition_authority.get("ground_partition")
            if isinstance(partition_authority, Mapping)
            else None
        )
        if (
            not isinstance(partition_authority, Mapping)
            or partition_authority.get("name") != "val"
            or target_access.get("record_digest")
            != partition_authority.get("target_access_record_digest")
            or not isinstance(ground_identity, Mapping)
            or ground_partition.get("record_digest") != ground_identity.get("receipt_record_digest")
        ):
            raise ValueError("external tuning target receipts differ from authority")
        context = _frozen_mapping(value["context"], "tuning context")
        if stable_digest(_plain(context)) != value["context_digest"]:
            raise ValueError("external tuning context digest differs")

        raw_census = value["task_census"]
        if not isinstance(raw_census, list) or any(
            not isinstance(row, list)
            or len(row) != 2
            or any(not isinstance(item, str) or not item for item in row)
            for row in raw_census
        ):
            raise ValueError("external tuning task census is malformed")
        census = tuple((row[0], row[1]) for row in raw_census)
        if census != tuple(sorted(census)) or len(census) != len(set(census)):
            raise ValueError("external tuning task census must be unique and sorted")
        if content_digest(raw_census) != value["task_census_digest"]:
            raise ValueError("external tuning task census digest differs")
        lineages = tuple(sorted({lineage for lineage, _ in census}))
        if content_digest(list(lineages)) != value["lineage_census_digest"]:
            raise ValueError("external tuning lineage census digest differs")

        population = _frozen_mapping(value["population"], "complete-system population")
        if stable_digest(_plain(population)) != value["population_digest"]:
            raise ValueError("external tuning population digest differs")
        try:
            # Reuse the complete-system normative parser so tuning cannot accept a weakened
            # population shape that the publication runner itself would reject.
            from isingfold.rl.complete_system import _population_from_payload

            typed_population = _population_from_payload(_plain(population))  # type: ignore[arg-type]
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError("external tuning population identity is invalid") from exc
        if typed_population.as_dict() != _plain(population):
            raise ValueError("external tuning population identity is not canonical")
        expected_instances = population.get("expected_instances")
        if expected_instances != tuple(tuple(row) for row in census):
            # Frozen JSON arrays become tuples recursively.
            raise ValueError("external tuning population and task census differ")
        if (
            population.get("expected_repetitions") != repetitions
            or typed_population.expected_repetitions != repetitions
        ):
            raise ValueError("external tuning population repetition denominator differs")
        if (
            population.get("denominator_scope") != ("all-policy-instances-before-initialization")
            or population.get("includes_initializer_failures") is not True
        ):
            raise ValueError("external tuning population changes the all-attempt denominator")
        strata = population.get("evaluation_strata")
        design = population.get("confirmatory_design")
        if (
            not isinstance(strata, tuple)
            or len(strata) != len(census)
            or not isinstance(design, Mapping)
        ):
            raise ValueError("external tuning population lacks strata or confirmatory design")
        if (
            any(
                not isinstance(row, Mapping)
                or row.get("learning_partition") != "val"
                or (row.get("lineage"), row.get("instance")) != census[index]
                for index, row in enumerate(strata)
            )
            or design.get("learning_partition") != "val"
        ):
            raise ValueError("external tuning population contains non-validation strata")

        raw_lineages = value["lineage_metrics"]
        if not isinstance(raw_lineages, list):
            raise ValueError("external tuning lineage metrics must be an array")
        lineage_rows = tuple(_lineage_row(row) for row in raw_lineages)
        if tuple(row["lineage"] for row in lineage_rows) != lineages:
            raise ValueError("external tuning lineage metrics and census differ")
        tasks_per_lineage = Counter(lineage for lineage, _ in census)
        for row in lineage_rows:
            expected_attempts = tasks_per_lineage[str(row["lineage"])] * repetitions
            if row["attempts"] != expected_attempts:
                raise ValueError("external tuning lineage omits task-by-repetition attempts")
        summary = ExternalTuningRun.summary_from_lineages(lineage_rows)
        raw_summary = value["summary"]
        if not isinstance(raw_summary, Mapping):
            raise ValueError("external tuning summary must be an object")
        _exact_keys(raw_summary, _SUMMARY_FIELDS, "external tuning summary")
        if _plain(raw_summary) != summary:
            raise ValueError("external tuning summary differs from lineage denominators")
        if summary["known_work_cap_violations"] != 0:
            raise ValueError("external tuning report contains a known work-cap violation")

        artifacts = value["artifacts"]
        if not isinstance(artifacts, Mapping):
            raise ValueError("external tuning artifacts must be an object")
        _exact_keys(artifacts, {"receipts", "terminal_evidence", "outcomes"}, "artifacts")
        frozen_artifacts: dict[str, object] = {}
        artifact_paths: set[str] = set()
        for artifact_name in ("receipts", "terminal_evidence", "outcomes"):
            artifact = artifacts[artifact_name]
            if not isinstance(artifact, Mapping):
                raise ValueError(f"external tuning {artifact_name} artifact is malformed")
            _exact_keys(artifact, _ARTIFACT_FIELDS, f"{artifact_name} artifact")
            artifact_path = _safe_relative_path(artifact["path"], f"{artifact_name} artifact path")
            if artifact_path in artifact_paths or not _is_digest(artifact["sha256"]):
                raise ValueError("external tuning artifact paths or SHA-256 pins are invalid")
            artifact_paths.add(artifact_path)
            if artifact["count"] != summary["attempts"]:
                raise ValueError("external tuning artifact count differs from attempts")
            frozen_artifacts[artifact_name] = _freeze(_plain(artifact))

        safe_report_path = _safe_relative_path(report_path, "external tuning report path")
        if not _is_digest(report_file_sha256):
            raise ValueError("external tuning report needs a SHA-256 file identity")
        plain = _plain(value)
        observed_file_sha = hashlib.sha256(canonical_json_bytes(plain) + b"\n").hexdigest()
        if not hmac.compare_digest(observed_file_sha, report_file_sha256):
            raise ValueError("external tuning report file SHA-256 differs")
        frozen = _freeze(plain)
        assert isinstance(frozen, Mapping)
        # Touch validated objects so static analysis catches accidental removal of these checks.
        del (
            runtime_platform,
            runtime_registry,
            selector,
            quality,
            context,
            population,
            frozen_artifacts,
            typed_population,
            wallclock_cap,
        )
        return cls(
            report=frozen,
            report_file_sha256=report_file_sha256,
            report_path=safe_report_path,
        )

    @property
    def candidate_index(self) -> int:
        return int(self.report["candidate_index"])

    @property
    def tuning_seed_index(self) -> int:
        return int(self.report["tuning_seed_index"])

    @property
    def lineage_metrics(self) -> tuple[Mapping[str, object], ...]:
        return self.report["lineage_metrics"]  # type: ignore[return-value]


def external_tuning_lineage_metrics(receipts: Sequence[object]) -> list[dict[str, object]]:
    """Reduce a complete validation cell to equal-lineage all-attempt sufficient statistics."""

    from isingfold.rl.external_pairing import ExternalCompleteSystemReceipt

    materialized = tuple(receipts)
    if not materialized or any(
        not isinstance(receipt, ExternalCompleteSystemReceipt)
        for receipt in materialized
    ):
        raise TypeError("external tuning metrics require typed complete-system receipts")
    typed = materialized  # narrowed by the fail-closed check above
    reference = typed[0]
    binding = reference.tuning_execution
    if binding is None or binding.mode != "validation-candidate":
        raise ValueError("external tuning metrics require a validation-candidate binding")
    expected_pairs = {
        (lineage, instance, repetition)
        for lineage, instance in reference.population.expected_instances
        for repetition in range(reference.population.expected_repetitions)
    }
    observed_pairs = {receipt.pair_key for receipt in typed}
    if len(observed_pairs) != len(typed) or observed_pairs != expected_pairs:
        raise ValueError("external tuning receipts do not cover the exact validation census")
    for receipt in typed:
        if (
            receipt.population != reference.population
            or receipt.backend != reference.backend
            or receipt.selector != reference.selector
            or receipt.config_digest != reference.config_digest
            or receipt.learned_config_digest != reference.learned_config_digest
            or receipt.context_digest != reference.context_digest
            or receipt.quality_authority_digest != reference.quality_authority_digest
            or receipt.runtime_identity_digest != reference.runtime_identity_digest
            or receipt.work_cap != reference.work_cap
            or receipt.training_seed_index != reference.training_seed_index
            or receipt.training_seed != reference.training_seed
            or receipt.tuning_execution != binding
            or receipt.wallclock_cap_seconds != reference.wallclock_cap_seconds
        ):
            raise ValueError("external tuning receipt authorities differ within one cell")
        if receipt.outcome.reason not in FAILURE_REASONS:
            raise ValueError("external tuning receipt has an unregistered terminal reason")

    rows: list[dict[str, object]] = []
    for lineage in sorted({receipt.lineage for receipt in typed}):
        members = tuple(receipt for receipt in typed if receipt.lineage == lineage)
        failures = {
            reason: sum(receipt.outcome.reason == reason for receipt in members)
            for reason in FAILURE_REASONS
        }
        work: dict[str, int | None] = {}
        for coordinate in WORK_FIELDS:
            values = tuple(
                getattr(receipt.observed_work, coordinate) for receipt in members
            )
            if coordinate in UNKNOWN_NATIVE_WORK_FIELDS:
                if any(value is not None for value in values):
                    raise ValueError("external tuning native work must remain unavailable")
                work[coordinate] = None
            else:
                if any(type(value) is not int or value < 0 for value in values):
                    raise ValueError("external tuning known work ledger is malformed")
                work[coordinate] = sum(int(value) for value in values)
        rows.append(
            {
                "lineage": lineage,
                "attempts": len(members),
                "utility_sum": math.fsum(float(receipt.outcome.utility) for receipt in members),
                "valid_returns": sum(bool(receipt.outcome.returned_valid) for receipt in members),
                "online_seconds_total": math.fsum(receipt.online_seconds for receipt in members),
                "failure_counts": failures,
                "work_totals": work,
                "wallclock_noncompliant_attempts": sum(
                    not receipt.wallclock_compliant for receipt in members
                ),
                "known_work_cap_violations": sum(
                    any(value is False for value in receipt.cap_compliance.values())
                    for receipt in members
                ),
            }
        )
    # Reuse the normative row validator before exposing the reduction to report builders.
    return [_lineage_row(row) for row in rows]


def build_external_tuning_report(
    *,
    registry: ExternalTuningRegistry,
    candidate_index: int,
    tuning_seed_index: int,
    receipts: Sequence[object],
    runtime_identity: Mapping[str, object],
    runtime_implementation_registry: Mapping[str, object],
    quality_authority: Mapping[str, object],
    target_access: Mapping[str, object],
    ground_partition_receipt: Mapping[str, object],
    context: Mapping[str, object],
    artifacts: Mapping[str, object],
) -> dict[str, object]:
    """Build and self-validate one canonical candidate-by-seed validation report."""

    from isingfold.rl.external_pairing import ExternalCompleteSystemReceipt

    if not isinstance(registry, ExternalTuningRegistry):
        raise TypeError("external tuning report requires an authenticated registry")
    candidate = registry.candidate_for_index(candidate_index)
    tuning_seed = registry.seed_for_index(tuning_seed_index)
    materialized = tuple(receipts)
    if not materialized or any(
        not isinstance(receipt, ExternalCompleteSystemReceipt) for receipt in materialized
    ):
        raise TypeError("external tuning report requires typed complete-system receipts")
    reference = materialized[0]
    expected_binding = ExternalTuningExecutionBinding.for_validation(registry, candidate_index)
    if (
        reference.tuning_execution != expected_binding
        or reference.training_seed_index != tuning_seed_index
        or reference.training_seed != tuning_seed
        or reference.config_digest != registry.external_config_digest
        or reference.population.expected_repetitions != registry.repetitions
        or reference.wallclock_cap_seconds != registry.online_wallclock_seconds
    ):
        raise ValueError("external tuning receipt differs from its registered cell")
    plain_runtime = _plain(runtime_identity)
    if not isinstance(plain_runtime, dict) or set(plain_runtime) != set(RUNTIME_IDENTITY_FIELDS):
        raise ValueError("external tuning runtime identity fields differ")
    if content_digest(plain_runtime) != reference.runtime_identity_digest:
        raise ValueError("external tuning runtime identity differs from its receipts")
    plain_runtime_registry = _plain(runtime_implementation_registry)
    if not isinstance(plain_runtime_registry, dict) or not plain_runtime_registry:
        raise ValueError("external tuning runtime implementation registry is empty")
    plain_quality = _plain(quality_authority)
    if not isinstance(plain_quality, dict) or not plain_quality:
        raise ValueError("external tuning quality authority is empty")
    if content_digest(plain_quality) != reference.quality_authority_digest:
        raise ValueError("external tuning quality authority differs from its receipts")
    plain_target_access = _plain(target_access)
    plain_ground_partition = _plain(ground_partition_receipt)
    if not isinstance(plain_target_access, dict) or not isinstance(plain_ground_partition, dict):
        raise ValueError("external tuning target receipts must be objects")
    _verify_record(plain_target_access, "external tuning target access")
    _verify_record(plain_ground_partition, "external tuning ground partition")
    plain_context = _plain(context)
    if not isinstance(plain_context, dict) or stable_digest(plain_context) != (
        reference.context_digest
    ):
        raise ValueError("external tuning context differs from its receipts")
    plain_artifacts = _plain(artifacts)
    if not isinstance(plain_artifacts, dict):
        raise ValueError("external tuning artifacts must be an object")

    lineage_metrics = external_tuning_lineage_metrics(materialized)
    population = reference.population.as_dict()
    task_census = [list(identity) for identity in reference.population.expected_instances]
    selector = reference.selector.as_dict()
    payload: dict[str, object] = {
        "schema": EXTERNAL_TUNING_RUN_SCHEMA,
        "schema_version": EXTERNAL_TUNING_RUN_VERSION,
        "status": "complete",
        "partition": "validation",
        "test_data_opened": False,
        "registry_file_sha256": registry.file_sha256,
        "registry_record_digest": registry.record_digest,
        "grid_file_sha256": registry.grid_file_sha256,
        "external_config_digest": registry.external_config_digest,
        "external_config_file_sha256": registry.external_config_file_sha256,
        "candidate_index": candidate_index,
        "candidate_id": candidate.candidate_id,
        "candidate": candidate.as_dict(),
        "candidate_digest": candidate.digest,
        "tuning_seed_index": tuning_seed_index,
        "tuning_seed": tuning_seed,
        "repetitions": registry.repetitions,
        "audit_reads": registry.audit_reads,
        "wallclock_cap_seconds": registry.online_wallclock_seconds,
        **plain_runtime,
        "runtime_implementation_registry": plain_runtime_registry,
        "runtime_implementation_digest": content_digest(plain_runtime_registry),
        "selector": selector,
        "selector_digest": content_digest(selector),
        "quality_authority": plain_quality,
        "quality_authority_digest": content_digest(plain_quality),
        "target_access": plain_target_access,
        "ground_partition_receipt": plain_ground_partition,
        "population": population,
        "population_digest": reference.population.digest,
        "context": plain_context,
        "context_digest": reference.context_digest,
        "task_census": task_census,
        "task_census_digest": content_digest(task_census),
        "lineage_census_digest": content_digest(
            sorted({lineage for lineage, _ in reference.population.expected_instances})
        ),
        "lineage_metrics": lineage_metrics,
        "artifacts": plain_artifacts,
        "summary": ExternalTuningRun.summary_from_lineages(lineage_metrics),
    }
    payload["record_digest"] = content_digest(payload)
    content = canonical_json_bytes(payload) + b"\n"
    ExternalTuningRun.from_mapping(
        payload,
        report_file_sha256=hashlib.sha256(content).hexdigest(),
        report_path="report.json",
    )
    return payload


def load_external_tuning_run(
    path: str | os.PathLike[str], *, expected_file_sha256: str, report_path: str | None = None
) -> ExternalTuningRun:
    """Load one canonical report against an independently supplied file digest."""

    if not _is_digest(expected_file_sha256):
        raise ValueError("external tuning report requires an out-of-band SHA-256")
    source = Path(path)
    content = source.read_bytes()
    observed = hashlib.sha256(content).hexdigest()
    if not hmac.compare_digest(observed, expected_file_sha256):
        raise ValueError("external tuning report differs from its out-of-band SHA-256")
    payload = _strict_json_bytes(content, "external tuning report")
    if canonical_json_bytes(payload) + b"\n" != content:
        raise ValueError("external tuning report is not canonical")
    return ExternalTuningRun.from_mapping(
        payload,
        report_file_sha256=observed,
        report_path=source.name if report_path is None else report_path,
    )


def _sum_work(rows: Sequence[Mapping[str, object]]) -> dict[str, int | None]:
    totals: dict[str, int | None] = {}
    for coordinate in WORK_FIELDS:
        if coordinate in UNKNOWN_NATIVE_WORK_FIELDS:
            totals[coordinate] = None
        else:
            totals[coordinate] = sum(
                int(row["summary"]["work_totals"][coordinate])
                for row in rows  # type: ignore[index]
            )
    return totals


def _runtime_identity_from_report(report: Mapping[str, object]) -> dict[str, object]:
    return {field: _plain(report[field]) for field in RUNTIME_IDENTITY_FIELDS}


def select_external_tuning(
    registry: ExternalTuningRegistry, runs: Sequence[ExternalTuningRun]
) -> dict[str, object]:
    """Select one candidate over the complete three-seed validation census."""

    if not isinstance(registry, ExternalTuningRegistry):
        raise TypeError("external tuning registry has the wrong type")
    expected_cells = len(registry.candidates) * len(registry.tuning_seeds)
    if len(runs) != expected_cells or any(not isinstance(run, ExternalTuningRun) for run in runs):
        raise ValueError("external tuning requires the complete candidate-by-seed census")
    cells = {(run.candidate_index, run.tuning_seed_index): run for run in runs}
    expected_keys = {
        (candidate_index, seed_index)
        for candidate_index in range(len(registry.candidates))
        for seed_index in range(len(registry.tuning_seeds))
    }
    if set(cells) != expected_keys or len(cells) != len(runs):
        raise ValueError("external tuning requires the complete candidate-by-seed census")

    reference = cells[(0, 0)].report
    scientific_authorities = (
        "selector",
        "selector_digest",
        "quality_authority",
        "quality_authority_digest",
        "target_access",
        "ground_partition_receipt",
        "population",
        "population_digest",
        "context",
        "context_digest",
        "task_census",
        "task_census_digest",
        "lineage_census_digest",
        "runtime_implementation_registry",
        "runtime_implementation_digest",
    )
    runtime_by_seed: list[dict[str, object]] = []
    for seed_index, tuning_seed in enumerate(registry.tuning_seeds):
        seed_reference = cells[(0, seed_index)].report
        expected_runtime = _runtime_identity_from_report(seed_reference)
        for candidate_index, candidate in enumerate(registry.candidates):
            run = cells[(candidate_index, seed_index)]
            report = run.report
            if (
                report["registry_file_sha256"] != registry.file_sha256
                or report["registry_record_digest"] != registry.record_digest
                or report["grid_file_sha256"] != registry.grid_file_sha256
                or report["external_config_digest"] != registry.external_config_digest
                or report["external_config_file_sha256"] != registry.external_config_file_sha256
                or report["candidate_index"] != candidate_index
                or report["candidate_id"] != candidate.candidate_id
                or report["candidate"] != _freeze(candidate.as_dict())
                or report["candidate_digest"] != candidate.digest
                or report["tuning_seed_index"] != seed_index
                or report["tuning_seed"] != tuning_seed
                or report["repetitions"] != registry.repetitions
                or report["audit_reads"] != registry.audit_reads
                or report["wallclock_cap_seconds"] != registry.online_wallclock_seconds
            ):
                raise ValueError("external tuning run differs from registry authority")
            if _runtime_identity_from_report(report) != expected_runtime:
                raise ValueError("external tuning runtime identity within tuning seed differs")
            if any(report[field] != reference[field] for field in scientific_authorities):
                raise ValueError("external tuning authority or census differs across runs")
        runtime_by_seed.append(
            {
                "tuning_seed_index": seed_index,
                "tuning_seed": tuning_seed,
                "runtime_identity": expected_runtime,
                "runtime_identity_digest": content_digest(expected_runtime),
            }
        )

    lineages = tuple(row[0] for row in reference["task_census"])  # type: ignore[index]
    unique_lineages = tuple(sorted(set(lineages)))
    if len(unique_lineages) < registry.minimum_independent_lineages:
        raise ValueError("external tuning does not meet its independent-lineage power threshold")
    candidate_aggregates: list[dict[str, object]] = []
    for candidate_index, candidate in enumerate(registry.candidates):
        ordered_runs = [cells[(candidate_index, seed_index)] for seed_index in range(3)]
        per_seed = []
        for run in ordered_runs:
            by_lineage = {str(row["lineage"]): row for row in run.lineage_metrics}
            if tuple(sorted(by_lineage)) != unique_lineages:
                raise ValueError("external tuning candidates do not share immutable lineages")
            per_seed.append(by_lineage)
        utility_matrix = np.asarray(
            [
                [
                    float(rows[lineage]["utility_sum"]) / int(rows[lineage]["attempts"])
                    for lineage in unique_lineages
                ]
                for rows in per_seed
            ],
            dtype=float,
        )
        validity_matrix = np.asarray(
            [
                [
                    int(rows[lineage]["valid_returns"]) / int(rows[lineage]["attempts"])
                    for lineage in unique_lineages
                ]
                for rows in per_seed
            ],
            dtype=float,
        )
        time_matrix = np.asarray(
            [
                [
                    float(rows[lineage]["online_seconds_total"]) / int(rows[lineage]["attempts"])
                    for lineage in unique_lineages
                ]
                for rows in per_seed
            ],
            dtype=float,
        )
        restart_matrix = np.asarray(
            [
                [
                    int(rows[lineage]["work_totals"]["restart_work"])  # type: ignore[index]
                    / int(rows[lineage]["attempts"])
                    for lineage in unique_lineages
                ]
                for rows in per_seed
            ],
            dtype=float,
        )
        bootstrap_low, bootstrap_high = crossed_bootstrap_bounds(
            utility_matrix,
            replicates=int(registry.bootstrap["replicates"]),
            seed=int(registry.bootstrap["seed"]),
            alpha=float(registry.bootstrap["two_sided_alpha"]) / 2.0,
        )
        reports = [run.report for run in ordered_runs]
        attempts = sum(int(report["summary"]["attempts"]) for report in reports)  # type: ignore[index]
        failures = {
            reason: sum(
                int(report["summary"]["failure_counts"][reason])  # type: ignore[index]
                for report in reports
            )
            for reason in FAILURE_REASONS
        }
        aggregate = {
            "candidate_index": candidate_index,
            "candidate_id": candidate.candidate_id,
            "candidate_digest": candidate.digest,
            "attempts": attempts,
            "independent_lineages": len(unique_lineages),
            "unconditional_utility_mean": float(np.mean(utility_matrix)),
            "utility_crossed_bootstrap_95_ci": [bootstrap_low, bootstrap_high],
            "valid_return_rate": float(np.mean(validity_matrix)),
            "mean_online_seconds": float(np.mean(time_matrix)),
            "mean_known_outer_restart_work": float(np.mean(restart_matrix)),
            "wallclock_noncompliant_attempts": sum(
                int(report["summary"]["wallclock_noncompliant_attempts"])  # type: ignore[index]
                for report in reports
            ),
            "known_work_cap_violations": 0,
            "failure_counts": failures,
            "work_totals": _sum_work(reports),
        }
        candidate_aggregates.append(aggregate)

    selected = max(
        candidate_aggregates,
        key=lambda row: (
            float(row["unconditional_utility_mean"]),
            float(row["valid_return_rate"]),
            -float(row["mean_online_seconds"]),
            -float(row["mean_known_outer_restart_work"]),
            -int(row["candidate_index"]),
        ),
    )
    selected_index = int(selected["candidate_index"])
    selected_candidate = registry.candidates[selected_index]
    ordered_runs = [cells[key] for key in sorted(cells)]
    total_attempts = sum(int(run.report["summary"]["attempts"]) for run in ordered_runs)  # type: ignore[index]
    total_failures = {
        reason: sum(
            int(run.report["summary"]["failure_counts"][reason])  # type: ignore[index]
            for run in ordered_runs
        )
        for reason in FAILURE_REASONS
    }
    receipt: dict[str, object] = {
        "schema": EXTERNAL_TUNING_SELECTION_SCHEMA,
        "schema_version": EXTERNAL_TUNING_SELECTION_VERSION,
        "status": "complete",
        "stage": "validation-only-baseline-preregistration",
        "partition": "validation",
        "test_data_opened": False,
        "registry_id": registry.registry_id,
        "registry_file_sha256": registry.file_sha256,
        "registry_record_digest": registry.record_digest,
        "grid_file_sha256": registry.grid_file_sha256,
        "external_config_digest": registry.external_config_digest,
        "external_config_file_sha256": registry.external_config_file_sha256,
        "tuning_seeds": list(registry.tuning_seeds),
        "repetitions": registry.repetitions,
        "audit_reads": registry.audit_reads,
        "wallclock_cap_seconds": registry.online_wallclock_seconds,
        "aggregation": registry.evaluation["aggregation"],
        "selection_rule": registry.selection["primary"],
        "tie_breaks": _plain(registry.selection["tie_breaks"]),
        "seed_selection_forbidden": True,
        "runtime_by_tuning_seed": runtime_by_seed,
        "runtime_implementation_registry": _plain(reference["runtime_implementation_registry"]),
        "runtime_implementation_digest": reference["runtime_implementation_digest"],
        "selector": _plain(reference["selector"]),
        "selector_digest": reference["selector_digest"],
        "quality_authority": _plain(reference["quality_authority"]),
        "quality_authority_digest": reference["quality_authority_digest"],
        "target_access": _plain(reference["target_access"]),
        "ground_partition_receipt": _plain(reference["ground_partition_receipt"]),
        "population": _plain(reference["population"]),
        "population_digest": reference["population_digest"],
        "context": _plain(reference["context"]),
        "context_digest": reference["context_digest"],
        "task_census": _plain(reference["task_census"]),
        "task_census_digest": reference["task_census_digest"],
        "lineage_census_digest": reference["lineage_census_digest"],
        "candidate_aggregates": candidate_aggregates,
        "selected_candidate_index": selected_index,
        "selected_candidate_id": selected_candidate.candidate_id,
        "selected_candidate": selected_candidate.as_dict(),
        "selected_candidate_digest": selected_candidate.digest,
        "source_reports": [
            {
                "candidate_index": run.candidate_index,
                "candidate_id": run.report["candidate_id"],
                "tuning_seed_index": run.tuning_seed_index,
                "tuning_seed": run.report["tuning_seed"],
                "path": run.report_path,
                "file_sha256": run.report_file_sha256,
                "record_digest": run.report["record_digest"],
            }
            for run in ordered_runs
        ],
        "power": {
            "registered_candidates": len(registry.candidates),
            "complete_candidate_seed_cells": expected_cells,
            "tuning_seed_count": len(registry.tuning_seeds),
            "independent_lineages": len(unique_lineages),
            "validation_tasks": len(reference["task_census"]),  # type: ignore[arg-type]
            "repetitions_per_task": registry.repetitions,
            "attempts": total_attempts,
            "bootstrap_replicates": registry.bootstrap["replicates"],
        },
        "failure_denominator": {
            **_plain(registry.failure_denominator),  # type: ignore[arg-type]
            "attempts": total_attempts,
            "observed_counts": total_failures,
        },
        "work_disclosure": {
            **_plain(registry.work_disclosure),  # type: ignore[arg-type]
            "observed_totals": _sum_work([run.report for run in ordered_runs]),
        },
        "time_denominator": {
            "unit": "complete-system-attempt",
            "attempts": total_attempts,
            "wallclock_cap_seconds_per_attempt": registry.online_wallclock_seconds,
            "online_seconds_total": math.fsum(
                float(run.report["summary"]["online_seconds_total"])  # type: ignore[index]
                for run in ordered_runs
            ),
            "wallclock_noncompliant_attempts": sum(
                int(run.report["summary"]["wallclock_noncompliant_attempts"])  # type: ignore[index]
                for run in ordered_runs
            ),
        },
        "test_use_requires_out_of_band_sha256": True,
    }
    receipt["record_digest"] = content_digest(receipt)
    return receipt


def _atomic_write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
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
            raise FileExistsError(path) from None
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def write_external_tuning_selection(
    path: str | os.PathLike[str],
    *,
    registry: ExternalTuningRegistry,
    runs: Sequence[ExternalTuningRun],
) -> tuple[dict[str, object], str]:
    """Atomically create the immutable, self-digested validation selection receipt."""

    receipt = select_external_tuning(registry, runs)
    content = canonical_json_bytes(receipt) + b"\n"
    _atomic_write_new(Path(path), content)
    return receipt, hashlib.sha256(content).hexdigest()


@dataclass(frozen=True)
class FrozenExternalTuningSelection:
    """Authenticated deployment baseline selected before test data are opened."""

    selected_candidate_id: str
    selected_candidate_index: int
    selected_candidate: ExternalTuningCandidate
    receipt_file_sha256: str
    record_digest: str
    registry_id: str
    registry_file_sha256: str
    registry_record_digest: str
    population_digest: str
    source_manifest_sha256: str
    selector_digest: str
    quality_authority_digest: str
    quality_authority: Mapping[str, object]
    target_access: Mapping[str, object]
    ground_partition_receipt: Mapping[str, object]
    context_digest: str
    runtime_implementation_digest: str
    test_data_opened: bool


@dataclass(frozen=True)
class ExternalTuningExecutionBinding:
    """Exact baseline strategy authority carried by every tuned runner receipt."""

    mode: str
    registry_id: str
    registry_file_sha256: str
    registry_record_digest: str
    candidate_index: int
    candidate: ExternalTuningCandidate
    selection_file_sha256: str | None = None
    selection_record_digest: str | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"validation-candidate", "frozen-deployment"}:
            raise ValueError("external tuning execution mode is invalid")
        if not self.registry_id or any(
            not _is_digest(value)
            for value in (self.registry_file_sha256, self.registry_record_digest)
        ):
            raise ValueError("external tuning execution registry authority is invalid")
        _integer(self.candidate_index, "external tuning execution candidate index")
        if not isinstance(self.candidate, ExternalTuningCandidate):
            raise TypeError("external tuning execution candidate has the wrong type")
        if (
            self.registry_id
            != "stock-minorminer-0.2.22-validation-tuning-hybrid-v1"
            or self.registry_file_sha256 != EXTERNAL_TUNING_REGISTRY_FILE_SHA256
            or self.registry_record_digest != EXTERNAL_TUNING_REGISTRY_RECORD_DIGEST
            or not 0 <= self.candidate_index < len(REGISTERED_CANDIDATE_IDENTITIES)
            or REGISTERED_CANDIDATE_IDENTITIES[self.candidate_index]
            != (self.candidate.candidate_id, self.candidate.digest)
        ):
            raise ValueError("external tuning execution differs from the finite v1 registry")
        selection_pins = (self.selection_file_sha256, self.selection_record_digest)
        if self.mode == "validation-candidate":
            if any(value is not None for value in selection_pins):
                raise ValueError("validation tuning cannot claim a frozen selection receipt")
        elif any(not _is_digest(value) for value in selection_pins):
            raise ValueError("deployment tuning requires both authenticated selection pins")

    @classmethod
    def for_validation(
        cls, registry: ExternalTuningRegistry, candidate_index: int
    ) -> ExternalTuningExecutionBinding:
        candidate = registry.candidate_for_index(candidate_index)
        return cls(
            mode="validation-candidate",
            registry_id=registry.registry_id,
            registry_file_sha256=registry.file_sha256,
            registry_record_digest=registry.record_digest,
            candidate_index=candidate_index,
            candidate=candidate,
        )

    @classmethod
    def from_mapping(cls, value: object) -> ExternalTuningExecutionBinding:
        expected = {
            "schema",
            "schema_version",
            "mode",
            "registry_id",
            "registry_file_sha256",
            "registry_record_digest",
            "candidate_index",
            "candidate_id",
            "candidate",
            "candidate_digest",
            "selection_file_sha256",
            "selection_record_digest",
        }
        if not isinstance(value, Mapping):
            raise ValueError("external tuning execution binding must be an object")
        _exact_keys(value, expected, "external tuning execution binding")
        if (
            value["schema"] != "isingfold.external-tuning-execution-binding"
            or value["schema_version"] != 1
        ):
            raise ValueError("unsupported external tuning execution binding schema")
        candidate = ExternalTuningCandidate.from_mapping(value["candidate"])
        if (
            value["candidate_id"] != candidate.candidate_id
            or value["candidate_digest"] != candidate.digest
        ):
            raise ValueError("external tuning execution candidate digest differs")
        return cls(
            mode=value["mode"],  # type: ignore[arg-type]
            registry_id=value["registry_id"],  # type: ignore[arg-type]
            registry_file_sha256=value["registry_file_sha256"],  # type: ignore[arg-type]
            registry_record_digest=value["registry_record_digest"],  # type: ignore[arg-type]
            candidate_index=value["candidate_index"],  # type: ignore[arg-type]
            candidate=candidate,
            selection_file_sha256=value["selection_file_sha256"],  # type: ignore[arg-type]
            selection_record_digest=value["selection_record_digest"],  # type: ignore[arg-type]
        )

    @classmethod
    def for_deployment(
        cls, selection: FrozenExternalTuningSelection
    ) -> ExternalTuningExecutionBinding:
        if not isinstance(selection, FrozenExternalTuningSelection):
            raise TypeError("external tuning deployment selection has the wrong type")
        return cls(
            mode="frozen-deployment",
            registry_id=selection.registry_id,
            registry_file_sha256=selection.registry_file_sha256,
            registry_record_digest=selection.registry_record_digest,
            candidate_index=selection.selected_candidate_index,
            candidate=selection.selected_candidate,
            selection_file_sha256=selection.receipt_file_sha256,
            selection_record_digest=selection.record_digest,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": "isingfold.external-tuning-execution-binding",
            "schema_version": 1,
            "mode": self.mode,
            "registry_id": self.registry_id,
            "registry_file_sha256": self.registry_file_sha256,
            "registry_record_digest": self.registry_record_digest,
            "candidate_index": self.candidate_index,
            "candidate_id": self.candidate.candidate_id,
            "candidate": self.candidate.as_dict(),
            "candidate_digest": self.candidate.digest,
            "selection_file_sha256": self.selection_file_sha256,
            "selection_record_digest": self.selection_record_digest,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.as_dict())


def load_external_tuning_selection(
    path: str | os.PathLike[str],
    *,
    expected_file_sha256: str,
    registry: ExternalTuningRegistry,
) -> FrozenExternalTuningSelection:
    """Authenticate a frozen selection before any publication test target is loaded."""

    if not _is_digest(expected_file_sha256):
        raise ValueError("external tuning selection requires an out-of-band SHA-256")
    content = Path(path).read_bytes()
    observed = hashlib.sha256(content).hexdigest()
    if not hmac.compare_digest(observed, expected_file_sha256):
        raise ValueError("external tuning selection out-of-band SHA-256 differs")
    payload = _strict_json_bytes(content, "external tuning selection")
    if canonical_json_bytes(payload) + b"\n" != content:
        raise ValueError("external tuning selection is not canonical")
    record_digest = _verify_record(payload, "external tuning selection")
    if (
        payload.get("schema") != EXTERNAL_TUNING_SELECTION_SCHEMA
        or payload.get("schema_version") != EXTERNAL_TUNING_SELECTION_VERSION
        or payload.get("status") != "complete"
        or payload.get("stage") != "validation-only-baseline-preregistration"
        or payload.get("partition") != "validation"
        or payload.get("test_data_opened") is not False
        or payload.get("test_use_requires_out_of_band_sha256") is not True
        or payload.get("seed_selection_forbidden") is not True
    ):
        raise ValueError("external tuning selection is not a sealed validation-only receipt")
    if (
        payload.get("registry_id") != registry.registry_id
        or payload.get("registry_file_sha256") != registry.file_sha256
        or payload.get("registry_record_digest") != registry.record_digest
        or payload.get("grid_file_sha256") != registry.grid_file_sha256
        or payload.get("external_config_digest") != registry.external_config_digest
        or payload.get("external_config_file_sha256") != registry.external_config_file_sha256
        or payload.get("tuning_seeds") != list(registry.tuning_seeds)
        or payload.get("repetitions") != registry.repetitions
        or payload.get("audit_reads") != registry.audit_reads
        or payload.get("wallclock_cap_seconds") != registry.online_wallclock_seconds
    ):
        raise ValueError("external tuning selection differs from registry authority")
    index = _integer(payload.get("selected_candidate_index"), "selected candidate index")
    candidate = registry.candidate_for_index(index)
    if (
        payload.get("selected_candidate_id") != candidate.candidate_id
        or payload.get("selected_candidate") != candidate.as_dict()
        or payload.get("selected_candidate_digest") != candidate.digest
    ):
        raise ValueError("external tuning selected candidate differs from its registry")
    sources = payload.get("source_reports")
    aggregates = payload.get("candidate_aggregates")
    power = payload.get("power")
    if (
        not isinstance(sources, list)
        or len(sources) != len(registry.candidates) * len(registry.tuning_seeds)
        or not isinstance(aggregates, list)
        or len(aggregates) != len(registry.candidates)
        or not isinstance(power, Mapping)
        or power.get("independent_lineages", 0) < registry.minimum_independent_lineages
        or power.get("complete_candidate_seed_cells") != len(sources)
    ):
        raise ValueError("external tuning selection has an incomplete power or source census")
    for label in (
        "runtime_implementation_digest",
        "selector_digest",
        "quality_authority_digest",
        "population_digest",
        "context_digest",
        "task_census_digest",
        "lineage_census_digest",
    ):
        if not _is_digest(payload.get(label)):
            raise ValueError(f"external tuning selection {label} is invalid")
    quality_authority = _frozen_mapping(
        payload.get("quality_authority"),
        "external tuning selection quality authority",
    )
    target_access = _frozen_mapping(
        payload.get("target_access"),
        "external tuning selection target access",
    )
    ground_partition_receipt = _frozen_mapping(
        payload.get("ground_partition_receipt"),
        "external tuning selection ground partition",
    )
    if content_digest(quality_authority) != payload.get("quality_authority_digest"):
        raise ValueError("external tuning selection quality authority digest differs")
    _verify_record(target_access, "external tuning selection target access")
    _verify_record(
        ground_partition_receipt,
        "external tuning selection ground partition",
    )
    raw_population = payload.get("population")
    if not isinstance(raw_population, Mapping) or not _is_digest(
        raw_population.get("source_manifest_sha256")
    ):
        raise ValueError("external tuning selection population authority is invalid")
    return FrozenExternalTuningSelection(
        selected_candidate_id=candidate.candidate_id,
        selected_candidate_index=index,
        selected_candidate=candidate,
        receipt_file_sha256=observed,
        record_digest=record_digest,
        registry_file_sha256=registry.file_sha256,
        registry_record_digest=registry.record_digest,
        registry_id=registry.registry_id,
        population_digest=str(payload["population_digest"]),
        source_manifest_sha256=str(raw_population["source_manifest_sha256"]),
        selector_digest=str(payload["selector_digest"]),
        quality_authority_digest=str(payload["quality_authority_digest"]),
        quality_authority=quality_authority,
        target_access=target_access,
        ground_partition_receipt=ground_partition_receipt,
        context_digest=str(payload["context_digest"]),
        runtime_implementation_digest=str(payload["runtime_implementation_digest"]),
        test_data_opened=False,
    )


__all__ = [
    "EXTERNAL_TUNING_REGISTRY_FILE_SHA256",
    "EXTERNAL_TUNING_REGISTRY_RECORD_DIGEST",
    "ExternalTuningCandidate",
    "ExternalTuningExecutionBinding",
    "ExternalTuningRegistry",
    "ExternalTuningRun",
    "FrozenExternalTuningSelection",
    "build_external_tuning_report",
    "external_tuning_lineage_metrics",
    "load_external_tuning_registry",
    "load_external_tuning_run",
    "load_external_tuning_selection",
    "select_external_tuning",
    "write_external_tuning_selection",
]
