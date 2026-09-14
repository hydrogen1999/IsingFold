"""Outcome-blind corpus-design v2 publisher for the IsingFold wire boundary.

The builder turns frozen lineage facts, task provenance identities, and measured
difficulty signals into the registered corpus design, its six evidence artifacts,
and the source-publication index consumed by EmbedBench's IsingFold publication
bridge.  It never reads ground energies, sampler outcomes, or IsingFold runtime code.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Literal

from embedbench.candidate_bank import canonical_json_bytes, content_digest

PRODUCTION_TRAIN_FLOOR = 1024
PRODUCTION_VALIDATION_FLOOR = 512
PRODUCTION_TEST_FLOOR = 1546
PRODUCTION_VALIDATION_TUNING_MINIMUM = 128
PARTITIONS = ("train", "val", "test")
DIFFICULTY_FIELDS = (
    "embedding_difficulty",
    "sampling_difficulty",
    "decision_difficulty",
)
STRATUM_FIELDS = (
    "application_family",
    "problem_origin",
    "host_family",
    "fault_status",
    "distribution_regime",
    "calibration_status",
    *DIFFICULTY_FIELDS,
)
STRATUM_ARTIFACTS = ("authority", "budget", "evidence", "origin", "panel", "protocol")
TRANSFORM_KINDS = frozenset(
    {"identity", "gauge", "relabel", "topology", "fault", "mechanism", "composed"}
)
_HEX = frozenset("0123456789abcdef")
_OUTCOME_DERIVED_METRIC_TOKENS = (
    "audit",
    "ground_energy",
    "label",
    "outcome",
    "p_solve",
    "reference_energy",
    "residual",
    "reward",
    "solution_quality",
)


class CorpusDesignError(ValueError):
    """The supplied facts cannot form the registered scientific corpus design."""


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be non-empty text")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{name} contains a Unicode surrogate")
    return value


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return 0.0 if result == 0.0 else result


def _exact_mapping(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise CorpusDesignError(f"{name} must be an exact object")
    actual = set(value)
    if actual != fields:
        raise CorpusDesignError(
            f"{name} schema fields differ: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return dict(value)


def _record(payload: Mapping[str, object]) -> dict[str, object]:
    return {**payload, "record_digest": content_digest(payload)}


def _raw_record(payload: Mapping[str, object]) -> tuple[dict[str, object], bytes]:
    record = _record(payload)
    return record, canonical_json_bytes(record) + b"\n"


def _file_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class DifficultyRule:
    """One frozen outcome-blind threshold over a measured structural signal."""

    field: str
    metric: str
    hard_if: Literal["greater-than-or-equal", "less-than-or-equal"]
    threshold: float

    def __post_init__(self) -> None:
        if self.field not in DIFFICULTY_FIELDS:
            raise ValueError(f"unsupported difficulty field {self.field!r}")
        metric = _text(self.metric, "difficulty metric")
        if any(token in metric.casefold() for token in _OUTCOME_DERIVED_METRIC_TOKENS):
            raise ValueError("difficulty metric must be outcome-blind")
        object.__setattr__(self, "metric", metric)
        if self.hard_if not in {"greater-than-or-equal", "less-than-or-equal"}:
            raise ValueError("difficulty comparator is unsupported")
        object.__setattr__(self, "threshold", _finite(self.threshold, "difficulty threshold"))

    def as_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "hard_if": self.hard_if,
            "metric": self.metric,
            "threshold": self.threshold,
        }


@dataclass(frozen=True, slots=True)
class MeasurementBudget:
    """Frozen resource cap under which difficulty measurements were collected."""

    measurement_budget_id: str
    decision_evaluations: int
    embedding_attempts: int
    sampler_reads: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "measurement_budget_id",
            _text(self.measurement_budget_id, "measurement budget ID"),
        )
        for field in ("decision_evaluations", "embedding_attempts", "sampler_reads"):
            object.__setattr__(self, field, _positive_int(getattr(self, field), field))


@dataclass(frozen=True, slots=True)
class LineageFact:
    """Publisher-owned facts and measurements for one immutable base lineage."""

    base_lineage_key: str
    application_family: str
    generator_id: str
    generator_implementation_sha256: str
    generator_kind: Literal["application", "synthetic"]
    source_instance_record_digests: tuple[str, ...]
    measurements: tuple[tuple[str, float], ...]

    def __post_init__(self) -> None:
        for field in ("base_lineage_key", "application_family", "generator_id"):
            object.__setattr__(self, field, _text(getattr(self, field), field))
        object.__setattr__(
            self,
            "generator_implementation_sha256",
            _sha256(
                self.generator_implementation_sha256,
                "generator implementation SHA-256",
            ),
        )
        if self.generator_kind not in {"application", "synthetic"}:
            raise ValueError("generator kind must be application or synthetic")
        if type(self.source_instance_record_digests) is not tuple or not (
            self.source_instance_record_digests
        ):
            raise ValueError("source instance record digests must be a nonempty tuple")
        source_digests = tuple(
            _sha256(value, "source instance record digest")
            for value in self.source_instance_record_digests
        )
        if source_digests != tuple(sorted(set(source_digests))):
            raise ValueError("source instance record digests must be a sorted unique tuple")
        object.__setattr__(self, "source_instance_record_digests", source_digests)
        if type(self.measurements) is not tuple or not self.measurements:
            raise ValueError("lineage measurements must be a nonempty tuple")
        normalized: list[tuple[str, float]] = []
        for index, pair in enumerate(self.measurements):
            if type(pair) is not tuple or len(pair) != 2:
                raise ValueError(f"lineage measurement {index} must be a two-item tuple")
            normalized.append(
                (
                    _text(pair[0], f"lineage measurement {index} name"),
                    _finite(pair[1], f"lineage measurement {index} value"),
                )
            )
        if normalized != sorted(normalized) or len({name for name, _ in normalized}) != len(
            normalized
        ):
            raise ValueError("lineage measurements must be sorted and unique by metric")
        object.__setattr__(self, "measurements", tuple(normalized))

    @property
    def problem_origin(self) -> str:
        return "application-derived" if self.generator_kind == "application" else "synthetic"


@dataclass(frozen=True, slots=True)
class TaskFact:
    """Target-free source provenance for one CandidateBank group."""

    group_id: str
    base_parent_lineage: str
    active_topology_identity: Mapping[str, object]
    nominal_topology_identity: Mapping[str, object]
    fault_identity: Mapping[str, object]
    calibration_identity: Mapping[str, object]
    descendant_transform_identity: Mapping[str, object]
    distribution: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _text(self.group_id, "group ID"))
        object.__setattr__(
            self,
            "base_parent_lineage",
            _text(self.base_parent_lineage, "base parent lineage"),
        )


@dataclass(frozen=True, slots=True)
class CorpusDesignProfile:
    """Explicit statistical floor profile; production floors cannot be overridden."""

    kind: Literal["production", "test-only"]
    train_floor: int
    validation_floor: int
    test_floor: int
    validation_tuning_minimum: int
    reason: str | None

    def __post_init__(self) -> None:
        for field in (
            "train_floor",
            "validation_floor",
            "test_floor",
            "validation_tuning_minimum",
        ):
            object.__setattr__(self, field, _positive_int(getattr(self, field), field))
        if self.validation_tuning_minimum > self.validation_floor:
            raise ValueError("validation tuning minimum exceeds the validation floor")
        if self.kind == "production":
            expected = (
                PRODUCTION_TRAIN_FLOOR,
                PRODUCTION_VALIDATION_FLOOR,
                PRODUCTION_TEST_FLOOR,
                PRODUCTION_VALIDATION_TUNING_MINIMUM,
            )
            observed = (
                self.train_floor,
                self.validation_floor,
                self.test_floor,
                self.validation_tuning_minimum,
            )
            if observed != expected:
                raise ValueError("production floors are fixed and cannot be lowered")
            if self.reason is not None:
                raise ValueError("production profile cannot carry a test-profile reason")
        elif self.kind == "test-only":
            object.__setattr__(
                self,
                "reason",
                _text(self.reason, "explicit test-profile reason"),
            )
        else:
            raise ValueError("corpus-design profile kind is unsupported")

    @classmethod
    def production(cls) -> CorpusDesignProfile:
        return cls(
            kind="production",
            train_floor=PRODUCTION_TRAIN_FLOOR,
            validation_floor=PRODUCTION_VALIDATION_FLOOR,
            test_floor=PRODUCTION_TEST_FLOOR,
            validation_tuning_minimum=PRODUCTION_VALIDATION_TUNING_MINIMUM,
            reason=None,
        )

    @classmethod
    def explicit_test(
        cls,
        *,
        train_floor: int,
        validation_floor: int,
        test_floor: int,
        validation_tuning_minimum: int,
        reason: str,
    ) -> CorpusDesignProfile:
        return cls(
            kind="test-only",
            train_floor=train_floor,
            validation_floor=validation_floor,
            test_floor=test_floor,
            validation_tuning_minimum=validation_tuning_minimum,
            reason=reason,
        )


@dataclass(frozen=True, slots=True)
class CorpusDesignReceipt:
    """Content identities and census for an atomically published design bundle."""

    output_directory: Path
    corpus_design_path: Path
    corpus_design_sha256: str
    corpus_design_record_digest: str
    publication_index_path: Path
    publication_index_sha256: str
    publication_index_record_digest: str
    base_lineage_count: int
    task_count: int
    test_only: bool


def _normalize_rules(rules: Sequence[DifficultyRule]) -> tuple[DifficultyRule, ...]:
    if isinstance(rules, (str, bytes)) or not isinstance(rules, Sequence):
        raise TypeError("difficulty_rules must be a sequence")
    normalized = tuple(rules)
    if any(not isinstance(rule, DifficultyRule) for rule in normalized):
        raise TypeError("every difficulty rule must be a DifficultyRule")
    normalized = tuple(sorted(normalized, key=lambda rule: rule.field))
    if tuple(rule.field for rule in normalized) != tuple(sorted(DIFFICULTY_FIELDS)):
        raise CorpusDesignError("difficulty rules must cover exactly three registered fields")
    if len({rule.metric for rule in normalized}) != len(normalized):
        raise CorpusDesignError("difficulty-rule metrics must be unique")
    return normalized


def _normalize_task(task: TaskFact) -> dict[str, object]:
    active = _exact_mapping(
        task.active_topology_identity,
        {"host_artifact_sha256", "host_sha256", "topology"},
        "active topology identity",
    )
    active["host_artifact_sha256"] = _sha256(
        active["host_artifact_sha256"], "active host artifact SHA-256"
    )
    active["host_sha256"] = _sha256(active["host_sha256"], "active host SHA-256")
    active["topology"] = _text(active["topology"], "active topology")

    nominal = _exact_mapping(
        task.nominal_topology_identity,
        {"pristine_host_sha256", "size", "topology"},
        "nominal topology identity",
    )
    nominal["pristine_host_sha256"] = _sha256(
        nominal["pristine_host_sha256"], "pristine host SHA-256"
    )
    nominal["size"] = _positive_int(nominal["size"], "nominal topology size")
    nominal["topology"] = _text(nominal["topology"], "nominal topology")
    if nominal["topology"] != active["topology"]:
        raise CorpusDesignError("nominal and active topology identities disagree")

    transform = _exact_mapping(
        task.descendant_transform_identity,
        {"kinds", "transform_sha256"},
        "descendant transform identity",
    )
    kinds = transform["kinds"]
    if (
        type(kinds) is not list
        or not kinds
        or kinds != sorted(set(kinds))
        or any(type(kind) is not str or kind not in TRANSFORM_KINDS for kind in kinds)
        or ("identity" in kinds and kinds != ["identity"])
    ):
        raise CorpusDesignError("descendant transform kinds are not a registered sorted set")
    transform["transform_sha256"] = _sha256(
        transform["transform_sha256"], "descendant transform SHA-256"
    )

    fault = _exact_mapping(
        task.fault_identity,
        {"fault_mask_sha256", "status"},
        "fault identity",
    )
    fault["fault_mask_sha256"] = _sha256(
        fault["fault_mask_sha256"], "fault mask SHA-256"
    )
    if fault["status"] not in {"none", "faulted"}:
        raise CorpusDesignError("fault status must be none or faulted")
    if fault["fault_mask_sha256"] != active["host_artifact_sha256"]:
        raise CorpusDesignError("fault mask and active host artifact identities disagree")
    if (fault["status"] == "faulted") != ("fault" in kinds):
        raise CorpusDesignError("fault status and descendant transform disagree")

    calibration = _exact_mapping(
        task.calibration_identity,
        {"calibration_sha256", "status"},
        "calibration identity",
    )
    if calibration["status"] == "not_applicable":
        if calibration["calibration_sha256"] is not None:
            raise CorpusDesignError("not-applicable calibration cannot carry a digest")
    elif calibration["status"] == "recorded":
        calibration["calibration_sha256"] = _sha256(
            calibration["calibration_sha256"], "calibration SHA-256"
        )
    else:
        raise CorpusDesignError("calibration status must be explicit")

    distribution = _exact_mapping(
        task.distribution,
        {"learning_partition", "regime", "source_partition", "stratum"},
        "distribution identity",
    )
    if distribution["learning_partition"] not in PARTITIONS:
        raise CorpusDesignError("learning partition is unsupported")
    if distribution["regime"] not in {"iid", "ood"}:
        raise CorpusDesignError("distribution regime must be iid or ood")
    if distribution["regime"] == "ood" and distribution["learning_partition"] != "test":
        raise CorpusDesignError("OOD conditions are allowed only in the sealed test partition")
    distribution["source_partition"] = _text(
        distribution["source_partition"], "source partition"
    )
    distribution["stratum"] = _text(distribution["stratum"], "distribution stratum")

    return {
        "active_topology_identity": active,
        "base_parent_lineage": task.base_parent_lineage,
        "calibration_identity": calibration,
        "descendant_transform_identity": transform,
        "distribution": distribution,
        "fault_identity": fault,
        "group_id": task.group_id,
        "nominal_topology_identity": nominal,
    }


def _difficulty_labels(
    lineages: Sequence[LineageFact],
    rules: Sequence[DifficultyRule],
) -> dict[str, dict[str, str]]:
    expected_metrics = {rule.metric for rule in rules}
    labels: dict[str, dict[str, str]] = {}
    for lineage in lineages:
        metrics = dict(lineage.measurements)
        if set(metrics) != expected_metrics:
            raise CorpusDesignError(
                f"lineage {lineage.base_lineage_key!r} measurements differ from the protocol"
            )
        current: dict[str, str] = {}
        for rule in rules:
            value = metrics[rule.metric]
            hard = (
                value >= rule.threshold
                if rule.hard_if == "greater-than-or-equal"
                else value <= rule.threshold
            )
            current[rule.field] = "hard" if hard else "easy"
        labels[lineage.base_lineage_key] = current
    return labels


def _partition_filter(
    registry: Sequence[Mapping[str, object]], partition: str
) -> dict[str, list[str]]:
    return {
        field: sorted(
            {
                str(row[field])
                for row in registry
                if row["learning_partition"] == partition
            }
        )
        for field in STRATUM_FIELDS
    }


def _power_minimum(
    *,
    alpha: float,
    target_power: float,
    discordance: float,
    separation: float,
    two_sided: bool = False,
) -> int:
    quantile = 1.0 - alpha / 2.0 if two_sided else 1.0 - alpha
    return max(
        1,
        math.ceil(
            discordance
            * (NormalDist().inv_cdf(quantile) + NormalDist().inv_cdf(target_power)) ** 2
            / separation**2
        ),
    )


def _precision_minimum(*, confidence: float, half_width: float, variance: float) -> int:
    quantile = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    return max(1, math.ceil(variance * quantile**2 / half_width**2))


def _statistical_targets(
    registry: Sequence[Mapping[str, object]],
    profile: CorpusDesignProfile,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    test_filter = _partition_filter(registry, "test")
    validation_filter = _partition_filter(registry, "val")
    if profile.kind == "production":
        test_power = {
            "alpha": 0.05,
            "assumed_discordance": 0.1,
            "assumed_true_difference": 0.0,
            "noninferiority_margin": 0.02,
            "power_separation": 0.02,
            "target_power": 0.8,
        }
        validation_power = {
            "alpha": 0.05,
            "assumed_discordance": 0.1,
            "assumed_true_difference": 0.05,
            "noninferiority_margin": 0.02,
            "power_separation": 0.07,
            "target_power": 0.8,
        }
        test_confidence, test_half_width = 0.95, 0.05
        validation_confidence, validation_half_width = 0.95, 0.2
    else:
        test_power = {
            "alpha": 0.49,
            "assumed_discordance": 0.01,
            "assumed_true_difference": 0.0,
            "noninferiority_margin": 0.99,
            "power_separation": 0.99,
            "target_power": 0.51,
        }
        validation_power = {
            "alpha": 0.05,
            "assumed_discordance": 0.01,
            "assumed_true_difference": 0.47,
            "noninferiority_margin": 0.02,
            "power_separation": 0.49,
            "target_power": 0.51,
        }
        test_confidence, test_half_width = 0.51, 0.99
        validation_confidence, validation_half_width = 0.51, 0.99

    test_power_minimum = max(
        profile.test_floor,
        _power_minimum(
            alpha=float(test_power["alpha"]),
            target_power=float(test_power["target_power"]),
            discordance=float(test_power["assumed_discordance"]),
            separation=float(test_power["power_separation"]),
        ),
    )
    validation_power_minimum = max(
        profile.validation_tuning_minimum,
        _power_minimum(
            alpha=float(validation_power["alpha"]),
            target_power=float(validation_power["target_power"]),
            discordance=float(validation_power["assumed_discordance"]),
            separation=float(validation_power["power_separation"]),
        ),
    )
    power_targets = [
        {
            **test_power,
            "alternative": "one-sided-noninferiority",
            "endpoint": "valid-return-noninferiority",
            "filter": test_filter,
            "learning_partition": "test",
            "method": "paired-binary-normal-approximation",
            "minimum_base_lineages": test_power_minimum,
            "target_id": "test-valid-return-noninferiority",
        },
        {
            **validation_power,
            "alternative": "one-sided-noninferiority",
            "endpoint": "valid-return-noninferiority",
            "filter": validation_filter,
            "learning_partition": "val",
            "method": "paired-binary-normal-approximation",
            "minimum_base_lineages": validation_power_minimum,
            "target_id": "validation-valid-return-noninferiority",
        },
    ]

    test_precision_minimum = _precision_minimum(
        confidence=test_confidence,
        half_width=test_half_width,
        variance=1.0,
    )
    validation_precision_minimum = max(
        profile.validation_tuning_minimum,
        _precision_minimum(
            confidence=validation_confidence,
            half_width=validation_half_width,
            variance=1.0,
        ),
    )
    precision_targets = [
        {
            "confidence_level": test_confidence,
            "endpoint": "learned-minus-stock-unconditional-if-q3-s0",
            "filter": test_filter,
            "half_width": test_half_width,
            "learning_partition": "test",
            "method": "bounded-paired-difference-worst-case-normal",
            "minimum_base_lineages": test_precision_minimum,
            "outcome_bounds": [-1.0, 1.0],
            "target_id": "test-learned-minus-stock-if-q3-s0-precision",
            "variance_bound": 1.0,
        },
        {
            "confidence_level": validation_confidence,
            "endpoint": "learned-minus-stock-unconditional-if-q3-s0",
            "filter": validation_filter,
            "half_width": validation_half_width,
            "learning_partition": "val",
            "method": "bounded-paired-difference-worst-case-normal",
            "minimum_base_lineages": validation_precision_minimum,
            "outcome_bounds": [-1.0, 1.0],
            "target_id": "validation-paired-utility-precision",
            "variance_bound": 1.0,
        },
    ]
    return power_targets, precision_targets


def _stratum_quotas(registry: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    members: dict[tuple[str, ...], set[str]] = {}
    for row in registry:
        key = (str(row["learning_partition"]), *(str(row[field]) for field in STRATUM_FIELDS))
        members.setdefault(key, set()).add(str(row["base_lineage_key"]))
    quotas: list[dict[str, object]] = []
    for key in sorted(members):
        partition = key[0]
        design_filter = {
            field: [key[index + 1]] for index, field in enumerate(STRATUM_FIELDS)
        }
        quota_digest = content_digest(
            {"filter": design_filter, "learning_partition": partition}
        )[:16]
        quotas.append(
            {
                "filter": design_filter,
                "learning_partition": partition,
                "minimum_base_lineages": len(members[key]),
                "quota_id": f"quota-{partition}-{quota_digest}",
            }
        )
    quotas.sort(key=lambda row: str(row["quota_id"]))
    if len({row["quota_id"] for row in quotas}) != len(quotas):
        raise CorpusDesignError("stratum quota identity collision")
    return quotas


def _write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def build_corpus_design_v2(
    *,
    lineages: Sequence[LineageFact],
    tasks: Sequence[TaskFact],
    output_directory: str | os.PathLike[str],
    publisher_id: str,
    source_release_id: str,
    source_release_manifest_sha256: str,
    split_manifest_sha256: str,
    measurement_budget: MeasurementBudget,
    difficulty_rules: Sequence[DifficultyRule],
    profile: CorpusDesignProfile | None = None,
) -> CorpusDesignReceipt:
    """Atomically publish one registered design bundle from target-free frozen facts."""

    destination = Path(output_directory)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"corpus-design output already exists: {destination}")
    publisher = _text(publisher_id, "publisher ID")
    release_id = _text(source_release_id, "source release ID")
    release_sha = _sha256(
        source_release_manifest_sha256, "source release manifest SHA-256"
    )
    split_sha = _sha256(split_manifest_sha256, "split manifest SHA-256")
    if not isinstance(measurement_budget, MeasurementBudget):
        raise TypeError("measurement_budget must be a MeasurementBudget")
    selected_profile = CorpusDesignProfile.production() if profile is None else profile
    if not isinstance(selected_profile, CorpusDesignProfile):
        raise TypeError("profile must be a CorpusDesignProfile")
    rules = _normalize_rules(difficulty_rules)

    lineage_rows = tuple(lineages)
    task_rows = tuple(tasks)
    if not lineage_rows or any(not isinstance(row, LineageFact) for row in lineage_rows):
        raise TypeError("lineages must be a nonempty sequence of LineageFact")
    if not task_rows or any(not isinstance(row, TaskFact) for row in task_rows):
        raise TypeError("tasks must be a nonempty sequence of TaskFact")
    lineage_by_id = {row.base_lineage_key: row for row in lineage_rows}
    if len(lineage_by_id) != len(lineage_rows):
        raise CorpusDesignError("lineage facts repeat a base lineage")
    normalized_tasks = [_normalize_task(task) for task in task_rows]
    normalized_tasks.sort(key=lambda row: str(row["group_id"]))
    if len({row["group_id"] for row in normalized_tasks}) != len(normalized_tasks):
        raise CorpusDesignError("task facts repeat a CandidateBank group")
    unknown_lineages = {
        str(row["base_parent_lineage"])
        for row in normalized_tasks
        if row["base_parent_lineage"] not in lineage_by_id
    }
    if unknown_lineages:
        raise CorpusDesignError(
            f"task facts reference unknown lineages: {sorted(unknown_lineages)}"
        )
    observed_lineages = {str(row["base_parent_lineage"]) for row in normalized_tasks}
    if observed_lineages != set(lineage_by_id):
        raise CorpusDesignError("every lineage fact must have at least one task fact")

    lineage_partition: dict[str, str] = {}
    transform_parent: dict[str, str] = {}
    for task in normalized_tasks:
        base = str(task["base_parent_lineage"])
        partition = str(task["distribution"]["learning_partition"])
        if lineage_partition.setdefault(base, partition) != partition:
            raise CorpusDesignError("one base lineage appears in multiple learning partitions")
        transform_sha = str(task["descendant_transform_identity"]["transform_sha256"])
        if transform_parent.setdefault(transform_sha, base) != base:
            raise CorpusDesignError("one transform identity maps to multiple base lineages")

    labels = _difficulty_labels(lineage_rows, rules)
    registry_by_condition: dict[tuple[object, ...], dict[str, object]] = {}
    for task in normalized_tasks:
        base = str(task["base_parent_lineage"])
        lineage = lineage_by_id[base]
        distribution = task["distribution"]
        active = task["active_topology_identity"]
        fault = task["fault_identity"]
        calibration = task["calibration_identity"]
        row: dict[str, object] = {
            "application_family": lineage.application_family,
            "base_lineage_key": base,
            "calibration_sha256": calibration["calibration_sha256"],
            "calibration_status": calibration["status"],
            "decision_difficulty": labels[base]["decision_difficulty"],
            "distribution_regime": distribution["regime"],
            "embedding_difficulty": labels[base]["embedding_difficulty"],
            "fault_status": fault["status"],
            "host_family": active["topology"],
            "learning_partition": distribution["learning_partition"],
            "problem_origin": lineage.problem_origin,
            "sampling_difficulty": labels[base]["sampling_difficulty"],
        }
        condition = (
            base,
            row["learning_partition"],
            row["application_family"],
            row["host_family"],
            row["fault_status"],
            row["distribution_regime"],
            row["calibration_status"],
            row["calibration_sha256"],
        )
        previous = registry_by_condition.setdefault(condition, row)
        if previous != row:
            raise CorpusDesignError("one structural condition has conflicting lineage facts")
    registry = list(registry_by_condition.values())
    registry.sort(
        key=lambda row: (
            str(row["base_lineage_key"]),
            str(row["learning_partition"]),
            *(str(row[field]) for field in STRATUM_FIELDS),
            "" if row["calibration_sha256"] is None else str(row["calibration_sha256"]),
        )
    )

    partition_lineages = {
        partition: sorted(
            base
            for base, observed_partition in lineage_partition.items()
            if observed_partition == partition
        )
        for partition in PARTITIONS
    }
    floors = {
        "train": selected_profile.train_floor,
        "val": selected_profile.validation_floor,
        "test": selected_profile.test_floor,
    }
    for partition in PARTITIONS:
        actual = len(partition_lineages[partition])
        if actual < floors[partition]:
            raise CorpusDesignError(
                f"{partition} has {actual} base lineages but requires {floors[partition]}"
            )

    host_families = {str(row["host_family"]) for row in registry}
    if len(host_families) < 2:
        raise CorpusDesignError("scientific design requires at least two host families")
    origins = {row.problem_origin for row in lineage_rows}
    if origins != {"application-derived", "synthetic"}:
        raise CorpusDesignError("scientific design requires both problem origins")
    origins_by_family: dict[str, set[str]] = {}
    for lineage in lineage_rows:
        origins_by_family.setdefault(lineage.application_family, set()).add(
            lineage.problem_origin
        )
    if any(len(values) != 1 for values in origins_by_family.values()):
        raise CorpusDesignError("one application family maps to multiple problem origins")
    if not any(row["fault_status"] == "faulted" for row in registry):
        raise CorpusDesignError("scientific design requires faulted coverage")
    for field in DIFFICULTY_FIELDS:
        if {str(row[field]) for row in registry} != {"easy", "hard"}:
            raise CorpusDesignError(f"{field} requires both easy and hard coverage")
    if not any(row["distribution_regime"] == "ood" for row in registry):
        raise CorpusDesignError("scientific design requires sealed-test OOD coverage")

    axes = {
        field: sorted({str(row[field]) for row in registry}) for field in STRATUM_FIELDS
    }
    quotas = _stratum_quotas(registry)
    power_targets, precision_targets = _statistical_targets(registry, selected_profile)

    budget_payload: dict[str, object] = {
        "limits": {
            "decision_evaluations": measurement_budget.decision_evaluations,
            "embedding_attempts": measurement_budget.embedding_attempts,
            "sampler_reads": measurement_budget.sampler_reads,
        },
        "measurement_budget_id": measurement_budget.measurement_budget_id,
        "schema": "embedbench.difficulty-budget",
        "schema_version": 1,
    }
    budget_record, budget_raw = _raw_record(budget_payload)
    protocol_payload: dict[str, object] = {
        "outcome_blind": True,
        "rules": [rule.as_dict() for rule in rules],
        "schema": "embedbench.difficulty-protocol",
        "schema_version": 1,
    }
    protocol_record, protocol_raw = _raw_record(protocol_payload)
    panel_payload: dict[str, object] = {
        "base_lineages": sorted(lineage_by_id),
        "sampling_frame": "all-registered-lineages",
        "schema": "embedbench.difficulty-panel",
        "schema_version": 1,
        "selection_outcome_blind": True,
    }
    panel_record, panel_raw = _raw_record(panel_payload)
    measurement_rows = []
    for lineage in sorted(lineage_rows, key=lambda row: row.base_lineage_key):
        measurement_payload: dict[str, object] = {
            "base_lineage_key": lineage.base_lineage_key,
            "metrics": [[name, value] for name, value in lineage.measurements],
        }
        measurement_rows.append(_record(measurement_payload))
    evidence_payload: dict[str, object] = {
        "budget_record_digest": budget_record["record_digest"],
        "measurements": measurement_rows,
        "panel_record_digest": panel_record["record_digest"],
        "protocol_record_digest": protocol_record["record_digest"],
        "schema": "embedbench.difficulty-evidence",
        "schema_version": 1,
    }
    evidence_record, evidence_raw = _raw_record(evidence_payload)
    origin_rows = []
    for lineage in sorted(lineage_rows, key=lambda row: row.base_lineage_key):
        origin_payload: dict[str, object] = {
            "application_family": lineage.application_family,
            "base_lineage_key": lineage.base_lineage_key,
            "generator": {
                "generator_id": lineage.generator_id,
                "implementation_sha256": lineage.generator_implementation_sha256,
                "kind": lineage.generator_kind,
            },
            "source_instance_record_digests": list(lineage.source_instance_record_digests),
        }
        origin_rows.append(_record(origin_payload))
    origin_payload: dict[str, object] = {
        "publisher_id": publisher,
        "records": origin_rows,
        "schema": "embedbench.origin-provenance",
        "schema_version": 1,
        "source_release_id": release_id,
        "source_release_manifest_sha256": release_sha,
    }
    origin_record, origin_raw = _raw_record(origin_payload)

    preliminary_records = {
        "budget": (budget_record, budget_raw),
        "evidence": (evidence_record, evidence_raw),
        "origin": (origin_record, origin_raw),
        "panel": (panel_record, panel_raw),
        "protocol": (protocol_record, protocol_raw),
    }
    authority_payload: dict[str, object] = {
        "artifacts": {
            f"{name}_{suffix}": (
                preliminary_records[name][0]["record_digest"]
                if suffix == "record_digest"
                else _file_sha256(preliminary_records[name][1])
            )
            for name in sorted(preliminary_records)
            for suffix in ("record_digest", "sha256")
        },
        "publisher_id": publisher,
        "schema": "embedbench.stratum-publisher-authority",
        "schema_version": 1,
        "source_release_id": release_id,
        "source_release_manifest_sha256": release_sha,
        "statement": "publisher-attests-origin-and-outcome-blind-difficulty-evidence",
    }
    authority_record, authority_raw = _raw_record(authority_payload)
    artifact_records = {
        **preliminary_records,
        "authority": (authority_record, authority_raw),
    }
    artifact_descriptors = {
        name: {
            "path": f"strata/{name}.json",
            "sha256": _file_sha256(artifact_records[name][1]),
        }
        for name in STRATUM_ARTIFACTS
    }
    design_payload: dict[str, object] = {
        "axis_values": axes,
        "corpus_design_version": "if-core-v2",
        "difficulty_calibration": {
            **artifact_descriptors,
            "outcome_blind": True,
            "publisher_id": publisher,
        },
        "independent_unit": "immutable-base-lineage",
        "lineage_registry": registry,
        "minimum_partition_base_lineages": {
            "test": selected_profile.test_floor,
            "train": selected_profile.train_floor,
            "val": selected_profile.validation_floor,
        },
        "partition_quotas": {
            partition: len(partition_lineages[partition]) for partition in ("test", "train", "val")
        },
        "power_targets": power_targets,
        "precision_targets": precision_targets,
        "schema": "isingfold.corpus-design",
        "schema_version": 2,
        "stratum_quotas": quotas,
    }
    design_record, design_raw = _raw_record(design_payload)
    publication_payload: dict[str, object] = {
        "corpus_design": {
            "path": "corpus_design_v2.json",
            "sha256": _file_sha256(design_raw),
        },
        "schema": "embedbench.isingfold-publication-index",
        "schema_version": 1,
        "source_release_id": release_id,
        "source_release_manifest_sha256": release_sha,
        "split_manifest_sha256": split_sha,
        "strata": artifact_descriptors,
        "tasks": normalized_tasks,
    }
    publication_record, publication_raw = _raw_record(publication_payload)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.design-", dir=destination.parent)
    )
    try:
        for name in ("budget", "evidence", "origin", "panel", "protocol", "authority"):
            _write_new(temporary / "strata" / f"{name}.json", artifact_records[name][1])
        _write_new(temporary / "corpus_design_v2.json", design_raw)
        # The publication index commits the complete design tree and is written last.
        _write_new(temporary / "publication-index.json", publication_raw)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"corpus-design output already exists: {destination}")
        os.replace(temporary, destination)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    root = destination.resolve(strict=True)
    return CorpusDesignReceipt(
        output_directory=root,
        corpus_design_path=root / "corpus_design_v2.json",
        corpus_design_sha256=_file_sha256(design_raw),
        corpus_design_record_digest=str(design_record["record_digest"]),
        publication_index_path=root / "publication-index.json",
        publication_index_sha256=_file_sha256(publication_raw),
        publication_index_record_digest=str(publication_record["record_digest"]),
        base_lineage_count=len(lineage_by_id),
        task_count=len(normalized_tasks),
        test_only=selected_profile.kind == "test-only",
    )


__all__ = [
    "CorpusDesignError",
    "CorpusDesignProfile",
    "CorpusDesignReceipt",
    "DifficultyRule",
    "LineageFact",
    "MeasurementBudget",
    "PRODUCTION_TEST_FLOOR",
    "PRODUCTION_TRAIN_FLOOR",
    "PRODUCTION_VALIDATION_FLOOR",
    "PRODUCTION_VALIDATION_TUNING_MINIMUM",
    "TaskFact",
    "build_corpus_design_v2",
]
