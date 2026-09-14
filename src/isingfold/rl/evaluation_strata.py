"""Authenticated scientific strata and confirmatory evaluation contracts.

The prepared scientific corpus already authenticates the prospective corpus design. This module
projects that design onto the exact pre-initialization policy-instance denominator used by
complete-system evaluation.  The projection is outcome blind: size bins depend only on the
registered nominal topology size, and every other coordinate comes from the authenticated
prepared provenance and condition registry.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from types import MappingProxyType
from isingfold.rl.contracts import stable_digest

EVALUATION_STRATUM_SCHEMA = "isingfold.evaluation-stratum"
EVALUATION_STRATUM_VERSION = 1
CONFIRMATORY_DESIGN_SCHEMA = "isingfold.confirmatory-evaluation-design"
CONFIRMATORY_DESIGN_VERSION = 1
SIZE_BIN_PROTOCOL = "nominal-topology-size-dyadic-v1"
SIZE_BIN_BOUNDARY_CONVENTION = "inclusive-[2^k,2^(k+1)-1]-zero-padded-v1"
INDEPENDENT_UNIT = "immutable-base-lineage"
MULTIPLICITY_METHOD = "prespecified-alpha-spending-v1"
FAMILYWISE_ALPHA = 0.05
REGISTERED_NONINFERIORITY_MARGIN = 0.02
MINIMUM_INFORMATIVE_DISCORDANT_LINEAGES = 2

STRATUM_AXES = (
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

_SHA256_LENGTH = 64


def _is_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != _SHA256_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _digest(value: object, *, label: str) -> str:
    if not _is_digest(value):
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value  # type: ignore[return-value]


def _probability(value: object, *, label: str, upper: float = 1.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < upper:
        raise ValueError(f"{label} must lie strictly between zero and {upper}")
    return result


def _exact_keys(payload: Mapping[str, object], expected: set[str], *, label: str) -> None:
    if set(payload) != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - set(payload))}, "
            f"unknown={sorted(set(payload) - expected)}"
        )


def nominal_size_bin(nominal_size: int) -> str:
    """Return the deterministic dyadic bin containing a positive nominal host size."""

    if isinstance(nominal_size, bool) or not isinstance(nominal_size, int) or nominal_size <= 0:
        raise ValueError("nominal topology size must be a positive integer")
    lower = 1 << (nominal_size.bit_length() - 1)
    upper = 2 * lower - 1
    return f"{lower:06d}-{upper:06d}"


@dataclass(frozen=True)
class EvaluationStratum:
    """Outcome-blind scientific stratum for one sealed policy instance."""

    lineage: str
    instance: str
    learning_partition: str
    application_family: str
    problem_origin: str
    host_family: str
    fault_status: str
    distribution_regime: str
    calibration_status: str
    calibration_sha256: str | None
    embedding_difficulty: str
    sampling_difficulty: str
    decision_difficulty: str
    nominal_size: int
    source_registry_row_digest: str
    source_provenance_record_digests: tuple[str, ...]
    size_bin: str = field(init=False)
    size_bin_protocol: str = field(init=False, default=SIZE_BIN_PROTOCOL)
    size_bin_boundary_convention: str = field(
        init=False, default=SIZE_BIN_BOUNDARY_CONVENTION
    )

    def __post_init__(self) -> None:
        for name in (
            "lineage",
            "instance",
            "application_family",
            "host_family",
            "embedding_difficulty",
            "sampling_difficulty",
            "decision_difficulty",
        ):
            _text(getattr(self, name), label=f"evaluation stratum {name}")
        if self.learning_partition not in {"train", "val", "test"}:
            raise ValueError("evaluation stratum learning partition is unsupported")
        if self.problem_origin not in {"application-derived", "synthetic"}:
            raise ValueError("evaluation stratum problem origin is unsupported")
        if self.fault_status not in {"none", "faulted"}:
            raise ValueError("evaluation stratum fault status is unsupported")
        if self.distribution_regime not in {"iid", "ood"}:
            raise ValueError("evaluation stratum distribution regime is unsupported")
        if self.calibration_status not in {"not_applicable", "recorded"}:
            raise ValueError("evaluation stratum calibration status is unsupported")
        if self.calibration_status == "recorded":
            _digest(self.calibration_sha256, label="recorded calibration identity")
        elif self.calibration_sha256 is not None:
            raise ValueError("not-applicable calibration cannot carry a calibration digest")
        _digest(self.source_registry_row_digest, label="source registry row identity")
        if (
            not isinstance(self.source_provenance_record_digests, tuple)
            or not self.source_provenance_record_digests
            or self.source_provenance_record_digests
            != tuple(sorted(set(self.source_provenance_record_digests)))
            or any(not _is_digest(value) for value in self.source_provenance_record_digests)
        ):
            raise ValueError(
                "source provenance record identities must be a nonempty canonical digest tuple"
            )
        object.__setattr__(self, "size_bin", nominal_size_bin(self.nominal_size))

    @property
    def identity(self) -> tuple[str, str]:
        return (self.lineage, self.instance)

    def _payload(self) -> dict[str, object]:
        return {
            "schema": EVALUATION_STRATUM_SCHEMA,
            "schema_version": EVALUATION_STRATUM_VERSION,
            "lineage": self.lineage,
            "instance": self.instance,
            "learning_partition": self.learning_partition,
            "application_family": self.application_family,
            "problem_origin": self.problem_origin,
            "host_family": self.host_family,
            "fault_status": self.fault_status,
            "distribution_regime": self.distribution_regime,
            "calibration_status": self.calibration_status,
            "calibration_sha256": self.calibration_sha256,
            "embedding_difficulty": self.embedding_difficulty,
            "sampling_difficulty": self.sampling_difficulty,
            "decision_difficulty": self.decision_difficulty,
            "nominal_size": self.nominal_size,
            "size_bin": self.size_bin,
            "size_bin_protocol": self.size_bin_protocol,
            "size_bin_boundary_convention": self.size_bin_boundary_convention,
            "source_registry_row_digest": self.source_registry_row_digest,
            "source_provenance_record_digests": list(
                self.source_provenance_record_digests
            ),
        }

    @property
    def record_digest(self) -> str:
        return stable_digest(self._payload())

    def as_dict(self) -> dict[str, object]:
        return {**self._payload(), "record_digest": self.record_digest}

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> EvaluationStratum:
        expected = {
            "schema",
            "schema_version",
            "lineage",
            "instance",
            "learning_partition",
            *STRATUM_AXES,
            "calibration_sha256",
            "nominal_size",
            "size_bin",
            "size_bin_protocol",
            "size_bin_boundary_convention",
            "source_registry_row_digest",
            "source_provenance_record_digests",
            "record_digest",
        }
        _exact_keys(value, expected, label="evaluation stratum")
        if (
            value["schema"] != EVALUATION_STRATUM_SCHEMA
            or value["schema_version"] != EVALUATION_STRATUM_VERSION
        ):
            raise ValueError("unsupported evaluation stratum schema")
        raw_provenance = value["source_provenance_record_digests"]
        if not isinstance(raw_provenance, list):
            raise ValueError("evaluation stratum provenance identities must be a list")
        try:
            result = cls(
                lineage=value["lineage"],  # type: ignore[arg-type]
                instance=value["instance"],  # type: ignore[arg-type]
                learning_partition=value["learning_partition"],  # type: ignore[arg-type]
                application_family=value["application_family"],  # type: ignore[arg-type]
                problem_origin=value["problem_origin"],  # type: ignore[arg-type]
                host_family=value["host_family"],  # type: ignore[arg-type]
                fault_status=value["fault_status"],  # type: ignore[arg-type]
                distribution_regime=value["distribution_regime"],  # type: ignore[arg-type]
                calibration_status=value["calibration_status"],  # type: ignore[arg-type]
                calibration_sha256=value["calibration_sha256"],  # type: ignore[arg-type]
                embedding_difficulty=value["embedding_difficulty"],  # type: ignore[arg-type]
                sampling_difficulty=value["sampling_difficulty"],  # type: ignore[arg-type]
                decision_difficulty=value["decision_difficulty"],  # type: ignore[arg-type]
                nominal_size=value["nominal_size"],  # type: ignore[arg-type]
                source_registry_row_digest=value["source_registry_row_digest"],  # type: ignore[arg-type]
                source_provenance_record_digests=tuple(raw_provenance),  # type: ignore[arg-type]
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("evaluation stratum is semantically invalid") from exc
        if value["size_bin_protocol"] != SIZE_BIN_PROTOCOL:
            raise ValueError("evaluation stratum size bin protocol is unsupported")
        if value["size_bin_boundary_convention"] != SIZE_BIN_BOUNDARY_CONVENTION:
            raise ValueError("evaluation stratum size bin boundary convention is unsupported")
        if value["size_bin"] != result.size_bin:
            raise ValueError("evaluation stratum size bin disagrees with nominal size")
        if value["record_digest"] != result.record_digest:
            raise ValueError("evaluation stratum record digest is invalid")
        return result


@dataclass(frozen=True)
class EvaluationStratumFilter:
    """Exact registered filter over the eight prepared scientific axes."""

    application_family: tuple[str, ...]
    problem_origin: tuple[str, ...]
    host_family: tuple[str, ...]
    fault_status: tuple[str, ...]
    distribution_regime: tuple[str, ...]
    calibration_status: tuple[str, ...]
    embedding_difficulty: tuple[str, ...]
    sampling_difficulty: tuple[str, ...]
    decision_difficulty: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in STRATUM_AXES:
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or not values
                or values != tuple(sorted(set(values)))
                or any(not isinstance(value, str) or not value for value in values)
            ):
                raise ValueError(f"registered filter {name} must be canonical and nonempty")
        if any(
            value not in {"application-derived", "synthetic"}
            for value in self.problem_origin
        ):
            raise ValueError("registered filter contains an unsupported problem origin")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> EvaluationStratumFilter:
        _exact_keys(value, set(STRATUM_AXES), label="registered stratum filter")
        fields: dict[str, tuple[str, ...]] = {}
        for name in STRATUM_AXES:
            raw = value[name]
            if not isinstance(raw, list):
                raise ValueError(f"registered filter {name} must be a list")
            fields[name] = tuple(raw)  # type: ignore[arg-type]
        return cls(**fields)

    def as_dict(self) -> dict[str, list[str]]:
        return {name: list(getattr(self, name)) for name in STRATUM_AXES}

    def matches(self, stratum: EvaluationStratum) -> bool:
        return all(getattr(stratum, name) in getattr(self, name) for name in STRATUM_AXES)


@dataclass(frozen=True)
class PowerTarget:
    """One exact prepared-corpus power target used by confirmatory paired inference."""

    target_id: str
    endpoint: str
    alternative: str
    alpha: float
    target_power: float
    assumed_discordance: float
    assumed_true_difference: float
    noninferiority_margin: float
    power_separation: float
    design_filter: EvaluationStratumFilter
    learning_partition: str
    method: str
    minimum_base_lineages: int

    def __post_init__(self) -> None:
        _text(self.target_id, label="power target identity")
        if self.endpoint != "valid-return-noninferiority":
            raise ValueError("unsupported confirmatory power target endpoint")
        if self.alternative != "one-sided-noninferiority":
            raise ValueError("valid-return noninferiority must be one sided")
        if self.method != "paired-binary-normal-approximation":
            raise ValueError("unsupported confirmatory power target method")
        if self.learning_partition not in {"val", "test"}:
            raise ValueError("confirmatory power target must address validation or test")
        alpha = _probability(self.alpha, label="power target alpha", upper=0.5)
        target_power = _probability(self.target_power, label="power target power")
        if target_power <= 0.5:
            raise ValueError("power target power must exceed one half")
        discordance = _probability(
            self.assumed_discordance, label="power target assumed discordance"
        )
        if (
            isinstance(self.assumed_true_difference, bool)
            or not isinstance(self.assumed_true_difference, (int, float))
            or not math.isfinite(float(self.assumed_true_difference))
            or not -1.0 < float(self.assumed_true_difference) < 1.0
        ):
            raise ValueError("power target assumed true difference must lie between -1 and 1")
        if self.noninferiority_margin != REGISTERED_NONINFERIORITY_MARGIN:
            raise ValueError("power target does not use the registered noninferiority margin")
        separation = _probability(
            self.power_separation, label="power target null-boundary separation"
        )
        if not math.isclose(
            separation,
            float(self.assumed_true_difference) + self.noninferiority_margin,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("power target has an inconsistent null-boundary separation")
        if (
            isinstance(self.minimum_base_lineages, bool)
            or not isinstance(self.minimum_base_lineages, int)
            or self.minimum_base_lineages <= 0
        ):
            raise ValueError("power target minimum base lineages must be positive")
        required = math.ceil(
            discordance
            * (NormalDist().inv_cdf(1.0 - alpha) + NormalDist().inv_cdf(target_power)) ** 2
            / separation**2
        )
        if self.minimum_base_lineages < max(1, required):
            raise ValueError("power target is below its registered power calculation")
        if not isinstance(self.design_filter, EvaluationStratumFilter):
            raise TypeError("power target filter has the wrong type")

    def as_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "endpoint": self.endpoint,
            "alternative": self.alternative,
            "alpha": self.alpha,
            "target_power": self.target_power,
            "assumed_discordance": self.assumed_discordance,
            "assumed_true_difference": self.assumed_true_difference,
            "noninferiority_margin": self.noninferiority_margin,
            "power_separation": self.power_separation,
            "filter": self.design_filter.as_dict(),
            "learning_partition": self.learning_partition,
            "method": self.method,
            "minimum_base_lineages": self.minimum_base_lineages,
        }

    @property
    def record_digest(self) -> str:
        return stable_digest(self.as_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PowerTarget:
        expected = {
            "target_id",
            "endpoint",
            "alternative",
            "alpha",
            "target_power",
            "assumed_discordance",
            "assumed_true_difference",
            "noninferiority_margin",
            "power_separation",
            "filter",
            "learning_partition",
            "method",
            "minimum_base_lineages",
        }
        _exact_keys(value, expected, label="power target")
        raw_filter = value["filter"]
        if not isinstance(raw_filter, Mapping):
            raise ValueError("power target filter must be an object")
        return cls(
            target_id=value["target_id"],  # type: ignore[arg-type]
            endpoint=value["endpoint"],  # type: ignore[arg-type]
            alternative=value["alternative"],  # type: ignore[arg-type]
            alpha=value["alpha"],  # type: ignore[arg-type]
            target_power=value["target_power"],  # type: ignore[arg-type]
            assumed_discordance=value["assumed_discordance"],  # type: ignore[arg-type]
            assumed_true_difference=value["assumed_true_difference"],  # type: ignore[arg-type]
            noninferiority_margin=value["noninferiority_margin"],  # type: ignore[arg-type]
            power_separation=value["power_separation"],  # type: ignore[arg-type]
            design_filter=EvaluationStratumFilter.from_mapping(raw_filter),
            learning_partition=value["learning_partition"],  # type: ignore[arg-type]
            method=value["method"],  # type: ignore[arg-type]
            minimum_base_lineages=value["minimum_base_lineages"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class PrecisionTarget:
    """Registered precision contract for the paired primary utility difference."""

    target_id: str
    endpoint: str
    learning_partition: str
    design_filter: EvaluationStratumFilter
    method: str
    confidence_level: float
    half_width: float
    minimum_base_lineages: int
    outcome_bounds: tuple[float, float]
    variance_bound: float

    def __post_init__(self) -> None:
        _text(self.target_id, label="precision target identity")
        if self.endpoint != "learned-minus-stock-unconditional-if-q3-s0":
            raise ValueError("unsupported confirmatory precision target endpoint")
        if self.method != "bounded-paired-difference-worst-case-normal":
            raise ValueError("unsupported confirmatory precision target method")
        if self.learning_partition not in {"val", "test"}:
            raise ValueError("confirmatory precision target must address validation or test")
        confidence = _probability(
            self.confidence_level, label="precision target confidence level"
        )
        if confidence <= 0.5:
            raise ValueError("precision target confidence level must exceed one half")
        half_width = _probability(self.half_width, label="precision target half width")
        if self.outcome_bounds != (-1.0, 1.0):
            raise ValueError("paired-difference precision target must use bounds [-1,1]")
        if self.variance_bound != 1.0:
            raise ValueError("paired-difference precision target must use variance bound one")
        if (
            isinstance(self.minimum_base_lineages, bool)
            or not isinstance(self.minimum_base_lineages, int)
            or self.minimum_base_lineages <= 0
        ):
            raise ValueError("precision target minimum base lineages must be positive")
        required = math.ceil(
            self.variance_bound
            * NormalDist().inv_cdf(0.5 + confidence / 2.0) ** 2
            / half_width**2
        )
        if self.minimum_base_lineages < max(1, required):
            raise ValueError("precision target is below its registered precision calculation")
        if not isinstance(self.design_filter, EvaluationStratumFilter):
            raise TypeError("precision target filter has the wrong type")

    def as_dict(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "endpoint": self.endpoint,
            "learning_partition": self.learning_partition,
            "filter": self.design_filter.as_dict(),
            "method": self.method,
            "confidence_level": self.confidence_level,
            "half_width": self.half_width,
            "minimum_base_lineages": self.minimum_base_lineages,
            "outcome_bounds": list(self.outcome_bounds),
            "variance_bound": self.variance_bound,
        }

    @property
    def record_digest(self) -> str:
        return stable_digest(self.as_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PrecisionTarget:
        expected = {
            "target_id",
            "endpoint",
            "learning_partition",
            "filter",
            "method",
            "confidence_level",
            "half_width",
            "minimum_base_lineages",
            "outcome_bounds",
            "variance_bound",
        }
        _exact_keys(value, expected, label="precision target")
        raw_filter = value["filter"]
        raw_bounds = value["outcome_bounds"]
        if not isinstance(raw_filter, Mapping):
            raise ValueError("precision target filter must be an object")
        if not isinstance(raw_bounds, list) or len(raw_bounds) != 2:
            raise ValueError("precision target outcome bounds must contain two values")
        return cls(
            target_id=value["target_id"],  # type: ignore[arg-type]
            endpoint=value["endpoint"],  # type: ignore[arg-type]
            learning_partition=value["learning_partition"],  # type: ignore[arg-type]
            design_filter=EvaluationStratumFilter.from_mapping(raw_filter),
            method=value["method"],  # type: ignore[arg-type]
            confidence_level=value["confidence_level"],  # type: ignore[arg-type]
            half_width=value["half_width"],  # type: ignore[arg-type]
            minimum_base_lineages=value["minimum_base_lineages"],  # type: ignore[arg-type]
            outcome_bounds=tuple(raw_bounds),  # type: ignore[arg-type]
            variance_bound=value["variance_bound"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ConfirmatoryEvaluationDesign:
    """Exact active power family projected from an authenticated prepared receipt."""

    source_manifest_sha256: str
    source_manifest_record_digest: str
    source_power_targets_digest: str
    source_precision_targets_digest: str
    learning_partition: str
    noninferiority_margin: float
    power_targets: tuple[PowerTarget, ...]
    precision_targets: tuple[PrecisionTarget, ...]
    familywise_alpha: float = FAMILYWISE_ALPHA
    multiplicity_method: str = MULTIPLICITY_METHOD
    independent_unit: str = INDEPENDENT_UNIT
    minimum_informative_discordant_lineages: int = MINIMUM_INFORMATIVE_DISCORDANT_LINEAGES

    def __post_init__(self) -> None:
        _digest(self.source_manifest_sha256, label="source corpus-design file identity")
        _digest(
            self.source_manifest_record_digest, label="source corpus-design record identity"
        )
        _digest(self.source_power_targets_digest, label="source power-target registry identity")
        _digest(
            self.source_precision_targets_digest,
            label="source precision-target registry identity",
        )
        if self.learning_partition not in {"val", "test"}:
            raise ValueError("confirmatory design must address validation or test")
        if self.noninferiority_margin != REGISTERED_NONINFERIORITY_MARGIN:
            raise ValueError("confirmatory noninferiority margin is not the registered grid margin")
        if self.familywise_alpha != FAMILYWISE_ALPHA:
            raise ValueError("confirmatory design familywise alpha is not registered")
        if self.multiplicity_method != MULTIPLICITY_METHOD:
            raise ValueError("confirmatory design multiplicity method is not registered")
        if self.independent_unit != INDEPENDENT_UNIT:
            raise ValueError("confirmatory design independent unit is not registered")
        if self.minimum_informative_discordant_lineages != (
            MINIMUM_INFORMATIVE_DISCORDANT_LINEAGES
        ):
            raise ValueError("confirmatory design discordance guard is not registered")
        if (
            not isinstance(self.power_targets, tuple)
            or not self.power_targets
            or any(not isinstance(target, PowerTarget) for target in self.power_targets)
        ):
            raise ValueError("confirmatory design needs typed power targets")
        target_ids = tuple(target.target_id for target in self.power_targets)
        if target_ids != tuple(sorted(set(target_ids))):
            raise ValueError("confirmatory power targets must be unique and canonically sorted")
        if any(target.learning_partition != self.learning_partition for target in self.power_targets):
            raise ValueError("confirmatory target partition differs from its active design")
        if any(
            target.noninferiority_margin != self.noninferiority_margin
            for target in self.power_targets
        ):
            raise ValueError("confirmatory target margin differs from the frozen grid margin")
        if sum(target.alpha for target in self.power_targets) > self.familywise_alpha + 1e-12:
            raise ValueError("confirmatory target alpha spending exceeds the familywise alpha")
        if (
            not isinstance(self.precision_targets, tuple)
            or len(self.precision_targets) != 1
            or any(not isinstance(target, PrecisionTarget) for target in self.precision_targets)
        ):
            raise ValueError("confirmatory design needs exactly one paired primary precision target")
        precision_ids = tuple(target.target_id for target in self.precision_targets)
        if precision_ids != tuple(sorted(set(precision_ids))):
            raise ValueError("confirmatory precision targets must be unique and sorted")
        if any(
            target.learning_partition != self.learning_partition
            for target in self.precision_targets
        ):
            raise ValueError("confirmatory precision partition differs from its active design")

    def _payload(self) -> dict[str, object]:
        return {
            "schema": CONFIRMATORY_DESIGN_SCHEMA,
            "schema_version": CONFIRMATORY_DESIGN_VERSION,
            "source_manifest_sha256": self.source_manifest_sha256,
            "source_manifest_record_digest": self.source_manifest_record_digest,
            "source_power_targets_digest": self.source_power_targets_digest,
            "source_precision_targets_digest": self.source_precision_targets_digest,
            "learning_partition": self.learning_partition,
            "noninferiority_margin": self.noninferiority_margin,
            "familywise_alpha": self.familywise_alpha,
            "multiplicity_method": self.multiplicity_method,
            "independent_unit": self.independent_unit,
            "minimum_informative_discordant_lineages": (
                self.minimum_informative_discordant_lineages
            ),
            "power_targets": [target.as_dict() for target in self.power_targets],
            "power_target_record_digests": [
                target.record_digest for target in self.power_targets
            ],
            "precision_targets": [target.as_dict() for target in self.precision_targets],
            "precision_target_record_digests": [
                target.record_digest for target in self.precision_targets
            ],
        }

    @property
    def record_digest(self) -> str:
        return stable_digest(self._payload())

    def as_dict(self) -> dict[str, object]:
        return {**self._payload(), "record_digest": self.record_digest}

    @classmethod
    def from_prepared_receipt(
        cls,
        receipt: Mapping[str, object],
        *,
        partition: str,
        noninferiority_margin: float,
    ) -> ConfirmatoryEvaluationDesign:
        normalized_partition = "val" if partition == "validation" else partition
        source_manifest_sha256 = _digest(
            receipt.get("manifest_sha256"), label="prepared corpus-design file identity"
        )
        source_manifest_record_digest = _digest(
            receipt.get("manifest_record_digest"),
            label="prepared corpus-design record identity",
        )
        raw_targets = receipt.get("power_targets")
        if (
            not isinstance(raw_targets, Sequence)
            or isinstance(raw_targets, (str, bytes, bytearray))
            or any(not isinstance(target, Mapping) for target in raw_targets)
        ):
            raise ValueError("prepared corpus-design power targets are malformed")
        raw_precision_targets = receipt.get("precision_targets")
        if (
            not isinstance(raw_precision_targets, Sequence)
            or isinstance(raw_precision_targets, (str, bytes, bytearray))
            or any(not isinstance(target, Mapping) for target in raw_precision_targets)
        ):
            raise ValueError("prepared corpus-design precision targets are malformed")
        active = tuple(
            PowerTarget.from_mapping(target)
            for target in raw_targets
            if target.get("learning_partition") == normalized_partition
        )
        active_precision = tuple(
            PrecisionTarget.from_mapping(target)
            for target in raw_precision_targets
            if target.get("learning_partition") == normalized_partition
            and target.get("endpoint")
            == "learned-minus-stock-unconditional-if-q3-s0"
            and target.get("method")
            == "bounded-paired-difference-worst-case-normal"
        )
        return cls(
            source_manifest_sha256=source_manifest_sha256,
            source_manifest_record_digest=source_manifest_record_digest,
            source_power_targets_digest=stable_digest(list(raw_targets)),
            source_precision_targets_digest=stable_digest(list(raw_precision_targets)),
            learning_partition=normalized_partition,
            noninferiority_margin=noninferiority_margin,
            power_targets=active,
            precision_targets=active_precision,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ConfirmatoryEvaluationDesign:
        expected = {
            "schema",
            "schema_version",
            "source_manifest_sha256",
            "source_manifest_record_digest",
            "source_power_targets_digest",
            "source_precision_targets_digest",
            "learning_partition",
            "noninferiority_margin",
            "familywise_alpha",
            "multiplicity_method",
            "independent_unit",
            "minimum_informative_discordant_lineages",
            "power_targets",
            "power_target_record_digests",
            "precision_targets",
            "precision_target_record_digests",
            "record_digest",
        }
        _exact_keys(value, expected, label="confirmatory evaluation design")
        if (
            value["schema"] != CONFIRMATORY_DESIGN_SCHEMA
            or value["schema_version"] != CONFIRMATORY_DESIGN_VERSION
        ):
            raise ValueError("unsupported confirmatory evaluation design schema")
        raw_targets = value["power_targets"]
        raw_digests = value["power_target_record_digests"]
        raw_precision_targets = value["precision_targets"]
        raw_precision_digests = value["precision_target_record_digests"]
        if (
            not isinstance(raw_targets, list)
            or not isinstance(raw_digests, list)
            or any(not isinstance(target, Mapping) for target in raw_targets)
            or not isinstance(raw_precision_targets, list)
            or not isinstance(raw_precision_digests, list)
            or any(not isinstance(target, Mapping) for target in raw_precision_targets)
        ):
            raise ValueError("confirmatory evaluation targets are malformed")
        targets = tuple(PowerTarget.from_mapping(target) for target in raw_targets)
        if raw_digests != [target.record_digest for target in targets]:
            raise ValueError("confirmatory power target record digests are invalid")
        precision_targets = tuple(
            PrecisionTarget.from_mapping(target) for target in raw_precision_targets
        )
        if raw_precision_digests != [target.record_digest for target in precision_targets]:
            raise ValueError("confirmatory precision target record digests are invalid")
        try:
            result = cls(
                source_manifest_sha256=value["source_manifest_sha256"],  # type: ignore[arg-type]
                source_manifest_record_digest=value[  # type: ignore[arg-type]
                    "source_manifest_record_digest"
                ],
                source_power_targets_digest=value["source_power_targets_digest"],  # type: ignore[arg-type]
                source_precision_targets_digest=value[  # type: ignore[arg-type]
                    "source_precision_targets_digest"
                ],
                learning_partition=value["learning_partition"],  # type: ignore[arg-type]
                noninferiority_margin=value["noninferiority_margin"],  # type: ignore[arg-type]
                power_targets=targets,
                precision_targets=precision_targets,
                familywise_alpha=value["familywise_alpha"],  # type: ignore[arg-type]
                multiplicity_method=value["multiplicity_method"],  # type: ignore[arg-type]
                independent_unit=value["independent_unit"],  # type: ignore[arg-type]
                minimum_informative_discordant_lineages=value[  # type: ignore[arg-type]
                    "minimum_informative_discordant_lineages"
                ],
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("confirmatory evaluation design is semantically invalid") from exc
        if value["record_digest"] != result.record_digest:
            raise ValueError("confirmatory evaluation design record digest is invalid")
        return result


def evaluation_strata_from_prepared(
    prepared: Sequence[object], *, partition: str
) -> tuple[EvaluationStratum, ...]:
    """Project prepared candidate rows onto unique policy-instance strata."""

    normalized_partition = "val" if partition == "validation" else partition
    if normalized_partition not in {"val", "test"}:
        raise ValueError("complete-system strata require validation or test")
    grouped: dict[tuple[str, str], list[object]] = {}
    for item in prepared:
        if getattr(item, "partition", None) != normalized_partition:
            continue
        task = getattr(item, "task", None)
        lineage = getattr(task, "lineage", None)
        instance = getattr(item, "instance_id", None)
        if not isinstance(lineage, str) or not lineage or not isinstance(instance, str) or not instance:
            raise ValueError("prepared evaluation row lacks its lineage or policy instance")
        grouped.setdefault((lineage, instance), []).append(item)
    if not grouped:
        raise ValueError("prepared evaluation partition has no policy instances")

    result: list[EvaluationStratum] = []
    for (lineage, instance), rows in sorted(grouped.items()):
        projections: list[dict[str, object]] = []
        provenance_digests: set[str] = set()
        for item in rows:
            condition = getattr(item, "design_condition", None)
            provenance = getattr(item, "provenance", None)
            if condition is None or provenance is None:
                raise ValueError("prepared evaluation row lacks authenticated v3 design provenance")
            if (
                condition.base_lineage_key != lineage
                or condition.learning_partition != normalized_partition
                or condition.host_family != provenance.active_topology
                or condition.fault_status != provenance.fault_status
                or condition.distribution_regime != provenance.distribution_regime
                or condition.calibration_status != provenance.calibration_status
                or condition.calibration_sha256 != provenance.calibration_sha256
            ):
                raise ValueError("prepared condition disagrees with its authenticated provenance")
            provenance_digests.add(
                _digest(provenance.record_digest, label="prepared provenance record identity")
            )
            projections.append(
                {
                    "learning_partition": condition.learning_partition,
                    "application_family": condition.application_family,
                    "problem_origin": condition.problem_origin,
                    "host_family": condition.host_family,
                    "fault_status": condition.fault_status,
                    "distribution_regime": condition.distribution_regime,
                    "calibration_status": condition.calibration_status,
                    "calibration_sha256": condition.calibration_sha256,
                    "embedding_difficulty": condition.embedding_difficulty,
                    "sampling_difficulty": condition.sampling_difficulty,
                    "decision_difficulty": condition.decision_difficulty,
                    "nominal_size": provenance.nominal_size,
                    "source_registry_row_digest": condition.registry_row_digest,
                }
            )
        first = projections[0]
        if any(projection != first for projection in projections[1:]):
            raise ValueError("candidate rows for one policy instance have different strata")
        result.append(
            EvaluationStratum(
                lineage=lineage,
                instance=instance,
                source_provenance_record_digests=tuple(sorted(provenance_digests)),
                **first,  # type: ignore[arg-type]
            )
        )
    return tuple(result)


def evaluation_contract_from_prepared(
    prepared: Sequence[object],
    *,
    corpus_design_receipt: Mapping[str, object],
    partition: str,
    noninferiority_margin: float,
) -> tuple[tuple[EvaluationStratum, ...], ConfirmatoryEvaluationDesign]:
    """Build the two values embedded in a complete-system population identity."""

    return (
        evaluation_strata_from_prepared(prepared, partition=partition),
        ConfirmatoryEvaluationDesign.from_prepared_receipt(
            corpus_design_receipt,
            partition=partition,
            noninferiority_margin=noninferiority_margin,
        ),
    )


def strata_by_identity(
    strata: Sequence[EvaluationStratum],
) -> Mapping[tuple[str, str], EvaluationStratum]:
    result: dict[tuple[str, str], EvaluationStratum] = {}
    for stratum in strata:
        if not isinstance(stratum, EvaluationStratum):
            raise TypeError("evaluation strata must contain typed rows")
        if stratum.identity in result:
            raise ValueError("evaluation strata contain duplicate policy identities")
        result[stratum.identity] = stratum
    return MappingProxyType(result)


def preregistered_subgroups(
    strata: Sequence[EvaluationStratum],
) -> Mapping[str, Mapping[str, frozenset[tuple[str, str]]]]:
    """Return stable subgroup membership without observing model outcomes."""

    dimensions: dict[str, dict[str, set[tuple[str, str]]]] = {
        "application": {},
        "problem_origin": {},
        "topology": {},
        "fault": {},
        "distribution": {},
        "calibration": {},
        "embedding_hardness": {},
        "sampling_hardness": {},
        "decision_hardness": {},
        "joint_hardness": {},
        "size_bin": {},
    }
    for stratum in strata_by_identity(strata).values():
        values = {
            "application": stratum.application_family,
            "problem_origin": stratum.problem_origin,
            "topology": stratum.host_family,
            "fault": stratum.fault_status,
            "distribution": stratum.distribution_regime,
            "calibration": stratum.calibration_status,
            "embedding_hardness": stratum.embedding_difficulty,
            "sampling_hardness": stratum.sampling_difficulty,
            "decision_hardness": stratum.decision_difficulty,
            "joint_hardness": (
                f"embedding={stratum.embedding_difficulty}|"
                f"sampling={stratum.sampling_difficulty}|"
                f"decision={stratum.decision_difficulty}"
            ),
            "size_bin": stratum.size_bin,
        }
        for dimension, value in values.items():
            dimensions[dimension].setdefault(value, set()).add(stratum.identity)
    return MappingProxyType(
        {
            dimension: MappingProxyType(
                {
                    value: frozenset(identities)
                    for value, identities in sorted(groups.items())
                }
            )
            for dimension, groups in dimensions.items()
        }
    )


__all__ = [
    "CONFIRMATORY_DESIGN_SCHEMA",
    "ConfirmatoryEvaluationDesign",
    "EVALUATION_STRATUM_SCHEMA",
    "EvaluationStratum",
    "EvaluationStratumFilter",
    "FAMILYWISE_ALPHA",
    "INDEPENDENT_UNIT",
    "MINIMUM_INFORMATIVE_DISCORDANT_LINEAGES",
    "MULTIPLICITY_METHOD",
    "PowerTarget",
    "PrecisionTarget",
    "REGISTERED_NONINFERIORITY_MARGIN",
    "SIZE_BIN_BOUNDARY_CONVENTION",
    "SIZE_BIN_PROTOCOL",
    "STRATUM_AXES",
    "evaluation_contract_from_prepared",
    "evaluation_strata_from_prepared",
    "nominal_size_bin",
    "preregistered_subgroups",
    "strata_by_identity",
]
