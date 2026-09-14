"""Pure protocol core for train-only quality continuation-count resolution.

This module deliberately performs no target access and no CLI I/O. It owns the frozen v1 config,
the deterministic sample primitives, and the exact finite-population decision rule. Target-bearing
plan and worker commands must build on these functions without weakening their checks.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


QUALITY_RESOLUTION_CONFIG_SCHEMA = "isingfold.quality-resolution-config"
QUALITY_RESOLUTION_CONFIG_VERSION = 1
QUALITY_RESOLUTION_STUDY_ID = "if-quality-resolution-v1"
RESOLUTION_LINEAGE_SAMPLE_DOMAIN = "quality-resolution-lineage-sample-v1"
REGISTERED_CANDIDATE_CONTINUATIONS = (12, 16, 24, 32, 48, 64, 96, 128)
BEST_ACTION_ALPHA = 0.05
FAMILYWISE_ALPHA_NUMERATOR = 1
FAMILYWISE_ALPHA_DENOMINATOR = 20
BONFERRONI_COMPARISONS = len(REGISTERED_CANDIDATE_CONTINUATIONS)
BONFERRONI_ALPHA_NUMERATOR = FAMILYWISE_ALPHA_NUMERATOR
BONFERRONI_ALPHA_DENOMINATOR = FAMILYWISE_ALPHA_DENOMINATOR * BONFERRONI_COMPARISONS
MAX_SEED = 2**63

_LOWERCASE_HEX = frozenset("0123456789abcdef")
_CONFIG_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "study_id",
        "partition",
        "production_lineages",
        "study_lineages",
        "tasks_per_lineage_cap",
        "states_per_lineage_cap",
        "evaluated_actions",
        "candidate_continuations",
        "reward_reads",
        "quality_seed",
        "sampling_seed",
        "familywise_alpha",
        "minimum_resolved_rows",
        "minimum_resolved_lineages",
        "lineages_per_shard",
    }
)


class QualityResolutionError(ValueError):
    """Raised when a quality-resolution protocol value fails closed."""


@dataclass(frozen=True)
class ContinuationDelta:
    """One immutable half-open continuation-index range in the registered ladder."""

    stage_index: int
    lower: int
    upper: int

    @property
    def count(self) -> int:
        return self.upper - self.lower


@dataclass(frozen=True)
class QualityResolutionConfig:
    """Strict in-memory representation of the registered v1 config."""

    study_id: str
    partition: str
    production_lineages: str
    study_lineages: int
    tasks_per_lineage_cap: int
    states_per_lineage_cap: int
    evaluated_actions: int
    candidate_continuations: tuple[int, ...]
    reward_reads: int
    quality_seed: int
    sampling_seed: int
    familywise_alpha: float
    minimum_resolved_rows: int
    minimum_resolved_lineages: int
    lineages_per_shard: int

    @property
    def continuation_deltas(self) -> tuple[ContinuationDelta, ...]:
        return continuation_delta_ranges(self.candidate_continuations)


@dataclass(frozen=True)
class CandidateResolution:
    """Exact finite-population result for one registered continuation count."""

    continuations: int
    sampled_resolved_lineages: int
    lower_population_bound: int
    passes: bool


@dataclass(frozen=True)
class ResolutionSelection:
    """Deterministic progressive decision over a prefix of the registered ladder."""

    results: tuple[CandidateResolution, ...]
    selected_continuations: int | None
    next_continuations: int | None
    advance: bool
    terminal: bool
    stop_reason: str
    familywise_alpha_numerator: int = FAMILYWISE_ALPHA_NUMERATOR
    familywise_alpha_denominator: int = FAMILYWISE_ALPHA_DENOMINATOR
    per_comparison_alpha_numerator: int = BONFERRONI_ALPHA_NUMERATOR
    per_comparison_alpha_denominator: int = BONFERRONI_ALPHA_DENOMINATOR


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise QualityResolutionError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise QualityResolutionError(f"{name} must be a nonnegative integer")
    return value


def _digest(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _LOWERCASE_HEX for character in value)
    ):
        raise QualityResolutionError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _probability(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualityResolutionError(f"{name} must be numeric")
    probability = float(value)
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise QualityResolutionError(f"{name} must lie strictly between zero and one")
    return probability


def _strict_json_object(path: Path) -> dict[str, object]:
    if path.is_symlink():
        raise QualityResolutionError("quality-resolution config must not be a symlink")
    if not path.is_file():
        raise QualityResolutionError("quality-resolution config is missing or not a regular file")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise QualityResolutionError(
                    f"quality-resolution config contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise QualityResolutionError(
            f"quality-resolution config contains non-finite number {token}"
        )

    try:
        raw = path.read_bytes().decode("utf-8")
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except QualityResolutionError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualityResolutionError("quality-resolution config is invalid JSON") from error
    if not isinstance(value, dict):
        raise QualityResolutionError("quality-resolution config must be a JSON object")
    return value


def load_quality_resolution_config(path: str | Path) -> QualityResolutionConfig:
    """Load the only publication-eligible continuation-resolution config."""

    return validate_quality_resolution_config(_strict_json_object(Path(path)))


def validate_quality_resolution_config(raw: Mapping[str, object]) -> QualityResolutionConfig:
    """Reject every schema or scientific-protocol drift from v1."""

    if not isinstance(raw, Mapping) or set(raw) != _CONFIG_FIELDS:
        observed = set(raw) if isinstance(raw, Mapping) else set()
        raise QualityResolutionError(
            "quality-resolution config schema fields differ: "
            f"missing={sorted(_CONFIG_FIELDS - observed)}, "
            f"unknown={sorted(observed - _CONFIG_FIELDS)}"
        )
    expected_scalars: dict[str, object] = {
        "schema": QUALITY_RESOLUTION_CONFIG_SCHEMA,
        "schema_version": QUALITY_RESOLUTION_CONFIG_VERSION,
        "study_id": QUALITY_RESOLUTION_STUDY_ID,
        "partition": "train",
        "production_lineages": "all",
        "study_lineages": 128,
        "tasks_per_lineage_cap": 1,
        "states_per_lineage_cap": 4,
        "evaluated_actions": 8,
        "reward_reads": 256,
        "quality_seed": 907,
        "sampling_seed": 1907,
        "familywise_alpha": BEST_ACTION_ALPHA,
        "minimum_resolved_rows": 128,
        "minimum_resolved_lineages": 128,
        "lineages_per_shard": 2,
    }
    for field, expected in expected_scalars.items():
        observed = raw[field]
        if isinstance(expected, int) and (
            isinstance(observed, bool) or not isinstance(observed, int)
        ):
            raise QualityResolutionError(f"quality-resolution {field} has an invalid type")
        if isinstance(expected, float) and (
            isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
        ):
            raise QualityResolutionError(f"quality-resolution {field} has an invalid type")
        if observed != expected:
            if field == "production_lineages":
                raise QualityResolutionError(
                    "quality-resolution production must use all train lineages"
                )
            raise QualityResolutionError(
                f"quality-resolution {field} differs from registered value {expected!r}"
            )
    raw_ladder = raw["candidate_continuations"]
    if not isinstance(raw_ladder, list) or tuple(raw_ladder) != REGISTERED_CANDIDATE_CONTINUATIONS:
        raise QualityResolutionError(
            "quality-resolution candidate_continuations differ from the registered ladder"
        )
    continuation_delta_ranges(raw_ladder)
    return QualityResolutionConfig(
        study_id=QUALITY_RESOLUTION_STUDY_ID,
        partition="train",
        production_lineages="all",
        study_lineages=128,
        tasks_per_lineage_cap=1,
        states_per_lineage_cap=4,
        evaluated_actions=8,
        candidate_continuations=REGISTERED_CANDIDATE_CONTINUATIONS,
        reward_reads=256,
        quality_seed=907,
        sampling_seed=1907,
        familywise_alpha=BEST_ACTION_ALPHA,
        minimum_resolved_rows=128,
        minimum_resolved_lineages=128,
        lineages_per_shard=2,
    )


def hoeffding_radius(continuations: int, arm_count: int, *, alpha: float) -> float:
    """Return the current simultaneous bounded-mean interval radius."""

    count = _positive_int(continuations, "continuations")
    arms = _positive_int(arm_count, "arm_count")
    probability = _probability(alpha, "alpha")
    return math.sqrt(math.log(2.0 * arms / probability) / (2.0 * count))


def bounded_resolution_possible(continuations: int, arm_count: int, *, alpha: float) -> bool:
    """Whether any bounded-reward row can exclude one of at least two arms."""

    radius = hoeffding_radius(continuations, arm_count, alpha=alpha)
    arms = _positive_int(arm_count, "arm_count")
    if arms < 2:
        return False
    return 2.0 * radius < 1.0


def minimum_continuations_for_possible_resolution(arm_count: int, *, alpha: float) -> int:
    """Smallest integer denominator for which an extreme bounded row may resolve."""

    arms = _positive_int(arm_count, "arm_count")
    if arms < 2:
        raise QualityResolutionError("at least two arms are required for a resolution comparison")
    probability = _probability(alpha, "alpha")
    # The strict inequality is C > 2 log(2 A / alpha). Check the computed boundary again to
    # protect the result from a floating representation that lands next to an integer.
    threshold = 2.0 * math.log(2.0 * arms / probability)
    candidate = math.floor(threshold) + 1
    while not bounded_resolution_possible(candidate, arms, alpha=alpha):
        candidate += 1
    while candidate > 1 and bounded_resolution_possible(candidate - 1, arms, alpha=alpha):
        candidate -= 1
    return candidate


def continuation_delta_ranges(ladder: Sequence[int]) -> tuple[ContinuationDelta, ...]:
    """Turn a strictly increasing cumulative ladder into immutable half-open deltas."""

    if isinstance(ladder, (str, bytes)) or not isinstance(ladder, Sequence) or not ladder:
        raise QualityResolutionError("continuation ladder must be a nonempty sequence")
    deltas: list[ContinuationDelta] = []
    lower = 0
    for stage_index, raw_upper in enumerate(ladder):
        upper = _positive_int(raw_upper, f"continuation ladder stage {stage_index}")
        if upper <= lower:
            raise QualityResolutionError("continuation ladder must be strictly increasing")
        deltas.append(ContinuationDelta(stage_index=stage_index, lower=lower, upper=upper))
        lower = upper
    return tuple(deltas)


def domain_separated_seed63(root: int, domain: str, *parts: object) -> int:
    """Derive one deterministic 63-bit seed with the existing quality framing."""

    seed_root = _nonnegative_int(root, "seed root")
    if seed_root >= MAX_SEED:
        raise QualityResolutionError("seed root must be a 63-bit integer")
    if not isinstance(domain, str) or not domain:
        raise QualityResolutionError("seed domain must be nonempty text")
    try:
        encoded = json.dumps(
            {"domain": domain, "parts": [seed_root, *parts], "version": 1},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise QualityResolutionError("seed coordinates must be finite canonical JSON") from error
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & (MAX_SEED - 1)


def require_unique_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    """Validate a complete seed registry and reject any collision."""

    validated: list[int] = []
    seen: set[int] = set()
    for position, raw_seed in enumerate(seeds):
        seed = _nonnegative_int(raw_seed, f"seed at position {position}")
        if seed >= MAX_SEED:
            raise QualityResolutionError("seed registry contains a value outside 63 bits")
        if seed in seen:
            raise QualityResolutionError(f"seed registry collision at value {seed}")
        seen.add(seed)
        validated.append(seed)
    return tuple(validated)


def resolution_lineage_key(
    lineage: str,
    *,
    sampling_seed: int,
    source_corpus_manifest_sha256: str,
    production_plan_digest: str,
) -> str:
    """Return the fixed outcome-blind hash-permutation key for one lineage."""

    if not isinstance(lineage, str) or not lineage:
        raise QualityResolutionError("resolution lineage must be nonempty text")
    seed = _nonnegative_int(sampling_seed, "sampling_seed")
    if seed >= MAX_SEED:
        raise QualityResolutionError("sampling_seed must be a 63-bit integer")
    payload = {
        "domain": RESOLUTION_LINEAGE_SAMPLE_DOMAIN,
        "seed": seed,
        "source_corpus_manifest_sha256": _digest(
            source_corpus_manifest_sha256,
            "source corpus manifest SHA-256",
        ),
        "production_plan_digest": _digest(production_plan_digest, "production plan digest"),
        "base_lineage": lineage,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def select_resolution_lineages(
    lineages: Iterable[str],
    *,
    sample_size: int,
    sampling_seed: int,
    source_corpus_manifest_sha256: str,
    production_plan_digest: str,
) -> tuple[str, ...]:
    """Select an exact deterministic sample without replacement from the full plan."""

    if isinstance(lineages, (str, bytes)):
        raise QualityResolutionError("resolution lineages must be an iterable of identities")
    population = tuple(lineages)
    if any(not isinstance(lineage, str) or not lineage for lineage in population):
        raise QualityResolutionError("resolution lineages must be nonempty text")
    if len(set(population)) != len(population):
        raise QualityResolutionError("resolution population contains duplicate lineages")
    count = _positive_int(sample_size, "sample_size")
    if count > len(population):
        raise QualityResolutionError("resolution sample requests more lineages than are available")
    keyed = [
        (
            resolution_lineage_key(
                lineage,
                sampling_seed=sampling_seed,
                source_corpus_manifest_sha256=source_corpus_manifest_sha256,
                production_plan_digest=production_plan_digest,
            ),
            lineage,
        )
        for lineage in population
    ]
    keys = [key for key, _lineage in keyed]
    if len(set(keys)) != len(keys):
        raise QualityResolutionError("resolution lineage sampling hash collision")
    keyed.sort()
    return tuple(lineage for _key, lineage in keyed[:count])


def _population_inputs(
    population_size: int,
    sample_size: int,
    observed_successes: int,
) -> tuple[int, int, int]:
    population = _positive_int(population_size, "population_size")
    sample = _positive_int(sample_size, "sample_size")
    observed = _nonnegative_int(observed_successes, "observed_successes")
    if sample > population:
        raise QualityResolutionError("sample_size cannot exceed population_size")
    if observed > sample:
        raise QualityResolutionError("observed_successes cannot exceed sample_size")
    return population, sample, observed


def hypergeometric_upper_tail_counts(
    *,
    population_size: int,
    population_successes: int,
    sample_size: int,
    observed_successes: int,
) -> tuple[int, int]:
    """Return exact numerator and denominator for the inclusive upper tail P[X >= x]."""

    population, sample, observed = _population_inputs(
        population_size,
        sample_size,
        observed_successes,
    )
    successes = _nonnegative_int(population_successes, "population_successes")
    if successes > population:
        raise QualityResolutionError("population_successes cannot exceed population_size")
    lower = max(observed, sample - (population - successes), 0)
    upper = min(sample, successes)
    numerator = sum(
        math.comb(successes, count) * math.comb(population - successes, sample - count)
        for count in range(lower, upper + 1)
    )
    return numerator, math.comb(population, sample)


def hypergeometric_lower_success_bound(
    *,
    population_size: int,
    sample_size: int,
    observed_successes: int,
    alpha_numerator: int,
    alpha_denominator: int,
) -> int:
    """Invert the inclusive upper tail to obtain an exact one-sided lower count bound."""

    population, sample, observed = _population_inputs(
        population_size,
        sample_size,
        observed_successes,
    )
    numerator_alpha = _positive_int(alpha_numerator, "alpha_numerator")
    denominator_alpha = _positive_int(alpha_denominator, "alpha_denominator")
    if numerator_alpha >= denominator_alpha:
        raise QualityResolutionError("alpha must lie strictly between zero and one")
    if observed == 0:
        return 0
    maximum = population - sample + observed
    for population_successes in range(observed, maximum + 1):
        tail_numerator, tail_denominator = hypergeometric_upper_tail_counts(
            population_size=population,
            population_successes=population_successes,
            sample_size=sample,
            observed_successes=observed,
        )
        if denominator_alpha * tail_numerator >= numerator_alpha * tail_denominator:
            return population_successes
    raise RuntimeError("finite-population lower-bound inversion exhausted its valid support")


def select_continuation_count(
    *,
    population_size: int,
    sample_size: int,
    resolved_lineages_by_count: Mapping[int, int],
    minimum_resolved_lineages: int,
) -> ResolutionSelection:
    """Apply the registered Bonferroni rule to a completed ladder prefix."""

    population, sample, _ = _population_inputs(population_size, sample_size, 0)
    minimum = _positive_int(minimum_resolved_lineages, "minimum_resolved_lineages")
    if minimum > population:
        raise QualityResolutionError(
            "minimum_resolved_lineages cannot exceed the finite population"
        )
    if not isinstance(resolved_lineages_by_count, Mapping) or not resolved_lineages_by_count:
        raise QualityResolutionError("candidate results must be a nonempty ladder prefix")
    completed_count = len(resolved_lineages_by_count)
    if completed_count > len(REGISTERED_CANDIDATE_CONTINUATIONS):
        raise QualityResolutionError("candidate results exceed the registered ladder")
    completed = REGISTERED_CANDIDATE_CONTINUATIONS[:completed_count]
    if set(resolved_lineages_by_count) != set(completed):
        raise QualityResolutionError("candidate results must form a registered ladder prefix")

    results: list[CandidateResolution] = []
    selected: int | None = None
    for continuations in completed:
        sampled_resolved = _nonnegative_int(
            resolved_lineages_by_count[continuations],
            f"resolved lineages at C={continuations}",
        )
        if sampled_resolved > sample:
            raise QualityResolutionError(
                f"resolved lineages at C={continuations} exceed sample_size"
            )
        lower = hypergeometric_lower_success_bound(
            population_size=population,
            sample_size=sample,
            observed_successes=sampled_resolved,
            alpha_numerator=BONFERRONI_ALPHA_NUMERATOR,
            alpha_denominator=BONFERRONI_ALPHA_DENOMINATOR,
        )
        passes = lower >= minimum
        results.append(
            CandidateResolution(
                continuations=continuations,
                sampled_resolved_lineages=sampled_resolved,
                lower_population_bound=lower,
                passes=passes,
            )
        )
        if selected is None and passes:
            selected = continuations

    if selected is not None and selected != completed[-1]:
        raise QualityResolutionError(
            "progressive continuation study continued after an earlier passing count"
        )
    if selected is not None:
        return ResolutionSelection(
            results=tuple(results),
            selected_continuations=selected,
            next_continuations=None,
            advance=True,
            terminal=True,
            stop_reason="registered-count-qualified",
        )
    if completed_count == len(REGISTERED_CANDIDATE_CONTINUATIONS):
        return ResolutionSelection(
            results=tuple(results),
            selected_continuations=None,
            next_continuations=None,
            advance=False,
            terminal=True,
            stop_reason="no-registered-count-qualified",
        )
    return ResolutionSelection(
        results=tuple(results),
        selected_continuations=None,
        next_continuations=REGISTERED_CANDIDATE_CONTINUATIONS[completed_count],
        advance=False,
        terminal=False,
        stop_reason="next-registered-count-required",
    )


__all__ = [
    "BEST_ACTION_ALPHA",
    "BONFERRONI_ALPHA_DENOMINATOR",
    "BONFERRONI_ALPHA_NUMERATOR",
    "BONFERRONI_COMPARISONS",
    "CandidateResolution",
    "ContinuationDelta",
    "QualityResolutionConfig",
    "QualityResolutionError",
    "REGISTERED_CANDIDATE_CONTINUATIONS",
    "RESOLUTION_LINEAGE_SAMPLE_DOMAIN",
    "ResolutionSelection",
    "bounded_resolution_possible",
    "continuation_delta_ranges",
    "domain_separated_seed63",
    "hoeffding_radius",
    "hypergeometric_lower_success_bound",
    "hypergeometric_upper_tail_counts",
    "load_quality_resolution_config",
    "minimum_continuations_for_possible_resolution",
    "require_unique_seeds",
    "resolution_lineage_key",
    "select_continuation_count",
    "select_resolution_lineages",
    "validate_quality_resolution_config",
]
