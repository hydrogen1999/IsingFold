from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import pytest

from isingfold.rl.data.quality_resolution import (
    BONFERRONI_ALPHA_DENOMINATOR,
    BONFERRONI_ALPHA_NUMERATOR,
    REGISTERED_CANDIDATE_CONTINUATIONS,
    ContinuationDelta,
    QualityResolutionError,
    bounded_resolution_possible,
    continuation_delta_ranges,
    domain_separated_seed63,
    hypergeometric_lower_success_bound,
    hypergeometric_upper_tail_counts,
    load_quality_resolution_config,
    minimum_continuations_for_possible_resolution,
    require_unique_seeds,
    resolution_lineage_key,
    select_continuation_count,
    select_resolution_lineages,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "quality_resolution_v1.json"


def _registered_config() -> dict[str, object]:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "quality-resolution.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _reference_upper_tail(
    population: int,
    population_successes: int,
    sample: int,
    observed_successes: int,
) -> Fraction:
    numerator = 0
    lower = max(observed_successes, 0, sample - (population - population_successes))
    upper = min(sample, population_successes)
    for successes in range(lower, upper + 1):
        numerator += math.comb(population_successes, successes) * math.comb(
            population - population_successes,
            sample - successes,
        )
    return Fraction(numerator, math.comb(population, sample))


def _reference_lower_bound(
    population: int,
    sample: int,
    observed_successes: int,
    alpha: Fraction,
) -> int:
    if observed_successes == 0:
        return 0
    for population_successes in range(
        observed_successes,
        population - sample + observed_successes + 1,
    ):
        if (
            _reference_upper_tail(
                population,
                population_successes,
                sample,
                observed_successes,
            )
            >= alpha
        ):
            return population_successes
    raise AssertionError("the feasible finite-population range must contain a lower bound")


def test_registered_resolution_config_loads_with_exact_frozen_values() -> None:
    config = load_quality_resolution_config(CONFIG)

    assert config.study_id == "if-quality-resolution-v1"
    assert config.partition == "train"
    assert config.production_lineages == "all"
    assert config.study_lineages == 128
    assert config.tasks_per_lineage_cap == 1
    assert config.states_per_lineage_cap == 4
    assert config.evaluated_actions == 8
    assert config.candidate_continuations == REGISTERED_CANDIDATE_CONTINUATIONS
    assert config.reward_reads == 256
    assert config.quality_seed == 907
    assert config.sampling_seed == 1907
    assert config.familywise_alpha == 0.05
    assert config.minimum_resolved_rows == 128
    assert config.minimum_resolved_lineages == 128
    assert config.lineages_per_shard == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda raw: raw.__setitem__("unknown", 1), "schema fields differ"),
        (lambda raw: raw.__setitem__("partition", "validation"), "partition"),
        (lambda raw: raw.__setitem__("production_lineages", 128), "all train lineages"),
        (lambda raw: raw.__setitem__("study_lineages", True), "study_lineages"),
        (
            lambda raw: raw.__setitem__("candidate_continuations", [8, 12, 16, 24, 32, 48, 64, 96]),
            "candidate_continuations",
        ),
        (lambda raw: raw.__setitem__("familywise_alpha", 0.1), "familywise_alpha"),
        (lambda raw: raw.__setitem__("quality_seed", 1907), "quality_seed"),
    ],
)
def test_resolution_config_rejects_any_schema_or_protocol_drift(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    raw = _registered_config()
    mutation(raw)

    with pytest.raises(QualityResolutionError, match=message):
        load_quality_resolution_config(_write_config(tmp_path, raw))


def test_resolution_config_rejects_duplicate_keys_nonfinite_values_and_symlinks(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    with pytest.raises(QualityResolutionError, match="duplicate key"):
        load_quality_resolution_config(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"familywise_alpha":NaN}', encoding="utf-8")
    with pytest.raises(QualityResolutionError, match="non-finite"):
        load_quality_resolution_config(nonfinite)

    source = _write_config(tmp_path, _registered_config())
    link = tmp_path / "linked-config.json"
    link.symlink_to(source)
    with pytest.raises(QualityResolutionError, match="symlink"):
        load_quality_resolution_config(link)


def test_eight_arm_hoeffding_rule_cannot_resolve_at_or_below_eleven() -> None:
    assert minimum_continuations_for_possible_resolution(8, alpha=0.05) == 12
    assert all(
        not bounded_resolution_possible(continuations=count, arm_count=8, alpha=0.05)
        for count in range(1, 12)
    )
    assert bounded_resolution_possible(continuations=12, arm_count=8, alpha=0.05)


def test_possible_resolution_thresholds_match_two_and_four_arm_boundaries() -> None:
    assert minimum_continuations_for_possible_resolution(2, alpha=0.05) == 9
    assert minimum_continuations_for_possible_resolution(4, alpha=0.05) == 11
    assert not bounded_resolution_possible(continuations=8, arm_count=2, alpha=0.05)


@pytest.mark.parametrize("alpha", [True, "0.05", 0.0, 1.0, math.nan])
def test_hoeffding_possibility_rejects_invalid_alpha_before_computing(alpha) -> None:
    with pytest.raises(QualityResolutionError, match="alpha"):
        minimum_continuations_for_possible_resolution(8, alpha=alpha)


def test_singleton_support_never_resolves_but_still_validates_denominators() -> None:
    assert not bounded_resolution_possible(continuations=128, arm_count=1, alpha=0.05)
    with pytest.raises(QualityResolutionError, match="continuations"):
        bounded_resolution_possible(continuations=0, arm_count=1, alpha=0.05)
    with pytest.raises(QualityResolutionError, match="alpha"):
        bounded_resolution_possible(continuations=128, arm_count=1, alpha=0.0)


def test_registered_ladder_expands_as_exact_nested_immutable_deltas() -> None:
    assert continuation_delta_ranges(REGISTERED_CANDIDATE_CONTINUATIONS) == (
        ContinuationDelta(stage_index=0, lower=0, upper=12),
        ContinuationDelta(stage_index=1, lower=12, upper=16),
        ContinuationDelta(stage_index=2, lower=16, upper=24),
        ContinuationDelta(stage_index=3, lower=24, upper=32),
        ContinuationDelta(stage_index=4, lower=32, upper=48),
        ContinuationDelta(stage_index=5, lower=48, upper=64),
        ContinuationDelta(stage_index=6, lower=64, upper=96),
        ContinuationDelta(stage_index=7, lower=96, upper=128),
    )


@pytest.mark.parametrize(
    "ladder",
    [(), (12, 12), (12, 11), (12, True), (12, 16.0)],
)
def test_delta_registry_rejects_empty_nonintegral_or_nonincreasing_ladders(ladder) -> None:
    with pytest.raises(QualityResolutionError):
        continuation_delta_ranges(ladder)


def test_hypergeometric_upper_tail_is_inclusive_and_exact_on_boundary() -> None:
    numerator, denominator = hypergeometric_upper_tail_counts(
        population_size=2,
        population_successes=1,
        sample_size=1,
        observed_successes=1,
    )
    assert Fraction(numerator, denominator) == Fraction(1, 2)
    assert (
        hypergeometric_lower_success_bound(
            population_size=2,
            sample_size=1,
            observed_successes=1,
            alpha_numerator=1,
            alpha_denominator=2,
        )
        == 1
    )
    assert (
        hypergeometric_lower_success_bound(
            population_size=2,
            sample_size=1,
            observed_successes=1,
            alpha_numerator=2,
            alpha_denominator=3,
        )
        == 2
    )


def test_exact_hypergeometric_lower_inversion_matches_exhaustive_small_oracle() -> None:
    alpha = Fraction(1, 7)
    for population in range(1, 11):
        for sample in range(1, population + 1):
            for observed in range(sample + 1):
                assert hypergeometric_lower_success_bound(
                    population_size=population,
                    sample_size=sample,
                    observed_successes=observed,
                    alpha_numerator=alpha.numerator,
                    alpha_denominator=alpha.denominator,
                ) == _reference_lower_bound(population, sample, observed, alpha)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "population_size": True,
            "sample_size": 1,
            "observed_successes": 1,
        },
        {
            "population_size": 4,
            "sample_size": 5,
            "observed_successes": 1,
        },
        {
            "population_size": 4,
            "sample_size": 2,
            "observed_successes": 3,
        },
    ],
)
def test_hypergeometric_lower_inversion_rejects_invalid_denominators(kwargs) -> None:
    with pytest.raises(QualityResolutionError):
        hypergeometric_lower_success_bound(
            **kwargs,
            alpha_numerator=1,
            alpha_denominator=20,
        )


def test_bonferroni_selection_uses_all_eight_candidates_and_smallest_passing_count() -> None:
    population = 1024
    sample = 128
    threshold_x = next(
        observed
        for observed in range(sample + 1)
        if _reference_lower_bound(
            population,
            sample,
            observed,
            Fraction(BONFERRONI_ALPHA_NUMERATOR, BONFERRONI_ALPHA_DENOMINATOR),
        )
        >= 128
    )
    successes = {
        12: threshold_x - 1,
        16: threshold_x,
    }

    decision = select_continuation_count(
        population_size=population,
        sample_size=sample,
        resolved_lineages_by_count=successes,
        minimum_resolved_lineages=128,
    )

    assert BONFERRONI_ALPHA_NUMERATOR == 1
    assert BONFERRONI_ALPHA_DENOMINATOR == 160
    assert decision.selected_continuations == 16
    assert decision.advance is True
    assert decision.terminal is True
    assert decision.stop_reason == "registered-count-qualified"
    assert [result.continuations for result in decision.results] == [12, 16]
    assert decision.results[0].passes is False
    assert decision.results[1].passes is True


def test_progressive_selection_requests_next_stage_then_fails_at_end_of_ladder() -> None:
    partial = select_continuation_count(
        population_size=1024,
        sample_size=128,
        resolved_lineages_by_count={12: 0, 16: 0},
        minimum_resolved_lineages=128,
    )
    assert partial.selected_continuations is None
    assert partial.advance is False
    assert partial.terminal is False
    assert partial.stop_reason == "next-registered-count-required"
    assert partial.next_continuations == 24

    exhausted = select_continuation_count(
        population_size=1024,
        sample_size=128,
        resolved_lineages_by_count={count: 0 for count in REGISTERED_CANDIDATE_CONTINUATIONS},
        minimum_resolved_lineages=128,
    )
    assert exhausted.selected_continuations is None
    assert exhausted.advance is False
    assert exhausted.terminal is True
    assert exhausted.stop_reason == "no-registered-count-qualified"
    assert exhausted.next_continuations is None


def test_selection_rejects_a_nonprefix_candidate_result_registry() -> None:
    with pytest.raises(QualityResolutionError, match="prefix"):
        select_continuation_count(
            population_size=1024,
            sample_size=128,
            resolved_lineages_by_count={12: 1, 24: 2},
            minimum_resolved_lineages=128,
        )


def test_progressive_selection_rejects_results_generated_after_an_earlier_pass() -> None:
    with pytest.raises(QualityResolutionError, match="continued after"):
        select_continuation_count(
            population_size=1024,
            sample_size=128,
            resolved_lineages_by_count={12: 128, 16: 128},
            minimum_resolved_lineages=128,
        )


def test_domain_separated_seed_matches_canonical_sha256_framing() -> None:
    expected_payload = {
        "domain": "fixture-domain-v1",
        "parts": [1907, "lineage-a", 4],
        "version": 1,
    }
    encoded = json.dumps(
        expected_payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    expected = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & (2**63 - 1)

    assert domain_separated_seed63(1907, "fixture-domain-v1", "lineage-a", 4) == expected
    assert domain_separated_seed63(1907, "other-domain-v1", "lineage-a", 4) != expected


def test_resolution_lineage_sample_is_deterministic_order_invariant_and_exact() -> None:
    lineages = [f"lineage-{index:04d}" for index in range(256)]
    source_digest = "a" * 64
    plan_digest = "b" * 64
    selected = select_resolution_lineages(
        lineages,
        sample_size=128,
        sampling_seed=1907,
        source_corpus_manifest_sha256=source_digest,
        production_plan_digest=plan_digest,
    )
    reversed_selected = select_resolution_lineages(
        reversed(lineages),
        sample_size=128,
        sampling_seed=1907,
        source_corpus_manifest_sha256=source_digest,
        production_plan_digest=plan_digest,
    )
    expected = tuple(
        sorted(
            lineages,
            key=lambda lineage: (
                resolution_lineage_key(
                    lineage,
                    sampling_seed=1907,
                    source_corpus_manifest_sha256=source_digest,
                    production_plan_digest=plan_digest,
                ),
                lineage,
            ),
        )[:128]
    )

    assert selected == reversed_selected == expected
    assert len(selected) == len(set(selected)) == 128


def test_resolution_sample_and_seed_registry_fail_closed_on_ambiguity() -> None:
    with pytest.raises(QualityResolutionError, match="duplicate"):
        select_resolution_lineages(
            ["same", "same"],
            sample_size=1,
            sampling_seed=1907,
            source_corpus_manifest_sha256="a" * 64,
            production_plan_digest="b" * 64,
        )
    with pytest.raises(QualityResolutionError, match="more lineages"):
        select_resolution_lineages(
            ["only"],
            sample_size=2,
            sampling_seed=1907,
            source_corpus_manifest_sha256="a" * 64,
            production_plan_digest="b" * 64,
        )
    with pytest.raises(QualityResolutionError, match="collision"):
        require_unique_seeds([1, 2, 1])
