from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest
from embedbench.isingfold_corpus_plan import (
    CandidateScreeningProtocol,
    CorpusGenerationPlan,
    build_production_corpus_plan,
    read_corpus_plan,
    write_corpus_plan,
)
from embedbench.isingfold_corpus_shard import (
    LineageRequest,
    ShardConfig,
    lineage_shard,
    prospect_lineage,
)


def _prospective_slot(config: ShardConfig, request: LineageRequest) -> int:
    """Pickle-safe worker for the complete production-plan census."""

    return prospect_lineage(config, request).prospective_slot


def _production_prospect_config(
    *, root_seed: int, screening: CandidateScreeningProtocol
) -> ShardConfig:
    return ShardConfig(
        root_seed=root_seed,
        shard_index=0,
        shard_count=1,
        attempt_slots=screening.attempt_slots,
        incumbent_search_slots=screening.incumbent_search_slots,
        max_split_search=screening.max_split_search,
        strengths=screening.strengths,
        reads=screening.reads,
        sweeps=screening.sweeps,
    )


def test_production_plan_has_registered_denominators_and_diverse_hard_ood_support() -> None:
    plan = build_production_corpus_plan(root_seed=260912, shard_count=64)
    counts = Counter(item.request.partition for item in plan.lineages)
    document = plan.to_dict()

    assert document["schema_version"] == 2
    assert document["prospective_identity_policy"] == {
        "enforcement": "pre-generation-fail-closed",
        "identity_fields": ["problem_sha256", "split_unit_id"],
        "map_entry_fields": [
            "lineage_id",
            "prospective_slot",
            "split_unit_id",
            "problem_sha256",
        ],
        "map_protocol": "canonical-prospective-identity-map-v1",
        "requirement": "global-uniqueness-across-planned-lineages",
    }
    assert document["release_scope"] == {
        "application_family_coverage": "graphcut-portfolio-jobshop-in-train-val-test",
        "ood_claim_scope": "registered-topology-host-scale-fault-and-distribution-regime-shifts",
        "profile_id": "isingfold-corpus-v4",
        "separate_unseen_family_profile": "docs/HARD_OOD_CORPUS_SPEC.md",
        "unseen_application_family_ood": False,
    }
    assert counts == {"train": 1024, "val": 512, "test": 1546}
    assert len({item.request.lineage_id for item in plan.lineages}) == 3082
    assert {item.request.topology for item in plan.lineages} == {
        "chimera",
        "pegasus",
        "zephyr",
    }
    assert {item.request.origin for item in plan.lineages} == {
        "application-derived",
        "synthetic-ink-drop",
    }
    assert {item.difficulty_intent for item in plan.lineages} == {"easy", "hard"}
    assert any(item.request.qubit_fraction > 0.0 for item in plan.lineages)
    assert any(item.request.qubit_fraction == 0.0 for item in plan.lineages)
    assert {
        item.request.application_family
        for item in plan.lineages
        if item.request.origin == "application-derived"
    } == {"graphcut", "jobshop", "portfolio"}
    assert {
        item.request.ink_drop_mode
        for item in plan.lineages
        if item.request.origin == "synthetic-ink-drop"
    } == {"compact", "elongated", "cut_congested", "near_capacity"}

    ood = [item for item in plan.lineages if item.request.distribution_regime == "ood"]
    assert ood
    assert all(item.request.partition == "test" for item in ood)
    assert all(item.shift_axes for item in ood)
    test_regimes = {
        item.request.distribution_regime
        for item in plan.lineages
        if item.request.partition == "test"
    }
    assert test_regimes == {
        "iid",
        "ood",
    }
    assert all(
        item.request.ink_drop_mode in {"compact", "elongated"}
        for item in plan.lineages
        if item.request.origin == "synthetic-ink-drop"
        and item.request.distribution_regime == "iid"
    )
    assert all(
        item.request.ink_drop_mode in {"cut_congested", "near_capacity"}
        for item in ood
        if item.request.origin == "synthetic-ink-drop"
    )

    shard_counts = Counter(
        lineage_shard(item.request.lineage_id, plan.shard_count) for item in plan.lineages
    )
    assert set(shard_counts) == set(range(plan.shard_count))
    assert max(shard_counts.values()) - min(shard_counts.values()) < 40


def test_production_plan_rejects_any_empty_shard() -> None:
    with pytest.raises(ValueError, match="every shard index must contain at least one lineage"):
        build_production_corpus_plan(root_seed=260912, shard_count=3083)


def test_production_prospect_budget_covers_known_chimera_ood_regression() -> None:
    plan = build_production_corpus_plan(root_seed=260912, shard_count=64)
    planned = next(
        item
        for item in plan.lineages
        if item.request.lineage_id == "if-prod-v4-test-1261-f42b15af179e7a09"
    )

    prospect = prospect_lineage(
        _production_prospect_config(root_seed=plan.root_seed, screening=plan.screening),
        planned.request,
    )

    assert plan.screening.max_split_search == 1024
    assert prospect.prospective_slot == 76
    assert prospect.request == planned.request


def test_production_plan_rejects_an_underbudgeted_prospective_protocol() -> None:
    screening = CandidateScreeningProtocol(max_split_search=736)

    with pytest.raises(ValueError, match="at least 1024 fixed split-search slots"):
        build_production_corpus_plan(
            root_seed=260912,
            shard_count=64,
            screening=screening,
        )


def test_complete_production_plan_is_prospectively_generatable() -> None:
    """Census every planned lineage without consulting proposal or quality outcomes."""

    plan = build_production_corpus_plan(root_seed=260912, shard_count=64)
    config = _production_prospect_config(
        root_seed=plan.root_seed,
        screening=plan.screening,
    )
    context_name = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    worker_count = min(10, os.cpu_count() or 1)
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context(context_name),
    ) as pool:
        slots = tuple(
            pool.map(
                _prospective_slot,
                (config for _ in plan.lineages),
                (item.request for item in plan.lineages),
                chunksize=8,
            )
        )

    assert len(slots) == 3082
    assert max(slots) == 597
    assert all(0 <= slot < plan.screening.max_split_search for slot in slots)


def test_plan_is_deterministic_but_root_seed_is_scientifically_visible() -> None:
    first = build_production_corpus_plan(root_seed=260912, shard_count=32)
    second = build_production_corpus_plan(root_seed=260912, shard_count=32)
    other = build_production_corpus_plan(root_seed=260913, shard_count=32)

    assert first.to_dict() == second.to_dict()
    assert first.record_digest == second.record_digest
    assert first.record_digest != other.record_digest
    assert first.lineages[0].request.lineage_id != other.lineages[0].request.lineage_id


def test_plan_round_trip_is_canonical_pinned_and_fail_closed(tmp_path: Path) -> None:
    plan = build_production_corpus_plan(root_seed=260912, shard_count=16)
    path = tmp_path / "corpus_plan.json"
    receipt = write_corpus_plan(plan, path)

    assert receipt.path == path.resolve()
    assert receipt.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.read_bytes().endswith(b"\n")
    assert read_corpus_plan(path, expected_sha256=receipt.sha256) == plan

    document = json.loads(path.read_bytes())
    document["lineages"][0]["difficulty_intent"] = "easy"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="SHA-256"):
        read_corpus_plan(path, expected_sha256=receipt.sha256)


def test_plan_loader_rejects_forged_ood_and_duplicate_lineages() -> None:
    plan = build_production_corpus_plan(root_seed=260912, shard_count=8)
    document = plan.to_dict()
    document["lineages"][0]["request"]["distribution_regime"] = "ood"
    with pytest.raises(ValueError, match="OOD|digest"):
        CorpusGenerationPlan.from_dict(document)

    document = plan.to_dict()
    document["lineages"][1] = document["lineages"][0]
    with pytest.raises(ValueError, match="unique|digest"):
        CorpusGenerationPlan.from_dict(document)
