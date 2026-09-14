"""Artifact-contract tests for sharded exact quality replay."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from isingfold.rl.data.import_embedbench import content_digest
from isingfold.rl.data.quality_preflight import (
    QualityPreflightError,
    build_quality_preflight_plan,
    load_quality_preflight_plan,
    load_trusted_quality_preflight_replay_bundle,
    merge_quality_preflight_shards,
    publish_quality_preflight_shard,
)


PIN = "a" * 64


def _row(index: int, trajectories: int = 4) -> dict[str, object]:
    return {
        "lineage": f"lineage-{index}",
        "source_row_record_digest": f"{index + 1:064x}",
        "task_id": f"task-{index}",
        "trajectory_count": trajectories,
    }


def _source(index: int, row: dict[str, object]) -> dict[str, object]:
    return {
        "manifest_record_digest": f"{index + 10:064x}",
        "manifest_sha256": f"{index + 20:064x}",
        "records_sha256": f"{index + 30:064x}",
        "rows": [row],
        "selected_lineages": [row["lineage"]],
        "selected_task_ids": [row["task_id"]],
        "shard_index": index,
    }


def _identity() -> dict[str, object]:
    return {
        "authority": {
            "ground_partition_receipt": {"record_digest": "1" * 64},
            "quality_authority": {"record_digest": "2" * 64},
            "target_access": {"record_digest": "3" * 64},
        },
        "context": {"digest": "4" * 64},
        "corpus": {"manifest_sha256": "5" * 64},
        "device_parity": {"raw_sha256": "6" * 64},
        "implementation": {"quality_contract_digest": "7" * 64},
        "resolution": {
            "production_protocol_digest": "8" * 64,
            "record_digest": "9" * 64,
            "selected_continuations": 12,
            "sha256": "a" * 64,
        },
        "selector": {"selector_digest": "b" * 64},
    }


def _make_plan(tmp_path: Path):
    rows = [_row(0), _row(1, trajectories=8)]
    path = tmp_path / "plan.json"
    sha = build_quality_preflight_plan(
        source_shards=[_source(index, row) for index, row in enumerate(rows)],
        shard_count=2,
        identity=_identity(),
        output_path=path,
    )
    return load_quality_preflight_plan(path, expected_sha256=sha), sha, rows


def _evidence(row: dict[str, object]) -> dict[str, object]:
    expected = content_digest(
        [
            {
                "action_index": 0,
                "continuation_index": index,
                "receipt_digest": f"{index + 40:064x}",
                "seed": index + 100,
            }
            for index in range(int(row["trajectory_count"]))
        ]
    )
    return {
        "expected_continuation_root_digest": expected,
        "lineage": row["lineage"],
        "match_count": row["trajectory_count"],
        "pass": True,
        "replayed_continuation_root_digest": expected,
        "source_row_record_digest": row["source_row_record_digest"],
        "task_id": row["task_id"],
        "trajectory_count": row["trajectory_count"],
    }


def test_plan_is_externally_pinned_and_rejects_duplicate_row_coverage(tmp_path: Path) -> None:
    plan, sha, rows = _make_plan(tmp_path)

    assert plan.raw_sha256 == sha
    assert plan.as_dict()["row_count"] == 2
    assert plan.as_dict()["trajectory_count"] == 12
    duplicate_digest = {
        **rows[1],
        "source_row_record_digest": rows[0]["source_row_record_digest"],
    }
    with pytest.raises(QualityPreflightError, match="source row.*duplicate"):
        build_quality_preflight_plan(
            source_shards=[_source(0, rows[0]), _source(1, duplicate_digest)],
            shard_count=2,
            identity=_identity(),
            output_path=tmp_path / "duplicate.json",
        )


def test_replay_shards_merge_to_a_complete_pinned_bundle(tmp_path: Path) -> None:
    plan, plan_sha, rows = _make_plan(tmp_path)
    pins: list[tuple[Path, str]] = []
    for index, row in enumerate(rows):
        root = tmp_path / f"replay-{index}"
        manifest_sha = publish_quality_preflight_shard(
            plan,
            expected_plan_sha256=plan_sha,
            shard_index=index,
            evidence=[_evidence(row)],
            runtime_identity={"host_class": "fixture", "runtime_sha256": PIN},
            output_directory=root,
        )
        pins.append((root, manifest_sha))

    bundle_path = tmp_path / "bundle.json"
    bundle_sha = merge_quality_preflight_shards(
        plan,
        expected_plan_sha256=plan_sha,
        replay_shards=pins,
        output_path=bundle_path,
    )
    bundle = load_trusted_quality_preflight_replay_bundle(
        bundle_path,
        expected_sha256=bundle_sha,
        source_shards=[_source(index, row) for index, row in enumerate(rows)],
    )

    assert bundle["pass"] is True
    assert bundle["trajectory_count"] == 12
    assert bundle["matching_count"] == 12
    assert set(bundle["trusted_rows"]) == {
        rows[0]["source_row_record_digest"],
        rows[1]["source_row_record_digest"],
    }
    assert hashlib.sha256(bundle_path.read_bytes()).hexdigest() == bundle_sha


def test_replay_bundle_rejects_missing_or_unpinned_shards(tmp_path: Path) -> None:
    plan, plan_sha, rows = _make_plan(tmp_path)
    root = tmp_path / "replay-0"
    manifest_sha = publish_quality_preflight_shard(
        plan,
        expected_plan_sha256=plan_sha,
        shard_index=0,
        evidence=[_evidence(rows[0])],
        runtime_identity={"host_class": "fixture", "runtime_sha256": PIN},
        output_directory=root,
    )

    with pytest.raises(QualityPreflightError, match="missing replay shard"):
        merge_quality_preflight_shards(
            plan,
            expected_plan_sha256=plan_sha,
            replay_shards=[(root, manifest_sha)],
            output_path=tmp_path / "incomplete.json",
        )
    with pytest.raises(QualityPreflightError, match="externally pinned"):
        merge_quality_preflight_shards(
            plan,
            expected_plan_sha256=plan_sha,
            replay_shards=[(root, "f" * 64), (root, manifest_sha)],
            output_path=tmp_path / "bad-pin.json",
        )


def test_quality_loader_emits_exact_evidence_and_accepts_only_row_scoped_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import isingfold.rl.cli as cli
    from isingfold.rl.data import quality
    from tests.unit.test_rl_quality_integrity import (
        _make_quality_artifact,
        _pin,
    )

    corpus, labels, selector, context = _make_quality_artifact(tmp_path, monkeypatch)
    evidence: list[dict[str, object]] = []
    _decoded, manifest = cli._load_quality_labels(
        labels,
        corpus=corpus,
        selector=selector,
        context=context,
        enforce_resolution=False,
        quality_attestation_pin=_pin(tmp_path, corpus=corpus),
        _replay_evidence=evidence,
        allow_diagnostic_legacy=True,
    )
    assert len(evidence) == manifest["record_count"]
    assert all(item["pass"] is True for item in evidence)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("trusted replay unexpectedly executed a continuation")

    monkeypatch.setattr(quality, "run_continuation", forbidden)
    monkeypatch.setattr(
        cli,
        "_quality_implementation_contract",
        lambda _context: manifest["label_protocol"]["implementation_contract"],
    )
    trusted = {}
    for item in evidence:
        body = {
            "schema": "isingfold.quality-preflight-row-evidence",
            "schema_version": 1,
            **item,
        }
        record = {**body, "record_digest": content_digest(body)}
        trusted[str(item["source_row_record_digest"])] = record
    _decoded, trusted_manifest = cli._load_quality_labels(
        labels,
        corpus=corpus,
        selector=selector,
        context=context,
        enforce_resolution=False,
        quality_attestation_pin=_pin(tmp_path, corpus=corpus),
        _trusted_replay_rows=trusted,
        allow_diagnostic_legacy=True,
    )
    summary = trusted_manifest["loader_summary"]
    assert summary["continuation_replay_mode"] == "pinned-quality-replay-bundle"
    assert summary["continuation_replays_executed"] == 0
    assert (
        summary["continuation_replay_matches"] == summary["continuation_trajectories_authenticated"]
    )


def test_parallel_preflight_aggregation_reconstructs_the_global_denominators() -> None:
    import isingfold.rl.cli as cli

    base = {
        "minimum_resolved_rows": 2,
        "minimum_resolved_lineages": 2,
        "passes_resolution": False,
        "continuation_replay_mode": "full-exact-continuation-and-evaluator",
    }
    summaries = [
        {
            **base,
            "records_total": 2,
            "records_used": 1,
            "fully_unresolved_skipped": 1,
            "label_lineages": 1,
            "resolved_lineages": 1,
            "resolved_tasks": 1,
            "unresolved_lineages": [],
            "continuation_trajectories_authenticated": 8,
            "continuation_replays_executed": 8,
            "continuation_replay_matches": 8,
        },
        {
            **base,
            "records_total": 1,
            "records_used": 1,
            "fully_unresolved_skipped": 0,
            "label_lineages": 1,
            "resolved_lineages": 1,
            "resolved_tasks": 1,
            "unresolved_lineages": [],
            "continuation_trajectories_authenticated": 4,
            "continuation_replays_executed": 4,
            "continuation_replay_matches": 4,
        },
    ]
    manifest = {
        "independent_denominators": {
            "continuation_trajectories": 12,
            "fully_unresolved_rows": 1,
            "lineages_with_records": 2,
            "records_total": 3,
            "resolved_rows": 2,
            "resolved_lineages": 2,
        }
    }

    aggregate = cli._aggregate_quality_preflight_summaries(
        summaries,
        manifest=manifest,
        min_resolved_rows=2,
        min_resolved_lineages=2,
    )

    assert aggregate["passes_resolution"] is True
    assert aggregate["records_total"] == 3
    assert aggregate["continuation_replays_executed"] == 12


def test_quality_merge_consumes_the_complete_bundle_without_replaying_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import isingfold.rl.cli as cli
    from isingfold.rl.data import quality
    from tests.unit.test_rl_quality_integrity import (
        _make_quality_shards,
        _merge_args,
        _pin,
    )

    corpus, quality_shards, selector, context = _make_quality_shards(
        tmp_path, monkeypatch, lineage_count=2, shard_count=2
    )
    headers = cli._quality_shard_headers(
        [str(root) for root in quality_shards],
        allow_diagnostic_legacy=True,
    )
    sources = cli._quality_preflight_source_shards(
        headers,
        expected_manifest_sha256=[header[3] for header in headers],
        allow_diagnostic_legacy=True,
    )
    plan_path = tmp_path / "preflight-plan.json"
    plan_sha = build_quality_preflight_plan(
        source_shards=sources,
        shard_count=2,
        identity=_identity(),
        output_path=plan_path,
    )
    plan = load_quality_preflight_plan(plan_path, expected_sha256=plan_sha)
    replay_pins: list[tuple[Path, str]] = []
    for index, quality_root in enumerate(quality_shards):
        evidence: list[dict[str, object]] = []
        cli._load_quality_labels(
            quality_root,
            corpus=corpus,
            selector=selector,
            context=context,
            enforce_resolution=False,
            quality_attestation_pin=_pin(tmp_path, corpus=corpus),
            _allow_shard=True,
            _replay_evidence=evidence,
            allow_diagnostic_legacy=True,
        )
        replay_root = tmp_path / f"replay-{index}"
        replay_sha = publish_quality_preflight_shard(
            plan,
            expected_plan_sha256=plan_sha,
            shard_index=index,
            evidence=evidence,
            runtime_identity={"host_class": "fixture", "runtime_sha256": PIN},
            output_directory=replay_root,
        )
        replay_pins.append((replay_root, replay_sha))
    bundle_path = tmp_path / "bundle.json"
    bundle_sha = merge_quality_preflight_shards(
        plan,
        expected_plan_sha256=plan_sha,
        replay_shards=replay_pins,
        output_path=bundle_path,
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("trusted merge unexpectedly replayed a continuation")

    common_manifest = cli._strict_json(quality_shards[0] / "manifest.json")
    monkeypatch.setattr(quality, "run_continuation", forbidden)
    monkeypatch.setattr(
        cli,
        "_quality_implementation_contract",
        lambda _context: common_manifest["label_protocol"]["implementation_contract"],
    )
    merged = tmp_path / "merged"
    args = _merge_args(corpus, quality_shards, merged)
    args.trusted_replay_bundle = str(bundle_path)
    args.expected_trusted_replay_bundle_sha256 = bundle_sha
    args.func(args)

    assert (merged / "manifest.json").is_file()
