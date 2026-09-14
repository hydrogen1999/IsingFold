from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import embedbench.isingfold_corpus_cli as corpus_cli
import embedbench.isingfold_corpus_plan as corpus_plan
import pytest
from embedbench.candidate_bank import canonical_json_bytes, content_digest
from embedbench.isingfold_corpus_cli import main
from embedbench.isingfold_corpus_plan import (
    CandidateScreeningProtocol,
    CorpusGenerationPlan,
    ProspectiveIdentityPolicy,
    ReleaseScope,
    read_corpus_plan,
    write_corpus_plan,
)
from embedbench.isingfold_design import (
    CorpusDesignProfile,
    DifficultyRule,
    MeasurementBudget,
)

SOURCE_SHA256 = "1" * 64


def _required_preflight_args(path: Path) -> list[str]:
    return [
        "--expected-source-sha256",
        SOURCE_SHA256,
        "--preflight",
        str(path),
        "--expected-preflight-sha256",
        "2" * 64,
        "--expected-preflight-record-digest",
        "3" * 64,
        "--expected-identity-map-digest",
        "4" * 64,
        "--expected-generation-provenance-digest",
        "5" * 64,
    ]


def _make_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[Path, str]:
    path = tmp_path / "plan.json"
    assert main(["plan", "--root-seed", "260912", "--shard-count", "8", "--out", str(path)]) == 0
    receipt = json.loads(capsys.readouterr().out)
    return path, receipt["sha256"]


def test_plan_command_publishes_a_pinned_production_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, digest = _make_plan(tmp_path, capsys)

    plan = read_corpus_plan(path, expected_sha256=digest)
    assert len(plan.lineages) == 3082
    assert plan.shard_count == 8


def test_generate_shard_uses_the_exact_pinned_plan_and_explicit_screening_budget(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, digest = _make_plan(tmp_path, capsys)
    captured: dict[str, object] = {}
    sentinel = object()

    def generate(config, requests):
        captured["config"] = config
        captured["requests"] = requests
        return sentinel

    def write(shard, output):
        assert shard is sentinel
        captured["output"] = output
        return SimpleNamespace(shard_index=3, lineage_count=49)

    monkeypatch.setattr("embedbench.isingfold_corpus_cli.generate_corpus_shard", generate)
    monkeypatch.setattr("embedbench.isingfold_corpus_cli.write_corpus_shard", write)
    monkeypatch.setattr(
        "embedbench.isingfold_corpus_cli.read_prospective_preflight",
        lambda *args, **kwargs: SimpleNamespace(),
    )
    destination = tmp_path / "shard-3"

    assert main(
        [
            "generate-shard",
            "--plan",
            str(path),
            "--expected-plan-sha256",
            digest,
            *_required_preflight_args(tmp_path / "preflight.json"),
            "--index",
            "3",
            "--out",
            str(destination),
        ]
    ) == 0

    plan = read_corpus_plan(path, expected_sha256=digest)
    config = captured["config"]
    assert config.shard_index == 3
    assert config.shard_count == 8
    assert config.reads == plan.screening.reads
    assert config.sweeps == plan.screening.sweeps
    assert len(captured["requests"]) == 3082
    assert captured["output"] == destination


def test_merge_command_binds_the_complete_shard_census_to_the_plan(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, digest = _make_plan(tmp_path, capsys)
    shard_root = tmp_path / "shards"
    shard_root.mkdir()
    for index in range(8):
        (shard_root / f"shard-{index:04d}-of-0008").mkdir()
    captured: dict[str, object] = {}

    def merge(shards, output, **kwargs):
        captured.update(shards=shards, output=output, **kwargs)
        return SimpleNamespace(shard_count=8, lineage_count=3082)

    monkeypatch.setattr("embedbench.isingfold_corpus_cli.merge_corpus_shards", merge)
    destination = tmp_path / "release"
    assert main(
        [
            "merge",
            "--plan",
            str(path),
            "--expected-plan-sha256",
            digest,
            "--shard-root",
            str(shard_root),
            "--out",
            str(destination),
        ]
    ) == 0

    plan = read_corpus_plan(path, expected_sha256=digest)
    assert captured["expected_shard_count"] == 8
    assert captured["expected_plan_record_digest"] == plan.record_digest
    assert len(captured["expected_lineage_requests"]) == 3082
    assert captured["output"] == destination


def test_generate_rejects_an_out_of_range_index_before_generation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, digest = _make_plan(tmp_path, capsys)
    with pytest.raises(ValueError, match="shard index"):
        main(
            [
                "generate-shard",
                "--plan",
                str(path),
                "--expected-plan-sha256",
                digest,
                *_required_preflight_args(tmp_path / "unused-preflight.json"),
                "--index",
                "8",
                "--out",
                str(tmp_path / "bad"),
            ]
        )


def test_plan_digest_is_the_hash_of_exact_published_bytes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path, digest = _make_plan(tmp_path, capsys)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()


def test_real_cli_pipeline_publishes_plan_bound_release_and_design(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise every real CLI boundary with a small, explicitly test-only census."""

    root_seed = 260912
    shard_count = 2
    source = corpus_plan.build_production_corpus_plan(
        root_seed=root_seed,
        shard_count=shard_count,
    )
    selected_ordinals = {
        "train": {0, 1, 8, 9, 10, 11},
        "val": {0, 1, 8, 9, 10, 11},
        "test": {0, 1, 8, 9, 16, 17},
    }
    lineages = tuple(
        sorted(
            (
                item
                for item in source.lineages
                if int(item.request.lineage_id.split("-")[-2])
                in selected_ordinals[item.request.partition]
            ),
            key=lambda item: item.request.lineage_id,
        )
    )
    assert len(lineages) == 18

    screening = CandidateScreeningProtocol(
        attempt_slots=10,
        incumbent_search_slots=4,
        max_split_search=1024,
        strengths=(1.0,),
        reads=2,
        sweeps=5,
    )
    budget = MeasurementBudget(
        measurement_budget_id="isingfold-cli-integration-smoke-v1",
        decision_evaluations=screening.attempt_slots,
        embedding_attempts=screening.attempt_slots + screening.incumbent_search_slots,
        sampler_reads=screening.reads * len(screening.strengths) * 2,
    )
    rules = (
        DifficultyRule(
            field="decision_difficulty",
            metric="candidate_resource_spread",
            hard_if="greater-than-or-equal",
            threshold=0.01,
        ),
        DifficultyRule(
            field="embedding_difficulty",
            metric="host_fill_fraction",
            hard_if="greater-than-or-equal",
            threshold=0.08,
        ),
        DifficultyRule(
            field="sampling_difficulty",
            metric="logical_edge_density",
            hard_if="greater-than-or-equal",
            threshold=0.5,
        ),
    )
    prospective_identity_policy = ProspectiveIdentityPolicy()
    release_scope = ReleaseScope()
    payload = {
        "difficulty_rules": [rule.as_dict() for rule in rules],
        "lineages": [item.to_dict() for item in lineages],
        "measurement_budget": asdict(budget),
        "prospective_identity_policy": prospective_identity_policy.to_dict(),
        "release_scope": release_scope.to_dict(),
        "root_seed": root_seed,
        "schema": "embedbench.isingfold-corpus-generation-plan",
        "schema_version": 2,
        "screening": screening.to_dict(),
        "shard_count": shard_count,
    }
    monkeypatch.setattr(
        corpus_plan,
        "_PARTITION_COUNTS",
        {"train": 6, "val": 6, "test": 6},
    )
    plan = CorpusGenerationPlan(
        root_seed=root_seed,
        shard_count=shard_count,
        screening=screening,
        measurement_budget=budget,
        difficulty_rules=rules,
        lineages=lineages,
        prospective_identity_policy=prospective_identity_policy,
        release_scope=release_scope,
        record_digest=content_digest(payload),
    )
    plan_path = tmp_path / "plan.json"
    plan_write = write_corpus_plan(plan, plan_path)
    assert plan_write.sha256 == hashlib.sha256(plan_path.read_bytes()).hexdigest()

    preflight_path = tmp_path / "prospective-preflight.json"
    assert main(
        [
            "preflight",
            "--plan",
            str(plan_path),
            "--expected-plan-sha256",
            plan_write.sha256,
            "--expected-source-sha256",
            SOURCE_SHA256,
            "--workers",
            "1",
            "--out",
            str(preflight_path),
        ]
    ) == 0
    preflight_receipt = json.loads(capsys.readouterr().out)
    required_preflight_args = [
        "--expected-source-sha256",
        SOURCE_SHA256,
        "--preflight",
        str(preflight_path),
        "--expected-preflight-sha256",
        preflight_receipt["preflight_sha256"],
        "--expected-preflight-record-digest",
        preflight_receipt["preflight_record_digest"],
        "--expected-identity-map-digest",
        preflight_receipt["prospective_identity_map_digest"],
        "--expected-generation-provenance-digest",
        preflight_receipt["generation_provenance_digest"],
    ]

    shard_root = tmp_path / "shards"
    shard_receipts = []
    for index in range(shard_count):
        shard_path = shard_root / f"shard-{index:04d}-of-{shard_count:04d}"
        assert main(
            [
                "generate-shard",
                "--plan",
                str(plan_path),
                "--expected-plan-sha256",
                plan_write.sha256,
                *required_preflight_args,
                "--index",
                str(index),
                "--out",
                str(shard_path),
            ]
        ) == 0
        receipt = json.loads(capsys.readouterr().out)
        assert receipt["shard_sha256"] == hashlib.sha256(
            (shard_path / "corpus_shard.json").read_bytes()
        ).hexdigest()
        shard_receipts.append(receipt)
    assert [receipt["shard_index"] for receipt in shard_receipts] == [0, 1]
    assert sum(receipt["lineage_count"] for receipt in shard_receipts) == len(lineages)

    release = tmp_path / "release"
    assert main(
        [
            "merge",
            "--plan",
            str(plan_path),
            "--expected-plan-sha256",
            plan_write.sha256,
            "--shard-root",
            str(shard_root),
            "--out",
            str(release),
        ]
    ) == 0
    merge_receipt = json.loads(capsys.readouterr().out)
    release_manifest = release / "release_manifest.json"
    release_manifest_sha256 = hashlib.sha256(release_manifest.read_bytes()).hexdigest()
    assert merge_receipt["release_manifest_sha256"] == release_manifest_sha256
    assert merge_receipt["lineage_count"] == len(lineages)
    assert merge_receipt["shard_count"] == shard_count
    release_document = json.loads(release_manifest.read_bytes())
    assert release_document["source_plan"] == {
        "lineage_request_count": len(lineages),
        "record_digest": plan.record_digest,
    }

    real_build_design = corpus_cli.build_corpus_design_v2

    def build_test_design(**kwargs: object):
        return real_build_design(
            **kwargs,
            profile=CorpusDesignProfile.explicit_test(
                train_floor=6,
                validation_floor=6,
                test_floor=6,
                validation_tuning_minimum=1,
                reason="real CLI integration fixture over generated corpus shards",
            ),
        )

    monkeypatch.setattr(corpus_cli, "build_corpus_design_v2", build_test_design)
    design_root = tmp_path / "design"
    assert main(
        [
            "design",
            "--plan",
            str(plan_path),
            "--expected-plan-sha256",
            plan_write.sha256,
            "--release",
            str(release),
            "--expected-release-manifest-sha256",
            release_manifest_sha256,
            "--publisher-id",
            "embedbench-cli-integration-authority",
            "--source-release-id",
            "embedbench-cli-integration-release",
            "--out",
            str(design_root),
        ]
    ) == 0
    design_receipt = json.loads(capsys.readouterr().out)
    design_path = design_root / "corpus_design_v2.json"
    publication_path = design_root / "publication-index.json"
    design_document = json.loads(design_path.read_bytes())
    publication_document = json.loads(publication_path.read_bytes())

    assert design_path.read_bytes() == canonical_json_bytes(design_document) + b"\n"
    assert design_document["schema"] == "isingfold.corpus-design"
    assert design_document["schema_version"] == 2
    assert design_document["partition_quotas"] == {"test": 6, "train": 6, "val": 6}
    assert design_receipt["test_only"] is True
    assert design_receipt["base_lineage_count"] == len(lineages)
    assert design_receipt["task_count"] == len(lineages)
    assert design_receipt["corpus_design_sha256"] == hashlib.sha256(
        design_path.read_bytes()
    ).hexdigest()
    assert design_receipt["publication_index_sha256"] == hashlib.sha256(
        publication_path.read_bytes()
    ).hexdigest()
    assert publication_document["source_release_manifest_sha256"] == (
        release_manifest_sha256
    )
    assert publication_document["split_manifest_sha256"] == plan_write.sha256
    assert publication_document["corpus_design"]["sha256"] == design_receipt[
        "corpus_design_sha256"
    ]
