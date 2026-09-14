from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
from embedbench.candidate_bank import (
    BankManifest,
    canonical_json_bytes,
    content_digest,
    read_bank,
)
from embedbench.isingfold_corpus_io import merge_corpus_shards, write_corpus_shard
from embedbench.isingfold_corpus_shard import (
    CorpusShard,
    LineageRequest,
    ShardConfig,
    generate_corpus_shard,
)
from embedbench.reference_export import export_problem_references


def _request(index: int) -> LineageRequest:
    lineage_id = ("io-lineage-0-0", "io-lineage-1-2")[index]
    return LineageRequest(
        lineage_id=lineage_id,
        application_family="graphcut",
        origin="application-derived",
        partition="train",
        distribution_regime="iid",
        topology="chimera",
        host_size=2,
        n_variables=3,
    )


def _shard(index: int, count: int = 2, root_seed: int = 260912) -> CorpusShard:
    config = ShardConfig(
        root_seed=root_seed,
        shard_index=index,
        shard_count=count,
        attempt_slots=10,
        incumbent_search_slots=4,
        max_split_search=128,
        strengths=(1.0,),
        reads=2,
        sweeps=5,
    )
    request = _request(index)
    return generate_corpus_shard(config, (request,))


def _json(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert raw == canonical_json_bytes(value) + b"\n"
    return value


def _checksums(path: Path) -> dict[str, str]:
    return {
        filename: digest
        for digest, filename in (
            line.split("  ", 1)
            for line in path.read_text(encoding="utf-8").splitlines()
        )
    }


def _reauthenticate_shard_directory(directory: Path) -> None:
    shard_raw = (directory / "corpus_shard.json").read_bytes()
    manifest = json.loads((directory / "shard_manifest.json").read_bytes())
    manifest["artifact"] = {
        "byte_count": len(shard_raw),
        "path": "corpus_shard.json",
        "sha256": hashlib.sha256(shard_raw).hexdigest(),
    }
    manifest["record_digest"] = content_digest(
        {key: value for key, value in manifest.items() if key != "record_digest"}
    )
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    (directory / "shard_manifest.json").write_bytes(manifest_raw)
    authenticated = {
        "corpus_shard.json": shard_raw,
        "shard_manifest.json": manifest_raw,
    }
    (directory / "SHA256SUMS").write_text(
        "".join(
            f"{hashlib.sha256(authenticated[name]).hexdigest()}  {name}\n"
            for name in sorted(authenticated)
        ),
        encoding="utf-8",
    )


def _jsonl(path: Path) -> list[dict[str, object]]:
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    assert raw == b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    return rows


def test_corpus_shard_writer_is_atomic_and_authenticated(tmp_path: Path) -> None:
    shard = _shard(0)
    destination = tmp_path / "shard-0"

    receipt = write_corpus_shard(shard, destination)

    assert {item.name for item in destination.iterdir()} == {
        "SHA256SUMS",
        "corpus_shard.json",
        "shard_manifest.json",
    }
    document = _json(destination / "corpus_shard.json")
    assert canonical_json_bytes(CorpusShard.from_dict(document).to_dict()) == canonical_json_bytes(
        shard.to_dict()
    )
    manifest = _json(destination / "shard_manifest.json")
    assert manifest["record_digest"] == content_digest(
        {key: value for key, value in manifest.items() if key != "record_digest"}
    )
    assert manifest["shard_record_digest"] == shard.record_digest
    assert manifest["accounting"]["attempts"] == len(shard.lineages[0].group.attempts)
    checksums = _checksums(destination / "SHA256SUMS")
    assert list(checksums) == ["corpus_shard.json", "shard_manifest.json"]
    assert checksums == {
        filename: hashlib.sha256((destination / filename).read_bytes()).hexdigest()
        for filename in checksums
    }
    assert receipt.output_directory == destination.resolve()
    assert receipt.shard_index == 0
    assert receipt.lineage_count == 1

    with pytest.raises(FileExistsError, match="already exists"):
        write_corpus_shard(shard, destination)


def test_merge_is_complete_deterministic_and_consumable_by_existing_boundaries(
    tmp_path: Path,
) -> None:
    shards = [tmp_path / f"shard-{index}" for index in range(2)]
    generated = [_shard(index) for index in range(2)]
    for shard, destination in zip(generated, shards, strict=True):
        write_corpus_shard(shard, destination)

    first = tmp_path / "merged-first"
    second = tmp_path / "merged-second"
    receipt = merge_corpus_shards(
        tuple(reversed(shards)),
        first,
        expected_shard_count=2,
        expected_plan_record_digest="a" * 64,
        expected_lineage_requests=tuple(_request(index) for index in range(2)),
    )
    merge_corpus_shards(
        shards,
        second,
        expected_shard_count=2,
        expected_plan_record_digest="a" * 64,
        expected_lineage_requests=tuple(_request(index) for index in range(2)),
    )

    expected_files = {
        "SHA256SUMS",
        "candidate_bank_v2.jsonl",
        "candidate_bank_v2.manifest.json",
        "lineage_facts.jsonl",
        "quality_isingfold_exact.jsonl",
        "quality_isingfold_exact.jsonl.manifest.json",
        "release_manifest.json",
        "task_facts.jsonl",
    }
    assert {path.name for path in first.iterdir()} == expected_files
    assert {
        name: (first / name).read_bytes() for name in expected_files
    } == {name: (second / name).read_bytes() for name in expected_files}

    bank_manifest = BankManifest(**_json(first / "candidate_bank_v2.manifest.json"))
    instances, groups = read_bank(first / "candidate_bank_v2.jsonl", bank_manifest)
    assert len(instances) == len(groups) == 2
    assert {group.instance_id for group in groups} == {
        instance.instance_id for instance in instances
    }
    assert sum(len(group.attempts) for group in groups) == 20

    lineage_rows = _jsonl(first / "lineage_facts.jsonl")
    task_rows = _jsonl(first / "task_facts.jsonl")
    assert [row["schema"] for row in lineage_rows] == [
        "embedbench.isingfold-lineage-fact"
    ] * 2
    assert [row["schema"] for row in task_rows] == [
        "embedbench.isingfold-task-fact"
    ] * 2
    for row in (*lineage_rows, *task_rows):
        assert row["record_digest"] == content_digest(
            {key: value for key, value in row.items() if key != "record_digest"}
        )

    quality_rows = _jsonl(first / "quality_isingfold_exact.jsonl")
    assert len(quality_rows) == 2
    assert all(set(row) == {"instance_id", "mode", "problem"} for row in quality_rows)
    quality_manifest = _json(first / "quality_isingfold_exact.jsonl.manifest.json")
    assert quality_manifest["record_count"] == 2
    assert len(quality_manifest["references"]) == 2
    assert quality_manifest["record_digest"] == content_digest(
        {key: value for key, value in quality_manifest.items() if key != "record_digest"}
    )
    exported = export_problem_references(
        [first / "quality_isingfold_exact.jsonl"],
        checksums_path=first / "SHA256SUMS",
        output_dir=tmp_path / "references",
    )
    assert exported["certified_problems"] == 2

    release = _json(first / "release_manifest.json")
    assert release["record_digest"] == content_digest(
        {key: value for key, value in release.items() if key != "record_digest"}
    )
    assert release["counts"] == {
        "candidate_groups": 2,
        "instances": 2,
        "lineage_facts": 2,
        "lineages": 2,
        "problems": 2,
        "split_units": 2,
        "task_facts": 2,
    }
    assert release["accounting"]["attempts"] == 20
    assert release["source_plan"] == {
        "lineage_request_count": 2,
        "record_digest": "a" * 64,
    }
    checksums = _checksums(first / "SHA256SUMS")
    assert set(checksums) == expected_files - {"SHA256SUMS"}
    assert checksums == {
        filename: hashlib.sha256((first / filename).read_bytes()).hexdigest()
        for filename in checksums
    }
    assert receipt.output_directory == first.resolve()
    assert receipt.shard_count == receipt.lineage_count == 2
    assert receipt.release_manifest_sha256 == hashlib.sha256(
        (first / "release_manifest.json").read_bytes()
    ).hexdigest()


def test_merge_rejects_incomplete_duplicate_or_mixed_shard_sets(tmp_path: Path) -> None:
    shard_zero = tmp_path / "shard-zero"
    shard_one = tmp_path / "shard-one"
    write_corpus_shard(_shard(0), shard_zero)
    write_corpus_shard(_shard(1), shard_one)

    with pytest.raises(ValueError, match="shard-index coverage.*exact"):
        merge_corpus_shards(
            [shard_zero],
            tmp_path / "missing",
            expected_shard_count=2,
        )
    with pytest.raises(ValueError, match="shard-index coverage.*exact"):
        merge_corpus_shards(
            [shard_zero, shard_zero],
            tmp_path / "duplicate",
            expected_shard_count=2,
        )

    mixed = tmp_path / "mixed-seed-one"
    write_corpus_shard(_shard(1, root_seed=260913), mixed)
    with pytest.raises(ValueError, match="different generation configurations"):
        merge_corpus_shards(
            [shard_zero, mixed],
            tmp_path / "mixed",
            expected_shard_count=2,
        )


def test_merge_rejects_a_shard_census_that_differs_from_the_prospective_plan(
    tmp_path: Path,
) -> None:
    shards = [tmp_path / f"shard-{index}" for index in range(2)]
    for index, destination in enumerate(shards):
        write_corpus_shard(_shard(index), destination)
    forged = replace(_request(1), lineage_id="unplanned-lineage")

    with pytest.raises(ValueError, match="prospective plan"):
        merge_corpus_shards(
            shards,
            tmp_path / "plan-mismatch",
            expected_shard_count=2,
            expected_plan_record_digest="b" * 64,
            expected_lineage_requests=(_request(0), forged),
        )


def test_merge_rejects_reauthenticated_noncanonical_or_semantic_tampering(
    tmp_path: Path,
) -> None:
    noncanonical = tmp_path / "noncanonical"
    write_corpus_shard(_shard(0), noncanonical)
    shard_path = noncanonical / "corpus_shard.json"
    shard_path.write_bytes(shard_path.read_bytes() + b" ")
    _reauthenticate_shard_directory(noncanonical)
    with pytest.raises(ValueError, match="not canonical JSON"):
        merge_corpus_shards(
            [noncanonical],
            tmp_path / "noncanonical-output",
            expected_shard_count=1,
        )

    semantic = tmp_path / "semantic"
    write_corpus_shard(_shard(0), semantic)
    document = json.loads((semantic / "corpus_shard.json").read_bytes())
    document["lineages"][0]["exact_reference"]["energy"] += 1.0
    document["record_digest"] = content_digest(
        {key: value for key, value in document.items() if key != "record_digest"}
    )
    (semantic / "corpus_shard.json").write_bytes(canonical_json_bytes(document) + b"\n")
    manifest = json.loads((semantic / "shard_manifest.json").read_bytes())
    manifest["shard_record_digest"] = document["record_digest"]
    (semantic / "shard_manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
    _reauthenticate_shard_directory(semantic)
    with pytest.raises(ValueError, match="exact reference|reference identities"):
        merge_corpus_shards(
            [semantic],
            tmp_path / "semantic-output",
            expected_shard_count=1,
        )


def test_merge_rejects_unaccounted_files_and_writer_cleans_failed_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shard_dir = tmp_path / "unexpected"
    write_corpus_shard(_shard(0), shard_dir)
    (shard_dir / "untracked.txt").write_text("not authenticated", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory differs"):
        merge_corpus_shards(
            [shard_dir],
            tmp_path / "unexpected-output",
            expected_shard_count=1,
        )

    import embedbench.isingfold_corpus_io as corpus_io

    real_write = corpus_io._write_new
    calls = 0

    def fail_second_write(path: Path, raw: bytes) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected staged-write failure")
        real_write(path, raw)

    monkeypatch.setattr(corpus_io, "_write_new", fail_second_write)
    destination = tmp_path / "failed-publication"
    with pytest.raises(RuntimeError, match="injected"):
        write_corpus_shard(_shard(0), destination)
    assert not destination.exists()
    assert not list(tmp_path.glob(".failed-publication.staging-*"))
