from __future__ import annotations

import hashlib
import json
from pathlib import Path

import embedbench.isingfold_corpus_preflight as preflight
import pytest
from embedbench.candidate_bank import canonical_json_bytes, content_digest, derive_split_unit_id
from embedbench.isingfold_corpus_cli import main
from embedbench.isingfold_corpus_shard import ProspectiveLineage

SOURCE_SHA256 = "1" * 64


def _write_plan(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> tuple[Path, str]:
    path = tmp_path / "corpus_plan.json"
    assert main(["plan", "--out", str(path)]) == 0
    return path, json.loads(capsys.readouterr().out)["sha256"]


def _slot_for(request: object) -> int:
    partition = {"train": 0, "val": 10, "test": 20}[request.partition]
    topology = {"chimera": 0, "pegasus": 2, "zephyr": 4}[request.topology]
    origin = 0 if request.origin == "application-derived" else 1
    return partition + topology + origin


def _fake_prospect(request: object, *, identity: str | None = None) -> ProspectiveLineage:
    nodes = (0, 1)
    edges = ((0, 1),)
    token = identity or request.lineage_id
    coefficient = int(content_digest({"prospective-test-token": token})[:12], 16) / 2**48
    h = ((0, coefficient), (1, -coefficient))
    j = ((0, 1, 1.0),)
    return ProspectiveLineage(
        request=request,
        prospective_seed=0,
        prospective_slot=_slot_for(request),
        split_unit_id=derive_split_unit_id(
            logical_nodes=nodes,
            logical_edges=edges,
            h=h,
            j=j,
        ),
        logical_nodes=nodes,
        logical_edges=edges,
        h=h,
        j=j,
        origin_metadata={},
        realized_host_artifact={},
    )


def test_preflight_command_publishes_a_reproducible_plan_bound_census(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, plan_sha256 = _write_plan(tmp_path, capsys)
    monkeypatch.setattr(
        preflight,
        "prospect_lineage",
        lambda config, request: _fake_prospect(request),
    )
    first = tmp_path / "preflight-a.json"
    second = tmp_path / "preflight-b.json"

    for destination in (first, second):
        assert main(
            [
                "preflight",
                "--plan",
                str(plan_path),
                    "--expected-plan-sha256",
                    plan_sha256,
                    "--expected-source-sha256",
                    SOURCE_SHA256,
                "--workers",
                "1",
                "--out",
                str(destination),
            ]
        ) == 0
        emitted = json.loads(capsys.readouterr().out)
        assert emitted["preflight_sha256"] == hashlib.sha256(
            destination.read_bytes()
        ).hexdigest()

    assert first.read_bytes() == second.read_bytes()
    document = json.loads(first.read_bytes())
    assert first.read_bytes() == canonical_json_bytes(document) + b"\n"
    assert set(document) == {
        "census",
        "generation_provenance",
        "lineage_count",
        "plan",
        "protocol",
        "prospective_identity_map",
        "record_digest",
        "schema",
        "schema_version",
        "source",
        "slot_statistics",
    }
    assert document["schema"] == "embedbench.isingfold-prospective-preflight"
    assert document["schema_version"] == 2
    assert document["plan"] == {
        "record_digest": json.loads(plan_path.read_bytes())["record_digest"],
        "sha256": plan_sha256,
    }
    assert document["source"] == {"sha256": SOURCE_SHA256}
    assert document["lineage_count"] == 3082
    assert document["protocol"] == {
        "operation": "prospect-lineage-only-no-proposal-or-quality-v2",
        "quantile_definition": "nearest-rank-ceiling-v1",
        "search_slots": 1024,
    }
    identity_map = document["prospective_identity_map"]
    assert identity_map["count"] == 3082
    assert identity_map["schema"] == "embedbench.isingfold-prospective-identity-map"
    assert identity_map["schema_version"] == 1
    assert identity_map["protocol"] == "canonical-prospective-identity-map-v1"
    assert len(identity_map["entries"]) == 3082
    assert identity_map["entries"] == sorted(
        identity_map["entries"], key=lambda item: item["lineage_id"]
    )
    assert len({item["split_unit_id"] for item in identity_map["entries"]}) == 3082
    assert len({item["problem_sha256"] for item in identity_map["entries"]}) == 3082
    assert identity_map["record_digest"] == content_digest(
        {key: value for key, value in identity_map.items() if key != "record_digest"}
    )
    assert document["slot_statistics"] == {
        "maximum": 25,
        "minimum": 0,
        "quantiles": [
            {"percentile": 50, "slot": 20},
            {"percentile": 90, "slot": 24},
            {"percentile": 95, "slot": 25},
            {"percentile": 99, "slot": 25},
        ],
    }
    assert len(document["census"]) == 18
    assert sum(row["lineage_count"] for row in document["census"]) == 3082
    partition_order = {"train": 0, "val": 1, "test": 2}
    topology_order = {"chimera": 0, "pegasus": 1, "zephyr": 2}
    origin_order = {"application-derived": 0, "synthetic-ink-drop": 1}
    assert document["census"] == sorted(
        document["census"],
        key=lambda row: (
            partition_order[row["partition"]],
            topology_order[row["topology"]],
            origin_order[row["origin"]],
        ),
    )
    assert document["record_digest"] == content_digest(
        {key: value for key, value in document.items() if key != "record_digest"}
    )
    assert not ({"hostname", "timestamp", "workers"} & set(document))

    assert main(
        [
            "verify-preflight",
            "--plan",
            str(plan_path),
            "--expected-plan-sha256",
            plan_sha256,
            "--expected-source-sha256",
            SOURCE_SHA256,
            "--preflight",
            str(first),
            "--expected-preflight-sha256",
            hashlib.sha256(first.read_bytes()).hexdigest(),
            "--expected-preflight-record-digest",
            document["record_digest"],
            "--expected-identity-map-digest",
            identity_map["record_digest"],
            "--expected-generation-provenance-digest",
            document["generation_provenance"]["record_digest"],
            "--workers",
            "1",
        ]
    ) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["prospective_identity_count"] == 3082
    assert verified["prospective_identity_map_digest"] == identity_map["record_digest"]


def test_preflight_fails_closed_without_partial_or_overwritten_receipts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, plan_sha256 = _write_plan(tmp_path, capsys)
    failing_id = json.loads(plan_path.read_bytes())["lineages"][-1]["request"]["lineage_id"]

    def fail_one(config: object, request: object) -> object:
        if request.lineage_id == failing_id:
            raise ValueError("deterministic prospective exhaustion")
        return _fake_prospect(request)

    monkeypatch.setattr(preflight, "prospect_lineage", fail_one)
    destination = tmp_path / "prospective_preflight.json"
    with pytest.raises(ValueError, match=f"1 unresolved lineage.*{failing_id}"):
        main(
            [
                "preflight",
                "--plan",
                str(plan_path),
                "--expected-plan-sha256",
                plan_sha256,
                "--expected-source-sha256",
                SOURCE_SHA256,
                "--workers",
                "1",
                "--out",
                str(destination),
            ]
        )
    assert not destination.exists()

    destination.write_bytes(b"do-not-overwrite\n")
    monkeypatch.setattr(
        preflight,
        "prospect_lineage",
        lambda config, request: _fake_prospect(request),
    )
    with pytest.raises(FileExistsError, match="already exists"):
        main(
            [
                "preflight",
                "--plan",
                str(plan_path),
                "--expected-plan-sha256",
                plan_sha256,
                "--expected-source-sha256",
                SOURCE_SHA256,
                "--workers",
                "1",
                "--out",
                str(destination),
            ]
        )
    assert destination.read_bytes() == b"do-not-overwrite\n"


def test_preflight_rejects_global_prospective_identity_collisions_before_publish(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, plan_sha256 = _write_plan(tmp_path, capsys)
    plan_document = json.loads(plan_path.read_bytes())
    collision_ids = {
        item["request"]["lineage_id"] for item in plan_document["lineages"][:2]
    }

    def collide(config: object, request: object) -> ProspectiveLineage:
        identity = (
            "reproduced-v1-jobshop-collision"
            if request.lineage_id in collision_ids
            else None
        )
        return _fake_prospect(request, identity=identity)

    monkeypatch.setattr(preflight, "prospect_lineage", collide)
    destination = tmp_path / "prospective_preflight.json"

    with pytest.raises(ValueError, match="duplicate prospective (problem|split-unit) identity"):
        main(
            [
                "preflight",
                "--plan",
                str(plan_path),
                "--expected-plan-sha256",
                plan_sha256,
                "--expected-source-sha256",
                SOURCE_SHA256,
                "--workers",
                "1",
                "--out",
                str(destination),
            ]
        )
    assert not destination.exists()


def test_preflight_fails_if_generation_provenance_changes_during_census(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, plan_sha256 = _write_plan(tmp_path, capsys)
    monkeypatch.setattr(
        preflight,
        "prospect_lineage",
        lambda config, request: _fake_prospect(request),
    )
    calls = iter(
        (
            {"protocol": "embedbench.isingfold-corpus-generator-v4", "revision": 1},
            {"protocol": "embedbench.isingfold-corpus-generator-v4", "revision": 2},
        )
    )
    monkeypatch.setattr(preflight, "_generation_provenance", lambda: next(calls))
    destination = tmp_path / "prospective_preflight.json"

    with pytest.raises(ValueError, match="generation provenance changed"):
        main(
            [
                "preflight",
                "--plan",
                str(plan_path),
                "--expected-plan-sha256",
                plan_sha256,
                "--expected-source-sha256",
                SOURCE_SHA256,
                "--workers",
                "1",
                "--out",
                str(destination),
            ]
        )
    assert not destination.exists()


@pytest.mark.parametrize("workers", ["0", "65", "not-an-integer"])
def test_preflight_command_rejects_out_of_contract_worker_counts(workers: str) -> None:
    with pytest.raises(SystemExit):
        main(
            [
                "preflight",
                "--plan",
                "unused.json",
                "--expected-plan-sha256",
                "0" * 64,
                "--workers",
                workers,
                "--out",
                "unused-receipt.json",
            ]
        )
