from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import embedbench.structural_release as structural_release
import pytest
from embedbench.candidate_bank import canonical_json_bytes, content_digest
from embedbench.structural_release import (
    DECISIONS_FILENAME,
    MANIFEST_FILENAME,
    StructuralReleasePlan,
    build_production_plan,
    generate_structural_shard,
    make_structural_plan_row,
    merge_structural_shards,
    shard_index_for,
)


def _tiny_plan(root_seed: int = 19, instances: int = 3) -> StructuralReleasePlan:
    rows = [
        make_structural_plan_row(
            root_seed=root_seed,
            topology="chimera",
            size=2,
            mode=("compact", "elongated", "near_capacity")[index % 3],
            split=("train", "val", "test")[index % 3],
            ood=index % 3 == 2,
            difficulty="easy" if index != 2 else "hard",
            replicate=index,
            faulted=False,
            n_vars=6,
            chain_size=3,
            k_in_play=3,
            radius=1,
            l_cap=4,
            max_window_free=18,
            max_nodes=200_000,
            samples_per_instance=4,
            max_actions=8,
        )
        for index in range(instances)
    ]
    return StructuralReleasePlan.build(root_seed=root_seed, rows=rows)


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def _generate_all(plan: StructuralReleasePlan, root: Path, count: int) -> list[Path]:
    directories = []
    for index in range(count):
        directory = root / f"shard-{index:03d}"
        generate_structural_shard(plan, index, count, directory)
        directories.append(directory)
    return directories


def test_production_plan_is_deterministic_and_covers_registered_axes() -> None:
    first = build_production_plan(root_seed=260912, replicates_per_cell=1)
    second = build_production_plan(root_seed=260912, replicates_per_cell=1)
    assert first.to_dict() == second.to_dict()
    assert len(first.rows) == 3 * 4 * 2 * 3 * 2
    assert {row.topology for row in first.rows} == {"chimera", "pegasus", "zephyr"}
    assert {row.mode for row in first.rows} == {
        "compact",
        "elongated",
        "cut_congested",
        "near_capacity",
    }
    assert {row.difficulty for row in first.rows} == {"easy", "hard"}
    assert {row.split for row in first.rows} == {"train", "val", "test"}
    assert {bool(row.removed_nodes) for row in first.rows} == {False, True}
    assert all(not row.ood or row.split == "test" for row in first.rows)
    assert all(row.ood == (row.split == "test") for row in first.rows)
    assert all(
        (row.host_condition == "faulted_nodes") == bool(row.removed_nodes)
        for row in first.rows
    )
    assert all(
        len(row.removed_nodes) == 1
        for row in first.rows
        if row.host_condition == "faulted_nodes"
    )
    cells = {
        (
            row.topology,
            row.mode,
            row.difficulty,
            row.host_condition,
            row.replicate,
            row.split,
        ): row
        for row in first.rows
    }
    for key, test_row in cells.items():
        if key[-1] != "test":
            continue
        train_row = cells[(*key[:-1], "train")]
        assert test_row.size > train_row.size
        assert test_row.n_vars > train_row.n_vars


def test_plan_round_trip_is_closed_and_immutable() -> None:
    plan = _tiny_plan(instances=2)
    restored = StructuralReleasePlan.from_dict(plan.to_dict())
    assert restored == plan
    assert restored.plan_digest == plan.plan_digest
    with pytest.raises(FrozenInstanceError):
        restored.root_seed = 0  # type: ignore[misc]
    altered = dict(plan.to_dict())
    altered["root_seed"] = plan.root_seed + 1
    with pytest.raises(ValueError, match="plan_digest"):
        StructuralReleasePlan.from_dict(altered)


@pytest.mark.parametrize("shard_count", [1, 2, 7, 19])
def test_sharding_never_changes_scientific_identity(shard_count: int) -> None:
    plan = _tiny_plan(instances=3)
    assigned = {
        row.lineage_id: shard_index_for(row.lineage_id, shard_count)
        for row in plan.rows
    }
    assert set(assigned) == {row.lineage_id for row in plan.rows}
    assert all(0 <= index < shard_count for index in assigned.values())
    # No shard coordinate participates in either stable identifier or generator seed.
    assert plan == _tiny_plan(instances=3)


def test_generation_wraps_exact_labels_and_complete_witness_provenance(tmp_path: Path) -> None:
    plan = _tiny_plan(instances=3)
    shard = tmp_path / "shard"
    receipt = generate_structural_shard(plan, 0, 1, shard)
    records = _read_jsonl(receipt.decisions_path)
    assert records
    for record in records:
        assert record["label_authority"] == {
            "algorithm": "exact_branch_and_bound_window_v1",
            "budget": {"max_nodes": record["plan_row"]["max_nodes"]},
            "capacity_constraint": {
                "q_cap_slack": record["plan_row"]["q_cap_slack"]
            },
            "scope": "complete_in_play_variables_inside_recorded_window",
            "status": "exact",
        }
        decision = record["decision"]
        assert len(decision["actions"]) == len(decision["values"])
        assert all(len(value) == 3 and value[0] in (0, 1) for value in decision["values"])
        assert decision["values"][decision["actions"].index(decision["best_action"])] == max(
            decision["values"]
        )
        witness = record["planted_witness"]
        assert witness["validation_errors"] == []
        assert witness["chains"]
        assert witness["host_nodes"] and witness["host_edges"]
        assert witness["seed"] == record["plan_row"]["generator_seed"]
        assert record["record_id"] == "sdr-" + content_digest(
            {key: value for key, value in record.items() if key != "record_id"}
        )
    manifest = json.loads(receipt.manifest_path.read_text())
    assert StructuralReleasePlan.from_dict(manifest["plan"]) == plan
    assert manifest["counts"]["records"] == len(records)
    assert manifest["attrition"]["dropped_aborted"] >= 0
    assert manifest["label_policy"]["unknown_or_aborted_records_emitted"] == 0
    assert set(path.name for path in shard.iterdir()) == {
        "SHA256SUMS",
        DECISIONS_FILENAME,
        MANIFEST_FILENAME,
    }


def test_realized_node_faults_are_applied_not_merely_claimed(tmp_path: Path) -> None:
    row = make_structural_plan_row(
        root_seed=91,
        topology="chimera",
        size=2,
        mode="compact",
        split="train",
        ood=False,
        difficulty="easy",
        replicate=0,
        faulted=True,
        n_vars=5,
        chain_size=2,
        k_in_play=3,
        radius=1,
        l_cap=3,
        max_window_free=16,
        max_nodes=100_000,
        samples_per_instance=3,
        max_actions=8,
    )
    plan = StructuralReleasePlan.build(root_seed=91, rows=[row])
    receipt = generate_structural_shard(plan, 0, 1, tmp_path / "faulted")
    manifest = json.loads(receipt.manifest_path.read_text())
    result = manifest["lineage_results"][0]
    assert result["host_realization"]["fault_intended"] is True
    assert result["host_realization"]["removed_nodes"] == list(row.removed_nodes)
    assert (
        result["host_realization"]["ideal_node_count"] - 1
        == result["host_realization"]["realized_node_count"]
    )
    for record in _read_jsonl(receipt.decisions_path):
        assert not (set(record["planted_witness"]["host_nodes"]) & set(row.removed_nodes))


def test_aborted_exact_search_is_attrition_never_a_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = _tiny_plan(instances=1)

    def aborted_samples(_pe, _cfg, _rng, stats, _instance_id):
        stats.samples_attempted += 1
        stats.dropped_aborted += 1
        return iter(())

    monkeypatch.setattr(
        structural_release, "decision_samples_from_witness", aborted_samples
    )
    receipt = generate_structural_shard(plan, 0, 1, tmp_path / "aborted")
    assert receipt.decisions_path.read_bytes() == b""
    manifest = json.loads(receipt.manifest_path.read_text())
    assert manifest["attrition"]["dropped_aborted"] == 1
    assert manifest["counts"]["records"] == 0
    assert manifest["label_policy"]["unknown_or_aborted_records_emitted"] == 0


def test_merge_is_byte_identical_for_shard_order_and_shard_count(tmp_path: Path) -> None:
    plan = _tiny_plan(instances=3)
    one = _generate_all(plan, tmp_path / "one", 1)
    three = _generate_all(plan, tmp_path / "three", 3)
    receipt_a = merge_structural_shards(plan, one, tmp_path / "release-a")
    receipt_b = merge_structural_shards(plan, list(reversed(three)), tmp_path / "release-b")
    names_a = sorted(path.name for path in receipt_a.output_directory.iterdir())
    names_b = sorted(path.name for path in receipt_b.output_directory.iterdir())
    assert names_a == names_b
    assert {
        name: (receipt_a.output_directory / name).read_bytes() for name in names_a
    } == {
        name: (receipt_b.output_directory / name).read_bytes() for name in names_b
    }


def test_merge_requires_exact_untampered_shard_coverage(tmp_path: Path) -> None:
    plan = _tiny_plan(instances=3)
    shards = _generate_all(plan, tmp_path / "shards", 3)
    with pytest.raises(ValueError, match="coverage"):
        merge_structural_shards(plan, shards[:2], tmp_path / "missing")
    (shards[1] / DECISIONS_FILENAME).write_bytes(
        (shards[1] / DECISIONS_FILENAME).read_bytes() + b"\n"
    )
    with pytest.raises(ValueError, match="SHA-256"):
        merge_structural_shards(plan, shards, tmp_path / "tampered")


def test_merge_rejects_reauthenticated_action_value_forgery(tmp_path: Path) -> None:
    plan = _tiny_plan(instances=3)
    shard = _generate_all(plan, tmp_path / "source-forgery", 1)[0]
    records = _read_jsonl(shard / DECISIONS_FILENAME)
    assert records
    record = records[0]
    decision = record["decision"]
    assert isinstance(decision, dict)
    actions = decision["actions"]
    values = decision["values"]
    assert isinstance(actions, list) and isinstance(values, list) and len(values) >= 2
    values[0], values[1] = values[1], values[0]
    order = sorted(range(len(actions)), key=lambda index: values[index], reverse=True)
    decision["best_action"] = actions[order[0]]
    component = next(
        index for index in range(3) if values[order[0]][index] != values[order[1]][index]
    )
    decision["margin_kind"] = ("feasibility", "qubits", "max_chain")[component]
    decision["margin"] = values[order[0]][component] - values[order[1]][component]
    decision["greedy_agrees"] = (
        values[actions.index(decision["greedy_action"])] == values[order[0]]
    )
    record["instance_id"] = "sdi-" + content_digest(
        {
            "decision": decision,
            "domain": "embedbench.structural-decision-instance",
            "kept_index": record["decision_index"],
            "lineage_id": record["lineage_id"],
            "schema_version": structural_release.SCHEMA_VERSION,
        }
    )
    record["record_id"] = "sdr-" + content_digest(
        {key: value for key, value in record.items() if key != "record_id"}
    )
    (shard / DECISIONS_FILENAME).write_bytes(
        b"".join(canonical_json_bytes(item) + b"\n" for item in records)
    )
    _reauthenticate_shard(shard)

    with pytest.raises(ValueError, match="exact replay"):
        merge_structural_shards(plan, [shard], tmp_path / "forged-release")


def test_merge_rejects_symlinks_and_unexpected_files(tmp_path: Path) -> None:
    plan = _tiny_plan(instances=1)
    shard = _generate_all(plan, tmp_path / "input", 1)[0]
    target = tmp_path / "copy.jsonl"
    target.write_bytes((shard / DECISIONS_FILENAME).read_bytes())
    (shard / DECISIONS_FILENAME).unlink()
    (shard / DECISIONS_FILENAME).symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        merge_structural_shards(plan, [shard], tmp_path / "symlink")

    clean = _generate_all(plan, tmp_path / "input-clean", 1)[0]
    (clean / "unregistered.txt").write_text("surprise")
    with pytest.raises(ValueError, match="unexpected"):
        merge_structural_shards(plan, [clean], tmp_path / "unexpected")


def test_publication_never_replaces_a_concurrently_reserved_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "published"
    original_write = structural_release._write_new
    writes = 0

    def reserve_after_staging(path: Path, raw: bytes) -> None:
        nonlocal writes
        original_write(path, raw)
        writes += 1
        if writes == 3:
            destination.mkdir()

    monkeypatch.setattr(structural_release, "_write_new", reserve_after_staging)

    with pytest.raises(FileExistsError):
        structural_release._publish_directory(
            destination,
            {
                DECISIONS_FILENAME: b"",
                MANIFEST_FILENAME: b"{}\n",
            },
        )
    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_merge_rejects_reauthenticated_unknown_manifest_fields(tmp_path: Path) -> None:
    plan = _tiny_plan(instances=1)
    shard = _generate_all(plan, tmp_path / "source-manifest", 1)[0]
    manifest_path = shard / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["unregistered_claim"] = "not allowed"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    _reauthenticate_shard(shard)
    with pytest.raises(ValueError, match="fields differ"):
        merge_structural_shards(plan, [shard], tmp_path / "unknown-manifest")


def _reauthenticate_shard(directory: Path) -> None:
    decisions = directory / DECISIONS_FILENAME
    manifest_path = directory / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    raw = decisions.read_bytes()
    manifest["artifact"] = {
        "byte_count": len(raw),
        "path": DECISIONS_FILENAME,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    payload = {key: value for key, value in manifest.items() if key != "record_digest"}
    manifest["record_digest"] = content_digest(payload)
    manifest_raw = canonical_json_bytes(manifest) + b"\n"
    manifest_path.write_bytes(manifest_raw)
    checksums = (
        f"{hashlib.sha256(raw).hexdigest()}  {DECISIONS_FILENAME}\n"
        f"{hashlib.sha256(manifest_raw).hexdigest()}  {MANIFEST_FILENAME}\n"
    )
    (directory / "SHA256SUMS").write_text(checksums)


@pytest.mark.parametrize("duplicated_field", ["instance_id", "record_id"])
def test_merge_rejects_reauthenticated_duplicate_identifiers(
    tmp_path: Path, duplicated_field: str
) -> None:
    plan = _tiny_plan(instances=3)
    shard = _generate_all(plan, tmp_path / "source", 1)[0]
    records = _read_jsonl(shard / DECISIONS_FILENAME)
    if len(records) < 2:
        pytest.skip("deterministic tiny fixture produced fewer than two certified records")
    records[1][duplicated_field] = records[0][duplicated_field]
    if duplicated_field == "instance_id":
        records[1]["record_id"] = "sdr-" + content_digest(
            {key: value for key, value in records[1].items() if key != "record_id"}
        )
    (shard / DECISIONS_FILENAME).write_bytes(
        b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    )
    _reauthenticate_shard(shard)
    with pytest.raises(ValueError, match=duplicated_field):
        merge_structural_shards(plan, [shard], tmp_path / "duplicate")


def test_invalid_or_fabricated_metadata_is_rejected() -> None:
    plan = _tiny_plan(instances=1)
    row = plan.rows[0]
    with pytest.raises(ValueError, match="removed_nodes"):
        StructuralReleasePlan.build(
            root_seed=plan.root_seed,
            rows=[replace(row, host_condition="faulted_nodes", removed_nodes=())],
        )
    with pytest.raises(ValueError, match="OOD"):
        StructuralReleasePlan.build(
            root_seed=plan.root_seed,
            rows=[replace(row, ood=True, split="train")],
        )
