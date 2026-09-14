from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import embedbench.repair_groups as repair_groups
import pytest
from embedbench.candidate_bank import InstanceRecord, content_digest
from embedbench.repair_groups import (
    GeneratedRepairGroup,
    generate_repaired_group,
    validate_generated_group,
)


def _instance() -> InstanceRecord:
    return InstanceRecord.create(
        family="persistence-test",
        topology="cycle-C8",
        logical_nodes=(0, 1, 2),
        logical_edges=((0, 1), (1, 2)),
        host_nodes=tuple(range(8)),
        host_edges=tuple((node, (node + 1) % 8) for node in range(8)),
        h=((0, -0.5), (1, 0.25), (2, 0.75)),
        j=((0, 1, -1.0), (1, 2, 0.5)),
    )


def _draft(
    monkeypatch: pytest.MonkeyPatch,
    instance: InstanceRecord,
    *,
    group_seed: int,
    outcomes: tuple[SimpleNamespace, ...],
) -> GeneratedRepairGroup:
    monkeypatch.setattr(
        repair_groups,
        "_run_repair_attempt",
        lambda **kwargs: outcomes[kwargs["slot"]],
    )
    return generate_repaired_group(
        instance,
        ((0,), (1,), (2,)),
        group_seed=group_seed,
        attempt_slots=len(outcomes),
        max_transitions_per_attempt=20,
        repairer=lambda *args, **kwargs: {},
        repairer_id="tests.persistence-repairer-v1",
    )


def _accepted_draft(
    monkeypatch: pytest.MonkeyPatch,
    instance: InstanceRecord,
    *,
    group_seed: int = 41,
) -> GeneratedRepairGroup:
    return _draft(
        monkeypatch,
        instance,
        group_seed=group_seed,
        outcomes=(
            SimpleNamespace(chains=((0,), (1,), (2, 3)), transitions=2, reason="success"),
            SimpleNamespace(chains=((7,), (0,), (1, 2)), transitions=3, reason="success"),
            SimpleNamespace(chains=None, transitions=20, reason="transition_limit"),
        ),
    )


def _rejected_draft(
    monkeypatch: pytest.MonkeyPatch,
    instance: InstanceRecord,
    *,
    group_seed: int = 43,
) -> GeneratedRepairGroup:
    return _draft(
        monkeypatch,
        instance,
        group_seed=group_seed,
        outcomes=(
            SimpleNamespace(chains=None, transitions=20, reason="transition_limit"),
            SimpleNamespace(chains=None, transitions=7, reason="no_candidate"),
        ),
    )


def _redigest(group: GeneratedRepairGroup) -> GeneratedRepairGroup:
    payload = group.to_dict()
    payload.pop("record_digest", None)
    return replace(group, record_digest=content_digest(payload))


def test_generated_repair_group_round_trips_with_its_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _instance()
    draft = _accepted_draft(monkeypatch, instance)

    payload = draft.to_dict()
    restored = GeneratedRepairGroup.from_dict(payload, instance=instance)

    assert payload["record_digest"] == draft.record_digest
    assert restored == draft
    assert restored.to_dict() == payload


def test_generated_repair_group_rejects_digest_and_schema_tampering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _instance()
    payload = _accepted_draft(monkeypatch, instance).to_dict()

    changed_digest = dict(payload)
    changed_digest["record_digest"] = "0" * 64
    with pytest.raises(ValueError, match="digest"):
        GeneratedRepairGroup.from_dict(changed_digest, instance=instance)

    unknown_field = dict(payload)
    unknown_field["unexpected"] = True
    with pytest.raises(ValueError, match="schema|unknown|fields"):
        GeneratedRepairGroup.from_dict(unknown_field, instance=instance)


def test_shard_round_trip_accounts_for_sealable_underfull_and_failed_attempts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    instance = _instance()
    accepted = _accepted_draft(monkeypatch, instance)
    rejected = _rejected_draft(monkeypatch, instance)
    path = tmp_path / "repair-drafts.jsonl"

    written = repair_groups.write_repair_draft_shard(
        path,
        instances=(instance,),
        drafts=(rejected, accepted),
    )
    loaded_instances, loaded_drafts, loaded_manifest = repair_groups.read_repair_draft_shard(path)

    assert repair_groups.repair_draft_manifest_path(path).is_file()
    assert loaded_manifest == written
    assert loaded_instances == (instance,)
    assert loaded_drafts == tuple(sorted((accepted, rejected), key=lambda item: item.group_id))
    assert written.instance_count == 1
    assert written.draft_count == 2
    assert written.accepted_count == 1
    assert written.rejected_count == 1
    assert written.underfull_count == 1
    assert written.drafts_with_repair_failure_count == 2
    assert dict(written.attempt_status_counts) == {
        "duplicate": 0,
        "no_change": 0,
        "repair_failed": 3,
        "valid": 2,
    }
    assert sum(dict(written.split_counts).values()) == 2


def test_append_is_atomic_and_rejects_duplicate_group_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    instance = _instance()
    accepted = _accepted_draft(monkeypatch, instance)
    rejected = _rejected_draft(monkeypatch, instance)
    path = tmp_path / "repair-drafts.jsonl"
    repair_groups.write_repair_draft_shard(path, instances=(instance,), drafts=(accepted,))

    repair_groups.append_repair_draft_shard(path, instances=(), drafts=(rejected,))
    _, drafts, manifest = repair_groups.read_repair_draft_shard(path)
    before_data = path.read_bytes()
    before_manifest = repair_groups.repair_draft_manifest_path(path).read_bytes()

    assert {draft.group_id for draft in drafts} == {accepted.group_id, rejected.group_id}
    assert manifest.draft_count == 2
    with pytest.raises(ValueError, match="duplicate group_id"):
        repair_groups.append_repair_draft_shard(path, instances=(), drafts=(accepted,))
    assert path.read_bytes() == before_data
    assert repair_groups.repair_draft_manifest_path(path).read_bytes() == before_manifest


def test_writer_rejects_duplicates_before_publishing_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    instance = _instance()
    draft = _accepted_draft(monkeypatch, instance)
    path = tmp_path / "duplicate.jsonl"

    with pytest.raises(ValueError, match="duplicate group_id"):
        repair_groups.write_repair_draft_shard(
            path,
            instances=(instance,),
            drafts=(draft, draft),
        )

    assert not path.exists()
    assert not repair_groups.repair_draft_manifest_path(path).exists()


def test_reader_rejects_data_and_manifest_tampering(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    instance = _instance()
    draft = _accepted_draft(monkeypatch, instance)
    path = tmp_path / "tamper.jsonl"
    repair_groups.write_repair_draft_shard(path, instances=(instance,), drafts=(draft,))

    original_data = path.read_bytes()
    path.write_bytes(original_data + b"\n")
    with pytest.raises(ValueError, match="checksum|byte"):
        repair_groups.read_repair_draft_shard(path)

    path.write_bytes(original_data)
    manifest_path = repair_groups.repair_draft_manifest_path(path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["accepted_count"] = 0
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|digest|count"):
        repair_groups.read_repair_draft_shard(path)


def test_validation_recomputes_edge_choice_and_enforces_transition_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _instance()
    draft = _accepted_draft(monkeypatch, instance)
    first = draft.attempts[0]
    other_edge = next(edge for edge in instance.logical_edges if edge != first.neighborhood)

    wrong_edge = _redigest(
        replace(
            draft,
            attempts=(replace(first, neighborhood=other_edge), *draft.attempts[1:]),
        )
    )
    with pytest.raises(ValueError, match="neighborhood|edge.*seed|derived"):
        validate_generated_group(wrong_edge, instance)

    over_cap = _redigest(
        replace(
            draft,
            attempts=(replace(first, transitions=21), *draft.attempts[1:]),
        )
    )
    with pytest.raises(ValueError, match="transition|cap|maximum"):
        validate_generated_group(over_cap, instance)


def test_validation_requires_the_exact_generation_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = _instance()
    draft = _accepted_draft(monkeypatch, instance)
    tampered = _redigest(replace(draft, protocol=(*draft.protocol, ("unexpected", True))))

    with pytest.raises(ValueError, match="protocol.*fields|unexpected"):
        validate_generated_group(tampered, instance)
