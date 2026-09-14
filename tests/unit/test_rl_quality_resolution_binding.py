from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

import isingfold.rl.data.quality_resolution_plan as plan_module
from isingfold.rl.contracts import stable_digest
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.quality_resolution import QualityResolutionError
from isingfold.rl.data.quality_resolution_binding import (
    QUALITY_RESOLUTION_BINDING_SCHEMA,
    QUALITY_TRAINING_INPUT_READINESS_SCHEMA,
    load_quality_resolution_binding,
    load_quality_training_input_readiness,
    verify_quality_resolution_binding,
    verify_quality_training_input_readiness,
)
from tests.unit.test_rl_quality_resolution_merge import (
    GROUND_PARTITION,
    QUALITY_AUTHORITY,
    QUALITY_CONTRACT,
    TARGET_ACCESS,
    _plan_sha,
    _two_action_plan,
    _write_stage_zero_sources,
)
from isingfold.rl.data.quality_resolution_merge import (
    merge_quality_resolution,
    publish_quality_resolution_shard_pin_registry,
)


def _record(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "record_digest": content_digest(payload)}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_quality_manifest(path: Path, plan, *, continuations: int = 12) -> str:
    record = plan.as_dict()
    lineages = sorted(
        item["base_lineage"] for item in record["production_plan"]["lineages"]
    )
    tasks = sorted(
        task
        for item in record["production_plan"]["lineages"]
        for task in item["task_ids"]
    )
    label_protocol = {
        "requested_lineages": 0,
        "tasks_per_lineage_cap": 1,
        "states_per_lineage_cap": 4,
        "evaluated_actions": 8,
        "continuations": continuations,
        "reward_reads": 256,
        "seed": 907,
        "implementation_contract": QUALITY_CONTRACT,
    }
    sampling = {
        "available_lineages": 128,
        "requested_lineages": 0,
        "selected_lineages": lineages,
        "selected_task_ids": tasks,
        "tasks_per_lineage_cap": 1,
        "states_per_lineage_cap": 4,
    }
    trajectories = 128 * 2 * continuations
    denominators = {
        "selected_lineages": 128,
        "selected_tasks": 128,
        "lineages_with_records": 128,
        "tasks_with_records": 128,
        "records_total": 128,
        "resolved_rows": 128,
        "fully_unresolved_rows": 0,
        "resolved_lineages": 128,
        "continuation_trajectories": trajectories,
        "valid_continuations": trajectories,
    }
    payload: dict[str, object] = {
        "schema": "isingfold.quality-label-corpus",
        "schema_version": 8,
        "source_corpus_manifest_sha256": record["prepared_corpus"]["manifest_sha256"],
        "quality_authority": QUALITY_AUTHORITY,
        "target_access": TARGET_ACCESS,
        "ground_partition_receipt": GROUND_PARTITION,
        "selector_digest": record["selector"]["selector_digest"],
        "normalizer_digest": record["selector"]["normalizer_digest"],
        "context": record["context"]["snapshot"],
        "context_digest": record["context"]["digest"],
        "partition": "train",
        "record_count": 128,
        "records_sha256": "a" * 64,
        "lineages": lineages,
        "task_ids": tasks,
        "sampling_receipt": sampling,
        "independent_denominators": denominators,
        "label_protocol": label_protocol,
        "initializer_bank": record["production_plan"]["initializer_bank"],
        "quality_resolution_plan": {
            "raw_sha256": _plan_sha(plan),
            "record_digest": record["record_digest"],
        },
    }
    manifest = _record(payload)
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return _sha(path)


def _write_preflight(path: Path, quality_manifest_path: Path, plan) -> str:
    manifest = __import__("json").loads(quality_manifest_path.read_text())
    denominators = manifest["independent_denominators"]
    trajectories = denominators["continuation_trajectories"]
    resolution = {
        "records_total": 128,
        "records_used": 128,
        "fully_unresolved_skipped": 0,
        "label_lineages": 128,
        "resolved_lineages": 128,
        "resolved_tasks": 128,
        "unresolved_lineages": [],
        "minimum_resolved_rows": 128,
        "minimum_resolved_lineages": 128,
        "passes_resolution": True,
        "continuation_trajectories_authenticated": trajectories,
        "continuation_replays_executed": trajectories,
        "continuation_replay_matches": trajectories,
        "continuation_replay_mode": "full-exact-continuation-and-evaluator",
    }
    payload: dict[str, object] = {
        "schema": "isingfold.quality-resolution-preflight",
        "schema_version": 3,
        "source_quality_manifest_sha256": _sha(quality_manifest_path),
        "source_quality_manifest_record_digest": manifest["record_digest"],
        "source_quality_records_sha256": manifest["records_sha256"],
        "source_corpus_manifest_sha256": plan.as_dict()["prepared_corpus"][
            "manifest_sha256"
        ],
        "quality_authority": QUALITY_AUTHORITY,
        "selector_digest": plan.as_dict()["selector"]["selector_digest"],
        "normalizer_digest": plan.as_dict()["selector"]["normalizer_digest"],
        "context": plan.as_dict()["context"]["snapshot"],
        "context_digest": plan.as_dict()["context"]["digest"],
        "implementation_contract": QUALITY_CONTRACT,
        "label_protocol_digest": stable_digest(manifest["label_protocol"]),
        "minimum_resolved_rows": 128,
        "minimum_resolved_lineages": 128,
        "full_replay": {
            "mode": "full-exact-continuation-and-evaluator",
            "continuation_trajectories": trajectories,
            "continuation_replays_executed": trajectories,
            "continuation_replay_matches": trajectories,
        },
        "resolution": resolution,
        "advance": True,
        "initializer_bank": manifest["initializer_bank"],
        "quality_resolution_plan": manifest["quality_resolution_plan"],
    }
    receipt = _record(payload)
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return _sha(path)


@pytest.fixture()
def phase_three_artifacts(tmp_path: Path) -> dict[str, Any]:
    old_minimum = plan_module.MINIMUM_PRODUCTION_LINEAGES
    plan_module.MINIMUM_PRODUCTION_LINEAGES = 128
    try:
        plan = _two_action_plan()
    finally:
        plan_module.MINIMUM_PRODUCTION_LINEAGES = old_minimum
    pins = _write_stage_zero_sources(tmp_path, plan)
    registry = tmp_path / "pins.json"
    registry_sha = publish_quality_resolution_shard_pin_registry(
        registry,
        plan=plan,
        expected_plan_sha256=_plan_sha(plan),
        completed_stage_index=0,
        pins=pins,
    )
    study = tmp_path / "study.json"
    study_sha = merge_quality_resolution(
        plan,
        expected_plan_sha256=_plan_sha(plan),
        pin_registry_path=registry,
        expected_pin_registry_sha256=registry_sha,
        output_path=study,
    )
    quality_manifest = tmp_path / "quality-manifest.json"
    quality_manifest_sha = _write_quality_manifest(quality_manifest, plan)
    return {
        "plan": plan,
        "study": study,
        "study_sha": study_sha,
        "quality_manifest": quality_manifest,
        "quality_manifest_sha": quality_manifest_sha,
        "tmp_path": tmp_path,
    }


def test_binding_requires_terminal_selection_and_exact_all_train_quality_protocol(
    phase_three_artifacts: dict[str, Any],
) -> None:
    item = phase_three_artifacts
    binding_path = item["tmp_path"] / "binding.json"
    binding_sha = verify_quality_resolution_binding(
        plan=item["plan"],
        expected_plan_sha256=_plan_sha(item["plan"]),
        resolution_receipt_path=item["study"],
        expected_resolution_receipt_sha256=item["study_sha"],
        quality_manifest_path=item["quality_manifest"],
        expected_quality_manifest_sha256=item["quality_manifest_sha"],
        output_path=binding_path,
    )
    binding = load_quality_resolution_binding(
        binding_path,
        expected_binding_sha256=binding_sha,
        plan=item["plan"],
        expected_plan_sha256=_plan_sha(item["plan"]),
        resolution_receipt_path=item["study"],
        expected_resolution_receipt_sha256=item["study_sha"],
        quality_manifest_path=item["quality_manifest"],
        expected_quality_manifest_sha256=item["quality_manifest_sha"],
    ).as_dict()
    assert binding["schema"] == QUALITY_RESOLUTION_BINDING_SCHEMA
    assert binding["selected_continuations"] == 12
    assert binding["checks"] == {
        "all_train_population": True,
        "authority": True,
        "context_and_selector": True,
        "production_protocol": True,
        "schema": True,
    }
    assert binding["pass"] is True

    wrong_manifest = item["tmp_path"] / "wrong-continuations.json"
    _write_quality_manifest(wrong_manifest, item["plan"], continuations=16)
    with pytest.raises(QualityResolutionError, match="production protocol"):
        verify_quality_resolution_binding(
            plan=item["plan"],
            expected_plan_sha256=_plan_sha(item["plan"]),
            resolution_receipt_path=item["study"],
            expected_resolution_receipt_sha256=item["study_sha"],
            quality_manifest_path=wrong_manifest,
            expected_quality_manifest_sha256=_sha(wrong_manifest),
            output_path=item["tmp_path"] / "wrong-binding.json",
        )
    assert not (item["tmp_path"] / "wrong-binding.json").exists()

    substituted = __import__("json").loads(item["quality_manifest"].read_text())
    substituted["initializer_bank"]["manifest_sha256"] = "f" * 64
    substituted_body = {
        key: value for key, value in substituted.items() if key != "record_digest"
    }
    substituted["record_digest"] = content_digest(substituted_body)
    substituted_path = item["tmp_path"] / "substituted-bank-manifest.json"
    substituted_path.write_bytes(canonical_json_bytes(substituted) + b"\n")
    with pytest.raises(QualityResolutionError, match="initializer bank"):
        verify_quality_resolution_binding(
            plan=item["plan"],
            expected_plan_sha256=_plan_sha(item["plan"]),
            resolution_receipt_path=item["study"],
            expected_resolution_receipt_sha256=item["study_sha"],
            quality_manifest_path=substituted_path,
            expected_quality_manifest_sha256=_sha(substituted_path),
            output_path=item["tmp_path"] / "substituted-bank-binding.json",
        )


def test_pinned_full_replay_preflight_produces_explicit_training_input_readiness(
    phase_three_artifacts: dict[str, Any],
) -> None:
    item = phase_three_artifacts
    binding_path = item["tmp_path"] / "binding-for-readiness.json"
    binding_sha = verify_quality_resolution_binding(
        plan=item["plan"],
        expected_plan_sha256=_plan_sha(item["plan"]),
        resolution_receipt_path=item["study"],
        expected_resolution_receipt_sha256=item["study_sha"],
        quality_manifest_path=item["quality_manifest"],
        expected_quality_manifest_sha256=item["quality_manifest_sha"],
        output_path=binding_path,
    )
    preflight_path = item["tmp_path"] / "preflight.json"
    preflight_sha = _write_preflight(
        preflight_path, item["quality_manifest"], item["plan"]
    )
    readiness_path = item["tmp_path"] / "quality-training-input-readiness.json"
    readiness_sha = verify_quality_training_input_readiness(
        plan=item["plan"],
        expected_plan_sha256=_plan_sha(item["plan"]),
        resolution_receipt_path=item["study"],
        expected_resolution_receipt_sha256=item["study_sha"],
        binding_path=binding_path,
        expected_binding_sha256=binding_sha,
        quality_manifest_path=item["quality_manifest"],
        expected_quality_manifest_sha256=item["quality_manifest_sha"],
        quality_preflight_path=preflight_path,
        expected_quality_preflight_sha256=preflight_sha,
        output_path=readiness_path,
    )
    readiness = load_quality_training_input_readiness(
        readiness_path,
        expected_readiness_sha256=readiness_sha,
        binding_path=binding_path,
        expected_binding_sha256=binding_sha,
        quality_preflight_path=preflight_path,
        expected_quality_preflight_sha256=preflight_sha,
    ).as_dict()
    assert readiness["schema"] == QUALITY_TRAINING_INPUT_READINESS_SCHEMA
    assert readiness["scope"] == "quality-supervision-input-only"
    assert readiness["pass"] is True
    assert readiness["selected_continuations"] == 12
    assert readiness["resolved_rows"] == 128
    assert readiness["resolved_lineages"] == 128

    forged_readiness = __import__("json").loads(readiness_path.read_text())
    forged_readiness["selected_continuations"] = 16
    forged_body = {
        key: value for key, value in forged_readiness.items() if key != "record_digest"
    }
    forged_readiness["record_digest"] = content_digest(forged_body)
    forged_path = item["tmp_path"] / "forged-readiness.json"
    forged_path.write_bytes(canonical_json_bytes(forged_readiness) + b"\n")
    with pytest.raises(QualityResolutionError, match="cannot be reproduced"):
        load_quality_training_input_readiness(
            forged_path,
            expected_readiness_sha256=_sha(forged_path),
            binding_path=binding_path,
            expected_binding_sha256=binding_sha,
            quality_preflight_path=preflight_path,
            expected_quality_preflight_sha256=preflight_sha,
        )

    tampered = __import__("json").loads(preflight_path.read_text())
    tampered["full_replay"]["continuation_replay_matches"] -= 1
    body = {key: value for key, value in tampered.items() if key != "record_digest"}
    tampered["record_digest"] = content_digest(body)
    bad_preflight = item["tmp_path"] / "bad-preflight.json"
    bad_preflight.write_bytes(canonical_json_bytes(tampered) + b"\n")
    with pytest.raises(QualityResolutionError, match="complete exact replay"):
        verify_quality_training_input_readiness(
            plan=item["plan"],
            expected_plan_sha256=_plan_sha(item["plan"]),
            resolution_receipt_path=item["study"],
            expected_resolution_receipt_sha256=item["study_sha"],
            binding_path=binding_path,
            expected_binding_sha256=binding_sha,
            quality_manifest_path=item["quality_manifest"],
            expected_quality_manifest_sha256=item["quality_manifest_sha"],
            quality_preflight_path=bad_preflight,
            expected_quality_preflight_sha256=_sha(bad_preflight),
            output_path=item["tmp_path"] / "bad-readiness.json",
        )

    substituted_preflight = __import__("json").loads(preflight_path.read_text())
    substituted_preflight["initializer_bank"]["manifest_sha256"] = "f" * 64
    body = {
        key: value
        for key, value in substituted_preflight.items()
        if key != "record_digest"
    }
    substituted_preflight["record_digest"] = content_digest(body)
    bad_bank_preflight = item["tmp_path"] / "bad-bank-preflight.json"
    bad_bank_preflight.write_bytes(canonical_json_bytes(substituted_preflight) + b"\n")
    with pytest.raises(QualityResolutionError, match="another bound quality"):
        verify_quality_training_input_readiness(
            plan=item["plan"],
            expected_plan_sha256=_plan_sha(item["plan"]),
            resolution_receipt_path=item["study"],
            expected_resolution_receipt_sha256=item["study_sha"],
            binding_path=binding_path,
            expected_binding_sha256=binding_sha,
            quality_manifest_path=item["quality_manifest"],
            expected_quality_manifest_sha256=item["quality_manifest_sha"],
            quality_preflight_path=bad_bank_preflight,
            expected_quality_preflight_sha256=_sha(bad_bank_preflight),
            output_path=item["tmp_path"] / "bad-bank-readiness.json",
        )
