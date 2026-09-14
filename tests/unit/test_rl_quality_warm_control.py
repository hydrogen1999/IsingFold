"""Fail-closed contracts for the post-selection direct-Q warm control."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from isingfold.rl.data.import_embedbench import content_digest
from isingfold.rl.cli import _quality_warm_source_binding, build_parser
from isingfold.rl.contracts import stable_digest
from isingfold.rl.quality_warm_control import (
    QUALITY_WARM_CONTROL_PROFILE_ID,
    build_quality_warm_control_binding,
    build_quality_warm_control_plan,
    load_quality_warm_control_registry,
    validate_quality_warm_control_binding,
)

ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "configs" / "rl_grid_hybrid_v1.json"
REGISTRY = ROOT / "configs" / "quality_warm_control_hybrid_v1.json"
SHA = "a" * 64


def _plan():
    registry = load_quality_warm_control_registry(REGISTRY, main_grid_path=GRID)
    return registry, build_quality_warm_control_plan(
        registry,
        selected_simpler="if-dual",
        representation_selection_sha256="1" * 64,
        representation_selection_record_digest="2" * 64,
        corpus_manifest_sha256="3" * 64,
        selector_digest="4" * 64,
        normalizer_digest="5" * 64,
        quality_manifest_sha256="6" * 64,
        quality_preflight_sha256="7" * 64,
        quality_preflight_record_digest="8" * 64,
    )


def test_registry_is_exactly_three_paired_ifcore_control_cells() -> None:
    registry, plan = _plan()

    assert registry.main_grid_sha256 == (
        "d3a7cd99c0c96c1c2f7fdd974d0856aee94ccbbe7a01f30b15d4724632712d09"
    )
    assert registry.model_family == "if-core"
    assert registry.quality_prior_mode == "bounded-centered-v1"
    assert registry.training_seeds == (1103, 2207, 3301)
    assert registry.training_protocol.epochs == 200
    assert registry.training_protocol.minibatch_records == 32
    assert registry.training_protocol.learning_rate == pytest.approx(3e-4)
    assert registry.training_protocol.weight_decay == 0.0
    assert registry.treatment_profile.profile_id == "full-qmu-v4"
    assert registry.control_profile.profile_id == QUALITY_WARM_CONTROL_PROFILE_ID
    assert registry.treatment_profile.weights.action_value == 1.0
    assert registry.control_profile.weights.action_value == 0.0
    assert registry.control_profile.weights.commit_delta == 0.0

    assert len(plan.cells) == 3
    assert [cell.seed for cell in plan.cells] == [1103, 2207, 3301]
    assert [cell.source_treatment_cell_id for cell in plan.cells] == [
        "rep-006-if-core-s1103",
        "rep-007-if-core-s2207",
        "rep-008-if-core-s3301",
    ]
    assert len({cell.control_cell_id for cell in plan.cells}) == 3
    assert plan.seed_selection_forbidden is True
    assert plan.selection_effect == "diagnostic-only-no-main-grid-reselection"
    assert plan.evaluation_partition == "validation"
    assert len(plan.plan_digest) == 64


def test_registry_rejects_duplicate_keys_tamper_self_rehash_and_grid_drift(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema": 1, "schema": 2}')
    with pytest.raises(ValueError, match="duplicate key"):
        load_quality_warm_control_registry(duplicate, main_grid_path=GRID)

    changed = json.loads(REGISTRY.read_text())
    changed["training_protocol"]["epochs"] = 199
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="record digest"):
        load_quality_warm_control_registry(tampered, main_grid_path=GRID)

    changed["record_digest"] = content_digest(
        {key: value for key, value in changed.items() if key != "record_digest"}
    )
    self_rehashed = tmp_path / "self-rehashed.json"
    self_rehashed.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="pinned v1 record digest"):
        load_quality_warm_control_registry(self_rehashed, main_grid_path=GRID)

    grid = json.loads(GRID.read_text())
    grid["stages"]["representation"].pop()
    changed_grid = tmp_path / "grid.json"
    changed_grid.write_text(json.dumps(grid))
    with pytest.raises(ValueError, match="pinned main-grid digest"):
        load_quality_warm_control_registry(REGISTRY, main_grid_path=changed_grid)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("selected_simpler", "if-core", "selected simpler"),
        ("representation_selection_sha256", "self-asserted", "SHA-256"),
        ("quality_preflight_record_digest", "not-a-digest", "SHA-256"),
    ],
)
def test_plan_rejects_invalid_selection_or_data_binding(
    field: str,
    value: str,
    message: str,
) -> None:
    registry = load_quality_warm_control_registry(REGISTRY, main_grid_path=GRID)
    kwargs = {
        "selected_simpler": "if-dual",
        "representation_selection_sha256": "1" * 64,
        "representation_selection_record_digest": "2" * 64,
        "corpus_manifest_sha256": "3" * 64,
        "selector_digest": "4" * 64,
        "normalizer_digest": "5" * 64,
        "quality_manifest_sha256": "6" * 64,
        "quality_preflight_sha256": "7" * 64,
        "quality_preflight_record_digest": "8" * 64,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        build_quality_warm_control_plan(registry, **kwargs)


def test_control_binding_is_signed_and_identity_checked() -> None:
    registry, plan = _plan()
    cell = plan.cells[1]
    binding = build_quality_warm_control_binding(
        registry,
        plan,
        cell=cell,
        source_treatment_run_record_digest="9" * 64,
        source_treatment_checkpoint_payload_digest="a" * 64,
        runtime_implementation_digest="b" * 64,
    )

    assert validate_quality_warm_control_binding(
        binding,
        expected_profile_id=QUALITY_WARM_CONTROL_PROFILE_ID,
        expected_seed=2207,
        expected_model_family="if-core",
    ) == binding

    tampered = dict(binding)
    tampered["seed"] = 1103
    with pytest.raises(ValueError, match="record digest"):
        validate_quality_warm_control_binding(
            tampered,
            expected_profile_id=QUALITY_WARM_CONTROL_PROFILE_ID,
            expected_seed=2207,
            expected_model_family="if-core",
        )


def test_same_seed_profiles_are_parameter_identical_by_construction() -> None:
    registry, plan = _plan()

    assert registry.same_parameter_schema is True
    assert registry.same_initial_state_for_seed is True
    assert registry.same_training_data_order is True
    assert all(cell.model_signature_sha256 == registry.model_signature_sha256 for cell in plan.cells)
    assert all(cell.model_family == registry.model_family for cell in plan.cells)


def test_dedicated_cli_exposes_no_scientific_hyperparameter_or_partition_override() -> None:
    parser = build_parser()
    command = next(
        action
        for action in parser._actions
        if getattr(action, "dest", None) == "command"
    ).choices["warm-quality-control-cell"]
    destinations = {action.dest for action in command._actions}

    assert {
        "config",
        "expected_config_sha256",
        "grid",
        "expected_grid_sha256",
        "representation_selection_receipt",
        "expected_representation_selection_sha256",
        "representation_run_root",
        "run_root",
        "index",
    } <= destinations
    assert not {
        "seed",
        "epochs",
        "minibatch",
        "learning_rate",
        "weight_decay",
        "model_family",
        "quality_prior_mode",
        "warm_start_loss_profile_id",
        "partition",
    } & destinations


def test_hpc_launchers_preserve_apollo_and_goose_scheduler_boundaries() -> None:
    apollo = (ROOT / "scripts" / "apollo_quality_warm_control.sh").read_text()
    goose = (ROOT / "scripts" / "goose_quality_warm_control.sbatch").read_text()

    assert "sbatch" not in apollo
    assert "warm-quality-control-cell" in apollo
    assert "FIRST and LAST" in apollo
    assert "SLURM_JOB_ID" in goose and "SLURM_ARRAY_TASK_ID" in goose
    assert "/opt/slurm/bin/srun" in goose
    assert "warm-quality-control-cell" in goose
    assert "submit exactly --array=0-2" in goose
    for launcher in (apollo, goose):
        assert "--seed" not in launcher
        assert "--epochs" not in launcher
        assert "--model-family" not in launcher
        assert "--quality-prior-mode" not in launcher


def test_source_treatment_must_match_control_runtime_data_and_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from isingfold.rl.ppo import (
        WARM_START_ACTION_VALUE_TARGET,
        WARM_START_ACTION_VALUE_WEIGHTING,
        WARM_START_ACTOR_CRITIC_LOSS,
        WARM_START_COMMIT_DELTA_TARGET,
        WARM_START_FULL_LOSS_PROFILE,
        WARM_START_REDUCTION,
        WARM_START_UTILITY_TARGET,
        WARM_START_UTILITY_WEIGHTING,
    )

    registry, plan = _plan()
    cell = plan.cells[0]
    source = (
        tmp_path / "main" / "representation" / cell.source_treatment_cell_id
    )
    source.mkdir(parents=True)
    (source / "checkpoint.pt").write_bytes(b"authenticated by selection payload")
    runtime = {"schema": "fixture-runtime"}
    runtime_digest = stable_digest(runtime)
    protocol = registry.training_protocol
    receipt = {
        "schema": "isingfold.training-run",
        "schema_version": 4,
        "phase": "representation",
        "model_family": "if-core",
        "method": "supervised-ranking",
        "complete": True,
        "checkpoint_payload_digest": "a" * 64,
        "record_digest": "9" * 64,
        "runtime_implementation_registry": runtime,
        "runtime_implementation_digest": runtime_digest,
        "counters": {"epochs": 200, "optimizer_steps": 200, "records": 128},
        "experiment_contract": {
            "phase": "representation",
            "method": "supervised-ranking",
            "model_family": "if-core",
            "model": {"fixture": "if-core"},
            "seed": 1103,
            "grid_manifest_sha256": registry.main_grid_sha256,
            "grid_cell": cell.source_treatment_cell_id,
            "corpus_manifest_sha256": plan.corpus_manifest_sha256,
            "selector_digest": plan.selector_digest,
            "normalizer_digest": plan.normalizer_digest,
            "quality_preflight_receipt_sha256": plan.quality_preflight_sha256,
            "quality_preflight_record_digest": plan.quality_preflight_record_digest,
            "device_type": "cuda",
            "deterministic_algorithms": True,
            "hyperparameters": {
                "epochs": protocol.epochs,
                "minibatch_records": protocol.minibatch_records,
                "learning_rate": protocol.learning_rate,
                "weight_decay": protocol.weight_decay,
                "quality_manifest_sha256": plan.quality_manifest_sha256,
                "minimum_resolved_rows": protocol.minimum_resolved_rows,
                "minimum_resolved_lineages": protocol.minimum_resolved_lineages,
                "quality_preflight_receipt_sha256": plan.quality_preflight_sha256,
                "quality_preflight_record_digest": plan.quality_preflight_record_digest,
                "warm_start_loss": WARM_START_ACTOR_CRITIC_LOSS,
                "warm_start_loss_profile": WARM_START_FULL_LOSS_PROFILE.contract(),
                "warm_start_rank_coefficient": 1.0,
                "warm_start_action_value_target": WARM_START_ACTION_VALUE_TARGET,
                "warm_start_action_value_weighting": WARM_START_ACTION_VALUE_WEIGHTING,
                "warm_start_action_value_coefficient": 1.0,
                "warm_start_commit_delta_target": WARM_START_COMMIT_DELTA_TARGET,
                "warm_start_commit_delta_coefficient": 0.5,
                "warm_start_reduction": WARM_START_REDUCTION,
                "warm_start_utility_target": WARM_START_UTILITY_TARGET,
                "warm_start_utility_weighting": WARM_START_UTILITY_WEIGHTING,
                "warm_start_utility_coefficient": 0.5,
                "warm_start_diagnostic_binding": None,
            },
        },
    }
    monkeypatch.setattr("isingfold.rl.cli._strict_json", lambda _path: receipt)
    monkeypatch.setattr("isingfold.rl.cli._verify_record", lambda *_args: None)
    monkeypatch.setattr("isingfold.rl.cli._model", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        "isingfold.rl.cli._model_identity",
        lambda *_args, **_kwargs: {"fixture": "if-core"},
    )
    monkeypatch.setattr(
        "isingfold.rl.checkpoint.runtime_implementation_registry",
        lambda: runtime,
    )
    args = SimpleNamespace(
        representation_run_root=str(tmp_path / "main"),
        deterministic=True,
    )

    binding = _quality_warm_source_binding(
        args=args,
        registry=registry,
        plan=plan,
        cell=cell,
        source_checkpoint_payload_digest="a" * 64,
        control_device_type="cuda",
    )
    assert binding["source_treatment_cell_id"] == cell.source_treatment_cell_id
    assert binding["runtime_implementation_digest"] == runtime_digest

    receipt["counters"]["optimizer_steps"] = 199
    with pytest.raises(ValueError, match="incomplete budget"):
        _quality_warm_source_binding(
            args=args,
            registry=registry,
            plan=plan,
            cell=cell,
            source_checkpoint_payload_digest="a" * 64,
            control_device_type="cuda",
        )
