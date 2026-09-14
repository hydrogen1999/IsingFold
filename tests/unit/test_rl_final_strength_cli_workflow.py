"""Outcome-blind orchestration checks for final-strength CLI commands."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from isingfold.rl.final_strength_cli import (
    plan_final_strength_audit,
    run_final_strength_audit_shard_command,
    source_arguments,
)


def test_source_arguments_reject_incomplete_census_before_file_access() -> None:
    args = SimpleNamespace(
        learned_evaluation=["missing-learned-a", "missing-learned-b"],
        expected_learned_report_sha256=["1" * 64, "2" * 64],
        external_evaluation=["missing-stock-a", "missing-stock-b", "missing-stock-c"],
        expected_external_report_sha256=["3" * 64, "4" * 64, "5" * 64],
    )

    with pytest.raises(ValueError, match="exactly three learned roots and pins"):
        source_arguments(args)


def test_source_arguments_reject_noncanonical_external_pin() -> None:
    args = SimpleNamespace(
        learned_evaluation=["learned-a", "learned-b", "learned-c"],
        expected_learned_report_sha256=["1" * 64, "2" * 64, "3" * 64],
        external_evaluation=["stock-a", "stock-b", "stock-c"],
        expected_external_report_sha256=["4" * 64, "5" * 64, "6" * 63 + "G"],
    )

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        source_arguments(args)


def test_source_arguments_preserve_root_pin_alignment() -> None:
    args = SimpleNamespace(
        learned_evaluation=["learned-a", "learned-b", "learned-c"],
        expected_learned_report_sha256=["1" * 64, "2" * 64, "3" * 64],
        external_evaluation=["stock-a", "stock-b", "stock-c"],
        expected_external_report_sha256=["4" * 64, "5" * 64, "6" * 64],
    )

    sources = source_arguments(args)

    assert sources.learned_directories == (
        "learned-a",
        "learned-b",
        "learned-c",
    )
    assert sources.stock_report_sha256s == ("4" * 64, "5" * 64, "6" * 64)


def test_final_strength_launchers_respect_apollo_and_goose_execution_models() -> None:
    repository = Path(__file__).resolve().parents[2]
    apollo = (repository / "scripts/apollo_final_strength_audit.sh").read_text()
    goose = (repository / "scripts/goose_final_strength_audit.sbatch").read_text()

    assert "srun" not in apollo and "sbatch" not in apollo
    assert "run-final-strength-audit-shard" in apollo
    assert "#SBATCH --array=0-127" in goose
    assert "/opt/slurm/bin/srun" in goose
    assert "run-final-strength-audit-shard" in goose
    for source_index in range(3):
        assert f"LEARNED_COMPLETE_{source_index}" in apollo
        assert f"LEARNED_COMPLETE_{source_index}" in goose
        assert f"STOCK_COMPLETE_{source_index}" in apollo
        assert f"STOCK_COMPLETE_{source_index}" in goose


def test_plan_reaches_only_public_test_partition_before_subset_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import isingfold.rl.cli as cli
    import isingfold.rl.data.ground_certificate as ground
    import isingfold.rl.data.prepared as prepared
    import isingfold.rl.experiment_selection as selection
    import isingfold.rl.external_tuning as tuning
    import isingfold.rl.final_strength_audit as audit

    class PublicBoundaryReached(RuntimeError):
        pass

    monkeypatch.setattr(
        cli,
        "_load_grid",
        lambda path: (
            {
                "complete_system_evaluation": {
                    "partition": "test",
                    "evaluation_seed": 55079,
                    "repetitions": 4,
                    "feasibility_noninferiority_margin": 0.02,
                }
            },
            "1" * 64,
        ),
    )
    monkeypatch.setattr(
        selection,
        "load_rl_value_freeze",
        lambda **kwargs: SimpleNamespace(grid_manifest_sha256="1" * 64),
    )
    monkeypatch.setattr(tuning, "load_external_tuning_registry", lambda *a, **k: object())
    monkeypatch.setattr(tuning, "load_external_tuning_selection", lambda *a, **k: object())
    monkeypatch.setattr(
        tuning.ExternalTuningExecutionBinding,
        "for_deployment",
        staticmethod(lambda value: object()),
    )
    monkeypatch.setattr(audit, "load_final_strength_audit_config", lambda *a, **k: object())
    monkeypatch.setattr(cli, "_publisher_attestation_pin", lambda args: object())
    root = SimpleNamespace()
    monkeypatch.setattr(ground, "load_ground_certificate_root", lambda *a, **k: root)
    authority = SimpleNamespace(as_dict=lambda: {})
    monkeypatch.setattr(
        ground,
        "project_ground_root_quality_authority",
        lambda loaded, partition: (authority, authority),
    )

    def public_only(*args, **kwargs):
        assert kwargs == {"partition": "test", "include_evaluator": False}
        raise PublicBoundaryReached

    monkeypatch.setattr(prepared, "load_prepared_partition", public_only)
    args = SimpleNamespace(
        out=tmp_path / "plan.json",
        grid="grid.json",
        rl_value_selection_receipt="rl.json",
        expected_selection_sha256="2" * 64,
        tuning_registry="tuning.json",
        expected_tuning_registry_sha256="3" * 64,
        external_config="external.json",
        external_tuning_selection="stock.json",
        expected_external_tuning_selection_sha256="4" * 64,
        audit_config="audit.json",
        expected_audit_config_sha256="5" * 64,
        ground_certificate_root="root.json",
        expected_ground_certificate_root_sha256="6" * 64,
        corpus="prepared-v4",
    )

    with pytest.raises(PublicBoundaryReached):
        plan_final_strength_audit(args)


def test_shard_authenticates_execution_pin_before_opening_test_targets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import isingfold.rl.data.ground_certificate as ground
    import isingfold.rl.final_strength_audit as audit

    class ExecutionPinRejected(RuntimeError):
        pass

    sealed_plan = SimpleNamespace(
        plan=SimpleNamespace(publication_eligible=True, shard_count=128)
    )
    monkeypatch.setattr(audit, "load_final_strength_audit_plan", lambda *a, **k: sealed_plan)

    def reject_execution(*args, **kwargs):
        raise ExecutionPinRejected

    monkeypatch.setattr(audit, "load_final_strength_audit_execution_manifest", reject_execution)
    opened = False

    def forbidden_target_open(*args, **kwargs):
        nonlocal opened
        opened = True
        raise AssertionError("test targets opened before execution authentication")

    monkeypatch.setattr(ground, "load_ground_certificate_partition", forbidden_target_open)
    args = SimpleNamespace(
        out=tmp_path / "shard",
        plan="plan.json",
        expected_plan_sha256="1" * 64,
        execution_manifest="execution.json",
        expected_execution_manifest_sha256="2" * 64,
        shard_index=0,
        learned_evaluation=["learned-a", "learned-b", "learned-c"],
        expected_learned_report_sha256=["3" * 64, "4" * 64, "5" * 64],
        external_evaluation=["stock-a", "stock-b", "stock-c"],
        expected_external_report_sha256=["6" * 64, "7" * 64, "8" * 64],
    )

    with pytest.raises(ExecutionPinRejected):
        run_final_strength_audit_shard_command(args)
    assert opened is False
