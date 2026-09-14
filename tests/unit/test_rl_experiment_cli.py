from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from isingfold.rl import cli
from isingfold.rl.cli import build_parser
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.ppo import PPOConfig


ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "configs" / "rl_grid_hybrid_v1.json"


def _quality_args() -> list[str]:
    return [
        "--quality-attestation",
        "publisher-attestation.json",
        "--expected-quality-attestation-digest",
        "a" * 64,
        "--expected-quality-publisher-id",
        "test-publisher",
        "--ground-certificate-root",
        "ground-certificate-root.json",
        "--expected-ground-certificate-root-sha256",
        "b" * 64,
    ]


def test_select_rl_value_delegates_only_registered_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_freeze(**kwargs):
        captured.update(kwargs)
        return {"schema": "fixture", "record_digest": "a" * 64}

    monkeypatch.setattr("isingfold.rl.experiment_selection.freeze_rl_value_experiment", fake_freeze)
    args = build_parser().parse_args(
        [
            "select-rl-value",
            "--grid",
            str(GRID),
            "--representation-selection-receipt",
            "representation.json",
            "--evaluations-root",
            "evaluations",
            "--out",
            str(tmp_path / "freeze.json"),
        ]
    )

    args.func(args)

    assert captured == {
        "grid_path": str(GRID),
        "representation_receipt_path": "representation.json",
        "evaluations_root": "evaluations",
        "output_path": str(tmp_path / "freeze.json"),
    }


def test_rl_value_evaluation_cell_uses_only_the_grid_protocol(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluations = []
    validated = []
    monkeypatch.setattr(
        cli,
        "_load_representation_selection",
        lambda *args, **kwargs: ("if-dual", "a" * 64, "b" * 64),
    )
    monkeypatch.setattr(
        cli,
        "_validate_rl_value_run_for_evaluation",
        lambda *args, **kwargs: validated.append((args, kwargs)),
    )
    monkeypatch.setattr(cli, "cmd_evaluate", evaluations.append)
    common = [
        "--grid",
        str(GRID),
        "--corpus",
        "prepared",
        "--selector",
        "selector",
        *_quality_args(),
        "--run-root",
        str(tmp_path / "runs"),
        "--evaluations-root",
        str(tmp_path / "evaluations"),
        "--complete-config",
        str(ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json"),
        "--bootstrap-bank",
        "validation-bootstrap-bank",
        "--expected-bootstrap-plan-sha256",
        "c" * 64,
        "--expected-bootstrap-manifest-sha256",
        "d" * 64,
        "--representation-selection-receipt",
        "representation.json",
        "--complete-config",
        str(ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json"),
        "--bootstrap-bank",
        "validation-bootstrap-bank",
        "--expected-bootstrap-plan-sha256",
        "c" * 64,
        "--expected-bootstrap-manifest-sha256",
        "d" * 64,
        "--device",
        "cpu",
        "--threads",
        "3",
    ]
    for index in (0, 17):
        args = build_parser().parse_args(["evaluate-rl-value-cell", "--index", str(index), *common])
        args.func(args)

    assert len(validated) == 2
    assert len(evaluations) == 2
    first, last = evaluations
    assert first.partition == last.partition == "validation"
    assert first.seed == last.seed == 44021
    assert first.repetitions == last.repetitions == 4
    assert first.audit_reads == last.audit_reads == 4096
    assert first.margin == last.margin == 0.02
    assert first.greedy is last.greedy is False
    assert first.deterministic is last.deterministic is True
    assert first.threads == last.threads == 3
    assert first.checkpoint.endswith(
        "rl_value/rl-000-selected-simpler-supervised-s1103/checkpoint.pt"
    )
    assert last.checkpoint.endswith("rl_value/rl-017-if-core-scratch-ppo-s3301/checkpoint.pt")
    assert first.out.endswith("rl-000-selected-simpler-supervised-s1103")
    assert last.out.endswith("rl-017-if-core-scratch-ppo-s3301")
    assert validated[0][1]["expected_model_family"] == "if-dual"
    assert validated[1][1]["expected_model_family"] == "if-core"


@pytest.mark.parametrize(
    "forbidden",
    [
        ["--partition", "test"],
        ["--seed", "9"],
        ["--greedy"],
        ["--no-deterministic"],
        ["--selected-simpler", "if-mlp"],
    ],
)
def test_rl_value_evaluation_cell_exposes_no_scientific_overrides(
    forbidden: list[str],
) -> None:
    base = [
        "evaluate-rl-value-cell",
        "--grid",
        str(GRID),
        "--index",
        "0",
        "--corpus",
        "prepared",
        "--selector",
        "selector",
        *_quality_args(),
        "--run-root",
        "runs",
        "--evaluations-root",
        "evaluations",
        "--representation-selection-receipt",
        "representation.json",
    ]

    with pytest.raises(SystemExit):
        build_parser().parse_args([*base, *forbidden])


def test_rl_value_run_binding_rejects_another_representation_receipt(
    tmp_path: Path,
) -> None:
    grid_digest = hashlib.sha256(GRID.read_bytes()).hexdigest()
    cell = json.loads(GRID.read_text())["stages"]["rl_value"][0]
    checkpoint = tmp_path / "rl_value" / cell["cell_id"] / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint-fixture")
    ppo_config = PPOConfig()
    contract = {
        "phase": "rl-value",
        "model_family": "if-dual",
        "method": cell["method"],
        "seed": cell["seed"],
        "grid_manifest_sha256": grid_digest,
        "grid_cell": cell["cell_id"],
        "selection_receipt_sha256": "a" * 64,
        "selection_record_digest": "b" * 64,
        "selection_mode": "validated-receipt",
        "hyperparameters": {
            **{
                field: getattr(ppo_config, field)
                for field in ppo_config.__dataclass_fields__
            },
            "ppo_collection_schedule": cli._ppo_collection_schedule_contract(),
            "ppo_update_control": cli._ppo_update_control_contract(ppo_config),
        },
    }
    payload = {
        "schema": "isingfold.training-run",
        "schema_version": cli.RUN_SCHEMA_VERSION,
        "phase": "rl-value",
        "model_family": "if-dual",
        "method": cell["method"],
        "experiment_contract": contract,
        "checkpoint_payload_digest": "c" * 64,
        "complete": True,
    }
    run = {**payload, "record_digest": content_digest(payload)}
    (checkpoint.parent / "run.json").write_bytes(canonical_json_bytes(run) + b"\n")

    cli._validate_rl_value_run_for_evaluation(
        checkpoint,
        expected_cell=cell,
        expected_model_family="if-dual",
        grid_digest=grid_digest,
        selection_receipt_sha256="a" * 64,
        selection_record_digest="b" * 64,
    )
    with pytest.raises(ValueError, match="representation selection"):
        cli._validate_rl_value_run_for_evaluation(
            checkpoint,
            expected_cell=cell,
            expected_model_family="if-dual",
            grid_digest=grid_digest,
            selection_receipt_sha256="d" * 64,
            selection_record_digest="b" * 64,
        )


def test_cli_grid_loader_rejects_an_incomplete_rl_value_configuration(
    tmp_path: Path,
) -> None:
    grid = json.loads(GRID.read_text())
    grid["stages"]["rl_value"][0]["method"] = "unregistered-method"
    path = tmp_path / "bad-grid.json"
    path.write_text(json.dumps(grid))

    with pytest.raises(ValueError, match="RL-value.*configuration|training method"):
        cli._load_grid(path)


def test_rl_value_hpc_launchers_preserve_scheduler_boundary() -> None:
    apollo = ROOT / "scripts" / "apollo_rl_value_eval.sh"
    goose = ROOT / "scripts" / "goose_rl_value_eval.sbatch"
    for script in (apollo, goose):
        subprocess.run(["bash", "-n", str(script)], check=True)
        text = script.read_text()
        assert "evaluate-rl-value-cell" in text
        assert "REPRESENTATION_SELECTION_RECEIPT" in text
        assert "--seed" not in text
        assert "--partition" not in text
        assert "--greedy" not in text
    assert "sbatch" not in apollo.read_text()
    goose_text = goose.read_text()
    assert "#SBATCH --array=0-17" in goose_text
    assert "SLURM_JOB_ID" in goose_text
    assert "SLURM_ARRAY_TASK_ID" in goose_text
    assert "/opt/slurm/bin/srun" in goose_text

    result = subprocess.run(["bash", str(goose)], check=False, capture_output=True)
    assert result.returncode == 69
    assert b"Slurm array job" in result.stderr
