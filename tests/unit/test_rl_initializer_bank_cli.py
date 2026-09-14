from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import isingfold.rl.cli as cli
import isingfold.rl.complete_system as complete_system
import isingfold.rl.data.prepared as prepared_module
import isingfold.rl.initializer_bank as bank_module
from isingfold.rl.cli import build_parser


ROOT = Path(__file__).resolve().parents[2]


def test_bootstrap_presets_keep_representation_and_rl_value_seed_domains_separate() -> None:
    grid = json.loads((ROOT / "configs" / "rl_grid_hybrid_v1.json").read_text())

    representation = cli._bootstrap_protocol_from_grid(grid, "representation-validation")
    rl_value = cli._bootstrap_protocol_from_grid(grid, "validation")

    assert representation[1:] == ("val", 33049, 4)
    assert rl_value[1:] == ("val", 44021, 4)
    assert representation[0] == grid["representation_evaluation"]
    assert rl_value[0] == grid["rl_value_evaluation"]


def test_initializer_bank_commands_expose_pinned_resumable_lifecycle() -> None:
    parser = build_parser()

    plan = parser.parse_args(
        [
            "plan-initializer-bank",
            "--corpus",
            "prepared-v4",
            "--config",
            "complete.json",
            "--training-seed",
            "1103",
            "--episode-count",
            "6400",
            "--max-draws-per-conditional-episode",
            "8",
            "--out",
            "bank-plan.json",
        ]
    )
    generate = parser.parse_args(
        [
            "generate-initializer-bank-shard",
            "--corpus",
            "prepared-v4",
            "--config",
            "complete.json",
            "--plan",
            "bank-plan.json",
            "--expected-plan-sha256",
            "a" * 64,
            "--bank",
            "initializer-bank",
            "--shard-index",
            "3",
            "--shard-count",
            "16",
        ]
    )
    seal = parser.parse_args(
        [
            "seal-initializer-bank",
            "--corpus",
            "prepared-v4",
            "--config",
            "complete.json",
            "--plan",
            "bank-plan.json",
            "--expected-plan-sha256",
            "a" * 64,
            "--bank",
            "initializer-bank",
        ]
    )

    assert plan.func.__name__ == "cmd_plan_initializer_bank"
    assert plan.episode_count == 6400
    assert generate.func.__name__ == "cmd_generate_initializer_bank_shard"
    assert (generate.shard_index, generate.shard_count) == (3, 16)
    assert seal.func.__name__ == "cmd_seal_initializer_bank"


def test_evaluation_bootstrap_bank_commands_expose_pinned_k2_lifecycle() -> None:
    parser = build_parser()
    common = [
        "--grid",
        "grid.json",
        "--corpus",
        "prepared-v4",
        "--config",
        "complete-cache-v2.json",
        "--protocol-preset",
        "validation",
    ]
    plan = parser.parse_args(["plan-bootstrap-bank", *common, "--out", "bank/plan.json"])
    generate = parser.parse_args(
        [
            "generate-bootstrap-bank-shard",
            *common,
            "--plan",
            "bank/plan.json",
            "--expected-plan-sha256",
            "a" * 64,
            "--bank",
            "bank",
            "--shard-index",
            "3",
            "--shard-count",
            "16",
        ]
    )
    seal = parser.parse_args(
        [
            "seal-bootstrap-bank",
            *common,
            "--plan",
            "bank/plan.json",
            "--expected-plan-sha256",
            "a" * 64,
            "--bank",
            "bank",
        ]
    )

    assert plan.func is cli.cmd_plan_bootstrap_bank
    assert generate.func is cli.cmd_generate_bootstrap_bank_shard
    assert (generate.shard_index, generate.shard_count) == (3, 16)
    assert seal.func is cli.cmd_seal_bootstrap_bank

    representation = parser.parse_args(
        [
            "plan-bootstrap-bank",
            *common[:-1],
            "representation-validation",
            "--out",
            "representation-bank/plan.json",
        ]
    )
    assert representation.protocol_preset == "representation-validation"


def test_initializer_bank_shard_bounds_are_checked_before_runtime_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "generate-initializer-bank-shard",
            "--corpus",
            "prepared-v4",
            "--config",
            "complete.json",
            "--plan",
            "bank-plan.json",
            "--expected-plan-sha256",
            "a" * 64,
            "--bank",
            "initializer-bank",
            "--shard-index",
            "4",
            "--shard-count",
            "4",
        ]
    )
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: pytest.fail("invalid shard bounds must fail before file access"),
    )

    with pytest.raises(ValueError, match="shard index"):
        args.func(args)


def test_initializer_bank_runtime_inputs_open_public_train_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[tuple[str, bool]] = []
    tasks = (object(),)
    backend = object()
    runtime = {"runtime": "identity"}
    context = object()

    def load(corpus, *, partition, include_evaluator):
        assert corpus == "prepared-v4"
        opened.append((partition, include_evaluator))
        return SimpleNamespace(tasks=tasks, target_access=None)

    monkeypatch.setattr(prepared_module, "load_prepared_partition", load)
    monkeypatch.setattr(
        complete_system, "LACMinorminerInitializerBackend", lambda: backend
    )
    monkeypatch.setattr(
        bank_module, "lac_runtime_implementation_manifest", lambda value: runtime
    )
    monkeypatch.setattr(cli, "_context", lambda *args, **kwargs: context)
    monkeypatch.setattr(cli, "_corpus_manifest_digest", lambda *args: "a" * 64)

    result = cli._initializer_bank_public_inputs(
        SimpleNamespace(
            corpus="prepared-v4",
            config=str(ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json"),
            qubit_cap=None,
        )
    )

    assert opened == [("train", False)]
    assert result[0] == tasks
    assert result[1:4] == ("a" * 64, backend, runtime)
    assert result[-1] is context


def test_initializer_bank_hpc_launchers_respect_host_schedulers() -> None:
    apollo = (ROOT / "scripts" / "apollo_initializer_bank.sh").read_text()
    goose = (ROOT / "scripts" / "goose_initializer_bank.sbatch").read_text()

    assert "sbatch" not in apollo
    assert "generate-initializer-bank-shard" in apollo
    assert "SLURM_JOB_ID" in goose and "SLURM_ARRAY_TASK_ID" in goose
    assert "/opt/slurm/bin/srun" in goose
    assert "generate-initializer-bank-shard" in goose
    for text in (apollo, goose):
        assert "--expected-plan-sha256" in text
        assert "verify_runtime_source.sh" in text


def test_packed_goose_initializer_launcher_uses_one_cpu_bound_multi_task_step() -> None:
    packed = (
        ROOT / "scripts" / "goose_initializer_bank_packed.sbatch"
    ).read_text()

    assert "SLURM_ARRAY_TASK_ID" in packed
    assert "requires one non-array Slurm job" in packed
    assert "ISINGFOLD_INITIALIZER_BANK_WORKERS" in packed
    assert '--ntasks="$workers"' in packed
    assert "--cpu-bind=cores" in packed
    assert "SLURM_PROCID" in packed
    assert 'index=$((index + workers))' in packed
    assert 'run_shard "$index" &' not in packed


def test_bootstrap_bank_hpc_launchers_pin_runtime_and_respect_schedulers() -> None:
    apollo = (ROOT / "scripts" / "apollo_bootstrap_bank.sh").read_text()
    goose = (ROOT / "scripts" / "goose_bootstrap_bank.sbatch").read_text()

    assert "sbatch" not in apollo
    assert "generate-bootstrap-bank-shard" in apollo
    assert "SLURM_JOB_ID" in goose and "SLURM_ARRAY_TASK_ID" in goose
    assert "/opt/slurm/bin/srun" in goose
    assert "generate-bootstrap-bank-shard" in goose
    for text in (apollo, goose):
        assert "--grid" in text
        assert "--protocol-preset" in text
        assert "--expected-plan-sha256" in text
        assert "verify_runtime_source.sh" in text


def test_complete_training_launchers_require_seed_matched_bank_pins() -> None:
    launchers = (
        ROOT / "scripts" / "apollo_complete_train.sh",
        ROOT / "scripts" / "goose_complete_train.sbatch",
    )

    for launcher in launchers:
        text = launcher.read_text()
        assert "ISINGFOLD_INITIALIZER_BANK_ROOT" in text
        assert "ISINGFOLD_INITIALIZER_BANK_MANIFEST_SHA256_" in text
        assert "ISINGFOLD_COMPLETE_CONFIG" in text
        assert '--initializer-bank "$initializer_bank"' in text
        assert '--expected-initializer-bank-manifest-sha256 "$initializer_bank_sha256"' in text
        assert '--complete-config "$ISINGFOLD_COMPLETE_CONFIG"' in text


def test_rl_value_grid_launchers_require_seed_matched_bank_pins() -> None:
    launchers = (
        ROOT / "scripts" / "apollo_rl_grid.sh",
        ROOT / "scripts" / "goose_rl_grid.sbatch",
    )

    for launcher in launchers:
        text = launcher.read_text()
        assert "ISINGFOLD_INITIALIZER_BANK_ROOT" in text
        assert "ISINGFOLD_INITIALIZER_BANK_MANIFEST_SHA256_" in text
        assert "ISINGFOLD_COMPLETE_CONFIG" in text
        assert '--initializer-bank "$initializer_bank"' in text
        assert '--expected-initializer-bank-manifest-sha256 "$initializer_bank_sha256"' in text
        assert '--complete-config "$ISINGFOLD_COMPLETE_CONFIG"' in text
