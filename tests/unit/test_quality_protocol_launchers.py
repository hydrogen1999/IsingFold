"""Static contracts for the post-initializer-bank quality workflow launchers."""

from __future__ import annotations

from pathlib import Path
import subprocess

from isingfold.rl.cli import build_parser


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"


def _text(name: str) -> str:
    return (SCRIPTS / name).read_text(encoding="utf-8")


def test_apollo_quality_launchers_are_direct_and_fail_closed() -> None:
    launchers = (
        "apollo_quality_control.sh",
        "apollo_quality_resolution_shards.sh",
        "apollo_quality_preflight_shards.sh",
    )
    for name in launchers:
        text = _text(name)
        assert "sbatch" not in text
        assert "SLURM_JOB_ID" in text
        assert 'source "$script_dir/publication_runtime.sh"' in text
        assert "require_apollo_publication_runtime" in text
        assert "verify_runtime_source.sh" in text
        assert "${ISINGFOLD_EXECUTION_RUNTIME_SHA256:?" in text


def test_goose_quality_launchers_require_compute_nodes_and_srun() -> None:
    launchers = (
        "goose_quality_control.sbatch",
        "goose_quality_resolution_shards.sbatch",
        "goose_quality_resolution_shards_packed.sbatch",
        "goose_quality_preflight_shards.sbatch",
        "goose_quality_preflight_shards_packed.sbatch",
    )
    for name in launchers:
        text = _text(name)
        assert "SLURM_JOB_ID" in text
        assert "never on the login node" in text
        assert 'script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}' in text
        assert 'source "$script_dir/publication_runtime.sh"' in text
        assert "require_goose_publication_runtime" in text
        assert "/opt/slurm/bin/srun" in text
        assert '"${container_exec[@]}"' in text
        assert "${ISINGFOLD_EXECUTION_RUNTIME_SHA256:?" in text


def test_quality_control_plane_covers_every_serial_protocol_transition() -> None:
    commands = (
        "audit-quality-selector-device-parity",
        "plan-quality-resolution",
        "publish-quality-resolution-verifier-identity",
        "publish-quality-resolution-shard-pins",
        "merge-quality-resolution",
        "publish-quality-capacity-budget",
        "plan-quality-capacity-canary",
        "quality-capacity-canary",
        "verify-quality-capacity-canary",
        "plan-quality-preflight-shards",
        "merge-quality-preflight-shards",
        "merge-quality-labels",
        "quality-preflight",
        "verify-quality-resolution-binding",
        "verify-quality-training-input-readiness",
        "gates",
    )
    text = _text("quality_control_command.sh")
    for command in commands:
        assert command in text, command


def test_resolution_workers_execute_and_independently_verify_each_shard() -> None:
    for launcher in (
        "apollo_quality_resolution_shards.sh",
        "goose_quality_resolution_shards.sbatch",
        "goose_quality_resolution_shards_packed.sbatch",
    ):
        text = _text(launcher)
        assert "run-quality-resolution-shard" in text
        assert "verify-quality-resolution-shard" in text
        assert "EXPECTED_QUALITY_RESOLUTION_DELTA_MANIFEST_SHA256" in text
        assert "EXPECTED_QUALITY_RESOLUTION_VERIFIER_IDENTITY_SHA256" in text


def test_preflight_workers_cover_replay_shards_only() -> None:
    for launcher in (
        "apollo_quality_preflight_shards.sh",
        "goose_quality_preflight_shards.sbatch",
        "goose_quality_preflight_shards_packed.sbatch",
    ):
        text = _text(launcher)
        assert "run-quality-preflight-shard" in text
        assert "--shard-index" in text
        assert "--execution-runtime-sha256" in text
        assert " quality-preflight \\" not in text


def test_control_plane_requires_external_pin_lists_for_collection_steps() -> None:
    dispatcher = _text("quality_control_command.sh")
    common = _text("quality_protocol_common.sh")
    assert "ISINGFOLD_QUALITY_RESOLUTION_PIN_FILE" in dispatcher
    assert "ISINGFOLD_QUALITY_SHARD_PIN_FILE" in dispatcher
    assert "ISINGFOLD_QUALITY_PREFLIGHT_REPLAY_PIN_FILE" in dispatcher
    assert "read -r" in common
    assert "-L" in common


def test_resolution_verification_never_self_pins_generated_delta() -> None:
    for launcher in (
        "apollo_quality_resolution_shards.sh",
        "goose_quality_resolution_shards.sbatch",
        "goose_quality_resolution_shards_packed.sbatch",
    ):
        text = _text(launcher)
        assert "ISINGFOLD_QUALITY_RESOLUTION_DELTA_PIN_FILE" in text
        assert "quality_lookup_manifest_pin" in text
        assert 'publication_file_sha256 "$delta_manifest"' not in text


def test_goose_packed_quality_workers_fit_two_job_association() -> None:
    for launcher in (
        "goose_quality_resolution_shards_packed.sbatch",
        "goose_quality_preflight_shards_packed.sbatch",
    ):
        text = _text(launcher)
        assert "#SBATCH --array=0-1" in text
        assert '"${SLURM_ARRAY_TASK_COUNT:-}" != "2"' in text
        assert '! "$SLURM_ARRAY_TASK_ID" =~ ^[01]$' in text
        assert "registered_shard_count=64" in text
        assert "index=$SLURM_ARRAY_TASK_ID" in text
        assert "index=$((index + SLURM_ARRAY_TASK_COUNT))" in text
        assert 'if [[ "$processed" != "32" ]]' in text
        assert 'verify_runtime_source.sh" "$container_python" gpu' in text
        assert 'if [[ -e "$output" || -L "$output" ]]' in text


def test_goose_packed_resolution_verify_uses_external_delta_pins() -> None:
    text = _text("goose_quality_resolution_shards_packed.sbatch")
    assert "ISINGFOLD_QUALITY_RESOLUTION_DELTA_PIN_FILE" in text
    assert "EXPECTED_QUALITY_RESOLUTION_DELTA_PIN_FILE_SHA256" in text
    assert 'publication_file_sha256 "$ISINGFOLD_QUALITY_RESOLUTION_DELTA_PIN_FILE"' in text
    assert "quality_require_regular_pin_file" in text
    assert "quality_lookup_manifest_pin" in text
    assert "--expected-delta-manifest-sha256" in text
    assert "--expected-verifier-identity-sha256" in text
    assert 'publication_file_sha256 "$delta_manifest"' not in text


def test_manifest_pin_reader_and_lookup_fail_closed(tmp_path: Path) -> None:
    artifact = tmp_path / "shard-0"
    artifact.mkdir()
    pins = tmp_path / "pins.tsv"
    pins.write_text(f"{artifact}\t{'a' * 64}\n", encoding="utf-8")
    script = (
        'set -euo pipefail; source "$1"; '
        'quality_require_regular_pin_file PIN_FILE; '
        'quality_lookup_manifest_pin "$2" "$PIN_FILE"; '
        'printf "%s" "$quality_lookup_sha256"'
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "pin-test",
            str(SCRIPTS / "quality_protocol_common.sh"),
            str(artifact),
        ],
        env={"PIN_FILE": str(pins)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "a" * 64

    symlink = tmp_path / "pins-link.tsv"
    symlink.symlink_to(pins)
    rejected = subprocess.run(
        [
            "bash",
            "-c",
            script,
            "pin-test",
            str(SCRIPTS / "quality_protocol_common.sh"),
            str(artifact),
        ],
        env={"PIN_FILE": str(symlink)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 78
    assert "non-symlink regular file" in rejected.stderr


def test_manifest_pin_lookup_rejects_duplicate_external_authority(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "shard-0"
    pins = tmp_path / "pins.tsv"
    pins.write_text(
        f"{artifact}\t{'a' * 64}\n{artifact}\t{'b' * 64}\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; quality_lookup_manifest_pin "$2" "$3"',
            "pin-test",
            str(SCRIPTS / "quality_protocol_common.sh"),
            str(artifact),
            str(pins),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 64
    assert "duplicate path" in result.stderr


def test_control_dispatch_builds_a_complete_capacity_budget_command(
    tmp_path: Path,
) -> None:
    fake_cli = tmp_path / "fake-cli"
    fake_cli.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$FAKE_CLI_LOG"\n',
        encoding="utf-8",
    )
    fake_cli.chmod(0o755)
    log = tmp_path / "args.log"
    out = tmp_path / "capacity-budget.json"
    digest = "a" * 64
    environment = {
        "FAKE_CLI": str(fake_cli),
        "FAKE_CLI_LOG": str(log),
        "ISINGFOLD_CORPUS": "/input/corpus",
        "ISINGFOLD_SELECTOR_BUNDLE": "/input/selector",
        "QUALITY_ATTESTATION": "/input/attestation.json",
        "EXPECTED_QUALITY_ATTESTATION_DIGEST": digest,
        "EXPECTED_QUALITY_PUBLISHER_ID": "publisher",
        "GROUND_CERTIFICATE_ROOT": "/input/ground.json",
        "EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256": digest,
        "ISINGFOLD_QUALITY_INITIALIZER_BANK": "/input/bank",
        "EXPECTED_QUALITY_INITIALIZER_BANK_MANIFEST_SHA256": digest,
        "ISINGFOLD_COMPLETE_CONFIG": "/input/config.json",
        "ISINGFOLD_EXECUTION_RUNTIME_SHA256": digest,
        "ISINGFOLD_QUALITY_ACCELERATOR": "gpu",
        "ISINGFOLD_QUALITY_OPERATION": "publish-capacity-budget",
        "ISINGFOLD_QUALITY_OUT": str(out),
        "ISINGFOLD_CAPACITY_BUDGET_ID": "paper-budget-v1",
        "ISINGFOLD_CAPACITY_MAX_ARTIFACT_BYTES": "1000",
        "ISINGFOLD_CAPACITY_MAX_CPU_SECONDS": "2000",
        "ISINGFOLD_CAPACITY_MAX_ELAPSED_SECONDS": "3000",
        "ISINGFOLD_CAPACITY_AVAILABLE_WORKERS": "16",
        "ISINGFOLD_CAPACITY_MIN_SCRATCH_FREE_BYTES": "4000",
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; '
            "quality_initialize_protocol_arguments; source \"$2\"; "
            'quality_cli=("$FAKE_CLI"); run_quality_control_operation',
            "control-test",
            str(SCRIPTS / "quality_protocol_common.sh"),
            str(SCRIPTS / "quality_control_command.sh"),
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    arguments = log.read_text(encoding="utf-8").splitlines()
    assert arguments == [
        "publish-quality-capacity-budget",
        "--budget-id",
        "paper-budget-v1",
        "--maximum-artifact-bytes",
        "1000",
        "--maximum-cpu-seconds",
        "2000",
        "--maximum-elapsed-seconds",
        "3000",
        "--available-workers",
        "16",
        "--minimum-scratch-free-bytes",
        "4000",
        "--out",
        str(out),
    ]
    parsed = build_parser().parse_args(arguments)
    assert parsed.command == "publish-quality-capacity-budget"


def test_goose_output_scope_rejects_escape_and_accepts_child(tmp_path: Path) -> None:
    root = tmp_path / "work"
    root.mkdir()
    inside = root / "receipt.json"
    outside = tmp_path / "escape.json"
    command = (
        'set -euo pipefail; source "$1"; '
        "quality_require_output_under_root OUTPUT WORK_ROOT"
    )
    accepted = subprocess.run(
        ["bash", "-c", command, "scope-test", str(SCRIPTS / "quality_protocol_common.sh")],
        env={"OUTPUT": str(inside), "WORK_ROOT": str(root)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr

    rejected = subprocess.run(
        ["bash", "-c", command, "scope-test", str(SCRIPTS / "quality_protocol_common.sh")],
        env={"OUTPUT": str(outside), "WORK_ROOT": str(root)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 78
    assert "must be contained" in rejected.stderr


def test_output_root_creation_rejects_symlink_alias(tmp_path: Path) -> None:
    physical = tmp_path / "physical"
    physical.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(physical, target_is_directory=True)
    command = (
        'set -euo pipefail; source "$1"; quality_prepare_output_root OUTPUT_ROOT'
    )
    rejected = subprocess.run(
        ["bash", "-c", command, "root-test", str(SCRIPTS / "quality_protocol_common.sh")],
        env={"OUTPUT_ROOT": str(alias)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert rejected.returncode == 78
    assert "non-symlink directory" in rejected.stderr
