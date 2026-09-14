from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import subprocess

import pytest

from isingfold.rl.cli import build_parser


ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts" / "publication_runtime.sh"
GOOSE_LAUNCHERS = tuple(
    path
    for path in sorted((ROOT / "scripts").glob("goose_*.sbatch"))
    if path.name
    not in {
        "goose_build_isingfold_runtime.sbatch",
        "goose_verify_isingfold_runtime.sbatch",
    }
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_environment(tmp_path: Path) -> dict[str, str]:
    apptainer = tmp_path / "apptainer"
    apptainer.write_text("#!/usr/bin/env bash\nexit 0\n")
    apptainer.chmod(0o755)
    image = tmp_path / "isingfold.sif"
    image.write_bytes(b"pinned test image")
    output_base = tmp_path / "publication-output"
    output_base.mkdir(exist_ok=True)
    build_receipt = tmp_path / "build-receipt.json"
    build_receipt.write_bytes(b"pinned build receipt")
    manifest = tmp_path / "wheelhouse.manifest.json"
    manifest.write_bytes(b"pinned wheelhouse manifest")
    project_wheel = tmp_path / "isingfold-0.1.0-fixture.whl"
    project_wheel.write_bytes(b"pinned project wheel")
    return {
        **os.environ,
        "SLURM_JOB_ID": "123",
        "SLURM_JOB_PARTITION": "gpu",
        "SLURM_CPUS_PER_TASK": "4",
        "ISINGFOLD_APPTAINER": str(apptainer),
        "ISINGFOLD_APPTAINER_SHA256": _sha256(apptainer),
        "ISINGFOLD_CONTAINER_IMAGE": str(image),
        "ISINGFOLD_CONTAINER_IMAGE_SHA256": _sha256(image),
        "ISINGFOLD_CONTAINER_PYTHON": "/opt/isingfold/venv/bin/python",
        "ISINGFOLD_GOOSE_OUTPUT_BASE": str(output_base),
        "ISINGFOLD_RUNTIME_BUILD_RECEIPT": str(build_receipt),
        "ISINGFOLD_RUNTIME_EXPECTED_BUILD_RECEIPT_SHA256": _sha256(build_receipt),
        "ISINGFOLD_RUNTIME_WHEELHOUSE_MANIFEST": str(manifest),
        "ISINGFOLD_RUNTIME_EXPECTED_WHEELHOUSE_MANIFEST_SHA256": _sha256(manifest),
        "ISINGFOLD_RUNTIME_PROJECT_WHEEL": str(project_wheel),
        "ISINGFOLD_RUNTIME_EXPECTED_PROJECT_WHEEL_SHA256": _sha256(project_wheel),
        "ISINGFOLD_RUNTIME_EXPECTED_NATIVE_EXTENSION_SHA256": "1" * 64,
        "ISINGFOLD_RUNTIME_EXPECTED_INSTALLATION_SHA256": "2" * 64,
        "ISINGFOLD_RUNTIME_EXPECTED_PYTHON_SHA256": "4" * 64,
        "ISINGFOLD_RUNTIME_EXPECTED_SOURCE_SHA256": "3" * 64,
        "PYTHONPATH": "/host/must/not/leak",
        "APPTAINERENV_PYTHONPATH": "/injected/must/not/leak",
        "APPTAINER_MOUNT": "type=bind,src=/,dst=/host-root",
    }


def _prepare_runtime(
    environment: dict[str, str],
    *,
    accelerator: str = "cpu",
    output_root: str | None = None,
) -> subprocess.CompletedProcess[str]:
    configured_base = environment.get("ISINGFOLD_GOOSE_OUTPUT_BASE", "/invalid-output-base")
    root = output_root or str(Path(configured_base) / "cell")
    if output_root is None and Path(configured_base).is_dir():
        Path(root).mkdir(exist_ok=True)
    return subprocess.run(
        [
            "bash",
            "-c",
            (
                'set -euo pipefail; source "$1/publication_runtime.sh"; '
                'threads=${TEST_THREADS:-}; '
                'require_goose_publication_runtime "$1" "$2" "$3"; '
                "printf 'python=<%s>\\n' \"$container_python\"; "
                "printf 'exec='; printf '<%s>' \"${container_exec[@]}\"; "
                "printf '\\nenv=<%s>|<%s>|<%s>\\n' "
                '"${PYTHONPATH-unset}" "${APPTAINERENV_PYTHONPATH-unset}" '
                '"${APPTAINER_MOUNT-unset}"'
            ),
            "isingfold-goose-runtime-test",
            str(ROOT / "scripts"),
            accelerator,
            root,
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_goose_runtime_helper_authenticates_slurm_apptainer_and_image(
    tmp_path: Path,
) -> None:
    result = _prepare_runtime(_runtime_environment(tmp_path))

    assert result.returncode == 0, result.stderr
    assert "python=</opt/isingfold/venv/bin/python>" in result.stdout
    assert "<--cleanenv><--containall><--no-home>" in result.stdout
    assert "<--no-mount><bind-paths,hostfs>" in result.stdout
    assert "<--pwd>" in result.stdout
    assert "<--nv>" not in result.stdout
    assert "env=<unset>|<unset>|<unset>" in result.stdout

    gpu = _prepare_runtime(_runtime_environment(tmp_path), accelerator="gpu")
    assert gpu.returncode == 0, gpu.stderr
    assert "<--nv>" in gpu.stdout


def test_goose_runtime_helper_rejects_login_node_and_digest_drift(
    tmp_path: Path,
) -> None:
    environment = _runtime_environment(tmp_path)
    environment.pop("SLURM_JOB_ID")
    login = _prepare_runtime(environment)
    assert login.returncode == 69
    assert "Slurm allocation" in login.stderr

    environment = _runtime_environment(tmp_path)
    environment["ISINGFOLD_CONTAINER_IMAGE_SHA256"] = "0" * 64
    drift = _prepare_runtime(environment)
    assert drift.returncode == 78
    assert "image differs" in drift.stderr


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"ISINGFOLD_APPTAINER": None}, "ISINGFOLD_APPTAINER"),
        ({"ISINGFOLD_APPTAINER_SHA256": "bad"}, "64-hex"),
        ({"ISINGFOLD_APPTAINER_SHA256": "0" * 64}, "runtime differs"),
        ({"ISINGFOLD_CONTAINER_IMAGE": None}, "ISINGFOLD_CONTAINER_IMAGE"),
        ({"ISINGFOLD_CONTAINER_IMAGE_SHA256": "bad"}, "64-hex"),
        ({"ISINGFOLD_CONTAINER_PYTHON": "python3"}, "/opt/isingfold/venv/bin/python"),
        ({"ISINGFOLD_GOOSE_OUTPUT_BASE": None}, "OUTPUT_BASE"),
    ],
)
def test_goose_runtime_helper_rejects_missing_or_drifting_pins(
    tmp_path: Path,
    mutation: dict[str, str | None],
    message: str,
) -> None:
    environment = _runtime_environment(tmp_path)
    for name, value in mutation.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value

    result = _prepare_runtime(environment)

    assert result.returncode != 0
    assert message in result.stderr


def test_goose_runtime_helper_limits_writes_to_the_approved_base(
    tmp_path: Path,
) -> None:
    environment = _runtime_environment(tmp_path)
    outside = tmp_path / "outside" / "cell"
    rejected = _prepare_runtime(environment, output_root=str(outside))
    assert rejected.returncode == 78
    assert "inside ISINGFOLD_GOOSE_OUTPUT_BASE" in rejected.stderr
    assert not outside.exists()

    outside.mkdir(parents=True)
    rejected_existing = _prepare_runtime(environment, output_root=str(outside))
    assert rejected_existing.returncode == 78
    assert "inside ISINGFOLD_GOOSE_OUTPUT_BASE" in rejected_existing.stderr

    inside = Path(environment["ISINGFOLD_GOOSE_OUTPUT_BASE"]) / "new" / "cell"
    accepted_inside = _prepare_runtime(environment, output_root=str(inside))
    assert accepted_inside.returncode == 0, accepted_inside.stderr
    assert inside.is_dir()

    escaped_parent = tmp_path / "symlink-target"
    escaped_parent.mkdir()
    link = Path(environment["ISINGFOLD_GOOSE_OUTPUT_BASE"]) / "link"
    link.symlink_to(escaped_parent, target_is_directory=True)
    escaped = escaped_parent / "cell"
    rejected_symlink = _prepare_runtime(environment, output_root=str(link / "cell"))
    assert rejected_symlink.returncode == 78
    assert "symbolic links" in rejected_symlink.stderr
    assert not escaped.exists()

    environment["ISINGFOLD_GOOSE_OUTPUT_BASE"] = str(ROOT.parent)
    source_ancestor = _prepare_runtime(environment, output_root=str(ROOT / "runs"))
    assert source_ancestor.returncode == 78
    assert "contain the source root" in source_ancestor.stderr

    environment["ISINGFOLD_GOOSE_OUTPUT_BASE"] = str(ROOT / "src")
    source_subtree = _prepare_runtime(environment, output_root=str(ROOT / "src"))
    assert source_subtree.returncode == 78
    assert "must be under runs" in source_subtree.stderr

    environment = _runtime_environment(tmp_path)
    environment["ISINGFOLD_GOOSE_OUTPUT_BASE"] = str(tmp_path / "not-created")
    missing_base = _prepare_runtime(environment)
    assert missing_base.returncode == 78
    assert "exist and be writable before sbatch" in missing_base.stderr


def test_goose_runtime_helper_uses_read_only_inputs_and_exact_writable_output(
    tmp_path: Path,
) -> None:
    environment = _runtime_environment(tmp_path)
    read_only = tmp_path / "published-input"
    read_only.mkdir()
    environment["ISINGFOLD_GOOSE_INPUT_ROOTS"] = str(read_only)
    output = Path(environment["ISINGFOLD_GOOSE_OUTPUT_BASE"]) / "cell"
    output.mkdir()

    result = _prepare_runtime(environment, output_root=str(output))

    assert result.returncode == 0, result.stderr
    assert f"{ROOT}:{ROOT}:ro" in result.stdout
    assert f"{read_only}:{read_only}:ro" in result.stdout
    assert f"{output}:{output}:rw" in result.stdout
    output_base = environment["ISINGFOLD_GOOSE_OUTPUT_BASE"]
    assert f"{output_base}:{output_base}:ro" in result.stdout
    assert f"{output_base}:{output_base}:rw" not in result.stdout


def test_goose_runtime_helper_rejects_python_shadow_and_thread_oversubscription(
    tmp_path: Path,
) -> None:
    environment = _runtime_environment(tmp_path)
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    environment["ISINGFOLD_GOOSE_INPUT_ROOTS"] = "/opt/isingfold"
    shadowed = _prepare_runtime(environment)
    assert shadowed.returncode == 78
    assert "shadow" in shadowed.stderr

    environment = _runtime_environment(tmp_path)
    environment["TEST_THREADS"] = "5"
    oversubscribed = _prepare_runtime(environment)
    assert oversubscribed.returncode == 78
    assert "SLURM_CPUS_PER_TASK" in oversubscribed.stderr


def test_every_goose_launcher_enters_the_pinned_container() -> None:
    assert GOOSE_LAUNCHERS
    for launcher in GOOSE_LAUNCHERS:
        text = launcher.read_text()
        assert 'script_dir=${ISINGFOLD_SCRIPT_DIR:?set ISINGFOLD_SCRIPT_DIR}' in text, (
            launcher.name
        )
        assert 'dirname -- "${BASH_SOURCE[0]}"' not in text, launcher.name
        source_index = text.index('source "$script_dir/publication_runtime.sh"')
        runtime_index = text.index("require_goose_publication_runtime")
        source_guard_index = text.index("verify_runtime_source.sh")
        assert source_index < runtime_index < source_guard_index, launcher.name
        assert '"${container_exec[@]}"' in text, launcher.name
        assert text.count("/opt/slurm/bin/srun") == text.count(
            '"${container_exec[@]}"'
        ), launcher.name
        assert '"$python_bin" -m isingfold.rl.cli' not in text, launcher.name
        assert "ISINGFOLD_PYTHON:-python3" not in text, launcher.name
        log_directives = [
            line
            for line in text.splitlines()
            if line.startswith(("#SBATCH --output=", "#SBATCH --error="))
        ]
        assert len(log_directives) == 2, launcher.name
        assert all("/" not in line.split("=", 1)[1] for line in log_directives), (
            launcher.name
        )

    cpu_launchers = {
        "goose_bootstrap_bank.sbatch",
        "goose_bootstrap_bank_packed.sbatch",
        "goose_seal_bootstrap_bank.sbatch",
        "goose_initializer_bank.sbatch",
        "goose_initializer_bank_packed.sbatch",
        "goose_selector_labels.sbatch",
        "goose_seal_initializer_bank.sbatch",
        "goose_final_strength_audit.sbatch",
    }
    for launcher in GOOSE_LAUNCHERS:
        text = launcher.read_text()
        if launcher.name == "goose_quality_shards.sbatch":
            assert "require_quality_accelerator_control" in text
            assert '"$script_dir" "$quality_accelerator"' in text
            assert '#SBATCH --gres=gpu:1' in text
            continue
        mode = "cpu" if launcher.name in cpu_launchers else "gpu"
        assert re.search(
            rf'require_goose_publication_runtime\s+(?:\\\s+)?"\$script_dir"'
            rf"\s+(?:\\\s+)?{mode}",
            text,
        ), launcher.name
        assert ("#SBATCH --gres=gpu" in text) is (mode == "gpu")


def test_quality_launcher_derives_a_nonoverlapping_two_way_stride() -> None:
    text = (ROOT / "scripts" / "goose_quality_shards.sbatch").read_text()
    assert 'if [[ "$step" != "$SLURM_ARRAY_TASK_COUNT" ]]' in text
    assert "requires exactly the registered two-element Goose array" in text


def test_initializer_bank_sealer_is_single_job_and_binds_public_inputs() -> None:
    text = (ROOT / "scripts" / "goose_seal_initializer_bank.sbatch").read_text()

    assert '[[ -z "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]' in text
    assert "seal-initializer-bank" in text
    for argument in (
        '--corpus "$ISINGFOLD_CORPUS"',
        '--config "$ISINGFOLD_COMPLETE_CONFIG"',
        '--plan "$ISINGFOLD_INITIALIZER_BANK_PLAN"',
        '--expected-plan-sha256 "$ISINGFOLD_INITIALIZER_BANK_PLAN_SHA256"',
        '--bank "$ISINGFOLD_INITIALIZER_BANK"',
    ):
        assert argument in text


def test_bootstrap_bank_packed_launcher_uses_one_allocation_and_round_robin() -> None:
    text = (ROOT / "scripts" / "goose_bootstrap_bank_packed.sbatch").read_text()

    assert "#SBATCH --array" not in text
    assert '[[ -z "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]' in text
    assert '--ntasks="$workers" --cpus-per-task=1' in text
    assert "index=${SLURM_PROCID:?}" in text
    assert "index=$((index + workers))" in text
    assert "generate-bootstrap-bank-shard" in text
    assert '"$ISINGFOLD_BOOTSTRAP_BANK_LOG_ROOT/shard-$index.err"' in text
    assert '"$ISINGFOLD_BOOTSTRAP_BANK_LOG_ROOT/shard-$index.out"' in text


def test_bootstrap_bank_sealer_is_single_job_and_binds_public_plan() -> None:
    text = (ROOT / "scripts" / "goose_seal_bootstrap_bank.sbatch").read_text()

    assert "#SBATCH --array" not in text
    assert '[[ -z "${SLURM_JOB_ID:-}" || -n "${SLURM_ARRAY_TASK_ID:-}" ]]' in text
    assert "seal-bootstrap-bank" in text
    for argument in (
        '--grid "$ISINGFOLD_GRID"',
        '--corpus "$ISINGFOLD_CORPUS"',
        '--config "$ISINGFOLD_COMPLETE_CONFIG"',
        '--protocol-preset "$ISINGFOLD_BOOTSTRAP_PRESET"',
        '--plan "$ISINGFOLD_BOOTSTRAP_BANK_PLAN"',
        '--expected-plan-sha256 "$ISINGFOLD_BOOTSTRAP_BANK_PLAN_SHA256"',
        '--bank "$ISINGFOLD_BOOTSTRAP_BANK"',
    ):
        assert argument in text


def test_selector_launchers_separate_cpu_labeling_from_gpu_fitting() -> None:
    labels = (ROOT / "scripts" / "goose_selector_labels.sbatch").read_text()
    fit = (ROOT / "scripts" / "goose_fit_selector.sbatch").read_text()

    assert "label-selector-data" in labels
    assert "#SBATCH --gres=gpu" not in labels
    assert "fit-selector" in fit
    assert "#SBATCH --gres=gpu:1" in fit
    assert '--device cuda' in fit
    assert "ISINGFOLD_SELECTOR_LABEL_WORK_ROOT" in labels
    assert (
        'require_goose_publication_runtime "$script_dir" cpu '
        '"$ISINGFOLD_SELECTOR_LABEL_WORK_ROOT"'
    ) in labels
    assert "ISINGFOLD_SELECTOR_BUNDLE_WORK_ROOT" in fit
    assert (
        'require_goose_publication_runtime "$script_dir" gpu '
        '"$ISINGFOLD_SELECTOR_BUNDLE_WORK_ROOT"'
    ) in fit
    assert '-e "$ISINGFOLD_SELECTOR_LABELS"' in labels
    assert '-e "$ISINGFOLD_SELECTOR_BUNDLE"' in fit
    for text in (labels, fit):
        assert '--quality-attestation "$QUALITY_ATTESTATION"' in text
        assert '--ground-certificate-root "$GROUND_CERTIFICATE_ROOT"' in text


def test_evaluation_shard_launcher_owns_the_validated_thread_count() -> None:
    text = (ROOT / "scripts" / "goose_evaluation_shards.sbatch").read_text()
    assert 'threads=${ISINGFOLD_THREADS:-${SLURM_CPUS_PER_TASK:-}}' in text
    assert '"$argument" == "--threads"' in text
    workflow_index = text.index('"${workflow_args[@]}"')
    validated_index = text.index('--threads "$threads"')
    assert workflow_index < validated_index


def test_cli_disables_long_option_abbreviations_for_every_command() -> None:
    parser = build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    assert parser.allow_abbrev is False
    assert subparsers.choices
    assert all(command.allow_abbrev is False for command in subparsers.choices.values())
