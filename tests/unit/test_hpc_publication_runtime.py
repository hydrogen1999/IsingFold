"""Executable contracts for publication launchers on HPC systems."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
PUBLICATION_RUNTIME = SCRIPTS / "publication_runtime.sh"


def _fake_venv(tmp_path: Path) -> tuple[Path, Path]:
    venv = tmp_path / "publication-venv"
    binary = venv / "bin" / "python"
    binary.parent.mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = /fixture\n")
    binary.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == "-I" && ${2:-} == "-m" && ${3:-} == "pip" \
   && ${4:-} == "check" ]]; then
  printf 'pip check\\n' >> "$FAKE_PIP_LOG"
  exit "${FAKE_PIP_EXIT:-0}"
fi
exec "$FAKE_REAL_PYTHON" "$@"
"""
    )
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    return binary, venv


def _runtime_environment(tmp_path: Path) -> tuple[dict[str, str], Path, Path]:
    python_bin, _ = _fake_venv(tmp_path)
    lock = tmp_path / "requirements.lock"
    lock.write_text("isingfold-publication-fixture==1\n")
    pip_log = tmp_path / "pip-check.log"
    environment = os.environ.copy()
    for name in (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "ISINGFOLD_PYTHON",
        "ISINGFOLD_ENV_LOCK",
        "ISINGFOLD_ENV_LOCK_SHA256",
        "FAKE_PIP_EXIT",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "ISINGFOLD_PYTHON": str(python_bin),
            "ISINGFOLD_ENV_LOCK": str(lock),
            "ISINGFOLD_ENV_LOCK_SHA256": hashlib.sha256(lock.read_bytes()).hexdigest(),
            "FAKE_PIP_LOG": str(pip_log),
            "FAKE_REAL_PYTHON": sys.executable,
        }
    )
    return environment, python_bin, pip_log


def _run_publication_runtime(environment: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; require_apollo_publication_runtime; '
            'printf "python_bin=%s\\n" "$python_bin"',
            "publication-runtime-test",
            str(PUBLICATION_RUNTIME),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_quality_accelerator_control(
    environment: dict[str, str], accelerator: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "/bin/bash",
            "-c",
            'set -euo pipefail; source "$1"; '
            'require_quality_accelerator_control "$2" "$3"',
            "quality-accelerator-test",
            str(PUBLICATION_RUNTIME),
            accelerator,
            sys.executable,
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def _run_cuda_argument_control(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "/bin/bash",
            "-c",
            'set -euo pipefail; source "$1"; shift; '
            'require_cuda_cli_arguments "$@"',
            "cuda-argument-test",
            str(PUBLICATION_RUNTIME),
            *arguments,
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def test_every_apollo_launcher_uses_publication_runtime_before_source_guard() -> None:
    launchers = [
        path
        for path in sorted(SCRIPTS.glob("apollo_*.sh"))
        if path.name != "apollo_build_isingfold_runtime.sh"
    ]

    assert launchers
    for launcher in launchers:
        text = launcher.read_text()
        source_index = text.index('source "$script_dir/publication_runtime.sh"')
        runtime_index = text.index("require_apollo_publication_runtime")
        source_guard_index = text.index("verify_runtime_source.sh")
        assert source_index < runtime_index < source_guard_index, launcher.name
        assert "ISINGFOLD_PYTHON:-python3" not in text, launcher.name


def test_apollo_publication_runtime_accepts_only_pinned_venv_and_lock(
    tmp_path: Path,
) -> None:
    environment, python_bin, pip_log = _runtime_environment(tmp_path)

    completed = _run_publication_runtime(environment)

    assert completed.returncode == 0, completed.stderr
    assert f"python_bin={python_bin}" in completed.stdout
    assert pip_log.read_text() == "pip check\n"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"SLURM_JOB_ID": "1234"}, "rejects Slurm"),
        ({"ISINGFOLD_PYTHON": None}, "ISINGFOLD_PYTHON"),
        ({"ISINGFOLD_PYTHON": "python3"}, "absolute executable"),
        ({"ISINGFOLD_ENV_LOCK": None}, "ISINGFOLD_ENV_LOCK"),
        ({"ISINGFOLD_ENV_LOCK_SHA256": "not-a-digest"}, "64-hex"),
        ({"ISINGFOLD_ENV_LOCK_SHA256": "0" * 64}, "digest mismatch"),
        ({"FAKE_PIP_EXIT": "7"}, "pip check failed"),
    ],
)
def test_apollo_publication_runtime_fails_closed(
    tmp_path: Path,
    mutation: dict[str, str | None],
    message: str,
) -> None:
    environment, _, _ = _runtime_environment(tmp_path)
    for name, value in mutation.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value

    completed = _run_publication_runtime(environment)

    assert completed.returncode != 0
    assert message in completed.stderr


def test_apollo_publication_runtime_rejects_non_venv_executable(
    tmp_path: Path,
) -> None:
    environment, python_bin, _ = _runtime_environment(tmp_path)
    (python_bin.parents[1] / "pyvenv.cfg").unlink()

    completed = _run_publication_runtime(environment)

    assert completed.returncode != 0
    assert "pyvenv.cfg" in completed.stderr


def test_quality_cpu_requires_an_externally_pinned_exact_parity_receipt(
    tmp_path: Path,
) -> None:
    environment = os.environ.copy()
    assert _run_quality_accelerator_control(environment, "gpu").returncode == 0

    missing = _run_quality_accelerator_control(environment, "cpu")
    assert missing.returncode == 78
    assert "parity receipt" in missing.stderr

    payload = {
        "all_equal": True,
        "mismatch_count": 0,
        "schema": "isingfold.quality-selector-device-parity",
        "schema_version": 1,
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    receipt = tmp_path / "selector-parity.json"
    receipt.write_text(
        json.dumps(
            {**payload, "record_digest": hashlib.sha256(canonical).hexdigest()},
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    environment.update(
        {
            "ISINGFOLD_QUALITY_SELECTOR_PARITY_RECEIPT": str(receipt),
            "ISINGFOLD_QUALITY_SELECTOR_PARITY_RECEIPT_SHA256": hashlib.sha256(
                receipt.read_bytes()
            ).hexdigest(),
        }
    )

    accepted = _run_quality_accelerator_control(environment, "cpu")
    assert accepted.returncode == 0, accepted.stderr


def test_model_evaluation_requires_exactly_one_cuda_device_argument() -> None:
    assert _run_cuda_argument_control("--device", "cuda").returncode == 0
    assert _run_cuda_argument_control("--device=cuda").returncode == 0
    for arguments in (
        (),
        ("--device", "cpu"),
        ("--device=cpu",),
        ("--device", "cuda", "--device=cuda"),
    ):
        rejected = _run_cuda_argument_control(*arguments)
        assert rejected.returncode == 78
        assert "device cuda" in rejected.stderr

    for launcher_name in (
        "apollo_evaluation_shards.sh",
        "goose_evaluation_shards.sbatch",
    ):
        text = (SCRIPTS / launcher_name).read_text(encoding="utf-8")
        assert "require_cuda_cli_arguments" in text


def test_apollo_selector_launchers_keep_labeling_cpu_and_fitting_cuda() -> None:
    labels = (SCRIPTS / "apollo_selector_labels.sh").read_text()
    fit = (SCRIPTS / "apollo_fit_selector.sh").read_text()

    assert "label-selector-data" in labels
    assert 'verify_runtime_source.sh" "$python_bin" cpu' in labels
    assert "fit-selector" in fit
    assert 'verify_runtime_source.sh" "$python_bin" gpu' in fit
    assert '--device cuda' in fit


def test_apollo_initializer_sealer_binds_the_exact_public_plan() -> None:
    text = (SCRIPTS / "apollo_seal_initializer_bank.sh").read_text()

    assert "seal-initializer-bank" in text
    for argument in (
        '--corpus "$corpus"',
        '--config "$config"',
        '--plan "$plan"',
        '--expected-plan-sha256 "$plan_sha256"',
        '--bank "$bank"',
    ):
        assert argument in text
