"""Runtime-identity and immutable-output contracts for quality launchers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64


def _runtime_receipt(digest: str = DIGEST) -> str:
    return json.dumps(
        {
            "checkpoint_runtime_registry_sha256": digest,
            "schema": "isingfold.runtime-validation-receipt",
            "schema_version": 1,
        },
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _bind_runtime(receipt: str, expected: str) -> subprocess.CompletedProcess[str]:
    command = (
        'set -euo pipefail; source "$1"; '
        'quality_bind_verified_runtime_identity "$2" "$3"; '
        'printf "%s" "$quality_verified_runtime_sha256"'
    )
    return subprocess.run(
        [
            "bash",
            "-c",
            command,
            "runtime-binding-test",
            str(SCRIPTS / "quality_protocol_common.sh"),
            receipt,
            sys.executable,
        ],
        env={"ISINGFOLD_EXECUTION_RUNTIME_SHA256": expected},
        text=True,
        capture_output=True,
        check=False,
    )


def test_runtime_receipt_binds_the_execution_identity() -> None:
    result = _bind_runtime(_runtime_receipt(), DIGEST)

    assert result.returncode == 0, result.stderr
    assert result.stdout == DIGEST


def test_runtime_receipt_rejects_a_declared_identity_mismatch() -> None:
    result = _bind_runtime(_runtime_receipt(), OTHER_DIGEST)

    assert result.returncode == 78
    assert "differs from the verified runtime" in result.stderr


def test_runtime_receipt_parser_rejects_duplicate_json_keys() -> None:
    receipt = (
        '{"checkpoint_runtime_registry_sha256":"'
        + DIGEST
        + '","checkpoint_runtime_registry_sha256":"'
        + OTHER_DIGEST
        + '","schema":"isingfold.runtime-validation-receipt","schema_version":1}'
    )

    result = _bind_runtime(receipt, DIGEST)

    assert result.returncode == 78
    assert "not strict canonical JSON" in result.stderr


def test_all_quality_launchers_bind_the_captured_runtime_before_cli() -> None:
    launchers = (
        "apollo_quality_control.sh",
        "goose_quality_control.sbatch",
        "apollo_quality_resolution_shards.sh",
        "goose_quality_resolution_shards.sbatch",
        "goose_quality_resolution_shards_packed.sbatch",
        "apollo_quality_preflight_shards.sh",
        "goose_quality_preflight_shards.sbatch",
        "goose_quality_preflight_shards_packed.sbatch",
    )
    for name in launchers:
        text = (SCRIPTS / name).read_text(encoding="utf-8")
        capture = text.index("runtime_validation_json=$(")
        binding = text.index("quality_bind_verified_runtime_identity")
        cli = text.index("-m isingfold.rl.cli")
        assert capture < binding < cli, name


def test_verifier_publication_rejects_a_runtime_not_observed_by_launcher(
    tmp_path: Path,
) -> None:
    out = tmp_path / "verifier.json"
    command = (
        'set -euo pipefail; source "$1"; source "$2"; '
        'quality_verified_runtime_sha256="$3"; quality_cli=(/usr/bin/true); '
        "run_quality_control_operation"
    )
    environment = {
        "ISINGFOLD_QUALITY_OPERATION": "publish-resolution-verifier",
        "ISINGFOLD_QUALITY_OUT": str(out),
        "ISINGFOLD_QUALITY_RESOLUTION_PLAN": "/input/plan.json",
        "EXPECTED_QUALITY_RESOLUTION_PLAN_SHA256": DIGEST,
        "ISINGFOLD_RESOLUTION_VERIFIER_RUNTIME_SHA256": OTHER_DIGEST,
        "ISINGFOLD_RESOLUTION_ATTESTOR_ID": "independent-verifier",
    }

    result = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "verifier-runtime-test",
            str(SCRIPTS / "quality_protocol_common.sh"),
            str(SCRIPTS / "quality_control_command.sh"),
            DIGEST,
        ],
        env={**os.environ, **environment},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 78
    assert "verifier runtime must equal the verified execution runtime" in result.stderr
    assert not out.exists()


def test_capacity_verification_dispatches_to_an_explicit_output() -> None:
    text = (SCRIPTS / "quality_control_command.sh").read_text(encoding="utf-8")
    branch = text.split("    verify-capacity)", 1)[1].split("      ;;", 1)[0]

    assert '--out "$ISINGFOLD_QUALITY_OUT"' in branch
    assert '> "$ISINGFOLD_QUALITY_OUT"' not in branch
