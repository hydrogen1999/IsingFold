"""CLI surface for the preregistered post-freeze strength audit."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from isingfold.rl.cli import (
    build_parser,
    cmd_merge_final_strength_audit,
    cmd_plan_final_strength_audit,
    cmd_run_final_strength_audit_shard,
    cmd_seal_final_strength_audit_execution,
)


_DIGESTS = tuple(str(index) * 64 for index in range(1, 10))


def _source_arguments() -> list[str]:
    result: list[str] = []
    for index in range(3):
        result.extend(("--learned-evaluation", f"learned-{index}"))
        result.extend(("--expected-learned-report-sha256", _DIGESTS[index]))
        result.extend(("--external-evaluation", f"stock-{index}"))
        result.extend(("--expected-external-report-sha256", _DIGESTS[index + 3]))
    return result


def _test_access_arguments() -> list[str]:
    return [
        "--corpus",
        "prepared-v4",
        "--quality-attestation",
        "publisher.json",
        "--expected-quality-attestation-digest",
        _DIGESTS[6],
        "--expected-quality-publisher-id",
        "publisher-one",
        "--ground-certificate-root",
        "ground-root.json",
        "--expected-ground-certificate-root-sha256",
        _DIGESTS[7],
    ]


def test_final_strength_plan_parser_has_no_post_test_or_free_statistical_inputs() -> None:
    parsed = build_parser().parse_args(
        [
            "plan-final-strength-audit",
            "--grid",
            "grid.json",
            "--corpus",
            "prepared-v4",
            "--selector",
            "selector",
            "--quality-attestation",
            "publisher.json",
            "--expected-quality-attestation-digest",
            _DIGESTS[0],
            "--expected-quality-publisher-id",
            "publisher-one",
            "--ground-certificate-root",
            "ground-root.json",
            "--expected-ground-certificate-root-sha256",
            _DIGESTS[1],
            "--audit-config",
            "audit-config.json",
            "--expected-audit-config-sha256",
            _DIGESTS[2],
            "--rl-value-selection-receipt",
            "rl-freeze.json",
            "--expected-selection-sha256",
            _DIGESTS[3],
            "--external-config",
            "external.json",
            "--tuning-registry",
            "tuning.json",
            "--expected-tuning-registry-sha256",
            _DIGESTS[4],
            "--external-tuning-selection",
            "stock-freeze.json",
            "--expected-external-tuning-selection-sha256",
            _DIGESTS[5],
            "--out",
            "plan.json",
        ]
    )

    assert parsed.func is cmd_plan_final_strength_audit
    assert parsed.command == "plan-final-strength-audit"
    assert not hasattr(parsed, "learned_evaluation")
    assert not hasattr(parsed, "external_evaluation")
    assert not hasattr(parsed, "reads_per_block")
    assert not hasattr(parsed, "audit_seed")
    assert not hasattr(parsed, "shard_count")


def test_final_strength_seal_and_shard_parsers_require_six_external_report_pins() -> None:
    common = [
        "--plan",
        "plan.json",
        "--expected-plan-sha256",
        _DIGESTS[8],
        *_test_access_arguments(),
        *_source_arguments(),
    ]
    sealed = build_parser().parse_args(
        [
            "seal-final-strength-audit-execution",
            *common,
            "--out",
            "execution.json",
        ]
    )
    shard = build_parser().parse_args(
        [
            "run-final-strength-audit-shard",
            *common,
            "--execution-manifest",
            "execution.json",
            "--expected-execution-manifest-sha256",
            _DIGESTS[0],
            "--shard-index",
            "17",
            "--out",
            "shard-017",
        ]
    )

    assert sealed.func is cmd_seal_final_strength_audit_execution
    assert shard.func is cmd_run_final_strength_audit_shard
    assert len(sealed.learned_evaluation) == 3
    assert len(sealed.expected_learned_report_sha256) == 3
    assert len(sealed.external_evaluation) == 3
    assert len(sealed.expected_external_report_sha256) == 3
    assert shard.shard_index == 17


def test_final_strength_merge_parser_exposes_only_sealed_inputs() -> None:
    parsed = build_parser().parse_args(
        [
            "merge-final-strength-audit",
            "--plan",
            "plan.json",
            "--expected-plan-sha256",
            "1" * 64,
            "--execution-manifest",
            "execution.json",
            "--expected-execution-manifest-sha256",
            "2" * 64,
            "--shard",
            "shard-000",
            "--expected-shard-receipt-sha256",
            "3" * 64,
            "--out",
            "final-audit",
        ]
    )

    assert parsed.command == "merge-final-strength-audit"
    assert parsed.shard == ["shard-000"]
    assert not hasattr(parsed, "seed")
    assert not hasattr(parsed, "reads")
    assert not hasattr(parsed, "bootstrap_replicates")


def test_final_strength_merge_rejects_unpaired_shard_pin_before_file_access() -> None:
    with pytest.raises(ValueError, match="one external receipt SHA-256"):
        cmd_merge_final_strength_audit(
            SimpleNamespace(
                shard=["shard-a", "shard-b"],
                expected_shard_receipt_sha256=["1" * 64],
                plan="missing-plan.json",
                expected_plan_sha256="2" * 64,
                execution_manifest="missing-execution.json",
                expected_execution_manifest_sha256="3" * 64,
                out="unused",
            )
        )
