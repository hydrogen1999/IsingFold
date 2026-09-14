from __future__ import annotations

import hashlib
import io
import os
import signal
import sys
import time
import zipfile
from contextlib import suppress
from dataclasses import fields
from pathlib import Path
from typing import cast

import embedbench.continuation_runner as runner_module
import embedbench.hard_ood_protocols as protocols_module
import pytest
from embedbench.continuation_runner import (
    CONTINUATION_RUN_RECEIPT_SCHEMA,
    MAX_CONTINUATION_INPUT_BYTES,
    MAX_CONTINUATION_OUTPUT_BYTES,
    MAX_CONTINUATION_STDERR_BYTES,
    MAX_CONTINUATION_STDOUT_BYTES,
    ContinuationInvocationReceipt,
    ContinuationRunReceipt,
    ContinuationRunResult,
    run_verified_continuation,
    verify_continuation_run_receipt,
)
from embedbench.hard_ood_protocols import (
    CANDIDATE_CHAIN_ORDER_RULE,
    CONTINUATION_COMMAND_TEMPLATE,
    CONTINUATION_INPUT_BYTE_LIMIT,
    CONTINUATION_INPUT_SCHEMA,
    CONTINUATION_NODE_BUDGET,
    CONTINUATION_NODE_COUNT_RULE,
    CONTINUATION_OBJECTIVE_ORDER,
    CONTINUATION_OUTPUT_BYTE_LIMIT,
    CONTINUATION_OUTPUT_SCHEMA,
    CONTINUATION_PATH_DEPTH_LIMIT,
    CONTINUATION_PROTOCOL_SCHEMA,
    CONTINUATION_RETAINED_BUNDLE_BYTE_LIMIT,
    CONTINUATION_STDERR_BYTE_LIMIT,
    CONTINUATION_STDOUT_BYTE_LIMIT,
    CONTINUATION_WATCHDOG_SECONDS,
    CONTINUATION_ZIP_MEMBER_LIMIT,
    CONTINUATION_ZIP_UNCOMPRESSED_BYTE_LIMIT,
    VARIABLE_ORDER_RULE,
    verify_continuation_protocol,
)
from embedbench.hard_ood_provenance import SOURCE_BUNDLE_SCHEMA, verify_source_bundle
from embedbench.hard_ood_schema import canonical_bytes

_RELEASE = "embedbench-hard-ood-v1.0.0"
_STATE = {"state_id": "state-0001"}
_INPUT_BYTES = canonical_bytes({"schema": CONTINUATION_INPUT_SCHEMA, "states": [_STATE]})
_INPUT_SHA256 = hashlib.sha256(_INPUT_BYTES).hexdigest()
_STATE_SHA256 = hashlib.sha256(canonical_bytes(_STATE)).hexdigest()


def _output_bytes(certificate: dict[str, object]) -> bytes:
    return canonical_bytes(
        {
            "certificate": certificate,
            "input_sha256": _INPUT_SHA256,
            "schema": CONTINUATION_OUTPUT_SCHEMA,
            "state_sha256": _STATE_SHA256,
        }
    )


def _zipapp(main_source: str) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        info = zipfile.ZipInfo("__main__.py", date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o100644 << 16
        archive.writestr(info, main_source.encode("utf-8"))
    return output.getvalue()


def _source_bundle(
    tmp_path: Path,
    role: str,
    files: dict[str, bytes],
    *,
    executable_paths: frozenset[str] = frozenset(),
):
    root = tmp_path / role
    for relative_path, payload in files.items():
        target = root / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        if relative_path in executable_paths:
            target.chmod(0o755)
    manifest_bytes = canonical_bytes(
        {
            "schema": SOURCE_BUNDLE_SCHEMA,
            "schema_version": 1,
            "release_id": _RELEASE,
            "role": role,
            "files": [
                {
                    "relative_path": relative_path,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "byte_count": len(payload),
                    "executable": relative_path in executable_paths,
                }
                for relative_path, payload in sorted(files.items())
            ],
        }
    )
    bundle = verify_source_bundle(
        root,
        manifest_bytes,
        expected_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        expected_role=role,
    )
    return bundle, root


def _continuation_protocol(
    tmp_path: Path,
    main_source: str,
    *,
    checker_extra_files: dict[str, bytes] | None = None,
    runtime_extra_files: dict[str, bytes] | None = None,
    protocol_overrides: dict[str, object] | None = None,
):
    checker_files = {"checker.pyz": _zipapp(main_source)}
    checker_files.update(checker_extra_files or {})
    checker, checker_root = _source_bundle(
        tmp_path,
        "continuation_checker",
        checker_files,
    )
    runtime_files = {"bin/python3": Path(sys.executable).read_bytes()}
    runtime_files.update(runtime_extra_files or {})
    runtime, runtime_root = _source_bundle(
        tmp_path,
        "portable_runtime",
        runtime_files,
        executable_paths=frozenset({"bin/python3"}),
    )
    document = {
        "schema": CONTINUATION_PROTOCOL_SCHEMA,
        "schema_version": 1,
        "release_id": _RELEASE,
        "protocol_id": "exact-continuation-v1",
        "checker_source_bundle_sha256": checker.manifest_sha256,
        "runtime_bundle_sha256": runtime.manifest_sha256,
        "runtime_python": "bin/python3",
        "checker_zipapp": "checker.pyz",
        "command_template": list(CONTINUATION_COMMAND_TEMPLATE),
        "clean_environment": {},
        "input_schema": CONTINUATION_INPUT_SCHEMA,
        "output_schema": CONTINUATION_OUTPUT_SCHEMA,
        "node_budget": CONTINUATION_NODE_BUDGET,
        "node_count_rule": CONTINUATION_NODE_COUNT_RULE,
        "max_window_free": 28,
        "maximum_chain_length": 6,
        "variable_order_rule": VARIABLE_ORDER_RULE,
        "candidate_chain_order_rule": CANDIDATE_CHAIN_ORDER_RULE,
        "objective_order": list(CONTINUATION_OBJECTIVE_ORDER),
        "safety_watchdog_seconds": CONTINUATION_WATCHDOG_SECONDS,
        "input_byte_limit": CONTINUATION_INPUT_BYTE_LIMIT,
        "output_byte_limit": CONTINUATION_OUTPUT_BYTE_LIMIT,
        "stdout_byte_limit": CONTINUATION_STDOUT_BYTE_LIMIT,
        "stderr_byte_limit": CONTINUATION_STDERR_BYTE_LIMIT,
        "retained_bundle_byte_limit": CONTINUATION_RETAINED_BUNDLE_BYTE_LIMIT,
        "zip_member_limit": CONTINUATION_ZIP_MEMBER_LIMIT,
        "zip_uncompressed_byte_limit": CONTINUATION_ZIP_UNCOMPRESSED_BYTE_LIMIT,
        "path_depth_limit": CONTINUATION_PATH_DEPTH_LIMIT,
    }
    document.update(protocol_overrides or {})
    protocol_bytes = canonical_bytes(document)
    protocol_sha256 = hashlib.sha256(protocol_bytes).hexdigest()
    protocol = verify_continuation_protocol(
        protocol_bytes,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
        checker_source_bundle=checker,
        runtime_bundle=runtime,
    )
    return protocol, protocol_sha256, checker_root, runtime_root


_DETERMINISTIC_CHECKER = f"""
import hashlib
import json
import os
import pathlib
import sys

input_bytes = pathlib.Path(sys.argv[1]).read_bytes()
input_document = json.loads(input_bytes)
state_bytes = json.dumps(
    input_document["states"][0], sort_keys=True, separators=(",", ":")
).encode("utf-8")
output = {{
    "certificate": {{
        "attacker_visible": "EMBEDBENCH_ATTACKER" in os.environ,
        "completion_feasible": False,
    }},
    "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
    "schema": {CONTINUATION_OUTPUT_SCHEMA!r},
    "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
}}
pathlib.Path(sys.argv[2]).write_bytes(
    json.dumps(output, sort_keys=True, separators=(",", ":")).encode("utf-8")
)
sys.stdout.buffer.write(b"checker stdout\\n")
sys.stderr.buffer.write(b"checker stderr\\n")
"""


def test_runner_stages_retained_bytes_uses_frozen_absolute_argv_and_empty_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, protocol_sha256, checker_root, runtime_root = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )
    monkeypatch.setenv("EMBEDBENCH_ATTACKER", "must-not-cross-exec")
    real_popen = runner_module.subprocess.Popen
    spawn_arguments: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def recording_spawn(command, **kwargs):
        spawn_arguments.append((command, kwargs))
        return real_popen(command, **kwargs)

    monkeypatch.setattr(runner_module.subprocess, "Popen", recording_spawn)

    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    expected_output = _output_bytes(
        {
            "attacker_visible": False,
            "completion_feasible": False,
        }
    )
    assert result.scientific_output_bytes == expected_output
    assert result.receipt.status == "accepted"
    assert result.receipt.failure_classification is None
    assert result.receipt.publication_output_match is True
    assert result.receipt.watchdog_seconds == CONTINUATION_WATCHDOG_SECONDS
    assert result.receipt.input_byte_limit == MAX_CONTINUATION_INPUT_BYTES
    assert result.receipt.output_byte_limit == MAX_CONTINUATION_OUTPUT_BYTES
    assert result.receipt.stdout_byte_limit == MAX_CONTINUATION_STDOUT_BYTES
    assert result.receipt.stderr_byte_limit == MAX_CONTINUATION_STDERR_BYTES
    assert result.receipt.retained_bundle_byte_limit == CONTINUATION_RETAINED_BUNDLE_BYTE_LIMIT
    assert result.receipt.zip_member_limit == CONTINUATION_ZIP_MEMBER_LIMIT
    assert result.receipt.zip_uncompressed_byte_limit == CONTINUATION_ZIP_UNCOMPRESSED_BYTE_LIMIT
    assert result.receipt.path_depth_limit == CONTINUATION_PATH_DEPTH_LIMIT
    assert result.receipt.primary.classification == "success"
    assert result.receipt.publication_rerun is not None
    assert result.receipt.publication_rerun.classification == "success"

    for invocation in (result.receipt.primary, result.receipt.publication_rerun):
        assert invocation.command[1] == "-I"
        assert len(invocation.command) == len(CONTINUATION_COMMAND_TEMPLATE)
        assert all(Path(invocation.command[index]).is_absolute() for index in (0, 2, 3, 4))
        assert str(checker_root) not in " ".join(invocation.command)
        assert str(runtime_root) not in " ".join(invocation.command)
        assert invocation.input_sha256 == hashlib.sha256(_INPUT_BYTES).hexdigest()
        assert invocation.output_sha256 == hashlib.sha256(expected_output).hexdigest()
        assert invocation.stdout_sha256 == hashlib.sha256(b"checker stdout\n").hexdigest()
        assert invocation.stderr_sha256 == hashlib.sha256(b"checker stderr\n").hexdigest()
        assert invocation.termination_mode == "exited"
        assert invocation.exit_code == 0
        assert invocation.signal_number is None

    assert result.receipt.primary.command != result.receipt.publication_rerun.command
    assert [arguments[0] for arguments in spawn_arguments] == [
        result.receipt.primary.command,
        result.receipt.publication_rerun.command,
    ]
    assert all(arguments[1]["env"] == {} for arguments in spawn_arguments)
    assert all(arguments[1]["shell"] is False for arguments in spawn_arguments)
    assert all(arguments[1]["start_new_session"] is True for arguments in spawn_arguments)
    assert not Path(result.receipt.primary.command[0]).exists()
    assert not Path(result.receipt.publication_rerun.command[0]).exists()
    assert result.receipt.schema == CONTINUATION_RUN_RECEIPT_SCHEMA
    assert result.receipt.canonical_bytes == canonical_bytes(result.receipt.to_dict())
    assert result.receipt.digest == hashlib.sha256(result.receipt.canonical_bytes).hexdigest()
    assert result.receipt.to_bytes() == result.receipt.canonical_bytes
    assert result.receipt.sha256 == result.receipt.digest


def test_runner_uses_retained_checker_snapshot_not_a_mutated_external_root(
    tmp_path: Path,
) -> None:
    protocol, protocol_sha256, checker_root, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )
    (checker_root / "checker.pyz").write_bytes(_zipapp("raise SystemExit(91)\n"))

    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    assert result.receipt.status == "accepted"
    assert result.receipt.primary.exit_code == 0


def test_runner_consumes_atomic_protocol_snapshot_without_legacy_to_dict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )

    def forbidden_to_dict(*args, **kwargs):
        raise AssertionError("runner used the legacy check-then-read protocol API")

    monkeypatch.setattr(protocols_module.VerifiedContinuationProtocol, "to_dict", forbidden_to_dict)
    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    assert result.receipt.status == "accepted"


@pytest.mark.parametrize(
    ("sha256", "release", "message"),
    [
        ("f" * 64, _RELEASE, "external commitment"),
        (None, "wrong-release", "release"),
    ],
)
def test_runner_rejects_wrong_external_protocol_trust_roots_before_spawning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sha256: str | None,
    release: str,
    message: str,
) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )
    spawn_called = False

    def forbidden_spawn(*args, **kwargs):
        nonlocal spawn_called
        spawn_called = True
        raise AssertionError("untrusted protocol reached subprocess creation")

    monkeypatch.setattr("embedbench.continuation_runner.subprocess.Popen", forbidden_spawn)
    with pytest.raises(ValueError, match=message):
        run_verified_continuation(
            protocol,
            _INPUT_BYTES,
            expected_protocol_sha256=protocol_sha256 if sha256 is None else sha256,
            expected_release_id=release,
        )
    assert spawn_called is False


def test_runner_rejects_a_tampered_verified_protocol_before_spawning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )
    object.__setattr__(protocol, "runtime_python", "bin/attacker")
    spawn_called = False

    def forbidden_spawn(*args, **kwargs):
        nonlocal spawn_called
        spawn_called = True
        raise AssertionError("tampered protocol reached subprocess creation")

    monkeypatch.setattr("embedbench.continuation_runner.subprocess.Popen", forbidden_spawn)
    with pytest.raises(ValueError, match="retained protocol bytes"):
        run_verified_continuation(
            protocol,
            _INPUT_BYTES,
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
        )
    assert spawn_called is False


def test_runner_requires_canonical_input_with_the_registered_schema(tmp_path: Path) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )

    with pytest.raises(ValueError, match="canonical"):
        run_verified_continuation(
            protocol,
            b'{"states":[{"state_id":"state-0001"}],"schema":"embedbench.continuation-batch-input-v1"}',
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
        )
    with pytest.raises(ValueError, match="schema"):
        run_verified_continuation(
            protocol,
            canonical_bytes({"schema": "wrong", "states": [_STATE]}),
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
        )


@pytest.mark.parametrize(
    "document",
    [
        {"schema": CONTINUATION_INPUT_SCHEMA, "states": []},
        {"schema": CONTINUATION_INPUT_SCHEMA, "states": [_STATE, {"state_id": "two"}]},
        {"schema": CONTINUATION_INPUT_SCHEMA, "states": ["not-an-object"]},
        {"schema": CONTINUATION_INPUT_SCHEMA, "states": [{}]},
        {"schema": CONTINUATION_INPUT_SCHEMA, "states": [_STATE], "extra": True},
    ],
)
def test_runner_requires_one_nonempty_state_in_a_closed_input_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    document: dict[str, object],
) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )
    spawn_called = False

    def forbidden_spawn(*args, **kwargs):
        nonlocal spawn_called
        spawn_called = True
        raise AssertionError("invalid input reached subprocess creation")

    monkeypatch.setattr(runner_module.subprocess, "Popen", forbidden_spawn)
    with pytest.raises((TypeError, ValueError), match="one nonempty state|fields differ"):
        run_verified_continuation(
            protocol,
            canonical_bytes(document),
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
        )
    assert spawn_called is False


def test_oversized_input_is_rejected_before_spawning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_limit = len(_INPUT_BYTES) - 1
    monkeypatch.setattr(protocols_module, "CONTINUATION_INPUT_BYTE_LIMIT", input_limit)
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
        protocol_overrides={"input_byte_limit": input_limit},
    )

    def forbidden_spawn(*args, **kwargs):
        raise AssertionError("oversized input reached subprocess creation")

    monkeypatch.setattr(runner_module.subprocess, "Popen", forbidden_spawn)
    with pytest.raises(ValueError, match="input exceeds"):
        run_verified_continuation(
            protocol,
            _INPUT_BYTES,
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
        )


def test_exact_input_output_and_stream_byte_limits_are_inclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_output = _output_bytes(
        {
            "attacker_visible": False,
            "completion_feasible": False,
        }
    )
    limits = {
        "input_byte_limit": len(_INPUT_BYTES),
        "output_byte_limit": len(expected_output),
        "stdout_byte_limit": len(b"checker stdout\n"),
        "stderr_byte_limit": len(b"checker stderr\n"),
    }
    for field_name, value in limits.items():
        monkeypatch.setattr(protocols_module, f"CONTINUATION_{field_name.upper()}", value)
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
        protocol_overrides=limits,
    )

    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    assert result.receipt.status == "accepted"
    assert result.scientific_output_bytes == expected_output


def test_runner_stages_all_retained_runtime_and_checker_files(tmp_path: Path) -> None:
    checker_source = f"""
import hashlib
import json
import pathlib
import sys

checker_root = pathlib.Path(sys.argv[0]).parent
runtime_root = pathlib.Path(sys.executable).parent.parent
input_bytes = pathlib.Path(sys.argv[1]).read_bytes()
state = json.loads(input_bytes)["states"][0]
state_bytes = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
output = {{
    "certificate": {{
        "checker_resource": (checker_root / "data" / "identity.txt").read_text(),
        "runtime_resource": (runtime_root / "share" / "identity.txt").read_text(),
    }},
    "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
    "schema": {CONTINUATION_OUTPUT_SCHEMA!r},
    "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
}}
pathlib.Path(sys.argv[2]).write_bytes(
    json.dumps(output, sort_keys=True, separators=(",", ":")).encode("utf-8")
)
"""
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        checker_source,
        checker_extra_files={"data/identity.txt": b"retained-checker"},
        runtime_extra_files={"share/identity.txt": b"retained-runtime"},
    )

    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    assert result.scientific_output_bytes == _output_bytes(
        {
            "checker_resource": "retained-checker",
            "runtime_resource": "retained-runtime",
        }
    )


def test_runner_capsules_cannot_be_constructed_outside_the_runner(tmp_path: Path) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )
    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    invocation_values = {
        field.name: getattr(result.receipt.primary, field.name)
        for field in fields(ContinuationInvocationReceipt)
        if field.name != "_seal"
    }
    receipt_values = {
        field.name: getattr(result.receipt, field.name)
        for field in fields(ContinuationRunReceipt)
        if field.name != "_seal"
    }
    result_values = {
        field.name: getattr(result, field.name)
        for field in fields(ContinuationRunResult)
        if field.name != "_seal"
    }

    with pytest.raises(TypeError):
        ContinuationInvocationReceipt()
    with pytest.raises(TypeError):
        ContinuationRunReceipt()
    with pytest.raises(TypeError):
        ContinuationRunResult()
    with pytest.raises(TypeError):
        ContinuationInvocationReceipt(**invocation_values)
    with pytest.raises(TypeError):
        ContinuationRunReceipt(**receipt_values)
    with pytest.raises(TypeError):
        ContinuationRunResult(**result_values)


def test_canonical_receipt_verifier_requires_independent_roots_and_output_bytes(
    tmp_path: Path,
) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )
    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    verified = verify_continuation_run_receipt(
        result.receipt.canonical_bytes,
        expected_receipt_sha256=result.receipt.sha256,
        protocol=protocol,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
        expected_checker_source_bundle_sha256=protocol.checker_source_bundle_sha256,
        expected_runtime_bundle_sha256=protocol.runtime_bundle_sha256,
        expected_input_bytes=_INPUT_BYTES,
        expected_scientific_output_bytes=result.scientific_output_bytes,
    )
    assert verified.canonical_bytes == result.receipt.canonical_bytes

    with pytest.raises(ValueError, match="external commitment"):
        verify_continuation_run_receipt(
            result.receipt.canonical_bytes,
            expected_receipt_sha256="f" * 64,
            protocol=protocol,
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
            expected_checker_source_bundle_sha256=protocol.checker_source_bundle_sha256,
            expected_runtime_bundle_sha256=protocol.runtime_bundle_sha256,
            expected_input_bytes=_INPUT_BYTES,
            expected_scientific_output_bytes=result.scientific_output_bytes,
        )

    with pytest.raises(ValueError, match="runtime bundle"):
        verify_continuation_run_receipt(
            result.receipt.canonical_bytes,
            expected_receipt_sha256=result.receipt.sha256,
            protocol=protocol,
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
            expected_checker_source_bundle_sha256=protocol.checker_source_bundle_sha256,
            expected_runtime_bundle_sha256="f" * 64,
            expected_input_bytes=_INPUT_BYTES,
            expected_scientific_output_bytes=result.scientific_output_bytes,
        )

    with pytest.raises(ValueError, match="checker bundle"):
        verify_continuation_run_receipt(
            result.receipt.canonical_bytes,
            expected_receipt_sha256=result.receipt.sha256,
            protocol=protocol,
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
            expected_checker_source_bundle_sha256="f" * 64,
            expected_runtime_bundle_sha256=protocol.runtime_bundle_sha256,
            expected_input_bytes=_INPUT_BYTES,
            expected_scientific_output_bytes=result.scientific_output_bytes,
        )

    with pytest.raises(ValueError, match="scientific output"):
        verify_continuation_run_receipt(
            result.receipt.canonical_bytes,
            expected_receipt_sha256=result.receipt.sha256,
            protocol=protocol,
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
            expected_checker_source_bundle_sha256=protocol.checker_source_bundle_sha256,
            expected_runtime_bundle_sha256=protocol.runtime_bundle_sha256,
            expected_input_bytes=_INPUT_BYTES,
            expected_scientific_output_bytes=_output_bytes({"wrong": True}),
        )


@pytest.mark.parametrize(
    ("primary_count", "publication_count"),
    [(0, 1), (1, 1)],
)
def test_canonical_receipt_verifier_rejects_forged_accepted_output_counts(
    tmp_path: Path,
    primary_count: int,
    publication_count: int,
) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )
    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )
    assert result.scientific_output_bytes is not None
    assert len(result.scientific_output_bytes) == 290

    forged = result.receipt.to_dict()
    primary = dict(cast(dict[str, object], forged["primary"]))
    publication = dict(cast(dict[str, object], forged["publication_rerun"]))
    primary["output_byte_count"] = primary_count
    publication["output_byte_count"] = publication_count
    forged["primary"] = primary
    forged["publication_rerun"] = publication
    forged_bytes = canonical_bytes(forged)

    with pytest.raises(ValueError, match="output byte count"):
        verify_continuation_run_receipt(
            forged_bytes,
            expected_receipt_sha256=hashlib.sha256(forged_bytes).hexdigest(),
            protocol=protocol,
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
            expected_checker_source_bundle_sha256=protocol.checker_source_bundle_sha256,
            expected_runtime_bundle_sha256=protocol.runtime_bundle_sha256,
            expected_input_bytes=_INPUT_BYTES,
            expected_scientific_output_bytes=result.scientific_output_bytes,
        )


def test_canonical_receipt_verifier_rejects_success_output_above_limit_in_failure(
    tmp_path: Path,
) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )
    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    forged = result.receipt.to_dict()
    primary = dict(cast(dict[str, object], forged["primary"]))
    publication = dict(cast(dict[str, object], forged["publication_rerun"]))
    primary["output_byte_count"] = result.receipt.output_byte_limit + 1
    publication["classification"] = "nonzero_exit"
    publication["exit_code"] = 17
    forged["primary"] = primary
    forged["publication_rerun"] = publication
    forged["publication_output_match"] = None
    forged["status"] = "environment_failure"
    forged["failure_classification"] = "publication_nonzero_exit"
    forged["scientific_output_sha256"] = None
    forged_bytes = canonical_bytes(forged)

    with pytest.raises(ValueError, match="primary exact output exceeds"):
        verify_continuation_run_receipt(
            forged_bytes,
            expected_receipt_sha256=hashlib.sha256(forged_bytes).hexdigest(),
            protocol=protocol,
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
            expected_checker_source_bundle_sha256=protocol.checker_source_bundle_sha256,
            expected_runtime_bundle_sha256=protocol.runtime_bundle_sha256,
            expected_input_bytes=_INPUT_BYTES,
            expected_scientific_output_bytes=None,
        )


def test_canonical_receipt_verifier_rejects_invalid_output_above_limit(
    tmp_path: Path,
) -> None:
    checker_source = """
import pathlib
import sys

pathlib.Path(sys.argv[2]).write_bytes(b"{}")
"""
    protocol, protocol_sha256, _, _ = _continuation_protocol(tmp_path, checker_source)
    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )
    assert result.receipt.primary.classification == "invalid_output"

    forged = result.receipt.to_dict()
    primary = dict(cast(dict[str, object], forged["primary"]))
    primary["output_byte_count"] = result.receipt.output_byte_limit + 1
    forged["primary"] = primary
    forged_bytes = canonical_bytes(forged)

    with pytest.raises(ValueError, match="primary exact output exceeds"):
        verify_continuation_run_receipt(
            forged_bytes,
            expected_receipt_sha256=hashlib.sha256(forged_bytes).hexdigest(),
            protocol=protocol,
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
            expected_checker_source_bundle_sha256=protocol.checker_source_bundle_sha256,
            expected_runtime_bundle_sha256=protocol.runtime_bundle_sha256,
            expected_input_bytes=_INPUT_BYTES,
            expected_scientific_output_bytes=None,
        )


def test_canonical_receipt_verifier_accepts_a_bound_failure_without_output(
    tmp_path: Path,
) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        "raise SystemExit(17)\n",
    )
    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    verified = verify_continuation_run_receipt(
        result.receipt.canonical_bytes,
        expected_receipt_sha256=result.receipt.sha256,
        protocol=protocol,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
        expected_checker_source_bundle_sha256=protocol.checker_source_bundle_sha256,
        expected_runtime_bundle_sha256=protocol.runtime_bundle_sha256,
        expected_input_bytes=_INPUT_BYTES,
        expected_scientific_output_bytes=None,
    )

    assert verified.status == "environment_failure"
    assert verified.failure_classification == "primary_nonzero_exit"


def test_canonical_receipt_verifier_rejects_reused_publication_invocation(
    tmp_path: Path,
) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )
    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )
    forged = result.receipt.to_dict()
    forged["publication_rerun"] = forged["primary"]
    forged_bytes = canonical_bytes(forged)

    with pytest.raises(ValueError, match="distinct fresh run directories"):
        verify_continuation_run_receipt(
            forged_bytes,
            expected_receipt_sha256=hashlib.sha256(forged_bytes).hexdigest(),
            protocol=protocol,
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
            expected_checker_source_bundle_sha256=protocol.checker_source_bundle_sha256,
            expected_runtime_bundle_sha256=protocol.runtime_bundle_sha256,
            expected_input_bytes=_INPUT_BYTES,
            expected_scientific_output_bytes=result.scientific_output_bytes,
        )


@pytest.mark.parametrize(
    "cap_field",
    [
        "input_byte_limit",
        "output_byte_limit",
        "stdout_byte_limit",
        "stderr_byte_limit",
        "retained_bundle_byte_limit",
        "zip_member_limit",
        "zip_uncompressed_byte_limit",
        "path_depth_limit",
    ],
)
def test_canonical_receipt_verifier_rejects_cap_rebinding(
    tmp_path: Path,
    cap_field: str,
) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )
    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )
    forged = result.receipt.to_dict()
    forged[cap_field] = getattr(result.receipt, cap_field) + 1
    forged_bytes = canonical_bytes(forged)

    with pytest.raises(ValueError, match=f"wrong {cap_field}"):
        verify_continuation_run_receipt(
            forged_bytes,
            expected_receipt_sha256=hashlib.sha256(forged_bytes).hexdigest(),
            protocol=protocol,
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
            expected_checker_source_bundle_sha256=protocol.checker_source_bundle_sha256,
            expected_runtime_bundle_sha256=protocol.runtime_bundle_sha256,
            expected_input_bytes=_INPUT_BYTES,
            expected_scientific_output_bytes=result.scientific_output_bytes,
        )


def test_canonical_receipt_verifier_rejects_nonexceeding_output_limit_count(
    tmp_path: Path,
) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )
    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )
    forged = result.receipt.to_dict()
    primary = dict(cast(dict[str, object], forged["primary"]))
    primary.update(
        {
            "classification": "output_limit_exceeded",
            "output_byte_count": result.receipt.output_byte_limit,
            "output_sha256": None,
        }
    )
    forged.update(
        {
            "failure_classification": "primary_output_limit_exceeded",
            "primary": primary,
            "publication_output_match": None,
            "publication_rerun": None,
            "scientific_output_sha256": None,
            "status": "environment_failure",
        }
    )
    forged_bytes = canonical_bytes(forged)

    with pytest.raises(ValueError, match="output limit classification requires"):
        verify_continuation_run_receipt(
            forged_bytes,
            expected_receipt_sha256=hashlib.sha256(forged_bytes).hexdigest(),
            protocol=protocol,
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
            expected_checker_source_bundle_sha256=protocol.checker_source_bundle_sha256,
            expected_runtime_bundle_sha256=protocol.runtime_bundle_sha256,
            expected_input_bytes=_INPUT_BYTES,
            expected_scientific_output_bytes=None,
        )


@pytest.mark.parametrize("stream_name", ["stdout", "stderr"])
def test_canonical_receipt_verifier_rejects_short_overflow_capture(
    tmp_path: Path,
    stream_name: str,
) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )
    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )
    forged = result.receipt.to_dict()
    primary = dict(cast(dict[str, object], forged["primary"]))
    stream_limit = getattr(result.receipt, f"{stream_name}_byte_limit")
    primary.update(
        {
            "classification": f"{stream_name}_limit_exceeded",
            "exit_code": None,
            f"{stream_name}_byte_count": stream_limit - 1,
            f"{stream_name}_overflow": True,
            "termination_mode": "capture_limit_exceeded",
        }
    )
    forged.update(
        {
            "failure_classification": f"primary_{stream_name}_limit_exceeded",
            "primary": primary,
            "publication_output_match": None,
            "publication_rerun": None,
            "scientific_output_sha256": None,
            "status": "environment_failure",
        }
    )
    forged_bytes = canonical_bytes(forged)

    with pytest.raises(ValueError, match=f"{stream_name} overflow requires"):
        verify_continuation_run_receipt(
            forged_bytes,
            expected_receipt_sha256=hashlib.sha256(forged_bytes).hexdigest(),
            protocol=protocol,
            expected_protocol_sha256=protocol_sha256,
            expected_release_id=_RELEASE,
            expected_checker_source_bundle_sha256=protocol.checker_source_bundle_sha256,
            expected_runtime_bundle_sha256=protocol.runtime_bundle_sha256,
            expected_input_bytes=_INPUT_BYTES,
            expected_scientific_output_bytes=None,
        )


def test_primary_staging_failure_is_a_canonical_environment_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )

    def fail_staging(*args, **kwargs):
        raise OSError("forced staging failure")

    def forbidden_spawn(*args, **kwargs):
        raise AssertionError("staging failure reached subprocess creation")

    monkeypatch.setattr(runner_module, "_stage_bundle", fail_staging)
    monkeypatch.setattr(runner_module.subprocess, "Popen", forbidden_spawn)
    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    assert result.scientific_output_bytes is None
    assert result.receipt.status == "environment_failure"
    assert result.receipt.failure_classification == "primary_staging_error"
    assert result.receipt.primary.classification == "staging_error"
    assert result.receipt.primary.termination_mode == "staging_error"
    assert result.receipt.primary.output_present is False
    verified = verify_continuation_run_receipt(
        result.receipt.canonical_bytes,
        expected_receipt_sha256=result.receipt.sha256,
        protocol=protocol,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
        expected_checker_source_bundle_sha256=protocol.checker_source_bundle_sha256,
        expected_runtime_bundle_sha256=protocol.runtime_bundle_sha256,
        expected_input_bytes=_INPUT_BYTES,
        expected_scientific_output_bytes=None,
    )
    assert verified.failure_classification == "primary_staging_error"


def test_publication_staging_failure_is_receipted_after_a_successful_primary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )
    real_stage_bundle = runner_module._stage_bundle
    stage_calls = 0

    def fail_publication_staging(*args, **kwargs):
        nonlocal stage_calls
        stage_calls += 1
        if stage_calls == 3:
            raise OSError("forced publication staging failure")
        return real_stage_bundle(*args, **kwargs)

    monkeypatch.setattr(runner_module, "_stage_bundle", fail_publication_staging)
    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    assert result.scientific_output_bytes is None
    assert result.receipt.primary.classification == "success"
    assert result.receipt.publication_rerun is not None
    assert result.receipt.publication_rerun.classification == "staging_error"
    assert result.receipt.failure_classification == "publication_staging_error"


def test_nonzero_exit_is_an_environment_failure_never_infeasibility(tmp_path: Path) -> None:
    emitted_output = canonical_bytes(
        {"completion_feasible": False, "schema": CONTINUATION_OUTPUT_SCHEMA}
    )
    checker_source = f"""
import pathlib
import sys

pathlib.Path(sys.argv[2]).write_bytes({emitted_output!r})
raise SystemExit(23)
"""
    protocol, protocol_sha256, _, _ = _continuation_protocol(tmp_path, checker_source)

    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    assert result.scientific_output_bytes is None
    assert result.receipt.status == "environment_failure"
    assert result.receipt.failure_classification == "primary_nonzero_exit"
    assert result.receipt.primary.classification == "nonzero_exit"
    assert result.receipt.primary.termination_mode == "exited"
    assert result.receipt.primary.exit_code == 23
    assert result.receipt.primary.signal_number is None
    assert result.receipt.primary.output_sha256 == hashlib.sha256(emitted_output).hexdigest()
    assert result.receipt.publication_rerun is None
    assert b"infeasible" not in result.receipt.canonical_bytes


def test_process_failure_remains_receiptable_when_its_output_is_unsafe(
    tmp_path: Path,
) -> None:
    checker_source = """
import os
import sys

os.symlink('/dev/null', sys.argv[2])
raise SystemExit(23)
"""
    protocol, protocol_sha256, _, _ = _continuation_protocol(tmp_path, checker_source)

    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    assert result.receipt.failure_classification == "primary_nonzero_exit"
    assert result.receipt.primary.classification == "nonzero_exit"
    assert result.receipt.primary.output_present is True
    assert result.receipt.primary.output_sha256 is None
    assert result.receipt.primary.output_byte_count is None


def test_signal_is_an_environment_failure_never_infeasibility(tmp_path: Path) -> None:
    checker_source = """
import os
import signal

os.kill(os.getpid(), signal.SIGTERM)
"""
    protocol, protocol_sha256, _, _ = _continuation_protocol(tmp_path, checker_source)

    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    assert result.scientific_output_bytes is None
    assert result.receipt.status == "environment_failure"
    assert result.receipt.failure_classification == "primary_signal"
    assert result.receipt.primary.classification == "signal"
    assert result.receipt.primary.termination_mode == "signal"
    assert result.receipt.primary.exit_code is None
    assert result.receipt.primary.signal_number == signal.SIGTERM
    assert result.receipt.publication_rerun is None


def test_spawn_error_is_receipted_without_exposing_scientific_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )

    def fail_spawn(*args, **kwargs):
        raise PermissionError("forced spawn failure")

    monkeypatch.setattr(runner_module.subprocess, "Popen", fail_spawn)
    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    assert result.scientific_output_bytes is None
    assert result.receipt.status == "environment_failure"
    assert result.receipt.failure_classification == "primary_spawn_error"
    assert result.receipt.primary.classification == "spawn_error"
    assert result.receipt.primary.termination_mode == "spawn_error"
    assert result.receipt.primary.exit_code is None
    assert result.receipt.primary.signal_number is None
    assert result.receipt.primary.stdout_sha256 == hashlib.sha256(b"").hexdigest()
    assert result.receipt.primary.stderr_sha256 == hashlib.sha256(b"").hexdigest()


def test_watchdog_timeout_is_receipted_without_exposing_scientific_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        _DETERMINISTIC_CHECKER,
    )

    def forced_timeout(
        command,
        *,
        working_directory,
        watchdog_seconds,
        stdout_byte_limit,
        stderr_byte_limit,
    ):
        assert watchdog_seconds == CONTINUATION_WATCHDOG_SECONDS
        assert stdout_byte_limit == MAX_CONTINUATION_STDOUT_BYTES
        assert stderr_byte_limit == MAX_CONTINUATION_STDERR_BYTES
        return runner_module._ProcessCapture("watchdog_timeout", None, b"partial", b"")

    monkeypatch.setattr(runner_module, "_run_process", forced_timeout)
    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    assert result.scientific_output_bytes is None
    assert result.receipt.status == "environment_failure"
    assert result.receipt.failure_classification == "primary_watchdog_timeout"
    assert result.receipt.primary.classification == "watchdog_timeout"
    assert result.receipt.primary.termination_mode == "watchdog_timeout"
    assert result.receipt.primary.exit_code is None
    assert result.receipt.primary.signal_number is None
    assert result.receipt.primary.stdout_sha256 == hashlib.sha256(b"partial").hexdigest()


def test_process_watchdog_forcibly_terminates_a_real_subprocess(tmp_path: Path) -> None:
    capture = runner_module._run_process(
        (
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ),
        working_directory=tmp_path,
        watchdog_seconds=0.05,
    )

    assert capture.termination_mode == "watchdog_timeout"
    assert capture.returncode is None
    assert capture.stdout == b""


def test_runner_does_not_block_if_an_escaped_child_holds_capture_pipes(
    tmp_path: Path,
) -> None:
    escaped_pid_path = tmp_path / "escaped.pid"
    escaped_source = (
        f"import os,time;open({str(escaped_pid_path)!r},'w').write(str(os.getpid()));time.sleep(30)"
    )
    source = (
        "import subprocess,sys;"
        "subprocess.Popen([sys.executable,'-c',"
        f"{escaped_source!r}],start_new_session=True)"
    )
    started = time.monotonic()

    capture = runner_module._run_process(
        (sys.executable, "-c", source),
        working_directory=tmp_path,
        watchdog_seconds=5,
    )
    elapsed = time.monotonic() - started
    for _ in range(100):
        if escaped_pid_path.exists():
            break
        time.sleep(0.01)
    escaped_pid = int(escaped_pid_path.read_text(encoding="utf-8"))
    try:
        assert elapsed < 3.0
        assert capture.termination_mode == "runner_error"
        os.kill(escaped_pid, 0)
    finally:
        with suppress(ProcessLookupError):
            os.kill(escaped_pid, signal.SIGKILL)


def test_publication_rerun_rejects_nondeterministic_scientific_output(
    tmp_path: Path,
) -> None:
    checker_source = f"""
import hashlib
import json
import os
import pathlib
import sys

input_bytes = pathlib.Path(sys.argv[1]).read_bytes()
state = json.loads(input_bytes)["states"][0]
state_bytes = json.dumps(state, sort_keys=True, separators=(",", ":")).encode()
output = {{
    "certificate": {{"run_directory": os.getcwd()}},
    "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
    "schema": {CONTINUATION_OUTPUT_SCHEMA!r},
    "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
}}
pathlib.Path(sys.argv[2]).write_bytes(
    json.dumps(output, sort_keys=True, separators=(",", ":")).encode("utf-8")
)
"""
    protocol, protocol_sha256, _, _ = _continuation_protocol(tmp_path, checker_source)

    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    assert result.scientific_output_bytes is None
    assert result.receipt.status == "environment_failure"
    assert result.receipt.failure_classification == "publication_output_mismatch"
    assert result.receipt.publication_output_match is False
    assert result.receipt.primary.classification == "success"
    assert result.receipt.publication_rerun is not None
    assert result.receipt.publication_rerun.classification == "success"
    assert result.receipt.primary.output_sha256 != result.receipt.publication_rerun.output_sha256
    assert result.receipt.primary.command != result.receipt.publication_rerun.command


@pytest.mark.parametrize(
    ("checker_source", "classification"),
    [
        ("raise SystemExit(0)\n", "missing_output"),
        (
            "import pathlib,sys\npathlib.Path(sys.argv[2]).write_bytes(b'not-json')\n",
            "invalid_output",
        ),
    ],
)
def test_zero_exit_without_a_valid_canonical_certificate_fails_closed(
    tmp_path: Path,
    checker_source: str,
    classification: str,
) -> None:
    protocol, protocol_sha256, _, _ = _continuation_protocol(tmp_path, checker_source)

    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    assert result.scientific_output_bytes is None
    assert result.receipt.status == "environment_failure"
    assert result.receipt.primary.classification == classification
    assert result.receipt.failure_classification == f"primary_{classification}"


def test_output_symlink_is_never_followed_or_accepted(tmp_path: Path) -> None:
    checker_source = """
import os
import sys

os.symlink('/dev/null', sys.argv[2])
"""
    protocol, protocol_sha256, _, _ = _continuation_protocol(tmp_path, checker_source)

    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    assert result.scientific_output_bytes is None
    assert result.receipt.status == "environment_failure"
    assert result.receipt.failure_classification == "primary_unsafe_output"
    assert result.receipt.primary.output_present is True
    assert result.receipt.primary.output_sha256 is None


@pytest.mark.parametrize(
    "output_bytes",
    [
        canonical_bytes({"schema": CONTINUATION_OUTPUT_SCHEMA}),
        canonical_bytes(
            {
                "certificate": {"completion_feasible": False},
                "input_sha256": "f" * 64,
                "schema": CONTINUATION_OUTPUT_SCHEMA,
                "state_sha256": _STATE_SHA256,
            }
        ),
        canonical_bytes(
            {
                "certificate": {"completion_feasible": False},
                "input_sha256": _INPUT_SHA256,
                "schema": CONTINUATION_OUTPUT_SCHEMA,
                "state_sha256": "f" * 64,
            }
        ),
        canonical_bytes(
            {
                "certificate": {},
                "input_sha256": _INPUT_SHA256,
                "schema": CONTINUATION_OUTPUT_SCHEMA,
                "state_sha256": _STATE_SHA256,
            }
        ),
        canonical_bytes(
            {
                "certificate": {"completion_feasible": False},
                "extra": True,
                "input_sha256": _INPUT_SHA256,
                "schema": CONTINUATION_OUTPUT_SCHEMA,
                "state_sha256": _STATE_SHA256,
            }
        ),
    ],
)
def test_schema_only_unbound_or_open_output_envelopes_are_rejected(
    tmp_path: Path,
    output_bytes: bytes,
) -> None:
    checker_source = f"""
import pathlib
import sys

pathlib.Path(sys.argv[2]).write_bytes({output_bytes!r})
"""
    protocol, protocol_sha256, _, _ = _continuation_protocol(tmp_path, checker_source)

    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    assert result.scientific_output_bytes is None
    assert result.receipt.failure_classification == "primary_invalid_output"
    assert result.receipt.primary.classification == "invalid_output"


def test_output_fifo_is_rejected_without_blocking_after_the_watchdog(tmp_path: Path) -> None:
    checker_source = """
import os
import sys

os.mkfifo(sys.argv[2])
"""
    protocol, protocol_sha256, _, _ = _continuation_protocol(tmp_path, checker_source)
    started = time.monotonic()

    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    assert time.monotonic() - started < 2.0
    assert result.scientific_output_bytes is None
    assert result.receipt.failure_classification == "primary_unsafe_output"
    assert result.receipt.primary.classification == "unsafe_output"


def test_renamed_io_parent_and_ancestor_symlink_cannot_redirect_output(
    tmp_path: Path,
) -> None:
    external_root = (tmp_path / "external-output").resolve()
    external_root.mkdir()
    external_output = external_root / "output.json"
    output_bytes = _output_bytes({"completion_feasible": False})
    checker_source = f"""
import pathlib
import sys

root = pathlib.Path.cwd()
(root / "io").rename(root / "io-original")
(root / "io").symlink_to(pathlib.Path({str(external_root)!r}), target_is_directory=True)
pathlib.Path(sys.argv[2]).write_bytes({output_bytes!r})
"""
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path / "protocol",
        checker_source,
    )

    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    assert external_output.read_bytes() == output_bytes
    assert result.scientific_output_bytes is None
    assert result.receipt.failure_classification == "primary_unsafe_io_directory"
    assert result.receipt.primary.classification == "unsafe_io_directory"


def test_sparse_oversized_output_is_rejected_without_reading_its_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_limit = 1024
    monkeypatch.setattr(protocols_module, "CONTINUATION_OUTPUT_BYTE_LIMIT", output_limit)
    checker_source = f"""
import os
import sys

with open(sys.argv[2], "wb") as output:
    output.truncate({output_limit + 1})
"""
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        checker_source,
        protocol_overrides={"output_byte_limit": output_limit},
    )

    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    assert result.scientific_output_bytes is None
    assert result.receipt.failure_classification == "primary_output_limit_exceeded"
    assert result.receipt.primary.classification == "output_limit_exceeded"
    assert result.receipt.primary.output_present is True
    assert result.receipt.primary.output_sha256 is None
    assert result.receipt.primary.output_byte_count == output_limit + 1


@pytest.mark.parametrize(
    ("stream_name", "classification"),
    [
        ("stdout", "stdout_limit_exceeded"),
        ("stderr", "stderr_limit_exceeded"),
    ],
)
def test_checker_stream_overflow_is_bounded_killed_and_receipted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stream_name: str,
    classification: str,
) -> None:
    stream_limit = 1024
    protocol_field = f"{stream_name}_byte_limit"
    monkeypatch.setattr(
        protocols_module,
        f"CONTINUATION_{stream_name.upper()}_BYTE_LIMIT",
        stream_limit,
    )
    output_bytes = _output_bytes({"completion_feasible": False})
    checker_source = f"""
import pathlib
import sys

pathlib.Path(sys.argv[2]).write_bytes({output_bytes!r})
sys.{stream_name}.buffer.write(b"x" * {stream_limit + 4096})
sys.{stream_name}.buffer.flush()
"""
    protocol, protocol_sha256, _, _ = _continuation_protocol(
        tmp_path,
        checker_source,
        protocol_overrides={protocol_field: stream_limit},
    )

    result = run_verified_continuation(
        protocol,
        _INPUT_BYTES,
        expected_protocol_sha256=protocol_sha256,
        expected_release_id=_RELEASE,
    )

    invocation = result.receipt.primary
    assert result.scientific_output_bytes is None
    assert result.receipt.failure_classification == f"primary_{classification}"
    assert invocation.classification == classification
    assert invocation.termination_mode == "capture_limit_exceeded"
    assert getattr(invocation, f"{stream_name}_overflow") is True
    assert getattr(invocation, f"{stream_name}_byte_count") == stream_limit
