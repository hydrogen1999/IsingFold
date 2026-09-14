"""Fail-closed fresh-process execution for verified continuation protocols."""

from __future__ import annotations

import hashlib
import os
import selectors
import signal
import stat
import subprocess
import tempfile
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Self, cast

from embedbench.hard_ood_protocols import (
    CHECKER_ZIPAPP_PATH,
    CONTINUATION_COMMAND_TEMPLATE,
    CONTINUATION_INPUT_BYTE_LIMIT,
    CONTINUATION_OUTPUT_BYTE_LIMIT,
    CONTINUATION_STDERR_BYTE_LIMIT,
    CONTINUATION_STDOUT_BYTE_LIMIT,
    RUNTIME_PYTHON_PATH,
    ContinuationProtocolSnapshot,
    VerifiedContinuationProtocol,
    snapshot_continuation_protocol,
)
from embedbench.hard_ood_provenance import VerifiedSourceFile, parse_canonical_json_bytes
from embedbench.hard_ood_schema import (
    canonical_bytes as _canonical_bytes,
)
from embedbench.hard_ood_schema import (
    require_exact_keys,
)

CONTINUATION_RUN_RECEIPT_SCHEMA = "embedbench.continuation-run-receipt-v1"
CONTINUATION_RUN_RECEIPT_SCHEMA_VERSION = 1
EMPTY_EXEC_ENVIRONMENT = "empty-exec-environment-v1"
MAX_CONTINUATION_INPUT_BYTES = CONTINUATION_INPUT_BYTE_LIMIT
MAX_CONTINUATION_OUTPUT_BYTES = CONTINUATION_OUTPUT_BYTE_LIMIT
MAX_CONTINUATION_STDOUT_BYTES = CONTINUATION_STDOUT_BYTE_LIMIT
MAX_CONTINUATION_STDERR_BYTES = CONTINUATION_STDERR_BYTE_LIMIT

_SHA256_ALPHABET = frozenset("0123456789abcdef")
_TERMINATION_MODES = frozenset(
    {
        "exited",
        "signal",
        "watchdog_timeout",
        "capture_limit_exceeded",
        "spawn_error",
        "runner_error",
        "staging_error",
    }
)
_CLASSIFICATIONS = frozenset(
    {
        "success",
        "nonzero_exit",
        "signal",
        "watchdog_timeout",
        "spawn_error",
        "runner_error",
        "staging_error",
        "stdout_limit_exceeded",
        "stderr_limit_exceeded",
        "missing_output",
        "unsafe_io_directory",
        "unsafe_output",
        "output_limit_exceeded",
        "invalid_output",
    }
)
_CAPTURE_DRAIN_GRACE_SECONDS = 1.0
_CAPTURE_CHUNK_BYTES = 64 * 1024
_PROCESS_POLL_SECONDS = 0.02
_INPUT_FIELDS = frozenset({"schema", "states"})
_OUTPUT_FIELDS = frozenset({"schema", "input_sha256", "state_sha256", "certificate"})
_INVOCATION_RECEIPT_FIELDS = frozenset(
    {
        "classification",
        "command",
        "exit_code",
        "input_byte_count",
        "input_sha256",
        "output_byte_count",
        "output_present",
        "output_sha256",
        "signal_number",
        "stderr_byte_count",
        "stderr_overflow",
        "stderr_sha256",
        "stdout_byte_count",
        "stdout_overflow",
        "stdout_sha256",
        "termination_mode",
    }
)
_RUN_RECEIPT_FIELDS = frozenset(
    {
        "checker_source_bundle_sha256",
        "checker_zipapp_sha256",
        "environment",
        "failure_classification",
        "input_byte_count",
        "input_byte_limit",
        "input_schema",
        "input_sha256",
        "output_byte_limit",
        "output_schema",
        "primary",
        "protocol_sha256",
        "publication_output_match",
        "publication_rerun",
        "release_id",
        "runtime_bundle_sha256",
        "runtime_python_sha256",
        "schema",
        "schema_version",
        "scientific_output_sha256",
        "status",
        "stderr_byte_limit",
        "stdout_byte_limit",
        "retained_bundle_byte_limit",
        "zip_member_limit",
        "zip_uncompressed_byte_limit",
        "path_depth_limit",
        "watchdog_seconds",
    }
)
_INVOCATION_RECEIPT_SEAL = object()
_RUN_RECEIPT_SEAL = object()
_RUN_RESULT_SEAL = object()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_ALPHABET for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _require_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_positive_timeout(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"{name} must be a positive number")
    return float(value)


def _require_text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def _optional_text(value: object, name: str) -> str | None:
    return None if value is None else _require_text(value, name)


def _optional_nonnegative_int(value: object, name: str) -> int | None:
    return None if value is None else _require_nonnegative_int(value, name)


def _optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer or null")
    return value


def _require_bool(value: object, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _optional_bool(value: object, name: str) -> bool | None:
    return None if value is None else _require_bool(value, name)


def _optional_sha256(value: object, name: str) -> str | None:
    return None if value is None else _require_sha256(value, name)


def _parse_input_binding(
    input_bytes: bytes,
    *,
    input_schema: str,
    input_byte_limit: int,
) -> tuple[str, str]:
    if len(input_bytes) > input_byte_limit:
        raise ValueError("continuation input exceeds the registered runner byte limit")
    document = require_exact_keys(
        parse_canonical_json_bytes(input_bytes, name="continuation input"),
        _INPUT_FIELDS,
        "continuation input",
    )
    if type(document["schema"]) is not str or document["schema"] != input_schema:
        raise ValueError("continuation input schema does not match the verified protocol")
    states = document["states"]
    if type(states) is not list or len(states) != 1 or type(states[0]) is not dict or not states[0]:
        raise ValueError("continuation input requires exactly one nonempty state object")
    return _sha256(input_bytes), _sha256(_canonical_bytes(states[0]))


def _validate_output_envelope(
    output_bytes: bytes,
    *,
    output_schema: str,
    input_sha256: str,
    state_sha256: str,
) -> None:
    document = require_exact_keys(
        parse_canonical_json_bytes(output_bytes, name="continuation output"),
        _OUTPUT_FIELDS,
        "continuation output",
    )
    if type(document["schema"]) is not str or document["schema"] != output_schema:
        raise ValueError("continuation output schema does not match the verified protocol")
    if type(document["input_sha256"]) is not str or document["input_sha256"] != input_sha256:
        raise ValueError("continuation output does not bind the exact input bytes")
    if type(document["state_sha256"]) is not str or document["state_sha256"] != state_sha256:
        raise ValueError("continuation output does not bind the exact state bytes")
    certificate = document["certificate"]
    if type(certificate) is not dict or not certificate:
        raise ValueError("continuation output requires one nonempty certificate object")


def _invocation_run_root(command: tuple[str, ...]) -> Path:
    if (
        type(command) is not tuple
        or len(command) != len(CONTINUATION_COMMAND_TEMPLATE)
        or any(type(argument) is not str or not argument for argument in command)
        or command[1] != "-I"
    ):
        raise ValueError("command must be the absolute frozen continuation argv")
    paths = tuple(Path(command[index]) for index in (0, 2, 3, 4))
    if any(
        not path.is_absolute() or any(part in {".", ".."} for part in path.parts) for path in paths
    ):
        raise ValueError("command must use normalized absolute continuation paths")
    input_path = Path(command[3])
    run_root = input_path.parent.parent
    expected = (
        run_root / "runtime" / RUNTIME_PYTHON_PATH,
        run_root / "checker" / CHECKER_ZIPAPP_PATH,
        run_root / "io" / "input.json",
        run_root / "io" / "output.json",
    )
    if paths != expected:
        raise ValueError("command paths do not share one exact staged run directory")
    return run_root


@dataclass(frozen=True, slots=True, init=False)
class ContinuationInvocationReceipt:
    """Canonical provenance for one fresh checker subprocess."""

    command: tuple[str, ...]
    termination_mode: str
    classification: str
    exit_code: int | None
    signal_number: int | None
    input_sha256: str
    input_byte_count: int
    output_present: bool
    output_sha256: str | None
    output_byte_count: int | None
    stdout_sha256: str
    stdout_byte_count: int
    stdout_overflow: bool
    stderr_sha256: str
    stderr_byte_count: int
    stderr_overflow: bool
    _seal: object

    def __new__(cls) -> Self:
        raise TypeError("invocation receipts can only be constructed by the runner")

    def __post_init__(self) -> None:
        if self._seal is not _INVOCATION_RECEIPT_SEAL:
            raise TypeError("invocation receipts can only be constructed by the runner")
        _invocation_run_root(self.command)
        if (
            type(self.termination_mode) is not str
            or self.termination_mode not in _TERMINATION_MODES
        ):
            raise ValueError("termination_mode is unregistered")
        if type(self.classification) is not str or self.classification not in _CLASSIFICATIONS:
            raise ValueError("classification is unregistered")
        if self.exit_code is not None:
            _require_nonnegative_int(self.exit_code, "exit_code")
        if self.signal_number is not None and (
            type(self.signal_number) is not int or self.signal_number <= 0
        ):
            raise ValueError("signal_number must be a positive integer or null")
        _require_sha256(self.input_sha256, "input_sha256")
        _require_nonnegative_int(self.input_byte_count, "input_byte_count")
        if type(self.output_present) is not bool:
            raise TypeError("output_present must be a boolean")
        if self.output_sha256 is not None:
            _require_sha256(self.output_sha256, "output_sha256")
        if self.output_byte_count is not None:
            _require_nonnegative_int(self.output_byte_count, "output_byte_count")
        _require_sha256(self.stdout_sha256, "stdout_sha256")
        _require_nonnegative_int(self.stdout_byte_count, "stdout_byte_count")
        _require_bool(self.stdout_overflow, "stdout_overflow")
        _require_sha256(self.stderr_sha256, "stderr_sha256")
        _require_nonnegative_int(self.stderr_byte_count, "stderr_byte_count")
        _require_bool(self.stderr_overflow, "stderr_overflow")

        if self.termination_mode == "exited":
            if self.exit_code is None or self.signal_number is not None:
                raise ValueError("an exited process requires only an exit code")
        elif self.termination_mode == "signal":
            if self.exit_code is not None or self.signal_number is None:
                raise ValueError("a signalled process requires only a signal number")
        elif self.exit_code is not None or self.signal_number is not None:
            raise ValueError("non-exit termination cannot carry exit or signal status")

        expected_termination = {
            "success": "exited",
            "nonzero_exit": "exited",
            "signal": "signal",
            "watchdog_timeout": "watchdog_timeout",
            "stdout_limit_exceeded": "capture_limit_exceeded",
            "stderr_limit_exceeded": "capture_limit_exceeded",
            "spawn_error": "spawn_error",
            "runner_error": "runner_error",
            "staging_error": "staging_error",
            "missing_output": "exited",
            "unsafe_io_directory": "exited",
            "unsafe_output": "exited",
            "output_limit_exceeded": "exited",
            "invalid_output": "exited",
        }[self.classification]
        if self.termination_mode != expected_termination:
            raise ValueError("classification and termination_mode disagree")
        if self.classification == "success" and (
            self.exit_code != 0
            or not self.output_present
            or self.output_sha256 is None
            or self.output_byte_count is None
        ):
            raise ValueError("success requires exit code zero and an exact output")
        if (
            self.classification
            in {
                "missing_output",
                "unsafe_io_directory",
                "unsafe_output",
                "output_limit_exceeded",
                "invalid_output",
            }
            and self.exit_code != 0
        ):
            raise ValueError("output failure requires exit code zero")
        if self.classification == "nonzero_exit" and (
            self.exit_code is None or self.exit_code == 0
        ):
            raise ValueError("nonzero_exit requires a nonzero exit code")
        if self.classification == "staging_error" and self.output_present:
            raise ValueError("staging_error cannot carry checker output")
        if self.classification in {"spawn_error", "staging_error"} and (
            self.output_present
            or self.stdout_byte_count != 0
            or self.stderr_byte_count != 0
            or self.stdout_sha256 != _sha256(b"")
            or self.stderr_sha256 != _sha256(b"")
            or self.stdout_overflow
            or self.stderr_overflow
        ):
            raise ValueError("a process-free failure requires empty output and stream captures")
        if self.classification == "missing_output" and self.output_present:
            raise ValueError("missing_output requires an absent output")
        if self.classification == "unsafe_output" and (
            not self.output_present
            or self.output_sha256 is not None
            or self.output_byte_count is not None
        ):
            raise ValueError("unsafe_output requires an unhashable present output")
        if self.classification == "output_limit_exceeded" and (
            not self.output_present
            or self.output_sha256 is not None
            or self.output_byte_count is None
        ):
            raise ValueError(
                "output_limit_exceeded requires a present output and its observed size"
            )
        if self.classification == "invalid_output" and (
            not self.output_present or self.output_sha256 is None or self.output_byte_count is None
        ):
            raise ValueError("invalid_output requires exact rejected output bytes")
        if self.output_present:
            if self.output_sha256 is not None and self.output_byte_count is None:
                raise ValueError("a hashed output requires an exact byte count")
            if self.classification in {"success", "invalid_output"} and (
                self.output_sha256 is None or self.output_byte_count is None
            ):
                raise ValueError("a parsed output requires an exact hash and byte count")
        elif self.output_sha256 is not None or self.output_byte_count is not None:
            raise ValueError("an absent output cannot carry a hash or byte count")
        if self.stdout_overflow or self.stderr_overflow:
            expected_capture_classification = (
                "stdout_limit_exceeded" if self.stdout_overflow else "stderr_limit_exceeded"
            )
            if self.classification != expected_capture_classification:
                raise ValueError(
                    "stream overflow flags disagree with the invocation classification"
                )
        elif self.classification in {"stdout_limit_exceeded", "stderr_limit_exceeded"}:
            raise ValueError("stream limit classification requires its overflow flag")

    def to_dict(self) -> dict[str, object]:
        return {
            "classification": self.classification,
            "command": list(self.command),
            "exit_code": self.exit_code,
            "input_byte_count": self.input_byte_count,
            "input_sha256": self.input_sha256,
            "output_byte_count": self.output_byte_count,
            "output_present": self.output_present,
            "output_sha256": self.output_sha256,
            "signal_number": self.signal_number,
            "stderr_byte_count": self.stderr_byte_count,
            "stderr_overflow": self.stderr_overflow,
            "stderr_sha256": self.stderr_sha256,
            "stdout_byte_count": self.stdout_byte_count,
            "stdout_overflow": self.stdout_overflow,
            "stdout_sha256": self.stdout_sha256,
            "termination_mode": self.termination_mode,
        }


@dataclass(frozen=True, slots=True, init=False)
class ContinuationRunReceipt:
    """Typed, canonical receipt for a primary run and publication replay."""

    release_id: str
    protocol_sha256: str
    checker_source_bundle_sha256: str
    runtime_bundle_sha256: str
    checker_zipapp_sha256: str
    runtime_python_sha256: str
    input_schema: str
    output_schema: str
    watchdog_seconds: int
    input_byte_limit: int
    output_byte_limit: int
    stdout_byte_limit: int
    stderr_byte_limit: int
    retained_bundle_byte_limit: int
    zip_member_limit: int
    zip_uncompressed_byte_limit: int
    path_depth_limit: int
    input_sha256: str
    input_byte_count: int
    primary: ContinuationInvocationReceipt
    publication_rerun: ContinuationInvocationReceipt | None
    publication_output_match: bool | None
    status: str
    failure_classification: str | None
    scientific_output_sha256: str | None
    _seal: object

    def __new__(cls) -> Self:
        raise TypeError("run receipts can only be constructed by the runner or verifier")

    @property
    def schema(self) -> str:
        return CONTINUATION_RUN_RECEIPT_SCHEMA

    @property
    def schema_version(self) -> int:
        return CONTINUATION_RUN_RECEIPT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self._seal is not _RUN_RECEIPT_SEAL:
            raise TypeError("run receipts can only be constructed by the runner or verifier")
        for text_value, text_name in (
            (self.release_id, "release_id"),
            (self.input_schema, "input_schema"),
            (self.output_schema, "output_schema"),
        ):
            if type(text_value) is not str or not text_value:
                raise ValueError(f"{text_name} must be nonempty text")
        for digest_value, digest_name in (
            (self.protocol_sha256, "protocol_sha256"),
            (self.checker_source_bundle_sha256, "checker_source_bundle_sha256"),
            (self.runtime_bundle_sha256, "runtime_bundle_sha256"),
            (self.checker_zipapp_sha256, "checker_zipapp_sha256"),
            (self.runtime_python_sha256, "runtime_python_sha256"),
            (self.input_sha256, "input_sha256"),
        ):
            _require_sha256(digest_value, digest_name)
        _require_nonnegative_int(self.input_byte_count, "input_byte_count")
        _require_positive_int(self.watchdog_seconds, "watchdog_seconds")
        for limit_value, limit_name in (
            (self.input_byte_limit, "input_byte_limit"),
            (self.output_byte_limit, "output_byte_limit"),
            (self.stdout_byte_limit, "stdout_byte_limit"),
            (self.stderr_byte_limit, "stderr_byte_limit"),
            (self.retained_bundle_byte_limit, "retained_bundle_byte_limit"),
            (self.zip_member_limit, "zip_member_limit"),
            (self.zip_uncompressed_byte_limit, "zip_uncompressed_byte_limit"),
            (self.path_depth_limit, "path_depth_limit"),
        ):
            _require_positive_int(limit_value, limit_name)
        if self.input_byte_count > self.input_byte_limit:
            raise ValueError("receipted input exceeds the runner byte limit")
        if type(self.primary) is not ContinuationInvocationReceipt:
            raise TypeError("primary must be a ContinuationInvocationReceipt")
        if (
            self.publication_rerun is not None
            and type(self.publication_rerun) is not ContinuationInvocationReceipt
        ):
            raise TypeError("publication_rerun must be a ContinuationInvocationReceipt or null")
        if (
            self.publication_output_match is not None
            and type(self.publication_output_match) is not bool
        ):
            raise TypeError("publication_output_match must be a boolean or null")
        if self.status not in {"accepted", "environment_failure"}:
            raise ValueError("status is unregistered")
        if self.failure_classification is not None and (
            type(self.failure_classification) is not str or not self.failure_classification
        ):
            raise ValueError("failure_classification must be nonempty text or null")
        if self.scientific_output_sha256 is not None:
            _require_sha256(self.scientific_output_sha256, "scientific_output_sha256")

        if self.primary.input_sha256 != self.input_sha256 or (
            self.primary.input_byte_count != self.input_byte_count
        ):
            raise ValueError("primary receipt does not bind the top-level input")
        if self.publication_rerun is not None and (
            self.publication_rerun.input_sha256 != self.input_sha256
            or self.publication_rerun.input_byte_count != self.input_byte_count
        ):
            raise ValueError("publication receipt does not bind the top-level input")
        for invocation_name, invocation in (
            ("primary", self.primary),
            ("publication", self.publication_rerun),
        ):
            if invocation is None:
                continue
            if invocation.stdout_byte_count > self.stdout_byte_limit:
                raise ValueError(
                    f"{invocation_name} stdout capture exceeds the receipted byte limit"
                )
            if invocation.stderr_byte_count > self.stderr_byte_limit:
                raise ValueError(
                    f"{invocation_name} stderr capture exceeds the receipted byte limit"
                )
            if (
                invocation.stdout_overflow
                and invocation.stdout_byte_count != self.stdout_byte_limit
            ):
                raise ValueError(
                    f"{invocation_name} stdout overflow requires capture exactly at "
                    "the receipted byte limit"
                )
            if (
                invocation.stderr_overflow
                and invocation.stderr_byte_count != self.stderr_byte_limit
            ):
                raise ValueError(
                    f"{invocation_name} stderr overflow requires capture exactly at "
                    "the receipted byte limit"
                )
            if invocation.classification == "output_limit_exceeded" and (
                invocation.output_byte_count is None
                or invocation.output_byte_count <= self.output_byte_limit
            ):
                raise ValueError(
                    f"{invocation_name} output limit classification requires a byte count "
                    "above the receipted byte limit"
                )
            if (
                invocation.classification in {"success", "invalid_output"}
                and invocation.output_byte_count is not None
                and invocation.output_byte_count > self.output_byte_limit
            ):
                raise ValueError(
                    f"{invocation_name} exact output exceeds the receipted byte limit"
                )
        primary_root = _invocation_run_root(self.primary.command)
        if not primary_root.name.startswith("embedbench-continuation-primary-"):
            raise ValueError("primary receipt has an invalid fresh run directory")
        if self.publication_rerun is not None:
            publication_root = _invocation_run_root(self.publication_rerun.command)
            if primary_root == publication_root:
                raise ValueError("publication replay requires distinct fresh run directories")
            if not publication_root.name.startswith("embedbench-continuation-publication-"):
                raise ValueError("publication receipt has an invalid fresh run directory")
        if self.status == "accepted":
            if (
                self.failure_classification is not None
                or self.publication_output_match is not True
                or self.primary.classification != "success"
                or self.publication_rerun is None
                or self.publication_rerun.classification != "success"
                or self.scientific_output_sha256 is None
                or self.primary.output_sha256 != self.scientific_output_sha256
                or self.publication_rerun.output_sha256 != self.scientific_output_sha256
                or self.primary.output_byte_count is None
                or self.primary.output_byte_count > self.output_byte_limit
                or self.publication_rerun.output_byte_count is None
                or self.publication_rerun.output_byte_count > self.output_byte_limit
            ):
                raise ValueError("accepted receipt requires identical successful publication runs")
            if (
                self.publication_rerun is not None
                and self.primary.output_byte_count != self.publication_rerun.output_byte_count
            ):
                raise ValueError("accepted receipt requires matching output byte counts")
        elif self.failure_classification is None or self.scientific_output_sha256 is not None:
            raise ValueError("environment_failure requires a classification and no scientific hash")
        elif self.primary.classification != "success":
            if (
                self.publication_rerun is not None
                or self.publication_output_match is not None
                or self.failure_classification != f"primary_{self.primary.classification}"
            ):
                raise ValueError("failure receipt has inconsistent primary-failure topology")
        elif self.publication_rerun is None:
            raise ValueError("failure receipt is missing its publication rerun")
        elif self.publication_rerun.classification != "success":
            if (
                self.publication_output_match is not None
                or self.failure_classification
                != f"publication_{self.publication_rerun.classification}"
            ):
                raise ValueError("failure receipt has inconsistent publication-failure topology")
        elif (
            self.publication_output_match is not False
            or self.failure_classification != "publication_output_mismatch"
        ):
            raise ValueError("failure receipt has inconsistent replay-mismatch topology")

    def to_dict(self) -> dict[str, object]:
        return {
            "checker_source_bundle_sha256": self.checker_source_bundle_sha256,
            "checker_zipapp_sha256": self.checker_zipapp_sha256,
            "environment": EMPTY_EXEC_ENVIRONMENT,
            "failure_classification": self.failure_classification,
            "input_byte_count": self.input_byte_count,
            "input_byte_limit": self.input_byte_limit,
            "input_schema": self.input_schema,
            "input_sha256": self.input_sha256,
            "output_schema": self.output_schema,
            "output_byte_limit": self.output_byte_limit,
            "primary": self.primary.to_dict(),
            "protocol_sha256": self.protocol_sha256,
            "publication_output_match": self.publication_output_match,
            "publication_rerun": (
                None if self.publication_rerun is None else self.publication_rerun.to_dict()
            ),
            "release_id": self.release_id,
            "runtime_bundle_sha256": self.runtime_bundle_sha256,
            "runtime_python_sha256": self.runtime_python_sha256,
            "schema": self.schema,
            "schema_version": self.schema_version,
            "scientific_output_sha256": self.scientific_output_sha256,
            "status": self.status,
            "stderr_byte_limit": self.stderr_byte_limit,
            "stdout_byte_limit": self.stdout_byte_limit,
            "retained_bundle_byte_limit": self.retained_bundle_byte_limit,
            "zip_member_limit": self.zip_member_limit,
            "zip_uncompressed_byte_limit": self.zip_uncompressed_byte_limit,
            "path_depth_limit": self.path_depth_limit,
            "watchdog_seconds": self.watchdog_seconds,
        }

    @property
    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def to_bytes(self) -> bytes:
        """Return the exact canonical JSON receipt bytes."""

        return self.canonical_bytes

    @property
    def digest(self) -> str:
        return _sha256(self.canonical_bytes)

    @property
    def sha256(self) -> str:
        """Return the receipt digest using the package's conventional property name."""

        return self.digest


@dataclass(frozen=True, slots=True, init=False)
class ContinuationRunResult:
    """A receipt plus output bytes exposed only after a reproducible publication replay."""

    receipt: ContinuationRunReceipt
    scientific_output_bytes: bytes | None
    _seal: object

    def __new__(cls) -> Self:
        raise TypeError("continuation results can only be constructed by the runner")

    def __post_init__(self) -> None:
        if self._seal is not _RUN_RESULT_SEAL:
            raise TypeError("continuation results can only be constructed by the runner")
        if type(self.receipt) is not ContinuationRunReceipt:
            raise TypeError("receipt must be a ContinuationRunReceipt")
        if (
            self.scientific_output_bytes is not None
            and type(self.scientific_output_bytes) is not bytes
        ):
            raise TypeError("scientific_output_bytes must be exact bytes or null")
        if self.receipt.status == "accepted":
            if self.scientific_output_bytes is None or (
                _sha256(self.scientific_output_bytes) != self.receipt.scientific_output_sha256
            ):
                raise ValueError(
                    "accepted result must expose the exactly receipted scientific bytes"
                )
            expected_output_byte_count = len(self.scientific_output_bytes)
            publication = self.receipt.publication_rerun
            if (
                self.receipt.primary.output_byte_count != expected_output_byte_count
                or publication is None
                or publication.output_byte_count != expected_output_byte_count
            ):
                raise ValueError(
                    "accepted result output byte counts must match the scientific bytes"
                )
        elif self.scientific_output_bytes is not None:
            raise ValueError("failed execution must not expose output as scientific evidence")


def _make_invocation_receipt(
    *,
    command: tuple[str, ...],
    termination_mode: str,
    classification: str,
    exit_code: int | None,
    signal_number: int | None,
    input_sha256: str,
    input_byte_count: int,
    output_present: bool,
    output_sha256: str | None,
    output_byte_count: int | None,
    stdout_sha256: str,
    stdout_byte_count: int,
    stdout_overflow: bool,
    stderr_sha256: str,
    stderr_byte_count: int,
    stderr_overflow: bool,
) -> ContinuationInvocationReceipt:
    receipt = object.__new__(ContinuationInvocationReceipt)
    object.__setattr__(receipt, "command", command)
    object.__setattr__(receipt, "termination_mode", termination_mode)
    object.__setattr__(receipt, "classification", classification)
    object.__setattr__(receipt, "exit_code", exit_code)
    object.__setattr__(receipt, "signal_number", signal_number)
    object.__setattr__(receipt, "input_sha256", input_sha256)
    object.__setattr__(receipt, "input_byte_count", input_byte_count)
    object.__setattr__(receipt, "output_present", output_present)
    object.__setattr__(receipt, "output_sha256", output_sha256)
    object.__setattr__(receipt, "output_byte_count", output_byte_count)
    object.__setattr__(receipt, "stdout_sha256", stdout_sha256)
    object.__setattr__(receipt, "stdout_byte_count", stdout_byte_count)
    object.__setattr__(receipt, "stdout_overflow", stdout_overflow)
    object.__setattr__(receipt, "stderr_sha256", stderr_sha256)
    object.__setattr__(receipt, "stderr_byte_count", stderr_byte_count)
    object.__setattr__(receipt, "stderr_overflow", stderr_overflow)
    object.__setattr__(receipt, "_seal", _INVOCATION_RECEIPT_SEAL)
    receipt.__post_init__()
    return receipt


def _make_run_receipt(
    *,
    release_id: str,
    protocol_sha256: str,
    checker_source_bundle_sha256: str,
    runtime_bundle_sha256: str,
    checker_zipapp_sha256: str,
    runtime_python_sha256: str,
    input_schema: str,
    output_schema: str,
    watchdog_seconds: int,
    input_byte_limit: int,
    output_byte_limit: int,
    stdout_byte_limit: int,
    stderr_byte_limit: int,
    retained_bundle_byte_limit: int,
    zip_member_limit: int,
    zip_uncompressed_byte_limit: int,
    path_depth_limit: int,
    input_sha256: str,
    input_byte_count: int,
    primary: ContinuationInvocationReceipt,
    publication_rerun: ContinuationInvocationReceipt | None,
    publication_output_match: bool | None,
    status: str,
    failure_classification: str | None,
    scientific_output_sha256: str | None,
) -> ContinuationRunReceipt:
    receipt = object.__new__(ContinuationRunReceipt)
    object.__setattr__(receipt, "release_id", release_id)
    object.__setattr__(receipt, "protocol_sha256", protocol_sha256)
    object.__setattr__(
        receipt,
        "checker_source_bundle_sha256",
        checker_source_bundle_sha256,
    )
    object.__setattr__(receipt, "runtime_bundle_sha256", runtime_bundle_sha256)
    object.__setattr__(receipt, "checker_zipapp_sha256", checker_zipapp_sha256)
    object.__setattr__(receipt, "runtime_python_sha256", runtime_python_sha256)
    object.__setattr__(receipt, "input_schema", input_schema)
    object.__setattr__(receipt, "output_schema", output_schema)
    object.__setattr__(receipt, "watchdog_seconds", watchdog_seconds)
    object.__setattr__(receipt, "input_byte_limit", input_byte_limit)
    object.__setattr__(receipt, "output_byte_limit", output_byte_limit)
    object.__setattr__(receipt, "stdout_byte_limit", stdout_byte_limit)
    object.__setattr__(receipt, "stderr_byte_limit", stderr_byte_limit)
    object.__setattr__(receipt, "retained_bundle_byte_limit", retained_bundle_byte_limit)
    object.__setattr__(receipt, "zip_member_limit", zip_member_limit)
    object.__setattr__(receipt, "zip_uncompressed_byte_limit", zip_uncompressed_byte_limit)
    object.__setattr__(receipt, "path_depth_limit", path_depth_limit)
    object.__setattr__(receipt, "input_sha256", input_sha256)
    object.__setattr__(receipt, "input_byte_count", input_byte_count)
    object.__setattr__(receipt, "primary", primary)
    object.__setattr__(receipt, "publication_rerun", publication_rerun)
    object.__setattr__(receipt, "publication_output_match", publication_output_match)
    object.__setattr__(receipt, "status", status)
    object.__setattr__(receipt, "failure_classification", failure_classification)
    object.__setattr__(receipt, "scientific_output_sha256", scientific_output_sha256)
    object.__setattr__(receipt, "_seal", _RUN_RECEIPT_SEAL)
    receipt.__post_init__()
    return receipt


def _make_run_result(
    receipt: ContinuationRunReceipt,
    scientific_output_bytes: bytes | None,
) -> ContinuationRunResult:
    result = object.__new__(ContinuationRunResult)
    object.__setattr__(result, "receipt", receipt)
    object.__setattr__(result, "scientific_output_bytes", scientific_output_bytes)
    object.__setattr__(result, "_seal", _RUN_RESULT_SEAL)
    result.__post_init__()
    return result


@dataclass(frozen=True, slots=True)
class _ProcessCapture:
    termination_mode: str
    returncode: int | None
    stdout: bytes
    stderr: bytes
    stdout_overflow: bool = False
    stderr_overflow: bool = False


@dataclass(frozen=True, slots=True)
class _FileCapture:
    present: bool
    safe: bool
    payload: bytes | None
    limit_exceeded: bool = False
    observed_byte_count: int | None = None


@dataclass(frozen=True, slots=True)
class _InvocationResult:
    receipt: ContinuationInvocationReceipt
    output_bytes: bytes | None


def _snapshot_protocol(
    protocol: VerifiedContinuationProtocol,
    *,
    expected_protocol_sha256: str,
    expected_release_id: str,
) -> ContinuationProtocolSnapshot:
    return snapshot_continuation_protocol(
        protocol,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_release_id=expected_release_id,
    )


def _write_exact_file(path: Path, payload: bytes, *, executable: bool) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o700 if executable else 0o600)
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short write while staging retained continuation bytes")
            written += count
        os.fchmod(descriptor, 0o700 if executable else 0o600)
    finally:
        os.close(descriptor)


def _write_exact_file_at(
    directory_fd: int,
    name: str,
    payload: bytes,
    *,
    executable: bool,
) -> None:
    if type(name) is not str or not name or "/" in name or name in {".", ".."}:
        raise ValueError("fd-relative staged name must be one safe path component")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(
        name,
        flags,
        0o700 if executable else 0o600,
        dir_fd=directory_fd,
    )
    try:
        view = memoryview(payload)
        written = 0
        while written < len(view):
            count = os.write(descriptor, view[written:])
            if count <= 0:
                raise OSError("short write while staging retained continuation bytes")
            written += count
        os.fchmod(descriptor, 0o700 if executable else 0o600)
    finally:
        os.close(descriptor)


def _stable_file_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _stable_directory_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
    )


def _directory_path_matches_fd(
    path: Path,
    directory_fd: int,
    expected_identity: tuple[int, int, int, int],
) -> bool:
    try:
        held = os.fstat(directory_fd)
        visible = os.stat(path, follow_symlinks=False)
    except OSError:
        return False
    return (
        stat.S_ISDIR(held.st_mode)
        and stat.S_ISDIR(visible.st_mode)
        and _stable_directory_identity(held) == expected_identity
        and _stable_directory_identity(visible) == expected_identity
    )


def _capture_open_regular_file(
    descriptor: int,
    *,
    maximum_bytes: int | None,
) -> _FileCapture:
    before = os.fstat(descriptor)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid != os.getuid():
        return _FileCapture(True, False, None)
    if maximum_bytes is not None and before.st_size > maximum_bytes:
        return _FileCapture(
            True,
            False,
            None,
            limit_exceeded=True,
            observed_byte_count=before.st_size,
        )
    chunks: list[bytes] = []
    captured_count = 0
    while True:
        read_size = 1024 * 1024
        if maximum_bytes is not None:
            read_size = min(read_size, maximum_bytes - captured_count + 1)
        chunk = os.read(descriptor, read_size)
        if not chunk:
            break
        captured_count += len(chunk)
        if maximum_bytes is not None and captured_count > maximum_bytes:
            return _FileCapture(
                True,
                False,
                None,
                limit_exceeded=True,
                observed_byte_count=captured_count,
            )
        chunks.append(chunk)
    after = os.fstat(descriptor)
    if _stable_file_identity(before) != _stable_file_identity(after):
        return _FileCapture(True, False, None)
    payload = b"".join(chunks)
    return _FileCapture(
        True,
        True,
        payload,
        observed_byte_count=len(payload),
    )


def _read_regular_file(path: Path, *, maximum_bytes: int | None = None) -> _FileCapture:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return _FileCapture(False, True, None)
    except OSError:
        return _FileCapture(os.path.lexists(path), False, None)
    try:
        return _capture_open_regular_file(descriptor, maximum_bytes=maximum_bytes)
    finally:
        os.close(descriptor)


def _entry_exists_at(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError:
        return True
    return True


def _read_regular_file_at(
    directory_fd: int,
    name: str,
    *,
    maximum_bytes: int,
) -> _FileCapture:
    if type(name) is not str or not name or "/" in name or name in {".", ".."}:
        raise ValueError("fd-relative output name must be one safe path component")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(name, flags, dir_fd=directory_fd)
    except FileNotFoundError:
        return _FileCapture(False, True, None)
    except OSError:
        return _FileCapture(_entry_exists_at(directory_fd, name), False, None)
    try:
        return _capture_open_regular_file(descriptor, maximum_bytes=maximum_bytes)
    except OSError:
        return _FileCapture(True, False, None)
    finally:
        os.close(descriptor)


def _stage_bundle(root: Path, files: tuple[VerifiedSourceFile, ...]) -> None:
    root.mkdir(mode=0o700)
    for entry in files:
        target = root.joinpath(*PurePosixPath(entry.relative_path).parts)
        _write_exact_file(target, entry.payload, executable=entry.executable)
        captured = _read_regular_file(target)
        if not captured.safe or captured.payload != entry.payload:
            raise OSError("staged continuation bytes failed exact readback")


def _kill_process_group(process: subprocess.Popen[bytes]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        with suppress(ProcessLookupError):
            process.kill()


@dataclass(slots=True)
class _StreamCapture:
    stream: BinaryIO
    byte_limit: int
    payload: bytearray
    overflow: bool = False


def _consume_ready_stream(state: _StreamCapture) -> bool:
    """Consume one nonblocking chunk; return true only at EOF."""

    try:
        chunk = os.read(state.stream.fileno(), _CAPTURE_CHUNK_BYTES)
    except BlockingIOError:
        return False
    if not chunk:
        return True
    remaining = max(0, state.byte_limit - len(state.payload))
    state.payload.extend(chunk[:remaining])
    if len(chunk) > remaining:
        state.overflow = True
    return False


def _close_selector_stream(
    selector: selectors.BaseSelector,
    state: _StreamCapture,
) -> None:
    with suppress(KeyError, ValueError):
        selector.unregister(state.stream)
    with suppress(OSError):
        state.stream.close()


def _drain_ready_streams(
    selector: selectors.BaseSelector,
    *,
    wait_seconds: float,
) -> None:
    for key, _events in selector.select(wait_seconds):
        state = cast(_StreamCapture, key.data)
        if _consume_ready_stream(state):
            _close_selector_stream(selector, state)


def _run_process(
    command: tuple[str, ...],
    *,
    working_directory: Path,
    watchdog_seconds: int | float,
    stdout_byte_limit: int = MAX_CONTINUATION_STDOUT_BYTES,
    stderr_byte_limit: int = MAX_CONTINUATION_STDERR_BYTES,
) -> _ProcessCapture:
    watchdog_timeout = _require_positive_timeout(watchdog_seconds, "watchdog_seconds")
    _require_positive_int(stdout_byte_limit, "stdout_byte_limit")
    _require_positive_int(stderr_byte_limit, "stderr_byte_limit")
    try:
        process = subprocess.Popen(
            command,
            cwd=os.fspath(working_directory),
            env={},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            start_new_session=True,
            shell=False,
        )
    except OSError:
        return _ProcessCapture("spawn_error", None, b"", b"")
    if process.stdout is None or process.stderr is None:
        _kill_process_group(process)
        return _ProcessCapture("runner_error", None, b"", b"")

    stdout = _StreamCapture(cast(BinaryIO, process.stdout), stdout_byte_limit, bytearray())
    stderr = _StreamCapture(cast(BinaryIO, process.stderr), stderr_byte_limit, bytearray())
    selector = selectors.DefaultSelector()
    termination_mode: str | None = None
    returncode: int | None = None
    process_exited_at: float | None = None
    deadline = time.monotonic() + watchdog_timeout
    try:
        for state in (stdout, stderr):
            os.set_blocking(state.stream.fileno(), False)
            selector.register(state.stream, selectors.EVENT_READ, state)

        while True:
            polled = process.poll()
            if type(polled) is int and returncode is None:
                returncode = polled
                process_exited_at = time.monotonic()
                _kill_process_group(process)
            if stdout.overflow or stderr.overflow:
                termination_mode = "capture_limit_exceeded"
                break
            if not selector.get_map() and returncode is not None:
                break

            now = time.monotonic()
            active_deadline = deadline
            if process_exited_at is not None:
                active_deadline = min(
                    active_deadline,
                    process_exited_at + _CAPTURE_DRAIN_GRACE_SECONDS,
                )
            if now >= active_deadline:
                termination_mode = "watchdog_timeout" if returncode is None else "runner_error"
                break
            wait_seconds = min(_PROCESS_POLL_SECONDS, active_deadline - now)
            if selector.get_map():
                _drain_ready_streams(selector, wait_seconds=wait_seconds)
            else:
                time.sleep(wait_seconds)
    except (OSError, ValueError):
        termination_mode = "runner_error"

    if termination_mode is not None:
        _kill_process_group(process)
    with suppress(OSError, subprocess.TimeoutExpired):
        process.wait(timeout=_CAPTURE_DRAIN_GRACE_SECONDS)

    drain_deadline = time.monotonic() + _CAPTURE_DRAIN_GRACE_SECONDS
    capture_incomplete = False
    try:
        while selector.get_map() and time.monotonic() < drain_deadline:
            _drain_ready_streams(
                selector,
                wait_seconds=min(
                    _PROCESS_POLL_SECONDS,
                    max(0.0, drain_deadline - time.monotonic()),
                ),
            )
    except (OSError, ValueError):
        if termination_mode is None:
            termination_mode = "runner_error"
    finally:
        capture_incomplete = bool(selector.get_map())
        for state in (stdout, stderr):
            _close_selector_stream(selector, state)
        selector.close()

    if stdout.overflow or stderr.overflow:
        termination_mode = "capture_limit_exceeded"
    if termination_mode is None and capture_incomplete:
        termination_mode = "runner_error"
    if termination_mode is None and type(returncode) is not int:
        returncode = process.returncode
        if type(returncode) is not int:
            termination_mode = "runner_error"
    if termination_mode is not None:
        return _ProcessCapture(
            termination_mode,
            None,
            bytes(stdout.payload),
            bytes(stderr.payload),
            stdout.overflow,
            stderr.overflow,
        )
    final_returncode = cast(int, returncode)
    mode = "signal" if final_returncode < 0 else "exited"
    return _ProcessCapture(
        mode,
        final_returncode,
        bytes(stdout.payload),
        bytes(stderr.payload),
        False,
        False,
    )


def _classify(
    process: _ProcessCapture,
    output: _FileCapture,
    *,
    io_directory_safe: bool,
    output_schema: str,
    input_sha256: str,
    state_sha256: str,
) -> tuple[str, int | None, int | None]:
    if process.termination_mode == "spawn_error":
        return "spawn_error", None, None
    if process.termination_mode == "watchdog_timeout":
        return "watchdog_timeout", None, None
    if process.termination_mode == "capture_limit_exceeded":
        if process.stdout_overflow:
            return "stdout_limit_exceeded", None, None
        if process.stderr_overflow:
            return "stderr_limit_exceeded", None, None
        raise RuntimeError("capture limit termination is missing its overflow flag")
    if process.termination_mode == "runner_error":
        return "runner_error", None, None
    if process.termination_mode == "signal":
        if process.returncode is None or process.returncode >= 0:
            raise RuntimeError("signalled subprocess did not report a negative return code")
        return "signal", None, -process.returncode
    if process.returncode is None or process.returncode < 0:
        raise RuntimeError("exited subprocess did not report a nonnegative return code")
    if process.returncode != 0:
        return "nonzero_exit", process.returncode, None
    if not io_directory_safe:
        return "unsafe_io_directory", 0, None
    if not output.present:
        return "missing_output", 0, None
    if output.limit_exceeded:
        return "output_limit_exceeded", 0, None
    if not output.safe or output.payload is None:
        return "unsafe_output", 0, None
    try:
        _validate_output_envelope(
            output.payload,
            output_schema=output_schema,
            input_sha256=input_sha256,
            state_sha256=state_sha256,
        )
    except (TypeError, ValueError):
        return "invalid_output", 0, None
    return "success", 0, None


def _invoke_once(
    snapshot: ContinuationProtocolSnapshot,
    input_bytes: bytes,
    *,
    input_sha256: str,
    state_sha256: str,
    run_kind: str,
) -> _InvocationResult:
    with tempfile.TemporaryDirectory(prefix=f"embedbench-continuation-{run_kind}-") as raw_root:
        root = Path(raw_root).resolve(strict=True)
        runtime_root = root / "runtime"
        checker_root = root / "checker"
        io_root = root / "io"
        input_path = io_root / "input.json"
        output_path = io_root / "output.json"
        substitutions = {
            "runtime_root": os.fspath(runtime_root),
            "runtime_python": snapshot.runtime_python,
            "checker_root": os.fspath(checker_root),
            "checker_zipapp": snapshot.checker_zipapp,
            "input_path": os.fspath(input_path),
            "output_path": os.fspath(output_path),
        }
        command = tuple(
            component.format_map(substitutions) for component in snapshot.command_template
        )
        expected_command = (
            os.fspath(runtime_root.joinpath(*PurePosixPath(snapshot.runtime_python).parts)),
            "-I",
            os.fspath(checker_root.joinpath(*PurePosixPath(snapshot.checker_zipapp).parts)),
            os.fspath(input_path),
            os.fspath(output_path),
        )
        if command != expected_command or _invocation_run_root(command) != root:
            raise RuntimeError("verified command does not resolve to the frozen absolute argv")

        io_fd: int | None = None
        try:
            os.chmod(root, 0o700)
            _stage_bundle(runtime_root, snapshot.runtime_files)
            _stage_bundle(checker_root, snapshot.checker_files)
            io_root.mkdir(mode=0o700)
            io_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
            io_fd = os.open(io_root, io_flags)
            io_identity = _stable_directory_identity(os.fstat(io_fd))
            _write_exact_file_at(io_fd, "input.json", input_bytes, executable=False)
            captured_input = _read_regular_file_at(
                io_fd,
                "input.json",
                maximum_bytes=snapshot.input_byte_limit,
            )
            if not captured_input.safe or captured_input.payload != input_bytes:
                raise OSError("staged continuation input failed exact readback")
        except (OSError, RuntimeError, ValueError):
            if io_fd is not None:
                with suppress(OSError):
                    os.close(io_fd)
            receipt = _make_invocation_receipt(
                command=command,
                termination_mode="staging_error",
                classification="staging_error",
                exit_code=None,
                signal_number=None,
                input_sha256=input_sha256,
                input_byte_count=len(input_bytes),
                output_present=False,
                output_sha256=None,
                output_byte_count=None,
                stdout_sha256=_sha256(b""),
                stdout_byte_count=0,
                stdout_overflow=False,
                stderr_sha256=_sha256(b""),
                stderr_byte_count=0,
                stderr_overflow=False,
            )
            return _InvocationResult(receipt, None)
        try:
            process = _run_process(
                command,
                working_directory=root,
                watchdog_seconds=snapshot.watchdog_seconds,
                stdout_byte_limit=snapshot.stdout_byte_limit,
                stderr_byte_limit=snapshot.stderr_byte_limit,
            )
            io_directory_safe = _directory_path_matches_fd(
                io_root,
                io_fd,
                io_identity,
            )
            output = _read_regular_file_at(
                io_fd,
                "output.json",
                maximum_bytes=snapshot.output_byte_limit,
            )
        finally:
            if io_fd is not None:
                with suppress(OSError):
                    os.close(io_fd)

        classification, exit_code, signal_number = _classify(
            process,
            output,
            io_directory_safe=io_directory_safe,
            output_schema=snapshot.output_schema,
            input_sha256=input_sha256,
            state_sha256=state_sha256,
        )
        output_sha256 = None
        output_byte_count = output.observed_byte_count
        if output.safe and output.payload is not None:
            output_sha256 = _sha256(output.payload)
            output_byte_count = len(output.payload)
        receipt = _make_invocation_receipt(
            command=command,
            termination_mode=process.termination_mode,
            classification=classification,
            exit_code=exit_code,
            signal_number=signal_number,
            input_sha256=input_sha256,
            input_byte_count=len(input_bytes),
            output_present=output.present,
            output_sha256=output_sha256,
            output_byte_count=output_byte_count,
            stdout_sha256=_sha256(process.stdout),
            stdout_byte_count=len(process.stdout),
            stdout_overflow=process.stdout_overflow,
            stderr_sha256=_sha256(process.stderr),
            stderr_byte_count=len(process.stderr),
            stderr_overflow=process.stderr_overflow,
        )
        return _InvocationResult(
            receipt,
            output.payload if classification == "success" else None,
        )


def _failure_receipt(
    snapshot: ContinuationProtocolSnapshot,
    input_bytes: bytes,
    primary: ContinuationInvocationReceipt,
    *,
    publication_rerun: ContinuationInvocationReceipt | None,
    publication_output_match: bool | None,
    failure_classification: str,
) -> ContinuationRunReceipt:
    return _make_run_receipt(
        release_id=snapshot.release_id,
        protocol_sha256=snapshot.protocol_sha256,
        checker_source_bundle_sha256=snapshot.checker_source_bundle_sha256,
        runtime_bundle_sha256=snapshot.runtime_bundle_sha256,
        checker_zipapp_sha256=snapshot.checker_zipapp_sha256,
        runtime_python_sha256=snapshot.runtime_python_sha256,
        input_schema=snapshot.input_schema,
        output_schema=snapshot.output_schema,
        watchdog_seconds=snapshot.watchdog_seconds,
        input_byte_limit=snapshot.input_byte_limit,
        output_byte_limit=snapshot.output_byte_limit,
        stdout_byte_limit=snapshot.stdout_byte_limit,
        stderr_byte_limit=snapshot.stderr_byte_limit,
        retained_bundle_byte_limit=snapshot.retained_bundle_byte_limit,
        zip_member_limit=snapshot.zip_member_limit,
        zip_uncompressed_byte_limit=snapshot.zip_uncompressed_byte_limit,
        path_depth_limit=snapshot.path_depth_limit,
        input_sha256=_sha256(input_bytes),
        input_byte_count=len(input_bytes),
        primary=primary,
        publication_rerun=publication_rerun,
        publication_output_match=publication_output_match,
        status="environment_failure",
        failure_classification=failure_classification,
        scientific_output_sha256=None,
    )


def run_verified_continuation(
    protocol: VerifiedContinuationProtocol,
    input_bytes: bytes,
    *,
    expected_protocol_sha256: str,
    expected_release_id: str,
) -> ContinuationRunResult:
    """Execute and independently replay one canonical continuation input.

    Only byte-identical, schema-valid output from two successful fresh processes is returned
    as scientific evidence. Every process or publication failure returns no scientific bytes.
    Trust-root and canonical-input failures raise before subprocess creation.
    """

    snapshot = _snapshot_protocol(
        protocol,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_release_id=expected_release_id,
    )
    if type(input_bytes) is not bytes:
        raise TypeError("continuation input must be exact bytes")
    input_sha256, state_sha256 = _parse_input_binding(
        input_bytes,
        input_schema=snapshot.input_schema,
        input_byte_limit=snapshot.input_byte_limit,
    )

    primary_result = _invoke_once(
        snapshot,
        input_bytes,
        input_sha256=input_sha256,
        state_sha256=state_sha256,
        run_kind="primary",
    )
    primary = primary_result.receipt
    if primary.classification != "success":
        receipt = _failure_receipt(
            snapshot,
            input_bytes,
            primary,
            publication_rerun=None,
            publication_output_match=None,
            failure_classification=f"primary_{primary.classification}",
        )
        return _make_run_result(receipt, None)

    replay_snapshot = _snapshot_protocol(
        protocol,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_release_id=expected_release_id,
    )
    if replay_snapshot != snapshot:
        raise ValueError("verified continuation protocol changed before publication replay")
    replay_result = _invoke_once(
        replay_snapshot,
        input_bytes,
        input_sha256=input_sha256,
        state_sha256=state_sha256,
        run_kind="publication",
    )
    replay = replay_result.receipt
    if replay.classification != "success":
        receipt = _failure_receipt(
            snapshot,
            input_bytes,
            primary,
            publication_rerun=replay,
            publication_output_match=None,
            failure_classification=f"publication_{replay.classification}",
        )
        return _make_run_result(receipt, None)

    output_match = (
        primary_result.output_bytes == replay_result.output_bytes
        and primary.output_sha256 == replay.output_sha256
    )
    if not output_match:
        receipt = _failure_receipt(
            snapshot,
            input_bytes,
            primary,
            publication_rerun=replay,
            publication_output_match=False,
            failure_classification="publication_output_mismatch",
        )
        return _make_run_result(receipt, None)

    if primary_result.output_bytes is None or primary.output_sha256 is None:
        raise RuntimeError("successful continuation invocation lost its scientific output")
    receipt = _make_run_receipt(
        release_id=snapshot.release_id,
        protocol_sha256=snapshot.protocol_sha256,
        checker_source_bundle_sha256=snapshot.checker_source_bundle_sha256,
        runtime_bundle_sha256=snapshot.runtime_bundle_sha256,
        checker_zipapp_sha256=snapshot.checker_zipapp_sha256,
        runtime_python_sha256=snapshot.runtime_python_sha256,
        input_schema=snapshot.input_schema,
        output_schema=snapshot.output_schema,
        watchdog_seconds=snapshot.watchdog_seconds,
        input_byte_limit=snapshot.input_byte_limit,
        output_byte_limit=snapshot.output_byte_limit,
        stdout_byte_limit=snapshot.stdout_byte_limit,
        stderr_byte_limit=snapshot.stderr_byte_limit,
        retained_bundle_byte_limit=snapshot.retained_bundle_byte_limit,
        zip_member_limit=snapshot.zip_member_limit,
        zip_uncompressed_byte_limit=snapshot.zip_uncompressed_byte_limit,
        path_depth_limit=snapshot.path_depth_limit,
        input_sha256=_sha256(input_bytes),
        input_byte_count=len(input_bytes),
        primary=primary,
        publication_rerun=replay,
        publication_output_match=True,
        status="accepted",
        failure_classification=None,
        scientific_output_sha256=primary.output_sha256,
    )
    return _make_run_result(receipt, primary_result.output_bytes)


def _parse_invocation_receipt(document: object) -> ContinuationInvocationReceipt:
    fields = require_exact_keys(
        document,
        _INVOCATION_RECEIPT_FIELDS,
        "continuation invocation receipt",
    )
    raw_command = fields["command"]
    if type(raw_command) is not list or any(type(item) is not str for item in raw_command):
        raise TypeError("continuation invocation command must be an exact JSON array of text")
    return _make_invocation_receipt(
        command=tuple(raw_command),
        termination_mode=_require_text(fields["termination_mode"], "termination_mode"),
        classification=_require_text(fields["classification"], "classification"),
        exit_code=_optional_nonnegative_int(fields["exit_code"], "exit_code"),
        signal_number=_optional_positive_int(fields["signal_number"], "signal_number"),
        input_sha256=_require_sha256(fields["input_sha256"], "input_sha256"),
        input_byte_count=_require_nonnegative_int(
            fields["input_byte_count"],
            "input_byte_count",
        ),
        output_present=_require_bool(fields["output_present"], "output_present"),
        output_sha256=_optional_sha256(fields["output_sha256"], "output_sha256"),
        output_byte_count=_optional_nonnegative_int(
            fields["output_byte_count"],
            "output_byte_count",
        ),
        stdout_sha256=_require_sha256(fields["stdout_sha256"], "stdout_sha256"),
        stdout_byte_count=_require_nonnegative_int(
            fields["stdout_byte_count"],
            "stdout_byte_count",
        ),
        stdout_overflow=_require_bool(fields["stdout_overflow"], "stdout_overflow"),
        stderr_sha256=_require_sha256(fields["stderr_sha256"], "stderr_sha256"),
        stderr_byte_count=_require_nonnegative_int(
            fields["stderr_byte_count"],
            "stderr_byte_count",
        ),
        stderr_overflow=_require_bool(fields["stderr_overflow"], "stderr_overflow"),
    )


def verify_continuation_run_receipt(
    receipt_bytes: bytes,
    *,
    expected_receipt_sha256: str,
    protocol: VerifiedContinuationProtocol,
    expected_protocol_sha256: str,
    expected_release_id: str,
    expected_checker_source_bundle_sha256: str,
    expected_runtime_bundle_sha256: str,
    expected_input_bytes: bytes,
    expected_scientific_output_bytes: bytes | None,
) -> ContinuationRunReceipt:
    """Verify canonical receipt bytes against independently supplied trust roots.

    This authenticates receipt topology and exact artifact bindings. It does not replace the
    publication gate's required fresh execution of the independently pinned checker.
    """

    if type(receipt_bytes) is not bytes:
        raise TypeError("continuation run receipt must be exact bytes")
    external_receipt_sha256 = _require_sha256(
        expected_receipt_sha256,
        "expected_receipt_sha256",
    )
    if _sha256(receipt_bytes) != external_receipt_sha256:
        raise ValueError("continuation run receipt disagrees with its external commitment")
    snapshot = _snapshot_protocol(
        protocol,
        expected_protocol_sha256=expected_protocol_sha256,
        expected_release_id=expected_release_id,
    )
    external_checker_sha256 = _require_sha256(
        expected_checker_source_bundle_sha256,
        "expected_checker_source_bundle_sha256",
    )
    if snapshot.checker_source_bundle_sha256 != external_checker_sha256:
        raise ValueError("continuation checker bundle disagrees with its external commitment")
    external_runtime_sha256 = _require_sha256(
        expected_runtime_bundle_sha256,
        "expected_runtime_bundle_sha256",
    )
    if snapshot.runtime_bundle_sha256 != external_runtime_sha256:
        raise ValueError("continuation runtime bundle disagrees with its external commitment")
    if type(expected_input_bytes) is not bytes:
        raise TypeError("expected continuation input must be exact bytes")
    input_sha256, state_sha256 = _parse_input_binding(
        expected_input_bytes,
        input_schema=snapshot.input_schema,
        input_byte_limit=snapshot.input_byte_limit,
    )

    document = require_exact_keys(
        parse_canonical_json_bytes(receipt_bytes, name="continuation run receipt"),
        _RUN_RECEIPT_FIELDS,
        "continuation run receipt",
    )
    if document["schema"] != CONTINUATION_RUN_RECEIPT_SCHEMA:
        raise ValueError("continuation run receipt has the wrong schema")
    if document["schema_version"] != CONTINUATION_RUN_RECEIPT_SCHEMA_VERSION:
        raise ValueError("continuation run receipt has the wrong schema version")
    if document["environment"] != EMPTY_EXEC_ENVIRONMENT:
        raise ValueError("continuation run receipt has the wrong execution environment")
    expected_scalars: dict[str, object] = {
        "release_id": snapshot.release_id,
        "protocol_sha256": snapshot.protocol_sha256,
        "checker_source_bundle_sha256": snapshot.checker_source_bundle_sha256,
        "runtime_bundle_sha256": snapshot.runtime_bundle_sha256,
        "checker_zipapp_sha256": snapshot.checker_zipapp_sha256,
        "runtime_python_sha256": snapshot.runtime_python_sha256,
        "input_schema": snapshot.input_schema,
        "output_schema": snapshot.output_schema,
        "watchdog_seconds": snapshot.watchdog_seconds,
        "input_byte_limit": snapshot.input_byte_limit,
        "output_byte_limit": snapshot.output_byte_limit,
        "stdout_byte_limit": snapshot.stdout_byte_limit,
        "stderr_byte_limit": snapshot.stderr_byte_limit,
        "retained_bundle_byte_limit": snapshot.retained_bundle_byte_limit,
        "zip_member_limit": snapshot.zip_member_limit,
        "zip_uncompressed_byte_limit": snapshot.zip_uncompressed_byte_limit,
        "path_depth_limit": snapshot.path_depth_limit,
        "input_sha256": input_sha256,
        "input_byte_count": len(expected_input_bytes),
    }
    for name, expected in expected_scalars.items():
        if type(document[name]) is not type(expected) or document[name] != expected:
            raise ValueError(f"continuation run receipt has the wrong {name}")

    primary = _parse_invocation_receipt(document["primary"])
    raw_publication = document["publication_rerun"]
    publication = None if raw_publication is None else _parse_invocation_receipt(raw_publication)
    receipt = _make_run_receipt(
        release_id=_require_text(document["release_id"], "release_id"),
        protocol_sha256=_require_sha256(document["protocol_sha256"], "protocol_sha256"),
        checker_source_bundle_sha256=_require_sha256(
            document["checker_source_bundle_sha256"],
            "checker_source_bundle_sha256",
        ),
        runtime_bundle_sha256=_require_sha256(
            document["runtime_bundle_sha256"],
            "runtime_bundle_sha256",
        ),
        checker_zipapp_sha256=_require_sha256(
            document["checker_zipapp_sha256"],
            "checker_zipapp_sha256",
        ),
        runtime_python_sha256=_require_sha256(
            document["runtime_python_sha256"],
            "runtime_python_sha256",
        ),
        input_schema=_require_text(document["input_schema"], "input_schema"),
        output_schema=_require_text(document["output_schema"], "output_schema"),
        watchdog_seconds=_require_positive_int(
            document["watchdog_seconds"],
            "watchdog_seconds",
        ),
        input_byte_limit=_require_positive_int(
            document["input_byte_limit"],
            "input_byte_limit",
        ),
        output_byte_limit=_require_positive_int(
            document["output_byte_limit"],
            "output_byte_limit",
        ),
        stdout_byte_limit=_require_positive_int(
            document["stdout_byte_limit"],
            "stdout_byte_limit",
        ),
        stderr_byte_limit=_require_positive_int(
            document["stderr_byte_limit"],
            "stderr_byte_limit",
        ),
        retained_bundle_byte_limit=_require_positive_int(
            document["retained_bundle_byte_limit"],
            "retained_bundle_byte_limit",
        ),
        zip_member_limit=_require_positive_int(
            document["zip_member_limit"],
            "zip_member_limit",
        ),
        zip_uncompressed_byte_limit=_require_positive_int(
            document["zip_uncompressed_byte_limit"],
            "zip_uncompressed_byte_limit",
        ),
        path_depth_limit=_require_positive_int(
            document["path_depth_limit"],
            "path_depth_limit",
        ),
        input_sha256=_require_sha256(document["input_sha256"], "input_sha256"),
        input_byte_count=_require_nonnegative_int(
            document["input_byte_count"],
            "input_byte_count",
        ),
        primary=primary,
        publication_rerun=publication,
        publication_output_match=_optional_bool(
            document["publication_output_match"],
            "publication_output_match",
        ),
        status=_require_text(document["status"], "status"),
        failure_classification=_optional_text(
            document["failure_classification"],
            "failure_classification",
        ),
        scientific_output_sha256=_optional_sha256(
            document["scientific_output_sha256"],
            "scientific_output_sha256",
        ),
    )
    if receipt.canonical_bytes != receipt_bytes:
        raise ValueError("continuation run receipt fields do not round-trip exactly")
    if receipt.status == "accepted":
        if type(expected_scientific_output_bytes) is not bytes:
            raise TypeError("accepted receipt requires independently supplied output bytes")
        if len(expected_scientific_output_bytes) > snapshot.output_byte_limit:
            raise ValueError("expected continuation output exceeds the runner byte limit")
        expected_output_byte_count = len(expected_scientific_output_bytes)
        if (
            receipt.primary.output_byte_count != expected_output_byte_count
            or receipt.publication_rerun is None
            or receipt.publication_rerun.output_byte_count != expected_output_byte_count
        ):
            raise ValueError(
                "accepted receipt output byte counts disagree with independently supplied "
                "scientific output"
            )
        _validate_output_envelope(
            expected_scientific_output_bytes,
            output_schema=snapshot.output_schema,
            input_sha256=input_sha256,
            state_sha256=state_sha256,
        )
        if _sha256(expected_scientific_output_bytes) != receipt.scientific_output_sha256:
            raise ValueError("scientific output disagrees with the verified receipt")
    elif expected_scientific_output_bytes is not None:
        raise ValueError("failed receipt cannot bind scientific output bytes")
    return receipt


__all__ = [
    "CONTINUATION_RUN_RECEIPT_SCHEMA",
    "CONTINUATION_RUN_RECEIPT_SCHEMA_VERSION",
    "MAX_CONTINUATION_INPUT_BYTES",
    "MAX_CONTINUATION_OUTPUT_BYTES",
    "MAX_CONTINUATION_STDERR_BYTES",
    "MAX_CONTINUATION_STDOUT_BYTES",
    "ContinuationInvocationReceipt",
    "ContinuationRunReceipt",
    "ContinuationRunResult",
    "run_verified_continuation",
    "verify_continuation_run_receipt",
]
