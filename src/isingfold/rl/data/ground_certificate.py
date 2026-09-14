"""Executable, externally pinned ground-certificate preflight.

Integrity-only evidence manifests prove that certificate bytes were not changed; they do not prove
the claimed Ising ground energy. This module closes that gap through a standalone subprocess
boundary. A verifier whose executable, source, and environment artifacts are independently pinned
receives only canonical public-instance data, the claimed evaluator target, and the exact
certificate bytes. No EmbedBench generator code is imported.

The verifier protocol is deliberately narrow: one canonical JSON request on standard input, one
canonical JSON result on standard output, no arguments, no stderr, and a scrubbed deterministic
environment. Schema v1 supports ``planted_proof`` and ``exact_enumeration`` only.
``certified_optimal`` remains fail-closed until a checker-specific protocol is registered.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
import platform
import signal
import shutil
import struct
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.prepared import (
    PreparedTask,
    TargetAccessReceipt,
    load_prepared_partition,
    load_prepared_tasks,
)
from isingfold.rl.data.quality_attestation import (
    EVIDENCE_KIND_BY_REFERENCE_STATUS,
    PUBLISHER_ATTESTATION_VERSION_V2,
    QUALITY_EVIDENCE_SCHEMA,
    PublisherAttestation,
    QualityAttestationPin,
    load_publisher_attestation,
    validate_quality_evidence,
)

GROUND_CERTIFICATE_PREFLIGHT_SCHEMA = "isingfold.ground-certificate-preflight"
GROUND_CERTIFICATE_PREFLIGHT_VERSION = 1
GROUND_CERTIFICATE_REQUEST_SCHEMA = "isingfold.ground-certificate-verifier-request"
GROUND_CERTIFICATE_RESULT_SCHEMA = "isingfold.ground-certificate-verifier-result"
GROUND_CERTIFICATE_PROTOCOL = "isingfold-ground-certificate-subprocess-v1"
GROUND_CERTIFICATE_PROTOCOL_V2 = "isingfold-ground-certificate-isolated-runtime-v2"
GROUND_CERTIFICATE_ROOT_SCHEMA = "isingfold.ground-certificate-root"
GROUND_CERTIFICATE_ROOT_VERSION = 1
GROUND_CERTIFICATE_PARTITION_SCHEMA = "isingfold.ground-certificate-partition"
GROUND_CERTIFICATE_PARTITION_VERSION = 1
VERIFIER_BUILD_ATTESTATION_SCHEMA = "isingfold.verifier-build-attestation"
VERIFIER_BUILD_ATTESTATION_VERSION = 1
GLOBAL_QUALITY_AUTHORITY_SCHEMA = "isingfold.global-quality-authority"
PARTITION_QUALITY_AUTHORITY_SCHEMA = "isingfold.partition-quality-authority"
SUPPORTED_REFERENCE_STATUSES = frozenset({"exact_enumeration", "planted_proof"})
_HEX = frozenset("0123456789abcdef")
_MAX_CERTIFICATE_BYTES = 64 * 1024 * 1024
_MAX_RESULT_BYTES = 64 * 1024
_DESIGN_CENSUS_FIELDS = (
    "application_family",
    "problem_origin",
    "host_family",
    "fault_status",
    "distribution_regime",
    "calibration_status",
    "embedding_difficulty",
    "sampling_difficulty",
    "decision_difficulty",
)
_SCRUBBED_ENVIRONMENT = {
    "LANG": "C",
    "LC_ALL": "C",
    "PYTHONHASHSEED": "0",
    "TZ": "UTC",
}


class GroundCertificateError(ValueError):
    """A verifier pin, protocol execution, or ground-certificate result is invalid."""


@dataclass(frozen=True)
class GroundCertificateVerifierPin:
    """Out-of-band identities for one closed standalone verifier."""

    executable_path: str | Path
    expected_executable_sha256: str
    source_path: str | Path
    expected_source_sha256: str
    environment_path: str | Path
    expected_environment_sha256: str
    expected_name: str
    expected_version: str
    execution_mode: str = "test-only-host"
    runtime_path: str | Path | None = None
    expected_runtime_sha256: str | None = None
    build_attestation_path: str | Path | None = None
    expected_build_attestation_sha256: str | None = None


@dataclass(frozen=True)
class _VerifierExecution:
    mode: str
    environment_path: Path
    runtime_path: Path | None


@dataclass(frozen=True)
class GroundCertificateRootReceipt:
    """Authenticated public root; its record contains no target rows or energies."""

    path: Path
    sha256: str
    record_digest: str
    target_authority_record_digest: str
    verifier_identity_digest: str
    record: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        raw = canonical_json_bytes(self.record) + b"\n"
        record = _strict_object(raw, "ground-certificate root receipt")
        digest = _verify_record(record, "ground-certificate root receipt")
        if digest != self.record_digest or hashlib.sha256(raw).hexdigest() != self.sha256:
            raise GroundCertificateError("ground-certificate root dataclass identity mismatch")
        return record


@dataclass(frozen=True)
class GroundCertificatePartitionLoad:
    """One explicitly opened and root-authenticated certificate partition."""

    root: GroundCertificateRootReceipt
    partition: str
    tasks: tuple[PreparedTask, ...]
    target_access: TargetAccessReceipt
    receipt: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        raw = canonical_json_bytes(self.receipt) + b"\n"
        record = _strict_object(raw, "ground-certificate partition receipt")
        _verify_record(record, "ground-certificate partition receipt")
        return record


@dataclass(frozen=True)
class GlobalQualityAuthority:
    """Cross-partition authority identity safe to compare at every lifecycle stage."""

    publication_id: str
    publisher_id: str
    publisher_attestation_record_digest: str
    target_authority_record_digest: str
    ground_root_receipt_sha256: str
    ground_root_record_digest: str
    verifier_identity_digest: str
    record_digest: str

    def as_dict(self) -> dict[str, Any]:
        for name, digest in (
            ("publisher attestation", self.publisher_attestation_record_digest),
            ("target authority", self.target_authority_record_digest),
            ("ground root file", self.ground_root_receipt_sha256),
            ("ground root record", self.ground_root_record_digest),
            ("verifier identity", self.verifier_identity_digest),
        ):
            _sha256(digest, f"global quality-authority {name} digest")
        _text(self.publication_id, "global quality-authority publication ID")
        _text(self.publisher_id, "global quality-authority publisher ID")
        payload = {
            "ground_root": {
                "receipt_sha256": self.ground_root_receipt_sha256,
                "record_digest": self.ground_root_record_digest,
                "verifier_identity_digest": self.verifier_identity_digest,
            },
            "publication_id": self.publication_id,
            "publisher_attestation_record_digest": (
                self.publisher_attestation_record_digest
            ),
            "publisher_id": self.publisher_id,
            "schema": GLOBAL_QUALITY_AUTHORITY_SCHEMA,
            "schema_version": 1,
            "target_authority_record_digest": self.target_authority_record_digest,
        }
        observed = content_digest(payload)
        if observed != self.record_digest:
            raise GroundCertificateError("global quality-authority digest mismatch")
        return {**payload, "record_digest": observed}


@dataclass(frozen=True)
class PartitionQualityAuthority:
    """One partition view; train and test views must never be equated."""

    name: str
    target_access_record_digest: str
    evidence_manifest_record_digest: str
    evidence_manifest_sha256: str
    target_set_digest: str
    target_count: int
    ground_partition_receipt_record_digest: str
    ground_partition_receipt_sha256: str
    accepted_count: int
    instance_set_digest: str
    record_digest: str

    def as_dict(self) -> dict[str, Any]:
        if self.name not in {"train", "val", "test"}:
            raise GroundCertificateError("partition quality-authority name is invalid")
        for name, digest in (
            ("target access", self.target_access_record_digest),
            ("evidence manifest record", self.evidence_manifest_record_digest),
            ("evidence manifest file", self.evidence_manifest_sha256),
            ("target set", self.target_set_digest),
            ("ground partition record", self.ground_partition_receipt_record_digest),
            ("ground partition file", self.ground_partition_receipt_sha256),
            ("instance set", self.instance_set_digest),
        ):
            _sha256(digest, f"partition quality-authority {name} digest")
        if (
            isinstance(self.target_count, bool)
            or not isinstance(self.target_count, int)
            or self.target_count <= 0
            or self.accepted_count != self.target_count
        ):
            raise GroundCertificateError(
                "partition quality authority must prove every positive target count"
            )
        payload = {
            "evidence_manifest_record_digest": self.evidence_manifest_record_digest,
            "evidence_manifest_sha256": self.evidence_manifest_sha256,
            "ground_partition": {
                "accepted_count": self.accepted_count,
                "instance_set_digest": self.instance_set_digest,
                "receipt_record_digest": self.ground_partition_receipt_record_digest,
                "receipt_sha256": self.ground_partition_receipt_sha256,
            },
            "name": self.name,
            "schema": PARTITION_QUALITY_AUTHORITY_SCHEMA,
            "schema_version": 1,
            "target_access_record_digest": self.target_access_record_digest,
            "target_count": self.target_count,
            "target_set_digest": self.target_set_digest,
        }
        observed = content_digest(payload)
        if observed != self.record_digest:
            raise GroundCertificateError("partition quality-authority digest mismatch")
        return {**payload, "record_digest": observed}


def _strict_object(raw: bytes, name: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise GroundCertificateError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise GroundCertificateError(f"{name} contains non-finite number {token}")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GroundCertificateError(f"{name} is invalid JSON") from exc
    if not isinstance(value, dict):
        raise GroundCertificateError(f"{name} must be a JSON object")
    _check_finite(value, name)
    return value


def _check_finite(value: object, name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise GroundCertificateError(f"{name} contains a non-finite number")
    if isinstance(value, Mapping):
        for item in value.values():
            _check_finite(item, name)
    elif isinstance(value, list):
        for item in value:
            _check_finite(item, name)


def _read(path: Path, name: str) -> bytes:
    try:
        if not path.is_file():
            raise GroundCertificateError(f"{name} is missing or is not a regular file")
        return path.read_bytes()
    except OSError as exc:
        raise GroundCertificateError(f"cannot read {name}: {exc}") from exc


def _read_jsonl(path: Path, name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(_read(path, name).splitlines(), start=1):
        if not line.strip():
            raise GroundCertificateError(f"{name} line {line_number} is blank")
        rows.append(_strict_object(line, f"{name} line {line_number}"))
    if not rows:
        raise GroundCertificateError(f"{name} must be nonempty")
    return rows


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise GroundCertificateError(
            f"{name} schema differs: missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise GroundCertificateError(f"{name} must be nonempty text")
    return value


def _sha256(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise GroundCertificateError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _verify_record(record: Mapping[str, object], name: str) -> str:
    digest = _sha256(record.get("record_digest"), f"{name} record digest")
    payload = {key: value for key, value in record.items() if key != "record_digest"}
    if not hmac.compare_digest(digest, content_digest(payload)):
        raise GroundCertificateError(f"{name} record digest mismatch")
    return digest


def _pinned_file(
    path_value: str | Path,
    expected_value: str,
    name: str,
) -> tuple[Path, bytes, str]:
    expected = _sha256(expected_value, f"expected {name} SHA-256")
    try:
        path = Path(path_value).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise GroundCertificateError(f"{name} is missing") from exc
    raw = _read(path, name)
    observed = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(observed, expected):
        raise GroundCertificateError(f"{name} differs from its out-of-band pin")
    return path, raw, observed


def _elf_has_program_interpreter(raw: bytes) -> bool:
    """Parse the ELF program table and report whether PT_INTERP is present."""

    if len(raw) < 52 or raw[:4] != b"\x7fELF":
        raise GroundCertificateError("static verifier must be an ELF executable")
    elf_class = raw[4]
    data_encoding = raw[5]
    if elf_class not in {1, 2} or data_encoding not in {1, 2}:
        raise GroundCertificateError("static verifier has an unsupported ELF header")
    byte_order = "<" if data_encoding == 1 else ">"
    if elf_class == 1:
        if len(raw) < 52:
            raise GroundCertificateError("static verifier ELF header is truncated")
        program_offset = struct.unpack_from(f"{byte_order}I", raw, 28)[0]
        entry_size = struct.unpack_from(f"{byte_order}H", raw, 42)[0]
        entry_count = struct.unpack_from(f"{byte_order}H", raw, 44)[0]
    else:
        if len(raw) < 64:
            raise GroundCertificateError("static verifier ELF header is truncated")
        program_offset = struct.unpack_from(f"{byte_order}Q", raw, 32)[0]
        entry_size = struct.unpack_from(f"{byte_order}H", raw, 54)[0]
        entry_count = struct.unpack_from(f"{byte_order}H", raw, 56)[0]
    if entry_count <= 0 or entry_size < 4:
        raise GroundCertificateError("static verifier ELF program table is missing")
    table_end = program_offset + entry_size * entry_count
    if program_offset <= 0 or table_end > len(raw):
        raise GroundCertificateError("static verifier ELF program table is truncated")
    return any(
        struct.unpack_from(f"{byte_order}I", raw, program_offset + index * entry_size)[0] == 3
        for index in range(entry_count)
    )


def _load_build_attestation(
    pin: GroundCertificateVerifierPin,
    *,
    executable_sha256: str,
    source_sha256: str,
) -> tuple[dict[str, object], Path, str]:
    if pin.build_attestation_path is None or pin.expected_build_attestation_sha256 is None:
        raise GroundCertificateError(
            "publication verifier requires a pinned reproducible-build attestation"
        )
    path, raw, file_sha256 = _pinned_file(
        pin.build_attestation_path,
        pin.expected_build_attestation_sha256,
        "verifier build attestation",
    )
    record = _strict_object(raw, "verifier build attestation")
    if raw != canonical_json_bytes(record) + b"\n":
        raise GroundCertificateError("verifier build attestation is not canonical JSON")
    _exact_keys(
        record,
        {
            "build_recipe_sha256",
            "executable_sha256",
            "independent_reproduction_count",
            "record_digest",
            "reproduced_executable_sha256",
            "schema",
            "schema_version",
            "source_sha256",
            "toolchain_image_sha256",
        },
        "verifier build attestation",
    )
    _verify_record(record, "verifier build attestation")
    if (
        record["schema"] != VERIFIER_BUILD_ATTESTATION_SCHEMA
        or record["schema_version"] != VERIFIER_BUILD_ATTESTATION_VERSION
    ):
        raise GroundCertificateError("unsupported verifier build attestation")
    for name in (
        "build_recipe_sha256",
        "executable_sha256",
        "reproduced_executable_sha256",
        "source_sha256",
        "toolchain_image_sha256",
    ):
        _sha256(record[name], f"verifier build {name}")
    count = record["independent_reproduction_count"]
    if isinstance(count, bool) or not isinstance(count, int) or count < 2:
        raise GroundCertificateError(
            "verifier build needs at least two independent byte-identical reproductions"
        )
    if (
        record["executable_sha256"] != executable_sha256
        or record["reproduced_executable_sha256"] != executable_sha256
        or record["source_sha256"] != source_sha256
    ):
        raise GroundCertificateError(
            "verifier build attestation does not bind the pinned source and executable"
        )
    return record, path, file_sha256


def _safe_relative_path(value: object, name: str) -> str:
    text = _text(value, name)
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or "\\" in text
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise GroundCertificateError(f"{name} must be a safe relative path")
    return text


def _bound_file(root: Path, relative: str, name: str) -> Path:
    root = root.resolve()
    try:
        path = root.joinpath(*PurePosixPath(relative).parts).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise GroundCertificateError(f"{name} is missing") from exc
    if not path.is_relative_to(root) or not path.is_file():
        raise GroundCertificateError(f"{name} escapes its authenticated evidence root")
    return path


def _verifier_identity(
    pin: GroundCertificateVerifierPin,
    *,
    publication: bool = False,
) -> tuple[
    dict[str, object],
    bytes,
    tuple[tuple[Path, str], ...],
    _VerifierExecution,
]:
    if not isinstance(pin, GroundCertificateVerifierPin):
        raise TypeError("verifier_pin must be GroundCertificateVerifierPin")
    executable_path, executable, executable_sha256 = _pinned_file(
        pin.executable_path,
        pin.expected_executable_sha256,
        "ground verifier executable",
    )
    if not os.access(executable_path, os.X_OK):
        raise GroundCertificateError("ground verifier executable is not executable")
    source_path, _source, source_sha256 = _pinned_file(
        pin.source_path,
        pin.expected_source_sha256,
        "ground verifier source",
    )
    environment_path, environment, environment_sha256 = _pinned_file(
        pin.environment_path,
        pin.expected_environment_sha256,
        "ground verifier environment",
    )
    name = _text(pin.expected_name, "expected ground verifier name")
    version = _text(pin.expected_version, "expected ground verifier version")
    mode = _text(pin.execution_mode, "ground verifier execution mode")
    base_paths: tuple[tuple[Path, str], ...] = (
        (executable_path, executable_sha256),
        (source_path, source_sha256),
        (environment_path, environment_sha256),
    )
    if mode == "test-only-host":
        if publication:
            raise GroundCertificateError(
                "test-only host verifier cannot authorize a publication receipt"
            )
        identity: dict[str, object] = {
            "environment_sha256": environment_sha256,
            "executable_sha256": executable_sha256,
            "name": name,
            "protocol": GROUND_CERTIFICATE_PROTOCOL,
            "source_sha256": source_sha256,
            "version": version,
        }
        return (
            identity,
            executable,
            base_paths,
            _VerifierExecution(
                mode=mode,
                environment_path=environment_path,
                runtime_path=None,
            ),
        )
    if not publication:
        raise GroundCertificateError(
            "isolated publication verifier modes require the partitioned v2 protocol"
        )
    if mode not in {"static-elf", "apptainer"}:
        raise GroundCertificateError(
            "publication verifier execution mode must be static-elf or apptainer"
        )
    if not executable.startswith(b"\x7fELF"):
        raise GroundCertificateError(
            "publication verifier must be a compiled ELF binary, not a script"
        )
    build, build_path, build_sha256 = _load_build_attestation(
        pin,
        executable_sha256=executable_sha256,
        source_sha256=source_sha256,
    )
    runtime_path: Path | None = None
    runtime_sha256: str | None = None
    pinned_paths = (*base_paths, (build_path, build_sha256))
    if mode == "static-elf":
        if _elf_has_program_interpreter(executable):
            raise GroundCertificateError(
                "static verifier contains PT_INTERP and depends on an unpinned loader"
            )
        environment_record = _strict_object(environment, "static runtime environment")
        if environment != canonical_json_bytes(environment_record) + b"\n":
            raise GroundCertificateError("static runtime environment is not canonical JSON")
        _exact_keys(
            environment_record,
            {
                "kernel_release",
                "kernel_system",
                "machine",
                "record_digest",
                "schema",
                "schema_version",
            },
            "static runtime environment",
        )
        _verify_record(environment_record, "static runtime environment")
        if (
            environment_record["schema"] != "isingfold.static-runtime-environment"
            or environment_record["schema_version"] != 1
        ):
            raise GroundCertificateError("unsupported static runtime environment")
        expected_host = {
            "kernel_release": platform.release(),
            "kernel_system": platform.system(),
            "machine": platform.machine(),
        }
        if any(environment_record[key] != value for key, value in expected_host.items()):
            raise GroundCertificateError(
                "static runtime environment does not identify the executing host"
            )
    else:
        if pin.runtime_path is None or pin.expected_runtime_sha256 is None:
            raise GroundCertificateError(
                "apptainer publication requires a pinned runtime executable"
            )
        runtime_path, runtime, runtime_sha256 = _pinned_file(
            pin.runtime_path,
            pin.expected_runtime_sha256,
            "Apptainer runtime",
        )
        if not os.access(runtime_path, os.X_OK):
            raise GroundCertificateError("Apptainer runtime is not executable")
        if not runtime.startswith(b"\x7fELF"):
            raise GroundCertificateError(
                "Apptainer runtime must be a pinned compiled ELF binary"
            )
        if not environment:
            raise GroundCertificateError("Apptainer image must be nonempty")
        pinned_paths = (*pinned_paths, (runtime_path, runtime_sha256))

    identity = {
        "build_attestation_record_digest": build["record_digest"],
        "build_attestation_sha256": build_sha256,
        "environment_sha256": environment_sha256,
        "execution_mode": mode,
        "executable_sha256": executable_sha256,
        "name": name,
        "protocol": GROUND_CERTIFICATE_PROTOCOL_V2,
        "runtime_sha256": runtime_sha256,
        "source_sha256": source_sha256,
        "version": version,
    }
    return (
        identity,
        executable,
        pinned_paths,
        _VerifierExecution(
            mode=mode,
            environment_path=environment_path,
            runtime_path=runtime_path,
        ),
    )


def _validate_timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GroundCertificateError("verifier timeout must be numeric")
    timeout = float(value)
    if not math.isfinite(timeout) or not 0.0 < timeout <= 3600.0:
        raise GroundCertificateError("verifier timeout must be in (0, 3600] seconds")
    return timeout


def _run_verifier(
    executable: bytes,
    request: Mapping[str, object],
    *,
    verifier_identity: Mapping[str, object],
    execution: _VerifierExecution,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str]:
    request_raw = canonical_json_bytes(request) + b"\n"
    with tempfile.TemporaryDirectory(prefix="isingfold-ground-verifier-") as directory:
        executable_path = Path(directory) / "verifier"
        executable_path.write_bytes(executable)
        executable_path.chmod(0o500)
        if execution.mode in {"test-only-host", "static-elf"}:
            command = [str(executable_path)]
            environment = dict(_SCRUBBED_ENVIRONMENT)
        elif execution.mode == "apptainer":
            if execution.runtime_path is None:
                raise RuntimeError("Apptainer verifier execution lost its runtime")
            command = [
                str(execution.runtime_path),
                "exec",
                "--containall",
                "--cleanenv",
                "--no-home",
                "--bind",
                f"{executable_path}:/isingfold/verifier:ro",
                str(execution.environment_path),
                "/isingfold/verifier",
            ]
            environment = {
                **_SCRUBBED_ENVIRONMENT,
                "APPTAINER_CACHEDIR": directory,
                "TMPDIR": directory,
            }
        else:
            raise RuntimeError("unsupported validated verifier execution mode")
        try:
            process = subprocess.Popen(  # noqa: S603 - executable bytes are externally pinned.
                command,
                cwd=directory,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as exc:
            raise GroundCertificateError("cannot execute the pinned ground verifier") from exc
        try:
            stdout, stderr = process.communicate(request_raw, timeout=timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise GroundCertificateError("ground verifier exceeded its pinned timeout") from exc
    if process.returncode != 0:
        raise GroundCertificateError(
            f"ground verifier exited with status {process.returncode}"
        )
    if stderr:
        raise GroundCertificateError("ground verifier emitted forbidden stderr output")
    if not stdout or len(stdout) > _MAX_RESULT_BYTES:
        raise GroundCertificateError("ground verifier result is empty or exceeds its size cap")
    result = _strict_object(stdout, "ground verifier result")
    if stdout != canonical_json_bytes(result) + b"\n":
        raise GroundCertificateError("ground verifier result is not canonical JSON")
    _exact_keys(
        result,
        {
            "accepted",
            "claimed_reference_energy",
            "instance_id",
            "reason_code",
            "record_digest",
            "request_digest",
            "schema",
            "schema_version",
            "verifier_identity",
        },
        "ground verifier result",
    )
    _verify_record(result, "ground verifier result")
    if (
        result["schema"] != GROUND_CERTIFICATE_RESULT_SCHEMA
        or result["schema_version"] != 1
    ):
        raise GroundCertificateError("unsupported ground verifier result schema")
    if result["verifier_identity"] != dict(verifier_identity):
        raise GroundCertificateError("ground verifier result has the wrong verifier identity")
    if result["request_digest"] != request["record_digest"]:
        raise GroundCertificateError("ground verifier result has the wrong request identity")
    public_instance = request["public_instance"]
    if not isinstance(public_instance, Mapping):
        raise RuntimeError("ground verifier request lost its public instance")
    if result["instance_id"] != public_instance["instance_id"]:
        raise GroundCertificateError("ground verifier result has the wrong instance identity")
    if result["claimed_reference_energy"] != request["claimed_reference_energy"]:
        raise GroundCertificateError("ground verifier result changed the claimed energy")
    if not isinstance(result["accepted"], bool):
        raise GroundCertificateError("ground verifier acceptance must be Boolean")
    reason = _text(result["reason_code"], "ground verifier reason code")
    if result["accepted"] and reason != "accepted":
        raise GroundCertificateError("accepted ground verifier result has the wrong reason code")
    if not result["accepted"]:
        raise GroundCertificateError(
            f"ground verifier rejected {result['instance_id']!r}: {reason}"
        )
    return result, hashlib.sha256(stdout).hexdigest()


def _evidence_rows(
    attestation: PublisherAttestation,
    *,
    target_by_instance: Mapping[str, Mapping[str, object]],
    verifier_identity: Mapping[str, object],
    partition: str | None = None,
) -> tuple[dict[str, tuple[Mapping[str, object], str, bytes]], bytes]:
    if attestation.schema_version == 2:
        if partition not in {"train", "val", "test"}:
            raise GroundCertificateError(
                "partitioned quality evidence requires one explicit partition"
            )
        identity = attestation.evidence_manifests[partition]
        evidence_relative = _safe_relative_path(
            identity.path,
            "quality evidence manifest path",
        )
        expected_evidence_sha256 = identity.sha256
        expected_manifest_keys = {
            "evaluator_targets_sha256",
            "evidence",
            "partition",
            "record_digest",
            "schema",
            "schema_version",
            "target_count",
            "target_set_digest",
        }
        expected_manifest_version = 2
    else:
        if partition is not None:
            raise GroundCertificateError(
                "legacy quality evidence has no partition identity"
            )
        evidence_relative = _safe_relative_path(
            attestation.evidence_manifest_path,
            "quality evidence manifest path",
        )
        if attestation.evidence_manifest_sha256 is None:
            raise RuntimeError("legacy attestation lost its evidence-manifest digest")
        expected_evidence_sha256 = attestation.evidence_manifest_sha256
        expected_manifest_keys = {
            "evaluator_targets_sha256",
            "evidence",
            "record_digest",
            "schema",
            "schema_version",
            "target_count",
            "target_set_digest",
        }
        expected_manifest_version = 1
    evidence_path = _bound_file(
        attestation.path.parent,
        evidence_relative,
        "quality evidence manifest",
    )
    evidence_raw = _read(evidence_path, "quality evidence manifest")
    if not hmac.compare_digest(
        hashlib.sha256(evidence_raw).hexdigest(),
        expected_evidence_sha256,
    ):
        raise GroundCertificateError("quality evidence manifest changed after authentication")
    manifest = _strict_object(evidence_raw, "quality evidence manifest")
    _exact_keys(manifest, expected_manifest_keys, "quality evidence manifest")
    _verify_record(manifest, "quality evidence manifest")
    if (
        manifest["schema"] != QUALITY_EVIDENCE_SCHEMA
        or manifest["schema_version"] != expected_manifest_version
        or (partition is not None and manifest["partition"] != partition)
    ):
        raise GroundCertificateError("unsupported quality evidence manifest schema")
    raw_rows = manifest["evidence"]
    if not isinstance(raw_rows, list):
        raise GroundCertificateError("quality evidence rows must be a list")
    rows: dict[str, tuple[Mapping[str, object], str, bytes]] = {}
    expected_row_keys = {
        "artifact_path",
        "artifact_sha256",
        "certificate_digest",
        "evidence_kind",
        "instance_id",
        "target_record_digest",
        "verifier",
    }
    for row in raw_rows:
        if not isinstance(row, Mapping):
            raise GroundCertificateError("quality evidence row must be an object")
        _exact_keys(row, expected_row_keys, "quality evidence row")
        instance_id = _text(row["instance_id"], "quality evidence instance ID")
        if instance_id in rows:
            raise GroundCertificateError("quality evidence repeats an instance ID")
        target = target_by_instance.get(instance_id)
        if target is None:
            raise GroundCertificateError("quality evidence references an unknown target")
        if row["target_record_digest"] != target["record_digest"]:
            raise GroundCertificateError("quality evidence references the wrong target")
        artifact_sha256 = _sha256(row["artifact_sha256"], "certificate artifact SHA-256")
        if (
            row["certificate_digest"] != target["certificate_digest"]
            or artifact_sha256 != target["certificate_digest"]
        ):
            raise GroundCertificateError("quality evidence certificate identity mismatch")
        status = str(target["reference_status"])
        if row["evidence_kind"] != EVIDENCE_KIND_BY_REFERENCE_STATUS[status]:
            raise GroundCertificateError("quality evidence kind differs from reference status")
        declared_verifier = row["verifier"]
        if not isinstance(declared_verifier, Mapping):
            raise GroundCertificateError("quality evidence verifier must be an object")
        _exact_keys(
            declared_verifier,
            {"implementation_sha256", "name", "version"},
            "quality evidence verifier",
        )
        if (
            declared_verifier["implementation_sha256"] != verifier_identity["source_sha256"]
            or declared_verifier["name"] != verifier_identity["name"]
            or declared_verifier["version"] != verifier_identity["version"]
        ):
            raise GroundCertificateError(
                "quality evidence verifier differs from the out-of-band verifier pin"
            )
        artifact_relative = _safe_relative_path(
            row["artifact_path"],
            "quality evidence artifact path",
        )
        artifact_path = _bound_file(
            evidence_path.parent,
            artifact_relative,
            "quality evidence artifact",
        )
        artifact = _read(artifact_path, "quality evidence artifact")
        if len(artifact) > _MAX_CERTIFICATE_BYTES:
            raise GroundCertificateError("quality evidence artifact exceeds its size cap")
        if not hmac.compare_digest(hashlib.sha256(artifact).hexdigest(), artifact_sha256):
            raise GroundCertificateError("quality evidence artifact changed after authentication")
        rows[instance_id] = (row, artifact_relative, artifact)
    if set(rows) != set(target_by_instance):
        raise GroundCertificateError("quality evidence does not exactly cover every target")
    return rows, evidence_raw


def _task_by_instance(tasks: Sequence[PreparedTask]) -> dict[str, PreparedTask]:
    grouped: dict[str, list[PreparedTask]] = {}
    for task in tasks:
        grouped.setdefault(task.instance_id, []).append(task)
    result: dict[str, PreparedTask] = {}
    for instance_id, rows in grouped.items():
        identities = {
            (
                row.partition,
                None
                if row.design_condition is None
                else row.design_condition.registry_row_digest,
            )
            for row in rows
        }
        if len(identities) != 1 or rows[0].design_condition is None:
            raise GroundCertificateError(
                "one public instance has ambiguous or missing scientific design identity"
            )
        result[instance_id] = rows[0]
    return result


def _validated_receipt_verifier(raw: object) -> dict[str, str]:
    if not isinstance(raw, Mapping):
        raise GroundCertificateError("ground-certificate receipt verifier must be an object")
    _exact_keys(
        raw,
        {
            "environment_sha256",
            "executable_sha256",
            "name",
            "protocol",
            "source_sha256",
            "version",
        },
        "ground-certificate receipt verifier",
    )
    identity = {
        "environment_sha256": _sha256(
            raw["environment_sha256"],
            "receipt verifier environment SHA-256",
        ),
        "executable_sha256": _sha256(
            raw["executable_sha256"],
            "receipt verifier executable SHA-256",
        ),
        "name": _text(raw["name"], "receipt verifier name"),
        "protocol": _text(raw["protocol"], "receipt verifier protocol"),
        "source_sha256": _sha256(
            raw["source_sha256"],
            "receipt verifier source SHA-256",
        ),
        "version": _text(raw["version"], "receipt verifier version"),
    }
    if identity["protocol"] != GROUND_CERTIFICATE_PROTOCOL:
        raise GroundCertificateError("ground-certificate receipt uses another verifier protocol")
    return identity


def _validated_publication_verifier(raw: object) -> dict[str, object]:
    if not isinstance(raw, Mapping):
        raise GroundCertificateError("publication verifier identity must be an object")
    _exact_keys(
        raw,
        {
            "build_attestation_record_digest",
            "build_attestation_sha256",
            "environment_sha256",
            "execution_mode",
            "executable_sha256",
            "name",
            "protocol",
            "runtime_sha256",
            "source_sha256",
            "version",
        },
        "publication verifier identity",
    )
    mode = _text(raw["execution_mode"], "publication verifier execution mode")
    if mode not in {"static-elf", "apptainer"}:
        raise GroundCertificateError("publication receipt uses a non-isolated verifier")
    runtime = raw["runtime_sha256"]
    if mode == "static-elf":
        if runtime is not None:
            raise GroundCertificateError("static verifier unexpectedly binds a runtime launcher")
    else:
        runtime = _sha256(runtime, "publication verifier runtime SHA-256")
    identity: dict[str, object] = {
        "build_attestation_record_digest": _sha256(
            raw["build_attestation_record_digest"],
            "publication verifier build-attestation record digest",
        ),
        "build_attestation_sha256": _sha256(
            raw["build_attestation_sha256"],
            "publication verifier build-attestation SHA-256",
        ),
        "environment_sha256": _sha256(
            raw["environment_sha256"], "publication verifier environment SHA-256"
        ),
        "execution_mode": mode,
        "executable_sha256": _sha256(
            raw["executable_sha256"], "publication verifier executable SHA-256"
        ),
        "name": _text(raw["name"], "publication verifier name"),
        "protocol": _text(raw["protocol"], "publication verifier protocol"),
        "runtime_sha256": runtime,
        "source_sha256": _sha256(
            raw["source_sha256"], "publication verifier source SHA-256"
        ),
        "version": _text(raw["version"], "publication verifier version"),
    }
    if identity["protocol"] != GROUND_CERTIFICATE_PROTOCOL_V2:
        raise GroundCertificateError("publication receipt uses another verifier protocol")
    return identity


def _public_v4_authorities(
    corpus: Path,
    quality_attestation_pin: QualityAttestationPin,
) -> tuple[
    dict[str, Any],
    bytes,
    str,
    str,
    PublisherAttestation,
    bytes,
    str,
]:
    """Read only the public manifest and publisher statement for a v4 corpus."""

    manifest_path = corpus / "manifest.json"
    manifest_raw = _read(manifest_path, "prepared manifest")
    manifest = _strict_object(manifest_raw, "prepared manifest")
    if manifest_raw != canonical_json_bytes(manifest) + b"\n":
        raise GroundCertificateError("prepared manifest is not canonical JSON")
    manifest_record_digest = _verify_record(manifest, "prepared manifest")
    if (
        manifest.get("schema") != "isingfold.prepared-candidate-bank"
        or manifest.get("schema_version") != 4
    ):
        raise GroundCertificateError(
            "partitioned ground publication requires prepared schema v4"
        )
    target_authority = manifest.get("target_authority")
    if not isinstance(target_authority, Mapping):
        raise GroundCertificateError("prepared manifest lacks target authority")
    _exact_keys(
        target_authority,
        {"partitions", "record_digest", "schema", "schema_version", "total_targets"},
        "prepared target authority",
    )
    target_authority_digest = _verify_record(
        target_authority, "prepared target authority"
    )
    if (
        target_authority["schema"] != "isingfold.partitioned-target-authority"
        or target_authority["schema_version"] != 1
    ):
        raise GroundCertificateError("unsupported prepared target authority")
    descriptors = target_authority["partitions"]
    if not isinstance(descriptors, Mapping) or set(descriptors) != {
        "train",
        "val",
        "test",
    }:
        raise GroundCertificateError("prepared target authority partition set is invalid")
    total = 0
    for partition in ("train", "val", "test"):
        descriptor = descriptors[partition]
        if not isinstance(descriptor, Mapping):
            raise GroundCertificateError("prepared target descriptor must be an object")
        _exact_keys(
            descriptor,
            {"path", "records", "sha256", "target_set_digest"},
            "prepared target descriptor",
        )
        if descriptor["path"] != f"targets/{partition}.jsonl":
            raise GroundCertificateError("prepared target descriptor crosses partitions")
        count = descriptor["records"]
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise GroundCertificateError("prepared target descriptor count must be positive")
        total += count
        _sha256(descriptor["sha256"], "prepared target file SHA-256")
        _sha256(descriptor["target_set_digest"], "prepared target-set digest")
    if target_authority["total_targets"] != total:
        raise GroundCertificateError("prepared target-authority total is inconsistent")
    attestation = load_publisher_attestation(
        quality_attestation_pin,
        prepared_manifest_path=manifest_path,
    )
    if attestation.schema_version != PUBLISHER_ATTESTATION_VERSION_V2:
        raise GroundCertificateError(
            "partitioned ground publication requires publisher attestation v2"
        )
    if attestation.target_authority_record_digest != target_authority_digest:
        raise GroundCertificateError("publisher binds another target authority")
    attestation_raw = _read(attestation.path, "publisher attestation")
    attestation_record = _strict_object(attestation_raw, "publisher attestation")
    if _verify_record(attestation_record, "publisher attestation") != attestation.digest:
        raise GroundCertificateError("publisher attestation changed after authentication")
    return (
        manifest,
        manifest_raw,
        manifest_record_digest,
        hashlib.sha256(manifest_raw).hexdigest(),
        attestation,
        attestation_raw,
        hashlib.sha256(attestation_raw).hexdigest(),
    )


def _categorical_counts(
    rows: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[field])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _atomic_publish(path: Path, raw: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"ground-certificate receipt already exists: {path}")
    if not path.parent.is_dir():
        raise FileNotFoundError(f"ground-certificate receipt parent does not exist: {path.parent}")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(
                f"ground-certificate receipt already exists: {path}"
            ) from None
    finally:
        temporary.unlink(missing_ok=True)


def _policy_rows_by_instance(corpus: Path) -> dict[str, Mapping[str, object]]:
    rows = _read_jsonl(corpus / "policy_instances.jsonl", "public policy instances")
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        _verify_record(row, "public policy instance")
        instance_id = _text(row.get("instance_id"), "public policy instance ID")
        if instance_id in result:
            raise GroundCertificateError("public policy instances repeat an instance ID")
        result[instance_id] = row
    return result


def _execute_ground_partition(
    corpus: Path,
    *,
    partition: str,
    quality_attestation_pin: QualityAttestationPin,
    attestation: PublisherAttestation,
    verifier_identity: Mapping[str, object],
    executable: bytes,
    execution: _VerifierExecution,
    timeout_seconds: float,
) -> dict[str, Any]:
    loaded = load_prepared_partition(
        corpus,
        partition=partition,
        include_evaluator=True,
        quality_attestation_pin=quality_attestation_pin,
    )
    access = loaded.target_access
    if access is None:
        raise RuntimeError("partitioned ground verification opened no target capability")
    access_record = access.as_dict()
    tasks_by_instance = _task_by_instance(loaded.tasks)
    target_path = corpus.joinpath(*PurePosixPath(access.target_path).parts)
    target_rows = _read_jsonl(target_path, f"{partition} evaluator targets")
    target_by_instance: dict[str, Mapping[str, object]] = {}
    for target in target_rows:
        _verify_record(target, "evaluator target")
        instance_id = _text(target.get("instance_id"), "evaluator target instance ID")
        if target.get("learning_partition") != partition:
            raise GroundCertificateError("evaluator target crosses its ground partition")
        if instance_id in target_by_instance:
            raise GroundCertificateError("evaluator targets repeat an instance ID")
        target_by_instance[instance_id] = target
    evidence_by_instance, _ = _evidence_rows(
        attestation,
        target_by_instance=target_by_instance,
        verifier_identity=verifier_identity,
        partition=partition,
    )
    all_policies = _policy_rows_by_instance(corpus)
    expected_instances = set(target_by_instance)
    if (
        set(tasks_by_instance) != expected_instances
        or set(evidence_by_instance) != expected_instances
        or not expected_instances.issubset(all_policies)
    ):
        raise GroundCertificateError(
            "partition certificate inputs do not cover one common instance census"
        )

    target_receipts: list[dict[str, Any]] = []
    for instance_id in sorted(expected_instances):
        target = target_by_instance[instance_id]
        status = _text(target["reference_status"], "target reference status")
        if status not in SUPPORTED_REFERENCE_STATUSES:
            raise GroundCertificateError(
                f"reference status {status!r} is not supported by ground-certificate protocol v2"
            )
        evidence, artifact_relative, artifact = evidence_by_instance[instance_id]
        certificate = {
            "artifact_base64": base64.b64encode(artifact).decode("ascii"),
            "artifact_sha256": evidence["artifact_sha256"],
            "evidence_kind": evidence["evidence_kind"],
            "size_bytes": len(artifact),
        }
        request_payload = {
            "certificate": certificate,
            "claimed_reference_energy": target["reference_energy"],
            "evaluator_target": target,
            "public_instance": all_policies[instance_id],
            "reference_status": status,
            "schema": GROUND_CERTIFICATE_REQUEST_SCHEMA,
            "schema_version": 1,
            "target_record_digest": target["record_digest"],
            "verifier_identity": verifier_identity,
        }
        request = {**request_payload, "record_digest": content_digest(request_payload)}
        request_raw = canonical_json_bytes(request) + b"\n"
        result, result_sha256 = _run_verifier(
            executable,
            request,
            verifier_identity=verifier_identity,
            execution=execution,
            timeout_seconds=timeout_seconds,
        )
        task = tasks_by_instance[instance_id]
        condition = task.design_condition
        if condition is None:
            raise GroundCertificateError("partition task lacks scientific design identity")
        target_receipts.append(
            {
                "artifact_path": artifact_relative,
                "artifact_sha256": evidence["artifact_sha256"],
                "artifact_size_bytes": len(artifact),
                "base_lineage_key": condition.base_lineage_key,
                "claimed_reference_energy": target["reference_energy"],
                "design_condition": asdict(condition),
                "evidence_kind": evidence["evidence_kind"],
                "instance_id": instance_id,
                "learning_partition": partition,
                "public_instance_record_digest": all_policies[instance_id][
                    "record_digest"
                ],
                "reference_status": status,
                "request_record_digest": request["record_digest"],
                "request_sha256": hashlib.sha256(request_raw).hexdigest(),
                "result_record_digest": result["record_digest"],
                "result_sha256": result_sha256,
                "status": "accepted",
                "target_record_digest": target["record_digest"],
                "verifier_result": result,
            }
        )

    base_lineages = sorted({str(row["base_lineage_key"]) for row in target_receipts})
    census = {
        "accepted_count": len(target_receipts),
        "artifact_bytes": sum(int(row["artifact_size_bytes"]) for row in target_receipts),
        "base_lineage_count": len(base_lineages),
        "base_lineage_set_digest": content_digest(base_lineages),
        "by_design_axis": {
            field: _categorical_counts(
                [row["design_condition"] for row in target_receipts], field
            )
            for field in _DESIGN_CENSUS_FIELDS
        },
        "by_evidence_kind": _categorical_counts(target_receipts, "evidence_kind"),
        "by_reference_status": _categorical_counts(target_receipts, "reference_status"),
        "instance_set_digest": content_digest(sorted(expected_instances)),
        "target_count": len(target_receipts),
        "unique_artifact_count": len(
            {str(row["artifact_sha256"]) for row in target_receipts}
        ),
    }
    payload = {
        "authority": access_record,
        "census": census,
        "partition": partition,
        "protocol": GROUND_CERTIFICATE_PROTOCOL_V2,
        "schema": GROUND_CERTIFICATE_PARTITION_SCHEMA,
        "schema_version": GROUND_CERTIFICATE_PARTITION_VERSION,
        "targets": target_receipts,
        "verifier": dict(verifier_identity),
    }
    return {**payload, "record_digest": content_digest(payload)}


def verify_ground_certificate_partitions(
    corpus_directory: str | Path,
    *,
    quality_attestation_pin: QualityAttestationPin,
    verifier_pin: GroundCertificateVerifierPin,
    output_directory: str | Path,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Verify v4 targets in isolation and publish a target-free public root last."""

    output = Path(output_directory)
    if output.exists():
        raise FileExistsError(f"ground-certificate output already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(
            f"ground-certificate output parent does not exist: {output.parent}"
        )
    timeout = _validate_timeout(timeout_seconds)
    verifier_identity, executable, pinned_paths, execution = _verifier_identity(
        verifier_pin,
        publication=True,
    )
    corpus = Path(corpus_directory)
    (
        prepared_manifest,
        prepared_manifest_raw,
        prepared_manifest_record_digest,
        prepared_manifest_sha256,
        attestation,
        attestation_raw,
        attestation_sha256,
    ) = _public_v4_authorities(corpus, quality_attestation_pin)

    partition_records = {
        partition: _execute_ground_partition(
            corpus,
            partition=partition,
            quality_attestation_pin=quality_attestation_pin,
            attestation=attestation,
            verifier_identity=verifier_identity,
            executable=executable,
            execution=execution,
            timeout_seconds=timeout,
        )
        for partition in ("train", "val", "test")
    }
    for path, expected_sha256 in pinned_paths:
        if not hmac.compare_digest(
            hashlib.sha256(_read(path, "pinned verifier artifact")).hexdigest(),
            expected_sha256,
        ):
            raise GroundCertificateError("a pinned verifier artifact changed during preflight")
    if _read(corpus / "manifest.json", "prepared manifest") != prepared_manifest_raw:
        raise GroundCertificateError("prepared manifest changed during certificate preflight")
    if _read(attestation.path, "publisher attestation") != attestation_raw:
        raise GroundCertificateError("publisher attestation changed during certificate preflight")

    raw_by_partition = {
        partition: canonical_json_bytes(record) + b"\n"
        for partition, record in partition_records.items()
    }
    descriptors: dict[str, dict[str, object]] = {}
    for partition, record in partition_records.items():
        authority = record["authority"]
        census = record["census"]
        descriptors[partition] = {
            "accepted_count": census["accepted_count"],
            "evidence_manifest_record_digest": authority[
                "evidence_manifest_record_digest"
            ],
            "evidence_manifest_sha256": authority["evidence_manifest_sha256"],
            "instance_set_digest": census["instance_set_digest"],
            "path": f"partitions/{partition}.json",
            "receipt_record_digest": record["record_digest"],
            "sha256": hashlib.sha256(raw_by_partition[partition]).hexdigest(),
            "target_access_record_digest": authority["record_digest"],
            "target_count": census["target_count"],
            "target_set_digest": authority["target_set_digest"],
        }
    corpus_design = prepared_manifest.get("corpus_design")
    if not isinstance(corpus_design, Mapping):
        raise GroundCertificateError("prepared v4 corpus lacks scientific design identity")
    target_authority = prepared_manifest["target_authority"]
    root_payload = {
        "authority": {
            "publication_id": attestation.publication_id,
            "publisher_attestation_record_digest": attestation.digest,
            "publisher_attestation_sha256": attestation_sha256,
            "publisher_id": attestation.publisher_id,
            "target_authority_record_digest": target_authority["record_digest"],
            "total_target_count": target_authority["total_targets"],
        },
        "census": {
            "accepted_count": sum(
                int(record["census"]["accepted_count"])
                for record in partition_records.values()
            ),
            "by_partition": {
                partition: int(partition_records[partition]["census"]["target_count"])
                for partition in ("train", "val", "test")
            },
            "target_count": sum(
                int(record["census"]["target_count"])
                for record in partition_records.values()
            ),
        },
        "partitions": descriptors,
        "prepared_corpus": {
            "corpus_design_manifest_record_digest": corpus_design[
                "manifest_record_digest"
            ],
            "corpus_design_manifest_sha256": corpus_design["manifest_sha256"],
            "manifest_record_digest": prepared_manifest_record_digest,
            "manifest_sha256": prepared_manifest_sha256,
            "schema_version": prepared_manifest["schema_version"],
        },
        "protocol": GROUND_CERTIFICATE_PROTOCOL_V2,
        "schema": GROUND_CERTIFICATE_ROOT_SCHEMA,
        "schema_version": GROUND_CERTIFICATE_ROOT_VERSION,
        "verifier": dict(verifier_identity),
        "verifier_identity_digest": content_digest(verifier_identity),
    }
    root = {**root_payload, "record_digest": content_digest(root_payload)}
    root_raw = canonical_json_bytes(root) + b"\n"

    created = False
    try:
        output.mkdir()
        created = True
        (output / "partitions").mkdir()
        for partition in ("train", "val", "test"):
            _atomic_publish(
                output / "partitions" / f"{partition}.json",
                raw_by_partition[partition],
            )
        # Root is the publication marker and is intentionally written last.
        _atomic_publish(output / "root.json", root_raw)
    except Exception:
        if created:
            shutil.rmtree(output)
        raise
    return root


def load_ground_certificate_root(
    path: str | Path,
    *,
    expected_sha256: str,
    corpus_directory: str | Path,
    quality_attestation_pin: QualityAttestationPin,
) -> GroundCertificateRootReceipt:
    """Authenticate the public v2 root without opening any sealed partition artifact."""

    expected = _sha256(expected_sha256, "expected ground-certificate root SHA-256")
    root_path = Path(path)
    raw = _read(root_path, "ground-certificate root receipt")
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(observed_sha256, expected):
        raise GroundCertificateError(
            "ground-certificate root differs from its out-of-band SHA-256 pin"
        )
    root = _strict_object(raw, "ground-certificate root receipt")
    if raw != canonical_json_bytes(root) + b"\n":
        raise GroundCertificateError("ground-certificate root is not canonical JSON")
    _exact_keys(
        root,
        {
            "authority",
            "census",
            "partitions",
            "prepared_corpus",
            "protocol",
            "record_digest",
            "schema",
            "schema_version",
            "verifier",
            "verifier_identity_digest",
        },
        "ground-certificate root receipt",
    )
    root_record_digest = _verify_record(root, "ground-certificate root receipt")
    if (
        root["schema"] != GROUND_CERTIFICATE_ROOT_SCHEMA
        or root["schema_version"] != GROUND_CERTIFICATE_ROOT_VERSION
        or root["protocol"] != GROUND_CERTIFICATE_PROTOCOL_V2
    ):
        raise GroundCertificateError("unsupported ground-certificate root receipt")
    verifier = _validated_publication_verifier(root["verifier"])
    verifier_identity_digest = _sha256(
        root["verifier_identity_digest"], "root verifier-identity digest"
    )
    if verifier_identity_digest != content_digest(verifier):
        raise GroundCertificateError("root verifier-identity digest mismatch")

    corpus = Path(corpus_directory)
    (
        manifest,
        manifest_raw,
        manifest_record_digest,
        manifest_sha256,
        attestation,
        attestation_raw,
        attestation_sha256,
    ) = _public_v4_authorities(corpus, quality_attestation_pin)
    corpus_design = manifest.get("corpus_design")
    if not isinstance(corpus_design, Mapping):
        raise GroundCertificateError("prepared v4 corpus lacks scientific design identity")
    expected_prepared = {
        "corpus_design_manifest_record_digest": corpus_design["manifest_record_digest"],
        "corpus_design_manifest_sha256": corpus_design["manifest_sha256"],
        "manifest_record_digest": manifest_record_digest,
        "manifest_sha256": manifest_sha256,
        "schema_version": 4,
    }
    if root["prepared_corpus"] != expected_prepared:
        raise GroundCertificateError("ground-certificate root belongs to another corpus")
    target_authority = manifest["target_authority"]
    expected_authority = {
        "publication_id": attestation.publication_id,
        "publisher_attestation_record_digest": attestation.digest,
        "publisher_attestation_sha256": attestation_sha256,
        "publisher_id": attestation.publisher_id,
        "target_authority_record_digest": target_authority["record_digest"],
        "total_target_count": target_authority["total_targets"],
    }
    if root["authority"] != expected_authority:
        raise GroundCertificateError("ground-certificate root belongs to another authority")

    partitions = root["partitions"]
    if not isinstance(partitions, Mapping) or set(partitions) != {
        "train",
        "val",
        "test",
    }:
        raise GroundCertificateError("ground-certificate root partition set is invalid")
    descriptor_keys = {
        "accepted_count",
        "evidence_manifest_record_digest",
        "evidence_manifest_sha256",
        "instance_set_digest",
        "path",
        "receipt_record_digest",
        "sha256",
        "target_access_record_digest",
        "target_count",
        "target_set_digest",
    }
    counts: dict[str, int] = {}
    for partition in ("train", "val", "test"):
        descriptor = partitions[partition]
        if not isinstance(descriptor, Mapping):
            raise GroundCertificateError("ground partition descriptor must be an object")
        _exact_keys(descriptor, descriptor_keys, "ground partition descriptor")
        if descriptor["path"] != f"partitions/{partition}.json":
            raise GroundCertificateError("ground partition descriptor crosses partitions")
        for field in (
            "evidence_manifest_record_digest",
            "evidence_manifest_sha256",
            "instance_set_digest",
            "receipt_record_digest",
            "sha256",
            "target_access_record_digest",
            "target_set_digest",
        ):
            _sha256(descriptor[field], f"ground partition {field}")
        count = descriptor["target_count"]
        accepted = descriptor["accepted_count"]
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or accepted != count
        ):
            raise GroundCertificateError("ground partition is not fully accepted")
        prepared_descriptor = target_authority["partitions"][partition]
        if (
            count != prepared_descriptor["records"]
            or descriptor["target_set_digest"]
            != prepared_descriptor["target_set_digest"]
        ):
            raise GroundCertificateError(
                "ground partition descriptor differs from public target commitment"
            )
        counts[partition] = count
    expected_census = {
        "accepted_count": sum(counts.values()),
        "by_partition": counts,
        "target_count": sum(counts.values()),
    }
    if root["census"] != expected_census:
        raise GroundCertificateError("ground-certificate root census is inconsistent")
    if (
        _read(root_path, "ground-certificate root receipt") != raw
        or _read(corpus / "manifest.json", "prepared manifest") != manifest_raw
        or _read(attestation.path, "publisher attestation") != attestation_raw
    ):
        raise GroundCertificateError(
            "a public authority artifact changed while loading the ground root"
        )
    return GroundCertificateRootReceipt(
        path=root_path.resolve(),
        sha256=observed_sha256,
        record_digest=root_record_digest,
        target_authority_record_digest=target_authority["record_digest"],
        verifier_identity_digest=verifier_identity_digest,
        record=root,
    )


def load_ground_certificate_partition(
    root_path: str | Path,
    *,
    expected_root_sha256: str,
    partition: str,
    corpus_directory: str | Path,
    quality_attestation_pin: QualityAttestationPin,
) -> GroundCertificatePartitionLoad:
    """Open and authenticate exactly one root-authorized certificate partition."""

    if partition == "validation":
        partition = "val"
    if partition not in {"train", "val", "test"}:
        raise GroundCertificateError("ground-certificate partition is invalid")
    root = load_ground_certificate_root(
        root_path,
        expected_sha256=expected_root_sha256,
        corpus_directory=corpus_directory,
        quality_attestation_pin=quality_attestation_pin,
    )
    descriptor = root.record["partitions"][partition]
    relative = _safe_relative_path(descriptor["path"], "ground partition receipt path")
    receipt_path = _bound_file(root.path.parent, relative, "ground partition receipt")
    raw = _read(receipt_path, "ground partition receipt")
    if hashlib.sha256(raw).hexdigest() != descriptor["sha256"]:
        raise GroundCertificateError("ground partition receipt checksum mismatch")
    receipt = _strict_object(raw, "ground partition receipt")
    if raw != canonical_json_bytes(receipt) + b"\n":
        raise GroundCertificateError("ground partition receipt is not canonical JSON")
    _exact_keys(
        receipt,
        {
            "authority",
            "census",
            "partition",
            "protocol",
            "record_digest",
            "schema",
            "schema_version",
            "targets",
            "verifier",
        },
        "ground partition receipt",
    )
    receipt_digest = _verify_record(receipt, "ground partition receipt")
    if (
        receipt["schema"] != GROUND_CERTIFICATE_PARTITION_SCHEMA
        or receipt["schema_version"] != GROUND_CERTIFICATE_PARTITION_VERSION
        or receipt["protocol"] != GROUND_CERTIFICATE_PROTOCOL_V2
        or receipt["partition"] != partition
        or receipt_digest != descriptor["receipt_record_digest"]
        or receipt["verifier"] != root.record["verifier"]
    ):
        raise GroundCertificateError("ground partition receipt has the wrong identity")

    loaded = load_prepared_partition(
        corpus_directory,
        partition=partition,
        include_evaluator=True,
        quality_attestation_pin=quality_attestation_pin,
    )
    access = loaded.target_access
    if access is None:
        raise RuntimeError("partitioned ground load produced no target-access receipt")
    access_record = access.as_dict()
    if (
        receipt["authority"] != access_record
        or access.record_digest != descriptor["target_access_record_digest"]
        or access.evidence_manifest_record_digest
        != descriptor["evidence_manifest_record_digest"]
        or access.evidence_manifest_sha256 != descriptor["evidence_manifest_sha256"]
        or access.target_count != descriptor["target_count"]
        or access.target_set_digest != descriptor["target_set_digest"]
    ):
        raise GroundCertificateError(
            "ground partition receipt belongs to another target access"
        )

    rows = receipt["targets"]
    if not isinstance(rows, list) or len(rows) != access.target_count:
        raise GroundCertificateError("ground partition target census is invalid")
    task_by_instance = _task_by_instance(loaded.tasks)
    row_by_instance: dict[str, Mapping[str, object]] = {}
    target_keys = {
        "artifact_path",
        "artifact_sha256",
        "artifact_size_bytes",
        "base_lineage_key",
        "claimed_reference_energy",
        "design_condition",
        "evidence_kind",
        "instance_id",
        "learning_partition",
        "public_instance_record_digest",
        "reference_status",
        "request_record_digest",
        "request_sha256",
        "result_record_digest",
        "result_sha256",
        "status",
        "target_record_digest",
        "verifier_result",
    }
    for row in rows:
        if not isinstance(row, Mapping):
            raise GroundCertificateError("ground partition target receipt must be an object")
        _exact_keys(row, target_keys, "ground partition target receipt")
        instance_id = _text(row["instance_id"], "ground partition target instance ID")
        if instance_id in row_by_instance:
            raise GroundCertificateError("ground partition repeats a target instance")
        task = task_by_instance.get(instance_id)
        if task is None:
            raise GroundCertificateError("ground partition contains an unknown target instance")
        result = row["verifier_result"]
        if not isinstance(result, Mapping):
            raise GroundCertificateError("ground partition verifier result must be an object")
        result_digest = _verify_record(result, "ground partition verifier result")
        if (
            row["learning_partition"] != partition
            or row["status"] != "accepted"
            or row["claimed_reference_energy"] != task.task.ground_energy
            or row["reference_status"] != task.reference_status
            or row["artifact_sha256"] != task.certificate_digest
            or result.get("accepted") is not True
            or result.get("reason_code") != "accepted"
            or result.get("instance_id") != instance_id
            or result.get("claimed_reference_energy") != task.task.ground_energy
            or result.get("verifier_identity") != root.record["verifier"]
            or result_digest != row["result_record_digest"]
        ):
            raise GroundCertificateError(
                "ground partition does not prove its current target acceptance"
            )
        row_by_instance[instance_id] = row
    if list(row_by_instance) != sorted(task_by_instance) or set(row_by_instance) != set(
        task_by_instance
    ):
        raise GroundCertificateError("ground partition target ordering or coverage is invalid")
    census = receipt["census"]
    if not isinstance(census, Mapping):
        raise GroundCertificateError("ground partition census must be an object")
    if (
        census.get("target_count") != len(rows)
        or census.get("accepted_count") != len(rows)
        or census.get("instance_set_digest")
        != content_digest(sorted(row_by_instance))
        or census.get("target_count") != descriptor["target_count"]
        or census.get("accepted_count") != descriptor["accepted_count"]
        or census.get("instance_set_digest") != descriptor["instance_set_digest"]
    ):
        raise GroundCertificateError("ground partition census differs from its root")
    if (
        _read(receipt_path, "ground partition receipt") != raw
        or _read(root.path, "ground-certificate root receipt")
        != canonical_json_bytes(root.record) + b"\n"
    ):
        raise GroundCertificateError("ground receipt changed during partition loading")
    return GroundCertificatePartitionLoad(
        root=root,
        partition=partition,
        tasks=loaded.tasks,
        target_access=access,
        receipt=receipt,
    )


def project_ground_root_quality_authority(
    root: GroundCertificateRootReceipt,
    *,
    partition: str | None = None,
) -> tuple[GlobalQualityAuthority, PartitionQualityAuthority | None]:
    """Project comparable global authority and an optional public partition commitment."""

    if not isinstance(root, GroundCertificateRootReceipt):
        raise TypeError("root must be GroundCertificateRootReceipt")
    record = root.as_dict()
    authority = record["authority"]
    if not isinstance(authority, Mapping):
        raise RuntimeError("validated ground root lost its authority")
    global_payload = {
        "ground_root": {
            "receipt_sha256": root.sha256,
            "record_digest": root.record_digest,
            "verifier_identity_digest": root.verifier_identity_digest,
        },
        "publication_id": authority["publication_id"],
        "publisher_attestation_record_digest": authority[
            "publisher_attestation_record_digest"
        ],
        "publisher_id": authority["publisher_id"],
        "schema": GLOBAL_QUALITY_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "target_authority_record_digest": root.target_authority_record_digest,
    }
    global_authority = GlobalQualityAuthority(
        publication_id=str(authority["publication_id"]),
        publisher_id=str(authority["publisher_id"]),
        publisher_attestation_record_digest=str(
            authority["publisher_attestation_record_digest"]
        ),
        target_authority_record_digest=root.target_authority_record_digest,
        ground_root_receipt_sha256=root.sha256,
        ground_root_record_digest=root.record_digest,
        verifier_identity_digest=root.verifier_identity_digest,
        record_digest=content_digest(global_payload),
    )
    global_authority.as_dict()
    if partition is None:
        return global_authority, None
    if partition == "validation":
        partition = "val"
    if partition not in {"train", "val", "test"}:
        raise GroundCertificateError("quality-authority partition is invalid")
    descriptor = record["partitions"][partition]
    if not isinstance(descriptor, Mapping):
        raise RuntimeError("validated ground root lost its partition descriptor")
    partition_payload = {
        "evidence_manifest_record_digest": descriptor[
            "evidence_manifest_record_digest"
        ],
        "evidence_manifest_sha256": descriptor["evidence_manifest_sha256"],
        "ground_partition": {
            "accepted_count": descriptor["accepted_count"],
            "instance_set_digest": descriptor["instance_set_digest"],
            "receipt_record_digest": descriptor["receipt_record_digest"],
            "receipt_sha256": descriptor["sha256"],
        },
        "name": partition,
        "schema": PARTITION_QUALITY_AUTHORITY_SCHEMA,
        "schema_version": 1,
        "target_access_record_digest": descriptor["target_access_record_digest"],
        "target_count": descriptor["target_count"],
        "target_set_digest": descriptor["target_set_digest"],
    }
    partition_authority = PartitionQualityAuthority(
        name=partition,
        target_access_record_digest=str(descriptor["target_access_record_digest"]),
        evidence_manifest_record_digest=str(
            descriptor["evidence_manifest_record_digest"]
        ),
        evidence_manifest_sha256=str(descriptor["evidence_manifest_sha256"]),
        target_set_digest=str(descriptor["target_set_digest"]),
        target_count=int(descriptor["target_count"]),
        ground_partition_receipt_record_digest=str(
            descriptor["receipt_record_digest"]
        ),
        ground_partition_receipt_sha256=str(descriptor["sha256"]),
        accepted_count=int(descriptor["accepted_count"]),
        instance_set_digest=str(descriptor["instance_set_digest"]),
        record_digest=content_digest(partition_payload),
    )
    partition_authority.as_dict()
    return global_authority, partition_authority


def project_ground_partition_quality_authority(
    loaded: GroundCertificatePartitionLoad,
) -> tuple[GlobalQualityAuthority, PartitionQualityAuthority]:
    """Project and rebind an actually opened partition to its public root descriptor."""

    if not isinstance(loaded, GroundCertificatePartitionLoad):
        raise TypeError("loaded must be GroundCertificatePartitionLoad")
    global_authority, partition_authority = project_ground_root_quality_authority(
        loaded.root,
        partition=loaded.partition,
    )
    if partition_authority is None:
        raise RuntimeError("selected partition projection disappeared")
    access = loaded.target_access.as_dict()
    if (
        access["target_authority_record_digest"]
        != global_authority.target_authority_record_digest
        or access["record_digest"] != partition_authority.target_access_record_digest
        or access["evidence_manifest_record_digest"]
        != partition_authority.evidence_manifest_record_digest
        or access["evidence_manifest_sha256"]
        != partition_authority.evidence_manifest_sha256
        or access["target_set_digest"] != partition_authority.target_set_digest
        or access["target_count"] != partition_authority.target_count
    ):
        raise GroundCertificateError(
            "opened target capability differs from its projected partition authority"
        )
    return global_authority, partition_authority


def verify_ground_certificates(
    corpus_directory: str | Path,
    *,
    quality_attestation_pin: QualityAttestationPin,
    verifier_pin: GroundCertificateVerifierPin,
    output_path: str | Path,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Execute every supported certificate exactly once and atomically publish a receipt."""

    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"ground-certificate receipt already exists: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(
            f"ground-certificate receipt parent does not exist: {output.parent}"
        )
    timeout = _validate_timeout(timeout_seconds)
    verifier_identity, executable, pinned_paths, execution = _verifier_identity(
        verifier_pin
    )
    corpus = Path(corpus_directory)
    prepared_manifest_path = corpus / "manifest.json"
    prepared_manifest_raw = _read(prepared_manifest_path, "prepared manifest")
    prepared_manifest = _strict_object(prepared_manifest_raw, "prepared manifest")
    prepared_manifest_record_digest = _verify_record(prepared_manifest, "prepared manifest")
    prepared_manifest_sha256 = hashlib.sha256(prepared_manifest_raw).hexdigest()

    tasks = load_prepared_tasks(
        corpus,
        include_evaluator=True,
        quality_attestation_pin=quality_attestation_pin,
    )
    tasks_by_instance = _task_by_instance(tasks)
    attestation = load_publisher_attestation(
        quality_attestation_pin,
        prepared_manifest_path=prepared_manifest_path,
    )
    attestation_raw = _read(attestation.path, "publisher attestation")
    attestation_record = _strict_object(attestation_raw, "publisher attestation")
    if _verify_record(attestation_record, "publisher attestation") != attestation.digest:
        raise GroundCertificateError("publisher attestation changed after authentication")
    attestation_sha256 = hashlib.sha256(attestation_raw).hexdigest()
    target_rows = _read_jsonl(corpus / "evaluator_targets.jsonl", "evaluator targets")
    target_by_instance: dict[str, Mapping[str, object]] = {}
    for target in target_rows:
        _verify_record(target, "evaluator target")
        instance_id = _text(target.get("instance_id"), "evaluator target instance ID")
        if instance_id in target_by_instance:
            raise GroundCertificateError("evaluator targets repeat an instance ID")
        target_by_instance[instance_id] = target
    evidence_receipt = validate_quality_evidence(
        attestation,
        evaluator_targets_path=corpus / "evaluator_targets.jsonl",
        targets=target_rows,
    )
    evidence_by_instance, evidence_manifest_raw = _evidence_rows(
        attestation,
        target_by_instance=target_by_instance,
        verifier_identity=verifier_identity,
    )
    policy_rows = _read_jsonl(corpus / "policy_instances.jsonl", "public policy instances")
    policy_by_instance: dict[str, Mapping[str, object]] = {}
    for policy in policy_rows:
        _verify_record(policy, "public policy instance")
        instance_id = _text(policy.get("instance_id"), "public policy instance ID")
        if instance_id in policy_by_instance:
            raise GroundCertificateError("public policy instances repeat an instance ID")
        policy_by_instance[instance_id] = policy
    expected_instances = set(target_by_instance)
    if (
        set(policy_by_instance) != expected_instances
        or set(tasks_by_instance) != expected_instances
        or set(evidence_by_instance) != expected_instances
    ):
        raise GroundCertificateError(
            "certificate preflight inputs do not exactly cover one common instance census"
        )

    target_receipts: list[dict[str, Any]] = []
    for instance_id in sorted(expected_instances):
        target = target_by_instance[instance_id]
        status = _text(target["reference_status"], "target reference status")
        if status not in SUPPORTED_REFERENCE_STATUSES:
            raise GroundCertificateError(
                f"reference status {status!r} is not supported by ground-certificate protocol v1"
            )
        evidence, artifact_relative, artifact = evidence_by_instance[instance_id]
        certificate = {
            "artifact_base64": base64.b64encode(artifact).decode("ascii"),
            "artifact_sha256": evidence["artifact_sha256"],
            "evidence_kind": evidence["evidence_kind"],
            "size_bytes": len(artifact),
        }
        request_payload = {
            "certificate": certificate,
            "claimed_reference_energy": target["reference_energy"],
            "evaluator_target": target,
            "public_instance": policy_by_instance[instance_id],
            "reference_status": status,
            "schema": GROUND_CERTIFICATE_REQUEST_SCHEMA,
            "schema_version": 1,
            "target_record_digest": target["record_digest"],
            "verifier_identity": verifier_identity,
        }
        request = {**request_payload, "record_digest": content_digest(request_payload)}
        request_raw = canonical_json_bytes(request) + b"\n"
        result, result_sha256 = _run_verifier(
            executable,
            request,
            verifier_identity=verifier_identity,
            execution=execution,
            timeout_seconds=timeout,
        )
        prepared_task = tasks_by_instance[instance_id]
        design_condition = prepared_task.design_condition
        if design_condition is None:  # Narrowed by _task_by_instance.
            raise RuntimeError("scientific design identity disappeared during preflight")
        target_receipts.append(
            {
                "artifact_path": artifact_relative,
                "artifact_sha256": evidence["artifact_sha256"],
                "artifact_size_bytes": len(artifact),
                "base_lineage_key": design_condition.base_lineage_key,
                "claimed_reference_energy": target["reference_energy"],
                "design_condition": asdict(design_condition),
                "evidence_kind": evidence["evidence_kind"],
                "instance_id": instance_id,
                "learning_partition": prepared_task.partition,
                "public_instance_record_digest": policy_by_instance[instance_id][
                    "record_digest"
                ],
                "reference_status": status,
                "request_record_digest": request["record_digest"],
                "request_sha256": hashlib.sha256(request_raw).hexdigest(),
                "result_record_digest": result["record_digest"],
                "result_sha256": result_sha256,
                "status": "accepted",
                "target_record_digest": target["record_digest"],
                "verifier_result": result,
            }
        )

    for path, expected_sha256 in pinned_paths:
        if not hmac.compare_digest(
            hashlib.sha256(_read(path, "pinned verifier file")).hexdigest(),
            expected_sha256,
        ):
            raise GroundCertificateError("a pinned verifier artifact changed during preflight")
    if _read(prepared_manifest_path, "prepared manifest") != prepared_manifest_raw:
        raise GroundCertificateError("prepared manifest changed during certificate preflight")
    if _read(attestation.path, "publisher attestation") != attestation_raw:
        raise GroundCertificateError("publisher attestation changed during certificate preflight")
    evidence_path = _bound_file(
        attestation.path.parent,
        _safe_relative_path(
            attestation.evidence_manifest_path,
            "quality evidence manifest path",
        ),
        "quality evidence manifest",
    )
    if _read(evidence_path, "quality evidence manifest") != evidence_manifest_raw:
        raise GroundCertificateError("quality evidence manifest changed during preflight")
    outputs = prepared_manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise GroundCertificateError("prepared manifest output registry is invalid")
    for filename, output_identity in outputs.items():
        if (
            not isinstance(filename, str)
            or not isinstance(output_identity, Mapping)
            or not isinstance(output_identity.get("sha256"), str)
            or hashlib.sha256(_read(corpus / filename, f"prepared output {filename}")).hexdigest()
            != output_identity["sha256"]
        ):
            raise GroundCertificateError("a prepared corpus output changed during preflight")

    def counts(field: str) -> dict[str, int]:
        values: dict[str, int] = {}
        for row in target_receipts:
            value = str(row[field])
            values[value] = values.get(value, 0) + 1
        return dict(sorted(values.items()))

    def design_counts(field: str) -> dict[str, int]:
        values: dict[str, int] = {}
        for row in target_receipts:
            condition = row["design_condition"]
            if not isinstance(condition, Mapping):
                raise RuntimeError("target receipt lost its design condition")
            value = str(condition[field])
            values[value] = values.get(value, 0) + 1
        return dict(sorted(values.items()))

    base_lineages = sorted({str(row["base_lineage_key"]) for row in target_receipts})
    corpus_design = prepared_manifest["corpus_design"]
    receipt_payload = {
        "authority": {
            "evidence_manifest_record_digest": evidence_receipt.evidence_manifest_digest,
            "evidence_manifest_sha256": evidence_receipt.evidence_manifest_sha256,
            "publication_id": attestation.publication_id,
            "publisher_attestation_record_digest": attestation.digest,
            "publisher_attestation_sha256": attestation_sha256,
            "publisher_id": attestation.publisher_id,
            "target_count": evidence_receipt.target_count,
            "target_set_digest": evidence_receipt.target_set_digest,
        },
        "census": {
            "accepted_count": len(target_receipts),
            "artifact_bytes": sum(row["artifact_size_bytes"] for row in target_receipts),
            "base_lineage_count": len(base_lineages),
            "base_lineage_set_digest": content_digest(base_lineages),
            "by_design_axis": {
                field: design_counts(field) for field in _DESIGN_CENSUS_FIELDS
            },
            "by_evidence_kind": counts("evidence_kind"),
            "by_learning_partition": counts("learning_partition"),
            "by_reference_status": counts("reference_status"),
            "instance_set_digest": content_digest(sorted(expected_instances)),
            "target_count": len(target_receipts),
            "unique_artifact_count": len(
                {row["artifact_sha256"] for row in target_receipts}
            ),
        },
        "prepared_corpus": {
            "corpus_design_manifest_record_digest": corpus_design[
                "manifest_record_digest"
            ],
            "corpus_design_manifest_sha256": corpus_design["manifest_sha256"],
            "manifest_record_digest": prepared_manifest_record_digest,
            "manifest_sha256": prepared_manifest_sha256,
            "schema_version": prepared_manifest["schema_version"],
        },
        "protocol": GROUND_CERTIFICATE_PROTOCOL,
        "schema": GROUND_CERTIFICATE_PREFLIGHT_SCHEMA,
        "schema_version": GROUND_CERTIFICATE_PREFLIGHT_VERSION,
        "targets": target_receipts,
        "verifier": verifier_identity,
    }
    receipt = {**receipt_payload, "record_digest": content_digest(receipt_payload)}
    _atomic_publish(output, canonical_json_bytes(receipt) + b"\n")
    return receipt


def load_ground_certificate_preflight(
    path: str | Path,
    *,
    expected_sha256: str,
    corpus_directory: str | Path,
    quality_attestation_pin: QualityAttestationPin,
) -> dict[str, Any]:
    """Authenticate a pinned preflight receipt against the current corpus and authority.

    This loader does not execute the external verifier again. Its out-of-band receipt-file pin,
    canonical record validation, exact target reconstruction, and current corpus/authority checks
    make the already executed preflight safe to bind into downstream scientific receipts.
    """

    expected_receipt_sha256 = _sha256(
        expected_sha256,
        "expected ground-certificate preflight SHA-256",
    )
    receipt_path = Path(path)
    receipt_raw = _read(receipt_path, "ground-certificate preflight receipt")
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    if not hmac.compare_digest(receipt_sha256, expected_receipt_sha256):
        raise GroundCertificateError(
            "ground-certificate preflight receipt differs from its out-of-band SHA-256 pin"
        )
    receipt = _strict_object(receipt_raw, "ground-certificate preflight receipt")
    if receipt_raw != canonical_json_bytes(receipt) + b"\n":
        raise GroundCertificateError(
            "ground-certificate preflight receipt is not canonical JSON"
        )
    _exact_keys(
        receipt,
        {
            "authority",
            "census",
            "prepared_corpus",
            "protocol",
            "record_digest",
            "schema",
            "schema_version",
            "targets",
            "verifier",
        },
        "ground-certificate preflight receipt",
    )
    _verify_record(receipt, "ground-certificate preflight receipt")
    if (
        receipt["schema"] != GROUND_CERTIFICATE_PREFLIGHT_SCHEMA
        or receipt["schema_version"] != GROUND_CERTIFICATE_PREFLIGHT_VERSION
        or receipt["protocol"] != GROUND_CERTIFICATE_PROTOCOL
    ):
        raise GroundCertificateError("unsupported ground-certificate preflight receipt")
    verifier_identity = _validated_receipt_verifier(receipt["verifier"])

    corpus = Path(corpus_directory)
    prepared_manifest_path = corpus / "manifest.json"
    prepared_manifest_raw = _read(prepared_manifest_path, "prepared manifest")
    prepared_manifest = _strict_object(prepared_manifest_raw, "prepared manifest")
    prepared_manifest_record_digest = _verify_record(prepared_manifest, "prepared manifest")
    prepared_manifest_sha256 = hashlib.sha256(prepared_manifest_raw).hexdigest()
    tasks = load_prepared_tasks(
        corpus,
        include_evaluator=True,
        quality_attestation_pin=quality_attestation_pin,
    )
    tasks_by_instance = _task_by_instance(tasks)

    attestation = load_publisher_attestation(
        quality_attestation_pin,
        prepared_manifest_path=prepared_manifest_path,
    )
    attestation_raw = _read(attestation.path, "publisher attestation")
    attestation_record = _strict_object(attestation_raw, "publisher attestation")
    if _verify_record(attestation_record, "publisher attestation") != attestation.digest:
        raise GroundCertificateError("publisher attestation changed after authentication")
    attestation_sha256 = hashlib.sha256(attestation_raw).hexdigest()

    target_path = corpus / "evaluator_targets.jsonl"
    target_raw = _read(target_path, "evaluator targets")
    target_rows = _read_jsonl(target_path, "evaluator targets")
    target_by_instance: dict[str, Mapping[str, object]] = {}
    for target in target_rows:
        _verify_record(target, "evaluator target")
        instance_id = _text(target.get("instance_id"), "evaluator target instance ID")
        if instance_id in target_by_instance:
            raise GroundCertificateError("evaluator targets repeat an instance ID")
        target_by_instance[instance_id] = target
    evidence_receipt = validate_quality_evidence(
        attestation,
        evaluator_targets_path=target_path,
        targets=target_rows,
    )
    evidence_by_instance, evidence_manifest_raw = _evidence_rows(
        attestation,
        target_by_instance=target_by_instance,
        verifier_identity=verifier_identity,
    )
    evidence_path = _bound_file(
        attestation.path.parent,
        _safe_relative_path(
            attestation.evidence_manifest_path,
            "quality evidence manifest path",
        ),
        "quality evidence manifest",
    )

    policy_path = corpus / "policy_instances.jsonl"
    policy_raw = _read(policy_path, "public policy instances")
    policy_rows = _read_jsonl(policy_path, "public policy instances")
    policy_by_instance: dict[str, Mapping[str, object]] = {}
    for policy in policy_rows:
        _verify_record(policy, "public policy instance")
        instance_id = _text(policy.get("instance_id"), "public policy instance ID")
        if instance_id in policy_by_instance:
            raise GroundCertificateError("public policy instances repeat an instance ID")
        policy_by_instance[instance_id] = policy
    expected_instances = set(target_by_instance)
    if (
        not expected_instances
        or set(policy_by_instance) != expected_instances
        or set(tasks_by_instance) != expected_instances
        or set(evidence_by_instance) != expected_instances
    ):
        raise GroundCertificateError(
            "current certificate inputs do not exactly cover one common instance census"
        )

    corpus_design = prepared_manifest.get("corpus_design")
    if not isinstance(corpus_design, Mapping):
        raise GroundCertificateError("current prepared corpus lacks its scientific design receipt")
    expected_prepared_corpus = {
        "corpus_design_manifest_record_digest": corpus_design[
            "manifest_record_digest"
        ],
        "corpus_design_manifest_sha256": corpus_design["manifest_sha256"],
        "manifest_record_digest": prepared_manifest_record_digest,
        "manifest_sha256": prepared_manifest_sha256,
        "schema_version": prepared_manifest["schema_version"],
    }
    if receipt["prepared_corpus"] != expected_prepared_corpus:
        raise GroundCertificateError(
            "ground-certificate preflight belongs to another prepared corpus"
        )
    authority = receipt["authority"]
    expected_authority = {
        "evidence_manifest_record_digest": evidence_receipt.evidence_manifest_digest,
        "evidence_manifest_sha256": evidence_receipt.evidence_manifest_sha256,
        "publication_id": attestation.publication_id,
        "publisher_attestation_record_digest": attestation.digest,
        "publisher_attestation_sha256": attestation_sha256,
        "publisher_id": attestation.publisher_id,
        "target_count": evidence_receipt.target_count,
        "target_set_digest": evidence_receipt.target_set_digest,
    }
    if authority != expected_authority:
        raise GroundCertificateError(
            "ground-certificate preflight belongs to another quality authority"
        )

    raw_target_receipts = receipt["targets"]
    if not isinstance(raw_target_receipts, list) or not raw_target_receipts:
        raise GroundCertificateError("ground-certificate preflight targets must be nonempty")
    target_receipt_by_instance: dict[str, Mapping[str, object]] = {}
    target_receipt_keys = {
        "artifact_path",
        "artifact_sha256",
        "artifact_size_bytes",
        "base_lineage_key",
        "claimed_reference_energy",
        "design_condition",
        "evidence_kind",
        "instance_id",
        "learning_partition",
        "public_instance_record_digest",
        "reference_status",
        "request_record_digest",
        "request_sha256",
        "result_record_digest",
        "result_sha256",
        "status",
        "target_record_digest",
        "verifier_result",
    }
    for row in raw_target_receipts:
        if not isinstance(row, Mapping):
            raise GroundCertificateError("ground-certificate target receipts must be objects")
        _exact_keys(row, target_receipt_keys, "ground-certificate target receipt")
        instance_id = _text(row["instance_id"], "ground-certificate target instance ID")
        if instance_id in target_receipt_by_instance:
            raise GroundCertificateError("ground-certificate receipt repeats an instance ID")
        target_receipt_by_instance[instance_id] = row
    if (
        list(target_receipt_by_instance) != sorted(expected_instances)
        or set(target_receipt_by_instance) != expected_instances
    ):
        raise GroundCertificateError(
            "ground-certificate receipt does not cover the full current target census"
        )

    validated_rows: list[dict[str, Any]] = []
    artifact_snapshots: list[tuple[Path, bytes]] = []
    for instance_id in sorted(expected_instances):
        row = target_receipt_by_instance[instance_id]
        target = target_by_instance[instance_id]
        policy = policy_by_instance[instance_id]
        task = tasks_by_instance[instance_id]
        condition = task.design_condition
        if condition is None:
            raise GroundCertificateError("current task lacks a scientific design identity")
        evidence, artifact_relative, artifact = evidence_by_instance[instance_id]
        artifact_path = _bound_file(
            evidence_path.parent,
            artifact_relative,
            "quality evidence artifact",
        )
        artifact_snapshots.append((artifact_path, artifact))
        status = _text(target["reference_status"], "target reference status")
        if status not in SUPPORTED_REFERENCE_STATUSES:
            raise GroundCertificateError(
                f"reference status {status!r} is not supported by ground-certificate protocol v1"
            )
        certificate = {
            "artifact_base64": base64.b64encode(artifact).decode("ascii"),
            "artifact_sha256": evidence["artifact_sha256"],
            "evidence_kind": evidence["evidence_kind"],
            "size_bytes": len(artifact),
        }
        request_payload = {
            "certificate": certificate,
            "claimed_reference_energy": target["reference_energy"],
            "evaluator_target": target,
            "public_instance": policy,
            "reference_status": status,
            "schema": GROUND_CERTIFICATE_REQUEST_SCHEMA,
            "schema_version": 1,
            "target_record_digest": target["record_digest"],
            "verifier_identity": verifier_identity,
        }
        request = {**request_payload, "record_digest": content_digest(request_payload)}
        request_sha256 = hashlib.sha256(canonical_json_bytes(request) + b"\n").hexdigest()

        result = row["verifier_result"]
        if not isinstance(result, Mapping):
            raise GroundCertificateError("embedded ground-verifier result must be an object")
        _exact_keys(
            result,
            {
                "accepted",
                "claimed_reference_energy",
                "instance_id",
                "reason_code",
                "record_digest",
                "request_digest",
                "schema",
                "schema_version",
                "verifier_identity",
            },
            "embedded ground-verifier result",
        )
        result_record_digest = _verify_record(result, "embedded ground-verifier result")
        if (
            result["schema"] != GROUND_CERTIFICATE_RESULT_SCHEMA
            or result["schema_version"] != 1
            or result["verifier_identity"] != verifier_identity
            or result["request_digest"] != request["record_digest"]
            or result["instance_id"] != instance_id
            or result["claimed_reference_energy"] != target["reference_energy"]
            or result["accepted"] is not True
            or result["reason_code"] != "accepted"
        ):
            raise GroundCertificateError(
                "embedded ground-verifier result does not prove acceptance of its current target"
            )
        result_sha256 = hashlib.sha256(
            canonical_json_bytes(result) + b"\n"
        ).hexdigest()
        expected_row = {
            "artifact_path": artifact_relative,
            "artifact_sha256": evidence["artifact_sha256"],
            "artifact_size_bytes": len(artifact),
            "base_lineage_key": condition.base_lineage_key,
            "claimed_reference_energy": target["reference_energy"],
            "design_condition": asdict(condition),
            "evidence_kind": evidence["evidence_kind"],
            "instance_id": instance_id,
            "learning_partition": task.partition,
            "public_instance_record_digest": policy["record_digest"],
            "reference_status": status,
            "request_record_digest": request["record_digest"],
            "request_sha256": request_sha256,
            "result_record_digest": result_record_digest,
            "result_sha256": result_sha256,
            "status": "accepted",
            "target_record_digest": target["record_digest"],
            "verifier_result": result,
        }
        if row != expected_row:
            raise GroundCertificateError(
                "ground-certificate target receipt differs from its current target evidence"
            )
        validated_rows.append(expected_row)

    base_lineages = sorted({str(row["base_lineage_key"]) for row in validated_rows})
    expected_census = {
        "accepted_count": len(validated_rows),
        "artifact_bytes": sum(int(row["artifact_size_bytes"]) for row in validated_rows),
        "base_lineage_count": len(base_lineages),
        "base_lineage_set_digest": content_digest(base_lineages),
        "by_design_axis": {
            field: _categorical_counts(
                [row["design_condition"] for row in validated_rows],
                field,
            )
            for field in _DESIGN_CENSUS_FIELDS
        },
        "by_evidence_kind": _categorical_counts(validated_rows, "evidence_kind"),
        "by_learning_partition": _categorical_counts(
            validated_rows,
            "learning_partition",
        ),
        "by_reference_status": _categorical_counts(validated_rows, "reference_status"),
        "instance_set_digest": content_digest(sorted(expected_instances)),
        "target_count": len(validated_rows),
        "unique_artifact_count": len(
            {str(row["artifact_sha256"]) for row in validated_rows}
        ),
    }
    if receipt["census"] != expected_census:
        raise GroundCertificateError(
            "ground-certificate receipt census differs from the full current target census"
        )

    outputs = prepared_manifest.get("outputs")
    if not isinstance(outputs, Mapping):
        raise GroundCertificateError("prepared manifest output registry is invalid")
    outputs_unchanged = all(
        isinstance(filename, str)
        and isinstance(identity, Mapping)
        and isinstance(identity.get("sha256"), str)
        and hashlib.sha256(_read(corpus / filename, f"prepared output {filename}")).hexdigest()
        == identity["sha256"]
        for filename, identity in outputs.items()
    )
    unchanged = (
        _read(receipt_path, "ground-certificate preflight receipt") == receipt_raw
        and _read(prepared_manifest_path, "prepared manifest") == prepared_manifest_raw
        and _read(attestation.path, "publisher attestation") == attestation_raw
        and _read(evidence_path, "quality evidence manifest") == evidence_manifest_raw
        and _read(target_path, "evaluator targets") == target_raw
        and _read(policy_path, "public policy instances") == policy_raw
        and outputs_unchanged
        and all(
            _read(item_path, "quality evidence artifact") == raw
            for item_path, raw in artifact_snapshots
        )
    )
    if not unchanged:
        raise GroundCertificateError(
            "a bound artifact changed while loading the ground-certificate preflight"
        )
    return receipt


__all__ = [
    "GROUND_CERTIFICATE_PREFLIGHT_SCHEMA",
    "GROUND_CERTIFICATE_PROTOCOL",
    "GROUND_CERTIFICATE_PROTOCOL_V2",
    "GROUND_CERTIFICATE_ROOT_SCHEMA",
    "GroundCertificateError",
    "GroundCertificatePartitionLoad",
    "GroundCertificateRootReceipt",
    "GroundCertificateVerifierPin",
    "GlobalQualityAuthority",
    "PartitionQualityAuthority",
    "SUPPORTED_REFERENCE_STATUSES",
    "load_ground_certificate_partition",
    "load_ground_certificate_preflight",
    "load_ground_certificate_root",
    "project_ground_partition_quality_authority",
    "project_ground_root_quality_authority",
    "verify_ground_certificate_partitions",
    "verify_ground_certificates",
]
