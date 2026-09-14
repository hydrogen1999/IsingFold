"""Closed, byte-authenticated provenance boundaries for the hard/OOD release."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Self

from embedbench.hard_ood_schema import canonical_bytes, require_exact_keys

SOURCE_BUNDLE_SCHEMA = "embedbench.source-bundle"
SOURCE_BUNDLE_SCHEMA_VERSION = 1
MAX_SOURCE_BUNDLE_FILES = 10_000
MAX_SOURCE_BUNDLE_MANIFEST_BYTES = 16 * 1024 * 1024
MAX_SOURCE_BUNDLE_TOTAL_BYTES = 1024 * 1024 * 1024
MAX_SOURCE_BUNDLE_PATH_DEPTH = 32
SOURCE_BUNDLE_ROLES = frozenset(
    {
        "generation",
        "minorminer",
        "cpp_baseline",
        "continuation_checker",
        "portable_runtime",
    }
)

_SOURCE_BUNDLE_FIELDS = frozenset({"schema", "schema_version", "release_id", "role", "files"})
_SOURCE_FILE_FIELDS = frozenset({"relative_path", "sha256", "byte_count", "executable"})
_SHA256_ALPHABET = frozenset("0123456789abcdef")
_SAFE_PATH_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+\Z")
_BUNDLE_SEAL = object()
_SECURE_IO_AVAILABLE = (
    hasattr(os, "O_NOFOLLOW")
    and hasattr(os, "O_DIRECTORY")
    and os.open in os.supports_dir_fd
    and os.stat in os.supports_dir_fd
    and os.stat in os.supports_follow_symlinks
    and os.scandir in os.supports_fd
)
_DIRECTORY_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
if _SECURE_IO_AVAILABLE:
    _DIRECTORY_FLAGS |= os.O_DIRECTORY | os.O_NOFOLLOW
    _FILE_FLAGS |= os.O_NOFOLLOW


def _require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _SHA256_ALPHABET for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be nonempty text")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{name} contains a Unicode surrogate")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _closed_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _decode_canonical_value(value: object) -> object:
    """Invert the canonical float tags after duplicate-key-safe JSON parsing."""

    if type(value) is list:
        return [_decode_canonical_value(item) for item in value]
    if type(value) is not dict:
        return value
    if set(value) == {"__float64_hex__"}:
        encoded = value["__float64_hex__"]
        if type(encoded) is not str:
            raise TypeError("canonical binary64 tag must contain exact text")
        try:
            decoded = float.fromhex(encoded)
        except ValueError as error:
            raise ValueError("canonical binary64 tag is invalid") from error
        if not math.isfinite(decoded):
            raise ValueError("canonical binary64 tag must be finite")
        normalized = 0.0 if decoded == 0.0 else decoded
        if normalized.hex().lower() != encoded:
            raise ValueError("canonical binary64 tag is not normalized")
        return normalized
    if "__float64_hex__" in value:
        raise ValueError("reserved canonical binary64 key is ambiguous")
    return {key: _decode_canonical_value(item) for key, item in value.items()}


def parse_canonical_json_bytes(payload: object, *, name: str) -> dict[str, object]:
    """Parse one exact canonical JSON object from immutable bytes."""

    if type(payload) is not bytes:
        raise TypeError(f"{name} must be exact bytes")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} must be valid UTF-8") from error
    try:
        raw_value = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_closed_object,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} must be valid JSON") from error
    if type(raw_value) is not dict:
        raise TypeError(f"{name} must contain one JSON object")
    value = _decode_canonical_value(raw_value)
    if type(value) is not dict:
        raise TypeError(f"{name} must contain one JSON object")
    if canonical_bytes(value) != payload:
        raise ValueError(f"{name} must use exact canonical JSON bytes")
    return value


def _relative_parts(value: object) -> tuple[str, ...]:
    path_text = _require_nonempty_text(value, "relative_path")
    if "\\" in path_text or "\x00" in path_text:
        raise ValueError("relative_path must use safe POSIX separators")
    path = PurePosixPath(path_text)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("relative_path must be a normalized relative POSIX path")
    if path.as_posix() != path_text:
        raise ValueError("relative_path must be normalized")
    if any(_SAFE_PATH_COMPONENT.fullmatch(part) is None for part in path.parts):
        raise ValueError("relative_path must use portable ASCII path characters")
    if len(path.parts) > MAX_SOURCE_BUNDLE_PATH_DEPTH:
        raise ValueError("relative_path exceeds the registered path depth limit")
    return path.parts


def validate_relative_path(value: object) -> str:
    """Return one normalized, portable bundle-relative path or fail closed."""

    return "/".join(_relative_parts(value))


def _stable_identity(metadata: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _require_secure_io_capabilities() -> None:
    if not _SECURE_IO_AVAILABLE:
        raise RuntimeError(
            "source bundle verification requires POSIX O_NOFOLLOW, O_DIRECTORY, "
            "fd-relative open/stat, no-follow stat, and fd-based scandir"
        )


def _open_root_without_symlinks(root: Path) -> int:
    """Open every lexical root component with ``O_NOFOLLOW``."""

    _require_secure_io_capabilities()
    raw = os.fspath(root)
    if "\x00" in raw:
        raise ValueError("source bundle root contains a null byte")
    lexical = Path(os.path.abspath(raw))
    if ".." in root.parts:
        raise ValueError("source bundle root must not contain parent traversal")
    directory_fd = os.open(os.sep, _DIRECTORY_FLAGS)
    try:
        for component in lexical.parts[1:]:
            next_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def _read_open_regular_file(
    file_fd: int,
    initial: os.stat_result,
    *,
    expected_byte_count: int,
) -> tuple[bytes, bool]:
    if expected_byte_count > MAX_SOURCE_BUNDLE_TOTAL_BYTES:
        raise ValueError("source bundle file exceeds the registered byte limit")
    if initial.st_size != expected_byte_count:
        raise ValueError("source bundle byte count mismatch before file read")
    opened = os.fstat(file_fd)
    if not stat.S_ISREG(opened.st_mode) or (
        opened.st_dev,
        opened.st_ino,
    ) != (
        initial.st_dev,
        initial.st_ino,
    ):
        raise ValueError("source bundle entry changed before it was opened")
    if opened.st_size != expected_byte_count:
        raise ValueError("source bundle byte count mismatch before file read")
    chunks: list[bytes] = []
    captured_byte_count = 0
    capture_limit = expected_byte_count + 1
    while captured_byte_count < capture_limit:
        chunk = os.read(file_fd, min(1024 * 1024, capture_limit - captured_byte_count))
        if not chunk:
            break
        captured_byte_count += len(chunk)
        chunks.append(chunk)
    if captured_byte_count > expected_byte_count:
        raise ValueError("source bundle file exceeded its declared byte count while being read")
    after = os.fstat(file_fd)
    if _stable_identity(opened) != _stable_identity(after):
        raise ValueError("source bundle file changed while it was being read")
    return b"".join(chunks), bool(opened.st_mode & 0o111)


def _snapshot_directory_tree(
    root_fd: int,
    *,
    declared_byte_counts: dict[str, int],
) -> dict[str, tuple[bytes, bool]]:
    """Read a closed regular-file tree without following a link at any depth."""

    snapshots: dict[str, tuple[bytes, bool]] = {}
    declared_directories = frozenset(
        "/".join(parts[:index])
        for path in declared_byte_counts
        for parts in (_relative_parts(path),)
        for index in range(1, len(parts))
    )
    declared_children: dict[tuple[str, ...], set[str]] = {}
    for path in declared_byte_counts:
        parts = _relative_parts(path)
        for index, child in enumerate(parts):
            declared_children.setdefault(parts[:index], set()).add(child)

    def visit(directory_fd: int, prefix: tuple[str, ...]) -> None:
        before = os.fstat(directory_fd)
        if not stat.S_ISDIR(before.st_mode):
            raise ValueError("source bundle tree contains a non-directory")
        expected_names = declared_children.get(prefix, set())
        seen_names: set[str] = set()
        with os.scandir(directory_fd) as entries:
            for entry in entries:
                name = entry.name
                if name not in expected_names:
                    unexpected = entry.stat(follow_symlinks=False)
                    if stat.S_ISLNK(unexpected.st_mode):
                        raise OSError("source bundle tree must not contain symlinks")
                    if stat.S_ISDIR(unexpected.st_mode):
                        raise ValueError("source bundle tree contains an undeclared directory")
                    if stat.S_ISREG(unexpected.st_mode):
                        raise ValueError("source bundle tree contains an undeclared file")
                    raise ValueError("source bundle tree contains an undeclared special entry")
                if name in seen_names or len(seen_names) >= len(expected_names):
                    raise ValueError("source bundle directory inventory exceeds its manifest")
                seen_names.add(name)
                parts = (*prefix, name)
                _relative_parts("/".join(parts))
                metadata = entry.stat(follow_symlinks=False)
                if stat.S_ISLNK(metadata.st_mode):
                    raise OSError("source bundle tree must not contain symlinks")
                if stat.S_ISDIR(metadata.st_mode):
                    if "/".join(parts) not in declared_directories:
                        raise ValueError("source bundle tree contains an undeclared directory")
                    child_fd = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
                    try:
                        opened = os.fstat(child_fd)
                        if (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino):
                            raise ValueError("source bundle directory changed while opening")
                        visit(child_fd, parts)
                    finally:
                        os.close(child_fd)
                    continue
                if not stat.S_ISREG(metadata.st_mode):
                    raise ValueError("source bundle entries must be regular files or directories")
                normalized = "/".join(parts)
                if normalized not in declared_byte_counts:
                    raise ValueError("source bundle tree contains an undeclared file")
                file_fd = os.open(name, _FILE_FLAGS, dir_fd=directory_fd)
                try:
                    snapshots[normalized] = _read_open_regular_file(
                        file_fd,
                        metadata,
                        expected_byte_count=declared_byte_counts[normalized],
                    )
                finally:
                    os.close(file_fd)
        if seen_names != expected_names:
            raise ValueError("source bundle directory is missing a declared entry")
        after = os.fstat(directory_fd)
        if _stable_identity(before) != _stable_identity(after):
            raise ValueError("source bundle directory changed while it was being read")

    visit(root_fd, ())
    return snapshots


@dataclass(frozen=True, slots=True)
class VerifiedSourceFile:
    relative_path: str
    sha256: str
    payload: bytes
    executable: bool


@dataclass(frozen=True, slots=True, init=False)
class VerifiedSourceBundle:
    """A source manifest and the exact regular-file bytes it authenticated."""

    release_id: str
    role: str
    manifest_sha256: str
    files: tuple[VerifiedSourceFile, ...]
    _manifest_payload: bytes
    _seal: object

    def file_bytes(
        self,
        relative_path: str,
        *,
        expected_manifest_sha256: str,
        expected_role: str,
        expected_release_id: str,
    ) -> bytes:
        return self.file_entry(
            relative_path,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_role=expected_role,
            expected_release_id=expected_release_id,
        ).payload

    def file_entry(
        self,
        relative_path: str,
        *,
        expected_manifest_sha256: str,
        expected_role: str,
        expected_release_id: str,
    ) -> VerifiedSourceFile:
        validate_source_bundle(
            self,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_role=expected_role,
            expected_release_id=expected_release_id,
        )
        parts = _relative_parts(relative_path)
        normalized = "/".join(parts)
        for entry in self.files:
            if entry.relative_path == normalized:
                return entry
        raise KeyError(relative_path)


@dataclass(frozen=True, slots=True, init=False)
class SourceBundleSnapshot:
    """Detached manifest and file bytes captured from one verified bundle capsule."""

    release_id: str
    role: str
    manifest_sha256: str
    manifest_bytes: bytes
    files: tuple[VerifiedSourceFile, ...]
    total_payload_bytes: int
    _seal: object

    def __new__(cls) -> Self:
        raise TypeError("source bundle snapshots can only be constructed by the verifier")

    def file_entry(self, relative_path: str) -> VerifiedSourceFile:
        if self._seal is not _SNAPSHOT_SEAL:
            raise TypeError("source bundle snapshot is not verifier-produced")
        normalized = validate_relative_path(relative_path)
        for entry in self.files:
            if entry.relative_path == normalized:
                return entry
        raise KeyError(relative_path)


_SNAPSHOT_SEAL = object()


def _source_bundle_identity(value: VerifiedSourceBundle) -> tuple[object, ...]:
    """Capture scalar state without retaining mutable capsule aliases."""

    manifest_payload = value._manifest_payload
    retained_files = value.files
    identity: list[object] = [
        value._seal,
        value.release_id,
        value.role,
        value.manifest_sha256,
        id(manifest_payload),
        manifest_payload,
        id(retained_files),
        len(retained_files) if type(retained_files) is tuple else -1,
    ]
    if type(retained_files) is tuple:
        for entry in retained_files:
            if type(entry) is not VerifiedSourceFile:
                identity.extend((id(entry), None))
                continue
            payload = entry.payload
            identity.extend(
                (
                    id(entry),
                    entry.relative_path,
                    entry.sha256,
                    id(payload),
                    payload,
                    entry.executable,
                )
            )
    return tuple(identity)


def snapshot_source_bundle(
    value: object,
    *,
    expected_manifest_sha256: str,
    expected_role: str,
    expected_release_id: str,
    maximum_total_payload_bytes: int = MAX_SOURCE_BUNDLE_TOTAL_BYTES,
    maximum_path_depth: int = MAX_SOURCE_BUNDLE_PATH_DEPTH,
) -> SourceBundleSnapshot:
    """Reparse, recompute, and detach one retained bundle under external roots.

    The before/after capsule identity check rejects check/use mutation. The returned object
    contains newly constructed file records and no alias to the source capsule's tuple or file
    records.
    """

    if type(value) is not VerifiedSourceBundle or value._seal is not _BUNDLE_SEAL:
        raise TypeError("source bundle must be produced by verify_source_bundle")
    if type(maximum_total_payload_bytes) is not int or maximum_total_payload_bytes <= 0:
        raise ValueError("maximum_total_payload_bytes must be a positive integer")
    if type(maximum_path_depth) is not int or maximum_path_depth <= 0:
        raise ValueError("maximum_path_depth must be a positive integer")
    before_identity = _source_bundle_identity(value)
    manifest_payload = value._manifest_payload
    retained_files = value.files
    if type(manifest_payload) is not bytes or type(retained_files) is not tuple:
        raise ValueError("verified source bundle fields disagree with retained manifest bytes")
    if len(manifest_payload) > MAX_SOURCE_BUNDLE_MANIFEST_BYTES:
        raise ValueError("verified source bundle manifest exceeds the registered byte limit")
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    external_sha256 = _require_sha256(
        expected_manifest_sha256,
        "expected_manifest_sha256",
    )
    if manifest_sha256 != external_sha256:
        raise ValueError("verified source bundle disagrees with its external commitment")

    document = require_exact_keys(
        parse_canonical_json_bytes(manifest_payload, name="retained source bundle manifest"),
        _SOURCE_BUNDLE_FIELDS,
        "retained source bundle manifest",
    )
    if document["schema"] != SOURCE_BUNDLE_SCHEMA or type(document["schema"]) is not str:
        raise ValueError("verified source bundle fields disagree with retained manifest bytes")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SOURCE_BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError("verified source bundle fields disagree with retained manifest bytes")
    release_id = _require_nonempty_text(document["release_id"], "release_id")
    role = _require_nonempty_text(document["role"], "role")
    if role not in SOURCE_BUNDLE_ROLES:
        raise ValueError("verified source bundle fields disagree with retained manifest bytes")
    if role != _require_nonempty_text(expected_role, "expected_role"):
        raise ValueError("verified source bundle has the wrong role")
    if release_id != _require_nonempty_text(expected_release_id, "expected_release_id"):
        raise ValueError("verified source bundle has the wrong release identity")
    if (
        value.release_id != release_id
        or value.role != role
        or value.manifest_sha256 != manifest_sha256
    ):
        raise ValueError("verified source bundle fields disagree with retained manifest bytes")

    raw_files = document["files"]
    if type(raw_files) is not list or not raw_files or len(raw_files) != len(retained_files):
        raise ValueError("verified source bundle fields disagree with retained manifest bytes")
    if len(raw_files) > MAX_SOURCE_BUNDLE_FILES:
        raise ValueError("verified source bundle file count exceeds the registered limit")

    captured_files: list[VerifiedSourceFile] = []
    recomputed_entries: list[dict[str, object]] = []
    total_payload_bytes = 0
    previous_path: str | None = None
    for index, (raw_entry, retained_entry) in enumerate(
        zip(raw_files, retained_files, strict=True)
    ):
        entry = require_exact_keys(
            raw_entry,
            _SOURCE_FILE_FIELDS,
            f"retained source bundle files[{index}]",
        )
        relative_path = "/".join(_relative_parts(entry["relative_path"]))
        if len(PurePosixPath(relative_path).parts) > maximum_path_depth:
            raise ValueError("verified source bundle path exceeds the protocol depth limit")
        if previous_path is not None and relative_path <= previous_path:
            raise ValueError("verified source bundle fields disagree with retained manifest bytes")
        previous_path = relative_path
        if type(retained_entry) is not VerifiedSourceFile:
            raise ValueError("verified source bundle fields disagree with retained manifest bytes")
        payload = retained_entry.payload
        executable = retained_entry.executable
        if type(payload) is not bytes or type(executable) is not bool:
            raise ValueError("verified source bundle fields disagree with retained manifest bytes")
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        payload_count = len(payload)
        recorded_sha256 = _require_sha256(
            entry["sha256"],
            f"retained source bundle files[{index}].sha256",
        )
        recorded_count = _require_nonnegative_int(
            entry["byte_count"],
            f"retained source bundle files[{index}].byte_count",
        )
        if type(entry["executable"]) is not bool:
            raise TypeError(f"retained source bundle files[{index}].executable must be a boolean")
        total_payload_bytes += payload_count
        if total_payload_bytes > maximum_total_payload_bytes:
            raise ValueError("verified source bundle exceeds the aggregate payload byte limit")
        if (
            entry["relative_path"] != relative_path
            or recorded_sha256 != payload_sha256
            or recorded_count != payload_count
            or entry["executable"] is not executable
            or retained_entry.relative_path != relative_path
            or retained_entry.sha256 != payload_sha256
        ):
            raise ValueError("verified source bundle fields disagree with retained manifest bytes")
        captured = VerifiedSourceFile(
            relative_path=relative_path,
            sha256=payload_sha256,
            payload=bytes(payload),
            executable=executable,
        )
        captured_files.append(captured)
        recomputed_entries.append(
            {
                "relative_path": relative_path,
                "sha256": payload_sha256,
                "byte_count": payload_count,
                "executable": executable,
            }
        )

    recomputed_manifest = canonical_bytes(
        {
            "schema": SOURCE_BUNDLE_SCHEMA,
            "schema_version": SOURCE_BUNDLE_SCHEMA_VERSION,
            "release_id": release_id,
            "role": role,
            "files": recomputed_entries,
        }
    )
    if recomputed_manifest != manifest_payload:
        raise ValueError("verified source bundle manifest does not match captured files")
    if _source_bundle_identity(value) != before_identity:
        raise ValueError("verified source bundle changed while it was being snapshotted")

    snapshot = object.__new__(SourceBundleSnapshot)
    object.__setattr__(snapshot, "release_id", release_id)
    object.__setattr__(snapshot, "role", role)
    object.__setattr__(snapshot, "manifest_sha256", manifest_sha256)
    object.__setattr__(snapshot, "manifest_bytes", bytes(manifest_payload))
    object.__setattr__(snapshot, "files", tuple(captured_files))
    object.__setattr__(snapshot, "total_payload_bytes", total_payload_bytes)
    object.__setattr__(snapshot, "_seal", _SNAPSHOT_SEAL)
    return snapshot


def verify_source_bundle(
    root: Path,
    manifest_bytes: bytes,
    *,
    expected_manifest_sha256: str,
    expected_role: str,
) -> VerifiedSourceBundle:
    """Authenticate a canonical manifest and securely snapshot every named source file."""

    _require_secure_io_capabilities()
    if not isinstance(root, Path):
        raise TypeError("source bundle root must be a pathlib.Path")
    if type(manifest_bytes) is not bytes:
        raise TypeError("source bundle manifest must be exact bytes")
    if len(manifest_bytes) > MAX_SOURCE_BUNDLE_MANIFEST_BYTES:
        raise ValueError("source bundle manifest exceeds the registered byte limit")
    expected = _require_sha256(expected_manifest_sha256, "expected_manifest_sha256")
    if hashlib.sha256(manifest_bytes).hexdigest() != expected:
        raise ValueError("source bundle manifest digest disagrees with its external commitment")
    document = require_exact_keys(
        parse_canonical_json_bytes(manifest_bytes, name="source bundle manifest"),
        _SOURCE_BUNDLE_FIELDS,
        "source bundle manifest",
    )
    if type(document["schema"]) is not str or document["schema"] != SOURCE_BUNDLE_SCHEMA:
        raise ValueError(f"source bundle requires schema {SOURCE_BUNDLE_SCHEMA!r}")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SOURCE_BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError(f"source bundle requires schema_version {SOURCE_BUNDLE_SCHEMA_VERSION}")
    release_id = _require_nonempty_text(document["release_id"], "release_id")
    role = _require_nonempty_text(document["role"], "role")
    required_role = _require_nonempty_text(expected_role, "expected_role")
    if role not in SOURCE_BUNDLE_ROLES or role != required_role:
        raise ValueError("source bundle role is unregistered or disagrees with expected_role")
    raw_files = document["files"]
    if type(raw_files) is not list or not raw_files:
        raise TypeError("source bundle files must be a nonempty exact JSON array")
    if len(raw_files) > MAX_SOURCE_BUNDLE_FILES:
        raise ValueError("source bundle file count exceeds the registered limit")

    descriptors: list[tuple[str, tuple[str, ...], str, int, bool]] = []
    for index, raw_entry in enumerate(raw_files):
        entry = require_exact_keys(
            raw_entry,
            _SOURCE_FILE_FIELDS,
            f"source bundle files[{index}]",
        )
        parts = _relative_parts(entry["relative_path"])
        relative_path = "/".join(parts)
        executable = entry["executable"]
        if type(executable) is not bool:
            raise TypeError(f"source bundle files[{index}].executable must be a boolean")
        descriptors.append(
            (
                relative_path,
                parts,
                _require_sha256(entry["sha256"], f"source bundle files[{index}].sha256"),
                _require_nonnegative_int(
                    entry["byte_count"],
                    f"source bundle files[{index}].byte_count",
                ),
                executable,
            )
        )
    paths = tuple(descriptor[0] for descriptor in descriptors)
    if paths != tuple(sorted(paths)) or len(paths) != len(set(paths)):
        raise ValueError("source bundle files must be sorted and duplicate-free")
    if sum(descriptor[3] for descriptor in descriptors) > MAX_SOURCE_BUNDLE_TOTAL_BYTES:
        raise ValueError("source bundle exceeds the aggregate payload byte limit")

    root_fd = _open_root_without_symlinks(root)
    verified_files: list[VerifiedSourceFile] = []
    try:
        if not stat.S_ISDIR(os.fstat(root_fd).st_mode):
            raise ValueError("source bundle root must be a directory")
        snapshots = _snapshot_directory_tree(
            root_fd,
            declared_byte_counts={
                relative_path: recorded_count
                for relative_path, _parts, _digest, recorded_count, _executable in descriptors
            },
        )
        if set(snapshots) != set(paths):
            unexpected = sorted(set(snapshots) - set(paths))
            missing = sorted(set(paths) - set(snapshots))
            raise ValueError(
                "source bundle closed inventory mismatch: "
                f"unexpected={unexpected!r}, missing={missing!r}"
            )
        for relative_path, _parts, recorded_sha256, recorded_count, executable in descriptors:
            payload, actual_executable = snapshots[relative_path]
            if len(payload) != recorded_count:
                raise ValueError(f"source bundle byte count mismatch for {relative_path!r}")
            actual_sha256 = hashlib.sha256(payload).hexdigest()
            if actual_sha256 != recorded_sha256:
                raise ValueError(f"source bundle digest mismatch for {relative_path!r}")
            if actual_executable is not executable:
                raise ValueError(f"source bundle executable intent mismatch for {relative_path!r}")
            verified_files.append(
                VerifiedSourceFile(
                    relative_path=relative_path,
                    sha256=actual_sha256,
                    payload=payload,
                    executable=executable,
                )
            )
    finally:
        os.close(root_fd)

    verified = object.__new__(VerifiedSourceBundle)
    object.__setattr__(verified, "release_id", release_id)
    object.__setattr__(verified, "role", role)
    object.__setattr__(verified, "manifest_sha256", expected)
    object.__setattr__(verified, "files", tuple(verified_files))
    object.__setattr__(verified, "_manifest_payload", bytes(manifest_bytes))
    object.__setattr__(verified, "_seal", _BUNDLE_SEAL)
    return verified


def validate_source_bundle(
    value: object,
    *,
    expected_manifest_sha256: str,
    expected_role: str,
    expected_release_id: str,
) -> VerifiedSourceBundle:
    """Recheck retained bytes against independently supplied trust roots."""

    if type(value) is not VerifiedSourceBundle or value._seal is not _BUNDLE_SEAL:
        raise TypeError("source bundle must be produced by verify_source_bundle")
    if type(value._manifest_payload) is not bytes:
        raise ValueError("verified source bundle fields disagree with retained manifest bytes")
    if len(value._manifest_payload) > MAX_SOURCE_BUNDLE_MANIFEST_BYTES:
        raise ValueError("verified source bundle manifest exceeds the registered byte limit")
    manifest_sha256 = hashlib.sha256(value._manifest_payload).hexdigest()
    if value.manifest_sha256 != manifest_sha256:
        raise ValueError("verified source bundle fields disagree with retained manifest bytes")
    externally_expected = _require_sha256(
        expected_manifest_sha256,
        "expected_manifest_sha256",
    )
    if manifest_sha256 != externally_expected:
        raise ValueError("verified source bundle disagrees with its external commitment")

    document = require_exact_keys(
        parse_canonical_json_bytes(value._manifest_payload, name="retained source bundle manifest"),
        _SOURCE_BUNDLE_FIELDS,
        "retained source bundle manifest",
    )
    if document["schema"] != SOURCE_BUNDLE_SCHEMA or type(document["schema"]) is not str:
        raise ValueError("verified source bundle fields disagree with retained manifest bytes")
    if (
        type(document["schema_version"]) is not int
        or document["schema_version"] != SOURCE_BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError("verified source bundle fields disagree with retained manifest bytes")
    release_id = _require_nonempty_text(document["release_id"], "release_id")
    role = _require_nonempty_text(document["role"], "role")
    if role not in SOURCE_BUNDLE_ROLES:
        raise ValueError("verified source bundle fields disagree with retained manifest bytes")
    if value.release_id != release_id or value.role != role or type(value.files) is not tuple:
        raise ValueError("verified source bundle fields disagree with retained manifest bytes")
    if role != _require_nonempty_text(expected_role, "expected_role"):
        raise ValueError("verified source bundle has the wrong role")
    if release_id != _require_nonempty_text(
        expected_release_id,
        "expected_release_id",
    ):
        raise ValueError("verified source bundle has the wrong release identity")

    raw_files = document["files"]
    if type(raw_files) is not list or not raw_files:
        raise ValueError("verified source bundle fields disagree with retained manifest bytes")
    if len(raw_files) > MAX_SOURCE_BUNDLE_FILES:
        raise ValueError("verified source bundle file count exceeds the registered limit")
    if len(raw_files) != len(value.files):
        raise ValueError("verified source bundle fields disagree with retained manifest bytes")
    previous_path: str | None = None
    total_payload_bytes = 0
    for index, (raw_entry, verified_entry) in enumerate(zip(raw_files, value.files, strict=True)):
        entry = require_exact_keys(
            raw_entry,
            _SOURCE_FILE_FIELDS,
            f"retained source bundle files[{index}]",
        )
        relative_path = "/".join(_relative_parts(entry["relative_path"]))
        recorded_sha256 = _require_sha256(
            entry["sha256"],
            f"retained source bundle files[{index}].sha256",
        )
        recorded_count = _require_nonnegative_int(
            entry["byte_count"],
            f"retained source bundle files[{index}].byte_count",
        )
        executable = entry["executable"]
        if type(executable) is not bool:
            raise TypeError(f"retained source bundle files[{index}].executable must be a boolean")
        if previous_path is not None and relative_path <= previous_path:
            raise ValueError("verified source bundle fields disagree with retained manifest bytes")
        previous_path = relative_path
        total_payload_bytes += recorded_count
        if total_payload_bytes > MAX_SOURCE_BUNDLE_TOTAL_BYTES:
            raise ValueError("verified source bundle exceeds the aggregate payload byte limit")
        if (
            type(verified_entry) is not VerifiedSourceFile
            or verified_entry.relative_path != relative_path
            or verified_entry.sha256 != recorded_sha256
            or verified_entry.executable is not executable
            or type(verified_entry.payload) is not bytes
            or len(verified_entry.payload) != recorded_count
            or hashlib.sha256(verified_entry.payload).hexdigest() != recorded_sha256
        ):
            raise ValueError("verified source bundle fields disagree with retained manifest bytes")
    return value


__all__ = [
    "MAX_SOURCE_BUNDLE_FILES",
    "MAX_SOURCE_BUNDLE_MANIFEST_BYTES",
    "MAX_SOURCE_BUNDLE_PATH_DEPTH",
    "MAX_SOURCE_BUNDLE_TOTAL_BYTES",
    "SOURCE_BUNDLE_ROLES",
    "SOURCE_BUNDLE_SCHEMA",
    "SOURCE_BUNDLE_SCHEMA_VERSION",
    "SourceBundleSnapshot",
    "VerifiedSourceBundle",
    "VerifiedSourceFile",
    "parse_canonical_json_bytes",
    "snapshot_source_bundle",
    "validate_source_bundle",
    "verify_source_bundle",
]
