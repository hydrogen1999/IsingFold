"""Canonical identities and strict schema primitives for the hard/OOD corpus.

This module is additive and deliberately independent of the legacy candidate-bank
serializer.  Hard/OOD content identities encode binary64 values by ``float.hex``
and reserve that tag against user-authored mappings, as required by the release
specification.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import re
import stat
import tempfile
import unicodedata
from bisect import bisect_left
from collections import OrderedDict
from collections.abc import Collection, Iterable, Iterator
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, BinaryIO, Literal, cast

import numpy as np

SeedPurpose = Literal[
    "problem",
    "host",
    "witness",
    "solver_attempt",
    "mechanism",
    "focus",
    "candidate_sample",
    "screen_label",
    "refine_label",
    "locked_label",
    "decode_tie",
    "audit_label",
    "audit_bootstrap",
]

SEED_PURPOSES: frozenset[str] = frozenset(
    {
        "problem",
        "host",
        "witness",
        "solver_attempt",
        "mechanism",
        "focus",
        "candidate_sample",
        "screen_label",
        "refine_label",
        "locked_label",
        "decode_tie",
        "audit_label",
        "audit_bootstrap",
    }
)
SEED_REQUEST_FIELDS: frozenset[str] = frozenset(
    {
        "release_id",
        "purpose",
        "partition",
        "panel",
        "cell",
        "task_type",
        "problem_sha256",
        "state_sha256",
        "candidate_index",
        "strength_index",
        "label_stage",
        "replicate",
    }
)
ARTIFACT_ENTRY_FIELDS: frozenset[str] = frozenset(
    {"relative_path", "sha256", "byte_count", "record_count", "schema_version"}
)
SEED_OPTION_FIELDS: frozenset[str] = frozenset(
    {
        "task_type",
        "problem_sha256",
        "state_sha256",
        "candidate_index",
        "strength_index",
        "label_stage",
    }
)
TASK_TYPES: frozenset[str] = frozenset({"terminal_quality", "partial_structural"})
LABEL_STAGES: frozenset[str] = frozenset({"screen", "refine", "locked", "audit"})
SEED_REGISTRY_SCHEMA = "embedbench.seed-registry-stage"
SEED_REGISTRY_SCHEMA_VERSION = 1
SEED_REGISTRY_FIELDS: frozenset[str] = frozenset(
    {"release_id", "stage_id", "parent_registry_sha256", "entries"}
)
SEED_REGISTRY_ENTRY_FIELDS: frozenset[str] = frozenset(
    {
        "request",
        "canonical_preimage",
        "seed_key_sha256",
        "seed_words",
        "seed32_required",
        "seed32",
        "collision_counter",
    }
)
SEED_REGISTRATION_FIELDS: frozenset[str] = frozenset({"request", "seed32_required"})
SEED_REGISTRY_SHARD_FIELDS: frozenset[str] = frozenset(
    {
        "relative_path",
        "sha256",
        "byte_count",
        "record_count",
        "schema_version",
        "first_seed_key_sha256",
        "last_seed_key_sha256",
    }
)
SEED_REGISTRY_ROOT_SCHEMA = "embedbench.seed-registry-root"
SEED_REGISTRY_ROOT_SCHEMA_VERSION = 1
SEED_REGISTRY_ROOT_FIELDS: frozenset[str] = frozenset(
    {
        "release_id",
        "stage_id",
        "parent_root_sha256",
        "entry_count",
        "shard_entry_limit",
        "occupied_seed32_count",
        "occupied_seed32_sha256",
        "shards",
    }
)
SEED_REGISTRY_STAGE_ARTIFACT_SCHEMA = "embedbench.seed-registry-stage-artifact"
SEED_REGISTRY_STAGE_ARTIFACT_SCHEMA_VERSION = 1
SEED_REGISTRY_STAGE_ARTIFACT_FIELDS: frozenset[str] = frozenset(
    {
        "stage_id",
        "visibility_class",
        "parent_root_sha256",
        "root_sha256",
        "root_artifact",
        "shard_artifacts",
    }
)
SEED_REGISTRY_ARTIFACT_MANIFEST_SCHEMA = "embedbench.seed-registry-artifact-manifest"
SEED_REGISTRY_ARTIFACT_MANIFEST_SCHEMA_VERSION = 1
SEED_REGISTRY_ARTIFACT_MANIFEST_FIELDS: frozenset[str] = frozenset(
    {"release_id", "terminal_root_sha256", "stages"}
)
SEED_REGISTRY_VISIBILITY_CLASSES: frozenset[str] = frozenset({"public", "custodian_private"})
SEED_REGISTRY_ENTRY_SCHEMA_VERSION = 1
SEED_REGISTRY_ROOT_FILENAME = "SEED_REGISTRY_ROOT.json"
MAX_SEED_REGISTRY_SHARD_ENTRIES = 50_000
MAX_SEED_REGISTRY_SORT_CHUNK_ENTRIES = 50_000
DEFAULT_SEED_REGISTRY_SHARD_ENTRIES = 50_000
DEFAULT_SEED_REGISTRY_SORT_CHUNK_ENTRIES = 50_000
MAX_SEED_REGISTRY_ENTRY_BYTES = 1_048_576
MAX_SEED_REGISTRY_ROOT_BYTES = 8_388_608
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_WINDOWS_DRIVE_PATTERN = re.compile(r"[A-Za-z]:")
_SEED_SHARD_PATH_PATTERN = re.compile(r"part-[0-9]{6}\.jsonl\Z")
_UINT32_LIMIT = 2**32
_SECURE_NOFOLLOW_AVAILABLE = hasattr(os, "O_NOFOLLOW") and hasattr(os, "O_DIRECTORY")
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
_VERIFICATION_SEAL = object()


@dataclass(frozen=True, slots=True)
class SeedFieldPolicy:
    """Required/null matrix and registered categorical values for one purpose."""

    required_fields: frozenset[str]
    task_types: frozenset[str]
    label_stages: frozenset[str]

    def __post_init__(self) -> None:
        if not self.required_fields <= SEED_OPTION_FIELDS:
            raise ValueError("seed policy contains an unknown optional field")
        if bool(self.task_types) != ("task_type" in self.required_fields):
            raise ValueError("seed policy task_types disagree with required_fields")
        if bool(self.label_stages) != ("label_stage" in self.required_fields):
            raise ValueError("seed policy label_stages disagree with required_fields")
        if not self.task_types <= TASK_TYPES:
            raise ValueError("seed policy contains an unknown task_type")
        if not self.label_stages <= LABEL_STAGES:
            raise ValueError("seed policy contains an unknown label_stage")

    @property
    def null_fields(self) -> frozenset[str]:
        return SEED_OPTION_FIELDS - self.required_fields


def _seed_policy(
    required_fields: Collection[str],
    *,
    task_types: Collection[str] = (),
    label_stages: Collection[str] = (),
) -> SeedFieldPolicy:
    return SeedFieldPolicy(
        required_fields=frozenset(required_fields),
        task_types=frozenset(task_types),
        label_stages=frozenset(label_stages),
    )


_QUALITY_TASK_TYPE = frozenset({"terminal_quality"})
_ALL_QUALITY_FIELDS = SEED_OPTION_FIELDS
SEED_FIELD_POLICIES = MappingProxyType(
    {
        "problem": _seed_policy(set()),
        "host": _seed_policy(set()),
        "witness": _seed_policy({"problem_sha256"}),
        "solver_attempt": _seed_policy({"problem_sha256"}),
        "mechanism": _seed_policy({"problem_sha256", "state_sha256"}),
        "focus": _seed_policy({"problem_sha256"}),
        "candidate_sample": _seed_policy({"problem_sha256"}),
        "screen_label": _seed_policy(
            _ALL_QUALITY_FIELDS,
            task_types=_QUALITY_TASK_TYPE,
            label_stages={"screen"},
        ),
        "refine_label": _seed_policy(
            _ALL_QUALITY_FIELDS,
            task_types=_QUALITY_TASK_TYPE,
            label_stages={"refine"},
        ),
        "locked_label": _seed_policy(
            _ALL_QUALITY_FIELDS,
            task_types=_QUALITY_TASK_TYPE,
            label_stages={"locked"},
        ),
        "decode_tie": _seed_policy(
            _ALL_QUALITY_FIELDS,
            task_types=_QUALITY_TASK_TYPE,
            label_stages=LABEL_STAGES,
        ),
        "audit_label": _seed_policy(
            _ALL_QUALITY_FIELDS,
            task_types=_QUALITY_TASK_TYPE,
            label_stages={"audit"},
        ),
        "audit_bootstrap": _seed_policy(set()),
    }
)

if frozenset(SEED_FIELD_POLICIES) != SEED_PURPOSES:
    raise RuntimeError("seed purpose policy matrix is incomplete")


def _reject_unicode_surrogates(value: str, name: str) -> None:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError(f"{name} contains a Unicode surrogate code point")


def _normalize_unicode_scalar(value: str, name: str) -> str:
    _reject_unicode_surrogates(value, name)
    return unicodedata.normalize("NFC", value)


def canonical_value(value: object) -> object:
    """Return the hard/OOD canonical JSON digest view defined by the spec."""

    value_type = type(value)
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("non-finite float")
        normalized = 0.0 if value == 0.0 else value
        return {"__float64_hex__": normalized.hex().lower()}
    if value is None or value_type is bool or value_type is int:
        return value
    if value_type is str:
        return _normalize_unicode_scalar(value, "canonical string")
    if value_type is list or value_type is tuple:
        return [canonical_value(item) for item in value]
    if value_type is dict:
        if not all(type(key) is str for key in value):
            raise TypeError("canonical object keys must be strings")
        keys = [_normalize_unicode_scalar(key, "canonical object key") for key in value]
        if len(keys) != len(set(keys)) or "__float64_hex__" in keys:
            raise ValueError("duplicate normalized or reserved key")
        return {
            _normalize_unicode_scalar(key, "canonical object key"): canonical_value(item)
            for key, item in value.items()
        }
    raise TypeError(f"unsupported canonical type: {type(value)!r}")


def canonical_bytes(value: object) -> bytes:
    """Serialize ``value`` to the platform-independent canonical UTF-8 bytes."""

    return json.dumps(
        canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    """Hash the canonical representation of a hard/OOD identity object."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_exact_keys(
    value: object,
    expected: Collection[str],
    name: str,
) -> dict[str, Any]:
    """Require a JSON object with exactly ``expected`` raw field names."""

    if type(value) is not dict:
        raise TypeError(f"{name} must be a JSON object")
    if not all(type(key) is str for key in value):
        raise TypeError(f"{name} keys must be strings")
    expected_keys = set(expected)
    if not all(isinstance(key, str) for key in expected_keys):
        raise TypeError("expected schema keys must be strings")
    actual = set(value)
    if actual != expected_keys:
        missing = sorted(expected_keys - actual)
        unknown = sorted(actual - expected_keys)
        raise ValueError(f"{name} schema fields differ: missing={missing}, unknown={unknown}")
    return cast(dict[str, Any], value)


def validate_versioned_object(
    value: object,
    *,
    schema: str,
    schema_version: int,
    fields: Collection[str],
    name: str,
) -> dict[str, Any]:
    """Validate the discriminator, version, and exact fields of a schema object."""

    if type(schema) is not str or not schema:
        raise ValueError("expected schema must be a non-empty string")
    if type(schema_version) is not int or schema_version <= 0:
        raise ValueError("expected schema_version must be a positive integer")
    document = require_exact_keys(
        value,
        {"schema", "schema_version", *fields},
        name,
    )
    actual_schema = document["schema"]
    if type(actual_schema) is not str or actual_schema != schema:
        raise ValueError(f"{name} requires schema {schema!r}")
    actual_version = document["schema_version"]
    if type(actual_version) is not int or actual_version != schema_version:
        raise ValueError(f"{name} requires schema_version {schema_version}")
    return document


def _nonempty_string(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return _normalize_unicode_scalar(value, name)


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty_string(value, name)


def _sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _optional_sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _sha256(value, name)


def _nonnegative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_nonnegative_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    try:
        return _nonnegative_int(value, name)
    except ValueError as error:
        raise ValueError(f"{name} must be a non-negative integer or null") from error


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _bounded_registry_limit(value: object, name: str, maximum: int) -> int:
    result = _positive_int(value, name)
    if result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _relative_posix_path(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("relative_path must be a normalized relative POSIX path")
    _reject_unicode_surrogates(value, "relative_path")
    if unicodedata.normalize("NFC", value) != value:
        raise ValueError("relative_path must be NFC-normalized")
    if any(
        character != "\x00" and unicodedata.category(character) in {"Cc", "Cf"}
        for character in value
    ):
        raise ValueError("relative_path must not contain a control character")
    if _WINDOWS_DRIVE_PATTERN.match(value) is not None:
        raise ValueError("relative_path must not use Windows drive syntax")
    path = value
    parsed = PurePosixPath(path)
    if (
        path == "."
        or path.startswith("/")
        or "\x00" in path
        or "\\" in path
        or parsed.as_posix() != path
        or any(part in {"", ".", ".."} for part in parsed.parts)
    ):
        raise ValueError("relative_path must be a normalized relative POSIX path")
    return path


def _safe_stage_id(value: object) -> str:
    stage_id = _nonempty_string(value, "stage_id")
    if (
        stage_id in {".", ".."}
        or "/" in stage_id
        or "\\" in stage_id
        or any(unicodedata.category(character) in {"Cc", "Cf"} for character in stage_id)
    ):
        raise ValueError("stage_id must be one safe path component")
    return stage_id


@dataclass(frozen=True, slots=True)
class ReleaseIdentity:
    """The immutable release namespace shared by all hard/OOD artifacts."""

    release_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "release_id", _nonempty_string(self.release_id, "release_id"))

    def to_dict(self) -> dict[str, object]:
        return {"release_id": self.release_id}

    @classmethod
    def from_dict(cls, value: object) -> ReleaseIdentity:
        document = require_exact_keys(value, {"release_id"}, "release identity")
        return cls(release_id=document["release_id"])


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    """Manifest identity for one immutable raw artifact payload."""

    relative_path: str
    sha256: str
    byte_count: int
    record_count: int
    schema_version: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "relative_path", _relative_posix_path(self.relative_path))
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))
        object.__setattr__(
            self,
            "byte_count",
            _nonnegative_int(self.byte_count, "byte_count"),
        )
        object.__setattr__(
            self,
            "record_count",
            _nonnegative_int(self.record_count, "record_count"),
        )
        object.__setattr__(
            self,
            "schema_version",
            _positive_int(self.schema_version, "schema_version"),
        )

    @classmethod
    def from_payload(
        cls,
        *,
        relative_path: str,
        payload: bytes,
        record_count: int,
        schema_version: int,
    ) -> ArtifactEntry:
        """Construct an entry whose digest and byte count bind exact file bytes."""

        if type(payload) is not bytes:
            raise TypeError("artifact payload must be bytes")
        return cls(
            relative_path=relative_path,
            sha256=hashlib.sha256(payload).hexdigest(),
            byte_count=len(payload),
            record_count=record_count,
            schema_version=schema_version,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "record_count": self.record_count,
            "schema_version": self.schema_version,
        }

    @classmethod
    def from_dict(cls, value: object) -> ArtifactEntry:
        document = require_exact_keys(value, ARTIFACT_ENTRY_FIELDS, "artifact entry")
        return cls(
            relative_path=document["relative_path"],
            sha256=document["sha256"],
            byte_count=document["byte_count"],
            record_count=document["record_count"],
            schema_version=document["schema_version"],
        )


@dataclass(frozen=True, slots=True)
class SeedRegistryStageArtifact:
    """Manifest identity for one complete immutable seed-registry stage."""

    stage_id: str
    visibility_class: str
    parent_root_sha256: str | None
    root_sha256: str
    root_artifact: ArtifactEntry
    shard_artifacts: tuple[ArtifactEntry, ...]

    def __post_init__(self) -> None:
        stage_id = _safe_stage_id(self.stage_id)
        visibility = _nonempty_string(self.visibility_class, "visibility_class")
        if visibility not in SEED_REGISTRY_VISIBILITY_CLASSES:
            raise ValueError("unregistered seed-registry visibility_class")
        parent_digest = _optional_sha256(self.parent_root_sha256, "parent_root_sha256")
        root_digest = _sha256(self.root_sha256, "root_sha256")
        if type(self.root_artifact) is not ArtifactEntry:
            raise TypeError("root_artifact must be an ArtifactEntry")
        if self.root_artifact.sha256 != root_digest:
            raise ValueError("root_artifact digest does not match root_sha256")
        if self.root_artifact.record_count != 1:
            raise ValueError("root_artifact must describe exactly one root record")
        if self.root_artifact.schema_version != SEED_REGISTRY_ROOT_SCHEMA_VERSION:
            raise ValueError("root_artifact has the wrong schema_version")
        expected_prefix = PurePosixPath("seed_registry") / stage_id
        root_path = PurePosixPath(self.root_artifact.relative_path)
        if root_path != expected_prefix / SEED_REGISTRY_ROOT_FILENAME:
            raise ValueError("root_artifact path does not match its stage_id")
        shards = tuple(self.shard_artifacts)
        if not shards or any(type(artifact) is not ArtifactEntry for artifact in shards):
            raise ValueError("shard_artifacts must contain ArtifactEntry values")
        for index, artifact in enumerate(shards):
            expected_path = expected_prefix / f"part-{index:06d}.jsonl"
            if PurePosixPath(artifact.relative_path) != expected_path:
                raise ValueError("shard_artifact paths must be complete and consecutive")
            if artifact.record_count <= 0:
                raise ValueError("shard_artifact must contain at least one record")
            if artifact.schema_version != SEED_REGISTRY_ENTRY_SCHEMA_VERSION:
                raise ValueError("shard_artifact has the wrong schema_version")
        object.__setattr__(self, "stage_id", stage_id)
        object.__setattr__(self, "visibility_class", visibility)
        object.__setattr__(self, "parent_root_sha256", parent_digest)
        object.__setattr__(self, "root_sha256", root_digest)
        object.__setattr__(self, "shard_artifacts", shards)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SEED_REGISTRY_STAGE_ARTIFACT_SCHEMA,
            "schema_version": SEED_REGISTRY_STAGE_ARTIFACT_SCHEMA_VERSION,
            "stage_id": self.stage_id,
            "visibility_class": self.visibility_class,
            "parent_root_sha256": self.parent_root_sha256,
            "root_sha256": self.root_sha256,
            "root_artifact": self.root_artifact.to_dict(),
            "shard_artifacts": [artifact.to_dict() for artifact in self.shard_artifacts],
        }

    @classmethod
    def from_dict(cls, value: object) -> SeedRegistryStageArtifact:
        document = validate_versioned_object(
            value,
            schema=SEED_REGISTRY_STAGE_ARTIFACT_SCHEMA,
            schema_version=SEED_REGISTRY_STAGE_ARTIFACT_SCHEMA_VERSION,
            fields=SEED_REGISTRY_STAGE_ARTIFACT_FIELDS,
            name="seed registry stage artifact",
        )
        raw_shards = document["shard_artifacts"]
        if type(raw_shards) is not list:
            raise ValueError("shard_artifacts must be a JSON array")
        return cls(
            stage_id=document["stage_id"],
            visibility_class=document["visibility_class"],
            parent_root_sha256=document["parent_root_sha256"],
            root_sha256=document["root_sha256"],
            root_artifact=ArtifactEntry.from_dict(document["root_artifact"]),
            shard_artifacts=tuple(ArtifactEntry.from_dict(item) for item in raw_shards),
        )


@dataclass(frozen=True, slots=True)
class SeedRegistryArtifactManifest:
    """Ordered, externally committed artifact chain for all registry stages."""

    release_id: str
    terminal_root_sha256: str
    stages: tuple[SeedRegistryStageArtifact, ...]

    def __post_init__(self) -> None:
        release_id = _nonempty_string(self.release_id, "release_id")
        terminal_digest = _sha256(self.terminal_root_sha256, "terminal_root_sha256")
        stages = tuple(self.stages)
        if not stages or any(type(stage) is not SeedRegistryStageArtifact for stage in stages):
            raise ValueError("seed registry artifact manifest requires stage artifacts")
        if len({stage.stage_id for stage in stages}) != len(stages):
            raise ValueError("seed registry artifact manifest repeats a stage_id")
        previous_digest: str | None = None
        for stage in stages:
            if stage.parent_root_sha256 != previous_digest:
                raise ValueError("seed registry artifacts do not form an ordered parent chain")
            previous_digest = stage.root_sha256
        if previous_digest != terminal_digest:
            raise ValueError("terminal_root_sha256 does not identify the final stage")
        object.__setattr__(self, "release_id", release_id)
        object.__setattr__(self, "terminal_root_sha256", terminal_digest)
        object.__setattr__(self, "stages", stages)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SEED_REGISTRY_ARTIFACT_MANIFEST_SCHEMA,
            "schema_version": SEED_REGISTRY_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            "release_id": self.release_id,
            "terminal_root_sha256": self.terminal_root_sha256,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def to_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_dict(cls, value: object) -> SeedRegistryArtifactManifest:
        document = validate_versioned_object(
            value,
            schema=SEED_REGISTRY_ARTIFACT_MANIFEST_SCHEMA,
            schema_version=SEED_REGISTRY_ARTIFACT_MANIFEST_SCHEMA_VERSION,
            fields=SEED_REGISTRY_ARTIFACT_MANIFEST_FIELDS,
            name="seed registry artifact manifest",
        )
        raw_stages = document["stages"]
        if type(raw_stages) is not list:
            raise ValueError("seed registry artifact stages must be a JSON array")
        return cls(
            release_id=document["release_id"],
            terminal_root_sha256=document["terminal_root_sha256"],
            stages=tuple(SeedRegistryStageArtifact.from_dict(item) for item in raw_stages),
        )

    @classmethod
    def from_bytes(cls, payload: bytes) -> SeedRegistryArtifactManifest:
        if type(payload) is not bytes:
            raise TypeError("seed registry artifact manifest payload must be exact bytes")
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("seed registry artifact manifest must be valid UTF-8 JSON") from error
        manifest = cls.from_dict(document)
        if manifest.to_bytes() != payload:
            raise ValueError("seed registry artifact manifest must use canonical bytes")
        return manifest


def validate_seed_registry_artifact_manifest(
    value: object,
    *,
    expected_manifest_sha256: str,
    expected_terminal_root_sha256: str,
) -> SeedRegistryArtifactManifest:
    """Parse a complete chain and bind both independently committed digests."""

    expected_manifest = _sha256(
        expected_manifest_sha256,
        "expected artifact manifest SHA-256",
    )
    expected_terminal = _sha256(
        expected_terminal_root_sha256,
        "expected terminal root SHA-256",
    )
    if type(value) is SeedRegistryArtifactManifest:
        manifest = SeedRegistryArtifactManifest.from_dict(value.to_dict())
    else:
        manifest = SeedRegistryArtifactManifest.from_dict(value)
    if manifest.sha256 != expected_manifest:
        raise ValueError("seed registry artifact manifest SHA-256 does not match its commitment")
    if manifest.terminal_root_sha256 != expected_terminal:
        raise ValueError("seed registry manifest does not match expected terminal root SHA-256")
    return manifest


@dataclass(frozen=True, slots=True)
class SeedRequest:
    """Complete, shard-independent namespace for one stochastic request."""

    release_id: str
    purpose: SeedPurpose
    partition: str
    panel: str
    cell: str
    task_type: str | None
    problem_sha256: str | None
    state_sha256: str | None
    candidate_index: int | None
    strength_index: int | None
    label_stage: str | None
    replicate: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "release_id", _nonempty_string(self.release_id, "release_id"))
        if type(self.purpose) is not str or self.purpose not in SEED_PURPOSES:
            raise ValueError(f"unregistered seed purpose: {self.purpose!r}")
        object.__setattr__(self, "partition", _nonempty_string(self.partition, "partition"))
        object.__setattr__(self, "panel", _nonempty_string(self.panel, "panel"))
        object.__setattr__(self, "cell", _nonempty_string(self.cell, "cell"))
        object.__setattr__(self, "task_type", _optional_string(self.task_type, "task_type"))
        object.__setattr__(
            self,
            "problem_sha256",
            _optional_sha256(self.problem_sha256, "problem_sha256"),
        )
        object.__setattr__(
            self,
            "state_sha256",
            _optional_sha256(self.state_sha256, "state_sha256"),
        )
        object.__setattr__(
            self,
            "candidate_index",
            _optional_nonnegative_int(self.candidate_index, "candidate_index"),
        )
        object.__setattr__(
            self,
            "strength_index",
            _optional_nonnegative_int(self.strength_index, "strength_index"),
        )
        object.__setattr__(
            self,
            "label_stage",
            _optional_string(self.label_stage, "label_stage"),
        )
        object.__setattr__(self, "replicate", _nonnegative_int(self.replicate, "replicate"))
        policy = SEED_FIELD_POLICIES[self.purpose]
        for field_name in sorted(policy.required_fields):
            if getattr(self, field_name) is None:
                raise ValueError(f"{field_name} must be non-null for seed purpose {self.purpose!r}")
        for field_name in sorted(policy.null_fields):
            if getattr(self, field_name) is not None:
                raise ValueError(f"{field_name} must be null for seed purpose {self.purpose!r}")
        if self.task_type is not None and self.task_type not in policy.task_types:
            raise ValueError(f"task_type is invalid for seed purpose {self.purpose!r}")
        if self.label_stage is not None and self.label_stage not in policy.label_stages:
            raise ValueError(f"label_stage is invalid for seed purpose {self.purpose!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "release_id": self.release_id,
            "purpose": self.purpose,
            "partition": self.partition,
            "panel": self.panel,
            "cell": self.cell,
            "task_type": self.task_type,
            "problem_sha256": self.problem_sha256,
            "state_sha256": self.state_sha256,
            "candidate_index": self.candidate_index,
            "strength_index": self.strength_index,
            "label_stage": self.label_stage,
            "replicate": self.replicate,
        }

    @classmethod
    def from_dict(cls, value: object) -> SeedRequest:
        document = require_exact_keys(value, SEED_REQUEST_FIELDS, "seed request")
        return cls(
            release_id=document["release_id"],
            purpose=document["purpose"],
            partition=document["partition"],
            panel=document["panel"],
            cell=document["cell"],
            task_type=document["task_type"],
            problem_sha256=document["problem_sha256"],
            state_sha256=document["state_sha256"],
            candidate_index=document["candidate_index"],
            strength_index=document["strength_index"],
            label_stage=document["label_stage"],
            replicate=document["replicate"],
        )

    def canonical_preimage(self) -> bytes:
        """Return the exact UTF-8 preimage recorded by the seed registry."""

        return canonical_bytes(self.to_dict())

    @property
    def seed_key(self) -> bytes:
        """Return the raw 32-byte SHA-256 key used by PCG64DXSM seeding."""

        return hashlib.sha256(self.canonical_preimage()).digest()

    @property
    def seed_key_hex(self) -> str:
        return self.seed_key.hex()

    @property
    def seed_words(self) -> tuple[int, int, int, int, int, int, int, int]:
        """Return the eight unsigned big-endian words passed to SeedSequence."""

        key = self.seed_key
        words = tuple(int.from_bytes(key[offset : offset + 4], "big") for offset in range(0, 32, 4))
        return cast(tuple[int, int, int, int, int, int, int, int], words)


@dataclass(frozen=True, slots=True)
class SeedRegistration:
    """Plan one request as native-PCG or third-party-32 before execution."""

    request: SeedRequest
    seed32_required: bool

    def __post_init__(self) -> None:
        if type(self.request) is not SeedRequest:
            raise TypeError("seed registration request must be a SeedRequest")
        if type(self.seed32_required) is not bool:
            raise ValueError("seed32_required must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "request": self.request.to_dict(),
            "seed32_required": self.seed32_required,
        }

    @classmethod
    def from_dict(cls, value: object) -> SeedRegistration:
        document = require_exact_keys(value, SEED_REGISTRATION_FIELDS, "seed registration")
        return cls(
            request=SeedRequest.from_dict(document["request"]),
            seed32_required=document["seed32_required"],
        )


def _uint32(value: object, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result >= _UINT32_LIMIT:
        raise ValueError(f"{name} must fit in an unsigned 32-bit integer")
    return result


def _optional_uint32(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _uint32(value, name)


def _seed_key_bytes(value: object) -> bytes:
    if type(value) is not bytes:
        raise TypeError("seed keys must be exact bytes")
    if len(value) != 32:
        raise ValueError("seed keys must contain exactly 32 bytes")
    return value


@dataclass(frozen=True, slots=True)
class Seed32Allocation:
    """One deterministic third-party seed assignment."""

    seed_key: bytes
    seed32: int
    collision_counter: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "seed_key", _seed_key_bytes(self.seed_key))
        object.__setattr__(self, "seed32", _uint32(self.seed32, "seed32"))
        object.__setattr__(
            self,
            "collision_counter",
            _uint32(self.collision_counter, "collision_counter"),
        )


def _next_seed32(seed_key: bytes, occupied: set[int]) -> tuple[int, int]:
    direct = int.from_bytes(seed_key[:4], "big")
    if direct not in occupied:
        return direct, 0
    for counter in range(1, _UINT32_LIMIT):
        candidate = int.from_bytes(
            hashlib.sha256(seed_key + counter.to_bytes(4, "big")).digest()[:4],
            "big",
        )
        if candidate not in occupied:
            return candidate, counter
    raise RuntimeError("the 32-bit seed namespace is exhausted")  # pragma: no cover


def allocate_seed32(
    seed_keys: Collection[bytes],
    *,
    occupied_seed32: Collection[int] = (),
) -> tuple[Seed32Allocation, ...]:
    """Allocate new 32-bit seeds without changing inherited assignments."""

    keys = tuple(_seed_key_bytes(key) for key in seed_keys)
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate seed key")
    inherited = tuple(_uint32(value, "occupied seed32") for value in occupied_seed32)
    if len(inherited) != len(set(inherited)):
        raise ValueError("duplicate occupied seed32")
    used = set(inherited)
    allocations: list[Seed32Allocation] = []
    for key in sorted(keys):
        seed32, collision_counter = _next_seed32(key, used)
        used.add(seed32)
        allocations.append(
            Seed32Allocation(
                seed_key=key,
                seed32=seed32,
                collision_counter=collision_counter,
            )
        )
    return tuple(allocations)


@dataclass(frozen=True, slots=True)
class SeedRegistryEntry:
    """Persisted derivations for one canonical seed request."""

    request: SeedRequest
    canonical_preimage: str
    seed_key_sha256: str
    seed_words: tuple[int, int, int, int, int, int, int, int]
    seed32_required: bool
    seed32: int | None
    collision_counter: int | None

    def __post_init__(self) -> None:
        if type(self.request) is not SeedRequest:
            raise TypeError("seed registry request must be a SeedRequest")
        expected_preimage = self.request.canonical_preimage().decode("ascii")
        if type(self.canonical_preimage) is not str or self.canonical_preimage != expected_preimage:
            raise ValueError("seed registry canonical_preimage does not match its request")
        expected_key = self.request.seed_key_hex
        if _sha256(self.seed_key_sha256, "seed_key_sha256") != expected_key:
            raise ValueError("seed registry seed_key_sha256 does not match its request")
        try:
            words = tuple(_uint32(word, "seed word") for word in self.seed_words)
        except TypeError as error:
            raise ValueError("seed_words must contain eight unsigned 32-bit integers") from error
        if len(words) != 8 or words != self.request.seed_words:
            raise ValueError("seed registry seed_words do not match its request")
        object.__setattr__(
            self,
            "seed_words",
            cast(tuple[int, int, int, int, int, int, int, int], words),
        )
        if type(self.seed32_required) is not bool:
            raise ValueError("seed32_required must be a boolean")
        seed32 = _optional_uint32(self.seed32, "seed32")
        collision_counter = _optional_uint32(self.collision_counter, "collision_counter")
        if self.seed32_required:
            if seed32 is None or collision_counter is None:
                raise ValueError("a required 32-bit seed needs seed32 and collision_counter")
        elif seed32 is not None or collision_counter is not None:
            raise ValueError("a PCG-only seed must have null 32-bit allocation fields")
        object.__setattr__(self, "seed32", seed32)
        object.__setattr__(self, "collision_counter", collision_counter)

    @classmethod
    def from_request(
        cls,
        request: SeedRequest,
        allocation: Seed32Allocation | None = None,
    ) -> SeedRegistryEntry:
        if allocation is not None and allocation.seed_key != request.seed_key:
            raise ValueError("seed allocation does not match its request")
        return cls(
            request=request,
            canonical_preimage=request.canonical_preimage().decode("ascii"),
            seed_key_sha256=request.seed_key_hex,
            seed_words=request.seed_words,
            seed32_required=allocation is not None,
            seed32=None if allocation is None else allocation.seed32,
            collision_counter=None if allocation is None else allocation.collision_counter,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "request": self.request.to_dict(),
            "canonical_preimage": self.canonical_preimage,
            "seed_key_sha256": self.seed_key_sha256,
            "seed_words": list(self.seed_words),
            "seed32_required": self.seed32_required,
            "seed32": self.seed32,
            "collision_counter": self.collision_counter,
        }

    @classmethod
    def from_dict(cls, value: object) -> SeedRegistryEntry:
        document = require_exact_keys(value, SEED_REGISTRY_ENTRY_FIELDS, "seed registry entry")
        raw_words = document["seed_words"]
        if type(raw_words) is not list:
            raise ValueError("seed_words must be a JSON array")
        return cls(
            request=SeedRequest.from_dict(document["request"]),
            canonical_preimage=document["canonical_preimage"],
            seed_key_sha256=document["seed_key_sha256"],
            seed_words=tuple(raw_words),
            seed32_required=document["seed32_required"],
            seed32=document["seed32"],
            collision_counter=document["collision_counter"],
        )


@dataclass(frozen=True, slots=True)
class SeedRegistryShard:
    """Digest and key-range descriptor for one canonical JSONL shard."""

    relative_path: str
    sha256: str
    byte_count: int
    record_count: int
    schema_version: int
    first_seed_key_sha256: str
    last_seed_key_sha256: str

    def __post_init__(self) -> None:
        relative_path = _relative_posix_path(self.relative_path)
        if _SEED_SHARD_PATH_PATTERN.fullmatch(relative_path) is None:
            raise ValueError("seed registry shard path must be part-NNNNNN.jsonl")
        first_key = _sha256(self.first_seed_key_sha256, "first_seed_key_sha256")
        last_key = _sha256(self.last_seed_key_sha256, "last_seed_key_sha256")
        if first_key > last_key:
            raise ValueError("seed registry shard key range is reversed")
        schema_version = _positive_int(self.schema_version, "schema_version")
        if schema_version != SEED_REGISTRY_ENTRY_SCHEMA_VERSION:
            raise ValueError(
                f"seed registry shard requires schema_version {SEED_REGISTRY_ENTRY_SCHEMA_VERSION}"
            )
        object.__setattr__(self, "relative_path", relative_path)
        object.__setattr__(self, "sha256", _sha256(self.sha256, "sha256"))
        object.__setattr__(self, "byte_count", _positive_int(self.byte_count, "byte_count"))
        object.__setattr__(
            self,
            "record_count",
            _positive_int(self.record_count, "record_count"),
        )
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "first_seed_key_sha256", first_key)
        object.__setattr__(self, "last_seed_key_sha256", last_key)

    def to_dict(self) -> dict[str, object]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "byte_count": self.byte_count,
            "record_count": self.record_count,
            "schema_version": self.schema_version,
            "first_seed_key_sha256": self.first_seed_key_sha256,
            "last_seed_key_sha256": self.last_seed_key_sha256,
        }

    @classmethod
    def from_dict(cls, value: object) -> SeedRegistryShard:
        document = require_exact_keys(value, SEED_REGISTRY_SHARD_FIELDS, "seed registry shard")
        return cls(
            relative_path=document["relative_path"],
            sha256=document["sha256"],
            byte_count=document["byte_count"],
            record_count=document["record_count"],
            schema_version=document["schema_version"],
            first_seed_key_sha256=document["first_seed_key_sha256"],
            last_seed_key_sha256=document["last_seed_key_sha256"],
        )


def _seed32_inventory_sha256(values: Collection[int]) -> str:
    normalized = tuple(sorted(_uint32(value, "occupied seed32") for value in values))
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate occupied seed32")
    return canonical_sha256(list(normalized))


@dataclass(frozen=True, slots=True)
class SeedRegistryRoot:
    """Small canonical root for a stage's deterministic entry shards."""

    release_id: str
    stage_id: str
    parent_root_sha256: str | None
    entry_count: int
    shard_entry_limit: int
    occupied_seed32_count: int
    occupied_seed32_sha256: str
    shards: tuple[SeedRegistryShard, ...]

    def __post_init__(self) -> None:
        release_id = _nonempty_string(self.release_id, "release_id")
        stage_id = _safe_stage_id(self.stage_id)
        parent_digest = (
            None
            if self.parent_root_sha256 is None
            else _sha256(self.parent_root_sha256, "parent_root_sha256")
        )
        entry_count = _positive_int(self.entry_count, "entry_count")
        shard_entry_limit = _bounded_registry_limit(
            self.shard_entry_limit,
            "shard_entry_limit",
            MAX_SEED_REGISTRY_SHARD_ENTRIES,
        )
        occupied_count = _nonnegative_int(
            self.occupied_seed32_count,
            "occupied_seed32_count",
        )
        occupied_digest = _sha256(
            self.occupied_seed32_sha256,
            "occupied_seed32_sha256",
        )
        shards = tuple(self.shards)
        if not shards or any(type(shard) is not SeedRegistryShard for shard in shards):
            raise ValueError("seed registry root requires SeedRegistryShard values")
        if sum(shard.record_count for shard in shards) != entry_count:
            raise ValueError("seed registry root entry_count does not match its shards")
        previous_last: str | None = None
        for index, shard in enumerate(shards):
            expected_path = f"part-{index:06d}.jsonl"
            if shard.relative_path != expected_path:
                raise ValueError("seed registry shard paths must be consecutive")
            if shard.record_count > shard_entry_limit:
                raise ValueError("seed registry shard exceeds shard_entry_limit")
            if previous_last is not None and shard.first_seed_key_sha256 <= previous_last:
                raise ValueError("seed registry shard key ranges overlap or are unsorted")
            previous_last = shard.last_seed_key_sha256
        object.__setattr__(self, "release_id", release_id)
        object.__setattr__(self, "stage_id", stage_id)
        object.__setattr__(self, "parent_root_sha256", parent_digest)
        object.__setattr__(self, "entry_count", entry_count)
        object.__setattr__(self, "shard_entry_limit", shard_entry_limit)
        object.__setattr__(self, "occupied_seed32_count", occupied_count)
        object.__setattr__(self, "occupied_seed32_sha256", occupied_digest)
        object.__setattr__(self, "shards", shards)
        if len(canonical_bytes(self.to_dict())) > MAX_SEED_REGISTRY_ROOT_BYTES:
            raise ValueError("seed registry root exceeds the 8 MiB byte limit")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SEED_REGISTRY_ROOT_SCHEMA,
            "schema_version": SEED_REGISTRY_ROOT_SCHEMA_VERSION,
            "release_id": self.release_id,
            "stage_id": self.stage_id,
            "parent_root_sha256": self.parent_root_sha256,
            "entry_count": self.entry_count,
            "shard_entry_limit": self.shard_entry_limit,
            "occupied_seed32_count": self.occupied_seed32_count,
            "occupied_seed32_sha256": self.occupied_seed32_sha256,
            "shards": [shard.to_dict() for shard in self.shards],
        }

    @classmethod
    def from_dict(cls, value: object) -> SeedRegistryRoot:
        document = validate_versioned_object(
            value,
            schema=SEED_REGISTRY_ROOT_SCHEMA,
            schema_version=SEED_REGISTRY_ROOT_SCHEMA_VERSION,
            fields=SEED_REGISTRY_ROOT_FIELDS,
            name="seed registry root",
        )
        raw_shards = document["shards"]
        if type(raw_shards) is not list:
            raise ValueError("seed registry root shards must be a JSON array")
        return cls(
            release_id=document["release_id"],
            stage_id=document["stage_id"],
            parent_root_sha256=document["parent_root_sha256"],
            entry_count=document["entry_count"],
            shard_entry_limit=document["shard_entry_limit"],
            occupied_seed32_count=document["occupied_seed32_count"],
            occupied_seed32_sha256=document["occupied_seed32_sha256"],
            shards=tuple(SeedRegistryShard.from_dict(shard) for shard in raw_shards),
        )

    def to_bytes(self) -> bytes:
        payload = canonical_bytes(self.to_dict())
        if len(payload) > MAX_SEED_REGISTRY_ROOT_BYTES:
            raise ValueError("seed registry root exceeds the 8 MiB byte limit")
        return payload

    @classmethod
    def from_bytes(cls, payload: bytes) -> SeedRegistryRoot:
        if type(payload) is not bytes:
            raise TypeError("seed registry root payload must be exact bytes")
        if len(payload) > MAX_SEED_REGISTRY_ROOT_BYTES:
            raise ValueError("seed registry root exceeds the byte limit")
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("seed registry root must be valid UTF-8 JSON") from error
        root = cls.from_dict(document)
        if root.to_bytes() != payload:
            raise ValueError("seed registry root must use canonical bytes")
        return root

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()


def _validate_stage_artifact_against_root(
    stage_artifact: SeedRegistryStageArtifact,
    root: SeedRegistryRoot,
) -> None:
    if type(stage_artifact) is not SeedRegistryStageArtifact:
        raise TypeError("stage_artifact must be a SeedRegistryStageArtifact")
    if (
        stage_artifact.stage_id != root.stage_id
        or stage_artifact.parent_root_sha256 != root.parent_root_sha256
        or stage_artifact.root_sha256 != root.sha256
    ):
        raise ValueError("stage artifact identity does not match the seed-registry root")
    root_payload = root.to_bytes()
    if (
        stage_artifact.root_artifact.byte_count != len(root_payload)
        or stage_artifact.root_artifact.schema_version != SEED_REGISTRY_ROOT_SCHEMA_VERSION
    ):
        raise ValueError("root artifact metadata does not match the seed-registry root")
    if len(stage_artifact.shard_artifacts) != len(root.shards):
        raise ValueError("stage artifact does not list every seed-registry shard")
    for artifact, shard in zip(stage_artifact.shard_artifacts, root.shards, strict=True):
        if (
            artifact.sha256 != shard.sha256
            or artifact.byte_count != shard.byte_count
            or artifact.record_count != shard.record_count
            or artifact.schema_version != shard.schema_version
        ):
            raise ValueError("shard artifact metadata does not match the seed-registry root")


@dataclass(frozen=True, slots=True, init=False)
class SeedRegistryVerification:
    """Internally sealed bounded state produced only by full artifact verification."""

    root: SeedRegistryRoot
    occupied_seed32: frozenset[int]
    _directory: Path = field(repr=False, compare=False)
    _parent: SeedRegistryVerification | None = field(repr=False, compare=False)
    _expected_root_sha256: str = field(repr=False, compare=False)
    _seal: object = field(repr=False, compare=False)

    @classmethod
    def _verified(
        cls,
        *,
        root: SeedRegistryRoot,
        occupied_seed32: Collection[int],
        directory: Path,
        parent: SeedRegistryVerification | None,
        expected_root_sha256: str,
    ) -> SeedRegistryVerification:
        if type(root) is not SeedRegistryRoot:
            raise TypeError("verified seed registry root must be a SeedRegistryRoot")
        expected = _sha256(expected_root_sha256, "expected root SHA-256")
        if root.sha256 != expected:
            raise ValueError("seed registry does not match expected root SHA-256")
        if parent is not None:
            _require_verified_summary(parent)
        if (
            parent is not None
            and type(occupied_seed32) in {set, frozenset}
            and len(occupied_seed32) == len(parent.occupied_seed32)
            and all(_uint32(value, "occupied seed32") == value for value in occupied_seed32)
            and occupied_seed32 == parent.occupied_seed32
        ):
            occupied = parent.occupied_seed32
        else:
            occupied = frozenset(_uint32(value, "occupied seed32") for value in occupied_seed32)
        if len(occupied) != root.occupied_seed32_count:
            raise ValueError("verified occupied seed32 count does not match its root")
        if _seed32_inventory_sha256(occupied) != root.occupied_seed32_sha256:
            raise ValueError("verified occupied seed32 digest does not match its root")
        if not isinstance(directory, Path):
            raise TypeError("verified seed registry directory must be a Path")
        if parent is None:
            if root.parent_root_sha256 is not None:
                raise ValueError("verified seed registry requires its parent")
        else:
            if root.release_id != parent.root.release_id:
                raise ValueError("verified parent must use the same release_id")
            if root.parent_root_sha256 != parent.root.sha256:
                raise ValueError("verified parent digest does not match the root")
            ancestor: SeedRegistryVerification | None = parent
            while ancestor is not None:
                if ancestor.root.stage_id == root.stage_id:
                    raise ValueError("verified stage_id repeats in the ancestor chain")
                ancestor = ancestor._parent
        result = object.__new__(cls)
        object.__setattr__(result, "root", root)
        object.__setattr__(result, "occupied_seed32", occupied)
        object.__setattr__(result, "_directory", directory)
        object.__setattr__(result, "_parent", parent)
        object.__setattr__(result, "_expected_root_sha256", expected)
        object.__setattr__(result, "_seal", _VERIFICATION_SEAL)
        return result


def _require_verified_summary(value: object) -> SeedRegistryVerification:
    if type(value) is not SeedRegistryVerification or value._seal is not _VERIFICATION_SEAL:
        raise TypeError("parent must be an internally verified seed-registry summary")
    if value.root.sha256 != value._expected_root_sha256:
        raise ValueError("verified seed-registry summary lost its expected root binding")
    if (
        len(value.occupied_seed32) != value.root.occupied_seed32_count
        or _seed32_inventory_sha256(value.occupied_seed32) != value.root.occupied_seed32_sha256
    ):
        raise ValueError("verified seed-registry summary lost its occupied-seed binding")
    return value


@dataclass(frozen=True, slots=True)
class SeedRegistry:
    """Test-scale reference model; production artifacts use the sharded API."""

    release_id: str
    stage_id: str
    parent_registry_sha256: str | None
    entries: tuple[SeedRegistryEntry, ...]
    _parent: SeedRegistry | None = field(default=None, repr=False, compare=False)
    _seed_keys: tuple[str, ...] = field(init=False, repr=False, compare=False)
    _occupied_seed32: frozenset[int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        release_id = _nonempty_string(self.release_id, "release_id")
        stage_id = _safe_stage_id(self.stage_id)
        parent_digest = (
            None
            if self.parent_registry_sha256 is None
            else _sha256(self.parent_registry_sha256, "parent_registry_sha256")
        )
        parent = self._parent
        if parent is None:
            if parent_digest is not None:
                raise ValueError("parent seed registry is required by parent_registry_sha256")
            inherited_occupied: frozenset[int] = frozenset()
        else:
            if type(parent) is not SeedRegistry:
                raise TypeError("parent seed registry must be a SeedRegistry")
            if parent.release_id != release_id:
                raise ValueError("parent seed registry must use the same release_id")
            ancestor: SeedRegistry | None = parent
            while ancestor is not None:
                if ancestor.stage_id == stage_id:
                    raise ValueError(
                        "seed registry stage_id is already present in its ancestor chain"
                    )
                ancestor = ancestor._parent
            if parent_digest != parent.sha256:
                raise ValueError("parent_registry_sha256 does not match the parent registry")
            inherited_occupied = parent.occupied_seed32
        entries = tuple(self.entries)
        if not entries:
            raise ValueError("seed registry requires at least one seed request")
        if any(type(entry) is not SeedRegistryEntry for entry in entries):
            raise TypeError("seed registry entries must be SeedRegistryEntry values")
        if any(entry.request.release_id != release_id for entry in entries):
            raise ValueError("seed registry entries must use one release_id")
        requests = tuple(entry.request for entry in entries)
        preimages = tuple(request.canonical_preimage() for request in requests)
        if len(preimages) != len(set(preimages)):
            raise ValueError("duplicate seed request")
        seed_keys = tuple(entry.seed_key_sha256 for entry in entries)
        if seed_keys != tuple(sorted(seed_keys)) or len(seed_keys) != len(set(seed_keys)):
            raise ValueError("seed registry entries must have unique sorted seed keys")
        if parent is not None:
            for entry in entries:
                inherited = parent._entry_for_seed_key(entry.seed_key_sha256)
                if inherited is None:
                    continue
                if inherited.request == entry.request:
                    raise ValueError("seed request is already registered in a parent stage")
                raise RuntimeError("distinct seed requests produced one SHA-256 key")
        allocated_requests = tuple(entry.request for entry in entries if entry.seed32_required)
        expected_allocations = allocate_seed32(
            [request.seed_key for request in allocated_requests],
            occupied_seed32=inherited_occupied,
        )
        expected_by_key = {allocation.seed_key: allocation for allocation in expected_allocations}
        for entry in entries:
            if not entry.seed32_required:
                continue
            expected = expected_by_key[entry.request.seed_key]
            if (
                entry.seed32 != expected.seed32
                or entry.collision_counter != expected.collision_counter
            ):
                raise ValueError("seed registry allocation fields do not match the seed plan")
        object.__setattr__(self, "release_id", release_id)
        object.__setattr__(self, "stage_id", stage_id)
        object.__setattr__(self, "parent_registry_sha256", parent_digest)
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "_seed_keys", seed_keys)
        new_occupied = {cast(int, entry.seed32) for entry in entries if entry.seed32_required}
        object.__setattr__(
            self,
            "_occupied_seed32",
            frozenset((*inherited_occupied, *new_occupied)),
        )

    @classmethod
    def build(
        cls,
        requests: Collection[SeedRequest],
        *,
        stage_id: str,
        parent: SeedRegistry | None = None,
        third_party_32_requests: Collection[SeedRequest] = (),
    ) -> SeedRegistry:
        request_values = tuple(requests)
        if not request_values:
            raise ValueError("seed registry requires at least one seed request")
        if any(type(request) is not SeedRequest for request in request_values):
            raise TypeError("seed registry requires SeedRequest values")
        preimages = tuple(request.canonical_preimage() for request in request_values)
        if len(preimages) != len(set(preimages)):
            raise ValueError("duplicate seed request")
        release_ids = {request.release_id for request in request_values}
        if len(release_ids) != 1:
            raise ValueError("seed registry requests must use one release_id")
        allocated_requests = tuple(third_party_32_requests)
        if any(type(request) is not SeedRequest for request in allocated_requests):
            raise TypeError("third-party seed plan requires SeedRequest values")
        allocated_keys = tuple(request.seed_key for request in allocated_requests)
        if len(allocated_keys) != len(set(allocated_keys)):
            raise ValueError("duplicate third-party seed request")
        request_by_key = {request.seed_key: request for request in request_values}
        for request in allocated_requests:
            planned = request_by_key.get(request.seed_key)
            if planned is None or planned != request:
                raise ValueError("third-party seed request is not in this registry stage")
        inherited_occupied = frozenset() if parent is None else parent.occupied_seed32
        allocations = allocate_seed32(
            allocated_keys,
            occupied_seed32=inherited_occupied,
        )
        allocations_by_key = {allocation.seed_key: allocation for allocation in allocations}
        entries = tuple(
            SeedRegistryEntry.from_request(
                request,
                allocations_by_key.get(request.seed_key),
            )
            for request in sorted(request_values, key=lambda item: item.seed_key)
        )
        return cls(
            release_id=next(iter(release_ids)),
            stage_id=stage_id,
            parent_registry_sha256=None if parent is None else parent.sha256,
            entries=entries,
            _parent=parent,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": SEED_REGISTRY_SCHEMA,
            "schema_version": SEED_REGISTRY_SCHEMA_VERSION,
            "release_id": self.release_id,
            "stage_id": self.stage_id,
            "parent_registry_sha256": self.parent_registry_sha256,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(
        cls,
        value: object,
        *,
        parent: SeedRegistry | None = None,
    ) -> SeedRegistry:
        document = validate_versioned_object(
            value,
            schema=SEED_REGISTRY_SCHEMA,
            schema_version=SEED_REGISTRY_SCHEMA_VERSION,
            fields=SEED_REGISTRY_FIELDS,
            name="seed registry",
        )
        raw_entries = document["entries"]
        if type(raw_entries) is not list:
            raise ValueError("seed registry entries must be a JSON array")
        return cls(
            release_id=document["release_id"],
            stage_id=document["stage_id"],
            parent_registry_sha256=document["parent_registry_sha256"],
            entries=tuple(SeedRegistryEntry.from_dict(entry) for entry in raw_entries),
            _parent=parent,
        )

    def to_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())

    @classmethod
    def from_bytes(
        cls,
        payload: bytes,
        *,
        parent: SeedRegistry | None = None,
    ) -> SeedRegistry:
        if type(payload) is not bytes:
            raise TypeError("seed registry payload must be exact bytes")
        try:
            document = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("seed registry payload must be valid UTF-8 JSON") from error
        registry = cls.from_dict(document, parent=parent)
        if registry.to_bytes() != payload:
            raise ValueError("seed registry payload must use canonical seed-registry bytes")
        return registry

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @property
    def occupied_seed32(self) -> frozenset[int]:
        return self._occupied_seed32

    def _entry_for_seed_key(self, seed_key_hex: str) -> SeedRegistryEntry | None:
        index = bisect_left(self._seed_keys, seed_key_hex)
        if index < len(self.entries) and self._seed_keys[index] == seed_key_hex:
            return self.entries[index]
        if self._parent is None:
            return None
        return self._parent._entry_for_seed_key(seed_key_hex)

    def require(self, request: SeedRequest) -> SeedRegistryEntry:
        if type(request) is not SeedRequest:
            raise TypeError("registered requests must be SeedRequest values")
        entry = self._entry_for_seed_key(request.seed_key_hex)
        if entry is None:
            raise KeyError("seed request is not registered")
        if entry.request != request:
            raise RuntimeError("distinct seed requests produced one SHA-256 key")
        return entry

    def rng_for(self, request: SeedRequest) -> np.random.Generator:
        entry = self.require(request)
        seed_sequence = np.random.SeedSequence(entry.seed_words)
        return np.random.Generator(np.random.PCG64DXSM(seed_sequence))

    def third_party_seed32_for(self, request: SeedRequest) -> int:
        entry = self.require(request)
        if not entry.seed32_required:
            raise ValueError("seed request does not require a 32-bit seed")
        return cast(int, entry.seed32)


def _lexical_absolute_path(value: str | os.PathLike[str]) -> Path:
    raw = os.fspath(value)
    if type(raw) is not str or not raw or "\x00" in raw:
        raise ValueError("seed registry path must be a non-empty text path")
    return Path(os.path.abspath(raw))


@contextmanager
def _open_directory_fd(
    value: str | os.PathLike[str],
    *,
    create: bool = False,
) -> Iterator[int]:
    """Open every path component with O_NOFOLLOW and retain the final directory FD."""

    if not _SECURE_NOFOLLOW_AVAILABLE:
        raise RuntimeError("secure registry I/O requires POSIX O_NOFOLLOW and O_DIRECTORY")
    path = _lexical_absolute_path(value)
    current_fd = os.open(os.sep, _DIRECTORY_OPEN_FLAGS)
    try:
        for component in path.parts[1:]:
            try:
                next_fd = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise ValueError(f"secure directory does not exist: {path}") from None
                with suppress(FileExistsError):
                    os.mkdir(component, mode=0o755, dir_fd=current_fd)
                try:
                    next_fd = os.open(component, _DIRECTORY_OPEN_FLAGS, dir_fd=current_fd)
                except OSError as error:
                    raise ValueError(
                        f"secure directory path rejects a symlink or non-directory: {path}"
                    ) from error
            except OSError as error:
                raise ValueError(
                    f"secure directory path rejects a symlink or non-directory: {path}"
                ) from error
            metadata = os.fstat(next_fd)
            if not stat.S_ISDIR(metadata.st_mode):
                os.close(next_fd)
                raise ValueError(f"secure directory path contains a non-directory: {path}")
            os.close(current_fd)
            current_fd = next_fd
        yield current_fd
    finally:
        os.close(current_fd)


def _open_regular_file_at(directory_fd: int, name: str) -> int:
    """Open one direct child without following links and validate the opened FD."""

    if not _SECURE_NOFOLLOW_AVAILABLE:
        raise RuntimeError("secure registry I/O requires POSIX O_NOFOLLOW and O_DIRECTORY")
    if type(name) is not str or not name or "/" in name or name in {".", ".."}:
        raise ValueError("secure artifact name must be one direct path component")
    try:
        descriptor = os.open(name, _FILE_OPEN_FLAGS, dir_fd=directory_fd)
    except OSError as error:
        raise ValueError(f"artifact {name!r} is missing, linked, or unreadable") from error
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        os.close(descriptor)
        raise ValueError(f"artifact {name!r} must be a regular file")
    return descriptor


def _read_regular_file_at(directory_fd: int, name: str, byte_limit: int) -> bytes:
    descriptor = _open_regular_file_at(directory_fd, name)
    with os.fdopen(descriptor, "rb") as stream:
        payload = stream.read(byte_limit + 1)
    if len(payload) > byte_limit:
        raise ValueError(f"artifact {name!r} exceeds its byte limit")
    return payload


def _verify_directory_inventory(directory_fd: int, expected_names: Collection[str]) -> None:
    remaining = set(expected_names)
    with os.scandir(directory_fd) as entries:
        for entry in entries:
            if entry.name not in remaining:
                raise ValueError(f"seed registry directory has unexpected artifact {entry.name!r}")
            remaining.remove(entry.name)
    if remaining:
        raise ValueError(f"seed registry directory is missing artifacts: {sorted(remaining)!r}")


def _clear_staging_directory_fd(directory_fd: int) -> None:
    """Remove a private staging tree without following a renamed lexical path."""

    with os.scandir(directory_fd) as entries:
        for entry in entries:
            name = entry.name
            metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISDIR(metadata.st_mode):
                os.unlink(name, dir_fd=directory_fd)
                continue
            child_fd = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=directory_fd)
            try:
                child_stat = os.fstat(child_fd)
                child_identity = (child_stat.st_dev, child_stat.st_ino)
                _clear_staging_directory_fd(child_fd)
                current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) != child_identity:
                    raise ValueError("seed registry staging directory changed during cleanup")
                os.rmdir(name, dir_fd=directory_fd)
            finally:
                os.close(child_fd)


def _remove_anchored_staging_directory(
    *,
    parent_fd: int,
    staging_name: str,
    staging_fd: int,
    staging_identity: tuple[int, int],
) -> None:
    _clear_staging_directory_fd(staging_fd)
    try:
        current = os.stat(staging_name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    if (current.st_dev, current.st_ino) != staging_identity:
        raise ValueError("seed registry staging directory changed during cleanup")
    os.rmdir(staging_name, dir_fd=parent_fd)


def _publish_directory_no_replace(
    payload_directory: Path,
    *,
    target_name: str,
    parent_fd: int,
    parent_path: Path,
) -> None:
    """Reserve the destination atomically and publish immutable files with root last."""

    if not target_name or target_name in {".", ".."} or "/" in target_name:
        raise ValueError("seed registry output requires a direct child directory")
    try:
        os.mkdir(target_name, mode=0o755, dir_fd=parent_fd)
    except FileExistsError:
        raise FileExistsError(f"seed registry output already exists: {target_name}") from None
    target_fd: int | None = None
    target_identity: tuple[int, int] | None = None
    parent_stat = os.fstat(parent_fd)
    parent_identity = (parent_stat.st_dev, parent_stat.st_ino)
    published_names: list[str] = []
    try:
        target_fd = os.open(target_name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
        target_stat = os.fstat(target_fd)
        target_identity = (target_stat.st_dev, target_stat.st_ino)
        with _open_directory_fd(payload_directory) as payload_fd:
            names = os.listdir(payload_fd)
            expected_root = [name for name in names if name == SEED_REGISTRY_ROOT_FILENAME]
            shard_names = sorted(name for name in names if _SEED_SHARD_PATH_PATTERN.fullmatch(name))
            if len(expected_root) != 1 or len(names) != len(shard_names) + 1:
                raise ValueError("staged seed registry contains unbound artifacts")
            for name in (*shard_names, SEED_REGISTRY_ROOT_FILENAME):
                source_fd = _open_regular_file_at(payload_fd, name)
                try:
                    source_stat = os.fstat(source_fd)
                    os.fsync(source_fd)
                    os.link(
                        name,
                        name,
                        src_dir_fd=payload_fd,
                        dst_dir_fd=target_fd,
                        follow_symlinks=False,
                    )
                    published_names.append(name)
                    published_fd = _open_regular_file_at(target_fd, name)
                    try:
                        published_stat = os.fstat(published_fd)
                        if (
                            published_stat.st_dev != source_stat.st_dev
                            or published_stat.st_ino != source_stat.st_ino
                        ):
                            raise ValueError("published artifact differs from its opened source")
                    finally:
                        os.close(published_fd)
                finally:
                    os.close(source_fd)
        os.fsync(target_fd)
        os.fsync(parent_fd)
        published_directory = os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
        if (published_directory.st_dev, published_directory.st_ino) != target_identity:
            raise FileExistsError("seed registry publication target changed during publication")
        try:
            with _open_directory_fd(parent_path) as current_parent_fd:
                current_parent_stat = os.fstat(current_parent_fd)
                if (current_parent_stat.st_dev, current_parent_stat.st_ino) != parent_identity:
                    raise ValueError("seed registry publication parent changed during publication")
                visible_target = os.stat(
                    target_name,
                    dir_fd=current_parent_fd,
                    follow_symlinks=False,
                )
                if (visible_target.st_dev, visible_target.st_ino) != target_identity:
                    raise ValueError("seed registry publication parent changed during publication")
        except (OSError, ValueError) as error:
            raise ValueError(
                "seed registry publication parent changed during publication"
            ) from error
    except BaseException:
        if target_fd is not None and target_identity is not None:
            try:
                current = os.stat(target_name, dir_fd=parent_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == target_identity:
                    for name in reversed(published_names):
                        with suppress(FileNotFoundError):
                            os.unlink(name, dir_fd=target_fd)
                    os.rmdir(target_name, dir_fd=parent_fd)
            except OSError:
                pass
        raise
    finally:
        if target_fd is not None:
            os.close(target_fd)


def _write_registration_chunk(
    registrations: list[SeedRegistration],
    directory: Path,
    index: int,
) -> Path:
    registrations.sort(key=lambda registration: registration.request.seed_key)
    path = directory / f"sort-{index:06d}.jsonl"
    with path.open("xb") as stream:
        for registration in registrations:
            payload = canonical_bytes(registration.to_dict())
            if len(payload) > MAX_SEED_REGISTRY_ENTRY_BYTES:
                raise ValueError("seed registration exceeds the line byte limit")
            stream.write(payload)
            stream.write(b"\n")
    return path


def _registration_chunks(
    registrations: Iterable[SeedRegistration],
    directory: Path,
    chunk_entry_limit: int,
) -> tuple[Path, ...]:
    limit = _bounded_registry_limit(
        chunk_entry_limit,
        "sort_chunk_limit",
        MAX_SEED_REGISTRY_SORT_CHUNK_ENTRIES,
    )
    paths: list[Path] = []
    chunk: list[SeedRegistration] = []
    for registration in registrations:
        if type(registration) is not SeedRegistration:
            raise TypeError("seed shard writer requires SeedRegistration values")
        chunk.append(registration)
        if len(chunk) == limit:
            paths.append(_write_registration_chunk(chunk, directory, len(paths)))
            chunk = []
    if chunk:
        paths.append(_write_registration_chunk(chunk, directory, len(paths)))
    if not paths:
        raise ValueError("seed shard writer requires at least one registration")
    return tuple(paths)


def _read_canonical_registration_line(stream: BinaryIO, source: Path) -> SeedRegistration | None:
    line = stream.readline(MAX_SEED_REGISTRY_ENTRY_BYTES + 2)
    if not line:
        return None
    if len(line) > MAX_SEED_REGISTRY_ENTRY_BYTES + 1 or not line.endswith(b"\n"):
        raise ValueError(f"seed registration line is invalid in {source.name}")
    payload = line[:-1]
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"seed registration line is invalid in {source.name}") from error
    registration = SeedRegistration.from_dict(document)
    if canonical_bytes(registration.to_dict()) != payload:
        raise ValueError(f"seed registration line is not canonical in {source.name}")
    return registration


def _merge_registration_chunks(paths: Collection[Path]) -> Iterator[SeedRegistration]:
    with ExitStack() as stack:
        streams = [stack.enter_context(path.open("rb")) for path in paths]
        heap: list[tuple[bytes, int, SeedRegistration]] = []
        for index, (path, stream) in enumerate(zip(paths, streams, strict=True)):
            registration = _read_canonical_registration_line(stream, path)
            if registration is not None:
                heapq.heappush(heap, (registration.request.seed_key, index, registration))
        while heap:
            _, index, registration = heapq.heappop(heap)
            yield registration
            next_registration = _read_canonical_registration_line(streams[index], paths[index])
            if next_registration is not None:
                heapq.heappush(
                    heap,
                    (next_registration.request.seed_key, index, next_registration),
                )


def _root_from_directory_fd(directory_fd: int) -> SeedRegistryRoot:
    payload = _read_regular_file_at(
        directory_fd,
        SEED_REGISTRY_ROOT_FILENAME,
        MAX_SEED_REGISTRY_ROOT_BYTES,
    )
    return SeedRegistryRoot.from_bytes(payload)


def _require_bound_root(
    verification: SeedRegistryVerification,
    directory_fd: int,
) -> None:
    _require_verified_summary(verification)
    actual = _root_from_directory_fd(directory_fd)
    if actual.sha256 != verification._expected_root_sha256 or actual != verification.root:
        raise ValueError("verified seed-registry root changed after validation")
    _verify_directory_inventory(
        directory_fd,
        {
            SEED_REGISTRY_ROOT_FILENAME,
            *(shard.relative_path for shard in verification.root.shards),
        },
    )


def _validated_verified_chain(
    terminal: SeedRegistryVerification,
) -> tuple[SeedRegistryVerification, ...]:
    """Revalidate every root and link before a summary chain becomes trusted state."""

    stages: list[SeedRegistryVerification] = []
    seen_objects: set[int] = set()
    seen_stage_ids: set[str] = set()
    expected_release_id: str | None = None
    child: SeedRegistryVerification | None = None
    current: SeedRegistryVerification | None = terminal
    while current is not None:
        current = _require_verified_summary(current)
        identity = id(current)
        if identity in seen_objects:
            raise ValueError("verified seed-registry parent chain contains a cycle")
        seen_objects.add(identity)
        with _open_directory_fd(current._directory) as directory_fd:
            _require_bound_root(current, directory_fd)
        if expected_release_id is None:
            expected_release_id = current.root.release_id
        elif current.root.release_id != expected_release_id:
            raise ValueError("verified seed-registry parent chain changes release_id")
        if current.root.stage_id in seen_stage_ids:
            raise ValueError("verified seed-registry parent chain repeats a stage_id")
        seen_stage_ids.add(current.root.stage_id)
        if child is not None:
            if child.root.parent_root_sha256 != current.root.sha256:
                raise ValueError("verified seed-registry parent chain has a broken digest link")
            if not current.occupied_seed32 <= child.occupied_seed32:
                raise ValueError("verified seed-registry parent chain loses an occupied seed32")
        stages.append(current)
        child = current
        current = current._parent
    if not stages or stages[-1].root.parent_root_sha256 is not None:
        raise ValueError("verified seed-registry parent chain does not end at a foundation root")
    return tuple(stages)


def _iter_verified_shard_entries_at(
    directory_fd: int,
    shard: SeedRegistryShard,
    *,
    changed_message: str,
) -> Iterator[SeedRegistryEntry]:
    descriptor = _open_regular_file_at(directory_fd, shard.relative_path)
    digest = hashlib.sha256()
    byte_count = 0
    record_count = 0
    first_key: str | None = None
    last_key: str | None = None
    previous_key: str | None = None
    with os.fdopen(descriptor, "rb") as stream:
        while True:
            line = stream.readline(MAX_SEED_REGISTRY_ENTRY_BYTES + 2)
            if not line:
                break
            digest.update(line)
            byte_count += len(line)
            if len(line) > MAX_SEED_REGISTRY_ENTRY_BYTES + 1 or not line.endswith(b"\n"):
                raise ValueError(changed_message)
            try:
                document = json.loads(line[:-1].decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(changed_message) from error
            entry = SeedRegistryEntry.from_dict(document)
            if canonical_bytes(entry.to_dict()) != line[:-1]:
                raise ValueError(changed_message)
            key = entry.seed_key_sha256
            if previous_key is not None and key <= previous_key:
                raise ValueError(changed_message)
            if first_key is None:
                first_key = key
            last_key = key
            previous_key = key
            record_count += 1
            yield entry
    if (
        digest.hexdigest() != shard.sha256
        or byte_count != shard.byte_count
        or record_count != shard.record_count
        or first_key != shard.first_seed_key_sha256
        or last_key != shard.last_seed_key_sha256
    ):
        raise ValueError(changed_message)


def _read_verified_shard_entries(
    verification: SeedRegistryVerification,
    shard_index: int,
) -> tuple[SeedRegistryEntry, ...]:
    _require_verified_summary(verification)
    if type(shard_index) is not int or not 0 <= shard_index < len(verification.root.shards):
        raise IndexError("seed registry shard index is out of range")
    with _open_directory_fd(verification._directory) as directory_fd:
        _require_bound_root(verification, directory_fd)
        return tuple(
            _iter_verified_shard_entries_at(
                directory_fd,
                verification.root.shards[shard_index],
                changed_message="verified seed-registry shard changed after validation",
            )
        )


def _iter_verified_stage_seed_keys(
    verification: SeedRegistryVerification,
) -> Iterator[bytes]:
    _require_verified_summary(verification)
    with _open_directory_fd(verification._directory) as directory_fd:
        _require_bound_root(verification, directory_fd)
        previous_key: str | None = None
        for shard in verification.root.shards:
            for entry in _iter_verified_shard_entries_at(
                directory_fd,
                shard,
                changed_message="verified parent seed-registry shard changed after validation",
            ):
                if previous_key is not None and entry.seed_key_sha256 <= previous_key:
                    raise ValueError("verified parent seed-registry shard changed after validation")
                previous_key = entry.seed_key_sha256
                yield entry.request.seed_key


def _iter_ancestor_seed_keys(
    parent: SeedRegistryVerification | None,
) -> Iterator[bytes]:
    stages: list[SeedRegistryVerification] = []
    current = parent
    while current is not None:
        stages.append(current)
        current = current._parent
    if not stages:
        return
    yield from heapq.merge(*(_iter_verified_stage_seed_keys(stage) for stage in stages))


def _reject_ancestor_registrations(
    registrations: Iterable[SeedRegistration],
    parent: SeedRegistryVerification | None,
) -> Iterator[SeedRegistration]:
    ancestor_keys = _iter_ancestor_seed_keys(parent)
    ancestor_key = next(ancestor_keys, None)
    for registration in registrations:
        key = registration.request.seed_key
        while ancestor_key is not None and ancestor_key < key:
            ancestor_key = next(ancestor_keys, None)
        if ancestor_key == key:
            raise ValueError("seed request is already registered in an ancestor stage")
        yield registration
    for _ in ancestor_keys:
        pass


def _entries_from_sorted_registrations(
    registrations: Iterable[SeedRegistration],
    *,
    release_id: str,
    occupied_seed32: set[int],
) -> Iterator[SeedRegistryEntry]:
    previous_key: bytes | None = None
    for registration in registrations:
        request = registration.request
        if request.release_id != release_id:
            raise ValueError("seed registrations must use one release_id")
        key = request.seed_key
        if previous_key == key:
            raise ValueError("duplicate seed request in registry stage")
        if previous_key is not None and key < previous_key:
            raise ValueError("seed registrations must be sorted by seed key")
        allocation: Seed32Allocation | None = None
        if registration.seed32_required:
            seed32, collision_counter = _next_seed32(key, occupied_seed32)
            allocation = Seed32Allocation(
                seed_key=key,
                seed32=seed32,
                collision_counter=collision_counter,
            )
            occupied_seed32.add(seed32)
        yield SeedRegistryEntry.from_request(request, allocation)
        previous_key = key


def _write_entry_shards(
    entries: Iterator[SeedRegistryEntry],
    directory: Path,
    shard_entry_limit: int,
) -> tuple[tuple[SeedRegistryShard, ...], int]:
    shard_entry_limit = _bounded_registry_limit(
        shard_entry_limit,
        "shard_entry_limit",
        MAX_SEED_REGISTRY_SHARD_ENTRIES,
    )
    descriptors: list[SeedRegistryShard] = []
    total_count = 0
    exhausted = False
    while not exhausted:
        relative_path = f"part-{len(descriptors):06d}.jsonl"
        path = directory / relative_path
        digest = hashlib.sha256()
        byte_count = 0
        count = 0
        first_key: str | None = None
        last_key: str | None = None
        with path.open("xb") as stream:
            for _ in range(shard_entry_limit):
                try:
                    entry = next(entries)
                except StopIteration:
                    exhausted = True
                    break
                payload = canonical_bytes(entry.to_dict())
                if len(payload) > MAX_SEED_REGISTRY_ENTRY_BYTES:
                    raise ValueError("seed registry entry exceeds the line byte limit")
                line = payload + b"\n"
                stream.write(line)
                digest.update(line)
                byte_count += len(line)
                count += 1
                total_count += 1
                if first_key is None:
                    first_key = entry.seed_key_sha256
                last_key = entry.seed_key_sha256
        if count == 0:
            path.unlink()
            break
        descriptors.append(
            SeedRegistryShard(
                relative_path=relative_path,
                sha256=digest.hexdigest(),
                byte_count=byte_count,
                record_count=count,
                schema_version=SEED_REGISTRY_ENTRY_SCHEMA_VERSION,
                first_seed_key_sha256=cast(str, first_key),
                last_seed_key_sha256=cast(str, last_key),
            )
        )
    return tuple(descriptors), total_count


def _validate_parent_root(
    *,
    release_id: str,
    stage_id: str,
    parent: SeedRegistryVerification | None,
) -> tuple[str | None, set[int]]:
    if parent is None:
        return None, set()
    parent = _require_verified_summary(parent)
    if parent.root.release_id != release_id:
        raise ValueError("parent seed-registry root must use the same release_id")
    for ancestor in _validated_verified_chain(parent):
        if ancestor.root.stage_id == stage_id:
            raise ValueError("seed registry stage_id is already present in its ancestor chain")
    return parent.root.sha256, set(parent.occupied_seed32)


def write_seed_registry_shards(
    registrations: Iterable[SeedRegistration],
    *,
    directory: str | os.PathLike[str],
    release_id: str,
    stage_id: str,
    parent: SeedRegistryVerification | None = None,
    shard_entry_limit: int = DEFAULT_SEED_REGISTRY_SHARD_ENTRIES,
    sort_chunk_limit: int = DEFAULT_SEED_REGISTRY_SORT_CHUNK_ENTRIES,
) -> SeedRegistryRoot:
    """Externally sort and atomically write one bounded-memory registry stage."""

    normalized_release = _nonempty_string(release_id, "release_id")
    normalized_stage = _safe_stage_id(stage_id)
    shard_limit = _bounded_registry_limit(
        shard_entry_limit,
        "shard_entry_limit",
        MAX_SEED_REGISTRY_SHARD_ENTRIES,
    )
    sort_limit = _bounded_registry_limit(
        sort_chunk_limit,
        "sort_chunk_limit",
        MAX_SEED_REGISTRY_SORT_CHUNK_ENTRIES,
    )
    parent_digest, occupied = _validate_parent_root(
        release_id=normalized_release,
        stage_id=normalized_stage,
        parent=parent,
    )
    target = _lexical_absolute_path(directory)
    if target.name in {"", ".", ".."}:
        raise ValueError("seed registry output must name a child directory")
    root: SeedRegistryRoot
    with _open_directory_fd(target.parent, create=True) as parent_fd:
        try:
            os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"seed registry output already exists: {target}")
        staging_fd: int | None = None
        staging_name: str | None = None
        staging_identity: tuple[int, int] | None = None
        try:
            with tempfile.TemporaryDirectory(
                prefix=".seed-registry-", dir=target.parent
            ) as temporary:
                work = Path(temporary)
                staging_name = work.name
                try:
                    staging_fd = os.open(
                        staging_name,
                        _DIRECTORY_OPEN_FLAGS,
                        dir_fd=parent_fd,
                    )
                except OSError as error:
                    raise ValueError(
                        "seed registry staging parent changed during creation"
                    ) from error
                staging_stat = os.fstat(staging_fd)
                staging_identity = (staging_stat.st_dev, staging_stat.st_ino)
                sort_directory = work / "sort"
                payload_directory = work / "payload"
                sort_directory.mkdir()
                payload_directory.mkdir()
                chunks = _registration_chunks(registrations, sort_directory, sort_limit)
                entries = _entries_from_sorted_registrations(
                    _reject_ancestor_registrations(_merge_registration_chunks(chunks), parent),
                    release_id=normalized_release,
                    occupied_seed32=occupied,
                )
                shards, entry_count = _write_entry_shards(
                    entries,
                    payload_directory,
                    shard_limit,
                )
                root = SeedRegistryRoot(
                    release_id=normalized_release,
                    stage_id=normalized_stage,
                    parent_root_sha256=parent_digest,
                    entry_count=entry_count,
                    shard_entry_limit=shard_limit,
                    occupied_seed32_count=len(occupied),
                    occupied_seed32_sha256=_seed32_inventory_sha256(occupied),
                    shards=shards,
                )
                (payload_directory / SEED_REGISTRY_ROOT_FILENAME).write_bytes(root.to_bytes())
                _publish_directory_no_replace(
                    payload_directory,
                    target_name=target.name,
                    parent_fd=parent_fd,
                    parent_path=target.parent,
                )
        finally:
            if staging_fd is not None and staging_name is not None and staging_identity is not None:
                try:
                    _remove_anchored_staging_directory(
                        parent_fd=parent_fd,
                        staging_name=staging_name,
                        staging_fd=staging_fd,
                        staging_identity=staging_identity,
                    )
                finally:
                    os.close(staging_fd)
    return root


def verify_seed_registry_shards(
    directory: str | os.PathLike[str],
    *,
    expected_root_sha256: str,
    parent: SeedRegistryVerification | None = None,
    stage_artifact: SeedRegistryStageArtifact | None = None,
) -> SeedRegistryVerification:
    """Verify exact stage bytes against an external digest with bounded retained state."""

    expected_digest = _sha256(expected_root_sha256, "expected root SHA-256")
    root_directory = _lexical_absolute_path(directory)
    with _open_directory_fd(root_directory) as directory_fd:
        root = _root_from_directory_fd(directory_fd)
        if root.sha256 != expected_digest:
            raise ValueError("seed registry does not match expected root SHA-256")
        expected_parent_digest, occupied = _validate_parent_root(
            release_id=root.release_id,
            stage_id=root.stage_id,
            parent=parent,
        )
        if root.parent_root_sha256 != expected_parent_digest:
            if root.parent_root_sha256 is not None and parent is None:
                raise ValueError("parent seed-registry root is required")
            raise ValueError("parent_root_sha256 does not match the parent seed-registry root")
        if stage_artifact is not None:
            _validate_stage_artifact_against_root(stage_artifact, root)
        expected_names = {
            SEED_REGISTRY_ROOT_FILENAME,
            *(shard.relative_path for shard in root.shards),
        }
        _verify_directory_inventory(directory_fd, expected_names)

        total_count = 0
        previous_key: str | None = None
        ancestor_keys = _iter_ancestor_seed_keys(parent)
        ancestor_key = next(ancestor_keys, None)
        for shard in root.shards:
            for entry in _iter_verified_shard_entries_at(
                directory_fd,
                shard,
                changed_message="seed registry shard bytes do not match their root descriptor",
            ):
                if entry.request.release_id != root.release_id:
                    raise ValueError("seed registry shard entry uses a different release_id")
                key = entry.seed_key_sha256
                if previous_key is not None and key <= previous_key:
                    raise ValueError("seed registry shard entries are duplicate or unsorted")
                raw_key = entry.request.seed_key
                while ancestor_key is not None and ancestor_key < raw_key:
                    ancestor_key = next(ancestor_keys, None)
                if ancestor_key == raw_key:
                    raise ValueError("seed request is already registered in an ancestor stage")
                if entry.seed32_required:
                    expected_seed32, expected_counter = _next_seed32(raw_key, occupied)
                    if (
                        entry.seed32 != expected_seed32
                        or entry.collision_counter != expected_counter
                    ):
                        raise ValueError(
                            "seed registry shard allocation does not match the stage plan"
                        )
                    occupied.add(expected_seed32)
                previous_key = key
                total_count += 1
        if total_count != root.entry_count:
            raise ValueError("seed registry entry count does not match its root")
        for _ in ancestor_keys:
            pass
        if len(occupied) != root.occupied_seed32_count:
            raise ValueError("occupied seed32 count does not match the registry root")
        if _seed32_inventory_sha256(occupied) != root.occupied_seed32_sha256:
            raise ValueError("occupied seed32 digest does not match the registry root")
    return SeedRegistryVerification._verified(
        root=root,
        occupied_seed32=occupied,
        directory=root_directory,
        parent=parent,
        expected_root_sha256=expected_digest,
    )


def verify_seed_registry_artifact_chain(
    value: object,
    *,
    artifact_root: str | os.PathLike[str],
    expected_manifest_sha256: str,
    expected_terminal_root_sha256: str,
) -> SeedRegistryVerification:
    """Verify a manifest's complete ordered root/shard chain from committed bytes."""

    manifest = validate_seed_registry_artifact_manifest(
        value,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_terminal_root_sha256=expected_terminal_root_sha256,
    )
    base = _lexical_absolute_path(artifact_root)
    parent: SeedRegistryVerification | None = None
    for stage in manifest.stages:
        parent = verify_seed_registry_shards(
            base / "seed_registry" / stage.stage_id,
            expected_root_sha256=stage.root_sha256,
            parent=parent,
            stage_artifact=stage,
        )
        if parent.root.release_id != manifest.release_id:
            raise ValueError("seed registry stage release_id does not match its artifact manifest")
    return cast(SeedRegistryVerification, parent)


class VerifiedSeedResolver:
    """Resolve requests through a sealed chain with a bounded verified-shard LRU."""

    def __init__(
        self,
        verification: SeedRegistryVerification,
        *,
        expected_terminal_root_sha256: str,
        max_cached_shards: int = 4,
    ) -> None:
        terminal = _require_verified_summary(verification)
        expected = _sha256(
            expected_terminal_root_sha256,
            "expected terminal root SHA-256",
        )
        if terminal.root.sha256 != expected:
            raise ValueError("verification does not match expected terminal root SHA-256")
        maximum = _bounded_registry_limit(max_cached_shards, "max_cached_shards", 64)
        self._stages = _validated_verified_chain(terminal)
        self._last_keys = tuple(
            tuple(shard.last_seed_key_sha256 for shard in stage.root.shards)
            for stage in self._stages
        )
        self._max_cached_shards = maximum
        self._terminal_root_sha256 = expected
        self._cache: OrderedDict[
            tuple[str, int],
            dict[str, SeedRegistryEntry],
        ] = OrderedDict()

    def _candidate_shard(self, stage_index: int, seed_key_hex: str) -> int | None:
        stage = self._stages[stage_index]
        shard_index = bisect_left(self._last_keys[stage_index], seed_key_hex)
        if shard_index == len(stage.root.shards):
            return None
        shard = stage.root.shards[shard_index]
        if seed_key_hex < shard.first_seed_key_sha256:
            return None
        return shard_index

    def _entries(self, stage_index: int, shard_index: int) -> dict[str, SeedRegistryEntry]:
        stage = self._stages[stage_index]
        cache_key = (stage.root.sha256, shard_index)
        cached = self._cache.pop(cache_key, None)
        if cached is not None:
            self._cache[cache_key] = cached
            return cached
        entries = _read_verified_shard_entries(stage, shard_index)
        indexed = {entry.seed_key_sha256: entry for entry in entries}
        if len(indexed) != len(entries):
            raise ValueError("verified shard contains a duplicate seed key")
        self._cache[cache_key] = indexed
        if len(self._cache) > self._max_cached_shards:
            self._cache.popitem(last=False)
        return indexed

    def resolve_many(self, requests: Iterable[SeedRequest]) -> tuple[SeedRegistryEntry, ...]:
        """Resolve a batch, grouping all keys that share one stage shard."""

        planned = tuple(requests)
        if any(type(request) is not SeedRequest for request in planned):
            raise TypeError("registered requests must be SeedRequest values")
        if not planned:
            return ()
        resolved: list[SeedRegistryEntry | None] = [None] * len(planned)
        unresolved = list(range(len(planned)))
        for stage_index, _stage in enumerate(self._stages):
            groups: dict[int, list[int]] = {}
            next_unresolved: list[int] = []
            for request_index in unresolved:
                shard_index = self._candidate_shard(
                    stage_index,
                    planned[request_index].seed_key_hex,
                )
                if shard_index is None:
                    next_unresolved.append(request_index)
                else:
                    groups.setdefault(shard_index, []).append(request_index)
            for shard_index, request_indices in groups.items():
                entries = self._entries(stage_index, shard_index)
                for request_index in request_indices:
                    request = planned[request_index]
                    entry = entries.get(request.seed_key_hex)
                    if entry is None:
                        next_unresolved.append(request_index)
                    elif entry.request != request:
                        raise RuntimeError("distinct seed requests produced one SHA-256 key")
                    else:
                        resolved[request_index] = entry
            unresolved = next_unresolved
            if not unresolved:
                break
        if unresolved:
            raise KeyError("seed request is not registered")
        return cast(tuple[SeedRegistryEntry, ...], tuple(resolved))

    def resolve(self, request: SeedRequest) -> SeedRegistryEntry:
        return self.resolve_many((request,))[0]


def validate_seed_resolver(
    value: object,
    *,
    expected_terminal_root_sha256: str,
) -> VerifiedSeedResolver:
    """Replay a resolver's verified chain against an independent terminal-root commitment."""

    if type(value) is not VerifiedSeedResolver:
        raise TypeError("resolver must be an exact VerifiedSeedResolver")
    expected = _sha256(
        expected_terminal_root_sha256,
        "expected terminal root SHA-256",
    )
    if value._terminal_root_sha256 != expected:
        raise ValueError("seed resolver does not match its external terminal-root commitment")
    if type(value._stages) is not tuple or not value._stages:
        raise ValueError("seed resolver has an invalid verified stage chain")
    # _validated_verified_chain stores the terminal stage first so lookups
    # prefer the newest registration before walking toward the foundation.
    terminal = _require_verified_summary(value._stages[0])
    replayed_stages = _validated_verified_chain(terminal)
    if len(replayed_stages) != len(value._stages) or any(
        replayed is not retained
        for replayed, retained in zip(replayed_stages, value._stages, strict=True)
    ):
        raise ValueError("seed resolver verified stage chain was replaced")
    if terminal.root.sha256 != expected:
        raise ValueError("seed resolver does not match its external terminal-root commitment")
    expected_last_keys = tuple(
        tuple(shard.last_seed_key_sha256 for shard in stage.root.shards)
        for stage in replayed_stages
    )
    if value._last_keys != expected_last_keys:
        raise ValueError("seed resolver shard index disagrees with its verified stage chain")
    _bounded_registry_limit(value._max_cached_shards, "max_cached_shards", 64)
    if type(value._cache) is not OrderedDict:
        raise ValueError("seed resolver cache has an invalid representation")
    value._cache.clear()
    return value


def require_sharded_seed_request(
    resolver: VerifiedSeedResolver,
    request: SeedRequest,
) -> SeedRegistryEntry:
    """Resolve one request through an explicitly reusable verified resolver."""

    if type(resolver) is not VerifiedSeedResolver:
        raise TypeError("resolver must be a VerifiedSeedResolver")
    return resolver.resolve(request)


def rng_for_sharded_seed_request(
    resolver: VerifiedSeedResolver,
    request: SeedRequest,
) -> np.random.Generator:
    entry = require_sharded_seed_request(resolver, request)
    return np.random.Generator(np.random.PCG64DXSM(np.random.SeedSequence(entry.seed_words)))


def third_party_seed32_for_sharded_seed_request(
    resolver: VerifiedSeedResolver,
    request: SeedRequest,
) -> int:
    entry = require_sharded_seed_request(resolver, request)
    if not entry.seed32_required:
        raise ValueError("seed request does not require a 32-bit seed")
    return cast(int, entry.seed32)


__all__ = [
    "ARTIFACT_ENTRY_FIELDS",
    "SEED_FIELD_POLICIES",
    "SEED_REGISTRY_ARTIFACT_MANIFEST_SCHEMA",
    "SEED_OPTION_FIELDS",
    "SEED_PURPOSES",
    "SEED_REGISTRY_ROOT_FILENAME",
    "SEED_REGISTRY_STAGE_ARTIFACT_SCHEMA",
    "SEED_REQUEST_FIELDS",
    "ArtifactEntry",
    "ReleaseIdentity",
    "Seed32Allocation",
    "SeedFieldPolicy",
    "SeedPurpose",
    "SeedRegistration",
    "SeedRegistryArtifactManifest",
    "SeedRegistryEntry",
    "SeedRegistryRoot",
    "SeedRegistryShard",
    "SeedRegistryStageArtifact",
    "SeedRegistryVerification",
    "SeedRequest",
    "VerifiedSeedResolver",
    "allocate_seed32",
    "canonical_bytes",
    "canonical_sha256",
    "canonical_value",
    "require_exact_keys",
    "require_sharded_seed_request",
    "rng_for_sharded_seed_request",
    "third_party_seed32_for_sharded_seed_request",
    "validate_seed_resolver",
    "validate_versioned_object",
    "validate_seed_registry_artifact_manifest",
    "verify_seed_registry_shards",
    "verify_seed_registry_artifact_chain",
    "write_seed_registry_shards",
]
