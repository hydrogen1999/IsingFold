"""Atomic, authenticated I/O for independently generated IsingFold corpus shards.

The on-disk boundary is intentionally stricter than an ordinary JSON dump.  A
shard is rehydrated through :class:`CorpusShard` before publication, every JSON
document uses the package canonical encoding, and every regular artifact except
the checksum table itself is committed by ``SHA256SUMS``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from embedbench.candidate_bank import canonical_json_bytes, content_digest, write_bank
from embedbench.isingfold_corpus_shard import CorpusShard, GeneratedLineage, LineageRequest

SHARD_FILENAME = "corpus_shard.json"
SHARD_MANIFEST_FILENAME = "shard_manifest.json"
CHECKSUM_FILENAME = "SHA256SUMS"
SHARD_MANIFEST_SCHEMA = "embedbench.isingfold-corpus-shard-manifest"
SCHEMA_VERSION = 1
CANDIDATE_BANK_FILENAME = "candidate_bank_v2.jsonl"
CANDIDATE_BANK_MANIFEST_FILENAME = "candidate_bank_v2.manifest.json"
LINEAGE_FACTS_FILENAME = "lineage_facts.jsonl"
TASK_FACTS_FILENAME = "task_facts.jsonl"
QUALITY_FILENAME = "quality_isingfold_exact.jsonl"
QUALITY_MANIFEST_FILENAME = f"{QUALITY_FILENAME}.manifest.json"
RELEASE_MANIFEST_FILENAME = "release_manifest.json"
RELEASE_MANIFEST_SCHEMA = "embedbench.isingfold-corpus-release"
QUALITY_MANIFEST_SCHEMA = "embedbench.isingfold-exact-quality-manifest"
LINEAGE_FACT_SCHEMA = "embedbench.isingfold-lineage-fact"
TASK_FACT_SCHEMA = "embedbench.isingfold-task-fact"
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class CorpusShardWriteReceipt:
    """Content identities returned after one shard directory is durable."""

    output_directory: Path
    shard_path: Path
    shard_manifest_path: Path
    checksums_path: Path
    shard_sha256: str
    shard_manifest_sha256: str
    shard_record_digest: str
    shard_index: int
    shard_count: int
    lineage_count: int


@dataclass(frozen=True, slots=True)
class CorpusMergeReceipt:
    """Authenticated artifact paths and census for a merged source release."""

    output_directory: Path
    candidate_bank_path: Path
    candidate_bank_manifest_path: Path
    lineage_facts_path: Path
    task_facts_path: Path
    quality_path: Path
    quality_manifest_path: Path
    release_manifest_path: Path
    checksums_path: Path
    release_manifest_sha256: str
    release_manifest_record_digest: str
    candidate_bank_manifest_record_digest: str
    shard_count: int
    lineage_count: int


@dataclass(frozen=True, slots=True)
class _LoadedShard:
    shard: CorpusShard
    shard_sha256: str
    manifest_sha256: str
    manifest_record_digest: str


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _record(payload: dict[str, Any]) -> dict[str, Any]:
    return {**payload, "record_digest": content_digest(payload)}


def _canonical_document(value: object) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _strict_json_document(raw: bytes, name: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise ValueError(f"{name} contains non-finite number {token}")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"{name} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain one JSON object")
    if raw != _canonical_document(value):
        raise ValueError(f"{name} is not canonical JSON with one terminal newline")
    return value


def _exact_mapping(value: object, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be an exact object")
    actual = set(value)
    if actual != fields:
        raise ValueError(
            f"{name} schema fields differ: missing={sorted(fields - actual)}, "
            f"unknown={sorted(actual - fields)}"
        )
    return dict(value)


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _read_regular(path: Path, name: str) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        raise ValueError(f"{name} is missing") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{name} must be a regular file, not a symlink or directory")
    return path.read_bytes()


def _write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _publish_directory(destination: Path, files: dict[str, bytes]) -> None:
    if _path_exists(destination):
        raise FileExistsError(f"corpus output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        for filename, raw in sorted(files.items()):
            _write_new(temporary / filename, raw)
        _fsync_directory(temporary)
        if _path_exists(destination):
            raise FileExistsError(f"corpus output already exists: {destination}")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _require_unique(values: Iterable[str], name: str) -> None:
    observed = tuple(values)
    if len(observed) != len(set(observed)):
        raise ValueError(f"corpus shards repeat {name}")


def _validate_unique_lineages(lineages: Sequence[GeneratedLineage]) -> None:
    _require_unique((row.request.lineage_id for row in lineages), "lineage_id")
    _require_unique(
        (row.lineage_fact.base_lineage_key for row in lineages),
        "base lineage identity",
    )
    _require_unique((row.instance.split_unit_id for row in lineages), "split_unit_id")
    _require_unique((row.instance.instance_id for row in lineages), "instance_id")
    _require_unique((row.group.group_id for row in lineages), "group_id")
    _require_unique(
        (row.reference_source.problem_digest for row in lineages),
        "problem identity",
    )


def _attempt_accounting(lineages: Sequence[GeneratedLineage]) -> dict[str, Any]:
    attempts = [attempt for row in lineages for attempt in row.group.attempts]
    status_counts = Counter(attempt.status for attempt in attempts)
    reason_counts = Counter(attempt.reason for attempt in attempts)
    return {
        "attempt_reason_counts": [
            {"count": reason_counts[reason], "reason": reason}
            for reason in sorted(
                reason_counts,
                key=lambda value: (value is not None, "" if value is None else value),
            )
        ],
        "attempt_status_counts": {
            status: status_counts[status]
            for status in ("duplicate", "no_change", "repair_failed", "valid")
        },
        "attempts": len(attempts),
        "candidate_alternatives": sum(len(row.group.candidates) for row in lineages),
        "incumbents": len(lineages),
        "transitions": sum(attempt.transitions for attempt in attempts),
    }


def _shard_manifest(shard: CorpusShard, shard_raw: bytes) -> dict[str, Any]:
    payload = {
        "accounting": _attempt_accounting(shard.lineages),
        "artifact": {
            "byte_count": len(shard_raw),
            "path": SHARD_FILENAME,
            "sha256": _sha256_bytes(shard_raw),
        },
        "counts": {
            "candidate_groups": len(shard.lineages),
            "instances": len(shard.lineages),
            "lineages": len(shard.lineages),
            "problems": len(shard.lineages),
            "split_units": len(shard.lineages),
        },
        "root_seed": shard.config.root_seed,
        "schema": SHARD_MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "shard_count": shard.shard_count,
        "shard_index": shard.shard_index,
        "shard_record_digest": shard.record_digest,
    }
    return _record(payload)


def _checksum_bytes(files: dict[str, bytes]) -> bytes:
    return "".join(
        f"{_sha256_bytes(files[filename])}  {filename}\n" for filename in sorted(files)
    ).encode("utf-8")


def _validate_shard_manifest(
    value: Mapping[str, Any],
    *,
    shard: CorpusShard,
    shard_raw: bytes,
) -> str:
    document = _exact_mapping(
        value,
        {
            "accounting",
            "artifact",
            "counts",
            "record_digest",
            "root_seed",
            "schema",
            "schema_version",
            "shard_count",
            "shard_index",
            "shard_record_digest",
        },
        "shard manifest",
    )
    if document["schema"] != SHARD_MANIFEST_SCHEMA or document["schema_version"] != 1:
        raise ValueError("unsupported shard manifest schema")
    digest = _require_sha256(document["record_digest"], "shard manifest record digest")
    if digest != content_digest(
        {key: item for key, item in document.items() if key != "record_digest"}
    ):
        raise ValueError("shard manifest record digest mismatch")
    artifact = _exact_mapping(
        document["artifact"], {"byte_count", "path", "sha256"}, "shard artifact"
    )
    expected_artifact = {
        "byte_count": len(shard_raw),
        "path": SHARD_FILENAME,
        "sha256": _sha256_bytes(shard_raw),
    }
    if artifact != expected_artifact:
        raise ValueError("shard manifest artifact identity mismatch")
    expected_counts = {
        "candidate_groups": len(shard.lineages),
        "instances": len(shard.lineages),
        "lineages": len(shard.lineages),
        "problems": len(shard.lineages),
        "split_units": len(shard.lineages),
    }
    if document["counts"] != expected_counts:
        raise ValueError("shard manifest counts mismatch")
    if document["accounting"] != _attempt_accounting(shard.lineages):
        raise ValueError("shard manifest attempt accounting mismatch")
    if (
        document["root_seed"] != shard.config.root_seed
        or document["shard_count"] != shard.shard_count
        or document["shard_index"] != shard.shard_index
        or document["shard_record_digest"] != shard.record_digest
    ):
        raise ValueError("shard manifest coordinates or record identity mismatch")
    if canonical_json_bytes(document) != canonical_json_bytes(
        _shard_manifest(shard, shard_raw)
    ):
        raise ValueError("shard manifest does not exactly reproduce derived accounting")
    return digest


def _load_corpus_shard(directory: Path) -> _LoadedShard:
    try:
        metadata = directory.lstat()
    except FileNotFoundError:
        raise ValueError(f"shard directory is missing: {directory}") from None
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"shard input must be a real directory: {directory}")
    expected = {SHARD_FILENAME, SHARD_MANIFEST_FILENAME, CHECKSUM_FILENAME}
    entries = {path.name for path in directory.iterdir()}
    if entries != expected:
        raise ValueError(
            "shard directory inventory differs: "
            f"missing={sorted(expected - entries)}, unknown={sorted(entries - expected)}"
        )
    shard_raw = _read_regular(directory / SHARD_FILENAME, "corpus shard")
    manifest_raw = _read_regular(directory / SHARD_MANIFEST_FILENAME, "shard manifest")
    checksum_raw = _read_regular(directory / CHECKSUM_FILENAME, "shard SHA256SUMS")
    authenticated = {
        SHARD_FILENAME: shard_raw,
        SHARD_MANIFEST_FILENAME: manifest_raw,
    }
    if checksum_raw != _checksum_bytes(authenticated):
        raise ValueError("shard SHA256SUMS is noncanonical or has a checksum mismatch")
    shard_document = _strict_json_document(shard_raw, "corpus shard")
    shard = CorpusShard.from_dict(shard_document)
    manifest = _strict_json_document(manifest_raw, "shard manifest")
    manifest_record_digest = _validate_shard_manifest(
        manifest,
        shard=shard,
        shard_raw=shard_raw,
    )
    _validate_unique_lineages(shard.lineages)
    return _LoadedShard(
        shard=shard,
        shard_sha256=_sha256_bytes(shard_raw),
        manifest_sha256=_sha256_bytes(manifest_raw),
        manifest_record_digest=manifest_record_digest,
    )


def _fact_row(schema: str, fact: object) -> dict[str, Any]:
    payload = {
        "fact": asdict(fact),
        "schema": schema,
        "schema_version": SCHEMA_VERSION,
    }
    return _record(payload)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _candidate_bank_artifacts(
    lineages: Sequence[GeneratedLineage],
    *,
    parent: Path,
) -> tuple[bytes, bytes, str]:
    temporary = Path(tempfile.mkdtemp(prefix=".isingfold-bank-", dir=parent))
    try:
        bank_path = temporary / CANDIDATE_BANK_FILENAME
        manifest = write_bank(
            bank_path,
            instances=[row.instance for row in lineages],
            groups=[row.group for row in lineages],
        )
        return (
            bank_path.read_bytes(),
            _canonical_document(asdict(manifest)),
            manifest.record_digest,
        )
    finally:
        shutil.rmtree(temporary)


def _artifact_inventory(files: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        filename: {
            "byte_count": len(files[filename]),
            "sha256": _sha256_bytes(files[filename]),
        }
        for filename in sorted(files)
    }


def _generation_config(shard: CorpusShard) -> dict[str, Any]:
    result = asdict(shard.config)
    result.pop("shard_index")
    return result


def write_corpus_shard(
    shard: CorpusShard,
    output_dir: str | os.PathLike[str],
) -> CorpusShardWriteReceipt:
    """Validate and atomically publish one self-authenticating corpus shard."""

    if not isinstance(shard, CorpusShard):
        raise TypeError("shard must be a CorpusShard")
    validated = CorpusShard.from_dict(shard.to_dict())
    _validate_unique_lineages(validated.lineages)
    shard_raw = _canonical_document(validated.to_dict())
    manifest_raw = _canonical_document(_shard_manifest(validated, shard_raw))
    authenticated = {
        SHARD_FILENAME: shard_raw,
        SHARD_MANIFEST_FILENAME: manifest_raw,
    }
    files = {**authenticated, CHECKSUM_FILENAME: _checksum_bytes(authenticated)}
    destination = Path(output_dir)
    _publish_directory(destination, files)
    root = destination.resolve(strict=True)
    return CorpusShardWriteReceipt(
        output_directory=root,
        shard_path=root / SHARD_FILENAME,
        shard_manifest_path=root / SHARD_MANIFEST_FILENAME,
        checksums_path=root / CHECKSUM_FILENAME,
        shard_sha256=_sha256_bytes(shard_raw),
        shard_manifest_sha256=_sha256_bytes(manifest_raw),
        shard_record_digest=validated.record_digest,
        shard_index=validated.shard_index,
        shard_count=validated.shard_count,
        lineage_count=len(validated.lineages),
    )


def merge_corpus_shards(
    shard_dirs: Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    *,
    expected_shard_count: int,
    expected_plan_record_digest: str | None = None,
    expected_lineage_requests: Sequence[LineageRequest] | None = None,
) -> CorpusMergeReceipt:
    """Validate complete shard coverage and atomically publish one source release.

    Input directory order has no effect on the emitted bytes.  The merge refuses
    partial coverage, mixed generation configurations, duplicate scientific units,
    unauthenticated bytes, noncanonical JSON, symlinks, and unexpected shard files.
    """

    if type(expected_shard_count) is not int or expected_shard_count <= 0:
        raise ValueError("expected_shard_count must be a positive integer")
    if isinstance(shard_dirs, (str, bytes)) or not isinstance(shard_dirs, Sequence):
        raise TypeError("shard_dirs must be a sequence of paths")
    paths = tuple(Path(path) for path in shard_dirs)
    if not paths:
        raise ValueError("at least one shard directory is required")
    loaded = tuple(_load_corpus_shard(path) for path in paths)
    indices = [item.shard.shard_index for item in loaded]
    expected_indices = list(range(expected_shard_count))
    if len(indices) != expected_shard_count or sorted(indices) != expected_indices:
        raise ValueError(
            "shard-index coverage must be exact: "
            f"expected={expected_indices}, observed={sorted(indices)}"
        )
    ordered = tuple(sorted(loaded, key=lambda item: item.shard.shard_index))
    if any(item.shard.shard_count != expected_shard_count for item in ordered):
        raise ValueError("a shard declares a different shard_count")
    generation_config = _generation_config(ordered[0].shard)
    if any(_generation_config(item.shard) != generation_config for item in ordered[1:]):
        raise ValueError("corpus shards use different generation configurations")

    lineages = tuple(
        sorted(
            (row for item in ordered for row in item.shard.lineages),
            key=lambda row: row.request.lineage_id,
        )
    )
    _validate_unique_lineages(lineages)
    if (expected_plan_record_digest is None) != (expected_lineage_requests is None):
        raise ValueError(
            "prospective plan digest and lineage requests must be supplied together"
        )
    source_plan: dict[str, object] | None = None
    if expected_plan_record_digest is not None and expected_lineage_requests is not None:
        plan_digest = _require_sha256(
            expected_plan_record_digest,
            "prospective plan record digest",
        )
        if isinstance(expected_lineage_requests, (str, bytes)) or not isinstance(
            expected_lineage_requests, Sequence
        ):
            raise TypeError("expected_lineage_requests must be a sequence")
        planned = tuple(expected_lineage_requests)
        if not planned or any(not isinstance(item, LineageRequest) for item in planned):
            raise TypeError("prospective plan must contain LineageRequest values")
        planned_by_id = {item.lineage_id: item for item in planned}
        if len(planned_by_id) != len(planned):
            raise ValueError("prospective plan repeats a lineage request")
        actual_by_id = {item.request.lineage_id: item.request for item in lineages}
        if set(actual_by_id) != set(planned_by_id) or any(
            canonical_json_bytes(asdict(actual_by_id[lineage_id]))
            != canonical_json_bytes(asdict(planned_by_id[lineage_id]))
            for lineage_id in set(actual_by_id) & set(planned_by_id)
        ):
            raise ValueError("merged shard census differs from the prospective plan")
        source_plan = {
            "lineage_request_count": len(planned),
            "record_digest": plan_digest,
        }

    destination = Path(output_dir)
    if _path_exists(destination):
        raise FileExistsError(f"corpus output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    bank_raw, bank_manifest_raw, bank_manifest_record_digest = _candidate_bank_artifacts(
        lineages,
        parent=destination.parent,
    )
    lineage_fact_rows = tuple(
        _fact_row(LINEAGE_FACT_SCHEMA, row.lineage_fact)
        for row in sorted(lineages, key=lambda item: item.lineage_fact.base_lineage_key)
    )
    task_fact_rows = tuple(
        _fact_row(TASK_FACT_SCHEMA, row.task_fact)
        for row in sorted(lineages, key=lambda item: item.task_fact.group_id)
    )
    reference_rows = tuple(
        sorted(lineages, key=lambda item: item.reference_source.problem_digest)
    )
    quality_raw = b"".join(row.reference_source.canonical_bytes + b"\n" for row in reference_rows)
    quality_payload = {
        "file": QUALITY_FILENAME,
        "record_count": len(reference_rows),
        "references": [
            {
                "authority_sha256": row.reference_source.authority_sha256,
                "instance_id": row.instance.instance_id,
                "problem_digest": row.reference_source.problem_digest,
                "source_record_sha256": row.reference_source.record_sha256,
            }
            for row in reference_rows
        ],
        "schema": QUALITY_MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "sha256": _sha256_bytes(quality_raw),
    }
    quality_manifest_raw = _canonical_document(_record(quality_payload))
    payload_files = {
        CANDIDATE_BANK_FILENAME: bank_raw,
        CANDIDATE_BANK_MANIFEST_FILENAME: bank_manifest_raw,
        LINEAGE_FACTS_FILENAME: _jsonl_bytes(lineage_fact_rows),
        TASK_FACTS_FILENAME: _jsonl_bytes(task_fact_rows),
        QUALITY_FILENAME: quality_raw,
        QUALITY_MANIFEST_FILENAME: quality_manifest_raw,
    }
    release_payload = {
        "accounting": _attempt_accounting(lineages),
        "artifacts": _artifact_inventory(payload_files),
        "candidate_bank_manifest_record_digest": bank_manifest_record_digest,
        "checksum_policy": {
            "algorithm": "sha256",
            "coverage": "every-regular-output-except-SHA256SUMS",
            "path": CHECKSUM_FILENAME,
        },
        "counts": {
            "candidate_groups": len(lineages),
            "instances": len(lineages),
            "lineage_facts": len(lineage_fact_rows),
            "lineages": len(lineages),
            "problems": len(reference_rows),
            "split_units": len(lineages),
            "task_facts": len(task_fact_rows),
        },
        "generation_config": generation_config,
        "schema": RELEASE_MANIFEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "shard_count": expected_shard_count,
        "source_plan": source_plan,
        "source_shards": [
            {
                "manifest_record_digest": item.manifest_record_digest,
                "manifest_sha256": item.manifest_sha256,
                "shard_index": item.shard.shard_index,
                "shard_record_digest": item.shard.record_digest,
                "shard_sha256": item.shard_sha256,
            }
            for item in ordered
        ],
    }
    release_manifest = _record(release_payload)
    release_manifest_raw = _canonical_document(release_manifest)
    authenticated = {
        **payload_files,
        RELEASE_MANIFEST_FILENAME: release_manifest_raw,
    }
    files = {**authenticated, CHECKSUM_FILENAME: _checksum_bytes(authenticated)}
    _publish_directory(destination, files)
    root = destination.resolve(strict=True)
    return CorpusMergeReceipt(
        output_directory=root,
        candidate_bank_path=root / CANDIDATE_BANK_FILENAME,
        candidate_bank_manifest_path=root / CANDIDATE_BANK_MANIFEST_FILENAME,
        lineage_facts_path=root / LINEAGE_FACTS_FILENAME,
        task_facts_path=root / TASK_FACTS_FILENAME,
        quality_path=root / QUALITY_FILENAME,
        quality_manifest_path=root / QUALITY_MANIFEST_FILENAME,
        release_manifest_path=root / RELEASE_MANIFEST_FILENAME,
        checksums_path=root / CHECKSUM_FILENAME,
        release_manifest_sha256=_sha256_bytes(release_manifest_raw),
        release_manifest_record_digest=release_manifest["record_digest"],
        candidate_bank_manifest_record_digest=bank_manifest_record_digest,
        shard_count=expected_shard_count,
        lineage_count=len(lineages),
    )


__all__ = [
    "CorpusMergeReceipt",
    "CorpusShardWriteReceipt",
    "merge_corpus_shards",
    "write_corpus_shard",
]
