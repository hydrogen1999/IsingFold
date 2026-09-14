"""Outcome-blind prospective eligibility gate for a pinned IsingFold corpus plan.

This boundary calls only :func:`prospect_lineage`.  It never proposes embeddings,
runs a sampler, observes a quality label, or changes the frozen lineage census.
Execution parallelism is intentionally absent from the published record because it
cannot change the deterministic scientific result.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import stat
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from embedbench.candidate_bank import canonical_json_bytes, content_digest
from embedbench.isingfold_corpus_plan import CorpusGenerationPlan
from embedbench.isingfold_corpus_shard import (
    LineageRequest,
    ShardConfig,
    _generation_provenance,
    prospect_lineage,
    reference_problem_digest,
)

PREFLIGHT_SCHEMA = "embedbench.isingfold-prospective-preflight"
PREFLIGHT_SCHEMA_VERSION = 2
IDENTITY_MAP_SCHEMA = "embedbench.isingfold-prospective-identity-map"
IDENTITY_MAP_SCHEMA_VERSION = 1
MAX_PREFLIGHT_WORKERS = 64
DEFAULT_PREFLIGHT_WORKERS = 10
_PERCENTILES = (50, 90, 95, 99)
_PARTITION_ORDER = {"train": 0, "val": 1, "test": 2}
_TOPOLOGY_ORDER = {"chimera": 0, "pegasus": 1, "zephyr": 2}
_ORIGIN_ORDER = {"application-derived": 0, "synthetic-ink-drop": 1}
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class ProspectivePreflightWriteReceipt:
    """External file identity returned after atomic, create-once publication."""

    preflight_path: Path
    preflight_sha256: str
    preflight_record_digest: str
    generation_provenance_digest: str
    source_sha256: str
    lineage_count: int
    maximum_prospective_slot: int
    prospective_identity_count: int
    prospective_identity_map_digest: str


@dataclass(frozen=True, slots=True)
class ProspectivePreflightVerificationReceipt:
    """Authenticated result of replaying every prospective identity."""

    preflight_path: Path
    preflight_sha256: str
    preflight_record_digest: str
    generation_provenance_digest: str
    source_sha256: str
    lineage_count: int
    maximum_prospective_slot: int
    prospective_identity_count: int
    prospective_identity_map_digest: str


@dataclass(frozen=True, slots=True)
class ProspectivePreflightReadReceipt:
    """Authenticated result of a cheap, non-replaying preflight read."""

    preflight_path: Path
    preflight_sha256: str
    preflight_record_digest: str
    generation_provenance_digest: str
    source_sha256: str
    lineage_count: int
    maximum_prospective_slot: int
    prospective_identity_count: int
    prospective_identity_map_digest: str


@dataclass(frozen=True, slots=True)
class _ProspectiveIdentity:
    lineage_id: str
    prospective_slot: int
    split_unit_id: str
    problem_sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "lineage_id": self.lineage_id,
            "prospective_slot": self.prospective_slot,
            "split_unit_id": self.split_unit_id,
            "problem_sha256": self.problem_sha256,
        }


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in _HEX for char in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def require_worker_count(value: object) -> int:
    """Validate the bounded operational parallelism knob."""

    if type(value) is not int or not 1 <= value <= MAX_PREFLIGHT_WORKERS:
        raise ValueError(
            f"workers must be an integer in [1, {MAX_PREFLIGHT_WORKERS}]"
        )
    return value


def _config(plan: CorpusGenerationPlan) -> ShardConfig:
    screening = plan.screening
    return ShardConfig(
        root_seed=plan.root_seed,
        shard_index=0,
        shard_count=1,
        attempt_slots=screening.attempt_slots,
        incumbent_search_slots=screening.incumbent_search_slots,
        max_split_search=screening.max_split_search,
        strengths=screening.strengths,
        reads=screening.reads,
        sweeps=screening.sweeps,
    )


def _prospect_one(
    config: ShardConfig,
    request: LineageRequest,
) -> tuple[str, int | None, str | None, str | None, str | None]:
    try:
        prospective = prospect_lineage(config, request)
    except Exception as error:  # Every unresolved row must be reported before publication.
        return request.lineage_id, None, None, None, f"{type(error).__name__}: {error}"
    slot = prospective.prospective_slot
    if type(slot) is not int or not 0 <= slot < config.max_split_search:
        return (
            request.lineage_id,
            None,
            None,
            None,
            "invalid prospective slot returned by generator",
        )
    split_unit_id = prospective.split_unit_id
    if type(split_unit_id) is not str or not split_unit_id:
        return request.lineage_id, None, None, None, "invalid split-unit identity returned"
    try:
        problem_sha256 = reference_problem_digest(prospective)
    except Exception as error:
        return (
            request.lineage_id,
            None,
            None,
            None,
            f"invalid prospective logical problem: {type(error).__name__}: {error}",
        )
    return request.lineage_id, slot, split_unit_id, problem_sha256, None


def _duplicate_identity_groups(
    prospects: tuple[_ProspectiveIdentity, ...],
    field: str,
) -> list[tuple[str, tuple[str, ...]]]:
    grouped: dict[str, list[str]] = {}
    for prospect in prospects:
        value = getattr(prospect, field)
        grouped.setdefault(value, []).append(prospect.lineage_id)
    return sorted(
        (value, tuple(sorted(lineage_ids)))
        for value, lineage_ids in grouped.items()
        if len(lineage_ids) > 1
    )


def _require_global_identity_uniqueness(
    prospects: tuple[_ProspectiveIdentity, ...],
) -> None:
    collisions = []
    for field, label in (
        ("split_unit_id", "split-unit"),
        ("problem_sha256", "problem"),
    ):
        for value, lineage_ids in _duplicate_identity_groups(prospects, field):
            collisions.append((label, value, lineage_ids))
    if not collisions:
        return
    preview = "; ".join(
        f"{label}={value} lineages={','.join(lineage_ids)}"
        for label, value, lineage_ids in collisions[:10]
    )
    suffix = "" if len(collisions) <= 10 else f"; plus {len(collisions) - 10} more"
    first_label = collisions[0][0]
    raise ValueError(
        f"duplicate prospective {first_label} identity across pinned plan: {preview}{suffix}"
    )


def _run_prospects(
    plan: CorpusGenerationPlan,
    *,
    workers: int,
) -> tuple[_ProspectiveIdentity, ...]:
    selected_workers = require_worker_count(workers)
    config = _config(plan)
    requests = tuple(item.request for item in plan.lineages)
    if selected_workers == 1:
        raw_results = tuple(_prospect_one(config, request) for request in requests)
    else:
        method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
        with ProcessPoolExecutor(
            max_workers=selected_workers,
            mp_context=multiprocessing.get_context(method),
        ) as pool:
            raw_results = tuple(
                pool.map(
                    _prospect_one,
                    (config for _ in requests),
                    requests,
                    chunksize=8,
                )
            )
    failures = sorted(
        (lineage_id, error)
        for lineage_id, slot, split_unit_id, problem_sha256, error in raw_results
        if slot is None or error is not None
    )
    if failures:
        preview = "; ".join(
            f"{lineage_id} ({error})" for lineage_id, error in failures[:10]
        )
        suffix = "" if len(failures) <= 10 else f"; plus {len(failures) - 10} more"
        noun = "lineage" if len(failures) == 1 else "lineages"
        raise ValueError(
            f"prospective preflight found {len(failures)} unresolved {noun}: "
            f"{preview}{suffix}"
        )
    results = tuple(
        sorted(
            (
                _ProspectiveIdentity(
                    lineage_id=lineage_id,
                    prospective_slot=slot,
                    split_unit_id=split_unit_id,
                    problem_sha256=problem_sha256,
                )
                for lineage_id, slot, split_unit_id, problem_sha256, error in raw_results
                if slot is not None
                and split_unit_id is not None
                and problem_sha256 is not None
                and error is None
            ),
            key=lambda item: item.lineage_id,
        )
    )
    expected_ids = tuple(sorted(request.lineage_id for request in requests))
    if tuple(item.lineage_id for item in results) != expected_ids:
        raise ValueError("prospective preflight result census differs from the pinned plan")
    _require_global_identity_uniqueness(results)
    return results


def _nearest_rank(slots: tuple[int, ...], percentile: int) -> int:
    rank = (percentile * len(slots) + 99) // 100
    return slots[rank - 1]


def _census(plan: CorpusGenerationPlan) -> list[dict[str, object]]:
    counts = Counter(
        (item.request.partition, item.request.topology, item.request.origin)
        for item in plan.lineages
    )
    keys = sorted(
        counts,
        key=lambda key: (
            _PARTITION_ORDER[key[0]],
            _TOPOLOGY_ORDER[key[1]],
            _ORIGIN_ORDER[key[2]],
        ),
    )
    return [
        {
            "lineage_count": counts[partition, topology, origin],
            "origin": origin,
            "partition": partition,
            "topology": topology,
        }
        for partition, topology, origin in keys
    ]


def _document(
    plan: CorpusGenerationPlan,
    *,
    expected_plan_sha256: str,
    expected_source_sha256: str,
    generation_provenance: dict[str, object],
    prospects: tuple[_ProspectiveIdentity, ...],
) -> dict[str, Any]:
    slots = tuple(sorted(item.prospective_slot for item in prospects))
    if not slots or len(slots) != len(plan.lineages):
        raise ValueError("prospective preflight requires one resolved slot per planned lineage")
    entries = [item.to_dict() for item in prospects]
    identity_payload: dict[str, Any] = {
        "count": len(entries),
        "entries": entries,
        "protocol": plan.prospective_identity_policy.map_protocol,
        "schema": IDENTITY_MAP_SCHEMA,
        "schema_version": IDENTITY_MAP_SCHEMA_VERSION,
    }
    identity_map = {
        **identity_payload,
        "record_digest": content_digest(identity_payload),
    }
    payload: dict[str, Any] = {
        "census": _census(plan),
        "generation_provenance": {
            "protocol": generation_provenance["protocol"],
            "record_digest": content_digest(generation_provenance),
        },
        "lineage_count": len(slots),
        "plan": {
            "record_digest": plan.record_digest,
            "sha256": expected_plan_sha256,
        },
        "protocol": {
            "operation": "prospect-lineage-only-no-proposal-or-quality-v2",
            "quantile_definition": "nearest-rank-ceiling-v1",
            "search_slots": plan.screening.max_split_search,
        },
        "prospective_identity_map": identity_map,
        "schema": PREFLIGHT_SCHEMA,
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "source": {"sha256": expected_source_sha256},
        "slot_statistics": {
            "maximum": slots[-1],
            "minimum": slots[0],
            "quantiles": [
                {"percentile": percentile, "slot": _nearest_rank(slots, percentile)}
                for percentile in _PERCENTILES
            ],
        },
    }
    return {**payload, "record_digest": content_digest(payload)}


def _path_exists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _strict_json(raw: bytes, name: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{name} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        document = json.loads(
            raw,
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite number {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    if type(document) is not dict or raw != canonical_json_bytes(document) + b"\n":
        raise ValueError(f"{name} must be canonical JSON with one terminal line feed")
    return document


def _read_regular(path: Path, name: str) -> bytes:
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{name} must be a regular file, not a link")
        raw = path.read_bytes()
        after = path.lstat()
    except OSError as error:
        raise ValueError(f"cannot read {name}: {error}") from error
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or len(raw) != before.st_size:
        raise ValueError(f"{name} changed while it was being read")
    return raw


def _receipt(
    receipt_type: type[ProspectivePreflightWriteReceipt]
    | type[ProspectivePreflightVerificationReceipt]
    | type[ProspectivePreflightReadReceipt],
    *,
    path: Path,
    raw: bytes,
    document: dict[str, Any],
) -> (
    ProspectivePreflightWriteReceipt
    | ProspectivePreflightVerificationReceipt
    | ProspectivePreflightReadReceipt
):
    identity_map = document["prospective_identity_map"]
    provenance = document["generation_provenance"]
    return receipt_type(
        preflight_path=path,
        preflight_sha256=hashlib.sha256(raw).hexdigest(),
        preflight_record_digest=document["record_digest"],
        generation_provenance_digest=provenance["record_digest"],
        source_sha256=document["source"]["sha256"],
        lineage_count=document["lineage_count"],
        maximum_prospective_slot=document["slot_statistics"]["maximum"],
        prospective_identity_count=identity_map["count"],
        prospective_identity_map_digest=identity_map["record_digest"],
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_new_file(destination: Path, raw: bytes) -> Path:
    """Publish complete bytes atomically without replacing any filesystem object."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    parent_metadata = destination.parent.lstat()
    if not stat.S_ISDIR(parent_metadata.st_mode) or destination.parent.is_symlink():
        raise ValueError("preflight output parent must be a real directory")
    if _path_exists(destination):
        raise FileExistsError(f"preflight output already exists: {destination}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.staging-",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise FileExistsError(
                f"preflight output already exists: {destination}"
            ) from None
        _fsync_directory(destination.parent)
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
    return destination.resolve(strict=True)


def publish_prospective_preflight(
    plan: CorpusGenerationPlan,
    *,
    expected_plan_sha256: str,
    expected_source_sha256: str,
    workers: int,
    output_path: str | os.PathLike[str],
) -> ProspectivePreflightWriteReceipt:
    """Prospect the exact pinned census and atomically publish its receipt."""

    if not isinstance(plan, CorpusGenerationPlan):
        raise TypeError("plan must be a CorpusGenerationPlan")
    expected = _require_sha256(expected_plan_sha256, "expected plan SHA-256")
    expected_source = _require_sha256(expected_source_sha256, "expected source SHA-256")
    plan_raw = canonical_json_bytes(plan.to_dict()) + b"\n"
    if hashlib.sha256(plan_raw).hexdigest() != expected:
        raise ValueError("expected plan SHA-256 does not bind the supplied plan")
    destination = Path(output_path)
    if _path_exists(destination):
        raise FileExistsError(f"preflight output already exists: {destination}")
    provenance_before = _generation_provenance()
    prospects = _run_prospects(plan, workers=workers)
    provenance_after = _generation_provenance()
    if provenance_before != provenance_after:
        raise ValueError("generation provenance changed during prospective preflight")
    document = _document(
        plan,
        expected_plan_sha256=expected,
        expected_source_sha256=expected_source,
        generation_provenance=provenance_before,
        prospects=prospects,
    )
    raw = canonical_json_bytes(document) + b"\n"
    published = _atomic_new_file(destination, raw)
    return _receipt(
        ProspectivePreflightWriteReceipt,
        path=published,
        raw=raw,
        document=document,
    )


def verify_prospective_preflight(
    plan: CorpusGenerationPlan,
    *,
    expected_plan_sha256: str,
    expected_source_sha256: str,
    preflight_path: str | os.PathLike[str],
    expected_preflight_sha256: str,
    expected_preflight_record_digest: str,
    expected_identity_map_digest: str,
    expected_generation_provenance_digest: str,
    workers: int,
) -> ProspectivePreflightVerificationReceipt:
    """Strictly authenticate and replay a complete prospective preflight."""

    if not isinstance(plan, CorpusGenerationPlan):
        raise TypeError("plan must be a CorpusGenerationPlan")
    expected_plan = _require_sha256(expected_plan_sha256, "expected plan SHA-256")
    plan_raw = canonical_json_bytes(plan.to_dict()) + b"\n"
    if hashlib.sha256(plan_raw).hexdigest() != expected_plan:
        raise ValueError("expected plan SHA-256 does not bind the supplied plan")
    read_receipt = read_prospective_preflight(
        plan,
        expected_plan_sha256=expected_plan,
        expected_source_sha256=expected_source_sha256,
        preflight_path=preflight_path,
        expected_preflight_sha256=expected_preflight_sha256,
        expected_preflight_record_digest=expected_preflight_record_digest,
        expected_identity_map_digest=expected_identity_map_digest,
        expected_generation_provenance_digest=expected_generation_provenance_digest,
    )
    source = read_receipt.preflight_path
    raw = _read_regular(source, "prospective preflight")
    observed = _strict_json(raw, "prospective preflight")
    provenance_before = _generation_provenance()
    prospects = _run_prospects(plan, workers=workers)
    provenance_after = _generation_provenance()
    if provenance_before != provenance_after:
        raise ValueError("generation provenance changed during prospective preflight replay")
    expected_document = _document(
        plan,
        expected_plan_sha256=expected_plan,
        expected_source_sha256=read_receipt.source_sha256,
        generation_provenance=provenance_before,
        prospects=prospects,
    )
    if observed != expected_document:
        raise ValueError(
            "prospective preflight differs from a full replay under current generation provenance"
        )
    resolved = source.resolve(strict=True)
    receipt = _receipt(
        ProspectivePreflightVerificationReceipt,
        path=resolved,
        raw=raw,
        document=observed,
    )
    if not isinstance(receipt, ProspectivePreflightVerificationReceipt):
        raise RuntimeError("prospective preflight verifier returned the wrong receipt type")
    return receipt


def read_prospective_preflight(
    plan: CorpusGenerationPlan,
    *,
    expected_plan_sha256: str,
    expected_source_sha256: str,
    preflight_path: str | os.PathLike[str],
    expected_preflight_sha256: str,
    expected_preflight_record_digest: str,
    expected_identity_map_digest: str,
    expected_generation_provenance_digest: str,
) -> ProspectivePreflightReadReceipt:
    """Authenticate all preflight commitments without recomputing prospective rows."""

    if not isinstance(plan, CorpusGenerationPlan):
        raise TypeError("plan must be a CorpusGenerationPlan")
    expected_plan = _require_sha256(expected_plan_sha256, "expected plan SHA-256")
    expected_source = _require_sha256(expected_source_sha256, "expected source SHA-256")
    expected_preflight = _require_sha256(
        expected_preflight_sha256,
        "expected preflight SHA-256",
    )
    expected_record = _require_sha256(
        expected_preflight_record_digest,
        "expected preflight record digest",
    )
    expected_map = _require_sha256(
        expected_identity_map_digest,
        "expected prospective identity map digest",
    )
    expected_provenance = _require_sha256(
        expected_generation_provenance_digest,
        "expected generation provenance digest",
    )
    plan_raw = canonical_json_bytes(plan.to_dict()) + b"\n"
    if hashlib.sha256(plan_raw).hexdigest() != expected_plan:
        raise ValueError("expected plan SHA-256 does not bind the supplied plan")

    source = Path(preflight_path)
    raw = _read_regular(source, "prospective preflight")
    if hashlib.sha256(raw).hexdigest() != expected_preflight:
        raise ValueError("prospective preflight SHA-256 mismatch")
    document = _strict_json(raw, "prospective preflight")
    if document.get("schema") != PREFLIGHT_SCHEMA or document.get(
        "schema_version"
    ) != PREFLIGHT_SCHEMA_VERSION:
        raise ValueError("unsupported prospective preflight schema")
    if document.get("record_digest") != expected_record:
        raise ValueError("prospective preflight record digest differs from commitment")
    payload = {key: value for key, value in document.items() if key != "record_digest"}
    if content_digest(payload) != expected_record:
        raise ValueError("prospective preflight record digest mismatch")

    raw_map = document.get("prospective_identity_map")
    if type(raw_map) is not dict or set(raw_map) != {
        "count",
        "entries",
        "protocol",
        "record_digest",
        "schema",
        "schema_version",
    }:
        raise ValueError("prospective identity map schema fields differ")
    if (
        raw_map["schema"] != IDENTITY_MAP_SCHEMA
        or raw_map["schema_version"] != IDENTITY_MAP_SCHEMA_VERSION
        or raw_map["protocol"] != plan.prospective_identity_policy.map_protocol
    ):
        raise ValueError("unsupported prospective identity map schema or protocol")
    map_payload = {key: value for key, value in raw_map.items() if key != "record_digest"}
    if raw_map["record_digest"] != expected_map or content_digest(map_payload) != expected_map:
        raise ValueError("prospective identity map digest mismatch")
    raw_entries = raw_map["entries"]
    if type(raw_entries) is not list or raw_map["count"] != len(raw_entries):
        raise ValueError("prospective identity map count differs from its entries")
    prospects: list[_ProspectiveIdentity] = []
    for entry in raw_entries:
        if type(entry) is not dict or set(entry) != set(
            plan.prospective_identity_policy.map_entry_fields
        ):
            raise ValueError("prospective identity map entry fields differ")
        lineage_id = entry["lineage_id"]
        slot = entry["prospective_slot"]
        split_unit_id = entry["split_unit_id"]
        problem_sha256 = entry["problem_sha256"]
        if type(lineage_id) is not str or not lineage_id:
            raise ValueError("prospective identity lineage ID must be nonempty text")
        if type(slot) is not int or not 0 <= slot < plan.screening.max_split_search:
            raise ValueError("prospective identity slot lies outside the plan")
        if type(split_unit_id) is not str or not split_unit_id:
            raise ValueError("prospective split-unit identity must be nonempty text")
        _require_sha256(problem_sha256, "prospective problem identity")
        prospects.append(
            _ProspectiveIdentity(
                lineage_id=lineage_id,
                prospective_slot=slot,
                split_unit_id=split_unit_id,
                problem_sha256=problem_sha256,
            )
        )
    identities = tuple(prospects)
    expected_ids = tuple(item.request.lineage_id for item in plan.lineages)
    if tuple(item.lineage_id for item in identities) != expected_ids:
        raise ValueError("prospective identity map differs from the exact planned census")
    _require_global_identity_uniqueness(identities)

    provenance_before = _generation_provenance()
    provenance_after = _generation_provenance()
    if provenance_before != provenance_after:
        raise ValueError("generation provenance changed while reading prospective preflight")
    provenance_digest = content_digest(provenance_before)
    if provenance_digest != expected_provenance:
        raise ValueError("current generation provenance differs from external commitment")
    expected_document = _document(
        plan,
        expected_plan_sha256=expected_plan,
        expected_source_sha256=expected_source,
        generation_provenance=provenance_before,
        prospects=identities,
    )
    if document != expected_document:
        raise ValueError("prospective preflight commitments or statistics are inconsistent")
    resolved = source.resolve(strict=True)
    receipt = _receipt(
        ProspectivePreflightReadReceipt,
        path=resolved,
        raw=raw,
        document=document,
    )
    if not isinstance(receipt, ProspectivePreflightReadReceipt):
        raise RuntimeError("prospective preflight reader returned the wrong receipt type")
    return receipt


__all__ = [
    "DEFAULT_PREFLIGHT_WORKERS",
    "MAX_PREFLIGHT_WORKERS",
    "IDENTITY_MAP_SCHEMA",
    "IDENTITY_MAP_SCHEMA_VERSION",
    "PREFLIGHT_SCHEMA",
    "PREFLIGHT_SCHEMA_VERSION",
    "ProspectivePreflightWriteReceipt",
    "ProspectivePreflightVerificationReceipt",
    "ProspectivePreflightReadReceipt",
    "publish_prospective_preflight",
    "read_prospective_preflight",
    "require_worker_count",
    "verify_prospective_preflight",
]
