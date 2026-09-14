"""Deterministic, target-free construction of the Gate 1 conformance corpus.

The registry is deliberately small because Gate 1 runs exponential independent checks.  Its
small size must not make it hand-pickable: producer, verifier, and release-gate loading all call
the selector in this module and authenticate both the prepared manifest and registry raw bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.prepared import (
    PreparedDesignCondition,
    PreparedTask,
    load_prepared_partition,
)

EXACT_CONFORMANCE_CORPUS_SCHEMA = "isingfold.exact-conformance-corpus"
EXACT_CONFORMANCE_CORPUS_VERSION = 1
EXACT_CONFORMANCE_CORPUS_ID = "if-gate1-exact-v1"
EXACT_CONFORMANCE_PURPOSE = "bounded-independent-structural-and-program-conformance"
EXACT_CONFORMANCE_MAX_TASKS = 8
EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES = 12
EXACT_CONFORMANCE_SELECTION_SEED = 0
EXACT_CONFORMANCE_TIE_DOMAIN = "exact-conformance-task-tie-v1"
EXACT_CONFORMANCE_SELECTOR_VERSION = "exact-conformance-selector-v1"
EXACT_CONFORMANCE_VERIFICATION_SCHEMA = "isingfold.exact-conformance-verification"
EXACT_CONFORMANCE_VERIFICATION_VERSION = 1

EXACT_CONFORMANCE_AXES = (
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

_REGISTRY_FIELDS = {
    "corpus_id",
    "max_logical_variables_per_task",
    "max_tasks",
    "purpose",
    "record_digest",
    "schema",
    "schema_version",
    "source_corpus_manifest_sha256",
    "task_ids",
}
_DESIGN_METADATA_FIELDS = {
    "base_lineage_key",
    "calibration_sha256",
    "learning_partition",
    "registry_row_digest",
}
_PUBLIC_PREPARED_FILES = (
    "initializers.jsonl",
    "policy_instances.jsonl",
    "provenance.jsonl",
    "splits.json",
)


class ExactConformanceError(ValueError):
    """An exact-conformance trust or selection invariant was violated."""


@dataclass(frozen=True)
class _Candidate:
    task: PreparedTask
    task_id: str
    base_lineage: str
    coordinates: tuple[tuple[str, str | bool], ...]
    stratum_digest: str


@dataclass(frozen=True)
class ExactConformanceSelection:
    """The deterministic selection plus diagnostics used by all three consumers."""

    tasks: tuple[PreparedTask, ...]
    task_ids: tuple[str, ...]
    greedy_task_ids: tuple[str, ...]
    stratum_digests: Mapping[str, str]
    source_manifest_sha256: str
    source_population_census: Mapping[str, object]
    selected_marginal_census: Mapping[str, object]


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_sha256(value: object, label: str) -> str:
    if not _is_sha256(value):
        raise ExactConformanceError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ExactConformanceError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise ExactConformanceError(f"{label} contains non-finite number {token}")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ExactConformanceError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ExactConformanceError(f"{label} must contain one JSON object")
    try:
        canonical = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise ExactConformanceError(f"{label} is not finite canonical JSON") from error
    if raw != canonical + b"\n":
        raise ExactConformanceError(
            f"{label} is not canonical JSON followed by exactly one newline"
        )
    return value


def _absolute_without_symlink(path: Path, *, must_exist: bool) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    probe = absolute if must_exist else absolute.parent
    try:
        resolved = probe.resolve(strict=True)
    except FileNotFoundError as error:
        raise ExactConformanceError(f"path does not exist: {probe}") from error
    if resolved != probe:
        raise ExactConformanceError(f"path resolves through a symlink: {probe}")
    return absolute


def _read_regular_file(path: str | os.PathLike[str], label: str) -> tuple[Path, bytes]:
    source = _absolute_without_symlink(Path(path), must_exist=True)
    before = source.lstat()
    if stat.S_ISLNK(before.st_mode):
        raise ExactConformanceError(f"{label} may not be a symlink")
    if not stat.S_ISREG(before.st_mode):
        raise ExactConformanceError(f"{label} must be a regular file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(source, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise ExactConformanceError(f"{label} changed while it was opened")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    after = source.lstat()
    if stat.S_ISLNK(after.st_mode) or (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns):
        raise ExactConformanceError(f"{label} changed while it was read")
    return source, b"".join(chunks)


def _verify_record(record: Mapping[str, Any], label: str) -> None:
    recorded = _require_sha256(record.get("record_digest"), f"{label} record_digest")
    payload = {key: value for key, value in record.items() if key != "record_digest"}
    if recorded != content_digest(payload):
        raise ExactConformanceError(f"{label} record digest mismatch")


def _publish_new_file(path: str | os.PathLike[str], raw: bytes) -> Path:
    destination = _absolute_without_symlink(Path(path), must_exist=False)
    parent = destination.parent
    if not parent.is_dir():
        raise ExactConformanceError(f"publication parent is not a directory: {parent}")
    try:
        current = destination.lstat()
    except FileNotFoundError:
        current = None
    if current is not None:
        if stat.S_ISLNK(current.st_mode):
            raise ExactConformanceError(
                f"publication destination may not be a symlink: {destination}"
            )
        raise FileExistsError(f"publication destination already exists: {destination}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fchmod(handle.fileno(), 0o644)
            os.fsync(handle.fileno())
        # A hard-link publish has the no-replace atomicity that os.replace lacks.  The
        # sibling temporary and destination are necessarily on the same filesystem.
        os.link(temporary, destination, follow_symlinks=False)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def read_regular_file(
    path: str | os.PathLike[str], label: str
) -> tuple[Path, bytes]:
    """Read immutable artifact bytes without following symlinks."""

    return _read_regular_file(path, label)


def publish_new_file(path: str | os.PathLike[str], raw: bytes) -> Path:
    """Atomically publish one immutable file without replacement."""

    return _publish_new_file(path, raw)


def _seed_digest(
    root: int,
    domain: str,
    parts: Sequence[str],
) -> str:
    if type(root) is not int or root < 0:
        raise ExactConformanceError("seed root must be a nonnegative integer")
    if type(domain) is not str or not domain:
        raise ExactConformanceError("seed domain must be a nonempty string")
    if any(type(part) is not str or not part for part in parts):
        raise ExactConformanceError("seed parts must be nonempty strings")
    return _sha256(
        canonical_json_bytes(
            {"domain": domain, "parts": [root, *parts], "version": 1}
        )
    )


def _condition_coordinates(
    condition: PreparedDesignCondition,
) -> tuple[tuple[str, str | bool], ...]:
    observed = tuple(
        field.name for field in fields(condition) if field.name not in _DESIGN_METADATA_FIELDS
    )
    if observed != EXACT_CONFORMANCE_AXES:
        raise ExactConformanceError(
            "prepared design condition coordinate schema differs from the nine fixed axes"
        )
    coordinates: list[tuple[str, str | bool]] = []
    for axis in EXACT_CONFORMANCE_AXES:
        value = getattr(condition, axis)
        if type(value) not in {str, bool} or (type(value) is str and not value):
            raise ExactConformanceError(f"prepared design condition has an invalid {axis} level")
        coordinates.append((axis, value))
    return tuple(coordinates)


def _has_complete_witness(task: PreparedTask) -> bool:
    witness = task.task.initial_embedding
    if witness is None:
        witness = task.task.witness
    if not isinstance(witness, Mapping) or not witness:
        return False
    logical_nodes = set(task.task.logical.nodes)
    if set(witness) != logical_nodes or not logical_nodes:
        return False
    return all(
        isinstance(chain, (set, frozenset, tuple, list)) and len(chain) > 0
        for chain in witness.values()
    )


def _candidate(task: PreparedTask) -> _Candidate | None:
    if task.task.name != task.task_id:
        raise ExactConformanceError("prepared task ID differs from its embedded task identity")
    if task.prepared_schema_version != 4 or task.partition != "val":
        return None
    if task.corpus_scope != "production-designed-v4":
        raise ExactConformanceError("prepared schema-v4 task has an incompatible corpus scope")
    if task.task.logical.number_of_nodes() > EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES:
        return None
    if not _has_complete_witness(task):
        return None
    condition = task.design_condition
    if condition is None:
        return None
    if not isinstance(condition, PreparedDesignCondition):
        raise ExactConformanceError("task design condition has the wrong record type")
    if condition.learning_partition != "val":
        raise ExactConformanceError(
            "prepared task and design condition disagree on validation partition"
        )
    lineage = condition.base_lineage_key
    if type(lineage) is not str or not lineage:
        raise ExactConformanceError("prepared task has an empty immutable base lineage")
    if task.task.lineage != lineage:
        raise ExactConformanceError(
            "prepared task lineage differs from its design-condition base lineage"
        )
    if task.provenance is not None and task.provenance.base_parent_lineage != lineage:
        raise ExactConformanceError(
            "prepared task provenance differs from its design-condition base lineage"
        )
    coordinates = _condition_coordinates(condition)
    coordinate_object = {axis: value for axis, value in coordinates}
    return _Candidate(
        task=task,
        task_id=task.task_id,
        base_lineage=lineage,
        coordinates=coordinates,
        stratum_digest=content_digest(coordinate_object),
    )


def _census(candidates: Sequence[_Candidate]) -> dict[str, object]:
    by_axis: dict[str, list[dict[str, object]]] = {}
    for axis_index, axis in enumerate(EXACT_CONFORMANCE_AXES):
        task_counts: Counter[str | bool] = Counter()
        lineage_members: dict[str | bool, set[str]] = {}
        for candidate in candidates:
            level = candidate.coordinates[axis_index][1]
            task_counts[level] += 1
            lineage_members.setdefault(level, set()).add(candidate.base_lineage)
        ordered_levels = sorted(task_counts, key=canonical_json_bytes)
        by_axis[axis] = [
            {
                "base_lineage_count": len(lineage_members[level]),
                "level": level,
                "task_count": task_counts[level],
            }
            for level in ordered_levels
        ]

    joint_task_counts: Counter[tuple[tuple[str, str | bool], ...]] = Counter()
    joint_lineages: dict[tuple[tuple[str, str | bool], ...], set[str]] = {}
    for candidate in candidates:
        joint_task_counts[candidate.coordinates] += 1
        joint_lineages.setdefault(candidate.coordinates, set()).add(candidate.base_lineage)
    ordered_joints = sorted(
        joint_task_counts,
        key=lambda coordinates: canonical_json_bytes(dict(coordinates)),
    )
    return {
        "by_axis": by_axis,
        "eligible_base_lineage_count": len({candidate.base_lineage for candidate in candidates}),
        "eligible_task_count": len(candidates),
        "joint_strata": [
            {
                "base_lineage_count": len(joint_lineages[coordinates]),
                "coordinates": dict(coordinates),
                "stratum_digest": content_digest(dict(coordinates)),
                "task_count": joint_task_counts[coordinates],
            }
            for coordinates in ordered_joints
        ],
    }


def select_exact_conformance_tasks(
    prepared: Sequence[PreparedTask],
    source_corpus_manifest_sha256: str,
) -> ExactConformanceSelection:
    """Select exactly eight diverse public validation tasks, independent of input order."""

    source_sha = _require_sha256(source_corpus_manifest_sha256, "source corpus manifest SHA-256")
    task_ids: set[str] = set()
    candidates: list[_Candidate] = []
    for task in prepared:
        if not isinstance(task, PreparedTask):
            raise ExactConformanceError("prepared population contains a non-PreparedTask value")
        if type(task.task_id) is not str or not task.task_id:
            raise ExactConformanceError("prepared validation task has an empty task ID")
        if task.task_id in task_ids:
            raise ExactConformanceError(
                "prepared validation population contains duplicate task IDs"
            )
        task_ids.add(task.task_id)
        candidate = _candidate(task)
        if candidate is not None:
            candidates.append(candidate)

    eligible_lineages = {candidate.base_lineage for candidate in candidates}
    if len(candidates) < EXACT_CONFORMANCE_MAX_TASKS:
        raise ExactConformanceError(
            "exact-conformance selection requires at least 8 eligible validation tasks"
        )
    if len(eligible_lineages) < EXACT_CONFORMANCE_MAX_TASKS:
        raise ExactConformanceError(
            "exact-conformance selection requires at least 8 independent base lineages"
        )

    tie_digests = {
        candidate.task_id: _seed_digest(
            EXACT_CONFORMANCE_SELECTION_SEED,
            EXACT_CONFORMANCE_TIE_DOMAIN,
            (
                source_sha,
                candidate.stratum_digest,
                candidate.base_lineage,
                candidate.task_id,
            ),
        )
        for candidate in candidates
    }
    if len(set(tie_digests.values())) != len(tie_digests):
        raise ExactConformanceError(
            "exact-conformance candidate tie derivation contains a SHA-256 collision"
        )
    stratum_registry: dict[str, tuple[tuple[str, str | bool], ...]] = {}
    for candidate in candidates:
        previous = stratum_registry.setdefault(candidate.stratum_digest, candidate.coordinates)
        if previous != candidate.coordinates:
            raise ExactConformanceError(
                "exact-conformance stratum registry contains a SHA-256 collision"
            )

    axis_loads: Counter[tuple[str, str | bool]] = Counter()
    joint_loads: Counter[tuple[tuple[str, str | bool], ...]] = Counter()
    selected_lineages: set[str] = set()
    greedy: list[_Candidate] = []
    for _ in range(EXACT_CONFORMANCE_MAX_TASKS):
        available = [
            candidate for candidate in candidates if candidate.base_lineage not in selected_lineages
        ]
        if not available:
            raise ExactConformanceError(
                "exact-conformance selector exhausted independent base lineages"
            )

        def selection_key(candidate: _Candidate) -> tuple[object, ...]:
            loads = [axis_loads[level] for level in candidate.coordinates]
            return (
                -sum(load == 0 for load in loads),
                max(loads),
                sum(loads),
                joint_loads[candidate.coordinates],
                tie_digests[candidate.task_id],
                candidate.base_lineage,
                candidate.task_id,
            )

        chosen = min(available, key=selection_key)
        greedy.append(chosen)
        selected_lineages.add(chosen.base_lineage)
        axis_loads.update(chosen.coordinates)
        joint_loads[chosen.coordinates] += 1

    ordered = sorted(greedy, key=lambda candidate: candidate.task_id)
    ordered_ids = tuple(candidate.task_id for candidate in ordered)
    source_census = _census(candidates)
    source_census.update(
        {
            "ineligible_task_count": len(prepared) - len(candidates),
            "loaded_base_lineage_count": len(
                {
                    task.task.lineage
                    for task in prepared
                    if type(task.task.lineage) is str and task.task.lineage
                }
            ),
            "loaded_task_count": len(prepared),
        }
    )
    selected_census = _census(ordered)
    selected_census.update(
        {
            "selected_base_lineage_count": len(selected_lineages),
            "selected_task_count": len(ordered),
        }
    )
    return ExactConformanceSelection(
        tasks=tuple(candidate.task for candidate in ordered),
        task_ids=ordered_ids,
        greedy_task_ids=tuple(candidate.task_id for candidate in greedy),
        stratum_digests={
            candidate.task_id: candidate.stratum_digest
            for candidate in sorted(candidates, key=lambda item: item.task_id)
        },
        source_manifest_sha256=source_sha,
        source_population_census=source_census,
        selected_marginal_census=selected_census,
    )


def build_exact_conformance_registry(
    selection: ExactConformanceSelection,
    source_corpus_manifest_sha256: str,
) -> dict[str, object]:
    """Build the closed-schema registry bound to a recomputed selection."""

    source_sha = _require_sha256(source_corpus_manifest_sha256, "source corpus manifest SHA-256")
    if selection.source_manifest_sha256 != source_sha:
        raise ExactConformanceError("selection and source manifest pins differ")
    if len(selection.task_ids) != EXACT_CONFORMANCE_MAX_TASKS or selection.task_ids != tuple(
        sorted(set(selection.task_ids))
    ):
        raise ExactConformanceError("selection is not an exact sorted eight-task set")
    payload: dict[str, object] = {
        "schema": EXACT_CONFORMANCE_CORPUS_SCHEMA,
        "schema_version": EXACT_CONFORMANCE_CORPUS_VERSION,
        "corpus_id": EXACT_CONFORMANCE_CORPUS_ID,
        "purpose": EXACT_CONFORMANCE_PURPOSE,
        "source_corpus_manifest_sha256": source_sha,
        "task_ids": list(selection.task_ids),
        "max_tasks": EXACT_CONFORMANCE_MAX_TASKS,
        "max_logical_variables_per_task": (EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES),
    }
    return {**payload, "record_digest": content_digest(payload)}


def selector_implementation_identity() -> dict[str, object]:
    """Return the auditable identity of the single selector implementation."""

    _, source_raw = _read_regular_file(Path(__file__), "exact-conformance implementation")
    source_sha = hashlib.sha256(source_raw).hexdigest()
    return {
        "implementation": ("isingfold.rl.data.exact_conformance.select_exact_conformance_tasks"),
        "selection_seed": EXACT_CONFORMANCE_SELECTION_SEED,
        "source_sha256": source_sha,
        "tie_domain": EXACT_CONFORMANCE_TIE_DOMAIN,
        "version": EXACT_CONFORMANCE_SELECTOR_VERSION,
    }


def _require_unchanged_selector_implementation(
    expected: Mapping[str, object],
) -> None:
    if selector_implementation_identity() != dict(expected):
        raise ExactConformanceError(
            "exact-conformance selector implementation changed during execution"
        )


def _validate_registry(record: Mapping[str, Any], source_sha: str) -> tuple[str, ...]:
    if set(record) != _REGISTRY_FIELDS:
        raise ExactConformanceError("exact-conformance registry schema fields differ")
    _verify_record(record, "exact-conformance registry")
    if (
        record["schema"] != EXACT_CONFORMANCE_CORPUS_SCHEMA
        or type(record["schema_version"]) is not int
        or record["schema_version"] != EXACT_CONFORMANCE_CORPUS_VERSION
        or record["corpus_id"] != EXACT_CONFORMANCE_CORPUS_ID
        or record["purpose"] != EXACT_CONFORMANCE_PURPOSE
        or record["source_corpus_manifest_sha256"] != source_sha
        or type(record["max_tasks"]) is not int
        or record["max_tasks"] != EXACT_CONFORMANCE_MAX_TASKS
        or type(record["max_logical_variables_per_task"]) is not int
        or record["max_logical_variables_per_task"] != EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES
    ):
        raise ExactConformanceError("exact-conformance registry has an incompatible identity")
    task_ids = record["task_ids"]
    if (
        not isinstance(task_ids, list)
        or len(task_ids) != EXACT_CONFORMANCE_MAX_TASKS
        or any(type(task_id) is not str or not task_id for task_id in task_ids)
        or task_ids != sorted(set(task_ids))
    ):
        raise ExactConformanceError(
            "exact-conformance registry must contain exactly eight sorted unique task IDs"
        )
    return tuple(task_ids)


def load_exact_conformance_registry(
    path: str | os.PathLike[str],
    *,
    expected_registry_sha256: str,
    prepared: Sequence[PreparedTask],
    source_corpus_manifest_sha256: str,
) -> tuple[list[PreparedTask], dict[str, object]]:
    """Authenticate a registry and reproduce its selection for verifier and Gate 1."""

    expected_sha = _require_sha256(
        expected_registry_sha256, "expected exact-conformance registry SHA-256"
    )
    source_sha = _require_sha256(source_corpus_manifest_sha256, "source corpus manifest SHA-256")
    _, raw = _read_regular_file(path, "exact-conformance registry")
    observed_sha = _sha256(raw)
    if observed_sha != expected_sha:
        raise ExactConformanceError(
            "exact-conformance registry differs from its externally pinned SHA-256"
        )
    record = _strict_json_object(raw, "exact-conformance registry")
    registry_task_ids = _validate_registry(record, source_sha)
    selection = select_exact_conformance_tasks(prepared, source_sha)
    if registry_task_ids != selection.task_ids:
        raise ExactConformanceError(
            "exact-conformance registry differs from the deterministic selector result"
        )
    base_lineages = sorted(task.task.lineage for task in selection.tasks)
    if (
        any(type(lineage) is not str or not lineage for lineage in base_lineages)
        or len(set(base_lineages)) != EXACT_CONFORMANCE_MAX_TASKS
    ):
        raise ExactConformanceError(
            "exact-conformance selection does not contain eight distinct base lineages"
        )
    identity: dict[str, object] = {
        "base_lineages": base_lineages,
        "corpus_id": EXACT_CONFORMANCE_CORPUS_ID,
        "exact_reproduction": True,
        "file_sha256": observed_sha,
        "max_logical_variables_per_task": (EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES),
        "max_tasks": EXACT_CONFORMANCE_MAX_TASKS,
        "path": str(Path(path)),
        "record_digest": record["record_digest"],
        "selected_marginal_census": selection.selected_marginal_census,
        "selector_implementation": selector_implementation_identity(),
        "source_population_census": selection.source_population_census,
        "source_corpus_manifest_sha256": source_sha,
        "task_count": len(selection.task_ids),
        "task_ids": list(selection.task_ids),
    }
    return list(selection.tasks), identity


def _authenticate_public_validation(
    corpus: str | os.PathLike[str],
    expected_corpus_manifest_sha256: str,
) -> tuple[tuple[PreparedTask, ...], dict[str, str]]:
    expected_sha = _require_sha256(
        expected_corpus_manifest_sha256, "expected prepared manifest SHA-256"
    )
    root = _absolute_without_symlink(Path(corpus), must_exist=True)
    if not root.is_dir():
        raise ExactConformanceError("prepared corpus root must be a directory")
    _, raw = _read_regular_file(root / "manifest.json", "prepared manifest")
    observed_sha = _sha256(raw)
    if observed_sha != expected_sha:
        raise ExactConformanceError("prepared manifest differs from its externally pinned SHA-256")
    manifest = _strict_json_object(raw, "prepared manifest")
    _verify_record(manifest, "prepared manifest")
    if (
        manifest.get("schema") != "isingfold.prepared-candidate-bank"
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 4
    ):
        raise ExactConformanceError("exact-conformance production requires prepared schema v4")
    public_file_sha256s: dict[str, str] = {}
    for filename in _PUBLIC_PREPARED_FILES:
        _, public_raw = _read_regular_file(root / filename, f"prepared public file {filename}")
        public_file_sha256s[filename] = _sha256(public_raw)
    loaded = load_prepared_partition(
        root,
        partition="val",
        include_evaluator=False,
    )
    _, raw_after_load = _read_regular_file(root / "manifest.json", "prepared manifest")
    if raw_after_load != raw:
        raise ExactConformanceError(
            "prepared manifest changed while the public validation partition was loaded"
        )
    for filename, expected_public_sha in public_file_sha256s.items():
        _, public_raw = _read_regular_file(root / filename, f"prepared public file {filename}")
        if _sha256(public_raw) != expected_public_sha:
            raise ExactConformanceError(
                f"prepared public file {filename} changed while validation was loaded"
            )
    if loaded.target_access is not None:
        raise ExactConformanceError(
            "public exact-conformance loading unexpectedly opened evaluator targets"
        )
    return loaded.tasks, {
        "manifest_record_digest": str(manifest["record_digest"]),
        "manifest_sha256": observed_sha,
    }


def load_authenticated_exact_conformance_registry(
    corpus: str | os.PathLike[str],
    *,
    expected_corpus_manifest_sha256: str,
    registry: str | os.PathLike[str],
    expected_registry_sha256: str,
) -> tuple[list[PreparedTask], dict[str, object]]:
    """Authenticate public prepared-v4 input and reproduce one pinned registry."""

    implementation_identity = selector_implementation_identity()
    tasks, source_identity = _authenticate_public_validation(
        corpus, expected_corpus_manifest_sha256
    )
    selected, registry_identity = load_exact_conformance_registry(
        registry,
        expected_registry_sha256=expected_registry_sha256,
        prepared=tasks,
        source_corpus_manifest_sha256=source_identity["manifest_sha256"],
    )
    _require_unchanged_selector_implementation(implementation_identity)
    if registry_identity["selector_implementation"] != implementation_identity:
        raise ExactConformanceError(
            "exact-conformance selector implementation changed while the registry was loaded"
        )
    return selected, registry_identity


def publish_exact_conformance_registry(
    corpus: str | os.PathLike[str],
    *,
    expected_corpus_manifest_sha256: str,
    out: str | os.PathLike[str],
) -> dict[str, object]:
    """Produce and immutably publish the only registry accepted by Gate 1."""

    implementation_identity = selector_implementation_identity()
    tasks, source_identity = _authenticate_public_validation(
        corpus, expected_corpus_manifest_sha256
    )
    selection = select_exact_conformance_tasks(tasks, source_identity["manifest_sha256"])
    registry = build_exact_conformance_registry(selection, source_identity["manifest_sha256"])
    raw = canonical_json_bytes(registry) + b"\n"
    _require_unchanged_selector_implementation(implementation_identity)
    destination = _publish_new_file(out, raw)
    return {
        "corpus_id": EXACT_CONFORMANCE_CORPUS_ID,
        "file_sha256": _sha256(raw),
        "path": str(destination),
        "record_digest": registry["record_digest"],
        "selector_implementation": implementation_identity,
        "source_corpus": source_identity,
        "source_population_census": selection.source_population_census,
        "selected_marginal_census": selection.selected_marginal_census,
        "task_ids": list(selection.task_ids),
    }


def verify_exact_conformance_registry(
    corpus: str | os.PathLike[str],
    *,
    expected_corpus_manifest_sha256: str,
    registry: str | os.PathLike[str],
    expected_registry_sha256: str,
    receipt_out: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, object], dict[str, object] | None]:
    """Independently reproduce the selection and optionally publish its receipt."""

    implementation_identity = selector_implementation_identity()
    tasks, source_identity = _authenticate_public_validation(
        corpus, expected_corpus_manifest_sha256
    )
    selected, registry_identity = load_exact_conformance_registry(
        registry,
        expected_registry_sha256=expected_registry_sha256,
        prepared=tasks,
        source_corpus_manifest_sha256=source_identity["manifest_sha256"],
    )
    selection = select_exact_conformance_tasks(tasks, source_identity["manifest_sha256"])
    if tuple(task.task_id for task in selected) != selection.task_ids:
        raise ExactConformanceError("verified registry selection changed after reproduction")
    _require_unchanged_selector_implementation(implementation_identity)
    payload: dict[str, object] = {
        "corpus_id": EXACT_CONFORMANCE_CORPUS_ID,
        "exact_reproduction": True,
        "registry": {
            "file_sha256": registry_identity["file_sha256"],
            "record_digest": registry_identity["record_digest"],
        },
        "schema": EXACT_CONFORMANCE_VERIFICATION_SCHEMA,
        "schema_version": EXACT_CONFORMANCE_VERIFICATION_VERSION,
        "selected_marginal_census": selection.selected_marginal_census,
        "selected_task_ids": list(selection.task_ids),
        "selector_implementation": implementation_identity,
        "source_corpus": source_identity,
        "source_population_census": selection.source_population_census,
    }
    receipt = {**payload, "record_digest": content_digest(payload)}
    publication: dict[str, object] | None = None
    if receipt_out is not None:
        raw = canonical_json_bytes(receipt) + b"\n"
        destination = _publish_new_file(receipt_out, raw)
        publication = {
            "receipt_file_sha256": _sha256(raw),
            "receipt_path": str(destination),
            "receipt_record_digest": receipt["record_digest"],
        }
    return receipt, publication


__all__ = [
    "EXACT_CONFORMANCE_AXES",
    "EXACT_CONFORMANCE_CORPUS_ID",
    "EXACT_CONFORMANCE_CORPUS_SCHEMA",
    "EXACT_CONFORMANCE_CORPUS_VERSION",
    "EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES",
    "EXACT_CONFORMANCE_MAX_TASKS",
    "EXACT_CONFORMANCE_PURPOSE",
    "EXACT_CONFORMANCE_SELECTION_SEED",
    "EXACT_CONFORMANCE_SELECTOR_VERSION",
    "EXACT_CONFORMANCE_TIE_DOMAIN",
    "ExactConformanceError",
    "ExactConformanceSelection",
    "build_exact_conformance_registry",
    "load_authenticated_exact_conformance_registry",
    "load_exact_conformance_registry",
    "publish_new_file",
    "publish_exact_conformance_registry",
    "read_regular_file",
    "select_exact_conformance_tasks",
    "selector_implementation_identity",
    "verify_exact_conformance_registry",
]
