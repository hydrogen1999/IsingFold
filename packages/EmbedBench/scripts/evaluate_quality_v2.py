#!/usr/bin/env python3
"""Evaluate one frozen Quality V2 checkpoint on the locked independent-audit test set.

This entry point deliberately has no training, validation, architecture-search, or release-
label mode. Candidate choices are made from model outputs plus exact feasibility and complete-
embedding qubit costs. Independently rescored schema-v2 audit probabilities are consulted only
after every policy has selected its candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
from quality_v2_paper_contract import (
    AUDIT_SOURCE_HASH_ALGORITHM,
    PaperAuditContract,
    PaperPreregistration,
    audit_source_sha256,
    load_paper_audit_contract,
    load_paper_preregistration,
    preregistered_policy_freeze_binding,
    require_exact_json,
    selection_audit_binding,
    selection_checkpoint_binding,
    strict_json_loads,
    validate_runtime_provenance,
)
from training_artifacts import runtime_provenance
from training_splits import (
    data_provenance,
    load_split_records,
    quality_problem_digest,
    quality_problem_id,
)

PAPER_ARTIFACT_SCHEMA = "embedbench.quality-v2-paper-evaluation"
PAPER_ARTIFACT_SCHEMA_VERSION = 2
SELECTION_ARTIFACT_SCHEMA = "embedbench.training-grid-selection"
SELECTION_ARTIFACT_SCHEMA_VERSION = 3
SPLIT_SCHEMA = "embedbench.split-manifest"
SPLIT_SCHEMA_VERSION = 2
PRIMARY_METRIC = "mean_finite_budget_regret"
CONFIGURATION_RULE = "lowest_canonical_cpu_replay_mean_over_all_registered_seeds"
PAPER_EVALUATION_RULE = "evaluate_every_registered_seed_checkpoint_of_winning_configuration"
REGISTERED_BUDGET_RATIOS: tuple[float | None, ...] = (1.0, 1.1, 1.25, 1.5, None)
PAPER_AUDIT_ROW_FIELDS = frozenset(
    {
        "audit_schema",
        "audit_schema_version",
        "file",
        "corpus_sha256",
        "instance_id",
        "focus",
        "candidate_signature",
        "resource_index",
        "original_index",
        "high_read_best",
        "high_read_scores",
        "reads",
        "strength_p_solve",
        "strength_success_counts",
        "provenance",
        "paper_binding",
    }
)
AUDIT_RELEASE_MANIFEST_SCHEMA = "embedbench.quality-v2-audit-release"
AUDIT_RELEASE_MANIFEST_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class _LabelFreePaperRow:
    input_index: int
    record: Mapping[str, Any]
    output: Any
    exact: Any
    resource_index: int
    original_index: int | None
    record_key: str
    problem_digest: str


@dataclass(frozen=True)
class _PaperRow(_LabelFreePaperRow):
    audit_scores: tuple[float, ...]
    audit_strength_p_solve: tuple[tuple[float, ...], ...]
    audit_strength_success_counts: tuple[tuple[int, ...], ...]
    audit_candidate_signature: str
    audit_provenance: Mapping[str, object]
    audit_paper_binding: Mapping[str, object]


@dataclass(frozen=True)
class RegisteredPaperCheckpoint:
    seed: int
    cell_id: str
    checkpoint: str
    checkpoint_sha256: str
    resolved_path: Path

    def public(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "cell_id": self.cell_id,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


@dataclass(frozen=True)
class SelectionArtifact:
    path: Path
    sha256: str
    root: Path
    document: Mapping[str, Any]
    registered_seeds: tuple[int, ...]
    winner_params: Mapping[str, Any]
    paper_checkpoints: tuple[RegisteredPaperCheckpoint, ...]


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_registered_policy_freeze(
    path: str | Path,
    expected_sha256: str,
    expected_payload: bytes,
) -> dict[str, object]:
    source = Path(path)
    payload = _capture_bytes(source, "label-free policy freeze")
    if _bytes_sha256(payload) != expected_sha256:
        raise ValueError("policy-freeze SHA-256 does not match the registered bytes")
    sidecar = source.with_suffix(".sha256")
    sidecar_payload = _capture_bytes(sidecar, "policy-freeze checksum sidecar")
    try:
        fields = sidecar_payload.decode("utf-8").split()
    except UnicodeDecodeError as error:
        raise ValueError("policy-freeze checksum sidecar is not UTF-8") from error
    if fields != [expected_sha256, source.name]:
        raise ValueError("policy-freeze checksum sidecar does not match the registered bytes")
    document = strict_json_loads(payload, location=f"label-free policy freeze {source}")
    if not isinstance(document, dict):
        raise ValueError("label-free policy freeze must be a JSON object")
    if payload != expected_payload:
        raise ValueError("fresh label-free policy replay is not byte-identical to phase 1")
    return document


def _capture_bytes(path: str | Path, kind: str) -> bytes:
    source = Path(path)
    try:
        return source.read_bytes()
    except OSError as error:
        raise ValueError(f"invalid {kind} {source}") from error


@contextmanager
def _captured_release_inputs(
    files: Sequence[str | Path],
    splits: str | Path,
) -> Iterator[tuple[list[Path], Path]]:
    """Yield a private immutable snapshot of corpus, split, and host-manifest bytes."""

    original_files = sorted((Path(path) for path in files), key=lambda path: path.name)
    names = [path.name for path in original_files]
    if len(names) != len(set(names)):
        raise ValueError("paper evaluation corpus inputs must have unique basenames")
    split_path = Path(splits)
    split_payload = _capture_bytes(split_path, "split manifest")
    split_document = strict_json_loads(
        split_payload,
        location=f"split manifest {split_path}",
    )
    if not isinstance(split_document, Mapping):
        raise ValueError(f"split manifest {split_path} must be an object")
    captured: list[tuple[Path, bytes]] = [(split_path, split_payload)]
    for path in original_files:
        corpus_payload = _capture_bytes(path, "quality corpus")
        for line_number, line in enumerate(corpus_payload.splitlines(), start=1):
            row = strict_json_loads(
                line,
                location=f"quality corpus {path}:{line_number}",
            )
            if not isinstance(row, Mapping):
                raise ValueError(f"quality corpus {path}:{line_number} row must be an object")
        captured.append((path, corpus_payload))
        manifest = Path(f"{path}.manifest.json")
        manifest_payload = _capture_bytes(manifest, "exact host generator manifest")
        manifest_document = strict_json_loads(
            manifest_payload,
            location=f"exact host generator manifest {manifest}",
        )
        if not isinstance(manifest_document, Mapping):
            raise ValueError(f"exact host generator manifest {manifest} must be an object")
        captured.append((manifest, manifest_payload))
    bundle_digest = hashlib.sha256()
    for path, payload in captured:
        name = path.name.encode("utf-8")
        bundle_digest.update(len(name).to_bytes(8, "big"))
        bundle_digest.update(name)
        bundle_digest.update(len(payload).to_bytes(8, "big"))
        bundle_digest.update(payload)
    with tempfile.TemporaryDirectory(
        prefix=f"embedbench-paper-inputs-{bundle_digest.hexdigest()[:16]}-"
    ) as directory:
        staged_root = Path(directory)
        for path, payload in captured:
            destination = staged_root / path.name
            if destination.exists():
                raise ValueError(f"paper input snapshot repeats basename {path.name!r}")
            destination.write_bytes(payload)
        yield [staged_root / path.name for path in original_files], staged_root / split_path.name


@contextmanager
def _captured_checkpoint(
    path: str | Path,
    expected_sha256: str,
) -> Iterator[tuple[Path, bytes]]:
    source = Path(path)
    payload = _capture_bytes(source, "Quality V2 checkpoint")
    actual = _bytes_sha256(payload)
    if actual != expected_sha256:
        raise ValueError("frozen checkpoint SHA-256 does not match the captured bytes")
    with tempfile.TemporaryDirectory(prefix="embedbench-paper-checkpoint-") as directory:
        staged = Path(directory) / f"{actual}{source.suffix}"
        staged.write_bytes(payload)
        yield staged, payload


@contextmanager
def _captured_audit_inputs(
    paths: Sequence[str | Path],
    manifests: Mapping[str, Mapping[str, object]],
    manifest_artifacts: Mapping[str, Mapping[str, str]],
) -> Iterator[tuple[list[Path], list[dict[str, str]]]]:
    originals = [Path(path) for path in paths]
    names = [path.name for path in originals]
    if len(names) != len(set(names)):
        raise ValueError("paper audit inputs must have unique basenames")
    captured = [(path, _capture_bytes(path, "paper audit labels")) for path in originals]
    combined = hashlib.sha256()
    for path, payload in captured:
        combined.update(path.name.encode("utf-8"))
        combined.update(payload)
    with tempfile.TemporaryDirectory(
        prefix=f"embedbench-paper-audit-{combined.hexdigest()[:16]}-"
    ) as directory:
        staged_root = Path(directory)
        artifacts: list[dict[str, str]] = []
        staged_paths: list[Path] = []
        for original, payload in captured:
            digest = _bytes_sha256(payload)
            manifest = manifests.get(original.name)
            if not isinstance(manifest, Mapping):
                raise RuntimeError(f"validated audit manifest disappeared for {original.name}")
            jsonl = manifest.get("jsonl")
            jsonl_bytes = jsonl.get("bytes") if isinstance(jsonl, Mapping) else None
            if (
                not isinstance(jsonl, Mapping)
                or set(jsonl) != {"file", "sha256", "bytes"}
                or jsonl.get("file") != original.name
                or jsonl.get("sha256") != digest
                or type(jsonl_bytes) is not int
                or jsonl_bytes != len(payload)
            ):
                raise ValueError(f"audit shard payload disagrees with manifest for {original.name}")
            records = manifest.get("records")
            assert isinstance(records, list)
            lines = payload.splitlines(keepends=True)
            if len(lines) != len(records):
                raise ValueError(
                    f"audit shard row count disagrees with manifest for {original.name}"
                )
            for index, (line, record) in enumerate(zip(lines, records, strict=True)):
                assert isinstance(record, Mapping)
                if record.get("bytes") != len(line) or record.get("sha256") != _bytes_sha256(line):
                    raise ValueError(
                        f"audit shard record {index} disagrees with manifest for {original.name}"
                    )
                decoded = strict_json_loads(
                    line,
                    location=f"audit shard {original.name} row {index}",
                )
                if not isinstance(decoded, Mapping):
                    raise ValueError(f"audit shard {original.name} row {index} is not an object")
                filename = decoded.get("file")
                instance_id = decoded.get("instance_id")
                focus = decoded.get("focus")
                if (
                    not isinstance(filename, str)
                    or Path(filename).name != filename
                    or not isinstance(instance_id, str)
                    or not instance_id
                    or isinstance(focus, bool)
                    or not isinstance(focus, int)
                ):
                    raise ValueError(
                        f"audit shard {original.name} row {index} has an invalid identity"
                    )
                identity_bytes = _canonical_audit_identity(filename, instance_id, focus)
                identity = identity_bytes.decode("utf-8")
                shard_count = manifest.get("shard_count")
                shard_index = manifest.get("shard_index")
                if (
                    record.get("identity") != identity
                    or record.get("relative_path")
                    != (
                        f"{original.name}.records/{hashlib.sha256(identity_bytes).hexdigest()}.json"
                    )
                    or isinstance(shard_count, bool)
                    or not isinstance(shard_count, int)
                    or isinstance(shard_index, bool)
                    or not isinstance(shard_index, int)
                    or int.from_bytes(hashlib.sha256(identity_bytes).digest(), "big") % shard_count
                    != shard_index
                ):
                    raise ValueError(
                        f"audit shard {original.name} row {index} identity disagrees with manifest"
                    )
                decoded_fields = set(decoded)
                if decoded_fields != PAPER_AUDIT_ROW_FIELDS:
                    raise ValueError(
                        f"audit shard {original.name} row {index} has missing or unexpected fields"
                    )
            staged = staged_root / original.name
            staged.write_bytes(payload)
            staged_paths.append(staged)
            manifest_artifact = manifest_artifacts[original.name]
            artifacts.append(
                {
                    "file": original.name,
                    "sha256": digest,
                    "manifest_file": manifest_artifact["file"],
                    "manifest_sha256": manifest_artifact["sha256"],
                }
            )
        yield staged_paths, artifacts


def _canonical_audit_identity(filename: str, instance_id: str, focus: int) -> bytes:
    return json.dumps(
        [filename, instance_id, focus],
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _validate_audit_manifest_sidecars(
    paths: Sequence[str | Path],
    *,
    selection: SelectionArtifact,
    audit_contract: PaperAuditContract,
    preregistration: PaperPreregistration,
    records: Sequence[_LabelFreePaperRow],
    release_shards: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, Mapping[str, object]], dict[str, dict[str, str]]]:
    execution = audit_contract.audit_execution
    shard_count = execution.get("shard_count")
    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count <= 0:
        raise RuntimeError("paper contract has an invalid audit shard count")
    if len(paths) != shard_count:
        raise ValueError(f"paper evaluation requires exactly {shard_count} audit shards")
    expected_contract = audit_contract.public_binding()
    expected_preregistration = preregistration.public_binding()
    expected_selection = selection_audit_binding(
        selection.document,
        selection_file=selection.path.name,
        selection_sha256=selection.sha256,
    )
    expected_source = {
        "algorithm": AUDIT_SOURCE_HASH_ALGORITHM,
        "sha256": selection.document["audit_source_sha256"],
    }
    manifests: dict[str, Mapping[str, object]] = {}
    manifest_artifacts: dict[str, dict[str, str]] = {}
    identities: list[str] = []
    seen_shards: set[int] = set()
    for raw_path in paths:
        path = Path(raw_path)
        if path.name in manifests:
            raise ValueError("paper audit shard basenames must be unique")
        manifest_path = path.with_name(f"{path.name}.manifest.json")
        payload = _capture_bytes(manifest_path, "paper audit shard manifest")
        release_entry = release_shards.get(path.name)
        if (
            not isinstance(release_entry, Mapping)
            or release_entry.get("manifest_file") != manifest_path.name
            or release_entry.get("manifest_sha256") != _bytes_sha256(payload)
            or release_entry.get("manifest_bytes") != len(payload)
        ):
            raise ValueError(
                f"audit shard manifest {manifest_path} differs from the external release commitment"
            )
        document = strict_json_loads(
            payload,
            location=f"paper audit shard manifest {manifest_path}",
        )
        expected_fields = {
            "schema",
            "schema_version",
            "audit_contract",
            "preregistration",
            "selection",
            "audit_source",
            "shard_count",
            "shard_index",
            "assignment",
            "canonical_audit_identity",
            "merge_order",
            "complete_fixed_test_record_count",
            "record_count",
            "records",
            "jsonl",
        }
        if not isinstance(document, Mapping) or set(document) != expected_fields:
            raise ValueError(f"audit shard manifest {manifest_path} has invalid fields")
        require_exact_json(
            document.get("audit_contract"),
            expected_contract,
            location=f"audit shard manifest {manifest_path} contract binding",
        )
        require_exact_json(
            document.get("preregistration"),
            expected_preregistration,
            location=f"audit shard manifest {manifest_path} preregistration binding",
        )
        require_exact_json(
            document.get("selection"),
            expected_selection,
            location=f"audit shard manifest {manifest_path} selection binding",
        )
        require_exact_json(
            document.get("audit_source"),
            expected_source,
            location=f"audit shard manifest {manifest_path} source binding",
        )
        shard_index = document.get("shard_index")
        complete_record_count = document.get("complete_fixed_test_record_count")
        if (
            document.get("schema") != "embedbench.quality-v2-audit-shard"
            or type(document.get("schema_version")) is not int
            or document.get("schema_version") != 2
            or document.get("shard_count") != shard_count
            or isinstance(shard_index, bool)
            or not isinstance(shard_index, int)
            or not 0 <= shard_index < shard_count
            or shard_index in seen_shards
            or document.get("assignment") != execution["assignment"]
            or document.get("canonical_audit_identity") != execution["canonical_audit_identity"]
            or document.get("merge_order") != execution["merge_order"]
            or type(complete_record_count) is not int
            or complete_record_count != len(records)
        ):
            raise ValueError(f"audit shard manifest {manifest_path} violates the frozen contract")
        jsonl = document.get("jsonl")
        if (
            not isinstance(jsonl, Mapping)
            or jsonl.get("file") != release_entry.get("file")
            or jsonl.get("sha256") != release_entry.get("sha256")
            or jsonl.get("bytes") != release_entry.get("bytes")
        ):
            raise ValueError(
                f"audit shard manifest {manifest_path} disagrees with the external "
                "release commitment"
            )
        raw_records = document.get("records")
        record_count = document.get("record_count")
        if (
            not isinstance(raw_records, list)
            or type(record_count) is not int
            or record_count != len(raw_records)
        ):
            raise ValueError(f"audit shard manifest {manifest_path} has invalid record coverage")
        shard_identities: list[str] = []
        for index, entry in enumerate(raw_records):
            if not isinstance(entry, Mapping) or set(entry) != {
                "identity",
                "relative_path",
                "sha256",
                "bytes",
            }:
                raise ValueError(f"audit shard manifest {manifest_path} record {index} is invalid")
            identity = entry.get("identity")
            relative_path = entry.get("relative_path")
            byte_count = entry.get("bytes")
            if (
                not isinstance(identity, str)
                or not isinstance(relative_path, str)
                or not relative_path
                or Path(relative_path).is_absolute()
                or any(part in {"", ".", ".."} for part in Path(relative_path).parts)
                or not _valid_sha256(entry.get("sha256"))
                or isinstance(byte_count, bool)
                or not isinstance(byte_count, int)
                or byte_count <= 0
            ):
                raise ValueError(f"audit shard manifest {manifest_path} record {index} is invalid")
            identity_value = strict_json_loads(
                identity,
                location=f"audit shard manifest {manifest_path} record identity",
            )
            if (
                not isinstance(identity_value, list)
                or len(identity_value) != 3
                or not isinstance(identity_value[0], str)
                or not isinstance(identity_value[1], str)
                or isinstance(identity_value[2], bool)
                or not isinstance(identity_value[2], int)
                or _canonical_audit_identity(*identity_value).decode() != identity
                or int.from_bytes(hashlib.sha256(identity.encode()).digest(), "big") % shard_count
                != shard_index
            ):
                raise ValueError(
                    f"audit shard manifest {manifest_path} record assignment is invalid"
                )
            shard_identities.append(identity)
        if shard_identities != sorted(shard_identities):
            raise ValueError(f"audit shard manifest {manifest_path} is not canonically ordered")
        identities.extend(shard_identities)
        seen_shards.add(shard_index)
        manifests[path.name] = document
        manifest_artifacts[path.name] = {
            "file": manifest_path.name,
            "sha256": _bytes_sha256(payload),
        }
    if seen_shards != set(range(shard_count)):
        raise ValueError("paper audit manifests do not cover every registered shard exactly once")
    expected_identities = sorted(
        _canonical_audit_identity(
            str(row.record["_file"]),
            str(row.record["instance_id"]),
            int(row.record["focus"]),
        ).decode()
        for row in records
    )
    if sorted(identities) != expected_identities or len(identities) != len(set(identities)):
        raise ValueError("paper audit manifests do not exactly cover the fixed test identities")
    return manifests, manifest_artifacts


def _load_audit_release_commitment(
    path: str | Path,
    expected_sha256: str,
    audit_paths: Sequence[str | Path],
    *,
    selection: SelectionArtifact,
    audit_contract: PaperAuditContract,
    preregistration: PaperPreregistration,
) -> tuple[dict[str, Mapping[str, object]], dict[str, object]]:
    """Verify the independently supplied root before opening any committed shard payload."""

    source = Path(path)
    payload = _capture_bytes(source, "external audit-release commitment")
    actual_sha256 = _bytes_sha256(payload)
    if actual_sha256 != expected_sha256:
        raise ValueError("audit-release manifest differs from its independently supplied SHA-256")
    document = strict_json_loads(payload, location=f"audit-release manifest {source}")
    expected_fields = {
        "schema",
        "schema_version",
        "commitment_scope",
        "audit_contract",
        "preregistration",
        "selection",
        "audit_source",
        "shard_count",
        "shards",
    }
    expected_contract = audit_contract.public_binding()
    expected_selection = selection_audit_binding(
        selection.document,
        selection_file=selection.path.name,
        selection_sha256=selection.sha256,
    )
    expected_source = {
        "algorithm": AUDIT_SOURCE_HASH_ALGORITHM,
        "sha256": selection.document["audit_source_sha256"],
    }
    shard_count = audit_contract.audit_execution.get("shard_count")
    if (
        not isinstance(document, Mapping)
        or set(document) != expected_fields
        or document.get("schema") != AUDIT_RELEASE_MANIFEST_SCHEMA
        or type(document.get("schema_version")) is not int
        or document.get("schema_version") != AUDIT_RELEASE_MANIFEST_SCHEMA_VERSION
        or document.get("commitment_scope") != "exact_audit_jsonl_and_shard_manifest_bytes"
        or document.get("shard_count") != shard_count
    ):
        raise ValueError("audit-release manifest violates the frozen release schema")
    require_exact_json(
        document.get("audit_contract"),
        expected_contract,
        location="audit-release manifest contract binding",
    )
    require_exact_json(
        document.get("preregistration"),
        preregistration.public_binding(),
        location="audit-release manifest preregistration binding",
    )
    require_exact_json(
        document.get("selection"),
        expected_selection,
        location="audit-release manifest selection binding",
    )
    require_exact_json(
        document.get("audit_source"),
        expected_source,
        location="audit-release manifest verifier-source binding",
    )
    raw_shards = document.get("shards")
    if not isinstance(raw_shards, list) or len(raw_shards) != shard_count:
        raise ValueError("audit-release manifest lacks complete registered shard coverage")
    supplied_files = [Path(raw).name for raw in audit_paths]
    if len(supplied_files) != len(set(supplied_files)):
        raise ValueError("paper audit inputs must have unique basenames")
    by_file: dict[str, Mapping[str, object]] = {}
    expected_entry_fields = {
        "shard_index",
        "file",
        "sha256",
        "bytes",
        "manifest_file",
        "manifest_sha256",
        "manifest_bytes",
    }
    for expected_index, entry in enumerate(raw_shards):
        if not isinstance(entry, Mapping) or set(entry) != expected_entry_fields:
            raise ValueError("audit-release manifest contains an invalid shard entry")
        filename = entry.get("file")
        byte_count = entry.get("bytes")
        manifest_bytes = entry.get("manifest_bytes")
        if (
            entry.get("shard_index") != expected_index
            or not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or filename in by_file
            or not _valid_sha256(entry.get("sha256"))
            or type(byte_count) is not int
            or byte_count < 0
            or entry.get("manifest_file") != f"{filename}.manifest.json"
            or not _valid_sha256(entry.get("manifest_sha256"))
            or type(manifest_bytes) is not int
            or manifest_bytes <= 0
        ):
            raise ValueError("audit-release manifest contains an invalid shard entry")
        by_file[filename] = entry
    if set(by_file) != set(supplied_files):
        raise ValueError("audit-release manifest does not match the supplied audit shards")
    return by_file, {
        "schema": AUDIT_RELEASE_MANIFEST_SCHEMA,
        "schema_version": AUDIT_RELEASE_MANIFEST_SCHEMA_VERSION,
        "file": source.name,
        "sha256": actual_sha256,
        "trust_model": "sha256_supplied_out_of_band",
        "shard_count": shard_count,
    }


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_number(value: object, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{location} must be finite and numeric")
    return float(value)


def _finite_fraction(value: object, location: str) -> Fraction:
    _finite_number(value, location)
    return Fraction(str(value))


def _independent_exhaustive_ground_energy(
    problem: Mapping[str, object],
    *,
    problem_digest: str,
    cache: dict[str, tuple[str, float]],
    max_variables: int,
    batch_states: int,
) -> float:
    """Enumerate every spin assignment without trusting ``problem.e0`` or audit output."""

    raw_h = problem.get("h")
    raw_j = problem.get("J")
    if not isinstance(raw_h, Mapping) or not isinstance(raw_j, list):
        raise ValueError("paper audit problem has an invalid Ising payload")
    if not raw_h:
        raise ValueError("paper audit exhaustive problem has no variables")
    h: dict[int, float] = {}
    for raw_node, raw_bias in raw_h.items():
        if not isinstance(raw_node, str):
            raise ValueError("paper audit problem.h keys must be canonical integer strings")
        try:
            node = int(raw_node)
        except ValueError as error:
            raise ValueError(
                "paper audit problem.h keys must be canonical integer strings"
            ) from error
        if str(node) != raw_node or node in h:
            raise ValueError("paper audit problem.h keys must be canonical integer strings")
        h[node] = _finite_number(raw_bias, f"problem.h[{raw_node!r}]")
    nodes = sorted(h)
    if len(nodes) > max_variables:
        raise ValueError("paper audit exhaustive ground reference exceeds its registered bound")
    node_index = {node: index for index, node in enumerate(nodes)}
    couplers: list[tuple[int, int, float]] = []
    seen_edges: set[tuple[int, int]] = set()
    for index, edge in enumerate(raw_j):
        if not isinstance(edge, list) or len(edge) != 3:
            raise ValueError(f"problem.J[{index}] must be a canonical edge triple")
        left, right, raw_weight = edge
        if (
            isinstance(left, bool)
            or not isinstance(left, int)
            or isinstance(right, bool)
            or not isinstance(right, int)
            or left >= right
            or left not in h
            or right not in h
            or (left, right) in seen_edges
        ):
            raise ValueError(f"problem.J[{index}] is not a canonical logical edge")
        seen_edges.add((left, right))
        couplers.append(
            (
                node_index[left],
                node_index[right],
                _finite_number(raw_weight, f"problem.J[{index}][2]"),
            )
        )

    canonical_payload = json.dumps(
        {
            "h": [[node, h[node]] for node in nodes],
            "J": [[nodes[u], nodes[v], weight] for u, v, weight in couplers],
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    problem_signature = _bytes_sha256(canonical_payload)
    cached = cache.get(problem_digest)
    if cached is not None:
        if cached[0] != problem_signature:
            raise ValueError("paper audit problem digest aliases different Ising payloads")
        return cached[1]

    magnitude_bound = sum(abs(value) for value in h.values()) + sum(
        abs(weight) for _, _, weight in couplers
    )
    if not math.isfinite(magnitude_bound):
        raise ValueError("paper audit exhaustive energy bound is not finite")
    fields = np.asarray([h[node] for node in nodes], dtype=np.float64)
    bit_positions = np.arange(len(nodes), dtype=np.uint64)
    state_count = 1 << len(nodes)
    best = math.inf
    for first in range(0, state_count, batch_states):
        stop = min(first + batch_states, state_count)
        states = np.arange(first, stop, dtype=np.uint64)
        bits = ((states[:, None] >> bit_positions) & 1).astype(np.int8, copy=False)
        spins = 1 - 2 * bits
        energies = spins @ fields
        for left, right, weight in couplers:
            energies += weight * spins[:, left] * spins[:, right]
        candidate = float(energies.min())
        if candidate < best:
            best = candidate
    if not math.isfinite(best) or best < -magnitude_bound - 1e-9:
        raise RuntimeError("independent exhaustive ground computation became invalid")
    cache[problem_digest] = (problem_signature, best)
    return best


def _validate_min_cut_ground_certificate(
    raw_h: Mapping[object, object],
    raw_j: Sequence[object],
    ground: Mapping[str, object],
    registered: Mapping[str, object],
) -> None:
    import networkx as nx

    certificate = ground.get("certificate")
    required_certificate_fields = {
        "algorithm",
        "capacity_scale",
        "cut_value_scaled",
        "energy_numerator",
        "energy_denominator",
        "source_variables",
        "sink_variables",
        "witness_sha256",
    }
    if not isinstance(certificate, Mapping) or set(certificate) != required_certificate_fields:
        raise ValueError("paper audit min-cut ground certificate is incomplete")
    if certificate.get("algorithm") != registered.get("algorithm"):
        raise ValueError("paper audit min-cut certificate algorithm is not registered")
    h: dict[int, Fraction] = {}
    for raw_node, raw_bias in raw_h.items():
        if not isinstance(raw_node, str):
            raise ValueError("paper audit problem.h keys must be canonical integer strings")
        try:
            node = int(raw_node)
        except ValueError as error:
            raise ValueError(
                "paper audit problem.h keys must be canonical integer strings"
            ) from error
        if str(node) != raw_node or node in h:
            raise ValueError("paper audit problem.h keys must be canonical integer strings")
        h[node] = _finite_fraction(raw_bias, f"problem.h[{raw_node!r}]")
    if not h:
        raise ValueError("paper audit min-cut problem has no variables")
    j: dict[tuple[int, int], Fraction] = {}
    for index, edge in enumerate(raw_j):
        if not isinstance(edge, list) or len(edge) != 3:
            raise ValueError(f"problem.J[{index}] must be a canonical edge triple")
        u, v, raw_weight = edge
        if (
            isinstance(u, bool)
            or not isinstance(u, int)
            or isinstance(v, bool)
            or not isinstance(v, int)
            or u >= v
        ):
            raise ValueError(f"problem.J[{index}] is not a canonical logical edge")
        weight = _finite_fraction(raw_weight, f"problem.J[{index}]")
        if weight > 0:
            raise ValueError("paper audit min-cut certificate requires ferromagnetic couplers")
        key = (u, v)
        if key in j or u not in h or v not in h:
            raise ValueError("paper audit min-cut certificate has an invalid logical edge")
        j[key] = weight

    def strict_integer(name: str, *, positive: bool = False) -> int:
        value = certificate.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"paper audit min-cut certificate {name} must be an integer")
        if positive and value <= 0:
            raise ValueError(f"paper audit min-cut certificate {name} must be positive")
        return value

    scale = strict_integer("capacity_scale", positive=True)
    expected_scale = 1
    for value in (*h.values(), *j.values()):
        expected_scale = math.lcm(expected_scale, value.denominator)
    if scale != expected_scale:
        raise ValueError("paper audit min-cut certificate capacity scale is not canonical")
    cut_value = strict_integer("cut_value_scaled")
    numerator = strict_integer("energy_numerator")
    denominator = strict_integer("energy_denominator", positive=True)
    exact_energy = Fraction(numerator, denominator)
    if exact_energy.numerator != numerator or exact_energy.denominator != denominator:
        raise ValueError("paper audit min-cut certificate energy fraction is not canonical")

    def partition(name: str) -> list[int]:
        value = certificate.get(name)
        if (
            not isinstance(value, list)
            or any(isinstance(node, bool) or not isinstance(node, int) for node in value)
            or value != sorted(set(value))
        ):
            raise ValueError(f"paper audit min-cut certificate {name} is not canonical")
        return value

    source_variables = partition("source_variables")
    sink_variables = partition("sink_variables")
    if set(source_variables) & set(sink_variables) or set(source_variables) | set(
        sink_variables
    ) != set(h):
        raise ValueError("paper audit min-cut partition does not cover every variable once")
    spins = {node: (-1 if node in set(source_variables) else 1) for node in sorted(h)}
    witness_payload = json.dumps(
        [[node, spins[node]] for node in sorted(spins)],
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if certificate.get("witness_sha256") != _bytes_sha256(witness_payload):
        raise ValueError("paper audit min-cut witness SHA-256 is invalid")
    witness_energy = sum(h[node] * spins[node] for node in h) + sum(
        weight * spins[u] * spins[v] for (u, v), weight in j.items()
    )
    if witness_energy != exact_energy:
        raise ValueError("paper audit min-cut witness energy is inconsistent")
    if not math.isclose(
        float(exact_energy),
        _finite_number(ground.get("energy"), "paper audit ground energy"),
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise ValueError("paper audit min-cut rational energy is inconsistent")

    source = ("embedbench", "source")
    sink = ("embedbench", "sink")
    graph = nx.DiGraph()
    graph.add_nodes_from((source, sink, *h))
    for node, field in h.items():
        source_capacity = int(max(2 * field, Fraction(0)) * scale)
        sink_capacity = int(max(-2 * field, Fraction(0)) * scale)
        if source_capacity:
            graph.add_edge(source, node, capacity=source_capacity)
        if sink_capacity:
            graph.add_edge(node, sink, capacity=sink_capacity)
    for (u, v), weight in j.items():
        capacity = int((-2 * weight) * scale)
        if capacity:
            graph.add_edge(u, v, capacity=capacity)
            graph.add_edge(v, u, capacity=capacity)
    recomputed_cut, _ = nx.minimum_cut(
        graph,
        source,
        sink,
        capacity="capacity",
        flow_func=nx.algorithms.flow.preflow_push,
    )
    if isinstance(recomputed_cut, bool) or int(recomputed_cut) != cut_value:
        raise ValueError("paper audit min-cut certificate is not globally minimal")
    reduction_energy = (
        Fraction(cut_value, scale) - sum(abs(field) for field in h.values()) + sum(j.values())
    )
    if reduction_energy != exact_energy:
        raise ValueError("paper audit min-cut cut value and rational energy disagree")


def _certificate_problem_from_audit_payload(problem: Mapping[str, object]):
    """Build the exact WP6 problem identity from the authenticated audit payload."""

    from paper_verifier.ground_certificate import IsingProblem

    raw_h = problem.get("h")
    raw_j = problem.get("J")
    if type(raw_h) is not dict or type(raw_j) is not list:
        raise ValueError("paper audit certificate problem has an invalid Ising payload")
    if not raw_h:
        raise ValueError("paper audit certificate problem has no variables")
    h: dict[int, float] = {}
    for raw_node, raw_bias in raw_h.items():
        if type(raw_node) is not str:
            raise ValueError(
                "paper audit certificate problem.h keys must be canonical integer strings"
            )
        try:
            node = int(raw_node)
        except ValueError as error:
            raise ValueError(
                "paper audit certificate problem.h keys must be canonical integer strings"
            ) from error
        if str(node) != raw_node or node in h:
            raise ValueError(
                "paper audit certificate problem.h keys must be canonical integer strings"
            )
        if type(raw_bias) is not float:
            raise TypeError("paper audit certificate problem coefficients must be binary64 floats")
        h[node] = _finite_number(raw_bias, f"problem.h[{raw_node!r}]")

    quadratic: list[tuple[int, int, float]] = []
    seen_edges: set[tuple[int, int]] = set()
    for index, raw_edge in enumerate(raw_j):
        if type(raw_edge) is not list or len(raw_edge) != 3:
            raise ValueError(f"problem.J[{index}] must be a canonical edge triple")
        left, right, raw_weight = raw_edge
        if (
            type(left) is not int
            or type(right) is not int
            or left >= right
            or left not in h
            or right not in h
            or (left, right) in seen_edges
        ):
            raise ValueError(f"problem.J[{index}] is not a canonical logical edge")
        if type(raw_weight) is not float:
            raise TypeError("paper audit certificate problem coefficients must be binary64 floats")
        seen_edges.add((left, right))
        quadratic.append((left, right, _finite_number(raw_weight, f"problem.J[{index}][2]")))

    try:
        return IsingProblem(
            variables=tuple(sorted(h)),
            linear=tuple((node, h[node]) for node in sorted(h)),
            quadratic=tuple(quadratic),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("paper audit certificate problem is not canonical") from error


def _validate_embedded_ground_certificate(
    problem: Mapping[str, object],
    ground: Mapping[str, object],
    registered: Mapping[str, object],
    *,
    problem_digest: str,
    certificate_cache: dict[str, tuple[str, float]],
) -> None:
    """Replay a closed, inline WP6 proof without trusting any receipt or path."""

    expected_fields = registered.get("envelope_fields")
    if (
        type(expected_fields) is not list
        or type(registered.get("format_version")) is not int
        or registered.get("format_version") != 1
        or any(type(field) is not str for field in expected_fields)
        or len(expected_fields) != len(set(expected_fields))
        or type(ground) is not dict
        or set(ground) != set(expected_fields)
    ):
        raise ValueError("paper audit embedded ground certificate envelope is not closed")
    certificate = ground.get("certificate")
    proof = ground.get("proof")
    certificate_sha256 = ground.get("certificate_sha256")
    proof_sha256 = ground.get("proof_artifact_sha256")
    if type(certificate) is not dict or type(proof) is not dict:
        raise TypeError("paper audit embedded ground certificate artifacts must be JSON objects")
    if not _valid_sha256(certificate_sha256) or not _valid_sha256(proof_sha256):
        raise ValueError("paper audit embedded ground certificate digests are invalid")
    if certificate.get("certificate_sha256") != certificate_sha256:
        raise ValueError("paper audit embedded certificate repeats a mismatched digest")
    if (
        certificate.get("schema") != registered.get("certificate_schema")
        or type(certificate.get("schema_version")) is not int
        or certificate.get("schema_version") != registered.get("certificate_schema_version")
    ):
        raise ValueError("paper audit embedded ground certificate schema is not registered")
    accepted_statuses = registered.get("accepted_statuses")
    status = certificate.get("status")
    if (
        type(accepted_statuses) is not list
        or any(type(value) is not str for value in accepted_statuses)
        or type(status) is not str
        or status not in accepted_statuses
    ):
        raise ValueError(
            "paper audit embedded ground certificate status is paper-ineligible; "
            "only planted_proof and exact_enumeration are accepted"
        )
    if status == "exact_enumeration" and certificate.get("method") != (
        "gray_code_exhaustive_enumeration"
    ):
        raise ValueError("paper audit exact-enumeration certificate method is not registered")

    from paper_verifier.ground_certificate import GroundStateCertificate
    from paper_verifier.schema import canonical_sha256

    try:
        parsed_certificate = GroundStateCertificate.from_dict(certificate)
    except (TypeError, ValueError) as error:
        raise ValueError("paper audit embedded certificate digest did not verify") from error
    if parsed_certificate.digest != certificate_sha256:
        raise ValueError("paper audit embedded certificate digest did not verify")
    if canonical_sha256(proof) != proof_sha256:
        raise ValueError("paper audit embedded proof artifact digest did not verify")

    certified_problem = _certificate_problem_from_audit_payload(problem)
    cache_key = f"embedded:{problem_digest}:{certificate_sha256}:{proof_sha256}"
    cached = certificate_cache.get(cache_key)
    if cached is not None:
        if cached[0] != certified_problem.problem_sha256:
            raise ValueError("paper audit problem digest aliases different certified problems")
        exact_energy = cached[1]
    else:
        from paper_verifier.ground_certificate import verify_ground_state_certificate

        try:
            verified = verify_ground_state_certificate(
                certified_problem,
                certificate,
                proof,
                expected_certificate_sha256=str(certificate_sha256),
                expected_proof_artifact_sha256=str(proof_sha256),
            )
        except (TypeError, ValueError) as error:
            raise ValueError("paper audit embedded ground-state proof did not verify") from error
        if verified.status != status or not verified.quality_eligible:
            raise ValueError("paper audit embedded ground-state proof is not quality eligible")
        exact_energy = verified.energy.to_float()
        certificate_cache[cache_key] = (certified_problem.problem_sha256, exact_energy)

    raw_reference = problem.get("e0")
    if type(raw_reference) is not float:
        raise TypeError("paper audit certificate problem.e0 must be a binary64 float")
    if not math.isclose(
        exact_energy,
        _finite_number(raw_reference, "problem.e0"),
        rel_tol=1e-10,
        abs_tol=1e-10,
    ):
        raise ValueError("problem.e0 disagrees with the embedded exact ground certificate")


def _validate_ground_reference(
    problem: Mapping[str, object],
    ground: object,
    audit_contract: PaperAuditContract,
    *,
    problem_digest: str,
    exhaustive_cache: dict[str, tuple[str, float]],
) -> None:
    if not isinstance(ground, Mapping):
        raise ValueError("paper audit row has no certified exact ground reference")
    raw_h = problem.get("h")
    raw_j = problem.get("J")
    if not isinstance(raw_h, Mapping) or not isinstance(raw_j, list):
        raise ValueError("paper audit problem has an invalid Ising payload")
    registry = audit_contract.evaluation["accepted_ground_references"]
    embedded_registration = (
        registry.get("embedded_ground_state_certificate_v1")
        if isinstance(registry, Mapping)
        else None
    )
    if type(ground) is dict and set(ground) == {
        "certificate",
        "proof",
        "certificate_sha256",
        "proof_artifact_sha256",
    }:
        if not isinstance(embedded_registration, Mapping):
            raise ValueError("paper audit embedded ground certificate is not registered")
        _validate_embedded_ground_certificate(
            problem,
            ground,
            embedded_registration,
            problem_digest=problem_digest,
            certificate_cache=exhaustive_cache,
        )
        return

    method = ground.get("method")
    registered = registry.get(method) if isinstance(registry, Mapping) else None
    if (
        ground.get("status") != "certified_exact"
        or not isinstance(method, str)
        or not isinstance(registered, Mapping)
        or isinstance(ground.get("n_variables"), bool)
        or not isinstance(ground.get("n_variables"), int)
        or ground.get("n_variables") != len(raw_h)
        or not math.isclose(
            _finite_number(ground.get("energy"), "paper audit ground energy"),
            _finite_number(problem.get("e0"), "problem.e0"),
            rel_tol=1e-10,
            abs_tol=1e-10,
        )
    ):
        raise ValueError("paper audit requires a registered certified exact ground reference")
    if method == "exhaustive_enumeration":
        if (
            set(ground) != {"status", "method", "energy", "n_variables"}
            or registered.get("format_version") != "legacy_v1_1_v1_2"
            or registered.get("certificate") != "not_required"
            or registered.get("enumeration") != "all_spin_assignments_vectorized_float64_v1"
            or isinstance(registered.get("max_variables"), bool)
            or not isinstance(registered.get("max_variables"), int)
            or isinstance(registered.get("batch_states"), bool)
            or not isinstance(registered.get("batch_states"), int)
            or int(registered["max_variables"]) <= 0
            or int(registered["batch_states"]) <= 0
        ):
            raise ValueError("paper audit exhaustive ground reference has unexpected evidence")
        recomputed = _independent_exhaustive_ground_energy(
            problem,
            problem_digest=problem_digest,
            cache=exhaustive_cache,
            max_variables=int(registered["max_variables"]),
            batch_states=int(registered["batch_states"]),
        )
        if not math.isclose(
            recomputed,
            _finite_number(ground.get("energy"), "paper audit ground energy"),
            rel_tol=1e-10,
            abs_tol=1e-10,
        ):
            raise ValueError(
                "paper audit exhaustive ground energy disagrees with independent enumeration"
            )
    elif method == "ferromagnetic_s_t_min_cut":
        if (
            set(ground) != {"status", "method", "energy", "n_variables", "certificate"}
            or registered.get("format_version") != "legacy_v1_1_v1_2"
        ):
            raise ValueError("paper audit min-cut ground reference has unexpected fields")
        _validate_min_cut_ground_certificate(raw_h, raw_j, ground, registered)
    else:
        raise ValueError("paper audit ground-reference method is not implemented")


_GROUND_REFERENCE_FORMATS = (
    "legacy_exhaustive_enumeration_v1_1_v1_2",
    "legacy_ferromagnetic_s_t_min_cut_v1_1_v1_2",
    "wp6_exact_enumeration_v1",
    "wp6_planted_proof_v1",
)


def _ground_reference_coverage(
    rows: Sequence[_PaperRow],
    audit_contract: PaperAuditContract,
) -> dict[str, object]:
    counts = {name: 0 for name in _GROUND_REFERENCE_FORMATS}
    for row in rows:
        ground = row.audit_paper_binding.get("ground_reference")
        if not isinstance(ground, Mapping):  # pragma: no cover - validated construction guard
            raise RuntimeError("validated paper row lost its ground reference")
        if set(ground) == {
            "certificate",
            "proof",
            "certificate_sha256",
            "proof_artifact_sha256",
        }:
            certificate = ground.get("certificate")
            status = certificate.get("status") if isinstance(certificate, Mapping) else None
            if status == "exact_enumeration":
                name = "wp6_exact_enumeration_v1"
            elif status == "planted_proof":
                name = "wp6_planted_proof_v1"
            else:  # pragma: no cover - rejected by proof validation
                raise RuntimeError("validated WP6 ground reference has an ineligible status")
        elif ground.get("method") == "exhaustive_enumeration":
            name = "legacy_exhaustive_enumeration_v1_1_v1_2"
        elif ground.get("method") == "ferromagnetic_s_t_min_cut":
            name = "legacy_ferromagnetic_s_t_min_cut_v1_1_v1_2"
        else:  # pragma: no cover - rejected by ground-reference validation
            raise RuntimeError("validated ground reference has an unregistered format")
        counts[name] += 1
    return {
        "policy": audit_contract.evaluation["ground_reference_policy"],
        "accepted_registry": dict(audit_contract.evaluation["accepted_ground_references"]),
        "generic_certified_optimal_accepted": False,
        "records": len(rows),
        "by_format": counts,
    }


def _load_audit_paper_bindings(
    paths: Sequence[str | Path],
    key_function,
    expected_policy_indices: Mapping[tuple[str, str, int], tuple[int, int]],
    *,
    expected_candidate_counts: Mapping[tuple[str, str, int], int],
    registered_strength_count: int,
    registered_reads_per_strength: int,
) -> tuple[
    dict[tuple[str, str, int], Mapping[str, object]],
    dict[tuple[str, str, int], tuple[tuple[float, ...], ...]],
    dict[tuple[str, str, int], tuple[tuple[int, ...], ...]],
]:
    bindings: dict[tuple[str, str, int], Mapping[str, object]] = {}
    strength_scores: dict[tuple[str, str, int], tuple[tuple[float, ...], ...]] = {}
    strength_counts: dict[tuple[str, str, int], tuple[tuple[int, ...], ...]] = {}
    for raw_path in paths:
        path = Path(raw_path)
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                row = strict_json_loads(
                    line,
                    location=f"{path}:{line_number} audit row",
                )
                if not isinstance(row, Mapping):
                    raise ValueError(f"{path}:{line_number}: audit row must be an object")
                key = key_function(row)
                expected_indices = expected_policy_indices.get(key)
                expected_candidate_count = expected_candidate_counts.get(key)
                resource_index = row.get("resource_index")
                original_index = row.get("original_index")
                scores = row.get("high_read_scores")
                high_read_best = row.get("high_read_best")
                if (
                    expected_indices is None
                    or isinstance(resource_index, bool)
                    or not isinstance(resource_index, int)
                    or isinstance(original_index, bool)
                    or not isinstance(original_index, int)
                    or (resource_index, original_index) != expected_indices
                ):
                    raise ValueError(
                        f"{path}:{line_number}: audit policy indices disagree with the "
                        "frozen quality record"
                    )
                if (
                    not isinstance(scores, list)
                    or expected_candidate_count is None
                    or expected_candidate_count <= 0
                    or len(scores) != expected_candidate_count
                ):
                    raise ValueError(
                        f"{path}:{line_number}: audit candidate support disagrees with the "
                        "frozen quality record"
                    )
                if (
                    any(
                        isinstance(score, bool)
                        or not isinstance(score, (int, float))
                        or not math.isfinite(float(score))
                        for score in scores
                    )
                    or isinstance(high_read_best, bool)
                    or not isinstance(high_read_best, int)
                    or high_read_best
                    != max(range(len(scores)), key=lambda index: float(scores[index]))
                ):
                    raise ValueError(f"{path}:{line_number}: audit high_read_best is inconsistent")
                raw_strength_scores = row.get("strength_p_solve")
                raw_strength_counts = row.get("strength_success_counts")
                reads = row.get("reads")
                if (
                    isinstance(reads, bool)
                    or not isinstance(reads, int)
                    or reads <= 0
                    or reads != registered_reads_per_strength
                    or not isinstance(raw_strength_scores, list)
                    or len(raw_strength_scores) != len(scores)
                    or not isinstance(raw_strength_counts, list)
                    or len(raw_strength_counts) != len(scores)
                    or any(
                        not isinstance(candidate_scores, list)
                        or len(candidate_scores) != registered_strength_count
                        for candidate_scores in raw_strength_scores
                    )
                    or any(
                        not isinstance(candidate_counts, list)
                        or len(candidate_counts) != registered_strength_count
                        for candidate_counts in raw_strength_counts
                    )
                    or any(
                        isinstance(score, bool)
                        or not isinstance(score, (int, float))
                        or not math.isfinite(float(score))
                        or not 0.0 <= float(score) <= 1.0
                        for candidate_scores in raw_strength_scores
                        for score in candidate_scores
                    )
                    or any(
                        isinstance(count, bool)
                        or not isinstance(count, int)
                        or not 0 <= count <= reads
                        for candidate_counts in raw_strength_counts
                        for count in candidate_counts
                    )
                ):
                    if reads != registered_reads_per_strength:
                        raise ValueError(
                            f"{path}:{line_number}: audit_labels.reads_per_strength "
                            "disagrees with the top-level read count"
                        )
                    raise ValueError(
                        f"{path}:{line_number}: audit requires complete integer success-count "
                        "evidence for every candidate and registered strength"
                    )
                authenticated_strength_scores = tuple(
                    tuple(float(score) for score in candidate_scores)
                    for candidate_scores in raw_strength_scores
                )
                authenticated_strength_counts = tuple(
                    tuple(int(count) for count in candidate_counts)
                    for candidate_counts in raw_strength_counts
                )
                if any(
                    count / reads != score
                    for candidate_scores, candidate_counts in zip(
                        authenticated_strength_scores,
                        authenticated_strength_counts,
                        strict=True,
                    )
                    for score, count in zip(candidate_scores, candidate_counts, strict=True)
                ):
                    raise ValueError(
                        f"{path}:{line_number}: audit probabilities disagree with integer "
                        "success-count evidence"
                    )
                if any(
                    max(candidate_scores) != float(aggregate)
                    for candidate_scores, aggregate in zip(
                        authenticated_strength_scores,
                        scores,
                        strict=True,
                    )
                ):
                    raise ValueError(
                        f"{path}:{line_number}: audit strength_p_solve disagrees "
                        "with high_read_scores"
                    )
                binding = row.get("paper_binding")
                if not isinstance(binding, Mapping):
                    raise ValueError(f"{path}:{line_number}: paper audit row has no paper binding")
                if key in bindings:
                    raise ValueError(f"{path}:{line_number}: duplicate paper audit binding")
                bindings[key] = dict(binding)
                strength_scores[key] = authenticated_strength_scores
                strength_counts[key] = authenticated_strength_counts
    return bindings, strength_scores, strength_counts


def _validate_audit_paper_binding(
    record: Mapping[str, Any],
    binding: Mapping[str, object],
    *,
    selection: SelectionArtifact,
    audit_contract: PaperAuditContract,
    preregistration: PaperPreregistration,
    exhaustive_cache: dict[str, tuple[str, float]],
) -> None:
    expected_fields = {
        "audit_contract",
        "preregistration",
        "selection",
        "audit_source",
        "runtime_provenance",
        "ground_reference",
        "realized_strengths",
        "realized_seeds",
    }
    if set(binding) != expected_fields:
        raise ValueError("paper audit binding has missing or unexpected fields")
    require_exact_json(
        binding.get("audit_contract"),
        audit_contract.public_binding(),
        location="paper audit contract binding",
    )
    require_exact_json(
        binding.get("preregistration"),
        preregistration.public_binding(),
        location="paper audit preregistration binding",
    )
    expected_selection = selection_audit_binding(
        selection.document,
        selection_file=selection.path.name,
        selection_sha256=selection.sha256,
    )
    require_exact_json(
        binding.get("selection"),
        expected_selection,
        location="paper audit selection binding",
    )
    expected_source = {
        "algorithm": AUDIT_SOURCE_HASH_ALGORITHM,
        "sha256": selection.document["audit_source_sha256"],
    }
    require_exact_json(
        binding.get("audit_source"),
        expected_source,
        location="paper audit source binding",
    )

    validate_runtime_provenance(
        binding.get("runtime_provenance"),
        location="paper audit runtime",
        expected_device=str(audit_contract.evaluation["device"]),
    )

    problem = record.get("problem")
    ground = binding.get("ground_reference")
    if not isinstance(problem, Mapping):
        raise ValueError("paper audit row has no certified exact ground reference")
    _validate_ground_reference(
        problem,
        ground,
        audit_contract,
        problem_digest=quality_problem_digest(record),
        exhaustive_cache=exhaustive_cache,
    )

    from embedbench.embedding import LogicalProblem
    from embedbench.surrogate import default_strength_grid

    raw_h = problem.get("h")
    raw_j = problem.get("J")
    assert isinstance(raw_h, Mapping)
    if not isinstance(raw_j, list):
        raise ValueError("paper audit problem.J must be a list")
    logical = LogicalProblem.from_dicts(
        {int(key): value for key, value in raw_h.items()},
        {(u, v): weight for u, v, weight in raw_j},
    )
    expected_strengths = default_strength_grid(logical, 4)
    strengths = binding.get("realized_strengths")
    if not isinstance(strengths, list):
        raise ValueError("paper audit realized strength schedule is stale")
    require_exact_json(
        strengths,
        expected_strengths,
        location="paper audit realized strength schedule",
    )
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("paper audit record has no candidates")
    base_seed = int(audit_contract.audit_labels["base_seed"])
    expected_seeds = [
        [
            (base_seed + 7919 * candidate_index + 31 * strength_index) % (2**31)
            for strength_index in range(len(expected_strengths))
        ]
        for candidate_index in range(len(candidates))
    ]
    require_exact_json(
        binding.get("realized_seeds"),
        expected_seeds,
        location="paper audit realized seed schedule",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("freeze", "score"),
        help="phase 1 writes the label-free policy freeze; phase 2 verifies it and scores",
    )
    parser.add_argument("files", nargs="+", help="immutable quality release JSONL files")
    parser.add_argument("--splits", required=True, help="fixed schema-v2 split manifest")
    parser.add_argument("--checkpoint", required=True, help="one selected Quality V2 checkpoint")
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="pre-registered SHA-256 of the frozen checkpoint bytes",
    )
    parser.add_argument(
        "--selection",
        required=True,
        help="validation-only training-grid selection artifact",
    )
    parser.add_argument(
        "--selection-sha256",
        required=True,
        help="pre-registered SHA-256 of the immutable selection artifact",
    )
    parser.add_argument(
        "--selection-root",
        default=".",
        help="staged root against which winner.paper_checkpoints paths resolve",
    )
    parser.add_argument(
        "--audit-labels",
        nargs="*",
        default=[],
        help="phase-2 externally committed schema-v2 high-read audit JSONL files",
    )
    parser.add_argument(
        "--audit-release-manifest",
        help="phase-2 audit-release manifest whose digest was retained outside the release",
    )
    parser.add_argument(
        "--audit-release-manifest-sha256",
        help="independently supplied SHA-256 of --audit-release-manifest",
    )
    parser.add_argument(
        "--policy-freeze",
        help="phase-1 label-free policy freeze; required for phase 2",
    )
    parser.add_argument(
        "--preregistration-manifest",
        help="single paper preregistration manifest; required for phase 2",
    )
    parser.add_argument(
        "--preregistration-manifest-sha256",
        help="mandatory digest retained independently before audit-label generation",
    )
    parser.add_argument(
        "--audit-contract",
        required=True,
        help="pre-frozen schema-v1 Quality V2 paper-audit contract",
    )
    parser.add_argument(
        "--audit-contract-sha256",
        required=True,
        help="pre-registered SHA-256 of the immutable paper-audit contract",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--partition",
        choices=("test",),
        default="test",
        help="locked by protocol; development partitions are not accepted",
    )
    parser.add_argument(
        "--evaluation-mode",
        choices=("paper",),
        default="paper",
        help="locked by protocol; release-label development evaluation is not accepted",
    )
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(argv)
    if not _valid_sha256(args.checkpoint_sha256):
        parser.error("--checkpoint-sha256 must be lowercase hexadecimal SHA-256")
    if not _valid_sha256(args.selection_sha256):
        parser.error("--selection-sha256 must be lowercase hexadecimal SHA-256")
    if not _valid_sha256(args.audit_contract_sha256):
        parser.error("--audit-contract-sha256 must be lowercase hexadecimal SHA-256")
    if args.phase == "freeze":
        if args.audit_labels:
            parser.error("phase freeze refuses --audit-labels")
        if (
            args.audit_release_manifest is not None
            or args.audit_release_manifest_sha256 is not None
        ):
            parser.error("phase freeze refuses an audit-release commitment")
        if args.policy_freeze is not None:
            parser.error("phase freeze refuses a pre-existing policy freeze")
        if (
            args.preregistration_manifest is not None
            or args.preregistration_manifest_sha256 is not None
        ):
            parser.error("phase freeze refuses a paper preregistration manifest")
    else:
        if not args.audit_labels:
            parser.error("phase score requires --audit-labels")
        if not args.audit_release_manifest or not _valid_sha256(args.audit_release_manifest_sha256):
            parser.error(
                "phase score requires --audit-release-manifest and lowercase hexadecimal "
                "--audit-release-manifest-sha256"
            )
        if not args.policy_freeze:
            parser.error("phase score requires --policy-freeze")
        if not args.preregistration_manifest or not _valid_sha256(
            args.preregistration_manifest_sha256
        ):
            parser.error(
                "phase score requires --preregistration-manifest and lowercase hexadecimal "
                "--preregistration-manifest-sha256 supplied out of band"
            )
    return args


def _load_schema_v2_manifest(path: str | Path) -> dict[str, Any]:
    split_path = Path(path)
    try:
        payload = split_path.read_bytes()
    except OSError as error:
        raise ValueError(f"invalid split manifest {split_path}") from error
    document = strict_json_loads(payload, location=f"split manifest {split_path}")
    if not isinstance(document, dict):
        raise ValueError(f"{split_path}: split manifest must be an object")
    if (
        document.get("schema") != SPLIT_SCHEMA
        or document.get("schema_version") != SPLIT_SCHEMA_VERSION
    ):
        raise ValueError("paper evaluation requires embedbench.split-manifest schema version 2")
    return document


def _fixed_split_provenance(
    files: Sequence[str | Path],
    splits: str | Path,
) -> tuple[dict[str, Any], dict[str, object]]:
    """Validate immutable input identities without materializing any corpus row."""

    document = _load_schema_v2_manifest(splits)
    split_provenance = document.get("provenance")
    split_inputs = split_provenance.get("inputs") if isinstance(split_provenance, Mapping) else None
    if not isinstance(split_inputs, list):
        raise ValueError("schema-v2 split manifest has no provenance inputs")
    committed_files = {entry.get("file") for entry in split_inputs if isinstance(entry, Mapping)}
    supplied_files = {Path(path).name for path in files}
    if committed_files != supplied_files or len(committed_files) != len(split_inputs):
        raise ValueError(
            "paper evaluation requires every corpus input committed by the fixed split"
        )
    return document, data_provenance(files, splits)


def _load_locked_test_records(
    files: Sequence[str | Path],
    splits: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, object], dict[str, int]]:
    """Validate the complete split contract, then materialize only tagged test records."""

    document, provenance = _fixed_split_provenance(files, splits)
    partitions = load_split_records(
        files,
        splits,
        group_key=quality_problem_id,
        record_group_key=quality_problem_digest,
    )
    split_tables = document.get("splits")
    if not isinstance(split_tables, Mapping):
        raise ValueError(f"{splits}: expected an object field named 'splits'")

    test_records: list[dict[str, Any]] = []
    for raw_path in files:
        path = Path(raw_path)
        table = split_tables.get(path.name)
        if not isinstance(table, Mapping):
            raise ValueError(f"{splits}: no split table for {path.name}")
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                record = strict_json_loads(
                    line,
                    location=f"{path}:{line_number} quality row",
                )
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: quality row must be an object")
                instance_id = record.get("instance_id")
                if table.get(str(instance_id)) == "test":
                    tagged = dict(record)
                    tagged["_file"] = path.name
                    tagged["_line_number"] = line_number
                    test_records.append(tagged)

    if len(test_records) != len(partitions.test):
        raise RuntimeError("locked test materialization disagrees with the validated split")
    if not test_records:
        raise ValueError("paper evaluation requires a non-empty fixed test partition")
    return (
        test_records,
        provenance,
        {
            "train": len(partitions.train),
            "val": len(partitions.val),
            "test": len(partitions.test),
        },
    )


def _provenance_inputs(value: object, location: str) -> dict[str, str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{location} corpus inputs must be a list")
    result: dict[str, str] = {}
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{location} corpus input {index} must be an object")
        filename = entry.get("file")
        digest = entry.get("sha256")
        if not isinstance(filename, str) or not filename or not _valid_sha256(digest):
            raise ValueError(f"{location} corpus input {index} is invalid")
        if filename in result:
            raise ValueError(f"{location} repeats corpus input {filename!r}")
        result[filename] = str(digest)
    return result


def _validated_data_provenance(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{location} data provenance must be an object")
    split = value.get("split_manifest")
    if not isinstance(split, Mapping):
        raise ValueError(f"{location} data provenance has no split manifest")
    split_file = split.get("file")
    if (
        not isinstance(split_file, str)
        or not split_file
        or Path(split_file).name != split_file
        or not _valid_sha256(split.get("sha256"))
        or split.get("schema") != SPLIT_SCHEMA
        or split.get("schema_version") != SPLIT_SCHEMA_VERSION
    ):
        raise ValueError(f"{location} split provenance is invalid")
    inputs = _provenance_inputs(value.get("corpus_inputs"), location)
    if not inputs:
        raise ValueError(f"{location} corpus provenance is empty")
    if any(Path(filename).name != filename for filename in inputs):
        raise ValueError(f"{location} corpus provenance paths must be basenames")
    return {
        "split_manifest": {
            "file": split_file,
            "sha256": split["sha256"],
            "schema": SPLIT_SCHEMA,
            "schema_version": SPLIT_SCHEMA_VERSION,
        },
        "corpus_inputs": [
            {"file": filename, "sha256": digest} for filename, digest in sorted(inputs.items())
        ],
    }


def _selection_checkpoint_path(checkpoint: object, root: Path) -> tuple[str, Path]:
    if not isinstance(checkpoint, str) or not checkpoint:
        raise ValueError("selection winner checkpoint path must be a non-empty string")
    relative = Path(checkpoint)
    if (
        relative.is_absolute()
        or relative.as_posix() != checkpoint
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError("selection winner checkpoint path must be canonical and relative")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("selection winner checkpoint path escapes --selection-root")
    return checkpoint, resolved


def load_selection_artifact(
    path: str | Path,
    expected_sha256: str,
    root: str | Path,
    audit_contract: PaperAuditContract,
    *,
    selection_payload: bytes | None = None,
) -> SelectionArtifact:
    selection_path = Path(path)
    if selection_payload is None:
        selection_payload = _capture_bytes(selection_path, "selection artifact")
    actual_sha256 = _bytes_sha256(selection_payload)
    if actual_sha256 != expected_sha256:
        raise ValueError("selection SHA-256 does not match --selection-sha256")
    document = strict_json_loads(
        selection_payload,
        location=f"selection artifact {selection_path}",
    )
    if not isinstance(document, dict):
        raise ValueError("selection artifact must be a JSON object")
    if (
        document.get("schema") != SELECTION_ARTIFACT_SCHEMA
        or document.get("schema_version") != SELECTION_ARTIFACT_SCHEMA_VERSION
    ):
        raise ValueError("paper evaluation requires a training-grid selection schema version 3")
    require_exact_json(
        document.get("paper_audit_contract"),
        audit_contract.public_binding(),
        location="selection paper-audit contract binding",
    )
    expected_selection = audit_contract.selection
    selected_contract = {
        "artifact_schema": document.get("schema"),
        "artifact_schema_version": document.get("schema_version"),
        "grid_id": document.get("grid_id"),
        "grid_path": document.get("grid_path"),
        "stage": document.get("stage"),
        "registered_cell_count": document.get("registered_cell_count"),
        "registered_seeds": document.get("registered_seeds"),
        "validation_only": all(
            document.get(field) is expected
            for field, expected in {
                "test_records_parsed": True,
                "test_records_parsed_for_partition_routing_only": True,
                "test_records_encoded": False,
                "test_labels_used_for_selection": False,
                "test_evaluated": False,
            }.items()
        ),
    }
    selection_mismatches = [
        field
        for field, expected in expected_selection.items()
        if selected_contract.get(field) != expected
    ]
    if selection_mismatches:
        raise ValueError(
            "selection artifact differs from the frozen paper selection contract: "
            f"{selection_mismatches}"
        )
    current_audit_source = audit_source_sha256(Path(__file__).parents[1])
    if (
        document.get("audit_source_hash_algorithm") != AUDIT_SOURCE_HASH_ALGORITHM
        or document.get("audit_source_sha256") != current_audit_source
    ):
        raise ValueError("selection artifact has stale paper-audit source provenance")

    required_hashes = ("grid_sha256", "source_sha256", "selector_sha256")
    invalid_hashes = [name for name in required_hashes if not _valid_sha256(document.get(name))]
    if invalid_hashes:
        raise ValueError(f"selection artifact has invalid provenance hashes: {invalid_hashes}")
    selector = Path(__file__).with_name("select_training_grid.py")
    if document["selector_sha256"] != _sha256(selector):
        raise ValueError("selection artifact has stale selector provenance")
    for field in ("grid_id", "grid_file", "stage"):
        value = document.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"selection artifact has invalid {field}")
    if Path(str(document["grid_file"])).name != document["grid_file"]:
        raise ValueError("selection grid_file must be a basename")

    expected_contract = {
        "primary_metric": PRIMARY_METRIC,
        "direction": "minimize",
        "configuration_rule": CONFIGURATION_RULE,
        "validation_metric_source": "canonical_cpu_checkpoint_replay",
        "paper_evaluation_rule": PAPER_EVALUATION_RULE,
        "test_records_parsed": True,
        "test_records_parsed_for_partition_routing_only": True,
        "test_records_encoded": False,
        "test_labels_used_for_selection": False,
        "test_evaluated": False,
    }
    mismatches = [
        field for field, expected in expected_contract.items() if document.get(field) != expected
    ]
    if mismatches:
        raise ValueError(f"selection artifact violates the validation-only contract: {mismatches}")
    validation_replay_registration = audit_contract.training_registration.get("validation_replay")
    if not isinstance(validation_replay_registration, Mapping):
        raise ValueError("paper audit contract has no canonical validation replay")
    require_exact_json(
        document.get("validation_replay_policy"),
        validation_replay_registration.get("canonical_runtime"),
        location="selection canonical validation replay policy",
    )
    _validated_data_provenance(document.get("data_provenance"), "selection")

    raw_seeds = document.get("registered_seeds")
    if (
        not isinstance(raw_seeds, list)
        or len(raw_seeds) < 2
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in raw_seeds)
        or len(set(raw_seeds)) != len(raw_seeds)
    ):
        raise ValueError("selection artifact has invalid registered seeds")
    registered_seeds = tuple(raw_seeds)

    winner = document.get("winner")
    if not isinstance(winner, Mapping):
        raise ValueError("selection artifact has no winner object")
    winner_params = winner.get("params")
    if not isinstance(winner_params, Mapping):
        raise ValueError("selection winner has no parameter object")
    for field in ("arch", "objective_variant"):
        if not isinstance(winner_params.get(field), str) or not winner_params[field]:
            raise ValueError(f"selection winner has invalid {field}")

    raw_checkpoints = winner.get("paper_checkpoints")
    if not isinstance(raw_checkpoints, list) or len(raw_checkpoints) != len(registered_seeds):
        raise ValueError(
            "selection winner.paper_checkpoints must cover every registered seed exactly once"
        )
    selection_root = Path(root).resolve()
    checkpoints: list[RegisteredPaperCheckpoint] = []
    cell_ids: set[str] = set()
    resolved_paths: set[Path] = set()
    for expected_seed, row in zip(registered_seeds, raw_checkpoints, strict=True):
        if not isinstance(row, Mapping) or set(row) != {
            "seed",
            "cell_id",
            "checkpoint",
            "checkpoint_sha256",
        }:
            raise ValueError("selection winner.paper_checkpoints contains an invalid entry")
        seed = row.get("seed")
        cell_id = row.get("cell_id")
        if seed != expected_seed or isinstance(seed, bool):
            raise ValueError(
                "selection winner.paper_checkpoints does not follow registered seed order"
            )
        if not isinstance(cell_id, str) or not cell_id or cell_id in cell_ids:
            raise ValueError("selection winner.paper_checkpoints has an invalid cell ID")
        checkpoint, resolved = _selection_checkpoint_path(row.get("checkpoint"), selection_root)
        checkpoint_sha256 = row.get("checkpoint_sha256")
        if not _valid_sha256(checkpoint_sha256):
            raise ValueError("selection winner.paper_checkpoints has an invalid SHA-256")
        if resolved in resolved_paths:
            raise ValueError("selection winner.paper_checkpoints repeats a checkpoint path")
        cell_ids.add(cell_id)
        resolved_paths.add(resolved)
        checkpoints.append(
            RegisteredPaperCheckpoint(
                seed=int(seed),
                cell_id=cell_id,
                checkpoint=checkpoint,
                checkpoint_sha256=str(checkpoint_sha256),
                resolved_path=resolved,
            )
        )

    selected_seed = winner.get("selected_seed")
    selected = next(
        (checkpoint for checkpoint in checkpoints if checkpoint.seed == selected_seed),
        None,
    )
    if selected is None or any(
        winner.get(field) != expected
        for field, expected in {
            "selected_cell_id": selected.cell_id,
            "checkpoint": selected.checkpoint,
            "checkpoint_sha256": selected.checkpoint_sha256,
        }.items()
    ):
        raise ValueError("selection deployment checkpoint disagrees with winner.paper_checkpoints")

    return SelectionArtifact(
        path=selection_path.resolve(),
        sha256=actual_sha256,
        root=selection_root,
        document=document,
        registered_seeds=registered_seeds,
        winner_params=dict(winner_params),
        paper_checkpoints=tuple(checkpoints),
    )


def selection_public_binding(
    selection: SelectionArtifact,
    checkpoint: RegisteredPaperCheckpoint,
) -> dict[str, object]:
    return selection_checkpoint_binding(
        selection.document,
        selection_file=selection.path.name,
        selection_sha256=selection.sha256,
        checkpoint=checkpoint.public(),
    )


def bind_registered_checkpoint(
    selection: SelectionArtifact,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    *,
    checkpoint_payload: bytes | None = None,
) -> RegisteredPaperCheckpoint:
    resolved = Path(checkpoint_path).resolve()
    registered = next(
        (entry for entry in selection.paper_checkpoints if entry.resolved_path == resolved),
        None,
    )
    if registered is None:
        raise ValueError("checkpoint is not registered in selection winner.paper_checkpoints")
    if checkpoint_sha256 != registered.checkpoint_sha256:
        raise ValueError(
            "frozen checkpoint SHA-256 does not match winner.paper_checkpoints registration"
        )
    actual_sha256 = (
        _bytes_sha256(checkpoint_payload) if checkpoint_payload is not None else _sha256(resolved)
    )
    if actual_sha256 != registered.checkpoint_sha256:
        raise ValueError("frozen checkpoint bytes do not match winner.paper_checkpoints SHA-256")
    return registered


def _validate_selection_data_contract(
    selection: SelectionArtifact,
    current_provenance: Mapping[str, object],
) -> None:
    selected = _validated_data_provenance(selection.document.get("data_provenance"), "selection")
    current = _validated_data_provenance(current_provenance, "evaluation")
    if selected["split_manifest"] != current["split_manifest"]:
        raise ValueError("selection split provenance does not match evaluation inputs")
    if selected["corpus_inputs"] != current["corpus_inputs"]:
        raise ValueError("selection corpus provenance does not match evaluation inputs")


def _validate_checkpoint_selection_binding(
    metadata: Mapping[str, Any],
    arch: str,
    selection: SelectionArtifact,
    checkpoint: RegisteredPaperCheckpoint,
) -> None:
    seed = metadata.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed != checkpoint.seed:
        raise ValueError("checkpoint seed disagrees with winner.paper_checkpoints registration")
    if arch != selection.winner_params["arch"]:
        raise ValueError("checkpoint architecture disagrees with selection winner")
    if metadata.get("objective_variant") != selection.winner_params["objective_variant"]:
        raise ValueError("checkpoint objective variant disagrees with selection winner")


def _validate_checkpoint_data_contract(
    metadata: Mapping[str, Any],
    current_provenance: Mapping[str, object],
    *,
    split_sha256: str,
) -> None:
    if metadata.get("split_sha256") != split_sha256:
        raise ValueError("checkpoint was selected against a different split manifest")
    checkpoint_provenance = metadata.get("data_provenance")
    if not isinstance(checkpoint_provenance, Mapping):
        raise ValueError("Quality V2 checkpoint has no data_provenance contract")
    checkpoint_split = checkpoint_provenance.get("split_manifest")
    current_split = current_provenance.get("split_manifest")
    if not isinstance(checkpoint_split, Mapping) or not isinstance(current_split, Mapping):
        raise ValueError("paper evaluation requires schema-v2 split provenance")
    for field in ("file", "sha256", "schema", "schema_version"):
        if checkpoint_split.get(field) != current_split.get(field):
            raise ValueError(f"checkpoint split provenance mismatch for {field}")
    checkpoint_inputs = _provenance_inputs(checkpoint_provenance.get("corpus_inputs"), "checkpoint")
    current_inputs = _provenance_inputs(current_provenance.get("corpus_inputs"), "evaluation")
    if checkpoint_inputs != current_inputs:
        raise ValueError("checkpoint corpus provenance does not match evaluation inputs")


def _validate_checkpoint_protocol(
    metadata: Mapping[str, Any],
    arch: str,
    *,
    model_neighbour_feats: bool,
) -> dict[str, Any]:
    if metadata.get("evaluation_support") != "full":
        raise ValueError("paper evaluation requires a full-support Quality V2 checkpoint")
    if metadata.get("primary_validation_metric") != "mean_finite_budget_regret":
        raise ValueError("checkpoint was not selected with the registered validation metric")
    if metadata.get("test_evaluated") is not False:
        raise ValueError("checkpoint metadata does not prove the test partition stayed locked")
    if metadata.get("test_partition_encoded") is not False:
        raise ValueError("checkpoint metadata does not prove the test partition stayed unencoded")
    if metadata.get("selection_contract") != (
        "quality_mean_or_lcb_subject_to_exact_feasibility_and_budget-v1"
    ):
        raise ValueError("checkpoint has an unsupported Quality V2 selection contract")
    if metadata.get("registered_budget_ratios") != list(REGISTERED_BUDGET_RATIOS):
        raise ValueError("checkpoint registered budget ratios disagree with paper protocol")
    if metadata.get("auxiliary_heads_used_for_selection") is not False:
        raise ValueError("paper selector must not consume auxiliary heads")
    training_scope = metadata.get("training_scope")
    if (
        not isinstance(training_scope, Mapping)
        or training_scope.get("limit") is not None
        or training_scope.get("full_fixed_train_validation") is not True
    ):
        raise ValueError(
            "paper evaluation requires a checkpoint trained on full fixed train/validation"
        )
    training_run_id = metadata.get("training_run_id")
    if (
        not isinstance(training_run_id, str)
        or len(training_run_id) != 32
        or any(character not in "0123456789abcdef" for character in training_run_id)
    ):
        raise ValueError("checkpoint training run ID is missing or invalid")
    best_epoch = metadata.get("best_epoch")
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or best_epoch <= 0:
        raise ValueError("checkpoint best epoch must be a positive integer")
    selected_metric = metadata.get("selected_validation_metric")
    if (
        isinstance(selected_metric, bool)
        or not isinstance(selected_metric, (int, float))
        or not math.isfinite(float(selected_metric))
        or float(selected_metric) < 0.0
    ):
        raise ValueError("checkpoint selected validation metric must be finite and non-negative")
    raw_lcb_z = metadata.get("lcb_z")
    if (
        isinstance(raw_lcb_z, bool)
        or not isinstance(raw_lcb_z, (int, float))
        or not math.isfinite(float(raw_lcb_z))
        or float(raw_lcb_z) < 0.0
    ):
        raise ValueError("checkpoint lcb_z must be an explicit finite non-negative number")

    preprocessing = metadata.get("preprocessing")
    if not isinstance(preprocessing, Mapping):
        raise ValueError("Quality V2 checkpoint has no preprocessing contract")
    required = {
        "deploy_view",
        "deploy_max_free",
        "neighbour_feats",
        "encoder",
        "hamiltonian_context",
    }
    if not required <= preprocessing.keys():
        raise ValueError("Quality V2 checkpoint preprocessing contract is incomplete")
    deploy_view = preprocessing.get("deploy_view")
    neighbour_feats = preprocessing.get("neighbour_feats")
    deploy_max_free = preprocessing.get("deploy_max_free")
    encoder = preprocessing.get("encoder")
    if not isinstance(deploy_view, bool) or not isinstance(neighbour_feats, bool):
        raise ValueError("checkpoint preprocessing flags must be boolean")
    if (
        isinstance(deploy_max_free, bool)
        or not isinstance(deploy_max_free, int)
        or deploy_max_free < 0
    ):
        raise ValueError("checkpoint deploy_max_free must be a non-negative integer")
    expected_encoder = "heterogeneous" if arch == "hetero" else "chain"
    if encoder != expected_encoder:
        raise ValueError("checkpoint encoder metadata disagrees with its model architecture")
    if arch == "hetero" and neighbour_feats:
        raise ValueError("heterogeneous Quality V2 does not support neighbour features")
    if neighbour_feats != model_neighbour_feats:
        raise ValueError(
            "checkpoint neighbour-features preprocessing disagrees with its model config"
        )
    from embedbench.hamiltonian_context import hamiltonian_context_contract

    if preprocessing.get("hamiltonian_context") != hamiltonian_context_contract():
        raise ValueError("checkpoint Hamiltonian-context contract is incompatible")
    return dict(preprocessing)


def _validate_deployment_host_contract(
    metadata: Mapping[str, Any],
    preprocessing: Mapping[str, Any],
    files: Sequence[str | Path],
) -> None:
    contract = metadata.get("deployment_host_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("Quality V2 checkpoint has no deployment host contract")
    if not preprocessing["deploy_view"]:
        if contract.get("mode") != "not_applied":
            raise ValueError("non-deploy-view checkpoint has an inconsistent host contract")
        return
    if contract.get("mode") != "pristine_topology_reconstruction":
        raise ValueError("deploy-view paper evaluation requires a pristine host contract")
    raw_sources = contract.get("source_manifests")
    if not isinstance(raw_sources, list):
        raise ValueError("deploy-view checkpoint host contract has no source manifests")
    expected: dict[str, Mapping[str, object]] = {}
    for entry in raw_sources:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("file"), str):
            raise ValueError("deploy-view checkpoint contains an invalid source manifest")
        filename = str(entry["file"])
        if filename in expected:
            raise ValueError(f"deploy-view checkpoint repeats source manifest {filename!r}")
        expected[filename] = entry
    if set(expected) != {Path(path).name for path in files}:
        raise ValueError("deploy-view checkpoint host contract does not cover this corpus")

    for raw_path in files:
        corpus = Path(raw_path)
        manifest = Path(f"{corpus}.manifest.json")
        try:
            payload = manifest.read_bytes()
        except OSError as error:
            raise ValueError(f"invalid generator manifest {manifest}") from error
        document = strict_json_loads(payload, location=f"generator manifest {manifest}")
        config = document.get("config") if isinstance(document, Mapping) else None
        if not isinstance(config, Mapping):
            raise ValueError(f"generator manifest {manifest} has no config object")
        corpus_digest = _sha256(corpus)
        entry = expected[corpus.name]
        if document.get("sha256") != corpus_digest:
            raise ValueError(f"generator manifest {manifest} does not match its corpus")
        if entry.get("corpus_sha256") != corpus_digest:
            raise ValueError(
                "checkpoint deployment contract was recorded for different corpus bytes"
            )
        if entry.get("manifest_sha256") != _sha256(manifest):
            raise ValueError("checkpoint deployment contract was recorded for a different manifest")
        for field in ("defect_qubits", "defect_couplers"):
            value = config.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"generator manifest {manifest} has invalid {field}")
            if float(value) != 0.0 or entry.get(field) != 0.0:
                raise ValueError("deploy-view evaluation refuses unreconstructable host defects")


def _verified_exact_host_contract(
    files: Sequence[str | Path],
) -> tuple[dict[str, object], dict[str, Mapping[str, object]]]:
    """Establish that pristine reconstruction is sound for exact paper masks.

    A topology name alone cannot reconstruct a sampled defective host. The immutable release
    generator manifests therefore have to prove that every evaluated corpus was generated on
    the pristine host. Their hashes are retained in the result artifact for auditability.
    """

    sources: list[dict[str, object]] = []
    by_file: dict[str, Mapping[str, object]] = {}
    for raw_path in files:
        corpus = Path(raw_path)
        manifest = Path(f"{corpus}.manifest.json")
        try:
            payload = manifest.read_bytes()
        except OSError as error:
            raise ValueError(
                f"exact host verification requires generator manifest {manifest}"
            ) from error
        document = strict_json_loads(payload, location=f"generator manifest {manifest}")
        config = document.get("config") if isinstance(document, Mapping) else None
        if not isinstance(config, Mapping):
            raise ValueError(f"exact host manifest {manifest} has no config object")
        corpus_digest = _sha256(corpus)
        if document.get("sha256") != corpus_digest:
            raise ValueError(f"exact host manifest {manifest} does not match its corpus")
        topology = config.get("topology")
        size = config.get("size")
        if not isinstance(topology, str) or not topology:
            raise ValueError(f"exact host manifest {manifest} has no topology")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ValueError(f"exact host manifest {manifest} has invalid size")
        defect_values: dict[str, float] = {}
        for field in ("defect_qubits", "defect_couplers"):
            value = config.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError(f"exact host manifest {manifest} has invalid {field}")
            defect_values[field] = float(value)
        if any(value != 0.0 for value in defect_values.values()):
            raise ValueError(
                "exact host masks refuse a defective corpus without the realized host graph"
            )
        source = {
            "file": corpus.name,
            "corpus_sha256": corpus_digest,
            "manifest_sha256": _sha256(manifest),
            "topology": topology,
            "size": size,
            **defect_values,
        }
        sources.append(source)
        by_file[corpus.name] = source
    return {"mode": "verified_pristine_manifests", "sources": sources}, by_file


def _validate_record_host_identity(
    record: Mapping[str, Any],
    exact_host_sources: Mapping[str, Mapping[str, object]],
) -> None:
    source = exact_host_sources.get(str(record.get("_file")))
    if source is None:
        raise ValueError("test record is absent from the exact host contract")
    if record.get("topology") != source.get("topology") or record.get("size") != source.get("size"):
        raise ValueError("test record topology/size disagrees with its exact host manifest")


def _validate_policy_indices(record: Mapping[str, Any], exact: Any) -> tuple[int, int | None]:
    candidate_count = len(exact)
    resource = record.get("resource_index")
    if isinstance(resource, bool) or not isinstance(resource, int):
        raise ValueError("quality record resource_index must be an integer")
    if not 0 <= resource < candidate_count:
        raise ValueError("quality record resource_index lies outside candidate support")
    valid = [index for index in range(candidate_count) if bool(exact.feasible[index])]
    if not valid:
        raise ValueError("quality record has no exactly feasible candidate")
    expected_resource = min(
        valid,
        key=lambda index: (
            int(exact.total_qubits[index]) - exact.frozen_total_qubits,
            int(exact.cycle_rank[index])
            + int(exact.total_qubits[index])
            - exact.frozen_total_qubits
            - 1,
            index,
        ),
    )
    if resource != expected_resource:
        raise ValueError("resource_index disagrees with the registered resource policy")
    if not bool(exact.feasible[resource]):
        raise ValueError("registered resource candidate is exactly infeasible")

    original = record.get("original_index", -1)
    if isinstance(original, bool) or not isinstance(original, int):
        raise ValueError("quality record original_index must be an integer")
    if original == -1:
        return resource, None
    if not 0 <= original < candidate_count:
        raise ValueError("quality record original_index lies outside candidate support")
    return resource, original


_MODEL_INPUT_FIELDS = frozenset(
    {
        "instance_id",
        "focus",
        "n_vars",
        "window_nodes",
        "window_edges",
        "frozen",
        "all_chains",
        "frozen_adjacency",
        "neighbours",
        "edge_J",
        "neighbour_degree",
        "neighbour_chain_size",
        "neighbour_h",
        "candidates",
        "Q",
        "source",
        "topology",
        "size",
        "difficulty",
        "l_cap",
        "focus_h",
        "problem",
        "defect_qubits",
        "defect_couplers",
    }
)


def _sanitized_model_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Project a corpus row onto the registered label-free model-input schema."""

    sanitized = {key: value for key, value in record.items() if key in _MODEL_INPUT_FIELDS}
    problem = sanitized.get("problem")
    if not isinstance(problem, Mapping):
        raise ValueError("model input requires a problem object")
    if "h" not in problem or "J" not in problem:
        raise ValueError("model input requires problem.h and problem.J")
    sanitized["problem"] = {"h": problem["h"], "J": problem["J"]}
    return sanitized


def _prepare_record(record: Mapping[str, Any], preprocessing: Mapping[str, Any], hosts: dict):
    if not preprocessing["deploy_view"]:
        return dict(record)
    from embedbench.models_chain import deployment_view
    from embedbench.structural import host_graph

    topology = record.get("topology")
    size = record.get("size")
    if not isinstance(topology, str) or isinstance(size, bool) or not isinstance(size, int):
        raise ValueError("deploy-view test record requires topology and integer size")
    key = (topology, size)
    if key not in hosts:
        hosts[key] = host_graph(topology, size)
    return deployment_view(
        dict(record),
        hosts[key],
        max_free=int(preprocessing["deploy_max_free"]),
    )


def _forward_record(model, record: Mapping[str, Any], preprocessing: Mapping[str, Any], device):
    """Run only label-free model inputs; release p_solve/stage values are never read."""

    arch = model.config.arch
    if arch == "hetero":
        from embedbench.models_hetero import encode_hetero_input

        return model(
            encode_hetero_input(
                dict(record),
                require_hamiltonian_context=True,
            )
        )

    from embedbench.models_chain import encode_chain
    from embedbench.models_quality_v2 import tensors_from_chain

    candidate_count = len(record["candidates"])
    model_record = dict(record)
    model_record["p_solve"] = [0.0] * candidate_count
    model_record["stage"] = [2] * candidate_count
    model_record["best_index"] = 0
    model_record["resource_index"] = 0
    model_record["original_index"] = -1
    encoded = encode_chain(
        model_record,
        use_neighbour_feats=bool(preprocessing["neighbour_feats"]),
        require_hamiltonian_context=True,
    )
    return model(*tensors_from_chain(encoded, device))


def _lcb_scores(output: Any, lcb_z: float) -> np.ndarray:
    mean = output.quality_mean.detach().cpu().numpy().astype(np.float64, copy=False)
    concentration = (
        output.quality_concentration.detach().cpu().numpy().astype(np.float64, copy=False)
    )
    variance = mean * (1.0 - mean) / (concentration + 1.0)
    return mean - lcb_z * np.sqrt(np.maximum(variance, 0.0))


def _record_key(record: Mapping[str, Any]) -> str:
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("quality record has no candidate list")
    identity = [
        str(record.get("_file")),
        str(record.get("instance_id")),
        record.get("focus"),
        quality_problem_digest(record),
        candidates,
    ]
    return _bytes_sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    )


def _record_keyed_random_choice(
    row: _LabelFreePaperRow,
    eligible: Sequence[int],
    *,
    namespace: str,
    random_seed: int,
) -> int:
    if not eligible:
        raise ValueError("record-keyed random baseline requires an eligible candidate")
    payload = f"{random_seed}:{namespace}:{row.record_key}".encode()
    offset = int.from_bytes(hashlib.sha256(payload).digest(), "big") % len(eligible)
    return int(eligible[offset])


def _selection_summary(
    selections: list[int | None],
    rows: Sequence[_PaperRow],
    eligible_sets: Sequence[Sequence[int]],
    *,
    top_tolerance: float,
) -> dict[str, object]:
    regrets: list[float] = []
    qualities: list[float] = []
    qubits: list[int] = []
    top_hits: list[bool] = []
    per_record: list[dict[str, object]] = []
    for selected, row, eligible in zip(selections, rows, eligible_sets, strict=True):
        if selected is None:
            per_record.append(
                {
                    "record_index": row.input_index,
                    "problem_digest": row.problem_digest,
                    "selection_index": None,
                    "p_solve": None,
                    "best_eligible_p_solve": None,
                    "regret": None,
                    "top": None,
                    "total_qubits": None,
                }
            )
            continue
        if selected not in eligible:
            raise RuntimeError("frozen label-free selection lies outside its eligible support")
        best = max(row.audit_scores[index] for index in eligible)
        quality = row.audit_scores[selected]
        regret = best - quality
        total_qubits = int(row.exact.total_qubits[selected])
        top = quality >= best - top_tolerance
        regrets.append(regret)
        qualities.append(quality)
        qubits.append(total_qubits)
        top_hits.append(top)
        per_record.append(
            {
                "record_index": row.input_index,
                "problem_digest": row.problem_digest,
                "selection_index": selected,
                "p_solve": quality,
                "best_eligible_p_solve": best,
                "regret": regret,
                "top": top,
                "total_qubits": total_qubits,
            }
        )
    selected_count = len(regrets)
    return {
        "regret": float(np.mean(regrets)) if regrets else None,
        "mean_p_solve": float(np.mean(qualities)) if qualities else None,
        "top": float(np.mean(top_hits)) if top_hits else None,
        "mean_total_qubits": float(np.mean(qubits)) if qubits else None,
        "n_selected": selected_count,
        "selection_indices": selections,
        "per_record": per_record,
    }


def _freeze_full_support(
    rows: Sequence[_LabelFreePaperRow],
    *,
    lcb_z: float,
    random_seed: int,
) -> dict[str, object]:
    eligible_sets: list[list[int]] = []
    selections = {name: [] for name in ("mean", "lcb", "resource", "original", "random")}
    for row in rows:
        eligible = [index for index, feasible in enumerate(row.exact.feasible) if feasible]
        if not eligible:
            raise ValueError("paper full-support evaluation has no feasible candidate")
        eligible_sets.append(eligible)
        mean = row.output.quality_mean.detach().cpu().numpy()
        lcb = _lcb_scores(row.output, lcb_z)
        selections["mean"].append(max(eligible, key=lambda index: (float(mean[index]), -index)))
        selections["lcb"].append(max(eligible, key=lambda index: (float(lcb[index]), -index)))
        selections["resource"].append(row.resource_index)
        selections["original"].append(
            row.original_index
            if row.record.get("source") == "minorminer" and row.original_index in eligible
            else None
        )
        selections["random"].append(
            _record_keyed_random_choice(
                row,
                eligible,
                namespace="full",
                random_seed=random_seed,
            )
        )
    return {
        "exact_mask": "minor_embedding_feasible",
        "n_records": len(rows),
        "total_presented_candidates": sum(len(row.exact) for row in rows),
        "total_eligible_candidates": sum(len(indices) for indices in eligible_sets),
        "eligible_indices": eligible_sets,
        "support_uses_labels": False,
        "selection_indices": selections,
    }


def _score_full_support(
    rows: Sequence[_PaperRow],
    frozen: Mapping[str, object],
    *,
    top_tolerance: float,
) -> dict[str, object]:
    eligible_sets = frozen["eligible_indices"]
    selections = frozen["selection_indices"]
    assert isinstance(eligible_sets, list)
    assert isinstance(selections, Mapping)
    return {key: value for key, value in frozen.items() if key != "selection_indices"} | {
        "selectors": {
            name: _selection_summary(
                list(indices),
                rows,
                eligible_sets,
                top_tolerance=top_tolerance,
            )
            for name, indices in selections.items()
        }
    }


def _budget_key(ratio: float | None) -> str:
    return "uncapped" if ratio is None else f"{ratio:.2f}x"


def _freeze_budget_sweep(
    all_rows: Sequence[_LabelFreePaperRow],
    *,
    lcb_z: float,
    random_seed: int,
) -> dict[str, object]:
    rows: list[_LabelFreePaperRow] = []
    included: list[int] = []
    excluded: list[int] = []
    reasons = {
        "non_minorminer_source": 0,
        "missing_original_reference": 0,
        "infeasible_original_reference": 0,
    }
    for row in all_rows:
        if row.record.get("source") != "minorminer":
            reason = "non_minorminer_source"
        elif row.original_index is None:
            reason = "missing_original_reference"
        elif not bool(row.exact.feasible[row.original_index]):
            reason = "infeasible_original_reference"
        else:
            reason = None
        if reason is None:
            rows.append(row)
            included.append(row.input_index)
        else:
            reasons[reason] += 1
            excluded.append(row.input_index)
    if not rows:
        raise ValueError("paper B/Q_MM evaluation has no valid stock minorminer reference")

    results: dict[str, object] = {}
    for ratio in REGISTERED_BUDGET_RATIOS:
        key = _budget_key(ratio)
        eligible_sets: list[list[int]] = []
        selections = {name: [] for name in ("mean", "lcb", "resource", "original", "random")}
        budgets: list[int | None] = []
        reference_qubits: list[int] = []
        no_survivor = 0
        for row in rows:
            assert row.original_index is not None
            q_mm = int(row.exact.total_qubits[row.original_index])
            budget = (
                int(row.exact.total_qubits.max())
                if ratio is None
                else int(math.floor(float(ratio) * q_mm + 1e-9))
            )
            eligible = [
                index
                for index in range(len(row.exact))
                if bool(row.exact.feasible[index]) and int(row.exact.total_qubits[index]) <= budget
            ]
            budgets.append(None if ratio is None else budget)
            reference_qubits.append(q_mm)
            eligible_sets.append(eligible)
            if not eligible:
                no_survivor += 1
                for name in selections:
                    selections[name].append(None)
                continue
            mean = row.output.quality_mean.detach().cpu().numpy()
            lcb = _lcb_scores(row.output, lcb_z)
            selections["mean"].append(max(eligible, key=lambda index: (float(mean[index]), -index)))
            selections["lcb"].append(max(eligible, key=lambda index: (float(lcb[index]), -index)))
            selections["resource"].append(
                row.resource_index if row.resource_index in eligible else None
            )
            selections["original"].append(row.original_index)
            selections["random"].append(
                _record_keyed_random_choice(
                    row,
                    eligible,
                    namespace=f"budget:{key}",
                    random_seed=random_seed,
                )
            )
        results[key] = {
            "ratio": ratio,
            "reference_policy": "stock_minorminer_original_total_qubits",
            "n_records": len(rows),
            "input_record_indices": [row.input_index for row in rows],
            "reference_total_qubits": reference_qubits,
            "budgets": budgets,
            "eligible_indices": eligible_sets,
            "mean_eligible_candidates": float(np.mean([len(x) for x in eligible_sets])),
            "minimum_eligible_candidates": min(len(x) for x in eligible_sets),
            "maximum_eligible_candidates": max(len(x) for x in eligible_sets),
            "no_survivor": no_survivor,
            "no_survivor_rate": no_survivor / len(rows),
            "selection_indices": selections,
        }
    return {
        "candidate_support": "full",
        "support_uses_labels": False,
        "exact_mask": "minor_embedding_feasible_and_full_embedding_total_qubits",
        "budget_contract": "B/Q_MM",
        "budget_reference": "stock_minorminer_original_total_qubits",
        "primary_record_policy": "valid_stock_minorminer_original_reference_only",
        "primary_record_coverage": {
            "input_records": len(all_rows),
            "included_records": len(rows),
            "excluded_records": len(all_rows) - len(rows),
            "coverage_rate": len(rows) / len(all_rows),
            "included_input_indices": included,
            "excluded_input_indices": excluded,
            "exclusion_reasons": reasons,
        },
        "budget_ratios": list(REGISTERED_BUDGET_RATIOS),
        "selectors": ["mean", "lcb"],
        "primary_selector": "mean",
        "primary_metric_name": "mean_finite_budget_regret",
        "lcb_z": lcb_z,
        "lcb_is_secondary_until_calibrated": True,
        "budgets": results,
    }


def _score_budget_sweep(
    all_rows: Sequence[_PaperRow],
    frozen: Mapping[str, object],
    *,
    top_tolerance: float,
) -> dict[str, object]:
    coverage = frozen["primary_record_coverage"]
    assert isinstance(coverage, Mapping)
    included = coverage["included_input_indices"]
    assert isinstance(included, list)
    by_input_index = {row.input_index: row for row in all_rows}
    rows = [by_input_index[index] for index in included]
    raw_budgets = frozen["budgets"]
    assert isinstance(raw_budgets, Mapping)
    budgets: dict[str, object] = {}
    for key, raw_result in raw_budgets.items():
        assert isinstance(raw_result, Mapping)
        eligible_sets = raw_result["eligible_indices"]
        selections = raw_result["selection_indices"]
        assert isinstance(eligible_sets, list)
        assert isinstance(selections, Mapping)
        budgets[str(key)] = {
            field: value for field, value in raw_result.items() if field != "selection_indices"
        } | {
            "selectors": {
                name: _selection_summary(
                    list(indices),
                    rows,
                    eligible_sets,
                    top_tolerance=top_tolerance,
                )
                for name, indices in selections.items()
            }
        }
    finite_regrets = [
        budgets[_budget_key(ratio)]["selectors"]["mean"]["regret"]
        for ratio in REGISTERED_BUDGET_RATIOS
        if ratio is not None
    ]
    if any(value is None for value in finite_regrets):
        raise RuntimeError("registered B/Q_MM selector failed to produce a paper metric")
    return {key: value for key, value in frozen.items() if key != "budgets"} | {
        "primary_metric": float(np.mean(finite_regrets)),
        "budgets": budgets,
    }


def _endpoint_status(available: int, required: int) -> str:
    if required == 0 or available == 0:
        return "unavailable"
    return "available" if available == required else "partial"


def _selection_survival(
    selections: Sequence[object],
    eligible_sets: Sequence[Sequence[int]],
) -> dict[str, object]:
    if len(selections) != len(eligible_sets):
        raise RuntimeError("secondary endpoint selection support is misaligned")
    selected_count = 0
    for selected, eligible in zip(selections, eligible_sets, strict=True):
        if selected is None:
            continue
        if isinstance(selected, bool) or not isinstance(selected, int) or selected not in eligible:
            raise RuntimeError("secondary endpoint found an invalid frozen selection")
        selected_count += 1
    total = len(eligible_sets)
    return {
        "records": total,
        "selected_records": selected_count,
        "selection_survival_rate": selected_count / total if total else 0.0,
        "selected_exact_feasibility_rate": 1.0 if selected_count else None,
    }


def _secondary_exact_feasibility_budget_survival(
    rows: Sequence[_PaperRow],
    frozen_full: Mapping[str, object],
    frozen_budget: Mapping[str, object],
    *,
    contract: Mapping[str, object],
) -> dict[str, object]:
    expected_source = "independent_complete_embedding_validation"
    if contract.get("source") != expected_source:
        raise RuntimeError("secondary feasibility endpoint implementation is not registered")
    full_eligible = frozen_full.get("eligible_indices")
    full_selections = frozen_full.get("selection_indices")
    if not isinstance(full_eligible, list) or not isinstance(full_selections, Mapping):
        raise RuntimeError("secondary feasibility endpoint has no frozen full support")
    presented = sum(len(row.exact) for row in rows)
    feasible = sum(int(np.count_nonzero(row.exact.feasible)) for row in rows)
    full_survivors = sum(bool(indices) for indices in full_eligible)
    selectors = tuple(contract.get("_selectors", ()))
    full_selector_results = {
        selector: _selection_survival(list(full_selections[selector]), full_eligible)
        for selector in selectors
    }

    raw_budgets = frozen_budget.get("budgets")
    if not isinstance(raw_budgets, Mapping):
        raise RuntimeError("secondary feasibility endpoint has no frozen budget support")
    budget_results: dict[str, object] = {}
    for key, raw_budget in raw_budgets.items():
        if not isinstance(raw_budget, Mapping):
            raise RuntimeError("secondary feasibility endpoint has an invalid budget")
        input_indices = raw_budget.get("input_record_indices")
        eligible_sets = raw_budget.get("eligible_indices")
        selections = raw_budget.get("selection_indices")
        if (
            not isinstance(input_indices, list)
            or not isinstance(eligible_sets, list)
            or not isinstance(selections, Mapping)
            or len(input_indices) != len(eligible_sets)
        ):
            raise RuntimeError("secondary feasibility endpoint budget support is misaligned")
        by_index = {row.input_index: row for row in rows}
        presented_here = sum(len(by_index[index].exact) for index in input_indices)
        eligible_count = sum(len(indices) for indices in eligible_sets)
        survivors = sum(bool(indices) for indices in eligible_sets)
        total = len(eligible_sets)
        budget_results[str(key)] = {
            "records": total,
            "records_with_survivor": survivors,
            "record_survival_rate": survivors / total if total else 0.0,
            "presented_candidates": presented_here,
            "eligible_candidates": eligible_count,
            "candidate_budget_survival_rate": (
                eligible_count / presented_here if presented_here else 0.0
            ),
            "selectors": {
                selector: _selection_survival(list(selections[selector]), eligible_sets)
                for selector in selectors
            },
        }
    return {
        "status": "available",
        "source": expected_source,
        "coverage": {
            "records": len(rows),
            "authenticated_records": len(rows),
            "coverage_rate": 1.0,
        },
        "full_support": {
            "records": len(rows),
            "records_with_survivor": full_survivors,
            "record_survival_rate": full_survivors / len(rows) if rows else 0.0,
            "presented_candidates": presented,
            "feasible_candidates": feasible,
            "candidate_feasibility_rate": feasible / presented if presented else 0.0,
            "selectors": full_selector_results,
        },
        "budgets": budget_results,
    }


def _terminal_resource_summary(
    rows: Sequence[_PaperRow],
    selections: Sequence[object],
) -> dict[str, object]:
    if len(rows) != len(selections):
        raise RuntimeError("terminal-resource endpoint support is misaligned")
    total_deltas: list[int] = []
    max_chain_deltas: list[int] = []
    selected_records = 0
    for row, selected in zip(rows, selections, strict=True):
        if selected is None:
            continue
        if isinstance(selected, bool) or not isinstance(selected, int):
            raise RuntimeError("terminal-resource endpoint has an invalid selection")
        if row.original_index is None:
            raise RuntimeError("terminal-resource endpoint lost its stock reference")
        selected_records += 1
        reference = row.original_index
        total_deltas.append(
            int(row.exact.total_qubits[selected]) - int(row.exact.total_qubits[reference])
        )
        max_chain_deltas.append(
            int(row.exact.max_chain[selected]) - int(row.exact.max_chain[reference])
        )
    required = len(rows)
    return {
        "status": _endpoint_status(selected_records, required),
        "reference_records": required,
        "paired_records": selected_records,
        "coverage_rate": selected_records / required if required else 0.0,
        "mean_total_qubits_delta_vs_stock_original": (
            float(np.mean(total_deltas)) if total_deltas else None
        ),
        "mean_max_chain_delta_vs_stock_original": (
            float(np.mean(max_chain_deltas)) if max_chain_deltas else None
        ),
    }


def _secondary_terminal_resources(
    rows: Sequence[_PaperRow],
    frozen_budget: Mapping[str, object],
    *,
    contract: Mapping[str, object],
) -> dict[str, object]:
    expected_source = "independent_complete_embedding_validation"
    if (
        contract.get("source") != expected_source
        or contract.get("reference") != "stock_minorminer_original"
    ):
        raise RuntimeError("secondary terminal-resource endpoint implementation is not registered")
    by_index = {row.input_index: row for row in rows}
    raw_budgets = frozen_budget.get("budgets")
    if not isinstance(raw_budgets, Mapping):
        raise RuntimeError("terminal-resource endpoint has no frozen budget support")
    selectors = tuple(contract.get("_selectors", ()))
    budgets: dict[str, object] = {}
    reference_record_count: int | None = None
    for key, raw_budget in raw_budgets.items():
        if not isinstance(raw_budget, Mapping):
            raise RuntimeError("terminal-resource endpoint has an invalid budget")
        input_indices = raw_budget.get("input_record_indices")
        selection_indices = raw_budget.get("selection_indices")
        if not isinstance(input_indices, list) or not isinstance(selection_indices, Mapping):
            raise RuntimeError("terminal-resource endpoint budget support is misaligned")
        budget_rows = [by_index[index] for index in input_indices]
        if reference_record_count is None:
            reference_record_count = len(budget_rows)
        elif reference_record_count != len(budget_rows):
            raise RuntimeError("terminal-resource endpoint changes its reference cohort")
        budgets[str(key)] = {
            "selectors": {
                selector: _terminal_resource_summary(
                    budget_rows,
                    list(selection_indices[selector]),
                )
                for selector in selectors
            }
        }
    return {
        "status": "available",
        "source": expected_source,
        "reference": "stock_minorminer_original",
        "coverage": {
            "records": reference_record_count or 0,
            "authenticated_records": reference_record_count or 0,
            "coverage_rate": 1.0 if reference_record_count else 0.0,
        },
        "budgets": budgets,
    }


def _residual_host_metrics(
    row: _PaperRow,
    candidate_index: int,
    *,
    exact_host_sources: Mapping[str, Mapping[str, object]],
    host_cache: dict[tuple[str, int], object],
) -> dict[str, float | int] | None:
    source = exact_host_sources.get(str(row.record.get("_file")))
    if source is None:
        return None
    raw_all_chains = row.record.get("all_chains")
    raw_candidates = row.record.get("candidates")
    if not isinstance(raw_all_chains, Mapping) or not isinstance(raw_candidates, list):
        return None
    topology = source.get("topology")
    size = source.get("size")
    if not isinstance(topology, str) or isinstance(size, bool) or not isinstance(size, int):
        raise RuntimeError("authenticated host source became malformed")
    key = (topology, size)
    if key not in host_cache:
        from embedbench.structural import host_graph

        host_cache[key] = host_graph(*key)
    host = host_cache[key]
    try:
        candidate = raw_candidates[candidate_index]
    except IndexError as error:
        raise RuntimeError(
            "residual-connectivity selection lies outside candidate support"
        ) from error
    if not isinstance(candidate, list):
        raise RuntimeError("authenticated candidate chain is malformed")
    used: set[int] = set()
    for chain in (*raw_all_chains.values(), candidate):
        if not isinstance(chain, list) or any(
            isinstance(qubit, bool) or not isinstance(qubit, int) for qubit in chain
        ):
            raise RuntimeError("authenticated complete embedding is malformed")
        used.update(chain)
    host_nodes = set(host.nodes)
    if not used <= host_nodes:
        raise RuntimeError("authenticated complete embedding leaves the realized host")
    remaining = host_nodes - used
    if remaining:
        import networkx as nx

        components = list(nx.connected_components(host.subgraph(remaining)))
        largest = max(map(len, components))
    else:
        components = []
        largest = 0
    return {
        "remaining_free_qubits": len(remaining),
        "largest_remaining_component": largest,
        "largest_component_fraction": largest / len(remaining) if remaining else 0.0,
        "remaining_component_count": len(components),
    }


def _residual_connectivity_summary(
    rows: Sequence[_PaperRow],
    selections: Sequence[object],
    *,
    exact_host_sources: Mapping[str, Mapping[str, object]],
    host_cache: dict[tuple[str, int], object],
) -> dict[str, object]:
    if len(rows) != len(selections):
        raise RuntimeError("residual-connectivity endpoint support is misaligned")
    delta_fields = {
        "remaining_free_qubits": "mean_remaining_free_qubits_delta_vs_stock_original",
        "largest_remaining_component": ("mean_largest_remaining_component_delta_vs_stock_original"),
        "largest_component_fraction": ("mean_largest_component_fraction_delta_vs_stock_original"),
        "remaining_component_count": ("mean_remaining_component_count_delta_vs_stock_original"),
    }
    absolute_fields = {
        "remaining_free_qubits": "mean_remaining_free_qubits",
        "largest_remaining_component": "mean_largest_remaining_component",
        "largest_component_fraction": "mean_largest_component_fraction",
        "remaining_component_count": "mean_remaining_component_count",
    }
    per_record_absolute_fields = {
        field: output.removeprefix("mean_") for field, output in absolute_fields.items()
    }
    per_record_delta_fields = {
        field: output.removeprefix("mean_") for field, output in delta_fields.items()
    }
    deltas: dict[str, list[float]] = {field: [] for field in delta_fields}
    observed_values: dict[str, list[float]] = {field: [] for field in absolute_fields}
    per_record: list[dict[str, object]] = []
    selected_records = 0
    authenticated_records = 0
    for row, selected in zip(rows, selections, strict=True):
        if selected is None:
            per_record.append(
                {
                    "record_index": row.input_index,
                    "problem_digest": row.problem_digest,
                    "selection_index": None,
                    "reference_index": row.original_index,
                    "authenticated": False,
                    "paired": False,
                    **{field: None for field in per_record_absolute_fields.values()},
                    **{field: None for field in per_record_delta_fields.values()},
                }
            )
            continue
        if isinstance(selected, bool) or not isinstance(selected, int):
            raise RuntimeError("residual-connectivity endpoint has an invalid selection")
        if row.original_index is None:
            raise RuntimeError("residual-connectivity endpoint lost its stock reference")
        selected_records += 1
        observed = _residual_host_metrics(
            row,
            selected,
            exact_host_sources=exact_host_sources,
            host_cache=host_cache,
        )
        reference = _residual_host_metrics(
            row,
            row.original_index,
            exact_host_sources=exact_host_sources,
            host_cache=host_cache,
        )
        if observed is None or reference is None:
            per_record.append(
                {
                    "record_index": row.input_index,
                    "problem_digest": row.problem_digest,
                    "selection_index": selected,
                    "reference_index": row.original_index,
                    "authenticated": False,
                    "paired": False,
                    **{field: None for field in per_record_absolute_fields.values()},
                    **{field: None for field in per_record_delta_fields.values()},
                }
            )
            continue
        authenticated_records += 1
        record_metrics: dict[str, object] = {
            "record_index": row.input_index,
            "problem_digest": row.problem_digest,
            "selection_index": selected,
            "reference_index": row.original_index,
            "authenticated": True,
            "paired": True,
        }
        for field in delta_fields:
            observed_values[field].append(float(observed[field]))
            deltas[field].append(float(observed[field]) - float(reference[field]))
            record_metrics[per_record_absolute_fields[field]] = float(observed[field])
            record_metrics[per_record_delta_fields[field]] = float(observed[field]) - float(
                reference[field]
            )
        per_record.append(record_metrics)
    result: dict[str, object] = {
        "status": _endpoint_status(authenticated_records, selected_records),
        "support_records": len(rows),
        "selected_records": selected_records,
        "authenticated_records": authenticated_records,
        "coverage_rate": authenticated_records / selected_records if selected_records else 0.0,
        "paired_records": authenticated_records,
        "pairing_excluded_records": len(rows) - authenticated_records,
        "paired_coverage_rate": authenticated_records / len(rows) if rows else 0.0,
        "unavailable_reason": (
            "no_frozen_selection_on_registered_support"
            if selected_records == 0
            else (
                None
                if authenticated_records == selected_records
                else "missing_authenticated_realized_host_or_complete_embedding"
            )
        ),
        "per_record": per_record,
    }
    result.update(
        {
            output_field: float(np.mean(deltas[input_field])) if deltas[input_field] else None
            for input_field, output_field in delta_fields.items()
        }
    )
    result.update(
        {
            output_field: (
                float(np.mean(observed_values[input_field]))
                if observed_values[input_field]
                else None
            )
            for input_field, output_field in absolute_fields.items()
        }
    )
    return result


def _secondary_residual_connectivity(
    rows: Sequence[_PaperRow],
    frozen_budget: Mapping[str, object],
    *,
    exact_host_sources: Mapping[str, Mapping[str, object]],
    contract: Mapping[str, object],
) -> dict[str, object]:
    expected_source = "authenticated_realized_host_minus_complete_embedding"
    if (
        contract.get("source") != expected_source
        or contract.get("reference") != "stock_minorminer_original"
        or contract.get("missing_input_action") != "unavailable_with_explicit_coverage"
    ):
        raise RuntimeError(
            "secondary residual-connectivity endpoint implementation is not registered"
        )
    by_index = {row.input_index: row for row in rows}
    raw_budgets = frozen_budget.get("budgets")
    if not isinstance(raw_budgets, Mapping):
        raise RuntimeError("residual-connectivity endpoint has no frozen budget support")
    selectors = tuple(contract.get("_selectors", ()))
    host_cache: dict[tuple[str, int], object] = {}
    budgets: dict[str, object] = {}
    statuses: list[str] = []
    for key, raw_budget in raw_budgets.items():
        if not isinstance(raw_budget, Mapping):
            raise RuntimeError("residual-connectivity endpoint has an invalid budget")
        input_indices = raw_budget.get("input_record_indices")
        selection_indices = raw_budget.get("selection_indices")
        if not isinstance(input_indices, list) or not isinstance(selection_indices, Mapping):
            raise RuntimeError("residual-connectivity endpoint budget support is misaligned")
        budget_rows = [by_index[index] for index in input_indices]
        summaries = {
            selector: _residual_connectivity_summary(
                budget_rows,
                list(selection_indices[selector]),
                exact_host_sources=exact_host_sources,
                host_cache=host_cache,
            )
            for selector in selectors
        }
        statuses.extend(str(summary["status"]) for summary in summaries.values())
        budgets[str(key)] = {"selectors": summaries}
    status = "available"
    if statuses and all(value == "unavailable" for value in statuses):
        status = "unavailable"
    elif any(value != "available" for value in statuses):
        status = "partial"
    authenticated_records = sum(
        _residual_host_metrics(
            row,
            0,
            exact_host_sources=exact_host_sources,
            host_cache=host_cache,
        )
        is not None
        for row in rows
    )
    return {
        "status": status,
        "source": expected_source,
        "reference": "stock_minorminer_original",
        "coverage": {
            "records": len(rows),
            "authenticated_records": authenticated_records,
            "coverage_rate": authenticated_records / len(rows) if rows else 0.0,
        },
        "budgets": budgets,
    }


def _strength_robustness_summary(
    rows: Sequence[_PaperRow],
    selections: Sequence[object],
) -> dict[str, object]:
    if len(rows) != len(selections):
        raise RuntimeError("strength-robustness endpoint support is misaligned")
    worst_case: list[float] = []
    spreads: list[float] = []
    worst_case_deltas: list[float] = []
    spread_deltas: list[float] = []
    per_record: list[dict[str, object]] = []
    selected_records = 0
    authenticated_records = 0
    paired_records = 0
    for row, selected in zip(rows, selections, strict=True):
        stock_reference = row.original_index if row.record.get("source") == "minorminer" else None
        if selected is None:
            per_record.append(
                {
                    "record_index": row.input_index,
                    "problem_digest": row.problem_digest,
                    "selection_index": None,
                    "reference_index": stock_reference,
                    "authenticated": False,
                    "paired": False,
                    "worst_case_p_solve": None,
                    "p_solve_spread": None,
                    "stock_original_worst_case_p_solve": None,
                    "stock_original_p_solve_spread": None,
                    "worst_case_p_solve_delta_vs_stock_original": None,
                    "p_solve_spread_delta_vs_stock_original": None,
                }
            )
            continue
        if isinstance(selected, bool) or not isinstance(selected, int):
            raise RuntimeError("strength-robustness endpoint has an invalid selection")
        selected_records += 1
        matrix = row.audit_strength_p_solve
        if matrix is None:
            per_record.append(
                {
                    "record_index": row.input_index,
                    "problem_digest": row.problem_digest,
                    "selection_index": selected,
                    "reference_index": stock_reference,
                    "authenticated": False,
                    "paired": False,
                    "worst_case_p_solve": None,
                    "p_solve_spread": None,
                    "stock_original_worst_case_p_solve": None,
                    "stock_original_p_solve_spread": None,
                    "worst_case_p_solve_delta_vs_stock_original": None,
                    "p_solve_spread_delta_vs_stock_original": None,
                }
            )
            continue
        try:
            values = matrix[selected]
        except IndexError as error:
            raise RuntimeError(
                "strength-robustness selection lies outside authenticated outcomes"
            ) from error
        authenticated_records += 1
        observed_worst = min(values)
        observed_spread = max(values) - min(values)
        worst_case.append(observed_worst)
        spreads.append(observed_spread)
        reference_worst: float | None = None
        reference_spread: float | None = None
        worst_delta: float | None = None
        spread_delta: float | None = None
        if stock_reference is not None:
            try:
                reference_values = matrix[stock_reference]
            except IndexError as error:
                raise RuntimeError(
                    "strength-robustness stock reference lies outside authenticated outcomes"
                ) from error
            reference_worst = min(reference_values)
            reference_spread = max(reference_values) - min(reference_values)
            worst_delta = observed_worst - reference_worst
            spread_delta = observed_spread - reference_spread
            worst_case_deltas.append(worst_delta)
            spread_deltas.append(spread_delta)
            paired_records += 1
        per_record.append(
            {
                "record_index": row.input_index,
                "problem_digest": row.problem_digest,
                "selection_index": selected,
                "reference_index": stock_reference,
                "authenticated": True,
                "paired": stock_reference is not None,
                "worst_case_p_solve": observed_worst,
                "p_solve_spread": observed_spread,
                "stock_original_worst_case_p_solve": reference_worst,
                "stock_original_p_solve_spread": reference_spread,
                "worst_case_p_solve_delta_vs_stock_original": worst_delta,
                "p_solve_spread_delta_vs_stock_original": spread_delta,
            }
        )
    return {
        "status": _endpoint_status(authenticated_records, selected_records),
        "support_records": len(rows),
        "selected_records": selected_records,
        "selection_excluded_records": len(rows) - selected_records,
        "authenticated_records": authenticated_records,
        "missing_outcome_records": selected_records - authenticated_records,
        "coverage_rate": authenticated_records / selected_records if selected_records else 0.0,
        "support_coverage_rate": authenticated_records / len(rows) if rows else 0.0,
        "paired_records": paired_records,
        "pairing_excluded_records": len(rows) - paired_records,
        "paired_coverage_rate": paired_records / len(rows) if rows else 0.0,
        "unavailable_reason": (
            "no_frozen_selection_on_registered_support"
            if selected_records == 0
            else (
                None
                if authenticated_records == selected_records
                else "missing_authenticated_per_strength_p_solve"
            )
        ),
        "mean_worst_case_p_solve": float(np.mean(worst_case)) if worst_case else None,
        "mean_p_solve_spread": float(np.mean(spreads)) if spreads else None,
        "mean_worst_case_p_solve_delta_vs_stock_original": (
            float(np.mean(worst_case_deltas)) if worst_case_deltas else None
        ),
        "mean_p_solve_spread_delta_vs_stock_original": (
            float(np.mean(spread_deltas)) if spread_deltas else None
        ),
        "per_record": per_record,
    }


def _secondary_strength_robustness(
    rows: Sequence[_PaperRow],
    frozen_full: Mapping[str, object],
    frozen_budget: Mapping[str, object],
    *,
    contract: Mapping[str, object],
) -> dict[str, object]:
    expected_source = "authenticated_four_registered_strength_p_solve_outcomes"
    if (
        contract.get("source") != expected_source
        or contract.get("registered_strength_count") != 4
        or contract.get("support") != "complete_fixed_test_and_registered_bqmm_budgets"
        or contract.get("missing_input_action") != "unavailable_with_explicit_coverage"
    ):
        raise RuntimeError(
            "secondary strength-robustness endpoint implementation is not registered"
        )
    by_index = {row.input_index: row for row in rows}
    full_selections = frozen_full.get("selection_indices")
    full_eligible = frozen_full.get("eligible_indices")
    if not isinstance(full_selections, Mapping) or not isinstance(full_eligible, list):
        raise RuntimeError("strength-robustness endpoint has no frozen full support")
    if len(full_eligible) != len(rows) or any(not indices for indices in full_eligible):
        raise RuntimeError("strength-robustness full support is incomplete")
    raw_budgets = frozen_budget.get("budgets")
    if not isinstance(raw_budgets, Mapping):
        raise RuntimeError("strength-robustness endpoint has no frozen budget support")
    selectors = tuple(contract.get("_selectors", ()))
    full_summaries = {
        selector: _strength_robustness_summary(
            rows,
            list(full_selections[selector]),
        )
        for selector in selectors
    }
    budgets: dict[str, object] = {}
    statuses: list[str] = [str(summary["status"]) for summary in full_summaries.values()]
    for key, raw_budget in raw_budgets.items():
        if not isinstance(raw_budget, Mapping):
            raise RuntimeError("strength-robustness endpoint has an invalid budget")
        input_indices = raw_budget.get("input_record_indices")
        selection_indices = raw_budget.get("selection_indices")
        if not isinstance(input_indices, list) or not isinstance(selection_indices, Mapping):
            raise RuntimeError("strength-robustness endpoint budget support is misaligned")
        budget_rows = [by_index[index] for index in input_indices]
        summaries = {
            selector: _strength_robustness_summary(
                budget_rows,
                list(selection_indices[selector]),
            )
            for selector in selectors
        }
        statuses.extend(str(summary["status"]) for summary in summaries.values())
        budgets[str(key)] = {"selectors": summaries}
    status = "available"
    if statuses and all(value == "unavailable" for value in statuses):
        status = "unavailable"
    elif any(value != "available" for value in statuses):
        status = "partial"
    authenticated_records = sum(row.audit_strength_p_solve is not None for row in rows)
    primary_coverage = frozen_budget.get("primary_record_coverage")
    if not isinstance(primary_coverage, Mapping):
        raise RuntimeError("strength-robustness endpoint has no budget-cohort coverage")
    for field in ("input_records", "included_records", "excluded_records"):
        value = primary_coverage.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError("strength-robustness budget-cohort coverage is malformed")
    if primary_coverage["input_records"] != len(rows) or primary_coverage[
        "included_records"
    ] + primary_coverage["excluded_records"] != len(rows):
        raise RuntimeError("strength-robustness budget-cohort coverage is inconsistent")
    return {
        "status": status,
        "source": expected_source,
        "registered_strength_count": 4,
        "coverage": {
            "records": len(rows),
            "authenticated_records": authenticated_records,
            "coverage_rate": authenticated_records / len(rows) if rows else 0.0,
        },
        "support": {
            "full_support_records": len(rows),
            "full_support_excluded_records": 0,
            "budget_conditioned_records": primary_coverage["included_records"],
            "budget_excluded_records": primary_coverage["excluded_records"],
            "budget_coverage_rate": primary_coverage["coverage_rate"],
            "budget_exclusion_reasons": primary_coverage["exclusion_reasons"],
        },
        "full_support": {
            "record_support": "complete_fixed_test_partition",
            "records": len(rows),
            "selectors": full_summaries,
        },
        "budgets": budgets,
    }


def _secondary_endpoints(
    rows: Sequence[_PaperRow],
    frozen_full: Mapping[str, object],
    frozen_budget: Mapping[str, object],
    *,
    exact_host_sources: Mapping[str, Mapping[str, object]],
    audit_contract: PaperAuditContract,
) -> dict[str, object]:
    registration = audit_contract.secondary_endpoints
    if (
        registration.get("selection_role") != "report_only_after_label_free_policy_freeze"
        or registration.get("availability_policy")
        != "fail_closed_on_malformed_authenticated_input_else_status_and_coverage"
        or registration.get("selectors") != ["mean", "lcb", "resource", "original", "random"]
    ):
        raise RuntimeError("secondary endpoint implementation disagrees with registration")
    selectors = list(registration["selectors"])

    def endpoint_contract(name: str) -> dict[str, object]:
        value = registration.get(name)
        if not isinstance(value, Mapping):
            raise RuntimeError(f"secondary endpoint {name} is not registered")
        return dict(value) | {"_selectors": selectors}

    return {
        "selection_role": registration["selection_role"],
        "availability_policy": registration["availability_policy"],
        "registration": dict(registration),
        "exact_feasibility_budget_survival": (
            _secondary_exact_feasibility_budget_survival(
                rows,
                frozen_full,
                frozen_budget,
                contract=endpoint_contract("exact_feasibility_budget_survival"),
            )
        ),
        "terminal_resources": _secondary_terminal_resources(
            rows,
            frozen_budget,
            contract=endpoint_contract("terminal_resources"),
        ),
        "residual_connectivity": _secondary_residual_connectivity(
            rows,
            frozen_budget,
            exact_host_sources=exact_host_sources,
            contract=endpoint_contract("residual_connectivity"),
        ),
        "strength_robustness": _secondary_strength_robustness(
            rows,
            frozen_full,
            frozen_budget,
            contract=endpoint_contract("strength_robustness"),
        ),
    }


def _label_free_plan(
    rows: Sequence[_LabelFreePaperRow],
    full_support: Mapping[str, object],
    budget_sweep: Mapping[str, object],
) -> dict[str, object]:
    from embedbench.models_chain import quality_candidate_signature

    budgets = budget_sweep["budgets"]
    assert isinstance(budgets, Mapping)
    return {
        "schema": "embedbench.quality-v2-label-free-policy-freeze",
        "schema_version": 2,
        "records": [
            {
                "input_index": row.input_index,
                "file": row.record["_file"],
                "line_number": row.record["_line_number"],
                "instance_id": row.record["instance_id"],
                "focus": row.record["focus"],
                "problem_digest": row.problem_digest,
                "record_key": row.record_key,
                "candidate_signature": quality_candidate_signature(row.record),
                "candidate_count": len(row.exact),
                "exact_feasible": [bool(value) for value in row.exact.feasible.tolist()],
                "total_qubits": [int(value) for value in row.exact.total_qubits.tolist()],
                "resource_index": row.resource_index,
                "original_index": row.original_index,
            }
            for row in rows
        ],
        "full_support": {
            "eligible_indices": full_support["eligible_indices"],
            "selection_indices": full_support["selection_indices"],
        },
        "budgets": {
            key: {
                "input_record_indices": value["input_record_indices"],
                "eligible_indices": value["eligible_indices"],
                "selection_indices": value["selection_indices"],
            }
            for key, value in budgets.items()
        },
    }


def evaluate(args: argparse.Namespace) -> dict[str, object]:
    import torch
    from embedbench.models_chain import (
        AUDIT_AGGREGATION,
        AUDIT_GENERATOR,
        AUDIT_SCHEMA,
        AUDIT_SCHEMA_VERSION,
        AUDIT_SEED_SCHEDULE,
        AUDIT_STRENGTH_SCHEDULE,
        MIN_HIGH_READS_PER_STRENGTH,
        REGISTERED_STRENGTH_COUNT,
        load_independent_audit_scores,
        quality_audit_key,
        verify_independent_audit_scores,
    )
    from embedbench.models_quality_v2 import (
        derive_exact_candidate_metrics,
        load_quality_v2_model,
        select_quality_candidate,
    )

    audit_contract = load_paper_audit_contract(
        args.audit_contract,
        args.audit_contract_sha256,
    )
    contract_audit = audit_contract.audit_labels
    implementation_audit = {
        "schema": AUDIT_SCHEMA,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "generator": AUDIT_GENERATOR,
        "reads_per_strength": contract_audit["reads_per_strength"],
        "num_sweeps": contract_audit["num_sweeps"],
        "base_seed": contract_audit["base_seed"],
        "seed_schedule": AUDIT_SEED_SCHEDULE,
        "registered_strength_count": REGISTERED_STRENGTH_COUNT,
        "strength_schedule": AUDIT_STRENGTH_SCHEDULE,
        "aggregation": AUDIT_AGGREGATION,
        "score_evidence": "integer_success_counts_per_candidate_strength",
        "probability_derivation": "success_count_divided_by_reads_per_strength_binary64",
    }
    if implementation_audit != contract_audit or int(contract_audit["reads_per_strength"]) < int(
        MIN_HIGH_READS_PER_STRENGTH
    ):
        raise RuntimeError(
            "paper evaluator implementation disagrees with the frozen audit-label contract"
        )
    if (
        list(REGISTERED_BUDGET_RATIOS) != audit_contract.evaluation["registered_budget_ratios"]
        or audit_contract.evaluation["primary_metric"] != PRIMARY_METRIC
    ):
        raise RuntimeError(
            "paper evaluator implementation disagrees with the frozen evaluation contract"
        )
    selection_payload = _capture_bytes(args.selection, "selection artifact")
    from select_training_grid import revalidate_selection_artifact

    revalidated_selection = revalidate_selection_artifact(
        args.selection,
        args.selection_sha256,
        root=args.selection_root,
        audit_contract=audit_contract,
        selection_payload=selection_payload,
    )
    selection = load_selection_artifact(
        args.selection,
        args.selection_sha256,
        args.selection_root,
        audit_contract,
        selection_payload=selection_payload,
    )
    require_exact_json(
        selection.document,
        revalidated_selection,
        location="captured selection after complete grid revalidation",
    )
    required_seed_count = int(audit_contract.model_seed_aggregation["registered_seed_count"])
    if len(selection.registered_seeds) != required_seed_count:
        raise ValueError("selection registered seeds disagree with the frozen paper audit contract")
    with _captured_checkpoint(args.checkpoint, args.checkpoint_sha256) as (
        captured_checkpoint,
        checkpoint_payload,
    ):
        registered_checkpoint = bind_registered_checkpoint(
            selection,
            args.checkpoint,
            args.checkpoint_sha256,
            checkpoint_payload=checkpoint_payload,
        )
        model, metadata = load_quality_v2_model(captured_checkpoint)
    checkpoint_path = registered_checkpoint.resolved_path
    actual_checkpoint_sha = registered_checkpoint.checkpoint_sha256
    arch = model.config.arch
    _validate_checkpoint_selection_binding(
        metadata,
        arch,
        selection,
        registered_checkpoint,
    )
    preprocessing = _validate_checkpoint_protocol(
        metadata,
        arch,
        model_neighbour_feats=bool(model.config.neighbour_feats),
    )
    preregistration: PaperPreregistration | None = None
    registered_policy_freeze_sha256: str | None = None
    if args.phase == "score":
        preregistration = load_paper_preregistration(
            args.preregistration_manifest,
            args.preregistration_manifest_sha256,
            audit_contract=audit_contract,
            selection_document=selection.document,
            selection_file=selection.path.name,
            selection_sha256=selection.sha256,
            audit_source_root=Path(__file__).parents[1],
        )
        freeze_binding = preregistered_policy_freeze_binding(
            preregistration,
            registered_checkpoint.seed,
        )
        if Path(args.policy_freeze).name != freeze_binding["file"]:
            raise ValueError(
                "policy freeze does not match its authenticated paper preregistration entry"
            )
        registered_policy_freeze_sha256 = str(freeze_binding["sha256"])
    raw_lcb_z = metadata["lcb_z"]
    if isinstance(raw_lcb_z, bool) or not isinstance(raw_lcb_z, (int, float)):
        raise ValueError("checkpoint lcb_z must be numeric")
    lcb_z = float(raw_lcb_z)
    if (
        not math.isfinite(lcb_z)
        or lcb_z < 0.0
        or lcb_z != float(audit_contract.evaluation["lcb_z"])
    ):
        raise ValueError("checkpoint lcb_z disagrees with the frozen paper contract")
    top_tolerance = float(audit_contract.evaluation["top_tolerance"])
    random_seed = int(audit_contract.evaluation["random_seed"])
    if (
        audit_contract.evaluation["random_baseline_policy"]
        != "sha256_record_key_modulo_eligible_v1"
    ):
        raise RuntimeError("paper evaluator random-baseline implementation is not registered")

    with _captured_release_inputs(args.files, args.splits) as (captured_files, captured_splits):
        _, current_provenance = _fixed_split_provenance(captured_files, captured_splits)
        _validate_selection_data_contract(selection, current_provenance)
        split_sha = _sha256(captured_splits)
        _validate_checkpoint_data_contract(
            metadata,
            current_provenance,
            split_sha256=split_sha,
        )
        exact_host_contract, exact_host_sources = _verified_exact_host_contract(captured_files)
        _validate_deployment_host_contract(metadata, preprocessing, captured_files)
        test_records, loaded_provenance, partition_counts = _load_locked_test_records(
            captured_files,
            captured_splits,
        )
        if _validated_data_provenance(
            loaded_provenance, "materialized evaluation"
        ) != _validated_data_provenance(current_provenance, "evaluation"):
            raise RuntimeError("test materialization changed the validated data provenance")
        corpus_sha = _provenance_inputs(current_provenance["corpus_inputs"], "evaluation")

        device = torch.device(str(audit_contract.evaluation["device"]))
        model = model.to(device)
        model.eval()
        evaluation_runtime = validate_runtime_provenance(
            runtime_provenance(device),
            location="paper evaluation runtime",
            expected_device=str(audit_contract.evaluation["device"]),
        )
        label_free_rows: list[_LabelFreePaperRow] = []
        hosts: dict[tuple[str, int], object] = {}
        with torch.no_grad():
            for input_index, record in enumerate(test_records):
                _validate_record_host_identity(record, exact_host_sources)
                model_record = _sanitized_model_record(record)
                prepared = _prepare_record(model_record, preprocessing, hosts)
                exact = derive_exact_candidate_metrics(prepared)
                resource_index, original_index = _validate_policy_indices(record, exact)
                output = _forward_record(model, prepared, preprocessing, device)
                select_quality_candidate(
                    output,
                    exact,
                    budget=int(exact.total_qubits.max()),
                    statistic=str(audit_contract.evaluation["selection_statistic"]),
                )
                label_free_rows.append(
                    _LabelFreePaperRow(
                        input_index=input_index,
                        record=record,
                        output=output,
                        exact=exact,
                        resource_index=resource_index,
                        original_index=original_index,
                        record_key=_record_key(record),
                        problem_digest=quality_problem_digest(record),
                    )
                )

        frozen_full = _freeze_full_support(
            label_free_rows,
            lcb_z=lcb_z,
            random_seed=random_seed,
        )
        frozen_budget = _freeze_budget_sweep(
            label_free_rows,
            lcb_z=lcb_z,
            random_seed=random_seed,
        )
        frozen_plan = _label_free_plan(label_free_rows, frozen_full, frozen_budget)
        freeze_checkpoint = {
            "file": checkpoint_path.name,
            "sha256": actual_checkpoint_sha,
            "artifact_schema": metadata["artifact_schema"],
            "artifact_schema_version": metadata["artifact_schema_version"],
            "architecture": arch,
            "objective_variant": metadata.get("objective_variant"),
            "seed": metadata.get("seed"),
            "training_run_id": metadata["training_run_id"],
            "best_epoch": metadata["best_epoch"],
            "selected_validation_metric": metadata["selected_validation_metric"],
            "split_sha256": split_sha,
            "training_scope": metadata["training_scope"],
        }
        frozen_plan.update(
            {
                "phase": "label_free_policy_freeze",
                "audit_contract": audit_contract.public_binding(),
                "selection": selection_public_binding(selection, registered_checkpoint),
                "checkpoint": freeze_checkpoint,
                "data_provenance": current_provenance,
                "runtime_provenance": evaluation_runtime,
                "preprocessing": preprocessing,
                "deployment_host_contract": metadata["deployment_host_contract"],
                "exact_host_contract": exact_host_contract,
                "partition_record_counts": partition_counts,
                "n_test": len(label_free_rows),
                "audit_paths_opened": False,
                "release_quality_labels_consumed": False,
            }
        )
        frozen_payload = _canonical_json_bytes(frozen_plan)
        frozen_sha256 = _bytes_sha256(frozen_payload)

        if args.phase == "freeze":
            return frozen_plan

        assert preregistration is not None
        assert registered_policy_freeze_sha256 is not None
        _load_registered_policy_freeze(
            args.policy_freeze,
            registered_policy_freeze_sha256,
            frozen_payload,
        )

        release_shards, audit_release_commitment = _load_audit_release_commitment(
            args.audit_release_manifest,
            args.audit_release_manifest_sha256,
            args.audit_labels,
            selection=selection,
            audit_contract=audit_contract,
            preregistration=preregistration,
        )
        audit_manifests, audit_manifest_artifacts = _validate_audit_manifest_sidecars(
            args.audit_labels,
            selection=selection,
            audit_contract=audit_contract,
            preregistration=preregistration,
            records=label_free_rows,
            release_shards=release_shards,
        )
        with _captured_audit_inputs(
            args.audit_labels,
            audit_manifests,
            audit_manifest_artifacts,
        ) as (
            captured_audit_paths,
            audit_artifacts,
        ):
            audit_lookup = load_independent_audit_scores(captured_audit_paths)
            expected_policy_indices = {
                quality_audit_key(row.record): (
                    row.resource_index,
                    row.original_index if row.original_index is not None else -1,
                )
                for row in label_free_rows
            }
            expected_candidate_counts = {
                quality_audit_key(row.record): len(row.exact) for row in label_free_rows
            }
            (
                audit_paper_bindings,
                audit_strength_scores,
                audit_strength_success_counts,
            ) = _load_audit_paper_bindings(
                captured_audit_paths,
                quality_audit_key,
                expected_policy_indices,
                expected_candidate_counts=expected_candidate_counts,
                registered_strength_count=int(
                    audit_contract.secondary_endpoints["strength_robustness"][
                        "registered_strength_count"
                    ]
                ),
                registered_reads_per_strength=int(contract_audit["reads_per_strength"]),
            )
            expected_audit_keys = [quality_audit_key(row.record) for row in label_free_rows]
            if len(set(expected_audit_keys)) != len(expected_audit_keys):
                raise ValueError("fixed test partition repeats an audit identity")
            expected_audit_key_set = set(expected_audit_keys)
            if (
                set(audit_lookup) != expected_audit_key_set
                or set(audit_paper_bindings) != expected_audit_key_set
                or set(audit_strength_scores) != expected_audit_key_set
                or set(audit_strength_success_counts) != expected_audit_key_set
            ):
                missing = sorted(expected_audit_key_set - set(audit_lookup))
                unexpected = sorted(set(audit_lookup) - expected_audit_key_set)
                raise ValueError(
                    "paper audit labels must match exactly the fixed test record support; "
                    f"missing={missing}, unexpected={unexpected}"
                )
            rows: list[_PaperRow] = []
            used_audit_keys = set()
            exhaustive_ground_cache: dict[str, tuple[str, float]] = {}
            for label_free in label_free_rows:
                record = label_free.record
                key = quality_audit_key(record)
                evidence = audit_lookup[key]
                if (
                    evidence.audit_schema != contract_audit["schema"]
                    or evidence.audit_schema_version != contract_audit["schema_version"]
                ):
                    raise ValueError(
                        f"audit row for {key!r} requires {contract_audit['schema']} "
                        f"schema version {contract_audit['schema_version']} from the frozen "
                        "contract"
                    )
                verified = verify_independent_audit_scores(
                    record,
                    evidence,
                    corpus_sha[str(record["_file"])],
                )
                audit_contract.validate_audit_provenance(
                    verified.provenance,
                    location=f"audit row for {key!r}",
                )
                paper_binding = audit_paper_bindings[key]
                _validate_audit_paper_binding(
                    record,
                    paper_binding,
                    selection=selection,
                    audit_contract=audit_contract,
                    preregistration=preregistration,
                    exhaustive_cache=exhaustive_ground_cache,
                )
                if len(verified.scores) != len(label_free.exact):
                    raise ValueError("verified audit vector does not cover full candidate support")
                used_audit_keys.add(key)
                rows.append(
                    _PaperRow(
                        input_index=label_free.input_index,
                        record=label_free.record,
                        output=label_free.output,
                        exact=label_free.exact,
                        resource_index=label_free.resource_index,
                        original_index=label_free.original_index,
                        record_key=label_free.record_key,
                        problem_digest=label_free.problem_digest,
                        audit_scores=tuple(float(value) for value in verified.scores),
                        audit_strength_p_solve=audit_strength_scores[key],
                        audit_strength_success_counts=audit_strength_success_counts[key],
                        audit_candidate_signature=verified.candidate_signature,
                        audit_provenance=dict(verified.provenance),
                        audit_paper_binding=dict(paper_binding),
                    )
                )
            if used_audit_keys != set(audit_lookup):
                raise RuntimeError("paper evaluation did not consume exact signed audit support")

        audit_runtimes = {
            json.dumps(
                row.audit_paper_binding["runtime_provenance"],
                sort_keys=True,
                separators=(",", ":"),
            )
            for row in rows
        }
        if len(audit_runtimes) != 1:
            raise ValueError("paper audit rows used inconsistent runtime provenance")
        audit_runtime = validate_runtime_provenance(
            rows[0].audit_paper_binding["runtime_provenance"],
            location="paper audit runtime",
            expected_device=str(audit_contract.evaluation["device"]),
        )
        full_support = _score_full_support(
            rows,
            frozen_full,
            top_tolerance=top_tolerance,
        )
        budget_sweep = _score_budget_sweep(
            rows,
            frozen_budget,
            top_tolerance=top_tolerance,
        )
        secondary_endpoints = _secondary_endpoints(
            rows,
            frozen_full,
            frozen_budget,
            exact_host_sources=exact_host_sources,
            audit_contract=audit_contract,
        )
        ground_reference_coverage = _ground_reference_coverage(rows, audit_contract)
        total_candidates = sum(len(row.exact) for row in rows)

    return {
        "artifact_schema": PAPER_ARTIFACT_SCHEMA,
        "artifact_schema_version": PAPER_ARTIFACT_SCHEMA_VERSION,
        "evaluation_mode": "paper",
        "partition": "test",
        "provisional": audit_contract.evaluation["provisional"],
        "evaluation_design": audit_contract.evaluation["evaluation_design"],
        "legacy_test_exposure": audit_contract.evaluation["legacy_test_exposure"],
        "label_fidelity": "independent-audit",
        "release_labels_consumed": False,
        "record_support": audit_contract.evaluation["record_support"],
        "candidate_support": audit_contract.evaluation["candidate_support"],
        "support_uses_labels": audit_contract.evaluation["support_uses_labels"],
        "audit_contract": audit_contract.public_binding(),
        "preregistration": preregistration.public_binding(),
        "selection": selection_public_binding(selection, registered_checkpoint),
        "checkpoint": freeze_checkpoint,
        "data_provenance": current_provenance,
        "runtime_provenance": evaluation_runtime,
        "audit_release_commitment": audit_release_commitment,
        "audit_artifacts": audit_artifacts,
        "audit_source": rows[0].audit_paper_binding["audit_source"],
        "audit_runtime_provenance": audit_runtime,
        "preprocessing": preprocessing,
        "deployment_host_contract": metadata["deployment_host_contract"],
        "exact_host_contract": exact_host_contract,
        "partition_record_counts": partition_counts,
        "n_test": len(rows),
        "n_train_encoded": 0,
        "n_val_encoded": 0,
        "audit_coverage": {
            "records_with_complete_support": len(rows),
            "record_fraction_complete": 1.0,
            "verified_records": len(rows),
            "labeled_candidates": total_candidates,
            "supported_candidates": total_candidates,
            "candidate_fraction": 1.0,
        },
        "ground_reference_coverage": ground_reference_coverage,
        "evaluated_records": [
            {
                "record_index": row.input_index,
                "file": row.record["_file"],
                "line_number": row.record["_line_number"],
                "instance_id": row.record["instance_id"],
                "focus": row.record["focus"],
                "problem_digest": row.problem_digest,
                "record_key": row.record_key,
                "corpus_sha256": corpus_sha[str(row.record["_file"])],
                "candidate_signature": row.audit_candidate_signature,
                "candidate_count": len(row.exact),
                "audit_provenance": row.audit_provenance,
                "ground_reference": row.audit_paper_binding["ground_reference"],
                "realized_strengths": row.audit_paper_binding["realized_strengths"],
                "realized_seeds": row.audit_paper_binding["realized_seeds"],
                "strength_success_counts": [
                    list(candidate_counts) for candidate_counts in row.audit_strength_success_counts
                ],
            }
            for row in rows
        ],
        "top_tolerance": top_tolerance,
        "random_seed": random_seed,
        "random_baseline_policy": audit_contract.evaluation["random_baseline_policy"],
        "lcb_z": lcb_z,
        "selection_statistic": audit_contract.evaluation["selection_statistic"],
        "label_free_policy_freeze": {
            "schema": frozen_plan["schema"],
            "schema_version": frozen_plan["schema_version"],
            "file": Path(args.policy_freeze).name,
            "sha256": frozen_sha256,
            "record_count": len(rows),
            "preregistration_manifest_caller_supplied_sha256_verified": True,
            "policy_freeze_listed_in_preregistration": True,
            "policy_freeze_bytes_match_preregistration": True,
            "byte_exact_phase_2_replay_verified": True,
            "audit_inputs_opened_after_preregistration_and_replay": True,
        },
        "full_support_metrics": full_support,
        "budget_sweep": budget_sweep,
        "secondary_endpoints": secondary_endpoints,
    }


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def _write_output(args: argparse.Namespace, result: Mapping[str, object]) -> None:
    destination = Path(args.out)
    if args.phase == "freeze":
        payload = _canonical_json_bytes(result)
        _atomic_write_bytes(destination, payload)
        digest = _bytes_sha256(payload)
        sidecar = destination.with_suffix(".sha256")
        _atomic_write_bytes(sidecar, f"{digest}  {destination.name}\n".encode())
    else:
        _atomic_write_bytes(
            destination,
            (json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8"),
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    result = evaluate(args)
    _write_output(args, result)
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
