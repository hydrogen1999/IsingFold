"""Production command line for IsingFold data preparation, training and evaluation.

Scientific commands consume only authenticated artifacts produced by ``prepare``.  The
historical in-package generator remains available as ``dev-generate`` for local smoke tests;
its output is deliberately not accepted as a production corpus.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import hmac
import inspect
import json
import math
import os
import platform
import random
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from isingfold.rl.contracts import Context, Mode, stable_digest
from isingfold.rl.data.import_embedbench import (
    canonical_json_bytes,
    content_digest,
    prepare_candidate_bank_v4,
)
from isingfold.rl.data.prepared import (
    PreparedTask,
    load_prepared_corpus_design,
)
from isingfold.rl.data.quality_attestation import QualityAttestationPin
from isingfold.rl.evaluation_strata import evaluation_contract_from_prepared

CLI_SCHEMA_VERSION = 1
RUN_SCHEMA = "isingfold.training-run"
RUN_SCHEMA_VERSION = 4
EVALUATION_REPORT_SCHEMA_VERSION = 4
WARM_START_HISTORY_SCHEMA = "isingfold.warm-start-history"
WARM_START_HISTORY_SCHEMA_VERSION = 4
QUALITY_SCHEMA = "isingfold.quality-label-corpus"
QUALITY_SCHEMA_VERSION = 8
QUALITY_DIAGNOSTIC_SCHEMA_VERSION = 7
QUALITY_ROW_SCHEMA = "isingfold.quality-counterfactual"
QUALITY_ROW_SCHEMA_VERSION = 7
QUALITY_SHARD_SCHEMA = "isingfold.quality-label-shard"
QUALITY_SHARD_SCHEMA_VERSION = 6
QUALITY_DIAGNOSTIC_SHARD_SCHEMA_VERSION = 5
QUALITY_MERGED_SCHEMA = "isingfold.quality-label-merged-corpus"
QUALITY_MERGED_SCHEMA_VERSION = 6
QUALITY_DIAGNOSTIC_MERGED_SCHEMA_VERSION = 5
QUALITY_SHARD_ASSIGNMENT = "round-robin-lineage-plan-v1"
QUALITY_PREFLIGHT_SCHEMA = "isingfold.quality-resolution-preflight"
QUALITY_PREFLIGHT_SCHEMA_VERSION = 3
QUALITY_DIAGNOSTIC_PREFLIGHT_SCHEMA_VERSION = 2
QUALITY_DIAGNOSTIC_STATUS_SCHEMA = "isingfold.quality-label-diagnostic-status"
QUALITY_DIAGNOSTIC_STATUS_VERSION = 1
SELECTOR_BUNDLE_SCHEMA = "isingfold.strength-selector-bundle"
SELECTOR_BUNDLE_SCHEMA_VERSION = 4
NORMALIZER_SCHEMA = "isingfold.if-q3-s0-selector-normalizer"
REPRESENTATION_SELECTION_SCHEMA = "isingfold.representation-selection"
REPRESENTATION_SELECTION_SCHEMA_VERSION = 4
REPRESENTATION_SELECTION_RULE = (
    "equal-seed/equal-base-lineage unconditional utility among families passing "
    "the preregistered one-sided feasibility constraint; lower online time then "
    "IF-MLP as deterministic ties"
)
GATE_RECEIPT_SCHEMA = "isingfold.release-gates"
GATE_RECEIPT_SCHEMA_VERSION = 7
_GATE_RECEIPT_FIELDS = frozenset(
    {
        "advance",
        "construction_ready",
        "context",
        "context_digest",
        "gate_profile",
        "gates",
        "grid_manifest_sha256",
        "ground_partition_receipt",
        "mandatory_gate_names",
        "optional_diagnostics",
        "parameters",
        "partition",
        "quality_authority",
        "quality_preflight_receipt_sha256",
        "quality_preflight_record_digest",
        "record_digest",
        "schema",
        "schema_version",
        "seed",
        "source_corpus_manifest_sha256",
        "source_selector_digest",
        "source_selector_fit_receipt_sha256",
        "source_selector_labels_manifest_record_digest",
        "source_selector_labels_manifest_sha256",
        "target_access",
    }
)
_EXACT_GATE_RESULT_FIELDS = frozenset(
    {
        "authenticated_exact_corpus",
        "eligible_instances",
        "exact_bounds",
        "independent_oracle",
        "independent_program_oracle",
        "instances_requested",
        "negative_structural_agreement",
        "negative_structural_checked",
        "negative_structural_unknown",
        "overlap_return_rejection_rate",
        "overlap_search_admissible_rate",
        "overlap_search_checked",
        "pass",
        "positive_structural_agreement",
        "positive_structural_checked",
        "program_faithful_rate",
        "registered_population",
        "structural_agreement",
        "structural_checked",
        "structural_unknown",
        "witness_valid_rate",
    }
)
_EXACT_CORPUS_IDENTITY_FIELDS = frozenset(
    {
        "base_lineages",
        "corpus_id",
        "exact_reproduction",
        "file_sha256",
        "max_logical_variables_per_task",
        "max_tasks",
        "path",
        "record_digest",
        "selected_marginal_census",
        "selector_implementation",
        "source_corpus_manifest_sha256",
        "source_population_census",
        "task_count",
        "task_ids",
    }
)
PRODUCTION_REWARD_READS = 256
DEFAULT_AUDIT_READS = 4096
PPO_TASK_SAMPLING_SCHEMA = "isingfold.ppo-task-sampling"
PPO_TASK_SAMPLING_VERSION = 1
PPO_LINEAGE_SCHEDULE_RULE = "canonical-base-lineage-round-robin-v1"
PPO_WITHIN_LINEAGE_RULE = "seeded-offset-cyclic-task-v1"
PPO_WITHIN_LINEAGE_SEED_DOMAIN = "ppo-within-base-lineage-task-offset-v1"
SCALABLE_GATE_SAMPLE_SCHEMA = "isingfold.scalable-gate-sample"
SCALABLE_GATE_SAMPLE_VERSION = 2
COMPLETE_CACHE_SELECTION_MODE = "rl-value-validation-freeze-complete-cache-k2-v2"


@dataclasses.dataclass(frozen=True)
class _VerifiedQualityAttestationPin(QualityAttestationPin):
    """Publisher pin augmented by the public, partition-sealed ground root."""

    ground_certificate_root_path: str
    ground_certificate_root_sha256: str
    ground_certificate_root_record_digest: str
    corpus_directory: str
    global_quality_authority: Mapping[str, object]


# Kept torch-free so data preparation remains available without the optional ``rl`` extra.
# ``_model`` verifies this CLI registry against ``model.MODEL_FAMILIES`` before construction.
REGISTERED_MODEL_FAMILIES = ("if-mlp", "if-dual", "if-core")
TRAINING_METHODS = ("supervised-only", "ppo-warm-start", "ppo-from-scratch")


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def _registered_reward_reads(value: str) -> int:
    result = _positive_int(value)
    if result != PRODUCTION_REWARD_READS:
        raise argparse.ArgumentTypeError(
            f"production training reward reads are fixed at {PRODUCTION_REWARD_READS}"
        )
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return result


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return result


def _nonnegative_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative and finite")
    return result


def _probability(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("value must be finite and in [0, 1]")
    return result


def _open_probability(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise argparse.ArgumentTypeError("value must be finite and strictly between 0 and 1")
    return result


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_lower_sha256(value: object) -> bool:
    return bool(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_lower_sha256(value: object, label: str) -> str:
    if not _is_lower_sha256(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _runtime_platform_identity() -> dict[str, object]:
    """Stable host-class fields needed before comparing measured online latency."""

    processor = platform.processor().strip()
    cpuinfo = Path("/proc/cpuinfo")
    if not processor and cpuinfo.is_file():
        try:
            for line in cpuinfo.read_text(encoding="utf-8").splitlines():
                if line.lower().startswith("model name") and ":" in line:
                    processor = line.split(":", 1)[1].strip()
                    break
        except OSError:
            processor = ""
    return {
        "hostname": platform.node() or "unspecified",
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": processor or "unspecified",
        "logical_cpu_count": os.cpu_count(),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }


def _inference_device_name(device, runtime_platform: Mapping[str, object]) -> str:
    if device.type == "cuda":
        import torch

        return str(torch.cuda.get_device_name(device))
    if device.type == "mps":
        return f"Apple-{runtime_platform['machine']}"
    return str(runtime_platform["processor"])


def _validate_registered_complete_config(
    protocol: Mapping[str, object],
    *,
    arm: str,
    semantic_digest: str,
    file_sha256: str,
) -> None:
    """Fail before test loading unless a config matches both preregistered pins."""

    if arm not in {"learned", "external"}:
        raise ValueError("unknown complete-system config arm")
    expected_digest = protocol.get(f"{arm}_config_digest")
    expected_sha = protocol.get(f"{arm}_config_file_sha256")
    if semantic_digest != expected_digest or file_sha256 != expected_sha:
        raise ValueError(
            f"{arm} complete-system config differs from the preregistered semantic/file pins"
        )


def _strict_json_object(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise ValueError(f"{label} contains non-finite number {token}")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain one JSON object")
    # Canonicalisation performs a recursive finite-number and key-type check.
    canonical_json_bytes(value)
    return value


def _strict_json(path: str | os.PathLike[str]) -> dict[str, Any]:
    source = Path(path)
    return _strict_json_object(source.read_bytes(), str(source))


def _publisher_attestation_pin(args: argparse.Namespace) -> QualityAttestationPin:
    """Construct the explicit publisher trust anchor without opening evaluator targets."""

    path = getattr(args, "quality_attestation", None)
    digest = getattr(args, "expected_quality_attestation_digest", None)
    publisher_id = getattr(args, "expected_quality_publisher_id", None)
    if not all(isinstance(value, str) and value for value in (path, digest, publisher_id)):
        raise ValueError(
            "trusted target access requires --quality-attestation, "
            "--expected-quality-attestation-digest, and --expected-quality-publisher-id"
        )
    return QualityAttestationPin(
        path=path,
        expected_digest=digest,
        expected_publisher_id=publisher_id,
    )


def _quality_attestation_pin(args: argparse.Namespace) -> _VerifiedQualityAttestationPin:
    """Authenticate the publisher and target-free ground root without opening targets."""

    from isingfold.rl.data.ground_certificate import (
        load_ground_certificate_root,
        project_ground_root_quality_authority,
    )

    publisher_pin = _publisher_attestation_pin(args)
    root_path = getattr(args, "ground_certificate_root", None)
    root_sha256 = getattr(
        args,
        "expected_ground_certificate_root_sha256",
        None,
    )
    corpus = getattr(args, "corpus", None)
    if (
        not all(
            isinstance(value, (str, os.PathLike)) and str(value) for value in (root_path, corpus)
        )
        or not isinstance(root_sha256, str)
        or not root_sha256
    ):
        raise ValueError(
            "trusted target access requires --ground-certificate-root and "
            "--expected-ground-certificate-root-sha256 for the current --corpus"
        )
    root = load_ground_certificate_root(
        root_path,
        expected_sha256=root_sha256,
        corpus_directory=corpus,
        quality_attestation_pin=publisher_pin,
    )
    global_authority, partition_authority = project_ground_root_quality_authority(root)
    if partition_authority is not None:
        raise RuntimeError("target-free ground-root projection unexpectedly opened a partition")
    return _VerifiedQualityAttestationPin(
        path=publisher_pin.path,
        expected_digest=publisher_pin.expected_digest,
        expected_publisher_id=publisher_pin.expected_publisher_id,
        ground_certificate_root_path=str(Path(root_path)),
        ground_certificate_root_sha256=root_sha256,
        ground_certificate_root_record_digest=root.record_digest,
        corpus_directory=str(Path(corpus)),
        global_quality_authority=global_authority.as_dict(),
    )


def _normalize_quality_partition(partition: str) -> str:
    normalized = "val" if partition == "validation" else partition
    if normalized not in {"train", "val", "test"}:
        raise ValueError(f"unsupported quality partition {partition!r}")
    return normalized


def _quality_authority_binding(
    global_authority: Mapping[str, object],
    partition_authority: Mapping[str, object],
    *,
    role: str,
) -> dict[str, object]:
    """Seal one opened target partition under a cross-partition-safe authority."""

    if role not in {"training_partition", "evaluation_partition"}:
        raise ValueError("quality-authority binding role is invalid")
    payload: dict[str, object] = {
        "schema": "isingfold.quality-authority-binding",
        "schema_version": 2,
        "global": dict(global_authority),
        role: dict(partition_authority),
    }
    binding = {**payload, "record_digest": content_digest(payload)}
    _validate_quality_authority_binding(binding, expected_role=role)
    return binding


def _validate_quality_authority_binding(
    authority: object,
    *,
    expected_role: str | None = None,
) -> dict[str, object]:
    """Validate the canonical v2 global-plus-partition authority contract."""

    if not isinstance(authority, dict):
        raise ValueError("quality authority must be an object")
    roles = {"training_partition", "evaluation_partition"} & set(authority)
    if len(roles) != 1:
        raise ValueError("quality authority must bind exactly one partition role")
    role = next(iter(roles))
    if expected_role is not None and role != expected_role:
        raise ValueError(f"quality authority must bind {expected_role}")
    expected_fields = {"schema", "schema_version", "global", role, "record_digest"}
    if set(authority) != expected_fields:
        raise ValueError("quality-authority binding schema differs")
    _verify_record(authority, "quality-authority binding")
    if (
        authority["schema"] != "isingfold.quality-authority-binding"
        or authority["schema_version"] != 2
    ):
        raise ValueError("unsupported quality-authority binding")
    global_authority = authority["global"]
    partition_authority = authority[role]
    if not isinstance(global_authority, dict) or not isinstance(partition_authority, dict):
        raise ValueError("quality-authority components must be objects")
    _verify_record(global_authority, "global quality authority")
    _verify_record(partition_authority, "partition quality authority")
    if (
        global_authority.get("schema") != "isingfold.global-quality-authority"
        or global_authority.get("schema_version") != 1
        or partition_authority.get("schema") != "isingfold.partition-quality-authority"
        or partition_authority.get("schema_version") != 1
    ):
        raise ValueError("quality-authority component schema differs")
    expected_name = "train" if role == "training_partition" else None
    if expected_name is not None and partition_authority.get("name") != expected_name:
        raise ValueError("training quality authority must bind the train partition")
    if partition_authority.get("name") not in {"train", "val", "test"}:
        raise ValueError("quality authority has an invalid partition name")
    return authority


def _load_quality_partition(
    corpus: str | os.PathLike[str],
    *,
    partition: str,
    pin: _VerifiedQualityAttestationPin,
    role: str | None = None,
) -> tuple[
    tuple[PreparedTask, ...],
    dict[str, object],
    dict[str, object],
    dict[str, object],
]:
    """Open exactly one certified target partition and retain every access receipt."""

    from isingfold.rl.data.ground_certificate import (
        load_ground_certificate_partition,
        project_ground_partition_quality_authority,
    )

    normalized = _normalize_quality_partition(partition)
    if Path(corpus).resolve() != Path(pin.corpus_directory).resolve():
        raise ValueError("quality pin is bound to another prepared corpus")
    loaded = load_ground_certificate_partition(
        pin.ground_certificate_root_path,
        expected_root_sha256=pin.ground_certificate_root_sha256,
        partition=normalized,
        corpus_directory=corpus,
        quality_attestation_pin=QualityAttestationPin(
            path=pin.path,
            expected_digest=pin.expected_digest,
            expected_publisher_id=pin.expected_publisher_id,
        ),
    )
    global_authority, partition_authority = project_ground_partition_quality_authority(loaded)
    if global_authority.as_dict() != dict(pin.global_quality_authority):
        raise ValueError("opened partition differs from the pinned global authority")
    resolved_role = role or (
        "training_partition" if normalized == "train" else "evaluation_partition"
    )
    binding = _quality_authority_binding(
        global_authority.as_dict(),
        partition_authority.as_dict(),
        role=resolved_role,
    )
    return (
        loaded.tasks,
        binding,
        loaded.target_access.as_dict(),
        loaded.as_dict(),
    )


def _quality_authority_identity(
    prepared: Sequence[PreparedTask], *, pin: object
) -> dict[str, object]:
    """Compatibility projection for an already authenticated single partition."""

    if not prepared:
        raise ValueError("quality authority cannot be derived from an empty task set")
    if not isinstance(pin, _VerifiedQualityAttestationPin):
        raise TypeError("quality authority requires a verified ground-root pin")
    partitions = {_normalize_quality_partition(item.partition) for item in prepared}
    if len(partitions) != 1:
        raise ValueError("quality authority cannot merge distinct target partitions")
    partition = next(iter(partitions))
    from isingfold.rl.data.ground_certificate import (
        load_ground_certificate_root,
        project_ground_root_quality_authority,
    )

    root = load_ground_certificate_root(
        pin.ground_certificate_root_path,
        expected_sha256=pin.ground_certificate_root_sha256,
        corpus_directory=pin.corpus_directory,
        quality_attestation_pin=QualityAttestationPin(
            path=pin.path,
            expected_digest=pin.expected_digest,
            expected_publisher_id=pin.expected_publisher_id,
        ),
    )
    global_authority, partition_authority = project_ground_root_quality_authority(
        root, partition=partition
    )
    if partition_authority is None:
        raise RuntimeError("partition quality-authority projection disappeared")
    role = "training_partition" if partition == "train" else "evaluation_partition"
    return _quality_authority_binding(
        global_authority.as_dict(), partition_authority.as_dict(), role=role
    )


def _verify_record(record: Mapping[str, Any], label: str) -> None:
    recorded = record.get("record_digest")
    if not isinstance(recorded, str) or len(recorded) != 64:
        raise ValueError(f"{label} has no valid record_digest")
    payload = {key: value for key, value in record.items() if key != "record_digest"}
    if recorded != content_digest(payload):
        raise ValueError(f"{label} record digest mismatch")


def _atomic_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _publish_gate_receipt(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> Path:
    """Publish the release-gate receipt with atomic no-replace semantics."""

    from isingfold.rl.data.exact_conformance import publish_new_file

    return publish_new_file(path, canonical_json_bytes(payload) + b"\n")


def _with_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**payload, "record_digest": content_digest(payload)}


def _load_exact_conformance_tasks(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    corpus: str | os.PathLike[str],
    expected_corpus_manifest_sha256: str,
) -> tuple[list[PreparedTask], dict[str, object]]:
    """Authenticate and recompute the bounded corpus used by Gate 1."""

    from isingfold.rl.data.exact_conformance import (
        load_authenticated_exact_conformance_registry,
    )

    return load_authenticated_exact_conformance_registry(
        corpus,
        expected_corpus_manifest_sha256=expected_corpus_manifest_sha256,
        registry=path,
        expected_registry_sha256=expected_sha256,
    )


def _stratified_gate_sample(
    prepared: Sequence[PreparedTask],
    *,
    count: int,
    seed: int,
) -> tuple[list[PreparedTask], dict[str, object]]:
    """Select one deterministic representative per base lineage, balanced over strata."""

    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("scalable gate sample count must be a positive integer")
    by_lineage: dict[str, list[PreparedTask]] = {}
    for item in prepared:
        lineage = item.task.lineage
        if not isinstance(lineage, str) or not lineage:
            raise ValueError("scalable gate sampling requires nonempty base lineages")
        by_lineage.setdefault(lineage, []).append(item)
    if count > len(by_lineage):
        raise ValueError("scalable gate sample requests more tasks than independent base lineages")

    def stratum_coordinates(item: PreparedTask) -> dict[str, object]:
        if item.design_condition is None:
            return {"legacy_unspecified": True}
        snapshot = _jsonable(item.design_condition)
        if not isinstance(snapshot, dict):  # pragma: no cover - dataclass/mapping contract
            raise TypeError("prepared design condition must encode as an object")
        # These fields identify a particular lineage/condition record.  Including them in
        # the stratum key would turn every lineage into its own stratum and defeat balancing.
        excluded = {
            "base_lineage_key",
            "calibration_sha256",
            "learning_partition",
            "registry_row_digest",
        }
        coordinates = {key: snapshot[key] for key in sorted(snapshot) if key not in excluded}
        if not coordinates:
            raise ValueError("prepared design condition has no categorical stratum coordinates")
        return coordinates

    candidates: list[tuple[str, PreparedTask, dict[str, object], str]] = []
    stratum_registry: dict[str, dict[str, object]] = {}
    for item in prepared:
        lineage = item.task.lineage
        coordinates = stratum_coordinates(item)
        stratum_digest = content_digest(coordinates)
        previous = stratum_registry.setdefault(stratum_digest, coordinates)
        if previous != coordinates:  # pragma: no cover - SHA-256 collision guard
            raise RuntimeError("categorical gate strata have a digest collision")
        candidates.append((lineage, item, coordinates, stratum_digest))

    axis_field_sets = {tuple(sorted(coordinates)) for _, _, coordinates, _ in candidates}
    if len(axis_field_sets) != 1:
        raise ValueError("scalable gate rows do not share one categorical stratum schema")
    axis_fields = next(iter(axis_field_sets))
    if any(
        not isinstance(coordinates[field], (str, bool))
        for _, _, coordinates, _ in candidates
        for field in axis_fields
    ):
        raise ValueError("scalable gate stratum coordinates must be categorical scalars")

    def level_label(value: object) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, bool):
            return "true" if value else "false"
        raise TypeError("unreachable noncategorical scalable-gate coordinate")

    source_axis_lineages: dict[str, dict[str, set[str]]] = {field: {} for field in axis_fields}
    for lineage, _, coordinates, _ in candidates:
        for field in axis_fields:
            label = level_label(coordinates[field])
            source_axis_lineages[field].setdefault(label, set()).add(lineage)

    selected_axis_counts: dict[str, dict[str, int]] = {field: {} for field in axis_fields}
    selected_stratum_counts = {stratum: 0 for stratum in stratum_registry}
    selected: list[PreparedTask] = []
    selected_lineages: set[str] = set()
    selected_strata: list[str] = []
    while len(selected) < count:
        eligible = [row for row in candidates if row[0] not in selected_lineages]
        if not eligible:
            raise RuntimeError("scalable gate sampler exhausted before its requested census")

        # Cover authenticated marginal levels first.  Joint-stratum round robin alone can take
        # several distinct cells that are nevertheless all easy, pristine, or from one host.
        # The remaining terms minimise marginal and joint-stratum reuse before a seeded tie.
        def selection_key(
            row: tuple[str, PreparedTask, dict[str, object], str],
        ) -> tuple[object, ...]:
            lineage, item, coordinates, stratum = row
            loads = tuple(
                selected_axis_counts[field].get(level_label(coordinates[field]), 0)
                for field in axis_fields
            )
            uncovered = sum(load == 0 for load in loads)
            return (
                -uncovered,
                max(loads, default=0),
                sum(loads),
                selected_stratum_counts[stratum],
                content_digest(
                    {
                        "domain": "scalable-gate-marginal-balance-tie-v1",
                        "seed": seed,
                        "stratum": stratum,
                        "lineage": lineage,
                        "task_id": item.task_id,
                    }
                ),
                lineage,
                item.task_id,
            )

        lineage, item, coordinates, stratum = min(eligible, key=selection_key)
        selected.append(item)
        selected_lineages.add(lineage)
        selected_strata.append(stratum)
        selected_stratum_counts[stratum] += 1
        for field in axis_fields:
            label = level_label(coordinates[field])
            selected_axis_counts[field][label] = selected_axis_counts[field].get(label, 0) + 1
    receipt = {
        "schema": SCALABLE_GATE_SAMPLE_SCHEMA,
        "schema_version": SCALABLE_GATE_SAMPLE_VERSION,
        "seed": seed,
        "independent_unit": "immutable-base-lineage",
        "strategy": "deterministic-greedy-authenticated-marginal-balance-v1",
        "source_task_count": len(prepared),
        "source_lineage_count": len(by_lineage),
        "source_stratum_count": len(stratum_registry),
        "source_axis_lineage_census": {
            field: {
                level: len(lineages)
                for level, lineages in sorted(source_axis_lineages[field].items())
            }
            for field in axis_fields
        },
        "stratum_registry": [
            {
                "stratum_digest": stratum,
                "coordinates": stratum_registry[stratum],
            }
            for stratum in sorted(stratum_registry)
        ],
        "selected_task_ids": [item.task_id for item in selected],
        "selected_base_lineages": [item.task.lineage for item in selected],
        "selected_stratum_digests": selected_strata,
        "selected_stratum_census": {
            stratum: selected_stratum_counts[stratum]
            for stratum in sorted(selected_stratum_counts)
            if selected_stratum_counts[stratum]
        },
        "selected_axis_census": {
            field: dict(sorted(selected_axis_counts[field].items())) for field in axis_fields
        },
    }
    return selected, {**receipt, "sampling_digest": content_digest(receipt)}


def _jsonable(value: object) -> object:
    """Create a canonical JSON snapshot without deepcopying immutable mapping proxies."""

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name)) for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"cannot snapshot {type(value).__name__} as canonical JSON")


def _context_snapshot(context: Context) -> dict[str, object]:
    snapshot = _jsonable(context)
    if not isinstance(snapshot, dict):  # pragma: no cover - Context is a dataclass by contract
        raise TypeError("Context snapshot must be a mapping")
    return snapshot


def _corpus_manifest_digest(corpus: str | os.PathLike[str]) -> str:
    manifest = Path(corpus) / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"prepared corpus manifest is missing: {manifest}")
    return _sha256_file(manifest)


def _registered_qubit_cap(corpus: str | os.PathLike[str], requested: int | None) -> int:
    manifest = _strict_json(Path(corpus) / "manifest.json")
    cap = manifest.get("qubit_cap")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise ValueError("prepared corpus has an invalid qubit cap")
    if requested is not None and requested != cap:
        raise ValueError(
            f"requested qubit cap {requested} differs from authenticated corpus cap {cap}"
        )
    return cap


def _context(args: argparse.Namespace, corpus: str | os.PathLike[str] | None = None) -> Context:
    requested = getattr(args, "qubit_cap", None)
    if corpus is None:
        if requested is None:
            raise ValueError("qubit cap is required without a prepared corpus")
        cap = requested
    else:
        cap = _registered_qubit_cap(corpus, requested)
    return Context(qubit_cap=cap, n_est_reads=PRODUCTION_REWARD_READS)


def _resolve_training_policy_context(
    base_context: Context,
    *,
    complete_system_no_restart: bool,
    deployment_initializer_bank: bool,
    complete_config: object | None,
) -> tuple[Context, str | None]:
    """Bind training support to one registered initializer/restart contract.

    The historical complete-system confirmation arm disabled policy restarts because it
    could not replay a native initializer.  Profile-I deployment-bank training instead
    consumes the persistent K=2 restart cache and must retain the registered restart
    allowance in the actor context.  Treating both paths as merely "bank backed" silently
    erased the restart actions from the latter task.
    """

    from isingfold.rl.complete_system import (
        COMPLETE_POLICY_RESTART_MODE,
        LEGACY_COMPLETE_POLICY_RESTART_MODE,
        CompleteSystemConfig,
        complete_policy_context,
    )

    if complete_system_no_restart and deployment_initializer_bank:
        raise ValueError(
            "legacy no-restart and persistent-cache deployment modes are mutually exclusive"
        )
    if not complete_system_no_restart and not deployment_initializer_bank:
        if complete_config is not None:
            raise ValueError("a complete-system config requires a registered bank policy")
        return base_context, None
    if not isinstance(complete_config, CompleteSystemConfig):
        raise TypeError("bank-backed training requires a parsed complete-system config")

    if complete_system_no_restart:
        if complete_config.policy_restart_mode != LEGACY_COMPLETE_POLICY_RESTART_MODE:
            raise ValueError("legacy complete-system training requires the no-restart config")
        return complete_policy_context(base_context), complete_config.policy_restart_mode

    if complete_config.policy_restart_mode != COMPLETE_POLICY_RESTART_MODE:
        raise ValueError(
            "deployment initializer-bank training requires the persistent K=2 restart cache"
        )
    return base_context, complete_config.policy_restart_mode


def _partition_roots(split: dict[str, object], partition: str) -> set[str]:
    """Resolve one registered partition and reject missing, empty or overlapping roots."""

    aliases = ("validation", "val") if partition in {"validation", "val"} else (partition,)
    present = [name for name in aliases if name in split]
    if len(present) != 1:
        raise ValueError(f"partition {partition!r} is missing or ambiguously aliased")
    values = split[present[0]]
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(item, str) or not item for item in values)
    ):
        raise ValueError(f"partition {partition!r} must be a non-empty list of lineage IDs")
    roots = set(values)
    if len(roots) != len(values):
        raise ValueError(f"partition {partition!r} contains duplicate lineage IDs")
    memberships: dict[str, list[str]] = {}
    for name, raw in split.items():
        if name not in {"train", "validation", "val", "test", "ood"} or not isinstance(raw, list):
            continue
        canonical = "validation" if name == "val" else name
        for root in raw:
            if isinstance(root, str):
                memberships.setdefault(root, []).append(canonical)
    overlaps = {root: names for root, names in memberships.items() if len(set(names)) > 1}
    if overlaps:
        raise ValueError(f"lineage leakage across registered partitions: {overlaps}")
    return roots


def _select_tasks(tasks, split: dict[str, object], partition: str):
    """Compatibility helper for old tests; production uses ``load_prepared_tasks``."""

    roots = _partition_roots(split, partition)
    selected = [task for task in tasks if task.lineage in roots]
    if not selected:
        raise ValueError(f"partition {partition!r} has no matching tasks in the corpus")
    registered = set()
    for name in ("train", "validation", "test"):
        if name in split or (name == "validation" and "val" in split):
            registered.update(_partition_roots(split, name))
    unknown = sorted({task.lineage for task in tasks if task.lineage not in registered})
    if unknown:
        raise ValueError(f"corpus contains unassigned task lineages: {unknown[:8]}")
    return selected


def _resolve_device(requested: str):
    import torch

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise ValueError("MPS was requested but is unavailable")
    return device


def _seed_runtime(seed: int, *, deterministic: bool, threads: int) -> None:
    import torch

    if isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(threads, bool) or threads <= 0:
        raise ValueError("threads must be a positive integer")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(bool(deterministic))
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic


def _domain_seed(root: int, domain: str, *parts: object) -> int:
    raw = canonical_json_bytes(
        {"domain": domain, "parts": [root, *parts], "schema_version": CLI_SCHEMA_VERSION}
    )
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & (2**31 - 1)


class _LineageEqualTaskScheduler:
    """Deterministic PPO task schedule with one vote per immutable base lineage.

    Generated variants and realized host conditions are nested observations of one base
    logical lineage.  They therefore cannot occupy extra slots in the outer PPO episode
    schedule.  The outer schedule is exact round robin; the inner schedule cycles through
    that lineage's canonical task registry from a separately seeded offset.
    """

    def __init__(self, tasks: Sequence[PreparedTask], *, seed: int) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("PPO task-sampling seed must be a nonnegative integer")
        if not tasks:
            raise ValueError("PPO task sampling requires at least one prepared task")
        grouped: dict[str, list[PreparedTask]] = {}
        seen_task_ids: set[str] = set()
        for item in tasks:
            if not isinstance(item, PreparedTask):
                raise TypeError("PPO task sampling requires typed PreparedTask records")
            lineage = item.task.lineage
            if not isinstance(lineage, str) or not lineage:
                raise ValueError("PPO task sampling requires nonempty base lineages")
            if item.provenance is not None and item.provenance.base_parent_lineage != lineage:
                raise ValueError("prepared task lineage differs from its authenticated base parent")
            if item.task_id in seen_task_ids:
                raise ValueError("PPO task sampling registry repeats a task ID")
            seen_task_ids.add(item.task_id)
            grouped.setdefault(lineage, []).append(item)
        self.seed = seed
        self.lineages = tuple(sorted(grouped))
        self.tasks_by_lineage = {
            lineage: tuple(sorted(grouped[lineage], key=lambda item: item.task_id))
            for lineage in self.lineages
        }
        task_registry = {
            lineage: [item.task_id for item in self.tasks_by_lineage[lineage]]
            for lineage in self.lineages
        }
        rule = {
            "independent_unit": "immutable-base-lineage",
            "lineage_schedule": PPO_LINEAGE_SCHEDULE_RULE,
            "within_lineage_schedule": PPO_WITHIN_LINEAGE_RULE,
            "within_lineage_seed_domain": PPO_WITHIN_LINEAGE_SEED_DOMAIN,
            "initializer_retry_policy": "same-episode-index-same-task-new-environment-seed",
        }
        self.receipt = {
            "schema": PPO_TASK_SAMPLING_SCHEMA,
            "schema_version": PPO_TASK_SAMPLING_VERSION,
            "seed": seed,
            "rule": rule,
            "rule_digest": content_digest(rule),
            "ordered_lineages": list(self.lineages),
            "lineage_count": len(self.lineages),
            "task_count": len(tasks),
            "lineage_task_counts": {
                lineage: len(self.tasks_by_lineage[lineage]) for lineage in self.lineages
            },
            "lineage_task_registry_digest": content_digest(task_registry),
        }
        self.digest = content_digest(self.receipt)

    def select(self, episode_schedule_index: int) -> PreparedTask:
        if (
            isinstance(episode_schedule_index, bool)
            or not isinstance(episode_schedule_index, int)
            or episode_schedule_index < 0
        ):
            raise ValueError("PPO episode schedule index must be a nonnegative integer")
        lineage_count = len(self.lineages)
        lineage = self.lineages[episode_schedule_index % lineage_count]
        lineage_occurrence = episode_schedule_index // lineage_count
        candidates = self.tasks_by_lineage[lineage]
        offset = _domain_seed(
            self.seed,
            PPO_WITHIN_LINEAGE_SEED_DOMAIN,
            lineage,
        ) % len(candidates)
        return candidates[(offset + lineage_occurrence) % len(candidates)]


def _proposal_version() -> str:
    from isingfold.rl import proposal

    declared = getattr(proposal, "PROPOSAL_VERSION", None)
    if isinstance(declared, str) and declared:
        return declared
    # Until proposal.py publishes a semantic version, source identity prevents an unsafe resume.
    return f"source-sha256:{_sha256_file(proposal.__file__)}"


def _implementation_identity(target: object) -> dict[str, str]:
    source = inspect.getsourcefile(target)
    if source is None or not Path(source).is_file():
        raise RuntimeError(f"cannot identify implementation source for {target!r}")
    module = getattr(target, "__module__", None)
    qualname = getattr(target, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(qualname, str):
        raise RuntimeError(f"cannot identify implementation name for {target!r}")
    return {
        "implementation": f"{module}.{qualname}",
        "source_sha256": _sha256_file(source),
    }


def _quality_implementation_contract(ctx: Context) -> dict[str, object]:
    """Bind every executable choice that defines a counterfactual Q^mu label."""

    from isingfold.rl.data import quality
    from isingfold.rl.data.action_certificate import (
        APPLIED_ACTION_SCHEMA,
        ENVELOPE_SCHEMA,
        SCHEMA_VERSION as ACTION_CERTIFICATE_SCHEMA_VERSION,
        StateActionEnvelopeV1,
        apply_envelope_action,
    )
    from isingfold.rl.data.prepared import PreparedTask
    from isingfold.rl.env import task_initializer
    from isingfold.rl.evaluator import sample_program
    from isingfold.rl.program import compile_registry
    from isingfold.rl.strength import StrengthSelectorModel
    from isingfold.rl.strength_tensorize import build_strength_inputs
    from isingfold.rl.validate import p_return

    return {
        "proposal_generator_version": _proposal_version(),
        "initializer": {
            "artifact_schema": "isingfold.profile-i-initializer",
            "artifact_schema_version": 1,
            "prepared_adapter": _implementation_identity(PreparedTask.initializer),
            "runtime_initializer": _implementation_identity(task_initializer),
        },
        "prefix_policy": {
            **_implementation_identity(_quality_prefixes),
            "rule": "uniform-among-legal-nonterminal-actions",
        },
        "continuation_policy": {
            **_implementation_identity(quality.random_masked_policy),
            "policy_id": quality.CONTINUATION_POLICY_ID,
            "rule": "uniform-among-exact-legal-actions",
            "max_steps": quality.CONTINUATION_MAX_STEPS,
            "rng_rule": "domain-separated-from-recorded-continuation-seed",
            "runner": _implementation_identity(quality.run_continuation),
        },
        "action_subset_policy": {
            **_implementation_identity(quality.deterministic_action_sample),
            "policy_id": quality.ACTION_SUBSET_POLICY_ID,
            "rule": (
                "protect-archive-zero-commit-then-sample-uniformly-without-replacement-"
                "from-all-remaining-legal-actions"
            ),
            "stored_quantity": "exact-first-order-inclusion-probability-per-selected-action",
        },
        "archive_semantics": {
            "mode": Mode.IMPROVEMENT.value,
            "protected_initial_slots": ctx.archive_protected,
            "mutable_fifo_slots": ctx.archive_fifo,
            "return_rule": "selected-realized-valid-only; protected-initial-fallback",
        },
        "labeler": _implementation_identity(quality.label_decision),
        "structural_action_bridge": {
            "envelope_schema": ENVELOPE_SCHEMA,
            "applied_action_schema": APPLIED_ACTION_SCHEMA,
            "schema_version": ACTION_CERTIFICATE_SCHEMA_VERSION,
            "envelope_capture": _implementation_identity(StateActionEnvelopeV1.capture),
            "successor_replay": _implementation_identity(apply_envelope_action),
            "terminal_actions": "envelope-bound-with-no-workspace-successor",
        },
        "terminal_semantics": {
            "evaluator": _implementation_identity(sample_program),
            "compiler": _implementation_identity(compile_registry),
            "return_validator": _implementation_identity(p_return),
            "selector_runtime": _implementation_identity(StrengthSelectorModel.select_embedding),
            "selector_tensorizer": _implementation_identity(build_strength_inputs),
        },
        "best_action_rule": {
            "rule_id": quality.BEST_ACTION_RULE_ID,
            "familywise_alpha": quality.BEST_ACTION_ALPHA,
            "decision": "retain-actions-whose-UCB-reaches-best-LCB",
            "fully_unresolved_rows": ("retain-qmu-and-state-value-mask-only-tied-best-ranking"),
        },
        "label_version": quality.QUALITY_LABEL_VERSION,
    }


def _model(
    model_family: str,
    *,
    improvement_mode: bool = True,
    quality_prior_mode: str | None = None,
):
    """Construct a registered family through the single normative model factory."""

    from isingfold.rl.model import (
        MODEL_FAMILIES,
        QUALITY_POLICY_PRIOR_DEFAULT,
        build_model,
    )

    if tuple(MODEL_FAMILIES) != REGISTERED_MODEL_FAMILIES:
        raise RuntimeError("CLI and model-factory family registries differ")
    return build_model(
        model_family,
        improvement_mode=improvement_mode,
        quality_prior_mode=(quality_prior_mode or QUALITY_POLICY_PRIOR_DEFAULT),
    )


def _model_identity(model: Any, model_family: str) -> dict[str, object]:
    """Bind a checkpoint to the concrete object returned by ``build_model``."""

    from isingfold.rl.model import (
        ACTION_QUALITY_HEAD,
        ACTION_QUALITY_POLICY_COUPLING_SCHEMA,
        quality_policy_prior_contract,
    )

    if model_family not in REGISTERED_MODEL_FAMILIES:
        raise ValueError(f"unknown registered model family {model_family!r}")
    declared = getattr(model, "model_family", None)
    canonical_declared = declared.strip().lower() if isinstance(declared, str) else None
    if canonical_declared != model_family:
        raise ValueError(
            f"model factory returned family {declared!r} for requested {model_family!r}"
        )
    local_blocks = getattr(model, "local_l", None)
    hardware_blocks = getattr(model, "local_h", None)
    fusion_blocks = getattr(model, "fusion", None)
    if local_blocks is None or hardware_blocks is None or fusion_blocks is None:
        raise TypeError("registered model does not expose its normative block structure")
    if len(local_blocks) != len(hardware_blocks):
        raise ValueError("logical and hardware local-block counts differ")
    quality_prior_mode = getattr(model, "quality_prior_mode", None)
    if not isinstance(quality_prior_mode, str):
        raise ValueError("registered model has no quality-prior mode")
    expected_quality_coupling = quality_policy_prior_contract(quality_prior_mode)
    if (
        getattr(model, "action_quality_head", None) != ACTION_QUALITY_HEAD
        or getattr(model, "action_quality_policy_coupling", None) != expected_quality_coupling
        or expected_quality_coupling.get("schema") != ACTION_QUALITY_POLICY_COUPLING_SCHEMA
    ):
        raise ValueError("registered model action-quality policy prior differs")
    projector = getattr(model, "proj_logical", None)
    try:
        width = int(projector[0].out_features)
    except (AttributeError, IndexError, TypeError) as exc:
        raise TypeError("registered model does not expose its hidden width") from exc
    parameter_count = (
        int(model.parameter_count())
        if hasattr(model, "parameter_count")
        else sum(parameter.numel() for parameter in model.parameters())
    )
    implementation_source = inspect.getsourcefile(type(model))
    if implementation_source is None or not Path(implementation_source).is_file():
        raise RuntimeError("registered model implementation source cannot be identified")
    return {
        "family": model_family,
        "declared_family": declared,
        "factory": "isingfold.rl.model.build_model",
        "implementation": f"{type(model).__module__}.{type(model).__qualname__}",
        "implementation_source_sha256": _sha256_file(implementation_source),
        "width": width,
        "local_blocks": len(local_blocks),
        "fusion_blocks": len(fusion_blocks),
        "improvement_mode": bool(getattr(model, "improvement_mode", False)),
        "action_quality_head": ACTION_QUALITY_HEAD,
        "quality_prior_mode": quality_prior_mode,
        "action_quality_policy_coupling": expected_quality_coupling,
        "parameter_count": parameter_count,
    }


@dataclasses.dataclass(frozen=True)
class SelectorBundle:
    model: Any
    selector_digest: str
    normalizer: dict[str, object]
    normalizer_digest: str
    coefficient_scale: float
    corpus_manifest_sha256: str
    quality_authority: dict[str, object]
    target_access: dict[str, object]
    ground_partition_receipt: dict[str, object]
    root: Path


def _load_selector_bundle(
    path: str | os.PathLike[str], *, corpus: str | os.PathLike[str]
) -> SelectorBundle:
    from isingfold.rl.data.selector_labels import terminal_program_mixture_registry
    from isingfold.rl.strength import StrengthSelectorModel

    root = Path(path)
    if not root.is_dir():
        raise ValueError("production selector must be a selector-bundle directory")
    expected_files = {"selector.pt", "normalizer.json", "fit_receipt.json"}
    actual_files = {item.name for item in root.iterdir() if item.is_file()}
    if actual_files != expected_files:
        raise ValueError(
            f"selector bundle file set differs: missing={sorted(expected_files - actual_files)}, "
            f"unknown={sorted(actual_files - expected_files)}"
        )
    normalizer_record = _strict_json(root / "normalizer.json")
    _verify_record(normalizer_record, "selector normalizer")
    normalizer = {key: value for key, value in normalizer_record.items() if key != "record_digest"}
    if normalizer.get("schema") != NORMALIZER_SCHEMA or normalizer.get("schema_version") != 1:
        raise ValueError("unsupported selector normalizer schema")
    normalizer_digest = content_digest(normalizer)
    scale = normalizer.get("coefficient_scale")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("selector normalizer has an invalid coefficient scale")

    receipt = _strict_json(root / "fit_receipt.json")
    _verify_record(receipt, "selector fit receipt")
    if (
        receipt.get("schema") != SELECTOR_BUNDLE_SCHEMA
        or receipt.get("schema_version") != SELECTOR_BUNDLE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported selector-bundle receipt")
    if receipt.get("terminal_program_mixture") != terminal_program_mixture_registry():
        raise ValueError("selector bundle was not fitted on the frozen terminal-program mixture")
    corpus_digest = _corpus_manifest_digest(corpus)
    if receipt.get("source_corpus_manifest_sha256") != corpus_digest:
        raise ValueError("selector bundle was fitted against a different prepared corpus")
    quality_authority = receipt.get("quality_authority")
    quality_authority = _validate_quality_authority_binding(
        quality_authority,
        expected_role="training_partition",
    )
    target_access = receipt.get("target_access")
    ground_partition_receipt = receipt.get("ground_partition_receipt")
    if not isinstance(target_access, dict) or not isinstance(ground_partition_receipt, dict):
        raise ValueError("selector bundle has no retained train target-access receipts")
    _verify_record(target_access, "selector target access")
    _verify_record(ground_partition_receipt, "selector ground partition")
    partition_authority = quality_authority["training_partition"]
    if not isinstance(partition_authority, dict):
        raise RuntimeError("selector train authority lost its partition record")
    ground_identity = partition_authority.get("ground_partition")
    if (
        target_access.get("record_digest") != partition_authority.get("target_access_record_digest")
        or not isinstance(ground_identity, dict)
        or ground_partition_receipt.get("record_digest")
        != ground_identity.get("receipt_record_digest")
    ):
        raise ValueError("selector target access differs from its quality authority")
    if receipt.get("normalizer_digest") != normalizer_digest:
        raise ValueError("selector receipt and normalizer digest differ")
    selector_path = root / "selector.pt"
    if receipt.get("selector_file_sha256") != _sha256_file(selector_path):
        raise ValueError("selector artifact file checksum mismatch")

    model = StrengthSelectorModel.load(selector_path)
    if not model.frozen or not model.deployment_ready:
        raise ValueError("PPO requires a frozen deployment-ready graph strength selector")
    selector_digest = model.artifact_digest()
    if receipt.get("selector_digest") != selector_digest:
        raise ValueError("selector receipt and model content digest differ")
    if model.normalizer_digest != normalizer_digest:
        raise ValueError("selector model was fitted with a different feature normalizer")
    recorded_strength_scale = receipt.get("strength_transform_scale")
    if (
        isinstance(recorded_strength_scale, bool)
        or not isinstance(recorded_strength_scale, (int, float))
        or not math.isclose(
            float(model.strength_transform_scale.item()),
            float(recorded_strength_scale),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError("selector artifact and receipt strength transforms differ")
    recorded_coefficient_scale = receipt.get("coefficient_transform_scale")
    if (
        isinstance(recorded_coefficient_scale, bool)
        or not isinstance(recorded_coefficient_scale, (int, float))
        or not math.isfinite(recorded_coefficient_scale)
        or recorded_coefficient_scale <= 0.0
    ):
        raise ValueError("selector receipt has an invalid coefficient transform")
    if not math.isclose(
        float(model.coefficient_transform_scale.item()),
        float(recorded_coefficient_scale),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("selector artifact and receipt coefficient transforms differ")
    if not math.isclose(
        float(recorded_coefficient_scale),
        float(scale),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("selector artifact and normalizer coefficient scales differ")
    return SelectorBundle(
        model=model,
        selector_digest=selector_digest,
        normalizer=normalizer,
        normalizer_digest=normalizer_digest,
        coefficient_scale=float(scale),
        corpus_manifest_sha256=corpus_digest,
        quality_authority=dict(quality_authority),
        target_access=dict(target_access),
        ground_partition_receipt=dict(ground_partition_receipt),
        root=root,
    )


def _bind_selector_quality_authority(
    bundle: SelectorBundle,
    prepared: Sequence[PreparedTask],
    *,
    pin: object,
) -> dict[str, object]:
    authority = _quality_authority_identity(prepared, pin=pin)
    selector_authority = _validate_quality_authority_binding(
        bundle.quality_authority,
        expected_role="training_partition",
    )
    current_authority = _validate_quality_authority_binding(authority)
    if selector_authority["global"] != current_authority["global"]:
        raise ValueError("selector bundle belongs to a different pinned quality authority")
    if "training_partition" in current_authority and selector_authority != current_authority:
        raise ValueError("selector bundle belongs to a different train target partition")
    return authority


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate a smoke-only corpus which production commands intentionally reject."""

    from isingfold.rl.data.generate import generate_instances, write_corpus

    instances = generate_instances(
        host_family=args.host,
        host_size=args.host_size,
        n_instances=args.instances,
        n_variables=args.variables,
        chain_size=args.chain_size,
        fault_rate=args.fault_rate,
        alpha=args.alpha,
        weight_choices=tuple(float(weight) for weight in args.weights.split(",")),
        clause_length=args.clause_length,
        seed=args.seed,
    )
    ood = (lambda lineage: lineage.family == "bottleneck") if args.ood_family else None
    manifest = write_corpus(
        instances,
        args.out,
        ctx=Context(qubit_cap=args.qubit_cap, n_est_reads=args.strength_reads),
        strength_reads=args.strength_reads,
        seed=args.seed,
        ood_predicate=ood,
    )
    print(json.dumps({"scope": "development-smoke-only", "manifest": manifest}, indent=1))


def cmd_prepare(args: argparse.Namespace) -> None:
    """Authenticate and partition-seal a CandidateBank-v2 export."""

    manifest = prepare_candidate_bank_v4(
        args.bank,
        args.bank_manifest,
        args.evaluator_targets,
        args.provenance,
        args.corpus_design_manifest,
        args.out,
        expected_corpus_design_sha256=args.expected_corpus_design_sha256,
        qubit_cap=args.qubit_cap,
    )
    print(json.dumps(manifest, indent=1))


def cmd_make_exact_conformance_corpus(args: argparse.Namespace) -> None:
    """Publish the deterministic public-validation corpus used by Gate 1."""

    from isingfold.rl.data.exact_conformance import (
        publish_exact_conformance_registry,
    )

    receipt = publish_exact_conformance_registry(
        args.corpus,
        expected_corpus_manifest_sha256=(args.expected_corpus_manifest_sha256),
        out=args.out,
    )
    print(json.dumps(receipt, indent=1))


def cmd_verify_exact_conformance_corpus(args: argparse.Namespace) -> None:
    """Recompute a pinned Gate 1 corpus without opening evaluator targets."""

    from isingfold.rl.data.exact_conformance import (
        verify_exact_conformance_registry,
    )

    receipt, publication = verify_exact_conformance_registry(
        args.corpus,
        expected_corpus_manifest_sha256=(args.expected_corpus_manifest_sha256),
        registry=args.registry,
        expected_registry_sha256=args.expected_registry_sha256,
        receipt_out=args.receipt_out,
    )
    print(json.dumps(receipt if publication is None else publication, indent=1))


def _initializer_bank_public_inputs(args: argparse.Namespace):
    """Authenticate target-free train inputs and the exact deployment initializer."""

    from isingfold.rl.complete_system import (
        CompleteSystemConfig,
        LACMinorminerInitializerBackend,
    )
    from isingfold.rl.data.prepared import load_prepared_partition
    from isingfold.rl.initializer_bank import lac_runtime_implementation_manifest

    loaded = load_prepared_partition(
        args.corpus,
        partition="train",
        include_evaluator=False,
    )
    if loaded.target_access is not None:
        raise RuntimeError("initializer-bank planning unexpectedly opened evaluator targets")
    config = CompleteSystemConfig.from_mapping(_strict_json(args.config))
    context = _context(args, args.corpus)
    initializer = LACMinorminerInitializerBackend()
    runtime = lac_runtime_implementation_manifest(initializer)
    return (
        loaded.tasks,
        _corpus_manifest_digest(args.corpus),
        initializer,
        runtime,
        config,
        context,
    )


def cmd_plan_initializer_bank(args: argparse.Namespace) -> None:
    """Seal a target-free, lineage-equal deployment-initializer schedule."""

    from isingfold.rl.initializer_bank import (
        build_initializer_bank_plan,
        write_initializer_bank_plan,
    )

    tasks, manifest_sha, initializer, runtime, config, context = _initializer_bank_public_inputs(
        args
    )
    plan = build_initializer_bank_plan(
        tasks,
        prepared_manifest_sha256=manifest_sha,
        training_seed=args.training_seed,
        episode_schedule_start=args.episode_schedule_start,
        episode_count=args.episode_count,
        max_draws_per_conditional_episode=args.max_draws_per_conditional_episode,
        initializer=initializer,
        runtime_implementation_manifest=runtime,
        config=config,
        context=context,
    )
    plan_sha256 = write_initializer_bank_plan(args.out, plan)
    print(
        json.dumps(
            {
                "initializer_bank_plan": str(Path(args.out)),
                "plan_sha256": plan_sha256,
                "plan_record_digest": plan.record_digest,
                "conditional_episode_count": len(plan.episodes),
                "evaluator_targets_opened": False,
            },
            indent=1,
        )
    )


def _load_initializer_bank_plan_inputs(args: argparse.Namespace):
    """Reload one externally pinned initializer-bank plan against live public inputs."""

    from isingfold.rl.initializer_bank import load_initializer_bank_plan

    tasks, manifest_sha, initializer, runtime, config, context = _initializer_bank_public_inputs(
        args
    )
    plan = load_initializer_bank_plan(
        args.plan,
        expected_plan_sha256=args.expected_plan_sha256,
        prepared_tasks=tasks,
        prepared_manifest_sha256=manifest_sha,
        initializer=initializer,
        runtime_implementation_manifest=runtime,
        config=config,
        context=context,
    )
    return plan, tasks, initializer, runtime, config, context


def cmd_generate_initializer_bank_shard(args: argparse.Namespace) -> None:
    """Generate one deterministic, resumable shard of a sealed initializer bank."""

    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("initializer-bank shard index must lie in [0, shard_count)")
    from isingfold.rl.initializer_bank import generate_initializer_bank

    plan, tasks, initializer, runtime, config, context = _load_initializer_bank_plan_inputs(args)
    indices = tuple(
        episode.episode_schedule_index
        for position, episode in enumerate(plan.episodes)
        if position % args.shard_count == args.shard_index
    )
    if not indices:
        raise ValueError("initializer-bank shard has no conditional episodes")
    snapshots = generate_initializer_bank(
        args.bank,
        plan,
        tasks,
        initializer=initializer,
        runtime_implementation_manifest=runtime,
        config=config,
        context=context,
        episode_indices=indices,
    )
    print(
        json.dumps(
            {
                "bank": str(Path(args.bank)),
                "plan_record_digest": plan.record_digest,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "conditional_episode_count": len(indices),
                "snapshot_count": len(snapshots),
                "successful_draws": sum(snapshot.success for snapshot in snapshots),
                "failed_draws": sum(not snapshot.success for snapshot in snapshots),
            },
            indent=1,
        )
    )


def cmd_seal_initializer_bank(args: argparse.Namespace) -> None:
    """Seal a complete initializer bank after all conditional episodes resolve."""

    from isingfold.rl.initializer_bank import seal_initializer_bank

    plan, *_ = _load_initializer_bank_plan_inputs(args)
    manifest_sha256 = seal_initializer_bank(args.bank, plan)
    print(
        json.dumps(
            {
                "bank": str(Path(args.bank)),
                "manifest_sha256": manifest_sha256,
                "plan_record_digest": plan.record_digest,
                "conditional_episode_count": len(plan.episodes),
            },
            indent=1,
        )
    )


def _bootstrap_protocol_from_grid(
    grid: Mapping[str, object], protocol_preset: str
) -> tuple[Mapping[str, object], str, int, int]:
    """Resolve one registered target-free bootstrap population from the staged grid."""

    if protocol_preset == "representation-validation":
        protocol = grid.get("representation_evaluation")
        partition = "val"
    elif protocol_preset == "validation":
        protocol = grid.get("rl_value_evaluation")
        partition = "val"
    elif protocol_preset == "final-test":
        protocol = grid.get("complete_system_evaluation")
        partition = "test"
    else:
        raise ValueError(
            "bootstrap protocol must be representation-validation, validation, or final-test"
        )
    if not isinstance(protocol, Mapping):  # pragma: no cover - guarded by _load_grid
        raise ValueError("staged grid omits the requested bootstrap protocol")
    return (
        protocol,
        partition,
        int(protocol["evaluation_seed"]),
        int(protocol["repetitions"]),
    )


def _bootstrap_bank_public_inputs(args: argparse.Namespace):
    """Authenticate a target-free evaluation census and exact K=2 runtime contract."""

    from isingfold.rl.complete_system import (
        CompleteSystemConfig,
        LACMinorminerInitializerBackend,
    )
    from isingfold.rl.data.prepared import load_prepared_partition
    from isingfold.rl.initializer_bank import lac_runtime_implementation_manifest
    from isingfold.rl.validation_bootstrap_bank import (
        bootstrap_context_digest,
        bootstrap_same_support_contract_digest,
    )

    grid, grid_sha256 = _load_grid(args.grid)
    protocol, partition, evaluation_seed, repetitions = _bootstrap_protocol_from_grid(
        grid, args.protocol_preset
    )
    public = load_prepared_partition(
        args.corpus,
        partition=partition,
        include_evaluator=False,
    )
    if public.target_access is not None:
        raise RuntimeError("bootstrap-bank access unexpectedly opened evaluator targets")
    manifest = _strict_json(Path(args.corpus) / "manifest.json")
    split_registry = _strict_json(Path(args.corpus) / "splits.json")
    public_tasks = _preinitialization_population_tasks(
        public.tasks,
        manifest=manifest,
        partition=partition,
    )
    config = CompleteSystemConfig.from_mapping(_strict_json(args.config))
    context = _context(args, args.corpus)
    initializer = LACMinorminerInitializerBackend()
    runtime = lac_runtime_implementation_manifest(initializer)
    support_digest = bootstrap_same_support_contract_digest(
        partition=partition,
        evaluation_seed=evaluation_seed,
        repetitions=repetitions,
        config_digest=config.digest,
        context_digest=bootstrap_context_digest(context),
    )
    return (
        public.tasks,
        {task.name: task for task in public_tasks},
        manifest,
        split_registry,
        _corpus_manifest_digest(args.corpus),
        grid_sha256,
        stable_digest(protocol),
        support_digest,
        partition,
        evaluation_seed,
        repetitions,
        initializer,
        runtime,
        config,
        context,
    )


def cmd_plan_bootstrap_bank(args: argparse.Namespace) -> None:
    """Seal one validation or final-test bootstrap census before target access."""

    from isingfold.rl.validation_bootstrap_bank import (
        build_bootstrap_plan,
        write_bootstrap_plan,
    )

    (
        prepared,
        _,
        manifest,
        split_registry,
        manifest_sha,
        grid_sha,
        protocol_digest,
        support_digest,
        partition,
        evaluation_seed,
        repetitions,
        initializer,
        runtime,
        config,
        context,
    ) = _bootstrap_bank_public_inputs(args)
    plan = build_bootstrap_plan(
        prepared,
        prepared_manifest=manifest,
        prepared_split_registry=split_registry,
        prepared_manifest_sha256=manifest_sha,
        protocol_registry_sha256=grid_sha,
        protocol_record_digest=protocol_digest,
        same_support_contract_digest=support_digest,
        partition=partition,
        evaluation_seed=evaluation_seed,
        repetitions=repetitions,
        initializer=initializer,
        runtime_implementation_manifest=runtime,
        config=config,
        context=context,
    )
    plan_sha256 = write_bootstrap_plan(args.out, plan)
    print(
        json.dumps(
            {
                "bootstrap_bank_plan": str(Path(args.out)),
                "protocol_preset": plan.protocol_preset,
                "plan_sha256": plan_sha256,
                "plan_record_digest": plan.record_digest,
                "same_support_contract_digest": plan.same_support_contract_digest,
                "denominator_count": len(plan.census),
                "evaluator_targets_opened": False,
            },
            indent=1,
        )
    )


def _load_bootstrap_bank_plan_inputs(args: argparse.Namespace):
    from isingfold.rl.validation_bootstrap_bank import load_bootstrap_plan

    inputs = _bootstrap_bank_public_inputs(args)
    (
        prepared,
        public_tasks,
        manifest,
        split_registry,
        manifest_sha,
        grid_sha,
        protocol_digest,
        support_digest,
        partition,
        evaluation_seed,
        repetitions,
        initializer,
        runtime,
        config,
        context,
    ) = inputs
    plan = load_bootstrap_plan(
        args.plan,
        expected_plan_sha256=args.expected_plan_sha256,
        prepared_tasks=prepared,
        prepared_manifest=manifest,
        prepared_split_registry=split_registry,
        prepared_manifest_sha256=manifest_sha,
        protocol_registry_sha256=grid_sha,
        protocol_record_digest=protocol_digest,
        same_support_contract_digest=support_digest,
        partition=partition,
        evaluation_seed=evaluation_seed,
        repetitions=repetitions,
        initializer=initializer,
        runtime_implementation_manifest=runtime,
        config=config,
        context=context,
    )
    return plan, public_tasks, initializer, config, context


def cmd_generate_bootstrap_bank_shard(args: argparse.Namespace) -> None:
    """Execute a disjoint deterministic shard of the target-free bootstrap census."""

    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("bootstrap-bank shard index must lie in [0, shard_count)")
    from isingfold.rl.validation_bootstrap_bank import (
        bootstrap_record_from_outcome,
        execute_bootstrap_row,
        publish_bootstrap_record,
    )

    plan, public_tasks, initializer, config, context = _load_bootstrap_bank_plan_inputs(args)
    selected = tuple(
        row for row in plan.census if row.census_index % args.shard_count == args.shard_index
    )
    if not selected:
        raise ValueError("bootstrap-bank shard has no census rows")
    success = 0
    for row in selected:
        try:
            task = public_tasks[row.instance_id]
        except KeyError as exc:  # pragma: no cover - plan authentication guards this
            raise RuntimeError("bootstrap plan names an unknown public task") from exc
        outcome = execute_bootstrap_row(
            plan,
            row.row_key,
            task,
            initializer=initializer,
            config=config,
            context=context,
        )
        record = bootstrap_record_from_outcome(plan, row.row_key, outcome)
        publish_bootstrap_record(args.bank, plan, record)
        success += int(record.actor_invocation_permitted)
    print(
        json.dumps(
            {
                "bank": str(Path(args.bank)),
                "protocol_preset": plan.protocol_preset,
                "plan_record_digest": plan.record_digest,
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "row_count": len(selected),
                "initial_success_count": success,
                "initial_failure_count": len(selected) - success,
                "evaluator_targets_opened": False,
            },
            indent=1,
        )
    )


def cmd_seal_bootstrap_bank(args: argparse.Namespace) -> None:
    """Seal a complete target-free bootstrap bank and print its external pin."""

    from isingfold.rl.validation_bootstrap_bank import seal_bootstrap_bank

    plan, *_ = _load_bootstrap_bank_plan_inputs(args)
    manifest_sha256 = seal_bootstrap_bank(args.bank, plan)
    print(
        json.dumps(
            {
                "bank": str(Path(args.bank)),
                "protocol_preset": plan.protocol_preset,
                "manifest_sha256": manifest_sha256,
                "plan_record_digest": plan.record_digest,
                "denominator_count": len(plan.census),
            },
            indent=1,
        )
    )


def cmd_verify_ground_certificates(args: argparse.Namespace) -> None:
    """Execute the externally pinned verifier over every authenticated ground target."""

    from isingfold.rl.data.ground_certificate import (
        GroundCertificateVerifierPin,
        verify_ground_certificate_partitions,
    )

    verifier_pin = GroundCertificateVerifierPin(
        executable_path=args.verifier_executable,
        expected_executable_sha256=args.expected_verifier_executable_sha256,
        source_path=args.verifier_source,
        expected_source_sha256=args.expected_verifier_source_sha256,
        environment_path=args.verifier_environment,
        expected_environment_sha256=args.expected_verifier_environment_sha256,
        expected_name=args.expected_verifier_name,
        expected_version=args.expected_verifier_version,
        execution_mode=args.verifier_execution_mode,
        runtime_path=args.verifier_runtime,
        expected_runtime_sha256=args.expected_verifier_runtime_sha256,
        build_attestation_path=args.verifier_build_attestation,
        expected_build_attestation_sha256=(args.expected_verifier_build_attestation_sha256),
    )
    receipt = verify_ground_certificate_partitions(
        args.corpus,
        quality_attestation_pin=_publisher_attestation_pin(args),
        verifier_pin=verifier_pin,
        output_directory=args.out,
        timeout_seconds=args.verifier_timeout_seconds,
    )
    print(json.dumps(receipt, indent=1))


def cmd_prepare_release_v1(args: argparse.Namespace) -> None:
    """Authenticate the exact-compatible legacy release and publish production artifacts."""

    from isingfold.rl.data.import_release_v1 import prepare_release_v1

    manifest = prepare_release_v1(
        args.quality_corpus,
        args.split_manifest,
        args.checksums,
        args.problem_references,
        args.out,
        qubit_cap=args.qubit_cap,
    )
    print(json.dumps(manifest, indent=1))


def cmd_label_selector_data(args: argparse.Namespace) -> None:
    """Publish fit labels or an explicitly authorized post-freeze audit label set."""

    from isingfold.rl.data.selector_labels import build_selector_labels

    expected_normalizer_digest = None
    audit_authorization = None
    audit_bundle: SelectorBundle | None = None
    protected_inputs = (
        getattr(args, "selector", None),
        getattr(args, "grid", None),
        getattr(args, "rl_value_selection_receipt", None),
        getattr(args, "expected_rl_value_selection_sha256", None),
    )
    if args.audit_mode:
        if not all(protected_inputs):
            raise ValueError(
                "production audit labels require --selector, --grid, and an authenticated "
                "externally pinned RL-value-selection receipt"
            )
        audit_args = argparse.Namespace(
            partition="audit_test",
            grid=args.grid,
            rl_value_selection_receipt=args.rl_value_selection_receipt,
            expected_rl_value_selection_sha256=(args.expected_rl_value_selection_sha256),
        )
        audit_authorization, _ = _selector_audit_access(audit_args)
        audit_bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
        _bind_selector_audit_freeze(
            args.rl_value_selection_receipt,
            bundle=audit_bundle,
        )
        expected_normalizer_digest = audit_bundle.normalizer_digest
    elif any(value is not None for value in protected_inputs):
        raise ValueError(
            "--selector, --grid, and RL-value freeze arguments are valid only with --audit-mode"
        )
    quality_pin = _quality_attestation_pin(args)
    train_tasks, train_authority, _, _ = _load_quality_partition(
        args.corpus,
        partition="train",
        pin=quality_pin,
        role="training_partition",
    )
    selector_tasks: tuple[PreparedTask, ...] = train_tasks
    selector_authority: dict[str, object] = train_authority
    if args.audit_mode:
        val_tasks, val_authority, _, _ = _load_quality_partition(
            args.corpus,
            partition="val",
            pin=quality_pin,
            role="evaluation_partition",
        )
        test_tasks, test_authority, _, _ = _load_quality_partition(
            args.corpus,
            partition="test",
            pin=quality_pin,
            role="evaluation_partition",
        )
        if not (train_authority["global"] == val_authority["global"] == test_authority["global"]):
            raise ValueError("selector audit partitions have different global authorities")
        audit_payload = {
            "schema": "isingfold.quality-authority-binding",
            "schema_version": 2,
            "global": train_authority["global"],
            "training_partition": train_authority["training_partition"],
            "audit_partitions": {
                "val": val_authority["evaluation_partition"],
                "test": test_authority["evaluation_partition"],
            },
        }
        selector_authority = {
            **audit_payload,
            "record_digest": content_digest(audit_payload),
        }
        selector_tasks = (*train_tasks, *val_tasks, *test_tasks)
    ctx = _context(args, args.corpus)
    manifest = build_selector_labels(
        args.corpus,
        args.out,
        prepared_tasks=selector_tasks,
        quality_authority=selector_authority,
        context=ctx,
        sample_seed=args.seed,
        split_seed=args.split_seed,
        calibration_fraction=args.calibration_fraction,
        audit_mode=args.audit_mode,
        audit_authorization=audit_authorization,
        expected_normalizer_digest=expected_normalizer_digest,
    )
    if audit_bundle is not None:
        audit_manifest_authority = manifest["quality_authority"]
        if (
            audit_bundle.quality_authority["global"] != audit_manifest_authority["global"]
            or audit_bundle.quality_authority["training_partition"]
            != audit_manifest_authority["training_partition"]
        ):
            raise ValueError("selector audit labels and frozen selector use different authorities")
    print(
        json.dumps(
            {
                "selector_labels": str(Path(args.out)),
                "manifest_record_digest": manifest["record_digest"],
                "partitions": manifest["partitions"],
                "audit_mode": manifest["audit_mode"],
            },
            indent=1,
        )
    )


def cmd_fit_selector(args: argparse.Namespace) -> None:
    """Fit and atomically publish IF-Q3-S0 from authenticated offline labels."""

    from isingfold.rl.data.selector_labels import (
        load_selector_metadata,
        load_selector_records,
    )
    from isingfold.rl.strength import fit_selector, selector_regret

    quality_pin = _quality_attestation_pin(args)
    (
        authority_tasks,
        quality_authority,
        target_access,
        ground_partition_receipt,
    ) = _load_quality_partition(
        args.corpus,
        partition="train",
        pin=quality_pin,
        role="training_partition",
    )
    ctx = _context(args, args.corpus)
    metadata = load_selector_metadata(args.selector_labels)
    corpus_digest = _corpus_manifest_digest(args.corpus)
    if metadata.source_prepared_manifest_sha256 != corpus_digest:
        raise ValueError("selector labels belong to a different prepared corpus")
    if dict(metadata.quality_authority) != quality_authority:
        raise ValueError("selector labels belong to a different pinned quality authority")
    if metadata.context_digest != content_digest(_context_snapshot(ctx)):
        raise ValueError("selector-label context differs from the requested production context")
    if metadata.reads_per_strength != ctx.n_est_reads:
        raise ValueError("selector-label read count differs from the requested production context")
    fit_records = load_selector_records(args.selector_labels, partition="train")
    calibration_records = load_selector_records(args.selector_labels, partition="calibration")
    normalizer_record = _strict_json(Path(args.selector_labels) / "normalizer.json")
    _verify_record(normalizer_record, "selector-label normalizer")
    normalizer = {key: value for key, value in normalizer_record.items() if key != "record_digest"}
    if (
        normalizer.get("schema") != NORMALIZER_SCHEMA
        or normalizer.get("schema_version") != CLI_SCHEMA_VERSION
        or content_digest(normalizer) != metadata.normalizer_digest
    ):
        raise ValueError("selector-label normalizer identity differs from its manifest")
    coefficient_scale = metadata.coefficient_scale
    strength_transform_scale = coefficient_scale
    device = _resolve_device(args.device)
    _seed_runtime(args.seed, deterministic=args.deterministic, threads=args.threads)
    fit_history: list[dict[str, object]] = []
    model = fit_selector(
        fit_records,
        calibration_records=calibration_records,
        epochs=args.epochs,
        graph_learning_rate=args.learning_rate,
        l2=args.weight_decay,
        seed=args.seed,
        strength_transform_scale=strength_transform_scale,
        coefficient_transform_scale=coefficient_scale,
        normalizer_digest=metadata.normalizer_digest,
        device=device,
        graph_minibatch=args.graph_minibatch,
        gradient_norm=1.0,
        history=fit_history,
    )
    if not model.deployment_ready:
        raise RuntimeError("graph selector fitting did not produce a deployment-ready artifact")

    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"selector bundle already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        model.save(temporary / "selector.pt")
        _atomic_json(temporary / "normalizer.json", _with_digest(normalizer))
        receipt_payload = {
            "schema": SELECTOR_BUNDLE_SCHEMA,
            "schema_version": SELECTOR_BUNDLE_SCHEMA_VERSION,
            "source_corpus_manifest_sha256": corpus_digest,
            "quality_authority": quality_authority,
            "target_access": target_access,
            "ground_partition_receipt": ground_partition_receipt,
            "source_selector_labels_dataset_version": metadata.dataset_version,
            "source_selector_labels_manifest_sha256": metadata.manifest_sha256,
            "source_selector_labels_manifest_record_digest": (metadata.manifest_record_digest),
            "terminal_program_mixture": dict(metadata.terminal_program_mixture),
            "selector_label_context_digest": metadata.context_digest,
            "selector_digest": model.artifact_digest(),
            "selector_file_sha256": _sha256_file(temporary / "selector.pt"),
            "normalizer_digest": metadata.normalizer_digest,
            "strength_transform_scale": strength_transform_scale,
            "coefficient_transform_scale": coefficient_scale,
            "fit_lineages": sorted({record.lineage for record in fit_records}),
            "calibration_lineages": sorted({record.lineage for record in calibration_records}),
            "label_protocol": {
                "reads_per_strength": metadata.reads_per_strength,
                "sweeps": ctx.num_sweeps,
                "four_strength_ratios": list(ctx.strength_ratios),
            },
            "optimizer": {
                "name": "AdamW",
                "device_type": next(model.parameters()).device.type,
                "training_seed": args.seed,
                "epochs": args.epochs,
                "graph_minibatch_records": args.graph_minibatch,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "gradient_norm": 1.0,
                "optimizer_steps": sum(int(row["optimizer_steps"]) for row in fit_history),
            },
            "training_history": fit_history,
            "fit_metrics": selector_regret(model, fit_records),
            "calibration_metrics": selector_regret(model, calibration_records),
        }
        _atomic_json(temporary / "fit_receipt.json", _with_digest(receipt_payload))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(
        json.dumps(
            {
                "selector_bundle": str(destination),
                "selector_digest": model.artifact_digest(),
                "fit_records": len(fit_records),
                "calibration_records": len(calibration_records),
                "device": str(device),
            },
            indent=1,
        )
    )


def _selector_audit_access(args: argparse.Namespace):
    """Authenticate the final validation freeze before returning an audit-test capability."""

    from isingfold.rl.data.selector_labels import SelectorAuditAuthorization
    from isingfold.rl.experiment_selection import load_rl_value_freeze

    grid_path = getattr(args, "grid", None)
    selection_path = getattr(args, "rl_value_selection_receipt", None)
    expected_sha256 = getattr(args, "expected_rl_value_selection_sha256", None)
    if args.partition == "audit_test":
        if not grid_path or not selection_path or not expected_sha256:
            raise ValueError(
                "audit_test requires --grid, --rl-value-selection-receipt, and its "
                "externally pinned --expected-rl-value-selection-sha256"
            )
        selection = load_rl_value_freeze(
            receipt_path=selection_path,
            grid_path=grid_path,
            expected_sha256=expected_sha256,
        )
        authorization = SelectorAuditAuthorization(
            grid_manifest_sha256=selection.grid_manifest_sha256,
            rl_value_selection_receipt_sha256=selection.receipt_sha256,
            rl_value_selection_record_digest=selection.record_digest,
            representation_selection_receipt_sha256=(selection.representation_selection_sha256),
            representation_selection_record_digest=(
                selection.representation_selection_record_digest
            ),
            selected_model_family=selection.model_family,
            selected_method=selection.method,
        ).validate()
        return authorization, {
            "rl_value_selection_required": True,
            "grid_manifest_sha256": selection.grid_manifest_sha256,
            "rl_value_selection_receipt_sha256": selection.receipt_sha256,
            "rl_value_selection_record_digest": selection.record_digest,
            "representation_selection_receipt_sha256": (selection.representation_selection_sha256),
            "representation_selection_record_digest": (
                selection.representation_selection_record_digest
            ),
            "selected_model_family": selection.model_family,
            "selected_method": selection.method,
        }
    if any(value is not None for value in (grid_path, selection_path, expected_sha256)):
        raise ValueError("RL-value freeze arguments are reserved for audit_test")
    return None, {
        "rl_value_selection_required": False,
        "grid_manifest_sha256": None,
        "rl_value_selection_receipt_sha256": None,
        "rl_value_selection_record_digest": None,
        "representation_selection_receipt_sha256": None,
        "representation_selection_record_digest": None,
        "selected_model_family": None,
        "selected_method": None,
    }


def _bind_selector_audit_freeze(
    receipt_path: str | os.PathLike[str],
    *,
    bundle: SelectorBundle,
) -> None:
    """Require the final unsealing decision to concern this exact corpus and selector."""

    receipt = _strict_json(receipt_path)
    _verify_record(receipt, "RL-value-selection receipt")
    protocol = receipt.get("matched_validation_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("RL-value-selection receipt has no validation protocol identity")
    if protocol.get("selector_digest") != bundle.selector_digest:
        raise ValueError("RL-value-selection receipt belongs to a different frozen selector")
    if protocol.get("corpus_manifest_sha256") != bundle.corpus_manifest_sha256:
        raise ValueError("RL-value-selection receipt belongs to a different prepared corpus")
    matched = protocol.get("matched_metadata")
    if matched is not None and (
        not isinstance(matched, Mapping) or matched.get("selector") != bundle.selector_digest
    ):
        raise ValueError("RL-value-selection evaluation protocol names a different frozen selector")


def cmd_audit_selector(args: argparse.Namespace) -> None:
    """Audit a frozen selector without exposing held-out rows to fitting or calibration."""

    # This check deliberately precedes every access to selector-label metadata or records.
    authorization, access_control = _selector_audit_access(args)
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"selector-audit output already exists: {destination}")

    from isingfold.rl import selector_audit
    from isingfold.rl.data.selector_labels import (
        load_selector_metadata,
        load_selector_records,
        validate_selector_quality_authority,
    )

    _seed_runtime(0, deterministic=True, threads=1)
    bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
    if authorization is not None:
        _bind_selector_audit_freeze(
            args.rl_value_selection_receipt,
            bundle=bundle,
        )
    metadata = load_selector_metadata(args.selector_labels, audit_mode=True)
    if metadata.source_prepared_manifest_sha256 != bundle.corpus_manifest_sha256:
        raise ValueError("selector audit labels belong to a different prepared corpus")
    audit_authority = validate_selector_quality_authority(
        metadata.quality_authority,
        audit_mode=True,
    )
    selector_authority = _validate_quality_authority_binding(
        bundle.quality_authority,
        expected_role="training_partition",
    )
    if (
        audit_authority["global"] != selector_authority["global"]
        or audit_authority["training_partition"] != selector_authority["training_partition"]
    ):
        raise ValueError("selector audit labels belong to a different quality authority")
    if metadata.normalizer_digest != bundle.normalizer_digest:
        raise ValueError("selector audit labels use a different frozen normalizer")

    fit_receipt_path = bundle.root / "fit_receipt.json"
    fit_receipt = _strict_json(fit_receipt_path)
    _verify_record(fit_receipt, "selector fit receipt")
    if fit_receipt.get("selector_label_context_digest") != metadata.context_digest:
        raise ValueError("selector fit and audit labels use different objective contexts")
    if fit_receipt.get("normalizer_digest") != metadata.normalizer_digest:
        raise ValueError("selector fit and audit labels use different preprocessing")
    if fit_receipt.get("selector_digest") != bundle.selector_digest:
        raise ValueError("selector fit receipt differs from the loaded frozen selector")
    fit_labels_sha256 = fit_receipt.get("source_selector_labels_manifest_sha256")
    fit_labels_record_digest = fit_receipt.get("source_selector_labels_manifest_record_digest")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in (fit_labels_sha256, fit_labels_record_digest)
    ):
        raise ValueError("selector fit receipt has invalid fitting-label provenance")

    partition_receipt = metadata.partitions.get(args.partition)
    if not isinstance(partition_receipt, Mapping):
        raise ValueError("selector audit partition has no authenticated census")
    expected_records = partition_receipt.get("records")
    expected_lineages = partition_receipt.get("lineages")
    if (
        isinstance(expected_records, bool)
        or not isinstance(expected_records, int)
        or expected_records <= 0
        or not isinstance(expected_lineages, list)
        or not expected_lineages
        or any(not isinstance(lineage, str) or not lineage for lineage in expected_lineages)
    ):
        raise ValueError("selector audit partition census is empty or malformed")
    fit_lineages = fit_receipt.get("fit_lineages")
    calibration_lineages = fit_receipt.get("calibration_lineages")
    if any(
        not isinstance(values, list)
        or not values
        or any(not isinstance(lineage, str) or not lineage for lineage in values)
        for values in (fit_lineages, calibration_lineages)
    ):
        raise ValueError("selector fit receipt has malformed lineage provenance")
    if set(fit_lineages) & set(calibration_lineages):
        raise ValueError("selector fitting and calibration lineages overlap")
    if (set(fit_lineages) | set(calibration_lineages)) & set(expected_lineages):
        raise ValueError("selector audit lineages leaked into fitting or calibration")

    records = load_selector_records(
        args.selector_labels,
        partition=args.partition,
        audit_mode=True,
        audit_authorization=authorization,
    )
    if len(records) != expected_records or sorted({record.lineage for record in records}) != sorted(
        expected_lineages
    ):
        raise ValueError("loaded selector audit rows differ from the authenticated census")
    rows, aggregate = selector_audit.audit_selector_records(
        bundle.model,
        records,
        partition=args.partition,
    )
    provenance = {
        "source_corpus_manifest_sha256": bundle.corpus_manifest_sha256,
        "selector_digest": bundle.selector_digest,
        "selector_file_sha256": _sha256_file(bundle.root / "selector.pt"),
        "selector_fit_receipt_sha256": _sha256_file(fit_receipt_path),
        "selector_fit_record_digest": fit_receipt["record_digest"],
        "selector_fit_labels_manifest_sha256": fit_receipt.get(
            "source_selector_labels_manifest_sha256"
        ),
        "selector_fit_labels_manifest_record_digest": fit_labels_record_digest,
        "audit_labels_dataset_version": metadata.dataset_version,
        "audit_labels_manifest_sha256": metadata.manifest_sha256,
        "audit_labels_manifest_record_digest": metadata.manifest_record_digest,
        "audit_labels_context_digest": metadata.context_digest,
        "normalizer_digest": metadata.normalizer_digest,
        "reads_per_strength": metadata.reads_per_strength,
        "partition_records": expected_records,
        "partition_lineages": sorted(expected_lineages),
        "inference_protocol": {
            "device_type": "cpu",
            "deterministic_algorithms": True,
            "threads": 1,
            "forward_passes_per_record": 1,
            "tie_break": "lowest-strength-index",
        },
        "runtime_platform": _runtime_platform_identity(),
        "implementation_sha256": _sha256_file(selector_audit.__file__),
    }
    receipt = selector_audit.publish_selector_audit(
        destination,
        rows=rows,
        aggregate=aggregate,
        provenance=provenance,
        access_control=access_control,
    )
    print(
        json.dumps(
            {
                "selector_audit": str(destination),
                "partition": args.partition,
                "records": aggregate["records"],
                "lineages": aggregate["lineages"],
                "selected_regret_mean": aggregate["selected_regret_mean"],
                "top1_rate": aggregate["top1_rate"],
                "receipt_record_digest": receipt["record_digest"],
            },
            indent=1,
        )
    )


def cmd_audit_qpsi_ordering(args: argparse.Namespace) -> None:
    """Publish a train-only diagnostic of q_psi ordering across terminal embeddings."""

    if args.allow_legacy_pilot:
        raise ValueError("qpsi ordering audit requires provenance-complete production artifacts")
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"qpsi-audit output already exists: {destination}")

    from isingfold.rl import evaluator, program, qpsi_audit, strength, strength_tensorize
    from isingfold.rl.checkpoint import runtime_implementation_registry

    _seed_runtime(args.bootstrap_seed, deterministic=True, threads=1)
    ctx = _context(args, args.corpus)
    bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
    source_quality_manifest = _strict_json(Path(args.quality_labels) / "manifest.json")
    initializer_bank, initializer_bank_manifest_sha256 = (
        _quality_initializer_bank_for_manifest(
            args,
            manifest=source_quality_manifest,
            context=ctx,
        )
    )
    bundle.model.to(_resolve_device("cpu")).eval()
    quality_pin = _quality_attestation_pin(args)
    prepared, opened_authority, target_access, ground_partition = _load_quality_partition(
        args.corpus,
        partition="train",
        pin=quality_pin,
        role="training_partition",
    )
    bound_authority = _bind_selector_quality_authority(
        bundle,
        prepared,
        pin=quality_pin,
    )
    if opened_authority != bound_authority:
        raise ValueError("opened train targets and frozen selector use different authorities")

    preflight = _load_quality_preflight_receipt(
        args.quality_preflight_receipt,
        expected_sha256=args.expected_quality_preflight_sha256,
        quality_labels=args.quality_labels,
        corpus=args.corpus,
        selector=bundle,
        context=ctx,
        min_resolved_rows=args.min_resolved_rows,
        min_resolved_lineages=args.min_resolved_lineages,
    )
    _decoded, loaded_manifest = _load_quality_labels(
        args.quality_labels,
        corpus=args.corpus,
        selector=bundle,
        context=ctx,
        min_resolved_rows=args.min_resolved_rows,
        min_resolved_lineages=args.min_resolved_lineages,
        enforce_resolution=True,
        require_provenance=not args.allow_legacy_pilot,
        quality_attestation_pin=quality_pin,
        trusted_preflight=preflight,
        _prepared_train_tasks=prepared,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=(initializer_bank_manifest_sha256),
    )
    quality_root = Path(args.quality_labels)
    manifest_path = quality_root / "manifest.json"
    records_path = quality_root / "records.jsonl"
    manifest = _strict_json(manifest_path)
    _verify_record(manifest, "qpsi source quality manifest")
    if manifest.get("quality_authority") != opened_authority:
        raise ValueError("quality source and opened train targets use different authorities")
    if loaded_manifest.get("record_digest") != manifest.get("record_digest"):
        raise ValueError("authenticated quality manifest changed before qpsi audit")
    raw_rows = _quality_rows_from_artifact(quality_root, manifest)
    rows, aggregate = qpsi_audit.audit_qpsi_ordering(
        bundle.model,
        raw_rows,
        prepared,
        ctx,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )

    fit_path = bundle.root / "fit_receipt.json"
    fit_receipt = _strict_json(fit_path)
    _verify_record(fit_receipt, "selector fit receipt")
    label_protocol = manifest.get("label_protocol")
    if not isinstance(label_protocol, Mapping) or not isinstance(
        label_protocol.get("implementation_contract"), Mapping
    ):
        raise ValueError("quality source has no executable implementation contract")
    task_by_id = {item.task_id: item for item in prepared}
    task_protocols: list[dict[str, str]] = []
    for task_id in sorted({str(row["task_id"]) for row in rows}):
        evaluator_digest = task_by_id[task_id].evaluator_protocol_digest
        if not _is_lower_sha256(evaluator_digest):
            raise ValueError("qpsi task has no authenticated evaluator protocol digest")
        task_protocols.append({"task_id": task_id, "evaluator_protocol_digest": evaluator_digest})
    implementation_sources: dict[str, str] = {}
    for name, module in {
        "evaluator": evaluator,
        "program": program,
        "qpsi_audit": qpsi_audit,
        "selector": strength,
        "selector_tensorizer": strength_tensorize,
    }.items():
        source = getattr(module, "__file__", None)
        if not isinstance(source, str) or not Path(source).is_file():
            raise RuntimeError(f"cannot identify {name} implementation source")
        implementation_sources[name] = _sha256_file(source)
    runtime = runtime_implementation_registry()
    quality_contract = dict(label_protocol["implementation_contract"])
    provenance = {
        "partition": "train",
        "sealed_validation_or_test_opened": False,
        "source_corpus_manifest_sha256": bundle.corpus_manifest_sha256,
        "source_quality_manifest_sha256": _sha256_file(manifest_path),
        "source_quality_manifest_record_digest": manifest["record_digest"],
        "source_quality_records_sha256": _sha256_file(records_path),
        "quality_preflight_receipt_sha256": _sha256_file(args.quality_preflight_receipt),
        "quality_preflight_record_digest": preflight["record_digest"],
        "selector_digest": bundle.selector_digest,
        "selector_file_sha256": _sha256_file(bundle.root / "selector.pt"),
        "selector_fit_receipt_sha256": _sha256_file(fit_path),
        "selector_fit_record_digest": fit_receipt["record_digest"],
        "normalizer_digest": bundle.normalizer_digest,
        "quality_authority": opened_authority,
        "target_access": target_access,
        "ground_partition_receipt": ground_partition,
        "quality_implementation_contract": quality_contract,
        "quality_implementation_contract_digest": content_digest(quality_contract),
        "task_evaluator_protocols": task_protocols,
        "task_evaluator_protocols_digest": content_digest(task_protocols),
        "runtime_implementation_registry": runtime,
        "runtime_implementation_digest": content_digest(runtime),
        "implementation_sources": implementation_sources,
    }
    receipt = qpsi_audit.publish_qpsi_ordering_audit(
        destination,
        rows=rows,
        aggregate=aggregate,
        provenance=provenance,
    )
    primary = aggregate["equal_lineage_then_equal_instance"]
    print(
        json.dumps(
            {
                "qpsi_audit": str(destination),
                "scope": "DIAGNOSTIC_ONLY",
                "comparable_opportunities": aggregate["comparable_opportunities"],
                "lineages": aggregate["lineages"],
                "spearman_rho": primary["spearman_rho"],
                "pairwise_sign_accuracy": primary["pairwise_sign_accuracy"],
                "qpsi_top1_regret": primary["qpsi_top1_regret"],
                "qpsi_top_minus_last": primary["qpsi_top_minus_last"],
                "archive_policy_authorized": False,
                "receipt_record_digest": receipt["record_digest"],
            },
            indent=1,
        )
    )


def _quality_record_payload(record, item: PreparedTask) -> dict[str, object]:
    """Publish one independently authenticated v7 row with its prepared identities."""

    payload = {
        "schema": QUALITY_ROW_SCHEMA,
        "schema_version": QUALITY_ROW_SCHEMA_VERSION,
        "task_id": record.instance,
        "instance_id": item.instance_id,
        "lineage": record.lineage,
        "partition": item.partition,
        "initializer_record_digest": item.initializer_record_digest,
        "reference_status": item.reference_status,
        "certificate_digest": item.certificate_digest,
        "evaluator_protocol_digest": item.evaluator_protocol_digest,
        "environment_seed": record.environment_seed,
        "prefix": list(record.prefix),
        "support": list(record.support),
        "evaluated": [dataclasses.asdict(action) for action in record.evaluated],
        "best_actions": list(record.best_actions()),
        "continuation_policy": record.continuation_policy,
        "action_envelope_record_digest": record.action_envelope_record_digest,
        "state_fingerprint": record.state_fingerprint,
        "support_fingerprint": record.support_fingerprint,
        "context_version": record.context_version,
        "context_digest": record.context_digest,
        "charged_work_receipt": record.charged_work_receipt,
        "exact_state": record.exact_state,
        "observation": record.observation,
        "label_version": record.label_version,
    }
    return _with_digest(payload)


def _quality_action_provenance_fingerprint(
    item: PreparedTask,
    *,
    source_corpus_manifest_sha256: str,
) -> str:
    """Bind one state/action envelope to its authenticated prepared-task authority."""

    return content_digest(
        {
            "schema": "isingfold.quality-action-provenance",
            "schema_version": 1,
            "source_corpus_manifest_sha256": source_corpus_manifest_sha256,
            "task_id": item.task_id,
            "instance_id": item.instance_id,
            "partition": item.partition,
            "initializer_record_digest": item.initializer_record_digest,
            "certificate_digest": item.certificate_digest,
            "evaluator_protocol_digest": item.evaluator_protocol_digest,
        }
    )


def _quality_prefixes(
    item: PreparedTask,
    ctx: Context,
    selector: object,
    *,
    count: int,
    seed: int,
) -> tuple[tuple[int, ...], ...]:
    """Collect real replayable prefixes from one frozen environment trajectory."""

    from isingfold.rl.contracts import DecisionState
    from isingfold.rl.env import EmbeddingEnv
    from isingfold.rl.proposal import LEGACY_ONLINE_INITIALIZER_RESTARTS_V1

    if isinstance(count, bool) or count <= 0:
        raise ValueError("states per quality instance must be positive")
    environment = EmbeddingEnv(
        item.task,
        ctx,
        mode=Mode.IMPROVEMENT,
        initializer=item.initializer(),
        selector=selector,
        reward_reads=ctx.n_est_reads,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=seed,
    )
    result = environment.reset(seed)
    prefix: list[int] = []
    prefixes: list[tuple[int, ...]] = []
    rng = np.random.default_rng(_domain_seed(seed, "quality-prefix-policy", item.task_id))
    while isinstance(result, DecisionState) and len(prefixes) < count:
        prefixes.append(tuple(prefix))
        continuing = [
            index
            for index, (candidate, legal) in enumerate(
                zip(result.candidates, result.legal_mask, strict=True)
            )
            if legal and not candidate.opcode.is_terminal
        ]
        if not continuing:
            break
        chosen = int(rng.choice(continuing))
        prefix.append(chosen)
        result = environment.step(
            result, chosen, evaluate_training_reward=False
        ).next_decision_or_terminal
    return tuple(prefixes)


def _quality_sampling_plan(
    tasks: Sequence[PreparedTask],
    *,
    lineage_count: int,
    tasks_per_lineage: int,
    seed: int,
) -> tuple[tuple[PreparedTask, ...], ...]:
    """Choose independent lineages first, then cap tasks within each lineage.

    Input ordering never changes the sample.  ``lineage_count=0`` means every lineage in the
    authenticated partition; a positive request is exact and fails instead of overstating the
    independent denominator when the partition is smaller.
    """

    if isinstance(lineage_count, bool) or not isinstance(lineage_count, int) or lineage_count < 0:
        raise ValueError("quality instance count must be a nonnegative lineage count")
    if (
        isinstance(tasks_per_lineage, bool)
        or not isinstance(tasks_per_lineage, int)
        or tasks_per_lineage <= 0
    ):
        raise ValueError("tasks per lineage must be a positive integer")
    grouped: dict[str, list[PreparedTask]] = {}
    for item in tasks:
        lineage = item.task.lineage
        if not isinstance(lineage, str) or not lineage:
            raise ValueError("prepared quality task has no lineage identity")
        grouped.setdefault(lineage, []).append(item)
    ordered_lineages = sorted(
        grouped,
        key=lambda lineage: (
            _domain_seed(seed, "quality-lineage-sample", lineage),
            lineage,
        ),
    )
    if lineage_count > len(ordered_lineages):
        raise ValueError(
            f"requested {lineage_count} independent lineages but the partition has only "
            f"{len(ordered_lineages)}"
        )
    if lineage_count:
        ordered_lineages = ordered_lineages[:lineage_count]
    selected: list[tuple[PreparedTask, ...]] = []
    for lineage in ordered_lineages:
        ordered_tasks = sorted(
            grouped[lineage],
            key=lambda item: (
                _domain_seed(seed, "quality-task-sample", lineage, item.task_id),
                item.task_id,
            ),
        )
        selected.append(tuple(ordered_tasks[:tasks_per_lineage]))
    return tuple(selected)


def _quality_plan_shard(
    plan: Sequence[tuple[PreparedTask, ...]],
    *,
    shard_index: int,
    shard_count: int,
) -> tuple[tuple[PreparedTask, ...], ...]:
    """Assign whole independent lineages to one deterministic round-robin shard."""

    if isinstance(shard_count, bool) or not isinstance(shard_count, int) or shard_count <= 0:
        raise ValueError("quality shard count must be a positive integer")
    if shard_count > len(plan):
        raise ValueError("quality shard count must be smaller than or equal to selected lineages")
    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or shard_index < 0
        or shard_index >= shard_count
    ):
        raise ValueError(
            "quality shard index must be nonnegative and strictly less than shard count"
        )
    return tuple(plan[shard_index::shard_count])


def _quality_active_groups(
    plan: Sequence[tuple[PreparedTask, ...]], states_per_lineage: int
) -> tuple[tuple[PreparedTask, ...], ...]:
    return tuple(group[: min(len(group), states_per_lineage)] for group in plan)


def _quality_sampling_receipt(
    tasks: Sequence[PreparedTask],
    plan: Sequence[tuple[PreparedTask, ...]],
    *,
    requested_lineages: int,
    tasks_per_lineage: int,
    states_per_lineage: int,
) -> dict[str, object]:
    active_groups = _quality_active_groups(plan, states_per_lineage)
    return {
        "available_lineages": len({item.task.lineage for item in tasks}),
        "requested_lineages": requested_lineages,
        "selected_lineages": sorted(group[0].task.lineage for group in plan),
        "selected_task_ids": sorted(item.task_id for group in active_groups for item in group),
        "tasks_per_lineage_cap": tasks_per_lineage,
        "states_per_lineage_cap": states_per_lineage,
    }


def _quality_sharding_receipt(
    global_sampling: Mapping[str, object],
    shard_plan: Sequence[tuple[PreparedTask, ...]],
    *,
    shard_index: int,
    shard_count: int,
    states_per_lineage: int,
) -> dict[str, object]:
    active_groups = _quality_active_groups(shard_plan, states_per_lineage)
    return {
        "assignment": QUALITY_SHARD_ASSIGNMENT,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "global_sampling_plan_digest": content_digest(global_sampling),
        "selected_lineages": sorted(group[0].task.lineage for group in shard_plan),
        "selected_task_ids": sorted(item.task_id for group in active_groups for item in group),
    }


def _quality_record_statistics(
    records: Sequence[Mapping[str, object]],
    *,
    selected_lineages: int,
    selected_tasks: int,
) -> tuple[list[str], list[str], dict[str, int]]:
    """Recompute all manifest censuses from authenticated v7 row content."""

    statistics = _QualityRecordStatisticsAccumulator()
    for record in records:
        statistics.add(record)
    return statistics.finish(
        selected_lineages=selected_lineages,
        selected_tasks=selected_tasks,
    )


@dataclasses.dataclass
class _QualityRecordStatisticsAccumulator:
    """Bounded row-summary state used by both monolithic and streaming publishers."""

    record_lineages: set[str] = dataclasses.field(default_factory=set)
    record_task_ids: set[str] = dataclasses.field(default_factory=set)
    resolved_lineages: set[str] = dataclasses.field(default_factory=set)
    records_total: int = 0
    resolved_rows: int = 0
    continuation_trajectories: int = 0
    valid_continuations: int = 0

    def add(self, record: Mapping[str, object]) -> None:
        lineage = str(record["lineage"])
        self.record_lineages.add(lineage)
        self.record_task_ids.add(str(record["task_id"]))
        self.records_total += 1
        evaluated = record["evaluated"]
        resolved = set(record["best_actions"]) != {
            action["action_index"] for action in evaluated
        }
        if resolved:
            self.resolved_rows += 1
            self.resolved_lineages.add(lineage)
        self.continuation_trajectories += sum(
            int(action["continuations"]) for action in evaluated
        )
        self.valid_continuations += sum(
            int(action["valid_returns"]) for action in evaluated
        )

    def finish(
        self,
        *,
        selected_lineages: int,
        selected_tasks: int,
    ) -> tuple[list[str], list[str], dict[str, int]]:
        record_lineages = sorted(self.record_lineages)
        record_task_ids = sorted(self.record_task_ids)
        denominators = {
            "selected_lineages": selected_lineages,
            "selected_tasks": selected_tasks,
            "lineages_with_records": len(record_lineages),
            "tasks_with_records": len(record_task_ids),
            "records_total": self.records_total,
            "resolved_rows": self.resolved_rows,
            "fully_unresolved_rows": self.records_total - self.resolved_rows,
            "resolved_lineages": len(self.resolved_lineages),
            "continuation_trajectories": self.continuation_trajectories,
            "valid_continuations": self.valid_continuations,
        }
        return record_lineages, record_task_ids, denominators


def _quality_record_sort_key(record: Mapping[str, object]) -> tuple[object, ...]:
    return (
        str(record["lineage"]),
        str(record["task_id"]),
        tuple(record["prefix"]),
        str(record["state_fingerprint"]),
    )


def _quality_manifest_is_publication(manifest: Mapping[str, object]) -> bool:
    schema = manifest.get("schema")
    version = manifest.get("schema_version")
    publication_versions = {
        QUALITY_SCHEMA: QUALITY_SCHEMA_VERSION,
        QUALITY_SHARD_SCHEMA: QUALITY_SHARD_SCHEMA_VERSION,
        QUALITY_MERGED_SCHEMA: QUALITY_MERGED_SCHEMA_VERSION,
    }
    diagnostic_versions = {
        QUALITY_SCHEMA: QUALITY_DIAGNOSTIC_SCHEMA_VERSION,
        QUALITY_SHARD_SCHEMA: QUALITY_DIAGNOSTIC_SHARD_SCHEMA_VERSION,
        QUALITY_MERGED_SCHEMA: QUALITY_DIAGNOSTIC_MERGED_SCHEMA_VERSION,
    }
    if schema not in publication_versions:
        raise ValueError("unsupported quality-label schema")
    if version == publication_versions[schema]:
        return True
    if version == diagnostic_versions[schema]:
        return False
    raise ValueError("unsupported quality-label schema version")


def _quality_manifest_identity(manifest: Mapping[str, object]) -> dict[str, object]:
    """Fields that must be byte-equivalent across every member of a shard set."""

    fields = (
        "source_corpus_manifest_sha256",
        "quality_authority",
        "target_access",
        "ground_partition_receipt",
        "selector_digest",
        "normalizer_digest",
        "context",
        "context_digest",
        "partition",
        "sampling_receipt",
        "label_protocol",
    )
    if manifest.get("schema_version") in {
        QUALITY_SCHEMA_VERSION,
        QUALITY_SHARD_SCHEMA_VERSION,
        QUALITY_MERGED_SCHEMA_VERSION,
    }:
        fields = (*fields, "initializer_bank", "quality_resolution_plan")
    missing = [key for key in fields if key not in manifest]
    if missing:
        raise ValueError(f"quality shard manifest is missing common identity field(s): {missing}")
    return {key: manifest[key] for key in fields}


@dataclasses.dataclass(frozen=True)
class _QualityJsonlFileSnapshot:
    device: int
    inode: int
    size: int
    mtime_ns: int


@dataclasses.dataclass(frozen=True)
class _QualityJsonlRange:
    lineage: str
    start: int
    end: int
    first_line_number: int
    record_count: int


@dataclasses.dataclass(frozen=True)
class _QualityJsonlSelection:
    snapshot: _QualityJsonlFileSnapshot
    records_sha256: str
    total_record_count: int
    ranges: tuple[_QualityJsonlRange, ...]
    covers_entire_file: bool

    @property
    def lineages(self) -> frozenset[str]:
        return frozenset(item.lineage for item in self.ranges)


@dataclasses.dataclass(frozen=True)
class _QualityJsonlIndex:
    snapshot: _QualityJsonlFileSnapshot
    records_sha256: str
    record_count: int
    task_ids: frozenset[str]
    valid_continuations: int
    ordered_ranges: tuple[_QualityJsonlRange, ...]

    @property
    def lineages(self) -> frozenset[str]:
        return frozenset(item.lineage for item in self.ordered_ranges)

    def select(self, lineages: frozenset[str] | None = None) -> _QualityJsonlSelection:
        if lineages is None:
            ranges = self.ordered_ranges
            covers_entire_file = True
        else:
            ranges = tuple(item for item in self.ordered_ranges if item.lineage in lineages)
            covers_entire_file = self.lineages == lineages
        return _QualityJsonlSelection(
            snapshot=self.snapshot,
            records_sha256=self.records_sha256,
            total_record_count=self.record_count,
            ranges=ranges,
            covers_entire_file=covers_entire_file,
        )


def _quality_jsonl_snapshot(raw: os.stat_result) -> _QualityJsonlFileSnapshot:
    return _QualityJsonlFileSnapshot(
        device=int(raw.st_dev),
        inode=int(raw.st_ino),
        size=int(raw.st_size),
        mtime_ns=int(raw.st_mtime_ns),
    )


def _decode_quality_jsonl_line(line: bytes, line_number: int) -> dict[str, Any]:
    if not line.strip():
        raise ValueError(f"quality-label JSON line {line_number} is blank")
    try:
        row = _strict_json_object(line, f"quality-label JSON line {line_number}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid quality-label JSON at line {line_number}") from exc
    _verify_record(row, f"quality record line {line_number}")
    return row


def _build_quality_jsonl_index(
    root: Path,
    manifest: Mapping[str, object],
) -> _QualityJsonlIndex:
    """Authenticate a JSONL artifact while retaining only compact lineage byte ranges."""

    path = root / "records.jsonl"
    expected_sha256 = manifest.get("records_sha256")
    expected_count = manifest.get("record_count")
    if not isinstance(expected_sha256, str):
        raise ValueError("quality-label manifest has no record checksum")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
    ):
        raise ValueError("quality-label manifest has an invalid record count")
    digest = hashlib.sha256()
    ranges: list[_QualityJsonlRange] = []
    active_lineage: str | None = None
    active_start = 0
    active_first_line = 0
    active_count = 0
    record_count = 0
    task_ids: set[str] = set()
    valid_continuations = 0
    with path.open("rb") as handle:
        snapshot = _quality_jsonl_snapshot(os.fstat(handle.fileno()))
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            digest.update(line)
            line_number = record_count + 1
            row = _decode_quality_jsonl_line(line, line_number)
            lineage = row.get("lineage")
            if not isinstance(lineage, str) or not lineage:
                raise ValueError(f"quality record line {line_number} has no lineage identity")
            task_id = row.get("task_id")
            evaluated = row.get("evaluated")
            if not isinstance(task_id, str) or not task_id:
                raise ValueError(f"quality record line {line_number} has no task identity")
            if not isinstance(evaluated, list) or any(
                not isinstance(action, Mapping)
                or isinstance(action.get("valid_returns"), bool)
                or not isinstance(action.get("valid_returns"), int)
                or action["valid_returns"] < 0
                for action in evaluated
            ):
                raise ValueError(
                    f"quality record line {line_number} has malformed valid continuations"
                )
            task_ids.add(task_id)
            valid_continuations += sum(action["valid_returns"] for action in evaluated)
            if active_lineage != lineage:
                if active_lineage is not None:
                    ranges.append(
                        _QualityJsonlRange(
                            lineage=active_lineage,
                            start=active_start,
                            end=start,
                            first_line_number=active_first_line,
                            record_count=active_count,
                        )
                    )
                active_lineage = lineage
                active_start = start
                active_first_line = line_number
                active_count = 0
            active_count += 1
            record_count += 1
        if active_lineage is not None:
            ranges.append(
                _QualityJsonlRange(
                    lineage=active_lineage,
                    start=active_start,
                    end=handle.tell(),
                    first_line_number=active_first_line,
                    record_count=active_count,
                )
            )
        if _quality_jsonl_snapshot(os.fstat(handle.fileno())) != snapshot:
            raise ValueError("quality-label records changed while they were being indexed")
    if _quality_jsonl_snapshot(path.stat()) != snapshot:
        raise ValueError("quality-label records changed while they were being indexed")
    records_sha256 = digest.hexdigest()
    if not hmac.compare_digest(records_sha256, expected_sha256):
        raise ValueError("quality-label record checksum mismatch")
    if record_count != expected_count:
        raise ValueError("quality-label record count differs from its manifest")
    return _QualityJsonlIndex(
        snapshot=snapshot,
        records_sha256=records_sha256,
        record_count=record_count,
        task_ids=frozenset(task_ids),
        valid_continuations=valid_continuations,
        ordered_ranges=tuple(ranges),
    )


def _iter_quality_jsonl_selection(
    root: Path,
    selection: _QualityJsonlSelection,
):
    """Yield selected rows by byte range without materializing unrelated JSONL content."""

    path = root / "records.jsonl"
    digest = hashlib.sha256() if selection.covers_entire_file else None
    yielded = 0
    with path.open("rb") as handle:
        if _quality_jsonl_snapshot(os.fstat(handle.fileno())) != selection.snapshot:
            raise ValueError("quality-label records changed after authentication")
        for item in selection.ranges:
            handle.seek(item.start)
            range_count = 0
            while handle.tell() < item.end:
                line = handle.readline()
                if not line or handle.tell() > item.end:
                    raise ValueError("quality-label byte-range index no longer matches the file")
                if digest is not None:
                    digest.update(line)
                line_number = item.first_line_number + range_count
                row = _decode_quality_jsonl_line(line, line_number)
                if row.get("lineage") != item.lineage:
                    raise ValueError("quality-label lineage byte-range index is inconsistent")
                range_count += 1
                yielded += 1
                yield line_number, row
            if handle.tell() != item.end or range_count != item.record_count:
                raise ValueError("quality-label byte-range record census is inconsistent")
        if _quality_jsonl_snapshot(os.fstat(handle.fileno())) != selection.snapshot:
            raise ValueError("quality-label records changed during selected replay")
    if _quality_jsonl_snapshot(path.stat()) != selection.snapshot:
        raise ValueError("quality-label records changed during selected replay")
    expected_yielded = sum(item.record_count for item in selection.ranges)
    if yielded != expected_yielded:
        raise ValueError("quality-label selected record census is inconsistent")
    if selection.covers_entire_file:
        if yielded != selection.total_record_count:
            raise ValueError("quality-label full-file selection is incomplete")
        assert digest is not None
        if not hmac.compare_digest(digest.hexdigest(), selection.records_sha256):
            raise ValueError("quality-label records changed during authenticated replay")


def _assert_quality_jsonl_index_current(root: Path, index: _QualityJsonlIndex) -> None:
    path = root / "records.jsonl"
    if (
        _quality_jsonl_snapshot(path.stat()) != index.snapshot
        or not hmac.compare_digest(_sha256_file(path), index.records_sha256)
    ):
        raise ValueError("quality-label records changed after authentication")


def _quality_rows_from_artifact(root: Path, manifest: Mapping[str, object]) -> list[dict[str, Any]]:
    """Read authenticated rows without materializing the JSONL byte stream."""

    index = _build_quality_jsonl_index(root, manifest)
    return [row for _, row in _iter_quality_jsonl_selection(root, index.select())]


@dataclasses.dataclass(frozen=True)
class _QualityResolutionLabelAuthorization:
    plan: Any
    receipt: Mapping[str, object]
    continuations: int
    capacity_capability: Mapping[str, object]


def _quality_diagnostic_receipt(
    *,
    continuations: int,
    continuation_source: str,
    manifest_sha256: str,
    manifest_record_digest: str,
    records_sha256: str,
) -> dict[str, object]:
    """Seal the non-publication status of a label run lacking study authorization."""

    if isinstance(continuations, bool) or not isinstance(continuations, int) or continuations <= 0:
        raise ValueError("diagnostic quality continuations must be a positive integer")
    if continuation_source not in {"legacy-default-diagnostic", "manual-diagnostic"}:
        raise ValueError("unknown diagnostic quality continuation source")
    payload: dict[str, object] = {
        "schema": QUALITY_DIAGNOSTIC_STATUS_SCHEMA,
        "schema_version": QUALITY_DIAGNOSTIC_STATUS_VERSION,
        "mode": "diagnostic-only",
        "publication_eligible": False,
        "resolution_authorization_present": False,
        "reason": "missing-pinned-resolution-and-capacity-authorization",
        "effective_continuations": continuations,
        "continuation_source": continuation_source,
        "source_quality_manifest_sha256": _require_lower_sha256(
            manifest_sha256, "diagnostic source manifest SHA-256"
        ),
        "source_quality_manifest_record_digest": _require_lower_sha256(
            manifest_record_digest, "diagnostic source manifest record digest"
        ),
        "source_quality_records_sha256": _require_lower_sha256(
            records_sha256, "diagnostic source records SHA-256"
        ),
    }
    return _with_digest(payload)


def _validate_quality_diagnostic_artifact(
    root: Path,
    *,
    manifest: Mapping[str, object],
) -> None:
    """Require the explicit non-publication receipt for a legacy initializer artifact."""

    status_root = root / "DIAGNOSTIC_ONLY"
    status_path = status_root / "receipt.json"
    if (
        not status_root.is_dir()
        or status_root.is_symlink()
        or {item.name for item in status_root.iterdir()} != {"receipt.json"}
        or not status_path.is_file()
        or status_path.is_symlink()
    ):
        raise ValueError("legacy quality labels require an explicit diagnostic-only receipt")
    receipt = _strict_json(status_path)
    _verify_record(receipt, "diagnostic quality-label receipt")
    expected_fields = {
        "continuation_source",
        "effective_continuations",
        "mode",
        "publication_eligible",
        "reason",
        "record_digest",
        "resolution_authorization_present",
        "schema",
        "schema_version",
        "source_quality_manifest_record_digest",
        "source_quality_manifest_sha256",
        "source_quality_records_sha256",
    }
    if (
        set(receipt) != expected_fields
        or receipt.get("schema") != QUALITY_DIAGNOSTIC_STATUS_SCHEMA
        or receipt.get("schema_version") != QUALITY_DIAGNOSTIC_STATUS_VERSION
        or receipt.get("mode") != "diagnostic-only"
        or receipt.get("publication_eligible") is not False
        or receipt.get("resolution_authorization_present") is not False
        or receipt.get("reason") != "missing-pinned-resolution-and-capacity-authorization"
        or receipt.get("source_quality_manifest_sha256")
        != _sha256_file(root / "manifest.json")
        or receipt.get("source_quality_manifest_record_digest") != manifest.get("record_digest")
        or receipt.get("source_quality_records_sha256")
        != _sha256_file(root / "records.jsonl")
    ):
        raise ValueError("legacy quality diagnostic receipt does not bind this exact artifact")


def _quality_resolution_label_authorization(
    args: argparse.Namespace, ctx: Context
) -> _QualityResolutionLabelAuthorization | None:
    """Authenticate the publication protocol before opening the train targets."""

    fields = (
        "resolution_plan",
        "expected_resolution_plan_sha256",
        "resolution_receipt",
        "expected_resolution_receipt_sha256",
        "capacity_selection",
        "expected_capacity_selection_sha256",
        "capacity_budget",
        "expected_capacity_budget_sha256",
        "capacity_canary",
        "expected_capacity_canary_sha256",
    )
    values = tuple(getattr(args, field, None) for field in fields)
    present = tuple(isinstance(value, str) and bool(value) for value in values)
    if not any(present):
        return None
    if not all(present):
        raise ValueError(
            "publication quality labeling requires the resolution plan, terminal study, "
            "capacity selection, budget, passing canary, and every out-of-band raw-file pin"
        )

    from isingfold.rl.data.quality_resolution_merge import (
        load_quality_resolution_study_receipt,
    )

    plan = _load_pinned_quality_resolution_plan(values[0], values[1])
    study = load_quality_resolution_study_receipt(
        values[2],
        expected_receipt_sha256=values[3],
        plan=plan,
        expected_plan_sha256=values[1],
    )
    from isingfold.rl.data.quality_capacity import (
        require_passing_quality_capacity_canary,
    )

    capacity_capability = require_passing_quality_capacity_canary(
        values[8],
        expected_canary_sha256=values[9],
        plan=plan,
        expected_plan_sha256=values[1],
        resolution_receipt=study,
        selection_path=values[4],
        expected_selection_sha256=values[5],
        budget_path=values[6],
        expected_budget_sha256=values[7],
    )
    plan_record = plan.as_dict()
    receipt = study.as_dict()
    selected = receipt.get("selected_continuations")
    if (
        receipt.get("terminal") is not True
        or receipt.get("advance") is not True
        or isinstance(selected, bool)
        or not isinstance(selected, int)
        or selected <= 0
    ):
        raise ValueError(
            "publication quality labeling requires the first passing terminal resolution receipt"
        )
    manual = getattr(args, "continuations", None)
    if manual is not None and manual != selected:
        raise ValueError(
            "manual --continuations differs from the externally pinned resolution decision"
        )
    protocol = receipt.get("production_protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("resolution receipt has no production protocol")
    expected_contract_digest = content_digest(_quality_implementation_contract(ctx))
    if (
        capacity_capability.get("full_quality_launch_authorized") is not True
        or capacity_capability.get("selected_continuations") != selected
        or capacity_capability.get("quality_implementation_contract_digest")
        != expected_contract_digest
    ):
        raise ValueError(
            "quality-capacity capability differs from the selected production protocol"
        )
    expected_protocol = {
        "continuations": selected,
        "evaluated_actions": args.actions,
        "partition": "train",
        "quality_implementation_contract_digest": expected_contract_digest,
        "requested_lineages": args.instances,
        "reward_reads": args.reward_reads,
        "seed": args.seed,
        "selector_device": args.device,
        "states_per_lineage_cap": args.states_per_lineage,
        "tasks_per_lineage_cap": args.tasks_per_lineage,
    }
    if dict(protocol) != expected_protocol:
        raise ValueError("label-quality arguments differ from the sealed production protocol")
    plan_context = plan_record.get("context")
    if (
        not isinstance(plan_context, Mapping)
        or plan_context.get("snapshot") != _context_snapshot(ctx)
        or plan_context.get("digest") != content_digest(_context_snapshot(ctx))
    ):
        raise ValueError("label-quality context differs from the resolution plan")
    prepared = plan_record.get("prepared_corpus")
    if not isinstance(prepared, Mapping) or prepared.get(
        "manifest_sha256"
    ) != _corpus_manifest_digest(args.corpus):
        raise ValueError("label-quality corpus differs from the resolution plan")
    return _QualityResolutionLabelAuthorization(
        plan=plan,
        receipt=receipt,
        continuations=selected,
        capacity_capability=capacity_capability,
    )


def _validate_quality_resolution_label_runtime(
    authorization: _QualityResolutionLabelAuthorization,
    *,
    bundle: SelectorBundle,
    quality_authority: Mapping[str, object],
    target_access: Mapping[str, object],
    ground_partition_receipt: Mapping[str, object],
) -> None:
    """Ensure the opened train capability is exactly the study-authorized authority."""

    plan_record = authorization.plan.as_dict()
    selector = plan_record.get("selector")
    if (
        not isinstance(selector, Mapping)
        or selector.get("selector_digest") != bundle.selector_digest
        or selector.get("normalizer_digest") != bundle.normalizer_digest
    ):
        raise ValueError("quality selector differs from the resolution plan")
    receipt_authority = authorization.receipt.get("quality_authority")
    if not isinstance(receipt_authority, Mapping):
        raise ValueError("resolution receipt has no executed train authority")
    train_authority = quality_authority.get("training_partition")
    ground_identity = (
        train_authority.get("ground_partition") if isinstance(train_authority, Mapping) else None
    )
    if not isinstance(ground_identity, Mapping):
        raise ValueError("opened train authority has no ground-partition identity")
    observed = {
        "ground_partition_receipt_record_digest": ground_partition_receipt.get("record_digest"),
        "ground_partition_receipt_sha256": ground_identity.get("receipt_sha256"),
        "quality_authority_record_digest": quality_authority.get("record_digest"),
        "target_access_record_digest": target_access.get("record_digest"),
    }
    if dict(receipt_authority) != observed:
        raise ValueError("opened quality target authority differs from the resolution receipt")


def cmd_label_quality(args: argparse.Namespace) -> None:
    """Create a checksummed warm-start label corpus from authenticated training tasks."""

    from isingfold.rl.data.prepared import load_prepared_partition
    from isingfold.rl.data.quality import (
        label_decision,
        quality_initializer_bank_contract,
        validate_publication_quality_record_initializer_binding,
    )

    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"quality-label corpus already exists: {destination}")
    partition = _normalize_quality_partition(args.partition)
    if partition != "train":
        raise ValueError("quality warm-start labels may open the train partition only")
    ctx = _context(args, args.corpus)
    authorization = _quality_resolution_label_authorization(args, ctx)
    bank_arguments = (
        getattr(args, "initializer_bank", None),
        getattr(args, "expected_initializer_bank_manifest_sha256", None),
        getattr(args, "complete_config", None),
    )
    if authorization is None and any(value is not None for value in bank_arguments):
        raise ValueError(
            "initializer-bank inputs require the complete publication quality authorization"
        )
    initializer_bank = None
    initializer_bank_manifest_sha256 = None
    initializer_bank_contract = None
    if authorization is not None:
        public = load_prepared_partition(
            args.corpus,
            partition="train",
            include_evaluator=False,
        )
        if public.target_access is not None:
            raise RuntimeError("publication quality initializer-bank load opened evaluator targets")
        initializer_bank, initializer_bank_manifest_sha256 = _load_quality_initializer_bank(
            args,
            public_tasks=public.tasks,
            context=ctx,
        )
        initializer_bank_contract = quality_initializer_bank_contract(
            initializer_bank,
            expected_manifest_sha256=initializer_bank_manifest_sha256,
        )
        planned_bank = authorization.plan.as_dict()["production_plan"]["initializer_bank"]
        if initializer_bank_contract != planned_bank:
            raise ValueError(
                "publication quality initializer bank differs from the resolution plan"
            )
        capability = authorization.capacity_capability
        if (
            capability.get("initializer_bank_manifest_sha256")
            != initializer_bank_contract["manifest_sha256"]
            or capability.get("initializer_bank_contract_record_digest")
            != initializer_bank_contract["record_digest"]
        ):
            raise ValueError("quality-capacity capability belongs to another initializer bank")
    continuations = (
        authorization.continuations
        if authorization is not None
        else (2 if args.continuations is None else args.continuations)
    )
    quality_pin = _quality_attestation_pin(args)
    tasks, quality_authority, target_access, ground_partition_receipt = _load_quality_partition(
        args.corpus,
        partition=partition,
        pin=quality_pin,
        role="training_partition",
    )
    bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
    _bind_selector_quality_authority(bundle, tasks, pin=quality_pin)
    if authorization is not None:
        _validate_quality_resolution_label_runtime(
            authorization,
            bundle=bundle,
            quality_authority=quality_authority,
            target_access=target_access,
            ground_partition_receipt=ground_partition_receipt,
        )
    _seed_runtime(args.seed, deterministic=args.deterministic, threads=args.threads)
    bundle.model.to(_resolve_device(args.device)).eval()
    source_corpus_manifest_sha256 = _corpus_manifest_digest(args.corpus)
    global_plan = _quality_sampling_plan(
        tasks,
        lineage_count=args.instances,
        tasks_per_lineage=args.tasks_per_lineage,
        seed=args.seed,
    )
    shard_index = getattr(args, "shard_index", None)
    shard_count = getattr(args, "shard_count", None)
    if (shard_index is None) != (shard_count is None):
        raise ValueError("--shard-index and --shard-count must be supplied together")
    global_sampling = _quality_sampling_receipt(
        tasks,
        global_plan,
        requested_lineages=args.instances,
        tasks_per_lineage=args.tasks_per_lineage,
        states_per_lineage=args.states_per_lineage,
    )
    sharding_receipt = None
    plan = global_plan
    if shard_index is not None and shard_count is not None:
        plan = _quality_plan_shard(
            global_plan,
            shard_index=shard_index,
            shard_count=shard_count,
        )
        sharding_receipt = _quality_sharding_receipt(
            global_sampling,
            plan,
            shard_index=shard_index,
            shard_count=shard_count,
            states_per_lineage=args.states_per_lineage,
        )
    publication_rows_by_task: dict[str, list[Mapping[str, object]]] = {}
    if authorization is not None:
        production_rows = authorization.plan.as_dict()["production_plan"]["rows"]
        if not isinstance(production_rows, list):
            raise ValueError("quality-resolution plan has no production row registry")
        for planned_row in production_rows:
            if not isinstance(planned_row, Mapping):
                raise ValueError("quality-resolution production row is malformed")
            publication_rows_by_task.setdefault(str(planned_row["task_id"]), []).append(planned_row)
        for planned_rows in publication_rows_by_task.values():
            planned_rows.sort(key=lambda row: int(row["state_schedule_index"]))
        expected_rows_by_task: dict[str, int] = {}
        for group in global_plan:
            active = group[: min(len(group), args.states_per_lineage)]
            base, extra = divmod(args.states_per_lineage, len(active))
            for task_position, item in enumerate(active):
                if item.task_id in expected_rows_by_task:
                    raise ValueError("publication quality sampling selected a task more than once")
                expected_rows_by_task[item.task_id] = base + (1 if task_position < extra else 0)
        if set(expected_rows_by_task) != set(publication_rows_by_task) or any(
            len(publication_rows_by_task[task_id]) != expected_count
            for task_id, expected_count in expected_rows_by_task.items()
        ):
            raise ValueError("publication quality sampling differs from the sealed resolution plan")
    records: list[dict[str, object]] = []
    selected_tasks: list[PreparedTask] = []
    selected_lineages: list[str] = []
    progress_started = time.monotonic()
    for group_position, group in enumerate(plan, start=1):
        lineage = group[0].task.lineage
        selected_lineages.append(lineage)
        active = group[: min(len(group), args.states_per_lineage)]
        selected_tasks.extend(active)
        base, extra = divmod(args.states_per_lineage, len(active))
        for task_position, item in enumerate(active):
            state_cap = base + (1 if task_position < extra else 0)
            if authorization is None:
                environment_seed = _domain_seed(
                    args.seed, "quality-environment", lineage, item.task_id
                )
                prefixes = _quality_prefixes(
                    item,
                    ctx,
                    bundle.model,
                    count=state_cap,
                    seed=environment_seed,
                )
                state_specs: tuple[Mapping[str, object] | None, ...] = tuple(
                    {
                        "environment_seed": environment_seed,
                        "prefix": prefix,
                    }
                    for prefix in prefixes
                )
            else:
                planned_rows = publication_rows_by_task.get(item.task_id, [])
                if len(planned_rows) != state_cap:
                    raise ValueError(
                        "publication quality state census differs from the resolution plan"
                    )
                state_specs = tuple(planned_rows)
            for planned_row in state_specs:
                if planned_row is None:  # pragma: no cover - state specs are mappings
                    raise RuntimeError("quality state specification disappeared")
                prefix = tuple(int(index) for index in planned_row["prefix"])
                environment_seed = int(planned_row["environment_seed"])
                episode_index = (
                    int(planned_row["initializer_bank_episode_index"])
                    if authorization is not None
                    else None
                )
                provenance_fingerprint = (
                    str(planned_row["action_provenance_fingerprint"])
                    if authorization is not None
                    else _quality_action_provenance_fingerprint(
                        item,
                        source_corpus_manifest_sha256=(source_corpus_manifest_sha256),
                    )
                )
                record = label_decision(
                    item.task,
                    ctx,
                    initializer=(item.initializer() if authorization is None else None),
                    initializer_bank=initializer_bank,
                    expected_initializer_bank_manifest_sha256=(initializer_bank_manifest_sha256),
                    initializer_bank_episode_index=episode_index,
                    prepared_task=(item if authorization is not None else None),
                    target_access=(target_access if authorization is not None else None),
                    ground_partition_receipt=(
                        ground_partition_receipt if authorization is not None else None
                    ),
                    selector=bundle.model,
                    prefix=prefix,
                    seed=environment_seed,
                    provenance_fingerprint=provenance_fingerprint,
                    evaluated_actions=args.actions,
                    continuations=continuations,
                    reward_reads=args.reward_reads,
                )
                if record is None and authorization is not None:
                    raise RuntimeError("sealed publication quality row is no longer replayable")
                if record is not None:
                    if authorization is not None:
                        if initializer_bank_contract is None:
                            raise RuntimeError("publication quality bank contract disappeared")
                        binding = validate_publication_quality_record_initializer_binding(
                            record,
                            expected_initializer_bank_contract=(initializer_bank_contract),
                        )
                        planned_actions = planned_row["actions"]
                        if (
                            binding["episode_schedule_index"] != episode_index
                            or binding["bootstrap_record_digest"]
                            != planned_row["initializer_bootstrap_record_digest"]
                            or record.state_fingerprint != planned_row["state_fingerprint"]
                            or record.support_fingerprint != planned_row["support_fingerprint"]
                            or record.action_envelope_record_digest
                            != planned_row["action_envelope_record_digest"]
                            or not isinstance(planned_actions, list)
                            or tuple(action.action_index for action in record.evaluated)
                            != tuple(
                                int(action["action_index"])
                                for action in planned_actions
                                if isinstance(action, Mapping)
                            )
                        ):
                            raise ValueError("publication quality row differs from its sealed plan")
                    records.append(_quality_record_payload(record, item))
        print(
            json.dumps(
                {
                    "event": "quality-label-progress",
                    "shard_index": shard_index,
                    "completed_lineages": group_position,
                    "assigned_lineages": len(plan),
                    "records": len(records),
                    "elapsed_seconds": round(time.monotonic() - progress_started, 1),
                }
            ),
            flush=True,
        )

    if not records:
        raise RuntimeError("no replayable quality decision was labelled")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        raw = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
        (temporary / "records.jsonl").write_bytes(raw)
        record_lineages, record_task_ids, denominators = _quality_record_statistics(
            records,
            selected_lineages=len(selected_lineages),
            selected_tasks=len(selected_tasks),
        )
        label_protocol = {
            "requested_lineages": args.instances,
            "tasks_per_lineage_cap": args.tasks_per_lineage,
            "states_per_lineage_cap": args.states_per_lineage,
            "evaluated_actions": args.actions,
            "continuations": continuations,
            "reward_reads": args.reward_reads,
            "seed": args.seed,
            "implementation_contract": _quality_implementation_contract(ctx),
        }
        manifest_payload: dict[str, object] = {
            "schema": QUALITY_SHARD_SCHEMA if sharding_receipt else QUALITY_SCHEMA,
            "schema_version": (
                (
                    QUALITY_SHARD_SCHEMA_VERSION
                    if authorization is not None
                    else QUALITY_DIAGNOSTIC_SHARD_SCHEMA_VERSION
                )
                if sharding_receipt
                else (
                    QUALITY_SCHEMA_VERSION
                    if authorization is not None
                    else QUALITY_DIAGNOSTIC_SCHEMA_VERSION
                )
            ),
            "source_corpus_manifest_sha256": source_corpus_manifest_sha256,
            "quality_authority": quality_authority,
            "target_access": target_access,
            "ground_partition_receipt": ground_partition_receipt,
            "selector_digest": bundle.selector_digest,
            "normalizer_digest": bundle.normalizer_digest,
            "context": _context_snapshot(ctx),
            "context_digest": stable_digest(_context_snapshot(ctx)),
            "partition": args.partition,
            "record_count": len(records),
            "records_sha256": hashlib.sha256(raw).hexdigest(),
            "lineages": record_lineages,
            "task_ids": record_task_ids,
            "sampling_receipt": global_sampling,
            "independent_denominators": denominators,
            "label_protocol": label_protocol,
        }
        if authorization is not None:
            if initializer_bank_contract is None:
                raise RuntimeError("publication quality bank contract disappeared")
            manifest_payload["initializer_bank"] = initializer_bank_contract
            manifest_payload["quality_resolution_plan"] = {
                "raw_sha256": _require_lower_sha256(
                    args.expected_resolution_plan_sha256,
                    "publication quality resolution-plan SHA-256",
                ),
                "record_digest": authorization.plan.as_dict()["record_digest"],
            }
        if sharding_receipt is not None:
            manifest_payload["sharding_receipt"] = sharding_receipt
        manifest_record = _with_digest(manifest_payload)
        _atomic_json(temporary / "manifest.json", manifest_record)
        diagnostic_receipt: dict[str, object] | None = None
        if authorization is None:
            diagnostic_receipt = _quality_diagnostic_receipt(
                continuations=continuations,
                continuation_source=(
                    "legacy-default-diagnostic"
                    if args.continuations is None
                    else "manual-diagnostic"
                ),
                manifest_sha256=_sha256_file(temporary / "manifest.json"),
                manifest_record_digest=str(manifest_record["record_digest"]),
                records_sha256=str(manifest_record["records_sha256"]),
            )
            _atomic_json(
                temporary / "DIAGNOSTIC_ONLY" / "receipt.json",
                diagnostic_receipt,
            )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    result: dict[str, object] = {
        "records": len(records),
        "lineages": len(record_lineages),
        "resolved_rows": denominators["resolved_rows"],
        "resolved_lineages": denominators["resolved_lineages"],
        "mode": "production" if authorization is not None else "diagnostic-only",
        "publication_eligible": authorization is not None,
        "out": str(destination),
    }
    if authorization is None:
        diagnostic_path = destination / "DIAGNOSTIC_ONLY" / "receipt.json"
        result["diagnostic_receipt"] = str(diagnostic_path)
        result["diagnostic_receipt_sha256"] = _sha256_file(diagnostic_path)
    print(json.dumps(result, indent=1))


def _quality_shard_headers(
    shard_paths: Sequence[str | os.PathLike[str]],
    *,
    allow_diagnostic_legacy: bool = False,
) -> list[tuple[int, Path, dict[str, Any], str]]:
    """Authenticate the shard-set envelope before any expensive exact replay."""

    if not shard_paths:
        raise ValueError("at least one quality shard is required")
    headers: list[tuple[int, Path, dict[str, Any], str]] = []
    expected_count: int | None = None
    expected_identity: dict[str, object] | None = None
    seen_indices: set[int] = set()
    for source in shard_paths:
        root = Path(source)
        if not root.is_dir():
            raise ValueError(f"quality shard directory does not exist: {root}")
        manifest_path = root / "manifest.json"
        manifest_raw = manifest_path.read_bytes()
        manifest = _strict_json_object(manifest_raw, str(manifest_path))
        _verify_record(manifest, f"quality shard manifest {root}")
        version = manifest.get("schema_version")
        publication = version == QUALITY_SHARD_SCHEMA_VERSION
        diagnostic = version == QUALITY_DIAGNOSTIC_SHARD_SCHEMA_VERSION
        if manifest.get("schema") != QUALITY_SHARD_SCHEMA or not (
            publication or diagnostic and allow_diagnostic_legacy
        ):
            raise ValueError(f"merge input is not a registered quality shard: {root}")
        if diagnostic:
            _validate_quality_diagnostic_artifact(root, manifest=manifest)
        sharding = manifest.get("sharding_receipt")
        if not isinstance(sharding, dict):
            raise ValueError("quality shard has no sharding receipt")
        shard_index = sharding.get("shard_index")
        shard_count = sharding.get("shard_count")
        if (
            isinstance(shard_index, bool)
            or not isinstance(shard_index, int)
            or shard_index < 0
            or isinstance(shard_count, bool)
            or not isinstance(shard_count, int)
            or shard_count <= 0
            or shard_index >= shard_count
        ):
            raise ValueError("quality shard index/count is malformed")
        if shard_index in seen_indices:
            raise ValueError(f"duplicate shard index {shard_index}")
        seen_indices.add(shard_index)
        if expected_count is None:
            expected_count = shard_count
        elif shard_count != expected_count:
            raise ValueError("quality shards disagree on shard count")
        identity = _quality_manifest_identity(manifest)
        if expected_identity is None:
            expected_identity = identity
        elif identity != expected_identity:
            raise ValueError(
                "quality shards disagree on corpus, selector, context, or label protocol"
            )
        headers.append((shard_index, root, manifest, hashlib.sha256(manifest_raw).hexdigest()))
    assert expected_count is not None
    missing = sorted(set(range(expected_count)) - seen_indices)
    if missing:
        raise ValueError(f"quality merge is incomplete; missing shard indices {missing}")
    return sorted(headers)


def cmd_merge_quality_labels(args: argparse.Namespace) -> None:
    """Exact-replay every shard and atomically publish one training-eligible corpus."""

    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"merged quality-label corpus already exists: {destination}")
    allow_diagnostic_legacy = bool(getattr(args, "allow_legacy_pilot", False))
    headers = _quality_shard_headers(
        args.shard,
        allow_diagnostic_legacy=allow_diagnostic_legacy,
    )
    publication = headers[0][2]["schema_version"] == QUALITY_SHARD_SCHEMA_VERSION
    replay_bundle_path = getattr(args, "trusted_replay_bundle", None)
    replay_bundle_sha256 = getattr(args, "expected_trusted_replay_bundle_sha256", None)
    if (replay_bundle_path is None) != (replay_bundle_sha256 is None):
        raise ValueError(
            "trusted replay bundle and its external SHA-256 pin must be supplied together"
        )
    trusted_replay_rows: Mapping[str, Mapping[str, object]] | None = None
    trusted_replay_capability: Mapping[str, object] | None = None
    if replay_bundle_path is not None and replay_bundle_sha256 is not None:
        from isingfold.rl.data.quality_preflight import (
            load_trusted_quality_preflight_replay_bundle,
        )

        sources = _quality_preflight_source_shards(
            headers,
            expected_manifest_sha256=[header[3] for header in headers],
            allow_diagnostic_legacy=allow_diagnostic_legacy,
        )
        trusted_replay_capability = load_trusted_quality_preflight_replay_bundle(
            replay_bundle_path,
            expected_sha256=replay_bundle_sha256,
            source_shards=sources,
        )
        trusted_replay_rows = trusted_replay_capability["trusted_rows"]  # type: ignore[assignment]
    ctx = _context(args, args.corpus)
    bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
    _seed_runtime(args.seed, deterministic=args.deterministic, threads=args.threads)
    bundle.model.to(_resolve_device(args.device)).eval()
    initializer_bank = None
    initializer_bank_manifest_sha256 = None
    if publication:
        from isingfold.rl.data.prepared import load_prepared_partition

        public = load_prepared_partition(
            args.corpus,
            partition="train",
            include_evaluator=False,
        )
        if public.target_access is not None:
            raise RuntimeError("quality merge bank load opened evaluator targets")
        initializer_bank, initializer_bank_manifest_sha256 = _load_quality_initializer_bank(
            args,
            public_tasks=public.tasks,
            context=ctx,
        )
    elif any(
        getattr(args, name, None) is not None
        for name in (
            "initializer_bank",
            "expected_initializer_bank_manifest_sha256",
            "complete_config",
        )
    ):
        raise ValueError("diagnostic quality merge cannot consume a publication initializer bank")
    quality_pin = _quality_attestation_pin(args)
    (
        prepared_train_tasks,
        quality_authority,
        target_access,
        ground_partition_receipt,
    ) = _load_quality_partition(
        args.corpus,
        partition="train",
        pin=quality_pin,
        role="training_partition",
    )
    if (
        _bind_selector_quality_authority(bundle, prepared_train_tasks, pin=quality_pin)
        != quality_authority
    ):
        raise ValueError("selector and quality-label merge authorities differ")
    for _, _, manifest, _ in headers:
        if (
            manifest.get("quality_authority") != quality_authority
            or manifest.get("target_access") != target_access
            or manifest.get("ground_partition_receipt") != ground_partition_receipt
        ):
            raise ValueError("quality shard target-access authority differs")

    source_receipts: list[dict[str, object]] = []
    source_indexes: dict[int, _QualityJsonlIndex] = {}
    source_roots: dict[int, Path] = {}
    lineage_owner: dict[str, int] = {}
    authenticated_records = 0
    progress_started = time.monotonic()
    for shard_position, (shard_index, root, manifest, manifest_sha256) in enumerate(
        headers, start=1
    ):
        record_index = _build_quality_jsonl_index(root, manifest)
        shard_trusted_rows = None
        if trusted_replay_rows is not None:
            shard_digests = {
                str(row["record_digest"])
                for _, row in _iter_quality_jsonl_selection(root, record_index.select())
            }
            shard_trusted_rows = {
                digest: trusted_replay_rows[digest]
                for digest in shard_digests
                if digest in trusted_replay_rows
            }
            if set(shard_trusted_rows) != shard_digests:
                raise ValueError("trusted replay bundle omits a quality-shard row")
        _load_quality_labels(
            root,
            corpus=args.corpus,
            selector=bundle,
            context=ctx,
            enforce_resolution=False,
            _allow_shard=True,
            _prepared_train_tasks=prepared_train_tasks,
            quality_attestation_pin=quality_pin,
            _trusted_replay_rows=shard_trusted_rows,
            initializer_bank=initializer_bank,
            expected_initializer_bank_manifest_sha256=(
                initializer_bank_manifest_sha256
            ),
            _record_selection=record_index.select(),
            _collect_decoded=False,
            allow_diagnostic_legacy=allow_diagnostic_legacy,
        )
        if _sha256_file(root / "manifest.json") != manifest_sha256:
            raise ValueError("quality shard manifest changed while it was being authenticated")
        sharding = manifest["sharding_receipt"]
        for lineage in sharding["selected_lineages"]:
            previous = lineage_owner.setdefault(lineage, shard_index)
            if previous != shard_index:
                raise ValueError(
                    f"quality shards duplicate lineage {lineage!r} in indices "
                    f"{previous} and {shard_index}"
                )
        source_indexes[shard_index] = record_index
        source_roots[shard_index] = root
        authenticated_records += record_index.record_count
        source_receipts.append(
            {
                "shard_index": shard_index,
                "manifest_sha256": manifest_sha256,
                "manifest_record_digest": manifest["record_digest"],
                "records_sha256": manifest["records_sha256"],
                "record_count": manifest["record_count"],
                "selected_lineages": sharding["selected_lineages"],
                "selected_task_ids": sharding["selected_task_ids"],
            }
        )
        print(
            json.dumps(
                {
                    "event": "quality-merge-progress",
                    "completed_shards": shard_position,
                    "expected_shards": len(headers),
                    "records": authenticated_records,
                    "elapsed_seconds": round(time.monotonic() - progress_started, 1),
                }
            ),
            flush=True,
        )

    common = headers[0][2]
    sampling = common["sampling_receipt"]
    expected_lineages = set(sampling["selected_lineages"])
    if set(lineage_owner) != expected_lineages:
        missing = sorted(expected_lineages - set(lineage_owner))
        extra = sorted(set(lineage_owner) - expected_lineages)
        raise ValueError(
            f"quality shard lineage union differs from the global plan; missing={missing}, extra={extra}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        statistics = _QualityRecordStatisticsAccumulator()
        seen_states: set[tuple[str, str]] = set()
        seen_continuation_seeds: set[int] = set()
        records_sha256 = hashlib.sha256()
        merged_record_count = 0
        with (temporary / "records.jsonl").open("wb") as output:
            for lineage in sorted(expected_lineages):
                shard_index = lineage_owner[lineage]
                record_index = source_indexes[shard_index]
                rows = [
                    row
                    for _, row in _iter_quality_jsonl_selection(
                        source_roots[shard_index],
                        record_index.select(frozenset({lineage})),
                    )
                ]
                rows.sort(key=_quality_record_sort_key)
                for row in rows:
                    if row.get("lineage") != lineage:
                        raise ValueError("quality shard row is outside its exact lineage subset")
                    state_key = (str(row["task_id"]), str(row["state_fingerprint"]))
                    if state_key in seen_states:
                        raise ValueError("quality shards contain a duplicate task/state row")
                    seen_states.add(state_key)
                    for action in row["evaluated"]:
                        for continuation_seed in action["continuation_seeds"]:
                            if continuation_seed in seen_continuation_seeds:
                                raise ValueError(
                                    "quality shards reuse a continuation seed across task/state rows"
                                )
                            seen_continuation_seeds.add(continuation_seed)
                    statistics.add(row)
                    encoded = canonical_json_bytes(row) + b"\n"
                    output.write(encoded)
                    records_sha256.update(encoded)
                    merged_record_count += 1
        if merged_record_count == 0:
            raise RuntimeError("complete quality shard set contains no replayable records")
        if merged_record_count != authenticated_records:
            raise ValueError("quality shard record counts differ from streamed merge output")
        for shard_index, root, _manifest, manifest_sha256 in headers:
            _assert_quality_jsonl_index_current(root, source_indexes[shard_index])
            if not hmac.compare_digest(_sha256_file(root / "manifest.json"), manifest_sha256):
                raise ValueError("quality shard manifest changed during streamed merge")
        record_lineages, record_task_ids, denominators = statistics.finish(
            selected_lineages=len(sampling["selected_lineages"]),
            selected_tasks=len(sampling["selected_task_ids"]),
        )
        manifest_payload = {
            "schema": QUALITY_MERGED_SCHEMA,
            "schema_version": (
                QUALITY_MERGED_SCHEMA_VERSION
                if publication
                else QUALITY_DIAGNOSTIC_MERGED_SCHEMA_VERSION
            ),
            **_quality_manifest_identity(common),
            "record_count": merged_record_count,
            "records_sha256": records_sha256.hexdigest(),
            "lineages": record_lineages,
            "task_ids": record_task_ids,
            "independent_denominators": denominators,
            "merge_receipt": {
                "assignment": QUALITY_SHARD_ASSIGNMENT,
                "shard_count": len(headers),
                "global_sampling_plan_digest": content_digest(sampling),
                "source_shards": source_receipts,
            },
        }
        merged_manifest = _with_digest(manifest_payload)
        _atomic_json(temporary / "manifest.json", merged_manifest)
        if not publication:
            diagnostic = _quality_diagnostic_receipt(
                continuations=int(common["label_protocol"]["continuations"]),
                continuation_source="manual-diagnostic",
                manifest_sha256=_sha256_file(temporary / "manifest.json"),
                manifest_record_digest=str(merged_manifest["record_digest"]),
                records_sha256=str(merged_manifest["records_sha256"]),
            )
            _atomic_json(temporary / "DIAGNOSTIC_ONLY" / "receipt.json", diagnostic)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(
        json.dumps(
            {
                "shards": len(headers),
                "records": merged_record_count,
                "lineages": len(record_lineages),
                "resolved_rows": denominators["resolved_rows"],
                "resolved_lineages": denominators["resolved_lineages"],
                "mode": (
                    "publication" if publication else "diagnostic-serial-replay"
                ),
                "trusted_replay_bundle_sha256": (
                    trusted_replay_capability.get("bundle_sha256")
                    if trusted_replay_capability is not None
                    else None
                ),
                "out": str(destination),
            },
            indent=1,
        )
    )


def _decode_observation(raw: object):
    from isingfold.rl.tensorize import Observation

    if not isinstance(raw, dict):
        raise ValueError("quality label has no complete observation mapping")
    expected = {field.name for field in dataclasses.fields(Observation)}
    if set(raw) != expected:
        raise ValueError("quality observation fields differ from the registered tensor schema")
    values: dict[str, object] = {}
    for name, value in raw.items():
        if name in {"qubit_ids", "logical_ids"}:
            if not isinstance(value, list):
                raise ValueError(f"observation {name} must be a list")
            values[name] = tuple(value)
        elif name in {"legal_mask", "real_action_mask"}:
            values[name] = np.asarray(value, dtype=bool)
        elif name.startswith("index_") or name.endswith("_roles"):
            values[name] = np.asarray(value, dtype=np.int64)
        else:
            values[name] = np.asarray(value, dtype=np.float32)
    return Observation(**values)


def _load_quality_labels(
    path: str | os.PathLike[str],
    *,
    corpus: str | os.PathLike[str],
    selector: SelectorBundle,
    context: Context,
    min_resolved_rows: int = 1,
    min_resolved_lineages: int = 1,
    enforce_resolution: bool = True,
    require_provenance: bool = True,
    quality_attestation_pin: object | None = None,
    trusted_preflight: Mapping[str, Any] | None = None,
    _allow_shard: bool = False,
    _prepared_train_tasks: Sequence[PreparedTask] | None = None,
    _replay_evidence: list[dict[str, object]] | None = None,
    _trusted_replay_rows: Mapping[str, Mapping[str, object]] | None = None,
    _replay_lineages: frozenset[str] | None = None,
    _record_selection: _QualityJsonlSelection | None = None,
    _collect_decoded: bool = True,
    initializer_bank: object | None = None,
    expected_initializer_bank_manifest_sha256: str | None = None,
    allow_diagnostic_legacy: bool = False,
) -> tuple[
    list[tuple[object, tuple[int, ...], tuple[int, ...], object]],
    dict[str, Any],
]:
    """Authenticate quality-v7 rows and optionally consume a pinned full-replay receipt.

    Decision states and every retained receipt are always reconstructed and checked.  The costly
    continuation/evaluator rerun is skipped only after ``trusted_preflight`` or row-scoped replay
    evidence has been authenticated out of band.  The two trust capabilities are mutually
    exclusive; replay workers may request compact evidence from an actual full rerun.
    """

    from isingfold.rl.data import quality
    from isingfold.rl.data.action_certificate import apply_envelope_action
    from isingfold.rl.ppo import WarmStartActionValueTarget, warm_start_utility_target

    for name, value in (
        ("minimum resolved rows", min_resolved_rows),
        ("minimum resolved lineages", min_resolved_lineages),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if trusted_preflight is not None and _trusted_replay_rows is not None:
        raise ValueError("full-preflight and row-scoped replay trust cannot be combined")
    if _replay_evidence is not None and (
        trusted_preflight is not None or _trusted_replay_rows is not None
    ):
        raise ValueError("replay evidence can be emitted only by an actual continuation replay")
    if _replay_evidence:
        raise ValueError("replay evidence destination must be empty")
    if _replay_lineages is not None and (
        not _replay_lineages
        or any(not isinstance(lineage, str) or not lineage for lineage in _replay_lineages)
        or trusted_preflight is not None
        or _trusted_replay_rows is not None
    ):
        raise ValueError("partial replay requires a nonempty lineage set and no trusted evidence")
    if not isinstance(_collect_decoded, bool):
        raise TypeError("quality decoded-row collection flag must be boolean")
    if _record_selection is not None and not isinstance(
        _record_selection, _QualityJsonlSelection
    ):
        raise TypeError("quality record selection has an invalid type")
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"quality-label directory does not exist: {root}")
    if {item.name for item in root.iterdir() if item.is_file()} != {
        "manifest.json",
        "records.jsonl",
    }:
        raise ValueError(
            "quality-label directory must contain exactly manifest.json and records.jsonl"
        )
    manifest = _strict_json(root / "manifest.json")
    _verify_record(manifest, "quality-label manifest")
    if trusted_preflight is not None:
        if (
            trusted_preflight.get("advance") is not True
            or trusted_preflight.get("source_quality_manifest_sha256")
            != _sha256_file(root / "manifest.json")
            or trusted_preflight.get("source_quality_manifest_record_digest")
            != manifest.get("record_digest")
            or trusted_preflight.get("source_quality_records_sha256")
            != _sha256_file(root / "records.jsonl")
        ):
            raise ValueError("quality preflight does not authenticate this exact label artifact")
    base_manifest_fields = {
        "schema",
        "schema_version",
        "source_corpus_manifest_sha256",
        "quality_authority",
        "target_access",
        "ground_partition_receipt",
        "selector_digest",
        "normalizer_digest",
        "context",
        "context_digest",
        "partition",
        "record_count",
        "records_sha256",
        "lineages",
        "task_ids",
        "sampling_receipt",
        "independent_denominators",
        "label_protocol",
        "record_digest",
    }
    manifest_schema = manifest.get("schema")
    publication = _quality_manifest_is_publication(manifest)
    if not publication and not allow_diagnostic_legacy:
        raise ValueError(
            "legacy quality labels are diagnostic-only and require explicit diagnostic opt-in"
        )
    publication_fields = {"initializer_bank", "quality_resolution_plan"} if publication else set()
    if manifest_schema == QUALITY_SHARD_SCHEMA:
        if not _allow_shard:
            raise ValueError(
                "quality-label shards must be merged completely before preflight or training"
            )
        expected_manifest_fields = base_manifest_fields | {"sharding_receipt"} | publication_fields
    elif manifest_schema == QUALITY_MERGED_SCHEMA:
        expected_manifest_fields = base_manifest_fields | {"merge_receipt"} | publication_fields
    else:
        expected_manifest_fields = base_manifest_fields | publication_fields
    if set(manifest) != expected_manifest_fields:
        if manifest_schema == QUALITY_SCHEMA:
            raise ValueError("quality-label manifest fields differ from its exact schema")
        raise ValueError("quality-label manifest fields differ from its exact registered schema")
    if publication:
        bank_contract = manifest["initializer_bank"]
        resolution_plan_identity = manifest["quality_resolution_plan"]
        try:
            quality.validate_quality_initializer_bank_contract(
                bank_contract,
                require_publication=True,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(str(error)) from error
        if (
            not isinstance(resolution_plan_identity, Mapping)
            or set(resolution_plan_identity) != {"raw_sha256", "record_digest"}
        ):
            raise ValueError("quality labels have no exact resolution-plan identity")
        for name, value in resolution_plan_identity.items():
            _require_lower_sha256(value, f"quality-label resolution-plan {name}")
        if initializer_bank is None or expected_initializer_bank_manifest_sha256 is None:
            raise ValueError(
                "publication quality-label replay requires the live initializer bank and "
                "its external manifest pin"
            )
        try:
            live_bank_contract = quality.quality_initializer_bank_contract(
                initializer_bank,
                expected_manifest_sha256=expected_initializer_bank_manifest_sha256,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(str(error)) from error
        if live_bank_contract != bank_contract:
            raise ValueError("quality-label initializer bank differs from its manifest")
        if trusted_preflight is not None and (
            trusted_preflight.get("initializer_bank") != bank_contract
            or trusted_preflight.get("quality_resolution_plan") != resolution_plan_identity
        ):
            raise ValueError("quality preflight belongs to another initializer environment")
    else:
        if initializer_bank is not None or expected_initializer_bank_manifest_sha256 is not None:
            raise ValueError("diagnostic quality labels cannot consume a publication bank")
        _validate_quality_diagnostic_artifact(root, manifest=manifest)
    if manifest.get("source_corpus_manifest_sha256") != _corpus_manifest_digest(corpus):
        raise ValueError("quality labels belong to a different prepared corpus")
    if manifest.get("selector_digest") != selector.selector_digest:
        raise ValueError("quality labels were generated with a different selector")
    if manifest.get("normalizer_digest") != selector.normalizer_digest:
        raise ValueError("quality labels use a different feature normalizer")
    context_snapshot = _context_snapshot(context)
    expected_context_digest = stable_digest(context_snapshot)
    if (
        manifest.get("context") != context_snapshot
        or manifest.get("context_digest") != expected_context_digest
    ):
        raise ValueError("quality labels use a different environment context")
    if manifest.get("partition") != "train":
        raise ValueError("warm-start quality labels must belong to the prepared train partition")
    label_protocol = manifest.get("label_protocol")
    expected_protocol_fields = {
        "requested_lineages",
        "tasks_per_lineage_cap",
        "states_per_lineage_cap",
        "evaluated_actions",
        "continuations",
        "reward_reads",
        "seed",
        "implementation_contract",
    }
    if not isinstance(label_protocol, dict) or set(label_protocol) != expected_protocol_fields:
        raise ValueError("quality-label protocol fields differ from the registered schema")
    if label_protocol.get("implementation_contract") != _quality_implementation_contract(context):
        raise ValueError("quality labels use a different executable labeling protocol")
    for field_name in (
        "tasks_per_lineage_cap",
        "states_per_lineage_cap",
        "evaluated_actions",
        "continuations",
        "reward_reads",
    ):
        value = label_protocol.get(field_name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"quality-label protocol {field_name} must be positive")
    requested_lineages = label_protocol.get("requested_lineages")
    if (
        isinstance(requested_lineages, bool)
        or not isinstance(requested_lineages, int)
        or requested_lineages < 0
    ):
        raise ValueError("quality-label requested lineage count must be nonnegative")
    if label_protocol["reward_reads"] != context.n_est_reads:
        raise ValueError("quality-label reward reads differ from the registered context")
    label_seed = label_protocol.get("seed")
    if isinstance(label_seed, bool) or not isinstance(label_seed, int) or label_seed < 0:
        raise ValueError("quality-label seed must be a nonnegative integer")

    del require_provenance  # Prepared-v4 production loads enforce provenance unconditionally.
    if _prepared_train_tasks is not None:
        train_tasks = list(_prepared_train_tasks)
    else:
        if not isinstance(quality_attestation_pin, _VerifiedQualityAttestationPin):
            raise TypeError("quality-label loading requires a verified ground-root pin")
        loaded_tasks, _, _, _ = _load_quality_partition(
            corpus,
            partition="train",
            pin=quality_attestation_pin,
            role="training_partition",
        )
        train_tasks = list(loaded_tasks)
    task_by_id = {item.task_id: item for item in train_tasks}
    if len(task_by_id) != len(train_tasks):
        raise ValueError("prepared train corpus repeats a task identity")
    observed_authority = _quality_authority_identity(train_tasks, pin=quality_attestation_pin)
    if manifest.get("quality_authority") != observed_authority:
        raise ValueError("quality labels belong to a different pinned quality authority")
    target_access = manifest.get("target_access")
    ground_partition_receipt = manifest.get("ground_partition_receipt")
    if not isinstance(target_access, dict) or not isinstance(ground_partition_receipt, dict):
        raise ValueError("quality labels have no retained target-access receipts")
    _verify_record(target_access, "quality-label target access")
    _verify_record(ground_partition_receipt, "quality-label ground partition")
    partition_authority = observed_authority["training_partition"]
    if not isinstance(partition_authority, dict):
        raise RuntimeError("validated training authority lost its partition record")
    ground_identity = partition_authority.get("ground_partition")
    if (
        target_access.get("record_digest") != partition_authority.get("target_access_record_digest")
        or not isinstance(ground_identity, dict)
        or ground_partition_receipt.get("record_digest")
        != ground_identity.get("receipt_record_digest")
    ):
        raise ValueError("quality-label target access differs from its authority binding")
    plan = _quality_sampling_plan(
        train_tasks,
        lineage_count=requested_lineages,
        tasks_per_lineage=label_protocol["tasks_per_lineage_cap"],
        seed=label_seed,
    )
    global_selected_lineages = sorted(group[0].task.lineage for group in plan)
    global_active_groups = _quality_active_groups(plan, label_protocol["states_per_lineage_cap"])
    global_selected_tasks = sorted(item.task_id for group in global_active_groups for item in group)
    sampling = manifest.get("sampling_receipt")
    expected_sampling_fields = {
        "available_lineages",
        "requested_lineages",
        "selected_lineages",
        "selected_task_ids",
        "tasks_per_lineage_cap",
        "states_per_lineage_cap",
    }
    if not isinstance(sampling, dict) or set(sampling) != expected_sampling_fields:
        raise ValueError("quality sampling receipt differs from the exact v7 schema")
    expected_sampling = {
        "available_lineages": len({item.task.lineage for item in train_tasks}),
        "requested_lineages": requested_lineages,
        "selected_lineages": global_selected_lineages,
        "selected_task_ids": global_selected_tasks,
        "tasks_per_lineage_cap": label_protocol["tasks_per_lineage_cap"],
        "states_per_lineage_cap": label_protocol["states_per_lineage_cap"],
    }
    if sampling != expected_sampling:
        raise ValueError("quality sampling receipt cannot be reproduced from the train corpus")

    record_plan = plan
    if manifest_schema == QUALITY_SHARD_SCHEMA:
        sharding = manifest.get("sharding_receipt")
        sharding_fields = {
            "assignment",
            "shard_index",
            "shard_count",
            "global_sampling_plan_digest",
            "selected_lineages",
            "selected_task_ids",
        }
        if not isinstance(sharding, dict) or set(sharding) != sharding_fields:
            raise ValueError("quality sharding receipt differs from its exact schema")
        if sharding.get("assignment") != QUALITY_SHARD_ASSIGNMENT:
            raise ValueError("quality shard uses an unsupported lineage assignment")
        record_plan = _quality_plan_shard(
            plan,
            shard_index=sharding.get("shard_index"),
            shard_count=sharding.get("shard_count"),
        )
        expected_sharding = _quality_sharding_receipt(
            expected_sampling,
            record_plan,
            shard_index=sharding["shard_index"],
            shard_count=sharding["shard_count"],
            states_per_lineage=label_protocol["states_per_lineage_cap"],
        )
        if sharding != expected_sharding:
            raise ValueError(
                "quality shard lineage subset cannot be reproduced from the global plan"
            )
    elif manifest_schema == QUALITY_MERGED_SCHEMA:
        merge_receipt = manifest.get("merge_receipt")
        merge_fields = {
            "assignment",
            "shard_count",
            "global_sampling_plan_digest",
            "source_shards",
        }
        if not isinstance(merge_receipt, dict) or set(merge_receipt) != merge_fields:
            raise ValueError("quality merge receipt differs from its exact schema")
        if merge_receipt.get("assignment") != QUALITY_SHARD_ASSIGNMENT:
            raise ValueError("quality merge uses an unsupported lineage assignment")
        shard_count = merge_receipt.get("shard_count")
        sources = merge_receipt.get("source_shards")
        if (
            isinstance(shard_count, bool)
            or not isinstance(shard_count, int)
            or shard_count <= 0
            or shard_count > len(plan)
            or not isinstance(sources, list)
            or len(sources) != shard_count
        ):
            raise ValueError("quality merge does not contain a complete shard census")
        if merge_receipt.get("global_sampling_plan_digest") != content_digest(expected_sampling):
            raise ValueError("quality merge refers to a different global sampling plan")
        source_fields = {
            "shard_index",
            "manifest_sha256",
            "manifest_record_digest",
            "records_sha256",
            "record_count",
            "selected_lineages",
            "selected_task_ids",
        }
        covered_lineages: set[str] = set()
        covered_tasks: set[str] = set()
        source_indices: list[int] = []
        source_record_count = 0
        for source in sources:
            if not isinstance(source, dict) or set(source) != source_fields:
                raise ValueError("quality merge source-shard receipt differs from its exact schema")
            shard_index = source.get("shard_index")
            if (
                isinstance(shard_index, bool)
                or not isinstance(shard_index, int)
                or shard_index < 0
                or shard_index >= shard_count
            ):
                raise ValueError("quality merge source has an invalid shard index")
            source_indices.append(shard_index)
            expected_shard_plan = _quality_plan_shard(
                plan, shard_index=shard_index, shard_count=shard_count
            )
            expected_source = _quality_sharding_receipt(
                expected_sampling,
                expected_shard_plan,
                shard_index=shard_index,
                shard_count=shard_count,
                states_per_lineage=label_protocol["states_per_lineage_cap"],
            )
            if source.get("selected_lineages") != expected_source["selected_lineages"]:
                raise ValueError("quality merge source lineage subset is not reproducible")
            if source.get("selected_task_ids") != expected_source["selected_task_ids"]:
                raise ValueError("quality merge source task subset is not reproducible")
            source_lineages = set(source["selected_lineages"])
            if covered_lineages & source_lineages:
                raise ValueError("quality merge source shards duplicate independent lineages")
            covered_lineages.update(source_lineages)
            source_tasks = set(source["selected_task_ids"])
            if covered_tasks & source_tasks:
                raise ValueError("quality merge source shards duplicate selected tasks")
            covered_tasks.update(source_tasks)
            record_count = source.get("record_count")
            if (
                isinstance(record_count, bool)
                or not isinstance(record_count, int)
                or record_count <= 0
            ):
                raise ValueError("quality merge source has an invalid record count")
            source_record_count += record_count
            digests = (
                source.get("manifest_sha256"),
                source.get("manifest_record_digest"),
                source.get("records_sha256"),
            )
            if any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in digests
            ):
                raise ValueError("quality merge source has invalid authenticated digests")
        if source_indices != list(range(shard_count)):
            raise ValueError("quality merge source shards are missing, repeated, or out of order")
        if covered_lineages != set(global_selected_lineages):
            raise ValueError("quality merge source shards do not cover the global lineage plan")
        if covered_tasks != set(global_selected_tasks):
            raise ValueError("quality merge source shards do not cover the global task plan")
        merged_record_count = manifest.get("record_count")
        if (
            isinstance(merged_record_count, bool)
            or not isinstance(merged_record_count, int)
            or source_record_count != merged_record_count
        ):
            raise ValueError("quality merge source record counts do not match the merged corpus")
    expected_selected_lineages = sorted(group[0].task.lineage for group in record_plan)
    active_groups = _quality_active_groups(record_plan, label_protocol["states_per_lineage_cap"])
    expected_selected_tasks = sorted(item.task_id for group in active_groups for item in group)
    expected_selected_lineage_set = frozenset(expected_selected_lineages)
    expected_selected_task_set = frozenset(expected_selected_tasks)

    record_count = manifest.get("record_count")
    if isinstance(record_count, bool) or not isinstance(record_count, int) or record_count <= 0:
        raise ValueError("quality-label manifest has an invalid record count")
    if _record_selection is None:
        record_index = _build_quality_jsonl_index(root, manifest)
        record_selection = record_index.select(_replay_lineages)
    else:
        record_selection = _record_selection
        if (
            record_selection.records_sha256 != manifest.get("records_sha256")
            or record_selection.total_record_count != record_count
        ):
            raise ValueError("quality record selection belongs to another artifact")
        if _replay_lineages is None:
            if not record_selection.covers_entire_file:
                raise ValueError("full quality replay requires a complete record selection")
        elif record_selection.lineages != _replay_lineages:
            raise ValueError("partial quality replay selection differs from its lineage assignment")
    decoded: list[tuple[object, tuple[int, ...], tuple[int, ...], object]] = []
    file_total_records = record_selection.total_record_count
    total_records = 0
    resolved_records = 0
    unresolved_records = 0
    record_lineages: set[str] = set()
    record_tasks: set[str] = set()
    resolved_lineages: set[str] = set()
    resolved_tasks: set[str] = set()
    seen_states: set[tuple[str, str]] = set()
    authenticated_row_digests: set[str] = set()
    seen_continuation_seeds: set[int] = set()
    continuation_trajectories = 0
    continuation_replays_executed = 0
    continuation_replay_matches = 0
    valid_continuations = 0
    records_per_lineage: dict[str, int] = {}
    row_fields = {
        "schema",
        "schema_version",
        "task_id",
        "instance_id",
        "lineage",
        "partition",
        "initializer_record_digest",
        "reference_status",
        "certificate_digest",
        "evaluator_protocol_digest",
        "environment_seed",
        "prefix",
        "support",
        "evaluated",
        "best_actions",
        "continuation_policy",
        "action_envelope_record_digest",
        "state_fingerprint",
        "support_fingerprint",
        "context_version",
        "context_digest",
        "charged_work_receipt",
        "exact_state",
        "observation",
        "label_version",
        "record_digest",
    }
    action_fields = {field.name for field in dataclasses.fields(quality.ActionQuality)}
    for line_number, record in _iter_quality_jsonl_selection(root, record_selection):
        if set(record) != row_fields:
            raise ValueError(f"quality record line {line_number} differs from the exact v7 schema")
        if (
            record["schema"] != QUALITY_ROW_SCHEMA
            or record["schema_version"] != QUALITY_ROW_SCHEMA_VERSION
        ):
            raise ValueError("unsupported quality counterfactual row schema")
        implementation = label_protocol["implementation_contract"]
        if (
            record.get("label_version") != implementation["label_version"]
            or record.get("continuation_policy")
            != implementation["continuation_policy"]["policy_id"]
        ):
            raise ValueError("quality record uses another label or continuation-policy version")
        task_id = record["task_id"]
        if not isinstance(task_id, str) or task_id not in task_by_id:
            raise ValueError("quality record task does not belong to the prepared train corpus")
        item = task_by_id[task_id]
        lineage = record["lineage"]
        identity = {
            "instance_id": item.instance_id,
            "lineage": item.task.lineage,
            "partition": item.partition,
            "initializer_record_digest": item.initializer_record_digest,
            "reference_status": item.reference_status,
            "certificate_digest": item.certificate_digest,
            "evaluator_protocol_digest": item.evaluator_protocol_digest,
        }
        if any(record[field] != value for field, value in identity.items()):
            raise ValueError("quality record lineage or prepared-task identities disagree")
        if task_id not in expected_selected_task_set or lineage not in expected_selected_lineage_set:
            raise ValueError("quality record is outside the authenticated sampling plan")
        if _replay_lineages is not None and lineage not in _replay_lineages:
            continue
        total_records += 1
        if _trusted_replay_rows is not None:
            authenticated_row_digests.add(str(record["record_digest"]))
        environment_seed = record["environment_seed"]
        if type(environment_seed) is not int or not 0 <= environment_seed < 2**63:
            raise ValueError("quality record environment seed is malformed")
        if not publication and environment_seed != _domain_seed(
            label_seed, "quality-environment", lineage, task_id
        ):
            raise ValueError("quality record environment seed differs from the sampling registry")
        prefix = record["prefix"]
        if not isinstance(prefix, list) or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in prefix
        ):
            raise ValueError("quality record prefix is not a nonnegative action-index sequence")
        if (
            record["context_version"] != context.context_version
            or record["context_digest"] != expected_context_digest
        ):
            raise ValueError("quality record context identity differs from its manifest")
        initializer_episode_index = None
        expected_initializer_binding = None
        if publication:
            raw_actions = record.get("evaluated")
            if not isinstance(raw_actions, list):
                raise ValueError("publication quality record has no evaluated action list")
            raw_receipts = [
                receipt
                for raw_action in raw_actions
                if isinstance(raw_action, Mapping)
                for receipt in raw_action.get("continuation_receipts", ())
            ]
            if not raw_receipts:
                raise ValueError("publication quality record has no continuation receipts")
            for receipt in raw_receipts:
                quality.validate_continuation_receipt(receipt)
                binding = receipt["initializer_binding"]
                quality.validate_quality_initializer_binding(binding)
                if expected_initializer_binding is None:
                    expected_initializer_binding = dict(binding)
                elif dict(binding) != expected_initializer_binding:
                    raise ValueError(
                        "publication quality record mixes initializer-bank episode bindings"
                    )
            initializer_episode_index = expected_initializer_binding[
                "episode_schedule_index"
            ]
        if publication:
            from isingfold.rl.data.quality_resolution_plan import (
                quality_action_provenance_fingerprint,
            )

            provenance_fingerprint = quality_action_provenance_fingerprint(
                item,
                manifest["source_corpus_manifest_sha256"],
            )
        else:
            provenance_fingerprint = _quality_action_provenance_fingerprint(
                item,
                source_corpus_manifest_sha256=manifest["source_corpus_manifest_sha256"],
            )
        bound_replay = quality.replay_decision_with_action_envelope(
            item.task,
            context,
            initializer=(None if publication else item.initializer()),
            initializer_bank=(initializer_bank if publication else None),
            expected_initializer_bank_manifest_sha256=(
                expected_initializer_bank_manifest_sha256 if publication else None
            ),
            initializer_bank_episode_index=initializer_episode_index,
            prepared_task=(item if publication else None),
            target_access=(target_access if publication else None),
            ground_partition_receipt=(ground_partition_receipt if publication else None),
            selector=selector.model,
            prefix=prefix,
            seed=environment_seed,
            reward_reads=label_protocol["reward_reads"],
            provenance_fingerprint=provenance_fingerprint,
        )
        if bound_replay is None:
            raise ValueError("quality record prefix cannot be replayed to a decision state")
        replayed, action_envelope = bound_replay
        if record["action_envelope_record_digest"] != action_envelope.record_digest:
            raise ValueError("quality record differs from its exact state/action envelope")
        expected_support, expected_exact_state, expected_observation = quality.decision_snapshot(
            replayed
        )
        if content_digest(record["support"]) != content_digest(list(expected_support)):
            raise ValueError("quality record support differs from exact replay")
        if content_digest(record["exact_state"]) != content_digest(expected_exact_state):
            raise ValueError("quality record exact state differs from exact replay")
        if content_digest(record["observation"]) != content_digest(expected_observation):
            raise ValueError("quality record observation differs from exact replay")
        if (
            record["state_fingerprint"] != replayed.state_fingerprint
            or record["support_fingerprint"] != replayed.support_fingerprint
        ):
            raise ValueError("quality record state/support fingerprints differ from exact replay")
        if record["charged_work_receipt"] != replayed.charged_work_receipt.as_dict():
            raise ValueError("quality record charged-work receipt differs from exact replay")
        state_key = (task_id, replayed.state_fingerprint)
        if state_key in seen_states:
            raise ValueError("quality-label corpus repeats one task/state counterfactual")
        seen_states.add(state_key)
        observation = _decode_observation(record["observation"])
        evaluated_raw = record.get("evaluated")
        best_raw = record.get("best_actions")
        if not isinstance(evaluated_raw, list) or not isinstance(best_raw, list):
            raise ValueError("quality record lacks evaluated and best-action sets")
        evaluated_actions: list[quality.ActionQuality] = []
        expected_receipt_identities: list[dict[str, object]] = []
        replayed_receipt_identities: list[dict[str, object]] = []
        row_replay_matches = 0
        for raw_action in evaluated_raw:
            if not isinstance(raw_action, dict) or set(raw_action) != action_fields:
                raise ValueError("quality action label differs from the exact v7 schema")
            numeric = (raw_action["q_mu"], raw_action["inclusion_probability"])
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in numeric
            ):
                raise ValueError("quality action has a non-finite scalar")
            if not isinstance(raw_action["continuation_seeds"], list):
                raise ValueError("quality continuation seeds must be an array")
            if not isinstance(raw_action["continuation_rewards"], list):
                raise ValueError("quality continuation rewards must be an array")
            if not isinstance(raw_action["continuation_valid"], list):
                raise ValueError("quality continuation-valid flags must be an array")
            if not isinstance(raw_action["continuation_receipts"], list):
                raise ValueError("quality continuation receipts must be an array")
            action = quality.ActionQuality(
                action_index=raw_action["action_index"],
                payload_key=raw_action["payload_key"],
                opcode=raw_action["opcode"],
                q_mu=float(raw_action["q_mu"]),
                continuations=raw_action["continuations"],
                valid_returns=raw_action["valid_returns"],
                inclusion_probability=float(raw_action["inclusion_probability"]),
                selected_payload_digest=raw_action["selected_payload_digest"],
                applied_action_record_digest=raw_action["applied_action_record_digest"],
                continuation_seeds=tuple(raw_action["continuation_seeds"]),
                continuation_rewards=tuple(raw_action["continuation_rewards"]),
                continuation_valid=tuple(raw_action["continuation_valid"]),
                continuation_receipts=tuple(raw_action["continuation_receipts"]),
            )
            action.validate()
            evaluated_actions.append(action)
        if publication:
            if expected_initializer_binding is None:
                raise RuntimeError("publication initializer binding disappeared during replay")
            publication_record = quality.QualityRecord(
                instance=task_id,
                lineage=lineage,
                prefix=tuple(prefix),
                support=tuple(record["support"]),
                evaluated=tuple(evaluated_actions),
                continuation_policy=record["continuation_policy"],
                action_envelope_record_digest=record["action_envelope_record_digest"],
                state_fingerprint=record["state_fingerprint"],
                support_fingerprint=record["support_fingerprint"],
                context_version=record["context_version"],
                charged_work_receipt=record["charged_work_receipt"],
                exact_state=record["exact_state"],
                observation=record["observation"],
                environment_seed=environment_seed,
                context_digest=record["context_digest"],
                label_version=record["label_version"],
            )
            binding = quality.validate_publication_quality_record_initializer_binding(
                publication_record,
                expected_initializer_bank_contract=manifest["initializer_bank"],
                expected_initializer_binding=expected_initializer_binding,
            )
            if binding["episode_schedule_index"] != initializer_episode_index:
                raise ValueError("publication quality row changed initializer episode")
        evaluated = tuple(action.action_index for action in evaluated_actions)
        if not evaluated or tuple(sorted(set(evaluated))) != evaluated:
            raise ValueError("quality evaluated action indices are empty, repeated, or unsorted")
        action_sample = quality.deterministic_action_sample(
            replayed,
            evaluated_actions=label_protocol["evaluated_actions"],
            seed=environment_seed,
        )
        expected_evaluated = action_sample.action_indices
        if evaluated != expected_evaluated:
            raise ValueError(
                "quality evaluated-action subset differs from the registered deterministic draw"
            )
        if not isinstance(best_raw, list) or any(
            isinstance(index, bool) or not isinstance(index, int) for index in best_raw
        ):
            raise ValueError("quality best-action set is not an integer index list")
        best = tuple(best_raw)
        if not best or tuple(sorted(set(best))) != best or not set(best) <= set(evaluated):
            raise ValueError("quality record has an invalid tied-best action target")
        if any(index < 0 or index >= observation.n_actions for index in evaluated):
            raise ValueError("quality label action is outside its stored support")
        if any(not observation.legal_mask[index] for index in evaluated):
            raise ValueError("quality label evaluates a masked action")
        legal_count = sum(bool(value) for value in replayed.legal_mask)
        expected_evaluated_count = min(label_protocol["evaluated_actions"], legal_count)
        if len(evaluated_actions) != expected_evaluated_count:
            raise ValueError("quality evaluated-action count differs from its registered cap")
        for action in evaluated_actions:
            support_row = expected_support[action.action_index]
            bound_action = action_envelope.candidates[action.action_index]
            expected_applied_digest = None
            if not bound_action.opcode.is_terminal:
                expected_applied = apply_envelope_action(action_envelope, action.action_index)
                expected_applied.verify_against(action_envelope)
                expected_applied_digest = expected_applied.record_digest
            if (
                action.payload_key != support_row["payload_key"]
                or action.opcode != support_row["opcode"]
                or support_row["legal"] is not True
                or action.selected_payload_digest != bound_action.payload_digest
                or action.applied_action_record_digest != expected_applied_digest
            ):
                raise ValueError(
                    "quality action payload/opcode/support or successor binding disagrees"
                )
            if action.continuations != label_protocol["continuations"]:
                raise ValueError("quality action continuation count differs from its protocol")
            if not math.isclose(
                action.inclusion_probability,
                action_sample.inclusion_probability(action.action_index),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError("quality action inclusion probability is not reproducible")
            expected_seeds = tuple(
                quality.continuation_seed(
                    environment_seed,
                    task_id,
                    replayed.state_fingerprint,
                    action.action_index,
                    continuation_index,
                )
                for continuation_index in range(action.continuations)
            )
            if action.continuation_seeds != expected_seeds:
                raise ValueError("quality continuation seed registry cannot be reproduced")
            for continuation_index, continuation_seed in enumerate(action.continuation_seeds):
                if continuation_seed in seen_continuation_seeds:
                    raise ValueError("quality continuation seed is reused across the corpus")
                seen_continuation_seeds.add(continuation_seed)
                expected_receipt_identities.append(
                    {
                        "action_index": action.action_index,
                        "continuation_index": continuation_index,
                        "receipt_digest": content_digest(
                            action.continuation_receipts[continuation_index]
                        ),
                        "seed": continuation_seed,
                    }
                )
                if trusted_preflight is None and _trusted_replay_rows is None:
                    replayed_outcome = quality.run_continuation(
                        item.task,
                        context,
                        initializer=(None if publication else item.initializer()),
                        initializer_bank=(initializer_bank if publication else None),
                        expected_initializer_bank_manifest_sha256=(
                            expected_initializer_bank_manifest_sha256
                            if publication
                            else None
                        ),
                        initializer_bank_episode_index=initializer_episode_index,
                        prepared_task=(item if publication else None),
                        target_access=(target_access if publication else None),
                        ground_partition_receipt=(
                            ground_partition_receipt if publication else None
                        ),
                        selector=selector.model,
                        prefix=(*prefix, action.action_index),
                        seed=environment_seed,
                        continuation_seed=continuation_seed,
                        reward_reads=label_protocol["reward_reads"],
                        max_steps=quality.CONTINUATION_MAX_STEPS,
                    )
                    continuation_replays_executed += 1
                    if replayed_outcome is None or replayed_outcome.receipt is None:
                        raise ValueError("quality continuation cannot be exactly replayed")
                    if content_digest(replayed_outcome.receipt) != content_digest(
                        action.continuation_receipts[continuation_index]
                    ):
                        raise ValueError("quality continuation outcome differs from exact replay")
                    continuation_replay_matches += 1
                    row_replay_matches += 1
                    replayed_receipt_identities.append(
                        {
                            "action_index": action.action_index,
                            "continuation_index": continuation_index,
                            "receipt_digest": content_digest(replayed_outcome.receipt),
                            "seed": continuation_seed,
                        }
                    )
            continuation_trajectories += action.continuations
            valid_continuations += action.valid_returns
        expected_receipt_root = content_digest(expected_receipt_identities)
        if _trusted_replay_rows is not None:
            trusted = _trusted_replay_rows.get(str(record["record_digest"]))
            if not isinstance(trusted, Mapping):
                raise ValueError("trusted replay bundle omits a quality source row")
            trusted_fields = {
                "expected_continuation_root_digest",
                "lineage",
                "match_count",
                "pass",
                "record_digest",
                "replayed_continuation_root_digest",
                "schema",
                "schema_version",
                "source_row_record_digest",
                "task_id",
                "trajectory_count",
            }
            row_trajectory_count = len(expected_receipt_identities)
            if (
                set(trusted) != trusted_fields
                or trusted.get("source_row_record_digest") != record["record_digest"]
                or trusted.get("lineage") != lineage
                or trusted.get("task_id") != task_id
                or trusted.get("trajectory_count") != row_trajectory_count
                or trusted.get("match_count") != row_trajectory_count
                or trusted.get("expected_continuation_root_digest") != expected_receipt_root
                or trusted.get("replayed_continuation_root_digest") != expected_receipt_root
                or trusted.get("pass") is not True
            ):
                raise ValueError("trusted replay evidence differs from the exact quality row")
            _verify_record(trusted, "trusted quality replay row")
            continuation_replay_matches += row_trajectory_count
        elif _replay_evidence is not None:
            replayed_receipt_root = content_digest(replayed_receipt_identities)
            _replay_evidence.append(
                {
                    "expected_continuation_root_digest": expected_receipt_root,
                    "lineage": lineage,
                    "match_count": row_replay_matches,
                    "pass": (
                        row_replay_matches == len(expected_receipt_identities)
                        and replayed_receipt_root == expected_receipt_root
                    ),
                    "replayed_continuation_root_digest": replayed_receipt_root,
                    "source_row_record_digest": record["record_digest"],
                    "task_id": task_id,
                    "trajectory_count": len(expected_receipt_identities),
                }
            )
        recomputed_best = quality.QualityRecord(
            instance=task_id,
            lineage=lineage,
            prefix=tuple(prefix),
            support=tuple(expected_support),
            evaluated=tuple(evaluated_actions),
            continuation_policy=record["continuation_policy"],
            action_envelope_record_digest=action_envelope.record_digest,
        ).best_actions()
        if best != recomputed_best:
            raise ValueError("quality plausible-best set differs from simultaneous uncertainty")
        protected_commit_indices = tuple(
            action.action_index
            for action in evaluated_actions
            if action.opcode == "COMMIT"
            and action_envelope.candidates[action.action_index].archive_ref == 0
        )
        if len(protected_commit_indices) > 1:
            raise ValueError("quality row evaluates multiple protected COMMIT anchors")
        action_value_target = WarmStartActionValueTarget(
            action_indices=evaluated,
            q_mu=tuple(action.q_mu for action in evaluated_actions),
            continuation_counts=tuple(action.continuations for action in evaluated_actions),
            inclusion_probabilities=tuple(
                action.inclusion_probability for action in evaluated_actions
            ),
            legal_action_count=legal_count,
            commit_index=(protected_commit_indices[0] if protected_commit_indices else None),
            support_size=observation.n_actions,
        )
        utility_target = warm_start_utility_target(
            q_mu=[action.q_mu for action in evaluated_actions],
            continuation_counts=[action.continuations for action in evaluated_actions],
            inclusion_probabilities=[action.inclusion_probability for action in evaluated_actions],
            action_value_target=action_value_target,
            legal_action_count=legal_count,
        )
        record_lineages.add(lineage)
        record_tasks.add(task_id)
        records_per_lineage[lineage] = records_per_lineage.get(lineage, 0) + 1
        # Every authenticated target calibrates the state-utility critic.  A fully
        # unresolved plausible-best set has no actor preference and is masked by the
        # warm-start ranking loss, rather than being discarded here.
        if _collect_decoded:
            decoded.append((observation, best, evaluated, utility_target))
        if set(best) == set(evaluated):
            unresolved_records += 1
            continue
        resolved_records += 1
        resolved_lineages.add(lineage)
        resolved_tasks.add(task_id)
    if file_total_records != record_count:
        raise ValueError("quality-label record count differs from its manifest")
    if any(
        count > label_protocol["states_per_lineage_cap"] for count in records_per_lineage.values()
    ):
        raise ValueError("quality records exceed the states-per-lineage cap")
    if _replay_lineages is not None and record_lineages != set(_replay_lineages):
        raise ValueError("partial replay lineage assignment has missing records")
    if _replay_lineages is None and manifest.get("lineages") != sorted(record_lineages):
        raise ValueError("quality-label lineage census differs from its records")
    if _replay_lineages is None and manifest.get("task_ids") != sorted(record_tasks):
        raise ValueError("quality-label task census differs from its records")
    denominators = manifest.get("independent_denominators")
    expected_denominator_fields = {
        "selected_lineages",
        "selected_tasks",
        "lineages_with_records",
        "tasks_with_records",
        "records_total",
        "resolved_rows",
        "fully_unresolved_rows",
        "resolved_lineages",
        "continuation_trajectories",
        "valid_continuations",
    }
    if not isinstance(denominators, dict) or set(denominators) != expected_denominator_fields:
        raise ValueError("quality independent denominators differ from the exact v7 schema")
    expected_denominators = {
        "selected_lineages": len(expected_selected_lineages),
        "selected_tasks": len(expected_selected_tasks),
        "lineages_with_records": len(record_lineages),
        "tasks_with_records": len(record_tasks),
        "records_total": total_records,
        "resolved_rows": resolved_records,
        "fully_unresolved_rows": unresolved_records,
        "resolved_lineages": len(resolved_lineages),
        "continuation_trajectories": continuation_trajectories,
        "valid_continuations": valid_continuations,
    }
    if _replay_lineages is None and denominators != expected_denominators:
        raise ValueError("quality independent denominators cannot be recomputed")
    if _trusted_replay_rows is not None and set(_trusted_replay_rows) != authenticated_row_digests:
        raise ValueError("trusted replay evidence row coverage differs from the quality artifact")
    loaded_manifest = dict(manifest)
    # Checkpoints must bind every lineage that contributes critic supervision, not only
    # lineages that also contain a resolved actor-ranking row.
    loaded_manifest["lineages"] = sorted(record_lineages)
    summary = {
        "records_total": total_records,
        # This established preflight field remains the actor-resolution denominator.
        # ``decoded`` now contains all authenticated critic rows.
        "records_used": resolved_records,
        "fully_unresolved_skipped": unresolved_records,
        "label_lineages": len(record_lineages),
        "resolved_lineages": len(resolved_lineages),
        "resolved_tasks": len(resolved_tasks),
        "unresolved_lineages": sorted(record_lineages - resolved_lineages),
        "minimum_resolved_rows": min_resolved_rows,
        "minimum_resolved_lineages": min_resolved_lineages,
        "passes_resolution": (
            resolved_records >= min_resolved_rows
            and len(resolved_lineages) >= min_resolved_lineages
        ),
        "continuation_trajectories_authenticated": continuation_trajectories,
        "continuation_replays_executed": continuation_replays_executed,
        "continuation_replay_matches": continuation_replay_matches,
        "continuation_replay_mode": (
            "pinned-full-replay-receipt"
            if trusted_preflight is not None
            else (
                "pinned-quality-replay-bundle"
                if _trusted_replay_rows is not None
                else "full-exact-continuation-and-evaluator"
            )
        ),
    }
    loaded_manifest["loader_summary"] = summary
    if enforce_resolution and not summary["passes_resolution"]:
        raise ValueError(
            "quality resolution preflight failed: "
            f"resolved rows {resolved_records}/{total_records} "
            f"(required {min_resolved_rows}); "
            f"resolved independent lineages {len(resolved_lineages)}/{len(record_lineages)} "
            f"(required {min_resolved_lineages}). Increase the preregistered continuation "
            "count or expand the train-lineage sample; do not invent a winner."
        )
    return decoded, loaded_manifest


def _load_quality_preflight_receipt(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    quality_labels: str | os.PathLike[str],
    corpus: str | os.PathLike[str],
    selector: SelectorBundle,
    context: Context,
    min_resolved_rows: int,
    min_resolved_lineages: int,
    allow_diagnostic_legacy: bool = False,
) -> dict[str, Any]:
    """Authenticate the one full stochastic replay that all scientific cells reuse."""

    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
    ):
        raise ValueError("expected quality-preflight SHA-256 is not a lowercase digest")
    receipt_path = Path(path)
    if not hmac.compare_digest(_sha256_file(receipt_path), expected_sha256):
        raise ValueError("quality-preflight receipt differs from its externally pinned SHA-256")
    receipt = _strict_json(receipt_path)
    _verify_record(receipt, "quality-preflight receipt")
    label_root = Path(quality_labels)
    manifest = _strict_json(label_root / "manifest.json")
    _verify_record(manifest, "quality-label manifest pinned by preflight")
    publication = _quality_manifest_is_publication(manifest)
    if not publication and not allow_diagnostic_legacy:
        raise ValueError("diagnostic quality preflight requires explicit diagnostic opt-in")
    expected_fields = {
        "schema",
        "schema_version",
        "source_quality_manifest_sha256",
        "source_quality_manifest_record_digest",
        "source_quality_records_sha256",
        "source_corpus_manifest_sha256",
        "quality_authority",
        "selector_digest",
        "normalizer_digest",
        "context",
        "context_digest",
        "implementation_contract",
        "label_protocol_digest",
        "minimum_resolved_rows",
        "minimum_resolved_lineages",
        "full_replay",
        "resolution",
        "advance",
        "record_digest",
    }
    if publication:
        expected_fields |= {"initializer_bank", "quality_resolution_plan"}
    if set(receipt) != expected_fields:
        raise ValueError("quality-preflight receipt differs from its exact schema")
    current_contract = _quality_implementation_contract(context)
    expected_preflight_version = (
        QUALITY_PREFLIGHT_SCHEMA_VERSION
        if publication
        else QUALITY_DIAGNOSTIC_PREFLIGHT_SCHEMA_VERSION
    )
    if (
        receipt.get("schema") != QUALITY_PREFLIGHT_SCHEMA
        or receipt.get("schema_version") != expected_preflight_version
        or receipt.get("advance") is not True
        or receipt.get("source_quality_manifest_sha256")
        != _sha256_file(label_root / "manifest.json")
        or receipt.get("source_quality_manifest_record_digest") != manifest.get("record_digest")
        or receipt.get("source_quality_records_sha256")
        != _sha256_file(label_root / "records.jsonl")
        or receipt.get("source_corpus_manifest_sha256") != _corpus_manifest_digest(corpus)
        or receipt.get("quality_authority") != selector.quality_authority
        or receipt.get("selector_digest") != selector.selector_digest
        or receipt.get("normalizer_digest") != selector.normalizer_digest
        or receipt.get("context") != _context_snapshot(context)
        or receipt.get("context_digest") != stable_digest(_context_snapshot(context))
        or receipt.get("implementation_contract") != current_contract
        or receipt.get("label_protocol_digest") != stable_digest(manifest.get("label_protocol"))
        or receipt.get("minimum_resolved_rows") != min_resolved_rows
        or receipt.get("minimum_resolved_lineages") != min_resolved_lineages
        or publication
        and (
            receipt.get("initializer_bank") != manifest.get("initializer_bank")
            or receipt.get("quality_resolution_plan")
            != manifest.get("quality_resolution_plan")
        )
    ):
        raise ValueError("quality-preflight receipt belongs to another scientific experiment")
    replay = receipt.get("full_replay")
    resolution = receipt.get("resolution")
    denominators = manifest.get("independent_denominators")
    if not isinstance(denominators, dict):
        raise ValueError("quality-label manifest has no authenticated denominator registry")
    if (
        not isinstance(replay, dict)
        or set(replay)
        != {
            "mode",
            "continuation_trajectories",
            "continuation_replays_executed",
            "continuation_replay_matches",
        }
        or replay.get("mode") != "full-exact-continuation-and-evaluator"
        or not isinstance(resolution, dict)
        or resolution.get("passes_resolution") is not True
        or replay.get("continuation_trajectories") != denominators.get("continuation_trajectories")
        or replay.get("continuation_replays_executed") != replay.get("continuation_trajectories")
        or replay.get("continuation_replay_matches") != replay.get("continuation_trajectories")
    ):
        raise ValueError("quality-preflight receipt does not prove one complete exact replay")
    return receipt


def _quality_preflight_process_worker(payload: Mapping[str, object]) -> dict[str, Any]:
    """Fresh-process entry point for one invocation-scoped whole-lineage replay."""

    raw_args = payload.get("args")
    raw_lineages = payload.get("lineages")
    raw_selection = payload.get("record_selection")
    if (
        not isinstance(raw_args, Mapping)
        or not isinstance(raw_lineages, list)
        or not isinstance(raw_selection, _QualityJsonlSelection)
    ):
        raise ValueError("quality-preflight worker payload is malformed")
    args = argparse.Namespace(**dict(raw_args))
    lineages = frozenset(str(lineage) for lineage in raw_lineages)
    if raw_selection.lineages != lineages:
        raise ValueError("quality-preflight worker ranges differ from its lineage assignment")
    ctx = _context(args, args.corpus)
    bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
    source_manifest = _strict_json(Path(args.quality_labels) / "manifest.json")
    initializer_bank, initializer_bank_manifest_sha256 = (
        _quality_initializer_bank_for_manifest(
            args,
            manifest=source_manifest,
            context=ctx,
        )
    )
    _seed_runtime(args.seed, deterministic=args.deterministic, threads=1)
    bundle.model.to(_resolve_device(args.device)).eval()
    quality_pin = _quality_attestation_pin(args)
    _records, manifest = _load_quality_labels(
        args.quality_labels,
        corpus=args.corpus,
        selector=bundle,
        context=ctx,
        min_resolved_rows=args.min_resolved_rows,
        min_resolved_lineages=args.min_resolved_lineages,
        enforce_resolution=False,
        require_provenance=not args.allow_legacy_pilot,
        quality_attestation_pin=quality_pin,
        _replay_lineages=lineages,
        _record_selection=raw_selection,
        _collect_decoded=False,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=(initializer_bank_manifest_sha256),
        allow_diagnostic_legacy=bool(getattr(args, "allow_legacy_pilot", False)),
    )
    if bundle.quality_authority != manifest["quality_authority"]:
        raise ValueError("quality-preflight worker selector and labels use different authorities")
    summary = manifest["loader_summary"]
    if summary["continuation_replay_mode"] != "full-exact-continuation-and-evaluator":
        raise ValueError("quality-preflight child did not perform a fresh exact replay")
    return summary


def _aggregate_quality_preflight_summaries(
    summaries: Sequence[Mapping[str, Any]],
    *,
    manifest: Mapping[str, Any],
    min_resolved_rows: int,
    min_resolved_lineages: int,
) -> dict[str, Any]:
    """Reconstruct the strict serial summary from disjoint whole-lineage children."""

    if not summaries:
        raise ValueError("parallel quality preflight produced no worker summaries")
    expected_fields = {
        "records_total",
        "records_used",
        "fully_unresolved_skipped",
        "label_lineages",
        "resolved_lineages",
        "resolved_tasks",
        "unresolved_lineages",
        "minimum_resolved_rows",
        "minimum_resolved_lineages",
        "passes_resolution",
        "continuation_trajectories_authenticated",
        "continuation_replays_executed",
        "continuation_replay_matches",
        "continuation_replay_mode",
    }
    if any(set(summary) != expected_fields for summary in summaries):
        raise ValueError("quality-preflight worker summary fields differ")
    unresolved = [
        str(lineage) for summary in summaries for lineage in summary["unresolved_lineages"]
    ]
    if len(unresolved) != len(set(unresolved)):
        raise ValueError("quality-preflight workers overlap an unresolved lineage")

    def total(field: str) -> int:
        values = [summary[field] for summary in summaries]
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise ValueError(f"quality-preflight worker {field} is malformed")
        return sum(values)

    result = {
        "records_total": total("records_total"),
        "records_used": total("records_used"),
        "fully_unresolved_skipped": total("fully_unresolved_skipped"),
        "label_lineages": total("label_lineages"),
        "resolved_lineages": total("resolved_lineages"),
        "resolved_tasks": total("resolved_tasks"),
        "unresolved_lineages": sorted(unresolved),
        "minimum_resolved_rows": min_resolved_rows,
        "minimum_resolved_lineages": min_resolved_lineages,
        "continuation_trajectories_authenticated": total("continuation_trajectories_authenticated"),
        "continuation_replays_executed": total("continuation_replays_executed"),
        "continuation_replay_matches": total("continuation_replay_matches"),
        "continuation_replay_mode": "full-exact-continuation-and-evaluator",
    }
    result["passes_resolution"] = bool(
        result["records_used"] >= min_resolved_rows
        and result["resolved_lineages"] >= min_resolved_lineages
    )
    denominators = manifest.get("independent_denominators")
    if not isinstance(denominators, Mapping):
        raise ValueError("quality-preflight source has no denominator registry")
    expected = {
        "continuation_trajectories_authenticated": denominators.get("continuation_trajectories"),
        "fully_unresolved_skipped": denominators.get("fully_unresolved_rows"),
        "label_lineages": denominators.get("lineages_with_records"),
        "records_total": denominators.get("records_total"),
        "records_used": denominators.get("resolved_rows"),
        "resolved_lineages": denominators.get("resolved_lineages"),
    }
    if any(result[key] != value for key, value in expected.items()):
        raise ValueError("parallel quality-preflight totals differ from the label manifest")
    trajectories = result["continuation_trajectories_authenticated"]
    if (
        result["continuation_replays_executed"] != trajectories
        or result["continuation_replay_matches"] != trajectories
    ):
        raise ValueError("parallel quality preflight did not replay every continuation exactly")
    return result


def _parallel_quality_preflight(
    args: argparse.Namespace,
    *,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one invocation-scoped multiprocessing replay with whole-lineage ownership."""

    import concurrent.futures
    import multiprocessing

    root = Path(args.quality_labels)
    record_index = _build_quality_jsonl_index(root, manifest)
    denominators = manifest.get("independent_denominators")
    if (
        manifest.get("lineages") != sorted(record_index.lineages)
        or manifest.get("task_ids") != sorted(record_index.task_ids)
        or not isinstance(denominators, Mapping)
        or denominators.get("tasks_with_records") != len(record_index.task_ids)
        or denominators.get("valid_continuations") != record_index.valid_continuations
    ):
        raise ValueError("parallel quality-preflight record census differs from its manifest")
    lineages = sorted(record_index.lineages)
    if not lineages:
        raise ValueError("quality preflight has no record-bearing lineages")
    worker_count = min(args.workers, len(lineages))
    assignments = [lineages[index::worker_count] for index in range(worker_count)]
    child_args = {
        "allow_legacy_pilot": bool(getattr(args, "allow_legacy_pilot", False)),
        "corpus": args.corpus,
        "deterministic": args.deterministic,
        "device": args.device,
        "expected_ground_certificate_root_sha256": (args.expected_ground_certificate_root_sha256),
        "expected_quality_attestation_digest": args.expected_quality_attestation_digest,
        "expected_quality_publisher_id": args.expected_quality_publisher_id,
        "ground_certificate_root": args.ground_certificate_root,
        "min_resolved_lineages": args.min_resolved_lineages,
        "min_resolved_rows": args.min_resolved_rows,
        "quality_attestation": args.quality_attestation,
        "quality_labels": args.quality_labels,
        "initializer_bank": getattr(args, "initializer_bank", None),
        "expected_initializer_bank_manifest_sha256": getattr(
            args,
            "expected_initializer_bank_manifest_sha256",
            None,
        ),
        "complete_config": getattr(args, "complete_config", None),
        "qubit_cap": args.qubit_cap,
        "seed": args.seed,
        "selector": args.selector,
    }
    summaries: list[dict[str, Any] | None] = [None] * worker_count
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count, mp_context=context
    ) as executor:
        futures = {
            executor.submit(
                _quality_preflight_process_worker,
                {
                    "args": child_args,
                    "lineages": lineage_assignment,
                    "record_selection": record_index.select(
                        frozenset(lineage_assignment)
                    ),
                },
            ): index
            for index, lineage_assignment in enumerate(assignments)
        }
        for future in concurrent.futures.as_completed(futures):
            index = futures[future]
            summaries[index] = future.result()
            print(
                json.dumps(
                    {
                        "event": "quality-preflight-worker-complete",
                        "completed_workers": sum(item is not None for item in summaries),
                        "workers": worker_count,
                        "worker_index": index,
                    }
                ),
                flush=True,
            )
    result = _aggregate_quality_preflight_summaries(
        [summary for summary in summaries if summary is not None],
        manifest=manifest,
        min_resolved_rows=args.min_resolved_rows,
        min_resolved_lineages=args.min_resolved_lineages,
    )
    _assert_quality_jsonl_index_current(root, record_index)
    return result


def cmd_quality_preflight(args: argparse.Namespace) -> None:
    """Replay and authenticate every label before any model or optimizer is created."""

    ctx = _context(args, args.corpus)
    bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
    source_manifest = _strict_json(Path(args.quality_labels) / "manifest.json")
    initializer_bank, initializer_bank_manifest_sha256 = (
        _quality_initializer_bank_for_manifest(
            args,
            manifest=source_manifest,
            context=ctx,
        )
    )
    _seed_runtime(args.seed, deterministic=args.deterministic, threads=args.threads)
    quality_pin = _quality_attestation_pin(args)
    if args.workers == 1:
        bundle.model.to(_resolve_device(args.device)).eval()
        _records, manifest = _load_quality_labels(
            args.quality_labels,
            corpus=args.corpus,
            selector=bundle,
            context=ctx,
            min_resolved_rows=args.min_resolved_rows,
            min_resolved_lineages=args.min_resolved_lineages,
            enforce_resolution=False,
            require_provenance=not getattr(args, "allow_legacy_pilot", False),
            quality_attestation_pin=quality_pin,
            initializer_bank=initializer_bank,
            expected_initializer_bank_manifest_sha256=(
                initializer_bank_manifest_sha256
            ),
            allow_diagnostic_legacy=bool(getattr(args, "allow_legacy_pilot", False)),
        )
        summary = manifest["loader_summary"]
    else:
        root = Path(args.quality_labels)
        manifest = _strict_json(root / "manifest.json")
        _verify_record(manifest, "parallel quality-preflight source manifest")
        label_protocol = manifest.get("label_protocol")
        expected_merged_version = (
            QUALITY_MERGED_SCHEMA_VERSION
            if _quality_manifest_is_publication(manifest)
            else QUALITY_DIAGNOSTIC_MERGED_SCHEMA_VERSION
        )
        if (
            manifest.get("schema") != QUALITY_MERGED_SCHEMA
            or manifest.get("schema_version") != expected_merged_version
            or manifest.get("source_corpus_manifest_sha256") != _corpus_manifest_digest(args.corpus)
            or manifest.get("selector_digest") != bundle.selector_digest
            or manifest.get("normalizer_digest") != bundle.normalizer_digest
            or manifest.get("context") != _context_snapshot(ctx)
            or manifest.get("context_digest") != stable_digest(_context_snapshot(ctx))
            or not isinstance(label_protocol, Mapping)
            or label_protocol.get("implementation_contract")
            != _quality_implementation_contract(ctx)
        ):
            raise ValueError(
                "parallel quality preflight requires the exact current merged quality-v7 artifact"
            )
        train_tasks, quality_authority, target_access, ground_partition = _load_quality_partition(
            args.corpus,
            partition="train",
            pin=quality_pin,
            role="training_partition",
        )
        if (
            _bind_selector_quality_authority(bundle, train_tasks, pin=quality_pin)
            != quality_authority
            or manifest.get("quality_authority") != quality_authority
            or manifest.get("target_access") != target_access
            or manifest.get("ground_partition_receipt") != ground_partition
        ):
            raise ValueError("parallel quality preflight target authority differs")
        summary = _parallel_quality_preflight(args, manifest=manifest)
    if bundle.quality_authority != manifest["quality_authority"]:
        raise ValueError("quality preflight selector and labels use different authorities")
    source_manifest_path = Path(args.quality_labels) / "manifest.json"
    source_records_path = Path(args.quality_labels) / "records.jsonl"
    authenticated_manifest = dict(manifest)
    authenticated_manifest.pop("loader_summary", None)
    if _strict_json(source_manifest_path) != authenticated_manifest:
        raise ValueError("quality-label manifest changed during preflight")
    source_manifest_sha256 = _sha256_file(source_manifest_path)
    source_records_sha256 = _sha256_file(source_records_path)
    if not hmac.compare_digest(
        source_records_sha256,
        str(authenticated_manifest.get("records_sha256")),
    ):
        raise ValueError("quality-label records changed during preflight")
    full_replay = {
        "mode": summary["continuation_replay_mode"],
        "continuation_trajectories": summary["continuation_trajectories_authenticated"],
        "continuation_replays_executed": summary["continuation_replays_executed"],
        "continuation_replay_matches": summary["continuation_replay_matches"],
    }
    publication = _quality_manifest_is_publication(manifest)
    receipt_payload: dict[str, object] = {
            "schema": QUALITY_PREFLIGHT_SCHEMA,
            "schema_version": (
                QUALITY_PREFLIGHT_SCHEMA_VERSION
                if publication
                else QUALITY_DIAGNOSTIC_PREFLIGHT_SCHEMA_VERSION
            ),
            "source_quality_manifest_sha256": source_manifest_sha256,
            "source_quality_manifest_record_digest": manifest["record_digest"],
            "source_quality_records_sha256": source_records_sha256,
            "source_corpus_manifest_sha256": _corpus_manifest_digest(args.corpus),
            "quality_authority": manifest["quality_authority"],
            "selector_digest": bundle.selector_digest,
            "normalizer_digest": bundle.normalizer_digest,
            "context": _context_snapshot(ctx),
            "context_digest": stable_digest(_context_snapshot(ctx)),
            "implementation_contract": _quality_implementation_contract(ctx),
            "label_protocol_digest": stable_digest(manifest["label_protocol"]),
            "minimum_resolved_rows": args.min_resolved_rows,
            "minimum_resolved_lineages": args.min_resolved_lineages,
            "full_replay": full_replay,
            "resolution": summary,
            "advance": bool(summary["passes_resolution"]),
    }
    if publication:
        receipt_payload["initializer_bank"] = manifest["initializer_bank"]
        receipt_payload["quality_resolution_plan"] = manifest["quality_resolution_plan"]
    receipt = _with_digest(receipt_payload)
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"quality preflight receipt already exists: {destination}")
    _atomic_json(destination, receipt)
    print(json.dumps(receipt, indent=1))
    if receipt["advance"] is not True:
        raise ValueError(
            "quality resolution preflight failed; retain the authenticated report, increase "
            "the preregistered continuation count or sample, and regenerate labels"
        )


def _ppo_config(args: argparse.Namespace):
    from isingfold.rl.ppo import PPOConfig

    return PPOConfig(
        episodes_per_batch=args.episodes,
        epochs=args.ppo_epochs,
        minibatch=args.minibatch,
        clip=0.2,
        learning_rate=args.learning_rate,
        gae_lambda=0.95,
        entropy_initial=0.01,
        entropy_floor=0.001,
        entropy_floor_at=0.8,
        grad_norm=0.5,
        kl_target=0.01,
        kl_stop=0.02,
        kl_backtrack_factor=0.5,
        kl_max_backtracks=3,
        value_weight=0.5,
        reference_length=32,
        failure_constraint_enabled=False,
        seed=args.seed,
    )


def _ppo_collection_schedule_contract() -> dict[str, object]:
    """Return the immutable behavior-collection scheduler identity."""

    from isingfold.rl.ppo import (
        COLLECTION_SCHEDULE,
        COLLECTION_SCHEDULE_SCHEMA,
        COLLECTION_SCHEDULE_SCHEMA_VERSION,
    )

    return {
        "schema": COLLECTION_SCHEDULE_SCHEMA,
        "schema_version": COLLECTION_SCHEDULE_SCHEMA_VERSION,
        "schedule": COLLECTION_SCHEDULE,
    }


def _ppo_update_control_contract(config: object) -> dict[str, object]:
    """Bind the KL transaction and entropy anti-collapse rules to a training run."""

    from isingfold.rl.ppo import (
        ENTROPY_NORMALIZATION,
        ENTROPY_NORMALIZATION_SCHEMA_VERSION,
        PPO_UPDATE_CONTROL,
        PPO_UPDATE_CONTROL_SCHEMA,
        PPO_UPDATE_CONTROL_SCHEMA_VERSION,
        PPOConfig,
    )

    if not isinstance(config, PPOConfig):
        raise TypeError("PPO update-control contract requires a validated PPOConfig")
    return {
        "schema": PPO_UPDATE_CONTROL_SCHEMA,
        "schema_version": PPO_UPDATE_CONTROL_SCHEMA_VERSION,
        "rule": PPO_UPDATE_CONTROL,
        "empirical_kl": "full-legal-support categorical KL(old_behavior || current)",
        "soft_target": config.kl_target,
        "hard_limit": config.kl_stop,
        "epoch_transaction": "snapshot-model-optimizer-and-rng; commit-only-below-hard-limit",
        "backtrack_factor": config.kl_backtrack_factor,
        "maximum_backtracks_per_epoch": config.kl_max_backtracks,
        "soft_target_action": "stop-remaining-epochs-after-accepted-epoch",
        "exhaustion_action": "rollback-to-last-safe-boundary-and-lower-next-learning-rate",
        "entropy_normalization": ENTROPY_NORMALIZATION,
        "entropy_normalization_schema_version": ENTROPY_NORMALIZATION_SCHEMA_VERSION,
        "entropy_schedule": "linear-from-initial-to-positive-floor",
        "entropy_initial": config.entropy_initial,
        "entropy_floor": config.entropy_floor,
        "entropy_floor_at_fraction": config.entropy_floor_at,
    }


def _require_current_ppo_collection_schedule(contract: Mapping[str, object]) -> None:
    """Reject RL artifacts collected with the retired scalar RNG schedule."""

    hyperparameters = contract.get("hyperparameters")
    if (
        not isinstance(hyperparameters, Mapping)
        or hyperparameters.get("ppo_collection_schedule") != _ppo_collection_schedule_contract()
    ):
        raise ValueError("RL-value artifact lacks the registered batched PPO collection schedule")


def _require_current_ppo_update_control(contract: Mapping[str, object]) -> None:
    """Reject RL artifacts that lack the hard KL and entropy anti-collapse contract."""

    from isingfold.rl.ppo import PPOConfig

    hyperparameters = contract.get("hyperparameters")
    if not isinstance(hyperparameters, Mapping):
        raise ValueError("RL-value artifact lacks PPO hyperparameters")
    config_fields = {field.name for field in dataclasses.fields(PPOConfig)}
    if not config_fields <= set(hyperparameters):
        raise ValueError("RL-value artifact lacks the current PPO update-control fields")
    try:
        config = PPOConfig(**{name: hyperparameters[name] for name in sorted(config_fields)})
    except (TypeError, ValueError) as error:
        raise ValueError("RL-value artifact has invalid PPO update-control fields") from error
    if hyperparameters.get("ppo_update_control") != _ppo_update_control_contract(config):
        raise ValueError("RL-value artifact lacks the registered PPO update-control contract")


def _experiment_contract(
    *,
    phase: str,
    model_family: str,
    model: Any,
    method: str,
    context: Context,
    corpus_manifest_sha256: str,
    quality_authority: Mapping[str, object],
    target_access: Mapping[str, object],
    ground_partition_receipt: Mapping[str, object],
    selector_digest: str,
    normalizer_digest: str,
    seed: int,
    device_type: str,
    deterministic: bool,
    hyperparameters: Mapping[str, object],
    grid_manifest_sha256: str | None = None,
    grid_cell: str | None = None,
    selection_receipt_sha256: str | None = None,
    selection_record_digest: str | None = None,
    selection_mode: str | None = None,
    gate_receipt_sha256: str | None = None,
    gate_record_digest: str | None = None,
    gate_profile: str | None = None,
    quality_preflight_receipt_sha256: str | None = None,
    quality_preflight_record_digest: str | None = None,
) -> dict[str, object]:
    return {
        "schema": "isingfold.experiment-contract",
        "schema_version": CLI_SCHEMA_VERSION,
        "phase": phase,
        "model_family": model_family,
        "method": method,
        "model": _model_identity(model, model_family),
        "environment": _context_snapshot(context),
        "endpoint": context.endpoint,
        "corpus_manifest_sha256": corpus_manifest_sha256,
        "quality_authority": dict(quality_authority),
        "target_access": dict(target_access),
        "ground_partition_receipt": dict(ground_partition_receipt),
        "selector_digest": selector_digest,
        "normalizer_digest": normalizer_digest,
        "seed": seed,
        "device_type": device_type,
        "float_dtype": "float32",
        "deterministic_algorithms": deterministic,
        "hyperparameters": dict(hyperparameters),
        "grid_manifest_sha256": grid_manifest_sha256,
        "grid_cell": grid_cell,
        "selection_receipt_sha256": selection_receipt_sha256,
        "selection_record_digest": selection_record_digest,
        "selection_mode": selection_mode,
        "gate_receipt_sha256": gate_receipt_sha256,
        "gate_record_digest": gate_record_digest,
        "gate_profile": gate_profile,
        "quality_preflight_receipt_sha256": quality_preflight_receipt_sha256,
        "quality_preflight_record_digest": quality_preflight_record_digest,
    }


def _run_record(
    *,
    phase: str,
    model_family: str,
    method: str,
    contract: Mapping[str, object],
    normalizer: Mapping[str, object],
    proposal_version: str,
    selector_digest: str,
    training_lineages: Sequence[str],
    checkpoint_metadata,
    complete: bool,
) -> dict[str, object]:
    return _with_digest(
        {
            "schema": RUN_SCHEMA,
            "schema_version": RUN_SCHEMA_VERSION,
            "phase": phase,
            "model_family": model_family,
            "method": method,
            "experiment_contract": dict(contract),
            "feature_normalizer": dict(normalizer),
            "proposal_version": proposal_version,
            "selector_digest": selector_digest,
            "training_lineages": sorted(set(training_lineages)),
            "checkpoint_payload_digest": checkpoint_metadata.payload_digest,
            "runtime_implementation_digest": (checkpoint_metadata.runtime_implementation_digest),
            "runtime_implementation_registry": (
                checkpoint_metadata.runtime_implementation_registry
            ),
            "counters": checkpoint_metadata.counters,
            "trainer_state": checkpoint_metadata.trainer_state,
            "complete": complete,
        }
    )


def _save_training_state(
    run_dir: Path,
    *,
    model,
    optimizer,
    contract: Mapping[str, object],
    bundle: SelectorBundle,
    lineages: Sequence[str],
    counters: Mapping[str, int],
    phase: str,
    model_family: str,
    method: str,
    complete: bool,
    trainer_state: Mapping[str, object] | None = None,
):
    from isingfold.rl.checkpoint import save_checkpoint

    proposal_version = _proposal_version()
    metadata = save_checkpoint(
        run_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        context=contract,
        feature_normalizer=bundle.normalizer,
        proposal_version=proposal_version,
        selector_digest=bundle.selector_digest,
        training_lineages=lineages,
        counters=counters,
        trainer_state=trainer_state,
    )
    record = _run_record(
        phase=phase,
        model_family=model_family,
        method=method,
        contract=contract,
        normalizer=bundle.normalizer,
        proposal_version=proposal_version,
        selector_digest=bundle.selector_digest,
        training_lineages=lineages,
        checkpoint_metadata=metadata,
        complete=complete,
    )
    _atomic_json(run_dir / "run.json", record)
    return metadata


def _resume_training_state(
    run_dir: Path,
    *,
    model,
    optimizer,
    contract: Mapping[str, object],
    bundle: SelectorBundle,
    lineages: Sequence[str],
    expected_trainer_state_schema: Mapping[str, str] | None = None,
):
    from isingfold.rl.checkpoint import load_checkpoint

    record = _strict_json(run_dir / "run.json")
    _verify_record(record, "training-run receipt")
    if record.get("schema") != RUN_SCHEMA or record.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ValueError("unsupported training-run receipt")
    if record.get("experiment_contract") != dict(contract):
        raise ValueError("resume experiment contract differs from the existing run")
    runtime_digest = record.get("runtime_implementation_digest")
    if not isinstance(runtime_digest, str) or len(runtime_digest) != 64:
        raise ValueError("training-run receipt has no valid runtime implementation digest")
    metadata = load_checkpoint(
        run_dir / "checkpoint.pt",
        model=model,
        optimizer=optimizer,
        expected_context=contract,
        expected_feature_normalizer=bundle.normalizer,
        expected_proposal_version=_proposal_version(),
        expected_selector_digest=bundle.selector_digest,
        expected_training_lineages=lineages,
        expected_trainer_state_schema=expected_trainer_state_schema,
        expected_runtime_digest=runtime_digest,
        restore_rng=True,
        map_location="cpu",
    )
    if metadata.payload_digest != record.get("checkpoint_payload_digest"):
        raise ValueError("training-run receipt references a different checkpoint payload")
    if record.get("runtime_implementation_registry") != metadata.runtime_implementation_registry:
        raise ValueError("training-run receipt references a different runtime registry")
    if record.get("trainer_state") != metadata.trainer_state:
        raise ValueError("training-run receipt references different mutable trainer state")
    return metadata


def _prepare_run_dir(path: str | os.PathLike[str], *, resume: bool) -> Path:
    destination = Path(path)
    if resume:
        if (
            not (destination / "checkpoint.pt").is_file()
            or not (destination / "run.json").is_file()
        ):
            raise ValueError("resume requires checkpoint.pt and run.json in the run directory")
    else:
        if destination.exists():
            raise FileExistsError(f"refusing to mix a new run into existing path: {destination}")
        destination.mkdir(parents=True)
    return destination


def cmd_warm_start(args: argparse.Namespace) -> None:
    """Initialize actor ranking and utility value from train-only counterfactuals."""

    import torch

    from isingfold.rl.ppo import (
        WARM_START_ACTION_VALUE_TARGET,
        WARM_START_ACTION_VALUE_WEIGHTING,
        WARM_START_ACTOR_CRITIC_LOSS,
        WARM_START_COMMIT_DELTA_TARGET,
        WARM_START_FULL_PROFILE_ID,
        WARM_START_REDUCTION,
        WARM_START_UTILITY_TARGET,
        WARM_START_UTILITY_WEIGHTING,
        WarmStartCorpusDenominators,
        warm_start_actor_critic_loss,
        warm_start_loss_profile,
    )

    if args.epochs <= 0 or args.minibatch <= 0:
        raise ValueError("warm-start epochs and minibatch must be positive")
    loss_profile = warm_start_loss_profile(
        getattr(args, "warm_start_loss_profile_id", WARM_START_FULL_PROFILE_ID)
    )
    diagnostic_binding = getattr(args, "warm_start_diagnostic_binding", None)
    if not loss_profile.production_transfer_eligible:
        from isingfold.rl.quality_warm_control import validate_quality_warm_control_binding

        diagnostic_binding = validate_quality_warm_control_binding(
            diagnostic_binding,
            expected_profile_id=loss_profile.profile_id,
            expected_seed=args.seed,
            expected_model_family=args.model_family,
        )
    elif diagnostic_binding is not None:
        raise ValueError("production warm-start profile forbids a diagnostic-control binding")

    ctx = _context(args, args.corpus)
    bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
    source_quality_manifest = _strict_json(Path(args.quality_labels) / "manifest.json")
    initializer_bank, initializer_bank_manifest_sha256 = (
        _quality_initializer_bank_for_manifest(
            args,
            manifest=source_quality_manifest,
            context=ctx,
            bank_attribute="quality_initializer_bank",
            pin_attribute="expected_quality_initializer_bank_manifest_sha256",
            config_attribute="quality_complete_config",
        )
    )
    quality_pin = _quality_attestation_pin(args)
    quality_preflight = _load_quality_preflight_receipt(
        args.quality_preflight_receipt,
        expected_sha256=args.expected_quality_preflight_sha256,
        quality_labels=args.quality_labels,
        corpus=args.corpus,
        selector=bundle,
        context=ctx,
        min_resolved_rows=args.min_resolved_rows,
        min_resolved_lineages=args.min_resolved_lineages,
        allow_diagnostic_legacy=bool(getattr(args, "allow_legacy_pilot", False)),
    )
    records, quality_manifest = _load_quality_labels(
        args.quality_labels,
        corpus=args.corpus,
        selector=bundle,
        context=ctx,
        min_resolved_rows=args.min_resolved_rows,
        min_resolved_lineages=args.min_resolved_lineages,
        require_provenance=not getattr(args, "allow_legacy_pilot", False),
        quality_attestation_pin=quality_pin,
        trusted_preflight=quality_preflight,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=(initializer_bank_manifest_sha256),
        allow_diagnostic_legacy=bool(getattr(args, "allow_legacy_pilot", False)),
    )
    if bundle.quality_authority != quality_manifest["quality_authority"]:
        raise ValueError("warm-start selector and labels use different quality authorities")
    actor_ranking_records = sum(
        bool(best) and bool(evaluated) and set(best) != set(evaluated)
        for _observation, best, evaluated, _target in records
    )
    if actor_ranking_records != quality_manifest["loader_summary"]["records_used"]:
        raise RuntimeError("warm-start actor-ranking denominator differs from quality preflight")
    action_value_targets = [record[3].action_value for record in records]
    if any(target is None for target in action_value_targets):
        raise RuntimeError("authenticated quality row omitted action-value supervision")
    action_value_records = len(action_value_targets)
    action_value_actions = sum(
        len(target.action_indices) for target in action_value_targets if target is not None
    )
    commit_delta_records = sum(
        target is not None and target.commit_index is not None and len(target.action_indices) > 1
        for target in action_value_targets
    )
    corpus_denominators = WarmStartCorpusDenominators(
        actor_ranking_records=actor_ranking_records,
        utility_effective_count=math.fsum(
            float(record[3].effective_count) for record in records
        ),
        action_value_records=action_value_records,
        commit_delta_records=commit_delta_records,
    )
    denominator_contract = dataclasses.asdict(corpus_denominators)
    print(json.dumps({"quality_resolution": quality_manifest["loader_summary"]}, indent=1))
    device = _resolve_device(args.device)
    _seed_runtime(args.seed, deterministic=args.deterministic, threads=args.threads)
    model = _model(
        args.model_family,
        quality_prior_mode=getattr(args, "quality_prior_mode", None),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-5,
        weight_decay=args.weight_decay,
    )
    lineages = tuple(sorted(set(quality_manifest["lineages"])))
    hyperparameters = {
        "epochs": args.epochs,
        "minibatch_records": args.minibatch,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "quality_manifest_sha256": _sha256_file(Path(args.quality_labels) / "manifest.json"),
        "quality_loader_summary": quality_manifest["loader_summary"],
        "minimum_resolved_rows": args.min_resolved_rows,
        "minimum_resolved_lineages": args.min_resolved_lineages,
        "quality_preflight_receipt_sha256": _sha256_file(args.quality_preflight_receipt),
        "quality_preflight_record_digest": quality_preflight["record_digest"],
        "warm_start_loss": WARM_START_ACTOR_CRITIC_LOSS,
        "warm_start_loss_profile": loss_profile.contract(),
        "warm_start_rank_coefficient": loss_profile.weights.rank,
        "warm_start_action_value_target": WARM_START_ACTION_VALUE_TARGET,
        "warm_start_action_value_weighting": WARM_START_ACTION_VALUE_WEIGHTING,
        "warm_start_action_value_coefficient": loss_profile.weights.action_value,
        "warm_start_commit_delta_target": WARM_START_COMMIT_DELTA_TARGET,
        "warm_start_commit_delta_coefficient": loss_profile.weights.commit_delta,
        "warm_start_reduction": WARM_START_REDUCTION,
        "warm_start_utility_target": WARM_START_UTILITY_TARGET,
        "warm_start_utility_weighting": WARM_START_UTILITY_WEIGHTING,
        "warm_start_utility_coefficient": loss_profile.weights.utility,
        "warm_start_diagnostic_binding": diagnostic_binding,
        "warm_start_actor_ranking_records": actor_ranking_records,
        "warm_start_utility_critic_records": len(records),
        "warm_start_action_value_records": action_value_records,
        "warm_start_action_value_actions": action_value_actions,
        "warm_start_commit_delta_records": commit_delta_records,
        "warm_start_corpus_denominators": denominator_contract,
    }
    contract = _experiment_contract(
        phase="representation",
        model_family=args.model_family,
        model=model,
        method="supervised-ranking",
        context=ctx,
        corpus_manifest_sha256=bundle.corpus_manifest_sha256,
        quality_authority=quality_manifest["quality_authority"],
        target_access=quality_manifest["target_access"],
        ground_partition_receipt=quality_manifest["ground_partition_receipt"],
        selector_digest=bundle.selector_digest,
        normalizer_digest=bundle.normalizer_digest,
        seed=args.seed,
        device_type=device.type,
        deterministic=args.deterministic,
        hyperparameters=hyperparameters,
        grid_manifest_sha256=getattr(args, "grid_manifest_sha256", None),
        grid_cell=getattr(args, "grid_cell", None),
        selection_receipt_sha256=getattr(args, "selection_receipt_sha256", None),
        selection_record_digest=getattr(args, "selection_record_digest", None),
        selection_mode=getattr(args, "selection_mode", None),
        gate_receipt_sha256=getattr(args, "gate_receipt_sha256", None),
        gate_record_digest=getattr(args, "gate_record_digest", None),
        gate_profile=getattr(args, "gate_profile", None),
        quality_preflight_receipt_sha256=_sha256_file(args.quality_preflight_receipt),
        quality_preflight_record_digest=quality_preflight["record_digest"],
    )
    run_dir = _prepare_run_dir(args.out, resume=args.resume)
    epochs_done = 0
    optimizer_steps = 0
    if args.resume:
        metadata = _resume_training_state(
            run_dir,
            model=model,
            optimizer=optimizer,
            contract=contract,
            bundle=bundle,
            lineages=lineages,
        )
        epochs_done = metadata.counters.get("epochs", 0)
        optimizer_steps = metadata.counters.get("optimizer_steps", 0)

    history_path = run_dir / "history.json"
    history_payload = _strict_json(history_path) if history_path.exists() else None
    if history_payload is not None and (
        history_payload.get("schema") != WARM_START_HISTORY_SCHEMA
        or history_payload.get("schema_version") != WARM_START_HISTORY_SCHEMA_VERSION
    ):
        raise ValueError("warm-start history uses a retired reduction schema")
    history = history_payload.get("epochs", []) if history_payload is not None else []
    if not isinstance(history, list) or len(history) != epochs_done:
        raise ValueError("warm-start history and checkpoint counters differ")
    for epoch in range(epochs_done, args.epochs):
        order = np.random.default_rng(
            _domain_seed(args.seed, "warm-start-order", epoch)
        ).permutation(len(records))
        detached_loss_rows: list[torch.Tensor] = []
        utility_effective_counts_seen: list[float] = []
        action_effective_counts_seen: list[float] = []
        actor_ranking_rows = 0
        action_value_rows = 0
        commit_delta_rows = 0
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(order), args.minibatch):
            indices = order[start : start + args.minibatch]
            observations = [records[int(index)][0] for index in indices]
            best = [records[int(index)][1] for index in indices]
            evaluated = [records[int(index)][2] for index in indices]
            utility_targets = [records[int(index)][3].value for index in indices]
            effective_counts = [records[int(index)][3].effective_count for index in indices]
            minibatch_action_targets = [records[int(index)][3].action_value for index in indices]
            losses = warm_start_actor_critic_loss(
                model,
                observations,
                best,
                evaluated,
                utility_targets,
                effective_counts,
                action_value_targets=minibatch_action_targets,
                device=device,
                corpus_denominators=corpus_denominators,
                loss_profile=loss_profile,
            )
            losses.total.backward()
            detached_loss_rows.append(
                torch.stack(
                    (
                        losses.rank,
                        losses.action_value,
                        losses.commit_delta,
                        losses.utility,
                        losses.total,
                    )
                ).detach()
            )
            utility_effective_counts_seen.append(losses.effective_count)
            action_effective_counts_seen.append(losses.action_effective_count)
            actor_ranking_rows += sum(
                bool(row_best)
                and bool(row_evaluated)
                and set(row_best) != set(row_evaluated)
                for row_best, row_evaluated in zip(best, evaluated, strict=True)
            )
            action_value_rows += sum(
                target is not None for target in minibatch_action_targets
            )
            commit_delta_rows += losses.commit_delta_rows
        utility_effective_count = math.fsum(utility_effective_counts_seen)
        action_effective_count = math.fsum(action_effective_counts_seen)
        detached_loss_matrix = torch.stack(detached_loss_rows)
        if not torch.isfinite(detached_loss_matrix).all():
            raise RuntimeError("non-finite supervised actor-critic loss")
        if not math.isclose(
            utility_effective_count,
            corpus_denominators.utility_effective_count,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise RuntimeError("warm-start epoch omitted authenticated utility mass")
        if (
            actor_ranking_rows != corpus_denominators.actor_ranking_records
            or action_value_rows != corpus_denominators.action_value_records
            or commit_delta_rows != corpus_denominators.commit_delta_records
        ):
            raise RuntimeError("warm-start epoch omitted authenticated supervised rows")
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        if not torch.isfinite(gradient_norm):
            raise RuntimeError("non-finite supervised actor-critic gradient")
        optimizer.step()
        optimizer_steps += 1
        (
            corpus_rank_loss,
            corpus_action_value_loss,
            corpus_commit_delta_loss,
            corpus_utility_loss,
            corpus_total_loss,
        ) = (
            float(value)
            for value in detached_loss_matrix.sum(dim=0).cpu().tolist()
        )
        history.append(
            {
                "epoch": epoch + 1,
                "corpus_rank_loss": corpus_rank_loss,
                "corpus_action_value_loss": corpus_action_value_loss,
                "corpus_commit_delta_loss": corpus_commit_delta_loss,
                "corpus_utility_loss": corpus_utility_loss,
                "corpus_total_loss": corpus_total_loss,
                "utility_effective_count": utility_effective_count,
                "action_effective_count": action_effective_count,
                "actor_ranking_rows": actor_ranking_rows,
                "action_value_rows": action_value_rows,
                "commit_delta_rows": commit_delta_rows,
                "memory_minibatches": len(detached_loss_rows),
                "gradient_norm_before_clip": float(gradient_norm.detach().item()),
            }
        )
        _atomic_json(
            history_path,
            {
                "schema": WARM_START_HISTORY_SCHEMA,
                "schema_version": WARM_START_HISTORY_SCHEMA_VERSION,
                "epochs": history,
            },
        )
        _save_training_state(
            run_dir,
            model=model,
            optimizer=optimizer,
            contract=contract,
            bundle=bundle,
            lineages=lineages,
            counters={
                "epochs": epoch + 1,
                "optimizer_steps": optimizer_steps,
                "records": len(records),
            },
            phase="representation",
            model_family=args.model_family,
            method="supervised-ranking",
            complete=epoch + 1 == args.epochs,
        )
        print(json.dumps(history[-1]))


def _load_transfer(
    checkpoint: str | os.PathLike[str],
    *,
    model,
    bundle: SelectorBundle,
    corpus: str | os.PathLike[str],
    model_family: str,
    expected_seed: int,
    expected_grid_cell: str,
    expected_grid_manifest_sha256: str,
    expected_quality_preflight_receipt_sha256: str,
    expected_quality_preflight_record_digest: str,
    expected_checkpoint_payload_digest: str | None = None,
) -> str:
    """Install a representation state only after binding it to caller-pinned provenance."""

    import torch

    from isingfold.rl.checkpoint import load_checkpoint
    from isingfold.rl.ppo import (
        WARM_START_ACTION_VALUE_COEFFICIENT,
        WARM_START_ACTION_VALUE_TARGET,
        WARM_START_ACTION_VALUE_WEIGHTING,
        WARM_START_ACTOR_CRITIC_LOSS,
        WARM_START_COMMIT_DELTA_COEFFICIENT,
        WARM_START_COMMIT_DELTA_TARGET,
        WARM_START_FULL_LOSS_PROFILE,
        WARM_START_RANK_COEFFICIENT,
        WARM_START_REDUCTION,
        WARM_START_UTILITY_COEFFICIENT,
        WARM_START_UTILITY_TARGET,
        WARM_START_UTILITY_WEIGHTING,
    )

    if (
        isinstance(expected_seed, bool)
        or not isinstance(expected_seed, int)
        or expected_seed < 0
    ):
        raise ValueError("warm-start transfer requires an expected nonnegative seed")
    if not isinstance(expected_grid_cell, str) or not expected_grid_cell:
        raise ValueError("warm-start transfer requires an expected representation grid cell")
    expected_digests = {
        "grid manifest": expected_grid_manifest_sha256,
        "quality preflight receipt": expected_quality_preflight_receipt_sha256,
        "quality preflight record": expected_quality_preflight_record_digest,
    }
    for label, digest in expected_digests.items():
        if not _is_lower_sha256(digest):
            raise ValueError(f"warm-start transfer requires an expected {label} SHA-256")
    if expected_checkpoint_payload_digest is not None and not _is_lower_sha256(
        expected_checkpoint_payload_digest
    ):
        raise ValueError("warm-start transfer has an invalid selected checkpoint payload digest")

    path = Path(checkpoint)
    record = _strict_json(path.parent / "run.json")
    _verify_record(record, "warm-start run receipt")
    if (
        record.get("schema") != RUN_SCHEMA
        or record.get("schema_version") != RUN_SCHEMA_VERSION
        or record.get("phase") != "representation"
        or record.get("model_family") != model_family
        or record.get("method") != "supervised-ranking"
        or record.get("complete") is not True
    ):
        raise ValueError("warm-start checkpoint is not a completed matching representation run")
    contract = record.get("experiment_contract")
    normalizer = record.get("feature_normalizer")
    lineages = record.get("training_lineages")
    if (
        not isinstance(contract, dict)
        or not isinstance(normalizer, dict)
        or not isinstance(lineages, list)
    ):
        raise ValueError("warm-start run receipt is malformed")
    if (
        contract.get("phase") != record["phase"]
        or contract.get("method") != record["method"]
        or contract.get("model_family") != model_family
        or contract.get("model") != _model_identity(model, model_family)
    ):
        raise ValueError("warm-start run and model experiment identities differ")
    if contract.get("seed") != expected_seed:
        raise ValueError("warm-start checkpoint belongs to another training seed")
    if contract.get("grid_cell") != expected_grid_cell:
        raise ValueError("warm-start checkpoint belongs to another representation grid cell")
    if contract.get("grid_manifest_sha256") != expected_grid_manifest_sha256:
        raise ValueError("warm-start checkpoint belongs to another grid manifest")
    if (
        contract.get("quality_preflight_receipt_sha256")
        != expected_quality_preflight_receipt_sha256
        or contract.get("quality_preflight_record_digest")
        != expected_quality_preflight_record_digest
    ):
        raise ValueError("warm-start checkpoint belongs to another trusted quality preflight")
    if contract.get("corpus_manifest_sha256") != _corpus_manifest_digest(corpus):
        raise ValueError("warm-start checkpoint belongs to another prepared corpus")
    if (
        contract.get("quality_authority") != bundle.quality_authority
        or contract.get("target_access") != bundle.target_access
        or contract.get("ground_partition_receipt") != bundle.ground_partition_receipt
    ):
        raise ValueError(
            "warm-start checkpoint uses a different quality authority or train capability"
        )
    if (
        contract.get("selector_digest") != bundle.selector_digest
        or contract.get("normalizer_digest") != bundle.normalizer_digest
    ):
        raise ValueError("warm-start experiment contract uses a different selector or normalizer")
    hyperparameters = contract.get("hyperparameters")
    expected_warm_start_contract = {
        "warm_start_loss": WARM_START_ACTOR_CRITIC_LOSS,
        "warm_start_loss_profile": WARM_START_FULL_LOSS_PROFILE.contract(),
        "warm_start_rank_coefficient": WARM_START_RANK_COEFFICIENT,
        "warm_start_action_value_target": WARM_START_ACTION_VALUE_TARGET,
        "warm_start_action_value_weighting": WARM_START_ACTION_VALUE_WEIGHTING,
        "warm_start_action_value_coefficient": WARM_START_ACTION_VALUE_COEFFICIENT,
        "warm_start_commit_delta_target": WARM_START_COMMIT_DELTA_TARGET,
        "warm_start_commit_delta_coefficient": WARM_START_COMMIT_DELTA_COEFFICIENT,
        "warm_start_reduction": WARM_START_REDUCTION,
        "warm_start_utility_target": WARM_START_UTILITY_TARGET,
        "warm_start_utility_weighting": WARM_START_UTILITY_WEIGHTING,
        "warm_start_utility_coefficient": WARM_START_UTILITY_COEFFICIENT,
        "warm_start_diagnostic_binding": None,
    }
    if not isinstance(hyperparameters, Mapping) or any(
        hyperparameters.get(name) != expected
        for name, expected in expected_warm_start_contract.items()
    ):
        raise ValueError("warm-start checkpoint lacks the current actor-critic loss contract")
    corpus_denominators = hyperparameters.get("warm_start_corpus_denominators")
    denominator_keys = {
        "actor_ranking_records",
        "utility_effective_count",
        "action_value_records",
        "commit_delta_records",
    }
    if (
        not isinstance(corpus_denominators, Mapping)
        or set(corpus_denominators) != denominator_keys
        or corpus_denominators.get("actor_ranking_records")
        != hyperparameters.get("warm_start_actor_ranking_records")
        or corpus_denominators.get("action_value_records")
        != hyperparameters.get("warm_start_action_value_records")
        or corpus_denominators.get("commit_delta_records")
        != hyperparameters.get("warm_start_commit_delta_records")
    ):
        raise ValueError("warm-start checkpoint has inconsistent corpus denominators")
    integer_denominators = (
        corpus_denominators["actor_ranking_records"],
        corpus_denominators["action_value_records"],
        corpus_denominators["commit_delta_records"],
    )
    duplicated_counts = (
        hyperparameters.get("warm_start_actor_ranking_records"),
        hyperparameters.get("warm_start_action_value_records"),
        hyperparameters.get("warm_start_commit_delta_records"),
    )
    utility_records = hyperparameters.get("warm_start_utility_critic_records")
    utility_effective_count = corpus_denominators["utility_effective_count"]
    if (
        any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (*integer_denominators, *duplicated_counts)
        )
        or isinstance(utility_records, bool)
        or not isinstance(utility_records, int)
        or utility_records <= 0
        or any(value > utility_records for value in integer_denominators)
        or isinstance(utility_effective_count, bool)
        or not isinstance(utility_effective_count, (int, float))
        or not math.isfinite(utility_effective_count)
        or utility_effective_count <= 0.0
        or corpus_denominators["commit_delta_records"]
        > corpus_denominators["action_value_records"]
    ):
        raise ValueError("warm-start checkpoint has invalid corpus denominators")
    preflight_sha256 = contract.get("quality_preflight_receipt_sha256")
    preflight_record_digest = contract.get("quality_preflight_record_digest")
    if (
        not _is_lower_sha256(preflight_sha256)
        or not _is_lower_sha256(preflight_record_digest)
        or hyperparameters.get("quality_preflight_receipt_sha256") != preflight_sha256
        or hyperparameters.get("quality_preflight_record_digest") != preflight_record_digest
    ):
        raise ValueError("warm-start checkpoint has inconsistent quality preflight pins")
    if record.get("selector_digest") != bundle.selector_digest or normalizer != bundle.normalizer:
        raise ValueError("warm-start checkpoint uses a different selector or normalizer")
    if record.get("proposal_version") != _proposal_version():
        raise ValueError("warm-start checkpoint uses a different proposal implementation")
    runtime_digest = record.get("runtime_implementation_digest")
    if not isinstance(runtime_digest, str) or len(runtime_digest) != 64:
        raise ValueError("warm-start receipt has no valid runtime implementation digest")
    receipt_payload_digest = record.get("checkpoint_payload_digest")
    if not _is_lower_sha256(receipt_payload_digest):
        raise ValueError("warm-start receipt has no valid checkpoint payload digest")
    if (
        expected_checkpoint_payload_digest is not None
        and receipt_payload_digest != expected_checkpoint_payload_digest
    ):
        raise ValueError("warm-start receipt differs from the selected checkpoint payload")
    temporary_optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    metadata = load_checkpoint(
        path,
        model=model,
        optimizer=temporary_optimizer,
        expected_context=contract,
        expected_feature_normalizer=normalizer,
        expected_proposal_version=record["proposal_version"],
        expected_selector_digest=bundle.selector_digest,
        expected_training_lineages=lineages,
        expected_runtime_digest=runtime_digest,
        restore_rng=False,
        map_location="cpu",
    )
    if metadata.payload_digest != receipt_payload_digest:
        raise ValueError("warm-start receipt references a different checkpoint")
    if record.get("runtime_implementation_registry") != metadata.runtime_implementation_registry:
        raise ValueError("warm-start receipt references a different runtime registry")
    if record.get("trainer_state") != metadata.trainer_state or metadata.trainer_state:
        raise ValueError("representation checkpoint must have empty trainer state")
    return metadata.payload_digest


def cmd_train(args: argparse.Namespace) -> None:
    """Train or resume one complete-episode Profile-I masked-PPO run."""

    from isingfold.rl.env import EmbeddingEnv
    from isingfold.rl.checkpoint import PPO_TRAINER_STATE_SCHEMA
    from isingfold.rl.ppo import PPOTrainer, collect

    if args.method != "supervised-only" and args.updates <= 0:
        raise ValueError("PPO methods require a positive update count")
    if args.method == "supervised-only" and args.updates != 0:
        raise ValueError("supervised-only requires --updates 0")
    if args.episodes <= 0 or args.ppo_epochs <= 0 or args.minibatch <= 0:
        raise ValueError("episodes, PPO epochs and minibatch must be positive")
    complete_system_policy = bool(getattr(args, "complete_system_no_restart", False))
    deployment_initializer_bank = bool(getattr(args, "deployment_initializer_bank", False))
    bank_backed_policy = complete_system_policy or deployment_initializer_bank
    if complete_system_policy and deployment_initializer_bank:
        raise ValueError(
            "legacy no-restart and persistent-cache deployment modes are mutually exclusive"
        )
    if args.method != "supervised-only" and not bank_backed_policy:
        raise ValueError(
            "registered Profile-I PPO requires an authenticated initializer bank; "
            "dispatch it through a staged grid or complete-system training command"
        )
    complete_selection_binding = getattr(args, "complete_selection_binding", None)
    if complete_system_policy and not isinstance(complete_selection_binding, dict):
        raise ValueError(
            "complete-system training must be dispatched by complete-system-train-cell "
            "with an authenticated three-seed RL-value freeze"
        )
    if complete_selection_binding is not None and (
        not isinstance(complete_selection_binding, dict) or not bank_backed_policy
    ):
        raise ValueError(
            "a complete-system selection binding requires registered bank-backed training"
        )
    initializer_bank_path = getattr(args, "initializer_bank", None)
    initializer_bank_pin = getattr(args, "expected_initializer_bank_manifest_sha256", None)
    complete_config_path = getattr(args, "complete_config", None)
    if bank_backed_policy and not all(
        isinstance(value, str) and value
        for value in (initializer_bank_path, initializer_bank_pin, complete_config_path)
    ):
        raise ValueError(
            "deployment-policy PPO requires an initializer bank, its external manifest "
            "SHA-256 pin, and the complete-system config"
        )
    if not bank_backed_policy and any(
        value is not None
        for value in (initializer_bank_path, initializer_bank_pin, complete_config_path)
    ):
        raise ValueError(
            "initializer-bank inputs are reserved for bank-backed deployment-policy training"
        )

    complete_config = None
    if bank_backed_policy:
        from isingfold.rl.complete_system import CompleteSystemConfig

        complete_config = CompleteSystemConfig.from_mapping(_strict_json(complete_config_path))

    quality_pin = _quality_attestation_pin(args)
    tasks, quality_authority, target_access, ground_partition_receipt = _load_quality_partition(
        args.corpus,
        partition="train",
        pin=quality_pin,
        role="training_partition",
    )
    base_ctx = _context(args, args.corpus)
    ctx, policy_restart_mode = _resolve_training_policy_context(
        base_ctx,
        complete_system_no_restart=complete_system_policy,
        deployment_initializer_bank=deployment_initializer_bank,
        complete_config=complete_config,
    )
    bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
    if _bind_selector_quality_authority(bundle, tasks, pin=quality_pin) != quality_authority:
        raise ValueError("selector and PPO training target authorities differ")
    device = _resolve_device(args.device)
    _seed_runtime(args.seed, deterministic=args.deterministic, threads=args.threads)
    model = _model(
        args.model_family,
        quality_prior_mode=getattr(args, "quality_prior_mode", None),
    )
    warm_digest: str | None = None
    if args.method in {"supervised-only", "ppo-warm-start"}:
        if not args.warm_start:
            raise ValueError(f"{args.method} requires --warm-start")
        warm_digest = _load_transfer(
            args.warm_start,
            model=model,
            bundle=bundle,
            corpus=args.corpus,
            model_family=args.model_family,
            expected_seed=args.seed,
            expected_grid_cell=getattr(args, "warm_start_grid_cell", None),
            expected_grid_manifest_sha256=getattr(args, "grid_manifest_sha256", None),
            expected_quality_preflight_receipt_sha256=getattr(
                args, "quality_preflight_receipt_sha256", None
            ),
            expected_quality_preflight_record_digest=getattr(
                args, "quality_preflight_record_digest", None
            ),
            expected_checkpoint_payload_digest=getattr(
                args, "warm_start_checkpoint_payload_digest", None
            ),
        )
        # Transfer must not inherit the source optimizer or RNG stream.
        _seed_runtime(args.seed, deterministic=args.deterministic, threads=args.threads)
    elif args.warm_start:
        raise ValueError("ppo-from-scratch forbids --warm-start")

    config = _ppo_config(args)
    total_updates = 0 if args.method == "supervised-only" else args.updates
    trainer = PPOTrainer(
        model,
        config,
        mode=Mode.IMPROVEMENT,
        total_updates=max(1, total_updates),
        device=device,
    )
    task_scheduler = _LineageEqualTaskScheduler(tasks, seed=args.seed)
    deployment_bank = None
    bank_training_tasks: dict[str, PreparedTask] = {}
    initializer_bank_access: dict[str, object] | None = None
    if bank_backed_policy and total_updates > 0:
        from isingfold.rl.complete_system import LACMinorminerInitializerBackend
        from isingfold.rl.data.prepared import load_prepared_partition
        from isingfold.rl.initializer_bank import (
            lac_runtime_implementation_manifest,
            load_initializer_bank,
        )

        public = load_prepared_partition(
            args.corpus,
            partition="train",
            include_evaluator=False,
        )
        if public.target_access is not None:
            raise RuntimeError("complete-system training public-bank load opened evaluator targets")
        if complete_config is None:  # pragma: no cover - guarded above
            raise RuntimeError("deployment bank config was not parsed")
        initializer_backend = LACMinorminerInitializerBackend()
        initializer_runtime = lac_runtime_implementation_manifest(initializer_backend)
        deployment_bank = load_initializer_bank(
            initializer_bank_path,
            expected_manifest_sha256=initializer_bank_pin,
            prepared_tasks=public.tasks,
            prepared_manifest_sha256=_corpus_manifest_digest(args.corpus),
            initializer=initializer_backend,
            runtime_implementation_manifest=initializer_runtime,
            config=complete_config,
            context=ctx,
        )
        expected_episode_count = total_updates * config.episodes_per_batch
        if (
            deployment_bank.plan.training_seed != args.seed
            or deployment_bank.plan.episode_schedule_start != 0
            or deployment_bank.plan.episode_schedule_stop_exclusive != expected_episode_count
        ):
            raise ValueError(
                "initializer bank differs from the exact training seed or PPO schedule"
            )
        for identity in deployment_bank.plan.tasks:
            matching = sorted(
                (
                    item
                    for item in tasks
                    if item.instance_id == identity.instance_id
                    and item.task_id in identity.source_task_ids
                ),
                key=lambda item: item.task_id,
            )
            if not matching:
                raise ValueError(
                    "initializer bank instance has no matching authenticated train target"
                )
            bank_training_tasks[identity.instance_id] = matching[0]
        if set(bank_training_tasks) != {
            identity.instance_id for identity in deployment_bank.plan.tasks
        }:
            raise ValueError("initializer bank target join is incomplete")
        initializer_bank_access = deployment_bank.access_receipt.as_dict()

    lineages = (
        tuple(sorted({identity.base_lineage for identity in deployment_bank.plan.tasks}))
        if deployment_bank is not None
        else task_scheduler.lineages
    )
    task_sampling = (
        dict(deployment_bank.plan.scheduler)
        if deployment_bank is not None
        else task_scheduler.receipt
    )
    task_sampling_digest = (
        deployment_bank.plan.record_digest if deployment_bank is not None else task_scheduler.digest
    )
    hyperparameters = {
        **_jsonable(config),
        "ppo_collection_schedule": _ppo_collection_schedule_contract(),
        "ppo_update_control": _ppo_update_control_contract(config),
        "total_updates": total_updates,
        "warm_start_payload_digest": warm_digest,
        "task_sampling": task_sampling,
        "task_sampling_digest": task_sampling_digest,
        "initializer_bank_access": initializer_bank_access,
        "initializer_bank_unused_for_supervised_only": bool(
            bank_backed_policy and total_updates == 0
        ),
        "deployment_initializer_bank": bank_backed_policy,
        "complete_system_policy_restart_mode": policy_restart_mode,
        "complete_system_selection_binding": complete_selection_binding,
    }
    contract = _experiment_contract(
        phase="rl-value",
        model_family=args.model_family,
        model=model,
        method=args.method,
        context=ctx,
        corpus_manifest_sha256=bundle.corpus_manifest_sha256,
        quality_authority=quality_authority,
        target_access=target_access,
        ground_partition_receipt=ground_partition_receipt,
        selector_digest=bundle.selector_digest,
        normalizer_digest=bundle.normalizer_digest,
        seed=args.seed,
        device_type=device.type,
        deterministic=args.deterministic,
        hyperparameters=hyperparameters,
        grid_manifest_sha256=getattr(args, "grid_manifest_sha256", None),
        grid_cell=getattr(args, "grid_cell", None),
        selection_receipt_sha256=getattr(args, "selection_receipt_sha256", None),
        selection_record_digest=getattr(args, "selection_record_digest", None),
        selection_mode=getattr(args, "selection_mode", None),
        gate_receipt_sha256=getattr(args, "gate_receipt_sha256", None),
        gate_record_digest=getattr(args, "gate_record_digest", None),
        gate_profile=getattr(args, "gate_profile", None),
        quality_preflight_receipt_sha256=getattr(args, "quality_preflight_receipt_sha256", None),
        quality_preflight_record_digest=getattr(args, "quality_preflight_record_digest", None),
    )
    run_dir = _prepare_run_dir(args.out, resume=args.resume)
    counters = {"updates": 0, "episodes": 0, "transitions": 0, "init_failures": 0}
    if args.resume:
        metadata = _resume_training_state(
            run_dir,
            model=model,
            optimizer=trainer.optimizer,
            contract=contract,
            bundle=bundle,
            lineages=lineages,
            expected_trainer_state_schema=PPO_TRAINER_STATE_SCHEMA,
        )
        counters.update(metadata.counters)
        trainer_state = metadata.trainer_state
        if trainer_state.get("updates_done") != counters["updates"]:
            raise ValueError("checkpoint trainer update state differs from progress counters")
        restored_lambda = trainer_state.get("lambda_f")
        if (
            isinstance(restored_lambda, bool)
            or not isinstance(restored_lambda, (int, float))
            or not math.isfinite(restored_lambda)
            or restored_lambda < 0.0
        ):
            raise ValueError("checkpoint trainer lambda_f is invalid")
        trainer.updates_done = int(trainer_state["updates_done"])
        trainer.lambda_f = float(restored_lambda)

    history_path = run_dir / "history.json"
    if history_path.exists():
        stored_history = _strict_json(history_path)
        if (
            set(stored_history) != {"schema_version", "updates"}
            or stored_history.get("schema_version") != 2
        ):
            raise ValueError("PPO history uses an incompatible update-control schema")
        history = stored_history.get("updates")
    else:
        history = []
    if not isinstance(history, list) or len(history) != counters["updates"]:
        raise ValueError("PPO history and checkpoint counters differ")

    bundle.model.to(device).eval()

    def factory(environment_seed: int, episode_schedule_index: int):
        if deployment_bank is not None:
            del environment_seed
            snapshot = deployment_bank.snapshot(episode_schedule_index)
            return deployment_bank.environment(
                episode_schedule_index,
                context=ctx,
                selector=bundle.model,
                reward_reads=args.reward_reads,
                training_task=bank_training_tasks[snapshot.instance_id],
                target_access=target_access,
                ground_partition_receipt=ground_partition_receipt,
            )
        item = task_scheduler.select(episode_schedule_index)
        return EmbeddingEnv(
            item.task,
            ctx,
            mode=Mode.IMPROVEMENT,
            initializer=item.initializer(),
            selector=bundle.model,
            reward_reads=args.reward_reads,
            seed=environment_seed,
        )

    if total_updates == 0:
        _atomic_json(history_path, {"schema_version": 2, "updates": history})
        metadata = _save_training_state(
            run_dir,
            model=model,
            optimizer=trainer.optimizer,
            contract=contract,
            bundle=bundle,
            lineages=lineages,
            counters=counters,
            phase="rl-value",
            model_family=args.model_family,
            method=args.method,
            complete=True,
            trainer_state={
                "lambda_f": float(trainer.lambda_f),
                "updates_done": int(trainer.updates_done),
            },
        )
        print(
            json.dumps(
                {
                    "method": args.method,
                    "updates": 0,
                    "checkpoint_payload_digest": metadata.payload_digest,
                }
            )
        )
        return

    print(
        json.dumps(
            {
                "model_family": args.model_family,
                "method": args.method,
                "parameters": model.parameter_count(),
                "train_tasks": len(tasks),
                "device": str(device),
                "resume_from_update": counters["updates"],
            }
        )
    )
    for update in range(counters["updates"], total_updates):
        snapshot = trainer.behaviour_snapshot()
        buffer, collection = collect(
            factory,
            snapshot,
            config,
            update_index=update,
            device=device,
        )
        if deployment_bank is not None:
            schedule_start = update * config.episodes_per_batch
            rejected_draws = sum(
                len(deployment_bank.draws(index)) - 1
                for index in range(
                    schedule_start,
                    schedule_start + config.episodes_per_batch,
                )
            )
            collection["init_failures"] = float(rejected_draws)
            collection["initializer_bank_rejected_draws"] = float(rejected_draws)
        if buffer.n_transitions <= 0:
            raise RuntimeError("complete-episode collection produced no transitions")
        replay_error = trainer.replay_check(buffer)
        logs = trainer.update(buffer)
        logs.update(
            {
                "update": update + 1,
                "pre_update_replay_max_abs_logprob_error": replay_error,
                **{f"collect_{key}": value for key, value in collection.items()},
            }
        )
        history.append(logs)
        counters["updates"] = update + 1
        counters["episodes"] += buffer.n_episodes
        counters["transitions"] += buffer.n_transitions
        counters["init_failures"] += int(collection.get("init_failures", 0.0))
        _atomic_json(history_path, {"schema_version": 2, "updates": history})
        _save_training_state(
            run_dir,
            model=model,
            optimizer=trainer.optimizer,
            contract=contract,
            bundle=bundle,
            lineages=lineages,
            counters=counters,
            phase="rl-value",
            model_family=args.model_family,
            method=args.method,
            complete=update + 1 == total_updates,
            trainer_state={
                "lambda_f": float(trainer.lambda_f),
                "updates_done": int(trainer.updates_done),
            },
        )
        print(
            json.dumps(
                {
                    key: round(value, 6) if isinstance(value, float) else value
                    for key, value in logs.items()
                }
            )
        )


def _load_evaluation_model(args: argparse.Namespace, bundle: SelectorBundle):
    import torch

    from isingfold.rl.checkpoint import PPO_TRAINER_STATE_SCHEMA, load_checkpoint

    checkpoint = Path(args.checkpoint)
    receipt = _strict_json(checkpoint.parent / "run.json")
    _verify_record(receipt, "training-run receipt")
    if (
        receipt.get("schema") != RUN_SCHEMA
        or receipt.get("schema_version") != RUN_SCHEMA_VERSION
        or receipt.get("phase") not in {"representation", "rl-value"}
        or receipt.get("complete") is not True
    ):
        raise ValueError("evaluation requires a completed scientific training run")
    model_family = receipt.get("model_family")
    if model_family not in REGISTERED_MODEL_FAMILIES:
        raise ValueError("training receipt names an unknown model family")
    contract = receipt.get("experiment_contract")
    normalizer = receipt.get("feature_normalizer")
    lineages = receipt.get("training_lineages")
    if (
        not isinstance(contract, dict)
        or not isinstance(normalizer, dict)
        or not isinstance(lineages, list)
    ):
        raise ValueError("training-run receipt is malformed")
    if contract.get("corpus_manifest_sha256") != _corpus_manifest_digest(args.corpus):
        raise ValueError("checkpoint belongs to a different prepared corpus")
    if receipt.get("selector_digest") != bundle.selector_digest or normalizer != bundle.normalizer:
        raise ValueError("checkpoint uses a different selector or normalizer")
    if receipt.get("proposal_version") != _proposal_version():
        raise ValueError("checkpoint uses a different proposal implementation")
    runtime_digest = receipt.get("runtime_implementation_digest")
    if not isinstance(runtime_digest, str) or len(runtime_digest) != 64:
        raise ValueError("training receipt has no valid runtime implementation digest")
    model_contract = contract.get("model")
    if not isinstance(model_contract, Mapping):
        raise ValueError("checkpoint experiment contract has no model identity")
    quality_prior_mode = model_contract.get("quality_prior_mode")
    if not isinstance(quality_prior_mode, str):
        raise ValueError("checkpoint model identity has no quality-prior mode")
    model = _model(model_family, quality_prior_mode=quality_prior_mode)
    for field_name in ("phase", "method", "model_family"):
        if contract.get(field_name) != receipt.get(field_name):
            raise ValueError(f"training receipt and experiment contract disagree on {field_name}")
    if model_contract != _model_identity(model, model_family):
        raise ValueError("checkpoint model implementation differs from its experiment contract")
    if receipt["phase"] == "rl-value":
        _require_current_ppo_collection_schedule(contract)
        _require_current_ppo_update_control(contract)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    metadata = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        expected_context=contract,
        expected_feature_normalizer=normalizer,
        expected_proposal_version=receipt["proposal_version"],
        expected_selector_digest=bundle.selector_digest,
        expected_training_lineages=lineages,
        expected_trainer_state_schema=(
            PPO_TRAINER_STATE_SCHEMA if receipt["phase"] == "rl-value" else {}
        ),
        expected_runtime_digest=runtime_digest,
        restore_rng=False,
        map_location="cpu",
    )
    if metadata.payload_digest != receipt.get("checkpoint_payload_digest"):
        raise ValueError("training receipt references a different checkpoint payload")
    if receipt.get("runtime_implementation_registry") != metadata.runtime_implementation_registry:
        raise ValueError("training receipt references a different runtime registry")
    if receipt.get("trainer_state") != metadata.trainer_state:
        raise ValueError("training receipt references different mutable trainer state")
    return model, receipt


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run paired held-out endpoints and write raw outcomes plus an aggregate receipt."""

    if args.partition == "test":
        raise PermissionError(
            "generic evaluate cannot open the sealed test partition; use the frozen "
            "evaluate-complete-system-cell path with exact RL-value-selection, config, "
            "population, receipt, and terminal-evidence pins"
        )

    bootstrap_bank = None
    bootstrap_plan = None
    scientific_authorization = getattr(args, "scientific_evaluation_authorization", None)
    if scientific_authorization is not None and getattr(args, "bootstrap_bank", None) is None:
        raise ValueError("registered Profile-I evaluation requires an authenticated bootstrap bank")
    if getattr(args, "bootstrap_bank", None) is not None:
        complete_config_path = getattr(args, "complete_config", None)
        if complete_config_path is None:
            raise ValueError("K=2 evaluation requires its complete-system config")
        bootstrap_protocol_preset = (
            "representation-validation"
            if isinstance(scientific_authorization, Mapping)
            and scientific_authorization.get("stage") == "representation"
            else "validation"
        )
        bootstrap_bank, bootstrap_plan = _load_pinned_evaluation_bootstrap_bank(
            args,
            protocol_preset=bootstrap_protocol_preset,
            config_path=complete_config_path,
        )

    from isingfold.rl import evaluate as evaluation_module
    from isingfold.rl.evaluate import (
        EVALUATION_RECEIPT_VERSION,
        ComparisonFamily,
        ComparisonSpec,
        EvaluationArm,
        baseline_method_metadata,
        quality_aware_controller,
        compare_arms,
        first_commit_controller,
        random_masked_controller,
        resource_first_controller,
        run_controller,
        secondary_metrics,
        torch_controller,
        write_episode_receipts,
    )
    from isingfold.rl.proposal import router_initializer

    quality_pin = _quality_attestation_pin(args)
    (
        prepared,
        quality_authority,
        target_access,
        ground_partition_receipt,
    ) = _load_quality_partition(
        args.corpus,
        partition=args.partition,
        pin=quality_pin,
        role="evaluation_partition",
    )
    if bootstrap_bank is None:
        population = [item.task for item in prepared]
    else:
        population = _preinitialization_population_tasks(
            prepared,
            manifest=_strict_json(Path(args.corpus) / "manifest.json"),
            partition=args.partition,
        )
    ctx = _context(args, args.corpus)
    if bool(getattr(args, "deployment_policy_context", False)):
        from isingfold.rl.complete_system import complete_policy_context

        ctx = complete_policy_context(ctx)
    bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
    if _bind_selector_quality_authority(bundle, prepared, pin=quality_pin) != quality_authority:
        raise ValueError("selector and evaluation partition authorities differ")
    device = _resolve_device(args.device)
    _seed_runtime(args.seed, deterministic=args.deterministic, threads=args.threads)
    model, training_receipt = _load_evaluation_model(args, bundle)
    training_contract = training_receipt["experiment_contract"]
    if scientific_authorization is not None:
        expected_authorization = {
            "stage": training_receipt["phase"],
            "grid_manifest_sha256": training_contract.get("grid_manifest_sha256"),
            "grid_cell": training_contract.get("grid_cell"),
            "partition": "validation",
        }
        if scientific_authorization != expected_authorization:
            raise ValueError(
                "scientific evaluation authorization differs from the training/grid contract"
            )
    runtime_implementation_registry = training_receipt.get("runtime_implementation_registry")
    runtime_implementation_digest = training_receipt.get("runtime_implementation_digest")
    if not isinstance(
        runtime_implementation_registry, dict
    ) or runtime_implementation_digest != stable_digest(runtime_implementation_registry):
        raise ValueError("evaluation training run has an inconsistent runtime registry")
    quality_preflight_receipt_sha256 = training_contract.get("quality_preflight_receipt_sha256")
    quality_preflight_record_digest = training_contract.get("quality_preflight_record_digest")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in (quality_preflight_receipt_sha256, quality_preflight_record_digest)
    ):
        raise ValueError("evaluation training run has no authenticated quality preflight")
    training_authority = _validate_quality_authority_binding(
        training_contract.get("quality_authority"),
        expected_role="training_partition",
    )
    if training_authority["global"] != quality_authority["global"]:
        raise ValueError("checkpoint belongs to a different pinned quality authority")
    if training_contract.get("environment") != _context_snapshot(ctx):
        raise ValueError(
            "checkpoint environment context differs from the requested evaluation context"
        )
    gate_profile = training_contract.get("gate_profile")
    gate_receipt_sha256 = training_contract.get("gate_receipt_sha256")
    gate_record_digest = training_contract.get("gate_record_digest")
    selection_receipt_sha256 = training_contract.get("selection_receipt_sha256")
    selection_record_digest = training_contract.get("selection_record_digest")
    selection_mode = training_contract.get("selection_mode")
    if training_receipt["phase"] == "representation" and (
        gate_profile not in {"profile-i", "profile-i+profile-c"}
        or not isinstance(gate_receipt_sha256, str)
        or len(gate_receipt_sha256) != 64
        or not isinstance(gate_record_digest, str)
        or len(gate_record_digest) != 64
    ):
        raise ValueError("representation evaluation requires an authenticated Profile-I gate")
    model.to(device).eval()
    bundle.model.to(device).eval()
    runtime_platform = _runtime_platform_identity()
    if device.type == "cuda":
        import torch

        inference_device_name = torch.cuda.get_device_name(device)
    elif device.type == "mps":
        inference_device_name = f"Apple-{runtime_platform['machine']}"
    else:
        inference_device_name = str(runtime_platform["processor"])
    complete_receipts: dict[str, Sequence[object]] | None = None
    if bootstrap_bank is None:
        common = {
            "initializer": router_initializer(),
            "selector": bundle.model,
            "reward_reads": args.audit_reads,
            "seed": args.seed,
            "repetitions": args.repetitions,
        }
        reference = run_controller(population, ctx, first_commit_controller, **common)
        arms = {
            "return_initial": reference,
            "random_masked": run_controller(population, ctx, random_masked_controller, **common),
            "classical_resource_first": run_controller(
                population, ctx, resource_first_controller, **common
            ),
            "classical_quality_aware": run_controller(
                population, ctx, quality_aware_controller, **common
            ),
            "policy": run_controller(
                population,
                ctx,
                torch_controller(model, device=device, greedy=args.greedy),
                **common,
            ),
        }
    else:
        from isingfold.rl.complete_system import (
            CompletePopulationIdentity,
            CompleteSystemConfig,
            FrozenComponentIdentity,
            LACMinorminerInitializerBackend,
            run_complete_system,
            task_population_digest,
        )

        assert bootstrap_plan is not None
        complete_config = CompleteSystemConfig.from_mapping(_strict_json(args.complete_config))
        if complete_config.audit_reads != args.audit_reads:
            raise ValueError("K=2 complete-system config changes validation audit reads")
        manifest_sha = _corpus_manifest_digest(args.corpus)
        evaluation_strata, confirmatory_design = evaluation_contract_from_prepared(
            prepared,
            corpus_design_receipt=load_prepared_corpus_design(args.corpus),
            partition="val",
            noninferiority_margin=float(args.margin),
        )
        population_identity = CompletePopulationIdentity(
            population_id=(
                "preinitialization-"
                + stable_digest(
                    {
                        "manifest": manifest_sha,
                        "partition": "val",
                        "evaluation_seed": args.seed,
                        "repetitions": args.repetitions,
                    }
                )
            ),
            source_manifest_sha256=manifest_sha,
            task_payload_sha256=task_population_digest(population),
            expected_instances=tuple(
                sorted((task.lineage or task.name, task.name) for task in population)
            ),
            expected_repetitions=args.repetitions,
            evaluation_strata=evaluation_strata,
            confirmatory_design=confirmatory_design,
            evaluation_seed=args.seed,
        )
        selector_fit = _strict_json(bundle.root / "fit_receipt.json")
        _verify_record(selector_fit, "selector fit receipt")
        selector_identity = FrozenComponentIdentity(
            component_id="isingfold-if-q3-s0-strength-selector",
            version=str(bundle.model.version),
            implementation=(
                f"{type(bundle.model).__module__}.{type(bundle.model).__qualname__}:"
                f"fit-{selector_fit['record_digest']}"
            ),
            artifact_sha256=_sha256_file(bundle.root / "selector.pt"),
        )
        controller_functions = {
            "return_initial": first_commit_controller,
            "random_masked": random_masked_controller,
            "classical_resource_first": resource_first_controller,
            "classical_quality_aware": quality_aware_controller,
            "policy": torch_controller(model, device=device, greedy=args.greedy),
        }
        initializer_backend = LACMinorminerInitializerBackend()
        arms = {}
        complete_receipts = {}
        implementation_sha = _sha256_file(evaluation_module.__file__)
        for arm_name, arm_controller in controller_functions.items():
            if arm_name == "policy":
                component_identity = FrozenComponentIdentity(
                    component_id=(
                        f"isingfold-policy/{training_receipt['model_family']}/"
                        f"{training_receipt['method']}"
                    ),
                    version=str(training_receipt["checkpoint_payload_digest"]),
                    implementation=(
                        f"{training_contract['model']['implementation']}:"
                        f"runtime-{runtime_implementation_digest}"
                    ),
                    artifact_sha256=_sha256_file(args.checkpoint),
                )
            else:
                component_identity = FrozenComponentIdentity(
                    component_id=f"isingfold-controller/{arm_name}",
                    version="1",
                    implementation=(f"{arm_controller.__module__}.{arm_controller.__qualname__}"),
                    artifact_sha256=implementation_sha,
                )
            arm_outcomes, arm_complete_receipts = run_complete_system(
                population,
                ctx,
                initializer_backend,
                complete_config,
                controller=arm_controller,
                selector=bundle.model,
                controller_identity=component_identity,
                selector_identity=selector_identity,
                population=population_identity,
                seed=args.seed,
                repetitions=args.repetitions,
                max_steps=ctx.caps.decisions,
                validation_bootstrap_bank=bootstrap_bank,
                same_support_contract_digest=(bootstrap_plan.same_support_contract_digest),
                bootstrap_consumer_id=(
                    f"{training_receipt['phase']}/{training_contract['grid_cell']}/{arm_name}"
                ),
            )
            arms[arm_name] = arm_outcomes
            complete_receipts[arm_name] = arm_complete_receipts
    methods = baseline_method_metadata()
    protocol_metadata = {
        "population": f"{bundle.corpus_manifest_sha256}:{args.partition}",
        "quality_authority": quality_authority,
        "target_access": target_access,
        "ground_partition_receipt": ground_partition_receipt,
        "budget": stable_digest(_context_snapshot(ctx)),
        "selector": bundle.selector_digest,
        "proposal": _proposal_version(),
        "support": "same-conditional-generator-and-exact-mask-contract",
        "policy_seed": args.seed,
        "evaluator_reads": args.audit_reads,
        "initializer": (
            "authenticated-profile-i"
            if bootstrap_bank is None
            else "authenticated-persistent-upfront-lac-cache-k2-v2"
        ),
        "repetitions": args.repetitions,
        "inference_device_type": device.type,
        "inference_device_name": inference_device_name,
        "inference_threads": args.threads,
        "runtime_platform": runtime_platform,
        "baseline_controller_registry": stable_digest(methods),
        "evaluation_receipt_schema": EVALUATION_RECEIPT_VERSION,
        "evaluation_implementation_sha256": _sha256_file(evaluation_module.__file__),
        "quality_preflight_receipt_sha256": quality_preflight_receipt_sha256,
        "quality_preflight_record_digest": quality_preflight_record_digest,
        "runtime_implementation_registry": runtime_implementation_registry,
        "runtime_implementation_digest": runtime_implementation_digest,
    }
    if bootstrap_bank is not None:
        assert bootstrap_plan is not None
        protocol_metadata.update(
            {
                "bootstrap_bank_access": bootstrap_bank.access_receipt.as_dict(),
                "same_support_contract_digest": (bootstrap_plan.same_support_contract_digest),
            }
        )
    methods["policy"] = {
        "method_id": f"{training_receipt['model_family']}:{training_receipt['method']}",
        "selection_rule": (
            "greedy-masked-policy" if args.greedy else "categorical-temperature-one"
        ),
        "tie_break": "smallest-action-index" if args.greedy else "registered-episode-rng",
        "support_contract": "environment-materialised-masked-actions",
        "online_evaluator_feedback": False,
        "evaluator_oracle": False,
        "checkpoint_payload_digest": training_receipt["checkpoint_payload_digest"],
    }
    evaluation_arms = {
        name: EvaluationArm(
            name,
            "baseline" if name != "policy" else "learned",
            tuple(rows),
            {**protocol_metadata, "method": methods[name]},
        )
        for name, rows in arms.items()
    }
    method = training_receipt.get("method")
    learned_family = (
        ComparisonFamily.PPO_INCREMENT
        if method in {"ppo-warm-start", "ppo-from-scratch"}
        else ComparisonFamily.SUPERVISED_REPRESENTATION
    )

    def comparison(
        name: str,
        treatment: str,
        reference_name: str,
        family: ComparisonFamily,
    ) -> dict[str, object]:
        return compare_arms(
            evaluation_arms,
            ComparisonSpec(
                name,
                treatment=treatment,
                reference=reference_name,
                family=family,
                matched_metadata=tuple(protocol_metadata),
                margin=args.margin,
                seed=args.seed,
            ),
        ).as_dict()

    comparisons = {
        "policy_vs_return_initial": comparison(
            "policy_vs_return_initial", "policy", "return_initial", learned_family
        ),
        "random_vs_return_initial": comparison(
            "random_vs_return_initial",
            "random_masked",
            "return_initial",
            ComparisonFamily.PROPOSAL_HEADROOM,
        ),
        "resource_first_vs_random": comparison(
            "resource_first_vs_random",
            "classical_resource_first",
            "random_masked",
            ComparisonFamily.CLASSICAL_SCORING,
        ),
        "quality_aware_vs_resource_first": comparison(
            "quality_aware_vs_resource_first",
            "classical_quality_aware",
            "classical_resource_first",
            ComparisonFamily.CLASSICAL_SCORING,
        ),
        "policy_vs_resource_first": comparison(
            "policy_vs_resource_first",
            "policy",
            "classical_resource_first",
            ComparisonFamily.LEARNED_VS_CLASSICAL,
        ),
        "policy_vs_quality_aware": comparison(
            "policy_vs_quality_aware",
            "policy",
            "classical_quality_aware",
            ComparisonFamily.LEARNED_VS_CLASSICAL,
        ),
    }
    report: dict[str, object] = {
        "schema": (
            "isingfold.paired-evaluation"
            if scientific_authorization is not None
            else "isingfold.diagnostic-paired-evaluation"
        ),
        "schema_version": (
            EVALUATION_REPORT_SCHEMA_VERSION if scientific_authorization is not None else 1
        ),
        "population": len(population),
        "partition": args.partition,
        "repetitions": args.repetitions,
        "model_family": training_receipt["model_family"],
        "training_method": training_receipt["method"],
        "training_seed": training_contract["seed"],
        "grid_cell": training_contract["grid_cell"],
        "grid_manifest_sha256": training_contract["grid_manifest_sha256"],
        "selection_receipt_sha256": selection_receipt_sha256,
        "selection_record_digest": selection_record_digest,
        "selection_mode": selection_mode,
        "gate_profile": gate_profile,
        "gate_receipt_sha256": gate_receipt_sha256,
        "gate_record_digest": gate_record_digest,
        "evaluation_seed": args.seed,
        "inference_device_type": protocol_metadata["inference_device_type"],
        "inference_device_name": protocol_metadata["inference_device_name"],
        "runtime_platform": protocol_metadata["runtime_platform"],
        "checkpoint_payload_digest": training_receipt["checkpoint_payload_digest"],
        "selector_digest": bundle.selector_digest,
        "corpus_manifest_sha256": bundle.corpus_manifest_sha256,
        "quality_authority": quality_authority,
        "target_access": target_access,
        "ground_partition_receipt": ground_partition_receipt,
        "quality_preflight_receipt_sha256": quality_preflight_receipt_sha256,
        "quality_preflight_record_digest": quality_preflight_record_digest,
        "runtime_implementation_registry": runtime_implementation_registry,
        "runtime_implementation_digest": runtime_implementation_digest,
        "deployment_rule": "greedy" if args.greedy else "categorical-temperature-one",
        "matched_metadata": protocol_metadata,
        "methods": methods,
        "external_system_boundary": (
            "Stock minorminer is not a same-support controller. Evaluate it through a "
            "separate equal-budget external-system protocol; no minorminer outcome is "
            "fabricated by this command."
        ),
        "comparisons": comparisons,
    }
    if bootstrap_bank is not None:
        assert bootstrap_plan is not None
        report["bootstrap_bank_access"] = bootstrap_bank.access_receipt.as_dict()
        report["same_support_contract_digest"] = bootstrap_plan.same_support_contract_digest
    if scientific_authorization is None:
        report["scientific_eligible"] = False
        report["diagnostic_reason"] = (
            "direct generic evaluation lacks an internal staged-grid authorization"
        )
    for name, outcomes in arms.items():
        report[name] = secondary_metrics(outcomes)

    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"evaluation output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        report["arm_receipts"] = {
            name: {
                "path": f"{name}.jsonl",
                "sha256": write_episode_receipts(temporary / f"{name}.jsonl", outcomes),
                "count": len(outcomes),
                "method": methods[name],
                "protocol": protocol_metadata,
            }
            for name, outcomes in arms.items()
        }
        if complete_receipts is not None:
            from isingfold.rl.complete_system import write_complete_system_receipts

            report["complete_system_receipts"] = {
                name: {
                    "path": f"{name}.complete.jsonl",
                    "sha256": write_complete_system_receipts(
                        temporary / f"{name}.complete.jsonl", rows
                    ),
                    "count": len(rows),
                }
                for name, rows in complete_receipts.items()
            }
        _atomic_json(temporary / "report.json", _with_digest(report))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps(report, indent=1, default=float))


def cmd_evaluate_external(args: argparse.Namespace) -> None:
    """Run stock minorminer as an external complete system, never a support ablation."""

    if args.partition == "test":
        raise PermissionError(
            "generic evaluate-external cannot open the sealed test partition; use "
            "evaluate-external-complete-system with exact grid/config/learned-pairing pins"
        )

    from isingfold.rl import external as external_module
    from isingfold.rl.evaluate import secondary_metrics, write_episode_receipts
    from isingfold.rl.external import (
        ExternalBaselineConfig,
        StockMinorminerBackend,
        external_method_metadata,
        run_external_system,
        write_external_attempt_receipts,
    )

    config_path = Path(args.config)
    config = ExternalBaselineConfig.from_mapping(_strict_json(config_path))
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"external evaluation output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    availability = StockMinorminerBackend.probe(expected_version=config.expected_backend_version)
    common_report: dict[str, object] = {
        "schema": "isingfold.external-system-evaluation",
        "schema_version": 1,
        "comparison_scope": "external-complete-system",
        "same_support_causal_ablation": False,
        "requested_backend": config.backend,
        "config": config.as_dict(),
        "config_digest": config.digest,
        "config_file_sha256": _sha256_file(config_path),
        "requested_corpus": str(args.corpus),
        "requested_selector": str(args.selector),
        "partition": args.partition,
        "repetitions": args.repetitions,
        "evaluation_seed": args.seed,
        "availability": availability.as_dict(),
        "external_implementation_sha256": _sha256_file(external_module.__file__),
        "failure_estimand": "unconditional-all-sealed-attempts-failure-is-utility-zero",
        "no_fallback": True,
        "scientific_eligible": False,
        "diagnostic_reason": (
            "free diagnostic seed/repetition interface; not a frozen paper endpoint"
        ),
    }
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        if not availability.available or availability.identity is None:
            report = _with_digest(
                {
                    **common_report,
                    "status": "unavailable",
                    "population": None,
                    "method": None,
                    "summary": None,
                    "artifacts": {},
                }
            )
            _atomic_json(temporary / "report.json", report)
            os.replace(temporary, destination)
            print(json.dumps(report, indent=1, default=float))
            return

        quality_pin = _quality_attestation_pin(args)
        (
            prepared,
            quality_authority,
            target_access,
            ground_partition_receipt,
        ) = _load_quality_partition(
            args.corpus,
            partition=args.partition,
            pin=quality_pin,
            role="evaluation_partition",
        )
        population = [item.task for item in prepared]
        ctx = _context(args, args.corpus)
        bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
        if _bind_selector_quality_authority(bundle, prepared, pin=quality_pin) != quality_authority:
            raise ValueError("selector and external diagnostic authorities differ")
        # Native minorminer threads are fixed by the external config.  One CPU thread also
        # keeps frozen-selector inference deterministic and visible in the receipt.
        _seed_runtime(args.seed, deterministic=True, threads=1)
        bundle.model.eval()
        backend = StockMinorminerBackend(availability.identity)
        outcomes, attempt_receipts = run_external_system(
            population,
            ctx,
            backend,
            config,
            selector=bundle.model,
            seed=args.seed,
            repetitions=args.repetitions,
        )
        if len(outcomes) != len(population) * args.repetitions:
            raise RuntimeError("external baseline omitted a task/repetition system attempt")
        outcome_sha = write_episode_receipts(temporary / "outcomes.jsonl", outcomes)
        attempt_sha = write_external_attempt_receipts(
            temporary / "attempts.jsonl", attempt_receipts
        )
        method = external_method_metadata(config, availability.identity)
        online_times = [float(item.online_seconds) for item in attempt_receipts]
        embedding_times = [float(item.embedding_phase_seconds) for item in attempt_receipts]
        postprocessing_times = [float(item.postprocessing_seconds) for item in attempt_receipts]
        evaluator_times = [
            float(item.evaluator_seconds)
            for item in attempt_receipts
            if item.evaluator_seconds is not None
        ]
        report = _with_digest(
            {
                **common_report,
                "status": "complete",
                "population": len(population),
                "corpus_manifest_sha256": bundle.corpus_manifest_sha256,
                "quality_authority": quality_authority,
                "target_access": target_access,
                "ground_partition_receipt": ground_partition_receipt,
                "selector_digest": bundle.selector_digest,
                "selector_fit_receipt_sha256": _sha256_file(bundle.root / "fit_receipt.json"),
                "context": _context_snapshot(ctx),
                "context_digest": content_digest(_context_snapshot(ctx)),
                "runtime_platform": _runtime_platform_identity(),
                "method": method,
                "summary": {
                    **{
                        key: (
                            None
                            if key
                            in {
                                "work_route_expansions_mean",
                                "work_materializations_mean",
                                "work_cut_edge_visits_mean",
                            }
                            else value
                        )
                        for key, value in secondary_metrics(outcomes).items()
                    },
                    "embedding_phase_seconds_mean": float(np.mean(embedding_times)),
                    "postprocessing_seconds_mean": float(np.mean(postprocessing_times)),
                    "online_seconds_mean": float(np.mean(online_times)),
                    "online_seconds_total": float(sum(online_times)),
                    "evaluator_seconds_mean_valid": (
                        float(np.mean(evaluator_times)) if evaluator_times else None
                    ),
                },
                "artifacts": {
                    "outcomes": {
                        "path": "outcomes.jsonl",
                        "sha256": outcome_sha,
                        "count": len(outcomes),
                    },
                    "attempts": {
                        "path": "attempts.jsonl",
                        "sha256": attempt_sha,
                        "count": len(attempt_receipts),
                    },
                },
            }
        )
        _atomic_json(temporary / "report.json", report)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(json.dumps(report, indent=1, default=float))


def cmd_evaluate_external_complete_system(args: argparse.Namespace) -> None:
    """Run stock minorminer on the exact frozen complete-system test census."""

    from isingfold.rl import external_pairing as external_pairing_module
    from isingfold.rl.complete_system import (
        CompletePopulationIdentity,
        CompleteSystemConfig,
        FrozenComponentIdentity,
        task_population_digest,
    )
    from isingfold.rl.checkpoint import runtime_implementation_registry
    from isingfold.rl.evaluate import secondary_metrics
    from isingfold.rl.external import StockMinorminerBackend
    from isingfold.rl.external_pairing import (
        EXTERNAL_COMPLETE_REPORT_SCHEMA,
        EXTERNAL_COMPLETE_REPORT_VERSION,
        AuthenticatedExternalCompleteRun,
        ExternalCompleteSystemConfig,
        context_snapshot,
        external_complete_summary,
        read_external_complete_outcomes,
        read_external_complete_receipts,
        read_external_complete_evidence,
        runtime_identity,
        run_external_complete_system,
        write_external_complete_evidence,
        write_external_complete_outcomes,
        write_external_complete_receipts,
    )
    from isingfold.rl.external_tuning import (
        ExternalTuningExecutionBinding,
        load_external_tuning_registry,
        load_external_tuning_selection,
    )

    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"external complete-system output already exists: {destination}")
    grid, grid_digest = _load_grid(args.grid)
    protocol = grid["complete_system_evaluation"]
    external_config_path = Path(args.config)
    external_config_payload = _strict_json(external_config_path)
    external_config = ExternalCompleteSystemConfig.from_mapping(external_config_payload)
    learned_config_path = Path(args.learned_config)
    learned_config_payload = _strict_json(learned_config_path)
    learned_config = CompleteSystemConfig.from_mapping(learned_config_payload)
    _validate_registered_complete_config(
        protocol,
        arm="external",
        semantic_digest=external_config.digest,
        file_sha256=_sha256_file(external_config_path),
    )
    _validate_registered_complete_config(
        protocol,
        arm="learned",
        semantic_digest=learned_config.digest,
        file_sha256=_sha256_file(learned_config_path),
    )
    tuning_registry = load_external_tuning_registry(
        args.tuning_registry,
        expected_file_sha256=args.expected_tuning_registry_sha256,
        grid_path=args.grid,
        external_config_path=args.config,
    )
    tuning_selection = load_external_tuning_selection(
        args.external_tuning_selection,
        expected_file_sha256=args.expected_external_tuning_selection_sha256,
        registry=tuning_registry,
    )
    tuning_execution = ExternalTuningExecutionBinding.for_deployment(tuning_selection)
    runtime_registry = runtime_implementation_registry()
    if content_digest(runtime_registry) != (tuning_selection.runtime_implementation_digest):
        raise ValueError("publication runtime sources differ from validation baseline tuning")
    if args.index not in range(3):
        raise ValueError("external complete-system index must be zero through two")
    training_seed = (1103, 2207, 3301)[args.index]
    availability = StockMinorminerBackend.probe(
        expected_version=external_config.expected_backend_version
    )
    if not availability.available or availability.identity is None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        report = _with_digest(
            {
                "schema": EXTERNAL_COMPLETE_REPORT_SCHEMA,
                "schema_version": EXTERNAL_COMPLETE_REPORT_VERSION,
                "status": "unavailable",
                "sealed_test_opened": False,
                "no_fallback": True,
                "availability": availability.as_dict(),
                "grid_manifest_sha256": grid_digest,
                "external_config": external_config.as_dict(),
                "external_config_digest": external_config.digest,
                "external_config_file_sha256": _sha256_file(external_config_path),
                "learned_config": learned_config.as_dict(),
                "learned_config_digest": learned_config.digest,
                "learned_config_file_sha256": _sha256_file(learned_config_path),
                "external_tuning_execution": tuning_execution.as_dict(),
                "external_tuning_execution_digest": tuning_execution.digest,
                "artifacts": {},
            }
        )
        destination.mkdir(parents=True)
        _atomic_json(destination / "report.json", report)
        print(json.dumps(report, indent=1, default=float))
        return

    manifest_path = Path(args.corpus) / "manifest.json"
    manifest = _strict_json(manifest_path)
    manifest_sha = _sha256_file(manifest_path)
    if manifest_sha != tuning_selection.source_manifest_sha256:
        raise ValueError("publication corpus differs from the corpus used for baseline tuning")
    quality_pin = _quality_attestation_pin(args)
    (
        prepared,
        quality_authority,
        target_access,
        ground_partition_receipt,
    ) = _load_quality_partition(
        args.corpus,
        partition=str(protocol["partition"]),
        pin=quality_pin,
        role="evaluation_partition",
    )
    tasks = _preinitialization_population_tasks(
        prepared,
        manifest=manifest,
        partition=str(protocol["partition"]),
    )
    evaluation_strata, confirmatory_design = evaluation_contract_from_prepared(
        prepared,
        corpus_design_receipt=load_prepared_corpus_design(args.corpus),
        partition=str(protocol["partition"]),
        noninferiority_margin=float(protocol["feasibility_noninferiority_margin"]),
    )
    context = _context(args, args.corpus)
    external_config.validate_symmetric_envelope(learned_config, context)
    if stable_digest(context_snapshot(context)) != tuning_selection.context_digest:
        raise ValueError("publication context differs from the context used for baseline tuning")
    if external_config.audit_reads != protocol["audit_reads"]:
        raise ValueError("external config and complete-system protocol audit reads differ")
    bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
    if _bind_selector_quality_authority(bundle, prepared, pin=quality_pin) != quality_authority:
        raise ValueError("selector and external test authorities differ")
    selector_fit_path = bundle.root / "fit_receipt.json"
    selector_fit = _strict_json(selector_fit_path)
    _verify_record(selector_fit, "selector fit receipt")
    selector_identity = FrozenComponentIdentity(
        component_id="isingfold-if-q3-s0-strength-selector",
        version=str(bundle.model.version),
        implementation=(
            f"{type(bundle.model).__module__}.{type(bundle.model).__qualname__}:"
            f"fit-{selector_fit['record_digest']}"
        ),
        artifact_sha256=_sha256_file(bundle.root / "selector.pt"),
    )
    if content_digest(selector_identity.as_dict()) != tuning_selection.selector_digest:
        raise ValueError("publication selector differs from the selector used for baseline tuning")
    tuning_quality_authority = _validate_quality_authority_binding(
        tuning_selection.quality_authority,
        expected_role="evaluation_partition",
    )
    if tuning_quality_authority["global"] != quality_authority["global"]:
        raise ValueError("publication quality authority differs from validation baseline tuning")
    evaluation_seed = int(protocol["evaluation_seed"])
    repetitions = int(protocol["repetitions"])
    population = CompletePopulationIdentity(
        population_id=(
            "preinitialization-"
            + stable_digest(
                {
                    "manifest": manifest_sha,
                    "partition": protocol["partition"],
                    "evaluation_seed": evaluation_seed,
                    "repetitions": repetitions,
                }
            )
        ),
        source_manifest_sha256=manifest_sha,
        task_payload_sha256=task_population_digest(tasks),
        expected_instances=tuple(sorted((task.lineage or task.name, task.name) for task in tasks)),
        expected_repetitions=repetitions,
        evaluation_strata=evaluation_strata,
        confirmatory_design=confirmatory_design,
        evaluation_seed=evaluation_seed,
    )
    device = _resolve_device(args.device)
    _seed_runtime(evaluation_seed, deterministic=args.deterministic, threads=args.threads)
    bundle.model.to(device).eval()
    runtime_platform = _runtime_platform_identity()
    compute_identity = runtime_identity(
        runtime_platform=runtime_platform,
        inference_device_type=device.type,
        inference_device_name=_inference_device_name(device, runtime_platform),
        inference_threads=args.threads,
        deterministic=args.deterministic,
    )
    backend = StockMinorminerBackend(availability.identity)
    outcomes, receipts = run_external_complete_system(
        tasks,
        context,
        backend,
        external_config,
        learned_config=learned_config,
        selector=bundle.model,
        selector_identity=selector_identity,
        population=population,
        quality_authority=quality_authority,
        seed=evaluation_seed,
        repetitions=repetitions,
        training_seed_index=args.index,
        training_seed=training_seed,
        compute_identity=compute_identity,
        tuning_execution=tuning_execution,
    )
    expected_attempts = len(tasks) * repetitions
    if len(outcomes) != expected_attempts or len(receipts) != expected_attempts:
        raise RuntimeError("external complete runner changed its sealed attempt census")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.external-", dir=destination.parent)
    )
    try:
        outcome_sha = write_external_complete_outcomes(temporary / "outcomes.jsonl", outcomes)
        receipt_sha = write_external_complete_receipts(
            temporary / "external_receipts.jsonl", receipts
        )
        evidence_sha = write_external_complete_evidence(
            temporary / "terminal_evidence.jsonl", receipts
        )
        recovered_outcomes = read_external_complete_outcomes(
            temporary / "outcomes.jsonl", expected_sha256=outcome_sha
        )
        recovered_receipts = read_external_complete_receipts(
            temporary / "external_receipts.jsonl",
            outcomes=recovered_outcomes,
            expected_backend=availability.identity,
            expected_selector=selector_identity,
            expected_population=population,
            expected_config_digest=external_config.digest,
            expected_learned_config_digest=learned_config.digest,
            expected_context_digest=stable_digest(context_snapshot(context)),
            expected_quality_authority_digest=content_digest(quality_authority),
            expected_work_cap=context.caps,
            expected_training_seed_index=args.index,
            expected_training_seed=training_seed,
            expected_runtime_identity_digest=content_digest(compute_identity),
            expected_tuning_execution=tuning_execution,
            expected_sha256=receipt_sha,
        )
        recovered_evidence = read_external_complete_evidence(
            temporary / "terminal_evidence.jsonl",
            receipts=recovered_receipts,
            tasks=tasks,
            context=context,
            expected_sha256=evidence_sha,
        )
        report_payload: dict[str, object] = {
            "schema": EXTERNAL_COMPLETE_REPORT_SCHEMA,
            "schema_version": EXTERNAL_COMPLETE_REPORT_VERSION,
            "status": "complete",
            "partition": "test",
            "sealed_test_opened": True,
            "no_fallback": True,
            "evaluation_protocol": protocol,
            "training_seed_index": args.index,
            "training_seed": training_seed,
            "grid_manifest_sha256": grid_digest,
            "source_corpus_manifest_sha256": manifest_sha,
            "quality_authority": quality_authority,
            "target_access": target_access,
            "ground_partition_receipt": ground_partition_receipt,
            "context": context_snapshot(context),
            "context_digest": stable_digest(context_snapshot(context)),
            "work_cap": context.caps.as_dict(),
            "population": population.as_dict(),
            "population_digest": population.digest,
            "selector": selector_identity.as_dict(),
            "selector_digest": content_digest(selector_identity.as_dict()),
            "external_tuning_execution": tuning_execution.as_dict(),
            "external_tuning_execution_digest": tuning_execution.digest,
            "backend": availability.identity.as_dict(),
            "external_config": external_config.as_dict(),
            "external_config_digest": external_config.digest,
            "external_config_file_sha256": _sha256_file(external_config_path),
            "learned_config": learned_config.as_dict(),
            "learned_config_digest": learned_config.digest,
            "learned_config_file_sha256": _sha256_file(learned_config_path),
            **dict(compute_identity),
            "runtime_implementation_registry": runtime_registry,
            "runtime_implementation_digest": content_digest(runtime_registry),
            "external_pairing_implementation_sha256": _sha256_file(
                external_pairing_module.__file__
            ),
            "method": {
                "method_id": availability.identity.method_id,
                "comparison_scope": "paired-external-complete-system",
                "selection_rule": tuning_execution.candidate.candidate_ranking,
                "online_evaluator_feedback": False,
                "total_online_wallclock_seconds": external_config.online_wallclock_seconds,
                "final_evaluator": "one-fresh-independent-4096-read-block-after-selection",
                "native_internal_work": "unavailable-retained-as-null",
                "adapter_process_model": ("one-persistent-python-worker-per-task-repetition"),
                "latency_scope": (
                    "same-machine-rowwise-total-online-wallclock-includes-adapter-startup"
                ),
            },
            "summary": external_complete_summary(recovered_outcomes, recovered_receipts),
            "artifacts": {
                "receipts": {
                    "path": "external_receipts.jsonl",
                    "sha256": receipt_sha,
                    "count": len(receipts),
                },
                "terminal_evidence": {
                    "path": "terminal_evidence.jsonl",
                    "sha256": evidence_sha,
                    "count": len(receipts),
                },
                "outcomes": {
                    "path": "outcomes.jsonl",
                    "sha256": outcome_sha,
                    "count": len(outcomes),
                },
            },
        }
        report = _with_digest(report_payload)
        AuthenticatedExternalCompleteRun(
            report=report,
            receipts=tuple(recovered_receipts),
            evidence=recovered_evidence,
            receipt_file_sha256=receipt_sha,
            evidence_file_sha256=evidence_sha,
            outcome_file_sha256=outcome_sha,
        )
        _atomic_json(temporary / "report.json", report)
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(
        json.dumps(
            {
                "population": len(tasks),
                "attempts": len(outcomes),
                "metrics": secondary_metrics(outcomes),
                "out": str(destination),
            },
            indent=1,
        )
    )


def cmd_evaluate_external_tuning_cell(args: argparse.Namespace) -> None:
    """Run one finite stock-minorminer candidate/seed cell on validation only."""

    from isingfold.rl.checkpoint import runtime_implementation_registry
    from isingfold.rl.complete_system import (
        CompletePopulationIdentity,
        CompleteSystemConfig,
        FrozenComponentIdentity,
        task_population_digest,
    )
    from isingfold.rl.external import StockMinorminerBackend
    from isingfold.rl.external_pairing import (
        ExternalCompleteSystemConfig,
        context_snapshot,
        read_external_complete_evidence,
        read_external_complete_outcomes,
        read_external_complete_receipts,
        run_external_complete_system,
        runtime_identity,
        write_external_complete_evidence,
        write_external_complete_outcomes,
        write_external_complete_receipts,
    )
    from isingfold.rl.external_tuning import (
        ExternalTuningExecutionBinding,
        build_external_tuning_report,
        load_external_tuning_registry,
    )

    registry = load_external_tuning_registry(
        args.registry,
        expected_file_sha256=args.expected_registry_sha256,
        grid_path=args.grid,
        external_config_path=args.external_config,
    )
    candidate = registry.candidate_for_index(args.index)
    tuning_seed = registry.seed_for_index(args.tuning_seed_index)
    destination = Path(args.out) / candidate.candidate_id / f"seed-{args.tuning_seed_index}"
    if destination.exists():
        raise FileExistsError(f"external tuning cell output already exists: {destination}")

    grid, grid_digest = _load_grid(args.grid)
    if grid_digest != registry.grid_file_sha256:
        raise ValueError("external tuning registry and staged grid file identities differ")
    protocol = grid["complete_system_evaluation"]
    external_config_path = Path(args.external_config)
    external_config = ExternalCompleteSystemConfig.from_mapping(_strict_json(external_config_path))
    learned_config_path = Path(args.learned_config)
    learned_config = CompleteSystemConfig.from_mapping(_strict_json(learned_config_path))
    _validate_registered_complete_config(
        protocol,
        arm="external",
        semantic_digest=external_config.digest,
        file_sha256=_sha256_file(external_config_path),
    )
    _validate_registered_complete_config(
        protocol,
        arm="learned",
        semantic_digest=learned_config.digest,
        file_sha256=_sha256_file(learned_config_path),
    )
    availability = StockMinorminerBackend.probe(
        expected_version=external_config.expected_backend_version
    )
    if not availability.available or availability.identity is None:
        raise RuntimeError(
            "the preregistered stock-minorminer backend is unavailable; "
            "validation tuning has no fallback"
        )

    manifest_path = Path(args.corpus) / "manifest.json"
    manifest = _strict_json(manifest_path)
    manifest_sha = _sha256_file(manifest_path)
    quality_pin = _quality_attestation_pin(args)
    (
        prepared,
        quality_authority,
        target_access,
        ground_partition_receipt,
    ) = _load_quality_partition(
        args.corpus,
        partition="val",
        pin=quality_pin,
        role="evaluation_partition",
    )
    tasks = _preinitialization_population_tasks(
        prepared,
        manifest=manifest,
        partition="val",
    )
    evaluation_strata, confirmatory_design = evaluation_contract_from_prepared(
        prepared,
        corpus_design_receipt=load_prepared_corpus_design(args.corpus),
        partition="val",
        noninferiority_margin=float(protocol["feasibility_noninferiority_margin"]),
    )
    context = _context(args, args.corpus)
    external_config.validate_symmetric_envelope(learned_config, context)
    if external_config.audit_reads != registry.audit_reads:
        raise ValueError("external config and tuning registry audit-read blocks differ")
    bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
    if _bind_selector_quality_authority(bundle, prepared, pin=quality_pin) != quality_authority:
        raise ValueError("selector and external tuning authorities differ")
    selector_fit_path = bundle.root / "fit_receipt.json"
    selector_fit = _strict_json(selector_fit_path)
    _verify_record(selector_fit, "selector fit receipt")
    selector_identity = FrozenComponentIdentity(
        component_id="isingfold-if-q3-s0-strength-selector",
        version=str(bundle.model.version),
        implementation=(
            f"{type(bundle.model).__module__}.{type(bundle.model).__qualname__}:"
            f"fit-{selector_fit['record_digest']}"
        ),
        artifact_sha256=_sha256_file(bundle.root / "selector.pt"),
    )
    population_seed = int(protocol["evaluation_seed"])
    population = CompletePopulationIdentity(
        population_id=(
            "external-tuning-validation-"
            + stable_digest(
                {
                    "manifest": manifest_sha,
                    "partition": "val",
                    "evaluation_seed": population_seed,
                    "repetitions": registry.repetitions,
                }
            )
        ),
        source_manifest_sha256=manifest_sha,
        task_payload_sha256=task_population_digest(tasks),
        expected_instances=tuple(sorted((task.lineage or task.name, task.name) for task in tasks)),
        expected_repetitions=registry.repetitions,
        evaluation_strata=evaluation_strata,
        confirmatory_design=confirmatory_design,
        evaluation_seed=population_seed,
    )
    device = _resolve_device(args.device)
    _seed_runtime(tuning_seed, deterministic=args.deterministic, threads=args.threads)
    bundle.model.to(device).eval()
    runtime_platform = _runtime_platform_identity()
    compute_identity = runtime_identity(
        runtime_platform=runtime_platform,
        inference_device_type=device.type,
        inference_device_name=_inference_device_name(device, runtime_platform),
        inference_threads=args.threads,
        deterministic=args.deterministic,
    )
    tuning_execution = ExternalTuningExecutionBinding.for_validation(registry, args.index)
    backend = StockMinorminerBackend(availability.identity)
    outcomes, receipts = run_external_complete_system(
        tasks,
        context,
        backend,
        external_config,
        learned_config=learned_config,
        selector=bundle.model,
        selector_identity=selector_identity,
        population=population,
        quality_authority=quality_authority,
        seed=population_seed,
        repetitions=registry.repetitions,
        training_seed_index=args.tuning_seed_index,
        training_seed=tuning_seed,
        compute_identity=compute_identity,
        tuning_execution=tuning_execution,
    )
    expected_attempts = len(tasks) * registry.repetitions
    if len(outcomes) != expected_attempts or len(receipts) != expected_attempts:
        raise RuntimeError("external tuning runner changed its all-attempt denominator")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tuning-", dir=destination.parent)
    )
    try:
        outcome_sha = write_external_complete_outcomes(temporary / "outcomes.jsonl", outcomes)
        receipt_sha = write_external_complete_receipts(
            temporary / "external_tuning_receipts.jsonl", receipts
        )
        evidence_sha = write_external_complete_evidence(
            temporary / "terminal_evidence.jsonl", receipts
        )
        recovered_outcomes = read_external_complete_outcomes(
            temporary / "outcomes.jsonl", expected_sha256=outcome_sha
        )
        recovered_receipts = read_external_complete_receipts(
            temporary / "external_tuning_receipts.jsonl",
            outcomes=recovered_outcomes,
            expected_backend=availability.identity,
            expected_selector=selector_identity,
            expected_population=population,
            expected_config_digest=external_config.digest,
            expected_learned_config_digest=learned_config.digest,
            expected_context_digest=stable_digest(context_snapshot(context)),
            expected_quality_authority_digest=content_digest(quality_authority),
            expected_work_cap=context.caps,
            expected_training_seed_index=args.tuning_seed_index,
            expected_training_seed=tuning_seed,
            expected_runtime_identity_digest=content_digest(compute_identity),
            expected_tuning_execution=tuning_execution,
            expected_sha256=receipt_sha,
        )
        recovered_evidence = read_external_complete_evidence(
            temporary / "terminal_evidence.jsonl",
            receipts=recovered_receipts,
            tasks=tasks,
            context=context,
            expected_sha256=evidence_sha,
        )
        if len(recovered_evidence) != expected_attempts:
            raise RuntimeError("external tuning evidence changed its attempt census")
        runtime_registry = runtime_implementation_registry()
        report = build_external_tuning_report(
            registry=registry,
            candidate_index=args.index,
            tuning_seed_index=args.tuning_seed_index,
            receipts=recovered_receipts,
            runtime_identity=compute_identity,
            runtime_implementation_registry=runtime_registry,
            quality_authority=quality_authority,
            target_access=target_access,
            ground_partition_receipt=ground_partition_receipt,
            context=context_snapshot(context),
            artifacts={
                "receipts": {
                    "path": "external_tuning_receipts.jsonl",
                    "sha256": receipt_sha,
                    "count": len(receipts),
                },
                "terminal_evidence": {
                    "path": "terminal_evidence.jsonl",
                    "sha256": evidence_sha,
                    "count": len(receipts),
                },
                "outcomes": {
                    "path": "outcomes.jsonl",
                    "sha256": outcome_sha,
                    "count": len(outcomes),
                },
            },
        )
        _atomic_json(temporary / "report.json", report)
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(
        json.dumps(
            {
                "candidate_index": args.index,
                "candidate_id": candidate.candidate_id,
                "tuning_seed_index": args.tuning_seed_index,
                "tuning_seed": tuning_seed,
                "validation_population": len(tasks),
                "attempts": expected_attempts,
                "out": str(destination),
            },
            indent=1,
        )
    )


def cmd_select_external_tuning(args: argparse.Namespace) -> None:
    """Freeze one stock-minorminer strategy from the complete validation census."""

    from isingfold.rl.external_tuning import (
        load_external_tuning_registry,
        load_external_tuning_run,
        write_external_tuning_selection,
    )

    registry = load_external_tuning_registry(
        args.registry,
        expected_file_sha256=args.expected_registry_sha256,
        grid_path=args.grid,
        external_config_path=args.external_config,
    )
    evaluations_root = Path(args.evaluations_root)
    runs = []
    for candidate_index, candidate in enumerate(registry.candidates):
        for seed_index, _ in enumerate(registry.tuning_seeds):
            relative = Path(candidate.candidate_id) / f"seed-{seed_index}" / "report.json"
            report_path = evaluations_root / relative
            report_sha256 = _sha256_file(report_path)
            run = load_external_tuning_run(
                report_path,
                expected_file_sha256=report_sha256,
                report_path=relative.as_posix(),
            )
            if run.candidate_index != candidate_index or run.tuning_seed_index != seed_index:
                raise ValueError("external tuning report path and registered cell differ")
            for artifact_name in ("receipts", "terminal_evidence", "outcomes"):
                artifact = run.report["artifacts"][artifact_name]
                artifact_path = report_path.parent / str(artifact["path"])
                if _sha256_file(artifact_path) != artifact["sha256"]:
                    raise ValueError(
                        f"external tuning {artifact_name} artifact differs from report pin"
                    )
                with artifact_path.open("rb") as handle:
                    observed_count = sum(1 for _ in handle)
                if observed_count != artifact["count"]:
                    raise ValueError(f"external tuning {artifact_name} artifact count differs")
            runs.append(run)
    receipt, receipt_sha256 = write_external_tuning_selection(
        args.out,
        registry=registry,
        runs=runs,
    )
    print(
        json.dumps(
            {
                "external_tuning_selection": receipt,
                "selection_file_sha256": receipt_sha256,
                "test_use_requires_this_out_of_band_sha256": True,
            },
            indent=1,
        )
    )


def cmd_gates(args: argparse.Namespace) -> None:
    """Run implementation gates on authenticated train-side or validation-side tasks."""

    from isingfold.rl.gates import (
        gate_authenticated_k2_support_coverage,
        gate_conformance,
        gate_profile_c_readiness,
        gate_profile_i_signal,
        gate_selector_discrimination,
        gate_support_headroom,
    )
    from isingfold.rl.data.prepared import load_prepared_partition
    from isingfold.rl.data.selector_labels import load_selector_metadata

    quality_pin = _quality_attestation_pin(args)
    ctx = _context(args, args.corpus)
    bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
    resolution_plan = _load_pinned_quality_resolution_plan(args.plan, args.expected_plan_sha256)
    public_train = load_prepared_partition(
        args.corpus,
        partition="train",
        include_evaluator=False,
    )
    if public_train.target_access is not None:
        raise RuntimeError("Gate 2 target-free train load opened evaluator targets")
    initializer_bank, initializer_bank_manifest_sha256 = _load_quality_initializer_bank(
        args,
        public_tasks=public_train.tasks,
        context=ctx,
    )
    full_support = gate_authenticated_k2_support_coverage(
        resolution_plan,
        expected_plan_sha256=args.expected_plan_sha256,
        public_prepared=public_train.tasks,
        context=ctx,
        selector=bundle.model,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=(initializer_bank_manifest_sha256),
    )
    grid, grid_digest = _load_grid(args.grid)
    resolution_protocol = grid["quality_resolution"]
    quality_preflight = _load_quality_preflight_receipt(
        args.quality_preflight_receipt,
        expected_sha256=args.expected_quality_preflight_sha256,
        quality_labels=args.quality_labels,
        corpus=args.corpus,
        selector=bundle,
        context=ctx,
        min_resolved_rows=int(resolution_protocol["min_resolved_rows"]),
        min_resolved_lineages=int(resolution_protocol["min_resolved_lineages"]),
    )
    exact_prepared, exact_corpus_identity = _load_exact_conformance_tasks(
        args.exact_conformance_corpus,
        expected_sha256=args.expected_exact_conformance_sha256,
        corpus=args.corpus,
        expected_corpus_manifest_sha256=bundle.corpus_manifest_sha256,
    )
    (
        prepared,
        quality_authority,
        target_access,
        ground_partition_receipt,
    ) = _load_quality_partition(
        args.corpus,
        partition=args.partition,
        pin=quality_pin,
    )
    if _bind_selector_quality_authority(bundle, prepared, pin=quality_pin) != quality_authority:
        raise ValueError("selector and release-gate authorities differ")
    scalable_prepared, scalable_sampling = _stratified_gate_sample(
        prepared,
        count=args.instances,
        seed=args.seed,
    )
    selected_tasks = [item.task for item in scalable_prepared]
    exact_tasks = [item.task for item in exact_prepared]
    conformance = gate_conformance(exact_tasks, ctx)
    conformance["authenticated_exact_corpus"] = exact_corpus_identity
    validation_headroom = gate_support_headroom(
        selected_tasks,
        ctx,
        selector=bundle.model,
        reads=args.reward_reads,
        seed=args.seed,
        stratum_ids=tuple(scalable_sampling["selected_stratum_digests"]),
    )
    gate_results = {
        "gate_1_exact_conformance": conformance,
        "gate_2_authenticated_k2_full_support": full_support,
        "gate_3_selector_discrimination": gate_selector_discrimination(
            args.selector_labels,
            bundle.model,
            expected_source_manifest_sha256=bundle.corpus_manifest_sha256,
            expected_context_digest=content_digest(_context_snapshot(ctx)),
        ),
        "gate_4_profile_i_signal": gate_profile_i_signal(
            selected_tasks,
            ctx,
            selector=bundle.model,
            reads=args.reward_reads,
            seed=args.seed,
        ),
    }
    profile_c = gate_profile_c_readiness(
        selected_tasks,
        ctx,
        selector=bundle.model,
        reads=args.reward_reads,
        seed=args.seed,
    )
    advance = bool(validation_headroom.get("pass")) and all(
        bool(value.get("pass")) for value in gate_results.values() if isinstance(value, dict)
    )
    selector_label_metadata = load_selector_metadata(args.selector_labels)
    label_authority = _validate_quality_authority_binding(
        dict(selector_label_metadata.quality_authority),
        expected_role="training_partition",
    )
    if label_authority["global"] != quality_authority["global"]:
        raise ValueError("gate labels belong to a different pinned quality authority")
    report = _with_digest(
        {
            "schema": GATE_RECEIPT_SCHEMA,
            "schema_version": GATE_RECEIPT_SCHEMA_VERSION,
            "source_corpus_manifest_sha256": bundle.corpus_manifest_sha256,
            "quality_authority": quality_authority,
            "target_access": target_access,
            "ground_partition_receipt": ground_partition_receipt,
            "source_selector_digest": bundle.selector_digest,
            "source_selector_fit_receipt_sha256": _sha256_file(bundle.root / "fit_receipt.json"),
            "source_selector_labels_manifest_sha256": (selector_label_metadata.manifest_sha256),
            "source_selector_labels_manifest_record_digest": (
                selector_label_metadata.manifest_record_digest
            ),
            "grid_manifest_sha256": grid_digest,
            "quality_preflight_receipt_sha256": _sha256_file(args.quality_preflight_receipt),
            "quality_preflight_record_digest": quality_preflight["record_digest"],
            "context": _context_snapshot(ctx),
            "context_digest": content_digest(_context_snapshot(ctx)),
            "partition": args.partition,
            "seed": args.seed,
            "parameters": {
                "instances_requested": args.instances,
                "reward_reads": args.reward_reads,
                "profile_episodes_per_instance": 2,
                "broad_reference_batches_per_instance": 8,
                "exact_conformance_corpus": exact_corpus_identity,
                "quality_resolution_plan": {
                    "initializer_bank_contract_record_digest": full_support["initializer_bank"][
                        "record_digest"
                    ],
                    "initializer_bank_manifest_sha256": full_support["initializer_bank"][
                        "manifest_sha256"
                    ],
                    "raw_sha256": args.expected_plan_sha256,
                    "record_digest": resolution_plan.as_dict()["record_digest"],
                },
                "scalable_gate_sampling": scalable_sampling,
            },
            "gate_profile": "profile-i",
            "mandatory_gate_names": list(gate_results),
            "gates": gate_results,
            "optional_diagnostics": {
                "profile_c_construction_readiness": profile_c,
                "validation_support_headroom": validation_headroom,
            },
            "construction_ready": bool(profile_c.get("pass")),
            "advance": advance,
        }
    )
    _publish_gate_receipt(args.out, report)
    print(json.dumps(report, indent=1, default=float))


def _read_representation_complete_receipts(
    path: Path, *, expected_sha256: str
) -> tuple[object, ...]:
    """Narrow testable boundary around the strict complete-system receipt parser."""

    from isingfold.rl.complete_system import read_complete_system_receipts

    return tuple(read_complete_system_receipts(path, expected_sha256=expected_sha256))


def _evaluation_policy_receipt(
    report_path: Path,
    *,
    expected_cell: Mapping[str, object],
    grid_digest: str,
    expected_protocol: Mapping[str, object],
) -> tuple[dict[str, Any], list[Any], list[Any], dict[str, object]]:
    """Authenticate one K=2 representation report and every same-support arm."""

    from isingfold.rl import evaluate as evaluation_module
    from isingfold.rl.evaluate import (
        EVALUATION_RECEIPT_VERSION,
        baseline_method_metadata,
        read_episode_receipts,
        secondary_metrics,
    )

    if not report_path.is_file():
        raise FileNotFoundError(
            f"missing validation evaluation for {expected_cell['cell_id']}: {report_path}"
        )
    report = _strict_json(report_path)
    _verify_record(report, f"representation evaluation {expected_cell['cell_id']}")
    if (
        report.get("schema") != "isingfold.paired-evaluation"
        or report.get("schema_version") != EVALUATION_REPORT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported representation evaluation report")
    if report.get("partition") != "validation":
        raise ValueError("representation selection accepts validation evaluations only")
    expected_identity = {
        "partition": expected_protocol["partition"],
        "model_family": expected_cell["model_family"],
        "training_seed": expected_cell["seed"],
        "training_method": "supervised-ranking",
        "grid_cell": expected_cell["cell_id"],
        "grid_manifest_sha256": grid_digest,
        "evaluation_seed": expected_protocol["evaluation_seed"],
        "deployment_rule": expected_protocol["deployment_rule"],
        "repetitions": expected_protocol["repetitions"],
    }
    observed_identity = {name: report.get(name) for name in expected_identity}
    if observed_identity != expected_identity:
        raise ValueError(
            f"evaluation identity differs for {expected_cell['cell_id']}: {observed_identity}"
        )
    gate_profile = report.get("gate_profile")
    gate_receipt_sha256 = report.get("gate_receipt_sha256")
    gate_record_digest = report.get("gate_record_digest")
    if (
        gate_profile not in {"profile-i", "profile-i+profile-c"}
        or not isinstance(gate_receipt_sha256, str)
        or len(gate_receipt_sha256) != 64
        or not isinstance(gate_record_digest, str)
        or len(gate_record_digest) != 64
    ):
        raise ValueError("representation evaluation has no authenticated Profile-I gate")
    population = report.get("population")
    repetitions = report.get("repetitions")
    if (
        isinstance(population, bool)
        or not isinstance(population, int)
        or population <= 0
        or isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions <= 0
    ):
        raise ValueError("representation evaluation has invalid population denominators")
    arm_receipts = report.get("arm_receipts")
    required_arms = (
        "return_initial",
        "random_masked",
        "classical_resource_first",
        "classical_quality_aware",
        "policy",
    )
    if not isinstance(arm_receipts, dict) or any(
        not isinstance(arm_receipts.get(name), dict) for name in required_arms
    ):
        raise ValueError("representation evaluation needs every same-support raw arm")
    for name in required_arms:
        if set(arm_receipts[name]) != {"path", "sha256", "count", "method", "protocol"}:
            raise ValueError(f"{name} receipt has an invalid schema")
    methods = report.get("methods")
    matched_metadata = report.get("matched_metadata")
    if (
        not isinstance(methods, dict)
        or set(methods) != set(required_arms)
        or any(not isinstance(methods.get(name), dict) for name in required_arms)
        or not isinstance(matched_metadata, dict)
        or any(
            arm_receipts[name]["method"] != methods[name]
            or arm_receipts[name]["protocol"] != matched_metadata
            for name in required_arms
        )
    ):
        raise ValueError("evaluation-arm method/protocol identity differs from its report")
    required_protocol = {
        "population",
        "quality_authority",
        "target_access",
        "ground_partition_receipt",
        "budget",
        "selector",
        "proposal",
        "support",
        "policy_seed",
        "evaluator_reads",
        "initializer",
        "repetitions",
        "inference_device_type",
        "inference_device_name",
        "inference_threads",
        "runtime_platform",
        "baseline_controller_registry",
        "evaluation_receipt_schema",
        "evaluation_implementation_sha256",
        "quality_preflight_receipt_sha256",
        "quality_preflight_record_digest",
        "runtime_implementation_registry",
        "runtime_implementation_digest",
        "bootstrap_bank_access",
        "same_support_contract_digest",
    }
    if set(matched_metadata) != required_protocol:
        raise ValueError("representation evaluation has incomplete matched metadata")
    corpus_digest = report.get("corpus_manifest_sha256")
    selector_digest = report.get("selector_digest")
    quality_authority = report.get("quality_authority")
    target_access = report.get("target_access")
    ground_partition_receipt = report.get("ground_partition_receipt")
    if (
        not isinstance(corpus_digest, str)
        or len(corpus_digest) != 64
        or not isinstance(selector_digest, str)
        or len(selector_digest) != 64
        or not isinstance(quality_authority, dict)
        or matched_metadata["quality_authority"] != quality_authority
        or matched_metadata["target_access"] != target_access
        or matched_metadata["ground_partition_receipt"] != ground_partition_receipt
        or matched_metadata["population"] != f"{corpus_digest}:{expected_protocol['partition']}"
        or matched_metadata["selector"] != selector_digest
        or matched_metadata["policy_seed"] != expected_protocol["evaluation_seed"]
        or matched_metadata["evaluator_reads"] != expected_protocol["audit_reads"]
        or matched_metadata["repetitions"] != expected_protocol["repetitions"]
        or matched_metadata["support"] != "same-conditional-generator-and-exact-mask-contract"
        or matched_metadata["initializer"]
        != "authenticated-persistent-upfront-lac-cache-k2-v2"
    ):
        raise ValueError(
            "representation evaluation changes the registered population/search/read protocol"
        )
    bootstrap_access = matched_metadata["bootstrap_bank_access"]
    support_digest = matched_metadata["same_support_contract_digest"]
    if (
        not isinstance(bootstrap_access, dict)
        or bootstrap_access.get("schema") != "isingfold.rl-value-bootstrap-access"
        or bootstrap_access.get("schema_version") != 1
        or bootstrap_access.get("protocol_preset") != "representation-validation"
        or bootstrap_access.get("opened_evaluator_targets") is not False
        or bootstrap_access.get("denominator_count") != population * repetitions
        or bootstrap_access.get("prepared_manifest_sha256") != corpus_digest
        or bootstrap_access.get("protocol_registry_sha256") != grid_digest
        or bootstrap_access.get("protocol_record_digest")
        != content_digest(expected_protocol)
        or bootstrap_access.get("same_support_contract_digest") != support_digest
        or report.get("bootstrap_bank_access") != bootstrap_access
        or report.get("same_support_contract_digest") != support_digest
    ):
        raise ValueError(
            "representation evaluation lacks its authenticated representation-validation "
            "bootstrap bank"
        )
    _verify_record(bootstrap_access, "representation bootstrap-bank access")
    _require_lower_sha256(support_digest, "representation same-support contract")
    if not isinstance(target_access, dict) or not isinstance(ground_partition_receipt, dict):
        raise ValueError("representation evaluation omits target-access receipts")
    _verify_record(target_access, "representation evaluation target access")
    _verify_record(
        ground_partition_receipt,
        "representation evaluation ground partition",
    )
    evaluation_partition = quality_authority.get("evaluation_partition")
    ground_identity = (
        evaluation_partition.get("ground_partition")
        if isinstance(evaluation_partition, dict)
        else None
    )
    if (
        not isinstance(evaluation_partition, dict)
        or target_access.get("record_digest")
        != evaluation_partition.get("target_access_record_digest")
        or not isinstance(ground_identity, dict)
        or ground_partition_receipt.get("record_digest")
        != ground_identity.get("receipt_record_digest")
    ):
        raise ValueError("representation target access differs from quality authority")
    if matched_metadata["evaluation_receipt_schema"] != EVALUATION_RECEIPT_VERSION:
        raise ValueError("representation evaluation uses an unsupported receipt schema")
    if (
        not isinstance(matched_metadata["runtime_implementation_registry"], dict)
        or matched_metadata["runtime_implementation_digest"]
        != stable_digest(matched_metadata["runtime_implementation_registry"])
        or report.get("runtime_implementation_registry")
        != matched_metadata["runtime_implementation_registry"]
        or report.get("runtime_implementation_digest")
        != matched_metadata["runtime_implementation_digest"]
    ):
        raise ValueError("representation evaluation has an inconsistent runtime registry")
    if matched_metadata["evaluation_implementation_sha256"] != _sha256_file(
        evaluation_module.__file__
    ):
        raise ValueError("representation evaluation implementation digest differs")
    for digest_field in (
        "budget",
        "selector",
        "baseline_controller_registry",
        "quality_preflight_receipt_sha256",
        "quality_preflight_record_digest",
    ):
        _require_lower_sha256(
            matched_metadata[digest_field], f"representation protocol {digest_field}"
        )
    runtime = matched_metadata.get("runtime_platform")
    if not isinstance(runtime, dict) or set(runtime) != {
        "hostname",
        "system",
        "release",
        "machine",
        "processor",
        "logical_cpu_count",
        "slurm_partition",
    }:
        raise ValueError("representation evaluation has malformed runtime-platform identity")
    logical_cpu_count = runtime["logical_cpu_count"]
    slurm_partition = runtime["slurm_partition"]
    if (
        isinstance(logical_cpu_count, bool)
        or not isinstance(logical_cpu_count, int)
        or logical_cpu_count <= 0
        or (slurm_partition is not None and not isinstance(slurm_partition, str))
    ):
        raise ValueError("representation evaluation has invalid runtime-platform values")
    expected_count = population * repetitions
    loaded_arms: dict[str, list[Any]] = {}
    for name in required_arms:
        arm_receipt = arm_receipts[name]
        relative = arm_receipt["path"]
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or len(Path(relative).parts) != 1
        ):
            raise ValueError(f"{name} receipt path must be one local filename")
        if arm_receipt["count"] != expected_count:
            raise ValueError(f"{name} receipt count differs from population times repetitions")
        loaded = read_episode_receipts(
            report_path.parent / relative,
            expected_sha256=arm_receipt["sha256"],
        )
        if len(loaded) != expected_count:
            raise ValueError(f"raw {name} outcome count differs from its receipt")
        identities = {(row.lineage, row.instance) for row in loaded}
        if len(identities) != population:
            raise ValueError(f"raw {name} population denominator differs from its report")
        for identity in identities:
            observed_repetitions = sorted(
                row.repetition for row in loaded if (row.lineage, row.instance) == identity
            )
            if observed_repetitions != list(range(repetitions)):
                raise ValueError(f"raw {name} has an incomplete repetition census")
        if report.get(name) != secondary_metrics(loaded):
            raise ValueError(f"{name} summary differs from authenticated raw outcomes")
        loaded_arms[name] = loaded

    expected_baselines = baseline_method_metadata()
    if any(methods[name] != expected_baselines[name] for name in expected_baselines):
        raise ValueError("representation evaluation changes a registered baseline method")
    policy_method = methods["policy"]
    if (
        policy_method.get("method_id")
        != f"{expected_cell['model_family']}:supervised-ranking"
        or policy_method.get("checkpoint_payload_digest")
        != report.get("checkpoint_payload_digest")
        or policy_method.get("selection_rule") != expected_protocol["deployment_rule"]
        or policy_method.get("support_contract")
        != "environment-materialised-masked-actions"
        or policy_method.get("online_evaluator_feedback") is not False
        or policy_method.get("evaluator_oracle") is not False
    ):
        raise ValueError("representation policy method differs from its registered cell")

    complete_registry = report.get("complete_system_receipts")
    if not isinstance(complete_registry, dict) or set(complete_registry) != set(required_arms):
        raise ValueError(
            "representation evaluation lacks complete receipts for every same-support arm"
        )
    complete_by_arm: dict[str, dict[tuple[str, str, int], object]] = {}
    for arm in required_arms:
        metadata = complete_registry[arm]
        expected_path = f"{arm}.complete.jsonl"
        if (
            not isinstance(metadata, dict)
            or set(metadata) != {"path", "sha256", "count"}
            or metadata.get("path") != expected_path
            or metadata.get("count") != expected_count
        ):
            raise ValueError(f"representation evaluation has invalid {arm} complete metadata")
        complete = _read_representation_complete_receipts(
            report_path.parent / expected_path,
            expected_sha256=_require_lower_sha256(
                metadata.get("sha256"), f"{arm} complete receipt digest"
            ),
        )
        complete_pairs = {
            getattr(receipt, "outcome").pair_key: receipt for receipt in complete
        }
        ordinary_pairs = {row.pair_key: row for row in loaded_arms[arm]}
        if (
            len(complete) != expected_count
            or len(complete_pairs) != expected_count
            or set(complete_pairs) != set(ordinary_pairs)
        ):
            raise ValueError(f"representation evaluation changes the {arm} complete census")
        for pair_key, ordinary in ordinary_pairs.items():
            receipt = complete_pairs[pair_key]
            complete_outcome = getattr(receipt, "outcome", None)
            binding = getattr(receipt, "bootstrap_binding", None)
            clone = binding.get("clone") if isinstance(binding, Mapping) else None
            if (
                not hasattr(complete_outcome, "as_dict")
                or complete_outcome.as_dict() != ordinary.as_dict()
            ):
                raise ValueError(
                    "representation evaluation changes an outcome across receipt layers"
                )
            if (
                not isinstance(clone, Mapping)
                or clone.get("consumer_id")
                != f"representation/{expected_cell['cell_id']}/{arm}"
                or clone.get("bank_access_record_digest")
                != bootstrap_access["record_digest"]
                or clone.get("same_support_contract_digest") != support_digest
            ):
                raise ValueError(
                    f"representation evaluation has an invalid {arm} bootstrap clone"
                )
        complete_by_arm[arm] = complete_pairs

    bootstrap_identity_fields = (
        "bank_access_record_digest",
        "same_support_contract_digest",
        "bootstrap_record_digest",
        "bootstrap_outcome_record_digest",
        "bootstrap_payload_sha256",
        "row_key",
    )
    policy_pairs = set(complete_by_arm["policy"])
    if any(set(complete_by_arm[arm]) != policy_pairs for arm in required_arms):
        raise ValueError("representation evaluation changes the paired census across arms")
    for pair_key in sorted(policy_pairs):
        identities = []
        for arm in required_arms:
            binding = getattr(complete_by_arm[arm][pair_key], "bootstrap_binding")
            clone = binding["clone"]
            identity = tuple(clone.get(field) for field in bootstrap_identity_fields)
            if any(not _is_lower_sha256(value) for value in identity):
                raise ValueError("representation bootstrap clone has an invalid identity digest")
            identities.append(identity)
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError(
                "representation evaluation changes bootstrap support across arms "
                f"at {pair_key!r}"
            )

    policy_by_pair = {outcome.pair_key: outcome for outcome in loaded_arms["policy"]}
    initial_by_pair = {outcome.pair_key: outcome for outcome in loaded_arms["return_initial"]}
    if len(policy_by_pair) != len(loaded_arms["policy"]) or len(initial_by_pair) != len(
        loaded_arms["return_initial"]
    ):
        raise ValueError("policy or return-initial receipts repeat an exact task/repetition pair")
    if set(policy_by_pair) != set(initial_by_pair):
        raise ValueError("policy and return-initial receipts do not share exact pairs")
    for key in sorted(policy_by_pair):
        policy_row = policy_by_pair[key]
        initial_row = initial_by_pair[key]
        if (
            policy_row.population_eligible != initial_row.population_eligible
            or policy_row.episode_seed != initial_row.episode_seed
        ):
            raise ValueError(f"policy/reference population or search seed differs at {key!r}")
        if (
            policy_row.returned_valid
            and initial_row.returned_valid
            and policy_row.evaluator_seed != initial_row.evaluator_seed
        ):
            raise ValueError(f"policy/reference evaluator seed differs at {key!r}")

    outcomes = loaded_arms["policy"]
    recomputed = secondary_metrics(outcomes)
    eligible = [outcome for outcome in outcomes if outcome.population_eligible]
    if not eligible:
        raise ValueError("representation evaluation has no eligible policy episodes")
    online = [outcome.online_seconds for outcome in eligible]
    if any(value is None or not math.isfinite(value) or value < 0.0 for value in online):
        raise ValueError("representation selection requires complete online timing")
    metrics = {
        "valid_return_rate": float(recomputed["valid_return_rate"]),
        "utility_mean": float(recomputed["utility_mean"]),
        "online_seconds_mean": float(np.mean(online)),
        "eligible_episodes": len(eligible),
        "attempts": len(outcomes),
    }
    return report, outcomes, loaded_arms["return_initial"], metrics


def _equal_seed_lineage_aggregate(
    runs: Sequence[tuple[int, Sequence[Any], Sequence[Any]]],
    *,
    margin: float,
    alpha: float,
    bootstrap: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    """Aggregate registered training seeds, clustering uncertainty by base lineage."""

    if not runs:
        raise ValueError("representation family has no authenticated runs")
    per_seed_lineage: list[dict[str, tuple[float, float, float, float]]] = []
    for _training_seed, outcomes, references in sorted(runs, key=lambda row: row[0]):
        policy_by_pair = {row.pair_key: row for row in outcomes}
        reference_by_pair = {row.pair_key: row for row in references}
        if len(policy_by_pair) != len(outcomes) or set(policy_by_pair) != set(reference_by_pair):
            raise ValueError("representation policy/reference census differs within a seed")
        grouped: dict[str, list[tuple[Any, Any]]] = {}
        for pair_key, outcome in policy_by_pair.items():
            reference = reference_by_pair[pair_key]
            if outcome.population_eligible:
                grouped.setdefault(outcome.lineage or outcome.instance, []).append(
                    (outcome, reference)
                )
        if not grouped:
            raise ValueError("representation run has no eligible independent lineage")
        per_seed_lineage.append(
            {
                lineage: (
                    float(np.mean([float(row.returned_valid) for row, _ in rows])),
                    float(np.mean([float(reference.returned_valid) for _, reference in rows])),
                    float(np.mean([float(row.utility) for row, _ in rows])),
                    float(np.mean([float(row.online_seconds) for row, _ in rows])),
                )
                for lineage, rows in grouped.items()
            }
        )
    lineages = sorted(per_seed_lineage[0])
    if any(sorted(rows) != lineages for rows in per_seed_lineage[1:]):
        raise ValueError("representation training seeds do not share base lineages")
    seed_lineage_values = np.asarray(
        [[rows[lineage] for lineage in lineages] for rows in per_seed_lineage],
        dtype=float,
    )
    lineage_values = np.mean(seed_lineage_values, axis=0)
    feasibility_delta = lineage_values[:, 0] - lineage_values[:, 1]
    from isingfold.rl.experiment_selection import crossed_bootstrap_bounds

    feasibility_ci_low, feasibility_ci_high = crossed_bootstrap_bounds(
        seed_lineage_values[:, :, 0] - seed_lineage_values[:, :, 1],
        replicates=bootstrap,
        seed=bootstrap_seed,
        alpha=alpha,
    )
    return {
        "independent_lineages": len(lineages),
        "valid_return_rate": float(np.mean(lineage_values[:, 0])),
        "reference_valid_return_rate": float(np.mean(lineage_values[:, 1])),
        "utility_mean": float(np.mean(lineage_values[:, 2])),
        "online_seconds_mean": float(np.mean(lineage_values[:, 3])),
        "feasibility_delta_vs_return_initial": float(np.mean(feasibility_delta)),
        "feasibility_ci_low": feasibility_ci_low,
        "feasibility_ci_high": feasibility_ci_high,
        "feasibility_noninferior": bool(feasibility_ci_low > -margin),
    }


def cmd_select_representation(args: argparse.Namespace) -> None:
    """Freeze the three-seed validation decision used by every RL-value cell."""

    from isingfold.rl.experiment_selection import _split_protocol_identity

    grid, grid_digest = _load_grid(args.grid)
    cells = grid["stages"]["representation"]
    expected_families = set(REGISTERED_MODEL_FAMILIES)
    by_family: dict[str, list[dict[str, object]]] = {
        family: [] for family in REGISTERED_MODEL_FAMILIES
    }
    family_runs: dict[str, list[tuple[int, Sequence[Any], Sequence[Any]]]] = {
        family: [] for family in REGISTERED_MODEL_FAMILIES
    }
    source_reports: list[dict[str, object]] = []
    evaluation_protocol = grid["representation_evaluation"]
    protocol_identity: dict[str, object] | None = None
    runtime_identity_by_seed: dict[str, dict[str, Any]] = {}
    pair_identity: tuple[tuple[object, ...], ...] | None = None
    reference_identity: tuple[tuple[object, ...], ...] | None = None
    for cell in cells:
        family = cell["model_family"]
        if family not in expected_families:
            raise ValueError("representation grid contains an unregistered model family")
        report_path = Path(args.evaluations_root) / cell["cell_id"] / "report.json"
        report, outcomes, reference, metrics = _evaluation_policy_receipt(
            report_path,
            expected_cell=cell,
            grid_digest=grid_digest,
            expected_protocol=evaluation_protocol,
        )
        scientific_metadata, runtime_identity = _split_protocol_identity(report["matched_metadata"])
        current_protocol = {
            "population": report["population"],
            "repetitions": report["repetitions"],
            "selector_digest": report.get("selector_digest"),
            "corpus_manifest_sha256": report.get("corpus_manifest_sha256"),
            "deployment_rule": report.get("deployment_rule"),
            "gate_profile": report.get("gate_profile"),
            "gate_receipt_sha256": report.get("gate_receipt_sha256"),
            "gate_record_digest": report.get("gate_record_digest"),
            "matched_metadata": scientific_metadata,
        }
        if protocol_identity is None:
            protocol_identity = current_protocol
        elif current_protocol != protocol_identity:
            raise ValueError(
                "representation evaluations do not share one scientific protocol identity"
            )
        seed_key = str(cell["seed"])
        registered_runtime = runtime_identity_by_seed.get(seed_key)
        if registered_runtime is None:
            runtime_identity_by_seed[seed_key] = runtime_identity
        elif runtime_identity != registered_runtime:
            raise ValueError(
                "representation evaluations change runtime identity within training seed "
                f"{cell['seed']}"
            )
        current_pairs = tuple(
            sorted(
                (
                    *outcome.pair_key,
                    outcome.population_eligible,
                    outcome.episode_seed,
                )
                for outcome in outcomes
            )
        )
        if pair_identity is None:
            pair_identity = current_pairs
        elif current_pairs != pair_identity:
            raise ValueError("representation evaluations do not share exact task/seed pairs")
        from isingfold.rl.experiment_selection import reference_outcome_signature

        current_reference = reference_outcome_signature(reference)
        if reference_identity is None:
            reference_identity = current_reference
        elif current_reference != reference_identity:
            raise ValueError("representation evaluations do not share one return-initial reference")
        seed_row = {
            "cell_id": cell["cell_id"],
            "seed": cell["seed"],
            **metrics,
        }
        by_family[family].append(seed_row)
        family_runs[family].append((int(cell["seed"]), outcomes, reference))
        source_reports.append(
            {
                "cell_id": cell["cell_id"],
                "model_family": family,
                "seed": cell["seed"],
                "report_path": str(report_path),
                "report_sha256": _sha256_file(report_path),
                "report_record_digest": report["record_digest"],
                "policy_receipt_sha256": report["arm_receipts"]["policy"]["sha256"],
                "checkpoint_payload_digest": report["checkpoint_payload_digest"],
                "runtime_implementation_registry": report["runtime_implementation_registry"],
                "runtime_implementation_digest": report["runtime_implementation_digest"],
                "runtime_identity": runtime_identity,
            }
        )

    margin = float(evaluation_protocol["margin"])
    alpha = float(evaluation_protocol["feasibility_alpha"])
    bootstrap = int(evaluation_protocol["feasibility_bootstrap"])
    bootstrap_seed = int(evaluation_protocol["selection_bootstrap_seed"])
    aggregates: dict[str, dict[str, object]] = {}
    for family, rows in by_family.items():
        expected_seeds = sorted(
            int(cell["seed"]) for cell in cells if cell["model_family"] == family
        )
        observed_seeds = sorted(int(row["seed"]) for row in rows)
        if len(rows) != 3 or observed_seeds != expected_seeds or len(set(observed_seeds)) != 3:
            raise ValueError(f"model family {family} lacks its three registered seeds")
        aggregate = _equal_seed_lineage_aggregate(
            family_runs[family],
            margin=margin,
            alpha=alpha,
            bootstrap=bootstrap,
            bootstrap_seed=bootstrap_seed,
        )
        aggregates[family] = {
            "seed_count": len(rows),
            "seeds": observed_seeds,
            **aggregate,
            "per_seed": sorted(rows, key=lambda row: int(row["seed"])),
        }

    simpler = ("if-mlp", "if-dual")
    eligible_simpler = [
        family for family in simpler if aggregates[family]["feasibility_noninferior"] is True
    ]
    if not eligible_simpler:
        raise ValueError(
            "no simpler representation passes the preregistered feasibility constraint"
        )
    if aggregates["if-core"]["feasibility_noninferior"] is not True:
        raise ValueError("IF-Core fails the preregistered feasibility constraint")
    selected = max(
        eligible_simpler,
        key=lambda family: (
            float(aggregates[family]["utility_mean"]),
            -float(aggregates[family]["online_seconds_mean"]),
            family == "if-mlp",
        ),
    )
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"representation-selection receipt already exists: {destination}")
    receipt = _with_digest(
        {
            "schema": REPRESENTATION_SELECTION_SCHEMA,
            "schema_version": REPRESENTATION_SELECTION_SCHEMA_VERSION,
            "grid_manifest_sha256": grid_digest,
            "stage": "representation",
            "partition": "validation",
            "selection_rule": REPRESENTATION_SELECTION_RULE,
            "feasibility_constraint": {
                "reference": evaluation_protocol["feasibility_reference"],
                "margin": margin,
                "one_sided_alpha": alpha,
                "cluster_unit": "immutable-base-lineage",
                "training_seed_treatment": "equal-weight-crossed-resampling-with-replacement",
                "lineage_treatment": "equal-weight-crossed-resampling-with-replacement",
                "bootstrap_replicates": bootstrap,
                "bootstrap_seed": bootstrap_seed,
                "pass_rule": "lower_bound_strictly_greater_than_negative_margin",
                "eligible_simpler": eligible_simpler,
                "if_core_pass": True,
            },
            "seed_selection_forbidden": True,
            "selected_simpler": selected,
            "retained_families": [selected, "if-core"],
            "gate_profile": protocol_identity["gate_profile"],
            "gate_receipt_sha256": protocol_identity["gate_receipt_sha256"],
            "gate_record_digest": protocol_identity["gate_record_digest"],
            "quality_preflight_receipt_sha256": protocol_identity["matched_metadata"][
                "quality_preflight_receipt_sha256"
            ],
            "quality_preflight_record_digest": protocol_identity["matched_metadata"][
                "quality_preflight_record_digest"
            ],
            "family_aggregates": aggregates,
            "protocol_identity": protocol_identity,
            "runtime_identity_by_training_seed": runtime_identity_by_seed,
            "runtime_implementation_registry": protocol_identity["matched_metadata"][
                "runtime_implementation_registry"
            ],
            "runtime_implementation_digest": protocol_identity["matched_metadata"][
                "runtime_implementation_digest"
            ],
            "source_reports": sorted(source_reports, key=lambda row: str(row["cell_id"])),
        }
    )
    _atomic_json(destination, receipt)
    print(json.dumps(receipt, indent=1))


def _load_grid(path: str | os.PathLike[str]) -> tuple[dict[str, Any], str]:
    from isingfold.rl.model import quality_policy_prior_contract

    grid = _strict_json(path)
    if (
        grid.get("schema") != "isingfold.staged-grid"
        or grid.get("schema_version") != 2
        or grid.get("name") != "if-core-v2-profile-i-hybrid-chimera-registered"
    ):
        raise ValueError("unsupported staged-grid manifest")
    stages = grid.get("stages")
    if not isinstance(stages, dict):
        raise ValueError("staged-grid manifest has no stages")
    quality_prior = grid.get("quality_policy_prior")
    if not isinstance(quality_prior, dict) or not isinstance(
        quality_prior.get("mode"), str
    ):
        raise ValueError("staged-grid manifest has no registered quality-policy prior")
    expected_quality_prior = quality_policy_prior_contract(quality_prior["mode"])
    if quality_prior != expected_quality_prior:
        raise ValueError(
            "staged-grid quality-policy prior does not match the implementation contract"
        )
    if quality_prior["publication_eligible"] is not True:
        raise ValueError("scientific staged-grid cannot use a diagnostic quality prior")
    expected = {"representation": 9, "rl_value": 18}
    for stage, count in expected.items():
        cells = stages.get(stage)
        if not isinstance(cells, list) or len(cells) != count:
            raise ValueError(f"grid stage {stage!r} must contain exactly {count} cells")
        ids = [cell.get("cell_id") for cell in cells if isinstance(cell, dict)]
        if len(ids) != count or len(set(ids)) != count:
            raise ValueError(f"grid stage {stage!r} has malformed or duplicate cell IDs")
        families = [cell.get("model_family") for cell in cells if isinstance(cell, dict)]
        allowed = set(REGISTERED_MODEL_FAMILIES)
        if stage == "rl_value":
            allowed.add("selected-simpler")
        unknown = sorted({family for family in families if family not in allowed})
        if len(families) != count or unknown:
            raise ValueError(f"grid stage {stage!r} has unknown model families: {unknown}")
        if any("quality_prior_mode" in cell for cell in cells):
            raise ValueError(
                "quality-prior mode is grid-global; per-cell overrides are forbidden"
            )
    representation_cells = stages["representation"]
    representation_seed_registry: tuple[int, ...] | None = None
    for family in REGISTERED_MODEL_FAMILIES:
        seeds = tuple(
            sorted(
                cell.get("seed")
                for cell in representation_cells
                if cell.get("model_family") == family
                and isinstance(cell.get("seed"), int)
                and not isinstance(cell.get("seed"), bool)
            )
        )
        if len(seeds) != 3 or len(set(seeds)) != 3:
            raise ValueError(f"representation family {family} lacks three registered seeds")
        if representation_seed_registry is None:
            representation_seed_registry = seeds
        elif seeds != representation_seed_registry:
            raise ValueError("representation families do not share one seed registry")
    rl_cells = stages["rl_value"]
    if any(
        cell.get("model_family") not in {"selected-simpler", "if-core"}
        or cell.get("method") not in TRAINING_METHODS
        for cell in rl_cells
    ):
        raise ValueError("RL-value grid contains an unregistered family or training method")
    rl_seed_registry: tuple[int, ...] | None = None
    for family in ("selected-simpler", "if-core"):
        for method in TRAINING_METHODS:
            seeds = tuple(
                sorted(
                    cell.get("seed")
                    for cell in rl_cells
                    if cell.get("model_family") == family
                    and cell.get("method") == method
                    and isinstance(cell.get("seed"), int)
                    and not isinstance(cell.get("seed"), bool)
                )
            )
            if len(seeds) != 3 or len(set(seeds)) != 3:
                raise ValueError(
                    f"RL-value configuration {family}/{method} lacks three registered seeds"
                )
            if rl_seed_registry is None:
                rl_seed_registry = seeds
            elif seeds != rl_seed_registry:
                raise ValueError("RL-value configurations do not share one seed registry")
    if rl_seed_registry != representation_seed_registry:
        raise ValueError("representation and RL-value stages use different seed registries")
    quality_resolution = grid.get("quality_resolution")
    if not isinstance(quality_resolution, dict) or set(quality_resolution) != {
        "min_resolved_rows",
        "min_resolved_lineages",
    }:
        raise ValueError("staged grid has no exact quality-resolution threshold")
    if any(
        isinstance(quality_resolution[field], bool)
        or not isinstance(quality_resolution[field], int)
        or quality_resolution[field] <= 1
        for field in ("min_resolved_rows", "min_resolved_lineages")
    ):
        raise ValueError("scientific grid quality-resolution minima must both exceed one")
    evaluation = grid.get("representation_evaluation")
    if not isinstance(evaluation, dict) or set(evaluation) != {
        "partition",
        "evaluation_seed",
        "repetitions",
        "audit_reads",
        "margin",
        "deployment_rule",
        "feasibility_reference",
        "feasibility_alpha",
        "feasibility_bootstrap",
        "selection_bootstrap_seed",
    }:
        raise ValueError("staged grid has no exact representation-evaluation protocol")
    if (
        evaluation["partition"] != "validation"
        or evaluation["deployment_rule"] != "categorical-temperature-one"
        or evaluation["feasibility_reference"] != "return_initial"
        or isinstance(evaluation["evaluation_seed"], bool)
        or not isinstance(evaluation["evaluation_seed"], int)
        or evaluation["evaluation_seed"] < 0
        or isinstance(evaluation["repetitions"], bool)
        or not isinstance(evaluation["repetitions"], int)
        or evaluation["repetitions"] <= 0
        or isinstance(evaluation["audit_reads"], bool)
        or not isinstance(evaluation["audit_reads"], int)
        or evaluation["audit_reads"] <= 0
        or isinstance(evaluation["margin"], bool)
        or not isinstance(evaluation["margin"], (int, float))
        or not 0.0 <= float(evaluation["margin"]) <= 1.0
        or isinstance(evaluation["feasibility_alpha"], bool)
        or not isinstance(evaluation["feasibility_alpha"], (int, float))
        or not 0.0 < float(evaluation["feasibility_alpha"]) < 0.5
        or isinstance(evaluation["feasibility_bootstrap"], bool)
        or not isinstance(evaluation["feasibility_bootstrap"], int)
        or evaluation["feasibility_bootstrap"] < 1_000
        or isinstance(evaluation["selection_bootstrap_seed"], bool)
        or not isinstance(evaluation["selection_bootstrap_seed"], int)
        or evaluation["selection_bootstrap_seed"] < 0
    ):
        raise ValueError("staged grid representation-evaluation protocol is malformed")
    rl_evaluation = grid.get("rl_value_evaluation")
    rl_evaluation_fields = {
        "partition",
        "evaluation_seed",
        "repetitions",
        "audit_reads",
        "margin",
        "deployment_rule",
        "feasibility_reference",
        "feasibility_alpha",
        "feasibility_bootstrap",
        "selection_bootstrap_seed",
        "aggregation",
        "primary_metric",
        "tie_break",
    }
    if not isinstance(rl_evaluation, dict) or set(rl_evaluation) != rl_evaluation_fields:
        raise ValueError("staged grid has no exact RL-value evaluation protocol")
    if (
        rl_evaluation["partition"] != "validation"
        or rl_evaluation["deployment_rule"] != "categorical-temperature-one"
        or rl_evaluation["feasibility_reference"] != "return_initial"
        or rl_evaluation["aggregation"] != "equal-training-seed-then-equal-immutable-base-lineage"
        or rl_evaluation["primary_metric"] != "unconditional-if-q3-s0-utility"
        or rl_evaluation["tie_break"] != "lower-online-seconds-then-registered-configuration-order"
        or isinstance(rl_evaluation["evaluation_seed"], bool)
        or not isinstance(rl_evaluation["evaluation_seed"], int)
        or rl_evaluation["evaluation_seed"] < 0
        or isinstance(rl_evaluation["repetitions"], bool)
        or not isinstance(rl_evaluation["repetitions"], int)
        or rl_evaluation["repetitions"] <= 0
        or isinstance(rl_evaluation["audit_reads"], bool)
        or not isinstance(rl_evaluation["audit_reads"], int)
        or rl_evaluation["audit_reads"] <= 0
        or isinstance(rl_evaluation["margin"], bool)
        or not isinstance(rl_evaluation["margin"], (int, float))
        or not math.isclose(float(rl_evaluation["margin"]), 0.02, rel_tol=0.0, abs_tol=1e-15)
        or isinstance(rl_evaluation["feasibility_alpha"], bool)
        or not isinstance(rl_evaluation["feasibility_alpha"], (int, float))
        or not 0.0 < float(rl_evaluation["feasibility_alpha"]) < 0.5
        or isinstance(rl_evaluation["feasibility_bootstrap"], bool)
        or not isinstance(rl_evaluation["feasibility_bootstrap"], int)
        or rl_evaluation["feasibility_bootstrap"] < 1_000
        or isinstance(rl_evaluation["selection_bootstrap_seed"], bool)
        or not isinstance(rl_evaluation["selection_bootstrap_seed"], int)
        or rl_evaluation["selection_bootstrap_seed"] < 0
    ):
        raise ValueError("staged grid RL-value evaluation protocol is malformed")
    if evaluation["evaluation_seed"] == rl_evaluation["evaluation_seed"]:
        raise ValueError(
            "representation screening and RL-value selection require disjoint "
            "evaluation seed domains"
        )
    complete_evaluation = grid.get("complete_system_evaluation")
    complete_fields = {
        "partition",
        "evaluation_seed",
        "repetitions",
        "audit_reads",
        "deployment_rule",
        "population_scope",
        "aggregation",
        "primary_metric",
        "bootstrap_replicates",
        "bootstrap_seed",
        "two_sided_alpha",
        "feasibility_noninferiority_margin",
        "learned_config_digest",
        "learned_config_file_sha256",
        "external_config_digest",
        "external_config_file_sha256",
    }
    if not isinstance(complete_evaluation, dict) or set(complete_evaluation) != complete_fields:
        raise ValueError("staged grid has no exact complete-system evaluation protocol")
    if (
        complete_evaluation["partition"] != "test"
        or complete_evaluation["deployment_rule"] != "categorical-temperature-one"
        or complete_evaluation["population_scope"] != "all-policy-instances-before-initialization"
        or complete_evaluation["aggregation"]
        != "equal-training-seed-then-equal-immutable-base-lineage"
        or complete_evaluation["primary_metric"] != "unconditional-if-q3-s0-utility"
        or isinstance(complete_evaluation["evaluation_seed"], bool)
        or not isinstance(complete_evaluation["evaluation_seed"], int)
        or complete_evaluation["evaluation_seed"] < 0
        or isinstance(complete_evaluation["repetitions"], bool)
        or not isinstance(complete_evaluation["repetitions"], int)
        or complete_evaluation["repetitions"] <= 0
        or isinstance(complete_evaluation["audit_reads"], bool)
        or not isinstance(complete_evaluation["audit_reads"], int)
        or complete_evaluation["audit_reads"] <= 0
        or isinstance(complete_evaluation["bootstrap_replicates"], bool)
        or not isinstance(complete_evaluation["bootstrap_replicates"], int)
        or complete_evaluation["bootstrap_replicates"] < 1_000
        or isinstance(complete_evaluation["bootstrap_seed"], bool)
        or not isinstance(complete_evaluation["bootstrap_seed"], int)
        or complete_evaluation["bootstrap_seed"] < 0
        or isinstance(complete_evaluation["two_sided_alpha"], bool)
        or not isinstance(complete_evaluation["two_sided_alpha"], (int, float))
        or not 0.0 < float(complete_evaluation["two_sided_alpha"]) < 0.5
        or isinstance(complete_evaluation["feasibility_noninferiority_margin"], bool)
        or not isinstance(complete_evaluation["feasibility_noninferiority_margin"], (int, float))
        or not math.isclose(
            float(complete_evaluation["feasibility_noninferiority_margin"]),
            0.02,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or any(
            not isinstance(complete_evaluation[name], str) or len(complete_evaluation[name]) != 64
            for name in (
                "learned_config_digest",
                "learned_config_file_sha256",
                "external_config_digest",
                "external_config_file_sha256",
            )
        )
    ):
        raise ValueError("staged grid complete-system evaluation protocol is malformed")
    return grid, _sha256_file(path)


def _load_representation_selection(
    path: str | os.PathLike[str],
    *,
    grid: Mapping[str, object],
    grid_digest: str,
    expected_quality_preflight_sha256: str | None = None,
    expected_quality_preflight_record_digest: str | None = None,
) -> tuple[str, str, str]:
    """Authenticate the validation-only family decision consumed by RL-value cells."""

    from isingfold.rl.experiment_selection import (
        _validate_runtime_identity,
        _validate_runtime_identity_by_seed,
        _validate_scientific_protocol_identity,
    )

    receipt_path = Path(path)
    receipt = _strict_json(receipt_path)
    _verify_record(receipt, "representation-selection receipt")
    if (
        receipt.get("schema") != REPRESENTATION_SELECTION_SCHEMA
        or receipt.get("schema_version") != REPRESENTATION_SELECTION_SCHEMA_VERSION
        or receipt.get("stage") != "representation"
        or receipt.get("partition") != "validation"
        or receipt.get("seed_selection_forbidden") is not True
        or receipt.get("grid_manifest_sha256") != grid_digest
    ):
        raise ValueError("representation-selection receipt has an incompatible identity")
    selected = receipt.get("selected_simpler")
    if selected not in {"if-mlp", "if-dual"}:
        raise ValueError("representation-selection receipt has no registered simpler family")
    if receipt.get("retained_families") != [selected, "if-core"]:
        raise ValueError("representation-selection receipt has an invalid retained-family set")
    if (
        receipt.get("gate_profile") not in {"profile-i", "profile-i+profile-c"}
        or not isinstance(receipt.get("gate_receipt_sha256"), str)
        or len(receipt["gate_receipt_sha256"]) != 64
        or not isinstance(receipt.get("gate_record_digest"), str)
        or len(receipt["gate_record_digest"]) != 64
    ):
        raise ValueError("representation-selection receipt has no Profile-I gate binding")
    source_reports = receipt.get("source_reports")
    representation = (
        grid.get("stages", {}).get("representation")
        if isinstance(grid.get("stages"), dict)
        else None
    )
    if not isinstance(source_reports, list) or not isinstance(representation, list):
        raise ValueError("representation-selection receipt has no source-report census")
    expected_cells = {
        (cell["cell_id"], cell["model_family"], cell["seed"])
        for cell in representation
        if isinstance(cell, dict)
    }
    observed_cells = {
        (row.get("cell_id"), row.get("model_family"), row.get("seed"))
        for row in source_reports
        if isinstance(row, dict)
    }
    if len(source_reports) != len(expected_cells) or observed_cells != expected_cells:
        raise ValueError("representation-selection receipt does not cover the complete grid")
    protocol_identity = receipt.get("protocol_identity")
    expected_protocol_fields = {
        "population",
        "repetitions",
        "selector_digest",
        "corpus_manifest_sha256",
        "deployment_rule",
        "gate_profile",
        "gate_receipt_sha256",
        "gate_record_digest",
        "matched_metadata",
    }
    if (
        not isinstance(protocol_identity, dict)
        or set(protocol_identity) != expected_protocol_fields
    ):
        raise ValueError("representation-selection receipt has no exact protocol identity")
    matched_metadata = protocol_identity.get("matched_metadata")
    if (
        not isinstance(matched_metadata, dict)
        or receipt.get("quality_preflight_receipt_sha256")
        != matched_metadata.get("quality_preflight_receipt_sha256")
        or receipt.get("quality_preflight_record_digest")
        != matched_metadata.get("quality_preflight_record_digest")
    ):
        raise ValueError("representation-selection quality-preflight binding is inconsistent")
    _validate_scientific_protocol_identity(protocol_identity["matched_metadata"])
    from isingfold.rl.checkpoint import runtime_implementation_registry

    pinned_runtime_registry = receipt.get("runtime_implementation_registry")
    pinned_runtime_digest = receipt.get("runtime_implementation_digest")
    current_runtime_registry = runtime_implementation_registry()
    if (
        not isinstance(pinned_runtime_registry, dict)
        or pinned_runtime_digest != stable_digest(pinned_runtime_registry)
        or pinned_runtime_registry
        != protocol_identity["matched_metadata"].get("runtime_implementation_registry")
        or pinned_runtime_digest
        != protocol_identity["matched_metadata"].get("runtime_implementation_digest")
        or current_runtime_registry != pinned_runtime_registry
    ):
        raise ValueError(
            "all 27 validation cells require byte-identical model/env/PPO sources and "
            "dependency versions; the representation runtime registry differs"
        )
    matched_preflight_sha = protocol_identity["matched_metadata"].get(
        "quality_preflight_receipt_sha256"
    )
    matched_preflight_record = protocol_identity["matched_metadata"].get(
        "quality_preflight_record_digest"
    )
    if (
        expected_quality_preflight_sha256 is not None
        and matched_preflight_sha != expected_quality_preflight_sha256
    ) or (
        expected_quality_preflight_record_digest is not None
        and matched_preflight_record != expected_quality_preflight_record_digest
    ):
        raise ValueError("representation selection belongs to another pinned quality preflight")
    training_seeds = sorted({int(cell["seed"]) for cell in representation})
    runtime_by_seed = _validate_runtime_identity_by_seed(
        receipt.get("runtime_identity_by_training_seed"),
        expected_seeds=training_seeds,
    )
    for row in source_reports:
        assert isinstance(row, dict)
        seed = int(row["seed"])
        if _validate_runtime_identity(row.get("runtime_identity")) != runtime_by_seed[str(seed)]:
            raise ValueError(f"representation source runtime differs within training seed {seed}")
        if (
            row.get("runtime_implementation_registry") != pinned_runtime_registry
            or row.get("runtime_implementation_digest") != pinned_runtime_digest
        ):
            raise ValueError("all 27 validation cells require one byte-identical runtime registry")
    aggregates = receipt.get("family_aggregates")
    if not isinstance(aggregates, dict) or set(aggregates) != set(REGISTERED_MODEL_FAMILIES):
        raise ValueError("representation-selection receipt has incomplete family aggregates")
    evaluation = grid.get("representation_evaluation")
    if not isinstance(evaluation, dict):
        raise ValueError("representation-selection grid lacks its evaluation protocol")
    margin = float(evaluation["margin"])
    constraint = receipt.get("feasibility_constraint")
    constraint_fields = {
        "reference",
        "margin",
        "one_sided_alpha",
        "cluster_unit",
        "training_seed_treatment",
        "lineage_treatment",
        "bootstrap_replicates",
        "bootstrap_seed",
        "pass_rule",
        "eligible_simpler",
        "if_core_pass",
    }
    expected_constraint = {
        "reference": evaluation["feasibility_reference"],
        "margin": margin,
        "one_sided_alpha": float(evaluation["feasibility_alpha"]),
        "cluster_unit": "immutable-base-lineage",
        "training_seed_treatment": "equal-weight-crossed-resampling-with-replacement",
        "lineage_treatment": "equal-weight-crossed-resampling-with-replacement",
        "bootstrap_replicates": int(evaluation["feasibility_bootstrap"]),
        "bootstrap_seed": int(evaluation["selection_bootstrap_seed"]),
        "pass_rule": "lower_bound_strictly_greater_than_negative_margin",
    }
    if not isinstance(constraint, dict) or set(constraint) != constraint_fields:
        raise ValueError("representation-selection receipt has no exact feasibility constraint")
    if any(constraint.get(name) != value for name, value in expected_constraint.items()):
        raise ValueError("representation-selection feasibility protocol differs from its grid")
    aggregate_fields = {
        "seed_count",
        "seeds",
        "independent_lineages",
        "valid_return_rate",
        "reference_valid_return_rate",
        "utility_mean",
        "online_seconds_mean",
        "feasibility_delta_vs_return_initial",
        "feasibility_ci_low",
        "feasibility_ci_high",
        "feasibility_noninferior",
        "per_seed",
    }
    for family, aggregate in aggregates.items():
        if not isinstance(aggregate, dict) or set(aggregate) != aggregate_fields:
            raise ValueError(f"representation aggregate for {family} has an invalid schema")
        numeric = (
            aggregate["valid_return_rate"],
            aggregate["reference_valid_return_rate"],
            aggregate["utility_mean"],
            aggregate["online_seconds_mean"],
            aggregate["feasibility_delta_vs_return_initial"],
            aggregate["feasibility_ci_low"],
            aggregate["feasibility_ci_high"],
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in numeric
        ):
            raise ValueError(f"representation aggregate for {family} has non-finite metrics")
        if not math.isclose(
            float(aggregate["feasibility_delta_vs_return_initial"]),
            float(aggregate["valid_return_rate"])
            - float(aggregate["reference_valid_return_rate"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"representation aggregate for {family} has inconsistent feasibility")
        expected_pass = float(aggregate["feasibility_ci_low"]) > -margin
        if aggregate["feasibility_noninferior"] is not expected_pass:
            raise ValueError(f"representation aggregate for {family} has inconsistent gate status")
    simpler = ("if-mlp", "if-dual")
    eligible = [
        family for family in simpler if aggregates[family]["feasibility_noninferior"] is True
    ]
    if not eligible or constraint["eligible_simpler"] != eligible:
        raise ValueError("representation-selection feasible-family shortlist is inconsistent")
    if (
        constraint["if_core_pass"] is not True
        or aggregates["if-core"]["feasibility_noninferior"] is not True
    ):
        raise ValueError("representation-selection IF-Core feasibility gate did not pass")
    expected_selected = max(
        eligible,
        key=lambda family: (
            float(aggregates[family]["utility_mean"]),
            -float(aggregates[family]["online_seconds_mean"]),
            family == "if-mlp",
        ),
    )
    if (
        receipt.get("selection_rule") != REPRESENTATION_SELECTION_RULE
        or selected != expected_selected
    ):
        raise ValueError("representation-selection decision does not follow its registered rule")
    record_digest = receipt.get("record_digest")
    if not isinstance(record_digest, str) or len(record_digest) != 64:
        raise ValueError("representation-selection receipt has an invalid record digest")
    return selected, _sha256_file(receipt_path), record_digest


def _representation_checkpoint_payload_digest(
    path: str | os.PathLike[str],
    *,
    expected_selection_sha256: str,
    expected_selection_record_digest: str,
    grid_cell: str,
    model_family: str,
    seed: int,
) -> str:
    """Read one evaluated representation payload from an authenticated selection."""

    if not _is_lower_sha256(expected_selection_sha256) or not _is_lower_sha256(
        expected_selection_record_digest
    ):
        raise ValueError("representation payload lookup requires pinned selection digests")
    receipt_path = Path(path)
    if _sha256_file(receipt_path) != expected_selection_sha256:
        raise ValueError("representation selection changed before payload lookup")
    receipt = _strict_json(receipt_path)
    _verify_record(receipt, "representation-selection receipt")
    if (
        receipt.get("schema") != REPRESENTATION_SELECTION_SCHEMA
        or receipt.get("schema_version") != REPRESENTATION_SELECTION_SCHEMA_VERSION
        or receipt.get("record_digest") != expected_selection_record_digest
    ):
        raise ValueError("representation payload lookup received another selection receipt")
    source_reports = receipt.get("source_reports")
    if not isinstance(source_reports, list):
        raise ValueError("representation selection has no source-report census")
    matches = [
        row
        for row in source_reports
        if isinstance(row, dict)
        and row.get("cell_id") == grid_cell
        and row.get("model_family") == model_family
        and row.get("seed") == seed
    ]
    if len(matches) != 1:
        raise ValueError("representation selection does not identify one warm-start checkpoint")
    payload_digest = matches[0].get("checkpoint_payload_digest")
    if not _is_lower_sha256(payload_digest):
        raise ValueError("selected representation has no valid checkpoint payload digest")
    return payload_digest


def _validate_exact_gate_result(
    result: object,
    *,
    source_corpus_manifest_sha256: str,
) -> dict[str, object]:
    from isingfold.rl.data.exact_conformance import (
        EXACT_CONFORMANCE_CORPUS_ID,
        EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES,
        EXACT_CONFORMANCE_MAX_TASKS,
        EXACT_CONFORMANCE_SELECTION_SEED,
        EXACT_CONFORMANCE_SELECTOR_VERSION,
        EXACT_CONFORMANCE_TIE_DOMAIN,
    )

    if not isinstance(result, dict) or set(result) != _EXACT_GATE_RESULT_FIELDS:
        raise ValueError("Gate 1 exact-conformance result schema differs")
    identity = result["authenticated_exact_corpus"]
    if not isinstance(identity, dict) or set(identity) != _EXACT_CORPUS_IDENTITY_FIELDS:
        raise ValueError("Gate 1 authenticated exact-corpus identity schema differs")

    task_ids = identity["task_ids"]
    if (
        not isinstance(task_ids, list)
        or len(task_ids) != EXACT_CONFORMANCE_MAX_TASKS
        or any(type(task_id) is not str or not task_id for task_id in task_ids)
        or task_ids != sorted(set(task_ids))
    ):
        raise ValueError("Gate 1 identity must contain eight sorted unique task IDs")
    base_lineages = identity["base_lineages"]
    if (
        not isinstance(base_lineages, list)
        or len(base_lineages) != EXACT_CONFORMANCE_MAX_TASKS
        or any(type(lineage) is not str or not lineage for lineage in base_lineages)
        or base_lineages != sorted(set(base_lineages))
    ):
        raise ValueError("Gate 1 identity must contain eight sorted unique base lineages")

    selector_identity = identity["selector_implementation"]
    expected_selector_fields = {
        "implementation",
        "selection_seed",
        "source_sha256",
        "tie_domain",
        "version",
    }
    if (
        not isinstance(selector_identity, dict)
        or set(selector_identity) != expected_selector_fields
    ):
        raise ValueError("Gate 1 selector implementation identity schema differs")
    if (
        selector_identity["implementation"]
        != "isingfold.rl.data.exact_conformance.select_exact_conformance_tasks"
        or type(selector_identity["selection_seed"]) is not int
        or selector_identity["selection_seed"] != EXACT_CONFORMANCE_SELECTION_SEED
        or selector_identity["tie_domain"] != EXACT_CONFORMANCE_TIE_DOMAIN
        or selector_identity["version"] != EXACT_CONFORMANCE_SELECTOR_VERSION
        or not _is_lower_sha256(selector_identity["source_sha256"])
    ):
        raise ValueError("Gate 1 selector implementation identity differs")

    exact_bounds = result["exact_bounds"]
    registered = result["registered_population"]
    if exact_bounds != {
        "max_tasks": EXACT_CONFORMANCE_MAX_TASKS,
        "max_logical_variables_per_task": EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES,
    }:
        raise ValueError("Gate 1 exact bounds differ from the registered protocol")
    if registered != {
        "exact_task_count": True,
        "task_count": EXACT_CONFORMANCE_MAX_TASKS,
        "unique_task_id_count": EXACT_CONFORMANCE_MAX_TASKS,
        "unique_base_lineage_count": EXACT_CONFORMANCE_MAX_TASKS,
    }:
        raise ValueError("Gate 1 registered population is not exactly eight independent tasks")

    selected_census = identity["selected_marginal_census"]
    if (
        not isinstance(selected_census, dict)
        or selected_census.get("selected_task_count") != EXACT_CONFORMANCE_MAX_TASKS
        or selected_census.get("selected_base_lineage_count") != EXACT_CONFORMANCE_MAX_TASKS
    ):
        raise ValueError("Gate 1 selected census is not exactly eight independent tasks")
    if not isinstance(identity["source_population_census"], dict):
        raise ValueError("Gate 1 source population census must be an object")
    if (
        identity["corpus_id"] != EXACT_CONFORMANCE_CORPUS_ID
        or identity["exact_reproduction"] is not True
        or type(identity["task_count"]) is not int
        or identity["task_count"] != EXACT_CONFORMANCE_MAX_TASKS
        or type(identity["max_tasks"]) is not int
        or identity["max_tasks"] != EXACT_CONFORMANCE_MAX_TASKS
        or type(identity["max_logical_variables_per_task"]) is not int
        or identity["max_logical_variables_per_task"] != EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES
        or identity["source_corpus_manifest_sha256"] != source_corpus_manifest_sha256
        or not _is_lower_sha256(identity["file_sha256"])
        or not _is_lower_sha256(identity["record_digest"])
        or type(identity["path"]) is not str
        or not identity["path"]
        or type(result["instances_requested"]) is not int
        or result["instances_requested"] != EXACT_CONFORMANCE_MAX_TASKS
        or type(result["eligible_instances"]) is not int
        or result["eligible_instances"] != EXACT_CONFORMANCE_MAX_TASKS
        or result["pass"] is not True
    ):
        raise ValueError("Gate 1 exact-conformance identity differs from the registered protocol")
    return identity


def _validate_support_headroom_gate_result(
    result: object,
    *,
    scalable_sampling: object,
    reward_reads: object,
) -> None:
    """Fail closed unless Gate 2 carries the registered v6 headroom evidence."""

    from statistics import NormalDist

    from isingfold.rl.gates import (
        SUPPORT_HEADROOM_INFERENCE_PROTOCOL,
        SUPPORT_HEADROOM_BERNOULLI_DIFFERENCE_VARIANCE_BOUND,
        SUPPORT_HEADROOM_MINIMUM_EFFECT,
        SUPPORT_HEADROOM_MINIMUM_STRATUM_COVERAGE,
        SUPPORT_HEADROOM_ONE_SIDED_ALPHA,
    )

    required_result_fields = {
        "eligible_instances",
        "headroom_confirmation_blocks",
        "headroom_inference",
        "instances_requested",
        "mean_supported_headroom_over_initial",
        "pass",
        "supported_headroom_se",
        "thresholds",
    }
    if (
        not isinstance(result, dict)
        or not required_result_fields <= set(result)
        or result.get("pass") is not True
    ):
        raise ValueError("Gate 2 lacks a passing support-headroom result")
    expected_thresholds = {
        "min_feasibility_support_recall": 0.5,
        "min_within_support_quality_spread": SUPPORT_HEADROOM_MINIMUM_EFFECT,
        "max_broad_pool_quality_advantage": 0.05,
        "min_mean_headroom_over_initial": SUPPORT_HEADROOM_MINIMUM_EFFECT,
        "headroom_one_sided_alpha": SUPPORT_HEADROOM_ONE_SIDED_ALPHA,
        "min_per_stratum_headroom_coverage": (SUPPORT_HEADROOM_MINIMUM_STRATUM_COVERAGE),
    }
    if result["thresholds"] != expected_thresholds:
        raise ValueError("Gate 2 thresholds differ from the registered protocol")

    inference = result["headroom_inference"]
    inference_fields = {
        "all_strata_meet_minimum_coverage",
        "between_lineage_standard_error",
        "confirmation_protocol",
        "eligible_instances",
        "evaluator_reads_per_confirmation_block",
        "evaluator_standard_error_upper_bound",
        "evaluator_variance_upper_bound_per_lineage",
        "headroom_one_sided_lower_confidence_bound",
        "independent_unit",
        "instances_meeting_minimum_headroom",
        "mean_supported_headroom_over_initial",
        "minimum_mean_headroom_over_initial",
        "minimum_per_stratum_coverage",
        "normal_quantile",
        "note",
        "one_sided_alpha",
        "overall_instance_coverage",
        "pass",
        "per_stratum_headroom",
        "protocol",
        "supported_headroom_se",
    }
    if not isinstance(inference, dict) or set(inference) != inference_fields:
        raise ValueError("Gate 2 headroom-inference schema differs")
    if (
        inference["protocol"] != SUPPORT_HEADROOM_INFERENCE_PROTOCOL
        or inference["independent_unit"] != "one-selected-task-per-immutable-base-lineage"
        or inference["one_sided_alpha"] != SUPPORT_HEADROOM_ONE_SIDED_ALPHA
        or inference["minimum_mean_headroom_over_initial"] != SUPPORT_HEADROOM_MINIMUM_EFFECT
        or inference["minimum_per_stratum_coverage"] != SUPPORT_HEADROOM_MINIMUM_STRATUM_COVERAGE
        or inference["all_strata_meet_minimum_coverage"] is not True
        or inference["pass"] is not True
    ):
        raise ValueError("Gate 2 headroom inference differs from the registered decision rule")

    def finite_number(value: object) -> bool:
        return type(value) in {int, float} and math.isfinite(float(value))

    mean = inference["mean_supported_headroom_over_initial"]
    between_standard_error = inference["between_lineage_standard_error"]
    evaluator_standard_error = inference["evaluator_standard_error_upper_bound"]
    standard_error = inference["supported_headroom_se"]
    lower_bound = inference["headroom_one_sided_lower_confidence_bound"]
    quantile = inference["normal_quantile"]
    if not all(
        finite_number(value)
        for value in (
            mean,
            between_standard_error,
            evaluator_standard_error,
            standard_error,
            lower_bound,
            quantile,
        )
    ):
        raise ValueError("Gate 2 headroom inference has a non-finite statistic")
    if isinstance(reward_reads, bool) or not isinstance(reward_reads, int) or reward_reads <= 0:
        raise ValueError("Gate 2 reward-read denominator is malformed")
    expected_quantile = NormalDist().inv_cdf(1.0 - SUPPORT_HEADROOM_ONE_SIDED_ALPHA)
    eligible = inference["eligible_instances"]
    if type(eligible) is not int or eligible <= 0:
        raise ValueError("Gate 2 headroom denominator is malformed")
    expected_evaluator_se = math.sqrt(
        SUPPORT_HEADROOM_BERNOULLI_DIFFERENCE_VARIANCE_BOUND / (reward_reads * eligible)
    )
    expected_standard_error = math.sqrt(
        float(between_standard_error) ** 2 + expected_evaluator_se**2
    )
    expected_lower = float(mean) - expected_quantile * float(standard_error)
    if (
        inference["confirmation_protocol"]
        != "select-on-screening-block-rescore-on-independent-block-v1"
        or inference["evaluator_reads_per_confirmation_block"] != reward_reads
        or not math.isclose(
            float(inference["evaluator_variance_upper_bound_per_lineage"]),
            SUPPORT_HEADROOM_BERNOULLI_DIFFERENCE_VARIANCE_BOUND / reward_reads,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or float(between_standard_error) < 0.0
        or not math.isclose(
            float(evaluator_standard_error),
            expected_evaluator_se,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(standard_error),
            expected_standard_error,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or float(standard_error) < 0.0
        or not math.isclose(float(quantile), expected_quantile, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(float(lower_bound), expected_lower, rel_tol=0.0, abs_tol=1e-12)
        or float(lower_bound) < SUPPORT_HEADROOM_MINIMUM_EFFECT
        or result["mean_supported_headroom_over_initial"] != mean
        or result["supported_headroom_se"] != standard_error
    ):
        raise ValueError("Gate 2 headroom lower bound is inconsistent or insufficient")

    if not isinstance(scalable_sampling, dict):
        raise ValueError("Gate 2 has no authenticated scalable-sampling receipt")
    selected_strata = scalable_sampling.get("selected_stratum_digests")
    if (
        not isinstance(selected_strata, list)
        or not selected_strata
        or any(not _is_lower_sha256(value) for value in selected_strata)
    ):
        raise ValueError("Gate 2 scalable-sampling strata are malformed")
    if (
        eligible != len(selected_strata)
        or result["eligible_instances"] != eligible
        or result["instances_requested"] != eligible
        or result["headroom_confirmation_blocks"] != 2 * eligible
    ):
        raise ValueError("Gate 2 headroom denominator differs from sampled base lineages")

    per_stratum = inference["per_stratum_headroom"]
    if not isinstance(per_stratum, dict) or set(per_stratum) != set(selected_strata):
        raise ValueError("Gate 2 headroom strata differ from authenticated sampling")
    total_successes = 0
    for stratum, expected_count in {
        key: selected_strata.count(key) for key in set(selected_strata)
    }.items():
        row = per_stratum[stratum]
        row_fields = {
            "coverage",
            "eligible_instances",
            "instances_meeting_minimum_headroom",
            "mean_supported_headroom_over_initial",
            "pass",
        }
        if not isinstance(row, dict) or set(row) != row_fields:
            raise ValueError("Gate 2 per-stratum headroom schema differs")
        successes = row["instances_meeting_minimum_headroom"]
        coverage = row["coverage"]
        if (
            row["eligible_instances"] != expected_count
            or type(successes) is not int
            or not 0 <= successes <= expected_count
            or not finite_number(coverage)
            or not math.isclose(
                float(coverage), successes / expected_count, rel_tol=0.0, abs_tol=1e-12
            )
            or row["pass"] is not (float(coverage) >= SUPPORT_HEADROOM_MINIMUM_STRATUM_COVERAGE)
        ):
            raise ValueError("Gate 2 per-stratum headroom result is inconsistent")
        total_successes += successes
    if (
        inference["instances_meeting_minimum_headroom"] != total_successes
        or not finite_number(inference["overall_instance_coverage"])
        or not math.isclose(
            float(inference["overall_instance_coverage"]),
            total_successes / eligible,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Gate 2 overall headroom coverage is inconsistent")


def _validate_authenticated_k2_support_gate_result(
    result: object,
    *,
    plan_identity: object,
) -> None:
    """Validate the target-free, exact full-support Gate 2 receipt."""

    from isingfold.rl.data.quality import (
        validate_quality_initializer_bank_contract,
    )
    from isingfold.rl.contracts import OPCODES
    from isingfold.rl.gates import (
        SUPPORT_FULL_AUDIT_MINIMUM_LINEAGES,
        SUPPORT_FULL_AUDIT_PROTOCOL,
    )

    fields = {
        "audited_candidate_count",
        "candidate_count",
        "candidate_selection",
        "family_counts",
        "initializer_bank",
        "legal_candidate_count",
        "legal_family_counts",
        "legal_opcode_counts",
        "minimum_independent_lineages",
        "missing_required_families",
        "opcode_counts",
        "pass",
        "plan_record_digest",
        "plan_sha256",
        "protocol",
        "record_digest",
        "required_families",
        "rows",
        "sampled_independent_lineages",
        "sampled_lineages_digest",
        "schema",
        "schema_version",
        "target_accessed",
    }
    if not isinstance(result, dict) or set(result) != fields:
        raise ValueError("Gate 2 authenticated full-support schema differs")
    _verify_record(result, "Gate 2 authenticated full-support result")
    if (
        result["schema"] != "isingfold.gate2-authenticated-full-support"
        or result["schema_version"] != 1
        or result["protocol"] != SUPPORT_FULL_AUDIT_PROTOCOL
        or result["candidate_selection"] != "all-legal-materialized-candidates-no-truncation"
        or result["target_accessed"] is not False
        or result["pass"] is not True
        or result["minimum_independent_lineages"] != SUPPORT_FULL_AUDIT_MINIMUM_LINEAGES
    ):
        raise ValueError("Gate 2 differs from the registered target-free protocol")
    if (
        not isinstance(plan_identity, dict)
        or set(plan_identity)
        != {
            "initializer_bank_contract_record_digest",
            "initializer_bank_manifest_sha256",
            "raw_sha256",
            "record_digest",
        }
        or result["plan_sha256"] != plan_identity["raw_sha256"]
        or result["plan_record_digest"] != plan_identity["record_digest"]
    ):
        raise ValueError("Gate 2 differs from the externally pinned resolution plan")
    for name, value in plan_identity.items():
        _require_lower_sha256(value, f"Gate 2 resolution-plan {name}")
    bank = result["initializer_bank"]
    if not isinstance(bank, Mapping):
        raise ValueError("Gate 2 omits its initializer-bank contract")
    try:
        validate_quality_initializer_bank_contract(bank, require_publication=True)
    except (TypeError, ValueError) as error:
        raise ValueError(str(error)) from error
    if (
        bank["manifest_sha256"] != plan_identity["initializer_bank_manifest_sha256"]
        or bank["record_digest"] != plan_identity["initializer_bank_contract_record_digest"]
    ):
        raise ValueError("Gate 2 initializer bank differs from its resolution-plan binding")

    sampled = result["sampled_independent_lineages"]
    candidate_count = result["candidate_count"]
    legal_count = result["legal_candidate_count"]
    audited_count = result["audited_candidate_count"]
    if (
        type(sampled) is not int
        or sampled < SUPPORT_FULL_AUDIT_MINIMUM_LINEAGES
        or type(candidate_count) is not int
        or type(legal_count) is not int
        or type(audited_count) is not int
        or candidate_count < legal_count
        or legal_count <= 0
        or audited_count != legal_count
        or result["missing_required_families"] != []
    ):
        raise ValueError("Gate 2 full-support denominators are inconsistent")
    required = result["required_families"]
    legal_families = result["legal_family_counts"]
    if (
        not isinstance(required, list)
        or not required
        or any(not isinstance(family, str) or not family for family in required)
        or len(set(required)) != len(required)
        or not isinstance(legal_families, dict)
        or any(
            type(legal_families.get(family)) is not int or legal_families[family] <= 0
            for family in required
        )
    ):
        raise ValueError("Gate 2 does not cover every required legal action family")
    rows = result["rows"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("Gate 2 has no exact replay rows")
    row_fields = {
        "audited_indices",
        "audited_legal_candidate_count",
        "base_lineage",
        "candidate_count",
        "candidate_payload_root",
        "family_counts",
        "initializer_bank_episode_index",
        "initializer_bootstrap_record_digest",
        "instance_id",
        "legal_candidate_count",
        "legal_family_counts",
        "legal_indices",
        "legal_opcode_counts",
        "opcode_counts",
        "record_digest",
        "row_id",
        "state_fingerprint",
        "support_fingerprint",
        "task_id",
    }
    row_ids: set[str] = set()
    lineages: set[str] = set()
    summed_candidates = 0
    summed_legal = 0
    summed_families: dict[str, int] = {}
    summed_legal_families: dict[str, int] = {}
    summed_opcodes: dict[str, int] = {}
    summed_legal_opcodes: dict[str, int] = {}

    def add_counts(total: dict[str, int], value: object, *, label: str) -> None:
        if not isinstance(value, dict) or any(
            not isinstance(key, str) or not key or type(count) is not int or count < 0
            for key, count in value.items()
        ):
            raise ValueError(f"Gate 2 {label} census is malformed")
        for key, count in value.items():
            total[key] = total.get(key, 0) + count

    for row in rows:
        if not isinstance(row, dict) or set(row) != row_fields:
            raise ValueError("Gate 2 exact replay row schema differs")
        _verify_record(row, "Gate 2 exact replay row")
        row_id = row["row_id"]
        lineage = row["base_lineage"]
        legal_indices = row["legal_indices"]
        candidate_count_row = row["candidate_count"]
        legal_count_row = row["legal_candidate_count"]
        episode_index = row["initializer_bank_episode_index"]
        if (
            not isinstance(row_id, str)
            or not row_id
            or row_id in row_ids
            or not isinstance(lineage, str)
            or not lineage
            or not isinstance(row["task_id"], str)
            or not row["task_id"]
            or not isinstance(row["instance_id"], str)
            or not row["instance_id"]
            or not isinstance(legal_indices, list)
            or any(type(index) is not int for index in legal_indices)
            or len(set(legal_indices)) != len(legal_indices)
            or row["audited_indices"] != legal_indices
            or type(candidate_count_row) is not int
            or type(legal_count_row) is not int
            or candidate_count_row < legal_count_row
            or legal_count_row <= 0
            or len(legal_indices) != legal_count_row
            or row["audited_legal_candidate_count"] != legal_count_row
            or legal_indices != sorted(legal_indices)
            or legal_indices
            and (legal_indices[0] < 0 or legal_indices[-1] >= candidate_count_row)
            or type(episode_index) is not int
            or not bank["episode_schedule_start"]
            <= episode_index
            < bank["episode_schedule_stop_exclusive"]
        ):
            raise ValueError("Gate 2 exact replay row is inconsistent")
        for digest_field in (
            "candidate_payload_root",
            "initializer_bootstrap_record_digest",
            "state_fingerprint",
            "support_fingerprint",
        ):
            _require_lower_sha256(row[digest_field], f"Gate 2 row {digest_field}")
        row_family_counts = row["family_counts"]
        row_legal_family_counts = row["legal_family_counts"]
        row_opcode_counts = row["opcode_counts"]
        row_legal_opcode_counts = row["legal_opcode_counts"]
        count_maps = (
            row_family_counts,
            row_legal_family_counts,
            row_opcode_counts,
            row_legal_opcode_counts,
        )
        if any(
            not isinstance(counts, dict)
            or any(
                not isinstance(key, str) or not key or type(count) is not int or count < 0
                for key, count in counts.items()
            )
            for counts in count_maps
        ):
            raise ValueError("Gate 2 exact replay row census is malformed")
        if (
            set(row_opcode_counts) != set(OPCODES)
            or set(row_legal_opcode_counts) != set(OPCODES)
            or sum(row_family_counts.values()) != candidate_count_row
            or sum(row_opcode_counts.values()) != candidate_count_row
            or sum(row_legal_family_counts.values()) != legal_count_row
            or sum(row_legal_opcode_counts.values()) != legal_count_row
            or not set(row_legal_family_counts) <= set(row_family_counts)
            or any(
                row_legal_family_counts[family] > row_family_counts[family]
                for family in row_legal_family_counts
            )
        ):
            raise ValueError("Gate 2 exact replay row census is inconsistent")
        row_ids.add(row_id)
        lineages.add(lineage)
        summed_candidates += candidate_count_row
        summed_legal += legal_count_row
        add_counts(summed_families, row_family_counts, label="row family")
        add_counts(
            summed_legal_families,
            row_legal_family_counts,
            label="row legal-family",
        )
        add_counts(summed_opcodes, row_opcode_counts, label="row opcode")
        add_counts(
            summed_legal_opcodes,
            row_legal_opcode_counts,
            label="row legal-opcode",
        )
    if (
        len(lineages) != sampled
        or summed_candidates != candidate_count
        or summed_legal != legal_count
        or result["sampled_lineages_digest"] != content_digest(sorted(lineages))
        or result["family_counts"] != dict(sorted(summed_families.items()))
        or result["legal_family_counts"] != dict(sorted(summed_legal_families.items()))
        or result["opcode_counts"] != dict(sorted(summed_opcodes.items()))
        or result["legal_opcode_counts"] != dict(sorted(summed_legal_opcodes.items()))
    ):
        raise ValueError("Gate 2 aggregate differs from its exact replay rows")


def _load_gate_receipt(
    path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    corpus: str | os.PathLike[str],
    selector: str | os.PathLike[str],
    context: Context,
    require_profile_c: bool = False,
    expected_grid_manifest_sha256: str | None = None,
    expected_quality_preflight_sha256: str | None = None,
    expected_quality_preflight_record_digest: str | None = None,
) -> tuple[str, str]:
    """Authenticate the selected gate scope before representation training."""

    from isingfold.rl.data.exact_conformance import read_regular_file

    expected_file_sha256 = _require_lower_sha256(
        expected_sha256, "expected release-gate receipt SHA-256"
    )
    receipt_path, receipt_raw = read_regular_file(path, "release-gate receipt")
    observed_file_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    if not hmac.compare_digest(observed_file_sha256, expected_file_sha256):
        raise ValueError("release-gate receipt differs from its externally pinned SHA-256")
    receipt = _strict_json_object(receipt_raw, str(receipt_path))
    if receipt_raw != canonical_json_bytes(receipt) + b"\n":
        raise ValueError("release-gate receipt is not canonical JSON with one newline")
    if set(receipt) != _GATE_RECEIPT_FIELDS:
        raise ValueError("release-gate receipt schema fields differ")
    _verify_record(receipt, "release-gate receipt")
    selector_fit_path = Path(selector) / "fit_receipt.json"
    if not selector_fit_path.is_file():
        raise ValueError("selector bundle has no fit receipt for gate binding")
    selector_fit = _strict_json(selector_fit_path)
    _verify_record(selector_fit, "selector fit receipt")
    gate_authority = _validate_quality_authority_binding(
        receipt.get("quality_authority"),
        expected_role="evaluation_partition",
    )
    selector_authority = _validate_quality_authority_binding(
        selector_fit.get("quality_authority"),
        expected_role="training_partition",
    )
    if (
        receipt.get("schema") != GATE_RECEIPT_SCHEMA
        or receipt.get("schema_version") != GATE_RECEIPT_SCHEMA_VERSION
        or receipt.get("partition") != "validation"
        or receipt.get("gate_profile") != "profile-i"
        or receipt.get("advance") is not True
        or receipt.get("source_corpus_manifest_sha256") != _corpus_manifest_digest(corpus)
        or gate_authority["global"] != selector_authority["global"]
        or receipt.get("source_selector_digest") != selector_fit.get("selector_digest")
        or receipt.get("source_selector_fit_receipt_sha256") != _sha256_file(selector_fit_path)
        or receipt.get("context") != _context_snapshot(context)
        or receipt.get("context_digest") != content_digest(_context_snapshot(context))
        or receipt.get("grid_manifest_sha256") != expected_grid_manifest_sha256
        or receipt.get("quality_preflight_receipt_sha256") != expected_quality_preflight_sha256
        or receipt.get("quality_preflight_record_digest")
        != expected_quality_preflight_record_digest
    ):
        raise ValueError("release-gate receipt is failing or belongs to another experiment")
    for name in (
        "context_digest",
        "grid_manifest_sha256",
        "source_corpus_manifest_sha256",
        "source_selector_digest",
        "source_selector_fit_receipt_sha256",
        "source_selector_labels_manifest_sha256",
        "source_selector_labels_manifest_record_digest",
        "quality_preflight_receipt_sha256",
        "quality_preflight_record_digest",
    ):
        _require_lower_sha256(receipt.get(name), f"release-gate {name}")
    target_access = receipt.get("target_access")
    ground_partition_receipt = receipt.get("ground_partition_receipt")
    if not isinstance(target_access, dict) or not isinstance(ground_partition_receipt, dict):
        raise ValueError("release-gate receipt omits target-access receipts")
    _verify_record(target_access, "release-gate target access")
    _verify_record(ground_partition_receipt, "release-gate ground partition")
    partition_authority = gate_authority["evaluation_partition"]
    if not isinstance(partition_authority, dict):
        raise RuntimeError("release-gate authority lost its partition record")
    ground_identity = partition_authority.get("ground_partition")
    if (
        target_access.get("record_digest") != partition_authority.get("target_access_record_digest")
        or not isinstance(ground_identity, dict)
        or ground_partition_receipt.get("record_digest")
        != ground_identity.get("receipt_record_digest")
    ):
        raise ValueError("release-gate target access differs from quality authority")
    gates = receipt.get("gates")
    mandatory_names = [
        "gate_1_exact_conformance",
        "gate_2_authenticated_k2_full_support",
        "gate_3_selector_discrimination",
        "gate_4_profile_i_signal",
    ]
    if (
        not isinstance(gates, dict)
        or receipt.get("mandatory_gate_names") != mandatory_names
        or list(gates) != mandatory_names
        or not all(
            isinstance(result, dict) and result.get("pass") is True for result in gates.values()
        )
    ):
        raise ValueError("release-gate receipt does not contain four passing Profile-I gates")
    exact_identity = _validate_exact_gate_result(
        gates["gate_1_exact_conformance"],
        source_corpus_manifest_sha256=receipt["source_corpus_manifest_sha256"],
    )
    parameters = receipt.get("parameters")
    expected_parameter_fields = {
        "broad_reference_batches_per_instance",
        "exact_conformance_corpus",
        "instances_requested",
        "profile_episodes_per_instance",
        "quality_resolution_plan",
        "reward_reads",
        "scalable_gate_sampling",
    }
    if (
        not isinstance(parameters, dict)
        or set(parameters) != expected_parameter_fields
        or parameters["exact_conformance_corpus"] != exact_identity
    ):
        raise ValueError("release-gate parameters do not bind the authenticated Gate 1 corpus")
    _validate_authenticated_k2_support_gate_result(
        gates["gate_2_authenticated_k2_full_support"],
        plan_identity=parameters["quality_resolution_plan"],
    )
    optional = receipt.get("optional_diagnostics")
    if not isinstance(optional, dict) or set(optional) != {
        "profile_c_construction_readiness",
        "validation_support_headroom",
    }:
        raise ValueError("release-gate receipt has no separate Profile-C diagnostic")
    _validate_support_headroom_gate_result(
        optional["validation_support_headroom"],
        scalable_sampling=parameters["scalable_gate_sampling"],
        reward_reads=parameters["reward_reads"],
    )
    profile_c = optional["profile_c_construction_readiness"]
    if not isinstance(profile_c, dict) or not isinstance(profile_c.get("pass"), bool):
        raise ValueError("release-gate receipt has a malformed Profile-C diagnostic")
    if receipt.get("construction_ready") is not profile_c["pass"]:
        raise ValueError("release-gate receipt disagrees with its Profile-C diagnostic")
    if require_profile_c and profile_c["pass"] is not True:
        raise ValueError("explicit Profile-C gate requirement is not satisfied")
    record_digest = receipt.get("record_digest")
    if not isinstance(record_digest, str) or len(record_digest) != 64:
        raise ValueError("release-gate receipt has an invalid record digest")
    binding: dict[str, object] = {
        "gate_profile": "profile-i",
        "source_corpus_manifest_sha256": receipt["source_corpus_manifest_sha256"],
        "quality_authority": receipt["quality_authority"],
        "target_access": target_access,
        "ground_partition_receipt": ground_partition_receipt,
        "source_selector_digest": receipt["source_selector_digest"],
        "source_selector_fit_receipt_sha256": receipt["source_selector_fit_receipt_sha256"],
        "context_digest": receipt["context_digest"],
        "partition": receipt["partition"],
        "seed": receipt["seed"],
        "parameters": receipt["parameters"],
        "grid_manifest_sha256": receipt["grid_manifest_sha256"],
        "quality_preflight_receipt_sha256": receipt["quality_preflight_receipt_sha256"],
        "quality_preflight_record_digest": receipt["quality_preflight_record_digest"],
        "gates": gates,
    }
    if require_profile_c:
        binding["gate_profile"] = "profile-i+profile-c"
        binding["profile_c_construction_readiness"] = profile_c
    return observed_file_sha256, content_digest(binding)


def _quality_warm_source_binding(
    *,
    args: argparse.Namespace,
    registry,
    plan,
    cell,
    source_checkpoint_payload_digest: str,
    control_device_type: str,
) -> dict[str, object]:
    """Authenticate the full-Q treatment paired with one control seed."""

    from isingfold.rl.checkpoint import runtime_implementation_registry
    from isingfold.rl.ppo import (
        WARM_START_ACTION_VALUE_TARGET,
        WARM_START_ACTION_VALUE_WEIGHTING,
        WARM_START_ACTOR_CRITIC_LOSS,
        WARM_START_COMMIT_DELTA_TARGET,
        WARM_START_FULL_LOSS_PROFILE,
        WARM_START_REDUCTION,
        WARM_START_UTILITY_TARGET,
        WARM_START_UTILITY_WEIGHTING,
    )
    from isingfold.rl.quality_warm_control import (
        build_quality_warm_control_binding,
    )

    source_dir = (
        Path(args.representation_run_root)
        / "representation"
        / cell.source_treatment_cell_id
    )
    source_checkpoint = source_dir / "checkpoint.pt"
    if not source_checkpoint.is_file():
        raise ValueError("quality warm control requires the authenticated source checkpoint")
    receipt = _strict_json(source_dir / "run.json")
    _verify_record(receipt, "quality warm-control source treatment")
    if (
        receipt.get("schema") != RUN_SCHEMA
        or receipt.get("schema_version") != RUN_SCHEMA_VERSION
        or receipt.get("phase") != "representation"
        or receipt.get("model_family") != registry.model_family
        or receipt.get("method") != "supervised-ranking"
        or receipt.get("complete") is not True
        or receipt.get("checkpoint_payload_digest")
        != source_checkpoint_payload_digest
    ):
        raise ValueError("quality warm-control source treatment has an incompatible identity")
    contract = receipt.get("experiment_contract")
    counters = receipt.get("counters")
    if not isinstance(contract, Mapping) or not isinstance(counters, Mapping):
        raise ValueError("quality warm-control source treatment is malformed")
    expected_model = _model_identity(
        _model(
            registry.model_family,
            quality_prior_mode=registry.quality_prior_mode,
        ),
        registry.model_family,
    )
    if (
        contract.get("phase") != "representation"
        or contract.get("method") != "supervised-ranking"
        or contract.get("model_family") != registry.model_family
        or contract.get("model") != expected_model
        or contract.get("seed") != cell.seed
        or contract.get("grid_manifest_sha256") != registry.main_grid_sha256
        or contract.get("grid_cell") != cell.source_treatment_cell_id
        or contract.get("corpus_manifest_sha256") != plan.corpus_manifest_sha256
        or contract.get("selector_digest") != plan.selector_digest
        or contract.get("normalizer_digest") != plan.normalizer_digest
        or contract.get("quality_preflight_receipt_sha256")
        != plan.quality_preflight_sha256
        or contract.get("quality_preflight_record_digest")
        != plan.quality_preflight_record_digest
        or contract.get("device_type") != control_device_type
        or contract.get("deterministic_algorithms") is not args.deterministic
    ):
        raise ValueError("quality warm-control source and control scientific identities differ")
    protocol = registry.training_protocol
    hyperparameters = contract.get("hyperparameters")
    expected_hyperparameters = {
        "epochs": protocol.epochs,
        "minibatch_records": protocol.minibatch_records,
        "learning_rate": protocol.learning_rate,
        "weight_decay": protocol.weight_decay,
        "quality_manifest_sha256": plan.quality_manifest_sha256,
        "minimum_resolved_rows": protocol.minimum_resolved_rows,
        "minimum_resolved_lineages": protocol.minimum_resolved_lineages,
        "quality_preflight_receipt_sha256": plan.quality_preflight_sha256,
        "quality_preflight_record_digest": plan.quality_preflight_record_digest,
        "warm_start_loss": WARM_START_ACTOR_CRITIC_LOSS,
        "warm_start_loss_profile": WARM_START_FULL_LOSS_PROFILE.contract(),
        "warm_start_rank_coefficient": WARM_START_FULL_LOSS_PROFILE.weights.rank,
        "warm_start_action_value_target": WARM_START_ACTION_VALUE_TARGET,
        "warm_start_action_value_weighting": WARM_START_ACTION_VALUE_WEIGHTING,
        "warm_start_action_value_coefficient": (
            WARM_START_FULL_LOSS_PROFILE.weights.action_value
        ),
        "warm_start_commit_delta_target": WARM_START_COMMIT_DELTA_TARGET,
        "warm_start_commit_delta_coefficient": (
            WARM_START_FULL_LOSS_PROFILE.weights.commit_delta
        ),
        "warm_start_reduction": WARM_START_REDUCTION,
        "warm_start_utility_target": WARM_START_UTILITY_TARGET,
        "warm_start_utility_weighting": WARM_START_UTILITY_WEIGHTING,
        "warm_start_utility_coefficient": WARM_START_FULL_LOSS_PROFILE.weights.utility,
        "warm_start_diagnostic_binding": None,
    }
    if not isinstance(hyperparameters, Mapping) or any(
        hyperparameters.get(name) != value
        for name, value in expected_hyperparameters.items()
    ):
        raise ValueError("quality warm-control source treatment changes the warm objective")
    if (
        counters.get("epochs") != protocol.epochs
        or counters.get("optimizer_steps") != protocol.epochs
        or isinstance(counters.get("records"), bool)
        or not isinstance(counters.get("records"), int)
        or counters["records"] <= 0
    ):
        raise ValueError("quality warm-control source treatment has an incomplete budget")
    current_runtime = runtime_implementation_registry()
    runtime_digest = stable_digest(current_runtime)
    if (
        receipt.get("runtime_implementation_registry") != current_runtime
        or receipt.get("runtime_implementation_digest") != runtime_digest
    ):
        raise ValueError("quality warm-control source treatment uses another runtime")
    source_record_digest = receipt.get("record_digest")
    if not _is_lower_sha256(source_record_digest):
        raise ValueError("quality warm-control source treatment has no record digest")
    return build_quality_warm_control_binding(
        registry,
        plan,
        cell=cell,
        source_treatment_run_record_digest=source_record_digest,
        source_treatment_checkpoint_payload_digest=source_checkpoint_payload_digest,
        runtime_implementation_digest=runtime_digest,
    )


def cmd_warm_quality_control_cell(args: argparse.Namespace) -> None:
    """Train one fixed-seed rank-plus-value control after representation selection."""

    from isingfold.rl.quality_warm_control import (
        build_quality_warm_control_plan,
        load_quality_warm_control_registry,
    )

    expected_config_sha256 = _require_lower_sha256(
        args.expected_config_sha256,
        "quality warm-control config SHA-256",
    )
    registry = load_quality_warm_control_registry(
        args.config,
        main_grid_path=args.grid,
    )
    if registry.file_sha256 != expected_config_sha256:
        raise ValueError("quality warm-control config differs from its external SHA-256 pin")
    grid, grid_digest = _load_grid(args.grid)
    if grid_digest != _require_lower_sha256(
        args.expected_grid_sha256,
        "quality warm-control grid SHA-256",
    ):
        raise ValueError("quality warm-control main grid differs from its external pin")
    if grid_digest != registry.main_grid_sha256:
        raise ValueError("quality warm-control registry and main grid differ")
    if not 0 <= args.index < len(registry.training_seeds):
        raise ValueError("quality warm-control cell index is outside [0, 3)")

    context = _context(args, args.corpus)
    selector = _load_selector_bundle(args.selector, corpus=args.corpus)
    quality_preflight = _load_quality_preflight_receipt(
        args.quality_preflight_receipt,
        expected_sha256=args.expected_quality_preflight_sha256,
        quality_labels=args.quality_labels,
        corpus=args.corpus,
        selector=selector,
        context=context,
        min_resolved_rows=registry.training_protocol.minimum_resolved_rows,
        min_resolved_lineages=registry.training_protocol.minimum_resolved_lineages,
    )
    quality_preflight_sha256 = _sha256_file(args.quality_preflight_receipt)
    expected_selection_sha256 = _require_lower_sha256(
        args.expected_representation_selection_sha256,
        "representation-selection SHA-256",
    )
    selected_simpler, selection_sha256, selection_record_digest = (
        _load_representation_selection(
            args.representation_selection_receipt,
            grid=grid,
            grid_digest=grid_digest,
            expected_quality_preflight_sha256=quality_preflight_sha256,
            expected_quality_preflight_record_digest=quality_preflight["record_digest"],
        )
    )
    if selection_sha256 != expected_selection_sha256:
        raise ValueError("representation selection differs from its external SHA-256 pin")
    quality_manifest_sha256 = _sha256_file(Path(args.quality_labels) / "manifest.json")
    plan = build_quality_warm_control_plan(
        registry,
        selected_simpler=selected_simpler,
        representation_selection_sha256=selection_sha256,
        representation_selection_record_digest=selection_record_digest,
        corpus_manifest_sha256=selector.corpus_manifest_sha256,
        selector_digest=selector.selector_digest,
        normalizer_digest=selector.normalizer_digest,
        quality_manifest_sha256=quality_manifest_sha256,
        quality_preflight_sha256=quality_preflight_sha256,
        quality_preflight_record_digest=quality_preflight["record_digest"],
    )
    cell = plan.cells[args.index]
    source_payload_digest = _representation_checkpoint_payload_digest(
        args.representation_selection_receipt,
        expected_selection_sha256=selection_sha256,
        expected_selection_record_digest=selection_record_digest,
        grid_cell=cell.source_treatment_cell_id,
        model_family=cell.model_family,
        seed=cell.seed,
    )
    control_device_type = _resolve_device(args.device).type
    binding = _quality_warm_source_binding(
        args=args,
        registry=registry,
        plan=plan,
        cell=cell,
        source_checkpoint_payload_digest=source_payload_digest,
        control_device_type=control_device_type,
    )
    run_dir = Path(args.run_root) / "quality_warm_control" / cell.control_cell_id
    namespace = argparse.Namespace(
        corpus=args.corpus,
        selector=args.selector,
        quality_initializer_bank=args.quality_initializer_bank,
        expected_quality_initializer_bank_manifest_sha256=(
            args.expected_quality_initializer_bank_manifest_sha256
        ),
        quality_complete_config=args.quality_complete_config,
        quality_attestation=args.quality_attestation,
        expected_quality_attestation_digest=args.expected_quality_attestation_digest,
        expected_quality_publisher_id=args.expected_quality_publisher_id,
        ground_certificate_root=args.ground_certificate_root,
        expected_ground_certificate_root_sha256=(
            args.expected_ground_certificate_root_sha256
        ),
        quality_preflight_receipt=args.quality_preflight_receipt,
        expected_quality_preflight_sha256=args.expected_quality_preflight_sha256,
        allow_legacy_pilot=False,
        qubit_cap=None,
        quality_labels=args.quality_labels,
        seed=cell.seed,
        device=args.device,
        deterministic=args.deterministic,
        threads=args.threads,
        out=str(run_dir),
        resume=(run_dir / "checkpoint.pt").is_file(),
        grid_manifest_sha256=grid_digest,
        grid_cell=cell.control_cell_id,
        selection_receipt_sha256=selection_sha256,
        selection_record_digest=selection_record_digest,
        selection_mode="post-selection-direct-qmu-warm-control-v1",
        gate_receipt_sha256=None,
        gate_record_digest=None,
        gate_profile=None,
        model_family=cell.model_family,
        quality_prior_mode=registry.quality_prior_mode,
        epochs=registry.training_protocol.epochs,
        minibatch=registry.training_protocol.minibatch_records,
        learning_rate=registry.training_protocol.learning_rate,
        weight_decay=registry.training_protocol.weight_decay,
        min_resolved_rows=registry.training_protocol.minimum_resolved_rows,
        min_resolved_lineages=registry.training_protocol.minimum_resolved_lineages,
        warm_start_loss_profile_id=cell.profile_id,
        warm_start_diagnostic_binding=binding,
    )
    cmd_warm_start(namespace)


def cmd_grid_cell(args: argparse.Namespace) -> None:
    """Execute one immutable cell from the 9 plus 18 staged pilot registry."""

    grid, grid_digest = _load_grid(args.grid)
    cells = grid["stages"][args.stage]
    if not 0 <= args.index < len(cells):
        raise ValueError(f"cell index is outside stage {args.stage!r}")
    if args.stage == "representation":
        if args.gate_receipt is None:
            raise ValueError("representation stage requires a passing release-gate receipt")
        if (
            args.selection_receipt is not None
            or args.selected_simpler is not None
            or args.diagnostic_manual_selection
        ):
            raise ValueError("representation cells cannot consume an RL-value family selection")
    else:
        if args.gate_receipt is not None or args.require_profile_c_gate:
            raise ValueError("RL-value cells inherit gates through representation selection")
        if args.selection_receipt is not None and (
            args.selected_simpler is not None or args.diagnostic_manual_selection
        ):
            raise ValueError("selection receipt cannot be combined with a manual diagnostic choice")
        if args.selected_simpler is not None and not args.diagnostic_manual_selection:
            raise ValueError("manual simpler-family selection requires explicit diagnostic opt-in")
        if args.selection_receipt is None and args.selected_simpler is None:
            raise ValueError("RL-value stage requires a representation selection receipt")
    initializer_bank_inputs = (
        args.initializer_bank,
        args.expected_initializer_bank_manifest_sha256,
        args.complete_config,
    )
    if args.stage == "representation" and any(
        value is not None for value in initializer_bank_inputs
    ):
        raise ValueError("representation cells cannot consume a deployment initializer bank")
    if args.stage == "rl_value" and not all(
        isinstance(value, str) and value for value in initializer_bank_inputs
    ):
        raise ValueError(
            "RL-value cells require a deployment initializer bank, its external manifest "
            "SHA-256 pin, and the complete-system config"
        )
    if args.stage == "rl_value":
        from isingfold.rl.complete_system import CompleteSystemConfig

        deployment_config = CompleteSystemConfig.from_mapping(_strict_json(args.complete_config))
        _validate_registered_complete_config(
            grid["complete_system_evaluation"],
            arm="learned",
            semantic_digest=deployment_config.digest,
            file_sha256=_sha256_file(args.complete_config),
        )
    grid_context = _context(args, args.corpus)
    grid_selector = _load_selector_bundle(args.selector, corpus=args.corpus)
    resolution_protocol = grid["quality_resolution"]
    quality_preflight = _load_quality_preflight_receipt(
        args.quality_preflight_receipt,
        expected_sha256=args.expected_quality_preflight_sha256,
        quality_labels=args.quality_labels,
        corpus=args.corpus,
        selector=grid_selector,
        context=grid_context,
        min_resolved_rows=int(resolution_protocol["min_resolved_rows"]),
        min_resolved_lineages=int(resolution_protocol["min_resolved_lineages"]),
    )
    quality_preflight_sha256 = _sha256_file(args.quality_preflight_receipt)
    cell = dict(cells[args.index])
    quality_prior_mode = str(grid["quality_policy_prior"]["mode"])
    model_family = cell["model_family"]
    selection_receipt_sha256: str | None = None
    selection_record_digest: str | None = None
    selection_mode: str | None = None
    gate_receipt_sha256: str | None = None
    gate_record_digest: str | None = None
    gate_profile: str | None = None
    if args.stage == "rl_value":
        if args.selection_receipt is not None:
            if args.selected_simpler is not None or args.diagnostic_manual_selection:
                raise ValueError(
                    "selection receipt cannot be combined with a manual diagnostic choice"
                )
            selected_simpler, selection_receipt_sha256, selection_record_digest = (
                _load_representation_selection(
                    args.selection_receipt,
                    grid=grid,
                    grid_digest=grid_digest,
                    expected_quality_preflight_sha256=quality_preflight_sha256,
                    expected_quality_preflight_record_digest=quality_preflight["record_digest"],
                )
            )
            selection_mode = "validated-receipt"
        elif args.selected_simpler is not None:
            if not args.diagnostic_manual_selection:
                raise ValueError(
                    "manual simpler-family selection requires explicit diagnostic opt-in"
                )
            selected_simpler = args.selected_simpler
            selection_mode = "diagnostic-manual"
        else:
            raise ValueError("RL-value stage requires a representation selection receipt")
        if model_family == "selected-simpler":
            model_family = selected_simpler
    elif (
        args.selection_receipt is not None
        or args.selected_simpler is not None
        or args.diagnostic_manual_selection
    ):
        raise ValueError("representation cells cannot consume an RL-value family selection")
    if args.stage == "representation":
        if args.gate_receipt is None:
            raise ValueError("representation stage requires a passing release-gate receipt")
        if args.expected_gate_receipt_sha256 is None:
            raise ValueError("representation stage requires an externally pinned gate receipt")
        gate_receipt_sha256, gate_record_digest = _load_gate_receipt(
            args.gate_receipt,
            expected_sha256=args.expected_gate_receipt_sha256,
            corpus=args.corpus,
            selector=args.selector,
            context=grid_context,
            require_profile_c=args.require_profile_c_gate,
            expected_grid_manifest_sha256=grid_digest,
            expected_quality_preflight_sha256=quality_preflight_sha256,
            expected_quality_preflight_record_digest=quality_preflight["record_digest"],
        )
        gate_profile = "profile-i+profile-c" if args.require_profile_c_gate else "profile-i"
    elif (
        args.gate_receipt is not None
        or args.expected_gate_receipt_sha256 is not None
        or args.require_profile_c_gate
    ):
        raise ValueError("RL-value cells inherit gates through representation selection")
    seed = int(cell["seed"])
    run_dir = Path(args.run_root) / args.stage / cell["cell_id"]
    resume = (run_dir / "checkpoint.pt").is_file()
    common = {
        "corpus": args.corpus,
        "selector": args.selector,
        "quality_initializer_bank": args.quality_initializer_bank,
        "expected_quality_initializer_bank_manifest_sha256": (
            args.expected_quality_initializer_bank_manifest_sha256
        ),
        "quality_complete_config": args.quality_complete_config,
        "quality_attestation": args.quality_attestation,
        "expected_quality_attestation_digest": (args.expected_quality_attestation_digest),
        "expected_quality_publisher_id": args.expected_quality_publisher_id,
        "ground_certificate_root": args.ground_certificate_root,
        "expected_ground_certificate_root_sha256": (args.expected_ground_certificate_root_sha256),
        "quality_preflight_receipt": args.quality_preflight_receipt,
        "expected_quality_preflight_sha256": args.expected_quality_preflight_sha256,
        "quality_preflight_receipt_sha256": quality_preflight_sha256,
        "quality_preflight_record_digest": quality_preflight["record_digest"],
        "qubit_cap": None,
        "reward_reads": int(grid["environment"]["reward_reads"]),
        "seed": seed,
        "device": args.device,
        "deterministic": args.deterministic,
        "threads": args.threads,
        "out": str(run_dir),
        "resume": resume,
        "quality_prior_mode": quality_prior_mode,
        "grid_manifest_sha256": grid_digest,
        "grid_cell": cell["cell_id"],
        "selection_receipt_sha256": selection_receipt_sha256,
        "selection_record_digest": selection_record_digest,
        "selection_mode": selection_mode,
        "gate_receipt_sha256": gate_receipt_sha256,
        "gate_record_digest": gate_record_digest,
        "gate_profile": gate_profile,
    }
    if args.stage == "representation":
        namespace = argparse.Namespace(
            **common,
            model_family=model_family,
            quality_labels=args.quality_labels,
            epochs=int(cell["epochs"]),
            minibatch=int(cell["minibatch"]),
            learning_rate=float(cell["learning_rate"]),
            weight_decay=float(cell["weight_decay"]),
            min_resolved_rows=int(grid["quality_resolution"]["min_resolved_rows"]),
            min_resolved_lineages=int(grid["quality_resolution"]["min_resolved_lineages"]),
        )
        cmd_warm_start(namespace)
        return

    method = cell["method"]
    warm_start = None
    warm_start_grid_cell = None
    warm_start_checkpoint_payload_digest = None
    if method in {"supervised-only", "ppo-warm-start"}:
        warm_cell = grid["representation_lookup"][model_family][str(seed)]
        warm_start = str(Path(args.run_root) / "representation" / warm_cell / "checkpoint.pt")
        warm_start_grid_cell = warm_cell
        if args.selection_receipt is not None:
            if selection_receipt_sha256 is None or selection_record_digest is None:
                raise RuntimeError("validated representation selection lost its digest binding")
            warm_start_checkpoint_payload_digest = (
                _representation_checkpoint_payload_digest(
                    args.selection_receipt,
                    expected_selection_sha256=selection_receipt_sha256,
                    expected_selection_record_digest=selection_record_digest,
                    grid_cell=warm_cell,
                    model_family=model_family,
                    seed=seed,
                )
            )
    namespace = argparse.Namespace(
        **common,
        model_family=model_family,
        method=method,
        warm_start=warm_start,
        warm_start_grid_cell=warm_start_grid_cell,
        warm_start_checkpoint_payload_digest=warm_start_checkpoint_payload_digest,
        updates=int(cell["updates"]),
        episodes=int(cell["episodes"]),
        ppo_epochs=int(cell["ppo_epochs"]),
        minibatch=int(cell["minibatch"]),
        learning_rate=float(cell["learning_rate"]),
        deployment_initializer_bank=True,
        initializer_bank=args.initializer_bank,
        expected_initializer_bank_manifest_sha256=(args.expected_initializer_bank_manifest_sha256),
        complete_config=args.complete_config,
    )
    cmd_train(namespace)


def _complete_selection_binding(selection, index: int) -> dict[str, object]:
    source_cell_id, seed, selected_source_digest = selection.cell_for_index(index)
    return {
        "schema": "isingfold.complete-system-selected-training",
        "schema_version": 3,
        "selection_receipt_sha256": selection.receipt_sha256,
        "selection_record_digest": selection.record_digest,
        "grid_manifest_sha256": selection.grid_manifest_sha256,
        "selected_model_family": selection.model_family,
        "selected_grid_model_family": selection.grid_model_family,
        "selected_method": selection.method,
        "all_training_seeds": list(selection.training_seeds),
        "all_source_cell_ids": list(selection.cell_ids),
        "all_source_checkpoint_payload_digests": list(selection.checkpoint_payload_digests),
        "runtime_implementation_registry": dict(selection.runtime_implementation_registry),
        "runtime_implementation_digest": selection.runtime_implementation_digest,
        "quality_preflight_receipt_sha256": (selection.quality_preflight_receipt_sha256),
        "quality_preflight_record_digest": selection.quality_preflight_record_digest,
        "source_cell_id": source_cell_id,
        "source_checkpoint_payload_digest": selected_source_digest,
        "training_seed": seed,
        "seed_selection_forbidden": True,
        "selected_validation_checkpoint_reused": False,
        "fresh_representation_stage_required": selection.method
        in {"supervised-only", "ppo-warm-start"},
        "retraining_rule": "fresh-selected-configuration-on-persistent-k2-cache-support",
    }


def cmd_complete_system_train_cell(args: argparse.Namespace) -> None:
    """Retrain one seed of the frozen three-seed configuration on deployment support."""

    from isingfold.rl.experiment_selection import load_rl_value_freeze

    grid, grid_digest = _load_grid(args.grid)
    preflight_context = _context(args, args.corpus)
    preflight_selector = _load_selector_bundle(args.selector, corpus=args.corpus)
    quality_resolution = grid["quality_resolution"]
    quality_preflight = _load_quality_preflight_receipt(
        args.quality_preflight_receipt,
        expected_sha256=args.expected_quality_preflight_sha256,
        quality_labels=args.quality_labels,
        corpus=args.corpus,
        selector=preflight_selector,
        context=preflight_context,
        min_resolved_rows=int(quality_resolution["min_resolved_rows"]),
        min_resolved_lineages=int(quality_resolution["min_resolved_lineages"]),
    )
    quality_preflight_sha256 = _sha256_file(args.quality_preflight_receipt)
    selection = load_rl_value_freeze(
        receipt_path=args.rl_value_selection_receipt,
        grid_path=args.grid,
        expected_sha256=args.expected_selection_sha256,
    )
    from isingfold.rl.checkpoint import runtime_implementation_registry

    current_runtime_registry = runtime_implementation_registry()
    if (
        current_runtime_registry != selection.runtime_implementation_registry
        or stable_digest(current_runtime_registry) != selection.runtime_implementation_digest
    ):
        raise ValueError(
            "complete-system retraining requires the byte-identical model/env/PPO source "
            "and dependency registry frozen across all 27 validation cells"
        )
    if (
        quality_preflight_sha256 != selection.quality_preflight_receipt_sha256
        or quality_preflight["record_digest"] != selection.quality_preflight_record_digest
    ):
        raise ValueError("complete-system retraining uses another quality-preflight pin")
    if selection.grid_manifest_sha256 != grid_digest:
        raise ValueError("complete-system selection and staged grid digests differ")
    source_cell_id, seed, _ = selection.cell_for_index(args.index)
    raw_cells = grid["stages"]["rl_value"]
    matching = [cell for cell in raw_cells if cell.get("cell_id") == source_cell_id]
    if len(matching) != 1:
        raise ValueError("selected complete-system cell is absent from the staged grid")
    cell = dict(matching[0])
    quality_prior_mode = str(grid["quality_policy_prior"]["mode"])
    if (
        cell.get("model_family") != selection.grid_model_family
        or cell.get("method") != selection.method
        or cell.get("seed") != seed
    ):
        raise ValueError("selected complete-system cell changes its frozen configuration")

    warm_start: str | None = None
    warm_cell: str | None = None
    fresh_representation_output = (
        Path(args.run_root) / "complete_system_representation" / source_cell_id
    )
    output = Path(args.run_root) / "complete_system" / source_cell_id
    if output.exists() or (
        selection.method in {"supervised-only", "ppo-warm-start"}
        and fresh_representation_output.exists()
    ):
        raise FileExistsError(
            "complete-system confirmation is fresh-only: use a new run root; resume and "
            "selected-checkpoint reuse are forbidden"
        )
    if selection.method in {"supervised-only", "ppo-warm-start"}:
        warm_cell = grid["representation_lookup"][selection.model_family][str(seed)]
        raw_representation_cells = grid["stages"]["representation"]
        representation_matches = [
            item for item in raw_representation_cells if item.get("cell_id") == warm_cell
        ]
        if len(representation_matches) != 1:
            raise ValueError("selected representation cell is absent from the staged grid")
        representation_cell = dict(representation_matches[0])
        if (
            representation_cell.get("model_family") != selection.model_family
            or representation_cell.get("seed") != seed
        ):
            raise ValueError("selected representation cell changes family or seed")
        warm_namespace = argparse.Namespace(
            corpus=args.corpus,
            selector=args.selector,
            quality_initializer_bank=args.quality_initializer_bank,
            expected_quality_initializer_bank_manifest_sha256=(
                args.expected_quality_initializer_bank_manifest_sha256
            ),
            quality_complete_config=args.quality_complete_config,
            quality_attestation=args.quality_attestation,
            expected_quality_attestation_digest=(args.expected_quality_attestation_digest),
            expected_quality_publisher_id=args.expected_quality_publisher_id,
            ground_certificate_root=args.ground_certificate_root,
            expected_ground_certificate_root_sha256=(args.expected_ground_certificate_root_sha256),
            quality_preflight_receipt=args.quality_preflight_receipt,
            expected_quality_preflight_sha256=args.expected_quality_preflight_sha256,
            quality_preflight_receipt_sha256=quality_preflight_sha256,
            quality_preflight_record_digest=quality_preflight["record_digest"],
            allow_legacy_pilot=False,
            qubit_cap=None,
            quality_labels=args.quality_labels,
            seed=seed,
            device=args.device,
            deterministic=args.deterministic,
            threads=args.threads,
            out=str(fresh_representation_output),
            resume=False,
            grid_manifest_sha256=grid_digest,
            grid_cell=warm_cell,
            selection_receipt_sha256=selection.receipt_sha256,
            selection_record_digest=selection.record_digest,
            selection_mode="fresh-complete-system-representation-v1",
            gate_receipt_sha256=None,
            gate_record_digest=None,
            gate_profile=None,
            model_family=selection.model_family,
            quality_prior_mode=quality_prior_mode,
            epochs=int(representation_cell["epochs"]),
            minibatch=int(representation_cell["minibatch"]),
            learning_rate=float(representation_cell["learning_rate"]),
            weight_decay=float(representation_cell["weight_decay"]),
            min_resolved_rows=int(grid["quality_resolution"]["min_resolved_rows"]),
            min_resolved_lineages=int(grid["quality_resolution"]["min_resolved_lineages"]),
        )
        cmd_warm_start(warm_namespace)
        warm_start = str(fresh_representation_output / "checkpoint.pt")
    binding = _complete_selection_binding(selection, args.index)
    namespace = argparse.Namespace(
        corpus=args.corpus,
        selector=args.selector,
        quality_attestation=args.quality_attestation,
        expected_quality_attestation_digest=(args.expected_quality_attestation_digest),
        expected_quality_publisher_id=args.expected_quality_publisher_id,
        ground_certificate_root=args.ground_certificate_root,
        expected_ground_certificate_root_sha256=(args.expected_ground_certificate_root_sha256),
        quality_preflight_receipt_sha256=quality_preflight_sha256,
        quality_preflight_record_digest=quality_preflight["record_digest"],
        allow_legacy_pilot=False,
        qubit_cap=None,
        reward_reads=int(grid["environment"]["reward_reads"]),
        seed=seed,
        device=args.device,
        deterministic=args.deterministic,
        threads=args.threads,
        out=str(output),
        resume=False,
        grid_manifest_sha256=grid_digest,
        grid_cell=source_cell_id,
        selection_receipt_sha256=selection.receipt_sha256,
        selection_record_digest=selection.record_digest,
        selection_mode=COMPLETE_CACHE_SELECTION_MODE,
        gate_receipt_sha256=None,
        gate_record_digest=None,
        gate_profile=None,
        model_family=selection.model_family,
        quality_prior_mode=quality_prior_mode,
        method=selection.method,
        warm_start=warm_start,
        warm_start_grid_cell=warm_cell,
        warm_start_checkpoint_payload_digest=None,
        updates=int(cell["updates"]),
        episodes=int(cell["episodes"]),
        ppo_epochs=int(cell["ppo_epochs"]),
        minibatch=int(cell["minibatch"]),
        learning_rate=float(cell["learning_rate"]),
        complete_system_no_restart=False,
        deployment_initializer_bank=True,
        complete_selection_binding=binding,
        initializer_bank=args.initializer_bank,
        expected_initializer_bank_manifest_sha256=(args.expected_initializer_bank_manifest_sha256),
        complete_config=args.complete_config,
    )
    cmd_train(namespace)


def _preinitialization_population_tasks(
    prepared: Sequence[PreparedTask],
    *,
    manifest: Mapping[str, object],
    partition: str,
) -> list[Any]:
    """Project candidate-group rows back to the full pre-initialization instance census."""

    if manifest.get("schema_version") != 4 or manifest.get("corpus_scope") != (
        "production-designed-v4"
    ):
        raise ValueError("complete-system evaluation requires production prepared schema v4")
    if partition == "validation":
        partition = "val"
    if partition not in {"val", "test"}:
        raise ValueError("complete-system confirmation accepts validation or test only")
    target_authority = manifest.get("target_authority")
    partition_descriptors = (
        target_authority.get("partitions") if isinstance(target_authority, dict) else None
    )
    descriptor = (
        partition_descriptors.get(partition) if isinstance(partition_descriptors, dict) else None
    )
    if not isinstance(descriptor, dict):
        raise ValueError("prepared manifest has no selected-partition target census")
    if descriptor.get("path") != f"targets/{partition}.jsonl":
        raise ValueError("prepared target descriptor crosses the selected partition")
    expected_instances = descriptor.get("records")
    if (
        isinstance(expected_instances, bool)
        or not isinstance(expected_instances, int)
        or expected_instances <= 0
    ):
        raise ValueError("prepared manifest has an invalid selected-partition target count")

    grouped: dict[str, list[PreparedTask]] = {}
    for item in prepared:
        if (
            item.prepared_schema_version != 4
            or item.corpus_scope != "production-designed-v4"
            or item.provenance is None
            or item.design_condition is None
        ):
            raise ValueError("complete-system population contains a non-production task")
        if item.partition != partition:
            raise ValueError("complete-system population crosses its opened target partition")
        grouped.setdefault(item.instance_id, []).append(item)
    if len(grouped) != expected_instances:
        raise ValueError(
            "prepared candidate groups do not cover the complete pre-initialization "
            "policy-instance denominator"
        )

    population: list[Any] = []
    for instance_id, rows in sorted(grouped.items()):
        partitions = {item.partition for item in rows}
        lineages = {item.task.lineage for item in rows}
        targets = {
            (
                item.task.ground_energy,
                item.reference_status,
                item.certificate_digest,
                item.evaluator_protocol_digest,
            )
            for item in rows
        }
        if len(partitions) != 1 or len(lineages) != 1 or None in lineages or len(targets) != 1:
            raise ValueError(
                f"prepared groups for {instance_id!r} disagree on split, lineage, or target"
            )
        canonical = [
            dataclasses.replace(item.task, name=instance_id, initial_embedding=None)
            for item in rows
        ]
        from isingfold.rl.complete_system import task_population_digest

        identities = {task_population_digest([task]) for task in canonical}
        if len(identities) != 1:
            raise ValueError(f"prepared groups for {instance_id!r} disagree on public task content")
        population.append(canonical[0])
    if not population:
        raise ValueError(f"complete-system partition {partition!r} has no policy instances")
    return population


def _load_complete_evaluation_model(
    args: argparse.Namespace,
    bundle: SelectorBundle,
    selection,
):
    """Load one selected confirmation checkpoint on persistent K=2 cache support."""

    from isingfold.rl.complete_system import COMPLETE_POLICY_RESTART_MODE

    source_cell_id, seed, _ = selection.cell_for_index(args.index)
    checkpoint = Path(args.run_root) / "complete_system" / source_cell_id / "checkpoint.pt"
    model, receipt = _load_evaluation_model(
        argparse.Namespace(checkpoint=str(checkpoint), corpus=args.corpus), bundle
    )
    contract = receipt.get("experiment_contract")
    if not isinstance(contract, dict):
        raise ValueError("complete-system checkpoint has no experiment contract")
    expected = {
        "phase": "rl-value",
        "model_family": selection.model_family,
        "method": selection.method,
        "seed": seed,
        "grid_manifest_sha256": selection.grid_manifest_sha256,
        "grid_cell": source_cell_id,
        "selection_receipt_sha256": selection.receipt_sha256,
        "selection_record_digest": selection.record_digest,
        "selection_mode": COMPLETE_CACHE_SELECTION_MODE,
    }
    if any(contract.get(name) != value for name, value in expected.items()):
        raise ValueError("complete-system checkpoint differs from its selected seed contract")
    hyperparameters = contract.get("hyperparameters")
    expected_binding = _complete_selection_binding(selection, args.index)
    if (
        not isinstance(hyperparameters, dict)
        or hyperparameters.get("complete_system_policy_restart_mode")
        != COMPLETE_POLICY_RESTART_MODE
        or hyperparameters.get("complete_system_selection_binding") != expected_binding
    ):
        raise ValueError("complete-system checkpoint lacks its exact selection/support binding")
    requested_context = _context(args, args.corpus)
    if contract.get("environment") != _context_snapshot(requested_context):
        raise ValueError("complete-system checkpoint was trained under another environment")
    return model, receipt, checkpoint


def _load_pinned_evaluation_bootstrap_bank(
    args: argparse.Namespace,
    *,
    protocol_preset: str,
    config_path: str | os.PathLike[str],
):
    """Authenticate the target-free bank before a caller acquires target access."""

    from isingfold.rl.validation_bootstrap_bank import (
        load_bootstrap_bank,
        load_bootstrap_plan,
    )

    public_args = argparse.Namespace(
        **{
            **vars(args),
            "protocol_preset": protocol_preset,
            "config": str(config_path),
        }
    )
    (
        prepared,
        public_tasks,
        manifest,
        split_registry,
        manifest_sha,
        grid_sha,
        protocol_digest,
        support_digest,
        partition,
        evaluation_seed,
        repetitions,
        initializer,
        runtime,
        config,
        context,
    ) = _bootstrap_bank_public_inputs(public_args)
    bank_root = Path(args.bootstrap_bank)
    plan = load_bootstrap_plan(
        bank_root / "plan.json",
        expected_plan_sha256=args.expected_bootstrap_plan_sha256,
        prepared_tasks=prepared,
        prepared_manifest=manifest,
        prepared_split_registry=split_registry,
        prepared_manifest_sha256=manifest_sha,
        protocol_registry_sha256=grid_sha,
        protocol_record_digest=protocol_digest,
        same_support_contract_digest=support_digest,
        partition=partition,
        evaluation_seed=evaluation_seed,
        repetitions=repetitions,
        initializer=initializer,
        runtime_implementation_manifest=runtime,
        config=config,
        context=context,
    )
    bank = load_bootstrap_bank(
        bank_root,
        plan=plan,
        expected_plan_sha256=args.expected_bootstrap_plan_sha256,
        expected_manifest_sha256=args.expected_bootstrap_manifest_sha256,
    )
    if len(bank) != len(plan.census) or set(public_tasks) != {
        task.instance_id for task in plan.tasks
    }:
        raise ValueError("bootstrap bank does not cover its complete public population")
    return bank, plan


def cmd_evaluate_complete_system_cell(args: argparse.Namespace) -> None:
    """Open one selected seed on the frozen pre-initialization test population."""

    from isingfold.rl.complete_system import (
        CompletePopulationIdentity,
        CompleteSystemConfig,
        FrozenComponentIdentity,
        LACMinorminerInitializerBackend,
        complete_system_metrics,
        complete_system_method_metadata,
        read_complete_system_evidence,
        read_complete_system_outcomes,
        read_complete_system_receipts,
        run_complete_system,
        task_population_digest,
        write_complete_system_evidence,
        write_complete_system_outcomes,
        write_complete_system_receipts,
    )
    from isingfold.rl.evaluate import torch_controller
    from isingfold.rl.experiment_selection import load_rl_value_freeze

    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"complete-system evaluation output already exists: {destination}")
    grid, grid_digest = _load_grid(args.grid)
    evaluation_protocol = grid["complete_system_evaluation"]
    config_path = Path(args.config)
    config_payload = _strict_json(config_path)
    config = CompleteSystemConfig.from_mapping(config_payload)
    _validate_registered_complete_config(
        evaluation_protocol,
        arm="learned",
        semantic_digest=config.digest,
        file_sha256=_sha256_file(config_path),
    )
    partition = str(evaluation_protocol["partition"])
    repetitions = int(evaluation_protocol["repetitions"])
    evaluation_seed = int(evaluation_protocol["evaluation_seed"])
    selection = load_rl_value_freeze(
        receipt_path=args.rl_value_selection_receipt,
        grid_path=args.grid,
        expected_sha256=args.expected_selection_sha256,
    )
    bootstrap_bank, bootstrap_plan = _load_pinned_evaluation_bootstrap_bank(
        args,
        protocol_preset="final-test",
        config_path=config_path,
    )
    source_cell_id, training_seed, _ = selection.cell_for_index(args.index)
    manifest_path = Path(args.corpus) / "manifest.json"
    manifest = _strict_json(manifest_path)
    quality_pin = _quality_attestation_pin(args)
    (
        prepared,
        quality_authority,
        target_access,
        ground_partition_receipt,
    ) = _load_quality_partition(
        args.corpus,
        partition=partition,
        pin=quality_pin,
        role="evaluation_partition",
    )
    tasks = _preinitialization_population_tasks(
        prepared,
        manifest=manifest,
        partition=partition,
    )
    evaluation_strata, confirmatory_design = evaluation_contract_from_prepared(
        prepared,
        corpus_design_receipt=load_prepared_corpus_design(args.corpus),
        partition=partition,
        noninferiority_margin=float(evaluation_protocol["feasibility_noninferiority_margin"]),
    )
    ctx = _context(args, args.corpus)
    if (
        config.audit_reads != ctx.audit_reads
        or config.audit_reads != evaluation_protocol["audit_reads"]
    ):
        raise ValueError("complete-system config, grid and environment audit reads differ")
    bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
    if _bind_selector_quality_authority(bundle, prepared, pin=quality_pin) != quality_authority:
        raise ValueError("selector and complete-system test authorities differ")
    device = _resolve_device(args.device)
    _seed_runtime(evaluation_seed, deterministic=args.deterministic, threads=args.threads)
    model, training_receipt, checkpoint = _load_complete_evaluation_model(args, bundle, selection)
    training_authority = _validate_quality_authority_binding(
        training_receipt["experiment_contract"].get("quality_authority"),
        expected_role="training_partition",
    )
    if training_authority["global"] != quality_authority["global"]:
        raise ValueError("complete-system checkpoint uses a different quality authority")
    model.to(device).eval()
    bundle.model.to(device).eval()

    initializer = LACMinorminerInitializerBackend()
    controller_identity = FrozenComponentIdentity(
        component_id=(
            f"isingfold-policy/{selection.model_family}/{selection.method}/seed-{training_seed}"
        ),
        version=str(training_receipt["checkpoint_payload_digest"]),
        implementation=(
            f"{training_receipt['experiment_contract']['model']['implementation']}:"
            f"runtime-{training_receipt['runtime_implementation_digest']}"
        ),
        artifact_sha256=_sha256_file(checkpoint),
    )
    selector_fit_path = bundle.root / "fit_receipt.json"
    selector_fit = _strict_json(selector_fit_path)
    _verify_record(selector_fit, "selector fit receipt")
    selector_identity = FrozenComponentIdentity(
        component_id="isingfold-if-q3-s0-strength-selector",
        version=str(bundle.model.version),
        implementation=(
            f"{type(bundle.model).__module__}.{type(bundle.model).__qualname__}:"
            f"fit-{selector_fit['record_digest']}"
        ),
        artifact_sha256=_sha256_file(bundle.root / "selector.pt"),
    )
    manifest_sha = _sha256_file(manifest_path)
    population = CompletePopulationIdentity(
        population_id=(
            "preinitialization-"
            + stable_digest(
                {
                    "manifest": manifest_sha,
                    "partition": partition,
                    "evaluation_seed": evaluation_seed,
                    "repetitions": repetitions,
                }
            )
        ),
        source_manifest_sha256=manifest_sha,
        task_payload_sha256=task_population_digest(tasks),
        expected_instances=tuple(sorted((task.lineage or task.name, task.name) for task in tasks)),
        expected_repetitions=repetitions,
        evaluation_strata=evaluation_strata,
        confirmatory_design=confirmatory_design,
        evaluation_seed=evaluation_seed,
    )
    outcomes, receipts = run_complete_system(
        tasks,
        ctx,
        initializer,
        config,
        controller=torch_controller(model, device=device, greedy=False),
        selector=bundle.model,
        controller_identity=controller_identity,
        selector_identity=selector_identity,
        population=population,
        seed=evaluation_seed,
        repetitions=repetitions,
        max_steps=ctx.caps.decisions,
        validation_bootstrap_bank=bootstrap_bank,
        same_support_contract_digest=(bootstrap_plan.same_support_contract_digest),
        bootstrap_consumer_id=(f"complete-system/{source_cell_id}/training-seed-{training_seed}"),
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.complete-", dir=destination.parent)
    )
    try:
        receipts_sha = write_complete_system_receipts(
            temporary / "complete_receipts.jsonl", receipts
        )
        evidence_sha = write_complete_system_evidence(
            temporary / "terminal_evidence.jsonl", receipts
        )
        outcomes_sha = write_complete_system_outcomes(temporary / "outcomes.jsonl", receipts)
        recovered_receipts = read_complete_system_receipts(
            temporary / "complete_receipts.jsonl",
            expected_sha256=receipts_sha,
        )
        recovered_evidence = read_complete_system_evidence(
            temporary / "terminal_evidence.jsonl",
            receipts=recovered_receipts,
            tasks=tasks,
            context=ctx,
            expected_sha256=evidence_sha,
        )
        recovered_outcomes = read_complete_system_outcomes(
            temporary / "outcomes.jsonl",
            receipts=recovered_receipts,
            expected_sha256=outcomes_sha,
        )
        if (
            len(recovered_receipts) != len(receipts)
            or len(recovered_evidence) != len(receipts)
            or len(recovered_outcomes) != len(receipts)
        ):
            raise RuntimeError("complete-system evidence self-verification changed its census")
        from isingfold.rl.checkpoint import runtime_implementation_registry

        runtime_platform = _runtime_platform_identity()
        runtime_registry = runtime_implementation_registry()
        report_payload = {
            "schema": "isingfold.complete-system-evaluation",
            "schema_version": 4,
            "partition": partition,
            "sealed_test_opened": True,
            "evaluation_protocol": evaluation_protocol,
            "deployment_rule": evaluation_protocol["deployment_rule"],
            "training_seed_index": args.index,
            "training_seed": training_seed,
            "source_cell_id": source_cell_id,
            "selection_binding": _complete_selection_binding(selection, args.index),
            "source_corpus_manifest_sha256": manifest_sha,
            "quality_authority": quality_authority,
            "target_access": target_access,
            "ground_partition_receipt": ground_partition_receipt,
            "complete_system_config_sha256": _sha256_file(config_path),
            "bootstrap_bank_access": bootstrap_bank.access_receipt.as_dict(),
            "same_support_contract_digest": (bootstrap_plan.same_support_contract_digest),
            "context": _context_snapshot(ctx),
            "context_digest": stable_digest(_context_snapshot(ctx)),
            "population": population.as_dict(),
            "population_digest": population.digest,
            "controller": controller_identity.as_dict(),
            "selector": selector_identity.as_dict(),
            "initializer": initializer.identity.as_dict(),
            "method": complete_system_method_metadata(
                config,
                initializer.identity,
                controller_identity,
                selector_identity,
                population,
            ),
            "runtime_platform": runtime_platform,
            "inference_device_type": device.type,
            "inference_device_name": _inference_device_name(device, runtime_platform),
            "inference_threads": args.threads,
            "deterministic": args.deterministic,
            "runtime_implementation_registry": runtime_registry,
            "runtime_implementation_digest": content_digest(runtime_registry),
            "repetitions": repetitions,
            "metrics": complete_system_metrics(receipts),
            "artifacts": {
                "complete_receipts": {
                    "path": "complete_receipts.jsonl",
                    "sha256": receipts_sha,
                    "count": len(receipts),
                },
                "terminal_evidence": {
                    "path": "terminal_evidence.jsonl",
                    "sha256": evidence_sha,
                    "count": len(receipts),
                },
                "outcomes": {
                    "path": "outcomes.jsonl",
                    "sha256": outcomes_sha,
                    "count": len(outcomes),
                },
            },
        }
        _atomic_json(temporary / "report.json", _with_digest(report_payload))
        os.rename(temporary, destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(
        json.dumps(
            {
                "cell": source_cell_id,
                "training_seed": training_seed,
                "population": len(tasks),
                "attempts": len(outcomes),
                "metrics": complete_system_metrics(receipts),
                "out": str(destination),
            },
            indent=1,
        )
    )


def _validate_complete_receipt_bank_bindings(
    receipts: Sequence[object],
    *,
    bank_access_record_digest: str,
    same_support_contract_digest: str,
    consumer_id: str,
) -> None:
    """Require every complete receipt to prove the externally pinned bank clone."""

    for receipt in receipts:
        binding = getattr(receipt, "bootstrap_binding", None)
        clone = binding.get("clone") if isinstance(binding, Mapping) else None
        if (
            not isinstance(clone, Mapping)
            or clone.get("consumer_id") != consumer_id
            or clone.get("bank_access_record_digest") != bank_access_record_digest
            or clone.get("same_support_contract_digest") != same_support_contract_digest
        ):
            raise ValueError(
                "complete-system receipt is not bound to the externally pinned bootstrap bank"
            )


def cmd_aggregate_complete_system(args: argparse.Namespace) -> None:
    """Authenticate and aggregate all three frozen complete-system test runs."""

    from isingfold.rl.complete_system import (
        CompleteSystemConfig,
        read_complete_system_evidence,
        read_complete_system_outcomes,
        read_complete_system_receipts,
    )
    from isingfold.rl.complete_system_aggregate import (
        AuthenticatedCompleteSystemSeedRun,
        aggregate_complete_system_seeds,
    )
    from isingfold.rl.experiment_selection import load_rl_value_freeze

    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"complete-system aggregate already exists: {destination}")
    grid, grid_digest = _load_grid(args.grid)
    evaluation_protocol = grid["complete_system_evaluation"]
    config_path = Path(args.config)
    config_payload = _strict_json(config_path)
    config = CompleteSystemConfig.from_mapping(config_payload)
    config_sha = _sha256_file(config_path)
    _validate_registered_complete_config(
        evaluation_protocol,
        arm="learned",
        semantic_digest=config.digest,
        file_sha256=config_sha,
    )
    selection = load_rl_value_freeze(
        receipt_path=args.rl_value_selection_receipt,
        grid_path=args.grid,
        expected_sha256=args.expected_selection_sha256,
    )
    if selection.grid_manifest_sha256 != grid_digest:
        raise ValueError("complete-system selection and staged grid digests differ")

    # Authenticate the complete, target-free support bank before opening any
    # evaluator-only quality targets below.
    bootstrap_bank, bootstrap_plan = _load_pinned_evaluation_bootstrap_bank(
        args,
        protocol_preset="final-test",
        config_path=config_path,
    )

    manifest_path = Path(args.corpus) / "manifest.json"
    manifest = _strict_json(manifest_path)
    manifest_sha = _sha256_file(manifest_path)
    quality_pin = _quality_attestation_pin(args)
    (
        prepared,
        quality_authority,
        target_access,
        ground_partition_receipt,
    ) = _load_quality_partition(
        args.corpus,
        partition=str(evaluation_protocol["partition"]),
        pin=quality_pin,
        role="evaluation_partition",
    )
    tasks = _preinitialization_population_tasks(
        prepared,
        manifest=manifest,
        partition=str(evaluation_protocol["partition"]),
    )
    ctx = _context(args, args.corpus)
    context = _context_snapshot(ctx)
    context_digest = stable_digest(context)
    if (
        config.audit_reads != ctx.audit_reads
        or config.audit_reads != evaluation_protocol["audit_reads"]
    ):
        raise ValueError("complete-system config, grid and environment audit reads differ")

    evaluation_root = Path(args.evaluation_root)
    authenticated_runs = []
    expected_artifacts = {
        "complete_receipts": "complete_receipts.jsonl",
        "terminal_evidence": "terminal_evidence.jsonl",
        "outcomes": "outcomes.jsonl",
    }
    for index in range(3):
        source_cell_id, training_seed, _ = selection.cell_for_index(index)
        run_root = evaluation_root / f"seed-{index}"
        report_path = run_root / "report.json"
        report = _strict_json(report_path)
        _verify_record(report, f"complete-system report {source_cell_id}")
        expected_identity = {
            "schema": "isingfold.complete-system-evaluation",
            "schema_version": 4,
            "partition": evaluation_protocol["partition"],
            "sealed_test_opened": True,
            "evaluation_protocol": evaluation_protocol,
            "deployment_rule": evaluation_protocol["deployment_rule"],
            "training_seed_index": index,
            "training_seed": training_seed,
            "source_cell_id": source_cell_id,
            "selection_binding": _complete_selection_binding(selection, index),
            "source_corpus_manifest_sha256": manifest_sha,
            "quality_authority": quality_authority,
            "target_access": target_access,
            "ground_partition_receipt": ground_partition_receipt,
            "complete_system_config_sha256": config_sha,
            "bootstrap_bank_access": bootstrap_bank.access_receipt.as_dict(),
            "same_support_contract_digest": (bootstrap_plan.same_support_contract_digest),
            "context": context,
            "context_digest": context_digest,
            "repetitions": evaluation_protocol["repetitions"],
        }
        differences = {
            name: {"expected": value, "observed": report.get(name)}
            for name, value in expected_identity.items()
            if report.get(name) != value
        }
        if differences:
            raise ValueError(
                f"complete-system report {source_cell_id} differs from the externally "
                f"pinned experiment: {differences}"
            )
        method = report.get("method")
        if not isinstance(method, dict) or method.get("config") != config_payload:
            raise ValueError(f"complete-system report {source_cell_id} uses another config payload")
        artifacts = report.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != set(expected_artifacts):
            raise ValueError(
                f"complete-system report {source_cell_id} has an incomplete artifact registry"
            )
        for name, filename in expected_artifacts.items():
            metadata = artifacts.get(name)
            if (
                not isinstance(metadata, dict)
                or metadata.get("path") != filename
                or not isinstance(metadata.get("sha256"), str)
                or len(metadata["sha256"]) != 64
            ):
                raise ValueError(
                    f"complete-system report {source_cell_id} has invalid {name} metadata"
                )
        receipts = read_complete_system_receipts(
            run_root / expected_artifacts["complete_receipts"],
            expected_sha256=artifacts["complete_receipts"]["sha256"],
        )
        _validate_complete_receipt_bank_bindings(
            receipts,
            bank_access_record_digest=bootstrap_bank.access_receipt.record_digest,
            same_support_contract_digest=(bootstrap_plan.same_support_contract_digest),
            consumer_id=(f"complete-system/{source_cell_id}/training-seed-{training_seed}"),
        )
        evidence = read_complete_system_evidence(
            run_root / expected_artifacts["terminal_evidence"],
            receipts=receipts,
            tasks=tasks,
            context=ctx,
            expected_sha256=artifacts["terminal_evidence"]["sha256"],
        )
        outcomes = read_complete_system_outcomes(
            run_root / expected_artifacts["outcomes"],
            receipts=receipts,
            expected_sha256=artifacts["outcomes"]["sha256"],
        )
        if len(evidence) != len(receipts) or len(outcomes) != len(receipts):
            raise ValueError(f"complete-system report {source_cell_id} changes its artifact census")
        run = AuthenticatedCompleteSystemSeedRun(report, receipts, evidence)
        if run.report_file_sha256 != _sha256_file(report_path):
            raise ValueError(f"complete-system report {source_cell_id} is not canonical JSON")
        authenticated_runs.append(run)

    aggregate = aggregate_complete_system_seeds(authenticated_runs)
    _atomic_json(destination, aggregate)
    if _strict_json(destination) != aggregate:
        raise RuntimeError("complete-system aggregate failed its write/read self-check")
    print(
        json.dumps(
            {
                "training_seeds": aggregate["training_seeds"],
                "independent_lineages": aggregate["independent_lineages"],
                "unconditional_utility_mean": aggregate["unconditional_utility_mean"],
                "confidence_interval": aggregate["unconditional_utility_confidence_interval"],
                "out": str(destination),
            },
            indent=1,
        )
    )


def cmd_aggregate_learned_vs_stock(args: argparse.Namespace) -> None:
    """Replay evidence and aggregate three row-matched learned/stock runs."""

    from isingfold.rl.complete_system import (
        CompleteSystemConfig,
        read_complete_system_evidence,
        read_complete_system_outcomes,
        read_complete_system_receipts,
    )
    from isingfold.rl.complete_system_aggregate import AuthenticatedCompleteSystemSeedRun
    from isingfold.rl.external_pairing import (
        ExternalCompleteSystemConfig,
        aggregate_learned_vs_stock,
        load_authenticated_external_complete_run,
    )
    from isingfold.rl.external_tuning import (
        ExternalTuningExecutionBinding,
        load_external_tuning_registry,
        load_external_tuning_selection,
    )

    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"paired complete-system aggregate already exists: {destination}")
    learned_roots = tuple(Path(path) for path in args.learned_evaluation)
    learned_pins = tuple(args.expected_learned_report_sha256)
    stock_roots = tuple(Path(path) for path in args.external_evaluation)
    stock_pins = tuple(args.expected_external_report_sha256)
    if len(learned_roots) != 3 or len(learned_pins) != 3:
        raise ValueError(
            "paired complete-system aggregation requires exactly three learned roots and pins"
        )
    if len(stock_roots) != 3 or len(stock_pins) != 3:
        raise ValueError(
            "paired complete-system aggregation requires exactly three stock roots and pins"
        )
    grid, grid_digest = _load_grid(args.grid)
    protocol = grid["complete_system_evaluation"]
    learned_config_path = Path(args.learned_config)
    learned_config = CompleteSystemConfig.from_mapping(_strict_json(learned_config_path))
    external_config_path = Path(args.external_config)
    external_config = ExternalCompleteSystemConfig.from_mapping(_strict_json(external_config_path))
    _validate_registered_complete_config(
        protocol,
        arm="learned",
        semantic_digest=learned_config.digest,
        file_sha256=_sha256_file(learned_config_path),
    )
    _validate_registered_complete_config(
        protocol,
        arm="external",
        semantic_digest=external_config.digest,
        file_sha256=_sha256_file(external_config_path),
    )
    tuning_registry = load_external_tuning_registry(
        args.tuning_registry,
        expected_file_sha256=args.expected_tuning_registry_sha256,
        grid_path=args.grid,
        external_config_path=args.external_config,
    )
    tuning_selection = load_external_tuning_selection(
        args.external_tuning_selection,
        expected_file_sha256=args.expected_external_tuning_selection_sha256,
        registry=tuning_registry,
    )
    expected_tuning_execution = ExternalTuningExecutionBinding.for_deployment(tuning_selection)

    manifest_path = Path(args.corpus) / "manifest.json"
    manifest = _strict_json(manifest_path)
    manifest_sha = _sha256_file(manifest_path)
    if manifest_sha != tuning_selection.source_manifest_sha256:
        raise ValueError("paired aggregate corpus differs from baseline tuning")
    quality_pin = _quality_attestation_pin(args)
    (
        prepared,
        quality_authority,
        target_access,
        ground_partition_receipt,
    ) = _load_quality_partition(
        args.corpus,
        partition=str(protocol["partition"]),
        pin=quality_pin,
        role="evaluation_partition",
    )
    tasks = _preinitialization_population_tasks(
        prepared,
        manifest=manifest,
        partition=str(protocol["partition"]),
    )
    ctx = _context(args, args.corpus)
    external_config.validate_symmetric_envelope(learned_config, ctx)
    if stable_digest(_context_snapshot(ctx)) != tuning_selection.context_digest:
        raise ValueError("paired aggregate context differs from baseline tuning")
    tuning_quality_authority = _validate_quality_authority_binding(
        tuning_selection.quality_authority,
        expected_role="evaluation_partition",
    )
    if tuning_quality_authority["global"] != quality_authority["global"]:
        raise ValueError("paired aggregate quality authority differs from baseline tuning")
    expected_report_identity = {
        "evaluation_protocol": protocol,
        "source_corpus_manifest_sha256": manifest_sha,
        "quality_authority": quality_authority,
        "target_access": target_access,
        "ground_partition_receipt": ground_partition_receipt,
        "context": _context_snapshot(ctx),
        "context_digest": stable_digest(_context_snapshot(ctx)),
        "complete_system_config_sha256": _sha256_file(learned_config_path),
    }
    authenticated = []
    for root, expected_report_sha in zip(learned_roots, learned_pins, strict=True):
        if not isinstance(expected_report_sha, str) or len(expected_report_sha) != 64:
            raise ValueError("learned report pin must be a SHA-256 digest")
        report_path = root / "report.json"
        if _sha256_file(report_path) != expected_report_sha:
            raise ValueError(f"learned report SHA-256 differs from its pin: {root}")
        report = _strict_json(report_path)
        if canonical_json_bytes(report) + b"\n" != report_path.read_bytes():
            raise ValueError(f"learned report is not canonical JSON: {root}")
        _verify_record(report, f"learned complete-system report {root.name}")
        artifacts = report.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != {
            "complete_receipts",
            "terminal_evidence",
            "outcomes",
        }:
            raise ValueError(f"learned report has an incomplete artifact registry: {root}")
        receipt_entry = artifacts["complete_receipts"]
        evidence_entry = artifacts["terminal_evidence"]
        outcome_entry = artifacts["outcomes"]
        if (
            not isinstance(receipt_entry, dict)
            or receipt_entry.get("path") != "complete_receipts.jsonl"
            or not isinstance(outcome_entry, dict)
            or outcome_entry.get("path") != "outcomes.jsonl"
            or not isinstance(evidence_entry, dict)
            or evidence_entry.get("path") != "terminal_evidence.jsonl"
        ):
            raise ValueError(f"learned report changes registered artifact paths: {root}")
        receipts = read_complete_system_receipts(
            root / "complete_receipts.jsonl",
            expected_sha256=receipt_entry.get("sha256"),
        )
        outcomes = read_complete_system_outcomes(
            root / "outcomes.jsonl",
            receipts=receipts,
            expected_sha256=outcome_entry.get("sha256"),
        )
        evidence = read_complete_system_evidence(
            root / "terminal_evidence.jsonl",
            receipts=receipts,
            tasks=tasks,
            context=ctx,
            expected_sha256=evidence_entry.get("sha256"),
        )
        if len(outcomes) != len(receipts) or len(evidence) != len(receipts):
            raise ValueError(f"learned outcome and receipt censuses differ: {root}")
        differences = {
            name: {"expected": value, "observed": report.get(name)}
            for name, value in expected_report_identity.items()
            if report.get(name) != value
        }
        if differences:
            raise ValueError(
                f"learned report differs from authenticated corpus/config: {differences}"
            )
        run = AuthenticatedCompleteSystemSeedRun(report, tuple(receipts), tuple(evidence))
        if run.report_file_sha256 != expected_report_sha:
            raise ValueError(f"learned report file identity is noncanonical: {root}")
        authenticated.append(run)

    stock = tuple(
        load_authenticated_external_complete_run(
            root,
            expected_report_sha256=pin,
            tasks=tasks,
            context=ctx,
        )
        for root, pin in zip(stock_roots, stock_pins, strict=True)
    )
    if any(run.receipts[0].tuning_execution != expected_tuning_execution for run in stock):
        raise ValueError("stock reports do not use the frozen validation-selected strategy")
    if any(
        content_digest(run.report["selector"]) != tuning_selection.selector_digest
        for run in (*authenticated, *stock)
    ):
        raise ValueError("paired reports use another baseline-tuning selector")
    if any(
        run.report["runtime_implementation_digest"]
        != tuning_selection.runtime_implementation_digest
        for run in (*authenticated, *stock)
    ):
        raise ValueError("paired reports use runtime sources different from baseline tuning")
    aggregate = aggregate_learned_vs_stock(authenticated, stock)
    _atomic_json(destination, aggregate)
    if _strict_json(destination) != aggregate:
        raise RuntimeError("paired complete-system aggregate failed its write/read self-check")
    print(
        json.dumps(
            {
                "training_seeds": aggregate["training_seeds"],
                "independent_lineages": aggregate["independent_lineages"],
                "utility_difference": aggregate["unconditional_utility_difference"],
                "utility_confidence_interval": aggregate[
                    "unconditional_utility_difference_confidence_interval"
                ],
                "feasibility_noninferiority": aggregate["feasibility_noninferiority"],
                "out": str(destination),
            },
            indent=1,
        )
    )


def cmd_representation_eval_cell(args: argparse.Namespace) -> None:
    """Create one of the nine preregistered validation reports used for selection."""

    grid, grid_digest = _load_grid(args.grid)
    cells = grid["stages"]["representation"]
    if not 0 <= args.index < len(cells):
        raise ValueError("representation evaluation index is outside the nine-cell registry")
    cell = cells[args.index]
    protocol = grid["representation_evaluation"]
    namespace = argparse.Namespace(
        corpus=args.corpus,
        selector=args.selector,
        quality_attestation=args.quality_attestation,
        expected_quality_attestation_digest=(args.expected_quality_attestation_digest),
        expected_quality_publisher_id=args.expected_quality_publisher_id,
        ground_certificate_root=args.ground_certificate_root,
        expected_ground_certificate_root_sha256=(args.expected_ground_certificate_root_sha256),
        qubit_cap=None,
        checkpoint=str(Path(args.run_root) / "representation" / cell["cell_id"] / "checkpoint.pt"),
        out=str(Path(args.evaluations_root) / cell["cell_id"]),
        partition=protocol["partition"],
        repetitions=int(protocol["repetitions"]),
        audit_reads=int(protocol["audit_reads"]),
        margin=float(protocol["margin"]),
        greedy=False,
        seed=int(protocol["evaluation_seed"]),
        device=args.device,
        threads=args.threads,
        deterministic=args.deterministic,
        deployment_policy_context=False,
        bootstrap_bank=args.bootstrap_bank,
        expected_bootstrap_plan_sha256=args.expected_bootstrap_plan_sha256,
        expected_bootstrap_manifest_sha256=(args.expected_bootstrap_manifest_sha256),
        complete_config=args.complete_config,
        scientific_evaluation_authorization={
            "stage": "representation",
            "grid_manifest_sha256": grid_digest,
            "grid_cell": cell["cell_id"],
            "partition": "validation",
        },
    )
    cmd_evaluate(namespace)


def cmd_select_rl_value(args: argparse.Namespace) -> None:
    """Freeze the complete validation-only decision over the eighteen RL-value cells."""

    from isingfold.rl.experiment_selection import freeze_rl_value_experiment

    receipt = freeze_rl_value_experiment(
        grid_path=args.grid,
        representation_receipt_path=args.representation_selection_receipt,
        evaluations_root=args.evaluations_root,
        output_path=args.out,
    )
    print(json.dumps(receipt, indent=1))


def _validate_rl_value_run_for_evaluation(
    checkpoint: Path,
    *,
    expected_cell: Mapping[str, object],
    expected_model_family: str,
    grid_digest: str,
    selection_receipt_sha256: str,
    selection_record_digest: str,
) -> dict[str, Any]:
    """Bind one completed RL-value checkpoint to its registered cell and family freeze."""

    if not checkpoint.is_file():
        raise FileNotFoundError(f"registered RL-value checkpoint is missing: {checkpoint}")
    run_path = checkpoint.parent / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"registered RL-value run receipt is missing: {run_path}")
    receipt = _strict_json(run_path)
    _verify_record(receipt, "RL-value training-run receipt")
    if (
        receipt.get("schema") != RUN_SCHEMA
        or receipt.get("schema_version") != RUN_SCHEMA_VERSION
        or receipt.get("phase") != "rl-value"
        or receipt.get("complete") is not True
        or receipt.get("model_family") != expected_model_family
        or receipt.get("method") != expected_cell.get("method")
    ):
        raise ValueError("checkpoint is not the completed registered RL-value training run")
    contract = receipt.get("experiment_contract")
    if not isinstance(contract, dict):
        raise ValueError("RL-value training run has no experiment contract")
    expected_contract = {
        "phase": "rl-value",
        "model_family": expected_model_family,
        "method": expected_cell.get("method"),
        "seed": expected_cell.get("seed"),
        "grid_manifest_sha256": grid_digest,
        "grid_cell": expected_cell.get("cell_id"),
        "selection_mode": "validated-receipt",
    }
    observed_contract = {field: contract.get(field) for field in expected_contract}
    if observed_contract != expected_contract:
        raise ValueError("RL-value checkpoint differs from its registered grid cell")
    if (
        contract.get("selection_receipt_sha256") != selection_receipt_sha256
        or contract.get("selection_record_digest") != selection_record_digest
    ):
        raise ValueError("RL-value checkpoint belongs to another representation selection")
    _require_current_ppo_collection_schedule(contract)
    _require_current_ppo_update_control(contract)
    checkpoint_payload_digest = receipt.get("checkpoint_payload_digest")
    if (
        not isinstance(checkpoint_payload_digest, str)
        or len(checkpoint_payload_digest) != 64
        or any(character not in "0123456789abcdef" for character in checkpoint_payload_digest)
    ):
        raise ValueError("RL-value training run has no valid checkpoint payload digest")
    return receipt


def cmd_rl_value_eval_cell(args: argparse.Namespace) -> None:
    """Evaluate one registered RL-value cell on validation without scientific overrides."""

    grid, grid_digest = _load_grid(args.grid)
    cells = grid["stages"]["rl_value"]
    if not 0 <= args.index < len(cells):
        raise ValueError("RL-value evaluation index is outside the eighteen-cell registry")
    cell = cells[args.index]
    selected_simpler, selection_sha256, selection_record_digest = _load_representation_selection(
        args.representation_selection_receipt,
        grid=grid,
        grid_digest=grid_digest,
    )
    grid_family = cell["model_family"]
    model_family = selected_simpler if grid_family == "selected-simpler" else grid_family
    checkpoint = Path(args.run_root) / "rl_value" / cell["cell_id"] / "checkpoint.pt"
    _validate_rl_value_run_for_evaluation(
        checkpoint,
        expected_cell=cell,
        expected_model_family=model_family,
        grid_digest=grid_digest,
        selection_receipt_sha256=selection_sha256,
        selection_record_digest=selection_record_digest,
    )
    protocol = grid["rl_value_evaluation"]
    if protocol["partition"] != "validation":  # Defense after strict grid loading.
        raise ValueError("RL-value model selection can evaluate validation only")
    if protocol["deployment_rule"] != "categorical-temperature-one":
        raise ValueError("RL-value model selection forbids greedy deployment")
    namespace = argparse.Namespace(
        corpus=args.corpus,
        selector=args.selector,
        quality_attestation=args.quality_attestation,
        expected_quality_attestation_digest=(args.expected_quality_attestation_digest),
        expected_quality_publisher_id=args.expected_quality_publisher_id,
        ground_certificate_root=args.ground_certificate_root,
        expected_ground_certificate_root_sha256=(args.expected_ground_certificate_root_sha256),
        qubit_cap=None,
        checkpoint=str(checkpoint),
        out=str(Path(args.evaluations_root) / cell["cell_id"]),
        partition="validation",
        repetitions=int(protocol["repetitions"]),
        audit_reads=int(protocol["audit_reads"]),
        margin=float(protocol["margin"]),
        greedy=False,
        seed=int(protocol["evaluation_seed"]),
        device=args.device,
        threads=args.threads,
        deterministic=True,
        deployment_policy_context=False,
        bootstrap_bank=args.bootstrap_bank,
        expected_bootstrap_plan_sha256=args.expected_bootstrap_plan_sha256,
        expected_bootstrap_manifest_sha256=(args.expected_bootstrap_manifest_sha256),
        complete_config=args.complete_config,
        scientific_evaluation_authorization={
            "stage": "rl-value",
            "grid_manifest_sha256": grid_digest,
            "grid_cell": cell["cell_id"],
            "partition": "validation",
        },
    )
    cmd_evaluate(namespace)


def cmd_plan_final_strength_audit(args: argparse.Namespace) -> None:
    """Seal the outcome-blind final-strength subset before test targets are opened."""

    from isingfold.rl.final_strength_cli import plan_final_strength_audit

    plan_final_strength_audit(args)


def cmd_seal_final_strength_audit_execution(args: argparse.Namespace) -> None:
    """Authenticate test targets and all six frozen source runs, then seal execution."""

    from isingfold.rl.final_strength_cli import seal_final_strength_audit_execution

    seal_final_strength_audit_execution(args)


def cmd_run_final_strength_audit_shard(args: argparse.Namespace) -> None:
    """Run one deterministic shard under the sealed plan and execution manifest."""

    from isingfold.rl.final_strength_cli import run_final_strength_audit_shard_command

    run_final_strength_audit_shard_command(args)


def cmd_merge_final_strength_audit(args: argparse.Namespace) -> None:
    """Authenticate every audit shard and publish one canonical final receipt."""

    from isingfold.rl.final_strength_audit import (
        load_final_strength_audit_execution_manifest,
        load_final_strength_audit_plan,
        load_final_strength_audit_shard,
        merge_final_strength_audit_shards,
        publish_final_strength_audit,
    )

    shard_directories = tuple(args.shard)
    shard_pins = tuple(args.expected_shard_receipt_sha256)
    if len(shard_directories) != len(shard_pins):
        raise ValueError(
            "final-strength merge requires one external receipt SHA-256 per shard directory"
        )
    sealed_plan = load_final_strength_audit_plan(
        args.plan,
        expected_sha256=args.expected_plan_sha256,
    )
    if len(shard_directories) != sealed_plan.plan.shard_count:
        raise ValueError(
            "final-strength merge requires the complete registered shard census: "
            f"expected={sealed_plan.plan.shard_count}, observed={len(shard_directories)}"
        )
    sealed_execution = load_final_strength_audit_execution_manifest(
        args.execution_manifest,
        expected_sha256=args.expected_execution_manifest_sha256,
        sealed_plan=sealed_plan,
    )
    shards = tuple(
        load_final_strength_audit_shard(root, expected_receipt_sha256=pin)
        for root, pin in zip(shard_directories, shard_pins, strict=True)
    )
    result = merge_final_strength_audit_shards(
        sealed_plan,
        sealed_execution,
        shards,
    )
    receipt = publish_final_strength_audit(
        args.out,
        sealed_plan=sealed_plan,
        sealed_execution_manifest=sealed_execution,
        result=result,
    )
    receipt_path = Path(args.out) / "receipt.json"
    print(
        json.dumps(
            {
                "publication_eligible": receipt["publication_eligible"],
                "rows": receipt["raw_rows"]["count"],
                "record_digest": receipt["record_digest"],
                "receipt_file_sha256": _sha256_file(receipt_path),
                "out": str(Path(args.out)),
            },
            indent=1,
        )
    )


def _selector_parity_device_identity(device: object, *, requested: str) -> dict[str, object]:
    """Record the exact CPU or CUDA device used by one parity pass."""

    import torch

    device_type = getattr(device, "type", None)
    if requested == "cpu":
        if device_type != "cpu":
            raise ValueError("selector parity CPU device did not resolve to CPU")
        platform_identity = _runtime_platform_identity()
        return {
            "capability": None,
            "index": None,
            "name": str(platform_identity["processor"]),
            "requested": requested,
            "type": "cpu",
        }
    if requested != "cuda" or device_type != "cuda":
        raise ValueError("selector parity accelerator device must resolve to CUDA")
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return {
        "capability": list(torch.cuda.get_device_capability(index)),
        "index": int(index),
        "name": str(torch.cuda.get_device_name(index)),
        "requested": requested,
        "type": "cuda",
    }


def _selector_parity_runtime_identity(args: argparse.Namespace) -> dict[str, object]:
    """Bind source, dependency, execution-runtime and host identities for parity."""

    import torch
    import isingfold.rl.data.quality_selector_device_parity as parity_module
    import isingfold.rl.program as program_module
    import isingfold.rl.strength as selector_module
    import isingfold.rl.strength_tensorize as tensorizer_module

    def source_sha256(module: object, label: str) -> str:
        path = getattr(module, "__file__", None)
        if not isinstance(path, str) or not Path(path).is_file():
            raise RuntimeError(f"selector parity cannot identify {label} source")
        return _sha256_file(path)

    cuda_version = torch.version.cuda
    if not isinstance(cuda_version, str) or not cuda_version:
        raise RuntimeError("selector parity CUDA runtime version is unavailable")
    return {
        "cuda_runtime_version": cuda_version,
        "deterministic_algorithms": bool(args.deterministic),
        "execution_runtime_sha256": _require_lower_sha256(
            args.execution_runtime_sha256,
            "selector parity execution runtime SHA-256",
        ),
        "implementation": {
            "parity_module_sha256": source_sha256(parity_module, "parity"),
            "program_module_sha256": source_sha256(program_module, "program compiler"),
            "selector_module_sha256": source_sha256(selector_module, "selector"),
            "tensorizer_module_sha256": source_sha256(tensorizer_module, "tensorizer"),
        },
        "platform": _runtime_platform_identity(),
        "python_version": platform.python_version(),
        "threads": int(args.threads),
        "torch_version": str(torch.__version__),
    }


def cmd_audit_quality_selector_device_parity(args: argparse.Namespace) -> None:
    """Audit the frozen selector on every target-free all-train quality state."""

    from isingfold.rl.data.prepared import load_prepared_partition
    from isingfold.rl.data.quality_selector_device_parity import (
        build_quality_selector_device_parity,
        load_quality_selector_parity_config,
        publish_quality_selector_device_parity,
    )
    from isingfold.rl.data.quality_resolution_plan import (
        materialize_resolution_production_plan,
    )

    if args.deterministic is not True or args.threads != 1:
        raise ValueError("selector parity requires deterministic one-thread execution")
    config, config_sha256, config_digest, config_record = (
        load_quality_selector_parity_config(
            args.quality_protocol_config,
            expected_sha256=args.expected_quality_protocol_config_sha256,
        )
    )
    public = load_prepared_partition(
        args.corpus,
        partition="train",
        include_evaluator=False,
    )
    if public.target_access is not None:
        raise RuntimeError("selector parity unexpectedly opened evaluator targets")
    context = _context(args, args.corpus)
    bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
    initializer_bank, initializer_bank_sha256 = _load_quality_initializer_bank(
        args,
        public_tasks=public.tasks,
        context=context,
    )
    _seed_runtime(config.quality_seed, deterministic=True, threads=1)
    cpu_device = _resolve_device(args.cpu_device)
    accelerator_device = _resolve_device(args.accelerator_device)
    cpu_identity = _selector_parity_device_identity(cpu_device, requested=args.cpu_device)
    accelerator_identity = _selector_parity_device_identity(
        accelerator_device,
        requested=args.accelerator_device,
    )
    bundle.model.to(cpu_device).eval()
    production = materialize_resolution_production_plan(
        public.tasks,
        context=context,
        selector=bundle.model,
        config=config,
        source_corpus_manifest_sha256=_corpus_manifest_digest(args.corpus),
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=initializer_bank_sha256,
    )
    production_record = production.as_dict()
    source_census = production_record["source_census"]
    if not isinstance(source_census, Mapping):
        raise RuntimeError("selector parity production plan lost its public census")
    manifest = _strict_json(Path(args.corpus) / "manifest.json")
    _verify_record(manifest, "selector parity prepared manifest")
    if manifest.get("schema_version") != 4:
        raise ValueError("selector parity requires prepared schema v4")
    fit_path = bundle.root / "fit_receipt.json"
    fit_receipt = _strict_json(fit_path)
    _verify_record(fit_receipt, "selector parity fit receipt")
    receipt = build_quality_selector_device_parity(
        public.tasks,
        production,
        context=context,
        selector=bundle.model,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=initializer_bank_sha256,
        cpu_device=cpu_device,
        accelerator_device=accelerator_device,
        corpus={
            "manifest_record_digest": _require_lower_sha256(
                manifest.get("record_digest"), "prepared manifest record digest"
            ),
            "manifest_sha256": _corpus_manifest_digest(args.corpus),
            "train_census_record_digest": source_census["record_digest"],
            "train_lineage_count": source_census["lineage_count"],
            "train_task_count": source_census["task_count"],
        },
        selector_identity={
            "fit_receipt_record_digest": _require_lower_sha256(
                fit_receipt.get("record_digest"), "selector fit receipt record digest"
            ),
            "fit_receipt_sha256": _sha256_file(fit_path),
            "normalizer_digest": bundle.normalizer_digest,
            "normalizer_file_sha256": _sha256_file(bundle.root / "normalizer.json"),
            "selector_digest": bundle.selector_digest,
            "selector_file_sha256": _sha256_file(bundle.root / "selector.pt"),
        },
        quality_protocol={
            "config": config_record,
            "config_digest": config_digest,
            "config_sha256": config_sha256,
            "production_plan_digest": production_record["record_digest"],
            "production_row_count": production_record["row_count"],
        },
        context_identity={
            "digest": content_digest(_context_snapshot(context)),
            "snapshot": _context_snapshot(context),
        },
        initializer_bank_identity=production_record["initializer_bank"],
        runtime=_selector_parity_runtime_identity(args),
        cpu_device_identity=cpu_identity,
        accelerator_device_identity=accelerator_identity,
    )
    raw_sha256 = publish_quality_selector_device_parity(args.out, receipt)
    print(
        json.dumps(
            {
                "all_equal": receipt["all_equal"],
                "mismatch_count": receipt["mismatch_count"],
                "receipt_sha256": raw_sha256,
                "record_digest": receipt["record_digest"],
                "out": str(Path(args.out)),
            },
            indent=1,
        )
    )


def _load_quality_selector_device_parity(
    path: str | os.PathLike[str],
    expected_sha256: str,
    *,
    device: str,
) -> tuple[dict[str, Any], str]:
    """Authenticate the externally pinned CPU/CUDA selector-parity decision."""
    from isingfold.rl.data.quality_selector_device_parity import (
        load_quality_selector_device_parity,
    )

    expected = _require_lower_sha256(expected_sha256, "quality selector-parity receipt pin")
    record = load_quality_selector_device_parity(path, expected_sha256=expected)
    observed = expected
    mismatch_count = record.get("mismatch_count")
    all_equal = record.get("all_equal")
    if (
        isinstance(mismatch_count, bool)
        or not isinstance(mismatch_count, int)
        or mismatch_count < 0
        or not isinstance(all_equal, bool)
        or all_equal != (mismatch_count == 0)
    ):
        raise ValueError("quality selector-parity result is inconsistent")
    if device not in {"cpu", "cuda"}:
        raise ValueError("quality selector device must be cpu or cuda")
    if device == "cpu" and (not all_equal or mismatch_count != 0):
        raise ValueError("CPU quality inference requires exact CPU/CUDA selector parity")
    return record, observed


def _load_pinned_quality_resolution_plan(path: str | os.PathLike[str], expected_sha256: str):
    """Load a resolution plan through its external raw pin and embedded upstream pins."""

    from isingfold.rl.data.quality_resolution_plan import load_quality_resolution_plan

    source = Path(path)
    if source.is_symlink():
        raise ValueError("quality-resolution plan may not be a symbolic link")
    if not source.is_file():
        raise FileNotFoundError(f"quality-resolution plan is missing: {source}")
    expected = _require_lower_sha256(expected_sha256, "quality-resolution plan pin")
    raw = source.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(observed, expected):
        raise ValueError("quality-resolution plan differs from its out-of-band pin")
    record = _strict_json_object(raw, "quality-resolution plan")
    if raw != canonical_json_bytes(record) + b"\n":
        raise ValueError("quality-resolution plan must be canonical JSON followed by one newline")

    sections: dict[str, Mapping[str, object]] = {}
    for name in ("config", "grid", "prepared_corpus", "publisher", "ground_root", "selector"):
        section = record.get(name)
        if not isinstance(section, Mapping):
            raise ValueError(f"quality-resolution plan {name} section is malformed")
        sections[name] = section
    return load_quality_resolution_plan(
        source,
        expected_plan_sha256=expected,
        expected_config_sha256=_require_lower_sha256(
            sections["config"].get("raw_sha256"),
            "quality-resolution plan config pin",
        ),
        expected_grid_sha256=_require_lower_sha256(
            sections["grid"].get("raw_sha256"),
            "quality-resolution plan grid pin",
        ),
        expected_prepared_manifest_sha256=_require_lower_sha256(
            sections["prepared_corpus"].get("manifest_sha256"),
            "quality-resolution plan prepared-manifest pin",
        ),
        expected_prepared_train_census_record_digest=_require_lower_sha256(
            sections["prepared_corpus"].get("train_census_record_digest"),
            "quality-resolution plan train-census digest",
        ),
        expected_publisher_attestation_sha256=_require_lower_sha256(
            sections["publisher"].get("attestation_sha256"),
            "quality-resolution plan publisher-attestation pin",
        ),
        expected_ground_root_sha256=_require_lower_sha256(
            sections["ground_root"].get("sha256"),
            "quality-resolution plan ground-root pin",
        ),
        expected_selector_file_sha256=_require_lower_sha256(
            sections["selector"].get("selector_file_sha256"),
            "quality-resolution plan selector-file pin",
        ),
        expected_selector_device_parity_sha256=_require_lower_sha256(
            sections["selector"].get("device_parity_sha256"),
            "quality-resolution plan selector-parity pin",
        ),
    )


def _quality_resolution_source_sha256(module: object, label: str) -> str:
    source_path = getattr(module, "__file__", None)
    if not isinstance(source_path, str) or not source_path:
        raise RuntimeError(f"{label} has no concrete source file")
    source = Path(source_path)
    if source.is_symlink() or not source.is_file():
        raise RuntimeError(f"{label} source is not an immutable regular file")
    return _sha256_file(source)


def cmd_publish_quality_resolution_verifier_identity(args: argparse.Namespace) -> None:
    """Publish the pinned independent replay-verifier identity for shard checks."""

    from isingfold.rl.data.quality_resolution_delta import (
        publish_quality_resolution_verifier_identity,
    )

    plan = _load_pinned_quality_resolution_plan(args.plan, args.expected_plan_sha256)
    raw_sha256 = publish_quality_resolution_verifier_identity(
        plan,
        verification_runtime_sha256=args.verification_runtime_sha256,
        attestor_id=args.attestor_id,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "out": str(Path(args.out)),
                "verifier_identity_sha256": raw_sha256,
            },
            indent=1,
        )
    )


def _load_quality_initializer_bank(
    args: argparse.Namespace,
    *,
    public_tasks: Sequence[PreparedTask],
    context: Context,
    bank_attribute: str = "initializer_bank",
    pin_attribute: str = "expected_initializer_bank_manifest_sha256",
    config_attribute: str = "complete_config",
):
    """Load the externally pinned target-free K=2 bank used by quality workflows."""

    from isingfold.rl.complete_system import (
        CompleteSystemConfig,
        LACMinorminerInitializerBackend,
    )
    from isingfold.rl.initializer_bank import (
        lac_runtime_implementation_manifest,
        load_initializer_bank,
    )

    bank_path = getattr(args, bank_attribute, None)
    manifest_pin = getattr(args, pin_attribute, None)
    complete_config_path = getattr(args, config_attribute, None)
    if not all(
        isinstance(value, str) and value
        for value in (bank_path, manifest_pin, complete_config_path)
    ):
        raise ValueError(
            "quality publication requires an initializer bank, its external manifest "
            "SHA-256 pin, and the complete-system config"
        )
    manifest_pin = _require_lower_sha256(manifest_pin, "quality initializer-bank manifest pin")
    complete_config = CompleteSystemConfig.from_mapping(_strict_json(complete_config_path))
    initializer = LACMinorminerInitializerBackend()
    runtime = lac_runtime_implementation_manifest(initializer)
    bank = load_initializer_bank(
        bank_path,
        expected_manifest_sha256=manifest_pin,
        prepared_tasks=public_tasks,
        prepared_manifest_sha256=_corpus_manifest_digest(args.corpus),
        initializer=initializer,
        runtime_implementation_manifest=runtime,
        config=complete_config,
        context=context,
    )
    return bank, manifest_pin


def _quality_initializer_bank_for_manifest(
    args: argparse.Namespace,
    *,
    manifest: Mapping[str, object],
    context: Context,
    bank_attribute: str = "initializer_bank",
    pin_attribute: str = "expected_initializer_bank_manifest_sha256",
    config_attribute: str = "complete_config",
):
    """Load the live bank for publication replay, or require explicit diagnostic isolation."""

    if not _quality_manifest_is_publication(manifest):
        if any(
            getattr(args, name, None) is not None
            for name in (bank_attribute, pin_attribute, config_attribute)
        ):
            raise ValueError("diagnostic quality replay cannot consume a publication bank")
        return None, None

    from isingfold.rl.data.prepared import load_prepared_partition

    public = load_prepared_partition(
        args.corpus,
        partition="train",
        include_evaluator=False,
    )
    if public.target_access is not None:
        raise RuntimeError("quality replay initializer-bank load opened evaluator targets")
    return _load_quality_initializer_bank(
        args,
        public_tasks=public.tasks,
        context=context,
        bank_attribute=bank_attribute,
        pin_attribute=pin_attribute,
        config_attribute=config_attribute,
    )


def _quality_resolution_implementation(ctx: Context) -> dict[str, object]:
    from isingfold.rl.data import quality as quality_module
    from isingfold.rl.data import quality_resolution_delta as delta_module

    contract = _quality_implementation_contract(ctx)
    return {
        "planner": "quality-resolution-plan-v1",
        "quality_implementation_contract_digest": content_digest(contract),
        "quality_module_sha256": _quality_resolution_source_sha256(
            quality_module, "quality implementation"
        ),
        "quality_resolution_delta_module_sha256": _quality_resolution_source_sha256(
            delta_module, "quality-resolution delta implementation"
        ),
    }


def cmd_plan_quality_resolution(args: argparse.Namespace) -> None:
    """Seal the complete target-free all-train continuation-resolution plan."""

    from isingfold.rl.data.prepared import load_prepared_partition
    from isingfold.rl.data.quality_resolution_plan import (
        QualityResolutionPlanAuthority,
        build_quality_resolution_plan,
        load_quality_resolution_planning_inputs,
        materialize_resolution_production_plan,
        publish_quality_resolution_plan,
    )

    inputs = load_quality_resolution_planning_inputs(
        args.config,
        expected_config_sha256=args.expected_config_sha256,
        grid_path=args.grid,
        expected_grid_sha256=args.expected_grid_sha256,
    )
    parity, parity_sha256 = _load_quality_selector_device_parity(
        args.selector_device_parity,
        args.expected_selector_device_parity_sha256,
        device=args.device,
    )
    quality_pin = _quality_attestation_pin(args)
    public = load_prepared_partition(
        args.corpus,
        partition="train",
        include_evaluator=False,
    )
    if public.target_access is not None:
        raise RuntimeError("quality-resolution planning unexpectedly opened evaluator targets")
    ctx = _context(args, args.corpus)
    bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
    _bind_selector_quality_authority(bundle, public.tasks, pin=quality_pin)
    _seed_runtime(
        inputs.config.quality_seed,
        deterministic=args.deterministic,
        threads=args.threads,
    )
    device = _resolve_device(args.device)
    if device.type != args.device:
        raise RuntimeError("quality-resolution selector device resolution changed")
    bundle.model.to(device).eval()

    manifest_path = Path(args.corpus) / "manifest.json"
    manifest = _strict_json(manifest_path)
    _verify_record(manifest, "prepared-v4 manifest")
    if manifest.get("schema_version") != 4:
        raise ValueError("quality-resolution planning requires prepared schema v4")
    design = load_prepared_corpus_design(args.corpus)
    design_sha256 = _require_lower_sha256(
        design.get("manifest_sha256"), "corpus-design manifest SHA-256"
    )
    source_manifest_sha256 = _sha256_file(manifest_path)
    initializer_bank, initializer_bank_manifest_sha256 = _load_quality_initializer_bank(
        args,
        public_tasks=public.tasks,
        context=ctx,
    )
    production = materialize_resolution_production_plan(
        public.tasks,
        context=ctx,
        selector=bundle.model,
        config=inputs.config,
        source_corpus_manifest_sha256=source_manifest_sha256,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=(initializer_bank_manifest_sha256),
    )
    fit_receipt_path = bundle.root / "fit_receipt.json"
    fit_receipt = _strict_json(fit_receipt_path)
    _verify_record(fit_receipt, "selector fit receipt")
    global_authority = dict(quality_pin.global_quality_authority)
    _verify_record(global_authority, "global quality authority")
    ground_identity = global_authority.get("ground_root")
    if not isinstance(ground_identity, Mapping):
        raise ValueError("global quality authority has no ground-root identity")
    publisher_id = global_authority.get("publisher_id")
    if not isinstance(publisher_id, str) or not publisher_id:
        raise ValueError("global quality authority has no publisher identity")
    implementation = _quality_resolution_implementation(ctx)
    context_snapshot = _context_snapshot(ctx)
    authority = QualityResolutionPlanAuthority(
        prepared_manifest_sha256=source_manifest_sha256,
        prepared_manifest_record_digest=_require_lower_sha256(
            manifest.get("record_digest"), "prepared manifest record digest"
        ),
        prepared_train_census_record_digest=production.source_census.record_digest,
        corpus_design_manifest_sha256=design_sha256,
        publisher_id=publisher_id,
        publisher_attestation_sha256=_sha256_file(quality_pin.path),
        publisher_attestation_record_digest=_require_lower_sha256(
            global_authority.get("publisher_attestation_record_digest"),
            "publisher attestation record digest",
        ),
        target_authority_record_digest=_require_lower_sha256(
            global_authority.get("target_authority_record_digest"),
            "target-authority record digest",
        ),
        ground_root_sha256=quality_pin.ground_certificate_root_sha256,
        ground_root_record_digest=quality_pin.ground_certificate_root_record_digest,
        verifier_identity_digest=_require_lower_sha256(
            ground_identity.get("verifier_identity_digest"),
            "ground verifier identity digest",
        ),
        selector_digest=bundle.selector_digest,
        selector_file_sha256=_sha256_file(bundle.root / "selector.pt"),
        selector_fit_receipt_sha256=_sha256_file(fit_receipt_path),
        selector_fit_record_digest=_require_lower_sha256(
            fit_receipt.get("record_digest"), "selector-fit record digest"
        ),
        normalizer_digest=bundle.normalizer_digest,
        selector_device=args.device,
        selector_device_parity_sha256=parity_sha256,
        selector_device_parity_record_digest=_require_lower_sha256(
            parity.get("record_digest"), "selector-parity record digest"
        ),
        context=context_snapshot,
        context_digest=content_digest(context_snapshot),
        implementation=implementation,
        implementation_digest=content_digest(implementation),
    )
    plan = build_quality_resolution_plan(inputs, authority, production)
    plan_sha256 = publish_quality_resolution_plan(args.out, plan)
    record = plan.as_dict()
    print(
        json.dumps(
            {
                "evaluator_targets_opened": False,
                "lineages": record["production_plan"]["lineage_count"],
                "plan_sha256": plan_sha256,
                "record_digest": record["record_digest"],
                "sample_lineages": record["sample"]["sample_size"],
                "out": str(Path(args.out)),
            },
            indent=1,
        )
    )


def _validate_quality_resolution_work_indices(
    plan: object, *, stage_index: int, shard_index: int
) -> None:
    record = plan.as_dict()
    stages = record.get("stages")
    shards = record.get("shards")
    if (
        isinstance(stage_index, bool)
        or not isinstance(stage_index, int)
        or not isinstance(stages, list)
        or stage_index < 0
        or stage_index >= len(stages)
    ):
        raise ValueError("quality-resolution stage index is outside the sealed plan")
    if (
        isinstance(shard_index, bool)
        or not isinstance(shard_index, int)
        or not isinstance(shards, list)
        or shard_index < 0
        or shard_index >= len(shards)
    ):
        raise ValueError("quality-resolution shard index is outside the sealed plan")


def _load_quality_resolution_target_records(
    corpus: str | os.PathLike[str], target_access: Mapping[str, object]
) -> tuple[dict[str, Any], ...]:
    """Read exactly the authenticated train target file at the execution boundary."""

    relative = target_access.get("target_path")
    if relative != "targets/train.jsonl":
        raise ValueError("quality-resolution execution may open only targets/train.jsonl")
    expected_sha256 = _require_lower_sha256(
        target_access.get("target_sha256"), "train target-file SHA-256"
    )
    source = Path(corpus) / relative
    if source.is_symlink() or not source.is_file():
        raise ValueError("authenticated train target file is not a regular file")
    raw = source.read_bytes()
    if not hmac.compare_digest(hashlib.sha256(raw).hexdigest(), expected_sha256):
        raise ValueError("train target file differs from its authenticated access receipt")
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("train target JSONL must end with exactly one record newline")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise ValueError(f"train target JSONL line {line_number} is blank")
        row = _strict_json_object(line, f"train target JSONL line {line_number}")
        if line != canonical_json_bytes(row):
            raise ValueError(f"train target JSONL line {line_number} is not canonical")
        _verify_record(row, f"train target JSONL line {line_number}")
        rows.append(row)
    expected_count = target_access.get("target_count")
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
        or len(rows) != expected_count
    ):
        raise ValueError("train target JSONL count differs from its access receipt")
    return tuple(rows)


def _quality_resolution_public_runtime_inputs(args: argparse.Namespace, plan: object):
    """Authenticate every public runtime identity without opening evaluator targets."""

    from isingfold.rl.data import quality as quality_module
    from isingfold.rl.data import quality_resolution_delta as delta_module
    from isingfold.rl.data.prepared import load_prepared_partition

    plan_record = plan.as_dict()
    quality_pin = _quality_attestation_pin(args)
    public = load_prepared_partition(
        args.corpus,
        partition="train",
        include_evaluator=False,
    )
    if public.target_access is not None:
        raise RuntimeError(
            "quality-resolution pre-execution checks unexpectedly opened evaluator targets"
        )
    ctx = _context(args, args.corpus)
    bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
    _bind_selector_quality_authority(bundle, public.tasks, pin=quality_pin)
    prepared_section = plan_record.get("prepared_corpus")
    publisher_section = plan_record.get("publisher")
    ground_section = plan_record.get("ground_root")
    selector_section = plan_record.get("selector")
    context_section = plan_record.get("context")
    implementation_section = plan_record.get("implementation")
    registry = (
        implementation_section.get("registry")
        if isinstance(implementation_section, Mapping)
        else None
    )
    if not all(
        isinstance(section, Mapping)
        for section in (
            prepared_section,
            publisher_section,
            ground_section,
            selector_section,
            context_section,
            registry,
        )
    ):
        raise ValueError("quality-resolution plan execution identity is malformed")
    current_global = dict(quality_pin.global_quality_authority)
    current_ground = current_global.get("ground_root")
    if not isinstance(current_ground, Mapping):
        raise ValueError("current global quality authority has no ground-root identity")
    if (
        prepared_section.get("manifest_sha256") != _corpus_manifest_digest(args.corpus)
        or publisher_section.get("publisher_id") != current_global.get("publisher_id")
        or publisher_section.get("attestation_sha256") != _sha256_file(quality_pin.path)
        or publisher_section.get("attestation_record_digest")
        != current_global.get("publisher_attestation_record_digest")
        or publisher_section.get("target_authority_record_digest")
        != current_global.get("target_authority_record_digest")
        or ground_section.get("sha256") != current_ground.get("receipt_sha256")
        or ground_section.get("record_digest") != current_ground.get("record_digest")
        or ground_section.get("verifier_identity_digest")
        != current_ground.get("verifier_identity_digest")
    ):
        raise ValueError(
            "quality-resolution corpus or target-free authority differs from the sealed plan"
        )
    fit_receipt_path = bundle.root / "fit_receipt.json"
    fit_receipt = _strict_json(fit_receipt_path)
    _verify_record(fit_receipt, "selector fit receipt")
    if (
        selector_section.get("selector_digest") != bundle.selector_digest
        or selector_section.get("normalizer_digest") != bundle.normalizer_digest
        or selector_section.get("selector_file_sha256") != _sha256_file(bundle.root / "selector.pt")
        or selector_section.get("fit_receipt_sha256") != _sha256_file(fit_receipt_path)
        or selector_section.get("fit_receipt_record_digest") != fit_receipt.get("record_digest")
        or context_section.get("snapshot") != _context_snapshot(ctx)
        or context_section.get("digest") != content_digest(_context_snapshot(ctx))
    ):
        raise ValueError("quality-resolution selector or context differs from the sealed plan")
    device = _resolve_device(args.device)
    if device.type not in {"cpu", "cuda"} or device.type != selector_section.get("device"):
        raise ValueError("quality-resolution execution device differs from the sealed plan")
    bundle.model.to(device).eval()
    config = plan_record.get("config")
    registered = config.get("registered") if isinstance(config, Mapping) else None
    if not isinstance(registered, Mapping):
        raise ValueError("quality-resolution plan config is malformed")
    quality_seed = registered.get("quality_seed")
    if isinstance(quality_seed, bool) or not isinstance(quality_seed, int) or quality_seed < 0:
        raise ValueError("quality-resolution plan seed is invalid")
    _seed_runtime(
        quality_seed,
        deterministic=args.deterministic,
        threads=args.threads,
    )
    current_contract_digest = content_digest(_quality_implementation_contract(ctx))
    if current_contract_digest != registry.get("quality_implementation_contract_digest"):
        raise ValueError("loaded quality implementation differs from the sealed plan")
    quality_module_sha256 = _quality_resolution_source_sha256(
        quality_module, "quality implementation"
    )
    delta_module_sha256 = _quality_resolution_source_sha256(
        delta_module, "quality-resolution delta implementation"
    )
    if (
        registry.get("quality_module_sha256") != quality_module_sha256
        or registry.get("quality_resolution_delta_module_sha256") != delta_module_sha256
    ):
        raise ValueError("loaded quality-resolution source differs from the sealed plan")
    runtime_sha256 = getattr(args, "execution_runtime_sha256", None)
    execution_identity = None
    if runtime_sha256 is not None:
        execution_identity = {
            "delta_module_sha256": delta_module_sha256,
            "device": device.type,
            "quality_implementation_contract_digest": current_contract_digest,
            "quality_module_sha256": quality_module_sha256,
            "runtime_sha256": _require_lower_sha256(
                runtime_sha256,
                "quality-resolution execution runtime SHA-256",
            ),
            "selector_digest": bundle.selector_digest,
            "threads": args.threads,
        }
    return (
        quality_pin,
        public.tasks,
        ctx,
        bundle,
        device,
        execution_identity,
    )


def _quality_resolution_execution_inputs(args: argparse.Namespace):
    """Inject train outcomes only after the sealed plan and public identities are valid."""

    from isingfold.rl.data.quality_resolution_delta import (
        QualityResolutionExecutionAuthority,
    )

    plan = _load_pinned_quality_resolution_plan(args.plan, args.expected_plan_sha256)
    stage_index = getattr(args, "stage_index", None)
    shard_index = getattr(args, "shard_index", None)
    if (stage_index is None) != (shard_index is None):
        raise ValueError("quality-resolution stage and shard indices must be supplied together")
    if stage_index is not None and shard_index is not None:
        _validate_quality_resolution_work_indices(
            plan, stage_index=stage_index, shard_index=shard_index
        )
    (
        quality_pin,
        public_tasks,
        ctx,
        bundle,
        _device,
        execution_identity,
    ) = _quality_resolution_public_runtime_inputs(args, plan)
    initializer_bank, initializer_bank_manifest_sha256 = _load_quality_initializer_bank(
        args,
        public_tasks=public_tasks,
        context=ctx,
    )
    if not isinstance(execution_identity, Mapping):
        raise ValueError(
            "target-bearing quality-resolution execution requires a runtime SHA-256 pin"
        )
    tasks, quality_authority, target_access, ground_partition = _load_quality_partition(
        args.corpus,
        partition="train",
        pin=quality_pin,
        role="training_partition",
    )
    selector_authority = _bind_selector_quality_authority(bundle, tasks, pin=quality_pin)
    if selector_authority != quality_authority:
        raise ValueError("selector and execution train authorities differ")
    authority = QualityResolutionExecutionAuthority(
        quality_authority=quality_authority,
        target_access=target_access,
        ground_partition_receipt=ground_partition,
        execution_identity=execution_identity,
        execution_identity_digest=content_digest(execution_identity),
    )
    target_records = _load_quality_resolution_target_records(args.corpus, target_access)
    return (
        plan,
        tasks,
        target_records,
        authority,
        ctx,
        bundle.model,
        initializer_bank,
        initializer_bank_manifest_sha256,
    )


def cmd_run_quality_resolution_shard(args: argparse.Namespace) -> None:
    """Execute one immutable continuation-delta shard from the sealed plan."""

    from isingfold.rl.data.quality_resolution_delta import (
        run_quality_resolution_delta_shard,
    )

    plan, tasks, targets, authority, ctx, selector, initializer_bank, bank_pin = (
        _quality_resolution_execution_inputs(args)
    )
    artifact = run_quality_resolution_delta_shard(
        plan,
        expected_plan_sha256=args.expected_plan_sha256,
        stage_index=args.stage_index,
        shard_index=args.shard_index,
        prepared=tasks,
        target_records=targets,
        authority=authority,
        context=ctx,
        selector=selector,
        output_directory=args.out,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=bank_pin,
    )
    print(
        json.dumps(
            {
                "manifest_sha256": artifact.manifest_sha256,
                "record_count": artifact.manifest["record_count"],
                "record_digest": artifact.manifest["record_digest"],
                "out": str(Path(args.out)),
            },
            indent=1,
        )
    )


def cmd_verify_quality_resolution_shard(args: argparse.Namespace) -> None:
    """Independently replay one externally pinned continuation-delta shard."""

    from isingfold.rl.data.quality_resolution_delta import (
        verify_quality_resolution_delta_shard,
    )

    plan, tasks, targets, authority, ctx, selector, initializer_bank, bank_pin = (
        _quality_resolution_execution_inputs(args)
    )
    receipt = verify_quality_resolution_delta_shard(
        args.delta_root,
        expected_manifest_sha256=args.expected_delta_manifest_sha256,
        plan=plan,
        expected_plan_sha256=args.expected_plan_sha256,
        prepared=tasks,
        target_records=targets,
        authority=authority,
        context=ctx,
        selector=selector,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=bank_pin,
        verifier_identity_path=args.verifier_identity,
        expected_verifier_identity_sha256=args.expected_verifier_identity_sha256,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "pass": receipt["pass"],
                "record_digest": receipt["record_digest"],
                "receipt_sha256": _sha256_file(args.out),
                "out": str(Path(args.out)),
            },
            indent=1,
        )
    )


def cmd_publish_quality_resolution_shard_pins(args: argparse.Namespace) -> None:
    """Seal the complete externally collected delta and verification pin census."""

    from isingfold.rl.data.quality_resolution_merge import (
        QualityResolutionShardPin,
        publish_quality_resolution_shard_pin_registry,
    )

    plan = _load_pinned_quality_resolution_plan(args.plan, args.expected_plan_sha256)
    expected_fields = {
        "delta_manifest_sha256",
        "delta_root",
        "shard_index",
        "stage_index",
        "verification_path",
        "verification_sha256",
    }
    pins = []
    for position, raw in enumerate(args.pin):
        record = _strict_json_object(
            raw.encode("utf-8"), f"quality-resolution shard pin {position}"
        )
        if set(record) != expected_fields:
            raise ValueError("quality-resolution shard-pin CLI schema differs")
        pins.append(QualityResolutionShardPin(**record))
    raw_sha256 = publish_quality_resolution_shard_pin_registry(
        args.out,
        plan=plan,
        expected_plan_sha256=args.expected_plan_sha256,
        completed_stage_index=args.completed_stage_index,
        pins=pins,
    )
    print(json.dumps({"pin_registry_sha256": raw_sha256, "out": str(Path(args.out))}, indent=1))


def cmd_merge_quality_resolution(args: argparse.Namespace) -> None:
    """Merge one externally pinned complete progressive stage into a study receipt."""

    from isingfold.rl.data.quality_resolution_merge import merge_quality_resolution

    plan = _load_pinned_quality_resolution_plan(args.plan, args.expected_plan_sha256)
    raw_sha256 = merge_quality_resolution(
        plan,
        expected_plan_sha256=args.expected_plan_sha256,
        pin_registry_path=args.pin_registry,
        expected_pin_registry_sha256=args.expected_pin_registry_sha256,
        output_path=args.out,
    )
    print(
        json.dumps({"resolution_receipt_sha256": raw_sha256, "out": str(Path(args.out))}, indent=1)
    )


def _load_quality_resolution_study_from_args(args: argparse.Namespace, plan: object):
    from isingfold.rl.data.quality_resolution_merge import (
        load_quality_resolution_study_receipt,
    )

    return load_quality_resolution_study_receipt(
        args.resolution_receipt,
        expected_receipt_sha256=args.expected_resolution_receipt_sha256,
        plan=plan,
        expected_plan_sha256=args.expected_plan_sha256,
    )


def cmd_publish_quality_capacity_budget(args: argparse.Namespace) -> None:
    """Publish the immutable operator capacity limits before canary outcomes."""

    from isingfold.rl.data.quality_capacity import (
        QualityCapacityBudget,
        publish_quality_capacity_budget,
    )

    budget = QualityCapacityBudget(
        budget_id=args.budget_id,
        maximum_artifact_bytes=args.maximum_artifact_bytes,
        maximum_cpu_seconds=args.maximum_cpu_seconds,
        maximum_elapsed_seconds=args.maximum_elapsed_seconds,
        available_workers=args.available_workers,
        minimum_scratch_free_bytes=args.minimum_scratch_free_bytes,
    )
    raw_sha256 = publish_quality_capacity_budget(args.out, budget)
    print(json.dumps({"capacity_budget_sha256": raw_sha256, "out": str(Path(args.out))}, indent=1))


def cmd_plan_quality_capacity_canary(args: argparse.Namespace) -> None:
    """Seal the outcome-blind 16-lineage canary selection before target access."""

    from isingfold.rl.data.quality_capacity import (
        build_quality_capacity_canary_selection,
    )

    plan = _load_pinned_quality_resolution_plan(args.plan, args.expected_plan_sha256)
    study = _load_quality_resolution_study_from_args(args, plan)
    (
        _quality_pin,
        public_tasks,
        ctx,
        bundle,
        _device,
        execution_identity,
    ) = _quality_resolution_public_runtime_inputs(args, plan)
    initializer_bank, initializer_bank_manifest_sha256 = _load_quality_initializer_bank(
        args,
        public_tasks=public_tasks,
        context=ctx,
    )
    if execution_identity is not None:
        raise RuntimeError("target-free capacity selection acquired an execution identity")
    raw_sha256 = build_quality_capacity_canary_selection(
        plan,
        expected_plan_sha256=args.expected_plan_sha256,
        resolution_receipt=study,
        public_prepared=public_tasks,
        context=ctx,
        selector=bundle.model,
        output_path=args.out,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=(initializer_bank_manifest_sha256),
    )
    print(
        json.dumps(
            {
                "capacity_selection_sha256": raw_sha256,
                "evaluator_targets_opened": False,
                "out": str(Path(args.out)),
            },
            indent=1,
        )
    )


def cmd_quality_capacity_canary(args: argparse.Namespace) -> None:
    """Execute the selected production-like canary under a sealed budget."""

    from isingfold.rl.data import quality_capacity as capacity_module
    from isingfold.rl.data.quality_capacity import (
        QUALITY_CAPACITY_MEASUREMENT_PROTOCOL,
        run_quality_capacity_canary,
    )

    plan, tasks, targets, authority, ctx, selector, initializer_bank, bank_pin = (
        _quality_resolution_execution_inputs(args)
    )
    study = _load_quality_resolution_study_from_args(args, plan)
    plan_record = plan.as_dict()
    registry = plan_record["implementation"]["registry"]
    runtime_identity = {
        "capacity_module_sha256": _quality_resolution_source_sha256(
            capacity_module, "quality-capacity implementation"
        ),
        "continuation_runner": "isingfold.rl.data.quality.run_continuation",
        "execution_runtime_sha256": args.execution_runtime_sha256,
        "host_class": args.host_class,
        "measurement_clock": QUALITY_CAPACITY_MEASUREMENT_PROTOCOL,
        "quality_implementation_contract_digest": registry[
            "quality_implementation_contract_digest"
        ],
        "quality_module_sha256": registry["quality_module_sha256"],
    }
    raw_sha256 = run_quality_capacity_canary(
        plan,
        expected_plan_sha256=args.expected_plan_sha256,
        resolution_receipt=study,
        selection_path=args.capacity_selection,
        expected_selection_sha256=args.expected_capacity_selection_sha256,
        budget_path=args.capacity_budget,
        expected_budget_sha256=args.expected_capacity_budget_sha256,
        prepared=tasks,
        target_records=targets,
        authority=authority,
        context=ctx,
        selector=selector,
        runtime_identity=runtime_identity,
        runtime_identity_digest=content_digest(runtime_identity),
        scratch_directory=args.scratch_directory,
        output_path=args.out,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=bank_pin,
    )
    print(json.dumps({"capacity_canary_sha256": raw_sha256, "out": str(Path(args.out))}, indent=1))


def cmd_verify_quality_capacity_canary(args: argparse.Namespace) -> None:
    """Reconstruct a canary and publish the narrow full-generation capability."""

    from isingfold.rl.data.exact_conformance import publish_new_file
    from isingfold.rl.data.quality_capacity import (
        require_passing_quality_capacity_canary,
    )

    plan = _load_pinned_quality_resolution_plan(args.plan, args.expected_plan_sha256)
    study = _load_quality_resolution_study_from_args(args, plan)
    capability = require_passing_quality_capacity_canary(
        args.capacity_canary,
        expected_canary_sha256=args.expected_capacity_canary_sha256,
        plan=plan,
        expected_plan_sha256=args.expected_plan_sha256,
        resolution_receipt=study,
        selection_path=args.capacity_selection,
        expected_selection_sha256=args.expected_capacity_selection_sha256,
        budget_path=args.capacity_budget,
        expected_budget_sha256=args.expected_capacity_budget_sha256,
    )
    raw = canonical_json_bytes(capability) + b"\n"
    destination = publish_new_file(args.out, raw)
    print(
        json.dumps(
            {
                "capacity_capability_sha256": hashlib.sha256(raw).hexdigest(),
                "out": str(destination),
            },
            indent=1,
        )
    )


def _quality_shard_headers_from_root(
    root: str | os.PathLike[str],
) -> list[tuple[int, Path, dict[str, Any], str]]:
    """Discover one strict, complete quality-shard set beneath a registered root."""

    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("quality-shard root must be a regular directory")
    entries = tuple(directory.iterdir())
    if not entries or any(
        entry.is_symlink() or not entry.is_dir() or not (entry / "manifest.json").is_file()
        for entry in entries
    ):
        raise ValueError("quality-shard root must contain only non-symlink shard directories")
    return _quality_shard_headers([str(entry) for entry in entries])


def _quality_preflight_source_shards(
    headers: Sequence[tuple[int, Path, Mapping[str, Any], str]],
    *,
    expected_manifest_sha256: Sequence[str],
    allow_diagnostic_legacy: bool = False,
) -> list[dict[str, object]]:
    """Build the target-free row census after checking every external shard pin."""

    if len(expected_manifest_sha256) != len(headers):
        raise ValueError("quality-shard manifest pin count differs from the complete shard census")
    sources: list[dict[str, object]] = []
    for expected_index, (shard_index, root, manifest, observed_sha256) in enumerate(headers):
        expected_sha256 = _require_lower_sha256(
            expected_manifest_sha256[expected_index],
            f"quality shard {shard_index} manifest SHA-256",
        )
        if not hmac.compare_digest(observed_sha256, expected_sha256):
            raise ValueError(
                f"quality shard {shard_index} differs from its externally pinned manifest"
            )
        diagnostic = root / "DIAGNOSTIC_ONLY" / "receipt.json"
        if diagnostic.exists():
            receipt = _strict_json(diagnostic)
            _verify_record(receipt, f"quality shard {shard_index} diagnostic status")
            if (
                receipt.get("publication_eligible") is not True
                and not allow_diagnostic_legacy
            ):
                raise ValueError(f"quality shard {shard_index} is explicitly diagnostic-only")
        if shard_index != expected_index:
            raise ValueError("quality shard census is missing or out of canonical order")
        rows = _quality_rows_from_artifact(root, manifest)
        row_census: list[dict[str, object]] = []
        for row in rows:
            evaluated = row.get("evaluated")
            if not isinstance(evaluated, list) or not evaluated:
                raise ValueError("quality source row has no evaluated-action census")
            trajectories = 0
            for action in evaluated:
                if not isinstance(action, Mapping):
                    raise ValueError("quality source action is not an object")
                continuations = action.get("continuations")
                if (
                    isinstance(continuations, bool)
                    or not isinstance(continuations, int)
                    or continuations <= 0
                    or any(
                        not isinstance(action.get(field), list)
                        or len(action[field]) != continuations
                        for field in (
                            "continuation_receipts",
                            "continuation_rewards",
                            "continuation_seeds",
                            "continuation_valid",
                        )
                    )
                ):
                    raise ValueError("quality source continuation census is malformed")
                trajectories += continuations
            row_census.append(
                {
                    "lineage": str(row["lineage"]),
                    "source_row_record_digest": str(row["record_digest"]),
                    "task_id": str(row["task_id"]),
                    "trajectory_count": trajectories,
                }
            )
        sharding = manifest["sharding_receipt"]
        sources.append(
            {
                "manifest_record_digest": manifest["record_digest"],
                "manifest_sha256": observed_sha256,
                "records_sha256": manifest["records_sha256"],
                "rows": row_census,
                "selected_lineages": sharding["selected_lineages"],
                "selected_task_ids": sharding["selected_task_ids"],
                "shard_index": shard_index,
            }
        )
    return sources


def _quality_preflight_plan_identity(
    *,
    resolution_plan: object,
    resolution_plan_sha256: str,
    study: object,
    study_sha256: str,
    common_manifest: Mapping[str, object],
    bundle: SelectorBundle,
    context: Context,
) -> dict[str, object]:
    """Bind replay scheduling to the resolution decision and quality-v7 identity."""

    plan_record = resolution_plan.as_dict()
    study_record = study.as_dict()
    label_protocol = common_manifest.get("label_protocol")
    production_protocol = study_record.get("production_protocol")
    if not isinstance(label_protocol, Mapping) or not isinstance(production_protocol, Mapping):
        raise ValueError("quality replay has no registered labeling protocol")
    observed_protocol = {
        "continuations": label_protocol.get("continuations"),
        "evaluated_actions": label_protocol.get("evaluated_actions"),
        "partition": common_manifest.get("partition"),
        "quality_implementation_contract_digest": content_digest(
            label_protocol.get("implementation_contract")
        ),
        "requested_lineages": label_protocol.get("requested_lineages"),
        "reward_reads": label_protocol.get("reward_reads"),
        "seed": label_protocol.get("seed"),
        "selector_device": plan_record["selector"]["device"],
        "states_per_lineage_cap": label_protocol.get("states_per_lineage_cap"),
        "tasks_per_lineage_cap": label_protocol.get("tasks_per_lineage_cap"),
    }
    if (
        study_record.get("terminal") is not True
        or study_record.get("advance") is not True
        or observed_protocol != production_protocol
        or label_protocol.get("requested_lineages") != 0
    ):
        raise ValueError("quality shards differ from the passing all-train resolution protocol")
    current_context = _context_snapshot(context)
    if (
        common_manifest.get("source_corpus_manifest_sha256")
        != plan_record["prepared_corpus"]["manifest_sha256"]
        or common_manifest.get("selector_digest") != bundle.selector_digest
        or common_manifest.get("normalizer_digest") != bundle.normalizer_digest
        or common_manifest.get("context") != current_context
        or common_manifest.get("context_digest") != content_digest(current_context)
        or common_manifest.get("quality_authority") != bundle.quality_authority
    ):
        raise ValueError("quality shards differ from the sealed public replay identity")
    return {
        "authority": {
            "ground_partition_receipt": common_manifest["ground_partition_receipt"],
            "quality_authority": common_manifest["quality_authority"],
            "target_access": common_manifest["target_access"],
        },
        "context": {
            "digest": content_digest(current_context),
            "snapshot": current_context,
        },
        "corpus": {"manifest_sha256": common_manifest["source_corpus_manifest_sha256"]},
        "device_parity": {
            "record_digest": plan_record["selector"]["device_parity_record_digest"],
            "raw_sha256": plan_record["selector"]["device_parity_sha256"],
        },
        "implementation": plan_record["implementation"],
        "resolution": {
            "plan_record_digest": plan_record["record_digest"],
            "plan_sha256": resolution_plan_sha256,
            "production_protocol_digest": content_digest(production_protocol),
            "record_digest": study_record["record_digest"],
            "selected_continuations": study_record["selected_continuations"],
            "sha256": study_sha256,
        },
        "selector": {
            "device": plan_record["selector"]["device"],
            "normalizer_digest": bundle.normalizer_digest,
            "selector_digest": bundle.selector_digest,
        },
    }


def cmd_plan_quality_preflight_shards(args: argparse.Namespace) -> None:
    """Publish a target-free exact replay plan from pinned production quality shards."""

    from isingfold.rl.data.quality_preflight import build_quality_preflight_plan
    from isingfold.rl.data.quality_resolution_merge import (
        load_quality_resolution_study_receipt,
    )

    resolution_plan = _load_pinned_quality_resolution_plan(
        args.resolution_plan, args.expected_resolution_plan_sha256
    )
    study = load_quality_resolution_study_receipt(
        args.resolution_receipt,
        expected_receipt_sha256=args.expected_resolution_receipt_sha256,
        plan=resolution_plan,
        expected_plan_sha256=args.expected_resolution_plan_sha256,
    )
    (
        _quality_pin,
        _public_tasks,
        ctx,
        bundle,
        _device,
        execution_identity,
    ) = _quality_resolution_public_runtime_inputs(args, resolution_plan)
    if execution_identity is not None:
        raise RuntimeError("target-free replay planning acquired an execution identity")
    headers = _quality_shard_headers_from_root(args.quality_shard_root)
    sources = _quality_preflight_source_shards(
        headers,
        expected_manifest_sha256=args.expected_quality_shard_manifest_sha256,
    )
    if args.shard_count != len(headers):
        raise ValueError("replay shard count must equal the whole-lineage quality-shard count")
    identity = _quality_preflight_plan_identity(
        resolution_plan=resolution_plan,
        resolution_plan_sha256=args.expected_resolution_plan_sha256,
        study=study,
        study_sha256=args.expected_resolution_receipt_sha256,
        common_manifest=headers[0][2],
        bundle=bundle,
        context=ctx,
    )
    raw_sha256 = build_quality_preflight_plan(
        source_shards=sources,
        shard_count=args.shard_count,
        identity=identity,
        output_path=args.out,
    )
    print(
        json.dumps(
            {
                "evaluator_targets_opened": False,
                "quality_preflight_plan_sha256": raw_sha256,
                "out": str(Path(args.out)),
            },
            indent=1,
        )
    )


def _load_quality_preflight_worker_public_inputs(
    args: argparse.Namespace, plan_record: Mapping[str, object]
) -> tuple[
    _VerifiedQualityAttestationPin,
    list[PreparedTask],
    Context,
    SelectorBundle,
    Path,
]:
    """Authenticate every public worker input before opening the train targets."""

    from isingfold.rl.data.prepared import load_prepared_partition

    quality_pin = _quality_attestation_pin(args)
    public = load_prepared_partition(args.corpus, partition="train", include_evaluator=False)
    if public.target_access is not None:
        raise RuntimeError("quality replay public validation opened evaluator targets")
    ctx = _context(args, args.corpus)
    bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
    public_authority = _bind_selector_quality_authority(bundle, public.tasks, pin=quality_pin)
    identity = plan_record.get("identity")
    if not isinstance(identity, Mapping):
        raise ValueError("quality-preflight plan identity is missing")
    selector_identity = identity.get("selector")
    context_identity = identity.get("context")
    corpus_identity = identity.get("corpus")
    authority = identity.get("authority")
    if not all(
        isinstance(value, Mapping)
        for value in (selector_identity, context_identity, corpus_identity, authority)
    ):
        raise ValueError("quality-preflight public identity is malformed")
    current_context = _context_snapshot(ctx)
    if (
        corpus_identity.get("manifest_sha256") != _corpus_manifest_digest(args.corpus)
        or selector_identity.get("selector_digest") != bundle.selector_digest
        or selector_identity.get("normalizer_digest") != bundle.normalizer_digest
        or selector_identity.get("device") != args.device
        or context_identity.get("snapshot") != current_context
        or context_identity.get("digest") != content_digest(current_context)
        or authority.get("quality_authority") != public_authority
    ):
        raise ValueError("quality-preflight worker differs from its sealed public plan")
    headers = _quality_shard_headers_from_root(args.quality_shard_root)
    planned_sources = plan_record.get("source_shards")
    if not isinstance(planned_sources, list):
        raise ValueError("quality-preflight plan source census is malformed")
    sources = _quality_preflight_source_shards(
        headers,
        expected_manifest_sha256=[str(item["manifest_sha256"]) for item in planned_sources],
    )
    expected_sources = [
        {
            "manifest_record_digest": source["manifest_record_digest"],
            "manifest_sha256": source["manifest_sha256"],
            "records_sha256": source["records_sha256"],
            "rows": plan_record["assignments"][index]["rows"],
            "selected_lineages": source["selected_lineages"],
            "selected_task_ids": source["selected_task_ids"],
            "shard_index": source["shard_index"],
        }
        for index, source in enumerate(planned_sources)
    ]
    if sources != expected_sources:
        raise ValueError("quality-preflight source shards differ from the sealed plan")
    index = args.shard_index
    if index >= len(headers):
        raise ValueError("quality-preflight worker shard index is outside the plan")
    return quality_pin, list(public.tasks), ctx, bundle, headers[index][1]


def cmd_run_quality_preflight_shard(args: argparse.Namespace) -> None:
    """Run one exact continuation replay and publish compact row evidence."""

    from isingfold.rl.data import quality as quality_module
    from isingfold.rl.data import quality_preflight as preflight_module
    from isingfold.rl.data.quality_preflight import (
        load_quality_preflight_plan,
        publish_quality_preflight_shard,
    )

    plan = load_quality_preflight_plan(args.plan, expected_sha256=args.expected_plan_sha256)
    plan_record = plan.as_dict()
    quality_pin, _public_tasks, ctx, bundle, quality_root = (
        _load_quality_preflight_worker_public_inputs(args, plan_record)
    )
    source_manifest = _strict_json(quality_root / "manifest.json")
    initializer_bank, initializer_bank_manifest_sha256 = (
        _quality_initializer_bank_for_manifest(
            args,
            manifest=source_manifest,
            context=ctx,
        )
    )
    tasks, quality_authority, target_access, ground_partition = _load_quality_partition(
        args.corpus,
        partition="train",
        pin=quality_pin,
        role="training_partition",
    )
    if _bind_selector_quality_authority(bundle, tasks, pin=quality_pin) != quality_authority:
        raise ValueError("quality replay selector and train authorities differ")
    authority = plan_record["identity"]["authority"]
    if (
        authority["quality_authority"] != quality_authority
        or authority["target_access"] != target_access
        or authority["ground_partition_receipt"] != ground_partition
    ):
        raise ValueError("quality replay target authority differs from the sealed plan")
    label_protocol = _strict_json(quality_root / "manifest.json")["label_protocol"]
    _seed_runtime(
        int(label_protocol["seed"]),
        deterministic=args.deterministic,
        threads=args.threads,
    )
    bundle.model.to(_resolve_device(args.device)).eval()
    evidence: list[dict[str, object]] = []
    _records, loaded = _load_quality_labels(
        quality_root,
        corpus=args.corpus,
        selector=bundle,
        context=ctx,
        enforce_resolution=False,
        quality_attestation_pin=quality_pin,
        _allow_shard=True,
        _prepared_train_tasks=tasks,
        _replay_evidence=evidence,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=(initializer_bank_manifest_sha256),
        allow_diagnostic_legacy=bool(getattr(args, "allow_legacy_pilot", False)),
    )
    summary = loaded["loader_summary"]
    if (
        summary["continuation_replays_executed"]
        != summary["continuation_trajectories_authenticated"]
        or summary["continuation_replay_matches"]
        != summary["continuation_trajectories_authenticated"]
    ):
        raise ValueError("quality replay worker did not execute one complete exact replay")
    runtime_identity = {
        "cli_module_sha256": _sha256_file(__file__),
        "device": args.device,
        "execution_runtime_sha256": _require_lower_sha256(
            args.execution_runtime_sha256,
            "quality-preflight execution runtime SHA-256",
        ),
        "preflight_module_sha256": _sha256_file(preflight_module.__file__),
        "quality_module_sha256": _sha256_file(quality_module.__file__),
        "threads": args.threads,
    }
    manifest_sha256 = publish_quality_preflight_shard(
        plan,
        expected_plan_sha256=args.expected_plan_sha256,
        shard_index=args.shard_index,
        evidence=evidence,
        runtime_identity=runtime_identity,
        output_directory=args.out,
    )
    print(
        json.dumps(
            {
                "manifest_sha256": manifest_sha256,
                "replayed_count": summary["continuation_replays_executed"],
                "out": str(Path(args.out)),
            },
            indent=1,
        )
    )


def cmd_merge_quality_preflight_shards(args: argparse.Namespace) -> None:
    """Merge one complete externally pinned exact-replay shard census."""

    from isingfold.rl.data.quality_preflight import (
        load_quality_preflight_plan,
        merge_quality_preflight_shards,
    )

    if len(args.replay_shard) != len(args.expected_replay_shard_manifest_sha256):
        raise ValueError("replay shard paths and external manifest pins must align")
    plan = load_quality_preflight_plan(args.plan, expected_sha256=args.expected_plan_sha256)
    raw_sha256 = merge_quality_preflight_shards(
        plan,
        expected_plan_sha256=args.expected_plan_sha256,
        replay_shards=list(
            zip(
                args.replay_shard,
                args.expected_replay_shard_manifest_sha256,
                strict=True,
            )
        ),
        output_path=args.out,
    )
    print(
        json.dumps(
            {"replay_bundle_sha256": raw_sha256, "out": str(Path(args.out))},
            indent=1,
        )
    )


def cmd_verify_quality_resolution_binding(args: argparse.Namespace) -> None:
    """Bind a terminal study receipt to an unchanged production quality-v7 manifest."""

    from isingfold.rl.data.quality_resolution_binding import (
        verify_quality_resolution_binding,
    )

    plan = _load_pinned_quality_resolution_plan(args.plan, args.expected_plan_sha256)
    raw_sha256 = verify_quality_resolution_binding(
        plan=plan,
        expected_plan_sha256=args.expected_plan_sha256,
        resolution_receipt_path=args.resolution_receipt,
        expected_resolution_receipt_sha256=args.expected_resolution_receipt_sha256,
        quality_manifest_path=args.quality_manifest,
        expected_quality_manifest_sha256=args.expected_quality_manifest_sha256,
        output_path=args.out,
    )
    print(json.dumps({"binding_sha256": raw_sha256, "out": str(Path(args.out))}, indent=1))


def cmd_verify_quality_training_input_readiness(args: argparse.Namespace) -> None:
    """Bind the resolution decision, quality-v7 manifest, and exact preflight."""

    from isingfold.rl.data.quality_resolution_binding import (
        verify_quality_training_input_readiness,
    )

    plan = _load_pinned_quality_resolution_plan(args.plan, args.expected_plan_sha256)
    raw_sha256 = verify_quality_training_input_readiness(
        plan=plan,
        expected_plan_sha256=args.expected_plan_sha256,
        resolution_receipt_path=args.resolution_receipt,
        expected_resolution_receipt_sha256=args.expected_resolution_receipt_sha256,
        binding_path=args.binding,
        expected_binding_sha256=args.expected_binding_sha256,
        quality_manifest_path=args.quality_manifest,
        expected_quality_manifest_sha256=args.expected_quality_manifest_sha256,
        quality_preflight_path=args.quality_preflight,
        expected_quality_preflight_sha256=args.expected_quality_preflight_sha256,
        output_path=args.out,
    )
    print(json.dumps({"readiness_sha256": raw_sha256, "out": str(Path(args.out))}, indent=1))


def _compute_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--threads", type=_positive_int, default=1)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable deterministic PyTorch algorithms (default: enabled)",
    )


def _runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=_nonnegative_int, default=0)
    _compute_arguments(parser)


def _quality_attestation_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_ground_root: bool = True,
) -> None:
    parser.add_argument(
        "--quality-attestation",
        required=True,
        help="publisher-attestation JSON obtained with the prepared corpus",
    )
    parser.add_argument(
        "--expected-quality-attestation-digest",
        required=True,
        help="out-of-band pinned record digest for the publisher attestation",
    )
    parser.add_argument(
        "--expected-quality-publisher-id",
        required=True,
        help="out-of-band pinned publisher identity",
    )
    if require_ground_root:
        parser.add_argument(
            "--ground-certificate-root",
            required=True,
            help="target-free root committing all independently verified target partitions",
        )
        parser.add_argument(
            "--expected-ground-certificate-root-sha256",
            required=True,
            help="out-of-band SHA-256 pin for the ground-certificate root receipt",
        )


def _quality_preflight_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quality-preflight-receipt",
        required=True,
        help="authenticated output of the one full quality continuation/evaluator replay",
    )
    parser.add_argument(
        "--expected-quality-preflight-sha256",
        required=True,
        help="externally pinned SHA-256 of the quality-preflight receipt",
    )


def _final_strength_test_access_arguments(parser: argparse.ArgumentParser) -> None:
    """Arguments that open only the authenticated test target partition."""

    parser.add_argument("--corpus", required=True)
    _quality_attestation_arguments(parser, require_ground_root=False)
    parser.add_argument(
        "--ground-certificate-root",
        required=True,
        help="outcome-blind root receipt committing all ground-certificate partitions",
    )
    parser.add_argument(
        "--expected-ground-certificate-root-sha256",
        required=True,
        help="out-of-band SHA-256 pin for the ground-certificate root receipt",
    )
    parser.add_argument("--qubit-cap", type=_positive_int, default=None)


def _final_strength_source_arguments(parser: argparse.ArgumentParser) -> None:
    """Externally pinned exact three-seed learned and tuned-stock source census."""

    parser.add_argument(
        "--learned-evaluation",
        action="append",
        required=True,
        help="learned complete-system seed directory; repeat exactly three times",
    )
    parser.add_argument(
        "--expected-learned-report-sha256",
        action="append",
        required=True,
        help="out-of-band report.json SHA-256 aligned with --learned-evaluation",
    )
    parser.add_argument(
        "--external-evaluation",
        action="append",
        required=True,
        help="tuned-stock complete-system seed directory; repeat exactly three times",
    )
    parser.add_argument(
        "--expected-external-report-sha256",
        action="append",
        required=True,
        help="out-of-band report.json SHA-256 aligned with --external-evaluation",
    )


def _prepared_arguments(
    parser: argparse.ArgumentParser,
    *,
    selector: bool = True,
    opens_quality_targets: bool = True,
) -> None:
    parser.add_argument("--corpus", required=True, help="authenticated output of prepare")
    if selector:
        parser.add_argument("--selector", required=True, help="frozen selector-bundle directory")
    if opens_quality_targets:
        _quality_attestation_arguments(parser)
    parser.add_argument(
        "--qubit-cap", type=_positive_int, default=None, help="optional equality assertion"
    )
    parser.add_argument(
        "--allow-legacy-pilot",
        action="store_true",
        help=(
            "diagnostic only: explicitly accept prepared schema v1; scientific grid commands "
            "never expose this override"
        ),
    )


def _reward_reads_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--reward-reads",
        type=_registered_reward_reads,
        default=PRODUCTION_REWARD_READS,
        help=f"training-side terminal reads, fixed at {PRODUCTION_REWARD_READS}",
    )


def _quality_resolution_plan_reference_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument("--plan", required=True, help="externally pinned resolution-plan JSON")
    parser.add_argument(
        "--expected-plan-sha256",
        required=True,
        help="out-of-band SHA-256 of the exact resolution-plan bytes",
    )


def _quality_resolution_study_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--resolution-receipt", required=True)
    parser.add_argument("--expected-resolution-receipt-sha256", required=True)


def _quality_capacity_artifact_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--capacity-selection", required=True)
    parser.add_argument("--expected-capacity-selection-sha256", required=True)
    parser.add_argument("--capacity-budget", required=True)
    parser.add_argument("--expected-capacity-budget-sha256", required=True)


def _quality_initializer_bank_arguments(
    parser: argparse.ArgumentParser,
    *,
    required: bool = True,
) -> None:
    parser.add_argument(
        "--initializer-bank",
        required=required,
        default=None,
        help="sealed target-free persistent-K=2 initializer bank",
    )
    parser.add_argument(
        "--expected-initializer-bank-manifest-sha256",
        required=required,
        default=None,
        help="out-of-band SHA-256 pin of the initializer-bank manifest",
    )
    parser.add_argument(
        "--complete-config",
        required=required,
        default=None,
        help="complete-system configuration used to generate the initializer bank",
    )


def _quality_label_bank_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quality-initializer-bank",
        default=None,
        help="sealed initializer bank used to generate and replay publication quality labels",
    )
    parser.add_argument(
        "--expected-quality-initializer-bank-manifest-sha256",
        default=None,
        help="out-of-band SHA-256 pin for the publication quality-label initializer bank",
    )
    parser.add_argument(
        "--quality-complete-config",
        default=None,
        help="complete-system config used to generate the quality-label initializer bank",
    )


def _quality_resolution_compute_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--device",
        required=True,
        choices=("cpu", "cuda"),
        help="selector device sealed by the parity decision and resolution plan",
    )
    parser.add_argument("--threads", type=_positive_int, default=1)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable deterministic PyTorch algorithms (default: enabled)",
    )


def _quality_resolution_execution_arguments(parser: argparse.ArgumentParser) -> None:
    _quality_resolution_plan_reference_arguments(parser)
    parser.add_argument("--stage-index", type=_nonnegative_int, required=True)
    parser.add_argument("--shard-index", type=_nonnegative_int, required=True)
    parser.add_argument("--corpus", required=True, help="authenticated prepared-v4 corpus")
    parser.add_argument("--selector", required=True, help="frozen selector-bundle directory")
    _quality_attestation_arguments(parser)
    _quality_resolution_compute_arguments(parser)
    _quality_initializer_bank_arguments(parser)
    parser.add_argument(
        "--execution-runtime-sha256",
        required=True,
        help="out-of-band SHA-256 identity of the frozen execution runtime",
    )
    parser.add_argument("--qubit-cap", type=_positive_int, default=None)


class _ExactArgumentParser(argparse.ArgumentParser):
    """Disable ambiguous option-prefix expansion for reproducible CLI invocations."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        kwargs.setdefault("allow_abbrev", False)
        super().__init__(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    from isingfold.rl.model import (
        QUALITY_POLICY_PRIOR_DEFAULT,
        QUALITY_POLICY_PRIOR_MODES,
    )

    parser = _ExactArgumentParser(prog="isingfold-rl", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser(
        "prepare",
        help=(
            "verify a provenance-complete CandidateBank-v2 export and publish "
            "partition-sealed prepared schema v4"
        ),
    )
    prepare.add_argument("--bank", required=True)
    prepare.add_argument("--bank-manifest", required=True)
    prepare.add_argument("--evaluator-targets", required=True)
    prepare.add_argument(
        "--provenance",
        required=True,
        help="closed-schema per-task topology/fault/calibration/lineage provenance sidecar",
    )
    prepare.add_argument(
        "--corpus-design-manifest",
        required=True,
        help="registered lineage/condition/quota/power design authority",
    )
    prepare.add_argument(
        "--expected-corpus-design-sha256",
        required=True,
        help="out-of-band SHA-256 pin for the corpus-design manifest",
    )
    prepare.add_argument("--out", required=True)
    prepare.add_argument("--qubit-cap", type=_positive_int, required=True)
    prepare.set_defaults(func=cmd_prepare)

    exact_make = sub.add_parser(
        "make-exact-conformance-corpus",
        help="deterministically publish the target-free eight-task Gate 1 registry",
    )
    exact_make.add_argument("--corpus", required=True)
    exact_make.add_argument(
        "--expected-corpus-manifest-sha256",
        required=True,
        help="out-of-band SHA-256 of the prepared-v4 manifest.json bytes",
    )
    exact_make.add_argument("--out", required=True)
    exact_make.set_defaults(func=cmd_make_exact_conformance_corpus)

    exact_verify = sub.add_parser(
        "verify-exact-conformance-corpus",
        help="independently reproduce a pinned target-free Gate 1 registry",
    )
    exact_verify.add_argument("--corpus", required=True)
    exact_verify.add_argument(
        "--expected-corpus-manifest-sha256",
        required=True,
        help="out-of-band SHA-256 of the prepared-v4 manifest.json bytes",
    )
    exact_verify.add_argument("--registry", required=True)
    exact_verify.add_argument("--expected-registry-sha256", required=True)
    exact_verify.add_argument(
        "--receipt-out",
        default=None,
        help="optional immutable verification receipt (required for publication archives)",
    )
    exact_verify.set_defaults(func=cmd_verify_exact_conformance_corpus)

    initializer_plan = sub.add_parser(
        "plan-initializer-bank",
        help="seal a target-free conditional deployment-initializer schedule",
    )
    initializer_plan.add_argument("--corpus", required=True)
    initializer_plan.add_argument("--config", required=True)
    initializer_plan.add_argument("--training-seed", type=_nonnegative_int, required=True)
    initializer_plan.add_argument("--episode-schedule-start", type=_nonnegative_int, default=0)
    initializer_plan.add_argument("--episode-count", type=_positive_int, required=True)
    initializer_plan.add_argument(
        "--max-draws-per-conditional-episode", type=_positive_int, required=True
    )
    initializer_plan.add_argument("--qubit-cap", type=_positive_int, default=None)
    initializer_plan.add_argument("--out", required=True)
    initializer_plan.set_defaults(func=cmd_plan_initializer_bank)

    def add_initializer_bank_worker_arguments(worker: argparse.ArgumentParser) -> None:
        worker.add_argument("--corpus", required=True)
        worker.add_argument("--config", required=True)
        worker.add_argument("--plan", required=True)
        worker.add_argument("--expected-plan-sha256", required=True)
        worker.add_argument("--bank", required=True)
        worker.add_argument("--qubit-cap", type=_positive_int, default=None)

    initializer_generate = sub.add_parser(
        "generate-initializer-bank-shard",
        help="generate or resume one deterministic initializer-bank shard",
    )
    add_initializer_bank_worker_arguments(initializer_generate)
    initializer_generate.add_argument("--shard-index", type=_nonnegative_int, required=True)
    initializer_generate.add_argument("--shard-count", type=_positive_int, required=True)
    initializer_generate.set_defaults(func=cmd_generate_initializer_bank_shard)

    initializer_seal = sub.add_parser(
        "seal-initializer-bank",
        help="seal a complete initializer bank and emit its external manifest pin",
    )
    add_initializer_bank_worker_arguments(initializer_seal)
    initializer_seal.set_defaults(func=cmd_seal_initializer_bank)

    bootstrap_plan = sub.add_parser(
        "plan-bootstrap-bank",
        help="seal a target-free validation or final-test K=2 bootstrap census",
    )
    bootstrap_plan.add_argument("--grid", required=True)
    bootstrap_plan.add_argument("--corpus", required=True)
    bootstrap_plan.add_argument("--config", required=True)
    bootstrap_plan.add_argument(
        "--protocol-preset",
        required=True,
        choices=("representation-validation", "validation", "final-test"),
    )
    bootstrap_plan.add_argument("--qubit-cap", type=_positive_int, default=None)
    bootstrap_plan.add_argument("--out", required=True)
    bootstrap_plan.set_defaults(func=cmd_plan_bootstrap_bank)

    def add_bootstrap_bank_worker_arguments(worker: argparse.ArgumentParser) -> None:
        worker.add_argument("--grid", required=True)
        worker.add_argument("--corpus", required=True)
        worker.add_argument("--config", required=True)
        worker.add_argument(
            "--protocol-preset",
            required=True,
            choices=("representation-validation", "validation", "final-test"),
        )
        worker.add_argument("--plan", required=True)
        worker.add_argument("--expected-plan-sha256", required=True)
        worker.add_argument("--bank", required=True)
        worker.add_argument("--qubit-cap", type=_positive_int, default=None)

    bootstrap_generate = sub.add_parser(
        "generate-bootstrap-bank-shard",
        help="generate one deterministic shard of a target-free K=2 bootstrap bank",
    )
    add_bootstrap_bank_worker_arguments(bootstrap_generate)
    bootstrap_generate.add_argument("--shard-index", type=_nonnegative_int, required=True)
    bootstrap_generate.add_argument("--shard-count", type=_positive_int, required=True)
    bootstrap_generate.set_defaults(func=cmd_generate_bootstrap_bank_shard)

    bootstrap_seal = sub.add_parser(
        "seal-bootstrap-bank",
        help="seal a complete validation or final-test K=2 bootstrap bank",
    )
    add_bootstrap_bank_worker_arguments(bootstrap_seal)
    bootstrap_seal.set_defaults(func=cmd_seal_bootstrap_bank)

    ground_certificates = sub.add_parser(
        "verify-ground-certificates",
        help="execute a pinned independent checker over every authenticated ground target",
    )
    ground_certificates.add_argument("--corpus", required=True)
    _quality_attestation_arguments(
        ground_certificates,
        require_ground_root=False,
    )
    ground_certificates.add_argument("--verifier-executable", required=True)
    ground_certificates.add_argument("--expected-verifier-executable-sha256", required=True)
    ground_certificates.add_argument("--verifier-source", required=True)
    ground_certificates.add_argument("--expected-verifier-source-sha256", required=True)
    ground_certificates.add_argument("--verifier-environment", required=True)
    ground_certificates.add_argument("--expected-verifier-environment-sha256", required=True)
    ground_certificates.add_argument("--expected-verifier-name", required=True)
    ground_certificates.add_argument("--expected-verifier-version", required=True)
    ground_certificates.add_argument(
        "--verifier-execution-mode",
        required=True,
        choices=("static-elf", "apptainer"),
    )
    ground_certificates.add_argument("--verifier-runtime")
    ground_certificates.add_argument("--expected-verifier-runtime-sha256")
    ground_certificates.add_argument("--verifier-build-attestation", required=True)
    ground_certificates.add_argument("--expected-verifier-build-attestation-sha256", required=True)
    ground_certificates.add_argument(
        "--verifier-timeout-seconds", type=_positive_float, default=30.0
    )
    ground_certificates.add_argument("--out", required=True)
    ground_certificates.set_defaults(func=cmd_verify_ground_certificates)

    release_v1 = sub.add_parser(
        "prepare-release-v1",
        help="diagnostic only: adapt the legacy release-v1.1 pilot corpus",
    )
    release_v1.add_argument(
        "--quality-corpus",
        action="append",
        required=True,
        help="repeat once per selected non-app quality_*.jsonl corpus",
    )
    release_v1.add_argument("--split-manifest", required=True)
    release_v1.add_argument("--checksums", required=True)
    release_v1.add_argument("--problem-references", required=True)
    release_v1.add_argument("--out", required=True)
    release_v1.add_argument("--qubit-cap", type=_positive_int, required=True)
    release_v1.set_defaults(func=cmd_prepare_release_v1)

    selector_labels = sub.add_parser(
        "label-selector-data",
        help="publish authenticated four-strength graph/count labels",
    )
    _prepared_arguments(selector_labels, selector=False)
    selector_labels.add_argument("--out", required=True)
    selector_labels.add_argument("--seed", type=_nonnegative_int, default=0)
    selector_labels.add_argument("--split-seed", type=_nonnegative_int, default=0)
    selector_labels.add_argument("--calibration-fraction", type=_open_probability, default=0.2)
    selector_labels.add_argument(
        "--audit-mode",
        action="store_true",
        help="after the final RL-value freeze, also label held-out audit partitions",
    )
    selector_labels.add_argument(
        "--selector",
        default=None,
        help="frozen bundle required only with --audit-mode",
    )
    selector_labels.add_argument(
        "--grid",
        default=None,
        help="registered grid required only with --audit-mode",
    )
    selector_labels.add_argument(
        "--rl-value-selection-receipt",
        default=None,
        help="authenticated final validation freeze required only with --audit-mode",
    )
    selector_labels.add_argument(
        "--expected-rl-value-selection-sha256",
        default=None,
        help="external SHA-256 pin for --rl-value-selection-receipt",
    )
    selector_labels.set_defaults(func=cmd_label_selector_data)

    generate = sub.add_parser(
        "dev-generate",
        help="local smoke data only; output is rejected by production train/evaluate",
    )
    _runtime_arguments(generate)
    generate.add_argument("--out", required=True)
    generate.add_argument("--qubit-cap", type=_positive_int, default=160)
    generate.add_argument("--host", default="grid")
    generate.add_argument("--host-size", type=_positive_int, default=10)
    generate.add_argument("--instances", type=_positive_int, default=16)
    generate.add_argument("--variables", type=_positive_int, default=12)
    generate.add_argument("--chain-size", type=_positive_int, default=3)
    generate.add_argument("--fault-rate", type=float, default=0.02)
    generate.add_argument("--alpha", type=float, default=0.5)
    generate.add_argument("--weights", default="1.0")
    generate.add_argument("--clause-length", type=_positive_int, default=6)
    generate.add_argument("--strength-reads", type=_positive_int, default=128)
    generate.add_argument("--ood-family", action="store_true")
    generate.set_defaults(func=cmd_generate)

    selector = sub.add_parser(
        "fit-selector", help="fit and freeze IF-Q3-S0 from authenticated offline labels"
    )
    _runtime_arguments(selector)
    _prepared_arguments(selector, selector=False)
    selector.add_argument("--selector-labels", required=True)
    selector.add_argument("--out", required=True, help="new immutable selector-bundle directory")
    selector.add_argument("--epochs", type=_positive_int, default=100)
    selector.add_argument(
        "--graph-minibatch",
        type=_positive_int,
        default=32,
        help="count-complete embedding records per selector optimizer step",
    )
    selector.add_argument("--learning-rate", type=_positive_float, default=3e-4)
    selector.add_argument("--weight-decay", type=_nonnegative_float, default=1e-3)
    selector.set_defaults(func=cmd_fit_selector)

    selector_audit = sub.add_parser(
        "audit-selector",
        help="post-freeze strength-selector regret audit; never used for fitting",
    )
    selector_audit.add_argument("--corpus", required=True)
    selector_audit.add_argument("--selector", required=True)
    selector_audit.add_argument("--selector-labels", required=True)
    selector_audit.add_argument(
        "--partition",
        choices=("audit_val", "audit_test"),
        default="audit_val",
    )
    selector_audit.add_argument(
        "--grid",
        default=None,
        help="registered grid manifest; mandatory only for audit_test unsealing",
    )
    selector_audit.add_argument(
        "--rl-value-selection-receipt",
        default=None,
        help="frozen final RL-value decision; mandatory for audit_test",
    )
    selector_audit.add_argument(
        "--expected-rl-value-selection-sha256",
        default=None,
        help="external SHA-256 pin for the final RL-value freeze",
    )
    selector_audit.add_argument("--out", required=True)
    selector_audit.set_defaults(func=cmd_audit_selector)

    qpsi_audit = sub.add_parser(
        "audit-qpsi-ordering",
        help="train-only cross-embedding ordering diagnostic for the frozen q_psi score",
    )
    _prepared_arguments(qpsi_audit)
    qpsi_audit.add_argument("--quality-labels", required=True)
    _quality_initializer_bank_arguments(qpsi_audit)
    _quality_preflight_arguments(qpsi_audit)
    qpsi_audit.add_argument("--out", required=True)
    qpsi_audit.add_argument("--min-resolved-rows", type=_positive_int, default=1)
    qpsi_audit.add_argument("--min-resolved-lineages", type=_positive_int, default=1)
    qpsi_audit.add_argument(
        "--bootstrap-replicates",
        type=_positive_int,
        default=20_000,
    )
    qpsi_audit.add_argument(
        "--bootstrap-seed",
        type=_nonnegative_int,
        default=26_090_601,
    )
    qpsi_audit.set_defaults(func=cmd_audit_qpsi_ordering)

    selector_parity = sub.add_parser(
        "audit-quality-selector-device-parity",
        help="compare CPU and CUDA selector decisions on the complete train quality schedule",
    )
    selector_parity.add_argument("--corpus", required=True)
    selector_parity.add_argument("--selector", required=True)
    selector_parity.add_argument("--quality-protocol-config", required=True)
    selector_parity.add_argument(
        "--expected-quality-protocol-config-sha256",
        required=True,
        help="out-of-band SHA-256 of the exact quality-resolution config bytes",
    )
    _quality_initializer_bank_arguments(selector_parity)
    selector_parity.add_argument("--cpu-device", required=True, choices=("cpu",))
    selector_parity.add_argument(
        "--accelerator-device", required=True, choices=("cuda",)
    )
    selector_parity.add_argument(
        "--execution-runtime-sha256",
        required=True,
        help="out-of-band identity of the frozen publication runtime",
    )
    selector_parity.add_argument("--threads", type=_positive_int, default=1)
    selector_parity.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    selector_parity.add_argument("--qubit-cap", type=_positive_int, default=None)
    selector_parity.add_argument("--out", required=True)
    selector_parity.set_defaults(func=cmd_audit_quality_selector_device_parity)

    resolution_plan = sub.add_parser(
        "plan-quality-resolution",
        help="seal the target-free all-train continuation-resolution plan",
    )
    resolution_plan.add_argument("--config", required=True)
    resolution_plan.add_argument("--expected-config-sha256", required=True)
    resolution_plan.add_argument("--grid", required=True)
    resolution_plan.add_argument("--expected-grid-sha256", required=True)
    resolution_plan.add_argument("--corpus", required=True)
    resolution_plan.add_argument("--selector", required=True)
    _quality_attestation_arguments(resolution_plan)
    resolution_plan.add_argument("--selector-device-parity", required=True)
    resolution_plan.add_argument("--expected-selector-device-parity-sha256", required=True)
    _quality_resolution_compute_arguments(resolution_plan)
    _quality_initializer_bank_arguments(resolution_plan)
    resolution_plan.add_argument("--qubit-cap", type=_positive_int, default=None)
    resolution_plan.add_argument("--out", required=True)
    resolution_plan.set_defaults(func=cmd_plan_quality_resolution)

    resolution_verifier_identity = sub.add_parser(
        "publish-quality-resolution-verifier-identity",
        help="publish the independently transported resolution-shard verifier identity",
    )
    _quality_resolution_plan_reference_arguments(resolution_verifier_identity)
    resolution_verifier_identity.add_argument(
        "--verification-runtime-sha256",
        required=True,
        help="out-of-band SHA-256 identity of the frozen verifier runtime",
    )
    resolution_verifier_identity.add_argument("--attestor-id", required=True)
    resolution_verifier_identity.add_argument("--out", required=True)
    resolution_verifier_identity.set_defaults(
        func=cmd_publish_quality_resolution_verifier_identity
    )

    resolution_run = sub.add_parser(
        "run-quality-resolution-shard",
        help="execute one immutable continuation-delta shard",
    )
    _quality_resolution_execution_arguments(resolution_run)
    resolution_run.add_argument("--out", required=True)
    resolution_run.set_defaults(func=cmd_run_quality_resolution_shard)

    resolution_verify = sub.add_parser(
        "verify-quality-resolution-shard",
        help="independently replay one externally pinned continuation-delta shard",
    )
    _quality_resolution_execution_arguments(resolution_verify)
    resolution_verify.add_argument("--delta-root", required=True)
    resolution_verify.add_argument("--expected-delta-manifest-sha256", required=True)
    resolution_verify.add_argument("--verifier-identity", required=True)
    resolution_verify.add_argument("--expected-verifier-identity-sha256", required=True)
    resolution_verify.add_argument("--out", required=True)
    resolution_verify.set_defaults(func=cmd_verify_quality_resolution_shard)

    resolution_pins = sub.add_parser(
        "publish-quality-resolution-shard-pins",
        help="seal a complete stage-prefix census of externally collected shard pins",
    )
    _quality_resolution_plan_reference_arguments(resolution_pins)
    resolution_pins.add_argument("--completed-stage-index", type=_nonnegative_int, required=True)
    resolution_pins.add_argument(
        "--pin",
        action="append",
        required=True,
        help=(
            "one JSON object containing stage_index, shard_index, delta_root, "
            "delta_manifest_sha256, verification_path, and verification_sha256"
        ),
    )
    resolution_pins.add_argument("--out", required=True)
    resolution_pins.set_defaults(func=cmd_publish_quality_resolution_shard_pins)

    resolution_merge = sub.add_parser(
        "merge-quality-resolution",
        help="authenticate and merge one complete progressive resolution stage",
    )
    _quality_resolution_plan_reference_arguments(resolution_merge)
    resolution_merge.add_argument("--pin-registry", required=True)
    resolution_merge.add_argument("--expected-pin-registry-sha256", required=True)
    resolution_merge.add_argument("--out", required=True)
    resolution_merge.set_defaults(func=cmd_merge_quality_resolution)

    resolution_binding = sub.add_parser(
        "verify-quality-resolution-binding",
        help="bind a terminal resolution decision to a production quality-v7 manifest",
    )
    _quality_resolution_plan_reference_arguments(resolution_binding)
    resolution_binding.add_argument("--resolution-receipt", required=True)
    resolution_binding.add_argument("--expected-resolution-receipt-sha256", required=True)
    resolution_binding.add_argument("--quality-manifest", required=True)
    resolution_binding.add_argument("--expected-quality-manifest-sha256", required=True)
    resolution_binding.add_argument("--out", required=True)
    resolution_binding.set_defaults(func=cmd_verify_quality_resolution_binding)

    resolution_readiness = sub.add_parser(
        "verify-quality-training-input-readiness",
        help="bind quality resolution, production labels, and exact preflight for training",
    )
    _quality_resolution_plan_reference_arguments(resolution_readiness)
    resolution_readiness.add_argument("--resolution-receipt", required=True)
    resolution_readiness.add_argument("--expected-resolution-receipt-sha256", required=True)
    resolution_readiness.add_argument("--binding", required=True)
    resolution_readiness.add_argument("--expected-binding-sha256", required=True)
    resolution_readiness.add_argument("--quality-manifest", required=True)
    resolution_readiness.add_argument("--expected-quality-manifest-sha256", required=True)
    resolution_readiness.add_argument("--quality-preflight", required=True)
    resolution_readiness.add_argument("--expected-quality-preflight-sha256", required=True)
    resolution_readiness.add_argument("--out", required=True)
    resolution_readiness.set_defaults(func=cmd_verify_quality_training_input_readiness)

    capacity_budget = sub.add_parser(
        "publish-quality-capacity-budget",
        help="seal operator capacity limits before observing the production-like canary",
    )
    capacity_budget.add_argument("--budget-id", required=True)
    capacity_budget.add_argument("--maximum-artifact-bytes", type=_positive_int, required=True)
    capacity_budget.add_argument("--maximum-cpu-seconds", type=_positive_int, required=True)
    capacity_budget.add_argument("--maximum-elapsed-seconds", type=_positive_int, required=True)
    capacity_budget.add_argument("--available-workers", type=_positive_int, required=True)
    capacity_budget.add_argument("--minimum-scratch-free-bytes", type=_positive_int, required=True)
    capacity_budget.add_argument("--out", required=True)
    capacity_budget.set_defaults(func=cmd_publish_quality_capacity_budget)

    capacity_plan = sub.add_parser(
        "plan-quality-capacity-canary",
        help="seal the target-free 16-lineage production-like canary selection",
    )
    _quality_resolution_plan_reference_arguments(capacity_plan)
    _quality_resolution_study_arguments(capacity_plan)
    capacity_plan.add_argument("--corpus", required=True)
    capacity_plan.add_argument("--selector", required=True)
    _quality_attestation_arguments(capacity_plan)
    _quality_resolution_compute_arguments(capacity_plan)
    _quality_initializer_bank_arguments(capacity_plan)
    capacity_plan.add_argument("--qubit-cap", type=_positive_int, default=None)
    capacity_plan.add_argument("--out", required=True)
    capacity_plan.set_defaults(func=cmd_plan_quality_capacity_canary)

    capacity_run = sub.add_parser(
        "quality-capacity-canary",
        help="execute the selected production-like canary under the sealed budget",
    )
    _quality_resolution_plan_reference_arguments(capacity_run)
    _quality_resolution_study_arguments(capacity_run)
    capacity_run.add_argument("--corpus", required=True)
    capacity_run.add_argument("--selector", required=True)
    _quality_attestation_arguments(capacity_run)
    _quality_resolution_compute_arguments(capacity_run)
    _quality_initializer_bank_arguments(capacity_run)
    capacity_run.add_argument("--execution-runtime-sha256", required=True)
    capacity_run.add_argument("--host-class", required=True)
    capacity_run.add_argument("--qubit-cap", type=_positive_int, default=None)
    _quality_capacity_artifact_arguments(capacity_run)
    capacity_run.add_argument("--scratch-directory", required=True)
    capacity_run.add_argument("--out", required=True)
    capacity_run.set_defaults(func=cmd_quality_capacity_canary)

    capacity_verify = sub.add_parser(
        "verify-quality-capacity-canary",
        help="reconstruct the canary and require a passing full-generation capability",
    )
    _quality_resolution_plan_reference_arguments(capacity_verify)
    _quality_resolution_study_arguments(capacity_verify)
    _quality_capacity_artifact_arguments(capacity_verify)
    capacity_verify.add_argument("--capacity-canary", required=True)
    capacity_verify.add_argument("--expected-capacity-canary-sha256", required=True)
    capacity_verify.add_argument("--out", required=True)
    capacity_verify.set_defaults(func=cmd_verify_quality_capacity_canary)

    replay_plan = sub.add_parser(
        "plan-quality-preflight-shards",
        help="seal a target-free whole-lineage exact-replay plan for quality shards",
    )
    replay_plan.add_argument("--resolution-plan", required=True)
    replay_plan.add_argument("--expected-resolution-plan-sha256", required=True)
    _quality_resolution_study_arguments(replay_plan)
    replay_plan.add_argument("--corpus", required=True)
    replay_plan.add_argument("--selector", required=True)
    _quality_attestation_arguments(replay_plan)
    replay_plan.add_argument("--quality-shard-root", required=True)
    replay_plan.add_argument(
        "--expected-quality-shard-manifest-sha256",
        action="append",
        required=True,
        help="external manifest pin in canonical quality-shard index order",
    )
    replay_plan.add_argument("--shard-count", type=_positive_int, required=True)
    _quality_resolution_compute_arguments(replay_plan)
    replay_plan.add_argument("--qubit-cap", type=_positive_int, default=None)
    replay_plan.add_argument("--out", required=True)
    replay_plan.set_defaults(func=cmd_plan_quality_preflight_shards)

    replay_run = sub.add_parser(
        "run-quality-preflight-shard",
        help="execute and publish one exact quality continuation-replay shard",
    )
    _quality_resolution_plan_reference_arguments(replay_run)
    replay_run.add_argument("--shard-index", type=_nonnegative_int, required=True)
    replay_run.add_argument("--corpus", required=True)
    replay_run.add_argument("--selector", required=True)
    _quality_attestation_arguments(replay_run)
    replay_run.add_argument("--quality-shard-root", required=True)
    _quality_initializer_bank_arguments(replay_run, required=False)
    replay_run.add_argument("--execution-runtime-sha256", required=True)
    _quality_resolution_compute_arguments(replay_run)
    replay_run.add_argument("--qubit-cap", type=_positive_int, default=None)
    replay_run.add_argument("--out", required=True)
    replay_run.set_defaults(func=cmd_run_quality_preflight_shard)

    replay_merge = sub.add_parser(
        "merge-quality-preflight-shards",
        help="publish one trusted bundle from a complete pinned replay-shard census",
    )
    _quality_resolution_plan_reference_arguments(replay_merge)
    replay_merge.add_argument("--replay-shard", action="append", required=True)
    replay_merge.add_argument(
        "--expected-replay-shard-manifest-sha256",
        action="append",
        required=True,
    )
    replay_merge.add_argument("--out", required=True)
    replay_merge.set_defaults(func=cmd_merge_quality_preflight_shards)

    quality = sub.add_parser(
        "label-quality", help="make authenticated counterfactual warm-start labels"
    )
    _runtime_arguments(quality)
    _prepared_arguments(quality)
    _reward_reads_argument(quality)
    quality.add_argument("--out", required=True, help="new immutable quality-label directory")
    quality.add_argument(
        "--instances",
        type=_nonnegative_int,
        default=128,
        help="independent train-lineage count; 0 means every train lineage",
    )
    quality.add_argument("--tasks-per-lineage", type=_positive_int, default=1)
    quality.add_argument(
        "--states-per-lineage",
        "--states",
        dest="states_per_lineage",
        type=_positive_int,
        default=4,
        help="maximum stored decision states across all sampled tasks in one lineage",
    )
    quality.add_argument("--actions", type=_positive_int, default=8)
    quality.add_argument(
        "--continuations",
        type=_positive_int,
        default=None,
        help=(
            "diagnostic-only manual continuations; publication reads the count from the "
            "externally pinned resolution receipt"
        ),
    )
    _quality_initializer_bank_arguments(quality, required=False)
    quality.add_argument(
        "--resolution-plan",
        default=None,
        help="publication-only target-free quality-resolution plan",
    )
    quality.add_argument(
        "--expected-resolution-plan-sha256",
        default=None,
        help="out-of-band raw SHA-256 pin for --resolution-plan",
    )
    quality.add_argument(
        "--resolution-receipt",
        default=None,
        help="publication-only terminal continuation-resolution study receipt",
    )
    quality.add_argument(
        "--expected-resolution-receipt-sha256",
        default=None,
        help="out-of-band raw SHA-256 pin for --resolution-receipt",
    )
    quality.add_argument(
        "--capacity-selection",
        default=None,
        help="publication-only target-free quality-capacity selection",
    )
    quality.add_argument(
        "--expected-capacity-selection-sha256",
        default=None,
        help="out-of-band raw SHA-256 pin for --capacity-selection",
    )
    quality.add_argument(
        "--capacity-budget",
        default=None,
        help="publication-only preregistered quality-capacity budget",
    )
    quality.add_argument(
        "--expected-capacity-budget-sha256",
        default=None,
        help="out-of-band raw SHA-256 pin for --capacity-budget",
    )
    quality.add_argument(
        "--capacity-canary",
        default=None,
        help="publication-only passing quality-capacity canary",
    )
    quality.add_argument(
        "--expected-capacity-canary-sha256",
        default=None,
        help="out-of-band raw SHA-256 pin for --capacity-canary",
    )
    quality.add_argument(
        "--shard-index",
        type=_nonnegative_int,
        default=None,
        help="zero-based lineage-shard index; must be paired with --shard-count",
    )
    quality.add_argument(
        "--shard-count",
        type=_positive_int,
        default=None,
        help="total deterministic lineage shards; must be paired with --shard-index",
    )
    quality.add_argument("--partition", choices=("train", "validation"), default="train")
    quality.set_defaults(func=cmd_label_quality)

    merge_quality = sub.add_parser(
        "merge-quality-labels",
        help="exact-replay a complete deterministic shard set and publish one merged corpus",
    )
    _runtime_arguments(merge_quality)
    _prepared_arguments(merge_quality)
    merge_quality.add_argument(
        "--shard",
        action="append",
        required=True,
        help="quality shard directory; repeat once for every expected shard",
    )
    merge_quality.add_argument(
        "--out", required=True, help="new immutable merged quality-label directory"
    )
    _quality_initializer_bank_arguments(merge_quality, required=False)
    merge_quality.add_argument("--trusted-replay-bundle", default=None)
    merge_quality.add_argument("--expected-trusted-replay-bundle-sha256", default=None)
    merge_quality.set_defaults(func=cmd_merge_quality_labels)

    preflight = sub.add_parser(
        "quality-preflight",
        help="authenticate, exact-replay and resolution-check counterfactual labels",
    )
    _runtime_arguments(preflight)
    _prepared_arguments(preflight)
    preflight.add_argument("--quality-labels", required=True)
    _quality_initializer_bank_arguments(preflight, required=False)
    preflight.add_argument("--out", required=True)
    preflight.add_argument("--min-resolved-rows", type=_positive_int, default=1)
    preflight.add_argument("--min-resolved-lineages", type=_positive_int, default=1)
    preflight.add_argument(
        "--workers",
        type=_positive_int,
        default=1,
        help="bounded whole-lineage worker count; changes scheduling only",
    )
    preflight.set_defaults(func=cmd_quality_preflight)

    warm = sub.add_parser(
        "warm-start", help="supervised actor-ranking and utility-critic initialization"
    )
    _runtime_arguments(warm)
    _prepared_arguments(warm)
    warm.add_argument("--quality-labels", required=True)
    _quality_label_bank_arguments(warm)
    _quality_preflight_arguments(warm)
    warm.add_argument("--out", required=True)
    warm.add_argument("--model-family", choices=REGISTERED_MODEL_FAMILIES, required=True)
    warm.add_argument(
        "--quality-prior-mode",
        choices=QUALITY_POLICY_PRIOR_MODES,
        default=QUALITY_POLICY_PRIOR_DEFAULT,
    )
    warm.add_argument("--epochs", type=_positive_int, default=200)
    warm.add_argument("--minibatch", type=_positive_int, default=32)
    warm.add_argument("--learning-rate", type=_positive_float, default=3e-4)
    warm.add_argument("--weight-decay", type=_nonnegative_float, default=0.0)
    warm.add_argument("--min-resolved-rows", type=_positive_int, default=1)
    warm.add_argument("--min-resolved-lineages", type=_positive_int, default=1)
    warm.add_argument("--resume", action="store_true")
    warm.set_defaults(func=cmd_warm_start)

    warm_control = sub.add_parser(
        "warm-quality-control-cell",
        help="run one fixed post-selection rank-plus-value warm control",
    )
    _compute_arguments(warm_control)
    warm_control.add_argument("--config", required=True)
    warm_control.add_argument("--expected-config-sha256", required=True)
    warm_control.add_argument("--grid", required=True)
    warm_control.add_argument("--expected-grid-sha256", required=True)
    warm_control.add_argument("--corpus", required=True)
    warm_control.add_argument("--selector", required=True)
    _quality_attestation_arguments(warm_control)
    warm_control.add_argument("--quality-labels", required=True)
    _quality_label_bank_arguments(warm_control)
    _quality_preflight_arguments(warm_control)
    warm_control.add_argument("--representation-selection-receipt", required=True)
    warm_control.add_argument(
        "--expected-representation-selection-sha256",
        required=True,
    )
    warm_control.add_argument("--representation-run-root", required=True)
    warm_control.add_argument("--run-root", required=True)
    warm_control.add_argument("--index", type=_nonnegative_int, required=True)
    warm_control.set_defaults(func=cmd_warm_quality_control_cell)

    train = sub.add_parser("train", help="complete-episode Profile-I masked PPO")
    _runtime_arguments(train)
    _prepared_arguments(train)
    _reward_reads_argument(train)
    train.add_argument("--out", required=True)
    train.add_argument("--model-family", choices=REGISTERED_MODEL_FAMILIES, default="if-core")
    train.add_argument(
        "--quality-prior-mode",
        choices=QUALITY_POLICY_PRIOR_MODES,
        default=QUALITY_POLICY_PRIOR_DEFAULT,
    )
    train.add_argument("--method", choices=TRAINING_METHODS, default="ppo-warm-start")
    train.add_argument("--warm-start", default=None)
    train.add_argument("--updates", type=_nonnegative_int, required=True)
    train.add_argument("--episodes", type=_positive_int, default=64)
    train.add_argument("--ppo-epochs", type=_positive_int, default=4)
    train.add_argument("--minibatch", type=_positive_int, default=256)
    train.add_argument("--learning-rate", type=_positive_float, default=3e-4)
    train.add_argument("--resume", action="store_true")
    train.set_defaults(func=cmd_train)

    complete_train = sub.add_parser(
        "complete-system-train-cell",
        help="train one of all three seeds selected for complete-system confirmation",
    )
    _compute_arguments(complete_train)
    complete_train.add_argument("--grid", required=True)
    complete_train.add_argument("--corpus", required=True)
    complete_train.add_argument("--selector", required=True)
    _quality_attestation_arguments(complete_train)
    complete_train.add_argument(
        "--quality-labels",
        required=True,
        help="authenticated quality-v7 corpus used for fresh supervised initialization",
    )
    _quality_label_bank_arguments(complete_train)
    _quality_preflight_arguments(complete_train)
    complete_train.add_argument("--run-root", required=True)
    complete_train.add_argument(
        "--initializer-bank",
        required=True,
        help="sealed deployment-initializer bank for this selected training seed",
    )
    complete_train.add_argument(
        "--expected-initializer-bank-manifest-sha256",
        required=True,
        help="out-of-band SHA-256 of the selected bank manifest",
    )
    complete_train.add_argument(
        "--complete-config",
        required=True,
        help="complete-system initializer and online-budget configuration",
    )
    complete_train.add_argument(
        "--rl-value-selection-receipt",
        required=True,
        help="validation-only three-seed configuration freeze",
    )
    complete_train.add_argument(
        "--expected-selection-sha256",
        required=True,
        help="externally pinned SHA-256 of the RL-value freeze",
    )
    complete_train.add_argument("--index", required=True, type=_nonnegative_int)
    complete_train.set_defaults(func=cmd_complete_system_train_cell)

    complete_evaluate = sub.add_parser(
        "evaluate-complete-system-cell",
        help="evaluate one frozen training seed on the pre-initialization denominator",
    )
    _compute_arguments(complete_evaluate)
    complete_evaluate.add_argument("--grid", required=True)
    complete_evaluate.add_argument("--corpus", required=True)
    complete_evaluate.add_argument("--selector", required=True)
    _quality_attestation_arguments(complete_evaluate)
    complete_evaluate.add_argument("--run-root", required=True)
    complete_evaluate.add_argument("--config", required=True)
    complete_evaluate.add_argument(
        "--bootstrap-bank",
        required=True,
        help="sealed target-free final-test K=2 bootstrap bank",
    )
    complete_evaluate.add_argument(
        "--expected-bootstrap-plan-sha256",
        required=True,
        help="out-of-band SHA-256 of bootstrap-bank plan.json",
    )
    complete_evaluate.add_argument(
        "--expected-bootstrap-manifest-sha256",
        required=True,
        help="out-of-band SHA-256 of bootstrap-bank manifest.json",
    )
    complete_evaluate.add_argument("--out", required=True)
    complete_evaluate.add_argument("--qubit-cap", type=_positive_int, default=None)
    complete_evaluate.add_argument("--rl-value-selection-receipt", required=True)
    complete_evaluate.add_argument("--expected-selection-sha256", required=True)
    complete_evaluate.add_argument("--index", required=True, type=_nonnegative_int)
    complete_evaluate.set_defaults(func=cmd_evaluate_complete_system_cell)

    complete_aggregate = sub.add_parser(
        "aggregate-complete-system",
        help="authenticate and aggregate the frozen three-seed complete-system endpoint",
    )
    complete_aggregate.add_argument("--grid", required=True)
    complete_aggregate.add_argument("--corpus", required=True)
    _quality_attestation_arguments(complete_aggregate)
    complete_aggregate.add_argument("--evaluation-root", required=True)
    complete_aggregate.add_argument("--config", required=True)
    complete_aggregate.add_argument("--bootstrap-bank", required=True)
    complete_aggregate.add_argument("--expected-bootstrap-plan-sha256", required=True)
    complete_aggregate.add_argument("--expected-bootstrap-manifest-sha256", required=True)
    complete_aggregate.add_argument("--rl-value-selection-receipt", required=True)
    complete_aggregate.add_argument("--expected-selection-sha256", required=True)
    complete_aggregate.add_argument("--out", required=True)
    complete_aggregate.set_defaults(func=cmd_aggregate_complete_system)

    paired_aggregate = sub.add_parser(
        "aggregate-learned-vs-stock",
        help="paired three-seed learned-minus-stock complete-system inference",
    )
    paired_aggregate.add_argument("--grid", required=True)
    paired_aggregate.add_argument("--corpus", required=True)
    _quality_attestation_arguments(paired_aggregate)
    paired_aggregate.add_argument("--learned-config", required=True)
    paired_aggregate.add_argument("--external-config", required=True)
    paired_aggregate.add_argument("--tuning-registry", required=True)
    paired_aggregate.add_argument("--expected-tuning-registry-sha256", required=True)
    paired_aggregate.add_argument("--external-tuning-selection", required=True)
    paired_aggregate.add_argument("--expected-external-tuning-selection-sha256", required=True)
    paired_aggregate.add_argument("--qubit-cap", type=_positive_int, default=None)
    paired_aggregate.add_argument(
        "--learned-evaluation",
        action="append",
        required=True,
        help="one learned seed evaluation directory; provide exactly three",
    )
    paired_aggregate.add_argument(
        "--expected-learned-report-sha256",
        action="append",
        required=True,
        help="out-of-band report.json SHA-256 aligned with --learned-evaluation",
    )
    paired_aggregate.add_argument(
        "--external-evaluation",
        action="append",
        required=True,
        help="one stock evaluation directory; provide exactly three",
    )
    paired_aggregate.add_argument(
        "--expected-external-report-sha256",
        action="append",
        required=True,
        help="out-of-band report SHA-256 aligned with --external-evaluation",
    )
    paired_aggregate.add_argument("--out", required=True)
    paired_aggregate.set_defaults(func=cmd_aggregate_learned_vs_stock)

    evaluate = sub.add_parser("evaluate", help="diagnostic validation evaluation with raw outcomes")
    _runtime_arguments(evaluate)
    _prepared_arguments(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--out", required=True)
    evaluate.add_argument("--partition", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--repetitions", type=_positive_int, default=1)
    evaluate.add_argument("--audit-reads", type=_positive_int, default=DEFAULT_AUDIT_READS)
    evaluate.add_argument("--margin", type=_probability, default=0.02)
    evaluate.add_argument(
        "--greedy", action="store_true", help="diagnostic only; default is sampling"
    )
    evaluate.set_defaults(func=cmd_evaluate)

    external = sub.add_parser(
        "evaluate-external",
        help="diagnostic stock-minorminer validation interface",
    )
    _prepared_arguments(external)
    external.add_argument("--config", required=True)
    external.add_argument("--out", required=True)
    external.add_argument("--partition", choices=("validation", "test"), default="validation")
    external.add_argument("--repetitions", type=_positive_int, default=1)
    external.add_argument("--seed", type=_nonnegative_int, default=0)
    external.set_defaults(func=cmd_evaluate_external)

    external_complete = sub.add_parser(
        "evaluate-external-complete-system",
        help="stock minorminer on the exact frozen pre-initialization test census",
    )
    _compute_arguments(external_complete)
    external_complete.add_argument("--grid", required=True)
    external_complete.add_argument("--corpus", required=True)
    external_complete.add_argument("--selector", required=True)
    _quality_attestation_arguments(external_complete)
    external_complete.add_argument("--config", required=True)
    external_complete.add_argument("--learned-config", required=True)
    external_complete.add_argument("--tuning-registry", required=True)
    external_complete.add_argument(
        "--expected-tuning-registry-sha256",
        required=True,
        help="out-of-band SHA-256 of the immutable validation tuning registry",
    )
    external_complete.add_argument("--external-tuning-selection", required=True)
    external_complete.add_argument(
        "--expected-external-tuning-selection-sha256",
        required=True,
        help="out-of-band SHA-256 frozen before the publication test target is opened",
    )
    external_complete.add_argument("--index", required=True, type=_nonnegative_int)
    external_complete.add_argument("--out", required=True)
    external_complete.add_argument("--qubit-cap", type=_positive_int, default=None)
    external_complete.set_defaults(func=cmd_evaluate_external_complete_system)

    external_tuning = sub.add_parser(
        "evaluate-external-tuning-cell",
        help="run one preregistered stock-minorminer candidate/seed on validation only",
    )
    _compute_arguments(external_tuning)
    external_tuning.add_argument("--registry", required=True)
    external_tuning.add_argument(
        "--expected-registry-sha256",
        required=True,
        help="out-of-band SHA-256 of the immutable finite tuning registry",
    )
    external_tuning.add_argument("--grid", required=True)
    external_tuning.add_argument("--index", required=True, type=_nonnegative_int)
    external_tuning.add_argument("--tuning-seed-index", required=True, type=_nonnegative_int)
    external_tuning.add_argument("--corpus", required=True)
    external_tuning.add_argument("--selector", required=True)
    _quality_attestation_arguments(external_tuning)
    external_tuning.add_argument("--external-config", required=True)
    external_tuning.add_argument("--learned-config", required=True)
    external_tuning.add_argument("--out", required=True)
    external_tuning.add_argument("--qubit-cap", type=_positive_int, default=None)
    external_tuning.set_defaults(func=cmd_evaluate_external_tuning_cell)

    external_tuning_selection = sub.add_parser(
        "select-external-tuning",
        help=(
            "authenticate the complete validation-only stock-minorminer grid and freeze "
            "one deployment strategy"
        ),
    )
    external_tuning_selection.add_argument("--registry", required=True)
    external_tuning_selection.add_argument(
        "--expected-registry-sha256",
        required=True,
        help="out-of-band SHA-256 of the immutable finite tuning registry",
    )
    external_tuning_selection.add_argument("--grid", required=True)
    external_tuning_selection.add_argument("--external-config", required=True)
    external_tuning_selection.add_argument("--evaluations-root", required=True)
    external_tuning_selection.add_argument("--out", required=True)
    external_tuning_selection.set_defaults(func=cmd_select_external_tuning)

    selection = sub.add_parser(
        "select-representation",
        help="freeze the complete three-seed validation representation decision",
    )
    selection.add_argument("--grid", required=True)
    selection.add_argument("--evaluations-root", required=True)
    selection.add_argument("--out", required=True)
    selection.set_defaults(func=cmd_select_representation)

    representation_eval = sub.add_parser(
        "evaluate-representation-cell",
        help="run one preregistered validation evaluation for the 9-cell representation grid",
    )
    representation_eval.add_argument("--grid", required=True)
    representation_eval.add_argument("--index", required=True, type=_nonnegative_int)
    representation_eval.add_argument("--corpus", required=True)
    representation_eval.add_argument("--selector", required=True)
    _quality_attestation_arguments(representation_eval)
    representation_eval.add_argument("--run-root", required=True)
    representation_eval.add_argument("--evaluations-root", required=True)
    representation_eval.add_argument(
        "--complete-config",
        required=True,
        help="registered persistent K=2 complete-system configuration",
    )
    representation_eval.add_argument(
        "--bootstrap-bank",
        required=True,
        help="sealed target-free validation K=2 bootstrap bank",
    )
    representation_eval.add_argument(
        "--expected-bootstrap-plan-sha256",
        required=True,
    )
    representation_eval.add_argument(
        "--expected-bootstrap-manifest-sha256",
        required=True,
    )
    representation_eval.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda", "mps")
    )
    representation_eval.add_argument("--threads", type=_positive_int, default=1)
    representation_eval.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    representation_eval.set_defaults(func=cmd_representation_eval_cell)

    rl_value_eval = sub.add_parser(
        "evaluate-rl-value-cell",
        help="run one locked validation evaluation for the 18-cell RL-value grid",
    )
    rl_value_eval.add_argument("--grid", required=True)
    rl_value_eval.add_argument("--index", required=True, type=_nonnegative_int)
    rl_value_eval.add_argument("--corpus", required=True)
    rl_value_eval.add_argument("--selector", required=True)
    _quality_attestation_arguments(rl_value_eval)
    rl_value_eval.add_argument("--run-root", required=True)
    rl_value_eval.add_argument("--evaluations-root", required=True)
    rl_value_eval.add_argument(
        "--complete-config",
        required=True,
        help="registered persistent K=2 complete-system configuration",
    )
    rl_value_eval.add_argument(
        "--bootstrap-bank",
        required=True,
        help="sealed target-free validation K=2 bootstrap bank",
    )
    rl_value_eval.add_argument(
        "--expected-bootstrap-plan-sha256",
        required=True,
    )
    rl_value_eval.add_argument(
        "--expected-bootstrap-manifest-sha256",
        required=True,
    )
    rl_value_eval.add_argument(
        "--representation-selection-receipt",
        "--selection-receipt",
        dest="representation_selection_receipt",
        required=True,
        help="authenticated validation-only output of select-representation",
    )
    rl_value_eval.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    rl_value_eval.add_argument("--threads", type=_positive_int, default=1)
    rl_value_eval.set_defaults(func=cmd_rl_value_eval_cell)

    rl_value_selection = sub.add_parser(
        "select-rl-value",
        help="authenticate all 18 validation reports and freeze one three-seed configuration",
    )
    rl_value_selection.add_argument("--grid", required=True)
    rl_value_selection.add_argument(
        "--representation-selection-receipt",
        "--selection-receipt",
        dest="representation_selection_receipt",
        required=True,
    )
    rl_value_selection.add_argument("--evaluations-root", required=True)
    rl_value_selection.add_argument("--out", required=True)
    rl_value_selection.set_defaults(func=cmd_select_rl_value)

    gates = sub.add_parser("gates", help="release gates before architecture selection")
    _runtime_arguments(gates)
    _prepared_arguments(gates)
    _reward_reads_argument(gates)
    gates.add_argument("--grid", required=True)
    _quality_resolution_plan_reference_arguments(gates)
    _quality_initializer_bank_arguments(gates)
    gates.add_argument("--quality-labels", required=True)
    _quality_preflight_arguments(gates)
    gates.add_argument("--selector-labels", required=True)
    gates.add_argument(
        "--exact-conformance-corpus",
        required=True,
        help="authenticated bounded task registry dedicated to exponential Gate 1 checks",
    )
    gates.add_argument(
        "--expected-exact-conformance-sha256",
        required=True,
        help="out-of-band SHA-256 pin for the exact-conformance corpus",
    )
    gates.add_argument("--out", required=True, help="new immutable release-gate receipt")
    gates.add_argument("--instances", type=_positive_int, default=6)
    gates.add_argument("--partition", choices=("train", "validation"), default="validation")
    gates.set_defaults(func=cmd_gates)

    grid = sub.add_parser("grid-cell", help="run one registered 9 plus 18 staged-grid cell")
    _runtime_arguments(grid)
    grid.add_argument("--grid", required=True)
    grid.add_argument("--stage", required=True, choices=("representation", "rl_value"))
    grid.add_argument("--index", required=True, type=_nonnegative_int)
    grid.add_argument("--corpus", required=True)
    grid.add_argument("--selector", required=True)
    _quality_attestation_arguments(grid)
    grid.add_argument("--quality-labels", required=True)
    _quality_label_bank_arguments(grid)
    _quality_preflight_arguments(grid)
    grid.add_argument("--run-root", required=True)
    grid.add_argument(
        "--initializer-bank",
        default=None,
        help="sealed deployment-initializer bank for the RL-value cell's training seed",
    )
    grid.add_argument(
        "--expected-initializer-bank-manifest-sha256",
        default=None,
        help="out-of-band SHA-256 of the selected deployment bank manifest",
    )
    grid.add_argument(
        "--complete-config",
        default=None,
        help="registered complete-system initializer and online-budget configuration",
    )
    grid.add_argument(
        "--gate-receipt",
        default=None,
        help="passing mandatory Profile-I gates; required for representation cells",
    )
    grid.add_argument(
        "--expected-gate-receipt-sha256",
        default=None,
        help="out-of-band SHA-256 of the immutable release-gate receipt",
    )
    grid.add_argument(
        "--require-profile-c-gate",
        action="store_true",
        help=(
            "also require the experimental empty-start Profile-C diagnostic; "
            "disabled for the Profile-I representation screen"
        ),
    )
    grid.add_argument(
        "--selection-receipt",
        default=None,
        help="validated output of select-representation; required for scientific RL-value cells",
    )
    grid.add_argument(
        "--selected-simpler",
        choices=("if-mlp", "if-dual"),
        default=None,
        help="manual diagnostic override; never accepted as a scientific selection",
    )
    grid.add_argument(
        "--diagnostic-manual-selection",
        action="store_true",
        help="mark --selected-simpler as a non-scientific diagnostic override",
    )
    grid.set_defaults(func=cmd_grid_cell)

    final_strength_plan = sub.add_parser(
        "plan-final-strength-audit",
        help="seal the outcome-blind strength-audit subset before test-target access",
    )
    final_strength_plan.add_argument("--grid", required=True)
    final_strength_plan.add_argument("--selector", required=True)
    final_strength_plan.add_argument("--corpus", required=True)
    _quality_attestation_arguments(
        final_strength_plan,
        require_ground_root=False,
    )
    final_strength_plan.add_argument("--ground-certificate-root", required=True)
    final_strength_plan.add_argument("--expected-ground-certificate-root-sha256", required=True)
    final_strength_plan.add_argument("--audit-config", required=True)
    final_strength_plan.add_argument("--expected-audit-config-sha256", required=True)
    final_strength_plan.add_argument("--rl-value-selection-receipt", required=True)
    final_strength_plan.add_argument("--expected-selection-sha256", required=True)
    final_strength_plan.add_argument("--external-config", required=True)
    final_strength_plan.add_argument("--tuning-registry", required=True)
    final_strength_plan.add_argument("--expected-tuning-registry-sha256", required=True)
    final_strength_plan.add_argument("--external-tuning-selection", required=True)
    final_strength_plan.add_argument("--expected-external-tuning-selection-sha256", required=True)
    final_strength_plan.add_argument("--qubit-cap", type=_positive_int, default=None)
    final_strength_plan.add_argument("--out", required=True)
    final_strength_plan.set_defaults(func=cmd_plan_final_strength_audit)

    final_strength_seal = sub.add_parser(
        "seal-final-strength-audit-execution",
        help="authenticate test access and all six source runs before any audit read",
    )
    final_strength_seal.add_argument("--plan", required=True)
    final_strength_seal.add_argument("--expected-plan-sha256", required=True)
    _final_strength_test_access_arguments(final_strength_seal)
    _final_strength_source_arguments(final_strength_seal)
    final_strength_seal.add_argument("--out", required=True)
    final_strength_seal.set_defaults(func=cmd_seal_final_strength_audit_execution)

    final_strength_shard = sub.add_parser(
        "run-final-strength-audit-shard",
        help="run one immutable paired-key shard under the sealed audit execution",
    )
    final_strength_shard.add_argument("--plan", required=True)
    final_strength_shard.add_argument("--expected-plan-sha256", required=True)
    final_strength_shard.add_argument("--execution-manifest", required=True)
    final_strength_shard.add_argument("--expected-execution-manifest-sha256", required=True)
    _final_strength_test_access_arguments(final_strength_shard)
    _final_strength_source_arguments(final_strength_shard)
    final_strength_shard.add_argument("--shard-index", required=True, type=_nonnegative_int)
    final_strength_shard.add_argument("--out", required=True)
    final_strength_shard.set_defaults(func=cmd_run_final_strength_audit_shard)

    final_strength_merge = sub.add_parser(
        "merge-final-strength-audit",
        help="authenticate all registered strength-audit shards and publish the final result",
    )
    final_strength_merge.add_argument("--plan", required=True)
    final_strength_merge.add_argument("--expected-plan-sha256", required=True)
    final_strength_merge.add_argument("--execution-manifest", required=True)
    final_strength_merge.add_argument("--expected-execution-manifest-sha256", required=True)
    final_strength_merge.add_argument(
        "--shard",
        action="append",
        required=True,
        help="immutable audit-shard directory; repeat for every registered shard",
    )
    final_strength_merge.add_argument(
        "--expected-shard-receipt-sha256",
        action="append",
        required=True,
        help="out-of-band receipt.json SHA-256 aligned with each --shard",
    )
    final_strength_merge.add_argument("--out", required=True)
    final_strength_merge.set_defaults(func=cmd_merge_final_strength_audit)

    from isingfold.rl.evaluation_workflow_cli import (
        register_evaluation_shard_commands,
    )

    register_evaluation_shard_commands(sub)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
