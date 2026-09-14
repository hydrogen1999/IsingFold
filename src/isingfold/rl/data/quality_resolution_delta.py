"""Immutable continuation-delta execution for quality-resolution plan shards.

Planning remains target-free.  This module is the explicit execution boundary at which an
authenticated train target capability and its ground-certificate partition are injected.
Each worker evaluates only its registered half-open continuation range and stores compact,
independently replayable evidence rather than copying full quality-v7 receipts.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

from isingfold.rl.contracts import Context
from isingfold.rl.data.exact_conformance import (
    ExactConformanceError,
    publish_new_file,
    read_regular_file,
)
from isingfold.rl.data.import_embedbench import (
    CERTIFIED_REFERENCE_STATUSES,
    canonical_json_bytes,
    content_digest,
)
from isingfold.rl.data.prepared import PreparedTask
import isingfold.rl.data.quality as quality_module
from isingfold.rl.data.quality import (
    CONTINUATION_MAX_STEPS,
    QUALITY_INITIALIZER_BANK_PROTOCOL,
    ContinuationResult,
    deterministic_action_sample,
    quality_initializer_bank_contract,
    random_masked_policy,
    replay_decision_with_action_envelope,
    run_continuation,
    validate_continuation_receipt,
    validate_quality_initializer_bank_contract,
    validate_quality_initializer_binding,
)
from isingfold.rl.data.action_certificate import apply_envelope_action
from isingfold.rl.data.quality_attestation import quality_target_set_digest
from isingfold.rl.data.quality_resolution import QualityResolutionError
from isingfold.rl.data.quality_resolution_plan import (
    QualityResolutionPlan,
    quality_action_provenance_fingerprint,
)
from isingfold.rl.env import StrengthSelector
from isingfold.rl.initializer_bank import InitializerSnapshotBank


QUALITY_RESOLUTION_DELTA_SCHEMA = "isingfold.quality-resolution-delta-shard"
QUALITY_RESOLUTION_DELTA_VERSION = 2
QUALITY_RESOLUTION_DELTA_ROW_SCHEMA = "isingfold.quality-resolution-delta-row"
QUALITY_RESOLUTION_DELTA_ROW_VERSION = 2
QUALITY_RESOLUTION_VERIFICATION_SCHEMA = "isingfold.quality-resolution-verification-shard"
QUALITY_RESOLUTION_VERIFICATION_VERSION = 2
QUALITY_RESOLUTION_VERIFIER_IDENTITY_SCHEMA = "isingfold.quality-resolution-verifier-identity"
QUALITY_RESOLUTION_VERIFIER_IDENTITY_VERSION = 1
QUALITY_RESOLUTION_RESUME_SCHEMA = "isingfold.quality-resolution-resume-journal"
QUALITY_RESOLUTION_RESUME_VERSION = 2

_HEX = frozenset("0123456789abcdef")
_TARGET_ACCESS_FIELDS = {
    "evidence_manifest_record_digest",
    "evidence_manifest_sha256",
    "opened_files",
    "partition",
    "prepared_manifest_record_digest",
    "prepared_manifest_sha256",
    "publisher_attestation_digest",
    "publisher_id",
    "record_digest",
    "target_authority_record_digest",
    "target_count",
    "target_path",
    "target_set_digest",
    "target_sha256",
}
_DELTA_ROW_FIELDS = {
    "action_index",
    "action_payload_digest",
    "base_lineage",
    "compiled_program_digests",
    "continuation_index",
    "continuation_receipt_digest",
    "continuation_seed",
    "execution_identity_digest",
    "instance_id",
    "initializer_binding_record_digest",
    "plan_record_digest",
    "plan_sha256",
    "record_digest",
    "returned_valid",
    "reward",
    "row_id",
    "schema",
    "schema_version",
    "selected_embedding_digest",
    "selected_strength_index",
    "shard_index",
    "stage_index",
    "state_fingerprint",
    "task_id",
}
_DELTA_MANIFEST_FIELDS = {
    "assigned_lineages",
    "assigned_row_ids",
    "assigned_task_ids",
    "complete",
    "context_digest",
    "cumulative_continuation_range",
    "delta_continuation_range",
    "execution_identity",
    "execution_identity_digest",
    "ground_partition_receipt_record_digest",
    "ground_partition_receipt_sha256",
    "initializer_bank_contract_record_digest",
    "initializer_bank_manifest_sha256",
    "plan_record_digest",
    "plan_sha256",
    "quality_authority_record_digest",
    "record_count",
    "record_digest",
    "record_set_digest",
    "records_sha256",
    "schema",
    "schema_version",
    "shard_count",
    "shard_index",
    "stage_index",
    "target_access_record_digest",
}
_TARGET_ROW_FIELDS = {
    "certificate_digest",
    "evaluator_protocol_digest",
    "instance_id",
    "instance_record_digest",
    "learning_partition",
    "record_digest",
    "reference_energy",
    "reference_status",
    "schema",
    "schema_version",
}
_EXECUTION_IDENTITY_FIELDS = {
    "delta_module_sha256",
    "device",
    "quality_implementation_contract_digest",
    "quality_module_sha256",
    "runtime_sha256",
    "selector_digest",
    "threads",
}
_VERIFIER_IDENTITY_FIELDS = {
    "attestor_id",
    "continuation_runner",
    "quality_implementation_contract_digest",
    "quality_module_sha256",
    "record_digest",
    "schema",
    "schema_version",
    "verification_module_sha256",
    "verification_runtime_sha256",
}


def _digest(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise QualityResolutionError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise QualityResolutionError(f"{label} must be nonempty text")
    return value


def _index(value: object, label: str, *, upper: int | None = None) -> int:
    if type(value) is not int or value < 0 or (upper is not None and value >= upper):
        raise QualityResolutionError(f"{label} is outside its registered range")
    return value


def _verify_record(record: Mapping[str, object], label: str) -> str:
    digest = _digest(record.get("record_digest"), f"{label} record digest")
    payload = {key: value for key, value in record.items() if key != "record_digest"}
    if not hmac.compare_digest(digest, content_digest(payload)):
        raise QualityResolutionError(f"{label} record digest mismatch")
    return digest


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise QualityResolutionError("authenticated mappings require text keys")
        return {key: _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise QualityResolutionError("authenticated identities require finite numbers")
        return value
    if value is None or type(value) in {str, bool, int}:
        return value
    raise QualityResolutionError(
        f"cannot encode {type(value).__name__} as an authenticated identity"
    )


def _freeze(value: object) -> object:
    normalized = _jsonable(value)
    if isinstance(normalized, dict):
        return MappingProxyType({key: _freeze(item) for key, item in normalized.items()})
    if isinstance(normalized, list):
        return tuple(_freeze(item) for item in normalized)
    return normalized


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_thaw(item) for item in value]
    return value


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise QualityResolutionError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def constant(token: str) -> None:
        raise QualityResolutionError(f"{label} contains non-finite number {token}")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
    except QualityResolutionError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualityResolutionError(f"{label} is invalid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise QualityResolutionError(f"{label} must contain one JSON object")
    if raw != canonical_json_bytes(value) + b"\n":
        raise QualityResolutionError(f"{label} is not canonical JSON followed by one newline")
    return value


def _read_pinned_json(
    path: str | Path,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], bytes, str]:
    expected = _digest(expected_sha256, f"expected {label} SHA-256")
    try:
        _resolved, raw = read_regular_file(path, label)
    except (ExactConformanceError, OSError) as error:
        raise QualityResolutionError(str(error)) from error
    observed = hashlib.sha256(raw).hexdigest()
    if not hmac.compare_digest(observed, expected):
        raise QualityResolutionError(f"{label} differs from its out-of-band pin")
    return _strict_json(raw, label), raw, observed


def _publish_file(path: Path, record: Mapping[str, object]) -> str:
    raw = canonical_json_bytes(record) + b"\n"
    try:
        publish_new_file(path, raw)
    except ExactConformanceError as error:
        raise QualityResolutionError(str(error)) from error
    return hashlib.sha256(raw).hexdigest()


def _module_sha256(module_file: object, label: str) -> str:
    if type(module_file) is not str or not module_file:
        raise QualityResolutionError(f"{label} has no concrete source file")
    source = Path(module_file)
    if source.is_symlink():
        raise QualityResolutionError(f"{label} source may not be a symbolic link")
    resolved = source.resolve()
    if not resolved.is_file():
        raise QualityResolutionError(f"{label} source is not a regular file")
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def _require_registered_continuation_runner(
    continuation_runner: Callable[..., ContinuationResult | None],
    *,
    plan_record: Mapping[str, object],
    authority: QualityResolutionExecutionAuthority,
) -> None:
    """Bind publication to the loaded registered callable and exact source bytes."""

    if continuation_runner is not run_continuation:
        raise QualityResolutionError(
            "quality-resolution publication requires the registered run_continuation callable"
        )
    implementation = plan_record.get("implementation")
    registry = implementation.get("registry") if isinstance(implementation, Mapping) else None
    if not isinstance(registry, Mapping):
        raise QualityResolutionError("resolution plan implementation registry is malformed")
    quality_sha = _module_sha256(quality_module.__file__, "quality implementation")
    delta_sha = _module_sha256(__file__, "quality-resolution delta implementation")
    contract = _digest(
        registry.get("quality_implementation_contract_digest"),
        "planned quality implementation contract digest",
    )
    if (
        registry.get("quality_module_sha256") != quality_sha
        or registry.get("quality_resolution_delta_module_sha256") != delta_sha
        or authority.execution_identity.get("quality_module_sha256") != quality_sha
        or authority.execution_identity.get("delta_module_sha256") != delta_sha
        or authority.execution_identity.get("quality_implementation_contract_digest") != contract
    ):
        raise QualityResolutionError(
            "loaded quality-resolution implementation differs from its sealed identities"
        )


def _load_verifier_identity(
    path: str | Path,
    expected_sha256: str,
    *,
    plan_record: Mapping[str, object],
) -> tuple[dict[str, Any], str]:
    """Load an independently transported verifier identity under a raw-byte pin."""

    identity, _raw, observed_sha = _read_pinned_json(
        path,
        expected_sha256,
        "quality-resolution verifier identity",
    )
    if set(identity) != _VERIFIER_IDENTITY_FIELDS:
        raise QualityResolutionError("quality-resolution verifier identity schema differs")
    _verify_record(identity, "quality-resolution verifier identity")
    if (
        identity.get("schema") != QUALITY_RESOLUTION_VERIFIER_IDENTITY_SCHEMA
        or identity.get("schema_version") != QUALITY_RESOLUTION_VERIFIER_IDENTITY_VERSION
        or identity.get("continuation_runner") != "isingfold.rl.data.quality.run_continuation"
    ):
        raise QualityResolutionError("unsupported quality-resolution verifier identity")
    _text(identity.get("attestor_id"), "quality-resolution verifier attestor ID")
    _digest(
        identity.get("verification_runtime_sha256"),
        "quality-resolution verifier runtime SHA-256",
    )
    implementation = plan_record.get("implementation")
    registry = implementation.get("registry") if isinstance(implementation, Mapping) else None
    if not isinstance(registry, Mapping):
        raise QualityResolutionError("resolution plan implementation registry is malformed")
    quality_sha = _module_sha256(quality_module.__file__, "quality implementation")
    verifier_sha = _module_sha256(__file__, "quality-resolution verifier implementation")
    if (
        identity.get("quality_module_sha256") != quality_sha
        or identity.get("quality_module_sha256") != registry.get("quality_module_sha256")
        or identity.get("verification_module_sha256") != verifier_sha
        or identity.get("verification_module_sha256")
        != registry.get("quality_resolution_delta_module_sha256")
        or identity.get("quality_implementation_contract_digest")
        != registry.get("quality_implementation_contract_digest")
    ):
        raise QualityResolutionError(
            "verifier identity differs from loaded and planned implementation bytes"
        )
    return identity, observed_sha


def publish_quality_resolution_verifier_identity(
    plan: QualityResolutionPlan,
    *,
    verification_runtime_sha256: str,
    attestor_id: str,
    output_path: str | Path,
) -> str:
    """Publish the independently transported verifier identity required by replay.

    The publisher reads the implementation registry from the already authenticated
    resolution plan, then requires those hashes to match the source modules loaded by
    this verifier process.  The runtime identity remains an external operator pin.
    """

    plan_record = plan.as_dict()
    implementation = plan_record.get("implementation")
    registry = implementation.get("registry") if isinstance(implementation, Mapping) else None
    if not isinstance(registry, Mapping):
        raise QualityResolutionError("resolution plan implementation registry is malformed")

    quality_sha256 = _module_sha256(quality_module.__file__, "quality implementation")
    verification_sha256 = _module_sha256(
        __file__,
        "quality-resolution verifier implementation",
    )
    contract_digest = _digest(
        registry.get("quality_implementation_contract_digest"),
        "planned quality implementation contract digest",
    )
    if (
        registry.get("quality_module_sha256") != quality_sha256
        or registry.get("quality_resolution_delta_module_sha256") != verification_sha256
    ):
        raise QualityResolutionError(
            "loaded verifier implementation differs from the resolution plan registry"
        )

    payload: dict[str, object] = {
        "attestor_id": _text(attestor_id, "quality-resolution verifier attestor ID"),
        "continuation_runner": "isingfold.rl.data.quality.run_continuation",
        "quality_implementation_contract_digest": contract_digest,
        "quality_module_sha256": quality_sha256,
        "schema": QUALITY_RESOLUTION_VERIFIER_IDENTITY_SCHEMA,
        "schema_version": QUALITY_RESOLUTION_VERIFIER_IDENTITY_VERSION,
        "verification_module_sha256": verification_sha256,
        "verification_runtime_sha256": _digest(
            verification_runtime_sha256,
            "quality-resolution verifier runtime SHA-256",
        ),
    }
    record = {**payload, "record_digest": content_digest(payload)}
    return _publish_file(Path(output_path), record)


@dataclass(frozen=True, slots=True)
class QualityResolutionExecutionAuthority:
    """The target capability and runtime identity injected only at execution."""

    quality_authority: Mapping[str, object]
    target_access: Mapping[str, object]
    ground_partition_receipt: Mapping[str, object]
    execution_identity: Mapping[str, object]
    execution_identity_digest: str

    def __post_init__(self) -> None:
        quality = _jsonable(self.quality_authority)
        target = _jsonable(self.target_access)
        ground = _jsonable(self.ground_partition_receipt)
        execution = _jsonable(self.execution_identity)
        if not all(isinstance(item, dict) for item in (quality, target, ground, execution)):
            raise QualityResolutionError("quality execution identities must be objects")
        if set(quality) != {
            "global",
            "record_digest",
            "schema",
            "schema_version",
            "training_partition",
        }:
            raise QualityResolutionError("training quality-authority schema differs")
        if (
            quality["schema"] != "isingfold.quality-authority-binding"
            or quality["schema_version"] != 2
        ):
            raise QualityResolutionError("unsupported training quality authority")
        _verify_record(quality, "training quality authority")
        global_authority = quality["global"]
        partition_authority = quality["training_partition"]
        if not isinstance(global_authority, dict) or not isinstance(partition_authority, dict):
            raise QualityResolutionError("quality-authority components must be objects")
        _verify_record(global_authority, "global quality authority")
        _verify_record(partition_authority, "train partition quality authority")
        if (
            global_authority.get("schema") != "isingfold.global-quality-authority"
            or global_authority.get("schema_version") != 1
            or partition_authority.get("schema") != "isingfold.partition-quality-authority"
            or partition_authority.get("schema_version") != 1
            or partition_authority.get("name") != "train"
        ):
            raise QualityResolutionError("quality authority does not bind the train partition")

        if set(target) != _TARGET_ACCESS_FIELDS:
            raise QualityResolutionError("train target-access schema differs")
        _verify_record(target, "train target access")
        if target.get("partition") != "train" or target.get("target_path") != "targets/train.jsonl":
            raise QualityResolutionError("resolution execution requires train target access")
        if type(target.get("target_count")) is not int or int(target["target_count"]) <= 0:
            raise QualityResolutionError("train target access has an invalid target count")
        opened = target.get("opened_files")
        if not isinstance(opened, list) or not opened:
            raise QualityResolutionError("train target access has no opened-file evidence")
        prepared_open = {
            (
                item.get("authority_root"),
                item.get("role"),
                item.get("relative_path"),
                item.get("sha256"),
            )
            for item in opened
            if isinstance(item, dict) and item.get("authority_root") == "prepared"
        }
        if prepared_open != {
            ("prepared", "evaluator-targets", "targets/train.jsonl", target["target_sha256"])
        }:
            raise QualityResolutionError("train target access opened the wrong prepared file")

        _verify_record(ground, "ground train partition")
        if (
            ground.get("schema") != "isingfold.ground-certificate-partition"
            or ground.get("schema_version") != 1
            or ground.get("partition") != "train"
            or ground.get("authority") != target
        ):
            raise QualityResolutionError(
                "ground authority does not certify the injected train targets"
            )
        census = ground.get("census")
        if (
            not isinstance(census, dict)
            or census.get("target_count") != target["target_count"]
            or census.get("accepted_count") != target["target_count"]
        ):
            raise QualityResolutionError("ground train census does not accept every target")

        ground_identity = partition_authority.get("ground_partition")
        if not isinstance(ground_identity, dict):
            raise QualityResolutionError("train quality authority has no ground-partition identity")
        if (
            partition_authority.get("target_access_record_digest") != target["record_digest"]
            or partition_authority.get("target_count") != target["target_count"]
            or partition_authority.get("target_set_digest") != target["target_set_digest"]
            or ground_identity.get("receipt_record_digest") != ground["record_digest"]
            or ground_identity.get("accepted_count") != target["target_count"]
        ):
            raise QualityResolutionError("execution authorities are not mutually bound")
        expected_execution_digest = _digest(
            self.execution_identity_digest, "execution identity digest"
        )
        if set(execution) != _EXECUTION_IDENTITY_FIELDS:
            raise QualityResolutionError("quality-resolution execution identity schema differs")
        if content_digest(execution) != expected_execution_digest:
            raise QualityResolutionError("quality-resolution execution identity digest mismatch")
        if execution.get("device") not in {"cpu", "cuda"}:
            raise QualityResolutionError("quality-resolution execution device is invalid")
        if type(execution.get("threads")) is not int or int(execution["threads"]) <= 0:
            raise QualityResolutionError("quality-resolution execution thread count is invalid")
        _digest(execution.get("selector_digest"), "execution selector digest")
        _digest(execution.get("runtime_sha256"), "execution runtime SHA-256")
        _digest(execution.get("quality_module_sha256"), "quality implementation SHA-256")
        _digest(
            execution.get("delta_module_sha256"),
            "quality-resolution delta implementation SHA-256",
        )
        _digest(
            execution.get("quality_implementation_contract_digest"),
            "quality implementation contract digest",
        )
        object.__setattr__(self, "quality_authority", _freeze(quality))
        object.__setattr__(self, "target_access", _freeze(target))
        object.__setattr__(self, "ground_partition_receipt", _freeze(ground))
        object.__setattr__(self, "execution_identity", _freeze(execution))

    def validate_against_plan(self, plan: QualityResolutionPlan) -> None:
        record = plan.as_dict()
        global_authority = self.quality_authority["global"]
        partition_authority = self.quality_authority["training_partition"]
        target = self.target_access
        ground = self.ground_partition_receipt
        expected_ground = record["ground_root"]
        expected_publisher = record["publisher"]
        expected_prepared = record["prepared_corpus"]
        expected_selector = record["selector"]
        expected_implementation = record["implementation"]
        if not all(
            isinstance(value, Mapping)
            for value in (
                global_authority,
                partition_authority,
                expected_ground,
                expected_publisher,
                expected_prepared,
                expected_selector,
                expected_implementation,
            )
        ):
            raise QualityResolutionError("plan or execution authority lost its object shape")
        root = global_authority.get("ground_root")
        ground_partition = partition_authority.get("ground_partition")
        if not isinstance(root, Mapping) or not isinstance(ground_partition, Mapping):
            raise QualityResolutionError("execution quality authority is incomplete")
        if (
            root.get("receipt_sha256") != expected_ground.get("sha256")
            or root.get("record_digest") != expected_ground.get("record_digest")
            or root.get("verifier_identity_digest")
            != expected_ground.get("verifier_identity_digest")
            or global_authority.get("publisher_id") != expected_publisher.get("publisher_id")
            or global_authority.get("publisher_attestation_record_digest")
            != expected_publisher.get("attestation_record_digest")
            or global_authority.get("target_authority_record_digest")
            != expected_publisher.get("target_authority_record_digest")
        ):
            raise QualityResolutionError("execution global authority differs from the plan")
        if (
            target.get("prepared_manifest_sha256") != expected_prepared.get("manifest_sha256")
            or target.get("prepared_manifest_record_digest")
            != expected_prepared.get("manifest_record_digest")
            or target.get("publisher_id") != expected_publisher.get("publisher_id")
            or target.get("publisher_attestation_digest")
            != expected_publisher.get("attestation_record_digest")
            or target.get("target_authority_record_digest")
            != expected_publisher.get("target_authority_record_digest")
        ):
            raise QualityResolutionError("injected train target access differs from the plan")
        ground_raw = canonical_json_bytes(_thaw(ground)) + b"\n"
        if ground_partition.get("receipt_sha256") != hashlib.sha256(
            ground_raw
        ).hexdigest() or ground_partition.get("receipt_record_digest") != ground.get(
            "record_digest"
        ):
            raise QualityResolutionError("ground train receipt differs from its authority pin")
        if self.execution_identity.get("selector_digest") != expected_selector.get(
            "selector_digest"
        ) or self.execution_identity.get("device") != expected_selector.get("device"):
            raise QualityResolutionError("execution selector identity differs from the plan")
        registry = expected_implementation.get("registry")
        if not isinstance(registry, Mapping) or (
            self.execution_identity.get("quality_module_sha256")
            != registry.get("quality_module_sha256")
            or self.execution_identity.get("delta_module_sha256")
            != registry.get("quality_resolution_delta_module_sha256")
            or self.execution_identity.get("quality_implementation_contract_digest")
            != registry.get("quality_implementation_contract_digest")
        ):
            raise QualityResolutionError("execution quality implementation differs from the plan")

    def identity_fields(self) -> dict[str, object]:
        partition = self.quality_authority["training_partition"]
        ground_identity = partition["ground_partition"]
        return {
            "execution_identity": _thaw(self.execution_identity),
            "execution_identity_digest": self.execution_identity_digest,
            "ground_partition_receipt_record_digest": self.ground_partition_receipt[
                "record_digest"
            ],
            "ground_partition_receipt_sha256": ground_identity["receipt_sha256"],
            "quality_authority_record_digest": self.quality_authority["record_digest"],
            "target_access_record_digest": self.target_access["record_digest"],
        }


@dataclass(frozen=True, slots=True)
class QualityResolutionDeltaArtifact:
    root: Path
    manifest: Mapping[str, object]
    records: tuple[Mapping[str, object], ...]
    manifest_sha256: str


@dataclass(frozen=True, slots=True)
class _DeltaWork:
    work_index: int
    stage_index: int
    shard_index: int
    lower: int
    upper: int
    row: Mapping[str, object]
    action: Mapping[str, object]
    continuation_index: int
    continuation_seed: int

    @property
    def key(self) -> str:
        return (
            f"{self.row['row_id']}:{int(self.action['action_index']):08d}:"
            f"{self.continuation_index:08d}"
        )


def _plan_identity(plan: QualityResolutionPlan, expected_sha256: str) -> tuple[dict[str, Any], str]:
    if not isinstance(plan, QualityResolutionPlan):
        raise TypeError("plan must be QualityResolutionPlan")
    expected = _digest(expected_sha256, "expected quality-resolution plan SHA-256")
    record = plan.as_dict()
    observed = hashlib.sha256(canonical_json_bytes(record) + b"\n").hexdigest()
    if not hmac.compare_digest(observed, expected):
        raise QualityResolutionError("quality-resolution plan differs from its out-of-band pin")
    _verify_record(record, "quality-resolution plan")
    return record, observed


def _context_identity(
    context: Context, plan_record: Mapping[str, object]
) -> tuple[dict[str, Any], str]:
    if not isinstance(context, Context):
        raise TypeError("context must be Context")
    snapshot = _jsonable(context)
    if not isinstance(snapshot, dict):
        raise RuntimeError("Context did not serialize as an object")
    digest = content_digest(snapshot)
    expected = plan_record.get("context")
    if (
        not isinstance(expected, Mapping)
        or expected.get("snapshot") != snapshot
        or expected.get("digest") != digest
    ):
        raise QualityResolutionError("execution Context differs from the resolution plan")
    return snapshot, digest


def _initializer_bank_identity(
    plan_record: Mapping[str, object],
    initializer_bank: InitializerSnapshotBank | None,
    expected_manifest_sha256: str | None,
    *,
    allow_test_bank: bool,
) -> dict[str, object]:
    """Bind delta execution to the exact bank already sealed into the plan.

    This is deliberately a consumer-side contract.  It accepts the existing initializer-bank
    schema and LAC runtime identity, and does not require a new PPO/model-source identity.
    """

    if initializer_bank is None or expected_manifest_sha256 is None:
        raise QualityResolutionError("quality-resolution delta requires a pinned initializer bank")
    production = plan_record.get("production_plan")
    planned = production.get("initializer_bank") if isinstance(production, Mapping) else None
    if not isinstance(planned, Mapping):
        raise QualityResolutionError("quality-resolution plan omits its initializer-bank contract")
    try:
        validate_quality_initializer_bank_contract(
            planned,
            require_publication=not allow_test_bank,
        )
        live = quality_initializer_bank_contract(
            initializer_bank,
            expected_manifest_sha256=expected_manifest_sha256,
            allow_test_bank=allow_test_bank,
        )
    except (TypeError, ValueError) as error:
        raise QualityResolutionError(str(error)) from error
    if live != dict(planned):
        raise QualityResolutionError("live initializer bank differs from the resolution plan")
    return live


def _delta_work(
    plan_record: Mapping[str, object], stage_index: int, shard_index: int
) -> tuple[Mapping[str, object], Mapping[str, object], tuple[_DeltaWork, ...]]:
    stages = plan_record.get("stages")
    shards = plan_record.get("shards")
    production = plan_record.get("production_plan")
    if (
        not isinstance(stages, list)
        or not isinstance(shards, list)
        or not isinstance(production, Mapping)
    ):
        raise QualityResolutionError("resolution plan schedule is malformed")
    stage_position = _index(stage_index, "resolution stage index", upper=len(stages))
    shard_position = _index(shard_index, "resolution shard index", upper=len(shards))
    stage = stages[stage_position]
    shard = shards[shard_position]
    if not isinstance(stage, Mapping) or not isinstance(shard, Mapping):
        raise QualityResolutionError("resolution plan stage or shard is malformed")
    if stage.get("stage_index") != stage_position or shard.get("shard_index") != shard_position:
        raise QualityResolutionError("resolution plan schedule index is noncanonical")
    delta = stage.get("continuation_range")
    if (
        not isinstance(delta, list)
        or len(delta) != 2
        or any(type(value) is not int for value in delta)
        or not 0 <= delta[0] < delta[1]
    ):
        raise QualityResolutionError("resolution stage has an invalid half-open delta")
    lower, upper = delta
    rows = production.get("rows")
    if not isinstance(rows, list):
        raise QualityResolutionError("resolution production rows are malformed")
    row_by_id = {row.get("row_id"): row for row in rows if isinstance(row, Mapping)}
    assigned_ids = shard.get("row_ids")
    if not isinstance(assigned_ids, list) or any(
        row_id not in row_by_id for row_id in assigned_ids
    ):
        raise QualityResolutionError("resolution shard refers to an unknown row")
    work: list[_DeltaWork] = []
    for row_id in sorted(assigned_ids):
        row = row_by_id[row_id]
        actions = row.get("actions")
        if not isinstance(actions, list):
            raise QualityResolutionError("planned resolution row has malformed actions")
        for action in sorted(actions, key=lambda item: item["action_index"]):
            seeds = action.get("continuation_seeds")
            if not isinstance(seeds, list) or len(seeds) < upper:
                raise QualityResolutionError("planned action seed tape is incomplete")
            for continuation_index in range(lower, upper):
                work.append(
                    _DeltaWork(
                        work_index=len(work),
                        stage_index=stage_position,
                        shard_index=shard_position,
                        lower=lower,
                        upper=upper,
                        row=row,
                        action=action,
                        continuation_index=continuation_index,
                        continuation_seed=seeds[continuation_index],
                    )
                )
    return stage, shard, tuple(work)


def _authenticated_target_rows(
    target_records: Sequence[Mapping[str, object]] | None,
    *,
    authority: QualityResolutionExecutionAuthority,
) -> dict[str, Mapping[str, object]]:
    if (
        target_records is None
        or isinstance(target_records, (str, bytes))
        or not isinstance(target_records, Sequence)
        or not target_records
    ):
        raise QualityResolutionError(
            "delta execution requires the complete authenticated train target-row census"
        )
    rows: list[Mapping[str, object]] = []
    by_instance: dict[str, Mapping[str, object]] = {}
    for row in target_records:
        if not isinstance(row, Mapping) or set(row) != _TARGET_ROW_FIELDS:
            raise QualityResolutionError("delta evaluator-target row schema differs")
        _verify_record(row, "delta evaluator-target row")
        if (
            row.get("schema") != "isingfold.evaluator-target"
            or row.get("schema_version") != 2
            or row.get("learning_partition") != "train"
            or row.get("reference_status") not in CERTIFIED_REFERENCE_STATUSES
        ):
            raise QualityResolutionError("delta evaluator target is not certified train v2")
        instance_id = _text(row.get("instance_id"), "delta target instance ID")
        if instance_id in by_instance:
            raise QualityResolutionError("delta evaluator targets repeat an instance ID")
        _digest(row.get("instance_record_digest"), "delta target instance record digest")
        _digest(row.get("certificate_digest"), "delta target certificate digest")
        _digest(
            row.get("evaluator_protocol_digest"),
            "delta target evaluator protocol digest",
        )
        energy = row.get("reference_energy")
        if (
            isinstance(energy, bool)
            or not isinstance(energy, (int, float))
            or not math.isfinite(float(energy))
        ):
            raise QualityResolutionError("delta target reference energy is not finite")
        rows.append(row)
        by_instance[instance_id] = row
    if [str(row["instance_id"]) for row in rows] != sorted(by_instance):
        raise QualityResolutionError("delta evaluator targets are not canonically ordered")
    target = authority.target_access
    if len(rows) != target.get("target_count"):
        raise QualityResolutionError("delta target count differs from target authority")
    if quality_target_set_digest(rows, partition="train") != target.get("target_set_digest"):
        raise QualityResolutionError("delta target set differs from target authority")
    raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    if hashlib.sha256(raw).hexdigest() != target.get("target_sha256"):
        raise QualityResolutionError("delta target bytes differ from target authority")
    return by_instance


def _target_tasks(
    prepared: Sequence[PreparedTask],
    work: Sequence[_DeltaWork],
    authority: QualityResolutionExecutionAuthority,
    target_records: Sequence[Mapping[str, object]] | None = None,
) -> dict[str, PreparedTask]:
    if isinstance(prepared, (str, bytes)) or not isinstance(prepared, Sequence) or not prepared:
        raise QualityResolutionError("delta execution needs target-bearing train tasks")
    expected_task_ids = {str(item.row["task_id"]) for item in work}
    by_id: dict[str, PreparedTask] = {}
    target = authority.target_access
    for item in prepared:
        if not isinstance(item, PreparedTask):
            raise TypeError("delta execution tasks must be PreparedTask values")
        if item.task_id in by_id:
            raise QualityResolutionError("delta execution received duplicate task IDs")
        by_id[item.task_id] = item
    missing = expected_task_ids - set(by_id)
    if missing:
        raise QualityResolutionError(
            f"delta execution is missing planned task(s): {sorted(missing)}"
        )
    for task_id in expected_task_ids:
        item = by_id[task_id]
        planned_rows = [entry.row for entry in work if entry.row["task_id"] == task_id]
        expected_lineages = {str(row["base_lineage"]) for row in planned_rows}
        if (
            item.partition != "train"
            or item.task.name != item.task_id
            or item.task.lineage not in expected_lineages
        ):
            raise QualityResolutionError("target-bearing task crosses its planned train identity")
        if (
            item.task.ground_energy is None
            or item.reference_status is None
            or item.certificate_digest is None
            or item.evaluator_protocol_digest is None
            or item.provenance is None
        ):
            raise QualityResolutionError("delta execution task lacks its evaluator target")
        for label, digest in (
            ("certificate", item.certificate_digest),
            ("evaluator protocol", item.evaluator_protocol_digest),
        ):
            _digest(digest, f"delta task {label} digest")
        metadata = (
            item.quality_attestation_digest,
            item.quality_evidence_manifest_digest,
            item.quality_evidence_manifest_sha256,
            item.quality_target_set_digest,
            item.quality_target_count,
        )
        if any(value is None for value in metadata):
            raise QualityResolutionError("delta execution task lacks authenticated target metadata")
        if (
            item.quality_attestation_digest != target["publisher_attestation_digest"]
            or item.quality_evidence_manifest_digest != target["evidence_manifest_record_digest"]
            or item.quality_evidence_manifest_sha256 != target["evidence_manifest_sha256"]
            or item.quality_target_set_digest != target["target_set_digest"]
            or item.quality_target_count != target["target_count"]
        ):
            raise QualityResolutionError("delta task target metadata differs from target access")
    targets = _authenticated_target_rows(target_records, authority=authority)
    if len(by_id) != target["target_count"] or set(targets) != {
        item.instance_id for item in by_id.values()
    }:
        raise QualityResolutionError(
            "delta target rows do not exactly cover the supplied train task census"
        )
    for task_id in expected_task_ids:
        item = by_id[task_id]
        row = targets.get(item.instance_id)
        if row is None:
            raise QualityResolutionError("delta task has no authenticated evaluator-target row")
        if (
            item.provenance is None
            or item.provenance.instance_record_digest != row["instance_record_digest"]
            or float(item.task.ground_energy) != float(row["reference_energy"])
            or item.reference_status != row["reference_status"]
            or item.certificate_digest != row["certificate_digest"]
            or item.evaluator_protocol_digest != row["evaluator_protocol_digest"]
        ):
            raise QualityResolutionError(
                "delta PreparedTask target values differ from authenticated target row"
            )
    return {task_id: by_id[task_id] for task_id in sorted(expected_task_ids)}


def _validate_planned_rows(
    work: Sequence[_DeltaWork],
    tasks: Mapping[str, PreparedTask],
    context: Context,
    selector: StrengthSelector,
    source_corpus_manifest_sha256: str,
    initializer_bank: InitializerSnapshotBank,
    initializer_bank_manifest_sha256: str,
    authority: QualityResolutionExecutionAuthority,
    allow_test_initializer_bank: bool,
) -> None:
    """Replay each public state once and prove its stored action envelope is still exact."""

    representative: dict[str, _DeltaWork] = {}
    for item in work:
        representative.setdefault(str(item.row["row_id"]), item)
    for row_id in sorted(representative):
        item = representative[row_id]
        row = item.row
        task = tasks[str(row["task_id"])]
        provenance = quality_action_provenance_fingerprint(task, source_corpus_manifest_sha256)
        if provenance != row.get("action_provenance_fingerprint"):
            raise QualityResolutionError(
                "target-bearing task differs from its target-free action provenance"
            )
        replayed = replay_decision_with_action_envelope(
            task.task,
            context,
            initializer=None,
            initializer_bank=initializer_bank,
            expected_initializer_bank_manifest_sha256=(initializer_bank_manifest_sha256),
            initializer_bank_episode_index=row["initializer_bank_episode_index"],
            prepared_task=task,
            target_access=authority.target_access,
            ground_partition_receipt=authority.ground_partition_receipt,
            allow_test_initializer_bank=allow_test_initializer_bank,
            selector=selector,
            prefix=row["prefix"],
            seed=row["environment_seed"],
            reward_reads=context.n_est_reads,
            provenance_fingerprint=provenance,
        )
        if replayed is None:
            raise QualityResolutionError("planned resolution state is no longer replayable")
        decision, envelope = replayed
        if (
            decision.state_fingerprint != row.get("state_fingerprint")
            or decision.support_fingerprint != row.get("support_fingerprint")
            or envelope.record_digest != row.get("action_envelope_record_digest")
        ):
            raise QualityResolutionError(
                "live resolution state or support differs from its target-free plan"
            )
        planned_actions = row.get("actions")
        if not isinstance(planned_actions, list):
            raise QualityResolutionError("planned resolution action registry is malformed")
        action_sample = deterministic_action_sample(
            decision,
            evaluated_actions=len(planned_actions),
            seed=row["environment_seed"],
        )
        chosen = action_sample.action_indices
        if chosen != tuple(action["action_index"] for action in planned_actions):
            raise QualityResolutionError(
                "live evaluated-action subset differs from its target-free plan"
            )
        for planned in planned_actions:
            action_index = int(planned["action_index"])
            bound = envelope.candidates[action_index]
            applied_digest = None
            if not bound.opcode.is_terminal:
                applied = apply_envelope_action(envelope, action_index)
                applied.verify_against(envelope)
                applied_digest = applied.record_digest
            if (
                bound.payload_key != planned.get("payload_key")
                or bound.opcode.value != planned.get("opcode")
                or bound.payload_digest != planned.get("selected_payload_digest")
                or applied_digest != planned.get("applied_action_record_digest")
            ):
                raise QualityResolutionError(
                    "live action identity differs from its target-free plan"
                )


def _compact_projection(outcome: ContinuationResult) -> dict[str, object]:
    receipt = outcome.receipt
    if not isinstance(receipt, Mapping):
        raise QualityResolutionError("quality continuation omitted its exact receipt")
    validate_continuation_receipt(receipt)
    initializer_binding = receipt.get("initializer_binding")
    if not isinstance(initializer_binding, Mapping):
        raise QualityResolutionError("quality continuation omitted its initializer-bank binding")
    try:
        validate_quality_initializer_binding(initializer_binding)
    except ValueError as error:
        raise QualityResolutionError(str(error)) from error
    evidence = receipt.get("terminal_evidence")
    if outcome.returned_valid:
        if not isinstance(evidence, Mapping):
            raise QualityResolutionError("valid continuation omitted terminal evidence")
        embedding = evidence.get("embedding")
        programs = evidence.get("programs")
        if not isinstance(embedding, list) or not isinstance(programs, list) or len(programs) != 4:
            raise QualityResolutionError("valid continuation has malformed compact evidence")
        embedding_digest: str | None = content_digest(embedding)
        program_digests = [content_digest(program) for program in programs]
        selected_index = evidence.get("selected_index")
        if type(selected_index) is not int or not 0 <= selected_index < 4:
            raise QualityResolutionError("valid continuation selected index is invalid")
    else:
        if evidence is not None:
            raise QualityResolutionError(
                "invalid continuation unexpectedly carries terminal evidence"
            )
        embedding_digest = None
        program_digests = []
        selected_index = None
    return {
        "compiled_program_digests": program_digests,
        "continuation_receipt_digest": _digest(
            receipt.get("record_digest"), "continuation receipt digest"
        ),
        "initializer_binding_record_digest": _digest(
            initializer_binding.get("record_digest"),
            "initializer binding record digest",
        ),
        "returned_valid": outcome.returned_valid,
        "reward": float(outcome.reward if outcome.returned_valid else 0.0),
        "selected_embedding_digest": embedding_digest,
        "selected_strength_index": selected_index,
    }


def _execute_work(
    work: _DeltaWork,
    *,
    task: PreparedTask,
    context: Context,
    selector: StrengthSelector,
    plan_sha256: str,
    plan_record_digest: str,
    authority: QualityResolutionExecutionAuthority,
    initializer_bank: InitializerSnapshotBank,
    initializer_bank_manifest_sha256: str,
    initializer_bank_contract: Mapping[str, object],
    allow_test_initializer_bank: bool,
    continuation_runner: Callable[..., ContinuationResult | None],
) -> dict[str, object]:
    outcome = continuation_runner(
        task.task,
        context,
        initializer=None,
        initializer_bank=initializer_bank,
        expected_initializer_bank_manifest_sha256=initializer_bank_manifest_sha256,
        initializer_bank_episode_index=work.row["initializer_bank_episode_index"],
        prepared_task=task,
        target_access=authority.target_access,
        ground_partition_receipt=authority.ground_partition_receipt,
        allow_test_initializer_bank=allow_test_initializer_bank,
        selector=selector,
        prefix=(*work.row["prefix"], work.action["action_index"]),
        policy=random_masked_policy,
        seed=work.row["environment_seed"],
        continuation_seed=work.continuation_seed,
        reward_reads=context.n_est_reads,
        max_steps=CONTINUATION_MAX_STEPS,
    )
    if outcome is None:
        raise RuntimeError("a planned legal quality action failed exact continuation replay")
    compact = _compact_projection(outcome)
    receipt = outcome.receipt
    if (
        not isinstance(receipt, Mapping)
        or receipt.get("continuation_seed") != work.continuation_seed
    ):
        raise QualityResolutionError("continuation receipt differs from its planned seed")
    binding = receipt.get("initializer_binding")
    if (
        not isinstance(binding, Mapping)
        or binding.get("protocol") != QUALITY_INITIALIZER_BANK_PROTOCOL
        or binding.get("manifest_sha256") != initializer_bank_manifest_sha256
        or binding.get("bank_contract_record_digest")
        != initializer_bank_contract.get("record_digest")
        or binding.get("episode_schedule_index") != work.row.get("initializer_bank_episode_index")
        or binding.get("bootstrap_record_digest")
        != work.row.get("initializer_bootstrap_record_digest")
    ):
        raise QualityResolutionError(
            "continuation initializer binding differs from planned K=2 episode"
        )
    payload: dict[str, object] = {
        "action_index": work.action["action_index"],
        "action_payload_digest": work.action["selected_payload_digest"],
        "base_lineage": work.row["base_lineage"],
        "continuation_index": work.continuation_index,
        "continuation_seed": work.continuation_seed,
        "execution_identity_digest": authority.execution_identity_digest,
        "instance_id": work.row["instance_id"],
        "plan_record_digest": plan_record_digest,
        "plan_sha256": plan_sha256,
        "row_id": work.row["row_id"],
        "schema": QUALITY_RESOLUTION_DELTA_ROW_SCHEMA,
        "schema_version": QUALITY_RESOLUTION_DELTA_ROW_VERSION,
        "shard_index": work.shard_index,
        "stage_index": work.stage_index,
        "state_fingerprint": work.row["state_fingerprint"],
        "task_id": work.row["task_id"],
        **compact,
    }
    return {**payload, "record_digest": content_digest(payload)}


def _resume_identity(
    *,
    plan_sha256: str,
    plan_record_digest: str,
    stage: Mapping[str, object],
    shard: Mapping[str, object],
    context_digest: str,
    authority: QualityResolutionExecutionAuthority,
    initializer_bank_contract: Mapping[str, object],
    work: Sequence[_DeltaWork],
) -> dict[str, object]:
    payload = {
        "context_digest": context_digest,
        "execution_identity_digest": authority.execution_identity_digest,
        "expected_work_digest": content_digest([item.key for item in work]),
        "expected_work_items": len(work),
        "initializer_bank_contract_record_digest": initializer_bank_contract["record_digest"],
        "initializer_bank_manifest_sha256": initializer_bank_contract["manifest_sha256"],
        "plan_record_digest": plan_record_digest,
        "plan_sha256": plan_sha256,
        "quality_authority_record_digest": authority.quality_authority["record_digest"],
        "schema": QUALITY_RESOLUTION_RESUME_SCHEMA,
        "schema_version": QUALITY_RESOLUTION_RESUME_VERSION,
        "shard_index": shard["shard_index"],
        "stage_index": stage["stage_index"],
        "target_access_record_digest": authority.target_access["record_digest"],
    }
    return {**payload, "record_digest": content_digest(payload)}


def _load_resume_record(path: Path, expected: Mapping[str, object]) -> dict[str, Any]:
    try:
        _resolved, raw = read_regular_file(path, "quality-resolution resume journal")
    except (ExactConformanceError, OSError) as error:
        raise QualityResolutionError(str(error)) from error
    record = _strict_json(raw, "quality-resolution resume journal")
    _verify_record(record, "quality-resolution resume journal")
    if record != expected:
        raise QualityResolutionError("resume journal belongs to another delta execution")
    return record


def _validate_delta_row(
    row: Mapping[str, object],
    work: _DeltaWork,
    *,
    plan_sha256: str,
    plan_record_digest: str,
    execution_identity_digest: str,
) -> None:
    if set(row) != _DELTA_ROW_FIELDS:
        raise QualityResolutionError("quality-resolution delta-row schema differs")
    _verify_record(row, "quality-resolution delta row")
    if (
        row.get("schema") != QUALITY_RESOLUTION_DELTA_ROW_SCHEMA
        or row.get("schema_version") != QUALITY_RESOLUTION_DELTA_ROW_VERSION
        or row.get("plan_sha256") != plan_sha256
        or row.get("plan_record_digest") != plan_record_digest
        or row.get("execution_identity_digest") != execution_identity_digest
        or row.get("stage_index") != work.stage_index
        or row.get("shard_index") != work.shard_index
        or row.get("row_id") != work.row["row_id"]
        or row.get("base_lineage") != work.row["base_lineage"]
        or row.get("task_id") != work.row["task_id"]
        or row.get("instance_id") != work.row["instance_id"]
        or row.get("state_fingerprint") != work.row["state_fingerprint"]
        or row.get("action_index") != work.action["action_index"]
        or row.get("action_payload_digest") != work.action["selected_payload_digest"]
        or row.get("continuation_index") != work.continuation_index
        or row.get("continuation_seed") != work.continuation_seed
    ):
        raise QualityResolutionError("quality-resolution delta row differs from planned work")
    _digest(row.get("continuation_receipt_digest"), "continuation receipt digest")
    _digest(
        row.get("initializer_binding_record_digest"),
        "initializer binding record digest",
    )
    reward = row.get("reward")
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(float(reward))
        or not 0.0 <= float(reward) <= 1.0
        or type(row.get("returned_valid")) is not bool
    ):
        raise QualityResolutionError("quality-resolution delta reward is invalid")
    if row["returned_valid"]:
        _digest(row.get("selected_embedding_digest"), "selected embedding digest")
        programs = row.get("compiled_program_digests")
        if not isinstance(programs, list) or len(programs) != 4:
            raise QualityResolutionError("valid delta row requires four compiled programs")
        for digest in programs:
            _digest(digest, "compiled program digest")
        selected = row.get("selected_strength_index")
        if type(selected) is not int or not 0 <= selected < 4:
            raise QualityResolutionError("valid delta row selected strength is invalid")
    elif (
        float(reward) != 0.0
        or row.get("selected_embedding_digest") is not None
        or row.get("compiled_program_digests") != []
        or row.get("selected_strength_index") is not None
    ):
        raise QualityResolutionError("invalid delta row carries terminal quality evidence")


def _load_journal_row(
    path: Path,
    work: _DeltaWork,
    *,
    plan_sha256: str,
    plan_record_digest: str,
    execution_identity_digest: str,
) -> dict[str, Any]:
    try:
        _resolved, raw = read_regular_file(path, "quality-resolution resume row")
    except (ExactConformanceError, OSError) as error:
        raise QualityResolutionError(str(error)) from error
    row = _strict_json(raw, "quality-resolution resume row")
    _validate_delta_row(
        row,
        work,
        plan_sha256=plan_sha256,
        plan_record_digest=plan_record_digest,
        execution_identity_digest=execution_identity_digest,
    )
    return row


def _manifest(
    records: Sequence[Mapping[str, object]],
    records_raw: bytes,
    *,
    plan_sha256: str,
    plan_record_digest: str,
    stage: Mapping[str, object],
    shard: Mapping[str, object],
    context_digest: str,
    authority: QualityResolutionExecutionAuthority,
    initializer_bank_contract: Mapping[str, object],
) -> dict[str, object]:
    lower, upper = stage["continuation_range"]
    identities = authority.identity_fields()
    payload: dict[str, object] = {
        "assigned_lineages": list(shard["lineages"]),
        "assigned_row_ids": list(shard["row_ids"]),
        "assigned_task_ids": list(shard["task_ids"]),
        "complete": True,
        "context_digest": context_digest,
        "cumulative_continuation_range": [0, upper],
        "delta_continuation_range": [lower, upper],
        "initializer_bank_contract_record_digest": initializer_bank_contract["record_digest"],
        "initializer_bank_manifest_sha256": initializer_bank_contract["manifest_sha256"],
        **identities,
        "plan_record_digest": plan_record_digest,
        "plan_sha256": plan_sha256,
        "record_count": len(records),
        "record_set_digest": content_digest([row["record_digest"] for row in records]),
        "records_sha256": hashlib.sha256(records_raw).hexdigest(),
        "schema": QUALITY_RESOLUTION_DELTA_SCHEMA,
        "schema_version": QUALITY_RESOLUTION_DELTA_VERSION,
        "shard_count": shard["shard_count"],
        "shard_index": shard["shard_index"],
        "stage_index": stage["stage_index"],
    }
    return {**payload, "record_digest": content_digest(payload)}


def run_quality_resolution_delta_shard(
    plan: QualityResolutionPlan,
    *,
    expected_plan_sha256: str,
    stage_index: int,
    shard_index: int,
    prepared: Sequence[PreparedTask],
    target_records: Sequence[Mapping[str, object]],
    authority: QualityResolutionExecutionAuthority,
    context: Context,
    selector: StrengthSelector,
    output_directory: str | Path,
    initializer_bank: InitializerSnapshotBank | None = None,
    expected_initializer_bank_manifest_sha256: str | None = None,
    allow_test_initializer_bank: bool = False,
    continuation_runner: Callable[..., ContinuationResult | None] = run_continuation,
) -> QualityResolutionDeltaArtifact:
    """Run or resume one exact immutable delta; never execute earlier prefix indices."""

    plan_record, plan_sha256 = _plan_identity(plan, expected_plan_sha256)
    initializer_bank_contract = _initializer_bank_identity(
        plan_record,
        initializer_bank,
        expected_initializer_bank_manifest_sha256,
        allow_test_bank=allow_test_initializer_bank,
    )
    assert isinstance(initializer_bank, InitializerSnapshotBank)
    if not isinstance(authority, QualityResolutionExecutionAuthority):
        raise TypeError("authority must be QualityResolutionExecutionAuthority")
    authority.validate_against_plan(plan)
    _require_registered_continuation_runner(
        continuation_runner,
        plan_record=plan_record,
        authority=authority,
    )
    _snapshot, context_digest = _context_identity(context, plan_record)
    stage, shard, work = _delta_work(plan_record, stage_index, shard_index)
    tasks = _target_tasks(prepared, work, authority, target_records)
    _validate_planned_rows(
        work,
        tasks,
        context,
        selector,
        str(plan_record["prepared_corpus"]["manifest_sha256"]),
        initializer_bank,
        str(initializer_bank_contract["manifest_sha256"]),
        authority,
        allow_test_initializer_bank,
    )

    destination = Path(output_directory)
    if destination.exists():
        raise FileExistsError(f"quality-resolution delta already exists: {destination}")
    if not destination.parent.is_dir():
        raise FileNotFoundError(f"quality-resolution delta parent is missing: {destination.parent}")
    partial = destination.parent / f".{destination.name}.partial"
    if partial.is_symlink():
        raise QualityResolutionError("quality-resolution resume directory may not be a symlink")
    partial.mkdir(exist_ok=True)
    if not partial.is_dir():
        raise QualityResolutionError("quality-resolution resume root is not a directory")
    partial_entries = {path.name: path for path in partial.iterdir()}
    if not set(partial_entries) <= {"journal.json", "records"}:
        raise QualityResolutionError("resume directory contains unregistered entries")
    if any(path.is_symlink() for path in partial_entries.values()):
        raise QualityResolutionError("resume directory contains a symbolic link")
    records_directory = partial / "records"
    records_directory.mkdir(exist_ok=True)
    if records_directory.is_symlink() or not records_directory.is_dir():
        raise QualityResolutionError("quality-resolution resume records root is invalid")
    journal = _resume_identity(
        plan_sha256=plan_sha256,
        plan_record_digest=plan_record["record_digest"],
        stage=stage,
        shard=shard,
        context_digest=context_digest,
        authority=authority,
        initializer_bank_contract=initializer_bank_contract,
        work=work,
    )
    journal_path = partial / "journal.json"
    if journal_path.exists():
        _load_resume_record(journal_path, journal)
    else:
        _publish_file(journal_path, journal)

    records: list[dict[str, object]] = []
    expected_resume_names = {f"{item.work_index:08d}.json" for item in work}
    observed_resume = tuple(records_directory.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in observed_resume):
        raise QualityResolutionError("resume evidence must contain regular files only")
    observed_resume_names = {path.name for path in observed_resume}
    if not observed_resume_names <= expected_resume_names:
        raise QualityResolutionError("resume directory contains unregistered evidence rows")
    for item in work:
        record_path = records_directory / f"{item.work_index:08d}.json"
        if record_path.exists():
            record = _load_journal_row(
                record_path,
                item,
                plan_sha256=plan_sha256,
                plan_record_digest=plan_record["record_digest"],
                execution_identity_digest=authority.execution_identity_digest,
            )
            replayed = _execute_work(
                item,
                task=tasks[str(item.row["task_id"])],
                context=context,
                selector=selector,
                plan_sha256=plan_sha256,
                plan_record_digest=plan_record["record_digest"],
                authority=authority,
                initializer_bank=initializer_bank,
                initializer_bank_manifest_sha256=str(initializer_bank_contract["manifest_sha256"]),
                initializer_bank_contract=initializer_bank_contract,
                allow_test_initializer_bank=allow_test_initializer_bank,
                continuation_runner=continuation_runner,
            )
            if record != replayed:
                raise QualityResolutionError(
                    "resume row differs from exact authenticated continuation replay"
                )
        else:
            record = _execute_work(
                item,
                task=tasks[str(item.row["task_id"])],
                context=context,
                selector=selector,
                plan_sha256=plan_sha256,
                plan_record_digest=plan_record["record_digest"],
                authority=authority,
                initializer_bank=initializer_bank,
                initializer_bank_manifest_sha256=str(initializer_bank_contract["manifest_sha256"]),
                initializer_bank_contract=initializer_bank_contract,
                allow_test_initializer_bank=allow_test_initializer_bank,
                continuation_runner=continuation_runner,
            )
            _validate_delta_row(
                record,
                item,
                plan_sha256=plan_sha256,
                plan_record_digest=plan_record["record_digest"],
                execution_identity_digest=authority.execution_identity_digest,
            )
            _publish_file(record_path, record)
        records.append(record)

    records_raw = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
    manifest = _manifest(
        records,
        records_raw,
        plan_sha256=plan_sha256,
        plan_record_digest=plan_record["record_digest"],
        stage=stage,
        shard=shard,
        context_digest=context_digest,
        authority=authority,
        initializer_bank_contract=initializer_bank_contract,
    )
    try:
        # mkdir is the portable atomic no-replace operation for a directory.  Publish the
        # manifest last so a crash can leave only an incomplete, fail-closed destination.
        destination.mkdir()
    except FileExistsError:
        raise FileExistsError(f"quality-resolution delta already exists: {destination}") from None
    try:
        publish_new_file(destination / "records.jsonl", records_raw)
        publish_new_file(destination / "manifest.json", canonical_json_bytes(manifest) + b"\n")
    except ExactConformanceError as error:
        raise QualityResolutionError(str(error)) from error
    shutil.rmtree(partial)
    manifest_sha256 = hashlib.sha256((destination / "manifest.json").read_bytes()).hexdigest()
    return QualityResolutionDeltaArtifact(
        root=destination,
        manifest=_freeze(manifest),  # type: ignore[arg-type]
        records=tuple(_freeze(record) for record in records),  # type: ignore[arg-type]
        manifest_sha256=manifest_sha256,
    )


def _load_delta_records(
    root: str | Path,
    expected_manifest_sha256: str,
    *,
    plan: QualityResolutionPlan,
    expected_plan_sha256: str,
    allow_test_initializer_bank: bool = False,
) -> QualityResolutionDeltaArtifact:
    plan_record, plan_sha256 = _plan_identity(plan, expected_plan_sha256)
    production = plan_record.get("production_plan")
    initializer_bank_contract = (
        production.get("initializer_bank") if isinstance(production, Mapping) else None
    )
    if not isinstance(initializer_bank_contract, Mapping):
        raise QualityResolutionError("quality-resolution plan omits its initializer-bank contract")
    try:
        validate_quality_initializer_bank_contract(
            initializer_bank_contract,
            require_publication=not allow_test_initializer_bank,
        )
    except ValueError as error:
        raise QualityResolutionError(str(error)) from error
    directory = Path(root)
    if directory.is_symlink() or not directory.is_dir():
        raise QualityResolutionError("quality-resolution delta root must be a real directory")
    entries = tuple(directory.iterdir())
    if {path.name for path in entries} != {"manifest.json", "records.jsonl"} or any(
        path.is_symlink() or not path.is_file() for path in entries
    ):
        raise QualityResolutionError(
            "quality-resolution delta directory inventory differs from its exact schema"
        )
    manifest, _raw, manifest_sha256 = _read_pinned_json(
        directory / "manifest.json",
        expected_manifest_sha256,
        "quality-resolution delta manifest",
    )
    if set(manifest) != _DELTA_MANIFEST_FIELDS:
        raise QualityResolutionError("quality-resolution delta manifest schema differs")
    _verify_record(manifest, "quality-resolution delta manifest")
    if (
        manifest.get("schema") != QUALITY_RESOLUTION_DELTA_SCHEMA
        or manifest.get("schema_version") != QUALITY_RESOLUTION_DELTA_VERSION
        or manifest.get("complete") is not True
        or manifest.get("plan_sha256") != plan_sha256
        or manifest.get("plan_record_digest") != plan_record["record_digest"]
        or manifest.get("initializer_bank_manifest_sha256")
        != initializer_bank_contract.get("manifest_sha256")
        or manifest.get("initializer_bank_contract_record_digest")
        != initializer_bank_contract.get("record_digest")
    ):
        raise QualityResolutionError("delta manifest differs from the pinned resolution plan")
    try:
        _resolved, records_raw = read_regular_file(
            directory / "records.jsonl", "quality-resolution delta records"
        )
    except (ExactConformanceError, OSError) as error:
        raise QualityResolutionError(str(error)) from error
    if hashlib.sha256(records_raw).hexdigest() != manifest.get("records_sha256"):
        raise QualityResolutionError("quality-resolution delta records changed after sealing")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(records_raw.splitlines(keepends=True), start=1):
        record = _strict_json(line, f"quality-resolution delta row {line_number}")
        _verify_record(record, f"quality-resolution delta row {line_number}")
        records.append(record)
    if len(records) != manifest.get("record_count") or content_digest(
        [record["record_digest"] for record in records]
    ) != manifest.get("record_set_digest"):
        raise QualityResolutionError("quality-resolution delta record census mismatch")
    stage, shard, work = _delta_work(
        plan_record,
        manifest.get("stage_index"),  # type: ignore[arg-type]
        manifest.get("shard_index"),  # type: ignore[arg-type]
    )
    expected_keys = [item.key for item in work]
    observed_keys = [
        f"{record['row_id']}:{int(record['action_index']):08d}:"
        f"{int(record['continuation_index']):08d}"
        for record in records
    ]
    if observed_keys != expected_keys:
        raise QualityResolutionError("quality-resolution delta work coverage differs from plan")
    for item, record in zip(work, records, strict=True):
        _validate_delta_row(
            record,
            item,
            plan_sha256=plan_sha256,
            plan_record_digest=plan_record["record_digest"],
            execution_identity_digest=manifest.get("execution_identity_digest"),  # type: ignore[arg-type]
        )
    expected_manifest = dict(
        _manifest(
            records,
            records_raw,
            plan_sha256=plan_sha256,
            plan_record_digest=plan_record["record_digest"],
            stage=stage,
            shard=shard,
            context_digest=manifest.get("context_digest"),  # type: ignore[arg-type]
            authority=_authority_stub_from_manifest(manifest),
            initializer_bank_contract=initializer_bank_contract,
        )
    )
    # The stub reconstruction above deliberately validates identity field shape.  The original
    # manifest remains authoritative for those externally authenticated digests.
    for field in (
        "assigned_lineages",
        "assigned_row_ids",
        "assigned_task_ids",
        "cumulative_continuation_range",
        "delta_continuation_range",
        "record_count",
        "record_set_digest",
        "records_sha256",
        "shard_count",
        "shard_index",
        "stage_index",
    ):
        if expected_manifest[field] != manifest.get(field):
            raise QualityResolutionError(f"delta manifest {field} differs from its reconstruction")
    return QualityResolutionDeltaArtifact(
        root=directory,
        manifest=_freeze(manifest),  # type: ignore[arg-type]
        records=tuple(_freeze(record) for record in records),  # type: ignore[arg-type]
        manifest_sha256=manifest_sha256,
    )


class _ManifestAuthorityStub:
    def __init__(self, manifest: Mapping[str, object]) -> None:
        self.execution_identity_digest = manifest["execution_identity_digest"]
        self.quality_authority = {"record_digest": manifest["quality_authority_record_digest"]}
        self.target_access = {"record_digest": manifest["target_access_record_digest"]}
        self.ground_partition_receipt = {
            "record_digest": manifest["ground_partition_receipt_record_digest"]
        }
        self.execution_identity = manifest["execution_identity"]
        self._ground_sha = manifest["ground_partition_receipt_sha256"]

    def identity_fields(self) -> dict[str, object]:
        return {
            "execution_identity": self.execution_identity,
            "execution_identity_digest": self.execution_identity_digest,
            "ground_partition_receipt_record_digest": self.ground_partition_receipt[
                "record_digest"
            ],
            "ground_partition_receipt_sha256": self._ground_sha,
            "quality_authority_record_digest": self.quality_authority["record_digest"],
            "target_access_record_digest": self.target_access["record_digest"],
        }


def _authority_stub_from_manifest(manifest: Mapping[str, object]) -> Any:
    return _ManifestAuthorityStub(manifest)


def load_cumulative_resolution_delta_records(
    artifacts: Sequence[tuple[str | Path, str]],
    *,
    plan: QualityResolutionPlan,
    expected_plan_sha256: str,
    allow_test_initializer_bank: bool = False,
) -> tuple[Mapping[str, object], ...]:
    """Load one shard's exact stage prefix and return [0,C) records in canonical order."""

    if not artifacts:
        raise QualityResolutionError("cumulative resolution load needs at least one delta")
    loaded = [
        _load_delta_records(
            root,
            pin,
            plan=plan,
            expected_plan_sha256=expected_plan_sha256,
            allow_test_initializer_bank=allow_test_initializer_bank,
        )
        for root, pin in artifacts
    ]
    loaded.sort(key=lambda artifact: int(artifact.manifest["stage_index"]))
    stages = [int(artifact.manifest["stage_index"]) for artifact in loaded]
    if stages != list(range(len(stages))):
        raise QualityResolutionError("resolution deltas do not form a complete stage prefix")
    shard_indices = {int(artifact.manifest["shard_index"]) for artifact in loaded}
    if len(shard_indices) != 1:
        raise QualityResolutionError("cumulative resolution deltas mix shard identities")
    expected_lower = 0
    records: list[Mapping[str, object]] = []
    for artifact in loaded:
        lower, upper = artifact.manifest["delta_continuation_range"]
        if lower != expected_lower:
            raise QualityResolutionError("resolution deltas are not a nested continuation prefix")
        expected_lower = int(upper)
        records.extend(artifact.records)
    records.sort(
        key=lambda row: (
            str(row["row_id"]),
            int(row["action_index"]),
            int(row["continuation_index"]),
        )
    )
    groups: dict[tuple[str, int], list[int]] = {}
    for row in records:
        groups.setdefault((str(row["row_id"]), int(row["action_index"])), []).append(
            int(row["continuation_index"])
        )
    if any(indices != list(range(expected_lower)) for indices in groups.values()):
        raise QualityResolutionError("cumulative records do not cover exact [0,C) prefixes")
    return tuple(records)


def verify_quality_resolution_delta_shard(
    delta_root: str | Path,
    *,
    expected_manifest_sha256: str,
    plan: QualityResolutionPlan,
    expected_plan_sha256: str,
    prepared: Sequence[PreparedTask],
    target_records: Sequence[Mapping[str, object]],
    authority: QualityResolutionExecutionAuthority,
    context: Context,
    selector: StrengthSelector,
    initializer_bank: InitializerSnapshotBank | None = None,
    expected_initializer_bank_manifest_sha256: str | None = None,
    allow_test_initializer_bank: bool = False,
    verifier_identity_path: str | Path,
    expected_verifier_identity_sha256: str,
    output_path: str | Path,
    continuation_runner: Callable[..., ContinuationResult | None] = run_continuation,
) -> dict[str, object]:
    """Independently replay a sealed delta and publish an immutable comparison receipt."""

    artifact = _load_delta_records(
        delta_root,
        expected_manifest_sha256,
        plan=plan,
        expected_plan_sha256=expected_plan_sha256,
        allow_test_initializer_bank=allow_test_initializer_bank,
    )
    plan_record, plan_sha256 = _plan_identity(plan, expected_plan_sha256)
    initializer_bank_contract = _initializer_bank_identity(
        plan_record,
        initializer_bank,
        expected_initializer_bank_manifest_sha256,
        allow_test_bank=allow_test_initializer_bank,
    )
    assert isinstance(initializer_bank, InitializerSnapshotBank)
    authority.validate_against_plan(plan)
    _require_registered_continuation_runner(
        continuation_runner,
        plan_record=plan_record,
        authority=authority,
    )
    _snapshot, context_digest = _context_identity(context, plan_record)
    if (
        artifact.manifest["execution_identity_digest"] != authority.execution_identity_digest
        or artifact.manifest["quality_authority_record_digest"]
        != authority.quality_authority["record_digest"]
        or artifact.manifest["target_access_record_digest"]
        != authority.target_access["record_digest"]
        or artifact.manifest["context_digest"] != context_digest
    ):
        raise QualityResolutionError("verification authority differs from delta generation")
    verifier, verifier_identity_sha256 = _load_verifier_identity(
        verifier_identity_path,
        expected_verifier_identity_sha256,
        plan_record=plan_record,
    )
    if (
        verifier["verification_runtime_sha256"]
        != authority.execution_identity["runtime_sha256"]
    ):
        raise QualityResolutionError(
            "quality-resolution verifier runtime differs from the execution runtime"
        )
    verifier_digest = _digest(verifier["record_digest"], "verifier identity record digest")
    stage, _shard, work = _delta_work(
        plan_record,
        int(artifact.manifest["stage_index"]),
        int(artifact.manifest["shard_index"]),
    )
    tasks = _target_tasks(prepared, work, authority, target_records)
    _validate_planned_rows(
        work,
        tasks,
        context,
        selector,
        str(plan_record["prepared_corpus"]["manifest_sha256"]),
        initializer_bank,
        str(initializer_bank_contract["manifest_sha256"]),
        authority,
        allow_test_initializer_bank,
    )
    observed_by_key = {
        (
            str(row["row_id"]),
            int(row["action_index"]),
            int(row["continuation_index"]),
        ): row
        for row in artifact.records
    }
    mismatches: list[dict[str, object]] = []
    for item in work:
        replay = _execute_work(
            item,
            task=tasks[str(item.row["task_id"])],
            context=context,
            selector=selector,
            plan_sha256=plan_sha256,
            plan_record_digest=plan_record["record_digest"],
            authority=authority,
            initializer_bank=initializer_bank,
            initializer_bank_manifest_sha256=str(initializer_bank_contract["manifest_sha256"]),
            initializer_bank_contract=initializer_bank_contract,
            allow_test_initializer_bank=allow_test_initializer_bank,
            continuation_runner=continuation_runner,
        )
        key = (str(item.row["row_id"]), int(item.action["action_index"]), item.continuation_index)
        observed = observed_by_key[key]
        if replay != _jsonable(observed):
            mismatches.append(
                {
                    "action_index": key[1],
                    "continuation_index": key[2],
                    "observed_record_digest": observed["record_digest"],
                    "replayed_record_digest": replay["record_digest"],
                    "row_id": key[0],
                }
            )
    payload: dict[str, object] = {
        "delta_manifest_record_digest": artifact.manifest["record_digest"],
        "delta_manifest_sha256": artifact.manifest_sha256,
        "execution_identity_digest": authority.execution_identity_digest,
        "matching_count": len(work) - len(mismatches),
        "mismatches": mismatches,
        "pass": not mismatches,
        "plan_record_digest": plan_record["record_digest"],
        "plan_sha256": plan_sha256,
        "replayed_count": len(work),
        "schema": QUALITY_RESOLUTION_VERIFICATION_SCHEMA,
        "schema_version": QUALITY_RESOLUTION_VERIFICATION_VERSION,
        "shard_index": artifact.manifest["shard_index"],
        "stage_index": stage["stage_index"],
        "verifier_identity": verifier,
        "verifier_identity_digest": verifier_digest,
        "verifier_identity_sha256": verifier_identity_sha256,
    }
    receipt = {**payload, "record_digest": content_digest(payload)}
    _publish_file(Path(output_path), receipt)
    return receipt


__all__ = [
    "QUALITY_RESOLUTION_DELTA_SCHEMA",
    "QUALITY_RESOLUTION_DELTA_VERSION",
    "QUALITY_RESOLUTION_VERIFICATION_SCHEMA",
    "QUALITY_RESOLUTION_VERIFICATION_VERSION",
    "QUALITY_RESOLUTION_VERIFIER_IDENTITY_SCHEMA",
    "QUALITY_RESOLUTION_VERIFIER_IDENTITY_VERSION",
    "QualityResolutionDeltaArtifact",
    "QualityResolutionExecutionAuthority",
    "load_cumulative_resolution_delta_records",
    "publish_quality_resolution_verifier_identity",
    "run_quality_resolution_delta_shard",
    "verify_quality_resolution_delta_shard",
]
