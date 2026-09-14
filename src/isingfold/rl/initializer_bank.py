"""Authenticated deployment-initializer snapshot banks for PPO training.

The learned complete system is deployed after a fresh LAC initializer run.  A PPO run
must therefore not train only on the hand-authored CandidateBank incumbents.  This module
seals a train-only episode schedule and the exact initializer runtime/configuration before
precomputing those deployments.  Evaluator targets are deliberately outside every bank
artifact and API used to construct the bank.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Hashable, Mapping, Sequence

from isingfold.rl.complete_system import (
    COMPLETE_POPULATION_TASK_DIGEST_SCHEMA,
    COMPLETE_POLICY_RESTART_MODE,
    LAC_INITIALIZER_METHOD_ID,
    CompleteInitializerResult,
    CompleteSystemBackendError,
    CompleteSystemConfig,
    InitializerAttemptReceipt,
    LACMinorminerInitializerBackend,
    PartialWorkVector,
    task_population_digest,
)
from isingfold.rl.contracts import (
    WORK_FIELDS,
    Context,
    Mode,
    RestartCacheSlot,
    WorkVector,
    stable_digest,
)
from isingfold.rl.data.import_embedbench import PREPARED_SCHEMA_VERSION_V4
from isingfold.rl.data.prepared import PreparedTask
from isingfold.rl.env import EmbeddingEnv, EmbeddingTask, StrengthSelector
from isingfold.rl.external import BackendIdentity, SearchStatus
from isingfold.rl.validate import p_embed

INITIALIZER_BANK_PLAN_SCHEMA = "isingfold.initializer-snapshot-bank-plan"
INITIALIZER_BANK_PLAN_VERSION = 2
INITIALIZER_BANK_PARTITION = "train"
INITIALIZER_BANK_CORPUS_SCOPE = "production-designed-v4"
INITIALIZER_BANK_SCHEDULE_RULE = "canonical-base-lineage-round-robin-v1"
INITIALIZER_BANK_WITHIN_LINEAGE_RULE = "seeded-offset-cyclic-instance-v1"
INITIALIZER_BANK_WITHIN_LINEAGE_SEED_DOMAIN = (
    "initializer-bank-within-base-lineage-instance-offset-v1"
)
INITIALIZER_BANK_SYSTEM_SEED_DOMAIN = "policy"
INITIALIZER_SNAPSHOT_SCHEMA = "isingfold.initializer-deployment-snapshot"
INITIALIZER_SNAPSHOT_VERSION = 2
INITIALIZER_BANK_MANIFEST_SCHEMA = "isingfold.initializer-snapshot-bank-manifest"
INITIALIZER_BANK_MANIFEST_VERSION = 2
INITIALIZER_BANK_ACCESS_SCHEMA = "isingfold.initializer-snapshot-bank-access"
INITIALIZER_BANK_ACCESS_VERSION = 2


class InitializerSnapshotUnavailable(RuntimeError):
    """The sealed deployment initializer failed for a scheduled training episode."""


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_digest(value: object, *, label: str) -> str:
    if not _is_digest(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    assert isinstance(value, str)
    return value


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("authenticated JSON mappings require string keys")
        return {key: _jsonable(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__} in initializer-bank identity")


def _freeze_json(value: object) -> object:
    normalized = _jsonable(value)
    if isinstance(normalized, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in normalized.items()})
    if isinstance(normalized, list):
        return tuple(_freeze_json(item) for item in normalized)
    return normalized


def _typed_identity(value: object) -> str:
    kind = type(value)
    return f"{kind.__module__}.{kind.__qualname__}:{value!r}"


def _pair(left: Hashable, right: Hashable) -> tuple[str, str]:
    first, second = _typed_identity(left), _typed_identity(right)
    return (first, second) if first <= second else (second, first)


def _public_task_payload(task: EmbeddingTask, *, instance_id: str) -> dict[str, object]:
    """Public deployment identity, intentionally excluding ground truth and incumbents."""

    clean = dataclasses.replace(
        task,
        name=instance_id,
        ground_energy=None,
        witness=None,
        initial_embedding=None,
    )
    return {
        "schema": "isingfold.initializer-bank-public-task",
        "schema_version": 1,
        "instance_id": instance_id,
        "base_lineage": task.lineage,
        "complete_population_task_digest_schema": COMPLETE_POPULATION_TASK_DIGEST_SCHEMA,
        "complete_population_task_digest": task_population_digest([clean]),
        "logical_node_order": [_typed_identity(node) for node in task.logical.nodes()],
        "host_node_order": [_typed_identity(node) for node in task.host.nodes()],
    }


def _public_task_digest(task: EmbeddingTask, *, instance_id: str) -> str:
    return stable_digest(_public_task_payload(task, instance_id=instance_id))


def public_task_digest(task: EmbeddingTask, *, instance_id: str) -> str:
    """Return the target-free public task identity used by every bootstrap bank."""

    return _public_task_digest(task, instance_id=instance_id)


_TARGET_ACCESS_FIELDS = frozenset(
    {
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
)
_GROUND_PARTITION_FIELDS = frozenset(
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
    }
)
_GROUND_TARGET_FIELDS = frozenset(
    {
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
)


def validate_target_task_binding(
    training_task: PreparedTask,
    *,
    target_access: Mapping[str, object],
    ground_partition_receipt: Mapping[str, object],
    expected_prepared_manifest_sha256: str,
) -> None:
    """Fail closed when joining a train-only quality target to a public bank row.

    Both receipt records have already crossed the external-authentication boundary in
    the CLI.  This adapter independently verifies their self-digests, partition/set
    relationship, and the exact evaluator-target row reconstructed from the typed task.
    """

    if not isinstance(training_task, PreparedTask):
        raise TypeError("target binding requires a typed PreparedTask")
    if not isinstance(target_access, Mapping) or set(target_access) != _TARGET_ACCESS_FIELDS:
        raise ValueError("target-access receipt fields differ from the authenticated schema")
    _verify_record(target_access, label="training target-access receipt")
    if (
        target_access.get("partition") != INITIALIZER_BANK_PARTITION
        or target_access.get("target_path") != "targets/train.jsonl"
        or target_access.get("prepared_manifest_sha256")
        != _require_digest(
            expected_prepared_manifest_sha256,
            label="initializer-bank prepared manifest",
        )
    ):
        raise ValueError("training target access differs from the bank corpus or partition")
    target_count = target_access.get("target_count")
    if isinstance(target_count, bool) or not isinstance(target_count, int) or target_count <= 0:
        raise ValueError("training target access has an invalid target count")
    for field in (
        "evidence_manifest_record_digest",
        "evidence_manifest_sha256",
        "prepared_manifest_record_digest",
        "publisher_attestation_digest",
        "target_authority_record_digest",
        "target_set_digest",
        "target_sha256",
    ):
        _require_digest(target_access.get(field), label=f"target-access {field}")
    opened_files = target_access.get("opened_files")
    if (
        not isinstance(opened_files, Sequence)
        or isinstance(opened_files, (str, bytes, bytearray))
        or not opened_files
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"authority_root", "role", "relative_path", "sha256"}
            or not _is_digest(item.get("sha256"))
            for item in opened_files
        )
    ):
        raise ValueError("training target access has invalid opened-file identities")
    if not any(
        item.get("authority_root") == "prepared"
        and item.get("role") == "evaluator-targets"
        and item.get("relative_path") == "targets/train.jsonl"
        and item.get("sha256") == target_access.get("target_sha256")
        for item in opened_files
        if isinstance(item, Mapping)
    ) or not any(
        item.get("authority_root") == "publisher"
        and item.get("role") == "quality-evidence-manifest"
        and item.get("sha256") == target_access.get("evidence_manifest_sha256")
        for item in opened_files
        if isinstance(item, Mapping)
    ):
        raise ValueError("training target access omits an authenticated source opening")

    if (
        not isinstance(ground_partition_receipt, Mapping)
        or set(ground_partition_receipt) != _GROUND_PARTITION_FIELDS
    ):
        raise ValueError("ground-partition receipt fields differ from the authenticated schema")
    _verify_record(ground_partition_receipt, label="training ground-partition receipt")
    if (
        ground_partition_receipt.get("schema")
        != "isingfold.ground-certificate-partition"
        or ground_partition_receipt.get("schema_version") != 1
        or ground_partition_receipt.get("protocol")
        != "isingfold-ground-certificate-isolated-runtime-v2"
        or ground_partition_receipt.get("partition") != INITIALIZER_BANK_PARTITION
        or ground_partition_receipt.get("authority") != target_access
    ):
        raise ValueError("ground-partition receipt differs from target-access authority")
    rows = ground_partition_receipt.get("targets")
    if (
        not isinstance(rows, Sequence)
        or isinstance(rows, (str, bytes, bytearray))
        or len(rows) != target_count
        or any(
            not isinstance(row, Mapping) or set(row) != _GROUND_TARGET_FIELDS
            for row in rows
        )
    ):
        raise ValueError("ground-partition target census differs from target access")
    row_pairs = []
    for raw_row in rows:
        assert isinstance(raw_row, Mapping)
        instance_id = raw_row.get("instance_id")
        target_digest = raw_row.get("target_record_digest")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError("ground-partition target has an invalid instance ID")
        row_pairs.append(
            {
                "instance_id": instance_id,
                "target_record_digest": _require_digest(
                    target_digest, label="ground-partition target record"
                ),
            }
        )
    if len({pair["instance_id"] for pair in row_pairs}) != len(row_pairs):
        raise ValueError("ground-partition target census repeats an instance")
    expected_set_digest = stable_digest(
        {
            "domain": "isingfold-partition-target-set-v1",
            "partition": INITIALIZER_BANK_PARTITION,
            "targets": sorted(row_pairs, key=lambda row: str(row["instance_id"])),
        }
    )
    if target_access.get("target_set_digest") != expected_set_digest:
        raise ValueError("ground-partition rows differ from the authenticated target set")
    census = ground_partition_receipt.get("census")
    if (
        not isinstance(census, Mapping)
        or census.get("target_count") != target_count
        or census.get("accepted_count") != target_count
        or census.get("instance_set_digest")
        != stable_digest(sorted(str(pair["instance_id"]) for pair in row_pairs))
    ):
        raise ValueError("ground-partition census differs from its authenticated rows")

    matching = [
        row for row in rows if isinstance(row, Mapping) and row.get("instance_id") == training_task.instance_id
    ]
    if len(matching) != 1:
        raise ValueError("training task has no unique authenticated ground target")
    target_row = matching[0]
    claimed = target_row.get("claimed_reference_energy")
    if (
        isinstance(claimed, bool)
        or not isinstance(claimed, (int, float))
        or not math.isfinite(float(claimed))
        or training_task.task.ground_energy is None
        or float(claimed) != float(training_task.task.ground_energy)
    ):
        raise ValueError("training task ground energy differs from authenticated target")
    if training_task.provenance is None or training_task.design_condition is None:
        raise ValueError("training target lacks prepared-v4 provenance and design binding")
    public_instance_record_digest = _require_digest(
        training_task.public_instance_record_digest,
        label="training public policy-instance record",
    )
    target_payload = {
        "certificate_digest": training_task.certificate_digest,
        "evaluator_protocol_digest": training_task.evaluator_protocol_digest,
        "instance_id": training_task.instance_id,
        "instance_record_digest": training_task.provenance.instance_record_digest,
        "learning_partition": INITIALIZER_BANK_PARTITION,
        "reference_energy": claimed,
        "reference_status": training_task.reference_status,
        "schema": "isingfold.evaluator-target",
        "schema_version": 2,
    }
    if (
        stable_digest(target_payload) != target_row.get("target_record_digest")
        or target_row.get("learning_partition") != INITIALIZER_BANK_PARTITION
        or target_row.get("status") != "accepted"
        or target_row.get("base_lineage_key")
        != training_task.design_condition.base_lineage_key
        or _jsonable(target_row.get("design_condition"))
        != _jsonable(training_task.design_condition)
        or target_row.get("public_instance_record_digest")
        != public_instance_record_digest
        or target_row.get("reference_status") != training_task.reference_status
        or target_row.get("artifact_sha256") != training_task.certificate_digest
    ):
        raise ValueError("training task quality fields differ from its exact target row")
    verifier_result = target_row.get("verifier_result")
    if not isinstance(verifier_result, Mapping):
        raise ValueError("training target has no authenticated verifier result")
    verifier_digest = _verify_record(
        verifier_result, label="training target verifier result"
    )
    if (
        verifier_digest != target_row.get("result_record_digest")
        or verifier_result.get("accepted") is not True
        or verifier_result.get("reason_code") != "accepted"
        or verifier_result.get("instance_id") != training_task.instance_id
        or verifier_result.get("claimed_reference_energy") != claimed
    ):
        raise ValueError("training target verifier result differs from the task")
    expected_quality = {
        "quality_attestation_digest": target_access.get(
            "publisher_attestation_digest"
        ),
        "quality_evidence_manifest_digest": target_access.get(
            "evidence_manifest_record_digest"
        ),
        "quality_evidence_manifest_sha256": target_access.get(
            "evidence_manifest_sha256"
        ),
        "quality_target_set_digest": target_access.get("target_set_digest"),
        "quality_target_count": target_count,
    }
    if any(getattr(training_task, name) != value for name, value in expected_quality.items()):
        raise ValueError("training task quality authority differs from target access")


def _domain_seed(root: int, domain: str, *parts: object) -> int:
    payload = {
        "domain": domain,
        "parts": [root, *parts],
        "schema_version": 1,
    }
    return int.from_bytes(
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()[:8],
        "big",
    ) & (2**31 - 1)


def _system_seed(training_seed: int, instance_id: str, lineage: str, episode: int) -> int:
    payload = {
        "base_seed": training_seed,
        "domain": INITIALIZER_BANK_SYSTEM_SEED_DOMAIN,
        "instance": instance_id,
        "lineage": lineage,
        "repetition": episode,
    }
    return int.from_bytes(
        hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()[:4],
        "big",
    ) % (2**31)


def _attempt_seed(system_seed: int, method_id: str, attempt_index: int) -> int:
    payload = f"{system_seed}:{method_id}:{attempt_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") % (2**31)


def _planned_draw(
    *,
    training_seed: int,
    episode_index: int,
    instance_id: str,
    lineage: str,
    slot_kind: str,
    slot_index: int,
    attempt_count: int,
    method_id: str,
) -> InitializerBankDraw:
    system_seed = _system_seed(
        training_seed,
        instance_id,
        lineage,
        _domain_seed(
            training_seed,
            f"initializer-bank-{slot_kind}-draw-v2",
            episode_index,
            slot_index,
        ),
    )
    attempt_seeds = tuple(
        _attempt_seed(system_seed, method_id, attempt_index)
        for attempt_index in range(attempt_count)
    )
    key_body = {
        "schema_version": INITIALIZER_BANK_PLAN_VERSION,
        "training_seed": training_seed,
        "conditional_episode_index": episode_index,
        "draw_index": slot_index,
        "slot_kind": slot_kind,
        "slot_index": slot_index,
        "instance_id": instance_id,
        "base_lineage": lineage,
        "system_seed": system_seed,
        "attempt_seeds": list(attempt_seeds),
    }
    return InitializerBankDraw(
        conditional_episode_index=episode_index,
        draw_index=slot_index,
        slot_kind=slot_kind,
        slot_index=slot_index,
        instance_id=instance_id,
        base_lineage=lineage,
        system_seed=system_seed,
        attempt_seeds=attempt_seeds,
        draw_key=stable_digest(key_body),
    )


def _validate_runtime_manifest(
    raw: Mapping[str, object], identity: BackendIdentity
) -> tuple[Mapping[str, object], str]:
    normalized = _jsonable(raw)
    if not isinstance(normalized, dict):
        raise ValueError("runtime implementation manifest must be an object")
    expected = {
        "schema",
        "schema_version",
        "native_extension_sha256",
        "python_source_sha256",
        "backend_info",
        "manifest_sha256",
    }
    if set(normalized) != expected:
        raise ValueError("runtime implementation manifest schema differs")
    if (
        normalized["schema"] != "lac-minorminer.runtime-implementation"
        or normalized["schema_version"] != 1
    ):
        raise ValueError("unsupported runtime implementation manifest")
    _require_digest(
        normalized["native_extension_sha256"], label="native extension identity"
    )
    sources = normalized["python_source_sha256"]
    if (
        not isinstance(sources, dict)
        or not sources
        or any(not isinstance(name, str) or not name for name in sources)
    ):
        raise ValueError("runtime Python source identity must be a nonempty object")
    for digest in sources.values():
        _require_digest(digest, label="runtime Python source identity")
    backend_info = normalized["backend_info"]
    if not isinstance(backend_info, dict):
        raise ValueError("runtime backend info must be an object")
    if backend_info.get("package_version") != identity.version:
        raise ValueError("runtime manifest package version differs from initializer identity")
    implementation_prefix = identity.implementation.rsplit(":runtime-sha256:", maxsplit=1)
    if len(implementation_prefix) != 2 or not implementation_prefix[0]:
        raise ValueError("initializer identity lacks a content-addressed runtime identity")
    if backend_info.get("backend") != implementation_prefix[0]:
        raise ValueError("runtime backend identity differs from initializer implementation")
    stated = _require_digest(
        normalized["manifest_sha256"], label="runtime implementation manifest digest"
    )
    body = {key: value for key, value in normalized.items() if key != "manifest_sha256"}
    if stable_digest(body) != stated:
        raise ValueError("runtime implementation manifest digest mismatch")
    if implementation_prefix[1] != stated:
        raise ValueError("initializer identity differs from runtime implementation manifest")
    frozen = _freeze_json(normalized)
    assert isinstance(frozen, Mapping)
    return frozen, stated


def _validate_backend_contract(
    initializer: object,
    config: CompleteSystemConfig,
    runtime_implementation_manifest: Mapping[str, object],
    *,
    allow_test_backend: bool,
) -> tuple[BackendIdentity, Mapping[str, object], str, bool]:
    identity = getattr(initializer, "identity", None)
    if not isinstance(identity, BackendIdentity):
        raise TypeError("initializer must expose a typed BackendIdentity")
    if identity.distribution != "lac-minorminer":
        raise ValueError("initializer backend must be lac-minorminer")
    if identity.method_id != LAC_INITIALIZER_METHOD_ID:
        raise ValueError("initializer method is not the registered LAC method")
    if identity.entrypoint != "lac_minorminer.find_embedding":
        raise ValueError("initializer entrypoint is not the registered LAC entrypoint")
    if config.initializer_backend != identity.distribution:
        raise ValueError("complete-system initializer backend differs from runtime")
    if config.initializer_method_id != identity.method_id:
        raise ValueError("complete-system initializer method differs from runtime")
    if config.expected_initializer_version != identity.version:
        raise ValueError("complete-system initializer version differs from runtime")
    frozen_manifest, manifest_digest = _validate_runtime_manifest(
        runtime_implementation_manifest, identity
    )
    production_backend = type(initializer) is LACMinorminerInitializerBackend
    if not production_backend and not allow_test_backend:
        raise TypeError("publication banks require the exact registered LAC backend type")
    return identity, frozen_manifest, manifest_digest, production_backend


def validate_initializer_backend_contract(
    initializer: object,
    config: CompleteSystemConfig,
    runtime_implementation_manifest: Mapping[str, object],
    *,
    allow_test_backend: bool = False,
) -> tuple[BackendIdentity, Mapping[str, object], str, bool]:
    """Validate the registered LAC runtime for train or validation bank builders."""

    return _validate_backend_contract(
        initializer,
        config,
        runtime_implementation_manifest,
        allow_test_backend=allow_test_backend,
    )


def lac_runtime_implementation_manifest(
    initializer: LACMinorminerInitializerBackend,
) -> Mapping[str, object]:
    """Return a verified read-only artifact manifest from the exact production adapter.

    ``complete_system`` currently keeps this value private because normal evaluation embeds
    it in every attempt receipt.  Bank planning needs the identity before the first call, so
    this narrow accessor performs an exact-type check and revalidates the content address.
    """

    if type(initializer) is not LACMinorminerInitializerBackend:
        raise TypeError("runtime manifest access requires the exact production LAC adapter")
    raw = getattr(initializer, "_implementation_manifest", None)
    if not isinstance(raw, Mapping):
        raise CompleteSystemBackendError(
            "production LAC adapter omitted its runtime implementation manifest"
        )
    frozen, _ = _validate_runtime_manifest(raw, initializer.identity)
    return frozen


@dataclass(frozen=True)
class InitializerBankTaskIdentity:
    instance_id: str
    base_lineage: str
    source_task_ids: tuple[str, ...]
    source_record_digests: tuple[str, ...]
    public_task_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.instance_id) is not str
            or not self.instance_id
            or type(self.base_lineage) is not str
            or not self.base_lineage
        ):
            raise ValueError("initializer bank task identity must be nonempty")
        source_task_ids = tuple(self.source_task_ids)
        source_record_digests = tuple(self.source_record_digests)
        if (
            not source_task_ids
            or len(source_task_ids) != len(source_record_digests)
            or any(type(value) is not str or not value for value in source_task_ids)
            or any(not _is_digest(value) for value in source_record_digests)
        ):
            raise ValueError("initializer bank task sources are malformed")
        _require_digest(self.public_task_digest, label="initializer bank public task")
        object.__setattr__(self, "source_task_ids", source_task_ids)
        object.__setattr__(
            self,
            "source_record_digests",
            source_record_digests,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "instance_id": self.instance_id,
            "base_lineage": self.base_lineage,
            "source_task_ids": list(self.source_task_ids),
            "source_record_digests": list(self.source_record_digests),
            "public_task_digest": self.public_task_digest,
        }


@dataclass(frozen=True)
class InitializerBankDraw:
    conditional_episode_index: int
    draw_index: int
    slot_kind: str
    slot_index: int
    instance_id: str
    base_lineage: str
    system_seed: int
    attempt_seeds: tuple[int, ...]
    draw_key: str

    def __post_init__(self) -> None:
        attempt_seeds = tuple(self.attempt_seeds)
        if (
            type(self.conditional_episode_index) is not int
            or self.conditional_episode_index < 0
            or type(self.draw_index) is not int
            or self.draw_index < 0
            or type(self.slot_index) is not int
            or self.slot_index != self.draw_index
            or self.slot_kind not in {"initial", "restart_cache"}
            or type(self.instance_id) is not str
            or not self.instance_id
            or type(self.base_lineage) is not str
            or not self.base_lineage
            or type(self.system_seed) is not int
            or not 0 <= self.system_seed < 2**31
            or not attempt_seeds
            or any(type(seed) is not int or not 0 <= seed < 2**31 for seed in attempt_seeds)
        ):
            raise ValueError("initializer bank draw is malformed")
        _require_digest(self.draw_key, label="initializer bank draw")
        object.__setattr__(self, "attempt_seeds", attempt_seeds)

    def as_dict(self) -> dict[str, object]:
        return {
            "conditional_episode_index": self.conditional_episode_index,
            "draw_index": self.draw_index,
            "slot_kind": self.slot_kind,
            "slot_index": self.slot_index,
            "instance_id": self.instance_id,
            "base_lineage": self.base_lineage,
            "system_seed": self.system_seed,
            "attempt_seeds": list(self.attempt_seeds),
            "draw_key": self.draw_key,
        }


@dataclass(frozen=True)
class InitializerBankEpisode:
    episode_schedule_index: int
    instance_id: str
    base_lineage: str
    draws: tuple[InitializerBankDraw, ...]
    restart_draws: tuple[InitializerBankDraw, ...]

    def __post_init__(self) -> None:
        draws = tuple(self.draws)
        restart_draws = tuple(self.restart_draws)
        if (
            type(self.episode_schedule_index) is not int
            or self.episode_schedule_index < 0
            or type(self.instance_id) is not str
            or not self.instance_id
            or type(self.base_lineage) is not str
            or not self.base_lineage
            or not draws
            or any(
                type(draw) is not InitializerBankDraw
                for draw in (*draws, *restart_draws)
            )
        ):
            raise TypeError("initializer bank episode draws must be typed draw records")
        for slot_kind, rows in (("initial", draws), ("restart_cache", restart_draws)):
            if any(
                draw.conditional_episode_index != self.episode_schedule_index
                or draw.instance_id != self.instance_id
                or draw.base_lineage != self.base_lineage
                or draw.slot_kind != slot_kind
                or draw.slot_index != index
                for index, draw in enumerate(rows)
            ):
                raise ValueError("initializer bank episode draw schedule is inconsistent")
        object.__setattr__(self, "draws", draws)
        object.__setattr__(self, "restart_draws", restart_draws)

    def as_dict(self) -> dict[str, object]:
        return {
            "episode_schedule_index": self.episode_schedule_index,
            "instance_id": self.instance_id,
            "base_lineage": self.base_lineage,
            "draws": [draw.as_dict() for draw in self.draws],
            "restart_draws": [draw.as_dict() for draw in self.restart_draws],
        }


@dataclass(frozen=True)
class InitializerBankPlan:
    prepared_manifest_sha256: str
    partition: str
    training_seed: int
    episode_schedule_start: int
    episode_schedule_stop_exclusive: int
    max_draws_per_conditional_episode: int
    restart_cache_slots_per_episode: int
    tasks: tuple[InitializerBankTaskIdentity, ...]
    episodes: tuple[InitializerBankEpisode, ...]
    scheduler: Mapping[str, object]
    initializer_identity: BackendIdentity
    runtime_implementation_manifest: Mapping[str, object]
    runtime_implementation_digest: str
    complete_system_config: Mapping[str, object]
    config_digest: str
    context: Mapping[str, object]
    context_digest: str
    policy_context_digest: str
    publication_eligible: bool
    _record_digest: str = dataclass_field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self) is not InitializerBankPlan:
            raise TypeError("initializer bank plan must use the exact plan type")
        tasks = tuple(self.tasks)
        episodes = tuple(self.episodes)
        if any(type(item) is not InitializerBankTaskIdentity for item in tasks):
            raise TypeError("initializer bank plan tasks must be typed identities")
        if any(type(item) is not InitializerBankEpisode for item in episodes):
            raise TypeError("initializer bank plan episodes must be typed records")
        if type(self.initializer_identity) is not BackendIdentity:
            raise TypeError("initializer bank plan requires an exact BackendIdentity")
        if (
            self.partition != INITIALIZER_BANK_PARTITION
            or type(self.training_seed) is not int
            or self.training_seed < 0
            or type(self.episode_schedule_start) is not int
            or self.episode_schedule_start < 0
            or type(self.episode_schedule_stop_exclusive) is not int
            or self.episode_schedule_stop_exclusive <= self.episode_schedule_start
            or type(self.max_draws_per_conditional_episode) is not int
            or self.max_draws_per_conditional_episode <= 0
            or type(self.restart_cache_slots_per_episode) is not int
            or self.restart_cache_slots_per_episode < 0
            or type(self.publication_eligible) is not bool
            or not tasks
            or len(episodes)
            != self.episode_schedule_stop_exclusive - self.episode_schedule_start
            or tuple(item.episode_schedule_index for item in episodes)
            != tuple(
                range(
                    self.episode_schedule_start,
                    self.episode_schedule_stop_exclusive,
                )
            )
        ):
            raise ValueError("initializer bank plan schedule is malformed")
        for value, label in (
            (self.prepared_manifest_sha256, "prepared manifest"),
            (self.runtime_implementation_digest, "runtime implementation"),
            (self.config_digest, "complete-system config"),
            (self.context_digest, "context"),
            (self.policy_context_digest, "policy context"),
        ):
            _require_digest(value, label=f"initializer bank plan {label}")
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "episodes", episodes)
        for name in (
            "scheduler",
            "runtime_implementation_manifest",
            "complete_system_config",
            "context",
        ):
            frozen = _freeze_json(getattr(self, name))
            if not isinstance(frozen, Mapping):
                raise TypeError(f"initializer bank plan {name} must be a mapping")
            object.__setattr__(self, name, frozen)
        object.__setattr__(self, "_record_digest", str(self.as_dict()["record_digest"]))

    def as_dict(self) -> dict[str, object]:
        body = {
            "schema": INITIALIZER_BANK_PLAN_SCHEMA,
            "schema_version": INITIALIZER_BANK_PLAN_VERSION,
            "prepared_manifest_sha256": self.prepared_manifest_sha256,
            "partition": self.partition,
            "prepared_schema_version": PREPARED_SCHEMA_VERSION_V4,
            "corpus_scope": INITIALIZER_BANK_CORPUS_SCOPE,
            "training_seed": self.training_seed,
            "episode_schedule_start": self.episode_schedule_start,
            "episode_schedule_stop_exclusive": self.episode_schedule_stop_exclusive,
            "max_draws_per_conditional_episode": (
                self.max_draws_per_conditional_episode
            ),
            "restart_cache_slots_per_episode": self.restart_cache_slots_per_episode,
            "tasks": [item.as_dict() for item in self.tasks],
            "episodes": [item.as_dict() for item in self.episodes],
            "scheduler": _jsonable(self.scheduler),
            "initializer_identity": self.initializer_identity.as_dict(),
            "runtime_implementation_manifest": _jsonable(
                self.runtime_implementation_manifest
            ),
            "runtime_implementation_digest": self.runtime_implementation_digest,
            "complete_system_config": _jsonable(self.complete_system_config),
            "config_digest": self.config_digest,
            "context": _jsonable(self.context),
            "context_digest": self.context_digest,
            "policy_context_digest": self.policy_context_digest,
            "publication_eligible": self.publication_eligible,
            "test_only_backend": not self.publication_eligible,
        }
        record_digest = getattr(self, "_record_digest", None)
        if record_digest is None:
            record_digest = stable_digest(body)
        return {**body, "record_digest": record_digest}

    @property
    def record_digest(self) -> str:
        return self._record_digest


def _validated_train_registry(
    prepared_tasks: Sequence[PreparedTask],
) -> tuple[
    tuple[InitializerBankTaskIdentity, ...],
    dict[str, EmbeddingTask],
    dict[str, tuple[str, ...]],
]:
    if not prepared_tasks:
        raise ValueError("initializer bank needs at least one train-partition task")
    grouped: dict[str, list[PreparedTask]] = {}
    seen_task_ids: set[str] = set()
    for item in prepared_tasks:
        if not isinstance(item, PreparedTask):
            raise TypeError("initializer bank inputs must be typed PreparedTask records")
        if item.partition != INITIALIZER_BANK_PARTITION:
            raise ValueError("initializer bank accepts the train partition only")
        if (
            item.prepared_schema_version != PREPARED_SCHEMA_VERSION_V4
            or item.corpus_scope != INITIALIZER_BANK_CORPUS_SCOPE
        ):
            raise ValueError("initializer bank requires production prepared schema v4")
        if item.provenance is None or item.design_condition is None:
            raise ValueError("initializer bank requires authenticated provenance and design")
        lineage = item.task.lineage
        if (
            not isinstance(lineage, str)
            or not lineage
            or item.provenance.base_parent_lineage != lineage
            or item.design_condition.base_lineage_key != lineage
        ):
            raise ValueError("prepared task differs from its authenticated base lineage")
        if item.design_condition.learning_partition != INITIALIZER_BANK_PARTITION:
            raise ValueError("initializer bank design condition is outside the train partition")
        _require_digest(
            item.public_instance_record_digest,
            label="initializer-bank public policy-instance record",
        )
        if (
            item.task.ground_energy is not None
            or item.reference_status is not None
            or item.certificate_digest is not None
            or item.evaluator_protocol_digest is not None
            or item.quality_attestation_digest is not None
            or item.quality_evidence_manifest_digest is not None
            or item.quality_evidence_manifest_sha256 is not None
            or item.quality_target_set_digest is not None
            or item.quality_target_count is not None
        ):
            raise ValueError("initializer bank construction cannot receive an evaluator target")
        if item.task_id in seen_task_ids:
            raise ValueError("initializer bank input repeats a prepared task ID")
        seen_task_ids.add(item.task_id)
        grouped.setdefault(item.instance_id, []).append(item)

    identities: list[InitializerBankTaskIdentity] = []
    public_tasks: dict[str, EmbeddingTask] = {}
    instances_by_lineage: dict[str, list[str]] = {}
    for instance_id, rows in sorted(grouped.items()):
        lineages = {row.task.lineage for row in rows}
        if len(lineages) != 1:
            raise ValueError("one policy instance crosses base lineages")
        lineage = next(iter(lineages))
        assert isinstance(lineage, str)
        digests = {
            _public_task_digest(row.task, instance_id=instance_id) for row in rows
        }
        if len(digests) != 1:
            raise ValueError("candidate rows disagree on their public policy instance")
        representative = min(rows, key=lambda row: row.task_id)
        public_tasks[instance_id] = dataclasses.replace(
            representative.task,
            name=instance_id,
            ground_energy=None,
            witness=None,
            initial_embedding=None,
        )
        ordered_rows = sorted(rows, key=lambda row: row.task_id)
        source_record_digests = tuple(
            stable_digest(
                {
                    "task_id": row.task_id,
                    "initializer_record_digest": row.initializer_record_digest,
                    "provenance_record_digest": row.provenance.record_digest,
                    "design_condition_digest": row.design_condition.registry_row_digest,
                }
            )
            for row in ordered_rows
        )
        identity = InitializerBankTaskIdentity(
            instance_id=instance_id,
            base_lineage=lineage,
            source_task_ids=tuple(row.task_id for row in ordered_rows),
            source_record_digests=source_record_digests,
            public_task_digest=next(iter(digests)),
        )
        identities.append(identity)
        instances_by_lineage.setdefault(lineage, []).append(instance_id)
    return (
        tuple(identities),
        public_tasks,
        {
            lineage: tuple(sorted(instance_ids))
            for lineage, instance_ids in sorted(instances_by_lineage.items())
        },
    )


def build_initializer_bank_plan(
    prepared_tasks: Sequence[PreparedTask],
    *,
    prepared_manifest_sha256: str,
    training_seed: int,
    episode_schedule_start: int,
    episode_count: int,
    max_draws_per_conditional_episode: int,
    initializer: object,
    runtime_implementation_manifest: Mapping[str, object],
    config: CompleteSystemConfig,
    context: Context,
    allow_test_backend: bool = False,
) -> InitializerBankPlan:
    """Seal a deterministic, lineage-equal train-only initializer episode schedule."""

    prepared_manifest_sha256 = _require_digest(
        prepared_manifest_sha256, label="prepared manifest identity"
    )
    if isinstance(training_seed, bool) or not isinstance(training_seed, int) or training_seed < 0:
        raise ValueError("training seed must be a nonnegative integer")
    if (
        isinstance(episode_schedule_start, bool)
        or not isinstance(episode_schedule_start, int)
        or episode_schedule_start < 0
    ):
        raise ValueError("episode schedule start must be a nonnegative integer")
    if isinstance(episode_count, bool) or not isinstance(episode_count, int) or episode_count <= 0:
        raise ValueError("initializer bank episode count must be positive")
    if (
        isinstance(max_draws_per_conditional_episode, bool)
        or not isinstance(max_draws_per_conditional_episode, int)
        or max_draws_per_conditional_episode <= 0
    ):
        raise ValueError("conditional initializer draw cap must be a positive integer")
    if not isinstance(config, CompleteSystemConfig) or not isinstance(context, Context):
        raise TypeError("initializer bank requires typed complete-system config and context")
    if config.audit_reads != context.audit_reads:
        raise ValueError("complete-system and context audit reads differ")
    if config.policy_restart_mode != COMPLETE_POLICY_RESTART_MODE:
        raise ValueError(
            "initializer restart-cache banks require the registered persistent cache mode"
        )
    if config.max_initializer_attempts > context.caps.restart_work:
        raise ValueError("initializer attempts exceed the context restart-work cap")
    identity, runtime, runtime_digest, production_backend = _validate_backend_contract(
        initializer,
        config,
        runtime_implementation_manifest,
        allow_test_backend=allow_test_backend,
    )
    task_identities, _, instances_by_lineage = _validated_train_registry(prepared_tasks)
    lineages = tuple(instances_by_lineage)
    task_registry = {
        lineage: list(instances_by_lineage[lineage]) for lineage in lineages
    }
    scheduler_body = {
        "independent_unit": "immutable-base-lineage",
        "lineage_schedule": INITIALIZER_BANK_SCHEDULE_RULE,
        "within_lineage_schedule": INITIALIZER_BANK_WITHIN_LINEAGE_RULE,
        "within_lineage_seed_domain": INITIALIZER_BANK_WITHIN_LINEAGE_SEED_DOMAIN,
        "initializer_retry_policy": (
            "conditional-initial-prefix-plus-fixed-native-restart-tape-v2"
        ),
        "ordered_lineages": list(lineages),
        "lineage_count": len(lineages),
        "instance_count": len(task_identities),
        "lineage_instance_registry_digest": stable_digest(task_registry),
    }
    scheduler = {**scheduler_body, "record_digest": stable_digest(scheduler_body)}
    episodes: list[InitializerBankEpisode] = []
    restart_draw_count = context.restart_allowance
    stop = episode_schedule_start + episode_count
    for episode_index in range(episode_schedule_start, stop):
        lineage = lineages[episode_index % len(lineages)]
        occurrence = episode_index // len(lineages)
        candidates = instances_by_lineage[lineage]
        offset = _domain_seed(
            training_seed,
            INITIALIZER_BANK_WITHIN_LINEAGE_SEED_DOMAIN,
            lineage,
        ) % len(candidates)
        instance_id = candidates[(offset + occurrence) % len(candidates)]
        draws = [
            _planned_draw(
                training_seed=training_seed,
                episode_index=episode_index,
                instance_id=instance_id,
                lineage=lineage,
                slot_kind="initial",
                slot_index=draw_index,
                attempt_count=config.max_initializer_attempts,
                method_id=identity.method_id,
            )
            for draw_index in range(max_draws_per_conditional_episode)
        ]
        restart_draws = [
            _planned_draw(
                training_seed=training_seed,
                episode_index=episode_index,
                instance_id=instance_id,
                lineage=lineage,
                slot_kind="restart_cache",
                slot_index=draw_index,
                attempt_count=config.max_initializer_attempts,
                method_id=identity.method_id,
            )
            for draw_index in range(restart_draw_count)
        ]
        episodes.append(
            InitializerBankEpisode(
                episode_schedule_index=episode_index,
                instance_id=instance_id,
                base_lineage=lineage,
                draws=tuple(draws),
                restart_draws=tuple(restart_draws),
            )
        )
    context_body = _jsonable(context)
    policy_context_body = context_body
    assert isinstance(context_body, dict) and isinstance(policy_context_body, dict)
    frozen_context = _freeze_json(context_body)
    frozen_config = _freeze_json(config.as_dict())
    frozen_scheduler = _freeze_json(scheduler)
    assert isinstance(frozen_context, Mapping)
    assert isinstance(frozen_config, Mapping)
    assert isinstance(frozen_scheduler, Mapping)
    return InitializerBankPlan(
        prepared_manifest_sha256=prepared_manifest_sha256,
        partition=INITIALIZER_BANK_PARTITION,
        training_seed=training_seed,
        episode_schedule_start=episode_schedule_start,
        episode_schedule_stop_exclusive=stop,
        max_draws_per_conditional_episode=max_draws_per_conditional_episode,
        restart_cache_slots_per_episode=restart_draw_count,
        tasks=task_identities,
        episodes=tuple(episodes),
        scheduler=frozen_scheduler,
        initializer_identity=identity,
        runtime_implementation_manifest=runtime,
        runtime_implementation_digest=runtime_digest,
        complete_system_config=frozen_config,
        config_digest=config.digest,
        context=frozen_context,
        context_digest=stable_digest(context_body),
        policy_context_digest=stable_digest(policy_context_body),
        publication_eligible=production_backend,
    )


def _canonical_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json(raw: bytes, *, label: str) -> dict[str, object]:
    def pairs(items: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"{label} contains non-finite number {token}")

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _verify_record(payload: Mapping[str, object], *, label: str) -> str:
    digest = _require_digest(payload.get("record_digest"), label=f"{label} record digest")
    body = {key: value for key, value in payload.items() if key != "record_digest"}
    if stable_digest(body) != digest:
        raise ValueError(f"{label} record digest mismatch")
    return digest


def _write_once(path: Path, payload: bytes) -> None:
    """Create an immutable artifact, or verify an identical resumed artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"immutable initializer-bank artifact is unsafe: {path.name}")
        if path.read_bytes() != payload:
            raise ValueError(f"immutable initializer-bank artifact differs: {path.name}")
        return
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_initializer_bank_plan(
    path: str | Path,
    plan: InitializerBankPlan,
) -> str:
    """Publish one canonical immutable plan and return its file SHA-256 pin."""

    if not isinstance(plan, InitializerBankPlan):
        raise TypeError("initializer bank plan publication requires a typed plan")
    payload = _canonical_bytes(plan.as_dict())
    _write_once(Path(path), payload)
    return _sha256_bytes(payload)


def load_initializer_bank_plan(
    path: str | Path,
    *,
    expected_plan_sha256: str,
    prepared_tasks: Sequence[PreparedTask],
    prepared_manifest_sha256: str,
    initializer: object,
    runtime_implementation_manifest: Mapping[str, object],
    config: CompleteSystemConfig,
    context: Context,
    allow_test_backend: bool = False,
) -> InitializerBankPlan:
    """Authenticate a serialized plan against both an external pin and live inputs."""

    expected_plan_sha256 = _require_digest(
        expected_plan_sha256, label="expected initializer bank plan"
    )
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("initializer bank plan is missing or unsafe")
    raw = source.read_bytes()
    if _sha256_bytes(raw) != expected_plan_sha256:
        raise ValueError("initializer bank plan SHA-256 pin mismatch")
    payload = _strict_json(raw, label="initializer bank plan")
    _verify_record(payload, label="initializer bank plan")
    if (
        payload.get("schema") != INITIALIZER_BANK_PLAN_SCHEMA
        or payload.get("schema_version") != INITIALIZER_BANK_PLAN_VERSION
        or payload.get("prepared_schema_version") != PREPARED_SCHEMA_VERSION_V4
        or payload.get("corpus_scope") != INITIALIZER_BANK_CORPUS_SCOPE
        or payload.get("partition") != INITIALIZER_BANK_PARTITION
    ):
        raise ValueError("unsupported initializer bank plan schema or partition")
    start = payload.get("episode_schedule_start")
    stop = payload.get("episode_schedule_stop_exclusive")
    draw_cap = payload.get("max_draws_per_conditional_episode")
    training_seed = payload.get("training_seed")
    if (
        type(start) is not int
        or type(stop) is not int
        or stop <= start
        or type(draw_cap) is not int
        or draw_cap <= 0
        or type(training_seed) is not int
        or training_seed < 0
    ):
        raise ValueError("initializer bank plan schedule is malformed")
    rebuilt = build_initializer_bank_plan(
        prepared_tasks,
        prepared_manifest_sha256=prepared_manifest_sha256,
        training_seed=training_seed,
        episode_schedule_start=start,
        episode_count=stop - start,
        max_draws_per_conditional_episode=draw_cap,
        initializer=initializer,
        runtime_implementation_manifest=runtime_implementation_manifest,
        config=config,
        context=context,
        allow_test_backend=allow_test_backend,
    )
    if rebuilt.as_dict() != payload:
        raise ValueError("initializer bank plan differs from current authenticated inputs")
    return rebuilt


def _work_from_mapping(raw: object, *, label: str) -> WorkVector:
    if not isinstance(raw, Mapping) or set(raw) != set(WORK_FIELDS):
        raise ValueError(f"{label} has missing or unknown work coordinates")
    values: dict[str, int] = {}
    for name in WORK_FIELDS:
        value = raw[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{label} work coordinates must be nonnegative integers")
        values[name] = value
    return WorkVector(**values)


def _sum_work(attempts: Sequence[InitializerAttemptReceipt]) -> WorkVector:
    total = WorkVector()
    for attempt in attempts:
        exact = attempt.total_work.to_work_vector()
        if exact is None:
            raise ValueError("initializer snapshot requires exact native work on every attempt")
        total = total + exact
    return total


def _embedding_digest(chains: Mapping[Hashable, Sequence[Hashable]]) -> str:
    rows = sorted(
        (
            _typed_identity(owner),
            sorted(_typed_identity(qubit) for qubit in chain),
        )
        for owner, chain in chains.items()
    )
    return stable_digest({"domain": "isingfold-complete-initializer-v1", "chains": rows})


def _normalise_embedding(
    raw: Mapping[Hashable, Sequence[Hashable]],
) -> dict[Hashable, frozenset[Hashable]]:
    if not isinstance(raw, Mapping):
        raise CompleteSystemBackendError("initializer embedding is not a mapping")
    try:
        return {owner: frozenset(chain) for owner, chain in raw.items()}
    except (TypeError, ValueError) as exc:
        raise CompleteSystemBackendError("initializer returned malformed branch sets") from exc


def _encode_embedding(
    task: EmbeddingTask,
    chains: Mapping[Hashable, Sequence[Hashable]],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    logical_nodes = tuple(task.logical.nodes())
    host_nodes = tuple(task.host.nodes())
    if set(chains) != set(logical_nodes):
        raise CompleteSystemBackendError("initializer embedding changed the logical domain")
    logical_index = {node: index for index, node in enumerate(logical_nodes)}
    host_index = {node: index for index, node in enumerate(host_nodes)}
    try:
        return tuple(
            sorted(
                (
                    logical_index[owner],
                    tuple(sorted(host_index[qubit] for qubit in chain)),
                )
                for owner, chain in chains.items()
            )
        )
    except KeyError as exc:
        raise CompleteSystemBackendError(
            "initializer embedding refers outside the authenticated host"
        ) from exc


def _decode_embedding(
    task: EmbeddingTask,
    encoded: Sequence[tuple[int, tuple[int, ...]]],
) -> dict[Hashable, frozenset[Hashable]]:
    logical_nodes = tuple(task.logical.nodes())
    host_nodes = tuple(task.host.nodes())
    if tuple(owner for owner, _ in encoded) != tuple(range(len(logical_nodes))):
        raise ValueError("snapshot embedding does not cover the logical node order")
    try:
        return {
            logical_nodes[owner]: frozenset(host_nodes[index] for index in chain)
            for owner, chain in encoded
        }
    except IndexError as exc:
        raise ValueError("snapshot embedding index lies outside the public task") from exc


@dataclass(frozen=True)
class InitializerSnapshot:
    plan_record_digest: str
    conditional_episode_index: int
    draw_index: int
    slot_kind: str
    slot_index: int
    draw_key: str
    instance_id: str
    base_lineage: str
    system_seed: int
    attempts: tuple[InitializerAttemptReceipt, ...]
    selected_attempt: int | None
    selected_embedding: tuple[tuple[int, tuple[int, ...]], ...] | None
    selected_embedding_digest: str | None
    initializer_work: WorkVector
    environment_budget_debit: WorkVector
    work_before: WorkVector
    work_after: WorkVector
    time_before_seconds: float
    time_after_seconds: float
    execution_status: str
    success: bool
    failure_reason: str | None
    initialization_seconds: float

    def __post_init__(self) -> None:
        _require_digest(self.plan_record_digest, label="snapshot plan identity")
        _require_digest(self.draw_key, label="snapshot draw identity")
        if (
            type(self.conditional_episode_index) is not int
            or self.conditional_episode_index < 0
            or type(self.draw_index) is not int
            or self.draw_index < 0
        ):
            raise ValueError("snapshot schedule indices must be nonnegative integers")
        if self.slot_kind not in {"initial", "restart_cache"}:
            raise ValueError("snapshot slot kind must be initial or restart_cache")
        if type(self.slot_index) is not int or self.slot_index < 0:
            raise ValueError("snapshot slot index must be a nonnegative integer")
        if self.draw_index != self.slot_index:
            raise ValueError("snapshot draw and slot indices differ")
        if (
            not isinstance(self.instance_id, str)
            or not self.instance_id
            or not isinstance(self.base_lineage, str)
            or not self.base_lineage
        ):
            raise ValueError("snapshot task identity must be nonempty")
        if type(self.system_seed) is not int or not 0 <= self.system_seed < 2**31:
            raise ValueError("snapshot system seed is outside its domain")
        if not isinstance(self.attempts, tuple) or any(
            not isinstance(attempt, InitializerAttemptReceipt) for attempt in self.attempts
        ):
            raise TypeError("snapshot attempts must be typed initializer receipts")
        if tuple(attempt.attempt_index for attempt in self.attempts) != tuple(
            range(len(self.attempts))
        ):
            raise ValueError("snapshot initializer attempts must be contiguous")
        if not isinstance(self.initializer_work, WorkVector) or not isinstance(
            self.environment_budget_debit, WorkVector
        ):
            raise TypeError("snapshot work ledgers must be exact WorkVectors")
        if not isinstance(self.work_before, WorkVector) or not isinstance(
            self.work_after, WorkVector
        ):
            raise TypeError("snapshot cumulative work ledger must be exact")
        if _sum_work(self.attempts) != self.initializer_work:
            raise ValueError("snapshot initializer work differs from its attempts")
        if self.work_after != self.work_before + self.initializer_work:
            raise ValueError("snapshot cumulative work ledger is inconsistent")
        if not self.work_before.is_nonnegative or not self.work_after.is_nonnegative:
            raise ValueError("snapshot cumulative work ledger cannot be negative")
        if isinstance(self.initialization_seconds, bool) or not math.isfinite(
            self.initialization_seconds
        ) or self.initialization_seconds < 0.0:
            raise ValueError("snapshot initialization time must be finite and nonnegative")
        if (
            isinstance(self.time_before_seconds, bool)
            or isinstance(self.time_after_seconds, bool)
            or not math.isfinite(self.time_before_seconds)
            or not math.isfinite(self.time_after_seconds)
            or self.time_before_seconds < 0.0
            or self.time_after_seconds
            != self.time_before_seconds + self.initialization_seconds
        ):
            raise ValueError("snapshot cumulative wallclock ledger is inconsistent")
        if self.execution_status not in {
            "SUCCESS",
            "FAILED",
            "BUDGET_NOT_INVOKED",
            "TIME_NOT_INVOKED",
        }:
            raise ValueError("snapshot has an unknown execution status")
        if type(self.success) is not bool:
            raise TypeError("snapshot success must be Boolean")
        if self.success:
            if (
                self.execution_status != "SUCCESS"
                or
                type(self.selected_attempt) is not int
                or self.selected_embedding is None
                or not _is_digest(self.selected_embedding_digest)
                or self.failure_reason is not None
                or self.environment_budget_debit != self.initializer_work
            ):
                raise ValueError("successful snapshot has an incomplete selected initializer")
            if tuple(owner for owner, _ in self.selected_embedding) != tuple(
                range(len(self.selected_embedding))
            ):
                raise ValueError("snapshot selected embedding rows are not canonical")
            if not 0 <= self.selected_attempt < len(self.attempts):
                raise ValueError("snapshot selected attempt lies outside its attempt ledger")
            selected = self.attempts[self.selected_attempt]
            if (
                selected.status != "VALID_CANDIDATE"
                or selected.embedding_digest != self.selected_embedding_digest
            ):
                raise ValueError("snapshot selected embedding differs from its attempt")
        elif (
            self.selected_attempt is not None
            or self.selected_embedding is not None
            or self.selected_embedding_digest is not None
            or not isinstance(self.failure_reason, str)
            or not self.failure_reason
            or self.environment_budget_debit
            != (
                self.initializer_work
                if self.slot_kind == "restart_cache"
                else WorkVector()
            )
        ):
            raise ValueError("failed snapshot cannot carry a selected initializer")
        if self.execution_status in {"BUDGET_NOT_INVOKED", "TIME_NOT_INVOKED"} and (
            self.attempts
            or self.initializer_work != WorkVector()
            or self.initialization_seconds != 0.0
        ):
            raise ValueError("uninvoked initializer slot cannot report execution work")

    def as_dict(self) -> dict[str, object]:
        body = {
            "schema": INITIALIZER_SNAPSHOT_SCHEMA,
            "schema_version": INITIALIZER_SNAPSHOT_VERSION,
            "plan_record_digest": self.plan_record_digest,
            "conditional_episode_index": self.conditional_episode_index,
            "draw_index": self.draw_index,
            "slot_kind": self.slot_kind,
            "slot_index": self.slot_index,
            "draw_key": self.draw_key,
            "instance_id": self.instance_id,
            "base_lineage": self.base_lineage,
            "system_seed": self.system_seed,
            "attempts": [attempt.as_dict() for attempt in self.attempts],
            "selected_attempt": self.selected_attempt,
            "selected_embedding": (
                None
                if self.selected_embedding is None
                else [
                    {"logical_index": owner, "host_indices": list(chain)}
                    for owner, chain in self.selected_embedding
                ]
            ),
            "selected_embedding_digest": self.selected_embedding_digest,
            "initializer_work": self.initializer_work.as_dict(),
            "environment_budget_debit": self.environment_budget_debit.as_dict(),
            "work_before": self.work_before.as_dict(),
            "work_after": self.work_after.as_dict(),
            "time_before_seconds": self.time_before_seconds,
            "time_after_seconds": self.time_after_seconds,
            "execution_status": self.execution_status,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "initialization_seconds": self.initialization_seconds,
            "evaluator_target_opened": False,
        }
        return {**body, "record_digest": stable_digest(body)}

    @property
    def record_digest(self) -> str:
        return str(self.as_dict()["record_digest"])


@dataclass(frozen=True)
class EpisodeBootstrapOutcome:
    """Authenticated pre-policy initializer and persistent restart-cache outcome."""

    episode_schedule_index: int
    instance_id: str
    base_lineage: str
    public_task_digest: str
    initial_snapshot: InitializerSnapshot
    restart_cache_snapshots: tuple[InitializerSnapshot, ...]
    restart_cache_slot_count: int
    initial_generation_debit: WorkVector
    restart_cache_fill_debit: WorkVector
    total_pre_policy_debit: WorkVector
    precomputed_online_seconds: float
    plan_record_digest: str
    manifest_record_digest: str
    config_digest: str
    context_digest: str

    def __post_init__(self) -> None:
        if self.initial_snapshot.slot_kind != "initial":
            raise ValueError("bootstrap outcome requires one initial snapshot")
        if (
            self.initial_snapshot.work_before != WorkVector()
            or self.initial_snapshot.time_before_seconds != 0.0
        ):
            raise ValueError("bootstrap initial snapshot must start at zero work and time")
        if (
            type(self.restart_cache_slot_count) is not int
            or self.restart_cache_slot_count < 0
        ):
            raise ValueError("bootstrap restart-cache slot count must be nonnegative")
        if self.initial_generation_debit != self.initial_snapshot.initializer_work:
            raise ValueError("bootstrap initial debit differs from its snapshot")
        if not self.initial_snapshot.success and self.restart_cache_snapshots:
            raise ValueError("failed initial bootstrap cannot carry restart-cache snapshots")
        if self.initial_snapshot.success and (
            len(self.restart_cache_snapshots) != self.restart_cache_slot_count
        ):
            raise ValueError("successful bootstrap must bind every restart-cache slot")
        expected_work = self.initial_snapshot.initializer_work
        expected_time = self.initial_snapshot.initialization_seconds
        cache_work = WorkVector()
        for index, snapshot in enumerate(self.restart_cache_snapshots):
            if (
                snapshot.slot_kind != "restart_cache"
                or snapshot.slot_index != index
                or snapshot.work_before != expected_work
                or snapshot.time_before_seconds != expected_time
            ):
                raise ValueError("bootstrap restart-cache sequence is not contiguous")
            cache_work = cache_work + snapshot.initializer_work
            expected_work = snapshot.work_after
            expected_time = snapshot.time_after_seconds
        if self.restart_cache_fill_debit != cache_work:
            raise ValueError("bootstrap cache-fill debit differs from its snapshots")
        if self.total_pre_policy_debit != (
            self.initial_generation_debit + self.restart_cache_fill_debit
        ):
            raise ValueError("bootstrap total pre-policy debit is inconsistent")
        final_work = (
            self.initial_snapshot.work_after
            if not self.restart_cache_snapshots
            else self.restart_cache_snapshots[-1].work_after
        )
        if self.total_pre_policy_debit != final_work:
            raise ValueError("bootstrap total debit differs from cumulative executed work")
        if self.precomputed_online_seconds != expected_time:
            raise ValueError("bootstrap cumulative wallclock differs from its snapshots")
        for digest in (
            self.public_task_digest,
            self.plan_record_digest,
            self.manifest_record_digest,
            self.config_digest,
            self.context_digest,
        ):
            _require_digest(digest, label="bootstrap identity")

    def as_dict(self) -> dict[str, object]:
        body = {
            "schema": "isingfold.episode-bootstrap-outcome",
            "schema_version": 1,
            "episode_schedule_index": self.episode_schedule_index,
            "instance_id": self.instance_id,
            "base_lineage": self.base_lineage,
            "public_task_digest": self.public_task_digest,
            "initial_snapshot": self.initial_snapshot.as_dict(),
            "restart_cache_snapshots": [
                snapshot.as_dict() for snapshot in self.restart_cache_snapshots
            ],
            "restart_cache_slot_count": self.restart_cache_slot_count,
            "initial_generation_debit": self.initial_generation_debit.as_dict(),
            "restart_cache_fill_debit": self.restart_cache_fill_debit.as_dict(),
            "total_pre_policy_debit": self.total_pre_policy_debit.as_dict(),
            "precomputed_online_seconds": self.precomputed_online_seconds,
            "plan_record_digest": self.plan_record_digest,
            "manifest_record_digest": self.manifest_record_digest,
            "config_digest": self.config_digest,
            "context_digest": self.context_digest,
            "opened_evaluator_targets": False,
        }
        return {**body, "record_digest": stable_digest(body)}

    @property
    def record_digest(self) -> str:
        return str(self.as_dict()["record_digest"])


def _execute_draw(
    draw: InitializerBankDraw,
    *,
    plan: InitializerBankPlan,
    task: EmbeddingTask,
    initializer: object,
    config: CompleteSystemConfig,
    context: Context,
    work_before: WorkVector = WorkVector(),
    time_before_seconds: float = 0.0,
) -> InitializerSnapshot:
    started = time.perf_counter()
    remaining_wallclock = max(
        0.0, config.online_wallclock_seconds - time_before_seconds
    )
    deadline = started + remaining_wallclock
    policy_context = context
    policy_entry_reserve = WorkVector(compiler_calls=4, validator_calls=1) + (
        policy_context.reserve
    )
    attempts: list[InitializerAttemptReceipt] = []
    candidates: list[
        tuple[int, dict[Hashable, frozenset[Hashable]], int, int, str]
    ] = []
    pre_call_budget_blocked = False
    time_not_invoked = remaining_wallclock <= 0.0
    for attempt_index, attempt_seed in enumerate(draw.attempt_seeds):
        remaining_seconds = deadline - time.perf_counter()
        if remaining_seconds <= 0.0:
            break
        if getattr(initializer, "identity", None) != plan.initializer_identity:
            raise CompleteSystemBackendError(
                "initializer identity changed after the bank plan was sealed"
            )
        exact_prior = work_before + _sum_work(attempts)
        call_reserve = WorkVector(restart_work=1, validator_calls=1) + policy_entry_reserve
        native_work_cap = context.caps - exact_prior - call_reserve
        if not native_work_cap.is_nonnegative:
            pre_call_budget_blocked = True
            break
        call_started = time.perf_counter()
        try:
            result = initializer.search(
                task.logical,
                task.host,
                seed=attempt_seed,
                timeout_seconds=remaining_seconds,
                parameters=config.initializer_parameters,
                work_cap=native_work_cap,
            )
        except CompleteSystemBackendError:
            raise
        except Exception as exc:
            raise CompleteSystemBackendError(
                f"initializer raised {type(exc).__name__}: {exc}"
            ) from exc
        observed_seconds = time.perf_counter() - call_started
        if getattr(initializer, "identity", None) != plan.initializer_identity:
            raise CompleteSystemBackendError(
                "initializer identity changed during a bank generation call"
            )
        if not isinstance(result, CompleteInitializerResult):
            raise CompleteSystemBackendError(
                "initializer returned a non-CompleteInitializerResult payload"
            )
        if result.status is SearchStatus.ERROR:
            raise CompleteSystemBackendError(result.detail or "initializer backend error")
        exact_backend_work = result.work.to_work_vector()
        if exact_backend_work is None:
            raise CompleteSystemBackendError(
                "LAC initializer bank requires all nine exact native work coordinates"
            )
        runtime_manifest = result.diagnostics.get("runtime_implementation_manifest")
        if _jsonable(runtime_manifest) != _jsonable(plan.runtime_implementation_manifest):
            raise CompleteSystemBackendError(
                "initializer diagnostics changed the sealed runtime implementation"
            )
        runner_work = WorkVector(restart_work=1)
        if result.budget_exhausted_coordinate is not None:
            attempts.append(
                InitializerAttemptReceipt(
                    attempt_index=attempt_index,
                    seed=attempt_seed,
                    status="WORK_BUDGET_EXHAUSTED",
                    reported_seconds=result.elapsed_seconds,
                    observed_seconds=observed_seconds,
                    backend_work=result.work,
                    runner_work=runner_work,
                    detail=(
                        "native prospective work cap exhausted: "
                        f"{result.budget_exhausted_coordinate}"
                    ),
                    backend_diagnostics=result.diagnostics,
                )
            )
            continue
        timed_out = observed_seconds > remaining_seconds + max(
            1e-9, remaining_seconds * 1e-9
        )
        if timed_out or result.status is SearchStatus.TIMED_OUT:
            attempts.append(
                InitializerAttemptReceipt(
                    attempt_index=attempt_index,
                    seed=attempt_seed,
                    status=SearchStatus.TIMED_OUT.value,
                    reported_seconds=result.elapsed_seconds,
                    observed_seconds=observed_seconds,
                    backend_work=result.work,
                    runner_work=runner_work,
                    detail=result.detail or "initializer reached the registered call deadline",
                    backend_diagnostics=result.diagnostics,
                )
            )
            if not (work_before + _sum_work(attempts)).fits_in(context.caps):
                break
            continue
        if result.status is SearchStatus.NO_EMBEDDING:
            attempts.append(
                InitializerAttemptReceipt(
                    attempt_index=attempt_index,
                    seed=attempt_seed,
                    status=SearchStatus.NO_EMBEDDING.value,
                    reported_seconds=result.elapsed_seconds,
                    observed_seconds=observed_seconds,
                    backend_work=result.work,
                    runner_work=runner_work,
                    detail=result.detail,
                    backend_diagnostics=result.diagnostics,
                )
            )
            if not (work_before + _sum_work(attempts)).fits_in(context.caps):
                break
            continue
        if result.status is not SearchStatus.EMBEDDING or result.embedding is None:
            raise CompleteSystemBackendError("initializer returned an inconsistent status")
        prospective = (
            work_before
            + _sum_work(attempts)
            + exact_backend_work
            + runner_work
            + WorkVector(validator_calls=1)
            + policy_entry_reserve
        )
        if not prospective.fits_in(context.caps):
            attempts.append(
                InitializerAttemptReceipt(
                    attempt_index=attempt_index,
                    seed=attempt_seed,
                    status="WORK_CAP_EXHAUSTED",
                    reported_seconds=result.elapsed_seconds,
                    observed_seconds=observed_seconds,
                    backend_work=result.work,
                    runner_work=runner_work,
                    detail=(
                        "initializer returned a candidate without enough registered work "
                        "for independent validation"
                    ),
                    backend_diagnostics=result.diagnostics,
                )
            )
            break
        chains = _normalise_embedding(result.embedding)
        validation = p_embed(chains, task.logical, task.host, context.qubit_cap)
        runner_work = runner_work + WorkVector(validator_calls=1)
        embedding_digest = _embedding_digest(chains)
        validation_digest = stable_digest(validation.as_dict())
        status = "VALID_CANDIDATE" if validation.valid else "INVALID_CANDIDATE"
        attempts.append(
            InitializerAttemptReceipt(
                attempt_index=attempt_index,
                seed=attempt_seed,
                status=status,
                reported_seconds=result.elapsed_seconds,
                observed_seconds=observed_seconds,
                backend_work=result.work,
                runner_work=runner_work,
                embedding_digest=embedding_digest,
                validation_digest=validation_digest,
                qubits=validation.qubits,
                max_chain=validation.max_chain,
                detail="; ".join(validation.reasons),
                backend_diagnostics=result.diagnostics,
            )
        )
        if validation.valid:
            candidates.append(
                (
                    attempt_index,
                    chains,
                    validation.qubits,
                    validation.max_chain,
                    embedding_digest,
                )
            )

    elapsed = (
        0.0
        if not attempts and (time_not_invoked or pre_call_budget_blocked)
        else time.perf_counter() - started
    )
    initializer_work = _sum_work(attempts)
    failure_reason: str | None = None
    selected_attempt: int | None = None
    selected_chains: dict[Hashable, frozenset[Hashable]] | None = None
    selected_digest: str | None = None
    if not candidates:
        if time_not_invoked and not attempts:
            failure_reason = "COMPLETE_SYSTEM_WALLCLOCK_BEFORE_CALL"
        elif not (work_before + initializer_work).fits_in(context.caps):
            failure_reason = "INITIALIZER_EXCEEDED_WORK_CAP"
        elif pre_call_budget_blocked:
            failure_reason = "INITIALIZER_WORK_CAP_BEFORE_CALL"
        elif any(item.status == "WORK_BUDGET_EXHAUSTED" for item in attempts):
            failure_reason = "INITIALIZER_NATIVE_WORK_BUDGET_EXHAUSTED"
        elif any(item.status == "WORK_CAP_EXHAUSTED" for item in attempts):
            failure_reason = "INITIALIZER_WORK_CAP_BEFORE_VALIDATION"
        else:
            failure_reason = "INITIALIZER_NO_VALID_EMBEDDING"
    else:
        selected_attempt, selected_chains, _, _, selected_digest = min(
            candidates,
            key=lambda item: (item[2], item[3], item[4], item[0]),
        )
        environment_initialization = WorkVector(compiler_calls=4, validator_calls=1)
        if not (
            work_before
            + initializer_work
            + environment_initialization
            + policy_context.reserve
        ).fits_in(policy_context.caps):
            failure_reason = (
                "INITIALIZER_EXCEEDED_WORK_CAP"
                if not initializer_work.fits_in(policy_context.caps)
                else "INITIALIZER_WORK_CAP_BEFORE_POLICY"
            )
            selected_attempt = None
            selected_chains = None
            selected_digest = None
        elif time.perf_counter() >= deadline:
            failure_reason = "COMPLETE_SYSTEM_WALLCLOCK_EXHAUSTED"
            selected_attempt = None
            selected_chains = None
            selected_digest = None
    success = failure_reason is None
    execution_status = (
        "SUCCESS"
        if success
        else (
            "BUDGET_NOT_INVOKED"
            if pre_call_budget_blocked and not attempts
            else (
                "TIME_NOT_INVOKED"
                if time_not_invoked and not attempts
                else "FAILED"
            )
        )
    )
    return InitializerSnapshot(
        plan_record_digest=plan.record_digest,
        conditional_episode_index=draw.conditional_episode_index,
        draw_index=draw.draw_index,
        slot_kind=draw.slot_kind,
        slot_index=draw.slot_index,
        draw_key=draw.draw_key,
        instance_id=draw.instance_id,
        base_lineage=draw.base_lineage,
        system_seed=draw.system_seed,
        attempts=tuple(attempts),
        selected_attempt=selected_attempt,
        selected_embedding=(
            None
            if selected_chains is None
            else _encode_embedding(task, selected_chains)
        ),
        selected_embedding_digest=selected_digest,
        initializer_work=initializer_work,
        environment_budget_debit=(
            initializer_work
            if success or draw.slot_kind == "restart_cache"
            else WorkVector()
        ),
        work_before=work_before,
        work_after=work_before + initializer_work,
        time_before_seconds=time_before_seconds,
        time_after_seconds=time_before_seconds + elapsed,
        execution_status=execution_status,
        success=success,
        failure_reason=failure_reason,
        initialization_seconds=elapsed,
    )


def execute_initializer_draw(
    draw: InitializerBankDraw,
    *,
    plan: object,
    task: EmbeddingTask,
    initializer: object,
    config: CompleteSystemConfig,
    context: Context,
    work_before: WorkVector = WorkVector(),
    time_before_seconds: float = 0.0,
) -> InitializerSnapshot:
    """Execute one sealed causal draw for a train or validation bootstrap plan.

    ``plan`` is intentionally protocol-typed: both bank plans expose an immutable record
    digest, initializer identity, and runtime manifest.  The internal executor revalidates
    all three on every native call.
    """

    for name in (
        "record_digest",
        "initializer_identity",
        "runtime_implementation_manifest",
    ):
        if not hasattr(plan, name):
            raise TypeError(f"initializer draw plan omits {name}")
    return _execute_draw(
        draw,
        plan=plan,  # type: ignore[arg-type]
        task=task,
        initializer=initializer,
        config=config,
        context=context,
        work_before=work_before,
        time_before_seconds=time_before_seconds,
    )


def _parse_attempt(raw: object) -> InitializerAttemptReceipt:
    if not isinstance(raw, Mapping):
        raise ValueError("initializer snapshot attempt must be an object")
    expected = {
        "attempt_index",
        "seed",
        "status",
        "reported_seconds",
        "observed_seconds",
        "backend_work",
        "backend_work_known_lower_bound",
        "runner_work",
        "total_work",
        "total_work_known_lower_bound",
        "embedding_digest",
        "validation_digest",
        "qubits",
        "max_chain",
        "detail",
        "backend_diagnostics",
    }
    if set(raw) != expected:
        raise ValueError("initializer snapshot attempt schema differs")
    backend_work = _work_from_mapping(raw["backend_work"], label="backend attempt")
    backend_lower = _work_from_mapping(
        raw["backend_work_known_lower_bound"], label="backend attempt lower bound"
    )
    if backend_work != backend_lower:
        raise ValueError("exact backend attempt work differs from its lower bound")
    runner_work = _work_from_mapping(raw["runner_work"], label="runner attempt")
    total = _work_from_mapping(raw["total_work"], label="total attempt")
    total_lower = _work_from_mapping(
        raw["total_work_known_lower_bound"], label="total attempt lower bound"
    )
    if total != backend_work + runner_work or total_lower != total:
        raise ValueError("initializer attempt total work is inconsistent")
    diagnostics = raw["backend_diagnostics"]
    if not isinstance(diagnostics, Mapping):
        raise ValueError("initializer attempt diagnostics must be an object")
    return InitializerAttemptReceipt(
        attempt_index=raw["attempt_index"],
        seed=raw["seed"],
        status=raw["status"],
        reported_seconds=raw["reported_seconds"],
        observed_seconds=raw["observed_seconds"],
        backend_work=PartialWorkVector.known(backend_work),
        runner_work=runner_work,
        embedding_digest=raw["embedding_digest"],
        validation_digest=raw["validation_digest"],
        qubits=raw["qubits"],
        max_chain=raw["max_chain"],
        detail=raw["detail"],
        backend_diagnostics=diagnostics,
    )


def _snapshot_from_payload(raw: Mapping[str, object]) -> InitializerSnapshot:
    expected = {
        "schema",
        "schema_version",
        "plan_record_digest",
        "conditional_episode_index",
        "draw_index",
        "slot_kind",
        "slot_index",
        "draw_key",
        "instance_id",
        "base_lineage",
        "system_seed",
        "attempts",
        "selected_attempt",
        "selected_embedding",
        "selected_embedding_digest",
        "initializer_work",
        "environment_budget_debit",
        "work_before",
        "work_after",
        "time_before_seconds",
        "time_after_seconds",
        "execution_status",
        "success",
        "failure_reason",
        "initialization_seconds",
        "evaluator_target_opened",
        "record_digest",
    }
    if set(raw) != expected:
        raise ValueError("initializer snapshot schema differs")
    _verify_record(raw, label="initializer snapshot")
    if (
        raw["schema"] != INITIALIZER_SNAPSHOT_SCHEMA
        or raw["schema_version"] != INITIALIZER_SNAPSHOT_VERSION
    ):
        raise ValueError("unsupported initializer snapshot schema")
    if raw["evaluator_target_opened"] is not False:
        raise ValueError("initializer snapshot claims evaluator-target access")
    attempts_raw = raw["attempts"]
    if not isinstance(attempts_raw, list):
        raise ValueError("initializer snapshot attempts must be a list")
    selected_raw = raw["selected_embedding"]
    selected: tuple[tuple[int, tuple[int, ...]], ...] | None = None
    if selected_raw is not None:
        if not isinstance(selected_raw, list):
            raise ValueError("initializer snapshot embedding must be a list or null")
        rows: list[tuple[int, tuple[int, ...]]] = []
        for row in selected_raw:
            if not isinstance(row, Mapping) or set(row) != {
                "logical_index",
                "host_indices",
            }:
                raise ValueError("initializer snapshot embedding row schema differs")
            logical_index = row["logical_index"]
            host_indices = row["host_indices"]
            if (
                type(logical_index) is not int
                or logical_index < 0
                or not isinstance(host_indices, list)
                or any(type(index) is not int or index < 0 for index in host_indices)
                or host_indices != sorted(set(host_indices))
            ):
                raise ValueError("initializer snapshot embedding indices are invalid")
            rows.append((logical_index, tuple(host_indices)))
        selected = tuple(rows)
    return InitializerSnapshot(
        plan_record_digest=raw["plan_record_digest"],
        conditional_episode_index=raw["conditional_episode_index"],
        draw_index=raw["draw_index"],
        slot_kind=raw["slot_kind"],
        slot_index=raw["slot_index"],
        draw_key=raw["draw_key"],
        instance_id=raw["instance_id"],
        base_lineage=raw["base_lineage"],
        system_seed=raw["system_seed"],
        attempts=tuple(_parse_attempt(item) for item in attempts_raw),
        selected_attempt=raw["selected_attempt"],
        selected_embedding=selected,
        selected_embedding_digest=raw["selected_embedding_digest"],
        initializer_work=_work_from_mapping(
            raw["initializer_work"], label="snapshot initializer"
        ),
        environment_budget_debit=_work_from_mapping(
            raw["environment_budget_debit"], label="snapshot environment debit"
        ),
        work_before=_work_from_mapping(raw["work_before"], label="snapshot work before"),
        work_after=_work_from_mapping(raw["work_after"], label="snapshot work after"),
        time_before_seconds=raw["time_before_seconds"],
        time_after_seconds=raw["time_after_seconds"],
        execution_status=raw["execution_status"],
        success=raw["success"],
        failure_reason=raw["failure_reason"],
        initialization_seconds=raw["initialization_seconds"],
    )


def episode_bootstrap_outcome_from_payload(
    raw: Mapping[str, object],
) -> EpisodeBootstrapOutcome:
    """Strictly authenticate and decode a serialized bootstrap/cache outcome."""

    expected = {
        "schema",
        "schema_version",
        "episode_schedule_index",
        "instance_id",
        "base_lineage",
        "public_task_digest",
        "initial_snapshot",
        "restart_cache_snapshots",
        "restart_cache_slot_count",
        "initial_generation_debit",
        "restart_cache_fill_debit",
        "total_pre_policy_debit",
        "precomputed_online_seconds",
        "plan_record_digest",
        "manifest_record_digest",
        "config_digest",
        "context_digest",
        "opened_evaluator_targets",
        "record_digest",
    }
    if set(raw) != expected:
        raise ValueError("episode bootstrap outcome schema differs")
    _verify_record(raw, label="episode bootstrap outcome")
    if (
        raw["schema"] != "isingfold.episode-bootstrap-outcome"
        or raw["schema_version"] != 1
        or raw["opened_evaluator_targets"] is not False
    ):
        raise ValueError("unsupported or unsafe episode bootstrap outcome")
    initial = raw["initial_snapshot"]
    cache = raw["restart_cache_snapshots"]
    if not isinstance(initial, Mapping) or not isinstance(cache, list):
        raise ValueError("episode bootstrap snapshots are malformed")
    if any(not isinstance(item, Mapping) for item in cache):
        raise ValueError("episode bootstrap cache snapshot is malformed")
    return EpisodeBootstrapOutcome(
        episode_schedule_index=raw["episode_schedule_index"],
        instance_id=raw["instance_id"],
        base_lineage=raw["base_lineage"],
        public_task_digest=raw["public_task_digest"],
        initial_snapshot=_snapshot_from_payload(initial),
        restart_cache_snapshots=tuple(
            _snapshot_from_payload(item)
            for item in cache
        ),
        restart_cache_slot_count=raw["restart_cache_slot_count"],
        initial_generation_debit=_work_from_mapping(
            raw["initial_generation_debit"], label="bootstrap initial debit"
        ),
        restart_cache_fill_debit=_work_from_mapping(
            raw["restart_cache_fill_debit"], label="bootstrap cache-fill debit"
        ),
        total_pre_policy_debit=_work_from_mapping(
            raw["total_pre_policy_debit"], label="bootstrap total debit"
        ),
        precomputed_online_seconds=raw["precomputed_online_seconds"],
        plan_record_digest=raw["plan_record_digest"],
        manifest_record_digest=raw["manifest_record_digest"],
        config_digest=raw["config_digest"],
        context_digest=raw["context_digest"],
    )


def environment_from_bootstrap_outcome(
    bootstrap: EpisodeBootstrapOutcome,
    *,
    task: EmbeddingTask,
    context: Context,
    selector: StrengthSelector,
    reward_reads: int | None = None,
) -> EmbeddingEnv:
    """Reconstruct one deployment environment from authenticated bootstrap evidence.

    This adapter is shared by train and validation consumers.  It does not generate or
    resample an initializer result.  Failed initial bootstraps are denominator outcomes and
    therefore cannot instantiate an actor environment.
    """

    if not isinstance(bootstrap, EpisodeBootstrapOutcome):
        raise TypeError("bootstrap must be an EpisodeBootstrapOutcome")
    if not isinstance(task, EmbeddingTask) or not isinstance(context, Context):
        raise TypeError("bootstrap environment needs a typed task and context")
    if stable_digest(_jsonable(context)) != bootstrap.context_digest:
        raise ValueError("environment context differs from bootstrap evidence")
    if (
        task.name != bootstrap.instance_id
        or task.lineage != bootstrap.base_lineage
        or _public_task_digest(task, instance_id=bootstrap.instance_id)
        != bootstrap.public_task_digest
    ):
        raise ValueError("environment task differs from bootstrap public identity")
    initial = bootstrap.initial_snapshot
    if not initial.success or initial.selected_embedding is None:
        raise InitializerSnapshotUnavailable(
            initial.failure_reason or "initializer failed"
        )
    if bootstrap.restart_cache_slot_count != context.restart_allowance:
        raise ValueError("bootstrap cache width differs from the registered restart allowance")

    initial_chains = _decode_embedding(task, initial.selected_embedding)
    cache_slots = tuple(
        RestartCacheSlot(
            slot_index=snapshot.slot_index,
            status=snapshot.execution_status,
            chains=(
                None
                if snapshot.selected_embedding is None
                else _decode_embedding(task, snapshot.selected_embedding)
            ),
            snapshot_record_digest=snapshot.record_digest,
            attempt_receipt_root=stable_digest(
                [attempt.as_dict() for attempt in snapshot.attempts]
            ),
        )
        for snapshot in bootstrap.restart_cache_snapshots
    )
    used = False

    def initializer(logical, host, seed):
        nonlocal used
        if logical is not task.logical or host is not task.host:
            raise CompleteSystemBackendError(
                "environment changed bootstrap snapshot graph identity"
            )
        if seed != initial.system_seed:
            raise CompleteSystemBackendError(
                "environment changed bootstrap snapshot system seed"
            )
        if used:
            raise CompleteSystemBackendError(
                "bootstrap initializer snapshot was requested more than once"
            )
        used = True
        return dict(initial_chains)

    from isingfold.rl.proposal import AUTHENTICATED_RESTART_CACHE_V1

    return EmbeddingEnv(
        task,
        context,
        mode=Mode.IMPROVEMENT,
        initializer=initializer,
        selector=selector,
        reward_reads=(context.n_est_reads if reward_reads is None else reward_reads),
        budget_debit=bootstrap.total_pre_policy_debit,
        initializer_precomputed=True,
        restart_cache=cache_slots,
        restart_cache_manifest_digest=bootstrap.manifest_record_digest,
        improvement_restart_protocol=AUTHENTICATED_RESTART_CACHE_V1,
        seed=initial.system_seed,
    )


def _validate_snapshot(
    snapshot: InitializerSnapshot,
    draw: InitializerBankDraw,
    *,
    plan: InitializerBankPlan,
    task: EmbeddingTask | None = None,
    context: Context | None = None,
) -> None:
    if (
        snapshot.plan_record_digest != plan.record_digest
        or snapshot.conditional_episode_index != draw.conditional_episode_index
        or snapshot.draw_index != draw.draw_index
        or snapshot.slot_kind != draw.slot_kind
        or snapshot.slot_index != draw.slot_index
        or snapshot.draw_key != draw.draw_key
        or snapshot.instance_id != draw.instance_id
        or snapshot.base_lineage != draw.base_lineage
        or snapshot.system_seed != draw.system_seed
    ):
        raise ValueError("initializer snapshot differs from its sealed draw identity")
    if draw.slot_kind == "initial" and (
        snapshot.work_before != WorkVector() or snapshot.time_before_seconds != 0.0
    ):
        raise ValueError("initial snapshot cannot inherit prior episode work or time")
    seeds = tuple(attempt.seed for attempt in snapshot.attempts)
    if seeds != draw.attempt_seeds[: len(seeds)]:
        raise ValueError("initializer snapshot attempt seeds differ from the sealed retry schedule")
    if len(seeds) > len(draw.attempt_seeds):
        raise ValueError("initializer snapshot exceeds the sealed attempt cap")
    for attempt in snapshot.attempts:
        runtime = attempt.backend_diagnostics.get("runtime_implementation_manifest")
        if _jsonable(runtime) != _jsonable(plan.runtime_implementation_manifest):
            raise ValueError("initializer snapshot attempt changed the runtime identity")
    valid_attempts = [
        attempt for attempt in snapshot.attempts if attempt.status == "VALID_CANDIDATE"
    ]
    if snapshot.success:
        if not valid_attempts:
            raise ValueError("successful initializer snapshot has no valid attempt")
        selected = min(
            valid_attempts,
            key=lambda attempt: (
                attempt.qubits,
                attempt.max_chain,
                attempt.embedding_digest,
                attempt.attempt_index,
            ),
        )
        if snapshot.selected_attempt != selected.attempt_index:
            raise ValueError("initializer snapshot changed resource-lexicographic selection")
    if task is not None:
        if snapshot.selected_embedding is not None:
            chains = _decode_embedding(task, snapshot.selected_embedding)
            validation = p_embed(chains, task.logical, task.host, context.qubit_cap if context else 0)
            if not validation.valid:
                raise ValueError("initializer snapshot selected embedding no longer validates")
            attempt = snapshot.attempts[snapshot.selected_attempt]
            if (
                _embedding_digest(chains) != snapshot.selected_embedding_digest
                or stable_digest(validation.as_dict()) != attempt.validation_digest
                or validation.qubits != attempt.qubits
                or validation.max_chain != attempt.max_chain
            ):
                raise ValueError("initializer snapshot selected embedding evidence differs")
        if _public_task_digest(task, instance_id=snapshot.instance_id) != next(
            item.public_task_digest
            for item in plan.tasks
            if item.instance_id == snapshot.instance_id
        ):
            raise ValueError("initializer snapshot public task identity differs")
    if context is not None and snapshot.success:
        policy_context = context
        if not (
            snapshot.work_after
            + WorkVector(compiler_calls=4, validator_calls=1)
            + policy_context.reserve
        ).fits_in(policy_context.caps):
            raise ValueError("initializer snapshot leaves insufficient policy work reserve")


def _snapshot_files(directory: Path) -> dict[str, list[tuple[Path, InitializerSnapshot]]]:
    snapshots = directory / "snapshots"
    if not snapshots.exists():
        return {}
    if not snapshots.is_dir():
        raise ValueError("initializer snapshot path is not a directory")
    result: dict[str, list[tuple[Path, InitializerSnapshot]]] = {}
    for path in sorted(snapshots.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise ValueError("initializer snapshot directory contains an unknown artifact")
        raw = _strict_json(path.read_bytes(), label=f"initializer snapshot {path.name}")
        snapshot = _snapshot_from_payload(raw)
        result.setdefault(snapshot.draw_key, []).append((path, snapshot))
    return result


def _validate_plan_execution_inputs(
    plan: InitializerBankPlan,
    prepared_tasks: Sequence[PreparedTask],
    *,
    initializer: object,
    runtime_implementation_manifest: Mapping[str, object],
    config: CompleteSystemConfig,
    context: Context,
    allow_test_backend: bool,
) -> dict[str, EmbeddingTask]:
    rebuilt = build_initializer_bank_plan(
        prepared_tasks,
        prepared_manifest_sha256=plan.prepared_manifest_sha256,
        training_seed=plan.training_seed,
        episode_schedule_start=plan.episode_schedule_start,
        episode_count=(
            plan.episode_schedule_stop_exclusive - plan.episode_schedule_start
        ),
        max_draws_per_conditional_episode=plan.max_draws_per_conditional_episode,
        initializer=initializer,
        runtime_implementation_manifest=runtime_implementation_manifest,
        config=config,
        context=context,
        allow_test_backend=allow_test_backend,
    )
    if rebuilt.as_dict() != plan.as_dict():
        raise ValueError("initializer bank plan differs from current authenticated inputs")
    _, tasks, _ = _validated_train_registry(prepared_tasks)
    return tasks


def generate_initializer_bank(
    directory: str | Path,
    plan: InitializerBankPlan,
    prepared_tasks: Sequence[PreparedTask],
    *,
    initializer: object,
    runtime_implementation_manifest: Mapping[str, object],
    config: CompleteSystemConfig,
    context: Context,
    episode_indices: Sequence[int] | None = None,
    allow_test_backend: bool = False,
) -> tuple[InitializerSnapshot, ...]:
    """Generate or resume complete initializer draws for selected conditional episodes."""

    if not isinstance(plan, InitializerBankPlan):
        raise TypeError("initializer bank generation requires a typed plan")
    root = Path(directory)
    if (root / "manifest.json").exists():
        raise ValueError("sealed initializer bank is immutable")
    tasks = _validate_plan_execution_inputs(
        plan,
        prepared_tasks,
        initializer=initializer,
        runtime_implementation_manifest=runtime_implementation_manifest,
        config=config,
        context=context,
        allow_test_backend=allow_test_backend,
    )
    write_initializer_bank_plan(root / "plan.json", plan)
    all_indices = {episode.episode_schedule_index for episode in plan.episodes}
    requested = all_indices if episode_indices is None else set(episode_indices)
    if any(type(index) is not int for index in requested) or not requested <= all_indices:
        raise ValueError("requested initializer-bank episode lies outside the sealed schedule")
    if episode_indices is not None and len(requested) != len(episode_indices):
        raise ValueError("initializer-bank generation repeats an episode index")
    files = _snapshot_files(root)
    known_draw_keys = {
        draw.draw_key
        for episode in plan.episodes
        for draw in (*episode.draws, *episode.restart_draws)
    }
    unknown = sorted(set(files) - known_draw_keys)
    if unknown:
        raise ValueError("initializer snapshot directory contains an unregistered draw")
    for draw_key, rows in files.items():
        if len(rows) != 1:
            raise ValueError(f"duplicate initializer snapshot for draw {draw_key}")

    returned: list[InitializerSnapshot] = []
    for episode in plan.episodes:
        if episode.episode_schedule_index not in requested:
            continue
        prefix: list[InitializerSnapshot] = []
        missing_seen = False
        for draw in episode.draws:
            existing = files.get(draw.draw_key, [])
            if existing:
                if missing_seen:
                    raise ValueError("initializer snapshot draw prefix contains a gap")
                snapshot = existing[0][1]
                _validate_snapshot(
                    snapshot,
                    draw,
                    plan=plan,
                    task=tasks[draw.instance_id],
                    context=context,
                )
                prefix.append(snapshot)
            else:
                missing_seen = True
        successes = [item for item in prefix if item.success]
        if len(successes) > 1 or (successes and prefix[-1] is not successes[0]):
            raise ValueError("conditional initializer draws continue after success")
        if not successes:
            for draw in episode.draws[len(prefix) :]:
                snapshot = _execute_draw(
                    draw,
                    plan=plan,
                    task=tasks[draw.instance_id],
                    initializer=initializer,
                    config=config,
                    context=context,
                )
                _validate_snapshot(
                    snapshot,
                    draw,
                    plan=plan,
                    task=tasks[draw.instance_id],
                    context=context,
                )
                payload = _canonical_bytes(snapshot.as_dict())
                filename = (
                    f"{draw.conditional_episode_index:012d}-{draw.slot_kind}-"
                    f"{draw.slot_index:04d}-{draw.draw_key}-{snapshot.record_digest}.json"
                )
                path = root / "snapshots" / filename
                _write_once(path, payload)
                files.setdefault(draw.draw_key, []).append((path, snapshot))
                prefix.append(snapshot)
                if snapshot.success:
                    break
        successes = [item for item in prefix if item.success]
        if not successes:
            if any(files.get(draw.draw_key) for draw in episode.restart_draws):
                raise ValueError(
                    "restart-cache snapshots exist before a successful initial bootstrap"
                )
            returned.extend(prefix)
            continue
        selected_initial = successes[0]
        cache_work = WorkVector()
        cache_seconds = 0.0
        missing_cache_seen = False
        for draw in episode.restart_draws:
            existing = files.get(draw.draw_key, [])
            if existing:
                if missing_cache_seen:
                    raise ValueError("restart-cache snapshot sequence contains a gap")
                snapshot = existing[0][1]
                _validate_snapshot(
                    snapshot,
                    draw,
                    plan=plan,
                    task=tasks[draw.instance_id],
                    context=context,
                )
            else:
                missing_cache_seen = True
                snapshot = _execute_draw(
                    draw,
                    plan=plan,
                    task=tasks[draw.instance_id],
                    initializer=initializer,
                    config=config,
                    context=context,
                    work_before=selected_initial.initializer_work + cache_work,
                    time_before_seconds=(
                        selected_initial.initialization_seconds + cache_seconds
                    ),
                )
                _validate_snapshot(
                    snapshot,
                    draw,
                    plan=plan,
                    task=tasks[draw.instance_id],
                    context=context,
                )
                payload = _canonical_bytes(snapshot.as_dict())
                filename = (
                    f"{draw.conditional_episode_index:012d}-{draw.slot_kind}-"
                    f"{draw.slot_index:04d}-{draw.draw_key}-{snapshot.record_digest}.json"
                )
                path = root / "snapshots" / filename
                _write_once(path, payload)
                files.setdefault(draw.draw_key, []).append((path, snapshot))
            expected_work_before = selected_initial.initializer_work + cache_work
            expected_time_before = (
                selected_initial.initialization_seconds + cache_seconds
            )
            if (
                snapshot.work_before != expected_work_before
                or snapshot.time_before_seconds != expected_time_before
            ):
                raise ValueError("restart-cache cumulative execution ledger differs")
            cache_work = cache_work + snapshot.initializer_work
            cache_seconds += snapshot.initialization_seconds
            returned.append(snapshot)
        returned.extend(prefix)
    return tuple(
        sorted(
            returned,
            key=lambda item: (
                item.conditional_episode_index,
                item.slot_kind != "initial",
                item.slot_index,
            ),
        )
    )


def _episode_resolution(
    episode: InitializerBankEpisode,
    files: Mapping[str, Sequence[tuple[Path, InitializerSnapshot]]],
    *,
    plan: InitializerBankPlan,
) -> tuple[list[tuple[Path, InitializerSnapshot]], InitializerSnapshot]:
    prefix: list[tuple[Path, InitializerSnapshot]] = []
    missing_seen = False
    for draw in episode.draws:
        rows = files.get(draw.draw_key, ())
        if len(rows) > 1:
            raise ValueError(f"duplicate initializer snapshot for draw {draw.draw_key}")
        if not rows:
            missing_seen = True
            continue
        if missing_seen:
            raise ValueError("initializer snapshot draw prefix contains a gap")
        path, snapshot = rows[0]
        _validate_snapshot(snapshot, draw, plan=plan)
        prefix.append((path, snapshot))
    if not prefix:
        raise ValueError(
            f"missing initializer snapshots for conditional episode "
            f"{episode.episode_schedule_index}"
        )
    successes = [(path, snapshot) for path, snapshot in prefix if snapshot.success]
    if not successes:
        if len(prefix) < len(episode.draws):
            raise ValueError(
                f"missing initializer snapshots for conditional episode "
                f"{episode.episode_schedule_index}"
            )
        raise InitializerSnapshotUnavailable(
            "conditional initializer episode exhausted its sealed draw cap without success"
        )
    if len(successes) != 1 or prefix[-1] != successes[0]:
        raise ValueError("conditional initializer draws continue after success")
    return prefix, successes[0][1]


def _restart_resolution(
    episode: InitializerBankEpisode,
    files: Mapping[str, Sequence[tuple[Path, InitializerSnapshot]]],
    *,
    plan: InitializerBankPlan,
    initial_snapshot: InitializerSnapshot,
) -> list[tuple[Path, InitializerSnapshot]]:
    rows: list[tuple[Path, InitializerSnapshot]] = []
    expected_work = initial_snapshot.initializer_work
    expected_time = initial_snapshot.initialization_seconds
    for draw in episode.restart_draws:
        matches = files.get(draw.draw_key, ())
        if len(matches) > 1:
            raise ValueError(f"duplicate initializer snapshot for draw {draw.draw_key}")
        if not matches:
            raise ValueError(
                "missing restart initializer snapshot for conditional episode "
                f"{episode.episode_schedule_index}, slot {draw.slot_index}"
            )
        path, snapshot = matches[0]
        _validate_snapshot(snapshot, draw, plan=plan)
        if (
            snapshot.work_before != expected_work
            or snapshot.time_before_seconds != expected_time
        ):
            raise ValueError("restart-cache cumulative execution ledger differs")
        rows.append((path, snapshot))
        expected_work = snapshot.work_after
        expected_time = snapshot.time_after_seconds
    return rows


def seal_initializer_bank(
    directory: str | Path,
    plan: InitializerBankPlan,
) -> str:
    """Seal a complete bank only when every requested PPO slot resolves successfully."""

    if not isinstance(plan, InitializerBankPlan):
        raise TypeError("initializer bank sealing requires a typed plan")
    root = Path(directory)
    plan_bytes = _canonical_bytes(plan.as_dict())
    plan_path = root / "plan.json"
    if not plan_path.is_file() or plan_path.read_bytes() != plan_bytes:
        raise ValueError("initializer bank plan artifact differs from the sealed plan")
    files = _snapshot_files(root)
    known_draw_keys = {
        draw.draw_key
        for episode in plan.episodes
        for draw in (*episode.draws, *episode.restart_draws)
    }
    unknown = sorted(set(files) - known_draw_keys)
    if unknown:
        raise ValueError("initializer snapshot directory contains an unregistered draw")
    for key, rows in files.items():
        if len(rows) > 1:
            raise ValueError(f"duplicate initializer snapshot for draw {key}")

    resolutions: list[dict[str, object]] = []
    file_rows: list[dict[str, object]] = []
    failure_digests: list[str] = []
    failed_draw_work = WorkVector()
    selected_initial_debit = WorkVector()
    restart_cache_fill_debit = WorkVector()
    total_pre_policy_debit = WorkVector()
    precomputed_online_seconds = 0.0
    generation_work = WorkVector()
    for episode in plan.episodes:
        prefix, selected = _episode_resolution(episode, files, plan=plan)
        restart_rows = _restart_resolution(
            episode,
            files,
            plan=plan,
            initial_snapshot=selected,
        )
        for path, snapshot in (*prefix, *restart_rows):
            raw = path.read_bytes()
            expected_name = (
                f"{snapshot.conditional_episode_index:012d}-{snapshot.slot_kind}-"
                f"{snapshot.slot_index:04d}-{snapshot.draw_key}-"
                f"{snapshot.record_digest}.json"
            )
            if path.name != expected_name:
                raise ValueError("initializer snapshot filename is not content addressed")
            file_rows.append(
                {
                    "conditional_episode_index": snapshot.conditional_episode_index,
                    "draw_index": snapshot.draw_index,
                    "slot_kind": snapshot.slot_kind,
                    "slot_index": snapshot.slot_index,
                    "draw_key": snapshot.draw_key,
                    "filename": f"snapshots/{path.name}",
                    "record_digest": snapshot.record_digest,
                    "sha256": _sha256_bytes(raw),
                    "success": snapshot.success,
                }
            )
            if not snapshot.success:
                failure_digests.append(snapshot.record_digest)
                failed_draw_work = failed_draw_work + snapshot.initializer_work
            generation_work = generation_work + snapshot.initializer_work
        rejected_work = WorkVector()
        for _, snapshot in prefix[:-1]:
            rejected_work = rejected_work + snapshot.initializer_work
        episode_cache_debit = _sum_snapshot_work(restart_rows)
        episode_total_debit = selected.environment_budget_debit + episode_cache_debit
        episode_seconds = (
            selected.initialization_seconds
            if not restart_rows
            else restart_rows[-1][1].time_after_seconds
        )
        selected_initial_debit = (
            selected_initial_debit + selected.environment_budget_debit
        )
        restart_cache_fill_debit = restart_cache_fill_debit + episode_cache_debit
        total_pre_policy_debit = total_pre_policy_debit + episode_total_debit
        precomputed_online_seconds += episode_seconds
        resolutions.append(
            {
                "conditional_episode_index": episode.episode_schedule_index,
                "instance_id": episode.instance_id,
                "base_lineage": episode.base_lineage,
                "selected_draw_index": selected.draw_index,
                "selected_record_digest": selected.record_digest,
                "draw_record_digests": [snapshot.record_digest for _, snapshot in prefix],
                "restart_record_digests": [
                    snapshot.record_digest for _, snapshot in restart_rows
                ],
                "restart_failure_count": sum(
                    not snapshot.success for _, snapshot in restart_rows
                ),
                "failed_draw_count": len(prefix) - 1,
                "rejected_draw_work": rejected_work.as_dict(),
                "initial_generation_debit": selected.environment_budget_debit.as_dict(),
                "restart_cache_fill_debit": episode_cache_debit.as_dict(),
                "total_pre_policy_debit": episode_total_debit.as_dict(),
                "precomputed_online_seconds": episode_seconds,
            }
        )
    resolution_digest = stable_digest(resolutions)
    snapshot_root_digest = stable_digest(file_rows)
    body = {
        "schema": INITIALIZER_BANK_MANIFEST_SCHEMA,
        "schema_version": INITIALIZER_BANK_MANIFEST_VERSION,
        "plan_record_digest": plan.record_digest,
        "plan_sha256": _sha256_bytes(plan_bytes),
        "prepared_manifest_sha256": plan.prepared_manifest_sha256,
        "partition": INITIALIZER_BANK_PARTITION,
        "training_seed": plan.training_seed,
        "conditional_episode_count": len(plan.episodes),
        "executed_draw_count": len(file_rows),
        "failed_draw_count": len(failure_digests),
        "max_draws_per_conditional_episode": plan.max_draws_per_conditional_episode,
        "restart_cache_slots_per_episode": plan.restart_cache_slots_per_episode,
        "resolutions": resolutions,
        "resolution_digest": resolution_digest,
        "snapshot_files": file_rows,
        "snapshot_root_digest": snapshot_root_digest,
        "failure_record_digests": failure_digests,
        "failed_draw_work": failed_draw_work.as_dict(),
        "selected_initial_debit": selected_initial_debit.as_dict(),
        "restart_cache_fill_debit": restart_cache_fill_debit.as_dict(),
        "total_pre_policy_debit": total_pre_policy_debit.as_dict(),
        "precomputed_online_seconds": precomputed_online_seconds,
        "generation_work": generation_work.as_dict(),
        "opened_evaluator_targets": False,
        "publication_eligible": plan.publication_eligible,
    }
    manifest = {**body, "record_digest": stable_digest(body)}
    payload = _canonical_bytes(manifest)
    _write_once(root / "manifest.json", payload)
    return _sha256_bytes(payload)


@dataclass(frozen=True)
class InitializerBankAccessReceipt:
    manifest_sha256: str
    manifest_record_digest: str
    plan_record_digest: str
    prepared_manifest_sha256: str
    partition: str
    conditional_episode_count: int
    executed_draw_count: int
    failed_draw_count: int
    failure_record_digests: tuple[str, ...]
    failed_draw_work: WorkVector
    selected_initial_debit: WorkVector
    restart_cache_fill_debit: WorkVector
    total_pre_policy_debit: WorkVector
    precomputed_online_seconds: float
    generation_work: WorkVector
    resolution_digest: str
    snapshot_root_digest: str
    opened_evaluator_targets: bool = False

    def as_dict(self) -> dict[str, object]:
        body = {
            "schema": INITIALIZER_BANK_ACCESS_SCHEMA,
            "schema_version": INITIALIZER_BANK_ACCESS_VERSION,
            "manifest_sha256": self.manifest_sha256,
            "manifest_record_digest": self.manifest_record_digest,
            "plan_record_digest": self.plan_record_digest,
            "prepared_manifest_sha256": self.prepared_manifest_sha256,
            "partition": self.partition,
            "conditional_episode_count": self.conditional_episode_count,
            "executed_draw_count": self.executed_draw_count,
            "failed_draw_count": self.failed_draw_count,
            "failure_record_digests": list(self.failure_record_digests),
            "failed_draw_work": self.failed_draw_work.as_dict(),
            "selected_initial_debit": self.selected_initial_debit.as_dict(),
            "restart_cache_fill_debit": self.restart_cache_fill_debit.as_dict(),
            "total_pre_policy_debit": self.total_pre_policy_debit.as_dict(),
            "precomputed_online_seconds": self.precomputed_online_seconds,
            "generation_work": self.generation_work.as_dict(),
            "resolution_digest": self.resolution_digest,
            "snapshot_root_digest": self.snapshot_root_digest,
            "opened_evaluator_targets": self.opened_evaluator_targets,
        }
        return {**body, "record_digest": stable_digest(body)}

    @property
    def record_digest(self) -> str:
        return str(self.as_dict()["record_digest"])


class InitializerSnapshotBank:
    """Verified conditional schedule backed by immutable deployment initializer draws."""

    def __init__(
        self,
        *,
        plan: InitializerBankPlan,
        tasks: Mapping[str, EmbeddingTask],
        draws_by_episode: Mapping[int, tuple[InitializerSnapshot, ...]],
        selected_by_episode: Mapping[int, InitializerSnapshot],
        restart_cache_by_episode: Mapping[int, tuple[InitializerSnapshot, ...]],
        access_receipt: InitializerBankAccessReceipt,
    ) -> None:
        self.plan = plan
        self._tasks = MappingProxyType(dict(tasks))
        self._draws = MappingProxyType(dict(draws_by_episode))
        self._selected = MappingProxyType(dict(selected_by_episode))
        self._restart_cache = MappingProxyType(dict(restart_cache_by_episode))
        self.access_receipt = access_receipt

    def __len__(self) -> int:
        return len(self._selected)

    def snapshot(self, episode_schedule_index: int) -> InitializerSnapshot:
        try:
            return self._selected[episode_schedule_index]
        except KeyError as exc:
            raise KeyError(
                f"episode {episode_schedule_index} is outside the sealed bank"
            ) from exc

    def draws(self, episode_schedule_index: int) -> tuple[InitializerSnapshot, ...]:
        try:
            return self._draws[episode_schedule_index]
        except KeyError as exc:
            raise KeyError(
                f"episode {episode_schedule_index} is outside the sealed bank"
            ) from exc

    def restart_cache_snapshots(
        self, episode_schedule_index: int
    ) -> tuple[InitializerSnapshot, ...]:
        try:
            return self._restart_cache[episode_schedule_index]
        except KeyError as exc:
            raise KeyError(
                f"episode {episode_schedule_index} is outside the sealed bank"
            ) from exc

    def bootstrap_outcome(
        self, episode_schedule_index: int
    ) -> EpisodeBootstrapOutcome:
        initial = self.snapshot(episode_schedule_index)
        cache = self.restart_cache_snapshots(episode_schedule_index)
        identity = next(
            item for item in self.plan.tasks if item.instance_id == initial.instance_id
        )
        cache_debit = WorkVector()
        for snapshot in cache:
            cache_debit = cache_debit + snapshot.initializer_work
        elapsed = (
            initial.initialization_seconds
            if not cache
            else cache[-1].time_after_seconds
        )
        return EpisodeBootstrapOutcome(
            episode_schedule_index=episode_schedule_index,
            instance_id=initial.instance_id,
            base_lineage=initial.base_lineage,
            public_task_digest=identity.public_task_digest,
            initial_snapshot=initial,
            restart_cache_snapshots=cache,
            restart_cache_slot_count=self.plan.restart_cache_slots_per_episode,
            initial_generation_debit=initial.initializer_work,
            restart_cache_fill_debit=cache_debit,
            total_pre_policy_debit=initial.initializer_work + cache_debit,
            precomputed_online_seconds=elapsed,
            plan_record_digest=self.plan.record_digest,
            manifest_record_digest=self.access_receipt.manifest_record_digest,
            config_digest=self.plan.config_digest,
            context_digest=self.plan.context_digest,
        )

    def environment(
        self,
        episode_schedule_index: int,
        *,
        context: Context,
        selector: StrengthSelector,
        reward_reads: int | None = None,
        training_task: PreparedTask | None = None,
        target_access: Mapping[str, object] | None = None,
        ground_partition_receipt: Mapping[str, object] | None = None,
    ) -> EmbeddingEnv:
        """Create the exact post-initializer environment for one conditional PPO slot.

        ``training_task`` may attach a train-only evaluator target for PPO reward after the
        bank has been authenticated. Its partition, source row, base lineage and public
        graph/problem identity must equal the bank. Neither the target nor its certificate
        is copied into a bank artifact or receipt.
        """

        if stable_digest(_jsonable(context)) != self.plan.context_digest:
            raise ValueError("environment context differs from the initializer bank")
        snapshot = self.snapshot(episode_schedule_index)
        public_task = self._tasks[snapshot.instance_id]
        if training_task is None:
            if target_access is not None or ground_partition_receipt is not None:
                raise ValueError("quality receipts require an explicit training target")
            task = public_task
        else:
            if target_access is None or ground_partition_receipt is None:
                raise ValueError(
                    "training target injection requires authenticated target-access and "
                    "ground-partition receipts"
                )
            identity = next(
                item for item in self.plan.tasks if item.instance_id == snapshot.instance_id
            )
            if (
                not isinstance(training_task, PreparedTask)
                or training_task.partition != INITIALIZER_BANK_PARTITION
                or training_task.prepared_schema_version != PREPARED_SCHEMA_VERSION_V4
                or training_task.corpus_scope != INITIALIZER_BANK_CORPUS_SCOPE
                or training_task.provenance is None
                or training_task.design_condition is None
                or training_task.task_id not in identity.source_task_ids
                or training_task.instance_id != snapshot.instance_id
                or training_task.task.lineage != snapshot.base_lineage
                or training_task.provenance.base_parent_lineage != snapshot.base_lineage
                or training_task.design_condition.base_lineage_key != snapshot.base_lineage
                or training_task.design_condition.learning_partition
                != INITIALIZER_BANK_PARTITION
                or _public_task_digest(
                    training_task.task, instance_id=snapshot.instance_id
                )
                != _public_task_digest(public_task, instance_id=snapshot.instance_id)
            ):
                raise ValueError(
                    "training task partition or public identity differs from bank snapshot"
                )
            validate_target_task_binding(
                training_task,
                target_access=target_access,
                ground_partition_receipt=ground_partition_receipt,
                expected_prepared_manifest_sha256=self.plan.prepared_manifest_sha256,
            )
            task = dataclasses.replace(
                training_task.task,
                name=snapshot.instance_id,
                witness=None,
                initial_embedding=None,
            )
        if snapshot.selected_embedding is None:
            raise InitializerSnapshotUnavailable(snapshot.failure_reason or "initializer failed")
        bootstrap = self.bootstrap_outcome(episode_schedule_index)
        return environment_from_bootstrap_outcome(
            bootstrap,
            task=task,
            context=context,
            selector=selector,
            reward_reads=reward_reads,
        )


def _manifest_exact_fields() -> set[str]:
    return {
        "schema",
        "schema_version",
        "plan_record_digest",
        "plan_sha256",
        "prepared_manifest_sha256",
        "partition",
        "training_seed",
        "conditional_episode_count",
        "executed_draw_count",
        "failed_draw_count",
        "max_draws_per_conditional_episode",
        "restart_cache_slots_per_episode",
        "resolutions",
        "resolution_digest",
        "snapshot_files",
        "snapshot_root_digest",
        "failure_record_digests",
        "failed_draw_work",
        "selected_initial_debit",
        "restart_cache_fill_debit",
        "total_pre_policy_debit",
        "precomputed_online_seconds",
        "generation_work",
        "opened_evaluator_targets",
        "publication_eligible",
        "record_digest",
    }


def _sum_snapshot_work(
    rows: Sequence[tuple[Path, InitializerSnapshot]],
) -> WorkVector:
    total = WorkVector()
    for _, snapshot in rows:
        total = total + snapshot.initializer_work
    return total


def load_initializer_bank(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
    prepared_tasks: Sequence[PreparedTask],
    prepared_manifest_sha256: str,
    initializer: object,
    runtime_implementation_manifest: Mapping[str, object],
    config: CompleteSystemConfig,
    context: Context,
    allow_test_backend: bool = False,
) -> InitializerSnapshotBank:
    """Open a bank against out-of-band pins and current train-only runtime identities."""

    expected_manifest_sha256 = _require_digest(
        expected_manifest_sha256, label="expected initializer bank manifest"
    )
    root = Path(directory)
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("initializer bank manifest is missing")
    manifest_bytes = manifest_path.read_bytes()
    if _sha256_bytes(manifest_bytes) != expected_manifest_sha256:
        raise ValueError("initializer bank manifest SHA-256 pin mismatch")
    manifest = _strict_json(manifest_bytes, label="initializer bank manifest")
    if set(manifest) != _manifest_exact_fields():
        raise ValueError("initializer bank manifest schema differs")
    manifest_record_digest = _verify_record(
        manifest, label="initializer bank manifest"
    )
    if (
        manifest["schema"] != INITIALIZER_BANK_MANIFEST_SCHEMA
        or manifest["schema_version"] != INITIALIZER_BANK_MANIFEST_VERSION
        or manifest["partition"] != INITIALIZER_BANK_PARTITION
        or manifest["opened_evaluator_targets"] is not False
    ):
        raise ValueError("unsupported or unsafe initializer bank manifest")
    for name in (
        "training_seed",
        "conditional_episode_count",
        "executed_draw_count",
        "failed_draw_count",
        "max_draws_per_conditional_episode",
        "restart_cache_slots_per_episode",
    ):
        value = manifest[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"initializer bank manifest {name} is not a count")
    if manifest["conditional_episode_count"] <= 0 or manifest[
        "max_draws_per_conditional_episode"
    ] <= 0:
        raise ValueError("initializer bank manifest has an empty conditional schedule")
    for name in (
        "plan_record_digest",
        "plan_sha256",
        "prepared_manifest_sha256",
        "resolution_digest",
        "snapshot_root_digest",
    ):
        _require_digest(manifest[name], label=f"initializer bank manifest {name}")
    plan_path = root / "plan.json"
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError("initializer bank plan is missing")
    plan_bytes = plan_path.read_bytes()
    if _sha256_bytes(plan_bytes) != manifest["plan_sha256"]:
        raise ValueError("initializer bank plan SHA-256 mismatch")
    plan_payload = _strict_json(plan_bytes, label="initializer bank plan")
    if _verify_record(plan_payload, label="initializer bank plan") != manifest[
        "plan_record_digest"
    ]:
        raise ValueError("initializer bank plan and manifest identities differ")
    start = plan_payload.get("episode_schedule_start")
    stop = plan_payload.get("episode_schedule_stop_exclusive")
    max_draws = plan_payload.get("max_draws_per_conditional_episode")
    training_seed = plan_payload.get("training_seed")
    if (
        type(start) is not int
        or type(stop) is not int
        or stop <= start
        or type(max_draws) is not int
        or max_draws <= 0
        or type(training_seed) is not int
        or training_seed < 0
    ):
        raise ValueError("initializer bank plan schedule is malformed")
    rebuilt = build_initializer_bank_plan(
        prepared_tasks,
        prepared_manifest_sha256=prepared_manifest_sha256,
        training_seed=training_seed,
        episode_schedule_start=start,
        episode_count=stop - start,
        max_draws_per_conditional_episode=max_draws,
        initializer=initializer,
        runtime_implementation_manifest=runtime_implementation_manifest,
        config=config,
        context=context,
        allow_test_backend=allow_test_backend,
    )
    if rebuilt.as_dict() != plan_payload:
        raise ValueError("initializer bank plan differs from current authenticated inputs")
    if (
        manifest["prepared_manifest_sha256"] != rebuilt.prepared_manifest_sha256
        or manifest["training_seed"] != rebuilt.training_seed
        or manifest["conditional_episode_count"] != len(rebuilt.episodes)
        or manifest["max_draws_per_conditional_episode"]
        != rebuilt.max_draws_per_conditional_episode
        or manifest["restart_cache_slots_per_episode"]
        != rebuilt.restart_cache_slots_per_episode
        or manifest["publication_eligible"] is not rebuilt.publication_eligible
    ):
        raise ValueError("initializer bank manifest differs from its authenticated plan")
    _, tasks, _ = _validated_train_registry(prepared_tasks)

    file_rows = manifest["snapshot_files"]
    resolutions = manifest["resolutions"]
    failure_digests = manifest["failure_record_digests"]
    if (
        not isinstance(file_rows, list)
        or not isinstance(resolutions, list)
        or not isinstance(failure_digests, list)
        or stable_digest(file_rows) != manifest["snapshot_root_digest"]
        or stable_digest(resolutions) != manifest["resolution_digest"]
    ):
        raise ValueError("initializer bank manifest roots are inconsistent")
    expected_filenames: set[str] = set()
    loaded_by_key: dict[str, tuple[Path, InitializerSnapshot]] = {}
    for row in file_rows:
        if not isinstance(row, Mapping) or set(row) != {
            "conditional_episode_index",
            "draw_index",
            "slot_kind",
            "slot_index",
            "draw_key",
            "filename",
            "record_digest",
            "sha256",
            "success",
        }:
            raise ValueError("initializer bank snapshot-file registry schema differs")
        filename = row["filename"]
        if (
            not isinstance(filename, str)
            or not filename.startswith("snapshots/")
            or Path(filename).is_absolute()
            or ".." in Path(filename).parts
            or filename in expected_filenames
        ):
            raise ValueError("initializer bank snapshot filename is unsafe or duplicated")
        expected_filenames.add(filename)
        path = root / filename
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"missing initializer snapshot {filename}")
        raw_bytes = path.read_bytes()
        if _sha256_bytes(raw_bytes) != row["sha256"]:
            raise ValueError("initializer snapshot SHA-256 mismatch")
        snapshot = _snapshot_from_payload(
            _strict_json(raw_bytes, label=f"initializer snapshot {filename}")
        )
        expected_filename = (
            f"snapshots/{snapshot.conditional_episode_index:012d}-"
            f"{snapshot.slot_kind}-{snapshot.slot_index:04d}-{snapshot.draw_key}-"
            f"{snapshot.record_digest}.json"
        )
        if (
            filename != expected_filename
            or not _is_digest(row["sha256"])
            or snapshot.record_digest != row["record_digest"]
            or snapshot.draw_key != row["draw_key"]
            or snapshot.conditional_episode_index != row["conditional_episode_index"]
            or snapshot.draw_index != row["draw_index"]
            or snapshot.slot_kind != row["slot_kind"]
            or snapshot.slot_index != row["slot_index"]
            or snapshot.success is not row["success"]
            or snapshot.draw_key in loaded_by_key
        ):
            raise ValueError("initializer snapshot differs from its manifest registry")
        loaded_by_key[snapshot.draw_key] = (path, snapshot)
    snapshot_dir = root / "snapshots"
    if snapshot_dir.is_symlink() or not snapshot_dir.is_dir():
        raise ValueError("initializer snapshot directory is missing or unsafe")
    snapshot_entries = tuple(snapshot_dir.iterdir())
    if any(
        path.is_symlink() or not path.is_file() or path.suffix != ".json"
        for path in snapshot_entries
    ):
        raise ValueError("initializer snapshot directory contains an unknown artifact")
    observed_filenames = {f"snapshots/{path.name}" for path in snapshot_entries}
    if observed_filenames != expected_filenames:
        missing = sorted(expected_filenames - observed_filenames)
        extra = sorted(observed_filenames - expected_filenames)
        raise ValueError(
            f"initializer snapshot file coverage differs: missing={missing}, extra={extra}"
        )
    known_draw_keys = {
        draw.draw_key
        for episode in rebuilt.episodes
        for draw in (*episode.draws, *episode.restart_draws)
    }
    if set(loaded_by_key) - known_draw_keys:
        raise ValueError("initializer bank contains an unregistered draw")

    draws_by_episode: dict[int, tuple[InitializerSnapshot, ...]] = {}
    selected_by_episode: dict[int, InitializerSnapshot] = {}
    restart_cache_by_episode: dict[int, tuple[InitializerSnapshot, ...]] = {}
    observed_resolutions: list[dict[str, object]] = []
    observed_failed_work = WorkVector()
    observed_selected_debit = WorkVector()
    observed_cache_debit = WorkVector()
    observed_total_pre_policy_debit = WorkVector()
    observed_precomputed_seconds = 0.0
    observed_generation_work = WorkVector()
    snapshot_rows_by_key = {
        key: (value,) for key, value in loaded_by_key.items()
    }
    for episode in rebuilt.episodes:
        prefix, selected = _episode_resolution(
            episode,
            snapshot_rows_by_key,
            plan=rebuilt,
        )
        restart_rows = _restart_resolution(
            episode,
            snapshot_rows_by_key,
            plan=rebuilt,
            initial_snapshot=selected,
        )
        for _, snapshot in prefix:
            _validate_snapshot(
                snapshot,
                episode.draws[snapshot.draw_index],
                plan=rebuilt,
                task=tasks[snapshot.instance_id],
                context=context,
            )
        draws_by_episode[episode.episode_schedule_index] = tuple(
            snapshot for _, snapshot in prefix
        )
        selected_by_episode[episode.episode_schedule_index] = selected
        restart_cache_by_episode[episode.episode_schedule_index] = tuple(
            snapshot for _, snapshot in restart_rows
        )
        episode_cache_debit = _sum_snapshot_work(restart_rows)
        episode_total_debit = selected.environment_budget_debit + episode_cache_debit
        episode_seconds = (
            selected.initialization_seconds
            if not restart_rows
            else restart_rows[-1][1].time_after_seconds
        )
        observed_resolutions.append(
            {
                "conditional_episode_index": episode.episode_schedule_index,
                "instance_id": episode.instance_id,
                "base_lineage": episode.base_lineage,
                "selected_draw_index": selected.draw_index,
                "selected_record_digest": selected.record_digest,
                "draw_record_digests": [snapshot.record_digest for _, snapshot in prefix],
                "restart_record_digests": [
                    snapshot.record_digest for _, snapshot in restart_rows
                ],
                "restart_failure_count": sum(
                    not snapshot.success for _, snapshot in restart_rows
                ),
                "failed_draw_count": len(prefix) - 1,
                "rejected_draw_work": _sum_snapshot_work(prefix[:-1]).as_dict(),
                "initial_generation_debit": selected.environment_budget_debit.as_dict(),
                "restart_cache_fill_debit": episode_cache_debit.as_dict(),
                "total_pre_policy_debit": episode_total_debit.as_dict(),
                "precomputed_online_seconds": episode_seconds,
            }
        )
        observed_failed_work = observed_failed_work + _sum_snapshot_work(
            tuple(row for row in (*prefix, *restart_rows) if not row[1].success)
        )
        observed_selected_debit = (
            observed_selected_debit + selected.environment_budget_debit
        )
        observed_cache_debit = observed_cache_debit + episode_cache_debit
        observed_total_pre_policy_debit = (
            observed_total_pre_policy_debit + episode_total_debit
        )
        observed_precomputed_seconds += episode_seconds
        observed_generation_work = observed_generation_work + _sum_snapshot_work(
            (*prefix, *restart_rows)
        )
    observed_failures = tuple(
        snapshot.record_digest
        for index in sorted(draws_by_episode)
        for snapshot in (
            *draws_by_episode[index],
            *tuple(
                loaded_by_key[draw.draw_key][1]
                for draw in rebuilt.episodes[index - rebuilt.episode_schedule_start].restart_draws
            ),
        )
        if not snapshot.success
    )
    if (
        resolutions != observed_resolutions
        or file_rows
        != sorted(
            file_rows,
            key=lambda row: (
                row["conditional_episode_index"],
                row["slot_kind"] != "initial",
                row["slot_index"],
            ),
        )
        or failure_digests != list(observed_failures)
        or manifest["executed_draw_count"]
        != sum(map(len, draws_by_episode.values()))
        + len(rebuilt.episodes) * rebuilt.restart_cache_slots_per_episode
        or manifest["failed_draw_count"] != len(observed_failures)
        or _work_from_mapping(
            manifest["failed_draw_work"], label="manifest failed-draw"
        )
        != observed_failed_work
        or _work_from_mapping(
            manifest["selected_initial_debit"],
            label="manifest selected initial debit",
        )
        != observed_selected_debit
        or _work_from_mapping(
            manifest["restart_cache_fill_debit"],
            label="manifest restart-cache fill debit",
        )
        != observed_cache_debit
        or _work_from_mapping(
            manifest["total_pre_policy_debit"],
            label="manifest total pre-policy debit",
        )
        != observed_total_pre_policy_debit
        or manifest["precomputed_online_seconds"] != observed_precomputed_seconds
        or _work_from_mapping(manifest["generation_work"], label="manifest generation")
        != observed_generation_work
    ):
        raise ValueError("initializer bank failure accounting differs from its draws")
    access = InitializerBankAccessReceipt(
        manifest_sha256=expected_manifest_sha256,
        manifest_record_digest=manifest_record_digest,
        plan_record_digest=rebuilt.record_digest,
        prepared_manifest_sha256=rebuilt.prepared_manifest_sha256,
        partition=INITIALIZER_BANK_PARTITION,
        conditional_episode_count=len(rebuilt.episodes),
        executed_draw_count=(
            sum(map(len, draws_by_episode.values()))
            + len(rebuilt.episodes) * rebuilt.restart_cache_slots_per_episode
        ),
        failed_draw_count=len(observed_failures),
        failure_record_digests=observed_failures,
        failed_draw_work=observed_failed_work,
        selected_initial_debit=observed_selected_debit,
        restart_cache_fill_debit=observed_cache_debit,
        total_pre_policy_debit=observed_total_pre_policy_debit,
        precomputed_online_seconds=observed_precomputed_seconds,
        generation_work=observed_generation_work,
        resolution_digest=str(manifest["resolution_digest"]),
        snapshot_root_digest=str(manifest["snapshot_root_digest"]),
        opened_evaluator_targets=False,
    )
    return InitializerSnapshotBank(
        plan=rebuilt,
        tasks=tasks,
        draws_by_episode=draws_by_episode,
        selected_by_episode=selected_by_episode,
        restart_cache_by_episode=restart_cache_by_episode,
        access_receipt=access,
    )


__all__ = [
    "EpisodeBootstrapOutcome",
    "InitializerBankAccessReceipt",
    "InitializerBankDraw",
    "InitializerBankEpisode",
    "InitializerBankPlan",
    "InitializerBankTaskIdentity",
    "InitializerSnapshot",
    "InitializerSnapshotBank",
    "InitializerSnapshotUnavailable",
    "build_initializer_bank_plan",
    "environment_from_bootstrap_outcome",
    "episode_bootstrap_outcome_from_payload",
    "execute_initializer_draw",
    "generate_initializer_bank",
    "lac_runtime_implementation_manifest",
    "load_initializer_bank",
    "load_initializer_bank_plan",
    "public_task_digest",
    "seal_initializer_bank",
    "validate_target_task_binding",
    "validate_initializer_backend_contract",
    "write_initializer_bank_plan",
]
