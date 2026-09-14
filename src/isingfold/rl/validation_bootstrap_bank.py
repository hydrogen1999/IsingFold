"""Target-free validation bootstrap banks shared by every same-support arm.

RL-value selection is paired only if every controller sees the same deployment initializer
and the same persistent restart cache. This module owns that immutable validation artifact.
It deliberately does not run LAC itself. Its execution boundary is the authenticated
``EpisodeBootstrapOutcome`` produced by the initializer-bank core. Evaluator targets are
opened only by the later evaluation runner, after this bank and its external pins validate.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import dataclasses
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Protocol, Sequence

from isingfold.rl.complete_system import (
    COMPLETE_POLICY_RESTART_MODE,
    LAC_INITIALIZER_METHOD_ID,
    CompleteSystemConfig,
)
from isingfold.rl.contracts import WORK_FIELDS, Context, WorkVector, stable_digest
from isingfold.rl.data.import_embedbench import PREPARED_SCHEMA_VERSION_V4
from isingfold.rl.data.prepared import PreparedTask
from isingfold.rl.env import EmbeddingTask
from isingfold.rl.external import BackendIdentity
from isingfold.rl.initializer_bank import (
    EpisodeBootstrapOutcome,
    InitializerBankDraw,
    InitializerBankTaskIdentity,
    InitializerSnapshot,
    episode_bootstrap_outcome_from_payload,
    execute_initializer_draw,
    public_task_digest,
    validate_initializer_backend_contract,
)
from isingfold.rl.proposal import PROPOSAL_VERSION

REPRESENTATION_EVALUATION_SEED = 33_049
RL_VALUE_EVALUATION_SEED = 44_021
RL_VALUE_FINAL_TEST_SEED = 55_079
RL_VALUE_REGISTERED_REPETITIONS = 4
VALIDATION_RESTART_CACHE_SLOTS = 2
VALIDATION_BOOTSTRAP_PARTITION = "val"
FINAL_TEST_BOOTSTRAP_PARTITION = "test"
VALIDATION_BOOTSTRAP_CORPUS_SCOPE = "production-designed-v4"
REPRESENTATION_VALIDATION_BOOTSTRAP_PRESET = "representation-validation"
VALIDATION_BOOTSTRAP_PRESET = "validation"
FINAL_TEST_BOOTSTRAP_PRESET = "final-test"


@dataclass(frozen=True)
class BootstrapProtocolPreset:
    name: str
    partition: str
    evaluation_seed: int
    repetitions: int


REGISTERED_BOOTSTRAP_PROTOCOLS = MappingProxyType(
    {
        REPRESENTATION_VALIDATION_BOOTSTRAP_PRESET: BootstrapProtocolPreset(
            name=REPRESENTATION_VALIDATION_BOOTSTRAP_PRESET,
            partition=VALIDATION_BOOTSTRAP_PARTITION,
            evaluation_seed=REPRESENTATION_EVALUATION_SEED,
            repetitions=RL_VALUE_REGISTERED_REPETITIONS,
        ),
        VALIDATION_BOOTSTRAP_PRESET: BootstrapProtocolPreset(
            name=VALIDATION_BOOTSTRAP_PRESET,
            partition=VALIDATION_BOOTSTRAP_PARTITION,
            evaluation_seed=RL_VALUE_EVALUATION_SEED,
            repetitions=RL_VALUE_REGISTERED_REPETITIONS,
        ),
        FINAL_TEST_BOOTSTRAP_PRESET: BootstrapProtocolPreset(
            name=FINAL_TEST_BOOTSTRAP_PRESET,
            partition=FINAL_TEST_BOOTSTRAP_PARTITION,
            evaluation_seed=RL_VALUE_FINAL_TEST_SEED,
            repetitions=RL_VALUE_REGISTERED_REPETITIONS,
        ),
    }
)

VALIDATION_BOOTSTRAP_PLAN_SCHEMA = "isingfold.rl-value-bootstrap-plan"
VALIDATION_BOOTSTRAP_PLAN_VERSION = 1
VALIDATION_BOOTSTRAP_RECORD_SCHEMA = "isingfold.rl-value-bootstrap-record"
VALIDATION_BOOTSTRAP_RECORD_VERSION = 1
VALIDATION_BOOTSTRAP_MANIFEST_SCHEMA = "isingfold.rl-value-bootstrap-manifest"
VALIDATION_BOOTSTRAP_MANIFEST_VERSION = 1
VALIDATION_BOOTSTRAP_ACCESS_SCHEMA = "isingfold.rl-value-bootstrap-access"
VALIDATION_BOOTSTRAP_ACCESS_VERSION = 1
VALIDATION_BOOTSTRAP_CLONE_SCHEMA = "isingfold.rl-value-bootstrap-clone"
VALIDATION_BOOTSTRAP_CLONE_VERSION = 1

BOOTSTRAP_SAME_SUPPORT_SCHEMA = "isingfold.bootstrap-same-support-contract"
BOOTSTRAP_SAME_SUPPORT_VERSION = 1
BOOTSTRAP_CLONE_RULE = "byte-identical-bootstrap-per-same-support-consumer-v1"
BOOTSTRAP_FAILURE_RULE = "denominator-eligible-zero-no-actor-no-cache-v1"

_POLICY_ENTRY_RESERVE = WorkVector(compiler_calls=4, validator_calls=1)
_FORBIDDEN_TARGET_KEYS = frozenset(
    {
        "ground_energy",
        "witness",
        "reference_status",
        "certificate_digest",
        "evaluator_protocol_digest",
        "quality_attestation_digest",
        "quality_evidence_manifest_digest",
        "quality_evidence_manifest_sha256",
        "quality_target_set_digest",
        "quality_target_count",
        "reference_energy",
        "claimed_reference_energy",
        "evaluator_target",
        "evaluator_targets",
        "evaluator_targets_sha256",
        "target_energy",
        "known_optimum",
        "optimum_energy",
        "ground_truth",
        "evaluator_hits",
        "utility",
    }
)


class BootstrapOutcomeProvider(Protocol):
    """Narrow adapter implemented by the authenticated initializer/cache executor."""

    def as_dict(self) -> dict[str, object]: ...


def _jsonable(value: object) -> object:
    """Normalize authenticated values without depending on another module's internals."""

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
    raise TypeError(
        f"cannot serialize {type(value).__name__} in validation-bootstrap identity"
    )


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _digest(value: object, label: str) -> str:
    if not _is_digest(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    assert isinstance(value, str)
    return value


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


def _file_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _strict_json(content: bytes, *, label: str) -> dict[str, object]:
    def pairs(items: Sequence[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"{label} contains non-finite number {token}")

    parsed = json.loads(content, object_pairs_hook=pairs, parse_constant=reject_constant)
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _verify_record(payload: Mapping[str, object], *, label: str) -> str:
    observed = _digest(payload.get("record_digest"), f"{label} record digest")
    body = {key: value for key, value in payload.items() if key != "record_digest"}
    if stable_digest(body) != observed:
        raise ValueError(f"{label} record digest mismatch")
    return observed


def _write_new(path: Path, content: bytes) -> None:
    """Create one artifact with O_EXCL; pre-existing bytes are never replaced."""

    parent = path.parent
    if parent.exists() and (parent.is_symlink() or not parent.is_dir()):
        raise ValueError(f"validation-bootstrap artifact parent is unsafe: {parent}")
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"validation-bootstrap artifact parent is unsafe: {parent}")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise FileExistsError(f"immutable validation-bootstrap artifact exists: {path}") from exc
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_bank_root(directory: Path, *, sealed: bool) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("validation-bootstrap root is missing or unsafe")
    allowed = {"plan.json", "records"}
    if sealed:
        allowed.add("manifest.json")
    unknown = sorted(path.name for path in directory.iterdir() if path.name not in allowed)
    if unknown:
        raise ValueError(
            "validation-bootstrap root contains an unknown top-level artifact: "
            f"{unknown[0]}"
        )


def _work(raw: object, *, label: str) -> WorkVector:
    if not isinstance(raw, Mapping) or set(raw) != set(WORK_FIELDS):
        raise ValueError(f"{label} has missing or unknown work coordinates")
    values: dict[str, int] = {}
    for name in WORK_FIELDS:
        value = raw[name]
        if type(value) is not int or value < 0:
            raise ValueError(f"{label} work coordinates must be nonnegative integers")
        values[name] = value
    return WorkVector(**values)


def _exact_nonnegative_int(value: object, *, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _exact_finite_float(value: object, *, label: str) -> float:
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative JSON float")
    return value


def _nonempty_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def registered_bootstrap_protocol(
    *, partition: str, evaluation_seed: int, repetitions: int
) -> BootstrapProtocolPreset:
    """Resolve exactly one registered target-free evaluation population."""

    if (
        not isinstance(partition, str)
        or type(evaluation_seed) is not int
        or type(repetitions) is not int
    ):
        raise ValueError("bootstrap protocol pins have invalid types")
    matches = [
        preset
        for preset in REGISTERED_BOOTSTRAP_PROTOCOLS.values()
        if (
            preset.partition == partition
            and preset.evaluation_seed == evaluation_seed
            and preset.repetitions == repetitions
        )
    ]
    if len(matches) != 1:
        raise ValueError(
            "bootstrap protocol is not a registered representation-validation, "
            "validation, or final-test preset"
        )
    return matches[0]


def bootstrap_context_digest(context: Context) -> str:
    if not isinstance(context, Context):
        raise TypeError("bootstrap context identity requires a typed Context")
    return stable_digest(_jsonable(context))


def bootstrap_same_support_contract(
    *,
    partition: str,
    evaluation_seed: int,
    repetitions: int,
    config_digest: str,
    context_digest: str,
    proposal_version: str = PROPOSAL_VERSION,
) -> dict[str, object]:
    """Build the canonical pairing contract shared by all arms in one population."""

    preset = registered_bootstrap_protocol(
        partition=partition,
        evaluation_seed=evaluation_seed,
        repetitions=repetitions,
    )
    config_digest = _digest(config_digest, "same-support config")
    context_digest = _digest(context_digest, "same-support context")
    proposal_version = _nonempty_text(
        proposal_version, label="same-support proposal version"
    )
    body = {
        "schema": BOOTSTRAP_SAME_SUPPORT_SCHEMA,
        "schema_version": BOOTSTRAP_SAME_SUPPORT_VERSION,
        "protocol_preset": preset.name,
        "partition": preset.partition,
        "evaluation_seed": preset.evaluation_seed,
        "repetitions": preset.repetitions,
        "config_digest": config_digest,
        "context_digest": context_digest,
        "restart_cache_slots": VALIDATION_RESTART_CACHE_SLOTS,
        "proposal_version": proposal_version,
        "census_unit": "immutable-base-lineage-instance-repetition-v1",
        "clone_rule": BOOTSTRAP_CLONE_RULE,
        "initial_failure_rule": BOOTSTRAP_FAILURE_RULE,
        "target_access": "forbidden-until-bank-authentication-v1",
    }
    return {**body, "record_digest": stable_digest(body)}


def bootstrap_same_support_contract_digest(
    *,
    partition: str,
    evaluation_seed: int,
    repetitions: int,
    config_digest: str,
    context_digest: str,
    proposal_version: str = PROPOSAL_VERSION,
) -> str:
    return str(
        bootstrap_same_support_contract(
            partition=partition,
            evaluation_seed=evaluation_seed,
            repetitions=repetitions,
            config_digest=config_digest,
            context_digest=context_digest,
            proposal_version=proposal_version,
        )["record_digest"]
    )


def _identity_seed(
    base_seed: int,
    domain: str,
    *,
    instance_id: str,
    base_lineage: str,
    repetition: int,
) -> int:
    payload = {
        "base_seed": base_seed,
        "domain": domain,
        "instance": instance_id,
        "lineage": base_lineage,
        "repetition": repetition,
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


def _assert_target_free(value: object, *, path: str = "bootstrap") -> None:
    """Reject evaluator-bearing data while allowing explicit false access flags."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _FORBIDDEN_TARGET_KEYS:
                raise ValueError(f"{path} contains forbidden evaluator target field {key!r}")
            if key in {"opened_evaluator_targets", "evaluator_target_opened"} and item is not False:
                raise ValueError(f"{path} claims evaluator-target access")
            _assert_target_free(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _assert_target_free(item, path=f"{path}[{index}]")


@dataclass(frozen=True)
class ValidationBootstrapCensusRow:
    census_index: int
    base_lineage: str
    instance_id: str
    repetition: int
    system_seed: int
    initial_attempt_seeds: tuple[int, ...]
    cache_system_seeds: tuple[int, ...]
    cache_attempt_seeds: tuple[tuple[int, ...], ...]
    row_key: str

    def as_dict(self) -> dict[str, object]:
        return {
            "census_index": self.census_index,
            "base_lineage": self.base_lineage,
            "instance_id": self.instance_id,
            "repetition": self.repetition,
            "system_seed": self.system_seed,
            "initial_attempt_seeds": list(self.initial_attempt_seeds),
            "cache_system_seeds": list(self.cache_system_seeds),
            "cache_attempt_seeds": [list(row) for row in self.cache_attempt_seeds],
            "row_key": self.row_key,
        }


@dataclass(frozen=True)
class ValidationBootstrapPlan:
    prepared_manifest_sha256: str
    protocol_registry_sha256: str
    protocol_record_digest: str
    same_support_contract_digest: str
    protocol_preset: str
    partition: str
    evaluation_seed: int
    repetitions: int
    restart_cache_slots: int
    tasks: tuple[InitializerBankTaskIdentity, ...]
    census: tuple[ValidationBootstrapCensusRow, ...]
    initializer_identity: BackendIdentity
    runtime_implementation_manifest: Mapping[str, object]
    runtime_implementation_digest: str
    complete_system_config: Mapping[str, object]
    config_digest: str
    context: Mapping[str, object]
    context_digest: str
    publication_eligible: bool

    @property
    def census_digest(self) -> str:
        return stable_digest([row.as_dict() for row in self.census])

    def as_dict(self) -> dict[str, object]:
        identity = getattr(self.initializer_identity, "as_dict", None)
        if not callable(identity):
            raise TypeError("validation-bootstrap initializer identity is not serializable")
        body = {
            "schema": VALIDATION_BOOTSTRAP_PLAN_SCHEMA,
            "schema_version": VALIDATION_BOOTSTRAP_PLAN_VERSION,
            "prepared_manifest_sha256": self.prepared_manifest_sha256,
            "prepared_schema_version": PREPARED_SCHEMA_VERSION_V4,
            "corpus_scope": VALIDATION_BOOTSTRAP_CORPUS_SCOPE,
            "protocol_registry_sha256": self.protocol_registry_sha256,
            "protocol_record_digest": self.protocol_record_digest,
            "same_support_contract_digest": self.same_support_contract_digest,
            "protocol_preset": self.protocol_preset,
            "partition": self.partition,
            "evaluation_seed": self.evaluation_seed,
            "repetitions": self.repetitions,
            "restart_cache_slots": self.restart_cache_slots,
            "census_order": "base-lineage-instance-repetition-lexicographic-v1",
            "tasks": [task.as_dict() for task in self.tasks],
            "census": [row.as_dict() for row in self.census],
            "census_digest": self.census_digest,
            "initializer_identity": identity(),
            "runtime_implementation_manifest": _jsonable(self.runtime_implementation_manifest),
            "runtime_implementation_digest": self.runtime_implementation_digest,
            "complete_system_config": _jsonable(self.complete_system_config),
            "config_digest": self.config_digest,
            "context": _jsonable(self.context),
            "context_digest": self.context_digest,
            "population_estimand": "unconditional-if-q3-s0-utility",
            "initial_failure_policy": BOOTSTRAP_FAILURE_RULE,
            "clone_policy": BOOTSTRAP_CLONE_RULE,
            "opened_evaluator_targets": False,
            "publication_eligible": self.publication_eligible,
            "test_only_backend": not self.publication_eligible,
        }
        return {**body, "record_digest": stable_digest(body)}

    @property
    def record_digest(self) -> str:
        return str(self.as_dict()["record_digest"])


def _validated_protocol_registry(
    prepared_tasks: Sequence[PreparedTask], *, partition: str
) -> tuple[InitializerBankTaskIdentity, ...]:
    if not prepared_tasks:
        raise ValueError("bootstrap needs at least one task in its registered partition")
    grouped: dict[str, list[PreparedTask]] = {}
    seen_task_ids: set[str] = set()
    for item in prepared_tasks:
        if not isinstance(item, PreparedTask):
            raise TypeError("validation bootstrap inputs must be typed PreparedTask records")
        if (
            item.partition != partition
            or item.design_condition is None
            or item.design_condition.learning_partition != partition
            or item.provenance is None
        ):
            raise ValueError("bootstrap task differs from its registered partition")
        if (
            item.prepared_schema_version != PREPARED_SCHEMA_VERSION_V4
            or item.corpus_scope != VALIDATION_BOOTSTRAP_CORPUS_SCOPE
        ):
            raise ValueError("validation bootstrap requires production prepared schema v4")
        lineage = item.task.lineage
        if (
            not isinstance(lineage, str)
            or not lineage
            or item.design_condition.base_lineage_key != lineage
            or item.provenance.base_parent_lineage != lineage
        ):
            raise ValueError("bootstrap task differs from its immutable base lineage")
        target_fields = (
            item.task.ground_energy,
            item.task.witness,
            item.reference_status,
            item.certificate_digest,
            item.evaluator_protocol_digest,
            item.quality_attestation_digest,
            item.quality_evidence_manifest_digest,
            item.quality_evidence_manifest_sha256,
            item.quality_target_set_digest,
            item.quality_target_count,
        )
        if any(value is not None for value in target_fields):
            raise ValueError("bootstrap construction cannot receive an evaluator target")
        if item.task_id in seen_task_ids:
            raise ValueError("bootstrap repeats a prepared task ID")
        seen_task_ids.add(item.task_id)
        grouped.setdefault(item.instance_id, []).append(item)

    identities: list[InitializerBankTaskIdentity] = []
    for instance_id, rows in sorted(grouped.items()):
        lineages = {row.task.lineage for row in rows}
        if len(lineages) != 1:
            raise ValueError("one bootstrap policy instance crosses base lineages")
        lineage = next(iter(lineages))
        assert isinstance(lineage, str)
        public_digests = {
            public_task_digest(row.task, instance_id=instance_id) for row in rows
        }
        if len(public_digests) != 1:
            raise ValueError("bootstrap rows disagree on their public policy instance")
        ordered = sorted(rows, key=lambda row: row.task_id)
        source_digests = tuple(
            stable_digest(
                {
                    "task_id": row.task_id,
                    "initializer_record_digest": row.initializer_record_digest,
                    "provenance_record_digest": row.provenance.record_digest,
                    "design_condition_digest": row.design_condition.registry_row_digest,
                }
            )
            for row in ordered
        )
        identities.append(
            InitializerBankTaskIdentity(
                instance_id=instance_id,
                base_lineage=lineage,
                source_task_ids=tuple(row.task_id for row in ordered),
                source_record_digests=source_digests,
                public_task_digest=next(iter(public_digests)),
            )
        )
    identities.sort(key=lambda row: (row.base_lineage, row.instance_id))
    return tuple(identities)


def _validate_prepared_census_authority(
    prepared_tasks: Sequence[PreparedTask],
    identities: Sequence[InitializerBankTaskIdentity],
    *,
    prepared_manifest: Mapping[str, object],
    prepared_split_registry: Mapping[str, object],
    prepared_manifest_sha256: str,
    partition: str,
) -> None:
    """Bind the plan census to authenticated prepared-v4 publication records.

    A manifest SHA by itself is only a caller claim.  The plan builder therefore
    reauthenticates the canonical manifest and its pinned split registry, then
    requires the supplied typed rows to be the *entire* registered partition.
    """

    if not isinstance(prepared_manifest, Mapping) or not isinstance(
        prepared_split_registry, Mapping
    ):
        raise TypeError("bootstrap census authority requires manifest and split mappings")
    _verify_record(prepared_manifest, label="prepared-v4 manifest")
    if not hmac.compare_digest(
        _file_sha256(_canonical_bytes(prepared_manifest)), prepared_manifest_sha256
    ):
        raise ValueError("prepared-v4 manifest bytes differ from their external SHA-256 pin")
    if (
        prepared_manifest.get("schema") != "isingfold.prepared-candidate-bank"
        or prepared_manifest.get("schema_version") != PREPARED_SCHEMA_VERSION_V4
        or prepared_manifest.get("corpus_scope")
        != VALIDATION_BOOTSTRAP_CORPUS_SCOPE
    ):
        raise ValueError("bootstrap census authority is not a production prepared-v4 manifest")

    _verify_record(prepared_split_registry, label="prepared-v4 split registry")
    if (
        prepared_split_registry.get("schema") != "isingfold.lineage-splits"
        or prepared_split_registry.get("schema_version")
        != PREPARED_SCHEMA_VERSION_V4
    ):
        raise ValueError("bootstrap census authority has an unsupported split registry")
    outputs = prepared_manifest.get("outputs")
    split_descriptor = outputs.get("splits.json") if isinstance(outputs, Mapping) else None
    if (
        not isinstance(split_descriptor, Mapping)
        or split_descriptor.get("records") != 1
        or split_descriptor.get("sha256")
        != _file_sha256(_canonical_bytes(prepared_split_registry))
    ):
        raise ValueError("prepared-v4 split registry differs from its manifest descriptor")
    registered_task_ids = prepared_split_registry.get(partition)
    supplied_task_ids = sorted(task.task_id for task in prepared_tasks)
    if (
        not isinstance(registered_task_ids, Sequence)
        or isinstance(registered_task_ids, (str, bytes, bytearray))
        or any(not isinstance(task_id, str) for task_id in registered_task_ids)
        or list(registered_task_ids) != sorted(registered_task_ids)
        or list(registered_task_ids) != supplied_task_ids
    ):
        raise ValueError(
            "bootstrap task rows do not equal the full authenticated partition census"
        )

    target_authority = prepared_manifest.get("target_authority")
    if not isinstance(target_authority, Mapping):
        raise ValueError("prepared-v4 manifest has no target-census authority")
    _verify_record(target_authority, label="prepared-v4 target authority")
    partition_descriptors = target_authority.get("partitions")
    target_descriptor = (
        partition_descriptors.get(partition)
        if isinstance(partition_descriptors, Mapping)
        else None
    )
    if (
        not isinstance(target_descriptor, Mapping)
        or target_descriptor.get("path") != f"targets/{partition}.jsonl"
        or target_descriptor.get("records") != len(identities)
    ):
        raise ValueError(
            "bootstrap policy-instance census differs from authenticated target authority"
        )
    counts = prepared_manifest.get("counts")
    by_partition = (
        counts.get("evaluator_targets_by_partition")
        if isinstance(counts, Mapping)
        else None
    )
    if (
        not isinstance(by_partition, Mapping)
        or by_partition.get(partition) != len(identities)
    ):
        raise ValueError("prepared-v4 manifest partition count differs from bootstrap census")
    target_output = (
        outputs.get(f"targets/{partition}.jsonl")
        if isinstance(outputs, Mapping)
        else None
    )
    if (
        not isinstance(target_output, Mapping)
        or target_output.get("records") != len(identities)
        or target_output.get("sha256") != target_descriptor.get("sha256")
    ):
        raise ValueError("prepared-v4 target output differs from its target authority")


def _census_row(
    *,
    census_index: int,
    identity: InitializerBankTaskIdentity,
    repetition: int,
    evaluation_seed: int,
    attempt_count: int,
    protocol_preset: str,
) -> ValidationBootstrapCensusRow:
    system_seed = _identity_seed(
        evaluation_seed,
        "policy",
        instance_id=identity.instance_id,
        base_lineage=identity.base_lineage,
        repetition=repetition,
    )
    initial_attempts = tuple(
        _attempt_seed(system_seed, LAC_INITIALIZER_METHOD_ID, index)
        for index in range(attempt_count)
    )
    cache_system_seeds = tuple(
        _identity_seed(
            evaluation_seed,
            f"policy-persistent-restart-cache-slot-{slot}-v1",
            instance_id=identity.instance_id,
            base_lineage=identity.base_lineage,
            repetition=repetition,
        )
        for slot in range(VALIDATION_RESTART_CACHE_SLOTS)
    )
    cache_attempts = tuple(
        tuple(
            _attempt_seed(cache_seed, LAC_INITIALIZER_METHOD_ID, index)
            for index in range(attempt_count)
        )
        for cache_seed in cache_system_seeds
    )
    key_body = {
        "schema": "isingfold.rl-value-bootstrap-census-row",
        "schema_version": 1,
        "protocol_preset": protocol_preset,
        "base_lineage": identity.base_lineage,
        "instance_id": identity.instance_id,
        "repetition": repetition,
        "evaluation_seed": evaluation_seed,
    }
    return ValidationBootstrapCensusRow(
        census_index=census_index,
        base_lineage=identity.base_lineage,
        instance_id=identity.instance_id,
        repetition=repetition,
        system_seed=system_seed,
        initial_attempt_seeds=initial_attempts,
        cache_system_seeds=cache_system_seeds,
        cache_attempt_seeds=cache_attempts,
        row_key=stable_digest(key_body),
    )


def build_bootstrap_plan(
    prepared_tasks: Sequence[PreparedTask],
    *,
    prepared_manifest: Mapping[str, object],
    prepared_split_registry: Mapping[str, object],
    prepared_manifest_sha256: str,
    protocol_registry_sha256: str,
    protocol_record_digest: str,
    same_support_contract_digest: str,
    partition: str,
    evaluation_seed: int,
    repetitions: int,
    initializer: object,
    runtime_implementation_manifest: Mapping[str, object],
    config: CompleteSystemConfig,
    context: Context,
    allow_test_backend: bool = False,
) -> ValidationBootstrapPlan:
    """Build one complete registered public census before any target is opened."""

    prepared_manifest_sha256 = _digest(
        prepared_manifest_sha256, "prepared manifest identity"
    )
    protocol_registry_sha256 = _digest(
        protocol_registry_sha256, "bootstrap protocol registry file"
    )
    protocol_record_digest = _digest(
        protocol_record_digest, "bootstrap protocol record"
    )
    same_support_contract_digest = _digest(
        same_support_contract_digest, "same-support contract"
    )
    preset = registered_bootstrap_protocol(
        partition=partition,
        evaluation_seed=evaluation_seed,
        repetitions=repetitions,
    )
    if not isinstance(config, CompleteSystemConfig) or not isinstance(context, Context):
        raise TypeError("validation bootstrap requires typed config and context")
    if config.policy_restart_mode != COMPLETE_POLICY_RESTART_MODE:
        raise ValueError(
            "bootstrap evaluation requires the registered persistent K=2 restart mode"
        )
    if context.restart_allowance != VALIDATION_RESTART_CACHE_SLOTS:
        raise ValueError("bootstrap evaluation fixes exactly two persistent cache slots")
    if config.audit_reads != context.audit_reads:
        raise ValueError("complete-system and context audit reads differ")
    if config.max_initializer_attempts > context.caps.restart_work:
        raise ValueError("initializer attempts exceed the registered restart-work cap")
    backend, runtime, runtime_digest, publication_eligible = validate_initializer_backend_contract(
        initializer,
        config,
        runtime_implementation_manifest,
        allow_test_backend=allow_test_backend,
    )
    if backend.method_id != LAC_INITIALIZER_METHOD_ID:
        raise ValueError("bootstrap evaluation requires the exact registered LAC initializer")
    tasks = _validated_protocol_registry(prepared_tasks, partition=preset.partition)
    _validate_prepared_census_authority(
        prepared_tasks,
        tasks,
        prepared_manifest=prepared_manifest,
        prepared_split_registry=prepared_split_registry,
        prepared_manifest_sha256=prepared_manifest_sha256,
        partition=preset.partition,
    )
    census: list[ValidationBootstrapCensusRow] = []
    for identity in tasks:
        for repetition in range(repetitions):
            census.append(
                _census_row(
                    census_index=len(census),
                    identity=identity,
                    repetition=repetition,
                    evaluation_seed=evaluation_seed,
                    attempt_count=config.max_initializer_attempts,
                    protocol_preset=preset.name,
                )
            )
    context_body = _jsonable(context)
    config_body = _jsonable(config.as_dict())
    assert isinstance(context_body, dict) and isinstance(config_body, dict)
    context_digest = bootstrap_context_digest(context)
    expected_support = bootstrap_same_support_contract_digest(
        partition=preset.partition,
        evaluation_seed=preset.evaluation_seed,
        repetitions=preset.repetitions,
        config_digest=config.digest,
        context_digest=context_digest,
    )
    if not hmac.compare_digest(same_support_contract_digest, expected_support):
        raise ValueError("same-support contract pin differs from registered semantics")
    return ValidationBootstrapPlan(
        prepared_manifest_sha256=prepared_manifest_sha256,
        protocol_registry_sha256=protocol_registry_sha256,
        protocol_record_digest=protocol_record_digest,
        same_support_contract_digest=same_support_contract_digest,
        protocol_preset=preset.name,
        partition=preset.partition,
        evaluation_seed=evaluation_seed,
        repetitions=repetitions,
        restart_cache_slots=VALIDATION_RESTART_CACHE_SLOTS,
        tasks=tasks,
        census=tuple(census),
        initializer_identity=backend,
        runtime_implementation_manifest=runtime,
        runtime_implementation_digest=runtime_digest,
        complete_system_config=MappingProxyType(config_body),
        config_digest=config.digest,
        context=MappingProxyType(context_body),
        context_digest=context_digest,
        publication_eligible=publication_eligible,
    )


def write_validation_bootstrap_plan(
    path: str | os.PathLike[str], plan: ValidationBootstrapPlan
) -> str:
    if not isinstance(plan, ValidationBootstrapPlan):
        raise TypeError("validation-bootstrap plan publication requires a typed plan")
    content = _canonical_bytes(plan.as_dict())
    _write_new(Path(path), content)
    return _file_sha256(content)


def load_bootstrap_plan(
    path: str | os.PathLike[str],
    *,
    expected_plan_sha256: str,
    prepared_tasks: Sequence[PreparedTask],
    prepared_manifest: Mapping[str, object],
    prepared_split_registry: Mapping[str, object],
    prepared_manifest_sha256: str,
    protocol_registry_sha256: str,
    protocol_record_digest: str,
    same_support_contract_digest: str,
    partition: str,
    evaluation_seed: int,
    repetitions: int,
    initializer: object,
    runtime_implementation_manifest: Mapping[str, object],
    config: CompleteSystemConfig,
    context: Context,
    allow_test_backend: bool = False,
) -> ValidationBootstrapPlan:
    expected = _digest(expected_plan_sha256, "expected validation-bootstrap plan SHA-256")
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("validation-bootstrap plan is missing or unsafe")
    content = source.read_bytes()
    if not hmac.compare_digest(_file_sha256(content), expected):
        raise ValueError("validation-bootstrap plan SHA-256 pin mismatch")
    payload = _strict_json(content, label="validation-bootstrap plan")
    _verify_record(payload, label="validation-bootstrap plan")
    rebuilt = build_bootstrap_plan(
        prepared_tasks,
        prepared_manifest=prepared_manifest,
        prepared_split_registry=prepared_split_registry,
        prepared_manifest_sha256=prepared_manifest_sha256,
        protocol_registry_sha256=protocol_registry_sha256,
        protocol_record_digest=protocol_record_digest,
        same_support_contract_digest=same_support_contract_digest,
        partition=partition,
        evaluation_seed=evaluation_seed,
        repetitions=repetitions,
        initializer=initializer,
        runtime_implementation_manifest=runtime_implementation_manifest,
        config=config,
        context=context,
        allow_test_backend=allow_test_backend,
    )
    if rebuilt.as_dict() != payload:
        raise ValueError("validation-bootstrap plan differs from authenticated public inputs")
    if content != _canonical_bytes(rebuilt.as_dict()):
        raise ValueError("validation-bootstrap plan is not canonical JSON")
    return rebuilt


@dataclass(frozen=True)
class ValidationBootstrapRecord:
    plan_record_digest: str
    row_key: str
    census_index: int
    base_lineage: str
    instance_id: str
    repetition: int
    evaluation_seed: int
    bootstrap_outcome_payload: Mapping[str, object]
    bootstrap_outcome_record_digest: str
    bootstrap_payload_sha256: str
    source_execution_plan_digest: str
    source_execution_manifest_digest: str
    initial_status: str
    cache_slot_count: int
    cache_success_count: int
    total_pre_policy_work: WorkVector
    precomputed_online_seconds: float
    denominator_eligible: bool
    fixed_utility: float | None
    actor_invocation_permitted: bool
    cache_invocation_permitted: bool

    def as_dict(self) -> dict[str, object]:
        body = {
            "schema": VALIDATION_BOOTSTRAP_RECORD_SCHEMA,
            "schema_version": VALIDATION_BOOTSTRAP_RECORD_VERSION,
            "plan_record_digest": self.plan_record_digest,
            "row_key": self.row_key,
            "census_index": self.census_index,
            "base_lineage": self.base_lineage,
            "instance_id": self.instance_id,
            "repetition": self.repetition,
            "evaluation_seed": self.evaluation_seed,
            "bootstrap_outcome": _jsonable(self.bootstrap_outcome_payload),
            "bootstrap_outcome_record_digest": self.bootstrap_outcome_record_digest,
            "bootstrap_payload_sha256": self.bootstrap_payload_sha256,
            "source_execution_plan_digest": self.source_execution_plan_digest,
            "source_execution_manifest_digest": self.source_execution_manifest_digest,
            "initial_status": self.initial_status,
            "cache_slot_count": self.cache_slot_count,
            "cache_success_count": self.cache_success_count,
            "total_pre_policy_work": self.total_pre_policy_work.as_dict(),
            "precomputed_online_seconds": self.precomputed_online_seconds,
            "denominator_eligible": self.denominator_eligible,
            "fixed_utility": self.fixed_utility,
            "actor_invocation_permitted": self.actor_invocation_permitted,
            "cache_invocation_permitted": self.cache_invocation_permitted,
            "opened_evaluator_targets": False,
        }
        return {**body, "record_digest": stable_digest(body)}

    @property
    def record_digest(self) -> str:
        return str(self.as_dict()["record_digest"])

    @property
    def bootstrap_payload_bytes(self) -> bytes:
        return _canonical_bytes(dict(self.bootstrap_outcome_payload))

    @property
    def bootstrap_outcome(self) -> EpisodeBootstrapOutcome:
        return episode_bootstrap_outcome_from_payload(dict(self.bootstrap_outcome_payload))


def _row_by_key(
    plan: ValidationBootstrapPlan, row_key: str
) -> ValidationBootstrapCensusRow:
    if not _is_digest(row_key):
        raise ValueError("validation-bootstrap row key must be a SHA-256 digest")
    matches = [row for row in plan.census if row.row_key == row_key]
    if len(matches) != 1:
        raise KeyError("validation-bootstrap row is outside the sealed census")
    return matches[0]


def _bootstrap_draw(
    plan: ValidationBootstrapPlan,
    row: ValidationBootstrapCensusRow,
    *,
    slot_kind: str,
    slot_index: int,
) -> InitializerBankDraw:
    if slot_kind == "initial":
        system_seed = row.system_seed
        attempt_seeds = row.initial_attempt_seeds
    elif slot_kind == "restart_cache" and 0 <= slot_index < plan.restart_cache_slots:
        system_seed = row.cache_system_seeds[slot_index]
        attempt_seeds = row.cache_attempt_seeds[slot_index]
    else:
        raise ValueError("bootstrap draw slot is outside the registered row")
    body = {
        "schema": "isingfold.rl-value-bootstrap-draw",
        "schema_version": 1,
        "plan_record_digest": plan.record_digest,
        "row_key": row.row_key,
        "census_index": row.census_index,
        "slot_kind": slot_kind,
        "slot_index": slot_index,
        "system_seed": system_seed,
        "attempt_seeds": list(attempt_seeds),
    }
    return InitializerBankDraw(
        conditional_episode_index=row.census_index,
        draw_index=slot_index,
        slot_kind=slot_kind,
        slot_index=slot_index,
        instance_id=row.instance_id,
        base_lineage=row.base_lineage,
        system_seed=system_seed,
        attempt_seeds=attempt_seeds,
        draw_key=stable_digest(body),
    )


def bootstrap_execution_record_digest(
    plan: ValidationBootstrapPlan,
    row_key: str,
    initial_snapshot: InitializerSnapshot,
    restart_cache_snapshots: Sequence[InitializerSnapshot],
) -> str:
    """Identity of the complete causal native transcript for one census row."""

    row = _row_by_key(plan, row_key)
    body = {
        "schema": "isingfold.rl-value-bootstrap-execution",
        "schema_version": 1,
        "plan_record_digest": plan.record_digest,
        "row_key": row.row_key,
        "initial_snapshot_record_digest": initial_snapshot.record_digest,
        "restart_cache_snapshot_record_digests": [
            snapshot.record_digest for snapshot in restart_cache_snapshots
        ],
    }
    return stable_digest(body)


def execute_bootstrap_row(
    plan: ValidationBootstrapPlan,
    row_key: str,
    public_task: EmbeddingTask,
    *,
    initializer: object,
    config: CompleteSystemConfig,
    context: Context,
    allow_test_backend: bool = False,
) -> EpisodeBootstrapOutcome:
    """Execute the exact initial LAC draw and causal persistent K=2 cache tape."""

    if not isinstance(plan, ValidationBootstrapPlan):
        raise TypeError("bootstrap execution requires a typed authenticated plan")
    if not isinstance(public_task, EmbeddingTask):
        raise TypeError("bootstrap execution requires a typed public EmbeddingTask")
    row = _row_by_key(plan, row_key)
    identity = next(item for item in plan.tasks if item.instance_id == row.instance_id)
    if (
        public_task.name != row.instance_id
        or public_task.lineage != row.base_lineage
        or public_task.ground_energy is not None
        or public_task.witness is not None
        or public_task.initial_embedding is not None
        or public_task_digest(public_task, instance_id=row.instance_id)
        != identity.public_task_digest
    ):
        raise ValueError("bootstrap execution task is not the sealed target-free public task")
    if (
        config.digest != plan.config_digest
        or bootstrap_context_digest(context) != plan.context_digest
    ):
        raise ValueError("bootstrap execution config or context differs from its plan")
    backend, runtime, runtime_digest, publication_eligible = (
        validate_initializer_backend_contract(
            initializer,
            config,
            plan.runtime_implementation_manifest,
            allow_test_backend=allow_test_backend,
        )
    )
    if (
        backend != plan.initializer_identity
        or _jsonable(runtime) != _jsonable(plan.runtime_implementation_manifest)
        or runtime_digest != plan.runtime_implementation_digest
        or publication_eligible != plan.publication_eligible
    ):
        raise ValueError("bootstrap execution changed its sealed initializer runtime")

    initial = execute_initializer_draw(
        _bootstrap_draw(plan, row, slot_kind="initial", slot_index=0),
        plan=plan,
        task=public_task,
        initializer=initializer,
        config=config,
        context=context,
    )
    if initial.execution_status not in {"SUCCESS", "FAILED"}:
        raise ValueError("registered bootstrap initial LAC draw was not invoked")
    cache: list[InitializerSnapshot] = []
    work = initial.work_after
    elapsed = initial.time_after_seconds
    if initial.success:
        for slot in range(plan.restart_cache_slots):
            snapshot = execute_initializer_draw(
                _bootstrap_draw(
                    plan, row, slot_kind="restart_cache", slot_index=slot
                ),
                plan=plan,
                task=public_task,
                initializer=initializer,
                config=config,
                context=context,
                work_before=work,
                time_before_seconds=elapsed,
            )
            cache.append(snapshot)
            work = snapshot.work_after
            elapsed = snapshot.time_after_seconds
    cache_debit = WorkVector()
    for snapshot in cache:
        cache_debit = cache_debit + snapshot.initializer_work
    outcome = EpisodeBootstrapOutcome(
        episode_schedule_index=row.census_index,
        instance_id=row.instance_id,
        base_lineage=row.base_lineage,
        public_task_digest=identity.public_task_digest,
        initial_snapshot=initial,
        restart_cache_snapshots=tuple(cache),
        restart_cache_slot_count=plan.restart_cache_slots,
        initial_generation_debit=initial.initializer_work,
        restart_cache_fill_debit=cache_debit,
        total_pre_policy_debit=initial.initializer_work + cache_debit,
        precomputed_online_seconds=elapsed,
        plan_record_digest=plan.record_digest,
        manifest_record_digest=bootstrap_execution_record_digest(
            plan, row.row_key, initial, cache
        ),
        config_digest=plan.config_digest,
        context_digest=plan.context_digest,
    )
    validation_bootstrap_record_from_outcome(plan, row.row_key, outcome)
    return outcome


def _outcome_from_provider(provider: BootstrapOutcomeProvider) -> EpisodeBootstrapOutcome:
    serializer = getattr(provider, "as_dict", None)
    if not callable(serializer):
        raise TypeError("bootstrap outcome provider must expose as_dict()")
    payload = serializer()
    if not isinstance(payload, Mapping):
        raise TypeError("bootstrap outcome provider returned a non-object payload")
    _assert_target_free(payload)
    return episode_bootstrap_outcome_from_payload(payload)


def _validate_attempt_tape(
    snapshot: object,
    planned: tuple[int, ...],
    *,
    label: str,
    runtime_manifest: Mapping[str, object],
) -> None:
    attempts = getattr(snapshot, "attempts", None)
    execution_status = getattr(snapshot, "execution_status", None)
    if not isinstance(attempts, tuple):
        raise TypeError(f"{label} initializer attempts have the wrong type")
    seeds = tuple(attempt.seed for attempt in attempts)
    if seeds != planned[: len(seeds)] or len(seeds) > len(planned):
        raise ValueError(f"{label} attempt seed schedule differs from the census")
    if execution_status in {"SUCCESS", "FAILED"} and not attempts:
        raise ValueError(f"{label} claims an invoked outcome without an attempt receipt")
    if execution_status in {"BUDGET_NOT_INVOKED", "TIME_NOT_INVOKED"} and attempts:
        raise ValueError(f"{label} uninvoked outcome contains attempt receipts")
    for attempt in attempts:
        observed_runtime = attempt.backend_diagnostics.get(
            "runtime_implementation_manifest"
        )
        if _jsonable(observed_runtime) != _jsonable(runtime_manifest):
            raise ValueError(f"{label} attempt changed the sealed LAC runtime identity")


def validation_bootstrap_record_from_outcome(
    plan: ValidationBootstrapPlan,
    row_key: str,
    outcome: BootstrapOutcomeProvider,
) -> ValidationBootstrapRecord:
    """Authenticate one causal initializer/cache outcome against one census row."""

    if not isinstance(plan, ValidationBootstrapPlan):
        raise TypeError("validation bootstrap record requires a typed plan")
    row = _row_by_key(plan, row_key)
    parsed = _outcome_from_provider(outcome)
    identity = next(
        task for task in plan.tasks if task.instance_id == row.instance_id
    )
    initial = parsed.initial_snapshot
    if (
        parsed.episode_schedule_index != row.census_index
        or parsed.instance_id != row.instance_id
        or parsed.base_lineage != row.base_lineage
        or parsed.public_task_digest != identity.public_task_digest
        or parsed.plan_record_digest != plan.record_digest
        or parsed.config_digest != plan.config_digest
        or parsed.context_digest != plan.context_digest
        or initial.conditional_episode_index != row.census_index
        or initial.instance_id != row.instance_id
        or initial.base_lineage != row.base_lineage
        or initial.slot_kind != "initial"
        or initial.slot_index != 0
        or initial.draw_index != 0
        or initial.system_seed != row.system_seed
        or initial.plan_record_digest != plan.record_digest
        or initial.draw_key
        != _bootstrap_draw(plan, row, slot_kind="initial", slot_index=0).draw_key
    ):
        raise ValueError("bootstrap outcome differs from its validation census identity")
    if parsed.restart_cache_slot_count != plan.restart_cache_slots:
        raise ValueError("bootstrap outcome changed the registered cache-slot count")
    _validate_attempt_tape(
        initial,
        row.initial_attempt_seeds,
        label="initial bootstrap",
        runtime_manifest=plan.runtime_implementation_manifest,
    )
    if initial.execution_status not in {"SUCCESS", "FAILED"}:
        raise ValueError("validation census requires one exact invoked LAC initial outcome")
    if initial.work_before != WorkVector() or initial.time_before_seconds != 0.0:
        raise ValueError("initial validation bootstrap did not start at zero work and time")

    cache = parsed.restart_cache_snapshots
    if initial.success:
        if len(cache) != plan.restart_cache_slots:
            raise ValueError("successful initial bootstrap requires exactly two cache slots")
    elif cache:
        raise ValueError("failed initial bootstrap must not invoke or populate cache slots")
    for slot, snapshot in enumerate(cache):
        if (
            snapshot.conditional_episode_index != row.census_index
            or snapshot.instance_id != row.instance_id
            or snapshot.base_lineage != row.base_lineage
            or snapshot.slot_kind != "restart_cache"
            or snapshot.slot_index != slot
            or snapshot.draw_index != slot
            or snapshot.system_seed != row.cache_system_seeds[slot]
            or snapshot.draw_key
            != _bootstrap_draw(
                plan, row, slot_kind="restart_cache", slot_index=slot
            ).draw_key
        ):
            raise ValueError("restart-cache slot seed schedule differs from the census")
        _validate_attempt_tape(
            snapshot,
            row.cache_attempt_seeds[slot],
            label=f"restart-cache slot {slot}",
            runtime_manifest=plan.runtime_implementation_manifest,
        )
        if snapshot.plan_record_digest != initial.plan_record_digest:
            raise ValueError("bootstrap/cache snapshots cross source execution plans")

    total_work = parsed.total_pre_policy_debit
    if not total_work.fits_in(_work(plan.context["caps"], label="validation context cap")):
        raise ValueError("bootstrap/cache generation exceeds the registered work cap")
    if initial.success and not (
        total_work + _POLICY_ENTRY_RESERVE + _work(
            plan.context["reserve"], label="validation context reserve"
        )
    ).fits_in(_work(plan.context["caps"], label="validation context cap")):
        raise ValueError("bootstrap/cache generation leaves insufficient policy terminal reserve")
    elapsed = parsed.precomputed_online_seconds
    wallclock_cap = plan.complete_system_config["online_wallclock_seconds"]
    if (
        isinstance(elapsed, bool)
        or not math.isfinite(elapsed)
        or elapsed < 0.0
        or not isinstance(wallclock_cap, (int, float))
        or elapsed > float(wallclock_cap) + max(1e-9, float(wallclock_cap) * 1e-9)
    ):
        raise ValueError("bootstrap/cache generation exceeds the cumulative wall-clock cap")

    payload = parsed.as_dict()
    if parsed.manifest_record_digest != bootstrap_execution_record_digest(
        plan, row.row_key, initial, cache
    ):
        raise ValueError("bootstrap execution transcript digest differs from its snapshots")
    _assert_target_free(payload)
    payload_bytes = _canonical_bytes(payload)
    failed = not initial.success
    return ValidationBootstrapRecord(
        plan_record_digest=plan.record_digest,
        row_key=row.row_key,
        census_index=row.census_index,
        base_lineage=row.base_lineage,
        instance_id=row.instance_id,
        repetition=row.repetition,
        evaluation_seed=plan.evaluation_seed,
        bootstrap_outcome_payload=MappingProxyType(payload),
        bootstrap_outcome_record_digest=parsed.record_digest,
        bootstrap_payload_sha256=_file_sha256(payload_bytes),
        source_execution_plan_digest=parsed.plan_record_digest,
        source_execution_manifest_digest=parsed.manifest_record_digest,
        initial_status=initial.execution_status,
        cache_slot_count=len(cache),
        cache_success_count=sum(snapshot.success for snapshot in cache),
        total_pre_policy_work=total_work,
        precomputed_online_seconds=elapsed,
        denominator_eligible=True,
        fixed_utility=0.0 if failed else None,
        actor_invocation_permitted=not failed,
        cache_invocation_permitted=not failed,
    )


def _record_from_payload(
    plan: ValidationBootstrapPlan, payload: Mapping[str, object]
) -> ValidationBootstrapRecord:
    expected = {
        "schema",
        "schema_version",
        "plan_record_digest",
        "row_key",
        "census_index",
        "base_lineage",
        "instance_id",
        "repetition",
        "evaluation_seed",
        "bootstrap_outcome",
        "bootstrap_outcome_record_digest",
        "bootstrap_payload_sha256",
        "source_execution_plan_digest",
        "source_execution_manifest_digest",
        "initial_status",
        "cache_slot_count",
        "cache_success_count",
        "total_pre_policy_work",
        "precomputed_online_seconds",
        "denominator_eligible",
        "fixed_utility",
        "actor_invocation_permitted",
        "cache_invocation_permitted",
        "opened_evaluator_targets",
        "record_digest",
    }
    if set(payload) != expected:
        raise ValueError("validation-bootstrap record has missing or unknown fields")
    if (
        payload["schema"] != VALIDATION_BOOTSTRAP_RECORD_SCHEMA
        or payload["schema_version"] != VALIDATION_BOOTSTRAP_RECORD_VERSION
        or payload["opened_evaluator_targets"] is not False
    ):
        raise ValueError("unsupported or target-bearing validation-bootstrap record")
    if type(payload["schema_version"]) is not int:
        raise ValueError("validation-bootstrap record schema version must be an integer")
    for name in (
        "census_index",
        "repetition",
        "evaluation_seed",
        "cache_slot_count",
        "cache_success_count",
    ):
        _exact_nonnegative_int(payload[name], label=f"validation-bootstrap record {name}")
    for name in (
        "denominator_eligible",
        "actor_invocation_permitted",
        "cache_invocation_permitted",
        "opened_evaluator_targets",
    ):
        if type(payload[name]) is not bool:
            raise ValueError(f"validation-bootstrap record {name} must be boolean")
    for name in (
        "plan_record_digest",
        "row_key",
        "bootstrap_outcome_record_digest",
        "bootstrap_payload_sha256",
        "source_execution_plan_digest",
        "source_execution_manifest_digest",
        "record_digest",
    ):
        _digest(payload[name], f"validation-bootstrap record {name}")
    for name in ("base_lineage", "instance_id", "initial_status"):
        _nonempty_text(payload[name], label=f"validation-bootstrap record {name}")
    _work(payload["total_pre_policy_work"], label="record total pre-policy work")
    _exact_finite_float(
        payload["precomputed_online_seconds"],
        label="record precomputed online seconds",
    )
    fixed_utility = payload["fixed_utility"]
    if fixed_utility is not None and type(fixed_utility) is not float:
        raise ValueError("validation-bootstrap fixed utility must be null or a JSON float")
    _verify_record(payload, label="validation-bootstrap record")
    bootstrap = payload["bootstrap_outcome"]
    if not isinstance(bootstrap, Mapping):
        raise ValueError("validation-bootstrap record outcome must be an object")
    rebuilt = validation_bootstrap_record_from_outcome(
        plan,
        str(payload["row_key"]),
        episode_bootstrap_outcome_from_payload(bootstrap),
    )
    if rebuilt.as_dict() != dict(payload):
        raise ValueError("validation-bootstrap record is not canonical")
    return rebuilt


def _record_filename(record: ValidationBootstrapRecord) -> str:
    return (
        f"{record.census_index:012d}-{record.row_key}-"
        f"{record.record_digest}.json"
    )


def publish_validation_bootstrap_record(
    directory: str | os.PathLike[str],
    plan: ValidationBootstrapPlan,
    record: ValidationBootstrapRecord,
) -> str:
    """Publish one content-addressed census row without replacement."""

    if not isinstance(plan, ValidationBootstrapPlan) or not isinstance(
        record, ValidationBootstrapRecord
    ):
        raise TypeError("validation-bootstrap publication requires typed plan and record")
    root = Path(directory)
    _validate_bank_root(root, sealed=False)
    plan_content = _canonical_bytes(plan.as_dict())
    plan_path = root / "plan.json"
    if plan_path.is_symlink() or not plan_path.is_file() or plan_path.read_bytes() != plan_content:
        raise ValueError("validation-bootstrap plan artifact is missing or differs")
    if (root / "manifest.json").exists():
        raise ValueError("sealed validation-bootstrap bank is immutable")
    _record_from_payload(plan, record.as_dict())
    content = _canonical_bytes(record.as_dict())
    _write_new(root / "records" / _record_filename(record), content)
    return _file_sha256(content)


def _read_record_files(
    directory: Path, plan: ValidationBootstrapPlan
) -> tuple[tuple[Path, bytes, ValidationBootstrapRecord], ...]:
    records_root = directory / "records"
    if not records_root.is_dir() or records_root.is_symlink():
        raise ValueError("validation-bootstrap record directory is missing or unsafe")
    rows: list[tuple[Path, bytes, ValidationBootstrapRecord]] = []
    seen: set[str] = set()
    for path in sorted(records_root.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file() or path.suffix != ".json":
            raise ValueError("validation-bootstrap record directory contains an unknown artifact")
        content = path.read_bytes()
        payload = _strict_json(content, label=f"validation-bootstrap record {path.name}")
        record = _record_from_payload(plan, payload)
        if content != _canonical_bytes(dict(payload)):
            raise ValueError("validation-bootstrap record is not canonical JSON")
        if path.name != _record_filename(record):
            raise ValueError("validation-bootstrap record filename is not content addressed")
        if record.row_key in seen:
            raise ValueError("validation-bootstrap bank repeats a census row")
        seen.add(record.row_key)
        rows.append((path, content, record))
    return tuple(rows)


def _sum_work(values: Sequence[WorkVector]) -> WorkVector:
    total = WorkVector()
    for value in values:
        total = total + value
    return total


def _manifest_payload(
    plan: ValidationBootstrapPlan,
    rows: Sequence[tuple[Path, bytes, ValidationBootstrapRecord]],
    *,
    plan_sha256: str,
) -> dict[str, object]:
    expected = {row.row_key for row in plan.census}
    observed = {record.row_key for _, _, record in rows}
    if observed != expected or len(rows) != len(plan.census):
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(
            "validation-bootstrap bank differs from the complete census: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    ordered = sorted(rows, key=lambda item: item[2].census_index)
    if [record.census_index for _, _, record in ordered] != list(range(len(plan.census))):
        raise ValueError("validation-bootstrap census indices are not complete and contiguous")
    files = [
        {
            "census_index": record.census_index,
            "row_key": record.row_key,
            "relative_path": f"records/{path.name}",
            "record_digest": record.record_digest,
            "bootstrap_outcome_record_digest": record.bootstrap_outcome_record_digest,
            "bootstrap_payload_sha256": record.bootstrap_payload_sha256,
            "sha256": _file_sha256(content),
            "initial_status": record.initial_status,
            "cache_slot_count": record.cache_slot_count,
        }
        for path, content, record in ordered
    ]
    records = [record for _, _, record in ordered]
    body = {
        "schema": VALIDATION_BOOTSTRAP_MANIFEST_SCHEMA,
        "schema_version": VALIDATION_BOOTSTRAP_MANIFEST_VERSION,
        "plan_record_digest": plan.record_digest,
        "plan_sha256": plan_sha256,
        "prepared_manifest_sha256": plan.prepared_manifest_sha256,
        "protocol_registry_sha256": plan.protocol_registry_sha256,
        "protocol_record_digest": plan.protocol_record_digest,
        "same_support_contract_digest": plan.same_support_contract_digest,
        "protocol_preset": plan.protocol_preset,
        "partition": plan.partition,
        "evaluation_seed": plan.evaluation_seed,
        "repetitions": plan.repetitions,
        "restart_cache_slots": plan.restart_cache_slots,
        "census_digest": plan.census_digest,
        "denominator_count": len(records),
        "denominator_eligible_count": sum(row.denominator_eligible for row in records),
        "initial_success_count": sum(row.actor_invocation_permitted for row in records),
        "initial_failure_count": sum(not row.actor_invocation_permitted for row in records),
        "actor_forbidden_count": sum(not row.actor_invocation_permitted for row in records),
        "cache_forbidden_count": sum(not row.cache_invocation_permitted for row in records),
        "fixed_zero_utility_count": sum(row.fixed_utility == 0.0 for row in records),
        "total_generation_work": _sum_work(
            [row.total_pre_policy_work for row in records]
        ).as_dict(),
        "total_precomputed_online_seconds": sum(
            row.precomputed_online_seconds for row in records
        ),
        "record_files": files,
        "record_root_digest": stable_digest(files),
        "source_execution_plan_root": stable_digest(
            [row.source_execution_plan_digest for row in records]
        ),
        "source_execution_manifest_root": stable_digest(
            [row.source_execution_manifest_digest for row in records]
        ),
        "opened_evaluator_targets": False,
        "publication_eligible": plan.publication_eligible,
    }
    return {**body, "record_digest": stable_digest(body)}


def _validate_manifest_types(manifest: Mapping[str, object]) -> None:
    if type(manifest["schema_version"]) is not int:
        raise ValueError("validation-bootstrap manifest schema version must be an integer")
    integer_fields = (
        "evaluation_seed",
        "repetitions",
        "restart_cache_slots",
        "denominator_count",
        "denominator_eligible_count",
        "initial_success_count",
        "initial_failure_count",
        "actor_forbidden_count",
        "cache_forbidden_count",
        "fixed_zero_utility_count",
    )
    if any(type(manifest[name]) is not int or manifest[name] < 0 for name in integer_fields):
        raise ValueError("validation-bootstrap manifest integer counts are malformed")
    for name in ("opened_evaluator_targets", "publication_eligible"):
        if type(manifest[name]) is not bool:
            raise ValueError(f"validation-bootstrap manifest {name} must be boolean")
    for name in (
        "plan_record_digest",
        "plan_sha256",
        "prepared_manifest_sha256",
        "protocol_registry_sha256",
        "protocol_record_digest",
        "same_support_contract_digest",
        "census_digest",
        "record_root_digest",
        "source_execution_plan_root",
        "source_execution_manifest_root",
        "record_digest",
    ):
        _digest(manifest[name], f"validation-bootstrap manifest {name}")
    _nonempty_text(
        manifest["protocol_preset"], label="bootstrap manifest protocol preset"
    )
    _nonempty_text(manifest["partition"], label="bootstrap manifest partition")
    _work(manifest["total_generation_work"], label="manifest total generation")
    _exact_finite_float(
        manifest["total_precomputed_online_seconds"],
        label="manifest total precomputed online seconds",
    )
    files = manifest["record_files"]
    if not isinstance(files, list):
        raise ValueError("validation-bootstrap manifest record files must be a JSON list")
    file_fields = {
        "census_index",
        "row_key",
        "relative_path",
        "record_digest",
        "bootstrap_outcome_record_digest",
        "bootstrap_payload_sha256",
        "sha256",
        "initial_status",
        "cache_slot_count",
    }
    for item in files:
        if not isinstance(item, Mapping) or set(item) != file_fields:
            raise ValueError("validation-bootstrap manifest record-file row is malformed")
        _exact_nonnegative_int(item["census_index"], label="record-file census index")
        _exact_nonnegative_int(item["cache_slot_count"], label="record-file cache slots")
        for name in (
            "row_key",
            "record_digest",
            "bootstrap_outcome_record_digest",
            "bootstrap_payload_sha256",
            "sha256",
        ):
            _digest(item[name], f"record-file {name}")
        relative = _nonempty_text(item["relative_path"], label="record-file path")
        if not relative.startswith("records/") or Path(relative).is_absolute():
            raise ValueError("validation-bootstrap manifest record-file path is unsafe")
        if item["initial_status"] not in {"SUCCESS", "FAILED"}:
            raise ValueError("validation-bootstrap manifest initial status is invalid")


def seal_validation_bootstrap_bank(
    directory: str | os.PathLike[str], plan: ValidationBootstrapPlan
) -> str:
    """Seal only a complete, target-free validation census."""

    if not isinstance(plan, ValidationBootstrapPlan):
        raise TypeError("validation-bootstrap sealing requires a typed plan")
    root = Path(directory)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"immutable validation-bootstrap artifact exists: {manifest_path}")
    _validate_bank_root(root, sealed=False)
    plan_content = _canonical_bytes(plan.as_dict())
    plan_path = root / "plan.json"
    if plan_path.is_symlink() or not plan_path.is_file() or plan_path.read_bytes() != plan_content:
        raise ValueError("validation-bootstrap plan artifact differs from the plan")
    rows = _read_record_files(root, plan)
    manifest = _manifest_payload(
        plan, rows, plan_sha256=_file_sha256(plan_content)
    )
    content = _canonical_bytes(manifest)
    _write_new(manifest_path, content)
    return _file_sha256(content)


@dataclass(frozen=True)
class ValidationBootstrapAccessReceipt:
    manifest_sha256: str
    manifest_record_digest: str
    plan_sha256: str
    plan_record_digest: str
    prepared_manifest_sha256: str
    protocol_registry_sha256: str
    protocol_record_digest: str
    same_support_contract_digest: str
    protocol_preset: str
    census_digest: str
    record_root_digest: str
    source_execution_plan_root: str
    source_execution_manifest_root: str
    denominator_count: int
    initial_success_count: int
    initial_failure_count: int
    total_generation_work: WorkVector
    total_precomputed_online_seconds: float

    def as_dict(self) -> dict[str, object]:
        body = {
            "schema": VALIDATION_BOOTSTRAP_ACCESS_SCHEMA,
            "schema_version": VALIDATION_BOOTSTRAP_ACCESS_VERSION,
            "manifest_sha256": self.manifest_sha256,
            "manifest_record_digest": self.manifest_record_digest,
            "plan_sha256": self.plan_sha256,
            "plan_record_digest": self.plan_record_digest,
            "prepared_manifest_sha256": self.prepared_manifest_sha256,
            "protocol_registry_sha256": self.protocol_registry_sha256,
            "protocol_record_digest": self.protocol_record_digest,
            "same_support_contract_digest": self.same_support_contract_digest,
            "protocol_preset": self.protocol_preset,
            "census_digest": self.census_digest,
            "record_root_digest": self.record_root_digest,
            "source_execution_plan_root": self.source_execution_plan_root,
            "source_execution_manifest_root": self.source_execution_manifest_root,
            "denominator_count": self.denominator_count,
            "initial_success_count": self.initial_success_count,
            "initial_failure_count": self.initial_failure_count,
            "total_generation_work": self.total_generation_work.as_dict(),
            "total_precomputed_online_seconds": self.total_precomputed_online_seconds,
            "opened_evaluator_targets": False,
        }
        return {**body, "record_digest": stable_digest(body)}

    @property
    def record_digest(self) -> str:
        return str(self.as_dict()["record_digest"])


@dataclass(frozen=True)
class ValidationBootstrapClone:
    consumer_id: str
    same_support_contract_digest: str
    bank_access_record_digest: str
    bootstrap_record: ValidationBootstrapRecord

    @property
    def bootstrap_record_digest(self) -> str:
        return self.bootstrap_record.record_digest

    @property
    def bootstrap_payload_sha256(self) -> str:
        return self.bootstrap_record.bootstrap_payload_sha256

    @property
    def bootstrap_payload_bytes(self) -> bytes:
        return self.bootstrap_record.bootstrap_payload_bytes

    def as_dict(self) -> dict[str, object]:
        body = {
            "schema": VALIDATION_BOOTSTRAP_CLONE_SCHEMA,
            "schema_version": VALIDATION_BOOTSTRAP_CLONE_VERSION,
            "consumer_id": self.consumer_id,
            "same_support_contract_digest": self.same_support_contract_digest,
            "bank_access_record_digest": self.bank_access_record_digest,
            "bootstrap_record_digest": self.bootstrap_record_digest,
            "bootstrap_outcome_record_digest": (
                self.bootstrap_record.bootstrap_outcome_record_digest
            ),
            "bootstrap_payload_sha256": self.bootstrap_payload_sha256,
            "row_key": self.bootstrap_record.row_key,
            "denominator_eligible": self.bootstrap_record.denominator_eligible,
            "fixed_utility": self.bootstrap_record.fixed_utility,
            "actor_invocation_permitted": (
                self.bootstrap_record.actor_invocation_permitted
            ),
            "cache_invocation_permitted": (
                self.bootstrap_record.cache_invocation_permitted
            ),
            "opened_evaluator_targets": False,
        }
        return {**body, "record_digest": stable_digest(body)}


class ValidationBootstrapBank:
    """Authenticated immutable rows with non-consuming, byte-identical clone access."""

    def __init__(
        self,
        *,
        plan: ValidationBootstrapPlan,
        records: Mapping[str, ValidationBootstrapRecord],
        access_receipt: ValidationBootstrapAccessReceipt,
    ) -> None:
        self.plan = plan
        self._records = MappingProxyType(dict(records))
        self.access_receipt = access_receipt

    def __len__(self) -> int:
        return len(self._records)

    def row_key(self, base_lineage: str, instance_id: str, repetition: int) -> str:
        matches = [
            row.row_key
            for row in self.plan.census
            if (
                row.base_lineage == base_lineage
                and row.instance_id == instance_id
                and row.repetition == repetition
            )
        ]
        if len(matches) != 1:
            raise KeyError("validation-bootstrap coordinates are outside the sealed census")
        return matches[0]

    def record(self, row_key: str) -> ValidationBootstrapRecord:
        try:
            return self._records[row_key]
        except KeyError as exc:
            raise KeyError("validation-bootstrap row is outside the authenticated bank") from exc

    def clone_for_consumer(
        self,
        row_key: str,
        *,
        consumer_id: str,
        expected_same_support_contract_digest: str,
    ) -> ValidationBootstrapClone:
        if not isinstance(consumer_id, str) or not consumer_id:
            raise ValueError("validation-bootstrap consumer ID must be nonempty")
        expected = _digest(
            expected_same_support_contract_digest, "expected same-support contract"
        )
        if not hmac.compare_digest(expected, self.plan.same_support_contract_digest):
            raise ValueError("validation-bootstrap consumer changed the same-support contract")
        return ValidationBootstrapClone(
            consumer_id=consumer_id,
            same_support_contract_digest=expected,
            bank_access_record_digest=self.access_receipt.record_digest,
            bootstrap_record=self.record(row_key),
        )


def load_validation_bootstrap_bank(
    directory: str | os.PathLike[str],
    *,
    plan: ValidationBootstrapPlan,
    expected_plan_sha256: str,
    expected_manifest_sha256: str,
) -> ValidationBootstrapBank:
    """Authenticate external plan/manifest pins before returning any bootstrap row."""

    if not isinstance(plan, ValidationBootstrapPlan):
        raise TypeError("validation-bootstrap loading requires an authenticated typed plan")
    expected_plan = _digest(expected_plan_sha256, "expected validation-bootstrap plan")
    expected_manifest = _digest(
        expected_manifest_sha256, "expected validation-bootstrap manifest"
    )
    root = Path(directory)
    _validate_bank_root(root, sealed=True)
    plan_path = root / "plan.json"
    manifest_path = root / "manifest.json"
    if any(path.is_symlink() or not path.is_file() for path in (plan_path, manifest_path)):
        raise ValueError("validation-bootstrap bank is missing or unsafe")
    plan_content = plan_path.read_bytes()
    if (
        not hmac.compare_digest(_file_sha256(plan_content), expected_plan)
        or plan_content != _canonical_bytes(plan.as_dict())
    ):
        raise ValueError("validation-bootstrap plan external pin or content differs")
    manifest_content = manifest_path.read_bytes()
    if not hmac.compare_digest(_file_sha256(manifest_content), expected_manifest):
        raise ValueError("validation-bootstrap manifest SHA-256 pin mismatch")
    manifest = _strict_json(manifest_content, label="validation-bootstrap manifest")
    expected_keys = {
        "schema",
        "schema_version",
        "plan_record_digest",
        "plan_sha256",
        "prepared_manifest_sha256",
        "protocol_registry_sha256",
        "protocol_record_digest",
        "same_support_contract_digest",
        "protocol_preset",
        "partition",
        "evaluation_seed",
        "repetitions",
        "restart_cache_slots",
        "census_digest",
        "denominator_count",
        "denominator_eligible_count",
        "initial_success_count",
        "initial_failure_count",
        "actor_forbidden_count",
        "cache_forbidden_count",
        "fixed_zero_utility_count",
        "total_generation_work",
        "total_precomputed_online_seconds",
        "record_files",
        "record_root_digest",
        "source_execution_plan_root",
        "source_execution_manifest_root",
        "opened_evaluator_targets",
        "publication_eligible",
        "record_digest",
    }
    if set(manifest) != expected_keys:
        raise ValueError("validation-bootstrap manifest has missing or unknown fields")
    if (
        manifest["schema"] != VALIDATION_BOOTSTRAP_MANIFEST_SCHEMA
        or manifest["schema_version"] != VALIDATION_BOOTSTRAP_MANIFEST_VERSION
        or manifest["opened_evaluator_targets"] is not False
    ):
        raise ValueError("unsupported or target-bearing validation-bootstrap manifest")
    _validate_manifest_types(manifest)
    manifest_record_digest = _verify_record(
        manifest, label="validation-bootstrap manifest"
    )
    rows = _read_record_files(root, plan)
    rebuilt = _manifest_payload(plan, rows, plan_sha256=expected_plan)
    if rebuilt != manifest:
        raise ValueError("validation-bootstrap manifest differs from authenticated records")
    if manifest_content != _canonical_bytes(rebuilt):
        raise ValueError("validation-bootstrap manifest is not canonical JSON")
    records = {record.row_key: record for _, _, record in rows}
    access = ValidationBootstrapAccessReceipt(
        manifest_sha256=expected_manifest,
        manifest_record_digest=manifest_record_digest,
        plan_sha256=expected_plan,
        plan_record_digest=plan.record_digest,
        prepared_manifest_sha256=plan.prepared_manifest_sha256,
        protocol_registry_sha256=plan.protocol_registry_sha256,
        protocol_record_digest=plan.protocol_record_digest,
        same_support_contract_digest=plan.same_support_contract_digest,
        protocol_preset=plan.protocol_preset,
        census_digest=plan.census_digest,
        record_root_digest=str(manifest["record_root_digest"]),
        source_execution_plan_root=str(manifest["source_execution_plan_root"]),
        source_execution_manifest_root=str(manifest["source_execution_manifest_root"]),
        denominator_count=int(manifest["denominator_count"]),
        initial_success_count=int(manifest["initial_success_count"]),
        initial_failure_count=int(manifest["initial_failure_count"]),
        total_generation_work=_work(
            manifest["total_generation_work"], label="manifest total generation"
        ),
        total_precomputed_online_seconds=float(
            manifest["total_precomputed_online_seconds"]
        ),
    )
    return ValidationBootstrapBank(plan=plan, records=records, access_receipt=access)


# Generic public names. The validation-prefixed names remain compatibility aliases while
# both registered populations use the same implementation and authenticated schemas.
BootstrapAccessReceipt = ValidationBootstrapAccessReceipt
BootstrapBank = ValidationBootstrapBank
BootstrapCensusRow = ValidationBootstrapCensusRow
BootstrapClone = ValidationBootstrapClone
BootstrapPlan = ValidationBootstrapPlan
BootstrapRecord = ValidationBootstrapRecord
bootstrap_record_from_outcome = validation_bootstrap_record_from_outcome
load_bootstrap_bank = load_validation_bootstrap_bank
publish_bootstrap_record = publish_validation_bootstrap_record
seal_bootstrap_bank = seal_validation_bootstrap_bank
write_bootstrap_plan = write_validation_bootstrap_plan
build_validation_bootstrap_plan = build_bootstrap_plan
load_validation_bootstrap_plan = load_bootstrap_plan


__all__ = [
    "BootstrapAccessReceipt",
    "BootstrapBank",
    "BootstrapCensusRow",
    "BootstrapClone",
    "BootstrapOutcomeProvider",
    "BootstrapPlan",
    "BootstrapProtocolPreset",
    "BootstrapRecord",
    "FINAL_TEST_BOOTSTRAP_PARTITION",
    "FINAL_TEST_BOOTSTRAP_PRESET",
    "REGISTERED_BOOTSTRAP_PROTOCOLS",
    "RL_VALUE_EVALUATION_SEED",
    "REPRESENTATION_EVALUATION_SEED",
    "REPRESENTATION_VALIDATION_BOOTSTRAP_PRESET",
    "RL_VALUE_FINAL_TEST_SEED",
    "RL_VALUE_REGISTERED_REPETITIONS",
    "VALIDATION_RESTART_CACHE_SLOTS",
    "ValidationBootstrapAccessReceipt",
    "ValidationBootstrapBank",
    "ValidationBootstrapCensusRow",
    "ValidationBootstrapClone",
    "ValidationBootstrapPlan",
    "ValidationBootstrapRecord",
    "bootstrap_context_digest",
    "bootstrap_execution_record_digest",
    "bootstrap_record_from_outcome",
    "bootstrap_same_support_contract",
    "bootstrap_same_support_contract_digest",
    "build_bootstrap_plan",
    "build_validation_bootstrap_plan",
    "execute_bootstrap_row",
    "load_bootstrap_bank",
    "load_bootstrap_plan",
    "load_validation_bootstrap_bank",
    "load_validation_bootstrap_plan",
    "publish_bootstrap_record",
    "publish_validation_bootstrap_record",
    "registered_bootstrap_protocol",
    "seal_bootstrap_bank",
    "seal_validation_bootstrap_bank",
    "validation_bootstrap_record_from_outcome",
    "write_bootstrap_plan",
    "write_validation_bootstrap_plan",
]
