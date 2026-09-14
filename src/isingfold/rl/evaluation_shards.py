"""Authenticated, lineage-coherent sharding for long evaluation workflows.

The execution plan commits to the complete pre-initialization population before a shard is
run.  Shards select whole immutable base lineages, while every attempt keeps the same
identity-derived random seed it would receive in an unsharded run.  A shard directory is an
atomic publication containing raw receipts, outcome projections, terminal evidence and a
small receipt that is authenticated out of band.  Merge never trusts a reported summary: it
rechecks the complete census and recomputes the workflow's unsharded sufficient statistics.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import TypeAlias

from isingfold.rl.complete_system import (
    COMPLETE_SYSTEM_OUTCOME_SCHEMA,
    COMPLETE_SYSTEM_OUTCOME_VERSION,
    CompletePopulationIdentity,
    CompleteSystemEvidenceRecord,
    CompleteSystemReceipt,
    _evidence_record_from_payload,
    _receipt_from_payload,
    complete_system_metrics,
    task_population_digest,
    verify_terminal_evidence,
)
from isingfold.rl.contracts import WORK_FIELDS, Context, WorkVector, stable_digest
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.env import EmbeddingTask
from isingfold.rl.evaluate import EpisodeOutcome, EvaluationProtocolError
from isingfold.rl.external_pairing import (
    ExternalCompleteSystemEvidenceRecord,
    ExternalCompleteSystemReceipt,
    _backend_from_payload,
    _component_from_payload,
    _population_from_payload,
    external_complete_summary,
    read_external_complete_outcomes,
    read_external_complete_receipts,
)
from isingfold.rl.external_tuning import (
    ExternalTuningExecutionBinding,
    ExternalTuningRun,
    external_tuning_lineage_metrics,
)

EVALUATION_PLAN_SCHEMA = "isingfold.resumable-evaluation-plan"
EVALUATION_PLAN_VERSION = 2
EVALUATION_SHARD_RECEIPT_SCHEMA = "isingfold.resumable-evaluation-shard-receipt"
EVALUATION_SHARD_RECEIPT_VERSION = 3
EVALUATION_MERGE_RECEIPT_SCHEMA = "isingfold.resumable-evaluation-merge-receipt"
EVALUATION_MERGE_RECEIPT_VERSION = 2
DEFAULT_MAX_LINEAGES_PER_SHARD = 32
SEED_ALGORITHM = "sha256-canonical-identity-first32-mod-2^31-v1"
REGISTERED_SEEDS = (1103, 2207, 3301)

PairKey: TypeAlias = tuple[str, str, int]
TypedReceipt: TypeAlias = CompleteSystemReceipt | ExternalCompleteSystemReceipt
TypedEvidence: TypeAlias = CompleteSystemEvidenceRecord | ExternalCompleteSystemEvidenceRecord


class EvaluationWorkflow(str, Enum):
    """The three publication workflows that share one sharding contract."""

    LEARNED_COMPLETE = "learned-complete-system"
    TUNED_STOCK_COMPLETE = "tuned-stock-complete-system"
    VALIDATION_TUNING = "external-validation-tuning"


def _is_digest(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _strict_json(content: bytes, *, label: str) -> dict[str, object]:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> object:
        raise ValueError(f"nonfinite JSON scalar {value!r}")

    try:
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvaluationProtocolError(f"{label} is not strict finite UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise EvaluationProtocolError(f"{label} must be a JSON object")
    if canonical_json_bytes(value) + b"\n" != content:
        raise EvaluationProtocolError(f"{label} is not canonical JSON")
    return value


def _strict_jsonl(content: bytes, *, label: str) -> tuple[Mapping[str, object], ...]:
    def no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> object:
        raise ValueError(f"nonfinite JSON scalar {value!r}")

    if not content:
        raise EvaluationProtocolError(f"{label} is empty")
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise EvaluationProtocolError(f"{label} is not UTF-8") from exc
    if not lines or any(not line for line in lines):
        raise EvaluationProtocolError(f"{label} is empty or contains blank rows")
    rows: list[Mapping[str, object]] = []
    for number, line in enumerate(lines, start=1):
        try:
            row = json.loads(
                line,
                object_pairs_hook=no_duplicates,
                parse_constant=reject_nonfinite,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise EvaluationProtocolError(
                f"{label} row {number} is not strict finite JSON"
            ) from exc
        if not isinstance(row, dict) or _raw_json_bytes(row) != line.encode():
            raise EvaluationProtocolError(f"{label} row {number} is not canonical JSON")
        rows.append(MappingProxyType(row))
    return tuple(rows)


def _plain(value: object) -> object:
    return json.loads(canonical_json_bytes(value))


def _stable(value: object) -> str:
    """Apply the repository digest to a plain canonical copy of proxy-backed JSON."""

    return stable_digest(_plain(value))


def _frozen_mapping(value: Mapping[str, object], *, label: str) -> Mapping[str, object]:
    plain = _plain(value)
    if not isinstance(plain, dict) or not plain:
        raise ValueError(f"{label} must be a nonempty canonical JSON object")
    return MappingProxyType(plain)


def _canonical_lines(rows: Sequence[Mapping[str, object]]) -> bytes:
    return b"".join(_raw_json_bytes(row) + b"\n" for row in rows)


def _raw_json_bytes(value: object) -> bytes:
    """Match the established raw JSONL writers, including signed zero and ASCII escapes."""

    serializable = dict(value) if isinstance(value, Mapping) else value
    return json.dumps(
        serializable,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def identity_seed(
    base_seed: int,
    domain: str,
    *,
    lineage: str,
    instance: str,
    repetition: int,
) -> int:
    """Return the registered attempt seed, independent of shard count and order."""

    if (
        type(base_seed) is not int
        or base_seed < 0
        or not isinstance(domain, str)
        or not domain
        or not isinstance(lineage, str)
        or not lineage
        or not isinstance(instance, str)
        or not instance
        or type(repetition) is not int
        or repetition < 0
    ):
        raise ValueError("seed identity is malformed")
    payload = {
        "base_seed": base_seed,
        "domain": domain,
        "instance": instance,
        "lineage": lineage,
        "repetition": repetition,
    }
    return int.from_bytes(hashlib.sha256(canonical_json_bytes(payload)).digest()[:4], "big") % (
        2**31
    )


def _coordinate_contract(
    workflow: EvaluationWorkflow, coordinates: Mapping[str, object]
) -> tuple[Mapping[str, object], int, str, str]:
    plain = _plain(coordinates)
    if not isinstance(plain, dict):
        raise ValueError("evaluation run coordinates must be an object")
    expected: set[str]
    if workflow is EvaluationWorkflow.LEARNED_COMPLETE:
        expected = {"training_seed_index", "training_seed", "source_cell_id"}
    elif workflow is EvaluationWorkflow.TUNED_STOCK_COMPLETE:
        expected = {"training_seed_index", "training_seed"}
    else:
        expected = {
            "candidate_index",
            "candidate_id",
            "tuning_seed_index",
            "tuning_seed",
        }
    if set(plain) != expected:
        raise ValueError(
            "evaluation run-coordinate fields differ: "
            f"missing={sorted(expected - set(plain))}, "
            f"unknown={sorted(set(plain) - expected)}"
        )
    if workflow is EvaluationWorkflow.VALIDATION_TUNING:
        index = plain["tuning_seed_index"]
        seed = plain["tuning_seed"]
        if (
            type(index) is not int
            or index not in range(len(REGISTERED_SEEDS))
            or seed != REGISTERED_SEEDS[index]
            or type(plain["candidate_index"]) is not int
            or plain["candidate_index"] < 0
            or not isinstance(plain["candidate_id"], str)
            or not plain["candidate_id"]
        ):
            raise ValueError("validation-tuning coordinates are invalid")
        base_seed = int(seed)
        system_domain = "external-tuning-policy"
        evaluator_domain = "external-tuning-final-evaluator"
    else:
        index = plain["training_seed_index"]
        seed = plain["training_seed"]
        if (
            type(index) is not int
            or index not in range(len(REGISTERED_SEEDS))
            or seed != REGISTERED_SEEDS[index]
        ):
            raise ValueError("complete-system coordinates use an unregistered training seed")
        if workflow is EvaluationWorkflow.LEARNED_COMPLETE and (
            not isinstance(plain["source_cell_id"], str) or not plain["source_cell_id"]
        ):
            raise ValueError("learned complete coordinates need a source cell")
        base_seed = -1
        system_domain = "policy"
        evaluator_domain = "final-evaluator"
    return MappingProxyType(plain), base_seed, system_domain, evaluator_domain


def learned_execution_contract(
    *,
    initializer: Mapping[str, object],
    controller: Mapping[str, object],
    selector: Mapping[str, object],
    config_digest: str,
    context_digest: str,
    policy_context_digest: str,
    work_cap: Mapping[str, object],
    wallclock_cap_seconds: float,
) -> Mapping[str, object]:
    """Build the pre-execution authority later matched against every learned receipt."""

    payload = {
        "contract_kind": "learned-complete-system-v1",
        "initializer": initializer,
        "controller": controller,
        "selector": selector,
        "config_digest": config_digest,
        "context_digest": context_digest,
        "policy_context_digest": policy_context_digest,
        "work_cap": work_cap,
        "wallclock_cap_seconds": wallclock_cap_seconds,
    }
    return _validate_execution_contract(payload, EvaluationWorkflow.LEARNED_COMPLETE)


def external_execution_contract(
    *,
    backend: Mapping[str, object],
    selector: Mapping[str, object],
    config_digest: str,
    learned_config_digest: str,
    context_digest: str,
    quality_authority_digest: str,
    work_cap: Mapping[str, object],
    wallclock_cap_seconds: float,
    training_seed_index: int,
    training_seed: int,
    tuning_execution: Mapping[str, object],
) -> Mapping[str, object]:
    """Build the pre-execution authority for stock deployment or validation tuning."""

    payload = {
        "contract_kind": "external-complete-system-v2",
        "backend": backend,
        "selector": selector,
        "config_digest": config_digest,
        "learned_config_digest": learned_config_digest,
        "context_digest": context_digest,
        "quality_authority_digest": quality_authority_digest,
        "work_cap": work_cap,
        "wallclock_cap_seconds": wallclock_cap_seconds,
        "training_seed_index": training_seed_index,
        "training_seed": training_seed,
        "tuning_execution": tuning_execution,
    }
    return _validate_execution_contract(payload, EvaluationWorkflow.TUNED_STOCK_COMPLETE)


def _validate_execution_contract(
    value: Mapping[str, object], workflow: EvaluationWorkflow
) -> Mapping[str, object]:
    plain = _plain(value)
    if not isinstance(plain, dict):
        raise ValueError("execution contract must be an object")
    learned_fields = {
        "contract_kind",
        "initializer",
        "controller",
        "selector",
        "config_digest",
        "context_digest",
        "policy_context_digest",
        "work_cap",
        "wallclock_cap_seconds",
    }
    external_fields = {
        "contract_kind",
        "backend",
        "selector",
        "config_digest",
        "learned_config_digest",
        "context_digest",
        "quality_authority_digest",
        "work_cap",
        "wallclock_cap_seconds",
        "training_seed_index",
        "training_seed",
        "tuning_execution",
    }
    expected = (
        learned_fields if workflow is EvaluationWorkflow.LEARNED_COMPLETE else external_fields
    )
    kind = (
        "learned-complete-system-v1"
        if workflow is EvaluationWorkflow.LEARNED_COMPLETE
        else "external-complete-system-v2"
    )
    if set(plain) != expected or plain.get("contract_kind") != kind:
        raise ValueError("execution contract fields or kind differ from the workflow")
    digest_fields = {"config_digest", "context_digest"}
    if workflow is EvaluationWorkflow.LEARNED_COMPLETE:
        digest_fields.add("policy_context_digest")
    else:
        digest_fields.update({"learned_config_digest", "quality_authority_digest"})
    if any(not _is_digest(plain.get(name)) for name in digest_fields):
        raise ValueError("execution contract contains an invalid authority digest")
    work = plain.get("work_cap")
    if (
        not isinstance(work, dict)
        or set(work) != set(WORK_FIELDS)
        or any(type(work[name]) is not int or work[name] < 0 for name in WORK_FIELDS)
    ):
        raise ValueError("execution contract work cap is invalid")
    wallclock = plain.get("wallclock_cap_seconds")
    if isinstance(wallclock, bool) or not isinstance(wallclock, (int, float)) or wallclock <= 0:
        raise ValueError("execution contract wall-clock cap is invalid")
    return MappingProxyType(plain)


def receipt_execution_contract(receipt: TypedReceipt) -> Mapping[str, object]:
    """Project one typed raw receipt onto fields fixed before outcome observation."""

    if isinstance(receipt, CompleteSystemReceipt):
        return learned_execution_contract(
            initializer=receipt.initializer.as_dict(),
            controller=receipt.controller.as_dict(),
            selector=receipt.selector.as_dict(),
            config_digest=receipt.config_digest,
            context_digest=receipt.context_digest,
            policy_context_digest=receipt.policy_context_digest,
            work_cap=receipt.work_cap.as_dict(),
            wallclock_cap_seconds=receipt.wallclock_cap_seconds,
        )
    if isinstance(receipt, ExternalCompleteSystemReceipt):
        if receipt.tuning_execution is None:
            raise EvaluationProtocolError("sharded external evaluation requires frozen tuning")
        return external_execution_contract(
            backend=receipt.backend.as_dict(),
            selector=receipt.selector.as_dict(),
            config_digest=receipt.config_digest,
            learned_config_digest=receipt.learned_config_digest,
            context_digest=receipt.context_digest,
            quality_authority_digest=receipt.quality_authority_digest,
            work_cap=receipt.work_cap.as_dict(),
            wallclock_cap_seconds=receipt.wallclock_cap_seconds,
            training_seed_index=receipt.training_seed_index,
            training_seed=receipt.training_seed,
            tuning_execution=receipt.tuning_execution.as_dict(),
        )
    raise TypeError("unsupported evaluation receipt type")


def _validated_evaluation_quality_authority(
    value: Mapping[str, object] | None,
    *,
    expected_partition: str,
    expected_count: int,
    expected_instance_set_digest: str,
) -> Mapping[str, object] | None:
    """Validate the target-free v2 evaluation-authority projection."""

    if value is None:
        return None
    plain = _plain(value)
    outer_fields = {
        "schema",
        "schema_version",
        "global",
        "evaluation_partition",
        "record_digest",
    }
    if not isinstance(plain, dict) or set(plain) != outer_fields:
        raise EvaluationProtocolError("evaluation quality-authority fields differ")
    outer_body = {key: item for key, item in plain.items() if key != "record_digest"}
    if (
        plain["schema"] != "isingfold.quality-authority-binding"
        or plain["schema_version"] != 2
        or plain["record_digest"] != content_digest(outer_body)
    ):
        raise EvaluationProtocolError("evaluation quality-authority binding is invalid")
    global_authority = plain["global"]
    partition_authority = plain["evaluation_partition"]
    global_fields = {
        "ground_root",
        "publication_id",
        "publisher_attestation_record_digest",
        "publisher_id",
        "record_digest",
        "schema",
        "schema_version",
        "target_authority_record_digest",
    }
    partition_fields = {
        "evidence_manifest_record_digest",
        "evidence_manifest_sha256",
        "ground_partition",
        "name",
        "record_digest",
        "schema",
        "schema_version",
        "target_access_record_digest",
        "target_count",
        "target_set_digest",
    }
    if not isinstance(global_authority, dict) or set(global_authority) != global_fields:
        raise EvaluationProtocolError("global quality-authority fields differ")
    if not isinstance(partition_authority, dict) or set(partition_authority) != partition_fields:
        raise EvaluationProtocolError("partition quality-authority fields differ")
    ground_root = global_authority["ground_root"]
    ground_partition = partition_authority["ground_partition"]
    if not isinstance(ground_root, dict) or set(ground_root) != {
        "receipt_sha256",
        "record_digest",
        "verifier_identity_digest",
    }:
        raise EvaluationProtocolError("global quality-authority ground root differs")
    if not isinstance(ground_partition, dict) or set(ground_partition) != {
        "accepted_count",
        "instance_set_digest",
        "receipt_record_digest",
        "receipt_sha256",
    }:
        raise EvaluationProtocolError("partition quality-authority ground proof differs")
    global_body = {key: item for key, item in global_authority.items() if key != "record_digest"}
    partition_body = {
        key: item for key, item in partition_authority.items() if key != "record_digest"
    }
    digests = (
        plain["record_digest"],
        global_authority["publisher_attestation_record_digest"],
        global_authority["target_authority_record_digest"],
        global_authority["record_digest"],
        ground_root["receipt_sha256"],
        ground_root["record_digest"],
        ground_root["verifier_identity_digest"],
        partition_authority["evidence_manifest_record_digest"],
        partition_authority["evidence_manifest_sha256"],
        partition_authority["record_digest"],
        partition_authority["target_access_record_digest"],
        partition_authority["target_set_digest"],
        ground_partition["instance_set_digest"],
        ground_partition["receipt_record_digest"],
        ground_partition["receipt_sha256"],
    )
    if (
        any(not _is_digest(item) for item in digests)
        or global_authority["schema"] != "isingfold.global-quality-authority"
        or global_authority["schema_version"] != 1
        or global_authority["record_digest"] != content_digest(global_body)
        or not isinstance(global_authority["publication_id"], str)
        or not global_authority["publication_id"]
        or not isinstance(global_authority["publisher_id"], str)
        or not global_authority["publisher_id"]
        or partition_authority["schema"] != "isingfold.partition-quality-authority"
        or partition_authority["schema_version"] != 1
        or partition_authority["record_digest"] != content_digest(partition_body)
        or partition_authority["name"] != expected_partition
        or type(partition_authority["target_count"]) is not int
        or partition_authority["target_count"] != expected_count
        or type(ground_partition["accepted_count"]) is not int
        or ground_partition["accepted_count"] != expected_count
        or ground_partition["instance_set_digest"] != expected_instance_set_digest
    ):
        raise EvaluationProtocolError(
            "evaluation quality authority differs from the sealed population"
        )
    return MappingProxyType(plain)


def validate_evaluation_compute_class(
    value: Mapping[str, object],
) -> Mapping[str, object]:
    """Validate a host-independent Apollo/Goose execution-class authority."""

    plain = _plain(value)
    expected_fields = {
        "schema",
        "schema_version",
        "cluster",
        "scheduler",
        "slurm_partition",
        "platform",
        "inference",
        "execution_environment",
        "publication_eligible",
        "record_digest",
    }
    if not isinstance(plain, dict) or set(plain) != expected_fields:
        raise ValueError("evaluation compute-class fields differ")
    body = {key: item for key, item in plain.items() if key != "record_digest"}
    if (
        plain["schema"] != "isingfold.evaluation-compute-class"
        or plain["schema_version"] != 1
        or plain["record_digest"] != content_digest(body)
    ):
        raise ValueError("evaluation compute-class schema or digest is invalid")
    platform_identity = plain["platform"]
    inference = plain["inference"]
    environment = plain["execution_environment"]
    if not isinstance(platform_identity, dict) or set(platform_identity) != {
        "system",
        "release",
        "machine",
        "processor",
    }:
        raise ValueError("evaluation compute-class platform fields differ")
    if any(
        not isinstance(platform_identity[name], str) or not platform_identity[name]
        for name in platform_identity
    ):
        raise ValueError("evaluation compute-class platform identity is malformed")
    if not isinstance(inference, dict) or set(inference) != {
        "device_type",
        "device_name",
        "threads",
        "deterministic",
    }:
        raise ValueError("evaluation compute-class inference fields differ")
    if (
        not isinstance(inference["device_type"], str)
        or not inference["device_type"]
        or not isinstance(inference["device_name"], str)
        or not inference["device_name"]
        or type(inference["threads"]) is not int
        or inference["threads"] <= 0
        or type(inference["deterministic"]) is not bool
    ):
        raise ValueError("evaluation compute-class inference identity is malformed")
    environment_fields = {
        "mode",
        "environment_lock_path",
        "environment_lock_sha256",
        "container_runtime_path",
        "container_runtime_sha256",
        "container_image_path",
        "container_image_sha256",
        "python_executable_sha256",
        "runtime_implementation_digest",
        "source_digest",
    }
    if not isinstance(environment, dict) or set(environment) != environment_fields:
        raise ValueError("evaluation execution-environment fields differ")
    if any(
        not _is_digest(environment[name])
        for name in (
            "python_executable_sha256",
            "runtime_implementation_digest",
            "source_digest",
        )
    ):
        raise ValueError("evaluation execution-environment identity is malformed")
    mode = environment["mode"]
    path_pin_pairs = (
        ("environment_lock_path", "environment_lock_sha256"),
        ("container_runtime_path", "container_runtime_sha256"),
        ("container_image_path", "container_image_sha256"),
    )
    for path_name, digest_name in path_pin_pairs:
        path_value = environment[path_name]
        digest_value = environment[digest_name]
        if (path_value is None) != (digest_value is None):
            raise ValueError("evaluation environment artifact path/pin is incomplete")
        if path_value is not None and (
            not isinstance(path_value, str) or not path_value or not _is_digest(digest_value)
        ):
            raise ValueError("evaluation environment artifact identity is malformed")
    pinned_venv = environment["environment_lock_path"] is not None
    pinned_container = (
        environment["container_runtime_path"] is not None
        and environment["container_image_path"] is not None
    )
    if mode == "pinned-venv":
        valid_mode = pinned_venv and not pinned_container
    elif mode == "apptainer":
        valid_mode = pinned_container and not pinned_venv
    elif mode == "bare-metal":
        valid_mode = not pinned_venv and not pinned_container
    else:
        valid_mode = False
    if not valid_mode:
        raise ValueError("evaluation execution mode and artifact pins disagree")
    cluster = plain["cluster"]
    scheduler = plain["scheduler"]
    partition = plain["slurm_partition"]
    if cluster == "goose":
        valid_cluster = scheduler == "slurm" and isinstance(partition, str) and bool(partition)
    elif cluster == "apollo":
        valid_cluster = scheduler == "direct" and partition is None
    else:
        valid_cluster = False
    expected_eligible = mode in {"pinned-venv", "apptainer"} and inference["deterministic"] is True
    if (
        not valid_cluster
        or type(plain["publication_eligible"]) is not bool
        or plain["publication_eligible"] is not expected_eligible
    ):
        raise ValueError("evaluation compute class is not a registered execution class")
    return MappingProxyType(plain)


def _validate_compute_provenance(
    plan: EvaluationExecutionPlan,
    value: Mapping[str, object] | None,
    *,
    shard_index: int,
) -> Mapping[str, object] | None:
    if plan.compute_class is None:
        if value is not None:
            raise EvaluationProtocolError(
                "diagnostic plan cannot acquire an unplanned compute provenance"
            )
        return None
    if value is None:
        raise EvaluationProtocolError("evaluation shard requires compute provenance")
    plain = _plain(value)
    expected_fields = {
        "schema",
        "schema_version",
        "compute_class",
        "compute_class_digest",
        "runtime_identity",
        "hostname",
        "node_id",
        "slurm_job_id",
        "slurm_array_task_id",
        "record_digest",
    }
    if not isinstance(plain, dict) or set(plain) != expected_fields:
        raise EvaluationProtocolError("evaluation compute-provenance fields differ")
    body = {key: item for key, item in plain.items() if key != "record_digest"}
    if (
        plain["schema"] != "isingfold.evaluation-compute-provenance"
        or plain["schema_version"] != 1
        or plain["record_digest"] != content_digest(body)
        or plain["compute_class_digest"] != plan.compute_class_digest
        or plain["compute_class"] != dict(plan.compute_class)
    ):
        raise EvaluationProtocolError("evaluation compute provenance differs from its plan")
    runtime = plain["runtime_identity"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "runtime_platform",
        "inference_device_type",
        "inference_device_name",
        "inference_threads",
        "deterministic",
    }:
        raise EvaluationProtocolError("evaluation runtime identity fields differ")
    runtime_platform = runtime["runtime_platform"]
    if not isinstance(runtime_platform, dict) or set(runtime_platform) != {
        "hostname",
        "system",
        "release",
        "machine",
        "processor",
        "logical_cpu_count",
        "slurm_partition",
    }:
        raise EvaluationProtocolError("evaluation runtime-platform fields differ")
    compute_class = plan.compute_class
    class_platform = compute_class["platform"]
    class_inference = compute_class["inference"]
    assert isinstance(class_platform, Mapping)
    assert isinstance(class_inference, Mapping)
    if (
        {name: runtime_platform[name] for name in class_platform} != dict(class_platform)
        or type(runtime_platform["logical_cpu_count"]) is not int
        or runtime_platform["logical_cpu_count"] <= 0
        or runtime["inference_device_type"] != class_inference["device_type"]
        or runtime["inference_device_name"] != class_inference["device_name"]
        or runtime["inference_threads"] != class_inference["threads"]
        or runtime["deterministic"] != class_inference["deterministic"]
        or plain["hostname"] != runtime_platform["hostname"]
        or not isinstance(plain["hostname"], str)
        or not plain["hostname"]
        or not isinstance(plain["node_id"], str)
        or not plain["node_id"]
    ):
        raise EvaluationProtocolError("evaluation runtime differs from its compute class")
    if compute_class["cluster"] == "goose":
        slurm_valid = (
            runtime_platform["slurm_partition"] == compute_class["slurm_partition"]
            and isinstance(plain["slurm_job_id"], str)
            and bool(plain["slurm_job_id"])
            and isinstance(plain["slurm_array_task_id"], str)
            and plain["slurm_array_task_id"] == str(shard_index)
        )
    else:
        slurm_valid = (
            runtime_platform["slurm_partition"] is None
            and plain["slurm_job_id"] is None
            and plain["slurm_array_task_id"] is None
        )
    if not slurm_valid:
        raise EvaluationProtocolError("evaluation scheduler provenance differs from its plan")
    return MappingProxyType(plain)


@dataclass(frozen=True)
class EvaluationShardSpec:
    shard_index: int
    shard_id: str
    lineages: tuple[str, ...]
    expected_instances: tuple[tuple[str, str], ...]
    expected_pair_keys: tuple[PairKey, ...]
    array_coordinates: Mapping[str, object]

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "shard_index": self.shard_index,
            "shard_id": self.shard_id,
            "lineages": list(self.lineages),
            "expected_instances": [list(value) for value in self.expected_instances],
            "expected_pair_keys": [list(value) for value in self.expected_pair_keys],
            "array_coordinates": dict(self.array_coordinates),
        }
        payload["record_digest"] = stable_digest(payload)
        return payload

    @property
    def digest(self) -> str:
        return str(self.as_dict()["record_digest"])


@dataclass(frozen=True)
class EvaluationExecutionPlan:
    workflow: EvaluationWorkflow
    population: Mapping[str, object]
    population_digest: str
    run_coordinates: Mapping[str, object]
    execution_contract: Mapping[str, object]
    quality_authority: Mapping[str, object] | None
    compute_class: Mapping[str, object] | None
    max_lineages_per_shard: int
    base_seed: int
    system_seed_domain: str
    evaluator_seed_domain: str
    shards: tuple[EvaluationShardSpec, ...]

    def __post_init__(self) -> None:
        workflow = EvaluationWorkflow(self.workflow)
        object.__setattr__(self, "workflow", workflow)
        population = _frozen_mapping(self.population, label="evaluation population")
        if _stable(population) != self.population_digest:
            raise ValueError("execution plan population digest mismatch")
        try:
            typed_population = _population_from_payload(population)
        except EvaluationProtocolError as exc:
            raise ValueError("execution plan population authority is invalid") from exc
        if typed_population.digest != self.population_digest:
            raise ValueError("execution plan typed population digest mismatch")
        expected_partition = "val" if workflow is EvaluationWorkflow.VALIDATION_TUNING else "test"
        if {row.learning_partition for row in typed_population.evaluation_strata} != {
            expected_partition
        }:
            raise ValueError("execution plan population partition differs from its workflow")
        object.__setattr__(self, "population", population)
        quality_authority = _validated_evaluation_quality_authority(
            self.quality_authority,
            expected_partition=expected_partition,
            expected_count=len(typed_population.expected_instances),
            expected_instance_set_digest=content_digest(
                sorted(instance for _, instance in typed_population.expected_instances)
            ),
        )
        object.__setattr__(self, "quality_authority", quality_authority)
        if self.compute_class is None:
            compute_class = None
        else:
            compute_class = validate_evaluation_compute_class(self.compute_class)
        object.__setattr__(self, "compute_class", compute_class)
        coordinates, coordinate_seed, system_domain, evaluator_domain = _coordinate_contract(
            workflow, self.run_coordinates
        )
        object.__setattr__(self, "run_coordinates", coordinates)
        contract = _validate_execution_contract(self.execution_contract, workflow)
        object.__setattr__(self, "execution_contract", contract)
        if (
            workflow is not EvaluationWorkflow.LEARNED_COMPLETE
            and quality_authority is not None
            and contract["quality_authority_digest"] != content_digest(quality_authority)
        ):
            raise ValueError(
                "external execution contract differs from the evaluation quality authority"
            )
        if workflow is not EvaluationWorkflow.LEARNED_COMPLETE:
            coordinate_prefix = (
                "tuning" if workflow is EvaluationWorkflow.VALIDATION_TUNING else "training"
            )
            if (
                contract["training_seed_index"] != coordinates[f"{coordinate_prefix}_seed_index"]
                or contract["training_seed"] != coordinates[f"{coordinate_prefix}_seed"]
            ):
                raise ValueError("execution contract and array seed coordinates differ")
            try:
                tuning = ExternalTuningExecutionBinding.from_mapping(contract["tuning_execution"])
            except (TypeError, ValueError) as exc:
                raise ValueError("execution plan tuning authority is invalid") from exc
            expected_mode = (
                "validation-candidate"
                if workflow is EvaluationWorkflow.VALIDATION_TUNING
                else "frozen-deployment"
            )
            if tuning.mode != expected_mode:
                raise ValueError("execution plan tuning mode differs from its workflow")
            if workflow is EvaluationWorkflow.VALIDATION_TUNING and (
                tuning.candidate_index != coordinates["candidate_index"]
                or tuning.candidate.candidate_id != coordinates["candidate_id"]
            ):
                raise ValueError("execution plan tuning candidate coordinates differ")
        if (
            type(self.max_lineages_per_shard) is not int
            or not 1 <= self.max_lineages_per_shard <= DEFAULT_MAX_LINEAGES_PER_SHARD
        ):
            raise ValueError("execution plan shard size must be between one and 32 lineages")
        population_seed = typed_population.evaluation_seed
        expected_seed = coordinate_seed if coordinate_seed >= 0 else population_seed
        if type(expected_seed) is not int or self.base_seed != expected_seed:
            raise ValueError("execution plan base seed differs from its workflow authority")
        if (
            self.system_seed_domain != system_domain
            or self.evaluator_seed_domain != evaluator_domain
        ):
            raise ValueError("execution plan seed domains differ from the workflow registry")
        instances = _population_instances(population)
        repetitions = population.get("expected_repetitions")
        if type(repetitions) is not int or repetitions <= 0:
            raise ValueError("execution plan population repetitions are invalid")
        expected = _build_shards(
            workflow=workflow,
            population_digest=self.population_digest,
            instances=instances,
            repetitions=repetitions,
            run_coordinates=coordinates,
            max_lineages_per_shard=self.max_lineages_per_shard,
        )
        if tuple(item.as_dict() for item in self.shards) != tuple(
            item.as_dict() for item in expected
        ):
            raise ValueError("execution plan shards are not the deterministic lineage partition")

    @property
    def expected_pair_keys(self) -> tuple[PairKey, ...]:
        return tuple(key for shard in self.shards for key in shard.expected_pair_keys)

    @property
    def execution_contract_digest(self) -> str:
        return _stable(self.execution_contract)

    @property
    def quality_authority_digest(self) -> str | None:
        if self.quality_authority is None:
            return None
        return content_digest(self.quality_authority)

    @property
    def compute_class_digest(self) -> str | None:
        if self.compute_class is None:
            return None
        return content_digest(self.compute_class)

    @property
    def publication_eligible(self) -> bool:
        return bool(
            self.quality_authority is not None
            and self.compute_class is not None
            and self.compute_class["publication_eligible"] is True
        )

    @property
    def seed_schedule(self) -> Mapping[str, object]:
        return MappingProxyType(
            {
                "algorithm": SEED_ALGORITHM,
                "base_seed": self.base_seed,
                "system_domain": self.system_seed_domain,
                "evaluator_domain": self.evaluator_seed_domain,
                "identity_fields": ["lineage", "instance", "repetition"],
                "excluded_fields": ["shard_index", "shard_count", "execution_order"],
            }
        )

    @property
    def seed_schedule_digest(self) -> str:
        return _stable(self.seed_schedule)

    def as_dict(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": EVALUATION_PLAN_SCHEMA,
            "schema_version": EVALUATION_PLAN_VERSION,
            "workflow": self.workflow.value,
            "population": dict(self.population),
            "population_digest": self.population_digest,
            "run_coordinates": dict(self.run_coordinates),
            "execution_contract": dict(self.execution_contract),
            "execution_contract_digest": self.execution_contract_digest,
            "quality_authority": (
                None if self.quality_authority is None else dict(self.quality_authority)
            ),
            "quality_authority_digest": self.quality_authority_digest,
            "compute_class": (None if self.compute_class is None else dict(self.compute_class)),
            "compute_class_digest": self.compute_class_digest,
            "publication_eligible": self.publication_eligible,
            "max_lineages_per_shard": self.max_lineages_per_shard,
            "seed_schedule": dict(self.seed_schedule),
            "seed_schedule_digest": self.seed_schedule_digest,
            "shards": [item.as_dict() for item in self.shards],
        }
        if include_digest:
            payload["record_digest"] = stable_digest(payload)
        return payload

    @property
    def record_digest(self) -> str:
        return str(self.as_dict()["record_digest"])

    def shard(self, shard_index: int) -> EvaluationShardSpec:
        if type(shard_index) is not int or shard_index not in range(len(self.shards)):
            raise IndexError("evaluation shard index is outside the sealed plan")
        return self.shards[shard_index]


def _population_instances(population: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    raw = population.get("expected_instances")
    if not isinstance(raw, list):
        raise ValueError("execution plan population has no instance census")
    instances: list[tuple[str, str]] = []
    for item in raw:
        if (
            not isinstance(item, list)
            or len(item) != 2
            or any(not isinstance(value, str) or not value for value in item)
        ):
            raise ValueError("execution plan population instance census is malformed")
        instances.append((item[0], item[1]))
    result = tuple(instances)
    if result != tuple(sorted(set(result))) or not result:
        raise ValueError("execution plan population instance census is not canonical")
    return result


def _build_shards(
    *,
    workflow: EvaluationWorkflow,
    population_digest: str,
    instances: tuple[tuple[str, str], ...],
    repetitions: int,
    run_coordinates: Mapping[str, object],
    max_lineages_per_shard: int,
) -> tuple[EvaluationShardSpec, ...]:
    lineages = tuple(sorted({lineage for lineage, _ in instances}))
    result: list[EvaluationShardSpec] = []
    for shard_index, start in enumerate(range(0, len(lineages), max_lineages_per_shard)):
        member_lineages = lineages[start : start + max_lineages_per_shard]
        member_set = set(member_lineages)
        members = tuple(item for item in instances if item[0] in member_set)
        keys = tuple(
            sorted(
                (lineage, instance, repetition)
                for lineage, instance in members
                for repetition in range(repetitions)
            )
        )
        array_coordinates = dict(run_coordinates)
        array_coordinates["shard_index"] = shard_index
        shard_id = _stable(
            {
                "workflow": workflow.value,
                "population_digest": population_digest,
                "run_coordinates": run_coordinates,
                "lineages": member_lineages,
            }
        )
        result.append(
            EvaluationShardSpec(
                shard_index=shard_index,
                shard_id=shard_id,
                lineages=member_lineages,
                expected_instances=members,
                expected_pair_keys=keys,
                array_coordinates=MappingProxyType(array_coordinates),
            )
        )
    return tuple(result)


def build_evaluation_plan(
    *,
    workflow: EvaluationWorkflow | str,
    population: CompletePopulationIdentity,
    run_coordinates: Mapping[str, object],
    execution_contract: Mapping[str, object],
    quality_authority: Mapping[str, object] | None,
    compute_class: Mapping[str, object] | None,
    max_lineages_per_shard: int = DEFAULT_MAX_LINEAGES_PER_SHARD,
) -> EvaluationExecutionPlan:
    """Create the deterministic full-population execution plan before any shard runs."""

    if not isinstance(population, CompletePopulationIdentity):
        raise TypeError("evaluation plan requires a typed complete population")
    typed_workflow = EvaluationWorkflow(workflow)
    coordinates, coordinate_seed, system_domain, evaluator_domain = _coordinate_contract(
        typed_workflow, run_coordinates
    )
    contract = _validate_execution_contract(execution_contract, typed_workflow)
    if (
        type(max_lineages_per_shard) is not int
        or not 1 <= max_lineages_per_shard <= DEFAULT_MAX_LINEAGES_PER_SHARD
    ):
        raise ValueError("max_lineages_per_shard must be between one and 32")
    population_payload = population.as_dict()
    base_seed = coordinate_seed if coordinate_seed >= 0 else population.evaluation_seed
    shards = _build_shards(
        workflow=typed_workflow,
        population_digest=population.digest,
        instances=population.expected_instances,
        repetitions=population.expected_repetitions,
        run_coordinates=coordinates,
        max_lineages_per_shard=max_lineages_per_shard,
    )
    return EvaluationExecutionPlan(
        workflow=typed_workflow,
        population=population_payload,
        population_digest=population.digest,
        run_coordinates=coordinates,
        execution_contract=contract,
        quality_authority=quality_authority,
        compute_class=compute_class,
        max_lineages_per_shard=max_lineages_per_shard,
        base_seed=base_seed,
        system_seed_domain=system_domain,
        evaluator_seed_domain=evaluator_domain,
        shards=shards,
    )


def select_tasks_for_shard(
    plan: EvaluationExecutionPlan,
    shard_index: int,
    tasks: Sequence[EmbeddingTask],
) -> tuple[EmbeddingTask, ...]:
    """Authenticate the full task payload, then return one whole-lineage slice."""

    identities = tuple(sorted((task.lineage or task.name, task.name) for task in tasks))
    if identities != _population_instances(plan.population):
        raise EvaluationProtocolError("task census differs from the sealed execution plan")
    if task_population_digest(tasks) != plan.population.get("task_payload_sha256"):
        raise EvaluationProtocolError("task payload differs from the sealed execution plan")
    lineages = set(plan.shard(shard_index).lineages)
    selected = tuple(task for task in tasks if (task.lineage or task.name) in lineages)
    observed = tuple(sorted((task.lineage or task.name, task.name) for task in selected))
    if observed != plan.shard(shard_index).expected_instances:
        raise EvaluationProtocolError("task slice is not lineage coherent")
    return selected


def _plan_from_mapping(value: Mapping[str, object]) -> EvaluationExecutionPlan:
    expected = {
        "schema",
        "schema_version",
        "workflow",
        "population",
        "population_digest",
        "run_coordinates",
        "execution_contract",
        "execution_contract_digest",
        "quality_authority",
        "quality_authority_digest",
        "compute_class",
        "compute_class_digest",
        "publication_eligible",
        "max_lineages_per_shard",
        "seed_schedule",
        "seed_schedule_digest",
        "shards",
        "record_digest",
    }
    if set(value) != expected:
        raise EvaluationProtocolError("execution plan fields differ")
    body = {key: item for key, item in value.items() if key != "record_digest"}
    if (
        value["schema"] != EVALUATION_PLAN_SCHEMA
        or value["schema_version"] != EVALUATION_PLAN_VERSION
        or value["record_digest"] != stable_digest(body)
        or not isinstance(value["population"], Mapping)
        or not isinstance(value["run_coordinates"], Mapping)
        or not isinstance(value["execution_contract"], Mapping)
        or (
            value["quality_authority"] is not None
            and not isinstance(value["quality_authority"], Mapping)
        )
        or (value["compute_class"] is not None and not isinstance(value["compute_class"], Mapping))
        or not isinstance(value["seed_schedule"], Mapping)
        or not isinstance(value["shards"], list)
    ):
        raise EvaluationProtocolError("execution plan schema or digest is invalid")
    specs: list[EvaluationShardSpec] = []
    for row in value["shards"]:
        if not isinstance(row, Mapping) or set(row) != {
            "shard_index",
            "shard_id",
            "lineages",
            "expected_instances",
            "expected_pair_keys",
            "array_coordinates",
            "record_digest",
        }:
            raise EvaluationProtocolError("execution shard specification is malformed")
        row_body = {key: item for key, item in row.items() if key != "record_digest"}
        if row["record_digest"] != stable_digest(row_body):
            raise EvaluationProtocolError("execution shard specification digest mismatch")
        try:
            specs.append(
                EvaluationShardSpec(
                    shard_index=row["shard_index"],  # type: ignore[arg-type]
                    shard_id=row["shard_id"],  # type: ignore[arg-type]
                    lineages=tuple(row["lineages"]),  # type: ignore[arg-type]
                    expected_instances=tuple(
                        tuple(item)
                        for item in row["expected_instances"]  # type: ignore[arg-type]
                    ),
                    expected_pair_keys=tuple(
                        tuple(item)
                        for item in row["expected_pair_keys"]  # type: ignore[arg-type]
                    ),
                    array_coordinates=MappingProxyType(dict(row["array_coordinates"])),  # type: ignore[arg-type]
                )
            )
        except (TypeError, ValueError) as exc:
            raise EvaluationProtocolError("execution shard specification is invalid") from exc
    seed_schedule = value["seed_schedule"]
    try:
        plan = EvaluationExecutionPlan(
            workflow=EvaluationWorkflow(value["workflow"]),
            population=value["population"],
            population_digest=value["population_digest"],  # type: ignore[arg-type]
            run_coordinates=value["run_coordinates"],
            execution_contract=value["execution_contract"],
            quality_authority=value["quality_authority"],  # type: ignore[arg-type]
            compute_class=value["compute_class"],  # type: ignore[arg-type]
            max_lineages_per_shard=value["max_lineages_per_shard"],  # type: ignore[arg-type]
            base_seed=seed_schedule["base_seed"],  # type: ignore[arg-type]
            system_seed_domain=seed_schedule["system_domain"],  # type: ignore[arg-type]
            evaluator_seed_domain=seed_schedule["evaluator_domain"],  # type: ignore[arg-type]
            shards=tuple(specs),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationProtocolError("execution plan is semantically invalid") from exc
    if plan.as_dict() != dict(value):
        raise EvaluationProtocolError("execution plan is not canonical under its schema")
    return plan


def _atomic_write_new(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise FileExistsError(path) from None
        temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)


def write_evaluation_plan(path: str | os.PathLike[str], plan: EvaluationExecutionPlan) -> str:
    """Atomically create a canonical plan and return its file SHA-256."""

    content = canonical_json_bytes(plan.as_dict()) + b"\n"
    _atomic_write_new(Path(path), content)
    return hashlib.sha256(content).hexdigest()


def load_evaluation_plan(
    path: str | os.PathLike[str], *, expected_sha256: str
) -> EvaluationExecutionPlan:
    """Load a plan only after checking its independently supplied file hash."""

    if not _is_digest(expected_sha256):
        raise ValueError("execution plan requires an out-of-band SHA-256")
    content = Path(path).read_bytes()
    observed = hashlib.sha256(content).hexdigest()
    if not hmac.compare_digest(observed, expected_sha256):
        raise EvaluationProtocolError("execution plan file differs from its SHA-256 pin")
    return _plan_from_mapping(_strict_json(content, label="execution plan"))


def _pair_key(row: Mapping[str, object]) -> PairKey:
    key = (row.get("lineage"), row.get("instance"), row.get("repetition"))
    if (
        not isinstance(key[0], str)
        or not key[0]
        or not isinstance(key[1], str)
        or not key[1]
        or type(key[2]) is not int
        or key[2] < 0
    ):
        raise EvaluationProtocolError("evaluation artifact row has an invalid pair key")
    return key  # type: ignore[return-value]


def _validate_inner_digest(row: Mapping[str, object], *, label: str) -> None:
    recorded = row.get("record_digest")
    body = {key: value for key, value in row.items() if key != "record_digest"}
    if not _is_digest(recorded) or recorded != stable_digest(body):
        raise EvaluationProtocolError(f"{label} row record digest mismatch")


def _learned_artifact_rows(
    receipts: Sequence[CompleteSystemReceipt],
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
]:
    receipt_rows: list[Mapping[str, object]] = []
    outcome_rows: list[Mapping[str, object]] = []
    evidence_rows: list[Mapping[str, object]] = []
    for receipt in sorted(receipts, key=lambda item: item.pair_key):
        receipt_payload = receipt.as_dict()
        outcome: dict[str, object] = {
            "schema": COMPLETE_SYSTEM_OUTCOME_SCHEMA,
            "schema_version": COMPLETE_SYSTEM_OUTCOME_VERSION,
            "instance": receipt.instance,
            "lineage": receipt.lineage,
            "repetition": receipt.repetition,
            "population_digest": receipt.population.digest,
            "complete_receipt_digest": receipt_payload["record_digest"],
            "outcome": receipt.outcome.as_dict(),
        }
        outcome["record_digest"] = stable_digest(outcome)
        evidence = CompleteSystemEvidenceRecord(
            instance=receipt.instance,
            lineage=receipt.lineage,
            repetition=receipt.repetition,
            population_digest=receipt.population.digest,
            complete_receipt_digest=str(receipt_payload["record_digest"]),
            terminal_evidence=receipt.terminal_evidence,
        )
        receipt_rows.append(MappingProxyType(receipt_payload))
        outcome_rows.append(MappingProxyType(outcome))
        evidence_rows.append(MappingProxyType(evidence.as_dict()))
    return tuple(receipt_rows), tuple(outcome_rows), tuple(evidence_rows)


def _external_artifact_rows(
    receipts: Sequence[ExternalCompleteSystemReceipt],
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
]:
    ordered = sorted(receipts, key=lambda item: item.pair_key)
    receipt_rows = tuple(MappingProxyType(item.as_dict()) for item in ordered)
    outcome_rows = tuple(MappingProxyType(item.outcome.as_dict()) for item in ordered)
    evidence_rows = tuple(
        MappingProxyType(
            ExternalCompleteSystemEvidenceRecord(
                instance=item.instance,
                lineage=item.lineage,
                repetition=item.repetition,
                population_digest=item.population.digest,
                external_receipt_digest=str(item.as_dict()["record_digest"]),
                terminal_evidence=item.terminal_evidence,
            ).as_dict()
        )
        for item in ordered
    )
    return receipt_rows, outcome_rows, evidence_rows


def _validate_shard_receipts(
    plan: EvaluationExecutionPlan,
    shard: EvaluationShardSpec,
    receipts: Sequence[TypedReceipt],
) -> None:
    if not receipts:
        raise EvaluationProtocolError("cannot publish an empty evaluation shard")
    learned = plan.workflow is EvaluationWorkflow.LEARNED_COMPLETE
    expected_type = CompleteSystemReceipt if learned else ExternalCompleteSystemReceipt
    if any(not isinstance(item, expected_type) for item in receipts):
        raise TypeError("evaluation shard receipt type differs from its workflow")
    keys = tuple(sorted(item.pair_key for item in receipts))
    if len(keys) != len(set(keys)) or keys != shard.expected_pair_keys:
        raise EvaluationProtocolError(
            "evaluation shard receipts do not cover exactly their lineage-coherent census"
        )
    for receipt in receipts:
        if receipt.population.digest != plan.population_digest:
            raise EvaluationProtocolError("evaluation shard receipt population drift")
        if dict(receipt_execution_contract(receipt)) != dict(plan.execution_contract):
            raise EvaluationProtocolError("evaluation shard receipt execution-contract drift")
        expected_system = identity_seed(
            plan.base_seed,
            plan.system_seed_domain,
            lineage=receipt.lineage,
            instance=receipt.instance,
            repetition=receipt.repetition,
        )
        if receipt.system_seed != expected_system:
            raise EvaluationProtocolError("evaluation shard system seed differs from its plan")
        if receipt.outcome.returned_valid:
            expected_evaluator = identity_seed(
                plan.base_seed,
                plan.evaluator_seed_domain,
                lineage=receipt.lineage,
                instance=receipt.instance,
                repetition=receipt.repetition,
            )
            if receipt.evaluator_seed != expected_evaluator:
                raise EvaluationProtocolError(
                    "evaluation shard evaluator seed differs from its plan"
                )


def _validate_target_access_receipt(
    plan: EvaluationExecutionPlan,
    value: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    if value is None:
        return None
    plain = _plain(value)
    expected_fields = {
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
    if not isinstance(plain, dict) or set(plain) != expected_fields:
        raise EvaluationProtocolError("evaluation shard target-access receipt fields differ")
    body = {key: item for key, item in plain.items() if key != "record_digest"}
    typed_population = _population_from_payload(plan.population)
    partition = typed_population.evaluation_strata[0].learning_partition
    expected_partition = "val" if plan.workflow is EvaluationWorkflow.VALIDATION_TUNING else "test"
    if (
        plain["record_digest"] != content_digest(body)
        or partition != expected_partition
        or plain["partition"] != expected_partition
        or plain["target_path"] != f"targets/{expected_partition}.jsonl"
        or plain["prepared_manifest_sha256"] != typed_population.source_manifest_sha256
        or type(plain["target_count"]) is not int
        or plain["target_count"] != len(typed_population.expected_instances)
    ):
        raise EvaluationProtocolError(
            "evaluation shard target access differs from its sealed population"
        )
    digest_fields = expected_fields - {
        "opened_files",
        "partition",
        "publisher_id",
        "record_digest",
        "target_count",
        "target_path",
    }
    if (
        any(not _is_digest(plain[name]) for name in digest_fields)
        or not isinstance(plain["publisher_id"], str)
        or not plain["publisher_id"]
        or not isinstance(plain["opened_files"], list)
        or not plain["opened_files"]
    ):
        raise EvaluationProtocolError("evaluation shard target-access authority is malformed")
    return MappingProxyType(plain)


def _validate_ground_certificate_authority(
    plan: EvaluationExecutionPlan,
    value: Mapping[str, object] | None,
    *,
    target_access_receipt: Mapping[str, object] | None,
) -> Mapping[str, object] | None:
    if value is None:
        if plan.quality_authority is not None:
            raise EvaluationProtocolError(
                "quality-authorized plan requires its opened ground authority"
            )
        if target_access_receipt is not None:
            raise EvaluationProtocolError("target access has no planned quality authority")
        return None
    if target_access_receipt is None:
        raise EvaluationProtocolError(
            "ground-certificate authority requires its target-access receipt"
        )
    expected_partition = "val" if plan.workflow is EvaluationWorkflow.VALIDATION_TUNING else "test"
    authority = _validated_evaluation_quality_authority(
        value,
        expected_partition=expected_partition,
        expected_count=len(_population_instances(plan.population)),
        expected_instance_set_digest=content_digest(
            sorted(instance for _, instance in _population_instances(plan.population))
        ),
    )
    if authority is None or plan.quality_authority is None:
        raise EvaluationProtocolError("ground authority was not precommitted by the plan")
    if dict(authority) != dict(plan.quality_authority):
        raise EvaluationProtocolError("opened quality authority differs from its sealed plan")
    global_authority = authority["global"]
    partition_authority = authority["evaluation_partition"]
    assert isinstance(global_authority, Mapping)
    assert isinstance(partition_authority, Mapping)
    if (
        global_authority["publisher_id"] != target_access_receipt["publisher_id"]
        or global_authority["publisher_attestation_record_digest"]
        != target_access_receipt["publisher_attestation_digest"]
        or global_authority["target_authority_record_digest"]
        != target_access_receipt["target_authority_record_digest"]
        or partition_authority["target_access_record_digest"]
        != target_access_receipt["record_digest"]
        or partition_authority["target_set_digest"] != target_access_receipt["target_set_digest"]
        or partition_authority["target_count"] != target_access_receipt["target_count"]
        or partition_authority["evidence_manifest_record_digest"]
        != target_access_receipt["evidence_manifest_record_digest"]
        or partition_authority["evidence_manifest_sha256"]
        != target_access_receipt["evidence_manifest_sha256"]
    ):
        raise EvaluationProtocolError(
            "partition quality-authority differs from target access or sealed population"
        )
    return authority


@dataclass(frozen=True)
class LoadedEvaluationShard:
    workflow: EvaluationWorkflow
    plan_record_digest: str
    shard_index: int
    shard_id: str
    receipt_file_sha256: str
    receipt_record_digest: str
    target_access_receipt: Mapping[str, object] | None
    ground_certificate_authority: Mapping[str, object] | None
    compute_provenance: Mapping[str, object] | None
    receipt_rows: tuple[Mapping[str, object], ...]
    outcome_rows: tuple[Mapping[str, object], ...]
    evidence_rows: tuple[Mapping[str, object], ...]
    receipts: tuple[TypedReceipt, ...]
    outcomes: tuple[EpisodeOutcome, ...]
    evidence: tuple[TypedEvidence, ...]

    @property
    def pair_keys(self) -> tuple[PairKey, ...]:
        return tuple(_pair_key(row) for row in self.receipt_rows)


def _shard_receipt_payload(
    *,
    plan: EvaluationExecutionPlan,
    shard: EvaluationShardSpec,
    artifact_contents: Mapping[str, bytes],
    count: int,
    target_access_receipt: Mapping[str, object] | None,
    ground_certificate_authority: Mapping[str, object] | None,
    compute_provenance: Mapping[str, object] | None,
) -> dict[str, object]:
    artifacts = {
        name: {
            "path": path,
            "sha256": hashlib.sha256(artifact_contents[path]).hexdigest(),
            "count": count,
        }
        for name, path in {
            "receipts": "receipts.jsonl",
            "outcomes": "outcomes.jsonl",
            "terminal_evidence": "terminal_evidence.jsonl",
        }.items()
    }
    payload: dict[str, object] = {
        "schema": EVALUATION_SHARD_RECEIPT_SCHEMA,
        "schema_version": EVALUATION_SHARD_RECEIPT_VERSION,
        "workflow": plan.workflow.value,
        "plan_record_digest": plan.record_digest,
        "population_digest": plan.population_digest,
        "shard_index": shard.shard_index,
        "shard_id": shard.shard_id,
        "shard_spec_digest": shard.digest,
        "array_coordinates": dict(shard.array_coordinates),
        "execution_contract_digest": plan.execution_contract_digest,
        "compute_class_digest": plan.compute_class_digest,
        "publication_eligible": plan.publication_eligible,
        "seed_schedule_digest": plan.seed_schedule_digest,
        "pair_key_census_digest": stable_digest([list(key) for key in shard.expected_pair_keys]),
        "target_access_receipt": (
            None if target_access_receipt is None else dict(target_access_receipt)
        ),
        "target_access_receipt_digest": (
            None if target_access_receipt is None else target_access_receipt["record_digest"]
        ),
        "ground_certificate_authority": (
            None if ground_certificate_authority is None else dict(ground_certificate_authority)
        ),
        "ground_certificate_authority_digest": (
            None if ground_certificate_authority is None else _stable(ground_certificate_authority)
        ),
        "compute_provenance": (None if compute_provenance is None else dict(compute_provenance)),
        "compute_provenance_digest": (
            None if compute_provenance is None else compute_provenance["record_digest"]
        ),
        "artifacts": artifacts,
    }
    payload["record_digest"] = stable_digest(payload)
    return payload


def publish_evaluation_shard(
    destination: str | os.PathLike[str],
    *,
    plan: EvaluationExecutionPlan,
    shard_index: int,
    receipts: Sequence[TypedReceipt],
    target_access_receipt: Mapping[str, object] | None = None,
    ground_certificate_authority: Mapping[str, object] | None = None,
    compute_provenance: Mapping[str, object] | None = None,
) -> tuple[Mapping[str, object], str]:
    """Atomically publish one complete shard directory and return its receipt pin."""

    shard = plan.shard(shard_index)
    materialized = tuple(receipts)
    _validate_shard_receipts(plan, shard, materialized)
    access = _validate_target_access_receipt(plan, target_access_receipt)
    ground = _validate_ground_certificate_authority(
        plan,
        ground_certificate_authority,
        target_access_receipt=access,
    )
    provenance = _validate_compute_provenance(
        plan,
        compute_provenance,
        shard_index=shard_index,
    )
    if plan.workflow is not EvaluationWorkflow.LEARNED_COMPLETE:
        assert provenance is not None
        runtime_identity = provenance["runtime_identity"]
        expected_runtime_digest = content_digest(runtime_identity)
        if any(
            not isinstance(receipt, ExternalCompleteSystemReceipt)
            or receipt.runtime_identity_digest != expected_runtime_digest
            for receipt in materialized
        ):
            raise EvaluationProtocolError(
                "external shard receipts differ from their runtime provenance"
            )
    if plan.workflow is EvaluationWorkflow.LEARNED_COMPLETE:
        receipt_rows, outcome_rows, evidence_rows = _learned_artifact_rows(
            materialized  # type: ignore[arg-type]
        )
    else:
        receipt_rows, outcome_rows, evidence_rows = _external_artifact_rows(
            materialized  # type: ignore[arg-type]
        )
    artifact_contents = {
        "receipts.jsonl": _canonical_lines(receipt_rows),
        "outcomes.jsonl": _canonical_lines(outcome_rows),
        "terminal_evidence.jsonl": _canonical_lines(evidence_rows),
    }
    shard_receipt = _shard_receipt_payload(
        plan=plan,
        shard=shard,
        artifact_contents=artifact_contents,
        count=len(materialized),
        target_access_receipt=access,
        ground_certificate_authority=ground,
        compute_provenance=provenance,
    )
    shard_content = canonical_json_bytes(shard_receipt) + b"\n"
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        raise FileExistsError(destination_path)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination_path.name}.", dir=destination_path.parent)
    )
    try:
        for name, content in artifact_contents.items():
            _atomic_write_new(temporary / name, content)
        _atomic_write_new(temporary / "shard_receipt.json", shard_content)
        os.rename(temporary, destination_path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return MappingProxyType(shard_receipt), hashlib.sha256(shard_content).hexdigest()


def _artifact_payloads(
    directory: Path,
    receipt: Mapping[str, object],
) -> tuple[
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
    tuple[Mapping[str, object], ...],
]:
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "receipts",
        "outcomes",
        "terminal_evidence",
    }:
        raise EvaluationProtocolError("evaluation shard artifact registry is incomplete")
    result: dict[str, tuple[Mapping[str, object], ...]] = {}
    expected_paths = {
        "receipts": "receipts.jsonl",
        "outcomes": "outcomes.jsonl",
        "terminal_evidence": "terminal_evidence.jsonl",
    }
    for name, expected_path in expected_paths.items():
        entry = artifacts[name]
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"path", "sha256", "count"}
            or entry.get("path") != expected_path
            or not _is_digest(entry.get("sha256"))
            or type(entry.get("count")) is not int
            or entry.get("count", 0) <= 0
        ):
            raise EvaluationProtocolError("evaluation shard artifact metadata is malformed")
        content = (directory / expected_path).read_bytes()
        if not hmac.compare_digest(hashlib.sha256(content).hexdigest(), str(entry["sha256"])):
            raise EvaluationProtocolError(f"evaluation shard {name} artifact digest mismatch")
        rows = _strict_jsonl(content, label=f"evaluation shard {name}")
        if len(rows) != entry["count"]:
            raise EvaluationProtocolError(f"evaluation shard {name} count mismatch")
        result[name] = rows
    counts = {len(value) for value in result.values()}
    if len(counts) != 1:
        raise EvaluationProtocolError("evaluation shard artifact counts differ")
    return result["receipts"], result["outcomes"], result["terminal_evidence"]


def _typed_learned_rows(
    receipt_rows: Sequence[Mapping[str, object]],
    outcome_rows: Sequence[Mapping[str, object]],
    evidence_rows: Sequence[Mapping[str, object]],
) -> tuple[
    tuple[CompleteSystemReceipt, ...],
    tuple[EpisodeOutcome, ...],
    tuple[CompleteSystemEvidenceRecord, ...],
]:
    receipts = tuple(_receipt_from_payload(row) for row in receipt_rows)
    receipt_by_key = {item.pair_key: item for item in receipts}
    outcomes: list[EpisodeOutcome] = []
    evidence: list[CompleteSystemEvidenceRecord] = []
    for row in outcome_rows:
        _validate_inner_digest(row, label="learned outcome")
        key = _pair_key(row)
        receipt = receipt_by_key.get(key)
        if receipt is None:
            raise EvaluationProtocolError("learned shard outcome has no receipt")
        expected: dict[str, object] = {
            "schema": COMPLETE_SYSTEM_OUTCOME_SCHEMA,
            "schema_version": COMPLETE_SYSTEM_OUTCOME_VERSION,
            "instance": receipt.instance,
            "lineage": receipt.lineage,
            "repetition": receipt.repetition,
            "population_digest": receipt.population.digest,
            "complete_receipt_digest": receipt.as_dict()["record_digest"],
            "outcome": receipt.outcome.as_dict(),
        }
        expected["record_digest"] = stable_digest(expected)
        if dict(row) != expected:
            raise EvaluationProtocolError("learned shard outcome link mismatch")
        outcomes.append(receipt.outcome)
    for row in evidence_rows:
        record = _evidence_record_from_payload(row)
        receipt = receipt_by_key.get(record.pair_key)
        terminal_digest = (
            None if record.terminal_evidence is None else record.terminal_evidence.digest
        )
        if (
            receipt is None
            or record.population_digest != receipt.population.digest
            or record.complete_receipt_digest != receipt.as_dict()["record_digest"]
            or terminal_digest != receipt.terminal_evidence_digest
        ):
            raise EvaluationProtocolError("learned shard terminal-evidence link mismatch")
        evidence.append(record)
    return receipts, tuple(outcomes), tuple(evidence)


def _typed_external_rows(
    directory: Path,
    plan: EvaluationExecutionPlan,
    receipt_rows: Sequence[Mapping[str, object]],
    evidence_rows: Sequence[Mapping[str, object]],
    *,
    expected_runtime_identity_digest: str,
) -> tuple[
    tuple[ExternalCompleteSystemReceipt, ...],
    tuple[EpisodeOutcome, ...],
    tuple[ExternalCompleteSystemEvidenceRecord, ...],
]:
    contract = plan.execution_contract
    population = _population_from_payload(plan.population)
    backend = _backend_from_payload(contract["backend"])
    selector = _component_from_payload(contract["selector"])
    work = contract["work_cap"]
    if not isinstance(work, Mapping):
        raise EvaluationProtocolError("external shard work authority is malformed")
    tuning = ExternalTuningExecutionBinding.from_mapping(contract["tuning_execution"])
    outcomes = tuple(
        read_external_complete_outcomes(
            directory / "outcomes.jsonl",
            expected_sha256=hashlib.sha256((directory / "outcomes.jsonl").read_bytes()).hexdigest(),
        )
    )
    receipts = tuple(
        read_external_complete_receipts(
            directory / "receipts.jsonl",
            outcomes=outcomes,
            expected_backend=backend,
            expected_selector=selector,
            expected_population=population,
            expected_config_digest=str(contract["config_digest"]),
            expected_learned_config_digest=str(contract["learned_config_digest"]),
            expected_context_digest=str(contract["context_digest"]),
            expected_quality_authority_digest=str(contract["quality_authority_digest"]),
            expected_work_cap=WorkVector(**{name: work[name] for name in WORK_FIELDS}),
            expected_sha256=hashlib.sha256((directory / "receipts.jsonl").read_bytes()).hexdigest(),
            expected_training_seed_index=int(contract["training_seed_index"]),
            expected_training_seed=int(contract["training_seed"]),
            expected_runtime_identity_digest=expected_runtime_identity_digest,
            expected_tuning_execution=tuning,
        )
    )
    if tuple(item.as_dict() for item in receipts) != tuple(dict(row) for row in receipt_rows):
        raise EvaluationProtocolError("external typed receipts differ from their raw rows")
    receipt_by_key = {item.pair_key: item for item in receipts}
    evidence: list[ExternalCompleteSystemEvidenceRecord] = []
    for row in evidence_rows:
        _validate_inner_digest(row, label="external evidence")
        raw_terminal = row.get("terminal_evidence")
        if raw_terminal is not None:
            # The detailed semantic replay is performed by the established final merged
            # evidence loader once full tasks and Context are supplied.  Here the receipt
            # digest and terminal digest still make mutation or cross-shard substitution fail.
            from isingfold.rl.complete_system import _terminal_evidence_from_payload

            if not isinstance(raw_terminal, Mapping):
                raise EvaluationProtocolError("external terminal evidence is malformed")
            terminal = _terminal_evidence_from_payload(raw_terminal)
        else:
            terminal = None
        try:
            record = ExternalCompleteSystemEvidenceRecord(
                instance=row["instance"],  # type: ignore[arg-type]
                lineage=row["lineage"],  # type: ignore[arg-type]
                repetition=row["repetition"],  # type: ignore[arg-type]
                population_digest=row["population_digest"],  # type: ignore[arg-type]
                external_receipt_digest=row["external_receipt_digest"],  # type: ignore[arg-type]
                terminal_evidence=terminal,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EvaluationProtocolError("external shard evidence is invalid") from exc
        receipt = receipt_by_key.get(record.pair_key)
        terminal_digest = None if terminal is None else terminal.digest
        if (
            receipt is None
            or record.population_digest != receipt.population.digest
            or record.external_receipt_digest != receipt.as_dict()["record_digest"]
            or terminal_digest != receipt.terminal_evidence_digest
            or record.as_dict() != dict(row)
        ):
            raise EvaluationProtocolError("external shard terminal-evidence link mismatch")
        evidence.append(record)
    return receipts, outcomes, tuple(evidence)


def load_evaluation_shard(
    directory: str | os.PathLike[str],
    *,
    plan: EvaluationExecutionPlan,
    expected_receipt_sha256: str,
) -> LoadedEvaluationShard:
    """Authenticate a shard against its external receipt pin and the sealed plan."""

    if not _is_digest(expected_receipt_sha256):
        raise ValueError("evaluation shard requires an out-of-band receipt SHA-256")
    source = Path(directory)
    expected_files = {
        "receipts.jsonl",
        "outcomes.jsonl",
        "terminal_evidence.jsonl",
        "shard_receipt.json",
    }
    if not source.is_dir() or {item.name for item in source.iterdir()} != expected_files:
        raise EvaluationProtocolError("evaluation shard directory census differs")
    receipt_content = (source / "shard_receipt.json").read_bytes()
    observed = hashlib.sha256(receipt_content).hexdigest()
    if not hmac.compare_digest(observed, expected_receipt_sha256):
        raise EvaluationProtocolError("evaluation shard receipt differs from its SHA-256 pin")
    receipt = _strict_json(receipt_content, label="evaluation shard receipt")
    expected_fields = {
        "schema",
        "schema_version",
        "workflow",
        "plan_record_digest",
        "population_digest",
        "shard_index",
        "shard_id",
        "shard_spec_digest",
        "array_coordinates",
        "execution_contract_digest",
        "compute_class_digest",
        "publication_eligible",
        "seed_schedule_digest",
        "pair_key_census_digest",
        "target_access_receipt",
        "target_access_receipt_digest",
        "ground_certificate_authority",
        "ground_certificate_authority_digest",
        "compute_provenance",
        "compute_provenance_digest",
        "artifacts",
        "record_digest",
    }
    body = {key: value for key, value in receipt.items() if key != "record_digest"}
    if (
        set(receipt) != expected_fields
        or receipt.get("schema") != EVALUATION_SHARD_RECEIPT_SCHEMA
        or receipt.get("schema_version") != EVALUATION_SHARD_RECEIPT_VERSION
        or receipt.get("record_digest") != stable_digest(body)
        or type(receipt.get("shard_index")) is not int
    ):
        raise EvaluationProtocolError("evaluation shard receipt schema or digest is invalid")
    shard = plan.shard(int(receipt["shard_index"]))
    expected_identity = {
        "workflow": plan.workflow.value,
        "plan_record_digest": plan.record_digest,
        "population_digest": plan.population_digest,
        "shard_id": shard.shard_id,
        "shard_spec_digest": shard.digest,
        "array_coordinates": dict(shard.array_coordinates),
        "execution_contract_digest": plan.execution_contract_digest,
        "compute_class_digest": plan.compute_class_digest,
        "publication_eligible": plan.publication_eligible,
        "seed_schedule_digest": plan.seed_schedule_digest,
        "pair_key_census_digest": stable_digest([list(key) for key in shard.expected_pair_keys]),
    }
    if any(receipt.get(name) != value for name, value in expected_identity.items()):
        raise EvaluationProtocolError("evaluation shard receipt differs from its sealed plan")
    access = _validate_target_access_receipt(
        plan,
        receipt.get("target_access_receipt"),  # type: ignore[arg-type]
    )
    expected_access_digest = None if access is None else access["record_digest"]
    if receipt.get("target_access_receipt_digest") != expected_access_digest:
        raise EvaluationProtocolError("evaluation shard target-access digest mismatch")
    ground = _validate_ground_certificate_authority(
        plan,
        receipt.get("ground_certificate_authority"),  # type: ignore[arg-type]
        target_access_receipt=access,
    )
    expected_ground_digest = None if ground is None else _stable(ground)
    if receipt.get("ground_certificate_authority_digest") != expected_ground_digest:
        raise EvaluationProtocolError("evaluation shard ground-certificate digest mismatch")
    provenance = _validate_compute_provenance(
        plan,
        receipt.get("compute_provenance"),  # type: ignore[arg-type]
        shard_index=shard.shard_index,
    )
    expected_provenance_digest = None if provenance is None else provenance["record_digest"]
    if receipt.get("compute_provenance_digest") != expected_provenance_digest:
        raise EvaluationProtocolError("evaluation shard compute-provenance digest mismatch")
    receipt_rows, outcome_rows, evidence_rows = _artifact_payloads(source, receipt)
    for label, rows in (
        ("receipt", receipt_rows),
        ("outcome", outcome_rows),
        ("evidence", evidence_rows),
    ):
        keys = tuple(_pair_key(row) for row in rows)
        if keys != shard.expected_pair_keys:
            raise EvaluationProtocolError(
                f"evaluation shard {label} census or canonical order differs"
            )
    if plan.workflow is EvaluationWorkflow.LEARNED_COMPLETE:
        typed_receipts, outcomes, evidence = _typed_learned_rows(
            receipt_rows, outcome_rows, evidence_rows
        )
    else:
        if provenance is None:
            raise EvaluationProtocolError("external shard has no runtime provenance")
        typed_receipts, outcomes, evidence = _typed_external_rows(
            source,
            plan,
            receipt_rows,
            evidence_rows,
            expected_runtime_identity_digest=content_digest(provenance["runtime_identity"]),
        )
    _validate_shard_receipts(plan, shard, typed_receipts)
    return LoadedEvaluationShard(
        workflow=plan.workflow,
        plan_record_digest=plan.record_digest,
        shard_index=shard.shard_index,
        shard_id=shard.shard_id,
        receipt_file_sha256=observed,
        receipt_record_digest=str(receipt["record_digest"]),
        target_access_receipt=access,
        ground_certificate_authority=ground,
        compute_provenance=provenance,
        receipt_rows=tuple(receipt_rows),
        outcome_rows=tuple(outcome_rows),
        evidence_rows=tuple(evidence_rows),
        receipts=tuple(typed_receipts),
        outcomes=tuple(outcomes),
        evidence=tuple(evidence),
    )


@dataclass(frozen=True)
class MergedEvaluationArtifacts:
    workflow: EvaluationWorkflow
    plan_record_digest: str
    shard_receipt_sha256: tuple[str, ...]
    target_access_receipt: Mapping[str, object] | None
    ground_certificate_authority: Mapping[str, object] | None
    compute_provenance: tuple[Mapping[str, object], ...]
    receipt_rows: tuple[Mapping[str, object], ...]
    outcome_rows: tuple[Mapping[str, object], ...]
    evidence_rows: tuple[Mapping[str, object], ...]
    receipts: tuple[TypedReceipt, ...]
    outcomes: tuple[EpisodeOutcome, ...]
    evidence: tuple[TypedEvidence, ...]
    recomputed_report_inputs: Mapping[str, object]

    @property
    def receipts_bytes(self) -> bytes:
        return _canonical_lines(self.receipt_rows)

    @property
    def outcomes_bytes(self) -> bytes:
        return _canonical_lines(self.outcome_rows)

    @property
    def evidence_bytes(self) -> bytes:
        return _canonical_lines(self.evidence_rows)

    @property
    def artifact_sha256(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "receipts": hashlib.sha256(self.receipts_bytes).hexdigest(),
                "outcomes": hashlib.sha256(self.outcomes_bytes).hexdigest(),
                "terminal_evidence": hashlib.sha256(self.evidence_bytes).hexdigest(),
            }
        )


def _sharded_external_tuning_lineage_metrics(
    receipts: Sequence[TypedReceipt],
) -> list[dict[str, object]]:
    """Use the normative reducer after projecting away authenticated node identity.

    ``merge_evaluation_shards`` has already checked every exact runtime against its
    shard provenance and every provenance against one plan compute class. The legacy
    monolithic reducer correctly rejects mixed runtime digests, so this private adapter
    normalizes only that one node-specific field and leaves all scientific authorities
    and observations under the reducer's existing fail-closed checks.
    """

    if not receipts or any(
        not isinstance(receipt, ExternalCompleteSystemReceipt) for receipt in receipts
    ):
        raise TypeError("sharded tuning metrics require external receipts")
    runtime_digest = receipts[0].runtime_identity_digest  # type: ignore[union-attr]
    projected = tuple(
        replace(receipt, runtime_identity_digest=runtime_digest) for receipt in receipts
    )
    return external_tuning_lineage_metrics(projected)


def merge_evaluation_shards(
    plan: EvaluationExecutionPlan,
    shards: Sequence[LoadedEvaluationShard],
    *,
    tasks: Sequence[EmbeddingTask] | None = None,
    context: Context | None = None,
) -> MergedEvaluationArtifacts:
    """Reject an incomplete/overlapping shard set and recompute unsharded report inputs."""

    materialized = tuple(shards)
    by_index = {shard.shard_index: shard for shard in materialized}
    expected_indices = set(range(len(plan.shards)))
    if len(by_index) != len(materialized):
        raise EvaluationProtocolError("evaluation merge contains duplicate shard indices")
    if set(by_index) != expected_indices:
        missing = sorted(expected_indices - set(by_index))
        extra = sorted(set(by_index) - expected_indices)
        raise EvaluationProtocolError(
            f"evaluation merge shard census differs: missing={missing}, extra={extra}"
        )
    all_keys: list[PairKey] = []
    for index, loaded in sorted(by_index.items()):
        spec = plan.shard(index)
        if (
            loaded.workflow is not plan.workflow
            or loaded.plan_record_digest != plan.record_digest
            or loaded.shard_id != spec.shard_id
            or loaded.pair_keys != spec.expected_pair_keys
        ):
            raise EvaluationProtocolError("evaluation merge contains shard identity drift")
        _validate_ground_certificate_authority(
            plan,
            loaded.ground_certificate_authority,
            target_access_receipt=loaded.target_access_receipt,
        )
        _validate_compute_provenance(
            plan,
            loaded.compute_provenance,
            shard_index=index,
        )
        all_keys.extend(loaded.pair_keys)
    if len(all_keys) != len(set(all_keys)):
        raise EvaluationProtocolError("evaluation merge contains overlapping pair keys")
    if tuple(sorted(all_keys)) != tuple(sorted(plan.expected_pair_keys)):
        raise EvaluationProtocolError("evaluation merge does not cover the full sealed census")
    ordered_shards = tuple(by_index[index] for index in range(len(plan.shards)))
    access_receipts = {
        None
        if shard.target_access_receipt is None
        else str(shard.target_access_receipt["record_digest"])
        for shard in ordered_shards
    }
    if len(access_receipts) != 1:
        raise EvaluationProtocolError("evaluation merge mixes target-access authorities")
    target_access = ordered_shards[0].target_access_receipt
    ground_authorities = {
        None
        if shard.ground_certificate_authority is None
        else _stable(shard.ground_certificate_authority)
        for shard in ordered_shards
    }
    if len(ground_authorities) != 1:
        raise EvaluationProtocolError("evaluation merge mixes ground-certificate authorities")
    ground_authority = ordered_shards[0].ground_certificate_authority
    compute_provenance = tuple(
        shard.compute_provenance for shard in ordered_shards if shard.compute_provenance is not None
    )
    if plan.compute_class is not None and len(compute_provenance) != len(ordered_shards):
        raise EvaluationProtocolError("evaluation merge is missing compute provenance")
    receipt_rows = tuple(
        sorted(
            (row for shard in ordered_shards for row in shard.receipt_rows),
            key=_pair_key,
        )
    )
    outcome_rows = tuple(
        sorted(
            (row for shard in ordered_shards for row in shard.outcome_rows),
            key=_pair_key,
        )
    )
    evidence_rows = tuple(
        sorted(
            (row for shard in ordered_shards for row in shard.evidence_rows),
            key=_pair_key,
        )
    )
    receipts = tuple(
        sorted(
            (item for shard in ordered_shards for item in shard.receipts),
            key=lambda item: item.pair_key,
        )
    )
    outcomes = tuple(
        sorted(
            (item for shard in ordered_shards for item in shard.outcomes),
            key=lambda item: item.pair_key,
        )
    )
    evidence = tuple(
        sorted(
            (item for shard in ordered_shards for item in shard.evidence),
            key=lambda item: item.pair_key,
        )
    )
    if plan.workflow is EvaluationWorkflow.LEARNED_COMPLETE:
        report_inputs: Mapping[str, object] = {
            "metrics": complete_system_metrics(receipts)  # type: ignore[arg-type]
        }
    elif plan.workflow is EvaluationWorkflow.TUNED_STOCK_COMPLETE:
        report_inputs = {
            "summary": external_complete_summary(
                outcomes,
                receipts,  # type: ignore[arg-type]
            )
        }
    else:
        lineage_metrics = _sharded_external_tuning_lineage_metrics(receipts)
        report_inputs = {
            "lineage_metrics": lineage_metrics,
            "summary": ExternalTuningRun.summary_from_lineages(lineage_metrics),
        }
    frozen_inputs = _frozen_mapping(report_inputs, label="recomputed report inputs")
    result = MergedEvaluationArtifacts(
        workflow=plan.workflow,
        plan_record_digest=plan.record_digest,
        shard_receipt_sha256=tuple(shard.receipt_file_sha256 for shard in ordered_shards),
        target_access_receipt=target_access,
        ground_certificate_authority=ground_authority,
        compute_provenance=compute_provenance,
        receipt_rows=receipt_rows,
        outcome_rows=outcome_rows,
        evidence_rows=evidence_rows,
        receipts=receipts,
        outcomes=outcomes,
        evidence=evidence,
        recomputed_report_inputs=frozen_inputs,
    )
    if (tasks is None) != (context is None):
        raise ValueError("tasks and context must be supplied together for evidence replay")
    if tasks is not None and context is not None:
        verify_merged_terminal_evidence(plan, result, tasks=tasks, context=context)
    return result


def verify_merged_terminal_evidence(
    plan: EvaluationExecutionPlan,
    merged: MergedEvaluationArtifacts,
    *,
    tasks: Sequence[EmbeddingTask],
    context: Context,
) -> None:
    """Recompile every valid merged terminal against the authenticated full task payload."""

    identities = tuple(sorted((task.lineage or task.name, task.name) for task in tasks))
    if identities != _population_instances(plan.population) or task_population_digest(
        tasks
    ) != plan.population.get("task_payload_sha256"):
        raise EvaluationProtocolError("terminal replay tasks differ from the sealed population")
    if merged.workflow is not plan.workflow or merged.plan_record_digest != plan.record_digest:
        raise EvaluationProtocolError("terminal replay merge differs from its execution plan")
    task_by_identity = {(task.lineage or task.name, task.name): task for task in tasks}
    receipt_by_key = {receipt.pair_key: receipt for receipt in merged.receipts}
    evidence_by_key = {record.pair_key: record for record in merged.evidence}
    if (
        len(receipt_by_key) != len(merged.receipts)
        or len(evidence_by_key) != len(merged.evidence)
        or set(receipt_by_key) != set(evidence_by_key)
    ):
        raise EvaluationProtocolError("terminal replay evidence census differs from receipts")
    for key, receipt in receipt_by_key.items():
        evidence = evidence_by_key[key]
        terminal = evidence.terminal_evidence
        if receipt.outcome.returned_valid != (terminal is not None):
            raise EvaluationProtocolError("terminal replay evidence presence differs from outcome")
        if terminal is not None:
            verify_terminal_evidence(
                terminal,
                task=task_by_identity[(key[0], key[1])],
                context=context,
                outcome=receipt.outcome,
            )


def publish_merged_evaluation(
    destination: str | os.PathLike[str],
    *,
    plan: EvaluationExecutionPlan,
    merged: MergedEvaluationArtifacts,
) -> tuple[Mapping[str, object], str]:
    """Atomically publish canonical unsharded raw inputs and a merge receipt."""

    if (
        merged.workflow is not plan.workflow
        or merged.plan_record_digest != plan.record_digest
        or len(merged.receipts) != len(plan.expected_pair_keys)
    ):
        raise EvaluationProtocolError("merged evaluation differs from its execution plan")
    artifacts = {
        "receipts": ("receipts.jsonl", merged.receipts_bytes),
        "outcomes": ("outcomes.jsonl", merged.outcomes_bytes),
        "terminal_evidence": ("terminal_evidence.jsonl", merged.evidence_bytes),
    }
    registry = {
        name: {
            "path": path,
            "sha256": hashlib.sha256(content).hexdigest(),
            "count": len(merged.receipts),
        }
        for name, (path, content) in artifacts.items()
    }
    payload: dict[str, object] = {
        "schema": EVALUATION_MERGE_RECEIPT_SCHEMA,
        "schema_version": EVALUATION_MERGE_RECEIPT_VERSION,
        "workflow": plan.workflow.value,
        "plan_record_digest": plan.record_digest,
        "population_digest": plan.population_digest,
        "shard_receipt_sha256": list(merged.shard_receipt_sha256),
        "target_access_receipt": (
            None if merged.target_access_receipt is None else dict(merged.target_access_receipt)
        ),
        "ground_certificate_authority": (
            None
            if merged.ground_certificate_authority is None
            else dict(merged.ground_certificate_authority)
        ),
        "quality_authority_digest": plan.quality_authority_digest,
        "compute_class": (None if plan.compute_class is None else dict(plan.compute_class)),
        "compute_class_digest": plan.compute_class_digest,
        "compute_provenance": [dict(item) for item in merged.compute_provenance],
        "publication_eligible": plan.publication_eligible,
        "pair_key_census_digest": stable_digest([list(key) for key in plan.expected_pair_keys]),
        "artifacts": registry,
        "recomputed_report_inputs": dict(merged.recomputed_report_inputs),
        "recomputed_report_inputs_digest": _stable(merged.recomputed_report_inputs),
    }
    payload["record_digest"] = stable_digest(payload)
    receipt_content = canonical_json_bytes(payload) + b"\n"
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        raise FileExistsError(destination_path)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination_path.name}.", dir=destination_path.parent)
    )
    try:
        for _name, (path, content) in artifacts.items():
            _atomic_write_new(temporary / path, content)
        _atomic_write_new(temporary / "merge_receipt.json", receipt_content)
        os.rename(temporary, destination_path)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return MappingProxyType(payload), hashlib.sha256(receipt_content).hexdigest()


__all__ = [
    "DEFAULT_MAX_LINEAGES_PER_SHARD",
    "EVALUATION_MERGE_RECEIPT_SCHEMA",
    "EVALUATION_PLAN_SCHEMA",
    "EVALUATION_SHARD_RECEIPT_SCHEMA",
    "EvaluationExecutionPlan",
    "EvaluationShardSpec",
    "EvaluationWorkflow",
    "LoadedEvaluationShard",
    "MergedEvaluationArtifacts",
    "build_evaluation_plan",
    "external_execution_contract",
    "identity_seed",
    "learned_execution_contract",
    "load_evaluation_plan",
    "load_evaluation_shard",
    "merge_evaluation_shards",
    "publish_evaluation_shard",
    "publish_merged_evaluation",
    "receipt_execution_contract",
    "select_tasks_for_shard",
    "validate_evaluation_compute_class",
    "verify_merged_terminal_evidence",
    "write_evaluation_plan",
]
