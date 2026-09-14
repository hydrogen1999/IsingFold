"""Authenticated offline labels for the independent IF-Q3-S0 strength selector.

The caller opens explicit certified evaluator partitions through the prepared-v4 and ground-root
trust boundary, then passes the authenticated tasks and their typed quality-authority binding to
this builder.  The builder materialises a frozen outcome-blind mixture of terminal embeddings,
compiles every retained terminal at all four strengths, and obtains one fresh count-complete read
block per program.  The source mixture never observes evaluator outcomes.  Its published graph
payload is the exact deployable selector input; reference energies and evaluator secrets never
enter that payload and are not returned by the loader.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import math
import os
import random
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from isingfold.rl.contracts import (
    Context,
    DecisionState,
    InitFailureRecord,
    Opcode,
    TerminalRecord,
    chain_key,
)
from isingfold.rl.data.import_embedbench import (
    CERTIFIED_REFERENCE_STATUSES,
    canonical_json_bytes,
    content_digest,
)
from isingfold.rl.data.prepared import PreparedTask
from isingfold.rl.evaluator import ReadBlock, sample_program
from isingfold.rl.env import EmbeddingEnv, fixed_strength_selector
from isingfold.rl.program import compile_registry, program_features, strength_registry
from isingfold.rl.proposal import LEGACY_ONLINE_INITIALIZER_RESTARTS_V1
from isingfold.rl.strength import (
    FEATURE_ORDER,
    N_STRENGTHS,
    StrengthGraphInput,
    StrengthRecord,
)
from isingfold.rl.strength_tensorize import build_strength_inputs

SCHEMA_VERSION = 4
RECORD_SCHEMA_VERSION = 2
NORMALIZER_SCHEMA_VERSION = 1
DATASET_SCHEMA = "isingfold.if-q3-s0-selector-labels"
RECORD_SCHEMA = "isingfold.if-q3-s0-selector-record"
NORMALIZER_SCHEMA = "isingfold.if-q3-s0-selector-normalizer"
DATASET_VERSION = "if-q3-s0-selector-labels-v4"
SEED_DOMAIN = "isingfold.if-q3-s0.selector-label-sampling.v2"
SPLIT_DOMAIN = "isingfold.if-q3-s0.selector-calibration-split.v1"
TERMINAL_SOURCE_SCHEMA = "isingfold.if-q3-s0-terminal-program-source"
TERMINAL_SOURCE_SCHEMA_VERSION = 1
TERMINAL_MIXTURE_SCHEMA = "isingfold.if-q3-s0-terminal-program-mixture"
TERMINAL_MIXTURE_SCHEMA_VERSION = 1
TERMINAL_MIXTURE_VERSION = "outcome-blind-uniform-rewrite-budget-mixture-v1"
TRAJECTORY_SEED_DOMAIN = "isingfold.if-q3-s0.terminal-source-trajectory.v1"
QUALITY_AUTHORITY_SCHEMA = "isingfold.quality-authority-binding"
QUALITY_AUTHORITY_SCHEMA_VERSION = 2
GLOBAL_QUALITY_AUTHORITY_SCHEMA = "isingfold.global-quality-authority"
PARTITION_QUALITY_AUTHORITY_SCHEMA = "isingfold.partition-quality-authority"

# This is a semantic registry, not a tunable hyperparameter.  Changing a slot, budget,
# action rule, or archive rule changes the population of terminal programs and therefore
# requires a new dataset version.
_TERMINAL_SOURCE_SPECS: tuple[tuple[str, int], ...] = (
    ("registered-initializer", 0),
    ("uniform-legal-rewrite", 1),
    ("uniform-legal-rewrite", 2),
    ("uniform-legal-rewrite", 4),
)

_FILES = frozenset({"records.jsonl", "normalizer.json", "manifest.json"})
_SELECTOR_PARTITIONS = frozenset({"train", "calibration", "audit_val", "audit_test"})
_AUDIT_PARTITIONS = frozenset({"audit_val", "audit_test"})
_HEX = frozenset("0123456789abcdef")

Evaluator = Callable[..., ReadBlock]


@dataclasses.dataclass(frozen=True)
class SelectorLabelMetadata:
    """Authenticated non-neural metadata needed to fit and publish a selector."""

    dataset_version: str
    manifest_sha256: str
    manifest_record_digest: str
    source_prepared_manifest_sha256: str
    quality_authority: Mapping[str, object]
    context_digest: str
    normalizer_digest: str
    coefficient_scale: float
    reads_per_strength: int
    partitions: Mapping[str, Mapping[str, object]]
    terminal_program_mixture: Mapping[str, object]


@dataclasses.dataclass(frozen=True)
class SelectorAuditAuthorization:
    """Capability issued only after the final RL-value decision is authenticated."""

    grid_manifest_sha256: str
    rl_value_selection_receipt_sha256: str
    rl_value_selection_record_digest: str
    representation_selection_receipt_sha256: str
    representation_selection_record_digest: str
    selected_model_family: str
    selected_method: str

    def validate(self) -> SelectorAuditAuthorization:
        for value, name in (
            (self.grid_manifest_sha256, "grid manifest digest"),
            (
                self.rl_value_selection_receipt_sha256,
                "RL-value-selection receipt digest",
            ),
            (
                self.rl_value_selection_record_digest,
                "RL-value-selection record digest",
            ),
            (
                self.representation_selection_receipt_sha256,
                "representation-selection receipt digest",
            ),
            (
                self.representation_selection_record_digest,
                "representation-selection record digest",
            ),
        ):
            if not _is_sha256(value):
                raise ValueError(f"selector audit authorization has an invalid {name}")
        if self.selected_model_family not in {"if-mlp", "if-dual", "if-core"}:
            raise ValueError("selector audit authorization has no registered model family")
        if self.selected_method not in {
            "supervised-only",
            "ppo-warm-start",
            "ppo-from-scratch",
        }:
            raise ValueError("selector audit authorization has no registered training method")
        return self


def _strict_object(raw: bytes, name: str) -> dict[str, Any]:
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
    except json.JSONDecodeError as error:
        raise ValueError(f"{name} is invalid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    # This recursively rejects overflowing JSON numbers and unsupported values.
    canonical_json_bytes(value)
    return value


def _read_json(path: Path, name: str) -> dict[str, Any]:
    return _strict_object(path.read_bytes(), name)


def _exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(
            f"{name} schema differs: missing={sorted(expected - set(value))}, "
            f"unknown={sorted(set(value) - expected)}"
        )


def _with_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**payload, "record_digest": content_digest(payload)}


def _verify_record(value: Mapping[str, Any], name: str) -> None:
    digest = value.get("record_digest")
    if not _is_sha256(digest):
        raise ValueError(f"{name} has an invalid record digest")
    payload = {key: item for key, item in value.items() if key != "record_digest"}
    if digest != content_digest(payload):
        raise ValueError(f"{name} record digest mismatch")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _authority_record(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    try:
        record = _strict_object(canonical_json_bytes(value), name)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} is not canonical JSON") from error
    _verify_record(record, name)
    return record


def _validate_global_quality_authority(value: object) -> dict[str, Any]:
    record = _authority_record(value, "selector global quality authority")
    _exact_keys(
        record,
        {
            "ground_root",
            "publication_id",
            "publisher_attestation_record_digest",
            "publisher_id",
            "record_digest",
            "schema",
            "schema_version",
            "target_authority_record_digest",
        },
        "selector global quality authority",
    )
    if (
        record["schema"] != GLOBAL_QUALITY_AUTHORITY_SCHEMA
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
    ):
        raise ValueError("selector global quality authority has an unsupported schema")
    if any(
        not isinstance(record[name], str) or not record[name]
        for name in ("publication_id", "publisher_id")
    ):
        raise ValueError("selector global quality authority IDs must be nonempty")
    if any(
        not _is_sha256(record[name])
        for name in (
            "publisher_attestation_record_digest",
            "target_authority_record_digest",
            "record_digest",
        )
    ):
        raise ValueError("selector global quality authority has an invalid digest")
    ground_root = record["ground_root"]
    if not isinstance(ground_root, dict):
        raise ValueError("selector global quality authority ground root must be an object")
    _exact_keys(
        ground_root,
        {"receipt_sha256", "record_digest", "verifier_identity_digest"},
        "selector global quality authority ground root",
    )
    if any(not _is_sha256(value) for value in ground_root.values()):
        raise ValueError("selector global quality authority ground root has an invalid digest")
    return record


def _validate_partition_quality_authority(
    value: object,
    *,
    expected_partition: str,
) -> dict[str, Any]:
    record = _authority_record(
        value,
        f"selector {expected_partition} partition quality authority",
    )
    _exact_keys(
        record,
        {
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
        },
        f"selector {expected_partition} partition quality authority",
    )
    if (
        record["schema"] != PARTITION_QUALITY_AUTHORITY_SCHEMA
        or type(record["schema_version"]) is not int
        or record["schema_version"] != 1
        or record["name"] != expected_partition
    ):
        raise ValueError(
            f"selector quality authority does not select {expected_partition!r}"
        )
    if any(
        not _is_sha256(record[name])
        for name in (
            "evidence_manifest_record_digest",
            "evidence_manifest_sha256",
            "record_digest",
            "target_access_record_digest",
            "target_set_digest",
        )
    ):
        raise ValueError("selector partition quality authority has an invalid digest")
    target_count = _positive_int(
        record["target_count"],
        f"selector {expected_partition} target count",
    )
    ground_partition = record["ground_partition"]
    if not isinstance(ground_partition, dict):
        raise ValueError("selector partition quality authority ground receipt must be an object")
    _exact_keys(
        ground_partition,
        {
            "accepted_count",
            "instance_set_digest",
            "receipt_record_digest",
            "receipt_sha256",
        },
        "selector partition quality authority ground receipt",
    )
    if any(
        not _is_sha256(ground_partition[name])
        for name in ("instance_set_digest", "receipt_record_digest", "receipt_sha256")
    ):
        raise ValueError("selector partition ground receipt has an invalid digest")
    accepted_count = _positive_int(
        ground_partition["accepted_count"],
        f"selector {expected_partition} accepted target count",
    )
    if accepted_count != target_count:
        raise ValueError("selector partition authority does not certify every target")
    return record


def validate_selector_quality_authority(
    value: Mapping[str, object],
    *,
    audit_mode: bool = False,
) -> dict[str, Any]:
    """Validate the exact v2 authority projection opened by the caller.

    Ordinary label construction accepts only the global identity and the opened train
    partition.  Post-freeze audit construction additionally requires explicit val and test
    projections.  This function never opens a corpus or infers authority from task fields.
    """

    if not isinstance(audit_mode, bool):
        raise ValueError("audit_mode must be Boolean")
    record = _authority_record(value, "selector quality-authority binding")
    expected = {
        "global",
        "record_digest",
        "schema",
        "schema_version",
        "training_partition",
    }
    if audit_mode:
        expected.add("audit_partitions")
    _exact_keys(record, expected, "selector quality-authority binding")
    if (
        record["schema"] != QUALITY_AUTHORITY_SCHEMA
        or type(record["schema_version"]) is not int
        or record["schema_version"] != QUALITY_AUTHORITY_SCHEMA_VERSION
    ):
        raise ValueError("selector quality-authority binding has an unsupported schema")
    _validate_global_quality_authority(record["global"])
    _validate_partition_quality_authority(
        record["training_partition"],
        expected_partition="train",
    )
    if audit_mode:
        audit_partitions = record["audit_partitions"]
        if not isinstance(audit_partitions, dict):
            raise ValueError("selector audit partition authorities must be an object")
        _exact_keys(
            audit_partitions,
            {"test", "val"},
            "selector audit partition authorities",
        )
        for partition in ("val", "test"):
            _validate_partition_quality_authority(
                audit_partitions[partition],
                expected_partition=partition,
            )
    return record


def _validated_selector_inputs(
    prepared_tasks: Sequence[PreparedTask],
    quality_authority: Mapping[str, object],
    *,
    audit_mode: bool,
) -> tuple[tuple[PreparedTask, ...], dict[str, Any]]:
    if isinstance(prepared_tasks, (str, bytes, bytearray)):
        raise TypeError("prepared_tasks must be a sequence of PreparedTask values")
    tasks = tuple(prepared_tasks)
    if not tasks or any(not isinstance(item, PreparedTask) for item in tasks):
        raise ValueError("selector labels require nonempty authenticated prepared tasks")
    if len({item.task_id for item in tasks}) != len(tasks):
        raise ValueError("selector prepared tasks repeat a task ID")
    expected_partitions = {"train", "val", "test"} if audit_mode else {"train"}
    observed_partitions = {item.partition for item in tasks}
    if observed_partitions != expected_partitions:
        if not audit_mode and observed_partitions - {"train"}:
            raise PermissionError(
                "ordinary selector-label construction accepts the train partition only"
            )
        raise ValueError(
            "selector audit construction requires explicit train, val, and test task sets"
        )
    authority = validate_selector_quality_authority(
        quality_authority,
        audit_mode=audit_mode,
    )
    projected = {"train": authority["training_partition"]}
    if audit_mode:
        projected.update(authority["audit_partitions"])
    global_authority = authority["global"]
    for partition in sorted(expected_partitions):
        partition_tasks = tuple(item for item in tasks if item.partition == partition)
        partition_authority = projected[partition]
        observed_count = len({item.instance_id for item in partition_tasks})
        if partition_authority["target_count"] != observed_count:
            raise ValueError(
                f"selector {partition} instance count differs from its partition authority"
            )
        for item in partition_tasks:
            if item.prepared_schema_version != 4:
                raise ValueError("selector target task is not authenticated prepared-v4 data")
            expected_fields = {
                "quality_attestation_digest": global_authority[
                    "publisher_attestation_record_digest"
                ],
                "quality_evidence_manifest_digest": partition_authority[
                    "evidence_manifest_record_digest"
                ],
                "quality_evidence_manifest_sha256": partition_authority[
                    "evidence_manifest_sha256"
                ],
                "quality_target_set_digest": partition_authority["target_set_digest"],
                "quality_target_count": partition_authority["target_count"],
            }
            if any(getattr(item, name) != expected for name, expected in expected_fields.items()):
                raise ValueError(
                    f"selector {partition} task differs from its partition authority"
                )
    return tasks, authority


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"cannot encode {type(value).__name__} in selector metadata")


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _selector_partition_assignment(
    tasks: Sequence[PreparedTask],
    *,
    split_seed: int,
    calibration_fraction: float,
    audit_mode: bool,
) -> dict[str, str]:
    if not math.isfinite(calibration_fraction) or not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be finite and strictly between zero and one")
    by_lineage: dict[str, str] = {}
    for item in tasks:
        if item.partition not in {"train", "val", "test"}:
            raise ValueError(
                f"prepared partition {item.partition!r} is unsupported; "
                "sealed OOD data requires a separately authenticated audit corpus"
            )
        lineage = item.task.lineage
        if not isinstance(lineage, str) or not lineage:
            raise ValueError("every selector task must have a non-empty logical lineage")
        previous = by_lineage.setdefault(lineage, item.partition)
        if previous != item.partition:
            raise ValueError("one logical lineage appears in multiple prepared partitions")

    train_lineages = sorted(lineage for lineage, part in by_lineage.items() if part == "train")
    if len(train_lineages) < 2:
        raise ValueError("selector fitting requires at least two prepared-train lineages")
    ordered = sorted(
        train_lineages,
        key=lambda lineage: (
            content_digest(
                {
                    "domain": SPLIT_DOMAIN,
                    "lineage": lineage,
                    "split_seed": split_seed,
                }
            ),
            lineage,
        ),
    )
    calibration_count = max(1, math.ceil(calibration_fraction * len(ordered)))
    calibration_count = min(calibration_count, len(ordered) - 1)
    calibration = set(ordered[-calibration_count:])
    assignment = {
        lineage: "calibration" if lineage in calibration else "train"
        for lineage in train_lineages
    }
    if audit_mode:
        assignment.update(
            {
                lineage: "audit_val" if part == "val" else "audit_test"
                for lineage, part in by_lineage.items()
                if part in {"val", "test"}
            }
        )
    return assignment


def _normalizer_payload(train_tasks: Sequence[PreparedTask]) -> dict[str, Any]:
    coefficients = [
        abs(float(value))
        for item in train_tasks
        for value in (*item.task.problem.h.values(), *item.task.problem.j.values())
        if float(value) != 0.0
    ]
    coefficient_scale = (
        math.sqrt(float(np.mean(np.square(coefficients)))) if coefficients else 1.0
    )
    coefficient_scale = max(1e-12, coefficient_scale)
    payload = {
        "coefficient_scale": coefficient_scale,
        "coefficient_scale_fit": "rms-absolute-nonzero-logical-coefficients",
        "coefficient_units": "logical-and-compiled-ising",
        "count_transform": "log1p",
        "fit_lineages": sorted({item.task.lineage for item in train_tasks}),
        "fit_partition": "selector-train-only",
        "knownness": "value-then-binary-knownness",
        "schema": NORMALIZER_SCHEMA,
        "schema_version": NORMALIZER_SCHEMA_VERSION,
        "signed_count_transform": "signed-log1p",
    }
    canonical_json_bytes(payload)
    return payload


def _seed_from_identity(
    root_seed: int,
    program_id: str,
    terminal_source_record_digest: str,
    strength_index: int,
    used: set[int],
) -> int:
    nonce = 0
    while True:
        digest = hashlib.sha256(
            canonical_json_bytes(
                {
                    "domain": SEED_DOMAIN,
                    "nonce": nonce,
                    "program_id": program_id,
                    "root_seed": root_seed,
                    "strength_index": strength_index,
                    "terminal_source_record_digest": terminal_source_record_digest,
                }
            )
        ).digest()
        candidate = int.from_bytes(digest[:8], "big") & (2**31 - 1)
        if candidate not in used:
            used.add(candidate)
            return candidate
        nonce += 1


def _seed(
    root_seed: int,
    program_id: str,
    terminal_source_record_digest: str,
    strength_index: int,
    used: set[int],
) -> int:
    return _seed_from_identity(
        root_seed,
        program_id,
        terminal_source_record_digest,
        strength_index,
        used,
    )


def terminal_program_mixture_registry() -> dict[str, object]:
    return {
        "schema": TERMINAL_MIXTURE_SCHEMA,
        "schema_version": TERMINAL_MIXTURE_SCHEMA_VERSION,
        "mixture_version": TERMINAL_MIXTURE_VERSION,
        "outcome_access": "forbidden-before-source-freeze",
        "state_changing_action_rule": "uniform-over-exact-legal-state-changing-support",
        "terminal_action_rule": "commit-current-valid-else-canonical-first-archive",
        "deduplication_rule": "terminal-embedding-chain-key-per-base-task",
        "strength_selector_during_terminalisation": "fixed-f2-index-1-no-evaluation",
        "sources": [
            {
                "source_slot": slot,
                "source_kind": source_kind,
                "rewrite_budget": rewrite_budget,
            }
            for slot, (source_kind, rewrite_budget) in enumerate(_TERMINAL_SOURCE_SPECS)
        ],
    }


def _trajectory_seed_from_identity(
    root_seed: int,
    task_id: str,
    initializer_record_digest: str,
    source_slot: int,
) -> int:
    digest = hashlib.sha256(
        canonical_json_bytes(
            {
                "domain": TRAJECTORY_SEED_DOMAIN,
                "initializer_record_digest": initializer_record_digest,
                "root_seed": root_seed,
                "source_slot": source_slot,
                "task_id": task_id,
            }
        )
    ).digest()
    return int.from_bytes(digest[:8], "big") & (2**63 - 1)


def _trajectory_seed(root_seed: int, task: PreparedTask, source_slot: int) -> int:
    return _trajectory_seed_from_identity(
        root_seed,
        task.task_id,
        task.initializer_record_digest,
        source_slot,
    )


def _typed_identity(value: object) -> dict[str, str]:
    kind = type(value)
    return {
        "type": f"{kind.__module__}.{kind.__qualname__}",
        "repr": repr(value),
    }


def _encoded_chains(chains: Mapping[object, Sequence[object]]) -> list[dict[str, object]]:
    def key(value: object) -> tuple[str, str]:
        identity = _typed_identity(value)
        return identity["type"], identity["repr"]

    return [
        {
            "logical": _typed_identity(node),
            "chain": [_typed_identity(qubit) for qubit in sorted(chains[node], key=key)],
        }
        for node in sorted(chains, key=key)
    ]


def _trace_action(
    decision: DecisionState,
    index: int,
    *,
    position: int,
    source: str,
) -> dict[str, object]:
    candidate = decision.candidates[index]
    return {
        "position": position,
        "source": source,
        "action_index": index,
        "opcode": candidate.opcode.value,
        "payload_key": candidate.payload_key,
        "state_fingerprint": decision.state_fingerprint,
        "support_fingerprint": decision.support_fingerprint,
    }


def _commit_index(decision: DecisionState) -> int | None:
    current_key = chain_key(decision.exact_state.chains)
    current_payload = f"COMMIT:{current_key}"
    legal = [
        index
        for index, (candidate, allowed) in enumerate(
            zip(decision.candidates, decision.legal_mask, strict=True)
        )
        if allowed and candidate.opcode is Opcode.COMMIT
    ]
    for index in legal:
        if decision.candidates[index].payload_key == current_payload:
            return index
    if not legal:
        return None
    return min(legal, key=lambda index: (decision.candidates[index].payload_key, index))


@dataclasses.dataclass(frozen=True)
class _TerminalSource:
    chains: Mapping[object, frozenset[object]]
    receipt_payload: Mapping[str, object]


def _terminal_source(
    item: PreparedTask,
    *,
    context: Context,
    coefficient_scale: float,
    root_seed: int,
    source_slot: int,
    source_kind: str,
    rewrite_budget: int,
) -> _TerminalSource | None:
    """Materialise one registered terminal without ever requesting an evaluator outcome."""

    trajectory_seed = _trajectory_seed(root_seed, item, source_slot)
    env = EmbeddingEnv(
        item.task,
        context,
        initializer=item.initializer(),
        selector=fixed_strength_selector(1),
        reward_reads=context.n_est_reads,
        coefficient_scale=coefficient_scale,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=trajectory_seed,
    )
    current = env.reset(trajectory_seed)
    if isinstance(current, (InitFailureRecord, TerminalRecord)):
        return None
    rng = random.Random(trajectory_seed)
    trace: list[dict[str, object]] = []
    state_changes = 0
    while isinstance(current, DecisionState) and state_changes < rewrite_budget:
        legal = [
            index
            for index, (candidate, allowed) in enumerate(
                zip(current.candidates, current.legal_mask, strict=True)
            )
            if allowed and candidate.changes_workspace
        ]
        if not legal:
            break
        chosen = rng.choice(legal)
        trace.append(
            _trace_action(
                current,
                chosen,
                position=len(trace),
                source="uniform-legal-state-changing",
            )
        )
        step = env.step(current, chosen, evaluate_training_reward=False)
        current = step.next_decision_or_terminal
        state_changes += 1

    if isinstance(current, DecisionState):
        commit = _commit_index(current)
        if commit is None:
            return None
        trace.append(
            _trace_action(
                current,
                commit,
                position=len(trace),
                source="outcome-blind-terminal-commit",
            )
        )
        current = env.step(current, commit, evaluate_training_reward=False).next_decision_or_terminal
    if not isinstance(current, TerminalRecord) or not current.returned_valid:
        return None
    if current.embedding is None or current.training_reward is not None or current.evaluator_counts:
        raise RuntimeError("outcome-blind terminal source accessed evaluator outcomes")
    frozen = {node: frozenset(chain) for node, chain in current.embedding.items()}
    encoded = _encoded_chains(frozen)
    receipt_payload: dict[str, object] = {
        "schema": TERMINAL_SOURCE_SCHEMA,
        "schema_version": TERMINAL_SOURCE_SCHEMA_VERSION,
        "mixture_version": TERMINAL_MIXTURE_VERSION,
        "source_slot": source_slot,
        "source_kind": source_kind,
        "rewrite_budget": rewrite_budget,
        "realized_state_changes": state_changes,
        "trajectory_seed": trajectory_seed,
        "trajectory_seed_domain": TRAJECTORY_SEED_DOMAIN,
        "outcome_access": "none",
        "base_task_id": item.task_id,
        "initializer_record_digest": item.initializer_record_digest,
        "action_trace": trace,
        "terminal_reason": current.terminal_reason.value,
        "returned_valid": True,
        "terminal_embedding": encoded,
        "terminal_embedding_digest": content_digest(encoded),
        "terminal_chain_key": chain_key(frozen),
        "validation_receipt": _jsonable(current.validation_receipt),
        "cumulative_work": current.cumulative_work.as_dict(),
    }
    canonical_json_bytes(receipt_payload)
    return _TerminalSource(chains=frozen, receipt_payload=receipt_payload)


def _tensor_payload(value: object, *, integer: bool) -> dict[str, Any]:
    dtype = np.dtype("<i8" if integer else "<f4")
    array = np.ascontiguousarray(np.asarray(value, dtype=dtype))
    if not integer and not np.isfinite(array).all():
        raise ValueError("selector tensor contains non-finite values")
    return {
        "data_base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        "dtype": "int64-le" if integer else "float32-le",
        "shape": list(array.shape),
    }


def _graph_payload(graph: StrengthGraphInput) -> dict[str, Any]:
    graph.validate()
    return {
        "claims": _tensor_payload(graph.claims, integer=False),
        "globals": _tensor_payload(graph.globals_, integer=False),
        "hardware": _tensor_payload(graph.hardware, integer=False),
        "hardware_edges": _tensor_payload(graph.hardware_edges, integer=False),
        "index_claims": _tensor_payload(graph.index_claims, integer=True),
        "index_hardware_edges": _tensor_payload(graph.index_hardware_edges, integer=True),
        "index_logical_edges": _tensor_payload(graph.index_logical_edges, integer=True),
        "logical": _tensor_payload(graph.logical, integer=False),
        "logical_edges": _tensor_payload(graph.logical_edges, integer=False),
        "normalizer_digest": graph.normalizer_digest,
        "scale": float(graph.scale),
        "strength": float(graph.strength),
        "strength_index": graph.strength_index,
    }


def _validate_read_block(block: object, *, reads: int, strength_index: int) -> ReadBlock:
    if not isinstance(block, ReadBlock):
        raise ValueError("selector evaluator returned no complete ReadBlock")
    if (
        isinstance(block.hits, bool)
        or not isinstance(block.hits, int)
        or isinstance(block.reads, bool)
        or not isinstance(block.reads, int)
    ):
        raise ValueError("selector evaluator counts must be integers")
    if block.reads != reads or not 0 <= block.hits <= block.reads:
        raise ValueError("selector evaluator returned an incomplete or invalid count block")
    if block.strength_index != strength_index:
        raise ValueError("selector evaluator returned a count for the wrong strength")
    if not math.isfinite(block.broken_fraction) or not 0.0 <= block.broken_fraction <= 1.0:
        raise ValueError("selector evaluator returned an invalid broken-chain fraction")
    if not math.isfinite(block.mean_residual):
        raise ValueError("selector evaluator returned a non-finite residual")
    return block


def _write_and_sync(path: Path, raw: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def build_selector_labels(
    prepared_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    prepared_tasks: Sequence[PreparedTask],
    quality_authority: Mapping[str, object],
    context: Context,
    sample_seed: int,
    split_seed: int = 0,
    calibration_fraction: float = 0.2,
    audit_mode: bool = False,
    audit_authorization: SelectorAuditAuthorization | None = None,
    expected_normalizer_digest: str | None = None,
    evaluator: Evaluator | None = None,
    evaluator_protocol: str = "sa-majority-fresh-count-block-v1",
) -> dict[str, Any]:
    """Build and atomically publish count-complete selector records.

    The caller must explicitly open authenticated prepared-v4 partitions and pass their exact
    quality-authority projection.  Only train may be passed for fitting and calibration.
    Validation and test are accepted only in post-freeze audit mode, where all three partition
    authorities must be bound into the supplied capability.
    """

    root_seed = _nonnegative_int(sample_seed, "sample_seed")
    split_root = _nonnegative_int(split_seed, "split_seed")
    if not isinstance(audit_mode, bool):
        raise ValueError("audit_mode must be Boolean")
    if audit_mode:
        if audit_authorization is None:
            raise PermissionError(
                "held-out selector-label generation requires a final RL-value freeze"
            )
        audit_authorization.validate()
    elif audit_authorization is not None:
        raise ValueError("selector audit authorization is valid only with audit_mode=True")
    if not isinstance(evaluator_protocol, str) or not evaluator_protocol:
        raise ValueError("evaluator_protocol must be a non-empty string")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"selector-label output already exists: {destination}")

    tasks, quality_authority = _validated_selector_inputs(
        prepared_tasks,
        quality_authority,
        audit_mode=audit_mode,
    )
    assignment = _selector_partition_assignment(
        tasks,
        split_seed=split_root,
        calibration_fraction=calibration_fraction,
        audit_mode=audit_mode,
    )
    selected = [item for item in tasks if item.task.lineage in assignment]
    train_tasks = [item for item in selected if assignment[item.task.lineage] == "train"]
    normalizer = _normalizer_payload(train_tasks)
    normalizer_digest = content_digest(normalizer)
    if (
        expected_normalizer_digest is not None
        and normalizer_digest != expected_normalizer_digest
    ):
        raise ValueError(
            "selector audit labels would use a different frozen normalizer; "
            "reuse the fitting split seed and calibration fraction"
        )
    coefficient_scale = float(normalizer["coefficient_scale"])
    reads = _positive_int(
        context.audit_reads if audit_mode else context.n_est_reads,
        "context.audit_reads" if audit_mode else "context.n_est_reads",
    )
    sampler = sample_program if evaluator is None else evaluator

    prepared_manifest_path = Path(prepared_dir) / "manifest.json"
    prepared_manifest = _read_json(prepared_manifest_path, "prepared manifest")
    _verify_record(prepared_manifest, "prepared manifest")
    if (
        prepared_manifest.get("schema") != "isingfold.prepared-candidate-bank"
        or prepared_manifest.get("schema_version") != 4
    ):
        raise ValueError("selector labels require a prepared CandidateBank v4 manifest")
    source_cap = prepared_manifest.get("qubit_cap")
    if (
        isinstance(source_cap, bool)
        or not isinstance(source_cap, int)
        or source_cap <= 0
        or source_cap != context.qubit_cap
    ):
        raise ValueError("selector context qubit cap differs from the prepared corpus")
    source_manifest_sha256 = hashlib.sha256(prepared_manifest_path.read_bytes()).hexdigest()

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.selector-labels-", dir=destination.parent)
    )
    used_seeds: set[int] = set()
    counts = {partition: 0 for partition in sorted(_SELECTOR_PARTITIONS)}
    lineages = {partition: set() for partition in sorted(_SELECTOR_PARTITIONS)}
    source_census: dict[str, dict[str, object]] = {
        str(slot): {
            "source_slot": slot,
            "source_kind": source_kind,
            "rewrite_budget": rewrite_budget,
            "attempted_tasks": 0,
            "terminalized_tasks": 0,
            "retained_records": 0,
            "duplicate_terminal_embeddings": 0,
            "failed_terminalizations": 0,
        }
        for slot, (source_kind, rewrite_budget) in enumerate(_TERMINAL_SOURCE_SPECS)
    }
    try:
        records_path = temporary / "records.jsonl"
        records_hasher = hashlib.sha256()
        record_count = 0
        with records_path.open("wb") as stream:
            ordered_tasks = sorted(
                selected,
                key=lambda item: (assignment[item.task.lineage], item.task_id),
            )
            for item in ordered_tasks:
                task = item.task
                if (
                    task.ground_energy is None
                    or not math.isfinite(float(task.ground_energy))
                    or item.reference_status not in CERTIFIED_REFERENCE_STATUSES
                    or not _is_sha256(item.certificate_digest)
                    or not _is_sha256(item.evaluator_protocol_digest)
                    or not _is_sha256(item.initializer_record_digest)
                ):
                    raise ValueError(f"task {item.task_id!r} has no complete certified target")
                partition = assignment[task.lineage]
                seen_terminal_embeddings: set[str] = set()
                retained_for_task = 0
                for source_slot, (source_kind, rewrite_budget) in enumerate(
                    _TERMINAL_SOURCE_SPECS
                ):
                    census = source_census[str(source_slot)]
                    census["attempted_tasks"] = int(census["attempted_tasks"]) + 1
                    source = _terminal_source(
                        item,
                        context=context,
                        coefficient_scale=coefficient_scale,
                        root_seed=root_seed,
                        source_slot=source_slot,
                        source_kind=source_kind,
                        rewrite_budget=rewrite_budget,
                    )
                    if source is None:
                        census["failed_terminalizations"] = (
                            int(census["failed_terminalizations"]) + 1
                        )
                        if source_slot == 0:
                            raise ValueError(
                                f"registered initializer failed to terminalize for task "
                                f"{item.task_id!r}"
                            )
                        continue
                    census["terminalized_tasks"] = int(census["terminalized_tasks"]) + 1
                    embedding_digest = str(
                        source.receipt_payload["terminal_embedding_digest"]
                    )
                    if embedding_digest in seen_terminal_embeddings:
                        census["duplicate_terminal_embeddings"] = (
                            int(census["duplicate_terminal_embeddings"]) + 1
                        )
                        continue
                    seen_terminal_embeddings.add(embedding_digest)
                    frozen_chains = {
                        node: frozenset(chain) for node, chain in source.chains.items()
                    }
                    strengths = strength_registry(
                        task.problem, context.strength_ratios, context.epsilon_strength
                    )
                    programs = compile_registry(
                        frozen_chains,
                        task.host,
                        task.problem,
                        strengths,
                        context.field_limit,
                        context.coupler_limit,
                    )
                    graph_inputs = build_strength_inputs(
                        ctx=context,
                        logical=task.logical,
                        host=task.host,
                        problem=task.problem,
                        chains=frozen_chains,
                        programs=programs,
                        coef_scale=coefficient_scale,
                        normalizer_digest=normalizer_digest,
                    )
                    features = tuple(
                        program_features(program, frozen_chains, task.problem)
                        for program in programs
                    )
                    feature_payload = [
                        {name: float(feature[name]) for name in FEATURE_ORDER}
                        for feature in features
                    ]
                    graph_payload = [_graph_payload(graph) for graph in graph_inputs]
                    program_registry_digest = content_digest(
                        {
                            "terminal_embedding_digest": embedding_digest,
                            "features": feature_payload,
                            "graph_inputs": graph_payload,
                        }
                    )
                    source_receipt = _with_digest(
                        {
                            **source.receipt_payload,
                            "program_registry_digest": program_registry_digest,
                        }
                    )
                    program_id = "selector-program-" + content_digest(
                        {
                            "base_task_id": item.task_id,
                            "source_receipt_record_digest": source_receipt["record_digest"],
                        }
                    )
                    hits: list[int] = []
                    totals: list[int] = []
                    sample_seeds: list[int] = []
                    for strength_index, program in enumerate(programs):
                        seed = _seed(
                            root_seed,
                            program_id,
                            str(source_receipt["record_digest"]),
                            strength_index,
                            used_seeds,
                        )
                        block = _validate_read_block(
                            sampler(
                                program,
                                frozen_chains,
                                task.problem,
                                float(task.ground_energy),
                                num_reads=reads,
                                seed=seed,
                                num_sweeps=context.num_sweeps,
                            ),
                            reads=reads,
                            strength_index=strength_index,
                        )
                        sample_seeds.append(seed)
                        hits.append(block.hits)
                        totals.append(block.reads)

                    record_without_id = {
                        "base_task_id": item.task_id,
                        "certificate_digest": item.certificate_digest,
                        "evaluator_protocol_digest": item.evaluator_protocol_digest,
                        "features": feature_payload,
                        "graph_inputs": graph_payload,
                        "hits": hits,
                        "initializer_record_digest": item.initializer_record_digest,
                        "instance_id": item.instance_id,
                        "lineage": task.lineage,
                        "program_id": program_id,
                        "program_registry_digest": program_registry_digest,
                        "reads": totals,
                        "reference_status": item.reference_status,
                        "sample_seeds": sample_seeds,
                        "schema": RECORD_SCHEMA,
                        "schema_version": RECORD_SCHEMA_VERSION,
                        "selector_partition": partition,
                        "source_partition": item.partition,
                        "terminal_source_receipt": source_receipt,
                        # StrengthRecord.task_id remains a unique sample identity.  The
                        # immutable source problem is retained separately as base_task_id.
                        "task_id": program_id,
                    }
                    record_id = "selector-record-" + content_digest(
                        {"domain": DATASET_VERSION, "record": record_without_id}
                    )
                    row = _with_digest({**record_without_id, "record_id": record_id})
                    raw = canonical_json_bytes(row) + b"\n"
                    stream.write(raw)
                    records_hasher.update(raw)
                    record_count += 1
                    retained_for_task += 1
                    census["retained_records"] = int(census["retained_records"]) + 1
                    counts[partition] += 1
                    lineages[partition].add(task.lineage)
                if retained_for_task == 0:
                    raise RuntimeError(
                        f"terminal source mixture retained no program for task {item.task_id!r}"
                    )
            stream.flush()
            os.fsync(stream.fileno())

        if sum(
            int(source_census[str(slot)]["retained_records"])
            for slot in range(1, len(_TERMINAL_SOURCE_SPECS))
        ) == 0:
            raise ValueError(
                "terminal source mixture produced no non-initializer program; "
                "the selector corpus cannot collapse to initializer-only labels"
            )

        normalizer_record = _with_digest(normalizer)
        normalizer_raw = canonical_json_bytes(normalizer_record) + b"\n"
        _write_and_sync(temporary / "normalizer.json", normalizer_raw)
        partition_receipts = {
            partition: {
                "lineages": sorted(lineages[partition]),
                "records": counts[partition],
            }
            for partition in sorted(_SELECTOR_PARTITIONS)
        }
        manifest_payload = {
            "audit_mode": audit_mode,
            "context": _jsonable(context),
            "context_digest": content_digest(_jsonable(context)),
            "dataset_version": DATASET_VERSION,
            "label_protocol": {
                "endpoint": context.endpoint,
                "evaluator": evaluator_protocol,
                "reads_per_strength": reads,
                "strength_ratios": list(context.strength_ratios),
                "sweeps": context.num_sweeps,
            },
            "normalizer_digest": normalizer_digest,
            "outputs": {
                "normalizer.json": {
                    "records": 1,
                    "sha256": hashlib.sha256(normalizer_raw).hexdigest(),
                },
                "records.jsonl": {
                    "records": record_count,
                    "sha256": records_hasher.hexdigest(),
                },
            },
            "partitions": partition_receipts,
            "sample_seed": root_seed,
            "schema": DATASET_SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "seed_domain": SEED_DOMAIN,
            "source_prepared_manifest_record_digest": prepared_manifest["record_digest"],
            "source_prepared_manifest_sha256": source_manifest_sha256,
            "quality_authority": quality_authority,
            "split_protocol": {
                "calibration_fraction": calibration_fraction,
                "domain": SPLIT_DOMAIN,
                "source": "prepared-train-lineages-only",
                "split_seed": split_root,
            },
            "terminal_program_mixture": terminal_program_mixture_registry(),
            "terminal_source_census": source_census,
        }
        manifest = _with_digest(manifest_payload)
        _write_and_sync(
            temporary / "manifest.json", canonical_json_bytes(manifest) + b"\n"
        )
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        except (AttributeError, OSError):
            pass
        else:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return manifest


def _decode_tensor(value: object, *, integer: bool, name: str) -> np.ndarray:
    if not isinstance(value, dict):
        raise ValueError(f"{name} tensor must be an object")
    _exact_keys(value, {"data_base64", "dtype", "shape"}, f"{name} tensor")
    expected_dtype = "int64-le" if integer else "float32-le"
    if value["dtype"] != expected_dtype:
        raise ValueError(f"{name} tensor dtype must be {expected_dtype}")
    shape = value["shape"]
    if (
        not isinstance(shape, list)
        or any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape)
    ):
        raise ValueError(f"{name} tensor has an invalid shape")
    encoded = value["data_base64"]
    if not isinstance(encoded, str):
        raise ValueError(f"{name} tensor data must be base64 text")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise ValueError(f"{name} tensor has invalid base64 data") from error
    dtype = np.dtype("<i8" if integer else "<f4")
    elements = math.prod(shape)
    if len(raw) != elements * dtype.itemsize:
        raise ValueError(f"{name} tensor byte count differs from its shape")
    array = np.frombuffer(raw, dtype=dtype).reshape(tuple(shape)).copy()
    if not integer and not np.isfinite(array).all():
        raise ValueError(f"{name} tensor contains non-finite values")
    return array


def _decode_graph(value: object, normalizer_digest: str) -> StrengthGraphInput:
    if not isinstance(value, dict):
        raise ValueError("selector graph input must be an object")
    _exact_keys(
        value,
        {
            "claims",
            "globals",
            "hardware",
            "hardware_edges",
            "index_claims",
            "index_hardware_edges",
            "index_logical_edges",
            "logical",
            "logical_edges",
            "normalizer_digest",
            "scale",
            "strength",
            "strength_index",
        },
        "selector graph input",
    )
    if value["normalizer_digest"] != normalizer_digest:
        raise ValueError("selector graph input uses a different normalizer")
    index = value["strength_index"]
    if isinstance(index, bool) or not isinstance(index, int):
        raise ValueError("selector graph strength index must be an integer")
    graph = StrengthGraphInput(
        logical=_decode_tensor(value["logical"], integer=False, name="logical"),
        hardware=_decode_tensor(value["hardware"], integer=False, name="hardware"),
        logical_edges=_decode_tensor(
            value["logical_edges"], integer=False, name="logical_edges"
        ),
        hardware_edges=_decode_tensor(
            value["hardware_edges"], integer=False, name="hardware_edges"
        ),
        claims=_decode_tensor(value["claims"], integer=False, name="claims"),
        globals_=_decode_tensor(value["globals"], integer=False, name="globals"),
        index_logical_edges=_decode_tensor(
            value["index_logical_edges"], integer=True, name="index_logical_edges"
        ),
        index_hardware_edges=_decode_tensor(
            value["index_hardware_edges"], integer=True, name="index_hardware_edges"
        ),
        index_claims=_decode_tensor(
            value["index_claims"], integer=True, name="index_claims"
        ),
        strength_index=index,
        strength=float(value["strength"]),
        scale=float(value["scale"]),
        normalizer_digest=normalizer_digest,
    )
    return graph.validate()


def _validate_terminal_source_receipt(
    receipt: object,
    *,
    root_seed: int,
    base_task_id: str,
    initializer_record_digest: str,
    features: object,
    graph_inputs: object,
) -> tuple[int, str]:
    if not isinstance(receipt, dict):
        raise ValueError("selector terminal source receipt must be an object")
    expected = {
        "action_trace",
        "base_task_id",
        "cumulative_work",
        "initializer_record_digest",
        "mixture_version",
        "outcome_access",
        "program_registry_digest",
        "realized_state_changes",
        "record_digest",
        "returned_valid",
        "rewrite_budget",
        "schema",
        "schema_version",
        "source_kind",
        "source_slot",
        "terminal_chain_key",
        "terminal_embedding",
        "terminal_embedding_digest",
        "terminal_reason",
        "trajectory_seed",
        "trajectory_seed_domain",
        "validation_receipt",
    }
    _exact_keys(receipt, expected, "selector terminal source receipt")
    _verify_record(receipt, "selector terminal source receipt")
    if (
        receipt["schema"] != TERMINAL_SOURCE_SCHEMA
        or receipt["schema_version"] != TERMINAL_SOURCE_SCHEMA_VERSION
        or receipt["mixture_version"] != TERMINAL_MIXTURE_VERSION
        or receipt["trajectory_seed_domain"] != TRAJECTORY_SEED_DOMAIN
        or receipt["outcome_access"] != "none"
        or receipt["returned_valid"] is not True
        or receipt["terminal_reason"] != "COMMIT"
        or receipt["base_task_id"] != base_task_id
        or receipt["initializer_record_digest"] != initializer_record_digest
    ):
        raise ValueError("selector terminal source receipt violates its frozen protocol")
    slot = receipt["source_slot"]
    budget = receipt["rewrite_budget"]
    realized = receipt["realized_state_changes"]
    if (
        type(slot) is not int
        or not 0 <= slot < len(_TERMINAL_SOURCE_SPECS)
        or type(budget) is not int
        or type(realized) is not int
        or not 0 <= realized <= budget
    ):
        raise ValueError("selector terminal source receipt has invalid budget fields")
    expected_kind, expected_budget = _TERMINAL_SOURCE_SPECS[slot]
    if receipt["source_kind"] != expected_kind or budget != expected_budget:
        raise ValueError("selector terminal source differs from the registered mixture slot")
    expected_trajectory_seed = _trajectory_seed_from_identity(
        root_seed,
        base_task_id,
        initializer_record_digest,
        slot,
    )
    if receipt["trajectory_seed"] != expected_trajectory_seed:
        raise ValueError("selector terminal trajectory seed violates its registered derivation")
    embedding = receipt["terminal_embedding"]
    if not isinstance(embedding, list) or receipt["terminal_embedding_digest"] != content_digest(
        embedding
    ):
        raise ValueError("selector terminal embedding digest mismatch")
    if not _is_sha256(receipt["terminal_chain_key"]):
        raise ValueError("selector terminal chain key is invalid")
    expected_program_digest = content_digest(
        {
            "terminal_embedding_digest": receipt["terminal_embedding_digest"],
            "features": features,
            "graph_inputs": graph_inputs,
        }
    )
    if receipt["program_registry_digest"] != expected_program_digest:
        raise ValueError("selector terminal receipt differs from its compiled program registry")
    trace = receipt["action_trace"]
    trace_keys = {
        "position",
        "source",
        "action_index",
        "opcode",
        "payload_key",
        "state_fingerprint",
        "support_fingerprint",
    }
    if (
        not isinstance(trace, list)
        or not trace
        or any(
            not isinstance(row, dict)
            or set(row) != trace_keys
            or row["position"] != position
            or type(row["action_index"]) is not int
            or row["action_index"] < 0
            or not isinstance(row["payload_key"], str)
            or not _is_sha256(row["state_fingerprint"])
            or not _is_sha256(row["support_fingerprint"])
            for position, row in enumerate(trace)
        )
        or trace[-1]["opcode"] != Opcode.COMMIT.value
        or trace[-1]["source"] != "outcome-blind-terminal-commit"
        or any(
            row["source"] != "uniform-legal-state-changing"
            or row["opcode"] in {Opcode.COMMIT.value, Opcode.STOP.value}
            for row in trace[:-1]
        )
        or len(trace) != realized + 1
    ):
        raise ValueError("selector terminal action trace is malformed")
    if not isinstance(receipt["validation_receipt"], dict) or not isinstance(
        receipt["cumulative_work"], dict
    ):
        raise ValueError("selector terminal validation/work receipt is malformed")
    return slot, str(receipt["record_digest"])


def _verified_dataset(
    directory: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"selector-label dataset does not exist: {directory}")
    actual = {path.name for path in directory.iterdir() if path.is_file()}
    if actual != _FILES:
        raise ValueError(
            f"selector-label file set differs: missing={sorted(_FILES - actual)}, "
            f"unknown={sorted(actual - _FILES)}"
        )
    manifest = _read_json(directory / "manifest.json", "selector-label manifest")
    _exact_keys(
        manifest,
        {
            "audit_mode",
            "context",
            "context_digest",
            "dataset_version",
            "label_protocol",
            "normalizer_digest",
            "outputs",
            "partitions",
            "quality_authority",
            "record_digest",
            "sample_seed",
            "schema",
            "schema_version",
            "seed_domain",
            "source_prepared_manifest_record_digest",
            "source_prepared_manifest_sha256",
            "split_protocol",
            "terminal_program_mixture",
            "terminal_source_census",
        },
        "selector-label manifest",
    )
    _verify_record(manifest, "selector-label manifest")
    if (
        manifest["schema"] != DATASET_SCHEMA
        or manifest["schema_version"] != SCHEMA_VERSION
        or manifest["dataset_version"] != DATASET_VERSION
        or manifest["seed_domain"] != SEED_DOMAIN
    ):
        raise ValueError("unsupported selector-label dataset version")
    if not isinstance(manifest["audit_mode"], bool):
        raise ValueError("selector-label audit_mode must be Boolean")
    if manifest["terminal_program_mixture"] != terminal_program_mixture_registry():
        raise ValueError("selector terminal program mixture differs from the frozen registry")
    census = manifest["terminal_source_census"]
    census_fields = {
        "source_slot",
        "source_kind",
        "rewrite_budget",
        "attempted_tasks",
        "terminalized_tasks",
        "retained_records",
        "duplicate_terminal_embeddings",
        "failed_terminalizations",
    }
    if not isinstance(census, dict) or set(census) != {
        str(slot) for slot in range(len(_TERMINAL_SOURCE_SPECS))
    }:
        raise ValueError("selector terminal source census has invalid mixture slots")
    for slot, (source_kind, rewrite_budget) in enumerate(_TERMINAL_SOURCE_SPECS):
        row = census[str(slot)]
        if not isinstance(row, dict):
            raise ValueError("selector terminal source census row must be an object")
        _exact_keys(row, census_fields, "selector terminal source census row")
        if (
            row["source_slot"] != slot
            or row["source_kind"] != source_kind
            or row["rewrite_budget"] != rewrite_budget
        ):
            raise ValueError("selector terminal source census differs from its registry")
        numeric = [
            _nonnegative_int(row[name], f"terminal source census {name}")
            for name in (
                "attempted_tasks",
                "terminalized_tasks",
                "retained_records",
                "duplicate_terminal_embeddings",
                "failed_terminalizations",
            )
        ]
        attempted, terminalized, retained, duplicates, failed = numeric
        if terminalized + failed != attempted or retained + duplicates != terminalized:
            raise ValueError("selector terminal source census arithmetic is inconsistent")
    _nonnegative_int(manifest["sample_seed"], "selector-label sample_seed")
    if manifest["context_digest"] != content_digest(manifest["context"]):
        raise ValueError("selector-label context digest mismatch")
    if not _is_sha256(manifest["source_prepared_manifest_record_digest"]) or not _is_sha256(
        manifest["source_prepared_manifest_sha256"]
    ):
        raise ValueError("selector-label source receipt is invalid")
    validate_selector_quality_authority(
        manifest["quality_authority"],
        audit_mode=manifest["audit_mode"],
    )
    outputs = manifest["outputs"]
    if not isinstance(outputs, dict) or set(outputs) != {"normalizer.json", "records.jsonl"}:
        raise ValueError("selector-label output registry mismatch")
    for filename in ("normalizer.json", "records.jsonl"):
        receipt = outputs[filename]
        if not isinstance(receipt, dict):
            raise ValueError(f"selector-label receipt for {filename} is invalid")
        _exact_keys(receipt, {"records", "sha256"}, f"selector-label receipt {filename}")
        _nonnegative_int(receipt["records"], f"selector-label receipt count {filename}")
        observed = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
        if receipt["sha256"] != observed:
            raise ValueError(f"selector-label checksum mismatch for {filename}")
    if outputs["normalizer.json"]["records"] != 1:
        raise ValueError("selector-label normalizer receipt must contain exactly one record")

    normalizer = _read_json(directory / "normalizer.json", "selector normalizer")
    _exact_keys(
        normalizer,
        {
            "coefficient_scale",
            "coefficient_scale_fit",
            "coefficient_units",
            "count_transform",
            "fit_lineages",
            "fit_partition",
            "knownness",
            "record_digest",
            "schema",
            "schema_version",
            "signed_count_transform",
        },
        "selector normalizer",
    )
    _verify_record(normalizer, "selector normalizer")
    if (
        normalizer["schema"] != NORMALIZER_SCHEMA
        or normalizer["schema_version"] != NORMALIZER_SCHEMA_VERSION
    ):
        raise ValueError("unsupported selector normalizer")
    normalizer_payload = {key: item for key, item in normalizer.items() if key != "record_digest"}
    if content_digest(normalizer_payload) != manifest["normalizer_digest"]:
        raise ValueError("selector-label normalizer digest mismatch")
    scale = normalizer["coefficient_scale"]
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("selector normalizer coefficient scale is invalid")

    raw_lines = (directory / "records.jsonl").read_bytes().splitlines()
    if any(not line.strip() for line in raw_lines):
        raise ValueError("selector-label records contain a blank line")
    rows = [
        _strict_object(line, f"selector-label record line {index}")
        for index, line in enumerate(raw_lines, start=1)
    ]
    if outputs["records.jsonl"]["records"] != len(rows):
        raise ValueError("selector-label record count differs from manifest")
    return manifest, normalizer_payload, rows


def _reject_unopened_audit(directory: Path, audit_mode: bool) -> None:
    if not isinstance(audit_mode, bool):
        raise ValueError("audit_mode must be Boolean")
    manifest = _read_json(directory / "manifest.json", "selector-label manifest")
    contains_audit = manifest.get("audit_mode")
    if not isinstance(contains_audit, bool):
        raise ValueError("selector-label audit_mode must be Boolean")
    if contains_audit and not audit_mode:
        raise PermissionError(
            "dataset contains held-out selector labels; explicit audit_mode=True is required"
        )


def load_selector_metadata(
    directory: str | os.PathLike[str], *, audit_mode: bool = False
) -> SelectorLabelMetadata:
    """Return authenticated preprocessing and provenance metadata for selector fitting."""

    root = Path(directory)
    _reject_unopened_audit(root, audit_mode)
    manifest, normalizer, _ = _verified_dataset(root)
    protocol = manifest["label_protocol"]
    if not isinstance(protocol, dict):
        raise ValueError("selector label protocol is invalid")
    return SelectorLabelMetadata(
        dataset_version=manifest["dataset_version"],
        manifest_sha256=hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        manifest_record_digest=manifest["record_digest"],
        source_prepared_manifest_sha256=manifest["source_prepared_manifest_sha256"],
        quality_authority=manifest["quality_authority"],
        context_digest=manifest["context_digest"],
        normalizer_digest=manifest["normalizer_digest"],
        coefficient_scale=float(normalizer["coefficient_scale"]),
        reads_per_strength=_positive_int(
            protocol.get("reads_per_strength"), "reads_per_strength"
        ),
        partitions=manifest["partitions"],
        terminal_program_mixture=manifest["terminal_program_mixture"],
    )


def load_selector_records(
    directory: str | os.PathLike[str],
    *,
    partition: str = "train",
    audit_mode: bool = False,
    audit_authorization: SelectorAuditAuthorization | None = None,
) -> tuple[StrengthRecord, ...]:
    """Load one authenticated partition as graph-complete :class:`StrengthRecord` objects."""

    if partition in {"test", "val", "validation", "ood"}:
        raise ValueError("use an explicit audit_* partition and audit_mode=True for held-out data")
    if partition not in _SELECTOR_PARTITIONS:
        raise ValueError(f"unknown selector partition {partition!r}")
    if partition in _AUDIT_PARTITIONS and not audit_mode:
        raise PermissionError("held-out selector records require explicit audit_mode=True")
    if partition == "audit_test":
        if audit_authorization is None:
            raise PermissionError(
                "audit_test remains sealed until the final RL-value freeze is authenticated"
            )
        audit_authorization.validate()
    elif audit_authorization is not None:
        raise ValueError("RL-value-selection authorization is valid only for audit_test")
    root = Path(directory)
    _reject_unopened_audit(root, audit_mode)
    manifest, normalizer, rows = _verified_dataset(root)
    if partition in _AUDIT_PARTITIONS and not manifest["audit_mode"]:
        raise ValueError("this selector dataset was published without audit records")

    partition_registry = manifest["partitions"]
    if not isinstance(partition_registry, dict) or set(partition_registry) != _SELECTOR_PARTITIONS:
        raise ValueError("selector-label partition registry mismatch")
    for name, receipt in partition_registry.items():
        if not isinstance(receipt, dict):
            raise ValueError(f"selector partition {name!r} receipt is invalid")
        _exact_keys(receipt, {"lineages", "records"}, f"selector partition {name}")
        if not isinstance(receipt["lineages"], list) or any(
            not isinstance(lineage, str) or not lineage for lineage in receipt["lineages"]
        ):
            raise ValueError(f"selector partition {name!r} has invalid lineages")
        _nonnegative_int(receipt["records"], f"selector partition {name} records")
    lineage_membership: dict[str, str] = {}
    for name, receipt in partition_registry.items():
        for lineage in receipt["lineages"]:
            previous = lineage_membership.setdefault(lineage, name)
            if previous != name:
                raise ValueError("selector train/calibration/audit lineage leakage")
    if normalizer["fit_lineages"] != partition_registry["train"]["lineages"]:
        raise ValueError("selector normalizer was not fitted on exactly the train lineages")

    protocol = manifest["label_protocol"]
    if not isinstance(protocol, dict):
        raise ValueError("selector label protocol is invalid")
    _exact_keys(
        protocol,
        {"endpoint", "evaluator", "reads_per_strength", "strength_ratios", "sweeps"},
        "selector label protocol",
    )
    expected_reads = _positive_int(protocol["reads_per_strength"], "reads_per_strength")
    _positive_int(protocol["sweeps"], "sweeps")
    if protocol["endpoint"] != "IF-Q3-S0":
        raise ValueError("selector label endpoint is not IF-Q3-S0")
    normalizer_digest = manifest["normalizer_digest"]
    if not _is_sha256(normalizer_digest):
        raise ValueError("selector normalizer identity is invalid")

    row_keys = {
        "base_task_id",
        "certificate_digest",
        "evaluator_protocol_digest",
        "features",
        "graph_inputs",
        "hits",
        "initializer_record_digest",
        "instance_id",
        "lineage",
        "program_id",
        "program_registry_digest",
        "reads",
        "record_digest",
        "record_id",
        "reference_status",
        "sample_seeds",
        "schema",
        "schema_version",
        "selector_partition",
        "source_partition",
        "terminal_source_receipt",
        "task_id",
    }
    seen_records: set[str] = set()
    seen_tasks: set[str] = set()
    seen_base_sources: set[tuple[str, int]] = set()
    seen_seeds: set[int] = set()
    observed_counts = {name: 0 for name in _SELECTOR_PARTITIONS}
    observed_lineages = {name: set() for name in _SELECTOR_PARTITIONS}
    observed_source_records = {
        str(slot): 0 for slot in range(len(_TERMINAL_SOURCE_SPECS))
    }
    observed_base_tasks: set[str] = set()
    decoded: list[tuple[str, StrengthRecord]] = []
    previous_order: tuple[str, str, int, str] | None = None
    root_seed = _nonnegative_int(manifest["sample_seed"], "selector-label sample_seed")
    for row in rows:
        _exact_keys(row, row_keys, "selector-label record")
        _verify_record(row, "selector-label record")
        if (
            row["schema"] != RECORD_SCHEMA
            or row["schema_version"] != RECORD_SCHEMA_VERSION
        ):
            raise ValueError("unsupported selector-label record version")
        record_id = row["record_id"]
        task_id = row["task_id"]
        program_id = row["program_id"]
        base_task_id = row["base_task_id"]
        if not isinstance(record_id, str) or record_id in seen_records:
            raise ValueError("duplicate or invalid selector record ID")
        if not isinstance(task_id, str) or not task_id or task_id in seen_tasks:
            raise ValueError("duplicate or invalid selector task ID")
        if (
            task_id != program_id
            or not isinstance(program_id, str)
            or not program_id.startswith("selector-program-")
            or not isinstance(base_task_id, str)
            or not base_task_id
        ):
            raise ValueError("selector program/base-task identity is invalid")
        identity_payload = {
            key: value for key, value in row.items() if key not in {"record_digest", "record_id"}
        }
        expected_id = "selector-record-" + content_digest(
            {"domain": DATASET_VERSION, "record": identity_payload}
        )
        if record_id != expected_id:
            raise ValueError("selector record ID does not match its content")
        seen_records.add(record_id)
        seen_tasks.add(task_id)
        observed_base_tasks.add(base_task_id)

        selector_partition = row["selector_partition"]
        if selector_partition not in _SELECTOR_PARTITIONS:
            raise ValueError("selector record has an invalid partition")
        source_expected = {
            "train": "train",
            "calibration": "train",
            "audit_val": "val",
            "audit_test": "test",
        }[selector_partition]
        if row["source_partition"] != source_expected:
            raise ValueError("selector partition is incompatible with the prepared partition")
        source_slot, source_receipt_digest = _validate_terminal_source_receipt(
            row["terminal_source_receipt"],
            root_seed=root_seed,
            base_task_id=base_task_id,
            initializer_record_digest=row["initializer_record_digest"],
            features=row["features"],
            graph_inputs=row["graph_inputs"],
        )
        if (base_task_id, source_slot) in seen_base_sources:
            raise ValueError("duplicate terminal mixture slot for one selector base task")
        seen_base_sources.add((base_task_id, source_slot))
        observed_source_records[str(source_slot)] += 1
        if row["program_registry_digest"] != row["terminal_source_receipt"].get(
            "program_registry_digest"
        ):
            raise ValueError("selector row and terminal source program digests differ")
        expected_program_id = "selector-program-" + content_digest(
            {
                "base_task_id": base_task_id,
                "source_receipt_record_digest": source_receipt_digest,
            }
        )
        if program_id != expected_program_id:
            raise ValueError("selector program ID does not match its terminal source")
        order = (selector_partition, base_task_id, source_slot, task_id)
        if previous_order is not None and order <= previous_order:
            raise ValueError("selector records are not in canonical partition/task/source order")
        previous_order = order
        lineage = row["lineage"]
        if not isinstance(lineage, str) or lineage_membership.get(lineage) != selector_partition:
            raise ValueError("selector record lineage disagrees with the manifest")
        if (
            row["reference_status"] not in CERTIFIED_REFERENCE_STATUSES
            or not _is_sha256(row["certificate_digest"])
            or not _is_sha256(row["evaluator_protocol_digest"])
            or not _is_sha256(row["initializer_record_digest"])
        ):
            raise ValueError("selector record has invalid trusted provenance")

        hits = row["hits"]
        reads = row["reads"]
        seeds = row["sample_seeds"]
        features = row["features"]
        graph_values = row["graph_inputs"]
        if not all(isinstance(value, list) and len(value) == N_STRENGTHS for value in (hits, reads, seeds, features, graph_values)):
            raise ValueError("selector record is not complete for all four strengths")
        integer_values: list[list[int]] = []
        for values, name in ((hits, "hits"), (reads, "reads"), (seeds, "sample seeds")):
            if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                raise ValueError(f"selector {name} must contain integers")
            integer_values.append(values)
        if any(total != expected_reads for total in reads) or any(
            hit < 0 or hit > total for hit, total in zip(hits, reads, strict=True)
        ):
            raise ValueError("selector record has incomplete or invalid counts")
        for strength_index, seed in enumerate(seeds):
            expected_seed = _seed_from_identity(
                root_seed,
                task_id,
                source_receipt_digest,
                strength_index,
                seen_seeds,
            )
            if seed != expected_seed:
                raise ValueError("selector sample seed violates the registered derivation")

        feature_maps: list[dict[str, float]] = []
        graphs: list[StrengthGraphInput] = []
        for index, (feature, graph_value) in enumerate(zip(features, graph_values, strict=True)):
            if not isinstance(feature, dict) or set(feature) != set(FEATURE_ORDER):
                raise ValueError("selector aggregate feature allowlist differs")
            converted = {name: float(feature[name]) for name in FEATURE_ORDER}
            if not np.isfinite(tuple(converted.values())).all():
                raise ValueError("selector aggregate features contain non-finite values")
            graph = _decode_graph(graph_value, normalizer_digest)
            if graph.strength_index != index:
                raise ValueError("selector graph inputs are not in strength-registry order")
            if converted["strength"] != graph.strength or converted["scale"] != graph.scale:
                raise ValueError("selector graph and aggregate program descriptors differ")
            feature_maps.append(converted)
            graphs.append(graph)
        strength_record = StrengthRecord(
            features=tuple(feature_maps),
            hits=tuple(hits),
            reads=tuple(reads),
            lineage=lineage,
            graph_inputs=tuple(graphs),
            task_id=task_id,
            source_record_id=record_id,
            selector_partition=selector_partition,
        )
        observed_counts[selector_partition] += 1
        observed_lineages[selector_partition].add(lineage)
        if selector_partition == partition:
            decoded.append((task_id, strength_record))

    for name, receipt in partition_registry.items():
        if receipt["records"] != observed_counts[name] or receipt["lineages"] != sorted(
            observed_lineages[name]
        ):
            raise ValueError(f"selector partition {name!r} receipt does not match records")
    census = manifest["terminal_source_census"]
    for slot in range(len(_TERMINAL_SOURCE_SPECS)):
        if census[str(slot)]["retained_records"] != observed_source_records[str(slot)]:
            raise ValueError("selector terminal source census differs from retained records")
        if census[str(slot)]["attempted_tasks"] != len(observed_base_tasks):
            raise ValueError("selector terminal source census differs from the base-task census")
    if sum(observed_source_records[str(slot)] for slot in range(1, len(_TERMINAL_SOURCE_SPECS))) == 0:
        raise ValueError("selector dataset collapsed to initializer-only terminal programs")
    if not decoded:
        raise ValueError(f"selector partition {partition!r} contains no records")
    return tuple(record for _, record in sorted(decoded, key=lambda item: item[0]))


__all__ = [
    "DATASET_SCHEMA",
    "DATASET_VERSION",
    "NORMALIZER_SCHEMA",
    "RECORD_SCHEMA",
    "SEED_DOMAIN",
    "SelectorLabelMetadata",
    "SelectorAuditAuthorization",
    "TERMINAL_MIXTURE_VERSION",
    "build_selector_labels",
    "load_selector_metadata",
    "load_selector_records",
    "terminal_program_mixture_registry",
    "validate_selector_quality_authority",
]
