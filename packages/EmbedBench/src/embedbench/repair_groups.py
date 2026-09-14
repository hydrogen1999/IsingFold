"""Deterministic, label-free repaired-candidate group generation.

Each fixed attempt slot starts from the same incumbent, samples one logical
edge, applies the native singleton-reset perturbation, and runs the pinned
classical resource repair.  Quality evaluation is deliberately absent from
this boundary; a later labeling step may seal successful drafts into a
candidate bank.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx

from .candidate_bank import (
    SCHEMA_VERSION,
    CandidateGroup,
    CandidateRecord,
    EvaluationCurve,
    InstanceRecord,
    RepairAttempt,
    _normalise_chains,
    assign_split,
    canonical_json_bytes,
    content_digest,
    derive_attempt_id,
    derive_candidate_id,
    derive_group_id,
    generation_payload,
    stable_seed,
    validate_group_against_instance,
    validate_instance,
)
from .objective import outcome_of

GENERATOR_VERSION = "uniform-edge-singleton-reset-resource-repair-v1"
REPAIR_DRAFT_SHARD_SCHEMA = "embedbench.repair-draft-shard"
GENERATION_PROTOCOL_FIELDS = frozenset(
    {
        "attempt_slots",
        "generator",
        "max_candidates",
        "max_transitions_per_attempt",
        "repairer",
    }
)
ATTEMPT_STATUSES = ("duplicate", "no_change", "repair_failed", "valid")
SPLITS = ("test", "train", "val")
Repairer = Callable[..., Mapping[str, Any]]


class InsufficientGroupBudget(ValueError):
    """Raised before slot zero when the complete fixed group is unaffordable."""


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(f"{name} schema fields differ: missing={missing}, unknown={unknown}")


def _require_sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError(f"{name} must be a SHA-256 digest") from error
    if value != value.lower():
        raise ValueError(f"{name} must use lowercase hexadecimal")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class GeneratedEmbedding:
    """One unlabeled embedding whose identity depends only on its chains."""

    candidate_id: str
    chains: tuple[tuple[int, ...], ...]
    total_qubits: int
    max_chain: int

    @classmethod
    def create(
        cls,
        group_id: str,
        chains: tuple[tuple[int, ...], ...],
    ) -> GeneratedEmbedding:
        canonical = _normalise_chains(chains)
        return cls(
            candidate_id=derive_candidate_id(group_id, canonical),
            chains=canonical,
            total_qubits=sum(map(len, canonical)),
            max_chain=max(map(len, canonical)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "chains": self.chains,
            "max_chain": self.max_chain,
            "schema_version": SCHEMA_VERSION,
            "total_qubits": self.total_qubits,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        group_id: str,
    ) -> GeneratedEmbedding:
        value = _require_mapping(value, "generated embedding")
        _require_exact_keys(
            value,
            {
                "candidate_id",
                "chains",
                "max_chain",
                "schema_version",
                "total_qubits",
            },
            "generated embedding",
        )
        if value["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"generated embedding requires schema_version {SCHEMA_VERSION}")
        try:
            chains = tuple(tuple(chain) for chain in value["chains"])
        except TypeError as error:
            raise ValueError("generated embedding chains must be a sequence") from error
        result = cls.create(group_id, chains)
        if canonical_json_bytes(result.to_dict()) != canonical_json_bytes(value):
            raise ValueError("generated embedding fields are not derived from its chains")
        return result


@dataclass(frozen=True, slots=True)
class GeneratedRepairGroup:
    """Complete structural draft, including unsuccessful fixed attempt slots."""

    group_id: str
    instance_id: str
    instance_record_digest: str
    split_unit_id: str
    group_seed: int
    protocol: tuple[tuple[str, Any], ...]
    incumbent: GeneratedEmbedding
    candidates: tuple[GeneratedEmbedding, ...]
    attempts: tuple[RepairAttempt, ...]
    rejection_reason: str | None
    record_digest: str

    @property
    def sealable(self) -> bool:
        return self.rejection_reason is None and len(self.candidates) >= 2

    def payload_dict(self) -> dict[str, Any]:
        """Return the pre-digest payload committed by ``record_digest``."""

        return generation_payload(
            group_id=self.group_id,
            instance_id=self.instance_id,
            instance_record_digest=self.instance_record_digest,
            split_unit_id=self.split_unit_id,
            group_seed=self.group_seed,
            protocol=self.protocol,
            incumbent_chains=self.incumbent.chains,
            candidate_chains=tuple(candidate.chains for candidate in self.candidates),
            attempts=self.attempts,
            rejection_reason=self.rejection_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        instance: InstanceRecord | None = None,
    ) -> GeneratedRepairGroup:
        value = _require_mapping(value, "generated repair group")
        _require_exact_keys(
            value,
            {
                "attempts",
                "candidates",
                "group_id",
                "group_seed",
                "incumbent",
                "instance_id",
                "instance_record_digest",
                "protocol",
                "record_digest",
                "rejection_reason",
                "schema_version",
                "split",
                "split_unit_id",
            },
            "generated repair group",
        )
        if value["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"generated repair group requires schema_version {SCHEMA_VERSION}")
        if value["split"] != assign_split(value["split_unit_id"]):
            raise ValueError("generated repair group split must be derived from split_unit_id")
        try:
            protocol = tuple(tuple(item) for item in value["protocol"])
            attempts = tuple(RepairAttempt.from_dict(item) for item in value["attempts"])
        except TypeError as error:
            raise ValueError("generated repair group sequences are malformed") from error
        group_id = value["group_id"]
        result = cls(
            group_id=group_id,
            instance_id=value["instance_id"],
            instance_record_digest=value["instance_record_digest"],
            split_unit_id=value["split_unit_id"],
            group_seed=value["group_seed"],
            protocol=protocol,
            incumbent=GeneratedEmbedding.from_dict(value["incumbent"], group_id=group_id),
            candidates=tuple(
                GeneratedEmbedding.from_dict(item, group_id=group_id)
                for item in value["candidates"]
            ),
            attempts=attempts,
            rejection_reason=value["rejection_reason"],
            record_digest=value["record_digest"],
        )
        _validate_generated_group_intrinsic(result)
        if instance is not None:
            validate_generated_group(result, instance)
        return result


@dataclass(frozen=True, slots=True)
class _AttemptOutcome:
    chains: tuple[tuple[int, ...], ...] | None
    transitions: int
    reason: str


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _repair_seed(
    group_seed: int,
    instance_id: str,
    incumbent_chains: tuple[tuple[int, ...], ...],
    slot: int,
) -> int:
    return stable_seed(
        group_seed,
        "uniform-edge-repair-v1",
        instance_id,
        content_digest(incumbent_chains),
        slot,
    )


def _graphs(instance: InstanceRecord) -> tuple[nx.Graph, nx.Graph]:
    source = nx.Graph()
    source.add_nodes_from(instance.logical_nodes)
    source.add_edges_from(instance.logical_edges)
    target = nx.Graph()
    target.add_nodes_from(instance.host_nodes)
    target.add_edges_from(instance.host_edges)
    return source, target


def _assert_valid_embedding(
    instance: InstanceRecord,
    chains: tuple[tuple[int, ...], ...],
    *,
    name: str,
) -> None:
    source, target = _graphs(instance)
    if len(chains) != len(instance.logical_nodes):
        raise ValueError(f"{name} has the wrong chain count")
    embedding = {
        logical: frozenset(chain)
        for logical, chain in zip(instance.logical_nodes, chains, strict=True)
    }
    try:
        valid = outcome_of(embedding, source, target)[0] == 1
    except nx.NetworkXError as error:
        raise ValueError(f"{name} contains a qubit outside the host graph") from error
    if not valid:
        raise ValueError(f"{name} is not a valid minor embedding")


def _run_repair_attempt(
    *,
    instance: InstanceRecord,
    incumbent_chains: tuple[tuple[int, ...], ...],
    neighborhood: tuple[int, int],
    repair_seed: int,
    slot: int,
    max_candidates: int,
    max_transitions_per_attempt: int,
    repairer: Repairer,
) -> _AttemptOutcome:
    """Invoke one external method primitive through a dependency-free payload."""

    source, target = _graphs(instance)
    incumbent = {
        logical: list(chain)
        for logical, chain in zip(instance.logical_nodes, incumbent_chains, strict=True)
    }
    raw = repairer(
        source,
        target,
        incumbent,
        neighborhood=neighborhood,
        random_seed=repair_seed,
        max_candidates=max_candidates,
        max_transitions=max_transitions_per_attempt,
    )
    if not isinstance(raw, Mapping):
        raise TypeError(f"repairer returned {type(raw).__name__}, expected a mapping")
    transitions = raw.get("transitions")
    reason = raw.get("reason")
    if isinstance(transitions, bool) or not isinstance(transitions, int) or transitions < 0:
        raise ValueError(f"repairer returned invalid transitions for slot {slot}")
    if transitions > max_transitions_per_attempt:
        raise ValueError(
            f"repairer transitions for slot {slot} exceed the per-attempt cap "
            f"{max_transitions_per_attempt}"
        )
    if not isinstance(reason, str) or not reason:
        raise ValueError(f"repairer returned invalid reason for slot {slot}")
    repaired = raw.get("chains")
    if repaired is None:
        return _AttemptOutcome(chains=None, transitions=transitions, reason=reason)
    if not isinstance(repaired, Mapping) or set(repaired) != set(instance.logical_nodes):
        raise ValueError(f"repairer returned incomplete chains for slot {slot}")
    canonical = _normalise_chains(tuple(repaired[node] for node in instance.logical_nodes))
    return _AttemptOutcome(
        chains=canonical,
        transitions=transitions,
        reason=reason,
    )


def generate_repaired_group(
    instance: InstanceRecord,
    incumbent_chains: tuple[tuple[int, ...], ...],
    *,
    group_seed: int,
    attempt_slots: int,
    max_candidates: int = 8,
    max_transitions_per_attempt: int = 200,
    available_transition_budget: int | None = None,
    repairer: Repairer,
    repairer_id: str,
) -> GeneratedRepairGroup:
    """Generate one complete label-free group under a fixed slot budget.

    The sampled edge triggers a native singleton reset.  Subsequent resource
    repair may change other logical chains because native eligibility is global.
    No attempt is resampled or omitted based on its outcome.
    """

    validate_instance(instance)
    if not callable(repairer):
        raise TypeError("repairer must be callable")
    if not isinstance(repairer_id, str) or not repairer_id:
        raise ValueError("repairer_id must be a non-empty versioned identifier")
    if isinstance(group_seed, bool) or not isinstance(group_seed, int):
        raise ValueError("group_seed must be an integer")
    attempt_slots = _positive_int(attempt_slots, "attempt_slots")
    max_candidates = _positive_int(max_candidates, "max_candidates")
    max_transitions_per_attempt = _positive_int(
        max_transitions_per_attempt,
        "max_transitions_per_attempt",
    )
    required_budget = attempt_slots * max_transitions_per_attempt
    if available_transition_budget is not None:
        if (
            isinstance(available_transition_budget, bool)
            or not isinstance(available_transition_budget, int)
            or available_transition_budget < 0
        ):
            raise ValueError("available_transition_budget must be a non-negative integer")
        if available_transition_budget < required_budget:
            raise InsufficientGroupBudget(
                "whole group budget is insufficient: "
                f"requires {required_budget} transitions, has {available_transition_budget}"
            )

    canonical_incumbent = _normalise_chains(incumbent_chains)
    _assert_valid_embedding(instance, canonical_incumbent, name="incumbent")
    protocol = (
        ("attempt_slots", attempt_slots),
        ("generator", GENERATOR_VERSION),
        ("max_candidates", max_candidates),
        ("max_transitions_per_attempt", max_transitions_per_attempt),
        ("repairer", repairer_id),
    )
    group_id = derive_group_id(
        instance.instance_id,
        canonical_incumbent,
        protocol,
        group_seed,
    )
    incumbent = GeneratedEmbedding.create(group_id, canonical_incumbent)
    edges = tuple(sorted(instance.logical_edges))
    if not edges:
        raise ValueError("repair generation requires at least one logical edge")

    candidates_by_chains: dict[tuple[tuple[int, ...], ...], GeneratedEmbedding] = {}
    attempts: list[RepairAttempt] = []
    for slot in range(attempt_slots):
        repair_seed = _repair_seed(
            group_seed,
            instance.instance_id,
            canonical_incumbent,
            slot,
        )
        edge_seed = stable_seed(repair_seed, "edge")
        neighborhood = random.Random(edge_seed).choice(edges)
        outcome = _run_repair_attempt(
            instance=instance,
            incumbent_chains=canonical_incumbent,
            neighborhood=neighborhood,
            repair_seed=repair_seed,
            slot=slot,
            max_candidates=max_candidates,
            max_transitions_per_attempt=max_transitions_per_attempt,
            repairer=repairer,
        )
        if outcome.chains is None:
            status = "repair_failed"
            candidate_id = None
            reason = outcome.reason
        else:
            repaired = _normalise_chains(outcome.chains)
            _assert_valid_embedding(instance, repaired, name=f"repair slot {slot}")
            if repaired == canonical_incumbent:
                status = "no_change"
                candidate_id = None
            elif repaired in candidates_by_chains:
                status = "duplicate"
                candidate_id = candidates_by_chains[repaired].candidate_id
            else:
                status = "valid"
                generated = GeneratedEmbedding.create(group_id, repaired)
                candidates_by_chains[repaired] = generated
                candidate_id = generated.candidate_id
            reason = None
        attempts.append(
            RepairAttempt(
                attempt_id=derive_attempt_id(group_id, slot),
                repair_seed=repair_seed,
                neighborhood=neighborhood,
                status=status,
                candidate_id=candidate_id,
                slot=slot,
                transitions=outcome.transitions,
                reason=reason,
            )
        )

    candidates = tuple(sorted(candidates_by_chains.values(), key=lambda item: item.candidate_id))
    rejection_reason = None if len(candidates) >= 2 else "fewer_than_2_unique_candidates"
    provisional = GeneratedRepairGroup(
        group_id=group_id,
        instance_id=instance.instance_id,
        instance_record_digest=instance.record_digest,
        split_unit_id=instance.split_unit_id,
        group_seed=group_seed,
        protocol=protocol,
        incumbent=incumbent,
        candidates=candidates,
        attempts=tuple(attempts),
        rejection_reason=rejection_reason,
        record_digest="0" * 64,
    )
    payload = provisional.payload_dict()
    return GeneratedRepairGroup(
        **{
            field: getattr(provisional, field)
            for field in provisional.__dataclass_fields__
            if field != "record_digest"
        },
        record_digest=content_digest(payload),
    )


def _validated_protocol(group: GeneratedRepairGroup) -> dict[str, Any]:
    try:
        protocol = dict(group.protocol)
    except (TypeError, ValueError) as error:
        raise ValueError("generation protocol must contain name/value pairs") from error
    if len(protocol) != len(group.protocol):
        raise ValueError("generation protocol fields must be unique")
    if set(protocol) != GENERATION_PROTOCOL_FIELDS:
        missing = sorted(GENERATION_PROTOCOL_FIELDS - set(protocol))
        unexpected = sorted(set(protocol) - GENERATION_PROTOCOL_FIELDS)
        raise ValueError(
            "generation protocol fields differ: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if tuple(sorted(group.protocol)) != group.protocol:
        raise ValueError("generation protocol fields must use canonical ordering")
    if protocol["generator"] != GENERATOR_VERSION:
        raise ValueError("generation protocol records an unsupported generator")
    _positive_int(protocol["attempt_slots"], "generation protocol attempt_slots")
    _positive_int(protocol["max_candidates"], "generation protocol max_candidates")
    _positive_int(
        protocol["max_transitions_per_attempt"],
        "generation protocol max_transitions_per_attempt",
    )
    if not isinstance(protocol["repairer"], str) or not protocol["repairer"]:
        raise ValueError("generation protocol repairer must be a non-empty identifier")
    return protocol


def _validate_generated_group_intrinsic(group: GeneratedRepairGroup) -> None:
    """Validate all draft invariants that do not require the instance graph."""

    for name in ("group_id", "instance_id", "split_unit_id"):
        value = getattr(group, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"generation draft {name} must be a non-empty string")
    _require_sha256(group.instance_record_digest, "generation draft instance_record_digest")
    _require_sha256(group.record_digest, "generation draft record_digest")
    if isinstance(group.group_seed, bool) or not isinstance(group.group_seed, int):
        raise ValueError("generation draft group_seed must be an integer")
    protocol = _validated_protocol(group)
    if group.group_id != derive_group_id(
        group.instance_id,
        group.incumbent.chains,
        group.protocol,
        group.group_seed,
    ):
        raise ValueError("generation draft group_id mismatch")
    if group.incumbent != GeneratedEmbedding.create(group.group_id, group.incumbent.chains):
        raise ValueError("generation draft incumbent identity mismatch")
    canonical_candidates = tuple(
        sorted(
            (GeneratedEmbedding.create(group.group_id, item.chains) for item in group.candidates),
            key=lambda item: item.candidate_id,
        )
    )
    if group.candidates != canonical_candidates:
        raise ValueError("generation draft candidates are not canonical")
    if len({item.candidate_id for item in group.candidates}) != len(group.candidates):
        raise ValueError("generation draft contains duplicate candidates")
    if group.incumbent.chains in {item.chains for item in group.candidates}:
        raise ValueError("generation draft candidate duplicates the incumbent")
    attempt_slots = protocol["attempt_slots"]
    maximum_transitions = protocol["max_transitions_per_attempt"]
    if len(group.attempts) != attempt_slots:
        raise ValueError("generation draft must retain every fixed attempt slot")
    if tuple(attempt.slot for attempt in group.attempts) != tuple(range(attempt_slots)):
        raise ValueError("generation draft attempt slots must be complete and contiguous")
    known_ids = {item.candidate_id for item in group.candidates}
    first_valid_ids: set[str] = set()
    for attempt in group.attempts:
        if attempt.transitions > maximum_transitions:
            raise ValueError("repair-attempt transitions exceed the recorded per-attempt cap")
        if attempt.attempt_id != derive_attempt_id(group.group_id, attempt.slot):
            raise ValueError("generation draft attempt identity mismatch")
        if attempt.status == "repair_failed":
            if attempt.reason is None:
                raise ValueError("failed repair attempt must retain its failure reason")
        elif attempt.reason is not None:
            raise ValueError("non-failed repair attempt cannot carry a failure reason")
        if attempt.status in {"valid", "duplicate"} and attempt.candidate_id not in known_ids:
            raise ValueError("generation draft attempt references an unknown candidate")
        if attempt.status == "valid":
            if attempt.candidate_id in first_valid_ids:
                raise ValueError("a candidate cannot have more than one first-valid attempt")
            first_valid_ids.add(attempt.candidate_id)
        elif attempt.status == "duplicate" and attempt.candidate_id not in first_valid_ids:
            raise ValueError("duplicate repair attempt must follow its first-valid attempt")
    if first_valid_ids != known_ids:
        raise ValueError("every generated candidate needs exactly one first-valid attempt")
    expected_rejection = None if len(group.candidates) >= 2 else "fewer_than_2_unique_candidates"
    if group.rejection_reason != expected_rejection:
        raise ValueError("generation draft rejection reason is inconsistent")
    if group.record_digest != content_digest(group.payload_dict()):
        raise ValueError("generation draft digest mismatch")


def validate_generated_group(group: GeneratedRepairGroup, instance: InstanceRecord) -> None:
    """Validate the complete structural draft before any labels are attached."""

    validate_instance(instance)
    _validate_generated_group_intrinsic(group)
    if group.instance_id != instance.instance_id:
        raise ValueError("generation draft references the wrong instance")
    if group.instance_record_digest != instance.record_digest:
        raise ValueError("generation draft references the wrong instance digest")
    if group.split_unit_id != instance.split_unit_id:
        raise ValueError("generation draft references the wrong split unit")
    _assert_valid_embedding(instance, group.incumbent.chains, name="incumbent")
    for candidate in group.candidates:
        _assert_valid_embedding(
            instance,
            candidate.chains,
            name=f"candidate {candidate.candidate_id}",
        )
    edges = tuple(sorted(instance.logical_edges))
    if not edges:
        raise ValueError("repair generation requires at least one logical edge")
    for attempt in group.attempts:
        if attempt.repair_seed != _repair_seed(
            group.group_seed,
            group.instance_id,
            group.incumbent.chains,
            attempt.slot,
        ):
            raise ValueError("generation draft repair seed mismatch")
        expected_neighborhood = random.Random(
            stable_seed(attempt.repair_seed, "edge")
        ).choice(edges)
        if attempt.neighborhood != expected_neighborhood:
            raise ValueError("generation draft neighborhood is not derived from the repair seed")


def seal_generated_group(
    group: GeneratedRepairGroup,
    instance: InstanceRecord,
    *,
    labels: Mapping[str, tuple[EvaluationCurve, EvaluationCurve]],
) -> CandidateGroup:
    """Attach labels to exactly one previously committed structural universe."""

    validate_generated_group(group, instance)
    if not group.sealable:
        raise ValueError("generation draft is not sealable")
    expected_ids = {item.candidate_id for item in (group.incumbent, *group.candidates)}
    if set(labels) != expected_ids:
        raise ValueError("labels must match exactly the committed candidate universe")

    def labeled(item: GeneratedEmbedding) -> CandidateRecord:
        decision, audit = labels[item.candidate_id]
        return CandidateRecord.create(
            group_id=group.group_id,
            candidate_id=item.candidate_id,
            chains=item.chains,
            decision=decision,
            audit=audit,
        )

    sealed = CandidateGroup.create(
        group_id=group.group_id,
        instance_id=group.instance_id,
        instance_record_digest=group.instance_record_digest,
        split_unit_id=group.split_unit_id,
        split=assign_split(group.split_unit_id),
        group_seed=group.group_seed,
        protocol=group.protocol,
        incumbent=labeled(group.incumbent),
        candidates=tuple(labeled(item) for item in group.candidates),
        attempts=group.attempts,
        generation_digest=group.record_digest,
    )
    validate_group_against_instance(sealed, instance)
    return sealed


def structural_bytes(group: GeneratedRepairGroup) -> bytes:
    """Return the canonical structural payload used by external shard writers."""

    return canonical_json_bytes(group.to_dict())


def _validate_count_pairs(
    value: Sequence[tuple[str, int]],
    *,
    name: str,
    exact_names: frozenset[str] | None = None,
    positive: bool = False,
) -> tuple[tuple[str, int], ...]:
    try:
        result = tuple((key, count) for key, count in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain name/count pairs") from error
    if any(not isinstance(key, str) or not key for key, _ in result):
        raise ValueError(f"{name} names must be non-empty strings")
    if len({key for key, _ in result}) != len(result):
        raise ValueError(f"{name} names must be unique")
    if result != tuple(sorted(result)):
        raise ValueError(f"{name} must use canonical ordering")
    for key, count in result:
        _nonnegative_int(count, f"{name}[{key!r}]")
        if positive and count == 0:
            raise ValueError(f"{name}[{key!r}] must be positive")
    if exact_names is not None and {key for key, _ in result} != exact_names:
        raise ValueError(f"{name} must contain exactly {sorted(exact_names)}")
    return result


@dataclass(frozen=True, slots=True)
class RepairDraftManifest:
    """Detached integrity and accounting record for one repair-draft shard."""

    jsonl_sha256: str
    byte_count: int
    instance_count: int
    draft_count: int
    accepted_count: int
    rejected_count: int
    underfull_count: int
    drafts_with_repair_failure_count: int
    attempt_status_counts: tuple[tuple[str, int], ...]
    rejection_reason_counts: tuple[tuple[str, int], ...]
    split_counts: tuple[tuple[str, int], ...]
    protocol_digests: tuple[str, ...]
    record_digest: str
    schema: str = REPAIR_DRAFT_SHARD_SCHEMA
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema != REPAIR_DRAFT_SHARD_SCHEMA:
            raise ValueError(f"repair-draft manifest requires schema {REPAIR_DRAFT_SHARD_SCHEMA!r}")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"repair-draft manifest requires schema_version {SCHEMA_VERSION}")
        _require_sha256(self.jsonl_sha256, "repair-draft manifest jsonl_sha256")
        _require_sha256(self.record_digest, "repair-draft manifest record_digest")
        for name in (
            "byte_count",
            "instance_count",
            "draft_count",
            "accepted_count",
            "rejected_count",
            "underfull_count",
            "drafts_with_repair_failure_count",
        ):
            _nonnegative_int(getattr(self, name), f"repair-draft manifest {name}")
        if self.draft_count == 0:
            raise ValueError("repair-draft manifest must account for at least one draft")
        if self.accepted_count + self.rejected_count != self.draft_count:
            raise ValueError("repair-draft manifest accepted/rejected counts are inconsistent")
        if self.underfull_count > self.rejected_count:
            raise ValueError("repair-draft manifest underfull count exceeds rejected count")
        if self.drafts_with_repair_failure_count > self.draft_count:
            raise ValueError("repair-draft manifest failure count exceeds draft count")
        attempt_counts = _validate_count_pairs(
            self.attempt_status_counts,
            name="repair-draft attempt_status_counts",
            exact_names=frozenset(ATTEMPT_STATUSES),
        )
        rejection_counts = _validate_count_pairs(
            self.rejection_reason_counts,
            name="repair-draft rejection_reason_counts",
            positive=True,
        )
        split_counts = _validate_count_pairs(
            self.split_counts,
            name="repair-draft split_counts",
            exact_names=frozenset(SPLITS),
        )
        if sum(count for _, count in rejection_counts) != self.rejected_count:
            raise ValueError("repair-draft manifest rejection-reason counts are inconsistent")
        if sum(count for _, count in split_counts) != self.draft_count:
            raise ValueError("repair-draft manifest split counts are inconsistent")
        if sum(count for _, count in attempt_counts) < self.draft_count:
            raise ValueError("repair-draft manifest attempt counts are inconsistent")
        if not self.protocol_digests:
            raise ValueError("repair-draft manifest must record at least one protocol digest")
        if tuple(sorted(set(self.protocol_digests))) != self.protocol_digests:
            raise ValueError("repair-draft manifest protocol digests must be sorted and unique")
        for digest in self.protocol_digests:
            _require_sha256(digest, "repair-draft protocol digest")

    def payload_dict(self) -> dict[str, Any]:
        return {
            "accepted_count": self.accepted_count,
            "attempt_status_counts": self.attempt_status_counts,
            "byte_count": self.byte_count,
            "draft_count": self.draft_count,
            "drafts_with_repair_failure_count": self.drafts_with_repair_failure_count,
            "instance_count": self.instance_count,
            "jsonl_sha256": self.jsonl_sha256,
            "protocol_digests": self.protocol_digests,
            "rejected_count": self.rejected_count,
            "rejection_reason_counts": self.rejection_reason_counts,
            "schema": self.schema,
            "schema_version": self.schema_version,
            "split_counts": self.split_counts,
            "underfull_count": self.underfull_count,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload_dict(), "record_digest": self.record_digest}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RepairDraftManifest:
        value = _require_mapping(value, "repair-draft manifest")
        _require_exact_keys(
            value,
            {
                "accepted_count",
                "attempt_status_counts",
                "byte_count",
                "draft_count",
                "drafts_with_repair_failure_count",
                "instance_count",
                "jsonl_sha256",
                "protocol_digests",
                "record_digest",
                "rejected_count",
                "rejection_reason_counts",
                "schema",
                "schema_version",
                "split_counts",
                "underfull_count",
            },
            "repair-draft manifest",
        )
        try:
            result = cls(
                jsonl_sha256=value["jsonl_sha256"],
                byte_count=value["byte_count"],
                instance_count=value["instance_count"],
                draft_count=value["draft_count"],
                accepted_count=value["accepted_count"],
                rejected_count=value["rejected_count"],
                underfull_count=value["underfull_count"],
                drafts_with_repair_failure_count=value[
                    "drafts_with_repair_failure_count"
                ],
                attempt_status_counts=tuple(map(tuple, value["attempt_status_counts"])),
                rejection_reason_counts=tuple(map(tuple, value["rejection_reason_counts"])),
                split_counts=tuple(map(tuple, value["split_counts"])),
                protocol_digests=tuple(value["protocol_digests"]),
                record_digest=value["record_digest"],
                schema=value["schema"],
                schema_version=value["schema_version"],
            )
        except TypeError as error:
            raise ValueError("repair-draft manifest sequences are malformed") from error
        if result.record_digest != content_digest(result.payload_dict()):
            raise ValueError("repair-draft manifest digest mismatch")
        return result


def _repair_draft_manifest(
    raw: bytes,
    instances: Sequence[InstanceRecord],
    drafts: Sequence[GeneratedRepairGroup],
) -> RepairDraftManifest:
    status_counts: Counter[str] = Counter(
        attempt.status for draft in drafts for attempt in draft.attempts
    )
    rejection_counts: Counter[str] = Counter(
        draft.rejection_reason
        for draft in drafts
        if draft.rejection_reason is not None
    )
    split_counts: Counter[str] = Counter(assign_split(draft.split_unit_id) for draft in drafts)
    provisional = RepairDraftManifest(
        jsonl_sha256=hashlib.sha256(raw).hexdigest(),
        byte_count=len(raw),
        instance_count=len(instances),
        draft_count=len(drafts),
        accepted_count=sum(draft.sealable for draft in drafts),
        rejected_count=sum(not draft.sealable for draft in drafts),
        underfull_count=sum(len(draft.candidates) < 2 for draft in drafts),
        drafts_with_repair_failure_count=sum(
            any(attempt.status == "repair_failed" for attempt in draft.attempts)
            for draft in drafts
        ),
        attempt_status_counts=tuple((status, status_counts[status]) for status in ATTEMPT_STATUSES),
        rejection_reason_counts=tuple(sorted(rejection_counts.items())),
        split_counts=tuple((split, split_counts[split]) for split in SPLITS),
        protocol_digests=tuple(
            sorted({content_digest({"protocol": draft.protocol}) for draft in drafts})
        ),
        record_digest="0" * 64,
    )
    return RepairDraftManifest(
        **{
            field: getattr(provisional, field)
            for field in provisional.__dataclass_fields__
            if field != "record_digest"
        },
        record_digest=content_digest(provisional.payload_dict()),
    )


def repair_draft_manifest_path(path: str | os.PathLike[str]) -> Path:
    """Return the conventional detached-manifest path for a JSONL shard."""

    return Path(f"{os.fspath(path)}.manifest.json")


def _manifest_destination(
    path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str] | None,
) -> Path:
    destination = Path(path)
    result = (
        repair_draft_manifest_path(destination) if manifest_path is None else Path(manifest_path)
    )
    if result == destination:
        raise ValueError("repair-draft data and manifest paths must differ")
    return result


def _strict_json_object(raw: bytes, name: str) -> dict[str, Any]:
    def reject_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(token: str) -> None:
        raise ValueError(f"{name} contains non-finite number {token}")

    try:
        value = json.loads(raw, object_pairs_hook=reject_pairs, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _normalise_shard_records(
    instances: Sequence[InstanceRecord],
    drafts: Sequence[GeneratedRepairGroup],
) -> tuple[tuple[InstanceRecord, ...], tuple[GeneratedRepairGroup, ...]]:
    instance_rows = tuple(sorted(instances, key=lambda item: item.instance_id))
    draft_rows = tuple(sorted(drafts, key=lambda item: item.group_id))
    if not draft_rows:
        raise ValueError("repair-draft shard requires at least one draft")
    if len({item.instance_id for item in instance_rows}) != len(instance_rows):
        raise ValueError("duplicate instance_id in repair-draft shard")
    if len({item.group_id for item in draft_rows}) != len(draft_rows):
        raise ValueError("duplicate group_id in repair-draft shard")
    by_id = {instance.instance_id: instance for instance in instance_rows}
    for instance in instance_rows:
        validate_instance(instance)
    for draft in draft_rows:
        if draft.instance_id not in by_id:
            raise ValueError(f"repair draft references unknown instance {draft.instance_id!r}")
        validate_generated_group(draft, by_id[draft.instance_id])
    return instance_rows, draft_rows


def _shard_bytes(
    instances: Sequence[InstanceRecord],
    drafts: Sequence[GeneratedRepairGroup],
) -> bytes:
    rows = [
        *({"kind": "instance", "record": item.to_dict()} for item in instances),
        *({"kind": "repair_draft", "record": item.to_dict()} for item in drafts),
    ]
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def _atomic_write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_repair_draft_shard(
    path: str | os.PathLike[str],
    *,
    instances: Sequence[InstanceRecord],
    drafts: Sequence[GeneratedRepairGroup],
    manifest_path: str | os.PathLike[str] | None = None,
) -> RepairDraftManifest:
    """Validate and atomically publish a deterministic draft shard and manifest."""

    destination = Path(path)
    manifest_destination = _manifest_destination(destination, manifest_path)
    instance_rows, draft_rows = _normalise_shard_records(instances, drafts)
    raw = _shard_bytes(instance_rows, draft_rows)
    manifest = _repair_draft_manifest(raw, instance_rows, draft_rows)
    manifest_raw = canonical_json_bytes(manifest.to_dict()) + b"\n"
    # Data is replaced first; the detached manifest is the completion marker.
    # A process interruption therefore fails closed on the next read.
    _atomic_write(destination, raw)
    _atomic_write(manifest_destination, manifest_raw)
    return manifest


def read_repair_draft_shard(
    path: str | os.PathLike[str],
    *,
    manifest_path: str | os.PathLike[str] | None = None,
) -> tuple[
    tuple[InstanceRecord, ...],
    tuple[GeneratedRepairGroup, ...],
    RepairDraftManifest,
]:
    """Read a shard only after validating bytes, records, graph semantics, and counts."""

    destination = Path(path)
    manifest_destination = _manifest_destination(destination, manifest_path)
    raw = destination.read_bytes()
    manifest_raw = manifest_destination.read_bytes()
    manifest_value = _strict_json_object(manifest_raw, "repair-draft manifest")
    manifest = RepairDraftManifest.from_dict(manifest_value)
    if manifest_raw != canonical_json_bytes(manifest.to_dict()) + b"\n":
        raise ValueError("repair-draft manifest does not use canonical encoding")
    if len(raw) != manifest.byte_count:
        raise ValueError("repair-draft shard byte count mismatch")
    if hashlib.sha256(raw).hexdigest() != manifest.jsonl_sha256:
        raise ValueError("repair-draft shard checksum mismatch")

    instances: list[InstanceRecord] = []
    drafts: list[GeneratedRepairGroup] = []
    reached_drafts = False
    for index, line in enumerate(raw.splitlines(), start=1):
        row = _strict_json_object(line, f"repair-draft shard line {index}")
        _require_exact_keys(row, {"kind", "record"}, f"repair-draft shard line {index}")
        if row["kind"] == "instance":
            if reached_drafts:
                raise ValueError("repair-draft shard instance rows must precede draft rows")
            instances.append(InstanceRecord.from_dict(row["record"]))
        elif row["kind"] == "repair_draft":
            reached_drafts = True
            drafts.append(GeneratedRepairGroup.from_dict(row["record"]))
        else:
            raise ValueError(f"repair-draft shard line {index} has unknown kind")

    instance_rows, draft_rows = _normalise_shard_records(instances, drafts)
    if _shard_bytes(instance_rows, draft_rows) != raw:
        raise ValueError("repair-draft shard rows are not canonically ordered or encoded")
    expected_manifest = _repair_draft_manifest(raw, instance_rows, draft_rows)
    if manifest != expected_manifest:
        raise ValueError("repair-draft manifest accounting mismatch")
    return instance_rows, draft_rows, manifest


def append_repair_draft_shard(
    path: str | os.PathLike[str],
    *,
    instances: Sequence[InstanceRecord],
    drafts: Sequence[GeneratedRepairGroup],
    manifest_path: str | os.PathLike[str] | None = None,
) -> RepairDraftManifest:
    """Append through a checked read/rewrite cycle for a single-writer workflow."""

    destination = Path(path)
    manifest_destination = _manifest_destination(destination, manifest_path)
    data_exists = destination.exists()
    manifest_exists = manifest_destination.exists()
    if data_exists != manifest_exists:
        raise ValueError("repair-draft shard is incomplete: data/manifest presence differs")
    if not data_exists:
        return write_repair_draft_shard(
            destination,
            instances=instances,
            drafts=drafts,
            manifest_path=manifest_destination,
        )
    existing_instances, existing_drafts, _ = read_repair_draft_shard(
        destination,
        manifest_path=manifest_destination,
    )
    new_instances = tuple(instances)
    new_drafts = tuple(drafts)
    existing_instance_ids = {item.instance_id for item in existing_instances}
    existing_group_ids = {item.group_id for item in existing_drafts}
    if any(item.instance_id in existing_instance_ids for item in new_instances):
        raise ValueError("duplicate instance_id while appending repair-draft shard")
    if any(item.group_id in existing_group_ids for item in new_drafts):
        raise ValueError("duplicate group_id while appending repair-draft shard")
    return write_repair_draft_shard(
        destination,
        instances=(*existing_instances, *new_instances),
        drafts=(*existing_drafts, *new_drafts),
        manifest_path=manifest_destination,
    )
