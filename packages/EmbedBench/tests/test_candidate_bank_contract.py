from __future__ import annotations

import hashlib
import math
from dataclasses import FrozenInstanceError, replace

import pytest
from embedbench.candidate_bank import (
    BankManifest,
    CandidateGroup,
    CandidateRecord,
    EvaluationCurve,
    InstanceRecord,
    QualityOutcome,
    RepairAttempt,
    assign_split,
    canonical_json_bytes,
    content_digest,
    derive_attempt_id,
    derive_generation_digest,
    derive_group_id,
    read_bank,
    stable_seed,
    validate_group,
    write_bank,
)

GROUP_SEED = 17
PROTOCOL = (("attempt_slots", 2), ("generator", "destroy-repair-v1"))
INCUMBENT_CHAINS = ((0,), (1,))
EVALUATION_PROTOCOL = (
    ("decoder", "majority-v1"),
    ("noise_model", "none"),
    ("reference_energy", "known-e0-v1"),
    ("sampler", "test-sa"),
    ("sampler_version", "test-v1"),
    ("schedule", "geometric-v1"),
    ("seed_derivation", "stable-v1"),
)


def _instance(
    *,
    topology: str = "chimera-C2",
    coupling: float = -1.0,
    host_edges: tuple[tuple[int, int], ...] = ((0, 1), (1, 2)),
    permuted: bool = False,
) -> InstanceRecord:
    logical_nodes = (1, 0) if permuted else (0, 1)
    logical_edges = ((1, 0),) if permuted else ((0, 1),)
    selected_host_edges = (
        tuple((second, first) for first, second in reversed(host_edges)) if permuted else host_edges
    )
    return InstanceRecord.create(
        family="application",
        topology=topology,
        logical_nodes=logical_nodes,
        logical_edges=logical_edges,
        host_nodes=(2, 1, 0) if permuted else (0, 1, 2),
        host_edges=selected_host_edges,
        h=((1, 0.25), (0, -0.5)) if permuted else ((0, -0.5), (1, 0.25)),
        j=((1, 0, coupling),) if permuted else ((0, 1, coupling),),
        metadata=(("defect_pattern", "none"),),
    )


def _curve(
    partition: str,
    *,
    seeds: tuple[int, ...] | None = None,
    strengths: tuple[float, ...] = (1.0, 2.0),
    offset: float = 0.0,
) -> EvaluationCurve:
    selected_seeds = seeds or ((101, 102) if partition == "decision" else (201, 202))
    return EvaluationCurve(
        partition=partition,
        strengths=strengths,
        seeds=selected_seeds,
        p_solve=((0.10 + offset, 0.20 + offset), (0.40 + offset, 0.50 + offset)),
        residual_mean=((1.2, 1.1), (0.8, 0.7)),
        reads=100,
        sweeps=200,
        objective="solve_probability_then_residual-v1",
        evaluation_protocol=EVALUATION_PROTOCOL,
    )


def _group_id(instance: InstanceRecord) -> str:
    return derive_group_id(instance.instance_id, INCUMBENT_CHAINS, PROTOCOL, GROUP_SEED)


def _candidate(
    group_id: str,
    chains: tuple[tuple[int, ...], ...],
    *,
    decision: EvaluationCurve | None = None,
    audit: EvaluationCurve | None = None,
) -> CandidateRecord:
    return CandidateRecord.create(
        group_id=group_id,
        chains=chains,
        decision=decision or _curve("decision"),
        audit=audit or _curve("audit"),
    )


def _group(
    *,
    candidates: tuple[CandidateRecord, ...] | None = None,
) -> CandidateGroup:
    instance = _instance()
    group_id = _group_id(instance)
    incumbent = _candidate(group_id, INCUMBENT_CHAINS)
    selected = candidates or (
        _candidate(group_id, ((0,), (1, 2))),
        _candidate(group_id, ((0, 1), (2,))),
    )
    attempts = tuple(
        RepairAttempt(
            attempt_id=derive_attempt_id(group_id, index),
            repair_seed=stable_seed(GROUP_SEED, "repair", group_id, index),
            neighborhood=(index % 2,),
            status="valid",
            candidate_id=candidate.candidate_id,
            slot=index,
        )
        for index, candidate in enumerate(selected)
    )
    generation_digest = derive_generation_digest(
        group_id=group_id,
        instance_id=instance.instance_id,
        instance_record_digest=instance.record_digest,
        split_unit_id=instance.split_unit_id,
        group_seed=GROUP_SEED,
        protocol=PROTOCOL,
        incumbent_chains=incumbent.chains,
        candidate_chains=tuple(candidate.chains for candidate in selected),
        attempts=attempts,
        rejection_reason=None,
    )
    return CandidateGroup.create(
        group_id=group_id,
        instance_id=instance.instance_id,
        instance_record_digest=instance.record_digest,
        split_unit_id=instance.split_unit_id,
        split=assign_split(instance.split_unit_id),
        group_seed=GROUP_SEED,
        protocol=PROTOCOL,
        incumbent=incumbent,
        candidates=selected,
        attempts=attempts,
        generation_digest=generation_digest,
    )


def test_canonical_json_and_digest_ignore_mapping_insertion_order() -> None:
    left = {"outer": {"z": 3, "a": 1}, "name": "inkdrop"}
    right = {"name": "inkdrop", "outer": {"a": 1, "z": 3}}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert content_digest(left) == content_digest(right)
    assert len(content_digest(left)) == 64


def test_stable_seed_is_deterministic_and_domain_separated() -> None:
    first = stable_seed(17, "repair", "group-7", 0)

    assert first == stable_seed(17, "repair", "group-7", 0)
    assert first != stable_seed(17, "evaluation", "group-7", 0)
    assert stable_seed(17, "repair", "ab", "c") != stable_seed(17, "repair", "a", "bc")
    assert 0 <= first < 2**63


def test_split_uses_pre_topology_unit_and_is_stable_across_variants() -> None:
    chimera = _instance(topology="chimera-C2")
    pegasus = _instance(
        topology="pegasus-P2",
        host_edges=((0, 1), (0, 2), (1, 2)),
    )

    assert chimera.split_unit_id == pegasus.split_unit_id
    assert chimera.instance_id != pegasus.instance_id
    assert assign_split(chimera.split_unit_id) == assign_split(pegasus.split_unit_id)


def test_candidate_bank_uses_canonical_train_val_test_thresholds() -> None:
    # Hash buckets are 2811, 7600, and 8663 respectively.
    assert assign_split("split-case-1") == "train"
    assert assign_split("split-case-2") == "val"
    assert assign_split("split-case-0") == "test"


def test_content_ids_are_canonical_and_separate_logical_from_host_identity() -> None:
    canonical = _instance()
    reordered = _instance(permuted=True)
    changed_problem = _instance(coupling=-0.75)
    changed_host = _instance(host_edges=((0, 1), (0, 2), (1, 2)))

    assert reordered == canonical
    assert changed_problem.split_unit_id != canonical.split_unit_id
    assert changed_problem.instance_id != canonical.instance_id
    assert changed_host.split_unit_id == canonical.split_unit_id
    assert changed_host.instance_id != canonical.instance_id


def test_candidate_identity_excludes_labels_but_record_digest_commits_to_them() -> None:
    group_id = _group_id(_instance())
    chains = ((0,), (1, 2))
    original = _candidate(group_id, chains)
    relabeled = _candidate(
        group_id,
        chains,
        decision=_curve("decision", offset=0.05),
        audit=_curve("audit", offset=0.05),
    )

    assert original.candidate_id == relabeled.candidate_id
    assert original.record_digest != relabeled.record_digest


def test_quality_outcome_selects_strength_only_on_decision_partition() -> None:
    decision = _curve("decision")
    audit = _curve("audit")

    assert decision.best_strength() == 2.0
    assert decision.best_outcome() == QualityOutcome(p_solve=0.45, residual_mean=0.75)
    assert audit.outcome_at_strength(decision.best_strength()) == QualityOutcome(
        p_solve=0.45,
        residual_mean=0.75,
    )
    with pytest.raises(ValueError, match="decision partition"):
        audit.best_outcome()


def test_records_are_typed_frozen_and_canonicalize_nested_sequences() -> None:
    instance = _instance()
    curve = _curve("decision")
    group = _group()
    candidate = group.candidates[0]
    attempt = group.attempts[0]

    for record in (instance, curve, candidate, attempt, group):
        with pytest.raises(FrozenInstanceError):
            record.__setattr__(next(iter(record.__dataclass_fields__)), "tampered")
    assert isinstance(candidate.chains, tuple)
    assert all(isinstance(chain, tuple) for chain in candidate.chains)
    assert candidate.total_qubits == 3
    assert candidate.max_chain == 2


@pytest.mark.parametrize("label", [math.nan, math.inf, -math.inf])
def test_evaluation_curve_rejects_nonfinite_labels(label: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        EvaluationCurve(
            partition="decision",
            strengths=(1.0,),
            seeds=(101,),
            p_solve=((label,),),
            residual_mean=None,
            reads=100,
            sweeps=200,
            objective="solve_probability_then_residual-v1",
            evaluation_protocol=EVALUATION_PROTOCOL,
        )


@pytest.mark.parametrize("missing", ["decision", "audit"])
def test_candidate_rejects_missing_decision_or_audit_labels(missing: str) -> None:
    labels = {"decision": _curve("decision"), "audit": _curve("audit")}
    labels[missing] = None

    with pytest.raises(ValueError, match=missing):
        CandidateRecord.create(
            group_id=_group_id(_instance()),
            chains=((0,), (1, 2)),
            decision=labels["decision"],
            audit=labels["audit"],
        )


def test_group_rejects_duplicate_candidate_ids_and_embeddings() -> None:
    group_id = _group_id(_instance())
    first = _candidate(group_id, ((0,), (1, 2)))
    different = ((0, 1), (2,))
    with pytest.raises(ValueError, match="candidate_id.*derived"):
        CandidateRecord.create(
            group_id=group_id,
            candidate_id=first.candidate_id,
            chains=different,
            decision=_curve("decision"),
            audit=_curve("audit"),
        )

    duplicate_embedding = (
        first,
        _candidate(group_id, first.chains),
    )
    with pytest.raises(ValueError, match="candidate_id|embedding"):
        _group(candidates=duplicate_embedding)


def test_group_requires_two_valid_unique_nonincumbent_candidates() -> None:
    group_id = _group_id(_instance())
    with pytest.raises(ValueError, match="at least two"):
        _group(candidates=(_candidate(group_id, ((0,), (1, 2))),))


@pytest.mark.parametrize("axis", ["strengths", "decision_seeds", "audit_seeds"])
def test_group_rejects_inconsistent_evaluation_grids(axis: str) -> None:
    group_id = _group_id(_instance())
    first = _candidate(group_id, ((0,), (1, 2)))
    decision = _curve("decision")
    audit = _curve("audit")
    if axis == "strengths":
        decision = _curve("decision", strengths=(1.0, 3.0))
        audit = _curve("audit", strengths=(1.0, 3.0))
    elif axis == "decision_seeds":
        decision = _curve("decision", seeds=(101, 103))
    else:
        audit = _curve("audit", seeds=(201, 203))
    second = CandidateRecord.create(
        group_id=group_id,
        chains=((0, 1), (2,)),
        decision=decision,
        audit=audit,
    )

    with pytest.raises(ValueError, match="grid"):
        _group(candidates=(first, second))


def test_group_validation_detects_id_and_digest_tampering() -> None:
    group = _group()
    with pytest.raises(ValueError, match="candidate_id.*derived"):
        replace(group.candidates[0], candidate_id="renamed")

    tampered_candidate = replace(group.candidates[0], record_digest="0" * 64)
    with pytest.raises(ValueError, match="digest"):
        validate_group(replace(group, candidates=(tampered_candidate, group.candidates[1])))
    with pytest.raises(ValueError, match="group_id.*derived"):
        validate_group(replace(group, group_id="renamed-group"))
    tampered_attempt = replace(group.attempts[0], attempt_id="renamed-attempt")
    with pytest.raises(ValueError, match="attempt_id.*derived"):
        validate_group(replace(group, attempts=(tampered_attempt, group.attempts[1])))
    with pytest.raises(ValueError, match="digest"):
        validate_group(replace(group, record_digest="0" * 64))


def test_jsonl_bank_round_trip_and_manifest_checksum_detection(tmp_path) -> None:
    path = tmp_path / "candidate-bank.jsonl"
    instance = _instance()
    group = _group()

    manifest = write_bank(path, instances=(instance,), groups=(group,))
    loaded_instances, loaded_groups = read_bank(path, manifest)

    assert isinstance(manifest, BankManifest)
    assert loaded_instances == (instance,)
    assert loaded_groups == (group,)
    assert manifest.instance_count == 1
    assert manifest.group_count == 1
    with pytest.raises(ValueError, match="checksum"):
        read_bank(path, replace(manifest, jsonl_sha256="0" * 64))


def test_reader_rejects_a_rehashed_bank_that_changes_split_unit_identity(tmp_path) -> None:
    path = tmp_path / "candidate-bank.jsonl"
    instance = _instance()
    record = instance.to_dict()
    record["split_unit_id"] = f"logical-{'0' * 64}"
    record["record_digest"] = content_digest(
        {key: value for key, value in record.items() if key != "record_digest"}
    )
    raw = canonical_json_bytes({"kind": "instance", "record": record}) + b"\n"
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    manifest_payload = {
        "group_count": 0,
        "instance_count": 1,
        "jsonl_sha256": digest,
        "schema_version": 1,
    }
    manifest = BankManifest(
        jsonl_sha256=digest,
        instance_count=1,
        group_count=0,
        record_digest=content_digest(manifest_payload),
    )

    with pytest.raises(ValueError, match="split_unit_id.*derived|logical problem"):
        read_bank(path, manifest)
