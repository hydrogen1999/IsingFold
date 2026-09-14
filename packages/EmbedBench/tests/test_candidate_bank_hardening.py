from __future__ import annotations

import hashlib

import pytest
from embedbench.candidate_bank import (
    SCHEMA_VERSION,
    BankManifest,
    CandidateGroup,
    CandidateRecord,
    EvaluationCurve,
    InstanceRecord,
    RepairAttempt,
    assign_split,
    canonical_json_bytes,
    content_digest,
    derive_attempt_id,
    derive_generation_digest,
    derive_group_id,
    fit_oracle_policy,
    read_bank,
    replay_policy,
    resource_policy,
    write_bank,
)

EVALUATION_PROTOCOL = (
    ("decoder", "majority-v1"),
    ("noise_model", "none"),
    ("reference_energy", "known-e0-v1"),
    ("sampler", "test-sa"),
    ("sampler_version", "test-v1"),
    ("schedule", "geometric-v1"),
    ("seed_derivation", "stable-v1"),
)


def _curve(
    partition: str,
    *,
    p_solve: tuple[tuple[float, ...], ...],
    residual_mean: tuple[tuple[float, ...], ...],
    strengths: tuple[float, ...] = (1.0, 2.0),
    seeds: tuple[int, ...] | None = None,
) -> EvaluationCurve:
    if seeds is None:
        seeds = (101, 103) if partition == "decision" else (201, 203)
    return EvaluationCurve(
        partition=partition,
        strengths=strengths,
        seeds=seeds,
        p_solve=p_solve,
        residual_mean=residual_mean,
        reads=400 if partition == "decision" else 4_000,
        sweeps=2_000,
        objective="solve_probability_then_residual-v1",
        evaluation_protocol=EVALUATION_PROTOCOL,
    )


def _record(
    group_id: str,
    _candidate_alias: str,
    chains: tuple[tuple[int, ...], ...],
    *,
    decision_p: tuple[tuple[float, ...], ...] = ((0.4, 0.4), (0.3, 0.3)),
    decision_residual: tuple[tuple[float, ...], ...] = ((0.6, 0.6), (0.7, 0.7)),
    audit_p: tuple[tuple[float, ...], ...] = ((0.4, 0.4), (0.3, 0.3)),
    audit_residual: tuple[tuple[float, ...], ...] = ((0.6, 0.6), (0.7, 0.7)),
    strengths: tuple[float, ...] = (1.0, 2.0),
    features: tuple[tuple[str, float], ...] | None = None,
) -> CandidateRecord:
    return CandidateRecord.create(
        group_id=group_id,
        candidate_id=None,
        chains=chains,
        decision=_curve(
            "decision",
            p_solve=decision_p,
            residual_mean=decision_residual,
            strengths=strengths,
        ),
        audit=_curve(
            "audit",
            p_solve=audit_p,
            residual_mean=audit_residual,
            strengths=strengths,
        ),
        features=features,
    )


def _instance() -> InstanceRecord:
    return InstanceRecord.create(
        family="hardening",
        topology="chimera-C1",
        logical_nodes=(0, 1),
        logical_edges=((0, 1),),
        host_nodes=(0, 1, 2, 3),
        host_edges=((0, 1), (1, 2), (2, 3)),
        h=((0, -0.5), (1, 0.25)),
        j=((0, 1, -1.0),),
    )


def _replay_instance(offset: float) -> InstanceRecord:
    nodes = tuple(range(9))
    return InstanceRecord.create(
        family="hardening-replay",
        topology="complete-K9",
        logical_nodes=(0, 1),
        logical_edges=((0, 1),),
        host_nodes=nodes,
        host_edges=tuple((left, right) for left in nodes for right in nodes if left < right),
        h=((0, offset), (1, -offset)),
        j=((0, 1, -1.0),),
    )


def _valid_attempts(
    group_id: str, candidates: tuple[CandidateRecord, ...]
) -> tuple[RepairAttempt, ...]:
    return tuple(
        RepairAttempt(
            attempt_id=derive_attempt_id(group_id, slot),
            repair_seed=100 + slot,
            neighborhood=(slot,),
            status="valid",
            candidate_id=candidate.candidate_id,
            slot=slot,
        )
        for slot, candidate in enumerate(candidates)
    )


def _create_group(
    *,
    instance_record_digest: str = "0" * 64,
    **values,
) -> CandidateGroup:
    generation_digest = derive_generation_digest(
        group_id=values["group_id"],
        instance_id=values["instance_id"],
        instance_record_digest=instance_record_digest,
        split_unit_id=values["split_unit_id"],
        group_seed=values["group_seed"],
        protocol=values["protocol"],
        incumbent_chains=values["incumbent"].chains,
        candidate_chains=tuple(candidate.chains for candidate in values["candidates"]),
        attempts=values["attempts"],
        rejection_reason=None,
    )
    return CandidateGroup.create(
        **values,
        instance_record_digest=instance_record_digest,
        generation_digest=generation_digest,
    )


def _embedding_group(invalid_chains: tuple[tuple[int, ...], ...]) -> CandidateGroup:
    instance = _instance()
    protocol = (("attempt_slots", 2), ("name", "hardening-v1"))
    group_seed = 17
    incumbent_chains = ((0,), (1,))
    group_id = derive_group_id(
        instance.instance_id,
        incumbent_chains,
        protocol,
        group_seed,
    )
    candidates = (
        _record(group_id, "invalid", invalid_chains),
        _record(group_id, "valid-control", ((0, 1), (2,))),
    )
    return _create_group(
        group_id=group_id,
        instance_id=instance.instance_id,
        instance_record_digest=instance.record_digest,
        split_unit_id=instance.split_unit_id,
        split=assign_split(instance.split_unit_id),
        group_seed=group_seed,
        protocol=protocol,
        incumbent=_record(group_id, "incumbent", incumbent_chains),
        candidates=candidates,
        attempts=_valid_attempts(group_id, candidates),
    )


def _write_raw_bank(path, instance: InstanceRecord, group: CandidateGroup) -> BankManifest:
    rows = (
        {"kind": "instance", "record": instance.to_dict()},
        {"kind": "group", "record": group.to_dict()},
    )
    raw = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    path.write_bytes(raw)
    checksum = hashlib.sha256(raw).hexdigest()
    payload = {
        "group_count": 1,
        "instance_count": 1,
        "jsonl_sha256": checksum,
        "schema_version": SCHEMA_VERSION,
    }
    return BankManifest(
        jsonl_sha256=checksum,
        instance_count=1,
        group_count=1,
        schema_version=SCHEMA_VERSION,
        record_digest=content_digest(payload),
    )


def test_replay_evaluates_audit_at_strength_selected_on_decision_partition() -> None:
    instance = _replay_instance(0.125)
    instance_id = instance.instance_id
    split_unit_id = instance.split_unit_id
    group_seed = 7
    protocol = (("attempt_slots", 2), ("name", "strength-selection-v1"))
    incumbent_chains = ((2,), (3, 4))
    group_id = derive_group_id(instance_id, incumbent_chains, protocol, group_seed)
    selected = _record(
        group_id,
        "selected",
        ((0,), (1,)),
        decision_p=((0.80, 0.80), (0.30, 0.30)),
        audit_p=((0.20, 0.20), (0.95, 0.95)),
    )
    candidates = (
        selected,
        _record(group_id, "distractor", ((5, 6), (7, 8))),
    )
    group = _create_group(
        group_id=group_id,
        instance_id=instance_id,
        instance_record_digest=instance.record_digest,
        split_unit_id=split_unit_id,
        split=assign_split(split_unit_id),
        group_seed=group_seed,
        protocol=protocol,
        incumbent=_record(
            group_id,
            "incumbent",
            incumbent_chains,
            decision_p=((0.10, 0.10), (0.10, 0.10)),
            audit_p=((0.10, 0.10), (0.10, 0.10)),
        ),
        candidates=candidates,
        attempts=_valid_attempts(group_id, candidates),
    )

    result = replay_policy((group,), resource_policy(), seed=11, instances=(instance,))[0]

    assert result.selected_candidate_id == selected.candidate_id
    assert result.decision_accepted is True
    assert result.audit_outcome == pytest.approx(0.20)


def test_fit_oracle_uses_residual_to_break_an_all_zero_p_solve_tie() -> None:
    instance = _replay_instance(0.25)
    instance_id = instance.instance_id
    split_unit_id = instance.split_unit_id
    group_seed = 13
    protocol = (("attempt_slots", 2), ("name", "hard-floor-v1"))
    incumbent_chains = ((0,), (1, 2))
    group_id = derive_group_id(instance_id, incumbent_chains, protocol, group_seed)
    zero_p = ((0.0, 0.0),)
    incumbent = _record(
        group_id,
        "incumbent",
        incumbent_chains,
        decision_p=zero_p,
        decision_residual=((0.50, 0.50),),
        audit_p=zero_p,
        audit_residual=((0.50, 0.50),),
        strengths=(1.0,),
    )
    bad = _record(
        group_id,
        "a-bad",
        ((3,), (4, 5)),
        decision_p=zero_p,
        decision_residual=((0.80, 0.80),),
        audit_p=zero_p,
        audit_residual=((0.80, 0.80),),
        strengths=(1.0,),
    )
    good = _record(
        group_id,
        "z-good",
        ((6,), (7, 8)),
        decision_p=zero_p,
        decision_residual=((0.20, 0.20),),
        audit_p=zero_p,
        audit_residual=((0.20, 0.20),),
        strengths=(1.0,),
    )
    candidates = (bad, good)
    group = _create_group(
        group_id=group_id,
        instance_id=instance_id,
        instance_record_digest=instance.record_digest,
        split_unit_id=split_unit_id,
        split=assign_split(split_unit_id),
        group_seed=group_seed,
        protocol=protocol,
        incumbent=incumbent,
        candidates=candidates,
        attempts=_valid_attempts(group_id, candidates),
    )

    result = replay_policy((group,), fit_oracle_policy(), seed=19, instances=(instance,))[0]

    assert result.selected_candidate_id == good.candidate_id
    assert result.decision_accepted is True
    assert result.final_candidate_id == good.candidate_id


@pytest.mark.parametrize(
    "feature_name",
    ("decision_p_solve", "audit_outcome", "quality_label", "target"),
)
def test_candidate_rejects_features_that_can_carry_quality_labels(feature_name: str) -> None:
    with pytest.raises(ValueError, match="feature|label|allow"):
        _record(
            "feature-leakage",
            "candidate",
            ((0,), (1,)),
            features=((feature_name, 0.99),),
        )


def test_candidate_rejects_overlap_between_decision_and_audit_seeds() -> None:
    decision = _curve(
        "decision",
        p_solve=((0.2, 0.3), (0.4, 0.5)),
        residual_mean=((0.8, 0.7), (0.6, 0.5)),
        seeds=(10, 11),
    )
    audit = _curve(
        "audit",
        p_solve=((0.2, 0.3), (0.4, 0.5)),
        residual_mean=((0.8, 0.7), (0.6, 0.5)),
        seeds=(11, 12),
    )

    with pytest.raises(ValueError, match="seed|overlap|disjoint"):
        CandidateRecord.create(
            group_id="overlapping-seeds",
            candidate_id=None,
            chains=((0,), (1,)),
            decision=decision,
            audit=audit,
        )


def test_candidate_requires_the_same_strength_grid_for_decision_and_audit() -> None:
    decision = _curve(
        "decision",
        p_solve=((0.2, 0.3), (0.4, 0.5)),
        residual_mean=((0.8, 0.7), (0.6, 0.5)),
        strengths=(1.0, 2.0),
    )
    audit = _curve(
        "audit",
        p_solve=((0.2, 0.3), (0.4, 0.5)),
        residual_mean=((0.8, 0.7), (0.6, 0.5)),
        strengths=(1.0, 3.0),
    )

    with pytest.raises(ValueError, match="strength|grid"):
        CandidateRecord.create(
            group_id="mismatched-strengths",
            candidate_id=None,
            chains=((0,), (1,)),
            decision=decision,
            audit=audit,
        )


def test_manifest_rejects_an_unknown_schema_version() -> None:
    with pytest.raises(ValueError, match="schema|version"):
        BankManifest(
            jsonl_sha256="0" * 64,
            instance_count=0,
            group_count=0,
            schema_version=SCHEMA_VERSION + 1,
            record_digest="0" * 64,
        )


def test_candidate_deserializer_rejects_an_unknown_schema_version() -> None:
    payload = _record("schema", "candidate", ((0,), (1,))).to_dict()
    payload["schema_version"] = SCHEMA_VERSION + 1

    with pytest.raises(ValueError, match="schema|version"):
        CandidateRecord.from_dict(payload)


def test_group_deserializer_rejects_an_unknown_schema_version() -> None:
    instance_id = "schema-instance"
    split_unit_id = "schema-logical"
    group_seed = 23
    protocol = (("attempt_slots", 2), ("name", "schema-v1"))
    incumbent_chains = ((0,), (1,))
    group_id = derive_group_id(instance_id, incumbent_chains, protocol, group_seed)
    candidates = (
        _record(group_id, "candidate-a", ((0, 1), (2,))),
        _record(group_id, "candidate-b", ((0,), (1, 2))),
    )
    group = _create_group(
        group_id=group_id,
        instance_id=instance_id,
        split_unit_id=split_unit_id,
        split=assign_split(split_unit_id),
        group_seed=group_seed,
        protocol=protocol,
        incumbent=_record(group_id, "incumbent", incumbent_chains),
        candidates=candidates,
        attempts=_valid_attempts(group_id, candidates),
    )
    payload = group.to_dict()
    payload["schema_version"] = SCHEMA_VERSION + 1

    with pytest.raises(ValueError, match="schema|version"):
        CandidateGroup.from_dict(payload)


def test_instance_deserializer_rejects_an_unknown_schema_version() -> None:
    payload = _instance().to_dict()
    payload["schema_version"] = SCHEMA_VERSION + 1

    with pytest.raises(ValueError, match="schema|version"):
        InstanceRecord.from_dict(payload)


def test_candidate_digest_matches_its_exact_serialized_payload() -> None:
    group_id = "self-describing-candidate"
    payload = _record(group_id, "candidate", ((0,), (1,))).to_dict()
    digest = payload.pop("record_digest")

    assert payload["group_id"] == group_id
    assert content_digest(payload) == digest


def test_candidate_deserializer_rejects_unknown_same_version_fields() -> None:
    payload = _record("strict-schema", "candidate", ((0,), (1,))).to_dict()
    payload["hidden_audit_label"] = 0.99

    with pytest.raises(ValueError, match="unknown|field|schema"):
        CandidateRecord.from_dict(payload)


def test_evaluation_curve_requires_complete_scientific_protocol() -> None:
    incomplete = tuple(item for item in EVALUATION_PROTOCOL if item[0] != "reference_energy")

    with pytest.raises(ValueError, match="evaluation_protocol|reference energy"):
        EvaluationCurve(
            partition="decision",
            strengths=(1.0,),
            seeds=(101,),
            p_solve=((0.5,),),
            residual_mean=((0.5,),),
            reads=400,
            sweeps=2_000,
            objective="solve_probability_then_residual-v1",
            evaluation_protocol=incomplete,
        )


def test_nested_metadata_is_deeply_immutable() -> None:
    instance = InstanceRecord.create(
        family="hardening",
        topology="edge",
        logical_nodes=(0, 1),
        logical_edges=((0, 1),),
        host_nodes=(0, 1),
        host_edges=((0, 1),),
        h=((0, -0.5), (1, 0.25)),
        j=((0, 1, -1.0),),
        metadata=(("nested", {"values": [1, 2]}),),
    )
    nested = dict(instance.metadata)["nested"]

    assert not isinstance(nested, (dict, list))
    with pytest.raises((AttributeError, TypeError)):
        nested[0][1].append(3)  # type: ignore[union-attr]


@pytest.mark.parametrize(
    "invalid_chains",
    (
        ((2, 3),),
        ((0,), (99,)),
        ((0, 2), (3,)),
        ((0,), (2, 3)),
    ),
    ids=("wrong-chain-count", "qubit-outside-host", "disconnected-chain", "missing-coupler"),
)
def test_write_bank_rejects_an_invalid_embedding_against_its_instance(
    tmp_path, invalid_chains: tuple[tuple[int, ...], ...]
) -> None:
    instance = _instance()
    group = _embedding_group(invalid_chains)

    with pytest.raises(ValueError, match="embedding|chain|host|logical|coupler"):
        write_bank(tmp_path / "invalid.jsonl", instances=(instance,), groups=(group,))


@pytest.mark.parametrize(
    "invalid_chains",
    (
        ((2, 3),),
        ((0,), (99,)),
        ((0, 2), (3,)),
        ((0,), (2, 3)),
    ),
    ids=("wrong-chain-count", "qubit-outside-host", "disconnected-chain", "missing-coupler"),
)
def test_read_bank_rejects_a_rehashed_invalid_embedding(
    tmp_path, invalid_chains: tuple[tuple[int, ...], ...]
) -> None:
    instance = _instance()
    group = _embedding_group(invalid_chains)
    path = tmp_path / "externally-produced-invalid.jsonl"
    manifest = _write_raw_bank(path, instance, group)

    with pytest.raises(ValueError, match="embedding|chain|host|logical|coupler"):
        read_bank(path, manifest)


def test_group_rejects_candidate_id_that_duplicates_the_incumbent_id() -> None:
    instance_id = "instance"
    split_unit_id = "logical"
    group_seed = 29
    protocol = (("attempt_slots", 2), ("name", "duplicate-id-v1"))
    incumbent_chains = ((0,), (1,))
    group_id = derive_group_id(instance_id, incumbent_chains, protocol, group_seed)
    incumbent = _record(group_id, "incumbent", incumbent_chains)
    candidates = (
        _record(group_id, "same-as-incumbent", incumbent.chains),
        _record(group_id, "other", ((0,), (1, 2))),
    )

    with pytest.raises(ValueError, match="candidate_id|incumbent"):
        _create_group(
            group_id=group_id,
            instance_id=instance_id,
            split_unit_id=split_unit_id,
            split=assign_split(split_unit_id),
            group_seed=group_seed,
            protocol=protocol,
            incumbent=incumbent,
            candidates=candidates,
            attempts=_valid_attempts(group_id, candidates),
        )


def test_group_rejects_duplicate_repair_attempt_ids() -> None:
    instance_id = "instance"
    split_unit_id = "logical"
    group_seed = 31
    protocol = (("attempt_slots", 2), ("name", "duplicate-attempt-v1"))
    incumbent_chains = ((0,), (1,))
    group_id = derive_group_id(instance_id, incumbent_chains, protocol, group_seed)
    candidates = (
        _record(group_id, "candidate-a", ((0, 1), (2,))),
        _record(group_id, "candidate-b", ((0,), (1, 2))),
    )
    attempts = (
        RepairAttempt("attempt", 31, (0,), "valid", candidates[0].candidate_id),
        RepairAttempt("attempt", 37, (1,), "valid", candidates[1].candidate_id),
    )

    with pytest.raises(ValueError, match="attempt_id|repair attempt"):
        _create_group(
            group_id=group_id,
            instance_id=instance_id,
            split_unit_id=split_unit_id,
            split=assign_split(split_unit_id),
            group_seed=group_seed,
            protocol=protocol,
            incumbent=_record(group_id, "incumbent", incumbent_chains),
            candidates=candidates,
            attempts=attempts,
        )


def test_instance_rejects_duplicate_h_coefficients_for_one_logical_node() -> None:
    with pytest.raises(ValueError, match="h|coefficient|duplicate|exactly"):
        InstanceRecord.create(
            split_unit_id="logical/duplicate-h",
            instance_id="duplicate-h/chimera-C1",
            family="hardening",
            topology="chimera-C1",
            logical_nodes=(0, 1),
            logical_edges=((0, 1),),
            host_nodes=(0, 1),
            host_edges=((0, 1),),
            h=((0, -0.5), (0, 0.5), (1, 0.25)),
            j=((0, 1, -1.0),),
        )


def test_instance_rejects_boolean_h_node_alias() -> None:
    with pytest.raises(ValueError, match="h|integer|logical"):
        InstanceRecord.create(
            family="hardening",
            topology="edge",
            logical_nodes=(0, 1),
            logical_edges=((0, 1),),
            host_nodes=(0, 1),
            host_edges=((0, 1),),
            h=((False, -0.5), (1, 0.25)),
            j=((0, 1, -1.0),),
        )


def test_writer_rejects_duplicate_group_ids_before_writing(tmp_path) -> None:
    instance = _instance()
    group = _embedding_group(((0,), (1, 2)))

    with pytest.raises(ValueError, match="duplicate group_id"):
        write_bank(
            tmp_path / "duplicate-groups.jsonl",
            instances=(instance,),
            groups=(group, group),
        )


def test_bank_validation_rejects_attempt_neighborhood_outside_logical_graph(tmp_path) -> None:
    instance = _instance()
    group = _embedding_group(((0,), (1, 2)))
    attempts = list(group.attempts)
    attempts[0] = RepairAttempt(
        attempt_id=attempts[0].attempt_id,
        repair_seed=attempts[0].repair_seed,
        neighborhood=(99,),
        status=attempts[0].status,
        candidate_id=attempts[0].candidate_id,
        slot=attempts[0].slot,
    )
    tampered = _create_group(
        instance_record_digest=group.instance_record_digest,
        group_id=group.group_id,
        instance_id=group.instance_id,
        split_unit_id=group.split_unit_id,
        split=group.split,
        group_seed=group.group_seed,
        protocol=group.protocol,
        incumbent=group.incumbent,
        candidates=group.candidates,
        attempts=attempts,
    )

    with pytest.raises(ValueError, match="neighborhood|logical"):
        write_bank(
            tmp_path / "invalid-neighborhood.jsonl", instances=(instance,), groups=(tampered,)
        )
