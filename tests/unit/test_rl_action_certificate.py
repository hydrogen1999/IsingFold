"""Conformance tests for environment-aligned structural action certificates."""

from __future__ import annotations

import json
from dataclasses import replace

import networkx as nx
import pytest

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import (
    Candidate,
    Context,
    DecisionState,
    Mode,
    Opcode,
    RestartCacheSlot,
    WorkVector,
    stable_digest,
)
from isingfold.rl.data.action_certificate import (
    APPLIED_ACTION_SCHEMA_VERSION,
    BOUND_CANDIDATE_SCHEMA,
    BOUND_CANDIDATE_SCHEMA_VERSION,
    ENVELOPE_SCHEMA_VERSION,
    ActionCertificateError,
    BoundCandidateV1,
    CertificateStatus,
    PersistentContinuationDomainV1,
    StateActionEnvelopeV1,
    apply_envelope_action,
    certify_persistent_completion,
    public_task_fingerprint,
)
from isingfold.rl.env import EmbeddingEnv, EmbeddingTask
from isingfold.rl.proposal import (
    AUTHENTICATED_RESTART_CACHE_V1,
    LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
    ProposalBatch,
    bound_successor_key,
)


def _candidate(
    chains: dict[int, frozenset[int]],
    *,
    opcode: Opcode,
    new_chains: dict[int, frozenset[int]],
    provenance: str,
) -> Candidate:
    affected = tuple(sorted(new_chains))
    successor = dict(chains)
    successor.update(new_chains)
    work = WorkVector(
        decisions=1,
        compiler_calls=4,
        validator_calls=1,
        feature_work=8 * len(affected),
    )
    return Candidate(
        opcode=opcode,
        affected=affected,
        old_chains={node: chains[node] for node in affected},
        new_chains=new_chains,
        work=work,
        payload_key=bound_successor_key(successor, work, restart=False),
        proposal_work=WorkVector(route_expansions=1, materializations=1),
        routes=tuple(((qubit,), node) for node, chain in new_chains.items() for qubit in chain),
        provenance=provenance,
        branch=object(),
    )


class _StaticGenerator:
    def __init__(self, factory):
        self._factory = factory

    def generate(
        self,
        chains,
        _rng,
        *,
        restarts_left=0,
        archive=(),
        restart_cache=(),
        allowance=None,
    ):
        del restarts_left, archive, restart_cache, allowance
        candidates = tuple(self._factory(dict(chains)))
        proposal_work = WorkVector()
        for candidate in candidates:
            proposal_work += candidate.proposal_work
        return ProposalBatch(candidates, proposal_work, {"test": len(candidates)}, 0)


def _construction_env(*, impossible_demand: bool = False) -> tuple[EmbeddingEnv, EmbeddingTask]:
    logical = nx.Graph([(0, 1)]) if impossible_demand else nx.empty_graph(1)
    host = nx.empty_graph(2) if impossible_demand else nx.path_graph(3)
    h = {node: float(node + 1) for node in logical.nodes}
    j = {(0, 1): -1.0} if impossible_demand else {}
    task = EmbeddingTask(
        "public-task",
        logical,
        host,
        LogicalProblem.from_dicts(h, j),
        ground_energy=-12345.0,
        lineage="base-lineage",
        witness={node: frozenset({0}) for node in logical.nodes},
    )
    env = EmbeddingEnv(
        task,
        Context(qubit_cap=host.number_of_nodes()),
        mode=Mode.CONSTRUCTION,
        reward_reads=4,
        seed=17,
    )
    if impossible_demand:
        env.generator = _StaticGenerator(
            lambda chains: (
                _candidate(
                    chains,
                    opcode=Opcode.PLACE,
                    new_chains={0: frozenset({0})},
                    provenance="place-zero",
                ),
            )
        )
    else:
        env.generator = _StaticGenerator(
            lambda chains: (
                _candidate(
                    chains,
                    opcode=Opcode.PLACE,
                    new_chains={0: frozenset({1})},
                    provenance="place-one",
                ),
                _candidate(
                    chains,
                    opcode=Opcode.PLACE,
                    new_chains={0: frozenset({2})},
                    provenance="place-two",
                ),
            )
        )
    return env, task


def _capture(env: EmbeddingEnv) -> tuple[DecisionState, StateActionEnvelopeV1]:
    decision = env.reset()
    assert isinstance(decision, DecisionState)
    envelope = StateActionEnvelopeV1.capture(
        env=env,
        decision=decision,
        provenance_fingerprint=stable_digest({"authority": "unit-test-public-provenance"}),
    )
    return decision, envelope


def _authenticated_restart_env() -> EmbeddingEnv:
    logical = nx.Graph([(0, 1)])
    host = nx.cycle_graph(6)
    task = EmbeddingTask(
        "authenticated-restart",
        logical,
        host,
        LogicalProblem.from_dicts({0: 0.0, 1: 0.0}, {(0, 1): -1.0}),
        ground_energy=-1.0,
        lineage="authenticated-restart-lineage",
    )
    initial = {0: frozenset({0}), 1: frozenset({1})}
    cached = {0: frozenset({2}), 1: frozenset({3})}
    slots = (
        RestartCacheSlot(
            slot_index=0,
            status="SUCCESS",
            chains=cached,
            snapshot_record_digest=stable_digest({"slot": 0}),
            attempt_receipt_root=stable_digest({"attempts": 0}),
        ),
        RestartCacheSlot(
            slot_index=1,
            status="FAILED",
            chains=None,
            snapshot_record_digest=stable_digest({"slot": 1}),
            attempt_receipt_root=stable_digest({"attempts": 1}),
        ),
    )
    return EmbeddingEnv(
        task,
        Context(qubit_cap=6, quotas={"restart": 2}),
        mode=Mode.IMPROVEMENT,
        initializer=lambda *_args: initial,
        restart_cache=slots,
        restart_cache_manifest_digest=stable_digest({"bank": "unit-test"}),
        improvement_restart_protocol=AUTHENTICATED_RESTART_CACHE_V1,
        seed=23,
    )


def test_envelope_binds_real_state_complete_support_and_no_evaluator_fields() -> None:
    env, task = _construction_env()
    decision, envelope = _capture(env)
    encoded = envelope.as_dict()

    assert encoded["schema"] == "isingfold.state-action-envelope"
    assert encoded["schema_version"] == ENVELOPE_SCHEMA_VERSION == 2
    assert encoded["task_fingerprint"] == public_task_fingerprint(task)
    assert encoded["state_fingerprint"] == decision.state_fingerprint
    assert encoded["support_fingerprint"] == decision.support_fingerprint
    assert len(encoded["candidates"]) == len(decision.candidates)
    assert [row["legal"] for row in encoded["candidates"]] == list(decision.legal_mask)
    assert encoded["record_digest"] == stable_digest(
        {key: value for key, value in encoded.items() if key != "record_digest"}
    )

    first = encoded["candidates"][0]
    assert set(first["payload"]) == {
        "affected",
        "archive_ref",
        "new_chains",
        "old_chains",
        "opcode",
        "payload_key",
        "proposal_provenance",
        "proposal_work",
        "restart_cache_after_digest",
        "restart_cache_slot",
        "routes",
        "schema",
        "schema_version",
        "target_conflict",
        "target_demand",
        "work",
    }
    assert first["payload"]["schema"] == BOUND_CANDIDATE_SCHEMA
    assert first["payload"]["schema_version"] == BOUND_CANDIDATE_SCHEMA_VERSION == 2
    raw = json.dumps(encoded, sort_keys=True)
    for forbidden in (
        "ground_energy",
        "witness",
        "initial_embedding",
        "observation",
        "random_state",
        "training_reward",
        "evaluator_counts",
        "evaluator_seed",
        "branch",
    ):
        assert forbidden not in raw

    changed_targets_only = replace(
        task,
        ground_energy=999.0,
        witness={0: frozenset({2})},
        initial_embedding={0: frozenset({0})},
    )
    assert public_task_fingerprint(changed_targets_only) == envelope.task_fingerprint


def test_authenticated_restart_certificate_binds_slot_and_post_consume_cache() -> None:
    env = _authenticated_restart_env()
    decision, envelope = _capture(env)
    restart_index = next(
        index
        for index, candidate in enumerate(decision.candidates)
        if candidate.opcode is Opcode.RESTART
    )
    source = decision.candidates[restart_index]
    bound = envelope.candidates[restart_index]

    assert source.restart_cache_slot == 0
    assert source.restart_cache_after_digest is not None
    assert bound.restart_cache_slot == source.restart_cache_slot
    assert bound.restart_cache_after_digest == source.restart_cache_after_digest
    assert bound.payload_dict()["restart_cache_slot"] == source.restart_cache_slot
    assert bound.payload_dict()["restart_cache_after_digest"] == source.restart_cache_after_digest

    applied = apply_envelope_action(envelope, restart_index)
    encoded = applied.as_dict()
    assert encoded["schema_version"] == APPLIED_ACTION_SCHEMA_VERSION == 2
    assert applied.selected_restart_cache_slot == source.restart_cache_slot
    assert applied.selected_restart_cache_after_digest == source.restart_cache_after_digest
    assert encoded["selected_restart_cache_slot"] == source.restart_cache_slot
    assert encoded["selected_restart_cache_after_digest"] == source.restart_cache_after_digest


def test_bound_candidate_rejects_partial_or_nonrestart_cache_metadata() -> None:
    env = _authenticated_restart_env()
    decision = env.reset()
    assert isinstance(decision, DecisionState)
    restart = next(
        candidate for candidate in decision.candidates if candidate.opcode is Opcode.RESTART
    )

    with pytest.raises(ActionCertificateError, match="together"):
        BoundCandidateV1.capture(
            index=0,
            candidate=replace(restart, restart_cache_after_digest=None),
            legal=True,
        )

    place_env, _task = _construction_env()
    place_decision = place_env.reset()
    assert isinstance(place_decision, DecisionState)
    place = next(
        candidate for candidate in place_decision.candidates if candidate.opcode is Opcode.PLACE
    )
    with pytest.raises(ActionCertificateError, match="valid only for RESTART"):
        BoundCandidateV1.capture(
            index=0,
            candidate=replace(
                place,
                restart_cache_slot=0,
                restart_cache_after_digest=stable_digest({"after": "forged"}),
            ),
            legal=True,
        )


def test_v1_envelope_digest_is_not_reinterpreted_as_v2() -> None:
    env = _authenticated_restart_env()
    _decision, envelope = _capture(env)
    legacy_payload = envelope.unsigned_dict()
    legacy_payload["schema_version"] = 1
    for row in legacy_payload["candidates"]:
        payload = row["payload"]
        assert isinstance(payload, dict)
        payload.pop("schema")
        payload.pop("schema_version")
        payload.pop("restart_cache_slot")
        payload.pop("restart_cache_after_digest")
        row["payload_digest"] = stable_digest(payload)
    legacy = replace(envelope, record_digest=stable_digest(legacy_payload))

    with pytest.raises(ActionCertificateError, match="digest mismatch"):
        legacy.verify_digest()


def test_resigned_restart_cache_digest_cannot_change_bound_transition() -> None:
    env = _authenticated_restart_env()
    _decision, envelope = _capture(env)
    restart_index = next(row.index for row in envelope.candidates if row.opcode is Opcode.RESTART)
    forged_row = replace(
        envelope.candidates[restart_index],
        restart_cache_after_digest="f" * 64,
    )
    rows = list(envelope.candidates)
    rows[restart_index] = forged_row
    forged_envelope = replace(envelope, candidates=tuple(rows), record_digest="0" * 64)
    forged_envelope = replace(
        forged_envelope,
        record_digest=stable_digest(forged_envelope.unsigned_dict()),
    )

    with pytest.raises(ActionCertificateError, match="payload_key"):
        apply_envelope_action(forged_envelope, restart_index)


def test_v1_applied_action_digest_is_not_reinterpreted_as_v2() -> None:
    env = _authenticated_restart_env()
    _decision, envelope = _capture(env)
    restart_index = next(row.index for row in envelope.candidates if row.opcode is Opcode.RESTART)
    applied = apply_envelope_action(envelope, restart_index)
    legacy_payload = applied.unsigned_dict()
    legacy_payload["schema_version"] = 1
    legacy_payload.pop("selected_restart_cache_slot")
    legacy_payload.pop("selected_restart_cache_after_digest")
    legacy = replace(applied, record_digest=stable_digest(legacy_payload))

    with pytest.raises(ActionCertificateError, match="digest mismatch"):
        legacy.verify_digest()


def test_envelope_rejects_a_support_payload_not_bound_by_the_environment() -> None:
    env, _task = _construction_env()
    decision = env.reset()
    assert isinstance(decision, DecisionState)
    candidates = list(decision.candidates)
    candidates[0] = replace(candidates[0], payload_key="forged")
    forged = replace(decision, candidates=tuple(candidates))

    with pytest.raises(ActionCertificateError, match="support"):
        StateActionEnvelopeV1.capture(
            env=env,
            decision=forged,
            provenance_fingerprint=stable_digest({"source": "test"}),
        )


def test_persistent_identity_rejects_process_local_object_repr() -> None:
    node = object()
    logical = nx.empty_graph()
    logical.add_node(node)
    host = nx.path_graph(2)
    task = EmbeddingTask(
        "noncanonical-node",
        logical,
        host,
        LogicalProblem.from_dicts({node: 0.0}, {}),
        ground_energy=0.0,
        lineage="noncanonical-node-lineage",
    )

    with pytest.raises(ActionCertificateError, match="canonical integer or text scalar"):
        public_task_fingerprint(task)


def test_selected_place_is_a_persistent_core_and_cannot_be_erased() -> None:
    env, task = _construction_env()
    _decision, envelope = _capture(env)
    place_two = next(
        row.index
        for row in envelope.candidates
        if row.opcode is Opcode.PLACE and row.new_chains[0] == frozenset({2})
    )

    applied = apply_envelope_action(envelope, place_two)
    domain = PersistentContinuationDomainV1.build(
        task=task,
        envelope=envelope,
        applied=applied,
        movable=(0,),
        window=frozenset(task.host.nodes),
        max_chain=1,
        qubit_cap=3,
        max_nodes=100,
    )
    certificate = certify_persistent_completion(
        task=task,
        envelope=envelope,
        applied=applied,
        domain=domain,
    )

    assert applied.chains_after[0] == frozenset({2})
    assert applied.persistent_cores[0] == frozenset({2})
    assert certificate.status is CertificateStatus.CERTIFIED_FEASIBLE
    assert certificate.feasible is True
    assert certificate.witness is not None
    assert certificate.witness[0] == frozenset({2})
    assert 2 in certificate.witness[0]
    assert 1 not in certificate.witness[0]


def test_infeasible_is_exact_only_after_exhaustion_and_node_limit_is_unknown() -> None:
    env, task = _construction_env(impossible_demand=True)
    _decision, envelope = _capture(env)
    action = next(row.index for row in envelope.candidates if row.opcode is Opcode.PLACE)
    applied = apply_envelope_action(envelope, action)

    exhausted_domain = PersistentContinuationDomainV1.build(
        task=task,
        envelope=envelope,
        applied=applied,
        movable=(0, 1),
        window=frozenset(task.host.nodes),
        max_chain=1,
        qubit_cap=2,
        max_nodes=100,
    )
    exhausted = certify_persistent_completion(
        task=task,
        envelope=envelope,
        applied=applied,
        domain=exhausted_domain,
    )
    assert exhausted.status is CertificateStatus.CERTIFIED_INFEASIBLE
    assert exhausted.feasible is False
    assert exhausted.exhausted is True

    capped_domain = PersistentContinuationDomainV1.build(
        task=task,
        envelope=envelope,
        applied=applied,
        movable=(0, 1),
        window=frozenset(task.host.nodes),
        max_chain=1,
        qubit_cap=2,
        max_nodes=1,
    )
    capped = certify_persistent_completion(
        task=task,
        envelope=envelope,
        applied=applied,
        domain=capped_domain,
    )
    assert capped.status is CertificateStatus.UNKNOWN_NODE_LIMIT
    assert capped.feasible is None
    assert capped.exhausted is False
    assert capped.nodes_explored == 1


def test_overlap_repair_is_explicitly_unsupported_not_a_negative_label() -> None:
    logical = nx.Graph([(0, 1)])
    host = nx.path_graph(10)
    problem = LogicalProblem.from_dicts({0: 0.0, 1: 0.0}, {(0, 1): -1.0})
    initial = {0: frozenset({0}), 1: frozenset({1})}
    task = EmbeddingTask(
        "overlap",
        logical,
        host,
        problem,
        ground_energy=-1.0,
        lineage="overlap-lineage",
        initial_embedding=initial,
    )
    env = EmbeddingEnv(
        task,
        Context(qubit_cap=10),
        mode=Mode.IMPROVEMENT,
        initializer=lambda *_args: initial,
        reward_reads=4,
        improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
        seed=9,
    )
    env.generator = _StaticGenerator(
        lambda chains: (
            _candidate(
                chains,
                opcode=Opcode.REWRITE_ONE,
                new_chains={0: frozenset({1})},
                provenance="deliberate-overlap",
            ),
        )
    )
    _decision, envelope = _capture(env)
    action = next(row.index for row in envelope.candidates if row.opcode is Opcode.REWRITE_ONE)
    applied = apply_envelope_action(envelope, action)
    domain = PersistentContinuationDomainV1.build(
        task=task,
        envelope=envelope,
        applied=applied,
        movable=(0, 1),
        window=frozenset(host.nodes),
        max_chain=2,
        qubit_cap=10,
        max_nodes=100,
    )

    result = certify_persistent_completion(
        task=task,
        envelope=envelope,
        applied=applied,
        domain=domain,
    )

    assert result.status is CertificateStatus.UNSUPPORTED_OVERLAP_REPAIR
    assert result.feasible is None
    assert result.exhausted is False
    assert result.nodes_explored == 0
    assert result.is_exact is False


def test_resigned_applied_action_cannot_detach_from_bound_envelope() -> None:
    env, task = _construction_env()
    _decision, envelope = _capture(env)
    place_two = next(
        row.index
        for row in envelope.candidates
        if row.opcode is Opcode.PLACE and row.new_chains[0] == frozenset({2})
    )
    genuine = apply_envelope_action(envelope, place_two)
    forged = replace(
        genuine,
        chains_after={0: frozenset({1})},
        persistent_cores={0: frozenset({1})},
        record_digest="0" * 64,
    )
    forged = replace(forged, record_digest=stable_digest(forged.unsigned_dict()))

    with pytest.raises(ActionCertificateError, match="bound envelope"):
        PersistentContinuationDomainV1.build(
            task=task,
            envelope=envelope,
            applied=forged,
            movable=(0,),
            window=frozenset(task.host.nodes),
            max_chain=1,
            qubit_cap=3,
            max_nodes=100,
        )
