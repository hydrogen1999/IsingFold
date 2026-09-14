from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Mapping

import pytest

from isingfold.rl.contracts import stable_digest
from isingfold.rl.data.quality import (
    QUALITY_INITIALIZER_BANK_PROTOCOL,
    ActionQuality,
    QualityRecord,
    commit_first_policy,
    continuation_seed,
    legacy_quality_initializer_binding,
    quality_initializer_bank_contract,
    run_continuation,
    validate_publication_quality_record_initializer_binding,
)
from isingfold.rl.data.quality_resolution import QualityResolutionError
from isingfold.rl.data.quality_resolution_plan import (
    materialize_resolution_production_plan,
)
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.initializer_bank import (
    generate_initializer_bank,
    load_initializer_bank,
    seal_initializer_bank,
)
from tests.unit.test_rl_initializer_bank import (
    _FakeLACInitializer,
    _bound_training_target,
    _build_plan,
    _config,
    _context,
    _prepared,
    _result,
    _valid_embedding,
)
from tests.unit.test_rl_quality_resolution_plan import _planning_inputs


def _resign(value: Mapping[str, object], **changes: object) -> dict[str, object]:
    body = {key: item for key, item in value.items() if key != "record_digest"}
    body.update(changes)
    return {**body, "record_digest": stable_digest(body)}


def _receipt_with_binding(
    receipt: Mapping[str, object], binding: Mapping[str, object]
) -> dict[str, object]:
    return _resign(receipt, initializer_binding=dict(binding))


def _quality_record(
    *,
    receipts: tuple[Mapping[str, object], ...],
    lineage: str = "lineage-0",
) -> QualityRecord:
    rewards = tuple(float(receipt["reward"]) for receipt in receipts)
    valid = tuple(bool(receipt["returned_valid"]) for receipt in receipts)
    seeds = tuple(int(receipt["continuation_seed"]) for receipt in receipts)
    action = ActionQuality(
        action_index=0,
        payload_key="commit",
        opcode="COMMIT",
        q_mu=sum(rewards) / len(rewards),
        continuations=len(receipts),
        valid_returns=sum(valid),
        inclusion_probability=1.0,
        selected_payload_digest="a" * 64,
        applied_action_record_digest=None,
        continuation_seeds=seeds,
        continuation_rewards=rewards,
        continuation_valid=valid,
        continuation_receipts=receipts,
    )
    return QualityRecord(
        instance="task-0",
        lineage=lineage,
        prefix=(),
        support=(),
        evaluated=(action,),
        continuation_policy="uniform-exact-legal-support-v1",
        action_envelope_record_digest="b" * 64,
    )


def _sealed_test_bank(root: Path, *, episode_count: int = 1):
    prepared = [_prepared()]
    loader_backend = _FakeLACInitializer([])
    plan = _build_plan(
        prepared,
        loader_backend,
        episode_count=episode_count,
        max_draws_per_episode=1,
    )
    generation_backend = _FakeLACInitializer(
        [_result(_valid_embedding()) for _ in range(6 * episode_count)]
    )
    generate_initializer_bank(
        root,
        plan,
        prepared,
        initializer=generation_backend,
        runtime_implementation_manifest=generation_backend.runtime_implementation_manifest,
        config=_config(),
        context=_context(),
        allow_test_backend=True,
    )
    manifest_sha256 = seal_initializer_bank(root, plan)
    bank = load_initializer_bank(
        root,
        expected_manifest_sha256=manifest_sha256,
        prepared_tasks=prepared,
        prepared_manifest_sha256=plan.prepared_manifest_sha256,
        initializer=loader_backend,
        runtime_implementation_manifest=loader_backend.runtime_implementation_manifest,
        config=_config(),
        context=_context(),
        allow_test_backend=True,
    )
    return prepared[0], bank, manifest_sha256


def _banked_receipt(public, bank, manifest_sha256: str, *, continuation_seed_value: int):
    training, target_access, ground_receipt = _bound_training_target(
        public,
        prepared_manifest_sha256=bank.plan.prepared_manifest_sha256,
    )
    bootstrap = bank.bootstrap_outcome(0)
    outcome = run_continuation(
        training.task,
        _context(),
        initializer=None,
        initializer_bank=bank,
        expected_initializer_bank_manifest_sha256=manifest_sha256,
        initializer_bank_episode_index=0,
        prepared_task=training,
        target_access=target_access,
        ground_partition_receipt=ground_receipt,
        allow_test_initializer_bank=True,
        selector=fixed_strength_selector(),
        prefix=(),
        policy=commit_first_policy,
        seed=bootstrap.initial_snapshot.system_seed,
        continuation_seed=continuation_seed_value,
        reward_reads=8,
        max_steps=2,
    )
    assert outcome is not None
    assert outcome.receipt is not None
    return outcome.receipt


def _publication_contract_and_binding(public, bank, manifest_sha256: str):
    receipt = _banked_receipt(
        public,
        bank,
        manifest_sha256,
        continuation_seed_value=7001,
    )
    contract = quality_initializer_bank_contract(
        bank,
        expected_manifest_sha256=manifest_sha256,
        allow_test_bank=True,
    )
    publication_contract = _resign(contract, publication_eligible=True)
    publication_binding = _resign(
        receipt["initializer_binding"],
        publication_eligible=True,
        bank_contract_record_digest=publication_contract["record_digest"],
    )
    return receipt, publication_contract, publication_binding


def test_quality_bank_contract_requires_the_external_manifest_pin_and_k2_cache(
    tmp_path: Path,
) -> None:
    public, bank, manifest_sha256 = _sealed_test_bank(tmp_path)

    with pytest.raises(ValueError, match="manifest.*pin"):
        quality_initializer_bank_contract(
            bank,
            expected_manifest_sha256="f" * 64,
            allow_test_bank=True,
        )
    with pytest.raises(ValueError, match="publication-eligible"):
        quality_initializer_bank_contract(
            bank,
            expected_manifest_sha256=manifest_sha256,
        )

    contract = quality_initializer_bank_contract(
        bank,
        expected_manifest_sha256=manifest_sha256,
        allow_test_bank=True,
    )

    assert contract["protocol"] == QUALITY_INITIALIZER_BANK_PROTOCOL
    assert contract["manifest_sha256"] == manifest_sha256
    assert contract["prepared_manifest_sha256"] == bank.plan.prepared_manifest_sha256
    assert contract["restart_cache_slots_per_episode"] == 2
    assert contract["opened_evaluator_data"] is False
    encoded = json.dumps(contract, sort_keys=True)
    assert "ground_energy" not in encoded
    assert "reference_energy" not in encoded
    assert public.task_id not in contract


def test_quality_bank_contract_rejects_an_access_receipt_that_claims_target_access(
    tmp_path: Path,
) -> None:
    _public, bank, manifest_sha256 = _sealed_test_bank(tmp_path)
    bank.access_receipt = dataclasses.replace(
        bank.access_receipt,
        opened_evaluator_targets=True,
    )

    with pytest.raises(ValueError, match="evaluator targets"):
        quality_initializer_bank_contract(
            bank,
            expected_manifest_sha256=manifest_sha256,
            allow_test_bank=True,
        )


def test_banked_quality_continuation_uses_the_sealed_bootstrap_and_receipts(
    tmp_path: Path,
) -> None:
    public, bank, manifest_sha256 = _sealed_test_bank(tmp_path)
    training, target_access, ground_receipt = _bound_training_target(
        public,
        prepared_manifest_sha256=bank.plan.prepared_manifest_sha256,
    )
    bootstrap = bank.bootstrap_outcome(0)

    outcome = run_continuation(
        training.task,
        _context(),
        initializer=None,
        initializer_bank=bank,
        expected_initializer_bank_manifest_sha256=manifest_sha256,
        initializer_bank_episode_index=0,
        prepared_task=training,
        target_access=target_access,
        ground_partition_receipt=ground_receipt,
        allow_test_initializer_bank=True,
        selector=fixed_strength_selector(),
        prefix=(),
        policy=commit_first_policy,
        seed=bootstrap.initial_snapshot.system_seed,
        continuation_seed=7001,
        reward_reads=8,
        max_steps=2,
    )

    assert outcome is not None
    assert outcome.returned_valid is True
    assert outcome.receipt is not None
    binding = outcome.receipt["initializer_binding"]
    assert binding["protocol"] == QUALITY_INITIALIZER_BANK_PROTOCOL
    assert binding["episode_schedule_index"] == 0
    assert binding["bootstrap_record_digest"] == bootstrap.record_digest
    assert binding["restart_snapshot_record_digests"] == [
        snapshot.record_digest for snapshot in bootstrap.restart_cache_snapshots
    ]


def test_publication_quality_record_requires_one_exact_authenticated_bank_binding(
    tmp_path: Path,
) -> None:
    public, bank, manifest_sha256 = _sealed_test_bank(tmp_path)
    receipt, contract, binding = _publication_contract_and_binding(public, bank, manifest_sha256)
    first = _receipt_with_binding(receipt, binding)
    second = _resign(
        _receipt_with_binding(receipt, binding),
        continuation_seed=7002,
    )
    record = _quality_record(
        receipts=(first, second),
        lineage=str(binding["base_lineage"]),
    )

    validated = validate_publication_quality_record_initializer_binding(
        record,
        expected_initializer_bank_contract=contract,
        expected_initializer_binding=binding,
    )

    assert validated == binding
    assert validated["publication_eligible"] is True
    assert validated["protocol"] == QUALITY_INITIALIZER_BANK_PROTOCOL


def test_publication_quality_record_rejects_legacy_and_test_bank_bindings(
    tmp_path: Path,
) -> None:
    public, bank, manifest_sha256 = _sealed_test_bank(tmp_path)
    receipt = _banked_receipt(
        public,
        bank,
        manifest_sha256,
        continuation_seed_value=7001,
    )
    legacy = _receipt_with_binding(receipt, legacy_quality_initializer_binding())

    with pytest.raises(ValueError, match="legacy.*publication"):
        validate_publication_quality_record_initializer_binding(_quality_record(receipts=(legacy,)))
    with pytest.raises(ValueError, match="publication-eligible"):
        validate_publication_quality_record_initializer_binding(
            _quality_record(receipts=(receipt,))
        )


def test_publication_quality_record_rejects_mixed_episode_bindings(
    tmp_path: Path,
) -> None:
    public, bank, manifest_sha256 = _sealed_test_bank(tmp_path)
    receipt, _contract, binding = _publication_contract_and_binding(public, bank, manifest_sha256)
    other_binding = _resign(
        binding,
        episode_schedule_index=int(binding["episode_schedule_index"]) + 1,
        bootstrap_record_digest="c" * 64,
    )
    first = _receipt_with_binding(receipt, binding)
    second = _resign(
        _receipt_with_binding(receipt, other_binding),
        continuation_seed=7002,
    )

    with pytest.raises(ValueError, match="exactly one initializer-bank binding"):
        validate_publication_quality_record_initializer_binding(
            _quality_record(
                receipts=(first, second),
                lineage=str(binding["base_lineage"]),
            )
        )


def test_publication_quality_record_rejects_expected_bank_or_episode_mismatch(
    tmp_path: Path,
) -> None:
    public, bank, manifest_sha256 = _sealed_test_bank(tmp_path)
    receipt, contract, binding = _publication_contract_and_binding(public, bank, manifest_sha256)
    record = _quality_record(
        receipts=(_receipt_with_binding(receipt, binding),),
        lineage=str(binding["base_lineage"]),
    )
    other_contract = _resign(contract, manifest_sha256="d" * 64)
    other_binding = _resign(binding, bootstrap_record_digest="e" * 64)

    with pytest.raises(ValueError, match="expected bank contract"):
        validate_publication_quality_record_initializer_binding(
            record,
            expected_initializer_bank_contract=other_contract,
        )
    with pytest.raises(ValueError, match="expected episode binding"):
        validate_publication_quality_record_initializer_binding(
            record,
            expected_initializer_binding=other_binding,
        )


def test_publication_quality_record_rejects_empty_or_invalid_evaluated_receipts() -> None:
    empty = QualityRecord(
        instance="task-0",
        lineage="lineage-0",
        prefix=(),
        support=(),
        evaluated=(),
        continuation_policy="uniform-exact-legal-support-v1",
        action_envelope_record_digest="b" * 64,
    )
    with pytest.raises(ValueError, match="no evaluated continuation receipts"):
        validate_publication_quality_record_initializer_binding(empty)

    legacy_binding = legacy_quality_initializer_binding()
    receipt = {
        "schema": "isingfold.quality-continuation",
        "schema_version": 2,
        "continuation_seed": 1,
        "continuation_policy": "uniform-exact-legal-support-v1",
        "continuation_steps": 0,
        "action_trace": [],
        "returned_valid": False,
        "terminal_reason": "STOP_NO_VALID",
        "reward": 0.0,
        "requested_reward_reads": 8,
        "validation_receipt": {"valid": False},
        "cumulative_work": {
            "initializer_calls": 0,
            "initializer_work": 0,
            "router_expansions": 0,
            "candidate_generations": 0,
            "candidate_materializations": 0,
            "sa_reads": 0,
            "sa_sweeps": 0,
        },
        "terminal_evidence": None,
        "initializer_binding": legacy_binding,
    }
    receipt["record_digest"] = stable_digest(receipt)
    invalid = _quality_record(receipts=(receipt,))
    invalid_action = dataclasses.replace(invalid.evaluated[0], q_mu=0.5)

    with pytest.raises(ValueError, match="quality mean differs"):
        validate_publication_quality_record_initializer_binding(
            dataclasses.replace(invalid, evaluated=(invalid_action,))
        )


def test_banked_quality_continuation_rejects_legacy_initializer_fallback(
    tmp_path: Path,
) -> None:
    public, bank, manifest_sha256 = _sealed_test_bank(tmp_path)
    bootstrap = bank.bootstrap_outcome(0)

    with pytest.raises(ValueError, match="exactly one initializer mode"):
        run_continuation(
            public.task,
            _context(),
            initializer=lambda *_args: _valid_embedding(),
            initializer_bank=bank,
            expected_initializer_bank_manifest_sha256=manifest_sha256,
            initializer_bank_episode_index=0,
            allow_test_initializer_bank=True,
            selector=fixed_strength_selector(),
            prefix=(),
            seed=bootstrap.initial_snapshot.system_seed,
            reward_reads=8,
            max_steps=1,
        )


def test_resolution_materializer_fails_closed_without_a_pinned_initializer_bank(
    tmp_path: Path,
) -> None:
    public, _bank, _manifest_sha256 = _sealed_test_bank(tmp_path)

    with pytest.raises(QualityResolutionError, match="initializer bank"):
        materialize_resolution_production_plan(
            [public],
            context=_context(),
            selector=fixed_strength_selector(),
            config=_planning_inputs().config,
            source_corpus_manifest_sha256="2" * 64,
        )


def test_resolution_materializer_binds_each_state_to_one_sealed_bank_episode(
    tmp_path: Path,
) -> None:
    public, bank, manifest_sha256 = _sealed_test_bank(tmp_path)

    production = materialize_resolution_production_plan(
        [public],
        context=_context(),
        selector=fixed_strength_selector(),
        config=_planning_inputs().config,
        source_corpus_manifest_sha256=bank.plan.prepared_manifest_sha256,
        initializer_bank=bank,
        expected_initializer_bank_manifest_sha256=manifest_sha256,
        allow_test_initializer_bank=True,
    )

    assert production.initializer_bank_contract["manifest_sha256"] == manifest_sha256
    assert production.initializer_bank_contract["restart_cache_slots_per_episode"] == 2
    assert production.rows
    bootstrap = bank.bootstrap_outcome(0)
    assert all(row.initializer_bank_episode_index == 0 for row in production.rows)
    assert all(
        row.initializer_bootstrap_record_digest == bootstrap.record_digest
        for row in production.rows
    )
    assert all(
        row.environment_seed == bootstrap.initial_snapshot.system_seed for row in production.rows
    )
    for row in production.rows:
        assert row.actions
        for action in row.actions:
            assert action.continuation_seeds == tuple(
                continuation_seed(
                    row.environment_seed,
                    row.task_id,
                    row.state_fingerprint,
                    action.action_index,
                    offset,
                )
                for offset in range(128)
            )


def test_resolution_materializer_hash_assigns_multi_episode_bank_without_first_bias(
    tmp_path: Path,
) -> None:
    public, bank, manifest_sha256 = _sealed_test_bank(tmp_path, episode_count=3)

    production = materialize_resolution_production_plan(
        [public],
        context=_context(),
        selector=fixed_strength_selector(),
        config=_planning_inputs().config,
        source_corpus_manifest_sha256=bank.plan.prepared_manifest_sha256,
        initializer_bank=bank,
        expected_initializer_bank_manifest_sha256=manifest_sha256,
        allow_test_initializer_bank=True,
    )

    assigned = {row.initializer_bank_episode_index for row in production.rows}
    assert assigned == {1}
    bootstrap = bank.bootstrap_outcome(1)
    assert all(
        row.initializer_bootstrap_record_digest == bootstrap.record_digest
        for row in production.rows
    )
