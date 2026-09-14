from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import pytest

import isingfold.rl.data.quality_resolution_plan as plan_module
import isingfold.rl.data.quality_resolution_delta as delta_module
import isingfold.rl.data.quality as quality_module
from isingfold.rl.contracts import Context, stable_digest
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.quality import quality_initializer_bank_contract
from isingfold.rl.data.quality_attestation import quality_target_set_digest
from isingfold.rl.data.quality_resolution import QualityResolutionError
from isingfold.rl.data.quality_resolution_delta import (
    QualityResolutionExecutionAuthority,
    load_cumulative_resolution_delta_records,
    run_quality_resolution_delta_shard,
    verify_quality_resolution_delta_shard,
)
from isingfold.rl.data.quality_resolution_plan import ResolutionProductionPlan
from isingfold.rl.env import fixed_strength_selector
from isingfold.rl.initializer_bank import (
    build_initializer_bank_plan,
    generate_initializer_bank,
    load_initializer_bank,
    seal_initializer_bank,
)
from tests.unit.quality_receipt_support import fake_continuation_result
from tests.unit.test_rl_quality_resolution_plan import (
    _planning_inputs,
    _production,
)
from tests.unit.test_rl_initializer_bank import (
    _FakeLACInitializer,
    _config as _initializer_config,
    _prepared as _initializer_prepared,
    _result as _initializer_result,
    _valid_embedding,
)


_STRICT_ROW_VALIDATOR = delta_module._validate_planned_rows


def _jsonable(value: object) -> object:
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _jsonable(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}
    if hasattr(value, "items"):
        return {str(key): _jsonable(item) for key, item in sorted(value.items())}  # type: ignore[union-attr]
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "value"):
        return value.value  # type: ignore[union-attr]
    return value


def _record(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "record_digest": content_digest(payload)}


def _context() -> Context:
    return Context(qubit_cap=4, n_est_reads=256, num_sweeps=1)


def _module_sha(module: object) -> str:
    path = Path(str(module.__file__)).resolve()  # type: ignore[union-attr]
    return hashlib.sha256(path.read_bytes()).hexdigest()


QUALITY_CONTRACT_DIGEST = content_digest({"contract": "quality-v7-fixture"})


def _plan_authority():
    from isingfold.rl.data.quality_resolution_plan import QualityResolutionPlanAuthority

    context = _jsonable(_context())
    assert isinstance(context, dict)
    implementation = {
        "planner": "quality-resolution-plan-v1",
        "quality_implementation_contract_digest": QUALITY_CONTRACT_DIGEST,
        "quality_module_sha256": _module_sha(quality_module),
        "quality_resolution_delta_module_sha256": _module_sha(delta_module),
    }
    return QualityResolutionPlanAuthority(
        prepared_manifest_sha256="2" * 64,
        prepared_manifest_record_digest="3" * 64,
        prepared_train_census_record_digest=_production().source_census.record_digest,
        corpus_design_manifest_sha256="4" * 64,
        publisher_id="fixture-publisher",
        publisher_attestation_sha256="5" * 64,
        publisher_attestation_record_digest="6" * 64,
        target_authority_record_digest="7" * 64,
        ground_root_sha256="8" * 64,
        ground_root_record_digest="9" * 64,
        verifier_identity_digest="a" * 64,
        selector_digest="b" * 64,
        selector_file_sha256="c" * 64,
        selector_fit_receipt_sha256="d" * 64,
        selector_fit_record_digest="e" * 64,
        normalizer_digest="f" * 64,
        selector_device="cpu",
        selector_device_parity_sha256="0" * 64,
        selector_device_parity_record_digest="1" * 64,
        context=context,
        context_digest=content_digest(context),
        implementation=implementation,
        implementation_digest=content_digest(implementation),
    )


def _plan():
    from isingfold.rl.data.quality_resolution_plan import build_quality_resolution_plan

    return build_quality_resolution_plan(_planning_inputs(), _plan_authority(), _production())


def _plan_sha(plan) -> str:
    return hashlib.sha256(canonical_json_bytes(plan.as_dict()) + b"\n").hexdigest()


def _public_tasks(plan, shard_index: int | None = 0):
    record = plan.as_dict()
    task_ids = (
        {row["task_id"] for row in record["production_plan"]["rows"]}
        if shard_index is None
        else set(record["shards"][shard_index]["task_ids"])
    )
    row_by_task = {row["task_id"]: row for row in record["production_plan"]["rows"]}
    prepared = []
    for task_id in sorted(task_ids):
        row = row_by_task[task_id]
        prepared.append(
            _initializer_prepared(
                task_id=task_id,
                instance_id=row["instance_id"],
                lineage=row["base_lineage"],
            )
        )
    return prepared


def _assigned_tasks(plan, shard_index: int = 0):
    prepared = [
        replace(
            public,
            task=replace(public.task, ground_energy=-2.75),
            reference_status="exact_enumeration",
            certificate_digest=hashlib.sha256(f"certificate:{public.task_id}".encode()).hexdigest(),
            evaluator_protocol_digest=hashlib.sha256(
                f"protocol:{public.task_id}".encode()
            ).hexdigest(),
        )
        for public in _public_tasks(plan, shard_index)
    ]
    target_rows = _target_rows_for_tasks(prepared)
    target_set_digest = quality_target_set_digest(target_rows, partition="train")
    return [
        replace(
            item,
            quality_attestation_digest=plan.as_dict()["publisher"]["attestation_record_digest"],
            quality_evidence_manifest_digest="a" * 64,
            quality_evidence_manifest_sha256="b" * 64,
            quality_target_set_digest=target_set_digest,
            quality_target_count=len(target_rows),
        )
        for item in prepared
    ]


def _public_bank_tasks(plan):
    return _public_tasks(plan, None)


def _banked_plan(tmp_path: Path):
    """Build a real test-only K=2 bank covering the synthetic plan census."""

    from isingfold.rl.data.quality_resolution_plan import (
        build_quality_resolution_plan,
    )

    provisional = _plan()
    public = _public_bank_tasks(provisional)
    loader_backend = _FakeLACInitializer([])
    bank_plan = build_initializer_bank_plan(
        public,
        prepared_manifest_sha256="2" * 64,
        training_seed=1103,
        episode_schedule_start=0,
        episode_count=len(public),
        max_draws_per_conditional_episode=1,
        initializer=loader_backend,
        runtime_implementation_manifest=loader_backend.runtime_implementation_manifest,
        config=_initializer_config(),
        context=_context(),
        allow_test_backend=True,
    )
    generation_backend = _FakeLACInitializer(
        [_initializer_result(_valid_embedding()) for _ in range(6 * len(public))]
    )
    bank_root = tmp_path / "initializer-bank"
    generate_initializer_bank(
        bank_root,
        bank_plan,
        public,
        initializer=generation_backend,
        runtime_implementation_manifest=generation_backend.runtime_implementation_manifest,
        config=_initializer_config(),
        context=_context(),
        allow_test_backend=True,
    )
    manifest_sha256 = seal_initializer_bank(bank_root, bank_plan)
    bank = load_initializer_bank(
        bank_root,
        expected_manifest_sha256=manifest_sha256,
        prepared_tasks=public,
        prepared_manifest_sha256="2" * 64,
        initializer=loader_backend,
        runtime_implementation_manifest=loader_backend.runtime_implementation_manifest,
        config=_initializer_config(),
        context=_context(),
        allow_test_backend=True,
    )
    contract = quality_initializer_bank_contract(
        bank,
        expected_manifest_sha256=manifest_sha256,
        allow_test_bank=True,
    )
    base = _production()
    schedule = {lineage.base_lineage: lineage.schedule_index for lineage in base.lineages}
    episode_by_instance = {
        episode.instance_id: episode.episode_schedule_index for episode in bank.plan.episodes
    }
    live_rows = []
    for item in public:
        episode_index = episode_by_instance[item.instance_id]
        bootstrap = bank.bootstrap_outcome(episode_index)
        provenance = plan_module.quality_action_provenance_fingerprint(
            item,
            "2" * 64,
        )
        replayed = quality_module.replay_decision_with_action_envelope(
            item.task,
            _context(),
            initializer=None,
            initializer_bank=bank,
            expected_initializer_bank_manifest_sha256=manifest_sha256,
            initializer_bank_episode_index=episode_index,
            allow_test_initializer_bank=True,
            selector=fixed_strength_selector(),
            prefix=(),
            seed=bootstrap.initial_snapshot.system_seed,
            reward_reads=_context().n_est_reads,
            provenance_fingerprint=provenance,
        )
        assert replayed is not None
        decision, envelope = replayed
        action_index = next(
            index
            for index, candidate in enumerate(envelope.candidates)
            if candidate.legal and candidate.opcode.value == "COMMIT"
        )
        bound = envelope.candidates[action_index]
        action = plan_module.PlannedResolutionAction(
            action_index=action_index,
            payload_key=bound.payload_key,
            opcode=bound.opcode.value,
            selected_payload_digest=bound.payload_digest,
            applied_action_record_digest=None,
            continuation_seeds=tuple(
                quality_module.continuation_seed(
                    bootstrap.initial_snapshot.system_seed,
                    item.task_id,
                    decision.state_fingerprint,
                    action_index,
                    offset,
                )
                for offset in range(128)
            ),
        )
        live_rows.append(
            plan_module.PlannedResolutionRow.create(
                base_lineage=item.task.lineage,
                task_id=item.task_id,
                instance_id=item.instance_id,
                lineage_schedule_index=schedule[item.task.lineage],
                state_schedule_index=0,
                environment_seed=bootstrap.initial_snapshot.system_seed,
                initializer_bank_episode_index=episode_index,
                initializer_bootstrap_record_digest=bootstrap.record_digest,
                prefix=(),
                state_fingerprint=decision.state_fingerprint,
                support_fingerprint=decision.support_fingerprint,
                action_envelope_record_digest=envelope.record_digest,
                action_provenance_fingerprint=provenance,
                actions=(action,),
            )
        )
    production = ResolutionProductionPlan(
        lineages=base.lineages,
        rows=tuple(live_rows),
        source_census=base.source_census,
        initializer_bank_contract=contract,
    )
    plan = build_quality_resolution_plan(
        _planning_inputs(),
        _plan_authority(),
        production,
    )
    return plan, bank, manifest_sha256


def _target_rows_for_tasks(tasks) -> list[dict[str, object]]:
    rows = []
    for item in sorted(tasks, key=lambda value: value.instance_id):
        assert item.provenance is not None
        rows.append(
            _record(
                {
                    "certificate_digest": item.certificate_digest,
                    "evaluator_protocol_digest": item.evaluator_protocol_digest,
                    "instance_id": item.instance_id,
                    "instance_record_digest": item.provenance.instance_record_digest,
                    "learning_partition": "train",
                    "reference_energy": item.task.ground_energy,
                    "reference_status": item.reference_status,
                    "schema": "isingfold.evaluator-target",
                    "schema_version": 2,
                }
            )
        )
    return rows


def _target_rows(plan) -> list[dict[str, object]]:
    return _target_rows_for_tasks(_assigned_tasks(plan))


def _verifier_identity(
    tmp_path: Path,
    plan,
    *,
    runtime_sha256: str = "0" * 64,
) -> tuple[Path, str]:
    registry = plan.as_dict()["implementation"]["registry"]
    payload = {
        "attestor_id": "fixture-independent-verifier",
        "continuation_runner": "isingfold.rl.data.quality.run_continuation",
        "quality_implementation_contract_digest": registry[
            "quality_implementation_contract_digest"
        ],
        "quality_module_sha256": _module_sha(quality_module),
        "schema": delta_module.QUALITY_RESOLUTION_VERIFIER_IDENTITY_SCHEMA,
        "schema_version": delta_module.QUALITY_RESOLUTION_VERIFIER_IDENTITY_VERSION,
        "verification_module_sha256": _module_sha(delta_module),
        "verification_runtime_sha256": runtime_sha256,
    }
    record = _record(payload)
    path = tmp_path / "verifier-identity.json"
    raw = canonical_json_bytes(record) + b"\n"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()


def test_verifier_identity_publisher_binds_loaded_implementation_and_runtime(
    tmp_path: Path,
) -> None:
    plan, _bank, _bank_sha256 = _banked_plan(tmp_path)
    destination = tmp_path / "published-verifier-identity.json"

    observed_sha256 = delta_module.publish_quality_resolution_verifier_identity(
        plan,
        verification_runtime_sha256="9" * 64,
        attestor_id="independent-verifier-a",
        output_path=destination,
    )

    raw = destination.read_bytes()
    record = json.loads(raw)
    registry = plan.as_dict()["implementation"]["registry"]
    assert observed_sha256 == hashlib.sha256(raw).hexdigest()
    assert record["attestor_id"] == "independent-verifier-a"
    assert record["verification_runtime_sha256"] == "9" * 64
    assert record["quality_module_sha256"] == registry["quality_module_sha256"]
    assert (
        record["verification_module_sha256"]
        == registry["quality_resolution_delta_module_sha256"]
    )
    assert record["record_digest"] == content_digest(
        {key: value for key, value in record.items() if key != "record_digest"}
    )

    loaded, loaded_sha256 = delta_module._load_verifier_identity(
        destination,
        observed_sha256,
        plan_record=plan.as_dict(),
    )
    assert loaded == record
    assert loaded_sha256 == observed_sha256


def _execution_authority(plan, *, tasks=None, target_rows=None):
    plan_record = plan.as_dict()
    tasks = _assigned_tasks(plan) if tasks is None else list(tasks)
    target_rows = _target_rows_for_tasks(tasks) if target_rows is None else list(target_rows)
    task_count = len(target_rows)
    target_raw_sha = hashlib.sha256(
        b"".join(canonical_json_bytes(row) + b"\n" for row in target_rows)
    ).hexdigest()
    target_set_digest = quality_target_set_digest(
        target_rows,
        partition="train",
    )
    target_access_payload: dict[str, object] = {
        "evidence_manifest_record_digest": "a" * 64,
        "evidence_manifest_sha256": "b" * 64,
        "opened_files": [
            {
                "authority_root": "prepared",
                "relative_path": "targets/train.jsonl",
                "role": "evaluator-targets",
                "sha256": target_raw_sha,
            },
            {
                "authority_root": "publisher",
                "relative_path": "quality-evidence/train.json",
                "role": "quality-evidence-manifest",
                "sha256": "b" * 64,
            },
        ],
        "partition": "train",
        "prepared_manifest_record_digest": plan_record["prepared_corpus"]["manifest_record_digest"],
        "prepared_manifest_sha256": plan_record["prepared_corpus"]["manifest_sha256"],
        "publisher_attestation_digest": plan_record["publisher"]["attestation_record_digest"],
        "publisher_id": plan_record["publisher"]["publisher_id"],
        "target_authority_record_digest": plan_record["publisher"][
            "target_authority_record_digest"
        ],
        "target_count": task_count,
        "target_path": "targets/train.jsonl",
        "target_set_digest": target_set_digest,
        "target_sha256": target_raw_sha,
    }
    target_access = _record(target_access_payload)
    target_by_instance = {str(row["instance_id"]): row for row in target_rows}
    ground_targets = []
    for item in sorted(tasks, key=lambda task: task.instance_id):
        assert item.provenance is not None and item.design_condition is not None
        verifier_result = _record(
            {
                "accepted": True,
                "claimed_reference_energy": item.task.ground_energy,
                "instance_id": item.instance_id,
                "reason_code": "accepted",
            }
        )
        ground_targets.append(
            {
                "artifact_path": f"certificates/train/{item.instance_id}.json",
                "artifact_sha256": item.certificate_digest,
                "artifact_size_bytes": 1,
                "base_lineage_key": item.design_condition.base_lineage_key,
                "claimed_reference_energy": item.task.ground_energy,
                "design_condition": _jsonable(item.design_condition),
                "evidence_kind": "exact",
                "instance_id": item.instance_id,
                "learning_partition": "train",
                "public_instance_record_digest": item.public_instance_record_digest,
                "reference_status": item.reference_status,
                "request_record_digest": stable_digest({"request": item.instance_id}),
                "request_sha256": stable_digest({"request-file": item.instance_id}),
                "result_record_digest": verifier_result["record_digest"],
                "result_sha256": stable_digest({"result-file": item.instance_id}),
                "status": "accepted",
                "target_record_digest": target_by_instance[item.instance_id]["record_digest"],
                "verifier_result": verifier_result,
            }
        )
    instance_set_digest = stable_digest(sorted(item.instance_id for item in tasks))
    ground_payload: dict[str, object] = {
        "authority": target_access,
        "census": {
            "accepted_count": task_count,
            "instance_set_digest": instance_set_digest,
            "target_count": task_count,
        },
        "partition": "train",
        "protocol": "isingfold-ground-certificate-isolated-runtime-v2",
        "schema": "isingfold.ground-certificate-partition",
        "schema_version": 1,
        "targets": ground_targets,
        "verifier": {"identity": "fixture"},
    }
    ground_partition = _record(ground_payload)
    ground_raw_sha = hashlib.sha256(canonical_json_bytes(ground_partition) + b"\n").hexdigest()

    global_authority = _record(
        {
            "ground_root": {
                "receipt_sha256": plan_record["ground_root"]["sha256"],
                "record_digest": plan_record["ground_root"]["record_digest"],
                "verifier_identity_digest": plan_record["ground_root"]["verifier_identity_digest"],
            },
            "publication_id": "fixture-publication",
            "publisher_attestation_record_digest": plan_record["publisher"][
                "attestation_record_digest"
            ],
            "publisher_id": plan_record["publisher"]["publisher_id"],
            "schema": "isingfold.global-quality-authority",
            "schema_version": 1,
            "target_authority_record_digest": plan_record["publisher"][
                "target_authority_record_digest"
            ],
        }
    )
    partition_authority = _record(
        {
            "evidence_manifest_record_digest": target_access["evidence_manifest_record_digest"],
            "evidence_manifest_sha256": target_access["evidence_manifest_sha256"],
            "ground_partition": {
                "accepted_count": task_count,
                "instance_set_digest": instance_set_digest,
                "receipt_record_digest": ground_partition["record_digest"],
                "receipt_sha256": ground_raw_sha,
            },
            "name": "train",
            "schema": "isingfold.partition-quality-authority",
            "schema_version": 1,
            "target_access_record_digest": target_access["record_digest"],
            "target_count": task_count,
            "target_set_digest": target_access["target_set_digest"],
        }
    )
    binding = _record(
        {
            "global": global_authority,
            "schema": "isingfold.quality-authority-binding",
            "schema_version": 2,
            "training_partition": partition_authority,
        }
    )
    execution_identity = {
        "delta_module_sha256": _module_sha(delta_module),
        "device": "cpu",
        "quality_implementation_contract_digest": QUALITY_CONTRACT_DIGEST,
        "quality_module_sha256": _module_sha(quality_module),
        "runtime_sha256": "0" * 64,
        "selector_digest": plan_record["selector"]["selector_digest"],
        "threads": 1,
    }
    return QualityResolutionExecutionAuthority(
        quality_authority=binding,
        target_access=target_access,
        ground_partition_receipt=ground_partition,
        execution_identity=execution_identity,
        execution_identity_digest=content_digest(execution_identity),
    )


class DeterministicRunner:
    def __init__(self, *, fail_after: int | None = None, reward_shift_seed: int | None = None):
        self.calls: list[dict[str, Any]] = []
        self.fail_after = fail_after
        self.reward_shift_seed = reward_shift_seed

    def __call__(self, task, context, **kwargs):
        del task, context
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("injected interruption")
        self.calls.append(dict(kwargs))
        seed = kwargs["continuation_seed"]
        reward = (seed % 4) / 4
        if seed == self.reward_shift_seed:
            reward = ((seed + 1) % 4) / 4
        return fake_continuation_result(
            continuation_seed=seed,
            reward=reward,
            reward_reads=kwargs["reward_reads"],
            prefix=kwargs["prefix"],
            chains={0: frozenset({0}), 1: frozenset({1})},
        )


@pytest.fixture(autouse=True)
def _small_phase_two_population(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plan_module, "MINIMUM_PRODUCTION_LINEAGES", 128)
    monkeypatch.setattr(delta_module, "_validate_planned_rows", lambda *_args: None)
    monkeypatch.setattr(
        plan_module,
        "validate_quality_initializer_bank_contract",
        lambda contract, **_kwargs: quality_module.validate_quality_initializer_bank_contract(
            contract,
            require_publication=False,
        ),
    )


def test_stage_one_executes_only_the_registered_half_open_delta(tmp_path: Path) -> None:
    plan, bank, bank_sha256 = _banked_plan(tmp_path)
    artifact = run_quality_resolution_delta_shard(
        plan,
        expected_plan_sha256=_plan_sha(plan),
        stage_index=1,
        shard_index=0,
        prepared=_assigned_tasks(plan),
        target_records=_target_rows(plan),
        authority=_execution_authority(plan),
        context=_context(),
        selector=fixed_strength_selector(),
        output_directory=tmp_path / "delta-stage-1",
        initializer_bank=bank,
        expected_initializer_bank_manifest_sha256=bank_sha256,
        allow_test_initializer_bank=True,
    )

    assert artifact.manifest["delta_continuation_range"] == (12, 16)
    assert artifact.manifest["cumulative_continuation_range"] == (0, 16)
    assert {row["continuation_index"] for row in artifact.records} == {12, 13, 14, 15}
    assert len(artifact.records) == 2 * 4
    assert all(
        row["continuation_seed"]
        in {
            action_seed
            for row in plan.as_dict()["production_plan"]["rows"]
            if row["row_id"] in plan.as_dict()["shards"][0]["row_ids"]
            for action in row["actions"]
            for action_seed in action["continuation_seeds"][12:16]
        }
        for row in artifact.records
    )


def test_delta_fails_closed_without_the_plan_bound_initializer_bank(
    tmp_path: Path,
) -> None:
    plan = _plan()

    with pytest.raises(QualityResolutionError, match="pinned initializer bank"):
        run_quality_resolution_delta_shard(
            plan,
            expected_plan_sha256=_plan_sha(plan),
            stage_index=0,
            shard_index=0,
            prepared=_assigned_tasks(plan),
            target_records=_target_rows(plan),
            authority=_execution_authority(plan),
            context=_context(),
            selector=fixed_strength_selector(),
            output_directory=tmp_path / "missing-bank",
        )
    assert not (tmp_path / "missing-bank").exists()


def test_strict_row_validator_replays_the_target_free_action_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, bank, bank_sha256 = _banked_plan(tmp_path)
    authority = _execution_authority(plan)
    target = _assigned_tasks(plan)[0]
    _stage, _shard, work = delta_module._delta_work(plan.as_dict(), 0, 0)
    target_work = next(item for item in work if item.row["task_id"] == target.task_id)

    def planned_sample(*_args, **_kwargs):
        indices = tuple(action["action_index"] for action in target_work.row["actions"])
        return quality_module.DeterministicActionSample(
            action_indices=indices,
            inclusion_probabilities=(1.0,) * len(indices),
            protected_commit_index=None,
        )

    monkeypatch.setattr(delta_module, "deterministic_action_sample", planned_sample)

    _STRICT_ROW_VALIDATOR(
        [target_work],
        {target.task_id: target},
        _context(),
        fixed_strength_selector(),
        "2" * 64,
        bank,
        bank_sha256,
        authority,
        True,
    )
    with pytest.raises(QualityResolutionError, match="action provenance"):
        _STRICT_ROW_VALIDATOR(
            [target_work],
            {target.task_id: replace(target, initializer_record_digest="f" * 64)},
            _context(),
            fixed_strength_selector(),
            "2" * 64,
            bank,
            bank_sha256,
            authority,
            True,
        )


def test_target_and_ground_authority_are_required_only_at_execution(tmp_path: Path) -> None:
    plan = _plan()
    authority = _execution_authority(plan)
    assert "target_access" not in (canonical_json_bytes(plan.as_dict()).decode("utf-8"))

    with pytest.raises(QualityResolutionError, match="train target access"):
        replace(
            authority,
            target_access=_record(
                {
                    **{
                        key: value
                        for key, value in authority.target_access.items()
                        if key != "record_digest"
                    },
                    "partition": "val",
                    "target_path": "targets/val.jsonl",
                }
            ),
        )


def test_target_tasks_require_complete_authenticated_quality_metadata() -> None:
    plan = _plan()
    plan_record = plan.as_dict()
    _stage, _shard, work = delta_module._delta_work(plan_record, 0, 0)

    with pytest.raises(QualityResolutionError, match="authenticated target metadata"):
        delta_module._target_tasks(
            [
                replace(
                    item,
                    quality_attestation_digest=None,
                    quality_evidence_manifest_digest=None,
                    quality_evidence_manifest_sha256=None,
                    quality_target_set_digest=None,
                    quality_target_count=None,
                )
                for item in _assigned_tasks(plan)
            ],
            work,
            _execution_authority(plan),
            _target_rows(plan),
        )


def test_publication_rejects_an_unregistered_continuation_runner_before_execution(
    tmp_path: Path,
) -> None:
    plan, bank, bank_sha256 = _banked_plan(tmp_path)
    runner = DeterministicRunner()

    with pytest.raises(QualityResolutionError, match="registered run_continuation"):
        run_quality_resolution_delta_shard(
            plan,
            expected_plan_sha256=_plan_sha(plan),
            stage_index=0,
            shard_index=0,
            prepared=_assigned_tasks(plan),
            target_records=_target_rows(plan),
            authority=_execution_authority(plan),
            context=_context(),
            selector=fixed_strength_selector(),
            output_directory=tmp_path / "forbidden-runner",
            initializer_bank=bank,
            expected_initializer_bank_manifest_sha256=bank_sha256,
            allow_test_initializer_bank=True,
            continuation_runner=runner,
        )

    assert runner.calls == []
    assert not (tmp_path / "forbidden-runner").exists()


def test_execution_rejects_target_value_tampering_before_continuation(
    tmp_path: Path,
) -> None:
    plan, bank, bank_sha256 = _banked_plan(tmp_path)
    tasks = _assigned_tasks(plan)
    tasks[0] = replace(tasks[0], task=replace(tasks[0].task, ground_energy=-99.0))

    with pytest.raises(QualityResolutionError, match="authenticated target row"):
        run_quality_resolution_delta_shard(
            plan,
            expected_plan_sha256=_plan_sha(plan),
            stage_index=1,
            shard_index=0,
            prepared=tasks,
            target_records=_target_rows(plan),
            authority=_execution_authority(plan),
            context=_context(),
            selector=fixed_strength_selector(),
            output_directory=tmp_path / "tampered-target",
            initializer_bank=bank,
            expected_initializer_bank_manifest_sha256=bank_sha256,
            allow_test_initializer_bank=True,
        )
    assert not (tmp_path / "tampered-target").exists()


def test_atomic_directory_claim_does_not_replace_a_racing_empty_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, bank, bank_sha256 = _banked_plan(tmp_path)
    destination = tmp_path / "racing-destination"
    destination.mkdir()
    original_exists = Path.exists
    first_destination_check = True

    def raced_exists(path: Path) -> bool:
        nonlocal first_destination_check
        if path == destination and first_destination_check:
            first_destination_check = False
            return False
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", raced_exists)
    with pytest.raises(FileExistsError, match="already exists"):
        run_quality_resolution_delta_shard(
            plan,
            expected_plan_sha256=_plan_sha(plan),
            stage_index=1,
            shard_index=0,
            prepared=_assigned_tasks(plan),
            target_records=_target_rows(plan),
            authority=_execution_authority(plan),
            context=_context(),
            selector=fixed_strength_selector(),
            output_directory=destination,
            initializer_bank=bank,
            expected_initializer_bank_manifest_sha256=bank_sha256,
            allow_test_initializer_bank=True,
        )
    assert list(destination.iterdir()) == []


def test_interrupted_shard_resumes_without_rewriting_completed_evidence(tmp_path: Path) -> None:
    plan, bank, bank_sha256 = _banked_plan(tmp_path)
    destination = tmp_path / "resumed"
    plan_record = plan.as_dict()
    authority = _execution_authority(plan)
    tasks = {item.task_id: item for item in _assigned_tasks(plan)}
    stage, shard, work = delta_module._delta_work(plan_record, 1, 0)
    partial = tmp_path / ".resumed.partial"
    records_directory = partial / "records"
    records_directory.mkdir(parents=True)
    context_digest = content_digest(_jsonable(_context()))
    journal = delta_module._resume_identity(
        plan_sha256=_plan_sha(plan),
        plan_record_digest=plan_record["record_digest"],
        stage=stage,
        shard=shard,
        context_digest=context_digest,
        authority=authority,
        initializer_bank_contract=plan_record["production_plan"]["initializer_bank"],
        work=work,
    )
    delta_module._publish_file(partial / "journal.json", journal)
    preserved: list[bytes] = []
    for item in work[:3]:
        row = delta_module._execute_work(
            item,
            task=tasks[str(item.row["task_id"])],
            context=_context(),
            selector=fixed_strength_selector(),
            plan_sha256=_plan_sha(plan),
            plan_record_digest=plan_record["record_digest"],
            authority=authority,
            initializer_bank=bank,
            initializer_bank_manifest_sha256=bank_sha256,
            initializer_bank_contract=plan_record["production_plan"]["initializer_bank"],
            allow_test_initializer_bank=True,
            continuation_runner=quality_module.run_continuation,
        )
        path = records_directory / f"{item.work_index:08d}.json"
        delta_module._publish_file(path, row)
        preserved.append(path.read_bytes())

    run_quality_resolution_delta_shard(
        plan,
        expected_plan_sha256=_plan_sha(plan),
        stage_index=1,
        shard_index=0,
        prepared=_assigned_tasks(plan),
        target_records=_target_rows(plan),
        authority=authority,
        context=_context(),
        selector=fixed_strength_selector(),
        output_directory=destination,
        initializer_bank=bank,
        expected_initializer_bank_manifest_sha256=bank_sha256,
        allow_test_initializer_bank=True,
    )
    fresh = run_quality_resolution_delta_shard(
        plan,
        expected_plan_sha256=_plan_sha(plan),
        stage_index=1,
        shard_index=0,
        prepared=_assigned_tasks(plan),
        target_records=_target_rows(plan),
        authority=_execution_authority(plan),
        context=_context(),
        selector=fixed_strength_selector(),
        output_directory=tmp_path / "fresh",
        initializer_bank=bank,
        expected_initializer_bank_manifest_sha256=bank_sha256,
        allow_test_initializer_bank=True,
    )

    assert preserved
    assert (destination / "records.jsonl").read_bytes() == (
        fresh.root / "records.jsonl"
    ).read_bytes()
    assert (destination / "manifest.json").read_bytes() == (
        fresh.root / "manifest.json"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        run_quality_resolution_delta_shard(
            plan,
            expected_plan_sha256=_plan_sha(plan),
            stage_index=1,
            shard_index=0,
            prepared=_assigned_tasks(plan),
            target_records=_target_rows(plan),
            authority=_execution_authority(plan),
            context=_context(),
            selector=fixed_strength_selector(),
            output_directory=destination,
            initializer_bank=bank,
            expected_initializer_bank_manifest_sha256=bank_sha256,
            allow_test_initializer_bank=True,
        )


def test_resume_rejects_a_locally_redigested_forged_compact_row(tmp_path: Path) -> None:
    plan, bank, bank_sha256 = _banked_plan(tmp_path)
    destination = tmp_path / "poisoned-resume"
    plan_record = plan.as_dict()
    authority = _execution_authority(plan)
    tasks = {item.task_id: item for item in _assigned_tasks(plan)}
    stage, shard, work = delta_module._delta_work(plan_record, 1, 0)
    partial = tmp_path / ".poisoned-resume.partial"
    records_directory = partial / "records"
    records_directory.mkdir(parents=True)
    journal = delta_module._resume_identity(
        plan_sha256=_plan_sha(plan),
        plan_record_digest=plan_record["record_digest"],
        stage=stage,
        shard=shard,
        context_digest=content_digest(_jsonable(_context())),
        authority=authority,
        initializer_bank_contract=plan_record["production_plan"]["initializer_bank"],
        work=work,
    )
    delta_module._publish_file(partial / "journal.json", journal)
    item = work[0]
    row = delta_module._execute_work(
        item,
        task=tasks[str(item.row["task_id"])],
        context=_context(),
        selector=fixed_strength_selector(),
        plan_sha256=_plan_sha(plan),
        plan_record_digest=plan_record["record_digest"],
        authority=authority,
        initializer_bank=bank,
        initializer_bank_manifest_sha256=bank_sha256,
        initializer_bank_contract=plan_record["production_plan"]["initializer_bank"],
        allow_test_initializer_bank=True,
        continuation_runner=quality_module.run_continuation,
    )
    row["continuation_receipt_digest"] = "0" * 64
    row["record_digest"] = content_digest(
        {key: value for key, value in row.items() if key != "record_digest"}
    )
    delta_module._publish_file(
        records_directory / f"{item.work_index:08d}.json",
        row,
    )

    with pytest.raises(QualityResolutionError, match="exact authenticated.*replay"):
        run_quality_resolution_delta_shard(
            plan,
            expected_plan_sha256=_plan_sha(plan),
            stage_index=1,
            shard_index=0,
            prepared=_assigned_tasks(plan),
            target_records=_target_rows(plan),
            authority=authority,
            context=_context(),
            selector=fixed_strength_selector(),
            output_directory=destination,
            initializer_bank=bank,
            expected_initializer_bank_manifest_sha256=bank_sha256,
            allow_test_initializer_bank=True,
        )
    assert not destination.exists()


def test_independent_verifier_replays_every_compact_record_and_detects_mismatch(
    tmp_path: Path,
) -> None:
    plan, bank, bank_sha256 = _banked_plan(tmp_path)
    artifact = run_quality_resolution_delta_shard(
        plan,
        expected_plan_sha256=_plan_sha(plan),
        stage_index=1,
        shard_index=0,
        prepared=_assigned_tasks(plan),
        target_records=_target_rows(plan),
        authority=_execution_authority(plan),
        context=_context(),
        selector=fixed_strength_selector(),
        output_directory=tmp_path / "delta",
        initializer_bank=bank,
        expected_initializer_bank_manifest_sha256=bank_sha256,
        allow_test_initializer_bank=True,
    )
    verifier_path, verifier_sha = _verifier_identity(tmp_path, plan)
    passed = verify_quality_resolution_delta_shard(
        artifact.root,
        expected_manifest_sha256=artifact.manifest_sha256,
        plan=plan,
        expected_plan_sha256=_plan_sha(plan),
        prepared=_assigned_tasks(plan),
        target_records=_target_rows(plan),
        authority=_execution_authority(plan),
        context=_context(),
        selector=fixed_strength_selector(),
        initializer_bank=bank,
        expected_initializer_bank_manifest_sha256=bank_sha256,
        allow_test_initializer_bank=True,
        verifier_identity_path=verifier_path,
        expected_verifier_identity_sha256=verifier_sha,
        output_path=tmp_path / "verified.json",
    )
    assert passed["pass"] is True
    assert passed["replayed_count"] == passed["matching_count"] == len(artifact.records)
    with pytest.raises(QualityResolutionError, match="verifier identity.*out-of-band pin"):
        verify_quality_resolution_delta_shard(
            artifact.root,
            expected_manifest_sha256=artifact.manifest_sha256,
            plan=plan,
            expected_plan_sha256=_plan_sha(plan),
            prepared=_assigned_tasks(plan),
            target_records=_target_rows(plan),
            authority=_execution_authority(plan),
            context=_context(),
            selector=fixed_strength_selector(),
            initializer_bank=bank,
            expected_initializer_bank_manifest_sha256=bank_sha256,
            allow_test_initializer_bank=True,
            verifier_identity_path=verifier_path,
            expected_verifier_identity_sha256="f" * 64,
            output_path=tmp_path / "untrusted-verifier.json",
        )

    changed = [dict(row) for row in artifact.records]
    changed[0]["continuation_receipt_digest"] = "0" * 64
    changed[0]["record_digest"] = content_digest(
        {key: value for key, value in changed[0].items() if key != "record_digest"}
    )
    changed_raw = b"".join(canonical_json_bytes(row) + b"\n" for row in changed)
    stage, shard, _work = delta_module._delta_work(plan.as_dict(), 1, 0)
    changed_manifest = delta_module._manifest(
        changed,
        changed_raw,
        plan_sha256=_plan_sha(plan),
        plan_record_digest=plan.as_dict()["record_digest"],
        stage=stage,
        shard=shard,
        context_digest=content_digest(_jsonable(_context())),
        authority=_execution_authority(plan),
        initializer_bank_contract=plan.as_dict()["production_plan"]["initializer_bank"],
    )
    (artifact.root / "records.jsonl").write_bytes(changed_raw)
    manifest_raw = canonical_json_bytes(changed_manifest) + b"\n"
    (artifact.root / "manifest.json").write_bytes(manifest_raw)
    failed = verify_quality_resolution_delta_shard(
        artifact.root,
        expected_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        plan=plan,
        expected_plan_sha256=_plan_sha(plan),
        prepared=_assigned_tasks(plan),
        target_records=_target_rows(plan),
        authority=_execution_authority(plan),
        context=_context(),
        selector=fixed_strength_selector(),
        initializer_bank=bank,
        expected_initializer_bank_manifest_sha256=bank_sha256,
        allow_test_initializer_bank=True,
        verifier_identity_path=verifier_path,
        expected_verifier_identity_sha256=verifier_sha,
        output_path=tmp_path / "mismatch.json",
    )
    assert failed["pass"] is False
    assert failed["matching_count"] == len(artifact.records) - 1
    assert len(failed["mismatches"]) == 1


def test_independent_verifier_rejects_a_claimed_runtime_it_is_not_using(
    tmp_path: Path,
) -> None:
    plan, bank, bank_sha256 = _banked_plan(tmp_path)
    authority = _execution_authority(plan)
    artifact = run_quality_resolution_delta_shard(
        plan,
        expected_plan_sha256=_plan_sha(plan),
        stage_index=1,
        shard_index=0,
        prepared=_assigned_tasks(plan),
        target_records=_target_rows(plan),
        authority=authority,
        context=_context(),
        selector=fixed_strength_selector(),
        output_directory=tmp_path / "delta-runtime-mismatch",
        initializer_bank=bank,
        expected_initializer_bank_manifest_sha256=bank_sha256,
        allow_test_initializer_bank=True,
    )
    verifier_path, verifier_sha = _verifier_identity(
        tmp_path,
        plan,
        runtime_sha256="9" * 64,
    )

    with pytest.raises(QualityResolutionError, match="verifier runtime.*execution runtime"):
        verify_quality_resolution_delta_shard(
            artifact.root,
            expected_manifest_sha256=artifact.manifest_sha256,
            plan=plan,
            expected_plan_sha256=_plan_sha(plan),
            prepared=_assigned_tasks(plan),
            target_records=_target_rows(plan),
            authority=authority,
            context=_context(),
            selector=fixed_strength_selector(),
            initializer_bank=bank,
            expected_initializer_bank_manifest_sha256=bank_sha256,
            allow_test_initializer_bank=True,
            verifier_identity_path=verifier_path,
            expected_verifier_identity_sha256=verifier_sha,
            output_path=tmp_path / "runtime-mismatch-verification.json",
        )


def test_cumulative_loader_requires_a_complete_stage_prefix(tmp_path: Path) -> None:
    plan, bank, bank_sha256 = _banked_plan(tmp_path)
    artifacts = []
    for stage_index in (0, 1):
        artifacts.append(
            run_quality_resolution_delta_shard(
                plan,
                expected_plan_sha256=_plan_sha(plan),
                stage_index=stage_index,
                shard_index=0,
                prepared=_assigned_tasks(plan),
                target_records=_target_rows(plan),
                authority=_execution_authority(plan),
                context=_context(),
                selector=fixed_strength_selector(),
                output_directory=tmp_path / f"delta-{stage_index}",
                initializer_bank=bank,
                expected_initializer_bank_manifest_sha256=bank_sha256,
                allow_test_initializer_bank=True,
            )
        )

    cumulative = load_cumulative_resolution_delta_records(
        [(artifact.root, artifact.manifest_sha256) for artifact in artifacts],
        plan=plan,
        expected_plan_sha256=_plan_sha(plan),
        allow_test_initializer_bank=True,
    )
    by_action: dict[tuple[str, int], list[int]] = {}
    for row in cumulative:
        by_action.setdefault((row["row_id"], row["action_index"]), []).append(
            row["continuation_index"]
        )
    assert by_action
    assert all(indices == list(range(16)) for indices in by_action.values())

    with pytest.raises(QualityResolutionError, match="complete stage prefix"):
        load_cumulative_resolution_delta_records(
            [(artifacts[1].root, artifacts[1].manifest_sha256)],
            plan=plan,
            expected_plan_sha256=_plan_sha(plan),
            allow_test_initializer_bank=True,
        )


def test_delta_loader_rejects_unknown_manifest_fields_and_directory_entries(
    tmp_path: Path,
) -> None:
    plan, bank, bank_sha256 = _banked_plan(tmp_path)
    artifact = run_quality_resolution_delta_shard(
        plan,
        expected_plan_sha256=_plan_sha(plan),
        stage_index=1,
        shard_index=0,
        prepared=_assigned_tasks(plan),
        target_records=_target_rows(plan),
        authority=_execution_authority(plan),
        context=_context(),
        selector=fixed_strength_selector(),
        output_directory=tmp_path / "strict-delta",
        initializer_bank=bank,
        expected_initializer_bank_manifest_sha256=bank_sha256,
        allow_test_initializer_bank=True,
    )
    (artifact.root / "unregistered.txt").write_text("unexpected")
    with pytest.raises(QualityResolutionError, match="directory inventory"):
        load_cumulative_resolution_delta_records(
            [(artifact.root, artifact.manifest_sha256)],
            plan=plan,
            expected_plan_sha256=_plan_sha(plan),
            allow_test_initializer_bank=True,
        )
    (artifact.root / "unregistered.txt").unlink()

    manifest = dict(artifact.manifest)
    manifest["unregistered"] = True
    manifest["record_digest"] = content_digest(
        {key: value for key, value in manifest.items() if key != "record_digest"}
    )
    raw = canonical_json_bytes(manifest) + b"\n"
    (artifact.root / "manifest.json").write_bytes(raw)
    with pytest.raises(QualityResolutionError, match="manifest schema"):
        load_cumulative_resolution_delta_records(
            [(artifact.root, hashlib.sha256(raw).hexdigest())],
            plan=plan,
            expected_plan_sha256=_plan_sha(plan),
            allow_test_initializer_bank=True,
        )
