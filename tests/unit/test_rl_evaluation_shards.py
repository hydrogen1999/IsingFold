"""Adversarial tests for resumable, lineage-coherent evaluation execution."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from tests.unit.evaluation_strata_support import evaluation_contract
from tests.unit.test_rl_complete_system import (
    FakeInitializer,
    _commit_initial,
    _component,
    _config,
    _partial_native_work,
    _tiny_task,
)
from tests.unit.test_rl_external_pairing import (
    _Backend,
    _learned_config,
    _selector,
    _stock_config,
    _task,
    _tuned_stock_default,
)
from isingfold.rl.complete_system import (
    CompleteInitializerResult,
    CompletePopulationIdentity,
    complete_policy_context,
    complete_system_metrics,
    run_complete_system,
    task_population_digest,
    write_complete_system_evidence,
    write_complete_system_outcomes,
    write_complete_system_receipts,
)
from isingfold.rl.complete_system import _context_digest
from isingfold.rl.contracts import Context, stable_digest
from isingfold.rl.data.import_embedbench import content_digest
from isingfold.rl.env import EmbeddingTask, fixed_strength_selector
from isingfold.rl.evaluate import EvaluationProtocolError
from isingfold.rl.evaluation_shards import (
    EvaluationWorkflow,
    build_evaluation_plan,
    identity_seed,
    learned_execution_contract,
    load_evaluation_plan,
    load_evaluation_shard,
    merge_evaluation_shards,
    publish_evaluation_shard,
    publish_merged_evaluation,
    receipt_execution_contract,
    select_tasks_for_shard,
    write_evaluation_plan,
)
from isingfold.rl.external import BackendSearchResult, SearchStatus
from isingfold.rl.evaluator import ReadBlock
from isingfold.rl.external_pairing import (
    external_complete_summary,
    run_external_complete_system,
    runtime_identity,
    write_external_complete_evidence,
    write_external_complete_outcomes,
    write_external_complete_receipts,
)
from isingfold.rl.external_tuning import ExternalTuningExecutionBinding


def _tasks(count: int, *, external: bool = False) -> tuple[EmbeddingTask, ...]:
    base = _task() if external else _tiny_task(prepared_initial=False)
    return tuple(
        dataclasses.replace(base, name=f"instance-{index:03d}", lineage=f"base-{index:03d}")
        for index in range(count)
    )


def _population(
    tasks: tuple[EmbeddingTask, ...],
    *,
    repetitions: int = 2,
    seed: int = 17,
    partition: str = "test",
) -> CompletePopulationIdentity:
    identities = tuple(sorted((task.lineage or task.name, task.name) for task in tasks))
    strata, design = evaluation_contract(identities, partition=partition)
    return CompletePopulationIdentity(
        population_id=f"sealed-{partition}-population",
        source_manifest_sha256=stable_digest({"manifest": partition}),
        task_payload_sha256=task_population_digest(tasks),
        expected_instances=identities,
        expected_repetitions=repetitions,
        evaluation_seed=seed,
        evaluation_strata=strata,
        confirmatory_design=design,
    )


def _failed_initializers(count: int) -> FakeInitializer:
    return FakeInitializer(
        [
            CompleteInitializerResult(
                SearchStatus.NO_EMBEDDING,
                None,
                elapsed_seconds=0.0,
                work=_partial_native_work(route_expansions=0),
            )
            for _ in range(count)
        ]
    )


def _compute_class(*, cluster: str = "apollo") -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "isingfold.evaluation-compute-class",
        "schema_version": 1,
        "cluster": cluster,
        "scheduler": "slurm" if cluster == "goose" else "direct",
        "slurm_partition": "gpu" if cluster == "goose" else None,
        "platform": {
            "system": "Linux",
            "release": "fixture",
            "machine": "x86_64",
            "processor": "fixture-cpu",
        },
        "inference": {
            "device_type": "cpu",
            "device_name": "fixture-cpu",
            "threads": 1,
            "deterministic": True,
        },
        "execution_environment": {
            "mode": "pinned-venv",
            "environment_lock_path": "/fixture/requirements.lock",
            "environment_lock_sha256": stable_digest({"environment": "lock"}),
            "container_runtime_path": None,
            "container_runtime_sha256": None,
            "container_image_path": None,
            "container_image_sha256": None,
            "python_executable_sha256": stable_digest({"python": "binary"}),
            "runtime_implementation_digest": stable_digest({"runtime": "registry"}),
            "source_digest": stable_digest({"evaluation": "source"}),
        },
        "publication_eligible": True,
    }
    return {**payload, "record_digest": content_digest(payload)}


def _runtime_identity(
    hostname: str = "apollo-node-a", *, cluster: str = "apollo"
) -> dict[str, object]:
    return dict(
        runtime_identity(
            runtime_platform={
                "hostname": hostname,
                "system": "Linux",
                "release": "fixture",
                "machine": "x86_64",
                "processor": "fixture-cpu",
                "logical_cpu_count": 8,
                "slurm_partition": "gpu" if cluster == "goose" else None,
            },
            inference_device_type="cpu",
            inference_device_name="fixture-cpu",
            inference_threads=1,
            deterministic=True,
        )
    )


def _compute_provenance(
    plan, hostname: str = "apollo-node-a", *, shard_index: int = 0
) -> dict[str, object]:
    cluster = plan.compute_class["cluster"]
    payload: dict[str, object] = {
        "schema": "isingfold.evaluation-compute-provenance",
        "schema_version": 1,
        "compute_class": dict(plan.compute_class),
        "compute_class_digest": plan.compute_class_digest,
        "runtime_identity": _runtime_identity(hostname, cluster=cluster),
        "hostname": hostname,
        "node_id": hostname,
        "slurm_job_id": "job-17" if cluster == "goose" else None,
        "slurm_array_task_id": str(shard_index) if cluster == "goose" else None,
    }
    return {**payload, "record_digest": content_digest(payload)}


def _learned_plan(tasks: tuple[EmbeddingTask, ...], *, shard_size: int = 2):
    population = _population(tasks)
    context = Context(qubit_cap=9)
    backend = _failed_initializers(1)
    config = _config()
    contract = learned_execution_contract(
        initializer=backend.identity.as_dict(),
        controller=_component("controller").as_dict(),
        selector=_component("selector").as_dict(),
        config_digest=config.digest,
        context_digest=_context_digest(context),
        policy_context_digest=_context_digest(complete_policy_context(context)),
        work_cap=context.caps.as_dict(),
        wallclock_cap_seconds=config.online_wallclock_seconds,
    )
    plan = build_evaluation_plan(
        workflow=EvaluationWorkflow.LEARNED_COMPLETE,
        population=population,
        run_coordinates={
            "training_seed_index": 0,
            "training_seed": 1103,
            "source_cell_id": "if-core-seed-1103",
        },
        execution_contract=contract,
        quality_authority=None,
        compute_class=_compute_class(),
        max_lineages_per_shard=shard_size,
    )
    return population, context, config, plan


def _run_learned(
    tasks: tuple[EmbeddingTask, ...],
    population: CompletePopulationIdentity,
    context: Context,
    *,
    execution_lineages: tuple[str, ...] | None = None,
):
    attempts = (
        len(tasks) * population.expected_repetitions
        if execution_lineages is None
        else sum((task.lineage or task.name) in execution_lineages for task in tasks)
        * population.expected_repetitions
    )
    return run_complete_system(
        tasks,
        context,
        _failed_initializers(attempts),
        _config(),
        controller=_commit_initial,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=population,
        seed=population.evaluation_seed,
        repetitions=population.expected_repetitions,
        execution_lineages=execution_lineages,
    )


def _target_and_ground_authority(plan):
    population = plan.population
    target_body = {
        "evidence_manifest_record_digest": stable_digest({"evidence": "record"}),
        "evidence_manifest_sha256": stable_digest({"evidence": "file"}),
        "opened_files": [
            {
                "authority_root": "prepared",
                "relative_path": "targets/test.jsonl",
                "role": "partition-targets",
                "sha256": stable_digest({"targets": "file"}),
            }
        ],
        "partition": "test",
        "prepared_manifest_record_digest": stable_digest({"manifest": "record"}),
        "prepared_manifest_sha256": population["source_manifest_sha256"],
        "publisher_attestation_digest": stable_digest({"publisher": "attestation"}),
        "publisher_id": "fixture-publisher",
        "target_authority_record_digest": stable_digest({"target": "authority"}),
        "target_count": len(population["expected_instances"]),
        "target_path": "targets/test.jsonl",
        "target_set_digest": stable_digest({"target": "set"}),
        "target_sha256": stable_digest({"target": "file"}),
    }
    target_access = {**target_body, "record_digest": content_digest(target_body)}
    global_body = {
        "ground_root": {
            "receipt_sha256": stable_digest({"ground": "root-file"}),
            "record_digest": stable_digest({"ground": "root-record"}),
            "verifier_identity_digest": stable_digest({"verifier": "identity"}),
        },
        "publication_id": "fixture-publication",
        "publisher_attestation_record_digest": target_body["publisher_attestation_digest"],
        "publisher_id": target_body["publisher_id"],
        "schema": "isingfold.global-quality-authority",
        "schema_version": 1,
        "target_authority_record_digest": target_body["target_authority_record_digest"],
    }
    global_authority = {**global_body, "record_digest": content_digest(global_body)}
    partition_body = {
        "evidence_manifest_record_digest": target_body["evidence_manifest_record_digest"],
        "evidence_manifest_sha256": target_body["evidence_manifest_sha256"],
        "ground_partition": {
            "accepted_count": target_body["target_count"],
            "instance_set_digest": content_digest(
                sorted(instance for _, instance in population["expected_instances"])
            ),
            "receipt_record_digest": stable_digest({"ground": "partition-record"}),
            "receipt_sha256": stable_digest({"ground": "partition-file"}),
        },
        "name": "test",
        "schema": "isingfold.partition-quality-authority",
        "schema_version": 1,
        "target_access_record_digest": target_access["record_digest"],
        "target_count": target_body["target_count"],
        "target_set_digest": target_body["target_set_digest"],
    }
    partition_authority = {
        **partition_body,
        "record_digest": content_digest(partition_body),
    }
    binding_body = {
        "schema": "isingfold.quality-authority-binding",
        "schema_version": 2,
        "global": global_authority,
        "evaluation_partition": partition_authority,
    }
    return target_access, {
        **binding_body,
        "record_digest": content_digest(binding_body),
    }


def test_plan_is_deterministic_lineage_coherent_and_capped_at_32(tmp_path: Path) -> None:
    tasks = _tasks(65)
    _, _, _, first = _learned_plan(tasks, shard_size=32)
    _, _, _, second = _learned_plan(tuple(reversed(tasks)), shard_size=32)

    assert first.as_dict() == second.as_dict()
    assert [len(shard.lineages) for shard in first.shards] == [32, 32, 1]
    assert all(
        {lineage for lineage, _, _ in shard.expected_pair_keys} == set(shard.lineages)
        for shard in first.shards
    )
    assert all(
        shard.array_coordinates
        == {
            "training_seed_index": 0,
            "training_seed": 1103,
            "source_cell_id": "if-core-seed-1103",
            "shard_index": shard.shard_index,
        }
        for shard in first.shards
    )

    plan_path = tmp_path / "plan.json"
    pin = write_evaluation_plan(plan_path, first)
    assert load_evaluation_plan(plan_path, expected_sha256=pin).as_dict() == first.as_dict()
    mutated = plan_path.read_bytes().replace(b"base-000", b"base-x00", 1)
    plan_path.write_bytes(mutated)
    with pytest.raises(EvaluationProtocolError, match="SHA-256 pin"):
        load_evaluation_plan(plan_path, expected_sha256=pin)


def test_select_tasks_authenticates_full_public_payload_before_slicing() -> None:
    tasks = _tasks(5)
    _, _, _, plan = _learned_plan(tasks)

    selected = select_tasks_for_shard(plan, 0, tuple(reversed(tasks)))

    assert {(task.lineage or task.name) for task in selected} == set(plan.shards[0].lineages)
    changed = dataclasses.replace(tasks[-1], name="mutated-instance")
    with pytest.raises(EvaluationProtocolError, match="task census"):
        select_tasks_for_shard(plan, 0, (*tasks[:-1], changed))


def test_learned_shards_merge_to_exact_monolithic_raw_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("isingfold.rl.complete_system.time.perf_counter", lambda: 0.0)
    tasks = _tasks(5)
    population, context, _, plan = _learned_plan(tasks)
    _, monolithic = _run_learned(tasks, population, context)
    loaded = []
    seed_by_pair = {receipt.pair_key: receipt.system_seed for receipt in monolithic}
    for shard in plan.shards:
        _, receipts = _run_learned(
            tasks,
            population,
            context,
            execution_lineages=shard.lineages,
        )
        assert dict(receipt_execution_contract(receipts[0])) == dict(plan.execution_contract)
        assert {receipt.pair_key: receipt.system_seed for receipt in receipts} == {
            key: seed_by_pair[key] for key in shard.expected_pair_keys
        }
        directory = tmp_path / f"shard-{shard.shard_index}"
        _, pin = publish_evaluation_shard(
            directory,
            plan=plan,
            shard_index=shard.shard_index,
            receipts=receipts,
            compute_provenance=_compute_provenance(plan),
        )
        loaded.append(load_evaluation_shard(directory, plan=plan, expected_receipt_sha256=pin))

    merged = merge_evaluation_shards(plan, tuple(reversed(loaded)), tasks=tasks, context=context)
    monolithic_dir = tmp_path / "monolithic"
    monolithic_dir.mkdir()
    write_complete_system_receipts(monolithic_dir / "receipts.jsonl", monolithic)
    write_complete_system_outcomes(monolithic_dir / "outcomes.jsonl", monolithic)
    write_complete_system_evidence(monolithic_dir / "terminal_evidence.jsonl", monolithic)

    assert merged.receipts_bytes == (monolithic_dir / "receipts.jsonl").read_bytes()
    assert merged.outcomes_bytes == (monolithic_dir / "outcomes.jsonl").read_bytes()
    assert merged.evidence_bytes == (monolithic_dir / "terminal_evidence.jsonl").read_bytes()
    assert merged.recomputed_report_inputs["metrics"] == complete_system_metrics(monolithic)
    published, merge_pin = publish_merged_evaluation(tmp_path / "merged", plan=plan, merged=merged)
    assert len(merge_pin) == 64
    assert published["record_digest"]


def test_shards_bind_typed_ground_authorities_without_ground_energies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("isingfold.rl.complete_system.time.perf_counter", lambda: 0.0)
    tasks = _tasks(1)
    population, context, _, plan = _learned_plan(tasks, shard_size=1)
    target_access, ground_authority = _target_and_ground_authority(plan)
    plan = dataclasses.replace(plan, quality_authority=ground_authority)
    assert plan.quality_authority == ground_authority
    assert plan.quality_authority_digest == content_digest(ground_authority)
    assert plan.publication_eligible is True
    assert "opened_files" not in str(plan.as_dict())
    _, receipts = _run_learned(
        tasks,
        population,
        context,
        execution_lineages=plan.shards[0].lineages,
    )

    directory = tmp_path / "bound-shard"
    _, pin = publish_evaluation_shard(
        directory,
        plan=plan,
        shard_index=0,
        receipts=receipts,
        target_access_receipt=target_access,
        ground_certificate_authority=ground_authority,
        compute_provenance=_compute_provenance(plan),
    )
    loaded = load_evaluation_shard(directory, plan=plan, expected_receipt_sha256=pin)

    assert dict(loaded.target_access_receipt or {}) == target_access
    assert dict(loaded.ground_certificate_authority or {}) == ground_authority
    assert "energy" not in (directory / "shard_receipt.json").read_text()

    changed = {
        **ground_authority,
        "evaluation_partition": {
            **ground_authority["evaluation_partition"],
            "name": "val",
        },
    }
    with pytest.raises(EvaluationProtocolError, match="quality-authority"):
        publish_evaluation_shard(
            tmp_path / "wrong-authority",
            plan=plan,
            shard_index=0,
            receipts=receipts,
            target_access_receipt=target_access,
            ground_certificate_authority=changed,
            compute_provenance=_compute_provenance(plan),
        )


def test_merge_accepts_distinct_goose_nodes_but_rejects_compute_class_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("isingfold.rl.complete_system.time.perf_counter", lambda: 0.0)
    tasks = _tasks(2)
    population, context, _, plan = _learned_plan(tasks, shard_size=1)
    plan = dataclasses.replace(plan, compute_class=_compute_class(cluster="goose"))
    loaded = []
    for shard, hostname in zip(plan.shards, ("goose-a", "goose-b"), strict=True):
        _, receipts = _run_learned(
            tasks,
            population,
            context,
            execution_lineages=shard.lineages,
        )
        directory = tmp_path / hostname
        _, pin = publish_evaluation_shard(
            directory,
            plan=plan,
            shard_index=shard.shard_index,
            receipts=receipts,
            compute_provenance=_compute_provenance(plan, hostname, shard_index=shard.shard_index),
        )
        loaded.append(load_evaluation_shard(directory, plan=plan, expected_receipt_sha256=pin))

    merged = merge_evaluation_shards(plan, loaded, tasks=tasks, context=context)
    assert [item["hostname"] for item in merged.compute_provenance] == [
        "goose-a",
        "goose-b",
    ]

    _, receipts = _run_learned(
        tasks,
        population,
        context,
        execution_lineages=plan.shards[0].lineages,
    )
    drift = _compute_provenance(plan, "goose-c")
    runtime = dict(drift["runtime_identity"])
    runtime_platform = dict(runtime["runtime_platform"])
    runtime_platform["processor"] = "different-cpu"
    runtime["runtime_platform"] = runtime_platform
    drift_body = {
        **drift,
        "runtime_identity": runtime,
    }
    drift_body.pop("record_digest")
    drift = {**drift_body, "record_digest": content_digest(drift_body)}
    with pytest.raises(EvaluationProtocolError, match="compute class"):
        publish_evaluation_shard(
            tmp_path / "drift",
            plan=plan,
            shard_index=0,
            receipts=receipts,
            compute_provenance=drift,
        )


def test_shard_loader_and_merge_fail_closed_on_mutation_missing_and_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("isingfold.rl.complete_system.time.perf_counter", lambda: 0.0)
    tasks = _tasks(3)
    population, context, _, plan = _learned_plan(tasks, shard_size=1)
    loaded = []
    pins = []
    for shard in plan.shards:
        _, receipts = _run_learned(tasks, population, context, execution_lineages=shard.lineages)
        directory = tmp_path / f"shard-{shard.shard_index}"
        _, pin = publish_evaluation_shard(
            directory,
            plan=plan,
            shard_index=shard.shard_index,
            receipts=receipts,
            compute_provenance=_compute_provenance(plan),
        )
        pins.append(pin)
        loaded.append(load_evaluation_shard(directory, plan=plan, expected_receipt_sha256=pin))

    with pytest.raises(EvaluationProtocolError, match="census differs"):
        merge_evaluation_shards(plan, loaded[:-1])
    with pytest.raises(EvaluationProtocolError, match="duplicate shard"):
        merge_evaluation_shards(plan, [loaded[0], loaded[0], *loaded[2:]])

    artifact = tmp_path / "shard-0" / "receipts.jsonl"
    artifact.write_bytes(artifact.read_bytes().replace(b"base-000", b"base-x00", 1))
    with pytest.raises(EvaluationProtocolError, match="artifact digest mismatch"):
        load_evaluation_shard(tmp_path / "shard-0", plan=plan, expected_receipt_sha256=pins[0])

    receipt_file = tmp_path / "shard-1" / "shard_receipt.json"
    receipt_file.write_bytes(receipt_file.read_bytes().replace(b'"workflow"', b'"workfloX"', 1))
    with pytest.raises(EvaluationProtocolError, match="SHA-256 pin"):
        load_evaluation_shard(tmp_path / "shard-1", plan=plan, expected_receipt_sha256=pins[1])


def test_merge_independently_replays_valid_terminal_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("isingfold.rl.complete_system.time.perf_counter", lambda: 0.0)
    monkeypatch.setattr(
        "isingfold.rl.complete_system.sample_program",
        lambda *args, **kwargs: ReadBlock(4096, 4096, 0.0, 0.0, 1),
    )
    tasks = _tasks(1)
    population, context, config, plan = _learned_plan(tasks, shard_size=1)
    backend = FakeInitializer(
        [
            CompleteInitializerResult(
                SearchStatus.EMBEDDING,
                {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2})},
                elapsed_seconds=0.0,
                work=_partial_native_work(route_expansions=0),
            )
            for _ in range(population.expected_repetitions)
        ]
    )
    _, receipts = run_complete_system(
        tasks,
        context,
        backend,
        config,
        controller=_commit_initial,
        selector=fixed_strength_selector(),
        controller_identity=_component("controller"),
        selector_identity=_component("selector"),
        population=population,
        seed=population.evaluation_seed,
        repetitions=population.expected_repetitions,
        execution_lineages=plan.shards[0].lineages,
    )
    assert all(receipt.terminal_evidence is not None for receipt in receipts)
    directory = tmp_path / "valid-shard"
    _, pin = publish_evaluation_shard(
        directory,
        plan=plan,
        shard_index=0,
        receipts=receipts,
        compute_provenance=_compute_provenance(plan),
    )
    loaded = load_evaluation_shard(directory, plan=plan, expected_receipt_sha256=pin)

    merged = merge_evaluation_shards(plan, [loaded], tasks=tasks, context=context)

    assert all(outcome.returned_valid for outcome in merged.outcomes)


def _run_external(
    tasks: tuple[EmbeddingTask, ...],
    population: CompletePopulationIdentity,
    *,
    tuning: ExternalTuningExecutionBinding,
    execution_lineages: tuple[str, ...] | None = None,
    hostname: str = "apollo-node-a",
    cluster: str = "apollo",
):
    context = Context(qubit_cap=4)
    return run_external_complete_system(
        tasks,
        context,
        _Backend(BackendSearchResult(SearchStatus.NO_EMBEDDING, None, 0.0)),
        _stock_config(),
        learned_config=_learned_config(),
        selector=fixed_strength_selector(),
        selector_identity=_selector(),
        population=population,
        quality_authority={"publisher": "fixture"},
        seed=population.evaluation_seed,
        repetitions=population.expected_repetitions,
        training_seed_index=0,
        training_seed=1103,
        compute_identity=_runtime_identity(hostname, cluster=cluster),
        tuning_execution=tuning,
        execution_lineages=execution_lineages,
    )


def test_tuned_stock_shards_have_monolithic_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("isingfold.rl.external_pairing.time.perf_counter", lambda: 0.0)
    tasks = _tasks(4, external=True)
    population = _population(tasks)
    tuning = _tuned_stock_default()
    _, monolithic = _run_external(tasks, population, tuning=tuning)
    plan = build_evaluation_plan(
        workflow=EvaluationWorkflow.TUNED_STOCK_COMPLETE,
        population=population,
        run_coordinates={"training_seed_index": 0, "training_seed": 1103},
        execution_contract=receipt_execution_contract(monolithic[0]),
        quality_authority=None,
        compute_class=_compute_class(),
        max_lineages_per_shard=2,
    )
    loaded = []
    for shard in plan.shards:
        _, receipts = _run_external(
            tasks, population, tuning=tuning, execution_lineages=shard.lineages
        )
        directory = tmp_path / f"stock-{shard.shard_index}"
        _, pin = publish_evaluation_shard(
            directory,
            plan=plan,
            shard_index=shard.shard_index,
            receipts=receipts,
            compute_provenance=_compute_provenance(plan),
        )
        loaded.append(load_evaluation_shard(directory, plan=plan, expected_receipt_sha256=pin))
    merged = merge_evaluation_shards(plan, loaded, tasks=tasks, context=Context(qubit_cap=4))
    monolithic_dir = tmp_path / "stock-monolithic"
    monolithic_dir.mkdir()
    write_external_complete_receipts(monolithic_dir / "receipts.jsonl", monolithic)
    write_external_complete_outcomes(
        monolithic_dir / "outcomes.jsonl", [receipt.outcome for receipt in monolithic]
    )
    write_external_complete_evidence(monolithic_dir / "terminal_evidence.jsonl", monolithic)

    assert merged.receipts_bytes == (monolithic_dir / "receipts.jsonl").read_bytes()
    assert merged.outcomes_bytes == (monolithic_dir / "outcomes.jsonl").read_bytes()
    assert merged.evidence_bytes == (monolithic_dir / "terminal_evidence.jsonl").read_bytes()
    assert merged.recomputed_report_inputs["summary"] == external_complete_summary(
        [receipt.outcome for receipt in monolithic], monolithic
    )


def test_external_shard_receipts_bind_exact_node_runtime_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("isingfold.rl.external_pairing.time.perf_counter", lambda: 0.0)
    tasks = _tasks(1, external=True)
    population = _population(tasks)
    tuning = _tuned_stock_default()
    _, receipts = _run_external(
        tasks,
        population,
        tuning=tuning,
        hostname="apollo-node-a",
    )
    plan = build_evaluation_plan(
        workflow=EvaluationWorkflow.TUNED_STOCK_COMPLETE,
        population=population,
        run_coordinates={"training_seed_index": 0, "training_seed": 1103},
        execution_contract=receipt_execution_contract(receipts[0]),
        quality_authority=None,
        compute_class=_compute_class(),
        max_lineages_per_shard=1,
    )

    with pytest.raises(EvaluationProtocolError, match="runtime provenance"):
        publish_evaluation_shard(
            tmp_path / "wrong-node",
            plan=plan,
            shard_index=0,
            receipts=receipts,
            compute_provenance=_compute_provenance(plan, "apollo-node-b"),
        )


def test_validation_tuning_array_is_candidate_seed_shard_and_recomputes_lineages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("isingfold.rl.external_pairing.time.perf_counter", lambda: 0.0)
    tasks = _tasks(3, external=True)
    population = _population(tasks, repetitions=1, partition="val")
    deployed = _tuned_stock_default()
    tuning = ExternalTuningExecutionBinding(
        mode="validation-candidate",
        registry_id=deployed.registry_id,
        registry_file_sha256=deployed.registry_file_sha256,
        registry_record_digest=deployed.registry_record_digest,
        candidate_index=deployed.candidate_index,
        candidate=deployed.candidate,
    )
    _, monolithic = _run_external(tasks, population, tuning=tuning)
    plan = build_evaluation_plan(
        workflow=EvaluationWorkflow.VALIDATION_TUNING,
        population=population,
        run_coordinates={
            "candidate_index": 0,
            "candidate_id": tuning.candidate.candidate_id,
            "tuning_seed_index": 0,
            "tuning_seed": 1103,
        },
        execution_contract=receipt_execution_contract(monolithic[0]),
        quality_authority=None,
        compute_class=_compute_class(cluster="goose"),
        max_lineages_per_shard=2,
    )
    assert plan.shards[0].array_coordinates == {
        "candidate_index": 0,
        "candidate_id": tuning.candidate.candidate_id,
        "tuning_seed_index": 0,
        "tuning_seed": 1103,
        "shard_index": 0,
    }
    assert monolithic[0].system_seed == identity_seed(
        1103,
        "external-tuning-policy",
        lineage=monolithic[0].lineage,
        instance=monolithic[0].instance,
        repetition=monolithic[0].repetition,
    )
    loaded = []
    for shard in plan.shards:
        hostname = f"goose-{shard.shard_index}"
        _, receipts = _run_external(
            tasks,
            population,
            tuning=tuning,
            execution_lineages=shard.lineages,
            hostname=hostname,
            cluster="goose",
        )
        directory = tmp_path / f"tuning-{shard.shard_index}"
        _, pin = publish_evaluation_shard(
            directory,
            plan=plan,
            shard_index=shard.shard_index,
            receipts=receipts,
            compute_provenance=_compute_provenance(plan, hostname, shard_index=shard.shard_index),
        )
        loaded.append(load_evaluation_shard(directory, plan=plan, expected_receipt_sha256=pin))
    merged = merge_evaluation_shards(plan, loaded, tasks=tasks, context=Context(qubit_cap=4))

    assert len(merged.recomputed_report_inputs["lineage_metrics"]) == 3
    assert merged.recomputed_report_inputs["summary"]["attempts"] == 3
