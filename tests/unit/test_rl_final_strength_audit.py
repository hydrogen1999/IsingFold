"""Final-arm four-strength audit contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import isingfold.rl.final_strength_audit as final_audit_module
from isingfold.rl.checkpoint import runtime_implementation_registry
from isingfold.rl.complete_system import CompletePopulationIdentity
from isingfold.rl.complete_system_aggregate import AuthenticatedCompleteSystemSeedRun
from isingfold.rl.contracts import Context, stable_digest
from isingfold.rl.data.import_embedbench import content_digest
from isingfold.rl.evaluator import ReadBlock
from isingfold.rl.external_pairing import AuthenticatedExternalCompleteRun
from isingfold.rl.external_tuning import (
    EXTERNAL_TUNING_REGISTRY_FILE_SHA256,
    EXTERNAL_TUNING_REGISTRY_RECORD_DIGEST,
    ExternalTuningCandidate,
    ExternalTuningExecutionBinding,
)
from isingfold.rl.final_strength_audit import (
    ARMS,
    BLOCK_DOMAINS,
    CONFIG_SCHEMA,
    CONFIG_VERSION,
    FinalStrengthAuditExecutionManifest,
    FinalStrengthAuditError,
    build_final_strength_audit_execution_manifest,
    build_final_strength_audit_plan,
    load_final_strength_audit_config,
    load_final_strength_audit_execution_manifest,
    load_final_strength_audit_shard,
    load_final_strength_audit_plan,
    merge_final_strength_audit_shards,
    publish_final_strength_audit,
    publish_final_strength_audit_shard,
    registered_final_strength_sampler_identity,
    run_final_strength_audit,
    run_final_strength_audit_shard,
    write_final_strength_audit_execution_manifest,
    write_final_strength_audit_plan,
)
from tests.unit import test_rl_complete_system_aggregate as fixtures
from tests.unit.evaluation_strata_support import evaluation_contract


def _stock_tuning_binding() -> dict[str, object]:
    candidate = ExternalTuningCandidate(
        candidate_id="time-quality-p20-v1",
        family="time-saturating-quality",
        outer_restart_policy="until-wallclock-or-outer-cap",
        outer_restart_cap=64,
        native_parameters={
            "tries": 1,
            "threads": 1,
            "max_no_improvement": 20,
            "chainlength_patience": 20,
        },
        candidate_ranking=(
            "frozen-selector-max-p_solve-then-qubits-max_chain-"
            "embedding_digest-restart_index"
        ),
        strength_rule="same-frozen-selector-argmax-four-programs",
        online_evaluator_feedback=False,
    )
    return ExternalTuningExecutionBinding(
        mode="frozen-deployment",
        registry_id="stock-minorminer-0.2.22-validation-tuning-hybrid-v1",
        registry_file_sha256=EXTERNAL_TUNING_REGISTRY_FILE_SHA256,
        registry_record_digest=EXTERNAL_TUNING_REGISTRY_RECORD_DIGEST,
        candidate_index=6,
        candidate=candidate,
        selection_file_sha256="3" * 64,
        selection_record_digest="4" * 64,
    ).as_dict()


def _rehash(report: dict[str, object]) -> dict[str, object]:
    payload = {key: value for key, value in report.items() if key != "record_digest"}
    return {**payload, "record_digest": content_digest(payload)}


def _production_config_payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": CONFIG_SCHEMA,
        "schema_version": CONFIG_VERSION,
        "registry_id": "final-strength-four-program-split-block-v1",
        "scope": "post-freeze-diagnostic-only",
        "reads_per_block": 4096,
        "audit_seed": 1907,
        "lineages_per_stratum_cap": 128,
        "minimum_base_lineages": 128,
        "shard_count": 64,
        "bootstrap_replicates": 20_000,
        "bootstrap_seed": 91307,
        "two_sided_alpha": 0.05,
        "sampler_id": "isingfold-sa-sample-program-v1",
        "monolithic_production_execution": False,
        "chronology": {
            "plan_sealed_before_test_outcomes": True,
            "six_source_runs_authenticated_before_execution_manifest": True,
            "execution_manifest_sealed_before_audit_reads": True,
            "audit_feedback_forbidden": True,
        },
    }
    return {**payload, "record_digest": content_digest(payload)}


def test_registered_config_and_sampler_are_typed_and_whole_file_pinned(tmp_path: Path) -> None:
    payload = _production_config_payload()
    raw = final_audit_module.canonical_json_bytes(payload) + b"\n"
    path = tmp_path / "audit-config.json"
    path.write_bytes(raw)
    pin = hashlib.sha256(raw).hexdigest()

    authenticated = load_final_strength_audit_config(path, expected_sha256=pin)
    sampler = registered_final_strength_sampler_identity()

    assert authenticated.config.as_dict() == payload
    assert authenticated.file_sha256 == pin
    assert sampler.sampler_id == payload["sampler_id"]
    assert sampler.callable_path == "isingfold.rl.evaluator.sample_program"
    assert len(sampler.module_source_sha256) == 64
    with pytest.raises(FinalStrengthAuditError, match="SHA-256 mismatch"):
        load_final_strength_audit_config(path, expected_sha256="0" * 64)

    weak = dict(payload)
    weak["minimum_base_lineages"] = 127
    weak = _rehash(weak)
    weak_raw = final_audit_module.canonical_json_bytes(weak) + b"\n"
    weak_path = tmp_path / "weak.json"
    weak_path.write_bytes(weak_raw)
    with pytest.raises(FinalStrengthAuditError, match="invalid"):
        load_final_strength_audit_config(
            weak_path, expected_sha256=hashlib.sha256(weak_raw).hexdigest()
        )


def test_production_plan_binds_quality_ground_learned_selection_and_capped_strata(
    monkeypatch, tmp_path: Path
) -> None:
    learned = fixtures._runs(monkeypatch, tmp_path / "learned")
    identities = tuple((f"lineage-{index:03d}", f"instance-{index:03d}") for index in range(128))
    strata, design = evaluation_contract(identities)
    population = CompletePopulationIdentity(
        population_id="sealed-production-test",
        source_manifest_sha256="9" * 64,
        task_payload_sha256=stable_digest({"tasks": identities}),
        expected_instances=identities,
        expected_repetitions=4,
        evaluation_seed=55079,
        evaluation_strata=strata,
        confirmatory_design=design,
    )
    payload = _production_config_payload()
    config_path = tmp_path / "production-config.json"
    raw = final_audit_module.canonical_json_bytes(payload) + b"\n"
    config_path.write_bytes(raw)
    authenticated_config = load_final_strength_audit_config(
        config_path, expected_sha256=hashlib.sha256(raw).hexdigest()
    )
    plan = build_final_strength_audit_plan(
        population,
        Context(qubit_cap=4),
        learned[0].receipts[0].selector,
        quality_authority=_quality_authority(),
        learned_selection=_learned_selection(),
        stock_tuning_execution=_stock_tuning_binding(),
        audit_config=authenticated_config,
    )

    assert plan.publication_eligible is True
    assert plan.quality_authority["global"]["ground_root"]["receipt_sha256"] == "e" * 64
    assert plan.quality_authority["evaluation_partition"]["name"] == "test"
    assert plan.learned_selection["selection_receipt_sha256"] == "1" * 64
    assert len({row.lineage for row in plan.selected_instances}) == 128
    assert dict(plan.selected_stratum_census) == {
        name: min(plan.lineages_per_stratum, count)
        for name, count in plan.source_stratum_census
    }

    wrong_ground_census = _quality_authority()
    partition = wrong_ground_census["evaluation_partition"]
    partition["ground_partition"]["instance_set_digest"] = "5" * 64
    partition["record_digest"] = content_digest(
        {name: value for name, value in partition.items() if name != "record_digest"}
    )
    wrong_ground_census["record_digest"] = content_digest(
        {
            name: value
            for name, value in wrong_ground_census.items()
            if name != "record_digest"
        }
    )
    with pytest.raises(ValueError, match="ground-instance census differs"):
        build_final_strength_audit_plan(
            population,
            Context(qubit_cap=4),
            learned[0].receipts[0].selector,
            quality_authority=wrong_ground_census,
            learned_selection=_learned_selection(),
            stock_tuning_execution=_stock_tuning_binding(),
            audit_config=authenticated_config,
        )


def _stock_clone(
    learned: AuthenticatedCompleteSystemSeedRun,
    binding: dict[str, object],
) -> AuthenticatedExternalCompleteRun:
    report = dict(learned.report)
    report.update(
        {
            "external_tuning_execution": binding,
            "external_tuning_execution_digest": content_digest(binding),
        }
    )
    clone = object.__new__(AuthenticatedExternalCompleteRun)
    object.__setattr__(clone, "report", _rehash(report))
    execution = ExternalTuningExecutionBinding.from_mapping(binding)
    runtime_identity = {
        name: learned.report[name]
        for name in (
            "runtime_platform",
            "inference_device_type",
            "inference_device_name",
            "inference_threads",
            "deterministic",
        )
    }
    receipts = tuple(
        SimpleNamespace(
            pair_key=receipt.pair_key,
            population=receipt.population,
            selector=receipt.selector,
            context_digest=receipt.context_digest,
            outcome=receipt.outcome,
            training_seed=learned.report["training_seed"],
            quality_authority_digest=content_digest(learned.report["quality_authority"]),
            runtime_identity_digest=content_digest(runtime_identity),
            tuning_execution=execution,
        )
        for receipt in learned.receipts
    )
    object.__setattr__(clone, "receipts", receipts)
    object.__setattr__(clone, "evidence", learned.evidence)
    object.__setattr__(clone, "receipt_file_sha256", "5" * 64)
    object.__setattr__(clone, "evidence_file_sha256", "6" * 64)
    object.__setattr__(clone, "outcome_file_sha256", "7" * 64)
    return clone


def _sealed_plan(monkeypatch, tmp_path: Path):
    learned = fixtures._runs(monkeypatch, tmp_path / "learned")
    binding = _stock_tuning_binding()
    stock = tuple(_stock_clone(run, binding) for run in learned)
    population = learned[0].receipts[0].population
    context = Context(qubit_cap=4)
    selector = learned[0].receipts[0].selector
    plan = build_final_strength_audit_plan(
        population,
        context,
        selector,
        quality_authority_digest=content_digest(learned[0].report["quality_authority"]),
        stock_tuning_execution=binding,
        reads_per_block=16,
        audit_seed=1907,
        lineages_per_stratum_cap=1,
        minimum_base_lineages=1,
    )
    path = tmp_path / "final-strength-plan.json"
    sha256 = write_final_strength_audit_plan(path, plan)
    sealed = load_final_strength_audit_plan(path, expected_sha256=sha256)
    return learned, stock, context, plan, sealed, path


def _source_report_pins(learned, stock) -> dict[str, str]:
    return {
        f"{arm}:{int(run.report['training_seed'])}": run.report_file_sha256
        for arm, runs in (("learned", learned), ("tuned-stock", stock))
        for run in runs
    }


def _target_access(population: CompletePopulationIdentity) -> dict[str, object]:
    payload: dict[str, object] = {
        "evidence_manifest_record_digest": "b" * 64,
        "evidence_manifest_sha256": "c" * 64,
        "opened_files": [
            {
                "authority_root": "prepared",
                "role": "evaluator-targets",
                "relative_path": "targets/test.jsonl",
                "sha256": "1" * 64,
            },
            {
                "authority_root": "publisher",
                "role": "quality-evidence-manifest",
                "relative_path": "evidence/manifest.json",
                "sha256": "c" * 64,
            },
        ],
        "partition": "test",
        "prepared_manifest_record_digest": "2" * 64,
        "prepared_manifest_sha256": population.source_manifest_sha256,
        "publisher_attestation_digest": "a" * 64,
        "publisher_id": "test-publisher",
        "target_authority_record_digest": "3" * 64,
        "target_count": len(population.expected_instances),
        "target_path": "targets/test.jsonl",
        "target_set_digest": "d" * 64,
        "target_sha256": "1" * 64,
    }
    return {**payload, "record_digest": content_digest(payload)}


def _resign_authenticated_shard(shard, *, mutate_receipt=None, mutate_rows=None):
    receipt = final_audit_module.json.loads(final_audit_module.canonical_json_bytes(shard.receipt))
    rows = final_audit_module.json.loads(
        b"[" + b",".join(final_audit_module.canonical_json_bytes(row) for row in shard.rows) + b"]"
    )
    if mutate_receipt is not None:
        mutate_receipt(receipt)
    if mutate_rows is not None:
        mutate_rows(rows)
    raw = b"".join(final_audit_module.canonical_json_bytes(row) + b"\n" for row in rows)
    receipt["raw_rows"]["sha256"] = hashlib.sha256(raw).hexdigest()
    receipt["raw_rows"]["count"] = len(rows)
    receipt = _rehash(receipt)
    receipt_raw = final_audit_module.canonical_json_bytes(receipt) + b"\n"
    return type(shard)(
        receipt=receipt,
        rows=tuple(rows),
        receipt_file_sha256=hashlib.sha256(receipt_raw).hexdigest(),
        rows_file_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _quality_authority() -> dict[str, object]:
    global_payload = {
        "ground_root": {
            "receipt_sha256": "e" * 64,
            "record_digest": "f" * 64,
            "verifier_identity_digest": "9" * 64,
        },
        "publication_id": "test-publication",
        "publisher_attestation_record_digest": "a" * 64,
        "publisher_id": "test-publisher",
        "schema": "isingfold.global-quality-authority",
        "schema_version": 1,
        "target_authority_record_digest": "3" * 64,
    }
    partition_payload = {
        "evidence_manifest_record_digest": "b" * 64,
        "evidence_manifest_sha256": "c" * 64,
        "ground_partition": {
            "accepted_count": 128,
            "instance_set_digest": content_digest(
                [f"instance-{index:03d}" for index in range(128)]
            ),
            "receipt_record_digest": "7" * 64,
            "receipt_sha256": "6" * 64,
        },
        "name": "test",
        "schema": "isingfold.partition-quality-authority",
        "schema_version": 1,
        "target_access_record_digest": "8" * 64,
        "target_count": 128,
        "target_set_digest": "d" * 64,
    }
    payload = {
        "schema": "isingfold.quality-authority-binding",
        "schema_version": 2,
        "global": {**global_payload, "record_digest": content_digest(global_payload)},
        "evaluation_partition": {
            **partition_payload,
            "record_digest": content_digest(partition_payload),
        },
    }
    return {**payload, "record_digest": content_digest(payload)}


def _learned_selection() -> dict[str, object]:
    registry = runtime_implementation_registry()
    return {
        "schema": "isingfold.final-strength-frozen-learned-selection",
        "schema_version": 1,
        "selection_receipt_sha256": "1" * 64,
        "selection_record_digest": "2" * 64,
        "grid_manifest_sha256": "3" * 64,
        "selected_model_family": "if-core",
        "selected_grid_model_family": "if-core",
        "selected_method": "ppo-warm-start",
        "training_seeds": [1103, 2207, 3301],
        "source_cell_ids": ["selected-cell-0", "selected-cell-1", "selected-cell-2"],
        "source_checkpoint_payload_digests": [
            stable_digest({"checkpoint": seed}) for seed in (1103, 2207, 3301)
        ],
        "representation_selection_sha256": "4" * 64,
        "representation_selection_record_digest": "5" * 64,
        "runtime_implementation_registry": registry,
        "runtime_implementation_digest": content_digest(registry),
        "quality_preflight_receipt_sha256": "7" * 64,
        "quality_preflight_record_digest": "8" * 64,
        "seed_selection_forbidden": True,
    }


def test_plan_is_canonical_balanced_and_requires_out_of_band_pin(
    monkeypatch, tmp_path: Path
) -> None:
    _, _, _, plan, sealed, path = _sealed_plan(monkeypatch, tmp_path)

    assert plan.lineages_per_stratum == 1
    assert {count for _, count in plan.source_stratum_census} >= {1}
    assert {key.arm for key in plan.opportunity_keys} == set(ARMS)
    assert {key.training_seed for key in plan.opportunity_keys} == {1103, 2207, 3301}
    assert len(plan.opportunity_keys) == (
        len(plan.selected_instances) * plan.population_repetitions * 3 * 2
    )
    assert sealed.plan.as_dict() == plan.as_dict()
    with pytest.raises(FinalStrengthAuditError, match="SHA-256 mismatch"):
        load_final_strength_audit_plan(path, expected_sha256="f" * 64)
    learned, _, context, _, _, _ = _sealed_plan(monkeypatch, tmp_path / "second")
    with pytest.raises(ValueError, match="too few independent base lineages"):
        build_final_strength_audit_plan(
            learned[0].receipts[0].population,
            context,
            learned[0].receipts[0].selector,
            quality_authority_digest=content_digest(
                learned[0].report["quality_authority"]
            ),
            stock_tuning_execution=_stock_tuning_binding(),
            reads_per_block=16,
            audit_seed=1907,
            lineages_per_stratum_cap=1,
            minimum_base_lineages=3,
        )


def test_audit_uses_fresh_a_b_blocks_and_keeps_invalid_opportunities(
    monkeypatch, tmp_path: Path
) -> None:
    learned, stock, context, plan, sealed, _ = _sealed_plan(monkeypatch, tmp_path)
    tasks = (
        fixtures._task("quality-one", "lineage-a", positive=True),
        fixtures._task("quality-zero-1", "lineage-b", positive=False),
        fixtures._task("quality-zero-2", "lineage-b", positive=False),
    )
    calls: list[tuple[int, int]] = []
    verified = 0
    verifier = final_audit_module.verify_terminal_evidence

    def ordered_verifier(*args, **kwargs):
        nonlocal verified
        verifier(*args, **kwargs)
        verified += 1

    monkeypatch.setattr(final_audit_module, "verify_terminal_evidence", ordered_verifier)

    # A ties indices 2 and 3 and must pick the lower index 2.  Independent B gives
    # index 2 rate 3/4 and deployed index 1 rate 1/4, so the estimate is 1/2.
    a_hits = (1, 4, 12, 12)
    b_hits = (2, 4, 12, 1)

    def sampler(program, chains, problem, ground, *, num_reads, seed, num_sweeps):
        del chains, problem, ground, num_sweeps
        call_index = len(calls)
        assert verified > call_index // 8
        calls.append((program.strength_index, seed))
        hits = (a_hits if call_index % 2 == 0 else b_hits)[program.strength_index]
        return ReadBlock(hits, num_reads, 0.0, 0.0, program.strength_index)

    result = run_final_strength_audit(
        sealed, learned, stock, tasks, context, sampler=sampler
    )

    valid = [row for row in result.rows if row["valid_return"] is True]
    invalid = [row for row in result.rows if row["valid_return"] is False]
    assert valid and invalid
    assert verified == len(valid)
    assert len(calls) == 8 * len(valid)
    assert len({seed for _, seed in calls}) == len(calls)
    primary_seeds = {
        record.terminal_evidence.evaluator_seed
        for run in (*learned, *stock)
        for record in run.evidence
        if record.terminal_evidence is not None
    }
    assert not ({seed for _, seed in calls} & primary_seeds)
    first = valid[0]
    assert first["oracle_a_strength_index"] == 2
    assert first["oracle_a_tie_indices"] == (2, 3)
    assert first["conditional_regret"] == pytest.approx(0.5)
    assert first["selected_oracle_agreement"] is False
    assert {block["domain"] for block in first["block_a"]} == {BLOCK_DOMAINS["A"]}
    assert {block["domain"] for block in first["block_b"]} == {BLOCK_DOMAINS["B"]}
    assert all(row["conditional_regret"] is None for row in invalid)
    assert all(row["reads_consumed"] == 0 for row in invalid)
    for arm in ARMS:
        arm_summary = result.aggregate["arms"][arm]
        assert arm_summary["opportunities"] == len(plan.opportunity_keys) // 2
        assert arm_summary["invalid_returns"] > 0
        assert arm_summary["conditional_regret"]["mean"] == pytest.approx(0.5)
        assert arm_summary["conditional_regret"]["median"] == pytest.approx(0.5)
        assert arm_summary["conditional_regret"]["p95"] == pytest.approx(0.5)
        assert arm_summary["conditional_regret"]["descriptive_crossed_bootstrap_95_ci"]
        assert "block_repeatability" in arm_summary
        assert "block_calibration" not in arm_summary
        sensitivity = arm_summary["invalid_worst_case_sensitivity"]
        assert sensitivity["lower"] < sensitivity["upper"]
    paired = result.aggregate["paired_learned_vs_tuned_stock"]
    assert paired["learned_minus_tuned_stock_conditional_regret"]["mean"] == pytest.approx(0.0)
    assert paired["joint_valid_coverage"] < 1.0
    assert paired["invalid_worst_case_sensitivity"]["definition"].endswith("[-2,2]")


def test_split_sample_regret_retains_negative_values_without_clipping(
    monkeypatch, tmp_path: Path
) -> None:
    learned, stock, context, _, sealed, _ = _sealed_plan(monkeypatch, tmp_path)
    tasks = (
        fixtures._task("quality-one", "lineage-a", positive=True),
        fixtures._task("quality-zero-1", "lineage-b", positive=False),
        fixtures._task("quality-zero-2", "lineage-b", positive=False),
    )
    call = 0

    def sampler(program, chains, problem, ground, *, num_reads, seed, num_sweeps):
        nonlocal call
        del chains, problem, ground, seed, num_sweeps
        # The fixture deploys strength 1. A selects strength 0 while independent B
        # favors deployed strength 1, so the unbiased split-sample estimate is -1.
        hits = (num_reads if program.strength_index == (0 if call % 2 == 0 else 1) else 0)
        call += 1
        return ReadBlock(hits, num_reads, 0.0, 0.0, program.strength_index)

    result = run_final_strength_audit(
        sealed, learned, stock, tasks, context, sampler=sampler
    )
    valid = [row for row in result.rows if row["valid_return"] is True]
    assert valid
    assert all(row["conditional_regret"] == -1.0 for row in valid)
    assert all(row["opportunity_sensitivity_range"] == (-1.0, -1.0) for row in valid)
    assert result.aggregate["arms"]["learned"]["conditional_regret"]["mean"] == -1.0


def test_audit_fails_before_sampling_on_tuning_population_or_runtime_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    learned, stock, context, _, sealed, _ = _sealed_plan(monkeypatch, tmp_path)
    tasks = (
        fixtures._task("quality-one", "lineage-a", positive=True),
        fixtures._task("quality-zero-1", "lineage-b", positive=False),
        fixtures._task("quality-zero-2", "lineage-b", positive=False),
    )
    called = False

    def sampler(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("contract mismatch must fail before opening audit reads")

    changed_report = dict(stock[0].report)
    changed_report["external_tuning_execution_digest"] = stable_digest({"other": "tuning"})
    object.__setattr__(stock[0], "report", _rehash(changed_report))
    with pytest.raises(FinalStrengthAuditError, match="tuned deployment"):
        run_final_strength_audit(
            sealed, learned, stock, tasks, context, sampler=sampler
        )
    assert called is False


def test_atomic_publication_binds_every_raw_row(monkeypatch, tmp_path: Path) -> None:
    learned, stock, context, plan, sealed, plan_path = _sealed_plan(monkeypatch, tmp_path)
    tasks = (
        fixtures._task("quality-one", "lineage-a", positive=True),
        fixtures._task("quality-zero-1", "lineage-b", positive=False),
        fixtures._task("quality-zero-2", "lineage-b", positive=False),
    )

    def sampler(program, chains, problem, ground, *, num_reads, seed, num_sweeps):
        del chains, problem, ground, seed, num_sweeps
        return ReadBlock(8, num_reads, 0.0, 0.0, program.strength_index)

    result = run_final_strength_audit(
        sealed, learned, stock, tasks, context, sampler=sampler
    )
    destination = tmp_path / "final-strength-audit"
    receipt = publish_final_strength_audit(
        destination, sealed_plan=sealed, result=result
    )

    raw = (destination / "rows.jsonl").read_bytes()
    assert receipt["plan"]["record_digest"] == plan.record_digest
    assert receipt["plan"]["file_sha256"] == hashlib.sha256(plan_path.read_bytes()).hexdigest()
    assert receipt["raw_rows"]["count"] == len(plan.opportunity_keys)
    assert receipt["raw_rows"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["record_digest"] == content_digest(
        {key: value for key, value in receipt.items() if key != "record_digest"}
    )
    with pytest.raises(FileExistsError):
        publish_final_strength_audit(destination, sealed_plan=sealed, result=result)


def test_execution_manifest_preallocates_unique_fresh_counterbalanced_schedule(
    monkeypatch, tmp_path: Path
) -> None:
    learned, stock, context, plan, sealed, _ = _sealed_plan(monkeypatch, tmp_path)
    tasks = (
        fixtures._task("quality-one", "lineage-a", positive=True),
        fixtures._task("quality-zero-1", "lineage-b", positive=False),
        fixtures._task("quality-zero-2", "lineage-b", positive=False),
    )
    manifest = build_final_strength_audit_execution_manifest(
        sealed,
        learned,
        stock,
        tasks,
        context,
        target_access=_target_access(learned[0].receipts[0].population),
        source_report_sha256_pins=_source_report_pins(learned, stock),
    )
    path = tmp_path / "execution-manifest.json"
    sha256 = write_final_strength_audit_execution_manifest(path, manifest)
    sealed_execution = load_final_strength_audit_execution_manifest(
        path, expected_sha256=sha256, sealed_plan=sealed
    )

    schedules = [
        item
        for opportunity in manifest.opportunity_execution
        for item in opportunity["seed_schedule"]
    ]
    seeds = [item["seed"] for item in schedules]
    primary = {
        evidence.terminal_evidence.evaluator_seed
        for run in (*learned, *stock)
        for evidence in run.evidence
        if evidence.terminal_evidence is not None
    }
    assert len(seeds) == len(set(seeds))
    assert not (set(seeds) & primary)
    assert all(0 <= seed < 2**31 for seed in seeds)
    assert all(
        sorted((item["execution_rank"] for item in opportunity["seed_schedule"]))
        == list(range(8))
        for opportunity in manifest.opportunity_execution
        if opportunity["valid_return"] is True
    )
    assert sealed_execution.manifest.record_digest == manifest.record_digest
    assert manifest.publication_eligible is False
    assert manifest.sampler_identity["sampler_id"] == plan.sampler_identity["sampler_id"]
    assert manifest.target_access["partition"] == "test"
    assert manifest.target_access_digest == manifest.target_access["record_digest"]

    tampered = manifest.as_dict()
    first = tampered["opportunity_execution"][0]
    first["seed_schedule"][1]["seed"] = first["seed_schedule"][0]["seed"]
    first["seed_schedule_digest"] = content_digest(first["seed_schedule"])
    first["record_digest"] = content_digest(
        {name: value for name, value in first.items() if name != "record_digest"}
    )
    tampered["opportunity_execution_digest"] = content_digest(
        tampered["opportunity_execution"]
    )
    tampered["record_digest"] = content_digest(
        {name: value for name, value in tampered.items() if name != "record_digest"}
    )
    with pytest.raises(ValueError, match="reuses"):
        FinalStrengthAuditExecutionManifest.from_mapping(tampered)

    ground_drift = manifest.as_dict()
    ground_drift["quality_authority"]["opaque_quality_authority_digest"] = "1" * 64
    ground_drift["quality_authority_digest"] = "1" * 64
    ground_drift["record_digest"] = content_digest(
        {name: value for name, value in ground_drift.items() if name != "record_digest"}
    )
    drifted = FinalStrengthAuditExecutionManifest.from_mapping(ground_drift)
    with pytest.raises(FinalStrengthAuditError, match="execution manifest differs"):
        drifted.validate_plan(sealed)

    target_drift = manifest.as_dict()
    target_drift["target_access"]["partition"] = "val"
    target_drift["target_access"]["record_digest"] = content_digest(
        {
            name: value
            for name, value in target_drift["target_access"].items()
            if name != "record_digest"
        }
    )
    target_drift["target_access_digest"] = target_drift["target_access"][
        "record_digest"
    ]
    target_drift["record_digest"] = content_digest(
        {name: value for name, value in target_drift.items() if name != "record_digest"}
    )
    with pytest.raises(ValueError, match="test partition"):
        FinalStrengthAuditExecutionManifest.from_mapping(target_drift)

    wrong_count = _target_access(learned[0].receipts[0].population)
    wrong_count["target_count"] += 1
    wrong_count["record_digest"] = content_digest(
        {name: value for name, value in wrong_count.items() if name != "record_digest"}
    )
    with pytest.raises(ValueError, match="target count differs"):
        build_final_strength_audit_execution_manifest(
            sealed,
            learned,
            stock,
            tasks,
            context,
            target_access=wrong_count,
            source_report_sha256_pins=_source_report_pins(learned, stock),
        )


def test_deterministic_shards_keep_pairs_complete_and_merge_recomputes(
    monkeypatch, tmp_path: Path
) -> None:
    learned, stock, context, _, _, _ = _sealed_plan(monkeypatch, tmp_path / "base")
    binding = _stock_tuning_binding()
    population = learned[0].receipts[0].population
    plan = build_final_strength_audit_plan(
        population,
        context,
        learned[0].receipts[0].selector,
        quality_authority_digest=content_digest(learned[0].report["quality_authority"]),
        stock_tuning_execution=binding,
        reads_per_block=16,
        audit_seed=1907,
        lineages_per_stratum_cap=1,
        minimum_base_lineages=1,
        shard_count=2,
    )
    plan_path = tmp_path / "plan.json"
    plan_sha = write_final_strength_audit_plan(plan_path, plan)
    sealed = load_final_strength_audit_plan(plan_path, expected_sha256=plan_sha)
    tasks = (
        fixtures._task("quality-one", "lineage-a", positive=True),
        fixtures._task("quality-zero-1", "lineage-b", positive=False),
        fixtures._task("quality-zero-2", "lineage-b", positive=False),
    )
    execution = build_final_strength_audit_execution_manifest(
        sealed,
        learned,
        stock,
        tasks,
        context,
        target_access=_target_access(population),
        source_report_sha256_pins=_source_report_pins(learned, stock),
    )
    execution_path = tmp_path / "execution.json"
    execution_sha = write_final_strength_audit_execution_manifest(
        execution_path, execution
    )
    sealed_execution = load_final_strength_audit_execution_manifest(
        execution_path, expected_sha256=execution_sha, sealed_plan=sealed
    )

    def sampler(program, chains, problem, ground, *, num_reads, seed, num_sweeps):
        del chains, problem, ground, seed, num_sweeps
        return ReadBlock(2 * program.strength_index, num_reads, 0.0, 0.0, program.strength_index)

    authenticated = []
    for index in range(2):
        shard = run_final_strength_audit_shard(
            sealed,
            sealed_execution,
            learned,
            stock,
            tasks,
            context,
            shard_index=index,
            sampler=sampler,
        )
        arms_by_pair: dict[tuple[object, ...], set[str]] = {}
        for row in shard.rows:
            key = row["audit_key"]
            pair = (key["lineage"], key["instance"], key["repetition"], key["training_seed"])
            arms_by_pair.setdefault(pair, set()).add(key["arm"])
        assert all(arms == set(ARMS) for arms in arms_by_pair.values())
        destination = tmp_path / f"shard-{index}"
        receipt = publish_final_strength_audit_shard(
            destination,
            sealed_plan=sealed,
            sealed_execution_manifest=sealed_execution,
            result=shard,
        )
        authenticated.append(
            load_final_strength_audit_shard(
                destination,
                expected_receipt_sha256=hashlib.sha256(
                    (destination / "receipt.json").read_bytes()
                ).hexdigest(),
            )
        )
        assert receipt["publication_eligible"] is False

    merged = merge_final_strength_audit_shards(
        sealed, sealed_execution, authenticated
    )
    assert len(merged.rows) == len(plan.opportunity_keys)
    assert merged.aggregate["opportunities"] == len(plan.opportunity_keys)
    assert merged.publication_eligible is False
    with pytest.raises(FinalStrengthAuditError, match="complete authenticated shard census"):
        merge_final_strength_audit_shards(sealed, sealed_execution, authenticated[:1])
    with pytest.raises(FinalStrengthAuditError, match="missing, duplicated"):
        merge_final_strength_audit_shards(
            sealed, sealed_execution, [authenticated[0], authenticated[0]]
        )

    def drift_ground(receipt):
        receipt["plan"]["quality_authority_digest"] = "0" * 64

    ground_drift = _resign_authenticated_shard(
        authenticated[1], mutate_receipt=drift_ground
    )
    with pytest.raises(FinalStrengthAuditError, match="ground"):
        merge_final_strength_audit_shards(
            sealed, sealed_execution, [authenticated[0], ground_drift]
        )

    def drift_source(receipt):
        receipt["source_runs"][0]["report_file_sha256"] = "0" * 64
        receipt["source_runs_digest"] = content_digest(receipt["source_runs"])

    source_drift = _resign_authenticated_shard(
        authenticated[1], mutate_receipt=drift_source
    )
    with pytest.raises(FinalStrengthAuditError, match="source"):
        merge_final_strength_audit_shards(
            sealed, sealed_execution, [authenticated[0], source_drift]
        )

    def drift_sampler(receipt):
        receipt["sampler_identity"]["callable_qualname"] += ".different"
        receipt["sampler_identity_digest"] = content_digest(receipt["sampler_identity"])

    sampler_digest = None

    def drift_sampler_rows(rows):
        nonlocal sampler_digest
        sampler_identity = final_audit_module.json.loads(
            final_audit_module.canonical_json_bytes(authenticated[1].receipt["sampler_identity"])
        )
        sampler_identity["callable_qualname"] += ".different"
        sampler_digest = content_digest(sampler_identity)
        for row in rows:
            row["sampler_identity_digest"] = sampler_digest
            row.update(_rehash(row))

    sampler_drift = _resign_authenticated_shard(
        authenticated[1],
        mutate_receipt=drift_sampler,
        mutate_rows=drift_sampler_rows,
    )
    assert sampler_drift.receipt["sampler_identity_digest"] == sampler_digest
    with pytest.raises(FinalStrengthAuditError, match="mix sampler"):
        merge_final_strength_audit_shards(
            sealed, sealed_execution, [authenticated[0], sampler_drift]
        )

    def reuse_seed(rows):
        valid = next(row for row in rows if row["valid_return"] is True)
        valid["block_b"][0]["seed"] = valid["block_a"][0]["seed"]
        valid.update(_rehash(valid))

    schedule_drift = _resign_authenticated_shard(
        authenticated[1], mutate_rows=reuse_seed
    )
    with pytest.raises(ValueError, match="seed/count contract"):
        merge_final_strength_audit_shards(
            sealed, sealed_execution, [authenticated[0], schedule_drift]
        )

    fake = object.__new__(type(merged))
    object.__setattr__(fake, "rows", merged.rows)
    object.__setattr__(fake, "aggregate", {"opportunities": 0})
    object.__setattr__(fake, "source_runs", merged.source_runs)
    object.__setattr__(fake, "publication_eligible", merged.publication_eligible)
    object.__setattr__(fake, "execution_manifest_digest", merged.execution_manifest_digest)
    object.__setattr__(fake, "sampler_identity_digest", merged.sampler_identity_digest)
    with pytest.raises(ValueError, match="aggregate"):
        publish_final_strength_audit(
            tmp_path / "fake-publish",
            sealed_plan=sealed,
            sealed_execution_manifest=sealed_execution,
            result=fake,
        )
