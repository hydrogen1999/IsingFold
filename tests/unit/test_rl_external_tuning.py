"""Validation-only stock-minorminer tuning and freeze contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from isingfold.rl.checkpoint import runtime_implementation_registry
from isingfold.rl.complete_system import CompletePopulationIdentity
from isingfold.rl.contracts import Context, stable_digest
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.evaluation_strata import (
    ConfirmatoryEvaluationDesign,
    EvaluationStratum,
)
from isingfold.rl.external_tuning import (
    build_external_tuning_report,
    EXTERNAL_TUNING_REGISTRY_FILE_SHA256,
    EXTERNAL_TUNING_REGISTRY_RECORD_DIGEST,
    ExternalTuningExecutionBinding,
    ExternalTuningRun,
    external_tuning_lineage_metrics,
    load_external_tuning_registry,
    load_external_tuning_selection,
    select_external_tuning,
    write_external_tuning_selection,
)


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "configs" / "external_minorminer_tuning_hybrid_v1.json"
GRID = ROOT / "configs" / "rl_grid_hybrid_v1.json"
EXTERNAL_CONFIG = ROOT / "configs" / "external_minorminer_complete_v1.json"
FAILURE_REASONS = (
    "EXTERNAL_VALID_RETURN",
    "EXTERNAL_NO_VALID_EMBEDDING",
    "EXTERNAL_COMPLETE_SYSTEM_WALLCLOCK_EXHAUSTED",
    "EXTERNAL_KNOWN_WORK_CAP_EXHAUSTED",
)


def _registry():
    return load_external_tuning_registry(
        REGISTRY,
        expected_file_sha256=EXTERNAL_TUNING_REGISTRY_FILE_SHA256,
        grid_path=GRID,
        external_config_path=EXTERNAL_CONFIG,
    )


def test_validation_runner_builds_a_canonical_tuning_report(monkeypatch) -> None:
    from tests.unit import test_rl_external_pairing as pairing_fixtures
    from tests.unit.evaluation_strata_support import evaluation_contract
    from isingfold.rl.complete_system import task_population_digest
    from isingfold.rl.external import BackendSearchResult, SearchStatus
    from isingfold.rl.external_pairing import (
        ExternalCompleteSystemConfig,
        context_snapshot,
        run_external_complete_system,
        runtime_identity,
    )

    registry = _registry()
    task = pairing_fixtures._task()
    identities = ((task.lineage or task.name, task.name),)
    strata, design = evaluation_contract(identities, partition="val")
    population = CompletePopulationIdentity(
        population_id="validation-tuning-population",
        source_manifest_sha256="b" * 64,
        task_payload_sha256=task_population_digest([task]),
        expected_instances=identities,
        expected_repetitions=registry.repetitions,
        evaluation_seed=55079,
        evaluation_strata=strata,
        confirmatory_design=design,
    )
    context = Context(qubit_cap=4)
    external_config = ExternalCompleteSystemConfig.from_mapping(
        json.loads(EXTERNAL_CONFIG.read_text())
    )
    target_access = _target_access("val", 1)
    ground_partition_receipt = _ground_partition_receipt("val", 1)
    quality_authority = _quality_authority(
        count=1,
        target_access=target_access,
        ground_partition_receipt=ground_partition_receipt,
    )
    compute = runtime_identity(
        runtime_platform={"system": "fixture", "hostname": "fixture-host"},
        inference_device_type="cpu",
        inference_device_name="fixture-cpu",
        inference_threads=1,
        deterministic=True,
    )
    binding = ExternalTuningExecutionBinding.for_validation(registry, 0)
    monkeypatch.setattr(
        "isingfold.rl.external_pairing.sample_program",
        lambda *args, **kwargs: pytest.fail("no-embedding tuning cannot sample"),
    )
    _, receipts = run_external_complete_system(
        [task],
        context,
        pairing_fixtures._Backend(
            BackendSearchResult(SearchStatus.NO_EMBEDDING, None, 0.0)
        ),
        external_config,
        learned_config=pairing_fixtures._learned_config(seconds=60.0),
        selector=pairing_fixtures.fixed_strength_selector(),
        selector_identity=pairing_fixtures._selector(),
        population=population,
        quality_authority=quality_authority,
        seed=population.evaluation_seed,
        repetitions=registry.repetitions,
        training_seed_index=0,
        training_seed=registry.tuning_seeds[0],
        compute_identity=compute,
        tuning_execution=binding,
    )
    metrics = external_tuning_lineage_metrics(receipts)
    assert metrics[0]["attempts"] == registry.repetitions
    assert metrics[0]["failure_counts"]["EXTERNAL_NO_VALID_EMBEDDING"] == (
        registry.repetitions
    )
    runtime_registry = runtime_implementation_registry()
    artifacts = {
        name: {"path": path, "sha256": digest * 64, "count": len(receipts)}
        for name, path, digest in (
            ("receipts", "external_tuning_receipts.jsonl", "1"),
            ("terminal_evidence", "terminal_evidence.jsonl", "2"),
            ("outcomes", "outcomes.jsonl", "3"),
        )
    }
    report = build_external_tuning_report(
        registry=registry,
        candidate_index=0,
        tuning_seed_index=0,
        receipts=receipts,
        runtime_identity=compute,
        runtime_implementation_registry=runtime_registry,
        quality_authority=quality_authority,
        target_access=target_access,
        ground_partition_receipt=ground_partition_receipt,
        context=context_snapshot(context),
        artifacts=artifacts,
    )
    content = canonical_json_bytes(report) + b"\n"
    loaded = ExternalTuningRun.from_mapping(
        report,
        report_file_sha256=hashlib.sha256(content).hexdigest(),
        report_path="report.json",
    )
    assert loaded.report["candidate_id"] == "stock-default-v1"
    assert loaded.report["summary"]["attempts"] == registry.repetitions


def _selector() -> dict[str, str]:
    return {
        "component_id": "isingfold-if-q3-s0-strength-selector",
        "version": "if-q3-s0-graph-selector-2",
        "implementation": "tests.frozen-selector",
        "artifact_sha256": "a" * 64,
    }


def _record(payload: dict[str, object]) -> dict[str, object]:
    return {**payload, "record_digest": content_digest(payload)}


def _target_access(partition: str, count: int) -> dict[str, object]:
    return _record(
        {
            "partition": partition,
            "schema": "test.target-access",
            "schema_version": 1,
            "target_count": count,
        }
    )


def _ground_partition_receipt(partition: str, count: int) -> dict[str, object]:
    return _record(
        {
            "accepted_count": count,
            "partition": partition,
            "schema": "test.ground-partition",
            "schema_version": 1,
        }
    )


def _quality_authority(
    *,
    count: int = 128,
    target_access: dict[str, object] | None = None,
    ground_partition_receipt: dict[str, object] | None = None,
) -> dict[str, object]:
    target = target_access or _target_access("val", count)
    ground = ground_partition_receipt or _ground_partition_receipt("val", count)
    global_authority = _record(
        {
            "ground_root": {
                "receipt_sha256": "1" * 64,
                "record_digest": "2" * 64,
                "verifier_identity_digest": "3" * 64,
            },
            "publication_id": "fixture-publication-v2",
            "publisher_attestation_record_digest": "b" * 64,
            "publisher_id": "fixture-publisher",
            "schema": "isingfold.global-quality-authority",
            "schema_version": 1,
            "target_authority_record_digest": "4" * 64,
        }
    )
    partition_authority = _record(
        {
            "evidence_manifest_record_digest": "c" * 64,
            "evidence_manifest_sha256": "d" * 64,
            "ground_partition": {
                "accepted_count": count,
                "instance_set_digest": "5" * 64,
                "receipt_record_digest": ground["record_digest"],
                "receipt_sha256": "6" * 64,
            },
            "name": "val",
            "schema": "isingfold.partition-quality-authority",
            "schema_version": 1,
            "target_access_record_digest": target["record_digest"],
            "target_count": count,
            "target_set_digest": "e" * 64,
        }
    )
    return _record(
        {
            "schema": "isingfold.quality-authority-binding",
            "schema_version": 2,
            "global": global_authority,
            "evaluation_partition": partition_authority,
        }
    )


def _task_census() -> list[list[str]]:
    return [[f"lineage-{index:03d}", f"task-{index:03d}"] for index in range(128)]


def _population(census: list[list[str]], *, repetitions: int) -> dict[str, object]:
    identities = tuple((lineage, instance) for lineage, instance in census)
    strata = tuple(
        EvaluationStratum(
            lineage=lineage,
            instance=instance,
            learning_partition="val",
            application_family="frustrated-loop",
            problem_origin="synthetic",
            host_family="pegasus",
            fault_status="none",
            distribution_regime="iid",
            calibration_status="not_applicable",
            calibration_sha256=None,
            embedding_difficulty="mixed",
            sampling_difficulty="mixed",
            decision_difficulty="mixed",
            nominal_size=index + 2,
            source_registry_row_digest=stable_digest({"registry": [lineage, instance]}),
            source_provenance_record_digests=(
                stable_digest({"provenance": [lineage, instance]}),
            ),
        )
        for index, (lineage, instance) in enumerate(identities)
    )
    design = ConfirmatoryEvaluationDesign.from_prepared_receipt(
        {
            "manifest_sha256": "7" * 64,
            "manifest_record_digest": "6" * 64,
            "power_targets": [
                {
                    "target_id": "validation-valid-return-noninferiority",
                    "endpoint": "valid-return-noninferiority",
                    "alternative": "one-sided-noninferiority",
                    "alpha": 0.05,
                    "target_power": 0.51,
                    "assumed_discordance": 0.01,
                    "assumed_true_difference": 0.47,
                    "noninferiority_margin": 0.02,
                    "power_separation": 0.49,
                    "filter": {
                        "application_family": ["frustrated-loop"],
                        "problem_origin": ["synthetic"],
                        "host_family": ["pegasus"],
                        "fault_status": ["none"],
                        "distribution_regime": ["iid"],
                        "calibration_status": ["not_applicable"],
                        "embedding_difficulty": ["mixed"],
                        "sampling_difficulty": ["mixed"],
                        "decision_difficulty": ["mixed"],
                    },
                    "learning_partition": "val",
                    "method": "paired-binary-normal-approximation",
                    "minimum_base_lineages": 128,
                }
            ],
            "precision_targets": [
                {
                    "target_id": "validation-paired-utility-precision",
                    "endpoint": "learned-minus-stock-unconditional-if-q3-s0",
                    "learning_partition": "val",
                    "filter": {
                        "application_family": ["frustrated-loop"],
                        "problem_origin": ["synthetic"],
                        "host_family": ["pegasus"],
                        "fault_status": ["none"],
                        "distribution_regime": ["iid"],
                        "calibration_status": ["not_applicable"],
                        "embedding_difficulty": ["mixed"],
                        "sampling_difficulty": ["mixed"],
                        "decision_difficulty": ["mixed"],
                    },
                    "method": "bounded-paired-difference-worst-case-normal",
                    "confidence_level": 0.95,
                    "half_width": 0.2,
                    "minimum_base_lineages": 128,
                    "outcome_bounds": [-1.0, 1.0],
                    "variance_bound": 1.0,
                }
            ],
        },
        partition="val",
        noninferiority_margin=0.02,
    )
    return CompletePopulationIdentity(
        population_id="external-tuning-validation-population-v1",
        source_manifest_sha256="9" * 64,
        task_payload_sha256="8" * 64,
        expected_instances=identities,
        expected_repetitions=repetitions,
        evaluation_seed=941,
        evaluation_strata=strata,
        confirmatory_design=design,
    ).as_dict()


def _work(*, valid: int, restarts: int) -> dict[str, int | None]:
    return {
        "decisions": None,
        "route_expansions": None,
        "materializations": None,
        "compiler_calls": 4 * valid,
        "validator_calls": 2 * valid,
        "cut_edge_visits": None,
        "restart_work": restarts,
        "evaluator_reads": 4096 * valid,
        "feature_work": 4 * valid,
    }


def _report_payload(
    registry,
    *,
    candidate_index: int,
    seed_index: int,
    utility: float,
    valid: bool = True,
    online_seconds: float = 1.0,
    runtime_label: str | None = None,
) -> dict[str, object]:
    candidate = registry.candidates[candidate_index]
    tuning_seed = registry.tuning_seeds[seed_index]
    runtime_registry = runtime_implementation_registry()
    selector = _selector()
    census = _task_census()
    target_access = _target_access("val", len(census))
    ground_partition_receipt = _ground_partition_receipt("val", len(census))
    quality = _quality_authority(
        count=len(census),
        target_access=target_access,
        ground_partition_receipt=ground_partition_receipt,
    )
    population = _population(census, repetitions=registry.repetitions)
    context = Context(qubit_cap=160)
    lineages = []
    for lineage, _ in census:
        attempts = registry.repetitions
        valid_returns = attempts if valid else 0
        failures = {reason: 0 for reason in FAILURE_REASONS}
        failures[
            "EXTERNAL_VALID_RETURN" if valid else "EXTERNAL_NO_VALID_EMBEDDING"
        ] = attempts
        lineages.append(
            {
                "lineage": lineage,
                "attempts": attempts,
                "utility_sum": attempts * utility if valid else 0.0,
                "valid_returns": valid_returns,
                "online_seconds_total": attempts * online_seconds,
                "failure_counts": failures,
                "work_totals": _work(
                    valid=valid_returns,
                    restarts=attempts * candidate.outer_restart_cap,
                ),
                "wallclock_noncompliant_attempts": 0,
                "known_work_cap_violations": 0,
            }
        )
    payload: dict[str, object] = {
        "schema": "isingfold.external-tuning-validation-run",
        "schema_version": 2,
        "status": "complete",
        "partition": "validation",
        "test_data_opened": False,
        "registry_file_sha256": registry.file_sha256,
        "registry_record_digest": registry.record_digest,
        "grid_file_sha256": registry.grid_file_sha256,
        "external_config_digest": registry.external_config_digest,
        "external_config_file_sha256": registry.external_config_file_sha256,
        "candidate_index": candidate_index,
        "candidate_id": candidate.candidate_id,
        "candidate": candidate.as_dict(),
        "candidate_digest": candidate.digest,
        "tuning_seed_index": seed_index,
        "tuning_seed": tuning_seed,
        "repetitions": registry.repetitions,
        "audit_reads": registry.audit_reads,
        "wallclock_cap_seconds": registry.online_wallclock_seconds,
        "runtime_platform": {
            "system": "fixture",
            "hostname": runtime_label or f"seed-host-{seed_index}",
        },
        "inference_device_type": "cpu",
        "inference_device_name": "fixture-cpu",
        "inference_threads": 1,
        "deterministic": True,
        "runtime_implementation_registry": runtime_registry,
        "runtime_implementation_digest": content_digest(runtime_registry),
        "selector": selector,
        "selector_digest": content_digest(selector),
        "quality_authority": quality,
        "quality_authority_digest": content_digest(quality),
        "target_access": target_access,
        "ground_partition_receipt": ground_partition_receipt,
        "population": population,
        "population_digest": stable_digest(population),
        "context": {
            "qubit_cap": context.qubit_cap,
            "work_cap": context.caps.as_dict(),
        },
        "context_digest": stable_digest(
            {"qubit_cap": context.qubit_cap, "work_cap": context.caps.as_dict()}
        ),
        "task_census": census,
        "task_census_digest": content_digest(census),
        "lineage_census_digest": content_digest([row[0] for row in census]),
        "lineage_metrics": lineages,
        "artifacts": {
            "receipts": {
                "path": "external_tuning_receipts.jsonl",
                "sha256": "1" * 64,
                "count": len(census) * registry.repetitions,
            },
            "terminal_evidence": {
                "path": "terminal_evidence.jsonl",
                "sha256": "2" * 64,
                "count": len(census) * registry.repetitions,
            },
            "outcomes": {
                "path": "outcomes.jsonl",
                "sha256": "3" * 64,
                "count": len(census) * registry.repetitions,
            },
        },
    }
    payload["summary"] = ExternalTuningRun.summary_from_lineages(lineages)
    payload["record_digest"] = content_digest(payload)
    return payload


def _runs(registry, utilities: list[float]) -> tuple[ExternalTuningRun, ...]:
    runs = []
    for candidate_index, utility in enumerate(utilities):
        for seed_index in range(3):
            payload = _report_payload(
                registry,
                candidate_index=candidate_index,
                seed_index=seed_index,
                utility=utility,
                online_seconds=1.0 + candidate_index,
            )
            runs.append(
                ExternalTuningRun.from_mapping(
                    payload,
                    report_file_sha256=hashlib.sha256(
                        canonical_json_bytes(payload) + b"\n"
                    ).hexdigest(),
                    report_path=(
                        f"{payload['candidate_id']}/seed-{seed_index}/report.json"
                    ),
                )
            )
    return tuple(runs)


def test_checked_in_registry_is_finite_pinned_and_covers_all_baseline_families() -> None:
    registry = _registry()

    assert registry.record_digest == EXTERNAL_TUNING_REGISTRY_RECORD_DIGEST
    assert registry.file_sha256 == EXTERNAL_TUNING_REGISTRY_FILE_SHA256
    assert registry.tuning_seeds == (1103, 2207, 3301)
    assert len(registry.candidates) == 7
    assert {candidate.family for candidate in registry.candidates} == {
        "stock-default",
        "time-saturating-resource",
        "time-saturating-quality",
    }
    stock = registry.candidates[0]
    assert stock.native_parameters["tries"] == 10
    assert stock.outer_restart_cap == 1
    assert all(not candidate.online_evaluator_feedback for candidate in registry.candidates)


def test_registry_rejects_rehashed_candidate_or_grid_drift(tmp_path: Path) -> None:
    payload = json.loads(REGISTRY.read_text())
    payload["candidates"][0]["native_parameters"]["tries"] = 9
    payload["record_digest"] = content_digest(
        {key: value for key, value in payload.items() if key != "record_digest"}
    )
    changed = tmp_path / "registry.json"
    changed.write_bytes(canonical_json_bytes(payload) + b"\n")
    changed_sha = hashlib.sha256(changed.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="fixed v1 registry"):
        load_external_tuning_registry(
            changed,
            expected_file_sha256=changed_sha,
            grid_path=GRID,
            external_config_path=EXTERNAL_CONFIG,
        )

    changed_grid = tmp_path / "grid.json"
    grid = json.loads(GRID.read_text())
    grid["description"] += " drift"
    changed_grid.write_bytes(canonical_json_bytes(grid) + b"\n")
    with pytest.raises(ValueError, match="grid.*pin"):
        load_external_tuning_registry(
            REGISTRY,
            expected_file_sha256=EXTERNAL_TUNING_REGISTRY_FILE_SHA256,
            grid_path=changed_grid,
            external_config_path=EXTERNAL_CONFIG,
        )


def test_tuning_run_rejects_test_access_and_runtime_mismatch_within_seed() -> None:
    registry = _registry()
    payload = _report_payload(
        registry, candidate_index=0, seed_index=0, utility=0.5
    )
    payload["partition"] = "test"
    payload["test_data_opened"] = True
    payload["record_digest"] = content_digest(
        {key: value for key, value in payload.items() if key != "record_digest"}
    )
    with pytest.raises(ValueError, match="validation-only|test"):
        ExternalTuningRun.from_mapping(
            payload, report_file_sha256="4" * 64, report_path="bad/report.json"
        )

    runs = list(_runs(registry, [0.4] * len(registry.candidates)))
    changed = _report_payload(
        registry,
        candidate_index=1,
        seed_index=0,
        utility=0.4,
        runtime_label="different-host",
    )
    runs[3] = ExternalTuningRun.from_mapping(
        changed,
        report_file_sha256=hashlib.sha256(
            canonical_json_bytes(changed) + b"\n"
        ).hexdigest(),
        report_path="time-resource-p5-v1/seed-0/report.json",
    )
    with pytest.raises(ValueError, match="runtime identity within tuning seed"):
        select_external_tuning(registry, runs)


def test_selection_uses_all_seeds_and_predeclared_tie_breaks() -> None:
    registry = _registry()
    utilities = [0.70, 0.71, 0.72, 0.73, 0.74, 0.80, 0.79]
    runs = list(_runs(registry, utilities))

    # A lucky single seed for the last candidate cannot beat the three-seed mean.
    lucky = _report_payload(
        registry,
        candidate_index=6,
        seed_index=0,
        utility=0.81,
        online_seconds=7.0,
    )
    runs[18] = ExternalTuningRun.from_mapping(
        lucky,
        report_file_sha256=hashlib.sha256(
            canonical_json_bytes(lucky) + b"\n"
        ).hexdigest(),
        report_path="time-quality-p20-v1/seed-0/report.json",
    )
    receipt = select_external_tuning(registry, runs)
    assert receipt["selected_candidate_id"] == "time-quality-p10-v1"
    assert receipt["seed_selection_forbidden"] is True
    assert receipt["tuning_seeds"] == [1103, 2207, 3301]
    assert len(receipt["source_reports"]) == 21
    assert receipt["power"]["independent_lineages"] == 128
    assert receipt["power"]["attempts"] == 21 * 128 * 4

    # Exact utility and validity ties go to lower time, then registry order.
    tied = _runs(registry, [0.8] * len(registry.candidates))
    tied_receipt = select_external_tuning(registry, tied)
    assert tied_receipt["selected_candidate_id"] == "stock-default-v1"


def test_selection_rejects_partial_census_authority_and_known_cap_violations() -> None:
    registry = _registry()
    runs = list(_runs(registry, [0.5] * len(registry.candidates)))
    with pytest.raises(ValueError, match="complete candidate.*seed census"):
        select_external_tuning(registry, runs[:-1])

    authority_drift = _report_payload(
        registry, candidate_index=1, seed_index=1, utility=0.5
    )
    authority_drift["quality_authority"]["publisher_id"] = "another-publisher"
    authority_drift["quality_authority_digest"] = content_digest(
        authority_drift["quality_authority"]
    )
    authority_drift["record_digest"] = content_digest(
        {
            key: value
            for key, value in authority_drift.items()
            if key != "record_digest"
        }
    )
    runs[4] = ExternalTuningRun.from_mapping(
        authority_drift,
        report_file_sha256=hashlib.sha256(
            canonical_json_bytes(authority_drift) + b"\n"
        ).hexdigest(),
        report_path="time-resource-p5-v1/seed-1/report.json",
    )
    with pytest.raises(ValueError, match="authority|census"):
        select_external_tuning(registry, runs)

    violation = _report_payload(
        registry, candidate_index=1, seed_index=1, utility=0.5
    )
    violation["lineage_metrics"][0]["known_work_cap_violations"] = 1
    violation["summary"] = ExternalTuningRun.summary_from_lineages(
        violation["lineage_metrics"]
    )
    violation["record_digest"] = content_digest(
        {key: value for key, value in violation.items() if key != "record_digest"}
    )
    with pytest.raises(ValueError, match="known work-cap violation"):
        ExternalTuningRun.from_mapping(
            violation,
            report_file_sha256="5" * 64,
            report_path="violating/report.json",
        )


def test_atomic_selection_receipt_requires_out_of_band_sha_before_loading(
    tmp_path: Path,
) -> None:
    registry = _registry()
    runs = _runs(registry, [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.78])
    destination = tmp_path / "selection.json"

    receipt, receipt_sha = write_external_tuning_selection(
        destination, registry=registry, runs=runs
    )
    assert destination.read_bytes() == canonical_json_bytes(receipt) + b"\n"
    assert receipt["record_digest"] == content_digest(
        {key: value for key, value in receipt.items() if key != "record_digest"}
    )
    with pytest.raises(FileExistsError):
        write_external_tuning_selection(destination, registry=registry, runs=runs)
    with pytest.raises(ValueError, match="out-of-band.*SHA-256"):
        load_external_tuning_selection(
            destination,
            expected_file_sha256="0" * 64,
            registry=registry,
        )

    frozen = load_external_tuning_selection(
        destination,
        expected_file_sha256=receipt_sha,
        registry=registry,
    )
    assert frozen.selected_candidate_id == "time-quality-p10-v1"
    assert frozen.selected_candidate == registry.candidates[5]
    assert frozen.test_data_opened is False


def test_selection_cli_exposes_no_test_partition_override() -> None:
    from isingfold.rl.cli import build_parser, cmd_select_external_tuning

    args = build_parser().parse_args(
        [
            "select-external-tuning",
            "--registry",
            str(REGISTRY),
            "--expected-registry-sha256",
            EXTERNAL_TUNING_REGISTRY_FILE_SHA256,
            "--grid",
            str(GRID),
            "--external-config",
            str(EXTERNAL_CONFIG),
            "--evaluations-root",
            "validation-reports",
            "--out",
            "selection.json",
        ]
    )

    assert args.func is cmd_select_external_tuning
    assert not hasattr(args, "partition")
