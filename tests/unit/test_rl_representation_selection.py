from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from isingfold.rl import cli
from isingfold.rl.cli import build_parser
from isingfold.rl.checkpoint import runtime_implementation_registry
from isingfold.rl.contracts import WorkVector, stable_digest
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.evaluate import (
    INITIALIZATION_FAILURE,
    EpisodeOutcome,
    baseline_method_metadata,
    secondary_metrics,
    write_episode_receipts,
)
from isingfold.rl.evaluate import EVALUATION_RECEIPT_VERSION
from tests.unit.test_rl_selector_labels import (
    _global_authority,
    _ground_partition_receipt,
    _partition_authority,
    _target_access,
)

ROOT = Path(__file__).resolve().parents[2]
QUALITY_PREFLIGHT_SHA256 = "6" * 64
QUALITY_PREFLIGHT_RECORD_DIGEST = "7" * 64


@pytest.fixture(autouse=True)
def _strict_complete_receipt_boundary_double(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep selection fixtures small while preserving file/hash/clone checks."""

    def read(path: Path, *, expected_sha256: str):
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected_sha256
        return tuple(
            SimpleNamespace(
                outcome=EpisodeOutcome.from_dict(payload["outcome"]),
                bootstrap_binding=payload["bootstrap_binding"],
            )
            for payload in (json.loads(line) for line in raw.decode("utf-8").splitlines())
        )

    monkeypatch.setattr(cli, "_read_representation_complete_receipts", read)


def _runtime_identity(label: str, *, slurm_partition: str | None = None) -> dict[str, object]:
    return {
        "inference_device_name": f"fixture-{label}",
        "inference_threads": 1,
        "runtime_platform": {
            "hostname": f"fixture-{label}-host",
            "system": "fixture",
            "release": "fixture",
            "machine": "fixture",
            "processor": label,
            "logical_cpu_count": 8,
            "slurm_partition": slurm_partition,
        },
    }


def _quality_args() -> list[str]:
    return [
        "--quality-attestation",
        "publisher-attestation.json",
        "--expected-quality-attestation-digest",
        "a" * 64,
        "--expected-quality-publisher-id",
        "test-publisher",
        "--ground-certificate-root",
        "ground-certificate-root.json",
        "--expected-ground-certificate-root-sha256",
        "b" * 64,
        "--quality-preflight-receipt",
        "quality-preflight.json",
        "--expected-quality-preflight-sha256",
        QUALITY_PREFLIGHT_SHA256,
    ]


def _outcome(
    *,
    utility: float,
    online_seconds: float,
    seed: int,
    index: int = 0,
    repetition: int = 0,
    returned_valid: bool = True,
) -> EpisodeOutcome:
    reads = 10
    hits = round(utility * reads)
    return EpisodeOutcome(
        instance=f"instance-{index}",
        lineage=f"lineage-{index}",
        returned_valid=returned_valid,
        utility=hits / reads if returned_valid else 0.0,
        qubits=6 if returned_valid else None,
        max_chain=2 if returned_valid else None,
        decisions=3,
        selected_strength=2.0 if returned_valid else None,
        reason="commit" if returned_valid else "no-valid-return",
        repetition=repetition,
        episode_seed=seed,
        evaluator_seed=seed + index + 1 if returned_valid else None,
        program_digest=(
            hashlib.sha256(f"program-{seed}-{index}".encode()).hexdigest()
            if returned_valid
            else None
        ),
        selected_strength_index=2 if returned_valid else None,
        evaluator_hits=hits if returned_valid else None,
        evaluator_reads=reads if returned_valid else None,
        work=WorkVector(
            decisions=3, evaluator_reads=reads if returned_valid else 0
        ),
        validation_digest=hashlib.sha256(
            f"validation-{seed}-{index}".encode()
        ).hexdigest(),
        online_seconds=online_seconds,
        evaluator_seconds=0.01 if returned_valid else None,
        broken_chain_fraction=0.0 if returned_valid else None,
        mean_energy_residual=0.1 if returned_valid else None,
        controller_calls=3,
        controller_seconds=online_seconds / 10.0,
    )


def _initialization_failure(*, seed: int, index: int, repetition: int) -> EpisodeOutcome:
    return EpisodeOutcome(
        instance=f"instance-{index}",
        lineage=f"lineage-{index}",
        returned_valid=False,
        utility=None,
        qubits=None,
        max_chain=None,
        decisions=0,
        selected_strength=None,
        reason="initializer-failed",
        repetition=repetition,
        episode_seed=seed,
        evaluator_seed=None,
        program_digest=None,
        selected_strength_index=None,
        evaluator_hits=None,
        evaluator_reads=None,
        work=WorkVector(),
        validation_digest=hashlib.sha256(
            f"init-failure-{seed}-{index}".encode()
        ).hexdigest(),
        online_seconds=0.01,
        evaluator_seconds=None,
        controller_calls=0,
        controller_seconds=0.0,
        outcome_kind=INITIALIZATION_FAILURE,
        population_eligible=False,
        overlap_events=0,
        overlap_decisions=0,
        repair_attempts=0,
        repair_successes=0,
        candidate_states=0,
        legal_actions_total=0,
        legal_opcode_types_total=0,
    )


def _write_evaluations(
    root: Path,
    *,
    partition: str = "validation",
    omitted_cell: str | None = None,
    gate_profile: str = "profile-i",
    family_valid_returns: dict[str, int] | None = None,
    population: int = 1,
    evaluation_seed: int = 33049,
    ineligible_cell: str | None = None,
    runtime_by_seed: dict[int, dict[str, object]] | None = None,
    runtime_mismatch_cell: str | None = None,
) -> None:
    grid_path = ROOT / "configs" / "rl_grid_hybrid_v1.json"
    grid = json.loads(grid_path.read_text())
    grid_digest = hashlib.sha256(grid_path.read_bytes()).hexdigest()
    runtime_registry = runtime_implementation_registry()
    runtime_registry_digest = content_digest(runtime_registry)
    family_utility = {
        "if-mlp": {1103: 0.9, 2207: 0.1, 3301: 0.1},
        "if-dual": {1103: 0.5, 2207: 0.5, 3301: 0.5},
        "if-core": {1103: 0.7, 2207: 0.7, 3301: 0.7},
    }
    valid_returns = family_valid_returns or {
        "if-mlp": population,
        "if-dual": population,
        "if-core": population,
    }
    authority_partition = "val" if partition == "validation" else partition
    authority_payload = {
        "schema": "isingfold.quality-authority-binding",
        "schema_version": 2,
        "global": _global_authority(),
        "evaluation_partition": _partition_authority(
            authority_partition,
            population,
        ),
    }
    quality_authority = {
        **authority_payload,
        "record_digest": content_digest(authority_payload),
    }
    target_access = _target_access(authority_partition, population)
    ground_partition_receipt = _ground_partition_receipt(
        authority_partition,
        population,
    )
    same_support_digest = "8" * 64
    bootstrap_access_body = {
        "schema": "isingfold.rl-value-bootstrap-access",
        "schema_version": 1,
        "manifest_sha256": "1" * 64,
        "manifest_record_digest": "2" * 64,
        "plan_sha256": "3" * 64,
        "plan_record_digest": "4" * 64,
        "prepared_manifest_sha256": "b" * 64,
        "protocol_registry_sha256": grid_digest,
        "protocol_record_digest": content_digest(grid["representation_evaluation"]),
        "same_support_contract_digest": same_support_digest,
        "protocol_preset": "representation-validation",
        "census_digest": "5" * 64,
        "record_root_digest": "9" * 64,
        "source_execution_plan_root": "c" * 64,
        "source_execution_manifest_root": "d" * 64,
        "denominator_count": population * grid["representation_evaluation"]["repetitions"],
        "initial_success_count": population * grid["representation_evaluation"]["repetitions"],
        "initial_failure_count": 0,
        "total_generation_work": {
            "decisions": 0,
            "route_expansions": 0,
            "materializations": 0,
            "compiler_calls": 0,
            "validator_calls": 0,
            "evaluator_reads": 0,
            "cut_edge_visits": 0,
            "restart_work": 0,
            "feature_work": 0,
        },
        "total_precomputed_online_seconds": 0.0,
        "opened_evaluator_targets": False,
    }
    bootstrap_access = {
        **bootstrap_access_body,
        "record_digest": content_digest(bootstrap_access_body),
    }
    for cell in grid["stages"]["representation"]:
        cell_id = cell["cell_id"]
        if cell_id == omitted_cell:
            continue
        family = cell["model_family"]
        seed = cell["seed"]
        cell_root = root / cell_id
        cell_root.mkdir(parents=True)
        repetitions = grid["representation_evaluation"]["repetitions"]
        outcomes = [
            _outcome(
                utility=family_utility[family][seed],
                online_seconds={"if-mlp": 1.0, "if-dual": 1.5, "if-core": 2.0}[family],
                seed=evaluation_seed + repetition,
                index=index,
                repetition=repetition,
                returned_valid=index < valid_returns[family],
            )
            for index in range(population)
            for repetition in range(repetitions)
        ]
        reference = [
            _outcome(
                utility=0.4,
                online_seconds=0.25,
                seed=evaluation_seed + repetition,
                index=index,
                repetition=repetition,
            )
            for index in range(population)
            for repetition in range(repetitions)
        ]
        if cell_id == ineligible_cell:
            for rows in (outcomes, reference):
                rows[0] = _initialization_failure(
                    seed=evaluation_seed,
                    index=0,
                    repetition=0,
                )
        arm_outcomes = {
            "return_initial": reference,
            "random_masked": list(reference),
            "classical_resource_first": list(reference),
            "classical_quality_aware": list(reference),
            "policy": outcomes,
        }
        arm_shas = {
            arm: write_episode_receipts(cell_root / f"{arm}.jsonl", rows)
            for arm, rows in arm_outcomes.items()
        }
        method = {
            "method_id": f"{family}:supervised-ranking",
            "selection_rule": "categorical-temperature-one",
            "tie_break": "registered-episode-rng",
            "support_contract": "environment-materialised-masked-actions",
            "online_evaluator_feedback": False,
            "evaluator_oracle": False,
            "checkpoint_payload_digest": hashlib.sha256(cell_id.encode()).hexdigest(),
        }
        methods = baseline_method_metadata()
        methods["policy"] = method
        runtime_identity = dict(
            (runtime_by_seed or {}).get(seed, _runtime_identity("cpu"))
        )
        if cell_id == runtime_mismatch_cell:
            runtime_identity["inference_device_name"] = "fixture-unmatched-runtime"
        protocol = {
            "population": "b" * 64 + ":validation",
            "quality_authority": quality_authority,
            "target_access": target_access,
            "ground_partition_receipt": ground_partition_receipt,
            "budget": "c" * 64,
            "selector": "a" * 64,
            "proposal": "proposal-fixture",
            "support": "same-conditional-generator-and-exact-mask-contract",
            "policy_seed": evaluation_seed,
            "evaluator_reads": 4096,
            "initializer": "authenticated-persistent-upfront-lac-cache-k2-v2",
            "repetitions": repetitions,
            "inference_device_type": "cpu",
            **runtime_identity,
            "baseline_controller_registry": "d" * 64,
            "evaluation_receipt_schema": EVALUATION_RECEIPT_VERSION,
            "evaluation_implementation_sha256": hashlib.sha256(
                (ROOT / "src" / "isingfold" / "rl" / "evaluate.py").read_bytes()
            ).hexdigest(),
            "quality_preflight_receipt_sha256": QUALITY_PREFLIGHT_SHA256,
            "quality_preflight_record_digest": QUALITY_PREFLIGHT_RECORD_DIGEST,
            "runtime_implementation_registry": runtime_registry,
            "runtime_implementation_digest": runtime_registry_digest,
            "bootstrap_bank_access": bootstrap_access,
            "same_support_contract_digest": same_support_digest,
        }
        payload = {
            "schema": "isingfold.paired-evaluation",
            "schema_version": 4,
            "population": population,
            "partition": partition,
            "repetitions": repetitions,
            "evaluation_seed": evaluation_seed,
            "checkpoint_payload_digest": hashlib.sha256(cell_id.encode()).hexdigest(),
            "selector_digest": "a" * 64,
            "corpus_manifest_sha256": "b" * 64,
            "quality_authority": quality_authority,
            "target_access": target_access,
            "ground_partition_receipt": ground_partition_receipt,
            "deployment_rule": "categorical-temperature-one",
            "matched_metadata": protocol,
            "methods": methods,
            "model_family": family,
            "training_seed": seed,
            "training_method": "supervised-ranking",
            "grid_cell": cell_id,
            "grid_manifest_sha256": grid_digest,
            "gate_profile": gate_profile,
            "gate_receipt_sha256": "e" * 64,
            "gate_record_digest": "f" * 64,
            "quality_preflight_receipt_sha256": QUALITY_PREFLIGHT_SHA256,
            "quality_preflight_record_digest": QUALITY_PREFLIGHT_RECORD_DIGEST,
            "runtime_implementation_registry": runtime_registry,
            "runtime_implementation_digest": runtime_registry_digest,
            "bootstrap_bank_access": bootstrap_access,
            "same_support_contract_digest": same_support_digest,
            "policy": secondary_metrics(outcomes),
            "return_initial": secondary_metrics(reference),
            "arm_receipts": {
                arm: {
                    "path": f"{arm}.jsonl",
                    "sha256": arm_shas[arm],
                    "count": len(rows),
                    "method": methods[arm],
                    "protocol": protocol,
                }
                for arm, rows in arm_outcomes.items()
            },
        }
        for arm, rows in arm_outcomes.items():
            payload[arm] = secondary_metrics(rows)
        complete_registry = {}
        for arm, rows in arm_outcomes.items():
            encoded = []
            for outcome in rows:
                pair_identity = {
                    "lineage": outcome.lineage,
                    "instance": outcome.instance,
                    "repetition": outcome.repetition,
                }
                clone = {
                    "consumer_id": f"representation/{cell_id}/{arm}",
                    "bank_access_record_digest": bootstrap_access["record_digest"],
                    "same_support_contract_digest": same_support_digest,
                    "bootstrap_record_digest": stable_digest(
                        {"bootstrap-record": pair_identity}
                    ),
                    "bootstrap_outcome_record_digest": stable_digest(
                        {"bootstrap-outcome": pair_identity}
                    ),
                    "bootstrap_payload_sha256": stable_digest(
                        {"bootstrap-payload": pair_identity}
                    ),
                    "row_key": stable_digest({"row": pair_identity}),
                }
                encoded.append(
                    canonical_json_bytes(
                        {
                            "outcome": outcome.as_dict(),
                            "bootstrap_binding": {"clone": clone},
                        }
                    )
                )
            complete_content = b"\n".join(encoded) + b"\n"
            complete_path = cell_root / f"{arm}.complete.jsonl"
            complete_path.write_bytes(complete_content)
            complete_registry[arm] = {
                "path": complete_path.name,
                "sha256": hashlib.sha256(complete_content).hexdigest(),
                "count": len(rows),
            }
        payload["complete_system_receipts"] = complete_registry
        record = {**payload, "record_digest": content_digest(payload)}
        (cell_root / "report.json").write_bytes(canonical_json_bytes(record) + b"\n")


def _selection_args(evaluations: Path, output: Path):
    return build_parser().parse_args(
        [
            "select-representation",
            "--grid",
            str(ROOT / "configs" / "rl_grid_hybrid_v1.json"),
            "--evaluations-root",
            str(evaluations),
            "--out",
            str(output),
        ]
    )


def test_selection_aggregates_all_three_seeds_instead_of_taking_best_seed(
    tmp_path: Path,
) -> None:
    evaluations = tmp_path / "evaluations"
    _write_evaluations(evaluations)
    output = tmp_path / "selection.json"

    args = _selection_args(evaluations, output)
    args.func(args)

    receipt = json.loads(output.read_text())
    assert receipt["schema_version"] == 4
    assert receipt["selected_simpler"] == "if-dual"
    assert receipt["family_aggregates"]["if-mlp"]["utility_mean"] == pytest.approx(
        1.1 / 3.0
    )
    assert receipt["family_aggregates"]["if-dual"]["utility_mean"] == 0.5
    assert receipt["family_aggregates"]["if-mlp"]["seed_count"] == 3
    assert receipt["family_aggregates"]["if-mlp"]["feasibility_noninferior"] is True
    assert receipt["feasibility_constraint"]["reference"] == "return_initial"
    assert receipt["feasibility_constraint"]["eligible_simpler"] == [
        "if-mlp",
        "if-dual",
    ]
    assert len(receipt["source_reports"]) == 9
    assert receipt["gate_profile"] == "profile-i"
    assert receipt["gate_receipt_sha256"] == "e" * 64
    assert receipt["gate_record_digest"] == "f" * 64
    unsigned = {key: value for key, value in receipt.items() if key != "record_digest"}
    assert receipt["record_digest"] == content_digest(unsigned)


def test_selection_rejects_missing_or_cross_arm_k2_bootstrap_receipts(
    tmp_path: Path,
) -> None:
    evaluations = tmp_path / "missing"
    _write_evaluations(evaluations)
    cell = "rep-000-if-mlp-s1103"
    report_path = evaluations / cell / "report.json"
    report = json.loads(report_path.read_text())
    report.pop("complete_system_receipts")
    report["record_digest"] = content_digest(
        {key: value for key, value in report.items() if key != "record_digest"}
    )
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    with pytest.raises(ValueError, match="complete receipts for every same-support arm"):
        _selection_args(evaluations, tmp_path / "missing-selection.json").func(
            _selection_args(evaluations, tmp_path / "missing-selection.json")
        )

    mismatch_root = tmp_path / "mismatch"
    _write_evaluations(mismatch_root)
    report_path = mismatch_root / cell / "report.json"
    report = json.loads(report_path.read_text())
    complete_path = mismatch_root / cell / "policy.complete.jsonl"
    rows = [json.loads(line) for line in complete_path.read_text().splitlines()]
    rows[0]["bootstrap_binding"]["clone"]["bootstrap_record_digest"] = "f" * 64
    content = b"\n".join(canonical_json_bytes(row) for row in rows) + b"\n"
    complete_path.write_bytes(content)
    report["complete_system_receipts"]["policy"]["sha256"] = hashlib.sha256(
        content
    ).hexdigest()
    report["record_digest"] = content_digest(
        {key: value for key, value in report.items() if key != "record_digest"}
    )
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    with pytest.raises(ValueError, match="changes bootstrap support across arms"):
        args = _selection_args(mismatch_root, tmp_path / "mismatch-selection.json")
        args.func(args)


def test_complete_evaluation_uses_stage_aware_bootstrap_consumer_namespace() -> None:
    source = (ROOT / "src" / "isingfold" / "rl" / "cli.py").read_text()
    assert 'f"{training_receipt[\'phase\']}/{training_contract[\'grid_cell\']}/{arm_name}"' in source


def test_selection_exposes_the_exact_evaluated_warm_start_payload(
    tmp_path: Path,
) -> None:
    evaluations = tmp_path / "evaluations"
    _write_evaluations(evaluations)
    output = tmp_path / "selection.json"
    args = _selection_args(evaluations, output)
    args.func(args)
    receipt = json.loads(output.read_text())

    payload_digest = cli._representation_checkpoint_payload_digest(
        output,
        expected_selection_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        expected_selection_record_digest=receipt["record_digest"],
        grid_cell="rep-003-if-dual-s1103",
        model_family="if-dual",
        seed=1103,
    )

    assert payload_digest == hashlib.sha256(b"rep-003-if-dual-s1103").hexdigest()


def test_selection_accepts_seed_blocked_runtime_identity(tmp_path: Path) -> None:
    evaluations = tmp_path / "evaluations"
    runtimes = {
        1103: _runtime_identity("apollo"),
        2207: _runtime_identity("goose", slurm_partition="gpu"),
        3301: _runtime_identity("apollo"),
    }
    _write_evaluations(evaluations, runtime_by_seed=runtimes)
    output = tmp_path / "selection.json"

    args = _selection_args(evaluations, output)
    args.func(args)

    receipt = json.loads(output.read_text())
    assert receipt["runtime_identity_by_training_seed"] == {
        str(seed): runtime for seed, runtime in runtimes.items()
    }
    matched = receipt["protocol_identity"]["matched_metadata"]
    assert matched["inference_device_type"] == "cpu"
    assert not {
        "inference_device_name",
        "inference_threads",
        "runtime_platform",
    } & set(matched)


def test_selection_rejects_runtime_mismatch_within_training_seed(tmp_path: Path) -> None:
    evaluations = tmp_path / "evaluations"
    _write_evaluations(
        evaluations,
        runtime_by_seed={
            1103: _runtime_identity("apollo"),
            2207: _runtime_identity("goose", slurm_partition="gpu"),
            3301: _runtime_identity("apollo"),
        },
        runtime_mismatch_cell="rep-003-if-dual-s1103",
    )

    with pytest.raises(ValueError, match="runtime identity within training seed 1103"):
        args = _selection_args(evaluations, tmp_path / "selection.json")
        args.func(args)


def test_representation_loader_rejects_runtime_map_source_disagreement(
    tmp_path: Path,
) -> None:
    evaluations = tmp_path / "evaluations"
    _write_evaluations(evaluations)
    output = tmp_path / "selection.json"
    args = _selection_args(evaluations, output)
    args.func(args)
    forged = json.loads(output.read_text())
    forged["runtime_identity_by_training_seed"]["1103"]["inference_device_name"] = (
        "forged-runtime"
    )
    unsigned = {key: value for key, value in forged.items() if key != "record_digest"}
    forged["record_digest"] = content_digest(unsigned)
    output.write_bytes(canonical_json_bytes(forged) + b"\n")
    grid, grid_digest = cli._load_grid(ROOT / "configs" / "rl_grid_hybrid_v1.json")

    with pytest.raises(ValueError, match="source runtime differs"):
        cli._load_representation_selection(output, grid=grid, grid_digest=grid_digest)


def test_representation_loader_rejects_rehashed_pre_runtime_schema(tmp_path: Path) -> None:
    evaluations = tmp_path / "evaluations"
    _write_evaluations(evaluations)
    output = tmp_path / "selection.json"
    args = _selection_args(evaluations, output)
    args.func(args)
    forged = json.loads(output.read_text())
    forged["schema_version"] = 1
    unsigned = {key: value for key, value in forged.items() if key != "record_digest"}
    forged["record_digest"] = content_digest(unsigned)
    output.write_bytes(canonical_json_bytes(forged) + b"\n")
    grid, grid_digest = cli._load_grid(ROOT / "configs" / "rl_grid_hybrid_v1.json")

    with pytest.raises(ValueError, match="incompatible identity"):
        cli._load_representation_selection(output, grid=grid, grid_digest=grid_digest)


def test_selection_maximizes_utility_only_within_feasibility_shortlist(
    tmp_path: Path,
) -> None:
    evaluations = tmp_path / "evaluations"
    _write_evaluations(
        evaluations,
        population=100,
        family_valid_returns={"if-mlp": 100, "if-dual": 90, "if-core": 100},
    )
    output = tmp_path / "selection.json"

    args = _selection_args(evaluations, output)
    args.func(args)

    receipt = json.loads(output.read_text())
    assert receipt["selected_simpler"] == "if-mlp"
    assert receipt["family_aggregates"]["if-dual"]["utility_mean"] > receipt[
        "family_aggregates"
    ]["if-mlp"]["utility_mean"]
    assert receipt["family_aggregates"]["if-dual"]["feasibility_noninferior"] is False
    assert receipt["feasibility_constraint"]["eligible_simpler"] == ["if-mlp"]


def test_selection_fails_closed_when_no_simpler_family_passes_feasibility(
    tmp_path: Path,
) -> None:
    evaluations = tmp_path / "evaluations"
    _write_evaluations(
        evaluations,
        population=100,
        family_valid_returns={"if-mlp": 90, "if-dual": 90, "if-core": 100},
    )
    output = tmp_path / "selection.json"

    with pytest.raises(ValueError, match="feasibility"):
        args = _selection_args(evaluations, output)
        args.func(args)
    assert not output.exists()


def test_rehashed_receipt_cannot_override_the_registered_selection_rule(
    tmp_path: Path,
) -> None:
    evaluations = tmp_path / "evaluations"
    _write_evaluations(evaluations)
    output = tmp_path / "selection.json"
    args = _selection_args(evaluations, output)
    args.func(args)
    forged = json.loads(output.read_text())
    forged["selected_simpler"] = "if-mlp"
    forged["retained_families"] = ["if-mlp", "if-core"]
    unsigned = {key: value for key, value in forged.items() if key != "record_digest"}
    forged["record_digest"] = content_digest(unsigned)
    output.write_bytes(canonical_json_bytes(forged) + b"\n")
    grid, grid_digest = cli._load_grid(ROOT / "configs" / "rl_grid_hybrid_v1.json")

    with pytest.raises(ValueError, match="registered rule"):
        cli._load_representation_selection(
            output,
            grid=grid,
            grid_digest=grid_digest,
        )


def test_selection_rejects_a_missing_registered_seed(tmp_path: Path) -> None:
    evaluations = tmp_path / "evaluations"
    _write_evaluations(evaluations, omitted_cell="rep-001-if-mlp-s2207")

    with pytest.raises((FileNotFoundError, ValueError), match="rep-001-if-mlp-s2207"):
        _selection_args(evaluations, tmp_path / "selection.json").func(
            _selection_args(evaluations, tmp_path / "selection.json")
        )


def test_selection_rejects_test_partition_receipts(tmp_path: Path) -> None:
    evaluations = tmp_path / "evaluations"
    _write_evaluations(evaluations, partition="test")

    with pytest.raises(ValueError, match="validation"):
        args = _selection_args(evaluations, tmp_path / "selection.json")
        args.func(args)


def test_selection_rejects_a_non_profile_i_gate_scope(tmp_path: Path) -> None:
    evaluations = tmp_path / "evaluations"
    _write_evaluations(evaluations, gate_profile="profile-c")

    with pytest.raises(ValueError, match="Profile-I gate"):
        args = _selection_args(evaluations, tmp_path / "selection.json")
        args.func(args)


def test_selection_rejects_nine_uniformly_wrong_evaluation_seeds(tmp_path: Path) -> None:
    evaluations = tmp_path / "evaluations"
    _write_evaluations(evaluations, evaluation_seed=44022)

    with pytest.raises(ValueError, match="identity differs"):
        args = _selection_args(evaluations, tmp_path / "selection.json")
        args.func(args)


def test_selection_rejects_cell_specific_population_denominator(tmp_path: Path) -> None:
    evaluations = tmp_path / "evaluations"
    _write_evaluations(
        evaluations,
        population=2,
        ineligible_cell="rep-008-if-core-s3301",
    )

    with pytest.raises(ValueError, match="exact task/seed pairs|return-initial reference"):
        args = _selection_args(evaluations, tmp_path / "selection.json")
        args.func(args)


def test_rl_grid_cell_requires_and_binds_the_selection_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evaluations = tmp_path / "evaluations"
    _write_evaluations(evaluations)
    selection = tmp_path / "selection.json"
    selection_args = _selection_args(evaluations, selection)
    selection_args.func(selection_args)
    captured = []
    monkeypatch.setattr(cli, "cmd_train", captured.append)
    monkeypatch.setattr(cli, "_context", lambda *args: object())
    monkeypatch.setattr(cli, "_load_selector_bundle", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        cli,
        "_load_quality_preflight_receipt",
        lambda *args, **kwargs: {"record_digest": QUALITY_PREFLIGHT_RECORD_DIGEST},
    )
    monkeypatch.setattr(
        cli,
        "_load_representation_selection",
        lambda *args, **kwargs: ("if-dual", "a" * 64, "b" * 64),
    )
    monkeypatch.setattr(
        cli,
        "_representation_checkpoint_payload_digest",
        lambda *args, **kwargs: "c" * 64,
    )
    complete_config = ROOT / "configs" / "complete_system_lac_hybrid_cache_v1.json"
    complete_config_sha256 = hashlib.sha256(complete_config.read_bytes()).hexdigest()
    monkeypatch.setattr(
        cli,
        "_sha256_file",
        lambda path: (
            complete_config_sha256
            if Path(path) == complete_config
            else QUALITY_PREFLIGHT_SHA256
        ),
    )
    base = [
        "grid-cell",
        "--grid",
        str(ROOT / "configs" / "rl_grid_hybrid_v1.json"),
        "--stage",
        "rl_value",
        "--index",
        "0",
        "--corpus",
        "prepared",
        "--selector",
        "selector",
        *_quality_args(),
        "--quality-labels",
        "quality",
        "--run-root",
        str(tmp_path / "runs"),
        "--initializer-bank",
        str(tmp_path / "initializer-banks" / "seed-0"),
        "--expected-initializer-bank-manifest-sha256",
        "f" * 64,
        "--complete-config",
        str(complete_config),
    ]

    with pytest.raises(ValueError, match="selection receipt"):
        missing = build_parser().parse_args(base)
        missing.func(missing)

    args = build_parser().parse_args(
        [*base, "--selection-receipt", str(selection)]
    )
    args.func(args)

    assert len(captured) == 1
    assert captured[0].model_family == "if-dual"
    assert len(captured[0].selection_receipt_sha256) == 64
    assert len(captured[0].selection_record_digest) == 64
    assert captured[0].warm_start_grid_cell == "rep-003-if-dual-s1103"
    assert captured[0].warm_start_checkpoint_payload_digest == "c" * 64


def test_manual_simpler_choice_requires_explicit_diagnostic_opt_in(tmp_path: Path) -> None:
    base = [
        "grid-cell",
        "--grid",
        str(ROOT / "configs" / "rl_grid_hybrid_v1.json"),
        "--stage",
        "rl_value",
        "--index",
        "0",
        "--corpus",
        "prepared",
        "--selector",
        "selector",
        *_quality_args(),
        "--quality-labels",
        "quality",
        "--run-root",
        str(tmp_path / "runs"),
        "--selected-simpler",
        "if-mlp",
    ]

    with pytest.raises(ValueError, match="diagnostic"):
        args = build_parser().parse_args(base)
        args.func(args)
