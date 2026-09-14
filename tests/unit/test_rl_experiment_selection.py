from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from isingfold.rl.checkpoint import runtime_implementation_registry
from isingfold.rl.contracts import WorkVector, stable_digest
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.evaluate import (
    EVALUATION_RECEIPT_VERSION,
    EpisodeOutcome,
    baseline_method_metadata,
    secondary_metrics,
    write_episode_receipts,
)
from isingfold.rl.experiment_selection import (
    _load_grid as _load_selection_grid,
    freeze_rl_value_experiment,
    load_rl_value_freeze,
)
from tests.unit.test_rl_selector_labels import (
    _global_authority,
    _ground_partition_receipt,
    _partition_authority,
    _target_access,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def _strict_complete_receipt_boundary_double(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep selection unit fixtures small; integration tests exercise the real parser."""

    from isingfold.rl import experiment_selection as selection_module

    def read(path: Path, *, expected_sha256: str):
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected_sha256
        result = []
        for line in raw.decode("utf-8").splitlines():
            payload = json.loads(line)
            result.append(
                SimpleNamespace(
                    outcome=EpisodeOutcome.from_dict(payload["outcome"]),
                    bootstrap_binding=payload["bootstrap_binding"],
                )
            )
        return tuple(result)

    monkeypatch.setattr(selection_module, "_read_complete_receipts_for_freeze", read)


def _evaluation_authority() -> dict[str, object]:
    payload = {
        "schema": "isingfold.quality-authority-binding",
        "schema_version": 2,
        "global": _global_authority(),
        "evaluation_partition": _partition_authority("val", 4),
    }
    return {**payload, "record_digest": content_digest(payload)}


QUALITY_AUTHORITY = _evaluation_authority()
QUALITY_TARGET_ACCESS = _target_access("val", 4)
QUALITY_GROUND_PARTITION_RECEIPT = _ground_partition_receipt("val", 4)
QUALITY_PREFLIGHT_SHA256 = "6" * 64
QUALITY_PREFLIGHT_RECORD_DIGEST = "7" * 64


def test_selection_loader_requires_registered_staged_grid_v2(tmp_path: Path) -> None:
    registered = ROOT / "configs" / "rl_grid_hybrid_v1.json"
    grid, _, _ = _load_selection_grid(registered)
    assert grid["schema_version"] == 2
    assert grid["name"] == "if-core-v2-profile-i-hybrid-chimera-registered"

    retired = json.loads(registered.read_text())
    retired["schema_version"] = 1
    path = tmp_path / "retired-grid.json"
    path.write_text(json.dumps(retired))
    with pytest.raises(ValueError, match="unsupported staged-grid"):
        _load_selection_grid(path)

    retired["schema_version"] = 2
    retired["name"] = "if-core-v1-profile-i-pilot"
    path.write_text(json.dumps(retired))
    with pytest.raises(ValueError, match="unsupported staged-grid"):
        _load_selection_grid(path)


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


def _outcome(
    *,
    utility: float,
    online_seconds: float,
    lineage_index: int,
    repetition: int,
    returned_valid: bool = True,
) -> EpisodeOutcome:
    reads = 100
    hits = round(utility * reads)
    episode_seed = 70_000 + 100 * lineage_index + repetition
    evaluator_seed = 80_000 + 100 * lineage_index + repetition
    return EpisodeOutcome(
        instance=f"instance-{lineage_index}",
        lineage=f"lineage-{lineage_index}",
        returned_valid=returned_valid,
        utility=hits / reads if returned_valid else 0.0,
        qubits=8 if returned_valid else None,
        max_chain=3 if returned_valid else None,
        decisions=4,
        selected_strength=2.0 if returned_valid else None,
        reason="commit" if returned_valid else "no-valid-return",
        repetition=repetition,
        episode_seed=episode_seed,
        evaluator_seed=evaluator_seed if returned_valid else None,
        program_digest=(
            hashlib.sha256(f"program-{lineage_index}-{repetition}-{utility}".encode()).hexdigest()
            if returned_valid
            else None
        ),
        selected_strength_index=2 if returned_valid else None,
        evaluator_hits=hits if returned_valid else None,
        evaluator_reads=reads if returned_valid else None,
        work=WorkVector(decisions=4, evaluator_reads=reads if returned_valid else 0),
        validation_digest=hashlib.sha256(
            f"validation-{lineage_index}-{repetition}-{returned_valid}".encode()
        ).hexdigest(),
        online_seconds=online_seconds,
        evaluator_seconds=0.01 if returned_valid else None,
        broken_chain_fraction=0.0 if returned_valid else None,
        mean_energy_residual=0.1 if returned_valid else None,
        controller_calls=4,
        controller_seconds=online_seconds / 2.0,
    )


def _representation_receipt(path: Path, grid_path: Path) -> None:
    grid = json.loads(grid_path.read_text())
    grid_digest = hashlib.sha256(grid_path.read_bytes()).hexdigest()
    protocol = grid["representation_evaluation"]
    seeds = [1103, 2207, 3301]
    runtime_by_seed = {
        str(seed): _runtime_identity("cpu") for seed in seeds
    }
    runtime_registry = runtime_implementation_registry()
    runtime_registry_digest = content_digest(runtime_registry)
    aggregates = {}
    for family, utility, cost in (
        ("if-mlp", 0.4, 1.0),
        ("if-dual", 0.5, 1.2),
        ("if-core", 0.6, 1.8),
    ):
        aggregates[family] = {
            "seed_count": 3,
            "seeds": seeds,
            "independent_lineages": 4,
            "valid_return_rate": 1.0,
            "utility_mean": utility,
            "online_seconds_mean": cost,
            "feasibility_delta_vs_return_initial": 0.0,
            "feasibility_ci_low": 0.0,
            "feasibility_ci_high": 0.0,
            "feasibility_noninferior": True,
            "per_seed": [
                {
                    "cell_id": cell["cell_id"],
                    "seed": cell["seed"],
                    "valid_return_rate": 1.0,
                    "utility_mean": utility,
                    "online_seconds_mean": cost,
                    "eligible_episodes": 4,
                    "attempts": 4,
                }
                for cell in grid["stages"]["representation"]
                if cell["model_family"] == family
            ],
        }
    payload = {
        "schema": "isingfold.representation-selection",
        "schema_version": 4,
        "grid_manifest_sha256": grid_digest,
        "stage": "representation",
        "partition": "validation",
        "selection_rule": (
            "equal-seed/equal-base-lineage unconditional utility among families passing "
            "the preregistered one-sided feasibility constraint; lower online time then "
            "IF-MLP as deterministic ties"
        ),
        "feasibility_constraint": {
            "reference": protocol["feasibility_reference"],
            "margin": float(protocol["margin"]),
            "one_sided_alpha": float(protocol["feasibility_alpha"]),
            "cluster_unit": "immutable-base-lineage",
            "training_seed_treatment": "equal-weight-crossed-resampling-with-replacement",
            "lineage_treatment": "equal-weight-crossed-resampling-with-replacement",
            "bootstrap_replicates": int(protocol["feasibility_bootstrap"]),
            "bootstrap_seed": int(protocol["selection_bootstrap_seed"]),
            "pass_rule": "lower_bound_strictly_greater_than_negative_margin",
            "eligible_simpler": ["if-mlp", "if-dual"],
            "if_core_pass": True,
        },
        "seed_selection_forbidden": True,
        "selected_simpler": "if-dual",
        "retained_families": ["if-dual", "if-core"],
        "gate_profile": "profile-i",
        "gate_receipt_sha256": "e" * 64,
        "gate_record_digest": "f" * 64,
        "quality_preflight_receipt_sha256": QUALITY_PREFLIGHT_SHA256,
        "quality_preflight_record_digest": QUALITY_PREFLIGHT_RECORD_DIGEST,
        "runtime_implementation_registry": runtime_registry,
        "runtime_implementation_digest": runtime_registry_digest,
        "family_aggregates": aggregates,
        "protocol_identity": {
            "population": 4,
            "repetitions": 1,
            "selector_digest": "a" * 64,
            "corpus_manifest_sha256": "b" * 64,
            "deployment_rule": "categorical-temperature-one",
            "gate_profile": "profile-i",
            "gate_receipt_sha256": "e" * 64,
            "gate_record_digest": "f" * 64,
            "matched_metadata": {
                "population": f"{'b' * 64}:validation",
                "quality_authority": QUALITY_AUTHORITY,
                "target_access": QUALITY_TARGET_ACCESS,
                "ground_partition_receipt": QUALITY_GROUND_PARTITION_RECEIPT,
                "budget": "c" * 64,
                "selector": "a" * 64,
                "proposal": "proposal-fixture",
                "support": "same-conditional-generator-and-exact-mask-contract",
                "policy_seed": 44021,
                "evaluator_reads": 4096,
                "initializer": "authenticated-profile-i",
                "repetitions": 1,
                "inference_device_type": "cpu",
                "baseline_controller_registry": "d" * 64,
                "evaluation_receipt_schema": EVALUATION_RECEIPT_VERSION,
                "evaluation_implementation_sha256": "9" * 64,
                "quality_preflight_receipt_sha256": QUALITY_PREFLIGHT_SHA256,
                "quality_preflight_record_digest": QUALITY_PREFLIGHT_RECORD_DIGEST,
                "runtime_implementation_registry": runtime_registry,
                "runtime_implementation_digest": runtime_registry_digest,
            },
        },
        "runtime_identity_by_training_seed": runtime_by_seed,
        "source_reports": [
            {
                "cell_id": cell["cell_id"],
                "model_family": cell["model_family"],
                "seed": cell["seed"],
                "report_path": f"representation/{cell['cell_id']}/report.json",
                "report_sha256": hashlib.sha256(cell["cell_id"].encode()).hexdigest(),
                "report_record_digest": hashlib.sha256(
                    f"record-{cell['cell_id']}".encode()
                ).hexdigest(),
                "policy_receipt_sha256": hashlib.sha256(
                    f"policy-{cell['cell_id']}".encode()
                ).hexdigest(),
                "checkpoint_payload_digest": hashlib.sha256(
                    f"checkpoint-{cell['cell_id']}".encode()
                ).hexdigest(),
                "runtime_identity": runtime_by_seed[str(cell["seed"])],
                "runtime_implementation_registry": runtime_registry,
                "runtime_implementation_digest": runtime_registry_digest,
            }
            for cell in grid["stages"]["representation"]
        ],
    }
    receipt = {**payload, "record_digest": content_digest(payload)}
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")


def _write_grid_artifacts(
    root: Path,
    *,
    omit_cell: str | None = None,
    mismatch_budget_cell: str | None = None,
    mismatch_selection_cell: str | None = None,
    partition: str = "validation",
    tie_utility_for_cost: bool = False,
    runtime_by_seed: dict[int, dict[str, object]] | None = None,
    runtime_mismatch_cell: str | None = None,
) -> tuple[Path, Path, Path]:
    grid_path = ROOT / "configs" / "rl_grid_hybrid_v1.json"
    grid = json.loads(grid_path.read_text())
    grid_digest = hashlib.sha256(grid_path.read_bytes()).hexdigest()
    selection_path = root / "representation-selection.json"
    _representation_receipt(selection_path, grid_path)
    representation = json.loads(selection_path.read_text())
    runtime_registry = representation["runtime_implementation_registry"]
    runtime_registry_digest = representation["runtime_implementation_digest"]
    representation_sha = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    evaluations_root = root / "evaluations"
    protocol_config = grid["rl_value_evaluation"]
    same_support_digest = "6" * 64
    bootstrap_access_body = {
        "schema": "isingfold.rl-value-bootstrap-access",
        "schema_version": 1,
        "manifest_sha256": "1" * 64,
        "manifest_record_digest": "2" * 64,
        "plan_sha256": "3" * 64,
        "plan_record_digest": "4" * 64,
        "prepared_manifest_sha256": "b" * 64,
        "protocol_registry_sha256": grid_digest,
        "protocol_record_digest": content_digest(protocol_config),
        "same_support_contract_digest": same_support_digest,
        "protocol_preset": "validation",
        "census_digest": "5" * 64,
        "record_root_digest": "7" * 64,
        "source_execution_plan_root": "8" * 64,
        "source_execution_manifest_root": "9" * 64,
        "denominator_count": 4 * protocol_config["repetitions"],
        "initial_success_count": 4 * protocol_config["repetitions"],
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
    utility_by_configuration = {
        ("selected-simpler", "supervised-only"): {1103: 0.99, 2207: 0.1, 3301: 0.1},
        ("selected-simpler", "ppo-warm-start"): {1103: 0.5, 2207: 0.5, 3301: 0.5},
        ("selected-simpler", "ppo-from-scratch"): {1103: 0.45, 2207: 0.45, 3301: 0.45},
        ("if-core", "supervised-only"): {1103: 0.48, 2207: 0.48, 3301: 0.48},
        ("if-core", "ppo-warm-start"): {1103: 1.0, 2207: 1.0, 3301: 1.0},
        ("if-core", "ppo-from-scratch"): {1103: 0.4, 2207: 0.4, 3301: 0.4},
    }
    cost_by_configuration = {
        key: 1.0 + index / 10.0 for index, key in enumerate(utility_by_configuration)
    }
    if tie_utility_for_cost:
        utility_by_configuration[("if-core", "supervised-only")] = {
            1103: 0.5,
            2207: 0.5,
            3301: 0.5,
        }
        cost_by_configuration[("if-core", "supervised-only")] = 0.25
    for cell in grid["stages"]["rl_value"]:
        cell_id = cell["cell_id"]
        if cell_id == omit_cell:
            continue
        grid_family = cell["model_family"]
        family = "if-dual" if grid_family == "selected-simpler" else grid_family
        method_name = cell["method"]
        seed = cell["seed"]
        utility = utility_by_configuration[(grid_family, method_name)][seed]
        cost = cost_by_configuration[(grid_family, method_name)]
        cell_root = evaluations_root / cell_id
        cell_root.mkdir(parents=True)
        policy = []
        reference = []
        for repetition in range(protocol_config["repetitions"]):
            for lineage_index in range(4):
                valid = not (
                    grid_family == "if-core"
                    and method_name == "ppo-warm-start"
                    and lineage_index == 0
                )
                policy.append(
                    _outcome(
                        utility=utility,
                        online_seconds=cost,
                        lineage_index=lineage_index,
                        repetition=repetition,
                        returned_valid=valid,
                    )
                )
                reference.append(
                    _outcome(
                        utility=0.3,
                        online_seconds=0.2,
                        lineage_index=lineage_index,
                        repetition=repetition,
                    )
                )
        arm_outcomes = {
            "return_initial": reference,
            "random_masked": list(reference),
            "classical_resource_first": list(reference),
            "classical_quality_aware": list(reference),
            "policy": policy,
        }
        arm_shas = {
            arm: write_episode_receipts(cell_root / f"{arm}.jsonl", outcomes)
            for arm, outcomes in arm_outcomes.items()
        }
        checkpoint_digest = hashlib.sha256(f"checkpoint-{cell_id}".encode()).hexdigest()
        policy_method = {
            "method_id": f"{family}:{method_name}",
            "selection_rule": "categorical-temperature-one",
            "tie_break": "registered-episode-rng",
            "support_contract": "environment-materialised-masked-actions",
            "online_evaluator_feedback": False,
            "evaluator_oracle": False,
            "checkpoint_payload_digest": checkpoint_digest,
        }
        methods = baseline_method_metadata()
        methods["policy"] = policy_method
        runtime_identity = dict(
            (runtime_by_seed or {}).get(seed, _runtime_identity("cpu"))
        )
        if cell_id == runtime_mismatch_cell:
            runtime_identity["inference_device_name"] = "fixture-unmatched-runtime"
        matched = {
            "population": f"{'b' * 64}:validation",
            "quality_authority": QUALITY_AUTHORITY,
            "target_access": QUALITY_TARGET_ACCESS,
            "ground_partition_receipt": QUALITY_GROUND_PARTITION_RECEIPT,
            "budget": "wrong" if cell_id == mismatch_budget_cell else "c" * 64,
            "selector": "a" * 64,
            "proposal": "proposal-fixture",
            "support": "same-conditional-generator-and-exact-mask-contract",
            "policy_seed": protocol_config["evaluation_seed"],
            "evaluator_reads": protocol_config["audit_reads"],
            "initializer": "authenticated-persistent-upfront-lac-cache-k2-v2",
            "repetitions": protocol_config["repetitions"],
            "inference_device_type": "cpu",
            **runtime_identity,
            "baseline_controller_registry": "d" * 64,
            "evaluation_receipt_schema": EVALUATION_RECEIPT_VERSION,
            "evaluation_implementation_sha256": "9" * 64,
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
            "population": 4,
            "partition": partition,
            "repetitions": protocol_config["repetitions"],
            "model_family": family,
            "training_method": method_name,
            "training_seed": seed,
            "grid_cell": cell_id,
            "grid_manifest_sha256": grid_digest,
            "selection_receipt_sha256": (
                "0" * 64 if cell_id == mismatch_selection_cell else representation_sha
            ),
            "selection_record_digest": representation["record_digest"],
            "selection_mode": "validated-receipt",
            "gate_profile": None,
            "gate_receipt_sha256": None,
            "gate_record_digest": None,
            "evaluation_seed": protocol_config["evaluation_seed"],
            "inference_device_type": "cpu",
            "inference_device_name": runtime_identity["inference_device_name"],
            "runtime_platform": matched["runtime_platform"],
            "checkpoint_payload_digest": checkpoint_digest,
            "selector_digest": "a" * 64,
            "corpus_manifest_sha256": "b" * 64,
            "quality_authority": QUALITY_AUTHORITY,
            "target_access": QUALITY_TARGET_ACCESS,
            "ground_partition_receipt": QUALITY_GROUND_PARTITION_RECEIPT,
            "quality_preflight_receipt_sha256": QUALITY_PREFLIGHT_SHA256,
            "quality_preflight_record_digest": QUALITY_PREFLIGHT_RECORD_DIGEST,
            "runtime_implementation_registry": runtime_registry,
            "runtime_implementation_digest": runtime_registry_digest,
            "deployment_rule": protocol_config["deployment_rule"],
            "bootstrap_bank_access": bootstrap_access,
            "same_support_contract_digest": same_support_digest,
            "matched_metadata": matched,
            "methods": methods,
            "arm_receipts": {
                arm: {
                    "path": f"{arm}.jsonl",
                    "sha256": arm_shas[arm],
                    "count": len(outcomes),
                    "method": methods[arm],
                    "protocol": matched,
                }
                for arm, outcomes in arm_outcomes.items()
            },
        }
        for arm, outcomes in arm_outcomes.items():
            payload[arm] = secondary_metrics(outcomes)
        complete_registry = {}
        for arm, outcomes in arm_outcomes.items():
            rows = []
            for outcome in outcomes:
                pair_identity = {
                    "lineage": outcome.lineage,
                    "instance": outcome.instance,
                    "repetition": outcome.repetition,
                }
                clone = {
                    "consumer_id": f"rl-value/{cell_id}/{arm}",
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
                rows.append(
                    canonical_json_bytes(
                        {
                            "outcome": outcome.as_dict(),
                            "bootstrap_binding": {"clone": clone},
                        }
                    )
                )
            complete_content = b"\n".join(rows) + b"\n"
            complete_path = cell_root / f"{arm}.complete.jsonl"
            complete_path.write_bytes(complete_content)
            complete_registry[arm] = {
                "path": complete_path.name,
                "sha256": hashlib.sha256(complete_content).hexdigest(),
                "count": len(outcomes),
            }
        payload["complete_system_receipts"] = complete_registry
        report = {**payload, "record_digest": content_digest(payload)}
        (cell_root / "report.json").write_bytes(canonical_json_bytes(report) + b"\n")
    return grid_path, selection_path, evaluations_root


def test_freeze_uses_all_seeds_and_feasibility_before_utility(tmp_path: Path) -> None:
    grid, representation, evaluations = _write_grid_artifacts(tmp_path)
    output = tmp_path / "rl-value-freeze.json"

    receipt = freeze_rl_value_experiment(
        grid_path=grid,
        representation_receipt_path=representation,
        evaluations_root=evaluations,
        output_path=output,
    )

    assert output.is_file()
    assert receipt["selected_configuration"]["model_family"] == "if-dual"
    assert receipt["selected_configuration"]["method"] == "ppo-warm-start"
    assert receipt["selected_configuration"]["training_seeds"] == [1103, 2207, 3301]
    assert receipt["seed_selection_forbidden"] is True
    aggregate = receipt["configuration_aggregates"]["if-dual/ppo-warm-start"]
    assert aggregate["unconditional_utility_mean"] == pytest.approx(0.5)
    best_seed_only = receipt["configuration_aggregates"]["if-dual/supervised-only"]
    assert best_seed_only["unconditional_utility_mean"] == pytest.approx(1.19 / 3.0)
    rejected = receipt["configuration_aggregates"]["if-core/ppo-warm-start"]
    assert rejected["unconditional_utility_mean"] > aggregate["unconditional_utility_mean"]
    assert rejected["feasibility_noninferior"] is False
    assert len(receipt["source_reports"]) == 18
    assert len({row["cell_id"] for row in receipt["source_reports"]}) == 18
    assert all(len(row["checkpoint"]["payload_digest"]) == 64 for row in receipt["source_reports"])
    assert receipt["selected_configuration"]["single_checkpoint_selected"] is False
    assert receipt["bootstrap_protocol"]["replicates"] == 20_000
    unsigned = {key: value for key, value in receipt.items() if key != "record_digest"}
    assert receipt["record_digest"] == content_digest(unsigned)
    assert json.loads(output.read_text()) == receipt

    repeated = freeze_rl_value_experiment(
        grid_path=grid,
        representation_receipt_path=representation,
        evaluations_root=evaluations,
        output_path=tmp_path / "same-evidence-second-freeze.json",
    )
    assert repeated == receipt


def test_freeze_rejects_missing_or_cross_arm_bootstrap_receipts(tmp_path: Path) -> None:
    grid, representation, evaluations = _write_grid_artifacts(tmp_path)
    cell = "rl-000-selected-simpler-supervised-s1103"
    report_path = evaluations / cell / "report.json"
    report = json.loads(report_path.read_text())
    report.pop("complete_system_receipts")
    report["record_digest"] = content_digest(
        {key: value for key, value in report.items() if key != "record_digest"}
    )
    report_path.write_bytes(canonical_json_bytes(report) + b"\n")
    with pytest.raises(ValueError, match="complete receipts for every same-support arm"):
        freeze_rl_value_experiment(
            grid_path=grid,
            representation_receipt_path=representation,
            evaluations_root=evaluations,
            output_path=tmp_path / "missing-freeze.json",
        )

    mismatch_root = tmp_path / "mismatch"
    mismatch_root.mkdir()
    grid, representation, evaluations = _write_grid_artifacts(mismatch_root)
    report_path = evaluations / cell / "report.json"
    report = json.loads(report_path.read_text())
    complete_path = evaluations / cell / "policy.complete.jsonl"
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
        freeze_rl_value_experiment(
            grid_path=grid,
            representation_receipt_path=representation,
            evaluations_root=evaluations,
            output_path=tmp_path / "mismatch-freeze.json",
        )


def test_freeze_accepts_seed_blocked_runtime_identity(tmp_path: Path) -> None:
    runtimes = {
        1103: _runtime_identity("apollo"),
        2207: _runtime_identity("goose", slurm_partition="gpu"),
        3301: _runtime_identity("apollo"),
    }
    grid, representation, evaluations = _write_grid_artifacts(
        tmp_path, runtime_by_seed=runtimes
    )

    receipt = freeze_rl_value_experiment(
        grid_path=grid,
        representation_receipt_path=representation,
        evaluations_root=evaluations,
        output_path=tmp_path / "rl-value-freeze.json",
    )

    assert receipt["schema_version"] == 4
    expected = {str(seed): runtime for seed, runtime in runtimes.items()}
    assert receipt["runtime_identity_by_training_seed"] == expected
    assert receipt["online_cost_comparability"] == {
        "required": True,
        "blocking_unit": "training-seed",
        "inference_device_type": "cpu",
        "runtime_identity_by_training_seed": expected,
    }
    assert "runtime_platform" not in receipt["matched_validation_protocol"]


def test_freeze_rejects_runtime_mismatch_within_training_seed(tmp_path: Path) -> None:
    grid, representation, evaluations = _write_grid_artifacts(
        tmp_path,
        runtime_by_seed={
            1103: _runtime_identity("apollo"),
            2207: _runtime_identity("goose", slurm_partition="gpu"),
            3301: _runtime_identity("apollo"),
        },
        runtime_mismatch_cell="rl-003-selected-simpler-warm-ppo-s1103",
    )

    with pytest.raises(ValueError, match="runtime identity within training seed 1103"):
        freeze_rl_value_experiment(
            grid_path=grid,
            representation_receipt_path=representation,
            evaluations_root=evaluations,
            output_path=tmp_path / "rl-value-freeze.json",
        )


def test_rl_value_loader_rejects_runtime_map_source_disagreement(tmp_path: Path) -> None:
    grid, representation, evaluations = _write_grid_artifacts(tmp_path)
    output = tmp_path / "rl-value-freeze.json"
    freeze_rl_value_experiment(
        grid_path=grid,
        representation_receipt_path=representation,
        evaluations_root=evaluations,
        output_path=output,
    )
    forged = json.loads(output.read_text())
    forged["runtime_identity_by_training_seed"]["1103"]["inference_device_name"] = (
        "forged-runtime"
    )
    forged["online_cost_comparability"]["runtime_identity_by_training_seed"] = forged[
        "runtime_identity_by_training_seed"
    ]
    unsigned = {key: value for key, value in forged.items() if key != "record_digest"}
    forged["record_digest"] = content_digest(unsigned)
    output.write_bytes(canonical_json_bytes(forged) + b"\n")

    with pytest.raises(ValueError, match="source runtime differs"):
        load_rl_value_freeze(
            receipt_path=output,
            grid_path=grid,
            expected_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        )


def test_rl_value_loader_rejects_rehashed_pre_runtime_schema(tmp_path: Path) -> None:
    grid, representation, evaluations = _write_grid_artifacts(tmp_path)
    output = tmp_path / "rl-value-freeze.json"
    freeze_rl_value_experiment(
        grid_path=grid,
        representation_receipt_path=representation,
        evaluations_root=evaluations,
        output_path=output,
    )
    forged = json.loads(output.read_text())
    forged["schema_version"] = 1
    unsigned = {key: value for key, value in forged.items() if key != "record_digest"}
    forged["record_digest"] = content_digest(unsigned)
    output.write_bytes(canonical_json_bytes(forged) + b"\n")

    with pytest.raises(ValueError, match="incompatible scientific identity"):
        load_rl_value_freeze(
            receipt_path=output,
            grid_path=grid,
            expected_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        )


def test_post_selection_loader_requires_external_pin_and_all_three_seeds(
    tmp_path: Path,
) -> None:
    grid, representation, evaluations = _write_grid_artifacts(tmp_path)
    output = tmp_path / "rl-value-freeze.json"
    receipt = freeze_rl_value_experiment(
        grid_path=grid,
        representation_receipt_path=representation,
        evaluations_root=evaluations,
        output_path=output,
    )
    expected_sha = hashlib.sha256(output.read_bytes()).hexdigest()

    selected = load_rl_value_freeze(
        receipt_path=output,
        grid_path=grid,
        expected_sha256=expected_sha,
    )

    assert selected.model_family == "if-dual"
    assert selected.method == "ppo-warm-start"
    assert selected.training_seeds == (1103, 2207, 3301)
    assert selected.checkpoint_payload_digests == tuple(
        receipt["selected_configuration"]["checkpoint_payload_digests"]
    )
    with pytest.raises(ValueError, match="pinned"):
        load_rl_value_freeze(
            receipt_path=output,
            grid_path=grid,
            expected_sha256="0" * 64,
        )


def test_post_selection_loader_rejects_rehashed_single_seed_choice(tmp_path: Path) -> None:
    grid, representation, evaluations = _write_grid_artifacts(tmp_path)
    output = tmp_path / "rl-value-freeze.json"
    freeze_rl_value_experiment(
        grid_path=grid,
        representation_receipt_path=representation,
        evaluations_root=evaluations,
        output_path=output,
    )
    forged = json.loads(output.read_text())
    forged["selected_configuration"]["training_seeds"] = [1103]
    unsigned = {key: value for key, value in forged.items() if key != "record_digest"}
    forged["record_digest"] = content_digest(unsigned)
    output.write_bytes(canonical_json_bytes(forged) + b"\n")

    with pytest.raises(ValueError, match="registered configuration decision"):
        load_rl_value_freeze(
            receipt_path=output,
            grid_path=grid,
            expected_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
        )


def test_freeze_fails_closed_on_incomplete_census(tmp_path: Path) -> None:
    missing = "rl-004-selected-simpler-warm-ppo-s2207"
    grid, representation, evaluations = _write_grid_artifacts(tmp_path, omit_cell=missing)
    output = tmp_path / "rl-value-freeze.json"

    with pytest.raises((FileNotFoundError, ValueError), match=missing):
        freeze_rl_value_experiment(
            grid_path=grid,
            representation_receipt_path=representation,
            evaluations_root=evaluations,
            output_path=output,
        )
    assert not output.exists()


def test_freeze_uses_lower_online_cost_only_after_primary_utility_tie(
    tmp_path: Path,
) -> None:
    grid, representation, evaluations = _write_grid_artifacts(tmp_path, tie_utility_for_cost=True)

    receipt = freeze_rl_value_experiment(
        grid_path=grid,
        representation_receipt_path=representation,
        evaluations_root=evaluations,
        output_path=tmp_path / "rl-value-freeze.json",
    )

    assert receipt["selected_configuration"]["model_family"] == "if-core"
    assert receipt["selected_configuration"]["method"] == "supervised-only"


def test_freeze_rejects_unmatched_search_protocol(tmp_path: Path) -> None:
    mismatched = "rl-017-if-core-scratch-ppo-s3301"
    grid, representation, evaluations = _write_grid_artifacts(
        tmp_path, mismatch_budget_cell=mismatched
    )

    with pytest.raises(ValueError, match="matched.*protocol|protocol.*match"):
        freeze_rl_value_experiment(
            grid_path=grid,
            representation_receipt_path=representation,
            evaluations_root=evaluations,
            output_path=tmp_path / "rl-value-freeze.json",
        )


def test_freeze_rejects_a_checkpoint_from_another_representation_freeze(
    tmp_path: Path,
) -> None:
    mismatched = "rl-017-if-core-scratch-ppo-s3301"
    grid, representation, evaluations = _write_grid_artifacts(
        tmp_path, mismatch_selection_cell=mismatched
    )

    with pytest.raises(ValueError, match="representation-selection receipt"):
        freeze_rl_value_experiment(
            grid_path=grid,
            representation_receipt_path=representation,
            evaluations_root=evaluations,
            output_path=tmp_path / "rl-value-freeze.json",
        )


def test_freeze_rejects_test_reports_without_writing_output(tmp_path: Path) -> None:
    grid, representation, evaluations = _write_grid_artifacts(tmp_path, partition="test")
    output = tmp_path / "rl-value-freeze.json"

    with pytest.raises(ValueError, match="validation"):
        freeze_rl_value_experiment(
            grid_path=grid,
            representation_receipt_path=representation,
            evaluations_root=evaluations,
            output_path=output,
        )
    assert not output.exists()


def test_freeze_is_immutable_and_rejects_rehashed_family_forgery(tmp_path: Path) -> None:
    grid, representation, evaluations = _write_grid_artifacts(tmp_path)
    output = tmp_path / "rl-value-freeze.json"
    freeze_rl_value_experiment(
        grid_path=grid,
        representation_receipt_path=representation,
        evaluations_root=evaluations,
        output_path=output,
    )

    with pytest.raises(FileExistsError):
        freeze_rl_value_experiment(
            grid_path=grid,
            representation_receipt_path=representation,
            evaluations_root=evaluations,
            output_path=output,
        )

    forged = json.loads(representation.read_text())
    forged["selected_simpler"] = "if-mlp"
    forged["retained_families"] = ["if-mlp", "if-core"]
    unsigned = {key: value for key, value in forged.items() if key != "record_digest"}
    forged["record_digest"] = content_digest(unsigned)
    representation.write_bytes(canonical_json_bytes(forged) + b"\n")
    with pytest.raises(ValueError, match="registered representation selection"):
        freeze_rl_value_experiment(
            grid_path=grid,
            representation_receipt_path=representation,
            evaluations_root=evaluations,
            output_path=tmp_path / "forged-freeze.json",
        )
