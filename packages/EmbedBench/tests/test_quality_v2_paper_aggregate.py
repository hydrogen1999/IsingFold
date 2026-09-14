from __future__ import annotations

import hashlib
import inspect
import json
import statistics
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "aggregate_quality_v2_paper.py"
AUDIT_CONTRACT = ROOT / "configs" / "quality_v2_paper_audit_v2.json"
_AGGREGATOR_TEST_DRIVER = r"""
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from aggregate_quality_v2_paper import _atomic_json, _parse_args, aggregate
import select_training_grid

args = _parse_args(sys.argv[2:])
revalidation_calls = 0
real_revalidate = select_training_grid.revalidate_selection_artifact
def counted_revalidate(*call_args, **call_kwargs):
    global revalidation_calls
    revalidation_calls += 1
    return real_revalidate(*call_args, **call_kwargs)
select_training_grid.revalidate_selection_artifact = counted_revalidate
artifact = aggregate(args)
# The aggregate preflight and every one of the four freeze plus four score evaluator
# invocations replay the complete production selection validation.  There is no injectable
# prevalidated-selection capability.
assert revalidation_calls == 9, revalidation_calls
_atomic_json(Path(args.out), artifact)
print(json.dumps(artifact["primary_metric"], sort_keys=True, allow_nan=False))
"""
sys.path.insert(0, str(ROOT / "scripts"))

from aggregate_quality_v2_paper import (  # noqa: E402
    _inference_summary,
    _secondary_endpoint_inference_summary,
    _secondary_endpoint_problem_values,
)
from evaluate_quality_v2 import evaluate  # noqa: E402
from quality_v2_paper_contract import (  # noqa: E402
    AUDIT_SOURCE_HASH_ALGORITHM,
    CORE_RUNTIME_DISTRIBUTIONS,
    EXPECTED_PAPER_AUDIT_CONTRACT,
    audit_source_sha256,
    load_paper_audit_contract,
)

BUDGETS = ("1.00x", "1.10x", "1.25x", "1.50x", "uncapped")
SELECTORS = ("mean", "lcb", "resource", "original", "random")
AUDIT_PROVENANCE = {
    "generator": "EmbedBench/scripts/rescore_quality.py",
    "reads_per_strength": 4000,
    "num_sweeps": 200,
    "base_seed": 0,
    "seed_schedule": "(base_seed + 7919*candidate_index + 31*strength_index) % 2**31",
    "registered_strength_count": 4,
    "strength_schedule": "default_strength_grid(problem, 4)",
    "aggregation": "max_p_solve_over_strength_schedule",
    "score_evidence": "integer_success_counts_per_candidate_strength",
    "probability_derivation": "success_count_divided_by_reads_per_strength_binary64",
}


def test_paper_evaluator_has_no_prevalidated_selection_bypass() -> None:
    assert tuple(inspect.signature(evaluate).parameters) == ("args",)
    source = (ROOT / "scripts" / "evaluate_quality_v2.py").read_text(encoding="utf-8")
    assert "_prevalidated_selection" not in source


def test_primary_mean_test_is_valid_for_asymmetric_bounded_null() -> None:
    # Under X=1/9 with probability .9 and X=-1 with probability .1, E[X]=0 and the
    # all-positive sample below occurs with probability .9**20 > .12. A valid mean-null
    # test therefore cannot assign it the old sign-flip p-value of about 1e-4.
    values = {f"{index:064x}": 1.0 / 9.0 for index in range(20)}

    result = _inference_summary(
        values,
        contract=EXPECTED_PAPER_AUDIT_CONTRACT["secondary_inference"],
        namespace="asymmetric-bounded-null",
    )

    assert result["status"] == "available"
    assert result["one_sided_p_value"] >= 0.9**20


def test_primary_mean_inference_requires_registered_problem_cluster_minimum() -> None:
    result = _inference_summary(
        {"a" * 64: 0.5},
        contract=EXPECTED_PAPER_AUDIT_CONTRACT["secondary_inference"],
        namespace="too-small",
    )

    assert result["status"] == "insufficient_problem_clusters"
    assert result["estimate"] == 0.5
    assert result["confidence_interval_95"] is None
    assert result["one_sided_p_value"] is None


def _paired_strength_seed_document(seed: int, *, second_record_paired: bool) -> dict:
    values = [0.1 * (seed + 1), -0.05]
    paired = [True, second_record_paired]
    rows = [
        {
            "record_index": index,
            "problem_digest": f"{index + 10:x}" * 64,
            "selection_index": 1,
            "reference_index": 0,
            "authenticated": pair,
            "paired": pair,
            "worst_case_p_solve_delta_vs_stock_original": value if pair else None,
        }
        for index, (value, pair) in enumerate(zip(values, paired, strict=True))
    ]
    paired_values = [value for value, pair in zip(values, paired, strict=True) if pair]
    summary = {
        "support_records": 2,
        "paired_records": len(paired_values),
        "pairing_excluded_records": 2 - len(paired_values),
        "paired_coverage_rate": len(paired_values) / 2,
        "mean_worst_case_p_solve_delta_vs_stock_original": statistics.fmean(paired_values),
        "per_record": rows,
    }
    return {
        "secondary_endpoints": {
            "strength_robustness": {
                "full_support": {"selectors": {"mean": summary}},
            }
        }
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_secondary_endpoint_inference_clusters_complete_pairs_and_reports_partial() -> None:
    seeds = (0, 1, 2, 3)
    by_seed = {
        seed: (
            Path(f"seed-{seed}.json"),
            f"{seed + 1:x}" * 64,
            _paired_strength_seed_document(seed, second_record_paired=seed != 3),
            {},
        )
        for seed in seeds
    }
    values, coverage = _secondary_endpoint_problem_values(
        by_seed,
        seeds,
        endpoint="strength_robustness",
        support="full_support",
        selector="mean",
        metric="worst_case_p_solve_delta_vs_stock_original",
    )
    contract = load_paper_audit_contract(AUDIT_CONTRACT, _sha256(AUDIT_CONTRACT))

    first = _secondary_endpoint_inference_summary(
        values,
        coverage=coverage,
        contract=contract.secondary_endpoint_inference,
        namespace="test:partial-strength",
    )
    second = _secondary_endpoint_inference_summary(
        values,
        coverage=coverage,
        contract=contract.secondary_endpoint_inference,
        namespace="test:partial-strength",
    )

    assert values == {"a" * 64: pytest.approx(0.25)}
    assert coverage == {
        "support_records": 2,
        "paired_records": 1,
        "excluded_records": 1,
        "paired_coverage_rate": 0.5,
        "support_problem_clusters": 2,
        "complete_case_problem_clusters": 1,
        "excluded_problem_clusters": 1,
        "problem_cluster_coverage_rate": 0.5,
    }
    assert first == second
    assert first["status"] == "insufficient_problem_clusters"
    assert first["coverage_status"] == "partial"
    assert first["estimate"] == pytest.approx(0.25)
    assert first["confidence_interval"] is None


def test_secondary_endpoint_inference_is_unavailable_without_complete_pairs() -> None:
    contract = load_paper_audit_contract(AUDIT_CONTRACT, _sha256(AUDIT_CONTRACT))

    summary = _secondary_endpoint_inference_summary(
        {},
        coverage={
            "support_records": 2,
            "paired_records": 0,
            "excluded_records": 2,
            "paired_coverage_rate": 0.0,
            "support_problem_clusters": 2,
            "complete_case_problem_clusters": 0,
            "excluded_problem_clusters": 2,
            "problem_cluster_coverage_rate": 0.0,
        },
        contract=contract.secondary_endpoint_inference,
        namespace="test:no-pairs",
    )

    assert summary["status"] == "unavailable"
    assert summary["estimate"] is None
    assert summary["confidence_interval"] is None
    assert summary["problem_values"] == {}
    assert summary["unavailable_reason"] == ("no_complete_pair_across_all_registered_model_seeds")


def test_secondary_endpoint_excludes_an_entire_incomplete_problem_cluster() -> None:
    seeds = (0, 1, 2, 3)
    documents = {
        seed: _paired_strength_seed_document(seed, second_record_paired=seed != 3) for seed in seeds
    }
    for document in documents.values():
        rows = document["secondary_endpoints"]["strength_robustness"]["full_support"]["selectors"][
            "mean"
        ]["per_record"]
        rows[1]["problem_digest"] = rows[0]["problem_digest"]
    by_seed = {
        seed: (Path(f"seed-{seed}.json"), f"{seed + 1:x}" * 64, document, {})
        for seed, document in documents.items()
    }

    values, coverage = _secondary_endpoint_problem_values(
        by_seed,
        seeds,
        endpoint="strength_robustness",
        support="full_support",
        selector="mean",
        metric="worst_case_p_solve_delta_vs_stock_original",
    )

    assert values == {}
    assert coverage["support_problem_clusters"] == 1
    assert coverage["complete_case_problem_clusters"] == 0
    assert coverage["problem_cluster_coverage_rate"] == 0.0


def test_secondary_endpoint_inference_rejects_scored_unpaired_rows() -> None:
    seeds = (0, 1, 2, 3)
    by_seed = {
        seed: (
            Path(f"seed-{seed}.json"),
            f"{seed + 1:x}" * 64,
            _paired_strength_seed_document(seed, second_record_paired=True),
            {},
        )
        for seed in seeds
    }
    forged = by_seed[0][2]["secondary_endpoints"]["strength_robustness"]["full_support"][
        "selectors"
    ]["mean"]
    forged["per_record"][1]["paired"] = False

    with pytest.raises(ValueError, match="scores an unpaired"):
        _secondary_endpoint_problem_values(
            by_seed,
            seeds,
            endpoint="strength_robustness",
            support="full_support",
            selector="mean",
            metric="worst_case_p_solve_delta_vs_stock_original",
        )


def _write_selection(tmp_path: Path) -> tuple[Path, str, dict]:
    seeds = [0, 1, 2, 3]
    checkpoints = [
        {
            "seed": seed,
            "cell_id": f"quality_value_v2_screen__arch-mpnn__objective_variant-full__seed-{seed}",
            "checkpoint": f"runs/checkpoints/quality-v2-seed-{seed}.pt",
            "checkpoint_sha256": f"{seed + 1:x}" * 64,
        }
        for seed in seeds
    ]
    provenance = {
        "split_manifest": {
            "file": "splits.json",
            "sha256": "a" * 64,
            "schema": "embedbench.split-manifest",
            "schema_version": 2,
        },
        "corpus_inputs": [{"file": "quality.jsonl", "sha256": "b" * 64}],
    }
    document = {
        "schema": "embedbench.training-grid-selection",
        "schema_version": 3,
        "paper_audit_contract": load_paper_audit_contract(
            AUDIT_CONTRACT, _sha256(AUDIT_CONTRACT)
        ).public_binding(),
        "audit_source_hash_algorithm": AUDIT_SOURCE_HASH_ALGORITHM,
        "audit_source_sha256": audit_source_sha256(ROOT),
        "grid_id": "isingfold-quality-value-v2-screen",
        "grid_file": "training_grid_quality_v2.json",
        "grid_path": "configs/training_grid_quality_v2.json",
        "grid_sha256": "c" * 64,
        "source_sha256": "d" * 64,
        "selector_sha256": _sha256(ROOT / "scripts" / "select_training_grid.py"),
        "data_provenance": provenance,
        "stage": "quality_value_v2_screen",
        "registered_cell_count": 80,
        "primary_metric": "mean_finite_budget_regret",
        "direction": "minimize",
        "registered_seeds": seeds,
        "configuration_rule": "lowest_canonical_cpu_replay_mean_over_all_registered_seeds",
        "validation_metric_source": "canonical_cpu_checkpoint_replay",
        "validation_replay_policy": load_paper_audit_contract(
            AUDIT_CONTRACT, _sha256(AUDIT_CONTRACT)
        ).training_registration["validation_replay"]["canonical_runtime"],
        "paper_evaluation_rule": (
            "evaluate_every_registered_seed_checkpoint_of_winning_configuration"
        ),
        "test_records_parsed": False,
        "test_labels_used_for_selection": False,
        "test_evaluated": False,
        "winner": {
            "params": {"arch": "mpnn", "objective_variant": "full"},
            "selected_seed": 0,
            "selected_cell_id": checkpoints[0]["cell_id"],
            "checkpoint": checkpoints[0]["checkpoint"],
            "checkpoint_sha256": checkpoints[0]["checkpoint_sha256"],
            "paper_checkpoints": checkpoints,
        },
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path, _sha256(path), document


def _selector_metrics(value: float) -> dict:
    result = {}
    for selector in SELECTORS:
        regret = value if selector in {"mean", "lcb"} else 0.5
        probability = 1.0 - regret
        result[selector] = {
            "regret": regret,
            "mean_p_solve": probability,
            "top": 0.0,
            "mean_total_qubits": 10.0,
            "n_selected": 5,
            "selection_indices": [0, 0, 0, 0, 0],
            "per_record": [
                {
                    "record_index": index,
                    "problem_digest": f"{index + 1:x}" * 64,
                    "selection_index": 0,
                    "p_solve": probability,
                    "best_eligible_p_solve": 1.0,
                    "regret": regret,
                    "top": False,
                    "total_qubits": 10,
                }
                for index in range(5)
            ],
        }
    return result


def _evaluation(document: dict, selection_sha: str, seed: int, value: float) -> dict:
    from embedbench.hamiltonian_context import hamiltonian_context_contract

    checkpoint = document["winner"]["paper_checkpoints"][seed]
    runtime = {
        "schema": "embedbench.runtime-environment",
        "schema_version": 1,
        "python": {
            "implementation": "CPython",
            "version": "3.12.0",
            "executable": "/usr/bin/python3",
        },
        "torch": {
            "version": "2.7.0",
            "cuda_available": False,
            "cuda_runtime_version": None,
            "cudnn_version": None,
        },
        "device": {"selected": "cpu", "type": "cpu", "index": None, "gpu": None},
        "packages": {name: None for name in CORE_RUNTIME_DISTRIBUTIONS},
    }
    full_selectors = _selector_metrics(value)
    primary_coverage = {
        "input_records": 5,
        "included_records": 5,
        "excluded_records": 0,
        "coverage_rate": 1.0,
        "included_input_indices": [0, 1, 2, 3, 4],
        "excluded_input_indices": [],
        "exclusion_reasons": {
            "non_minorminer_source": 0,
            "missing_original_reference": 0,
            "infeasible_original_reference": 0,
        },
    }
    selection_binding = {
        "file": "selection.json",
        "sha256": selection_sha,
        "artifact_schema": document["schema"],
        "artifact_schema_version": document["schema_version"],
        "grid_id": document["grid_id"],
        "grid_sha256": document["grid_sha256"],
        "source_sha256": document["source_sha256"],
        "selector_sha256": document["selector_sha256"],
        "stage": document["stage"],
        "registered_seeds": document["registered_seeds"],
        "winner_params": document["winner"]["params"],
        "paper_checkpoint": checkpoint,
    }
    return {
        "artifact_schema": "embedbench.quality-v2-paper-evaluation",
        "artifact_schema_version": 2,
        "evaluation_mode": "paper",
        "partition": "test",
        "provisional": True,
        "evaluation_design": "exploratory_legacy_iid",
        "legacy_test_exposure": ("membership_problem_sizes_and_e0_inspected_without_audit_p_solve"),
        "label_fidelity": "independent-audit",
        "release_labels_consumed": False,
        "record_support": "complete_fixed_test_partition",
        "candidate_support": "full",
        "support_uses_labels": False,
        "audit_contract": load_paper_audit_contract(
            AUDIT_CONTRACT,
            _sha256(AUDIT_CONTRACT),
        ).public_binding(),
        "audit_source": {
            "algorithm": AUDIT_SOURCE_HASH_ALGORITHM,
            "sha256": audit_source_sha256(ROOT),
        },
        "audit_runtime_provenance": runtime,
        "runtime_provenance": runtime,
        "selection": selection_binding,
        "checkpoint": {
            "file": Path(checkpoint["checkpoint"]).name,
            "sha256": checkpoint["checkpoint_sha256"],
            "artifact_schema": "embedbench.quality-value-v2",
            "artifact_schema_version": 2,
            "architecture": "mpnn",
            "objective_variant": "full",
            "seed": seed,
        },
        "data_provenance": document["data_provenance"],
        "audit_release_commitment": {
            "schema": "embedbench.quality-v2-audit-release",
            "schema_version": 1,
            "file": "audit-release-manifest.json",
            "sha256": "9" * 64,
            "trust_model": "sha256_supplied_out_of_band",
            "shard_count": 64,
        },
        "audit_artifacts": [
            {
                "file": f"audit-{index:02d}.jsonl",
                "sha256": "e" * 64,
                "manifest_file": f"audit-{index:02d}.jsonl.manifest.json",
                "manifest_sha256": "f" * 64,
            }
            for index in range(64)
        ],
        "preprocessing": {
            "deploy_view": False,
            "deploy_max_free": 8,
            "neighbour_feats": False,
            "encoder": "chain",
            "hamiltonian_context": hamiltonian_context_contract(),
        },
        "deployment_host_contract": {"mode": "not_applied", "source_manifests": []},
        "exact_host_contract": {"mode": "verified_pristine_manifests", "sources": []},
        "partition_record_counts": {"train": 10, "val": 5, "test": 5},
        "n_test": 5,
        "n_train_encoded": 0,
        "n_val_encoded": 0,
        "audit_coverage": {"record_fraction_complete": 1.0},
        "evaluated_records": [
            {
                "record_index": index,
                "problem_digest": f"{index + 1:x}" * 64,
                "record_key": f"{index + 6:x}" * 64,
                "audit_provenance": AUDIT_PROVENANCE,
            }
            for index in range(5)
        ],
        "top_tolerance": 0.02,
        "random_seed": 0,
        "random_baseline_policy": "sha256_record_key_modulo_eligible_v1",
        "lcb_z": 1.0,
        "selection_statistic": "mean",
        "label_free_policy_freeze": {
            "schema": "embedbench.quality-v2-label-free-policy-freeze",
            "schema_version": 1,
            "sha256": f"{seed + 1:x}" * 64,
            "record_count": 5,
            "audit_inputs_opened_after_freeze": True,
        },
        "full_support_metrics": {
            "exact_mask": "minor_embedding_feasible",
            "n_records": 5,
            "total_presented_candidates": 10,
            "total_eligible_candidates": 10,
            "support_uses_labels": False,
            "eligible_indices": [[0, 1] for _ in range(5)],
            "selectors": full_selectors,
        },
        "budget_sweep": {
            "candidate_support": "full",
            "support_uses_labels": False,
            "exact_mask": "minor_embedding_feasible_and_full_embedding_total_qubits",
            "budget_contract": "B/Q_MM",
            "budget_reference": "stock_minorminer_original_total_qubits",
            "primary_record_policy": "valid_stock_minorminer_original_reference_only",
            "primary_record_coverage": primary_coverage,
            "budget_ratios": [1.0, 1.1, 1.25, 1.5, None],
            "selectors": ["mean", "lcb"],
            "primary_selector": "mean",
            "primary_metric_name": "mean_finite_budget_regret",
            "primary_metric": value,
            "lcb_z": 1.0,
            "lcb_is_secondary_until_calibrated": True,
            "budgets": {
                budget: {
                    "ratio": None if budget == "uncapped" else float(budget.removesuffix("x")),
                    "reference_policy": "stock_minorminer_original_total_qubits",
                    "n_records": 5,
                    "input_record_indices": [0, 1, 2, 3, 4],
                    "reference_total_qubits": [10] * 5,
                    "budgets": [
                        None if budget == "uncapped" else int(10 * float(budget.removesuffix("x")))
                    ]
                    * 5,
                    "eligible_indices": [[0, 1] for _ in range(5)],
                    "no_survivor": 0,
                    "no_survivor_rate": 0.0,
                    "selectors": _selector_metrics(value),
                }
                for budget in BUDGETS
            },
        },
    }


def _write_evaluations(
    tmp_path: Path,
    document: dict,
    selection_sha: str,
    seeds: list[int],
) -> list[Path]:
    paths = []
    for seed in seeds:
        path = tmp_path / f"paper-seed-{seed}.json"
        path.write_text(
            json.dumps(_evaluation(document, selection_sha, seed, 0.1 * (seed + 1))),
            encoding="utf-8",
        )
        paths.append(path)
    return paths


def _command(selection: Path, selection_sha: str, evaluations: list[Path], out: Path) -> list[str]:
    # These paths deliberately do not exist.  A handwritten selection/summary must be
    # rejected while revalidating the 80-cell registration, before any data or audit
    # input can be opened.
    dummy_files = selection.parent / "must-not-open.jsonl"
    dummy_splits = selection.parent / "must-not-open-splits.json"
    dummy_audit = selection.parent / "must-not-open-audit.jsonl"
    dummy_audit_release = selection.parent / "must-not-open-audit-release.json"
    dummy_preregistration = selection.parent / "must-not-open-preregistration.json"
    freezes = [selection.parent / f"must-not-open-freeze-{seed}.json" for seed in range(4)]
    return [
        sys.executable,
        str(SCRIPT),
        "--selection",
        str(selection),
        "--selection-sha256",
        selection_sha,
        "--selection-root",
        str(selection.parent),
        "--files",
        str(dummy_files),
        "--splits",
        str(dummy_splits),
        "--audit-labels",
        str(dummy_audit),
        "--audit-release-manifest",
        str(dummy_audit_release),
        "--audit-release-manifest-sha256",
        "9" * 64,
        "--preregistration-manifest",
        str(dummy_preregistration),
        "--preregistration-manifest-sha256",
        "8" * 64,
        "--policy-freezes",
        *map(str, freezes),
        "--audit-contract",
        str(AUDIT_CONTRACT),
        "--audit-contract-sha256",
        _sha256(AUDIT_CONTRACT),
        "--evaluations",
        *map(str, evaluations),
        "--out",
        str(out),
    ]


def test_aggregate_replays_every_registered_winner_seed_before_reporting(
    tmp_path: Path,
) -> None:
    from tests.test_quality_v2_paper_evaluation import (
        _command as evaluation_command,
    )
    from tests.test_quality_v2_paper_evaluation import (
        _freeze_command,
        _refresh_preregistration,
        _rewrite_audit,
        _write_fixture,
    )

    fixture = _write_fixture(tmp_path)
    _rewrite_audit(
        fixture,
        lambda row: row.update(
            strength_p_solve=[
                [0.05, 0.10, 0.15, 0.20],
                [0.50, 0.60, 0.70, 0.80],
                [0.20, 0.40, 0.60, 0.70],
            ],
            strength_success_counts=[
                [200, 400, 600, 800],
                [2000, 2400, 2800, 3200],
                [800, 1600, 2400, 2800],
            ],
        ),
    )
    selection = Path(fixture["selection"])
    selection_sha = str(fixture["selection_sha"])
    selection_document = json.loads(selection.read_text(encoding="utf-8"))
    evaluations: list[Path] = []
    freezes = list(fixture["policy_freezes"])
    per_seed_fixtures: list[dict[str, object]] = []
    for index, checkpoint in enumerate(selection_document["winner"]["paper_checkpoints"]):
        seed = checkpoint["seed"]
        per_seed = dict(fixture)
        per_seed["checkpoint"] = selection.parent / checkpoint["checkpoint"]
        per_seed["checkpoint_sha"] = checkpoint["checkpoint_sha256"]
        per_seed["policy_freeze"] = freezes[index]
        per_seed["output"] = tmp_path / f"evaluation-seed-{seed}.json"
        frozen = subprocess.run(
            _freeze_command(per_seed),
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert frozen.returncode == 0, frozen.stderr
        per_seed_fixtures.append(per_seed)

    _refresh_preregistration(fixture)
    for per_seed in per_seed_fixtures:
        per_seed["preregistration_sha"] = fixture["preregistration_sha"]
        per_seed["preregistration_binding"] = fixture["preregistration_binding"]
        per_seed["audit_release_manifest_sha"] = fixture["audit_release_manifest_sha"]
        scored = subprocess.run(
            evaluation_command(per_seed),
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        assert scored.returncode == 0, scored.stderr
        evaluations.append(Path(per_seed["output"]))
    out = tmp_path / "aggregate.json"
    command = [
        sys.executable,
        "-c",
        _AGGREGATOR_TEST_DRIVER,
        str(ROOT / "scripts"),
        "--selection",
        str(selection),
        "--selection-sha256",
        selection_sha,
        "--selection-root",
        str(selection.parent),
        "--files",
        str(fixture["corpus"]),
        "--splits",
        str(fixture["splits"]),
        "--audit-labels",
        *(str(path) for path in fixture["audits"]),
        "--audit-release-manifest",
        str(fixture["audit_release_manifest"]),
        "--audit-release-manifest-sha256",
        str(fixture["audit_release_manifest_sha"]),
        "--preregistration-manifest",
        str(fixture["preregistration"]),
        "--preregistration-manifest-sha256",
        str(fixture["preregistration_sha"]),
        "--policy-freezes",
        *map(str, freezes),
        "--audit-contract",
        str(fixture["audit_contract"]),
        "--audit-contract-sha256",
        str(fixture["audit_contract_sha"]),
        "--evaluations",
        *map(str, evaluations),
        "--out",
        str(out),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["artifact_schema"] == "embedbench.quality-v2-paper-aggregate"
    assert result["artifact_schema_version"] == 2
    assert result["registered_seeds"] == [0, 1, 2, 3]
    assert result["audit_contract"]["sha256"] == fixture["audit_contract_sha"]
    assert [row["seed"] for row in result["evaluations"]] == [0, 1, 2, 3]
    assert result["primary_metric"]["name"] == "mean_finite_budget_regret"
    assert result["primary_metric"]["path"] == "budget_sweep.primary_metric"
    assert result["ground_reference_coverage"]["records"] == 1
    assert result["independent_revalidation"] == {
        **result["independent_revalidation"],
        "training_cells_replayed": 80,
        "winner_checkpoints_phase_1_replayed": 4,
        "winner_checkpoints_phase_2_replayed": 4,
        "audit_shards_validated_per_checkpoint": 64,
    }
    assert set(result["primary_metric"]["values_by_seed"]) == {"0", "1", "2", "3"}
    assert "full_support_metrics.selectors.mean.mean_p_solve" in result["metric_aggregates"]
    secondary = result["secondary_endpoints"]
    assert secondary["selection_role"] == "report_only_after_label_free_policy_freeze"
    assert all(
        row["strength_robustness"]["status"] == "available"
        for row in secondary["availability_by_seed"].values()
    )
    assert any(
        path.endswith("mean_worst_case_p_solve") for path in secondary["registered_metric_paths"]
    )
    assert (
        "secondary_endpoints.strength_robustness.full_support.selectors.original."
        "mean_worst_case_p_solve"
    ) in secondary["registered_metric_paths"]
    inference = result["secondary_problem_clustered_inference"]
    assert inference["contract"]["mean_inference"]["minimum_problem_clusters"] == 20
    assert set(inference["by_budget"]) == set(BUDGETS[:-1])
    assert all("holm_adjusted_p_value" in summary for summary in inference["by_budget"].values())
    endpoint_inference = result["secondary_endpoint_clustered_inference"]
    assert endpoint_inference["selection_role"] == (
        "report_only_never_selection_or_primary_pass_fail"
    )
    residual_inference = endpoint_inference["residual_connectivity"]["budgets"]["1.00x"]
    assert residual_inference["status"] == "insufficient_problem_clusters"
    residual_free = residual_inference["remaining_free_qubits_delta_vs_stock_original"]
    assert residual_free["n_problem_clusters"] == 1
    assert residual_free["confidence_interval"] is None
    assert len(residual_free["problem_values"]) == 1
    strength_inference = endpoint_inference["strength_robustness"]["full_support"]
    assert strength_inference["status"] == "insufficient_problem_clusters"
    strength_worst = strength_inference["worst_case_p_solve_delta_vs_stock_original"]
    assert strength_worst["n_problem_clusters"] == 1
    assert strength_worst["confidence_interval"] is None

    forged = json.loads(evaluations[0].read_text(encoding="utf-8"))
    forged["handwritten_summary"] = {"claim": "accept without replay"}
    evaluations[0].write_text(json.dumps(forged), encoding="utf-8")
    forged_out = tmp_path / "forged-aggregate.json"
    rejected = subprocess.run(
        [*command[:-1], str(forged_out)],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert rejected.returncode != 0
    assert "handwritten_summary" in rejected.stderr
    assert not forged_out.exists()


def test_aggregate_rejects_handwritten_summaries_before_opening_data(
    tmp_path: Path,
) -> None:
    selection, selection_sha, document = _write_selection(tmp_path)
    evaluations = _write_evaluations(tmp_path, document, selection_sha, [0, 1, 2, 3])

    completed = subprocess.run(
        _command(selection, selection_sha, evaluations, tmp_path / "aggregate.json"),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "training_grid_quality_v2.json" in completed.stderr
    assert "must-not-open" not in completed.stderr


def test_aggregate_authenticates_preregistration_before_any_audit_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import aggregate_quality_v2_paper as aggregator

    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}", encoding="utf-8")
    seeds = (0, 1, 2, 3)
    checkpoints = tuple(
        aggregator.paper_evaluator.RegisteredPaperCheckpoint(
            seed=seed,
            cell_id=f"cell-{seed}",
            checkpoint=f"checkpoint-{seed}.pt",
            checkpoint_sha256=f"{seed + 1:x}" * 64,
            resolved_path=(tmp_path / f"checkpoint-{seed}.pt").resolve(),
        )
        for seed in seeds
    )
    selection_document = {"registered_cell_count": 80}
    selection = aggregator.paper_evaluator.SelectionArtifact(
        path=selection_path.resolve(),
        sha256="a" * 64,
        root=tmp_path.resolve(),
        document=selection_document,
        registered_seeds=seeds,
        winner_params={"arch": "mpnn", "objective_variant": "full"},
        paper_checkpoints=checkpoints,
    )
    import select_training_grid

    monkeypatch.setattr(
        select_training_grid,
        "revalidate_selection_artifact",
        lambda *args, **kwargs: selection_document,
    )
    monkeypatch.setattr(
        aggregator.paper_evaluator,
        "load_selection_artifact",
        lambda *args, **kwargs: selection,
    )
    monkeypatch.setattr(
        aggregator,
        "load_paper_preregistration",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("paper preregistration does not match the independently supplied SHA-256")
        ),
    )
    monkeypatch.setattr(
        aggregator.paper_evaluator,
        "evaluate",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("audit replay started before preregistration authentication")
        ),
    )
    contract = ROOT / "configs" / "quality_v2_paper_audit_v2.json"
    args = SimpleNamespace(
        audit_contract=str(contract),
        audit_contract_sha256=_sha256(contract),
        selection=str(selection_path),
        selection_sha256="a" * 64,
        selection_root=str(tmp_path),
        policy_freezes=[str(tmp_path / f"freeze-{seed}.json") for seed in seeds],
        preregistration_manifest=str(tmp_path / "self-rehashed-preregistration.json"),
        preregistration_manifest_sha256="b" * 64,
        evaluations=[str(tmp_path / f"evaluation-{seed}.json") for seed in seeds],
        files=[str(tmp_path / "must-not-open-corpus.jsonl")],
        splits=str(tmp_path / "must-not-open-splits.json"),
        audit_labels=[str(tmp_path / "must-not-open-audit.jsonl")],
        audit_release_manifest=str(tmp_path / "must-not-open-release.json"),
        audit_release_manifest_sha256="c" * 64,
        out=str(tmp_path / "aggregate.json"),
    )

    with pytest.raises(ValueError, match="independently supplied SHA-256"):
        aggregator.aggregate(args)
