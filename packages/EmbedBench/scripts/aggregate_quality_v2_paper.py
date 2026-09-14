#!/usr/bin/env python3
"""Aggregate exactly all registered Quality V2 winner-seed paper evaluations.

Before trusting any supplied evaluation JSON, this entry point revalidates the complete
80-cell training grid, independently replays phase 1 for all four winner checkpoints, verifies
their registered policy-freeze bytes, validates all 64 externally committed audit shard byte
streams for every seed, and recomputes the evaluation metrics. Only byte-equivalent replayed
evaluation documents enter the aggregate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import evaluate_quality_v2 as paper_evaluator
import numpy as np
from quality_v2_paper_contract import (
    PaperAuditContract,
    PaperPreregistration,
    load_paper_audit_contract,
    load_paper_preregistration,
    preregistered_policy_freeze_binding,
    require_exact_json,
    strict_json_loads,
    validate_runtime_provenance,
    verify_preregistered_policy_freezes,
)

AGGREGATE_SCHEMA = "embedbench.quality-v2-paper-aggregate"
AGGREGATE_SCHEMA_VERSION = 2
SELECTORS = ("mean", "lcb", "resource", "original", "random")
SELECTOR_METRICS = ("regret", "mean_p_solve", "top", "mean_total_qubits")
BUDGET_KEYS = ("1.00x", "1.10x", "1.25x", "1.50x", "uncapped")
FINITE_BUDGET_KEYS = BUDGET_KEYS[:-1]
MODEL_SCHEMA = "embedbench.quality-value-v2"
MODEL_SCHEMA_VERSION = 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--selection-sha256", required=True)
    parser.add_argument("--selection-root", default=".")
    parser.add_argument("--files", nargs="+", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--audit-labels", nargs="+", required=True)
    parser.add_argument("--audit-release-manifest", required=True)
    parser.add_argument("--audit-release-manifest-sha256", required=True)
    parser.add_argument("--preregistration-manifest", required=True)
    parser.add_argument("--preregistration-manifest-sha256", required=True)
    parser.add_argument(
        "--policy-freezes",
        nargs="+",
        required=True,
        help="one phase-1 policy freeze in registered seed order",
    )
    parser.add_argument("--audit-contract", required=True)
    parser.add_argument("--audit-contract-sha256", required=True)
    parser.add_argument(
        "--evaluations",
        nargs="+",
        required=True,
        help="one paper-evaluation JSON artifact for every registered winner seed",
    )
    parser.add_argument("--out", required=True)
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(argv)
    if not paper_evaluator._valid_sha256(args.selection_sha256):
        parser.error("--selection-sha256 must be lowercase hexadecimal SHA-256")
    if not paper_evaluator._valid_sha256(args.audit_contract_sha256):
        parser.error("--audit-contract-sha256 must be lowercase hexadecimal SHA-256")
    if not paper_evaluator._valid_sha256(args.audit_release_manifest_sha256):
        parser.error("--audit-release-manifest-sha256 must be lowercase hexadecimal SHA-256")
    if not paper_evaluator._valid_sha256(args.preregistration_manifest_sha256):
        parser.error(
            "--preregistration-manifest-sha256 must be an independently supplied "
            "lowercase hexadecimal SHA-256"
        )
    return args


def _read_object(path: str | Path, kind: str) -> dict[str, Any]:
    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise ValueError(f"invalid {kind} {source}") from error
    return _object_from_payload(payload, source, kind)


def _object_from_payload(payload: bytes, source: Path, kind: str) -> dict[str, Any]:
    document = strict_json_loads(payload, location=f"{kind} {source}")
    if not isinstance(document, dict):
        raise ValueError(f"{kind} must be a JSON object: {source}")
    return document


def _finite_number(value: object, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{location} must be a finite numeric paper metric")
    return float(value)


def _validate_selector_summary(
    summary: Mapping[str, Any],
    *,
    source: Path,
    path: str,
    expected_record_indices: Sequence[int],
    problem_digests: Mapping[int, str],
    top_tolerance: float,
) -> None:
    selections = summary.get("selection_indices")
    per_record = summary.get("per_record")
    if (
        not isinstance(selections, list)
        or not isinstance(per_record, list)
        or len(selections) != len(expected_record_indices)
        or len(per_record) != len(expected_record_indices)
    ):
        raise ValueError(f"paper evaluation {source} has incomplete per-record {path}")
    regrets: list[float] = []
    probabilities: list[float] = []
    top_hits: list[bool] = []
    qubits: list[int] = []
    for position, (expected_index, selected, outcome) in enumerate(
        zip(expected_record_indices, selections, per_record, strict=True)
    ):
        if not isinstance(outcome, Mapping):
            raise ValueError(f"paper evaluation {source} has invalid {path}[{position}]")
        if (
            outcome.get("record_index") != expected_index
            or outcome.get("problem_digest") != problem_digests[expected_index]
            or outcome.get("selection_index") != selected
        ):
            raise ValueError(f"paper evaluation {source} misaligns {path}[{position}]")
        metric_fields = (
            "p_solve",
            "best_eligible_p_solve",
            "regret",
            "top",
            "total_qubits",
        )
        if selected is None:
            if any(outcome.get(field) is not None for field in metric_fields):
                raise ValueError(f"paper evaluation {source} scores an unselected {path} row")
            continue
        if isinstance(selected, bool) or not isinstance(selected, int) or selected < 0:
            raise ValueError(f"paper evaluation {source} has an invalid {path} selection")
        probability = _finite_number(outcome.get("p_solve"), f"{source}: {path}.p_solve")
        best = _finite_number(
            outcome.get("best_eligible_p_solve"),
            f"{source}: {path}.best_eligible_p_solve",
        )
        regret = _finite_number(outcome.get("regret"), f"{source}: {path}.regret")
        top = outcome.get("top")
        total_qubits = outcome.get("total_qubits")
        if (
            not 0.0 <= probability <= 1.0
            or not 0.0 <= best <= 1.0
            or regret < 0.0
            or not math.isclose(regret, best - probability, rel_tol=1e-12, abs_tol=1e-12)
            or not isinstance(top, bool)
            or top is not (probability >= best - top_tolerance)
            or isinstance(total_qubits, bool)
            or not isinstance(total_qubits, int)
            or total_qubits <= 0
        ):
            raise ValueError(f"paper evaluation {source} has inconsistent {path}[{position}]")
        regrets.append(regret)
        probabilities.append(probability)
        top_hits.append(top)
        qubits.append(total_qubits)
    expected = {
        "regret": statistics.fmean(regrets) if regrets else None,
        "mean_p_solve": statistics.fmean(probabilities) if probabilities else None,
        "top": statistics.fmean(top_hits) if top_hits else None,
        "mean_total_qubits": statistics.fmean(qubits) if qubits else None,
        "n_selected": len(regrets),
    }
    for field, value in expected.items():
        actual = summary.get(field)
        matches = (
            actual is value
            if value is None
            else (
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and math.isclose(float(actual), float(value), rel_tol=1e-12, abs_tol=1e-12)
            )
        )
        if not matches:
            raise ValueError(f"paper evaluation {source} has inconsistent aggregate {path}.{field}")


def _registered_metrics(
    document: Mapping[str, Any],
    source: Path,
    audit_contract: PaperAuditContract,
    problem_digests: Mapping[int, str],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    budget_sweep = document.get("budget_sweep")
    if not isinstance(budget_sweep, Mapping):
        raise ValueError(f"paper evaluation {source} has no budget sweep")
    if budget_sweep.get("primary_metric_name") != paper_evaluator.PRIMARY_METRIC:
        raise ValueError(f"paper evaluation {source} changed the primary metric")
    expected_budget_protocol = {
        "candidate_support": audit_contract.evaluation["candidate_support"],
        "support_uses_labels": audit_contract.evaluation["support_uses_labels"],
        "exact_mask": audit_contract.evaluation["exact_mask"],
        "budget_contract": audit_contract.evaluation["budget_contract"],
        "budget_reference": audit_contract.evaluation["budget_reference"],
        "primary_record_policy": audit_contract.evaluation["primary_record_policy"],
        "budget_ratios": audit_contract.evaluation["registered_budget_ratios"],
        "selectors": audit_contract.evaluation["reported_selection_statistics"],
        "primary_selector": audit_contract.evaluation["selection_statistic"],
        "primary_metric_name": audit_contract.evaluation["primary_metric"],
        "lcb_z": audit_contract.evaluation["lcb_z"],
    }
    budget_mismatches = [
        field
        for field, expected in expected_budget_protocol.items()
        if budget_sweep.get(field) != expected
    ]
    if budget_mismatches:
        raise ValueError(
            f"paper evaluation {source} changed its frozen budget protocol: {budget_mismatches}"
        )
    metrics["budget_sweep.primary_metric"] = _finite_number(
        budget_sweep.get("primary_metric"),
        f"{source}: budget_sweep.primary_metric",
    )

    full_support = document.get("full_support_metrics")
    full_selectors = full_support.get("selectors") if isinstance(full_support, Mapping) else None
    if not isinstance(full_selectors, Mapping):
        raise ValueError(f"paper evaluation {source} has no full-support selectors")
    for selector in SELECTORS:
        summary = full_selectors.get(selector)
        if not isinstance(summary, Mapping):
            raise ValueError(f"paper evaluation {source} is missing selector {selector!r}")
        for metric in SELECTOR_METRICS:
            path = f"full_support_metrics.selectors.{selector}.{metric}"
            metrics[path] = _finite_number(summary.get(metric), f"{source}: {path}")
        _validate_selector_summary(
            summary,
            source=source,
            path=f"full_support_metrics.selectors.{selector}",
            expected_record_indices=list(range(len(problem_digests))),
            problem_digests=problem_digests,
            top_tolerance=float(audit_contract.evaluation["top_tolerance"]),
        )

    budgets = budget_sweep.get("budgets")
    if not isinstance(budgets, Mapping) or set(budgets) != set(BUDGET_KEYS):
        raise ValueError(f"paper evaluation {source} changed the registered budget sweep")
    for budget in BUDGET_KEYS:
        result = budgets[budget]
        selectors = result.get("selectors") if isinstance(result, Mapping) else None
        if not isinstance(selectors, Mapping):
            raise ValueError(f"paper evaluation {source} has no selectors for budget {budget}")
        for selector in SELECTORS:
            summary = selectors.get(selector)
            if not isinstance(summary, Mapping):
                raise ValueError(
                    f"paper evaluation {source} is missing {selector!r} at budget {budget}"
                )
            for metric in SELECTOR_METRICS:
                path = f"budget_sweep.budgets.{budget}.selectors.{selector}.{metric}"
                metrics[path] = _finite_number(summary.get(metric), f"{source}: {path}")
            input_indices = result.get("input_record_indices")
            assert isinstance(input_indices, list)
            _validate_selector_summary(
                summary,
                source=source,
                path=f"budget_sweep.budgets.{budget}.selectors.{selector}",
                expected_record_indices=input_indices,
                problem_digests=problem_digests,
                top_tolerance=float(audit_contract.evaluation["top_tolerance"]),
            )

    recomputed_primary = statistics.fmean(
        metrics[f"budget_sweep.budgets.{budget}.selectors.mean.regret"]
        for budget in FINITE_BUDGET_KEYS
    )
    if not math.isclose(
        metrics["budget_sweep.primary_metric"],
        recomputed_primary,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(f"paper evaluation {source} has an inconsistent primary metric")
    metrics.update(
        _secondary_registered_metrics(
            document,
            source,
            audit_contract,
        )
    )
    return metrics


def _validated_endpoint_coverage(
    value: object,
    *,
    source: Path,
    path: str,
) -> dict[str, float | int]:
    if not isinstance(value, Mapping) or set(value) != {
        "records",
        "authenticated_records",
        "coverage_rate",
    }:
        raise ValueError(f"paper evaluation {source} has invalid {path} coverage")
    records = value.get("records")
    authenticated = value.get("authenticated_records")
    rate = value.get("coverage_rate")
    if (
        isinstance(records, bool)
        or not isinstance(records, int)
        or records < 0
        or isinstance(authenticated, bool)
        or not isinstance(authenticated, int)
        or not 0 <= authenticated <= records
        or isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(float(rate))
        or not math.isclose(
            float(rate),
            authenticated / records if records else 0.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"paper evaluation {source} has inconsistent {path} coverage")
    return {
        "records": records,
        "authenticated_records": authenticated,
        "coverage_rate": float(rate),
    }


def _optional_registered_metric(
    metrics: dict[str, float],
    summary: Mapping[str, object],
    field: str,
    *,
    source: Path,
    path: str,
) -> None:
    value = summary.get(field)
    if value is not None:
        metrics[f"{path}.{field}"] = _finite_number(value, f"{source}: {path}.{field}")


def _secondary_registered_metrics(
    document: Mapping[str, Any],
    source: Path,
    audit_contract: PaperAuditContract,
) -> dict[str, float]:
    secondary = document.get("secondary_endpoints")
    if not isinstance(secondary, Mapping) or set(secondary) != {
        "selection_role",
        "availability_policy",
        "registration",
        "exact_feasibility_budget_survival",
        "terminal_resources",
        "residual_connectivity",
        "strength_robustness",
    }:
        raise ValueError(f"paper evaluation {source} has invalid secondary endpoints")
    require_exact_json(
        secondary.get("registration"),
        audit_contract.secondary_endpoints,
        location=f"paper evaluation {source} secondary endpoint registration",
    )
    for field in ("selection_role", "availability_policy"):
        if secondary.get(field) != audit_contract.secondary_endpoints[field]:
            raise ValueError(f"paper evaluation {source} changed secondary endpoint {field}")
    selectors = tuple(audit_contract.secondary_endpoints["selectors"])
    if selectors != SELECTORS:
        raise RuntimeError("aggregator selector registry changed")
    metrics: dict[str, float] = {}

    feasibility = secondary.get("exact_feasibility_budget_survival")
    if not isinstance(feasibility, Mapping) or feasibility.get("status") != "available":
        raise ValueError(f"paper evaluation {source} lacks exact feasibility endpoints")
    _validated_endpoint_coverage(
        feasibility.get("coverage"),
        source=source,
        path="secondary_endpoints.exact_feasibility_budget_survival",
    )
    full = feasibility.get("full_support")
    if not isinstance(full, Mapping):
        raise ValueError(f"paper evaluation {source} lacks full-support feasibility")
    metrics[
        "secondary_endpoints.exact_feasibility_budget_survival.full_support."
        "candidate_feasibility_rate"
    ] = _finite_number(
        full.get("candidate_feasibility_rate"),
        f"{source}: candidate feasibility rate",
    )
    full_selectors = full.get("selectors")
    if not isinstance(full_selectors, Mapping) or set(full_selectors) != set(selectors):
        raise ValueError(f"paper evaluation {source} changed feasibility selectors")
    for selector in selectors:
        summary = full_selectors[selector]
        if not isinstance(summary, Mapping):
            raise ValueError(f"paper evaluation {source} has invalid feasibility selector")
        metrics[
            "secondary_endpoints.exact_feasibility_budget_survival.full_support.selectors."
            f"{selector}.selection_survival_rate"
        ] = _finite_number(
            summary.get("selection_survival_rate"),
            f"{source}: full-support {selector} selection survival",
        )
    feasibility_budgets = feasibility.get("budgets")
    if not isinstance(feasibility_budgets, Mapping) or set(feasibility_budgets) != set(BUDGET_KEYS):
        raise ValueError(f"paper evaluation {source} changed feasibility budget endpoints")
    for budget in BUDGET_KEYS:
        budget_summary = feasibility_budgets[budget]
        if not isinstance(budget_summary, Mapping):
            raise ValueError(f"paper evaluation {source} has invalid feasibility budget")
        base = f"secondary_endpoints.exact_feasibility_budget_survival.budgets.{budget}"
        metrics[f"{base}.record_survival_rate"] = _finite_number(
            budget_summary.get("record_survival_rate"),
            f"{source}: {base}.record_survival_rate",
        )
        budget_selectors = budget_summary.get("selectors")
        if not isinstance(budget_selectors, Mapping) or set(budget_selectors) != set(selectors):
            raise ValueError(f"paper evaluation {source} changed budget feasibility selectors")
        for selector in selectors:
            summary = budget_selectors[selector]
            if not isinstance(summary, Mapping):
                raise ValueError(f"paper evaluation {source} has invalid budget survival")
            metrics[f"{base}.selectors.{selector}.selection_survival_rate"] = _finite_number(
                summary.get("selection_survival_rate"),
                f"{source}: {base}.{selector}.selection_survival_rate",
            )

    endpoint_specs = {
        "terminal_resources": (
            "paired_records",
            tuple(audit_contract.secondary_endpoints["terminal_resources"]["metrics"]),
        ),
        "residual_connectivity": (
            "authenticated_records",
            tuple(audit_contract.secondary_endpoints["residual_connectivity"]["metrics"]),
        ),
        "strength_robustness": (
            "authenticated_records",
            tuple(audit_contract.secondary_endpoints["strength_robustness"]["metrics"]),
        ),
    }
    for endpoint_name, (_available_field, metric_fields) in endpoint_specs.items():
        endpoint = secondary.get(endpoint_name)
        if not isinstance(endpoint, Mapping) or endpoint.get("status") not in {
            "available",
            "partial",
            "unavailable",
        }:
            raise ValueError(f"paper evaluation {source} has invalid {endpoint_name} endpoint")
        _validated_endpoint_coverage(
            endpoint.get("coverage"),
            source=source,
            path=f"secondary_endpoints.{endpoint_name}",
        )
        budgets = endpoint.get("budgets")
        if not isinstance(budgets, Mapping) or set(budgets) != set(BUDGET_KEYS):
            raise ValueError(f"paper evaluation {source} changed {endpoint_name} budgets")
        for budget in BUDGET_KEYS:
            budget_summary = budgets[budget]
            endpoint_selectors = (
                budget_summary.get("selectors") if isinstance(budget_summary, Mapping) else None
            )
            if not isinstance(endpoint_selectors, Mapping) or set(endpoint_selectors) != set(
                selectors
            ):
                raise ValueError(f"paper evaluation {source} changed {endpoint_name} selectors")
            for selector in selectors:
                summary = endpoint_selectors[selector]
                if not isinstance(summary, Mapping) or summary.get("status") not in {
                    "available",
                    "partial",
                    "unavailable",
                }:
                    raise ValueError(
                        f"paper evaluation {source} has invalid {endpoint_name} summary"
                    )
                base = f"secondary_endpoints.{endpoint_name}.budgets.{budget}.selectors.{selector}"
                for field in metric_fields:
                    _optional_registered_metric(
                        metrics,
                        summary,
                        field,
                        source=source,
                        path=base,
                    )

    strength = secondary["strength_robustness"]
    assert isinstance(strength, Mapping)
    primary_coverage = document.get("budget_sweep")
    primary_coverage = (
        primary_coverage.get("primary_record_coverage")
        if isinstance(primary_coverage, Mapping)
        else None
    )
    if not isinstance(primary_coverage, Mapping):
        raise ValueError(f"paper evaluation {source} has no primary budget-cohort coverage")
    expected_strength_support = {
        "full_support_records": document.get("n_test"),
        "full_support_excluded_records": 0,
        "budget_conditioned_records": primary_coverage.get("included_records"),
        "budget_excluded_records": primary_coverage.get("excluded_records"),
        "budget_coverage_rate": primary_coverage.get("coverage_rate"),
        "budget_exclusion_reasons": primary_coverage.get("exclusion_reasons"),
    }
    require_exact_json(
        strength.get("support"),
        expected_strength_support,
        location=f"paper evaluation {source} strength-robustness support",
    )
    full_strength = strength.get("full_support")
    if (
        not isinstance(full_strength, Mapping)
        or full_strength.get("record_support") != "complete_fixed_test_partition"
        or full_strength.get("records") != document.get("n_test")
    ):
        raise ValueError(f"paper evaluation {source} changed full-support strength evidence")
    full_strength_selectors = full_strength.get("selectors")
    if not isinstance(full_strength_selectors, Mapping) or set(full_strength_selectors) != set(
        selectors
    ):
        raise ValueError(f"paper evaluation {source} changed full-support strength selectors")
    strength_metrics = tuple(audit_contract.secondary_endpoints["strength_robustness"]["metrics"])
    for selector in selectors:
        summary = full_strength_selectors[selector]
        if not isinstance(summary, Mapping) or summary.get("status") not in {
            "available",
            "partial",
            "unavailable",
        }:
            raise ValueError(f"paper evaluation {source} has invalid full-support strength summary")
        support_records = summary.get("support_records")
        selected_records = summary.get("selected_records")
        selection_excluded = summary.get("selection_excluded_records")
        authenticated_records = summary.get("authenticated_records")
        missing_outcomes = summary.get("missing_outcome_records")
        if (
            isinstance(support_records, bool)
            or not isinstance(support_records, int)
            or support_records != document.get("n_test")
            or isinstance(selected_records, bool)
            or not isinstance(selected_records, int)
            or isinstance(selection_excluded, bool)
            or not isinstance(selection_excluded, int)
            or selected_records + selection_excluded != support_records
            or isinstance(authenticated_records, bool)
            or not isinstance(authenticated_records, int)
            or isinstance(missing_outcomes, bool)
            or not isinstance(missing_outcomes, int)
            or authenticated_records + missing_outcomes != selected_records
        ):
            raise ValueError(
                f"paper evaluation {source} has inconsistent full-support strength coverage"
            )
        base = f"secondary_endpoints.strength_robustness.full_support.selectors.{selector}"
        for field in strength_metrics:
            _optional_registered_metric(
                metrics,
                summary,
                field,
                source=source,
                path=base,
            )
    return metrics


def _validate_preprocessing(
    value: object,
    selection: paper_evaluator.SelectionArtifact,
    source: Path,
) -> Mapping[str, Any]:
    from embedbench.hamiltonian_context import hamiltonian_context_contract

    if not isinstance(value, Mapping):
        raise ValueError(f"paper evaluation {source} has no preprocessing contract")
    if value.get("hamiltonian_context") != hamiltonian_context_contract():
        raise ValueError(f"paper evaluation {source} changed Hamiltonian-context preprocessing")
    expected_encoder = "heterogeneous" if selection.winner_params["arch"] == "hetero" else "chain"
    if value.get("encoder") != expected_encoder:
        raise ValueError(f"paper evaluation {source} changed the selected encoder")
    return value


def _validate_evaluation(
    document: Mapping[str, Any],
    source: Path,
    selection: paper_evaluator.SelectionArtifact,
    audit_contract: PaperAuditContract,
    preregistration: PaperPreregistration,
) -> tuple[
    paper_evaluator.RegisteredPaperCheckpoint,
    dict[str, float],
]:
    expected_protocol = {
        "artifact_schema": paper_evaluator.PAPER_ARTIFACT_SCHEMA,
        "artifact_schema_version": paper_evaluator.PAPER_ARTIFACT_SCHEMA_VERSION,
        "evaluation_mode": "paper",
        "partition": "test",
        "provisional": audit_contract.evaluation["provisional"],
        "evaluation_design": audit_contract.evaluation["evaluation_design"],
        "legacy_test_exposure": audit_contract.evaluation["legacy_test_exposure"],
        "label_fidelity": "independent-audit",
        "release_labels_consumed": False,
        "record_support": audit_contract.evaluation["record_support"],
        "candidate_support": audit_contract.evaluation["candidate_support"],
        "support_uses_labels": audit_contract.evaluation["support_uses_labels"],
        "top_tolerance": audit_contract.evaluation["top_tolerance"],
        "random_seed": audit_contract.evaluation["random_seed"],
        "random_baseline_policy": audit_contract.evaluation["random_baseline_policy"],
        "lcb_z": audit_contract.evaluation["lcb_z"],
        "selection_statistic": audit_contract.evaluation["selection_statistic"],
        "n_train_encoded": 0,
        "n_val_encoded": 0,
    }
    mismatches = [
        field for field, expected in expected_protocol.items() if document.get(field) != expected
    ]
    if mismatches:
        raise ValueError(f"paper evaluation {source} violates its protocol: {mismatches}")
    require_exact_json(
        document.get("audit_contract"),
        audit_contract.public_binding(),
        location=f"paper evaluation {source} audit contract binding",
    )
    require_exact_json(
        document.get("preregistration"),
        preregistration.public_binding(),
        location=f"paper evaluation {source} preregistration binding",
    )
    expected_source = {
        "algorithm": selection.document["audit_source_hash_algorithm"],
        "sha256": selection.document["audit_source_sha256"],
    }
    require_exact_json(
        document.get("audit_source"),
        expected_source,
        location=f"paper evaluation {source} audit source provenance",
    )
    for field in ("runtime_provenance", "audit_runtime_provenance"):
        validate_runtime_provenance(
            document.get(field),
            location=f"paper evaluation {source} {field}",
            expected_device=str(audit_contract.evaluation["device"]),
        )

    checkpoint = document.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"paper evaluation {source} has no checkpoint contract")
    seed = checkpoint.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"paper evaluation {source} has an invalid checkpoint seed")
    registered = next(
        (entry for entry in selection.paper_checkpoints if entry.seed == seed),
        None,
    )
    if registered is None:
        raise ValueError(f"paper evaluation {source} is not in selection winner.paper_checkpoints")
    expected_checkpoint = {
        "file": Path(registered.checkpoint).name,
        "sha256": registered.checkpoint_sha256,
        "artifact_schema": MODEL_SCHEMA,
        "artifact_schema_version": MODEL_SCHEMA_VERSION,
        "architecture": selection.winner_params["arch"],
        "objective_variant": selection.winner_params["objective_variant"],
        "seed": registered.seed,
    }
    checkpoint_mismatches = [
        field
        for field, expected in expected_checkpoint.items()
        if checkpoint.get(field) != expected
    ]
    if checkpoint_mismatches:
        raise ValueError(
            f"paper evaluation {source} checkpoint disagrees with "
            f"winner.paper_checkpoints or winner params: {checkpoint_mismatches}"
        )

    expected_selection = paper_evaluator.selection_public_binding(selection, registered)
    binding = document.get("selection")
    if not isinstance(binding, Mapping):
        raise ValueError(f"paper evaluation {source} has no selection binding")
    if binding.get("sha256") != selection.sha256:
        raise ValueError(f"paper evaluation {source} has a stale selection SHA-256")
    require_exact_json(
        binding,
        expected_selection,
        location=f"paper evaluation {source} selection binding",
    )

    selected_provenance = paper_evaluator._validated_data_provenance(
        selection.document.get("data_provenance"), "selection"
    )
    evaluation_provenance = paper_evaluator._validated_data_provenance(
        document.get("data_provenance"), f"paper evaluation {source}"
    )
    if evaluation_provenance != selected_provenance:
        raise ValueError(f"paper evaluation {source} data provenance is stale")
    _validate_preprocessing(document.get("preprocessing"), selection, source)

    n_test = document.get("n_test")
    evaluated_records = document.get("evaluated_records")
    if (
        isinstance(n_test, bool)
        or not isinstance(n_test, int)
        or n_test <= 0
        or not isinstance(evaluated_records, list)
        or len(evaluated_records) != n_test
    ):
        raise ValueError(f"paper evaluation {source} has incomplete evaluated-record evidence")
    problem_digests: dict[int, str] = {}
    record_keys: set[str] = set()
    for index, record in enumerate(evaluated_records):
        if (
            not isinstance(record, Mapping)
            or record.get("record_index") != index
            or not paper_evaluator._valid_sha256(record.get("problem_digest"))
            or not paper_evaluator._valid_sha256(record.get("record_key"))
            or record["record_key"] in record_keys
        ):
            raise ValueError(f"paper evaluation {source} has an invalid evaluated record")
        problem_digests[index] = str(record["problem_digest"])
        record_keys.add(str(record["record_key"]))
        audit_contract.validate_audit_provenance(
            record.get("audit_provenance"),
            location=f"paper evaluation {source} evaluated_records[{index}]",
        )
    coverage = document.get("audit_coverage")
    if not isinstance(coverage, Mapping) or coverage.get("record_fraction_complete") != 1.0:
        raise ValueError(f"paper evaluation {source} has incomplete independent-audit coverage")
    ground_coverage = document.get("ground_reference_coverage")
    if not isinstance(ground_coverage, Mapping):
        raise ValueError(f"paper evaluation {source} has no ground-reference coverage")
    require_exact_json(
        ground_coverage.get("accepted_registry"),
        audit_contract.evaluation["accepted_ground_references"],
        location=f"paper evaluation {source} ground-reference registry",
    )
    by_ground_format = ground_coverage.get("by_format")
    if (
        ground_coverage.get("policy") != audit_contract.evaluation["ground_reference_policy"]
        or ground_coverage.get("generic_certified_optimal_accepted") is not False
        or ground_coverage.get("records") != n_test
        or not isinstance(by_ground_format, Mapping)
        or set(by_ground_format) != set(paper_evaluator._GROUND_REFERENCE_FORMATS)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in by_ground_format.values()
        )
        or sum(by_ground_format.values()) != n_test
    ):
        raise ValueError(f"paper evaluation {source} has invalid ground-reference coverage")
    audit_release = document.get("audit_release_commitment")
    if (
        not isinstance(audit_release, Mapping)
        or audit_release.get("schema") != paper_evaluator.AUDIT_RELEASE_MANIFEST_SCHEMA
        or audit_release.get("schema_version")
        != paper_evaluator.AUDIT_RELEASE_MANIFEST_SCHEMA_VERSION
        or not isinstance(audit_release.get("file"), str)
        or not paper_evaluator._valid_sha256(audit_release.get("sha256"))
        or audit_release.get("trust_model") != "sha256_supplied_out_of_band"
        or audit_release.get("shard_count") != int(audit_contract.audit_execution["shard_count"])
    ):
        raise ValueError(f"paper evaluation {source} has no external audit-release commitment")
    audit_artifacts = document.get("audit_artifacts")
    if not isinstance(audit_artifacts, list) or not audit_artifacts:
        raise ValueError(f"paper evaluation {source} has no immutable audit artifact")
    artifact_files: set[str] = set()
    manifest_files: set[str] = set()
    for artifact in audit_artifacts:
        if (
            not isinstance(artifact, Mapping)
            or not isinstance(artifact.get("file"), str)
            or not artifact["file"]
            or not paper_evaluator._valid_sha256(artifact.get("sha256"))
            or not isinstance(artifact.get("manifest_file"), str)
            or not artifact["manifest_file"]
            or not paper_evaluator._valid_sha256(artifact.get("manifest_sha256"))
        ):
            raise ValueError(f"paper evaluation {source} has an invalid audit artifact")
        artifact_file = str(artifact["file"])
        manifest_file = str(artifact["manifest_file"])
        if (
            artifact_file in artifact_files
            or manifest_file in manifest_files
            or manifest_file != f"{artifact_file}.manifest.json"
        ):
            raise ValueError(f"paper evaluation {source} repeats audit-shard evidence")
        artifact_files.add(artifact_file)
        manifest_files.add(manifest_file)
    if len(audit_artifacts) != int(audit_contract.audit_execution["shard_count"]):
        raise ValueError(f"paper evaluation {source} lacks complete audit-shard evidence")

    freeze = document.get("label_free_policy_freeze")
    registered_freeze = preregistered_policy_freeze_binding(preregistration, registered.seed)
    if (
        not isinstance(freeze, Mapping)
        or freeze.get("schema") != "embedbench.quality-v2-label-free-policy-freeze"
        or freeze.get("schema_version") != 2
        or not paper_evaluator._valid_sha256(freeze.get("sha256"))
        or freeze.get("file") != registered_freeze["file"]
        or freeze.get("sha256") != registered_freeze["sha256"]
        or freeze.get("record_count") != n_test
        or freeze.get("preregistration_manifest_caller_supplied_sha256_verified") is not True
        or freeze.get("policy_freeze_listed_in_preregistration") is not True
        or freeze.get("policy_freeze_bytes_match_preregistration") is not True
        or freeze.get("byte_exact_phase_2_replay_verified") is not True
        or freeze.get("audit_inputs_opened_after_preregistration_and_replay") is not True
    ):
        raise ValueError(f"paper evaluation {source} lacks a label-free policy freeze")

    return registered, _registered_metrics(
        document,
        source,
        audit_contract,
        problem_digests,
    )


def _common_contract(document: Mapping[str, Any]) -> dict[str, object]:
    fields = (
        "data_provenance",
        "audit_release_commitment",
        "audit_artifacts",
        "preprocessing",
        "deployment_host_contract",
        "exact_host_contract",
        "partition_record_counts",
        "n_test",
        "audit_coverage",
        "ground_reference_coverage",
        "evaluated_records",
        "evaluation_design",
        "legacy_test_exposure",
        "top_tolerance",
        "random_seed",
        "random_baseline_policy",
        "lcb_z",
        "selection_statistic",
        "audit_contract",
        "preregistration",
    )
    return {field: document.get(field) for field in fields}


def _cross_seed_cohort(document: Mapping[str, Any]) -> dict[str, object]:
    """Extract every model-independent support and baseline field."""

    full = document.get("full_support_metrics")
    budget_sweep = document.get("budget_sweep")
    if not isinstance(full, Mapping) or not isinstance(budget_sweep, Mapping):
        raise ValueError("paper evaluation has no cross-seed cohort evidence")
    n_test = document.get("n_test")
    full_eligible = full.get("eligible_indices")
    if (
        isinstance(n_test, bool)
        or not isinstance(n_test, int)
        or full.get("exact_mask") != "minor_embedding_feasible"
        or full.get("n_records") != n_test
        or not isinstance(full_eligible, list)
        or len(full_eligible) != n_test
        or any(not isinstance(indices, list) or not indices for indices in full_eligible)
    ):
        raise ValueError("paper evaluation has invalid full-support cohort evidence")

    baseline_names = ("resource", "original", "random")

    def baselines(container: Mapping[str, Any]) -> dict[str, object]:
        selectors = container.get("selectors")
        if not isinstance(selectors, Mapping):
            raise ValueError("paper evaluation has no model-independent baseline selectors")
        return {name: selectors.get(name) for name in baseline_names}

    full_contract = {
        field: full.get(field)
        for field in (
            "exact_mask",
            "n_records",
            "total_presented_candidates",
            "total_eligible_candidates",
            "eligible_indices",
            "support_uses_labels",
        )
    }
    full_contract["baselines"] = baselines(full)
    budgets = budget_sweep.get("budgets")
    if not isinstance(budgets, Mapping):
        raise ValueError("paper evaluation has no budget cohort evidence")
    coverage = budget_sweep.get("primary_record_coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError("paper evaluation has no primary-record coverage evidence")
    included = coverage.get("included_input_indices")
    excluded = coverage.get("excluded_input_indices")
    reasons = coverage.get("exclusion_reasons")
    included_count = coverage.get("included_records")
    excluded_count = coverage.get("excluded_records")
    if (
        not isinstance(included, list)
        or not isinstance(excluded, list)
        or not isinstance(reasons, Mapping)
        or included_count != len(included)
        or excluded_count != len(excluded)
        or coverage.get("input_records") != n_test
        or len(included) + len(excluded) != n_test
        or set(included) & set(excluded)
        or set(included) | set(excluded) != set(range(n_test))
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in reasons.values()
        )
        or sum(reasons.values()) != len(excluded)
    ):
        raise ValueError("paper evaluation has invalid primary-record coverage evidence")

    budget_contracts = {}
    for key in BUDGET_KEYS:
        result = budgets.get(key)
        if not isinstance(result, Mapping):
            raise ValueError(f"paper evaluation has no cohort evidence for budget {key}")
        eligible = result.get("eligible_indices")
        reference_qubits = result.get("reference_total_qubits")
        realized_budgets = result.get("budgets")
        input_indices = result.get("input_record_indices")
        if (
            result.get("n_records") != included_count
            or input_indices != included
            or not isinstance(reference_qubits, list)
            or not isinstance(realized_budgets, list)
            or not isinstance(eligible, list)
            or not (
                len(reference_qubits) == len(realized_budgets) == len(eligible) == included_count
            )
            or any(not isinstance(indices, list) or not indices for indices in eligible)
            or result.get("no_survivor") != 0
            or result.get("no_survivor_rate") != 0.0
        ):
            raise ValueError(f"paper evaluation has invalid cohort evidence for budget {key}")
        cohort = {
            field: result.get(field)
            for field in (
                "ratio",
                "reference_policy",
                "n_records",
                "input_record_indices",
                "reference_total_qubits",
                "budgets",
                "eligible_indices",
                "no_survivor",
                "no_survivor_rate",
            )
        }
        cohort["baselines"] = baselines(result)
        budget_contracts[key] = cohort
    return {
        "full_support": full_contract,
        "budget_protocol": {
            field: budget_sweep.get(field)
            for field in (
                "candidate_support",
                "support_uses_labels",
                "exact_mask",
                "budget_contract",
                "budget_reference",
                "primary_record_policy",
                "primary_record_coverage",
                "budget_ratios",
            )
        },
        "budgets": budget_contracts,
    }


def _metric_summary(values_by_seed: Mapping[int, float]) -> dict[str, object]:
    values = list(values_by_seed.values())
    return {
        "mean": statistics.fmean(values),
        "sample_std": statistics.stdev(values),
        "minimum": min(values),
        "maximum": max(values),
        "values_by_seed": {str(seed): value for seed, value in values_by_seed.items()},
    }


def _problem_level_paired_deltas(
    by_seed: Mapping[int, tuple[Path, str, dict[str, Any], dict[str, float]]],
    registered_seeds: Sequence[int],
    budget: str,
) -> dict[str, float]:
    per_record: dict[int, list[float]] = {}
    problem_by_record: dict[int, str] = {}
    for seed in registered_seeds:
        document = by_seed[seed][2]
        result = document["budget_sweep"]["budgets"][budget]
        learned = result["selectors"]["mean"]["per_record"]
        reference = result["selectors"]["original"]["per_record"]
        for learned_row, reference_row in zip(learned, reference, strict=True):
            record_index = learned_row["record_index"]
            if (
                reference_row["record_index"] != record_index
                or reference_row["problem_digest"] != learned_row["problem_digest"]
                or learned_row["p_solve"] is None
                or reference_row["p_solve"] is None
            ):
                raise ValueError("paper evaluations have incomplete learned-vs-stock pairing")
            problem_by_record[record_index] = learned_row["problem_digest"]
            per_record.setdefault(record_index, []).append(
                float(learned_row["p_solve"]) - float(reference_row["p_solve"])
            )
    if any(len(values) != len(registered_seeds) for values in per_record.values()):
        raise ValueError("paper evaluations have incomplete model-seed pairing")
    grouped: dict[str, list[float]] = {}
    for record_index, values in per_record.items():
        grouped.setdefault(problem_by_record[record_index], []).append(statistics.fmean(values))
    return {
        problem_digest: statistics.fmean(values)
        for problem_digest, values in sorted(grouped.items())
    }


def _inference_summary(
    values_by_problem: Mapping[str, float],
    *,
    contract: Mapping[str, Any],
    namespace: str,
) -> dict[str, object]:
    del namespace  # The registered analytic bound has no pseudo-random namespace.
    values = [float(values_by_problem[key]) for key in sorted(values_by_problem)]
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("problem-clustered inference requires finite paired deltas")
    mean_inference = contract.get("mean_inference")
    paired_test = contract.get("paired_test")
    if (
        not isinstance(mean_inference, Mapping)
        or mean_inference.get("method") != "two_sided_hoeffding_confidence_interval"
        or mean_inference.get("bounded_support") != [-1.0, 1.0]
        or not isinstance(paired_test, Mapping)
        or paired_test.get("method") != "one_sided_hoeffding_bounded_mean_test"
        or paired_test.get("null") != "population_mean_p_solve_delta_less_than_or_equal_to_zero"
        or paired_test.get("alternative") != "learned_mean_p_solve_greater_than_stock_original"
        or contract.get("sampling_assumption") != "independent_problem_clusters"
    ):
        raise RuntimeError("problem-clustered mean inference is not the registered procedure")
    if any(not -1.0 <= value <= 1.0 for value in values):
        raise ValueError("problem-level p_solve deltas must lie in the registered [-1,1] bound")
    observed = statistics.fmean(values)
    confidence_level = float(mean_inference["confidence_level"])
    minimum_clusters = int(mean_inference["minimum_problem_clusters"])
    if not 0.0 < confidence_level < 1.0 or minimum_clusters <= 1:
        raise RuntimeError("registered bounded-mean inference parameters are invalid")
    available = len(values) >= minimum_clusters
    result: dict[str, object] = {
        "status": "available" if available else "insufficient_problem_clusters",
        "estimate": observed,
        "confidence_interval_95": None,
        "n_problem_clusters": len(values),
        "minimum_problem_clusters": minimum_clusters,
        "problem_values": {key: float(values_by_problem[key]) for key in sorted(values_by_problem)},
        "one_sided_p_value": None,
        "confidence_interval_method": mean_inference["method"],
        "test_method": paired_test["method"],
        "inference_scope": contract["inference_scope"],
        "sampling_assumption": contract["sampling_assumption"],
    }
    if not available:
        return result
    alpha = 1.0 - confidence_level
    radius = 2.0 * math.sqrt(math.log(2.0 / alpha) / (2.0 * len(values)))
    result["confidence_interval_95"] = [
        max(-1.0, observed - radius),
        min(1.0, observed + radius),
    ]
    result["one_sided_p_value"] = (
        1.0 if observed <= 0.0 else min(1.0, math.exp(-len(values) * observed**2 / 2.0))
    )
    return result


def _holm_adjust(raw_p_values: Mapping[str, float]) -> dict[str, float]:
    ordered = sorted(raw_p_values, key=lambda key: (raw_p_values[key], key))
    adjusted: dict[str, float] = {}
    running = 0.0
    family_size = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (family_size - rank) * raw_p_values[key])
        adjusted[key] = min(1.0, running)
    return adjusted


def _secondary_problem_inference(
    by_seed: Mapping[int, tuple[Path, str, dict[str, Any], dict[str, float]]],
    registered_seeds: Sequence[int],
    audit_contract: PaperAuditContract,
) -> dict[str, object]:
    contract = audit_contract.secondary_inference
    by_budget_values = {
        budget: _problem_level_paired_deltas(by_seed, registered_seeds, budget)
        for budget in FINITE_BUDGET_KEYS
    }
    by_budget = {
        budget: _inference_summary(
            values,
            contract=contract,
            namespace=f"budget:{budget}",
        )
        for budget, values in by_budget_values.items()
    }
    raw_p_values = {
        budget: float(summary["one_sided_p_value"])
        for budget, summary in by_budget.items()
        if summary["one_sided_p_value"] is not None
    }
    adjusted = _holm_adjust(raw_p_values) if len(raw_p_values) == len(by_budget) else {}
    alpha = float(contract["per_budget_multiplicity"]["familywise_alpha"])
    for budget, summary in by_budget.items():
        summary["holm_adjusted_p_value"] = adjusted.get(budget)
        summary["reject_after_holm"] = adjusted[budget] <= alpha if budget in adjusted else False
    problem_sets = {frozenset(values) for values in by_budget_values.values()}
    if len(problem_sets) != 1:
        raise ValueError("finite budgets do not share one problem-cluster inference cohort")
    problems = sorted(next(iter(problem_sets)))
    primary_values = {
        problem: statistics.fmean(
            by_budget_values[budget][problem] for budget in FINITE_BUDGET_KEYS
        )
        for problem in problems
    }
    problem_cluster_coverage = {
        "support_problem_clusters": len(problems),
        "complete_case_problem_clusters": len(problems),
        "excluded_problem_clusters": 0,
        "problem_cluster_coverage_rate": 1.0,
        "registered_model_seed_count": len(registered_seeds),
    }
    for summary in by_budget.values():
        summary["problem_cluster_coverage"] = dict(problem_cluster_coverage)
    return {
        "contract": dict(contract),
        "problem_cluster_coverage": problem_cluster_coverage,
        "primary_across_finite_budgets": _inference_summary(
            primary_values,
            contract=contract,
            namespace="primary_across_finite_budgets",
        ),
        "by_budget": by_budget,
    }


def _secondary_endpoint_selector_summary(
    document: Mapping[str, Any],
    *,
    endpoint: str,
    support: str,
    selector: str,
) -> Mapping[str, Any]:
    secondary = document.get("secondary_endpoints")
    if not isinstance(secondary, Mapping):
        raise ValueError("paper evaluation has no secondary endpoint evidence")
    endpoint_value = secondary.get(endpoint)
    if not isinstance(endpoint_value, Mapping):
        raise ValueError(f"paper evaluation has no {endpoint} evidence")
    if endpoint == "strength_robustness" and support == "full_support":
        support_value = endpoint_value.get("full_support")
    else:
        budgets = endpoint_value.get("budgets")
        support_value = budgets.get(support) if isinstance(budgets, Mapping) else None
    selectors = support_value.get("selectors") if isinstance(support_value, Mapping) else None
    summary = selectors.get(selector) if isinstance(selectors, Mapping) else None
    if not isinstance(summary, Mapping):
        raise ValueError(
            f"paper evaluation lacks {endpoint}/{support}/{selector} per-record evidence"
        )
    return summary


def _secondary_endpoint_problem_values(
    by_seed: Mapping[int, tuple[Path, str, dict[str, Any], dict[str, float]]],
    registered_seeds: Sequence[int],
    *,
    endpoint: str,
    support: str,
    selector: str,
    metric: str,
) -> tuple[dict[str, float], dict[str, int | float]]:
    mean_metric = f"mean_{metric}"
    expected_identities: list[tuple[int, str, object]] | None = None
    values_by_record: dict[int, list[float]] = {}
    complete_by_record: dict[int, bool] = {}
    problem_by_record: dict[int, str] = {}
    support_records: int | None = None
    for seed in registered_seeds:
        summary = _secondary_endpoint_selector_summary(
            by_seed[seed][2],
            endpoint=endpoint,
            support=support,
            selector=selector,
        )
        per_record = summary.get("per_record")
        reported_support = summary.get("support_records")
        reported_paired = summary.get("paired_records")
        reported_excluded = summary.get("pairing_excluded_records")
        reported_rate = summary.get("paired_coverage_rate")
        if (
            not isinstance(per_record, list)
            or isinstance(reported_support, bool)
            or not isinstance(reported_support, int)
            or reported_support < 0
            or len(per_record) != reported_support
            or isinstance(reported_paired, bool)
            or not isinstance(reported_paired, int)
            or reported_paired < 0
            or isinstance(reported_excluded, bool)
            or not isinstance(reported_excluded, int)
            or reported_excluded < 0
            or reported_paired + reported_excluded != reported_support
            or isinstance(reported_rate, bool)
            or not isinstance(reported_rate, (int, float))
            or not math.isfinite(float(reported_rate))
            or not math.isclose(
                float(reported_rate),
                reported_paired / reported_support if reported_support else 0.0,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"paper evaluation has malformed {endpoint}/{support} paired coverage")
        if support_records is None:
            support_records = reported_support
        elif reported_support != support_records:
            raise ValueError(
                f"paper evaluations change {endpoint}/{support} support across model seeds"
            )
        identities: list[tuple[int, str, object]] = []
        seed_values: list[float] = []
        seen_indices: set[int] = set()
        counted_paired = 0
        for position, row in enumerate(per_record):
            if not isinstance(row, Mapping):
                raise ValueError(
                    f"paper evaluation has malformed {endpoint}/{support} row {position}"
                )
            record_index = row.get("record_index")
            problem_digest = row.get("problem_digest")
            reference_index = row.get("reference_index")
            selected_index = row.get("selection_index")
            authenticated = row.get("authenticated")
            paired = row.get("paired")
            value = row.get(metric)
            if (
                isinstance(record_index, bool)
                or not isinstance(record_index, int)
                or record_index < 0
                or record_index in seen_indices
                or not isinstance(problem_digest, str)
                or not paper_evaluator._valid_sha256(problem_digest)
                or (
                    reference_index is not None
                    and (
                        isinstance(reference_index, bool)
                        or not isinstance(reference_index, int)
                        or reference_index < 0
                    )
                )
                or (
                    selected_index is not None
                    and (
                        isinstance(selected_index, bool)
                        or not isinstance(selected_index, int)
                        or selected_index < 0
                    )
                )
                or not isinstance(authenticated, bool)
                or not isinstance(paired, bool)
            ):
                raise ValueError(
                    f"paper evaluation has malformed {endpoint}/{support} row {position}"
                )
            seen_indices.add(record_index)
            identities.append((record_index, problem_digest, reference_index))
            if expected_identities is not None and identities[-1] != expected_identities[position]:
                raise ValueError(
                    f"paper evaluations change {endpoint}/{support} rows across model seeds"
                )
            if paired:
                if not authenticated or selected_index is None or reference_index is None:
                    raise ValueError(
                        f"paper evaluation has unauthenticated {endpoint}/{support} pair"
                    )
                numeric_value = _finite_number(
                    value,
                    f"{endpoint}/{support}/{metric} row {position}",
                )
                counted_paired += 1
                seed_values.append(numeric_value)
            elif value is not None:
                raise ValueError(f"paper evaluation scores an unpaired {endpoint}/{support} row")
            if expected_identities is None:
                values_by_record[record_index] = []
                complete_by_record[record_index] = True
                problem_by_record[record_index] = problem_digest
            if paired:
                values_by_record[record_index].append(float(value))
            else:
                complete_by_record[record_index] = False
        if counted_paired != reported_paired:
            raise ValueError(f"paper evaluation miscounts {endpoint}/{support} paired records")
        reported_mean = summary.get(mean_metric)
        expected_mean = statistics.fmean(seed_values) if seed_values else None
        if expected_mean is None:
            if reported_mean is not None:
                raise ValueError(
                    f"paper evaluation aggregates absent {endpoint}/{support}/{metric} pairs"
                )
        elif not math.isclose(
            _finite_number(reported_mean, f"{endpoint}/{support}/{mean_metric}"),
            expected_mean,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError(f"paper evaluation misaggregates {endpoint}/{support}/{mean_metric}")
        if expected_identities is None:
            expected_identities = identities
        elif identities != expected_identities:
            raise ValueError(
                f"paper evaluations change {endpoint}/{support} rows across model seeds"
            )
    if support_records is None or expected_identities is None:
        raise ValueError(f"paper evaluations have no {endpoint}/{support} support")
    paired_record_values = {
        record_index: statistics.fmean(values)
        for record_index, values in values_by_record.items()
        if complete_by_record[record_index] and len(values) == len(registered_seeds)
    }
    records_by_problem: dict[str, list[int]] = {}
    for record_index, problem_digest in problem_by_record.items():
        records_by_problem.setdefault(problem_digest, []).append(record_index)
    values_by_problem = {
        problem_digest: statistics.fmean(
            paired_record_values[record_index] for record_index in record_indices
        )
        for problem_digest, record_indices in sorted(records_by_problem.items())
        if all(record_index in paired_record_values for record_index in record_indices)
    }
    paired_records = len(paired_record_values)
    support_problem_clusters = len(records_by_problem)
    complete_problem_clusters = len(values_by_problem)
    return values_by_problem, {
        "support_records": support_records,
        "paired_records": paired_records,
        "excluded_records": support_records - paired_records,
        "paired_coverage_rate": paired_records / support_records if support_records else 0.0,
        "support_problem_clusters": support_problem_clusters,
        "complete_case_problem_clusters": complete_problem_clusters,
        "excluded_problem_clusters": support_problem_clusters - complete_problem_clusters,
        "problem_cluster_coverage_rate": (
            complete_problem_clusters / support_problem_clusters
            if support_problem_clusters
            else 0.0
        ),
    }


def _secondary_endpoint_inference_summary(
    values_by_problem: Mapping[str, float],
    *,
    coverage: Mapping[str, int | float],
    contract: Mapping[str, Any],
    namespace: str,
) -> dict[str, object]:
    support_records = int(coverage["support_records"])
    paired_records = int(coverage["paired_records"])
    values = [float(values_by_problem[key]) for key in sorted(values_by_problem)]
    bootstrap = contract["bootstrap"]
    resamples = int(bootstrap["resamples"])
    confidence_level = float(bootstrap["confidence_level"])
    minimum_clusters = int(bootstrap["minimum_problem_clusters"])
    status = (
        "unavailable"
        if not values
        else "available"
        if paired_records == support_records
        else "partial"
    )
    inference_status = (
        status if not values or len(values) >= minimum_clusters else "insufficient_problem_clusters"
    )
    result: dict[str, object] = {
        "status": inference_status,
        "coverage_status": status,
        "support_records": support_records,
        "paired_records": paired_records,
        "excluded_records": int(coverage["excluded_records"]),
        "paired_coverage_rate": float(coverage["paired_coverage_rate"]),
        "support_problem_clusters": int(coverage["support_problem_clusters"]),
        "complete_case_problem_clusters": int(coverage["complete_case_problem_clusters"]),
        "excluded_problem_clusters": int(coverage["excluded_problem_clusters"]),
        "problem_cluster_coverage_rate": float(coverage["problem_cluster_coverage_rate"]),
        "estimate": None,
        "confidence_interval": None,
        "confidence_level": confidence_level,
        "n_problem_clusters": len(values),
        "minimum_problem_clusters": minimum_clusters,
        "problem_values": {key: float(values_by_problem[key]) for key in sorted(values_by_problem)},
        "bootstrap_resamples": resamples,
        "unavailable_reason": (
            "no_complete_pair_across_all_registered_model_seeds"
            if not values
            else "fewer_than_registered_minimum_problem_clusters"
            if len(values) < minimum_clusters
            else None
        ),
    }
    if not values:
        return result
    if any(not math.isfinite(value) for value in values):
        raise ValueError("secondary endpoint inference requires finite problem-level deltas")
    estimate = statistics.fmean(values)
    result["estimate"] = estimate
    if len(values) < minimum_clusters:
        return result
    namespace_seed = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:8], "big")
    rng = random.Random(int(bootstrap["seed"]) + namespace_seed)
    bootstrap_values = [
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(resamples)
    ]
    alpha = 1.0 - confidence_level
    interval = np.quantile(
        np.asarray(bootstrap_values, dtype=np.float64),
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method=str(bootstrap["quantile_method"]),
    )
    result["confidence_interval"] = [float(interval[0]), float(interval[1])]
    return result


def _combined_secondary_endpoint_status(statuses: Sequence[str]) -> str:
    if not statuses or all(status == "unavailable" for status in statuses):
        return "unavailable"
    if all(status == "insufficient_problem_clusters" for status in statuses):
        return "insufficient_problem_clusters"
    if all(status == "available" for status in statuses):
        return "available"
    return "partial"


def _secondary_endpoint_clustered_inference(
    by_seed: Mapping[int, tuple[Path, str, dict[str, Any], dict[str, float]]],
    registered_seeds: Sequence[int],
    audit_contract: PaperAuditContract,
) -> dict[str, object]:
    contract = audit_contract.secondary_endpoint_inference
    endpoint_contracts = contract.get("endpoints")
    bootstrap = contract.get("bootstrap")
    if (
        contract.get("selection_role") != "report_only_never_selection_or_primary_pass_fail"
        or contract.get("selector") != "mean"
        or contract.get("reference") != "stock_minorminer_original"
        or contract.get("cluster_unit") != "quality_problem_digest"
        or contract.get("model_seed_reduction") != "arithmetic_mean_within_record"
        or contract.get("record_reduction") != "arithmetic_mean_within_problem_digest"
        or contract.get("missing_pair_action") != "unavailable_or_partial_with_explicit_coverage"
        or not isinstance(bootstrap, Mapping)
        or bootstrap.get("method") != "percentile_problem_cluster_bootstrap"
        or bootstrap.get("minimum_problem_clusters") != 20
        or bootstrap.get("interval_role") != "pointwise_descriptive_not_hypothesis_test"
        or bootstrap.get("inference_scope") != "conditional_on_four_registered_model_checkpoints"
        or not isinstance(endpoint_contracts, Mapping)
    ):
        raise RuntimeError("secondary endpoint inference implementation is not registered")
    selector = str(contract["selector"])

    def support_summary(endpoint: str, support: str, metrics: Sequence[str]) -> dict[str, object]:
        summaries: dict[str, object] = {}
        common_coverage: dict[str, int | float] | None = None
        statuses: list[str] = []
        for metric in metrics:
            values, coverage = _secondary_endpoint_problem_values(
                by_seed,
                registered_seeds,
                endpoint=endpoint,
                support=support,
                selector=selector,
                metric=metric,
            )
            if common_coverage is None:
                common_coverage = coverage
            elif coverage != common_coverage:
                raise ValueError(
                    f"paper evaluations change paired coverage across {endpoint}/{support} metrics"
                )
            summary = _secondary_endpoint_inference_summary(
                values,
                coverage=coverage,
                contract=contract,
                namespace=f"secondary_endpoint:{endpoint}:{support}:{metric}",
            )
            summaries[metric] = summary
            statuses.append(str(summary["status"]))
        if common_coverage is None:
            raise RuntimeError(f"secondary endpoint {endpoint} has no registered metrics")
        return {
            "status": _combined_secondary_endpoint_status(statuses),
            "coverage": dict(common_coverage),
            **summaries,
        }

    residual_contract = endpoint_contracts.get("residual_connectivity")
    strength_contract = endpoint_contracts.get("strength_robustness")
    if not isinstance(residual_contract, Mapping) or not isinstance(strength_contract, Mapping):
        raise RuntimeError("secondary endpoint inference registry is incomplete")
    residual_supports = tuple(residual_contract.get("supports", ()))
    strength_supports = tuple(strength_contract.get("supports", ()))
    residual_metrics = tuple(residual_contract.get("metrics", ()))
    strength_metrics = tuple(strength_contract.get("metrics", ()))
    if residual_supports != BUDGET_KEYS or strength_supports != ("full_support", *BUDGET_KEYS):
        raise RuntimeError("secondary endpoint inference support registry changed")
    residual_budgets = {
        support: support_summary("residual_connectivity", support, residual_metrics)
        for support in residual_supports
    }
    strength_full = support_summary(
        "strength_robustness",
        "full_support",
        strength_metrics,
    )
    strength_budgets = {
        support: support_summary("strength_robustness", support, strength_metrics)
        for support in BUDGET_KEYS
    }
    return {
        "selection_role": contract["selection_role"],
        "selector": selector,
        "reference": contract["reference"],
        "contract": dict(contract),
        "residual_connectivity": {
            "status": _combined_secondary_endpoint_status(
                [str(value["status"]) for value in residual_budgets.values()]
            ),
            "budgets": residual_budgets,
        },
        "strength_robustness": {
            "status": _combined_secondary_endpoint_status(
                [str(strength_full["status"])]
                + [str(value["status"]) for value in strength_budgets.values()]
            ),
            "full_support": strength_full,
            "budgets": strength_budgets,
        },
    }


def _paper_evaluation_args(
    args: argparse.Namespace,
    checkpoint: paper_evaluator.RegisteredPaperCheckpoint,
    *,
    phase: str,
    policy_freeze: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        phase=phase,
        files=list(args.files),
        splits=args.splits,
        checkpoint=str(checkpoint.resolved_path),
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        selection=args.selection,
        selection_sha256=args.selection_sha256,
        selection_root=args.selection_root,
        audit_labels=list(args.audit_labels) if phase == "score" else [],
        audit_release_manifest=(args.audit_release_manifest if phase == "score" else None),
        audit_release_manifest_sha256=(
            args.audit_release_manifest_sha256 if phase == "score" else None
        ),
        policy_freeze=policy_freeze,
        preregistration_manifest=(args.preregistration_manifest if phase == "score" else None),
        preregistration_manifest_sha256=(
            args.preregistration_manifest_sha256 if phase == "score" else None
        ),
        audit_contract=args.audit_contract,
        audit_contract_sha256=args.audit_contract_sha256,
        out=os.devnull,
        partition="test",
        evaluation_mode="paper",
    )


def aggregate(args: argparse.Namespace) -> dict[str, object]:
    audit_contract = load_paper_audit_contract(
        args.audit_contract,
        args.audit_contract_sha256,
    )
    try:
        selection_payload = Path(args.selection).read_bytes()
    except OSError as error:
        raise ValueError(f"invalid selection artifact {args.selection}") from error
    from select_training_grid import revalidate_selection_artifact

    revalidated_selection = revalidate_selection_artifact(
        args.selection,
        args.selection_sha256,
        root=args.selection_root,
        audit_contract=audit_contract,
        selection_payload=selection_payload,
    )
    selection = paper_evaluator.load_selection_artifact(
        args.selection,
        args.selection_sha256,
        args.selection_root,
        audit_contract,
        selection_payload=selection_payload,
    )
    require_exact_json(
        selection.document,
        revalidated_selection,
        location="aggregator selection artifact",
    )
    if len(selection.registered_seeds) != int(
        audit_contract.model_seed_aggregation["registered_seed_count"]
    ):
        raise ValueError("selection registered seeds disagree with the frozen paper audit contract")
    if len(args.policy_freezes) != len(selection.registered_seeds) or len(
        set(map(str, args.policy_freezes))
    ) != len(selection.registered_seeds):
        raise ValueError("paper aggregation requires one distinct freeze per registered seed")
    preregistration = load_paper_preregistration(
        args.preregistration_manifest,
        args.preregistration_manifest_sha256,
        audit_contract=audit_contract,
        selection_document=selection.document,
        selection_file=selection.path.name,
        selection_sha256=selection.sha256,
        audit_source_root=Path(__file__).parents[1],
    )
    verified_freezes = verify_preregistered_policy_freezes(
        preregistration,
        args.policy_freezes,
    )
    registered_freeze_sha256s = [str(entry["sha256"]) for entry in preregistration.policy_freezes]
    evaluation_paths = [Path(path).resolve() for path in args.evaluations]
    if len(evaluation_paths) != len(selection.registered_seeds) or len(
        set(evaluation_paths)
    ) != len(evaluation_paths):
        raise ValueError("paper aggregation requires exactly every registered seed evaluation once")

    # Complete phase 1 for all seeds before any audit-label path is opened.
    replayed_freezes: dict[int, dict[str, object]] = {}
    for index, checkpoint in enumerate(selection.paper_checkpoints):
        phase_one = paper_evaluator.evaluate(
            _paper_evaluation_args(args, checkpoint, phase="freeze"),
        )
        expected_payload = paper_evaluator._canonical_json_bytes(phase_one)
        paper_evaluator._load_registered_policy_freeze(
            args.policy_freezes[index],
            registered_freeze_sha256s[index],
            expected_payload,
        )
        replayed_freezes[checkpoint.seed] = phase_one

    # Only after every policy is frozen do we consume and independently rescore all shards.
    replayed_evaluations: dict[int, dict[str, object]] = {}
    for index, checkpoint in enumerate(selection.paper_checkpoints):
        replayed = paper_evaluator.evaluate(
            _paper_evaluation_args(
                args,
                checkpoint,
                phase="score",
                policy_freeze=args.policy_freezes[index],
            )
        )
        replayed_evaluations[checkpoint.seed] = replayed

    by_seed: dict[int, tuple[Path, str, dict[str, Any], dict[str, float]]] = {}
    common: dict[str, object] | None = None
    common_cohort: dict[str, object] | None = None
    common_runtime: dict[str, object] | None = None
    for source in evaluation_paths:
        try:
            evaluation_payload = source.read_bytes()
        except OSError as error:
            raise ValueError(f"invalid paper evaluation {source}") from error
        document = _object_from_payload(evaluation_payload, source, "paper evaluation")
        checkpoint, metrics = _validate_evaluation(
            document,
            source,
            selection,
            audit_contract,
            preregistration,
        )
        require_exact_json(
            document,
            replayed_evaluations[checkpoint.seed],
            location=f"paper evaluation {source}",
        )
        if checkpoint.seed in by_seed:
            raise ValueError(
                "paper aggregation requires exactly every registered seed evaluation once"
            )
        current_common = _common_contract(document)
        if common is None:
            common = current_common
        elif current_common != common:
            raise ValueError(f"paper evaluation {source} changes the common test/audit protocol")
        current_cohort = _cross_seed_cohort(document)
        if common_cohort is None:
            common_cohort = current_cohort
        elif current_cohort != common_cohort:
            raise ValueError(f"paper evaluation {source} changes the cross-seed cohort")
        current_runtime = {
            "runtime_provenance": document.get("runtime_provenance"),
            "audit_runtime_provenance": document.get("audit_runtime_provenance"),
            "audit_source": document.get("audit_source"),
        }
        if common_runtime is None:
            common_runtime = current_runtime
        elif current_runtime != common_runtime:
            raise ValueError(f"paper evaluation {source} changes runtime provenance")
        by_seed[checkpoint.seed] = (
            source,
            hashlib.sha256(evaluation_payload).hexdigest(),
            document,
            metrics,
        )

    if tuple(seed for seed in selection.registered_seeds if seed in by_seed) != (
        selection.registered_seeds
    ) or set(by_seed) != set(selection.registered_seeds):
        missing = [seed for seed in selection.registered_seeds if seed not in by_seed]
        unexpected = [seed for seed in by_seed if seed not in selection.registered_seeds]
        raise ValueError(
            "paper aggregation requires exactly every registered seed evaluation once; "
            f"missing={missing}, unexpected={unexpected}"
        )

    metric_paths = list(next(iter(by_seed.values()))[3])
    if any(list(row[3]) != metric_paths for row in by_seed.values()):
        raise RuntimeError("paper evaluations disagree on registered metric paths")
    aggregates = {
        path: _metric_summary({seed: by_seed[seed][3][path] for seed in selection.registered_seeds})
        for path in metric_paths
    }
    primary_path = "budget_sweep.primary_metric"
    primary = {
        "name": paper_evaluator.PRIMARY_METRIC,
        "path": primary_path,
        **aggregates[primary_path],
    }
    secondary_metric_paths = [
        path for path in metric_paths if path.startswith("secondary_endpoints.")
    ]
    secondary_availability_by_seed = {
        str(seed): {
            endpoint_name: {
                "status": by_seed[seed][2]["secondary_endpoints"][endpoint_name]["status"],
                "coverage": by_seed[seed][2]["secondary_endpoints"][endpoint_name]["coverage"],
            }
            for endpoint_name in (
                "exact_feasibility_budget_survival",
                "terminal_resources",
                "residual_connectivity",
                "strength_robustness",
            )
        }
        for seed in selection.registered_seeds
    }
    secondary_inference = _secondary_problem_inference(
        by_seed,
        selection.registered_seeds,
        audit_contract,
    )
    secondary_endpoint_inference = _secondary_endpoint_clustered_inference(
        by_seed,
        selection.registered_seeds,
        audit_contract,
    )
    if common is None:  # pragma: no cover - exact registered-seed coverage guard
        raise RuntimeError("paper aggregation has no common evaluation contract")
    document = selection.document
    return {
        "artifact_schema": AGGREGATE_SCHEMA,
        "artifact_schema_version": AGGREGATE_SCHEMA_VERSION,
        "evaluation_mode": "paper",
        "partition": "test",
        "provisional": audit_contract.evaluation["provisional"],
        "evaluation_design": audit_contract.evaluation["evaluation_design"],
        "legacy_test_exposure": audit_contract.evaluation["legacy_test_exposure"],
        "aggregation_contract": audit_contract.model_seed_aggregation["coverage"],
        "audit_contract": audit_contract.public_binding(),
        "preregistration": preregistration.public_binding(),
        "ground_reference_coverage": common["ground_reference_coverage"],
        "model_seed_aggregation": dict(audit_contract.model_seed_aggregation),
        "independent_revalidation": {
            "training_cells_replayed": selection.document["registered_cell_count"],
            "winner_checkpoints_phase_1_replayed": len(replayed_freezes),
            "winner_checkpoints_phase_2_replayed": len(replayed_evaluations),
            "audit_shards_validated_per_checkpoint": len(args.audit_labels),
            "preregistration_manifest_caller_supplied_sha256_verified": True,
            "all_policy_freeze_bytes_match_preregistration": (
                len(verified_freezes) == len(selection.registered_seeds)
            ),
            "policy_freezes": [
                {
                    "seed": checkpoint.seed,
                    "file": Path(args.policy_freezes[index]).name,
                    "sha256": registered_freeze_sha256s[index],
                    "fresh_replay_sha256": hashlib.sha256(
                        paper_evaluator._canonical_json_bytes(replayed_freezes[checkpoint.seed])
                    ).hexdigest(),
                }
                for index, checkpoint in enumerate(selection.paper_checkpoints)
            ],
        },
        "selection": {
            "file": selection.path.name,
            "sha256": selection.sha256,
            "artifact_schema": document["schema"],
            "artifact_schema_version": document["schema_version"],
            "grid_id": document["grid_id"],
            "grid_sha256": document["grid_sha256"],
            "source_sha256": document["source_sha256"],
            "selector_sha256": document["selector_sha256"],
            "stage": document["stage"],
            "winner_params": dict(selection.winner_params),
            "paper_checkpoints": [entry.public() for entry in selection.paper_checkpoints],
        },
        "registered_seeds": list(selection.registered_seeds),
        "n_seed_evaluations": len(by_seed),
        "evaluations": [
            {
                "seed": seed,
                "file": by_seed[seed][0].name,
                "sha256": by_seed[seed][1],
                "cell_id": selection.paper_checkpoints[index].cell_id,
                "checkpoint": selection.paper_checkpoints[index].checkpoint,
                "checkpoint_sha256": selection.paper_checkpoints[index].checkpoint_sha256,
            }
            for index, seed in enumerate(selection.registered_seeds)
        ],
        "standard_deviation": audit_contract.model_seed_aggregation["dispersion"],
        "registered_metric_paths": metric_paths,
        "primary_metric": primary,
        "metric_aggregates": aggregates,
        "secondary_endpoints": {
            "selection_role": audit_contract.secondary_endpoints["selection_role"],
            "availability_policy": audit_contract.secondary_endpoints["availability_policy"],
            "registration": dict(audit_contract.secondary_endpoints),
            "registered_metric_paths": secondary_metric_paths,
            "availability_by_seed": secondary_availability_by_seed,
        },
        "secondary_problem_clustered_inference": secondary_inference,
        "secondary_endpoint_clustered_inference": secondary_endpoint_inference,
    }


def _atomic_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    artifact = aggregate(args)
    _atomic_json(Path(args.out), artifact)
    print(json.dumps(artifact["primary_metric"], sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
