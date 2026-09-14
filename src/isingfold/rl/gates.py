"""The four release gates that decide whether to advance.

Spec: Rev2 section "A staged corpus plan and release gates" and the slices of section 8.2.
Each gate answers one question with a measurement, and a failing gate stops the next stage
rather than being skipped by a larger model.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import operator
import random
from collections import Counter
from dataclasses import replace
from pathlib import Path
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np
import networkx as nx

from isingfold.embedding import LogicalProblem
from isingfold.rl.contracts import (
    OPCODES,
    Context,
    DecisionState,
    InitFailureRecord,
    Mode,
    TerminalRecord,
    chain_key,
    stable_digest,
)
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.prepared import PreparedTask
from isingfold.rl.data.quality import (
    QUALITY_INITIALIZER_BANK_PROTOCOL,
    quality_initializer_bank_contract,
    replay_decision_with_action_envelope,
)
from isingfold.rl.data.quality_resolution_plan import (
    QUALITY_RESOLUTION_PLAN_SCHEMA,
    QUALITY_RESOLUTION_PLAN_VERSION,
    QualityResolutionPlan,
)
from isingfold.rl.data.exact_conformance import (
    EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES,
    EXACT_CONFORMANCE_MAX_TASKS,
)
from isingfold.rl.data.selector_labels import load_selector_metadata, load_selector_records
from isingfold.rl.data.program_oracle import (
    brute_force_program_identity,
    implementation_identity as program_oracle_identity,
)
from isingfold.rl.data.structural import ContinuationDomain, exact_feasibility
from isingfold.rl.data.structural_oracle import (
    brute_force_feasibility,
    implementation_identity as structural_oracle_identity,
)
from isingfold.rl.env import (
    EmbeddingEnv,
    EmbeddingTask,
    StrengthSelector,
    fixed_strength_selector,
    task_initializer,
)
from isingfold.rl.evaluator import sample_program
from isingfold.rl.program import check_faithfulness, program_features
from isingfold.rl.proposal import (
    LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
    router_initializer,
)
from isingfold.rl.initializer_bank import InitializerSnapshotBank
from isingfold.rl.validate import p_return, p_search
from isingfold.rl.strength import selector_regret
from isingfold.rl.strength_tensorize import build_strength_inputs


SUPPORT_HEADROOM_INFERENCE_PROTOCOL = (
    "independent-confirmation-normal-lineage-lcb-plus-all-strata-coverage-v1"
)
SUPPORT_HEADROOM_MINIMUM_EFFECT = 0.02
SUPPORT_HEADROOM_ONE_SIDED_ALPHA = 0.05
SUPPORT_HEADROOM_MINIMUM_STRATUM_COVERAGE = 0.5
SUPPORT_HEADROOM_BERNOULLI_DIFFERENCE_VARIANCE_BOUND = 0.5
SUPPORT_FULL_AUDIT_PROTOCOL = "authenticated-k2-full-legal-support-v1"
SUPPORT_FULL_AUDIT_MINIMUM_LINEAGES = 128
SUPPORT_FULL_AUDIT_REQUIRED_OPCODES = (
    "REWRITE_GROUP",
    "REPAIR_GROUP",
    "RESTART",
)


def _selector_scalar(value: object, name: str) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0.0:
        raise ValueError(f"graph selector {name} must be positive and finite")
    return converted


def _select_strength(
    selector: object,
    task: EmbeddingTask,
    ctx: Context,
    chains,
    programs,
    features,
) -> int:
    """Use the same graph-selector deployment boundary as the environment."""

    graph_attributes = (
        "select_embedding",
        "deployment_ready",
        "normalizer_digest",
        "coefficient_transform_scale",
    )
    if all(hasattr(selector, name) for name in graph_attributes):
        if selector.deployment_ready is not True:
            raise ValueError("graph strength selector is not deployment-ready")
        normalizer_digest = selector.normalizer_digest
        if not isinstance(normalizer_digest, str) or not normalizer_digest:
            raise ValueError("graph selector has no normalizer identity")
        graph_inputs = build_strength_inputs(
            ctx=ctx,
            logical=task.logical,
            host=task.host,
            problem=task.problem,
            chains=chains,
            programs=programs,
            coef_scale=_selector_scalar(
                selector.coefficient_transform_scale, "coefficient transform scale"
            ),
            normalizer_digest=normalizer_digest,
        )
        selected = selector.select_embedding(graph_inputs)
    else:
        selected = selector(programs, features)
    try:
        index = operator.index(selected)
    except TypeError as error:
        raise ValueError("strength selector returned a non-integral index") from error
    if not 0 <= index < len(programs):
        raise ValueError("strength selector returned an out-of-range index")
    return index


def gate_conformance(tasks: Sequence[EmbeddingTask], ctx: Context) -> dict[str, object]:
    """Gate 1: exact certificates agree, and programming reproduces the intended energy.

    Checks the witness against the independent validator, the four compiled programs against
    the coefficient-sum identity, and an exact structural label against a tiny enumeration.
    """

    if not tasks:
        raise ValueError("exact-conformance corpus cannot be empty")
    if len(tasks) > EXACT_CONFORMANCE_MAX_TASKS:
        raise ValueError(
            "exact-conformance corpus exceeds the registered bounded task cap "
            f"({EXACT_CONFORMANCE_MAX_TASKS})"
        )
    oversized = [
        task.name
        for task in tasks
        if task.logical.number_of_nodes() > EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES
    ]
    if oversized:
        raise ValueError(
            "exact-conformance corpus contains a task above the registered logical-size "
            f"cap ({EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES}): {sorted(oversized)}"
        )

    witness_ok = 0
    program_ok = 0
    program_oracle_ok = 0
    program_oracle_assignments = 0
    for task in tasks:
        witness = task.initial_embedding or task.witness
        if witness is None:
            continue
        receipt, programs = p_return(witness, task.logical, task.host, task.problem, ctx)
        witness_ok += int(receipt.valid)
        conventional_ok = bool(
            receipt.valid
            and len(programs) == len(ctx.strength_ratios)
            and all(
                check_faithfulness(
                    program,
                    witness,
                    task.host,
                    task.problem,
                    ctx.field_limit,
                    ctx.coupler_limit,
                ).ok
                for program in programs
            )
        )
        oracle_reports = tuple(
            brute_force_program_identity(
                witness,
                task.host,
                task.problem,
                program,
                field_limit=ctx.field_limit,
                coupler_limit=ctx.coupler_limit,
            )
            for program in programs
        )
        program_oracle_assignments += sum(report.assignments_checked for report in oracle_reports)
        independent_ok = bool(
            receipt.valid
            and len(oracle_reports) == len(ctx.strength_ratios)
            and all(report.ok for report in oracle_reports)
        )
        program_oracle_ok += int(independent_ok)
        program_ok += int(conventional_ok and independent_ok)

    # These motifs are deliberately fixed and tiny.  Running the independent oracle on an
    # arbitrary witness-derived window can become exponential and can accidentally omit the
    # boundary-coupler condition that the structural label is meant to certify.
    motif_logical = nx.Graph([("movable", "frozen")])
    motif_host = nx.Graph([(0, 1)])
    motif_chains = {"movable": frozenset((0,)), "frozen": frozenset((1,))}
    positive_domain = ContinuationDomain(
        ("movable",), frozenset((0,)), max_chain=1, qubit_cap=2, timeout_ms=1_500
    )
    negative_domain = ContinuationDomain(
        ("movable",), frozenset((1,)), max_chain=1, qubit_cap=2, timeout_ms=1_500
    )
    positive = exact_feasibility(motif_chains, motif_logical, motif_host, positive_domain)
    positive_oracle = brute_force_feasibility(
        motif_chains, motif_logical, motif_host, positive_domain
    )
    negative = exact_feasibility(motif_chains, motif_logical, motif_host, negative_domain)
    negative_oracle = brute_force_feasibility(
        motif_chains, motif_logical, motif_host, negative_domain
    )
    structural_checked = int(positive.feasible is not None)
    structural_agree = int(
        positive.feasible is not None
        and positive.feasible is positive_oracle.feasible
        and positive_oracle.feasible
    )
    negative_checked = int(negative.feasible is not None)
    negative_agree = int(
        negative.feasible is not None
        and negative.feasible is negative_oracle.feasible
        and not negative_oracle.feasible
    )
    oracle_assignments_checked = (
        positive_oracle.assignments_checked + negative_oracle.assignments_checked
    )

    # A three-qubit handoff requires the centre qubit to be claimed by both chains while a seam
    # is live.  O1 must admit that repair state, whereas terminal return must reject it.
    overlap_logical = nx.Graph([("left", "right")])
    overlap_host = nx.path_graph(3)
    overlapping = {
        "left": frozenset((0, 1)),
        "right": frozenset((1, 2)),
    }
    overlap_problem = LogicalProblem.from_dicts(
        {"left": 0.25, "right": -0.5}, {("left", "right"): -1.0}
    )
    overlap_ctx = replace(ctx, qubit_cap=max(3, ctx.qubit_cap))
    search_receipt = p_search(
        overlapping,
        overlap_logical,
        overlap_host,
        overlap_ctx.qubit_cap,
        overlap_ctx.overlap,
    )
    terminal_receipt, _ = p_return(
        overlapping, overlap_logical, overlap_host, overlap_problem, overlap_ctx
    )
    overlap_search_checked = 1
    overlap_search_admissible = int(search_receipt.valid)
    overlap_return_rejected = int(not terminal_receipt.valid)
    n = sum(1 for task in tasks if task.initial_embedding or task.witness)
    task_ids = [task.name for task in tasks]
    base_lineages = [task.lineage for task in tasks]
    unique_task_id_count = len(
        {task_id for task_id in task_ids if isinstance(task_id, str) and task_id}
    )
    unique_base_lineage_count = len(
        {lineage for lineage in base_lineages if isinstance(lineage, str) and lineage}
    )
    exact_task_count = len(tasks) == EXACT_CONFORMANCE_MAX_TASKS
    registered_population_ok = bool(
        exact_task_count
        and unique_task_id_count == EXACT_CONFORMANCE_MAX_TASKS
        and unique_base_lineage_count == EXACT_CONFORMANCE_MAX_TASKS
    )
    structural_unknown = 1 - structural_checked
    return {
        "instances_requested": len(tasks),
        "eligible_instances": n,
        "witness_valid_rate": witness_ok / n if n else 0.0,
        "program_faithful_rate": program_ok / n if n else 0.0,
        "structural_checked": structural_checked,
        "structural_unknown": structural_unknown,
        "structural_agreement": (
            structural_agree / structural_checked if structural_checked else 0.0
        ),
        "positive_structural_checked": structural_checked,
        "positive_structural_agreement": (
            structural_agree / structural_checked if structural_checked else 0.0
        ),
        "negative_structural_checked": negative_checked,
        "negative_structural_unknown": 1 - negative_checked,
        "negative_structural_agreement": (
            negative_agree / negative_checked if negative_checked else 0.0
        ),
        "overlap_search_checked": overlap_search_checked,
        "overlap_search_admissible_rate": (
            overlap_search_admissible / overlap_search_checked if overlap_search_checked else 0.0
        ),
        "overlap_return_rejection_rate": (
            overlap_return_rejected / overlap_search_checked if overlap_search_checked else 0.0
        ),
        "independent_oracle": {
            **structural_oracle_identity(),
            "assignments_checked": oracle_assignments_checked,
            "positive_cases": structural_checked,
            "negative_cases": negative_checked,
            "motifs": {
                "boundary_coupler_to_frozen_context": {
                    "expected": True,
                    "production": positive.feasible,
                    "oracle": positive_oracle.feasible,
                },
                "blocked_frozen_context": {
                    "expected": False,
                    "production": negative.feasible,
                    "oracle": negative_oracle.feasible,
                },
                "overlap_required_handoff": {
                    "search_expected": True,
                    "search_observed": search_receipt.valid,
                    "terminal_expected": False,
                    "terminal_observed": terminal_receipt.valid,
                },
            },
        },
        "independent_program_oracle": {
            **program_oracle_identity(),
            "eligible_instances": n,
            "passing_instances": program_oracle_ok,
            "logical_assignments_checked": program_oracle_assignments,
        },
        "exact_bounds": {
            "max_tasks": EXACT_CONFORMANCE_MAX_TASKS,
            "max_logical_variables_per_task": EXACT_CONFORMANCE_MAX_LOGICAL_VARIABLES,
        },
        "registered_population": {
            "exact_task_count": exact_task_count,
            "task_count": len(tasks),
            "unique_task_id_count": unique_task_id_count,
            "unique_base_lineage_count": unique_base_lineage_count,
        },
        "pass": bool(
            registered_population_ok
            and n == len(tasks)
            and witness_ok == n
            and program_ok == n
            and program_oracle_ok == n
            and structural_checked == 1
            and structural_unknown == 0
            and structural_agree == 1
            and negative_checked == 1
            and negative_agree == 1
            and overlap_search_checked == 1
            and overlap_search_admissible == 1
            and overlap_return_rejected == 1
        ),
    }


def support_headroom_inference(
    headrooms: Sequence[float],
    stratum_ids: Sequence[str],
    *,
    minimum_headroom: float = SUPPORT_HEADROOM_MINIMUM_EFFECT,
    one_sided_alpha: float = SUPPORT_HEADROOM_ONE_SIDED_ALPHA,
    minimum_stratum_coverage: float = SUPPORT_HEADROOM_MINIMUM_STRATUM_COVERAGE,
    evaluator_reads: int = 256,
) -> dict[str, object]:
    """Apply the registered uncertainty and coverage rule to oracle headroom.

    Each input row is an independently confirmed difference for one sampled base lineage.  The
    one-sided normal lower confidence bound adds the between-lineage standard error and a
    worst-case finite-read Bernoulli variance term.  It is an explicit screening rule over those
    audit differences, not a confidence interval for the unrestricted embedding optimum.  A
    sampled stratum passes coverage only when the registered fraction of its rows individually
    reach the meaningful-effect threshold.
    """

    values = np.asarray(headrooms, dtype=np.float64)
    strata = tuple(stratum_ids)
    if values.ndim != 1 or not len(values):
        raise ValueError("support-headroom inference requires a nonempty vector")
    if len(strata) != len(values):
        raise ValueError("support-headroom strata must align one-to-one with headroom rows")
    if not np.isfinite(values).all() or np.any(values < -1.0) or np.any(values > 1.0):
        raise ValueError("support headrooms must be finite probability differences in [-1, 1]")
    if any(not isinstance(stratum, str) or not stratum for stratum in strata):
        raise ValueError("support-headroom stratum IDs must be nonempty strings")
    if (
        isinstance(minimum_headroom, bool)
        or not isinstance(minimum_headroom, (int, float))
        or not math.isfinite(float(minimum_headroom))
        or not 0.0 <= float(minimum_headroom) <= 1.0
    ):
        raise ValueError("minimum support headroom must lie in [0, 1]")
    if (
        isinstance(one_sided_alpha, bool)
        or not isinstance(one_sided_alpha, (int, float))
        or not math.isfinite(float(one_sided_alpha))
        or not 0.0 < float(one_sided_alpha) < 0.5
    ):
        raise ValueError("support-headroom one-sided alpha must lie in (0, 0.5)")
    if (
        isinstance(minimum_stratum_coverage, bool)
        or not isinstance(minimum_stratum_coverage, (int, float))
        or not math.isfinite(float(minimum_stratum_coverage))
        or not 0.0 < float(minimum_stratum_coverage) <= 1.0
    ):
        raise ValueError("minimum support-headroom stratum coverage must lie in (0, 1]")
    if (
        isinstance(evaluator_reads, bool)
        or not isinstance(evaluator_reads, int)
        or evaluator_reads <= 0
    ):
        raise ValueError("support-headroom evaluator reads must be a positive integer")

    threshold = float(minimum_headroom)
    alpha = float(one_sided_alpha)
    coverage_threshold = float(minimum_stratum_coverage)
    mean = float(values.mean())
    between_lineage_standard_error = (
        float(values.std(ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None
    )
    evaluator_standard_error = math.sqrt(
        SUPPORT_HEADROOM_BERNOULLI_DIFFERENCE_VARIANCE_BOUND / (evaluator_reads * len(values))
    )
    standard_error = (
        math.sqrt(between_lineage_standard_error**2 + evaluator_standard_error**2)
        if between_lineage_standard_error is not None
        else None
    )
    quantile = float(NormalDist().inv_cdf(1.0 - alpha))
    lower_bound = mean - quantile * standard_error if standard_error is not None else None

    per_stratum: dict[str, dict[str, object]] = {}
    for stratum in sorted(set(strata)):
        selected = values[np.asarray([row == stratum for row in strata], dtype=bool)]
        successes = int(np.count_nonzero(selected >= threshold))
        coverage = successes / len(selected)
        per_stratum[stratum] = {
            "eligible_instances": int(len(selected)),
            "instances_meeting_minimum_headroom": successes,
            "coverage": coverage,
            "mean_supported_headroom_over_initial": float(selected.mean()),
            "pass": bool(coverage >= coverage_threshold),
        }
    all_strata_pass = all(row["pass"] is True for row in per_stratum.values())
    return {
        "protocol": SUPPORT_HEADROOM_INFERENCE_PROTOCOL,
        "independent_unit": "one-selected-task-per-immutable-base-lineage",
        "eligible_instances": int(len(values)),
        "mean_supported_headroom_over_initial": mean,
        "supported_headroom_se": standard_error,
        "between_lineage_standard_error": between_lineage_standard_error,
        "evaluator_standard_error_upper_bound": evaluator_standard_error,
        "evaluator_reads_per_confirmation_block": evaluator_reads,
        "evaluator_variance_upper_bound_per_lineage": (
            SUPPORT_HEADROOM_BERNOULLI_DIFFERENCE_VARIANCE_BOUND / evaluator_reads
        ),
        "confirmation_protocol": "select-on-screening-block-rescore-on-independent-block-v1",
        "headroom_one_sided_lower_confidence_bound": lower_bound,
        "normal_quantile": quantile,
        "one_sided_alpha": alpha,
        "minimum_mean_headroom_over_initial": threshold,
        "instances_meeting_minimum_headroom": int(np.count_nonzero(values >= threshold)),
        "overall_instance_coverage": float(np.mean(values >= threshold)),
        "minimum_per_stratum_coverage": coverage_threshold,
        "per_stratum_headroom": per_stratum,
        "all_strata_meet_minimum_coverage": all_strata_pass,
        "pass": bool(lower_bound is not None and lower_bound >= threshold and all_strata_pass),
        "note": (
            "The one-sided bound combines sampled-lineage variation with a conservative "
            "finite-read evaluator term. It screens frozen observed support and is not an "
            "optimal-embedding claim."
        ),
    }


def _pinned_resolution_plan(
    plan: QualityResolutionPlan,
    expected_plan_sha256: str,
) -> tuple[dict[str, Any], str]:
    if not isinstance(plan, QualityResolutionPlan):
        raise TypeError("support audit requires a QualityResolutionPlan")
    if (
        not isinstance(expected_plan_sha256, str)
        or len(expected_plan_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_plan_sha256)
    ):
        raise ValueError("support audit requires a lowercase resolution-plan SHA-256 pin")
    record = plan.as_dict()
    observed = hashlib.sha256(canonical_json_bytes(record) + b"\n").hexdigest()
    if not hmac.compare_digest(observed, expected_plan_sha256):
        raise ValueError("support audit resolution plan differs from its external pin")
    if (
        record.get("schema") != QUALITY_RESOLUTION_PLAN_SCHEMA
        or record.get("schema_version") != QUALITY_RESOLUTION_PLAN_VERSION
    ):
        raise ValueError("support audit requires the current quality-resolution plan")
    recorded_digest = record.get("record_digest")
    unsigned = {key: value for key, value in record.items() if key != "record_digest"}
    if not isinstance(recorded_digest, str) or not hmac.compare_digest(
        recorded_digest, content_digest(unsigned)
    ):
        raise ValueError("support audit resolution-plan record digest mismatch")
    return record, observed


def _target_free_train_task(item: PreparedTask) -> None:
    if not isinstance(item, PreparedTask):
        raise TypeError("support audit public corpus must contain PreparedTask records")
    if (
        item.partition != "train"
        or item.prepared_schema_version != 4
        or item.corpus_scope != "production-designed-v4"
    ):
        raise ValueError("support audit requires target-free production train tasks")
    if item.task.name != item.task_id or not item.task.lineage:
        raise ValueError("support audit prepared task identity is malformed")
    if item.provenance is None or item.design_condition is None:
        raise ValueError("support audit train task lacks authenticated provenance")
    if (
        item.provenance.base_parent_lineage != item.task.lineage
        or item.design_condition.base_lineage_key != item.task.lineage
        or item.design_condition.learning_partition != "train"
    ):
        raise ValueError("support audit task differs from its train lineage authority")
    evaluator_fields = (
        item.task.ground_energy,
        item.reference_status,
        item.certificate_digest,
        item.evaluator_protocol_digest,
        item.quality_attestation_digest,
        item.quality_evidence_manifest_digest,
        item.quality_evidence_manifest_sha256,
        item.quality_target_set_digest,
        item.quality_target_count,
    )
    if any(value is not None for value in evaluator_fields):
        raise ValueError("support audit opened evaluator data before target-free replay")


def _support_family(opcode: str, affected_count: int) -> str:
    if opcode == "REWRITE_GROUP":
        return f"REWRITE_GROUP:{affected_count}"
    return opcode


def gate_authenticated_k2_support_coverage(
    plan: QualityResolutionPlan,
    *,
    expected_plan_sha256: str,
    public_prepared: Sequence[PreparedTask],
    context: Context,
    selector: StrengthSelector,
    initializer_bank: InitializerSnapshotBank | None,
    expected_initializer_bank_manifest_sha256: str | None,
    allow_test_initializer_bank: bool = False,
) -> dict[str, object]:
    """Replay every sampled plan row and audit its complete legal candidate support.

    This target-free Gate 2 conjunct reuses the frozen 128-lineage resolution sample. It
    reconstructs each row from the exact persistent K=2 initializer episode recorded by the
    plan, then audits every legal candidate in the materialized support. No positional prefix,
    score, or outcome may select a smaller candidate subset.
    """

    record, plan_sha256 = _pinned_resolution_plan(plan, expected_plan_sha256)
    if not isinstance(context, Context):
        raise TypeError("support audit requires a Context")
    if initializer_bank is None or expected_initializer_bank_manifest_sha256 is None:
        raise ValueError("support audit requires a pinned persistent K=2 initializer bank")
    production = record.get("production_plan")
    sample = record.get("sample")
    prepared_identity = record.get("prepared_corpus")
    if not all(isinstance(section, Mapping) for section in (production, sample, prepared_identity)):
        raise ValueError("support audit resolution-plan sections are malformed")
    planned_bank = production.get("initializer_bank")
    if not isinstance(planned_bank, Mapping):
        raise ValueError("support audit plan has no initializer-bank contract")
    try:
        live_bank = quality_initializer_bank_contract(
            initializer_bank,
            expected_manifest_sha256=expected_initializer_bank_manifest_sha256,
            allow_test_bank=allow_test_initializer_bank,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(str(error)) from error
    if live_bank != dict(planned_bank):
        raise ValueError("support audit initializer bank differs from the resolution plan")
    if (
        live_bank.get("protocol") != QUALITY_INITIALIZER_BANK_PROTOCOL
        or live_bank.get("restart_cache_slots_per_episode") != 2
        or live_bank.get("partition") != "train"
        or live_bank.get("opened_evaluator_data") is not False
    ):
        raise ValueError("support audit initializer bank is not target-free persistent K=2")

    if isinstance(public_prepared, (str, bytes)) or not isinstance(public_prepared, Sequence):
        raise TypeError("support audit public corpus must be a sequence")
    public_by_task: dict[str, PreparedTask] = {}
    for item in public_prepared:
        _target_free_train_task(item)
        if item.task_id in public_by_task:
            raise ValueError("support audit public corpus repeats a task ID")
        public_by_task[item.task_id] = item

    source_census = production.get("source_census")
    census_lineages = source_census.get("lineages") if isinstance(source_census, Mapping) else None
    if not isinstance(census_lineages, list) or not census_lineages:
        raise ValueError("support audit plan has no complete train census")
    expected_tasks: set[str] = set()
    for lineage in census_lineages:
        task_ids = lineage.get("task_ids") if isinstance(lineage, Mapping) else None
        if (
            not isinstance(task_ids, list)
            or not task_ids
            or any(not isinstance(task_id, str) or not task_id for task_id in task_ids)
        ):
            raise ValueError("support audit train census is malformed")
        expected_tasks.update(task_ids)
    if set(public_by_task) != expected_tasks:
        raise ValueError("support audit public corpus differs from the plan train census")
    if prepared_identity.get("manifest_sha256") != live_bank.get(
        "prepared_manifest_sha256"
    ) or source_census.get("prepared_manifest_sha256") != live_bank.get("prepared_manifest_sha256"):
        raise ValueError("support audit plan and bank use different prepared corpora")

    selected_lineages = sample.get("selected_lineages")
    selected_row_ids = sample.get("selected_row_ids")
    if (
        not isinstance(selected_lineages, list)
        or len(selected_lineages) < SUPPORT_FULL_AUDIT_MINIMUM_LINEAGES
        or len(set(selected_lineages)) != len(selected_lineages)
        or any(not isinstance(lineage, str) or not lineage for lineage in selected_lineages)
    ):
        raise ValueError(
            "support audit requires at least "
            f"{SUPPORT_FULL_AUDIT_MINIMUM_LINEAGES} independent train lineages"
        )
    if (
        not isinstance(selected_row_ids, list)
        or not selected_row_ids
        or len(set(selected_row_ids)) != len(selected_row_ids)
        or any(not isinstance(row_id, str) or not row_id for row_id in selected_row_ids)
    ):
        raise ValueError("support audit selected-row registry is malformed")
    rows = production.get("rows")
    if not isinstance(rows, list):
        raise ValueError("support audit production rows are malformed")
    selected_id_set = set(selected_row_ids)
    selected_rows = [
        row for row in rows if isinstance(row, Mapping) and row.get("row_id") in selected_id_set
    ]
    if {row.get("row_id") for row in selected_rows} != selected_id_set:
        raise ValueError("support audit plan omits a selected production row")
    sampled_lineage_set = set(selected_lineages)
    row_lineages = {row.get("base_lineage") for row in selected_rows}
    if row_lineages != sampled_lineage_set:
        raise ValueError("support audit requires at least one row for every sampled lineage")

    opcode_counts: Counter[str] = Counter()
    legal_opcode_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    audited_family_counts: Counter[str] = Counter()
    audit_rows: list[dict[str, object]] = []
    for row in sorted(selected_rows, key=lambda value: str(value["row_id"])):
        task_id = row.get("task_id")
        item = public_by_task.get(task_id) if isinstance(task_id, str) else None
        if item is None or item.instance_id != row.get("instance_id"):
            raise ValueError("support audit row differs from its public prepared task")
        episode_index = row.get("initializer_bank_episode_index")
        environment_seed = row.get("environment_seed")
        prefix = row.get("prefix")
        provenance = row.get("action_provenance_fingerprint")
        if (
            type(episode_index) is not int
            or type(environment_seed) is not int
            or not isinstance(prefix, list)
            or any(type(index) is not int or index < 0 for index in prefix)
            or not isinstance(provenance, str)
            or not provenance
        ):
            raise ValueError("support audit row replay identity is malformed")
        replayed = replay_decision_with_action_envelope(
            item.task,
            context,
            initializer=None,
            initializer_bank=initializer_bank,
            expected_initializer_bank_manifest_sha256=(expected_initializer_bank_manifest_sha256),
            initializer_bank_episode_index=episode_index,
            allow_test_initializer_bank=allow_test_initializer_bank,
            selector=selector,
            prefix=tuple(prefix),
            seed=environment_seed,
            reward_reads=context.n_est_reads,
            provenance_fingerprint=provenance,
        )
        if replayed is None:
            raise ValueError("support audit could not replay a selected production row")
        decision, envelope = replayed
        if (
            envelope.record_digest != row.get("action_envelope_record_digest")
            or decision.state_fingerprint != row.get("state_fingerprint")
            or decision.support_fingerprint != row.get("support_fingerprint")
            or len(envelope.candidates) != len(decision.candidates)
        ):
            raise ValueError("support audit live row differs from its sealed plan")
        for planned_action in row.get("actions", ()):  # subset used by Q labels
            if not isinstance(planned_action, Mapping):
                raise ValueError("support audit planned action registry is malformed")
            action_index = planned_action.get("action_index")
            if type(action_index) is not int or not 0 <= action_index < len(envelope.candidates):
                raise ValueError("support audit planned action leaves the live support")
            bound = envelope.candidates[action_index]
            if (
                bound.payload_digest != planned_action.get("selected_payload_digest")
                or bound.payload_key != planned_action.get("payload_key")
                or bound.opcode.value != planned_action.get("opcode")
            ):
                raise ValueError("support audit planned action differs from live support")

        legal_indices: list[int] = []
        row_opcode_counts = Counter[str]()
        row_legal_opcode_counts = Counter[str]()
        row_family_counts = Counter[str]()
        row_audited_family_counts = Counter[str]()
        for index, bound in enumerate(envelope.candidates):
            if bound.index != index or bound.legal is not decision.legal_mask[index]:
                raise ValueError("support audit envelope positions differ from the legal mask")
            opcode = bound.opcode.value
            if opcode not in OPCODES:
                raise ValueError("support audit encountered an unregistered opcode")
            family = _support_family(opcode, len(bound.affected))
            opcode_counts[opcode] += 1
            row_opcode_counts[opcode] += 1
            family_counts[family] += 1
            row_family_counts[family] += 1
            if bound.legal:
                legal_indices.append(index)
                legal_opcode_counts[opcode] += 1
                row_legal_opcode_counts[opcode] += 1
                audited_family_counts[family] += 1
                row_audited_family_counts[family] += 1
        if not legal_indices:
            raise ValueError("support audit selected row has no legal candidate")
        row_payload: dict[str, object] = {
            "audited_indices": legal_indices,
            "audited_legal_candidate_count": len(legal_indices),
            "base_lineage": row["base_lineage"],
            "candidate_count": len(envelope.candidates),
            "candidate_payload_root": stable_digest(
                [candidate.payload_digest for candidate in envelope.candidates]
            ),
            "family_counts": dict(sorted(row_family_counts.items())),
            "initializer_bank_episode_index": episode_index,
            "initializer_bootstrap_record_digest": row["initializer_bootstrap_record_digest"],
            "instance_id": row["instance_id"],
            "legal_candidate_count": len(legal_indices),
            "legal_family_counts": dict(sorted(row_audited_family_counts.items())),
            "legal_indices": legal_indices,
            "legal_opcode_counts": {opcode: row_legal_opcode_counts[opcode] for opcode in OPCODES},
            "opcode_counts": {opcode: row_opcode_counts[opcode] for opcode in OPCODES},
            "row_id": row["row_id"],
            "state_fingerprint": row["state_fingerprint"],
            "support_fingerprint": row["support_fingerprint"],
            "task_id": row["task_id"],
        }
        audit_rows.append({**row_payload, "record_digest": content_digest(row_payload)})

    configured_group_families = tuple(
        f"REWRITE_GROUP:{size}"
        for size in context.group_sizes
        if int(context.quotas.get(f"group{size}", 0)) > 0
        and any(
            public_by_task[str(row["task_id"])].task.logical.number_of_nodes() >= size
            for row in selected_rows
        )
    )
    required_families = (
        *configured_group_families,
        "REPAIR_GROUP",
        "RESTART",
    )
    missing_families = tuple(
        family for family in required_families if audited_family_counts[family] == 0
    )
    payload: dict[str, object] = {
        "audited_candidate_count": sum(
            int(row["audited_legal_candidate_count"]) for row in audit_rows
        ),
        "candidate_count": sum(int(row["candidate_count"]) for row in audit_rows),
        "candidate_selection": "all-legal-materialized-candidates-no-truncation",
        "family_counts": dict(sorted(family_counts.items())),
        "initializer_bank": live_bank,
        "legal_candidate_count": sum(int(row["legal_candidate_count"]) for row in audit_rows),
        "legal_family_counts": dict(sorted(audited_family_counts.items())),
        "legal_opcode_counts": {opcode: legal_opcode_counts[opcode] for opcode in OPCODES},
        "minimum_independent_lineages": SUPPORT_FULL_AUDIT_MINIMUM_LINEAGES,
        "missing_required_families": list(missing_families),
        "opcode_counts": {opcode: opcode_counts[opcode] for opcode in OPCODES},
        "pass": not missing_families,
        "plan_record_digest": record["record_digest"],
        "plan_sha256": plan_sha256,
        "protocol": SUPPORT_FULL_AUDIT_PROTOCOL,
        "required_families": list(required_families),
        "rows": audit_rows,
        "sampled_independent_lineages": len(selected_lineages),
        "sampled_lineages_digest": content_digest(sorted(selected_lineages)),
        "schema": "isingfold.gate2-authenticated-full-support",
        "schema_version": 1,
        "target_accessed": False,
    }
    if payload["audited_candidate_count"] != payload["legal_candidate_count"]:
        raise RuntimeError("support audit silently dropped a legal candidate")
    return {**payload, "record_digest": content_digest(payload)}


def gate_support_headroom(
    tasks: Sequence[EmbeddingTask],
    ctx: Context,
    *,
    selector: StrengthSelector | None = None,
    reads: int = 128,
    seed: int = 0,
    meaningful_effect: float = 0.02,
    broad_reference_batches: int = 8,
    min_feasibility_recall: float = 0.5,
    max_broad_pool_advantage: float = 0.05,
    stratum_ids: Sequence[str] | None = None,
    headroom_one_sided_alpha: float = SUPPORT_HEADROOM_ONE_SIDED_ALPHA,
    min_stratum_headroom_coverage: float = SUPPORT_HEADROOM_MINIMUM_STRATUM_COVERAGE,
) -> dict[str, object]:
    """Gate 2: compare one policy support with a frozen broader observed proposal pool.

    The broader pool is the union of independently seeded batches from the same frozen,
    outcome-blind proposal generator. Every legal workspace-changing candidate in every batch is
    included; support order never acts as a hidden candidate limit. Candidate choice uses a
    screening block; headroom uses independent confirmation blocks for the selected rewrite and
    protected COMMIT. This work is an offline diagnostic budget and is never credited to a policy.
    Consequently this is an empirical pool audit, not a global feasibility recall or
    optimal-quality claim.
    """

    if broad_reference_batches < 2:
        raise ValueError("broad reference requires at least two frozen proposal batches")
    if stratum_ids is not None and len(stratum_ids) != len(tasks):
        raise ValueError("support audit stratum IDs must align one-to-one with tasks")

    selector = selector or fixed_strength_selector()
    initializer = router_initializer()
    gaps: list[float] = []
    spreads: list[float] = []
    recalls: list[float] = []
    headrooms: list[float] = []
    candidate_attempts = 0
    evaluator_calls = 0
    headroom_confirmation_blocks = 0
    eligible_strata: list[str] = []
    legal_workspace_opcode_counts: Counter[str] = Counter()
    for k, task in enumerate(tasks):
        registered: dict[str, object] = {}
        broader: dict[str, object] = {}
        start = None
        for batch in range(broad_reference_batches):
            batch_seed = seed + 1_000_003 * k + 7_919 * batch
            env = EmbeddingEnv(
                task,
                ctx,
                mode=Mode.IMPROVEMENT,
                initializer=task_initializer(task, initializer),
                selector=selector,
                reward_reads=reads,
                improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
                seed=batch_seed,
            )
            decision = env.reset(batch_seed)
            if not isinstance(decision, DecisionState) or env.state is None:
                continue
            if start is None:
                start = dict(env.state.archive[0].chains)
            state_candidates = [
                candidate
                for candidate, legal in zip(decision.candidates, decision.legal_mask, strict=True)
                if legal and candidate.changes_workspace
            ]
            candidate_attempts += len(state_candidates)
            for candidate in state_candidates:
                legal_workspace_opcode_counts[candidate.opcode.value] += 1
                successor = dict(env.state.chains)
                successor.update(candidate.new_chains)
                receipt, _ = p_return(successor, task.logical, task.host, task.problem, ctx)
                if not receipt.valid:
                    continue
                key = chain_key(successor)
                broader.setdefault(key, successor)
                if batch == 0:
                    registered.setdefault(key, successor)
        if start is None or not registered or not broader:
            continue
        if task.ground_energy is None:
            raise ValueError("support-headroom gate requires certified evaluator targets")
        screening_cache: dict[str, float] = {}

        def score(successor, *, confirmation: bool = False) -> float:
            nonlocal evaluator_calls
            key = chain_key(successor)
            if not confirmation and key in screening_cache:
                return screening_cache[key]
            receipt, programs = p_return(successor, task.logical, task.host, task.problem, ctx)
            if not receipt.valid:
                raise RuntimeError("broad-reference candidate changed validity before scoring")
            features = [program_features(p, successor, task.problem) for p in programs]
            index = _select_strength(selector, task, ctx, successor, programs, features)
            evaluator_seed = (
                seed + 1_000_003 * k + (1_610_612_741 if confirmation else 0) + int(key[:16], 16)
            ) % (2**31 - 1)
            block = sample_program(
                programs[index],
                successor,
                task.problem,
                task.ground_energy,
                num_reads=reads,
                seed=evaluator_seed,
                num_sweeps=ctx.num_sweeps,
            )
            evaluator_calls += 1
            if not confirmation:
                screening_cache[key] = block.rate
            return block.rate

        supported_scores = [(successor, score(successor)) for successor in registered.values()]
        supported_values = [value for _, value in supported_scores]
        broad_values = [score(successor) for successor in broader.values()]
        selected_successor, _ = min(
            supported_scores,
            key=lambda row: (-row[1], chain_key(row[0])),
        )
        confirmed_supported = score(selected_successor, confirmation=True)
        confirmed_base = score(start, confirmation=True)
        headroom_confirmation_blocks += 2
        recalls.append(len(set(registered) & set(broader)) / len(broader))
        gaps.append(max(broad_values) - max(supported_values))
        spreads.append(max(supported_values) - min(supported_values))
        headrooms.append(confirmed_supported - confirmed_base)
        eligible_strata.append(stratum_ids[k] if stratum_ids is not None else "unstratified")
    if not gaps:
        return {
            "signal_name": "frozen-broad-reference-support-audit",
            "instances_requested": len(tasks),
            "eligible_instances": 0,
            "pass": False,
            "reason": "no instance had both registered and broad valid successors",
        }

    def mean_se(values: Sequence[float]) -> tuple[float, float]:
        array = np.asarray(values, dtype=float)
        standard_error = float(array.std(ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
        return float(array.mean()), standard_error

    mean_recall, recall_se = mean_se(recalls)
    mean_gap, gap_se = mean_se(gaps)
    mean_spread, spread_se = mean_se(spreads)
    headroom_inference = support_headroom_inference(
        headrooms,
        eligible_strata,
        minimum_headroom=meaningful_effect,
        one_sided_alpha=headroom_one_sided_alpha,
        minimum_stratum_coverage=min_stratum_headroom_coverage,
        evaluator_reads=reads,
    )
    return {
        "signal_name": "frozen-broad-reference-support-audit",
        "instances_requested": len(tasks),
        "eligible_instances": len(gaps),
        "broad_reference_batches_per_instance": broad_reference_batches,
        "candidate_selection": "all-legal-materialized-workspace-candidates-no-truncation",
        "legal_workspace_opcode_counts": {
            opcode: legal_workspace_opcode_counts[opcode] for opcode in OPCODES
        },
        "offline_candidate_attempts": candidate_attempts,
        "offline_evaluator_calls": evaluator_calls,
        "offline_evaluator_reads": evaluator_calls * reads,
        "headroom_confirmation_blocks": headroom_confirmation_blocks,
        "mean_feasibility_support_recall": mean_recall,
        "feasibility_support_recall_se": recall_se,
        "mean_broad_pool_quality_advantage": mean_gap,
        "broad_pool_quality_advantage_se": gap_se,
        "mean_within_support_quality_spread": mean_spread,
        "within_support_quality_spread_se": spread_se,
        "mean_supported_headroom_over_initial": headroom_inference[
            "mean_supported_headroom_over_initial"
        ],
        "supported_headroom_se": headroom_inference["supported_headroom_se"],
        "headroom_inference": headroom_inference,
        "thresholds": {
            "min_feasibility_support_recall": min_feasibility_recall,
            "min_within_support_quality_spread": meaningful_effect,
            "max_broad_pool_quality_advantage": max_broad_pool_advantage,
            "min_mean_headroom_over_initial": meaningful_effect,
            "headroom_one_sided_alpha": headroom_one_sided_alpha,
            "min_per_stratum_headroom_coverage": min_stratum_headroom_coverage,
        },
        "pass": bool(
            len(tasks) > 0
            and len(gaps) == len(tasks)
            and mean_recall >= min_feasibility_recall
            and mean_spread >= meaningful_effect
            and mean_gap <= max_broad_pool_advantage
            and headroom_inference["pass"] is True
        ),
        "note": (
            "Headroom is independently confirmed after support screening. Recall and quality "
            "gap remain relative to a frozen observed proposal pool, not global feasibility or "
            "an optimal embedding."
        ),
    }


def gate_selector_discrimination(
    selector_labels: str | Path,
    selector: object,
    *,
    expected_source_manifest_sha256: str,
    expected_context_digest: str,
) -> dict[str, object]:
    """Gate 3: assess the frozen graph selector on training-side calibration labels."""

    metadata = load_selector_metadata(selector_labels)
    if metadata.source_prepared_manifest_sha256 != expected_source_manifest_sha256:
        raise ValueError("selector labels belong to a different prepared corpus")
    if metadata.context_digest != expected_context_digest:
        raise ValueError("selector-label context differs from the gate context")
    if not all(
        hasattr(selector, name)
        for name in (
            "select_embedding",
            "deployment_ready",
            "normalizer_digest",
            "coefficient_transform_scale",
        )
    ):
        raise ValueError("selector-discrimination gate requires a graph selector")
    if selector.deployment_ready is not True:
        raise ValueError("selector-discrimination gate requires a deployment-ready selector")
    if selector.normalizer_digest != metadata.normalizer_digest:
        raise ValueError("selector and selector-label normalizer identities differ")
    selector_scale = _selector_scalar(
        selector.coefficient_transform_scale, "coefficient transform scale"
    )
    if not math.isclose(selector_scale, metadata.coefficient_scale, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("selector and selector-label coefficient scales differ")

    records = load_selector_records(selector_labels, partition="calibration")
    if not records:
        return {
            "records": 0,
            "partition": "calibration",
            "selector_label_manifest_sha256": metadata.manifest_sha256,
            "pass": False,
            "reason": "empty selector calibration denominator",
        }
    rates = np.asarray(
        [[h / max(1, n) for h, n in zip(r.hits, r.reads, strict=True)] for r in records]
    )
    spread = float(np.mean(rates.max(axis=1) - rates.min(axis=1))) if len(rates) else 0.0
    report: dict[str, object] = {
        "records": len(records),
        "partition": "calibration",
        "mean_strength_spread": spread,
        "selector_label_manifest_sha256": metadata.manifest_sha256,
    }
    report.update(selector_regret(selector, records))
    report["random_estimator"] = "exact-uniform-four-strength-expectation"
    report["pass"] = bool(
        spread >= 0.02
        and report["selected_mean"] >= report["random_mean"]
        and report["selected_mean"] >= report["fixed_f2_mean"]
    )
    return report


def _run_valid_return_profile(
    tasks: Sequence[EmbeddingTask],
    ctx: Context,
    *,
    mode: Mode,
    selector: StrengthSelector | None = None,
    reads: int = 128,
    seed: int = 0,
    episodes: int = 2,
) -> dict[str, object]:
    """Run one validity profile without assigning release-gate semantics to it."""

    selector = selector or fixed_strength_selector()
    initializer = router_initializer()
    expected = len(tasks) * episodes
    rewards: list[float] = []
    valid = 0
    init_failures = 0
    continued = 0
    completed = 0
    for k, task in enumerate(tasks):
        for episode in range(episodes):
            episode_seed = (
                seed + (0 if mode is Mode.IMPROVEMENT else 10_000_019) + 101 * k + episode
            )
            env = EmbeddingEnv(
                task,
                ctx,
                mode=mode,
                initializer=(
                    task_initializer(task, initializer) if mode is Mode.IMPROVEMENT else None
                ),
                selector=selector,
                reward_reads=reads,
                improvement_restart_protocol=LEGACY_ONLINE_INITIALIZER_RESTARTS_V1,
                seed=episode_seed,
            )
            result = env.reset(episode_seed)
            if isinstance(result, InitFailureRecord):
                init_failures += 1
                continue
            rng = random.Random(episode_seed)
            steps = 0
            changed_workspace = False
            while isinstance(result, DecisionState) and steps < ctx.caps.decisions:
                legal = [index for index, legal in enumerate(result.legal_mask) if legal]
                if not legal:
                    break
                chosen = rng.choice(legal)
                changed_workspace = changed_workspace or result.candidates[chosen].changes_workspace
                result = env.step(result, chosen).next_decision_or_terminal
                steps += 1
            if changed_workspace:
                continued += 1
            if not isinstance(result, TerminalRecord):
                continue
            completed += 1
            reward = float(result.training_reward or 0.0) if result.returned_valid else 0.0
            rewards.append(reward)
            valid += int(result.returned_valid)
    padded_rewards = rewards + [0.0] * (expected - len(rewards))
    return {
        "mode": mode.value,
        "empty_start": mode is Mode.CONSTRUCTION,
        "attempted_episodes": expected,
        "completed_episodes": completed,
        "initialization_failures": init_failures,
        "valid_returns": valid,
        "valid_return_rate": valid / expected if expected else 0.0,
        "continuation_episode_share": continued / expected if expected else 0.0,
        "nonzero_utility_share": (
            float(np.mean([reward > 0.0 for reward in padded_rewards])) if padded_rewards else 0.0
        ),
        "unconditional_mean_utility": (float(np.mean(padded_rewards)) if padded_rewards else 0.0),
    }


def gate_profile_i_signal(
    tasks: Sequence[EmbeddingTask],
    ctx: Context,
    *,
    selector: StrengthSelector | None = None,
    reads: int = 128,
    seed: int = 0,
    episodes: int = 2,
    min_valid_rate: float = 0.9,
    min_continuation_share: float = 0.1,
    min_nonzero_utility_share: float = 0.1,
) -> dict[str, object]:
    """Mandatory Gate 4 for the first Profile-I representation screen."""

    report = _run_valid_return_profile(
        tasks,
        ctx,
        mode=Mode.IMPROVEMENT,
        selector=selector,
        reads=reads,
        seed=seed,
        episodes=episodes,
    )
    expected = len(tasks) * episodes
    report["thresholds"] = {
        "min_valid_return_rate": min_valid_rate,
        "min_continuation_episode_share": min_continuation_share,
        "min_nonzero_utility_share": min_nonzero_utility_share,
    }
    report["pass"] = bool(
        expected > 0
        and report["completed_episodes"] == expected
        and report["valid_return_rate"] >= min_valid_rate
        and report["continuation_episode_share"] >= min_continuation_share
        and report["nonzero_utility_share"] >= min_nonzero_utility_share
    )
    report["note"] = (
        "Mandatory Profile-I signal gate: valid fallback, actual continuation, and "
        "nonzero terminal utility are measured after successful initialization."
    )
    return report


def gate_profile_c_readiness(
    tasks: Sequence[EmbeddingTask],
    ctx: Context,
    *,
    selector: StrengthSelector | None = None,
    reads: int = 128,
    seed: int = 0,
    episodes: int = 2,
    min_valid_return_rate: float = 0.1,
) -> dict[str, object]:
    """Optional construction-readiness diagnostic; it never gates Profile-I by default."""

    report = _run_valid_return_profile(
        tasks,
        ctx,
        mode=Mode.CONSTRUCTION,
        selector=selector,
        reads=reads,
        seed=seed,
        episodes=episodes,
    )
    expected = len(tasks) * episodes
    report["thresholds"] = {"min_empty_start_valid_return_rate": min_valid_return_rate}
    report["pass"] = bool(
        expected > 0
        and report["completed_episodes"] == expected
        and report["valid_return_rate"] >= min_valid_return_rate
    )
    report["note"] = (
        "Experimental Profile-C starts with empty chains and no protected archive. "
        "This result gates construction experiments only."
    )
    return report


def gate_valid_returns(
    tasks: Sequence[EmbeddingTask],
    ctx: Context,
    *,
    selector: StrengthSelector | None = None,
    reads: int = 128,
    seed: int = 0,
    episodes: int = 2,
    min_profile_i_valid_rate: float = 0.9,
    min_profile_i_continuation_share: float = 0.1,
    min_profile_i_nonzero_utility_share: float = 0.1,
    min_profile_c_valid_return_rate: float = 0.1,
) -> dict[str, object]:
    """Compatibility report with Profile-I and Profile-C decisions kept independent."""

    profile_i = gate_profile_i_signal(
        tasks,
        ctx,
        selector=selector,
        reads=reads,
        seed=seed,
        episodes=episodes,
        min_valid_rate=min_profile_i_valid_rate,
        min_continuation_share=min_profile_i_continuation_share,
        min_nonzero_utility_share=min_profile_i_nonzero_utility_share,
    )
    profile_c = gate_profile_c_readiness(
        tasks,
        ctx,
        selector=selector,
        reads=reads,
        seed=seed,
        episodes=episodes,
        min_valid_return_rate=min_profile_c_valid_return_rate,
    )
    return {
        "profile_i": profile_i,
        "profile_c": profile_c,
        "pass": profile_i["pass"],
        "construction_ready": profile_c["pass"],
        "note": (
            "Profile-I is the mandatory first-stage decision. Profile-C is separately "
            "reported and becomes binding only for an explicitly requested construction stage."
        ),
    }
