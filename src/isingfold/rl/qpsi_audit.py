"""Train-only, fail-closed diagnostic for cross-embedding q_psi ordering.

This module does not train, calibrate, select a checkpoint, or authorize an archive
policy.  It asks one narrower question: among distinct valid terminal embeddings already
present in authenticated quality-continuation evidence for the same opportunity, does the
frozen IF-Q3-S0 predictor rank the empirically better endpoint higher?
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import tempfile
from collections.abc import Hashable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import torch

from isingfold.rl.checkpoint import runtime_implementation_registry
from isingfold.rl.contracts import Context, stable_digest
from isingfold.rl.data.import_embedbench import canonical_json_bytes, content_digest
from isingfold.rl.data.prepared import PreparedTask
from isingfold.rl.data.quality import (
    _canonical_program,
    _encoded_chains,
    _typed_identity,
    validate_continuation_receipt,
)
from isingfold.rl.program import Program
from isingfold.rl.strength import N_STRENGTHS
from isingfold.rl.strength_tensorize import build_strength_inputs
from isingfold.rl.validate import p_return

AUDIT_SCHEMA = "isingfold.qpsi-cross-embedding-audit"
AUDIT_ROW_SCHEMA = "isingfold.qpsi-cross-embedding-opportunity"
AUDIT_SCHEMA_VERSION = 1
MINIMUM_BOOTSTRAP_REPLICATES = 1_000
MINIMUM_CROSS_EMBEDDING_CONTRACT = (
    "minimum cross-embedding contract: each comparable train opportunity must carry "
    "at least two distinct valid terminal embeddings, all four canonical compiled "
    "programs, the frozen selector choice, independent evaluator count blocks, immutable "
    "task/instance/lineage identities, and outcome-blind receipt emission order"
)
_METRICS = (
    "spearman_rho",
    "pairwise_sign_accuracy",
    "qpsi_top1_regret",
    "qpsi_top_minus_last",
    "qpsi_top_beats_last_rate",
)
_SOURCE_KEYS = frozenset(
    {"evaluator", "program", "qpsi_audit", "selector", "selector_tensorizer"}
)


def _with_digest(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {**payload, "record_digest": content_digest(payload)}


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite_probabilities(values: object) -> tuple[float, ...]:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if array.shape != (N_STRENGTHS,) or not np.isfinite(array).all():
        raise ValueError("frozen selector must return four finite probabilities")
    result = tuple(float(value) for value in array)
    if any(value < 0.0 or value > 1.0 for value in result):
        raise ValueError("frozen selector probabilities must lie in [0, 1]")
    return result


def _scalar(value: object, *, name: str) -> float:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        value = value.detach().cpu().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _decode_embedding(
    encoded: object,
    prepared: PreparedTask,
) -> dict[Hashable, frozenset[Hashable]]:
    logical_by_identity = {
        _typed_identity(node): node for node in prepared.task.logical.nodes()
    }
    host_by_identity = {_typed_identity(qubit): qubit for qubit in prepared.task.host.nodes()}
    if len(logical_by_identity) != prepared.task.logical.number_of_nodes() or len(
        host_by_identity
    ) != prepared.task.host.number_of_nodes():
        raise ValueError("typed node identities are not injective for this task")
    if not isinstance(encoded, list):
        raise ValueError("terminal embedding encoding is malformed")
    chains: dict[Hashable, frozenset[Hashable]] = {}
    for item in encoded:
        if not isinstance(item, Mapping) or set(item) != {"logical", "chain"}:
            raise ValueError("terminal embedding encoding is malformed")
        logical_key = item["logical"]
        chain_keys = item["chain"]
        if (
            not isinstance(logical_key, str)
            or logical_key not in logical_by_identity
            or not isinstance(chain_keys, list)
            or any(not isinstance(key, str) for key in chain_keys)
        ):
            raise ValueError("terminal embedding does not match the prepared task")
        if logical_by_identity[logical_key] in chains:
            raise ValueError("terminal embedding repeats a logical variable")
        try:
            chain = frozenset(host_by_identity[key] for key in chain_keys)
        except (KeyError, TypeError) as error:
            raise ValueError("terminal embedding leaves the prepared active host") from error
        if len(chain) != len(chain_keys):
            raise ValueError("terminal embedding repeats a physical qubit")
        chains[logical_by_identity[logical_key]] = chain
    if _encoded_chains(chains) != encoded:
        raise ValueError("terminal embedding is not canonical or task-faithful")
    return chains


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    cursor = 0
    while cursor < len(array):
        stop = cursor + 1
        while stop < len(array) and array[order[stop]] == array[order[cursor]]:
            stop += 1
        ranks[order[cursor:stop]] = (cursor + stop - 1) / 2.0 + 1.0
        cursor = stop
    return ranks


def _spearman(scores: Sequence[float], outcomes: Sequence[float]) -> float | None:
    if len(scores) < 2 or len(set(scores)) < 2 or len(set(outcomes)) < 2:
        return None
    score_ranks = _average_ranks(scores)
    outcome_ranks = _average_ranks(outcomes)
    value = float(np.corrcoef(score_ranks, outcome_ranks)[0, 1])
    return value if math.isfinite(value) else None


def _pairwise_counts(
    scores: Sequence[float], outcomes: Sequence[float]
) -> dict[str, int]:
    counts = {
        "correct": 0,
        "incorrect": 0,
        "score_ties": 0,
        "target_ties": 0,
        "comparable": 0,
    }
    for left, right in combinations(range(len(scores)), 2):
        score_sign = np.sign(scores[left] - scores[right])
        target_sign = np.sign(outcomes[left] - outcomes[right])
        if score_sign == 0:
            counts["score_ties"] += 1
        if target_sign == 0:
            counts["target_ties"] += 1
        if score_sign == 0 or target_sign == 0:
            continue
        counts["comparable"] += 1
        counts["correct" if score_sign == target_sign else "incorrect"] += 1
    return counts


def _selector_inputs_and_score(
    selector: object,
    prepared: PreparedTask,
    chains: Mapping[Hashable, frozenset[Hashable]],
    programs: Sequence[Program],
    ctx: Context,
) -> tuple[tuple[float, ...], int]:
    scale = _scalar(
        getattr(selector, "coefficient_transform_scale", None),
        name="selector coefficient transform scale",
    )
    normalizer = getattr(selector, "normalizer_digest", None)
    if not isinstance(normalizer, str) or not normalizer:
        raise ValueError("frozen selector omits its normalizer identity")
    inputs = build_strength_inputs(
        ctx=ctx,
        logical=prepared.task.logical,
        host=prepared.task.host,
        problem=prepared.task.problem,
        chains=chains,
        programs=programs,
        coef_scale=scale,
        normalizer_digest=normalizer,
    )
    probabilities = _finite_probabilities(selector(inputs))  # type: ignore[operator]
    return probabilities, int(np.argmax(np.asarray(probabilities)))


def _terminal_occurrence(
    *,
    selector: object,
    prepared: PreparedTask,
    receipt: Mapping[str, object],
    order: int,
    ctx: Context,
) -> dict[str, Any] | None:
    validate_continuation_receipt(receipt)
    if receipt["returned_valid"] is False:
        return None
    evidence = receipt["terminal_evidence"]
    if not isinstance(evidence, Mapping):  # Defensive after the normative validator.
        raise ValueError("valid quality continuation omitted terminal evidence")
    chains = _decode_embedding(evidence["embedding"], prepared)
    validation, programs = p_return(
        chains,
        prepared.task.logical,
        prepared.task.host,
        prepared.task.problem,
        ctx,
    )
    if not validation.valid:
        raise ValueError("terminal evidence embedding fails current return validation")
    if receipt["validation_receipt"] != validation.as_dict():
        raise ValueError("terminal evidence validation receipt differs from current validation")
    canonical_programs = [_canonical_program(program) for program in programs]
    if evidence["programs"] != canonical_programs:
        raise ValueError("terminal evidence compiled programs do not match the current compiler")
    selected = evidence["selected_index"]
    if type(selected) is not int or not 0 <= selected < N_STRENGTHS:
        raise ValueError("terminal evidence selected strength index is invalid")
    if not math.isclose(
        _scalar(evidence["selected_strength"], name="selected strength"),
        float(programs[selected].strength),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("terminal evidence selected strength differs from compilation")
    block = evidence["evaluator_count_block"]
    if not isinstance(block, Mapping) or block.get("num_sweeps") != ctx.num_sweeps:
        raise ValueError("terminal evaluator program protocol differs from the registered context")
    probabilities, predicted = _selector_inputs_and_score(
        selector, prepared, chains, programs, ctx
    )
    if predicted != selected:
        raise ValueError("terminal evidence differs from the frozen selector choice")
    embedding_digest = stable_digest({"embedding": evidence["embedding"]})
    return {
        "embedding": evidence["embedding"],
        "embedding_digest": embedding_digest,
        "programs_digest": stable_digest({"programs": canonical_programs}),
        "selector_probabilities": list(probabilities),
        "selected_strength_index": selected,
        "qpsi_score": max(probabilities),
        "hits": int(block["hits"]),
        "reads": int(block["reads"]),
        "evaluator_seed": int(block["seed"]),
        "order": order,
        "continuation_receipt_digest": receipt["record_digest"],
    }


def _deduplicate_occurrences(
    occurrences: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for occurrence in occurrences:
        grouped.setdefault(str(occurrence["embedding_digest"]), []).append(occurrence)
    terminals: list[dict[str, Any]] = []
    for embedding_digest, group in sorted(grouped.items()):
        first = group[0]
        identity_fields = (
            "embedding",
            "programs_digest",
            "selector_probabilities",
            "selected_strength_index",
            "qpsi_score",
        )
        if any(any(row[field] != first[field] for field in identity_fields) for row in group):
            raise ValueError("one terminal embedding has inconsistent program or selector evidence")
        hits = sum(int(row["hits"]) for row in group)
        reads = sum(int(row["reads"]) for row in group)
        terminals.append(
            {
                "embedding_digest": embedding_digest,
                "embedding": first["embedding"],
                "programs_digest": first["programs_digest"],
                "selector_probabilities": first["selector_probabilities"],
                "selected_strength_index": first["selected_strength_index"],
                "qpsi_score": first["qpsi_score"],
                "hits": hits,
                "reads": reads,
                "empirical_success_rate": hits / reads,
                "independent_count_blocks": len(group),
                "first_emission_order": min(int(row["order"]) for row in group),
                "last_emission_order": max(int(row["order"]) for row in group),
                "evaluator_seeds": sorted(int(row["evaluator_seed"]) for row in group),
                "continuation_receipt_digests": sorted(
                    str(row["continuation_receipt_digest"]) for row in group
                ),
            }
        )
    return tuple(terminals)


def _opportunity_row(
    *,
    source: Mapping[str, object],
    terminals: Sequence[Mapping[str, Any]],
    invalid_terminal_receipts: int,
) -> dict[str, Any]:
    scores = [float(item["qpsi_score"]) for item in terminals]
    outcomes = [float(item["empirical_success_rate"]) for item in terminals]
    pairwise = _pairwise_counts(scores, outcomes)
    score_order = sorted(
        range(len(terminals)),
        key=lambda index: (-scores[index], str(terminals[index]["embedding_digest"])),
    )
    qpsi_index = score_order[0]
    last_index = max(
        range(len(terminals)),
        key=lambda index: (
            int(terminals[index]["last_emission_order"]),
            str(terminals[index]["embedding_digest"]),
        ),
    )
    oracle_rate = max(outcomes)
    qpsi_rate = outcomes[qpsi_index]
    last_rate = outcomes[last_index]
    if qpsi_rate > last_rate:
        comparison = "win"
    elif qpsi_rate < last_rate:
        comparison = "loss"
    else:
        comparison = "tie"
    payload = {
        "schema": AUDIT_ROW_SCHEMA,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "partition": "train",
        "task_id": source["task_id"],
        "instance_id": source["instance_id"],
        "lineage": source["lineage"],
        "state_fingerprint": source["state_fingerprint"],
        "source_quality_record_digest": source["record_digest"],
        "distinct_terminal_embeddings": len(terminals),
        "invalid_terminal_receipts": invalid_terminal_receipts,
        "terminal_embeddings": list(terminals),
        "spearman_rho": _spearman(scores, outcomes),
        "pairwise_counts": pairwise,
        "pairwise_sign_accuracy": (
            pairwise["correct"] / pairwise["comparable"]
            if pairwise["comparable"]
            else None
        ),
        "qpsi_top_embedding_digest": terminals[qpsi_index]["embedding_digest"],
        "qpsi_top_empirical_rate": qpsi_rate,
        "empirical_oracle_rate": oracle_rate,
        "qpsi_top1_regret": oracle_rate - qpsi_rate,
        "last_distinct_valid_embedding_digest": terminals[last_index][
            "embedding_digest"
        ],
        "last_distinct_valid_empirical_rate": last_rate,
        "qpsi_top_minus_last": qpsi_rate - last_rate,
        "qpsi_top_vs_last": comparison,
        "qpsi_top_beats_last_rate": float(comparison == "win"),
    }
    return _with_digest(payload)


def _metric_hierarchy(
    rows: Sequence[Mapping[str, Any]], metric: str
) -> tuple[float | None, dict[str, float]]:
    by_lineage: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        value = row[metric]
        if value is None:
            continue
        by_lineage.setdefault(str(row["lineage"]), {}).setdefault(
            str(row["instance_id"]), []
        ).append(float(value))
    lineage_values: dict[str, float] = {}
    for lineage, instances in sorted(by_lineage.items()):
        instance_values = [float(np.mean(values)) for values in instances.values()]
        if instance_values:
            lineage_values[lineage] = float(np.mean(instance_values))
    if not lineage_values:
        return None, {}
    return float(np.mean(list(lineage_values.values()))), lineage_values


def _bootstrap(
    lineage_values: Mapping[str, Mapping[str, float]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, Any]]:
    lineages = tuple(sorted(lineage_values))
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(lineages), size=(replicates, len(lineages)))
    result: dict[str, dict[str, Any]] = {}
    for metric in _METRICS:
        eligible = tuple(
            lineage for lineage in lineages if metric in lineage_values[lineage]
        )
        if len(eligible) != len(lineages):
            result[metric] = {
                "supported": False,
                "reason": "metric is not supported in every independent lineage",
                "lineages": len(eligible),
            }
            continue
        values = np.asarray(
            [lineage_values[lineage][metric] for lineage in lineages],
            dtype=np.float64,
        )
        distribution = values[samples].mean(axis=1)
        lower, upper = np.quantile(distribution, (0.025, 0.975))
        result[metric] = {
            "supported": True,
            "estimate": float(values.mean()),
            "lower": float(lower),
            "upper": float(upper),
            "confidence_level": 0.95,
            "replicates": replicates,
            "cluster_unit": "immutable-lineage",
        }
    return result


def _aggregate(
    rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    census: Mapping[str, int],
) -> dict[str, Any]:
    primary: dict[str, float | None] = {}
    per_lineage: dict[str, dict[str, float]] = {
        str(lineage): {} for lineage in sorted({row["lineage"] for row in rows})
    }
    support: dict[str, dict[str, int]] = {}
    for metric in _METRICS:
        value, lineage_values = _metric_hierarchy(rows, metric)
        primary[metric] = value
        support[metric] = {
            "opportunities": sum(row[metric] is not None for row in rows),
            "lineages": len(lineage_values),
        }
        for lineage, lineage_value in lineage_values.items():
            per_lineage[lineage][metric] = lineage_value
    pairwise = {
        key: sum(int(row["pairwise_counts"][key]) for row in rows)
        for key in ("correct", "incorrect", "score_ties", "target_ties", "comparable")
    }
    aggregate = {
        **dict(census),
        "instances": len({row["instance_id"] for row in rows}),
        "lineages": len(per_lineage),
        "exact_work_census": {
            "selector_forward_calls": census["valid_terminal_receipts"],
            "selector_program_inputs": N_STRENGTHS
            * census["valid_terminal_receipts"],
            "program_compiler_calls": N_STRENGTHS
            * census["valid_terminal_receipts"],
        },
        "equal_lineage_then_equal_instance": primary,
        "metric_support": support,
        "pairwise_counts": pairwise,
        "lineage_cluster_bootstrap": _bootstrap(
            per_lineage,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "bootstrap_seed": bootstrap_seed,
    }
    for value in primary.values():
        if value is not None and not math.isfinite(value):
            raise RuntimeError("qpsi audit produced a non-finite aggregate")
    return aggregate


def audit_qpsi_ordering(
    selector: object,
    quality_rows: Sequence[Mapping[str, object]],
    prepared_tasks: Sequence[PreparedTask],
    ctx: Context,
    *,
    bootstrap_replicates: int = 20_000,
    bootstrap_seed: int = 26_090_601,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Measure frozen q_psi ordering only where authenticated endpoints are comparable."""

    if not isinstance(ctx, Context):
        raise TypeError("qpsi ordering audit requires the registered Context")
    if not bool(getattr(selector, "frozen", False)) or not bool(
        getattr(selector, "deployment_ready", False)
    ):
        raise ValueError("qpsi ordering audit requires a frozen deployment-ready selector")
    if (
        isinstance(bootstrap_replicates, bool)
        or not isinstance(bootstrap_replicates, int)
        or bootstrap_replicates < MINIMUM_BOOTSTRAP_REPLICATES
    ):
        raise ValueError("qpsi ordering audit requires at least 1000 bootstrap replicates")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise ValueError("qpsi ordering bootstrap seed must be a nonnegative integer")
    if not quality_rows:
        raise ValueError(MINIMUM_CROSS_EMBEDDING_CONTRACT)

    by_task: dict[str, PreparedTask] = {}
    for prepared in prepared_tasks:
        if not isinstance(prepared, PreparedTask) or prepared.partition != "train":
            raise ValueError("qpsi ordering audit accepts prepared train tasks only")
        if prepared.task_id in by_task:
            raise ValueError("qpsi ordering audit received duplicate prepared task identities")
        by_task[prepared.task_id] = prepared

    output: list[dict[str, Any]] = []
    seen_records: set[str] = set()
    seen_opportunities: set[tuple[str, str]] = set()
    total_valid = 0
    total_invalid = 0
    total_distinct = 0
    excluded_opportunities = 0
    with torch.no_grad():
        for source in quality_rows:
            digest = source.get("record_digest")
            unsigned = {key: value for key, value in source.items() if key != "record_digest"}
            if not isinstance(digest, str) or digest != content_digest(unsigned):
                raise ValueError("quality source record digest mismatch")
            if digest in seen_records:
                raise ValueError("duplicate quality source record in qpsi audit")
            seen_records.add(digest)
            required = {
                "schema",
                "schema_version",
                "task_id",
                "instance_id",
                "lineage",
                "partition",
                "state_fingerprint",
                "evaluated",
            }
            if not required <= set(source):
                raise ValueError("quality source record omits qpsi audit identity fields")
            if (
                source["schema"] != "isingfold.quality-counterfactual"
                or source["schema_version"] != 7
                or source["partition"] != "train"
            ):
                raise ValueError("qpsi ordering audit accepts authenticated train quality v7 only")
            task_id = source["task_id"]
            if not isinstance(task_id, str) or task_id not in by_task:
                raise ValueError("quality source task is absent from the prepared train corpus")
            prepared = by_task[task_id]
            if any(
                not isinstance(source[field], str) or not source[field]
                for field in ("instance_id", "lineage", "state_fingerprint")
            ):
                raise ValueError("quality source has malformed opportunity identities")
            if (
                source["instance_id"] != prepared.instance_id
                or source["lineage"] != prepared.task.lineage
            ):
                raise ValueError("quality source instance or lineage differs from prepared task")
            opportunity = (task_id, str(source["state_fingerprint"]))
            if opportunity in seen_opportunities:
                raise ValueError("qpsi audit source repeats one decision opportunity")
            seen_opportunities.add(opportunity)
            evaluated = source["evaluated"]
            if not isinstance(evaluated, list):
                raise ValueError("quality source evaluated actions are malformed")
            occurrences: list[dict[str, Any]] = []
            evaluator_seeds: set[int] = set()
            order = 0
            invalid = 0
            for action in evaluated:
                if not isinstance(action, Mapping):
                    raise ValueError("quality source evaluated action is malformed")
                receipts = action.get("continuation_receipts")
                if not isinstance(receipts, list):
                    raise ValueError("quality source omits continuation receipts")
                for receipt in receipts:
                    if not isinstance(receipt, Mapping):
                        raise ValueError("quality continuation receipt is malformed")
                    occurrence = _terminal_occurrence(
                        selector=selector,
                        prepared=prepared,
                        receipt=receipt,
                        order=order,
                        ctx=ctx,
                    )
                    order += 1
                    if occurrence is None:
                        invalid += 1
                        continue
                    seed = int(occurrence["evaluator_seed"])
                    if seed in evaluator_seeds:
                        raise ValueError(
                            "terminal embeddings do not have independent evaluator count seeds"
                        )
                    evaluator_seeds.add(seed)
                    occurrences.append(occurrence)
            terminals = _deduplicate_occurrences(occurrences)
            total_valid += len(occurrences)
            total_invalid += invalid
            total_distinct += len(terminals)
            if len(terminals) < 2:
                excluded_opportunities += 1
                continue
            output.append(
                _opportunity_row(
                    source=source,
                    terminals=terminals,
                    invalid_terminal_receipts=invalid,
                )
            )

    if not output:
        raise ValueError(MINIMUM_CROSS_EMBEDDING_CONTRACT)
    lineages = {str(row["lineage"]) for row in output}
    if len(lineages) < 2:
        raise ValueError(
            "qpsi ordering audit requires at least two independent train lineages; "
            + MINIMUM_CROSS_EMBEDDING_CONTRACT
        )
    output.sort(
        key=lambda row: (
            str(row["lineage"]),
            str(row["instance_id"]),
            str(row["task_id"]),
            str(row["state_fingerprint"]),
        )
    )
    aggregate = _aggregate(
        output,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
        census={
            "source_opportunities": len(quality_rows),
            "comparable_opportunities": len(output),
            "excluded_noncomparable_opportunities": excluded_opportunities,
            "valid_terminal_receipts": total_valid,
            "invalid_terminal_receipts": total_invalid,
            "distinct_terminal_embeddings": total_distinct,
        },
    )
    return tuple(output), aggregate


def _validate_signed_record(value: object, *, name: str) -> None:
    if not isinstance(value, Mapping) or "record_digest" not in value:
        raise ValueError(f"qpsi provenance has no authenticated {name}")
    digest = value["record_digest"]
    unsigned = {key: item for key, item in value.items() if key != "record_digest"}
    if not isinstance(digest, str) or digest != content_digest(unsigned):
        raise ValueError(f"qpsi provenance {name} digest mismatch")


def _validate_provenance(
    provenance: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> None:
    required = {
        "partition",
        "sealed_validation_or_test_opened",
        "source_corpus_manifest_sha256",
        "source_quality_manifest_sha256",
        "source_quality_manifest_record_digest",
        "source_quality_records_sha256",
        "quality_preflight_receipt_sha256",
        "quality_preflight_record_digest",
        "selector_digest",
        "selector_file_sha256",
        "selector_fit_receipt_sha256",
        "selector_fit_record_digest",
        "normalizer_digest",
        "quality_authority",
        "target_access",
        "ground_partition_receipt",
        "quality_implementation_contract",
        "quality_implementation_contract_digest",
        "task_evaluator_protocols",
        "task_evaluator_protocols_digest",
        "runtime_implementation_registry",
        "runtime_implementation_digest",
        "implementation_sources",
    }
    if set(provenance) != required:
        raise ValueError("qpsi provenance fields differ from the registered schema")
    if (
        provenance["partition"] != "train"
        or provenance["sealed_validation_or_test_opened"] is not False
    ):
        raise ValueError("qpsi diagnostic is train-only and must not open sealed validation/test")
    digest_fields = required & {
        "source_corpus_manifest_sha256",
        "source_quality_manifest_sha256",
        "source_quality_manifest_record_digest",
        "source_quality_records_sha256",
        "quality_preflight_receipt_sha256",
        "quality_preflight_record_digest",
        "selector_digest",
        "selector_file_sha256",
        "selector_fit_receipt_sha256",
        "selector_fit_record_digest",
        "normalizer_digest",
    }
    if any(not _is_sha256(provenance[field]) for field in digest_fields):
        raise ValueError("qpsi provenance contains an invalid artifact digest")
    for field in ("quality_authority", "target_access", "ground_partition_receipt"):
        _validate_signed_record(provenance[field], name=field.replace("_", " "))

    quality_contract = provenance["quality_implementation_contract"]
    if (
        not isinstance(quality_contract, Mapping)
        or provenance["quality_implementation_contract_digest"]
        != content_digest(quality_contract)
    ):
        raise ValueError("qpsi provenance quality implementation contract mismatch")
    evaluator_protocols = provenance["task_evaluator_protocols"]
    if (
        not isinstance(evaluator_protocols, list)
        or provenance["task_evaluator_protocols_digest"]
        != content_digest(evaluator_protocols)
    ):
        raise ValueError("qpsi provenance evaluator protocol digest mismatch")
    task_ids: set[str] = set()
    for protocol in evaluator_protocols:
        if (
            not isinstance(protocol, Mapping)
            or set(protocol) != {"task_id", "evaluator_protocol_digest"}
            or not isinstance(protocol["task_id"], str)
            or not _is_sha256(protocol["evaluator_protocol_digest"])
            or protocol["task_id"] in task_ids
        ):
            raise ValueError("qpsi provenance evaluator protocol entries are malformed")
        task_ids.add(protocol["task_id"])
    if task_ids != {str(row["task_id"]) for row in rows}:
        raise ValueError("qpsi provenance evaluator protocols do not cover raw opportunities")

    runtime = provenance["runtime_implementation_registry"]
    if (
        not isinstance(runtime, Mapping)
        or dict(runtime) != runtime_implementation_registry()
        or provenance["runtime_implementation_digest"] != content_digest(runtime)
    ):
        raise ValueError("qpsi provenance runtime implementation registry mismatch")
    sources = provenance["implementation_sources"]
    if (
        not isinstance(sources, Mapping)
        or set(sources) != _SOURCE_KEYS
        or any(not _is_sha256(value) for value in sources.values())
    ):
        raise ValueError("qpsi provenance implementation source hashes are incomplete")
    from isingfold.rl import evaluator, program, qpsi_audit, strength, strength_tensorize

    expected_sources = {
        name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name, module in {
            "evaluator": evaluator,
            "program": program,
            "qpsi_audit": qpsi_audit,
            "selector": strength,
            "selector_tensorizer": strength_tensorize,
        }.items()
    }
    if dict(sources) != expected_sources:
        raise ValueError("qpsi provenance implementation source hash mismatch")


def _validate_raw_rows(
    rows: Sequence[Mapping[str, Any]], aggregate: Mapping[str, Any]
) -> None:
    if not rows:
        raise ValueError("qpsi audit cannot publish an empty diagnostic")
    source_records: set[str] = set()
    for row in rows:
        if (
            row.get("schema") != AUDIT_ROW_SCHEMA
            or row.get("schema_version") != AUDIT_SCHEMA_VERSION
            or row.get("partition") != "train"
        ):
            raise ValueError("raw qpsi audit row has an unsupported schema or partition")
        digest = row.get("record_digest")
        unsigned = {key: value for key, value in row.items() if key != "record_digest"}
        if not isinstance(digest, str) or digest != content_digest(unsigned):
            raise ValueError("raw qpsi audit row digest mismatch")
        source = row.get("source_quality_record_digest")
        if not _is_sha256(source) or source in source_records:
            raise ValueError("raw qpsi audit rows repeat or omit source quality identities")
        source_records.add(source)
    if (
        aggregate.get("comparable_opportunities") != len(rows)
        or aggregate.get("lineages") != len({row["lineage"] for row in rows})
        or aggregate.get("instances") != len({row["instance_id"] for row in rows})
    ):
        raise ValueError("qpsi audit aggregate/raw census differs")


def _write_sync(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def publish_qpsi_ordering_audit(
    destination: str | os.PathLike[str],
    *,
    rows: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish raw train-only diagnostics and a provenance-bound receipt."""

    target = Path(destination)
    if target.exists():
        raise FileExistsError(f"qpsi-audit output already exists: {target}")
    _validate_raw_rows(rows, aggregate)
    _validate_provenance(provenance, rows)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.qpsi-audit-", dir=target.parent))
    try:
        raw = b"".join(canonical_json_bytes(dict(row)) + b"\n" for row in rows)
        _write_sync(temporary / "opportunities.jsonl", raw)
        receipt = _with_digest(
            {
                "schema": AUDIT_SCHEMA,
                "schema_version": AUDIT_SCHEMA_VERSION,
                "scope": "DIAGNOSTIC_ONLY",
                "partition": "train",
                "sealed_validation_or_test_opened": False,
                "archive_policy_authorized": False,
                "training_or_calibration_authorized": False,
                "qpsi_score_contract": (
                    "maximum frozen IF-Q3-S0 success probability across the four "
                    "recompiled registered-strength programs"
                ),
                "target_contract": (
                    "aggregate hits divided by aggregate reads from independent authenticated "
                    "terminal evaluator count blocks for one distinct embedding"
                ),
                "last_distinct_valid_comparator": {
                    "available": True,
                    "contract": (
                        "last distinct valid terminal in outcome-blind quality-receipt emission "
                        "order within the same opportunity"
                    ),
                },
                "fifo_archive_comparator": {
                    "available": False,
                    "reason": (
                        "parallel counterfactual continuation receipts do not identify an "
                        "archive admission stream or its protected/FIFO slot"
                    ),
                },
                "limitations": [
                    "empirical top-1 is optimistic because it maximizes noisy finite-read rates",
                    "the diagnostic does not change the MDP, environment, or archive policy",
                    "train-only evidence cannot establish held-out end-to-end superiority",
                ],
                "aggregation_contract": (
                    "mean opportunities within instance, mean instances within immutable "
                    "lineage, then equal-weight mean across lineages"
                ),
                "raw_opportunities": {
                    "path": "opportunities.jsonl",
                    "schema": AUDIT_ROW_SCHEMA,
                    "count": len(rows),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
                "aggregate": dict(aggregate),
                "provenance": dict(provenance),
            }
        )
        _write_sync(temporary / "receipt.json", canonical_json_bytes(receipt) + b"\n")
        os.replace(temporary, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        except (AttributeError, OSError):
            pass
        else:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return receipt


__all__ = [
    "AUDIT_ROW_SCHEMA",
    "AUDIT_SCHEMA",
    "AUDIT_SCHEMA_VERSION",
    "MINIMUM_CROSS_EMBEDDING_CONTRACT",
    "audit_qpsi_ordering",
    "publish_qpsi_ordering_audit",
]
