"""Strict CLI orchestration for the post-freeze final-strength audit.

The statistical implementation is deliberately kept in ``final_strength_audit``.  This
module owns filesystem chronology: planning sees public corpus metadata only, while execution
sealing and shards open exactly the test target partition and authenticate all six source runs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True)
class FinalStrengthSourceArguments:
    """Aligned command-line roots and mandatory out-of-band report pins."""

    learned_directories: tuple[str, ...]
    learned_report_sha256s: tuple[str, ...]
    stock_directories: tuple[str, ...]
    stock_report_sha256s: tuple[str, ...]


def source_arguments(args: argparse.Namespace) -> FinalStrengthSourceArguments:
    """Validate the complete six-source census before any path is opened."""

    learned_directories = tuple(args.learned_evaluation)
    learned_pins = tuple(args.expected_learned_report_sha256)
    stock_directories = tuple(args.external_evaluation)
    stock_pins = tuple(args.expected_external_report_sha256)
    if len(learned_directories) != 3 or len(learned_pins) != 3:
        raise ValueError(
            "final-strength execution requires exactly three learned roots and pins"
        )
    if len(stock_directories) != 3 or len(stock_pins) != 3:
        raise ValueError(
            "final-strength execution requires exactly three tuned-stock roots and pins"
        )
    if any(not _is_digest(value) for value in (*learned_pins, *stock_pins)):
        raise ValueError("final-strength report pins must be lowercase SHA-256 digests")
    return FinalStrengthSourceArguments(
        learned_directories,
        learned_pins,
        stock_directories,
        stock_pins,
    )


def plan_final_strength_audit(args: argparse.Namespace) -> None:
    """Seal the public, outcome-blind test subset."""

    from isingfold.rl.cli import (
        _context,
        _load_grid,
        _load_selector_bundle,
        _preinitialization_population_tasks,
        _publisher_attestation_pin,
        _sha256_file,
        _strict_json,
        _verify_record,
    )
    from isingfold.rl.complete_system import (
        CompletePopulationIdentity,
        FrozenComponentIdentity,
        task_population_digest,
    )
    from isingfold.rl.contracts import stable_digest
    from isingfold.rl.data.ground_certificate import (
        load_ground_certificate_root,
        project_ground_root_quality_authority,
    )
    from isingfold.rl.data.prepared import (
        load_prepared_corpus_design,
        load_prepared_partition,
    )
    from isingfold.rl.evaluation_strata import evaluation_contract_from_prepared
    from isingfold.rl.experiment_selection import load_rl_value_freeze
    from isingfold.rl.external_tuning import (
        ExternalTuningExecutionBinding,
        load_external_tuning_registry,
        load_external_tuning_selection,
    )
    from isingfold.rl.final_strength_audit import (
        build_final_strength_audit_plan,
        load_final_strength_audit_config,
        write_final_strength_audit_plan,
    )
    from isingfold.rl.final_strength_workflow import frozen_learned_selection_payload

    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"final-strength audit plan already exists: {destination}")
    grid, grid_sha256 = _load_grid(args.grid)
    protocol = _test_protocol(grid)
    selection = load_rl_value_freeze(
        receipt_path=args.rl_value_selection_receipt,
        grid_path=args.grid,
        expected_sha256=args.expected_selection_sha256,
    )
    if selection.grid_manifest_sha256 != grid_sha256:
        raise ValueError("final-strength learned selection and grid identity differ")
    tuning_registry = load_external_tuning_registry(
        args.tuning_registry,
        expected_file_sha256=args.expected_tuning_registry_sha256,
        grid_path=args.grid,
        external_config_path=args.external_config,
    )
    tuning_selection = load_external_tuning_selection(
        args.external_tuning_selection,
        expected_file_sha256=args.expected_external_tuning_selection_sha256,
        registry=tuning_registry,
    )
    tuning_execution = ExternalTuningExecutionBinding.for_deployment(tuning_selection)
    audit_config = load_final_strength_audit_config(
        args.audit_config,
        expected_sha256=args.expected_audit_config_sha256,
    )

    publisher_pin = _publisher_attestation_pin(args)
    ground_root = load_ground_certificate_root(
        args.ground_certificate_root,
        expected_sha256=args.expected_ground_certificate_root_sha256,
        corpus_directory=args.corpus,
        quality_attestation_pin=publisher_pin,
    )
    global_authority, evaluation_authority = (
        project_ground_root_quality_authority(ground_root, partition="test")
    )
    if evaluation_authority is None:  # pragma: no cover - fixed partition contract
        raise RuntimeError("test quality-authority projection is unexpectedly absent")
    quality_authority = _quality_authority(
        global_authority.as_dict(), evaluation_authority.as_dict()
    )

    # This is the chronology boundary: the public v4 loader cannot open a target file.
    public_load = load_prepared_partition(
        args.corpus,
        partition="test",
        include_evaluator=False,
    )
    if public_load.target_access is not None:
        raise RuntimeError("outcome-blind planning unexpectedly obtained target access")
    manifest_path = Path(args.corpus) / "manifest.json"
    manifest = _strict_json(manifest_path)
    tasks = _preinitialization_population_tasks(
        public_load.tasks,
        manifest=manifest,
        partition="test",
    )
    evaluation_strata, confirmatory_design = evaluation_contract_from_prepared(
        public_load.tasks,
        corpus_design_receipt=load_prepared_corpus_design(args.corpus),
        partition="test",
        noninferiority_margin=float(protocol["feasibility_noninferiority_margin"]),
    )
    context = _context(args, args.corpus)
    manifest_sha256 = _sha256_file(manifest_path)
    population = CompletePopulationIdentity(
        population_id=(
            "preinitialization-"
            + stable_digest(
                {
                    "manifest": manifest_sha256,
                    "partition": "test",
                    "evaluation_seed": protocol["evaluation_seed"],
                    "repetitions": protocol["repetitions"],
                }
            )
        ),
        source_manifest_sha256=manifest_sha256,
        task_payload_sha256=task_population_digest(tasks),
        expected_instances=tuple(
            sorted((task.lineage or task.name, task.name) for task in tasks)
        ),
        expected_repetitions=int(protocol["repetitions"]),
        evaluation_strata=evaluation_strata,
        confirmatory_design=confirmatory_design,
        evaluation_seed=int(protocol["evaluation_seed"]),
    )
    bundle = _load_selector_bundle(args.selector, corpus=args.corpus)
    bundle_global = bundle.quality_authority.get("global")
    if bundle_global != global_authority.as_dict():
        raise ValueError(
            "selector and final-strength plan differ on global quality authority"
        )
    fit_receipt = _strict_json(bundle.root / "fit_receipt.json")
    _verify_record(fit_receipt, "selector fit receipt")
    selector_identity = FrozenComponentIdentity(
        component_id="isingfold-if-q3-s0-strength-selector",
        version=str(bundle.model.version),
        implementation=(
            f"{type(bundle.model).__module__}.{type(bundle.model).__qualname__}:"
            f"fit-{fit_receipt['record_digest']}"
        ),
        artifact_sha256=_sha256_file(bundle.root / "selector.pt"),
    )
    plan = build_final_strength_audit_plan(
        population,
        context,
        selector_identity,
        quality_authority=quality_authority,
        learned_selection=frozen_learned_selection_payload(selection),
        stock_tuning_execution=tuning_execution.as_dict(),
        audit_config=audit_config,
    )
    plan_sha256 = write_final_strength_audit_plan(destination, plan)
    print(
        json.dumps(
            {
                "outcome_blind": True,
                "publication_eligible": plan.publication_eligible,
                "selected_instances": len(plan.selected_instances),
                "shard_count": plan.shard_count,
                "record_digest": plan.record_digest,
                "file_sha256": plan_sha256,
                "out": str(destination),
            },
            indent=1,
        )
    )


def seal_final_strength_audit_execution(args: argparse.Namespace) -> None:
    """Open test targets, authenticate six sources, and seal exact audit work."""

    from isingfold.rl.final_strength_audit import (
        build_final_strength_audit_execution_manifest,
        write_final_strength_audit_execution_manifest,
    )

    sources, sealed_plan, opened, tasks, context = _authenticated_execution_inputs(args)
    execution = build_final_strength_audit_execution_manifest(
        sealed_plan,
        sources.learned,
        sources.stock,
        tasks,
        context,
        target_access=opened.target_access.as_dict(),
        source_report_sha256_pins=sources.report_sha256_pins,
    )
    manifest_sha256 = write_final_strength_audit_execution_manifest(args.out, execution)
    print(
        json.dumps(
            {
                "publication_eligible": execution.publication_eligible,
                "source_runs": len(execution.source_runs),
                "target_access_digest": execution.target_access_digest,
                "opportunities": len(execution.opportunity_execution),
                "record_digest": execution.record_digest,
                "file_sha256": manifest_sha256,
                "out": str(Path(args.out)),
            },
            indent=1,
        )
    )


def run_final_strength_audit_shard_command(args: argparse.Namespace) -> None:
    """Execute and atomically publish one deterministic audit shard."""

    from isingfold.rl.final_strength_audit import (
        load_final_strength_audit_execution_manifest,
        load_final_strength_audit_plan,
        publish_final_strength_audit_shard,
        run_final_strength_audit_shard,
    )
    from isingfold.rl.cli import _sha256_file

    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"final-strength audit shard already exists: {destination}")
    requested_sources = source_arguments(args)
    sealed_plan = load_final_strength_audit_plan(
        args.plan,
        expected_sha256=args.expected_plan_sha256,
    )
    if not sealed_plan.plan.publication_eligible:
        raise ValueError("final-strength CLI rejects non-publication audit plans")
    if not 0 <= args.shard_index < sealed_plan.plan.shard_count:
        raise ValueError("final-strength shard index is outside the sealed census")
    sealed_execution = load_final_strength_audit_execution_manifest(
        args.execution_manifest,
        expected_sha256=args.expected_execution_manifest_sha256,
        sealed_plan=sealed_plan,
    )
    sources, _, opened, tasks, context = _authenticated_execution_inputs(
        args,
        requested_sources=requested_sources,
        sealed_plan=sealed_plan,
    )
    if (
        opened.target_access.as_dict()["record_digest"]
        != sealed_execution.manifest.target_access_digest
    ):
        raise ValueError("current test-target access differs from the execution seal")
    result = run_final_strength_audit_shard(
        sealed_plan,
        sealed_execution,
        sources.learned,
        sources.stock,
        tasks,
        context,
        shard_index=args.shard_index,
    )
    receipt = publish_final_strength_audit_shard(
        destination,
        sealed_plan=sealed_plan,
        sealed_execution_manifest=sealed_execution,
        result=result,
    )
    print(
        json.dumps(
            {
                "publication_eligible": receipt["publication_eligible"],
                "shard_index": receipt["shard_index"],
                "rows": receipt["raw_rows"]["count"],
                "record_digest": receipt["record_digest"],
                "receipt_file_sha256": _sha256_file(destination / "receipt.json"),
                "out": str(destination),
            },
            indent=1,
        )
    )


def _test_protocol(grid: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol = grid.get("complete_system_evaluation")
    if not isinstance(protocol, Mapping) or protocol.get("partition") != "test":
        raise ValueError("final-strength audit requires the registered test protocol")
    evaluation_seed = protocol.get("evaluation_seed")
    repetitions = protocol.get("repetitions")
    if type(evaluation_seed) is not int or not 0 <= evaluation_seed < 2**31:
        raise ValueError("final-strength protocol evaluation seed is invalid")
    if type(repetitions) is not int or repetitions <= 0:
        raise ValueError("final-strength protocol repetitions must be positive")
    margin = protocol.get("feasibility_noninferiority_margin")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise ValueError("final-strength protocol has no noninferiority margin")
    return protocol


def _quality_authority(
    global_authority: Mapping[str, object],
    evaluation_partition: Mapping[str, object],
) -> dict[str, object]:
    from isingfold.rl.data.import_embedbench import content_digest

    payload: dict[str, object] = {
        "schema": "isingfold.quality-authority-binding",
        "schema_version": 2,
        "global": dict(global_authority),
        "evaluation_partition": dict(evaluation_partition),
    }
    return {**payload, "record_digest": content_digest(payload)}


def _authenticated_execution_inputs(
    args: argparse.Namespace,
    *,
    requested_sources: FinalStrengthSourceArguments | None = None,
    sealed_plan=None,
):
    """Open exactly test, prove ground authority, and authenticate six source runs."""

    from isingfold.rl.cli import (
        _context,
        _preinitialization_population_tasks,
        _publisher_attestation_pin,
        _strict_json,
    )
    from isingfold.rl.data.ground_certificate import (
        load_ground_certificate_partition,
        project_ground_partition_quality_authority,
    )
    from isingfold.rl.final_strength_audit import load_final_strength_audit_plan
    from isingfold.rl.final_strength_workflow import authenticate_final_strength_sources

    if requested_sources is None:
        requested_sources = source_arguments(args)
    destination = Path(args.out)
    if destination.exists():
        raise FileExistsError(f"final-strength output already exists: {destination}")
    if sealed_plan is None:
        sealed_plan = load_final_strength_audit_plan(
            args.plan,
            expected_sha256=args.expected_plan_sha256,
        )
    if not sealed_plan.plan.publication_eligible:
        raise ValueError("final-strength CLI rejects non-publication audit plans")
    publisher_pin = _publisher_attestation_pin(args)
    opened = load_ground_certificate_partition(
        args.ground_certificate_root,
        expected_root_sha256=args.expected_ground_certificate_root_sha256,
        partition="test",
        corpus_directory=args.corpus,
        quality_attestation_pin=publisher_pin,
    )
    global_authority, evaluation_authority = (
        project_ground_partition_quality_authority(opened)
    )
    current_authority = _quality_authority(
        global_authority.as_dict(), evaluation_authority.as_dict()
    )
    if current_authority != dict(sealed_plan.plan.quality_authority):
        raise ValueError("current test quality authority differs from the sealed plan")
    manifest = _strict_json(Path(args.corpus) / "manifest.json")
    tasks = _preinitialization_population_tasks(
        opened.tasks,
        manifest=manifest,
        partition="test",
    )
    context = _context(args, args.corpus)
    authenticated_sources = authenticate_final_strength_sources(
        learned_directories=requested_sources.learned_directories,
        learned_report_sha256s=requested_sources.learned_report_sha256s,
        stock_directories=requested_sources.stock_directories,
        stock_report_sha256s=requested_sources.stock_report_sha256s,
        tasks=tasks,
        context=context,
    )
    return authenticated_sources, sealed_plan, opened, tasks, context


__all__ = [
    "FinalStrengthSourceArguments",
    "plan_final_strength_audit",
    "run_final_strength_audit_shard_command",
    "seal_final_strength_audit_execution",
    "source_arguments",
]
