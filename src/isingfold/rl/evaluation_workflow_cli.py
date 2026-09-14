"""Thin production orchestration for resumable complete-system evaluation.

The plan command authenticates the entire public partition without opening evaluator
targets.  Run and merge independently open exactly the selected ground-certificate
partition under explicit external pins, reconstruct the full plan, and only then permit
one lineage-coherent shard or a complete merge.  Scientific algorithms remain in their
typed workflow modules; this file only joins their existing authorities at the CLI edge.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from isingfold.rl.complete_system import CompletePopulationIdentity
from isingfold.rl.contracts import Context
from isingfold.rl.data.prepared import PreparedTask, load_prepared_partition
from isingfold.rl.data.import_embedbench import content_digest
from isingfold.rl.env import EmbeddingTask
from isingfold.rl.evaluation_shards import (
    DEFAULT_MAX_LINEAGES_PER_SHARD,
    EvaluationExecutionPlan,
    EvaluationWorkflow,
    TypedReceipt,
    build_evaluation_plan,
    load_evaluation_plan,
    load_evaluation_shard,
    merge_evaluation_shards,
    publish_evaluation_shard,
    publish_merged_evaluation,
    select_tasks_for_shard,
    validate_evaluation_compute_class,
    write_evaluation_plan,
)

TargetMode = Literal["public", "ground"]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pinned_artifact(
    path_value: str | os.PathLike[str], expected_sha256: str, *, label: str
) -> tuple[str, str]:
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"{label} requires a lowercase SHA-256 pin")
    try:
        int(expected_sha256, 16)
        path = Path(path_value).resolve(strict=True)
    except (TypeError, ValueError, OSError, RuntimeError) as exc:
        raise ValueError(f"{label} path or SHA-256 pin is invalid") from exc
    if not path.is_file():
        raise ValueError(f"{label} is not a regular file")
    observed = _sha256_file(path)
    if not hmac.compare_digest(observed, expected_sha256):
        raise ValueError(f"{label} differs from its out-of-band SHA-256 pin")
    return str(path), observed


def _execution_environment_identity(args: argparse.Namespace) -> Mapping[str, object]:
    """Verify and identify the environment actually executing this command."""

    from isingfold.rl.checkpoint import runtime_implementation_registry

    mode = str(args.execution_mode)
    environment_lock_path: str | None = None
    environment_lock_sha256: str | None = None
    container_runtime_path: str | None = None
    container_runtime_sha256: str | None = None
    container_image_path: str | None = None
    container_image_sha256: str | None = None
    if mode == "pinned-venv":
        environment_lock_path, environment_lock_sha256 = _pinned_artifact(
            args.environment_lock,
            args.expected_environment_lock_sha256,
            label="environment lock",
        )
    elif mode == "apptainer":
        container_runtime_path, container_runtime_sha256 = _pinned_artifact(
            args.container_runtime,
            args.expected_container_runtime_sha256,
            label="Apptainer runtime",
        )
        container_image_path, container_image_sha256 = _pinned_artifact(
            args.container_image,
            args.expected_container_image_sha256,
            label="Apptainer image",
        )
    elif mode != "bare-metal":
        raise ValueError(f"unsupported evaluation execution mode {mode!r}")
    python_path = Path(sys.executable).resolve(strict=True)
    if not python_path.is_file():
        raise ValueError("evaluation Python executable is not a regular file")
    runtime_registry = runtime_implementation_registry()
    source_registry = {
        "isingfold/rl/evaluation_shards.py": hashlib.sha256(
            Path(__file__).with_name("evaluation_shards.py").read_bytes()
        ).hexdigest(),
        "isingfold/rl/evaluation_workflow_cli.py": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    return MappingProxyType(
        {
            "mode": mode,
            "environment_lock_path": environment_lock_path,
            "environment_lock_sha256": environment_lock_sha256,
            "container_runtime_path": container_runtime_path,
            "container_runtime_sha256": container_runtime_sha256,
            "container_image_path": container_image_path,
            "container_image_sha256": container_image_sha256,
            "python_executable_sha256": _sha256_file(python_path),
            "runtime_implementation_digest": content_digest(runtime_registry),
            "source_digest": content_digest(source_registry),
        }
    )


def _compute_class_and_provenance(
    args: argparse.Namespace,
    *,
    device: object,
) -> tuple[Mapping[str, object], Mapping[str, object]]:
    """Build host-independent class authority plus exact per-command provenance."""

    from isingfold.rl import cli as core
    from isingfold.rl.external_pairing import runtime_identity

    runtime_platform = core._runtime_platform_identity()
    device_type = str(getattr(device, "type", ""))
    runtime = runtime_identity(
        runtime_platform=runtime_platform,
        inference_device_type=device_type,
        inference_device_name=core._inference_device_name(device, runtime_platform),
        inference_threads=int(args.threads),
        deterministic=bool(args.deterministic),
    )
    cluster = str(args.cluster)
    scheduler = "slurm" if cluster == "goose" else "direct"
    slurm_partition = args.slurm_partition if cluster == "goose" else None
    environment = _execution_environment_identity(args)
    class_payload: dict[str, object] = {
        "schema": "isingfold.evaluation-compute-class",
        "schema_version": 1,
        "cluster": cluster,
        "scheduler": scheduler,
        "slurm_partition": slurm_partition,
        "platform": {
            name: runtime_platform[name] for name in ("system", "release", "machine", "processor")
        },
        "inference": {
            "device_type": runtime["inference_device_type"],
            "device_name": runtime["inference_device_name"],
            "threads": runtime["inference_threads"],
            "deterministic": runtime["deterministic"],
        },
        "execution_environment": dict(environment),
        "publication_eligible": (
            args.execution_mode in {"pinned-venv", "apptainer"} and args.deterministic is True
        ),
    }
    compute_class = validate_evaluation_compute_class(
        {**class_payload, "record_digest": content_digest(class_payload)}
    )
    provenance_payload: dict[str, object] = {
        "schema": "isingfold.evaluation-compute-provenance",
        "schema_version": 1,
        "compute_class": dict(compute_class),
        "compute_class_digest": content_digest(compute_class),
        "runtime_identity": dict(runtime),
        "hostname": runtime_platform["hostname"],
        "node_id": os.environ.get("SLURMD_NODENAME") or str(runtime_platform["hostname"]),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    provenance = {
        **provenance_payload,
        "record_digest": content_digest(provenance_payload),
    }
    return compute_class, MappingProxyType(provenance)


def _open_evaluation_partition(
    args: argparse.Namespace,
    *,
    partition: str,
    target_mode: TargetMode,
) -> tuple[
    tuple[PreparedTask, ...],
    Mapping[str, object] | None,
    Mapping[str, object] | None,
]:
    """Project a public authority, opening targets only in explicit ground mode."""

    from isingfold.rl import cli as core

    normalized = core._normalize_quality_partition(partition)
    if normalized not in {"val", "test"}:
        raise ValueError("evaluation sharding is restricted to val/test partitions")
    pin = core._quality_attestation_pin(args)

    if target_mode == "public":
        loaded = load_prepared_partition(
            args.corpus,
            partition=normalized,
            include_evaluator=False,
        )
        if loaded.target_access is not None:
            raise RuntimeError("public evaluation preparation unexpectedly opened targets")
        authority = core._quality_authority_identity(loaded.tasks, pin=pin)
        core._validate_quality_authority_binding(
            authority,
            expected_role="evaluation_partition",
        )
        return tuple(loaded.tasks), None, authority
    if target_mode != "ground":
        raise ValueError("target_mode must be 'public' or 'ground'")
    tasks, authority, access, _ground_receipt = core._load_quality_partition(
        args.corpus,
        partition=normalized,
        pin=pin,
        role="evaluation_partition",
    )
    core._validate_quality_authority_binding(
        authority,
        expected_role="evaluation_partition",
    )
    return tuple(tasks), access, authority


@dataclass(frozen=True)
class PreparedEvaluationWorkflow:
    """Fully authenticated inputs plus ephemeral runtime objects for one workflow cell."""

    workflow: EvaluationWorkflow
    population: CompletePopulationIdentity
    tasks: tuple[EmbeddingTask, ...]
    context: Context
    run_coordinates: Mapping[str, object]
    execution_contract: Mapping[str, object]
    target_access_receipt: Mapping[str, object] | None = None
    ground_certificate_authority: Mapping[str, object] | None = None
    compute_class: Mapping[str, object] | None = None
    compute_provenance: Mapping[str, object] | None = None
    runtime: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "workflow", EvaluationWorkflow(self.workflow))
        object.__setattr__(self, "tasks", tuple(self.tasks))
        object.__setattr__(self, "run_coordinates", MappingProxyType(dict(self.run_coordinates)))
        object.__setattr__(
            self, "execution_contract", MappingProxyType(dict(self.execution_contract))
        )
        for name in (
            "target_access_receipt",
            "ground_certificate_authority",
            "compute_class",
            "compute_provenance",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, MappingProxyType(dict(value)))
        object.__setattr__(self, "runtime", MappingProxyType(dict(self.runtime)))

        identities = tuple(sorted((task.lineage or task.name, task.name) for task in self.tasks))
        if identities != self.population.expected_instances:
            raise ValueError("prepared evaluation tasks differ from the sealed population")
        if self.target_access_receipt is not None and self.ground_certificate_authority is None:
            raise ValueError("prepared evaluation target access has no quality authority")


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def _nonnegative_int(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return result


def _lineage_cap(value: str) -> int:
    result = _positive_int(value)
    if result > DEFAULT_MAX_LINEAGES_PER_SHARD:
        raise argparse.ArgumentTypeError("value cannot exceed 32 lineages")
    return result


def _require_arguments(args: argparse.Namespace, names: Sequence[str], label: str) -> None:
    missing = [
        f"--{name.replace('_', '-')}"
        for name in names
        if getattr(args, name, None) is None
        or (isinstance(getattr(args, name, None), str) and not getattr(args, name).strip())
    ]
    if missing:
        raise ValueError(f"{label} required arguments are missing: {', '.join(missing)}")


def _workflow(value: object) -> EvaluationWorkflow:
    try:
        return EvaluationWorkflow(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported evaluation workflow {value!r}") from exc


def _validate_execution_environment_arguments(args: argparse.Namespace) -> None:
    _require_arguments(args, ("cluster", "execution_mode"), "execution environment")
    cluster = args.cluster
    command = getattr(args, "command", None)
    if cluster == "goose":
        _require_arguments(args, ("slurm_partition",), "Goose Slurm execution")
        if not os.environ.get("SLURM_JOB_ID"):
            raise ValueError("Goose evaluation must execute inside a Slurm allocation")
        if command == "run-evaluation-shard":
            array_index = os.environ.get("SLURM_ARRAY_TASK_ID")
            if not array_index:
                raise ValueError("Goose shard execution requires a Slurm array task")
            if array_index != str(getattr(args, "shard_index", "")):
                raise ValueError("Goose Slurm array index differs from --shard-index")
        observed_partition = os.environ.get("SLURM_JOB_PARTITION")
        if observed_partition != args.slurm_partition:
            raise ValueError("Goose Slurm partition differs from --slurm-partition compute class")
    elif cluster == "apollo":
        if getattr(args, "slurm_partition", None) is not None:
            raise ValueError("Apollo direct execution cannot declare a Slurm partition")
        if os.environ.get("SLURM_JOB_ID"):
            raise ValueError("Apollo direct execution cannot run inside Slurm")
    else:
        raise ValueError(f"unsupported evaluation cluster {cluster!r}")

    mode = args.execution_mode
    if mode == "pinned-venv":
        _require_arguments(
            args,
            ("environment_lock", "expected_environment_lock_sha256"),
            "pinned-venv execution",
        )
        if any(
            getattr(args, name, None) is not None
            for name in (
                "container_runtime",
                "expected_container_runtime_sha256",
                "container_image",
                "expected_container_image_sha256",
            )
        ):
            raise ValueError("pinned-venv execution cannot declare container pins")
    elif mode == "apptainer":
        _require_arguments(
            args,
            (
                "container_runtime",
                "expected_container_runtime_sha256",
                "container_image",
                "expected_container_image_sha256",
            ),
            "Apptainer execution",
        )
        if any(
            getattr(args, name, None) is not None
            for name in ("environment_lock", "expected_environment_lock_sha256")
        ):
            raise ValueError("Apptainer execution cannot declare a venv lock")
    elif mode == "bare-metal":
        if not getattr(args, "diagnostic_only", False):
            raise ValueError("bare-metal execution requires explicit --diagnostic-only")
        if any(
            getattr(args, name, None) is not None
            for name in (
                "environment_lock",
                "expected_environment_lock_sha256",
                "container_runtime",
                "expected_container_runtime_sha256",
                "container_image",
                "expected_container_image_sha256",
            )
        ):
            raise ValueError("bare-metal diagnostic execution cannot declare artifact pins")
    else:
        raise ValueError(f"unsupported evaluation execution mode {mode!r}")


def _validate_workflow_arguments(
    args: argparse.Namespace, *, target_mode: TargetMode
) -> EvaluationWorkflow:
    workflow = _workflow(getattr(args, "workflow", None))
    _require_arguments(
        args,
        ("grid", "corpus", "selector", "device", "threads", "deterministic"),
        "evaluation workflow",
    )
    _validate_execution_environment_arguments(args)
    if workflow is EvaluationWorkflow.LEARNED_COMPLETE:
        _require_arguments(
            args,
            (
                "config",
                "run_root",
                "rl_value_selection_receipt",
                "expected_selection_sha256",
                "index",
            ),
            "learned complete-system workflow",
        )
    elif workflow is EvaluationWorkflow.TUNED_STOCK_COMPLETE:
        _require_arguments(
            args,
            (
                "config",
                "learned_config",
                "tuning_registry",
                "expected_tuning_registry_sha256",
                "external_tuning_selection",
                "expected_external_tuning_selection_sha256",
                "index",
            ),
            "tuned stock complete-system workflow",
        )
    else:
        _require_arguments(
            args,
            (
                "registry",
                "expected_registry_sha256",
                "external_config",
                "learned_config",
                "index",
                "tuning_seed_index",
            ),
            "external validation-tuning workflow",
        )
    _require_arguments(
        args,
        (
            "quality_attestation",
            "expected_quality_attestation_digest",
            "expected_quality_publisher_id",
            "ground_certificate_root",
            "expected_ground_certificate_root_sha256",
        ),
        "ground-authorized evaluation",
    )
    if target_mode not in {"public", "ground"}:  # pragma: no cover - typed/API guard
        raise ValueError("target_mode must be 'public' or 'ground'")
    return workflow


def prepare_evaluation_workflow(
    args: argparse.Namespace, *, target_mode: TargetMode
) -> PreparedEvaluationWorkflow:
    """Authenticate and materialize one full workflow cell.

    ``public`` never opens evaluator targets.  ``ground`` opens exactly the workflow's
    ``val`` or ``test`` partition through its pinned ground-certificate root.
    """

    workflow = _validate_workflow_arguments(args, target_mode=target_mode)
    if workflow is EvaluationWorkflow.LEARNED_COMPLETE:
        return _prepare_learned(args, target_mode=target_mode)
    if workflow is EvaluationWorkflow.TUNED_STOCK_COMPLETE:
        return _prepare_tuned_stock(args, target_mode=target_mode)
    return _prepare_validation_tuning(args, target_mode=target_mode)


def _build_plan(
    prepared: PreparedEvaluationWorkflow, *, max_lineages_per_shard: int
) -> EvaluationExecutionPlan:
    return build_evaluation_plan(
        workflow=prepared.workflow,
        population=prepared.population,
        run_coordinates=prepared.run_coordinates,
        execution_contract=prepared.execution_contract,
        quality_authority=prepared.ground_certificate_authority,
        compute_class=prepared.compute_class,
        max_lineages_per_shard=max_lineages_per_shard,
    )


def _authenticate_prepared_plan(
    prepared: PreparedEvaluationWorkflow, plan: EvaluationExecutionPlan
) -> None:
    expected = _build_plan(prepared, max_lineages_per_shard=plan.max_lineages_per_shard)
    if expected.as_dict() != plan.as_dict():
        raise ValueError(
            "evaluation plan differs from the independently authenticated full population"
        )


def _authenticate_requested_workflow(plan: object, requested: EvaluationWorkflow) -> None:
    recorded = getattr(plan, "workflow", requested)
    if EvaluationWorkflow(recorded) is not requested:
        raise ValueError("requested workflow differs from the pinned evaluation plan")


def _require_ground_binding(prepared: PreparedEvaluationWorkflow) -> None:
    if (
        prepared.target_access_receipt is None
        or prepared.ground_certificate_authority is None
        or prepared.compute_class is None
        or prepared.compute_provenance is None
    ):
        raise ValueError(
            "run and merge require selected-partition and compute provenance authorities"
        )


def execute_evaluation_shard(
    prepared: PreparedEvaluationWorkflow,
    plan: EvaluationExecutionPlan,
    shard_index: int,
) -> tuple[TypedReceipt, ...]:
    """Authenticate the full population and run one whole-lineage shard."""

    _authenticate_prepared_plan(prepared, plan)
    select_tasks_for_shard(plan, shard_index, prepared.tasks)
    shard = plan.shard(shard_index)
    runtime = prepared.runtime
    if prepared.workflow is EvaluationWorkflow.LEARNED_COMPLETE:
        from isingfold.rl.complete_system import run_complete_system

        _, receipts = run_complete_system(
            prepared.tasks,
            prepared.context,
            runtime["initializer"],
            runtime["config"],
            controller=runtime["controller"],
            selector=runtime["selector"],
            controller_identity=runtime["controller_identity"],
            selector_identity=runtime["selector_identity"],
            population=prepared.population,
            seed=prepared.population.evaluation_seed,
            repetitions=prepared.population.expected_repetitions,
            max_steps=prepared.context.caps.decisions,
            execution_lineages=shard.lineages,
        )
    else:
        from isingfold.rl.external_pairing import run_external_complete_system

        _, receipts = run_external_complete_system(
            prepared.tasks,
            prepared.context,
            runtime["backend"],
            runtime["external_config"],
            learned_config=runtime["learned_config"],
            selector=runtime["selector"],
            selector_identity=runtime["selector_identity"],
            population=prepared.population,
            quality_authority=runtime["selector_quality_authority"],
            seed=prepared.population.evaluation_seed,
            repetitions=prepared.population.expected_repetitions,
            training_seed_index=runtime["training_seed_index"],
            training_seed=runtime["training_seed"],
            compute_identity=runtime["compute_identity"],
            tuning_execution=runtime["tuning_execution"],
            execution_lineages=shard.lineages,
        )
    return tuple(receipts)


def cmd_plan_evaluation_shards(args: argparse.Namespace) -> None:
    workflow = _validate_workflow_arguments(args, target_mode="public")
    prepared = prepare_evaluation_workflow(args, target_mode="public")
    if (
        prepared.target_access_receipt is not None
        or prepared.ground_certificate_authority is None
        or prepared.compute_class is None
    ):
        raise ValueError("evaluation planning requires target-free quality and compute authorities")
    plan = _build_plan(
        prepared,
        max_lineages_per_shard=int(args.max_lineages_per_shard),
    )
    pin = write_evaluation_plan(args.out, plan)
    print(
        json.dumps(
            {
                "workflow": workflow.value,
                "plan": str(args.out),
                "plan_sha256": pin,
                "publication_eligible": plan.publication_eligible,
            },
            indent=1,
        )
    )


def cmd_run_evaluation_shard(args: argparse.Namespace) -> None:
    workflow = _validate_workflow_arguments(args, target_mode="ground")
    plan = load_evaluation_plan(args.plan, expected_sha256=args.expected_plan_sha256)
    _authenticate_requested_workflow(plan, workflow)
    shard_index = int(args.shard_index)
    if shard_index not in range(len(plan.shards)):
        raise ValueError("requested shard index is outside the pinned plan")
    prepared = prepare_evaluation_workflow(args, target_mode="ground")
    _require_ground_binding(prepared)
    _authenticate_prepared_plan(prepared, plan)
    receipts = execute_evaluation_shard(prepared, plan, shard_index)
    receipt, pin = publish_evaluation_shard(
        args.out,
        plan=plan,
        shard_index=shard_index,
        receipts=receipts,
        target_access_receipt=prepared.target_access_receipt,
        ground_certificate_authority=prepared.ground_certificate_authority,
        compute_provenance=prepared.compute_provenance,
    )
    print(
        json.dumps(
            {
                "workflow": workflow.value,
                "shard": shard_index,
                "out": str(args.out),
                "shard_receipt_sha256": pin,
                "shard_receipt_record_digest": receipt["record_digest"],
                "attempts": len(receipts),
            },
            indent=1,
        )
    )


def cmd_merge_evaluation_shards(args: argparse.Namespace) -> None:
    workflow = _validate_workflow_arguments(args, target_mode="ground")
    plan = load_evaluation_plan(args.plan, expected_sha256=args.expected_plan_sha256)
    _authenticate_requested_workflow(plan, workflow)
    directories = tuple(args.shard)
    pins = tuple(args.expected_shard_receipt_sha256)
    if len(directories) != len(pins) or not directories or len(directories) != len(plan.shards):
        raise ValueError(
            "merge requires the complete plan shard census and one receipt pin per shard"
        )
    prepared = prepare_evaluation_workflow(args, target_mode="ground")
    _require_ground_binding(prepared)
    _authenticate_prepared_plan(prepared, plan)
    loaded = tuple(
        load_evaluation_shard(directory, plan=plan, expected_receipt_sha256=pin)
        for directory, pin in zip(directories, pins, strict=True)
    )
    expected_access = dict(prepared.target_access_receipt or {})
    expected_ground = dict(prepared.ground_certificate_authority or {})
    if any(
        dict(shard.target_access_receipt or {}) != expected_access
        or dict(shard.ground_certificate_authority or {}) != expected_ground
        for shard in loaded
    ):
        raise ValueError(
            "evaluation shard authority differs from the independently opened ground proof"
        )
    merged = merge_evaluation_shards(
        plan,
        loaded,
        tasks=prepared.tasks,
        context=prepared.context,
    )
    receipt, pin = publish_merged_evaluation(args.out, plan=plan, merged=merged)
    print(
        json.dumps(
            {
                "workflow": workflow.value,
                "out": str(args.out),
                "merge_receipt_sha256": pin,
                "merge_receipt_record_digest": receipt["record_digest"],
                "attempts": len(getattr(merged, "receipts", ())),
                "report_inputs": dict(getattr(merged, "recomputed_report_inputs", {})),
            },
            indent=1,
            default=float,
        )
    )


def _add_common_workflow_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workflow", required=True, choices=tuple(item.value for item in EvaluationWorkflow)
    )
    parser.add_argument("--grid", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--selector", required=True)
    parser.add_argument("--qubit-cap", type=_positive_int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--threads", type=_positive_int, default=1)
    parser.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cluster", required=True, choices=("apollo", "goose"))
    parser.add_argument("--slurm-partition")
    parser.add_argument(
        "--execution-mode",
        required=True,
        choices=("apptainer", "pinned-venv", "bare-metal"),
    )
    parser.add_argument("--environment-lock")
    parser.add_argument("--expected-environment-lock-sha256")
    parser.add_argument("--container-runtime")
    parser.add_argument("--expected-container-runtime-sha256")
    parser.add_argument("--container-image")
    parser.add_argument("--expected-container-image-sha256")
    parser.add_argument("--diagnostic-only", action="store_true")

    # Learned complete-system cell.
    parser.add_argument("--config")
    parser.add_argument("--run-root")
    parser.add_argument("--rl-value-selection-receipt")
    parser.add_argument("--expected-selection-sha256")

    # Tuned stock deployment cell.
    parser.add_argument("--learned-config")
    parser.add_argument("--tuning-registry")
    parser.add_argument("--expected-tuning-registry-sha256")
    parser.add_argument("--external-tuning-selection")
    parser.add_argument("--expected-external-tuning-selection-sha256")

    # Validation-tuning candidate/seed cell.
    parser.add_argument("--registry")
    parser.add_argument("--expected-registry-sha256")
    parser.add_argument("--external-config")
    parser.add_argument("--index", type=_nonnegative_int)
    parser.add_argument("--tuning-seed-index", type=_nonnegative_int)


def _add_ground_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--quality-attestation", required=True)
    parser.add_argument("--expected-quality-attestation-digest", required=True)
    parser.add_argument("--expected-quality-publisher-id", required=True)
    parser.add_argument("--ground-certificate-root", required=True)
    parser.add_argument("--expected-ground-certificate-root-sha256", required=True)


def register_evaluation_shard_commands(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Register the three thin commands on the repository's root CLI parser."""

    plan = subparsers.add_parser(
        "plan-evaluation-shards",
        help="seal a full public-only evaluation population into lineage shards",
    )
    _add_common_workflow_arguments(plan)
    _add_ground_arguments(plan)
    plan.add_argument(
        "--max-lineages-per-shard",
        type=_lineage_cap,
        default=DEFAULT_MAX_LINEAGES_PER_SHARD,
    )
    plan.add_argument("--out", required=True)
    plan.set_defaults(func=cmd_plan_evaluation_shards)

    run = subparsers.add_parser(
        "run-evaluation-shard",
        help="run one pinned lineage shard after selected-partition ground authorization",
    )
    _add_common_workflow_arguments(run)
    _add_ground_arguments(run)
    run.add_argument("--plan", required=True)
    run.add_argument("--expected-plan-sha256", required=True)
    run.add_argument("--shard-index", required=True, type=_nonnegative_int)
    run.add_argument("--out", required=True)
    run.set_defaults(func=cmd_run_evaluation_shard)

    merge = subparsers.add_parser(
        "merge-evaluation-shards",
        help="authenticate a complete shard census and recompute canonical report inputs",
    )
    _add_common_workflow_arguments(merge)
    _add_ground_arguments(merge)
    merge.add_argument("--plan", required=True)
    merge.add_argument("--expected-plan-sha256", required=True)
    merge.add_argument("--shard", action="append", required=True)
    merge.add_argument("--expected-shard-receipt-sha256", action="append", required=True)
    merge.add_argument("--out", required=True)
    merge.set_defaults(func=cmd_merge_evaluation_shards)


def _grid_protocol(args: argparse.Namespace):
    from isingfold.rl import cli as core

    grid, grid_digest = core._load_grid(args.grid)
    protocol = grid.get("complete_system_evaluation")
    if not isinstance(protocol, Mapping):
        raise ValueError("staged grid has no complete-system evaluation protocol")
    required = {
        "partition",
        "evaluation_seed",
        "repetitions",
        "audit_reads",
        "feasibility_noninferiority_margin",
    }
    if not required.issubset(protocol):
        raise ValueError("complete-system evaluation protocol fields are incomplete")
    return grid, grid_digest, protocol


def _selector_bundle_and_identity(args: argparse.Namespace):
    from isingfold.rl import cli as core
    from isingfold.rl.complete_system import FrozenComponentIdentity

    bundle = core._load_selector_bundle(args.selector, corpus=args.corpus)
    selector_fit_path = bundle.root / "fit_receipt.json"
    selector_fit = core._strict_json(selector_fit_path)
    core._verify_record(selector_fit, "selector fit receipt")
    identity = FrozenComponentIdentity(
        component_id="isingfold-if-q3-s0-strength-selector",
        version=str(bundle.model.version),
        implementation=(
            f"{type(bundle.model).__module__}.{type(bundle.model).__qualname__}:"
            f"fit-{selector_fit['record_digest']}"
        ),
        artifact_sha256=core._sha256_file(bundle.root / "selector.pt"),
    )
    return bundle, identity


def _validate_ground_selector_publisher(
    bundle: object,
    ground_certificate_authority: Mapping[str, object] | None,
) -> None:
    if ground_certificate_authority is None:
        raise ValueError("evaluation workflow has no precommitted quality authority")
    from isingfold.rl import cli as core

    bundle_authority = getattr(bundle, "quality_authority", None)
    if not isinstance(bundle_authority, Mapping):
        raise ValueError("selector has no training quality authority")
    training = core._validate_quality_authority_binding(
        dict(bundle_authority), expected_role="training_partition"
    )
    evaluation = core._validate_quality_authority_binding(
        dict(ground_certificate_authority), expected_role="evaluation_partition"
    )
    if training["global"] != evaluation["global"]:
        raise ValueError("selector training and evaluation authorities use different global roots")


def _population_inputs(
    args: argparse.Namespace,
    *,
    protocol: Mapping[str, object],
    partition: str,
    repetitions: int,
    population_prefix: str,
    target_mode: TargetMode,
):
    from isingfold.rl import cli as core
    from isingfold.rl.complete_system import (
        CompletePopulationIdentity,
        task_population_digest,
    )
    from isingfold.rl.data.prepared import load_prepared_corpus_design
    from isingfold.rl.evaluation_strata import evaluation_contract_from_prepared

    manifest_path = Path(args.corpus) / "manifest.json"
    manifest = core._strict_json(manifest_path)
    manifest_sha256 = core._sha256_file(manifest_path)
    prepared, target_access, ground_authority = _open_evaluation_partition(
        args,
        partition=partition,
        target_mode=target_mode,
    )
    tasks = tuple(
        core._preinitialization_population_tasks(
            prepared,
            manifest=manifest,
            partition=partition,
        )
    )
    strata, design = evaluation_contract_from_prepared(
        prepared,
        corpus_design_receipt=load_prepared_corpus_design(args.corpus),
        partition=partition,
        noninferiority_margin=float(protocol["feasibility_noninferiority_margin"]),
    )
    evaluation_seed = int(protocol["evaluation_seed"])
    population = CompletePopulationIdentity(
        population_id=(
            population_prefix
            + core.stable_digest(
                {
                    "manifest": manifest_sha256,
                    "partition": partition,
                    "evaluation_seed": evaluation_seed,
                    "repetitions": repetitions,
                }
            )
        ),
        source_manifest_sha256=manifest_sha256,
        task_payload_sha256=task_population_digest(tasks),
        expected_instances=tuple(sorted((task.lineage or task.name, task.name) for task in tasks)),
        expected_repetitions=repetitions,
        evaluation_seed=evaluation_seed,
        evaluation_strata=strata,
        confirmatory_design=design,
    )
    return prepared, tasks, population, target_access, ground_authority


def _prepare_learned(
    args: argparse.Namespace, *, target_mode: TargetMode
) -> PreparedEvaluationWorkflow:
    from isingfold.rl import cli as core
    from isingfold.rl.complete_system import (
        CompleteSystemConfig,
        FrozenComponentIdentity,
        LACMinorminerInitializerBackend,
        _context_digest,
        complete_policy_context,
    )
    from isingfold.rl.evaluate import torch_controller
    from isingfold.rl.evaluation_shards import learned_execution_contract
    from isingfold.rl.experiment_selection import load_rl_value_freeze

    _, grid_digest, protocol = _grid_protocol(args)
    partition = str(protocol["partition"])
    if partition != "test":
        raise ValueError("learned complete-system sharding is fixed to the test partition")
    repetitions = int(protocol["repetitions"])
    evaluation_seed = int(protocol["evaluation_seed"])

    config_path = Path(args.config)
    config = CompleteSystemConfig.from_mapping(core._strict_json(config_path))
    core._validate_registered_complete_config(
        protocol,
        arm="learned",
        semantic_digest=config.digest,
        file_sha256=core._sha256_file(config_path),
    )
    selection = load_rl_value_freeze(
        receipt_path=args.rl_value_selection_receipt,
        grid_path=args.grid,
        expected_sha256=args.expected_selection_sha256,
    )
    if selection.grid_manifest_sha256 != grid_digest:
        raise ValueError("learned selection and staged grid identities differ")
    source_cell_id, training_seed, _ = selection.cell_for_index(args.index)
    context = core._context(args, args.corpus)
    if config.audit_reads != context.audit_reads or config.audit_reads != protocol["audit_reads"]:
        raise ValueError("learned config, grid and environment audit reads differ")
    bundle, selector_identity = _selector_bundle_and_identity(args)
    device = core._resolve_device(args.device)
    core._seed_runtime(
        evaluation_seed,
        deterministic=args.deterministic,
        threads=args.threads,
    )
    compute_class, compute_provenance = _compute_class_and_provenance(args, device=device)
    model, training_receipt, checkpoint = core._load_complete_evaluation_model(
        args, bundle, selection
    )
    selector_quality_authority = core._validate_quality_authority_binding(
        dict(bundle.quality_authority), expected_role="training_partition"
    )
    checkpoint_quality_authority = core._validate_quality_authority_binding(
        training_receipt["experiment_contract"].get("quality_authority"),
        expected_role="training_partition",
    )
    if checkpoint_quality_authority != selector_quality_authority:
        raise ValueError("learned checkpoint uses a different selector quality authority")
    model.to(device).eval()
    bundle.model.to(device).eval()
    initializer = LACMinorminerInitializerBackend()
    controller_identity = FrozenComponentIdentity(
        component_id=(
            f"isingfold-policy/{selection.model_family}/{selection.method}/seed-{training_seed}"
        ),
        version=str(training_receipt["checkpoint_payload_digest"]),
        implementation=(
            f"{training_receipt['experiment_contract']['model']['implementation']}:"
            f"runtime-{training_receipt['runtime_implementation_digest']}"
        ),
        artifact_sha256=core._sha256_file(checkpoint),
    )
    _, tasks, population, target_access, ground_authority = _population_inputs(
        args,
        protocol=protocol,
        partition=partition,
        repetitions=repetitions,
        population_prefix="preinitialization-",
        target_mode=target_mode,
    )
    _validate_ground_selector_publisher(bundle, ground_authority)
    contract = learned_execution_contract(
        initializer=initializer.identity.as_dict(),
        controller=controller_identity.as_dict(),
        selector=selector_identity.as_dict(),
        config_digest=config.digest,
        context_digest=_context_digest(context),
        policy_context_digest=_context_digest(complete_policy_context(context)),
        work_cap=context.caps.as_dict(),
        wallclock_cap_seconds=config.online_wallclock_seconds,
    )
    return PreparedEvaluationWorkflow(
        workflow=EvaluationWorkflow.LEARNED_COMPLETE,
        population=population,
        tasks=tasks,
        context=context,
        run_coordinates={
            "training_seed_index": args.index,
            "training_seed": training_seed,
            "source_cell_id": source_cell_id,
        },
        execution_contract=contract,
        target_access_receipt=target_access,
        ground_certificate_authority=ground_authority,
        compute_class=compute_class,
        compute_provenance=compute_provenance,
        runtime={
            "initializer": initializer,
            "config": config,
            "controller": torch_controller(model, device=device, greedy=False),
            "selector": bundle.model,
            "controller_identity": controller_identity,
            "selector_identity": selector_identity,
        },
    )


def _external_components(
    args: argparse.Namespace,
    *,
    protocol: Mapping[str, object],
    external_config_path: Path,
    runtime_seed: int,
) -> dict[str, Any]:
    from isingfold.rl import cli as core
    from isingfold.rl.complete_system import CompleteSystemConfig
    from isingfold.rl.external import StockMinorminerBackend
    from isingfold.rl.external_pairing import (
        ExternalCompleteSystemConfig,
        context_snapshot,
    )

    external_config = ExternalCompleteSystemConfig.from_mapping(
        core._strict_json(external_config_path)
    )
    learned_config_path = Path(args.learned_config)
    learned_config = CompleteSystemConfig.from_mapping(core._strict_json(learned_config_path))
    core._validate_registered_complete_config(
        protocol,
        arm="external",
        semantic_digest=external_config.digest,
        file_sha256=core._sha256_file(external_config_path),
    )
    core._validate_registered_complete_config(
        protocol,
        arm="learned",
        semantic_digest=learned_config.digest,
        file_sha256=core._sha256_file(learned_config_path),
    )
    context = core._context(args, args.corpus)
    external_config.validate_symmetric_envelope(learned_config, context)
    if external_config.audit_reads != protocol["audit_reads"]:
        raise ValueError("external config and grid audit-read blocks differ")
    if args.deterministic is not True or args.threads != 1:
        raise ValueError(
            "publication external evaluation requires deterministic one-thread inference"
        )
    bundle, selector_identity = _selector_bundle_and_identity(args)
    device = core._resolve_device(args.device)
    core._seed_runtime(runtime_seed, deterministic=True, threads=1)
    bundle.model.to(device).eval()
    availability = StockMinorminerBackend.probe(
        expected_version=external_config.expected_backend_version
    )
    if not availability.available or availability.identity is None:
        raise RuntimeError(
            "the preregistered stock-minorminer backend is unavailable; no fallback exists"
        )
    backend = StockMinorminerBackend(availability.identity)
    compute_class, compute_provenance = _compute_class_and_provenance(args, device=device)
    compute_identity = compute_provenance["runtime_identity"]
    return {
        "external_config": external_config,
        "learned_config": learned_config,
        "context": context,
        "context_snapshot": context_snapshot(context),
        "bundle": bundle,
        "selector": bundle.model,
        "selector_identity": selector_identity,
        "backend": backend,
        "backend_identity": availability.identity,
        "compute_identity": compute_identity,
        "compute_class": compute_class,
        "compute_provenance": compute_provenance,
    }


def _external_contract(
    components: Mapping[str, Any],
    *,
    selector_quality_authority: Mapping[str, object],
    training_seed_index: int,
    training_seed: int,
    tuning_execution: object,
) -> Mapping[str, object]:
    from isingfold.rl.contracts import stable_digest
    from isingfold.rl.data.import_embedbench import content_digest
    from isingfold.rl.evaluation_shards import external_execution_contract

    context = components["context"]
    external_config = components["external_config"]
    learned_config = components["learned_config"]
    selector_identity = components["selector_identity"]
    backend_identity = components["backend_identity"]
    return external_execution_contract(
        backend=backend_identity.as_dict(),
        selector=selector_identity.as_dict(),
        config_digest=external_config.digest,
        learned_config_digest=learned_config.digest,
        context_digest=stable_digest(components["context_snapshot"]),
        quality_authority_digest=content_digest(selector_quality_authority),
        work_cap=context.caps.as_dict(),
        wallclock_cap_seconds=external_config.online_wallclock_seconds,
        training_seed_index=training_seed_index,
        training_seed=training_seed,
        tuning_execution=tuning_execution.as_dict(),
    )


def _prepare_tuned_stock(
    args: argparse.Namespace, *, target_mode: TargetMode
) -> PreparedEvaluationWorkflow:
    from isingfold.rl import cli as core
    from isingfold.rl.checkpoint import runtime_implementation_registry
    from isingfold.rl.data.import_embedbench import content_digest
    from isingfold.rl.external_tuning import (
        ExternalTuningExecutionBinding,
        load_external_tuning_registry,
        load_external_tuning_selection,
    )

    _, grid_digest, protocol = _grid_protocol(args)
    partition = str(protocol["partition"])
    if partition != "test":
        raise ValueError("tuned stock complete-system sharding is fixed to test")
    registry = load_external_tuning_registry(
        args.tuning_registry,
        expected_file_sha256=args.expected_tuning_registry_sha256,
        grid_path=args.grid,
        external_config_path=args.config,
    )
    selection = load_external_tuning_selection(
        args.external_tuning_selection,
        expected_file_sha256=args.expected_external_tuning_selection_sha256,
        registry=registry,
    )
    if args.index not in range(3):
        raise ValueError("tuned stock complete-system index must be zero through two")
    training_seed = (1103, 2207, 3301)[args.index]
    runtime_registry = runtime_implementation_registry()
    if content_digest(runtime_registry) != selection.runtime_implementation_digest:
        raise ValueError("runtime sources differ from validation baseline tuning")
    manifest_path = Path(args.corpus) / "manifest.json"
    if core._sha256_file(manifest_path) != selection.source_manifest_sha256:
        raise ValueError("publication corpus differs from validation baseline tuning")

    components = _external_components(
        args,
        protocol=protocol,
        external_config_path=Path(args.config),
        runtime_seed=int(protocol["evaluation_seed"]),
    )
    context = components["context"]
    bundle = components["bundle"]
    selector_identity = components["selector_identity"]
    if core.stable_digest(components["context_snapshot"]) != selection.context_digest:
        raise ValueError("publication context differs from validation baseline tuning")
    if content_digest(selector_identity.as_dict()) != selection.selector_digest:
        raise ValueError("publication selector differs from validation baseline tuning")
    selector_quality_authority = core._validate_quality_authority_binding(
        dict(bundle.quality_authority), expected_role="training_partition"
    )
    if content_digest(selector_quality_authority) != selection.quality_authority_digest:
        raise ValueError("publication quality authority differs from validation tuning")
    tuning_execution = ExternalTuningExecutionBinding.for_deployment(selection)
    repetitions = int(protocol["repetitions"])
    _, tasks, population, target_access, ground_authority = _population_inputs(
        args,
        protocol=protocol,
        partition=partition,
        repetitions=repetitions,
        population_prefix="preinitialization-",
        target_mode=target_mode,
    )
    _validate_ground_selector_publisher(bundle, ground_authority)
    contract = _external_contract(
        components,
        selector_quality_authority=ground_authority,
        training_seed_index=args.index,
        training_seed=training_seed,
        tuning_execution=tuning_execution,
    )
    return PreparedEvaluationWorkflow(
        workflow=EvaluationWorkflow.TUNED_STOCK_COMPLETE,
        population=population,
        tasks=tasks,
        context=context,
        run_coordinates={
            "training_seed_index": args.index,
            "training_seed": training_seed,
        },
        execution_contract=contract,
        target_access_receipt=target_access,
        ground_certificate_authority=ground_authority,
        compute_class=components["compute_class"],
        compute_provenance=components["compute_provenance"],
        runtime={
            **components,
            "selector_quality_authority": ground_authority,
            "training_quality_authority": selector_quality_authority,
            "training_seed_index": args.index,
            "training_seed": training_seed,
            "tuning_execution": tuning_execution,
        },
    )


def _prepare_validation_tuning(
    args: argparse.Namespace, *, target_mode: TargetMode
) -> PreparedEvaluationWorkflow:
    from isingfold.rl import cli as core
    from isingfold.rl.external_tuning import (
        ExternalTuningExecutionBinding,
        load_external_tuning_registry,
    )

    _, grid_digest, protocol = _grid_protocol(args)
    registry = load_external_tuning_registry(
        args.registry,
        expected_file_sha256=args.expected_registry_sha256,
        grid_path=args.grid,
        external_config_path=args.external_config,
    )
    if grid_digest != registry.grid_file_sha256:
        raise ValueError("external tuning registry and staged grid identities differ")
    candidate = registry.candidate_for_index(args.index)
    tuning_seed = registry.seed_for_index(args.tuning_seed_index)
    components = _external_components(
        args,
        protocol=protocol,
        external_config_path=Path(args.external_config),
        runtime_seed=tuning_seed,
    )
    external_config = components["external_config"]
    context = components["context"]
    bundle = components["bundle"]
    if (
        external_config.audit_reads != registry.audit_reads
        or external_config.online_wallclock_seconds != registry.online_wallclock_seconds
    ):
        raise ValueError("external config differs from the tuning registry envelope")
    selector_quality_authority = core._validate_quality_authority_binding(
        dict(bundle.quality_authority), expected_role="training_partition"
    )
    tuning_execution = ExternalTuningExecutionBinding.for_validation(registry, args.index)
    _, tasks, population, target_access, ground_authority = _population_inputs(
        args,
        protocol=protocol,
        partition="val",
        repetitions=registry.repetitions,
        population_prefix="external-tuning-validation-",
        target_mode=target_mode,
    )
    _validate_ground_selector_publisher(bundle, ground_authority)
    contract = _external_contract(
        components,
        selector_quality_authority=ground_authority,
        training_seed_index=args.tuning_seed_index,
        training_seed=tuning_seed,
        tuning_execution=tuning_execution,
    )
    return PreparedEvaluationWorkflow(
        workflow=EvaluationWorkflow.VALIDATION_TUNING,
        population=population,
        tasks=tasks,
        context=context,
        run_coordinates={
            "candidate_index": args.index,
            "candidate_id": candidate.candidate_id,
            "tuning_seed_index": args.tuning_seed_index,
            "tuning_seed": tuning_seed,
        },
        execution_contract=contract,
        target_access_receipt=target_access,
        ground_certificate_authority=ground_authority,
        compute_class=components["compute_class"],
        compute_provenance=components["compute_provenance"],
        runtime={
            **components,
            "selector_quality_authority": ground_authority,
            "training_quality_authority": selector_quality_authority,
            "training_seed_index": args.tuning_seed_index,
            "training_seed": tuning_seed,
            "tuning_execution": tuning_execution,
        },
    )


__all__ = [
    "PreparedEvaluationWorkflow",
    "cmd_merge_evaluation_shards",
    "cmd_plan_evaluation_shards",
    "cmd_run_evaluation_shard",
    "execute_evaluation_shard",
    "prepare_evaluation_workflow",
    "register_evaluation_shard_commands",
]
