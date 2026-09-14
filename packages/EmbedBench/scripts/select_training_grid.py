#!/usr/bin/env python3
"""Replay a complete validation-only grid and freeze one paper checkpoint.

GPU validation remains authenticated checkpoint-generation provenance. Every saved checkpoint
is then evaluated by one canonical deterministic CPU replay, and only those replay metrics enter
configuration or seed selection. The winning configuration minimizes its arithmetic mean over
every registered seed. Registered order breaks ties. Audit labels and test records are never
encoded or used for selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple

import run_training_grid as grid_runner
from quality_v2_paper_contract import (
    AUDIT_SOURCE_HASH_ALGORITHM,
    PaperAuditContract,
    audit_source_sha256,
    load_paper_audit_contract,
    require_exact_json,
    strict_json_loads,
    validate_runtime_provenance,
)
from training_splits import quality_problem_digest

SELECTION_SCHEMA = "embedbench.training-grid-selection"
SELECTION_SCHEMA_VERSION = 3
PRIMARY_METRIC = "mean_finite_budget_regret"
REGISTERED_BUDGET_RATIOS: tuple[float | None, ...] = (1.0, 1.1, 1.25, 1.5, None)
SELECTION_CONTRACT = "quality_mean_or_lcb_subject_to_exact_feasibility_and_budget-v1"
CANONICAL_VALIDATION_REPLAY_POLICY: dict[str, object] = {
    "schema": "embedbench.canonical-validation-replay-policy",
    "schema_version": 1,
    "device": "cpu",
    "model_parameter_dtype": "float32",
    "intraop_threads": 1,
    "interop_threads": 1,
    "deterministic_algorithms": True,
    "float32_matmul_precision": "highest",
    "selection_metric_source": "canonical_cpu_checkpoint_replay",
    "training_device_metric_role": (
        "authenticated_training_time_best_epoch_provenance_only_never_cross_cell_selection"
    ),
    "cross_device_sweep_equality_required": False,
}
_CANONICAL_REPLAY_RUNTIME_READY = False
_REGISTERED_VALUE_OPTIONS: dict[str, tuple[str, type]] = {
    "--hidden": ("hidden", int),
    "--layers": ("layers", int),
    "--heads": ("heads", int),
    "--lr": ("lr", float),
    "--tol": ("tol", float),
    "--rank-margin": ("rank_margin", float),
    "--stage1-rank-margin": ("stage1_rank_margin", float),
    "--lcb-z": ("lcb_z", float),
    "--stage1-weight": ("stage1_weight", float),
    "--lambda-connectivity": ("lambda_connectivity", float),
    "--lambda-robustness": ("lambda_robustness", float),
    "--deploy-max-free": ("deploy_max_free", int),
    "--evaluation-support": ("evaluation_support", str),
}
_REGISTERED_BOOLEAN_OPTIONS = {
    "--neighbour-feats": "neighbour_feats",
    "--deploy-view": "deploy_view",
}


class _TrainingPreflight(NamedTuple):
    document: dict[str, object]
    stage: Mapping[str, object]
    registration: Mapping[str, object]
    grid_sha256: str
    source_sha256: str


def _configure_canonical_replay_runtime() -> dict[str, object]:
    """Pin the process-wide Torch controls used by every validation replay."""

    import torch

    global _CANONICAL_REPLAY_RUNTIME_READY
    if not _CANONICAL_REPLAY_RUNTIME_READY:
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError as error:
                raise ValueError(
                    "canonical validation replay requires a fresh Python process so Torch "
                    "interop threads can be pinned before parallel work"
                ) from error
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        _CANONICAL_REPLAY_RUNTIME_READY = True
    observed = {
        **CANONICAL_VALIDATION_REPLAY_POLICY,
        "intraop_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    require_exact_json(
        observed,
        CANONICAL_VALIDATION_REPLAY_POLICY,
        location="canonical validation replay runtime",
    )
    return observed


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"artifact path is outside the staged root: {path}") from error


def _expected_paths(
    stage: Mapping[str, object], cell: grid_runner.GridCell, root: Path
) -> tuple[Path, Path, Path]:
    output_dir = stage.get("output_dir")
    checkpoint_dir = stage.get("checkpoint_dir")
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("grid stage output_dir must be a non-empty path string")
    if not isinstance(checkpoint_dir, str) or not checkpoint_dir:
        raise ValueError("grid stage checkpoint_dir must be a non-empty path string")
    result = root / output_dir / f"{cell.cell_id}.json"
    receipt = result.with_suffix(".receipt.json")
    checkpoint_prefix = root / checkpoint_dir / cell.cell_id
    checkpoint = grid_runner._expected_checkpoint(checkpoint_prefix, cell)
    return result, receipt, checkpoint


def _read_object(path: Path, kind: str) -> dict[str, object]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"invalid {kind} {path}") from error
    return _object_from_payload(payload, path=path, kind=kind)


def _object_from_payload(payload: bytes, *, path: Path, kind: str) -> dict[str, object]:
    document = strict_json_loads(payload, location=f"{kind} {path}")
    if not isinstance(document, dict):
        raise ValueError(f"{kind} must be a JSON object: {path}")
    return document


def _payload_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_full_training_scope(value: object) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("limit") is None
        and value.get("full_fixed_train_validation") is True
    )


def _runtime_compatibility_signature(
    value: object, *, expected_device_type: object, cell_id: str
) -> dict[str, object]:
    try:
        validated = validate_runtime_provenance(
            value,
            location=f"cell {cell_id}",
            expected_device=(
                str(expected_device_type) if expected_device_type is not None else None
            ),
        )
    except ValueError as error:
        raise ValueError(f"cell {cell_id} has invalid runtime provenance: {error}") from error
    python = validated["python"]
    torch = validated["torch"]
    packages = validated["packages"]
    assert isinstance(python, Mapping)
    assert isinstance(torch, Mapping)
    assert isinstance(packages, Mapping)
    return {
        "python_implementation": python.get("implementation"),
        "python_version": python.get("version"),
        "torch_version": torch.get("version"),
        "cuda_runtime_version": torch.get("cuda_runtime_version"),
        "cudnn_version": torch.get("cudnn_version"),
        "packages": dict(packages),
    }


def _expected_preprocessing(
    stage: Mapping[str, object], *, architecture: object
) -> dict[str, object]:
    from embedbench.hamiltonian_context import hamiltonian_context_contract

    extra_args = stage.get("extra_args", [])
    if not isinstance(extra_args, list):
        raise ValueError("stage extra_args must be a list")

    def integer_option(name: str, default: int) -> int:
        if name not in extra_args:
            return default
        index = extra_args.index(name)
        if index + 1 >= len(extra_args):
            raise ValueError(f"stage {name} has no value")
        try:
            value = int(extra_args[index + 1])
        except (TypeError, ValueError) as error:
            raise ValueError(f"stage {name} must be an integer") from error
        if value < 0:
            raise ValueError(f"stage {name} must be non-negative")
        return value

    return {
        "deploy_view": "--deploy-view" in extra_args,
        "deploy_max_free": integer_option("--deploy-max-free", 8),
        "neighbour_feats": "--neighbour-feats" in extra_args,
        "encoder": "heterogeneous" if architecture == "hetero" else "chain",
        "hamiltonian_context": hamiltonian_context_contract(),
    }


def _registered_trainer_args(stage: Mapping[str, object]) -> dict[str, object]:
    """Parse a complete, explicitly registered non-axis trainer CLI contract."""

    extra_args = stage.get("extra_args")
    if not isinstance(extra_args, list) or not all(isinstance(value, str) for value in extra_args):
        raise ValueError("stage extra_args must be a list of strings")
    parsed: dict[str, object] = {
        "neighbour_feats": False,
        "deploy_view": False,
        "limit": None,
    }
    seen: set[str] = set()
    index = 0
    while index < len(extra_args):
        option = extra_args[index]
        if option in seen:
            raise ValueError(f"stage repeats registered trainer option {option}")
        seen.add(option)
        if option in _REGISTERED_BOOLEAN_OPTIONS:
            parsed[_REGISTERED_BOOLEAN_OPTIONS[option]] = True
            index += 1
            continue
        specification = _REGISTERED_VALUE_OPTIONS.get(option)
        if specification is None:
            raise ValueError(f"stage has unsupported or unregistered trainer option {option}")
        if index + 1 >= len(extra_args) or extra_args[index + 1].startswith("--"):
            raise ValueError(f"stage trainer option {option} has no explicit value")
        destination, converter = specification
        raw_value = extra_args[index + 1]
        try:
            parsed[destination] = converter(raw_value)
        except ValueError as error:
            raise ValueError(f"stage trainer option {option} has an invalid value") from error
        index += 2

    missing = sorted(set(_REGISTERED_VALUE_OPTIONS) - seen)
    if missing:
        raise ValueError(
            f"stage must freeze every non-axis trainer hyperparameter; missing options: {missing}"
        )
    for name in ("hidden", "layers", "heads"):
        value = parsed[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"registered trainer {name} must be a positive integer")
    deploy_max_free = parsed["deploy_max_free"]
    if (
        isinstance(deploy_max_free, bool)
        or not isinstance(deploy_max_free, int)
        or deploy_max_free < 0
    ):
        raise ValueError("registered trainer deploy_max_free must be non-negative")
    for name in (
        "lr",
        "tol",
        "rank_margin",
        "stage1_rank_margin",
        "lcb_z",
        "stage1_weight",
        "lambda_connectivity",
        "lambda_robustness",
    ):
        value = parsed[name]
        if not isinstance(value, float) or not math.isfinite(value) or value < 0.0:
            raise ValueError(f"registered trainer {name} must be finite and non-negative")
    if parsed["lr"] <= 0.0:
        raise ValueError("registered trainer lr must be positive")
    if parsed["stage1_rank_margin"] < parsed["rank_margin"]:
        raise ValueError("registered stage1 rank margin must cover the stage2 rank margin")
    if parsed["evaluation_support"] != "full":
        raise ValueError("registered trainer evaluation support must be full")
    return parsed


def _relocated_path_matches(actual: object, expected: Path, root: Path) -> bool:
    if not isinstance(actual, str) or not actual:
        return False
    try:
        # Preserve the staged logical path.  Corpus directories are symlinked into frozen
        # Apollo/Goose source trees, so resolve() would escape the staged root and make two
        # semantically identical commands compare unequal.
        expected_absolute = Path(os.path.abspath(expected))
        root_absolute = Path(os.path.abspath(root))
        relative = expected_absolute.relative_to(root_absolute)
    except ValueError:
        return False
    actual_parts = Path(actual).parts
    relative_parts = relative.parts
    return (
        bool(relative_parts)
        and len(actual_parts) >= len(relative_parts)
        and (actual_parts[-len(relative_parts) :] == relative_parts)
    )


def _registered_device(stage: Mapping[str, object]) -> str:
    execution = stage.get("execution")
    device = execution.get("required_device_type") if isinstance(execution, Mapping) else None
    if device not in {"cpu", "cuda"}:
        raise ValueError("selection requires a registered cpu/cuda trainer device")
    return str(device)


def _validate_receipt_command(
    value: object,
    *,
    document: Mapping[str, object],
    stage: Mapping[str, object],
    cell: grid_runner.GridCell,
    root: Path,
) -> None:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(token, str) and token for token in value)
        or not Path(value[0]).name.startswith("python")
    ):
        raise ValueError(f"cell {cell.cell_id} has an invalid receipt command")
    device = _registered_device(stage)
    expected, _, _ = grid_runner.build_command(
        document,
        cell,
        root=root,
        python=value[0],
        device=device,
    )
    inputs, _ = grid_runner._stage_inputs(stage, root)
    path_positions = {1, *range(2, 2 + len(inputs))}
    for option in ("--splits", "--save", "--out"):
        path_positions.add(expected.index(option) + 1)
    if len(value) != len(expected):
        raise ValueError(f"cell {cell.cell_id} receipt command length changed")
    for index, (actual, registered) in enumerate(zip(value, expected, strict=True)):
        matches = (
            _relocated_path_matches(actual, Path(registered), root)
            if index in path_positions
            else actual == registered
        )
        if not matches:
            raise ValueError(
                f"cell {cell.cell_id} receipt command changed at semantic argv {index}"
            )


def _same_arg_value(actual: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return actual is expected
    if expected is None:
        return actual is None
    if isinstance(expected, int):
        return isinstance(actual, int) and not isinstance(actual, bool) and actual == expected
    if isinstance(expected, float):
        return (
            isinstance(actual, float)
            and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=0.0, abs_tol=0.0)
        )
    return type(actual) is type(expected) and actual == expected


def _validate_result_args(
    value: object,
    *,
    stage: Mapping[str, object],
    cell: grid_runner.GridCell,
    root: Path,
    result_path: Path,
    checkpoint_path: Path,
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"cell {cell.cell_id} result args are missing")
    registered = _registered_trainer_args(stage)
    inputs, split_manifest = grid_runner._stage_inputs(stage, root)
    epochs = stage.get("epochs")
    checkpoint_suffix = f"_{cell.params['arch']}_s{cell.params['seed']}.pt"
    checkpoint_text = str(checkpoint_path)
    if not checkpoint_text.endswith(checkpoint_suffix):
        raise RuntimeError("registered checkpoint naming contract changed")
    expected: dict[str, object] = {
        "files": [str(path) for path in inputs],
        "arch": cell.params["arch"],
        "objective_variant": cell.params["objective_variant"],
        "seeds": str(cell.params["seed"]),
        "epochs": epochs,
        **registered,
        "device": _registered_device(stage),
        "splits": str(split_manifest),
        "save": checkpoint_text.removesuffix(checkpoint_suffix),
        "out": str(result_path),
    }
    if set(value) != set(expected):
        raise ValueError(f"cell {cell.cell_id} result args fields changed")
    raw_files = value.get("files")
    if (
        not isinstance(raw_files, list)
        or len(raw_files) != len(inputs)
        or any(
            not _relocated_path_matches(actual, expected_path, root)
            for actual, expected_path in zip(raw_files, inputs, strict=True)
        )
    ):
        raise ValueError(f"cell {cell.cell_id} result args changed corpus inputs")
    for name, expected_path in {
        "splits": split_manifest,
        "save": Path(str(expected["save"])),
        "out": result_path,
    }.items():
        if not _relocated_path_matches(value.get(name), expected_path, root):
            raise ValueError(f"cell {cell.cell_id} result args changed {name}")
    path_fields = {"files", "splits", "save", "out"}
    mismatches = sorted(
        name
        for name, expected_value in expected.items()
        if name not in path_fields and not _same_arg_value(value.get(name), expected_value)
    )
    if mismatches:
        raise ValueError(f"cell {cell.cell_id} result args changed: {mismatches}")
    return registered


def _expected_deployment_host_contract(
    inputs: Sequence[Path], *, deploy_view: bool
) -> dict[str, object]:
    if not deploy_view:
        return {"mode": "not_applied", "source_manifests": []}
    sources: list[dict[str, object]] = []
    for corpus in inputs:
        manifest = Path(f"{corpus}.manifest.json")
        document = _read_object(manifest, "generator manifest")
        config = document.get("config")
        corpus_sha256 = grid_runner._sha256(corpus)
        if not isinstance(config, Mapping) or document.get("sha256") != corpus_sha256:
            raise ValueError(f"generator manifest does not match corpus {corpus}")
        defects: dict[str, float] = {}
        for field in ("defect_qubits", "defect_couplers"):
            value = config.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) != 0.0
            ):
                raise ValueError(f"deploy-view selection requires defect-free manifest {manifest}")
            defects[field] = float(value)
        sources.append(
            {
                "file": corpus.name,
                "corpus_sha256": corpus_sha256,
                "manifest_sha256": grid_runner._sha256(manifest),
                **defects,
            }
        )
    return {
        "mode": "pristine_topology_reconstruction",
        "source_manifests": sources,
    }


def _load_validation_records(
    inputs: Sequence[Path],
    split_manifest: Path,
) -> list[dict[str, Any]]:
    """Materialize only records assigned to the frozen validation partition."""

    split_document = _read_object(split_manifest, "split manifest")
    if (
        split_document.get("schema") != "embedbench.split-manifest"
        or split_document.get("schema_version") != 2
    ):
        raise ValueError(
            "training data provenance validation replay requires a schema-v2 split manifest"
        )
    split_tables = split_document.get("splits")
    if not isinstance(split_tables, Mapping) or set(split_tables) != {path.name for path in inputs}:
        raise ValueError("validation replay split tables do not exactly cover corpus inputs")
    records: list[dict[str, Any]] = []
    for path in inputs:
        table = split_tables.get(path.name)
        if not isinstance(table, Mapping):
            raise ValueError(f"validation replay has no split table for {path.name}")
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise ValueError(f"invalid validation replay corpus {path}") from error
        for line_number, line in enumerate(payload.splitlines(), start=1):
            record = strict_json_loads(
                line,
                location=f"validation replay {path}:{line_number}",
            )
            if not isinstance(record, dict):
                raise ValueError(f"validation replay {path}:{line_number} must be an object")
            instance_id = record.get("instance_id")
            if not isinstance(instance_id, str) or not instance_id:
                raise ValueError(
                    f"validation replay {path}:{line_number} has no canonical instance_id"
                )
            partition = table.get(instance_id)
            if partition not in {"train", "val", "test"}:
                raise ValueError(
                    f"validation replay {path}:{line_number} is absent from the split table"
                )
            if partition == "val":
                records.append(record)
    if not records:
        raise ValueError("validation replay requires a non-empty validation partition")
    return records


_VALIDATION_MODEL_FIELDS = frozenset(
    {
        "instance_id",
        "focus",
        "n_vars",
        "window_nodes",
        "window_edges",
        "frozen",
        "all_chains",
        "frozen_adjacency",
        "neighbours",
        "edge_J",
        "neighbour_degree",
        "neighbour_chain_size",
        "neighbour_h",
        "candidates",
        "Q",
        "source",
        "topology",
        "size",
        "difficulty",
        "l_cap",
        "focus_h",
        "problem",
        "defect_qubits",
        "defect_couplers",
    }
)


def _validation_model_record(record: Mapping[str, Any]) -> dict[str, Any]:
    projected = {key: value for key, value in record.items() if key in _VALIDATION_MODEL_FIELDS}
    problem = projected.get("problem")
    if not isinstance(problem, Mapping) or "h" not in problem or "J" not in problem:
        raise ValueError("validation model input requires problem.h and problem.J")
    projected["problem"] = {"h": problem["h"], "J": problem["J"]}
    return projected


def _replay_validation_checkpoint(
    model: Any,
    records: Sequence[Mapping[str, Any]],
    preprocessing: Mapping[str, object],
    *,
    lcb_z: float,
) -> tuple[dict[str, object], dict[str, object]]:
    """Independently reconstruct the complete registered validation sweep."""

    import torch

    runtime_policy = _configure_canonical_replay_runtime()
    from embedbench.models_chain import (
        deployment_view,
        encode_chain,
        quality_candidate_signature,
    )
    from embedbench.models_hetero import encode_hetero_input
    from embedbench.models_quality_v2 import (
        derive_exact_candidate_metrics,
        evaluate_quality_v2_budgets,
        tensors_from_chain,
    )
    from embedbench.structural import host_graph

    floating_parameter_dtypes = {
        parameter.dtype for parameter in model.parameters() if parameter.is_floating_point()
    }
    if floating_parameter_dtypes != {torch.float32}:
        raise ValueError("canonical validation replay requires float32 model parameters")
    model = model.to(torch.device("cpu"))
    model.eval()
    outputs = []
    labels: list[SimpleNamespace] = []
    exact_metrics = []
    evidence_records: list[dict[str, object]] = []
    hosts: dict[tuple[str, int], object] = {}
    with torch.no_grad():
        for record_index, record in enumerate(records):
            projected = _validation_model_record(record)
            if preprocessing.get("deploy_view") is True:
                topology = projected.get("topology")
                size = projected.get("size")
                if (
                    not isinstance(topology, str)
                    or isinstance(size, bool)
                    or not isinstance(size, int)
                ):
                    raise ValueError("validation deploy-view requires topology and integer size")
                host_key = (topology, size)
                if host_key not in hosts:
                    hosts[host_key] = host_graph(*host_key)
                projected = deployment_view(
                    projected,
                    hosts[host_key],
                    max_free=int(preprocessing["deploy_max_free"]),
                )
            exact = derive_exact_candidate_metrics(projected)
            if not bool(exact.feasible.all()):
                raise ValueError(
                    f"validation record {record.get('instance_id')!r} contains an "
                    "exactly infeasible candidate"
                )
            candidate_count = len(exact)
            if model.config.arch == "hetero":
                model_input = encode_hetero_input(
                    projected,
                    require_hamiltonian_context=True,
                )
                output = model(model_input)
            else:
                chain_input = dict(projected)
                chain_input.update(
                    p_solve=[0.0] * candidate_count,
                    stage=[2] * candidate_count,
                    best_index=0,
                    resource_index=0,
                    original_index=-1,
                )
                encoded = encode_chain(
                    chain_input,
                    use_neighbour_feats=bool(preprocessing["neighbour_feats"]),
                    require_hamiltonian_context=True,
                )
                output = model(*tensors_from_chain(encoded, torch.device("cpu")))
            label_row = SimpleNamespace(
                p=record.get("p_solve"),
                stage=record.get("stage"),
                resource_index=record.get("resource_index"),
                original_index=record.get("original_index", -1),
                source=record.get("source"),
            )
            outputs.append(output)
            labels.append(label_row)
            exact_metrics.append(exact)

            feasible = [bool(value) for value in exact.feasible.tolist()]
            total_qubits = [int(value) for value in exact.total_qubits.tolist()]
            original = label_row.original_index
            q_mm = (
                total_qubits[original]
                if label_row.source == "minorminer"
                and isinstance(original, int)
                and not isinstance(original, bool)
                and 0 <= original < candidate_count
                and feasible[original]
                else None
            )
            mean = output.quality_mean.detach().cpu().numpy()
            budgets: dict[str, object] = {}
            for ratio in REGISTERED_BUDGET_RATIOS:
                key = "uncapped" if ratio is None else f"{ratio:.2f}x"
                budget = (
                    max(total_qubits)
                    if ratio is None
                    else None
                    if q_mm is None
                    else int(math.floor(ratio * q_mm + 1e-9))
                )
                eligible = (
                    []
                    if budget is None
                    else [
                        index
                        for index in range(candidate_count)
                        if feasible[index] and total_qubits[index] <= budget
                    ]
                )
                selected = (
                    max(eligible, key=lambda index: (float(mean[index]), -index))
                    if eligible
                    else None
                )
                budgets[key] = {
                    "budget": None if ratio is None else budget,
                    "eligible_indices": eligible,
                    "mean_selection_index": selected,
                }
            evidence_records.append(
                {
                    "record_index": record_index,
                    "instance_id": record.get("instance_id"),
                    "focus": record.get("focus"),
                    "problem_digest": quality_problem_digest(record),
                    "candidate_signature": quality_candidate_signature(record),
                    "exact_feasible": feasible,
                    "total_qubits": total_qubits,
                    "resource_index": label_row.resource_index,
                    "original_index": label_row.original_index,
                    "q_mm": q_mm,
                    "budgets": budgets,
                }
            )

    sweep = evaluate_quality_v2_budgets(
        outputs,
        labels,
        exact_metrics,
        budget_ratios=REGISTERED_BUDGET_RATIOS,
        lcb_z=lcb_z,
        random_seed=0,
    )
    evidence = {
        "schema": "embedbench.quality-v2-validation-replay",
        "schema_version": 1,
        "runtime_policy": runtime_policy,
        "record_count": len(evidence_records),
        "records": evidence_records,
        "budget_sweep": sweep,
    }
    evidence_payload = json.dumps(
        evidence,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sweep, {
        "schema": evidence["schema"],
        "schema_version": evidence["schema_version"],
        "runtime_policy": runtime_policy,
        "record_count": len(evidence_records),
        "sha256": _payload_sha256(evidence_payload),
        "primary_metric": sweep["primary_metric"],
    }


def _validated_budget_sweep(
    sweep: Mapping[str, object], *, cell_id: str, validation_records: int
) -> tuple[float, dict[str, object]]:
    expected_fields = {
        "candidate_support": "full",
        "support_uses_labels": False,
        "evaluation_mode": "development",
        "provisional": True,
        "exact_mask": "minor_embedding_feasible_and_full_embedding_total_qubits",
        "release_Q_semantics": "release_focus_chain_Q",
        "budget_contract": "B/Q_MM",
        "budget_reference": "stock_minorminer_original_total_qubits",
        "primary_record_policy": "valid_stock_minorminer_original_reference_only",
        "budget_ratios": list(REGISTERED_BUDGET_RATIOS),
        "primary_selector": "mean",
        "primary_metric_name": PRIMARY_METRIC,
    }
    failed = sorted(
        field for field, expected in expected_fields.items() if sweep.get(field) != expected
    )
    if failed:
        raise ValueError(f"cell {cell_id} has invalid budget-sweep fields: {failed}")

    coverage = sweep.get("primary_record_coverage")
    if not isinstance(coverage, Mapping):
        raise ValueError(f"cell {cell_id} has no B/Q_MM coverage record")
    integer_fields = ("input_records", "included_records", "excluded_records")
    if any(
        isinstance(coverage.get(field), bool)
        or not isinstance(coverage.get(field), int)
        or int(coverage[field]) < 0
        for field in integer_fields
    ):
        raise ValueError(f"cell {cell_id} has invalid B/Q_MM coverage counts")
    input_records = int(coverage["input_records"])
    included = int(coverage["included_records"])
    excluded = int(coverage["excluded_records"])
    if input_records != validation_records or included <= 0 or included + excluded != input_records:
        raise ValueError(f"cell {cell_id} has invalid B/Q_MM validation coverage")
    rate = coverage.get("coverage_rate")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isclose(float(rate), included / input_records, abs_tol=1e-12)
    ):
        raise ValueError(f"cell {cell_id} has inconsistent B/Q_MM coverage rate")
    included_indices = coverage.get("included_input_indices")
    excluded_indices = coverage.get("excluded_input_indices")
    if (
        not isinstance(included_indices, list)
        or not isinstance(excluded_indices, list)
        or len(included_indices) != included
        or len(excluded_indices) != excluded
        or any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in included_indices + excluded_indices
        )
        or sorted(included_indices + excluded_indices) != list(range(input_records))
    ):
        raise ValueError(f"cell {cell_id} has inconsistent B/Q_MM record indices")
    reasons = coverage.get("exclusion_reasons")
    expected_reason_keys = {
        "non_minorminer_source",
        "missing_original_reference",
        "infeasible_original_reference",
    }
    if (
        not isinstance(reasons, Mapping)
        or set(reasons) != expected_reason_keys
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in reasons.values()
        )
        or sum(int(value) for value in reasons.values()) != excluded
    ):
        raise ValueError(f"cell {cell_id} has inconsistent B/Q_MM exclusion reasons")

    budgets = sweep.get("budgets")
    if not isinstance(budgets, Mapping):
        raise ValueError(f"cell {cell_id} has no registered per-budget results")
    finite_regrets: list[float] = []
    for ratio in REGISTERED_BUDGET_RATIOS:
        key = "uncapped" if ratio is None else f"{ratio:.2f}x"
        budget = budgets.get(key)
        if not isinstance(budget, Mapping):
            raise ValueError(f"cell {cell_id} is missing registered budget {key}")
        if (
            budget.get("ratio") != ratio
            or budget.get("reference_policy") != "stock_minorminer_original_total_qubits"
            or budget.get("n_records") != included
            or budget.get("reference_sources") != {"stock_minorminer_original": included}
        ):
            raise ValueError(f"cell {cell_id} has invalid registered budget {key}")
        if budget.get("input_record_indices") != coverage["included_input_indices"]:
            raise ValueError(f"cell {cell_id} budget {key} changes the covered cohort")
        aligned_fields = ("reference_indices", "reference_total_qubits", "budgets")
        if any(
            not isinstance(budget.get(field), list) or len(budget[field]) != included
            for field in aligned_fields
        ):
            raise ValueError(f"cell {cell_id} budget {key} has incomplete reference arrays")
        if budget.get("no_survivor") != 0 or budget.get("no_survivor_rate") != 0.0:
            raise ValueError(f"cell {cell_id} budget {key} has an incomplete survivor cohort")
        selectors = budget.get("selectors")
        mean = selectors.get("mean") if isinstance(selectors, Mapping) else None
        regret = mean.get("regret") if isinstance(mean, Mapping) else None
        if (
            isinstance(regret, bool)
            or not isinstance(regret, (int, float))
            or not math.isfinite(float(regret))
            or float(regret) < 0.0
        ):
            raise ValueError(f"cell {cell_id} has invalid mean regret at budget {key}")
        selection_indices = mean.get("selection_indices")
        if (
            mean.get("n_selected") != included
            or not isinstance(selection_indices, list)
            or len(selection_indices) != included
            or any(
                isinstance(index, bool) or not isinstance(index, int) for index in selection_indices
            )
        ):
            raise ValueError(f"cell {cell_id} budget {key} hides a selector subcohort")
        if ratio is not None:
            finite_regrets.append(float(regret))

    stored_metric = sweep.get("primary_metric")
    recomputed_metric = statistics.fmean(finite_regrets)
    if (
        isinstance(stored_metric, bool)
        or not isinstance(stored_metric, (int, float))
        or not math.isfinite(float(stored_metric))
        or float(stored_metric) < 0.0
        or not math.isclose(float(stored_metric), recomputed_metric, rel_tol=1e-12, abs_tol=1e-12)
    ):
        raise ValueError(f"cell {cell_id} primary metric disagrees with recomputed regrets")
    return recomputed_metric, dict(coverage)


def _model_independent_validation_evidence(sweep: Mapping[str, object]) -> dict[str, object]:
    """Project the stored/replayed fields that cannot depend on model predictions."""

    raw_budgets = sweep.get("budgets")
    if not isinstance(raw_budgets, Mapping):
        raise ValueError("validated budget sweep lost its per-budget evidence")
    budget_fields = (
        "ratio",
        "reference_policy",
        "reference_sources",
        "input_record_indices",
        "reference_indices",
        "reference_total_qubits",
        "budgets",
        "n_records",
        "mean_eligible_candidates",
        "minimum_eligible_candidates",
        "maximum_eligible_candidates",
        "no_survivor",
        "no_survivor_rate",
    )
    projected_budgets: dict[str, object] = {}
    for ratio in REGISTERED_BUDGET_RATIOS:
        key = "uncapped" if ratio is None else f"{ratio:.2f}x"
        budget = raw_budgets.get(key)
        if not isinstance(budget, Mapping):
            raise ValueError(f"validated budget sweep lost registered budget {key}")
        projected_budgets[key] = {field: budget.get(field) for field in budget_fields}
    return {
        "exact_mask": sweep.get("exact_mask"),
        "release_Q_semantics": sweep.get("release_Q_semantics"),
        "budget_contract": sweep.get("budget_contract"),
        "budget_reference": sweep.get("budget_reference"),
        "primary_record_policy": sweep.get("primary_record_policy"),
        "primary_record_coverage": sweep.get("primary_record_coverage"),
        "budget_ratios": sweep.get("budget_ratios"),
        "budgets": projected_budgets,
    }


def _cross_device_validation_diagnostic(
    training_sweep: Mapping[str, object],
    replay_sweep: Mapping[str, object],
    *,
    training_metric: float,
    replay_metric: float,
) -> dict[str, object]:
    """Record model-dependent numerical drift without using it as a gate or tie-break."""

    training_budgets = training_sweep.get("budgets")
    replay_budgets = replay_sweep.get("budgets")
    if not isinstance(training_budgets, Mapping) or not isinstance(replay_budgets, Mapping):
        raise ValueError("validated sweeps lost their cross-device diagnostic fields")
    diagnostics: dict[str, object] = {}
    total_disagreements = 0
    for ratio in REGISTERED_BUDGET_RATIOS:
        key = "uncapped" if ratio is None else f"{ratio:.2f}x"
        training_budget = training_budgets.get(key)
        replay_budget = replay_budgets.get(key)
        if not isinstance(training_budget, Mapping) or not isinstance(replay_budget, Mapping):
            raise ValueError(f"validated sweeps lost cross-device budget {key}")
        training_selectors = training_budget.get("selectors")
        replay_selectors = replay_budget.get("selectors")
        training_mean = (
            training_selectors.get("mean") if isinstance(training_selectors, Mapping) else None
        )
        replay_mean = (
            replay_selectors.get("mean") if isinstance(replay_selectors, Mapping) else None
        )
        if not isinstance(training_mean, Mapping) or not isinstance(replay_mean, Mapping):
            raise ValueError(f"validated sweeps lost mean-selector evidence at budget {key}")
        training_indices = training_mean.get("selection_indices")
        replay_indices = replay_mean.get("selection_indices")
        input_indices = training_budget.get("input_record_indices")
        if (
            not isinstance(training_indices, list)
            or not isinstance(replay_indices, list)
            or not isinstance(input_indices, list)
            or len(training_indices) != len(replay_indices)
            or len(training_indices) != len(input_indices)
        ):
            raise ValueError(f"validated sweeps have misaligned selections at budget {key}")
        disagreements = [
            {
                "input_record_index": input_index,
                "training_device_selection_index": training_index,
                "canonical_cpu_selection_index": replay_index,
            }
            for input_index, training_index, replay_index in zip(
                input_indices,
                training_indices,
                replay_indices,
                strict=True,
            )
            if training_index != replay_index
        ]
        total_disagreements += len(disagreements)
        diagnostics[key] = {
            "selection_index_disagreement_count": len(disagreements),
            "selection_index_disagreements": disagreements,
        }
    return {
        "role": "record_only_never_reject_rank_exclude_or_break_ties",
        "training_device_primary_metric": training_metric,
        "canonical_cpu_primary_metric": replay_metric,
        "canonical_cpu_minus_training_device_metric": replay_metric - training_metric,
        "selection_index_disagreement_count": total_disagreements,
        "budgets": diagnostics,
        "used_for_acceptance": False,
        "used_for_ranking": False,
        "used_for_exclusion": False,
        "used_for_tie_breaking": False,
    }


def _validated_cell(
    cell: grid_runner.GridCell,
    *,
    document: Mapping[str, object],
    stage: Mapping[str, object],
    root: Path,
    grid_id: object,
    grid_sha256: str,
    source_sha256: str,
    data_provenance: Mapping[str, object],
    expected_deployment_host_contract: Mapping[str, object],
    validation_records: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    result_path, receipt_path, checkpoint_path = _expected_paths(stage, cell, root)
    relative_paths = {
        path: _relative(path, root) for path in (result_path, receipt_path, checkpoint_path)
    }
    missing = [
        relative_paths[path]
        for path in (result_path, receipt_path, checkpoint_path)
        if not path.is_file()
    ]
    if missing:
        raise ValueError(f"incomplete training grid cell {cell.cell_id}: missing {missing}")

    try:
        result_payload = result_path.read_bytes()
        receipt_payload = receipt_path.read_bytes()
        checkpoint_payload = checkpoint_path.read_bytes()
    except OSError as error:
        raise ValueError(f"cell {cell.cell_id} artifacts changed during capture") from error
    result_sha256 = _payload_sha256(result_payload)
    receipt_sha256 = _payload_sha256(receipt_payload)
    checkpoint_sha256 = _payload_sha256(checkpoint_payload)
    receipt = _object_from_payload(
        receipt_payload,
        path=receipt_path,
        kind="training receipt",
    )
    expected_cell = json.loads(json.dumps(cell._asdict()))
    require_exact_json(
        receipt.get("data_provenance"),
        data_provenance,
        location=f"cell {cell.cell_id} receipt data provenance",
    )
    require_exact_json(
        receipt.get("cell"),
        expected_cell,
        location=f"cell {cell.cell_id} receipt cell",
    )
    receipt_checks = {
        "schema": receipt.get("schema") == "embedbench.training-receipt",
        "schema_version": type(receipt.get("schema_version")) is int
        and receipt.get("schema_version") == 1,
        "grid_id": receipt.get("grid_id") == grid_id,
        "grid_sha256": receipt.get("grid_sha256") == grid_sha256,
        "source_sha256": receipt.get("source_sha256") == source_sha256,
        "result_sha256": receipt.get("result_sha256") == result_sha256,
        "checkpoint_sha256": receipt.get("checkpoint_sha256") == checkpoint_sha256,
        "test_locked": receipt.get("test_locked") is True,
    }
    expected_site = grid_runner.expected_execution_site(stage, cell)
    if expected_site is not None:
        receipt_checks["execution_site"] = receipt.get("execution_site") == expected_site
    failed = sorted(name for name, passed in receipt_checks.items() if not passed)
    if failed:
        raise ValueError(f"cell {cell.cell_id} has invalid receipt fields: {failed}")
    if Path(str(receipt.get("result", ""))).name != result_path.name:
        raise ValueError(f"cell {cell.cell_id} receipt names a different result artifact")
    if Path(str(receipt.get("checkpoint", ""))).name != checkpoint_path.name:
        raise ValueError(f"cell {cell.cell_id} receipt names a different checkpoint")
    _validate_receipt_command(
        receipt.get("command"),
        document=document,
        stage=stage,
        cell=cell,
        root=root,
    )
    execution = stage.get("execution")
    slurm = receipt.get("slurm")
    if expected_site is not None:
        if not isinstance(execution, Mapping) or not isinstance(slurm, Mapping):
            raise ValueError(f"cell {cell.cell_id} has incomplete execution provenance")
        required_slurm = execution.get("slurm_required_sites", [])
        forbidden_slurm = execution.get("slurm_forbidden_sites", [])
        if expected_site in required_slurm and any(
            not isinstance(slurm.get(field), str) or not slurm[field]
            for field in ("array_job_id", "array_task_id", "job_id")
        ):
            raise ValueError(f"cell {cell.cell_id} is missing required Slurm provenance")
        if expected_site in forbidden_slurm and any(
            slurm.get(field) is not None for field in ("array_job_id", "array_task_id", "job_id")
        ):
            raise ValueError(f"cell {cell.cell_id} unexpectedly ran under Slurm")

    result = _object_from_payload(result_payload, path=result_path, kind="training result")
    if (
        result.get("artifact_schema") != "embedbench.quality-v2-training-results"
        or type(result.get("artifact_schema_version")) is not int
        or result.get("artifact_schema_version") != 1
    ):
        raise ValueError(f"cell {cell.cell_id} is not a Quality V2 training result")
    require_exact_json(
        result.get("data_provenance"),
        data_provenance,
        location=f"cell {cell.cell_id} result data provenance",
    )
    registered_trainer_args = _validate_result_args(
        result.get("args"),
        stage=stage,
        cell=cell,
        root=root,
        result_path=result_path,
        checkpoint_path=checkpoint_path,
    )
    expected_device_type = (
        execution.get("required_device_type") if isinstance(execution, Mapping) else None
    )
    runtime_signature = _runtime_compatibility_signature(
        result.get("runtime_provenance"),
        expected_device_type=expected_device_type,
        cell_id=cell.cell_id,
    )
    if not _is_full_training_scope(result.get("training_scope")):
        raise ValueError(f"cell {cell.cell_id} was not trained on full fixed train/validation")
    expected_preprocessing = _expected_preprocessing(stage, architecture=cell.params.get("arch"))
    require_exact_json(
        result.get("preprocessing"),
        expected_preprocessing,
        location=f"cell {cell.cell_id} result preprocessing",
    )
    require_exact_json(
        result.get("deployment_host_contract"),
        expected_deployment_host_contract,
        location=f"cell {cell.cell_id} result deployment host contract",
    )
    if result.get("test_locked") is not True or result.get("test_partition_encoded") is not False:
        raise ValueError(f"cell {cell.cell_id} does not prove that test stayed locked")

    rows = result.get("results")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ValueError(f"cell {cell.cell_id} must contain exactly one seed result")
    row = rows[0]
    for name, value in cell.params.items():
        require_exact_json(
            row.get(name),
            value,
            location=f"cell {cell.cell_id} result axis {name!r}",
        )
    for name in ("hidden", "layers", "heads"):
        require_exact_json(
            row.get(name),
            registered_trainer_args[name],
            location=f"cell {cell.cell_id} registered result argument {name}",
        )
    require_exact_json(
        row.get("preprocessing"),
        expected_preprocessing,
        location=f"cell {cell.cell_id} seed preprocessing",
    )
    require_exact_json(
        row.get("deployment_host_contract"),
        expected_deployment_host_contract,
        location=f"cell {cell.cell_id} seed deployment host contract",
    )
    if not _is_full_training_scope(row.get("training_scope")):
        raise ValueError(f"cell {cell.cell_id} seed was not trained on full fixed train/validation")
    require_exact_json(
        row.get("registered_budget_ratios"),
        list(REGISTERED_BUDGET_RATIOS),
        location=f"cell {cell.cell_id} registered budget ratios",
    )
    if (
        row.get("primary_validation_metric") != PRIMARY_METRIC
        or row.get("evaluation_support") != "full"
        or row.get("selection_contract") != SELECTION_CONTRACT
        or row.get("auxiliary_heads_used_for_selection") is not False
        or row.get("test_evaluated") is not False
        or row.get("test_partition_encoded") is not False
        or row.get("test") is not None
    ):
        raise ValueError(f"cell {cell.cell_id} violates the validation-only protocol")
    validation = row.get("validation")
    sweep = validation.get("budget_sweep") if isinstance(validation, Mapping) else None
    if not isinstance(sweep, Mapping):
        raise ValueError(f"cell {cell.cell_id} has no registered budget sweep")
    validation_record_count = row.get("n_val")
    if isinstance(validation_record_count, bool) or not isinstance(validation_record_count, int):
        raise ValueError(f"cell {cell.cell_id} has invalid validation record count")
    training_device_metric, training_device_coverage = _validated_budget_sweep(
        sweep,
        cell_id=cell.cell_id,
        validation_records=validation_record_count,
    )

    run_id = row.get("training_run_id")
    if (
        not isinstance(run_id, str)
        or len(run_id) != 32
        or any(character not in "0123456789abcdef" for character in run_id)
    ):
        raise ValueError(f"cell {cell.cell_id} has an invalid training run ID")
    try:
        from embedbench.models_quality_v2 import load_quality_v2_model

        with tempfile.TemporaryDirectory(prefix="embedbench-selection-checkpoint-") as directory:
            captured_checkpoint = Path(directory) / f"{checkpoint_sha256}{checkpoint_path.suffix}"
            captured_checkpoint.write_bytes(checkpoint_payload)
            checkpoint_model, checkpoint_metadata = load_quality_v2_model(captured_checkpoint)
    except Exception as error:
        raise ValueError(f"cell {cell.cell_id} has an invalid Quality V2 checkpoint") from error
    if checkpoint_model.config.arch != cell.params.get("arch"):
        raise ValueError(f"cell {cell.cell_id} checkpoint architecture changed")
    checkpoint_expectations = {
        "objective_variant": cell.params.get("objective_variant"),
        "seed": cell.params.get("seed"),
        "training_run_id": run_id,
        "best_epoch": row.get("best_epoch"),
        "split_sha256": data_provenance["split_manifest"]["sha256"],
        "data_provenance": data_provenance,
        "runtime_provenance": result.get("runtime_provenance"),
        "evaluation_support": "full",
        "registered_budget_ratios": list(REGISTERED_BUDGET_RATIOS),
        "primary_validation_metric": PRIMARY_METRIC,
        "lcb_z": registered_trainer_args["lcb_z"],
        "selection_contract": SELECTION_CONTRACT,
        "auxiliary_heads_used_for_selection": False,
        "test_evaluated": False,
        "test_partition_encoded": False,
        "training_scope": row.get("training_scope"),
        "preprocessing": row.get("preprocessing"),
        "deployment_host_contract": row.get("deployment_host_contract"),
        "loss_weights": row.get("loss_weights"),
    }
    for field, expected in checkpoint_expectations.items():
        label = "training run" if field == "training_run_id" else "metadata"
        require_exact_json(
            checkpoint_metadata.get(field),
            expected,
            location=f"cell {cell.cell_id} checkpoint {label} mismatch for {field}",
        )
    for name in ("hidden", "layers", "heads", "neighbour_feats"):
        require_exact_json(
            getattr(checkpoint_model.config, name),
            registered_trainer_args[name],
            location=f"cell {cell.cell_id} checkpoint model config {name}",
        )
    checkpoint_metric = checkpoint_metadata.get("selected_validation_metric")
    if (
        isinstance(checkpoint_metric, bool)
        or not isinstance(checkpoint_metric, (int, float))
        or not math.isclose(
            float(checkpoint_metric),
            training_device_metric,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"cell {cell.cell_id} checkpoint selected validation metric mismatch")

    replay_sweep, replay_evidence = _replay_validation_checkpoint(
        checkpoint_model,
        validation_records,
        expected_preprocessing,
        lcb_z=float(registered_trainer_args["lcb_z"]),
    )
    replay_metric, replay_coverage = _validated_budget_sweep(
        replay_sweep,
        cell_id=f"{cell.cell_id} canonical CPU replay",
        validation_records=validation_record_count,
    )
    require_exact_json(
        replay_coverage,
        training_device_coverage,
        location=f"cell {cell.cell_id} canonical replay cohort",
    )
    require_exact_json(
        _model_independent_validation_evidence(replay_sweep),
        _model_independent_validation_evidence(sweep),
        location=f"cell {cell.cell_id} model-independent validation evidence",
    )
    if not math.isfinite(float(replay_metric)):
        raise ValueError(f"cell {cell.cell_id} validation metric fails independent replay")

    return {
        "cell_id": cell.cell_id,
        "index": cell.index,
        "seed": cell.params["seed"],
        "validation_metric": float(replay_metric),
        "validation_metric_source": "canonical_cpu_checkpoint_replay",
        "training_device_validation_metric": float(training_device_metric),
        "training_device_validation_sweep_sha256": _payload_sha256(
            json.dumps(
                sweep,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ),
        "cross_device_validation_diagnostic": _cross_device_validation_diagnostic(
            sweep,
            replay_sweep,
            training_metric=float(training_device_metric),
            replay_metric=float(replay_metric),
        ),
        "result": relative_paths[result_path],
        "result_sha256": result_sha256,
        "checkpoint": relative_paths[checkpoint_path],
        "checkpoint_sha256": checkpoint_sha256,
        "coverage": dict(replay_coverage),
        "execution_site": expected_site,
        "hostname": receipt.get("hostname"),
        "slurm": dict(slurm) if isinstance(slurm, Mapping) else None,
        "receipt": relative_paths[receipt_path],
        "receipt_sha256": receipt_sha256,
        "runtime_provenance": result.get("runtime_provenance"),
        "runtime_compatibility_signature": runtime_signature,
        "validation_replay": replay_evidence,
    }


def _preflight_training_registration(
    document: Mapping[str, object],
    *,
    grid_path: Path,
    stage_name: str,
    root: Path,
    audit_contract: PaperAuditContract,
    grid_payload: bytes | None = None,
) -> _TrainingPreflight:
    """Verify the immutable grid and training-source digest before artifact access."""

    root = root.resolve()
    grid_path = grid_path.resolve()
    if not grid_path.is_relative_to(root):
        raise ValueError("training grid path is outside the staged root")
    if grid_payload is None:
        try:
            grid_payload = grid_path.read_bytes()
        except OSError as error:
            raise ValueError(f"invalid training grid {grid_path}") from error
    on_disk_document = _object_from_payload(
        grid_payload,
        path=grid_path,
        kind="training grid",
    )
    require_exact_json(
        document,
        on_disk_document,
        location="selection grid document",
    )
    registration = audit_contract.training_registration
    if not isinstance(registration, Mapping) or set(registration) != {
        "grid_sha256",
        "source_sha256",
        "selection",
        "stage",
        "data_provenance",
        "validation_replay",
    }:
        raise ValueError("paper audit contract has an incomplete training registration")
    grid_sha256 = _payload_sha256(grid_payload)
    if grid_sha256 != registration["grid_sha256"]:
        raise ValueError("training grid SHA-256 differs from its frozen registration")
    stage = grid_runner._stage(on_disk_document, stage_name)
    require_exact_json(
        on_disk_document.get("selection"),
        registration["selection"],
        location="training grid selection policy",
    )
    require_exact_json(
        stage,
        registration["stage"],
        location="training grid stage",
    )
    source_sha256 = grid_runner._source_sha256(root)
    if source_sha256 != registration["source_sha256"]:
        raise ValueError(
            "training source SHA-256 differs from its frozen registration: "
            f"expected {registration['source_sha256']}, observed {source_sha256}"
        )
    return _TrainingPreflight(
        document=on_disk_document,
        stage=stage,
        registration=registration,
        grid_sha256=grid_sha256,
        source_sha256=source_sha256,
    )


def select_grid(
    document: Mapping[str, object],
    *,
    grid_path: Path,
    stage_name: str,
    root: Path,
    audit_contract: PaperAuditContract,
    grid_payload: bytes | None = None,
) -> dict[str, object]:
    """Validate every cell without encoding, evaluating, or reading labels from test rows."""

    runtime_policy = _configure_canonical_replay_runtime()
    root = root.resolve()
    grid_path = grid_path.resolve()
    preflight = _preflight_training_registration(
        document,
        grid_path=grid_path,
        stage_name=stage_name,
        root=root,
        audit_contract=audit_contract,
        grid_payload=grid_payload,
    )
    document = preflight.document
    stage = preflight.stage
    registration = preflight.registration
    grid_sha256 = preflight.grid_sha256
    source_sha256 = preflight.source_sha256
    cells = grid_runner.expand_stage(document, stage_name)
    order = stage.get("axis_order")
    axes = stage.get("axes")
    if not isinstance(order, list) or not isinstance(axes, Mapping) or "seed" not in order:
        raise ValueError("selection requires one registered seed axis")
    selection = document.get("selection")
    if not isinstance(selection, Mapping):
        raise ValueError("training grid requires a selection contract")
    if selection.get("primary_metric") != PRIMARY_METRIC:
        raise ValueError(f"selection.primary_metric must be {PRIMARY_METRIC!r}")
    registered_seeds = selection.get("seeds")
    if not isinstance(registered_seeds, list) or registered_seeds != axes.get("seed"):
        raise ValueError("selection seeds must exactly match the ordered seed axis")
    if len(registered_seeds) < 2 or len(registered_seeds) != len(set(registered_seeds)):
        raise ValueError("selection requires at least two distinct registered seeds")
    registered_trainer_args = _registered_trainer_args(stage)
    if registered_trainer_args["lcb_z"] != audit_contract.evaluation["lcb_z"]:
        raise ValueError("training grid lcb_z disagrees with the frozen paper-audit contract")
    validation_replay_registration = registration.get("validation_replay")
    if not isinstance(validation_replay_registration, Mapping):
        raise ValueError("paper audit contract has no canonical validation replay")
    require_exact_json(
        CANONICAL_VALIDATION_REPLAY_POLICY,
        validation_replay_registration.get("canonical_runtime"),
        location="selector canonical validation replay policy",
    )

    expected_selection = audit_contract.selection
    actual_selection = {
        "artifact_schema": SELECTION_SCHEMA,
        "artifact_schema_version": SELECTION_SCHEMA_VERSION,
        "grid_id": document.get("grid_id"),
        "grid_path": _relative(grid_path, root),
        "stage": stage_name,
        "registered_cell_count": len(cells),
        "registered_seeds": registered_seeds,
        "validation_only": True,
    }
    mismatches = [
        field
        for field, expected in expected_selection.items()
        if actual_selection.get(field) != expected
    ]
    if mismatches:
        raise ValueError(
            f"training grid disagrees with the frozen paper-audit selection contract: {mismatches}"
        )

    inputs, split_manifest = grid_runner._stage_inputs(stage, root)
    # Parse the split and every routed row with the paper pipeline's strict decoder before
    # calling helpers from the frozen training runner.  The preregistered runner predates
    # duplicate-key and non-finite rejection and must remain byte-identical to its receipts.
    validation_records = _load_validation_records(inputs, split_manifest)
    data_provenance = grid_runner._data_provenance(inputs, split_manifest)
    require_exact_json(
        data_provenance,
        registration["data_provenance"],
        location="training data provenance",
    )
    deploy_view = bool(
        _expected_preprocessing(stage, architecture=cells[0].params.get("arch"))["deploy_view"]
    )
    expected_host_contract = _expected_deployment_host_contract(inputs, deploy_view=deploy_view)
    validated = [
        _validated_cell(
            cell,
            document=document,
            stage=stage,
            root=root,
            grid_id=document.get("grid_id"),
            grid_sha256=grid_sha256,
            source_sha256=source_sha256,
            data_provenance=data_provenance,
            expected_deployment_host_contract=expected_host_contract,
            validation_records=validation_records,
        )
        for cell in cells
    ]
    coverage_documents = {
        json.dumps(row["coverage"], sort_keys=True, separators=(",", ":")) for row in validated
    }
    if len(coverage_documents) != 1:
        raise ValueError("grid cells disagree on the B/Q_MM validation population")
    runtime_signatures = {
        json.dumps(
            row["runtime_compatibility_signature"],
            sort_keys=True,
            separators=(",", ":"),
        )
        for row in validated
    }
    if len(runtime_signatures) != 1:
        raise ValueError("grid cells used incompatible core runtime environments")

    config_axes = [axis for axis in order if axis != "seed"]
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    params_by_key: dict[tuple[object, ...], dict[str, object]] = {}
    order_by_key: dict[tuple[object, ...], int] = {}
    for cell, row in zip(cells, validated, strict=True):
        key = tuple(cell.params[axis] for axis in config_axes)
        groups.setdefault(key, []).append(row)
        params_by_key[key] = {axis: cell.params[axis] for axis in config_axes}
        order_by_key.setdefault(key, cell.index)

    execution = stage.get("execution")
    if isinstance(execution, Mapping):
        site_mapping = execution.get("remainder_to_site")
        if not isinstance(site_mapping, Mapping):
            raise ValueError("execution contract has no site mapping")
        registered_sites = set(site_mapping.values())
        if execution.get("require_each_configuration_all_sites") is True:
            for key, seed_rows in groups.items():
                if {row["execution_site"] for row in seed_rows} != registered_sites:
                    raise ValueError(
                        f"configuration {params_by_key[key]} is not crossed over all sites"
                    )
        balance_axes = execution.get("exact_balance_axes", [])
        if not isinstance(balance_axes, list) or any(axis not in axes for axis in balance_axes):
            raise ValueError("execution exact_balance_axes are invalid")
        for axis in balance_axes:
            for value in axes[axis]:
                counts = {
                    site: sum(
                        cell.params[axis] == value
                        and grid_runner.expected_execution_site(stage, cell) == site
                        for cell in cells
                    )
                    for site in registered_sites
                }
                if len(set(counts.values())) != 1:
                    raise ValueError(
                        f"execution assignment does not balance {axis}={value!r}: {counts}"
                    )

    summaries: list[dict[str, object]] = []
    for key, seed_rows in groups.items():
        seed_rows.sort(key=lambda row: registered_seeds.index(row["seed"]))
        if [row["seed"] for row in seed_rows] != registered_seeds:
            raise ValueError(f"configuration {params_by_key[key]} has incomplete seeds")
        values = [float(row["validation_metric"]) for row in seed_rows]
        summaries.append(
            {
                "params": params_by_key[key],
                "mean_validation_metric": statistics.fmean(values),
                "sample_std_validation_metric": statistics.stdev(values),
                "minimum_validation_metric": min(values),
                "maximum_validation_metric": max(values),
                "seeds": seed_rows,
                "_registered_order": order_by_key[key],
            }
        )
    summaries.sort(key=lambda row: int(row["_registered_order"]))
    winner = min(
        summaries,
        key=lambda row: (
            float(row["mean_validation_metric"]),
            int(row["_registered_order"]),
        ),
    )
    selected_seed_row = min(
        winner["seeds"],
        key=lambda row: (
            float(row["validation_metric"]),
            registered_seeds.index(row["seed"]),
        ),
    )
    public_summaries = [
        {key: value for key, value in summary.items() if key != "_registered_order"}
        for summary in summaries
    ]
    return {
        "schema": SELECTION_SCHEMA,
        "schema_version": SELECTION_SCHEMA_VERSION,
        "paper_audit_contract": audit_contract.public_binding(),
        "audit_source_hash_algorithm": AUDIT_SOURCE_HASH_ALGORITHM,
        "audit_source_sha256": audit_source_sha256(root),
        "grid_id": document.get("grid_id"),
        "grid_file": grid_path.name,
        "grid_path": _relative(grid_path, root),
        "grid_sha256": grid_sha256,
        "source_sha256": source_sha256,
        "selector_sha256": grid_runner._sha256(Path(__file__)),
        "data_provenance": data_provenance,
        "stage": stage_name,
        "registered_cell_count": len(cells),
        "execution_contract": stage.get("execution"),
        "primary_metric": PRIMARY_METRIC,
        "validation_metric_source": "canonical_cpu_checkpoint_replay",
        "validation_replay_policy": runtime_policy,
        "direction": "minimize",
        "registered_seeds": registered_seeds,
        "configuration_rule": "lowest_canonical_cpu_replay_mean_over_all_registered_seeds",
        "configuration_tie_break": "first_registered_grid_order",
        "checkpoint_rule": ("lowest_canonical_cpu_replay_metric_within_winning_configuration"),
        "checkpoint_tie_break": "first_registered_seed_order",
        "paper_evaluation_rule": (
            "evaluate_every_registered_seed_checkpoint_of_winning_configuration"
        ),
        "test_records_parsed": True,
        "test_records_parsed_for_partition_routing_only": True,
        "test_records_encoded": False,
        "test_labels_used_for_selection": False,
        "test_evaluated": False,
        "configurations": public_summaries,
        "winner": {
            "params": winner["params"],
            "mean_validation_metric": winner["mean_validation_metric"],
            "sample_std_validation_metric": winner["sample_std_validation_metric"],
            "selected_seed": selected_seed_row["seed"],
            "selected_cell_id": selected_seed_row["cell_id"],
            "selected_validation_metric": selected_seed_row["validation_metric"],
            "selected_validation_metric_source": "canonical_cpu_checkpoint_replay",
            "result": selected_seed_row["result"],
            "result_sha256": selected_seed_row["result_sha256"],
            "checkpoint": selected_seed_row["checkpoint"],
            "checkpoint_sha256": selected_seed_row["checkpoint_sha256"],
            "paper_checkpoints": [
                {
                    "seed": row["seed"],
                    "cell_id": row["cell_id"],
                    "checkpoint": row["checkpoint"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                }
                for row in winner["seeds"]
            ],
        },
    }


def revalidate_selection_artifact(
    path: str | Path,
    expected_sha256: str,
    *,
    root: str | Path,
    audit_contract: PaperAuditContract,
    selection_payload: bytes | None = None,
) -> dict[str, object]:
    """Rebuild a selection from its grid and every receipt before test access."""

    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ValueError("selection SHA-256 must be lowercase hexadecimal")
    source = Path(path)
    if selection_payload is None:
        try:
            selection_payload = source.read_bytes()
        except OSError as error:
            raise ValueError(f"invalid selection artifact {source}") from error
    if _payload_sha256(selection_payload) != expected_sha256:
        raise ValueError("selection SHA-256 does not match frozen bytes")
    selected = strict_json_loads(
        selection_payload,
        location=f"selection artifact {source}",
    )
    if not isinstance(selected, dict):
        raise ValueError("selection artifact must be a JSON object")
    if (
        selected.get("schema") != SELECTION_SCHEMA
        or selected.get("schema_version") != SELECTION_SCHEMA_VERSION
    ):
        raise ValueError("locked test requires training-grid selection schema version 3")
    require_exact_json(
        selected.get("paper_audit_contract"),
        audit_contract.public_binding(),
        location="selection paper-audit contract binding",
    )

    staged_root = Path(root).resolve()
    raw_grid_path = selected.get("grid_path")
    if not isinstance(raw_grid_path, str) or not raw_grid_path:
        raise ValueError("selection artifact has no canonical grid_path")
    relative_grid = Path(raw_grid_path)
    if (
        relative_grid.is_absolute()
        or relative_grid.as_posix() != raw_grid_path
        or any(part in {"", ".", ".."} for part in relative_grid.parts)
    ):
        raise ValueError("selection artifact grid_path must be canonical and relative")
    grid_path = (staged_root / relative_grid).resolve()
    if not grid_path.is_relative_to(staged_root):
        raise ValueError("selection artifact grid_path escapes the staged root")
    stage_name = selected.get("stage")
    if not isinstance(stage_name, str) or not stage_name:
        raise ValueError("selection artifact has no stage")
    try:
        grid_payload = grid_path.read_bytes()
    except OSError as error:
        raise ValueError(f"invalid training grid {grid_path}") from error
    document = _object_from_payload(grid_payload, path=grid_path, kind="training grid")
    regenerated = select_grid(
        document,
        grid_path=grid_path,
        stage_name=stage_name,
        root=staged_root,
        audit_contract=audit_contract,
        grid_payload=grid_payload,
    )
    require_exact_json(
        selected,
        regenerated,
        location="selection artifact after receipt-revalidated frozen grid replay",
    )
    return selected


def _atomic_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--audit-contract", required=True)
    parser.add_argument("--audit-contract-sha256", required=True)
    parser.add_argument("--out")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="verify the frozen grid and training-source digests without opening cell artifacts",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.preflight_only and args.out is None:
        parser.error("--out is required unless --preflight-only is set")
    root = Path(args.root).resolve()
    grid_path = Path(args.grid).resolve()
    document = _read_object(grid_path, "training grid")
    audit_contract = load_paper_audit_contract(
        args.audit_contract,
        args.audit_contract_sha256,
    )
    if args.preflight_only:
        preflight = _preflight_training_registration(
            document,
            grid_path=grid_path,
            stage_name=args.stage,
            root=root,
            audit_contract=audit_contract,
        )
        print(
            json.dumps(
                {
                    "grid_sha256": preflight.grid_sha256,
                    "source_sha256": preflight.source_sha256,
                },
                sort_keys=True,
            )
        )
        return 0
    artifact = select_grid(
        document,
        grid_path=grid_path,
        stage_name=args.stage,
        root=root,
        audit_contract=audit_contract,
    )
    _atomic_json(Path(args.out), artifact)
    print(json.dumps(artifact["winner"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
