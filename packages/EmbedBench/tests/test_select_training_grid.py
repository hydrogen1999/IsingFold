from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
REGISTERED_EXTRA_ARGS = [
    "--hidden",
    "8",
    "--layers",
    "1",
    "--heads",
    "1",
    "--lr",
    "0.001",
    "--tol",
    "0.05",
    "--rank-margin",
    "0.05",
    "--stage1-rank-margin",
    "0.10",
    "--lcb-z",
    "1.0",
    "--stage1-weight",
    "0.25",
    "--lambda-connectivity",
    "0.1",
    "--lambda-robustness",
    "0.1",
    "--deploy-max-free",
    "8",
    "--evaluation-support",
    "full",
]
CORE_RUNTIME_DISTRIBUTIONS = (
    "embedbench",
    "numpy",
    "scipy",
    "networkx",
    "dimod",
    "dwave-samplers",
    "dwave-networkx",
    "minorminer",
)
PAPER_PIPELINE_SCRIPTS = (
    "aggregate_quality_v2_paper.py",
    "evaluate_quality_v2.py",
    "quality_v2_paper_contract.py",
    "rescore_quality.py",
    "run_training_grid.py",
    "select_training_grid.py",
    "train_quality_v2.py",
    "training_artifacts.py",
    "training_splits.py",
)


def _load_modules():
    scripts = str(ROOT / "scripts")
    sys.path.insert(0, scripts)
    try:
        runner_spec = importlib.util.spec_from_file_location(
            "run_training_grid", ROOT / "scripts" / "run_training_grid.py"
        )
        assert runner_spec is not None and runner_spec.loader is not None
        runner = importlib.util.module_from_spec(runner_spec)
        sys.modules["run_training_grid"] = runner
        runner_spec.loader.exec_module(runner)
        selector_spec = importlib.util.spec_from_file_location(
            "select_training_grid", ROOT / "scripts" / "select_training_grid.py"
        )
        assert selector_spec is not None and selector_spec.loader is not None
        selector = importlib.util.module_from_spec(selector_spec)
        selector_spec.loader.exec_module(selector)
        runner._test_selector = selector
        return runner, selector
    finally:
        sys.path.remove(scripts)


def test_canonical_validation_runtime_is_idempotent_and_exact() -> None:
    _, selector = _load_modules()

    first = selector._configure_canonical_replay_runtime()
    second = selector._configure_canonical_replay_runtime()

    assert first == selector.CANONICAL_VALIDATION_REPLAY_POLICY
    assert second == first


def test_canonical_validation_runtime_rejects_initialized_interop_pool() -> None:
    program = """
import torch
torch.set_num_interop_threads(2)
import select_training_grid
select_training_grid._configure_canonical_replay_runtime()
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env={"PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}"},
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "fresh Python process" in completed.stderr


def test_relocated_path_match_uses_staged_symlink_path(tmp_path: Path) -> None:
    _, selector = _load_modules()
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "quality.jsonl").write_text("{}\n", encoding="utf-8")
    root = tmp_path / "stage"
    release = root / "runs" / "release"
    release.parent.mkdir(parents=True)
    release.symlink_to(raw, target_is_directory=True)

    expected = release / "quality.jsonl"
    relocated = "/remote/stage/runs/release/quality.jsonl"

    assert selector._relocated_path_matches(relocated, expected, root)


def test_validation_projection_is_safe_for_the_frozen_legacy_deployment_view() -> None:
    _, selector = _load_modules()
    record = {
        "instance_id": "validation-row",
        "focus": 0,
        "candidates": [[0], [1]],
        "problem": {"h": {"0": 0.25}, "J": [], "e0": -0.25},
        "p_solve": [0.9, 0.1],
        "stage": [2, 2],
        "best_index": 0,
        "resource_index": 0,
        "original_index": 1,
        "future_oracle_label": 0.99,
    }

    projected = selector._validation_model_record(record)

    assert not {
        "p_solve",
        "stage",
        "best_index",
        "resource_index",
        "original_index",
        "future_oracle_label",
    } & set(projected)
    assert projected["problem"] == {"h": {"0": 0.25}, "J": []}
    # The preregistered models_chain implementation branches through this exact access.
    legacy_current = (
        set(projected["candidates"][projected["original_index"]])
        if projected.get("original_index", -1) >= 0
        else set()
    )
    assert legacy_current == set()


def _fixture(tmp_path: Path):
    runner, selector = _load_modules()
    from quality_v2_paper_contract import PAPER_VERIFIER_PACKAGE_FILES

    (tmp_path / "src" / "embedbench").mkdir(parents=True)
    (tmp_path / "scripts").mkdir()
    (tmp_path / "runs" / "release").mkdir(parents=True)
    (tmp_path / "data").mkdir()
    (tmp_path / "src" / "embedbench" / "model.py").write_text("VALUE = 1\n")
    for script in PAPER_PIPELINE_SCRIPTS:
        (tmp_path / "scripts" / script).write_text(f"# {script}\n")
    for relative in PAPER_VERIFIER_PACKAGE_FILES:
        target = tmp_path / "scripts" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# {relative}\n", encoding="utf-8")
    corpus = tmp_path / "runs" / "release" / "quality.jsonl"
    validation_record = {
        "instance_id": "x",
        "focus": 0,
        "n_vars": 2,
        "window_nodes": [0, 1, 4],
        "window_edges": [[0, 4], [1, 4]],
        "frozen": {"2": [5]},
        "all_chains": {"2": [5]},
        "frozen_adjacency": {"0": [5], "1": [5]},
        "neighbours": [2],
        "edge_J": [-1.0],
        "neighbour_degree": [1],
        "neighbour_chain_size": [1],
        "neighbour_h": [0.0],
        "candidates": [[0], [1], [0, 4]],
        "Q": [1, 1, 2],
        "p_solve": [0.9, 0.7, 0.1],
        "stage": [2, 2, 2],
        "best_index": 0,
        "resource_index": 0,
        "original_index": 1,
        "source": "minorminer",
        "topology": "chimera",
        "size": 1,
        "difficulty": "test",
        "l_cap": 4,
        "focus_h": 0.25,
        "problem": {
            "h": {"0": 0.25, "2": 0.0},
            "J": [[0, 2, -1.0]],
            "e0": -1.25,
        },
    }
    corpus.write_text(json.dumps(validation_record) + "\n", encoding="utf-8")
    splits = tmp_path / "data" / "splits.json"
    splits.write_text(
        json.dumps(
            {
                "schema": "embedbench.split-manifest",
                "schema_version": 2,
                "provenance": {
                    "inputs": [
                        {
                            "file": corpus.name,
                            "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
                        }
                    ]
                },
                "splits": {corpus.name: {"x": "val"}},
            }
        ),
        encoding="utf-8",
    )
    grid = tmp_path / "grid.json"
    document = {
        "schema": "embedbench.training-grid",
        "schema_version": 1,
        "grid_id": "selection-test",
        "selection": {
            "primary_metric": "mean_finite_budget_regret",
            "seeds": [0, 1, 2],
        },
        "stages": {
            "screen": {
                "trainer": "scripts/train_quality_v2.py",
                "inputs": "runs/release/quality*.jsonl",
                "splits": "data/splits.json",
                "output_dir": "runs/grid/screen",
                "checkpoint_dir": "runs/grid/checkpoints",
                "epochs": 1,
                "axis_order": ["arch", "objective_variant", "seed"],
                "axes": {
                    "arch": ["mpnn", "gin"],
                    "objective_variant": ["full"],
                    "seed": [0, 1, 2],
                },
                "extra_args": REGISTERED_EXTRA_ARGS,
                "execution": {
                    "assignment": "cell_index_modulo",
                    "modulus": 1,
                    "remainder_to_site": {"0": "test-site"},
                    "required_device_type": "cuda",
                    "slurm_required_sites": [],
                    "slurm_forbidden_sites": [],
                },
            }
        },
    }
    grid.write_text(json.dumps(document), encoding="utf-8")
    cells = runner.expand_stage(document, "screen")
    data_provenance = runner._data_provenance((corpus,), splits)
    grid_sha256 = runner._sha256(grid)
    source_sha256 = runner._source_sha256(tmp_path)
    return runner, selector, document, grid, cells, data_provenance, grid_sha256, source_sha256


def _paper_contract(tmp_path: Path, selector, document: dict, grid: Path, cells: list):
    from quality_v2_paper_contract import EXPECTED_PAPER_AUDIT_CONTRACT

    contract_document = copy.deepcopy(EXPECTED_PAPER_AUDIT_CONTRACT)
    stage_name = next(iter(document["stages"]))
    contract_document["selection"] = {
        "artifact_schema": "embedbench.training-grid-selection",
        "artifact_schema_version": 3,
        "grid_id": document["grid_id"],
        "grid_path": grid.relative_to(tmp_path).as_posix(),
        "stage": stage_name,
        "registered_cell_count": len(cells),
        "registered_seeds": document["selection"]["seeds"],
        "validation_only": True,
    }
    stage = document["stages"][stage_name]
    inputs, split_manifest = selector.grid_runner._stage_inputs(stage, tmp_path)
    contract_document["training_registration"] = {
        "grid_sha256": selector.grid_runner._sha256(grid),
        "source_sha256": selector.grid_runner._source_sha256(tmp_path),
        "selection": copy.deepcopy(document["selection"]),
        "stage": copy.deepcopy(stage),
        "data_provenance": selector.grid_runner._data_provenance(inputs, split_manifest),
        "validation_replay": copy.deepcopy(
            EXPECTED_PAPER_AUDIT_CONTRACT["training_registration"]["validation_replay"]
        ),
    }
    path = tmp_path / "paper-audit-contract.json"
    path.write_text(json.dumps(contract_document), encoding="utf-8")
    return selector.PaperAuditContract(
        path=path.resolve(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checksum_path=path.with_suffix(".sha256").resolve(),
        checksum_sha256="0" * 64,
        document=contract_document,
    )


def _select_grid(selector, document: dict, grid: Path, stage_name: str, root: Path):
    cells = selector.grid_runner.expand_stage(document, stage_name)
    return selector.select_grid(
        document,
        grid_path=grid,
        stage_name=stage_name,
        root=root,
        audit_contract=_paper_contract(root, selector, document, grid, cells),
    )


def _write_cell(
    runner,
    root: Path,
    cell,
    metric: float,
    data_provenance: dict[str, object],
    grid_sha256: str,
    source_sha256: str,
    *,
    grid_path: Path | None = None,
    deploy_view: bool = False,
    deployment_host_contract_override: dict[str, object] | None = None,
) -> tuple[Path, Path, Path]:
    grid_path = grid_path or (root / "grid.json")
    document = json.loads(grid_path.read_text(encoding="utf-8"))
    stage = runner._stage(document, cell.stage)
    result = root / str(stage["output_dir"]) / f"{cell.cell_id}.json"
    receipt = result.with_suffix(".receipt.json")
    checkpoint = runner._expected_checkpoint(
        root / str(stage["checkpoint_dir"]) / cell.cell_id,
        cell,
    )
    result.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    pytest = __import__("pytest")
    pytest.importorskip("torch")
    from embedbench.hamiltonian_context import hamiltonian_context_contract
    from embedbench.models_quality_v2 import (
        build_quality_v2_model,
        save_quality_v2_model,
    )

    training_run_id = hashlib.sha256(cell.cell_id.encode()).hexdigest()[:32]
    preprocessing = {
        "deploy_view": deploy_view,
        "deploy_max_free": 8,
        "neighbour_feats": False,
        "encoder": "heterogeneous" if cell.params["arch"] == "hetero" else "chain",
        "hamiltonian_context": hamiltonian_context_contract(),
    }
    deployment_host_contract = deployment_host_contract_override or {
        "mode": "not_applied",
        "source_manifests": [],
    }
    loss_weights = {
        "quality_probability": 1.0,
        "within_state_rank": 0.5,
        "residual_connectivity_proxy": 0.1,
        "robustness": 0.1,
        "terminal_qubits": 0.0,
    }
    device_type = str(stage["execution"]["required_device_type"])
    runtime_provenance = {
        "schema": "embedbench.runtime-environment",
        "schema_version": 1,
        "python": {
            "implementation": "CPython",
            "version": "3.12.0",
            "executable": "/staged/python",
        },
        "torch": {
            "version": "2.7.0",
            "cuda_available": device_type == "cuda",
            "cuda_runtime_version": "12.8" if device_type == "cuda" else None,
            "cudnn_version": 90701 if device_type == "cuda" else None,
        },
        "device": {
            "selected": device_type,
            "type": device_type,
            "index": 0 if device_type == "cuda" else None,
            "gpu": (
                {
                    "name": "test-gpu",
                    "compute_capability": [9, 0],
                    "total_memory_bytes": 1,
                }
                if device_type == "cuda"
                else None
            ),
        },
        "packages": {name: None for name in CORE_RUNTIME_DISTRIBUTIONS},
    }
    budget_ratios = [1.0, 1.1, 1.25, 1.5, None]
    training_scope = {"limit": None, "full_fixed_train_validation": True}
    command, expected_result, expected_checkpoint = runner.build_command(
        document,
        cell,
        root=root,
        python="/staged/python",
        device=device_type,
    )
    assert expected_result == result
    assert expected_checkpoint == checkpoint

    remote_root = Path("/remote/staged/embedbench")

    def relocate(value: str) -> str:
        path = Path(value)
        try:
            relative = path.relative_to(root)
        except ValueError:
            return value
        return str(remote_root / relative)

    command = [relocate(value) for value in command]
    inputs, split_manifest = runner._stage_inputs(stage, root)
    result_args = {
        "files": [relocate(str(path)) for path in inputs],
        "arch": cell.params["arch"],
        "objective_variant": cell.params["objective_variant"],
        "seeds": str(cell.params["seed"]),
        "epochs": 1,
        "hidden": 8,
        "layers": 1,
        "heads": 1,
        "lr": 0.001,
        "tol": 0.05,
        "rank_margin": 0.05,
        "stage1_rank_margin": 0.10,
        "lcb_z": 1.0,
        "stage1_weight": 0.25,
        "lambda_connectivity": 0.1,
        "lambda_robustness": 0.1,
        "limit": None,
        "neighbour_feats": False,
        "deploy_view": deploy_view,
        "deploy_max_free": 8,
        "device": device_type,
        "splits": relocate(str(split_manifest)),
        "evaluation_support": "full",
        "save": relocate(str(root / str(stage["checkpoint_dir"]) / cell.cell_id)),
        "out": relocate(str(result)),
    }
    model = build_quality_v2_model(arch=cell.params["arch"], hidden=8, layers=1, heads=1)
    validation_records = runner._test_selector._load_validation_records(inputs, split_manifest)
    replay_sweep, _ = runner._test_selector._replay_validation_checkpoint(
        model,
        validation_records,
        preprocessing,
        lcb_z=1.0,
    )
    training_device_sweep = copy.deepcopy(replay_sweep)
    for key in ("1.00x", "1.10x", "1.25x", "1.50x"):
        training_device_sweep["budgets"][key]["selectors"]["mean"]["regret"] = metric
    training_device_sweep["primary_metric"] = metric
    save_quality_v2_model(
        model,
        checkpoint,
        metadata={
            "objective_variant": cell.params["objective_variant"],
            "seed": cell.params["seed"],
            "training_run_id": training_run_id,
            "best_epoch": 1,
            "selected_validation_metric": metric,
            "split_sha256": data_provenance["split_manifest"]["sha256"],
            "data_provenance": data_provenance,
            "runtime_provenance": runtime_provenance,
            "evaluation_support": "full",
            "registered_budget_ratios": budget_ratios,
            "primary_validation_metric": "mean_finite_budget_regret",
            "lcb_z": 1.0,
            "selection_contract": (
                "quality_mean_or_lcb_subject_to_exact_feasibility_and_budget-v1"
            ),
            "auxiliary_heads_used_for_selection": False,
            "test_evaluated": False,
            "test_partition_encoded": False,
            "training_scope": training_scope,
            "preprocessing": preprocessing,
            "deployment_host_contract": deployment_host_contract,
            "loss_weights": loss_weights,
        },
    )
    result.write_text(
        json.dumps(
            {
                "artifact_schema": "embedbench.quality-v2-training-results",
                "artifact_schema_version": 1,
                "args": result_args,
                "data_provenance": data_provenance,
                "runtime_provenance": runtime_provenance,
                "preprocessing": preprocessing,
                "deployment_host_contract": deployment_host_contract,
                "training_scope": training_scope,
                "test_locked": True,
                "test_partition_encoded": False,
                "results": [
                    {
                        **cell.params,
                        "hidden": 8,
                        "layers": 1,
                        "heads": 1,
                        "training_run_id": training_run_id,
                        "best_epoch": 1,
                        "primary_validation_metric": "mean_finite_budget_regret",
                        "training_scope": training_scope,
                        "evaluation_support": "full",
                        "registered_budget_ratios": budget_ratios,
                        "selection_contract": (
                            "quality_mean_or_lcb_subject_to_exact_feasibility_and_budget-v1"
                        ),
                        "auxiliary_heads_used_for_selection": False,
                        "preprocessing": preprocessing,
                        "deployment_host_contract": deployment_host_contract,
                        "loss_weights": loss_weights,
                        "test_evaluated": False,
                        "test_partition_encoded": False,
                        "test": None,
                        "n_val": len(validation_records),
                        "validation": {
                            "budget_sweep": training_device_sweep,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    receipt.write_text(
        json.dumps(
            {
                "schema": "embedbench.training-receipt",
                "schema_version": 1,
                "grid_id": document["grid_id"],
                "grid_sha256": grid_sha256,
                "source_sha256": source_sha256,
                "data_provenance": data_provenance,
                "cell": cell._asdict(),
                "command": command,
                "execution_site": runner.expected_execution_site(stage, cell),
                "slurm": {
                    "array_job_id": None,
                    "array_task_id": None,
                    "job_id": None,
                },
                "result": relocate(str(result)),
                "result_sha256": runner._sha256(result),
                "checkpoint": relocate(str(checkpoint)),
                "checkpoint_sha256": runner._sha256(checkpoint),
                "test_locked": True,
            }
        ),
        encoding="utf-8",
    )
    return result, receipt, checkpoint


def test_selects_lowest_replayed_seed_mean_then_freezes_best_validation_seed(
    tmp_path: Path,
) -> None:
    (
        runner,
        selector,
        document,
        grid,
        cells,
        provenance,
        grid_sha,
        source_sha,
    ) = _fixture(tmp_path)
    metrics = {
        "mpnn": [0.10, 0.50, 0.50],
        "gin": [0.30, 0.30, 0.30],
    }
    for cell in cells:
        _write_cell(
            runner,
            tmp_path,
            cell,
            metrics[cell.params["arch"]][cell.params["seed"]],
            provenance,
            grid_sha,
            source_sha,
        )

    contract = _paper_contract(tmp_path, selector, document, grid, cells)
    artifact = selector.select_grid(
        document,
        grid_path=grid,
        stage_name="screen",
        root=tmp_path,
        audit_contract=contract,
    )

    assert artifact["schema_version"] == 3
    assert artifact["paper_audit_contract"] == contract.public_binding()
    assert artifact["grid_path"] == "grid.json"
    assert artifact["registered_cell_count"] == len(cells)
    assert artifact["test_records_parsed"] is True
    assert artifact["test_records_parsed_for_partition_routing_only"] is True
    assert artifact["test_records_encoded"] is False
    assert artifact["test_labels_used_for_selection"] is False
    assert artifact["test_evaluated"] is False
    assert artifact["validation_metric_source"] == "canonical_cpu_checkpoint_replay"
    assert artifact["validation_replay_policy"] == (selector.CANONICAL_VALIDATION_REPLAY_POLICY)
    assert len(artifact["configurations"]) == 2
    winner_index = min(
        range(len(artifact["configurations"])),
        key=lambda index: (
            artifact["configurations"][index]["mean_validation_metric"],
            index,
        ),
    )
    expected_winner = artifact["configurations"][winner_index]
    assert artifact["winner"]["params"] == expected_winner["params"]
    expected_seed = min(
        expected_winner["seeds"],
        key=lambda row: (row["validation_metric"], [0, 1, 2].index(row["seed"])),
    )
    assert artifact["winner"]["selected_seed"] == expected_seed["seed"]
    assert [row["seed"] for row in artifact["winner"]["paper_checkpoints"]] == [0, 1, 2]
    assert artifact["winner"]["checkpoint_sha256"] == runner._sha256(
        tmp_path / artifact["winner"]["checkpoint"]
    )
    for configuration in artifact["configurations"]:
        assert math.isclose(
            configuration["mean_validation_metric"],
            statistics.fmean(row["validation_metric"] for row in configuration["seeds"]),
        )
        assert all(row["validation_replay"]["record_count"] == 1 for row in configuration["seeds"])
        assert all(
            row["validation_metric_source"] == "canonical_cpu_checkpoint_replay"
            for row in configuration["seeds"]
        )
        assert all(
            row["validation_replay"]["runtime_policy"]
            == selector.CANONICAL_VALIDATION_REPLAY_POLICY
            for row in configuration["seeds"]
        )

    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(artifact), encoding="utf-8")
    validated = selector.revalidate_selection_artifact(
        selection,
        hashlib.sha256(selection.read_bytes()).hexdigest(),
        root=tmp_path,
        audit_contract=contract,
    )
    assert validated == artifact

    captured = selection.read_bytes()
    selection.write_text("{}", encoding="utf-8")
    assert (
        selector.revalidate_selection_artifact(
            selection,
            hashlib.sha256(captured).hexdigest(),
            root=tmp_path,
            audit_contract=contract,
            selection_payload=captured,
        )
        == artifact
    )
    selection.write_bytes(captured)

    legacy = copy.deepcopy(artifact)
    legacy["schema_version"] = 2
    selection.write_text(json.dumps(legacy), encoding="utf-8")
    with pytest.raises(ValueError, match="selection schema version 3"):
        selector.revalidate_selection_artifact(
            selection,
            hashlib.sha256(selection.read_bytes()).hexdigest(),
            root=tmp_path,
            audit_contract=contract,
        )

    changed_replay = copy.deepcopy(artifact)
    changed_replay["configurations"][0]["seeds"][0]["validation_replay"]["sha256"] = "f" * 64
    selection.write_text(json.dumps(changed_replay), encoding="utf-8")
    with pytest.raises(ValueError, match="receipt-revalidated frozen grid replay"):
        selector.revalidate_selection_artifact(
            selection,
            hashlib.sha256(selection.read_bytes()).hexdigest(),
            root=tmp_path,
            audit_contract=contract,
        )
    selection.write_bytes(captured)

    receipt = tmp_path / artifact["configurations"][0]["seeds"][0]["receipt"]
    receipt.unlink()
    try:
        selector.revalidate_selection_artifact(
            selection,
            hashlib.sha256(selection.read_bytes()).hexdigest(),
            root=tmp_path,
            audit_contract=contract,
        )
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("selection revalidation must reopen every registered receipt")


def test_selection_rejects_unregistered_grid_before_opening_cell_artifacts(
    tmp_path: Path,
) -> None:
    _runner, selector, document, grid, cells, *_ = _fixture(tmp_path)
    contract = _paper_contract(tmp_path, selector, document, grid, cells)
    contract.document["training_registration"]["grid_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="grid SHA-256 differs"):
        selector.select_grid(
            document,
            grid_path=grid,
            stage_name="screen",
            root=tmp_path,
            audit_contract=contract,
        )


def test_selection_revalidation_rejects_boolean_contract_type_confusion(
    tmp_path: Path,
) -> None:
    (
        runner,
        selector,
        document,
        grid,
        cells,
        provenance,
        grid_sha,
        source_sha,
    ) = _fixture(tmp_path)
    for cell in cells:
        _write_cell(
            runner,
            tmp_path,
            cell,
            0.2,
            provenance,
            grid_sha,
            source_sha,
        )
    contract = _paper_contract(tmp_path, selector, document, grid, cells)
    artifact = selector.select_grid(
        document,
        grid_path=grid,
        stage_name="screen",
        root=tmp_path,
        audit_contract=contract,
    )
    # ``True == 1`` under ordinary Python mapping equality.  The artifact boundary
    # must nevertheless preserve the exact JSON type registered by the contract.
    artifact["paper_audit_contract"]["schema_version"] = True
    selection = tmp_path / "type-confused-selection.json"
    selection.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="paper-audit contract binding"):
        selector.revalidate_selection_artifact(
            selection,
            hashlib.sha256(selection.read_bytes()).hexdigest(),
            root=tmp_path,
            audit_contract=contract,
        )


def test_selection_rejects_boolean_seed_zero_across_all_training_artifacts(
    tmp_path: Path,
) -> None:
    (
        runner,
        selector,
        document,
        grid,
        cells,
        provenance,
        grid_sha,
        source_sha,
    ) = _fixture(tmp_path)
    artifacts = [
        _write_cell(
            runner,
            tmp_path,
            cell,
            0.2,
            provenance,
            grid_sha,
            source_sha,
        )
        for cell in cells
    ]
    result, receipt, checkpoint = artifacts[0]
    result_document = json.loads(result.read_text(encoding="utf-8"))
    result_document["results"][0]["seed"] = False
    result.write_text(json.dumps(result_document), encoding="utf-8")

    torch = pytest.importorskip("torch")
    checkpoint_document = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_document["meta"]["seed"] = False
    torch.save(checkpoint_document, checkpoint)

    receipt_document = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_document["cell"]["params"]["seed"] = False
    receipt_document["result_sha256"] = runner._sha256(result)
    receipt_document["checkpoint_sha256"] = runner._sha256(checkpoint)
    receipt.write_text(json.dumps(receipt_document), encoding="utf-8")

    with pytest.raises(ValueError, match="receipt cell"):
        _select_grid(selector, document, grid, "screen", tmp_path)


@pytest.mark.parametrize(
    ("artifact_kind", "message"),
    [
        ("result", "result axis 'seed'"),
        ("checkpoint", "checkpoint metadata mismatch"),
    ],
)
def test_selection_rejects_boolean_seed_zero_in_individual_training_artifacts(
    tmp_path: Path,
    artifact_kind: str,
    message: str,
) -> None:
    (
        runner,
        selector,
        document,
        grid,
        cells,
        provenance,
        grid_sha,
        source_sha,
    ) = _fixture(tmp_path)
    artifacts = [
        _write_cell(
            runner,
            tmp_path,
            cell,
            0.2,
            provenance,
            grid_sha,
            source_sha,
        )
        for cell in cells
    ]
    result, receipt, checkpoint = artifacts[0]
    receipt_document = json.loads(receipt.read_text(encoding="utf-8"))
    if artifact_kind == "result":
        result_document = json.loads(result.read_text(encoding="utf-8"))
        result_document["results"][0]["seed"] = False
        result.write_text(json.dumps(result_document), encoding="utf-8")
        receipt_document["result_sha256"] = runner._sha256(result)
    else:
        torch = pytest.importorskip("torch")
        checkpoint_document = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_document["meta"]["seed"] = False
        torch.save(checkpoint_document, checkpoint)
        receipt_document["checkpoint_sha256"] = runner._sha256(checkpoint)
    receipt.write_text(json.dumps(receipt_document), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _select_grid(selector, document, grid, "screen", tmp_path)


def test_training_source_preflight_fails_before_artifact_access_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _runner, selector, document, grid, cells, *_ = _fixture(tmp_path)
    contract = _paper_contract(tmp_path, selector, document, grid, cells)
    expected_source = contract.training_registration["source_sha256"]
    preflight = selector._preflight_training_registration(
        document,
        grid_path=grid,
        stage_name="screen",
        root=tmp_path,
        audit_contract=contract,
    )
    assert preflight.source_sha256 == expected_source

    (tmp_path / "src" / "embedbench" / "model.py").write_text("VALUE = 2\n")

    def forbidden_artifact_access(*_args, **_kwargs):
        raise AssertionError("source mismatch must fail before cell artifact access")

    monkeypatch.setattr(selector, "_validated_cell", forbidden_artifact_access)
    output = tmp_path / "selection.json"
    with pytest.raises(ValueError, match=f"expected {expected_source}, observed"):
        artifact = selector.select_grid(
            document,
            grid_path=grid,
            stage_name="screen",
            root=tmp_path,
            audit_contract=contract,
        )
        selector._atomic_json(output, artifact)
    assert not output.exists()


def test_selection_rejects_cell_symlink_escape_before_reading_it(tmp_path: Path) -> None:
    (
        runner,
        selector,
        document,
        grid,
        cells,
        provenance,
        grid_sha,
        source_sha,
    ) = _fixture(tmp_path)
    artifacts = [
        _write_cell(
            runner,
            tmp_path,
            cell,
            0.2,
            provenance,
            grid_sha,
            source_sha,
        )
        for cell in cells
    ]
    result = artifacts[0][0]
    outside = tmp_path.parent / f"{tmp_path.name}-outside-result.json"
    shutil.copyfile(result, outside)
    result.unlink()
    result.symlink_to(outside)

    with __import__("pytest").raises(ValueError, match="outside the staged root"):
        _select_grid(selector, document, grid, "screen", tmp_path)


def test_selection_fails_closed_on_incomplete_or_partial_grid(tmp_path: Path) -> None:
    (
        runner,
        selector,
        document,
        grid,
        cells,
        provenance,
        grid_sha,
        source_sha,
    ) = _fixture(tmp_path)
    artifacts = []
    for cell in cells:
        artifacts.append(
            _write_cell(
                runner,
                tmp_path,
                cell,
                0.2,
                provenance,
                grid_sha,
                source_sha,
            )
        )
    artifacts[-1][1].unlink()

    try:
        _select_grid(selector, document, grid, "screen", tmp_path)
    except ValueError as error:
        assert "incomplete" in str(error)
    else:
        raise AssertionError("incomplete grid must fail closed")

    _, receipt, _ = _write_cell(
        runner,
        tmp_path,
        cells[-1],
        0.2,
        provenance,
        grid_sha,
        source_sha,
    )
    result = artifacts[0][0]
    payload = json.loads(result.read_text())
    payload["training_scope"] = {"limit": 128, "full_fixed_train_validation": False}
    result.write_text(json.dumps(payload), encoding="utf-8")
    receipt_payload = json.loads(artifacts[0][1].read_text())
    receipt_payload["result_sha256"] = runner._sha256(result)
    artifacts[0][1].write_text(json.dumps(receipt_payload), encoding="utf-8")

    try:
        _select_grid(selector, document, grid, "screen", tmp_path)
    except ValueError as error:
        assert "full fixed train/validation" in str(error)
    else:
        raise AssertionError("partial training grid must fail closed")


def test_selection_binds_document_and_recomputes_registered_metric(tmp_path: Path) -> None:
    (
        runner,
        selector,
        document,
        grid,
        cells,
        provenance,
        grid_sha,
        source_sha,
    ) = _fixture(tmp_path)
    artifacts = [
        _write_cell(
            runner,
            tmp_path,
            cell,
            0.2,
            provenance,
            grid_sha,
            source_sha,
        )
        for cell in cells
    ]

    shortened = copy.deepcopy(document)
    shortened["stages"]["screen"]["axes"]["arch"] = ["mpnn"]
    try:
        _select_grid(selector, shortened, grid, "screen", tmp_path)
    except ValueError as error:
        assert "grid document" in str(error)
    else:
        raise AssertionError("selection document must be bound to grid bytes")

    result, receipt, _ = artifacts[0]
    payload = json.loads(result.read_text())
    stored_metric = payload["results"][0]["validation"]["budget_sweep"]["primary_metric"]
    payload["results"][0]["validation"]["budget_sweep"]["primary_metric"] = stored_metric + 0.123
    result.write_text(json.dumps(payload), encoding="utf-8")
    receipt_payload = json.loads(receipt.read_text())
    receipt_payload["result_sha256"] = runner._sha256(result)
    receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")

    try:
        _select_grid(selector, document, grid, "screen", tmp_path)
    except ValueError as error:
        assert "recomputed" in str(error)
    else:
        raise AssertionError("stored scalar must match per-budget regrets")


def test_selection_ranks_canonical_replay_not_training_device_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        _runner,
        selector,
        document,
        grid,
        cells,
        _provenance,
        _grid_sha,
        _source_sha,
    ) = _fixture(tmp_path)
    canonical = {
        "mpnn": [0.30, 0.10, 0.20],
        "gin": [0.40, 0.40, 0.40],
    }
    training_device = {
        "mpnn": [0.01, 0.90, 0.90],
        "gin": [0.20, 0.20, 0.20],
    }
    visited: list[str] = []

    def validated_cell(cell, **_kwargs):
        visited.append(cell.cell_id)
        arch = cell.params["arch"]
        seed = cell.params["seed"]
        return {
            "cell_id": cell.cell_id,
            "index": cell.index,
            "seed": seed,
            "validation_metric": canonical[arch][seed],
            "validation_metric_source": "canonical_cpu_checkpoint_replay",
            "training_device_validation_metric": training_device[arch][seed],
            "result": f"results/{cell.cell_id}.json",
            "result_sha256": "1" * 64,
            "checkpoint": f"checkpoints/{cell.cell_id}.pt",
            "checkpoint_sha256": "2" * 64,
            "coverage": {"registered": True},
            "execution_site": "test-site",
            "runtime_compatibility_signature": {"runtime": "fixed"},
            "validation_replay": {"record_count": 1},
        }

    monkeypatch.setattr(selector, "_validated_cell", validated_cell)
    artifact = _select_grid(selector, document, grid, "screen", tmp_path)

    assert visited == [cell.cell_id for cell in cells]
    assert artifact["winner"]["params"]["arch"] == "mpnn"
    assert artifact["winner"]["selected_seed"] == 1
    diagnostic_winner = min(
        artifact["configurations"],
        key=lambda configuration: statistics.fmean(
            row["training_device_validation_metric"] for row in configuration["seeds"]
        ),
    )
    assert diagnostic_winner["params"]["arch"] == "gin"


def test_selection_records_model_dependent_cross_device_drift(tmp_path: Path) -> None:
    (
        runner,
        selector,
        document,
        grid,
        cells,
        provenance,
        grid_sha,
        source_sha,
    ) = _fixture(tmp_path)
    artifacts = [
        _write_cell(
            runner,
            tmp_path,
            cell,
            0.2,
            provenance,
            grid_sha,
            source_sha,
        )
        for cell in cells
    ]
    result, receipt, checkpoint = artifacts[0]
    result_document = json.loads(result.read_text(encoding="utf-8"))
    sweep = result_document["results"][0]["validation"]["budget_sweep"]
    training_device_metric = float(sweep["primary_metric"]) + 0.123
    for key in ("1.00x", "1.10x", "1.25x", "1.50x"):
        mean = sweep["budgets"][key]["selectors"]["mean"]
        mean["regret"] = training_device_metric
        current = mean["selection_indices"][0]
        mean["selection_indices"][0] = 1 if current == 0 else 0
    sweep["primary_metric"] = training_device_metric
    result.write_text(json.dumps(result_document), encoding="utf-8")

    torch = pytest.importorskip("torch")
    checkpoint_document = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_document["meta"]["selected_validation_metric"] = training_device_metric
    torch.save(checkpoint_document, checkpoint)
    receipt_document = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_document["result_sha256"] = runner._sha256(result)
    receipt_document["checkpoint_sha256"] = runner._sha256(checkpoint)
    receipt.write_text(json.dumps(receipt_document), encoding="utf-8")

    artifact = _select_grid(selector, document, grid, "screen", tmp_path)
    rows = [row for configuration in artifact["configurations"] for row in configuration["seeds"]]
    changed = next(row for row in rows if row["cell_id"] == cells[0].cell_id)
    diagnostic = changed["cross_device_validation_diagnostic"]

    assert changed["training_device_validation_metric"] == training_device_metric
    assert changed["validation_metric"] != training_device_metric
    assert diagnostic["selection_index_disagreement_count"] == 4
    assert diagnostic["canonical_cpu_minus_training_device_metric"] == pytest.approx(
        changed["validation_metric"] - training_device_metric
    )
    assert diagnostic["used_for_acceptance"] is False
    assert diagnostic["used_for_ranking"] is False


def test_selection_rejects_changed_model_independent_budget_evidence(tmp_path: Path) -> None:
    (
        runner,
        selector,
        document,
        grid,
        cells,
        provenance,
        grid_sha,
        source_sha,
    ) = _fixture(tmp_path)
    artifacts = [
        _write_cell(
            runner,
            tmp_path,
            cell,
            0.2,
            provenance,
            grid_sha,
            source_sha,
        )
        for cell in cells
    ]
    result, receipt, _checkpoint = artifacts[0]
    result_document = json.loads(result.read_text(encoding="utf-8"))
    budget = result_document["results"][0]["validation"]["budget_sweep"]["budgets"]["1.00x"]
    budget["reference_total_qubits"][0] += 1
    result.write_text(json.dumps(result_document), encoding="utf-8")
    receipt_document = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_document["result_sha256"] = runner._sha256(result)
    receipt.write_text(json.dumps(receipt_document), encoding="utf-8")

    with pytest.raises(ValueError, match="model-independent validation evidence"):
        _select_grid(selector, document, grid, "screen", tmp_path)


def test_selection_still_binds_checkpoint_to_training_device_diagnostic(
    tmp_path: Path,
) -> None:
    (
        runner,
        selector,
        document,
        grid,
        cells,
        provenance,
        grid_sha,
        source_sha,
    ) = _fixture(tmp_path)
    artifacts = [
        _write_cell(
            runner,
            tmp_path,
            cell,
            0.2,
            provenance,
            grid_sha,
            source_sha,
        )
        for cell in cells
    ]
    _, receipt, checkpoint = artifacts[0]
    torch = pytest.importorskip("torch")
    checkpoint_document = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_document["meta"]["selected_validation_metric"] += 0.123
    torch.save(checkpoint_document, checkpoint)
    receipt_document = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_document["checkpoint_sha256"] = runner._sha256(checkpoint)
    receipt.write_text(json.dumps(receipt_document), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint selected validation metric mismatch"):
        _select_grid(selector, document, grid, "screen", tmp_path)


def test_selection_loads_checkpoint_and_binds_it_to_training_run(tmp_path: Path) -> None:
    (
        runner,
        selector,
        document,
        grid,
        cells,
        provenance,
        grid_sha,
        source_sha,
    ) = _fixture(tmp_path)
    artifacts = [
        _write_cell(
            runner,
            tmp_path,
            cell,
            0.2,
            provenance,
            grid_sha,
            source_sha,
        )
        for cell in cells
    ]
    _, receipt, checkpoint = artifacts[0]
    checkpoint.write_bytes(b"not-a-quality-v2-checkpoint")
    receipt_payload = json.loads(receipt.read_text())
    receipt_payload["checkpoint_sha256"] = runner._sha256(checkpoint)
    receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")

    try:
        _select_grid(selector, document, grid, "screen", tmp_path)
    except ValueError as error:
        assert "checkpoint" in str(error)
    else:
        raise AssertionError("corrupt checkpoint must fail closed")

    result, receipt, _ = _write_cell(
        runner,
        tmp_path,
        cells[0],
        0.2,
        provenance,
        grid_sha,
        source_sha,
    )
    result_payload = json.loads(result.read_text())
    result_payload["results"][0]["training_run_id"] = "f" * 32
    result.write_text(json.dumps(result_payload), encoding="utf-8")
    receipt_payload = json.loads(receipt.read_text())
    receipt_payload["result_sha256"] = runner._sha256(result)
    receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")

    try:
        _select_grid(selector, document, grid, "screen", tmp_path)
    except ValueError as error:
        assert "training run" in str(error)
    else:
        raise AssertionError("stale checkpoint/result pairing must fail closed")


def test_selection_rejects_hidden_per_budget_subcohort(tmp_path: Path) -> None:
    (
        runner,
        selector,
        document,
        grid,
        cells,
        provenance,
        grid_sha,
        source_sha,
    ) = _fixture(tmp_path)
    artifacts = [
        _write_cell(
            runner,
            tmp_path,
            cell,
            0.2,
            provenance,
            grid_sha,
            source_sha,
        )
        for cell in cells
    ]
    result, receipt, _ = artifacts[0]
    payload = json.loads(result.read_text())
    budget = payload["results"][0]["validation"]["budget_sweep"]["budgets"]["1.00x"]
    budget["selectors"]["mean"]["n_selected"] = 5
    receipt_payload = json.loads(receipt.read_text())
    result.write_text(json.dumps(payload), encoding="utf-8")
    receipt_payload["result_sha256"] = runner._sha256(result)
    receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")

    try:
        _select_grid(selector, document, grid, "screen", tmp_path)
    except ValueError as error:
        assert "cohort" in str(error)
    else:
        raise AssertionError("per-budget hidden subcohort must fail closed")


def test_selection_rejects_tampered_receipt_training_command(tmp_path: Path) -> None:
    (
        runner,
        selector,
        document,
        grid,
        cells,
        provenance,
        grid_sha,
        source_sha,
    ) = _fixture(tmp_path)
    artifacts = [
        _write_cell(
            runner,
            tmp_path,
            cell,
            0.2,
            provenance,
            grid_sha,
            source_sha,
        )
        for cell in cells
    ]
    receipt = artifacts[0][1]
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    lr_index = payload["command"].index("--lr") + 1
    payload["command"][lr_index] = "0.02"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    try:
        _select_grid(selector, document, grid, "screen", tmp_path)
    except ValueError as error:
        assert "receipt command" in str(error)
    else:
        raise AssertionError("tampered receipt command must fail closed")


def test_selection_rejects_tampered_result_trainer_arguments(tmp_path: Path) -> None:
    (
        runner,
        selector,
        document,
        grid,
        cells,
        provenance,
        grid_sha,
        source_sha,
    ) = _fixture(tmp_path)
    artifacts = [
        _write_cell(
            runner,
            tmp_path,
            cell,
            0.2,
            provenance,
            grid_sha,
            source_sha,
        )
        for cell in cells
    ]
    result, receipt, _ = artifacts[0]
    payload = json.loads(result.read_text(encoding="utf-8"))
    payload["args"]["hidden"] = 16
    result.write_text(json.dumps(payload), encoding="utf-8")
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_payload["result_sha256"] = runner._sha256(result)
    receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")

    try:
        _select_grid(selector, document, grid, "screen", tmp_path)
    except ValueError as error:
        assert "result args" in str(error)
    else:
        raise AssertionError("tampered trainer result args must fail closed")
