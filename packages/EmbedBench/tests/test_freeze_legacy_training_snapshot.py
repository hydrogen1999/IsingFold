from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "tools" / "legacy_provenance" / "freeze_legacy_training_snapshot.py"
STAGE = "quality_value_v2_screen"


def _load_module():
    spec = importlib.util.spec_from_file_location("freeze_legacy_training_snapshot", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_legacy_runner():
    path = ROOT / "scripts" / "run_training_grid.py"
    spec = importlib.util.spec_from_file_location("snapshot_test_legacy_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    scripts_path = str(ROOT / "scripts")
    sys.path.insert(0, scripts_path)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(scripts_path)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _grid_document() -> dict[str, object]:
    return {
        "schema": "embedbench.training-grid",
        "schema_version": 1,
        "grid_id": "isingfold-quality-value-v2-screen",
        "selection": {"seeds": [0, 1, 2, 3]},
        "stages": {
            STAGE: {
                "trainer": "scripts/train_quality_v2.py",
                "inputs": "runs/release_v1_1/quality_*.jsonl",
                "splits": "data/release_v1/splits_quality_problem_v2.json",
                "output_dir": "runs/training_grid_quality_v2/screen",
                "checkpoint_dir": "runs/training_grid_quality_v2/checkpoints/screen",
                "epochs": 2,
                "axis_order": ["objective_variant", "arch", "seed"],
                "axes": {
                    "arch": ["mpnn", "gin", "gatv2", "gps", "hetero"],
                    "objective_variant": [
                        "p_only",
                        "p_connectivity",
                        "p_robustness",
                        "full",
                    ],
                    "seed": [0, 1, 2, 3],
                },
                "extra_args": ["--hidden", "8", "--deploy-view"],
                "execution": {
                    "assignment": "cell_index_modulo",
                    "modulus": 2,
                    "remainder_to_site": {"0": "apollo", "1": "goose"},
                    "required_device_type": "cuda",
                    "slurm_required_sites": ["goose"],
                    "slurm_forbidden_sites": ["apollo"],
                },
            }
        },
    }


def _cell_id(objective: str, architecture: str, seed: int) -> str:
    return f"{STAGE}__objective_variant-{objective}__arch-{architecture}__seed-{seed}"


def _prepare_snapshot_fixture(tmp_path: Path, module):
    roots = [tmp_path / "apollo", tmp_path / "goose"]
    grid_path = tmp_path / "registered-grid.json"
    grid = _grid_document()
    _write_json(grid_path, grid)
    grid_payload = grid_path.read_bytes()
    grid_sha256 = _sha256(grid_path)

    corpus_payload = b'{"partition":"test","opaque_quality_label":"DO NOT PARSE"}\n'
    split_document = {
        "schema": "embedbench.split-manifest",
        "schema_version": 2,
        "provenance": {
            "inputs": [
                {
                    "file": "quality_fixture.jsonl",
                    "sha256": hashlib.sha256(corpus_payload).hexdigest(),
                }
            ]
        },
        "splits": {"quality_fixture.jsonl": {"opaque-problem": "test"}},
    }

    for staged_root in roots:
        files = {
            "src/embedbench/__init__.py": b"",
            "src/embedbench/model.py": b"MODEL = 'fixture'\n",
            "src/embedbench/models_quality_v2.py": (
                b"import torch\n"
                b"def build_quality_v2_model(**kwargs):\n"
                b"    return torch.nn.Linear(1, 1)\n"
            ),
            "scripts/train_quality_v2.py": b"# fixture trainer\n",
            "scripts/training_splits.py": b"# fixture split logic\n",
            "scripts/run_training_grid.py": b"# fixture launcher\n",
            "scripts/goose_training_grid.sbatch": b"#!/bin/bash\n# fixture Slurm wrapper\n",
            "pyproject.toml": b"[project]\nname='fixture'\n",
            "requirements.lock": b"torch==fixture --hash=sha256:fixture\n",
        }
        for relative, payload in files.items():
            path = staged_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        staged_grid = staged_root / "configs" / "training_grid_quality_v2.json"
        staged_grid.parent.mkdir(parents=True, exist_ok=True)
        staged_grid.write_bytes(grid_payload)
        corpus = staged_root / "runs" / "release_v1_1" / "quality_fixture.jsonl"
        corpus.parent.mkdir(parents=True, exist_ok=True)
        corpus.write_bytes(corpus_payload)
        _write_json(
            staged_root / "data" / "release_v1" / "splits_quality_problem_v2.json",
            split_document,
        )

    source_sha256, _ = module.legacy_source_inventory(roots[0])
    module.REGISTERED_LEGACY_SOURCE_SHA256 = source_sha256
    module.REGISTERED_GRID_SHA256 = grid_sha256
    module.REGISTERED_GOOSE_LAUNCH_WRAPPER_SHA256 = _sha256(
        roots[1] / "scripts" / "goose_training_grid.sbatch"
    )

    corpus_sha256 = hashlib.sha256(corpus_payload).hexdigest()
    split_path = roots[0] / "data" / "release_v1" / "splits_quality_problem_v2.json"
    module.REGISTERED_QUALITY_INPUTS = {
        "quality_fixture.jsonl": {
            "bytes": len(corpus_payload),
            "sha256": corpus_sha256,
        }
    }
    module.REGISTERED_QUALITY_SPLIT_SHA256 = _sha256(split_path)
    data_provenance = {
        "corpus_inputs": [{"file": "quality_fixture.jsonl", "sha256": corpus_sha256}],
        "split_manifest": {
            "file": "splits_quality_problem_v2.json",
            "schema": "embedbench.split-manifest",
            "schema_version": 2,
            "sha256": _sha256(split_path),
        },
    }
    runtime_provenance = {
        "schema": "embedbench.runtime-environment",
        "schema_version": 1,
        "python": {
            "implementation": "CPython",
            "version": "3.12.3",
            "executable": "/opt/fixture/bin/python3",
        },
        "torch": {
            "version": "2.11.0+cu128",
            "cuda_available": True,
            "cuda_runtime_version": "12.8",
            "cudnn_version": 91900,
        },
        "device": {
            "selected": "cuda",
            "type": "cuda",
            "index": 0,
            "gpu": {
                "name": "fixture-gpu",
                "compute_capability": [9, 0],
                "total_memory_bytes": 1024,
            },
        },
        "packages": {
            "embedbench": "0.1.0",
            "numpy": "2.4.4",
            "scipy": "1.18.0",
            "networkx": "3.6.1",
            "dimod": "0.12.22",
            "dwave-samplers": "1.8.0",
            "dwave-networkx": "0.8.19",
            "minorminer": "0.2.22",
        },
    }
    _, environment_sha256, _ = module._runtime_contract(
        runtime_provenance,
        required_device="cuda",
        cell_id="fixture",
    )

    index = 0
    for objective in ("p_only", "p_connectivity", "p_robustness", "full"):
        for architecture in ("mpnn", "gin", "gatv2", "gps", "hetero"):
            for seed in (0, 1, 2, 3):
                staged_root = roots[index % 2]
                site = "apollo" if index % 2 == 0 else "goose"
                cell_id = _cell_id(objective, architecture, seed)
                result_path = (
                    staged_root / "runs" / "training_grid_quality_v2" / "screen" / f"{cell_id}.json"
                )
                checkpoint_prefix = (
                    staged_root
                    / "runs"
                    / "training_grid_quality_v2"
                    / "checkpoints"
                    / "screen"
                    / cell_id
                )
                checkpoint_path = Path(f"{checkpoint_prefix}_{architecture}_s{seed}.pt")
                preprocessing = {
                    "deploy_view": True,
                    "deploy_max_free": 8,
                    "neighbour_feats": False,
                    "encoder": "heterogeneous" if architecture == "hetero" else "chain",
                    "hamiltonian_context": {"schema": "fixture-context"},
                }
                deployment_host_contract = {
                    "mode": "pristine_topology_reconstruction",
                    "source_manifests": [],
                }
                training_scope = {"limit": None, "full_fixed_train_validation": True}
                loss_weights = {
                    "quality_probability": 1.0,
                    "within_state_rank": 0.5,
                    "residual_connectivity_proxy": (
                        0.1 if objective in {"p_connectivity", "full"} else 0.0
                    ),
                    "robustness": 0.1 if objective in {"p_robustness", "full"} else 0.0,
                    "terminal_qubits": 0.0,
                }
                primary_metric = 0.01 + index / 10000.0
                training_run_id = hashlib.sha256(cell_id.encode()).hexdigest()[:32]
                result_args = {
                    "files": [str(staged_root / "runs" / "release_v1_1" / "quality_fixture.jsonl")],
                    "arch": architecture,
                    "objective_variant": objective,
                    "seeds": str(seed),
                    "epochs": 2,
                    "hidden": 8,
                    "layers": 3,
                    "heads": 4,
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
                    "deploy_view": True,
                    "deploy_max_free": 8,
                    "device": "cuda",
                    "splits": str(
                        staged_root / "data" / "release_v1" / "splits_quality_problem_v2.json"
                    ),
                    "evaluation_support": "full",
                    "save": str(checkpoint_prefix),
                    "out": str(result_path),
                }
                result = {
                    "artifact_schema": "embedbench.quality-v2-training-results",
                    "artifact_schema_version": 1,
                    "args": result_args,
                    "device": "cuda",
                    "data_provenance": data_provenance,
                    "split_sha256": data_provenance["split_manifest"]["sha256"],
                    "runtime_provenance": runtime_provenance,
                    "preprocessing": preprocessing,
                    "deployment_host_contract": deployment_host_contract,
                    "training_scope": training_scope,
                    "test_locked": True,
                    "test_partition_encoded": False,
                    "results": [
                        {
                            "objective_variant": objective,
                            "arch": architecture,
                            "seed": seed,
                            "hidden": 8,
                            "layers": 3,
                            "heads": 4,
                            "params": 2,
                            "device": "cuda",
                            "n_train": 10,
                            "n_val": 4,
                            "n_test": 3,
                            "best_epoch": 1,
                            "training_run_id": training_run_id,
                            "training_scope": training_scope,
                            "preprocessing": preprocessing,
                            "deployment_host_contract": deployment_host_contract,
                            "loss_weights": loss_weights,
                            "evaluation_support": "full",
                            "registered_budget_ratios": [1.0, 1.1, 1.25, 1.5, None],
                            "primary_validation_metric": "mean_finite_budget_regret",
                            "lcb_z": 1.0,
                            "lcb_is_secondary_until_calibrated": True,
                            "ranking_thresholds": {
                                "stage2_pair": 0.05,
                                "stage1_involved_pair": 0.10,
                            },
                            "selection_contract": (
                                "quality_mean_or_lcb_subject_to_exact_feasibility_and_budget-v1"
                            ),
                            "auxiliary_heads_used_for_selection": False,
                            "validation": {"budget_sweep": {"primary_metric": primary_metric}},
                            "test_evaluated": False,
                            "test_partition_encoded": False,
                            "test": None,
                            "seconds": 1.0,
                            "best_epoch_training_losses": {
                                "loss": 1.0,
                                "quality_probability": 0.5,
                                "within_state_rank": 0.25,
                                "future_capacity": 0.0,
                                "robustness": 0.0,
                                "terminal_qubits": 0.0,
                            },
                        }
                    ],
                }
                _write_json(result_path, result)
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                checkpoint_metadata = {
                    "artifact_schema": "embedbench.quality-value-v2",
                    "artifact_schema_version": 2,
                    "forward_inputs": [
                        "x",
                        "adjacency",
                        "candidate_masks",
                        "candidate_features",
                        "hamiltonian_context",
                    ],
                    "robustness_names": [
                        "minimum_logical_contact_count",
                        "mean_logical_contact_count",
                        "total_logical_contact_count",
                        "candidate_chain_cycle_rank",
                    ],
                    "model_config": {
                        "arch": architecture,
                        "hidden": 8,
                        "layers": 3,
                        "heads": 4,
                        "neighbour_feats": False,
                        "predict_terminal_qubits": False,
                        "robustness_dim": 4,
                    },
                    "objective_variant": objective,
                    "seed": seed,
                    "training_run_id": training_run_id,
                    "best_epoch": 1,
                    "selected_validation_metric": primary_metric,
                    "split_sha256": data_provenance["split_manifest"]["sha256"],
                    "data_provenance": data_provenance,
                    "runtime_provenance": runtime_provenance,
                    "device": "cuda",
                    "splits": str(
                        staged_root / "data" / "release_v1" / "splits_quality_problem_v2.json"
                    ),
                    "evaluation_support": "full",
                    "registered_budget_ratios": [1.0, 1.1, 1.25, 1.5, None],
                    "primary_validation_metric": "mean_finite_budget_regret",
                    "lcb_z": 1.0,
                    "lcb_is_secondary_until_calibrated": True,
                    "ranking_thresholds": {
                        "stage2_pair": 0.05,
                        "stage1_involved_pair": 0.10,
                    },
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
                    "neighbour_feats": False,
                }
                torch.save(
                    {"state": torch.nn.Linear(1, 1).state_dict(), "meta": checkpoint_metadata},
                    checkpoint_path,
                )
                cell = {
                    "stage": STAGE,
                    "index": index,
                    "cell_id": cell_id,
                    "params": {
                        "objective_variant": objective,
                        "arch": architecture,
                        "seed": seed,
                    },
                    "extra_args": ["--hidden", "8", "--deploy-view"],
                }
                corpus_path = staged_root / "runs" / "release_v1_1" / "quality_fixture.jsonl"
                staged_split = (
                    staged_root / "data" / "release_v1" / "splits_quality_problem_v2.json"
                )
                trainer = staged_root / "scripts" / "train_quality_v2.py"
                command = [
                    "/opt/fixture/bin/python3",
                    str(trainer),
                    str(corpus_path),
                    "--splits",
                    str(staged_split),
                    "--objective-variant",
                    objective,
                    "--arch",
                    architecture,
                    "--seeds",
                    str(seed),
                    "--epochs",
                    "2",
                    "--device",
                    "cuda",
                    "--save",
                    str(checkpoint_prefix),
                    "--hidden",
                    "8",
                    "--deploy-view",
                    "--out",
                    str(result_path),
                ]
                slurm = {
                    "array_job_id": "7" if site == "goose" else None,
                    "array_task_id": str((index // 2) % 2) if site == "goose" else None,
                    "job_id": str(7000 + index) if site == "goose" else None,
                }
                lock_sha256 = _sha256(staged_root / "requirements.lock")
                wrapper_sha256 = _sha256(staged_root / "scripts" / "goose_training_grid.sbatch")
                receipt = {
                    "schema": "embedbench.training-receipt",
                    "schema_version": 1,
                    "grid_id": grid["grid_id"],
                    "grid_sha256": grid_sha256,
                    "source_sha256": source_sha256,
                    "data_provenance": data_provenance,
                    "cell": cell,
                    "command": command,
                    "hostname": f"{site}.example",
                    "execution_site": site,
                    "slurm": slurm,
                    "started_unix": 1000.0 + index,
                    "finished_unix": 1001.0 + index,
                    "result": str(result_path),
                    "result_sha256": _sha256(result_path),
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": _sha256(checkpoint_path),
                    "test_locked": True,
                    "dependency_lock_sha256": lock_sha256,
                    "environment_sha256": environment_sha256,
                    "launch_wrapper_sha256": wrapper_sha256,
                }
                _write_json(result_path.with_suffix(".receipt.json"), receipt)
                index += 1

    return roots, grid_path, source_sha256, grid_sha256


def _receipt(roots: list[Path], cell_index: int) -> Path:
    staged_root = roots[cell_index % 2]
    receipts = sorted(
        (staged_root / "runs" / "training_grid_quality_v2" / "screen").glob("*.receipt.json")
    )
    for path in receipts:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document["cell"]["index"] == cell_index:
            return path
    raise AssertionError(f"fixture receipt {cell_index} not found")


def test_registered_grid_bytes_are_frozen() -> None:
    module = _load_module()

    assert _sha256(ROOT / "configs" / "training_grid_quality_v2.json") == (
        module.REGISTERED_GRID_SHA256
    )


def test_source_digest_exactly_matches_the_legacy_launcher_semantics(
    tmp_path: Path,
) -> None:
    module = _load_module()
    runner = _load_legacy_runner()
    source = tmp_path / "src" / "embedbench"
    scripts = tmp_path / "scripts"
    source.mkdir(parents=True)
    scripts.mkdir()
    (source / "z.py").write_bytes(b"Z = 1\n")
    (source / "a.py").write_bytes(b"A = 2\n")
    (scripts / "train_quality_v2.py").write_bytes(b"# trainer\n")
    (scripts / "training_splits.py").write_bytes(b"# split\n")
    (scripts / "run_training_grid.py").write_bytes(b"# launcher\n")

    reconstructed, inventory = module.legacy_source_inventory(tmp_path)

    assert reconstructed == runner._source_sha256(tmp_path)
    assert [entry["relative_path"] for entry in inventory] == sorted(
        entry["relative_path"] for entry in inventory
    )
    assert module.REGISTERED_LEGACY_SOURCE_SHA256 == (
        "294caa56d0bb04f716163b143a995ee176c8e72dd209c5f5f77caef92d72a264"
    )


def test_build_snapshot_is_canonical_complete_and_root_order_independent(
    tmp_path: Path,
) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, grid_sha256 = _prepare_snapshot_fixture(tmp_path, module)

    forward = module.build_snapshot(
        grid_path=grid_path,
        staged_roots=roots,
        expected_legacy_source_sha256=source_sha256,
        allow_unlocked_legacy=True,
    )
    reverse = module.build_snapshot(
        grid_path=grid_path,
        staged_roots=list(reversed(roots)),
        expected_legacy_source_sha256=source_sha256,
        allow_unlocked_legacy=True,
    )

    assert forward == reverse
    assert forward["schema"] == "embedbench.legacy-training-snapshot"
    assert forward["schema_version"] == 1
    assert forward["legacy_source_sha256"] == source_sha256
    assert forward["grid_sha256"] == grid_sha256
    assert forward["registered_cell_count"] == 80
    assert forward["registered_sites"] == ["apollo", "goose"]
    assert forward["logical_site_assignment_verified"] is True
    assert forward["physical_execution_site_attested"] is False
    assert forward["historical_staged_roots_by_site"] == {
        "apollo": str(roots[0]),
        "goose": str(roots[1]),
    }
    assert forward["execution_site_cell_counts"] == {"apollo": 40, "goose": 40}
    assert len(forward["cells"]) == 80
    assert [cell["index"] for cell in forward["cells"]] == list(range(80))
    assert {cell["seed"] for cell in forward["cells"]} == {0, 1, 2, 3}
    cell_keys = {
        (cell["objective_variant"], cell["architecture"], cell["seed"]) for cell in forward["cells"]
    }
    assert len(cell_keys) == 80
    for role in (
        "legacy_training_result",
        "legacy_training_receipt",
        "legacy_training_checkpoint",
    ):
        assert sum(entry["role"] == role for entry in forward["file_inventory"]) == 80
    assert forward["test_access"]["corpus_records_parsed_by_snapshot"] is False
    assert forward["test_access"]["legacy_split_loader_parsed_test_records"] is True
    assert forward["test_access"]["test_partition_encoded_by_training"] is False
    assert forward["test_access"]["test_metrics_evaluated_by_training"] is False
    assert forward["dependency_provenance"]["staged_lock_status"] == "present"
    assert forward["dependency_provenance"]["historical_receipts_authenticated"] is False
    assert forward["paper_reproducibility_eligible"] is False
    assert forward["final_paper_evaluation_eligible"] is False
    assert forward["launch_provenance"]["goose"]["wrapper_sha256"] == (
        module.REGISTERED_GOOSE_LAUNCH_WRAPPER_SHA256
    )
    assert forward["launch_provenance"]["goose"]["slurm_submission_command_attested"] is False
    assert forward["launch_provenance"]["apollo"]["mode"] == "direct_python"
    assert all(cell["checkpoint_semantics"]["parameter_count"] == 2 for cell in forward["cells"])
    assert forward["snapshot_manifest_sha256"] not in {
        source_sha256,
        grid_sha256,
    }
    assert forward["snapshot_manifest_sha256"] == module.inventory_sha256(forward["file_inventory"])


def test_snapshot_hashes_opaque_corpus_and_checkpoint_bytes_without_parsing_them(
    tmp_path: Path,
) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    corpus = roots[0] / "runs" / "release_v1_1" / "quality_fixture.jsonl"
    corpus.write_bytes(b"not JSON and deliberately opaque\n")
    new_hash = _sha256(corpus)
    peer = roots[1] / "runs" / "release_v1_1" / "quality_fixture.jsonl"
    peer.write_bytes(corpus.read_bytes())
    for root in roots:
        split_path = root / "data" / "release_v1" / "splits_quality_problem_v2.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        split["provenance"]["inputs"][0]["sha256"] = new_hash
        _write_json(split_path, split)
        split_sha256 = _sha256(split_path)
        for receipt_path in (root / "runs" / "training_grid_quality_v2" / "screen").glob(
            "*.receipt.json"
        ):
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["data_provenance"]["corpus_inputs"][0]["sha256"] = new_hash
            receipt["data_provenance"]["split_manifest"]["sha256"] = split_sha256
            result_path = receipt_path.with_name(
                receipt_path.name.removesuffix(".receipt.json") + ".json"
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["data_provenance"]["corpus_inputs"][0]["sha256"] = new_hash
            result["data_provenance"]["split_manifest"]["sha256"] = split_sha256
            result["split_sha256"] = split_sha256
            _write_json(result_path, result)
            checkpoint_path = Path(receipt["checkpoint"])
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            checkpoint["meta"]["data_provenance"] = result["data_provenance"]
            checkpoint["meta"]["split_sha256"] = split_sha256
            torch.save(checkpoint, checkpoint_path)
            receipt["result_sha256"] = _sha256(result_path)
            receipt["checkpoint_sha256"] = _sha256(checkpoint_path)
            _write_json(receipt_path, receipt)

    module.REGISTERED_QUALITY_INPUTS = {
        "quality_fixture.jsonl": {"bytes": len(corpus.read_bytes()), "sha256": new_hash}
    }
    module.REGISTERED_QUALITY_SPLIT_SHA256 = _sha256(
        roots[0] / "data" / "release_v1" / "splits_quality_problem_v2.json"
    )

    snapshot = module.build_snapshot(
        grid_path=grid_path,
        staged_roots=roots,
        expected_legacy_source_sha256=source_sha256,
        allow_unlocked_legacy=True,
    )

    assert snapshot["data_provenance"]["corpus_inputs"][0]["sha256"] == new_hash


def test_dirty_corpus_cannot_be_recertified_by_rewriting_local_receipts(
    tmp_path: Path,
) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    changed_payload = b"locally reconstructed substitute\n"
    changed_sha256 = hashlib.sha256(changed_payload).hexdigest()
    for root in roots:
        corpus = root / "runs" / "release_v1_1" / "quality_fixture.jsonl"
        corpus.write_bytes(changed_payload)
        split_path = root / "data" / "release_v1" / "splits_quality_problem_v2.json"
        split = json.loads(split_path.read_text(encoding="utf-8"))
        split["provenance"]["inputs"][0]["sha256"] = changed_sha256
        _write_json(split_path, split)
        split_sha256 = _sha256(split_path)
        for receipt_path in (root / "runs" / "training_grid_quality_v2" / "screen").glob(
            "*.receipt.json"
        ):
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["data_provenance"]["corpus_inputs"][0]["sha256"] = changed_sha256
            receipt["data_provenance"]["split_manifest"]["sha256"] = split_sha256
            result_path = receipt_path.with_name(
                receipt_path.name.removesuffix(".receipt.json") + ".json"
            )
            result = json.loads(result_path.read_text(encoding="utf-8"))
            result["data_provenance"] = receipt["data_provenance"]
            result["split_sha256"] = split_sha256
            _write_json(result_path, result)
            receipt["result_sha256"] = _sha256(result_path)
            _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="registered immutable corpus"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_relocated_release_directory_symlink_is_accepted_when_bytes_match(
    tmp_path: Path,
) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    logical_release = roots[0] / "runs" / "release_v1_1"
    relocated_release = tmp_path / "relocated" / "release_v1_1"
    relocated_release.parent.mkdir()
    logical_release.rename(relocated_release)
    logical_release.symlink_to(relocated_release, target_is_directory=True)

    snapshot = module.build_snapshot(
        grid_path=grid_path,
        staged_roots=roots,
        expected_legacy_source_sha256=source_sha256,
        allow_unlocked_legacy=True,
    )

    assert snapshot["registered_cell_count"] == 80
    corpus_inventory = [
        entry for entry in snapshot["file_inventory"] if entry["role"] == "legacy_corpus_input"
    ]
    assert [entry["relative_path"] for entry in corpus_inventory] == [
        "runs/release_v1_1/quality_fixture.jsonl"
    ]


def test_registered_individual_corpus_symlinks_are_accepted_when_bytes_match(
    tmp_path: Path,
) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    for index, staged_root in enumerate(roots):
        corpus = staged_root / "runs" / "release_v1_1" / "quality_fixture.jsonl"
        relocated = tmp_path / f"relocated-corpus-{index}.jsonl"
        corpus.rename(relocated)
        corpus.symlink_to(relocated)

    snapshot = module.build_snapshot(
        grid_path=grid_path,
        staged_roots=roots,
        expected_legacy_source_sha256=source_sha256,
        allow_unlocked_legacy=True,
    )

    assert snapshot["registered_cell_count"] == 80


def test_missing_registered_cell_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 17)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    Path(receipt["result"]).unlink()
    Path(receipt["checkpoint"]).unlink()
    receipt_path.unlink()

    with pytest.raises(ValueError, match="missing registered cells"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_duplicate_registered_cell_across_roots_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 0)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    destination_root = roots[1]
    for field in ("result", "checkpoint"):
        source = Path(receipt[field])
        relative = source.relative_to(roots[0])
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    relative_receipt = receipt_path.relative_to(roots[0])
    duplicate_receipt = destination_root / relative_receipt
    duplicate_receipt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(receipt_path, duplicate_receipt)

    with pytest.raises(ValueError, match="duplicate registered cell"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


@pytest.mark.parametrize("artifact", ["result", "checkpoint", "receipt"])
def test_tampered_artifact_is_rejected(tmp_path: Path, artifact: str) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 8)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    path = receipt_path if artifact == "receipt" else Path(receipt[artifact])
    path.write_bytes(path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="invalid|SHA-256|tampered"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_dirty_source_tree_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    (roots[1] / "src" / "embedbench" / "dirty.py").write_text("DIRTY = True\n", encoding="utf-8")

    with pytest.raises(ValueError, match="legacy source SHA-256"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


@pytest.mark.parametrize("binding", ["grid_sha256", "source_sha256"])
def test_stale_receipt_grid_or_source_binding_is_rejected(tmp_path: Path, binding: str) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 5)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt[binding] = "0" * 64
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match=binding):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_receipt_launch_command_must_match_the_registered_cell(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 5)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    epochs_index = receipt["command"].index("--epochs") + 1
    receipt["command"][epochs_index] = "999"
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="launch command changed"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_data_or_split_provenance_mismatch_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 11)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["data_provenance"]["split_manifest"]["sha256"] = "a" * 64
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="data_provenance"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_stale_top_level_result_split_binding_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 11)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result_path = Path(receipt["result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["split_sha256"] = "a" * 64
    _write_json(result_path, result)
    receipt["result_sha256"] = _sha256(result_path)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="split_sha256"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_symlinked_cell_artifact_is_rejected_as_a_substitute(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 8)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checkpoint = Path(receipt["checkpoint"])
    substitute = tmp_path / "substitute.pt"
    checkpoint.rename(substitute)
    checkpoint.symlink_to(substitute)

    with pytest.raises(ValueError, match="symlink substitute"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_changed_registered_grid_bytes_are_rejected(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    grid_path.write_bytes(grid_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="registered grid SHA-256"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_unregistered_orphan_receipt_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    stale = roots[0] / "runs" / "training_grid_quality_v2" / "screen" / "stale.receipt.json"
    _write_json(stale, {"schema": "embedbench.training-receipt"})

    with pytest.raises(ValueError, match="unregistered receipt"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_cli_refuses_stale_existing_manifest(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    out = tmp_path / "LEGACY_TRAINING_SNAPSHOT.json"
    out.write_text('{"stale":true}\n', encoding="utf-8")
    monkeypatch.setattr(module, "REGISTERED_LEGACY_SOURCE_SHA256", source_sha256)
    monkeypatch.setattr(module, "REGISTERED_GRID_SHA256", _sha256(grid_path))

    with pytest.raises(ValueError, match="existing snapshot is stale"):
        module.main(
            [
                "--expected-legacy-source-sha256",
                source_sha256,
                "--grid",
                str(grid_path),
                "--staged-root",
                f"apollo={roots[0]}",
                "--staged-root",
                f"goose={roots[1]}",
                "--allow-unlocked-legacy",
                "--out",
                str(out),
            ]
        )


def test_cli_writes_once_and_idempotently_verifies_identical_snapshot(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, grid_sha256 = _prepare_snapshot_fixture(tmp_path, module)
    out = tmp_path / "release" / "LEGACY_TRAINING_SNAPSHOT.json"
    monkeypatch.setattr(module, "REGISTERED_LEGACY_SOURCE_SHA256", source_sha256)
    monkeypatch.setattr(module, "REGISTERED_GRID_SHA256", grid_sha256)
    argv = [
        "--expected-legacy-source-sha256",
        source_sha256,
        "--grid",
        str(grid_path),
        "--staged-root",
        f"apollo={roots[0]}",
        "--staged-root",
        f"goose={roots[1]}",
        "--allow-unlocked-legacy",
        "--out",
        str(out),
    ]

    assert module.main(argv) == 0
    first = out.read_bytes()
    assert module.main(argv) == 0

    assert out.read_bytes() == first
    document = json.loads(first)
    assert document["registered_cell_count"] == 80


def _remove_receipt_bound_environment(roots: list[Path]) -> None:
    for staged_root in roots:
        lock = staged_root / "requirements.lock"
        if lock.exists():
            lock.unlink()
        for receipt_path in (staged_root / "runs" / "training_grid_quality_v2" / "screen").glob(
            "*.receipt.json"
        ):
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt.pop("dependency_lock_sha256", None)
            receipt.pop("environment_sha256", None)
            receipt.pop("launch_wrapper_sha256", None)
            _write_json(receipt_path, receipt)


def test_default_refuses_historical_receipts_without_explicit_legacy_authorization(
    tmp_path: Path,
) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    _remove_receipt_bound_environment(roots)

    with pytest.raises(ValueError, match="requires --allow-unlocked-legacy"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_explicit_unlocked_legacy_mode_downgrades_evidence_and_is_truthful(
    tmp_path: Path,
) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    _remove_receipt_bound_environment(roots)

    snapshot = module.build_snapshot(
        grid_path=grid_path,
        staged_roots=roots,
        expected_legacy_source_sha256=source_sha256,
        allow_unlocked_legacy=True,
    )

    assert snapshot["evidence_scope"] == "legacy_architecture_selection_only"
    assert snapshot["paper_reproducibility_eligible"] is False
    assert snapshot["final_paper_evaluation_eligible"] is False
    assert snapshot["legacy_environment_override_used"] is True
    assert snapshot["test_access"] == {
        "corpus_bytes_hashed_by_snapshot": True,
        "corpus_records_parsed_by_snapshot": False,
        "legacy_split_loader_parsed_test_records": True,
        "test_partition_encoded_by_training": False,
        "test_metrics_evaluated_by_training": False,
    }


def test_cli_unlocked_legacy_flag_writes_only_downgraded_evidence(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, grid_sha256 = _prepare_snapshot_fixture(tmp_path, module)
    _remove_receipt_bound_environment(roots)
    monkeypatch_values = {
        "REGISTERED_LEGACY_SOURCE_SHA256": source_sha256,
        "REGISTERED_GRID_SHA256": grid_sha256,
    }
    for name, value in monkeypatch_values.items():
        setattr(module, name, value)
    out = tmp_path / "legacy-snapshot.json"

    assert (
        module.main(
            [
                "--expected-legacy-source-sha256",
                source_sha256,
                "--grid",
                str(grid_path),
                "--staged-root",
                f"apollo={roots[0]}",
                "--staged-root",
                f"goose={roots[1]}",
                "--allow-unlocked-legacy",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    snapshot = json.loads(out.read_text(encoding="utf-8"))
    assert snapshot["paper_reproducibility_eligible"] is False
    assert snapshot["final_paper_evaluation_eligible"] is False


@pytest.mark.parametrize(
    ("command_index", "replacement"),
    [
        (0, "/attacker/python-backdoor"),
        (1, "/attacker/stage/scripts/train_quality_v2.py"),
        (1, "/tmp/../attacker/scripts/train_quality_v2.py"),
    ],
)
def test_receipt_command_rejects_fake_interpreter_suffix_spoof_and_traversal(
    tmp_path: Path, command_index: int, replacement: str
) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 5)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["command"][command_index] = replacement
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="interpreter|historical root|canonical absolute"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_cell_cannot_be_owned_by_a_root_qualified_for_another_site(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 0)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for source in (Path(receipt["result"]), Path(receipt["checkpoint"]), receipt_path):
        destination = roots[1] / source.relative_to(roots[0])
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)

    with pytest.raises(ValueError, match="owned by staged site"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


@pytest.mark.parametrize("task_id", ["999999", "not-a-task", "-1"])
def test_goose_slurm_task_must_be_numeric_and_match_the_registered_cell(
    tmp_path: Path, task_id: str
) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 5)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["slurm"]["array_task_id"] = task_id
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="Slurm array task"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_result_args_must_match_the_receipt_bound_training_command(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 8)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result_path = Path(receipt["result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["args"]["epochs"] = 999
    _write_json(result_path, result)
    receipt["result_sha256"] = _sha256(result_path)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="result args"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_nonfinite_result_metric_is_rejected_even_with_rewritten_receipt_hash(
    tmp_path: Path,
) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 8)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result_path = Path(receipt["result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["results"][0]["validation"] = {"budget_sweep": {"primary_metric": float("nan")}}
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["result_sha256"] = _sha256(result_path)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="non-finite JSON constant"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_zero_byte_checkpoint_is_rejected_even_with_rewritten_receipt_hash(
    tmp_path: Path,
) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 8)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(receipt["checkpoint"])
    checkpoint_path.write_bytes(b"")
    receipt["checkpoint_sha256"] = _sha256(checkpoint_path)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="empty checkpoint"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 8)
    payload = receipt_path.read_text(encoding="utf-8")
    needle = '  "source_sha256": '
    line = next(line for line in payload.splitlines() if line.startswith(needle))
    payload = payload.replace(line, f"{line}\n{line}", 1)
    receipt_path.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_split_manifest_symlink_substitute_is_rejected(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    for index, staged_root in enumerate(roots):
        split = staged_root / "data" / "release_v1" / "splits_quality_problem_v2.json"
        substitute = tmp_path / f"split-substitute-{index}.json"
        split.rename(substitute)
        split.symlink_to(substitute)

    with pytest.raises(ValueError, match="split manifest.*symlink"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_unqualified_cli_root_strings_are_rejected(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)

    with pytest.raises(ValueError, match="site-qualified"):
        module.main(
            [
                "--expected-legacy-source-sha256",
                source_sha256,
                "--grid",
                str(grid_path),
                "--staged-root",
                str(roots[0]),
                "--staged-root",
                str(roots[1]),
                "--out",
                str(tmp_path / "snapshot.json"),
            ]
        )


@pytest.mark.parametrize(
    "field",
    ["dependency_lock_sha256", "environment_sha256", "launch_wrapper_sha256"],
)
def test_missing_receipt_environment_binding_requires_explicit_legacy_mode(
    tmp_path: Path, field: str
) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 5)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop(field)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="requires --allow-unlocked-legacy"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_explicit_legacy_mode_never_accepts_a_wrong_recorded_environment_hash(
    tmp_path: Path,
) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 5)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["environment_sha256"] = "0" * 64
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="wrong runtime environment"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
            allow_unlocked_legacy=True,
        )


def test_runtime_environment_must_be_compatible_across_all_cells(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 5)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result_path = Path(receipt["result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["runtime_provenance"]["packages"]["numpy"] = "99.0"
    _, environment_sha256, _ = module._runtime_contract(
        result["runtime_provenance"], required_device="cuda", cell_id="test"
    )
    _write_json(result_path, result)
    receipt["result_sha256"] = _sha256(result_path)
    receipt["environment_sha256"] = environment_sha256
    checkpoint_path = Path(receipt["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["meta"]["runtime_provenance"] = result["runtime_provenance"]
    torch.save(checkpoint, checkpoint_path)
    receipt["checkpoint_sha256"] = _sha256(checkpoint_path)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="incompatible runtime environments"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


@pytest.mark.parametrize("corruption", ["metadata", "nonfinite_state"])
def test_checkpoint_semantics_are_validated_in_an_isolated_subprocess(
    tmp_path: Path, corruption: str
) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 8)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(receipt["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if corruption == "metadata":
        checkpoint["meta"]["seed"] = 999
    else:
        checkpoint["state"]["weight"].fill_(float("nan"))
    torch.save(checkpoint, checkpoint_path)
    receipt["checkpoint_sha256"] = _sha256(checkpoint_path)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="isolated checkpoint validation failed"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )


def test_registered_legacy_receipts_cannot_be_posthoc_upgraded_to_paper_evidence(
    tmp_path: Path,
) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)

    with pytest.raises(ValueError, match="requires --allow-unlocked-legacy"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
        )

    snapshot = module.build_snapshot(
        grid_path=grid_path,
        staged_roots=roots,
        expected_legacy_source_sha256=source_sha256,
        allow_unlocked_legacy=True,
    )
    assert snapshot["paper_reproducibility_eligible"] is False
    assert snapshot["final_paper_evaluation_eligible"] is False
    assert snapshot["historical_receipts_authenticated"] is False


def test_every_cell_at_one_site_must_share_one_exact_historical_root(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 0)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result_path = Path(receipt["result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    old_root = str(roots[0])
    alternate_root = "/alternate/stage"
    receipt["command"] = [
        alternate_root + token[len(old_root) :] if token.startswith(old_root) else token
        for token in receipt["command"]
    ]
    receipt["result"] = alternate_root + receipt["result"][len(old_root) :]
    receipt["checkpoint"] = alternate_root + receipt["checkpoint"][len(old_root) :]
    result["args"]["files"] = [
        alternate_root + token[len(old_root) :] if token.startswith(old_root) else token
        for token in result["args"]["files"]
    ]
    for field in ("splits", "save", "out"):
        result["args"][field] = alternate_root + result["args"][field][len(old_root) :]
    _write_json(result_path, result)
    receipt["result_sha256"] = _sha256(result_path)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="historical staged root"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
            allow_unlocked_legacy=True,
        )


@pytest.mark.parametrize("artifact", ["receipt", "result", "checkpoint"])
def test_seed_zero_never_accepts_boolean_false_as_exact_metadata(
    tmp_path: Path, artifact: str
) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 0)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if artifact == "receipt":
        receipt["cell"]["params"]["seed"] = False
    elif artifact == "result":
        result_path = Path(receipt["result"])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result["results"][0]["seed"] = False
        _write_json(result_path, result)
        receipt["result_sha256"] = _sha256(result_path)
    else:
        checkpoint_path = Path(receipt["checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        checkpoint["meta"]["seed"] = False
        torch.save(checkpoint, checkpoint_path)
        receipt["checkpoint_sha256"] = _sha256(checkpoint_path)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="cell|axes|metadata"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
            allow_unlocked_legacy=True,
        )


def test_json_number_overflow_is_rejected_before_result_validation(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 0)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result_path = Path(receipt["result"])
    payload = result_path.read_text(encoding="utf-8").replace('"seconds": 1.0', '"seconds": 1e999')
    assert "1e999" in payload
    result_path.write_text(payload, encoding="utf-8")
    receipt["result_sha256"] = _sha256(result_path)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="non-finite JSON number"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
            allow_unlocked_legacy=True,
        )


def test_result_rows_are_closed_against_unregistered_fields(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 0)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result_path = Path(receipt["result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["results"][0]["unregistered_metric"] = 1.0
    _write_json(result_path, result)
    receipt["result_sha256"] = _sha256(result_path)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="seed-result fields changed"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
            allow_unlocked_legacy=True,
        )


def test_objective_variant_requires_its_exact_registered_loss_weights(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 0)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result_path = Path(receipt["result"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["results"][0]["loss_weights"]["residual_connectivity_proxy"] = 0.1
    _write_json(result_path, result)
    checkpoint_path = Path(receipt["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["meta"]["loss_weights"] = result["results"][0]["loss_weights"]
    torch.save(checkpoint, checkpoint_path)
    receipt["result_sha256"] = _sha256(result_path)
    receipt["checkpoint_sha256"] = _sha256(checkpoint_path)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="loss weights"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
            allow_unlocked_legacy=True,
        )


def test_checkpoint_metadata_schema_is_closed(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 0)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(receipt["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["meta"]["unregistered_selection_hint"] = 0
    torch.save(checkpoint, checkpoint_path)
    receipt["checkpoint_sha256"] = _sha256(checkpoint_path)
    _write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="checkpoint metadata fields changed"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
            allow_unlocked_legacy=True,
        )


def test_checkpoint_semantics_use_the_same_bytes_as_the_inventory(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    receipt_path = _receipt(roots, 0)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(receipt["checkpoint"])
    valid_payload = checkpoint_path.read_bytes()
    invalid_payload = b"opaque-invalid-checkpoint-but-nonempty"
    checkpoint_path.write_bytes(invalid_payload)
    receipt["checkpoint_sha256"] = hashlib.sha256(invalid_payload).hexdigest()
    _write_json(receipt_path, receipt)
    real_run = module.subprocess.run

    def swap_only_while_the_child_runs(*args, **kwargs):
        checkpoint_path.write_bytes(valid_payload)
        try:
            return real_run(*args, **kwargs)
        finally:
            checkpoint_path.write_bytes(invalid_payload)

    monkeypatch.setattr(module.subprocess, "run", swap_only_while_the_child_runs)
    with pytest.raises(ValueError, match="isolated checkpoint validation failed"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
            allow_unlocked_legacy=True,
        )


def test_apollo_does_not_need_an_unused_copy_of_the_goose_wrapper(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    (roots[0] / "scripts" / "goose_training_grid.sbatch").unlink()

    snapshot = module.build_snapshot(
        grid_path=grid_path,
        staged_roots=roots,
        expected_legacy_source_sha256=source_sha256,
        allow_unlocked_legacy=True,
    )

    assert snapshot["launch_provenance"]["goose"]["wrapper_sha256"] == (
        module.REGISTERED_GOOSE_LAUNCH_WRAPPER_SHA256
    )


def test_goose_wrapper_must_match_the_registered_approved_sha256(tmp_path: Path) -> None:
    module = _load_module()
    roots, grid_path, source_sha256, _ = _prepare_snapshot_fixture(tmp_path, module)
    wrapper = roots[1] / "scripts" / "goose_training_grid.sbatch"
    wrapper.write_bytes(wrapper.read_bytes() + b"# post-hoc mutation\n")

    with pytest.raises(ValueError, match="registered Goose wrapper SHA-256"):
        module.build_snapshot(
            grid_path=grid_path,
            staged_roots=roots,
            expected_legacy_source_sha256=source_sha256,
            allow_unlocked_legacy=True,
        )
