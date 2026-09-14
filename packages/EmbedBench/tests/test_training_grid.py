from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]


def _load_module():
    path = ROOT / "scripts" / "run_training_grid.py"
    spec = importlib.util.spec_from_file_location("run_training_grid", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_registered_screen_has_expected_cells_and_keeps_test_locked() -> None:
    runner = _load_module()
    grid_path = ROOT / "configs" / "training_grid_v1.json"
    document = json.loads(grid_path.read_text(encoding="utf-8"))

    structural = runner.expand_stage(document, "structural_family_screen")
    quality = runner.expand_stage(document, "quality_chain_family_screen")

    assert len(structural) == 12
    assert len(quality) == 30
    assert len({cell.cell_id for cell in structural + quality}) == 42
    assert all("--evaluate-test" not in cell.extra_args for cell in structural + quality)
    assert all("--deploy-view" in cell.extra_args for cell in quality)


def test_quality_value_v2_grid_crosses_backbones_objectives_and_seeds() -> None:
    runner = _load_module()
    grid_path = ROOT / "configs" / "training_grid_quality_v2.json"
    document = json.loads(grid_path.read_text(encoding="utf-8"))

    cells = runner.expand_stage(document, "quality_value_v2_screen")
    stage = document["stages"]["quality_value_v2_screen"]

    assert len(cells) == 80
    assert {cell.params["arch"] for cell in cells} == {
        "mpnn",
        "gin",
        "gatv2",
        "gps",
        "hetero",
    }
    assert {cell.params["objective_variant"] for cell in cells} == {
        "p_only",
        "p_connectivity",
        "p_robustness",
        "full",
    }
    assert {cell.params["seed"] for cell in cells} == {0, 1, 2, 3}
    assert all("--evaluation-support" in cell.extra_args for cell in cells)
    assert all("full" in cell.extra_args for cell in cells)
    assert all("--deploy-view" in cell.extra_args for cell in cells)
    assert all("--evaluate-test" not in cell.extra_args for cell in cells)
    assert stage["extra_args"] == [
        "--hidden",
        "64",
        "--layers",
        "3",
        "--heads",
        "4",
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
        "--deploy-view",
    ]
    assert stage["inputs"] == "runs/release_v1_1/quality_*.jsonl"
    assert stage["splits"] == "data/release_v1/splits_quality_problem_v2.json"
    assert stage["axis_order"] == ["objective_variant", "arch", "seed"]
    assert document["selection"]["seeds"] == [0, 1, 2, 3]
    assert "all four seed checkpoints" in document["selection"]["paper_checkpoint_policy"]
    assert "not the sole paper estimate" in document["selection"]["deployment_checkpoint_policy"]

    assignments = {cell.index: runner.expected_execution_site(stage, cell) for cell in cells}
    assert list(assignments.values()).count("apollo") == 40
    assert list(assignments.values()).count("goose") == 40
    for architecture in {cell.params["arch"] for cell in cells}:
        sites = [assignments[cell.index] for cell in cells if cell.params["arch"] == architecture]
        assert sites.count("apollo") == sites.count("goose") == 8
    for objective in {cell.params["objective_variant"] for cell in cells}:
        sites = [
            assignments[cell.index]
            for cell in cells
            if cell.params["objective_variant"] == objective
        ]
        assert sites.count("apollo") == sites.count("goose") == 10
    for objective in {cell.params["objective_variant"] for cell in cells}:
        for architecture in {cell.params["arch"] for cell in cells}:
            sites = [
                assignments[cell.index]
                for cell in cells
                if cell.params["objective_variant"] == objective
                and cell.params["arch"] == architecture
            ]
            assert sites.count("apollo") == sites.count("goose") == 2
    assert stage["execution"]["seeds_per_configuration_per_site"] == 2


def test_source_digest_includes_quality_value_v2_trainer(tmp_path: Path) -> None:
    runner = _load_module()
    source = tmp_path / "src" / "embedbench"
    scripts = tmp_path / "scripts"
    source.mkdir(parents=True)
    scripts.mkdir()
    (source / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
    for name in (
        "train_chain.py",
        "train_structural.py",
        "train_quality_v2.py",
        "run_training_grid.py",
    ):
        (scripts / name).write_text(f"# {name}\n", encoding="utf-8")

    before = runner._source_sha256(tmp_path)
    (scripts / "train_quality_v2.py").write_text("# changed V2 trainer\n", encoding="utf-8")

    assert runner._source_sha256(tmp_path) != before


def test_grid_provenance_rejects_omitted_schema_v2_input(tmp_path: Path) -> None:
    runner = _load_module()
    first = tmp_path / "quality_first.jsonl"
    second = tmp_path / "quality_second.jsonl"
    manifest = tmp_path / "splits.json"
    first.write_text('{"instance_id":"first"}\n', encoding="utf-8")
    second.write_text('{"instance_id":"second"}\n', encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "schema": "embedbench.split-manifest",
                "schema_version": 2,
                "provenance": {
                    "inputs": [
                        {
                            "file": path.name,
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        }
                        for path in (first, second)
                    ]
                },
                "splits": {
                    first.name: {"first": "train"},
                    second.name: {"second": "val"},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires the exact corpus input set"):
        runner._data_provenance((first,), manifest)


def test_quality_grid_rejects_execution_on_the_wrong_registered_site() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_training_grid.py"),
            "--grid",
            str(ROOT / "configs" / "training_grid_quality_v2.json"),
            "--stage",
            "quality_value_v2_screen",
            "--index",
            "0",
            "--root",
            str(ROOT),
            "--execution-site",
            "goose",
            "--dry-run",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "assigned to 'apollo'" in completed.stderr


def test_cell_command_uses_fixed_manifest_and_one_training_seed(tmp_path: Path) -> None:
    runner = _load_module()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "data" / "release_v1_1").mkdir(parents=True)
    (tmp_path / "scripts" / "train_chain.py").write_text("", encoding="utf-8")
    for name in ("quality_a.jsonl", "quality_b.jsonl"):
        (tmp_path / "data" / "release_v1_1" / name).write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "data" / "release_v1_1" / "splits_quality_problem_v2.json"
    manifest.write_text("{}\n", encoding="utf-8")
    document = {
        "schema": "embedbench.training-grid",
        "schema_version": 1,
        "grid_id": "tiny",
        "stages": {
            "quality": {
                "trainer": "scripts/train_chain.py",
                "inputs": "data/release_v1_1/quality_*.jsonl",
                "splits": "data/release_v1_1/splits_quality_problem_v2.json",
                "output_dir": "runs/tiny/quality",
                "checkpoint_dir": "runs/tiny/checkpoints",
                "epochs": 2,
                "axis_order": ["arch", "loss", "seed"],
                "axes": {"arch": ["gin"], "loss": ["pairwise"], "seed": [7]},
                "extra_args": ["--deploy-view"],
            }
        },
    }
    cell = runner.expand_stage(document, "quality")[0]

    command, result_path, checkpoint_path = runner.build_command(
        document,
        cell,
        root=tmp_path,
        python="python-test",
        device="cpu",
    )

    assert command[0] == "python-test"
    assert command.count("--seeds") == 1
    assert command[command.index("--seeds") + 1] == "7"
    assert command[command.index("--splits") + 1] == str(manifest)
    assert "--evaluate-test" not in command
    assert "--deploy-view" in command
    assert command[-2:] == ["--out", str(result_path)]
    assert result_path.name == f"{cell.cell_id}.json"
    assert checkpoint_path.name == f"{cell.cell_id}_gin_s7.pt"


def _completed_artifacts(runner, tmp_path: Path):
    corpus = tmp_path / "quality.jsonl"
    splits = tmp_path / "splits.json"
    result = tmp_path / "cell.json"
    checkpoint = tmp_path / "cell_gin_s7.pt"
    receipt = tmp_path / "cell.receipt.json"
    corpus.write_text('{"instance_id":"a"}\n', encoding="utf-8")
    splits.write_text('{"splits":{"quality.jsonl":{"a":"train"}}}\n', encoding="utf-8")
    result.write_text('{"results":[]}\n', encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint-v1")
    provenance = runner._data_provenance((corpus,), splits)
    receipt.write_text(
        json.dumps(
            {
                "schema": "embedbench.training-receipt",
                "schema_version": 1,
                "grid_sha256": "grid-sha",
                "source_sha256": "source-sha",
                "data_provenance": provenance,
                "result_sha256": runner._sha256(result),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": runner._sha256(checkpoint),
                "test_locked": True,
            }
        ),
        encoding="utf-8",
    )
    return corpus, splits, result, checkpoint, receipt, provenance


def test_resume_rejects_changed_corpus_or_split_manifest(tmp_path: Path) -> None:
    runner = _load_module()
    corpus, splits, result, checkpoint, receipt, provenance = _completed_artifacts(runner, tmp_path)

    assert runner._completed(
        receipt,
        result,
        checkpoint,
        grid_sha256="grid-sha",
        source_sha256="source-sha",
        data_provenance=provenance,
    )
    corpus.write_text('{"instance_id":"changed"}\n', encoding="utf-8")
    changed_corpus = runner._data_provenance((corpus,), splits)
    assert not runner._completed(
        receipt,
        result,
        checkpoint,
        grid_sha256="grid-sha",
        source_sha256="source-sha",
        data_provenance=changed_corpus,
    )

    corpus.write_text('{"instance_id":"a"}\n', encoding="utf-8")
    splits.write_text(
        '{"note":"changed","splits":{"quality.jsonl":{"a":"train"}}}\n',
        encoding="utf-8",
    )
    changed_split = runner._data_provenance((corpus,), splits)
    assert not runner._completed(
        receipt,
        result,
        checkpoint,
        grid_sha256="grid-sha",
        source_sha256="source-sha",
        data_provenance=changed_split,
    )


def test_resume_rejects_missing_or_corrupt_checkpoint(tmp_path: Path) -> None:
    runner = _load_module()
    _, _, result, checkpoint, receipt, provenance = _completed_artifacts(runner, tmp_path)

    checkpoint.unlink()
    assert not runner._completed(
        receipt,
        result,
        checkpoint,
        grid_sha256="grid-sha",
        source_sha256="source-sha",
        data_provenance=provenance,
    )
    _, _, result, checkpoint, receipt, provenance = _completed_artifacts(runner, tmp_path)
    checkpoint.write_bytes(b"corrupt")
    assert not runner._completed(
        receipt,
        result,
        checkpoint,
        grid_sha256="grid-sha",
        source_sha256="source-sha",
        data_provenance=provenance,
    )
    _, _, result, checkpoint, receipt, provenance = _completed_artifacts(runner, tmp_path)
    receipt.write_text("[]\n", encoding="utf-8")
    assert not runner._completed(
        receipt,
        result,
        checkpoint,
        grid_sha256="grid-sha",
        source_sha256="source-sha",
        data_provenance=provenance,
    )


def test_goose_two_task_array_covers_deterministic_disjoint_shards(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    (work_root / "scripts").mkdir(parents=True)
    invocation_log = tmp_path / "invocations.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$GRID_TEST_LOG"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    script = ROOT / "scripts" / "goose_training_grid.sbatch"
    base_environment = {
        **os.environ,
        "GRID_TEST_LOG": str(invocation_log),
        "ISINGFOLD_GRID_SIZE": "5",
        "ISINGFOLD_GRID_STAGE": "quality_chain_family_screen",
        "ISINGFOLD_PYTHON": str(fake_python),
        "ISINGFOLD_WORK_ROOT": str(work_root),
        "SLURM_ARRAY_TASK_COUNT": "2",
    }

    for task_id in ("0", "1"):
        subprocess.run(
            ["bash", str(script)],
            check=True,
            env={**base_environment, "SLURM_ARRAY_TASK_ID": task_id},
        )

    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    indices = [int(line.split()[line.split().index("--index") + 1]) for line in invocations]
    assert indices == [0, 2, 4, 1, 3]
    assert sorted(indices) == list(range(5))


def test_goose_accepts_registered_grid_config_override(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    (work_root / "scripts").mkdir(parents=True)
    invocation_log = tmp_path / "invocations.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$GRID_TEST_LOG"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    script = ROOT / "scripts" / "goose_training_grid.sbatch"
    subprocess.run(
        ["bash", str(script)],
        check=True,
        env={
            **os.environ,
            "GRID_TEST_LOG": str(invocation_log),
            "ISINGFOLD_GRID_CONFIG": "configs/training_grid_quality_v2.json",
            "ISINGFOLD_GRID_SIZE": "1",
            "ISINGFOLD_GRID_STAGE": "quality_value_v2_screen",
            "ISINGFOLD_PYTHON": str(fake_python),
            "ISINGFOLD_WORK_ROOT": str(work_root),
            "SLURM_ARRAY_TASK_COUNT": "2",
            "SLURM_ARRAY_TASK_ID": "0",
        },
    )

    invocation = invocation_log.read_text(encoding="utf-8").splitlines()[0]
    assert "--grid configs/training_grid_quality_v2.json" in invocation


def test_goose_two_task_array_can_cover_a_disjoint_global_index_range(
    tmp_path: Path,
) -> None:
    work_root = tmp_path / "work"
    (work_root / "scripts").mkdir(parents=True)
    invocation_log = tmp_path / "invocations.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$GRID_TEST_LOG"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    script = ROOT / "scripts" / "goose_training_grid.sbatch"
    base_environment = {
        **os.environ,
        "GRID_TEST_LOG": str(invocation_log),
        "ISINGFOLD_GRID_FIRST": "1",
        "ISINGFOLD_GRID_LAST": "79",
        "ISINGFOLD_GRID_STEP": "2",
        "ISINGFOLD_GRID_SIZE": "80",
        "ISINGFOLD_GRID_STAGE": "quality_value_v2_screen",
        "ISINGFOLD_PYTHON": str(fake_python),
        "ISINGFOLD_WORK_ROOT": str(work_root),
        "SLURM_ARRAY_TASK_COUNT": "2",
    }

    for task_id in ("0", "1"):
        subprocess.run(
            ["bash", str(script)],
            check=True,
            env={**base_environment, "SLURM_ARRAY_TASK_ID": task_id},
        )

    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    indices = [int(line.split()[line.split().index("--index") + 1]) for line in invocations]
    assert indices == list(range(1, 80, 4)) + list(range(3, 80, 4))
    assert sorted(indices) == list(range(1, 80, 2))
    assert all("--execution-site goose" in invocation for invocation in invocations)


def test_goose_two_workers_per_task_cover_odd_quality_cells_once(
    tmp_path: Path,
) -> None:
    work_root = tmp_path / "work"
    (work_root / "scripts").mkdir(parents=True)
    invocation_log = tmp_path / "invocations.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        '#!/bin/bash\nprintf "%s|%s\\n" "$OMP_NUM_THREADS" "$*" >> "$GRID_TEST_LOG"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    script = ROOT / "scripts" / "goose_training_grid.sbatch"
    base_environment = {
        **os.environ,
        "GRID_TEST_LOG": str(invocation_log),
        "ISINGFOLD_GRID_FIRST": "1",
        "ISINGFOLD_GRID_LAST": "79",
        "ISINGFOLD_GRID_STEP": "2",
        "ISINGFOLD_GRID_SIZE": "80",
        "ISINGFOLD_GRID_STAGE": "quality_value_v2_screen",
        "ISINGFOLD_GRID_WORKERS_PER_TASK": "2",
        "ISINGFOLD_PYTHON": str(fake_python),
        "ISINGFOLD_WORK_ROOT": str(work_root),
        "SLURM_ARRAY_TASK_COUNT": "2",
    }

    for task_id in ("0", "1"):
        subprocess.run(
            ["bash", str(script)],
            check=True,
            env={**base_environment, "SLURM_ARRAY_TASK_ID": task_id},
        )

    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    worker_threads, commands = zip(*(line.split("|", 1) for line in invocations), strict=True)
    indices = [
        int(command.split()[command.split().index("--index") + 1])
        for command in commands
    ]
    assert len(indices) == 40
    assert sorted(indices) == list(range(1, 80, 2))
    assert len(indices) == len(set(indices))
    assert set(worker_threads) == {"4"}


def test_two_apollo_processes_cover_the_even_quality_grid_indices(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    (work_root / "scripts").mkdir(parents=True)
    invocation_log = tmp_path / "invocations.txt"
    fake_python = tmp_path / "fake-python"
    fake_python.write_text(
        '#!/bin/bash\nprintf "%s\\n" "$*" >> "$GRID_TEST_LOG"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    script = ROOT / "scripts" / "apollo_training_grid.sh"
    environment = {
        **os.environ,
        "GRID_TEST_LOG": str(invocation_log),
        "ISINGFOLD_GRID_CONFIG": "configs/training_grid_quality_v2.json",
        "ISINGFOLD_GRID_STEP": "4",
        "ISINGFOLD_PYTHON": str(fake_python),
        "ISINGFOLD_WORK_ROOT": str(work_root),
    }
    for first in ("0", "2"):
        subprocess.run(
            ["bash", str(script), "quality_value_v2_screen", first, "78"],
            check=True,
            env=environment,
        )

    invocations = invocation_log.read_text(encoding="utf-8").splitlines()
    indices = [int(line.split()[line.split().index("--index") + 1]) for line in invocations]
    assert indices == list(range(0, 80, 4)) + list(range(2, 80, 4))
    assert sorted(indices) == list(range(0, 80, 2))
    assert all(
        "--grid configs/training_grid_quality_v2.json" in invocation
        and "--execution-site apollo" in invocation
        for invocation in invocations
    )


def test_goose_script_rejects_any_array_size_other_than_two(tmp_path: Path) -> None:
    work_root = tmp_path / "work"
    work_root.mkdir()
    script = ROOT / "scripts" / "goose_training_grid.sbatch"
    completed = subprocess.run(
        ["bash", str(script)],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "ISINGFOLD_GRID_SIZE": "5",
            "ISINGFOLD_GRID_STAGE": "structural_family_screen",
            "ISINGFOLD_WORK_ROOT": str(work_root),
            "SLURM_ARRAY_TASK_COUNT": "3",
            "SLURM_ARRAY_TASK_ID": "0",
        },
    )

    assert completed.returncode != 0
    assert "exactly two" in completed.stderr
