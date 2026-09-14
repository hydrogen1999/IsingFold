from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]


def _load_training_artifacts():
    path = ROOT / "scripts" / "training_artifacts.py"
    spec = importlib.util.spec_from_file_location("artifact_training_artifacts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_script(monkeypatch: pytest.MonkeyPatch, name: str):
    scripts = ROOT / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    path = scripts / name
    spec = importlib.util.spec_from_file_location(f"artifact_test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_training_splits():
    path = ROOT / "scripts" / "training_splits.py"
    spec = importlib.util.spec_from_file_location("artifact_training_splits", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quality_record(index: int, split: str) -> dict:
    return {
        "instance_id": f"problem-{split}-{index}-m",
        "window_nodes": [0, 1],
        "window_edges": [[0, 1]],
        "frozen": {"2": [2]},
        "frozen_adjacency": {"0": [2], "1": [2]},
        "neighbours": [2],
        "edge_J": [-1.0],
        "neighbour_degree": [1],
        "neighbour_chain_size": [1],
        "candidates": [[0], [1]],
        "p_solve": [0.8, 0.1],
        "stage": [2, 2],
        "best_index": 0,
        "resource_index": 0,
        "original_index": 0,
        "source": "minorminer",
        "topology": "chimera",
        "size": 1,
        "difficulty": "test",
        "l_cap": 4,
        "focus_h": float(index + 1),
        "problem": {
            "h": {"0": float(index + 1), "2": 0.0},
            "J": [[0, 2, -1.0]],
            "e0": float(-index - 1),
        },
    }


def _write_v2_manifest(corpus: Path, manifest: Path, records: list[dict]) -> None:
    assignments = {
        str(record["instance_id"]): split
        for record, split in zip(records, ("train", "val", "test"), strict=True)
    }
    manifest.write_text(
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
                "splits": {corpus.name: assignments},
            }
        ),
        encoding="utf-8",
    )


def test_runtime_provenance_has_stable_json_safe_cpu_schema() -> None:
    import torch

    artifacts = _load_training_artifacts()

    provenance = artifacts.runtime_provenance(torch.device("cpu"))

    assert tuple(provenance) == (
        "schema",
        "schema_version",
        "python",
        "torch",
        "device",
        "packages",
    )
    assert tuple(provenance["python"]) == ("implementation", "version", "executable")
    assert tuple(provenance["torch"]) == (
        "version",
        "cuda_available",
        "cuda_runtime_version",
        "cudnn_version",
    )
    assert provenance["device"] == {
        "selected": "cpu",
        "type": "cpu",
        "index": None,
        "gpu": None,
    }
    assert tuple(provenance["packages"]) == (
        "embedbench",
        "numpy",
        "scipy",
        "networkx",
        "dimod",
        "dwave-samplers",
        "dwave-networkx",
        "minorminer",
    )
    json.dumps(provenance, sort_keys=True, allow_nan=False)


def test_runtime_provenance_records_allocated_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    artifacts = _load_training_artifacts()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: f"GPU-{index}")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (9, 0))
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda index: SimpleNamespace(total_memory=48 * 1024**3),
    )

    provenance = artifacts.runtime_provenance(torch.device("cuda"))

    assert provenance["device"] == {
        "selected": "cuda",
        "type": "cuda",
        "index": 2,
        "gpu": {
            "name": "GPU-2",
            "compute_capability": [9, 0],
            "total_memory_bytes": 48 * 1024**3,
        },
    }


def test_data_provenance_carries_verified_v2_input_hashes(tmp_path: Path) -> None:
    training_splits = _load_training_splits()
    corpus = tmp_path / "quality.jsonl"
    records = [_quality_record(index, split) for index, split in enumerate(("a", "b", "c"))]
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = tmp_path / "splits.json"
    _write_v2_manifest(corpus, manifest, records)

    provenance = training_splits.data_provenance([corpus], manifest)

    assert provenance["split_manifest"] == {
        "file": manifest.name,
        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "schema": "embedbench.split-manifest",
        "schema_version": 2,
    }
    assert provenance["corpus_inputs"] == [
        {
            "file": corpus.name,
            "sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        }
    ]


def test_structural_trainer_saves_portable_checkpoint_and_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch

    trainer = _load_script(monkeypatch, "train_structural.py")
    partitions = ([{"id": "train"}], [{"id": "val"}], [{"id": "test"}])
    model = torch.nn.Linear(2, 1)
    provenance = {
        "split_manifest": {"file": "splits.json", "sha256": "a" * 64},
        "corpus_inputs": [{"file": "corpus.jsonl", "sha256": "b" * 64}],
    }
    runtime = {"schema": "test-runtime", "schema_version": 1}
    output = tmp_path / "results" / "structural.json"
    save_prefix = tmp_path / "checkpoints" / "structural"

    monkeypatch.setattr(trainer, "load_split_records", lambda *args, **kwargs: partitions)
    monkeypatch.setattr(trainer, "data_provenance", lambda *args, **kwargs: provenance)
    monkeypatch.setattr(trainer, "runtime_provenance", lambda device: runtime)
    monkeypatch.setattr(trainer, "encode", lambda record: record)
    monkeypatch.setattr(trainer, "build_arch", lambda *args, **kwargs: model)
    monkeypatch.setattr(trainer, "train", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        trainer,
        "evaluate",
        lambda model, records, **kwargs: {"n": len(records)},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_structural.py",
            "corpus.jsonl",
            "--splits",
            "splits.json",
            "--arch",
            "gin",
            "--hidden",
            "12",
            "--layers",
            "2",
            "--heads",
            "2",
            "--device",
            "cpu",
            "--save",
            str(save_prefix),
            "--out",
            str(output),
        ],
    )

    trainer.main()

    checkpoint_path = Path(f"{save_prefix}_gin_s0.pt")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["meta"]["arch"] == "gin"
    assert checkpoint["meta"]["hidden"] == 12
    assert checkpoint["meta"]["layers"] == 2
    assert checkpoint["meta"]["heads"] == 2
    assert checkpoint["meta"]["data_provenance"] == provenance
    assert checkpoint["meta"]["runtime_provenance"] == runtime
    assert all(tensor.device.type == "cpu" for tensor in checkpoint["state"].values())
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["data_provenance"] == provenance
    assert result["runtime_provenance"] == runtime


def test_structural_loader_restores_non_mpnn_architecture(tmp_path: Path) -> None:
    import torch
    from embedbench.models_structural import build_arch, load_model

    model = build_arch("gin", hidden=8, layers=1, heads=1)
    checkpoint = tmp_path / "gin.pt"
    torch.save(
        {
            "state": model.state_dict(),
            "meta": {"arch": "gin", "hidden": 8, "layers": 1, "heads": 1},
        },
        checkpoint,
    )

    loaded = load_model(checkpoint)

    assert loaded.state_dict().keys() == model.state_dict().keys()


def test_chain_trainer_creates_checkpoint_dir_and_records_v2_provenance(
    tmp_path: Path,
) -> None:
    import torch

    records = [_quality_record(index, split) for index, split in enumerate(("a", "b", "c"))]
    corpus = tmp_path / "quality.jsonl"
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    manifest = tmp_path / "splits.json"
    _write_v2_manifest(corpus, manifest, records)
    output = tmp_path / "results" / "chain.json"
    save_prefix = tmp_path / "checkpoints" / "chain"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_chain.py"),
            str(corpus),
            "--splits",
            str(manifest),
            "--epochs",
            "1",
            "--hidden",
            "8",
            "--layers",
            "1",
            "--heads",
            "1",
            "--device",
            "cpu",
            "--save",
            str(save_prefix),
            "--out",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        Path(f"{save_prefix}_mpnn_s0.pt"),
        map_location="cpu",
        weights_only=False,
    )
    assert result["data_provenance"] == checkpoint["meta"]["data_provenance"]
    assert result["runtime_provenance"] == checkpoint["meta"]["runtime_provenance"]
    assert result["runtime_provenance"]["device"]["type"] == "cpu"
    assert result["data_provenance"]["corpus_inputs"][0]["file"] == corpus.name
    assert all(tensor.device.type == "cpu" for tensor in checkpoint["state"].values())


def test_chain_loader_keeps_preprocessing_local_to_each_scorer(tmp_path: Path) -> None:
    import torch
    from embedbench.models_chain import build_chain_arch
    from embedbench.models_hetero import load_scorer

    record = _quality_record(0, "test")
    plain_model = build_chain_arch(
        "gin", hidden=8, layers=1, heads=1, neighbour_feats=False
    )
    neighbour_model = build_chain_arch(
        "gin", hidden=8, layers=1, heads=1, neighbour_feats=True
    )
    plain_path = tmp_path / "plain.pt"
    neighbour_path = tmp_path / "neighbour.pt"
    torch.save(
        {
            "state": plain_model.state_dict(),
            "meta": {
                "arch": "gin",
                "hidden": 8,
                "layers": 1,
                "heads": 1,
                "preprocessing": {
                    "neighbour_feats": False,
                    "no_length_feats": True,
                    "deploy_view": False,
                },
            },
        },
        plain_path,
    )
    torch.save(
        {
            "state": neighbour_model.state_dict(),
            "meta": {
                "arch": "gin",
                "hidden": 8,
                "layers": 1,
                "heads": 1,
                "preprocessing": {
                    "neighbour_feats": True,
                    "no_length_feats": False,
                    "deploy_view": False,
                },
            },
        },
        neighbour_path,
    )

    plain_scorer = load_scorer(plain_path)
    neighbour_scorer = load_scorer(neighbour_path)

    assert len(neighbour_scorer(record)) == 2
    assert len(plain_scorer(record)) == 2


def test_chain_scorer_applies_no_length_preprocessing() -> None:
    import torch
    from embedbench.models_chain import ChainScorer

    class CaptureModel(torch.nn.Module):
        def forward(self, x, adj, masks, feats):
            self.features = feats.detach().clone()
            return torch.zeros(feats.shape[0])

    model = CaptureModel()
    scorer = ChainScorer(model, no_length_feats=True)

    scorer(_quality_record(0, "test"))

    assert torch.count_nonzero(model.features[:, 0]) == 0
    assert torch.count_nonzero(model.features[:, 9]) == 0


def test_deploy_view_checkpoint_transforms_corpus_but_not_seam_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import embedbench.models_chain as models_chain
    import embedbench.structural as structural
    import torch

    class CaptureModel(torch.nn.Module):
        def forward(self, x, adj, masks, feats):
            return torch.zeros(feats.shape[0])

    calls = []

    def fake_deployment_view(record, host):
        calls.append((record["source"], host))
        return {**record, "_deployment_view": True}

    monkeypatch.setattr(structural, "host_graph", lambda topology, size: (topology, size))
    monkeypatch.setattr(models_chain, "deployment_view", fake_deployment_view)
    scorer = models_chain.ChainScorer(CaptureModel(), deploy_view=True)

    scorer(_quality_record(0, "test"))
    seam_record = {**_quality_record(0, "test"), "source": "seam", "topology": "seam"}
    scorer(seam_record)

    assert calls == [("minorminer", ("chimera", 1))]


def test_non_deploy_view_checkpoint_rejects_seam_records(tmp_path: Path) -> None:
    import torch
    from embedbench.models_chain import build_chain_arch
    from embedbench.models_hetero import load_scorer

    model = build_chain_arch("mpnn", hidden=8, layers=1, heads=1)
    checkpoint = tmp_path / "raw-view.pt"
    torch.save(
        {
            "state": model.state_dict(),
            "meta": {
                "arch": "mpnn",
                "hidden": 8,
                "layers": 1,
                "heads": 1,
                "preprocessing": {
                    "neighbour_feats": False,
                    "no_length_feats": False,
                    "deploy_view": False,
                },
            },
        },
        checkpoint,
    )
    scorer = load_scorer(checkpoint)
    seam_record = {**_quality_record(0, "test"), "source": "seam", "topology": "seam"}

    with pytest.raises(ValueError, match="trained without deploy-view"):
        scorer(seam_record)


@pytest.mark.parametrize("flag", ["neighbour_feats", "no_length_feats"])
def test_hetero_loader_rejects_inapplicable_chain_feature_flags(
    tmp_path: Path,
    flag: str,
) -> None:
    import torch
    from embedbench.models_hetero import build_hetero, load_scorer

    model = build_hetero(hidden=8, layers=1, heads=1)
    checkpoint = tmp_path / "hetero.pt"
    torch.save(
        {
            "state": model.state_dict(),
            "meta": {
                "arch": "hetero",
                "hidden": 8,
                "layers": 1,
                "heads": 1,
                "preprocessing": {
                    "neighbour_feats": flag == "neighbour_feats",
                    "no_length_feats": flag == "no_length_feats",
                    "deploy_view": True,
                },
            },
        },
        checkpoint,
    )

    expected = "neighbour-feats" if flag == "neighbour_feats" else "no-length-feats"
    with pytest.raises(ValueError, match=rf"{expected}.*hetero"):
        load_scorer(checkpoint)
