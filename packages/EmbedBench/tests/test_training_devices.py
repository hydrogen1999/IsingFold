from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]


def _load_training_device():
    path = ROOT / "scripts" / "training_device.py"
    spec = importlib.util.spec_from_file_location("training_device", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_script(monkeypatch: pytest.MonkeyPatch, name: str):
    scripts = ROOT / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    path = scripts / name
    spec = importlib.util.spec_from_file_location(f"device_test_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("script", ["train_structural.py", "train_chain.py"])
def test_training_cli_exposes_device_selection(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / script), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--device" in result.stdout
    assert "{auto,cpu,cuda}" in result.stdout


def test_auto_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    training_device = _load_training_device()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert training_device.resolve_device("auto") == torch.device("cpu")


def test_explicit_cuda_fails_early_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch

    training_device = _load_training_device()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA was requested"):
        training_device.resolve_device("cuda")


def test_structural_trainer_seeds_before_model_initialization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import torch

    trainer = _load_script(monkeypatch, "train_structural.py")
    partitions = ([{"id": "train"}], [{"id": "val"}], [{"id": "test"}])
    events = []

    class FakeModel:
        def parameters(self):
            return []

    monkeypatch.setattr(trainer, "load_split_records", lambda *args, **kwargs: partitions)
    monkeypatch.setattr(
        trainer,
        "data_provenance",
        lambda *args, **kwargs: {
            "split_manifest": {"sha256": "split-digest"},
            "corpus_inputs": [],
        },
    )
    monkeypatch.setattr(trainer, "encode", lambda record: record)
    monkeypatch.setattr(
        trainer,
        "seed_device",
        lambda seed, device: events.append(("seed", seed, str(device))),
    )
    monkeypatch.setattr(
        trainer,
        "build_arch",
        lambda *args: events.append(("build",)) or FakeModel(),
    )
    monkeypatch.setattr(
        trainer,
        "train",
        lambda *args, **kwargs: events.append(("train", str(kwargs["device"]))) or 1,
    )
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
            "--device",
            "cpu",
            "--out",
            str(tmp_path / "result.json"),
        ],
    )

    trainer.main()

    assert events[:3] == [("seed", 0, str(torch.device("cpu"))), ("build",), ("train", "cpu")]


def test_length_balanced_pair_loss_stays_on_score_device() -> None:
    import torch
    from embedbench.models_chain import pair_loss

    scores = torch.empty(3, device="meta")
    loss = pair_loss(
        scores,
        [0.9, 0.5, 0.1],
        tol=0.01,
        stage=[2, 2, 2],
        lengths=[3, 1, 2],
        length_balance=True,
    )

    assert loss is not None
    assert loss.device.type == "meta"


def test_listwise_loss_stays_on_score_device() -> None:
    import torch
    from embedbench.models_hetero import listwise_loss

    loss = listwise_loss(torch.empty(2, device="meta"), [0.8, 0.2], [2, 1])

    assert loss.device.type == "meta"


def test_hetero_forward_uses_model_device() -> None:
    from embedbench.models_hetero import HeteroEncoded, build_hetero

    encoded = HeteroEncoded(
        xq=np.zeros((2, 10), dtype=np.float32),
        xv=np.zeros((2, 7), dtype=np.float32),
        Eqq=np.zeros((2, 2, 5), dtype=np.float32),
        Avv=np.zeros((2, 2), dtype=np.float32),
        Mvq=np.array([[0, 0], [0, 1]], dtype=np.float32),
        cand_masks=np.array([[1, 0]], dtype=np.float32),
        cand_contacts=[[]],
        focus_index=0,
        focus_h=0.5,
        p=[0.8],
        stage=[2],
        best_index=0,
        resource_index=0,
        original_index=0,
        source="test",
        instance_id="test-0",
        topology="test",
    )
    model = build_hetero(hidden=8, layers=1, heads=1).to("meta")

    scores = model(encoded)

    assert scores.device.type == "meta"
