from __future__ import annotations

import copy
import hashlib
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from isingfold.rl.checkpoint import (
    ACTION_SCHEMA,
    CHECKPOINT_SCHEMA,
    CHECKPOINT_VERSION,
    PPO_TRAINER_STATE_SCHEMA,
    RUNTIME_DEPENDENCIES,
    RUNTIME_IMPLEMENTATION_SCHEMA,
    RUNTIME_IMPLEMENTATION_VERSION,
    RUNTIME_MODULE_SOURCES,
    RUNTIME_NATIVE_MODULES,
    CheckpointCompatibilityError,
    CheckpointIntegrityError,
    load_checkpoint,
    runtime_implementation_registry,
    save_checkpoint,
)
from isingfold.rl.contracts import stable_digest


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _context(*, cap: int = 24) -> dict[str, object]:
    return {
        "context_version": "rev2-pilot-1",
        "qubit_cap": cap,
        "strength_ratios": [0.5, 1.0, 2.0, 4.0],
        "caps": {"decisions": 32, "evaluator_reads": 8192},
    }


def _normalizer(*, scale: float = 2.0) -> dict[str, object]:
    return {
        "version": "train-only-v1",
        "logical_h": {"transform": "signed_log1p", "scale": scale},
        "knownness": True,
    }


def _trained_pair() -> tuple[nn.Module, torch.optim.Optimizer]:
    model = nn.Sequential(nn.Linear(3, 5), nn.SiLU(), nn.Linear(5, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    loss = model(torch.tensor([[1.0, -2.0, 0.5]])).square().sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def _save(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        context=_context(),
        feature_normalizer=_normalizer(),
        proposal_version="proposal-rev2-v1",
        selector_digest=_digest("selector-v1"),
        training_lineages=("lineage-z", "lineage-a"),
        counters={"updates": 7, "episodes": 448, "transitions": 1931},
    )


def _load_kwargs(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, object]:
    return {
        "model": model,
        "optimizer": optimizer,
        "expected_context": _context(),
        "expected_feature_normalizer": _normalizer(),
        "expected_proposal_version": "proposal-rev2-v1",
        "expected_selector_digest": _digest("selector-v1"),
        "expected_training_lineages": ("lineage-a", "lineage-z"),
    }


def test_runtime_implementation_registry_is_canonical_and_complete() -> None:
    registry = runtime_implementation_registry()

    assert registry == runtime_implementation_registry()
    assert registry["schema"] == RUNTIME_IMPLEMENTATION_SCHEMA
    assert registry["schema_version"] == RUNTIME_IMPLEMENTATION_VERSION
    assert set(registry["modules"]) == {name for name, _ in RUNTIME_MODULE_SOURCES}
    assert set(registry["dependencies"]) == {"python", *RUNTIME_DEPENDENCIES}
    assert set(registry["native_artifacts"]) == set(RUNTIME_NATIVE_MODULES)
    assert "isingfold.rl.data.ground_certificate" in registry["modules"]
    assert "isingfold.rl.data.action_certificate" in registry["modules"]
    for name, relative_path in RUNTIME_MODULE_SOURCES:
        module = registry["modules"][name]
        assert module["source_path"] == relative_path
        assert not Path(module["source_path"]).is_absolute()
        assert len(module["sha256"]) == 64
    for artifact in registry["native_artifacts"].values():
        assert artifact["artifact_kind"] == "extension-module"
        assert len(artifact["sha256"]) == 64
    assert len(stable_digest(registry)) == 64


def test_checkpoint_rejects_runtime_source_drift_before_model_mutation(
    tmp_path: Path,
) -> None:
    model, optimizer = _trained_pair()
    path = tmp_path / "pilot.ckpt"
    registry = runtime_implementation_registry()
    changed = copy.deepcopy(registry)
    changed["modules"]["isingfold.rl.router"]["sha256"] = _digest("changed-router")
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        context=_context(),
        feature_normalizer=_normalizer(),
        proposal_version="proposal-rev2-v1",
        selector_digest=_digest("selector-v1"),
        training_lineages=("lineage-z", "lineage-a"),
        counters={"updates": 7},
        runtime_registry=changed,
    )
    for parameter in model.parameters():
        parameter.data.add_(17.0)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    with pytest.raises(CheckpointCompatibilityError, match="runtime implementation"):
        load_checkpoint(path, **_load_kwargs(model, optimizer))

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_checkpoint_supports_explicit_runtime_registry_injection(tmp_path: Path) -> None:
    model, optimizer = _trained_pair()
    path = tmp_path / "pilot.ckpt"
    registry = runtime_implementation_registry()
    registry["modules"]["isingfold.rl.router"]["sha256"] = _digest("fixture-router")
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        context=_context(),
        feature_normalizer=_normalizer(),
        proposal_version="proposal-rev2-v1",
        selector_digest=_digest("selector-v1"),
        training_lineages=("lineage-z", "lineage-a"),
        counters={"updates": 7},
        runtime_registry=registry,
    )

    loaded = load_checkpoint(
        path,
        **_load_kwargs(model, optimizer),
        expected_runtime_registry=registry,
        expected_runtime_digest=stable_digest(registry),
    )

    assert loaded.runtime_implementation_registry == registry


def test_checkpoint_accepts_an_explicit_expected_runtime_digest(tmp_path: Path) -> None:
    model, optimizer = _trained_pair()
    path = tmp_path / "pilot.ckpt"
    _save(path, model, optimizer)
    registry_digest = stable_digest(runtime_implementation_registry())

    loaded = load_checkpoint(
        path,
        **_load_kwargs(model, optimizer),
        expected_runtime_digest=registry_digest,
    )

    assert loaded.runtime_implementation_digest == registry_digest
    assert loaded.runtime_implementation_registry == runtime_implementation_registry()


def test_checkpoint_rejects_an_explicit_runtime_digest_mismatch(tmp_path: Path) -> None:
    model, optimizer = _trained_pair()
    path = tmp_path / "pilot.ckpt"
    _save(path, model, optimizer)

    with pytest.raises(CheckpointCompatibilityError, match="runtime implementation"):
        load_checkpoint(
            path,
            **_load_kwargs(model, optimizer),
            expected_runtime_digest=_digest("different-runtime"),
        )


def test_checkpoint_rejects_conflicting_expected_registry_and_digest(
    tmp_path: Path,
) -> None:
    model, optimizer = _trained_pair()
    path = tmp_path / "pilot.ckpt"
    _save(path, model, optimizer)

    with pytest.raises(ValueError, match="expected runtime implementation digest"):
        load_checkpoint(
            path,
            **_load_kwargs(model, optimizer),
            expected_runtime_registry=runtime_implementation_registry(),
            expected_runtime_digest=_digest("different-runtime"),
        )


def test_checkpoint_round_trip_restores_training_and_next_rng_draws(tmp_path: Path) -> None:
    random.seed(1103)
    np.random.seed(2207)
    torch.manual_seed(3301)
    model, optimizer = _trained_pair()
    saved_parameters = {name: value.detach().clone() for name, value in model.state_dict().items()}
    path = tmp_path / "pilot.ckpt"

    metadata = save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        context=_context(),
        feature_normalizer=_normalizer(),
        proposal_version="proposal-rev2-v1",
        selector_digest=_digest("selector-v1"),
        training_lineages=("lineage-z", "lineage-a"),
        counters={"updates": 7, "episodes": 448, "transitions": 1931},
    )
    expected_draws = (random.random(), np.random.random(), torch.rand(4))

    for parameter in model.parameters():
        parameter.data.add_(100.0)
    optimizer.param_groups[0]["lr"] = 0.9
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)

    loaded = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        expected_context=_context(),
        expected_feature_normalizer=_normalizer(),
        expected_proposal_version="proposal-rev2-v1",
        expected_selector_digest=_digest("selector-v1"),
        expected_training_lineages=("lineage-a", "lineage-z"),
    )

    assert path.is_file()
    assert not list(tmp_path.glob(".pilot.ckpt.*.tmp"))
    assert metadata == loaded
    assert loaded.schema == CHECKPOINT_SCHEMA
    assert loaded.schema_version == CHECKPOINT_VERSION
    assert loaded.action_schema == ACTION_SCHEMA
    assert loaded.context_snapshot == _context()
    assert loaded.feature_normalizer == _normalizer()
    assert loaded.training_lineages == ("lineage-a", "lineage-z")
    assert loaded.counters == {"episodes": 448, "transitions": 1931, "updates": 7}
    assert len(loaded.payload_digest) == 64
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, saved_parameters[name])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    assert optimizer.state
    assert random.random() == expected_draws[0]
    assert np.random.random() == expected_draws[1]
    torch.testing.assert_close(torch.rand(4), expected_draws[2])


def test_checkpoint_round_trip_binds_construction_trainer_state(tmp_path: Path) -> None:
    model, optimizer = _trained_pair()
    path = tmp_path / "construction.ckpt"
    trainer_state = {"lambda_f": 0.375, "updates_done": 7}

    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        context=_context(),
        feature_normalizer=_normalizer(),
        proposal_version="proposal-rev2-v1",
        selector_digest=_digest("selector-v1"),
        training_lineages=("lineage-a",),
        counters={"updates": 7},
        trainer_state=trainer_state,
    )
    loaded = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        expected_context=_context(),
        expected_feature_normalizer=_normalizer(),
        expected_proposal_version="proposal-rev2-v1",
        expected_selector_digest=_digest("selector-v1"),
        expected_training_lineages=("lineage-a",),
        expected_trainer_state_schema=PPO_TRAINER_STATE_SCHEMA,
    )

    assert loaded.trainer_state == trainer_state


def test_checkpoint_rejects_trainer_state_tampering(tmp_path: Path) -> None:
    model, optimizer = _trained_pair()
    path = tmp_path / "construction.ckpt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        context=_context(),
        feature_normalizer=_normalizer(),
        proposal_version="proposal-rev2-v1",
        selector_digest=_digest("selector-v1"),
        training_lineages=("lineage-a",),
        counters={"updates": 7},
        trainer_state={"lambda_f": 0.375, "updates_done": 7},
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["trainer_state"]["lambda_f"] = 9.0
    torch.save(payload, path)

    with pytest.raises(CheckpointIntegrityError, match="payload digest"):
        load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_context=_context(),
            expected_feature_normalizer=_normalizer(),
            expected_proposal_version="proposal-rev2-v1",
            expected_selector_digest=_digest("selector-v1"),
            expected_training_lineages=("lineage-a",),
            expected_trainer_state_schema=PPO_TRAINER_STATE_SCHEMA,
        )


def test_checkpoint_rejects_trainer_state_schema_before_mutation(tmp_path: Path) -> None:
    model, optimizer = _trained_pair()
    path = tmp_path / "pilot.ckpt"
    _save(path, model, optimizer)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    with pytest.raises(CheckpointCompatibilityError, match="trainer state schema"):
        load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_context=_context(),
            expected_feature_normalizer=_normalizer(),
            expected_proposal_version="proposal-rev2-v1",
            expected_selector_digest=_digest("selector-v1"),
            expected_training_lineages=("lineage-a", "lineage-z"),
            expected_trainer_state_schema=PPO_TRAINER_STATE_SCHEMA,
        )

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"expected_context": _context(cap=25)}, "context"),
        ({"expected_feature_normalizer": _normalizer(scale=3.0)}, "normalizer"),
        ({"expected_proposal_version": "proposal-rev2-v2"}, "proposal"),
        ({"expected_selector_digest": _digest("selector-v2")}, "selector"),
        ({"expected_action_schema": (*ACTION_SCHEMA[:-1], "HALT")}, "action schema"),
        ({"expected_training_lineages": ("lineage-other",)}, "lineage"),
    ],
)
def test_checkpoint_rejects_incompatible_experiment_before_mutation(
    tmp_path: Path, override: dict[str, object], match: str
) -> None:
    model, optimizer = _trained_pair()
    path = tmp_path / "pilot.ckpt"
    _save(path, model, optimizer)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    kwargs: dict[str, object] = {
        "model": model,
        "optimizer": optimizer,
        "expected_context": _context(),
        "expected_feature_normalizer": _normalizer(),
        "expected_proposal_version": "proposal-rev2-v1",
        "expected_selector_digest": _digest("selector-v1"),
        "expected_training_lineages": ("lineage-a", "lineage-z"),
    }
    kwargs.update(override)

    with pytest.raises(CheckpointCompatibilityError, match=match):
        load_checkpoint(path, **kwargs)

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])


def test_checkpoint_rejects_unknown_schema(tmp_path: Path) -> None:
    model, optimizer = _trained_pair()
    path = tmp_path / "pilot.ckpt"
    _save(path, model, optimizer)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["schema"] = "some-other-project.checkpoint"
    torch.save(payload, path)

    with pytest.raises(CheckpointCompatibilityError, match="schema"):
        load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_context=_context(),
            expected_feature_normalizer=_normalizer(),
            expected_proposal_version="proposal-rev2-v1",
            expected_selector_digest=_digest("selector-v1"),
            expected_training_lineages=("lineage-a", "lineage-z"),
        )


def test_checkpoint_rejects_pre_registry_schema_version(tmp_path: Path) -> None:
    model, optimizer = _trained_pair()
    path = tmp_path / "pilot.ckpt"
    _save(path, model, optimizer)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["schema_version"] = 2
    torch.save(payload, path)

    with pytest.raises(CheckpointCompatibilityError, match="schema version"):
        load_checkpoint(path, **_load_kwargs(model, optimizer))


def test_checkpoint_rejects_payload_digest_mismatch(tmp_path: Path) -> None:
    model, optimizer = _trained_pair()
    path = tmp_path / "pilot.ckpt"
    _save(path, model, optimizer)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["metadata"]["counters"]["updates"] = 8
    torch.save(payload, path)

    with pytest.raises(CheckpointIntegrityError, match="payload digest"):
        load_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            expected_context=_context(),
            expected_feature_normalizer=_normalizer(),
            expected_proposal_version="proposal-rev2-v1",
            expected_selector_digest=_digest("selector-v1"),
            expected_training_lineages=("lineage-a", "lineage-z"),
        )


def test_checkpoint_rejects_model_schema_mismatch_before_loading(tmp_path: Path) -> None:
    source, source_optimizer = _trained_pair()
    path = tmp_path / "pilot.ckpt"
    _save(path, source, source_optimizer)
    incompatible = nn.Linear(3, 2)
    incompatible_optimizer = torch.optim.AdamW(incompatible.parameters(), lr=3e-4)
    before = incompatible.weight.detach().clone()

    with pytest.raises(CheckpointCompatibilityError, match="model schema"):
        load_checkpoint(
            path,
            model=incompatible,
            optimizer=incompatible_optimizer,
            expected_context=_context(),
            expected_feature_normalizer=_normalizer(),
            expected_proposal_version="proposal-rev2-v1",
            expected_selector_digest=_digest("selector-v1"),
            expected_training_lineages=("lineage-a", "lineage-z"),
        )

    torch.testing.assert_close(incompatible.weight, before)


def test_checkpoint_rejects_nonfinite_or_invalid_metadata(tmp_path: Path) -> None:
    model, optimizer = _trained_pair()

    with pytest.raises(ValueError, match="finite"):
        save_checkpoint(
            tmp_path / "bad.ckpt",
            model=model,
            optimizer=optimizer,
            context=_context(),
            feature_normalizer={"scale": float("nan")},
            proposal_version="proposal-rev2-v1",
            selector_digest=_digest("selector-v1"),
            training_lineages=("lineage-a",),
            counters={"updates": 0},
        )
    with pytest.raises(ValueError, match="nonnegative integer"):
        save_checkpoint(
            tmp_path / "bad-counter.ckpt",
            model=model,
            optimizer=optimizer,
            context=_context(),
            feature_normalizer=_normalizer(),
            proposal_version="proposal-rev2-v1",
            selector_digest=_digest("selector-v1"),
            training_lineages=("lineage-a",),
            counters={"updates": -1},
        )
