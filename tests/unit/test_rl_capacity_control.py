"""Registered post-selection parameter-capacity controls for the model ladder."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isingfold.rl.capacity_control import (
    build_capacity_diagnostic_model,
    build_capacity_diagnostic_plan,
    load_capacity_control_registry,
    parameter_census,
)
from isingfold.rl.data.import_embedbench import content_digest
from isingfold.rl.model import MODEL_FAMILIES, build_model


ROOT = Path(__file__).resolve().parents[2]
GRID = ROOT / "configs" / "rl_grid_hybrid_v1.json"
REGISTRY = ROOT / "configs" / "rl_capacity_control_hybrid_v1.json"


def test_census_counts_trainable_parameters_from_instantiated_models() -> None:
    census = {row.model_id: row for row in parameter_census(improvement_mode=True)}

    assert {
        family: (row.trainable_parameters, row.total_parameters, row.frozen_parameters)
        for family, row in census.items()
    } == {
        "if-mlp": (1_930_083, 1_971_300, 41_217),
        "if-dual": (3_041_379, 3_082_596, 41_217),
        "if-core": (5_262_435, 5_303_652, 41_217),
    }
    assert all(len(row.parameter_signature_sha256) == 64 for row in census.values())


def test_registry_preserves_the_9_plus_18_grid_and_matches_runtime_capacity() -> None:
    before = GRID.read_bytes()

    registry = load_capacity_control_registry(REGISTRY, main_grid_path=GRID)

    assert GRID.read_bytes() == before
    assert registry.main_grid_cell_counts == {"representation": 9, "rl_value": 18}
    assert tuple(row.model_id for row in registry.main_census) == MODEL_FAMILIES
    assert tuple(row.variant_id for row in registry.variants) == (
        "if-core-v1-reference",
        "if-dual-depth9-v1",
    )
    reference, control = registry.variants
    assert reference.width == control.width == 128
    assert reference.trainable_parameters == 5_262_435
    assert control.trainable_parameters == 5_263_971
    assert registry.relative_parameter_gap == pytest.approx(1_536 / 5_262_435)
    assert registry.relative_parameter_gap <= registry.maximum_relative_parameter_gap
    assert (
        registry.training_protocol.epochs,
        registry.training_protocol.minibatch_records,
        registry.training_protocol.learning_rate,
        registry.training_protocol.weight_decay,
    ) == (200, 32, 3e-4, 0.0)
    assert registry.training_protocol.minimum_resolved_rows == 128
    assert registry.training_protocol.minimum_resolved_lineages == 128


def test_diagnostic_factory_is_separate_from_main_selection_factory() -> None:
    registry = load_capacity_control_registry(REGISTRY, main_grid_path=GRID)

    reference = build_capacity_diagnostic_model(registry, "if-core-v1-reference")
    control = build_capacity_diagnostic_model(registry, "if-dual-depth9-v1")

    assert (len(reference.local_l), len(reference.fusion)) == (3, 2)
    assert (len(control.local_l), len(control.fusion)) == (9, 0)
    assert MODEL_FAMILIES == ("if-mlp", "if-dual", "if-core")
    with pytest.raises(ValueError, match="model family"):
        build_model("if-dual-depth9-v1")
    with pytest.raises(ValueError, match="capacity diagnostic"):
        build_capacity_diagnostic_model(registry, "if-core-but-wider")


def test_diagnostic_plan_is_post_selection_only_and_never_selects_a_seed() -> None:
    registry = load_capacity_control_registry(REGISTRY, main_grid_path=GRID)
    selection_sha256 = "a" * 64
    selection_record_digest = "b" * 64

    plan = build_capacity_diagnostic_plan(
        registry,
        selected_simpler="if-dual",
        representation_selection_sha256=selection_sha256,
        representation_selection_record_digest=selection_record_digest,
    )

    assert len(plan.cells) == 6
    assert {cell.seed for cell in plan.cells} == {1103, 2207, 3301}
    assert {cell.variant_id for cell in plan.cells} == {
        "if-core-v1-reference",
        "if-dual-depth9-v1",
    }
    assert len({cell.cell_id for cell in plan.cells}) == 6
    assert plan.representation_selection_sha256 == selection_sha256
    assert plan.representation_selection_record_digest == selection_record_digest
    assert plan.selection_effect == "diagnostic-only-no-main-grid-reselection"
    assert plan.seed_selection_forbidden is True
    assert plan.training_protocol == registry.training_protocol
    assert all(cell.trainable_parameters > 0 for cell in plan.cells)
    assert all(len(cell.parameter_signature_sha256) == 64 for cell in plan.cells)
    assert "fresh_retraining_required_for_both_variants" in plan.execution_requirements

    with pytest.raises(ValueError, match="only registered when IF-Dual"):
        build_capacity_diagnostic_plan(
            registry,
            selected_simpler="if-mlp",
            representation_selection_sha256=selection_sha256,
            representation_selection_record_digest=selection_record_digest,
        )
    with pytest.raises(ValueError, match="SHA-256"):
        build_capacity_diagnostic_plan(
            registry,
            selected_simpler="if-dual",
            representation_selection_sha256="self-asserted",
            representation_selection_record_digest=selection_record_digest,
        )


def test_registry_rejects_grid_or_registration_drift(tmp_path: Path) -> None:
    grid = json.loads(GRID.read_text())
    grid["stages"]["representation"].pop()
    changed_grid = tmp_path / "grid.json"
    changed_grid.write_text(json.dumps(grid))
    with pytest.raises(ValueError, match="pinned main-grid digest"):
        load_capacity_control_registry(REGISTRY, main_grid_path=changed_grid)

    registration = json.loads(REGISTRY.read_text())
    registration["variants"][1]["expected_trainable_parameters"] += 1
    changed_registry = tmp_path / "capacity.json"
    changed_registry.write_text(json.dumps(registration))
    with pytest.raises(ValueError, match="record digest"):
        load_capacity_control_registry(changed_registry, main_grid_path=GRID)

    registration["record_digest"] = content_digest(
        {key: value for key, value in registration.items() if key != "record_digest"}
    )
    self_rehashed_registry = tmp_path / "self-rehashed-capacity.json"
    self_rehashed_registry.write_text(json.dumps(registration))
    with pytest.raises(ValueError, match="pinned v1 record digest"):
        load_capacity_control_registry(self_rehashed_registry, main_grid_path=GRID)
