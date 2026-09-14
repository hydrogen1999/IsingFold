"""Filesystem trust boundary for the final four-strength audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from isingfold.rl.checkpoint import runtime_implementation_registry
from isingfold.rl.contracts import Context
from isingfold.rl.data.import_embedbench import content_digest
from isingfold.rl.experiment_selection import FrozenRLValueSelection
from isingfold.rl.final_strength_workflow import (
    FinalStrengthWorkflowError,
    authenticate_final_strength_sources,
    frozen_learned_selection_payload,
    load_authenticated_learned_complete_run,
)


def _selection() -> FrozenRLValueSelection:
    runtime = runtime_implementation_registry()
    return FrozenRLValueSelection(
        receipt_sha256="1" * 64,
        record_digest="2" * 64,
        grid_manifest_sha256="3" * 64,
        model_family="if-core",
        grid_model_family="if-core",
        method="ppo-warm-start",
        training_seeds=(1103, 2207, 3301),
        cell_ids=("cell-a", "cell-b", "cell-c"),
        checkpoint_payload_digests=("4" * 64, "5" * 64, "6" * 64),
        representation_selection_sha256="7" * 64,
        representation_selection_record_digest="8" * 64,
        runtime_implementation_registry=runtime,
        runtime_implementation_digest=content_digest(runtime),
        quality_preflight_receipt_sha256="9" * 64,
        quality_preflight_record_digest="a" * 64,
    )


def test_frozen_selection_projection_is_complete_and_seed_selection_free() -> None:
    selection = _selection()
    payload = frozen_learned_selection_payload(selection)

    assert payload["selection_receipt_sha256"] == selection.receipt_sha256
    assert payload["source_cell_ids"] == ["cell-a", "cell-b", "cell-c"]
    assert payload["training_seeds"] == [1103, 2207, 3301]
    assert payload["seed_selection_forbidden"] is True
    assert payload["runtime_implementation_digest"] == content_digest(
        payload["runtime_implementation_registry"]
    )


def test_six_source_loader_rejects_partial_census_before_opening_files() -> None:
    with pytest.raises(FinalStrengthWorkflowError, match="three learned.*three tuned-stock"):
        authenticate_final_strength_sources(
            learned_directories=("missing",),
            learned_report_sha256s=("1" * 64,),
            stock_directories=("missing",) * 3,
            stock_report_sha256s=("2" * 64,) * 3,
            tasks=(),
            context=Context(qubit_cap=4),
        )


def test_learned_loader_requires_external_pin_before_report_parse(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    (root / "report.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(FinalStrengthWorkflowError, match="out-of-band"):
        load_authenticated_learned_complete_run(
            root,
            expected_report_sha256="0" * 64,
            tasks=(),
            context=Context(qubit_cap=4),
        )

