"""Quality Value V2 model, exact-descriptor, selector, and artifact contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parents[1]


def _quality_record(instance_id: str = "problem-train-0-m") -> dict:
    return {
        "instance_id": instance_id,
        "window_nodes": [12, 28, 2, 3],
        "window_edges": [[12, 28], [2, 28], [2, 3], [3, 28]],
        "frozen": {"1": [13], "2": [8, 30]},
        "all_chains": {"1": [13], "2": [8, 30]},
        "frozen_adjacency": {
            "12": [13, 30],
            "28": [13],
            "2": [30],
            "3": [30],
        },
        "neighbours": [1, 2],
        "edge_J": [-1.0, 0.5],
        "neighbour_degree": [2, 3],
        "neighbour_chain_size": [1, 2],
        "candidates": [[12], [28, 2], [28, 2, 3]],
        "Q": [1, 2, 3],
        "p_solve": [0.45, 0.60, 0.72],
        "stage": [1, 2, 2],
        "best_index": 2,
        "resource_index": 0,
        "original_index": 1,
        "source": "minorminer",
        "topology": "pegasus",
        "size": 2,
        "difficulty": "test",
        "l_cap": 4,
        "focus": 0,
        "focus_h": 0.25,
        "n_vars": 3,
        "problem": {
            "h": {"0": 0.25, "1": 0.0, "2": 0.0},
            "J": [[0, 1, -1.0], [0, 2, 0.5]],
            "e0": -1.75,
        },
    }


def _encoded(record: dict | None = None):
    from embedbench.models_chain import encode_chain

    return encode_chain(record or _quality_record())


def _deployment_quality_record(instance_id: str, focus_h: float) -> dict:
    """Small valid Chimera-2 seam whose corpus window is intentionally incomplete."""

    return {
        "instance_id": instance_id,
        "window_nodes": [4, 5, 8, 12],
        "window_edges": [[4, 12], [8, 12]],
        "frozen": {"1": [0], "2": [1]},
        "all_chains": {"1": [0], "2": [1]},
        "frozen_adjacency": {"4": [0, 1], "5": [0, 1]},
        "neighbours": [1, 2],
        "edge_J": [-1.0, 0.5],
        "neighbour_degree": [2, 2],
        "neighbour_chain_size": [1, 1],
        "neighbour_h": [0.0, 0.0],
        "candidates": [[4], [5], [4, 12], [4, 12, 8]],
        "Q": [1, 1, 2, 3],
        "p_solve": [0.40, 0.42, 0.60, 0.72],
        "stage": [1, 1, 2, 2],
        "best_index": 3,
        "resource_index": 0,
        "original_index": 2,
        "source": "minorminer",
        "topology": "chimera",
        "size": 2,
        "difficulty": "test",
        "l_cap": 4,
        "focus": 0,
        "focus_h": focus_h,
        "n_vars": 3,
        "problem": {
            "h": {"0": focus_h, "1": 0.0, "2": 0.0},
            "J": [[0, 1, -1.0], [0, 2, 0.5]],
            "e0": -1.75 - focus_h,
        },
    }


def test_exact_metrics_are_derived_without_mutating_release_record() -> None:
    from embedbench.models_quality_v2 import derive_exact_candidate_metrics

    record = _quality_record()
    before = copy.deepcopy(record)

    metrics = derive_exact_candidate_metrics(record)

    assert record == before
    np.testing.assert_array_equal(metrics.total_qubits, [4, 5, 6])
    np.testing.assert_array_equal(metrics.release_focus_chain_q, [1, 2, 3])
    assert metrics.frozen_total_qubits == 3
    np.testing.assert_array_equal(metrics.max_chain, [2, 2, 3])
    np.testing.assert_array_equal(metrics.contact_counts, [[1, 1], [1, 1], [1, 2]])
    np.testing.assert_array_equal(metrics.cycle_rank, [0, 0, 1])
    np.testing.assert_array_equal(metrics.feasible, [True, True, True])


def test_exact_metrics_reject_overlapping_frozen_chains() -> None:
    from embedbench.models_quality_v2 import derive_exact_candidate_metrics

    record = _quality_record()
    record["all_chains"]["2"] = [13, 30]

    with pytest.raises(ValueError, match="disjoint"):
        derive_exact_candidate_metrics(record)


def test_exact_metrics_fail_on_release_focus_chain_q_mismatch() -> None:
    from embedbench.models_quality_v2 import derive_exact_candidate_metrics

    record = _quality_record()
    record["Q"] = [1, 99, 3]

    with pytest.raises(ValueError, match="release_focus_chain_Q"):
        derive_exact_candidate_metrics(record)


@pytest.mark.parametrize("arch", ["mpnn", "gin", "gatv2", "gps"])
def test_backbones_emit_all_registered_candidate_heads_with_finite_gradients(
    arch: str,
) -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_quality_v2 import (
        ROBUSTNESS_NAMES,
        build_quality_v2_model,
        quality_v2_loss,
        tensors_from_chain,
    )

    encoded = _encoded()
    model = build_quality_v2_model(
        arch=arch,
        hidden=16,
        layers=1,
        heads=4,
        predict_terminal_qubits=True,
    )
    output = model(*tensors_from_chain(encoded))

    assert output.quality_logit.shape == (3,)
    assert output.quality_log_concentration.shape == (3,)
    assert output.quality_mean.shape == (3,)
    assert output.quality_concentration.shape == (3,)
    assert output.future_capacity.shape == (3,)
    assert output.robustness.shape == (3, len(ROBUSTNESS_NAMES))
    assert output.terminal_qubits is not None
    assert output.terminal_qubits.shape == (3,)

    losses = quality_v2_loss(
        output,
        p_solve=torch.tensor(encoded.p),
        stage=torch.tensor(encoded.stage),
        future_capacity=torch.tensor([0.75, 0.50, 0.25]),
        robustness=torch.tensor([[1.0, 1.0, 2.0, 0.0], [1.0, 1.5, 3.0, 0.0], [2.0, 2.5, 5.0, 1.0]]),
        terminal_qubits=torch.tensor([4.0, 5.0, 6.0]),
    )
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_auxiliary_losses_are_zero_when_every_auxiliary_target_is_masked() -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_quality_v2 import (
        build_quality_v2_model,
        quality_v2_loss,
        tensors_from_chain,
    )

    encoded = _encoded()
    model = build_quality_v2_model(hidden=16, layers=1)
    output = model(*tensors_from_chain(encoded))
    losses = quality_v2_loss(
        output,
        p_solve=torch.tensor(encoded.p),
        stage=torch.tensor(encoded.stage),
        future_capacity=torch.full((3,), float("nan")),
        future_capacity_mask=torch.zeros(3, dtype=torch.bool),
        robustness=torch.full((3, 4), float("nan")),
        robustness_mask=torch.zeros((3, 4), dtype=torch.bool),
    )
    losses["loss"].backward()

    assert losses["future_capacity"].item() == 0.0
    assert losses["robustness"].item() == 0.0
    assert losses["terminal_qubits"].item() == 0.0
    assert losses["quality_probability"].item() > 0.0
    assert losses["within_state_rank"].item() > 0.0
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_mixed_missing_auxiliary_targets_have_finite_losses_and_gradients() -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_quality_v2 import QualityV2Output, quality_v2_loss

    future_capacity = torch.tensor([0.25, 0.50, 0.75], requires_grad=True)
    robustness = torch.tensor(
        [[0.5, 1.0, 1.5, 2.0], [1.0, 1.5, 2.0, 2.5], [1.5, 2.0, 2.5, 3.0]],
        requires_grad=True,
    )
    robustness_target = torch.tensor(
        [
            [1.0, float("nan"), 2.0, 0.0],
            [float("nan"), 1.5, 3.0, float("nan")],
            [2.0, 2.5, float("nan"), 1.0],
        ]
    )
    terminal_qubits = torch.tensor([3.0, 4.0, 5.0], requires_grad=True)
    output = QualityV2Output(
        quality_logit=torch.zeros(3, requires_grad=True),
        quality_log_concentration=torch.zeros(3, requires_grad=True),
        future_capacity=future_capacity,
        robustness=robustness,
        terminal_qubits=terminal_qubits,
    )

    losses = quality_v2_loss(
        output,
        p_solve=torch.tensor([0.45, 0.60, 0.72]),
        stage=torch.tensor([1, 2, 2]),
        future_capacity=torch.tensor([0.75, float("nan"), 0.25]),
        robustness=robustness_target,
        terminal_qubits=torch.tensor([4.0, float("nan"), 6.0]),
    )
    losses["loss"].backward()

    assert all(torch.isfinite(losses[name]) for name in losses)
    assert future_capacity.grad is not None
    assert robustness.grad is not None
    assert terminal_qubits.grad is not None
    assert torch.isfinite(future_capacity.grad).all()
    assert torch.isfinite(robustness.grad).all()
    assert torch.isfinite(terminal_qubits.grad).all()
    assert future_capacity.grad[1].item() == 0.0
    assert robustness.grad[~torch.isfinite(robustness_target)].eq(0.0).all()
    assert terminal_qubits.grad[1].item() == 0.0


def test_masked_missing_quality_label_cannot_create_nan_loss_or_gradient() -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_quality_v2 import (
        build_quality_v2_model,
        quality_v2_loss,
        tensors_from_chain,
    )

    model = build_quality_v2_model(hidden=16, layers=1)
    output = model(*tensors_from_chain(_encoded()))
    losses = quality_v2_loss(
        output,
        p_solve=torch.tensor([0.45, float("nan"), 0.72]),
        p_solve_mask=torch.tensor([True, False, True]),
        stage=torch.tensor([1, 2, 2]),
    )
    losses["loss"].backward()

    assert torch.isfinite(losses["loss"])
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_ranking_uses_stricter_margin_when_either_label_is_stage1() -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_quality_v2 import QualityV2Output, quality_v2_loss

    def output() -> QualityV2Output:
        return QualityV2Output(
            quality_logit=torch.tensor([0.0, 0.0], requires_grad=True),
            quality_log_concentration=torch.zeros(2, requires_grad=True),
            future_capacity=torch.zeros(2),
            robustness=torch.zeros(2, 4),
            terminal_qubits=None,
        )

    stage2_noise = quality_v2_loss(
        output(), p_solve=torch.tensor([0.50, 0.54]), stage=torch.tensor([2, 2])
    )
    stage1_noise = quality_v2_loss(
        output(), p_solve=torch.tensor([0.50, 0.59]), stage=torch.tensor([2, 1])
    )
    stage1_signal = quality_v2_loss(
        output(), p_solve=torch.tensor([0.50, 0.61]), stage=torch.tensor([2, 1])
    )

    assert stage2_noise["within_state_rank"].item() == 0.0
    assert stage1_noise["within_state_rank"].item() == 0.0
    assert stage1_signal["within_state_rank"].item() > 0.0


def test_zero_rank_margin_never_creates_diagonal_self_pairs() -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_quality_v2 import QualityV2Output, quality_v2_loss

    output = QualityV2Output(
        quality_logit=torch.tensor([0.0, 0.0], requires_grad=True),
        quality_log_concentration=torch.zeros(2, requires_grad=True),
        future_capacity=torch.zeros(2),
        robustness=torch.zeros(2, 4),
        terminal_qubits=None,
    )
    losses = quality_v2_loss(
        output,
        p_solve=torch.tensor([0.5, 0.5]),
        stage=torch.tensor([2, 2]),
        rank_margin=0.0,
        stage1_rank_margin=0.0,
    )

    assert losses["within_state_rank"].item() == 0.0


def test_forward_is_invariant_to_every_label_and_index_field() -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_quality_v2 import build_quality_v2_model, tensors_from_chain

    torch.manual_seed(7)
    record = _quality_record()
    model = build_quality_v2_model(hidden=16, layers=1)
    model.eval()
    first = model(*tensors_from_chain(_encoded(record))).quality_logit.detach()

    relabeled = copy.deepcopy(record)
    relabeled.update(
        p_solve=[0.99, 0.01, 0.50],
        stage=[2, 1, 1],
        best_index=0,
        resource_index=2,
        original_index=-1,
    )
    second = model(*tensors_from_chain(_encoded(relabeled))).quality_logit.detach()

    assert torch.equal(first, second)


def test_chain_quality_v2_is_conditioned_on_nonfocus_hamiltonian_context() -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_quality_v2 import build_quality_v2_model, tensors_from_chain

    torch.manual_seed(29)
    model = build_quality_v2_model(hidden=16, layers=1)
    model.eval()
    record = _quality_record()
    first_encoded = _encoded(record)
    changed = copy.deepcopy(record)
    changed["problem"]["h"]["2"] = 4.0
    changed["problem"]["J"].append([1, 2, -0.75])
    second_encoded = _encoded(changed)

    assert not np.array_equal(
        first_encoded.hamiltonian_context,
        second_encoded.hamiltonian_context,
    )
    first = model(*tensors_from_chain(first_encoded)).quality_logit.detach()
    second = model(*tensors_from_chain(second_encoded)).quality_logit.detach()
    assert not torch.equal(first, second)


def test_chain_quality_encoder_can_require_full_problem_context() -> None:
    from embedbench.models_chain import encode_chain

    record = _quality_record()
    record.pop("problem")

    with pytest.raises(ValueError, match="record requires problem"):
        encode_chain(record, require_hamiltonian_context=True)


def test_selector_enforces_budget_feasibility_ties_and_no_survivor() -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_quality_v2 import (
        QualityV2Output,
        derive_exact_candidate_metrics,
        select_quality_candidate,
    )

    metrics = derive_exact_candidate_metrics(_quality_record())
    output = QualityV2Output(
        quality_logit=torch.tensor([0.0, 2.0, 4.0]),
        quality_log_concentration=torch.zeros(3),
        future_capacity=torch.zeros(3),
        robustness=torch.zeros(3, 4),
        terminal_qubits=None,
    )
    assert select_quality_candidate(output, metrics, budget=5).index == 1

    tied = QualityV2Output(
        quality_logit=torch.tensor([2.0, 2.0, 4.0]),
        quality_log_concentration=torch.zeros(3),
        future_capacity=torch.zeros(3),
        robustness=torch.zeros(3, 4),
        terminal_qubits=None,
    )
    assert select_quality_candidate(tied, metrics, budget=5).index == 0
    none = select_quality_candidate(tied, metrics, budget=3)
    assert none.index is None
    assert none.reason == "no_feasible_candidate_within_budget"

    disconnected = _quality_record()
    disconnected["candidates"][1] = [12, 2]
    disconnected_metrics = derive_exact_candidate_metrics(disconnected)
    assert disconnected_metrics.feasible.tolist() == [True, False, True]
    assert select_quality_candidate(output, disconnected_metrics, budget=5).index == 0


def test_selector_fails_closed_on_nonfinite_quality_predictions() -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_quality_v2 import (
        QualityV2Output,
        derive_exact_candidate_metrics,
        select_quality_candidate,
    )

    metrics = derive_exact_candidate_metrics(_quality_record())
    output = QualityV2Output(
        quality_logit=torch.tensor([0.0, float("nan"), 4.0]),
        quality_log_concentration=torch.zeros(3),
        future_capacity=torch.zeros(3),
        robustness=torch.zeros(3, 4),
        terminal_qubits=None,
    )

    with pytest.raises(ValueError, match="finite"):
        select_quality_candidate(output, metrics, budget=5)


def test_lcb_can_prefer_a_more_certain_candidate() -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_quality_v2 import (
        QualityV2Output,
        derive_exact_candidate_metrics,
        select_quality_candidate,
    )

    metrics = derive_exact_candidate_metrics(_quality_record())
    output = QualityV2Output(
        quality_logit=torch.tensor([1.5, 1.3, -4.0]),
        quality_log_concentration=torch.tensor([-5.0, 5.0, 0.0]),
        future_capacity=torch.zeros(3),
        robustness=torch.zeros(3, 4),
        terminal_qubits=None,
    )

    assert select_quality_candidate(output, metrics, budget=5, statistic="mean").index == 0
    assert (
        select_quality_candidate(output, metrics, budget=5, statistic="lcb", lcb_z=1.0).index == 1
    )


def test_auxiliary_predictions_cannot_change_quality_selector_formula() -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_quality_v2 import (
        QualityV2Output,
        derive_exact_candidate_metrics,
        select_quality_candidate,
    )

    metrics = derive_exact_candidate_metrics(_quality_record())
    common = {
        "quality_logit": torch.tensor([0.0, 2.0, 4.0]),
        "quality_log_concentration": torch.zeros(3),
        "terminal_qubits": None,
    }
    low_auxiliary = QualityV2Output(
        **common,
        future_capacity=torch.zeros(3),
        robustness=torch.zeros(3, 4),
    )
    high_auxiliary = QualityV2Output(
        **common,
        future_capacity=torch.tensor([1e6, -1e6, 1e6]),
        robustness=torch.full((3, 4), -1e6),
    )

    assert select_quality_candidate(low_auxiliary, metrics, budget=5).index == 1
    assert select_quality_candidate(high_auxiliary, metrics, budget=5).index == 1


def test_budget_sweep_masks_higher_scored_over_budget_candidate() -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_quality_v2 import (
        QualityV2Output,
        derive_exact_candidate_metrics,
        evaluate_quality_v2_budgets,
    )

    encoded = _encoded()
    metrics = derive_exact_candidate_metrics(_quality_record())
    output = QualityV2Output(
        quality_logit=torch.tensor([0.0, 2.0, 8.0]),
        quality_log_concentration=torch.zeros(3),
        future_capacity=torch.zeros(3),
        robustness=torch.zeros(3, 4),
        terminal_qubits=None,
    )

    result = evaluate_quality_v2_budgets(
        [output],
        [encoded],
        [metrics],
        budget_ratios=(1.0, None),
    )

    one_x = result["budgets"]["1.00x"]
    uncapped = result["budgets"]["uncapped"]
    assert one_x["reference_total_qubits"] == [5]
    assert one_x["budgets"] == [5]
    assert one_x["selectors"]["mean"]["selection_indices"] == [1]
    assert uncapped["selectors"]["mean"]["selection_indices"] == [2]
    assert result["candidate_support"] == "full"
    assert result["support_uses_labels"] is False
    assert result["primary_metric_name"] == "mean_finite_budget_regret"


def test_primary_budget_sweep_uses_only_valid_paired_minorminer_references() -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_chain import encode_chain
    from embedbench.models_quality_v2 import (
        QualityV2Output,
        derive_exact_candidate_metrics,
        evaluate_quality_v2_budgets,
    )

    valid_minorminer = _quality_record("problem-valid-mm")
    witness = _quality_record("problem-witness")
    witness["source"] = "witness"
    missing_original = _quality_record("problem-missing-mm")
    missing_original["original_index"] = -1
    infeasible_original = _quality_record("problem-infeasible-mm")
    infeasible_original["candidates"][1] = [12, 2]
    records = [valid_minorminer, witness, missing_original, infeasible_original]
    encoded = [encode_chain(record) for record in records]
    metrics = [derive_exact_candidate_metrics(record) for record in records]
    outputs = [
        QualityV2Output(
            quality_logit=torch.tensor([2.0, 0.0, 8.0]),
            quality_log_concentration=torch.zeros(3),
            future_capacity=torch.zeros(3),
            robustness=torch.zeros(3, 4),
            terminal_qubits=None,
        )
        for _ in records
    ]

    result = evaluate_quality_v2_budgets(
        outputs,
        encoded,
        metrics,
        budget_ratios=(1.0,),
    )

    assert result["budget_contract"] == "B/Q_MM"
    assert result["budget_reference"] == "stock_minorminer_original_total_qubits"
    coverage = result["primary_record_coverage"]
    assert coverage["input_records"] == 4
    assert coverage["included_records"] == 1
    assert coverage["excluded_records"] == 3
    assert coverage["coverage_rate"] == pytest.approx(0.25)
    assert coverage["included_input_indices"] == [0]
    assert coverage["excluded_input_indices"] == [1, 2, 3]
    assert coverage["exclusion_reasons"] == {
        "non_minorminer_source": 1,
        "missing_original_reference": 1,
        "infeasible_original_reference": 1,
    }

    primary = result["budgets"]["1.00x"]
    assert primary["n_records"] == 1
    assert primary["reference_policy"] == "stock_minorminer_original_total_qubits"
    assert primary["reference_sources"] == {"stock_minorminer_original": 1}
    assert primary["input_record_indices"] == [0]
    assert primary["reference_indices"] == [1]
    assert primary["reference_total_qubits"] == [5]
    assert primary["selectors"]["mean"]["regret"] == pytest.approx(0.15)
    assert result["primary_metric"] == pytest.approx(0.15)

    diagnostic = result["diagnostic_resource_relative_sweep"]
    assert diagnostic["diagnostic_only"] is True
    assert diagnostic["included_in_primary_metric"] is False
    assert diagnostic["budget_reference"] == "registered_resource_candidate_total_qubits"
    assert diagnostic["budgets"]["1.00x"]["n_records"] == 4
    assert diagnostic["budgets"]["1.00x"]["input_record_indices"] == [0, 1, 2, 3]
    assert diagnostic["budgets"]["1.00x"]["reference_indices"] == [0, 0, 0, 0]
    assert diagnostic["budgets"]["1.00x"]["selectors"]["mean"]["regret"] == 0.0


def test_diagnostic_resource_sweep_uses_registered_internal_coupler_tiebreak() -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_chain import encode_chain
    from embedbench.models_quality_v2 import (
        QualityV2Output,
        derive_exact_candidate_metrics,
        evaluate_quality_v2_budgets,
    )

    record = _quality_record()
    # Candidate 0 is a triangle (three internal couplers); candidate 1 is a path (two).
    # Both use three focus-chain qubits, so the release resource policy selects index 1.
    record["candidates"] = [[28, 2, 3], [12, 28, 2]]
    record["Q"] = [3, 3]
    record["p_solve"] = [0.8, 0.6]
    record["stage"] = [2, 2]
    record["best_index"] = 0
    record["resource_index"] = 1
    record["original_index"] = 0
    record["source"] = "witness"
    encoded = encode_chain(record)
    metrics = derive_exact_candidate_metrics(record)
    output = QualityV2Output(
        quality_logit=torch.tensor([2.0, 0.0]),
        quality_log_concentration=torch.zeros(2),
        future_capacity=torch.zeros(2),
        robustness=torch.zeros(2, 4),
        terminal_qubits=None,
    )

    result = evaluate_quality_v2_budgets([output], [encoded], [metrics], budget_ratios=(1.0,))

    assert result["budgets"]["1.00x"]["n_records"] == 0
    budget = result["diagnostic_resource_relative_sweep"]["budgets"]["1.00x"]
    assert budget["reference_indices"] == [1]
    assert budget["selectors"]["resource"]["selection_indices"] == [1]


def test_cpu_checkpoint_round_trip_preserves_predictions_and_contract(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_quality_v2 import (
        build_quality_v2_model,
        load_quality_v2_model,
        save_quality_v2_model,
        tensors_from_chain,
    )

    torch.manual_seed(3)
    tensors = tensors_from_chain(_encoded())
    model = build_quality_v2_model(arch="gin", hidden=16, layers=1)
    model.eval()
    expected = model(*tensors).quality_mean.detach()
    path = tmp_path / "model.pt"

    save_quality_v2_model(model, path, metadata={"split_sha256": "abc"})
    loaded, metadata = load_quality_v2_model(path)

    actual = loaded(*tensors).quality_mean.detach()
    assert torch.equal(expected, actual)
    assert metadata["artifact_schema"] == "embedbench.quality-value-v2"
    assert metadata["split_sha256"] == "abc"
    assert all(tensor.device.type == "cpu" for tensor in loaded.state_dict().values())


def test_trainer_keeps_fixed_test_locked_and_emits_one_row_per_seed(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    records = [
        _quality_record("problem-train-0-m"),
        _quality_record("problem-val-1-m"),
        _quality_record("problem-test-2-m"),
    ]
    records[1]["problem"]["h"]["0"] = 1.25
    records[2]["problem"]["h"]["0"] = 2.25
    corpus = tmp_path / "quality.jsonl"
    corpus.write_text("".join(json.dumps(record) + "\n" for record in records))
    import hashlib

    assignments = {
        record["instance_id"]: split
        for record, split in zip(records, ("train", "val", "test"), strict=True)
    }
    manifest = tmp_path / "splits.json"
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
        )
    )
    output = tmp_path / "result.json"
    checkpoint = tmp_path / "checkpoints" / "quality_v2"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_quality_v2.py"),
            str(corpus),
            "--splits",
            str(manifest),
            "--arch",
            "mpnn",
            "--objective-variant",
            "full",
            "--seeds",
            "0,1",
            "--epochs",
            "1",
            "--hidden",
            "8",
            "--layers",
            "1",
            "--device",
            "cpu",
            "--evaluation-support",
            "full",
            "--save",
            str(checkpoint),
            "--out",
            str(output),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    artifact = json.loads(output.read_text())
    assert len(artifact["results"]) == 2
    assert {row["seed"] for row in artifact["results"]} == {0, 1}
    assert all(row["test_evaluated"] is False for row in artifact["results"])
    assert all(row["test_partition_encoded"] is False for row in artifact["results"])
    assert all(
        row["training_scope"] == {"limit": None, "full_fixed_train_validation": True}
        for row in artifact["results"]
    )
    assert all(row["test"] is None for row in artifact["results"])
    assert all(
        row["primary_validation_metric"] == "mean_finite_budget_regret"
        for row in artifact["results"]
    )
    assert all(
        row["validation"]["budget_sweep"]["budget_ratios"] == [1.0, 1.1, 1.25, 1.5, None]
        for row in artifact["results"]
    )
    assert all(
        row["ranking_thresholds"] == {"stage2_pair": 0.05, "stage1_involved_pair": 0.10}
        for row in artifact["results"]
    )
    assert (tmp_path / "checkpoints" / "quality_v2_mpnn_s0.pt").exists()
    assert (tmp_path / "checkpoints" / "quality_v2_mpnn_s1.pt").exists()


def test_hetero_trainer_uses_deploy_view_records_contract_and_keeps_test_unencoded(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    from embedbench.models_quality_v2 import load_quality_v2_model

    records = [
        _deployment_quality_record("problem-train-0-m", 0.25),
        _deployment_quality_record("problem-val-1-m", 1.25),
        _deployment_quality_record("problem-test-2-m", 2.25),
    ]
    # A locked test record must never reach deployment-window reconstruction or encoding.
    records[2]["topology"] = "invalid-locked-test-topology"
    corpus = tmp_path / "quality_chimera2.jsonl"
    corpus.write_text("".join(json.dumps(record) + "\n" for record in records))
    corpus_sha256 = hashlib.sha256(corpus.read_bytes()).hexdigest()
    Path(f"{corpus}.manifest.json").write_text(
        json.dumps(
            {
                "config": {
                    "topology": "chimera",
                    "size": 2,
                    "defect_qubits": 0.0,
                    "defect_couplers": 0.0,
                },
                "sha256": corpus_sha256,
            }
        )
    )
    assignments = {
        record["instance_id"]: split
        for record, split in zip(records, ("train", "val", "test"), strict=True)
    }
    split_manifest = tmp_path / "splits.json"
    split_manifest.write_text(
        json.dumps(
            {
                "schema": "embedbench.split-manifest",
                "schema_version": 2,
                "provenance": {"inputs": [{"file": corpus.name, "sha256": corpus_sha256}]},
                "splits": {corpus.name: assignments},
            }
        )
    )
    output = tmp_path / "hetero-result.json"
    checkpoint = tmp_path / "checkpoints" / "quality_v2"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "train_quality_v2.py"),
            str(corpus),
            "--splits",
            str(split_manifest),
            "--arch",
            "hetero",
            "--seeds",
            "0",
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
            "--deploy-view",
            "--save",
            str(checkpoint),
            "--out",
            str(output),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text())
    row = artifact["results"][0]
    assert row["arch"] == "hetero"
    assert row["test_evaluated"] is False
    assert row["test_partition_encoded"] is False
    from embedbench.hamiltonian_context import hamiltonian_context_contract

    assert row["preprocessing"] == {
        "deploy_view": True,
        "deploy_max_free": 8,
        "neighbour_feats": False,
        "encoder": "heterogeneous",
        "hamiltonian_context": hamiltonian_context_contract(),
    }
    assert artifact["preprocessing"] == row["preprocessing"]
    assert artifact["deployment_host_contract"]["mode"] == ("pristine_topology_reconstruction")
    assert (
        artifact["deployment_host_contract"]["source_manifests"][0]["corpus_sha256"]
        == corpus_sha256
    )

    checkpoint_path = tmp_path / "checkpoints" / "quality_v2_hetero_s0.pt"
    loaded, metadata = load_quality_v2_model(checkpoint_path)
    assert loaded.config.arch == "hetero"
    assert isinstance(row["training_run_id"], str)
    assert len(row["training_run_id"]) == 32
    assert metadata["training_run_id"] == row["training_run_id"]
    assert (
        metadata["selected_validation_metric"]
        == row["validation"]["budget_sweep"]["primary_metric"]
    )
    assert metadata["best_epoch"] == row["best_epoch"]
    assert metadata["training_scope"] == {
        "limit": None,
        "full_fixed_train_validation": True,
    }
    assert metadata["preprocessing"] == row["preprocessing"]
    assert metadata["deployment_host_contract"] == artifact["deployment_host_contract"]


def test_deploy_view_rejects_missing_or_defective_generator_manifest(
    tmp_path: Path,
) -> None:
    pytest.importorskip("torch")
    record = _deployment_quality_record("problem-train-0-m", 0.25)
    corpus = tmp_path / "quality.jsonl"
    corpus.write_text(json.dumps(record) + "\n")
    corpus_sha256 = hashlib.sha256(corpus.read_bytes()).hexdigest()
    split_manifest = tmp_path / "splits.json"
    split_manifest.write_text(
        json.dumps(
            {
                "schema": "embedbench.split-manifest",
                "schema_version": 2,
                "provenance": {"inputs": [{"file": corpus.name, "sha256": corpus_sha256}]},
                "splits": {corpus.name: {record["instance_id"]: "train"}},
            }
        )
    )

    base_command = [
        sys.executable,
        str(ROOT / "scripts" / "train_quality_v2.py"),
        str(corpus),
        "--splits",
        str(split_manifest),
        "--epochs",
        "1",
        "--deploy-view",
        "--out",
        str(tmp_path / "result.json"),
    ]
    missing = subprocess.run(base_command, cwd=ROOT, capture_output=True, text=True)
    assert missing.returncode != 0
    assert "generator manifest" in missing.stderr

    Path(f"{corpus}.manifest.json").write_text(
        json.dumps(
            {
                "config": {"defect_qubits": 0.01, "defect_couplers": 0.0},
                "sha256": corpus_sha256,
            }
        )
    )
    defective = subprocess.run(base_command, cwd=ROOT, capture_output=True, text=True)
    assert defective.returncode != 0
    assert "defect-free" in defective.stderr
