"""Label-free heterogeneous-backbone contracts for Quality Value V2."""

from __future__ import annotations

import copy
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest


def _record() -> dict:
    return {
        "instance_id": "hetero-quality-0-m",
        "window_nodes": [0, 1, 2, 3],
        "window_edges": [[0, 1], [1, 2], [2, 3], [1, 3]],
        "frozen": {"1": [10], "2": [11, 12]},
        "frozen_adjacency": {
            "0": [10, 11],
            "1": [10],
            "2": [11, 12],
            "3": [10, 11],
        },
        "neighbours": [1, 2],
        "edge_J": [-1.0, 0.5],
        "neighbour_degree": [2, 3],
        "neighbour_chain_size": [1, 2],
        "neighbour_h": [0.0, -0.25],
        "candidates": [[0], [1, 2], [1, 2, 3]],
        "focus_h": 0.25,
        "p_solve": [0.45, 0.60, 0.72],
        "stage": [1, 2, 2],
        "best_index": 2,
        "resource_index": 0,
        "original_index": 1,
        "source": "minorminer",
        "topology": "chimera",
    }


def test_label_free_hetero_input_needs_no_training_metadata() -> None:
    from embedbench.models_hetero import HeteroModelInput, encode_hetero_input

    record = _record()
    for name in (
        "p_solve",
        "stage",
        "best_index",
        "resource_index",
        "original_index",
        "source",
        "instance_id",
        "topology",
    ):
        record.pop(name)

    model_input = encode_hetero_input(record)

    forbidden = {
        "p",
        "p_solve",
        "stage",
        "best_index",
        "resource_index",
        "original_index",
        "source",
        "instance_id",
        "topology",
    }
    assert forbidden.isdisjoint(field.name for field in fields(HeteroModelInput))
    assert model_input.candidate_masks.shape == (3, 7)
    assert np.count_nonzero(model_input.qubit_features[:, -1]) == 0


def test_quality_v2_hetero_forward_rejects_label_bearing_encoding() -> None:
    pytest.importorskip("torch")
    from embedbench.models_hetero import encode_hetero
    from embedbench.models_quality_v2_hetero import build_quality_v2_hetero_model

    model = build_quality_v2_hetero_model(hidden=8, layers=1, heads=1)

    with pytest.raises(TypeError, match="HeteroModelInput"):
        model(encode_hetero(_record()))


def test_training_encoding_is_explicitly_reduced_to_label_free_input() -> None:
    from embedbench.models_hetero import (
        HeteroModelInput,
        encode_hetero,
        model_input_from_hetero,
    )

    encoded = encode_hetero(_record())
    assert np.count_nonzero(encoded.xq[:, -1]) > 0

    model_input = model_input_from_hetero(encoded)

    assert isinstance(model_input, HeteroModelInput)
    assert np.count_nonzero(model_input.qubit_features[:, -1]) == 0
    assert not hasattr(model_input, "p")
    assert not hasattr(model_input, "stage")


def test_quality_v2_hetero_forward_is_label_and_index_invariant() -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_hetero import encode_hetero_input
    from embedbench.models_quality_v2_hetero import build_quality_v2_hetero_model

    torch.manual_seed(7)
    model = build_quality_v2_hetero_model(hidden=8, layers=1, heads=1)
    model.eval()
    first = model(encode_hetero_input(_record())).quality_logit.detach()

    relabeled = copy.deepcopy(_record())
    relabeled.update(
        p_solve=[0.99, 0.01, 0.50],
        stage=[2, 1, 1],
        best_index=0,
        resource_index=2,
        original_index=-1,
        source="witness",
        instance_id="different-label-row",
        topology="label-metadata-must-not-matter",
    )
    second = model(encode_hetero_input(relabeled)).quality_logit.detach()

    assert torch.equal(first, second)


def test_heterogeneous_quality_v2_uses_nonfocus_hamiltonian_context() -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_hetero import encode_hetero_input
    from embedbench.models_quality_v2_hetero import build_quality_v2_hetero_model

    record = _record()
    record["problem"] = {
        "h": {"0": 0.25, "1": 0.0, "2": -0.25},
        "J": [[0, 1, -1.0], [0, 2, 0.5]],
    }
    record["all_chains"] = copy.deepcopy(record["frozen"])
    changed = copy.deepcopy(record)
    changed["problem"]["h"]["2"] = 3.5
    changed["problem"]["J"].append([1, 2, -0.75])

    torch.manual_seed(31)
    model = build_quality_v2_hetero_model(hidden=8, layers=1, heads=1)
    model.eval()
    first_input = encode_hetero_input(record, require_hamiltonian_context=True)
    second_input = encode_hetero_input(changed, require_hamiltonian_context=True)

    assert not np.array_equal(
        first_input.hamiltonian_context,
        second_input.hamiltonian_context,
    )
    first = model(first_input).quality_logit.detach()
    second = model(second_input).quality_logit.detach()
    assert not torch.equal(first, second)


def test_heterogeneous_quality_encoder_includes_full_logical_problem_graph() -> None:
    from embedbench.models_hetero import encode_hetero_input

    record = _record()
    record["focus"] = 0
    record["frozen"] = {"1": [10]}
    record["all_chains"] = {"1": [10], "2": [11, 12], "3": [13]}
    record["neighbours"] = [1]
    record["edge_J"] = [-1.0]
    record["neighbour_degree"] = [2]
    record["neighbour_chain_size"] = [1]
    record["neighbour_h"] = [0.0]
    record["problem"] = {
        "h": {"0": 0.25, "1": 0.0, "2": -2.0, "3": 1.5},
        "J": [[0, 1, -1.0], [1, 2, 0.75], [2, 3, -0.5]],
    }

    model_input = encode_hetero_input(record, require_hamiltonian_context=True)

    assert model_input.variable_features.shape == (4, 7)
    assert model_input.membership.shape[0] == 4
    assert model_input.logical_adjacency[1, 2] > 0.0
    assert model_input.logical_adjacency[2, 3] < 0.0
    assert model_input.variable_features[2, 5] < 0.0
    assert model_input.variable_features[3, 5] > 0.0


def test_full_heterogeneous_encoder_accepts_zero_weight_structural_neighbour() -> None:
    """Inkdrop records may require a contact absent from the programmed Ising J."""

    from embedbench.models_hetero import encode_hetero_input

    record = _record()
    record["focus"] = 0
    record["all_chains"] = copy.deepcopy(record["frozen"])
    record["edge_J"] = [-1.0, 0.0]
    record["problem"] = {
        "h": {"0": 0.25, "1": 0.0, "2": -0.25},
        "J": [[0, 1, -1.0]],
    }

    model_input = encode_hetero_input(record, require_hamiltonian_context=True)

    assert model_input.logical_adjacency[0, 1] < 0.0
    assert model_input.logical_adjacency[0, 2] == 0.0
    assert any(
        coupling == 0.0
        for candidate_contacts in model_input.candidate_contacts
        for _, _, coupling in candidate_contacts
    )


def test_quality_v2_hetero_emits_registered_heads_with_finite_gradients() -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_hetero import encode_hetero_input
    from embedbench.models_quality_v2 import ROBUSTNESS_NAMES, quality_v2_loss
    from embedbench.models_quality_v2_hetero import build_quality_v2_hetero_model

    model = build_quality_v2_hetero_model(
        hidden=8,
        layers=1,
        heads=1,
        predict_terminal_qubits=True,
    )
    output = model(encode_hetero_input(_record()))

    assert model.config.arch == "hetero"
    assert not hasattr(model, "feasibility_head")
    assert output.quality_logit.shape == (3,)
    assert output.quality_log_concentration.shape == (3,)
    assert output.future_capacity.shape == (3,)
    assert output.robustness.shape == (3, len(ROBUSTNESS_NAMES))
    assert output.terminal_qubits is not None
    assert output.terminal_qubits.shape == (3,)

    losses = quality_v2_loss(
        output,
        p_solve=torch.tensor([0.45, 0.60, 0.72]),
        stage=torch.tensor([1, 2, 2]),
        future_capacity=torch.tensor([0.75, 0.50, 0.25]),
        robustness=torch.tensor([[1.0, 1.0, 2.0, 0.0], [1.0, 1.5, 3.0, 0.0], [2.0, 2.5, 5.0, 1.0]]),
        terminal_qubits=torch.tensor([4.0, 5.0, 6.0]),
    )
    losses["loss"].backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_public_builder_and_checkpoint_loader_dispatch_heterogeneous_v2(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_hetero import encode_hetero_input
    from embedbench.models_quality_v2 import (
        build_quality_v2_model,
        load_quality_v2_model,
        save_quality_v2_model,
    )

    torch.manual_seed(11)
    model_input = encode_hetero_input(_record())
    model = build_quality_v2_model(arch="hetero", hidden=8, layers=1, heads=1)
    model.eval()
    expected = model(model_input).quality_mean.detach()
    path = tmp_path / "hetero-v2.pt"

    save_quality_v2_model(model, path, metadata={"split_sha256": "abc"})
    loaded, metadata = load_quality_v2_model(path)

    actual = loaded(model_input).quality_mean.detach()
    assert torch.equal(expected, actual)
    assert loaded.config.arch == "hetero"
    assert metadata["forward_inputs"] == [
        "qubit_features",
        "variable_features",
        "coupler_features",
        "logical_adjacency",
        "membership",
        "candidate_masks",
        "candidate_contacts",
        "focus_index",
        "focus_h",
        "hamiltonian_context",
    ]

    with pytest.raises(ValueError, match="neighbour_feats"):
        build_quality_v2_model(
            arch="hetero",
            hidden=8,
            layers=1,
            heads=1,
            neighbour_feats=True,
        )


def test_batched_heterogeneous_message_passing_matches_one_graph_at_a_time() -> None:
    torch = pytest.importorskip("torch")
    from embedbench.models_hetero import build_hetero

    torch.manual_seed(23)
    model = build_hetero(hidden=8, layers=2, heads=1)
    batch, n_qubits, n_variables = 3, 5, 4
    xq = torch.randn(batch, n_qubits, 10)
    xv = torch.randn(batch, n_variables, 7)
    edges = torch.zeros(batch, n_qubits, n_qubits, 5)
    for left, right in ((0, 1), (1, 2), (2, 3), (3, 4)):
        edges[:, left, right, 0] = 1.0
        edges[:, right, left, 0] = 1.0
    logical = torch.rand(batch, n_variables, n_variables)
    logical = (logical + logical.transpose(1, 2)) / 2.0
    logical.diagonal(dim1=1, dim2=2).zero_()
    membership = torch.zeros(batch, n_variables, n_qubits)
    membership[:, 0, 0] = 1.0
    membership[:, 1, 1] = 1.0
    membership[:, 2, 2:4] = 1.0
    membership[:, 3, 4] = 1.0

    batched_q, batched_v = model.forward_many(xq, xv, edges, logical, membership)
    sequential = [
        model.forward_one(xq[index], xv[index], edges[index], logical[index], membership[index])
        for index in range(batch)
    ]

    torch.testing.assert_close(batched_q, torch.stack([item[0] for item in sequential]))
    torch.testing.assert_close(batched_v, torch.stack([item[1] for item in sequential]))
