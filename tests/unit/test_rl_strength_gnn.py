from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pytest
import networkx as nx

torch = pytest.importorskip("torch")

from isingfold.rl.strength import (  # noqa: E402
    StrengthGraphInput,
    StrengthRecord,
    StrengthSelectorModel,
    binomial_bce_with_logits,
    fit_selector,
)
from isingfold.embedding import LogicalProblem  # noqa: E402
from isingfold.rl.contracts import Context  # noqa: E402
from isingfold.rl.program import compile_registry, strength_registry  # noqa: E402
from isingfold.rl.strength_tensorize import build_strength_inputs  # noqa: E402


def _packed(rows: int, slots: int, value: float = 0.0) -> np.ndarray:
    array = np.zeros((rows, 2 * slots), dtype=np.float32)
    array[:, :slots] = value
    array[:, slots:] = 1.0
    return array


def _graph(strength_index: int) -> StrengthGraphInput:
    return StrengthGraphInput(
        logical=_packed(2, 20, 0.1),
        hardware=_packed(3, 18, 0.2 + 0.05 * strength_index),
        logical_edges=_packed(2, 6, 0.3),
        hardware_edges=_packed(4, 10, 0.4),
        claims=_packed(3, 6, 0.5),
        globals_=_packed(1, 32, 0.6)[0],
        index_logical_edges=np.asarray([[0, 1], [1, 0]], dtype=np.int64),
        index_hardware_edges=np.asarray([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=np.int64),
        index_claims=np.asarray([[0, 1, 1], [0, 1, 2]], dtype=np.int64),
        strength_index=strength_index,
        strength=(0.5, 1.0, 2.0, 4.0)[strength_index],
        scale=0.9,
    )


def _legacy_features(strength: float) -> dict[str, float]:
    return {
        "strength": strength,
        "scale": 0.9,
        "qubits": 4.0,
        "max_chain": 2.0,
        "mean_chain": 2.0,
        "single_qubit_fraction": 0.0,
        "max_field": 0.5,
        "max_coupling": 1.5,
        "mean_contacts": 1.0,
        "single_contact_fraction": 1.0,
        "strength_over_jmax": strength / 1.25,
        "chain_edges": 2.0,
    }


def test_selector_has_normative_independent_graph_architecture() -> None:
    model = StrengthSelectorModel()

    assert model.width == 128
    assert len(model.local_logical) == 3
    assert len(model.local_hardware) == 3
    assert len(model.ownership_fusion) == 2
    assert model.head[0].in_features == 582
    assert model.head[0].out_features == 128
    assert model.head[2].in_features == 128
    assert model.head[2].out_features == 64
    assert model.head[4].in_features == 64
    assert model.head[4].out_features == 1
    assert not any(
        "conflict" in name or "archive" in name or "action" in name
        for name, _ in model.named_modules()
    )


def test_selector_node_projectors_are_exact_linear_silu_layernorm() -> None:
    model = StrengthSelectorModel()

    for projector, input_width in (
        (model.logical_projector, 2 * 20),
        (model.hardware_projector, 2 * 18),
    ):
        assert [type(layer).__name__ for layer in projector] == [
            "Linear",
            "SiLU",
            "LayerNorm",
        ]
        assert projector[0].in_features == input_width
        assert projector[0].out_features == 128
        assert tuple(projector[2].normalized_shape) == (128,)


def test_four_conditioned_evaluations_share_weights_and_backpropagate() -> None:
    torch.manual_seed(3)
    model = StrengthSelectorModel()
    inputs = tuple(_graph(index) for index in range(4))

    probabilities = model(inputs)

    assert probabilities.shape == (4,)
    assert torch.isfinite(probabilities).all()
    assert torch.all((probabilities >= 0.0) & (probabilities <= 1.0))
    model.logits(inputs).sum().backward()
    assert model.logical_projector[0].weight.grad is not None
    assert model.hardware_projector[0].weight.grad is not None
    assert model.ownership_fusion[0].logical_to_hardware.message[0].weight.grad is not None
    assert model.head[0].weight.grad is not None


def test_deployable_masks_remove_search_age_and_tabu_channels() -> None:
    torch.manual_seed(7)
    model = StrengthSelectorModel()
    baseline = _graph(0)
    logical = baseline.logical.copy()
    hardware = baseline.hardware.copy()
    logical_edges = baseline.logical_edges.copy()
    claims = baseline.claims.copy()
    globals_ = baseline.globals_.copy()
    for array, slots in (
        (logical, (16, 17)),
        (hardware, (15,)),
        (logical_edges, (5,)),
        (claims, (0,)),
    ):
        scalar_count = array.shape[1] // 2
        for slot in slots:
            array[:, slot] = 9_999.0
            array[:, scalar_count + slot] = 1.0
    for slot in tuple(range(8, 12)) + (13,) + tuple(range(23, 32)):
        globals_[slot] = 9_999.0
        globals_[32 + slot] = 1.0
    poisoned = replace(
        baseline,
        logical=logical,
        hardware=hardware,
        logical_edges=logical_edges,
        claims=claims,
        globals_=globals_,
    )

    assert torch.equal(model.logits((baseline,)), model.logits((poisoned,)))


def test_binomial_loss_is_finite_for_zero_and_all_hit_blocks() -> None:
    logits = torch.tensor([-1_000.0, -2.0, 2.0, 1_000.0], requires_grad=True)
    hits = torch.tensor([0, 0, 16, 16])
    reads = torch.tensor([16, 16, 16, 16])

    loss = binomial_bce_with_logits(logits, hits, reads)

    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_freeze_tie_rule_and_serialization_digest(tmp_path) -> None:
    model = StrengthSelectorModel()
    with torch.no_grad():
        for parameter in model.head.parameters():
            parameter.zero_()
    inputs = tuple(_graph(index) for index in range(4))

    model.freeze()
    digest = model.artifact_digest()
    path = tmp_path / "selector.pt"
    model.save(path)
    restored = StrengthSelectorModel.load(path)

    assert model.select_embedding(inputs) == 0
    assert model.frozen
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert restored.frozen
    assert restored.artifact_digest() == digest
    assert torch.equal(restored(inputs), model(inputs))


def test_parameter_compatible_v1_selector_artifact_fails_closed(tmp_path) -> None:
    model = StrengthSelectorModel().freeze()
    path = tmp_path / "old-selector.pt"
    payload = {
        "format": "isingfold.strength-selector.graph.v1",
        "version": "if-q3-s0-graph-selector-1",
        "normalizer_digest": model.normalizer_digest,
        "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "digest": model.artifact_digest(),
        "frozen": True,
    }
    torch.save(payload, path)

    with pytest.raises(ValueError, match="unsupported strength-selector artifact format"):
        StrengthSelectorModel.load(path)

    payload["format"] = StrengthSelectorModel.FORMAT
    torch.save(payload, path)
    with pytest.raises(ValueError, match="unsupported strength-selector artifact version"):
        StrengthSelectorModel.load(path)


def test_graph_input_rejects_non_deployable_or_malformed_channels() -> None:
    graph = _graph(0)
    with pytest.raises(ValueError, match="logical feature width"):
        replace(graph, logical=np.zeros((2, 41), dtype=np.float32)).validate()
    with pytest.raises(ValueError, match="strength_index"):
        replace(graph, strength_index=4).validate()
    with pytest.raises(ValueError, match="claim endpoint"):
        replace(graph, index_claims=np.asarray([[0], [99]], dtype=np.int64)).validate()


def test_graph_record_fit_uses_binomial_targets_and_publishes_frozen_model() -> None:
    graphs = tuple(_graph(index) for index in range(4))
    record = StrengthRecord(
        features=tuple(_legacy_features(graph.strength) for graph in graphs),
        hits=(0, 4, 12, 16),
        reads=(16, 16, 16, 16),
        lineage="train-lineage",
        graph_inputs=graphs,
    )

    model = fit_selector(
        (record,),
        epochs=1,
        graph_learning_rate=1e-4,
        seed=11,
        device=torch.device("cpu"),
    )

    assert model.deployment_ready
    assert model.frozen
    assert next(model.parameters()).device.type == "cpu"
    assert torch.isfinite(model(graphs)).all()
    with pytest.raises(RuntimeError, match="complete graph"):
        model.predict(record.features)


def test_graph_selector_fit_minibatches_records_without_dropping_final_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graphs = tuple(_graph(index) for index in range(4))
    records = tuple(
        StrengthRecord(
            features=tuple(_legacy_features(graph.strength) for graph in graphs),
            hits=(offset, 4 + offset, 8 + offset, 12 + offset),
            reads=(16, 16, 16, 16),
            lineage=f"train-{offset}",
            graph_inputs=graphs,
        )
        for offset in range(5)
    )
    observed_batch_pairs: list[int] = []
    original = StrengthSelectorModel.raw_logits

    def traced(self, inputs):
        observed_batch_pairs.append(len(inputs))
        return original(self, inputs)

    monkeypatch.setattr(StrengthSelectorModel, "raw_logits", traced)
    history: list[dict[str, object]] = []
    fit_selector(
        records,
        epochs=1,
        graph_minibatch=2,
        graph_learning_rate=1e-4,
        seed=17,
        history=history,
    )

    assert observed_batch_pairs == [8, 8, 4]
    assert history[0]["records_seen"] == 5
    assert history[0]["program_pairs_seen"] == 20
    assert history[0]["optimizer_steps"] == 3
    assert history[0]["graph_minibatch_records"] == 2
    assert math.isfinite(float(history[0]["mean_binomial_bce"]))


def test_graph_selector_minibatch_fit_is_seed_deterministic() -> None:
    graphs = tuple(_graph(index) for index in range(4))
    records = tuple(
        StrengthRecord(
            features=tuple(_legacy_features(graph.strength) for graph in graphs),
            hits=(offset, 4 + offset, 8 + offset, 12 + offset),
            reads=(16, 16, 16, 16),
            lineage=f"train-{offset}",
            graph_inputs=graphs,
        )
        for offset in range(3)
    )
    first_history: list[dict[str, object]] = []
    second_history: list[dict[str, object]] = []

    first = fit_selector(
        records,
        epochs=1,
        graph_minibatch=2,
        graph_learning_rate=1e-4,
        seed=23,
        history=first_history,
    )
    second = fit_selector(
        records,
        epochs=1,
        graph_minibatch=2,
        graph_learning_rate=1e-4,
        seed=23,
        history=second_history,
    )

    assert first.artifact_digest() == second.artifact_digest()
    assert first_history == second_history


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_graph_record_fit_moves_model_and_tensorized_inputs_to_cuda() -> None:
    graphs = tuple(_graph(index) for index in range(4))
    record = StrengthRecord(
        features=tuple(_legacy_features(graph.strength) for graph in graphs),
        hits=(0, 4, 12, 16),
        reads=(16, 16, 16, 16),
        lineage="train-lineage",
        graph_inputs=graphs,
    )

    model = fit_selector(
        (record,),
        epochs=1,
        graph_learning_rate=1e-4,
        seed=11,
        device=torch.device("cuda"),
    )

    assert next(model.parameters()).device.type == "cuda"
    assert model(graphs).device.type == "cuda"


@pytest.mark.skipif(torch.cuda.is_available(), reason="test requires a CPU-only runtime")
def test_graph_record_fit_rejects_unavailable_cuda() -> None:
    graphs = tuple(_graph(index) for index in range(4))
    record = StrengthRecord(
        features=tuple(_legacy_features(graph.strength) for graph in graphs),
        hits=(0, 4, 12, 16),
        reads=(16, 16, 16, 16),
        lineage="train-lineage",
        graph_inputs=graphs,
    )

    with pytest.raises(ValueError, match="CUDA.*unavailable"):
        fit_selector((record,), epochs=0, device="cuda")


def test_program_tensorizer_builds_four_actual_program_graphs_and_masks_search_state() -> None:
    logical = nx.Graph([(0, 1)])
    host = nx.path_graph(4)
    problem = LogicalProblem.from_dicts({0: 1.0, 1: -0.5}, {(0, 1): 1.25})
    chains = {0: frozenset((0, 1)), 1: frozenset((2, 3))}
    context = Context(qubit_cap=4)
    strengths = strength_registry(problem, context.strength_ratios)
    programs = compile_registry(
        chains,
        host,
        problem,
        strengths,
        context.field_limit,
        context.coupler_limit,
    )

    inputs = build_strength_inputs(
        ctx=context,
        logical=logical,
        host=host,
        problem=problem,
        chains=chains,
        programs=programs,
    )

    assert len(inputs) == 4
    assert tuple(item.strength_index for item in inputs) == (0, 1, 2, 3)
    assert all(item.validate() is item for item in inputs)
    for item in inputs:
        assert np.all(item.logical[:, [16, 17, 36, 37]] == 0.0)
        assert np.all(item.hardware[:, [15, 33]] == 0.0)
        assert np.all(item.logical_edges[:, [5, 11]] == 0.0)
        assert np.all(item.claims[:, [0, 6]] == 0.0)
        disallowed_global = tuple(range(8, 12)) + (13,) + tuple(range(23, 32))
        assert np.all(item.globals_[list(disallowed_global)] == 0.0)
        assert np.all(item.globals_[[32 + slot for slot in disallowed_global]] == 0.0)
    # Program-coupling channels are recomputed for F_j rather than copied from F_1.
    assert not np.array_equal(inputs[0].hardware_edges, inputs[-1].hardware_edges)
