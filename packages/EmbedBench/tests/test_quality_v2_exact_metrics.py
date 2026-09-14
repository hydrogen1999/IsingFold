"""Adversarial checks for Quality V2's host-verified exact descriptors."""

from __future__ import annotations

import copy

import networkx as nx
import numpy as np
import pytest


def _full_record() -> dict:
    return {
        "instance_id": "exact-context",
        "topology": "chimera",
        "size": 2,
        "focus": 0,
        "n_vars": 3,
        "window_nodes": [0, 1],
        "window_edges": [],
        "all_chains": {"1": [10, 11], "2": [20]},
        "frozen": {"1": [10, 11], "2": [20]},
        # This encoder field is intentionally not trusted for exact feasibility.
        "frozen_adjacency": {"0": [10, 20], "1": [11, 20]},
        "neighbours": [1, 2],
        "candidates": [[0], [1]],
        "Q": [1, 1],
        "l_cap": 3,
        "problem": {
            "h": {"0": 0.0, "1": 0.0, "2": 0.0},
            "J": [[0, 1, -1.0], [0, 2, 0.5], [1, 2, -0.25]],
            "e0": -1.0,
        },
    }


def _full_host() -> nx.Graph:
    host = nx.Graph()
    host.add_nodes_from([0, 1, 10, 11, 20])
    host.add_edges_from(
        [
            (10, 11),  # frozen chain 1
            (11, 20),  # frozen logical edge 1--2
            (0, 10),
            (0, 20),
            (1, 11),
            (1, 20),
        ]
    )
    return host


@pytest.mark.parametrize("corruption", ["off_host", "disconnected"])
def test_exact_metrics_fail_closed_on_invalid_nonfocus_chain(corruption: str) -> None:
    from embedbench.models_quality_v2 import derive_exact_candidate_metrics

    record = _full_record()
    host = _full_host()
    if corruption == "off_host":
        record["all_chains"]["1"] = [10, 999]
    else:
        host.remove_edge(10, 11)

    with pytest.raises(ValueError, match="non-focus chain"):
        derive_exact_candidate_metrics(record, host=host)


def test_exact_metrics_fail_closed_on_unrealised_nonfocus_logical_edge() -> None:
    from embedbench.models_quality_v2 import derive_exact_candidate_metrics

    host = _full_host()
    host.remove_edge(11, 20)

    with pytest.raises(ValueError, match="non-focus logical edge"):
        derive_exact_candidate_metrics(_full_record(), host=host)


@pytest.mark.parametrize("corruption", ["focus_chain", "duplicate_logical_id"])
def test_exact_metrics_reject_ambiguous_all_chains(corruption: str) -> None:
    from embedbench.models_quality_v2 import derive_exact_candidate_metrics

    record = _full_record()
    if corruption == "focus_chain":
        record["all_chains"]["0"] = [0]
    else:
        record["all_chains"][1] = [11]

    with pytest.raises(ValueError, match="all_chains"):
        derive_exact_candidate_metrics(record, host=_full_host())


def test_candidate_feasibility_uses_host_not_claimed_frozen_adjacency() -> None:
    from embedbench.models_quality_v2 import derive_exact_candidate_metrics

    record = _full_record()
    host = _full_host()
    host.remove_edge(1, 20)
    # The serialized encoder feature still falsely claims the missing contact.
    assert 20 in record["frozen_adjacency"]["1"]

    metrics = derive_exact_candidate_metrics(record, host=host)

    np.testing.assert_array_equal(metrics.feasible, [True, False])
    np.testing.assert_array_equal(metrics.contact_counts, [[1, 1], [1, 0]])


def test_focus_structural_neighbour_is_required_even_when_its_j_is_zero() -> None:
    from embedbench.models_quality_v2 import derive_exact_candidate_metrics

    record = _full_record()
    record["problem"]["J"] = [edge for edge in record["problem"]["J"] if edge[:2] != [0, 2]]
    host = _full_host()
    host.remove_edge(1, 20)

    metrics = derive_exact_candidate_metrics(record, host=host)

    np.testing.assert_array_equal(metrics.feasible, [True, False])


def test_exact_metrics_reject_unknown_focus_neighbour() -> None:
    from embedbench.models_quality_v2 import derive_exact_candidate_metrics

    record = _full_record()
    record["neighbours"].append(99)

    with pytest.raises(ValueError, match="absent from all_chains"):
        derive_exact_candidate_metrics(record, host=_full_host())


def test_exact_metrics_do_not_mutate_the_explicit_host_or_record() -> None:
    from embedbench.models_quality_v2 import derive_exact_candidate_metrics

    record = _full_record()
    host = _full_host()
    before_record = copy.deepcopy(record)
    before_nodes = set(host.nodes)
    before_edges = set(host.edges)

    derive_exact_candidate_metrics(record, host=host)

    assert record == before_record
    assert set(host.nodes) == before_nodes
    assert set(host.edges) == before_edges


def test_residual_connectivity_distinguishes_equal_free_count_fragmentation() -> None:
    from embedbench.models_quality_v2 import (
        RESIDUAL_CONNECTIVITY_PROXY,
        derive_exact_candidate_metrics,
    )

    host = nx.Graph()
    host.add_nodes_from([0, 1, 2, 3, 4, 10])
    host.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (0, 10), (2, 10)])
    record = {
        "topology": "chimera",
        "size": 2,
        "focus": 0,
        "n_vars": 2,
        "window_nodes": [0, 1, 2, 3, 4],
        "window_edges": [[0, 1], [1, 2], [2, 3], [3, 4]],
        "all_chains": {"1": [10]},
        "frozen": {"1": [10]},
        "frozen_adjacency": {"0": [10], "2": [10]},
        "neighbours": [1],
        "candidates": [[0], [2]],
        "Q": [1, 1],
        "l_cap": 2,
        "problem": {
            "h": {"0": 0.0, "1": 0.0},
            "J": [[0, 1, -1.0]],
            "e0": -1.0,
        },
    }

    metrics = derive_exact_candidate_metrics(record, host=host)

    np.testing.assert_array_equal(metrics.feasible, [True, True])
    np.testing.assert_allclose(metrics.residual_connectivity, [0.8, 0.4])
    # Backward-compatible spelling exposes the same registered target.
    np.testing.assert_array_equal(metrics.residual_capacity, metrics.residual_connectivity)
    assert RESIDUAL_CONNECTIVITY_PROXY == (
        "largest_remaining_free_component_size_over_original_window_size"
    )


def test_pristine_reconstruction_rejects_explicit_defect_provenance() -> None:
    from embedbench.models_quality_v2 import derive_exact_candidate_metrics

    record = _full_record()
    record["defect_qubits"] = 0.05

    with pytest.raises(ValueError, match="explicit host"):
        derive_exact_candidate_metrics(record)
