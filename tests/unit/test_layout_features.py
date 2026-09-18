"""Representation regressions for all-free root policies, independent of any learner."""
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from candidate_features import FeatureContext, WIDTH as BASE_WIDTH
from layout_features import (
    FEATURE_INDEX, FEATURE_NAMES, FEATURE_VERSION, WIDTH,
    LayoutFeatureContext, _residual_capacities,
)


def task(host=None, logical=None, h=None, j=None):
    return SimpleNamespace(name="toy", host=host if host is not None else nx.path_graph(7),
                           logical=logical if logical is not None else nx.Graph([(0, 1), (0, 2)]),
                           problem=SimpleNamespace(h=h or {}, j=j or {}))


def value(vector, name):
    return vector[FEATURE_INDEX[name]]


@pytest.mark.parametrize("opcode,qubits", [("PLACE", [1]), ("ROUTE", [1, 2]), ("COMMIT", [])])
def test_preserves_every_original_channel_and_declares_new_schema(opcode, qubits):
    t = task(h={0: -2.0}, j={(0, 1): 2.0, (0, 2): -4.0})
    state = {1: {0}, 2: {6}}
    legacy = FeatureContext(t, 5).pair(0, qubits, state, opcode)
    current = LayoutFeatureContext(t, 5).pair(0, qubits, state, opcode)
    assert FEATURE_VERSION == "layout-v2"
    assert WIDTH == BASE_WIDTH + len(FEATURE_NAMES) == 65
    assert current.dtype == np.float32
    assert current.shape == (WIDTH,)
    np.testing.assert_array_equal(current[:BASE_WIDTH], legacy)
    if opcode != "PLACE":
        assert not current[BASE_WIDTH:].any()


def test_distinguishes_identical_local_features_with_different_remaining_capacity():
    t = task(host=nx.path_graph(25), logical=nx.Graph([(0, 1)]))
    old, new = FeatureContext(t), LayoutFeatureContext(t)
    close_block, far_block = {1: {5}}, {1: {20}}
    np.testing.assert_array_equal(old.pair(0, [0], close_block), old.pair(0, [0], far_block))
    a, b = new.pair(0, [0], close_block), new.pair(0, [0], far_block)
    assert value(a, "component_host_fraction") == pytest.approx(5 / 25)
    assert value(b, "component_host_fraction") == pytest.approx(20 / 25)
    assert value(a, "largest_branch_host_fraction") == pytest.approx(4 / 25)
    assert value(b, "largest_branch_host_fraction") == pytest.approx(19 / 25)


@pytest.mark.parametrize("seed", range(8))
def test_cached_branch_capacities_match_actual_reachability_after_removing_anchor(seed):
    graph = nx.gnp_random_graph(16, 0.15, seed=seed)
    adjacency = {q: tuple(graph.neighbors(q)) for q in graph}
    components, branches, articulation = _residual_capacities(adjacency)
    expected_articulation = set(nx.articulation_points(graph))
    for q in graph:
        assert components[q] == len(nx.node_connected_component(graph, q))
        assert articulation[q] == (q in expected_articulation)
        residual = graph.subgraph(set(graph) - {q})
        for neighbour in graph.neighbors(q):
            assert branches[q][neighbour] == len(nx.node_connected_component(residual, neighbour))


def test_branch_walk_does_not_fail_on_deep_residual_corridors():
    graph = nx.path_graph(2000)
    _, branches, articulation = _residual_capacities({q: tuple(graph.neighbors(q)) for q in graph})
    assert branches[1000] == {999: 1000, 1001: 999}
    assert articulation[1000] and not articulation[0]


def test_direction_channels_report_onward_capacity_not_sum_over_cyclic_paths():
    ctx = LayoutFeatureContext(task(host=nx.path_graph(5), logical=nx.empty_graph(1)))
    # Synthetic coordinate convention, explicitly supplied only for this toy.
    ctx.coords = {q: [q / 4.0, 0.0, 0.0, 0.0, 0.0] for q in ctx.host}
    row = ctx.pair(0, [2], {})
    assert value(row, "coord_0_negative_branch_fraction") == pytest.approx(2 / 5)
    assert value(row, "coord_0_positive_branch_fraction") == pytest.approx(2 / 5)
    assert value(row, "fragmentation_after_anchor") == pytest.approx(0.5)
    cycle = LayoutFeatureContext(task(host=nx.cycle_graph(5), logical=nx.empty_graph(1)))
    row = cycle.pair(0, [0], {})
    assert value(row, "largest_branch_host_fraction") == pytest.approx(4 / 5)
    assert value(row, "fragmentation_after_anchor") == 0


def test_swapping_strong_coupling_changes_preferred_direction_without_changing_legacy_inputs():
    left = task(j={(0, 1): 4.0, (0, 2): 1.0})
    right = task(j={(0, 1): 1.0, (0, 2): 4.0})
    chains = {1: {0}, 2: {6}}
    a, b = LayoutFeatureContext(left), LayoutFeatureContext(right)
    left_root_a, left_root_b = a.pair(0, [1], chains), b.pair(0, [1], chains)
    np.testing.assert_array_equal(left_root_a[:BASE_WIDTH], left_root_b[:BASE_WIDTH])
    assert value(left_root_a, "coupling_weighted_contact") == pytest.approx(0.8)
    assert value(left_root_b, "coupling_weighted_contact") == pytest.approx(0.2)
    assert value(left_root_a, "coupling_weighted_distance") < value(a.pair(0, [5], chains), "coupling_weighted_distance")
    assert value(left_root_b, "coupling_weighted_distance") > value(b.pair(0, [5], chains), "coupling_weighted_distance")


def test_routing_distance_cannot_cross_another_occupied_chain():
    logical = nx.Graph([(0, 1), (0, 2)])
    logical.add_node(3)
    ctx = LayoutFeatureContext(task(logical=logical, j={(0, 1): 4.0, (0, 2): 1.0}))
    row = ctx.pair(0, [1], {1: {0}, 2: {6}, 3: {3}})
    assert value(row, "coupling_weighted_unreachable") == pytest.approx(0.2)
    assert 1 not in ctx._distances[2]


def test_signed_field_and_coupling_information_is_retained():
    positive = LayoutFeatureContext(task(h={0: 2.0}, j={(0, 1): 4.0, (0, 2): 1.0}))
    negative = LayoutFeatureContext(task(h={0: -2.0}, j={(0, 1): -4.0, (0, 2): -1.0}))
    chains = {1: {0}, 2: {6}}
    a, b = positive.pair(0, [1], chains), negative.pair(0, [1], chains)
    np.testing.assert_array_equal(a[:BASE_WIDTH], b[:BASE_WIDTH])
    for feature in ("signed_field", "signed_coupling_proximity", "signed_coupling_contact"):
        assert value(a, feature) == -value(b, feature)


def test_state_cache_reuses_same_membership_and_rebuilds_after_same_length_rewrite():
    ctx = LayoutFeatureContext(task())
    chains = {1: {0}, 2: {6}}
    ctx.pair(0, [1], chains)
    components, distances = ctx._components, ctx._distances
    ctx.pair(0, [2], {2: {6}, 1: {0}})
    assert ctx._components is components and ctx._distances is distances
    chains[2] = {3}
    ctx.pair(0, [1], chains)
    assert ctx._components is not components and ctx._distances is not distances
    assert 6 in ctx._free and 3 not in ctx._free


def test_never_reads_witness_and_reports_supplied_budget_and_actual_occupancy():
    class DeploymentTask:
        host = nx.path_graph(7)
        logical = nx.Graph([(0, 1), (0, 2)])
        name = "toy"
        problem = SimpleNamespace(h={}, j={})

        @property
        def witness(self):
            raise AssertionError("inference must not read witness")

    row = LayoutFeatureContext(DeploymentTask(), budget=4).pair(0, [1], {1: {0}, 2: {6}})
    assert value(row, "occupancy") == pytest.approx(2 / 7)
    assert value(row, "unplaced_variable_fraction") == pytest.approx(1 / 3)
    assert value(row, "remaining_budget_host_fraction") == pytest.approx(2 / 7)
    assert value(row, "unmet_edge_fraction") == 1.0


def test_added_features_are_finite_and_bounded_for_isolated_nodes_and_exhausted_budget():
    ctx = LayoutFeatureContext(task(host=nx.empty_graph(4), logical=nx.empty_graph(3)), budget=1)
    row = ctx.pair(0, [3], {1: {0}, 2: {1}})
    assert np.isfinite(row).all()
    assert np.abs(row[BASE_WIDTH:]).max() <= 1.0
    assert value(row, "remaining_budget_host_fraction") < 0.0
    assert value(row, "largest_branch_host_fraction") == 0.0


def test_contacts_update_remaining_logical_demand():
    ctx = LayoutFeatureContext(task(logical=nx.path_graph(3)))
    row = ctx.pair(2, [6], {0: {0}, 1: {1}})
    assert value(row, "unmet_edge_fraction") == pytest.approx(0.5)
