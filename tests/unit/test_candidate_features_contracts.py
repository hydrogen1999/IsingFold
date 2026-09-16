"""Regression contracts for observations used by construction and contact-growth policies."""
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from candidate_features import FEATURE_SLICES, WIDTH, FeatureContext, frontier_features


def task(host=None, logical=None, h=None, j=None):
    return SimpleNamespace(
        name="toy", host=nx.path_graph(6) if host is None else host,
        logical=nx.path_graph(3) if logical is None else logical,
        problem=SimpleNamespace(h=h or {}, j=j or {}),
    )


def test_same_length_rewrite_invalidates_cached_occupancy_and_chain_owners():
    ctx = FeatureContext(task())
    chains = {0: {0}, 1: {2}}
    before = ctx.pair(0, [1], chains)
    chains[1] = {5}
    allowed, placed = ctx.state(chains)
    assert 2 in allowed and 5 not in allowed
    assert placed[1] == {5}
    after = ctx.pair(0, [1], chains)
    assert not np.array_equal(before, after)


def test_cache_preserves_variable_identity_and_is_independent_of_insertion_order():
    ctx = FeatureContext(task())
    ctx.state({1: {0}})
    _, placed = ctx.state({"1": {0}})
    assert "1" in placed and 1 not in placed
    allowed, placed = ctx.state({0: {0}, 1: {2}})
    allowed_again, placed_again = ctx.state({1: {2}, 0: {0}})
    assert allowed_again is allowed and placed_again is placed


def test_no_qubit_candidate_has_the_same_physics_columns_and_current_budget():
    ctx = FeatureContext(task(h={0: 2.0, 1: 4.0}, j={(0, 1): 2.0, (0, 2): 6.0}))
    chains = {0: {0}, 1: {4}}
    grow = ctx.pair(0, [1], chains)
    empty = ctx.pair(0, [], chains, "REWRITE_ONE")
    commit = ctx.pair(None, [], chains, "COMMIT")
    assert grow.shape == empty.shape == commit.shape == (WIDTH,)
    np.testing.assert_allclose(grow[FEATURE_SLICES["physics"]], [0.5, 1.0, 0.75])
    np.testing.assert_array_equal(empty[FEATURE_SLICES["physics"]], grow[FEATURE_SLICES["physics"]])
    np.testing.assert_allclose(commit[FEATURE_SLICES["budget"]], [2 / 6, 4 / 3])
    assert not commit[FEATURE_SLICES["physics"]].any()
    assert not grow[FEATURE_SLICES["reserved"]].any()
    assert not empty[FEATURE_SLICES["reserved"]].any()


def test_isolated_logical_variable_produces_finite_features():
    ctx = FeatureContext(task(logical=nx.empty_graph(1), h={0: 1.0}))
    assert np.isfinite(ctx.pair(0, [0], {})).all()


def test_budget_counts_distinct_physical_qubits_once():
    ctx = FeatureContext(task())
    # A candidate may include a chain attachment already occupied by this variable.
    features = ctx.pair(0, [0, 1, 1], {0: {0}, 1: {4}})
    np.testing.assert_allclose(features[FEATURE_SLICES["budget"]], [3 / 6, 3 / 3])


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf")])
def test_invalid_budget_is_rejected(budget):
    with pytest.raises(ValueError, match="budget"):
        FeatureContext(task(), budget)


def test_frontier_counts_leaf_space_even_if_walk_finishes_before_two_hops():
    values = frontier_features(nx.path_graph(2), nx.empty_graph(1), {}, 0, 0)
    np.testing.assert_allclose(values[:2], [1 / 64, 1 / 256])


def test_frontier_matches_residual_graph_distances_and_does_not_cross_occupied_chain():
    host = nx.path_graph(6)
    logical = nx.path_graph(3)
    chains = {0: {0}, 1: {3}, 2: {5}}
    values = frontier_features(host, logical, chains, 0, 1)
    allowed = set(host) - {q for c in chains.values() for q in c}
    distances = nx.single_source_shortest_path_length(host.subgraph(allowed), 1, cutoff=3)
    np.testing.assert_allclose(values[:2], [sum(0 < d <= 2 for d in distances.values()) / 64,
                                          sum(0 < d <= 3 for d in distances.values()) / 256])
    # Chain 1 blocks the corridor. Chain 2 is beyond it and is not locally visible.
    assert values[2] == 1 / 8


def test_frontier_cap_bounds_the_whole_walk_not_each_neighbor_loop():
    host = nx.balanced_tree(4, 3)
    values = frontier_features(host, nx.empty_graph(1), {}, 0, 0, cap=6)
    np.testing.assert_allclose(values[:2], [6 / 64, 6 / 256])


def test_frontier_serves_only_previously_unmet_logical_contacts():
    host = nx.path_graph(4)
    logical = nx.path_graph(2)
    before = frontier_features(host, logical, {0: {0}, 1: {2, 3}}, 0, 1)
    assert before[3] == 1 / 4
    after = frontier_features(host, logical, {0: {0, 1}, 1: {2}}, 0, 3)
    assert after[3] == 0
