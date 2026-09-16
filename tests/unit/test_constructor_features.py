"""Every constructor action must expose its bound successor without teacher labels."""
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from isingfold.rl.contracts import Candidate, Opcode, WorkVector, WORK_FIELDS
from constructor_features import (
    ConstructorFeatureContext, FEATURE_SLICES, FEATURE_VERSION, SUMMARY_INDEX,
    SUBTYPES, WIDTH,
)


class DeploymentTask:
    name = "toy"
    host = nx.path_graph(6)
    logical = nx.Graph([(0, 1)])
    problem = SimpleNamespace(h={0: 4.0, 1: -1.0}, j={(0, 1): -3.0})

    @property
    def witness(self):
        raise AssertionError("features must not read witness")

    @property
    def ground_energy(self):
        raise AssertionError("features must not read ground energy")


def action(opcode, current, updates=None, **kwargs):
    updates = updates or {}
    return Candidate(opcode=opcode, affected=tuple(updates),
                     old_chains={v: frozenset(current.get(v, ())) for v in updates},
                     new_chains={v: frozenset(c) for v, c in updates.items()},
                     work=WorkVector(decisions=1), payload_key="test", **kwargs)


def scalar(row, block, name):
    return row[FEATURE_SLICES[block]][SUMMARY_INDEX[name]]


def test_place_encodes_actual_new_root_and_reduces_unplaced_variables():
    current = {1: {5}}
    fc = ConstructorFeatureContext(DeploymentTask())
    row = fc.observe(action(Opcode.PLACE, current, {0: {1}}), current)
    assert FEATURE_VERSION == "constructor-v1"
    assert row.dtype == np.float32 and row.shape == (WIDTH,)
    assert np.isfinite(row).all()
    assert scalar(row, "before", "placed_fraction") == 0.5
    assert scalar(row, "after", "placed_fraction") == 1.0
    assert scalar(row, "delta", "occupied_fraction") == pytest.approx(1 / 6)
    assert row[FEATURE_SLICES["local_after"]].any()


def test_route_exposes_contact_creation_and_full_structural_completion():
    current = {0: {0}, 1: {4}}
    candidate = action(Opcode.ROUTE, current, {0: {0, 1, 2, 3}},
                       routes=(((0, 1, 2, 3), 0),), target_demand=(0, 1))
    row = ConstructorFeatureContext(DeploymentTask()).observe(candidate, current)
    assert scalar(row, "before", "realized_edge_fraction") == 0.0
    assert scalar(row, "after", "realized_edge_fraction") == 1.0
    assert scalar(row, "after", "structurally_complete") == 1.0
    assert scalar(row, "after", "signed_realized_coupling") == -1.0
    assert row[FEATURE_SLICES["edits"]][0] == pytest.approx(3 / 6)


@pytest.mark.parametrize("old,new,subtype", [({0, 1}, {0, 1, 2}, "grow"),
                                             ({0, 1, 2}, {0, 1}, "shrink"),
                                             ({0, 1}, {2, 3}, "rewrite")])
def test_grow_shrink_and_rewrite_are_derived_from_edits_not_provenance(old, new, subtype):
    current = {0: old, 1: {5}}
    candidate = action(Opcode.REWRITE_ONE, current, {0: new}, provenance="misleading:arbitrary")
    row = ConstructorFeatureContext(DeploymentTask()).observe(candidate, current)
    expected = np.zeros(len(SUBTYPES))
    expected[SUBTYPES.index(subtype)] = 1.0
    np.testing.assert_array_equal(row[FEATURE_SLICES["subtype"]], expected)
    assert row[FEATURE_SLICES["local_before"]].any()
    assert row[FEATURE_SLICES["local_after"]].any()
    assert row[FEATURE_SLICES["edits"]][0] == pytest.approx(len(new - old) / 6)
    assert row[FEATURE_SLICES["edits"]][1] == pytest.approx(len(old - new) / 6)


@pytest.mark.parametrize("opcode", [Opcode.REWRITE_GROUP, Opcode.REPAIR_GROUP])
def test_group_replacement_retains_unaffected_and_updates_all_affected_chains(opcode):
    current = {0: {0, 1}, 1: {1, 2}}
    candidate = action(opcode, current, {0: {0}, 1: {1, 2, 3}})
    row = ConstructorFeatureContext(DeploymentTask()).observe(candidate, current)
    assert scalar(row, "before", "overlap_fraction") == pytest.approx(1 / 6)
    assert scalar(row, "after", "overlap_fraction") == 0.0
    assert scalar(row, "after", "structurally_complete") == 1.0


def test_commit_uses_referenced_archive_and_can_distinguish_equal_resource_embeddings():
    current = {0: {4}, 1: {5}}
    archive = [SimpleNamespace(chains={0: {0, 1}, 1: {2, 3}}),
               SimpleNamespace(chains={0: {0}, 1: {1, 2, 3}})]
    state = SimpleNamespace(archive=archive)
    fc = ConstructorFeatureContext(DeploymentTask())
    a = fc.observe(action(Opcode.COMMIT, current, archive_ref=0), current, state=state)
    b = fc.observe(action(Opcode.COMMIT, current, archive_ref=1), current, state=state)
    assert scalar(a, "before", "occupied_fraction") == pytest.approx(2 / 6)
    assert scalar(a, "after", "occupied_fraction") == scalar(b, "after", "occupied_fraction") == pytest.approx(4 / 6)
    assert scalar(a, "after", "max_chain_fraction") == pytest.approx(2 / 6)
    assert scalar(b, "after", "max_chain_fraction") == pytest.approx(3 / 6)
    assert scalar(a, "after", "field_load_concentration") != scalar(b, "after", "field_load_concentration")
    assert not np.array_equal(a, b)


@pytest.mark.parametrize("ref", [None, -1, 1, True])
def test_commit_missing_archive_reference_never_silently_uses_workspace(ref):
    current = {0: {0}, 1: {1}}
    candidate = action(Opcode.COMMIT, current, archive_ref=ref)
    with pytest.raises(ValueError, match="archive"):
        ConstructorFeatureContext(DeploymentTask()).observe(
            candidate, current, state=SimpleNamespace(archive=[SimpleNamespace(chains=current)]))


def test_restart_exposes_empty_successor_and_stop_exposes_current_failure():
    current = {0: {0}, 1: {5}}
    fc = ConstructorFeatureContext(DeploymentTask())
    restart = fc.observe(action(Opcode.RESTART, current, {0: set(), 1: set()}), current)
    stop = fc.observe(action(Opcode.STOP, current), current)
    assert scalar(restart, "after", "placed_fraction") == 0
    assert scalar(restart, "after", "occupied_fraction") == 0
    assert scalar(stop, "after", "placed_fraction") == 1
    assert scalar(stop, "after", "structurally_complete") == 0
    np.testing.assert_array_equal(stop[FEATURE_SLICES["delta"]], np.zeros(len(SUMMARY_INDEX)))
    assert not np.array_equal(restart, stop)


def test_work_horizon_and_restarts_are_explicit_even_for_terminal_actions():
    current = {0: {0}}
    caps = WorkVector(**{name: 100 for name in WORK_FIELDS})
    reserve = WorkVector(**{name: 10 for name in WORK_FIELDS})
    state = SimpleNamespace(archive=[], remaining=WorkVector(**{name: 50 for name in WORK_FIELDS}),
                            restarts_left=1)
    ctx = SimpleNamespace(caps=caps, reserve=reserve, restart_allowance=2, max_commit=8)
    row = ConstructorFeatureContext(DeploymentTask()).observe(
        action(Opcode.STOP, current), current, state=state, ctx=ctx, steps_left=4, max_steps=10)
    np.testing.assert_allclose(row[FEATURE_SLICES["remaining_work"]], np.full(len(WORK_FIELDS), 0.5))
    np.testing.assert_allclose(row[FEATURE_SLICES["usable_work"]], np.full(len(WORK_FIELDS), 0.4))
    np.testing.assert_allclose(row[FEATURE_SLICES["context"]][:3], [0.4, 0.6, 0.5])


def test_cap_violations_and_disconnected_chains_are_not_observed_as_valid():
    current = {0: {0}, 1: {5}}
    row = ConstructorFeatureContext(DeploymentTask(), budget=2).observe(
        action(Opcode.REWRITE_ONE, current, {0: {0, 2}}), current)
    assert scalar(row, "after", "within_budget") == 0
    assert scalar(row, "after", "structurally_complete") == 0
    assert scalar(row, "after", "connected_chain_fraction") == 0.5


def test_same_length_chain_rewrite_rebuilds_summary_and_local_capacity():
    fc = ConstructorFeatureContext(DeploymentTask())
    a, b = {0: {0}, 1: {1}}, {0: {0}, 1: {5}}
    first = fc.observe(action(Opcode.STOP, a), a)
    second = fc.observe(action(Opcode.STOP, b), b)
    assert scalar(first, "before", "realized_edge_fraction") == 1.0
    assert scalar(second, "before", "realized_edge_fraction") == 0.0


@pytest.mark.parametrize("steps_left,max_steps", [(float("nan"), 5), (-1, 5), (3, 0), (2, float("inf"))])
def test_invalid_horizon_is_rejected(steps_left, max_steps):
    with pytest.raises(ValueError, match="horizon"):
        ConstructorFeatureContext(DeploymentTask()).observe(
            action(Opcode.STOP, {}), {}, steps_left=steps_left, max_steps=max_steps)
