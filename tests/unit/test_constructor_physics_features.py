"""Compact physics observations see coefficients and exact public successors only."""
from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from constructor_physics_features import (
    BASE_WIDTH, FEATURE_SLICES, SUMMARY_INDEX, WIDTH, PhysicsFeatures,
)
from constructor_tiny_gate import Features
from isingfold.rl.contracts import Candidate, Opcode, WorkVector


class PublicTask:
    def __init__(self, host, logical, h=None, j=None):
        self.host, self.logical = host, logical
        self.problem = SimpleNamespace(h=h or {}, j=j or {})

    @property
    def witness(self):
        raise AssertionError("privileged witness read")

    @property
    def initial_embedding(self):
        raise AssertionError("privileged initial embedding read")

    @property
    def ground_energy(self):
        raise AssertionError("outcome read")

    @property
    def evaluator(self):
        raise AssertionError("evaluator read")

    @property
    def prefix_source(self):
        raise AssertionError("training prefix read")


def action(opcode, updates=None, **kwargs):
    updates = updates or {}
    return Candidate(opcode=opcode, affected=tuple(updates), old_chains={},
                     new_chains={v: frozenset(c) for v, c in updates.items()},
                     work=WorkVector(decisions=1), payload_key="physics-test", **kwargs)


def scalar(row, block, name):
    return row[FEATURE_SLICES[block]][SUMMARY_INDEX[name]]


def toy():
    host = nx.Graph([(0, 1), (1, 2), (0, 3), (2, 4)])
    logical = nx.Graph([(0, 1), (0, 2)])
    return PublicTask(host, logical, h={0: .2}, j={(0, 1): 1., (0, 2): -1.})


def test_prefix_matches_local20_and_zero_padding_preserves_logits_for_every_opcode():
    task = toy()
    current = {0: {0, 1}, 1: {3}, 2: {4}}
    archive = SimpleNamespace(archive=[SimpleNamespace(chains={0: {0, 1, 2}, 1: {3}, 2: {4}})])
    candidates = [action(Opcode.PLACE, {0: {1}}), action(Opcode.ROUTE, {0: {0, 1, 2}}),
                  action(Opcode.REWRITE_ONE, {0: {1, 2}}),
                  action(Opcode.REWRITE_GROUP, {0: {1, 2}, 1: {3}}),
                  action(Opcode.REPAIR_GROUP, {0: {0, 1, 2}}),
                  action(Opcode.RESTART, {v: set() for v in current}),
                  action(Opcode.STOP), action(Opcode.COMMIT, archive_ref=0)]
    base, physics = Features(task, local_channels=True), PhysicsFeatures(task)
    weights = np.random.default_rng(5).normal(size=BASE_WIDTH).astype(np.float32)
    for candidate in candidates:
        row = physics.observe(candidate, current, state=archive, ctx=None, steps_left=3, max_steps=8)
        old = base.observe(candidate, current, state=archive, ctx=None, steps_left=3, max_steps=8)
        assert row.shape == (WIDTH,) == (32,) and row.dtype == np.float32
        assert np.isfinite(row).all()
        np.testing.assert_array_equal(row[:BASE_WIDTH], old)
        assert row @ np.pad(weights, (0, WIDTH - BASE_WIDTH)) == pytest.approx(old @ weights)


def test_public_magnitude_and_sign_change_physics_but_not_local20():
    task = toy()
    current = {0: {0, 1, 2}, 1: {3}, 2: {4}}
    stop = action(Opcode.STOP)
    first = PhysicsFeatures(task).observe(stop, current)
    task.problem.j = {(0, 1): 9., (0, 2): -1.}
    hot = PhysicsFeatures(task).observe(stop, current)
    np.testing.assert_array_equal(first[:BASE_WIDTH], hot[:BASE_WIDTH])
    assert scalar(hot, "after", "coefficient_load_concentration") > scalar(first, "after", "coefficient_load_concentration")
    assert scalar(hot, "after", "bridge_load_bottleneck_proxy") < scalar(first, "after", "bridge_load_bottleneck_proxy")
    assert scalar(hot, "after", "signed_realized_coupling") > scalar(first, "after", "signed_realized_coupling")
    task.problem.j = {(0, 1): -9., (0, 2): 1.}
    flipped = PhysicsFeatures(task).observe(stop, current)
    np.testing.assert_allclose(flipped[FEATURE_SLICES["after"]][:-1], hot[FEATURE_SLICES["after"]][:-1])
    assert scalar(flipped, "after", "signed_realized_coupling") == pytest.approx(-scalar(hot, "after", "signed_realized_coupling"))


def test_field_distribution_is_visible_without_couplings():
    task = PublicTask(nx.path_graph(4), nx.empty_graph(2), h={0: 1., 1: 1.})
    current = {0: {0}, 1: {1, 2, 3}}
    before = PhysicsFeatures(task).observe(action(Opcode.STOP), current)
    task.problem.h = {0: 9., 1: 1.}
    after = PhysicsFeatures(task).observe(action(Opcode.STOP), current)
    assert scalar(after, "after", "coefficient_load_concentration") > scalar(before, "after", "coefficient_load_concentration")


def test_commit_reads_selected_archive_and_distinguishes_equal_resource_embeddings():
    task = PublicTask(nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0),
                                (1, 4), (4, 5), (5, 2)]), nx.Graph([(0, 1)]),
                      h={0: 1.}, j={(0, 1): -2.})
    current = {0: {0}, 1: {3}}
    first, second = {0: {0, 1}, 1: {2, 3}}, {0: {0, 1}, 1: {4, 5}}
    state = SimpleNamespace(archive=[SimpleNamespace(chains=first), SimpleNamespace(chains=second)])
    fc = PhysicsFeatures(task)
    a = fc.observe(action(Opcode.COMMIT, archive_ref=0), current, state=state)
    b = fc.observe(action(Opcode.COMMIT, archive_ref=1), current, state=state)
    np.testing.assert_array_equal(a[FEATURE_SLICES["before"]], b[FEATURE_SLICES["before"]])
    np.testing.assert_array_equal(a[FEATURE_SLICES["after"]], fc.observe(action(Opcode.STOP), first)[FEATURE_SLICES["after"]])
    np.testing.assert_array_equal(b[FEATURE_SLICES["after"]], fc.observe(action(Opcode.STOP), second)[FEATURE_SLICES["after"]])
    assert scalar(a, "after", "weighted_contact_redundancy") == pytest.approx(.5)
    assert scalar(b, "after", "weighted_contact_redundancy") == 0.


@pytest.mark.parametrize("ref", [None, -1, True, 1])
def test_commit_rejects_invalid_reference(ref):
    with pytest.raises(ValueError, match="archive"):
        PhysicsFeatures(toy()).observe(action(Opcode.COMMIT, archive_ref=ref), {},
                                        state=SimpleNamespace(archive=[SimpleNamespace(chains={})]))


def test_restart_and_stop_expose_their_declared_successors():
    fc = PhysicsFeatures(toy())
    current = {0: {0, 1, 2}, 1: {3}, 2: {4}}
    stop = fc.observe(action(Opcode.STOP), current)
    np.testing.assert_array_equal(stop[FEATURE_SLICES["before"]], stop[FEATURE_SLICES["after"]])
    restart = fc.observe(action(Opcode.RESTART, {v: set() for v in current}), current)
    np.testing.assert_array_equal(restart[FEATURE_SLICES["before"]], stop[FEATURE_SLICES["before"]])
    np.testing.assert_array_equal(restart[FEATURE_SLICES["after"]], np.zeros(6))


def test_cycle_and_bridge_proxies_distinguish_path_from_cycle():
    logical = nx.empty_graph(1)
    chain = {0: {0, 1, 2, 3}}
    path = PhysicsFeatures(PublicTask(nx.path_graph(4), logical, h={0: 1.}))
    cycle = PhysicsFeatures(PublicTask(nx.cycle_graph(4), logical, h={0: 1.}))
    a, b = (fc.observe(action(Opcode.STOP), chain) for fc in (path, cycle))
    assert scalar(a, "after", "bridge_load_bottleneck_proxy") == pytest.approx(1.)
    assert scalar(b, "after", "bridge_load_bottleneck_proxy") == 0.
    assert scalar(a, "after", "cycle_redundancy") == 0.
    assert scalar(b, "after", "cycle_redundancy") == pytest.approx(.25)


def test_relabeling_and_coordinate_translation_do_not_change_features():
    task = toy()
    current, updates = {0: {0, 1}, 1: {3}, 2: {4}}, {0: {0, 1, 2}}
    original = PhysicsFeatures(task).observe(action(Opcode.REWRITE_ONE, updates), current)
    qmap = {q: (q + 100, 500 - q) for q in task.host}
    vmap = {v: f"logical-{10-v}" for v in task.logical}
    moved = PublicTask(nx.relabel_nodes(task.host, qmap), nx.relabel_nodes(task.logical, vmap),
                       h={vmap[v]: h for v, h in task.problem.h.items()},
                       j={(vmap[u], vmap[v]): j for (u, v), j in task.problem.j.items()})
    convert = lambda chains: {vmap[v]: {qmap[q] for q in chain} for v, chain in chains.items()}
    observed = PhysicsFeatures(moved).observe(action(Opcode.REWRITE_ONE, convert(updates)), convert(current))
    np.testing.assert_allclose(observed, original, atol=1e-7)


def test_same_size_rewrite_updates_cached_contacts_and_neighbor_loads():
    fc = PhysicsFeatures(toy())
    current = {0: {0, 1}, 1: {3}, 2: {4}}
    first = fc.observe(action(Opcode.REWRITE_ONE, {0: {1, 2}}), current)
    second = fc.observe(action(Opcode.REWRITE_ONE, {0: {0, 1}}), {**current, 0: {1, 2}})
    np.testing.assert_array_equal(first[FEATURE_SLICES["before"]], second[FEATURE_SLICES["after"]])
    np.testing.assert_array_equal(first[FEATURE_SLICES["after"]], second[FEATURE_SLICES["before"]])
    assert scalar(first, "before", "signed_realized_coupling") != scalar(first, "after", "signed_realized_coupling")


@pytest.mark.parametrize("scale", [0., 1e-300, 1e300])
def test_empty_zero_and_extreme_coefficients_produce_finite_features(scale):
    task = toy()
    task.problem.h = {0: scale}
    task.problem.j = {(0, 1): scale, (0, 2): -scale}
    fc = PhysicsFeatures(task)
    for current in ({}, {0: {0, 1, 2}, 1: {3}, 2: {4}}):
        row = fc.observe(action(Opcode.STOP), current)
        assert np.isfinite(row).all()
        assert np.max(np.abs(row[BASE_WIDTH:])) <= 1.


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_coefficients_are_rejected(bad):
    task = toy()
    task.problem.j[(0, 1)] = bad
    with pytest.raises(ValueError, match="finite"):
        PhysicsFeatures(task)


def test_observation_does_not_scan_host_or_invoke_completion_solver(monkeypatch):
    import minorminer

    class GuardedHost(nx.Graph):
        deny_scan = False

        def __iter__(self):
            if self.deny_scan:
                raise AssertionError("per-candidate full host scan")
            return super().__iter__()

    host = GuardedHost(nx.path_graph(1000))
    task = PublicTask(host, nx.Graph([(0, 1)]), j={(0, 1): -1.})
    fc = PhysicsFeatures(task)
    host.deny_scan = True

    def forbidden(*args, **kwargs):
        raise AssertionError("completion solver read")

    monkeypatch.setattr(minorminer, "find_embedding", forbidden)
    row = fc.observe(action(Opcode.REWRITE_ONE, {0: {0, 1, 2}}), {0: {0, 1}, 1: {3}})
    assert np.isfinite(row).all()


def test_bridge_proxy_uses_load_on_both_sides_and_handles_components():
    # Two internally cyclic blocks can still have a single loaded bottleneck.
    host = nx.Graph([(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)])
    task = PublicTask(host, nx.empty_graph(1), h={0: 1.})
    row = PhysicsFeatures(task).observe(action(Opcode.STOP), {0: set(host)})
    assert scalar(row, "after", "bridge_load_bottleneck_proxy") == pytest.approx(1.)
    assert scalar(row, "after", "cycle_redundancy") == pytest.approx(2. / 7.)
    # Invalid/disconnected candidates remain finite; the local prefix records
    # disconnection, and the proxy never invents a bridge between components.
    host = nx.Graph([(0, 1), (1, 2), (3, 4)])
    row = PhysicsFeatures(PublicTask(host, nx.empty_graph(1), h={0: 1.})).observe(
        action(Opcode.STOP), {0: set(host)})
    assert scalar(row, "after", "bridge_load_bottleneck_proxy") == pytest.approx(.4)
