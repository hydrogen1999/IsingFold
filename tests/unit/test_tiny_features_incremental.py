"""The incremental tiny features equal the direct computation, candidate for candidate."""
import importlib.util
from pathlib import Path

import numpy as np
import torch

_pc = Path(__file__).resolve().parents[2] / "probes" / "constructor_curriculum.py"
spec = importlib.util.spec_from_file_location("constructor_curriculum", _pc)
cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
from constructor_rollout import _context  # noqa: E402
from constructor_tiny_gate import Features, no_completion_solver  # noqa: E402
from isingfold.rl.contracts import DecisionState  # noqa: E402
from isingfold.rl.env import EmbeddingEnv, Mode, fixed_strength_selector  # noqa: E402


def _walk(task, steps, seed):
    fc = Features(task)
    ctx = _context(task, len(task.host), steps + 2, 8, None, 2)
    env = EmbeddingEnv(task, ctx, mode=Mode.CONSTRUCTION, initializer=None,
                       selector=fixed_strength_selector(), reward_reads=8, build_observation=False)
    env.generator.allow_satisfied_growth = True
    rng = np.random.default_rng(seed)
    dec = env.reset(seed)
    checked = 0
    while isinstance(dec, DecisionState) and checked < 400:
        legal = [i for i, ok in enumerate(dec.legal_mask) if ok]
        chains = env.state.chains
        for i in legal:
            cand = dec.candidates[i]
            fast = fc.observe(cand, chains, state=env.state, ctx=ctx, steps_left=steps, max_steps=steps)
            if cand.opcode.value == "COMMIT":
                after = dict(env.state.archive[cand.archive_ref].chains)
            else:
                after = dict(chains); after.update(cand.new_chains)
            ref_delta = fc.summary_reference(after) - fc.summary_reference(chains)
            assert np.allclose(fast[8:13], ref_delta, atol=1e-6), (cand.opcode, fast[8:13], ref_delta)
            checked += 1
        dec = env.step(dec, int(rng.choice(legal)), evaluate_training_reward=False).next_decision_or_terminal
    return checked


def test_incremental_summary_matches_reference_on_random_walks():
    for stage, seed in (("a", 31), ("b", 32), ("b", 33)):
        train, _ = cc.build_sets(stage, 1, 1, seed=seed)
        with no_completion_solver():
            assert _walk(train[0], 40, seed) > 20


def test_connectivity_cache_is_correct_for_split_chains():
    import networkx as nx
    train, _ = cc.build_sets("a", 1, 1, seed=34)
    fc = Features(train[0])
    host = train[0].host
    nodes = sorted(host)
    a, b = nodes[0], nodes[-1]
    assert fc._connected(frozenset({a})) is True
    assert fc._connected(frozenset()) is True
    joined = frozenset(nx.shortest_path(host, a, b))
    assert fc._connected(joined) is True
    if not host.has_edge(a, b):
        assert fc._connected(frozenset({a, b})) is False


def test_local_channels_separate_placements_that_leave_different_room():
    """Two placements realising the same edge but leaving different free space must differ."""
    import networkx as nx
    import constructor_curriculum as cc
    # a corridor: qubit 0 is a dead end, qubit 4 opens onto a wide area
    host = nx.Graph()
    host.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    host.add_edges_from([(4, 10 + i) for i in range(6)])
    host.add_edges_from([(10 + i, 20 + i) for i in range(6)])
    logical = nx.path_graph(3)
    task = cc.Task("corridor", logical, host, "toy")
    plain, local = Features(task), Features(task, local_channels=True)
    chains = {0: frozenset({2}), 1: frozenset(), 2: frozenset()}
    corner = {0: frozenset({2}), 1: frozenset({1}), 2: frozenset()}
    open_side = {0: frozenset({2}), 1: frozenset({3}), 2: frozenset()}
    assert np.allclose(plain.successor_summary(chains, corner, [1]),
                       plain.successor_summary(chains, open_side, [1]))
    a = local.local(chains, corner, [1])
    b = local.local(chains, open_side, [1])
    assert not np.allclose(a, b) and b[1] > a[1]
    assert len(a) == cc.LOCAL_CHANNELS


def test_local_features_are_the_tiny_ones_plus_four():
    import constructor_curriculum as cc
    train, _ = cc.build_sets("b", 1, 1, seed=61)
    task = train[0]
    plain, local = cc.make_features("tiny", task), cc.make_features("local", task)
    ctx = _context(task, len(task.host), 40, 8, None, 2)
    env = EmbeddingEnv(task, ctx, mode=Mode.CONSTRUCTION, initializer=None,
                       selector=fixed_strength_selector(), reward_reads=8, build_observation=False)
    env.generator.allow_satisfied_growth = True
    dec = env.reset(0)
    assert isinstance(dec, DecisionState)
    chains = env.state.chains
    for i, ok in enumerate(dec.legal_mask):
        if not ok:
            continue
        cand = dec.candidates[i]
        short = plain.observe(cand, chains, state=env.state, ctx=ctx, steps_left=1, max_steps=40)
        long = local.observe(cand, chains, state=env.state, ctx=ctx, steps_left=1, max_steps=40)
        assert len(long) == len(short) + cc.LOCAL_CHANNELS
        assert np.allclose(long[:len(short)], short)
        assert np.all(np.isfinite(long))
