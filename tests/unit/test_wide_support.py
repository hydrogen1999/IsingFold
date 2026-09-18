"""The wide construction support: registered by context version, frontier-complete PLACE,
and work caps that a long construction cannot exhaust."""
import importlib.util
from dataclasses import replace
from pathlib import Path

import networkx as nx
import numpy as np
import pytest
import torch

_pc = Path(__file__).resolve().parents[2] / "probes" / "constructor_curriculum.py"
spec = importlib.util.spec_from_file_location("constructor_curriculum", _pc)
cc = importlib.util.module_from_spec(spec); spec.loader.exec_module(cc)
from _context import WIDE_QUOTAS, construction_context, scale_caps_for_steps  # noqa: E402
from constructor_rollout import episode  # noqa: E402
from constructor_tiny_gate import no_completion_solver  # noqa: E402
from isingfold.rl.contracts import WIDE_STATE_CHANGING, WIDE_SUFFIX, DecisionState, Opcode  # noqa: E402
from isingfold.rl.env import EmbeddingEnv, Mode, fixed_strength_selector  # noqa: E402


def test_the_registry_admits_only_the_two_widths():
    narrow = construction_context(64, 8, 10)
    assert narrow.max_state_changing == 64 and narrow.padded_actions == 73
    wide = construction_context(64, 8, 10, wide=True)
    assert wide.max_state_changing == WIDE_STATE_CHANGING and wide.padded_actions == WIDE_STATE_CHANGING + 9
    assert WIDE_SUFFIX in wide.context_version and dict(wide.construction_quotas) == WIDE_QUOTAS
    with pytest.raises(ValueError):
        replace(narrow, max_state_changing=300, padded_actions=309)
    with pytest.raises(ValueError):
        replace(narrow, max_state_changing=WIDE_STATE_CHANGING, padded_actions=WIDE_STATE_CHANGING + 9)


def test_caps_scale_with_steps_and_variables():
    ctx = construction_context(64, 400, 700)
    scaled = scale_caps_for_steps(ctx, 400, 1500)
    assert scaled.caps.feature_work >= 1502 * 32 * (ctx.padded_actions + 408)
    assert scaled.caps.validator_calls >= 1502 * (ctx.max_state_changing + 8)
    assert scaled.caps.route_expansions >= 1502 * 64_000
    assert scaled.caps.decisions == ctx.caps.decisions


def _frontier_state():
    """A grid host, a star logical graph with sixteen leaves, the hub placed on a 3x3 block:
    sixteen unplaced variables adjacent to a placed one, twelve free roots around the block,
    more than the registered shortlist (eight variables, four roots each) can offer."""
    host = nx.convert_node_labels_to_integers(nx.grid_2d_graph(9, 9))
    logical = nx.star_graph(16)
    task = cc.Task("frontier", logical, host, "toy")
    block = frozenset(9 * r + c for r in (3, 4, 5) for c in (3, 4, 5))
    return task, {0: block}


@pytest.mark.parametrize("wide,min_vars", [(False, 1), (True, 12)])
def test_wide_support_offers_place_for_the_whole_frontier(wide, min_vars):
    task, placed = _frontier_state()
    ctx = construction_context(len(task.host), task.logical.number_of_nodes(),
                               task.logical.number_of_edges(), wide=wide)
    ctx = scale_caps_for_steps(ctx, task.logical.number_of_nodes(), 64)
    env = EmbeddingEnv(task, ctx, mode=Mode.CONSTRUCTION, initializer=lambda *_: placed,
                       selector=fixed_strength_selector(), reward_reads=8, build_observation=False)
    dec = env.reset(0)
    assert isinstance(dec, DecisionState)
    place_vars = {next(iter(c.new_chains)) for c in dec.candidates if c.opcode is Opcode.PLACE}
    n_state_changing = sum(1 for c in dec.candidates if c.changes_workspace)
    if wide:
        assert len(place_vars) >= min_vars and n_state_changing > 64
    else:
        assert len(place_vars) <= 8 and n_state_changing <= 64


def test_an_episode_runs_under_the_wide_support():
    train, _ = cc.build_sets("b", 1, 1, seed=21)
    task = train[0]
    actor = cc.make_actor("linear", 8, cc.FEATURE_WIDTHS["tiny"])
    with no_completion_solver(), torch.no_grad():
        rec = episode(task, actor, cc.make_features("tiny", task), 1., 40, np.random.default_rng(3), 60.,
                      train=True, objective="feasibility", evaluate_reward=False, wide=True)
    assert rec["steps"] >= 1 and rec["reason"]


def test_curriculum_accepts_the_wide_support():
    args = cc.parse(["--stage", "a", "--train", "1", "--heldout", "1", "--episodes", "2", "--iterations", "1",
                     "--eval-episodes", "1", "--seed", "22", "--max-steps", "12", "--support", "wide"])
    train, heldout = cc.build_sets("a", 1, 1, seed=22)
    with cc.no_completion_solver():
        summary = cc.run(args, train, heldout)
    assert 0. <= summary["heldout"]["final"] <= 1.


def test_ground_energy_enumerates_twenty_variables():
    from isingfold.embedding import LogicalProblem
    n = 18
    problem = LogicalProblem.from_dicts({i: 0. for i in range(n)}, {(i, i + 1): -1. for i in range(n - 1)})
    assert cc.exact_ground_energy(problem) == -(n - 1)


def test_the_wide_registration_keeps_every_recovery_family():
    from isingfold.rl.contracts import Opcode
    quotas = dict(construction_context(64, 8, 10, wide=True).construction_quotas)
    assert set(quotas) >= {"place", "route", "grow", "shrink", "rewrite", "repair", "restart"}
    assert all(v > 0 for v in quotas.values())
    task, placed = _frontier_state()
    ctx = construction_context(len(task.host), task.logical.number_of_nodes(),
                               task.logical.number_of_edges(), wide=True)
    ctx = scale_caps_for_steps(ctx, task.logical.number_of_nodes(), 64)
    env = EmbeddingEnv(task, ctx, mode=Mode.CONSTRUCTION, initializer=lambda *_: placed,
                       selector=fixed_strength_selector(), reward_reads=8, build_observation=False)
    dec = env.reset(0)
    assert isinstance(dec, DecisionState)
    opcodes = {c.opcode for c in dec.candidates}
    assert Opcode.PLACE in opcodes and Opcode.STOP in opcodes


def test_the_first_placement_offers_a_dense_spread_under_the_wide_budget():
    """From empty there is no frontier; the wide budget must still reach most of the host."""
    import networkx as nx
    from isingfold.rl.contracts import Opcode
    host = nx.convert_node_labels_to_integers(nx.grid_2d_graph(12, 12))
    logical = nx.cycle_graph(10)
    task = cc.Task("empty-start", logical, host, "toy")
    rows = {}
    for wide in (False, True):
        ctx = construction_context(len(task.host), task.logical.number_of_nodes(),
                                   task.logical.number_of_edges(), wide=wide)
        ctx = scale_caps_for_steps(ctx, task.logical.number_of_nodes(), 32)
        env = EmbeddingEnv(task, ctx, mode=Mode.CONSTRUCTION, initializer=None,
                           selector=fixed_strength_selector(), reward_reads=8, build_observation=False)
        dec = env.reset(0)
        places = [c for c in dec.candidates if c.opcode is Opcode.PLACE]
        roots = {next(iter(next(iter(c.new_chains.values())))) for c in places}
        variables = {next(iter(c.new_chains)) for c in places}
        rows[wide] = (len(roots), len(variables))
    narrow_roots, narrow_vars = rows[False]
    wide_roots, wide_vars = rows[True]
    assert wide_roots > narrow_roots and wide_roots >= 0.5 * len(host)
    assert wide_vars >= narrow_vars
