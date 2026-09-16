"""ADR-001: the program the scorer sees is the program the evaluator ran.

The successor scorer is trained on labels the environment produced by compiling an embedding
with the registered strength ratio times the instance's RMS coefficient scale. If the probe
that builds the scorer's input compiles with any other strength, the model fits labels of a
program it never sees. This pins the two compilers to each other on a toy fixture.
"""
import sys
from pathlib import Path

import networkx as nx
import pytest

PROBES = Path(__file__).resolve().parents[2] / "probes"
SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture(autouse=True)
def _paths(monkeypatch):
    monkeypatch.setenv("ISINGFOLD_SRC", str(SRC))
    monkeypatch.syspath_prepend(str(PROBES))
    monkeypatch.syspath_prepend(str(SRC))


def _fixture():
    from isingfold.embedding import LogicalProblem

    # A triangle with unequal couplings and one field, so mean|J| and RMS(h, J) differ.
    logical = nx.Graph([(0, 1), (1, 2), (0, 2)])
    problem = LogicalProblem.from_dicts({0: 0.7}, {(0, 1): 1.0, (1, 2): -0.25, (0, 2): 2.0})
    host = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)])
    chains = {0: frozenset({0}), 1: frozenset({1}), 2: frozenset({2, 3})}
    return logical, problem, host, chains


class _Task:
    def __init__(self, problem, host):
        self.problem = problem
        self.host = host


def test_probe_compiler_uses_the_environment_strength_registry():
    from isingfold.rl.program import compile_program, strength_registry
    from train_successor import compile_for
    from _context import host_context

    _, problem, host, chains = _fixture()
    ctx = host_context(8)
    index = 1
    strengths = strength_registry(problem, ctx.strength_ratios, ctx.epsilon_strength)
    expected = compile_program(chains, host, problem, strengths[index], index,
                               field_limit=ctx.field_limit, coupler_limit=ctx.coupler_limit)
    shown = compile_for(_Task(problem, host), ctx, chains, index=index)

    assert shown is not None
    assert shown.strength == pytest.approx(expected.strength)
    assert dict(shown.h_phys) == pytest.approx(dict(expected.h_phys))
    assert {k: v for k, v in shown.j_phys.items()} == pytest.approx(dict(expected.j_phys))
