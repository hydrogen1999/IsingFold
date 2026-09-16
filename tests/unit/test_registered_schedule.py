"""ADR-002: a Context used for labelling carries the registered annealing schedule.

Under the auto schedule the surrogate annealer derives beta from the programmed coefficients
and cancels the energy compression the objective is about. probes/_context.py is the one
place probes build their Context, so it is where the registered range must live.
"""
import os
import sys
from pathlib import Path

import pytest

PROBES = Path(__file__).resolve().parents[2] / "probes"
SRC = Path(__file__).resolve().parents[2] / "src"


@pytest.fixture(autouse=True)
def _paths(monkeypatch):
    monkeypatch.setenv("ISINGFOLD_SRC", str(SRC))
    monkeypatch.syspath_prepend(str(PROBES))
    monkeypatch.syspath_prepend(str(SRC))


def test_host_context_carries_the_registered_schedule():
    from _context import REGISTERED_BETA_RANGE, host_context

    ctx = host_context(680)
    assert ctx.beta_range == REGISTERED_BETA_RANGE
    assert REGISTERED_BETA_RANGE == (0.1, 2.0)


def test_host_context_can_still_be_asked_for_the_auto_schedule_explicitly():
    from _context import host_context

    ctx = host_context(680, beta_range=None)
    assert ctx.beta_range is None
