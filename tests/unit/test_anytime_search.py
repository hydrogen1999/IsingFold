from __future__ import annotations

import random
from types import SimpleNamespace

from lac_minorminer._search_loop import run_search
from lac_minorminer.components import ControlDecision
from lac_minorminer.diagnostics import SearchWorkCounters, TerminationReason


def _snapshot(chains, *, generation: int, valid: bool):
    return SimpleNamespace(
        chains=chains,
        generation=generation,
        valid=valid,
        max_occupancy=1 if valid else 2,
        total_excess_occupancy=0 if valid else 1,
        missing_source_edges=0,
        used_target_nodes=len({node for chain in chains for node in chain}),
    )


class ScriptedSession:
    def __init__(self) -> None:
        self.states = [
            _snapshot(((0,), (0,)), generation=1, valid=False),
            _snapshot(((0,), (1,)), generation=2, valid=True),
            _snapshot(((0,), (2,)), generation=3, valid=True),
        ]
        self.index = 0

    def snapshot(self):
        return self.states[self.index]

    def eligible_variables(self):
        return [] if self.snapshot().valid else [0]

    def restart(self) -> None:
        raise AssertionError("the one-attempt scripted search must not restart")

    def work_counters(self) -> SearchWorkCounters:
        # The scripted transition has no router/materializer; its index is the exact number
        # of proposal decisions consumed by this intentionally minimal test double.
        return SearchWorkCounters(decisions=self.index)


class ContinueControl:
    def decide(self, context):
        return ControlDecision.CONTINUE


class ScriptedOrchestrator:
    control_policy = ContinueControl()

    def transition(self, session, rng, *, allow_valid_refinement: bool = False):
        before = session.snapshot()
        if before.valid:
            assert allow_valid_refinement
        session.index += 1
        after = session.snapshot()
        return SimpleNamespace(
            logical_id=0,
            candidate_index=0,
            candidate_chains=(after.chains[0],),
            accepted=True,
            before=before,
            after=after,
        )


class TerminalValues:
    def score(self, snapshot) -> float:
        return {((0,), (1,)): 0.25, ((0,), (2,)): 0.75}[snapshot.chains]


def _run(*, anytime: bool):
    return run_search(
        ScriptedOrchestrator(),
        ScriptedSession(),
        random.Random(0),
        random_seed=0,
        tries=1,
        max_transitions=2,
        timeout=None,
        anytime=anytime,
        terminal_scorer=TerminalValues(),
    )


def test_first_valid_and_anytime_modes_return_their_documented_incumbents() -> None:
    first_valid = _run(anytime=False)
    anytime = _run(anytime=True)

    assert first_valid.success
    assert first_valid.termination_reason is TerminationReason.SUCCESS
    assert first_valid.transitions == 1
    assert [record.terminal_value for record in first_valid.incumbents] == [0.25]
    assert first_valid.best_incumbent is not None
    assert first_valid.best_incumbent.chains == ((0,), (1,))

    assert anytime.success
    assert anytime.termination_reason is TerminationReason.TRANSITION_LIMIT
    assert anytime.transitions == 2
    assert [record.terminal_value for record in anytime.incumbents] == [0.25, 0.75]
    assert anytime.best_incumbent is not None
    assert anytime.best_incumbent.chains == ((0,), (2,))
    assert anytime.structural_dict() == _run(anytime=True).structural_dict()
