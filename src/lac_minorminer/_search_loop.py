"""Hard-bounded main loop kept separate from replaceable operations."""

from __future__ import annotations

import time
from collections import Counter
import math
from numbers import Real
from typing import Any

from . import _core
from ._version import __version__
from .components import ControlContext, ControlDecision
from .diagnostics import (
    IncumbentRecord,
    RepairRecord,
    SearchDiagnostics,
    TerminationReason,
    TransitionRecord,
    WORK_COUNTER_FIELDS,
)


def _state_key(snapshot: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(chain) for chain in snapshot.chains)


def _work_budget_coordinate(error: BaseException) -> str:
    prefix = "native work budget exhausted: "
    message = str(error)
    if not message.startswith(prefix) or message[len(prefix) :] not in WORK_COUNTER_FIELDS:
        raise RuntimeError("native work-budget exception omitted a registered coordinate") from error
    return message[len(prefix) :]


def run_search(
    orchestrator,
    session,
    rng,
    *,
    random_seed: int,
    tries: int,
    max_transitions: int,
    timeout: float | None,
    anytime: bool = False,
    terminal_scorer=None,
) -> SearchDiagnostics:
    if not isinstance(anytime, bool):
        raise ValueError("anytime must be a boolean")
    if anytime and terminal_scorer is None:
        raise ValueError("anytime mode requires a terminal_scorer")
    if terminal_scorer is not None and not callable(getattr(terminal_scorer, "score", None)):
        raise TypeError("terminal_scorer must provide score(snapshot)")

    started = time.monotonic()
    trace: list[TransitionRecord] = []
    repair_trace: list[RepairRecord] = []
    incumbents: list[IncumbentRecord] = []
    incumbent_states: set[tuple[tuple[int, ...], ...]] = set()
    best_incumbent_index: int | None = None
    transitions = proposals = applied = discarded = 0
    attempts = 0
    reason = TerminationReason.TRIES_EXHAUSTED
    success = False
    terminal = False
    work_budget_coordinate: str | None = None

    def retain_incumbent(snapshot: Any, attempt: int, transition: int) -> None:
        nonlocal best_incumbent_index, success
        key = _state_key(snapshot)
        if key in incumbent_states:
            return
        terminal_value = None
        if terminal_scorer is not None:
            raw_value = terminal_scorer.score(snapshot)
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, Real)
                or not math.isfinite(raw_value)
            ):
                raise ValueError("terminal_scorer must return one finite real value")
            terminal_value = float(raw_value)
        incumbent_states.add(key)
        record = IncumbentRecord(
            attempt=attempt,
            transition=transition,
            generation=snapshot.generation,
            terminal_value=terminal_value,
            chains=key,
            used_target_nodes=snapshot.used_target_nodes,
            longest_chain=max((len(chain) for chain in key), default=0),
            total_chain_length=sum(len(chain) for chain in key),
        )
        incumbents.append(record)
        success = True
        if best_incumbent_index is None:
            best_incumbent_index = 0
        elif terminal_value is not None:
            current = incumbents[best_incumbent_index].terminal_value
            if current is None or terminal_value > current:
                best_incumbent_index = len(incumbents) - 1

    for attempt in range(1, tries + 1):
        attempts = attempt
        local_repairs = 0
        reason = TerminationReason.TRIES_EXHAUSTED
        if attempt > 1:
            try:
                session.restart()
            except _core.WorkBudgetExceeded as error:
                work_budget_coordinate = _work_budget_coordinate(error)
                reason, terminal = TerminationReason.WORK_BUDGET_EXHAUSTED, True
                break
        try:
            initial_snapshot = session.snapshot()
        except _core.WorkBudgetExceeded as error:
            work_budget_coordinate = _work_budget_coordinate(error)
            reason, terminal = TerminationReason.WORK_BUDGET_EXHAUSTED, True
            break
        visits = Counter({_state_key(initial_snapshot): 1})

        while True:
            try:
                snapshot = session.snapshot()
            except _core.WorkBudgetExceeded as error:
                work_budget_coordinate = _work_budget_coordinate(error)
                reason, terminal = TerminationReason.WORK_BUDGET_EXHAUSTED, True
                break
            if timeout is not None and time.monotonic() - started >= timeout:
                reason, terminal = TerminationReason.TIMEOUT, True
                break
            if snapshot.valid:
                retain_incumbent(snapshot, attempt, transitions)
                if not anytime:
                    reason, terminal = TerminationReason.SUCCESS, True
                    break
            if transitions >= max_transitions:
                reason, terminal = TerminationReason.TRANSITION_LIMIT, True
                break
            eligible = session.eligible_variables()
            can_refine_valid = anytime and snapshot.valid and bool(snapshot.chains)
            if not eligible and not can_refine_valid:
                reason = (
                    TerminationReason.SUCCESS if snapshot.valid else TerminationReason.NO_CANDIDATE
                )
                terminal = snapshot.valid
                break

            try:
                if anytime and snapshot.valid:
                    outcome = orchestrator.transition(session, rng, allow_valid_refinement=True)
                else:
                    outcome = orchestrator.transition(session, rng)
            except _core.WorkBudgetExceeded as error:
                work_budget_coordinate = _work_budget_coordinate(error)
                reason, terminal = TerminationReason.WORK_BUDGET_EXHAUSTED, True
                break
            transitions += 1
            proposals += 1
            applied += int(outcome.accepted)
            discarded += int(not outcome.accepted)
            selected_chain = (
                outcome.candidate_chains[outcome.candidate_index]
                if outcome.candidate_index is not None
                else None
            )
            trace.append(
                TransitionRecord(
                    attempt,
                    transitions,
                    outcome.logical_id,
                    outcome.candidate_index,
                    outcome.accepted,
                    outcome.before.generation,
                    outcome.after.generation,
                    outcome.after.max_occupancy,
                    outcome.after.total_excess_occupancy,
                    outcome.after.missing_source_edges,
                    selected_chain,
                    getattr(outcome, "policy_mode", None),
                    getattr(outcome, "policy_reason", None),
                )
            )
            if timeout is not None and time.monotonic() - started >= timeout:
                reason, terminal = TerminationReason.TIMEOUT, True
                break
            if outcome.after.valid:
                retain_incumbent(outcome.after, attempt, transitions)
                if not anytime:
                    reason, terminal = TerminationReason.SUCCESS, True
                    break

            key = _state_key(outcome.after)
            visits[key] += 1
            decision = orchestrator.control_policy.decide(
                ControlContext(
                    attempt,
                    transitions,
                    visits[key],
                    outcome.accepted,
                    bool(outcome.candidate_chains),
                )
            )
            if not isinstance(decision, ControlDecision):
                raise ValueError("control policy must return a ControlDecision")
            if decision is ControlDecision.STOP:
                reason, terminal = TerminationReason.CONTROL_STOP, True
                break
            wants_repair = decision is ControlDecision.RESTART or not outcome.candidate_chains
            if wants_repair and local_repairs < orchestrator.max_local_repairs:
                before_repair = outcome.after
                try:
                    logical_ids = orchestrator.repair(session, before_repair, rng)
                    after_repair = session.snapshot()
                except _core.WorkBudgetExceeded as error:
                    work_budget_coordinate = _work_budget_coordinate(error)
                    reason, terminal = TerminationReason.WORK_BUDGET_EXHAUSTED, True
                    break
                local_repairs += 1
                repair_trace.append(
                    RepairRecord(
                        attempt=attempt,
                        repair=len(repair_trace) + 1,
                        after_transition=transitions,
                        logical_ids=logical_ids,
                        before_generation=before_repair.generation,
                        after_generation=after_repair.generation,
                        before_max_occupancy=before_repair.max_occupancy,
                        after_max_occupancy=after_repair.max_occupancy,
                        before_total_excess_occupancy=before_repair.total_excess_occupancy,
                        after_total_excess_occupancy=after_repair.total_excess_occupancy,
                        before_missing_source_edges=before_repair.missing_source_edges,
                        after_missing_source_edges=after_repair.missing_source_edges,
                    )
                )
                visits = Counter({_state_key(after_repair): 1})
                continue
            if not outcome.candidate_chains:
                reason = TerminationReason.NO_CANDIDATE
                break
            if decision is ControlDecision.RESTART:
                break
        if terminal:
            break

    work = session.work_counters()
    if work.decisions != transitions:
        raise RuntimeError("native decision work disagrees with the consumed transition count")
    return SearchDiagnostics(
        success,
        reason,
        random_seed,
        attempts,
        transitions,
        proposals,
        applied,
        discarded,
        time.monotonic() - started,
        __version__,
        str(_core.backend_info()["backend"]),
        tuple(trace),
        tuple(incumbents),
        best_incumbent_index,
        anytime,
        tuple(repair_trace),
        work,
        work_cap=getattr(session, "work_cap", None),
        work_budget_exhausted_coordinate=work_budget_coordinate,
    )
