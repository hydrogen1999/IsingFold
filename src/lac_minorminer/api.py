"""Public Minorminer-shaped convenience API."""

from __future__ import annotations

import random
import time
from dataclasses import replace

from ._chimera_clique import find_chimera_clique_fallback
from ._core import backend_info
from ._graph_input import normalize_graph
from ._options import integer_option, seed_option, timeout_option
from ._version import __version__
from .components import ViolationNeighborhoodSelector
from .diagnostics import (
    IncumbentRecord,
    SearchDiagnostics,
    SearchWorkCounters,
    TerminationReason,
)
from .orchestrator import SearchOrchestrator
from .session import SearchSession
from .validation import validate_embedding


def find_embedding(
    source,
    target,
    *,
    random_seed: int | None = None,
    timeout: float | None = None,
    tries: int = 10,
    max_transitions: int = 10_000,
    max_candidates: int = 8,
    scorer=None,
    anytime: bool = False,
    terminal_scorer=None,
    search_profile: str = "v0",
    work_cap: SearchWorkCounters | None = None,
    return_diagnostics: bool = False,
):
    started = time.monotonic()
    resolved_seed = seed_option(random_seed, allow_none=True)
    timeout = timeout_option(timeout)
    tries = integer_option("tries", tries, minimum=1)
    max_transitions = integer_option("max_transitions", max_transitions, minimum=0)
    max_candidates = integer_option("max_candidates", max_candidates, minimum=1)
    if not isinstance(return_diagnostics, bool):
        raise ValueError("return_diagnostics must be a boolean")
    if not isinstance(anytime, bool):
        raise ValueError("anytime must be a boolean")
    if anytime and terminal_scorer is None:
        raise ValueError("anytime mode requires a terminal_scorer")
    supported_profiles = {"v0", "lns_v1", "hybrid_chimera_clique_v1"}
    if not isinstance(search_profile, str) or search_profile not in supported_profiles:
        raise ValueError(
            "search_profile must be 'v0', 'lns_v1', or "
            "'hybrid_chimera_clique_v1'"
        )
    if search_profile == "hybrid_chimera_clique_v1" and anytime:
        raise ValueError("hybrid_chimera_clique_v1 does not support anytime mode")
    if work_cap is not None and not isinstance(work_cap, SearchWorkCounters):
        raise TypeError("work_cap must be a SearchWorkCounters instance or None")

    source_graph = normalize_graph(source)
    target_graph = normalize_graph(target)
    session = SearchSession(
        source_graph,
        target_graph,
        random_seed=resolved_seed,
        max_candidates=max_candidates,
        work_cap=work_cap,
    )
    if source_graph.labels and not target_graph.labels:
        diagnostics = SearchDiagnostics(
            False,
            TerminationReason.EMPTY_TARGET,
            resolved_seed,
            0,
            0,
            0,
            0,
            0,
            time.monotonic() - started,
            __version__,
            str(backend_info()["backend"]),
            (),
            work_cap=work_cap,
            search_profile=search_profile,
            profile_detail="empty_target",
        )
        return ({}, diagnostics) if return_diagnostics else {}

    if search_profile == "lns_v1":
        orchestrator = SearchOrchestrator(
            scorer=scorer,
            repair_selector=ViolationNeighborhoodSelector(max_size=4),
            max_local_repairs=15,
        )
    else:
        orchestrator = SearchOrchestrator(scorer=scorer)
    remaining_timeout = (
        None if timeout is None else max(0.0, timeout - (time.monotonic() - started))
    )
    diagnostics = orchestrator.run(
        session,
        random.Random(resolved_seed),
        random_seed=resolved_seed,
        tries=tries,
        max_transitions=max_transitions,
        timeout=remaining_timeout,
        anytime=anytime,
        terminal_scorer=terminal_scorer,
    )
    if search_profile == "hybrid_chimera_clique_v1":
        if diagnostics.success:
            diagnostics = replace(
                diagnostics,
                search_profile=search_profile,
                profile_detail="v0_success",
            )
        elif diagnostics.termination_reason in {
            TerminationReason.TIMEOUT,
            TerminationReason.WORK_BUDGET_EXHAUSTED,
        }:
            diagnostics = replace(
                diagnostics,
                search_profile=search_profile,
                profile_detail=f"v0_{diagnostics.termination_reason.value}",
            )
        else:
            deadline = None if timeout is None else started + timeout
            structural = find_chimera_clique_fallback(
                source_graph,
                target_graph,
                random_seed=resolved_seed,
                initial_work=diagnostics.work,
                work_cap=work_cap,
                deadline=deadline,
            )
            diagnostics = replace(diagnostics, structural_fallback_invoked=True)
            if structural.budget_exhausted_coordinate is not None:
                diagnostics = replace(
                    diagnostics,
                    success=False,
                    termination_reason=TerminationReason.WORK_BUDGET_EXHAUSTED,
                    best_incumbent_index=None,
                    work=structural.work,
                    work_budget_exhausted_coordinate=(
                        structural.budget_exhausted_coordinate
                    ),
                    search_profile=search_profile,
                    profile_detail="v0_failed_chimera_clique_work_budget_exhausted",
                )
            elif structural.timed_out:
                diagnostics = replace(
                    diagnostics,
                    success=False,
                    termination_reason=TerminationReason.TIMEOUT,
                    best_incumbent_index=None,
                    work=structural.work,
                    search_profile=search_profile,
                    profile_detail="v0_failed_chimera_clique_timeout",
                )
            elif structural.chains is not None:
                chains = structural.chains
                generation = 1 + max(
                    (
                        *(record.after_generation for record in diagnostics.trace),
                        *(record.generation for record in diagnostics.incumbents),
                        0,
                    )
                )
                incumbent = IncumbentRecord(
                    attempt=max(1, diagnostics.attempts),
                    transition=diagnostics.transitions,
                    generation=generation,
                    terminal_value=None,
                    chains=chains,
                    used_target_nodes=len({node for chain in chains for node in chain}),
                    longest_chain=max(map(len, chains), default=0),
                    total_chain_length=sum(map(len, chains)),
                )
                diagnostics = replace(
                    diagnostics,
                    success=True,
                    termination_reason=TerminationReason.SUCCESS,
                    incumbents=diagnostics.incumbents + (incumbent,),
                    best_incumbent_index=len(diagnostics.incumbents),
                    work=structural.work,
                    search_profile=search_profile,
                    profile_detail="v0_failed_chimera_clique_success",
                )
            else:
                prefix = (
                    "v0_failed_chimera_clique"
                    if structural.applicable
                    else "v0_failure_chimera_clique_inapplicable"
                )
                diagnostics = replace(
                    diagnostics,
                    work=structural.work,
                    search_profile=search_profile,
                    profile_detail=f"{prefix}_{structural.detail}",
                )
    else:
        diagnostics = replace(
            diagnostics,
            search_profile=search_profile,
            profile_detail=search_profile,
        )
    best = diagnostics.best_incumbent
    embedding = (
        {
            logical: [target_graph.labels[target_id] for target_id in best.chains[logical_id]]
            for logical_id, logical in enumerate(source_graph.labels)
        }
        if diagnostics.success and best is not None
        else {}
    )
    if diagnostics.success:
        if work_cap is not None and diagnostics.work.validator_calls >= work_cap.validator_calls:
            diagnostics = replace(
                diagnostics,
                success=False,
                termination_reason=TerminationReason.WORK_BUDGET_EXHAUSTED,
                best_incumbent_index=None,
                work_budget_exhausted_coordinate="validator_calls",
                profile_detail=f"{diagnostics.profile_detail}_validator_budget_exhausted",
            )
            diagnostics = replace(diagnostics, elapsed_seconds=time.monotonic() - started)
            return ({}, diagnostics) if return_diagnostics else {}
        report = validate_embedding(source_graph, target_graph, embedding)
        diagnostics = replace(
            diagnostics,
            work=diagnostics.work.plus(validator_calls=1),
        )
        if not report.valid:
            raise RuntimeError(
                f"native validity disagreed with independent validator: {report.errors}"
            )
    diagnostics = replace(diagnostics, elapsed_seconds=time.monotonic() - started)
    return (embedding, diagnostics) if return_diagnostics else embedding
