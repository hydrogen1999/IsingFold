"""Measured selection around independently generated constructor outputs.

Both learned and baseline proposers obey this deadline/read protocol. This module
does not import a completion solver; the optional minorminer baseline is separate.
"""
from collections.abc import Mapping

import hashlib
import json
import time

import numpy as np

from constructor_objective import measure_terminal


def experiment_seed(seed, phase, *parts):
    payload = json.dumps([int(seed), phase, *parts], separators=(",", ":"))
    return int.from_bytes(hashlib.blake2s(payload.encode(), digest_size=4).digest(), "little") % (2 ** 31)


def split_by_lineage(tasks, fraction, seed):
    if not 0 < fraction < 1:
        raise ValueError("holdout fraction must lie between zero and one")
    groups = sorted({t.lineage or t.name for t in tasks})
    if len(groups) < 2:
        raise ValueError("at least two independent lineages are required")
    size = min(len(groups) - 1, max(1, int(len(groups) * fraction)))
    held = set(np.random.default_rng(seed).choice(groups, size=size, replace=False))
    return ([t for t in tasks if (t.lineage or t.name) not in held],
            [t for t in tasks if (t.lineage or t.name) in held])


def checkpoint_key(records, objective):
    if not records:
        raise ValueError("validation records cannot be empty")
    valid = [r for r in records if r["valid"]]
    coverage = len(valid) / len(records)
    if objective == "feasibility":
        return (coverage,)
    measured = [r["residual"] for r in valid if r["residual"] is not None]
    return (coverage, len(measured) / len(records),
            -float(np.mean(measured)) if measured else -float("inf"))


def _measurement_record(value, terminal, reads):
    """Normalize legacy residual callbacks or one paired read-block receipt.

    A mapping must report both metrics from the same call, plus its read/strength
    receipts. Scalar callbacks remain supported but cannot certify p_solve or
    actual read counts; requested reads are still conservatively charged.
    """
    if value is None:
        return {"residual": None, "p_solve": None, "reads_verified": 0}
    paired = isinstance(value, Mapping)
    if paired:
        required = {"residual", "p_solve", "reads", "strength_index"}
        if not required.issubset(value):
            raise ValueError("paired measurement requires residual/p_solve/read/strength receipts")
        if value["reads"] != reads or value["strength_index"] != terminal.selected_index:
            raise ValueError("paired measurement violated read/strength receipts")
        residual, p_solve = value["residual"], value["p_solve"]
        if residual is None or p_solve is None:
            raise ValueError("paired measurements must contain both finite metrics; use None for failure")
    else:
        residual, p_solve = value, None
    residual = float(residual)
    if not np.isfinite(residual) or residual < 0:
        raise ValueError("nonfinite/negative measurement residual")
    if paired:
        p_solve = float(p_solve)
        if not np.isfinite(p_solve) or not 0 <= p_solve <= 1:
            raise ValueError("measurement p_solve must be finite and in [0,1]")
    return {"residual": residual, "p_solve": p_solve,
            "reads_verified": reads if paired else 0}


def evaluate_search(task, propose, *, deadline, select_cap=6, selection_reads=256,
                    assessment_reads=256, objective="quality", seed=0, measure=None,
                    utility=None, reserved_seeds=(), select_measure=None):
    """Generate from scratch, select on pilot residual, independently assess once.

    ``propose(seed, seconds_left)`` returns a terminal record or None. Proposal
    includes feature extraction, policy decisions and validation/compilation.
    Late answers are discarded; reads requested for late/failed calls still count.
    ``measure`` may return a scalar residual (legacy), None (failure), or a mapping
    with residual, p_solve, reads and strength_index from one sampling block.
    ``select_measure`` optionally provides a separate public-information-only
    scalar selection score (nonnegative, lower is better). It defaults to
    ``measure`` for compatibility. Fresh assessment is reporting-only and cannot
    change the selected candidate.
    ``utility(task, residual)`` optionally defines the prespecified quality utility;
    failed construction/assessment always receives zero, separately from coverage.
    """
    counts = (select_cap, selection_reads, assessment_reads)
    if (not np.isfinite(deadline) or deadline <= 0
            or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in counts)):
        raise ValueError("positive finite deadline and read/selection caps are required")
    if objective not in ("quality", "feasibility"):
        raise ValueError("unknown evaluation objective")
    measure = measure_terminal if measure is None else measure
    legacy_selection = select_measure is None
    select_measure = measure if select_measure is None else select_measure
    score_semantics = "normalized_residual" if legacy_selection else "callback_lower_is_better"
    started = time.monotonic()
    seen, chosen, best = set(), None, float("inf")
    chosen_metadata = None
    used_seeds = set(reserved_seeds)

    def fresh_seed(phase, *parts):
        # Phase hashing alone has a nonzero collision probability in a 31-bit
        # sampler seed space. Resolve collisions explicitly within each search.
        result = experiment_seed(seed, phase, task.name, *parts)
        while result in used_seeds:
            result = (result + 1) % (2 ** 31)
        used_seeds.add(result)
        return result

    attempts = constructed = selection_used = measurements = measurement_failures = 0
    proposal_seeds, candidates, selection_receipts = [], [], []
    while time.monotonic() - started < deadline:
        left = deadline - (time.monotonic() - started)
        if left <= 0:
            break
        attempts += 1
        proposal_seed = fresh_seed("proposal", attempts)
        proposal_seeds.append(proposal_seed)
        try:
            terminal = propose(proposal_seed, left)
        except StopIteration:
            # A finite shortlist is exhausted; there is no reason to busy-wait.
            attempts -= 1
            proposal_seeds.pop()
            break
        if time.monotonic() - started > deadline:
            break
        if terminal is None or not terminal.returned_valid:
            continue
        constructed += 1
        # Distinct strengths of the same chains are distinct compiled programs.
        key = (frozenset((v, frozenset(c)) for v, c in terminal.embedding.items()),
               terminal.selected_index)
        if key in seen:
            continue
        seen.add(key)
        metadata = {"candidate_id": len(candidates), "proposal_attempt": attempts,
                    "proposal_seed": proposal_seed, "strength_index": terminal.selected_index,
                    "qubits": sum(len(c) for c in terminal.embedding.values()),
                    "longest_chain": max((len(c) for c in terminal.embedding.values()), default=0)}
        candidates.append(metadata)
        if objective == "feasibility":
            chosen, chosen_metadata = terminal, metadata
            break
        selection_used += selection_reads
        selection_seed = fresh_seed("selection", measurements)
        measured = _measurement_record(select_measure(task, terminal, selection_seed, selection_reads),
                                       terminal, selection_reads)
        measurements += 1
        on_time = time.monotonic() - started <= deadline
        selection_receipts.append({"candidate_id": metadata["candidate_id"],
                                   "seed": selection_seed, "reads_requested": selection_reads,
                                   "on_time": on_time, **measured,
                                   "score": measured["residual"], "score_semantics": score_semantics,
                                   "residual": measured["residual"] if legacy_selection else None})
        residual = measured["residual"]
        if residual is None:
            measurement_failures += 1
        if not on_time:
            break
        if residual is not None and residual < best:
            best, chosen, chosen_metadata = residual, terminal, metadata
        if measurements >= select_cap:
            break
    deployment_seconds = time.monotonic() - started
    fresh = {"residual": None, "p_solve": None, "reads_verified": 0}
    assessment_used, assessment_seed, assessment_calls = 0, None, 0
    if chosen is not None and objective == "quality":
        assessment_used, assessment_calls = assessment_reads, 1
        assessment_seed = fresh_seed("assessment")
        fresh = _measurement_record(measure(task, chosen, assessment_seed, assessment_reads),
                                    chosen, assessment_reads)
    assessment_ok = fresh["residual"] is not None
    quality_utility = None
    if objective == "quality":
        if not assessment_ok:
            quality_utility = 0.0
        elif utility is not None:
            quality_utility = float(utility(task, fresh["residual"]))
            if not np.isfinite(quality_utility):
                raise ValueError("quality utility must be finite")
    assessment_failures = int(assessment_calls > 0 and not assessment_ok)
    return {"valid": chosen is not None, "construction_valid": constructed > 0,
            "residual": fresh["residual"], "p_solve": fresh["p_solve"],
            "quality_utility": quality_utility, "assessment_ok": assessment_ok,
            "attempts": attempts, "constructed_attempts": constructed,
            "unique_candidates": len(seen), "selection_calls": measurements,
            "assessment_calls": assessment_calls, "selection_reads": selection_used,
            "assessment_reads": assessment_used, "total_reads": selection_used + assessment_used,
            "selection_reads_verified": sum(r["reads_verified"] for r in selection_receipts),
            "assessment_reads_verified": fresh["reads_verified"],
            "measurement_failures": measurement_failures,
            "selection_failures": measurement_failures, "assessment_failures": assessment_failures,
            "chosen_candidate": chosen_metadata,
            "chosen_qubits": chosen_metadata["qubits"] if chosen_metadata else None,
            "chosen_longest_chain": chosen_metadata["longest_chain"] if chosen_metadata else None,
            "candidate_qubits_mean": float(np.mean([c["qubits"] for c in candidates])) if candidates else None,
            "selection_score": best if chosen is not None and objective == "quality" else None,
            "selection_score_semantics": score_semantics,
            "selection_residual": best if legacy_selection and chosen is not None and objective == "quality" else None,
            "selection_receipts": selection_receipts, "assessment_seed": assessment_seed,
            "proposal_seeds": proposal_seeds,
            "budget": {"deadline_seconds": deadline, "select_cap": select_cap,
                       "selection_reads_per_call": selection_reads,
                       "assessment_reads_per_call": assessment_reads},
            "deployment_seconds": deployment_seconds,
            "total_seconds": time.monotonic() - started,
            "deadline_overrun_seconds": max(0.0, deployment_seconds - deadline)}


def evaluate_anytime_search(task, propose, *, deadline, select_cap=6, selection_reads=256,
                            assessment_reads=256, objective="quality", seed=0, measure=None,
                            utility=None, proposal_fraction=.5, shortlist_mode="diverse",
                            select_measure=None):
    """Use a declared proposal-time allocation before capped measured selection.

    Structural shortlisting spends no quality reads. All arms receive the same
    total deployment deadline, proposal fraction and pilot/assessment read caps.
    Only the retained shortlist is sampled; the fresh assessment stays outside
    deployment time and can never replace the selected candidate. This protocol
    enables faster proposers to explore more candidates, without treating the
    structural diversity filter as an oracle for downstream quality.
    """
    from constructor_shortlist import collect_shortlist

    counts = (select_cap, selection_reads, assessment_reads)
    if (not np.isfinite(deadline) or deadline <= 0
            or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in counts)):
        raise ValueError("positive finite deadline and read/selection caps are required")
    if not np.isfinite(proposal_fraction) or not 0 < proposal_fraction < 1:
        raise ValueError("proposal_fraction must be strictly between zero and one")
    if objective != "quality":
        raise ValueError("anytime shortlisting is a quality-evaluation protocol")
    started = time.monotonic()
    candidates, collection = collect_shortlist(task, propose, deadline=deadline * proposal_fraction,
                                               capacity=select_cap, seed=seed, clock=time.monotonic,
                                               mode=shortlist_mode)
    remaining = deadline - (time.monotonic() - started)
    # Empty and overrun pools must exit immediately, without a deadline busy-wait
    # or a late first pilot. The epsilon only satisfies evaluate_search's contract.
    eligible = candidates if remaining > 0 else ()
    iterator = iter(eligible)

    def retrieve(_seed, _seconds_left):
        return next(iterator)

    result = evaluate_search(task, retrieve, deadline=max(remaining, 1e-12),
                             select_cap=select_cap, selection_reads=selection_reads,
                             assessment_reads=assessment_reads, objective=objective,
                             seed=seed, measure=measure, utility=utility, select_measure=select_measure,
                             reserved_seeds=[m["proposal_seed"] for m in collection["retained_metadata"]])
    # Preserve the acquisition work, rather than reporting cheap shortlist reads
    # as if they were actual embedding-construction attempts.
    result["shortlist_retrieval_seeds"] = result["proposal_seeds"]
    result["proposal_seeds"] = [m["proposal_seed"] for m in collection["retained_metadata"]]
    result["proposal_seeds_scope"] = "retained_shortlist"
    result["proposal_seed_schedule"] = collection["proposal_seed_schedule"]
    result["shortlist_retrieval_attempts"] = result["attempts"]
    result["attempts"] = collection["attempts"]
    result["constructed_attempts"] = collection["constructed_attempts"]
    result["construction_valid"] = collection["constructed_attempts"] > 0
    result["unique_candidates_scope"] = "measured_shortlist"
    result["retained_candidates"] = len(candidates)
    result["candidate_qubits_mean"] = (float(np.mean([sum(len(c) for c in t.embedding.values())
                                                     for t in candidates])) if candidates else None)
    result["candidate_qubits_mean_scope"] = "retained_shortlist"
    chosen = result["chosen_candidate"]
    if chosen is not None:
        origin = collection["retained_metadata"][chosen["candidate_id"]]
        chosen.update({"shortlist_rank": chosen["candidate_id"],
                       "proposal_attempt": origin["proposal_attempt"],
                       "proposal_seed": origin["proposal_seed"]})
    result["proposal_collection"] = collection
    result["protocol"] = "deadline_shortlist_then_measured_selection"
    # evaluate_search's deployment stopwatch begins after collection. Its total
    # additionally contains the independent, reporting-only assessment block.
    acquisition_seconds = deadline - remaining
    result["selection_stage_seconds"] = result["deployment_seconds"]
    result["deployment_seconds"] += acquisition_seconds
    result["total_seconds"] = time.monotonic() - started
    result["deadline_overrun_seconds"] = max(0., result["deployment_seconds"] - deadline)
    result["budget"].update({"deadline_seconds": deadline,
                              "proposal_fraction": proposal_fraction,
                              "proposal_deadline_seconds": deadline * proposal_fraction,
                              "selection_seconds_available": max(0., remaining),
                              "shortlist_mode": shortlist_mode})
    return result
