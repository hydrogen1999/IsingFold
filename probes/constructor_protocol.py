"""Measured selection around independently generated constructor outputs.

Both learned and baseline proposers obey this deadline/read protocol. This module
does not import a completion solver; the optional minorminer baseline is separate.
"""
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


def evaluate_search(task, propose, *, deadline, select_cap=6, selection_reads=256,
                    assessment_reads=256, objective="quality", seed=0, measure=None):
    """propose(seed, seconds_left) returns a terminal record or None.

    Proposal includes feature extraction, policy decisions and validation/compilation.
    Late answers are discarded. Reads used by late/failed measurements still count.
    Fresh assessment is reporting-only and cannot change the returned candidate.
    """
    counts = (select_cap, selection_reads, assessment_reads)
    if (not np.isfinite(deadline) or deadline <= 0
            or not all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in counts)):
        raise ValueError("positive finite deadline and read/selection caps are required")
    if objective not in ("quality", "feasibility"):
        raise ValueError("unknown evaluation objective")
    measure = measure_terminal if measure is None else measure
    started = time.monotonic()
    seen, chosen, best = set(), None, float("inf")
    attempts = constructed = selection_used = measurements = measurement_failures = 0
    while time.monotonic() - started < deadline:
        attempts += 1
        left = deadline - (time.monotonic() - started)
        if left <= 0:
            break
        terminal = propose(experiment_seed(seed, "proposal", task.name, attempts), left)
        if time.monotonic() - started > deadline:
            break
        if terminal is None or not terminal.returned_valid:
            continue
        constructed += 1
        # Same chains at a different selected strength are distinct programs. The
        # compiler/evaluator context is fixed for every candidate in this call.
        key = (frozenset((v, frozenset(c)) for v, c in terminal.embedding.items()),
               terminal.selected_index)
        if key in seen:
            continue
        seen.add(key)
        if objective == "feasibility":
            chosen = terminal
            break
        selection_used += selection_reads
        residual = measure(task, terminal,
                           experiment_seed(seed, "selection", task.name, measurements), selection_reads)
        measurements += 1
        if residual is None:
            measurement_failures += 1
        elif not np.isfinite(residual) or residual < 0:
            raise ValueError("nonfinite/negative selection residual")
        if time.monotonic() - started > deadline:
            break
        if residual is not None and residual < best:
            best, chosen = residual, terminal
        if measurements >= select_cap:
            break
    deployment_seconds = time.monotonic() - started
    fresh, assessment_used = None, 0
    if chosen is not None and objective == "quality":
        assessment_used = assessment_reads
        fresh = measure(task, chosen, experiment_seed(seed, "assessment", task.name), assessment_reads)
        if fresh is not None and (not np.isfinite(fresh) or fresh < 0):
            raise ValueError("nonfinite/negative assessment residual")
    return {"valid": chosen is not None, "residual": fresh, "attempts": attempts,
            "constructed_attempts": constructed, "unique_candidates": len(seen),
            "selection_reads": selection_used, "assessment_reads": assessment_used,
            "measurement_failures": measurement_failures,
            "deployment_seconds": deployment_seconds,
            "total_seconds": time.monotonic() - started,
            "deadline_overrun_seconds": max(0.0, deployment_seconds - deadline)}
