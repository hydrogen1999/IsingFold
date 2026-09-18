"""Outcome-blind, bounded-memory proposal collection before measured selection.

The shortlist is a search ablation, not a prediction of solution quality. It never
samples an annealer, reads a reward, or ranks embeddings by their qubit count.
Every proposer uses the same declared wall-clock proposal budget. The caller must
reserve and charge measured selection against its own overall deployment deadline.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import time

import networkx as nx


SHORTLIST_VERSION = "structural-shortlist-v1"


def chain_assignment_distance(left, right):
    """Mean per-logical-chain Jaccard distance, independent of iteration order.

    A consistent relabeling of logical nodes or hardware qubits leaves the distance
    unchanged. Physical placements remain distinct; this does not identify graph
    automorphisms or exchange the identities of different logical variables.
    """
    if set(left) != set(right):
        raise ValueError("chain assignments must contain the same logical nodes")
    if not left:
        return 0.0
    distances = []
    for v in left:
        a, b = frozenset(left[v]), frozenset(right[v])
        union = a | b
        distances.append(len(a ^ b) / len(union) if union else 0.0)
    # fsum prevents a change of mapping iteration order from changing tie breaks.
    return math.fsum(distances) / len(distances)


def _program_key(terminal):
    return (frozenset((v, frozenset(chain)) for v, chain in terminal.embedding.items()),
            terminal.selected_index)


def _invalid_reason(task, terminal):
    """Independent graph-level gate; compilation remains the proposer's obligation."""
    if terminal is None or not getattr(terminal, "returned_valid", False):
        return "no_valid_terminal"
    embedding = getattr(terminal, "embedding", None)
    if embedding is None or set(embedding) != set(task.logical):
        return "incomplete_assignment"
    program = getattr(terminal, "selected_program", None)
    index = getattr(terminal, "selected_index", None)
    if (program is None or index is None
            or getattr(program, "strength_index", None) != index):
        return "missing_or_mismatched_program"
    occupied = set()
    for chain in embedding.values():
        nodes = set(chain)
        if not nodes:
            return "empty_chain"
        if not nodes.issubset(task.host):
            return "unknown_qubit"
        if nodes & occupied:
            return "overlapping_chains"
        if not nx.is_connected(task.host.subgraph(nodes)):
            return "disconnected_chain"
        occupied.update(nodes)
    for u, v in task.logical.edges():
        if not any(task.host.has_edge(p, q) for p in embedding[u] for q in embedding[v]):
            return "unrealized_logical_edge"
    return None


def _minimum_distance(terminals):
    if len(terminals) < 2:
        return 0.0
    return min(chain_assignment_distance(a.embedding, b.embedding)
               for i, a in enumerate(terminals) for b in terminals[i + 1:])


def _proposal_seed(seed, name, attempt):
    payload = json.dumps([int(seed), "proposal", name, attempt], separators=(",", ":"))
    return int.from_bytes(hashlib.blake2s(payload.encode(), digest_size=4).digest(),
                          "little") % (2 ** 31)


def collect_shortlist(task, propose, *, deadline, capacity=6, seed=0, clock=None,
                      mode="diverse"):
    """Return ``(terminal_tuple, receipt)`` after a declared proposal-duration budget.

    ``propose(seed, seconds_left)`` returns an immutable compiled terminal or None.
    ``deadline`` is a duration in seconds, not an absolute timestamp. All proposals,
    including failed/duplicate/rejected/late proposals, consume this same budget;
    validation and distance computation also count. A call that overruns cannot be
    preempted here, but its answer cannot enter the shortlist and its overrun is logged.

    ``mode='first'`` retains the first capacity distinct valid programs. ``'diverse'``
    additionally accepts a single-member replacement only when it strictly improves
    the minimum pairwise chain-assignment distance. Ties retain earlier proposals;
    capacity=1 therefore keeps the first valid program. This is a streaming heuristic,
    not a globally optimal k-dispersion solver and not a quality guarantee.

    Deduplication is exact against retained programs only. An evicted/rejected program
    can be proposed again, so receipts deliberately do not claim a historical unique
    count. Memory is bounded by capacity plus one transient proposal. Same chains at
    different strength indices are distinct programs but have zero structural distance.
    """
    if isinstance(deadline, bool) or not math.isfinite(deadline) or deadline <= 0:
        raise ValueError("deadline must be a positive finite duration")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
        raise ValueError("capacity must be a positive integer")
    if mode not in {"first", "diverse"}:
        raise ValueError("shortlist mode must be first or diverse")
    clock = time.monotonic if clock is None else clock
    started = clock()
    retained, retained_metadata, invalid_reasons = [], [], Counter()
    exhausted = False
    counts = {"attempts": 0, "constructed_attempts": 0, "invalid_attempts": 0,
              "duplicate_attempts": 0, "shortlist_replacements": 0,
              "shortlist_rejections": 0, "late_attempts": 0}
    while True:
        left = deadline - (clock() - started)
        if left <= 0:
            break
        attempt = counts["attempts"] + 1
        proposal_seed = _proposal_seed(seed, task.name, attempt)
        try:
            terminal = propose(proposal_seed, left)
        except StopIteration:
            exhausted = True
            break
        counts["attempts"] = attempt
        if clock() - started > deadline:
            counts["late_attempts"] += 1
            break
        reason = _invalid_reason(task, terminal)
        if clock() - started > deadline:
            counts["late_attempts"] += 1
            break
        if reason is not None:
            counts["invalid_attempts"] += 1
            invalid_reasons[reason] += 1
            continue
        key = _program_key(terminal)
        duplicate = any(key == _program_key(t) for t in retained)
        proposed, proposed_metadata = retained, retained_metadata
        metadata = {"proposal_attempt": attempt, "proposal_seed": proposal_seed}
        replaced = False
        if not duplicate:
            if len(retained) < capacity:
                proposed = [*retained, terminal]
                proposed_metadata = [*retained_metadata, metadata]
            elif mode == "diverse" and capacity > 1:
                best = _minimum_distance(retained)
                # Reverse order preserves earlier arrivals on equal improvements.
                for i in reversed(range(len(retained))):
                    replacement = retained[:i] + retained[i + 1:] + [terminal]
                    score = _minimum_distance(replacement)
                    if score > best + 1e-12:
                        proposed, best, replaced = replacement, score, True
                        proposed_metadata = (retained_metadata[:i] + retained_metadata[i + 1:]
                                             + [metadata])
        # A candidate whose shortlist processing ran late must not replace an
        # earlier on-time incumbent. Counts still expose that consumed attempt.
        if clock() - started > deadline:
            counts["late_attempts"] += 1
            break
        counts["constructed_attempts"] += 1
        if duplicate:
            counts["duplicate_attempts"] += 1
        elif replaced:
            counts["shortlist_replacements"] += 1
        elif proposed is retained:
            counts["shortlist_rejections"] += 1
        retained, retained_metadata = proposed, proposed_metadata
    elapsed = clock() - started
    receipt = {**counts, "shortlist_version": SHORTLIST_VERSION, "shortlist_mode": mode,
               "shortlist_capacity": capacity, "retained_candidates": len(retained),
               "dedup_scope": "retained_programs_only",
               "proposer_exhausted": exhausted, "retained_metadata": retained_metadata,
               "proposal_seed_schedule": {"scheme": "blake2s-experiment-seed-v1",
                                          "seed": int(seed), "phase": "proposal",
                                          "task": task.name, "attempts": counts["attempts"]},
               "invalid_reasons": dict(invalid_reasons),
               "proposal_deadline_seconds": float(deadline), "proposal_seconds": elapsed,
               "deadline_overrun_seconds": max(0.0, elapsed - deadline),
               "selection_reads": 0, "assessment_reads": 0,
               "validation": "complete_disjoint_connected_chains_and_all_logical_edges",
               "compilation_validation": "proposer_receipt_strength_index",
               "quality_outcomes_used": False}
    return tuple(retained), receipt
