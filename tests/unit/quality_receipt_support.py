"""Small deterministic continuation receipts for quality-artifact unit tests."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Hashable

from isingfold.rl.contracts import WorkVector, stable_digest
from isingfold.rl.data.quality import (
    CONTINUATION_POLICY_ID,
    CONTINUATION_RECEIPT_SCHEMA,
    CONTINUATION_RECEIPT_VERSION,
    ContinuationResult,
    legacy_quality_initializer_binding,
)


def fake_continuation_result(
    *,
    continuation_seed: int,
    reward: float,
    reward_reads: int,
    prefix: Sequence[int],
    chains: Mapping[Hashable, frozenset[Hashable]] | None,
    returned_valid: bool = True,
    selected_index: int = 0,
    num_sweeps: int = 1_000,
    steps: int = 1,
) -> ContinuationResult:
    """Build a schema-complete fake while production remains strict about missing receipts."""

    reward = float(reward if returned_valid else 0.0)
    hits = round(reward * reward_reads)
    if not math.isclose(hits / reward_reads, reward, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError("test reward must be exactly representable by the read denominator")
    trace = [
        {
            "position": position,
            "source": "forced-prefix",
            "action_index": action,
            "opcode": "TEST",
            "payload_key": f"test-action-{action}",
            "state_fingerprint": f"test-state-{position}",
            "support_fingerprint": f"test-support-{position}",
        }
        for position, action in enumerate(prefix)
    ]
    evidence = None
    if returned_valid:
        programs = [
            {
                "strength": float(index + 1),
                "strength_index": index,
                "scale": 1.0,
                "offset": 0.0,
                "h_phys": [],
                "j_phys": [],
                "chain_edges": [],
                "contact_counts": [],
            }
            for index in range(4)
        ]
        embedding: list[dict[str, object]] = []
        evidence = {
            "embedding": embedding,
            "programs": programs,
            "selected_index": selected_index,
            "selected_strength": float(selected_index + 1),
            "selected_program_digest": stable_digest(
                {"program": programs[selected_index], "embedding": embedding}
            ),
            "evaluator_count_block": {
                "seed": continuation_seed % (2**31),
                "hits": hits,
                "reads": reward_reads,
                "num_sweeps": num_sweeps,
            },
        }
    payload: dict[str, object] = {
        "schema": CONTINUATION_RECEIPT_SCHEMA,
        "schema_version": CONTINUATION_RECEIPT_VERSION,
        "continuation_seed": continuation_seed,
        "continuation_policy": CONTINUATION_POLICY_ID,
        "continuation_steps": steps,
        "action_trace": trace,
        "returned_valid": returned_valid,
        "terminal_reason": "COMMIT" if returned_valid else "STOP_NO_VALID",
        "reward": reward,
        "requested_reward_reads": reward_reads,
        "validation_receipt": {"valid": returned_valid},
        "cumulative_work": WorkVector().as_dict(),
        "terminal_evidence": evidence,
        "initializer_binding": legacy_quality_initializer_binding(),
    }
    receipt = {**payload, "record_digest": stable_digest(payload)}
    return ContinuationResult(
        returned_valid=returned_valid,
        reward=reward,
        embedding=chains,
        selected_index=selected_index if returned_valid else None,
        steps=steps,
        receipt=receipt,
    )


__all__ = ["fake_continuation_result"]
