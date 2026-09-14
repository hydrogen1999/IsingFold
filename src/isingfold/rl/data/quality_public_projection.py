"""Authority-neutral v2 projections of structural action certificates.

The v1 envelope and applied-action digests intentionally include provenance authority.  Planning
cannot know evaluator certificates, while quality execution must bind them.  Comparing those v1
digests across the boundary is therefore invalid.  These v2 projections retain every
deployment-visible task, state, support, action, successor, and work field while excluding only
the authority-dependent links.
"""

from __future__ import annotations

from typing import Any

from isingfold.rl.data.action_certificate import (
    AppliedStateActionV1,
    StateActionEnvelopeV1,
)
from isingfold.rl.data.import_embedbench import content_digest


QUALITY_PUBLIC_ENVELOPE_PROJECTION_SCHEMA = (
    "isingfold.quality-public-envelope-projection"
)
QUALITY_PUBLIC_APPLIED_PROJECTION_SCHEMA = (
    "isingfold.quality-public-applied-action-projection"
)
QUALITY_PUBLIC_PROJECTION_VERSION = 2

_ENVELOPE_PUBLIC_FIELDS = (
    "base_lineage",
    "candidates",
    "charged_work_receipt",
    "context_version",
    "mode",
    "state_chains",
    "state_fingerprint",
    "support_fingerprint",
    "task_fingerprint",
    "task_id",
)
_APPLIED_PUBLIC_FIELDS = (
    "chains_after",
    "chains_before",
    "persistence_semantics",
    "persistent_cores",
    "selected_index",
    "selected_opcode",
    "selected_payload_digest",
    "selected_payload_key",
    "state_fingerprint",
    "successor_fingerprint",
    "support_fingerprint",
    "task_fingerprint",
)


def public_envelope_projection(envelope: StateActionEnvelopeV1) -> dict[str, Any]:
    """Return the exact authority-neutral public envelope projection."""

    if not isinstance(envelope, StateActionEnvelopeV1):
        raise TypeError("envelope must be StateActionEnvelopeV1")
    envelope.verify_digest()
    source = envelope.unsigned_dict()
    missing = set(_ENVELOPE_PUBLIC_FIELDS) - set(source)
    if missing:  # pragma: no cover - protects against an upstream certificate drift
        raise ValueError(f"state/action envelope lost public fields: {sorted(missing)}")
    return {
        "schema": QUALITY_PUBLIC_ENVELOPE_PROJECTION_SCHEMA,
        "schema_version": QUALITY_PUBLIC_PROJECTION_VERSION,
        **{field: source[field] for field in _ENVELOPE_PUBLIC_FIELDS},
    }


def public_envelope_projection_digest(envelope: StateActionEnvelopeV1) -> str:
    """Digest the exact public envelope whitelist."""

    return content_digest(public_envelope_projection(envelope))


def public_applied_action_projection(
    applied: AppliedStateActionV1,
) -> dict[str, Any]:
    """Return the authority-neutral public successor projection.

    ``provenance_fingerprint`` and ``envelope_record_digest`` are excluded because both name the
    phase-local authority.  All state, selected-action, persistence, and successor fields remain.
    """

    if not isinstance(applied, AppliedStateActionV1):
        raise TypeError("applied must be AppliedStateActionV1")
    applied.verify_digest()
    source = applied.unsigned_dict()
    missing = set(_APPLIED_PUBLIC_FIELDS) - set(source)
    if missing:  # pragma: no cover - protects against an upstream certificate drift
        raise ValueError(f"applied action lost public fields: {sorted(missing)}")
    return {
        "schema": QUALITY_PUBLIC_APPLIED_PROJECTION_SCHEMA,
        "schema_version": QUALITY_PUBLIC_PROJECTION_VERSION,
        **{field: source[field] for field in _APPLIED_PUBLIC_FIELDS},
    }


def public_applied_action_projection_digest(applied: AppliedStateActionV1) -> str:
    """Digest the exact public applied-action whitelist."""

    return content_digest(public_applied_action_projection(applied))


__all__ = [
    "QUALITY_PUBLIC_APPLIED_PROJECTION_SCHEMA",
    "QUALITY_PUBLIC_ENVELOPE_PROJECTION_SCHEMA",
    "QUALITY_PUBLIC_PROJECTION_VERSION",
    "public_applied_action_projection",
    "public_applied_action_projection_digest",
    "public_envelope_projection",
    "public_envelope_projection_digest",
]
