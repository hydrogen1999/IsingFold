from __future__ import annotations

from dataclasses import replace

from isingfold.rl.contracts import Context
from isingfold.rl.data.action_certificate import apply_envelope_action
from isingfold.rl.data.quality import replay_decision_with_action_envelope
from isingfold.rl.data.quality_public_projection import (
    QUALITY_PUBLIC_APPLIED_PROJECTION_SCHEMA,
    QUALITY_PUBLIC_ENVELOPE_PROJECTION_SCHEMA,
    public_applied_action_projection,
    public_applied_action_projection_digest,
    public_envelope_projection,
    public_envelope_projection_digest,
)
from isingfold.rl.env import fixed_strength_selector
from tests.unit.test_rl_quality_resolution_plan import _public_prepared


def test_public_projection_is_equal_across_target_free_and_target_bound_envelopes() -> None:
    public = _public_prepared(0)
    target = replace(public, task=replace(public.task, ground_energy=-1.75))
    context = Context(qubit_cap=4, n_est_reads=256)
    first = replay_decision_with_action_envelope(
        public.task,
        context,
        initializer=public.initializer(),
        selector=fixed_strength_selector(),
        prefix=(),
        seed=907,
        reward_reads=256,
        provenance_fingerprint="1" * 64,
    )
    second = replay_decision_with_action_envelope(
        target.task,
        context,
        initializer=target.initializer(),
        selector=fixed_strength_selector(),
        prefix=(),
        seed=907,
        reward_reads=256,
        provenance_fingerprint="2" * 64,
    )
    assert first is not None and second is not None
    first_decision, first_envelope = first
    second_decision, second_envelope = second
    assert first_envelope.record_digest != second_envelope.record_digest
    assert public_envelope_projection(first_envelope) == public_envelope_projection(
        second_envelope
    )
    assert public_envelope_projection_digest(
        first_envelope
    ) == public_envelope_projection_digest(second_envelope)
    assert public_envelope_projection(first_envelope)["schema"] == (
        QUALITY_PUBLIC_ENVELOPE_PROJECTION_SCHEMA
    )

    nonterminal = next(
        index
        for index, (candidate, legal) in enumerate(
            zip(first_decision.candidates, first_decision.legal_mask, strict=True)
        )
        if legal and not candidate.opcode.is_terminal
    )
    first_applied = apply_envelope_action(first_envelope, nonterminal)
    second_applied = apply_envelope_action(second_envelope, nonterminal)
    assert first_applied.record_digest != second_applied.record_digest
    assert public_applied_action_projection(
        first_applied
    ) == public_applied_action_projection(second_applied)
    assert public_applied_action_projection_digest(
        first_applied
    ) == public_applied_action_projection_digest(second_applied)
    assert public_applied_action_projection(first_applied)["schema"] == (
        QUALITY_PUBLIC_APPLIED_PROJECTION_SCHEMA
    )


def test_projection_whitelist_retains_exact_action_and_work_coordinates() -> None:
    public = _public_prepared(1)
    replayed = replay_decision_with_action_envelope(
        public.task,
        Context(qubit_cap=4, n_est_reads=256),
        initializer=public.initializer(),
        selector=fixed_strength_selector(),
        prefix=(),
        seed=907,
        reward_reads=256,
        provenance_fingerprint="3" * 64,
    )
    assert replayed is not None
    decision, envelope = replayed
    projection = public_envelope_projection(envelope)

    assert projection["charged_work_receipt"] == decision.charged_work_receipt.as_dict()
    assert len(projection["candidates"]) == len(decision.candidates)
    assert [row["payload"]["work"] for row in projection["candidates"]] == [
        candidate.work.as_dict() for candidate in decision.candidates
    ]
    assert [row["payload"]["proposal_work"] for row in projection["candidates"]] == [
        candidate.proposal_work.as_dict() for candidate in decision.candidates
    ]
    assert "provenance_fingerprint" not in projection
    assert "record_digest" not in projection

