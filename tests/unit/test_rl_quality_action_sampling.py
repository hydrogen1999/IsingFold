from __future__ import annotations

import pytest

from isingfold.rl.contracts import Candidate, DecisionState, Opcode, WorkVector
from isingfold.rl.data.quality import deterministic_action_sample


def _candidate(
    index: int,
    *,
    opcode: Opcode = Opcode.REWRITE_GROUP,
    archive_ref: int | None = None,
) -> Candidate:
    return Candidate(
        opcode=opcode,
        affected=(),
        old_chains={},
        new_chains={},
        work=WorkVector(),
        payload_key=f"candidate-{index}",
        archive_ref=archive_ref,
    )


def _decision(
    candidates: tuple[Candidate, ...],
    legal_mask: tuple[bool, ...] | None = None,
) -> DecisionState:
    mask = legal_mask if legal_mask is not None else (True,) * len(candidates)
    return DecisionState(
        observation=None,
        candidates=candidates,
        legal_mask=mask,
        state_fingerprint="state-17",
        context_version="test",
        charged_work_receipt=WorkVector(),
        support_fingerprint="support-17",
    )


def test_action_sample_protects_incumbent_commit_and_records_exact_propensities() -> None:
    decision = _decision(
        (
            _candidate(0, opcode=Opcode.COMMIT, archive_ref=0),
            _candidate(1, opcode=Opcode.COMMIT, archive_ref=1),
            _candidate(2, opcode=Opcode.STOP),
            _candidate(3),
            _candidate(4),
        )
    )

    draw = deterministic_action_sample(decision, evaluated_actions=3, seed=91)

    assert draw.action_indices == tuple(sorted(draw.action_indices))
    assert draw.protected_commit_index == 0
    assert 0 in draw.action_indices
    assert len(draw.action_indices) == 3
    assert draw.inclusion_probability(0) == 1.0
    for index in draw.action_indices:
        if index != 0:
            assert draw.inclusion_probability(index) == pytest.approx(0.5)
    assert sum(
        1.0 / (decision.n_legal * draw.inclusion_probability(index))
        for index in draw.action_indices
    ) == pytest.approx(1.0)


def test_action_sample_gives_every_non_anchor_legal_opcode_positive_support() -> None:
    decision = _decision(
        (
            _candidate(0, opcode=Opcode.COMMIT, archive_ref=0),
            _candidate(1, opcode=Opcode.COMMIT, archive_ref=1),
            _candidate(2, opcode=Opcode.STOP),
            _candidate(3),
        )
    )

    seen: set[int] = set()
    for seed in range(256):
        draw = deterministic_action_sample(decision, evaluated_actions=2, seed=seed)
        seen.update(index for index in draw.action_indices if index != 0)
        assert draw.inclusion_probability(0) == 1.0
        assert all(
            draw.inclusion_probability(index) == pytest.approx(1.0 / 3.0)
            for index in draw.action_indices
            if index != 0
        )

    assert seen == {1, 2, 3}


def test_action_sample_without_protected_commit_is_uniform() -> None:
    decision = _decision(
        tuple(_candidate(index) for index in range(5)),
        legal_mask=(True, False, True, True, True),
    )

    draw = deterministic_action_sample(decision, evaluated_actions=2, seed=7)

    assert len(draw.action_indices) == 2
    assert draw.protected_commit_index is None
    assert all(index in {0, 2, 3, 4} for index in draw.action_indices)
    assert draw.inclusion_probabilities == pytest.approx((0.5, 0.5))


def test_action_sample_exhaustive_support_has_unit_propensities() -> None:
    decision = _decision(
        (
            _candidate(0, opcode=Opcode.COMMIT, archive_ref=0),
            _candidate(1, opcode=Opcode.STOP),
            _candidate(2),
        )
    )

    draw = deterministic_action_sample(decision, evaluated_actions=8, seed=7)

    assert draw.action_indices == (0, 1, 2)
    assert draw.inclusion_probabilities == (1.0, 1.0, 1.0)
    assert draw.protected_commit_index == 0


def test_action_sample_fails_when_anchor_would_zero_other_legal_propensities() -> None:
    decision = _decision(
        (
            _candidate(0, opcode=Opcode.COMMIT, archive_ref=0),
            _candidate(1),
        )
    )

    with pytest.raises(ValueError, match="at least two evaluated actions"):
        deterministic_action_sample(decision, evaluated_actions=1, seed=7)


def test_action_sample_rejects_duplicate_protected_commit_candidates() -> None:
    decision = _decision(
        (
            _candidate(0, opcode=Opcode.COMMIT, archive_ref=0),
            _candidate(1, opcode=Opcode.COMMIT, archive_ref=0),
            _candidate(2),
        )
    )

    with pytest.raises(ValueError, match="multiple legal protected COMMIT"):
        deterministic_action_sample(decision, evaluated_actions=2, seed=7)
