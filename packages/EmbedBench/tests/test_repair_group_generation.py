from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import pytest
from embedbench import repair_groups
from embedbench.candidate_bank import (
    EvaluationCurve,
    InstanceRecord,
    content_digest,
    derive_attempt_id,
)
from embedbench.repair_groups import (
    InsufficientGroupBudget,
    generate_repaired_group,
    seal_generated_group,
    validate_generated_group,
)

EVALUATION_PROTOCOL = (
    ("decoder", "majority-v1"),
    ("noise_model", "none"),
    ("reference_energy", "known-e0-v1"),
    ("sampler", "test-sa"),
    ("sampler_version", "test-v1"),
    ("schedule", "geometric-v1"),
    ("seed_derivation", "stable-v1"),
)


def _no_change_repairer(
    logical,
    host,
    incumbent,
    *,
    neighborhood,
    random_seed,
    max_candidates,
    max_transitions,
):
    del logical, host, neighborhood, random_seed, max_candidates, max_transitions
    return {"chains": incumbent, "transitions": 1, "reason": "success"}


def _instance() -> InstanceRecord:
    """A valid incumbent with enough host slack for several repaired embeddings."""

    return InstanceRecord.create(
        family="unit",
        topology="cycle-C8",
        logical_nodes=(0, 1, 2),
        logical_edges=((0, 1), (1, 2)),
        host_nodes=tuple(range(8)),
        host_edges=tuple((node, (node + 1) % 8) for node in range(8)),
        h=((0, 0.0), (1, 0.0), (2, 0.0)),
        j=((0, 1, -1.0), (1, 2, 1.0)),
        metadata=(),
    )


def _generate(**overrides):
    options = {
        "group_seed": 27101,
        "attempt_slots": 6,
        "max_candidates": 8,
        "max_transitions_per_attempt": 20,
        "repairer": _no_change_repairer,
        "repairer_id": "test-no-change-v1",
    }
    options.update(overrides)
    return generate_repaired_group(
        _instance(),
        ((0,), (1,), (2,)),
        **options,
    )


def _curve(partition: str, value: float) -> EvaluationCurve:
    return EvaluationCurve(
        partition=partition,
        strengths=(1.0,),
        seeds=(101, 103) if partition == "decision" else (201, 203),
        p_solve=((value, value),),
        residual_mean=((1.0 - value, 1.0 - value),),
        reads=400 if partition == "decision" else 4_000,
        sweeps=2_000,
        objective="solve_probability_then_residual-v1",
        evaluation_protocol=EVALUATION_PROTOCOL,
    )


def test_exact_group_rerun_is_structurally_identical() -> None:
    first = _generate()
    second = _generate()

    assert first == second
    assert first.record_digest == second.record_digest
    assert len(first.attempts) == 6
    assert [attempt.attempt_id for attempt in first.attempts] == [
        derive_attempt_id(first.group_id, slot) for slot in range(6)
    ]
    assert all(attempt.neighborhood in _instance().logical_edges for attempt in first.attempts)


def test_slot_seeds_and_prefix_outcomes_do_not_depend_on_later_slots() -> None:
    short = _generate(attempt_slots=3)
    long = _generate(attempt_slots=7)

    def attempt_projection(draft):
        chains_by_id = {candidate.candidate_id: candidate.chains for candidate in draft.candidates}
        return tuple(
            (
                attempt.repair_seed,
                attempt.neighborhood,
                attempt.status,
                chains_by_id.get(attempt.candidate_id),
                attempt.transitions,
                attempt.reason,
            )
            for attempt in draft.attempts
        )

    assert attempt_projection(short) == attempt_projection(long)[:3]


def test_every_fixed_slot_is_recorded_including_failure_duplicate_and_no_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repaired_a = ((0,), (1,), (2, 3))
    repaired_b = ((7,), (0,), (1, 2))
    outcomes = (
        SimpleNamespace(chains=repaired_a, transitions=3, reason="success"),
        SimpleNamespace(chains=None, transitions=20, reason="transition_limit"),
        SimpleNamespace(chains=repaired_a, transitions=4, reason="success"),
        SimpleNamespace(chains=((0,), (1,), (2,)), transitions=2, reason="success"),
        SimpleNamespace(chains=repaired_b, transitions=5, reason="success"),
    )
    observed_slots: list[int] = []

    def fake_repair_attempt(**kwargs):
        slot = kwargs["slot"]
        observed_slots.append(slot)
        return outcomes[slot]

    monkeypatch.setattr(repair_groups, "_run_repair_attempt", fake_repair_attempt)

    draft = _generate(attempt_slots=len(outcomes))

    assert observed_slots == list(range(len(outcomes)))
    assert [attempt.status for attempt in draft.attempts] == [
        "valid",
        "repair_failed",
        "duplicate",
        "no_change",
        "valid",
    ]
    assert [attempt.transitions for attempt in draft.attempts] == [3, 20, 4, 2, 5]
    assert draft.attempts[1].reason == "transition_limit"
    assert draft.attempts[0].candidate_id == draft.attempts[2].candidate_id
    assert draft.attempts[1].candidate_id is None
    assert draft.attempts[3].candidate_id is None
    assert len(draft.candidates) == 2
    assert draft.rejection_reason is None


def test_insufficient_budget_is_rejected_before_any_native_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def should_not_run(**kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError(f"unexpected repair call: {kwargs}")

    monkeypatch.setattr(repair_groups, "_run_repair_attempt", should_not_run)

    with pytest.raises(InsufficientGroupBudget, match="whole group|requires 50"):
        _generate(
            attempt_slots=5,
            max_transitions_per_attempt=10,
            available_transition_budget=49,
        )

    assert calls == 0


def test_repairer_cannot_report_more_than_the_per_attempt_transition_cap() -> None:
    def over_budget_repairer(
        logical,
        host,
        incumbent,
        *,
        neighborhood,
        random_seed,
        max_candidates,
        max_transitions,
    ):
        del logical, host, neighborhood, random_seed, max_candidates
        return {
            "chains": incumbent,
            "transitions": max_transitions + 1,
            "reason": "success",
        }

    with pytest.raises(ValueError, match="transitions.*cap|transition.*maximum"):
        _generate(
            attempt_slots=1,
            max_transitions_per_attempt=7,
            repairer=over_budget_repairer,
            repairer_id="test-over-budget-v1",
        )


def test_validation_rejects_rehashed_attempt_above_recorded_transition_cap() -> None:
    instance = _instance()
    draft = _generate(max_transitions_per_attempt=7)
    attempts = list(draft.attempts)
    attempts[0] = replace(attempts[0], transitions=8)
    tampered = replace(draft, attempts=tuple(attempts), record_digest="0" * 64)
    tampered = replace(tampered, record_digest=content_digest(tampered.to_dict()))

    with pytest.raises(ValueError, match="transitions.*cap|transition.*maximum"):
        validate_generated_group(tampered, instance)


def test_underfull_group_is_retained_as_an_unsealable_rejected_draft(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    only_candidate = ((0,), (1,), (2, 3))
    outcomes = (
        SimpleNamespace(chains=only_candidate, transitions=1, reason="success"),
        SimpleNamespace(chains=only_candidate, transitions=2, reason="success"),
        SimpleNamespace(chains=None, transitions=20, reason="transition_limit"),
        SimpleNamespace(chains=((0,), (1,), (2,)), transitions=1, reason="success"),
    )

    monkeypatch.setattr(
        repair_groups,
        "_run_repair_attempt",
        lambda **kwargs: outcomes[kwargs["slot"]],
    )

    draft = _generate(attempt_slots=len(outcomes))

    assert len(draft.attempts) == len(outcomes)
    assert len(draft.candidates) == 1
    assert draft.rejection_reason == "fewer_than_2_unique_candidates"
    assert draft.sealable is False


def test_incumbent_and_repaired_chains_are_canonicalized_before_hashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instance = InstanceRecord.create(
        family="unit",
        topology="path-P4",
        logical_nodes=(0, 1),
        logical_edges=((0, 1),),
        host_nodes=(0, 1, 2, 3),
        host_edges=((0, 1), (1, 2), (2, 3)),
        h=((0, 0.0), (1, 0.0)),
        j=((0, 1, -1.0),),
        metadata=(),
    )
    outcomes = (
        SimpleNamespace(chains=((2, 1), (0,)), transitions=1, reason="success"),
        SimpleNamespace(chains=((3,), (2, 1)), transitions=1, reason="success"),
    )
    monkeypatch.setattr(
        repair_groups,
        "_run_repair_attempt",
        lambda **kwargs: outcomes[kwargs["slot"]],
    )

    first = generate_repaired_group(
        instance,
        ((1, 0), (2,)),
        group_seed=7,
        attempt_slots=2,
        max_transitions_per_attempt=5,
        repairer=_no_change_repairer,
        repairer_id="test-no-change-v1",
    )
    second = generate_repaired_group(
        instance,
        ((0, 1), (2,)),
        group_seed=7,
        attempt_slots=2,
        max_transitions_per_attempt=5,
        repairer=_no_change_repairer,
        repairer_id="test-no-change-v1",
    )

    assert first.incumbent.chains == ((0, 1), (2,))
    assert {candidate.chains for candidate in first.candidates} == {
        ((1, 2), (0,)),
        ((3,), (1, 2)),
    }
    assert first.record_digest == second.record_digest


def test_generation_api_cannot_receive_labels_evaluators_or_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden = {
        "audit",
        "decision",
        "evaluator",
        "labeler",
        "model",
        "objective",
        "policy",
        "scorer",
        "screen_model",
    }
    assert forbidden.isdisjoint(inspect.signature(generate_repaired_group).parameters)

    candidates = (
        ((0,), (1,), (2, 3)),
        ((7,), (0,), (1, 2)),
    )
    monkeypatch.setattr(
        repair_groups,
        "_run_repair_attempt",
        lambda **kwargs: SimpleNamespace(
            chains=candidates[kwargs["slot"]], transitions=1, reason="success"
        ),
    )
    draft = _generate(attempt_slots=2)

    assert draft.sealable
    assert all(
        forbidden.isdisjoint(candidate.__dataclass_fields__)
        for candidate in (draft.incumbent, *draft.candidates)
    )

    with pytest.raises(TypeError, match="unexpected keyword argument 'model'"):
        generate_repaired_group(
            _instance(),
            ((0,), (1,), (2,)),
            group_seed=1,
            attempt_slots=2,
            max_transitions_per_attempt=5,
            repairer=_no_change_repairer,
            repairer_id="test-no-change-v1",
            model=object(),
        )


def test_sealing_commits_to_the_complete_prelabel_candidate_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = (
        SimpleNamespace(chains=((0,), (1,), (2, 3)), transitions=2, reason="success"),
        SimpleNamespace(chains=((7,), (0,), (1, 2)), transitions=3, reason="success"),
        SimpleNamespace(chains=None, transitions=20, reason="transition_limit"),
    )
    monkeypatch.setattr(
        repair_groups,
        "_run_repair_attempt",
        lambda **kwargs: outcomes[kwargs["slot"]],
    )
    instance = _instance()
    draft = generate_repaired_group(
        instance,
        ((0,), (1,), (2,)),
        group_seed=41,
        attempt_slots=3,
        max_transitions_per_attempt=20,
        repairer=_no_change_repairer,
        repairer_id="test-no-change-v1",
    )
    labels = {
        item.candidate_id: (_curve("decision", 0.6), _curve("audit", 0.55))
        for item in (draft.incumbent, *draft.candidates)
    }

    sealed = seal_generated_group(draft, instance, labels=labels)

    assert sealed.generation_digest == draft.record_digest
    assert tuple(candidate.candidate_id for candidate in sealed.candidates) == tuple(
        candidate.candidate_id for candidate in draft.candidates
    )
    assert sealed.attempts == draft.attempts

    omitted = replace(draft, candidates=draft.candidates[:-1])
    with pytest.raises(ValueError, match="generation|draft|digest"):
        seal_generated_group(omitted, instance, labels=labels)


def test_sealing_requires_labels_for_exactly_the_committed_universe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = (
        SimpleNamespace(chains=((0,), (1,), (2, 3)), transitions=2, reason="success"),
        SimpleNamespace(chains=((7,), (0,), (1, 2)), transitions=3, reason="success"),
    )
    monkeypatch.setattr(
        repair_groups,
        "_run_repair_attempt",
        lambda **kwargs: outcomes[kwargs["slot"]],
    )
    instance = _instance()
    draft = generate_repaired_group(
        instance,
        ((0,), (1,), (2,)),
        group_seed=43,
        attempt_slots=2,
        max_transitions_per_attempt=20,
        repairer=_no_change_repairer,
        repairer_id="test-no-change-v1",
    )
    labels = {
        item.candidate_id: (_curve("decision", 0.6), _curve("audit", 0.55))
        for item in (draft.incumbent, *draft.candidates)
    }
    labels.pop(draft.candidates[-1].candidate_id)

    with pytest.raises(ValueError, match="labels.*exact|candidate universe|missing"):
        seal_generated_group(draft, instance, labels=labels)
