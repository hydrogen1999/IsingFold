from __future__ import annotations

import copy
import random
from dataclasses import FrozenInstanceError, asdict, fields

import pytest
from embedbench.candidate_bank import (
    AcceptanceProtocol,
    CandidateGroup,
    CandidateRecord,
    EvaluationCurve,
    InstanceRecord,
    RepairAttempt,
    ReplayPolicy,
    UnlabeledAttemptView,
    UnlabeledCandidateView,
    UnlabeledGroupView,
    assign_split,
    audit_oracle_policy,
    content_digest,
    derive_attempt_id,
    derive_generation_digest,
    derive_group_id,
    fit_oracle_policy,
    max_chain_first_policy,
    random_policy,
    resource_policy,
)
from embedbench.candidate_bank import (
    filter_groups_by_split as _filter_groups_by_split,
)
from embedbench.candidate_bank import (
    replay_policy as _replay_policy,
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


def _curve(value: float, partition: str) -> EvaluationCurve:
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


def _record(group_id: str, _candidate_alias: str, chains, fit: float, audit: float):
    total_qubits = sum(map(len, chains))
    return CandidateRecord.create(
        group_id=group_id,
        candidate_id=None,
        chains=chains,
        decision=_curve(fit, "decision"),
        audit=_curve(audit, "audit"),
        features=(
            ("total_qubits", float(total_qubits)),
            ("max_chain", float(max(map(len, chains)))),
        ),
    )


_INSTANCES: dict[str, InstanceRecord] = {}


def _instance(problem_alias: str) -> InstanceRecord:
    offset = int(content_digest({"problem_alias": problem_alias})[:12], 16) / 16**12
    nodes = tuple(range(9))
    return InstanceRecord.create(
        family="replay-test",
        topology="complete-K9",
        logical_nodes=(0, 1),
        logical_edges=((0, 1),),
        host_nodes=nodes,
        host_edges=tuple((left, right) for left in nodes for right in nodes if left < right),
        h=((0, offset), (1, -offset)),
        j=((0, 1, -1.0),),
    )


def _create_group(*, instance: InstanceRecord, **values) -> CandidateGroup:
    instance_record_digest = instance.record_digest
    generation_digest = derive_generation_digest(
        group_id=values["group_id"],
        instance_id=values["instance_id"],
        instance_record_digest=instance_record_digest,
        split_unit_id=values["split_unit_id"],
        group_seed=values["group_seed"],
        protocol=values["protocol"],
        incumbent_chains=values["incumbent"].chains,
        candidate_chains=tuple(candidate.chains for candidate in values["candidates"]),
        attempts=values["attempts"],
        rejection_reason=None,
    )
    result = CandidateGroup.create(
        **values,
        instance_record_digest=instance_record_digest,
        generation_digest=generation_digest,
    )
    _INSTANCES[instance.instance_id] = instance
    return result


def replay_policy(groups, policy, **options):
    instances = tuple({_INSTANCES[group.instance_id] for group in groups})
    return _replay_policy(groups, policy, instances=instances, **options)


def filter_groups_by_split(groups, *, split):
    instances = tuple({_INSTANCES[group.instance_id] for group in groups})
    return _filter_groups_by_split(groups, split=split, instances=instances)


def _group(group_alias="g0", problem_alias="i0", split=None, specs=None):
    specs = specs or (
        ("a", ((0, 1), (2, 3)), 0.60, 0.20),
        ("b", ((0, 1, 2), (3, 4, 5)), 0.90, 0.80),
        ("c", ((0, 1, 3), (2, 4)), 0.40, 0.95),
    )
    instance = _instance(problem_alias)
    instance_id = instance.instance_id
    group_seed = 17
    protocol = (
        ("attempt_slots", len(specs)),
        ("name", "quality-replay-v1"),
        ("test_group", group_alias),
    )
    incumbent_chains = ((0, 1, 2), (3, 4))
    group_id = derive_group_id(instance_id, incumbent_chains, protocol, group_seed)
    incumbent = _record(group_id, "incumbent", incumbent_chains, 0.50, 0.55)
    candidates = tuple(_record(group_id, *spec) for spec in specs)
    attempts = tuple(
        RepairAttempt(
            attempt_id=derive_attempt_id(group_id, slot),
            repair_seed=100 + slot,
            neighborhood=(slot % 2,),
            status="valid",
            candidate_id=candidate.candidate_id,
            slot=slot,
        )
        for slot, candidate in enumerate(candidates)
    )
    return _create_group(
        instance=instance,
        group_id=group_id,
        instance_id=instance_id,
        split_unit_id=instance.split_unit_id,
        split=assign_split(instance.split_unit_id) if split is None else split,
        group_seed=group_seed,
        protocol=protocol,
        incumbent=incumbent,
        candidates=candidates,
        attempts=attempts,
    )


def test_evaluation_curve_prefers_p_solve_then_uses_residual_quality() -> None:
    measured = EvaluationCurve(
        "decision",
        (1.0, 2.0),
        (1, 2),
        ((0.1, 0.3), (0.4, 0.2)),
        ((4.0, 2.0), (1.0, 1.0)),
        400,
        2_000,
        "solve_probability_then_residual-v1",
        EVALUATION_PROTOCOL,
    )
    zeros = EvaluationCurve(
        "decision",
        (1.0,),
        (1, 2),
        ((0.0, 0.0),),
        ((0.01, 0.01),),
        400,
        2_000,
        "solve_probability_then_residual-v1",
        EVALUATION_PROTOCOL,
    )
    residual = EvaluationCurve(
        "decision",
        (1.0, 2.0),
        (1, 2),
        None,
        ((4.0, 2.0), (2.0, 2.0)),
        400,
        2_000,
        "residual_mean-v1",
        EVALUATION_PROTOCOL,
    )
    assert measured.best_outcome().p_solve == pytest.approx(0.30)
    assert measured.best_outcome().residual_mean == pytest.approx(1.0)
    assert zeros.best_outcome().p_solve == 0.0
    assert zeros.best_outcome().residual_mean == pytest.approx(0.01)
    assert residual.best_outcome().p_solve is None
    assert residual.best_outcome().residual_mean == pytest.approx(2.0)


def test_policies_receive_identical_candidates_but_deployable_code_sees_no_labels() -> None:
    seen = []

    def select(group: UnlabeledGroupView, rng: random.Random) -> str:
        del rng
        seen.append(group)
        return group.candidates[0].candidate_id

    group = _group()
    recording = replay_policy(
        (group,),
        ReplayPolicy("recording", select, implementation_id="tests.recording-v1"),
        seed=5,
    )[0]
    builtins = [
        replay_policy((group,), policy, seed=5)[0]
        for policy in (
            resource_policy(),
            random_policy(),
            fit_oracle_policy(),
            audit_oracle_policy(),
        )
    ]
    assert len(seen) == 1
    view = seen[0]
    assert isinstance(view, UnlabeledGroupView)
    assert isinstance(view.incumbent, UnlabeledCandidateView)
    assert all(isinstance(candidate, UnlabeledCandidateView) for candidate in view.candidates)
    assert all(isinstance(attempt, UnlabeledAttemptView) for attempt in view.attempts)
    assert all(
        not hasattr(candidate, label)
        for candidate in (view.incumbent, *view.candidates)
        for label in ("decision", "audit")
    )
    first_candidate = group.candidates[0]
    expected_ids = tuple(candidate.candidate_id for candidate in group.candidates)
    assert (
        view.candidates[0].candidate_id,
        view.candidates[0].chains,
        view.candidates[0].total_qubits,
        view.candidates[0].max_chain,
        view.candidates[0].features,
    ) == (
        first_candidate.candidate_id,
        first_candidate.chains,
        first_candidate.total_qubits,
        first_candidate.max_chain,
        first_candidate.features,
    )
    assert {result.eligible_candidate_ids for result in [recording, *builtins]} == {expected_ids}


def test_deployable_group_view_is_complete_immutable_and_has_no_leakage_fields() -> None:
    captured = []

    def select(view: UnlabeledGroupView, rng: random.Random) -> str:
        del rng
        captured.append(view)
        return view.candidates[0].candidate_id

    group = _group("context", "context-problem")
    instance = _INSTANCES[group.instance_id]
    replay_policy(
        (group,),
        ReplayPolicy("context", select, implementation_id="tests.context-v1"),
        seed=7,
    )
    view = captured[0]

    assert {field.name for field in fields(view)} == {
        "attempts",
        "candidates",
        "family",
        "h",
        "host_edges",
        "host_nodes",
        "incumbent",
        "j",
        "logical_edges",
        "logical_nodes",
        "topology",
    }
    assert (
        view.family,
        view.topology,
        view.logical_nodes,
        view.logical_edges,
        view.h,
        view.j,
        view.host_nodes,
        view.host_edges,
    ) == (
        instance.family,
        instance.topology,
        instance.logical_nodes,
        instance.logical_edges,
        instance.h,
        instance.j,
        instance.host_nodes,
        instance.host_edges,
    )
    assert view.incumbent == UnlabeledCandidateView.from_record(group.incumbent)
    assert tuple(
        (attempt.slot, attempt.neighborhood, attempt.status, attempt.candidate_id)
        for attempt in view.attempts
    ) == tuple(
        (attempt.slot, attempt.neighborhood, attempt.status, attempt.candidate_id)
        for attempt in group.attempts
    )

    payload = asdict(view)
    forbidden = {
        "audit",
        "audit_outcome",
        "audit_quality",
        "decision",
        "generation_digest",
        "group_id",
        "group_seed",
        "instance_id",
        "metadata",
        "p_solve",
        "protocol",
        "record_digest",
        "repair_seed",
        "residual_mean",
        "split",
        "split_unit_id",
    }

    def nested_keys(value):
        if isinstance(value, dict):
            return set(value).union(*(nested_keys(item) for item in value.values()))
        if isinstance(value, (list, tuple)):
            return set().union(*(nested_keys(item) for item in value))
        return set()

    assert forbidden.isdisjoint(nested_keys(payload))
    with pytest.raises(FrozenInstanceError):
        view.topology = "tampered"  # type: ignore[misc]
    with pytest.raises(TypeError):
        view.logical_edges[0][0] = 99  # type: ignore[index]


def test_relabeling_candidates_cannot_change_deployable_policy_input() -> None:
    seen = []

    def select(view: UnlabeledGroupView, rng: random.Random) -> str:
        del rng
        seen.append(view)
        return view.candidates[0].candidate_id

    chains = (((0, 1), (2, 3)), ((0, 1, 2), (3, 4, 5)))
    first = _group(
        "label-invariance",
        "same-problem",
        specs=(
            ("first", chains[0], 0.11, 0.22),
            ("second", chains[1], 0.33, 0.44),
        ),
    )
    relabeled = _group(
        "label-invariance",
        "same-problem",
        specs=(
            ("first", chains[0], 0.91, 0.82),
            ("second", chains[1], 0.73, 0.64),
        ),
    )
    policy = ReplayPolicy("capture", select, implementation_id="tests.capture-v1")

    replay_policy((first,), policy, seed=3)
    replay_policy((relabeled,), policy, seed=3)

    assert first.record_digest != relabeled.record_digest
    assert seen[0] == seen[1]


def test_q_cap_hides_attempt_references_to_ineligible_candidates() -> None:
    seen = []

    def select(view: UnlabeledGroupView, rng: random.Random) -> str:
        del rng
        seen.append(view)
        return view.candidates[0].candidate_id

    group = _group()
    replay_policy(
        (group,),
        ReplayPolicy("capped-view", select, implementation_id="tests.capped-view-v1"),
        seed=0,
        q_cap=5,
    )

    visible_ids = {candidate.candidate_id for candidate in seen[0].candidates}
    assert visible_ids == {
        candidate.candidate_id for candidate in group.candidates if candidate.total_qubits <= 5
    }
    assert all(
        attempt.candidate_id is None or attempt.candidate_id in visible_ids
        for attempt in seen[0].attempts
    )


def test_resource_ties_and_random_seed_are_deterministic() -> None:
    specs = (
        ("z", ((0, 1), (2, 3)), 0.6, 0.6),
        ("a", ((0, 2), (1, 3)), 0.6, 0.6),
        ("m", ((0, 3), (1, 2)), 0.6, 0.6),
    )
    forward = _group(specs=specs)
    reversed_group = _create_group(
        instance=_INSTANCES[forward.instance_id],
        group_id=forward.group_id,
        instance_id=forward.instance_id,
        split_unit_id=forward.split_unit_id,
        split=forward.split,
        group_seed=forward.group_seed,
        protocol=forward.protocol,
        incumbent=forward.incumbent,
        candidates=tuple(reversed(forward.candidates)),
        attempts=tuple(reversed(forward.attempts)),
    )
    assert tuple(candidate.candidate_id for candidate in forward.candidates) == tuple(
        sorted(candidate.candidate_id for candidate in forward.candidates)
    )
    assert forward == reversed_group
    same_seed = replay_policy((forward,), resource_policy(), seed=99)[0].selected_candidate_id
    assert (
        same_seed
        == replay_policy((reversed_group,), resource_policy(), seed=99)[0].selected_candidate_id
    )
    seeded_choices = {
        replay_policy((forward,), resource_policy(), seed=seed)[0].selected_candidate_id
        for seed in range(16)
    }
    assert len(seeded_choices) > 1
    first = replay_policy((forward,), random_policy(), seed=1)[0].selected_candidate_id
    assert first == replay_policy((forward,), random_policy(), seed=1)[0].selected_candidate_id
    assert first != replay_policy((forward,), random_policy(), seed=5)[0].selected_candidate_id


def test_resource_policy_matches_native_feasible_total_qubit_order() -> None:
    native_preferred_chains = ((0, 1, 2, 3), (4,))
    max_chain_preferred_chains = ((0, 1, 2), (3, 4, 5))
    group = _group(
        "native-resource-order",
        specs=(
            ("fewer-used-nodes", native_preferred_chains, 0.6, 0.6),
            ("shorter-max-chain", max_chain_preferred_chains, 0.6, 0.6),
        ),
    )
    native_preferred = next(
        candidate for candidate in group.candidates if candidate.chains == native_preferred_chains
    )

    result = replay_policy((group,), resource_policy(), seed=0)[0]

    assert (native_preferred.total_qubits, native_preferred.max_chain) == (5, 4)
    assert result.selected_candidate_id == native_preferred.candidate_id


def test_max_chain_first_heuristic_remains_explicitly_separate_from_native_resource() -> None:
    native_preferred_chains = ((0, 1, 2, 3), (4,))
    max_chain_preferred_chains = ((0, 1, 2), (3, 4, 5))
    group = _group(
        "separate-max-chain-heuristic",
        specs=(
            ("fewer-used-nodes", native_preferred_chains, 0.6, 0.6),
            ("shorter-max-chain", max_chain_preferred_chains, 0.6, 0.6),
        ),
    )
    max_chain_preferred = next(
        candidate
        for candidate in group.candidates
        if candidate.chains == max_chain_preferred_chains
    )

    native = replay_policy((group,), resource_policy(), seed=0)[0]
    heuristic = replay_policy((group,), max_chain_first_policy(), seed=0)[0]

    assert native.selected_candidate_id != heuristic.selected_candidate_id
    assert heuristic.selected_candidate_id == max_chain_preferred.candidate_id
    assert heuristic.policy_name == "max-chain-first"


def test_q_cap_only_filters_eligibility_and_does_not_rewrite_labels() -> None:
    group, frozen = _group(), copy.deepcopy(_group())
    capped = replay_policy((group,), fit_oracle_policy(), seed=0, q_cap=5)[0]
    uncapped = replay_policy((group,), fit_oracle_policy(), seed=0)[0]
    eligible = tuple(candidate for candidate in group.candidates if candidate.total_qubits <= 5)
    capped_best = max(eligible, key=lambda candidate: candidate.decision.best_outcome().p_solve)
    uncapped_best = max(
        group.candidates, key=lambda candidate: candidate.decision.best_outcome().p_solve
    )
    assert (capped.eligible_candidate_ids, capped.selected_candidate_id) == (
        tuple(candidate.candidate_id for candidate in eligible),
        capped_best.candidate_id,
    )
    assert uncapped.selected_candidate_id == uncapped_best.candidate_id
    assert group == frozen
    assert uncapped_best.decision.best_outcome().p_solve == pytest.approx(0.90)
    assert uncapped.audit_quality.p_solve == pytest.approx(0.80)
    assert uncapped.audit_quality.residual_mean == pytest.approx(0.20)


def test_one_choice_per_group_reports_decision_acceptance_and_audit_separately() -> None:
    audit_only = _group(
        "ga",
        specs=(
            ("audit-only", ((0, 1), (2, 3)), 0.40, 0.95),
            ("dominated-a", ((0, 1, 2), (3, 4, 5)), 0.10, 0.10),
        ),
    )
    decision_only = _group(
        "gd",
        "i1",
        specs=(
            ("decision-only", ((0, 1), (2, 3)), 0.60, 0.20),
            ("dominated-d", ((0, 1, 2), (3, 4, 5)), 0.10, 0.10),
        ),
    )
    results = replay_policy((audit_only, decision_only), resource_policy(), seed=0)
    assert len(results) == 2
    assert all(result.selected_candidate_id is not None for result in results)
    by_group = {result.group_id: result for result in results}
    assert by_group[audit_only.group_id].decision_accepted is False
    assert by_group[audit_only.group_id].audit_quality.p_solve == pytest.approx(0.55)
    assert by_group[decision_only.group_id].decision_accepted is True
    assert by_group[decision_only.group_id].audit_quality.p_solve == pytest.approx(0.20)


def test_acceptance_protocol_charges_and_reports_the_post_selection_evaluation() -> None:
    protocol = AcceptanceProtocol(
        name="one-shot-decision-improvement-v1",
        margin=0.20,
        decision_evaluations=1,
    )

    result = replay_policy(
        (_group(),),
        resource_policy(),
        seed=0,
        acceptance=protocol,
    )[0]

    assert result.decision_accepted is False
    assert result.decision_evaluations == 1
    assert result.acceptance_protocol == "one-shot-decision-improvement-v1"
    assert result.replay_seed == 0
    assert result.q_cap is None
    assert result.acceptance_margin == pytest.approx(0.20)
    assert (
        result.policy_implementation_id
        == "embedbench.native-resource-total-qubits-random-tie-v1"
    )
    assert result.policy_trust_boundary == "in-process-attested"
    assert result.audit_quality.p_solve == pytest.approx(0.55)
    assert result.audit_quality.residual_mean == pytest.approx(0.45)


def test_oracles_are_explicitly_nondeployable() -> None:
    fit, audit = fit_oracle_policy(), audit_oracle_policy()
    assert (fit.name, fit.deployable) == ("fit-oracle", False)
    assert (audit.name, audit.deployable) == ("audit-oracle", False)
    assert replay_policy((_group(),), fit, seed=0)[0].policy_deployable is False
    assert replay_policy((_group(),), audit, seed=0)[0].policy_deployable is False


def test_replay_policy_rejects_an_unknown_oracle_partition() -> None:
    with pytest.raises(ValueError, match="oracle_partition|decision|audit"):
        ReplayPolicy(
            "misspelled-oracle",
            None,
            implementation_id="tests.misspelled-oracle-v1",
            deployable=False,
            oracle_partition="decison",  # type: ignore[arg-type]
        )


def test_audit_oracle_optimizes_post_acceptance_audit_outcome() -> None:
    group = _group(
        "post-acceptance",
        specs=(
            ("raw-audit-best-but-rejected", ((0, 1), (2, 3)), 0.40, 0.99),
            ("accepted-audit-best", ((0, 1, 2), (3, 4, 5)), 0.90, 0.80),
        ),
    )

    result = replay_policy((group,), audit_oracle_policy(), seed=7)[0]

    accepted = max(
        group.candidates, key=lambda candidate: candidate.decision.best_outcome().p_solve
    )
    assert result.selected_candidate_id == accepted.candidate_id
    assert result.decision_accepted is True
    assert result.audit_quality.p_solve == pytest.approx(0.80)


def test_q_cap_rejects_a_group_whose_incumbent_exceeds_the_cap() -> None:
    with pytest.raises(ValueError, match="incumbent|q_cap"):
        replay_policy((_group(),), resource_policy(), seed=0, q_cap=4)


def test_split_filtering_rejects_one_logical_instance_in_multiple_splits() -> None:
    def problem_alias_for(target: str) -> str:
        return next(
            alias
            for index in range(10_000)
            if assign_split(_instance(alias := f"problem-{index}").split_unit_id) == target
        )

    train_alias = problem_alias_for("train")
    test_alias = problem_alias_for("test")
    groups = (_group("tr", train_alias, "train"), _group("te", test_alias, "test"))
    assert filter_groups_by_split(groups, split="train") == (groups[0],)
    with pytest.raises(ValueError, match="split.*derived|split_unit_id"):
        _group("leak", train_alias, "test")
