from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
from embedbench.candidate_protocol import (
    CandidateProtocolResult,
    FrozenChain,
    candidate_protocol_result_sha256,
    generate_candidate_bank,
)
from embedbench.ground_certificate import IsingProblem
from embedbench.hard_ood_schema import canonical_sha256
from embedbench.hardness import (
    CONTINUATION_CERTIFICATE_SCHEMA,
    HARDNESS_SCHEMA,
    PARTIAL_CONTEXT_SCHEMA,
    STARTING_EMBEDDING_SCHEMA,
    PartialContext,
    VerifiedHardness,
    VerifiedHost,
    VerifiedProblem,
    VerifiedStartingEmbedding,
    build_continuation_artifacts,
    build_partial_context_artifact,
    build_starting_embedding_artifact,
    continuation_checker_source_sha256,
    hardness_model_inputs,
    recompute_hardness,
    residual_hardness,
    validate_hardness,
    verify_continuation_evidence,
    verify_hardness,
    verify_host,
    verify_partial_context,
    verify_problem,
    verify_starting_embedding,
)
from embedbench.realized_host import build_realized_host_artifact

_HOST_SEED = bytes(range(32))
_SOURCE_SHA256 = "1" * 64
_REMOVAL_RULE_SHA256 = "2" * 64
_SOLVER_PROFILE_SHA256 = "3" * 64
_CANDIDATE_SEED = bytes(reversed(range(32)))
_CHECKER_ENVIRONMENT = {
    "schema": "embedbench.test-checker-environment",
    "schema_version": 1,
    "python": "test-runtime",
    "networkx": "test-version",
}


def _problem() -> IsingProblem:
    return IsingProblem(
        variables=(0, 1, 2),
        linear=((0, 0.0), (1, 0.0), (2, 0.0)),
        quadratic=((0, 1, -1.0), (1, 2, 1.0)),
    )


def _verified_problem() -> VerifiedProblem:
    problem = _problem()
    return verify_problem(problem, expected_problem_sha256=problem.problem_sha256)


def _host_artifact() -> dict[str, object]:
    return build_realized_host_artifact(
        topology="chimera",
        size=1,
        qubit_fraction=0.0,
        coupler_fraction=0.0,
        seed_key=_HOST_SEED,
    )


def _verified_host() -> VerifiedHost:
    artifact = _host_artifact()
    return verify_host(
        artifact,
        seed_key=_HOST_SEED,
        expected_host_artifact_sha256=artifact["host_artifact_sha256"],
        expected_host_sha256=artifact["host_sha256"],
    )


def _starting_artifact(problem: VerifiedProblem, host: VerifiedHost) -> dict[str, object]:
    return build_starting_embedding_artifact(
        problem=problem,
        host=host,
        source="witness",
        source_artifact_sha256=_SOURCE_SHA256,
        chains=(
            FrozenChain(0, (0,)),
            FrozenChain(1, (4,)),
            FrozenChain(2, (1,)),
        ),
    )


def _verified_starting(problem: VerifiedProblem, host: VerifiedHost) -> VerifiedStartingEmbedding:
    artifact = _starting_artifact(problem, host)
    return verify_starting_embedding(
        artifact,
        problem=problem,
        host=host,
        expected_starting_embedding_sha256=artifact["starting_embedding_sha256"],
    )


def _context_artifact(
    problem: VerifiedProblem,
    host: VerifiedHost,
    starting: VerifiedStartingEmbedding,
) -> dict[str, object]:
    return build_partial_context_artifact(
        problem=problem,
        host=host,
        starting_embedding=starting,
        witness_embedding=starting,
        focus=1,
        frozen_variables=(0,),
        window_nodes=(0, 1, 4, 5),
        window_edges=((0, 4), (0, 5), (1, 4), (1, 5)),
        l_cap=2,
        q_cap=None,
        q_cap_slack=0,
        removal_rule_sha256=_REMOVAL_RULE_SHA256,
    )


def _verified_context(
    problem: VerifiedProblem,
    host: VerifiedHost,
    starting: VerifiedStartingEmbedding,
) -> PartialContext:
    artifact = _context_artifact(problem, host, starting)
    return verify_partial_context(
        artifact,
        problem=problem,
        host=host,
        starting_embedding=starting,
        witness_embedding=starting,
        expected_partial_context_sha256=artifact["partial_context_sha256"],
    )


def _candidate_result(host: VerifiedHost, context: PartialContext) -> CandidateProtocolResult:
    result = generate_candidate_bank(
        host.graph(),
        window_nodes=context.window_nodes,
        frozen_chains=context.frozen_chains,
        required_logical_neighbors=context.required_focus_neighbors,
        original_focus_chain=context.original_focus_chain,
        l_cap=context.l_cap,
        q_cap=context.q_cap,
        candidate_sample_seed_key=_CANDIDATE_SEED,
    )
    assert result.attempt_status == "candidate_bank_ready"
    return result


def _continuation_inputs(
    problem: VerifiedProblem,
    host: VerifiedHost,
    context: PartialContext,
    result: CandidateProtocolResult,
) -> tuple[
    tuple[dict[str, object], ...],
    tuple[dict[str, object], ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str | None, ...],
]:
    certificates: list[dict[str, object]] = []
    acceptances: list[dict[str, object]] = []
    certificate_digests: list[str] = []
    acceptance_digests: list[str] = []
    assignment_digests: list[str | None] = []
    assert result.offered_to_full_indices is not None
    for canonical_index in result.offered_to_full_indices:
        certificate, acceptance = build_continuation_artifacts(
            problem=problem,
            host=host,
            partial_context=context,
            candidate_protocol_result=result,
            candidate_sample_seed_key=_CANDIDATE_SEED,
            expected_candidate_protocol_result_sha256=candidate_protocol_result_sha256(result),
            canonical_index=canonical_index,
            checker_environment=_CHECKER_ENVIRONMENT,
        )
        certificates.append(certificate)
        acceptances.append(acceptance)
        certificate_digests.append(certificate["certificate_sha256"])
        acceptance_digests.append(acceptance["checker_acceptance_sha256"])
        assignment_digests.append(certificate["assignment_sha256"])
    return (
        tuple(certificates),
        tuple(acceptances),
        tuple(certificate_digests),
        tuple(acceptance_digests),
        tuple(assignment_digests),
    )


def _recomputed_hardness() -> tuple[
    VerifiedHardness,
    VerifiedProblem,
    VerifiedHost,
    VerifiedStartingEmbedding,
    PartialContext,
    CandidateProtocolResult,
    tuple[dict[str, object], ...],
    dict[str, object],
]:
    problem = _verified_problem()
    host = _verified_host()
    starting = _verified_starting(problem, host)
    context = _verified_context(problem, host, starting)
    result = _candidate_result(host, context)
    certificates, acceptances, cert_digests, acceptance_digests, assignment_digests = (
        _continuation_inputs(problem, host, context, result)
    )
    arguments: dict[str, object] = {
        "profile_id": "unit-partial-profile-v1",
        "solver_profile_sha256": _SOLVER_PROFILE_SHA256,
        "problem": problem,
        "host": host,
        "starting_embedding": starting,
        "witness_embedding": starting,
        "partial_context": context,
        "expected_problem_sha256": problem.problem_sha256,
        "expected_host_artifact_sha256": host.host_artifact_sha256,
        "expected_host_sha256": host.host_sha256,
        "expected_starting_embedding_sha256": starting.starting_embedding_sha256,
        "expected_witness_embedding_sha256": starting.starting_embedding_sha256,
        "expected_partial_context_sha256": context.partial_context_sha256,
        "candidate_sample_seed_key": _CANDIDATE_SEED,
        "expected_candidate_protocol_result_sha256": candidate_protocol_result_sha256(result),
        "continuation_certificates": certificates,
        "checker_acceptances": acceptances,
        "expected_continuation_certificate_sha256s": cert_digests,
        "expected_checker_acceptance_sha256s": acceptance_digests,
        "expected_assignment_sha256s": assignment_digests,
        "checker_environment": _CHECKER_ENVIRONMENT,
        "expected_checker_source_sha256": continuation_checker_source_sha256(),
        "expected_checker_environment_sha256": canonical_sha256(_CHECKER_ENVIRONMENT),
    }
    record = recompute_hardness(**arguments)
    return record, problem, host, starting, context, result, certificates, arguments


def test_verified_boundaries_are_content_bound_and_immutable() -> None:
    problem = _verified_problem()
    host = _verified_host()
    starting_document = _starting_artifact(problem, host)
    starting = verify_starting_embedding(
        starting_document,
        problem=problem,
        host=host,
        expected_starting_embedding_sha256=starting_document["starting_embedding_sha256"],
    )
    context_document = _context_artifact(problem, host, starting)
    context = verify_partial_context(
        context_document,
        problem=problem,
        host=host,
        starting_embedding=starting,
        witness_embedding=starting,
        expected_partial_context_sha256=context_document["partial_context_sha256"],
    )

    starting_document["chains"][0][1].append(7)
    context_document["window_nodes"].append(7)

    assert starting.chains[0] == FrozenChain(0, (0,))
    assert context.window_nodes == (0, 1, 4, 5)
    assert context.frozen_chains == (FrozenChain(0, (0,)),)
    with pytest.raises((AttributeError, TypeError)):
        context.window_nodes += (7,)


def test_partial_context_separates_placed_focus_and_unplaced_variables() -> None:
    problem = _verified_problem()
    host = _verified_host()
    starting = _verified_starting(problem, host)

    context = _verified_context(problem, host, starting)

    assert context.focus == 1
    assert context.placed_variables == (0,)
    assert context.unplaced_variables == (2,)
    assert context.required_focus_neighbors == (0,)
    assert context.deferred_logical_edges == ((1, 2),)
    assert context.q_cap is None
    assert context.witness_total_qubits == 3
    assert context.terminal_q_cap == 3


def test_complete_starting_embedding_is_required_but_partial_context_is_not_complete() -> None:
    problem = _verified_problem()
    host = _verified_host()
    malformed = _starting_artifact(problem, host)
    malformed["chains"] = malformed["chains"][:-1]
    payload = {key: value for key, value in malformed.items() if key != "starting_embedding_sha256"}
    malformed["starting_embedding_sha256"] = canonical_sha256(payload)

    with pytest.raises(ValueError, match="every logical variable"):
        verify_starting_embedding(
            malformed,
            problem=problem,
            host=host,
            expected_starting_embedding_sha256=malformed["starting_embedding_sha256"],
        )

    starting = _verified_starting(problem, host)
    assert _verified_context(problem, host, starting).unplaced_variables == (2,)


def test_partial_context_requires_null_immediate_cap_and_derives_terminal_cap() -> None:
    problem = _verified_problem()
    host = _verified_host()
    starting = _verified_starting(problem, host)
    document = _context_artifact(problem, host, starting)
    document["q_cap"] = 3
    payload = {key: value for key, value in document.items() if key != "partial_context_sha256"}
    document["partial_context_sha256"] = canonical_sha256(payload)

    with pytest.raises(ValueError, match="q_cap must be null"):
        verify_partial_context(
            document,
            problem=problem,
            host=host,
            starting_embedding=starting,
            witness_embedding=starting,
            expected_partial_context_sha256=document["partial_context_sha256"],
        )


def test_terminal_quota_is_replayed_from_authenticated_witness_not_caller_shape() -> None:
    problem = _verified_problem()
    host = _verified_host()
    starting = _verified_starting(problem, host)
    document = _context_artifact(problem, host, starting)
    document["witness_total_qubits"] = 2
    document["terminal_q_cap"] = 2
    payload = {key: value for key, value in document.items() if key != "partial_context_sha256"}
    document["partial_context_sha256"] = canonical_sha256(payload)

    with pytest.raises(ValueError, match="authenticated witness embedding"):
        verify_partial_context(
            document,
            problem=problem,
            host=host,
            starting_embedding=starting,
            witness_embedding=starting,
            expected_partial_context_sha256=document["partial_context_sha256"],
        )


def test_terminal_quota_uses_registered_witness_not_selected_starting_embedding() -> None:
    problem = _verified_problem()
    host = _verified_host()
    witness = _verified_starting(problem, host)
    selected_artifact = build_starting_embedding_artifact(
        problem=problem,
        host=host,
        source="minorminer",
        source_artifact_sha256="4" * 64,
        chains=(
            FrozenChain(0, (0,)),
            FrozenChain(1, (1, 4)),
            FrozenChain(2, (5,)),
        ),
    )
    selected = verify_starting_embedding(
        selected_artifact,
        problem=problem,
        host=host,
        expected_starting_embedding_sha256=selected_artifact["starting_embedding_sha256"],
    )

    context_artifact = build_partial_context_artifact(
        problem=problem,
        host=host,
        starting_embedding=selected,
        witness_embedding=witness,
        focus=1,
        frozen_variables=(0,),
        window_nodes=(0, 1, 4, 5),
        window_edges=((0, 4), (0, 5), (1, 4), (1, 5)),
        l_cap=2,
        q_cap=None,
        q_cap_slack=0,
        removal_rule_sha256=_REMOVAL_RULE_SHA256,
    )
    context = verify_partial_context(
        context_artifact,
        problem=problem,
        host=host,
        starting_embedding=selected,
        witness_embedding=witness,
        expected_partial_context_sha256=context_artifact["partial_context_sha256"],
    )

    assert selected.total_qubits == 4
    assert witness.total_qubits == 3
    assert context.witness_total_qubits == 3
    assert context.terminal_q_cap == 3
    assert context.witness_embedding_sha256 == witness.starting_embedding_sha256
    with pytest.raises(ValueError, match="registered witness"):
        verify_partial_context(
            context_artifact,
            problem=problem,
            host=host,
            starting_embedding=selected,
            witness_embedding=selected,
            expected_partial_context_sha256=context_artifact["partial_context_sha256"],
        )


def test_rejects_singleton_logical_problem_for_exact_density() -> None:
    problem = IsingProblem(
        variables=(0,),
        linear=((0, 0.0),),
        quadratic=(),
    )

    with pytest.raises(ValueError, match="at least two"):
        verify_problem(problem, expected_problem_sha256=problem.problem_sha256)


def test_verified_host_uses_one_detached_artifact_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    import embedbench.hardness as hardness_module

    artifact = _host_artifact()
    original_validator = hardness_module.validate_realized_host_artifact

    def mutate_caller_after_validation(snapshot: object, **kwargs: object):
        graph = original_validator(snapshot, **kwargs)
        artifact["removed_nodes"].append(7)
        return graph

    monkeypatch.setattr(
        hardness_module,
        "validate_realized_host_artifact",
        mutate_caller_after_validation,
    )
    verified = verify_host(
        artifact,
        seed_key=_HOST_SEED,
        expected_host_artifact_sha256=artifact["host_artifact_sha256"],
        expected_host_sha256=artifact["host_sha256"],
    )

    assert verified.removed_nodes == ()
    assert 7 in verified.nodes
    assert verified.pristine_node_count == 8


def test_starting_and_partial_boundaries_do_not_reread_caller_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import embedbench.hardness as hardness_module

    problem = _verified_problem()
    host = _verified_host()
    starting_document = _starting_artifact(problem, host)
    original_embedding_validator = hardness_module.validate_minor_embedding

    def mutate_starting_after_validation(*args: object, **kwargs: object) -> None:
        original_embedding_validator(*args, **kwargs)
        starting_document["chains"][0][1].append(7)

    monkeypatch.setattr(
        hardness_module,
        "validate_minor_embedding",
        mutate_starting_after_validation,
    )
    starting = verify_starting_embedding(
        starting_document,
        problem=problem,
        host=host,
        expected_starting_embedding_sha256=starting_document["starting_embedding_sha256"],
    )
    assert starting.chain_for(0) == (0,)

    context_document = _context_artifact(problem, host, starting)
    original_window_validator = hardness_module.validate_window

    def mutate_context_after_validation(*args: object, **kwargs: object) -> None:
        original_window_validator(*args, **kwargs)
        context_document["window_nodes"].append(7)

    monkeypatch.setattr(
        hardness_module,
        "validate_window",
        mutate_context_after_validation,
    )
    context = verify_partial_context(
        context_document,
        problem=problem,
        host=host,
        starting_embedding=starting,
        witness_embedding=starting,
        expected_partial_context_sha256=context_document["partial_context_sha256"],
    )
    assert context.window_nodes == (0, 1, 4, 5)


def test_boundary_artifact_schemas_are_explicit() -> None:
    problem = _verified_problem()
    host = _verified_host()
    starting = _starting_artifact(problem, host)
    verified_starting = _verified_starting(problem, host)
    context = _context_artifact(problem, host, verified_starting)

    assert starting["schema"] == STARTING_EMBEDDING_SCHEMA
    assert context["schema"] == PARTIAL_CONTEXT_SCHEMA
    assert type(starting["starting_embedding_sha256"]) is str
    assert type(context["partial_context_sha256"]) is str


@pytest.mark.parametrize("invalid_version", [True, 1.0])
def test_external_boundaries_reject_non_integer_schema_versions(invalid_version: object) -> None:
    problem = _verified_problem()
    host = _verified_host()
    starting_document = _starting_artifact(problem, host)
    starting_document["schema_version"] = invalid_version
    starting_payload = {
        key: value for key, value in starting_document.items() if key != "starting_embedding_sha256"
    }
    starting_document["starting_embedding_sha256"] = canonical_sha256(starting_payload)
    with pytest.raises(ValueError, match="schema_version"):
        verify_starting_embedding(
            starting_document,
            problem=problem,
            host=host,
            expected_starting_embedding_sha256=starting_document["starting_embedding_sha256"],
        )

    starting = _verified_starting(problem, host)
    context_document = _context_artifact(problem, host, starting)
    context_document["schema_version"] = invalid_version
    context_payload = {
        key: value for key, value in context_document.items() if key != "partial_context_sha256"
    }
    context_document["partial_context_sha256"] = canonical_sha256(context_payload)
    with pytest.raises(ValueError, match="schema_version"):
        verify_partial_context(
            context_document,
            problem=problem,
            host=host,
            starting_embedding=starting,
            witness_embedding=starting,
            expected_partial_context_sha256=context_document["partial_context_sha256"],
        )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    [
        ("frozen_chains", [[0, (0,)]]),
        ("window_nodes", (0, 1, 4, 5)),
        ("window_edges", [(0, 4), (0, 5), (1, 4), (1, 5)]),
    ],
)
def test_partial_context_rejects_non_json_tuple_arrays(
    field_name: str,
    replacement: object,
) -> None:
    problem = _verified_problem()
    host = _verified_host()
    starting = _verified_starting(problem, host)
    document = _context_artifact(problem, host, starting)
    document[field_name] = replacement
    payload = {key: value for key, value in document.items() if key != "partial_context_sha256"}
    document["partial_context_sha256"] = canonical_sha256(payload)

    with pytest.raises(TypeError, match="exact JSON array"):
        verify_partial_context(
            document,
            problem=problem,
            host=host,
            starting_embedding=starting,
            witness_embedding=starting,
            expected_partial_context_sha256=document["partial_context_sha256"],
        )


def test_candidate_indexed_hardness_uses_post_replacement_partial_state() -> None:
    record, _, _, _, _, _, _, _ = _recomputed_hardness()
    document = record.to_dict()

    assert document["schema"] == HARDNESS_SCHEMA
    assert document["partial_context"] == {
        "placed_variables": 1,
        "unplaced_variables": 1,
        "focus": 1,
        "occupied_qubits": 1,
        "q_cap": None,
        "terminal_q_cap": 3,
    }
    candidates = {tuple(candidate["candidate"]): candidate for candidate in document["candidates"]}
    short = candidates[(4,)]
    long = candidates[(1, 4)]
    assert short["immediate"]["current_total_qubits"] == 2
    assert short["post_replacement_embedding"]["total_qubits"] == 2
    assert short["residual"]["free_node_numerator"] == 6
    assert long["immediate"]["current_total_qubits"] == 3
    assert long["post_replacement_embedding"]["total_qubits"] == 3
    assert long["residual"]["free_node_numerator"] == 5
    assert long["post_replacement_embedding"] != short["post_replacement_embedding"]
    assert long["residual"] != short["residual"]


def test_immediate_required_couplers_exclude_deferred_focus_edges() -> None:
    record, _, _, _, _, _, _, _ = _recomputed_hardness()
    document = record.to_dict()

    assert all(
        candidate["deferred_logical_edges"] == [[1, 2]] for candidate in document["candidates"]
    )
    assert all(
        candidate["immediate"]["realizes_every_required_coupler"]
        for candidate in document["candidates"]
    )
    assert all(candidate["immediate"]["within_q_cap"] for candidate in document["candidates"])
    assert all(candidate["immediate"]["feasible_now"] for candidate in document["candidates"])
    assert "feasible_candidate_fraction" not in document["decision"]
    assert "budget_survivor_fraction" not in document["decision"]


def test_terminal_cap_is_checked_only_by_exact_continuation() -> None:
    record, _, _, _, context, _, _, _ = _recomputed_hardness()
    document = record.to_dict()
    candidates = {tuple(candidate["candidate"]): candidate for candidate in document["candidates"]}

    assert context.q_cap is None
    assert candidates[(4,)]["continuation"]["completion_feasible"] is True
    assert candidates[(4,)]["continuation"]["terminal_total_qubits"] == 3
    assert candidates[(1, 4)]["immediate"]["feasible_now"] is True
    assert candidates[(1, 4)]["continuation"]["completion_feasible"] is False
    assert candidates[(1, 4)]["continuation"]["terminal_total_qubits"] is None


def test_reference_candidate_is_original_focus_chain_full_bank_index() -> None:
    record, _, _, _, context, result, _, _ = _recomputed_hardness()
    document = record.to_dict()
    assert result.full_candidates is not None

    assert document["reference_candidate_index"] == result.full_candidates.index(
        context.original_focus_chain
    )
    assert [candidate["canonical_index"] for candidate in document["candidates"]] == list(
        result.offered_to_full_indices
    )


def test_exact_completion_status_is_replayed_not_trusted_from_certificate() -> None:
    problem = _verified_problem()
    host = _verified_host()
    starting = _verified_starting(problem, host)
    context = _verified_context(problem, host, starting)
    result = _candidate_result(host, context)
    certificates, acceptances, cert_digests, acceptance_digests, assignment_digests = (
        _continuation_inputs(problem, host, context, result)
    )
    forged_certificates = list(copy.deepcopy(certificates))
    forged_acceptances = list(copy.deepcopy(acceptances))
    feasible_position = next(
        index
        for index, certificate in enumerate(forged_certificates)
        if certificate["completion_feasible"] is True
    )
    certificate = forged_certificates[feasible_position]
    for field_name in (
        "terminal_total_qubits",
        "terminal_maximum_chain_length",
        "largest_free_component_node_numerator",
        "largest_free_component_node_denominator",
        "largest_free_component_edge_connectivity",
        "largest_free_component_articulation_count",
        "minimum_logical_contact_multiplicity",
        "assignment_sha256",
    ):
        certificate[field_name] = None
    certificate["completion_feasible"] = False
    certificate_payload = {
        key: value for key, value in certificate.items() if key != "certificate_sha256"
    }
    certificate["certificate_sha256"] = canonical_sha256(certificate_payload)
    acceptance = forged_acceptances[feasible_position]
    acceptance["certificate_sha256"] = certificate["certificate_sha256"]
    acceptance["assignment_sha256"] = None
    acceptance_payload = {
        key: value for key, value in acceptance.items() if key != "checker_acceptance_sha256"
    }
    acceptance["checker_acceptance_sha256"] = canonical_sha256(acceptance_payload)
    forged_cert_digests = list(cert_digests)
    forged_acceptance_digests = list(acceptance_digests)
    forged_assignments = list(assignment_digests)
    forged_cert_digests[feasible_position] = certificate["certificate_sha256"]
    forged_acceptance_digests[feasible_position] = acceptance["checker_acceptance_sha256"]
    forged_assignments[feasible_position] = None

    with pytest.raises(ValueError, match="checker replay"):
        recompute_hardness(
            profile_id="unit-partial-profile-v1",
            solver_profile_sha256=_SOLVER_PROFILE_SHA256,
            problem=problem,
            host=host,
            starting_embedding=starting,
            witness_embedding=starting,
            partial_context=context,
            expected_problem_sha256=problem.problem_sha256,
            expected_host_artifact_sha256=host.host_artifact_sha256,
            expected_host_sha256=host.host_sha256,
            expected_starting_embedding_sha256=starting.starting_embedding_sha256,
            expected_witness_embedding_sha256=starting.starting_embedding_sha256,
            expected_partial_context_sha256=context.partial_context_sha256,
            candidate_sample_seed_key=_CANDIDATE_SEED,
            expected_candidate_protocol_result_sha256=candidate_protocol_result_sha256(result),
            continuation_certificates=tuple(forged_certificates),
            checker_acceptances=tuple(forged_acceptances),
            expected_continuation_certificate_sha256s=tuple(forged_cert_digests),
            expected_checker_acceptance_sha256s=tuple(forged_acceptance_digests),
            expected_assignment_sha256s=tuple(forged_assignments),
            checker_environment=_CHECKER_ENVIRONMENT,
            expected_checker_source_sha256=continuation_checker_source_sha256(),
            expected_checker_environment_sha256=canonical_sha256(_CHECKER_ENVIRONMENT),
        )


@pytest.mark.parametrize(
    "override",
    [
        {"expected_candidate_protocol_result_sha256": "f" * 64},
        {"expected_checker_source_sha256": "f" * 64},
        {"expected_checker_environment_sha256": "f" * 64},
    ],
)
def test_every_computation_source_requires_an_external_digest(override: dict[str, object]) -> None:
    problem = _verified_problem()
    host = _verified_host()
    starting = _verified_starting(problem, host)
    context = _verified_context(problem, host, starting)
    result = _candidate_result(host, context)
    certificates, acceptances, cert_digests, acceptance_digests, assignment_digests = (
        _continuation_inputs(problem, host, context, result)
    )
    arguments: dict[str, object] = {
        "profile_id": "unit-partial-profile-v1",
        "solver_profile_sha256": _SOLVER_PROFILE_SHA256,
        "problem": problem,
        "host": host,
        "starting_embedding": starting,
        "witness_embedding": starting,
        "partial_context": context,
        "expected_problem_sha256": problem.problem_sha256,
        "expected_host_artifact_sha256": host.host_artifact_sha256,
        "expected_host_sha256": host.host_sha256,
        "expected_starting_embedding_sha256": starting.starting_embedding_sha256,
        "expected_witness_embedding_sha256": starting.starting_embedding_sha256,
        "expected_partial_context_sha256": context.partial_context_sha256,
        "candidate_sample_seed_key": _CANDIDATE_SEED,
        "expected_candidate_protocol_result_sha256": candidate_protocol_result_sha256(result),
        "continuation_certificates": certificates,
        "checker_acceptances": acceptances,
        "expected_continuation_certificate_sha256s": cert_digests,
        "expected_checker_acceptance_sha256s": acceptance_digests,
        "expected_assignment_sha256s": assignment_digests,
        "checker_environment": _CHECKER_ENVIRONMENT,
        "expected_checker_source_sha256": continuation_checker_source_sha256(),
        "expected_checker_environment_sha256": canonical_sha256(_CHECKER_ENVIRONMENT),
    }
    arguments.update(override)

    with pytest.raises(ValueError, match="digest|source|environment"):
        recompute_hardness(**arguments)


def test_continuation_assignment_requires_an_independent_external_digest() -> None:
    problem = _verified_problem()
    host = _verified_host()
    starting = _verified_starting(problem, host)
    context = _verified_context(problem, host, starting)
    result = _candidate_result(host, context)
    certificates, acceptances, cert_digests, acceptance_digests, assignment_digests = (
        _continuation_inputs(problem, host, context, result)
    )
    assert result.offered_to_full_indices is not None
    assert assignment_digests[0] != "f" * 64

    with pytest.raises(ValueError, match="assignment digest"):
        verify_continuation_evidence(
            certificates[0],
            acceptances[0],
            problem=problem,
            host=host,
            partial_context=context,
            candidate_protocol_result=result,
            candidate_sample_seed_key=_CANDIDATE_SEED,
            expected_candidate_protocol_result_sha256=candidate_protocol_result_sha256(result),
            canonical_index=result.offered_to_full_indices[0],
            checker_environment=_CHECKER_ENVIRONMENT,
            expected_certificate_sha256=cert_digests[0],
            expected_checker_acceptance_sha256=acceptance_digests[0],
            expected_checker_source_sha256=continuation_checker_source_sha256(),
            expected_checker_environment_sha256=canonical_sha256(_CHECKER_ENVIRONMENT),
            expected_assignment_sha256="f" * 64,
        )


def test_continuation_rejects_cross_problem_context() -> None:
    original_problem = _verified_problem()
    altered_problem_value = IsingProblem(
        variables=(0, 1, 2),
        linear=((0, 0.0), (1, 0.0), (2, 0.0)),
        quadratic=((0, 1, -0.5), (1, 2, 1.0)),
    )
    altered_problem = verify_problem(
        altered_problem_value,
        expected_problem_sha256=altered_problem_value.problem_sha256,
    )
    host = _verified_host()
    starting = _verified_starting(original_problem, host)
    context = _verified_context(original_problem, host, starting)
    result = _candidate_result(host, context)
    assert result.offered_to_full_indices is not None

    with pytest.raises(ValueError, match="same problem and host identity"):
        build_continuation_artifacts(
            problem=altered_problem,
            host=host,
            partial_context=context,
            candidate_protocol_result=result,
            candidate_sample_seed_key=_CANDIDATE_SEED,
            expected_candidate_protocol_result_sha256=candidate_protocol_result_sha256(result),
            canonical_index=result.offered_to_full_indices[0],
            checker_environment=_CHECKER_ENVIRONMENT,
        )


def test_continuation_rejects_forged_candidate_facts_even_with_self_digest() -> None:
    problem = _verified_problem()
    host = _verified_host()
    starting = _verified_starting(problem, host)
    context = _verified_context(problem, host, starting)
    result = _candidate_result(host, context)
    assert result.full_candidate_facts is not None
    assert result.offered_to_full_indices is not None
    forged_facts = list(result.full_candidate_facts)
    forged_facts[0] = replace(
        forged_facts[0],
        current_total_qubits=forged_facts[0].current_total_qubits + 1,
    )
    forged_result = replace(result, full_candidate_facts=tuple(forged_facts))

    with pytest.raises(ValueError, match="exact replay"):
        build_continuation_artifacts(
            problem=problem,
            host=host,
            partial_context=context,
            candidate_protocol_result=forged_result,
            candidate_sample_seed_key=_CANDIDATE_SEED,
            expected_candidate_protocol_result_sha256=candidate_protocol_result_sha256(
                forged_result
            ),
            canonical_index=result.offered_to_full_indices[0],
            checker_environment=_CHECKER_ENVIRONMENT,
        )


def test_node_budget_exhaustion_never_becomes_an_exact_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import embedbench.hardness as hardness_module

    monkeypatch.setattr(hardness_module, "EXACT_COMPLETION_NODE_BUDGET", 1)
    problem = _verified_problem()
    host = _verified_host()
    starting = _verified_starting(problem, host)
    context = _verified_context(problem, host, starting)
    result = _candidate_result(host, context)
    certificates, acceptances, cert_digests, acceptance_digests, assignment_digests = (
        _continuation_inputs(problem, host, context, result)
    )

    record = recompute_hardness(
        profile_id="unit-partial-profile-v1",
        solver_profile_sha256=_SOLVER_PROFILE_SHA256,
        problem=problem,
        host=host,
        starting_embedding=starting,
        witness_embedding=starting,
        partial_context=context,
        expected_problem_sha256=problem.problem_sha256,
        expected_host_artifact_sha256=host.host_artifact_sha256,
        expected_host_sha256=host.host_sha256,
        expected_starting_embedding_sha256=starting.starting_embedding_sha256,
        expected_witness_embedding_sha256=starting.starting_embedding_sha256,
        expected_partial_context_sha256=context.partial_context_sha256,
        candidate_sample_seed_key=_CANDIDATE_SEED,
        expected_candidate_protocol_result_sha256=candidate_protocol_result_sha256(result),
        continuation_certificates=certificates,
        checker_acceptances=acceptances,
        expected_continuation_certificate_sha256s=cert_digests,
        expected_checker_acceptance_sha256s=acceptance_digests,
        expected_assignment_sha256s=assignment_digests,
        checker_environment=_CHECKER_ENVIRONMENT,
        expected_checker_source_sha256=continuation_checker_source_sha256(),
        expected_checker_environment_sha256=canonical_sha256(_CHECKER_ENVIRONMENT),
    )
    document = record.to_dict()

    exhausted = [
        candidate["continuation"]
        for candidate in document["candidates"]
        if candidate["continuation"]["exact_completion_status"] == "node_budget_exhausted"
    ]
    assert exhausted
    exact_fields = {
        "completion_feasible",
        "terminal_total_qubits",
        "terminal_maximum_chain_length",
        "largest_free_component_node_numerator",
        "largest_free_component_node_denominator",
        "largest_free_component_edge_connectivity",
        "largest_free_component_articulation_count",
        "minimum_logical_contact_multiplicity",
    }
    assert all(
        all(continuation[field] is None for field in exact_fields) for continuation in exhausted
    )
    assert document["decision"]["exact_completion_numerator"] is None
    assert document["decision"]["exact_completion_fraction"] is None
    assert document["decision"]["exact_completion_denominator"] == len(document["candidates"])


@pytest.mark.parametrize(
    ("node_budget", "expected_status", "expected_expanded"),
    [
        (2, "node_budget_exhausted", 2),
        (3, "complete", 3),
        (4, "complete", 3),
    ],
)
def test_exact_continuation_budget_counts_only_dfs_states(
    node_budget: int,
    expected_status: str,
    expected_expanded: int,
) -> None:
    import embedbench.hardness as hardness_module

    problem = _verified_problem()
    host = _verified_host()
    starting = _verified_starting(problem, host)
    context = _verified_context(problem, host, starting)
    result = _candidate_result(host, context)
    assert result.full_candidates is not None
    canonical_index = result.full_candidates.index((4,))

    replay = hardness_module._replay_continuation(
        problem=problem,
        host=host,
        context=context,
        canonical_index=canonical_index,
        candidate=(4,),
        node_budget=node_budget,
    )

    assert replay["exact_completion_status"] == expected_status
    assert replay["expanded_nodes"] == expected_expanded


def test_hardness_record_does_not_alias_continuation_documents() -> None:
    record, _, _, _, _, _, certificates, _ = _recomputed_hardness()
    before = record.to_dict()
    certificates[0]["completion_feasible"] = not certificates[0]["completion_feasible"]

    assert record.to_dict() == before
    assert validate_hardness(record) is record
    with pytest.raises((AttributeError, TypeError)):
        record.hardness_sha256 = "f" * 64


def test_hardness_has_no_public_document_mint_and_validates_closed_schema() -> None:
    import embedbench.hardness as hardness_module

    assert not hasattr(VerifiedHardness, "_from_document")
    forged_document = {
        "schema": HARDNESS_SCHEMA,
        "schema_version": 1,
        "decision": {"exact_completion_fraction": 1.0},
        "forged": True,
    }
    forged = object.__new__(VerifiedHardness)
    object.__setattr__(
        forged,
        "_serialized",
        json.dumps(forged_document, sort_keys=True, separators=(",", ":")),
    )
    object.__setattr__(forged, "hardness_sha256", canonical_sha256(forged_document))
    object.__setattr__(forged, "_seal", hardness_module._VERIFICATION_SEAL)

    with pytest.raises(ValueError, match="schema fields differ"):
        validate_hardness(forged)


def test_hardness_rejects_subjective_free_profile_id() -> None:
    _, _, _, _, _, _, _, arguments = _recomputed_hardness()
    arguments["profile_id"] = "easy"

    with pytest.raises(ValueError, match="subjective hardness label"):
        recompute_hardness(**arguments)


def test_model_view_is_positive_whitelist_without_provenance_or_continuation() -> None:
    record, _, _, _, _, _, _, _ = _recomputed_hardness()

    features = hardness_model_inputs(record)
    serialized = repr(features)

    assert "sources" not in features
    assert "profile_id" not in features
    assert "solver_profile_sha256" not in features
    assert "reference_candidate_index" not in features
    assert "starting_embedding" not in features
    assert "continuation" not in serialized
    assert "certificate_sha256" not in serialized
    assert features["candidates"][0]["deferred_logical_edges"] == [[1, 2]]


def test_stored_hardness_is_content_bound_and_recomputed() -> None:
    record, _, _, _, _, _, _, arguments = _recomputed_hardness()
    stored = record.to_dict()
    verified = verify_hardness(
        stored,
        expected_hardness_sha256=record.hardness_sha256,
        recompute_arguments=arguments,
    )
    assert verified.to_dict() == record.to_dict()
    assert verified is not record

    stored["decision"]["exact_completion_numerator"] = 0
    forged_digest = canonical_sha256(stored)
    with pytest.raises(ValueError, match="recomputed"):
        verify_hardness(
            stored,
            expected_hardness_sha256=forged_digest,
            recompute_arguments=arguments,
        )


def test_continuation_artifacts_bind_checker_environment_and_assignment() -> None:
    problem = _verified_problem()
    host = _verified_host()
    starting = _verified_starting(problem, host)
    context = _verified_context(problem, host, starting)
    result = _candidate_result(host, context)
    certificates, acceptances, _, _, _ = _continuation_inputs(problem, host, context, result)

    assert all(
        certificate["schema"] == CONTINUATION_CERTIFICATE_SCHEMA for certificate in certificates
    )
    assert all(
        acceptance["checker_source_sha256"] == continuation_checker_source_sha256()
        for acceptance in acceptances
    )
    assert all(
        acceptance["checker_environment_sha256"] == canonical_sha256(_CHECKER_ENVIRONMENT)
        for acceptance in acceptances
    )
    for certificate, acceptance in zip(certificates, acceptances, strict=True):
        assert acceptance["assignment_sha256"] == certificate["assignment_sha256"]


def test_residual_formula_preserves_exact_denominators_and_lcc_tie_break() -> None:
    graph = _verified_host().graph()
    residual = residual_hardness(graph, [0, 4])

    assert residual["free_node_denominator"] == 8
    assert residual["free_edge_denominator"] == 16
    assert residual["largest_free_component_node_denominator"] == 8
    assert residual["largest_free_component_edge_denominator"] == 16
