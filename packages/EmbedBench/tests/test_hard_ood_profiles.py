from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import pytest
from embedbench.candidate_protocol import (
    FrozenChain,
    candidate_protocol_result_sha256,
    generate_candidate_bank,
)
from embedbench.ground_certificate import (
    CheckerRun,
    IsingProblem,
    SolverRun,
    build_exact_enumeration_certificate,
)
from embedbench.hard_ood_profiles import (
    CELL_PROFILE_SCHEMA,
    HARD_OOD_RELEASE_ID,
    PLAN_ROW_SCHEMA,
    PROFILE_SET_SCHEMA,
    build_quota_entry,
    cell_matches,
    classify_registered_bands,
    defect_band,
    expected_quota_total,
    host_fill_band,
    logical_degree_band,
    quota_deficits,
    recompute_cell_facts,
    registered_cell_requirements,
    residual_lcc_band,
    validate_cell_facts,
    validate_cell_profile,
    validate_exact_quotas,
    validate_plan_row,
    validate_quota_entry,
    validate_registered_profile_set,
    verify_cell_facts,
    verify_plan_row,
    verify_quota_entry,
    verify_registered_profile_set,
)
from embedbench.hard_ood_schema import (
    SeedRegistration,
    SeedRequest,
    VerifiedSeedResolver,
    canonical_sha256,
    verify_seed_registry_shards,
    write_seed_registry_shards,
)
from embedbench.hardness import (
    build_continuation_artifacts,
    build_partial_context_artifact,
    build_starting_embedding_artifact,
    continuation_checker_source_sha256,
    recompute_hardness,
    verify_host,
    verify_partial_context,
    verify_problem,
    verify_starting_embedding,
)
from embedbench.realized_host import build_realized_host_artifact

_SEED_KEY = bytes(range(32))
_SHA256 = "2" * 64
_CANDIDATE_SEED = bytes(reversed(range(32)))
_CHECKER_ENVIRONMENT = {
    "schema": "embedbench.profile-test-checker-environment",
    "schema_version": 1,
}


def _verify_plan_document(
    document: dict[str, object],
    *,
    expected_plan_row_sha256: str,
    profile_set,
    registered_request: SeedRequest | None = None,
    seed32_required: bool = False,
    expected_seed_root_sha256: str | None = None,
):
    request = (
        SeedRequest.from_dict(document["problem_seed_request"])
        if registered_request is None
        else registered_request
    )
    with tempfile.TemporaryDirectory(prefix="embedbench-plan-seed-") as temporary:
        registry_directory = Path(temporary).resolve() / "registry"
        root = write_seed_registry_shards(
            [SeedRegistration(request, seed32_required=seed32_required)],
            directory=registry_directory,
            release_id=HARD_OOD_RELEASE_ID,
            stage_id="structure-foundation",
        )
        verified_registry = verify_seed_registry_shards(
            registry_directory,
            expected_root_sha256=root.sha256,
        )
        resolver = VerifiedSeedResolver(
            verified_registry,
            expected_terminal_root_sha256=root.sha256,
        )
        return verify_plan_row(
            document,
            expected_plan_row_sha256=expected_plan_row_sha256,
            profile_set=profile_set,
            seed_resolver=resolver,
            expected_seed_registry_terminal_root_sha256=(
                root.sha256 if expected_seed_root_sha256 is None else expected_seed_root_sha256
            ),
        )


def _ising_problem() -> IsingProblem:
    return IsingProblem(
        variables=(0, 1, 2),
        linear=((0, 0.0), (1, 0.0), (2, 0.0)),
        quadratic=((0, 1, -1.0), (1, 2, 1.0)),
    )


def _scientific_sources(profile_id: str = "capacity_transition_v1") -> dict[str, object]:
    ising_problem = _ising_problem()
    problem = verify_problem(ising_problem, expected_problem_sha256=ising_problem.problem_sha256)
    artifact = build_realized_host_artifact(
        topology="chimera",
        size=5,
        qubit_fraction=0.0,
        coupler_fraction=0.0,
        seed_key=_SEED_KEY,
    )
    host = verify_host(
        artifact,
        seed_key=_SEED_KEY,
        expected_host_artifact_sha256=artifact["host_artifact_sha256"],
        expected_host_sha256=artifact["host_sha256"],
    )
    starting_artifact = build_starting_embedding_artifact(
        problem=problem,
        host=host,
        source="witness",
        source_artifact_sha256="1" * 64,
        chains=(FrozenChain(0, (0,)), FrozenChain(1, (4,)), FrozenChain(2, (1,))),
    )
    starting = verify_starting_embedding(
        starting_artifact,
        problem=problem,
        host=host,
        expected_starting_embedding_sha256=starting_artifact["starting_embedding_sha256"],
    )
    context_artifact = build_partial_context_artifact(
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
        removal_rule_sha256="4" * 64,
    )
    context = verify_partial_context(
        context_artifact,
        problem=problem,
        host=host,
        starting_embedding=starting,
        witness_embedding=starting,
        expected_partial_context_sha256=context_artifact["partial_context_sha256"],
    )
    candidate_result = generate_candidate_bank(
        host.graph(),
        window_nodes=context.window_nodes,
        frozen_chains=context.frozen_chains,
        required_logical_neighbors=context.required_focus_neighbors,
        original_focus_chain=context.original_focus_chain,
        l_cap=context.l_cap,
        q_cap=context.q_cap,
        candidate_sample_seed_key=_CANDIDATE_SEED,
    )
    certificates = []
    acceptances = []
    certificate_digests = []
    acceptance_digests = []
    assignment_digests = []
    for canonical_index in candidate_result.offered_to_full_indices:
        certificate, acceptance = build_continuation_artifacts(
            problem=problem,
            host=host,
            partial_context=context,
            candidate_protocol_result=candidate_result,
            candidate_sample_seed_key=_CANDIDATE_SEED,
            expected_candidate_protocol_result_sha256=candidate_protocol_result_sha256(
                candidate_result
            ),
            canonical_index=canonical_index,
            checker_environment=_CHECKER_ENVIRONMENT,
        )
        certificates.append(certificate)
        acceptances.append(acceptance)
        certificate_digests.append(certificate["certificate_sha256"])
        acceptance_digests.append(acceptance["checker_acceptance_sha256"])
        assignment_digests.append(certificate["assignment_sha256"])
    hardness = recompute_hardness(
        profile_id=profile_id,
        solver_profile_sha256=_SHA256,
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
        expected_candidate_protocol_result_sha256=candidate_protocol_result_sha256(
            candidate_result
        ),
        continuation_certificates=tuple(certificates),
        checker_acceptances=tuple(acceptances),
        expected_continuation_certificate_sha256s=tuple(certificate_digests),
        expected_checker_acceptance_sha256s=tuple(acceptance_digests),
        expected_assignment_sha256s=tuple(assignment_digests),
        checker_environment=_CHECKER_ENVIRONMENT,
        expected_checker_source_sha256=continuation_checker_source_sha256(),
        expected_checker_environment_sha256=canonical_sha256(_CHECKER_ENVIRONMENT),
    )
    ground = build_exact_enumeration_certificate(
        ising_problem,
        solver=SolverRun(
            name="embedbench-gray-code-enumerator",
            version="1",
            command=("python3", "certify.py"),
            seed=None,
            deterministic_work_limit=1 << len(ising_problem.variables),
            safety_timeout_seconds=None,
        ),
        checker=CheckerRun(
            source_sha256="5" * 64,
            environment_sha256="6" * 64,
            command=("python3", "verify.py"),
            exit_code=0,
            log_sha256="7" * 64,
        ),
    )
    profile_document = _registered_profile_document()
    for cell in profile_document["cells"]:
        if cell["cell_id"] == "hard_dev/chimera-5/capacity_transition":
            cell["profile_id"] = profile_id
    profile_set = verify_registered_profile_set(
        profile_document,
        expected_profile_set_sha256=canonical_sha256(profile_document),
    )
    problem_seed_request = SeedRequest(
        release_id=HARD_OOD_RELEASE_ID,
        purpose="problem",
        partition="hard_dev",
        panel="capacity_transition",
        cell="hard_dev/chimera-5/capacity_transition",
        task_type=None,
        problem_sha256=None,
        state_sha256=None,
        candidate_index=None,
        strength_index=None,
        label_stage=None,
        replicate=0,
    )
    plan_document = {
        "schema": PLAN_ROW_SCHEMA,
        "schema_version": 1,
        "release_id": HARD_OOD_RELEASE_ID,
        "profile_set_sha256": profile_set.profile_set_sha256,
        "cell_id": "hard_dev/chimera-5/capacity_transition",
        "partition": "hard_dev",
        "panel": "capacity_transition",
        "ordinal": 0,
        "topology": "chimera",
        "size": 5,
        "logical_source": "ink_drop",
        "logical_family": "ink_drop_quotient",
        "n_vars": 3,
        "chain_size": 3,
        "l_cap": 2,
        "max_window_free": 28,
        "q_cap_slack": 0,
        "requested_qubit_numerator": 0,
        "requested_qubit_denominator": 1,
        "requested_coupler_numerator": 0,
        "requested_coupler_denominator": 1,
        "fill_subband": 0,
        "starting_source": "witness",
        "generator_arguments_sha256": "8" * 64,
        "problem_seed_request": problem_seed_request.to_dict(),
        "problem_seed_key_sha256": problem_seed_request.seed_key_hex,
        "state_slot": 0,
        "focus_quartile": 0,
    }
    plan_document["plan_row_sha256"] = canonical_sha256(plan_document)
    plan_row = _verify_plan_document(
        plan_document,
        expected_plan_row_sha256=plan_document["plan_row_sha256"],
        profile_set=profile_set,
    )
    return {
        "hardness": hardness,
        "problem": problem,
        "host": host,
        "starting_embedding": starting,
        "partial_context": context,
        "plan_row": plan_row,
        "expected_plan_row_sha256": plan_row.plan_row_sha256,
        "profile_set": profile_set,
        "ising_problem": ising_problem,
        "ground_certificate": ground.certificate.to_dict(),
        "ground_proof": ground.proof.to_dict(),
        "expected_ground_certificate_sha256": ground.certificate.digest,
        "expected_ground_proof_artifact_sha256": ground.proof.digest,
    }


def _facts(profile_id: str = "capacity_transition_v1"):
    return recompute_cell_facts(**_scientific_sources(profile_id))


def _profile(
    *,
    cell_id: str = "chimera-capacity",
    profile_id: str = "capacity_transition_v1",
    quota: int = 2,
) -> dict[str, object]:
    return {
        "schema": CELL_PROFILE_SCHEMA,
        "schema_version": 1,
        "profile_id": profile_id,
        "cell_id": cell_id,
        "partition": "hard_dev",
        "panel": "capacity_transition",
        "quota": quota,
        "ratio_predicates": [
            {
                "field": "hardness.starting_embedding.host_fill",
                "minimum": {"numerator": 0, "denominator": 1, "inclusive": True},
                "maximum": {"numerator": 1, "denominator": 2, "inclusive": False},
            },
            {
                "field": "requested_qubit_fraction",
                "minimum": {"numerator": 0, "denominator": 1, "inclusive": True},
                "maximum": {"numerator": 0, "denominator": 1, "inclusive": True},
            },
        ],
        "integer_predicates": [
            {
                "field": "chain_size",
                "minimum": {"value": 3, "inclusive": True},
                "maximum": {"value": 4, "inclusive": True},
            }
        ],
        "value_predicates": [
            {"field": "topology", "allowed_values": ["chimera"]},
            {"field": "ground_state_eligible", "allowed_values": [True]},
        ],
        "band_predicates": [
            {"axis": "host_fill", "allowed_bands": ["low"]},
            {"axis": "defects", "allowed_bands": ["none"]},
        ],
    }


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        (44, 100, "low"),
        (45, 100, "medium"),
        (59, 100, "medium"),
        (60, 100, "high"),
        (71, 100, "high"),
        (72, 100, "extreme"),
        (144, 200, "extreme"),
    ],
)
def test_host_fill_registered_boundaries(numerator: int, denominator: int, expected: str) -> None:
    assert host_fill_band(numerator, denominator) == expected


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        (29, 10, "sparse"),
        (3, 1, "moderate"),
        (44, 10, "moderate"),
        (45, 10, "dense"),
        (59, 10, "dense"),
        (6, 1, "very_dense"),
    ],
)
def test_logical_degree_registered_boundaries(
    numerator: int, denominator: int, expected: str
) -> None:
    assert logical_degree_band(numerator, denominator) == expected


@pytest.mark.parametrize(
    ("numerator", "denominator", "expected"),
    [
        (1, 2, "open"),
        (49, 100, "constrained"),
        (1, 4, "constrained"),
        (24, 100, "critical"),
        (0, 7, "critical"),
    ],
)
def test_residual_lcc_registered_boundaries(
    numerator: int, denominator: int, expected: str
) -> None:
    assert residual_lcc_band(numerator, denominator) == expected


@pytest.mark.parametrize(
    ("q_num", "q_den", "c_num", "c_den", "expected"),
    [
        (0, 10, 0, 10, "none"),
        (1, 50, 0, 10, "light"),
        (0, 10, 1, 20, "light"),
        (1, 50, 1, 20, "light"),
        (2, 50, 0, 10, "heavy"),
        (0, 10, 2, 20, "heavy"),
    ],
)
def test_defect_registered_boundaries(
    q_num: int, q_den: int, c_num: int, c_den: int, expected: str
) -> None:
    assert defect_band(q_num, q_den, c_num, c_den) == expected


def test_band_classification_uses_exact_counts_not_descriptive_decimals() -> None:
    hardness = _scientific_sources()["hardness"]
    assert classify_registered_bands(hardness) == {
        "host_fill": "low",
        "logical_degree": "sparse",
        "residual_lcc": "open",
        "defects": "none",
    }


@pytest.mark.parametrize(
    ("args", "error"),
    [
        ((True, 10), "integer"),
        ((1, 0), "positive"),
        ((-1, 10), "non-negative"),
        ((11, 10), "must not exceed"),
    ],
)
def test_fraction_band_inputs_are_strict(args: tuple[object, object], error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        host_fill_band(*args)


def test_recompute_cell_facts_derives_host_lcc_and_rejects_untrusted_shape() -> None:
    facts = _facts()
    document = facts.to_dict()

    assert document["host_largest_component_numerator"] == 200
    assert document["host_largest_component_denominator"] == 200
    assert document["host_largest_component_fraction"] == 1.0
    assert document["ground_state_eligible"] is True
    assert document["ground_certificate_status"] == "exact_enumeration"
    assert validate_cell_facts(facts) is facts

    document["p_solve"] = 0.9
    with pytest.raises(TypeError, match="VerifiedCellFacts"):
        validate_cell_facts(document)


def test_verify_cell_facts_recomputes_consistent_looking_tampering() -> None:
    source = _scientific_sources()
    facts = recompute_cell_facts(**source)
    document = facts.to_dict()
    assert (
        verify_cell_facts(
            document,
            expected_cell_facts_sha256=facts.cell_facts_sha256,
            recomputed=facts,
        )
        is facts
    )

    document["host_largest_component_numerator"] = 199
    document["host_largest_component_fraction"] = 199 / 200
    with pytest.raises(ValueError, match="do not equal authenticated recomputed"):
        verify_cell_facts(
            document,
            expected_cell_facts_sha256=canonical_sha256(document),
            recomputed=facts,
        )


def test_cell_fact_recomputation_binds_verified_source_identities() -> None:
    source = _scientific_sources()
    artifact = build_realized_host_artifact(
        topology="chimera",
        size=5,
        qubit_fraction=0.01,
        coupler_fraction=0.0,
        seed_key=bytes([10]) * 32,
    )
    foreign_host = verify_host(
        artifact,
        seed_key=bytes([10]) * 32,
        expected_host_artifact_sha256=artifact["host_artifact_sha256"],
        expected_host_sha256=artifact["host_sha256"],
    )
    source["host"] = foreign_host
    with pytest.raises(ValueError, match="hardness sources"):
        recompute_cell_facts(**source)


def test_cell_profile_is_closed_data_driven_and_exact() -> None:
    profile = _profile()

    checked = validate_cell_profile(profile)
    assert checked == profile
    profile["panel"] = "mutated-after-validation"
    assert checked["panel"] == "capacity_transition"
    profile["panel"] = "capacity_transition"
    assert cell_matches(profile, _facts())

    malformed = copy.deepcopy(profile)
    malformed["ratio_predicates"][0]["minimum"]["denominator"] = True
    with pytest.raises((TypeError, ValueError), match="denominator"):
        validate_cell_profile(malformed)

    malformed = copy.deepcopy(profile)
    malformed["future_threshold"] = 0.7
    with pytest.raises(ValueError, match="unknown=.*future_threshold"):
        validate_cell_profile(malformed)

    malformed = copy.deepcopy(profile)
    malformed["ratio_predicates"][0]["minimum"] = None
    malformed["ratio_predicates"][0]["maximum"] = None
    with pytest.raises(ValueError, match="at least one bound"):
        validate_cell_profile(malformed)


def test_cell_predicates_use_cross_multiplication_at_inclusive_boundaries() -> None:
    profile = _profile()
    facts = _facts()
    fill = facts.to_dict()["hardness"]["starting_embedding"]
    boundary = {
        "numerator": fill["host_fill_numerator"],
        "denominator": fill["host_fill_denominator"],
        "inclusive": True,
    }
    profile["ratio_predicates"][0]["minimum"] = boundary
    assert cell_matches(profile, facts)

    below = copy.deepcopy(profile)
    below["ratio_predicates"][0]["minimum"]["numerator"] += 1
    assert not cell_matches(below, facts)

    exclusive = copy.deepcopy(profile)
    exclusive["ratio_predicates"][0]["minimum"]["inclusive"] = False
    assert not cell_matches(exclusive, facts)


def test_profile_id_allowed_values_and_band_failures_do_not_match() -> None:
    profile = _profile()
    wrong_profile_facts = _facts("logical_stress_v1")
    assert not cell_matches(profile, wrong_profile_facts)

    wrong_topology = copy.deepcopy(profile)
    wrong_topology["value_predicates"][0]["allowed_values"] = ["pegasus"]
    assert not cell_matches(wrong_topology, _facts())

    wrong_band = copy.deepcopy(profile)
    wrong_band["band_predicates"][0]["allowed_bands"] = ["extreme"]
    assert not cell_matches(wrong_band, _facts())


def test_explicit_problem_sizes_are_data_driven_allowed_values() -> None:
    profile = _profile()
    profile["value_predicates"].append(
        {"field": "hardness.logical.variables", "allowed_values": [3, 4, 6, 8, 10]}
    )
    assert cell_matches(profile, _facts())

    profile["value_predicates"][-1]["allowed_values"] = [4, 6, 8, 10, 12]
    assert not cell_matches(profile, _facts())

    profile["value_predicates"][-1]["allowed_values"] = [True]
    with pytest.raises(TypeError, match="exact type int"):
        validate_cell_profile(profile)


def test_stored_cell_fact_tampering_cannot_be_resealed_by_the_caller() -> None:
    facts = _facts()
    document = facts.to_dict()
    document["requested_qubit_fraction"] = 0.25
    with pytest.raises(ValueError, match="independent binding"):
        verify_cell_facts(
            document,
            expected_cell_facts_sha256=facts.cell_facts_sha256,
            recomputed=facts,
        )


def _quota_entry(source: dict[str, object], suffix: str = "a"):
    return build_quota_entry(
        facts=recompute_cell_facts(**source),
        profile_set=source["profile_set"],
        expected_problem_iso_sha256=suffix * 64,
        expected_state_sha256=("b" if suffix != "b" else "c") * 64,
    )


def test_quota_helpers_enforce_exact_counts_predicates_and_distinct_unions() -> None:
    source = _scientific_sources()
    profile_set = source["profile_set"]
    cell_id = "hard_dev/chimera-5/capacity_transition"
    entry = _quota_entry(source)
    assert validate_quota_entry(entry) is entry
    assert entry.to_dict()["problem_group_id"] == (
        f"problem-sha256:{source['problem'].problem_sha256}"
    )

    assert expected_quota_total(profile_set) == 1_520
    deficits = quota_deficits((entry,), profile_set)
    assert deficits[cell_id] == 39
    assert sum(deficits.values()) == 1_519
    with pytest.raises(ValueError, match="quota mismatch"):
        validate_exact_quotas((entry,), profile_set)

    with pytest.raises(ValueError, match="problem_group_id occurs in multiple"):
        quota_deficits((entry, entry), profile_set)


def test_quota_entry_closed_schema_rejects_labels_solver_output_and_unknown_cell() -> None:
    source = _scientific_sources()
    profile_set = source["profile_set"]
    entry = _quota_entry(source, "d")
    stored = entry.to_dict()
    stored["label"] = 1
    with pytest.raises(TypeError, match="VerifiedQuotaEntry"):
        quota_deficits((stored,), profile_set)

    with pytest.raises(TypeError, match="immutable tuple"):
        quota_deficits([entry], profile_set)

    assert (
        verify_quota_entry(
            entry.to_dict(),
            expected_quota_entry_sha256=entry.quota_entry_sha256,
            recomputed=entry,
        )
        is entry
    )
    stored = entry.to_dict()
    stored["state_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="independent binding"):
        verify_quota_entry(
            stored,
            expected_quota_entry_sha256=entry.quota_entry_sha256,
            recomputed=entry,
        )


def test_profile_rejects_unknown_predicate_fields_bands_and_duplicate_rules() -> None:
    profile = _profile()
    profile["ratio_predicates"][0]["field"] = "label.p_solve"
    with pytest.raises(ValueError, match="unregistered ratio field"):
        validate_cell_profile(profile)

    profile = _profile()
    profile["band_predicates"][0]["allowed_bands"] = ["hard"]
    with pytest.raises(ValueError, match="unregistered band"):
        validate_cell_profile(profile)

    profile = _profile()
    profile["value_predicates"].append(copy.deepcopy(profile["value_predicates"][0]))
    with pytest.raises(ValueError, match="duplicate value predicate"):
        validate_cell_profile(profile)

    profile = _profile()
    profile["value_predicates"][0]["allowed_values"] = [f"chimera{chr(0xD800)}"]
    with pytest.raises(ValueError, match="Unicode surrogate"):
        validate_cell_profile(profile)


def _registered_profile_document() -> dict[str, object]:
    cells: list[dict[str, object]] = []
    for requirement in registered_cell_requirements():
        profile = _profile(
            cell_id=requirement.cell_id,
            profile_id=f"{requirement.panel}_v1",
            quota=requirement.quota,
        )
        profile["partition"] = requirement.partition
        profile["panel"] = requirement.panel
        cells.append(profile)
    return {
        "schema": PROFILE_SET_SCHEMA,
        "schema_version": 1,
        "release_id": HARD_OOD_RELEASE_ID,
        "generator_source_sha256": "a" * 64,
        "maximum_ordinal": 10_000,
        "cells": cells,
    }


def test_registered_layout_has_every_exact_release_cell_and_quota() -> None:
    requirements = registered_cell_requirements()
    assert len(requirements) == 37
    assert len({item.cell_id for item in requirements}) == 37
    totals: dict[str, int] = {}
    for item in requirements:
        totals[item.partition] = totals.get(item.partition, 0) + item.quota
    assert totals == {
        "hard_dev": 360,
        "mechanism_dev": 350,
        "mechanism_locked": 350,
        "composed_locked": 100,
        "locked_ood": 360,
    }


def test_profile_set_requires_independent_digest_and_detaches_caller_data() -> None:
    document = _registered_profile_document()
    digest = canonical_sha256(document)
    verified = verify_registered_profile_set(document, expected_profile_set_sha256=digest)
    assert validate_registered_profile_set(verified) is verified
    assert verified.profile_set_sha256 == digest

    document["cells"][0]["quota"] = 999
    assert verified.to_dict()["cells"][0]["quota"] == 40

    forged = _registered_profile_document()
    forged["cells"][0]["ratio_predicates"][0]["minimum"]["numerator"] = 2
    with pytest.raises(ValueError, match="independent binding"):
        verify_registered_profile_set(forged, expected_profile_set_sha256=digest)


def test_profile_set_rejects_empty_partial_unknown_and_reregistered_cells() -> None:
    document = _registered_profile_document()

    empty = copy.deepcopy(document)
    empty["cells"] = []
    with pytest.raises(ValueError, match="must not be empty"):
        verify_registered_profile_set(empty, expected_profile_set_sha256=canonical_sha256(empty))

    partial = copy.deepcopy(document)
    partial["cells"].pop()
    with pytest.raises(ValueError, match="cell registry differs"):
        verify_registered_profile_set(
            partial, expected_profile_set_sha256=canonical_sha256(partial)
        )

    unknown = copy.deepcopy(document)
    unknown["cells"][0]["cell_id"] = "locked_ood/invented/easy"
    with pytest.raises(ValueError, match="cell registry differs"):
        verify_registered_profile_set(
            unknown, expected_profile_set_sha256=canonical_sha256(unknown)
        )

    reregistered = copy.deepcopy(document)
    reregistered["cells"][0]["quota"] = 39
    with pytest.raises(ValueError, match="registration differs"):
        verify_registered_profile_set(
            reregistered,
            expected_profile_set_sha256=canonical_sha256(reregistered),
        )


def test_profile_set_rejects_unregistered_ordinal_cap_and_empty_predicates() -> None:
    document = _registered_profile_document()
    document["maximum_ordinal"] = 10_001
    with pytest.raises(ValueError, match="maximum_ordinal"):
        verify_registered_profile_set(
            document, expected_profile_set_sha256=canonical_sha256(document)
        )

    document = _registered_profile_document()
    cell = document["cells"][0]
    for field in (
        "ratio_predicates",
        "integer_predicates",
        "value_predicates",
        "band_predicates",
    ):
        cell[field] = []
    with pytest.raises(ValueError, match="no acceptance predicate"):
        verify_registered_profile_set(
            document, expected_profile_set_sha256=canonical_sha256(document)
        )


def test_plan_row_is_profile_bound_content_addressed_and_immutable() -> None:
    source = _scientific_sources()
    plan_row = source["plan_row"]
    assert (
        validate_plan_row(
            plan_row,
            expected_plan_row_sha256=plan_row.plan_row_sha256,
        )
        is plan_row
    )
    original = plan_row.to_dict()

    tampered = copy.deepcopy(original)
    tampered["ordinal"] = 1
    with pytest.raises(ValueError, match="problem_seed_request"):
        _verify_plan_document(
            tampered,
            expected_plan_row_sha256=original["plan_row_sha256"],
            profile_set=source["profile_set"],
        )
    replacement_request_document = copy.deepcopy(tampered["problem_seed_request"])
    replacement_request_document["replicate"] = 1
    replacement_request = SeedRequest.from_dict(replacement_request_document)
    tampered["problem_seed_request"] = replacement_request.to_dict()
    tampered["problem_seed_key_sha256"] = replacement_request.seed_key_hex
    tampered["plan_row_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "plan_row_sha256"}
    )
    verified = _verify_plan_document(
        tampered,
        expected_plan_row_sha256=tampered["plan_row_sha256"],
        profile_set=source["profile_set"],
        registered_request=replacement_request,
    )
    tampered["logical_source"] = "after-the-fact"
    assert verified.to_dict()["logical_source"] == "ink_drop"


def test_plan_row_whole_rebase_cannot_replace_external_commitment() -> None:
    source = _scientific_sources()
    honest = source["plan_row"]
    honest_digest = honest.plan_row_sha256
    attacker_document = honest.to_dict()
    attacker_document["ordinal"] = 1
    request_document = copy.deepcopy(attacker_document["problem_seed_request"])
    request_document["replicate"] = 1
    attacker_request = SeedRequest.from_dict(request_document)
    attacker_document["problem_seed_request"] = attacker_request.to_dict()
    attacker_document["problem_seed_key_sha256"] = attacker_request.seed_key_hex
    attacker_document["plan_row_sha256"] = canonical_sha256(
        {key: item for key, item in attacker_document.items() if key != "plan_row_sha256"}
    )
    attacker = _verify_plan_document(
        attacker_document,
        expected_plan_row_sha256=attacker_document["plan_row_sha256"],
        profile_set=source["profile_set"],
        registered_request=attacker_request,
    )

    object.__setattr__(honest, "_serialized", attacker._serialized)
    object.__setattr__(honest, "plan_row_sha256", attacker.plan_row_sha256)

    with pytest.raises(ValueError, match="content digest"):
        validate_plan_row(
            honest,
            expected_plan_row_sha256=honest_digest,
        )


def test_plan_row_rejects_wrong_registered_host_and_nonfinite_prefix() -> None:
    source = _scientific_sources()
    original = source["plan_row"].to_dict()
    for field, value, error in (
        ("topology", "pegasus", "topology"),
        ("size", 6, "size"),
        ("ordinal", 10_000, "finite prefix"),
    ):
        document = copy.deepcopy(original)
        document[field] = value
        document["plan_row_sha256"] = canonical_sha256(
            {key: item for key, item in document.items() if key != "plan_row_sha256"}
        )
        with pytest.raises(ValueError, match=error):
            _verify_plan_document(
                document,
                expected_plan_row_sha256=document["plan_row_sha256"],
                profile_set=source["profile_set"],
            )


def test_plan_row_requires_the_independent_seed_registry_root() -> None:
    source = _scientific_sources()
    document = source["plan_row"].to_dict()

    with pytest.raises(ValueError, match="external terminal-root commitment"):
        _verify_plan_document(
            document,
            expected_plan_row_sha256=document["plan_row_sha256"],
            profile_set=source["profile_set"],
            expected_seed_root_sha256="0" * 64,
        )


@pytest.mark.parametrize("plan_request_stage", ["foundation", "terminal"])
def test_plan_row_resolves_through_honest_multistage_seed_registry(
    tmp_path: Path,
    plan_request_stage: str,
) -> None:
    source = _scientific_sources()
    document = source["plan_row"].to_dict()
    request = SeedRequest.from_dict(document["problem_seed_request"])
    parent_request_document = request.to_dict()
    parent_request_document["replicate"] = 1
    other_request = SeedRequest.from_dict(parent_request_document)
    foundation_request = request if plan_request_stage == "foundation" else other_request
    terminal_request = other_request if plan_request_stage == "foundation" else request

    parent_directory = tmp_path.resolve() / "parent"
    parent_root = write_seed_registry_shards(
        [SeedRegistration(foundation_request, seed32_required=False)],
        directory=parent_directory,
        release_id=HARD_OOD_RELEASE_ID,
        stage_id="structure-foundation",
    )
    parent = verify_seed_registry_shards(
        parent_directory,
        expected_root_sha256=parent_root.sha256,
    )
    child_directory = tmp_path.resolve() / "child"
    child_root = write_seed_registry_shards(
        [SeedRegistration(terminal_request, seed32_required=False)],
        directory=child_directory,
        release_id=HARD_OOD_RELEASE_ID,
        stage_id="profile-plan",
        parent=parent,
    )
    child = verify_seed_registry_shards(
        child_directory,
        expected_root_sha256=child_root.sha256,
        parent=parent,
    )
    resolver = VerifiedSeedResolver(
        child,
        expected_terminal_root_sha256=child_root.sha256,
    )

    verified = verify_plan_row(
        document,
        expected_plan_row_sha256=document["plan_row_sha256"],
        profile_set=source["profile_set"],
        seed_resolver=resolver,
        expected_seed_registry_terminal_root_sha256=child_root.sha256,
    )

    assert verified.to_dict()["problem_seed_request"] == request.to_dict()


def test_plan_row_rejects_unregistered_or_seed32_problem_request() -> None:
    source = _scientific_sources()
    document = source["plan_row"].to_dict()
    registered_request_document = copy.deepcopy(document["problem_seed_request"])
    registered_request_document["replicate"] = 1
    other_request = SeedRequest.from_dict(registered_request_document)

    with pytest.raises(KeyError, match="not registered"):
        _verify_plan_document(
            document,
            expected_plan_row_sha256=document["plan_row_sha256"],
            profile_set=source["profile_set"],
            registered_request=other_request,
        )
    with pytest.raises(ValueError, match="native PCG"):
        _verify_plan_document(
            document,
            expected_plan_row_sha256=document["plan_row_sha256"],
            profile_set=source["profile_set"],
            seed32_required=True,
        )


@pytest.mark.parametrize(("field", "value"), [("state_slot", 4), ("focus_quartile", True)])
def test_plan_row_rejects_unregistered_structural_slots(field: str, value: object) -> None:
    source = _scientific_sources()
    document = source["plan_row"].to_dict()
    document[field] = value
    document["plan_row_sha256"] = canonical_sha256(
        {key: item for key, item in document.items() if key != "plan_row_sha256"}
    )

    with pytest.raises((TypeError, ValueError), match=field):
        _verify_plan_document(
            document,
            expected_plan_row_sha256=document["plan_row_sha256"],
            profile_set=source["profile_set"],
        )


def test_ground_eligibility_is_replayed_not_accepted_from_a_boolean() -> None:
    source = _scientific_sources()
    for field in (
        "ground_certificate",
        "ground_proof",
        "expected_ground_certificate_sha256",
        "expected_ground_proof_artifact_sha256",
    ):
        source[field] = None
    facts = recompute_cell_facts(**source)
    assert facts.to_dict()["ground_state_eligible"] is False
    profile = source["profile_set"].cell(source["plan_row"].to_dict()["cell_id"])
    assert not cell_matches(profile, facts)

    source = _scientific_sources()
    source["ground_certificate"] = copy.deepcopy(source["ground_certificate"])
    source["ground_certificate"]["energy"] += 1.0
    with pytest.raises(ValueError):
        recompute_cell_facts(**source)
