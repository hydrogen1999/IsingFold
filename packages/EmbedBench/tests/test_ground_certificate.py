from __future__ import annotations

import copy
import itertools
import math
import random
from dataclasses import FrozenInstanceError, replace

import pytest
from embedbench.embedding import LogicalProblem
from embedbench.ground_certificate import (
    CERTIFICATE_SCHEMA,
    CERTIFICATE_SCHEMA_VERSION,
    CERTIFIED_OPTIMAL_PROOF_SCHEMA,
    CERTIFIED_OPTIMAL_PROOF_SCHEMA_VERSION,
    CHECKER_ACCEPTANCE_SCHEMA,
    CHECKER_ACCEPTANCE_SCHEMA_VERSION,
    EXHAUSTIVE_PROOF_SCHEMA,
    EXHAUSTIVE_PROOF_SCHEMA_VERSION,
    PLANTED_PROOF_SCHEMA,
    PLANTED_PROOF_SCHEMA_VERSION,
    SPIN_ASSIGNMENT_SCHEMA,
    SPIN_ASSIGNMENT_SCHEMA_VERSION,
    UNCERTIFIED_REFERENCE_PROOF_SCHEMA,
    UNCERTIFIED_REFERENCE_PROOF_SCHEMA_VERSION,
    CheckerRun,
    CheckerTrust,
    Dyadic,
    IsingProblem,
    PlantedTerm,
    SolverRun,
    binary64_to_dyadic,
    build_certified_optimal_certificate,
    build_exact_enumeration_certificate,
    build_planted_certificate,
    build_uncertified_reference_certificate,
    exact_energy,
    verify_ground_state_certificate,
)
from embedbench.hard_ood_schema import canonical_sha256


def test_ground_certificate_schema_discriminators_are_pinned() -> None:
    assert (CERTIFICATE_SCHEMA, CERTIFICATE_SCHEMA_VERSION) == (
        "embedbench.ground-state-certificate",
        1,
    )
    assert (PLANTED_PROOF_SCHEMA, PLANTED_PROOF_SCHEMA_VERSION) == (
        "embedbench.ground-state-planted-proof",
        1,
    )
    assert (EXHAUSTIVE_PROOF_SCHEMA, EXHAUSTIVE_PROOF_SCHEMA_VERSION) == (
        "embedbench.ground-state-exhaustive-proof",
        1,
    )
    assert (CERTIFIED_OPTIMAL_PROOF_SCHEMA, CERTIFIED_OPTIMAL_PROOF_SCHEMA_VERSION) == (
        "embedbench.ground-state-certified-optimal-proof",
        1,
    )
    assert (
        UNCERTIFIED_REFERENCE_PROOF_SCHEMA,
        UNCERTIFIED_REFERENCE_PROOF_SCHEMA_VERSION,
    ) == ("embedbench.ground-state-uncertified-reference-proof", 1)
    assert (CHECKER_ACCEPTANCE_SCHEMA, CHECKER_ACCEPTANCE_SCHEMA_VERSION) == (
        "embedbench.ground-state-checker-acceptance",
        1,
    )
    assert (SPIN_ASSIGNMENT_SCHEMA, SPIN_ASSIGNMENT_SCHEMA_VERSION) == (
        "embedbench.spin-assignment",
        1,
    )


def _triangle_problem() -> IsingProblem:
    return IsingProblem(
        variables=(0, 1, 2),
        linear=((0, 0.0), (1, 0.0), (2, 0.0)),
        quadratic=((0, 1, -1.0), (0, 2, 1.0), (1, 2, -1.0)),
    )


def _large_mixed_problem() -> IsingProblem:
    return IsingProblem(
        variables=tuple(range(25)),
        linear=tuple((variable, -2.0) for variable in range(25)),
        quadratic=((0, 1, 1.0), (1, 2, -0.5)),
    )


def _large_all_positive_assignment() -> tuple[tuple[int, int], ...]:
    return tuple((variable, 1) for variable in range(25))


def _checker_run(*, exit_code: int = 0) -> CheckerRun:
    return CheckerRun(
        source_sha256="1" * 64,
        environment_sha256="2" * 64,
        command=("python3", "scripts/verify_ground_state.py", "proof.json"),
        exit_code=exit_code,
        log_sha256="3" * 64,
    )


def _exact_solver(state_count: int) -> SolverRun:
    return SolverRun(
        name="embedbench-gray-code-enumerator",
        version="1",
        command=("python3", "scripts/certify_hard_ood_ground_states.py", "problem.json"),
        seed=None,
        deterministic_work_limit=state_count,
        safety_timeout_seconds=None,
    )


def _branch_and_bound_solver(*, status_seed: int = 17) -> SolverRun:
    return SolverRun(
        name="registered-exact-bnb",
        version="4.2.0",
        command=("exact-bnb", "--proof", "proof.bin", "problem.json"),
        seed=status_seed,
        deterministic_work_limit=100_000_000,
        safety_timeout_seconds=None,
    )


def _checker_trust(*, source_sha256: str = "1" * 64) -> CheckerTrust:
    return CheckerTrust(
        proof_schema="registered-bnb-tree-v1",
        source_sha256=source_sha256,
        environment_sha256="2" * 64,
        command=("python3", "scripts/verify_ground_state.py", "proof.json"),
    )


def _triangle_planted_term() -> PlantedTerm:
    return PlantedTerm(
        term_index=0,
        linear=(),
        quadratic=(
            (0, 1, Dyadic(-1, 0)),
            (0, 2, Dyadic(1, 0)),
            (1, 2, Dyadic(-1, 0)),
        ),
        lower_bound=Dyadic(-1, 0),
    )


def _resign_certificate(document: dict[str, object]) -> str:
    document["certificate_sha256"] = canonical_sha256(
        {key: value for key, value in document.items() if key != "certificate_sha256"}
    )
    return str(document["certificate_sha256"])


def _bind_mutated_proof(
    certificate: dict[str, object], proof: dict[str, object]
) -> tuple[str, str]:
    proof_digest = canonical_sha256(proof)
    certificate["proof_artifact_sha256"] = proof_digest
    return _resign_certificate(certificate), proof_digest


def _resign_checker_acceptance(document: dict[str, object]) -> str:
    document["checker_acceptance_sha256"] = canonical_sha256(
        {key: value for key, value in document.items() if key != "checker_acceptance_sha256"}
    )
    return str(document["checker_acceptance_sha256"])


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, Dyadic(0, 0)),
        (-0.0, Dyadic(0, 0)),
        (0.5, Dyadic(1, -1)),
        (-12.5, Dyadic(-25, -1)),
        (0.1, Dyadic(3602879701896397, -55)),
        (float.fromhex("0x0.0000000000001p-1022"), Dyadic(1, -1074)),
    ],
)
def test_binary64_is_lifted_to_its_exact_canonical_dyadic(
    value: float,
    expected: Dyadic,
) -> None:
    assert binary64_to_dyadic(value) == expected


@pytest.mark.parametrize("value", [True, 1, math.inf, -math.inf, math.nan])
def test_binary64_conversion_rejects_non_binary64_or_nonfinite_input(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        binary64_to_dyadic(value)


def test_problem_identity_and_energy_use_exact_dyadic_arithmetic() -> None:
    problem = IsingProblem(
        variables=(0, 1),
        linear=((0, 0.1), (1, -0.2)),
        quadratic=((0, 1, 0.3),),
    )

    assert (
        problem.problem_sha256 == "67fc3c8caa1419421f9da70da4f78a4ea5926550cf5d057f7a17102c892d48b5"
    )
    assert exact_energy(problem, ((0, 1), (1, -1))) == Dyadic(1, -55)


def test_problem_snapshot_adapts_the_existing_logical_problem_convention() -> None:
    existing = LogicalProblem.from_dicts(
        {2: 0.0, 0: 0.25, 1: -0.5},
        {(2, 0): -1.0, (1, 2): 0.0},
    )

    problem = IsingProblem.from_logical_problem(existing)
    existing.h[0] = 999.0

    assert problem.variables == (0, 1, 2)
    assert problem.linear == ((0, 0.25), (1, -0.5), (2, 0.0))
    assert problem.quadratic == ((0, 2, -1.0),)
    assert exact_energy(problem, ((0, 1), (1, -1), (2, 1))) == Dyadic(-1, -2)


def test_energy_accumulation_uses_unbounded_integer_units() -> None:
    largest = float.fromhex("0x1.fffffffffffffp+1023")
    smallest = float.fromhex("0x0.0000000000001p-1022")
    problem = IsingProblem(
        variables=(0, 1),
        linear=((0, largest), (1, -largest)),
        quadratic=((0, 1, smallest),),
    )

    energy = exact_energy(problem, ((0, 1), (1, 1)))

    assert energy == Dyadic(1, -1074)


def test_large_exact_integer_with_negative_scale_has_a_finite_descriptive_float() -> None:
    value = Dyadic((1 << 2000) + 1, -2000)

    assert value.to_float() == 1.0


@pytest.mark.parametrize(
    "problem",
    [
        lambda: IsingProblem(variables=(0, 0), linear=((0, 0.0),), quadratic=()),
        lambda: IsingProblem(variables=(1, 0), linear=((0, 0.0), (1, 0.0)), quadratic=()),
        lambda: IsingProblem(variables=(0,), linear=((0, 0.0), (0, 1.0)), quadratic=()),
        lambda: IsingProblem(variables=(0,), linear=((0, True),), quadratic=()),
        lambda: IsingProblem(variables=(0,), linear=((0, math.nan),), quadratic=()),
        lambda: IsingProblem(
            variables=(0, 1),
            linear=((0, 0.0), (1, 0.0)),
            quadratic=((1, 0, 1.0),),
        ),
        lambda: IsingProblem(
            variables=(0, 1),
            linear=((0, 0.0), (1, 0.0)),
            quadratic=((0, 1, 0.0),),
        ),
    ],
)
def test_problem_schema_rejects_noncanonical_or_ambiguous_payloads(problem) -> None:
    with pytest.raises((TypeError, ValueError)):
        problem()


def test_assignment_is_closed_complete_and_bool_safe() -> None:
    problem = _triangle_problem()

    with pytest.raises(ValueError, match="exactly once"):
        exact_energy(problem, ((0, 1), (0, -1), (2, 1)))
    with pytest.raises(TypeError, match="spin"):
        exact_energy(problem, ((0, True), (1, -1), (2, 1)))
    with pytest.raises(ValueError, match="sorted"):
        exact_energy(problem, ((1, 1), (0, -1), (2, 1)))


def test_planted_proof_checks_every_term_and_round_trips_as_quality_eligible() -> None:
    problem = _triangle_problem()
    bundle = build_planted_certificate(
        problem,
        assignment=((0, 1), (1, 1), (2, 1)),
        terms=(_triangle_planted_term(),),
        method="frustrated_loop_planting",
        checker=_checker_run(),
    )

    verified = verify_ground_state_certificate(
        problem,
        bundle.certificate.to_dict(),
        bundle.proof.to_dict(),
        expected_certificate_sha256=bundle.certificate.digest,
        expected_proof_artifact_sha256=bundle.proof.digest,
    )

    assert verified.status == "planted_proof"
    assert verified.energy == Dyadic(-1, 0)
    assert verified.lower_bound == verified.upper_bound == verified.energy
    assert verified.quality_eligible is True
    assert bundle.proof.digest == "a54dd7943ac02577b07980b3eb4cf41c2b0d201afa10338cf8577baaff23e124"
    assert (
        bundle.certificate.digest
        == "aa55f4cbcd21f9482a9849c414c336a1687c22d61f0d346a7ff3b266200ec572"
    )


@pytest.mark.parametrize("forgery", ["term_bound", "assignment_digest", "decomposition"])
def test_planted_proof_rejects_deep_forgery_even_when_outer_hashes_are_resigned(
    forgery: str,
) -> None:
    problem = _triangle_problem()
    bundle = build_planted_certificate(
        problem,
        assignment=((0, 1), (1, 1), (2, 1)),
        terms=(_triangle_planted_term(),),
        method="frustrated_loop_planting",
        checker=_checker_run(),
    )
    certificate = copy.deepcopy(bundle.certificate.to_dict())
    proof = copy.deepcopy(bundle.proof.to_dict())
    if forgery == "term_bound":
        proof["terms"][0]["lower_bound_integer"] = -3
    elif forgery == "assignment_digest":
        proof["spin_assignment_sha256"] = "4" * 64
        certificate["spin_assignment_sha256"] = "4" * 64
    else:
        proof["terms"][0]["quadratic"][0][2]["integer"] = -3
    certificate_digest, proof_digest = _bind_mutated_proof(certificate, proof)

    with pytest.raises(ValueError):
        verify_ground_state_certificate(
            problem,
            certificate,
            proof,
            expected_certificate_sha256=certificate_digest,
            expected_proof_artifact_sha256=proof_digest,
        )


def test_planted_proof_rejects_an_assignment_that_does_not_attain_each_term_bound() -> None:
    with pytest.raises(ValueError, match="does not attain"):
        build_planted_certificate(
            _triangle_problem(),
            assignment=((0, 1), (1, -1), (2, 1)),
            terms=(_triangle_planted_term(),),
            method="frustrated_loop_planting",
            checker=_checker_run(),
        )


def test_planted_schemas_reject_unknown_fields_and_a_failed_checker() -> None:
    problem = _triangle_problem()
    with pytest.raises(ValueError, match="checker"):
        build_planted_certificate(
            problem,
            assignment=((0, 1), (1, 1), (2, 1)),
            terms=(_triangle_planted_term(),),
            method="frustrated_loop_planting",
            checker=_checker_run(exit_code=1),
        )

    bundle = build_planted_certificate(
        problem,
        assignment=((0, 1), (1, 1), (2, 1)),
        terms=(_triangle_planted_term(),),
        method="frustrated_loop_planting",
        checker=_checker_run(),
    )
    proof = bundle.proof.to_dict()
    proof["unregistered"] = True
    proof_digest = canonical_sha256(proof)
    certificate = bundle.certificate.to_dict()
    certificate["proof_artifact_sha256"] = proof_digest
    certificate_digest = _resign_certificate(certificate)
    with pytest.raises(ValueError, match="schema fields"):
        verify_ground_state_certificate(
            problem,
            certificate,
            proof,
            expected_certificate_sha256=certificate_digest,
            expected_proof_artifact_sha256=proof_digest,
        )


def test_exact_gray_code_certificate_recomputes_all_states_and_is_eligible() -> None:
    problem = IsingProblem(
        variables=(0, 1, 2),
        linear=((0, 0.5), (1, -0.25), (2, 0.125)),
        quadratic=((0, 1, 0.75), (1, 2, -0.5)),
    )
    bundle = build_exact_enumeration_certificate(
        problem,
        solver=_exact_solver(8),
        checker=_checker_run(),
    )

    verified = verify_ground_state_certificate(
        problem,
        bundle.certificate.to_dict(),
        bundle.proof.to_dict(),
        expected_certificate_sha256=bundle.certificate.digest,
        expected_proof_artifact_sha256=bundle.proof.digest,
    )
    brute_force = min(
        exact_energy(problem, tuple(zip(problem.variables, spins, strict=True)))
        for spins in itertools.product((-1, 1), repeat=3)
    )

    assert verified.status == "exact_enumeration"
    assert verified.energy == brute_force
    assert verified.quality_eligible is True
    assert bundle.proof.checked_state_count == 8
    assert bundle.proof.enumeration_order == "binary_reflected_gray_code_lsb_first_v1"
    assert bundle.proof.digest == "400816d0a5f4948fa7880044252d498d776ecc177cdd2dcd15375683ce67c7f2"
    assert (
        bundle.certificate.digest
        == "c0e84998efc0e0025ea783191a9f629d7e87f4dea379de1ccffdc7b1bd27b2e6"
    )


def test_exact_enumeration_uses_in_process_replay_without_checker_run_claims() -> None:
    problem = _triangle_problem()

    bundle = build_exact_enumeration_certificate(
        problem,
        solver=_exact_solver(8),
    )
    certificate = bundle.certificate.to_dict()

    assert certificate["checker_source_sha256"] is None
    assert certificate["checker_environment_sha256"] is None
    assert certificate["checker_command"] is None
    assert certificate["checker_exit_code"] is None
    assert certificate["checker_log_sha256"] is None
    verified = verify_ground_state_certificate(
        problem,
        certificate,
        bundle.proof.to_dict(),
        expected_certificate_sha256=bundle.certificate.digest,
        expected_proof_artifact_sha256=bundle.proof.digest,
    )
    assert verified.status == "exact_enumeration"
    assert verified.quality_eligible is True


def test_zero_optimum_has_exact_zero_gap_and_null_relative_gap() -> None:
    problem = IsingProblem(variables=(0,), linear=((0, 0.0),), quadratic=())
    bundle = build_exact_enumeration_certificate(
        problem,
        solver=_exact_solver(2),
        checker=_checker_run(),
    )

    certificate = bundle.certificate.to_dict()

    assert certificate["lower_bound_integer"] == 0
    assert certificate["upper_bound_integer"] == 0
    assert certificate["absolute_gap"] == 0.0
    assert certificate["relative_gap"] is None


@pytest.mark.parametrize("seed", range(20))
def test_gray_code_exact_minimum_matches_independent_cartesian_enumeration(seed: int) -> None:
    rng = random.Random(seed)
    n_vars = 1 + seed % 6
    problem = IsingProblem(
        variables=tuple(range(n_vars)),
        linear=tuple((variable, rng.uniform(-2.0, 2.0)) for variable in range(n_vars)),
        quadratic=tuple(
            (left, right, rng.choice((-1.0, 1.0)) * rng.uniform(0.01, 2.0))
            for left in range(n_vars)
            for right in range(left + 1, n_vars)
            if rng.random() < 0.55
        ),
    )
    state_count = 1 << n_vars
    bundle = build_exact_enumeration_certificate(
        problem,
        solver=_exact_solver(state_count),
        checker=_checker_run(),
    )
    independent = min(
        exact_energy(problem, tuple(zip(problem.variables, spins, strict=True)))
        for spins in itertools.product((-1, 1), repeat=n_vars)
    )

    assert bundle.certificate.energy == independent


@pytest.mark.parametrize(
    "field",
    [
        "checked_state_count",
        "minimum_energy_integer",
        "minimum_gray_step",
        "final_gray_word",
        "spin_assignment_sha256",
    ],
)
def test_exact_enumeration_rejects_a_resigned_forged_transcript(field: str) -> None:
    problem = _triangle_problem()
    bundle = build_exact_enumeration_certificate(
        problem,
        solver=_exact_solver(8),
        checker=_checker_run(),
    )
    certificate = copy.deepcopy(bundle.certificate.to_dict())
    proof = copy.deepcopy(bundle.proof.to_dict())
    if field == "spin_assignment_sha256":
        proof[field] = "5" * 64
        certificate[field] = "5" * 64
    else:
        proof[field] += 1
    certificate_digest, proof_digest = _bind_mutated_proof(certificate, proof)

    with pytest.raises(ValueError):
        verify_ground_state_certificate(
            problem,
            certificate,
            proof,
            expected_certificate_sha256=certificate_digest,
            expected_proof_artifact_sha256=proof_digest,
        )


def test_exact_enumeration_limit_and_work_count_are_fail_closed() -> None:
    too_large = IsingProblem(
        variables=tuple(range(25)),
        linear=tuple((variable, 0.0) for variable in range(25)),
        quadratic=(),
    )
    with pytest.raises(ValueError, match="24"):
        build_exact_enumeration_certificate(
            too_large,
            solver=_exact_solver(1 << 25),
            checker=_checker_run(),
        )

    with pytest.raises(ValueError, match="work limit"):
        build_exact_enumeration_certificate(
            _triangle_problem(),
            solver=_exact_solver(7),
            checker=_checker_run(),
        )


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("certificate", "energy_integer", True),
        ("certificate", "upper_bound", math.inf),
        ("proof", "n_vars", True),
        ("proof", "checked_state_count", 8.0),
    ],
)
def test_exact_certificate_rejects_bool_as_int_and_nonfinite_fields(
    target: str,
    field: str,
    value: object,
) -> None:
    problem = _triangle_problem()
    bundle = build_exact_enumeration_certificate(
        problem,
        solver=_exact_solver(8),
        checker=_checker_run(),
    )
    certificate = copy.deepcopy(bundle.certificate.to_dict())
    proof = copy.deepcopy(bundle.proof.to_dict())
    document = certificate if target == "certificate" else proof
    document[field] = value
    if target == "proof":
        certificate_digest, proof_digest = _bind_mutated_proof(certificate, proof)
    else:
        proof_digest = bundle.proof.digest
        try:
            certificate_digest = _resign_certificate(certificate)
        except ValueError:
            certificate_digest = bundle.certificate.digest
    with pytest.raises((TypeError, ValueError)):
        verify_ground_state_certificate(
            problem,
            certificate,
            proof,
            expected_certificate_sha256=certificate_digest,
            expected_proof_artifact_sha256=proof_digest,
        )


def test_checker_accepted_certified_optimal_proof_crosses_an_explicit_trust_boundary() -> None:
    problem = _large_mixed_problem()
    bundle = build_certified_optimal_certificate(
        problem,
        assignment=_large_all_positive_assignment(),
        lower_bound=Dyadic(-99, -1),
        upper_bound=Dyadic(-99, -1),
        solver_status="OPTIMAL",
        solver=_branch_and_bound_solver(),
        proof_schema="registered-bnb-tree-v1",
        proof_file_sha256="6" * 64,
        checker=_checker_run(),
    )
    assert bundle.checker_acceptance is not None

    verified = verify_ground_state_certificate(
        problem,
        bundle.certificate.to_dict(),
        bundle.proof.to_dict(),
        expected_certificate_sha256=bundle.certificate.digest,
        expected_proof_artifact_sha256=bundle.proof.digest,
        checker_acceptance=bundle.checker_acceptance.to_dict(),
        expected_checker_acceptance_sha256=bundle.checker_acceptance.digest,
        checker_trust=_checker_trust(),
    )

    assert verified.status == "certified_optimal"
    assert verified.quality_eligible is True
    assert (
        bundle.checker_acceptance.digest
        == "e75e435b1b2999b9f27e2cf3c9297df8b51d3ae6ba47d7547eb026911cc6a252"
    )


def test_small_nonplanted_problem_must_use_registered_gray_code_enumeration() -> None:
    with pytest.raises(ValueError, match="Gray-code"):
        build_certified_optimal_certificate(
            _triangle_problem(),
            assignment=((0, 1), (1, 1), (2, 1)),
            lower_bound=Dyadic(-1, 0),
            upper_bound=Dyadic(-1, 0),
            solver_status="OPTIMAL",
            solver=_branch_and_bound_solver(),
            proof_schema="registered-bnb-tree-v1",
            proof_file_sha256="6" * 64,
            checker=_checker_run(),
        )


@pytest.mark.parametrize("missing", ["acceptance", "expected_digest", "trust"])
def test_solver_optimal_status_is_never_trusted_without_independent_checker_binding(
    missing: str,
) -> None:
    problem = _large_mixed_problem()
    bundle = build_certified_optimal_certificate(
        problem,
        assignment=_large_all_positive_assignment(),
        lower_bound=Dyadic(-99, -1),
        upper_bound=Dyadic(-99, -1),
        solver_status="OPTIMAL",
        solver=_branch_and_bound_solver(),
        proof_schema="registered-bnb-tree-v1",
        proof_file_sha256="6" * 64,
        checker=_checker_run(),
    )
    assert bundle.checker_acceptance is not None
    kwargs = {
        "checker_acceptance": bundle.checker_acceptance.to_dict(),
        "expected_checker_acceptance_sha256": bundle.checker_acceptance.digest,
        "checker_trust": _checker_trust(),
    }
    if missing == "acceptance":
        kwargs["checker_acceptance"] = None
    elif missing == "expected_digest":
        kwargs["expected_checker_acceptance_sha256"] = None
    else:
        kwargs["checker_trust"] = None

    with pytest.raises(ValueError, match="checker"):
        verify_ground_state_certificate(
            problem,
            bundle.certificate.to_dict(),
            bundle.proof.to_dict(),
            expected_certificate_sha256=bundle.certificate.digest,
            expected_proof_artifact_sha256=bundle.proof.digest,
            **kwargs,
        )


def test_checker_acceptance_rejects_wrong_trusted_checker_and_bound_forgery() -> None:
    problem = _large_mixed_problem()
    bundle = build_certified_optimal_certificate(
        problem,
        assignment=_large_all_positive_assignment(),
        lower_bound=Dyadic(-99, -1),
        upper_bound=Dyadic(-99, -1),
        solver_status="OPTIMAL",
        solver=_branch_and_bound_solver(),
        proof_schema="registered-bnb-tree-v1",
        proof_file_sha256="6" * 64,
        checker=_checker_run(),
    )
    assert bundle.checker_acceptance is not None
    with pytest.raises(ValueError, match="checker"):
        verify_ground_state_certificate(
            problem,
            bundle.certificate.to_dict(),
            bundle.proof.to_dict(),
            expected_certificate_sha256=bundle.certificate.digest,
            expected_proof_artifact_sha256=bundle.proof.digest,
            checker_acceptance=bundle.checker_acceptance.to_dict(),
            expected_checker_acceptance_sha256=bundle.checker_acceptance.digest,
            checker_trust=_checker_trust(source_sha256="9" * 64),
        )

    forged = build_certified_optimal_certificate(
        problem,
        assignment=_large_all_positive_assignment(),
        lower_bound=Dyadic(-101, -1),
        upper_bound=Dyadic(-99, -1),
        solver_status="OPTIMAL",
        solver=_branch_and_bound_solver(),
        proof_schema="registered-bnb-tree-v1",
        proof_file_sha256="6" * 64,
        checker=_checker_run(),
    )
    with pytest.raises(ValueError, match="acceptance"):
        verify_ground_state_certificate(
            problem,
            forged.certificate.to_dict(),
            forged.proof.to_dict(),
            expected_certificate_sha256=forged.certificate.digest,
            expected_proof_artifact_sha256=forged.proof.digest,
            checker_acceptance=bundle.checker_acceptance.to_dict(),
            expected_checker_acceptance_sha256=bundle.checker_acceptance.digest,
            checker_trust=_checker_trust(),
        )


def test_checker_accepted_nonzero_gap_is_valid_metadata_but_not_quality_eligible() -> None:
    problem = _large_mixed_problem()
    bundle = build_certified_optimal_certificate(
        problem,
        assignment=_large_all_positive_assignment(),
        lower_bound=Dyadic(-101, -1),
        upper_bound=Dyadic(-99, -1),
        solver_status="WORK_LIMIT_REACHED",
        solver=_branch_and_bound_solver(),
        proof_schema="registered-bnb-tree-v1",
        proof_file_sha256="6" * 64,
        checker=_checker_run(),
    )
    assert bundle.checker_acceptance is not None

    verified = verify_ground_state_certificate(
        problem,
        bundle.certificate.to_dict(),
        bundle.proof.to_dict(),
        expected_certificate_sha256=bundle.certificate.digest,
        expected_proof_artifact_sha256=bundle.proof.digest,
        checker_acceptance=bundle.checker_acceptance.to_dict(),
        expected_checker_acceptance_sha256=bundle.checker_acceptance.digest,
        checker_trust=_checker_trust(),
    )

    assert verified.lower_bound == Dyadic(-101, -1)
    assert verified.upper_bound == Dyadic(-99, -1)
    assert verified.quality_eligible is False


def test_certified_optimal_verifier_enforces_registered_node_limit_after_ingestion() -> None:
    problem = _large_mixed_problem()
    bundle = build_certified_optimal_certificate(
        problem,
        assignment=_large_all_positive_assignment(),
        lower_bound=Dyadic(-99, -1),
        upper_bound=Dyadic(-99, -1),
        solver_status="OPTIMAL",
        solver=_branch_and_bound_solver(),
        proof_schema="registered-bnb-tree-v1",
        proof_file_sha256="6" * 64,
        checker=_checker_run(),
    )
    assert bundle.checker_acceptance is not None
    proof = copy.deepcopy(bundle.proof.to_dict())
    certificate = copy.deepcopy(bundle.certificate.to_dict())
    acceptance = copy.deepcopy(bundle.checker_acceptance.to_dict())
    proof["deterministic_work_limit"] = 100_000_001
    certificate["deterministic_work_limit"] = 100_000_001
    proof_digest = canonical_sha256(proof)
    certificate["proof_artifact_sha256"] = proof_digest
    acceptance["proof_artifact_sha256"] = proof_digest
    acceptance_digest = _resign_checker_acceptance(acceptance)
    certificate["checker_acceptance_sha256"] = acceptance_digest
    certificate_digest = _resign_certificate(certificate)

    with pytest.raises(ValueError, match="100,000,000"):
        verify_ground_state_certificate(
            problem,
            certificate,
            proof,
            expected_certificate_sha256=certificate_digest,
            expected_proof_artifact_sha256=proof_digest,
            checker_acceptance=acceptance,
            expected_checker_acceptance_sha256=acceptance_digest,
            checker_trust=_checker_trust(),
        )


def test_uncertified_reference_never_becomes_quality_eligible_even_at_the_true_optimum() -> None:
    problem = _triangle_problem()
    bundle = build_uncertified_reference_certificate(
        problem,
        assignment=((0, 1), (1, 1), (2, 1)),
        method="simulated_annealing_best_known",
        reason="heuristic upper bound only",
        solver=_branch_and_bound_solver(status_seed=23),
    )

    verified = verify_ground_state_certificate(
        problem,
        bundle.certificate.to_dict(),
        bundle.proof.to_dict(),
        expected_certificate_sha256=bundle.certificate.digest,
        expected_proof_artifact_sha256=bundle.proof.digest,
    )

    assert verified.energy == Dyadic(-1, 0)
    assert verified.lower_bound is None
    assert verified.quality_eligible is False


def test_uncertified_proof_cannot_be_relabelled_as_exact_by_resigning_outer_objects() -> None:
    problem = _triangle_problem()
    bundle = build_uncertified_reference_certificate(
        problem,
        assignment=((0, 1), (1, 1), (2, 1)),
        method="tabu_best_known",
        reason="no exact lower bound",
        solver=_branch_and_bound_solver(),
    )
    certificate = bundle.certificate.to_dict()
    certificate["status"] = "exact_enumeration"
    certificate["lower_bound"] = certificate["upper_bound"]
    certificate["lower_bound_integer"] = certificate["upper_bound_integer"]
    certificate["lower_bound_power_of_two"] = certificate["upper_bound_power_of_two"]
    certificate["absolute_gap"] = 0.0
    certificate["relative_gap"] = 0.0
    certificate["checker_source_sha256"] = "1" * 64
    certificate["checker_environment_sha256"] = "2" * 64
    certificate["checker_command"] = ["python3", "scripts/verify_ground_state.py", "proof.json"]
    certificate["checker_exit_code"] = 0
    certificate["checker_log_sha256"] = "3" * 64
    certificate_digest = _resign_certificate(certificate)

    with pytest.raises(ValueError):
        verify_ground_state_certificate(
            problem,
            certificate,
            bundle.proof.to_dict(),
            expected_certificate_sha256=certificate_digest,
            expected_proof_artifact_sha256=bundle.proof.digest,
        )


def test_missing_proof_and_wrong_schema_version_are_rejected() -> None:
    problem = _triangle_problem()
    bundle = build_uncertified_reference_certificate(
        problem,
        assignment=((0, 1), (1, 1), (2, 1)),
        method="best_known",
        reason="no proof",
        solver=_branch_and_bound_solver(),
    )
    with pytest.raises(TypeError, match="proof"):
        verify_ground_state_certificate(
            problem,
            bundle.certificate.to_dict(),
            None,
            expected_certificate_sha256=bundle.certificate.digest,
            expected_proof_artifact_sha256=bundle.proof.digest,
        )
    certificate = bundle.certificate.to_dict()
    certificate["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        verify_ground_state_certificate(
            problem,
            certificate,
            bundle.proof.to_dict(),
            expected_certificate_sha256=bundle.certificate.digest,
            expected_proof_artifact_sha256=bundle.proof.digest,
        )


def test_all_in_memory_artifact_contracts_are_frozen_and_validate_direct_construction() -> None:
    problem = _triangle_problem()
    exhaustive = build_exact_enumeration_certificate(
        problem,
        solver=_exact_solver(8),
        checker=_checker_run(),
    )
    planted = build_planted_certificate(
        problem,
        assignment=((0, 1), (1, 1), (2, 1)),
        terms=(_triangle_planted_term(),),
        method="frustrated_loop_planting",
        checker=_checker_run(),
    )
    certified = build_certified_optimal_certificate(
        _large_mixed_problem(),
        assignment=_large_all_positive_assignment(),
        lower_bound=Dyadic(-99, -1),
        upper_bound=Dyadic(-99, -1),
        solver_status="OPTIMAL",
        solver=_branch_and_bound_solver(),
        proof_schema="registered-bnb-tree-v1",
        proof_file_sha256="6" * 64,
        checker=_checker_run(),
    )
    uncertified = build_uncertified_reference_certificate(
        problem,
        assignment=((0, 1), (1, 1), (2, 1)),
        method="best_known",
        reason="not certified",
        solver=_branch_and_bound_solver(),
    )

    with pytest.raises(FrozenInstanceError):
        exhaustive.certificate.status = "uncertified_reference"
    with pytest.raises(ValueError):
        replace(exhaustive.proof, checked_state_count=7)
    with pytest.raises(ValueError):
        replace(planted.proof, spin_assignment_sha256="not-a-digest")
    with pytest.raises(ValueError):
        replace(certified.proof, solver_status="")
    with pytest.raises(ValueError):
        replace(uncertified.proof, reason="")


@pytest.mark.parametrize("power", [-1075, 1024, 1_000_000])
def test_proof_artifacts_reject_dyadic_scales_outside_finite_binary64_energy_domain(
    power: int,
) -> None:
    with pytest.raises(ValueError, match="binary64 energy domain"):
        PlantedTerm(
            term_index=0,
            linear=((0, Dyadic(1, power)),),
            quadratic=(),
            lower_bound=Dyadic(-1, 0),
        )
