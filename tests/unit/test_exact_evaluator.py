from __future__ import annotations

import itertools
import math
import random
from dataclasses import replace

import pytest

from isingfold_lac_b.evaluation import ExactEvaluator
from isingfold_lac_b.objective import HardwareGraph, IsingProblem, ObjectiveContext
from isingfold_lac_b.programming import Embedding, PhysicalProgram, program_embedding


def _single_chain_context() -> ObjectiveContext:
    return ObjectiveContext(
        problem=IsingProblem(linear=(-1.0,), quadratic=()),
        hardware=HardwareGraph(num_nodes=2, edges=((0, 1),)),
        beta_dev=1.0,
        j_max=1.0,
        f_domain=(0.0, 4.0),
        q_cap=2,
    )


def _direct_probabilities(
    context: ObjectiveContext, program: PhysicalProgram, strength: float
) -> tuple[float, float, float, float, float]:
    epsilon = context.scale_divisor(strength, chain_edge_count=program.total_chain_edges)
    weighted_states: list[tuple[tuple[int, ...], float]] = []
    for spins in itertools.product((-1, 1), repeat=program.num_active_spins):
        weight = math.exp(-context.beta_dev * program.energy(spins, strength=strength) / epsilon)
        weighted_states.append((spins, weight))

    partition = math.fsum(weight for _, weight in weighted_states)
    ground = min(
        context.problem.energy(logical)
        for logical in itertools.product((-1, 1), repeat=context.problem.num_variables)
    )
    p_consistent = (
        math.fsum(weight for spins, weight in weighted_states if program.is_chain_consistent(spins))
        / partition
    )
    p_solve = (
        math.fsum(
            weight
            for spins, weight in weighted_states
            if (logical := program.induced_logical_spins(spins)) is not None
            and context.problem.energy(logical) == pytest.approx(ground)
        )
        / partition
    )
    expected_broken = (
        math.fsum(program.broken_edge_count(spins) * weight for spins, weight in weighted_states)
        / partition
    )
    expected_e0 = (
        math.fsum(program.unpenalized_energy(spins) * weight for spins, weight in weighted_states)
        / partition
    )
    return math.log(partition), p_consistent, p_solve, expected_broken, expected_e0


def test_hand_enumeration_matches_partition_events_and_defect_coefficients() -> None:
    context = _single_chain_context()
    program = program_embedding(context, Embedding(chains=((0, 1),)))

    result = ExactEvaluator(max_active_spins=4).evaluate(context, program, strength=0.5)

    partition = math.exp(1.5) + 3.0 * math.exp(-0.5)
    assert result.log_partition == pytest.approx(math.log(partition))
    assert result.p_chain_consistent == pytest.approx((math.exp(1.5) + math.exp(-0.5)) / partition)
    assert result.p_solve == pytest.approx(math.exp(1.5) / partition)
    assert result.expected_broken_edges == pytest.approx(2.0 * math.exp(-0.5) / partition)
    assert result.expected_unpenalized_energy == pytest.approx(
        (-math.exp(1.5) + math.exp(-0.5)) / partition
    )
    assert result.logical_ground_energy == -1.0
    assert result.logical_ground_degeneracy == 1
    assert result.physical_state_count == 4
    assert result.evaluator_id == "exact-enumeration-defect-v1"
    assert result.objective_context_digest == context.digest
    assert result.physical_program_digest == program.digest
    assert result.verification_tolerance == 1e-11
    assert result.defect_log_coefficients == pytest.approx(
        (math.log(math.exp(1.0) + math.exp(-1.0)), math.log(2.0))
    )
    assert result.log_partition_from_defects == pytest.approx(result.log_partition)
    assert result.p_chain_consistent_from_defects == pytest.approx(result.p_chain_consistent)
    assert result.p_solve_from_faithful_identity == pytest.approx(result.p_solve)
    assert result.audit.connected_programmed_chains
    assert result.audit.hardware_compatible
    assert result.audit.faithful_programming
    assert result.audit.e0_independent_of_strength
    assert result.audit.defect_partition_identity_verified
    assert result.audit.defect_consistency_identity_verified
    assert result.audit.faithful_solve_identity_verified


@pytest.mark.parametrize("strength", [0.0, 0.4, 1.0, 2.7])
def test_direct_enumeration_matches_independent_calculation_across_scaling_knee(
    strength: float,
) -> None:
    context = ObjectiveContext(
        problem=IsingProblem(linear=(0.4, -0.7), quadratic=((0, 1, -1.3),)),
        hardware=HardwareGraph(num_nodes=3, edges=((0, 1), (0, 2), (1, 2))),
        beta_dev=1.7,
        j_max=1.0,
        f_domain=(0.0, 3.0),
        q_cap=3,
    )
    program = program_embedding(context, Embedding(chains=((0, 1), (2,))))

    result = ExactEvaluator().evaluate(context, program, strength=strength)
    expected = _direct_probabilities(context, program, strength)

    assert result.log_partition == pytest.approx(expected[0], abs=1e-12)
    assert result.p_chain_consistent == pytest.approx(expected[1], abs=1e-12)
    assert result.p_solve == pytest.approx(expected[2], abs=1e-12)
    assert result.expected_broken_edges == pytest.approx(expected[3], abs=1e-12)
    assert result.expected_unpenalized_energy == pytest.approx(expected[4], abs=1e-12)
    assert result.log_partition_from_defects == pytest.approx(expected[0], abs=1e-12)


def test_defect_reconstruction_matches_deterministically_generated_tiny_cases() -> None:
    generator = random.Random(20260904)
    hardware = HardwareGraph(
        num_nodes=4,
        edges=((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)),
    )
    evaluator = ExactEvaluator()

    for _ in range(12):
        context = ObjectiveContext(
            problem=IsingProblem(
                linear=(generator.uniform(-1.0, 1.0), generator.uniform(-1.0, 1.0)),
                quadratic=((0, 1, generator.uniform(-1.0, 1.0)),),
            ),
            hardware=hardware,
            beta_dev=generator.uniform(0.25, 2.0),
            j_max=1.0,
            f_domain=(0.0, 3.0),
            q_cap=4,
        )
        program = program_embedding(context, Embedding(chains=((0, 1), (2, 3))))
        strength = generator.uniform(0.0, 3.0)

        result = evaluator.evaluate(context, program, strength=strength)
        expected = _direct_probabilities(context, program, strength)

        assert result.log_partition_from_defects == pytest.approx(expected[0], abs=1e-11)
        assert result.p_chain_consistent_from_defects == pytest.approx(expected[1], abs=1e-11)
        assert result.p_solve_from_faithful_identity == pytest.approx(expected[2], abs=1e-11)


def test_all_singleton_program_is_consistent_and_strength_invariant() -> None:
    context = ObjectiveContext(
        problem=IsingProblem(linear=(0.2, -0.4), quadratic=((0, 1, 0.8),)),
        hardware=HardwareGraph(num_nodes=2, edges=((0, 1),)),
        beta_dev=2.0,
        j_max=1.0,
        f_domain=(0.0, 8.0),
        q_cap=2,
    )
    program = program_embedding(context, Embedding(chains=((0,), (1,))))
    evaluator = ExactEvaluator()

    at_zero = evaluator.evaluate(context, program, strength=0.0)
    at_eight = evaluator.evaluate(context, program, strength=8.0)

    assert at_zero.p_chain_consistent == 1.0
    assert at_eight.p_chain_consistent == 1.0
    assert at_zero.scale_divisor == at_eight.scale_divisor == 1.0
    assert at_zero.p_solve == pytest.approx(at_eight.p_solve)
    assert at_zero.log_partition == pytest.approx(at_eight.log_partition)
    assert len(at_zero.defect_log_coefficients) == 1


def test_unfaithful_program_uses_direct_probability_but_disables_faithful_identity() -> None:
    context = _single_chain_context()
    program = program_embedding(context, Embedding(chains=((0, 1),)))
    tampered = replace(program, linear=(1.5, -0.5))

    result = ExactEvaluator().evaluate(context, tampered, strength=0.75)

    assert not result.audit.faithful_programming
    assert result.p_solve_from_faithful_identity is None
    assert result.audit.faithful_solve_identity_verified is None
    assert result.audit.defect_partition_identity_verified
    assert result.audit.defect_consistency_identity_verified
    assert result.p_solve == pytest.approx(
        _direct_probabilities(context, tampered, 0.75)[2], abs=1e-12
    )


def test_disconnected_programmed_chain_is_rejected_before_using_defect_events() -> None:
    context = ObjectiveContext(
        problem=IsingProblem(linear=(0.0,), quadratic=()),
        hardware=HardwareGraph(num_nodes=3, edges=((0, 1), (1, 2))),
        beta_dev=1.0,
        j_max=1.0,
        f_domain=(0.0, 2.0),
        q_cap=3,
    )
    disconnected = PhysicalProgram(
        active_nodes=(0, 1, 2),
        chains=((0, 1, 2),),
        linear=(0.0, 0.0, 0.0),
        problem_quadratic=(),
        chain_edges_by_logical=(((0, 1),),),
    )

    with pytest.raises(ValueError, match="connected"):
        ExactEvaluator().evaluate(context, disconnected, strength=1.0)


def test_state_cap_invalid_strength_and_hardware_mismatch_are_rejected() -> None:
    context = _single_chain_context()
    program = program_embedding(context, Embedding(chains=((0, 1),)))

    with pytest.raises(ValueError, match="max_active_spins"):
        ExactEvaluator(max_active_spins=1).evaluate(context, program, strength=0.5)
    with pytest.raises(ValueError, match="strength"):
        ExactEvaluator().evaluate(context, program, strength=5.0)

    mismatched = replace(program, active_nodes=(0, 2))
    with pytest.raises(ValueError, match="hardware"):
        ExactEvaluator().evaluate(context, mismatched, strength=0.5)


def test_logsumexp_path_stays_finite_for_large_energy_exponents() -> None:
    context = ObjectiveContext(
        problem=IsingProblem(linear=(-10.0,), quadratic=()),
        hardware=HardwareGraph(num_nodes=1, edges=()),
        beta_dev=1000.0,
        j_max=1.0,
        f_domain=(0.0, 1.0),
        q_cap=1,
    )
    program = program_embedding(context, Embedding(chains=((0,),)))

    result = ExactEvaluator().evaluate(context, program, strength=0.0)

    assert math.isfinite(result.log_partition)
    assert result.log_partition == pytest.approx(10_000.0)
    assert result.p_solve == 1.0


def test_exact_success_does_not_use_relative_tolerance_at_large_energy_scale() -> None:
    context = ObjectiveContext(
        problem=IsingProblem(
            linear=(1e12, 1e12),
            quadratic=((0, 1, 1e12 - 0.5),),
        ),
        hardware=HardwareGraph(num_nodes=2, edges=((0, 1),)),
        beta_dev=1e-12,
        j_max=1.0,
        f_domain=(0.0, 1.0),
        q_cap=2,
    )
    program = program_embedding(context, Embedding(chains=((0,), (1,))))

    result = ExactEvaluator(verification_tolerance=1e-11).evaluate(context, program, strength=0.0)

    assert result.logical_ground_energy == -1e12 - 0.5
    assert result.logical_ground_degeneracy == 1
