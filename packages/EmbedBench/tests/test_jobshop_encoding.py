from __future__ import annotations

from collections.abc import Mapping
from itertools import pairwise

import numpy as np
import pytest
from embedbench.apps import Instance, jobshop

PRODUCTION_SHAPES = (
    (2, 1, 3),
    (2, 1, 4),
    (2, 1, 5),
    (2, 2, 3),
    (2, 1, 7),
    (2, 2, 4),
)


def _keyed(raw: Mapping[str, int]) -> dict[tuple[int, int], int]:
    return {
        tuple(int(part) for part in key.split("-")): value
        for key, value in raw.items()
    }


def _state_table(
    instance: Instance,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    derived = instance.derived
    encoding = derived["encoding"]
    variables = encoding["variable_order"]
    n_variables = len(variables)
    masks = np.arange(1 << n_variables, dtype=np.uint32)[:, None]
    bits = ((masks >> np.arange(n_variables, dtype=np.uint32)) & 1).astype(np.int64)

    qubo = np.full(len(bits), encoding["binary_offset_units"], dtype=np.int64)
    for node, units in encoding["qubo_linear_units"]:
        qubo += units * bits[:, node]
    for left, right, units in encoding["qubo_quadratic_units"]:
        qubo += units * bits[:, left] * bits[:, right]

    denominator = encoding["ising_coefficient_denominator"]
    spins = 2 * bits - 1
    ising_with_offset = np.full(len(bits), encoding["ising_offset_units"], dtype=np.int64)
    for node, coefficient in instance.problem.h.items():
        units = int(round(coefficient * denominator))
        assert coefficient == units / denominator
        ising_with_offset += units * spins[:, node]
    for (left, right), coefficient in instance.problem.j.items():
        units = int(round(coefficient * denominator))
        assert coefficient == units / denominator
        ising_with_offset += units * spins[:, left] * spins[:, right]

    by_operation: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for item in variables:
        by_operation.setdefault((item["job"], item["machine"]), []).append(
            (item["start"], item["node"])
        )
    valid = np.ones(len(bits), dtype=bool)
    direct_objective = np.zeros(len(bits), dtype=np.int64)
    selected_start: dict[tuple[int, int], np.ndarray] = {}
    proc = _keyed(derived["proc"])
    priority = _keyed(derived["objective"]["priority_units"])
    tardiness_weight = _keyed(derived["objective"]["tardiness_units"])
    due_dates = _keyed(derived["objective"]["due_dates"])
    horizon = instance.kwargs["horizon"]
    for operation, choices in by_operation.items():
        nodes = [node for _, node in choices]
        valid &= bits[:, nodes].sum(axis=1) == 1
        starts = sum(start * bits[:, node] for start, node in choices)
        selected_start[operation] = starts
        valid &= starts + proc[operation] <= horizon
        for start, node in choices:
            completion = start + proc[operation]
            cost = priority[operation] * completion + tardiness_weight[operation] * max(
                0, completion - due_dates[operation]
            )
            direct_objective += cost * bits[:, node]
    for raw_job, route in derived["routes"].items():
        job = int(raw_job)
        for first_machine, second_machine in pairwise(route):
            first, second = (job, first_machine), (job, second_machine)
            valid &= selected_start[second] >= selected_start[first] + proc[first]
    n_jobs = instance.kwargs["n_jobs"]
    n_machines = instance.kwargs["n_machines"]
    for machine in range(n_machines):
        for first_job in range(n_jobs):
            for second_job in range(first_job + 1, n_jobs):
                first, second = (first_job, machine), (second_job, machine)
                first_start = selected_start[first]
                second_start = selected_start[second]
                overlap = (first_start < second_start + proc[second]) & (
                    second_start < first_start + proc[first]
                )
                valid &= ~overlap
    return qubo, ising_with_offset, direct_objective, valid


@pytest.mark.parametrize(("n_jobs", "n_machines", "horizon"), PRODUCTION_SHAPES)
@pytest.mark.parametrize("seed", (0, 7, 31))
def test_jobshop_exact_qubo_ising_replay_and_all_ground_states_are_feasible(
    n_jobs: int,
    n_machines: int,
    horizon: int,
    seed: int,
) -> None:
    instance = jobshop(
        n_jobs=n_jobs,
        n_machines=n_machines,
        horizon=horizon,
        seed=seed,
    )
    qubo, ising_with_offset, direct_objective, valid = _state_table(instance)
    encoding = instance.derived["encoding"]

    assert np.array_equal(4 * qubo, ising_with_offset)
    assert valid.any()
    assert np.array_equal(qubo[valid], direct_objective[valid])
    assert qubo.min() == direct_objective[valid].min()
    assert np.all(qubo[valid] <= encoding["objective_upper_bound_units"])
    assert np.all(qubo[~valid] >= encoding["penalty_units"])
    assert encoding["penalty_units"] == encoding["objective_upper_bound_units"] + 1
    assert np.all(valid[qubo == qubo.min()])

    planted = instance.derived["planted_schedule"]
    planted_mask = 0
    for item in encoding["variable_order"]:
        operation = f"{item['job']}-{item['machine']}"
        if item["start"] == planted[operation]:
            planted_mask |= 1 << item["node"]
    assert valid[planted_mask]


def test_jobshop_semantic_seed_grid_has_distinct_exact_problem_coefficients() -> None:
    signatures = set()
    for seed in range(64):
        instance = jobshop(n_jobs=2, n_machines=1, horizon=3, seed=seed)
        signatures.add(
            (
                tuple(sorted(instance.problem.h.items())),
                tuple(sorted(instance.problem.j.items())),
            )
        )

    assert len(signatures) == 64
