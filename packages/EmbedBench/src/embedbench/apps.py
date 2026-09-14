"""Application-derived logical problems.

The standing constraint on this project is that instances come from applications, not from
MaxCut-style toys. Three families are implemented here, each reduced to an Ising problem in
the form eq. (2) consumes:

    graphcut   contrast-sensitive Potts binary MRF for image segmentation
    portfolio  cardinality-constrained binary mean-variance optimization with full
               positive-definite factor covariance
    jobshop    synthetic time-indexed scheduling with exact-one, precedence, and
               interval-overlap machine constraints

These are **synthetic instances drawn from application families**, not measurements of real
application data. That distinction is stated here because it belongs in any result that
uses them: the structure of h and J is application-shaped, the numbers are generated.

What matters for Gate L0 is that the load is structured rather than uniform. A family whose
|J_ij| were all equal would make a_q nearly constant across a chain by construction and would
tell L0 nothing, so each family below produces a genuinely spread coupling distribution and
`describe()` reports that spread.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import numpy as np

from embedbench.embedding import LogicalProblem


@dataclass(frozen=True)
class Instance:
    """One logical problem with the provenance needed to rebuild it."""

    problem: LogicalProblem
    family: str
    name: str
    seed: int
    kwargs: dict
    """Exactly the generator arguments, so FAMILIES[family](seed=seed, **kwargs) rebuilds it."""
    derived: dict
    """Values the generator computed rather than received. Recorded, never replayed."""

    def describe(self) -> dict:
        j = np.array([abs(v) for v in self.problem.j.values()]) if self.problem.j else np.zeros(1)
        h = np.array([abs(v) for v in self.problem.h.values()]) if self.problem.h else np.zeros(1)
        return {
            "name": self.name,
            "family": self.family,
            "seed": self.seed,
            "n": self.problem.n,
            "m": len(self.problem.j),
            "abs_J_mean": float(j.mean()),
            "abs_J_cv": float(j.std() / j.mean()) if j.mean() > 0 else 0.0,
            "abs_h_mean": float(h.mean()),
        }


def graphcut_mrf(
    rows: int = 5, cols: int = 5, seed: int = 0, noise: float = 0.6,
    connectivity: int = 4,
) -> Instance:
    """Binary MRF for foreground/background segmentation on a `rows` x `cols` grid.

    A disc defines the latent foreground and the observation is corrupted by Gaussian
    noise.  Under ``x=(1+s)/2``, ``h_i=1-2*y_i`` represents twice the squared unary
    data-fidelity term, up to an additive constant.  It is not identified as a literal
    log-likelihood ratio because no Gaussian variance or likelihood scale is fixed here.
    Smoothness couples neighbours with a strength that falls with the observed gradient,
    which gives ``|J|`` its spread: edges inside a region couple strongly and edges across
    the boundary weakly.  The latent disc is a data-generation variable, not a certificate
    that its labeling minimizes the noisy, regularized energy.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:rows, 0:cols]
    cy, cx = (rows - 1) / 2, (cols - 1) / 2
    radius = 0.30 * min(rows, cols)
    truth = ((yy - cy) ** 2 + (xx - cx) ** 2) <= radius**2
    observed = truth.astype(float) + rng.normal(0.0, noise, size=truth.shape)

    idx = {(r, c): r * cols + c for r in range(rows) for c in range(cols)}
    h = {idx[(r, c)]: float(1.0 - 2.0 * observed[r, c]) for r in range(rows) for c in range(cols)}

    j: dict[tuple[int, int], float] = {}
    beta = 1.4
    for r in range(rows):
        for c in range(cols):
            offsets = ((0, 1), (1, 0)) if connectivity == 4 else (
                (0, 1), (1, 0), (1, 1), (1, -1)
            )
            for dr, dc in offsets:
                r2, c2 = r + dr, c + dc
                if not (0 <= r2 < rows and 0 <= c2 < cols):
                    continue
                grad = abs(observed[r, c] - observed[r2, c2])
                # contrast-sensitive smoothness; ferromagnetic, so negative in Ising sign
                j[(idx[(r, c)], idx[(r2, c2)])] = -float(beta * np.exp(-(grad**2) / 0.5))

    return Instance(
        problem=LogicalProblem.from_dicts(h, j),
        family="graphcut",
        name=f"graphcut{connectivity}-{rows}x{cols}-{seed}",
        seed=seed,
        kwargs={"rows": rows, "cols": cols, "noise": noise, "connectivity": connectivity},
        derived={"beta": beta},
    )


def portfolio(n_assets: int = 16, seed: int = 0, n_factors: int = 3, risk: float = 1.0) -> Instance:
    """Cardinality-constrained binary mean-variance portfolio as an exact Ising model.

    The covariance is the full positive-definite integer matrix ``A A^T + D``; it is never
    thresholded or repaired with graph edges.  A rigorous global objective lower bound and
    an explicit cardinality-feasible witness determine a penalty that makes every QUBO
    ground state cardinality-valid.  The QUBO is converted with ``x=(1+s)/2`` and its exact
    energy offset and dyadic coefficient denominators are recorded.
    """
    if type(n_assets) is not int or n_assets <= 0:
        raise ValueError("n_assets must be a positive integer")
    if type(n_factors) is not int or n_factors <= 0:
        raise ValueError("n_factors must be a positive integer")
    if isinstance(risk, bool) or not isinstance(risk, (int, float)):
        raise ValueError("risk must be a positive finite number on the 1/256 grid")
    normalized_risk = float(risk)
    if not math.isfinite(normalized_risk) or normalized_risk <= 0.0:
        raise ValueError("risk must be a positive finite number on the 1/256 grid")
    qubo_denominator = 256
    risk_units = int(round(normalized_risk * qubo_denominator))
    if normalized_risk != risk_units / qubo_denominator or risk_units <= 0:
        raise ValueError("risk must be exactly representable on the 1/256 grid")

    rng = np.random.default_rng(seed)
    loadings = rng.integers(-2, 3, size=(n_assets, n_factors), dtype=np.int64)
    idiosyncratic = rng.integers(1, 5, size=n_assets, dtype=np.int64)
    covariance = loadings @ loadings.T + np.diag(idiosyncratic)
    expected_return_units = rng.integers(64, 513, size=n_assets, dtype=np.int64)
    cardinality = max(1, n_assets // 3)

    singleton_scores = [
        risk_units * int(covariance[index, index]) - int(expected_return_units[index])
        for index in range(n_assets)
    ]
    feasible_witness = tuple(
        sorted(
            sorted(
                range(n_assets),
                key=lambda index: (singleton_scores[index], index),
            )[:cardinality]
        )
    )
    witness = np.zeros(n_assets, dtype=np.int64)
    witness[list(feasible_witness)] = 1
    feasible_witness_objective_units = int(
        risk_units * (witness @ covariance @ witness) - expected_return_units @ witness
    )
    global_objective_lower_bound_units = -sum(
        max(0, int(value)) for value in expected_return_units
    )
    penalty_units = (
        feasible_witness_objective_units - global_objective_lower_bound_units + 1
    )
    if penalty_units <= 0:
        raise RuntimeError("portfolio penalty proof did not produce a positive penalty")

    qubo_linear = {
        index: (
            risk_units * int(covariance[index, index])
            - int(expected_return_units[index])
            + penalty_units * (1 - 2 * cardinality)
        )
        for index in range(n_assets)
    }
    qubo_quadratic = {
        (left, right): (
            2 * risk_units * int(covariance[left, right]) + 2 * penalty_units
        )
        for left in range(n_assets)
        for right in range(left + 1, n_assets)
    }
    binary_offset_units = penalty_units * cardinality**2
    ising_denominator = 4 * qubo_denominator
    h_units = {node: 2 * units for node, units in qubo_linear.items()}
    for (left, right), units in qubo_quadratic.items():
        h_units[left] += units
        h_units[right] += units
    j_units = dict(qubo_quadratic)
    ising_offset_units = (
        4 * binary_offset_units
        + 2 * sum(qubo_linear.values())
        + sum(qubo_quadratic.values())
    )
    h = {node: units / ising_denominator for node, units in h_units.items()}
    j = {edge: units / ising_denominator for edge, units in j_units.items()}
    return Instance(
        problem=LogicalProblem.from_dicts(h, j),
        family="portfolio",
        name=f"portfolio-{n_assets}-{seed}",
        seed=seed,
        kwargs={"n_assets": n_assets, "n_factors": n_factors, "risk": normalized_risk},
        derived={
            "cardinality": cardinality,
            "covariance": {
                "factor_loadings": loadings.tolist(),
                "idiosyncratic_diagonal": idiosyncratic.tolist(),
                "protocol": "integer-factor-positive-definite-v1",
                "sigma_units": covariance.tolist(),
            },
            "encoding": {
                "binary_offset_units": binary_offset_units,
                "constraint_protocol": "exact-cardinality-square-penalty-v1",
                "feasible_witness_objective_units": feasible_witness_objective_units,
                "global_objective_lower_bound_units": global_objective_lower_bound_units,
                "ising_coefficient_denominator": ising_denominator,
                "ising_offset_units": ising_offset_units,
                "penalty_units": penalty_units,
                "qubo_coefficient_denominator": qubo_denominator,
                "qubo_linear_units": [list(item) for item in sorted(qubo_linear.items())],
                "qubo_quadratic_units": [
                    [left, right, units]
                    for (left, right), units in sorted(qubo_quadratic.items())
                ],
                "spin_substitution": "x=(1+s)/2",
            },
            "objective": {
                "coefficient_denominator": qubo_denominator,
                "expected_return_units": expected_return_units.tolist(),
                "protocol": "cardinality-constrained-binary-mean-variance-v1",
                "risk_units": risk_units,
            },
            "feasible_witness": list(feasible_witness),
        },
    )


def jobshop(n_jobs: int = 4, n_machines: int = 3, seed: int = 0, horizon: int = 4) -> Instance:
    """Synthetic time-indexed job-shop QUBO converted exactly to an Ising model.

    The registered production family uses either two jobs on one machine with seeded
    processing durations that fit the horizon, or two jobs on two machines with unit
    durations and a constructive pipeline schedule.  The latter unit-duration restriction
    is deliberate: it guarantees feasibility for the smallest 2x2 horizon rather than
    silently emitting impossible scheduling instances.

    Binary variables ``x(operation, start)`` receive exact-one, invalid-late-start,
    precedence, and full interval-overlap penalties.  The integer penalty is one unit
    larger than an upper bound on the entire nonnegative weighted-completion/tardiness
    objective.  Consequently every QUBO ground state is feasible.  The returned Ising
    coefficients use ``x=(1+s)/2`` and the exact energy offset is recorded in ``derived``.
    """
    if any(type(value) is not int or value <= 0 for value in (n_jobs, n_machines, horizon)):
        raise ValueError("n_jobs, n_machines, and horizon must be positive integers")
    schedule_window = max(n_jobs, n_machines)
    if horizon < schedule_window:
        raise ValueError("horizon is too short for the constructive job-shop schedule")

    rng = np.random.default_rng(seed)
    ops = tuple((job, machine) for job in range(n_jobs) for machine in range(n_machines))
    proc = {op: 1 for op in ops}
    planted_start: dict[tuple[int, int], int] = {}
    routes: dict[int, tuple[int, ...]] = {}
    if n_machines == 1:
        job_order = tuple(int(job) for job in rng.permutation(n_jobs))
        slack = horizon - n_jobs
        extra = tuple(int(value) for value in rng.multinomial(slack, [1.0 / n_jobs] * n_jobs))
        for job, duration_extra in zip(job_order, extra, strict=True):
            proc[job, 0] += duration_extra
        cursor = 0
        for job in job_order:
            planted_start[job, 0] = cursor
            cursor += proc[job, 0]
        routes = {job: (0,) for job in range(n_jobs)}
    else:
        rotation = int(rng.integers(0, schedule_window))
        for time in range(schedule_window):
            for job in range(n_jobs):
                machine = (job + time + rotation) % schedule_window
                if machine < n_machines:
                    planted_start[job, machine] = time
        routes = {
            job: tuple(
                machine
                for _, machine in sorted(
                    (planted_start[job, machine], machine)
                    for machine in range(n_machines)
                )
            )
            for job in range(n_jobs)
        }

    priority_units = {op: int(rng.integers(64, 256)) for op in ops}
    tardiness_units = {op: int(rng.integers(128, 512)) for op in ops}
    due_dates = {op: int(rng.integers(1, horizon + 1)) for op in ops}
    qubo_denominator = 4096
    ising_denominator = 4 * qubo_denominator
    var = {
        (op, time): index
        for index, (op, time) in enumerate(
            (operation, time) for operation in ops for time in range(horizon)
        )
    }

    objective_units: dict[tuple[tuple[int, int], int], int] = {}
    for op in ops:
        for time in range(horizon):
            completion = time + proc[op]
            objective_units[op, time] = (
                priority_units[op] * completion
                + tardiness_units[op] * max(0, completion - due_dates[op])
            )
    objective_upper_bound_units = sum(
        max(
            objective_units[op, time]
            for time in range(horizon)
            if time + proc[op] <= horizon
        )
        for op in ops
    )
    penalty_units = objective_upper_bound_units + 1
    qubo_linear = {
        var[op, time]: objective_units[op, time]
        for op in ops
        for time in range(horizon)
    }
    qubo_quadratic: dict[tuple[int, int], int] = {}
    binary_offset_units = 0

    def add_pair(left: int, right: int, units: int) -> None:
        key = (left, right) if left < right else (right, left)
        qubo_quadratic[key] = qubo_quadratic.get(key, 0) + units

    for op in ops:
        binary_offset_units += penalty_units
        for time in range(horizon):
            node = var[op, time]
            qubo_linear[node] -= penalty_units
            if time + proc[op] > horizon:
                qubo_linear[node] += penalty_units
        for first_time in range(horizon):
            for second_time in range(first_time + 1, horizon):
                add_pair(
                    var[op, first_time],
                    var[op, second_time],
                    2 * penalty_units,
                )

    for job in range(n_jobs):
        route = routes[job]
        for first_machine, second_machine in pairwise(route):
            first_op = (job, first_machine)
            second_op = (job, second_machine)
            for first_time in range(horizon):
                for second_time in range(horizon):
                    if second_time < first_time + proc[first_op]:
                        add_pair(
                            var[first_op, first_time],
                            var[second_op, second_time],
                            penalty_units,
                        )

    for machine in range(n_machines):
        for first_job in range(n_jobs):
            for second_job in range(first_job + 1, n_jobs):
                first_op = (first_job, machine)
                second_op = (second_job, machine)
                for first_time in range(horizon):
                    first_end = first_time + proc[first_op]
                    for second_time in range(horizon):
                        second_end = second_time + proc[second_op]
                        if first_time < second_end and second_time < first_end:
                            add_pair(
                                var[first_op, first_time],
                                var[second_op, second_time],
                                penalty_units,
                            )

    h_units = {node: 2 * units for node, units in qubo_linear.items()}
    for (left, right), units in qubo_quadratic.items():
        h_units[left] += units
        h_units[right] += units
    j_units = dict(qubo_quadratic)
    ising_offset_units = (
        4 * binary_offset_units
        + 2 * sum(qubo_linear.values())
        + sum(qubo_quadratic.values())
    )
    h = {node: units / ising_denominator for node, units in h_units.items()}
    j = {edge: units / ising_denominator for edge, units in j_units.items()}

    def keyed(values: dict[tuple[int, int], int]) -> dict[str, int]:
        return {f"{job}-{machine}": values[job, machine] for job, machine in ops}

    variable_order = [
        {
            "job": op[0],
            "machine": op[1],
            "node": var[op, time],
            "start": time,
        }
        for op in ops
        for time in range(horizon)
    ]
    return Instance(
        problem=LogicalProblem.from_dicts(h, j),
        family="jobshop",
        name=f"jobshop-{n_jobs}x{n_machines}-{seed}",
        seed=seed,
        kwargs={"n_jobs": n_jobs, "n_machines": n_machines, "horizon": horizon},
        derived={
            "encoding": {
                "binary_offset_units": binary_offset_units,
                "constraint_protocol": (
                    "time-indexed-exact-one-late-precedence-interval-overlap-v2"
                ),
                "ising_coefficient_denominator": ising_denominator,
                "ising_offset_units": ising_offset_units,
                "objective_upper_bound_units": objective_upper_bound_units,
                "penalty_units": penalty_units,
                "qubo_coefficient_denominator": qubo_denominator,
                "qubo_linear_units": [list(item) for item in sorted(qubo_linear.items())],
                "qubo_quadratic_units": [
                    [left, right, units]
                    for (left, right), units in sorted(qubo_quadratic.items())
                ],
                "spin_substitution": "x=(1+s)/2",
                "variable_order": variable_order,
            },
            "objective": {
                "due_dates": keyed(due_dates),
                "priority_units": keyed(priority_units),
                "protocol": "weighted-completion-tardiness-v1",
                "tardiness_units": keyed(tardiness_units),
            },
            "planted_schedule": keyed(planted_start),
            "proc": keyed(proc),
            "routes": {str(job): list(routes[job]) for job in range(n_jobs)},
        },
    )


FAMILIES = {"graphcut": graphcut_mrf, "portfolio": portfolio, "jobshop": jobshop}
