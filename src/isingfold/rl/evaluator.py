"""The independent evaluator: fresh reads of the actually compiled program.

Spec: Rev2 Eq. (pj) and MODEL_SPEC Eq. (5). It samples the returned program bytes, decodes by
the registered decoder and compares against a certified logical optimum. The trusted
evaluator may know ``E_0``; the actor and the strength selector never do. A valid zero-hit
block is a legitimate reward zero, while a missing read is an experiment error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable, Mapping

import numpy as np

from isingfold.embedding import LogicalProblem
from isingfold.rl.program import Program

Node = Hashable
Qubit = Hashable


def energy_tolerance(problem: LogicalProblem) -> float:
    """``eps_I`` of MODEL_SPEC Eq. (5): scaled to the instance, not an absolute constant."""

    mass = sum(abs(v) for v in problem.h.values()) + sum(abs(v) for v in problem.j.values())
    return 1e-8 * max(1.0, mass)


def logical_energy(problem: LogicalProblem, spins: Mapping[Node, int]) -> float:
    total = sum(problem.h.get(i, 0.0) * spins[i] for i in spins)
    for (u, v), w in problem.j.items():
        total += w * spins[u] * spins[v]
    return float(total)


def majority_decode(
    sample: Mapping[Qubit, int],
    chains: Mapping[Node, frozenset[Qubit]],
    rng: np.random.Generator,
) -> tuple[dict[Node, int], int]:
    spins: dict[Node, int] = {}
    broken = 0
    for i, chain in chains.items():
        votes = sum(1 if sample[q] > 0 else -1 for q in chain)
        if votes == 0:
            spins[i] = 1 if rng.random() < 0.5 else -1
            broken += 1
        else:
            spins[i] = 1 if votes > 0 else -1
            if any((sample[q] > 0) != (spins[i] > 0) for q in chain):
                broken += 1
    return spins, broken


@dataclass(frozen=True)
class ReadBlock:
    hits: int
    reads: int
    broken_fraction: float
    mean_residual: float
    strength_index: int

    @property
    def rate(self) -> float:
        return self.hits / self.reads if self.reads else 0.0


def sample_program(
    program: Program,
    chains: Mapping[Node, frozenset[Qubit]],
    problem: LogicalProblem,
    ground_energy: float,
    *,
    num_reads: int,
    seed: int,
    num_sweeps: int = 200,
    tolerance: float | None = None,
    beta_range: tuple[float, float] | None = None,
) -> ReadBlock:
    """Draw a fresh independent block at one strength and count decoded ground states."""

    if isinstance(num_reads, bool) or not isinstance(num_reads, int) or num_reads <= 0:
        raise ValueError("num_reads must be a positive integer")
    if isinstance(num_sweeps, bool) or not isinstance(num_sweeps, int) or num_sweeps <= 0:
        raise ValueError("num_sweeps must be a positive integer")

    import dimod
    from dwave.samplers import SimulatedAnnealingSampler

    bqm = dimod.BinaryQuadraticModel(
        {q: float(v) for q, v in program.h_phys.items()},
        {(a, b): float(w) for (a, b), w in program.j_phys.items()},
        0.0,
        dimod.SPIN,
    )
    # Sorted, for the reason given in `program.compile_program`: a qubit that reaches the model
    # through this loop rather than through h_phys was being added in per-process hash order.
    for node in sorted(chains, key=str):
        for q in sorted(chains[node], key=str):
            if q not in bqm.variables:
                bqm.add_variable(q, 0.0)

    # Passing the range through matters: left to itself the sampler derives beta from h and J,
    # so a Hamiltonian multiplied by a common factor gets a schedule divided by it and the
    # rescaling has no effect on the samples at all.
    extra = {} if beta_range is None else {"beta_range": list(beta_range)}
    sampleset = SimulatedAnnealingSampler().sample(
        bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed, **extra
    )
    tol = energy_tolerance(problem) if tolerance is None else tolerance
    rng = np.random.default_rng(seed)
    variables = list(sampleset.variables)
    rows = np.asarray(sampleset.record.sample)
    if rows.ndim != 2 or rows.shape[0] != num_reads:
        raise ValueError(
            f"sampler read-count integrity failure: requested {num_reads}, received "
            f"{rows.shape[0] if rows.ndim else 0}"
        )
    if rows.shape[1] != len(variables):
        raise ValueError("sampler variable/read matrix shape mismatch")
    required_qubits = frozenset(q for chain in chains.values() for q in chain)
    if not required_qubits.issubset(variables):
        raise ValueError("sampler output is missing programmed chain variables")
    if not np.isin(rows, (-1, 1)).all():
        raise ValueError("sampler returned a non-SPIN sample")
    hits = 0
    broken_total = 0
    residual = 0.0
    denom = max(abs(ground_energy), 1e-9)
    for row in rows:
        sample = dict(zip(variables, row.tolist(), strict=True))
        spins, broken = majority_decode(sample, chains, rng)
        broken_total += broken
        e = logical_energy(problem, spins)
        if e <= ground_energy + tol:
            hits += 1
        gap = e - ground_energy
        if gap < 0.0:
            # A read that lands exactly on the exact ground state leaves a rounding residue of
            # order 1e-16, and the receipt validator rejects any negative mean, so an instance
            # solved on every read fails evaluation. Anything past the solver tolerance is a
            # real violation and still raises.
            if gap < -tol:
                raise ValueError("decoded energy fell below the exact ground energy")
            gap = 0.0
        residual += gap / denom
    n_chains = max(1, len(chains))
    return ReadBlock(
        hits=hits,
        reads=num_reads,
        broken_fraction=broken_total / (num_reads * n_chains),
        mean_residual=residual / num_reads,
        strength_index=program.strength_index,
    )


def reads_to_99(p: float) -> int | None:
    """``R_99 = log(0.01)/log(1-p)``: a read count, not an end-to-end time to solution."""

    if p <= 0.0:
        return None
    if p >= 1.0:
        return 1
    return int(math.ceil(math.log(0.01) / math.log(1.0 - p)))
