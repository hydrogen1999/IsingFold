"""Independent exhaustive oracle for compiled Ising-program conformance.

The production compiler and its coefficient-sum checker live in :mod:`isingfold.rl.program`.
This module deliberately imports neither.  For a small logical problem it enumerates every
logical spin assignment, expands that assignment onto the supplied chains, and compares the
physical-program energy with ``scale * logical_energy + offset``.  Gate 1 therefore cannot be
certified solely by two routines that share the compiler's implementation assumptions.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Mapping, Protocol

import networkx as nx

Node = Hashable
Qubit = Hashable
PROGRAM_ORACLE_VERSION = "program-energy-enumeration-v1"


class ProblemLike(Protocol):
    h: Mapping[Node, float]
    j: Mapping[tuple[Node, Node], float]


class ProgramLike(Protocol):
    scale: float
    offset: float
    h_phys: Mapping[Qubit, float]
    j_phys: Mapping[tuple[Qubit, Qubit], float]


@dataclass(frozen=True)
class ProgramOracleResult:
    ok: bool
    assignments_checked: int
    max_abs_error: float
    reasons: tuple[str, ...]


def implementation_identity() -> dict[str, object]:
    return {
        "implementation": "isingfold.rl.data.program_oracle.brute_force_program_identity",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "version": PROGRAM_ORACLE_VERSION,
    }


def brute_force_program_identity(
    chains: Mapping[Node, frozenset[Qubit]],
    host: nx.Graph,
    problem: ProblemLike,
    program: ProgramLike,
    *,
    field_limit: float,
    coupler_limit: float,
    atol: float = 1e-9,
) -> ProgramOracleResult:
    """Check one compiled program by exhaustive energy evaluation.

    This intentionally uses the public coefficient maps only.  It does not trust compiler
    contact counts, chain-edge registries, or a cached faithfulness report.
    """

    reasons: list[str] = []
    owners: dict[Qubit, Node] = {}
    for node, chain in chains.items():
        if not chain:
            reasons.append("empty chain")
        for qubit in chain:
            if qubit in owners:
                reasons.append("overlapping chains")
            owners[qubit] = node
    occupied = set(owners)
    if set(program.h_phys) != occupied:
        reasons.append("physical-field domain differs from occupied qubits")
    if any(
        left not in occupied or right not in occupied or not host.has_edge(left, right)
        for left, right in program.j_phys
    ):
        reasons.append("physical coupler is off-chain or off-host")

    scalar_values = [
        float(program.scale),
        float(program.offset),
        *(float(value) for value in program.h_phys.values()),
        *(float(value) for value in program.j_phys.values()),
    ]
    if not all(math.isfinite(value) for value in scalar_values):
        reasons.append("program contains non-finite coefficients")
    if float(program.scale) <= 0.0 or float(program.scale) > 1.0 + atol:
        reasons.append("program scale is outside the shrink-only interval")
    if max((abs(float(value)) for value in program.h_phys.values()), default=0.0) > (
        field_limit + atol
    ):
        reasons.append("physical field exceeds the registered limit")
    if max((abs(float(value)) for value in program.j_phys.values()), default=0.0) > (
        coupler_limit + atol
    ):
        reasons.append("physical coupler exceeds the registered limit")

    nodes = tuple(sorted(chains, key=lambda value: (type(value).__qualname__, repr(value))))
    if set(problem.h) - set(nodes) or any(
        left not in chains or right not in chains for left, right in problem.j
    ):
        reasons.append("problem support differs from the chain domain")
    if reasons:
        return ProgramOracleResult(False, 0, math.inf, tuple(sorted(set(reasons))))

    max_error = 0.0
    checked = 0
    for raw_spins in itertools.product((-1, 1), repeat=len(nodes)):
        checked += 1
        logical_spins = dict(zip(nodes, raw_spins, strict=True))
        physical_spins = {qubit: logical_spins[node] for qubit, node in owners.items()}
        logical_energy = sum(
            float(coefficient) * logical_spins[node]
            for node, coefficient in problem.h.items()
        ) + sum(
            float(coefficient) * logical_spins[left] * logical_spins[right]
            for (left, right), coefficient in problem.j.items()
        )
        physical_energy = sum(
            float(coefficient) * physical_spins[qubit]
            for qubit, coefficient in program.h_phys.items()
        ) + sum(
            float(coefficient) * physical_spins[left] * physical_spins[right]
            for (left, right), coefficient in program.j_phys.items()
        )
        expected = float(program.scale) * logical_energy + float(program.offset)
        error = abs(physical_energy - expected)
        max_error = max(max_error, error)
        if not math.isclose(physical_energy, expected, rel_tol=0.0, abs_tol=atol):
            reasons.append("aligned physical energy differs from the logical program identity")
            break
    return ProgramOracleResult(not reasons, checked, max_error, tuple(sorted(set(reasons))))


__all__ = [
    "PROGRAM_ORACLE_VERSION",
    "ProgramOracleResult",
    "brute_force_program_identity",
    "implementation_identity",
]
