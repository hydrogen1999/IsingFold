"""A solve-probability signal, built here rather than imported.

Section 7 of architecture/SPEC.tex makes solve probability the objective and refuses any
structural proxy for it. Gate L1 therefore needs a number per embedding, and this module
produces one: program the embedded Ising, anneal it classically, decode the chains by
majority, and count how often the logical ground state comes back.

Three modelling choices decide whether the number means anything.

**Autoscaling, and the fixed thermal scale it needs to mean anything.** A real annealer has
a bounded coupling range, so after the chain couplers are added at strength F the whole
Hamiltonian is rescaled to fit. The intent is that F competes with the problem: a chain held
too hard flattens the couplings that carry the answer.

Rescaling alone does **not** achieve that here, and the claim that it did was false.
`SimulatedAnnealingSampler` recomputes its temperature schedule from whatever model it is
handed, so the scaling cancels exactly; verified by bit-identical sample arrays with
autoscaling on and off at F = 0.4, 2 and 20. The competition only exists against a **fixed**
thermal scale, so `BETA_RANGE` is passed explicitly. With it, compressing the problem
couplings really does cost solve probability, which is the physics the surrogate is meant to
carry.

**Its own optimal F per embedding.** Section 7 requires it, so that no measured gap between
two embeddings is really a gap in strength selection. `solve_probability` sweeps a grid and
reports the maximum together with the whole curve.

**Majority decoding with random tie-breaking.** A broken chain has no logical value; taking
the majority and breaking ties by coin flip is the standard reading and it is what makes a
chain break cost solve probability rather than silently resolve in the model's favour.

This is a *surrogate*. It is classical simulated annealing, not hardware, and every number
it produces carries that qualification.
"""
from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import dimod
import numpy as np
from dwave.samplers import SimulatedAnnealingSampler, TabuSampler

from embedbench.embedding import Embedding, LogicalProblem, Node, Qubit
from embedbench.programming import program

EXACT_GROUND_STATE_MAX_N = 22
"""Above this, brute force over 2^N spin states stops being affordable."""

BETA_RANGE = (0.1, 8.0)
"""The fixed inverse-temperature schedule, in the units the autoscaled model lives in.

Fixed rather than derived, because a schedule recomputed per model cancels the rescaling
that represents the hardware's bounded coupling range. Every embedding and every chain
strength is annealed against the same thermal floor, so an embedding that needs a larger F
really does pay for it in compressed problem couplings."""


def physical_ising(
    embedding: Embedding,
    problem: LogicalProblem,
    chain_strength: float,
    autoscale: bool = True,
) -> dimod.BinaryQuadraticModel:
    """The embedded Ising problem at a given uniform chain strength.

    Couplings come from Pi(phi); each programmed chain coupler is added ferromagnetically at
    -F. With `autoscale`, everything is then divided by the largest absolute coefficient, so
    the model sits inside a fixed coupling range the way hardware does.
    """
    coeffs = program(embedding, problem)
    h = dict(coeffs.h_phys)
    j = dict(coeffs.j_phys)
    for chain_edges in embedding.chain_edges.values():
        for e in chain_edges:
            j[e] = j.get(e, 0.0) - abs(chain_strength)

    if autoscale:
        scale = max(
            max((abs(v) for v in h.values()), default=0.0),
            max((abs(v) for v in j.values()), default=0.0),
        )
        if scale > 0:
            h = {k: v / scale for k, v in h.items()}
            j = {k: v / scale for k, v in j.items()}
    return dimod.BinaryQuadraticModel(h, j, 0.0, dimod.SPIN)


def logical_energy(problem: LogicalProblem, spins: Mapping[Node, int]) -> float:
    """The logical objective, which is what a solution is scored against."""
    e = sum(problem.h.get(i, 0.0) * spins[i] for i in spins)
    e += sum(w * spins[u] * spins[v] for (u, v), w in problem.j.items())
    return float(e)


@dataclass(frozen=True)
class GroundState:
    energy: float
    exact: bool
    """True when found by exhaustive enumeration rather than by search."""
    agreement: int
    """How many independent restarts reached this energy. 1 means low confidence."""
    restarts: int


def logical_ground_state(
    problem: LogicalProblem, restarts: int = 12, timeout_ms: int = 2000, seed: int = 0
) -> GroundState:
    """The reference energy p_solve counts hits against.

    Exhaustive below `EXACT_GROUND_STATE_MAX_N`; above it, the best energy found by
    independent tabu restarts, with the number that agreed reported so a weak reference is
    visible rather than assumed. A p_solve measured against an unconfirmed reference is a
    p_solve against the best thing anyone found, and that has to be said out loud.
    """
    nodes = sorted(problem.h)
    if len(nodes) <= EXACT_GROUND_STATE_MAX_N:
        best = np.inf
        for bits in itertools.product((-1, 1), repeat=len(nodes)):
            e = logical_energy(problem, dict(zip(nodes, bits, strict=True)))
            best = min(best, e)
        return GroundState(energy=float(best), exact=True, agreement=1, restarts=1)

    bqm = dimod.BinaryQuadraticModel(dict(problem.h), dict(problem.j), 0.0, dimod.SPIN)
    sampler = TabuSampler()
    energies = []
    for r in range(restarts):
        ss = sampler.sample(bqm, num_reads=1, timeout=timeout_ms, seed=seed + r)
        energies.append(float(ss.record.energy.min()))
    best = min(energies)
    agree = sum(1 for e in energies if e <= best + 1e-9)
    return GroundState(energy=best, exact=False, agreement=agree, restarts=restarts)


def majority_decode(
    sample: Mapping[Qubit, int],
    chains: Mapping[Node, frozenset[Qubit]],
    rng: np.random.Generator,
) -> tuple[dict[Node, int], int]:
    """Decode physical spins to logical ones, returning the number of broken chains.

    A chain is broken when its qubits disagree. Ties are broken by a coin flip, which is the
    standard reading and the one that makes a break cost something.
    """
    out: dict[Node, int] = {}
    broken = 0
    for i, chain in chains.items():
        vals = [sample[q] for q in chain]
        total = sum(vals)
        if total != len(vals) and total != -len(vals):
            broken += 1
        if total > 0:
            out[i] = 1
        elif total < 0:
            out[i] = -1
        else:
            out[i] = int(rng.choice((-1, 1)))
    return out, broken


@dataclass(frozen=True)
class SolveResult:
    p_solve: float
    chain_strength: float
    broken_fraction: float
    num_reads: int
    mean_residual: float = 0.0
    """Mean over reads of (E - E0) / |E0| of the decoded logical state (the scale tier's
    objective, where p_solve is at the floor); 0 when every read is at the ground state."""


def solve_probability_at(
    embedding: Embedding,
    problem: LogicalProblem,
    ground_energy: float,
    chain_strength: float,
    num_reads: int = 200,
    num_sweeps: int = 200,
    seed: int = 0,
    tol: float = 1e-6,
    ice: float = 0.0,
    sampler: str = "sa",
) -> SolveResult:
    """Fraction of anneals that decode to the logical ground state, at one F. `ice` > 0 adds
    integrated-control-error style noise to the programmed coefficients: every physical h
    and J is perturbed by Gaussian noise with standard deviation `ice` times the largest
    |J| of the programmed problem, drawn once per call (one programming)."""
    bqm = physical_ising(embedding, problem, chain_strength)
    if ice > 0:
        nrng = np.random.default_rng(seed + 977)
        scale = ice * max([abs(v) for v in bqm.quadratic.values()] + [1e-9])
        for v in list(bqm.linear):
            bqm.set_linear(v, bqm.get_linear(v) + float(nrng.normal(0.0, scale)))
        for (u, v), j in list(bqm.quadratic.items()):
            bqm.set_quadratic(u, v, j + float(nrng.normal(0.0, scale)))
    if sampler == "sa":
        ss = SimulatedAnnealingSampler().sample(
            bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed, beta_range=BETA_RANGE
        )
    elif sampler in ("tabu", "sd"):
        # A second heuristic with different dynamics. Any claim about which embedding anneals
        # better has to survive a change of the machine that does the annealing, or it is a
        # statement about simulated annealing rather than about the embedding. Steepest descent
        # is started from random states, so its reads are independent like the annealer's.
        if sampler == "tabu":
            from dwave.samplers import TabuSampler as _S
            ss = _S().sample(bqm, num_reads=num_reads, seed=seed)
        else:
            from dwave.samplers import SteepestDescentSampler as _S
            ss = _S().sample(bqm, num_reads=num_reads, seed=seed)
    else:
        raise ValueError(f"unknown sampler {sampler}")
    variables = list(ss.variables)
    rng = np.random.default_rng(seed)
    hits = 0
    broken_total = 0
    resid = 0.0
    for row in ss.record.sample:
        sample = dict(zip(variables, row.tolist(), strict=True))
        spins, broken = majority_decode(sample, embedding.chains, rng)
        broken_total += broken
        e = logical_energy(problem, spins)
        if e <= ground_energy + tol:
            hits += 1
        resid += (e - ground_energy) / max(abs(ground_energy), 1e-9)
    n_chains = len(embedding.chains)
    return SolveResult(
        p_solve=hits / num_reads,
        chain_strength=chain_strength,
        broken_fraction=broken_total / (num_reads * n_chains),
        num_reads=num_reads,
        mean_residual=resid / num_reads,
    )


def decoded_energies(
    embedding: Embedding,
    problem: LogicalProblem,
    chain_strength: float,
    *,
    num_reads: int = 1000,
    num_sweeps: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """The decoded logical energy of every read, which is the spectrum a figure needs.

    `solve_probability_at` reduces the same reads to a hit count; this keeps them, so the
    distribution over logical energies at one chain strength can be plotted or fitted.
    """

    bqm = physical_ising(embedding, problem, chain_strength)
    ss = SimulatedAnnealingSampler().sample(
        bqm, num_reads=num_reads, num_sweeps=num_sweeps, seed=seed, beta_range=BETA_RANGE
    )
    variables = list(ss.variables)
    rng = np.random.default_rng(seed)
    out = []
    for row in ss.record.sample:
        spins, _ = majority_decode(dict(zip(variables, row.tolist(), strict=True)), embedding.chains, rng)
        out.append(logical_energy(problem, spins))
    return np.asarray(out, dtype=float)


def default_strength_grid(problem: LogicalProblem, n: int = 9) -> list[float]:
    """A grid of F spanning the scale the couplings actually set.

    Anchored on the mean absolute coupling rather than on absolute numbers, so the same grid
    is meaningful across families whose coefficients differ by orders of magnitude.
    """
    mags = [abs(v) for v in problem.j.values()] or [1.0]
    unit = float(np.mean(mags))
    return [float(unit * m) for m in np.geomspace(0.25, 8.0, n)]


@dataclass(frozen=True)
class EmbeddingScore:
    """One embedding's operational quality, at its own optimal chain strength."""

    p_solve: float
    best_chain_strength: float
    curve: tuple[tuple[float, float], ...]
    broken_fraction: float
    ground_exact: bool
    ground_agreement: int

    @property
    def strength_is_interior(self) -> bool:
        """True when the optimum is not at an endpoint of the grid.

        An optimum on the boundary means the grid did not contain it, so `p_solve` is a
        lower bound and comparing two embeddings on it can be comparing grid edges.
        """
        strengths = [f for f, _ in self.curve]
        return self.best_chain_strength not in (strengths[0], strengths[-1])


def score_embedding(
    embedding: Embedding,
    problem: LogicalProblem,
    ground: GroundState,
    strengths: Sequence[float] | None = None,
    num_reads: int = 200,
    num_sweeps: int = 200,
    seed: int = 0,
    ice: float = 0.0,
) -> EmbeddingScore:
    """Sweep F and report the best, as Section 7 requires of every embedding."""
    grid = list(strengths) if strengths is not None else default_strength_grid(problem)
    results = [
        solve_probability_at(
            embedding, problem, ground.energy, f,
            num_reads=num_reads, num_sweeps=num_sweeps, seed=seed + k, ice=ice,
        )
        for k, f in enumerate(grid)
    ]
    best = max(results, key=lambda r: r.p_solve)
    return EmbeddingScore(
        p_solve=best.p_solve,
        best_chain_strength=best.chain_strength,
        curve=tuple((r.chain_strength, r.p_solve) for r in results),
        broken_fraction=best.broken_fraction,
        ground_exact=ground.exact,
        ground_agreement=ground.agreement,
    )
