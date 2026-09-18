"""Quality labels of policy-created terminal programs, with no completion solver.

Only the label backend reads a ground-energy reference. The actor and construction
environment never need it. Residuals use the registered majority decoder/schedule.
"""
import math
from numbers import Integral

from _context import REGISTERED_BETA_RANGE


def _sample_terminal(task, terminal, seed, reads, reference_energy):
    """Validate the selected program and sample it against an explicitly supplied reference."""
    from isingfold.rl.evaluator import sample_program

    if terminal is None or not terminal.returned_valid or terminal.embedding is None or terminal.selected_program is None:
        raise ValueError("quality measurement requires a validated COMMIT")
    if not math.isfinite(reference_energy):
        raise ValueError("measurement reference must be finite")
    if terminal.selected_index != terminal.selected_program.strength_index:
        raise ValueError("terminal strength/program mismatch")
    block = sample_program(terminal.selected_program, terminal.embedding, task.problem,
                           reference_energy, num_reads=reads, seed=int(seed), num_sweeps=200,
                           beta_range=REGISTERED_BETA_RANGE)
    if block.reads != reads or block.strength_index != terminal.selected_index:
        raise ValueError("quality evaluator violated the declared read/strength contract")
    residual = float(block.mean_residual)
    if not math.isfinite(residual) or residual < 0:
        raise ValueError("quality residual must be finite and nonnegative")
    return block


def measure_terminal_metrics(task, terminal, seed, reads=256):
    """Measure the exact program selected by the constructor's COMMIT.

    Programming/receipt/sampler errors are surfaced, not converted into easy labels.
    No chains are grown, repaired, replaced or initialized in this function.
    Residual, hit rate and chain-break frequency describe the *same* sample
    block, so reporting another metric does not silently spend extra reads.
    """
    reference = task.ground_energy
    if reference is None:
        raise ValueError("residual labels require a certified ground energy; inference does not")
    block = _sample_terminal(task, terminal, seed, reads, reference)
    residual = float(block.mean_residual)
    if (isinstance(block.hits, bool) or not isinstance(block.hits, Integral)
            or not 0 <= block.hits <= reads):
        raise ValueError("quality hits must be an integer within the read count")
    broken_fraction = float(block.broken_fraction)
    if not math.isfinite(broken_fraction) or not 0 <= broken_fraction <= 1:
        raise ValueError("chain-break fraction must be finite and lie in [0, 1]")
    return {"residual": residual, "p_solve": int(block.hits) / reads,
            "hits": int(block.hits), "reads": int(block.reads),
            "strength_index": int(block.strength_index), "broken_fraction": broken_fraction}


def measure_terminal(task, terminal, seed, reads=256):
    """Backward-compatible scalar residual from one fully validated read block."""
    return measure_terminal_metrics(task, terminal, seed, reads)["residual"]


def measure_selection_energy(task, terminal, seed, reads=256):
    """Deployment selection score requiring only public logical coefficients.

    For S=sum|h|+sum|J|, every decoded energy satisfies E >= -S. Sampling with
    this public lower bound as the backend reference returns the affine score
    (mean(E)+S)/max(S,1e-9), which orders candidates by mean decoded energy
    within an instance. It is NOT a residual to a certified optimum. In
    particular, hits against -S are discarded and never called solve probability.
    Ground energy, witnesses and other evaluator-only task fields are not read.
    Selection and final assessment must use separate seeds/read blocks.
    """
    coefficients = [float(x) for x in (*task.problem.h.values(), *task.problem.j.values())]
    if not all(math.isfinite(x) for x in coefficients):
        raise ValueError("selection requires finite public logical coefficients")
    mass = sum(abs(x) for x in coefficients)
    if not math.isfinite(mass):
        raise ValueError("public coefficient mass must be finite")
    block = _sample_terminal(task, terminal, seed, reads, -mass)
    return float(block.mean_residual)


def quality_utility(task, residual):
    """Affine per-instance quality reward in [0.5,1]; invalid episodes receive 0.

    Ising energies lie in [-S,S], where S=sum|h|+sum|J|. Thus the existing normalized
    residual is at most 2S/max(|E_ground|,1e-9). Scaling is fixed for the instance;
    expected utility therefore orders valid policies by expected residual. It does
    not assert lexicographic feasibility in expectation across failed episodes.

    The valid-quality slope is -1/(2*bound). Small residual differences can thus
    produce small policy advantages, especially relative to valid/invalid credit.
    Fixed scaling of the whole policy gradient changes update units, not that
    relative trade-off or the sampler's signal-to-noise ratio. ``None`` denotes
    invalidity here; a valid COMMIT with a missing label must be rejected by the
    rollout/loss contract rather than passed to this low-level helper.
    """
    if residual is None:
        return 0.0
    residual = float(residual)
    if not math.isfinite(residual) or residual < 0:
        raise ValueError("quality residual must be finite and nonnegative")
    if task.ground_energy is None or not math.isfinite(task.ground_energy):
        raise ValueError("quality labels require a finite ground-energy reference")
    coefficients = [*task.problem.h.values(), *task.problem.j.values()]
    if not all(math.isfinite(float(x)) for x in coefficients):
        raise ValueError("logical coefficients must be finite")
    scale = sum(abs(float(x)) for x in coefficients)
    bound = 2 * scale / max(abs(float(task.ground_energy)), 1e-9)
    if bound == 0:
        if residual > 1e-9:
            raise ValueError("nonzero residual for a constant-zero Hamiltonian")
        return 1.0
    fraction = residual / bound
    if fraction > 1 + 1e-7:
        raise ValueError("residual exceeds the Ising coefficient bound")
    return 1.0 - 0.5 * min(1.0, fraction)
