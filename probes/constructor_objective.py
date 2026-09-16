"""Quality labels of policy-created terminal programs, with no completion solver.

Only the label backend reads a ground-energy reference. The actor and construction
environment never need it. Residuals use the registered majority decoder/schedule.
"""
import math

from _context import REGISTERED_BETA_RANGE


def measure_terminal(task, terminal, seed, reads=256):
    """Measure the exact program selected by the constructor's COMMIT.

    Programming/receipt/sampler errors are surfaced, not converted into easy labels.
    No chains are grown, repaired, replaced or initialized in this function.
    """
    from isingfold.rl.evaluator import sample_program

    if not terminal.returned_valid or terminal.embedding is None or terminal.selected_program is None:
        raise ValueError("quality measurement requires a validated COMMIT")
    if task.ground_energy is None:
        raise ValueError("residual labels require a certified ground energy; inference does not")
    if terminal.selected_index != terminal.selected_program.strength_index:
        raise ValueError("terminal strength/program mismatch")
    block = sample_program(terminal.selected_program, terminal.embedding, task.problem,
                           task.ground_energy, num_reads=reads, seed=int(seed), num_sweeps=200,
                           beta_range=REGISTERED_BETA_RANGE)
    if block.reads != reads or block.strength_index != terminal.selected_index:
        raise ValueError("quality evaluator violated the declared read/strength contract")
    residual = float(block.mean_residual)
    if not math.isfinite(residual) or residual < 0:
        raise ValueError("quality residual must be finite and nonnegative")
    return residual


def quality_utility(task, residual):
    """Affine per-instance quality reward in [0.5,1]; invalid episodes receive 0.

    Ising energies lie in [-S,S], where S=sum|h|+sum|J|. Thus the existing normalized
    residual is at most 2S/max(|E_ground|,1e-9). Scaling is fixed for the instance;
    expected utility therefore orders valid policies by expected residual. It does
    not assert lexicographic feasibility in expectation across failed episodes.
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
