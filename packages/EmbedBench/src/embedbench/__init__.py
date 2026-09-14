"""EmbedBench: certified decision data and downstream-objective evaluation for minor embedders.

Three tracks:
  structural   decision samples with exact labels (feasible completion, then fewest qubits,
               then shortest longest chain) on planted witnesses in real hardware graphs;
  quality      chain-seam samples labelled by a classical annealing surrogate of the solve
               probability, with a two-stage fresh-seed protocol;
  end-to-end   evaluate any embedder (a callable from (logical graph, host graph) to chains)
               on planted-Ising and application instances by feasibility, qubits, longest
               chain and surrogate solve probability, against stock minorminer.
No dependency on the rest of this repository.
"""
from embedbench.embedding import Embedding, LogicalProblem, Node, Qubit
from embedbench.objective import INFEASIBLE, Outcome, compare, outcome_of
from embedbench.exact import all_completions, best_completion, value_of_action
from embedbench.inkdrop import MODES, InkDropConfig, PlantedEmbedding, PlantedGenerationError, ink_drop
from embedbench.planted_ising import PlantedIsing, PlantingError, frustrated_loops
from embedbench.structural import GenConfig, generate, host_graph
from embedbench.quality_chain import ChainConfig, generate_chain
from embedbench.handtests import MOTIFS, build_motif, certify_motif
from embedbench.evaluate import EvalConfig, evaluate_embedder, stock_minorminer, decision_top1

__version__ = "0.1.0"
__all__ = [n for n in dir() if not n.startswith("_")]
