# Design note: a candidate-conditioned heterogeneous network on the programmed Ising (2026-09-09)

## What the decision looks like
A state is a valid or partial embedding: logical variables with couplings (h, J), hardware
qubits with couplers, and a mapping (chains). A candidate action replaces the focus
variable's chain. The registered terminal label is
`V(e) = max_{f in default_strength_grid(problem, 4)} p_solve(e; f)`: at each of exactly four
strengths, every chain coupler is programmed at `f`, every logical coupling is placed on one
contact coupler, and the problem is rescaled before annealing and majority decoding. This
release target is not a strength-conditioned response. Quality therefore depends on (a) the
programmed coefficients around the candidate, (b) how chain breaks under rescaling and
field splitting, (c) how the candidate changes contacts for its logical neighbours, and
(d) the rest of the embedding, which is fixed.

## What the current scorers see
Qubit-level structural features on the hardware window, a few coupling summaries, a mean
over the candidate's qubits. They do not see the full logical Hamiltonian, do not represent
the logical variables as nodes, and score candidates independently.

## The variant
1. **Full Hamiltonian context**: every backbone receives the versioned, label-free global
   context derived from all of `problem.h` and `problem.J`. It contains signed and magnitude
   coefficient statistics plus logical graph size, density, degree, component, cycle, and
   frustration-like summaries. It must not consume `e0`, labels, or baseline indices.
2. **Heterogeneous graph**: node types {qubit, variable}. Edge types are hardware coupler,
   nonzero logical coupling (feature: J), and membership (qubit in chain of variable).
   Variable nodes carry h, degree, chain size, "is focus", and "is frozen neighbour of
   focus". Qubit nodes carry the existing structural features. All logical variables and
   nonzero logical couplings remain present, not only the focus neighbourhood.
3. **Candidate conditioning**: for each candidate, the focus variable's membership edges are
   set to the candidate's qubits and the programmed coefficients are recomputed for the
   affected couplers only (cheap: intra-chain couplers of the candidate and its contacts).
   The rest of the graph is shared across candidates, so the batch of candidates is one
   graph with a small per-candidate delta.
4. **Message passing**: qubit->qubit over couplers with edge-feature gating (GINE-style),
   qubit->variable pooling over membership (mean and max), variable->qubit broadcast over
   membership, variable->variable over logical couplings. Three rounds.
5. **Physics prior as a feature, not a rule**: summarize the per-chain cut-load ratio across
   the four registered strengths rather than conditioning the terminal value model on one
   queried strength. Track A found that ratio added nothing on structural targets (L-31);
   here the target is the quality quantity it was derived to explain.
6. **Readout**: the focus variable node's state after passing, concatenated with the
   candidate's pooled qubit states and a cross-candidate attention over the candidate set
   (listwise context), then a scalar.
7. **Loss**: listwise softmax cross-entropy with soft targets proportional to exp(p/T) over
   the candidate set, each candidate weighted by its label reliability (stage-2 estimates
   weight 1, screening estimates weight 0.3); pairwise logistic kept as an ablation.

## Why it could beat the random order with evaluation
The scorer only has to identify, among the harness's proposals, the ones whose programmed
Hamiltonian is friendlier to decoding: strong contacts on high-|J| edges, low cut load,
short chains where the field is strong. Those are computable inputs the current scorer
does not receive.

## What is new here and what is not
Bipartite variable/constraint networks: Gasse et al. 2019 (MILP branching). Dual logical /
hardware GCNs: CHARME. Edge-featured passing: GINE. Listwise ranking losses: standard.
The combination targeted at an embedding decision, conditioned on candidate chains, with
programmed-Hamiltonian edge features and a chain-break prior, trained on certified
counterfactual labels, is not in the literature we found.

## Risks
The label noise floor (0.02 to 0.03 regret at 400 reads) may already be close to what any
scorer can reach; the gain must show as regret on the reliable set and as end-to-end p_solve
against the random order with evaluation at equal budget (the comparison it has so far
lost). Overfitting: 4k chain-seam states; the model must stay small (under 200k) and be
selected on validation with four optimization seeds. For the two-site screen, seeds 0 and 2
run sequentially through one direct, non-Slurm Apollo GPU process; seeds 1 and 3 run only
through the Goose Slurm array. This gives each architecture-objective configuration two
seeds per site.
