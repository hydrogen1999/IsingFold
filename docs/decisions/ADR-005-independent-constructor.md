# ADR-005: the method is an independent constructive RL embedder

Status: implements the author's explicit correction. Supersedes ADR-004 as the primary
method direction; historical hybrid experiments remain labelled comparison probes.

## Decision and boundaries

The learned method receives the logical Ising instance, active host and public budgets.
It starts with empty chains and an empty archive. The policy selects every placement,
route, rewrite/growth, restart and COMMIT action. A graph algorithm may materialize a
bounded path or connected replacement as an action candidate. It must not finish the
whole embedding on the policy's behalf. Minorminer exists only in a separate comparison
arm and is never called to repair, complete or rescue a policy episode.

Qubits are a capacity constraint. No witness-derived deployment cap, chain-length penalty
or per-qubit reward penalty is allowed in this trainer. Witnesses are not consulted by
the constructor, its features, or its validation budget. An archive may contain only
embeddings produced during this episode. A selected COMMIT is required; finding a valid
workspace does not force termination or silently return an unselected archive entry.

The current candidate-generator defaults elsewhere remain unchanged. This constructor
opts into growth even when all logical demands are already satisfied, under a distinct
context-version suffix. This creates the action needed to learn whether additional
qubits help quality. It does not assume that growth is beneficial.

## Implementation map

| Component | File | Responsibility |
|---|---|---|
| Trainer | `probes/train_constructor_rl.py` | Lineage split, on-policy updates, schema-bound checkpoints |
| Transitions | `probes/constructor_rollout.py` | Empty environment, actor-selected legal macro-actions and terminal handling |
| Observations | `probes/constructor_features.py` | Current/candidate-successor hardware and logical summaries |
| Actor/critic | `probes/layout_policy.py::LayoutActorCritic` | Contextual candidate-set policy and action-independent state baseline |
| Loss | `probes/constructor_learning.py` | Monte Carlo actor-critic or potential-corrected leave-one-out |
| Quality label | `probes/constructor_objective.py` | Measure the exact selected terminal program |
| Evaluation | `probes/constructor_protocol.py` | Shared proposal/selection deadline and independent assessment |
| Comparison | `probes/constructor_baseline.py` | Unhinted minorminer arm only |

The macro-action generator and independent validity gate remain in `src/isingfold/rl`.
No changes to hardware programming, strength ratios, registered annealing schedule or
decoder are part of this change.

## State, actions and output validity

The authoritative environment state contains chain assignments, the archive, work spent
and remaining, restart allowance and proposal randomness. Legal actions come from the
environment's materialized candidate batch. The actor receives only legal rows, and
the selected row's payload is applied with its bound state/support fingerprint.

PLACE assigns a seed/connected branch set. ROUTE assigns the transit qubits to named
chains to satisfy a logical edge. GROW and SHRINK are explicit REWRITE_ONE outcomes;
other rewrites and repairs can replace larger sets. RESTART consumes a token and clears
the workspace. COMMIT selects an independently revalidated archive embedding produced
during this run. STOP terminates without output when offered by the environment.

Default proposal quotas are place16, route16, grow8, shrink4, rewrite8, repair8 and
restart4; total state-changing support remains bounded by the core action capacity.
Quotas and deterministic path heuristics still restrict support. The policy chooses
among these candidates; this is not unconstrained generation of arbitrary chains.

Validity requires every variable to own a nonempty connected branch set, disjoint
ownership for the returned embedding, physical realization of every logical edge,
the declared qubit cap and compiler/selector admissibility. The existing environment
is the authority for relaxed internal search states and legal masks. An inconclusive
heuristic is not claimed to certify future completion.

Reset, candidate/feature generation, actor inference, transitions and final compilation
count toward the episode's monotonic deadline. Late results are rejected. Hitting an
external horizon or deadline without an on-time selected COMMIT is a failure, including
when an unselected valid archive exists. Terminal labels have separate training cost.

## Observations and network

The new `constructor-v1` observation has 230 channels. It includes legacy 35 channels,
all eight opcodes, growth/shrink/rewrite/restore indicators, local before/after residual
capacity and coupling features, full-state before/after/delta structural summaries,
action additions/removals, work remaining and reserved, horizon and restart availability.
COMMIT observes the referenced archive entry, not the current workspace by accident.

Structural summaries expose component space, chain connectivity, internal bridges and
cycles, realized demands, contacts and declared load proxies. Native-coordinate channels
are topology-index directions, not Euclidean compass directions. Local pooling uses at
most eight deterministic anchors. Contact/field load summaries are observable proxies,
not measured chain-break rates or a substitute for the physical compiler.

The shared encoder maps each candidate row to an embedding; mean/max candidate pooling
and support size supply context to the actor and scalar value head. The critic is
computed before sampling an action. It predicts return, not COMMIT quality or action Q.
This is a candidate-set network, not a GNN, and its finite summary is not claimed to be
a lossless Markov observation. No witness, exact ground energy or sampled quality label
is supplied to the actor. Raw qubit labels do not become supervised targets.

## Objective, shaping and loss

The default stage is `quality`. At a valid, measured COMMIT the registered evaluator
uses the exact selected program, majority decoding, 200 sweeps and beta_range=(0.1,2.0).
It reports the existing normalized residual

    r = E[decoded energy - ground energy] / max(|ground energy|, 1e-9).

Let S=sum|h|+sum|J| and B=2S/max(|ground energy|,1e-9). The training reward is

    R = 1 - 0.5 r/B        for valid measured output, S>0;
    R = 1                 for a zero Hamiltonian and zero residual;
    R = 0                 for failure or explicitly missing measurement.

The fixed per-instance affine transform maps valid rewards into [0.5,1] and preserves
expected residual ordering among valid policies on that instance. It reweights tasks
relative to raw residuals, so evaluation reports raw residual and coverage separately.
It does not give a lexicographic feasibility guarantee in expectation. Invalid sampler
receipts or numerical errors raise instead of being silently turned into training labels.
Certified ground energy is read only by the reward/assessment backend. Inference with
`evaluate_reward=False` needs no such label and makes no annealing call.

An optional explicit `feasibility` curriculum uses valid reward 1 and failed reward
0.25 times placement/demand progress. It is a different objective and must be labelled;
it cannot establish the downstream-quality claim. It has no qubit penalty.

Optional potential shaping uses gamma=1, state-only progress Phi(s), and
F(s,a,s')=Phi(s')-Phi(s). All absorbing endings, including timeouts, have Phi=0.
The empty initial state also has Phi=0, so the trajectory reward telescopes exactly
to R. Returns to go are G_t=R-Phi(s_t). In particular a dead-end trajectory cannot
retain free shaping credit. With a perfect potential-shifted LOO baseline, this shaping
does not create a new policy-gradient signal by itself.

For a fresh rollout, the default loss is

    A_t = stop_gradient(G_t - V(s_t))
    L_actor = -mean_episodes sum_t A_t log pi(a_t | s_t)
    L_value = mean_steps Huber(V(s_t), G_t)
    L = L_actor + c_v L_value - c_H mean_steps H(pi_t)/log(|A_t|).

One-action entropy is zero. COMMIT, STOP and selected RESTART participate in the same
credit assignment. For leave-one-out, the baseline at time t is the mean BASE return
of the other episodes on this task minus Phi(s_t); it excludes the current trajectory.
Actor credit is summed, not divided by trajectory length. One optimizer update consumes
each fresh batch, with gradient clipping. This is Monte Carlo actor-critic, not PPO.

## Training and evaluation protocol

Split whole lineages before training. Validate with separate deterministic RNG streams,
`torch.no_grad`, and no quality labels inside construction. Repeated policy episodes
produce complete candidate embeddings from empty. Selection measures candidates until
the shared deadline or read cap, then assesses the chosen output on a fresh block.
Minorminer, if enabled, independently starts unhinted and gets the same outer deadline,
read cap, decoder, strength policy, qubit budget and assessment stream. Its router time
is counted only in its own arm. Failed policy attempts never invoke that baseline.

Checkpoint selection is validation-only: validity, measured coverage, then lower policy
residual. Feasibility-stage selection uses validity. The initial checkpoint is saved even
with zero updates. Feature schemas bind checkpoints; the former 65-dimensional hybrid
checkpoint cannot silently initialize this 230-dimensional constructor.

The batch log records reward/advantage variation, critic diagnostics, entropy, gradient
norm, validity, measurement coverage, demand progress and episode length. Final test
lineages must be kept outside this tuning corpus. Repeated validation is not final-test
evidence. Report all failures; do not report only the successful-policy intersection.

## Execution

On the experiment host, activate the existing environment and run one stage at a time:

```bash
bash scripts/apollo/run_independent_constructor.sh "$CORPUS" runs/constructor/feas_s0 feasibility 0
bash scripts/apollo/run_independent_constructor.sh "$CORPUS" runs/constructor/quality_s0 quality 0 \
  --init runs/constructor/feas_s0.pt
```

For an isolated policy-only run add `--comparison none`; no minorminer arm is invoked.
Keep the same seed/holdout fraction between curriculum and quality stages. This preserves
the training/validation lineage split. Checkpoints retain cumulative training-lineage
provenance; initialization rejects any overlap with current validation. Legacy
checkpoints without verifiable provenance require explicit `--allow-unverified-init`
and their descendants remain marked unverified. Such validation is diagnostic, not
held-out evidence. The final quality run needs a measured-selection
minorminer comparison for the paper, multiple training seeds and untouched test data.
Legacy local-MLP ablation: `--actor local --features legacy --value-baseline loo`.

Do not begin a long quality run with a policy that never commits a valid embedding.
All failed quality rollouts receive 0; telescoping shaping cannot manufacture a quality
signal. First establish nonzero valid-COMMIT coverage using the explicit feasibility
curriculum on small tasks, then transfer the checkpoint with the same feature schema.
The log's `has_measured_quality_signal` exposes this prerequisite during fine-tuning.

## Evidence boundary

The focused regression gate passes 167 tests; its exact command and output are in
`results/audit/independent_constructor_unit.log`. Ruff, shell syntax and diff checks
also pass. Full unit-suite collection is blocked by the unavailable native
`lac_minorminer._core` extension.

Tests exercise actual graph-environment trajectories with minorminer calls and witness
access replaced by failures; actor-selected PLACE, ROUTE, growth, refinement and COMMIT
still succeed on toys. Separate tests cover post-valid growth, no automatic commit,
deadline failures, budget constraints, reward/gradient contracts and CLI checkpoints.
The tiny Monte Carlo learnability gate has a synthetic rewarded terminal action; it is
not an annealing-quality result. Full-scale training and A* competitiveness remain to
be established. The hybrid logs remain historical measurements of a different method.
