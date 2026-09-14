# Specification: Quality-signal v2 for the IsingFold embedder

## Objective

Build a deployment-legal learned embedder whose design is fixed from pre-test evidence. The system
must maximize unconditional terminal IF-Q3-S0 solve probability while treating minor-embedding
validity, programming validity, work caps, and comparison fairness as hard contracts.

The change addresses label sparsity and archive selection. It does not redefine quality as low qubit
count, short chains, feasibility, or the frozen selector prediction. Actual decoded solve probability
at the selected strength remains the ultimate reward.

## Assumptions

1. The frozen strength selector is trained only from authorized non-test labels and is available at
   deployment.
2. The prepared-v4 train, validation, and test partitions and their immutable lineage identities do
   not change.
3. Persistent K=2 initializer banks bind only initialization and can be reused after a model or loss
   schema bump when their own LAC, context, and manifest contracts still verify.
4. No production quality-v8 outcome, validation model-selection outcome, or test outcome has been
   opened when the v2 registries are frozen.
5. Apollo runs direct processes. Goose runs publication work only through Slurm.

## Scientific contract

### Ultimate objective

For task `x`, initializer outcome `z`, policy randomness `u`, and evaluator randomness `e`, define

```text
J(theta) = E[ valid_return * hits_IF-Q3-S0 / reads_IF-Q3-S0 ].
```

Initializer failure contributes zero to the unconditional complete-system outcome. Conditional PPO
starts only after an authenticated successful initializer, but final reporting restores the complete
population denominator. Connectivity and programmability are enforced before COMMIT. Qubit count
and maximum chain length are explanatory or tie-break metrics, never replacements for `J(theta)`.

### Deployment information boundary

Allowed inputs include the logical problem, faulty active host, current overlapping branch sets,
full ownership, candidate payloads, archive entries, budgets, deterministic memory, compiled program
features, and frozen selector predictions if their diagnostic gate passes.

Forbidden inputs include ground energy, evaluator hits, future return, counterfactual winner labels,
test membership, and evaluator-derived archive scores.

### Primary comparison

The final learned system is compared on untouched test lineages with stock minorminer selected from
the preregistered validation-only tuning family. Both systems use the same active host, program
compiler, four strengths, frozen selector, evaluator, decoder, read budget, and matched online
wall-clock envelope. Superiority requires the paired lineage-level confidence interval for learned
minus stock unconditional utility to exclude zero. Valid-return noninferiority must also pass.

## Architecture

### Stable components

- Native C++ proposal, routing, repair, overlap, validation, and exact work accounting.
- Fully materialized legal action support with at most 64 state-changing actions, eight COMMIT
  actions, and one STOP action.
- IF-MLP, IF-Dual, and IF-Core representation families.
- Masked categorical actor, state utility critic, optional construction failure critic, and PPO.
- Frozen IF-Q3-S0 graph strength selector.
- Protected valid initializer in improvement mode.

### Quality-v8 supervision

For a state `o`, include the protected COMMIT action `a0` and sample at most seven actions from all
other legal actions without replacement. The remaining pool includes state-changing actions, other
COMMIT actions, and STOP when the operating mode permits it. This gives every legal action positive
inclusion probability. Store each action's inclusion probability `pi(a|o)`.
For continuation policy `mu`, estimate

```text
Q_mu(o,a) = mean_m R_terminal(o, a, mu; m),        R_terminal in [0,1]
A_commit(o,a) = Q_mu(o,a) - Q_mu(o,a0).
```

Every retained action contributes to the action-value head. Every state contributes to the state
critic through the registered Horvitz-Thompson state-value estimator when coverage permits. A hard
tied-best actor target is optional and active only when the registered simultaneous comparison
resolves.

The action-value tower consumes the same 512-dimensional action-state context as the actor residual:

```text
q_logit(o,a) = MLP_Q([v_a, z_s, c_K, v_a * z_s])
q_hat(o,a)   = sigmoid(q_logit(o,a))
quality_prior(o,a) = 2 * q_hat(o,a) - 1
policy_logit = MLP_pi([v_a, z_s, c_K, v_a * z_s]) + quality_prior(o,a)
```

The main grid fixes the centered bounded transform, so the supervised quality estimate influences
the initial policy without contributing more than one logit unit in either direction. It is not a
mask and receives no evaluator feature at deployment. The grid stores this complete coupling
contract and forbids per-cell overrides. The registered post-selection ablation compares
auxiliary-only, centered bounded, clipped-logit, and detached centered-bounded coupling under the
same data and seed schedule. Raw-logit coupling is diagnostic-only because it has no finite output
bound. Under the main mode, the rank loss, PPO actor loss, and entropy loss differentiate through
both the actor residual and quality prior. If PPO runs without an auxiliary `Q_mu` loss, the bounded
output becomes an on-policy latent feature and is no longer interpreted as a calibrated estimate of
the frozen-continuation value. After PPO, only the sum of the actor residual and transformed quality
prior is behaviorally identifiable.

The primary warm-start loss is

```text
L_warm = lambda_q * L_Q
       + lambda_delta * L_commit_delta
       + lambda_v * L_state_value
       + lambda_rank * I_resolved * L_tied_best.
```

The registered initial coefficients are `lambda_q=1`, `lambda_delta=0.5`, `lambda_v=0.5`, and
`lambda_rank=1`. Action regression is normalized within each state using weights
`continuation_count / inclusion_probability`, then averaged equally across states. This prevents a
state with many legal candidates from silently dominating the representation fit.

The full authenticated corpus fixes the ranking-row count, utility confidence mass, action-value
row count, and protected-COMMIT delta-row count before optimization. Memory minibatches contribute
additive pieces divided by those global denominators. Gradients accumulate over the complete corpus,
then one clipped AdamW update is applied. The paper grid fixes 200 full-corpus updates and records the
exact census and summed loss decomposition after every update. A short final memory minibatch cannot
receive the weight of a full minibatch.

### Post-selection Q-label causal control

The main warm-versus-scratch comparison does not by itself identify the contribution of
counterfactual `Q_mu` labels. Warm start also supplies tied-best actor ranking and state-utility
calibration. Register a separate post-selection diagnostic with two paired loss profiles:

- `full-qmu-v4`: `lambda_q=1`, `lambda_delta=0.5`, `lambda_v=0.5`,
  `lambda_rank=1`;
- `rank-value-only-control-v1`: `lambda_q=0`, `lambda_delta=0`, `lambda_v=0.5`,
  `lambda_rank=1`.

Both profiles instantiate the same IF-Core parameters, including the action-quality tower, use the
same bounded-centered policy coupling, authenticated quality corpus, initializer bank, exact
corpus denominators, seeds `[1103,2207,3301]`, minibatch 32, AdamW settings, and 200 full-corpus
updates. The three authenticated full-loss IF-Core representation cells are the treatments. The
separate registry adds exactly three control trainings, one for each treatment seed. Paired cells
with the same seed begin from identical parameter and optimizer initialization and see the same
record permutation. This reuse is valid only when the treatment and control receipts bind the same
runtime implementation digest. If the warm-loss implementation changes after a treatment was run,
that treatment must be rerun. The control still permits rank gradients to reach the quality tower
through the bounded policy prior; it removes supervision from `Q_mu` targets rather than removing
model capacity or policy coupling.

This diagnostic runs only after the representation-selection receipt is sealed. Its result cannot
revise the main representation family, RL-value configuration, final system, or test protocol. The
registered comparison is full `Q_mu` supervision minus rank-plus-value control, aggregated first by
equal training seed and then by immutable base lineage on the existing validation protocol. It is
evidence about the causal value of Q labels, not a new model-selection stage. A control checkpoint
must carry a distinct signed loss-profile contract and must be rejected by production warm-start
transfer, which accepts only `full-qmu-v4`.

The executable registry is `configs/quality_warm_control_hybrid_v1.json`. It fixes IF-Core,
`bounded-centered-v1`, the three source treatment cell IDs, all loss weights, data-order contract,
training budget, and claim scope. `warm-quality-control-cell` accepts only a registry index and
artifact paths. It exposes no model, seed, coefficient, epoch, minibatch, optimizer, or partition
override. The control may be launched directly on Apollo or as a three-element Slurm array on
Goose. No control result exists until all three authenticated cell receipts and their paired
validation reports have been produced.

`L_Q` uses soft-target binary cross entropy on the quality logits with count-aware weights. It is a
proper scoring surrogate for the bounded mean continuation reward. It is not called a binomial
likelihood because one continuation reward is itself a decoded solve fraction. With a fixed
registered continuation count, inverse propensity corrects the protected-anchor sampling imbalance.
`L_commit_delta` compares bounded action and anchor predictions from the same state. It removes an
action-invariant state offset but is auxiliary evidence, not an additional independent continuation
sample. Exact coefficients, caps, and normalization are fixed in the registry before fitting. PPO
then optimizes the unchanged terminal reward. No offline row enters the PPO importance-ratio buffer.

### Evidence-gated selector-aware archive

Profile I retains FIFO until a cross-embedding diagnostic passes. The optional successor profile
stores detached `selector_max_p_solve` and `selector_strength_index` with each valid archive entry.
The protected slot remains first and immutable. Mutable entries are selected by frozen selector
score with deterministic structural and digest tie-breaks. All policy families receive identical
archive inputs and rules.

This profile must be trained and evaluated under newly generated continuation labels because archive
retention changes the MDP. It cannot reuse a quality label whose continuation used FIFO.

## Data and estimator design

### Action sampling

- Improvement with a legal protected COMMIT: inclusion probability 1 for that anchor.
- Non-anchor legal pool of size `M`: sample `min(7,M)` uniformly without replacement, with inclusion
  probability `min(1,7/M)` when `M>0`; if `M=0`, retain only the anchor.
- No protected COMMIT: sample up to eight actions uniformly from the full legal support and mark the
  row ineligible for the COMMIT-relative loss; do not substitute an arbitrary anchor silently.
- Record full legal support, sampled action identities, propensities, and sampling seed receipt.

The protected action is the unique legal `COMMIT` whose bound candidate references archive slot
zero. The non-anchor pool must contain every other legal action, including other `COMMIT` actions
and `STOP`. The draw is a deterministic function of the registered seed and exact state fingerprint,
but it is made before any continuation outcome is observed. Generation and authentication call the
same sampler. Every legal action has positive first-order inclusion probability. Illegal actions are
never sampled. The production profile uses one fixed registered continuation count for every
evaluated action in a row. Outcome-dependent stopping requires a new estimator and registry.

### State-value target

Let `N` be the exact number of legal actions, `E` the evaluated subset, `pi_a` the first-order
inclusion probability, and `q_a` the unbiased continuation mean. Define `w_a=1/(N*pi_a)`. The critic
target and its registered confidence weight are

```text
V_hat = sum_{a in E} w_a q_a
n_eff = 1 / [sum_{a in E} w_a^2 / C_a
             + sum_{a in E} (1-pi_a) w_a^2].
```

Both registered fixed-size samplers satisfy `sum_{a in E} w_a=1`, including the mixed-propensity
protected-COMMIT design. Therefore `V_hat` is unbiased for a uniformly drawn first action followed
by `mu`, and its realized value remains in `[0,1]`. The effective count is a conservative,
preregistered variance proxy. It is not a literal sample count. Full coverage with a common count
`C` gives `n_eff=N*C`; partial coverage leaves a nonzero uncertainty term even as continuation
counts grow. This formula requires independent sibling continuation streams. A paired-stream design
must register its covariance-aware estimator separately.

### Continuation randomness

Independent sibling streams remain valid but may be inefficient. A paired-stream successor must
bind one row-level stream index across sibling actions, record pairing IDs, and use a paired
estimator. Seed reuse under an independent-stream receipt is an integrity failure.

### Uncertainty

Store per-action continuation count, mean, sample variance, evaluator reads, and complete receipts.
Select continuation count on train only from a preregistered nested pilot. The selection criterion
targets action-value precision, not an arbitrary number of hard winners. Hard-rank resolution is
reported as an auxiliary diagnostic.

### Cross-embedding selector diagnostic

Use multiple distinct valid embeddings from the same immutable opportunity. Split evaluator reads
so the oracle choice and regret estimate do not reuse the same random block. Aggregate by immutable
base lineage. Publish raw comparable groups, exclusion reasons, rank metrics, retention simulation,
confidence intervals, runtime identity, selector identity, and a record digest.

## Commands

Current commands that must remain green:

```bash
.venv/bin/pytest -q tests/unit tests/integration
.venv/bin/ruff check src tests
bash -n scripts/*.sh scripts/*.sbatch
```

Proposed quality-v8 workflow after implementation:

```bash
python -I -m isingfold.rl.cli plan-quality-v8 \
  --corpus runs/prepared_v4 \
  --selector runs/selector_v4 \
  --initializer-bank runs/initializer_banks/seed-0 \
  --expected-initializer-bank-manifest-sha256 "$BANK_SHA256" \
  --config configs/quality_signal_hybrid_v1.json \
  --out runs/quality_v8_plan

python -I -m isingfold.rl.cli run-quality-v8-shard \
  --plan runs/quality_v8_plan/plan.json \
  --stage-index 0 \
  --shard-index 0 \
  --out runs/quality_v8_shards/stage-0/shard-0

python -I -m isingfold.rl.cli merge-quality-v8 \
  --plan runs/quality_v8_plan/plan.json \
  --pins runs/quality_v8_shards/pins.json \
  --out runs/quality_v8

python -I -m isingfold.rl.cli audit-selector-cross-embedding \
  --corpus runs/prepared_v4 \
  --selector runs/selector_v4 \
  --quality runs/quality_v8 \
  --partition validation \
  --config configs/selector_cross_embedding_v1.json \
  --out runs/selector_cross_embedding_v1
```

The command names above are acceptance targets. Publication launchers must use the pinned runtime,
absolute artifact paths, external SHA-256 pins, direct Apollo execution, and Slurm execution on
Goose.

## Project structure

```text
configs/                         frozen experiment and estimator registries
src/isingfold/rl/                environment, actor-critic, PPO, selector, evaluation
src/isingfold/rl/data/           quality plans, shards, merge, preflight, trust records
tests/unit/                      schema, estimator, loss, replay, and launcher contracts
tests/integration/               complete small-corpus workflows
scripts/                         Apollo direct and Goose Slurm publication launchers
documents/                       normative architecture, runbook, specs, and ADRs
runs/                            generated artifacts only; never source of hidden configuration
```

## Code style

Typed immutable records carry scientific identities. Validation is fail closed and canonical
digests exclude their own digest field.

```python
@dataclass(frozen=True)
class SampledActionValue:
    action_key: str
    inclusion_probability: float
    continuation_count: int
    q_mu: float
    sample_variance: float

    def __post_init__(self) -> None:
        if not 0.0 < self.inclusion_probability <= 1.0:
            raise ValueError("action inclusion probability must lie in (0, 1]")
        if not 0.0 <= self.q_mu <= 1.0:
            raise ValueError("bounded action value must lie in [0, 1]")
```

## Testing strategy

- Unit tests prove sampling propensities, COMMIT anchoring, Horvitz-Thompson targets, count-aware
  weights, paired seed domains,
  schema rejection, action-value masks, and archive tie-breaks.
- Gradient tests prove that unresolved rows update the state critic and action-value head but do not
  enter the tied-best actor term.
- Replay tests prove exact candidate, restart-cache, state, and support identities.
- Integration tests execute a complete tiny plan, shard, merge, warm start, PPO update, checkpoint,
  resume, and evaluation.
- Leakage tests replace ground/test targets and prove tensorized observations and deployment actions
  are unchanged.
- Runtime tests enforce CUDA for neural training, direct Apollo execution, and Slurm-only Goose
  execution.
- Full tests, Ruff, shell syntax checks, native C++ tests, and LaTeX builds run before source freeze.

## Boundaries

### Always

- Preserve the ultimate IF-Q3-S0 reward and hard validity rules.
- Record every stochastic domain, propensity, count, exclusion, and artifact digest.
- Aggregate uncertainty by immutable base lineage.
- Compare under the same frozen selector and evaluator.
- Treat diagnostic outputs as non-publication evidence unless their capability receipt authorizes
  the next stage.

### Require a new registered profile

- Changing archive retention, action support, continuation policy, candidate sampling, observation
  dimensions, training loss, seed schedule, or model-selection rule.
- Adding selector scores to actor inputs.
- Adding candidate self-attention or increasing GNN capacity.

### Never

- Use test outcomes for training, hyperparameter selection, archive ranking, or early stopping.
- Use evaluator feedback during deployment search.
- Treat selector predictions as measured terminal rewards.
- label unevaluated actions as inferior.
- continue a production run after a digest, replay, work-ledger, or validity mismatch.

## Success criteria

1. Unresolved authenticated rows train the state critic without training the tied-best actor term.
2. Quality-v8 emits usable action-value targets for every sampled action and verifies all inclusion
   probabilities and continuation receipts.
3. The cross-embedding selector diagnostic is immutable, lineage-aggregated, and test-isolated.
4. Selector-aware retention is unavailable unless its registered gate passes.
5. Every publication launcher assigns each grid cell exactly once on its registered host and device.
6. All focused and full verification commands pass under the frozen source.
7. No superiority claim is made until the sealed paired test interval clears zero and valid-return
   noninferiority passes.

## Open questions resolved by the pilot

- Whether the frozen selector ranks different embeddings well enough to improve top-seven
  retention over FIFO.
- Which continuation count reaches the registered action-value precision target.
- Whether paired sibling streams reduce variance on the actual continuation kernel.
- Whether IF-Core improves validation utility after stronger supervision, before testing any larger
  architecture.
