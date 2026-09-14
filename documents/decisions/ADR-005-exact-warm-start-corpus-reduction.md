# ADR-005: Exact corpus reduction for supervised policy warm start

## Status

Accepted.

## Date

2026-09-13

## Context

The supervised warm start combines four corpus objectives: resolved-action ranking, state utility,
per-action frozen-continuation value, and protected-COMMIT-relative value. The registered equations
normalize these terms by corpus-level masses. The implementation previously recomputed each
denominator inside every memory minibatch and applied one AdamW step per minibatch.

That behavior changed the objective. A minibatch containing one resolved ranking or delta row had
the same component weight as a minibatch containing many such rows. The final short minibatch was
also overweighted. For the utility critic, averaging minibatch ratios is not the corpus ratio. For
example, singleton rows with effective counts 1 and 100 and squared errors 0 and 1 yield an average
minibatch loss of 0.5, while the registered corpus loss is 100/101. Shuffling does not make a ratio
estimator exactly unbiased.

The scientific requirement is stronger than obtaining a finite training loss. At fixed model
parameters, splitting or permuting the corpus into memory minibatches must reconstruct the same
loss and gradient, up to floating-point summation order.

## Decision

Compute one validated `WarmStartCorpusDenominators` object after authenticated quality rows are
loaded and before training begins. It contains:

- `actor_ranking_records`, the number of rows with a nontrivial resolved best subset;
- `utility_effective_count`, the sum of state-utility confidence weights;
- `action_value_records`, the number of rows carrying action-value supervision;
- `commit_delta_records`, the number of rows with a protected COMMIT and at least one comparison.

For a memory minibatch (M), return additive contributions to the fixed corpus objective:

```text
L_rank(M)  = sum_{i in M, resolved} r_i / R
L_value(M) = lambda_V sum_{i in M} c_i (V_i - y_i)^2 / C
L_Q(M)     = lambda_Q sum_{i in M, Q-labelled} q_i / Q
L_delta(M) = lambda_delta sum_{i in M, delta-labelled} d_i / D
```

Here (R,C,Q,D) are the four frozen corpus denominators. Within one state,
(q_i) remains the `C/pi`-weighted binary cross-entropy normalized by the sum of those action
weights. Likewise, (d_i) remains the weighted mean squared error between predicted and target
COMMIT-relative differences. Action evidence is normalized within a state, then states are weighted
equally in the corpus action-value term.

Treat a minibatch only as a GPU memory partition. At each epoch:

1. clear gradients once;
2. run one minibatch forward and backward at a time, immediately freeing its graph;
3. verify that the epoch census reconstructs every corpus denominator;
4. check finite losses and gradients;
5. clip once and apply one optimizer step.

The model must remain fixed across all minibatches in the epoch. The reference IF-Core forward has
no dropout, batch normalization, cross-sample reduction, stochastic routing, or mutable forward
state. A train-mode regression test locks loss and gradient invariance under a shuffled 2+1
partition. Introducing any such operation requires a new reduction contract.

The loss identity is bumped from v3 to v4. The reduction identity is
`exact-full-corpus-gradient-accumulation-v1`. The four denominators are serialized inside the signed
experiment contract and checked before transfer. Warm-start history is renamed and bumped to
`isingfold.warm-start-history` v4; it records corpus losses rather than means of minibatch losses.

## Alternatives considered

### Keep locally normalized minibatch losses

This is fast and produces many AdamW updates, but it optimizes a batch-composition-dependent loss.
It was rejected because sparse ranking and delta rows, variable utility evidence, and the short tail
receive incorrect weights.

### Horvitz-Thompson-scaled stochastic minibatches

Uniform fixed-size minibatches can produce an unbiased gradient estimator by scaling each additive
corpus contribution by the inverse row inclusion probability. The final short batch needs its actual
inclusion probability or explicit resampling. This retains frequent optimizer updates and may train
faster, but a realized epoch remains order-dependent and noisier. It was rejected for the primary
warm-start contract in favor of an exactly auditable corpus gradient. It remains a valid future,
separately versioned optimization mode.

### Materialize one full GPU batch

This gives the same mathematical gradient but can exceed accelerator memory on heterogeneous graph
states. It was rejected because sequential backward over memory minibatches provides the same fixed
gradient without retaining all computation graphs.

## Consequences

- Short-tail size and sparse-label density no longer change component weights.
- Warm-start loss and gradients are invariant to memory partition and record permutation within
  numerical tolerance.
- GPU memory remains bounded by the configured minibatch, not corpus size.
- There is one AdamW update per corpus pass. The former 20-epoch schedule therefore becomes only 20
  updates and cannot be interpreted as equivalent to the old training run.
- The publication grid must freeze a new full-corpus optimizer-step budget before evaluation. The
  accepted main-run budget is 200 steps; checkpoints at earlier fixed steps are diagnostics, not a
  test-selected stopping rule.
- Distributed data parallel training would all-reduce every microbatch unless accumulation uses
  `no_sync`; the current single-device-per-cell launch avoids that overhead.
