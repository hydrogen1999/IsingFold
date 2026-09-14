# ADR-007: Direct Q-mu Warm-Supervision Control

## Context

The registered warm start combines four signals: tied-best action ranking, state utility,
per-action continuation quality, and the quality difference from protected COMMIT. A comparison
against PPO from scratch cannot identify which supervised signal caused an improvement. Removing
the action-quality head would also change parameter capacity and policy coupling, so it would not
isolate the value of direct counterfactual Q labels.

## Decision

Register a post-representation-selection diagnostic with two immutable loss profiles:

- `full-qmu-v4` uses weights `(rank=1, utility=0.5, action_value=1, commit_delta=0.5)`.
- `rank-value-only-control-v1` uses weights
  `(rank=1, utility=0.5, action_value=0, commit_delta=0)`.

Reuse the three authenticated full-loss IF-Core representation cells as treatments and train one
new control at each seed `1103`, `2207`, and `3301`. A pair must use the same IF-Core parameter
schema, bounded-centered action-quality prior, corpus, labels, exact preflight, selector,
normalizer, device type, source code, initialization seed, record order, optimizer, minibatch, and
200 full-corpus updates. Any implementation drift invalidates reuse and requires the treatment to
be rerun.

The registry and runner expose no scientific hyperparameter override. Each control receipt binds
its source treatment checkpoint and the sealed representation-selection receipt. Production PPO
transfer accepts only `full-qmu-v4` with no diagnostic binding.

## Consequences

The paired difference estimates the incremental contribution of direct Q-mu and COMMIT-delta
labels beyond ranking and state-value supervision. It does not compare quality against structure.
The control retains the action-quality tower, and ranking gradients can still reach it through the
bounded policy prior. Results are post-selection diagnostics on validation and cannot change any
main-grid choice or open the test partition.

The study adds three trainings rather than six, but the three existing treatments are reusable
only if their signed runtime and data identities match the controls exactly. All three seed pairs
must be reported without seed or checkpoint selection.
