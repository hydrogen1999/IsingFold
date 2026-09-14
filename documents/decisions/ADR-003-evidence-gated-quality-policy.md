# ADR-003: Evidence-gated quality supervision and archive selection

## Status

Accepted for the pre-training pilot. Quality-aware archive retention remains gated and is not yet
part of the publication MDP.

## Date

2026-09-13

## Context

IsingFold optimizes unconditional terminal IF-Q3-S0 solve probability under exact validity and work
constraints. The first policy design uses a protected initializer, seven mutable FIFO archive slots,
a frozen strength selector, tied-best counterfactual ranking, a state utility critic, and PPO.

Two independent pre-training audits found four material risks.

1. With eight actions and familywise alpha 0.05, the current Hoeffding rule at 128 continuations has
   radius 0.150108. Eliminating one arm requires an empirical gap above approximately 0.300216.
   Earlier seam studies measured smaller spreads on a related, but not identical, target. This does
   not prove that the new study will fail, but it makes a hard-winner-only corpus unnecessarily
   risky.
2. The quality row already contains bounded per-action continuation estimates, yet the loader drops
   an unresolved row before training the state critic. Ranking uncertainty therefore removes valid
   critic supervision.
3. The frozen selector is used to choose strength and to rank the quality-aware stock baseline, but
   its cross-embedding ordering ability has not been established. Adding its score to the learned
   policy or replacing FIFO before that measurement would change the MDP on unsupported evidence.
4. The learned system can discover more valid embeddings than fit in its archive. FIFO can discard
   an early high-quality embedding, but a selector-ranked replacement is beneficial only if the
   selector orders different embeddings reliably.

The audit also rejected two tempting changes. A larger IF-Core is not currently justified because
model capacity is not the identified bottleneck. A constant dueling term in categorical actor logits
would cancel under softmax and cannot change the policy.

An older, non-publication M4 embedding-selector experiment provides motivation but not authorization
for the new profile. Across 180 host-instance pools, keeping the seven embeddings with the learned
score instead of the last seven in generation order improved the best measured held-out score by
0.0275 on average. Equal-instance bootstrap over 60 base instances gave a two-sided 95 percent
interval of approximately [0.0157, 0.0392]. That scorer, corpus, and endpoint are not the current
frozen strength selector and IF-Q3-S0 protocol, so these numbers justify the new diagnostic rather
than proving that selector-ranked retention will work.

## Decision

Keep the ultimate reward, validity predicates, action grammar, evaluator, and untouched test
protocol unchanged. Improve the training signal in evidence-ordered stages.

### Stage 1: preserve valid critic supervision

Every authenticated quality row contributes its count-aware state-utility target. The actor
tied-best loss remains masked when the row is unresolved. The warm-start loss and receipt identity
must change so an old checkpoint cannot be presented as a new one.

### Stage 2: migrate from winner-only labels to bounded action values

Before full quality generation, register quality-v8 with:

- one protected COMMIT anchor whenever improvement mode exposes it;
- up to seven uniformly sampled remaining legal actions, including non-protected COMMIT actions;
- the exact inclusion probability for every retained action;
- all bounded continuation rewards and effective counts;
- a per-action action-value target for every sampled action;
- a primary count-aware action-value regression loss;
- a COMMIT-relative auxiliary target `Q_mu(o,a) - Q_mu(o,COMMIT)`;
- the old confident tied-best loss only as an auxiliary when its interval resolves.

The protected anchor is the unique legal COMMIT bound to archive slot zero. It has inclusion
probability one. The remaining pool contains every other legal action, including other COMMIT
actions and STOP. Sampling is uniform without replacement, outcome-blind, and reproducible from the
registered seed and exact state fingerprint. If no protected anchor exists, sample up to eight
actions uniformly from the full legal support and disable the COMMIT-relative loss for that row.
Every legal action must have positive first-order inclusion probability.

Unevaluated actions are never labeled negative. Inverse-propensity terms correct the registered
action sampling design. For state-value initialization, use the Horvitz-Thompson weight
`w_a=1/(N*pi_a)`, where `N` is the exact legal-action count. The registered fixed-size samplers must
satisfy `sum_a w_a=1` in every realized row. The confidence weight is
`1/[sum_a w_a^2/C_a + sum_a (1-pi_a)w_a^2]`. It is a bounded-variance proxy, not a literal number of
independent observations. The production profile fixes one continuation count per evaluated action
within a row. Outcome-dependent stopping requires a new estimator. Weight clipping, if used for
numerical stability, must be declared in the quality-v8 registry rather than selected from
validation outcomes.

Common random numbers may be used across sibling actions only after the seed domain, pairing
identity, and paired estimator are versioned together. Reusing seeds while retaining an independent
Hoeffding interpretation is forbidden.

The action-quality logit enters the categorical policy logit with fixed coefficient one. Therefore
the tied-best loss and subsequent PPO actor and entropy losses update both the actor residual and the
quality-prior tower. Without continued action-value supervision, the bounded head is interpreted as
a calibrated frozen-continuation value only at warm start, not after PPO. Only the combined policy
logit is behaviorally identifiable after PPO.

### Stage 3: measure selector ordering before changing the MDP

Run a frozen, target-isolated cross-embedding diagnostic on train or validation opportunities. For
each comparable opportunity, score multiple valid embeddings by the maximum frozen selector
probability and evaluate their actual IF-Q3-S0 outcomes with independently split read blocks. Report:

- comparable opportunity and embedding counts;
- per-opportunity and aggregate Kendall or Spearman rank correlation;
- pairwise sign accuracy with ties stated explicitly;
- top-1 regret against the split-block oracle;
- simulated best-retained quality for protected-plus-FIFO and protected-plus-selector-top-k;
- confidence intervals resampled by immutable base lineage;
- runtime and selector calls.

The diagnostic cannot use the test partition, alter selector parameters, train the actor, or revise
the terminal objective. Selector-aware retention advances only if its preregistered lineage-level
lower confidence bound over FIFO is positive and its cross-embedding rank statistic is positive.
Otherwise FIFO remains the publication archive policy.

### Stage 4: optional quality-aware archive profile

If Stage 3 passes, define a new profile rather than silently changing Profile I.

- The protected initializer is never evicted.
- Each valid admitted embedding stores a detached selector maximum and selected strength index.
- Mutable retention keeps the seven best entries under the frozen selector, then lower qubit count,
  lower maximum chain length, canonical embedding digest, and admission order.
- The score is a deployment-visible feature and retention key, not the terminal reward.
- COMMIT independently revalidates and reruns the frozen selector.
- Selector inference is charged in the online work and wall-clock receipts.
- All representation families receive the same score fields and archive rule.
- Observation, state fingerprint, action certificate, checkpoint, and artifact schemas are bumped.
- Old quality rows are re-tensorized only when their exact states contain enough authenticated
  information. Otherwise they are regenerated. Test data is never used for this migration.

## Alternatives considered

### Enlarge the GNN immediately

The current IF-Core already has typed logical and hardware message passing, ownership fusion,
phase-aware factors, route tokens, archive tokens, and legal candidate-set pooling. There is no
evidence that width or depth is the current limiting factor. This option was rejected before the
first diagnostic grid.

### Replace FIFO immediately

The frozen selector has strength-selection evidence but no completed cross-embedding ranking
certificate. Immediate replacement could make archive retention worse and would invalidate current
counterfactual labels. This option was rejected.

### Keep only hard tied-best rows

This discards bounded action-value observations when confidence intervals overlap. It is statistically
wasteful and can leave the warm start without enough rows. This option was rejected as the primary
supervision path; resolved ranks remain a useful auxiliary.

### Use evaluator outcomes online to retain archive entries

This leaks the training or test evaluator into deployment decisions and makes the comparison with
stock minorminer invalid. It is forbidden.

### Add a state-only constant to actor logits

The constant cancels exactly in masked categorical normalization. It cannot improve action ranking
and is rejected.

## Consequences

- The immediate code fix increases critic coverage without changing the deployed policy.
- Quality-v8 can use every authenticated sampled action even if no hard winner resolves.
- A selector-aware archive remains a measured hypothesis, not an assumed advantage.
- Changing retention requires new quality labels because it changes continuation dynamics.
- The paper can separate three causal contributions: better supervision, sequential PPO, and
  quality-aware retention.
- No stage guarantees superiority. The claim is supported only by the sealed paired test confidence
  interval against the validation-selected stock baseline.
