# ADR-002: Nested continuation resolution, all-train labels, and sharded exact replay

## Context

Quality-v7 uses simultaneous Hoeffding intervals over continuation returns. With eight evaluated
actions and familywise alpha 0.05, no count at or below 11 can exclude an action even under the
largest possible empirical reward gap. The existing CLI nevertheless accepts a free continuation
count and defaults to 2. The runbook example also requests 128 lineages while requiring 128 resolved
lineages, leaving no tolerance for an unresolved lineage.

At publication scale, full continuation receipts are expensive. A production-like measured canary
observed a mean of 37,678 serialized bytes and 1.479 CPU seconds per continuation. Under the upper
census of 1,024 lineages, four states, and eight actions, `C=64` projects about 73.59 GiB and 35.91
serial CPU-days before a new dense-corpus benchmark. The current quality merger and preflight each
repeat the full replay serially.

## Decision

Use a preregistered train-only continuation study and make publication quality labels cover all
eligible train lineages.

- Construct the complete `--instances 0` production plan first, with one task per lineage, four
  states, eight actions, 256 reads, root seed 907, and the frozen continuation policy.
- Select 128 lineages from that plan with an outcome-blind domain-separated hash permutation.
- Commit and externally pin sampling seed 1907 and its domain before opening quality outcomes.
  EmbedBench and every other data generator are forbidden from consuming that reserved domain or
  root/domain pair, so lineage identities cannot be adapted to the study sample.
- Use nested continuation prefixes at counts `[12,16,24,32,48,64,96,128]` and immutable delta
  stages. Study rows have exactly the same task, state, action, and seed identities as their
  production projections.
- For each count, invert the inclusive upper tail of the finite-population hypergeometric
  distribution with exact integer arithmetic. Use Bonferroni alpha `0.05/8` and select the smallest
  count whose simultaneous lower bound is at least 128 resolved lineages.
- Publish failure with no selected count if the registered ladder does not qualify. Do not extend
  the ladder, lower gates, or invent winners after observing results.
- Require an externally pinned resolution receipt for publication quality generation, then generate
  quality-v7 for all train lineages.
- Run a frozen selector CPU/CUDA parity audit first. Use CPU only after 100 percent selected-index
  and compiled-program-digest parity. Otherwise preserve CUDA selector semantics and multiplex
  CPU-heavy shard workers per GPU. Training remains CUDA.
- Require a post-freeze production-like capacity canary before full generation.
- Replay complete lineage shards in parallel. Bind every replay shard in an externally pinned
  canonical replay-bundle manifest. Let the quality merger consume that trusted evidence instead of
  performing a second serial stochastic replay.
- Emit the existing strict quality-preflight v2 artifact from one global command that directly
  executes a fresh full replay through bounded multiprocessing. V2 does not claim that persistent
  replay shards prove its counts because it has no field that can bind their Merkle root.

## Alternatives considered

### Fix a convenient continuation count

This is fast but supplies no evidence that ranking rows resolve. Counts below 12 are impossible for
an eight-action row under the current rule. It was rejected.

### Choose the count from validation performance

This leaks model-selection outcomes into label construction and weakens the sealed evaluation
claim. It was rejected. Resolution selection is train-only.

### Generate an independent 128-lineage pilot

An independently planned pilot can choose different tasks, states, actions, or seeds from the
all-train corpus, so its finite-population conclusion need not apply to production. It was rejected.
The study is a sampled projection of the all-train plan.

### Keep production at 128 lineages

Requiring all 128 sampled lineages to resolve is brittle and leaves no safety margin. It was
rejected in favor of all 1,024 train lineages and a conservative lower bound on the number resolved.

### Run every ladder count independently

Independent reruns waste computation and change Monte Carlo draws. They were rejected in favor of
nested prefixes and progressive immutable deltas.

### Keep serial replay in both merge and preflight

This duplicates the dominant computation and delays training. It was rejected in favor of one
complete sharded replay consumed by merge and one direct but internally parallel preflight replay.

### Let preflight v2 assert aggregate shard counts without binding shards

An aggregate assertion has no exact audit trail. It was rejected. The replay bundle binds all shard
pins and row coverage for quality merge, while preflight v2 reruns the work itself. If the final
receipt should reuse and carry the Merkle root, a deliberate preflight v3 migration is required.

### Deduplicate continuation receipts immediately

Content addressing could reduce storage substantially, but it changes quality-v7 bytes and every
strict consumer. It was deferred to a separately approved successor schema. The current decision
uses a mandatory capacity gate.

## Consequences

- Continuation count becomes a reproducible design decision rather than an unchecked CLI value.
- Bonferroni control remains valid under nested counts and registered early stopping.
- All-train labels provide room to satisfy both 128-row and 128-lineage gates.
- Storage and CPU cost can be large, but progressive stages avoid unnecessary higher counts and
  sharded replay uses Apollo and Goose concurrently.
- Quality-v7 and final quality-preflight v2 remain compatibility-stable.
- Publication archives gain resolution plan, study receipts, selector parity, capacity, replay
  bundle, and raw-file pins.
- Existing `--instances 128`, free continuation-count, and serial-merge workflows remain diagnostic
  only.
