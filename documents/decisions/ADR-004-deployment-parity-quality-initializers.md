# ADR-004: Deployment-parity initializer banks for quality supervision

## Context

The learned embedder is deployed as a policy inside a persistent environment. Each episode begins
from one sealed initializer snapshot and exposes exactly two precomputed restart snapshots. The
quality-supervision pipeline previously reconstructed states and counterfactual continuations with
the legacy online initializer path. That path can draw a different initial embedding and different
restart outcomes for each replay. A label produced under those dynamics does not estimate the
action value of the deployed policy environment.

This mismatch affects more than reproducibility. It changes the transition kernel, the reachable
state distribution, the meaning of a `RESTART` action, and therefore the target
`Q_U^mu(s,a)`. Increasing the continuation count cannot repair a target generated under the wrong
kernel.

## Decision

Publication quality supervision, continuation-count resolution, exact delta replay, and the
capacity canary must consume the same authenticated persistent K=2 initializer-bank abstraction as
deployment.

- The caller supplies a loaded `InitializerSnapshotBank` and the raw SHA-256 pin of its manifest.
  A filename or self-declared digest inside the artifact is not an authority.
- The quality consumer authenticates the existing bank manifest, access receipt, plan, prepared
  corpus, train partition, complete-system configuration digest, context digest, episode schedule,
  and exactly two restart-cache slots. This is a consumer contract over the current initializer-bank
  schema. It does not add a model-source digest or change the bank format.
- The target-free production plan assigns every planned task to a concrete sealed episode. Every
  planned row stores that episode index and its bootstrap-record digest. The environment seed is the
  sealed initial snapshot's system seed. Assignment uses the registered
  `sha256-task-modulo-sealed-episodes-v1` rule, rather than always selecting the earliest episode.
- A prepared row must match the exact source-record digest sealed for that task in the bank plan.
  Its prepared provenance, design-condition registry row, initializer identity, and planning
  `Context` therefore cannot be substituted after bank generation.
- Planning opens no evaluator target. Train targets and their authenticated ground-certificate
  authority enter only when the frozen plan is executed.
- Every continuation receipt records the global bank-contract identity and the episode-local
  initial and restart snapshot identities. Delta, resume, verification, selection, and canary
  artifacts bind the same contract.
- Publication-quality records pass a stricter boundary than diagnostic records: every retained
  continuation must use one identical publication-eligible bank and episode binding. Legacy,
  test-only, empty, or mixed-binding records fail closed.
- Publication CLI entry points for resolution planning, delta execution and verification,
  capacity planning and execution, full label generation, and Gate 2 require the bank path, its
  out-of-band manifest SHA-256, and the complete-system configuration. A production command cannot
  silently select the diagnostic online initializer.
- Resume journals are caches, not authorities. Every cached result row is exactly replayed against
  the authenticated bank before it can enter a newly sealed delta.
- Exact replay reconstructs the environment from the stored episode. It must not call an online
  initializer, redraw a restart, substitute another episode, or derive a new seed.
- The legacy online-initializer path remains available only for explicit diagnostics. Its receipt
  states `publication_eligible=false`. It cannot enter a publication plan.
- Gate 2 replays the target-free resolution sample, which contains at least 128 independent train
  lineages, against the same bank. It audits every legal materialized candidate. There is no
  candidate-count CLI truncation. The gate receipt binds both the resolution-plan identity and the
  initializer-bank contract.

The quality objective remains unchanged by this decision. Count-aware action-value targets, paired
sibling random streams, continuation-count selection, and alternative policy losses are separate
contracts layered above the same environment kernel.

## Validity boundary

A production operation fails closed when any of the following is missing or inconsistent:

1. the loaded bank or its caller-supplied manifest pin;
2. the prepared-corpus identity or train-only access receipt;
3. publication eligibility of the initializer backend;
4. the K=2 restart-cache width;
5. the row's episode index, bootstrap digest, public task digest, context, or environment seed;
6. train target-access and ground-certificate receipts for target-bearing execution;
7. the initializer binding embedded in a continuation, delta, resume journal, verifier replay,
   capacity selection, or canary.

Artifact-only inspection may validate a serialized binding against its signed parent artifact.
Live continuation replay additionally requires the loaded bank and the external pin. A serialized
bank contract is not permission to synthesize a replacement environment.

## Contract migration

The semantic change is explicit in the following consumer schemas:

| Contract | Version after this decision | Added binding |
|---|---:|---|
| continuation receipt | 2 | exact initializer episode and K=2 snapshots |
| state-action envelope, bound candidate, applied action | 2 | restart-cache slot and cache-after digest |
| quality label identity | `if-q3-s0-qmu-6` | new continuation-receipt semantics |
| publication quality manifest | 8 | exact bank contract and resolution-plan identity; rows remain v7 |
| publication quality shard and merged manifest | 6 | exact bank contract and resolution-plan identity |
| publication quality preflight | 3 | bank and plan identity copied from the exact replayed manifest |
| quality publication binding and training-input readiness | 2 | cross-check manifest, preflight, bank, and plan identities |
| resolution production plan and row | 2 | bank contract, episode index, bootstrap digest |
| resolution delta, row, verifier, and resume journal | 2 | plan-matching bank and episode binding |
| capacity selection and canary | 2 | plan-matching bank contract |
| release gate receipt | 7 | target-free full legal-support audit and plan-bound bank identity |

Readers accept the new versions exactly. They do not reinterpret old bytes as the new target. Old
quality, resolution, and canary artifacts must be regenerated from a pinned bank. They must not be
silently relabeled or upgraded by copying fields.

## Consequences

- Supervised quality targets and PPO deployment now share one initialization and restart transition
  kernel.
- Counterfactual siblings can replay the same immutable initial and restart material, which removes
  initializer redraws as an uncontrolled source of variance.
- Publication jobs require the bank to be generated and externally pinned before target-bearing
  quality execution begins.
- A legal cached `RESTART` must carry its slot and cache-after digest through the action certificate.
  Tampering with either field invalidates the certificate.
- Tests may use a test-only bank only through an explicit opt-in. That opt-in never changes the
  serialized `publication_eligible` fact.
- A diagnostic capacity canary may be inspected with an explicit test opt-in, but it can never
  mint the full-generation launch capability. A production launch capability names both the bank
  manifest SHA-256 and bank-contract record digest.

## Alternatives considered

### Keep online initializer restarts for labels

This preserves the old implementation but estimates a different MDP from deployment. It was
rejected.

### Store only the initializer seed

A seed does not authenticate the initializer implementation, accepted-draw history, snapshot bytes,
or restart cache. It was rejected in favor of sealed snapshot identities.

### Accept a bank path without an external manifest pin

This lets mutable local state become its own authority. It was rejected. Every live consumer needs
the caller-supplied SHA-256 pin.

### Change the initializer-bank schema to include the current neural model

Quality supervision needs the deployment initializer and transition kernel, not the future policy
weights that will be trained from those labels. Coupling the bank to a model-source digest would
also invalidate reusable banks after every model or loss change. It was rejected.
