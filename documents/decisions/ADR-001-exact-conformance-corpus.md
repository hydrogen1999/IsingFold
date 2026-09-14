# ADR-001: Deterministic bounded exact-conformance corpus

## Context

Release Gate 1 runs exponential independent program checks and therefore needs a small bounded task
set. The current schema authenticates a sorted list of at most eight task IDs, but the loader does
not prove how those tasks were chosen. An externally pinned hash protects bytes, not selection
validity. A hand-picked list could therefore enter Gate 1 while satisfying the old loader.

## Decision

Keep `isingfold.exact-conformance-corpus` schema version 1 and make its selection semantics fixed by
`corpus_id=if-gate1-exact-v1`.

- Load only the public prepared-v4 validation partition, without evaluator targets.
- Require at least eight eligible base lineages, a public witness, complete design coordinates, and
  at most 12 logical variables.
- Select exactly eight distinct lineages with the deterministic greedy marginal-coverage algorithm
  and domain-separated tie digest defined in
  `documents/EXACT_CONFORMANCE_AND_RESOLUTION_SPEC.md`.
- Fix the selection seed to 0 as part of the corpus identity.
- Require a raw registry SHA-256 recorded out of band.
- Make the producer, independent verifier, and Gate 1 loader call the same selection function. The
  loader rejects any alternate task list even when that list is canonical, self-digested, and
  externally pinned.
- Keep exact conformance itself in Gate 1. The registry asserts selection, not correctness.

## Alternatives considered

### Accept any externally pinned bounded list

This protects integrity but allows outcome-aware or convenience-based task selection. It was
rejected because it cannot support a paper claim about a fixed conformance protocol.

### Add selection metadata to schema v1

This would alter a strict compatibility schema without changing its version. It was rejected.
Selection rule, seed, and implementation evidence instead follow from the fixed corpus ID and the
verification receipt.

### Introduce exact-conformance schema v2

A v2 artifact could embed a richer selection census. It was rejected for this change because the
existing v1 fields are sufficient when every consumer recomputes the selection. A future v2 remains
appropriate if the task cap, partition, or selection axes change.

### Draw eight tasks uniformly at random

Uniform selection is simple but can omit host, fault, origin, or difficulty levels in a tiny set.
It was rejected in favor of deterministic marginal coverage with a hash tie.

## Consequences

- The exact registry is byte-reproducible and outcome-blind.
- A pin alone is no longer sufficient; prepared-v4 public data and the selection implementation are
  required to verify it.
- Gate 1 retains its eight-task and 12-variable bounds.
- Existing v1 registry files that do not equal the recomputed set are intentionally rejected.
- No validation target is exposed by corpus production or verification.
- Changing the rule or seed requires a new corpus identity and an explicit compatibility decision.

