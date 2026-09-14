# ADR-009: Bind evaluator targets to authenticated public policy instances

## Context

A prepared training task carries two distinct record identities.

1. `PreparedProvenance.instance_record_digest` identifies the instance in the source release.
2. `policy_instances.jsonl[*].record_digest` identifies the verified public policy-instance row in
   the prepared corpus.

The evaluator-target payload intentionally uses the source-release identity. The independently
verified ground-certificate row instead carries `public_instance_record_digest`, which refers to the
public policy-instance row. These digests are expected to differ. Treating them as interchangeable
caused bank-backed quality replay and PPO training to reject valid production data after all source
artifacts had otherwise authenticated successfully.

## Decision

`PreparedTask` carries both identities explicitly.

- The prepared loader sets `public_instance_record_digest` from the already schema-checked and
  self-digest-verified public policy-instance row.
- Evaluator-target reconstruction continues to use
  `PreparedProvenance.instance_record_digest`. Its bytes and semantics do not change.
- The train-only target join compares the ground-certificate row's
  `public_instance_record_digest` with `PreparedTask.public_instance_record_digest`.
- Both values must be lowercase SHA-256 digests. Missing, malformed, or substituted values fail
  closed.
- The public digest is not added to initializer-bank plan identity. A bank plan already binds the
  exact prepared-manifest SHA-256, the semantic public-task digest, the initializer row, the design
  condition, and the base lineage. The live target join adds the independently authenticated public
  row check when evaluator targets are opened.

The resulting identity flow is:

```text
source release instance row
  -> provenance.instance_record_digest
  -> evaluator-target record_digest

prepared policy_instances.jsonl row
  -> public_instance_record_digest
  -> ground-certificate target row
  -> live train-only target join
```

## Validity boundary

The target-bearing consumer must authenticate all of the following before executing an episode:

1. the prepared manifest and the exact `policy_instances.jsonl` bytes;
2. the publisher attestation and target-access receipt;
3. the ground-certificate partition receipt;
4. the evaluator target reconstructed with the source-release identity;
5. the ground target joined with the public policy-instance identity;
6. the initializer-bank manifest, plan, episode, task digest, and complete-system context.

Neither identity is derived from a normalized `EmbeddingTask`. Recomputing an identity from graph
content would erase the authenticated record boundary and is therefore forbidden.

## Compatibility

Existing sealed initializer banks remain valid because their plans already pin the prepared
manifest and target-free task semantics. Adding the new in-memory field does not change plan bytes,
snapshot bytes, or the target-free bank contract. Quality artifacts and PPO runs created by the
incorrect target join are not publication eligible. They must be regenerated with a runtime whose
source inventory contains this decision.

## Consequences

- Public task identity and source-release provenance have one unambiguous role each.
- Target-free bank generation continues to open no evaluator targets.
- A valid bank can be reused across the corrected runtime when its existing external manifest pin
  and prepared-corpus pin both verify.
- A forged public row, wrong prepared manifest, wrong evaluator target, or mismatched ground row is
  rejected independently.
- Production training must use a newly built runtime. A diagnostic hot patch is evidence that the
  code path executes, but it is not a publication runtime.

## Verification

The regression suite covers distinct source and public digests, malformed public digests, target
mutation, prepared loading, bank planning and loading, capacity replay, resolution-delta replay,
and bank-backed PPO. The complete local suite after the change passed 1,288 tests with one
CUDA-only test skipped on macOS.
