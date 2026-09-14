# ADR-006: Fail-closed provenance for warm-start transfer

## Status

Accepted.

## Date

2026-09-13

## Context

An RL-value run may initialize its model from a completed representation checkpoint. The checkpoint
loader already authenticated the checkpoint against the adjacent `run.json`, the corpus, selector,
normalizer, quality authority, model identity, loss contract, runtime registry, and recorded payload
digest. Those checks established internal consistency, but they did not establish that the caller
had selected the correct representation artifact.

In particular, a valid checkpoint from another registered seed, representation grid cell, grid
revision, or quality preflight could pass the old loader if its other contracts matched. A staged
RL-value run could also load a different valid checkpoint for the same family without comparing its
payload digest with the representation checkpoint authenticated by the validation selection
receipt. This is a provenance substitution risk, not a change to the scientific selection rule.

## Decision

Warm-start transfer is fail closed. Before checkpoint deserialization, `_load_transfer` requires and
checks the following caller-pinned source identity:

- the exact training seed;
- the exact representation grid cell identifier;
- the raw SHA-256 of the registered grid manifest;
- the raw SHA-256 and canonical record digest of the currently trusted quality preflight;
- the selected representation checkpoint payload digest when an authenticated selection receipt
  defines one.

Production transfer also accepts only the registered `full-qmu-v4` loss profile with its exact
rank, utility, action-value, and COMMIT-delta coefficients and no diagnostic binding. A valid
rank-value-only Q-label control checkpoint is evidence for an ablation, not a production policy
initializer.

The loader first validates the syntax of every expected digest. It then authenticates `run.json` and
compares its experiment contract with the expected seed, source cell, grid, and preflight. If a
selected payload digest is supplied, the receipt must name that exact payload before any Torch
checkpoint deserialization occurs. After deserialization, the computed checkpoint payload digest
must still equal the adjacent run receipt. The existing model, corpus, selector, normalizer, quality
authority, loss, and runtime checks remain mandatory.

For the registered staged grid, the representation-selection receipt is already authenticated by
the existing selection loader. A second bounded lookup binds the exact `(cell_id, model_family,
seed)` source row to its `checkpoint_payload_digest`. The raw file SHA-256 and canonical record
digest are rechecked during that lookup to prevent a changed receipt from being used between
validation and payload extraction.

Complete-system confirmation intentionally does not reuse the validation-selected RL-value
checkpoint. It trains a fresh representation initializer for the frozen family and seed. Transfer
therefore binds that fresh artifact to its expected representation cell, grid, and current quality
preflight, while the adjacent receipt and checkpoint loader bind its actual payload. The selected
validation payload remains recorded separately in the complete-system selection binding as source
evidence, with `selected_validation_checkpoint_reused` set to false.

Manual family overrides remain diagnostic. They receive the same seed, source-cell, grid, and
preflight checks, but have no validation-selected payload to pin. Such runs are not eligible for the
registered scientific result.

## Alternatives considered

### Trust the checkpoint path convention

The run-root layout encodes stage and cell identifiers, but a path is not an authenticated artifact
identity. A copied or replaced checkpoint could preserve the expected filename. This option was
rejected.

### Trust only the adjacent run receipt

This detects corruption and inconsistency between a checkpoint and its metadata, but a complete
and valid artifact from the wrong seed or grid remains self-consistent. This option was rejected.

### Reuse the selected validation checkpoint during complete-system confirmation

This would provide a direct selected-payload pin, but it would violate the frozen confirmation rule
that retrains every selected seed on the deployment initializer support. This option was rejected.

## Consequences

- A wrong-seed, wrong-cell, wrong-grid, stale-preflight, or substituted selected checkpoint is
  rejected before Torch deserialization.
- Scientific staged transfer now has an unbroken chain from the current quality preflight and grid
  to the representation selection source row, representation run receipt, and checkpoint payload.
- Direct warm-start transfer callers must supply the expected source provenance. Missing pins no
  longer fall back to values declared by the checkpoint itself.
- Because the CLI source is part of the runtime implementation registry, representation artifacts
  created before this change must be regenerated under the current source before registered
  transfer.
- The representation family selection rule, feasibility gate, utility ordering, training losses,
  and optimization schedule are unchanged.
