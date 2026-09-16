# Quality and occupancy contract audit

Base reviewed: `7c5d2252bd8c9c2b4c331e1ba5b93920cdbbf219`.
This is an implementation audit, not a new training result. Existing result logs are preserved.

## Objective and benchmark interpretation

The target is downstream solution quality under a declared budget. Qubit count is a constraint
and a reported operating point, not the default quantity minimized by the policy. Occupancy is
the number of distinct occupied physical qubits divided by the size of the available host.
At 90 percent actual occupancy, ten percent remain free, but not every free qubit is reachable
from every chain. Local legality does not guarantee that future demands can be completed.

Two experiments answer different questions:

1. A partial embedding with unmet demands tests completion in tight space.
2. A valid embedding, including a supplied planted witness, tests how to allocate its remaining
   space for quality. Supplying that witness to all arms is a controlled start, not evidence
   that the policy learned to construct it.

End-to-end deployment starts with empty chains. A witness or externally supplied complete
embedding is never a fallback deployment output. A COMMIT can return only a valid embedding
constructed during that run. The contact probe implements the second case as a training and
continuation diagnostic, not as an end-to-end deployment benchmark. Its contact-only candidate
support is a chosen action restriction, not a proof that the best extension is always in that
support. The supplied start is assessed separately as a reference and is not a selectable output
candidate. The current proposal compares bounded growth trajectories; it does not implement a
learned intermediate STOP or rip-up decision.

## Corrections

### State features

- Cache identity includes chain membership and logical-variable identity, so equal-length
  rewrites do not reuse stale occupancy.
- Empty candidates use the same coefficient channels as nonempty candidates. The existing
  35-channel width and nonempty ordering are preserved, including five reserved channels.
- Occupancy counts distinct qubits, isolated variables have a safe degree normalization,
  and invalid feature budgets are rejected.
- Frontier traversal counts short dead ends correctly and obeys its discovery cap. Capped
  counts are lower bounds; they are not a certificate of a blocked direction.

The coefficient channels are magnitude summaries, not a complete physical-program encoding.
Shape-compatible old checkpoints should still be re-evaluated because corrected observations
change behavior.

### Contact policy

The allowed spend can be zero when the start fills its budget. Both compared arms receive the
same number of growth-candidate slots; the supplied start is only a separately measured reference.
No keep-start fallback is added to the deployment action space. Failed measurements are counted,
with the declared failure convention and coverage reported, rather than deleting unfavorable
arms silently.

Policy-gradient updates use an action-independent leave-one-out baseline across independent
episodes of the same instance and the sum of trajectory log probabilities. Validation sampling
and measurement streams are separate from training. Splits are grouped by lineage, and logs
distinguish requested spend from actual occupancy and supplied-start availability.

### Hybrid completion and evaluation

Minor-embedding checks cover host membership, nonempty connected disjoint chains and every
logical edge. Router calls receive remaining time, and late results do not count as arriving
within the deadline. A native solver's cooperative timeout is not a hard process-kill guarantee;
actual elapsed time must still be reported.

Proposal, completion and selection consume the same declared deployment budget for both arms.
Independent final assessment is outside selection and reported separately. Explicit qubit caps
are constraints. A witness-derived legacy cap must be requested and identified, rather than
silently used as deployment information. Checkpoint selection and conditional quality reporting
must not depend only on instances where the competing baseline also succeeds.

### Adaptive allocation and scorer metadata

Selection and assessment use separate sampler streams. Read accounting includes unsuccessful
requests and never silently grants an arm extra reads. Learned-prior loading checks architecture,
candidate alignment and training-lineage provenance. An explicit exploratory override is not
a paper-valid substitute for matching corpus, context and split metadata.

The scorer and adaptive probe currently optimize the registered solve-rate endpoint. A scorer
trained on it does not automatically become an energy-residual predictor for high-fill tasks.

### Growth sweep

If several growth draws compete, their pilot measurements choose the candidate; a fresh block
assesses the winner. The baseline receives the same assessment reads. Extra pilot selection
cost is declared, so this sweep is not presented as a matched-total-cost learned-method win.
Rows are independent growths from the same start, not a nested sequence. Geometry and physical
programming change along with qubit use. Table columns and objective provenance are explicit.

## Evidence boundaries

- No remote training, SA benchmark or QPU run was launched by this audit.
- Registered contact-growth intervals in the existing logs include zero. They do not establish
  a positive gain or noninferiority of contact growth.
- The large scorer result is validation evidence against random. It is not an unopened test
  result or a demonstrated gain over plain successive halving.
- A finite-read measured-selection reference is not a theoretical learnable ceiling.
- These corrections require versioned reruns before replacing any historical number.

## Verification

The integrated targeted suite passed **69 tests**. It covers the five changed probes, candidate
features, prior provenance, compiler/schedule consistency, real toy routing, mocked-clock limits,
and training/checkpoint smoke tests with synthetic measurements. A deployment regression makes
the held-out witness inaccessible, so the default hybrid path cannot silently use it.

Command (Python 3.12, `OMP_NUM_THREADS=2`):

```sh
python -m pytest -q --tb=short \
  tests/unit/test_candidate_features_contracts.py \
  tests/unit/test_contact_policy_contracts.py tests/unit/test_contact_policy.py \
  tests/unit/test_hybrid_rl.py tests/unit/test_adaptive_allocation_contracts.py \
  tests/unit/test_budget_sweep_protocol.py tests/unit/test_prioritiser.py \
  tests/unit/test_cache_provenance.py tests/unit/test_probe_compile_matches_env.py \
  tests/unit/test_registered_schedule.py
```

Undefined-name checks (`ruff F821,F822,F823`) and `git diff --check` passed.
The full `tests/unit` suite did **not** pass: collection reported 12 test modules requiring the
unbuilt `lac_minorminer._core`, and five requiring `isingfold_lac_b`, which is absent from the
reviewed baseline source. A broad continue-on-collection-errors run encountered further failures
and was interrupted; it is not a completed full-suite result. Eleven sampled failures in unchanged
checkpoint/calibration/native-registry tests were reproduced as missing `_core`. This is not a
claim that every remaining legacy failure has been classified or fixed.

No training performance, final-test generalization or QPU behavior is certified by these tests.
