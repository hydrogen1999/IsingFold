# External Baseline Tuning and Preregistration

## Objective

IsingFold must compare its learned complete system against a strong stock-minorminer
baseline, not against an arbitrary untuned invocation. Baseline design choices are selected
once on authenticated validation data and frozen before any test task is loaded. The output is
an immutable selection receipt whose file SHA-256 must be recorded outside the evaluation
directory and supplied explicitly to every publication-facing stock test command.

Success means that the pipeline evaluates every registered candidate on the same validation
census and seed blocks, selects exactly one candidate without seed selection, and mechanically
rejects test access, authority drift, partial runs, post-hoc candidates, or evaluator feedback at
deployment.

## Scientific contract

The finite candidate registry contains three baseline families:

1. `stock-default`: one native minorminer call with the documented 0.2.22 defaults made
   explicit, including ten internal tries.
2. `time-saturating-resource`: independently seeded one-try calls until the common wall-clock
   deadline or the Context restart-work cap of 64, followed by resource-lexicographic selection.
3. `time-saturating-quality`: the same search schedule, but candidate embeddings are ranked by
   the maximum predicted success probability from the already frozen deployment strength
   selector. It opens no evaluator result. Ties use fewer qubits, shorter maximum chain,
   embedding digest, and restart index in that order. The selected strength is the frozen
   selector's argmax over the four compiled programs.

The registry includes a small, fixed patience grid for the two time-saturating families. No new
candidate may be added after validation results are inspected under the same registry identity.
Every candidate uses the same total online wall-clock limit, final audit-read count, task census,
quality authority, selector, context, and software registry.

The learned and stock systems do not claim equal internal algorithmic work. The stock
time-saturating candidates may use up to 64 native calls while the learned initializer has its
own registered attempt cap. Fairness is enforced by the same total online wall-clock and the
same prospective Context work caps; all exposed work is reported rather than equated.

## Validation and test boundary

- Tuning accepts only prepared production schema-v4 tasks in the `val` split.
- Every run and the final receipt state `partition=validation` and
  `test_data_opened=false`.
- A request naming `test`, a test task, a mixed validation/test census, or an artifact claiming
  that test was opened is an integrity error. It is never converted to utility zero.
- Ordinary search, timeout, known prospective work-cap, compilation, and invalid-return failures
  remain in the validation denominator with utility zero.
- The test runner must load and authenticate the tuning receipt and its caller-supplied expected
  file SHA-256 before loading prepared test targets.

## Selection estimand

Each registered candidate is evaluated under tuning seeds 1103, 2207, and 3301. Within each seed,
repetitions and instances are averaged inside immutable base lineage. Candidate utility is then
the equal-weight mean over seeds and lineages. The deterministic selection key is:

1. larger unconditional IF-Q3-S0 utility;
2. larger valid-return rate;
3. lower mean total online seconds;
4. lower known outer restart work;
5. earlier candidate position in the immutable registry.

Floating values are compared exactly after canonical finite-JSON decoding. Seed-specific winners
are diagnostic only and cannot change the selected candidate.

## Power, work, time, and failures

The registered design requires at least 128 independent validation lineages, all three tuning
seeds, four repetitions, and a complete candidate-by-seed-by-lineage census. The receipt reports
the achieved lineage count and attempt count. A 20,000-replicate crossed seed-by-lineage bootstrap
with a fixed RNG seed reports uncertainty but does not alter the predeclared deterministic
selection key.

Each run reports total and mean online seconds, the wall-clock cap, wall-clock exhaustion count,
known work-cap exhaustion count, and all ordinary terminal failure counts. Compiler, validator,
restart, evaluator-read, and feature-work totals are exact. Native internal decision, routing,
materialization, and cut-visit coordinates remain null because the public stock API does not expose
them. The paper must not claim equal internal work.

## Artifact identities

Every tuning report and the final selection receipt bind:

- tuning-registry semantic digest and file SHA-256;
- staged-grid semantic contents and file SHA-256;
- external complete-system config semantic digest and file SHA-256;
- full runtime source and dependency registry plus its digest;
- frozen selector identity and artifact digest;
- publisher quality authority and its digest;
- validation `TargetAccessReceipt` and matching ground-partition receipt;
- validation task census and immutable-lineage census digests;
- context and work-cap identity;
- complete candidate, seed, lineage, repetition, time, work, and failure denominators.

All JSON is canonical, finite, exact-schema, self-digested, atomically written, and immutable by
default. Source report file hashes are retained in the selection receipt.

## Commands

The publication workflow is:

```bash
python3 -m isingfold.rl.cli evaluate-external-tuning-cell \
  --registry configs/external_minorminer_tuning_hybrid_v1.json \
  --expected-registry-sha256 "$EXTERNAL_TUNING_REGISTRY_SHA256" \
  --grid configs/rl_grid_hybrid_v1.json \
  --index 0 \
  --tuning-seed-index 0 \
  --corpus runs/prepared_v4 \
  --selector runs/selector_v4 \
  --quality-attestation "$QUALITY_ATTESTATION" \
  --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST" \
  --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID" \
  --ground-certificate-root "$GROUND_CERTIFICATE_ROOT" \
  --expected-ground-certificate-root-sha256 \
    "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256" \
  --external-config configs/external_minorminer_complete_v1.json \
  --learned-config configs/complete_system_lac_hybrid_cache_v1.json \
  --out runs/external_tuning_v2

python3 -m isingfold.rl.cli select-external-tuning \
  --registry configs/external_minorminer_tuning_hybrid_v1.json \
  --expected-registry-sha256 "$EXTERNAL_TUNING_REGISTRY_SHA256" \
  --grid configs/rl_grid_hybrid_v1.json \
  --external-config configs/external_minorminer_complete_v1.json \
  --evaluations-root runs/external_tuning_v2 \
  --out runs/external_tuning_selection_v2.json

export EXTERNAL_TUNING_SELECTION=runs/external_tuning_selection_v2.json
export EXPECTED_EXTERNAL_TUNING_SELECTION_SHA256='<recorded outside the run directory>'
```

The two commands are unavailable for `--partition test`; the partition is fixed internally to
validation. Publication test evaluation later requires both the frozen receipt path and
`EXPECTED_EXTERNAL_TUNING_SELECTION_SHA256`. The evaluation command writes one cell under
`OUT_ROOT/<candidate-id>/seed-<seed-index>`. Run all seven candidates for a tuning seed on the
same machine so runtime differences cannot be mistaken for candidate differences. Apollo runs
`scripts/apollo_external_tuning.sh` directly. Goose submits
`scripts/goose_external_tuning.sbatch`; its three-array jobs each run all seven candidates
sequentially inside one Slurm allocation.

For long production runs, use `plan-evaluation-shards` with workflow
`external-validation-tuning`, then execute `run-evaluation-shard` through
`scripts/apollo_evaluation_shards.sh` or a Goose Slurm array through
`scripts/goose_evaluation_shards.sbatch`. The target-free plan fixes the complete validation
population, candidate and tuning-seed coordinates, compute class, and whole-lineage shards. Each
worker opens only validation targets and emits a shard receipt v3. `merge-evaluation-shards`
requires every pinned shard and recomputes the unsharded tuning inputs under merge receipt v2.
Publication runs use a pinned virtual-environment lock on Apollo or a pinned Apptainer executable
and image on Goose; bare-metal mode is diagnostic-only.

## Project structure and verification

- `configs/external_minorminer_tuning_hybrid_v1.json`: immutable finite registry.
- `src/isingfold/rl/external_tuning.py`: registry, report, aggregation, and receipt contracts.
- `src/isingfold/rl/cli.py`: validation evaluation and selection entry points.
- `tests/unit/test_rl_external_tuning.py`: fail-closed and deterministic-selection tests.
- `documents/TRAINING_OPERATIONS.md`: executable Apollo and Goose workflow.

Verification commands are:

```bash
.venv/bin/python -m pytest -q tests/unit/test_rl_external_tuning.py
.venv/bin/python -m ruff check src/isingfold/rl/external_tuning.py \
  tests/unit/test_rl_external_tuning.py
.venv/bin/python -m py_compile src/isingfold/rl/external_tuning.py
```

## Boundaries

- Always validate config and registry pins before evaluator-bearing validation tasks are loaded.
- Always evaluate every candidate and all three seeds; never select a seed.
- Ask before changing the candidate registry, power threshold, or selection rule after a receipt
  has been published.
- Never accept test rows for tuning, use evaluator outcomes during deployment reranking, infer a
  missing authority, overwrite an existing receipt, or turn integrity errors into method failures.
