# Exact Conformance, Quality Resolution, and Scalable Replay Specification

## 1. Purpose

This document specifies three protocols that must exist before the IsingFold paper workflow can
start the registered 9 plus 18 model grid:

1. production and independent verification of the bounded exact-conformance corpus used by release
   Gate 1;
2. a train-only continuation-count study that selects the Monte Carlo denominator for quality-v7
   labels without inspecting validation or test outcomes;
3. scalable exact replay of the full quality-v7 corpus, with replay evidence that can be trusted by
   both the quality merger and the existing quality-preflight v2 consumer.

The protocols fill operational gaps. They do not change the primary IsingFold objective, the
quality target `if-q3-s0-qmu-5`, the simultaneous plausible-best rule, the IF-Core model, the
9 plus 18 grid, or the validation and test selection rules.

The authoritative inputs are:

- `documents/ISINGFOLD_PROJECT_SPEC_EN.md`;
- `documents/ISINGFOLD_MODEL_SPEC.md`;
- `documents/TRAINING_OPERATIONS.md`;
- `configs/rl_grid_hybrid_v1.json`;
- prepared corpus schema v4, selector bundle v4, quality row v7, quality shard and merge envelope
  v5, quality preflight v2, and release gate v6.

If this proposal conflicts with one of those inputs, Section 15 identifies the conflict explicitly.
Implementation must not silently resolve such a conflict.

## 2. Explicit assumptions

1. Production uses prepared schema v4 and has at least 1,024 immutable train base lineages and 512
   immutable validation base lineages.
2. The exact-conformance registry is selected from the public validation partition. Selection does
   not open an evaluator target, a ground-partition receipt, or any quality outcome.
3. Gate 1 remains bounded to exactly eight registered tasks, each with at most 12 logical
   variables and a public initial witness.
4. The production quality plan uses all eligible train base lineages. In the existing CLI notation,
   this is `--instances 0`, not `--instances 128`.
5. The production quality protocol keeps one task per lineage, at most four stored states per
   lineage, at most eight evaluated actions per state, 256 reward reads, the frozen uniform legal
   continuation policy, and root seed 907.
6. A resolution-study row and its corresponding production row use the same lineage, task,
   prefix, state fingerprint, evaluated action set, environment seed, continuation policy, and
   ordered continuation seeds. The study samples rows from the full production plan. It does not
   build a separate 128-lineage plan.
7. Training and learned complete-system evaluation remain CUDA workloads. This document does not
   move model training to CPU.
8. The frozen strength selector may run on CPU during quality generation only after a complete
   train-corpus parity audit proves 100 percent equality between CPU and CUDA selected indices and
   compiled strength digests. Otherwise quality generation retains CUDA selector semantics.
9. JSON digests use the repository's canonical finite JSON encoding. A `record_digest` is the
   SHA-256 digest of the canonical payload without the `record_digest` field. A file SHA-256 is the
   digest of the exact bytes on disk and is a separate identity.
10. Every publication artifact is immutable, written through a sibling temporary path, flushed,
    atomically renamed, and rejected if the destination already exists or resolves through a
    symlink.

## 3. Invariants shared by all three protocols

### 3.1 Authority and partition boundaries

- Public planning may authenticate prepared-v4 public files, publisher attestation, selector
  metadata, and the target-free ground root. It must not open `targets/train.jsonl`,
  `targets/val.jsonl`, `targets/test.jsonl`, or a target-bearing ground-partition receipt.
- A worker that evaluates continuation utility opens only the train target partition through the
  existing publisher and ground-certificate checks. It retains the exact train
  `TargetAccessReceipt` and ground-partition receipt.
- Exact-conformance production and verification never need target access.
- Validation and test targets are forbidden in the continuation study, capacity canary, quality
  generation, quality replay, and quality merge.
- A self-digest establishes integrity, not authority. Every plan, registry, replay bundle, and
  final receipt that crosses a trust boundary also requires a caller-supplied lowercase raw-file
  SHA-256 obtained outside the file being authenticated.

### 3.2 Determinism and seed derivation

No implementation may use Python's randomized `hash`. A derived seed is:

```text
seed63(root, domain, parts...) =
  low_63_bits(SHA256(canonical_json({
    "domain": domain,
    "parts": [root, *parts],
    "version": 1
  }))[0:8])
```

The root is the first element of `parts`; it is not a separate JSON field. The exact byte order is
big-endian, matching the existing quality seed implementation. The
following domains are disjoint and fixed:

| Use | Domain |
|---|---|
| Exact-corpus task tie | `exact-conformance-task-tie-v1` |
| Resolution lineage sampling | `quality-resolution-lineage-sample-v1` |
| Resolution shard assignment tie | `quality-resolution-shard-tie-v1` |
| Production environment | existing `quality-environment` |
| Continuation | existing `quality-continuation` |
| Preflight shard assignment tie | `quality-preflight-shard-tie-v1` |

The exact-conformance selection seed is fixed to 0 by corpus identity
`if-gate1-exact-v1`. The quality root seed is fixed to 907 by the production protocol. The
resolution sampling seed is fixed to 1907 by `configs/quality_resolution_v1.json`. Changing a root,
domain, framing rule, or candidate ladder creates a new protocol identity.

The config bytes, sampling seed 1907, and sampling domain must be committed and externally pinned
before any quality outcome is opened. EmbedBench generation, prepared-corpus construction,
selector labeling, and every other data-generation process must not consume either the reserved
domain `quality-resolution-lineage-sample-v1` or the reserved root/domain pair containing 1907.
This separation prevents the generator from adapting lineage identities to the eventual study
sample. Accidental prior reuse invalidates the randomization claim and requires a new preregistered
sampling seed and study identity.

Every producer checks the complete derived-seed set for collisions before publishing. A collision
is an error even though its probability is negligible.

### 3.3 Canonical files and publication

- JSON must reject duplicate keys, nonfinite numbers, booleans where integers are required, unknown
  fields, blank JSONL rows, and noncanonical set ordering.
- JSONL is UTF-8, one canonical object plus one newline per record.
- Lists representing sets are unique and lexicographically sorted.
- Paths are regular files or directories beneath explicit roots. Inputs and destinations may not
  be symlinks.
- Directory artifacts write data files first and `manifest.json` last. A partially written
  directory is never accepted as an artifact.
- Publication uses no-replace semantics. Competing publishers yield one complete winner and one
  clean failure, never overwrite or merge partial output.

## 4. Why the continuation denominator cannot remain a guess

For a row with `A` evaluated actions and `C` continuation samples per action, the current
plausible-best rule uses

```text
r(C, A) = sqrt(log(2 A / 0.05) / (2 C)).
```

An action is removed only when its upper confidence endpoint is strictly below the largest lower
endpoint. Rewards are bounded in `[0, 1]`, so the largest possible empirical gap is 1. Therefore
any row can resolve only if `2 r(C,A) < 1`, equivalently

```text
C > 2 log(2 A / 0.05).
```

For the production cap `A = 8`, the right side is about 11.54. Thus `C <= 11` cannot resolve any
eight-arm row, even under the extreme empirical means 1 and 0. In particular, `C <= 8` is
mathematically incapable of producing ranking signal for an eight-arm row. The first admissible
integer is 12. For reference, the corresponding minima are 9 for two arms and 11 for four arms.
A singleton-action row is always fully plausible and never supplies ranking signal.

This is a possibility bound, not a claim that `C = 12` is adequate. Adequacy is selected by the
train-only finite-population study below and confirmed by complete exact preflight.

## 5. Protocol A: exact-conformance corpus

### 5.1 Objective and boundary

The producer creates the only task registry accepted by Gate 1. It selects a small, diverse,
outcome-blind set for exponential program checks. The verifier and the Gate 1 loader both recompute
the fixed selection from public prepared-v4 data. A correct SHA-256 over a different hand-picked
task list is rejected.

The registry proves selection identity only. It does not assert that exact conformance passed;
`gate_conformance` remains the authority for witness, programming, independent program oracle,
structural motif, and overlap-handoff checks.

### 5.2 Commands

```bash
export PREPARED_CORPUS=runs/prepared_v4
export EXPECTED_PREPARED_MANIFEST_SHA256='<out-of-band manifest.json SHA-256>'

isingfold-rl make-exact-conformance-corpus \
  --corpus "$PREPARED_CORPUS" \
  --expected-corpus-manifest-sha256 "$EXPECTED_PREPARED_MANIFEST_SHA256" \
  --out runs/exact_conformance_v1.json

export EXACT_CONFORMANCE_CORPUS=runs/exact_conformance_v1.json
export EXPECTED_EXACT_CONFORMANCE_SHA256='<out-of-band registry file SHA-256>'

isingfold-rl verify-exact-conformance-corpus \
  --corpus "$PREPARED_CORPUS" \
  --expected-corpus-manifest-sha256 "$EXPECTED_PREPARED_MANIFEST_SHA256" \
  --registry "$EXACT_CONFORMANCE_CORPUS" \
  --expected-registry-sha256 "$EXPECTED_EXACT_CONFORMANCE_SHA256" \
  --receipt-out runs/exact_conformance_verification_v1.json
```

Both commands are CPU-only and target-free. `--receipt-out` is optional for local diagnosis and
mandatory for a publication archive. The existing `gates` command still consumes the registry and
its external file pin. Its loader must perform the same recomputation as the verifier.

### 5.3 Fixed eligible population

Load the public `val` partition with `include_evaluator=False`. A task is eligible only when:

- it is prepared schema v4 and belongs to `val`;
- its task ID and immutable base lineage are unique;
- it has a nonempty public initial embedding or witness;
- its logical graph contains at most 12 variables;
- it has a complete `PreparedDesignCondition` with the nine registered categorical stratum fields.

Production requires at least eight eligible base lineages. Exactly eight tasks and eight different
base lineages are selected. A smaller output is an error even though the legacy loader previously
accepted one through eight tasks.

### 5.4 Deterministic selection algorithm

For each candidate, construct categorical coordinates from `PreparedDesignCondition` after
excluding `base_lineage_key`, `calibration_sha256`, `learning_partition`, and
`registry_row_digest`. The remaining fixed axes are:

```text
application_family, problem_origin, host_family, fault_status,
distribution_regime, calibration_status, embedding_difficulty,
sampling_difficulty, decision_difficulty
```

These names are the exact fields in the current `PreparedDesignCondition` dataclass and
`_STRATUM_FIELDS` registry. An alias or historical field name is not accepted.

All candidates must share that exact coordinate schema, and every value must be a string or
Boolean. Let `load(f, v)` be the number of already selected tasks with level `v` on axis `f`, and
let `joint_load(s)` be the count in the complete joint stratum. At each of eight iterations,
consider only candidates from unselected base lineages and choose the lexicographically smallest
key:

```text
(
  - number_of_axes_whose_current_level_is_uncovered,
  max_axis_level_load,
  sum_axis_level_loads,
  joint_stratum_load,
  SHA256(domain, seed=0, source_manifest_sha256, stratum_digest,
         base_lineage, task_id),
  base_lineage,
  task_id
)
```

Update all loads and repeat. The output file stores the selected task IDs sorted, not in greedy
selection order. Input enumeration order must not affect the result. This rule is identified by
`if-gate1-exact-v1`; the producer, verifier, and Gate 1 loader use one shared implementation rather
than three copies.

### 5.5 Registry schema v1

The registry keeps the exact field set already accepted by the current Gate 1 loader:

| Field | Type | Required value or meaning |
|---|---|---|
| `schema` | string | `isingfold.exact-conformance-corpus` |
| `schema_version` | integer | `1` |
| `corpus_id` | string | exactly `if-gate1-exact-v1` |
| `purpose` | string | `bounded-independent-structural-and-program-conformance` |
| `source_corpus_manifest_sha256` | lowercase SHA-256 | exact prepared `manifest.json` bytes |
| `task_ids` | array of strings | exactly the eight sorted recomputed IDs |
| `max_tasks` | integer | `8` |
| `max_logical_variables_per_task` | integer | `12` |
| `record_digest` | lowercase SHA-256 | canonical payload digest |

No extra field is allowed. The fixed `corpus_id` carries the selection rule and seed contract. The
verification receipt records the implementation identity, source manifest pin, source population
census, selected marginal census, registry raw-file SHA-256, and a Boolean exact-reproduction
result. The Gate 1 loader records the same implementation identity in its nested authenticated
corpus identity.

### 5.6 Acceptance criteria

- Two producer runs over byte-identical prepared-v4 input emit byte-identical registry files.
- Shuffling public task input does not change the selected set or bytes.
- Producer, verifier, and Gate 1 loader select the same eight IDs.
- A self-consistent and externally pinned alternate task list is rejected.
- No target file or ground-partition receipt is opened.
- Every selected task has a witness, a distinct base lineage, and at most 12 logical variables.
- The existing Gate 1 exact checks pass on the published registry.

## 6. Protocol B: continuation-count resolution study

### 6.1 Objective and frozen configuration

The study chooses the smallest continuation count with conservative evidence that the full fixed
train population contains at least 128 independently resolved lineages under the production seed
schedule. It uses no validation or test outcome. It does not replace the final quality preflight.

Create `configs/quality_resolution_v1.json` with this exact scientific content:

```json
{
  "schema": "isingfold.quality-resolution-config",
  "schema_version": 1,
  "study_id": "if-quality-resolution-v1",
  "partition": "train",
  "production_lineages": "all",
  "study_lineages": 128,
  "tasks_per_lineage_cap": 1,
  "states_per_lineage_cap": 4,
  "evaluated_actions": 8,
  "candidate_continuations": [12, 16, 24, 32, 48, 64, 96, 128],
  "reward_reads": 256,
  "quality_seed": 907,
  "sampling_seed": 1907,
  "familywise_alpha": 0.05,
  "minimum_resolved_rows": 128,
  "minimum_resolved_lineages": 128,
  "lineages_per_shard": 2
}
```

The JSON object above is the complete canonical config file. The planning command requires its
external raw-file SHA-256. The config, seed, and domain are frozen before any quality outcome is
opened. Any change requires a new config, study ID, and receipt. In particular, a failed ladder
cannot be extended after reading its results under the same identity.

### 6.2 Plan first, without target access

```bash
export RESOLUTION_CONFIG=configs/quality_resolution_v1.json
export EXPECTED_RESOLUTION_CONFIG_SHA256='<out-of-band config file SHA-256>'

isingfold-rl plan-quality-resolution \
  --config "$RESOLUTION_CONFIG" \
  --expected-config-sha256 "$EXPECTED_RESOLUTION_CONFIG_SHA256" \
  --grid configs/rl_grid_hybrid_v1.json \
  --corpus runs/prepared_v4 \
  --selector runs/selector_v4 \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  --out runs/quality_resolution_plan_v1.json
```

The planner authenticates only public prepared-v4 data, the selector bundle, publisher identity,
and the target-free ground root. It constructs the complete all-train quality sampling plan first.
It then materializes every planned task, prefix, state and support fingerprint, evaluated action
index, and ordered continuation seed through index 127. It must not obtain the study sample by
calling the 128-instance quality planner, because that can select different tasks or states.

The full-plan lineage order and per-lineage row schedule are fixed before study sampling. A study
lineage's row specification must be byte-identical to its projection from the all-train production
plan.

### 6.3 Deterministic 128-lineage sample

Let `N` be the number of base lineages in the complete all-train plan. Require `N >= 1,024`. Give
each lineage this key:

```text
SHA256(canonical_json({
  "domain": "quality-resolution-lineage-sample-v1",
  "seed": 1907,
  "source_corpus_manifest_sha256": ...,
  "production_plan_digest": ...,
  "base_lineage": ...
}))
```

Select the first 128 lineages by `(key, base_lineage)`. This is a fixed, outcome-blind simple sample
without replacement from the finite planned population. Publish and externally pin the plan before
any worker opens the train targets. Host, fault, origin, application family, and difficulty
censuses are reported as descriptive coverage checks, but they do not enter the continuation
selection rule.

### 6.4 Progressive immutable delta stages

Do not execute 128 continuations for every action immediately. Run the fixed ladder in order:

```text
stage 0: continuation indices [0, 12)
stage 1: continuation indices [12, 16)
stage 2: continuation indices [16, 24)
stage 3: continuation indices [24, 32)
stage 4: continuation indices [32, 48)
stage 5: continuation indices [48, 64)
stage 6: continuation indices [64, 96)
stage 7: continuation indices [96, 128)
```

Each stage writes a new immutable delta directory. Earlier outcomes are never rewritten. A stage
may begin only when all shard receipts for the preceding cumulative count have been merged and the
selection rule has failed at that count. Stop at the first passing count. This sequential decision
does not inflate the confidence level because all eight candidate comparisons and the stop rule are
registered in advance and controlled as one family.

Within a stage, split complete base lineages into 64 shards of two sampled lineages. Apollo runs
assigned shard indices directly. Goose computation runs only through Slurm. The same source,
runtime, plan, selector, corpus, publisher tree, ground root, and train authority must be
byte-identical on both hosts.

```bash
# Apollo, direct execution. Example even shard indices for one stage.
scripts/apollo_quality_resolution_shards.sh \
  "$RESOLUTION_PLAN" "$EXPECTED_RESOLUTION_PLAN_SHA256" \
  "$STAGE_INDEX" 0 62 2 runs/quality_resolution_shards_v1

# Goose, submitted from login and computed inside Slurm. Example odd indices.
sbatch --export=ALL,ISINGFOLD_RESOLUTION_PLAN="$RESOLUTION_PLAN",ISINGFOLD_EXPECTED_RESOLUTION_PLAN_SHA256="$EXPECTED_RESOLUTION_PLAN_SHA256",ISINGFOLD_RESOLUTION_STAGE="$STAGE_INDEX",ISINGFOLD_SHARD_FIRST=1,ISINGFOLD_SHARD_LAST=63,ISINGFOLD_SHARD_STEP=2,ISINGFOLD_RESOLUTION_OUTPUT=runs/quality_resolution_shards_v1 \
  scripts/goose_quality_resolution_shards.sbatch
```

Each launcher expands to the exact worker command:

```bash
isingfold-rl run-quality-resolution-shard \
  --plan "$RESOLUTION_PLAN" \
  --expected-plan-sha256 "$EXPECTED_RESOLUTION_PLAN_SHA256" \
  --stage-index "$STAGE_INDEX" \
  --shard-index "$SHARD_INDEX" \
  --corpus runs/prepared_v4 \
  --selector runs/selector_v4 \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  --device "$QUALITY_SELECTOR_DEVICE" \
  --threads 1 \
  --out "$SHARD_OUT"
```

### 6.5 Compact study evidence and exact verification

Resolution-study deltas must not copy the full production quality-v7 continuation receipt into a
second large corpus. For each planned continuation, the compact row retains:

- plan, stage, lineage, task, state, and action identity;
- continuation index and seed;
- bounded reward and valid-return flag;
- the source continuation receipt's canonical digest;
- selected embedding digest and the four compiled program digests;
- deterministic runtime and implementation identity;
- record digest.

The resolution worker executes the real continuation and computes those values. A distinct
`verify-quality-resolution-shard` command reruns every assigned continuation and requires exact
equality of reward, validity, selected embedding digest, program digests, and full continuation
receipt digest. Its immutable verification receipt binds the source delta manifest raw SHA-256 and
record digest. Generation and verification processes must not share an in-memory result.

The compact evidence is a design artifact only. It cannot be loaded as quality-v7 warm-start data.
Production quality-v7 rows are generated again at the selected denominator and later receive one
complete exact replay.

### 6.6 Plan, shard, and study-receipt schemas

#### Resolution plan v1

`isingfold.quality-resolution-plan` version 1 contains exactly:

- study and config identity, raw config SHA-256, grid SHA-256, and grid resolution minima;
- prepared manifest SHA-256 and record digest;
- publisher, target-free ground-root, selector, normalizer, context, and implementation identities;
- the complete production protocol and complete all-train plan digest;
- the sorted all-train lineage and task census;
- all planned row identities and ordered seeds through continuation index 127;
- the sample rule, seed, 128 selected lineages, and descriptive stratum census;
- the candidate ladder and delta ranges;
- the 64 exact whole-lineage shard assignments;
- the canonical payload `record_digest`.

#### Resolution delta shard v1

Each directory contains `records.jsonl` followed by `manifest.json`. The manifest schema
`isingfold.quality-resolution-delta-shard` version 1 contains exactly:

- plan raw SHA-256 and record digest;
- stage index, lower and upper continuation indices, shard index, and shard count;
- assigned lineages, task IDs, row IDs, and their set digests;
- source corpus, publisher, train target access, ground partition, selector, context, implementation,
  and runtime identities;
- record count, raw records SHA-256, record-set digest, and continuation count;
- completion Boolean and canonical `record_digest`.

#### Resolution verification shard v1

`isingfold.quality-resolution-verification-shard` version 1 contains the delta manifest identity,
the same authority and runtime identities, exact assigned row and continuation counts, replayed
count, matching count, mismatch list, `pass`, and `record_digest`. Publication requires `pass=true`.

#### Resolution study receipt v1

Merge one cumulative stage only after receiving every delta shard and every externally pinned
verification receipt for all stages through that count:

```bash
isingfold-rl merge-quality-resolution \
  --plan "$RESOLUTION_PLAN" \
  --expected-plan-sha256 "$EXPECTED_RESOLUTION_PLAN_SHA256" \
  --stage-root runs/quality_resolution_shards_v1 \
  --expected-shard-pin-registry "$EXPECTED_RESOLUTION_SHARD_PINS" \
  --out "runs/quality_resolution_stage_${STAGE_INDEX}_v1.json"
```

The output schema `isingfold.quality-continuation-resolution-study` version 1 contains exactly:

- plan, config, corpus, selector, context, quality-authority, and implementation identities;
- source delta and verification shard census with each raw-file SHA-256 and record digest;
- finite population size `N`, sample size `n=128`, and exact selected lineage IDs;
- the registered candidate ladder and familywise confidence specification;
- one result per completed candidate count, including sampled resolved rows, sampled resolved
  lineages `x`, unresolved lineages, and the simultaneous finite-population lower bound;
- selected continuation count or `null`;
- the complete production protocol that the selection authorizes;
- `advance`, stop reason, and canonical `record_digest`.

Every intermediate stage receipt is immutable. Only the first receipt with `advance=true`, or the
final `no-registered-count-qualified` receipt, is the terminal study receipt.

### 6.7 Exact finite-population selection rule

For candidate count `C`, define a sampled lineage as resolved when at least one of its planned rows
has a plausible-best set strictly smaller than its evaluated action set using exactly continuation
indices `[0,C)`. A planned lineage that yields no replayable row has value zero. Let:

- `N` be the number of lineages in the full all-train production plan;
- `n=128` be the fixed study sample;
- `x_C` be the number of resolved sampled lineages at `C`;
- `K=8` be the number of registered candidate counts;
- `alpha_C=0.05/K=0.00625`.

Treat the deterministic, preregistered hash order as the realized without-replacement
randomization. For an unknown finite-population success count `M`,
`X ~ Hypergeometric(N, M, n)`. The one-sided lower confidence bound is

```text
M_L(C) = min M in [x_C, N - n + x_C]
         such that P[X >= x_C | N, M, n] >= alpha_C.
```

For `x_C=0`, define `M_L(C)=0`. The tail is inclusive and points toward larger observed success
counts. Compute it by exact integer arithmetic using binomial coefficients and cross-multiplication;
do not add SciPy or use floating probability comparisons. Unit tests must cover the complete small
`N,n,x` state space against enumerated hypergeometric probabilities.

Concretely, for `alpha_C=1/160`, accept candidate population count `M` in the inversion exactly
when

```text
160 * sum[k=x_C..min(n,M)] choose(M,k) * choose(N-M,n-k)
  >= choose(N,n),
```

where terms outside the hypergeometric support are zero. Search increasing integer `M` and return
the first accepted value. This fixes both the tail direction and boundary equality.

Bonferroni over all eight registered counts gives simultaneous familywise coverage of at least
0.95, regardless of dependence from nested prefixes and optional stopping. Select the smallest
candidate `C` for which `M_L(C) >= 128`. Because every resolved lineage contains at least one
resolved row, this lower bound also implies at least 128 resolved rows in the same finite
population. The production preflight still checks both realized denominators exactly.

If no candidate qualifies at 128, publish `selected_continuations=null`, `advance=false`, and
`stop_reason=no-registered-count-qualified`. Do not choose 128 by default, reduce either threshold,
invent winners, or add a larger count after viewing results. A larger ladder requires a new
preregistered config and study identity.

### 6.8 Production binding

The publication form of `label-quality` requires the terminal resolution receipt and its external
raw-file pin. It reads the selected count from that receipt and rejects a separate manual count.
It verifies that all production fields match, then runs all train lineages:

```bash
isingfold-rl label-quality \
  --corpus runs/prepared_v4 \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  --selector runs/selector_v4 \
  --resolution-receipt "$QUALITY_RESOLUTION_RECEIPT" \
  --expected-resolution-receipt-sha256 "$EXPECTED_QUALITY_RESOLUTION_SHA256" \
  --instances 0 \
  --tasks-per-lineage 1 \
  --states-per-lineage 4 \
  --actions 8 \
  --seed 907 \
  --reward-reads 256 \
  --device "$QUALITY_SELECTOR_DEVICE" \
  --out runs/quality_shards_v7/...
```

Quality v7 remains unchanged. Its existing label protocol records the selected continuation count,
all-train request, seed, and implementation contract. A separate
`verify-quality-resolution-binding` command compares the externally pinned study receipt to the
quality-v7 manifest before preflight and gates. The study remains a design input rather than a new
field silently inserted into schema v7.

## 7. Selector-device parity and GPU policy

Before the resolution plan is sealed, execute an exact selector decision parity audit over the
complete public train task census and every state scheduled by the all-train quality plan:

```bash
isingfold-rl audit-quality-selector-device-parity \
  --corpus runs/prepared_v4 \
  --selector runs/selector_v4 \
  --quality-protocol-config configs/quality_resolution_v1.json \
  --cpu-device cpu \
  --accelerator-device cuda \
  --out runs/quality_selector_device_parity_v1.json
```

The receipt contains every row identity, CPU and CUDA selected strength index, selected embedding
digest, four compiled program digests, totals, mismatch list, runtime identities, and
`all_equal`. It is canonical, immutable, and externally pinned before the resolution plan.

- If every selected index and every compiled strength digest are equal, the resolution study,
  quality generation, and replay may pin selector inference to CPU. This is expected to scale well
  across many one-thread lineage workers.
- If any comparison differs, `Q^mu` has CUDA selector semantics. Keep selector inference on CUDA.
  Apollo may run several CPU-bound shard subprocesses sharing its one GPU. Goose requests one GPU
  per Slurm allocation and runs several one-thread shard subprocesses inside that allocation.
  Concurrency is fixed only after a post-freeze capacity benchmark.
- Training and PPO remain CUDA in both cases.

The device choice and parity-receipt pin are part of the resolution plan, every quality shard, every
replay shard, and the final preflight implementation identity. A device change creates a different
study and quality corpus.

## 8. Required capacity canary before full quality generation

Quality-v7 currently embeds a full terminal embedding and four compiled physical programs inside
each continuation receipt. This is scientifically auditable but storage-heavy. Existing
measurements justify a mandatory production-like capacity gate:

- a tiny valid-task measurement at 20 continuations and 256 reads observed 3.8 to 7.6 KB per
  receipt, median 4.9 KB;
- an old production-like portfolio task with 16 logical variables, a Zephyr host with 316 nodes and
  2,266 edges, a 25-qubit incumbent, and 256 reads observed five of five valid continuations, mean
  1.479 CPU seconds and mean 37,678 serialized bytes per receipt;
- using the full upper census `1,024 * 4 * 8 * C`, the latter measurement projects 393,216 receipts,
  13.80 GiB, and 6.73 serial CPU-days at `C=12`; 36.79 GiB and 17.95 days at `C=32`; 73.59 GiB and
  35.91 days at `C=64`; 147.18 GiB and 71.82 days at `C=128`.

These are measured canaries, not final estimates. New dense portfolio tasks, larger embeddings,
program size, invalid-return mix, filesystem behavior, and hardware can materially change them.

After the continuation count and selector device are frozen, run a deterministic canary on 16
publicly selected train lineages covering the largest public graph/program-size cells and all
available host, fault, and origin levels. The selection is outcome-blind and recorded in the
resolution plan. It executes the selected `C`, serializes real quality-v7 rows, and reports:

- raw bytes per row and per continuation receipt, mean, median, p95, and maximum;
- CPU seconds and wall seconds per continuation, mean, p95, and maximum;
- peak resident memory and temporary disk high-water mark;
- valid and invalid continuation counts;
- exact planned full-population upper census.

The capacity receipt computes guarded projections with a fixed safety factor 2.0:

```text
projected_bytes = 2 * max_observed_bytes_per_trajectory * upper_trajectory_census
projected_cpu_seconds = 2 * max_observed_cpu_seconds_per_trajectory * upper_trajectory_census
required_free_bytes = 3 * projected_bytes
```

Before canary target access, the operator must seal a capacity budget containing maximum artifact
bytes, maximum CPU-seconds, available worker count, maximum elapsed time, and scratch free bytes.
The canary passes only when all guarded projections fit that preregistered budget and observed free
space is at least `required_free_bytes`. No full production shard may launch on a failed or missing
capacity receipt. A budget change after seeing the canary creates a new execution identity and must
be disclosed; it cannot change the selected scientific continuation count.

The exact invocation is:

```bash
isingfold-rl quality-capacity-canary \
  --resolution-receipt "$QUALITY_RESOLUTION_RECEIPT" \
  --expected-resolution-receipt-sha256 "$EXPECTED_QUALITY_RESOLUTION_SHA256" \
  --capacity-budget configs/quality_capacity_budget_v1.json \
  --expected-capacity-budget-sha256 "$EXPECTED_QUALITY_CAPACITY_BUDGET_SHA256" \
  --corpus runs/prepared_v4 \
  --selector runs/selector_v4 \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  --device "$QUALITY_SELECTOR_DEVICE" \
  --out runs/quality_capacity_canary_v1.json
```

The capacity-budget schema `isingfold.quality-capacity-budget` version 1 contains exactly
`schema`, `schema_version`, `budget_id`, `maximum_artifact_bytes`, `maximum_cpu_seconds`,
`maximum_elapsed_seconds`, `available_workers`, `minimum_scratch_free_bytes`, and
`record_digest`. Every numeric budget is a positive integer in base units.

The canary schema `isingfold.quality-capacity-canary` version 1 contains exactly the resolution,
budget, corpus, selector, context, device-parity, train-authority, implementation, and runtime
identities; the fixed canary selection rule and 16 lineage IDs; actual row, action, continuation,
validity, byte, CPU-time, wall-time, memory, and disk censuses; the full-population upper census;
factor-2 guarded byte and CPU projections; factor-3 required free space; every individual budget
comparison; `pass`; and `record_digest`. The receipt and budget both require external raw-file
pins.

Receipt deduplication or a quality-v8 format may later reduce storage, but it is outside this
compatibility-preserving protocol and requires an explicit schema decision.

## 9. Scalable quality replay and preflight

### 9.1 Required order

The scalable order is:

1. generate immutable quality-v7 lineage shards using the selected `C` and all-train plan;
2. exact-replay each quality shard in parallel and publish an externally pinned replay shard;
3. merge quality-v7 shards by consuming their paired trusted replay evidence, without executing a
   second serial continuation replay;
4. run one global quality-preflight command whose internal worker pool directly executes a fresh
   complete replay of the canonical merged corpus and publishes the existing quality-preflight v2
   receipt;
5. externally pin the final quality-v7 manifest, preflight v2 file, and replay-bundle manifest
   before release gates.

A quality shard never enters training. Only the canonical merged quality-v7 corpus plus a passing
preflight v2 receipt can enter a grid cell.

### 9.2 Replay plan

```bash
isingfold-rl plan-quality-preflight-shards \
  --corpus runs/prepared_v4 \
  --selector runs/selector_v4 \
  --quality-shard-root runs/quality_shards_v7 \
  --resolution-receipt "$QUALITY_RESOLUTION_RECEIPT" \
  --expected-resolution-receipt-sha256 "$EXPECTED_QUALITY_RESOLUTION_SHA256" \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  --shard-count 64 \
  --out runs/quality_preflight_plan_v1.json
```

The planner authenticates the complete quality shard-header census without replaying stochastic
outcomes. It assigns whole lineages by their position in the canonical all-train plan modulo 64.
Every quality row key appears in exactly one replay shard. It records expected row and continuation
counts per shard so a missing, empty, duplicated, or overlapping result fails before merge.

### 9.3 Replay worker and evidence

```bash
isingfold-rl run-quality-preflight-shard \
  --plan runs/quality_preflight_plan_v1.json \
  --expected-plan-sha256 "$EXPECTED_QUALITY_PREFLIGHT_PLAN_SHA256" \
  --shard-index "$SHARD_INDEX" \
  --corpus runs/prepared_v4 \
  --selector runs/selector_v4 \
  --quality-shard-root runs/quality_shards_v7 \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  --device "$QUALITY_SELECTOR_DEVICE" \
  --threads 1 \
  --out "$PREFLIGHT_SHARD_OUT"
```

For each source row, the worker reconstructs the exact state/action envelope, validates every
stored seed and denominator, reruns every continuation, and compares the complete continuation
receipt digest, evaluator evidence, reward, validity, selected embedding digest, and program
digests. `evidence.jsonl` contains one compact row per source quality row with the source row digest,
ordered expected and replayed continuation-root digests, trajectory count, match count, and row
pass Boolean. It does not duplicate full programs.

The replay-shard manifest schema `isingfold.quality-preflight-shard` version 1 contains:

- plan, source quality-shard, resolution, corpus, selector, context, device-parity, quality
  authority, target-access, ground-partition, implementation, and runtime identities;
- exact shard index/count and assigned lineage, task, row, and continuation censuses;
- evidence file SHA-256 and record-set digest;
- replayed and matching continuation counts, mismatch list, `pass`, and `record_digest`.

Every replay shard requires a raw manifest SHA-256 recorded outside its directory before it is
used or transferred.

### 9.4 Replay bundle and quality merge

`merge-quality-preflight-shards` first validates all 64 replay shards and publishes a canonical
companion manifest, schema `isingfold.quality-preflight-replay-bundle` version 1. It binds the plan,
the exact quality-shard union, each replay-shard raw SHA-256 and record digest, every source row
digest exactly once, total trajectories, total exact matches, authority, implementation, runtime,
and a Merkle root over canonical replay-shard identities. The bundle is immutable and externally
pinned.

The quality merger then requires the replay bundle and its pin:

```bash
isingfold-rl merge-quality-labels \
  --corpus runs/prepared_v4 \
  --selector runs/selector_v4 \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  "${QUALITY_SHARD_ARGS[@]}" \
  --trusted-replay-bundle runs/quality_preflight_replay_bundle_v1.json \
  --expected-trusted-replay-bundle-sha256 "$EXPECTED_REPLAY_BUNDLE_SHA256" \
  --out runs/quality_v7
```

It still validates every quality row, seed, action envelope, denominator, shard assignment, source
hash, and exact row coverage. It may skip executing continuations only when the replay bundle proves
one matching replay for every continuation in every input row under the same authority,
implementation, context, selector device, and runtime. It streams canonical rows in lineage/task/
prefix/state order and must not retain the full corpus in memory.

If the bundle is absent, diagnostic merge may use the existing direct full replay. Publication
merge requires the bundle so exact stochastic work is parallelized rather than repeated serially.

### 9.5 Final quality-preflight v2 direct parallel replay

The strict v2 field set has no semantically valid field for an external replay-bundle root. It is
therefore incorrect to project a v2 receipt from independently produced shards while merely
asserting their aggregate counts. After canonical quality merge, one global command directly
replays the complete merged corpus through an internal bounded multiprocessing pool. The parent
process owns the authenticated inputs, assigns every whole lineage exactly once, receives exact
per-row results from its children, verifies complete coverage, recomputes all resolution
denominators, and alone publishes the receipt:

```bash
isingfold-rl quality-preflight \
  --corpus runs/prepared_v4 \
  --selector runs/selector_v4 \
  --quality-labels runs/quality_v7 \
  --min-resolved-rows 128 \
  --min-resolved-lineages 128 \
  --workers 64 \
  --out runs/quality_preflight_v2.json
```

`--workers` changes execution scheduling only. It is capped by the number of record-bearing
lineages, each child is single-threaded, child failures abort the parent, and child results are not
accepted from a previous invocation. This command does not consume the external replay bundle and
does not trust persistent shard assertions. It actually invokes the registered continuation and
evaluator once for every trajectory in the merged corpus during this invocation.

The final receipt remains `isingfold.quality-resolution-preflight` version 2. Its
`source_quality_*` identities name the merged quality-v7 artifact, `full_replay.mode` remains
`full-exact-continuation-and-evaluator`, and replayed and matching counts equal the exact trajectory
denominator. The separately pinned replay bundle remains the evidence that allowed quality merge to
avoid its former serial replay; it is not represented as evidence inside v2. A future workflow that
wants preflight itself to reuse persistent replay shards must introduce quality-preflight v3 and
update every strict consumer.

No confidence interval is computed at preflight. It is a complete deterministic census and exact
replay. `advance=true` only when all continuations match and both realized thresholds are met.
Failure receipts are retained, but cannot be consumed by gates or training.

### 9.6 Host scheduling

- Apollo executes assigned indices directly and must reject a Slurm environment.
- Goose uses `sbatch` and `srun` inside the pinned Apptainer image and must reject login-node
  computation.
- With proven CPU/CUDA selector parity, use up to 64 one-thread CPU replay workers. A single global
  preflight invocation runs on one host; Apollo and Goose may divide the earlier merge-trust replay
  shards, but they do not jointly contribute persistent results to one v2 receipt.
- Without parity, keep CUDA selector semantics and multiplex several one-thread CPU-heavy shard
  subprocesses per GPU. Apollo shares its one GPU only at the concurrency established by the
  capacity canary. Goose uses one GPU per Slurm allocation and the same registered subprocess
  count. Do not request one GPU per mostly CPU-bound continuation unless measurements justify it.
- A post-freeze throughput benchmark may choose concurrency, batch transfer size, and shard-host
  assignment. It cannot alter tasks, rows, actions, seeds, continuations, selector device, or any
  scientific outcome.

## 10. Project paths

The intended implementation footprint is:

```text
configs/quality_resolution_v1.json
src/isingfold/rl/data/exact_conformance.py
src/isingfold/rl/data/quality_resolution.py
src/isingfold/rl/data/quality_preflight.py
src/isingfold/rl/cli.py
scripts/apollo_quality_resolution_shards.sh
scripts/goose_quality_resolution_shards.sbatch
scripts/apollo_quality_preflight_shards.sh
scripts/goose_quality_preflight_shards.sbatch
tests/unit/test_rl_exact_conformance_protocol.py
tests/unit/test_rl_quality_resolution_protocol.py
tests/unit/test_rl_quality_preflight_shards.py
tests/integration/test_exact_conformance_to_gates.py
tests/integration/test_resolution_to_quality_preflight.py
```

Shared canonical JSON, strict parsing, file hashing, atomic publication, target authorization,
quality row validation, and implementation identity helpers must be reused. Do not copy those
security boundaries into scripts.

## 11. Code-style contract

The implementation should expose typed pure functions for deterministic decisions and thin CLI
adapters for I/O. Scientific decisions return complete receipts rather than printing unstructured
text. For example:

```python
@dataclass(frozen=True)
class ContinuationDecision:
    selected_continuations: int | None
    lower_population_bound: int
    familywise_alpha: float
    advance: bool


def select_continuation_count(
    *, population: int, sample: int, successes: Mapping[int, int]
) -> ContinuationDecision:
    """Apply the registered exact finite-population rule without I/O."""
```

Avoid hidden globals, unordered iteration, floating probability thresholds, unbounded worker
queues, and catch-all exception recovery. Error messages must name the failed artifact, expected
identity, and observed identity without exposing evaluator targets.

## 12. Test strategy

### 12.1 Exact conformance

- golden registry bytes for a fixed prepared-v4 fixture;
- order invariance, exact eight-task cardinality, distinct lineages, and logical-size cap;
- marginal-coverage greedy tie cases and SHA tie fallback;
- fixed corpus ID and seed, shared producer/verifier/loader result;
- rejection of an alternate but self-consistent and externally pinned list;
- wrong source manifest, wrong raw pin, missing witness, duplicate lineage, missing design fields,
  fewer than eight eligible lineages, unknown JSON fields, duplicate JSON keys, symlink, and race;
- target-open spy proving zero target and ground-partition access.

### 12.2 Resolution study

- mathematical regression proving no eight-arm row resolves for every `C <= 11`, including means
  1 versus 0, and proving possibility begins at 12;
- prefix nesting, so results at `C` equal the first `C` outcomes of every later stage;
- all-train plan projection equality for every sampled lineage, task, prefix, state, action, and
  seed;
- exact 128-lineage deterministic sample, order invariance, no duplicate lineage, and no target
  access by the planner;
- seed-domain separation and global collision checks;
- exhaustive small finite populations for the inclusive hypergeometric upper-tail inversion using
  exact arithmetic;
- Bonferroni family size fixed at eight and smallest-passing-count selection;
- no-qualified-count receipt with `advance=false`;
- progressive stage gap, overlap, retry, tamper, wrong plan pin, missing shard, duplicate shard,
  authority mismatch, and no-overwrite tests;
- train-only target access and explicit rejection of validation/test authority;
- CPU/CUDA selector parity pass and mismatch branches.

### 12.3 Capacity and scalable replay

- measured serialized-byte and CPU-time fixture calculations, upper census, safety factor, budget,
  free-space, and failed-capacity branches;
- exact whole-lineage assignment and every planned row exactly once;
- one quality shard paired with exactly one complete replay evidence set;
- continuation receipt, evaluator evidence, reward, validity, selected embedding, and program-digest
  tamper detection;
- source quality-shard, corpus, selector, context, device, implementation, runtime, target-access,
  and ground-authority mismatch rejection;
- missing, duplicate, overlapping, partial, failed, unpinned, and symlink replay shards;
- Merkle-root determinism and pin-registry order invariance;
- trusted replay merge parity against direct monolithic replay on a small fixture;
- streaming quality merge byte parity with the current canonical merge order;
- final strict preflight v2 field equality and compatibility with current gates, warm start, grid
  cell, and complete-system consumers;
- proof that planning and merging create no model or optimizer and do not open validation/test.

### 12.4 Verification commands

```bash
python -m pytest -q \
  tests/unit/test_rl_exact_conformance_protocol.py \
  tests/unit/test_rl_quality_resolution_protocol.py \
  tests/unit/test_rl_quality_preflight_shards.py

python -m pytest -q \
  tests/integration/test_exact_conformance_to_gates.py \
  tests/integration/test_resolution_to_quality_preflight.py

python -m pytest -q tests/unit/test_rl_gates.py \
  tests/unit/test_rl_cli_training.py \
  tests/unit/test_rl_quality_integrity.py

python -m ruff check src/isingfold/rl tests
python -m ruff format --check src/isingfold/rl tests
bash -n scripts/apollo_quality_resolution_shards.sh \
  scripts/goose_quality_resolution_shards.sbatch \
  scripts/apollo_quality_preflight_shards.sh \
  scripts/goose_quality_preflight_shards.sbatch
```

Run a two-host fixture before publication: assign disjoint lineage shards to Apollo and Goose,
transfer completed immutable outputs, merge twice with reversed input order, and require identical
final bytes and all exact replay matches.

## 13. Boundaries

### Always

- authenticate inputs before expensive work;
- select and shard only by immutable base lineage and public identities;
- preserve every inconclusive row and exclude it from ranking loss;
- use the selected denominator unchanged for all production lineages;
- retain failure artifacts and external pins;
- run the full compatibility and adversarial test suite before enabling the model grid.

### Requires explicit architecture approval

- changing an artifact schema or compatibility-chain version;
- changing the candidate ladder, confidence family, thresholds, sample size, or all-train rule;
- changing the continuation policy, reward reads, action cap, state cap, selector device, or seed;
- deduplicating or compressing semantic content in a way that changes quality-v7 bytes;
- adding a third-party statistics dependency;
- allowing trusted replay evidence to bypass any state, action, seed, or authority check.

### Never

- inspect validation or test outcomes during resolution selection;
- accept a hand-picked exact-conformance list because its hash is pinned;
- use `--instances 128` as the resolution study's independent production plan;
- treat `C <= 8` as a viable eight-action pilot for ranking signal;
- lower the 128-row or 128-lineage gates after observing results;
- invent a winner for a fully unresolved row;
- run Goose computation on the login node;
- change selector device without exact full-train parity evidence;
- let a global preflight receipt merely assert aggregate replay counts without binding exact replay
  shards.

## 14. Parallel implementation plan

The following streams can proceed in parallel after this spec and both ADRs are approved:

| Stream | Work | Dependency |
|---|---|---|
| A | exact selector, v1 producer, verifier, Gate 1 loader hardening, tests | prepared-v4 public loader |
| B | exact hypergeometric arithmetic, resolution config/plan/selection receipts, tests | quality plan helpers |
| C | progressive resolution worker and verification shards, Apollo/Goose launchers | frozen B plan schema |
| D | selector CPU/CUDA parity audit and capacity canary | full production plan identity |
| E | preflight plan, replay worker, replay bundle, streaming trusted quality merge | quality-v7 validator |
| F | direct multiprocessing preflight v2, gate/warm-start compatibility, end-to-end tests | canonical quality-v7 merge |

Dependencies impose this critical path:

```text
approve spec and ADRs
  -> implement A and B in parallel
  -> parity audit and resolution plan
  -> progressive study and selection
  -> production-like capacity canary
  -> all-train quality generation plus replay shards
  -> trusted streaming quality merge plus direct parallel preflight v2
  -> exact conformance Gate 1 and release gates
  -> 9 plus 18 model grid
```

No stream may edit another stream's schema without first updating this living spec.

## 15. Architecture conflicts and required reconciliations

1. The current exact-conformance loader authenticates schema v1 and task bounds but accepts any
   nonempty `corpus_id` and any externally pinned sorted task list. This violates deterministic
   selection authority. It must require `if-gate1-exact-v1` and recompute the eight IDs.
2. No current CLI produces or independently verifies the exact-conformance registry.
3. The current runbook quality example uses `--instances 128` while its gate requires 128 resolved
   independent lineages. That leaves no tolerance for one unresolved lineage. The example is
   identified as a template, but the paper workflow must change it to `--instances 0`.
4. The current quality CLI accepts a manual continuation count, defaults to 2, and has no resolution
   receipt input. For eight actions, every `C <= 11`, including the default and all values through
   8, is structurally incapable of resolving a row. Publication launch must require the externally
   pinned study receipt.
5. Calling the existing 128-instance quality planner for the study does not prove that sampled
   state/action/seed identities are projections of the all-train production plan. The new planner
   must construct all train lineages first and sample afterward.
6. The current quality merger and quality preflight each perform a serial full replay. At the
   all-train scale this duplicates the dominant work and can require tens to more than one hundred
   GiB of receipt payload. Trusted replay shards must be consumable by the quality merger so that
   merge is no longer serial. The strict v2 preflight then performs its own fresh complete replay
   through bounded multiprocessing.
7. Current preflight v2 has no field for replay-shard provenance. It therefore cannot claim that
   persistent replay shards prove its counts. This spec keeps v2 honest by making its global
   command directly execute every replay. The separately pinned replay bundle proves only the
   earlier quality merge. Reusing it as preflight evidence requires a deliberate preflight v3
   migration across all consumers, not an extra unregistered v2 field.
8. The current quality CLI permits CPU or CUDA selector execution without a complete device parity
   receipt. The selector device can change rare argmax decisions and therefore change `Q^mu`.
   Publication execution must implement the parity gate and pin one device semantic.
9. Quality-v7 repeats full terminal embedding and four compiled programs in each continuation
   receipt. Sharding fixes compute scalability but not the canonical storage expansion. The
   capacity gate is mandatory under v7; content-addressed deduplication requires a separately
   approved successor schema.

## 16. Risk register

| Risk | Failure mode | Required mitigation and trigger |
|---|---|---|
| Sampling randomization is not independent | A generator or analyst anticipates seed 1907 and changes lineage identities or contents | Reserve the domain and seed before quality outcomes, prohibit generator use, record the commit and external config pin; any prior reuse creates a new study identity |
| Eight tasks underrepresent a condition | Exact Gate 1 passes while a host or difficulty level is absent | Greedy marginal coverage over all authenticated design axes, publish the source and selected censuses, and keep scalable Gate 2 separate |
| Nested-stage corruption | A later stage changes an earlier reward, row, action, or seed | Immutable deltas, exact prefix equality, complete independent replay receipts, and rejection of overlaps or gaps |
| Statistical implementation error | Wrong hypergeometric tail or floating rounding selects an unsupported count | Inclusive upper-tail inversion with integer arithmetic, exhaustive small-population tests, and Bonferroni over all eight candidates |
| Selected count still misses realized gates | The simultaneous lower bound covers the finite plan but exact production outcomes differ | Require byte-identical study-to-production row and seed projections, all-train generation, then exact census preflight; no model starts on failure |
| Selector device changes `Q^mu` | CPU and CUDA choose different strength indices on rare inputs | Complete train-corpus parity receipt before CPU use; otherwise pin CUDA and multiplex CPU work per GPU |
| Quality-v7 exceeds storage or time | Repeated full programs make generation or replay impractical | Post-freeze 16-lineage canary, factor-2 guarded projections, factor-3 free-space requirement, progressive counts, and no launch on a failed capacity receipt |
| Trusted replay bypasses verification | A merger accepts aggregate counts without evidence for exact rows | Externally pinned replay bundle, Merkle root, every row and continuation exactly once, matching authority/runtime/device, and monolithic parity tests |
| Cross-host nondeterminism | Apollo and Goose replay the same seed differently | Byte-identical source and inputs, pinned runtimes, exact receipt-digest comparison, two-host fixture, and fail closed on any mismatch |
| Streaming merge changes canonical bytes | Scaling changes row order or manifest identities | Fixed lineage/task/prefix/state order and byte-parity tests against the current canonical merger |
| Worker retry overwrites evidence | A retry hides a failed or divergent first result | No-replace output, new attempt path, externally pinned winning shard, and explicit duplicate-attempt rejection |

Residual risk remains that quality-v7 storage is too large even after the smallest qualifying count.
The correct outcome is a documented capacity failure or a separately approved successor schema,
not a silent reduction in lineages, states, actions, reads, or continuation count.

## 17. Completion criteria before training

The model grid may start only when all of the following exist and pass:

1. a byte-reproducible exact-conformance v1 registry, verification receipt, and Gate 1 loader that
   rejects alternate pinned selections;
2. a pinned selector-device parity decision;
3. a pinned all-train resolution plan and a terminal study receipt selecting one registered count;
4. a passing post-freeze capacity receipt;
5. quality-v7 shards for all train lineages under the selected count and fixed seed schedule;
6. a complete externally pinned replay bundle with one exact matching replay per continuation;
7. one canonical merged quality-v7 corpus produced from those shards without a second serial
   stochastic replay;
8. a passing strict quality-preflight v2 receipt with at least 128 resolved rows and 128 resolved
   independent lineages;
9. passing exact conformance and release gates under the existing grid and authority chain;
10. all unit, integration, determinism, two-host, lint, format, and shell syntax checks in Section
    12.

Only then is starting the registered 9 plus 18 training grid paper-defensible.
