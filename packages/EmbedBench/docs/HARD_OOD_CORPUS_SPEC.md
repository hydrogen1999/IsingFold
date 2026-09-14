# Hard and Out-of-Distribution Corpus Specification

## 1. Objective

This specification defines a versioned hard and out-of-distribution corpus for EmbedBench.
The corpus supports three distinct experimental purposes:

1. train decision models on calibrated structural and solution-quality difficulty;
2. select a model on development data without using either base-test or OOD-test labels;
3. evaluate generalization to capacity pressure, larger hosts, unseen logical families, host
   defects, and certified failure mechanisms.

The existing `release_v1_1` corpus remains an immutable IID base corpus. It may be used for
architecture screening, ablations, and an IID test result. It is not, by itself, evidence of
hard-case or OOD generalization.

The words MUST, MUST NOT, SHOULD, and MAY are normative.

### 1.1 Release boundary

This document specifies a separate future or legacy hard/OOD benchmark profile. It is not
the generation manifest for the current `isingfold-corpus-v4` release. The v4 release makes
claims only about its registered topology/host-scale, fault, and distribution-regime shifts.
Graphcut, portfolio, and jobshop instances occur in training, validation, and test in v4, so
jobshop MUST NOT be described as an unseen-family OOD result for that release. The
unseen-family exclusions and matrix below apply only if this separate profile is built and
published under its own plan, split manifest, and release identity.

### 1.2 Scientific claims enabled by this corpus

If every acceptance gate in this specification passes, experiments may support claims about:

- quality-aware chain replacement under a fixed candidate-bank and compute contract;
- preservation of feasibility, residual connectivity, and host capacity;
- transfer across occupancy, logical density, chain length, window size, and host scale;
- robustness to explicitly serialized qubit and coupler defects;
- accuracy on randomized and composed versions of certified failure mechanisms;
- downstream solve probability under a frozen chain-strength and annealing protocol.

The corpus does not justify a claim of unrestricted minor embedding, arbitrary-size scaling,
or QPU advantage. Such claims require separate end-to-end and hardware evidence.

## 2. Design assumptions and boundaries

### 2.1 Assumptions

- A problem group, not a decision row, is the independent statistical unit.
- All seams, focus variables, starting embeddings, host variants, and topology realizations
  derived from the same logical Hamiltonian belong to one split.
- Structural feasibility is a hard constraint. Solution quality is compared only on exactly
  feasible candidates that satisfy the registered qubit budget.
- Quality means the registered maximum solve probability over exactly four chain strengths.
- Hardness is a vector of measured fields. It is never a free-text label or a single solver's
  failure probability.
- The base architecture grid produces a shortlist. Its trained weights are not the final
  hard/OOD checkpoints.

The corpus exposes two supervised tasks, `terminal_quality` and `partial_structural`. They use
separate target schemas and losses and MUST NOT be collapsed into one undocumented scalar. A
state may be referenced by one row of each task, but the rows share identity and split.

### 2.2 Always do

- Preserve every published corpus byte and publish changed bytes under a new release ID.
- Derive seeds from a release namespace so process order and HPC site cannot change data.
- Count quotas and confidence intervals by independent problem group.
- Retain all attempted decision states, including ambiguous and failed states.
- Verify every host, embedding, candidate bank, split, and label artifact before publication.
- Keep every locked target inaccessible until model selection and checkpoint hashes are
  frozen.

### 2.3 Ask before changing

- the four-strength schedule;
- the primary metric or qubit-budget ratios;
- any hard/OOD axis, quota, or acceptance threshold;
- the canonicalization used by a content digest;
- the model-selection rule after any validation result has been observed.

Any approved change requires a new specification version and new release ID.

### 2.4 Never do

- overwrite `release_v1`, `release_v1_1`, or their label artifacts;
- assign OOD membership with an IID hash threshold;
- use Minorminer actions as target labels;
- discard a row because its quality margin is small;
- reconstruct a defective host from only topology name and size;
- accept a heuristic energy as a certified ground-state energy;
- choose the best training seed after opening locked labels;
- tune candidate sampling, label fidelity, or evaluation tolerance on locked results.

## 3. Immutable base-corpus contract

The base input has the logical ID `embedbench-base-v1.1`. Its authoritative raw corpora are
the twelve JSONL files built in `runs/release_v1_1`. The authoritative quality split is the
problem-digest split, not an instance-ID split:

```text
data/release_v1/splits_quality_problem_v2.json
data/release_v1/splits_structural_v2.json
```

The hard/OOD release MUST record the SHA-256 of every base corpus and split manifest it uses.
It MUST NOT copy a manifest whose corpus digest disagrees with the referenced raw bytes.

The immutable base bytes are pinned below. These values were measured directly from the
Apollo `~/isingfold/EmbedBench/runs/release_v1_1` payload. The checked-in per-file manifests in
`data/release_v1/` describe an earlier byte generation and MUST NOT authenticate these v1.1
files.

| Base artifact | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `quality_chimera5_app.jsonl` | 79 | 347753 | `8b11660c068b626e4eae55c6d8d3ed31fdd335a48ebd682bd06727cc99b5410e` |
| `quality_chimera5_inkdrop.jsonl` | 467 | 1825147 | `15cc622b3d8d4cb4fd66a378a83d2d15a886f076344817515d2c250bedaa2cce` |
| `quality_chimera5_random.jsonl` | 50 | 167045 | `1a4403535fa4d3ecf15c4c9d89d60456b1bba2d0bf85f2012af9b9090d196368` |
| `quality_pegasus3_app.jsonl` | 196 | 1146306 | `f1d911990fed7773f624dcf1badf2c806c7a3d42a0fb1c33aeb246ff56a9e9ed` |
| `quality_pegasus3_inkdrop.jsonl` | 836 | 4237876 | `d2ffc17be0ab3770d24436d1e28335ed91711971303c35200183783bc4b9dfb3` |
| `quality_pegasus3_random.jsonl` | 101 | 472656 | `07871cdae0d8fb2858bc5c21af14dbee1c7e59807f724c54df52131d9dd1cd32` |
| `quality_zephyr2_app.jsonl` | 190 | 1165152 | `3bd996cef358e4b924bdc3d051e5815af34ec168b55af8abdd4db5fb4280e48d` |
| `quality_zephyr2_inkdrop.jsonl` | 878 | 4755319 | `b1b62bcf1a8637eb9bbb2ed2aefddbffb387fb451dc6145e37fdc1331e91cd7a` |
| `quality_zephyr2_random.jsonl` | 82 | 393094 | `3b82118599bfb10fe3300db11504d0dd15577574c12d08724b6d667c37278aca` |
| `structural_chimera5.jsonl` | 204 | 293822 | `540a1dbd3262d797dd35109fd3e0e74c8c9eebbc17472f1f9c8b4fd8f1cfc3cf` |
| `structural_pegasus3.jsonl` | 326 | 743472 | `86ee59a180c2ee649c517be70353744b1d0fcd65b068ce9f3b720e3d296a4cfe` |
| `structural_zephyr2.jsonl` | 301 | 729334 | `35b537d790b3f4aaa0184c038987c60bf943770f796609af8f829ceb71a9fdbb` |

The permitted split files are pinned independently:

| Split artifact | SHA-256 |
|---|---|
| `data/release_v1/splits_quality_problem_v2.json` | `cf3b7b44a9d3e95e1dc82b11870a9056492d4d3d7029f181e7737d07906a78f0` |
| `data/release_v1/splits_structural_v2.json` | `399b60f71f6af24e8217d1ff2aa360451a270a7c6e8c12b96660475be94b3a8f` |

`BASE_INPUTS.json` MUST reproduce this filename-to-digest mapping, add byte and row counts, map
every split entry to one of these bytes, and provide at least one content-addressed retrieval
URI. Before paper artifact publication, that URI MUST resolve outside a personal home
directory. A clean-checkout fetch command downloads to a content-addressed cache, verifies all
digests, and never writes into `release_v1` or `release_v1_1`.

The base corpus has these roles:

| Partition | Permitted use |
|---|---|
| base train | architecture screening and later hard-corpus retraining |
| base validation | base-grid selection and hard-corpus confirmation |
| base test | one confirmatory IID evaluation after the final selection lock |

No base test quality label may participate in architecture screening, hard-corpus design,
threshold calibration, retraining, or checkpoint selection.

### 3.1 Frozen legacy training snapshot

The legacy base-grid jobs identify their frozen staged source by `legacy_source_sha256`
`294caa56d0bb04f716163b143a995ee176c8e72dd209c5f5f77caef92d72a264`. This digest, rather than a
later Git commit or current working tree, is the training-source identity. The registered base
grid file is `configs/training_grid_quality_v2.json`, SHA-256
`10be13fe54669e3577c9c6a32f972d1cd94640f091bb262b03f302bd59bdc270`.

Hard/OOD implementation is additive. It MUST NOT edit, delete, rename, or overwrite any file
listed in the reconstructed legacy snapshot. New modules, scripts, configs, tests, and output
directories use the names defined in Sections 15 and 19. If a utility is needed, it is
implemented in a new hard/OOD module rather than changing legacy training behavior.

Before an existing base-screen result can seed the shortlist, a
`LEGACY_TRAINING_SNAPSHOT.json` MUST be reconstructed from the staged HPC roots and MUST bind
the expected source digest above, the sorted `(relative_path, file_sha256)` list for trainer,
model, preprocessing, metric, and split code, every staged dependency file, the exact trainer
argument vector recorded by each job, all result and checkpoint hashes, and the per-job
recorded source digest. Every job admitted to the grid MUST report the same expected digest.
A missing staged file, mismatched digest, untracked source file, or locally reconstructed
substitute invalidates that result. Semantic checkpoint validation MUST operate on captured
checkpoint and model-source bytes whose digests equal the inventory. It MUST NOT hash one file
view and load another.

The frozen launcher did not emit dependency-lock, runtime-environment, Goose-wrapper, or
Slurm-submission bindings in its receipts. These unsigned historical receipts are permanently
ineligible for a paper reproducibility claim and for final paper evaluation. Supplying
`--allow-unlocked-legacy` explicitly permits their use only as architecture-selection
evidence. Adding fields to a receipt after training, copying a dependency file into a staged
root, or synchronizing wrapper bytes cannot upgrade that scope.

The verifier requires one internally consistent historical command root per site. Its
site-qualified input roots establish logical ownership only and do not attest the physical
execution host. On Goose, the approved `scripts/goose_training_grid.sbatch` bytes are pinned by
SHA-256 `da45ee57952ecc2d6ffce15dda079c4d991863530cbcf6e893c232eeba9b6850` and checked only in the
Goose staged root. Apollo uses the direct `scripts/run_training_grid.py` launcher already bound
by the legacy source digest. Historical Slurm IDs remain receipt claims; the original `sbatch`
command, exported range and worker variables, array shape, and resource request are
unattested. No later document may present those inferred values as recorded launch facts.

The legacy digest algorithm is preserved exactly. It selects sorted
`src/embedbench/**/*.py`, `scripts/train_*.py`, `scripts/training_*.py`, and
`scripts/run_training_grid.py`; for each file it hashes the 8-byte big-endian path length,
UTF-8 relative path, 8-byte big-endian payload length, and raw payload bytes. The reconstructed
manifest also has a distinct `snapshot_manifest_sha256` over its canonical sorted
`(relative_path, file_sha256)` inventory. New hard/OOD artifacts use `source_digest` for that
canonical inventory digest. These three named fields are never treated as interchangeable.

## 4. Release identity and artifact layout

### 4.1 Release ID

The first release governed by this specification is:

```text
embedbench-hard-ood-v1.0.0
```

Any byte change to records, hosts, split assignments, candidate banks, labels, or protocol
documents requires a new semantic release ID. A patch version may correct metadata only when
record, host, split, and label bytes remain identical.

### 4.2 Repository metadata

```text
EmbedBench/data/releases/embedbench-hard-ood-v1.0.0/
  PREUNLOCK_MANIFEST.json
  RELEASE_MANIFEST.json
  BASE_INPUTS.json
  LEGACY_TRAINING_SNAPSHOT.json
  DATA_CARD.md
  SHA256SUMS.preunlock
  SHA256SUMS
  HARDNESS_SCHEMA.json
  LABEL_PROTOCOL.json
  TRAINING_PROTOCOL.json
  GENERATION_PROTOCOL.json
  SOLVER_PROTOCOL.json
  CONTINUATION_PROTOCOL.json
  ISOMORPHISM_CONTRACT.json
  source_bundles/
    generation.json
    minorminer.json
    cpp_baseline.json
    continuation_checker.json
    portable_runtime.json
  SEED_REGISTRY_ARTIFACT_MANIFEST.json
  COMPUTE_PLAN_DRAFT.json
  COMPUTE_PLAN.json
  profiles/
    hard_dev_v1.json
    mechanism_v1.json
    locked_ood_v1.json
  splits/
    hard_dev_v1.json
    mechanism_dev_v1.json
    mechanism_locked_v1.json
    locked_ood_v1.json
  commitments/
    LOCKED_LABEL_COMMITMENT.json
```

### 4.3 External data payload

```text
runs/embedbench-hard-ood-v1.0.0/
  hosts/
    <host_artifact_sha256>.json
  problems/
    <problem_sha256>.json
  records/
    hard_dev/<shard>.jsonl
    mechanism_dev/<shard>.jsonl
    locked_ood/<shard>.jsonl
  candidate_banks/
    <full_candidate_bank_sha256>.json.zst
  seed_registry/
    <stage_id>/
      SEED_REGISTRY_ROOT.json
      part-000000.jsonl
      ...
  labels/
    development/<shard>.jsonl
  certificates/
    ground_state/<certificate_sha256>.json
    mechanism_development/<certificate_sha256>.json
  manifests/
    <artifact>.manifest.json
```

The custodian-only payload uses a separate root:

```text
/restricted/embedbench-hard-ood-v1.0.0/
  locked_quality/<shard>.jsonl
  mechanism_locked/records/<shard>.jsonl
  mechanism_locked/labels/<shard>.jsonl
  mechanism_locked/certificates/<certificate_sha256>.json
  UNLOCK_RECEIPT.json
  ACCESS_LOG.jsonl
```

Locked quality labels, locked mechanism records, their exact structural labels, and their
mechanism certificates form the locked-target payload. They MUST live outside the source
checkout with read permissions restricted to the evaluation custodian. Only their
cryptographic commitment, protocol digest, expected row count, and expected candidate count
belong in the repository before unlock.

### 4.4 Required release-manifest fields

`RELEASE_MANIFEST.json` MUST contain:

```json
{
  "schema": "embedbench.release-manifest",
  "schema_version": 1,
  "release_id": "embedbench-hard-ood-v1.0.0",
  "preunlock_manifest_sha256": "<64 lowercase hex characters>",
  "preunlock_checksums_sha256": "<64 lowercase hex characters>",
  "generator_git_commit": "<40 lowercase hex characters>",
  "generator_source_sha256": "<64 lowercase hex characters>",
  "generator_tree_clean": true,
  "python_environment_sha256": "<64 lowercase hex characters>",
  "base_release": {
    "release_id": "embedbench-base-v1.1",
    "corpus_sha256": {},
    "split_sha256": {}
  },
  "seed_registry_terminal_root_sha256": "<64 lowercase hex characters>",
  "seed_registry_artifact_manifest_sha256": "<64 lowercase hex characters>",
  "hardness_schema_sha256": "<64 lowercase hex characters>",
  "isomorphism_contract_sha256": "<64 lowercase hex characters>",
  "solver_protocol_sha256": "<64 lowercase hex characters>",
  "training_protocol_sha256": "<64 lowercase hex characters>",
  "compute_plan_sha256": "<64 lowercase hex characters>",
  "profile_sha256": {},
  "split_sha256": {},
  "host_artifacts": [],
  "problem_artifacts": [],
  "record_artifacts": [],
  "candidate_bank_artifacts": [],
  "seed_registry_root_artifacts": [],
  "seed_registry_shard_artifacts": [],
  "ground_state_certificate_artifacts": [],
  "mechanism_certificate_artifacts": [],
  "development_label_artifacts": [],
  "locked_label_commitment_artifact": {},
  "locked_target_artifacts": [],
  "evaluation_artifacts": [],
  "label_protocol_sha256": "<64 lowercase hex characters>",
  "locked_label_commitment_sha256": "<64 lowercase hex characters>"
}
```

Every listed artifact entry MUST include relative path, SHA-256, byte count, record count,
and schema version. `SHA256SUMS.preunlock` covers every pre-unlock published file except itself
and is never changed. The post-unlock `SHA256SUMS` covers every final published file except
itself, including `SHA256SUMS.preunlock`, and is generated from the immutable publication
directory.

`python_environment_sha256` is the digest of a canonical environment object containing the
OCI image digest, Python implementation and ABI, exact lockfile digest, sorted installed
package names and versions, operating-system image digest, CPU architecture, and relevant
compiler/runtime versions. Host name, scheduler job ID, timestamps, and measured runtime are
provenance sidecars and do not enter this digest.

Publication has two manifests. `PREUNLOCK_MANIFEST.json` covers all public metadata, structure,
development labels, certificates, and the locked-label commitment. It lists locked payload
digests and counts but not a readable path. `RELEASE_MANIFEST.json` is created after evaluation,
includes every revealed locked-target and evaluation artifact, and records the byte-identical
pre-unlock manifest digest. Neither manifest is edited after publication: changing a covered
byte creates a new release ID, not an in-place revision.

`PREUNLOCK_MANIFEST.json` has schema `embedbench.preunlock-manifest` version 1 and contains the
same identity, environment, base, protocol, profile, split, host, problem, record, bank,
development-label, and public-certificate fields shown above, plus the locked commitment, but
no final-manifest predecessor fields, locked plaintext paths, or evaluation artifacts. The
final manifest's two predecessor digest fields above bind the raw pre-unlock manifest and
`SHA256SUMS.preunlock` bytes. Both schemas reject unknown or missing keys.

### 4.5 Locked-label commitment

A locked quality-label row represents exactly one `(partition, problem_group_id, state_sha256,
candidate_index, strength_index, block)` tuple and stores its success count, read count, and
protocol digest. Rows are sorted lexicographically by that tuple, serialized as canonical JSON
followed by one line-feed byte, and divided into consecutive shards of at most 100,000 rows.
No compression byte participates in the plaintext commitment.

`LOCKED_LABEL_COMMITMENT.json` MUST contain release ID, ordered partition IDs, every bound
problem, split, host, state, and bank digest, label-protocol digest, ordered plaintext shard
entries `(name, sha256, bytes, rows)`, total candidates, total strengths, total reads, creation
time in UTC, custodian ID, Ed25519 public-key fingerprint, and signature. Its
seed fields bind the public parent-root digest, every private stage-root digest, and the private
terminal-root digest used for generation. Its
payload entries also cover canonical locked mechanism records, exact labels, and proof files.
`payload_root_sha256` is the SHA-256 of the canonical ordered payload-entry array. The signature
covers the canonical commitment object with only the signature field omitted. Verification
reconstructs every plaintext shard, checks every entry and total, recomputes the root, and
verifies the signature before evaluation.

The commitment discloses no success count, probability, rank, winner, reliability class,
mechanism details, or aggregate.
The post-unlock manifest may add deterministic compressed archives, but their decompressed
canonical streams MUST match the committed plaintext shard digests.

### 4.6 Seed derivation

No sequential pseudorandom state may cross groups or shards. Every stochastic request has a
canonical seed request object:

```text
seed_key = SHA256(canonical_json({
  "release_id": release_id,
  "purpose": purpose,
  "partition": partition,
  "panel": panel,
  "cell": cell,
  "task_type": task_type_or_null,
  "problem_sha256": problem_sha256_or_null,
  "state_sha256": state_sha256_or_null,
  "candidate_index": candidate_index_or_null,
  "strength_index": strength_index_or_null,
  "label_stage": label_stage_or_null,
  "replicate": nonnegative_integer
}))
```

`purpose` is one of `problem`, `host`, `witness`, `solver_attempt`, `mechanism`,
`focus`, `candidate_sample`, `screen_label`, `refine_label`, `locked_label`, `decode_tie`,
`audit_label`, or `audit_bootstrap`. The six nullable fields have the exact policy below. `R`
means a non-null value is required; `null` means that any non-null value is invalid. A literal
in the table is the only permitted value. Generation and bootstrap purposes are task-agnostic;
their task type is null rather than a redundant spelling of information already fixed by the
purpose, panel, and cell.

| Purpose | `task_type` | `problem_sha256` | `state_sha256` | `candidate_index` | `strength_index` | `label_stage` |
|---|---|---|---|---|---|---|
| `problem` | null | null | null | null | null | null |
| `host` | null | null | null | null | null | null |
| `witness` | null | R | null | null | null | null |
| `solver_attempt` | null | R | null | null | null | null |
| `mechanism` | null | R | R | null | null | null |
| `focus` | null | R | null | null | null | null |
| `candidate_sample` | null | R | null | null | null | null |
| `screen_label` | `terminal_quality` | R | R | R | R | `screen` |
| `refine_label` | `terminal_quality` | R | R | R | R | `refine` |
| `locked_label` | `terminal_quality` | R | R | R | R | `locked` |
| `decode_tie` | `terminal_quality` | R | R | R | R | R |
| `audit_label` | `terminal_quality` | R | R | R | R | `audit` |
| `audit_bootstrap` | null | null | null | null | null | null |

For `decode_tie`, `label_stage` is exactly one of `screen`, `refine`, `locked`, or `audit` and
matches the sample whose decoded tie is being resolved. For label and tie requests, `replicate`
is the registered independent block ordinal. For generation it is the planned attempt or
realization ordinal, and for bootstrap it is the bootstrap replicate. `cell` is the immutable
fully qualified plan-cell identifier, not a display label. Consequently, changing shard count,
process count, execution host, or completion order never creates a new semantic request.
The focus and candidate-sampling requests deliberately have null `state_sha256`: focus choice
and offered-bank sampling happen before the final state identity, which binds the offered bank.
Their fully qualified `cell` includes the attempt and focus slots, avoiding a circular digest
while keeping distinct pre-state requests in distinct namespaces.
For `mechanism`, `state_sha256` is the already materialized parent-state digest, never the
digest of the transformed state being generated.

The registry is a chain of immutable stage deltas, not one mutable global file. A stage is
frozen before any stochastic operation assigned to that stage executes. Its root has schema
`embedbench.seed-registry-root`, version 1, and contains exactly `schema`, `schema_version`,
`release_id`, `stage_id`, `parent_root_sha256`, `entry_count`, `shard_entry_limit`,
`occupied_seed32_count`, `occupied_seed32_sha256`, and `shards`. The foundation root has a null
parent. Every later root stores the digest of the immediately preceding root, so its digest
transitively binds the complete lifecycle. Reusing a stage ID, changing release ID, or repeating
a canonical request from any ancestor is invalid.

Each stage stores only its new requests in `part-NNNNNN.jsonl`, with at most 50,000 entries per
shard. Entries are sorted globally within the stage by raw `seed_key` bytes. Every line is
`canonical_bytes(entry)` followed by one line-feed byte. A shard descriptor contains exactly
`relative_path`, `sha256`, `byte_count`, `record_count`, `schema_version`,
`first_seed_key_sha256`, and `last_seed_key_sha256`. Paths are consecutive from
`part-000000.jsonl`; ranges are disjoint and increasing. The root is canonical JSON without a
trailing line feed. It is limited to 8 MiB, and an entry line is limited to 1 MiB.

Each entry contains exactly `request`, `canonical_preimage`, `seed_key_sha256`, `seed_words`,
`seed32_required`, `seed32`, and `collision_counter`. `canonical_preimage` is the exact ASCII
JSON text produced by `canonical_bytes(request)`, whose bytes are UTF-8;
`seed_key_sha256` is its 64-character lowercase digest; and `seed_words` is the ordered array
of its eight unsigned big-endian 32-bit words. A native PCG request has
`seed32_required=false` and both 32-bit allocation fields null. A request whose frozen adapter
accepts only a 32-bit seed has `seed32_required=true` and both allocation fields non-null. The
adapter contract fixes this boolean before the stage root is written; changing it requires a
new release.

For each stage, new third-party requests are sorted by `seed_key`. Allocation starts with the
complete inherited occupied-32 set. The direct candidate is the first four digest bytes as an
unsigned big-endian integer. On collision, allocation tries
`SHA256(seed_key || uint32_big_endian(counter))`, starting at counter 1, until the value is
unused. The chosen counter and value are added to the occupied set before the next new
third-party request. Parent assignments are never recomputed, even when a child key sorts
before its parent key. PCG-only requests never enter the occupied set and therefore cannot
perturb a third-party seed. `occupied_seed32_sha256` is
`SHA256(canonical_bytes(sorted_occupied_seed32_array))`; its count and digest are cumulative.

`SEED_REGISTRY_ARTIFACT_MANIFEST.json` has schema
`embedbench.seed-registry-artifact-manifest`, version 1, and exactly `schema`, `schema_version`,
`release_id`, `terminal_root_sha256`, and `stages`. Each ordered stage item has schema
`embedbench.seed-registry-stage-artifact`, version 1, and exactly `schema`, `schema_version`,
`stage_id`, `visibility_class`, `parent_root_sha256`, `root_sha256`, `root_artifact`, and
`shard_artifacts`. Visibility is exactly `public` or `custodian_private`. The root and every
shard are full artifact entries that bind normalized relative path, digest, byte count, record
count, and schema version. Stage paths are complete and consecutive under
`seed_registry/<stage_id>/`. Stage IDs are unique safe path components. The ordered first
parent is null, every later parent equals the preceding root digest, and the final root equals
`terminal_root_sha256`.

The release manifest binds the artifact-manifest digest and independently stores the expected
terminal digest as `seed_registry_terminal_root_sha256`. Loading or verifying a chain requires
both externally supplied expected digests. The verifier reparses typed manifest values and
checks their canonical bytes, so changing even an allowed visibility value invalidates the
artifact-manifest commitment. Copying either expected digest from the untrusted artifact being
checked is invalid. The pre-unlock manifest binds every public root and shard. A custodian-only
stage is represented before unlock by its signed root and payload commitment, then added
byte-for-byte to the final artifact manifest after unlock.

Generation performs deterministic external merge sorting in chunks of at most 50,000
registrations, so the 1.7-million-entry quality plan is not materialized as one Python object or
one JSON document. The 50,000 bounds for both sort chunks and output shards are enforced by
constructors, writers, and readers; they are not caller-tunable maxima. The 8 MiB root bound is
also enforced before construction or publication, not only when a root is later loaded.

Verification requires an independently supplied expected root digest, streams one bounded line
at a time, retains only the cumulative occupied 32-bit set plus its count and digest, and checks
every byte, count, range, allocation, parent, and manifest artifact. It does not retain the list
of stage allocations or all seed keys. A stage with no new 32-bit allocation reuses its parent's
exact immutable occupied-set object rather than retaining another cumulative copy.
Verification summaries are constructor-sealed outputs of successful byte verification, but
Python object internals are not treated as a trust boundary. Every consumer securely rereads
each root and validates the complete parent-digest chain, common release identity, unique stage
IDs, cumulative occupied-set inclusion, absence of cycles, and termination at a null-parent
foundation. Both child writing and child verification additionally reread every parent shard
before using its occupied set or accepting its ancestry.

`VerifiedSeedResolver` is the only production lookup interface. It requires the independently
committed terminal-root digest, revalidates the complete summary chain before accepting it,
groups `resolve_many` requests by stage and shard, and verifies each selected shard once. It
keeps an explicit bounded LRU of verified shard contents, so a batch containing many keys in
one shard causes one shard read. RNG and third-party adapter helpers require a resolver rather
than a bare verification summary. The project RNG is
constructed only as
`Generator(PCG64DXSM(SeedSequence(tuple(seed_words))))`; a third-party adapter obtains only the
registered `seed32`. An unplanned request is a hard error, and code may not fall back to an ad
hoc integer seed. The in-memory monolithic registry is only a test-scale reference model and is
not part of the production API.

Verification and final publication open every ancestor directory relative to a held directory
descriptor with `O_NOFOLLOW`. Each file check opens the root or shard once, checks `fstat` on
that descriptor, and hashes and parses bytes from that same descriptor. Directory inventory is
streamed and fails immediately on an unexpected name before any shard is opened. Publication
atomically reserves a previously absent destination directory, links only new regular files
without replacement, and publishes the root last. Immediately before reporting success, the
writer securely reopens the lexical parent and requires its identity and visible target to
match the held descriptors. A renamed parent or a target created after planning is rejected;
an existing target is never replaced. The writer also retains a descriptor for its private
staging directory and removes that exact tree after failure, even if an ancestor was renamed.

Unknown or missing fields, duplicate requests within or across stages, duplicate seed keys,
mixed release IDs, noncanonical lines, unsorted ranges, altered parent digests, or any derived
field that does not recompute are invalid. Candidate, strength, label stage, and replicate
fields MUST be populated for every quality sample, so candidates, strengths, stages, and
independent blocks never intentionally share a stream. Golden vectors cover direct allocation,
same-stage collision resolution, child-stage collision resolution without parent renumbering,
the eight-word conversion, and PCG64DXSM output.

The required dependency order is:

| Registry stage | Identities already frozen | Requests executed after this root freezes |
|---|---|---|
| structure foundation | release, profiles, plan cells and ordinals | `problem`, `host` |
| structure solvers | verified problem and realized-host identities | `witness`, `solver_attempt`, `focus`, `candidate_sample` |
| mechanism development | verified parent-state identities | development `mechanism` |
| development labels | verified states and complete bank identities | `screen_label`, `refine_label`, development `audit_label`, and matching `decode_tie` |
| solver hardness | verified solver-profile cells and problem/host identities | any additional shared `solver_attempt` runs |
| locked public labels | public locked states, banks and label protocol | `locked_label`, locked-bank `audit_label`, and matching `decode_tie` |
| bootstrap | frozen cohorts and estimator cells | `audit_bootstrap` |

The custodian mirrors the necessary foundation, structure, and mechanism deltas for private
locked mechanisms, with the public terminal root as their ancestor. A child stage is not
planned until every content identity it names is verified, and it is immutable before any
request in that child executes. Thus no request names an identity produced by its own random
stream, and later state-dependent requests cannot renumber an earlier solver seed.

## 5. Statistical units and content identity

### 5.1 Problem identity

The canonical problem payload contains the sorted logical-variable IDs, sorted
integer-indexed `(u,h)` pairs, and sorted `(u,v,J)` triples with `u < v`. It excludes generator
family and arguments, ground-state energy, solver outputs, embeddings, topology, and labels.
Generator family and arguments remain required provenance fields outside the identity payload.

```text
problem_sha256 = SHA256(canonical_json(problem_payload))
problem_group_id = "problem-sha256:" || problem_sha256
```

Excluding generator provenance makes canonically identical Hamiltonians one statistical group,
even when two generators reach them independently. Excluding ground-state energy prevents a
later certification improvement from changing split identity. Coefficients MUST be finite
IEEE-754 binary64 values. Negative zero is normalized to positive zero, and every coefficient
is represented in the digest view by its lowercase hexadecimal `float.hex()` value under the
reserved `__float64_hex__` tag. Decimal formatting and input spelling therefore cannot change
identity.

Every declared variable has exactly one bias pair, including zero bias. Parallel logical
couplers are summed in exact dyadic arithmetic before binary64 conversion, and a resulting
zero coupler is omitted. Duplicate bias or coupler keys, self-couplers, undeclared endpoints,
and non-integer variable IDs are rejected rather than normalized silently.

Every problem also has `problem_iso_sha256` for leakage auditing under logical-node relabeling.
Construct a colored incidence graph with one vertex per logical variable and one vertex per
logical edge. A variable-vertex color is `(variable, h_hex)`; an edge-vertex color is
`(coupler, J_hex)`; each edge-vertex is adjacent to its two endpoint variable vertices. Sort
the distinct color tuples lexicographically and replace them by consecutive integer colors.
Canonicalize this colored graph with the vendored Traces canonical-labeling implementation
whose source digest is stored in `ISOMORPHISM_CONTRACT.json`, then hash its canonical color and
adjacency bytes. The release pins the implementation, compiler, command, and golden fixtures.
All cross-partition intersections by `problem_iso_sha256` MUST be empty.

### 5.2 Grouping rules

The following objects MUST share `problem_group_id` and split membership:

- every focus decision for the problem;
- witness, Minorminer, C++ baseline, and perturbed starting embeddings;
- every host topology or host size on which the same problem is evaluated;
- all repeated label seeds and chain strengths;
- all candidate-bank views of the same state.

A mechanism family also has `mechanism_parent_id`. All transformations of one sampled parent
state MUST remain in one split. Canonically identical states MUST NOT occur in more than one
partition, even when their parent IDs differ.

Mechanism states additionally carry `mechanism_iso_sha256`. It is computed by the same pinned
canonical-labeling implementation over a colored incidence representation of the realized
host, logical variables and couplers, focus role, window membership, frozen-chain membership,
and every numeric coefficient. One identically colored candidate-role vertex per candidate is
connected to its member host nodes; the original-candidate flag is a distinct color. The
canonical labeling removes host-node, logical-node, and non-original candidate order, but no
structural or numeric attribute. Cross-partition intersections by
`mechanism_parent_id` or `mechanism_iso_sha256` MUST be empty.

### 5.3 Record identity

```text
state_sha256 = SHA256(canonical_json({
  "problem_sha256": problem_sha256,
  "host_sha256": host_sha256,
  "starting_embedding": canonical_embedding,
  "focus": focus,
  "frozen_context": canonical_frozen_context,
  "window": sorted_window_nodes,
  "l_cap": l_cap,
  "q_cap": q_cap_or_null,
  "terminal_q_cap": terminal_q_cap_or_null,
  "candidate_protocol_result_sha256": candidate_protocol_result_sha256,
  "full_candidate_bank_sha256": full_candidate_bank_sha256_or_null,
  "offered_candidate_bank_sha256": offered_candidate_bank_sha256_or_null
}))
```

Labels and baseline decisions are excluded from `state_sha256`. Each row carries
`problem_group_id`, `problem_iso_sha256`, `state_sha256`, `host_sha256`,
`host_artifact_sha256`, both candidate-bank digests, `partition`, `panel`, and `cell_id`.
Every decision row also carries `candidate_protocol_result_sha256`. It is the SHA-256 digest
of the canonical bytes of the complete `embedbench.candidate-protocol-result` version 1
object. This binds every aligned per-candidate fact, status, cap, count, mapping, and bank
digest rather than authenticating only candidate node arrays.
Embeddings are sorted arrays of
`[logical_variable, sorted_chain_nodes]`; frozen contexts are sorted arrays of typed records;
all node collections are duplicate-free sorted arrays. Their complete JSON Schemas are part of
the release and reject unknown keys.

Each generation attempt has a separate `attempt_id`, computed from release ID, plan cell,
ordinal, starting-source slot, focus slot, and attempt seed. Multiple attempts may resolve to
the same `state_sha256`; only the lexicographically first attempt becomes a decision example,
while every attempt remains in the attempt ledger. An attempt row may have null state or bank
digests. A decision row may not.

## 6. Defective-host representation

### 6.1 Host artifact

Every realized host is stored exactly once as a content-addressed artifact:

```json
{
  "schema": "embedbench.realized-host",
  "schema_version": 1,
  "topology": "pegasus",
  "size": 3,
  "pristine_host_sha256": "<sha256>",
  "requested_defects": {
    "qubit_fraction": 0.02,
    "coupler_fraction": 0.05,
    "seed": 123456789
  },
  "removed_nodes": [1, 7],
  "removed_edges": [[2, 8], [4, 9]],
  "nodes": [0, 2, 3],
  "edges": [[0, 2], [2, 3]],
  "realized_statistics": {
    "qubit_fraction": 0.02,
    "coupler_fraction": 0.0498,
    "component_count": 1,
    "largest_component_fraction": 1.0
  },
  "host_sha256": "<sha256>",
  "host_artifact_sha256": "<sha256>"
}
```

The example arrays are abbreviated. Node IDs MUST be unsigned 64-bit integers and sorted. Each
edge MUST be sorted within the pair, and the edge list MUST be lexicographically sorted.
`host_sha256` hashes only the canonical realized-graph identity object
`{"schema":"embedbench.host-graph","schema_version":1,"nodes":[...],"edges":[...]}`.
Thus, the same physical fault mask cannot acquire another identity from different provenance.
`host_artifact_sha256` separately hashes the full object above with both self-digest fields
omitted, preserving requested-defect and seed provenance.

`pristine_host_sha256` hashes a separate canonical
`{"schema":"embedbench.pristine-host","schema_version":1,"nodes":[...],"edges":[...]}`
object generated by the pinned D-Wave NetworkX topology implementation. Topology and size are
provenance and are checked against these bytes but do not replace them. `host_fill_fraction`
is occupied embedding nodes divided by realized-host nodes, after all defects.

Defect counts and identities are exact. For a pristine graph `(V0,E0)`, the qubit count is
`floor(requested_qubit_fraction * |V0|)`. Nodes are ordered by
`SHA256(host_seed_bytes || ":node:" || canonical_node_bytes)`, and the first requested count is
removed. Let `E1` be the pristine edges whose endpoints survive. The explicit coupler-fault
count is `floor(requested_coupler_fraction * |E1|)`. Edges in `E1` are ordered by the
analogous `":edge:"` digest and the first requested count is removed. `removed_edges` contains
only these explicit coupler faults, not edges lost incidentally with a removed node. A nonzero
request that rounds to zero is invalid for this release.

`host_seed_bytes` is the raw 32-byte `seed_key` for the host request. For schema version 1,
`canonical_node_bytes(v)` is exactly the unsigned 64-bit big-endian encoding of `v`, containing
eight bytes and no length prefix or separator. Before edge encoding, endpoints are ordered so
that `u < v`; `canonical_edge_bytes(u,v)` is
`canonical_node_bytes(u) || canonical_node_bytes(v)`. The node ranking key is
`(SHA256(host_seed_bytes || ASCII(":node:") || canonical_node_bytes(v)), v)`. The edge ranking
key is
`(SHA256(host_seed_bytes || ASCII(":edge:") || canonical_edge_bytes(u,v)), u, v)`. SHA-256
values in these tuples are compared lexicographically as their raw 32 bytes. The numeric node
or edge suffix is the mandatory deterministic tie-break if two digests are equal. After
selecting the first requested count in ranking order, `removed_nodes` is sorted by unsigned
node ID and `removed_edges` is sorted lexicographically by `(u,v)`.

Defect counts are computed by multiplying the requested IEEE-754 binary64 fraction by the
exact graph cardinality using binary64 round-to-nearest, ties-to-even, and then applying
mathematical floor. All registered graph cardinalities are below `2^53`.

Realized qubit and coupler fractions use `|V0|` and `|E1|` as their respective denominators.
If the realized host violates a profile connectivity predicate, the attempt is retained as a
rejected host and the planner advances to the next registered host ordinal. It MUST NOT redraw
defects under the same seed.

### 6.2 Host validation

For every record, validation MUST prove:

- the full artifact digest matches `host_artifact_sha256` and its realized graph matches
  `host_sha256`;
- all window nodes, chain nodes, and candidate nodes exist in the realized host;
- all recorded window, chain, and contact edges exist in the realized host;
- chains are connected and pairwise disjoint;
- every logical edge has at least one realized contact coupler;
- requested and realized defect fractions agree with deterministic removal;
- generation used the realized host, not the pristine reconstruction.

The trainer and evaluator MUST refuse a nonzero defect profile when the realized host artifact
is absent. Topology and size are descriptive fields and are never sufficient to reconstruct a
defective host.

## 7. Ground-state certification and quality provenance

### 7.1 Ground-state record

Every problem used for a quality label MUST reference:

```json
{
  "schema": "embedbench.ground-state-certificate",
  "schema_version": 1,
  "problem_sha256": "<sha256>",
  "status": "planted_proof",
  "energy": -12.5,
  "spin_assignment_sha256": "<sha256>",
  "method": "frustrated_loop_planting",
  "solver": null,
  "solver_version": null,
  "solver_command": null,
  "solver_seed": null,
  "deterministic_work_limit": null,
  "safety_timeout_seconds": null,
  "lower_bound": -12.5,
  "upper_bound": -12.5,
  "relative_gap": 0.0,
  "absolute_gap": 0.0,
  "energy_integer": -25,
  "energy_power_of_two": -1,
  "lower_bound_integer": -25,
  "lower_bound_power_of_two": -1,
  "upper_bound_integer": -25,
  "upper_bound_power_of_two": -1,
  "proof_artifact_sha256": "<sha256>",
  "checker_source_sha256": "<sha256>",
  "checker_environment_sha256": "<sha256>",
  "checker_command": ["python3", "scripts/verify_ground_state.py", "..."],
  "checker_exit_code": 0,
  "checker_log_sha256": "<sha256>",
  "checker_acceptance_sha256": null,
  "certificate_sha256": "<sha256>"
}
```

Allowed `status` values are `planted_proof`, `exact_enumeration`, `certified_optimal`, and
`uncertified_reference`.

`certificate_sha256` is computed over the canonical certificate object with that field
omitted. A `planted_proof` MUST include a digest of the planted assignment, the lower bound for
every planted term, and a machine-checkable equality showing that the assignment attains the
sum of those bounds. Merely knowing the assignment used by a generator is not a proof.

Every binary64 Ising coefficient is a dyadic rational. The verifier converts all coefficients
to a common power-of-two denominator and performs energy and bound comparisons with unbounded
integers. `energy_integer * 2**energy_power_of_two` is the exact value; decimal `energy` is
descriptive. Zero gap means exact equality of the integer lower and upper bounds. Relative gap
is null when the certified optimum is zero and never controls eligibility.

### 7.2 Eligibility rules

- Locked quality evaluation requires `planted_proof`, `exact_enumeration`, or
  `certified_optimal`, with exact equality of integer lower and upper bounds. Relative gap is
  diagnostic only.
- Exhaustive Gray-code enumeration MUST be used for `n_vars <= 24`.
- Larger application problems MUST use a solver that returns a matching lower and upper
  bound and a persisted certificate. The registered exact branch-and-bound verifier may
  expand at most 100,000,000 nodes and has no wall-clock stopping condition.
- Tabu, simulated annealing, or best-known energy may be stored only as
  `uncertified_reference`. Such problems may support structural training but MUST NOT support
  locked `p_solve` claims.
- The certificate MUST preserve solver name, solver version, command, seed, deterministic work
  limit, nullable safety timeout, bounds, and a digest of any proof or assignment file.

The proof artifact is stored under its content digest and contains either all planted-term
bounds, the exhaustive-enumeration transcript header and checked state count, or a complete
branch-and-bound tree whose leaves carry independently recomputable lower bounds. The release
pins the proof schema, checker source digest, environment digest, invocation, and successful
checker log. A solver's claim or output assignment without a checker-accepted proof is
`uncertified_reference`.

A checker-acceptance receipt authenticates one prior checker execution but is not itself a
mathematical proof. The publication gate MUST execute the independently registered checker on
the published proof bytes and reproduce the accepted receipt. If that executable checker is
unavailable, `certified_optimal` is disabled and the release may use only planted or replayed
exhaustive certificates for quality-eligible problems.

### 7.3 Exact structural target

For candidate chain `c`, let `Q_now(c)` be the total number of physical qubits occupied by the
frozen context and `c`, and let `L_now(c)` be their maximum chain length. These are immediate
partial-state quantities, not resource measurements of a completed embedding. The
immediate-validity vector records connectedness, chain disjointness, realization of every
required coupler whose other endpoint is already placed in the frozen context, and
`length(c) <= l_cap` separately. Logical edges from the focus to unplaced variables are
recorded as deferred edges and are certified only by exact continuation. `minor_valid(c)` is
true only when the four immediate predicates hold. `within_q_cap(c)` records
`Q_now(c) <= q_cap`, and `feasible_now` is their conjunction. If `q_cap` is null, it denotes no
finite immediate-occupancy cap:
`within_q_cap(c)=true` and `feasible_now(c)=minor_valid(c)`.

For an exact feasible completion `e`, define its future-connectivity tuple:

```text
K(e) = (largest_free_component_node_fraction,
        largest_free_component_edge_connectivity,
        -largest_free_component_articulation_count,
        minimum_logical_contact_multiplicity)
```

The node fraction is stored as an exact integer numerator and denominator and compared by
cross multiplication. An empty or singleton free graph has edge connectivity zero. The
frozen-context branch-and-bound search assigns every unplaced logical variable and returns the
best reachable terminal outcome. It separately enforces nullable `terminal_q_cap` on the
completed embedding:

```text
T(c) = (completion_feasible, K(completion), -terminal_Q, -terminal_L_max)
```

Tuples are maximized lexicographically. The search uses at most 28 free window nodes, chain
cap 6, and exactly 10,000,000 expanded search nodes. One expanded node means one invocation of
the continuation DFS state, including the root and terminal states; enumerating or rejecting a
candidate chain inside a state does not consume another node. It has no wall-clock stopping
condition.
If that deterministic node budget is exhausted, the completion fields are null and
`exact_completion_status="node_budget_exhausted"`; no heuristic value may occupy an exact
field. The exact structural winner is the smallest canonical candidate index attaining the
maximum `T(c)`.

Continuation certificates MUST be produced in one fresh subprocess per state by the portable
runtime and checker source bundle pinned in `CONTINUATION_PROTOCOL.json`. The protocol fixes
the absolute command template, empty inherited environment, input and output schemas, node
budget, counting rule, traversal rules, objective order, safety watchdog, and all resource
ceilings. The exact ceilings are 8 MiB of canonical input, 32 MiB of canonical output, 1 MiB
each for captured stdout and stderr, 512 MiB combined across retained checker and runtime
payloads, 1,024 ZIP members, 64 MiB of total uncompressed ZIP payload, and 16 path components
for retained and ZIP-member paths. Every ceiling is an exact positive integer field in the
protocol document. Changing any ceiling changes the externally committed protocol SHA-256.

The sole runner-facing protocol API reparses retained protocol bytes against independently
supplied protocol SHA-256 and release identity. In the same call it recomputes each checker and
runtime manifest from newly captured file records, checks the protocol-bound manifest roots,
and compares capsule identity before and after capture. The runner stages only the returned
detached snapshot and never reads mutable private bundle fields. Whole-capsule replacement,
self-rehashing, or mutation between validation and use invalidates the snapshot. The classic
ZIP end-of-central-directory record and every central-directory header are counted and checked
before Python materializes member metadata, so a forged member count cannot bypass the bound.
Declared and streamed uncompressed totals are both checked before a ZIP entrypoint is accepted.

The runner, not the caller, records the command, runtime attestation, termination mode, exit
code, exact protocol ceilings, and hashes of stdout and stderr. A watchdog, signal, capture
overflow, spawn error, or staging error is `environment_failure`, never exact-search
exhaustion. Once protocol and input identity are available, a failure while staging retained
bytes or canonical input produces a canonical failure receipt. The publication gate reruns the
same canonical input in another fresh process and requires byte-identical scientific
certificate output. Reading mutable live module paths or hashing a caller-supplied environment
object is not execution attestation.

This process contract is not yet a hermetic-runtime claim. A copied interpreter may still load
host standard-library, dynamic-loader, or shared-library bytes, and a descendant that creates a
new session can escape process-group termination. Publication remains blocked until both
boundaries have enforceable containment and independent tests.

For the best and second-best candidates, `structural_margin_kind` is the first unequal tuple
component: `feasibility`, `lcc_fraction`, `edge_connectivity`, `articulation`,
`contact_multiplicity`, `qubits`, `max_chain`, or `none`.
`structural_margin_value` is the signed numeric difference on that component. A8 counts a
ranking-eligible decision row as feasibility-decisive exactly when its margin kind is
`feasibility`; ties and aborted completions are outside that denominator and are reported.

For quota-directed partial-structural states, immediate `q_cap` is null.
`terminal_q_cap = Q_witness_active + q_cap_slack`, where `Q_witness_active` is the total qubit
count of the registered witness completion restricted to the frozen, focus, and unplaced
logical variables represented by the state. `q_cap_slack` is the exact integer 0 or 1 named by
the profile. The witness completion and its count are independently replayed before the cap is
accepted. Complete replacement states with no unplaced variable MAY use an immediate `q_cap`;
in that case it equals `terminal_q_cap` and both checks must agree. Each supervised row carries
exactly one `task_type`; fields owned by the other task are null. This structural continuation
target is not a downstream solve-quality label.

### 7.4 Quality-label protocol

The terminal target is:

```text
V(e) = max over k in {0,1,2,3} of p_solve(e; strength_unit * multiplier[k])
```

The label artifact MUST store all four per-strength outcomes as well as the maximum. It MUST
record reads, sweeps, beta schedule, programming rule, rescaling rule, decoding rule, tie
tolerance, and seed derivation. A scalar maximum without its four supporting outcomes is
invalid.

For every candidate, strength, stage, and independent block, the primitive stored outcome is
`(success_count, read_count, broken_chain_count, seed_key)`. Probabilities and intervals are
derived fields that validators recompute from these counts. Counts from screening and
refinement are never pooled; the two blocks within the same refinement or locked stage may be
summed only for the registered primary point estimate.

`strength_unit` is the binary64 mean of `abs(J)` over sorted logical couplers, or exactly 1.0
when there is no coupler. The four multiplier hex values are
`0x1.0000000000000p-2`, `0x1.965fea53d6e3bp-1`,
`0x1.428a2f98d7289p+1`, and `0x1.0000000000000p+3`.

The programming map divides each logical bias equally over its chain and each logical
coupling equally over all realized contact couplers. Every selected chain edge receives
ferromagnetic coupling `-abs(strength)`. The complete physical Hamiltonian is divided by its
largest absolute coefficient, unless all coefficients are zero. The sampler uses the pinned
D-Wave simulated-annealing implementation, beta range `[0.1,8.0]`, 200 sweeps, no coefficient
noise, and the seed streams in Section 4.6. Decoding is majority vote with independent seeded
coin flips for exact ties. A read is a solve when its decoded logical energy is at most the
certificate energy plus `1e-6` in programmed logical units.

`LABEL_PROTOCOL.json` MUST freeze the sampler package and source digest, container image
digest, compiler and math-library versions, RNG and tie-break implementations, exact
coefficient operations, strength bytes, all constants above, and golden output fixtures. It
is hashed into the pre-unlock manifest before any development or locked label is generated.

The registered protocol for this release is:

| Use | Independent blocks | Reads per block, candidate, and strength | Total reads | Sweeps | Strengths |
|---|---:|---:|---:|---:|---:|
| development integrity screen | 2 | 50 | 100 | 200 | 4 |
| development refinement | 2 | 500 | 1,000 | 200 | 4 |
| locked evaluation | 2 | 2,000 | 4,000 | 200 | 4 |

Development refinement MUST use fresh seeds independent of screening. Locked labels MUST use
fresh seeds independent of all development labels.

The integrity screen labels every offered development candidate and is used only to validate
schema alignment, empirical variance, throughput, and absence of constant or duplicated RNG
streams. It neither filters candidates nor changes refinement quotas. Both refinement blocks
label every offered candidate. Locked block A and block B are mutually independent. Section
10.4 uses them for cross-fit audit estimates, and Section 11 requires agreement across their
independent reliability decisions.

### 7.5 Frozen multitask loss and sampling contract

`TRAINING_PROTOCOL.json` is frozen with Stage 0, before development labels. Every shortlisted
architecture receives a terminal-quality head, a structural listwise-score head, a completion
feasibility head, and component-regression heads. The task targets remain separate.

For a quality bank, let `y_i` be the combined same-stage refinement estimate `V(i)`, and clamp
the predicted probability `p_i` to `[1e-6,1-1e-6]`. The terminal loss is the mean fractional
binary cross-entropy over candidates plus 0.25 times a pairwise hinge loss. The hinge includes
ordered pairs with `abs(y_i-y_j) >= 0.05`, has margin 0.05, and is averaged over included pairs;
an empty pair set contributes zero.

For a structural bank, listwise cross-entropy targets the canonical exact winner from Section
7.3. Completion-feasibility binary cross-entropy covers every candidate. Smooth-L1 loss with
delta 0.1 covers each non-null continuation component, after normalizing LCC fraction directly,
edge connectivity by realized-host maximum degree, articulation count by realized-host node
count, contact multiplicity by realized-host maximum degree, terminal qubits by realized-host
node count, and terminal maximum chain by 6. The structural loss is listwise cross-entropy plus
0.5 times feasibility loss plus 0.1 times the mean of available component losses.

The total hard-corpus loss is terminal loss plus 0.25 times structural loss. Legacy objective
variants retain only their frozen base auxiliaries: connectivity weight 0.1 for
`p_connectivity` and `full`, robustness weight 0.1 for `p_robustness` and `full`, and zero
otherwise. A missing task or auxiliary target is masked and never imputed as zero.

Each optimizer cycle contains two uniformly sampled base quality groups, one uniformly sampled
hard quality group, and one uniformly sampled partial-structural group from hard development or
mechanism development. Sampling cycles without replacement through a seed-specific permutation
of each group list and reshuffles only after exhausting that list. Candidate losses are averaged
inside a state, states inside a group, and the four task slots inside a cycle. AdamW uses
learning rate 0.001, betas `(0.9,0.999)`, epsilon `1e-8`, and weight decay `1e-4`, with no
scheduler and global gradient-norm clipping at 1.0. Training uses float32 and deterministic
PyTorch algorithms. Legacy architecture constructors retain their frozen initialization; new
linear heads use Xavier-uniform weight with gain 1 and zero bias. Confirmation runs 25 epochs
and selects the lowest validation-regret epoch, breaking ties toward the earlier epoch. These
rules and golden first-step fixtures are stored in the protocol. A protocol byte or loss-code
change requires a new release before observing hard validation.

## 8. Calibrated hardness schema

Every record MUST carry a structured `hardness` object validated against
`HARDNESS_SCHEMA.json`. Raw measurements are authoritative. Bands are deterministic views of
those measurements.

### 8.1 Required raw fields

```json
{
  "schema": "embedbench.hardness",
  "schema_version": 1,
  "release_id": "embedbench-hard-ood-v1.0.0",
  "sources": {
    "profile_set_sha256": "<sha256>",
    "plan_row_sha256": "<sha256>",
    "generation_protocol_sha256": "<sha256>",
    "problem_sha256": "<sha256>",
    "host_artifact_sha256": "<sha256>",
    "host_sha256": "<sha256>",
    "witness_receipt_sha256": "<sha256>",
    "witness_embedding_sha256": "<sha256>",
    "starting_receipt_sha256": "<sha256>",
    "starting_embedding_sha256": "<sha256>",
    "partial_context_receipt_sha256": "<sha256>",
    "partial_context_sha256": "<sha256>",
    "candidate_receipt_sha256": "<sha256>",
    "candidate_protocol_result_sha256": "<sha256>",
    "solver_profile_sha256": "<sha256>",
    "continuation_protocol_sha256": "<sha256>",
    "continuation_batch_certificate_sha256": "<sha256>"
  },
  "host": {
    "nodes": 128,
    "edges": 680,
    "component_count": 1,
    "defect_qubit_numerator": 3,
    "defect_qubit_denominator": 128,
    "defect_coupler_numerator": 34,
    "defect_coupler_denominator": 680
  },
  "logical": {
    "variables": 28,
    "edges": 70,
    "average_degree_numerator": 140,
    "average_degree_denominator": 28,
    "density_numerator": 140,
    "density_denominator": 756,
    "maximum_degree": 9,
    "component_count": 1
  },
  "starting_embedding": {
    "chain_count": 28,
    "total_qubits": 78,
    "host_fill_numerator": 78,
    "host_fill_denominator": 128,
    "mean_chain_length_numerator": 78,
    "mean_chain_length_denominator": 28,
    "maximum_chain_length": 5,
    "chain_length_stddev": 0.892
  },
  "partial_context": {
    "placed_variables": 24,
    "unplaced_variables": 3,
    "focus": 7,
    "occupied_qubits": 66,
    "q_cap": null,
    "terminal_q_cap": 79
  },
  "reference_candidate_index": 12,
  "candidates": [
    {
      "canonical_index": 0,
      "candidate": [31, 32],
      "immediate": {
        "connected": true,
        "chain_disjoint": true,
        "realizes_every_required_coupler": true,
        "within_l_cap": true,
        "minor_valid": true,
        "current_total_qubits": 68,
        "current_maximum_chain_length": 5,
        "within_q_cap": true,
        "feasible_now": true
      },
      "post_replacement_embedding": {
        "chain_count": 25,
        "total_qubits": 68,
        "host_fill_numerator": 68,
        "host_fill_denominator": 128,
        "mean_chain_length_numerator": 68,
        "mean_chain_length_denominator": 25,
        "maximum_chain_length": 5,
        "chain_length_stddev": 0.846
      },
      "residual": {
        "free_node_numerator": 60,
        "free_node_denominator": 128,
        "free_edge_numerator": 174,
        "free_edge_denominator": 680,
        "largest_free_component_node_numerator": 40,
        "largest_free_component_node_denominator": 128,
        "largest_free_component_edge_numerator": 129,
        "largest_free_component_edge_denominator": 680,
        "largest_free_component_edge_connectivity": 3,
        "articulation_count": 8,
        "largest_free_component_articulation_count": 5
      },
      "deferred_logical_edges": [[7, 9], [7, 18]],
      "continuation": {
        "exact_completion_status": "complete",
        "certificate_sha256": "<sha256>",
        "checker_acceptance_sha256": "<sha256>",
        "completion_feasible": true,
        "terminal_total_qubits": 77,
        "terminal_maximum_chain_length": 5,
        "largest_free_component_node_numerator": 35,
        "largest_free_component_node_denominator": 128,
        "largest_free_component_edge_connectivity": 2,
        "largest_free_component_articulation_count": 6,
        "minimum_logical_contact_multiplicity": 1
      }
    }
  ],
  "decision": {
    "focus_degree": 7,
    "window_nodes": 24,
    "window_edges": 49,
    "full_candidate_count": 318,
    "offered_candidate_count": 64,
    "exact_completion_numerator": 12,
    "exact_completion_denominator": 64,
    "exact_completion_fraction": 0.1875
  }
}
```

No field may contain `easy`, `medium`, `hard`, `base`, or another subjective string unless it
is a deterministic band defined below.

The `starting_embedding` is a complete authenticated embedding selected before any focus or
candidate operation. Its fill and chain fields are the only embedding measurements used for
starting-state quota assignment. The independently registered `witness_embedding` supplies the
terminal resource cap and is never replaced by the selected starting source. The
`partial_context` is derived from the selected starting embedding by a registered removal rule
and names all placed, focus, and unplaced logical variables. It must contain at least one
unplaced variable for `partial_structural`; a terminal-quality replacement state may have none.

`candidates` has exactly one entry for every offered candidate and is ordered by canonical
full-bank index. Every candidate entry is independently recomputed after replacing the old
focus chain. `post_replacement_embedding` measures only currently placed chains, including the
candidate; it is never described as a terminal embedding. Chain-length standard deviation is
the population standard deviation over those currently placed chains. Terminal total qubits,
maximum chain length, and future connectivity occur only inside checker-accepted continuation
records. A candidate whose exact search exhausts its deterministic node budget has null
continuation measurements.

For each candidate, the free graph is the subgraph of the realized host induced by nodes not
used by the frozen context or candidate. `free_node` divides free nodes by realized-host nodes;
`free_edge` divides edges in the free induced graph by realized-host edges. Largest-component
node and edge fractions use those same realized-host denominators. Edge connectivity is the
exact global minimum edge cut of the largest free component and is zero for fewer than two
nodes. `articulation_count` counts articulation vertices in the full free graph, while
`largest_free_component_articulation_count` counts them only in its largest component. If
several free components have the same maximum node count, the largest component is the one
whose sorted unsigned-64 node tuple is lexicographically smallest.

`deferred_logical_edges` is the sorted set of focus couplers whose other endpoint is unplaced.
It is not part of immediate `minor_valid`. Every such edge must be realized in a feasible
checker-accepted terminal completion. Edges to already placed variables are immediate required
couplers and must be realized by the candidate itself.

Every fraction also stores its integer numerator and denominator; the excerpt shows the two
used by primary bands and targets. Validators compare thresholds by exact cross
multiplication. Decimal fraction fields are descriptive recomputations and never decide a
boundary.

Every source digest is verified against immutable source bytes before computation. Verifiers
take one detached snapshot and never reread or return aliases to caller-owned dictionaries or
graphs. Stored continuation status, feasibility, terminal measurements, and certificate
eligibility are checker outputs, not accepted caller booleans. The complete hardness object is
content addressed, and changing a source or candidate result requires recomputation.

The full and offered candidate banks contain only candidates satisfying the immediate minor
constraints by construction. Quota-directed partial states also have no immediate qubit cap.
Consequently, an offered-bank feasibility fraction or immediate-budget-survivor fraction would
be identically one and MUST NOT be reported as a hardness axis. `exact_completion_fraction`
divides candidates with a checker-accepted feasible completion by all offered candidates only
when every offered completion search finishes. It and its numerator are null when any search
exhausts its deterministic node budget; its denominator remains the offered-candidate count.
Null is distinct from zero and MUST NOT satisfy a numeric profile predicate.

### 8.2 Registered bands

| Axis | Band | Exact rule |
|---|---|---|
| host fill | low | `fill < 0.45` |
| host fill | medium | `0.45 <= fill < 0.60` |
| host fill | high | `0.60 <= fill < 0.72` |
| host fill | extreme | `fill >= 0.72` |
| logical degree | sparse | `average_degree < 3.0` |
| logical degree | moderate | `3.0 <= average_degree < 4.5` |
| logical degree | dense | `4.5 <= average_degree < 6.0` |
| logical degree | very_dense | `average_degree >= 6.0` |
| residual LCC | open | `largest_free_component_node_fraction >= 0.50` |
| residual LCC | constrained | `0.25 <= largest_free_component_node_fraction < 0.50` |
| residual LCC | critical | `largest_free_component_node_fraction < 0.25` |
| defects | none | both realized fractions are zero |
| defects | light | qubits `<= 0.02` and couplers `<= 0.05`, with at least one nonzero |
| defects | heavy | qubits `> 0.02` or couplers `> 0.05` |

The host-fill band always uses the complete `starting_embedding`. The residual-LCC band is a
candidate-level view. A state-level profile predicate must name a candidate index explicitly;
the registered default is `reference_candidate_index`, which is the original focus chain's
canonical full-bank index. Aggregating residual measurements across candidates without a
registered reducer is invalid.

Band boundaries MUST NOT be recalibrated from locked data.

### 8.3 Solver hardness profile

Hardness remains a vector. Each problem manifest MUST include, for each registered solver:

- success count and success rate;
- deterministic attempt count and seed list;
- resolved algorithm parameters, work limit, safety timeout, and reference executor;
- total qubits, maximum chain, chain imbalance, and runtime distributions on successes;
- failure categories;
- solver and dependency versions.

Minorminer and the project C++ baseline are required. A problem has exactly one solver-neutral
schedule of 32 requests. Each request uses `purpose="solver_attempt"`, the frozen
`panel="structure"`, the frozen `cell="paired-baseline-comparison"`, the verified release,
partition, and problem digest, and a replicate in `[0,32)`. No solver identity occurs in the
seed preimage. Every request is registered with `seed32_required=true`; the
`collision-managed-uint32-v1` projection passes that same allocated unsigned 32-bit integer to
both adapters. Both run receipts bind the same ordered request keys, registry-entry digests,
and schedule digest. The schedule digest is exactly
`SHA256(canonical_bytes([request_0, ..., request_31]))`, where each item is the complete seed
request and array order is replicate order 0 through 31. A registry-entry digest is exactly
`SHA256(canonical_bytes(entry))` over the complete persisted entry object. Their algorithmic
budgets are solver-specific and MUST NOT be described as equivalent work.
`SOLVER_PROTOCOL.json` freezes every resolved Minorminer parameter after library defaults and
the C++ expansion limit, ordering rule, compiler, and binary digest. Each solver bundle has a
registered native entrypoint. The Minorminer profile uses 10 tries with at most 100 inner
rounds per try. The C++ profile uses 10 tries with at most 10,000 state transitions per try.
The protocol also fixes `seed_schedule_rule`, `seed_projection`, work-limit unit, exact binary
digest, software and dependency versions, and the single-core Apollo reference executor.
Minorminer's internal timeout is disabled because its empty return does not distinguish a
timeout from heuristic failure. A wall-clock limit is a 10-minute safety watchdog on the named
Apollo reference CPU; the external runner records a watchdog event as
`environment_failure`, not evidence of algorithmic infeasibility. Runtime is stored in a
noncanonical provenance sidecar and never affects record identity. OCT and ATOM adapters MAY
report `not_run` with a machine-readable reason, but the paper MUST NOT claim solver-universal
hardness without their profiles. An independent exact feasibility profile is mandatory when
`n_vars <= 16` and the realized host has at most 128 nodes; it uses a 50,000,000-node
deterministic search limit and reports `node_budget_exhausted` rather than a timeout claim.

Solver outputs are analysis metadata, never training targets and never inputs to a learned
model.

The model-input schema is a positive whitelist of physical problem, realized-host, current
embedding, focus, window, frozen-context, candidate, and permitted Hamiltonian features.
It excludes the complete selected starting embedding, witness chains and totals, partition,
panel, quota cell, profile ID, generator source, attempt source, solver profiles, certificates,
labels, every content digest, and every future or counterfactual field. The complete starting
embedding remains available only to quota validation. Training receipts hash the model-input
schema and the encoded tensors.

### 8.4 Closed provenance and verification order

Every digest in a release-valid hardness record satisfies exactly one rule: it hashes bytes
supplied to the verifier and matches an independently committed digest; it comes from an
already verified parent object; or it is recomputed inside the current verification call. A
digest copied from the document being checked is never evidence for that document.

External JSON boundaries accept canonical UTF-8 bytes and invert the registered binary64
hexadecimal representation exactly. They reject duplicate keys,
noncanonical serialization, unknown fields, non-finite numbers, and a boolean or float used as
an integer schema version. Source-bundle manifests list sorted relative paths, file byte counts,
file digests, and executable intent. Paths use a portable ASCII subset. Verification requires
POSIX `O_NOFOLLOW`, `O_DIRECTORY`, fd-relative open and stat, no-follow stat, and fd-based
streaming `scandir`; an unsupported platform fails closed. Source manifests are limited to
16 MiB, 10,000 files, 1 GiB of aggregate declared payload, and 32 path components. These
limits are checked from canonical manifest fields before opening the bundle root. Verification
opens every directory component
without following links, permits only directories that are proper prefixes of declared files,
and rejects an undeclared file before reading it. A declared file's opened size must equal its
committed byte count before any payload read. The verifier then reads each file once, checks the
independently committed manifest digest, and retains the exact immutable bytes used by the next
stage. A continuation consumer additionally rebuilds the canonical manifest from detached
records and requires before-and-after capsule identity. Native entrypoints must pass a static
ELF or Mach-O header, segment, and entry-point
preflight. This preflight is not execution attestation. The registered source-bundle roles are generation, Minorminer,
C++ baseline, continuation checker, and `portable_runtime`; generic `solver` and `runtime` roles
are invalid.

The release verification order is:

```text
profile set + plan row + verified seed registry
  -> problem + realized host + generation protocol
  -> independently replayed witness completion
  -> solver profile and tagged starting-source selection
  -> replayed focus, removal, window, and cap construction
  -> replayed full and offered candidate banks
  -> fresh-process continuation batch and publication rerun
  -> internal hardness recomputation and stored-byte equality
```

A plan row contains the complete `problem_seed_request` and its
`problem_seed_key_sha256`; a naked caller-supplied seed digest is invalid. The request is
exactly `purpose="problem"`, uses the row's registered release, partition, panel, and fully
qualified cell, has all purpose-inapplicable fields null, and uses the row ordinal as
`replicate`. The verifier resolves it through an exact `VerifiedSeedResolver` against the
independently supplied terminal root digest. This request is PCG-only and MUST have
`seed32_required=false`. The row also fixes `state_slot` and `focus_quartile`, each as an exact
integer in `[0,4)`. These values are plan inputs to later deterministic construction replay,
not trusted outputs copied from a record.

The witness receipt binds the release, profile set, plan row, problem, host, generation
protocol, exact `purpose="witness"` seed request, generation input, complete chains, total
qubits, and generation status. The tagged starting receipt binds either that witness, the
registered 32-attempt Minorminer profile, the registered 32-attempt C++ profile, or a registered
witness perturbation. Solver selection is replayed by `(total_qubits, maximum_chain_length,
sum_squared_chain_lengths, canonical_embedding_bytes)`.

The partial-context construction receipt binds the exact `purpose="focus"` seed request,
state slot, degree quartile, focus, frozen variables and chains, induced window, `l_cap`,
`q_cap`, slack, and the independently verified witness cap. These values are derived from the
verified plan and generation protocol. A public release verifier does not accept free focus,
window, cap, slack, profile name, solver-profile digest, or candidate seed bytes.

The candidate receipt binds the exact registered `purpose="candidate_sample"` request, full
candidate-protocol result, compressed sidecars, and decompressed canonical bank digests. Public
continuation APIs accept this verified candidate receipt, not a bare in-memory result. The
stored hardness bytes are accepted only when the verifier executes this complete chain and
recomputes the same document internally. Passing an allegedly recomputed or sealed Python
object to the verifier is not a trust boundary.

## 9. Exact corpus axes and quotas

Quotas count accepted independent problem groups. Decision rows never satisfy a missing group
quota. Generation attrition is reported separately and does not reduce the target quota.

Every accepted group has exactly one `quota_cell_id` and may satisfy exactly one matrix cell.
The hard-development union MUST contain 360 distinct `problem_group_id` values, the
single-mechanism union 700 distinct values, the composed-mechanism union 100 distinct values,
and the locked-OOD union 360 distinct values. Re-evaluating one problem on another host is a
paired diagnostic and never fills another quota. All within-partition intersections of
`problem_group_id` across quota cells MUST be empty. The corresponding unions also have the
same required cardinalities under `problem_iso_sha256`; a node-relabelled duplicate cannot
fill a second quota.

All 360 hard-development groups and all 360 locked-OOD groups MUST have a checker-accepted
zero-gap ground-state certificate and exactly one registered quality state. An uncertified
problem remains in the attempt ledger but does not fill one of these quotas. Mechanism groups
are structural-only under this release.

### 9.1 Hard-development matrix

`hard_dev_v1` contains exactly 360 problem groups:

| Host | Capacity transition | Logical stress | Long context | Total |
|---|---:|---:|---:|---:|
| Chimera-5 | 40 | 40 | 40 | 120 |
| Pegasus-3 | 40 | 40 | 40 | 120 |
| Zephyr-2 | 40 | 40 | 40 | 120 |
| Total | 120 | 120 | 120 | 360 |

Each 40-group cell contributes exactly 30 hard-training groups and 10 hard-validation groups.
Subcells MUST be balanced between train and validation to within one group.

#### Capacity transition

- realized starting-embedding fill: `0.55 <= fill < 0.65`;
- pristine host;
- 20 ink-drop and 20 connected-random logical problems per host cell;
- exact balance over `(chain_size, l_cap)` in `{3,4} x {4,6}`: 10 groups per combination;
- structural states balance `q_cap_slack` between 0 and 1;
- logical average degree remains in `[2.5, 4.5)` to avoid a density confound.

#### Logical stress

- realized fill: `0.35 <= fill < 0.55`;
- pristine host, `chain_size=3`;
- 10 groups each from connected random degree 4, connected random degree 6, 8-connected
  graph-cut, and portfolio;
- `l_cap` is balanced 20 at 4 and 20 at 6 per host cell;
- accepted problems have average logical degree at least 4.0.

#### Long context

- realized fill: `0.35 <= fill < 0.55`;
- pristine host;
- exact balance over `(l_cap, max_window_free)` in `{5,6} x {20,28}`: 10 groups per
  combination;
- 20 ink-drop and 20 connected-random logical problems per host cell;
- chain sizes 3 and 4 are balanced 20/20 without changing the four subcell totals.

### 9.2 Certified mechanism matrix

The seven required families are `dead_end`, `articulation`, `cut_capacity`,
`multi_neighbour`, `high_degree_trap`, `short_chain_trap`, and `ordering`.

| Partition | Variants per family | Families | Total groups |
|---|---:|---:|---:|
| mechanism development | 50 | 7 | 350 |
| mechanism locked | 50 | 7 | 350 |
| composed locked | 100 total | two-trap and three-trap | 100 |

The composed panel has exactly 50 two-trap and 50 three-trap groups. Pairings are balanced so
each base family occurs in at least 20 composed groups.

Each single-mechanism cell uses 50 independently seeded parent problems and retains exactly one
certified state from each parent. A parent contributes at most one accepted problem group to
the entire release. Development and locked parents have disjoint seed namespaces and zero
overlap by problem, problem-isomorphism, state, and mechanism-isomorphism digests. The composed
panel likewise uses 100 independent parent constructions, one accepted group per parent. Thus,
relabelings of one motif cannot fill a family quota.

### 9.3 Locked OOD matrix

This matrix belongs to the separate profile defined by this document, not to
`isingfold-corpus-v4`.

`locked_ood_v1` contains exactly 360 problem groups:

| Panel | Cell definition | Groups per cell | Cells | Total |
|---|---|---:|---:|---:|
| extreme capacity | Chimera-5, Pegasus-3, Zephyr-2 | 30 | 3 | 90 |
| host scale | Chimera-6, Pegasus-4, Zephyr-3 | 30 | 3 | 90 |
| unseen logical family | jobshop, OCT-friendly, OCT-adversarial | 30 | 3 | 90 |
| host faults | Chimera-5, Pegasus-3, Zephyr-2 | 30 | 3 | 90 |

#### Extreme capacity

- realized fill: `0.72 <= fill <= 0.82`;
- chain sizes 3 and 4: 15 groups each per topology;
- `l_cap=6`;
- no host defects;
- logical average degree in `[2.5, 4.5)`.

#### Host scale

- topology sizes are excluded from every training and validation corpus;
- realized fill in `[0.45, 0.65)`;
- at least 32 logical variables;
- `l_cap=6` and `max_window_free=28`;
- ink-drop and connected-random groups are balanced 15/15 per topology.

#### Unseen logical family

- no jobshop, OCT-friendly, or OCT-adversarial problem may occur in base training,
  hard training, or either validation set;
- each family has 10 groups on each of Chimera-5, Pegasus-3, and Zephyr-2;
- realized fill in `[0.45, 0.65)`;
- every quality-eligible problem has a certified ground state.

#### Host faults

- 15 light and 15 heavy groups per topology;
- light request: qubit fraction 0.02, coupler fraction 0.05;
- heavy request: qubit fraction 0.05, coupler fraction 0.10;
- realized fill in `[0.45, 0.65)` after defects;
- planting, candidate enumeration, and verification use the realized host artifact;
- the realized host's largest component contains at least 90 percent of retained nodes.
- all 90 fault groups have distinct `host_sha256` masks; a mask is never shared by two problem
  groups or by light and heavy subcells.

### 9.4 Problem sizes and starting embeddings

Within every non-mechanism cell, accepted problems use the five explicit `n_vars` values in
the profile. Each value occurs eight times in a 40-group cell and six times in a 30-group cell.

For each problem, generation attempts these starting-state sources:

1. a planted or target-guided ink-drop witness;
2. Minorminer with the 32 registered solver-profile seeds;
3. the project C++ baseline with the 32 registered solver-profile seeds;
4. two feasibility-preserving perturbations of the witness.

All successes and failures are retained in the solver profile. For each solver, its registered
starting embedding is selected without quality labels by `(total_qubits, maximum_chain_length,
sum_of_squared_chain_lengths, canonical_embedding_bytes)` over its successful attempts. A Minorminer
failure never removes a planted-feasible problem from the structural quota.

For random and application graphs, target-guided ink-drop receives the logical graph as its
required contact graph. This breaks the coupling in which witness states occur only for
ink-drop quotient graphs.

Each non-mechanism problem has exactly four structural state slots, one for each logical-degree
quartile. Starting sources rotate across witness, registered Minorminer, registered C++
baseline, and first canonical valid witness perturbation using a seed-derived cyclic offset.
If a source or quartile is unavailable, its attempt is recorded and is not backfilled. This
bounds generation at four state attempts per problem without confounding one source with a
fixed degree quartile.

Exactly one of the four decision-state slots per quality-eligible problem receives quality
labels. Selection uses the predeclared source preference Minorminer, C++ baseline, witness,
then perturbation, taking the first `candidate_bank_ready` slot and never creating a fifth
state. Such fallback states are retained but are excluded from the
primary `B/Q_MM` cohort. They remain in the `B/Q_resource` diagnostic cohort. Mechanism groups
contain exactly one certified decision state and receive structural labels only in v1.0.0.

## 10. Candidate-bank protocol

### 10.1 Full enumeration

The generator enumerates every duplicate-free connected candidate chain within the registered
window and `l_cap` that is disjoint from frozen chains and touches every required neighboring
chain. A candidate is a sorted array of unsigned 64-bit host-node IDs encoded as its array
length followed by big-endian eight-byte IDs. Every array-length prefix is itself an unsigned
64-bit big-endian integer. Canonical bank bytes begin with an unsigned 64-bit big-endian
candidate-count prefix followed by the candidate encodings in canonical order. The full bank
is deduplicated and sorted by `(chain_length, qubit_array)` and stored in a compressed sidecar.
Reconstruction alone is not a substitute for publishing completed full-bank bytes.

Full and offered bank digests hash the uncompressed canonical candidate array. The artifact
manifest separately hashes compressed file bytes and freezes codec, version, level, frame
options, and decompressed byte count. A validator must reproduce the canonical array digest
after decompression.

The record includes:

- `enumeration_status`, `enumeration_cap`, `enumerated_prefix_count`, and prefix digest;
- nullable `full_candidate_count` and `full_candidate_bank_sha256`;
- `offered_candidate_count` and `offered_candidate_bank_sha256`;
- mapping from every offered candidate to its full-bank index.

The closed computation record has schema `embedbench.candidate-protocol-result`, version 1.
Its `full_candidate_facts[i]` MUST describe `full_candidates[i]`; the tuple position is the
canonical full-bank index. Candidate facts are recomputed from the realized host, frozen
context, original focus chain, `l_cap`, and nullable `q_cap`. They are not independently
trusted metadata. The canonical record is published as a content-addressed artifact, and
`candidate_protocol_result_sha256` hashes the entire canonical object. The enclosing state
identity binds that digest together with all inputs needed to reproduce the facts.

Traversal considers subset size from 1 through `l_cap`, then lexicographic combinations of
the sorted eligible window nodes, applying validity predicates only after choosing a subset.
It stops only after finding candidate 4,001. It
then records `enumeration_status="aborted_cap"`, `enumerated_prefix_count=4001`, the prefix
digest, and null full and offered-bank fields. The attempt remains in coverage accounting but
does not create a decision row or ranking example. For a complete traversal, status is
`complete` and both full fields are non-null. A mechanism attempt is accepted only when its
full traversal completes at no more than 512 candidates.

### 10.2 Full and truncated banks

- If the full bank has at most 64 candidates, every candidate is offered.
- If it has more than 64 candidates, exactly 64 candidates are offered.
- The original candidate counts inside the 64 slots. It is never appended as a 65th item.
- Every mechanism state uses its full candidate bank.
- If the current focus chain is not a valid member of the full bank, the attempt is
  `invalid_generation` and has no decision row.

### 10.3 Label-free deterministic truncation

For a bank larger than 64, the offered bank is constructed before any quality evaluation:

1. Add the original chain.
2. Add the candidate minimizing `(length, -internal_chain_edge_count, canonical_index)`.
3. Add the candidate maximizing `(largest_free_component_node_fraction,
   largest_free_component_edge_connectivity, -canonical_index)`.
4. For each required logical neighbor, count distinct realized host edges from the candidate
   to that neighbor's frozen chain. Sort this count vector. Add the candidate maximizing
   `(minimum_count, total_count, sorted_count_vector, -canonical_index)`.
5. Partition remaining candidates by `(chain_length, minimum_count, total_count)`.
6. Visit nonempty strata in lexicographic round-robin order. Within a stratum, order by
   `SHA256(canonical_json([candidate_sample_seed_key_hex, candidate_node_array]))`.
7. Stop at exactly 64 unique candidates.
8. Sort the selected candidates back into full-bank canonical order.

All connectivity fields in steps 3 and 4 use the realized host and hypothetical replaced
embedding under the denominators in Section 8. Ties use the full-bank canonical index. The
algorithm MUST NOT consume `p_solve`, ground-state energy, a baseline-selected index, or any
learned score.

### 10.4 Full-bank audit

Exactly four hard-development groups per 40-group cell and exactly three locked-OOD groups per
30-group cell form the full-bank audit subset. Eligible groups have a complete quality state
with 65 through 512 full candidates. Membership uses the lowest
`SHA256(canonical_json([problem_group_id,"full-bank-audit"]))` values in the cell before any
label is generated. Exactly that one registered quality state per selected group is audited,
and every full-bank candidate in it receives both registered label blocks. Audit aggregation
weights states equally, which is equivalent to group weighting because there is one state per
selected group.

Audit eligibility is a pre-label subquota. The planner consumes ordinals until it can choose
the exact cell size satisfying all joint balances plus the audit count, then selects the
lexicographically smallest valid assignment. A cell is not frozen before this succeeds; no
group is added, removed, or replaced after any quality label exists.

Let `f_A` and `o_A` be the canonical first empirical winners of the full and offered banks in
block A, with `f_B` and `o_B` defined analogously. State recall is one half of the two indicators
`V_A(o_A) >= V_A(f_A)-1e-5` and `V_B(o_B) >= V_B(f_B)-1e-5`. Define
`d_AB = V_B(f_A)-V_B(o_A)` and `d_BA = V_A(f_B)-V_A(o_B)`. State omission regret is exactly
`(max(0,d_AB)+max(0,d_BA))/2`. This order clips each independent directional estimate before
averaging.

Across the 36-group hard-development audit as a whole, with cell results also reported, the
64-candidate bank MUST achieve:

- at least 98 percent recall of the full-bank quality winner;
- mean full-bank omission regret at most 0.01 in solve probability;
- 95th-percentile omission regret at most 0.03.

The point 95th percentile is the nearest-rank element `ceil(0.95*n)` after sorting state
regrets. The gate uses 10,000 stratified group-bootstrap replicates, resampling four groups
with replacement inside each of the nine cells and weighting cells equally. Its seed uses the
`audit_bootstrap` purpose. One-sided percentile bounds use sorted replicate index 499 for the
5 percent lower bound and 9,499 for the 95 percent upper bound under zero-based indexing. The
lower recall bound must be at least 0.98; upper bounds for mean and nearest-rank 95th-percentile
regret must be at most 0.01 and 0.03. Failure requires a new candidate-bank
protocol and release version before selection. Locked-OOD audit results use the identical
estimators but are report-only. They cannot trigger data replacement, threshold changes, a
new bank, or another model-selection round.

## 11. No-margin, failure, and coverage protocol

Generation never silently drops an attempt. The immutable attempt ledger assigns the first
applicable status in this precedence order:

```text
starting_embedding_unavailable
invalid_generation
enumeration_aborted
no_candidate
single_candidate
candidate_bank_ready
```

Only `single_candidate` and `candidate_bank_ready` create decision rows and therefore carry
non-null state and candidate-bank digests. Ground-state eligibility is a separate field. A
label-derived value never enters a prepublished decision record.

The label artifact assigns one of `not_requested`, `uncertified`, `label_failed`,
`single_candidate`, `ambiguous`, or `reliable`. For a bank of `K >= 2` candidates, each label
block constructs 95 percent family-wise simultaneous intervals for all `4K` binomial rates by
inverting exact binomial tests with Holm step-down correction. The method identifier is
`holm_exact_binomial_v1`; its inversion tolerance, implementation digest, and golden fixtures
are frozen in the label protocol.

For candidate `i`, its empirical value and bounds are the maxima of its four per-strength
estimates, lower bounds, and upper bounds. A block supports candidate `b` only when `b` is the
canonical first empirical maximizer and its lower bound exceeds every other candidate's upper
bound by at least 0.01. A bank is `reliable` only when both independent blocks support the same
candidate. Otherwise it is `ambiguous`. Raw counts, intervals, supported winner, and margins
from both blocks remain in the private or development label artifact regardless of status.

Training MAY use reliable examples directly and ambiguous examples through interval,
distributional, or consistency losses. Locked reporting MUST include:

- attempted-state count;
- constructed-state count;
- complete-enumeration count;
- ranking-eligible count with at least two candidates;
- ground-state-certified and label-complete count;
- reliable-ranking count;
- model regret on ranking-eligible, ground-state-certified, label-complete banks;
- ranking accuracy only on the reliable subset;
- a risk-coverage curve for any abstaining policy;
- every exclusion count with a machine-readable reason.

No method may report only the reliable subset without reporting its coverage against all
attempted states.

Every coverage rate names numerator and denominator. Single-candidate banks are reported as
trivial-choice coverage but never enter ranking regret or ranking accuracy. No-candidate,
aborted, invalid, uncertified, and label-failed attempts remain in the attempted denominator
and are excluded from regret only under their named reason. No alternative denominator may be
introduced after locked-label commitment.

## 12. Hand-test randomization and certification

Each mechanism variant is generated from a parent motif through a deterministic combination
of:

- host-node relabeling;
- valid placement in a registered hardware-family subgraph;
- hardware graph automorphism when available;
- reversal or permutation of symmetric logical roles;
- equivalent witness orientation;
- label-free host decorations that preserve the intended local mechanism.

Each transformed state is independently certified by exhaustive completion. It is accepted
only when:

- the hypothesized winner is a certified winner;
- the certified margin is positive;
- all candidates and completions satisfy the realized host;
- its canonical state digest is new;
- its `mechanism_iso_sha256` is new across the whole release;
- the transformation metadata reconstructs the state byte-for-byte.

Rejected transformations remain in generation accounting. Development and locked variants
use distinct seed namespaces. No locked mechanism state or canonical equivalent may be used
for training, hyperparameter selection, or threshold calibration.

Composed motifs MUST prove that both or all three constituent bottlenecks affect at least one
candidate continuation. A disconnected union of independent motifs does not qualify.

`profiles/mechanism_v1.json` enumerates every parent ordinal and freezes the motif template
digest, coefficient choices, topology and size, placement region, role permutation,
automorphism index, orientation, and decoration parameters before generation. Each numeric
choice is an explicit finite list indexed by the seed key, never an open range. A composed
certificate includes counterfactual ablations removing each constituent trap; every removal
must change the exact winner or exact margin, which proves that the constituents interact in
the certified decision.

## 13. Split construction

### 13.1 Development split

Within each 40-group hard-development cell:

1. generate exactly 40 accepted groups under the cell contract;
2. sort groups by `SHA256(problem_group_id || ":hard-dev-split")`;
3. assign 30 to `hard_train` and 10 to `hard_val`;
4. preserve subcell balance to within one group by deterministic constrained assignment.

If constrained assignment has more than one solution, select the lexicographically smallest
ordered assignment vector.

### 13.2 Held-out rules

These held-out rules are normative only for this separate unseen-family profile. They do not
retroactively describe the family coverage of `isingfold-corpus-v4`.

- `mechanism_locked` and `locked_ood` contain test groups only.
- Extreme-capacity fill `[0.72, 0.82]` is absent from all training and validation corpora.
- Chimera-6, Pegasus-4, and Zephyr-3 are absent from all training and validation corpora.
- Jobshop, OCT-friendly, and OCT-adversarial families are absent from all training and
  validation corpora.
- Nonzero host defects are absent from all training and validation corpora for the strict
  fault-OOD claim. A separate robustness-training experiment requires a new split profile.
- All problem-group, mechanism-parent, and canonical-state overlap counts across partitions
  MUST be zero.
- All `problem_iso_sha256` and `mechanism_iso_sha256` overlap counts across development and
  locked partitions MUST be zero.

The split manifest records both inclusion and explicit exclusion predicates. A generic IID
hash rule is insufficient for an OOD panel.

### 13.3 Frozen generation plans

Each profile JSON contains its generator module source digest, schema version, complete
allowed parameter lists, joint balance table, maximum ordinal, and acceptance predicates. The
planner expands it into one row per ordinal with every graph-family argument, target fill bin,
`n_vars`, chain size, `l_cap`, window cap, defect request, starting-source rotation, and seed
key already resolved. Generation consumes this plan and performs no hidden random draw.

For every 40-group hard-development cell, the plan has five equal-width fill sub-bands and
uses exactly eight accepted groups per sub-band. Capacity cells cross the two logical sources,
two chain sizes, and two chain caps exactly within each sub-band. Logical-stress cells use each
of their four families ten times and each chain cap twenty times, with every family-chain-cap
pair used five times. Long-context cells use every `(source, chain_size, l_cap,
max_window_free)` tuple either two or three times according to the exact 40-row balance table
stored in the profile; all one-axis marginals are 20/20 and all two-axis marginals are 10/10.

For every 30-group locked cell, the plan uses six groups in each of five equal-width fill
sub-bands. Binary axes have 15/15 marginals and every pairwise joint count differs by at most
one. Unseen-family cells use ten groups on each registered base-size topology. Problem sizes
are explicit profile values, use at least five distinct values per cell, and each occurs six
times.

For each cell, scan ordinals from 0 upward and retain every attempt. After each ordinal, solve
a deterministic constrained subset problem for exactly 40 or 30 groups, as applicable. Its
constraints include all joint balances, distinct problem and isomorphism identities,
ground-state eligibility, the full-bank audit subquota, and, for hard development, A8 margin
counts. Stop at the first prefix admitting a solution and choose the lexicographically smallest
ordered ordinal vector among its solutions. This is the only accepted set; earlier individually
eligible groups may be absent from it without being deleted from the attempt ledger. No cell
is frozen and no quality label is generated before this prospective selection finishes.
Exhausting `maximum_ordinal=10000` without a solution fails the release rather than inviting
manual replacement.

## 14. Compute and publication stages

Stages are sequential with immutable outputs. Shards within a stage may run in parallel.

### Stage 0: Plan and freeze metadata

- validate profile files and exact quotas;
- create the problem and host request plan, write and verify the structure-foundation seed root,
  then prohibit any edit to that root or its shards;
- compute preliminary artifact digests;
- freeze label and training protocols and publish split exclusions;
- sign `COMPUTE_PLAN_DRAFT.json` for structural generation.

### Stage 1: Generate and verify structure

- resolve problem and host RNG only through the verified foundation root, then generate and
  verify realized hosts and logical problems;
- derive requests that name those verified identities, freeze the structure-solvers root, then
  generate starting embeddings and all attempted states;
- enumerate candidate banks;
- after parent states exist, freeze the mechanism-development root before generating any
  randomized mechanism transformation;
- run exact structural labels and development-mechanism certificates;
- retain every failure status;
- freeze the development-label root only after every referenced state and bank digest verifies;
- verify quotas and split isolation;
- sign the realized `COMPUTE_PLAN.json` before any label or training launch.

### Stage 2: Development quality labels

- run 100-read screening for development states;
- run 1,000-read fresh-seed refinement for all offered development candidates;
- run full-bank refinement on the preselected audit subset;
- freeze and verify the development-bootstrap root before computing the registered audit
  replicates;
- compute label reliability without deleting ambiguous states.

### Stage 3: Solver hardness profiles

- freeze the solver-hardness root before launching any additional stochastic baseline run;
- run all registered baselines under fixed budgets;
- publish the full solver vector and failure taxonomy;
- verify that solver outputs were not added to model inputs.

### Stage 4: Private locked labels and public commitment

- after public locked-OOD structural bytes and all locked plans are frozen, the custodian
  verifies the public parent root and freezes the locked-label root plus the dependency-ordered
  private mechanism roots;
- the independent custodian then generates both 2,000-read blocks for every offered locked
  candidate and full-bank audit candidate;
- the custodian generates and certifies locked mechanism records from their frozen private
  plan, retaining all rejected parent attempts in its private ledger;
- the custodian verifies exact private quotas and zero isomorphism overlap against the public
  digest sets before commitment;
- the custodian validates canonical row order, exact planned counts, and protocol digests;
- plaintext labels, temporary shards, logs, and summaries remain in custodian-only storage;
- the custodian publishes and signs only `LOCKED_LABEL_COMMITMENT.json` as specified in
  Section 4.5;
- finalize `PREUNLOCK_MANIFEST.json` and `SHA256SUMS.preunlock` before confirmation begins.

### Stage 5: Model confirmation, final refit, and selection lock

- confirm the base shortlist on hard development data;
- choose the fixed final epoch count and refit four fresh seeds;
- freeze the final architecture, training recipe, four checkpoint hashes, metric code hash,
  and evaluation command in `selection.json`;
- freeze any final-evaluation bootstrap requests and bind the terminal seed-root digest;
- bind the already published locked-label commitment into `selection.json`;
- sign and publish a single `SELECTION_LOCK.json` before requesting unlock.

### Stage 6: Unlock and one-pass confirmatory evaluation

- verify the signed commitment and one-time selection lock, then atomically bind that exact
  selection digest and release the existing committed bytes;
- evaluate all four registered training seeds;
- aggregate paired results by problem group;
- publish the unlock receipt, access log, and one immutable evaluation bundle.

### Stage 7: Optional QPU panel

The QPU panel uses a separately versioned programming and sampling contract. It cannot replace
the locked surrogate panel and cannot retroactively alter any selection decision.

### 14.1 Access-control contract

The protected asset is any locked plaintext or derived statistic before a valid selection
lock. The custodian uses a separate operating-system account and storage directory owned with
mode `0700`; Apollo and Goose training accounts have no ACL entry, mount, environment secret,
or readable scheduler log for that directory. Every create, hash, read, copy, and unlock event
is written to an append-only signed access log. A successful read by a development identity,
an unsigned commitment, or a missing audit interval invalidates the release. This controls
accidental and ordinary project access; it does not claim protection from a malicious system
administrator.

Base v1.1 test labels already coexist with train and validation rows in the raw payload, so
they are an access-audited IID holdout, not a cryptographically blind panel. A custodian builds
immutable train-validation and test views from the pinned split. Training jobs mount only the
train-validation view, and receipts list every loaded `problem_group_id`. The test view is
mounted only by the Stage 6 evaluator. The paper MUST state this weaker base-test threat model.

### 14.2 Compute bound and retries

The accepted corpus exposes 2,880 non-mechanism structural state slots, 800 mechanism states,
360 hard-development quality states, and 360 locked quality states. These are accepted-output
counts, not generation-attempt bounds. The prospective prefix scan permits at most 210,000
non-mechanism problem ordinals and 160,000 mechanism parent ordinals across all cells, with up
to four state attempts per non-mechanism ordinal. Candidate enumeration may test at most
499,177 subsets per state before filtering, exact continuation search may expand 10,000,000
nodes per labeled candidate, and a non-planted ground proof may expand 100,000,000 nodes per
problem. Draft planning MUST account for these worst-case work units even when expected
attrition is much lower.

With 64 offered candidates,
the integrity screen uses at most 9,216,000 annealer reads, development refinement uses at
most 92,160,000, and locked evaluation uses at most 368,640,000. The 36 development and 36
locked full-bank audit states add at most 70,963,200 and 258,048,000 reads respectively when
each full bank reaches 512 candidates. These bounds include four strengths and both blocks.

Stage 0 creates signed `COMPUTE_PLAN_DRAFT.json` with ordinal, subset-test, exact-search,
proof-search, storage, and site-allocation ceilings for structural generation. After Stage 1
has frozen state and bank counts, `COMPUTE_PLAN.json` records actual work already consumed and
exact upper sums for every remaining label, training, audit, and evaluation job, including
`4 * candidates * reads`. Both objects contain schema version, release ID, source/profile
digests, measured pilot throughput, estimated CPU/GPU hours, peak memory, output bytes, shard
count, site assignment, approver key fingerprint, and Ed25519 signature. Both are covered by
the pre-unlock manifest.

Each stage refuses launch when its applicable signed plan exceeds the approved allocation or
when observed cumulative work exceeds a ceiling. A shard writes to a temporary
content-addressed directory, validates, then moves
atomically into place. At most three infrastructure retries use identical bytes and seeds; a
different successful output for one shard key is a determinism failure. Scientific failure
never triggers a changed seed or hidden replacement.

## 15. Required command surface

The following CLI surface MUST be implemented. Exact paths may change only before the first
release manifest is frozen.

```bash
# Fetch and verify the pinned base without changing its source directory.
python3 scripts/fetch_hard_ood_inputs.py \
  --manifest data/releases/embedbench-hard-ood-v1.0.0/BASE_INPUTS.json \
  --cache runs/artifact-cache

# Reconstruct and verify the frozen legacy grid snapshot from staged HPC artifacts.
python3 tools/legacy_provenance/freeze_legacy_training_snapshot.py \
  --expected-legacy-source-sha256 294caa56d0bb04f716163b143a995ee176c8e72dd209c5f5f77caef92d72a264 \
  --grid configs/training_grid_quality_v2.json \
  --staged-root apollo=/staging/apollo/EmbedBench_qv2_global_20260910 \
  --staged-root goose=/staging/goose/EmbedBench_qv2_global_20260910 \
  --allow-unlocked-legacy \
  --out data/releases/embedbench-hard-ood-v1.0.0/LEGACY_TRAINING_SNAPSHOT.json

# Validate profiles and create immutable shard and compute plans.
python3 scripts/plan_hard_ood.py \
  --release-id embedbench-hard-ood-v1.0.0 \
  --profiles data/releases/embedbench-hard-ood-v1.0.0/profiles \
  --compute-draft data/releases/embedbench-hard-ood-v1.0.0/COMPUTE_PLAN_DRAFT.json \
  --out runs/embedbench-hard-ood-v1.0.0/plans

# Generate one deterministic CPU shard.
python3 scripts/generate_hard_ood_shard.py \
  --plan runs/embedbench-hard-ood-v1.0.0/plans/hard_dev.json \
  --shard-index 0 \
  --out runs/embedbench-hard-ood-v1.0.0/staging/hard_dev-000

# Validate and merge shards without changing record order.
python3 scripts/merge_hard_ood.py \
  --plan runs/embedbench-hard-ood-v1.0.0/plans/hard_dev.json \
  --shards runs/embedbench-hard-ood-v1.0.0/staging/hard_dev-* \
  --out runs/embedbench-hard-ood-v1.0.0/records/hard_dev

# Generate and independently verify proofs, then build solver vectors.
python3 scripts/certify_hard_ood_ground_states.py \
  --release runs/embedbench-hard-ood-v1.0.0 \
  --protocol data/releases/embedbench-hard-ood-v1.0.0/LABEL_PROTOCOL.json
python3 scripts/profile_hard_ood_solvers.py \
  --release runs/embedbench-hard-ood-v1.0.0 \
  --protocol data/releases/embedbench-hard-ood-v1.0.0/SOLVER_PROTOCOL.json

# Recompute schemas, digests, quotas, splits, hosts, banks, and certificates.
python3 scripts/validate_hard_ood_release.py \
  --release runs/embedbench-hard-ood-v1.0.0 \
  --metadata data/releases/embedbench-hard-ood-v1.0.0 \
  --phase structure

# Freeze and sign exact remaining work from realized state and bank counts.
python3 scripts/finalize_hard_ood_compute_plan.py \
  --release runs/embedbench-hard-ood-v1.0.0 \
  --draft data/releases/embedbench-hard-ood-v1.0.0/COMPUTE_PLAN_DRAFT.json \
  --out data/releases/embedbench-hard-ood-v1.0.0/COMPUTE_PLAN.json

# Produce both development label stages. This additive command refuses locked partitions.
python3 scripts/label_hard_ood_quality.py \
  --release runs/embedbench-hard-ood-v1.0.0 \
  --partition hard_dev \
  --stages integrity_screen,refinement \
  --compute-plan runs/embedbench-hard-ood-v1.0.0/plans/COMPUTE_PLAN.json \
  --out runs/embedbench-hard-ood-v1.0.0/labels/development

# Gate the bank protocol using development labels only.
python3 scripts/audit_hard_ood_candidate_bank.py \
  --partition hard_dev \
  --release runs/embedbench-hard-ood-v1.0.0 \
  --mode development-gate

# Run under the isolated custodian account before model confirmation.
python3 scripts/custodian_label_hard_ood.py \
  --partitions mechanism_locked,locked_ood \
  --release runs/embedbench-hard-ood-v1.0.0 \
  --private-out /restricted/embedbench-hard-ood-v1.0.0 \
  --commitment-out data/releases/embedbench-hard-ood-v1.0.0/commitments/LOCKED_LABEL_COMMITMENT.json

# Freeze the public predecessor manifest before any confirmation run.
python3 scripts/finalize_hard_ood_preunlock.py \
  --release runs/embedbench-hard-ood-v1.0.0 \
  --metadata data/releases/embedbench-hard-ood-v1.0.0

# Confirm the base shortlist on the hard-development split.
python3 scripts/confirm_quality_shortlist.py \
  --base-grid-snapshot data/releases/embedbench-hard-ood-v1.0.0/LEGACY_TRAINING_SNAPSHOT.json \
  --base-selection runs/training_grid_quality_v2/selection.json \
  --base-splits data/release_v1/splits_quality_problem_v2.json \
  --hard-splits data/releases/embedbench-hard-ood-v1.0.0/splits/hard_dev_v1.json \
  --top-k 3 --seeds 0,1,2,3 \
  --out runs/embedbench-hard-ood-v1.0.0/model_confirmation

# Refit four fresh checkpoints and create the immutable selection lock.
python3 scripts/refit_and_lock_hard_ood_winner.py \
  --confirmation runs/embedbench-hard-ood-v1.0.0/model_confirmation \
  --commitment data/releases/embedbench-hard-ood-v1.0.0/commitments/LOCKED_LABEL_COMMITMENT.json \
  --seeds 0,1,2,3 \
  --out runs/embedbench-hard-ood-v1.0.0/final

# Run once under the custodian account; a second request is rejected.
python3 scripts/custodian_unlock_hard_ood.py \
  --selection-lock runs/embedbench-hard-ood-v1.0.0/final/SELECTION_LOCK.json \
  --commitment data/releases/embedbench-hard-ood-v1.0.0/commitments/LOCKED_LABEL_COMMITMENT.json \
  --private-root /restricted/embedbench-hard-ood-v1.0.0

# Evaluate all panels only after commitment, selection, and unlock receipts verify.
python3 scripts/evaluate_hard_ood.py \
  --selection runs/embedbench-hard-ood-v1.0.0/final/selection.json \
  --selection-lock runs/embedbench-hard-ood-v1.0.0/final/SELECTION_LOCK.json \
  --release runs/embedbench-hard-ood-v1.0.0 \
  --base-test-split data/release_v1/splits_quality_problem_v2.json \
  --mechanism-split data/releases/embedbench-hard-ood-v1.0.0/splits/mechanism_locked_v1.json \
  --ood-split data/releases/embedbench-hard-ood-v1.0.0/splits/locked_ood_v1.json \
  --locked-targets /restricted/embedbench-hard-ood-v1.0.0 \
  --unlock-receipt /restricted/embedbench-hard-ood-v1.0.0/UNLOCK_RECEIPT.json \
  --out runs/embedbench-hard-ood-v1.0.0/evaluation

# Assemble the post-unlock manifest and verify every publication byte.
python3 scripts/publish_hard_ood_release.py \
  --release runs/embedbench-hard-ood-v1.0.0 \
  --metadata data/releases/embedbench-hard-ood-v1.0.0 \
  --evaluation runs/embedbench-hard-ood-v1.0.0/evaluation
```

Apollo executes shard commands directly and MUST NOT use Slurm. Goose executes generation,
labeling, and training arrays only through Slurm. A Goose launch has the form:

```bash
SHARD_LAST="$(python3 scripts/hard_ood_shard_count.py \
  --plan runs/embedbench-hard-ood-v1.0.0/plans/locked_ood.json --last-index)"
sbatch --array="0-${SHARD_LAST}" scripts/goose_hard_ood_array.sbatch \
  runs/embedbench-hard-ood-v1.0.0/plans/locked_ood.json
```

The Slurm wrapper MUST derive the shard solely from `SLURM_ARRAY_TASK_ID`, record the job ID,
and execute the same Python command and environment digest as Apollo. Site is provenance, not
a seed input.

## 16. Model shortlist, retraining, and evaluation

### 16.1 Objective and primary statistic

For a quality state, `Q_MM` is the total-qubit count of the registered Minorminer starting
embedding selected from 32 attempts without labels as specified in Section 9.4. For exact
rational budget ratios `R = {100/100, 110/100, 125/100, 150/100}`, define
`B_r = floor(r * Q_MM)`. A candidate is eligible exactly when it is an exact-valid minor
embedding and its full-embedding total qubits do not exceed `B_r`. The model selects the
eligible candidate with maximum predicted `V`; canonical candidate index breaks prediction
ties.

For each state and ratio, empirical quality regret is the largest measured `V` among eligible
candidates minus measured `V` for the selected candidate. The two same-stage label blocks are
summed before this point estimate. A selection outside the exact mask, a nonfinite prediction,
or no prediction has regret 1 and is an invalid selection. It is never imputed or omitted.
States with no valid `Q_MM`, or with no eligible candidate at any registered ratio, are fixed
before training, excluded as `missing_q_mm` or `no_budget_survivor` from the primary cohort for every configuration,
and reported in an uncapped and `B/Q_resource` diagnostic.

`mean_finite_budget_regret` aggregates in this exact order: arithmetic mean across the four
finite ratios within a state, mean across states in a problem group, mean across groups in a
panel, equal mean across registered panels, then mean across seeds 0, 1, 2, and 3. Candidate or
decision-row multiplicity never changes weight. Residual connectivity and robustness are
separate auxiliary targets and reported Pareto axes. They are not silently added to quality
by an unregistered scalar. Total qubits and maximum chain length are exact masks and reported
axes, so the learned policy may choose more qubits only within an explicit budget when doing
so predicts better solve quality.

### 16.2 Base screen and shortlist

The frozen grid evaluates 5 architectures, 4 objective variants, and 4 seeds, for exactly 80
cells. Each architecture-objective configuration is ranked by its four-seed arithmetic mean
of the already registered base-validation `mean_finite_budget_regret`. The validator requires
all 80 result, receipt, source, and checkpoint hashes from the legacy snapshot. A missing seed,
failed job, NaN, infinity, corpus limit, inconsistent evaluation population, or invalid exact
selection makes that configuration ineligible. At least three configurations must remain.

The shortlist is the best three distinct eligible configurations. Ties within `1e-6` use
lower worst-seed regret, fewer trainable parameters, then canonical configuration ID. No
base-test label is read. The shortlist artifact binds every input and receipt hash. Screening
weights are discarded.

### 16.3 Mechanism-development role

Within each mechanism family, the 50 development parents split deterministically into 40
`mechanism_train` and 10 `mechanism_val` groups by the same constrained hash method as Section
13.1. Mechanism training supervises the structural continuation and feasibility heads. It has
no `p_solve` target and never enters quality regret. Mechanism validation is a preregistered
constraint: a confirmed configuration must achieve at least 90 percent exact-winner accuracy
per family and 100 percent immediate validity after the exact mask. It may disqualify a model
but cannot improve its primary quality score.

### 16.4 Hard-development confirmation

Each shortlisted configuration is retrained from scratch with seeds 0, 1, 2, and 3 on
`base_train union hard_train union mechanism_train`. The frozen confirmation population is
the same for every configuration. Evaluation gives equal weight to exactly four quality
panels: base validation, capacity transition, logical stress, and long context. Groups receive
equal weight inside a panel.

The configuration with the lowest four-seed `mean_finite_budget_regret` wins subject to zero
invalid selections and the mechanism constraints in Section 16.3. Ties within `1e-6` use
lower worst-panel regret, fewer trainable parameters, then canonical configuration ID.
Uncapped, reliability-stratified, connectivity, and coverage results are diagnostics and do
not change selection.

Each confirmation seed stores its validation-selected epoch. After the configuration is
chosen, sort its four epoch values and freeze the third value as the upper median. This fixed
epoch count and the complete optimizer schedule are used for final refitting, which has no
early stopping.

### 16.5 Final refit and lock

The chosen configuration is refit from scratch for all four seeds on `base_train union base
validation union hard_train union hard_val union mechanism_train union mechanism_val`. No
confirmation checkpoint is reused. Base test, mechanism locked, and locked OOD remain absent.

`selection.json` MUST bind the legacy snapshot, shortlist and confirmation inputs, final
config and source digest, fixed epoch count, optimizer schedule, preprocessing, data, split,
host, bank and label-protocol digests, exact commands and environment digests, aggregation and
tie rules, all four checkpoint byte hashes, and the pre-existing locked-label commitment.

`SELECTION_LOCK.json` stores the release ID, raw `selection.json` SHA-256, pre-unlock manifest
SHA-256, locked commitment SHA-256, creation time, selection-key fingerprint, and Ed25519
signature over all preceding fields. It is published to append-only project storage before
the custodian is contacted. The custodian keeps a durable release-ID state machine that accepts
exactly one valid selection digest with an atomic compare-and-set from `sealed` to `unlocked`.
A second request, including the same digest, is rejected and audited.

`UNLOCK_RECEIPT.json` has schema `embedbench.unlock-receipt` version 1 and binds the release ID,
selection-lock digest, raw selection digest, commitment digest, released payload-root digest,
recipient evaluator identity, UTC release time, resulting one-time state, custodian key
fingerprint, and signature. The evaluator accepts no plaintext without this receipt. The final
manifest covers both signed objects, so preparing several locks, evaluating them, and choosing
one afterward cannot satisfy the release contract.

### 16.6 Confirmatory evaluation

All four final checkpoints are evaluated. For a group, the four seed-specific results are
averaged before panel aggregation. The paper also reports every seed and their standard
deviation, but never selects a paper seed after unlock. Confidence intervals use 10,000 paired
stratified bootstrap replicates over `problem_group_id`. A sampled group carries every state,
budget, method, model seed, and, for fault panels, its unique host mask together. The bootstrap
seed and implementation digest are frozen in `selection.json`.

Results are reported separately for:

- base IID test;
- each of the seven mechanism families;
- composed mechanisms;
- extreme capacity;
- host scale;
- each unseen logical family;
- light and heavy host faults;
- full-bank audit subsets;
- reliable and ambiguous label subsets with coverage.

The primary OOD aggregate weights the four OOD panels equally, then weights problem groups
equally within a panel. It never weights by the number of seams or candidates.

## 17. Acceptance gates

| Gate | Requirement | Failure consequence |
|---|---|---|
| A0: artifact identity | Pinned base inputs, staged legacy source, every published byte, schema, path, and count match the applicable pre-unlock or final manifest and `SHA256SUMS`; every seed stage, parent root, shard range, and occupied-32 digest verifies. | Do not publish or train. |
| A1: exact quotas | All Section 9 cell counts and required distinct-ID union cardinalities hold, with one quota cell per group. | Generate only additional pre-registered ordinals before labels. |
| A2: split isolation | Zero overlap by problem, problem isomorphism, mechanism parent, mechanism isomorphism, or state digest. | Rebuild all affected splits under a new release ID. |
| A3: calibrated fields | Every hardness field recomputes exactly; all profile predicates hold. | Reject the record artifact, not individual rows. |
| A4: host integrity | Every defective record verifies against its content-addressed realized host. | Exclude the whole host artifact and refill its cell before labels. |
| A5: ground-state proof | 100 percent of locked quality groups have zero-gap certification. | Structural-only use; no locked quality claim. |
| A6: candidate banks | Full and offered digests verify, and the development truncation audit passes the confidence-bound recall and regret limits. | Version and rebuild before model selection; locked audit is report-only. |
| A7: coverage | The fixed pre-label population retains every attempt and reason, and all planned eligible labels complete. Reliable coverage is reported without a pass threshold. | Invalidate the release; never add or replace groups using label outcomes. |
| A8: structural balance | In every hard-development host-profile cell, feasibility is the exact decisive margin for at least 30 percent of ranking-eligible capacity rows and 20 percent of logical-stress or long-context rows. | Continue the Section 13.3 prefix scan before cell freeze; fail at the ordinal cap. |
| A9: mechanisms | All 700 independent single-mechanism groups and 100 independent composed groups have positive checker-accepted certificates. | Consume new pre-registered parents before locked-target commitment. |
| A10: compute parity | Pure generation, canonicalization, exact structural search, and bank bytes match on Apollo and Goose for fixed fixtures; timing and stochastic label outputs are outside this byte-parity claim. | Fix nondeterminism before labels. |
| A11: escrow and selection lock | Locked bytes are generated and committed in isolation first; final recipe, epochs, four checkpoints, commands, source, data, and metric hashes verify before unlock. | Keep committed labels inaccessible and invalidate any premature access. |
| A12: reporting | Metrics are group-clustered, panel-balanced, and include coverage and every exclusion reason. | No paper-facing result bundle. |
| A13: compute plan | Realized jobs, reads, storage, retries, and allocations remain within the signed compute plan. | Do not launch the over-budget stage. |

## 18. Testing strategy

Tests live under `EmbedBench/tests/` and MUST include:

- schema tests for every artifact type;
- canonical JSON and digest golden tests;
- deterministic seed, lifecycle-extension, parent-collision, shard-order, and batched-resolver
  one-read tests;
- adversarial forged-summary, wrong-expected-digest, oversized-root, over-limit shard,
  unexpected-file, final-file symlink, ancestor-symlink, check/open race, and no-clobber
  publication tests;
- aggregate source-payload and path-depth tests that prove rejection occurs before file reads;
- continuation cap-rebinding, bounded ZIP-member, compressed-expansion, ZIP-depth,
  whole-capsule, self-rehash, check/use mutation, and staging-receipt tests;
- ordered stage-artifact manifest tests for missing shards, broken parent links, visibility,
  release identity, and independently supplied artifact-manifest and terminal digests;
- exact quota and constrained split tests;
- adversarial leakage tests across topology, source, state, problem identity, logical
  isomorphism, and mechanism isomorphism;
- defective-host reconstruction refusal and explicit-host verification tests;
- ground-state eligibility and certificate-chain tests;
- candidate canonicalization, 64-slot enforcement, and label-independence tests;
- full-bank audit membership and regret tests;
- Holm interval, independent-block, no-margin retention, and denominator tests;
- motif transformation and independent recertification tests;
- Apollo versus Goose deterministic parity tests using small fixtures;
- tests proving development identities can never read locked targets and the evaluator cannot
  read them before a valid selection lock.
- staged-manifest, signed commitment, unlock receipt, and committed-plaintext replay tests.

The minimum verification commands are:

```bash
python3 -m pytest -q tests/test_hard_ood_*.py
python3 scripts/validate_hard_ood_release.py \
  --release runs/embedbench-hard-ood-v1.0.0 \
  --metadata data/releases/embedbench-hard-ood-v1.0.0
git diff --check -- EmbedBench
```

Small fixtures MUST exercise every profile and failure status without containing any locked
quality label.

## 19. Project structure for implementation

```text
EmbedBench/src/embedbench/
  hard_ood_schema.py       # typed records and canonical serialization
  hard_ood_provenance.py   # secure canonical-byte and source-bundle verification
  hard_ood_protocols.py    # generation, solver, and continuation protocols
  hard_ood_profiles.py     # profile predicates and exact quotas
  realized_host.py         # defects, host artifacts, and verification
  hardness.py              # raw calibrated measurements and bands
  ground_state.py          # proof metadata and eligibility
  candidate_protocol.py    # full enumeration and deterministic offered bank
  mechanism_variants.py    # randomized and composed certified motifs

EmbedBench/scripts/
  fetch_hard_ood_inputs.py
  plan_hard_ood.py
  hard_ood_shard_count.py
  generate_hard_ood_shard.py
  merge_hard_ood.py
  validate_hard_ood_release.py
  finalize_hard_ood_compute_plan.py
  certify_hard_ood_ground_states.py
  verify_ground_state.py
  profile_hard_ood_solvers.py
  label_hard_ood_quality.py
  audit_hard_ood_candidate_bank.py
  custodian_label_hard_ood.py
  custodian_unlock_hard_ood.py
  finalize_hard_ood_preunlock.py
  confirm_quality_shortlist.py
  refit_and_lock_hard_ood_winner.py
  evaluate_hard_ood.py
  publish_hard_ood_release.py
  goose_hard_ood_array.sbatch

EmbedBench/tools/legacy_provenance/
  freeze_legacy_training_snapshot.py

EmbedBench/tests/
  test_hard_ood_schema.py
  test_hard_ood_provenance.py
  test_hard_ood_protocols.py
  test_hard_ood_profiles.py
  test_realized_host.py
  test_ground_state_certificates.py
  test_candidate_protocol.py
  test_mechanism_variants.py
  test_hard_ood_splits.py
  test_hard_ood_release.py
  test_hard_ood_selection_lock.py
```

Implementations SHOULD use frozen dataclasses for in-memory contracts and pure functions for
canonicalization. Validators MUST reject unknown keys for versioned schemas. A canonical
serializer returns bytes, not a platform-dependent string. `canonical_json(x)` in formulas
means `canonical_bytes(x)` below:

```python
import json
import math
import unicodedata


def normalize_unicode_scalar(value: str) -> str:
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        raise ValueError("Unicode surrogate code point")
    return unicodedata.normalize("NFC", value)


def canonical_value(value: object) -> object:
    value_type = type(value)
    if value_type is float:
        if not math.isfinite(value):
            raise ValueError("non-finite float")
        normalized = 0.0 if value == 0.0 else value
        return {"__float64_hex__": normalized.hex().lower()}
    if value is None or value_type is bool or value_type is int:
        return value
    if value_type is str:
        return normalize_unicode_scalar(value)
    if value_type is list or value_type is tuple:
        return [canonical_value(item) for item in value]
    if value_type is dict:
        if not all(type(key) is str for key in value):
            raise TypeError("canonical object keys must be strings")
        keys = [normalize_unicode_scalar(key) for key in value]
        if len(keys) != len(set(keys)) or "__float64_hex__" in keys:
            raise ValueError("duplicate normalized or reserved key")
        return {
            normalize_unicode_scalar(key): canonical_value(item) for key, item in value.items()
        }
    raise TypeError(f"unsupported canonical type: {type(value)!r}")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        canonical_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
```

Only the exact built-in types shown above are admitted. Subclasses of `float`, `int`, `bool`,
`str`, `list`, `tuple`, or `dict`, general mapping implementations, dataclasses, NumPy scalar
types, byte strings, sets, and arbitrary objects are rejected rather than coerced. Callers
convert typed records to an exact built-in dictionary explicitly before hashing. This prevents
custom iteration, conversion, equality, or serialization behavior from changing an identity.
Every string and object key must be a Unicode scalar sequence; isolated UTF-16 surrogate code
points are invalid even though Python can store them in `str` and escape them in JSON. A
versioned object's `schema` discriminator must have exact built-in type `str` before equality is
tested, so a subclass or equality proxy cannot impersonate a registered schema.

Every artifact `relative_path` is an NFC-normalized Unicode-scalar POSIX path. Absolute paths,
empty or dot segments, `..`, repeated separators, backslashes, NUL and other Unicode control
characters, and a leading Windows drive designator are invalid. Manifest loaders resolve paths
under the declared artifact root, reject symlinks, and bind the normalized path together with
the raw payload digest, byte count, record count, and schema version.

## 20. Implementation work packages

1. Materialize `BASE_INPUTS.json` and reconstruct `LEGACY_TRAINING_SNAPSHOT.json`.
  - Acceptance: all twelve base payloads, both safe splits, all 80 grid receipts, and source
    digest `294caa56d0bb04f716163b143a995ee176c8e72dd209c5f5f77caef92d72a264`
    verify without editing a legacy file. The snapshot records permanent
    architecture-selection-only scope and never paper reproducibility eligibility.
  - Verify: wrong-byte, stale-manifest, missing-cell, dirty-source, cross-root command,
    same-byte checkpoint capture, post-hoc receipt-field, and retrieval tests.
2. Define schemas, canonical bytes, content digests, and seed namespaces.
  - Acceptance: golden fixtures reproduce identical bytes on Apollo and Goose; every lifecycle
    stage freezes before execution; the 1.7-million-entry label plan writes and verifies with
    bounded memory.
  - Verify: schema, digest, parent-chain, extension-collision, shard-order, streaming, seed, and
    parity tests.
3. Implement realized-host artifacts and defect-aware verification.
  - Acceptance: every record, including a pristine-host record, requires the exact realized-host
    artifact plus independently committed graph and artifact digests. Pristine reconstruction is
    available only through an explicitly generation-only helper.
  - Verify: pristine, light-fault, heavy-fault, and tamper tests.
4. Implement structured hardness measurements and profile predicates.
  - Acceptance: every Section 8 field is recomputed from source artifacts.
  - Verify: threshold boundary and invalid-profile tests.
5. Implement problem, starting-state, and candidate-bank generation.
  - Acceptance: all attempts persist and offered banks contain at most 64 candidates.
  - Verify: enumeration, truncation, label-independence, and failure-retention tests.
6. Implement ground-state certificate ingestion and eligibility.
  - Acceptance: heuristic references cannot enter locked quality evaluation.
  - Verify: planted, exact, zero-gap, nonzero-gap, and missing-proof tests.
7. Implement mechanism randomization and composition.
  - Acceptance: every retained variant independently certifies a positive margin.
  - Verify: all seven families, two-trap, three-trap, collision, and reconstruction tests.
8. Implement quota-aware planning, shard generation, merge, and split construction.
  - Acceptance: Section 9 quotas are exact and all overlap audits are zero.
  - Verify: small plan, resume, duplicate shard, missing shard, and constrained split tests.
9. Implement staged labels and locked commitments.
  - Acceptance: development commands refuse locked partitions, custodian generation requires
    frozen structural digests, and unlock requires a verified selection lock.
  - Verify: access-control, protocol-digest, seed-separation, and tamper tests.
10. Implement shortlist confirmation, final refit receipts, and four-seed aggregation.
  - Acceptance: no checkpoint can be evaluated unless it is bound by `selection.json`.
  - Verify: command mismatch, checkpoint mismatch, missing seed, and panel-weight tests.
11. Build and independently validate `embedbench-hard-ood-v1.0.0`.
  - Acceptance: every gate A0 through A13 passes and the data card reports all attrition and
    coverage denominators.
  - Verify: full release validator plus a fresh-context artifact review.

## 21. Success criteria

The corpus is complete when all of the following are true:

1. `release_v1_1` remains byte-identical and is referenced by verified digests.
2. The hard-development, mechanism, and locked OOD matrices have the exact group quotas in
   Section 9.
3. Hardness fields are numeric, reproducible, and independent of a free-text difficulty tag.
4. Every defective state resolves to an immutable realized-host artifact.
5. Every locked quality group has a zero-gap ground-state certificate.
6. Candidate truncation is label-free and passes the full-bank audit thresholds.
7. Ambiguous and failed states remain in coverage accounting.
8. Every mechanism variant and composition has an independent exact certificate.
9. Base screening, hard confirmation, final refit, and locked evaluation are cryptographically
   separated.
10. Every stochastic operation resolves through a pre-execution verified stage root, and no
    child stage changes a parent request or 32-bit assignment.
11. All four registered final seeds are evaluated and aggregated by independent problem
    group.
12. Every gate A0 through A13 passes from a clean checkout using the published commands.
