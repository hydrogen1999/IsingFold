# IsingFold training operations

This runbook is the executable Profile-I contract for data preparation, model training,
validation selection, and sealed complete-system evaluation. It complements
[`ISINGFOLD_MODEL_SPEC.md`](ISINGFOLD_MODEL_SPEC.md): the model specification defines the state,
action, network, and loss; this file defines artifact order, trust boundaries, frozen grids, and
host-specific execution.

## Non-negotiable rules

1. Production experiments consume corpus-design-complete, partition-sealed prepared schema v4.
   Prepared v1/v2/v3 and
   `dev-generate` are diagnostic-only and cannot support hard, fault, calibration, or OOD claims.
2. The publisher-attestation digest and publisher ID are obtained out of band. Never copy either
   value from the attestation file that it is meant to authenticate.
3. IsingFold never creates or infers ground-energy authority. EmbedBench publishes physically
   partitioned evaluator targets, evidence manifests, certificate artifacts, and publisher
   attestation v2. IsingFold verifies that chain before a trusted process opens one named
   partition. Public planning may authenticate `root.json`, but must not read a partition receipt
   or target file.
4. The scientific search contains exactly nine representation cells followed by eighteen RL-value
   cells. Do not add, omit, or retry a cell under the same experiment identity.
5. Architecture and RL-value decisions aggregate all three registered training seeds and immutable
   base lineages. Selecting a best seed is forbidden.
6. Complete-system test outcomes remain sealed until the RL-value configuration,
   complete-system protocol, and analysis are frozen. The separately named selector
   `audit_test` partition opens only through the frozen RL-value receipt and cannot affect either
   main selection.
7. Apollo runs jobs directly because it has no Slurm. Goose runs compute only through `sbatch` and
   `srun`; never train or evaluate on the Goose login node.

## Publication runtime pins

Set the runtime pins before any production launcher. Apollo uses one project virtual environment
directly and rejects a Slurm context. The lock file and its digest identify the installed
environment; `ISINGFOLD_PYTHON` must name the interpreter inside that environment.

```bash
export ISINGFOLD_PYTHON=/opt/isingfold/venv/bin/python
export ISINGFOLD_ENV_LOCK=/opt/isingfold/environment.lock
export ISINGFOLD_ENV_LOCK_SHA256='<out-of-band environment-lock SHA-256>'
```

Goose runs only inside a Slurm allocation and invokes computation through `srun` in a pinned
Apptainer image. Both the runtime executable and image are file-hash authenticated before the
container starts.

```bash
export ISINGFOLD_APPTAINER=/opt/apptainer/bin/apptainer
export ISINGFOLD_APPTAINER_SHA256='<out-of-band Apptainer executable SHA-256>'
export ISINGFOLD_CONTAINER_IMAGE=/opt/isingfold/isingfold.sif
export ISINGFOLD_CONTAINER_IMAGE_SHA256='<out-of-band image SHA-256>'
export ISINGFOLD_CONTAINER_PYTHON=/opt/isingfold/venv/bin/python
export ISINGFOLD_GOOSE_OUTPUT_BASE="$(pwd -P)/runs"
export ISINGFOLD_SCRIPT_DIR="$(pwd -P)/scripts"
mkdir -p "$ISINGFOLD_GOOSE_OUTPUT_BASE"
```

Every `scripts/apollo_*.sh` or `scripts/goose_*.sbatch` publication launcher below requires the
corresponding variables. `ISINGFOLD_SCRIPT_DIR` is required because Slurm executes a private copy
of the submitted batch file from its spool directory; resolving helpers relative to
`BASH_SOURCE[0]` would therefore select the wrong directory. Export the variables through
`sbatch --export=ALL` on Goose. The bare host Python
path and Goose login node are diagnostic-only and cannot create publication artifacts. Direct CLI
examples use the installed `isingfold-rl` entry point for readability; execute them in the same
pinned environment class and record that class in every artifact that exposes runtime provenance.

The Goose guard starts Apptainer with a clean, contained environment, disables the host home bind,
mounts source and the approved output base read-only, and remounts only the output roots named by
the selected launcher read-write. Every output root must resolve inside the absolute
`ISINGFOLD_GOOSE_OUTPUT_BASE`; an in-repository base is allowed only below `runs/`. Set
`ISINGFOLD_GOOSE_INPUT_ROOTS` to a colon-separated list of absolute existing paths when an
authenticated input lives outside the source tree or output base. Those paths are mounted
read-only. The guard removes host Python and Apptainer environment injection and passes back only
the required Slurm identity fields. The registered image must expose its interpreter at exactly
`/opt/isingfold/venv/bin/python`; a host-mounted interpreter is rejected. Initializer-bank and final-strength workers use the CPU
container path; GPU launchers alone request `--nv`.

Goose log directives write into the existing submission directory, so a clean checkout does not
depend on a `slurm/` directory that Slurm would have to open before the job script starts. To place
logs elsewhere, create that directory before submission and supply absolute `sbatch --output` and
`--error` overrides.

## Artifact graph and schema contract

| Stage | Input authority | Published artifact | Required schema |
|---|---|---|---|
| Prepare | independent CandidateBank-v2 export, manifest, partitionable targets, provenance, out-of-band-pinned corpus design v2 | eight-file partition-sealed corpus plus complete design receipt | prepared v4 |
| Authenticate quality | publisher pin, prepared target authority, pinned verifier build/runtime | target-free root plus train/validation/test partition receipts | publisher attestation v2, ground protocol v2 |
| Plan initializer bank | public train tasks, LAC runtime, complete-system config | target-free lineage-equal conditional schedule | initializer-bank plan v2 |
| Generate and seal initializer bank | pinned plan and deterministic initializer draws | immutable snapshots, manifest, and external manifest pin | snapshot/bank/access v2 |
| Label selector | one authorized target partition | count-complete four-strength labels plus access receipts | selector-label manifest v4 |
| Fit selector | frozen train labels and train authority | immutable selector bundle | selector bundle v4 |
| Label quality | train partition, selector, authority, resolution plan, and pinned quality initializer bank | exact-replay counterfactual rows | publication manifest v8; row v7; shard and merge envelopes v6; `if-q3-s0-qmu-7` target |
| Preflight quality | canonical merged quality corpus, live quality bank, and external pin | exact bank-backed full-replay receipt | quality preflight v3 |
| Release gates | exact conformance corpus, selector, quality corpus, live quality bank, resolution plan, and preflight | immutable gate receipt | release gate v7 |
| Train | gates, quality rows, quality-bank replay contract, frozen grid | checkpoint, run receipt, warm history | checkpoint v3, run v4, warm history v4 |
| Select representation | nine validation reports | immutable representation receipt | representation selection v4 |
| Select RL value | eighteen validation reports | immutable RL-value receipt | RL-value freeze v4 |
| Confirm system | frozen selected configuration and per-seed initializer bank | three fresh training runs and three test reports | receipt v3, report v4 |
| Aggregate | authenticated three-seed reports and sidecars | one self-digested aggregate JSON | aggregate v2, crossed seed by lineage bootstrap |
| Compare stock | validation-tuned stock arm and learned arm | tuning run/selection v2, external report v4, paired aggregate v4 | partition-authorized receipts |
| Resume evaluation | target-free full-population plan, partition-authorized workers | lineage shards and canonical merge | plan v2, shard v3, merge v2 |
| Audit strengths | frozen six-run source census and A/B read blocks | plan, execution manifest, shards, merge, receipt | config v1, plan v2, execution v2, receipt v2 |

Schema versions are compatibility boundaries. Prepared v1/v2/v3, selector bundle v2/v3,
publication quality manifests before v8, quality shard or merge envelopes before v6, quality
preflight before v3, release gates before v7, representation or RL-value selection v3, and older
complete-system reports must not be renamed or re-signed to enter this workflow. Legacy quality
artifacts remain diagnostic-only and require an explicit diagnostic receipt.

Every command that can open evaluator targets requires the same three trust-anchor arguments:

```bash
export QUALITY_ATTESTATION=/data/EmbedBench/release/publisher_attestation.json
export EXPECTED_QUALITY_ATTESTATION_DIGEST='<digest from the trusted release registry>'
export EXPECTED_QUALITY_PUBLISHER_ID='<publisher ID from the trusted release registry>'

QUALITY_ARGS=(
  --quality-attestation "$QUALITY_ATTESTATION"
  --expected-quality-attestation-digest "$EXPECTED_QUALITY_ATTESTATION_DIGEST"
  --expected-quality-publisher-id "$EXPECTED_QUALITY_PUBLISHER_ID"
)
```

Publisher attestation v2 binds the prepared-manifest SHA-256, source release, partitioned target
authority, and separately hashed partition evidence. Each evidence manifest covers its targets
exactly once and binds them to real certificate artifacts and verifier identity. Public policy
loading checks commitments without reading target files. A target-bearing loader returns only the
requested partition and emits a `TargetAccessReceipt`.

## 1. Install and prepare production data

```bash
python3 -m pip install -e '.[rl,baseline]'

export CORPUS_DESIGN=/data/EmbedBench/corpus_design_v2.json
export EXPECTED_CORPUS_DESIGN_SHA256='<file digest from the trusted design registry>'

isingfold-rl prepare \
  --bank /data/EmbedBench/candidate_bank_v2.jsonl \
  --bank-manifest /data/EmbedBench/candidate_bank_v2.manifest.json \
  --evaluator-targets /data/EmbedBench/evaluator_targets.jsonl \
  --provenance /data/EmbedBench/isingfold_task_provenance.jsonl \
  --corpus-design-manifest "$CORPUS_DESIGN" \
  --expected-corpus-design-sha256 "$EXPECTED_CORPUS_DESIGN_SHA256" \
  --qubit-cap 160 \
  --out runs/prepared_v4
```

`prepare` publishes exactly eight immutable files: public policy instances, Profile-I initializers,
three physically separate evaluator-target files under `targets/`, splits, provenance, and
`manifest.json`. The output directory must not already exist. The independently obtained
design-file SHA-256 is mandatory; never read the expected digest from the file it authenticates.
The v4 manifest embeds the exact condition registry, partition target commitments, registered
quota/power/precision targets, and recomputed realized census. Structural validation follows
[`configs/corpus_design_v1.schema.json`](../configs/corpus_design_v1.schema.json); the adjacent
template is an authoring example and deliberately fails its production partition floors and
realistic power/precision minima until operators replace the placeholder identities and expand the
frozen lineage registry. The prepared
manifest authenticates the corpus and design, but ground-energy authority still comes only from the
independently pinned publisher attestation.

The signed `minimum_partition_base_lineages` registry cannot be lower than 1,024 train, 512
validation, and 1,546 sealed-test lineages. Each exact partition quota must meet its registered
floor, and the test floor must also meet the largest registered confirmatory target. Counts are over
unique immutable base lineages, so crossing one base over hosts or faults cannot inflate them.

Production provenance records the immutable base parent, descendant transform, nominal and active
host identities, realized fault mask, calibration identity, distribution stratum, and learning
partition. One base may appear under multiple registered host/fault conditions, but it is counted
once for quota and inferential power and must remain in one partition. Difficulty strata require
outcome-blind authority, panel, protocol, budget, and evidence digests. OOD conditions and their
dedicated quota are confined to sealed test. The design must realize at least two host families, a
faulted condition, and both `application-derived` and `synthetic` problem origins; concrete
`application_family` and host-family values must match the authenticated source. For one-sided
noninferiority, `power_separation = assumed_true_difference + noninferiority_margin`; the main
valid-return target records margin `0.02`, assumed difference `0.0`, separation `0.02`, and alpha
`0.05` explicitly. The importer recomputes its 1,546-lineage illustrative minimum at assumed
discordance `0.1` and target power `0.8`.

Both this confirmatory noninferiority target and the paired-utility precision target below must use
filters equal to every axis value realized in the sealed-test partition. Hard, OOD, and other cells
remain frozen quota-controlled subgroups; they do not replace the full-population primary targets.

The co-primary failure-aware utility target is the paired lineage difference
`learned-minus-stock-unconditional-if-q3-s0`. Since each difference lies in `[-1,1]`, the signed
design uses the conservative worst-case variance bound `1.0`; 95% confidence and half-width `0.05`
require 1,537 independent lineages. A superiority claim is CI-based and is made only if the frozen
paired interval excludes zero in the learned direction. This precision target does not promise that
the experiment will be positive.

Validation-only baseline tuning requires two separate named targets over exactly the full realized
validation population, including every crossed condition. The
`validation-valid-return-noninferiority` target uses the paired-binary normal approximation with
one-sided alpha `0.05`, power `0.8`, assumed discordance `0.1`, assumed true difference `0.05`,
margin `0.02`, and therefore separation `0.07`. The registered calculation
`ceil(0.1 * (z_0.95 + z_0.8)^2 / 0.07^2)` is 127, and the manifest requires at least 128 independent
validation lineages. The `validation-paired-utility-precision` target uses paired differences in
`[-1,1]`, worst-case variance bound `1.0`, 95% confidence, and half-width `0.2`; its calculation
`ceil(z_0.975^2 / 0.2^2)` is 97, with the same conservative registered minimum of 128. These are
prospective denominator and adequacy guards used to construct `CompletePopulationIdentity` from
prepared-v4. They are not post-selection power or confidence claims about tuning outcomes.

Before any scientific command opens those targets, run the independently pinned standalone checker
over the complete authenticated census:

```bash
export GROUND_VERIFIER=/opt/isingfold-trust/ground-certificate-verifier
export EXPECTED_GROUND_VERIFIER_EXECUTABLE_SHA256='<trusted executable file SHA-256>'
export GROUND_VERIFIER_SOURCE=/opt/isingfold-trust/ground-certificate-verifier.source
export EXPECTED_GROUND_VERIFIER_SOURCE_SHA256='<trusted source artifact SHA-256>'
export GROUND_VERIFIER_ENVIRONMENT=/opt/isingfold-trust/ground-verifier-environment.lock
export EXPECTED_GROUND_VERIFIER_ENVIRONMENT_SHA256='<trusted environment lock SHA-256>'
export GROUND_VERIFIER_BUILD_ATTESTATION=/opt/isingfold-trust/ground-verifier-build.json
export EXPECTED_GROUND_VERIFIER_BUILD_ATTESTATION_SHA256='<trusted build-attestation SHA-256>'

isingfold-rl verify-ground-certificates \
  --corpus runs/prepared_v4 \
  "${QUALITY_ARGS[@]}" \
  --verifier-executable "$GROUND_VERIFIER" \
  --expected-verifier-executable-sha256 "$EXPECTED_GROUND_VERIFIER_EXECUTABLE_SHA256" \
  --verifier-source "$GROUND_VERIFIER_SOURCE" \
  --expected-verifier-source-sha256 "$EXPECTED_GROUND_VERIFIER_SOURCE_SHA256" \
  --verifier-environment "$GROUND_VERIFIER_ENVIRONMENT" \
  --expected-verifier-environment-sha256 "$EXPECTED_GROUND_VERIFIER_ENVIRONMENT_SHA256" \
  --expected-verifier-name '<registered verifier name>' \
  --expected-verifier-version '<registered verifier version>' \
  --verifier-execution-mode static-elf \
  --verifier-build-attestation "$GROUND_VERIFIER_BUILD_ATTESTATION" \
  --expected-verifier-build-attestation-sha256 \
    "$EXPECTED_GROUND_VERIFIER_BUILD_ATTESTATION_SHA256" \
  --out runs/ground_certificates_v2

export GROUND_CERTIFICATE_ROOT=runs/ground_certificates_v2/root.json
export EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256='<freeze root.json SHA-256 out of band>'
GROUND_CERTIFICATE_ARGS=(
  --ground-certificate-root "$GROUND_CERTIFICATE_ROOT"
  --expected-ground-certificate-root-sha256 \
    "$EXPECTED_GROUND_CERTIFICATE_ROOT_SHA256"
)
```

The command executes every certificate exactly once in the pinned verifier environment. It writes
`partitions/train.json`, `partitions/val.json`, and `partitions/test.json`, then publishes
`root.json` last. The root is target-free and commits the complete census. Every downstream
scientific command receives both `GROUND_CERTIFICATE_ARGS` entries. Planning stops at the root.
Execution opens exactly one root-authorized partition, reconstructs its target binding, and retains
the partition `TargetAccessReceipt` and ground receipt in the produced artifact. For Apptainer
verification, also supply the pinned runtime path and SHA-256 required by the selected verifier
build attestation.

The following commands remain outside the paper path:

```bash
# Local smoke data, intentionally rejected by production training and evaluation.
isingfold-rl dev-generate --out runs/smoke --instances 16 --seed 7

# Legacy release-v1.1 adapter, accepted only by explicit diagnostic callers.
isingfold-rl prepare-release-v1 --help
```

## 2. Train and freeze the terminal strength selector

Selector fitting is independent of PPO. Each label contains four compiled program graphs and four
integer count pairs. The deployed selector sees only deployment-available instance, embedding,
program, strength, and hardware features. It never sees the certified energy, evaluator seed, or
true solve probability.

```bash
isingfold-rl label-selector-data \
  --corpus runs/prepared_v4 \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  --seed 503 \
  --split-seed 601 \
  --calibration-fraction 0.2 \
  --out runs/selector_labels_v4

isingfold-rl fit-selector \
  --corpus runs/prepared_v4 \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  --selector-labels runs/selector_labels_v4 \
  --epochs 100 \
  --graph-minibatch 32 \
  --seed 701 \
  --device cuda \
  --out runs/selector_v4
```

The checked launchers preserve the same split and optimizer contract. On Apollo, run
`scripts/apollo_selector_labels.sh CORPUS LABELS_OUT` and then
`scripts/apollo_fit_selector.sh CORPUS LABELS SELECTOR_OUT`. On Goose, submit
`scripts/goose_selector_labels.sbatch` first and submit `scripts/goose_fit_selector.sbatch` with
an `afterok` dependency on the label job. Selector label generation is CPU sampling. Selector
fitting is neural training and requires CUDA. Both launchers authenticate the train quality
authority and ground-certificate root before opening targets.

The selector-label manifest v4 and selector bundle v4 carry a global publisher/root identity plus
the train-partition quality authority. The bundle also retains the exact train
`TargetAccessReceipt` and ground-partition receipt. Audit mode is a distinct post-freeze operation
that may open validation and test; those authorities cannot be substituted for train authority or
used by `fit-selector`. Any mismatch with the pinned corpus aborts fitting, training, or evaluation.

## 3. Build exact-replay quality labels

The warm-start target is plausible-best membership under simultaneous Hoeffding intervals. An
unresolved row is retained for audit and excluded from ranking loss. It is never converted to a
winner by a tie-breaking heuristic. Determine the continuation count with a separately named
resolution study, then freeze it before producing the full artifact. The commands below are
protocol templates, not evidence that any particular continuation count passes the paper gate.

The former direct `--continuations` path is diagnostic-only. Publication first freezes a target-free
resolution plan against a distinct quality initializer bank, executes and independently verifies
the registered continuation ladder, and merges a terminal resolution receipt. The abbreviated
operator sequence is:

```bash
QUALITY_BANK_ARGS=(
  --initializer-bank runs/quality_initializer_bank_seed907
  --expected-initializer-bank-manifest-sha256 "$ISINGFOLD_QUALITY_BANK_SHA256"
  --complete-config configs/complete_system_lac_hybrid_cache_v1.json
)

isingfold-rl plan-quality-resolution \
  --config configs/quality_resolution_v1.json \
  --expected-config-sha256 "$ISINGFOLD_QUALITY_RESOLUTION_CONFIG_SHA256" \
  --grid configs/rl_grid_hybrid_v1.json \
  --expected-grid-sha256 "$ISINGFOLD_GRID_SHA256" \
  --corpus runs/prepared_v4 --selector runs/selector_v4 \
  "${QUALITY_ARGS[@]}" "${GROUND_CERTIFICATE_ARGS[@]}" \
  --selector-device-parity runs/selector_device_parity.json \
  --expected-selector-device-parity-sha256 "$ISINGFOLD_SELECTOR_PARITY_SHA256" \
  "${QUALITY_BANK_ARGS[@]}" --device cuda --out runs/quality_resolution_plan.json

# Execute run-quality-resolution-shard, verify-quality-resolution-shard,
# publish-quality-resolution-shard-pins, and merge-quality-resolution until terminal.

isingfold-rl label-quality \
  --corpus runs/prepared_v4 --selector runs/selector_v4 \
  "${QUALITY_ARGS[@]}" "${GROUND_CERTIFICATE_ARGS[@]}" \
  "${QUALITY_BANK_ARGS[@]}" \
  --resolution-plan runs/quality_resolution_plan.json \
  --expected-resolution-plan-sha256 "$ISINGFOLD_QUALITY_PLAN_SHA256" \
  --resolution-receipt runs/quality_resolution_receipt.json \
  --expected-resolution-receipt-sha256 "$ISINGFOLD_QUALITY_RESOLUTION_SHA256" \
  --capacity-selection runs/quality_capacity_selection.json \
  --expected-capacity-selection-sha256 "$ISINGFOLD_QUALITY_CAPACITY_SELECTION_SHA256" \
  --capacity-budget runs/quality_capacity_budget.json \
  --expected-capacity-budget-sha256 "$ISINGFOLD_QUALITY_CAPACITY_BUDGET_SHA256" \
  --capacity-canary runs/quality_capacity_canary.json \
  --expected-capacity-canary-sha256 "$ISINGFOLD_QUALITY_CAPACITY_CANARY_SHA256" \
  --instances 0 --tasks-per-lineage 1 --states-per-lineage 4 --actions 8 \
  --seed 907 --device cuda --out runs/quality_v8

isingfold-rl quality-preflight \
  --corpus runs/prepared_v4 --selector runs/selector_v4 \
  "${QUALITY_ARGS[@]}" "${GROUND_CERTIFICATE_ARGS[@]}" \
  "${QUALITY_BANK_ARGS[@]}" --quality-labels runs/quality_v8 \
  --min-resolved-rows 128 --min-resolved-lineages 128 \
  --device cpu --out runs/quality_preflight_v3.json
```

Publication quality manifest v8 binds row schema v7 to the prepared task, exact state/action
envelope, selected payload, workspace successor when one exists, evaluator target, continuation
seed schedule, observation, action outcomes, quality authority, resolution-plan identity, and one
externally pinned K=2 initializer bank. Its continuation target is `if-q3-s0-qmu-7`. Preflight v3
authenticates and exactly replays these fields from that live bank; it cannot fall back to the
legacy per-task initializer. The two minima count different denominators, and the scientific gate
requires both. No representation or RL-value cell may start until this exact preflight reaches at
least 128 resolved rows and 128 resolved independent lineages.

Freeze the continuation count through a separately named resolution study. That study is a design
input, not part of the production corpus. Do not lower the two resolution gates or relabel
inconclusive rows in response to an unfavorable resolution rate.

### Distributed quality generation

Sharding uses complete immutable lineages. A shard first reconstructs the global lineage plan and
then owns positions `I, I+N, I+2N, ...`. No lineage can cross shard boundaries.

```bash
# Publication uses every authenticated train lineage. A positive value is diagnostic-only.
export ISINGFOLD_QUALITY_LINEAGES=0
export ISINGFOLD_QUALITY_SEED=907
export ISINGFOLD_SHARD_CONCURRENCY=4
export ISINGFOLD_GPU_IDS=0,1,2,3

# Apollo: direct execution, assigned indices 0 through 7.
scripts/apollo_quality_shards.sh \
  runs/prepared_v4 runs/selector_v4 runs/quality_shards_v8 16 0 7

# Goose: submission from the login node, computation inside Slurm.
sbatch \
  --export=ALL,ISINGFOLD_CORPUS=runs/prepared_v4,ISINGFOLD_SELECTOR_BUNDLE=runs/selector_v4,ISINGFOLD_QUALITY_SHARD_ROOT=runs/quality_shards_v8,ISINGFOLD_QUALITY_SHARD_COUNT=16,ISINGFOLD_QUALITY_LINEAGES=0,ISINGFOLD_QUALITY_SEED=907,ISINGFOLD_SHARD_FIRST=8,ISINGFOLD_SHARD_LAST=15,ISINGFOLD_SHARD_STEP=2 \
  scripts/goose_quality_shards.sbatch
```

The three publisher variables and both ground-root variables must be exported on both hosts and
propagated by `sbatch --export=ALL`. If storage is not shared, transfer only completed immutable
shard directories. The repository revision, prepared manifest, selector bundle, publisher tree,
and ground-root tree must be byte-identical.

Merge only after all indices are present:

```bash
shard_args=()
for shard in runs/quality_shards_v8/shard-*-of-00016; do
  shard_args+=(--shard "$shard")
done

isingfold-rl merge-quality-labels \
  --corpus runs/prepared_v4 \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  --selector runs/selector_v4 \
  "${shard_args[@]}" \
  --device cpu \
  --out runs/quality_v8
```

The merge validates an exact, nonoverlapping, complete shard set and replays every quality-v7 row.
Shell globbing does not establish completeness. Preserve the source shards read-only because the
merged receipt binds their manifests and row hashes.

After the merge and preflight complete, record the preflight file SHA-256 outside its output
directory. All training and gate launchers receive these two values:

```bash
export QUALITY_PREFLIGHT_RECEIPT=runs/quality_preflight_v3.json
export EXPECTED_QUALITY_PREFLIGHT_SHA256='<externally recorded SHA-256>'

QUALITY_PREFLIGHT_ARGS=(
  --quality-preflight-receipt "$QUALITY_PREFLIGHT_RECEIPT"
  --expected-quality-preflight-sha256 "$EXPECTED_QUALITY_PREFLIGHT_SHA256"
)
```

## 4. Release gates

```bash
export EXACT_CONFORMANCE_CORPUS=/data/EmbedBench/exact_conformance_v1.json
export EXPECTED_EXACT_CONFORMANCE_SHA256='<out-of-band SHA-256 pin>'

isingfold-rl gates \
  --grid configs/rl_grid_hybrid_v1.json \
  --corpus runs/prepared_v4 \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  "${QUALITY_PREFLIGHT_ARGS[@]}" \
  --selector runs/selector_v4 \
  --selector-labels runs/selector_labels_v4 \
  --quality-labels runs/quality_v8 \
  --exact-conformance-corpus "$EXACT_CONFORMANCE_CORPUS" \
  --expected-exact-conformance-sha256 "$EXPECTED_EXACT_CONFORMANCE_SHA256" \
  --partition validation \
  --instances 6 \
  --candidates 12 \
  --seed 1009 \
  --device cuda \
  --out runs/gates_v7.json
```

The gate receipt v7 binds the same quality authority, quality-v8 corpus, exact quality-preflight
pin, and authenticated scalable-gate strata as the selector and labels. Gate 2 selects each
best-supported rewrite on a screening read block, then measures that rewrite and protected COMMIT
again on independent confirmation blocks. Its one-sided 95 percent normal lower confidence bound
adds the between-lineage standard error and the worst-case Bernoulli evaluator variance for a
difference of two 256-read rates. The bound must be at least 0.02. In every sampled joint stratum,
at least half of its rows must individually reach 0.02 confirmed headroom. Fewer than two eligible
independent lineages fail closed. The existing support-recall, within-support spread, and
broad-pool-gap requirements remain mandatory. These checks describe the frozen observed proposal
pool; they do not certify a global optimum. Mandatory Profile-I gates cover exact contracts,
candidate support, label signal, and valid-return learning signal.
Profile-C empty-start readiness is separate and does not enter this representation selection.
After publication, record the gate receipt SHA-256 outside `runs/`. Every representation consumer
must receive that independent pin; deriving it from the receipt at consumer startup would not
authenticate the artifact.

## 5. Frozen 9 plus 18 model grid

[`configs/rl_grid_hybrid_v1.json`](../configs/rl_grid_hybrid_v1.json) is the only current main-grid
registry. Its nine representation cells are the Cartesian product of three families and three
seeds:

The filename's `v1` identifies the first hybrid protocol revision. The authenticated payload is
staged-grid schema v2 with the exact name
`if-core-v2-profile-i-hybrid-chimera-registered`; schema v1 and the retired `profile-i-pilot` name
are rejected by every grid loader.

| Family | Seeds | Training method |
|---|---|---|
| IF-MLP | 1103, 2207, 3301 | tied-best actor ranking plus count-aware utility-critic calibration |
| IF-Dual | 1103, 2207, 3301 | tied-best actor ranking plus count-aware utility-critic calibration |
| IF-Core | 1103, 2207, 3301 | tied-best actor ranking plus count-aware utility-critic calibration |

Its eighteen RL-value cells use the selected simpler family and IF-Core, each under
`supervised-only`, `ppo-warm-start`, and `ppo-from-scratch`, at the same three seeds. The placeholder
`selected-simpler` is resolved only by the authenticated representation receipt. Manual resolution
is diagnostic and cannot create a scientific cell.

Each RL-value warm transfer is bound before checkpoint deserialization to the exact representation
seed and cell, the registered grid SHA-256, and the currently trusted quality-preflight file and
record digests. For receipt-selected cells, the loader also requires the checkpoint payload digest
from the matching `(cell_id, model_family, seed)` source report in the authenticated representation
selection. An internally valid checkpoint from another cell is therefore not interchangeable. See
[ADR-006](decisions/ADR-006-warm-start-transfer-provenance.md).

### Representation training and validation

Block host assignment by training seed. Every model family for seed 1103 runs on Apollo, every
family for seed 2207 runs on Goose, and every family for seed 3301 runs on Apollo. Keep the same
assignment for validation. This permits host names and platform details to differ across seeds,
while every family compared inside one seed has exactly the same runtime identity. The inference
device type remains a shared scientific control across all seeds.

```bash
export ISINGFOLD_GRID_CONFIG=configs/rl_grid_hybrid_v1.json
export ISINGFOLD_GATE_RECEIPT=runs/gates_v7.json
export EXPECTED_ISINGFOLD_GATE_RECEIPT_SHA256='<externally recorded gate-receipt SHA-256>'
export ISINGFOLD_GRID_STEP=3

# Apollo receives seed 1103 at indices 0,3,6 and seed 3301 at 2,5,8.
scripts/apollo_rl_grid.sh representation \
  runs/prepared_v4 runs/selector_v4 runs/quality_v8 runs/grid_v1 0 6
scripts/apollo_rl_grid.sh representation \
  runs/prepared_v4 runs/selector_v4 runs/quality_v8 runs/grid_v1 2 8

# Goose receives seed 2207 at indices 1,4,7 and executes through Slurm.
sbatch --array=0-2 \
  --export=ALL,ISINGFOLD_GRID_STAGE=representation,ISINGFOLD_GRID_FIRST=1,ISINGFOLD_GRID_STEP=3,ISINGFOLD_CORPUS=runs/prepared_v4,ISINGFOLD_SELECTOR_BUNDLE=runs/selector_v4,ISINGFOLD_QUALITY_LABELS=runs/quality_v8,ISINGFOLD_RUN_ROOT=runs/grid_v1,ISINGFOLD_GRID_CONFIG=configs/rl_grid_hybrid_v1.json,ISINGFOLD_GATE_RECEIPT=runs/gates_v7.json \
  scripts/goose_rl_grid.sbatch
```

Before evaluating those cells, create a dedicated target-free K=2 bank for the representation
seed domain. This bank is not interchangeable with the later RL-value validation bank: the
representation protocol uses evaluation seed 33049, while RL-value selection uses 44021. The
loader authenticates the preset, grid record, complete public census, seed schedule, K=2 cache,
runtime implementation, and external plan and manifest pins before opening validation targets.

```bash
isingfold-rl plan-bootstrap-bank \
  --grid configs/rl_grid_hybrid_v1.json \
  --corpus runs/prepared_v4 \
  --config configs/complete_system_lac_hybrid_cache_v1.json \
  --protocol-preset representation-validation \
  --out runs/bootstrap_representation_validation_v1/plan.json

export ISINGFOLD_VALIDATION_BOOTSTRAP_BANK=runs/bootstrap_representation_validation_v1
export EXPECTED_VALIDATION_BOOTSTRAP_PLAN_SHA256='<externally recorded SHA-256>'

# Generate all immutable shards on one publication runtime, then seal. The Goose packed
# launcher is suitable when the association cannot submit a large Slurm array.
export ISINGFOLD_GRID=configs/rl_grid_hybrid_v1.json
export ISINGFOLD_CORPUS=runs/prepared_v4
export ISINGFOLD_COMPLETE_CONFIG=configs/complete_system_lac_hybrid_cache_v1.json
export ISINGFOLD_BOOTSTRAP_PRESET=representation-validation
export ISINGFOLD_BOOTSTRAP_BANK_PLAN="$ISINGFOLD_VALIDATION_BOOTSTRAP_BANK/plan.json"
export ISINGFOLD_BOOTSTRAP_BANK_PLAN_SHA256="$EXPECTED_VALIDATION_BOOTSTRAP_PLAN_SHA256"
export ISINGFOLD_BOOTSTRAP_BANK="$ISINGFOLD_VALIDATION_BOOTSTRAP_BANK"
export ISINGFOLD_BOOTSTRAP_BANK_LOG_ROOT=runs/bootstrap_representation_validation_logs_v1
export ISINGFOLD_BOOTSTRAP_BANK_SHARD_COUNT=128
sbatch --export=ALL scripts/goose_bootstrap_bank_packed.sbatch

isingfold-rl seal-bootstrap-bank \
  --grid configs/rl_grid_hybrid_v1.json --corpus runs/prepared_v4 \
  --config configs/complete_system_lac_hybrid_cache_v1.json \
  --protocol-preset representation-validation \
  --plan "$ISINGFOLD_VALIDATION_BOOTSTRAP_BANK/plan.json" \
  --expected-plan-sha256 "$EXPECTED_VALIDATION_BOOTSTRAP_PLAN_SHA256" \
  --bank "$ISINGFOLD_VALIDATION_BOOTSTRAP_BANK"
export EXPECTED_VALIDATION_BOOTSTRAP_MANIFEST_SHA256='<externally recorded SHA-256>'
```

Evaluate exactly the same cells on validation:

```bash
export ISINGFOLD_GRID_STEP=3

scripts/apollo_representation_eval.sh \
  runs/prepared_v4 runs/selector_v4 runs/grid_v1 runs/representation_eval_v1 0 6
scripts/apollo_representation_eval.sh \
  runs/prepared_v4 runs/selector_v4 runs/grid_v1 runs/representation_eval_v1 2 8

sbatch --array=0-2 \
  --export=ALL,ISINGFOLD_GRID_FIRST=1,ISINGFOLD_GRID_STEP=3,ISINGFOLD_CORPUS=runs/prepared_v4,ISINGFOLD_SELECTOR_BUNDLE=runs/selector_v4,ISINGFOLD_RUN_ROOT=runs/grid_v1,ISINGFOLD_REP_EVALUATIONS_ROOT=runs/representation_eval_v1,ISINGFOLD_GRID_CONFIG=configs/rl_grid_hybrid_v1.json \
  scripts/goose_representation_eval.sbatch

isingfold-rl select-representation \
  --grid configs/rl_grid_hybrid_v1.json \
  --evaluations-root runs/representation_eval_v1 \
  --out runs/representation_selection_v4.json
```

Selection uses the registered validation seed, four repetitions, 4096 reads, categorical
temperature-one deployment, a 0.02 feasibility noninferiority margin, and 20,000 bootstrap
replicates. Each replicate independently resamples the three training seeds and immutable base
lineages with replacement, then applies equal weights across both axes. This crossed bootstrap
retains training-procedure uncertainty and lineage uncertainty. It never treats seed by lineage
rows as independent IID samples.

### RL-value training, validation, and freeze

Before starting this stage, build and seal the three deployment-initializer banks described in
"Sealed deployment-initializer banks" below. Seed indices 0, 1, and 2 bind training seeds 1103,
2207, and 3301 respectively. Every PPO cell reuses the read-only bank for its seed; the launcher
requires its external manifest pin, the grid dispatch authenticates the preregistered
complete-system configuration, and PPO setup authenticates the bank before its first rollout.
Supervised-only cells keep the same sealed interface and persistent K=2 restart-cache context.
Their receipts record whether the authenticated initializer and cache snapshots were consumed.
The validation bootstrap below uses preset `validation` and seed 44021. Do not point it at the
representation-validation bank even though both cover the same public validation partition.

```bash
export ISINGFOLD_SELECTION_RECEIPT=runs/representation_selection_v4.json
export ISINGFOLD_GRID_STEP=3
export ISINGFOLD_COMPLETE_CONFIG=configs/complete_system_lac_hybrid_cache_v1.json
export ISINGFOLD_INITIALIZER_BANK_ROOT=runs/initializer_banks
export ISINGFOLD_INITIALIZER_BANK_MANIFEST_SHA256_0='<externally recorded SHA-256>'
export ISINGFOLD_INITIALIZER_BANK_MANIFEST_SHA256_1='<externally recorded SHA-256>'
export ISINGFOLD_INITIALIZER_BANK_MANIFEST_SHA256_2='<externally recorded SHA-256>'

# Apollo receives seed 1103 at 0,3,...,15 and seed 3301 at 2,5,...,17.
scripts/apollo_rl_grid.sh rl_value \
  runs/prepared_v4 runs/selector_v4 runs/quality_v8 runs/grid_v1 0 15
scripts/apollo_rl_grid.sh rl_value \
  runs/prepared_v4 runs/selector_v4 runs/quality_v8 runs/grid_v1 2 17

# Goose receives seed 2207 at 1,4,...,16.
sbatch --array=0-5 \
  --export=ALL,ISINGFOLD_GRID_STAGE=rl_value,ISINGFOLD_GRID_FIRST=1,ISINGFOLD_GRID_STEP=3,ISINGFOLD_CORPUS=runs/prepared_v4,ISINGFOLD_SELECTOR_BUNDLE=runs/selector_v4,ISINGFOLD_QUALITY_LABELS=runs/quality_v8,ISINGFOLD_RUN_ROOT=runs/grid_v1,ISINGFOLD_SELECTION_RECEIPT=runs/representation_selection_v4.json,ISINGFOLD_GRID_CONFIG=configs/rl_grid_hybrid_v1.json \
  scripts/goose_rl_grid.sbatch

# Build this once, before validation quality targets are opened. The plan binds the public
# deployment envelope; the quality authority is authenticated separately by evaluation.
isingfold-rl plan-bootstrap-bank \
  --grid configs/rl_grid_hybrid_v1.json \
  --corpus runs/prepared_v4 \
  --config configs/complete_system_lac_hybrid_cache_v1.json \
  --protocol-preset validation \
  --out runs/bootstrap_validation_v1/plan.json
export ISINGFOLD_VALIDATION_BOOTSTRAP_BANK=runs/bootstrap_validation_v1
export EXPECTED_VALIDATION_BOOTSTRAP_PLAN_SHA256='<externally recorded SHA-256>'

# Split the exact 128-shard census without overlap. Apollo runs directly because it has no
# Slurm. Goose must submit the odd-index shards as a Slurm array; login-node execution fails.
scripts/apollo_bootstrap_bank.sh \
  configs/rl_grid_hybrid_v1.json runs/prepared_v4 \
  configs/complete_system_lac_hybrid_cache_v1.json validation \
  "$ISINGFOLD_VALIDATION_BOOTSTRAP_BANK/plan.json" \
  "$EXPECTED_VALIDATION_BOOTSTRAP_PLAN_SHA256" \
  "$ISINGFOLD_VALIDATION_BOOTSTRAP_BANK" 128 0 126 2

export ISINGFOLD_GRID=configs/rl_grid_hybrid_v1.json
export ISINGFOLD_CORPUS=runs/prepared_v4
export ISINGFOLD_COMPLETE_CONFIG=configs/complete_system_lac_hybrid_cache_v1.json
export ISINGFOLD_BOOTSTRAP_PRESET=validation
export ISINGFOLD_BOOTSTRAP_BANK_PLAN="$ISINGFOLD_VALIDATION_BOOTSTRAP_BANK/plan.json"
export ISINGFOLD_BOOTSTRAP_BANK_PLAN_SHA256="$EXPECTED_VALIDATION_BOOTSTRAP_PLAN_SHA256"
export ISINGFOLD_BOOTSTRAP_BANK="$ISINGFOLD_VALIDATION_BOOTSTRAP_BANK"
export ISINGFOLD_BOOTSTRAP_BANK_SHARD_COUNT=128
sbatch --array=1-127:2 --export=ALL scripts/goose_bootstrap_bank.sbatch

# If Apollo and Goose do not mount the same run root, copy each immutable shard directory
# plus its sidecars into this bank root. Do not rename or merge rows manually. Seal only after
# all indices 0,...,127 are present; sealing authenticates the complete census.
isingfold-rl seal-bootstrap-bank \
  --grid configs/rl_grid_hybrid_v1.json --corpus runs/prepared_v4 \
  --config configs/complete_system_lac_hybrid_cache_v1.json \
  --protocol-preset validation \
  --plan "$ISINGFOLD_VALIDATION_BOOTSTRAP_BANK/plan.json" \
  --expected-plan-sha256 "$EXPECTED_VALIDATION_BOOTSTRAP_PLAN_SHA256" \
  --bank "$ISINGFOLD_VALIDATION_BOOTSTRAP_BANK"
export EXPECTED_VALIDATION_BOOTSTRAP_MANIFEST_SHA256='<externally recorded SHA-256>'

export ISINGFOLD_GRID_STEP=3

scripts/apollo_rl_value_eval.sh \
  runs/prepared_v4 runs/selector_v4 runs/grid_v1 \
  runs/representation_selection_v4.json runs/rl_value_eval_v1 0 15
scripts/apollo_rl_value_eval.sh \
  runs/prepared_v4 runs/selector_v4 runs/grid_v1 \
  runs/representation_selection_v4.json runs/rl_value_eval_v1 2 17

sbatch --array=0-5 \
  --export=ALL,ISINGFOLD_GRID_FIRST=1,ISINGFOLD_GRID_STEP=3,ISINGFOLD_CORPUS=runs/prepared_v4,ISINGFOLD_SELECTOR_BUNDLE=runs/selector_v4,ISINGFOLD_RUN_ROOT=runs/grid_v1,ISINGFOLD_REPRESENTATION_SELECTION_RECEIPT=runs/representation_selection_v4.json,ISINGFOLD_RL_VALUE_EVALUATIONS_ROOT=runs/rl_value_eval_v1,ISINGFOLD_GRID_CONFIG=configs/rl_grid_hybrid_v1.json \
  scripts/goose_rl_value_eval.sbatch

isingfold-rl select-rl-value \
  --grid configs/rl_grid_hybrid_v1.json \
  --representation-selection-receipt runs/representation_selection_v4.json \
  --evaluations-root runs/rl_value_eval_v1 \
  --out runs/rl_value_selection_v4.json
```

Keep `ISINGFOLD_GRID_STEP=3` for both Apollo launchers and all Goose arrays. Do not move a cell to a
different host between training and validation, and do not split configurations of one training
seed across hosts. Synchronize only complete cell directories before either selection command.
Each freeze receipt records the runtime identity for each seed and rejects any same-seed mismatch.
The RL-value selector authenticates all eighteen reports and again uses a crossed seed by lineage
bootstrap. Its primary metric is unconditional IF-Q3-S0 utility; lower online seconds and then
registered configuration order are tie-breaks. The equal-seed aggregate makes latency comparison
a seed-blocked comparison even though Apollo and Goose report different host-specific identities.

Every training directory contains `checkpoint.pt`, `run.json`, and `history.json`. Warm-start
history `isingfold.warm-start-history` v4 records corpus-normalized Q, delta, rank, utility, and
total losses, the exact denominator census, memory-minibatch count, and one optimizer step per full
corpus pass. Memory minibatches bound GPU use but do not define the statistical reduction; see
[ADR-005](decisions/ADR-005-exact-warm-start-corpus-reduction.md). PPO history v2 records accepted
and rejected epoch attempts, rolled-back optimizer steps, full-buffer KL, the early-stop reason,
normalized entropy, and the effective learning rate. Checkpoint v3
binds model and optimizer state, Python/NumPy/Torch RNG state, experiment contract, normalizer,
action schema, proposal identity, selector and quality authority, training lineages, mutable PPO
state, and a digest of the runtime source registry. Run v4 binds the checkpoint payload digest.
Resume rejects any changed source, context, corpus, authority, support, family, schedule, or lineage
set.

The PPO update controls are optimizer-only. They do not revise IF-Q3-S0, terminal reward, any data
partition or label, deployment action selection, or the publication endpoint.

### Post-selection direct-Q warm control

The main grid trains the three IF-Core treatment cells with `full-qmu-v4`. After the
representation-selection receipt is sealed, run the registered rank-plus-value controls from
[`configs/quality_warm_control_hybrid_v1.json`](../configs/quality_warm_control_hybrid_v1.json). The config record
digest is `73f99e6f1b157d638d9348ec67a9cf0814dcb3518b3cdd63e56a38c5e6ad4fb0`; its current file SHA-256
is `afa85214c9f1b1aa20a90bc5652ac0c59d74a350c6814477409157b532cbac17`. Record the file SHA-256
outside the run tree before launching.

The control changes only `lambda_q` and `lambda_delta` from 1 and 0.5 to zero. IF-Core parameters,
bounded policy coupling, quality rows, exact corpus denominators, initial seed, permutation,
minibatch 32, optimizer, and 200 full-corpus updates remain fixed. The launcher authenticates each
source treatment receipt and requires the same runtime, device type, corpus, selector, normalizer,
quality manifest, preflight, and checkpoint identity. A control checkpoint is not eligible for PPO
warm transfer.

```bash
export EXPECTED_QUALITY_WARM_CONTROL_CONFIG_SHA256=\
'afa85214c9f1b1aa20a90bc5652ac0c59d74a350c6814477409157b532cbac17'
export EXPECTED_ISINGFOLD_GRID_SHA256=\
'd3a7cd99c0c96c1c2f7fdd974d0856aee94ccbbe7a01f30b15d4724632712d09'
export ISINGFOLD_SELECTION_RECEIPT=runs/representation_selection_v4.json
export EXPECTED_ISINGFOLD_SELECTION_RECEIPT_SHA256='<externally recorded SHA-256>'

# Assign complete paired seeds to one host. Apollo runs directly.
scripts/apollo_quality_warm_control.sh \
  runs/prepared_v4 runs/selector_v4 runs/quality_v8 runs/grid_v1 \
  runs/diagnostics_v1 0 0
scripts/apollo_quality_warm_control.sh \
  runs/prepared_v4 runs/selector_v4 runs/quality_v8 runs/grid_v1 \
  runs/diagnostics_v1 2 2

# Goose handles seed 2207 through Slurm.
sbatch --array=1-1 scripts/goose_quality_warm_control.sbatch
```

The launchers also require the publisher, ground-certificate, quality-preflight, and quality
initializer-bank environment variables used by the main grid. This is a diagnostic comparison on
validation. Aggregate all three seed pairs by immutable base lineage and report negative, null, and
positive effects. Do not use it to revise representation selection, RL-value selection, or the
sealed test protocol. See
[ADR-007](decisions/ADR-007-direct-qmu-warm-control.md).

### Selector audit after the RL-value freeze

The final RL-value receipt is the capability that permits test-side selector-audit labels. Record
its file SHA-256 outside the run directory before opening `audit_test`. These labels cannot be used
by `fit-selector` and cannot revise either model-selection receipt.

```bash
export ISINGFOLD_RL_FREEZE=runs/rl_value_selection_v4.json
export ISINGFOLD_RL_FREEZE_SHA256='<externally recorded SHA-256>'

isingfold-rl label-selector-data \
  --corpus runs/prepared_v4 \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  --seed 1709 \
  --split-seed 601 \
  --calibration-fraction 0.2 \
  --audit-mode \
  --selector runs/selector_v4 \
  --grid configs/rl_grid_hybrid_v1.json \
  --rl-value-selection-receipt "$ISINGFOLD_RL_FREEZE" \
  --expected-rl-value-selection-sha256 "$ISINGFOLD_RL_FREEZE_SHA256" \
  --out runs/selector_labels_audit_v4

isingfold-rl audit-selector \
  --corpus runs/prepared_v4 \
  --selector runs/selector_v4 \
  --selector-labels runs/selector_labels_audit_v4 \
  --partition audit_val \
  --out runs/selector_audit_val_v1

isingfold-rl audit-selector \
  --corpus runs/prepared_v4 \
  --selector runs/selector_v4 \
  --selector-labels runs/selector_labels_audit_v4 \
  --partition audit_test \
  --grid configs/rl_grid_hybrid_v1.json \
  --rl-value-selection-receipt "$ISINGFOLD_RL_FREEZE" \
  --expected-rl-value-selection-sha256 "$ISINGFOLD_RL_FREEZE_SHA256" \
  --out runs/selector_audit_test_v1
```

### Sealed deployment-initializer banks

Every PPO cell in the RL-value grid and every fresh complete-system PPO confirmation must train on
the same initializer distribution used at deployment, not on a hand-authored CandidateBank
incumbent. Build one target-free bank for each training seed before RL-value training. The bank is
then reused read-only by all matched family/method cells and by complete-system confirmation for
that seed. For a PPO cell in the registered grid, the plan contains
`updates * episodes = 100 * 64 = 6400` conditional episodes. If the method is `supervised-only`,
the command-line bank argument remains part of the selected-system interface but the run receipt
records that no PPO initializer snapshot was consumed.

For each conditional episode, the plan chooses an immutable base lineage by canonical round robin,
chooses an instance by a seeded cyclic offset, and fixes a finite sequence of complete LAC
initializer draws. Generation stops after the first successful draw. All rejected draws and their
work remain in the manifest. Exhausting the frozen draw cap for any episode prevents sealing. Bank
planning and generation load public train tasks only, set `opened_evaluator_targets=false`, and bind
the exact native extension, Python sources, complete-system config, work context, training seed, and
episode schedule.

Example for selected-seed index 0, whose registered training seed is 1103:

```bash
export ISINGFOLD_COMPLETE_CONFIG=configs/complete_system_lac_hybrid_cache_v1.json
export ISINGFOLD_INITIALIZER_BANK_DRAW_CAP='<positive integer frozen before generation>'

isingfold-rl plan-initializer-bank \
  --corpus runs/prepared_v4 \
  --config "$ISINGFOLD_COMPLETE_CONFIG" \
  --training-seed 1103 \
  --episode-schedule-start 0 \
  --episode-count 6400 \
  --max-draws-per-conditional-episode "$ISINGFOLD_INITIALIZER_BANK_DRAW_CAP" \
  --out runs/initializer_bank_plans/seed-0.json

export ISINGFOLD_INITIALIZER_BANK_PLAN=runs/initializer_bank_plans/seed-0.json
export ISINGFOLD_INITIALIZER_BANK_PLAN_SHA256='<recorded outside the bank tree>'

# Apollo is direct. This example generates all 16 deterministic shards.
scripts/apollo_initializer_bank.sh \
  runs/prepared_v4 "$ISINGFOLD_COMPLETE_CONFIG" \
  "$ISINGFOLD_INITIALIZER_BANK_PLAN" "$ISINGFOLD_INITIALIZER_BANK_PLAN_SHA256" \
  runs/initializer_banks/seed-0 16 0 15

isingfold-rl seal-initializer-bank \
  --corpus runs/prepared_v4 \
  --config "$ISINGFOLD_COMPLETE_CONFIG" \
  --plan "$ISINGFOLD_INITIALIZER_BANK_PLAN" \
  --expected-plan-sha256 "$ISINGFOLD_INITIALIZER_BANK_PLAN_SHA256" \
  --bank runs/initializer_banks/seed-0

export ISINGFOLD_INITIALIZER_BANK_ROOT=runs/initializer_banks
export ISINGFOLD_INITIALIZER_BANK_MANIFEST_SHA256_0='<manifest SHA-256 printed by seal>'
```

For seed index 1, submit `scripts/goose_initializer_bank.sbatch` as a Slurm array after exporting
`ISINGFOLD_CORPUS`, `ISINGFOLD_COMPLETE_CONFIG`, `ISINGFOLD_INITIALIZER_BANK_PLAN`,
`ISINGFOLD_INITIALIZER_BANK_PLAN_SHA256`, `ISINGFOLD_INITIALIZER_BANK`, and
`ISINGFOLD_INITIALIZER_BANK_SHARD_COUNT`. The script rejects login-node execution and uses `srun`.
If the Goose association limits the number of submitted array elements, use the registered
`scripts/goose_initializer_bank_packed.sbatch` fallback as one non-array Slurm job. Also export
`ISINGFOLD_INITIALIZER_BANK_LOG_ROOT` under the approved output base and set
`ISINGFOLD_INITIALIZER_BANK_WORKERS` no higher than `SLURM_CPUS_PER_TASK` or the shard count. The
packed launcher uses concurrent `srun --overlap --exact` steps and preserves the same immutable
shard identities; it is a scheduling change, not a scientific change.
Build seed index 2 with training seed 3301. Each bank has its own externally recorded plan and
manifest pins.

## 6. Fresh three-seed complete-system confirmation

The RL-value receipt freezes one configuration across all three training seeds. Confirmation does
not reuse a selected validation checkpoint. It retrains seeds 1103, 2207, and 3301 from the selected
configuration, with fresh output directories and no resume. Policy restarts are disabled because
native replay support is not part of this contract.

Pin the RL-value receipt hash in the experiment registry before opening test outcomes:

```bash
export ISINGFOLD_GRID_CONFIG=configs/rl_grid_hybrid_v1.json
export ISINGFOLD_RL_FREEZE=runs/rl_value_selection_v4.json
export ISINGFOLD_RL_FREEZE_SHA256='<externally recorded SHA-256>'
export ISINGFOLD_COMPLETE_CONFIG=configs/complete_system_lac_hybrid_cache_v1.json
export ISINGFOLD_INITIALIZER_BANK_ROOT=runs/initializer_banks
export ISINGFOLD_INITIALIZER_BANK_MANIFEST_SHA256_0='<externally recorded SHA-256>'
export ISINGFOLD_INITIALIZER_BANK_MANIFEST_SHA256_1='<externally recorded SHA-256>'
export ISINGFOLD_INITIALIZER_BANK_MANIFEST_SHA256_2='<externally recorded SHA-256>'

# Apollo example: selected-seed indices 0 and 2.
scripts/apollo_complete_train.sh \
  runs/prepared_v4 runs/selector_v4 runs/quality_v8 runs/complete_train_v1 \
  "$ISINGFOLD_RL_FREEZE" "$ISINGFOLD_RL_FREEZE_SHA256" 0 2 2

# Goose example: selected-seed index 1 only.
sbatch --array=1-1 \
  --export=ALL,ISINGFOLD_CORPUS=runs/prepared_v4,ISINGFOLD_SELECTOR_BUNDLE=runs/selector_v4,ISINGFOLD_QUALITY_LABELS=runs/quality_v8,ISINGFOLD_RUN_ROOT=runs/complete_train_v1,ISINGFOLD_RL_FREEZE=runs/rl_value_selection_v4.json,ISINGFOLD_RL_FREEZE_SHA256="$ISINGFOLD_RL_FREEZE_SHA256",ISINGFOLD_GRID_CONFIG=configs/rl_grid_hybrid_v1.json \
  scripts/goose_complete_train.sbatch
```

Evaluate each fresh checkpoint with
[`configs/complete_system_lac_hybrid_cache_v1.json`](../configs/complete_system_lac_hybrid_cache_v1.json). The denominator is
the full sealed policy-instance population before initialization. Initializer failures are ordinary
method failures with utility zero. Valid returns receive one fresh 4096-read evaluator block at the
strength chosen by the frozen selector. The report retains raw complete receipts, outcomes, terminal
evidence, work, seeds, and failure denominators.

The learned arm uses `lac-minorminer-hybrid-chimera-clique-v1-initializer-v4` and native
work-counter schema v3. Before each initializer call, the runner subtracts all prior work and
reserves the top-level invocation,
independent candidate screen, environment bootstrap, and terminal policy path. The exact remainder
is passed into C++, where every registered charge is checked prospectively. A native cap stop is an
explicit `WORK_BUDGET_EXHAUSTED` attempt with a named coordinate and a non-exceeding work snapshot.
The runtime initializer identity hashes the bytes of the loaded native extension plus every Python
source file in the imported `lac_minorminer` package. An unresolved extension or incomplete source
manifest aborts before evaluation. The registered native transition ceiling is 31, leaving the
32nd decision unit for the policy terminal path.

The staged grid pins both the semantic digest and the byte-level SHA-256 of the learned and stock
complete-system configurations. Every test-opening command validates the relevant config pins
before loading a test task. Editing either config after registration requires a new grid and a new
experiment, not an in-place rerun.

Before opening the final-test targets, repeat `plan-bootstrap-bank`, shard generation, and sealing
with `--protocol-preset final-test`. Export the independently recorded paths and pins as
`ISINGFOLD_FINAL_TEST_BOOTSTRAP_BANK`, `EXPECTED_FINAL_TEST_BOOTSTRAP_PLAN_SHA256`, and
`EXPECTED_FINAL_TEST_BOOTSTRAP_MANIFEST_SHA256`. Every learned seed consumes byte-identical
target-free initial/cache support from this bank. Evaluator targets and their authority remain a
separate post-authentication binding; the protocol does not claim equality of target-bearing full
envelopes.

Both complete-evaluation launchers write the fixed learned directory layout
`EVALUATION_ROOT/seed-0`, `EVALUATION_ROOT/seed-1`, and `EVALUATION_ROOT/seed-2`. Do not rename those
directories. `aggregate-complete-system` consumes that exact layout and rejects missing or aliased
seed directories. They also run the matching stock row immediately afterward under the same host,
device, thread, and determinism identity. Set `ISINGFOLD_EXTERNAL_COMPLETE_OUT_ROOT` to the stock
output root before either launcher.

```bash
# Apollo evaluates selected-seed indices 0 and 2 directly.
export ISINGFOLD_EXTERNAL_COMPLETE_OUT_ROOT=runs/external_complete_v4
scripts/apollo_complete_eval.sh \
  runs/prepared_v4 runs/selector_v4 runs/complete_train_v1 \
  configs/complete_system_lac_hybrid_cache_v1.json runs/complete_eval_v1 \
  "$ISINGFOLD_RL_FREEZE" "$ISINGFOLD_RL_FREEZE_SHA256" 0 2 2

# Goose evaluates selected-seed index 1 inside Slurm.
sbatch --array=1-1 \
  --export=ALL,ISINGFOLD_CORPUS=runs/prepared_v4,ISINGFOLD_SELECTOR_BUNDLE=runs/selector_v4,ISINGFOLD_RUN_ROOT=runs/complete_train_v1,ISINGFOLD_COMPLETE_CONFIG=configs/complete_system_lac_hybrid_cache_v1.json,ISINGFOLD_COMPLETE_OUT_ROOT=runs/complete_eval_v1,ISINGFOLD_EXTERNAL_COMPLETE_OUT_ROOT=runs/external_complete_v4,ISINGFOLD_RL_FREEZE=runs/rl_value_selection_v4.json,ISINGFOLD_RL_FREEZE_SHA256="$ISINGFOLD_RL_FREEZE_SHA256",ISINGFOLD_GRID_CONFIG=configs/rl_grid_hybrid_v1.json \
  scripts/goose_complete_eval.sbatch
```

After all three evaluation directories have been synchronized, aggregate locally. This command is
authentication and statistics only; it exposes no device, seed, partition, repetition, or compute
override.

```bash
isingfold-rl aggregate-complete-system \
  --grid configs/rl_grid_hybrid_v1.json \
  --corpus runs/prepared_v4 \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  --evaluation-root runs/complete_eval_v1 \
  --config configs/complete_system_lac_hybrid_cache_v1.json \
  --rl-value-selection-receipt "$ISINGFOLD_RL_FREEZE" \
  --expected-selection-sha256 "$ISINGFOLD_RL_FREEZE_SHA256" \
  --out runs/complete_system_aggregate_v2.json
```

Aggregation authenticates each report and its raw receipt, outcome, and terminal-evidence
sidecars; requires the exact three selected training seeds and a common population, context,
initializer, selector, quality authority, and protocol; and rejects missing or conditional-only
rows. Its two-sided 95 percent interval uses 20,000 independent-with-replacement crossed bootstrap
replicates over training seeds and immutable base lineages.

## 7. Capacity-control diagnostic

[`configs/rl_capacity_control_hybrid_v1.json`](../configs/rl_capacity_control_hybrid_v1.json) is outside the fixed
9 plus 18 selection. It activates only if the selected simpler family is IF-Dual. The registered
comparison freshly retrains, for all three paired seeds, the width-128 IF-Core reference and a
width-128 nine-local-block IF-Dual control. Their trainable-parameter gap is bounded by 0.1 percent
of the IF-Core reference count.

The diagnostic cannot change either main-grid receipt. It can support only the narrower claim that
ownership/conflict fusion was compared with additional within-stream depth at near-matched trainable
capacity. Depth and inference compute remain confounded. Therefore publish instantiated trainable
and total parameter counts, all three paired-seed outcomes, inference latency, and peak accelerator
memory. Do not call IF-MLP versus IF-Core capacity-controlled.

The registry is fail-closed through `load_capacity_control_registry` and
`build_capacity_diagnostic_plan` in `isingfold.rl.capacity_control`. A separate runner may consume
that authenticated plan, but its directories and reports must not be placed inside the nine-cell or
eighteen-cell roots.

## 8. External complete-system baseline

Stock minorminer is a separate whole-system arm, not a shared-support ablation. Its deployment
strategy is selected once on validation and frozen before any test target is loaded. The immutable
registry contains the documented stock default, time-saturating resource ranking, and
time-saturating frozen-selector quality ranking across a fixed patience grid. Every candidate uses
three tuning seeds, four repetitions, the exact full validation population, and at least 128
independent lineages. Candidate selection averages equally over tuning seed and immutable lineage;
it never selects a seed.

Record the checked-in registry SHA-256 outside the run directory, then execute all seven candidates
for each tuning seed. Keep the seven candidates for one seed on the same machine. Apollo runs
directly; Goose uses one three-row Slurm array whose array job loops over all seven candidates.

```bash
export EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256=\
'0ddeaa14f4179f0e0abae03b58d3e9aa869501c104fb12ada413c7b5465b9e62'

# Apollo example: tuning seeds 0 and 2, direct execution.
scripts/apollo_external_tuning.sh \
  runs/prepared_v4 runs/selector_v4 runs/external_tuning_v2 0 2 2

# Goose example: tuning seed 1, computation only through Slurm.
sbatch --array=1-1 \
  --export=ALL,ISINGFOLD_CORPUS=runs/prepared_v4,ISINGFOLD_SELECTOR_BUNDLE=runs/selector_v4,ISINGFOLD_EXTERNAL_TUNING_OUT_ROOT=runs/external_tuning_v2,EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256="$EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256" \
  scripts/goose_external_tuning.sbatch

isingfold-rl select-external-tuning \
  --registry configs/external_minorminer_tuning_hybrid_v1.json \
  --expected-registry-sha256 "$EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256" \
  --grid configs/rl_grid_hybrid_v1.json \
  --external-config configs/external_minorminer_complete_v1.json \
  --evaluations-root runs/external_tuning_v2 \
  --out runs/external_tuning_selection_v2.json

export EXTERNAL_TUNING_SELECTION=runs/external_tuning_selection_v2.json
export EXPECTED_EXTERNAL_TUNING_SELECTION_SHA256='<recorded outside the run directory>'
```

Only after those two selection pins are frozen may the test arm run. Three stock test runs are
required, one matched rowwise to learned training seeds 1103, 2207, and 3301. The paper command
projects prepared-v4 candidate-group rows to the same sealed pre-initialization policy-instance
census as the learned arm. Its test seed, four repetitions, audit reads, denominator, and bootstrap
protocol come from the registered grid and have no command-line override.

```bash
isingfold-rl evaluate-external-complete-system \
  --grid configs/rl_grid_hybrid_v1.json \
  --index 0 \
  --corpus runs/prepared_v4 \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  --selector runs/selector_v4 \
  --config configs/external_minorminer_complete_v1.json \
  --learned-config configs/complete_system_lac_hybrid_cache_v1.json \
  --tuning-registry configs/external_minorminer_tuning_hybrid_v1.json \
  --expected-tuning-registry-sha256 "$EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256" \
  --external-tuning-selection "$EXTERNAL_TUNING_SELECTION" \
  --expected-external-tuning-selection-sha256 \
    "$EXPECTED_EXTERNAL_TUNING_SELECTION_SHA256" \
  --device cuda \
  --threads 1 \
  --out runs/external_complete_v4/seed-0
```

The recommended launch path is the combined complete-evaluation launcher in Section 6 because it
runs each learned and stock row inside one host or one Goose allocation. The standalone Apollo
launcher takes `CORPUS SELECTOR OUT_ROOT FIRST LAST [STEP]`. The standalone Goose launcher is a
three-row Slurm array. A standalone Goose run is admissible for latency comparison only when its
reported hostname and complete compute identity exactly match the corresponding learned row.

The stock-default candidate makes one native call with ten internal tries. Time-saturating
candidates make independently seeded one-try calls until the 60-second deadline or the 64-call
Context cap. Resource variants rank by qubits and maximum chain. Quality variants compile all four
programs for each valid candidate and rank first by the frozen selector's maximum predicted
`p_solve`, then by resources. Neither ranking opens evaluator outcomes. The chosen candidate alone
receives one fresh 4096-read final block. Every timeout or ordinary failure remains utility zero with
a null evaluator seed. Every valid result retains its embedding, four programs, selected program,
validation receipt, selector reranking evidence, and evaluator count block. Native internal work
coordinates unavailable from the public API remain null. The systems share wall-clock and
prospective Context caps, but the paper does not claim equal internal algorithmic work.

Record SHA-256 pins for all three learned and all three stock `report.json` files outside the
evaluation directories. The final paired command accepts only those six pinned reports. It requires an
identical population, pair census, context/work cap, selector, quality authority, system-seed
schedule, evaluator-seed schedule, learned configuration identity, and symmetric online envelope.

```bash
isingfold-rl aggregate-learned-vs-stock \
  --grid configs/rl_grid_hybrid_v1.json \
  --corpus runs/prepared_v4 \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  --learned-config configs/complete_system_lac_hybrid_cache_v1.json \
  --external-config configs/external_minorminer_complete_v1.json \
  --tuning-registry configs/external_minorminer_tuning_hybrid_v1.json \
  --expected-tuning-registry-sha256 "$EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256" \
  --external-tuning-selection "$EXTERNAL_TUNING_SELECTION" \
  --expected-external-tuning-selection-sha256 \
    "$EXPECTED_EXTERNAL_TUNING_SELECTION_SHA256" \
  --learned-evaluation runs/complete_eval_v1/seed-0 \
  --expected-learned-report-sha256 "$LEARNED_REPORT_SHA_0" \
  --learned-evaluation runs/complete_eval_v1/seed-1 \
  --expected-learned-report-sha256 "$LEARNED_REPORT_SHA_1" \
  --learned-evaluation runs/complete_eval_v1/seed-2 \
  --expected-learned-report-sha256 "$LEARNED_REPORT_SHA_2" \
  --external-evaluation runs/external_complete_v4/seed-0 \
  --expected-external-report-sha256 "$STOCK_REPORT_SHA_0" \
  --external-evaluation runs/external_complete_v4/seed-1 \
  --expected-external-report-sha256 "$STOCK_REPORT_SHA_1" \
  --external-evaluation runs/external_complete_v4/seed-2 \
  --expected-external-report-sha256 "$STOCK_REPORT_SHA_2" \
  --out runs/learned_vs_stock_complete_v4.json
```

The primary estimate is the paired learned-minus-stock unconditional IF-Q3-S0 utility difference.
It first averages repetitions and instances within each immutable lineage, then gives equal weight
to each lineage and each of the three training seeds. A 20,000-replicate crossed seed-by-lineage
bootstrap produces the two-sided 95 percent utility interval and the one-sided feasibility lower
bound. The two confirmatory claims use fixed-sequence gatekeeping at familywise alpha 0.05. Global
feasibility noninferiority is tested first. Learned utility superiority is tested only if that gate
passes, its one-sided lower confidence bound is above zero, and its registered sample-size and
precision guards pass. If noninferiority fails, superiority is reported as not tested and false.
Each per-seed row reports learned and stock total online wall-clock seconds. The aggregator rejects
any row whose hostname, platform, device name/type, thread count, or deterministic setting differs.

The older `evaluate-external` command remains a diagnostic interface. Prepared-v4 input is also
projected to policy instances, but its free seed, partition, and repetition controls and its v1
receipt authority are not admissible for the paper comparison.

## 9. Resumable complete-system evaluation

Long evaluation uses the common `plan-evaluation-shards`, `run-evaluation-shard`, and
`merge-evaluation-shards` protocol. It supports `learned-complete-system`,
`tuned-stock-complete-system`, and `external-validation-tuning`. Create one plan per workflow cell
and compute class. A plan authenticates the publisher and target-free ground root, reconstructs the
entire public validation or test population, fixes run coordinates, derives seeds from lineage,
instance, and repetition identity, and partitions whole base lineages into groups of at most 32.
It does not open a target file or ground partition receipt.

The following example plans learned test seed index 0 on Apollo:

```bash
export ISINGFOLD_ENV_LOCK=/opt/isingfold/environment.lock
export ISINGFOLD_ENV_LOCK_SHA256='<out-of-band environment-lock SHA-256>'

isingfold-rl plan-evaluation-shards \
  --workflow learned-complete-system \
  --grid configs/rl_grid_hybrid_v1.json \
  --corpus runs/prepared_v4 \
  --selector runs/selector_v4 \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  --config configs/complete_system_lac_hybrid_cache_v1.json \
  --run-root runs/complete_train_v1 \
  --rl-value-selection-receipt "$ISINGFOLD_RL_FREEZE" \
  --expected-selection-sha256 "$ISINGFOLD_RL_FREEZE_SHA256" \
  --index 0 \
  --device cuda \
  --threads 1 \
  --cluster apollo \
  --execution-mode pinned-venv \
  --environment-lock "$ISINGFOLD_ENV_LOCK" \
  --expected-environment-lock-sha256 "$ISINGFOLD_ENV_LOCK_SHA256" \
  --max-lineages-per-shard 32 \
  --out runs/evaluation_plans/learned-seed-0.json

export LEARNED_EVAL_PLAN=runs/evaluation_plans/learned-seed-0.json
export LEARNED_EVAL_PLAN_SHA256='<plan SHA-256 printed by the planner>'
export LEARNED_EVAL_SHARD_COUNT='<length of the sealed plan shard list>'

scripts/apollo_evaluation_shards.sh \
  learned-complete-system "$LEARNED_EVAL_PLAN" "$LEARNED_EVAL_PLAN_SHA256" \
  "$LEARNED_EVAL_SHARD_COUNT" runs/evaluation_shards/learned-seed-0 -- \
  --grid configs/rl_grid_hybrid_v1.json \
  --corpus runs/prepared_v4 \
  --selector runs/selector_v4 \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  --config configs/complete_system_lac_hybrid_cache_v1.json \
  --run-root runs/complete_train_v1 \
  --rl-value-selection-receipt "$ISINGFOLD_RL_FREEZE" \
  --expected-selection-sha256 "$ISINGFOLD_RL_FREEZE_SHA256" \
  --index 0 \
  --device cuda \
  --threads 1
```

Record every `shard_receipt.json` SHA-256 outside the shard tree. Merge by repeating one `--shard`
and one aligned `--expected-shard-receipt-sha256` for every index in plan order, while repeating the
same workflow and compute-class arguments used for planning. The merger rejects a missing index,
duplicate index, overlapping key, changed authority, changed compute class, and noncanonical row
order. It reopens only the selected partition, replays terminal evidence, and writes canonical
unsharded receipts, outcomes, terminal evidence, recomputed report inputs, and merge receipt v2.

On Goose, submit `scripts/goose_evaluation_shards.sbatch` as a Slurm array whose bounds exactly
match the sealed shard count. Export a real `ISINGFOLD_APPTAINER` path and SHA-256 plus a real
`ISINGFOLD_CONTAINER_IMAGE` path and SHA-256. The launcher mounts those pinned bytes read-only,
records the Slurm partition and actual node, and invokes each worker with `srun`. Set
`ISINGFOLD_THREADS` to the thread count sealed in the plan; the launcher rejects a `--threads`
override in forwarded workflow arguments and rejects any count above `SLURM_CPUS_PER_TASK`. Do not
invent a container identity. `bare-metal` requires `--diagnostic-only` and produces no publication-eligible
plan. External stock and validation-tuning plans use the same three commands with their respective
tuning registry, frozen selection, external config, candidate index, and tuning-seed index.

## 10. Final four-strength A/B diagnostic

This post-freeze diagnostic is separate from the primary learned-versus-stock inference. It cannot
change training, representation selection, RL-value selection, selector fitting, stock tuning, or
the fixed-sequence superiority decision. The registered config is
`configs/final_strength_audit_v1.json`. Its semantic record digest is
`fda50e04d2b193a26015f6f2c65c645e79bb9aaa05b346b9112505c06768c89e`; its file SHA-256 is
`58497bdb40b2b83fb90208678f0644779753771045c9195df13d7697507e0c7c`.

The production protocol fixes audit seed 1907, bootstrap seed 170141, at least 128 distinct base
lineages, a cap of 16 selected lineages in each signed stratum, 4,096 reads per block, 20,000
bootstrap replicates, two-sided alpha 0.05, and exactly 128 shards. Execute the stages in this
order:

1. Load the config with its caller-supplied file pin and register the sampler identity.
2. Build, write, externally pin, and reload the outcome-blind plan before opening test outcomes.
3. Authenticate the three learned and three tuned-stock source runs and their caller-supplied
   report pins before constructing the execution manifest.
4. Build, write, externally pin, and reload the execution manifest before any audit read.
5. Run each shard index from 0 through 127 over complete paired opportunity groups. For every valid
   return and each of four strengths, block A chooses the empirical oracle with the lowest-index
   tie rule and independent block B estimates oracle-minus-deployed regret. Invalid returns stay in
   the coverage denominator and receive the registered worst-case sensitivity range.
6. Publish every shard immutably, record each shard receipt SHA-256 outside the shard tree, reload
   all 128 shards through those pins, merge once, and publish the final audit atomically.

Apollo executes its assigned shards directly. Goose executes only through Slurm. Both hosts use the
same sealed plan and execution-manifest bytes. Shard assignment is deterministic; the merger rejects
missing indices, duplicates, overlap, partial paired groups, row drift, authority mismatch, source
drift, and noncanonical order. A monolithic production execution, an individual shard, and a partial
merge are not scientific results. Audit feedback is forbidden.

```bash
export FINAL_STRENGTH_CONFIG=configs/final_strength_audit_v1.json
export FINAL_STRENGTH_CONFIG_SHA256=\
'58497bdb40b2b83fb90208678f0644779753771045c9195df13d7697507e0c7c'

isingfold-rl plan-final-strength-audit \
  --grid configs/rl_grid_hybrid_v1.json \
  --selector runs/selector_v4 \
  --corpus runs/prepared_v4 \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  --audit-config "$FINAL_STRENGTH_CONFIG" \
  --expected-audit-config-sha256 "$FINAL_STRENGTH_CONFIG_SHA256" \
  --rl-value-selection-receipt "$ISINGFOLD_RL_FREEZE" \
  --expected-selection-sha256 "$ISINGFOLD_RL_FREEZE_SHA256" \
  --external-config configs/external_minorminer_complete_v1.json \
  --tuning-registry configs/external_minorminer_tuning_hybrid_v1.json \
  --expected-tuning-registry-sha256 "$EXPECTED_EXTERNAL_TUNING_REGISTRY_SHA256" \
  --external-tuning-selection "$EXTERNAL_TUNING_SELECTION" \
  --expected-external-tuning-selection-sha256 \
    "$EXPECTED_EXTERNAL_TUNING_SELECTION_SHA256" \
  --out runs/final_strength_plan_v2.json

export ISINGFOLD_FINAL_STRENGTH_PLAN=runs/final_strength_plan_v2.json
export EXPECTED_FINAL_STRENGTH_PLAN_SHA256='<recorded outside the audit tree>'

FINAL_SOURCE_ARGS=(
  --learned-evaluation runs/complete_eval_v1/seed-0
  --expected-learned-report-sha256 "$LEARNED_REPORT_SHA_0"
  --learned-evaluation runs/complete_eval_v1/seed-1
  --expected-learned-report-sha256 "$LEARNED_REPORT_SHA_1"
  --learned-evaluation runs/complete_eval_v1/seed-2
  --expected-learned-report-sha256 "$LEARNED_REPORT_SHA_2"
  --external-evaluation runs/external_complete_v4/seed-0
  --expected-external-report-sha256 "$STOCK_REPORT_SHA_0"
  --external-evaluation runs/external_complete_v4/seed-1
  --expected-external-report-sha256 "$STOCK_REPORT_SHA_1"
  --external-evaluation runs/external_complete_v4/seed-2
  --expected-external-report-sha256 "$STOCK_REPORT_SHA_2"
)

isingfold-rl seal-final-strength-audit-execution \
  --plan "$ISINGFOLD_FINAL_STRENGTH_PLAN" \
  --expected-plan-sha256 "$EXPECTED_FINAL_STRENGTH_PLAN_SHA256" \
  --corpus runs/prepared_v4 \
  "${QUALITY_ARGS[@]}" \
  "${GROUND_CERTIFICATE_ARGS[@]}" \
  "${FINAL_SOURCE_ARGS[@]}" \
  --out runs/final_strength_execution_v2.json

export ISINGFOLD_FINAL_STRENGTH_EXECUTION=runs/final_strength_execution_v2.json
export EXPECTED_FINAL_STRENGTH_EXECUTION_SHA256='<recorded before audit reads>'
export ISINGFOLD_FINAL_STRENGTH_SHARD_ROOT=runs/final_strength_shards_v1
export ISINGFOLD_CORPUS=runs/prepared_v4
export LEARNED_COMPLETE_0=runs/complete_eval_v1/seed-0
export LEARNED_COMPLETE_1=runs/complete_eval_v1/seed-1
export LEARNED_COMPLETE_2=runs/complete_eval_v1/seed-2
export EXPECTED_LEARNED_REPORT_SHA256_0="$LEARNED_REPORT_SHA_0"
export EXPECTED_LEARNED_REPORT_SHA256_1="$LEARNED_REPORT_SHA_1"
export EXPECTED_LEARNED_REPORT_SHA256_2="$LEARNED_REPORT_SHA_2"
export STOCK_COMPLETE_0=runs/external_complete_v4/seed-0
export STOCK_COMPLETE_1=runs/external_complete_v4/seed-1
export STOCK_COMPLETE_2=runs/external_complete_v4/seed-2
export EXPECTED_STOCK_REPORT_SHA256_0="$STOCK_REPORT_SHA_0"
export EXPECTED_STOCK_REPORT_SHA256_1="$STOCK_REPORT_SHA_1"
export EXPECTED_STOCK_REPORT_SHA256_2="$STOCK_REPORT_SHA_2"

# Apollo runs its assigned range directly.
export ISINGFOLD_SHARD_FIRST=0 ISINGFOLD_SHARD_LAST=63 ISINGFOLD_SHARD_STEP=1
scripts/apollo_final_strength_audit.sh

# Goose handles the complementary range inside Slurm.
sbatch --array=64-127 scripts/goose_final_strength_audit.sbatch
```

Both launchers require the corpus, publisher/root pins, six source directories, and six report pins
as exported environment variables, using the names checked at the top of each launcher. To merge,
repeat aligned `--shard` and `--expected-shard-receipt-sha256` arguments for all 128 directories:

```bash
isingfold-rl merge-final-strength-audit \
  --plan "$ISINGFOLD_FINAL_STRENGTH_PLAN" \
  --expected-plan-sha256 "$EXPECTED_FINAL_STRENGTH_PLAN_SHA256" \
  --execution-manifest "$ISINGFOLD_FINAL_STRENGTH_EXECUTION" \
  --expected-execution-manifest-sha256 \
    "$EXPECTED_FINAL_STRENGTH_EXECUTION_SHA256" \
  "${FINAL_STRENGTH_SHARD_ARGS[@]}" \
  --out runs/final_strength_audit_v2
```

## 11. Reproducibility and release checks

Before accepting an artifact for a paper table, verify all of the following:

- The repository source digest, dependency registry, corpus manifest, quality authority, selector,
  grid, and selection receipts match their externally recorded hashes.
- Every expected cell exists exactly once; partial directories and informal retries are excluded.
- Representation and RL-value selection use all three seeds and crossed seed by lineage bootstrap.
- Complete-system confirmation uses fresh no-resume retraining for all three seeds and the
  pre-initialization test denominator.
- Every PPO confirmation seed authenticates a complete target-free initializer bank whose episode
  schedule equals `updates * episodes`; rejected draws and work remain visible.
- Every sharded evaluation plan is target-free, every worker opens one named partition, and the
  merge contains every planned lineage key exactly once.
- Ordinary initialization, search, validation, compilation, and sampling failures remain visible
  with utility zero. Integrity failures abort the artifact instead of becoming method failures.
- Primary tables report unconditional IF-Q3-S0 utility and valid-return rate. Secondary tables
  report qubit use, chain statistics, connectivity/cut descriptors, work, latency, and memory.
- Selector audit, empirical four-count oracle, greedy deployment, and capacity control are clearly
  labeled diagnostics and cannot revise the frozen main choice.

Deterministic algorithms and one CPU thread are the strict replay defaults. Cross-host GPU agreement
is diagnostic rather than bitwise evidence. Use CPU replay to check cross-site likelihood parity.
Training cost, online embedding work, and evaluator reads are separate resource dimensions.
