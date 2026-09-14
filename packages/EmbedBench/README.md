# EmbedBench

Certified structural decision data, surrogate-scored quality decision data, and downstream-objective evaluation for minor embedders.

Most embedding heuristics are judged by qubit counts. EmbedBench judges decisions and
embeddings by the objective that matters on an annealer: a feasible embedding first, then
the probability that the programmed problem decodes to the ground state. It ships the
generators, the certificates, the surrogate and the evaluation protocol from the IsingFold
study (record book entries L-67 to L-101), with no dependency on the rest of that repository. Two kinds of label: the structural track's labels are exact and re-certified (0 mismatches in 6046 brute-force checks on release v1); the quality track's labels are surrogate estimates with a noise floor of about 0.035 at 400 reads, not certificates.

## Tracks

**A. Structural decisions (exact labels).** A witness embedding is planted in a real hardware
graph (Chimera, Pegasus, Zephyr; optional defects). A decision state freezes all but `k`
variables, keeps a prefix of the focus chain, cuts a window, and labels every legal
single-qubit extension by exact enumeration of its completions under (feasible, fewest
qubits, shortest longest chain), with every logical edge to the frozen context enforced.
Records are re-certifiable from their own fields (`embedbench certify`). Metric: top-1 on
certified-best actions; reference rules: local greedy, random.

**B. Quality decisions at the chain seam (surrogate labels).** From a valid embedding
(witness, stock minorminer, or an application instance with real coefficients) remove one
chain, enumerate every valid replacement in a window, and score each by the surrogate solve
probability with a two-stage fresh-seed protocol (screening reads for all, refinement reads
for the screening top, the fewest-qubit replacement and the original chain; comparisons use
stage-2 estimates only). Metric: regret against the best of the reliable set; references:
fewest-qubit chain, original chain, random.

**C. End-to-end embedders.** Any callable `embed(logical, host, seed) -> {node: qubits}` is
run on planted-Ising random graphs, planted quotient graphs and application instances
(portfolio, graph-cut) and compared with stock minorminer by feasibility, qubits, longest
chain and surrogate solve probability (max over a chain-strength grid; fixed-schedule
simulated annealing, 200 sweeps, majority-vote decoding; 800 reads, fresh seed).

## Quick start

```bash
python3 -m pip install -e .            # models extra: pip install -e ".[models]"
embedbench motifs                       # or: python3 -m embedbench.cli motifs
                                         # seven hand-test motifs, each certified exactly
embedbench generate-structural --topology chimera --size 6 --n-instances 20 --jobs 8 --out runs/structural_c6.jsonl
embedbench certify runs/structural_c6.jsonl --limit 50
embedbench generate-chain --topology pegasus --size 3 --graph random --source minorminer --n-instances 20 --jobs 8 --out runs/chain_p3.jsonl
embedbench evaluate --topology chimera --size 5 --source app --n 10 --embedder mypkg.embed:my_embedder
```

Python:

```python
from embedbench import evaluate_embedder, EvalConfig, stock_minorminer
print(evaluate_embedder(my_embedder, EvalConfig(topology="pegasus", size=3, source="random", n=30)))
```

## Training on a release

Paper-facing training uses the release manifest rather than a seed-dependent random split:

```bash
python3 scripts/train_structural.py runs/release_v1_1/structural_*.jsonl \
  --splits runs/release_v1_1/splits.json --arch gatv2 --seeds 0,1,2,3 \
  --out runs/arch/structural_gatv2.json

# after generating the payload-grouped quality split manifest described below
python3 scripts/train_chain.py runs/release_v1_1/quality_*.jsonl \
  --splits runs/release_v1_1/splits_problem_v2.json \
  --arch gps --loss pairwise --seeds 0,1,2,3 \
  --out runs/arch_chain/quality_gps.json
```

With `--splits`, the training seed changes optimisation only; train, validation and test
membership stays fixed. The loader rejects missing IDs, unknown split names and any quality
problem whose canonical full Ising payload occurs across partitions. This covers
witness/minorminer state pairs and the same application or random problem embedded into
different host topologies. Omitting `--splits` retains the legacy random split for
exploratory runs.

Fixed-split architecture searches report validation metrics and keep test metrics unset.
Add `--evaluate-test` only for the final architecture and hyperparameters selected on
validation data.

### Quality Value V2

The V1 stage-2 shortlist metric is retained only as the explicit
`legacy-reliable` diagnostic. Quality Value V2 scores the complete presented candidate
support, marks the mixed-fidelity release metric as provisional, and applies exact
minor-embedding validity and full-embedding qubit-budget masks before selection. The
release field `Q` is the focus-chain length; V2 derives total embedding qubits from
`all_chains` and verifies this relationship for every record.

The registered terminal target is
`V(e) = max_{f in F_registered} p_solve(e; f)`, where
`F_registered = default_strength_grid(problem, 4)`. It is the best solve probability over
exactly four registered chain strengths, not a strength-conditioned probability. Release
labels and independent high-read audit labels use this same grid. The neural selector uses
only its predicted quality mean, or the separately reported uncertainty proxy. Exact
validity and `B/Q_MM` are hard masks. The residual-connectivity and contact-robustness heads
are auxiliary representation losses. Total qubits are measured exactly, so the protocol
does not assume that either shorter or longer embeddings are always better.

Every registered V2 backbone must receive the full deployment-available logical
Hamiltonian. The label-free global context contract in
`src/embedbench/hamiltonian_context.py` reads only `problem.h` and `problem.J`; it ignores
`e0`, labels, candidate winners, and baseline indices. Checkpoints and runtime adapters must
bind its schema version, ordered feature names, and dimension. The heterogeneous backbone
additionally represents every logical variable and nonzero logical coupling.

```bash
python3 scripts/train_quality_v2.py runs/release_v1_1/quality_*.jsonl \
  --splits data/release_v1/splits_quality_problem_v2.json \
  --arch gatv2 --objective-variant full --seeds 0,1,2,3 \
  --evaluation-support full --deploy-view \
  --out runs/quality_v2/gatv2_full.json
```

The development trainer evaluates the registered budget sweep
`{1.00, 1.10, 1.25, 1.50, uncapped}` relative to the exact total-qubit count of a valid
stock minorminer embedding. Rows without that paired `Q_MM` reference are excluded with
explicit coverage reasons; they are never silently rebased. A separately named
`B/Q_resource` sweep is diagnostic only. Checkpoints are selected by mean validation regret
across the finite `B/Q_MM` budgets, and the trainer never encodes or evaluates the fixed
test partition.

The registered screen contains five backbones (MPNN, GIN, GATv2, GPS, and a heterogeneous
logical/physical graph model), four objective variants (`p_only`, `p_connectivity`,
`p_robustness`, and `full`), and four seeds, for 80 development cells. `--deploy-view`
reconstructs only train and validation windows and
requires defect-free generator manifests whose corpus hashes match. The heterogeneous V2
forward boundary contains no labels or policy/oracle indices. A learned feasibility head is
not reported for this terminal corpus because every released candidate is exact-valid; the
native feasibility and budget masks remain authoritative.

Paper and audit evaluation require complete independent high-read labels bound to corpus
SHA-256, ordered candidate signature, instance ID, and focus, plus the read count, sweep
count, base seed, seed schedule, registered strength count (`4`), strength schedule, and
aggregation rule. Every per-strength probability must also carry its integer success count;
the evaluator derives the probability as `success_count / reads_per_strength` and rejects
fractional score-only evidence. Legacy high-read files without this evidence remain
unverified development evidence and must be regenerated before a paper-facing evaluation.
The machine-readable contract in `configs/quality_v2_paper_audit_v2.json` freezes those
label settings before the locked test is opened. It also freezes full record and candidate
support, top tolerance `0.02`, random-baseline seed `0`, mean as the primary learned
selection statistic, LCB coefficient `z=1.0`, the five `B/Q_MM` budget points, and arithmetic
aggregation over exactly four registered model seeds with sample standard deviation. Random
baselines are keyed by a canonical record digest rather than input position, so reordering
the corpus cannot change them. The adjacent contract checksum is a local integrity check.
The later externally registered preregistration manifest binds the contract bytes into the
paper trust root. The selector, rescoring job, evaluator, and aggregator reject contract drift.
The retained v1 contract is superseded: its exact CUDA-to-CPU sweep comparison failed before
any production selection artifact or test-label access. Paper runs must use v2.

The same contract predeclares report-only secondary endpoints that cannot select an
architecture or candidate. They include exact candidate feasibility and per-budget survival,
terminal total-qubit and maximum-chain deltas against the stock minorminer embedding, and
physical residual-connectivity deltas after removing each complete selected embedding from
the authenticated realized host. Residual connectivity reports remaining free qubits, the
largest remaining connected component, its fraction of all remaining qubits, and the number
of remaining components, both absolutely and relative to stock minorminer. These are
recomputed structural quantities, not learned proxy
labels. Audit rows retain the four per-strength `p_solve` outcomes and exact success counts
for every candidate. After the policy freeze, the evaluator reports their worst case and max-minus-min
spread on the complete feasible test support and, separately, on every registered `B/Q_MM`
budget cohort. Both views report paired deltas against the stock minorminer candidate and
retain the underlying per-record values. Positive worst-case `p_solve` delta is better;
negative spread delta is more robust. Both views expose selection exclusions, missing
outcomes, and coverage. Missing per-strength probability or integer-count evidence is
rejected. Missing authenticated host evidence reports `unavailable` or `partial` with
explicit coverage; malformed or internally inconsistent evidence is rejected.

Problem-clustered inference for mean `p_solve` deltas uses a distribution-free Hoeffding
bound on the registered `[-1,1]` support. It does not use a sign-flip test, which would
require an unregistered symmetry assumption. Confidence intervals and p-values are emitted
only with at least 20 distinct problem clusters and are conditional on the registered
independent-problem-cluster sampling assumption. Secondary endpoint bootstrap intervals are
pointwise descriptive, conditional on the four registered model checkpoints, and use the
same 20-cluster minimum.
The cryptographic bindings prove byte identity and execution ordering; they do not prove that
problem clusters are statistically independent. Independence must be justified by the corpus
construction and sampling design, so all stated intervals and tests remain conditional on that
design assumption.

The current legacy-IID test is explicitly `exploratory_legacy_iid` and every result from it
is provisional. Test membership, problem sizes, and `e0` were inspected during development,
although independent audit `p_solve` labels were not. It therefore cannot support a
confirmatory claim. Only a later custodian-held hard/OOD corpus that remains unopened until
the full protocol is frozen is designated confirmatory.

Run one registered cell through `scripts/run_training_grid.py`, or use the Apollo and Goose
wrappers. The registered axis order makes even and odd cell indices a crossed two-site
design: every architecture and objective variant is split equally between sites, and every
architecture-objective configuration has exactly two seeds on Apollo and two on Goose.
With the registered order, seeds 0 and 2 occupy even indices and seeds 1 and 3 occupy odd
indices. Apollo has one observed GPU and no Slurm: run one direct process sequentially over
the even indices. Goose jobs must go through Slurm; its mandatory two-task array divides the
odd sequence. After staging the same source and data tree on each host:

```bash
# Apollo only: one direct process on its single observed GPU; do not use Slurm.
export ISINGFOLD_WORK_ROOT="$PWD"
ISINGFOLD_GRID_CONFIG=configs/training_grid_quality_v2.json \
ISINGFOLD_GRID_STEP=2 CUDA_VISIBLE_DEVICES=0 \
  scripts/apollo_training_grid.sh quality_value_v2_screen 0 78

# Goose only: submit from the staged root; do not train on the login node.
export ISINGFOLD_WORK_ROOT="$PWD"
sbatch --array=0-1 \
  --export=ALL,ISINGFOLD_GRID_CONFIG=configs/training_grid_quality_v2.json,ISINGFOLD_GRID_STAGE=quality_value_v2_screen,ISINGFOLD_GRID_SIZE=80,ISINGFOLD_GRID_FIRST=1,ISINGFOLD_GRID_LAST=79,ISINGFOLD_GRID_STEP=2 \
  scripts/goose_training_grid.sbatch
```

Paper evaluation uses an exact frozen training root. Its `_source_sha256` must remain
`294caa56d0bb04f716163b143a995ee176c8e72dd209c5f5f77caef92d72a264`, the digest already
bound into every registered receipt and the paper contract. Into that root, the only runtime
files that may be synchronized are:

- `scripts/select_training_grid.py`
- `scripts/evaluate_quality_v2.py`
- `scripts/rescore_quality.py`
- `scripts/aggregate_quality_v2_paper.py`
- `scripts/quality_v2_paper_contract.py`
- `scripts/paper_verifier/__init__.py`
- `scripts/paper_verifier/schema.py`
- `scripts/paper_verifier/ground_certificate.py`
- `configs/quality_v2_paper_audit_v2.json` and its `.sha256` sidecar

Do not synchronize `src/`, `scripts/run_training_grid.py`, any `scripts/train_*.py` or
`scripts/training_*.py`, the training grid, corpus, splits, checkpoints, results, or receipts.
Those paths are part of the preregistered training-source or data identity. The synchronized
entrypoints and `paper_verifier` package form the mutable edge of a separate verifier layer.
The package contains the exact certificate and canonical-JSON checker needed by the paper
evaluator, outside the legacy training-source glob. `audit_source_sha256` starts from the five
production entrypoints and captures each local Python source file once through directory file
descriptors with `O_NOFOLLOW`. It parses and hashes the same retained bytes, recursively binds
every reachable module already present under `scripts/` and `src/`, includes package
initializers, and rejects symlinks in the source-root path or source tree, as well as an
inventory change during verification. Registered local namespace roots remain protected even
when their files are absent. Unresolved or ambiguous local imports, direct dynamic code-loading
primitives, and aliases or escaped references to those primitives fail closed.
The current development checkout is not the registered training root and must not be treated
as one merely because it contains newer verifier code. Stage the historical root first,
confirm its registered `_source_sha256`, and then synchronize only the entrypoint files above.
The closure digest authenticates their actual transitive dependencies from that staged tree;
it does not authorize replacing those dependencies. In particular, retained
`scripts/run_training_grid.py` and `src/embedbench/models_chain.py` are closure inputs but must
not be synchronized. This preserves the historical training digest while giving the verifier
closure its own separately checked digest.

The historical package initializer conservatively brings the dormant
`embedbench.objective_embedder` source into the static closure through a function-local import
in `embedbench.evaluate`. The paper entrypoints neither instantiate that class nor load its
packaged `models/*.joblib` resources. The closure verifier rejects normal transitive import or
call edges from the paper path into `ObjectiveEmbedder`; those unused joblib files are therefore
outside this paper audit binding. A future paper protocol that invokes that embedder must add
the model resources to its content manifest and define a new source algorithm version.

This source check assumes a stable staged filesystem and a trusted Python interpreter. Run the
five production entrypoints as fresh CLI processes from the authenticated staged root. Importing
them into a test harness that monkeypatches modules, replacing code after preflight, or executing
under a hostile interpreter is outside the claimed trust boundary. The protocol has no
test-only or caller-supplied prevalidated-selection path.

After every receipt is present, validate the contract and run the source preflight. It
recomputes the registered grid and training-source digests without opening any cell artifact
or writing a selection:

```bash
ROOT=/home/nguyencongt/isingfold/EmbedBench_qv2_global_20260910
PYTHON_BIN=/home/nguyencongt/isingfold/.venv/bin/python
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
export PYTHONHASHSEED=0
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
unset PYTHONSTARTUP PYTHONINSPECT

AUDIT_CONTRACT=configs/quality_v2_paper_audit_v2.json
AUDIT_CONTRACT_SHA256="$(awk '{print $1}' \
  configs/quality_v2_paper_audit_v2.sha256)"
(cd configs && shasum -a 256 -c quality_v2_paper_audit_v2.sha256)

"$PYTHON_BIN" scripts/select_training_grid.py \
  --grid configs/training_grid_quality_v2.json \
  --stage quality_value_v2_screen --root . \
  --audit-contract "$AUDIT_CONTRACT" \
  --audit-contract-sha256 "$AUDIT_CONTRACT_SHA256" \
  --preflight-only

"$PYTHON_BIN" scripts/select_training_grid.py \
  --grid configs/training_grid_quality_v2.json \
  --stage quality_value_v2_screen --root . \
  --audit-contract "$AUDIT_CONTRACT" \
  --audit-contract-sha256 "$AUDIT_CONTRACT_SHA256" \
  --out runs/training_grid_quality_v2/selection.json

SELECTION=runs/training_grid_quality_v2/selection.json
SELECTION_SHA256="$("$PYTHON_BIN" -c \
  'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1], "rb").read()).hexdigest())' \
  "$SELECTION")"
printf '%s  %s\n' "$SELECTION_SHA256" "$SELECTION" \
  > runs/training_grid_quality_v2/selection.sha256
```

The schema-v3 selector verifies all 80 registered results, loadable checkpoints, receipts,
normalized semantic training command, parsed trainer arguments, corpus hash, generator
manifest, split hash, source hash, core runtime, execution site, full-split marker, and common
`B/Q_MM` validation population. Absolute staged roots and Python paths may differ between Apollo
and Goose, but
all model, optimizer, loss, preprocessing, and evaluation arguments are explicit in the grid
and must agree exactly. The stored CUDA validation sweep remains required provenance for the
registered best-epoch checkpoint, and its primary metric must match checkpoint metadata. It is
not compared bitwise with another device and cannot rank configurations or deployment seeds.
Instead, the selector runs every frozen checkpoint through the same deterministic, float32,
single-thread CPU replay. This fresh replay is the sole source of cross-cell validation regret.
Model-dependent GPU-to-CPU differences are recorded as non-gating diagnostics and never cause
rejection, exclusion, ranking, or tie-breaking. Model-independent cohort, reference-index,
exact-cost, and budget fields must match exactly; any mismatch fails closed.
In a cross-host preflight on one frozen H100-trained checkpoint, Apollo and Goose produced
byte-identical sweep and replay-evidence SHA under the registered runtime. The all-checkpoint
claim remains withheld until all 80 replays are certified. No post-hoc numeric tolerance
participates in acceptance or ranking.
The selector chooses the lowest mean canonical replay regret across the registered seeds. The
lowest canonical replay regret within that configuration determines the deployment checkpoint;
all four seed checkpoints are frozen and independently evaluated for the paper result. The
paper reports their aggregate rather than the best seed alone. Ties use registered order.
The selector strictly parses corpus rows for manifest verification and partition routing, but
only encodes the validation partition. It independently replays every one of the 80 frozen
validation checkpoints on an allowlisted, label-free model view and binds each replay digest,
candidate signature, exact mask, total-qubit vector, `Q_MM`, budget, eligible set, and choice.
It never encodes or evaluates test records and never consumes test labels for selection. A
missing, changed, or
incomplete receipt, checkpoint, or replay result invalidates the selection instead of
falling back to the stored winner.

The paper evaluator is a strict two-phase protocol. Phase 1 completes a label-free replay over
the fixed test records, computes exact feasibility and qubit masks, model predictions, every
learned and baseline choice, and every budget-specific choice, then atomically writes a
canonical policy freeze and SHA-256 sidecar. It refuses audit paths. The sidecar detects local
copy errors but is not an independent registration. After all four freezes exist, one canonical
preregistration manifest binds the audit contract, verifier-source digest, selection artifact,
four checkpoint digests, and four freeze digests. Phase 2 requires the manifest digest supplied
from an out-of-band record, regenerates the relevant phase-1 bytes, and verifies byte equality
before it validates or opens any audit-label shard. The random baseline and all finite-budget
choices are therefore fixed without consulting the audit outcomes.
The generated evaluation artifact states only facts the process can establish mechanically:
the caller-supplied manifest digest matched, the selected freeze was listed in that manifest,
its bytes matched, replay was byte exact, and audit inputs were opened afterward. Whether the
caller obtained that digest from the required out-of-band log remains a custody fact recorded
outside the program, not an unconditional boolean asserted by the evaluator.

Run phase 1 for all four registered seeds before generating or exposing audit labels:

```bash
SELECTION=runs/training_grid_quality_v2/selection.json
SELECTION_SHA256="$(awk '{print $1}' runs/training_grid_quality_v2/selection.sha256)"

"$PYTHON_BIN" - "$SELECTION" "$SELECTION_SHA256" \
  "$AUDIT_CONTRACT" "$AUDIT_CONTRACT_SHA256" <<'PY'
import glob
import json
import subprocess
import sys

selection_path, selection_sha256, audit_contract, audit_contract_sha256 = sys.argv[1:]
selection = json.load(open(selection_path, encoding="utf-8"))
corpora = sorted(glob.glob("runs/release_v1_1/quality_*.jsonl"))
for entry in selection["winner"]["paper_checkpoints"]:
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_quality_v2.py",
            "--phase", "freeze",
            *corpora,
            "--splits", "data/release_v1/splits_quality_problem_v2.json",
            "--selection", selection_path,
            "--selection-sha256", selection_sha256,
            "--selection-root", ".",
            "--checkpoint", entry["checkpoint"],
            "--checkpoint-sha256", entry["checkpoint_sha256"],
            "--audit-contract", audit_contract,
            "--audit-contract-sha256", audit_contract_sha256,
            "--out",
            f"runs/training_grid_quality_v2/policy_freeze_seed_{entry['seed']}.json",
        ],
        check=True,
    )
PY

POLICY_FREEZES=(runs/training_grid_quality_v2/policy_freeze_seed_{0,1,2,3}.json)
PREREGISTRATION_MANIFEST=runs/training_grid_quality_v2/paper_preregistration.json
"$PYTHON_BIN" scripts/quality_v2_paper_contract.py \
  --audit-contract "$AUDIT_CONTRACT" \
  --audit-contract-sha256 "$AUDIT_CONTRACT_SHA256" \
  --selection "$SELECTION" --selection-sha256 "$SELECTION_SHA256" \
  --selection-root . --audit-source-root . \
  --policy-freezes "${POLICY_FREEZES[@]}" \
  --out "$PREREGISTRATION_MANIFEST"

# Copy the printed SHA-256 to an independent append-only evidence log before
# starting any audit-label job. Set this only from that external record.
PREREGISTRATION_MANIFEST_SHA256='<externally-registered-sha256>'
```

Only after externally registering that single manifest digest, generate the complete schema-v2
audit labels for the locked test partition. `--per-file` must cover every test record, not a
sample. Every rescorer process authenticates the manifest and all four freeze files before it
opens a corpus record for scoring:

```bash
# Direct-loop example for Apollo. On Goose, submit the same body as a Slurm 0-63
# array and set SHARD_INDEX="$SLURM_ARRAY_TASK_ID" inside the batch job.
for SHARD_INDEX in $(seq 0 63); do
  "$PYTHON_BIN" scripts/rescore_quality.py runs/release_v1_1/quality_*.jsonl \
    --splits data/release_v1/splits_quality_problem_v2.json \
    --split test --per-file 100000 --reads 4000 --seed 0 \
    --selection "$SELECTION" \
    --selection-sha256 "$SELECTION_SHA256" \
    --selection-root . \
    --audit-contract "$AUDIT_CONTRACT" \
    --audit-contract-sha256 "$AUDIT_CONTRACT_SHA256" \
    --preregistration-manifest "$PREREGISTRATION_MANIFEST" \
    --preregistration-manifest-sha256 "$PREREGISTRATION_MANIFEST_SHA256" \
    --policy-freezes "${POLICY_FREEZES[@]}" \
    --shard-count 64 --shard-index "$SHARD_INDEX" \
    --out "runs/release_v1_2/hiread_test_labels_v2_shard_${SHARD_INDEX}.jsonl"
done
```

After all 64 shards and their manifests exist, a custodian builds one release manifest that
commits to the exact bytes of every shard and every shard manifest. The printed SHA-256 must
be copied to an external immutable evidence log before anyone runs phase 2. A digest stored
only beside mutable audit files is not an independent commitment.

```bash
AUDIT_RELEASE_MANIFEST=runs/release_v1_2/hiread_test_labels_v2_release.json
"$PYTHON_BIN" - "$SELECTION" "$SELECTION_SHA256" \
  "$AUDIT_CONTRACT" "$AUDIT_CONTRACT_SHA256" \
  "$PREREGISTRATION_MANIFEST" "$PREREGISTRATION_MANIFEST_SHA256" \
  "$AUDIT_RELEASE_MANIFEST" <<'PY'
import glob
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from quality_v2_paper_contract import AUDIT_SOURCE_HASH_ALGORITHM, load_paper_audit_contract, load_paper_preregistration, selection_audit_binding
from rescore_quality import build_audit_release_commitment

(selection_path, selection_sha256, contract_path, contract_sha256,
 preregistration_path, preregistration_sha256, output) = sys.argv[1:]
selection = json.load(open(selection_path, encoding="utf-8"))
contract = load_paper_audit_contract(contract_path, contract_sha256)
preregistration = load_paper_preregistration(
    preregistration_path,
    preregistration_sha256,
    audit_contract=contract,
    selection_document=selection,
    selection_file=Path(selection_path).name,
    selection_sha256=selection_sha256,
    audit_source_root=".",
)
shards = sorted(glob.glob("runs/release_v1_2/hiread_test_labels_v2_shard_*.jsonl"))
_, payload = build_audit_release_commitment(
    shards,
    audit_contract_binding=contract.public_binding(),
    preregistration_binding=preregistration.public_binding(),
    selection_binding=selection_audit_binding(
        selection,
        selection_file=Path(selection_path).name,
        selection_sha256=selection_sha256,
    ),
    audit_source_binding={
        "algorithm": AUDIT_SOURCE_HASH_ALGORITHM,
        "sha256": selection["audit_source_sha256"],
    },
    expected_shard_count=contract.document["audit_execution"]["shard_count"],
)
Path(output).write_bytes(payload)
print(hashlib.sha256(payload).hexdigest())
PY

# Set this from the digest retained in the external immutable evidence log.
AUDIT_RELEASE_MANIFEST_SHA256='<externally-registered-sha256>'
```

Phase 2 independently regenerates and byte-compares each policy before it opens the 64 audit
shards:

```bash
"$PYTHON_BIN" - "$SELECTION" "$SELECTION_SHA256" \
  "$AUDIT_CONTRACT" "$AUDIT_CONTRACT_SHA256" \
  "$PREREGISTRATION_MANIFEST" "$PREREGISTRATION_MANIFEST_SHA256" \
  "$AUDIT_RELEASE_MANIFEST" "$AUDIT_RELEASE_MANIFEST_SHA256" <<'PY'
import glob
import json
import subprocess
import sys

(
    selection_path,
    selection_sha256,
    audit_contract,
    audit_contract_sha256,
    preregistration_manifest,
    preregistration_manifest_sha256,
    audit_release_manifest,
    audit_release_manifest_sha256,
) = sys.argv[1:]
selection = json.load(open(selection_path, encoding="utf-8"))
corpora = sorted(glob.glob("runs/release_v1_1/quality_*.jsonl"))
audit_shards = sorted(glob.glob("runs/release_v1_2/hiread_test_labels_v2_shard_*.jsonl"))
for entry in selection["winner"]["paper_checkpoints"]:
    freeze = f"runs/training_grid_quality_v2/policy_freeze_seed_{entry['seed']}.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_quality_v2.py",
            "--phase", "score",
            *corpora,
            "--splits", "data/release_v1/splits_quality_problem_v2.json",
            "--selection", selection_path,
            "--selection-sha256", selection_sha256,
            "--selection-root", ".",
            "--checkpoint", entry["checkpoint"],
            "--checkpoint-sha256", entry["checkpoint_sha256"],
            "--audit-labels", *audit_shards,
            "--audit-release-manifest", audit_release_manifest,
            "--audit-release-manifest-sha256", audit_release_manifest_sha256,
            "--policy-freeze", freeze,
            "--preregistration-manifest", preregistration_manifest,
            "--preregistration-manifest-sha256", preregistration_manifest_sha256,
            "--audit-contract", audit_contract,
            "--audit-contract-sha256", audit_contract_sha256,
            "--out",
            f"runs/training_grid_quality_v2/paper_test_seed_{entry['seed']}.json",
        ],
        check=True,
    )
PY

"$PYTHON_BIN" scripts/aggregate_quality_v2_paper.py \
  --selection "$SELECTION" --selection-sha256 "$SELECTION_SHA256" \
  --selection-root . \
  --files runs/release_v1_1/quality_*.jsonl \
  --splits data/release_v1/splits_quality_problem_v2.json \
  --audit-labels runs/release_v1_2/hiread_test_labels_v2_shard_*.jsonl \
  --audit-release-manifest "$AUDIT_RELEASE_MANIFEST" \
  --audit-release-manifest-sha256 "$AUDIT_RELEASE_MANIFEST_SHA256" \
  --preregistration-manifest "$PREREGISTRATION_MANIFEST" \
  --preregistration-manifest-sha256 "$PREREGISTRATION_MANIFEST_SHA256" \
  --policy-freezes "${POLICY_FREEZES[@]}" \
  --audit-contract "$AUDIT_CONTRACT" \
  --audit-contract-sha256 "$AUDIT_CONTRACT_SHA256" \
  --evaluations runs/training_grid_quality_v2/paper_test_seed_*.json \
  --out runs/training_grid_quality_v2/paper_test_aggregate.json
```

The 64 shards are assigned by the SHA-256 of canonical `[file, instance_id, focus]` bytes.
Each writes one atomic, content-addressed record per identity, a canonically ordered JSONL,
and a checksum manifest. Interrupted paper shards recompute every assigned row and never
trust a partial score cache. The evaluator first validates all 64 manifests and their
exactly-once coverage of the fixed test set, then verifies every decoded JSONL row against its
manifest identity, shard assignment, and externally committed JSONL bytes before parsing
labels. The per-record files are resumability artifacts used while generating a shard. They
are not authoritative phase-2 inputs, and the evaluator neither opens nor trusts them; their
manifest paths and hashes document how the canonical JSONL was assembled. It independently
recomputes certified ground energy with bounded, batched all-spin
enumeration for registered small problems or the registered ferromagnetic s-t min-cut case.
Hard/OOD rows may instead carry a closed inline envelope containing the WP6 certificate,
proof, and independent SHA-256 bindings for both artifacts. The evaluator recomputes both
digests and replays the proof against the authenticated Ising payload. Only `planted_proof`
and `exact_enumeration` are paper eligible in the first release. `certified_optimal`, solver
receipts, aliases, external artifact paths, and unknown envelope fields are rejected until an
executable independently registered branch-and-bound proof checker exists.
For v1.1/v1.2 compatibility, the previous independently replayed exhaustive-at-most-22 and
ferromagnetic min-cut formats remain explicitly version-gated. Evaluation artifacts report
their counts separately from WP6 planted and WP6 exact-enumeration certificates, so evidence
classes cannot be silently pooled.

Each audit row binds the contract, externally authenticated preregistration, receipt-revalidated
selection, paper-audit source digest, and complete CPU runtime provenance. Two source digests
have distinct roles. The historical training `_source_sha256` binds the package and registered
training pipeline used by the 80 receipts. The later `audit_source_sha256` binds the sorted,
length-prefixed bytes and relative paths of the complete local transitive import closure rooted
at `PAPER_AUDIT_ENTRYPOINTS`. It includes the five entrypoints plus every reachable module in
`scripts/` and `src/`; AST resolution and hashing consume one retained byte snapshot, and a
second metadata-only inventory scan detects concurrent source drift. It does not redefine or
replace the historical training root. Runtime
provenance binds Python, platform, device, and the exact versions of the eight registered
distributions. Inputs and checkpoints are evaluated from the same captured bytes that were
hashed. The evaluator rejects a path, digest, seed, architecture, objective variant, data
provenance, audit-label provenance, runtime, or evaluation setting that is not registered.
Tolerance, device, LCB coefficient, and evaluation randomness have no command-line override.

The aggregate command does not trust supplied summaries. It first revalidates and replays all
80 validation cells. Every one of its four phase-1 and four phase-2 evaluator invocations also
performs the same complete selection revalidation; no reusable Python capability can skip it.
It then replays phase 1 for all four winner checkpoints and verifies every
registered freeze byte before opening any audit path. It next consumes all 64 shards and
recomputes all four phase-2 results. A supplied evaluation is accepted only when it is exactly
equal to that recomputation. It also requires the same record cohort, coverage masks,
budgets, QMM values, model-independent baselines, audit runtime, and inference runtime. Its
paper artifact reports the across-seed mean and sample standard deviation. As secondary
exploratory inference, it reports problem-equal-weighted learned-minus-stock `p_solve`, a
distribution-free two-sided Hoeffding 95% confidence interval on the registered `[-1,1]`
support, and a one-sided Hoeffding bounded-mean p-value. Inference is emitted only with at
least 20 distinct problem clusters. The four finite budgets form one frozen Holm-Bonferroni
family. These uncertainty summaries remain provisional on the legacy-IID cohort.
The aggregate also reports across-seed summaries for every available predeclared secondary
endpoint and preserves per-seed availability and coverage. For residual-connectivity and
strength-robustness deltas, it first averages the four registered model seeds within each
record, then averages records with the same problem digest. It reports those paired
per-problem values with a deterministic 10,000-resample problem-cluster percentile bootstrap
interval. The contract freezes seed `260912` and linear quantiles. This analysis is
report-only and cannot change selection or primary pass/fail. A record must have a complete
learned-versus-stock pair for every model seed. Missing pairs produce explicit partial or
unavailable status and never silently change a denominator or become a synthetic label.

The legacy `splits.json` and `splits_v1.json` files used source-specific IDs and can place
identical problem payloads in different partitions. Quality V2 rejects them. Its registered
manifest is `data/release_v1/splits_quality_problem_v2.json`, generated from the canonical
problem-payload digest; structural training is unaffected.

## What the benchmark says about minorminer (from the study)

- Its local rule picks a certified-best structural action on 73 to 80% of decisions on
  Chimera and 39 to 48% on Pegasus and Zephyr.
- Its chains are within 0.05 of the best valid replacement in 82 to 96% of chain-seam
  states; mean regret 0.01 to 0.03.
- The released objective-guided embedder (`--embedder objective`: the best of ten
  minorminer embeddings by a 100-read surrogate probe, then destroy-and-repair moves of
  three coupled variables in which a learned screen ranks twenty repaired states per step
  and the surrogate evaluates the first; 100 evaluations in all, fresh-seed verification)
  raises solve probability on application graphs by +0.15 to +0.21 over stock minorminer on
  instances it never saw (Chimera, Pegasus, Zephyr; L-112, L-128), at 40 to 65% more qubits.
  The learned screen adds +0.031 over the same embedder without it at 40 evaluations
  (pre-registered, p = 1e-5) and is level at 100; `--screen-model none` gives the classical
  embedder. The released CHARME
  embedder is 0.17 to 0.23 below minorminer on the same instances.

## Reporting

A submission reports, per track and instance set: the config (topology, size, source,
seeds), the manifest digests of the datasets used, the metric with the registered
problem-clustered inference, and wall time. Primary mean `p_solve` deltas use the Hoeffding
interval and bounded-mean test described above. Bootstrap intervals are pointwise,
descriptive summaries for registered secondary endpoints only. Labels are
classical-surrogate quantities; a hardware panel script is provided separately in the study
repository and is not part of the benchmark's claims.

## Layout

`embedding.py`, `programming.py`, `surrogate.py` (types, programming, SA surrogate);
`inkdrop.py`, `planted_ising.py`, `apps.py` (instances); `objective.py`, `exact.py`
(objective and exact completion); `structural.py`, `quality_chain.py`, `quality_step.py`
(generators); `certify.py` (label audit); `handtests.py` (motifs); `models_structural.py`,
`models_chain.py`, `rebuild.py` (reference learned baselines and the rebuild loop);
`evaluate.py`, `cli.py`.
