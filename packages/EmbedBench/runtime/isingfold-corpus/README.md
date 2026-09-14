# IsingFold corpus runtime

This directory defines one corpus-generation runtime for Apollo and Goose.  It
matches the versions recorded by
`embedbench.isingfold_corpus_shard._generation_provenance`: CPython 3.12.3,
EmbedBench 0.1.0, dimod 0.12.22, dwave-networkx 0.8.19, dwave-samplers 1.8.0,
minorminer 0.2.22, networkx 3.6.1, numpy 2.4.4, and scipy 1.18.0.

`requirements-linux-x86_64-py312.lock` also pins the three transitive runtime
packages and the build tools.  It accepts only the named Linux x86_64 wheels and
their PyPI SHA-256 digests.  `runtime-lock.json` is the machine-readable contract.
The Apptainer definition pins the Docker manifest digest and exposes exactly
`/opt/isingfold/venv/bin/python`.

Version equality is not sufficient when a package is installed editable from a
modified checkout.  In particular, an editable Minorminer tree can still report
version 0.2.22.  Production Apollo generation must therefore use a clean virtual
environment built from this wheel lock.  It must not use a developer environment
with editable, VCS-installed, or direct-URL external dependencies.

## Apollo runtime

Apollo has no Slurm.  Build its virtual environment directly with a trusted
CPython 3.12.3 executable:

```bash
python3.12 -m venv /absolute/new/corpus-venv
/absolute/new/corpus-venv/bin/python -m pip install \
  --require-hashes \
  -r runtime/isingfold-corpus/requirements-linux-x86_64-py312.lock
/absolute/new/corpus-venv/bin/python -m pip install \
  --no-deps --no-build-isolation .
/absolute/new/corpus-venv/bin/python -m pip check
```

Confirm that `platform.python_version()` is exactly `3.12.3`.  Then compute the
source and provenance commitments with `scripts/verify_isingfold_corpus_runtime.sh`.
The provenance commands require a final `THREADS` argument; production uses the
runtime-contract value `1`.
Record the lock, source, Python executable, and provenance hashes outside the run
directory.  Run the same strict validator with
`--expected-python-executable /absolute/new/corpus-venv/bin/python`; it rejects
editable, VCS, and direct-URL external dependencies even when they report the
expected version.  Invoke `scripts/apollo_isingfold_corpus.sh` only after that
receipt succeeds.

```bash
PYTHONPATH=/absolute/staged/EmbedBench/src \
  /absolute/new/corpus-venv/bin/python \
  /absolute/staged/EmbedBench/scripts/validate_isingfold_corpus_image.py \
  --runtime-lock /absolute/staged/EmbedBench/runtime/isingfold-corpus/runtime-lock.json \
  --source-root /absolute/staged/EmbedBench \
  --expected-provenance-sha256 RECORDED_PROVENANCE_SHA256 \
  --expected-python-executable /absolute/new/corpus-venv/bin/python
```

## Prospective identity preflight

The plan is deterministic and can be published before a runtime exists.  The
authoritative preflight cannot.  It binds the current generator provenance, so it
must be created by the frozen Linux runtime, not by a developer Python environment.
For plan v3, publish preflight schema v2 once to a new path:

```bash
PYTHONPATH=/absolute/staged/EmbedBench/src \
  MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /absolute/frozen/python -m embedbench.isingfold_corpus_cli preflight \
    --plan /absolute/frozen/plan_v3/corpus_plan.json \
    --expected-plan-sha256 PLAN_SHA256 \
    --expected-source-sha256 SOURCE_SHA256 \
    --workers 10 \
    --out /absolute/frozen/plan_v3/prospective_preflight_v2.json
```

Record the emitted preflight file SHA-256, preflight record digest, identity-map
digest, and generation-provenance digest outside the plan and run directories.
Then run `verify-preflight` with all four values on Apollo and again inside the
pinned Goose image.  This command recomputes every prospective row.  Both sites
must report 3,082 identities and the same receipts before a canary starts.  A
developer-machine preflight can be retained under an explicit `audit_only` path,
but it is diagnostic and must never be supplied to either production launcher.

Preflight v2 rejects any repeated `split_unit_id` or exact logical
`problem_sha256` over the complete plan.  Shard generation also performs a cheap
authenticated read of the same full identity map before doing expensive work.

## Resource policy

Corpus prospecting and shard generation are CPU-bound.  They do not use a GPU and
must not request one.  Apollo parallelism is controlled by `CONCURRENCY`, with one
thread per worker.  Goose parallelism is a Slurm array with CPU and memory limits;
the compute script intentionally has no GPU resource directive.  GPUs belong to
the later neural-model training stage, which is outside this runtime contract.

## Goose image build

Goose compute must go through Slurm.  Do not execute the build launcher with
`bash`, and do not run `apptainer build` on the login node.  Stage an immutable
EmbedBench source tree, calculate all required hashes independently, export the
variables required at the top of
`scripts/goose_build_isingfold_corpus_runtime.sbatch`, and submit:

```bash
/opt/slurm/bin/sbatch --export=ALL \
  scripts/goose_build_isingfold_corpus_runtime.sbatch
```

The external commitments include the source inventory, definition, pip lock,
runtime lock, validator, build launcher, shared runtime helper, Apptainer
executable, and Apollo generation-provenance digest.  Recording operational files
separately is intentional because the corpus source inventory covers scientific
Python modules, not every operational script.

The job invokes `apptainer build --fakeroot` through `srun`.  Before publishing
the SIF, it binds the staged source read-only and runs
`scripts/validate_isingfold_corpus_image.py`.  That guard requires:

1. `/opt/isingfold/venv/bin/python` and Python 3.12.3;
2. exact installed versions for every locked distribution;
3. successful `python -m pip check`;
4. imports from the committed staged source, not the image fallback copy; and
5. a generation-provenance digest equal to the Apollo commitment.

The job never overwrites an image.  It publishes the SIF only after validation,
then writes `IMAGE.sha256` and `IMAGE.validation.json`.  Pass the recorded image
digest, installation inventory digest, and `/opt/isingfold/venv/bin/python` to
`scripts/goose_isingfold_corpus.sbatch` for generation.

## Cross-site canary gate

Production mode is closed until Apollo and Goose reproduce one full shard byte
for byte.  Run both generation launchers with `ISINGFOLD_CORPUS_MODE=canary-only`,
the same `SHARD_FIRST=SHARD_LAST`, `SHARD_STEP=1`, one worker or one Slurm array
task, and a dedicated site-local `CANARY_OUTPUT_ROOT`.  Every runtime, source,
plan, provenance, installation, and thread check still applies in this mode.

Copy each site's three shard files into new read-only comparison directories on
the clean Apollo runtime.  Publish the parity statement with the exact frozen
plan and the four externally recorded commitments:

```bash
python -m embedbench.isingfold_cross_site_canary publish \
  --apollo-shard /absolute/comparison/apollo/shard-N \
  --goose-shard /absolute/comparison/goose/shard-N \
  --plan /absolute/frozen/corpus_plan.json \
  --out /absolute/attestations/cross-site-canary.json \
  --plan-sha256 PLAN_SHA256 \
  --source-sha256 SOURCE_SHA256 \
  --generation-provenance-sha256 PROVENANCE_SHA256 \
  --installation-sha256 INSTALLATION_SHA256
```

The publisher uses the production shard loader, proves that the complete config
and request subset came from the authenticated plan, compares all three regular
files exactly, and publishes once without replacement.  Record the attestation
file SHA-256 outside all run trees.  Production launchers require
`ISINGFOLD_CORPUS_MODE=production`, the canonical attestation path and root, and
`ISINGFOLD_CORPUS_EXPECTED_PARITY_ATTESTATION_SHA256`; missing, legacy, changed,
or plan-detached attestations stop before corpus generation.

The definition is reproducible at the dependency and source-contract level.  A
built SIF can still contain build timestamps, so the published SIF SHA-256 remains
the final executable identity and must be recorded for every production run.
