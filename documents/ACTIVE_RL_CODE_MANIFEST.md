# Active IsingFold RL code boundary

This manifest defines which code is authoritative for new IsingFold training
and evaluation. A file being retained in the repository does not make it part
of the learned method.

## Canonical project root

`/Users/nguyencongt/Documents/IsingFold/IsingFold`

All local development commands must start from this root. HPC jobs execute an
immutable source snapshot derived from this tree and return artifacts to the
read-only mirror indexed in
`docs/provenance/HPC_ARTIFACT_INDEX_20260914.json`.

## Active implementation

| Path | Authority |
|---|---|
| `src/isingfold/rl/` | Environment, action grammar, IF-Core GNN actor-critic, PPO, checkpointing, quality objective, data trust boundary, and evaluation |
| `src/isingfold/` | Self-contained Ising programming, embedding, and surrogate primitives used by the RL runtime |
| `src/lac_minorminer/` | Replaceable Python orchestration seams used by the current native search core |
| `cpp/` | Native candidate generation, routing, validity, accounting, search session, and Python bindings |
| `packages/EmbedBench/` | Independently installable data generation, certification, publication, and baseline package |
| `configs/` | Registered experiment, objective, capacity, and quality protocol configurations |
| `scripts/` | Apollo direct launchers, Goose Slurm launchers, runtime freezing, and publication workflow |
| `tests/` and `packages/EmbedBench/tests/` | Normative unit, integration, trust-boundary, and protocol checks |
| `documents/` | Current model and system architecture contracts |

EmbedBench and IsingFold share one repository root but retain a one-way
publication boundary. EmbedBench publishes authenticated records. The learned
runtime consumes those records through `src/isingfold/rl/data/` and must not
import generator or evaluator-target code.

## Retained but not active

The following material is excluded from new model definitions and paper-facing
training commands:

- sibling workspaces `LAC_GPT/`, `LAC_B/`, and `LAC/`;
- root `probes/` experiments unless a specific probe is explicitly promoted
  into a versioned, tested entry point;
- `src/isingfold_lac_b/`, retained only for compatibility and historical tests;
- root `paper/` inherited analysis and any old result without an authenticated
  receipt;
- `packages/EmbedBench/tools/legacy_provenance/` and the packaged
  objective-guided embedder when reconstructing historical evidence.

These exclusions do not remove baseline implementations needed for comparison.
Stock minorminer, random/greedy controls, classical search variants, and the
frozen selector remain part of the evaluation suite, not part of the learned
policy.

## Last executed reproducible closure

- Last executed v10 source identity: `3900af306ed4dd9396b872d6322e0782a9934347c13764b477a10084a0ab8cee`
- Last executed v10 inventory: 212 selected scientific and operational files
- Current validated diagnostic checkpoint: Goose job 3279, seed 1103
- Artifact index: `docs/provenance/HPC_ARTIFACT_INDEX_20260914.json`

This digest identifies the immutable v10 execution closure, not the whole
current dirty monorepo. Any future training run requires a new freeze. The
checkpoint is mechanically valid but its flat training return does not establish
a learned improvement claim.
