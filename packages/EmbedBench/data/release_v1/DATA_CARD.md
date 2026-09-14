# EmbedBench release v1.1 (v1 built 2026-09-09 on apollo, `scripts/make_release.sh runs/release_v1 4`; v1.1 regenerates the quality corpora with the connected random-graph generator and records that carry the full problem and embedding, `runs/release_v1_1`)

Totals: 831 structural decisions (exact labels, re-certified) and 2879 quality decisions (surrogate labels).

Twelve JSONL corpora, 12 MB, each with a manifest (`*.manifest.json`: generator config,
seed, drop statistics, SHA-256 of the file) and a `SHA256SUMS` file. The JSONL files live on
the build host (`~/isingfold/EmbedBench/runs/release_v1`) and in the release archive; only
the manifests, checksums and the certification log are in the repository.

## Structural corpus: certified single-chain decisions with exact counterfactual labels

| file | records | instances | attempted | dropped (no margin) |
|---|---|---|---|---|
| structural_chimera5 | 204 | 240 | 1440 | 1236 |
| structural_pegasus3 | 326 | 240 | 1440 | 1113 |
| structural_zephyr2 | 301 | 240 | 1440 | 1137 |

Each record is a decision state (window, partial chains, frozen neighbours, feasible
actions) with the exact value of every action (feasible, minus qubits, minus longest chain)
by windowed branch-and-bound. Independent re-certification (`embedbench certify`, brute
force plus minorminer with fixed chains, 300 records per file): 6046 action values brute
checked, 0 mismatches; 520 infeasible actions attempted by minorminer, 0 contradictions
(`certify.log`).

## Quality corpus: chain-seam decisions scored by the surrogate

| file | records | instances | attempted | dropped (no margin) | generation failures |
|---|---|---|---|---|---|
| quality_chimera5_app | 79 | 120 | 480 | 125 | 0 |
| quality_chimera5_inkdrop | 467 | 146 | 1168 | 354 | 0 |
| quality_chimera5_random | 50 | 40 | 160 | 7 | 0 |
| quality_pegasus3_app | 196 | 120 | 480 | 42 | 0 |
| quality_pegasus3_inkdrop | 836 | 160 | 1280 | 66 | 0 |
| quality_pegasus3_random | 101 | 40 | 160 | 32 | 0 |
| quality_zephyr2_app | 190 | 120 | 480 | 55 | 0 |
| quality_zephyr2_inkdrop | 878 | 160 | 1280 | 152 | 0 |
| quality_zephyr2_random | 82 | 40 | 160 | 51 | 0 |

Each record is one chain of a valid embedding of a planted or application Ising problem,
with every valid replacement chain in its window and the surrogate solve probability of each
(two-stage scoring: 100-read screen, 400 reads for the top three, the resource-best and the
original, fresh seeds), the best index, the resource index and the original index. Records
whose best-versus-second margin is below the noise rule are dropped. The `random` family
was regenerated with the connected generator (v1.1; the first build lost 25 of 40 instances
per topology to disconnected G(n, m) draws).

## Provenance

Generator: `embedbench` package at commit of this file; seeds 100 (structural) and 200
(quality); per-instance seeds are SHA-256 of (seed, mode, index). Surrogate: fixed-schedule
simulated annealing, beta in [0.1, 8], 200 sweeps, majority-vote decoding, best chain
strength over a grid.

## Splits and reference baselines (v1.1)

`splits.json` is the original v1.1 manifest: SHA-256 of the source-specific instance ID,
with 0.7 / 0.1 / 0.2 train / validation / test thresholds. `splits_v1.json` is a byte-for-byte
archive of that manifest. It is retained for reproducibility, but it is not safe for new
quality-model training because identical `{h, J, e0}` problems can occur under different
source or topology IDs in different partitions.

`splits_v2.json` is the schema-v2 paper-facing manifest. It groups structural records by
instance ID and quality records by the canonical full `{h, J, e0}` payload digest, and it
records the SHA-256 of every input corpus. The training jobs may equivalently use the
subset manifests `splits_structural_v2.json` and `splits_quality_problem_v2.json`. The
loader verifies all corpus hashes and rejects cross-partition problem leakage.

The v2 quality membership is 2,055 / 291 / 533 records from 433 / 60 / 112 independent
problem groups. It is not identical to v1. Existing `baselines.json`,
`baselines_hiread.json`, and high-read labels describe the v1 test split only. They must be
regenerated for the v2 test membership before any confirmatory v2 test result is reported.
Build: `scripts/make_splits.py`, `scripts/release_baselines.py`.
