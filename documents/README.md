# IsingFold normative documentation

The files in this directory define the current IsingFold system, model, data, training, and
evaluation contracts. They take precedence over the earlier bilingual handbooks in `../docs/`,
which remain available as historical design background and evidence provenance.

## Authoritative sources

| Scope | Editable source | Distribution artifact |
|---|---|---|
| Current project specification, English | `ISINGFOLD_PROJECT_SPEC_EN.md` | `ISINGFOLD_PROJECT_SPEC_EN.tex` and `ISINGFOLD_PROJECT_SPEC_EN.pdf` |
| Current project specification, Vietnamese | `ISINGFOLD_PROJECT_SPEC_VI.md` | `ISINGFOLD_PROJECT_SPEC_VI.tex` and `ISINGFOLD_PROJECT_SPEC_VI.pdf` |
| System and experiment architecture | `IsingFold_Architecture_Rev2/main.tex` plus `IsingFold_Architecture_Rev2/sections/` | `ISINGFOLD_RL_ARCHITECTURE_EN.tex` and `ISINGFOLD_RL_ARCHITECTURE_EN.pdf` |
| IF-Core model and RL contract | `ISINGFOLD_MODEL_SPEC.md` and `ISINGFOLD_MODEL_SPEC.tex` | `ISINGFOLD_MODEL_SPEC.pdf` |
| Exact operational sequence | `TRAINING_OPERATIONS.md` | Executable command and artifact contract |
| Stock-minorminer validation tuning | `EXTERNAL_BASELINE_TUNING_SPEC.md` | Finite candidate and freeze protocol |

The modular architecture source is canonical. Its self-contained LaTeX copy exists for review,
submission, and archival use. The root-level and modular-directory copies of that file and PDF must
remain byte-identical after a release build.

## Scientific identity chain

The current compatibility chain is:

1. CandidateBank-v2 input, corpus-design manifest v2, and partition-sealed prepared schema v4;
2. publisher attestation v2 plus ground-certificate protocol v2, published as a target-free root
   with separate train, validation, and test receipts;
3. selector-label manifest v4 and selector bundle v4;
4. quality record v7 for semantic target `if-q3-s0-qmu-6`, continuation receipt v2, target-free
   resolution production plan and row v2, resolution delta and verification v2, capacity selection
   and canary v2, quality shard and merge envelopes v5, and quality preflight v2;
5. release gate v6, representation selection v4, and RL-value freeze v4;
6. scientific checkpoint v3 and training-run receipt v4;
7. learned complete-system receipt v3, report v4, and three-seed aggregate v2;
8. external tuning run and selection v2, external complete-system receipt v3 and report v4, and
   paired learned-versus-stock aggregate v4;
9. resumable evaluation plan v2, shard receipt v3, and merge receipt v2;
10. final-strength config v1, sampler identity v1, plan v2, row v1, execution manifest v2, shard
    v1, merge v1, and audit receipt v2.

Every target-opening stage carries a partition-specific `TargetAccessReceipt` and the matching
ground-partition authority. Planning authenticates only the public root and public population, so
it cannot expose evaluator targets. A digest proves byte identity only. It does not establish
authority or mathematical correctness by itself.

Publication quality planning and continuation replay additionally consume the exact externally
pinned persistent K=2 initializer bank used by deployment. The global contract binds the prepared
corpus, train partition, configuration, context, schedule, and bank manifest; each row and receipt
binds one episode and bootstrap snapshot. Legacy online initializer restarts are diagnostic-only.
See `decisions/ADR-004-deployment-parity-quality-initializers.md`.

## Experiment contract

The primary endpoint is failure-aware selected-strength IF-Q3-S0 utility. Valid returns are scored
by fresh majority-decoded ground-state solve probability at the strength chosen by a frozen
program-feature selector. Ordinary failures remain in the denominator with utility zero. Qubit use,
work, connectivity, latency, and memory are constraints or secondary axes, not replacements for the
quality objective.

The main model decision contains exactly nine representation cells and eighteen RL-value cells at
seeds 1103, 2207, and 3301. Test data remain sealed through both decisions. Complete-system
confirmation retrains all three seeds. PPO consumes a target-free, sealed deployment-initializer
bank whose deterministic conditional retry schedule supplies the post-initializer Profile-I state;
final evaluation restores the unconditional pre-initialization denominator. Stock minorminer is
tuned only on the complete validation population and frozen before test access. Publication
analysis tests valid-return noninferiority before unconditional utility superiority. The separate
four-strength diagnostic is registered in
`configs/final_strength_audit_v1.json`; it uses at least 128 lineages, 128 shards, 4,096 reads per
block, and independent A/B blocks. It cannot update training or either selection receipt and is not
primary superiority evidence.

## HPC boundary

Apollo commands run directly because Apollo has no Slurm. Goose computation runs only inside a
Slurm allocation. Initializer-bank generation, quality labeling, complete-system evaluation, stock
tuning, and final-strength auditing have deterministic shard workflows. Evaluation plans bind a
host-independent compute class; each shard also records its actual node. Publication execution uses
a pinned virtual environment on Apollo or a pinned Apptainer runtime and image on Goose. Bare-metal
mode is diagnostic-only. Mergers reject missing, duplicate, overlapping, or authority-mismatched
inputs and recompute canonical summaries before publication.

## Build and release

From `documents/IsingFold_Architecture_Rev2`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexpand --fatal --output ISINGFOLD_RL_ARCHITECTURE_EN.tex main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error ISINGFOLD_RL_ARCHITECTURE_EN.tex
```

From `documents/`:

```bash
/opt/anaconda3/bin/pandoc ISINGFOLD_PROJECT_SPEC_EN.md --standalone --toc \
  -V documentclass=article -V papersize=a4 -V fontsize=10pt \
  -V geometry:margin=22mm -V colorlinks=true -o ISINGFOLD_PROJECT_SPEC_EN.tex
/opt/anaconda3/bin/pandoc ISINGFOLD_PROJECT_SPEC_VI.md --standalone --toc \
  -V documentclass=article -V papersize=a4 -V fontsize=10pt \
  -V geometry:margin=22mm -V colorlinks=true -V fontenc=T5 \
  -o ISINGFOLD_PROJECT_SPEC_VI.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error ISINGFOLD_PROJECT_SPEC_EN.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error ISINGFOLD_PROJECT_SPEC_VI.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error ISINGFOLD_MODEL_SPEC.tex
```

Copy the self-contained architecture source and PDF to `documents/` only after successful builds,
then verify byte equality. Clean transient LaTeX files with `latexmk -c`. The
`IsingFold_Architecture_Rev2_Source.zip` archive contains the modular sources, self-contained source,
compiled PDF, README, Makefile, historical input-digest manifest, and documentary pilot registry.
Do not rewrite `source_digests.json`: it records the supplied historical source set, not current
runtime or release hashes.

## Vietnamese reader note

Tài liệu trong thư mục này là đặc tả chuẩn của IsingFold hiện tại. Các handbook song ngữ trong
`../docs/` chỉ là tài liệu nền lịch sử. Khi hai bộ tài liệu khác nhau, phải dùng architecture, model
specification, training operations và external-baseline specification trong `documents/`. Mọi kết
quả khoa học phải đi qua prepared-v4, ground-certificate protocol v2 với quyền truy cập theo từng
partition, initializer bank đã seal, grid 9 cộng 18 với ba seed, đánh giá complete-system,
minorminer được tune chỉ trên validation, và final audit bốn strength với hai block độc lập.
