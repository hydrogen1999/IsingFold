# IsingFold architecture

System, model, data, training, and evaluation architecture, 12 September 2026.

## Read first

The architecture decisions at the beginning fix the system boundary. Sections 1 to 3 define objective, environment and theory. Sections 4 to 6 specify neural tensors, PPO and generation. Sections 7 and 8 define evidence and implementation contracts. Appendix B contains worked checks. The exact execution sequence is in [`../TRAINING_OPERATIONS.md`](../TRAINING_OPERATIONS.md).

IF-Q3-S0 is the registered endpoint for this system; historical V2 labels retain their original semantics. Architecture definitions and experimental results remain separate artifacts. The scientific chain uses prepared-v4, publisher attestation v2, ground-certificate protocol v2 with a target-free root and partition receipts, selector labels and bundle v4, quality-v7 with shard/merge v5 and preflight v2 for semantic target `if-q3-s0-qmu-5`, release gate v6, representation and RL freeze v4, checkpoint v3, and training-run receipt v4. Initializer-bank, complete-system, tuned-stock, resumable-evaluation, paired, and final-strength artifact versions are listed in Appendix A.

## Build

Run `make` for the modular source, or run `pdflatex` twice on the supplied self-contained `ISINGFOLD_RL_ARCHITECTURE_EN.tex`. Rebuild the self-contained source with `latexpand --fatal --output ISINGFOLD_RL_ARCHITECTURE_EN.tex main.tex` before a release. Both forms require a standard TeX Live installation with TikZ, tcolorbox, latexmk and common math/table packages. No external figures or bibliography downloads are needed.

## Contents

- `main.tex` and `sections/`: editable modular document.
- `ISINGFOLD_RL_ARCHITECTURE_EN.tex`: self-contained equivalent.
- `ISINGFOLD_RL_ARCHITECTURE_EN.pdf`: compiled combined document.
- `pilot_registry.yaml`: historical design input; executable registries live under `configs/`.
- `source_digests.json`: historical input-provenance digests, not runtime or experiment identity; do not regenerate it from current files.

Giới thiệu ngắn: IsingFold dùng RL để cải thiện embedding khả thi bằng các thao tác nhiều chain, với validity do môi trường chính xác kiểm soát. Ink-drop và frustrated-loop cung cấp hai loại chứng chỉ khác nhau, không chứng minh embedding tối ưu. Prepared-v4 tách target theo partition; initializer bank được seal cho từng seed PPO; đánh giá dài được chia shard theo lineage và merge lại theo census đầy đủ. Quy trình thực nghiệm dùng grid cố định 9 cộng 18, ba seed đầy đủ, và đánh giá complete-system trên toàn bộ mẫu trước initialization.
