#!/usr/bin/env bash
# Build the EmbedBench v1 release corpora with manifests and a checksum file.
#   bash scripts/make_release.sh OUTDIR [JOBS]
set -u
OUT=${1:-runs/release_v1}; J=${2:-4}; mkdir -p "$OUT"
export OMP_NUM_THREADS=1
gen() { python -m embedbench.cli "$@" ; }
# structural corpus: certified single-chain decisions with exact counterfactual labels
for T in chimera:5 pegasus:3 zephyr:2; do
  top=${T%%:*}; sz=${T##*:}
  gen generate-structural --topology $top --size $sz --n-instances 60 --seed 100 --jobs $J --out "$OUT/structural_${top}${sz}.jsonl" > "$OUT/structural_${top}${sz}.log" 2>&1
done
# quality corpus: chain-seam decisions scored by the surrogate, three graph families, two sources
for T in chimera:5 pegasus:3 zephyr:2; do
  top=${T%%:*}; sz=${T##*:}
  for G in random inkdrop app; do
    gen generate-chain --topology $top --size $sz --graph $G --n-instances 40 --seed 200 --jobs $J --out "$OUT/quality_${top}${sz}_${G}.jsonl" > "$OUT/quality_${top}${sz}_${G}.log" 2>&1
  done
done
# independent re-certification of a sample of the structural corpus
gen certify "$OUT"/structural_*.jsonl --limit 300 > "$OUT/certify.log" 2>&1
( cd "$OUT" && sha256sum *.jsonl *.manifest.json > SHA256SUMS )
echo done > "$OUT/DONE"
