# The representation ablation on the diverse corpus, split by logical family rather than by
# lineage. The cache's own split holds out lineages of every family, so it tests transfer
# between coefficient draws of graphs the model has seen. Holding a family out tests transfer
# between graph structures, which is the claim any representation result has to survive.
#
# Five arms, one seed, on goose's four cores. Seed noise on this quantity is about 0.008, so
# this run can separate a large effect and cannot separate a small one; it is the checklist
# item, not the decision.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=1
CACHE=runs/diverse/labels.pkl
FAM=${1:-modular}
mkdir -p runs/diverse_abl
run () {
  name=$1; shift
  LOG=runs/diverse_abl/${name}_${FAM}.log
  if [ -s "$LOG" ] && grep -q "SUCCESSOR SCORER DONE" "$LOG"; then echo "skip $name"; return; fi
  python -u probes/train_successor.py --cache "$CACHE" --epochs 400 --width 64 \
    --learning-rate 1e-3 --clip 10.0 --rank-weight 1.0 --inner-fraction 0.0 \
    --holdout-family "$FAM" --seed 0 --out runs/diverse_abl/${name}_${FAM}.json "$@" \
    > "$LOG" 2>&1 &
  sleep 3
}
run F0
run Fpos   --coords
run Fphys  --physics
run Fspace --space
run Fall   --coords --physics --space
wait
echo "DIVERSE ABLATION FINISHED"
