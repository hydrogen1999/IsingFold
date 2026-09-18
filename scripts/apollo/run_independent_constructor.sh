#!/usr/bin/env bash
# Activate the project environment first; run one stage/seed on an experiment host.
set -euo pipefail
if [ "$#" -lt 4 ]; then
    echo "usage: $0 CORPUS OUT_PREFIX {feasibility|quality} SEED [trainer options]" >&2
    exit 2
fi
corpus=$1
out=$2
stage=$3
seed=$4
shift 4
case "$stage" in
    feasibility|quality) ;;
    *) echo "stage must be feasibility or quality" >&2; exit 2 ;;
esac
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
export ISINGFOLD_SRC="$repo_root/src"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
mkdir -p "$(dirname "$out")"
if [ -e "$out.pt" ] || [ -e "$out.log" ]; then
    echo "choose a fresh output prefix to preserve existing runs: $out" >&2
    exit 2
fi
python -u probes/train_constructor_rl.py \
    --corpus "$corpus" --out "$out.pt" --objective "$stage" --seed "$seed" \
    --actor contextual --features construction --value-baseline value \
    --iterations 200 --eval-every 10 --episodes-per-instance 4 \
    --instances-per-iteration 4 --episode-seconds 30 --deadline 60 \
    --max-steps 1000 --comparison minorminer --select-cap 6 \
    "$@" 2>&1 | tee "$out.log"
