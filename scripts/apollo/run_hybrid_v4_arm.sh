#!/usr/bin/env bash
# One opt-in ablation per invocation; execute on the registered experiment host.
set -euo pipefail
if [ "$#" -lt 4 ]; then
    echo "usage: $0 CORPUS OUT_PREFIX ARM SEED [additional trainer options]" >&2
    echo "ARM: legacy | support | capacity | context | critic | warm" >&2
    exit 2
fi
corpus=$1
out=$2
arm=$3
seed=$4
shift 4
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
export ISINGFOLD_SRC="$repo_root/src"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
actor=local
features=legacy
support=legacy
baseline=loo
warm=0
case "$arm" in
    legacy) ;;
    support) support=all_free ;;
    capacity) support=all_free; features=capacity ;;
    context) support=all_free; features=capacity; actor=contextual ;;
    critic) support=all_free; features=capacity; actor=contextual; baseline=value ;;
    warm) support=all_free; features=capacity; actor=contextual; baseline=value; warm=5 ;;
    *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac
mkdir -p "$(dirname "$out")"
if [ -e "$out.pt" ] || [ -e "$out.log" ]; then
    echo "output exists; choose a fresh prefix to preserve evidence: $out" >&2
    exit 2
fi
python -u probes/train_hybrid_rl.py \
    --corpus "$corpus" --out "$out.pt" --seed "$seed" --fast \
    --actor "$actor" --features "$features" --root-support "$support" \
    --baseline "$baseline" --warmstart-epochs "$warm" --entropy-coef 0 \
    --objective quality --iterations 100 --eval-every 10 \
    --episodes-per-instance 4 --instances-per-iteration 4 \
    --eval-deadline 60 --layout-router-secs 2 --layout-tries 2 --select-cap 6 \
    "$@" 2>&1 | tee "$out.log"
