#!/usr/bin/env bash
# Gate 2 of the constructor ladder: one stage and seed per invocation, on an experiment host.
set -euo pipefail
if [ "$#" -lt 3 ]; then
    echo "usage: $0 STAGE(a|b) SEED OUT_PREFIX [probe options]" >&2
    exit 2
fi
stage=$1; seed=$2; out=$3; shift 3
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
export ISINGFOLD_SRC="$repo_root/src"
export PYTHONPATH="$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
mkdir -p "$(dirname "$out")"
if [ -e "$out.log" ]; then
    echo "choose a fresh output prefix to preserve existing runs: $out" >&2
    exit 2
fi
python -u probes/constructor_curriculum.py --stage "$stage" --seed "$seed" --out "$out.pt" "$@" 2>&1 | tee "$out.log"
