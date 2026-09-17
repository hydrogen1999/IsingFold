#!/usr/bin/env bash
# Submit one curriculum run to Slurm on goose: NAME, then the probe's own arguments.
# A login session on goose is capped at one CPU; Slurm allocations are not.
set -euo pipefail
if [ "$#" -lt 2 ]; then
    echo "usage: $0 NAME PROBE_ARGS..." >&2
    exit 2
fi
name=$1; shift
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$root/runs/curriculum/slurm"
if [ -e "$root/runs/curriculum/$name.log" ]; then
    echo "choose a fresh name to preserve existing runs: $name" >&2
    exit 2
fi
job="$root/runs/curriculum/slurm/$name.sbatch"
{
    echo '#!/bin/bash'
    echo "#SBATCH -J $name"
    echo '#SBATCH -p gpu'
    echo '#SBATCH -c 1'
    echo '#SBATCH --mem=16G'
    echo '#SBATCH -t 3-00:00:00'
    echo "#SBATCH -o $root/runs/curriculum/slurm/$name.out"
    echo "cd $root"
    echo 'source ~/isingfold/.venv/bin/activate'
    echo 'export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1'
    printf 'exec python -u probes/constructor_curriculum.py'
    for a in "$@"; do printf ' %q' "$a"; done
    echo " > runs/curriculum/$name.log 2>&1"
} > "$job"
sbatch "$job"
