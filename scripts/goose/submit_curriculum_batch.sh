#!/usr/bin/env bash
# Submit several curriculum runs as ONE Slurm job on goose (the account allows only a couple of
# jobs in the queue at once): JOBNAME SPECFILE, where each spec line is NAME|PROBE ARGS...
# Every run gets its own core inside the allocation and its own log under runs/curriculum/.
set -euo pipefail
if [ "$#" -ne 2 ]; then
    echo "usage: $0 JOBNAME SPECFILE" >&2
    exit 2
fi
jobname=$1; spec=$2
root=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
mkdir -p "$root/runs/curriculum/slurm"
n=$(grep -cve '^\s*$' "$spec")
if [ "$n" -lt 1 ]; then
    echo "empty spec" >&2
    exit 2
fi
job="$root/runs/curriculum/slurm/$jobname.sbatch"
{
    echo '#!/bin/bash'
    echo "#SBATCH -J $jobname"
    echo '#SBATCH -p gpu'
    echo "#SBATCH -c $n"
    echo "#SBATCH --mem=$((8 * n))G"
    echo '#SBATCH -t 3-00:00:00'
    echo "#SBATCH -o $root/runs/curriculum/slurm/$jobname.out"
    echo "cd $root"
    echo 'source ~/isingfold/.venv/bin/activate'
    echo 'export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1'
    while IFS='|' read -r name args; do
        [ -z "$name" ] && continue
        if [ -e "$root/runs/curriculum/$name.log" ]; then
            echo "choose a fresh name to preserve existing runs: $name" >&2
            exit 2
        fi
        echo "python -u probes/constructor_curriculum.py $args > runs/curriculum/$name.log 2>&1 &"
    done < "$spec"
    echo 'wait'
} > "$job"
sbatch "$job"
