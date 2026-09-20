#!/usr/bin/env bash
# Reverse-start curriculum at the congested cell, with the schedule the measurement asks for.
#
# The first attempt started at 0.97 of a 110-state trajectory and stepped back by 0.1. Both are
# far too coarse here: a direct test on one instance finds a valid COMMIT from 0.999 and 0.99 of
# the walk and STOP_NO_VALID from 0.97, so two decisions earlier is already past what an untrained
# policy can finish at fill 0.90. It starts at the last state now, where only the COMMIT remains,
# and steps back one decision at a time.
#
# The empty-start mix is cut to 0.05. Those episodes run the whole 300-second deadline while an
# assisted one at the end of the walk takes seconds, so they set the iteration cost, and they
# contribute nothing until the start has walked most of the way back.
set -eu
cd "$HOME/prj_IsingFold_pr2"
source "$HOME/isingfold/.venv/bin/activate"
export ISINGFOLD_SRC="$HOME/prj_IsingFold_pr2/src"
export PYTHONPATH="$ISINGFOLD_SRC"
export OMP_NUM_THREADS=1
export ISINGFOLD_FAST_INTERNAL_ASSERTS=1
mkdir -p runs/reverse/coarse_schedule
mv runs/reverse/rev_*.log runs/reverse/coarse_schedule/ 2>/dev/null || true

run () {  # tag corpus cells init seed
  setsid nohup python -u probes/constructor_curriculum.py --stage corpus \
    --corpus "$HOME/prj_IsingFold/runs/small/$2/corpus" --cells "$3" \
    --features physics --expand-features --support wide \
    --train 8 --heldout 2 --manifest-split --init "runs/curriculum/$4" --seed "$5" \
    --iterations 400 --episodes 6 --instances-per-iteration 3 --eval-episodes 3 --eval-every 10 \
    --eval-sets heldout --max-steps 600 --episode-seconds 600 --train-episode-seconds 300 \
    --stop-bias -6 --prefix-unit trajectory --prefix-fraction 0.999:0.0 \
    --prefix-schedule mastery --mastery-step 0.01 --mastery-threshold 0.8 \
    --prefix-empty-mix 0.05 --out "runs/reverse/$1.pt" \
    > "runs/reverse/$1.log" 2>&1 < /dev/null &
}
run rev2_p3_f90_s0 pegasus3 fill90-a3.0 F_local_s0.pt 0
run rev2_p3_f90_s1 pegasus3 fill90-a3.0 F_local_s0.pt 1
run rev2_z2_f90_s0 zephyr2  fill90-a3.0 G_local_s0.pt 0
sleep 40
for f in runs/reverse/rev2_*.log; do
  printf "%-18s %s err | %s\n" "$(basename "$f" .log)" "$(grep -c Traceback "$f")" \
    "$(grep -o '"reverse_start".*' "$f" | head -1 | cut -c1-100)"
done
