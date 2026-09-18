#!/usr/bin/env bash
# Cloning did not converge in forty epochs. The loss fell from 3.411 to 2.522, which is the
# probability the actor puts on the witness-consistent candidate set rising from 3.3 to 8.0
# percent, and it was still falling. Two questions follow and this answers both at once: does it
# converge given time, and does the richer representation fit the teacher better?
#
# The physics arm starts from the local20 checkpoint zero-padded into the 32-channel schema, so
# its twelve new channels begin with no influence on any action logit and the comparison is the
# representation rather than the initialisation.
set -eu
cd "$HOME/prj_IsingFold_pr2"
source "$HOME/isingfold/.venv/bin/activate"
export ISINGFOLD_SRC="$HOME/prj_IsingFold_pr2/src"
export PYTHONPATH="$ISINGFOLD_SRC"
export OMP_NUM_THREADS=1
export ISINGFOLD_FAST_INTERNAL_ASSERTS=1

run () {  # tag features init extra
  setsid nohup python -u probes/constructor_clone.py \
    --corpus runs/quality/pegasus3_f30 --cells fill30-a3.0 \
    --train 20 --heldout 4 --features "$2" --support wide \
    --init "runs/curriculum/$3" $4 --epochs 250 --learning-rate 0.01 \
    --max-steps 900 --teacher-seconds 300 --eval-episodes 3 --eval-every 50 \
    --episode-seconds 120 --stop-bias -6 --out "runs/quality/$1.pt" \
    > "runs/quality/$1.log" 2>&1 < /dev/null &
}
run clone_long_local_p3   local   F_local_s0.pt ""
run clone_long_physics_p3 physics F_local_s0.pt "--expand-features"
sleep 25
for f in runs/quality/clone_long_*_p3.log; do
  printf "%-26s %s err, %s bytes\n" "$(basename "$f" .log)" "$(grep -c Traceback "$f")" "$(wc -c < "$f")"
done
