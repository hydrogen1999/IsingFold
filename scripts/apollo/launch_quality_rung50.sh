#!/usr/bin/env bash
# The quality cell itself, warm-started from the rung below rather than from the fragments.
#
# Starting from the 12-to-20 variable fragment checkpoints, held-out validity here was 0.0. The
# same checkpoints reach 0.75 to 1.00 at fill 0.30, which is 28 and 38 variables on the same two
# hosts under the same wide support, so the gap was the jump in size and not the policy. These
# runs start from that rung's validation-selected weights.
set -eu
cd "$HOME/prj_IsingFold_pr2"
source "$HOME/isingfold/.venv/bin/activate"
export ISINGFOLD_SRC="$HOME/prj_IsingFold_pr2/src"
export PYTHONPATH="$ISINGFOLD_SRC"
export OMP_NUM_THREADS=1
export ISINGFOLD_FAST_INTERNAL_ASSERTS=1

rung () {  # tag corpus init seed
  setsid nohup python -u probes/constructor_curriculum.py --stage corpus \
    --corpus "runs/quality/$2" --cells fill50-a3.0 --features local --support wide \
    --train 12 --heldout 6 --manifest-split --init "runs/quality/$3" --seed "$4" \
    --iterations 200 --episodes 4 --instances-per-iteration 2 --eval-episodes 3 --eval-every 2 \
    --eval-sets heldout --max-steps 300 --episode-seconds 120 --train-episode-seconds 60 \
    --stop-bias -6 --prefix-fraction 0.9:0.0 --prefix-schedule mastery --prefix-empty-mix 0.2 \
    --out "runs/quality/$1.pt" > "runs/quality/$1.log" 2>&1 < /dev/null &
}
rung r50_p3_s0 pegasus3_f50 r30_p3_s0.best.pt 0
rung r50_p3_s1 pegasus3_f50 r30_p3_s1.best.pt 1
rung r50_z2_s0 zephyr2_f50  r30_z2_s0.best.pt 0
rung r50_z2_s1 zephyr2_f50  r30_z2_s1.best.pt 1
sleep 25
for f in runs/quality/r50_*.log; do
  printf "%-12s %s err | %s\n" "$(basename "$f" .log)" "$(grep -c Traceback "$f")" \
    "$(grep -o '"rate": [0-9.]*' "$f" | tr '\n' ' ')"
done
