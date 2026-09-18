#!/usr/bin/env bash
# The rung the ladder was missing. From 28 variables the constructor reaches 0.75 to 1.00; at 50
# it reaches 0.0 even warm-started from that checkpoint. The same fill-0.50 corpora carry a
# longer-chain shape at alpha 2.0 with 39 variables, which sits between the two on the same hosts
# and the same wide support.
set -eu
cd "$HOME/prj_IsingFold_pr2"
source "$HOME/isingfold/.venv/bin/activate"
export ISINGFOLD_SRC="$HOME/prj_IsingFold_pr2/src"
export PYTHONPATH="$ISINGFOLD_SRC"
export OMP_NUM_THREADS=1
export ISINGFOLD_FAST_INTERNAL_ASSERTS=1
mkdir -p runs/quality/rung50_from_28
mv runs/quality/r50_*.log runs/quality/rung50_from_28/ 2>/dev/null || true

rung () {  # tag corpus init seed
  setsid nohup python -u probes/constructor_curriculum.py --stage corpus \
    --corpus "runs/quality/$2" --cells fill50-a2.0 --features local --support wide \
    --train 12 --heldout 6 --manifest-split --init "runs/quality/$3" --seed "$4" \
    --iterations 200 --episodes 4 --instances-per-iteration 2 --eval-episodes 3 --eval-every 2 \
    --eval-sets heldout --max-steps 280 --episode-seconds 100 --train-episode-seconds 50 \
    --stop-bias -6 --prefix-fraction 0.9:0.0 --prefix-schedule mastery --prefix-empty-mix 0.2 \
    --out "runs/quality/$1.pt" > "runs/quality/$1.log" 2>&1 < /dev/null &
}
rung r39_p3_s0 pegasus3_f50 r30_p3_s0.best.pt 0
rung r39_p3_s1 pegasus3_f50 r30_p3_s1.best.pt 1
rung r39_z2_s0 zephyr2_f50  r30_z2_s0.best.pt 0
rung r39_z2_s1 zephyr2_f50  r30_z2_s1.best.pt 1
sleep 25
for f in runs/quality/r39_*.log; do
  printf "%-12s %s err | %s\n" "$(basename "$f" .log)" "$(grep -c Traceback "$f")" \
    "$(grep -o '"rate": [0-9.]*' "$f" | tr '\n' ' ')"
done
