#!/usr/bin/env bash
# The rung between what the F and G checkpoints know and the quality cell.
#
# Those checkpoints learned 12 to 20 variables on 64 to 128 qubit fragments. The quality cell is
# 39 to 50 variables on the whole of Pegasus 3 and Zephyr 2, and the pilot that jumped straight
# there sat at zero. Fill 0.30 on the same two hosts gives 28 and 38 variables with the same
# topology and the same wide support, so only the size changes. minorminer is valid on every
# instance of it, which also makes it a second quality cell rather than only a stepping stone.
set -eu
cd "$HOME/prj_IsingFold_pr2"
source "$HOME/isingfold/.venv/bin/activate"
export ISINGFOLD_SRC="$HOME/prj_IsingFold_pr2/src"
export PYTHONPATH="$ISINGFOLD_SRC"
export OMP_NUM_THREADS=1
export ISINGFOLD_FAST_INTERNAL_ASSERTS=1
mkdir -p runs/quality/jumped_to_f50
mv runs/quality/qw_*.log runs/quality/jumped_to_f50/ 2>/dev/null || true

rung () {  # tag corpus init seed
  setsid nohup python -u probes/constructor_curriculum.py --stage corpus \
    --corpus "runs/quality/$2" --cells fill30-a3.0 --features local --support wide \
    --train 20 --heldout 4 --manifest-split --init "runs/curriculum/$3" --seed "$4" \
    --iterations 200 --episodes 4 --instances-per-iteration 2 --eval-episodes 3 --eval-every 2 \
    --eval-sets heldout --max-steps 250 --episode-seconds 90 --train-episode-seconds 45 \
    --stop-bias -6 --prefix-fraction 0.9:0.0 --prefix-schedule mastery --prefix-empty-mix 0.2 \
    --out "runs/quality/$1.pt" > "runs/quality/$1.log" 2>&1 < /dev/null &
}
rung r30_p3_s0 pegasus3_f30 F_local_s0.pt 0
rung r30_p3_s1 pegasus3_f30 F_local_s0.pt 1
rung r30_z2_s0 zephyr2_f30  G_local_s0.pt 0
rung r30_z2_s1 zephyr2_f30  G_local_s0.pt 1
sleep 20
for f in runs/quality/r30_*.log; do
  printf "%-12s %s err, %s bytes\n" "$(basename "$f" .log)" "$(grep -c Traceback "$f")" "$(wc -c < "$f")"
done
