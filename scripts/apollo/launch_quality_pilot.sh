#!/usr/bin/env bash
# Pilot loop at the quality cell, on the wide support.
#
# The registered 64-candidate support was adopted here on the strength of a hinted replay, which
# only shows that the witness's own moves are inside the offered set. The unhinted replay, which
# is the order a policy actually meets, gets stuck on 6 of 6 instances with as little as nine
# percent of the variables placed, and reports states with no legal PLACE at all while variables
# remain. The wide support reaches a valid COMMIT on 6 of 6 in the same 58 decisions, at 13.9 s
# against 4.0 s, offering 79 to 149 PLACE candidates a step instead of 25 to 33. The deadline
# below is four times that trajectory.
set -eu
cd "$HOME/prj_IsingFold_pr2"
source "$HOME/isingfold/.venv/bin/activate"
export ISINGFOLD_SRC="$HOME/prj_IsingFold_pr2/src"
export PYTHONPATH="$ISINGFOLD_SRC"
export OMP_NUM_THREADS=1
export ISINGFOLD_FAST_INTERNAL_ASSERTS=1
mkdir -p runs/quality/narrow_support
mv runs/quality/qf_*.log runs/quality/narrow_support/ 2>/dev/null || true

feas () {  # tag corpus init seed
  setsid nohup python -u probes/constructor_curriculum.py --stage corpus \
    --corpus "runs/quality/$2" --cells fill50-a3.0 --features local --support wide \
    --train 12 --heldout 6 --manifest-split --init "runs/curriculum/$3" --seed "$4" \
    --iterations 200 --episodes 4 --instances-per-iteration 2 --eval-episodes 3 --eval-every 1 \
    --eval-sets heldout --max-steps 300 --episode-seconds 120 --train-episode-seconds 60 \
    --stop-bias -6 --prefix-fraction 0.9:0.0 --prefix-schedule mastery --prefix-empty-mix 0.2 \
    --out "runs/quality/$1.pt" > "runs/quality/$1.log" 2>&1 < /dev/null &
}
feas qw_p3_s0 pegasus3_f50 F_local_s0.pt 0
feas qw_p3_s1 pegasus3_f50 F_local_s0.pt 1
feas qw_z2_s0 zephyr2_f50  G_local_s0.pt 0
feas qw_z2_s1 zephyr2_f50  G_local_s0.pt 1
sleep 20
for f in runs/quality/qw_*.log; do
  printf "%-12s %s err, %s bytes\n" "$(basename "$f" .log)" "$(grep -c Traceback "$f")" "$(wc -c < "$f")"
done
