#!/usr/bin/env bash
# Pilot loop at the quality cell. Small iterations and an evaluation every one of them, because
# the question is whether this cell learns at all, and a number every few minutes answers that
# faster than a number every eighty. The witness needs about 55 decisions here, roughly six
# seconds, so a thirty second deadline is five times the trajectory rather than a constraint.
set -eu
cd "$HOME/prj_IsingFold_pr2"
source "$HOME/isingfold/.venv/bin/activate"
export ISINGFOLD_SRC="$HOME/prj_IsingFold_pr2/src"
export PYTHONPATH="$ISINGFOLD_SRC"
export OMP_NUM_THREADS=1
export ISINGFOLD_FAST_INTERNAL_ASSERTS=1
mkdir -p runs/quality/slow_loop
mv runs/quality/qfeas_*.log runs/quality/slow_loop/ 2>/dev/null || true

feas () {  # tag corpus init seed
  setsid nohup python -u probes/constructor_curriculum.py --stage corpus \
    --corpus "runs/quality/$2" --cells fill50-a3.0 --features local \
    --train 12 --heldout 6 --manifest-split --init "runs/curriculum/$3" --seed "$4" \
    --iterations 200 --episodes 4 --instances-per-iteration 2 --eval-episodes 3 --eval-every 1 \
    --eval-sets heldout --max-steps 300 --episode-seconds 60 --train-episode-seconds 30 \
    --stop-bias -6 --prefix-fraction 0.9:0.0 --prefix-schedule mastery --prefix-empty-mix 0.2 \
    --out "runs/quality/$1.pt" > "runs/quality/$1.log" 2>&1 < /dev/null &
}
feas qf_p3_s0 pegasus3_f50 F_local_s0.pt 0
feas qf_p3_s1 pegasus3_f50 F_local_s0.pt 1
feas qf_z2_s0 zephyr2_f50  G_local_s0.pt 0
feas qf_z2_s1 zephyr2_f50  G_local_s0.pt 1
sleep 20
for f in runs/quality/qf_*.log; do
  printf "%-12s %s err, %s bytes\n" "$(basename "$f" .log)" "$(grep -c Traceback "$f")" "$(wc -c < "$f")"
done
