#!/usr/bin/env bash
# Behaviour cloning of the planted witness at the quality rung, as initialisation only.
#
# The placement diagnostic says the constructor's loss is where its chains sit, not how large
# they grow: a third of its qubits are removable and removing them recovers 0.0022 of an 0.080
# residual gap. Terminal reward has to fix a placement sixty decisions before it arrives, and the
# leave-one-out advantage gives every decision the same credit, so supervision of the placement
# itself is the cheapest lever available.
#
# The first attempt used a 250-decision teacher horizon and lost half the teachers to it, with
# stop reasons HORIZON at progress 0.71 to 0.97. The walk needs one ROUTE per logical edge on top
# of one PLACE per variable, so the horizon is raised well past that.
set -eu
cd "$HOME/prj_IsingFold_pr2"
source "$HOME/isingfold/.venv/bin/activate"
export ISINGFOLD_SRC="$HOME/prj_IsingFold_pr2/src"
export PYTHONPATH="$ISINGFOLD_SRC"
export OMP_NUM_THREADS=1
export ISINGFOLD_FAST_INTERNAL_ASSERTS=1
mkdir -p runs/quality/clone_horizon250
mv runs/quality/clone_*_f30.log runs/quality/clone_horizon250/ 2>/dev/null || true

for h in pegasus3 zephyr2; do
  init=F_local_s0.pt
  [ "$h" = zephyr2 ] && init=G_local_s0.pt
  setsid nohup python -u probes/constructor_clone.py \
    --corpus "runs/quality/${h}_f30" --cells fill30-a3.0 \
    --train 20 --heldout 4 --features local --support wide \
    --init "runs/curriculum/$init" --epochs 40 --learning-rate 0.01 \
    --max-steps 900 --teacher-seconds 300 --eval-episodes 3 --eval-every 10 \
    --episode-seconds 120 --stop-bias -6 --out "runs/quality/clone_${h}_f30.pt" \
    > "runs/quality/clone_${h}_f30.log" 2>&1 < /dev/null &
done
sleep 20
for f in runs/quality/clone_*_f30.log; do
  printf "%-24s %s err, horizon %s\n" "$(basename "$f" .log)" "$(grep -c Traceback "$f")" \
    "$(head -1 "$f" | grep -o '"max_steps": [0-9]*')"
done
