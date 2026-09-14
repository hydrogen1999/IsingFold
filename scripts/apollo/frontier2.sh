# The corrected measurement, on the test lineages only. Selection and assessment separated,
# failures counted, iterated improvement reported on its own instead of clipped by a maximum,
# and a noise control that says how large a difference this instrument can invent.
set -u
cd ~/isingfold_ladder3
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$HOME/isingfold_ladder3/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=2
python -u probes/frontier2.py --lineages 60 --split test --ladder 2,4,8,12,16,24,40 \
  --rounds 1,2,4,8 --draws 8 \
  --checkpoints if-core=runs/bestof3_if-core_s0/policy.pt,if-dual=runs/bestof3_if-dual_s0/policy.pt,if-mlp=runs/bestof3_if-mlp_s0/policy.pt \
  > runs/frontier2_test.log 2>&1
echo "FRONTIER2 EXIT $?" >> runs/frontier2_test.log
