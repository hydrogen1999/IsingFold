# Can a plain supervised scorer pick a good embedding without measuring it? If it can, the
# project's contribution is learned selection and the sequential controller is unnecessary. If
# it cannot, then operational quality is not a function of the structural features anyone would
# think to write down, which is the same wall the RL critic hit, and that is worth saying.
set -u
cd ~/isingfold_ladder3
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$HOME/isingfold_ladder3/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=2
python -u probes/scorer_baseline.py --train-lineages 200 --test-lineages 60 --train-draws 6 \
  --ladder 2,4,8,16 > runs/scorer_baseline.log 2>&1
echo "SCORER EXIT $?" >> runs/scorer_baseline.log
