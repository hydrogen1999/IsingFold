# The comparison Track B has never run: the learned embedder against stock minorminer, on the
# held-out lineages, through the same evaluator, at a matched number of attempts and with wall
# clock reported. The second minorminer arm spends roughly the same time as a policy arm.
set -u
cd ~/isingfold_ladder3
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$HOME/isingfold_ladder3/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=2
python -u probes/external_baseline.py --lineages 60 --k 8 --mm-tries 10 --mm-k-extra 40 \
  > runs/external_baseline.log 2>&1
