# A corpus the project has never seen, for the final number. The existing split was used as one
# pooled development set by the trainers, so neither half of it is a locked test set any more.
set -u
cd ~/isingfold_ladder3
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$HOME/isingfold_ladder3/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=2
python -u probes/gen_corpus.py --out runs/corpus_c4_locked --host chimera --host-size 4 \
  --variables 16 --chain-size 3 --fault-rate 0.0 --qubit-cap 120 --instances 600 \
  --alpha 0.9 --weights 0.4,1.0,2.5 --clause-length 5 --seed 20260914 \
  > runs/freshcorpus.log 2>&1
echo "FRESHCORPUS EXIT $?" >> runs/freshcorpus.log
