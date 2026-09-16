# RL on the constructive embedder, curriculum host: Pegasus 3 (144 qubits) and Zephyr 2 (160),
# fill 70 to 95 percent, short and medium chains, twelve instances a cell. From scratch here;
# the imitation-initialised run follows when the prioritiser exists. Episodes are seconds on
# these hosts, so the loop can run hundreds of iterations in a night.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src OMP_NUM_THREADS=2
mkdir -p runs/small runs/rl
for HS in "pegasus 3" "zephyr 2"; do
  set -- $HS
  [ -s runs/small/$1$2/corpus/instances.jsonl ] || \
    python -u probes/gen_fill_corpus.py --out runs/small/$1$2/corpus --host $1 --host-size $2 \
      --fills 0.70,0.80,0.85,0.90,0.95 --alphas 3.0,2.0 --per-cell 12 --seed 20260916 \
      > runs/small/$1$2_generate.log 2>&1
done
python -u probes/train_constructor_rl.py --corpus runs/small/pegasus3/corpus \
  --out runs/rl/pegasus3_scratch.pt --iterations 300 --instances-per-iteration 4 \
  --episodes-per-instance 2 --deadline 30 --eval-every 10 --seed 0 \
  > runs/rl/pegasus3_scratch.log 2>&1 &
python -u probes/train_constructor_rl.py --corpus runs/small/zephyr2/corpus \
  --out runs/rl/zephyr2_scratch.pt --iterations 300 --instances-per-iteration 4 \
  --episodes-per-instance 2 --deadline 30 --eval-every 10 --seed 0 \
  > runs/rl/zephyr2_scratch.log 2>&1 &
wait
echo "CONSTRUCTOR RL FINISHED"
