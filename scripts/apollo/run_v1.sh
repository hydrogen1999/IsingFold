# One full run from this checkout: corpus, then one training job per model family.
#
# The corpus seed is here so the digest is checkable. Chimera 4, 16 variables, chain size 3,
# 600 instances, seed 20260914 gives 25570ef89d385f2c.
#
# What this trains is elite imitation, not the actor-critic loss: probes/train_bestof.py keeps
# the best of K rollouts and minimises cross-entropy on their actions. There is no PPO ratio, no
# utility regression and no calibration loss on the quality head, so the utility head of the
# resulting checkpoint is untrained and must not be called a learned critic. An external audit
# made this point and it is correct; the name of the experiment now says what it is.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=3
mkdir -p runs/v1

CORPUS=runs/v1/corpus
if [ ! -s "$CORPUS/manifest.json" ]; then
  python -u probes/gen_corpus.py --out "$CORPUS" --host chimera --host-size 4 \
    --variables 16 --chain-size 3 --fault-rate 0.0 --qubit-cap 120 --instances 600 \
    --alpha 0.9 --weights 0.4,1.0,2.5 --clause-length 5 --seed 20260914 \
    > runs/v1/corpus.log 2>&1
fi
python - <<'PY'
import json
m = json.load(open("runs/v1/corpus/manifest.json"))
print("CORPUS digest", m["digest"], "instances", m["instances"], "split", m["split"])
PY

ROUNDS=${ROUNDS:-60}
pids=""; families=""
for F in if-core if-dual if-mlp; do
  OUT=runs/v1/bestof_$F
  LOG=runs/v1/train_${F}.log
  # A finished run is one whose last logged round is the last round, not one that left a file
  # behind. A crash log is not evidence of success, which is what the previous version of this
  # script treated it as.
  if [ -s "$LOG" ] && python - "$LOG" "$ROUNDS" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{") and "round" in l]
sys.exit(0 if rows and rows[-1]["round"] >= int(sys.argv[2]) - 1 else 1)
PY
  then
    echo "skip $F: $LOG already reaches round $((ROUNDS - 1))"
    continue
  fi
  rm -f "$LOG"
  python -u probes/train_bestof.py --corpus "$CORPUS" --family $F --out "$OUT" \
    --initializer minorminer --mm-tries 10 --rounds "$ROUNDS" --window 10 --lineages 24 --k 8 \
    --epochs 2 --learning-rate 3e-4 --seed "${SEED:-0}" --eval-every 5 --eval-instances 40 \
    > "$LOG" 2>&1 &
  pids="$pids $!"; families="$families $F"
  sleep 5
done

status=0
set -- $families
for pid in $pids; do
  fam=$1; shift
  if wait "$pid"; then
    if [ ! -s "runs/v1/bestof_$fam/policy.pt" ]; then
      echo "FAILED $fam: exited cleanly but wrote no checkpoint"; status=1
    else
      echo "ok $fam"
    fi
  else
    echo "FAILED $fam: exit $?; tail of its log:"; tail -3 "runs/v1/train_${fam}.log"; status=1
  fi
done
[ "$status" -eq 0 ] && echo "V1 TRAINING DONE" || echo "V1 TRAINING INCOMPLETE"
exit "$status"
