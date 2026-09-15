# The first run with both audits' P0 findings fixed. What differs from v2, in the order the
# findings were raised:
#
#   * the teacher accepts a trajectory on its re-measured gain over its own starting embedding,
#     not on the block that made it the winner;
#   * the K rollouts share a starting embedding but RESTART still reaches the real initializer,
#     so training meets the action deployment will give it;
#   * the in-training curve reads validation only;
#   * the advantage baseline leaves the episode out of its own baseline.
#
# The run identity is written next to the logs, because a launcher that skips on round count
# alone cannot tell one seed from another.
set -eu
cd "$(dirname "$0")/../.."
source ~/isingfold/.venv/bin/activate
export ISINGFOLD_SRC=$PWD/src
export PYTHONPATH=$ISINGFOLD_SRC
export OMP_NUM_THREADS=3
SEED=${SEED:-0}
ROUNDS=${ROUNDS:-60}
CORPUS=${CORPUS:-runs/v1/corpus}
OUTROOT=runs/v3/seed$SEED
mkdir -p "$OUTROOT"

python - "$CORPUS" "$OUTROOT" "$SEED" "$ROUNDS" <<'PY'
import hashlib, json, pathlib, subprocess, sys
corpus, outroot, seed, rounds = sys.argv[1:5]
src = pathlib.Path("src/isingfold/rl")
h = hashlib.sha256()
for f in sorted(src.rglob("*.py")):
    h.update(f.read_bytes())
receipt = {
    "corpus": corpus,
    "corpus_digest": json.loads((pathlib.Path(corpus) / "manifest.json").read_text())["digest"],
    "seed": int(seed),
    "rounds": int(rounds),
    "source_sha256": h.hexdigest()[:16],
    "trainer": "probes/train_bestof.py",
    "teacher_gate": "remeasured_gain > margin",
}
(pathlib.Path(outroot) / "run_identity.json").write_text(json.dumps(receipt, indent=1))
print("RUN IDENTITY " + json.dumps(receipt))
PY

pids=""; families=""
for F in if-core if-dual if-mlp; do
  LOG=$OUTROOT/train_${F}.log
  if [ -s "$LOG" ] && python - "$LOG" "$ROUNDS" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{") and "round" in l]
sys.exit(0 if rows and rows[-1]["round"] >= int(sys.argv[2]) - 1 else 1)
PY
  then
    echo "skip $F: already complete for seed $SEED"; continue
  fi
  rm -f "$LOG"
  python -u probes/train_bestof.py --corpus "$CORPUS" --family $F \
    --out "$OUTROOT/bestof_$F" --initializer minorminer --mm-tries 10 \
    --rounds "$ROUNDS" --window 10 --lineages 24 --k 8 --epochs 2 --learning-rate 3e-4 \
    --seed "$SEED" --dev-split validation --eval-every 5 --eval-instances 40 \
    > "$LOG" 2>&1 &
  pids="$pids $!"; families="$families $F"
  sleep 5
done

status=0
set -- $families
for pid in $pids; do
  fam=$1; shift
  if wait "$pid"; then
    [ -s "$OUTROOT/bestof_$fam/policy.pt" ] || { echo "FAILED $fam: no checkpoint"; status=1; }
  else
    echo "FAILED $fam: exit status nonzero; tail:"; tail -3 "$OUTROOT/train_${fam}.log"; status=1
  fi
done
[ "$status" -eq 0 ] && echo "V3 SEED $SEED DONE" || echo "V3 SEED $SEED INCOMPLETE"
exit "$status"
