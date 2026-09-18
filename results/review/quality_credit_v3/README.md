# Quality-credit integration pilot

This is a small development/integration experiment, not the paper benchmark or a final test.
Four P2 validation instances, six variables each, are reused across three continuation seeds.
The twelve runs share one cloned initialization (its SHA-256 is in every header and summary).
No witness or completion solver is used by the deployed policy. Minorminer is a separate arm.

| Final mean residual (lower is better) | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| Frozen clone | 0.032064 | 0.006185 | 0.005208 | 0.014486 |
| Continued feasibility | 0.032064 | 0.006185 | 0.013021 | 0.017090 |
| Quality RL, LOO | 0.007161 | 0.004069 | 0.006510 | 0.005914 |
| Quality RL, LOO + separate state critic | 0.010742 | 0.006185 | 0.006510 | 0.007813 |
| Minorminer, same selection/assessment protocol | 0.004395 | 0.006510 | 0.005046 | 0.005317 |

Every selected embedding is valid. LOO improves over frozen on two seeds, regresses on one;
it does not establish superiority over minorminer. The optional critic is not better than
LOO here and is retained as an ablation, not promoted as the default. Seed means are not
twelve independent problem observations: there are only four validation lineages and one
shared initialization. No confidence or generalization claim is made from this pilot.

Cloning used eight manifest-training instances and a monotone teacher. All eight teacher
walks finished. Held-out sampled feasibility was already 16/16 at initialization, was 15/16
at epoch 9, and returned to 16/16 after the fixed twenty epochs. This does not demonstrate
that cloning increased feasibility. RL used twenty updates, two instances/update and four
episodes/instance. Training, selection and assessment use 32, 32 and 256 SA reads respectively;
evaluation has a cap of two proposals and a fifteen-second deadline per arm.

Raw logs and corpus records are archived here. The run headers correctly record a dirty source
checkout at `97ebb97`; these are development results from the patch, not executions of that
unmodified commit. The summary records source hashes at archive time; critic-schema loading
validation was tightened after the first run without changing the pilot's newly initialized
critic or rollout/loss. No final-test instances were evaluated. Runs overlapped on one machine,
so elapsed times are not speed benchmarks. Binary checkpoints are excluded; recreate them below.

## Reproduction

From the repository root with its Python dependencies installed:

```bash
export PYTHONPATH=src:probes
python - <<'PY'
from isingfold.rl.data.generate import generate_instances, write_corpus
rows = generate_instances(host_family='pegasus', host_size=2, n_instances=24,
    n_variables=6, chain_size=2, fault_rate=0, seed=1809, max_attempts=30)
assert len(rows) == 24
write_corpus(rows, 'runs/quality_credit_review/corpus', seed=1809)
PY
python probes/constructor_clone.py --corpus runs/quality_credit_review/corpus \
  --train 8 --heldout 4 --seed 0 --features local --epochs 20 \
  --learning-rate 0.01 --max-steps 80 --teacher-seconds 5 --teacher-mode monotone \
  --eval-episodes 4 --eval-every 10 --episode-seconds 5 --support wide --stop-bias -6 \
  --out runs/quality_credit_review/clone.pt
```

For each seed in `0 1 2`, use the following common arguments and each arm's arguments:

```bash
python probes/constructor_curriculum.py --stage corpus \
  --corpus runs/quality_credit_review/corpus --manifest-split --data-seed 0 --seed 0 \
  --train 8 --heldout 4 --features local --init runs/quality_credit_review/clone.pt \
  --evaluation-objective quality --support wide --stop-bias 0 --max-steps 80 \
  --episode-seconds 5 --train-episode-seconds 5 --instances-per-iteration 2 --episodes 4 \
  --learning-rate 0.003 --eval-every 20 --eval-sets heldout --reward-reads 32 \
  --selection-reads 32 --assessment-reads 256 --select-cap 2 --deadline 15 \
  --comparison minorminer --baseline-tries 2 --baseline-router-seconds 1 \
  --objective quality --baseline loo --iterations 20 --out runs/quality_credit_review/quality_loo.pt
```

| Arm | objective | baseline | iterations |
|---|---|---|---:|
| frozen | quality | loo | 0 |
| feasibility | feasibility | loo | 20 |
| quality_loo | quality | loo | 20 |
| quality_loo_value | quality | loo_value | 20 |

Give every arm/seed its own output path. Preserve `stop_bias=0` during continuation: the clone
already carries its learned terminal-action logits. The archived source summary verifies
all twelve initialization hashes agree. Larger registered studies should use
`run_quality_study.py` and its immutable input snapshots, rather than these smoke commands.
