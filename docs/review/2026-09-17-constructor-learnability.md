# Constructor learnability: start with a small RL experiment that actually learns

## Conclusion

A simple RL policy can learn to construct a genuine minor embedding in the existing
empty-start environment. On K3 embedded into C5, a zero-initialized linear policy with
16 parameters and REINFORCE with a leave-one-out baseline improves valid-COMMIT rate
from 31/100 to 93/100,92/100,93/100 across three training seeds after 40 updates.

This is a same-instance feasibility overfit gate. It is deliberately small and symmetric;
it mainly establishes that useful action sequencing and terminal choice can be learned.
It does not establish spatial reasoning on difficult packing instances, generalization,
annealing-quality improvement, or competitiveness with a classical embedder.

The earlier 167 unit tests established software contracts; the earlier two-action synthetic
reward test established policy-gradient plumbing. Neither was this actual embedding
learnability gate. That gap should have been closed before launching larger runs.

## What the latest commit actually reports

Main commit `eca14e0abea447b003450a6bf66e60f2fadf18ad` changes STATUS, not the independent
constructor implementation. The running implementation is PR 2 head `d06bf43` in a
separate experiment-host checkout. The board reports:

- At iteration 9, all six runs have zero valid outputs on 30 held-out tasks each.
- At iteration 19, training and validation remain at zero validity, with no valid COMMIT
  in the reported 320 training episodes.
- Normalized entropy prints 1.00000; reported initial candidate logit spread is about 0.007.
- The training reward is about 0.04 with advantage standard deviation 0.02--0.03.
- A step takes about 0.5--0.8s; longer episodes and larger-learning-rate runs were added.

The corresponding new constructor raw logs are not committed under results. These
numbers are therefore board-reported observations, not independently re-extracted
learning curves. Nineteen optimizer updates without success are a warning about this
training setting, not evidence that RL cannot learn embedding.

## Three problems worth separating

### 1. Training starts beyond the demonstrated learnable regime

The phase 1 reward is 1 for a valid COMMIT and 0.25 times final placement/demand progress
otherwise. Thus it is not an all-zero reward problem, but a run without a successful
trajectory never sees the large completion bonus. Rewarding incomplete progress cannot
by itself tell the policy which early choices permit eventual completion.

Thirty seconds at the reported step cost leaves roughly 37--60 decisions. The observation
reports a 1000-step decision horizon, not the actual remaining wall-clock time. This
restricts exploration and creates a missing-budget observation. It does not mathematically
prove that a 20-variable task is impossible: 20 placements plus routing may or may not fit.
Use a generous watchdog with a fixed decision horizon for the initial learning gate;
measure real elapsed time for every benchmark. A later wall-clock-limited policy should
observe that budget explicitly.

### 2. A uniform candidate policy is not uniform over action families

The proposal generator refills unused family quotas. In a five-step diagnostic on
Pegasus3 (128 qubits) with a cycle20 logical graph, support evolved through
PLACE 12/GROW 10, PLACE 12/GROW 22, PLACE 12/GROW 32, then PLACE 12/GROW 42 with one each of
SHRINK, RESTART and STOP. At the last state a near-uniform candidate scorer therefore
places about 74% of its probability on GROW and 21% on PLACE. The nominal grow quota 8
is not a bound after refill. This can spend the trajectory expanding existing chains
while most logical variables remain unplaced.

This is a confirmed sampling-prior effect, not a claim that GROW is generally undesirable.
Growth is essential both to connectivity and to the downstream-quality objective.
The existing learned constructor also retains adjacency-limited PLACE proposals; the
all-free support of the historical root policy is a different implementation.

An isolated next ablation is a family-balanced policy: choose a family, then a candidate
within that family. Preserve all legal actions. Compare it with the flat scorer at the
same support and update budget before making it the default. A reduced grammar is also
possible in an explicitly labelled early curriculum, but it changes the learning task.

### 3. Most runtime is outside the neural network

A cProfile diagnostic on that five-step Pegasus3 toy took 2.189s:

| Component | Cumulative seconds | Interpretation |
|---|---:|---|
| Constructor 230-channel features |1.274| About 58% of profiled wall time |
| Environment preparation |0.840| Includes the next row |
| Full environment tensor observation |0.515| Built, then ignored by this constructor |
| Actor distribution calls |0.0087| Under 0.4% |
| Proposal generation |0.064| Part of the environment work |

These overlapping cumulative times must not be summed. cProfile perturbs timing;
this is a bottleneck diagnostic on a constructed instance, not a production benchmark.
The observed trace and exact diagnostic source are preserved in
`results/audit/constructor_support_profile.json` and the adjacent `.py` file.
The constructor recomputes its own features after the environment has already built a
full observation for the original model interface. Removing redundant observation work
is a targeted throughput ablation. Increasing neural model size will not solve it.

## What the gradient audit does and does not show

The current actor loss is differentiable and sums trajectory credit; it does not
accidentally detach the actor. At uniform logits,

    d log pi(a) / d z_j = 1[j=a] - 1/K.

Therefore entropy close to 1 does not imply a zero policy gradient. The entropy bonus's
gradient is exactly zero at uniform logits, and small nearby; the observed flat policy
is not evidence that an entropy coefficient 0.01 prevents all learning.

A separate initial-state diagnostic on a four-variable path/grid4x4, using actual
230-channel observations and three initialization seeds, found actor encoder gradient
norms 1.52e-4--1.79e-4, weighted value-loss norms 3.10e-3--5.39e-3, and weighted entropy
norms 3.0e-9--1.74e-8. Shared critic interference is consequently a plausible ablation,
not a demonstrated explanation for the remote runs: these are single-state toy gradients,
not full training-trajectory gradients.

The clean ablation keeps actor/features/support fixed and sets
`--value-baseline loo --entropy-coef 0`. It removes learned-critic interference and the
entropy term while retaining an action-independent baseline. Record actual actor and
critic gradient norms before interpreting an entropy plot.

Potential shaping also cannot manufacture completion signal here. With gamma 1 and zero
terminal potential, returns are R-Phi(s). The potential-shifted LOO baseline subtracts
the same Phi(s), so it cancels exactly from actor advantages. Raising shaping_coef
alone is not a remedy for this Monte Carlo training setting.

## The simple experiment that learned

The task is to embed triangle K3 into the five-node hardware cycle C5. No singleton
placement can realize all three logical edges because C5 contains no triangle. Every
successful output therefore needs a nonsingleton connected branch set. The existing
environment performs transitions and independent COMMIT validation.

The actor is just

    pi(a | s) = softmax(w^T x(s,a)), w in R^16.

Its 16 observations are eight opcode indicators, five observable successor-minus-current
structural statistics (placement, realized edges, occupancy, overlap, disconnected
chains), and three terminal/restart indicators conditioned on progress. These are
candidate features, not witness labels. Occupancy is an observation, not a reward penalty.

Training uses the existing feasibility reward, one fresh batch per update, 16 rollouts
on the same task, a leave-one-out baseline, Adam learning rate 0.03, clipping at 1, no
critic, no entropy bonus, no demonstrations, and no teacher. The full existing legal
macro-action support is retained. COMMIT, STOP and RESTART are sampled by the policy;
there is no forced COMMIT or hidden completion routine.

| Training seed | Before /100 evaluation episodes | After 40 updates /100 |
|---|---:|---:|
|0|31|93|
|1|31|92|
|2|31|93|

All actors start at exactly zero weights, hence the common initial result with identical
evaluation RNG seeds. Evaluation uses 100 stochastic episodes, seeds 100000--100099,
on the SAME task, before and after training. These are not 100 held-out instances.
No hyperparameter sweep was used for these three runs. Training performs 640 episodes
per seed. Raw logs and metadata are in `results/audit/constructor_tiny_seed*.log` and
`constructor_tiny_summary.json`. Runtime is hardware-dependent (approximately 40--52s
per complete run here). A minorminer call or witness/initial-embedding/ground-energy
access raises an exception throughout this diagnostic.

Reproduce on the PR branch:

```bash
python probes/constructor_tiny_gate.py --seed 0 --iterations 40 --episodes 16 --eval-episodes 100
python probes/constructor_tiny_gate.py --seed 1 --iterations 40 --episodes 16 --eval-episodes 100
python probes/constructor_tiny_gate.py --seed 2 --iterations 40 --episodes 16 --eval-episodes 100
```

The recorded runs used the preliminary copy of this diagnostic before packaging; the
summary records its SHA256 and the environment source commit. The packaged script
preserves its algorithm, observations, reward, RNG streams and defaults.

## Next gates, in order

1. Keep this real-embedding test as the learning smoke gate. Unit-test counts are not a
   substitute for it.
2. Train a small fixed set of 2--8-variable instances with explicit route and placement
   ambiguity; measure both training-set overfit and independent held-out graphs. Add
   distractor branches/dead ends so a policy that only avoids STOP cannot pass.
3. On the unchanged contextual model, compare value-baseline versus LOO. Separately
   compare flat candidate sampling versus family-balanced sampling. Keep features,
   support, instances and update count fixed within each ablation.
4. Profile and remove duplicated observation work, then remeasure transitions/second
   and valid COMMITs per unit time. Separate time limits from decision-horizon limits.
5. Increase size/difficulty only after success is observed on the previous stage.
   Quality training begins only with a constructor that produces valid outputs.
   Keep the quality objective, public capacity constraints, and empty-start deployment.

The learning algorithm does not need PPO at this stage. REINFORCE with a baseline has
precedent in neural combinatorial optimization (Kool et al., ICLR 2019,
https://arxiv.org/abs/1803.08475; POMO, NeurIPS 2020,
https://arxiv.org/abs/2010.16011). Those papers are methodological precedents, not evidence
that a particular IsingFold instance distribution is already learnable or that this
small gate transfers to Pegasus/Zephyr.
