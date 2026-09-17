# Critical review, 2026-09-17: the two open bottlenecks and the strongest A* story

Scope: bottleneck 1 (validity at 80 to 95 percent fill on Pegasus 6 and Zephyr 4, where
minorminer is 0) and bottleneck 2 (a learned quality component beyond spend-and-measure).
Written against the ladder results on `main` up to `480d08a` and the probe on branch
`feat/constructor-curriculum`. The project rule applies throughout: no direction is killed
on one run until bugs, scale, hyperparameters and power are ruled out.

## Bottleneck 1: why a policy that learns every rung to 20 variables can still read 0 at 300

Five candidate causes, in the order I would rank them after reading the code.

**1. The support may not contain the answer.** Every step offers at most 64 state-changing
candidates: 16 PLACE, 16 ROUTE, 8 GROW, 4 SHRINK, 8 REWRITE, 8 REPAIR, 4 RESTART
(`CONSTRUCTION_QUOTAS`). The PLACE shortlist is the generator's heuristic, constrained-first
around placed neighbours. Task 12 established that with the witness as the generator's
preference the environment builds a valid embedding on 48 of 48 fill instances, and with the
unhinted order it builds none. That result was read as "roots are the information", but it
also says something sharper: the unhinted shortlist may simply not contain the qubit the
witness would use. If at 90 percent fill the witness's next placement is inside the 16
offered candidates only a third of the time, no policy over this support can reach a valid
COMMIT, whatever its features or training. This is the first thing to measure and it costs
minutes: replay the witness through the construction API with the hint off and record, at
each PLACE and ROUTE step, whether the witness's action is in the generated batch
(coverage against progress, per fill cell). Coverage above 0.8 says training can in
principle succeed; coverage below 0.5 says widen the support first (PLACE 40, ROUTE 16,
GROW 6, SHRINK 2 fits the 64 cap; the PR's all-free root support is the other lever).

**2. The 16 channels are placement-blind by construction.** The tiny features are eight
opcode indicators, five global deltas (placed, realised, occupancy, overlap, disconnected)
and three terminal indicators. Two PLACE candidates for the same variable differ only in
their realised-edge delta and occupancy delta. What the linear actor learned on every rung
is therefore a greedy rule: prefer the candidate that realises the most edges now for the
fewest qubits, avoid STOP, COMMIT when complete. That rule generalises across topologies
because it is host-agnostic, and it is exactly the rule that fails at high fill, where the
right placement leaves room for later variables rather than realising an edge now. The
230-channel features carry residual capacity and coupling; they cost 0.74 s a step at
5,640 qubits and 0.13 s at 1,024, and their local part (`_local`, `_residual_capacities`)
is the remaining hot spot. A middle ground exists and is cheap: add to the 16 channels the
delta of the largest free component and of the number of free components, computed by the
adjacency-list walk that already exists in `constructor_features._free_component_sizes`.
That is two channels, host-size independent in cost after the first walk, and it is the
one piece of information a greedy placer needs to stop painting itself into corners.

**3. Horizon and wall time are below what a valid embedding needs.** A 428-variable
instance needs about n placements and about m routes, m being 1.5 to 2 n: 1,000 to 1,300
state-changing actions before a COMMIT is possible. The fill runs use `--max-steps 1500`
and `--episode-seconds 600`. The from-scratch episodes ended after 42 s by STOP, so the
per-step cost with hundreds of chains placed has never been measured; `_prepare` charges
feature work proportional to the number of chains and GROW candidates are drawn from every
chain. If a step costs 0.4 s at 400 chains, 1,300 steps is 520 s, and the deadline truncates
the episode at the last stretch even when every choice was right. Measure the step cost
with a 90 percent witness prefix on the fill corpora (the step-cost probe accepts an
initializer through `episode`), then set the horizon to 4 n and the deadline to cover it.
Until that is done a 0 at fill 90 or 95 is uninterpretable.

**4. Exploration, only for the cold starts.** A zero-weight actor is uniform over the
support, STOP is always offered, so the expected episode length is the support size, about
60 steps, and progress reward is 0.25 times 5 to 8 percent. That is the 0.013 to 0.020
reward in the scratch logs. It is not a verdict; it is the cold-start regime the smaller
rungs left after 20 iterations. The warm-started runs (from `F_linear_s0` and `G_linear_s0`)
already have STOP suppressed and are the runs that count. The cold prefix runs will
converge more slowly for this reason alone and should not be read before iteration 50.

**5. Credit assignment is uniform across a thousand steps.** With the leave-one-out
baseline, every decision in an episode receives the same advantage R minus the mean of the
other episodes' returns. Potential-based shaping does not change that: `constructor_loss`
subtracts the potential from both the return-to-go and the baseline, so the per-step
advantage is again R minus the others' mean (this is by design and correct, but it means
`--shaping-coef` is not a lever here). Over 1,000 steps the gradient is the sum of 1,000
score terms times one scalar; its variance grows with length while the signal does not.
Two remedies are cheap. First, a per-instance value baseline that depends on progress
(the contextual actor's critic exists; at lr 3e-3 it did not collapse and matched the
linear actor). Second, and simpler, an n-step truncation: credit each action with the
progress made in the next 50 steps rather than the whole episode. That changes the
objective toward greedy progress, which is a risk at high fill, so I rank it below the
value baseline.

### Approaches for bottleneck 1, with deciding experiments

A1. Support coverage by witness replay (diagnostic, apollo, under an hour). Instances: all
48 fill instances per host. Metric: fraction of witness PLACE and ROUTE actions present in
the unhinted batch, by fill cell and by progress decile. Success: coverage above 0.8 at
every fill. Failure: rebalance quotas toward PLACE and ROUTE and rerun; if coverage stays
low, the PR's all-free support must be ported to the constructor before any training.
Risk: none; this decides what the training runs can mean.

A2. Step cost and horizon at fill (diagnostic, apollo, under an hour). Step cost against
number of placed chains at 680 qubits with the 16-channel actor; then relaunch the
warm-started prefix runs with horizon 4 n and a deadline that covers it. Success: episodes
that end by COMMIT or by NO_ACTION rather than by deadline.

A3. Two free-space channels on the 16-channel actor (code, one day; then the warm-started
runs again). Mechanism above. Deciding experiment: stage b and the fragment rungs, three
seeds, then fill 80 warm-started. Success at the small rungs: no loss of held-out validity;
at fill 80: any held-out validity above 0 by iteration 50 where the 16-channel run has 0.
Risk: the linear actor may not exploit the channels; a small MLP is the fallback.

A4. Completion from a partial embedding as the deployed task (the user's test case 1). The
partial must not come from the witness. Two defensible sources: the constructor's own
greedy rollout truncated where it stalls, and minorminer on a random 90 percent subset of
the variables, which succeeds at that occupancy. The task is then "complete the remaining
variables in the remaining space", and the control is minorminer with the partial as fixed
chains under the same deadline. Deciding experiment: fill 90 on both hosts, 12 held-out
partials, policy completion rate against minorminer-fixed-chains completion rate. Success:
the policy completes where the router does not, with a paired difference whose interval
excludes zero. Risk: minorminer with fixed chains may complete easily at 90 percent (the
roots experiment says witness roots complete 0.83 to 1.00 at 90 percent), in which case
the deployed task is not where the router fails; run the control first.

A5. Behaviour cloning of the witness action sequence, then RL. Task 12 already produces the
teacher trajectories. The loss is the PR's `witness_prefix_loss` idea applied to every
construction step: maximise the log probability of the candidate that matches the witness's
next chain. Deciding experiment: fill 80, 8 train instances, 20 epochs, held-out validity
from empty; success is anything above 0 where the RL-only run has 0. Risk: with the 16
channels the teacher's choices are often indistinguishable (two placements with equal
deltas), so this needs A3's channels or the 230 channels; do A3 first.

A6. Best-of-K within a deadline as the platform's feasibility number. The evaluation
samples 2 to 5 episodes and reports the per-episode rate; the deployed constructor would
sample until the deadline and return the first valid embedding, exactly as minorminer
restarts. Report both numbers. No training needed; a per-episode rate of 0.1 becomes a
deployed rate of 0.65 at K = 10.

## Bottleneck 2: null of the method or null of the experiment

Three reasons to believe it is the experiment.

**Signal to noise of the quality reward.** The reward is 1 minus half the residual over its
Ising bound B, B = 2 S over the ground energy magnitude; on these problems B is about 2.4,
so a residual difference of 0.004 between two valid embeddings of the same instance is a
reward difference of about 0.0008. The residual is a 256-read mean; its standard error at
these sizes is of order 0.005 to 0.01, a reward noise of 0.001 to 0.002. The leave-one-out
advantage over six episodes of one instance is therefore mostly measurement noise, and the
policy gradient sums that noise over 20 to 60 decisions. Fifty to a hundred iterations
moving nothing is what this predicts. This is measurable in an hour: for one held-out
instance and one fixed embedding, the residual standard deviation across twenty seeds at
256 reads, against the standard deviation across the policy's own valid embeddings of that
instance. If noise is at or above signal, the reward is uninformative at 256 reads.

**The effect size in the rung is tiny.** Both arms sit at residual 0.02 to 0.03 on 4 to 8
variable problems; the ground state is found by almost any embedding. The regime where
embedding quality moves the objective is the modern corpora: 16 to 20 variables on
Pegasus 6 and Zephyr 4, p_solve 0.4 to 0.6, where measured best-of-8 selection gains
+0.15 p_solve and even the fewest-qubits rule gains +0.04. The quality experiments have
not been run there. The 16-channel actor is host-size independent (0.045 s a step at 512
qubits) and the 12 to 20 variable checkpoints exist, so nothing blocks it.

**The control is one point on a curve.** Two random growth qubits on minorminer's draw is
one operating point; the platform's own best is growth of 1 to 4 qubits with several
variants selected by measurement, and a heuristic that grows the chain of the strongest
coupled or highest degree variable is the natural next control. If the learned component
beats random growth but not heuristic growth, the paper reports what the policy learned
(a heuristic) rather than learning as a contribution, which is still a result.

### Approaches for bottleneck 2, with deciding experiments

B1. Reward signal-to-noise diagnostic (apollo, an hour). As above. If noise dominates:
1,024 reward reads and 12 episodes per instance, or the pairwise variant B4.

B2. The quality rung on the modern corpora (apollo, several hours, the experiment that
connects the ladder to the headline table). Held-out instances from the validation split of
`runs/modern/pegasus6` and `zephyr4` (30 each), the F and G 16-channel checkpoints and the
230-channel checkpoints from the size rung, arms: constructor best of 6, minorminer best
of 6, minorminer plus growth of 1 to 4 selected by measurement, heuristic growth; the
selection-rules protocol; p_solve as the score. Success: constructor minus the grown
control at or above +0.03 p_solve with the interval above zero on both hosts. Failure is
still a table the paper needs.

B3. Matched-spend controls (part of B2): growth of exactly the constructor's extra qubits
per instance, so the comparison is at equal resource; and heuristic growth. Cost: none
beyond B2.

B4. Pairwise preference reward (code, a day). Two valid embeddings of the same instance
from the same policy, measured with the same read budget; the better one gets +1, the other
-1; REINFORCE over pairs. A rank signal is far more robust to a noisy scalar than the
affine utility, and it is what the platform's selection already computes. Deciding
experiment: B2's protocol after 100 iterations on the modern corpus.

B5. Where a learned component is structurally necessary: at 85 percent and above no router
draw exists to grow from, so the growth control does not exist there. If bottleneck 1 is
solved, bottleneck 2 is answered in that regime by construction, and the quality question
becomes "does the constructor's embedding, selected by measurement, reach the witness's
residual" (the witness is the only other valid embedding). That is the paper's oral table if
it exists.

## The strongest defensible story today, and the one result that changes the tier

Established: the objective is measurable and not predictable from resource counts;
measured selection captures 90 percent of the best-of-8 ceiling on the target hardware and
the fewest-qubit rule a quarter; random growth of two qubits on the standard router's draw
cuts its residual threefold, so resource-first optimisation is the wrong objective; the
planted-fill benchmark and its regime tables show the standard tool returning nothing above
80 percent on short chains while the right roots complete every cell to 85 percent; and an
RL constructor built from a 16-weight linear actor learns to construct valid embeddings from
empty and generalises to unseen instances across Pegasus and Zephyr fragments up to 20
variables, with the honest control that its quality gain beyond spend-and-measure is
small. That is a solid ML-for-combinatorial-optimisation paper with a benchmark artifact,
and a weak oral, because the learned part does not yet do anything the platform cannot do
without it.

The single result that changes the tier is non-zero held-out validity at 85 percent fill or
above on Pegasus 6 or Zephyr 4, evaluated from empty or from a non-witness partial, where
the standard router is 0. Second is B2 with the constructor beating the grown control by
0.03 p_solve on the modern corpora. Either turns "learns and generalises" into "does what
the standard tool cannot".

## Ranking by information per compute-hour and execution order

1. A1 support coverage (minutes; decides whether any training at fill can succeed).
2. A2 step cost at fill and the horizon fix (minutes; decides whether episodes can finish).
3. B1 reward signal-to-noise (an hour; decides whether the quality reward means anything).
4. B2 quality on the modern corpora with three arms (hours; the headline table either way).
5. A6 best-of-K feasibility reporting (no compute; the deployed number).
6. A4 completion from a non-witness partial with the fixed-chains control (hours).
7. A3 two free-space channels, then the warm-started fill runs again (a day).
8. A5 behaviour cloning after A3 (a day).
9. B4 pairwise reward (a day).

apollo (32 cores, shared): A1, A2 and B1 now on two cores; B2 on four cores as soon as B1
reports; the warm-started prefix runs already running continue untouched and are read at
iteration 50, not before. goose (Slurm, two jobs): the fourteen-run batch continues; when
the horizon fix from A2 is known, the next batch is the fill runs with horizon 4 n and the
A4 completion runs, twelve to sixteen cores in one allocation. Nothing on either host is
killed on the strength of the current zeros.
