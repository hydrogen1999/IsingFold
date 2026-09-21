# Progress report

Learned minor-embedding for quantum annealers. This lists what has been built and what has been
measured. Measured numbers cite their log under `results/`; the full tables are in
`docs/paper/RESULTS.md` and the system description is in `docs/ARCHITECTURE.md`.

Intervals are 95 percent. All sampling is classical simulated annealing under a registered
schedule. There is no QPU result and none is claimed.

---

## 1. Where the project stands

The method is an **independent constructive embedder**: a policy that builds an embedding from an
empty hardware graph, choosing every action itself, with no external router completing or seeding
its output at training or deployment time.

**Working.** It learns to construct. On 24-variable instances on the full Pegasus 16 and Zephyr 15
it reaches held-out validity 1.00 and 0.95 from an empty host. On small hosts it reaches 1.00 at
28 variables across four seeds and 1.00 and 0.94 at 39 variables on two seeds.

**Not working.** It does not yet build *better* embeddings than the classical router. On a
30-instance held-out evaluation it is behind by +0.0359 [+0.0272, +0.0444] in residual energy, and
a quality-directed reward failed its pre-registered test against a matched control.

**The blocking question is now identified and it is architectural**, not a matter of more
training: the information that would tell a good construction decision from a bad one is not in
the observation the policy receives. Section 8 gives the measurement.

---

## 2. Data generation

| item | what was built |
|---|---|
| Instance families | ink-drop, modular, scale-free, dense, sparse, lattice, bipartite, clique |
| Planting | frustrated-loop Ising problems with a **certified ground energy**, so residual energy is exactly measurable rather than estimated |
| Witness embeddings | every instance ships with a valid embedding constructed at generation time, used as a curriculum teacher and never reachable at evaluation |
| Congestion corpora | the host is partitioned into chains first, so the instance is planted **at a chosen occupancy**, from 0.30 to 0.95 of the hardware |
| Chain-shape axis | a power-law parameter controls whether the target embedding has many short chains or fewer long ones |
| Splits | train, validation and test lists locked at generation; held-out instances deliberately carry **no witness** |
| Scale | corpora from 13 to 541 logical variables, on hosts from 40 to 7,440 qubits |

Generators: `probes/gen_corpus.py`, `probes/gen_fill_corpus.py`.

A benchmark result came out of this axis by itself. Under a shared 300-second wall clock with
unlimited restarts, the standard router is valid on **0 of 12** instances at every short-chain
cell from 0.85 occupancy upward on both hosts, while on long-chain cells at the same occupancy it
still succeeds on 0.25 to 1.00. **Chain shape decides embeddability more than occupancy does**,
which was not the project's own prior assumption. `results/control/anytime_*_300.log`.

---

## 3. Model construction: environment and action space

The environment presents, at each step, the set of legal actions as feature rows. Eight opcodes:
place a variable, route a logical edge, grow or shrink a chain by one qubit, rewrite or repair a
group, restart, and commit. An episode counts only if it ends in a commit whose embedding
validates.

Two action-set registrations were built and compared, 64 candidates and 512. **The wide one is
required**, and this was measured rather than assumed: replaying a known-good construction in the
order a policy actually meets, the 64-candidate support reaches a valid commit on 0 of 6 instances
at 0.50 occupancy and 1 of 4 at 0.90, and the 512-candidate support on 6 of 6 and 4 of 4.

Engineering: the per-decision cost at 400 chains was reduced from 4.2 s to 0.16 s through
incremental features, memoised chain connectivity and contact sets, and cached candidate
identities, with equality tests proving identical episodes.

---

## 4. Model construction: policy

The action set has no fixed size, so the policy is a **per-action scorer**: the same weights score
every candidate row and a softmax over the set gives the action distribution.

| class | structure |
|---|---|
| linear | one weight vector, zero-initialised, 16 to 32 parameters |
| MLP | one hidden layer with SiLU |
| contextual | candidate-set actor-critic with Deep Sets pooling and a value head |

Four observation schemas were built and compared: 16 channels, 20 with local capacity, **32 with
physics channels** (contact coverage, redundancy, load concentration, a bridge bottleneck proxy,
cycle redundancy, signed coupling), and 230 with a full description. A zero-padding loader lets a
narrower checkpoint initialise a wider schema without changing any action logit, so representation
can be ablated without confounding it with initialisation.

Ablation result: **four local-capacity channels buy what two hundred and fourteen more channels
buy**, 0.96 against 0.94 final held-out validity, at 0.087 seconds a step against 0.74.

---

## 5. Model training

Monte Carlo REINFORCE with a leave-one-out baseline within an instance. Two objectives:
feasibility, and a measured-quality objective whose reward is the decoded residual energy of the
committed embedding. **Neither reward contains a resource term**; qubit count and chain length are
constraints, never penalties. This is deliberate: the paper's thesis is that resource count is the
wrong objective.

Techniques built for the training problem:

| technique | what it does | status |
|---|---|---|
| Curriculum ladder | climbs instance size rung by rung, each warm-started from the one below | **works**, and is what made the larger cells learnable at all |
| Occupancy prefixes | starts training episodes from part of the witness, annealed away by a mastery gate | works at low occupancy |
| **Reverse-start prefixes** | starts from a state a successful construction actually passed through, k decisions from the end, stepping back as each depth is mastered | new, under test at high occupancy |
| Behaviour cloning | imitates the witness construction with a set-valued loss over all consistent actions | reaches validity 1.00, no quality gain |
| **Conditional quality** | draws until K valid episodes and centres only those, estimating the gradient of quality *given* validity rather than of a mixed utility | new, declared as an objective change |
| Sampling temperature | exposed on the training rollout only; evaluation is fixed so numbers stay comparable | calibrated |

---

## 6. Evaluation protocol

Measurement discipline was built before the comparisons, and several project claims were withdrawn
by it.

- Selection and assessment never share a read block.
- Deployment comparisons give every arm the **same wall-clock deadline**, and the router arm may
  restart for the whole of it.
- The deployed selector uses a **public energy score that never reads the certified optimum**, so
  it is a score a real deployment could compute.
- Held-out instances carry no witness.
- The evaluation's own noise was measured, not assumed: the paired per-instance spread is 0.023 to
  0.045, so 30 instances resolve an effect of 0.017 and 166 to 636 would be needed for 0.005. The
  threshold used is therefore the smallest this protocol can see.

---

## 7. Results by regime

### 7.1 Large hardware, low occupancy

Full Pegasus 16 (5,640 qubits) and Zephyr 15 (7,440), 24-variable instances at about 2 percent
occupancy, where the classical router succeeds in a tenth of a second.

| result | value |
|---|---|
| held-out validity from an empty host | **1.00** and **0.95** |
| locked test list, at the training budget | **0.60** and **0.81** |
| the same checkpoint before hardware-scale training | 0.38 and 0.73 |
| cross-topology transfer, trained on one lattice, tested on the other | -0.05 and **+0.03** |
| size generalisation, trained on 24 variables | 0.33 to 0.60 on 100-variable instances |

Reading: the constructor learns to build at hardware scale, the training is worth +0.22 and +0.08
over the checkpoint it started from, and a policy trained on one topology transfers to the other
at essentially no cost.

### 7.2 Small hosts, high occupancy

Pegasus 3 and Zephyr 2, 93 to 116 variables at 0.90 and 0.95 occupancy, short chains. This is the
regime where the classical router fails outright.

| result | value |
|---|---|
| classical router, 300-second deadline, ~450 restarts | **0 of 12** |
| the constructor, six hours of training | **0.00** |
| known-good construction exists and is reachable in the action set | 4 of 4, 107 decisions |

Reading: the cell is a benchmark nobody currently solves. The construction exists and the action
space contains it; the policy does not find it. The curriculum ladder, which fixed exactly this at
smaller sizes, has not been carried here: it currently reaches 39 variables at 0.50 occupancy.

### 7.3 Quality, where both arms are always valid

Pegasus 3 at 0.30 occupancy, 28 variables, 30 held-out instances, both arms valid on every
instance so the comparison is quality alone.

| arm | residual | solve probability |
|---|---|---|
| frozen policy | 0.0647 | 0.354 |
| **40 updates of the feasibility reward** | **0.0557** | **0.402** |
| 40 updates of the quality reward | 0.0590 | 0.373 |
| classical router | 0.0289 | 0.537 |

Paired: feasibility training improves on the frozen policy by **+0.0094 [+0.0025, +0.0163]**, an
established effect. The quality reward minus the matched feasibility reward is **-0.0033 [-0.0084,
+0.0018]**, which fails the pre-registered threshold and kills that configuration.

**Training the constructor improves the quality of what it builds. Rewarding it for quality does
not improve it further.**

---

## 8. Why the quality objective cannot work as posed

Three measurements, in order, locate the problem.

**A single construction decision is worth more than the whole performance gap.** Branching from
one prefix into four competing legal actions and finishing each with the same policy under
matched randomness, over 518 branch points: the residual spread across the four is **0.0491
[0.0467, 0.0515]**, against the 0.0268 that separates the policy from the router.

**The policy chooses at chance.** It picks the better action 0.230 of the time against a chance
rate of 0.250.

**And the information is not in the observation.** Fitted directly to the measured orderings, with
folds grouped by instance, a small network predicts the better action 0.290 of the time against
0.250. Capacity is ruled out separately: with eight teacher trajectories the network overfits, and
with thirty-nine it gains 0.03 over a linear model. The physics channels are ruled out too: the
32-channel and 20-channel schemas rank construction decisions alike, 0.267 against 0.261.

A terminal reward gives every decision in a trajectory the same credit, so a good final embedding
never identifies the decision that produced it. **The next change is to the action representation,
and no reward schedule substitutes for it.**

---

## 9. Scientific findings that stand independently of the method

These are measured, replicated, and do not depend on the learned policy succeeding.

| finding | evidence |
|---|---|
| **Resource count is a weak quality signal.** Within one instance, qubit count ranks candidate embeddings at 0.15 to 0.27 while a single 256-read measurement ranks them at 0.64 to 0.87 | two corpora, four host-topology combinations |
| **Chain breaking, not size, determines solution quality**, at 0.728 [0.568, 0.888] against 0.439 [0.303, 0.576] | 180 embeddings, within instance |
| **At a fixed chain length, internal cycle redundancy is what keeps a chain from breaking**, at -0.439 [-0.589, -0.289] | 496 chains in 41 groups |
| **Resource interventions move resources and not quality.** Pruning removes 34 percent of the qubits and recovers 0.0022 of an 0.080 gap | two independent interventions |
| **Solve probability is unmeasurable above about 100 variables** and deeper annealing does not extend it: 20,000 sweeps gives 0.084 and 200,000 gives 0.082 | three anneal depths, nine host sizes |
| **Chain shape decides embeddability more than occupancy does** | 300-second wall clock, both hosts |

---

## 10. Open items

1. **Carry the curriculum ladder to the congested cells.** It reaches 39 variables at 0.50
   occupancy and the target is 93 at 0.90. The reverse-start technique is the current attempt and
   has advanced two decisions back along a 110-decision construction.
2. **Change the action representation** so that a good branch is distinguishable from a bad one.
   This is the gate on every quality claim.
3. **Replicate anything that passes** on three seeds and fresh instances before it is reported.
4. **QPU evaluation** remains outside the current evidence.

---

## 11. Practice

Every experiment is recorded with the log that produced it and the number and the log are
committed together. Thirteen claims have been reported and then withdrawn by a later measurement,
each with the measurement that withdrew it, and the list is kept in `docs/paper/RESULTS.md`
rather than in an appendix. Reviews by an adversarial model were commissioned at eight points and
forced several of those withdrawals; they are in `docs/review/`.

Per-point data for every figure is exported to `data/figures/` with a manifest tying each file to
its source log, so any figure traces back to the run that produced it.
