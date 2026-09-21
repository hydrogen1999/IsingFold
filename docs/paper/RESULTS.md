# Results

Every number below cites the log it came from, under `results/`. Numbers in square brackets are
95 percent intervals. Where a result was later withdrawn it appears in the retractions section
with the measurement that withdrew it, because this project has withdrawn numbers before and a
list of what currently stands is the only way to tell a finding from a leftover.

All sampling is classical simulated annealing under the registered schedule, beta range 0.1 to
2.0 and 200 sweeps, with selection and assessment on disjoint read blocks. No QPU result exists
and none is claimed.

---

## 1. The thesis: resource count is a weak quality signal

Within one instance, across the candidate embeddings a deployed router actually produces,
correlated against a disjoint assessment block. This is the paper's central claim and it is
measured on two corpora and four host-and-topology combinations.

| signal | Pegasus 6 | Zephyr 4 | Pegasus 3 | Zephyr 2 |
|---|---|---|---|---|
| fewest qubits | 0.265 [+0.141, +0.390] | 0.213 [+0.063, +0.364] | 0.269 [+0.030, +0.508] | 0.148 [-0.058, +0.354] |
| **one 256-read measurement** | **0.814 [+0.723, +0.906]** | **0.872 [+0.804, +0.940]** | **0.821 [+0.719, +0.922]** | **0.639 [+0.442, +0.836]** |

Qubit count orders candidates above chance but at a quarter to a third the strength of a short
measurement, and on Zephyr 2 its interval crosses zero, so there it orders no better than chance.

In utility terms on the modern corpora, over the same eight draws an instance and thirty
instances a host: selecting by measurement gains +0.147 and +0.153 solve probability over the
first draw, selecting by fewest qubits gains +0.042 and +0.041, and an oracle over the same
draws gains +0.163 and +0.163. Measurement takes ninety percent of what is available; the
resource rule takes a quarter.

Why a weak rule still finds embeddings: the eight draws differ by 1.9 to 2.8 qubits, while the
perturbation experiment below needed 32 to 64 absorbed qubits before quality moved measurably.
Inside a router's own output distribution the resource variation is too small to explain the
quality variation.

`results/rules/*.log`, `results/quality/resource_rho_*.log`, `results/signal/ranking.log`,
`probes/signal_ranking.py`.

---

## 2. What actually determines quality

### 2.1 Chain breaking, not resource count

Nine embeddings of each of twenty instances, three checkpoints plus their pruned versions plus
minorminer, each assessed on its own 4096-read block, correlated within instance:

| predictor of residual | rank correlation |
|---|---|
| **broken-chain fraction** | **+0.728 [+0.568, +0.888]** |
| qubit count | +0.439 [+0.303, +0.576] |
| the two predictors with each other | +0.686 [+0.602, +0.769] |

### 2.2 At a fixed chain length, internal redundancy is what holds a chain

Each chain's break rate decoded over 2048 reads, 1347 chains over 16 instances. Singletons are
excluded from every score: a one-qubit chain cannot break and they are 53 percent of the chains,
and leaving them in made length alone look like the better predictor at 0.929 against 0.753.

Within an instance, over multi-qubit chains: articulation points +0.505 [+0.435, +0.575], length
+0.406 [+0.320, +0.492], internal edges +0.362 [+0.271, +0.453], heaviest contact load +0.204
[+0.112, +0.296].

Holding length fixed, inside each instance-and-length group of at least six chains, 496 chains in
41 groups:

| predictor at one length | rank correlation with break rate |
|---|---|
| **internal redundancy, edges beyond a spanning tree** | **-0.439 [-0.589, -0.289]** |
| articulation points | +0.326 [+0.153, +0.499] |
| load on the heaviest contact | +0.182 [+0.070, +0.295] |
| concentration of load | +0.001 [-0.097, +0.098] |
| total coupling mass | +0.017 [-0.116, +0.150] |

A chain containing a cycle breaks substantially less than a path of the same length, and that is
the largest controllable effect measured.

`results/quality/broken_*.log`, `results/quality/breakpred2_p3.log`, `probes/break_predictors.py`.

### 2.3 Resource interventions move resources and not quality

| intervention | qubits | broken fraction | residual |
|---|---|---|---|
| greedy pruning, every contact preserved | -34 percent | -12 percent | +0.0022 recovered of an 0.080 gap |
| cloning the witness construction, 250 epochs | -42 percent | -18 percent | -0.0200 [-0.0428, +0.0028] over twenty instances |

Two independent interventions on the resource axis, both effective on resources, both inert on
quality. The cloning effect reversed sign between an eight-instance sample that suggested it and
a twenty-instance replication.

`results/quality/excess_*.log`, `results/quality/excess20_*_p3.log`, `probes/policy_excess.py`.

---

## 3. What the constructor learns

### 3.1 Feasibility, from an empty host

Held-out, unseen instances, minorminer forbidden after generation, evaluation always from empty.

| cell | variables | fill | init | final |
|---|---|---|---|---|
| ink-drop, full Pegasus 16, 5,640 qubits | 24 | 0.02 | 0.10 | **1.00** |
| ink-drop, full Zephyr 15, 7,440 qubits | 24 | 0.02 | 0.35 | **0.95** |
| Pegasus 3 and Zephyr 2, four seeds | 28 | 0.30 | 0.75 to 1.00 | **1.00 on all four** |
| Pegasus 3, two seeds | 39 | 0.50 | 0.05 and 0.11 | **1.00 and 0.94** |
| Zephyr 2, two seeds | 39 | 0.50 | 0.00 and 0.05 | 0.61 and 0.44 |
| Pegasus 3 and Zephyr 2 | 93 to 116 | 0.90, 0.95 | 0.00 | **0.00** |

`results/inkdrop/ink24_*.log`, `results/quality/r30_*.log`, `results/quality/r39_*.log`,
`results/curriculum/w[23]_f9*.log`.

### 3.2 The locked test lists, at the training budget

Rerun at exactly 400 decisions and 300 seconds, five episodes an instance, on frozen checkpoints.
Each list holds instances at three sizes and the policies saw only 24-variable instances.

| checkpoint | Pegasus 16 test | 24 vars | 48 vars | 100 vars |
|---|---|---|---|---|
| trained on Pegasus 16 | **0.60** | 0.90 | 0.65 | 0.43 |
| trained on Zephyr 15, cross-topology | 0.55 | 0.90 | 0.62 | 0.33 |
| fragment checkpoint, never trained at scale | 0.38 | 0.70 | 0.33 | 0.33 |

| checkpoint | Zephyr 15 test | 24 vars | 48 vars | 100 vars |
|---|---|---|---|---|
| trained on Zephyr 15 | 0.81 | 0.93 | 1.00 | 0.52 |
| trained on Pegasus 16, cross-topology | **0.84** | 0.95 | 0.93 | 0.60 |
| fragment checkpoint, never trained at scale | 0.73 | 0.93 | 0.80 | 0.36 |

Training at hardware scale is worth +0.22 and +0.08 over the fragment checkpoint each run started
from. Cross-topology transfer costs -0.05 one way and +0.03 the other. A policy trained only on
24-variable instances reaches 0.33 to 0.60 on the 100-variable instances.

Both lists were opened once at a looser budget before this rerun, so they are development
diagnostics and not untouched confirmatory sets. `results/transfer/m_*.log`.

### 3.3 Quality, against the matched control

Three arms from the same physics32 checkpoint, the same forty updates, the same thirty held-out
instances, the same 300-second deployment protocol, differing only in the reward.

| arm | residual | solve probability | qubits | longest chain | valid |
|---|---|---|---|---|---|
| frozen | 0.0647 | 0.354 | 67.4 | 6.5 | 0.967 |
| **forty updates of the feasibility reward** | **0.0557** | **0.402** | 68.8 | 6.6 | 1.000 |
| forty updates of the quality reward | 0.0590 | 0.373 | 68.9 | 6.1 | 1.000 |
| minorminer | 0.0289 | 0.537 | 29.9 | 1.7 | 1.000 |

| paired over thirty instances | difference | interval |
|---|---|---|
| **feasibility minus quality, the deciding number** | **-0.0033** | **[-0.0084, +0.0018]** |
| quality minus frozen | +0.0060 | [-0.0025, +0.0145] |
| **feasibility minus frozen** | **+0.0094** | **[+0.0025, +0.0163]** |

The kill criterion, registered before the run, was an upper bound on the deciding number below
0.002. It is 0.0018, so the configuration is dead. Quality was better on ten of thirty instances,
so the sign is wrong and not only the size.

What survives is the control: training the constructor improves the quality of what it builds,
+0.0094 [+0.0025, +0.0163], with the interval clear of zero. Rewarding it for quality does not
improve it further. `results/quality_study/pegasus3_physics_*_s0.log`.

---

## 4. Why the quality reward cannot work as posed

### 4.1 One decision is worth twice the whole gap

From one construction prefix, four competing legal actions spread across the policy's own
ranking, each finished with the same policy under the same continuation randomness, each
completion measured. Forty-seven branch points over twelve instances.

| quantity | 47 branch points | **518 branch points, 102 instances** |
|---|---|---|
| **residual spread across four actions at one branch point** | 0.0572 [0.0438, 0.0706] | **0.0491 [0.0467, 0.0515]** |
| gap between the policy and minorminer | 0.0268 | 0.0268 |
| the policy's preferred action was the best | 14 of 47, 0.298 | **119 of 518, 0.230** |
| chance, with four actions | 0.250 | 0.250 |
| distance from chance | +0.76 standard errors | **-1.07 standard errors** |

A single decision carries nearly twice the entire performance gap. The policy's preference is at
or slightly below chance with respect to which action leads to the better embedding; the 0.298
seen on forty-seven points was noise and the larger sample puts it at 0.230. A terminal reward
gives every decision in the trajectory the same advantage, so a good final embedding never
identifies the choice that produced it.

**And direct supervision on the measured orderings barely helps.** Fitted to the measured best
action at each branch point, five folds grouped by instance:

| model | branch points not fitted | chance | branch points fitted |
|---|---|---|---|
| linear | 0.257 | 0.250 | 0.321 |
| small network | **0.290** | 0.250 | 0.692 |

At 518 records the network reaches 0.290 against a chance rate of 0.250, which is 2.1 standard
errors, a real effect and a tiny one. So the quality at a branch point is large, and the rows the
environment offers do not distinguish which branch carries it. That is a statement about the
representation, and no reward schedule changes it.

`results/quality/branch_p3.log`, `results/quality/branch_all.jsonl`, `probes/branch_compare.py`,
`probes/branch_learnable.py`.

### 4.2 Not capacity, and not the physics channels

Candidate rows cached at each witness decision, the witness-consistent ones marked, a linear
scorer and a small network fitted to the same records under the same ranking loss, split over
training instances because a held-out instance deliberately carries no witness.

| data | schema | model | fitted | unfitted | chance |
|---|---|---|---|---|---|
| 8 paths, 272 decisions | physics32 | linear | 0.276 | 0.278 | 0.031 |
| 8 paths, 272 decisions | physics32 | small network | 0.522 | 0.187 | 0.031 |
| 39 paths, 1293 decisions | physics32 | linear | 0.363 | 0.234 | 0.026 |
| 39 paths, 1293 decisions | physics32 | **small network** | 0.512 | **0.267** | 0.026 |
| 39 paths, 1293 decisions | local20 | linear | 0.354 | 0.249 | 0.026 |
| 39 paths, 1293 decisions | local20 | **small network** | 0.396 | **0.261** | 0.026 |

The observation carries real signal, nine to ten times chance. Capacity is not the binding
constraint: with eight paths the network overfits and with thirty-nine it gains only +0.03. And
physics32 ranks the same as local20, 0.267 against 0.261, so the twelve physics channels are not
carrying the witness's decision.

The witness is one valid solution among many, so a move that is not witness-consistent may still
be a good move and this understates how often the model picks something that works.

`results/quality/express4_p3.log`, `results/quality/express_big*.log`,
`probes/expressiveness_audit.py`.

---

## 5. The benchmark

### 5.1 Congestion under a wall clock

minorminer restarting until a shared 300-second deadline, twelve instances a cell, so the router
may spend the whole budget on restarts rather than a fixed try count.

| cell | Pegasus 3 at 60 s | at 300 s | attempts | Zephyr 2 at 300 s |
|---|---|---|---|---|
| fill 0.70, long chains | 1.00 | 1.00 | 1 | 1.00 |
| fill 0.80, short chains | 0.33 | 0.42 | 345 | 0.08 |
| fill 0.85, long chains | 1.00 | 1.00 | 10 | 1.00 |
| **fill 0.85, short chains** | **0.00** | **0.00** | 479 | **0.00** |
| fill 0.90, long chains | 0.50 | 0.58 | 280 | 0.83 |
| **fill 0.90, short chains** | **0.00** | **0.00** | 488 | **0.00** |
| fill 0.95, long chains | 0.17 | 0.25 | 410 | 0.25 |
| **fill 0.95, short chains** | **0.00** | **0.00** | 453 | **0.00** |

The shape of the target embedding decides this more than its occupancy does. High occupancy with
a short-chain target defeats the router; high occupancy with long chains does not.

`results/control/anytime_*_300.log`.

### 5.2 Where solve probability is measurable at all

Decoded solve probability falls off close to exponentially in the variable count at fixed
congestion, at a rate that did not move across three anneal depths. Fitted only on cells that
produced a resolvable rate, so the unresolved cells are excluded rather than modelled.

| sweeps | slope of log solve probability per variable | variables at 0.05 |
|---|---|---|
| 200 | -0.0499 | 66 |
| 2000 | -0.0383 | 83 |
| 20000 | -0.0457 | 80 |

Depth then stops buying anything: at 94 variables, 20000 sweeps gives 0.084 and 200000 gives
0.082. A local field on up to half the nodes, the one knob that leaves every edge alone, leaves
the rate at zero from 195 variables up.

Residual still separates embeddings where solve probability does not: at fill 0.50 with a
64-qubit perturbation it gives +0.0047 [+0.0022, +0.0073] at 274 variables, where solve
probability is exactly zero.

`results/frontier/*.log`, `probes/solvability_frontier.py`.

### 5.3 Named fill overstates the occupancy an embedding needs

The witness is one embedding at the named fill; greedy pruning yields a smaller one that still
works. Both are sufficient occupancies and neither is a minimum.

| host, named fill | variables | pruned fill | lower bound | minorminer valid at 200 tries |
|---|---|---|---|---|
| Pegasus 2, 0.90 | 30 | 0.84 | 0.75 | 0.58 |
| **Pegasus 3, 0.90** | **93** | **0.86** | **0.72** | **0.00** |
| Pegasus 6, 0.90 | 488 | 0.87 | 0.72 | 0.00 |
| Zephyr 1, 0.95 | 38 | 0.92 | 0.79 | 0.17 |
| **Zephyr 2, 0.90** | **116** | **0.87** | **0.72** | **0.00** |

`results/frontier/prune_*.log`, `probes/witness_prune.py`.

### 5.4 Reward channel calibration

Eight valid embeddings an instance, eight independent 256-read blocks each, the measurement
variance removed from the between-candidate variance, then the pool ranked with one block and the
winner scored on the disjoint ones. Twelve instances a host.

| host | channel | signal to noise at 256 reads | assessed gain from ranking |
|---|---|---|---|
| Pegasus 3 | solve probability | 2.75 | +0.038 |
| Pegasus 3 | **residual** | **19.1** | +0.027 |
| Zephyr 2 | solve probability | 0.73 | +0.007 |
| Zephyr 2 | **residual** | **39.8** | +0.032 |

Residual is a mean over 256 reads; solve probability is the rate of a rare event whose standard
error at that budget exceeds the differences between candidates. Residual trains, solve
probability assesses. `results/quality/reward_*.log`, `probes/reward_channel.py`.

### 5.5 How large a held-out set has to be

The same arm evaluated twice on the same instances gives a paired per-instance spread of 0.0231
and 0.0454, which is the evaluation's own noise. A paired comparison at 95 percent confidence and
80 percent power then needs:

| effect to detect | instances at sd 0.023 | at sd 0.045 | core-hours an evaluation at 24 instances |
|---|---|---|---|
| 0.005 | 166 | 636 | 69 to 265 |
| **0.017** | **15** | **55** | **10 to 20** |

A full study is three arms by three evaluations by three seeds. The 0.005 threshold would cost
1900 to 7000 core-hours of deployment alone, so 0.017 is the measurable bar and it is
thirty-seven percent of the gap to minorminer.

---

## 6. Ablations

Final held-out rate, not gain: arms within a rung start from different checkpoints, and the
local-feature arms begin at 0.42 and 0.54 where the others begin near 0.10.

| rung | observation | seeds | final |
|---|---|---|---|
| Zephyr 2 fragments, 64 to 128 qubits | 20 channels, 16 plus local capacity | 2 | **0.96 [0.931, 0.986]** |
| Zephyr 2 fragments, 64 to 128 | 230 channels | 1 | 0.94 |
| Zephyr 2 fragments, 64 to 128 | 16 channels | 1 | 0.88 |
| Pegasus 3 fragments, 64 to 128 | 20 channels | 2 | 0.83 |
| Pegasus 3 fragments, 64 to 128 | 16 channels | 1 | 0.80 |

At rung b, where three seeds exist for every arm and all start at 0.13, gains are comparable:
230 channels +0.760 [+0.696, +0.823], 16 channels +0.607 [+0.566, +0.648], contextual actor with
a leave-one-out baseline +0.388 [-0.093, +0.868] and with a value baseline +0.257 [-0.112,
+0.626]. Both contextual arms at the registered learning rate cross zero.

Four local-capacity channels buy what two hundred and fourteen more channels buy, at 0.087
seconds a step against 0.74. `results/ablation/feasibility.log`, `probes/ablation_table.py`.

---

## 7. Support and cost

| measurement | value |
|---|---|
| unhinted witness replay, registered 64-candidate support, fill 0.50 | **0 of 6**, states offering no legal placement while variables remain |
| unhinted witness replay, wide 512-candidate support, fill 0.50 | **6 of 6**, 58 decisions, 13.9 s |
| unhinted witness replay, registered support, fill 0.90 | 1 of 4 |
| unhinted witness replay, wide support, fill 0.90 | **4 of 4**, 107 decisions, 80.7 s |
| step cost, wide support, 115 qubits | 0.53 s |

The wide registration is required, and a hinted replay cannot establish that because it succeeds
under both. `results/quality/replay_*.log`, `results/control/replay_p3_f90_*.log`.

---

## 8. Retractions

Every one of these was reported and then withdrawn by a later measurement. They are listed so a
reader can tell what the record currently supports from what it once said.

| claim | what withdrew it |
|---|---|
| solve probability is exactly zero at scale | zero hits in 512 reads bounds the rate below 0.006, it does not establish zero |
| annealing deeper erases the embedding signal | the table compared each embedding's own best chain strength on the block it was scored on; at the registered strength the depth interaction is -0.0005 [-0.0036, +0.0026] |
| residual separates embeddings at the congested cell | twelve instances gave +0.0042 [+0.0008, +0.0076]; a fresh twenty-four gave +0.0004 [-0.0031, +0.0038] |
| the cold-start pair is a curriculum ablation | both logs hold only the initial evaluations and not one training iteration |
| the congested cell runs inside the registered support | that rested on hinted replays; unhinted, the registered support reaches a COMMIT on 0 of 6 |
| cloning closes 46 percent of the quality gap | eight instances gave +0.0366 [-0.0029, +0.0761]; twenty gave -0.0200 [-0.0428, +0.0028] |
| twenty updates of the feasibility reward take 53 percent of the gap | that was a mid-training evaluation on four instances; the same arm ended worse than it started |
| named fill is the certified congestion | pruning shows a fill-0.90 witness needs 0.84 to 0.87 |
| stage P and Z are the full Pegasus 16 and Zephyr 15 | they are 32 to 64 qubit fragments of Pegasus 3 and Zephyr 2 |
| qubit count is not a quality signal | it is, at rank correlation 0.15 to 0.27; measurement is three to four times stronger |

---

## 9. What is open

- The quality objective has failed its pre-registered test, and the branch measurement at 518
  points says why: the quality at a decision is large, 0.0491 [0.0467, 0.0515], and neither the
  policy nor a model fitted directly to the measured orderings can find it from the rows the
  environment offers, 0.290 against 0.250 by chance. The next thing to change is the action
  representation, not the reward.
- The congested cells, 93 to 116 variables at fill 0.90 and 0.95, have a benchmark result and no
  method result. minorminer is valid on 0 of 12 there under a 300-second wall clock and the
  constructor is at 0.00. A reverse-start curriculum, beginning at the last state of a successful
  forward walk and stepping back one decision at a time, is the technique under test; it has
  advanced from the final state to two decisions back so far.
- No QPU result exists. Every quality statement here is about classical simulated annealing under
  the registered schedule.
