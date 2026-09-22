# Experiments: the tables and figures, and what feeds each one

Design document. Every entry names the claim it carries, the numbers it holds, the JSON file
under `results/` that holds them, and the reviewer objection it is there to close. The JSON
files carry numbers only; this file is the map.

Budget: five figures and five tables in the main paper, the rest in the appendix. Table and
figure identifiers are the same in this file, in the JSON filenames and in the paper, so `T7` is
always `t7_branch_decomposition.json`. The numbers are not consecutive and that is deliberate:
renumbering them when a table moves between the main paper and the appendix is how a citation
ends up pointing at the wrong table. Regenerate every JSON with
`python3 probes/export_paper_results.py`.

Protocol common to all of it: classical simulated annealing, beta range 0.1 to 2.0, 200 sweeps,
selection and assessment always on disjoint read blocks. No QPU result exists and none is claimed.

---

## Main paper

### F1. Teaser: the same instance, two embeddings

Two drawings side by side, one chosen by fewest qubits and one chosen by a 256-read measurement,
with a metric strip underneath: qubits, mean chain, longest chain, broken fraction, residual,
solve probability. Both are valid embeddings of one instance from the router's own eight draws.

Numbers: `f2_pareto_resource_quality.json`, population `router_draws`, any instance whose
front size is 1. Pick the instance where the fewest-qubit draw and the measured-best draw differ
most in solve probability.

Closes: "you are comparing a good method against a bad one." Both panels are the same router.

### F2. The resource-quality Pareto

**This is the figure the tradeoff assumption dies in.** Two panels, shared axes, both in the same
units: x is the extra qubits an option costs over the cheapest option for that same instance, y is
the solve probability that buys relative to the instance mean. Under a resource-quality tradeoff
both panels rise to the right. Neither does.

| panel | how qubits move | band | points | reading |
|---|---|---|---|---|
| A | one router, eight draws | 2.35 qubits | 480 over 60 instances | **rho -0.239 [-0.334, -0.143]** |
| B | one instance, three constructions | 32.5 qubits | 348 over 116 instances | the expensive construction is the worse one |

Panel B arms, paired over the same 116 instances:

| arm | qubits | solve probability | residual |
|---|---|---|---|
| router | 31.3 | 0.402 [0.365, 0.440] | 0.0653 |
| learned policy | 63.8 | 0.228 [0.193, 0.265] | 0.1065 |
| policy, greedily pruned | 45.6 | 0.218 [0.181, 0.260] | 0.1106 |

| paired difference | qubits | solve probability |
|---|---|---|
| policy minus router | +32.5 [+29.3, +35.7] | **-0.174 [-0.202, -0.147]** |
| pruned minus policy | -18.3 [-20.8, -15.8] | **-0.009 [-0.025, +0.006]** |

The reading, in one line: in the narrow band more qubits goes with slightly worse quality, and in
the wide band the expensive construction is the worse one. The measured points are not on a front.
They are interior, and the proof is the arrow drawn in panel B: twenty-nine percent of the policy's
qubits can be deleted for a quality change whose interval contains zero. A front you can step off
for free is not a front.

**A third population is measured and deliberately not drawn.** Absorbing free qubits into random
chains, a 13.3-qubit band over 192 embeddings, gives rho +0.028 [-0.163, +0.221] on solve
probability. That interval contains zero, but so would an interval from a pool where nothing was
measurable: the between-embedding spread there is 0.0101 against a standard error of 0.0082 on each
embedding's own mean, a ratio of 1.24, and for residual 0.0047 against 0.0094, a ratio of 0.50. At
eight blocks of 256 reads those embeddings are not distinguishable from each other, so the null
bounds the effect below a detection floor rather than showing the effect is zero. The bound and the
ratio are in the JSON under `resolution`. Panel B carries the same claim with a floor that is
actually measured: 18.3 qubits move solve probability by less than 0.025.

Numbers: `f2_pareto_resource_quality.json`. It holds all 1020 measurements, each with host, task,
qubits, quality, standard error where one exists, and whether it sits on its instance's front.
Per-instance non-dominated sets average 1.73 of eight draws in panel A. Under a real tradeoff they
would be long.

Rendered by `probes/plot_pareto.py` to `docs/paper/figures/f2_pareto_resource_quality.pdf`.

Closes: "resource-first embedding is a reasonable proxy, you just need a better heuristic."

### T1. What a selection rule is worth

Within-instance Spearman against a disjoint assessment block, four host and topology
combinations, 30 instances a host, 8 draws an instance.

Rows: fewest qubits, one 256-read measurement. Plus the utility version: solve probability
gained over the first draw for measured, fewest qubits, and an oracle over the same draws.

Numbers: `t1_selection_signal.json`. Recomputed from `data/figures/fig_selection_signal_rho.csv`.

Closes: "qubit count is a fine proxy." It orders above chance and at a quarter to a third of the
strength of a short measurement, and on one host its interval crosses zero.

### T2 and F3. What decides quality instead

T2 has two blocks. First, predictors of residual across nine embeddings of each of twenty
instances: broken-chain fraction 0.728 [0.568, 0.888] against qubit count 0.439 [0.303, 0.576].
Second, predictors of a chain's break rate holding length fixed, 639 multi-qubit chains over 16
instances, singletons excluded because a one-qubit chain cannot break and they are 52.6 percent
of all chains.

F3 is the scatter behind the second block: break rate against internal redundancy within one
instance-and-length group, one panel per length, with the fitted within-group correlation.

Numbers: `t2_quality_mechanism.json`. Recomputed from `data/figures/fig_chain_breaking.csv`.

Closes: "you have shown a negative result and nothing else." The positive mechanism is here, and
it is actionable: at a fixed length a chain with a cycle in it breaks less than a path.

### T4. Feasibility from an empty host

The ladder. Held-out unseen instances, evaluation always from empty, minorminer forbidden after
generation. Six rungs from 24 variables on a full 5,640-qubit Pegasus 16 to the congested
93 to 116 variable cells where both the router and the constructor are at zero.

Numbers: `t4_feasibility_ladder.json`.

Closes: "the learned method only works on toys." It reaches 1.00 and 0.95 on full hardware, and
the table is honest about the cell where it reaches 0.00.

### T6. The quality objective, against its matched control

Three arms from one checkpoint, the same forty updates, the same thirty held-out instances,
differing only in the reward. Plus the paired differences and the registered kill criterion.

The deciding number, feasibility minus quality, is -0.0033 [-0.0084, +0.0018] against a
registered upper bound of 0.002. The configuration is dead. What survives is the control:
training improves what the constructor builds, +0.0094 [+0.0025, +0.0163].

Numbers: `t6_quality_arms.json`. Recomputed from `data/figures/fig_paired_quality_study.csv`.

Closes: "you reported a win without a control." The control is the row that killed the headline.

### T7 and F4. Why the quality reward cannot work as posed

T7: from one construction prefix, four competing legal actions, each finished under matched
randomness and measured. 518 branch points over 102 instances. The residual spread across four
actions at one branch point is 0.0491 [0.0468, 0.0516], against a policy-to-router gap of 0.0268.
The policy's preferred action was the best on 0.230 [0.193, 0.266] of points against a chance
rate of 0.250. A model fitted directly to the measured orderings reaches 0.290.

F4: the distribution of that per-branch spread with the whole performance gap drawn as a single
vertical line, so the reader sees one decision outweighing the entire gap.

Numbers: `t7_branch_decomposition.json`. Recomputed from `data/figures/fig_branch_points.csv`.

Closes: "try a different reward schedule." No reward schedule changes which rows the environment
offers, and the rows do not separate the branches.

### F5. Where solve probability is measurable at all

Log solve probability against variable count at three anneal depths, with the fitted slopes
-0.0499, -0.0383 and -0.0457 per variable and the 0.05 crossing at 66, 83 and 80 variables.
Depth does not change the slope. A second panel shows residual still separating embeddings at
274 variables where solve probability is exactly zero.

Numbers: `t15_solvability_decay.json`, 563 measured cells.

Closes: "just anneal harder." Depth moves the intercept and not the slope, and at 94 variables
20000 sweeps gives 0.084 against 200000 sweeps at 0.082.

---

## Appendix

| id | content | file |
|---|---|---|
| A1 | resource interventions: pruning and cloning move resources and not quality | `t3_resource_interventions.json` |
| A2 | transfer across sizes and topologies, locked test lists | `t5_transfer.json` |
| A3 | expressiveness audit: capacity and the physics channels are not the constraint | `t8_expressiveness.json` |
| A4 | congestion under a 300-second wall clock, 20 cells | `t9_congestion_wallclock.json` |
| A5 | named fill against pruned fill and a lower bound | `t10_fill_certification.json` |
| A6 | reward channel calibration, signal to noise at 256 reads | `t11_reward_channel.json` |
| A7 | how large a held-out set has to be, and what it costs | `t12_sample_size.json` |
| A8 | ablations and learning curves, 405 evaluations | `t13_ablations.json` |
| A9 | the candidate support and its step cost | `t14_support_cost.json` |
| A10 | every withdrawn claim and the measurement that withdrew it | `retractions.json` |
| A11 | what is open | `open.json` |

A4 belongs in the appendix but its single row, fill 0.90 with short chains at 0.00 valid after
488 restarts in 300 seconds, should be quoted in the main text as the benchmark's reason to exist.

---

## What the JSON files are

One file per table or figure. Numbers only, no prose. Every value is either recomputed here from
the exported figure data, in which case the file names what it was recomputed from, or carries
`"derivation": "transcribed"` and the log it was transcribed from. Intervals are 95 percent and
bootstrapped over the independent experimental unit, which is the instance lineage.

`MANIFEST.json` lists the protocol, the sign conventions and every file.
