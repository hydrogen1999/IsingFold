# Target results

> **Nothing in this file is a measurement.** Every number is a target: the value the method must
> reach for the corresponding claim to be made. It exists so the figures, the tables and the
> narrative can be laid out before the numbers arrive, and so that each target is fixed in advance
> rather than chosen after seeing a result.
>
> Measured values live in `RESULTS.md` and every one of them cites the log it came from. Nothing
> here cites a log, because no run produced it. If a number in this file ever appears without the
> word target beside it, it has been copied into the wrong document.

The shape of the claim: the learned constructor wins on **feasibility** and on **solve
probability**, on the **small hosts** where congestion binds and on the **full hardware** where
the problem is size rather than congestion.

---

## 1. The headline table, as it must read

### 1.1 Small hosts, Pegasus 3 and Zephyr 2

Thirty held-out instances a cell, empty starts, a shared 300-second deployment deadline, eight
256-read selection blocks, a fresh 4096-read assessment, minorminer as an anytime arm with the
same deadline.

| cell | arm | feasibility | solve probability | residual |
|---|---|---|---|---|
| fill 0.30, 28 vars | learned | target **≥ 0.97** | target **≥ 0.56** | target **≤ 0.027** |
| fill 0.30, 28 vars | anytime minorminer | measured 1.00 | measured 0.537 | measured 0.0289 |
| fill 0.50, 39 vars | learned | target **≥ 0.95** | target **≥ baseline + 0.03** | target **≤ baseline − 0.010** |
| fill 0.50, 39 vars | anytime minorminer | to be measured | to be measured | to be measured |
| **fill 0.90, 93 vars** | **learned** | target **≥ 0.50** | target **> 0**, any value | target reported |
| **fill 0.90, 93 vars** | **anytime minorminer** | measured **0.00 of 12** | undefined, no embedding | undefined |
| **fill 0.95, 102 vars** | **learned** | target **≥ 0.33** | target **> 0** | target reported |
| **fill 0.95, 102 vars** | **anytime minorminer** | measured **0.00 of 12** | undefined | undefined |

The two congested rows are the ones that carry the paper. There the comparison is not a margin,
it is the difference between a number and no number at all: the router returns nothing under a
five-minute wall clock with four hundred and fifty restarts, so any nonzero coverage is a
capability it does not have. The decision rule registered for them is **six of twelve completed
deployments**, with **one or fewer of twelve** killing the configuration.

### 1.2 Full hardware, Pegasus 16 and Zephyr 15

The locked test lists, evaluated once, at the training budget, on frozen checkpoints.

| list | arm | feasibility overall | 24 vars | 48 vars | 100 vars |
|---|---|---|---|---|---|
| Pegasus 16 test | learned, trained on Pegasus | target **≥ 0.80** | target ≥ 0.95 | target ≥ 0.85 | target **≥ 0.65** |
| Pegasus 16 test | fragment checkpoint | measured 0.38 | 0.70 | 0.33 | 0.33 |
| Zephyr 15 test | learned, trained on Zephyr | target **≥ 0.90** | target ≥ 0.95 | target ≥ 1.00 | target **≥ 0.75** |
| Zephyr 15 test | fragment checkpoint | measured 0.73 | 0.93 | 0.80 | 0.36 |
| Pegasus 16 test | learned, trained on **Zephyr** | target **≥ 0.75** | | | |
| Zephyr 15 test | learned, trained on **Pegasus** | target **≥ 0.85** | | | |

The cross-topology rows are the transfer claim: a policy trained on one lattice loses no more
than **0.05** on the other. That is already close to what the current checkpoints do, at -0.05 and
+0.03, so the target is to keep it while the absolute numbers rise.

---

## 2. Solve probability, the quality half

| cell | comparison | target |
|---|---|---|
| fill 0.30, 28 vars | learned pool minus anytime minorminer, paired | **≥ +0.017 residual**, interval clear of zero |
| fill 0.30, 28 vars | quality reward minus matched feasibility reward | **≥ +0.017**, interval clear of zero |
| fill 0.30, 28 vars | replication | same sign on **three seeds**, on **thirty fresh** lineages |
| fill 0.50, 39 vars | the same three, at the larger cell | same thresholds |
| full hardware, 24 vars | learned pool minus anytime minorminer | **≥ +0.010**, interval clear of zero |

Why 0.017 and not something rounder: the evaluation's own paired noise is 0.023 to 0.045 per
instance, so thirty instances resolve 0.017 and need 166 to 636 to resolve 0.005. The threshold
is the smallest effect this protocol can see, and it is thirty-seven percent of the 0.0268 that
currently separates the policy from the router.

---

## 3. What each figure will show, and which target it carries

| figure | data file | the target it must display |
|---|---|---|
| 1, motivation | `fig_selection_signal_rho.csv` | resource rank correlation well below measurement, on every host |
| 2, feasibility by cell | `fig_learning_curves.csv` | the congested cells rising off zero |
| 3, congestion | `fig_congestion_wallclock.csv` | the router at zero and the learned arm above it, short-chain cells |
| 4, quality, paired | `fig_paired_quality_study.csv` | the quality arm left of the feasibility arm by at least 0.017 |
| 5, mechanism | `fig_chain_breaking.csv` | the learned arm's chains moving toward the router's break rate |
| 6, where quality lives | `fig_branch_points.csv` | the policy's pick rate rising off the chance line |
| 7, regime map | `fig_solvability_frontier.csv` | which cells report solve probability and which report residual |

Figures 1, 3 and 7 can be drawn today: they carry measured data and no target. Figures 2, 4, 5 and
6 have their axes and their data files and are waiting on the numbers.

---

## 4. The distance from here

| quantity | measured today | target | gap |
|---|---|---|---|
| feasibility, fill 0.90, 93 vars | 0.00 | ≥ 0.50 | the whole claim |
| feasibility, Pegasus 16 test list | 0.60 | ≥ 0.80 | +0.20 |
| residual against the router, fill 0.30 | +0.0268 worse | ≥ 0.017 better | 0.044 |
| quality reward against feasibility reward | -0.0033, killed | ≥ +0.017 | 0.020, and the sign |
| the policy's pick rate at a branch point | 0.230 | above 0.250 and rising | at chance |

The last row is the one that gates the rest. A branch decision is worth 0.0491 [0.0467, 0.0515]
and the policy chooses at chance, while a model fitted directly to the measured answers reaches
only 0.290 against 0.250. Until the action representation distinguishes a good branch from a bad
one, the quality rows of this table cannot be filled in by any reward schedule.
