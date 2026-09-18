# The paper's claim chain, its tables, and where each number comes from

Written 2026-09-18 from the record in `STATUS.md` and the two pairs of critical reviews
(`docs/review/2026-09-17-*`, `docs/review/2026-09-18-*`). Every cell names the log that
fills it. A cell marked **pending** has an experiment running; a cell marked **missing** has
none and would have to be launched.

## The chain

**C1. Two valid embeddings of the same problem differ in solution quality, and the
difference is large.** Eight independent router draws of the same instance, each assessed on
a fresh 512-read block under the registered schedule: the best differs from the first by
+0.16 p_solve on both hosts (`results/rules/*.log`). Supported.

**C2. Resource count is a weak proxy for that quality.** Picking the draw with the fewest
qubits gains +0.042 and +0.041 p_solve over the first draw, and its interval touches zero;
the within-instance rank correlation between qubits and p_solve is -0.27 and -0.21
(`results/rules/*.log`). Supported.

**C3. Measuring is a strong proxy and it is cheap.** Picking by a 256-read measurement gains
+0.147 and +0.153, which is 90 percent of what an oracle on the assessment block would gain;
64 reads a draw already take 92 percent of it, and the gain grows with the pool
(`results/rules/*_k*_r*.log`). Supported.

**C4. The objective itself changes character with scale, and the benchmark reports both
halves.** Solve probability is 0.022 at 50 variables, 0.003 at 75 and exactly 0 from 250 on,
while the energy residual stays informative at 0.10 to 0.15 (`results/scale/objective_*.log`).
So p_solve is the score below about 100 variables and the residual above it. Supported.

**C5. There is a regime where the standard tool returns nothing.** On planted instances at 80
to 95 percent host occupancy with short chains, minorminer returns no embedding at 20 tries,
at 200 tries, and under 300 to 600 second anytime restarts, on Pegasus 6 and Zephyr 4, while
a valid embedding (the planted witness) exists (`results/fill/*_anytime*.log`,
`*_budget.log`, `results/seeded/*.log`). Supported.

**C6. A learned constructor builds embeddings from empty and generalises to unseen
instances.** A 16-weight linear actor trained by REINFORCE with a leave-one-out baseline,
inside the environment's own macro-action grammar, with no router at train or test time:
held-out valid-COMMIT rate rises from 0.05 to 0.17 up to 0.67 to 0.99 on every rung from 2
to 20 variables, three seeds where run (`results/curriculum/*.log`). Supported.

**C7. What it learns is a construction rule, not a hardware geometry.** Each host's
checkpoint reaches the same held-out rate on both hosts' unseen fragment sets, 0.82 on the
Pegasus set and 0.90 on the Zephyr set, whichever topology trained it
(`results/transfer/*.log`). Supported.

**C8. It scales to the hardware the paper names.** On 24-variable instances of the full
Pegasus 16 (5,640 qubits) and Zephyr 15 (7,440 qubits), evaluated from empty on the corpora'
locked validation lists, held-out validity goes from 0.10 and 0.35 at the fragment
checkpoint to 0.85 and 0.90 (`results/hardware/*.log`). Supported.

**C9. Where the standard tool works, it is still the right tool.** Under a shared wall-clock
deadline with the first valid embedding winning, minorminer covers 12 of 12 held-out modern
instances in 0.1 s while the constructor covers 0.42 to 0.75 at 60 s and 0.92 to 1.00 at
300 s (`results/deploy/modern_*.log`). Supported, and it belongs in the paper.

**C10. In the congested regime the learned constructor returns embeddings where the router
returns none.** **Pending**: twelve runs on the short-chain cells at 80, 85, 90 and 95
percent occupancy, both hosts, locked splits (`goose:runs/curriculum/sfill*`). The number
that decides the paper's tier.

**C11. Among valid embeddings the learned constructor's are better on the objective than
what the platform can build without learning.** Partially supported and currently negative
at scale: at the fragment rung the constructor beats "minorminer plus two random growth
qubits, then measured selection" by -0.008 [-0.015, -0.003] residual at equal qubits on
Zephyr and -0.004 [-0.011, +0.003] on Pegasus (`results/curriculum/{P,Z}q_lin230_s1.log`),
but on the modern corpora the fragment-trained constructor over-spends and loses
(`results/curriculum/modq_*.log`). **Pending**: training with the quality reward in that
regime.

## The tables

| Table | Rows | Columns | Source |
|---|---|---|---|
| 1. Selection rules | first draw, random, fewest qubits, shortest chain, measured, oracle | p_solve minus first draw, Pegasus 6 and Zephyr 4 | `results/rules/{pegasus6,zephyr4}.log` |
| 2. Reads and pool size | K = 4, 8, 16 by 64, 128, 256 reads | measured gain, oracle gain | `results/rules/*_k*_r*.log` |
| 3. The regime | fill 80 to 95 by chain length | minorminer at 20 and 200 tries and anytime 300 to 600 s, witness roots | `results/fill/*`, `results/seeded/*` |
| 4. The objective by scale | 50, 75, 250 to 440 variables | witness p_solve, witness residual, minorminer p_solve | `results/scale/objective_*.log` |
| 5. The ladder | 8 rungs from K3-into-C5 to 24 variables on the full hosts | held-out validity before and after, paired gain | `results/curriculum/*`, `results/hardware/*` |
| 6. Transfer | Pegasus-trained, Zephyr-trained | held-out rate on each host's set | `results/transfer/*.log` |
| 7. Deployment | 60 s, 300 s, both hosts | coverage and seconds to first valid, policy against minorminer | `results/deploy/modern_*.log` |
| 8. The congested regime | fill 80 to 95, both hosts | constructor coverage against minorminer coverage (0 by construction) | **pending** |
| 9. Quality against the non-learned best | constructor, minorminer, plus random growth, plus measured-selection growth, plus heuristic growth | paired residual, qubits, validity | `results/curriculum/*q_*.log`, **pending** at the modern scale |
| 10. Ablations | support, features, actor, baseline, curriculum | held-out validity at one rung, three seeds | partly in `results/curriculum/`, **missing** as one table |

## The figures

1. **The regime map.** x: host occupancy 80 to 95 percent; y: fraction of instances embedded;
   curves for minorminer at three budgets, for the witness (1.0 by construction) and for the
   learned constructor. Data: table 3 plus table 8 when it exists.
2. **Measure, do not count.** x: pool size K; y: p_solve gain over a single draw; curves for
   the fewest-qubit rule, the measured rule and the oracle, both hosts. Data: `results/rules/`.
3. **The ladder.** x: logical variables (log scale, 3 to 440); y: held-out validity from
   empty; one point per rung with its bootstrap interval, annotated with the host size.
   Data: `results/curriculum/`, `results/hardware/`.

## What to cut

The successor scorer and its learning curve, the hybrid root-policy experiments, the contact
policy, the pool-ceiling comparison, the adaptive-allocation prior, and the chronology of
withdrawn numbers. They are the record of how the direction was found and they are in
`STATUS.md`; the paper needs C1 to C11 and the growth control.

## The single result that decides the tier

C10 with an interval, on both hosts, at 85 percent occupancy or above, with table 7 printed
beside it so nobody can say the comparison was chosen after the fact.
