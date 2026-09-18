**I would not recommend an oral yet. The strongest current submission is a benchmark-and-measurement paper.** A substantial learned feasibility win could change that assessment, but finding 4 first needs an endpoint audit.

I read `STATUS.md` in full, both boards, the three recent reviews, and relevant sources and logs. No numerical work ran. Bibliography, fonts, anonymity, and page-limit checks were outside scope.

One evidence qualification: the local deep-anneal log contains only the 20,000-sweep result; the large-host pruning log contains only its header. I therefore treat the additional saturation and 488-variable results as author-reported, not independently verified from these copies. [pegasus3_deep.log:3](/Users/nguyencongt/Documents/prj_IsingFold/results/frontier/pegasus3_deep.log:3), [prune_pegasus6.log:1](/Users/nguyencongt/Documents/prj_IsingFold/results/frontier/prune_pegasus6.log:1)

**A. Replace the thesis with this sentence.**

> On the tested synthetic Pegasus and Zephyr candidate pools, qubit count provides a weak quality signal, while selection using a separate sampling block yields larger independently assessed gains under the declared classical simulated-annealing protocol.

That survives the correlation objection and matches the selection results: approximately +0.15 over the first draw for measurement, versus +0.04 for fewest qubits. [pegasus6.log:36](/Users/nguyencongt/Documents/prj_IsingFold/results/rules/pegasus6.log:36), [zephyr4.log:36](/Users/nguyencongt/Documents/prj_IsingFold/results/rules/zephyr4.log:36)

Three qualifications belong beside it:

- The qubit correlations are defined on 27 and 28 pools; longest-chain correlations cover only 12 and 2. Those chain results cannot support a general ranking of structural predictors. [pegasus6.log:40](/Users/nguyencongt/Documents/prj_IsingFold/results/rules/pegasus6.log:40), [zephyr4.log:40](/Users/nguyencongt/Documents/prj_IsingFold/results/rules/zephyr4.log:40)
- Your supplied narrow qubit spread limits the conclusion to local candidate selection. It says little about optimal quality across substantially different resource budgets.
- Measurement purchases additional information. An efficiency claim needs total deployment cost, including a resource selector allowed to reinvest its saved measurement budget.

The isolated finding is workshop-sized. A reproducible benchmark that separates construction difficulty, sampling difficulty, and measurement cost could support a main-conference poster. An oral needs a consequential capability or a broadly useful, well-established scientific finding. “Sampling predicts sampling outcomes better than counting qubits” is insufficient by itself.

**B. Finding 4 is not yet a clean demonstration that depth compensates for embedding defects.**

The earliest problem is **strength selection**, before mechanism.

The probe stores fixed-strength results separately from the best result across four strengths. Its human-readable residual column prints `residual_best`, chosen separately for each embedding on the same measurement block. That is not an independently assessed strength-selection policy. [solvability_frontier.py:84](/Users/nguyencongt/Documents/prj_IsingFold/probes/solvability_frontier.py:84), [solvability_frontier.py:199](/Users/nguyencongt/Documents/prj_IsingFold/probes/solvability_frontier.py:199)

A concrete example demonstrates why this matters. At 20,000 sweeps, instance 2 has registered-strength residuals **0.07213 and 0.08678**, but best-strength residuals **0.04319 and 0.04343**. Near equality in the latter columns does not establish equality at the registered strength. [gate_pegasus3_f90.log:6](/Users/nguyencongt/Documents/prj_IsingFold/results/frontier/gate_pegasus3_f90.log:6)

Your previous review explicitly flagged same-block strength maxima. That concern remains open for the frontier summaries. [review-three:45](/Users/nguyencongt/Documents/prj_IsingFold/docs/review/2026-09-18-codex-gpt6-astra-review-three-solvability.md:45)

There are four further attacks:

- **The intervention is mislabeled.** The Pegasus gate records **13 absorbed qubits**, and Zephyr records **16**, rather than 24. Report realized growth, not the requested dose. [Pegasus gate:3](/Users/nguyencongt/Documents/prj_IsingFold/results/frontier/gate_pegasus3_f90_registered.log:3), [Zephyr gate:3](/Users/nguyencongt/Documents/prj_IsingFold/results/frontier/gate_zephyr2_f90.log:3)
- **Growth does not isolate chain length.** It absorbs adjacent free qubits. The compiler then programs all internal chain edges, redistributes logical couplings across available contacts, and autoscales. Growth can therefore change redundancy, contact geometry, and physical coefficients together. Calling the result a “bad embedding” assumes the mechanism being tested. [solvability_frontier.py:42](/Users/nguyencongt/Documents/prj_IsingFold/probes/solvability_frontier.py:42), [program.py:108](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/program.py:108)
- **Majority decoding is a plausible moderator.** Tied votes receive random spins. Changing chain lengths changes opportunities for ties and disagreement, so the result may concern this decoder specifically. [evaluator.py:38](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/evaluator.py:38)
- **Residual normalization alone cannot explain the disappearance.** Its denominator is the same logical ground-energy magnitude for both embeddings and both depths on an instance. However, an aggregate can hide cancellation across instances. Inspect individual gaps and raw logical-energy differences. [evaluator.py:131](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/evaluator.py:131)

Statistically, a significant shallow result and nonsignificant deep result do not establish a significant depth interaction. The read counts also differ. Test the change in the paired embedding gap directly, then test whether the deep gap lies inside a declared equivalence margin. “ONLY depth” cannot follow from two tested depths. [STATUS.md:1394](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:1394)

**Cheapest falsifier: existing-log reanalysis on apollo, with zero new annealing.** Pair the twelve Pegasus instances across depths using only `residual_registered`. Report the deep gap and the shallow-minus-deep interaction, with intervals across instances.

My proposed falsification threshold is a deep residual disadvantage of **at least 0.003 with its paired interval above zero**. That rejects practical “erasure.” An interaction interval containing zero leaves attenuation unestablished. To support practical erasure, preregister a margin such as **±0.002** and require the deep interval entirely inside it.

If that survives, the cheapest mechanism extension is one equal-qubit rerouting alternative per instance, evaluated at both depths with equal reads. Save physical samples and decode them both by majority vote and by local-energy resolution of broken chains, which Ocean supports. A persistent deep rerouting gap defeats the general embedding claim; decoder-dependent disappearance confines it to the decoder. [Ocean decoding documentation](https://docs.dwavequantum.com/en/latest/ocean/api_ref_system/embedding.html)

Keeping 200 sweeps as a declared operating budget is defensible. Making it indispensable because it produces a favorable contrast is not.

**C. The best two-day bet is held-out empty-start construction near 100 variables.**

Specifically: **a frozen learned constructor achieves substantial deployment coverage on both Pegasus 3 and Zephyr 2 where equally budgeted controls fail.** Use the running congested jobs; the board’s runtime warning makes another 488-variable training campaign a poor two-day investment. [STATUS.md:1439](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:1439)

The smallest deciding version is:

- Named fill 0.90, one frozen checkpoint per topology, twelve fresh validation lineages per topology, selected before observing baseline outcomes.
- Empty starts, a **300-second deployment deadline**, equal CPU allocation and resource cap, with three deployment repetitions.
- Paired controls: tuned anytime minorminer, grown-minorminer, and a strong nonlearned controller using the same support and termination rules. Include the starting checkpoint to identify what congested training added.
- Bootstrap instances, treating deployment repetitions as repeated measurements.

**Success:** at least **50 percent coverage on each topology**, at least **20 percentage points above the strongest control**, and a positive paired interval.

**Kill this two-day method bet:** coverage at or below **one instance in twelve on either topology**, or a **nonpositive advantage over the nonlearned controller**. Intermediate outcomes remain preliminary. These are proposed decision thresholds, not power guarantees or oral acceptance criteria.

Two hundred tries is not a wall-time match. The wrapper passes `tries` and a seed; the final comparator should restart until the shared deadline, with restart and patience settings chosen on validation. [\_initializers.py:19](/Users/nguyencongt/Documents/prj_IsingFold/probes/_initializers.py:19), [minorminer API](https://docs.dwavequantum.com/en/latest/ocean/api_ref_system/generated/minorminer.find_embedding.html)

Where minorminer returns nothing, grown-minorminer also has no starting embedding. Report its failure and conditional quality as unavailable. Never substitute the witness. A learned validity advantage there establishes construction capability; it does not establish better quality among valid embeddings.

My ranking of the alternatives is:

| Candidate result | Value now |
|---|---|
| Congested held-out validity | Highest: establishes a useful learned capability |
| Cross-topology transfer at congestion | Next, after an in-distribution win |
| Targeted learning ablation | Necessary attribution; a full architecture table has lower immediate value |
| Benchmark without a learned win | Stronger current submission than an unsupported learned-method narrative |

Small-fragment transfer is already recorded. The open question is transfer at congestion, not whether any transfer exists. [STATUS.md:624](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:624)

This pilot would materially improve the oral case. Replication across training seeds and evidence beyond the planting distribution would still be needed.

**D. Cut the universal claims and the development diary.**

Cut “qubit count is not a quality signal,” “only the shallow schedule reveals embedding quality,” and the causal compensation story until the preceding audit passes.

Replace “the law” and depth-invariant decay with a descriptive trend over the tested ensemble. The reported fit excludes unresolved cells; that alone does not establish the claimed lower bound on the true slope or a universal solvability boundary. [STATUS.md:1320](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:1320)

Cut “certified requirement” for pruned occupancy. Greedy pruning supplies a smaller **sufficient** embedding, not a minimum. Also remove the universal claim that successful minorminer embeddings match pruning: one logged Pegasus instance uses 110 qubits against a 92-qubit pruned witness. [witness_prune.py:10](/Users/nguyencongt/Documents/prj_IsingFold/probes/witness_prune.py:10), [prune_pegasus.log:46](/Users/nguyencongt/Documents/prj_IsingFold/results/frontier/prune_pegasus.log:46)

Move retired hybrids, scorer variants, and ladder chronology to an appendix. Preserve the grown-control comparisons and modern-corpus losses prominently. [STATUS.md:534](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:534)

The most dangerous abstract/Figure 1 inference is that **objective-guided selection demonstrates a learned quality advantage at congestion**. Your selection experiment chooses among minorminer draws; congested learned validity remains pending. Those are separate results. [selection_rules.py:162](/Users/nguyencongt/Documents/prj_IsingFold/probes/selection_rules.py:162), [STATUS.md:1439](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:1439)

1. **Repair evidence first:** synchronize complete logs and identify the exact strength fields behind every frontier summary.
2. **On apollo, run the fixed-strength depth reanalysis:** test interaction and equivalence before buying new reads.
3. **Spend the two-day compute window on congested deployment coverage:** keep apollo within shared capacity and goose within two Slurm allocations.
4. **If coverage passes, test frozen cross-topology transfer and one matched learning ablation.** Otherwise lead with the benchmark-and-measurement contribution.
5. **Freeze claims, protocol, and checkpoints before opening the final test list.** Report failures alongside successes.
