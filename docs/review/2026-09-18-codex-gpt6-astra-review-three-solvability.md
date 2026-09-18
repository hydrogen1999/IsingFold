**I would not recommend an oral on this change alone.** Shrinking the host is defensible benchmark development, but “the same congestion” is unsupported. It restores a measurable endpoint while weakening the scaling claim identified in the previous review. [review-two.md:78](/Users/nguyencongt/Documents/prj_IsingFold/docs/review/2026-09-18-codex-gpt6-astra-review-two.md:78)

This is a static review of the named sources, prior reviews, and relevant status entries and logs. No numerical work ran. Bibliography, fonts, anonymity, and page-limit checks were outside scope.

**First, correct the diagnosis.** “Whatever the embedding” exceeds the evidence: zero hits in 512 independent reads gives an approximate 95 percent upper bound of 0.006 for that particular program, not proof of zero probability across embeddings. The record’s universal wording should go. [STATUS.md:564](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:564)

The size map also uses bins: the Pegasus “50” row contains instances with 54 to 73 variables. It is not a controlled measurement at exactly 50. [objective_pegasus.log:2](/Users/nguyencongt/Documents/prj_IsingFold/results/scale/objective_pegasus.log:2), [objective_pegasus.log:15](/Users/nguyencongt/Documents/prj_IsingFold/results/scale/objective_pegasus.log:15)

Loop density changes interaction structure, but **does not remove logical variables here**. Every support node survives through the field dictionary, including zero-field isolates. Edges survive only when their summed coefficient is nonzero, so even “every touched edge” is inaccurate because contributions can cancel. [planting.py:138](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/data/planting.py:138), [embedding.py:42](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/embedding.py:42)

**Shrinking is legitimate if you explicitly narrow the claim.** The assertion that shrinking leaves the logical graph untouched is false: the probe rebuilds the host, partition, and quotient for each size. [solvability_frontier.py:85](/Users/nguyencongt/Documents/prj_IsingFold/probes/solvability_frontier.py:85)

Equal fill fixes one ratio. It does not fix boundary effects, routing alternatives, graph structure, or the number of decisions a constructor must coordinate. More fundamentally, the planted witness certifies an upper bound on required occupancy. The generator’s explicit lower bound is only variables divided by host qubits. For your pilot, the immediate bounds are **70 to 90 percent**, not certified 90 percent congestion. [gen_fill_corpus.py:69](/Users/nguyencongt/Documents/prj_IsingFold/probes/gen_fill_corpus.py:69)

At 28 to 190 variables, you can establish compact constrained packing and embedding-sensitive sampling. You lose evidence that learned construction handles hundreds of interacting placement decisions near capacity. The killing reviewer sentence is:

> “The authors recover their preferred metric by reducing the problem size, then treat equal witness occupancy as evidence that the original scaling challenge was preserved.”

Retaining the large-instance track prevents that sentence from describing the whole paper.

**Minorminer failing at 28 variables is credible, but insufficient evidence of hard packing.** Small size does not guarantee heuristic success. However, this wrapper supplies only a seed and `tries`, leaving other search controls at defaults. It also embeds the interacting graph first and attaches isolated variables afterward, returning failure if no space remains. Separate those failure modes before attributing everything to congestion. [\_initializers.py:19](/Users/nguyencongt/Documents/prj_IsingFold/probes/_initializers.py:19)

The cheapest falsifier on **apollo** is to greedily remove redundant witness qubits while preserving connectivity and every logical contact, then ask minorminer to improve the complete witness using `initial_chains` and `skip_initialization`. Those controls explicitly support chain improvement. [minorminer API](https://docs.dwavequantum.com/en/latest/ocean/api_ref_system/generated/minorminer.find_embedding.html)

A valid result using 32 or fewer qubits immediately disproves a claim of required 90 percent occupancy. Simply returning the supplied witness proves nothing about search difficulty. Failure to shrink also proves nothing about optimality.

Next, try independent seeds and graph relabelings under a short fixed deadline, plus a bounded independent packing search. Easy independent solutions implicate minorminer’s search. A certified lower bound near 36 qubits supports tight packing. Persistent heuristic failure alone remains inconclusive. Witness-seeded minorminer belongs strictly in this diagnostic, never inside learned completion.

**A second annealing depth is defensible as a newly declared operating point.** It cannot retroactively inherit the original registration. ADR-002 makes the schedule part of the objective, and `Context` explicitly stores 200 sweeps. [ADR-002:15](/Users/nguyencongt/Documents/prj_IsingFold/docs/decisions/ADR-002-registered-schedule.md:15), [contracts.py:200](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/contracts.py:200)

Frame it as: “We calibrated an additional compute regime on development instances, then froze it before final evaluation.” Retain the original 200-sweep results, use identical depths for all arms within each regime, and charge selection as well as assessment work.

Keeping beta endpoints fixed preserves the compression question, but 20,000 sweeps changes the solver budget substantially. Calling both outputs `p_solve` does not make them the same objective under the same budget. Store the new depth in the context itself; the current probe passes its sweep override directly to sampling. [solvability_frontier.py:46](/Users/nguyencongt/Documents/prj_IsingFold/probes/solvability_frontier.py:46)

**Fields are defensible calibration, and they literally contain planted hints.** The implementation sets each selected field to `-b * planted_spin`. Thus every nonzero field reveals one planted spin; with positive weights and fields on every node, their signs reveal a complete optimum. [planting.py:129](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/data/planting.py:129)

That is not automatically disqualifying for an embedding benchmark. Planted solutions and tunable difficulty have established precedent in annealing benchmarks, including clause-density calibration. [Hen et al.](https://arxiv.org/abs/1502.01663) The defensible claim is performance on a declared synthetic ensemble. It cannot support broad claims about solving difficult, unstructured optimization problems.

At fixed support and seed, this knob preserves the logical graph because fields are drawn after loop construction. That is a cleaner structural control than shrinking. [planting.py:101](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/data/planting.py:101)

But it does **not** isolate logical difficulty perfectly. Fields change the RMS scale used for registered chain strengths, and can change physical autoscaling. [program.py:28](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/program.py:28), [program.py:153](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/program.py:153) They also remove ground-state degeneracy, so increased solve probability is not guaranteed.

Choose one field rate using development instances and neutral reference embeddings, publish zero-field results, and freeze it before learned comparisons. Check a simple field-guided classical solver to expose trivialization. Calibrating until the learned method wins would be indefensible.

**Nonzero probability is only the first gate.** The current probe measures all strengths and reports their same-block maximum. Its minorminer number is also a maximum, whereas `p_registered` uses index 1. These are different strength policies, and the maxima need independent assessment before supporting quality claims. [solvability_frontier.py:40](/Users/nguyencongt/Documents/prj_IsingFold/probes/solvability_frontier.py:40), [solvability_frontier.py:118](/Users/nguyencongt/Documents/prj_IsingFold/probes/solvability_frontier.py:118)

The literal minimum experiment is two valid embeddings of the **same instance**. On apollo, compare the witness with a valid alternative obtained through chain relocation or rerouting. Prefer equal qubit counts. Where available, include minorminer and grown-minorminer. Use a separate pilot block to identify the predicted better embedding, then two fresh 512-read blocks per embedding at the same fixed strength.

One pair establishes discrimination only for that instance. My smallest useful benchmark gate is twelve fresh validation lineages, with:

- Mean independently assessed advantage of at least **0.05 absolute solve probability**.
- A paired 95 percent interval across lineages entirely above zero.
- The result not driven by one exceptional instance.

Those are proposed advancement thresholds. A null means the chosen cell has not established useful discrimination.

Minorminer failure cannot serve as the “bad embedding.” Report its conditional quality as unavailable and its construction failure separately. A failure-inclusive deployment score may assign zero, but that demonstrates feasibility advantage. Every learned comparison must include grown-minorminer under the same deadline, resource cap, and selection budget.

Also preserve the distinction between per-read success and obtaining any success in a block. The evaluator reports hits divided by reads; block success would already be saturated around your pilot’s rate. [evaluator.py:65](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/evaluator.py:65)

**My concrete commitment would be Pegasus 2 as the compact congestion track:** 40 host qubits, actual witness occupancy 36/40, 28 logical variables, zero added fields, 200 sweeps, beta range 0.1 to 2.0, fixed strength ratio 1.0, and 512 fresh assessment reads per returned embedding. Use fresh instances; condition on variable count before observing solver outcomes because partition lengths are random. [gen_fill_corpus.py:55](/Users/nguyencongt/Documents/prj_IsingFold/probes/gen_fill_corpus.py:55)

Your reported 0.461 pilot is the only supplied configuration demonstrating comfortable measurement headroom without changing depth or adding hints. It earns this configuration a validation experiment. Packing and discrimination still need to pass before it earns a headline.

**One objective is not worth discarding the large-instance evidence.** Keep solve probability primary in the compact track and report it, including zero-hit bounds, on the large track. Retain residual and feasibility there as declared endpoints. Two transparent operating regimes are scientifically stronger than apparent uniformity purchased by abandoning the scaling question. All quality claims remain about classical simulated annealing on compiled programs. [evaluator.py:89](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/evaluator.py:89)

1. **Repair interpretation and measurement.** Correct the zero-probability, size-bin, and preserved-congestion claims; add comparable fixed-strength rates and fresh assessment.
2. **Audit packing on apollo.** Prune the witness, distinguish minorminer failure modes, and test independent search. A substantially smaller certificate defeats the required-fill claim.
3. **Run the twelve-lineage discrimination gate on apollo.** Failure blocks promotion of Pegasus 2 to the headline benchmark.
4. **Use goose for controlled large-instance calibration.** Finish the running depth sweep; test fields on fixed graphs before sacrificing scale. Freeze the protocol before final testing.
5. **Evaluate learned performance with grown-minorminer throughout.** Retain large-instance feasibility and residual results, including losses. Compact success alone does not satisfy the oral case.
