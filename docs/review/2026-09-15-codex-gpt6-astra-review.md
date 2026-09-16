The supported conclusion is **“the tested models do not demonstrate useful transfer.”** The stronger conclusions, “there is no learnable signal” and “a perfect local policy cannot help,” are not established. The record itself contains a reproducible within-state selection advantage, while the pool experiment tests one growth distribution. I also found a concrete mismatch between the program shown to the successor scorer and the program used to produce its labels. ([STATUS.md:26](STATUS.md:26), [pool_ceiling.py:123](probes/pool_ceiling.py:123), [train_successor.py:35](probes/train_successor.py:35))

I accept the established measurements. My disagreement is with what they rule out.

**A. The models observe coefficients, but the successor scorer observes the wrong compiled program.**

Neither model is restricted to graph structure:

| Quantity | Policy observation | Successor scorer |
|---|---|---|
| Logical \(h,J\) | Signed fields, signed couplings, absolute values and incident-load summaries. ([tensorize.py:365](src/isingfold/rl/tensorize.py:365), [tensorize.py:441](src/isingfold/rl/tensorize.py:441)) | Logical fields and couplings appear in chain and logical-edge features. ([successor_scorer.py:249](probes/successor_scorer.py:249), [successor_scorer.py:266](probes/successor_scorer.py:266)) |
| Programmed \(h^{phys},J^{phys}\) | Current physical fields and couplings, plus coefficients on candidate/archive edge-use records. ([tensorize.py:398](src/isingfold/rl/tensorize.py:398), [tensorize.py:480](src/isingfold/rl/tensorize.py:480), [tensorize.py:701](src/isingfold/rl/tensorize.py:701)) | Physical fields, signed physical couplings, chain membership and internal/external edge indicators. ([successor_scorer.py:208](probes/successor_scorer.py:208), [successor_scorer.py:228](probes/successor_scorer.py:228)) |
| Strength and autoscale | Registered strengths, hardware limits and current reference-program scale. ([tensorize.py:592](src/isingfold/rl/tensorize.py:592)) | Strength and scale explicitly enter the final readout. ([successor_scorer.py:341](probes/successor_scorer.py:341)) |
| Coupler load | Logical incident loads and maximum incident physical coupling. Total external physical load is recoverable from the graph, but is not an explicit hardware-node scalar. ([tensorize.py:370](src/isingfold/rl/tensorize.py:370), [tensorize.py:399](src/isingfold/rl/tensorize.py:399)) | Recoverable from weighted edges and membership. The optional physics features explicitly sum external load. ([successor_scorer.py:149](probes/successor_scorer.py:149)) |

The decisive defect is `compile_for()`. It says it constructs the program the evaluator runs, but uses

\[
F_{\text{shown}}=r_j\,\operatorname{mean}_{(u,v)}|J_{uv}|.
\]

The evaluator uses

\[
F_{\text{evaluated}}
=r_j\max\!\left(\epsilon,
\sqrt{\frac{\sum_i h_i^2+\sum_{uv}J_{uv}^2}{n+m}}\right).
\]

These differ even for zero-field, uniform-magnitude couplings. The environment registers the second formula; cached labels are obtained through that environment. ([train_successor.py:35](probes/train_successor.py:35), [program.py:28](src/isingfold/rl/program.py:28), [env.py:261](src/isingfold/rl/env.py:261), [train_quality.py:50](probes/train_quality.py:50))

I ran a read-only compilation check on the four local fixtures. The scorer’s strengths exceeded the evaluator’s by approximately **8.6%, 18.5%, 1.0% and 43.9%**. This changes physical chain couplings and the derived robustness features. Depending on which coefficient determines autoscale, it can also change every programmed coefficient. This invalidates the claim that the ablation presents the exact evaluated program. It does **not** prove that fixing it will restore transfer: raw logical coefficients still contain enough information to reconstruct the correct strength. ([fixture instances:1](audit/5/diagnostics/fixture_corpus/instances.jsonl:1), [train_successor.py:39](probes/train_successor.py:39), [program.py:115](src/isingfold/rl/program.py:115), [program.py:134](src/isingfold/rl/program.py:134))

The policy has a related representational limitation. Current, archive and successor coefficient views use strength index **0**, whereas the probe labels use the fixed selector’s index **1**. The policy receives all registered strengths, so this is not information-theoretic blindness, but it must reconstruct the effects of changing strength and autoscale. ([env.py:323](src/isingfold/rl/env.py:323), [tensorize.py:813](src/isingfold/rl/tensorize.py:813), [tensorize.py:842](src/isingfold/rl/tensorize.py:842), [env.py:81](src/isingfold/rl/env.py:81))

My proposed input correction is specific:

- Use the shared strength registry and compiler for both labels and observations, and require identical program digests.
- At the candidate/archive phase construction, expose the evaluated strength, its autoscale, and candidate-specific physical fields. The current factor features are structural, while the hardware-node fields describe the current program. ([tensorize.py:735](src/isingfold/rl/tensorize.py:735), [tensorize.py:807](src/isingfold/rl/tensorize.py:807))
- Add \(L_q=|h_q^{phys}|+\sum_{r\notin C(q)}|J_{qr}^{phys}|\) beside `j_inc` in hardware-node features, with corresponding candidate-phase summaries. Add chain cut/load margins to factor features. These are useful explicit computations, not previously absent underlying information. ([tensorize.py:399](src/isingfold/rl/tensorize.py:399), [successor_scorer.py:149](probes/successor_scorer.py:149))
- Correct the claimed “cheapest cut” feature: it considers singleton cuts and only one component of each bridge removal. It does not enumerate all cuts or both bridge sides. For short chains, exhaustive subset cuts are cheap enough to test. ([successor_scorer.py:172](probes/successor_scorer.py:172))

There is also a mismatch between the intended physics and the measured objective. The evaluator is classical simulated annealing. `beta_range` defaults to `None`, and the main quality, successor and pool probes do not override it. Thus their annealing schedule adapts to the coefficients and cancels uniform energy compression, despite the repository already demonstrating this failure mode. These measurements cannot establish the fixed-temperature resource-quality argument. ([evaluator.py:90](src/isingfold/rl/evaluator.py:90), [contracts.py:189](src/isingfold/rl/contracts.py:189), [train_quality.py:214](probes/train_quality.py:214), [train_successor.py:130](probes/train_successor.py:130), [pool_ceiling.py:70](probes/pool_ceiling.py:70), [STATUS.md:56](STATUS.md:56))

One further hypothesis deserves a cheap control. The sampler call leaves update order at its default, while the scorer pools graphs without encoding sampler order. D-Wave documents that sequential updates can introduce an order-dependent dynamical bias. Relabeling a logical graph could therefore change the finite-sweep target while leaving an invariant scorer’s prediction unchanged. This is a plausible missing variable, not a demonstrated explanation here. Test relabelings with independent reads and randomized update order before adding an order feature. ([evaluator.py:109](src/isingfold/rl/evaluator.py:109), [program.py:85](src/isingfold/rl/program.py:85), [successor_scorer.py:341](probes/successor_scorer.py:341), [official sampler documentation](https://docs.dwavequantum.com/en/latest/ocean/api_ref_samplers/generated/dwave.samplers.SimulatedAnnealingSampler.sample.html))

**B. Read noise weakens fine rankings, but does not explain away the demonstrated signal.**

The evaluator reports ground-state hits divided by reads. Under independent Bernoulli reads with stable probability \(p\), its sampling variance is \(p(1-p)/N\). The following are calculations from that measurement model, using the worst case \(p=0.5\). ([evaluator.py:128](src/isingfold/rl/evaluator.py:128), [evaluator.py:65](src/isingfold/rl/evaluator.py:65))

| Reads per candidate | Maximum standard error | Standard error of a two-candidate difference | Approximate misordering probability for a true +0.05 gap |
|---|---:|---:|---:|
| 256 | 0.0313 | 0.0442 | 13% |
| 512 | 0.0221 | 0.0313 | 5.5% |

For a true gap of only +0.02, the corresponding misordering probabilities are roughly 33% and 26%. Therefore these labels can support coarse selection while providing unstable supervision for near ties.

An existing reliability experiment **does** exist in the audit record: eight states from four tasks, 56 distinct successors, two independent 512-read blocks, within-state correlations 0.979–0.998, and mean block-to-block RMSE 0.0245. Its independently assessed selection advantage was +0.1835. The report explicitly limits this to the fixtures and immediate COMMIT quality. ([audit report:19](audit/3/ISINGFOLD_TRAINING_ROOT_CAUSE_VI.md:19), [audit report:34](audit/3/ISINGFOLD_TRAINING_ROOT_CAUSE_VI.md:34), [audit report:38](audit/3/ISINGFOLD_TRAINING_ROOT_CAUSE_VI.md:38))

That is evidence against “the labels are mostly noise.” It does not measure ranking reliability across the later hard and modern corpora. The frontier noise control measures one embedding repeatedly, not the ordering of a candidate pool. Its statement that smaller effects are inside the instrument’s noise is also too strong: a single-instance range is not the uncertainty of a mean paired effect over many instances. ([frontier2.py:210](probes/frontier2.py:210))

I would extend the existing experiment as follows:

1. Freeze and deduplicate eight candidates on each of 32 new lineages.
2. Obtain two independent 512-read blocks for **every** candidate; retain per-read outcomes to also analyze 256-read prefixes.
3. Report tie-aware rank correlation, top-choice agreement, and independent-block regret. Select with A and assess with B, then reverse.
4. Use the mean of all candidates’ assessment probabilities as the uniform-selection reference.
5. Bootstrap lineages, and estimate between-candidate variance separately from read variance.

This costs 262,144 reads before any optional high-read reference. It extends the fixture experiment without confusing candidate selection with assessment. ([audit report:21](audit/3/ISINGFOLD_TRAINING_ROOT_CAUSE_VI.md:21), [train_quality.py:385](probes/train_quality.py:385))

The flat learning curve establishes failure of this training recipe over these nested subsets. It does not rule out data scale generally. The subsets are lexicographic prefixes, both model seeds use the same data, and every point gets 400 full-batch updates with the same architecture and compilation mismatch. The largest run also retains nonzero training regret. None of this reverses the observed flatness; it limits its interpretation. ([train_successor.py:153](probes/train_successor.py:153), [run_curve.sh:21](scripts/apollo/run_curve.sh:21), [n200_seed0.log:14](results/curve/n200_seed0.log:14))

Given the independent-read oracle advantage and fixture reliability, I would prioritize **target consistency and generalization bias**, not simply buy more reads or declare sampling noise the cause.

**C. The pool result constrains the tested proposal distribution, not the whole MDP.**

The pool experiment grows each draw with `mode="contact"`, one power-law configuration and bounded chain lengths. It does not optimize over all rewrites, action sequences, intermediate invalid states, strengths or construction policies. Moreover, its “ceiling” selects using 256-read estimates. That is measured selection performance, not an omniscient upper bound. ([pool_ceiling.py:61](probes/pool_ceiling.py:61), [pool_ceiling.py:123](probes/pool_ceiling.py:123), [pool_ceiling.py:151](probes/pool_ceiling.py:151))

The defensible inference is: **replacing independent draws with these correlated grown copies loses useful diversity.** The inference that every perfect local policy is dominated by restarting is unsupported. A successful local policy could make different edits, use several steps, or spend less time per useful candidate. The existing environment includes group rewrites and repairs beyond the tested growth operation. ([results/modern/pegasus6/pool.log:5](results/modern/pegasus6/pool.log:5), [proposal.py:815](src/isingfold/rl/proposal.py:815))

There is also a remaining bug in the degradation experiment: `frontier2.improve()` passes an initializer that always returns the starting embedding. RESTART calls that same initializer. Training explicitly fixed this by pinning only the first call and using fresh minorminer draws afterward. Thus the frontier still changes the trained action’s semantics. This affects the degradation claim, not the established registered policy-versus-random result. ([frontier2.py:54](probes/frontier2.py:54), [frontier2.py:89](probes/frontier2.py:89), [proposal.py:587](src/isingfold/rl/proposal.py:587), [train_bestof.py:208](probes/train_bestof.py:208))

Two formulations escape the tested pool restriction:

- Construct placements and routes from an empty embedding, with bounded backtracking.
- Control a larger search operation: choose which region to remove, what to preserve, which demand to route, or how to initialize a new search.

Their value would have to be demonstrated at matched time. Neither is bounded by measuring one grown copy per minorminer draw.

Finally, the direct labels concern immediately valid successors. They do not supervise whether an initially worse or incomplete state enables a better completion. And `train_bestof.py` updates the actor with imitation loss, despite using the PPO collector. Its failure is not a direct test of every RL credit-assignment formulation. ([train_quality.py:146](probes/train_quality.py:146), [train_quality.py:50](probes/train_quality.py:50), [train_bestof.py:365](probes/train_bestof.py:365))

**D. Constructive learning should first target certified feasibility under a resource budget.**

There is already construction scaffolding, but simply switching modes is insufficient. Empty chains are supported; the standard evaluator hardcodes improvement mode; the default horizon is 32 decisions; and the 24-placement quota is consumed by roots for the first unplaced variable. The fill tasks contain hundreds of variables. A witness trajectory must first be representable by the action support and fit within the horizon. ([env.py:325](src/isingfold/rl/env.py:325), [evaluate.py:1084](src/isingfold/rl/evaluate.py:1084), [contracts.py:152](src/isingfold/rl/contracts.py:152), [contracts.py:225](src/isingfold/rl/contracts.py:225), [proposal.py:284](src/isingfold/rl/proposal.py:284), [pegasus6.log:5](results/fill/pegasus6.log:5))

My proposed formulation is:

| Component | Proposal and rationale |
|---|---|
| **State** | Logical graph; host graph and native coordinates; partial connected chains; occupancy; unplaced variables; unmet contacts; residual free components and directional capacity; remaining qubit budget \(B\), time and backtracking budget. Budget conditioning and completion heads are explicitly unfinished in the advisor checklist. ([MEETING_CHECKLIST.md:64](docs/MEETING_CHECKLIST.md:64), [MEETING_CHECKLIST.md:50](docs/MEETING_CHECKLIST.md:50)) |
| **Action** | Factorized choice of variable and root; extension by an adjacent free qubit; optional bounded path routing; bounded removal and rebuilding of a selected chain group. Expose all legal roots through a pointer head rather than the current lexicographic shortlist. Scale the horizon with placements and extensions. ([proposal.py:284](src/isingfold/rl/proposal.py:284), [contracts.py:153](src/isingfold/rl/contracts.py:153)) |
| **Reward** | Initially, terminal reward 1 for an independently validated embedding within \(B\), 0 otherwise. Optional potential shaping for placement/contact progress must telescope away, including at failure. Add downstream quality only after establishing a measurable endpoint; the current reward is sampled terminal solve probability. ([env.py:1534](src/isingfold/rl/env.py:1534)) |
| **Teacher** | Root each witness chain and reveal it through connected spanning-tree prefixes. Randomize variable order and valid growth order. Train on sets of witness-consistent actions rather than one arbitrary serialization. Witnesses certify successful completions; they do not certify that other actions have no completion. ([gen_fill_corpus.py:42](probes/gen_fill_corpus.py:42), [gen_fill_corpus.py:129](probes/gen_fill_corpus.py:129)) |
| **Recovery learning** | Start from partially revealed witnesses, then introduce policy-generated mistakes. Use bounded repair teachers where available. Label solver timeouts as “teacher failed within budget,” not “infeasible.” The current immediate-COMMIT dataset excludes successors it cannot evaluate, so it does not supply this supervision. ([train_quality.py:154](probes/train_quality.py:154)) |
| **Curriculum** | Progress from short residual completions to full construction, then increase fill through 70%, 80%, 85%, 90%, 95%. Vary chain-length distributions, graph families and host defects independently. Split base instances before generating permutations and trajectories. The existing lineage machinery supports grouped and explicit out-of-distribution splits. ([lineage.py:70](src/isingfold/rl/data/lineage.py:70)) |

Two qualifications are essential.

First, a witness at 95% fill establishes a **feasible upper bound** on required qubits, not a minimum requirement of 95%. Singleton witnesses make that distinction disappear because every logical variable needs a distinct physical qubit. For longer chains, report the lower bound \(n/|V_H|\) and witness fill separately. Also compute actual witness occupancy: the generator can discard components, but its printed witness column uses requested fill times host size. ([gen_fill_corpus.py:119](probes/gen_fill_corpus.py:119), [gen_fill_corpus.py:130](probes/gen_fill_corpus.py:130), [gen_fill_corpus.py:153](probes/gen_fill_corpus.py:153))

Second, the completed Zephyr witness table now reports **0.0000 observed solve probability in every cell**, including valid minorminer outputs. Consequently, construction success cannot immediately be advertised as improved ground-state success. At the default 512 reads, zero hits only gives an approximate individual 95% upper bound of 0.0058, not proof of \(p=0\). Establish useful measurement resolution before quality fine-tuning; energy residual can be a separately declared secondary target, not silently substituted for the registered objective. ([zephyr4_witness.log:51](results/fill/zephyr4_witness.log:51), [witness_quality.py:31](probes/witness_quality.py:31), [witness_quality.py:54](probes/witness_quality.py:54))

The fair baseline is an **anytime minorminer system at the same elapsed-time deadline**, with tries, restart schedule and initialization tuned on validation. Count every attempt, including failures. Charge policy inference, feature construction, routing, validation, backtracking and any solver calls. Report training cost separately. The current wrapper specifies tries but no explicit time deadline, and the larger-budget logs already show that feasibility depends on budget. ([_initializers.py:16](probes/_initializers.py:16), [zephyr4_budget.log:3](results/fill/zephyr4_budget.log:3))

Include the same constructive search with heuristic or random decisions, imitation without RL, and policy-assisted minorminer. For singleton-witness tasks, also test a subgraph-matching baseline, since that family admits such solutions. A gain over ten-try minorminer alone would not identify the value of learning.

**E. Remaining routes to favourable artefacts.**

These are current risks, not grounds for withdrawing the established measurements without their underlying receipts.

1. **Reporting selection scores as performance.** `registered_bar` still prints a maximum over noisy episode returns. Equal \(K\) does not guarantee equal winner’s-curse bias when arms have different success probabilities or correlated candidates. Its independently assessed column is the relevant endpoint. ([registered_bar.py:15](probes/registered_bar.py:15), [registered_bar.py:98](probes/registered_bar.py:98))

2. **The withdrawn growth inflation remains the executable default.** `budget_sweep` defaults to two repeats, measures the start once, retains each arm’s largest measured utility and reports that same utility. There is no independent assessment of that winner. ([budget_sweep.py:51](probes/budget_sweep.py:51), [budget_sweep.py:85](probes/budget_sweep.py:85), [budget_sweep.py:103](probes/budget_sweep.py:103))

3. **Salted seeds remain in the pool probe.** Assessment still uses `hash(name) % 5`. Furthermore, `--seed` changes the growth RNG but not the fixed minorminer seed schedules. Repeating this command with another seed is not an independent replication of the entire experiment. ([pool_ceiling.py:72](probes/pool_ceiling.py:72), [pool_ceiling.py:116](probes/pool_ceiling.py:116), [pool_ceiling.py:135](probes/pool_ceiling.py:135), [pool_ceiling.py:152](probes/pool_ceiling.py:152))

4. **“All pools are size K” is not enforced.** Failed draws and measurements disappear. Pools as small as three survive; the mixed pool takes the first successful parent/growth pairs. Record attempted, valid, measured and distinct candidate counts for every arm. The current checks do not certify equal pool sizes. ([pool_ceiling.py:115](probes/pool_ceiling.py:115), [pool_ceiling.py:130](probes/pool_ceiling.py:130), [pool_ceiling.py:143](probes/pool_ceiling.py:143))

5. **Missing outcomes can improve reported means.** Quality probes swallow evaluator exceptions and compare surviving intersections. The registered-bar headline means exclude failures, although its paired difference correctly assigns them zero. The general evaluator intentionally excludes initializer failures from its conditional policy population. None of these conditional means should be presented as whole-system feasibility performance. ([train_quality.py:56](probes/train_quality.py:56), [train_quality.py:438](probes/train_quality.py:438), [registered_bar.py:110](probes/registered_bar.py:110), [registered_bar.py:119](probes/registered_bar.py:119), [evaluate.py:1126](src/isingfold/rl/evaluate.py:1126))

6. **The cache can silently preserve obsolete labels and observations.** Its key omits corpus contents, split contents, context, compiler, resolver and evaluator versions. `train_successor` accepts the loaded cache without verifying those identities and constructs a new context. Its source hash excludes the probe containing the erroneous compiler. ([train_quality.py:231](probes/train_quality.py:231), [train_successor.py:128](probes/train_successor.py:128), [train_successor.py:305](probes/train_successor.py:305))

7. **I found no direct label leakage through the cache key into the network.** The key is checked or printed; the successor network consumes explicit graph tensors. The stored task does contain evaluator-only ground energy and witness data, so that boundary should be tested, but possession in the pickle is not evidence that the network sees them. The demonstrated cache problem is stale provenance, not a label-bearing input feature. ([train_successor.py:132](probes/train_successor.py:132), [successor_scorer.py:271](probes/successor_scorer.py:271), [generate.py:237](src/isingfold/rl/data/generate.py:237), [tensorize.py:6](src/isingfold/rl/tensorize.py:6))

8. **The representation comparison changes its random reference.** Quality evaluation uses an RNG advanced during dataset generation; cache reuse skips that advancement. Successor evaluation resets its RNG to zero. The supposedly shared reference therefore changes from 0.5947 to 0.5780 in the logged comparison. Compare methods directly on paired candidates and assessment blocks, not by subtracting different random controls. ([train_quality.py:216](probes/train_quality.py:216), [train_quality.py:238](probes/train_quality.py:238), [train_quality.py:393](probes/train_quality.py:393), [train_successor.py:235](probes/train_successor.py:235), [if-mlp.log:24](results/quality_v2/if-mlp.log:24), [successor.log:30](results/r2/successor.log:30))

9. **Fresh reads do not make training states held out.** Both trainers print training-state assessment first. The quality probe’s held-out set is validation, and the historical policy development curve pooled validation and test. Reusing the same assessment seeds across model seeds is useful pairing, but does not supply independent measurement replications or undo adaptive benchmark use. ([train_successor.py:301](probes/train_successor.py:301), [train_quality.py:213](probes/train_quality.py:213), [results/README.md:35](results/README.md:35), [train_successor.py:251](probes/train_successor.py:251))

10. **Some frontier positives spend more measurements.** The comparator chooses the nearest budget on a ladder rather than an exactly matched budget. The logged +0.0194 and +0.0275 compare ten blocks against eight. These cannot support an equal-measurement claim. ([frontier2.py:344](probes/frontier2.py:344), [frontier2.log:39](results/v3/frontier2.log:39), [frontier2.log:49](results/v3/frontier2.log:49))

11. **Wall-time accounting still mixes different work.** `per_draw` includes candidate generation and evaluation, then charges that time to the resource-only arm. Final-only improvement also receives `keep_secs`, the extra comparison cost needed by keep-better. Use separate measured timers for actual work in each arm. ([frontier2.py:187](probes/frontier2.py:187), [frontier2.py:259](probes/frontier2.py:259), [frontier2.py:329](probes/frontier2.py:329))

12. **Growth arms are not a nested, matched-budget frontier.** Every arm independently draws target lengths starting from the same base. Each child contains its parent, but successive rows need not contain one another or spend identical resources. The printed interpretation overstates what was constructed. ([powerlaw_embeddings.py:105](probes/powerlaw_embeddings.py:105), [budget_sweep.py:90](probes/budget_sweep.py:90), [budget_sweep.py:138](probes/budget_sweep.py:138))

13. **Teacher acceptance still selects on noise.** Remeasuring the winner fixes reuse of its selection block, but accepting when fresh gain exceeds zero turns that fresh block into another selection stage. It is training supervision, not unbiased evidence of accepted teachers’ gains. An untouched third block is required to assess that claim. ([train_bestof.py:298](probes/train_bestof.py:298), [train_bestof.py:306](probes/train_bestof.py:306))

14. **Population and difficulty claims can drift.** The pool bootstrap groups by task name rather than root lineage; that becomes optimistic when multiple variants share a root. Fill logs use requested occupancy, and frustrated-loop planting retains only nonzero supported couplings, so the final graph need not be the full quotient graph described in the generator’s introduction. ([pool_ceiling.py:154](probes/pool_ceiling.py:154), [pool_ceiling.py:162](probes/pool_ceiling.py:162), [gen_fill_corpus.py:128](probes/gen_fill_corpus.py:128), [planting.py:138](src/isingfold/rl/data/planting.py:138))

The previously wrong COMMIT resolution is repaired in current code. That particular bug should not be invoked again without showing that an old cache bypassed the repair. ([train_quality.py:102](probes/train_quality.py:102))

**F. My top five changes, ranked by expected contribution to the research goal.**

The thresholds below are proposed decision rules, not existing results.

| Rank | Change | Falsifiable prediction | Cheapest experiment that would kill the proposed next stage |
|---|---|---|---|
| **1** | Constructive search trained from witness completions, then RL for recovery. | On unseen 85% fill instances, learning improves validity by at least 10 percentage points over the same search without learning at matched time. | First replay every witness through the action API. Then train a small completion policy and test increasing missing fractions. If it cannot beat heuristic completion on unseen instances, stop before full construction training. Current placement support and horizon must be fixed first. ([proposal.py:284](src/isingfold/rl/proposal.py:284), [contracts.py:153](src/isingfold/rl/contracts.py:153)) |
| **2** | Register a physically relevant, measurable quality protocol. | A fixed schedule restores sensitivity to programmed energy scale, and some near-threshold regime provides measurable quality differences. | Run the existing scale control plus repeated witness measurements on a few instances at longer schedules. If quality remains unresolved at affordable cost, stop claiming a solve-probability contribution for that regime. Existing Zephyr results already trigger this concern. ([check_sampler_scale.py:59](probes/check_sampler_scale.py:59), [zephyr4_witness.log:51](results/fill/zephyr4_witness.log:51)) |
| **3** | Make model inputs and labels refer to the identical compiled program. | Correcting strength and candidate-program views produces a repeatable improvement over the old scorer on the same frozen pools. | Verify digest equality, then rerun old versus corrected encodings with identical labels, picks and assessment seeds. If transfer remains unchanged with a tight interval, reject compilation mismatch as the main explanation. ([train_successor.py:35](probes/train_successor.py:35), [program.py:37](src/isingfold/rl/program.py:37)) |
| **4** | Test generalization through physical invariances and controlled data expansion. | Corrected cut/load features, relabeling controls and randomized lineage coverage recover at least +0.03 held-out selection gain. | First perform relabeling and repeated-pool tests. Then compare randomized 200/800-lineage subsets with matched training convergence. If reliable labels coexist with a tight upper bound below +0.03, stop this surrogate design. That kills this scaling investment, not learnability in principle. The existing curve only covers lexicographic subsets of one cache. ([successor_scorer.py:172](probes/successor_scorer.py:172), [train_successor.py:153](probes/train_successor.py:153)) |
| **5** | Build the actual competitive benchmark: tuned baselines, fresh lineages, exact time budgets and separate feasibility/quality endpoints. | Any learned advantage survives stronger minorminer budgets, a nonlearned constructor and singleton-case matching baselines. | Before substantial training, run those baselines on a small certified set at several fixed deadlines. If they solve the target regime within the intended policy budget, reject the proposed “where minorminer fails” claim or reformulate it as a speed claim. The present corpus has only seven test instances per host. ([pegasus6.log:11](results/fill/pegasus6.log:11), [zephyr4_budget.log:3](results/fill/zephyr4_budget.log:3)) |

I would execute the cheap checks in ranks 2, 3 and 5 before investing heavily in rank 1.

I agree that the present results do not justify a competitive learned quality embedder. I disagree that the suspects are exhausted: the compilation mismatch is real, the energy-scale experiment still uses the wrong default schedule for its theory, and the pool result is being promoted beyond its tested action distribution. ([STATUS.md:130](STATUS.md:130), [train_successor.py:39](probes/train_successor.py:39), [contracts.py:189](src/isingfold/rl/contracts.py:189), [pool_ceiling.py:123](probes/pool_ceiling.py:123))

What would change my view is a corrected, independently assessed scorer with a reproducible useful gain, or a constructor that beats tuned search on untouched instances at equal time. If both fail under those conditions, I would stop pursuing this particular learned formulation.