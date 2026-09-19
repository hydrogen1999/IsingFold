**My best one-week bet is search-assisted policy improvement, preceded by a cheap test of the linear actor’s expressiveness.** Physics32 should be the shared starting observation. Neither a large network nor deeper search has yet earned its cost.

I read STATUS, the September 18 reviews, and relevant implementation. This checkout has no explicitly dated September 19 section, so I treat your newer measurements as supplied. No experiments or edits ran; submission-format checks were outside scope.

For **A**, my leading diagnosis for high fill is **search and credit assignment**, with **linear capacity the first alternative to test**. For quality, the historical representation omission is established, but adding physics32 does not establish that the remaining problem is solved.

Reachability establishes that successful decisions exist along the replayed paths. It does not establish that the observation distinguishes them, that one weight vector can order them, or that sampled trajectories discover them. [STATUS.md:1802](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:1802)

There is a specific capacity concern. A linear scorer can reward cycle redundancy directly. It cannot freely condition that preference on congestion, remaining demands, or combinations of bottleneck features. Moreover, physics32 pools chain summaries and describes immediate successors. At early placements, a future useful cycle may leave no distinctive current signal. [constructor_physics_features.py:176](/Users/nguyencongt/Documents/prj_IsingFold/probes/constructor_physics_features.py:176)

**The cheapest test is a linear feasibility problem over recorded witness decisions.**

Cache the exact legal candidate rows along several successful training-lineage replays. Ask whether one weight vector can satisfy `score(chosen) >= score(alternative) + 1` at every decision. This requires no annealing or RL.

Then fit the existing small MLP to the same records. Its implementation already exists. [constructor_curriculum.py:464](/Users/nguyencongt/Documents/prj_IsingFold/probes/constructor_curriculum.py:464)

| Observation | Interpretation and action |
|---|---|
| Linear ordering is infeasible; MLP reproduces the paths | Capacity is implicated. Use the small nonlinear scorer. |
| Actions with demonstrably different continuation outcomes have identical rows | Representation is insufficient. Add action-dependent residual-host and unplaced-neighbor context. More hidden units cannot separate identical inputs. |
| Linear fitting reproduces successful paths, but RL cannot learn those decisions | Exploration or credit assignment is implicated. Change the training technique. |
| Supervised fitting succeeds on recorded states, but deployment fails after deviations | Recovery and training-state coverage are implicated. Collect search targets on learner-generated states. |

The qualification matters: infeasibility proves inability to reproduce **that strict witness ordering**, not inability to construct any valid embedding. Account for interchangeable successful moves before declaring a capacity failure. Report whole-path completion, since respectable per-step accuracy can still produce almost no complete trajectories.

For **B**, **yes, policy-guided lookahead followed by training toward improved decisions is the right technique to test for both problems**. This is the policy-improvement loop formalized by Expert Iteration: search discovers better decisions, and the network learns to reproduce and guide them. It does not guarantee improvement here. [Anthony, Tian and Barber](https://arxiv.org/abs/1705.08439)

The present constructor samples one categorical action and receives terminal feedback. There is no intermediate planning loop. [constructor_rollout.py:172](/Users/nguyencongt/Documents/prj_IsingFold/probes/constructor_rollout.py:172)

**The cost rules out ordinary broad tree search.** Assuming each newly expanded state requires another 0.53-second decision-generation pass:

| Work at one construction decision | Approximate serial cost |
|---|---:|
| Generate and score the current candidate set | 0.53 seconds |
| Expand all 512 successors | 271 seconds |
| Expand two complete additional levels | 38.7 core-hours |

The 512 current rows are already scored together. The multiplication applies to generating successor states and their candidate sets, not to 512 neural-network evaluations. A value head using already available successor summaries could be much cheaper and should be timed separately.

A 107-decision trajectory costs about 57 seconds at 0.53 seconds per decision. A 300-second allowance leaves approximately four additional expansions per decision. Using the recorded 80.7-second replay instead leaves fewer than three, before measurement overhead. [STATUS.md:1602](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:1602), [STATUS.md:1808](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:1808)

Start with **two surviving branches and depth two at selected decisions**, governed by a total expansion budget. Spend more effort at consequential placements; reuse the chosen subtree. Deeper search belongs at a few branch points or in offline teacher generation.

This requires cached state branching. The older limited-discrepancy experiment already failed while fitting only one to three passes into its deadline. Replaying prefixes from empty would repeat that failure. [STATUS.md:397](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:397)

The value estimate must predict:

- Probability of completing within the remaining budget.
- Expected measured residual conditional on completion.

Train these from actual continuation outcomes, including failed attempts, and combine them using the declared utility. Completed training trajectories provide 256-read labels; internal search nodes use the learned estimate. Four full continuations of roughly 100 decisions already cost about 212 seconds plus 1,024 reads, so this cannot happen at every deployment decision.

A structural predictor around 0.5 is defensible as a **weak search prior or auxiliary prediction target**. It is not an accurate future-value estimate, a pruning certificate, or a replacement objective. Search can amplify its errors by selecting extreme predictions. Preserve exploration outside its favorites and judge its selections using independent measured residual.

For **C**, my ranking for high fill is:

| Rank | Technique | Concrete version and limitation |
|---|---|---|
| 1 | Reverse-start curriculum from successful trajectories | Save legal forward prefixes. Train forward completion with 1, 2, 4, 8, 16, then more decisions remaining. Advance after at least 80% completion across development instances; retain empty-start episodes. This supplies successful high-fill experience immediately. |
| 2 | Cached backtracking with a learned ordering | Save earlier placement states and unexplored alternatives. Backtrack to a relevant placement when continuation fails. Compare against independent restarts at equal total time. |
| 3 | Separate size and fill ladders | Increase logical size while space remains generous, then increase occupancy at fixed size. Make host restrictions explicit and retain comparable graph structure. This is useful training support, but does not resolve an unexpressible decision rule. |
| 4 | A different action space | Introduce atomic relocation or cycle-closing moves only if useful changes require more coordinated steps than affordable search can see. Factorization alone does not supply missing spatial information. |

“Reverse trajectories” should mean learning forward completion from progressively earlier legal states. Literal inverse actions need not exist: SHRINK preserves existing contacts and cannot remove singleton placements. [proposal.py:486](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/proposal.py:486)

The reverse-start version is more targeted than another diagonal jump from fragments to 100 variables. It exposes the terminal congestion that the current ladder misses. Your stronger Pegasus fill-0.50 results justify extending that ladder; the weaker Zephyr results argue against promoting both topologies together.

For a bounded high-fill pilot, use twelve unseen instances per topology under 300 seconds. **Advance at six completed deployments out of twelve, with a clear advantage over equal-time restarts. Stop that configuration at one or fewer out of twelve.** Intermediate results remain preliminary.

For **D**, terminal policy gradient can learn this in principle. **I would not allocate the week to terminal policy gradient alone.** Its current leave-one-out estimator gives all decisions in a trajectory the same advantage, so observing a useful terminal structure does not identify which early choice enabled it. [constructor_learning.py:159](/Users/nguyencongt/Documents/prj_IsingFold/probes/constructor_learning.py:159)

Directly teaching the network to reproduce a cycle count already present in its input adds little. Directly rewarding “more cycles” changes the objective and can encourage unnecessary growth. Your evidence concerns redundancy **at fixed length**, not unrestricted cycle accumulation.

I favor **supervision of action comparisons**, with structure determining which comparisons are informative:

- Branch from the same construction prefix into alternatives differing in redundancy or bottlenecks.
- Match chain length and total expenditure where feasible, and record changes in contacts and load.
- Complete both branches with the same continuation policy and matched continuation randomness.
- Teach the preference supported by measured terminal residual. Use measured per-chain breaking as an auxiliary target, if useful.

This is not a resource penalty: neither fewer qubits nor more cycles automatically earns preference.

The sentence “This explains the puzzle” currently goes beyond the evidence. Fixed-length associations motivate an intervention; they do not establish that adding redundancy causes the desired residual improvement. The earlier interventions already reduced breaking without the predicted energy benefit. [STATUS.md:2292](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:2292), [STATUS.md:2159](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:2159)

Use the branch-comparison data to test the mechanism and train the method together. There is no need for another detached correlation campaign.

For **E**, the single highest-value method change is **replace blind episode-level policy improvement with cached branch-and-compare supervision**. The expressiveness test comes first because it determines whether the student can learn the search’s decisions.

The smallest useful quality experiment is Pegasus 3, fill 0.30:

- Twelve training instances, four prefixes each, four competing actions, two continuations per action: **384 continuation attempts**.
- Include policy-preferred, structurally different, and exploratory alternatives.
- Measure completed continuations with 256 reads, fit the action preferences and value estimate, then test on newly encountered states.
- Compare against physics32 terminal-RL and continued-feasibility controls with matched training budgets. Also evaluate frozen weights with the same search, so additional planning cannot receive credit as learning.
- Freeze the chosen configuration before the thirty-instance held-out deployment comparison. Preserve the common 300-second deadline, selection-read allowance, and independent assessment. Include tuned minorminer.

Define improvement as comparator residual minus new-method residual.

**Success:** at least **0.017 improvement over the strongest matched learning control**, a positive paired 95% interval, at least 29/30 valid deployments, and no decrease in deployment solve probability. If the comparator remains 0.0647, the residual target is approximately **0.0477**. Competitive superiority additionally requires beating minorminer.

**Kill this configuration:** the improvement’s upper confidence bound is below **0.017**, or validity falls to 27/30 or worse. A wide interval is unresolved. Thirty instances are a practical screen, not guaranteed power across the supplied noise range. [STATUS.md:2338](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:2338)

Stop unchanged high-fill launches, compactness cloning, resource-restraint interventions, temperature sweeps justified by the withdrawn validity-noise story, and selecting favorable intermediate evaluations. Round seven’s 0.005 gate and attribution claim have been superseded. [review-seven:39](/Users/nguyencongt/Documents/prj_IsingFold/docs/review/2026-09-18-codex-gpt6-astra-review-seven-variance-and-representation.md:39), [STATUS.md:2305](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:2305)

Ranked spending for the week, in CPU core-hours, excluding queue delays:

1. **Expressiveness audit on apollo: 2–4 hours.** Cache successful paths, solve the linear ordering problem, compare the existing small MLP. Allow half a day.
2. **Branch-and-compare quality pilot on goose: 40–70 hours.** Includes matched controls and the thirty-instance evaluation. Allow two days to implement branching and accounting.
3. **Reverse-start high-fill training plus cached backtracking on goose: 90–130 hours.** One bounded pilot per topology. Twenty updates × 24 episodes × 300 seconds permits 40 training hours per topology.
4. **Replicate only a passing quality configuration: 80–140 hours on goose.** Second seed, fresh lineages, frozen-search attribution, and the strongest external baseline. Total planned allowance: approximately **210–345 core-hours**.
