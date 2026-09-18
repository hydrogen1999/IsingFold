**I would not bet the week on terminal reward closing this gap in 100 updates. The smallest credible rescue is better initialization, followed by a short, controlled quality-training test.** The baseline defeats the current learned-quality claim, but it does not defeat the premise that resource count is an incomplete quality proxy.

This was a static review of `STATUS.md`, prior critiques, implementation and committed logs. No experiments or files changed. Submission-format checks were outside scope.

For **A**, reliable measurement is necessary, but it does not solve exploration or credit assignment.

Every decision in an episode receives the same leave-one-out advantage. A good early placement and compensating late growth therefore receive identical credit. The stated potential shaping cancels exactly. Furthermore, the large Pegasus residual gap corresponds to only about **0.0079 in bounded utility**, so occasional invalid training episodes can dominate differences between valid embeddings. Deployment validity of 1.00 does not establish perfect training-episode validity. [constructor_learning.py:159](/Users/nguyencongt/Documents/prj_IsingFold/probes/constructor_learning.py:159), [pegasus3_local_frozen_s0.log:2](/Users/nguyencongt/Documents/prj_IsingFold/results/quality_study/pegasus3_local_frozen_s0.log:2)

A shared preference for unnecessary growth could change quickly. Discovering coordinated replacements for poor placements is a substantially harder problem. The previous Zephyr quality run improved residual by only 0.004671, with one seed and attribution still unresolved. That makes modest improvement a reasonable forecast, not closure of a 0.05 gap. [STATUS.md:9](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:9)

The statement **“The mechanism is not mysterious”** overstates the evidence. Long chains accompany the loss; the comparison does not establish whether they arise from bad placement, unnecessary growth, or both, nor whether chain breaks explain the energy loss. [STATUS.md:1916](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:1916)

Run three diagnostic interventions on development instances:

- Compare the first valid workspace with the selected final embedding. Large deterioration after first validity implicates refinement or stopping.
- Prune policy outputs while preserving validity, then independently reassess them. Substantial removable excess and recovered quality implicate redundant growth. Failure to prune does not establish optimality.
- Compare equally advanced policy and teacher placement prefixes, followed by the **same frozen constructor**. A teacher-prefix rescue implicates placement. These are diagnostics, not empty-start method results.

**SHRINK exists, but is a weak undo mechanism.** It removes one qubit, preserves connectivity and every existing contact, and cannot remove a singleton placement. Ordinary rewrites are unavailable until every variable is placed. Undoing the Pegasus excess through SHRINK alone would require roughly 54 additional decisions, assuming those deletions are legal. [proposal.py:486](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/proposal.py:486), [proposal.py:1200](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/proposal.py:1200)

Sixteen slots is not a 3.1 percent action probability: legality, logits and quota redistribution matter. Measure offered deletions, distinct chains covered and selected probability. If recovery coverage is limiting, reserve **64 SHRINK and 64 REWRITE slots after placement**, using the vacated PLACE allocation. This cannot repair a geometry for which no useful deletion exists. [proposal.py:1250](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/proposal.py:1250)

My smallest substantive change is **replace unrestricted feasibility pretraining with cloning of successful, compact minorminer constructions on training lineages**, then introduce quality reward immediately. This supplies placement supervision before asking terminal reward to discover an entirely different construction strategy.

That is legitimate initialization. The resulting method is solver-initialized RL; deployment still starts empty, selects every action and never invokes minorminer for completion. Report teacher-generation cost, and compare against both the frozen clone and its continued-feasibility counterpart. Cloning alone beating the old policy would establish an initialization benefit.

Use several demonstrations per task and set-valued action labels. Require replay through ordinary deployment support, without teacher-dependent candidate injection. Critically, the existing cloning loop retains any nonempty trajectory, including unsuccessful prefixes. For this rescue, retain successful COMMIT trajectories only. [constructor_clone.py:40](/Users/nguyencongt/Documents/prj_IsingFold/probes/constructor_clone.py:40), [constructor_clone.py:149](/Users/nguyencongt/Documents/prj_IsingFold/probes/constructor_clone.py:149)

**Advance only if the cloned initialization reaches at least 95 percent deployment validity and comes within 0.01 residual of minorminer on validation.** These are proposed gates, not predictions. If teacher-compatible and incompatible placements have identical features, more cloning cannot distinguish them; inspect that before extending training.

Keep residual as the reward. However, the supplied calibration logs concern **fill 0.50 router/growth pools**, not this fill 0.30 policy distribution. Briefly check policy outputs under the proposed cap instead of assuming the same SNR transfers. [reward_pegasus3_f50.log:1](/Users/nguyencongt/Documents/prj_IsingFold/results/quality/reward_pegasus3_f50.log:1), [reward_channel.py:94](/Users/nguyencongt/Documents/prj_IsingFold/probes/reward_channel.py:94)

For **B**, matching resources is **stronger for testing whether construction quality matters beyond total qubits, but narrower as a deployment claim**.

Define each instance’s budget \(B_i\) using a frozen minorminer reference procedure before inspecting policy outcomes or assessment reads. Supply only the count to the policy. Train with the same cap convention, mask successors exceeding it, and independently validate COMMIT against it. Leave the quality reward unchanged: no qubit coefficient, unused-budget bonus or chain-length term. The rollout already requires agreement between feature and environment budgets. [constructor_rollout.py:120](/Users/nguyencongt/Documents/prj_IsingFold/probes/constructor_rollout.py:120), [validate.py:321](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/validate.py:321)

Do not impose the tight cap only at evaluation and interpret failure as failure of quality learning. That primarily tests distribution shift in construction.

Also distinguish:

- **Equal caps:** the policy may use fewer qubits. This establishes performance at no greater resource use.
- **Exactly equal expenditure:** both accepted outputs use \(B_i\). Declare that separately; do not pad or prune outputs afterward.

Equal qubit count still permits different chain distributions and contacts, so call this objective-guided **construction**, unless further controls isolate placement specifically.

A minorminer-derived cap uses instance-specific solver information. It is acceptable for this controlled experiment. An independent deployment claim additionally needs an externally specified budget rule, or accounting for cap-generation cost. Retain the original common-cap comparison.

With equal validity, the deciding number is:

\[
\Delta r=\operatorname{mean}_i\bigl(r_{\text{policy},i}(B_i)-r_{\text{minorminer},i}(B_i)\bigr).
\]

My proposed success gate is **\(\Delta r\leq-0.005\), a paired 95 percent interval below zero, no deployment-validity loss, and superiority over the strongest admissible growth control too**. Require an additional quality-training benefit over the frozen and continued-feasibility controls. Bootstrap instances, not reads.

If the cap collapses validity, favorable residuals on surviving instances do not establish a deployment win. Report failure-inclusive utility and solve probability alongside conditional residual.

For **C**, my single most likely outcome is: **continued feasibility stays near the frozen policy; quality training modestly improves residual but still loses to minorminer.**

| Observed outcome | What it should trigger |
|---|---|
| Quality improves over both frozen and continued feasibility, but still loses to minorminer | One bounded initialization rescue. Report objective adaptation, not competitive superiority. |
| Both training arms improve similarly | Attribute progress to continued training or adaptation. The quality reward has not earned credit. |
| Quality shows no useful improvement or worsens | Inspect valid-episode advantages and reachable alternatives. Do not automatically buy another 80 identical updates. |
| Conditional residual improves while validity falls | Judge failure-inclusive performance first. Do not select only successful instances. |
| Residual improves but solve probability does not | Make an expected-energy claim only, if that endpoint was fixed before final testing. |
| Quality beats the strongest baseline and both learning controls | Replicate across three seeds and fresh lineages, then test transfer and sampler robustness. |

**Improvement that still loses is publishable evidence**, especially if it explains when objective training helps and why competitive construction remains difficult. Alone, it is unlikely to support the intended main-track algorithm claim. It belongs in a controlled empirical study or a narrower proof-of-concept paper. The implementation plan already distinguishes residual improvement from solve-probability improvement. [2026-09-18-quality-first-implementation.md:153](/Users/nguyencongt/Documents/prj_IsingFold/docs/plans/2026-09-18-quality-first-implementation.md:153)

For **D**, the strongest honest abstract sentence is:

> Across the tested Pegasus and Zephyr ensembles under a fixed classical simulated-annealing protocol, inexpensive energy measurements select higher-quality minor embeddings than resource-count rules, while successful learned construction from empty does not by itself deliver competitive solution quality.

This supports a main-track **empirical study of embedding objectives and measurement budgets**. The ranking and selection evidence supports that direction; it does not establish that learning is generally unnecessary. [STATUS.md:963](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:963), [STATUS.md:1823](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:1823)

The following ranking includes one rescue attempt and the **three experiments that would complete that empirical case**. Estimates are planning allowances in core-hours, excluding queue delays. The committed pilot records six episodes per instance and four instances per update, rather than four episodes. At its 45-second training cap, 100 updates allow about 30 construction core-hours per run before labels and evaluation. [pegasus3_local_continued_feasibility_s0.log:1](/Users/nguyencongt/Documents/prj_IsingFold/results/quality_study/pegasus3_local_continued_feasibility_s0.log:1)

1. **Diagnose placement versus growth and finish the current pilot.** First-valid comparisons, pruning, prefix interventions and actual SHRINK coverage on 12 development instances per host. **8–16 core-hours**, using a few reallocated apollo workers. Measure chain breaks rather than assuming their role.

2. **Run the cloning rescue with matched controls.** Successful training-only demonstrations, frozen-clone evaluation, then 20 quality and feasibility updates under the declared cap. **30–60 core-hours**, one goose allocation. Apply the validity and residual gates above. Only a passing configuration earns three-seed replication, approximately **350–500 additional core-hours** including evaluation.

3. **Experiment 1: confirm selection gains on fresh common pools.** Sixty fresh lineages per host; identical router pools; first, random, fewest-qubit, shortest-chain and measured selection; independent assessment. Include equal-qubit strata. **40–80 core-hours**, goose. Failure to beat resource selection confines the existing result to its development distributions.

4. **Experiment 2: measure the deployment cost frontier.** Compare resource selection, uniform measurement and successive halving at three fixed total budgets, charging proposals and selection. Let resource selectors reinvest saved time. **60–100 core-hours**, goose. If measurement loses after cost accounting, withdraw the efficiency claim.

5. **Experiment 3: test generality without retraining.** Add one independently generated logical-graph family and two predeclared sampler/strength conditions on both topologies. **40–80 core-hours**, the second goose allocation. If gains disappear, restrict the claim to the original ensemble and operating protocol.
