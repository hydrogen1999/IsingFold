**Spend the week separating quality learning from validity learning at Pegasus 3, fill 0.30.** Keep the learned constructor as the target. Today’s evidence credits continued feasibility training, however, and round six’s prediction that it would remain near the frozen policy was wrong. The comparator is now the matched continuation, currently 0.0633 residual. [review-six:58](/Users/nguyencongt/Documents/prj_IsingFold/docs/review/2026-09-18-codex-gpt6-astra-review-six-closing-the-quality-gap.md:58), [STATUS.md:2176](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:2176)

This is a static review of STATUS, including every September 18 section, recent reviews and relevant code. No experiments ran; submission-format checks were outside scope.

For **A**, I rank **(iii), explicitly reformulated as conditional quality optimization, then (i), then (ii)**. Calling uncorrected (iii) an unbiased baseline for the existing utility is wrong.

**(iii): Change the objective honestly.** For one instance, let \(V\) denote validity, \(p=P(V=1)\), and \(\mu=E[U\mid V=1]\). The present objective is

\[
J=p\mu,\qquad \nabla J=\mu\nabla p+p\nabla\mu.
\]

Even with the exact valid-episode mean, subtracting it only on valid episodes removes the \(\mu\nabla p\) term. That **biases the gradient of the original objective**. Leaving the current episode out of the mean does not repair outcome conditioning.

**An unbiased version exists:** estimate and restore the missing term, using independent estimates where products require them. This retains validity learning and does not automatically eliminate its variance. Outcome-dependent control variates require a correction. [Tucker et al.](https://proceedings.mlr.press/v80/tucker18a/tucker18a.pdf)

There is also an unbiased estimator for the **different, conditional objective**: collect a fixed number of independent valid trajectories from an unchanged policy, and use their leave-one-out centered returns with their trajectory scores. Its expectation is \(\nabla\mu\).

That is my recommendation: optimize conditional measured residual, with completion controlled separately. Count every failed attempt, retain feasibility updates when completion deteriorates, and assess every deployment. This preserves your scientific target while changing the troublesome scalarization. It must be documented as an objective change.

**(i): Cheapest implementation, uncertain exploration.** Lower temperature can suppress failures, but can also concentrate on an invalid greedy trajectory or eliminate useful variation among valid embeddings. Furthermore, the flag changes training temperature while evaluation remains at 1.0. The gradient is on-policy for the colder distribution, not the deployment distribution. [constructor_curriculum.py:914](/Users/nguyencongt/Documents/prj_IsingFold/probes/constructor_curriculum.py:914)

Your amplitude calculation implies roughly **99.994% validity** before validity noise falls below 0.0079, even before the finite-group baseline penalty. A few all-valid batches cannot establish that. Nor does the calculation establish that learning is mathematically impossible: return variance is not gradient variance, and the policy-to-minorminer gap is not a measured within-policy gradient signal.

**(ii): Useful curriculum, weak standalone variance fix.** A gate fixed from earlier observations avoids selecting gradients using their own outcomes. But multiplying gradients by an instance gate changes instance weighting. It is unbiased for that frozen, gated objective, **not automatically for the original population objective**. Hard exclusion prevents importance correction for excluded instances.

The existing machinery is also less than you describe: one aggregate assisted-success statistic lowers one prefix fraction. It does not maintain per-instance mastery of empty-start construction. [constructor_curriculum.py:750](/Users/nguyencongt/Documents/prj_IsingFold/probes/constructor_curriculum.py:750), [constructor_curriculum.py:789](/Users/nguyencongt/Documents/prj_IsingFold/probes/constructor_curriculum.py:789)

**Smallest useful decision experiment:** first calibrate eight training instances, sixteen episodes each, at temperatures 1.0, 0.7, 0.5, 0.3 and 0.15. Measure distinct valid embeddings twice with independent 256-read blocks. Report validity and noise-corrected residual variation **within each instance**. Establish gates from earlier batches, then test them on fresh episodes.

Repair the calibration first. It currently pools residuals across instances, omits the utility’s instance-specific divisor, and reports infinite signal-to-noise whenever observed failures are zero, even if useful variation vanished. [temperature_calibration.py:77](/Users/nguyencongt/Documents/prj_IsingFold/probes/temperature_calibration.py:77), [constructor_objective.py:109](/Users/nguyencongt/Documents/prj_IsingFold/probes/constructor_objective.py:109)

Then compare twenty updates of the three fixes against matched continued feasibility, from identical weights. The deciding number is

\[
G=\operatorname{mean}_i(r_{\mathrm{feasibility},i}-r_{\mathrm{quality},i}).
\]

Advance a fix at **\(G\ge0.005\)** with a positive paired interval and unchanged deployment coverage. Low advantage variance alone does not win this comparison.

For **B**, the leading hypothesis is **external forcing across weak internal cuts**. The resource interventions do not establish that length never matters, and the breaking correlation does not establish a causal quality lever. STATUS explicitly records that reducing breaking failed to deliver the predicted residual improvement. [STATUS.md:2151](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:2151)

Measure these quantities, in this order:

- **Weakest cut margin:** internal ferromagnetic coupling across a cut compared with the fields and external coupling load on its two sides. Start with bridge cuts and small cuts.
- **Load relative to strength:** maximum and RMS external load per physical qubit, divided by actual chain coupling; concentration of load on individual attachment points.
- **Contact geometry:** coefficient-weighted contact multiplicity, whether competing couplings attach across the same bottleneck, and their signed structure.
- **Internal connectivity:** bridges with their separated loads, articulation points, alternative paths, cycle redundancy and diameter.
- **Actual programmed scale:** compiler scaling and effective \(\beta F\). Fixed nominal strength need not mean fixed physical strength. [program.py:134](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/program.py:134)

Cut-based embedding bounds motivate these features, but do not certify finite-anneal break probabilities. [Fang and Warburton](https://link.springer.com/article/10.1007/s11128-020-02681-x)

The current representation has specific omissions worth testing. Its bridge channel is a normalized load fraction, rather than a strength margin. Its signed channel explicitly does not estimate frustration. It pools chain summaries, potentially hiding one dangerous chain. [constructor_physics_features.py:9](/Users/nguyencongt/Documents/prj_IsingFold/probes/constructor_physics_features.py:9), [constructor_physics_features.py:169](/Users/nguyencongt/Documents/prj_IsingFold/probes/constructor_physics_features.py:169)

**Cheapest supervised test:** reuse the 180 embeddings without new anneals. Use five folds grouped by the twenty underlying instances, keeping every chain and variant of an instance together. Compare regularized linear prediction and one shallow nonlinear predictor. Exclude singleton chains from the primary score; predicting their deterministic zero breaks is trivial.

Compare length/degree/load controls, the exact existing chain summaries, and the richer cut features. Score variation across alternative embeddings of the same logical chain. Separately predict embedding-level breaking from the **actual pooled actor observation**. Success with unpooled features does not establish that the actor receives the information.

The existing break probe uses eleven custom features, not the actor’s physics observation, so its advertised representation conclusion is unsupported. [break_predictors.py:40](/Users/nguyencongt/Documents/prj_IsingFold/probes/break_predictors.py:40), [break_predictors.py:214](/Users/nguyencongt/Documents/prj_IsingFold/probes/break_predictors.py:214)

My proposed screening thresholds are:

- **Information available:** held-out within-instance rank correlation at least **0.60**, lower bootstrap bound above **0.30**, and improvement of at least **0.20** over the control.
- **Evidence of missing representation:** richer features pass, while the actual observation’s upper bound remains below **0.30**.
- **Otherwise:** unresolved.

Bootstrap instances. If only the nonlinear predictor passes, actor expressiveness remains implicated. Even a complete pass does **not** prove that credit assignment is the sole problem: terminal information may be unavailable at early placement decisions. No honest threshold establishes your proposed binary conclusion.

Also inspect which breaks matter energetically. Breaking an unloaded tip can change unanimity without changing decoded energy. Keep break prediction as a diagnostic, not the replacement reward.

For **C**, today supports **feasibility-trained construction plus measured selection over its pool** as the description of what works. It does not yet establish that this learned pool beats the strongest classical pool. Keep quality reward as the experiment that could change that account.

The single result that flips attribution is a replicated positive \(G\) under the preceding gate, against continued feasibility using the same initialization, observation, selection protocol and training budget. Three training seeds and fresh lineages are necessary given today’s reversals. The current four-instance 0.0633 is a reference, not a universal target. [STATUS.md:1898](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:1898)

For **D**, run **conditional residual training on Pegasus 3, fill 0.30, short-chain cell, 128 available qubits and roughly thirty logical variables**. This is the cell with working construction and the mechanism data. Moving immediately to fill 0.50 reintroduces the unfinished size ladder. Congestion remains zero-valid after six hours. [STATUS.md:1849](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:1849), [STATUS.md:2230](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:2230)

Use physics32, zero-padded from the same feasibility checkpoint in every training arm. Use the pre-cloning checkpoint so witness imitation does not confound attribution. Compare conditional quality, the existing mixed utility, continued feasibility and frozen weights. Keep support, initialization, optimizer and episode limits identical; count conditional sampling rejections and report extra labeling cost.

Evaluate with 300 seconds, at most eight 256-read selection blocks, and 4,096 independent assessment reads under the registered strength and schedule. Include tuned anytime minorminer and its strongest growth/selection control, allowing fast baselines their full proposal allocation. Retain failure-inclusive solve probability and coverage.

**Success for quality learning:** \(G\ge0.005\), positive paired interval, consistent direction across three seeds, and no coverage loss. If continued feasibility repeats 0.0633, that means approximately **0.0583 or lower**, assessed on the same instances.

**Success for the intended competitive claim:** additionally beat the strongest external control by at least **0.005 residual**. Beating continued feasibility alone does not earn that claim.

**Kill this configuration:** after twenty updates, the upper confidence bound on \(G\) is below **0.002**, or deployment coverage deteriorates by more than two percentage points. A wide interval is inconclusive, not a negative result.

Stop further witness cloning aimed at compactness, qubit-restraint interventions, unchanged mixed-reward extensions, architecture sweeps and congested launches without intermediate rungs.

Ranked spending, in CPU core-hours, excluding queue delays:

1. **Repair and run the temperature comparison:** **8–16 hours on apollo**.
2. **Test exact observation sufficiency on cached embeddings:** **1–3 hours on apollo**, alongside calibration.
3. **Run the twenty-update comparison of variance fixes and controls:** **40–70 hours on goose**.
4. **Replicate only the surviving configuration across three seeds and sixty fresh test lineages:** **200–350 hours on goose**. Reserve second-topology work for a passing attribution result.
