**The best one-week bet is a learned proposer with a fixed measured selector.** The missing result is a quality advantage attributable to training that survives the strongest equally budgeted proposal baseline.

I read `STATUS.md` in full, the prior reviews, and relevant code and logs. No numerical work ran. Submission-format checks were outside scope.

For **A**, the abstract sentence to earn is:

> A reinforcement-learned constructor produces embedding pools that improve independently assessed ground-state success on unseen roughly 50-variable Pegasus problems, outperforming tuned anytime minorminer under matched time, qubit, and measurement budgets.

This is a prospective claim. The witness perturbation establishes sensitivity, not available improvement over minorminer’s best pool. The replicated quality evidence points to Pegasus 3 at named fill **0.50**, not 0.90. [STATUS.md:1425](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:1425)

Run one decisive experiment:

- **Cell:** Pegasus 3, 128 available qubits, named fill 0.50, approximately 50 logical variables. Report actual instance sizes and returned occupancy.
- **Protocol:** empty-start construction; 300 seconds per deployment on identical CPU allocations; eight distinct measured candidates; 256 selection reads each; registered strength and 200 sweeps; 4,096 fresh assessment reads. Use mean decoded energy for deployment selection.
- **Controls:** tuned minorminer; minorminer with random and coupling-guided growth, including zero growth; a diversity-oriented baseline described below; the frozen feasibility checkpoint; continued feasibility training with matched updates.
- **Replication:** three training seeds and 60 fresh test lineages, locked after development. Report failures as zero deployment success and show conditional quality separately.
- **Success:** at least **+0.020 absolute solve probability** over the strongest control, positive paired 95 percent interval, positive direction for every training seed, and at least 95 percent deployment validity. Quality training must additionally contribute **+0.010** over continued feasibility training.
- **Kill this week’s claim:** the advantage’s upper confidence bound is below **+0.010**, or quality training’s increment has an upper bound at or below zero. Intermediate outcomes remain unresolved.

These are proposed decision thresholds, not power guarantees. A single embedding beating one router draw is insufficient. Keep K=1 as an attribution diagnostic.

For **B**, useful diversity can constitute learned quality: the policy may learn a distribution whose upper tail makes limited measurement more valuable. But “stochastic construction plus selection” is already a strong alternative explanation.

The decisive control is **diversity-matched, nonlearned proposal generation**. Let minorminer restart, produce random or heuristic growth variants, and form an eight-program shortlist using structural distance. Match per-instance qubit expenditure, chain-length distributions, and pairwise structural distances as closely as feasible, without consulting assessment outcomes. Also tune the feasibility policy’s sampling temperature on validation to match the quality policy’s diversity.

Cross each proposer with uniform selection and the identical measured selector. Independently assess the unselected candidates as well as the selected winner. Report:

- Mean candidate quality.
- Candidate-quality variance and duplication.
- Selected quality at K=1, 2, 4 and 8.

A positive mean-quality difference supports “better typical candidates.” If the mean difference is established equivalent within **±0.005**, while best-of-eight improves, the contribution concerns the tail or dependence between proposals. Merely obtaining a nonsignificant mean difference does not establish equivalence.

The deciding learned contribution is **at least +0.010 selected solve probability**, with a positive interval, over both the diversity-matched control and temperature-tuned feasibility policy. If those controls reproduce the gain within ±0.005, the evidence supports selection exploiting readily available variation.

Structural distance matching alone cannot settle this: structural diversity and quality diversity are different quantities.

For **C**, **no new target cell has yet established that a 256-read training reward is reliable**. Pegasus 3 at fill 0.50 is the right candidate. Its replicated perturbation used 4,096 reads; an aggregate significant effect does not establish reliable within-instance episode rankings at 256 reads. [mech2_pegasus3_d32.log:1](/Users/nguyencongt/Documents/prj_IsingFold/results/frontier/mech2_pegasus3_d32.log:1)

The direct favorable residual-reliability evidence comes from the small fragment pilot, whose median variance signal-to-noise ratio was 11.3. That cannot be transferred automatically to 50-variable, matched-spend proposals. [snr_P_lin230.log:6](/Users/nguyencongt/Documents/prj_IsingFold/results/diag/snr_P_lin230.log:6)

**Solve probability should be the provisional reward at fill 0.50**, because it matches the desired claim and was the stronger observed discriminator. Settle this cheaply before training:

Take 12 development lineages, eight ordinary constructor outputs per lineage, and eight independent 256-read blocks per output. Include matched-spend alternatives. Every block supplies both hits and residual, so comparing channels requires no additional sampler calls. Estimate measurement noise and between-candidate variation **within each instance**. Rank using one block and assess against disjoint blocks; compare 256-read and aggregated 1,024-read measurements.

Advance a channel when its median variance signal-to-noise ratio exceeds one and its rankings produce positive independently assessed solve-probability gains. Prefer residual only if this experiment shows that it provides better supervision for the final solve-rate objective. If neither works at 1,024 reads, kill that training configuration.

Keep qubits as a constraint, not a reward penalty. Ground-state hits require a known optimum; training and assessment may use certificates. Deployment can select by mean decoded energy, whose ordering equals residual ordering within an instance without requiring the optimum. [evaluator.py:131](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/evaluator.py:131)

For **D**, these are the three most dangerous attacks.

**1. “You handicapped the fast baseline.”** The current evaluator stops after reaching its measurement cap. Consequently, a six-candidate comparison is not unrestricted anytime competition, even when both arms nominally receive the same deadline. [constructor_protocol.py:95](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/constructor_protocol.py:95)

Keep equal-K results to isolate proposal quality. The headline must also survive minorminer using the entire deadline for restarts and inexpensive shortlist construction. Under the same 2,048-read allowance, let validation choose among eight×256, sixteen×128, thirty-two×64, and successive halving. Give the learned system the same options.

Tune restart and chain-improvement patience, which are separate API controls. Charge feature extraction, inference, compilation, selection and failed attempts; discard late answers. Report training separately and compare deployment quality at 60 and 300 seconds. [Minorminer API](https://docs.dwavequantum.com/en/latest/ocean/api_ref_system/generated/minorminer.find_embedding.html)

**Answering evidence:** the learned advantage survives this stronger deployment comparison, rather than only equal candidate counts.

**2. “Fifty variables on a tailored distribution is too narrow.”** Fifty to 100 variables is not inherently disqualifying for a method paper. One planted cell with a small gain would nevertheless leave the contribution fragile.

The minimum persuasive extension is frozen-policy quality transfer to a second topology and an independently generated logical-graph family, plus a size curve through roughly 25, 50, 75 and 100 variables. Preserve the endpoint when it becomes unresolved. Do not enlarge the host and call that logical scaling: the full-host validation result contains four 24-variable tasks. [ink24_pegasus16_prefixinit_s0.log:21](/Users/nguyencongt/Documents/prj_IsingFold/results/inkdrop/ink24_pegasus16_prefixinit_s0.log:21)

**Answering evidence:** quality gains beyond the training distribution, strong attribution controls, and explicit operating limits. Feasibility transfer alone does not answer this attack.

**3. “You optimized a simulator peculiarity.”** Concede the absence of evidence about QPU quality. The evaluator explicitly invokes classical simulated annealing. [evaluator.py:89](/Users/nguyencongt/Documents/prj_IsingFold/src/isingfold/rl/evaluator.py:89)

Assess frozen policies under a second declared depth, randomized spin-update order, and a second common strength rule. Randomized order matters because the sampler documentation identifies possible ordering bias in sequential updates. [Sampler documentation](https://docs.dwavequantum.com/en/latest/ocean/api_ref_samplers/generated/dwave.samplers.SimulatedAnnealingSampler.sample.html)

**Answering evidence:** a reproducible learned optimization method across these classical operating conditions. This strengthens a main-track method claim; it supplies no evidence of quantum advantage or improved physical-QPU outcomes.

For **E**, cut universal claims about unpredictability, zero solve probability, depth-invariant laws, and residual discrimination at fill 0.90. Cut the abandoned-method chronology from the main narrative. Keep the modern-corpus losses and growth controls visible: they directly constrain the claimed contribution. [STATUS.md:534](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:534)

Put detailed feasibility ladders, host-size timing, frontier calibration, support diagnostics and condensed negative experiments in the appendix. Measured selection needs enough space to specify the method and its baseline, not a separate benchmark-paper narrative.

Repair two reporting errors before using the ablation evidence. The script labels P/Z fragment stages as full Pegasus 16/Zephyr 15. The cold hardware logs contain only initialization evaluations, so they do not establish failure after matched training. [ablation_table.py:13](/Users/nguyencongt/Documents/prj_IsingFold/probes/ablation_table.py:13), [Z_lin230_s1.log:3](/Users/nguyencongt/Documents/prj_IsingFold/results/curriculum/Z_lin230_s1.log:3), [cold Pegasus log:3](/Users/nguyencongt/Documents/prj_IsingFold/results/inkdrop/ink24_pegasus16_prefix_s0.log:3)

Budget roughly **500–900 core-hours**, subject to successful-rollout timing. Use at most eight shared apollo cores; use one goose allocation for six single-core training jobs and the other for evaluation. Queue delays are additional.

1. **Audit the exact protocol and support.** About half a day of engineering and **1–3 core-hours** on apollo. Start with 64 candidates, but verify empty-start, unhinted deployment support: both committed narrow replay logs actually say `hint: true` and `deployment: false`. They do not establish the advertised deployment reachability. [Pegasus replay:1](/Users/nguyencongt/Documents/prj_IsingFold/results/control/replay_pegasus3_narrow.log:1), [Zephyr replay:1](/Users/nguyencongt/Documents/prj_IsingFold/results/control/replay_zephyr2_narrow.log:1)

2. **Calibrate reward and baseline budgets.** **8–16 core-hours**, apollo. Run the repeated-block experiment and tune the nonlearned controls. Freeze the primary endpoint, selector and practical margins.

3. **Run a paired learning pilot.** **20–40 core-hours**, apollo or goose. Warm-start the cheap 20-channel actor; compare 20 quality updates with continued feasibility training. Require adequate valid-pool coverage and evidence that quality training changes assessed quality.

4. **Replicate the surviving configuration.** **300–500 core-hours**, goose. Three quality seeds and three matched feasibility continuations. One hundred updates of 24 episodes capped at 60 seconds allow 40 core-hours per run before measurement and evaluation. Avoid an architecture sweep.

5. **Test transfer and sampler robustness without retraining.** **40–80 core-hours**, the second goose allocation. Prioritize second-topology quality and one independent graph family over more congested feasibility runs.

6. **Run the frozen final comparison once.** **100–200 core-hours**, goose. Evaluate the locked lineages against the strongest anytime and diversity controls, retain failures, and write the abstract claim only if both the deployment advantage and the quality-training increment pass.
