I would not recommend an oral yet. **My leading diagnosis is insufficient information for spatial decisions, compounded by weak credit assignment and limited recovery.** Wider support removes one obstruction; it does not establish learnability.

I read the full status record, both reviews, the constructor sources at `deee7ba`, and the relevant committed logs. This was a static review. No numerical work ran. Bibliography, fonts, anonymity, and page-limit checks were outside this record-and-code review.

For A, the causes rank as follows.

**1. The support conclusion remains overstated.** “The wide support contains the witness’s path” describes a narrower experiment than empty-start deployment. Replay supplies one **complete witness chain** at initialization and uses a witness-derived cap. It also leaves satisfied-growth disabled, whereas deployment enables it. Repeat the audit through the actual rollout configuration, including its first placement. [witness_replay.py:39](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/witness_replay.py:39), [constructor_rollout.py:117](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/constructor_rollout.py:117)

“Every frontier variable” also does not mean every necessary root. Roots remain truncated; sufficiently large frontiers can lose variables too. The purported whole-frontier test requires only 12 of 16 variables. Conversely, failure to follow one witness under narrow support never proved that *every* valid embedding was unreachable. [proposal.py:344](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/src/isingfold/rl/proposal.py:344), [test_wide_support.py:43](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/tests/unit/test_wide_support.py:43)

The local Pegasus-90 replay log still contains two completed instances out of three, so the supplied record does not verify the third. [replay log:1](/Users/nguyencongt/Documents/prj_IsingFold/results/diag/replay_pegasus6_fill90-a3.0_wide.log:1)

**2. The tiny representation cannot distinguish many consequential placements.** For free singleton PLACE alternatives, occupancy change is identical. Equal realized-edge changes therefore produce identical rows, including at the first placement. An MLP over those same rows cannot separate them either. Normalizing placement progress by variable count further weakens transferred progress weights relative to opcode biases. [constructor_tiny_gate.py:115](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/constructor_tiny_gate.py:115), [constructor_tiny_gate.py:155](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/constructor_tiny_gate.py:155)

My hypothesis is that it learns useful action preferences but cannot reliably preserve scarce future routing opportunities. Add count-scaled progress and local features measuring remaining neighbor demand, available roots, and competition for those roots. Global free-component summaries alone may still tie; their current implementation walks the residual host for each uncached summary. [constructor_features.py:170](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/constructor_features.py:170), [constructor_features.py:194](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/constructor_features.py:194)

**3. Six episodes give six return observations despite thousands of decisions.** With six siblings, the implemented advantage at every step is

\[
A_{it}=R_i-\frac{1}{5}\sum_{j\ne i}R_j.
\]

Actor terms are summed over the trajectory; potential shaping cancels from this advantage. The estimator is legitimate, but supplies no direct distinction between a helpful early placement and a later mistake. Equal returns give zero actor signal. Failures still receive terminal progress reward, so zero valid episodes does not necessarily mean zero learning signal; that signal may favor incomplete dead ends. [constructor_learning.py:76](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/constructor_learning.py:76), [constructor_rollout.py:214](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/constructor_rollout.py:214)

**4. Mastery can stall for two distinct reasons.** There is no advance if the first level never passes, and no mechanism to make that level easier. Moreover, the gate aggregates assisted and empty-start episodes. With four groups of six, one unsuccessful empty group limits otherwise perfect assisted performance to 18/24, below 0.8. Advancement then depends partly on the randomly drawn mixture. [constructor_curriculum.py:626](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/constructor_curriculum.py:626)

Gate advancement on assisted completion separately, over repeated prefixes. Start with one, two, four, and eight missing chains. Keep empty starts as a separate training component. Also, the fraction counts *witness qubits*: 90 percent of an 80-percent-filled witness starts near 72 percent host occupancy. [constructor_curriculum.py:358](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/constructor_curriculum.py:358)

**5. Wide support removes recovery actions.** Its quotas contain PLACE, ROUTE, GROW, and SHRINK only. Zero-quota families are excluded, so ordinary rewrite, repair, and RESTART are absent. RESTART’s initialization bias therefore has no effect in this configuration. SHRINK cannot remove a singleton and preserves already realized demands. This is a material change beyond support width. [\_context.py:33](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/_context.py:33), [proposal.py:1227](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/src/isingfold/rl/proposal.py:1227), [proposal.py:489](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/src/isingfold/rl/proposal.py:489)

A 500-candidate softmax is not inherently beyond a linear actor. It creates multiplicity bias and exploration dilution: one preferred row needs about an 8.4-logit advantage over 499 equal competitors to receive 90 percent probability. Factorizing action subtype, variable, and root can help allocation, but cannot recover missing information. The current rollout uses one categorical distribution over legal rows. [constructor_rollout.py:149](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/constructor_rollout.py:149)

**6. The deadline is plausible for the teacher, restrictive for exploration.** At the supplied costs, 500 decisions require 80–150 seconds; 584 require at most about 175 seconds. Thus 300 seconds is not inherently insufficient. However, 2,000 decisions require 320–600 seconds before additional overhead. Training can truncate detours well before its nominal horizon. The rollout counts reset, features, inference, and terminal compilation, and rejects late COMMITs. [constructor_rollout.py:142](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/constructor_rollout.py:142), [constructor_rollout.py:198](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/constructor_rollout.py:198)

**Behavior cloning should precede further RL.** Keep RL only if it improves over cloning under a matched budget.

For witness \(W\), define \(T(s)\) as all legal advancing candidates whose complete successors satisfy \(C_v(s)\subseteq C_v(s')\subseteq W_v\) for every variable, plus valid terminal COMMITs. Exclude no-ops and backward moves. Use

\[
L_{\rm BC}=-\frac1{N}\sum_{\tau}\frac1{|\tau|}
\sum_{s\in\tau}\log\!\sum_{a\in T(s)}\pi_\theta(a\mid s).
\]

This rewards total acceptable probability without requiring an arbitrary ordering. Single-pick cross-entropy is a useful ablation. Store full replacement chains and the positive mask: the current dump saves one teacher index and a union of added qubits, losing ownership information needed for general successor reconstruction. [witness_replay.py:78](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/witness_replay.py:78)

Cloning remains witness supervision, not a completeness guarantee. Off-witness actions can be valid; teacher-step accuracy cannot substitute for empty-start completion.

The cheapest deciding diagnostic is a four-task audit, two training tasks per topology: exact-deployment teacher replay, then six sibling rollouts from empty, one-missing-chain, and the current prefix level. Record acceptable-action mass, identical-feature classes, subtype counts, actual occupancy, six returns, and termination times. Teacher failure identifies remaining support/runtime problems; mixed positive/negative feature ties identify missing information; successful supervised fitting followed by rollout failure points toward distribution shift and recovery. Instrument the actual rollout: the current support diagnostic groups GROW and SHRINK together through their shared REWRITE opcode. [constructor_diagnostics.py:73](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/constructor_diagnostics.py:73)

For B, the strongest claim chain is:

- Measured selection improves simulated-annealing outcomes substantially more than resource-only selection on the tested pools. Its assessment oracle is a noisy reference, not a theoretical ceiling. [STATUS.md:800](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:800)
- Planted instances supply known-feasible construction challenges where substantial minorminer budgets often fail.
- Independent construction is learnable on small held-out instances. That does not yet establish a deployed advantage over strong nonlearned construction controls. [STATUS.md:402](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:402)
- One Zephyr fragment checkpoint beats measured minorminer-plus-growth by −0.008 [−0.015, −0.003], with equal **mean** qubit counts. [Zq log:19](/Users/nguyencongt/Documents/prj_IsingFold/results/curriculum/Zq_lin230_s1.log:19)
- **Unsupported links:** reliable large-instance construction, cross-topology transfer, application transfer, a causal benefit from quality RL, and improved QPU outcomes.

The three attacks most likely to sink the paper are these.

**First: “The learned contribution disappears against a fair heuristic.”** The Zephyr result partially answers this, but one seed and equal mean spend do not isolate placement learning. A final-versus-control interval also does not establish improvement over the initial checkpoint. Pegasus loses validity, and modern-corpus transfer loses against growth. [Pq log:19](/Users/nguyencongt/Documents/prj_IsingFold/results/curriculum/Pq_lin230_s1.log:19), [STATUS.md:519](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:519)

Twenty tries alone is inadequate as the headline baseline. Existing 200-try and 600-second results strengthen the benchmark, while disproving blanket statements about failure above 85 percent: Zephyr medium-chain coverage remains 0.83 at 85 percent and 0.33 at 90 percent. [budget log:14](/Users/nguyencongt/Documents/prj_IsingFold/results/fill/zephyr4_budget.log:14), [anytime log:51](/Users/nguyencongt/Documents/prj_IsingFold/results/fill/zephyr4_anytime600.log:51)

Use validation-tuned anytime minorminer with equal CPU allocation, deadline, cap, and measurement budget. Tune restart strategy and search patience, not merely tries. These are distinct controls in the [official minorminer API](https://docs.dwavequantum.com/en/latest/ocean/api_ref_system/generated/minorminer.find_embedding.html).

**Second: “You reverse-engineered the benchmark around your teacher.”** Planted fill is defensible as a controlled stress benchmark. A witness certifies feasibility and an upper bound on required qubits; it generally does not certify minimum fill or realistic application structure. Singleton witnesses make the qubit lower bound exact. Actual occupancy must account for discarded components. [gen_fill_corpus.py:42](/Users/nguyencongt/Documents/prj_IsingFold/probes/gen_fill_corpus.py:42), [gen_fill_corpus.py:128](/Users/nguyencongt/Documents/prj_IsingFold/probes/gen_fill_corpus.py:128)

The inspected generator retains successful plantings regardless of minorminer’s outcome, so “the generator filters for minorminer failures” is **refuted**. Selecting only failures afterward would define a conditional challenge set, unsuitable for estimating general superiority. Publish the whole registered distribution and identify any failure-conditioned subset explicitly. [gen_fill_corpus.py:143](/Users/nguyencongt/Documents/prj_IsingFold/probes/gen_fill_corpus.py:143)

**Third: “Neither the scale nor the quantum relevance follows.”** Hardware-sized hosts with 24 logical variables and four validation tasks establish an implementation milestone. They do not establish large-problem performance or Pegasus-to-Zephyr transfer. [STATUS.md:542](/Users/nguyencongt/Documents/prj_IsingFold/STATUS.md:542)

Frame quality as **decoded residual under a fixed classical simulated-annealing evaluator on compiled Pegasus/Zephyr programs**. The backend actually invokes `SimulatedAnnealingSampler`. The embedding-feasibility claim stands independently; QPU quality and quantum advantage remain untested. [evaluator.py:89](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/src/isingfold/rl/evaluator.py:89)

Reward reliability closes the broad noise explanation on the sampled tasks. It does not establish reliable gradients for small differences at matched spend. The reported SNR is a **variance ratio**, estimated across ordinary policy outputs. [constructor_diagnostics.py:151](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/constructor_diagnostics.py:151)

For D, cut the hybrid-completion narrative, retired scorer/contact-policy branches, tiny-gate chronology, and claims that a finite unsuccessful run establishes a learning ceiling. Keep the modern-corpus losses and growth controls prominently.

The result that would make me argue for an oral is **a frozen empty-start constructor achieving at least 50 percent coverage on unseen congested 300–500-variable instances on both topologies within 600 seconds, beating the strongest matched baseline by at least 20 percentage points with a positive paired interval, and retaining a demonstrated advantage beyond the planting distribution**. One isolated success would not suffice.

For C, the ranking below excludes duplicate launches of the running fill and modern-quality trainings. Thresholds are proposed advancement gates, not observed results. Every learned evaluation includes measured growth and the appropriate same-support heuristic. Use apollo for short diagnostics and goose through `sbatch` for batches, one CPU per process, within its two-job limit. Keep final test data closed. Twenty updates × four tasks × six episodes × 300 seconds already permits **40 CPU-hours per arm per topology**, before evaluation.

1. **Exact-support, mastery, and deadline audit.** Run the four-task diagnostic above on apollo. Gate: all four teacher continuations finish through deployment support within 240 seconds; one-missing-chain completion reaches 80 percent. Failure identifies a prerequisite to fix before interpreting training. Compare subtype-preserving wide support with a small reserved recovery allocation.

2. **Cached cloning and feature-identifiability pilot.** On apollo, collect four training trajectories per topology; compare tiny versus local-capacity features and set-valued versus single-pick loss for ten epochs. Evaluate four unseen lineages per topology. Gate: at least 20 percent empty-start completion and a ten-point gain over the existing RL checkpoint and same-support heuristic. High teacher accuracy with zero completion rejects imitation accuracy as the deciding metric.

3. **Cross-topology transfer without updates.** Evaluate an existing Pegasus-trained checkpoint on eight size-matched Zephyr validation lineages, five episodes each; compare with a Zephyr-trained checkpoint and controls. Reverse direction only after the pilot. Gate: retain at least 80 percent of target-trained coverage and beat the nonlearned constructor by ten points. Target tuning would invalidate the zero-shot claim.

4. **Stronger growth controls and the causal quality-training comparison.** Reassess initial and final fragment checkpoints on the same twelve validation lineages using independent protocol repetitions. Compare fixed growth, coupling-based growth, and allocation across roots versus variants within the existing read budget. Gate: residual advantage at least 0.005 with an interval below zero, matched per-instance resource constraints, and no coverage loss. The final-minus-initial change needs its own interval.

5. **Actual deployed best-of-K feasibility.** On goose, use eight validation lineages per topology and shared 60/300/600-second deadlines. Count observed successful deployments, including all failed attempts and late outputs. Gate: at least 50 percent coverage and twenty points above the strongest control. Do not derive deployment coverage from the aggregate per-episode rate. The existing curriculum reports per-episode averages; the search protocol already supports stopping at the first valid output. [constructor_curriculum.py:580](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/constructor_curriculum.py:580), [constructor_protocol.py:62](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/constructor_protocol.py:62)

6. **Application-derived transfer.** Freeze the checkpoint and evaluate twelve scheduling and twelve graph-partitioning instances selected by size and provenance before baseline outcomes. Use certified-optimum cases for residual assessment, retaining uncertified cases for feasibility. Gate: ten-point coverage advantage, or residual improvement of 0.005 without coverage loss, across both collections. A null confines the claim to synthetic construction distributions.

7. **Ablation table with matched comparisons.** Run short pilots on the first reliably feasible congested rung, then replicate surviving differences with three seeds.

   | Axis | Deciding comparison |
   |---|---|
   | Support | 64 versus 512; separately restore recovery quotas |
   | Features | Tiny, count-scaled/local-capacity, 230 channels |
   | Actor | Linear versus MLP on identical informative rows |
   | Baseline | Contextual actor with LOO versus value baseline |
   | Curriculum | Fixed schedule, assisted-only mastery, cloning then RL |

   A component earns inclusion through ten points of coverage or halving CPU-hours to the same coverage. Otherwise simplify. Keep the actor fixed in the baseline comparison; the current value option requires the contextual actor. [constructor_curriculum.py:766](/private/tmp/claude-504/-Users-nguyencongt-Documents-IsingFold/61c7ad2b-8794-4865-a49b-277aa6136355/scratchpad/pr2/probes/constructor_curriculum.py:766)

8. **Separate scaling curves.** Evaluate frozen checkpoints against logical size on full hosts at 24/48/100 variables, and separately against actual/witness fill at 80/85/90/95 percent. Start with four independent validation lineages per cell. Report coverage, runtime, and controls together. Gate: an advantage survives two consecutive larger-size points and two adjacent fill levels. A collapse identifies the operating boundary.

9. **Quality-only refinement after the running modern jobs finish.** Freeze successful construction and train refinement for twenty updates from its own outputs. Pair every output with random and heuristic growth from that same start. Gate: 0.005 residual improvement with unchanged coverage. This decides whether quality learning deserves a separate module.

10. **Frozen final deployment table.** After the pilots, evaluate three training seeds on at least thirty independent test lineages per headline stratum, using tuned anytime and growth controls. Apply the oral criterion above. Tiny failure-conditioned cells or repeated test-driven adjustments would defeat this result.

11. **Separate measured-selection paper.** The platform result is separable as an empirical systems or benchmark paper; current evidence does not make it a standalone A* ML oral. Its smallest next test is replication on two application collections with total deployment cost reported. Gate: at least 0.05 solve-probability improvement over resource selection, with a positive interval. Prioritize the constructor first if its deployment gate passes.