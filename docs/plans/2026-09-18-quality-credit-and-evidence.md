# Quality credit and evidence integrity

## Fixed research contract

The learned policy constructs the embedding from empty. It chooses every construction,
refinement and final COMMIT action. Minorminer remains a comparison method only. Terminal
utility rewards independently sampled solution quality and assigns zero to failed quality
episodes; there is no qubit-count or chain-length penalty. Hardware capacity is a constraint.

The paper keeps two complementary regimes: the resource-abundant P16/Z15 ink-drop instances,
and the congested P6/Z4 fill-80/85/90/95 instances. The P3/Z2 fill-30 cell is a small learning
diagnostic, not a replacement for those benchmarks. Witness fill is a generation label;
an empty-start episode begins with no assigned chains and must report its own occupancy.

## What the audited results establish

This review starts at `97ebb97` and includes later main commits through `b661a6e`.

* The completed fill-30 frozen study has four validation instances per host and one seed.
  Policy residual is 0.0877 versus minorminer 0.0417 on P3, and 0.1128 versus 0.0590 on Z2.
  Both always produce a valid selected embedding on these instances. This is a quality loss.
  Committed continued-feasibility logs contain only headers; corresponding completed quality
  arms are absent. Launched jobs are not completed outcomes.
* The six `results/transfer/m_*.log` files each contain a header and an `init/train` evaluation
  on sixteen IDs. Every ID belongs to the header's train list; none belongs to its fifteen-ID
  held-out list. They do not support the STATUS table's locked-test interpretation.
* `results/quality/excess_clone_p3.log` has mean residual 0.0969 versus 0.1335 in the earlier
  eight-instance diagnostic. The normal interval for the paired improvement is
  [-0.0029, 0.0761]. This comparison is confounded: the earlier checkpoint is
  `r30_p3_s0.best.pt`, while cloning initializes from `F_local_s0.pt`. Four of the eight
  diagnostic IDs are in the clone's train list (two in retained successful demonstrations),
  and seven overlap the union of the two logged training sets. The 46% gap-closure number
  is descriptive arithmetic, not a held-out or isolated cloning effect.
* The follow-up on twenty instances (`55b7f71`) does not reproduce the quality contrast:
  mean improvement -0.0200, normal interval [-0.0428, 0.0028], while qubits fall by 9.3.
  Four instances repeat the first diagnostic; all twenty overlap the union of logged
  checkpoint training sets. Calling them fresh held-out evidence is unsupported.
* The later break-rate diagnostic (`0c85160`) reports 0.0734/0.0641/0.0187 broken fraction
  for policy/pruned/minorminer. Other measurements exactly match the previous twenty-task
  diagnostic. Correlation with quality does not establish that breakage exclusively causes
  the gap or rule out chain-length interactions. Both 250-epoch local20/physics32 clones
  completed and preserved 12/12 measured validity. The next commit `b661a6e` measures their
  residuals: local20 0.1017, physics32 0.0996, minorminer 0.0671 on the same reused twenty
  tasks. Physics-minus-local superiority is unestablished: improvement 0.00206 with t19
  interval [-0.02035, 0.02447]. Minorminer's identical record appears three times per task
  in the advertised 180-row correlation calculation. Keeping it once gives seven arm
  records per task (140 nominal records); mean within-instance break/residual correlation
  is 0.6631 and qubit/residual correlation 0.2815. This remains a descriptive development
  association, not 180 independent embeddings or an out-of-sample prediction result.
* The Pegasus clone's held-out validity is already 1.00 at initialization and remains 1.00.
  The Zephyr clone log contains teacher generation, not a completed training result.
* Greedy pruning recovers only 0.0022 mean residual over eight P3 instances, with substantial
  instance variation. This does not identify placement as the exclusive cause. Pruning also
  changes contacts, topology and coefficient allocation; mean cancellation is possible.

Training loss continuing to decrease is not a lower bound on future generalization or quality.
A valid trajectory is a feasibility demonstration, not evidence that every action optimizes
quality. The committed observations do not establish multi-seed superiority or QPU gains.

## A minimal RL change with an unchanged objective

The reference actor remains the same linear candidate scorer on local20 (physics32 is a
separate feature ablation). The new `--baseline loo_value` adds an independent training-only
critic. It sees the mean and maximum of the legal candidate rows, log support size, remaining
decision fraction, and current construction progress, all before the action is sampled.
Rows are detached and the critic has no shared parameters with the actor. The compact critic
observation is not claimed to be a sufficient Markov state.

Let R_i be terminal base utility of rollout i, V_phi(s) a base-return prediction, and K>=2
independent on-policy rollouts collected without updating either network. The baseline is

    b_it = V_phi(s_it) + mean_{j != i}[R_j - V_phi(s_j0)].
    A_it = R_i - b_it.
    L_actor = -(1/K) sum_i sum_t stopgrad(A_it) log pi_theta(a_it | s_it).

With potential shaping, both return and baseline subtract Phi(s_it); the same advantage
results. Failed episodes remain in the batch and retain R_i=0. The critic is trained by
Huber regression to R_i. Its zero-initialized output makes the first update exactly ordinary
LOO. Actor and critic use separate optimizers and gradient clipping; critic gradients cannot
rescale the actor update through a shared clipping norm. The critic is not evaluated during
deployment. Best-validation checkpoints save actor and critic from the same iteration.

For fixed pre-batch parameters, b_it is independent of the current sampled action conditional
on its pre-action history and the other independent rollouts. Thus

    E[b_it * grad_theta log pi_theta(a_it | s_it)] = 0.

The unclipped, unregularized score estimator retains the original expected-utility gradient.
This is not an unbiasedness claim about Adam or clipping, a convergence guarantee, or a new
RL algorithm. A poor critic can increase variance. The implementation records regression
error and explained variance; superiority requires the matched baseline ablation.

In particular, validity need not equal one for quality gradients to exist. With valid quality
q and success probability p, E[R]=p E[q | valid]. Both factors may depend on the policy at
p<1. Large validity variance can obscure a small quality signal, but a reward-standard-
deviation ratio is not gradient SNR and does not prove an episode sample-complexity bound.
Exact two-step tests include failures and demonstrate a nonzero quality gradient at p=0.5.

## Feasibility teacher and lineage contracts

The default cloning teacher is now `--teacher-mode monotone`. From a witness-consistent state
it labels only available witness-subset extensions that strictly increase assigned qubits
without removing existing assignments. If a valid COMMIT is available, that is the target.
This excludes grow/shrink cycles and prevents further refinement from competing with a
successful teacher COMMIT. It does not insert missing actions, seed a first chain, or guarantee
completion under a bounded support. Legacy `witness_set` remains a declared ablation.

Cloning defaults to the manifest's train and validation lineages. Random repartition is an
explicit exploratory mode. Checkpoints carry cumulative training lineage provenance, and
known train/held-out overlaps fail before training or evaluation. Legacy checkpoints with
unknown ancestors remain diagnostic; the new metadata cannot retrospectively certify them.
The pruning diagnostic uses an explicit manifest role and retains construction failures in
coverage and failure-inclusive utility denominators.

The quality-study runner snapshots initial checkpoints by content hash and creates one
immutable study manifest. All frozen, feasibility and quality arms, including separate
`--only` launches, reuse that snapshot. A changed source checkpoint, altered configuration,
corrupt snapshot or preexisting unregistered output requires a new study directory.

## Controlled next experiment

1. Generate or reserve clean lineage-disjoint data before any policy selection. Produce the
   initial local20 constructor on the manifest training split only; do not warm start from a
   legacy checkpoint whose training ancestry is unknown and then call it confirmatory.
2. Finish a fixed-epoch monotone-teacher initialization and save
   `runs/quality_clean/{pegasus3,zephyr2}_local_clone.pt`. Keep its own pre-cloning checkpoint
   for a matched BC ablation. A fixed epoch count is a declared protocol, not hindsight selection.
3. Run frozen / continued-feasibility / quality arms first with LOO, then with `loo_value`,
   using the same immutable initial weights. Compare the recorded initialization hashes across
   both study directories before pairing results. Set continuation `stop_bias=0`: cloning
   already initialized and trained those logits. Fix all other knobs. The independent critic's
   additional training time must be reported. Isolate local20 versus physics32 afterwards.
4. Advance only after independent validation quality improves relative to both controls
   without hiding failures. Replicate across seeds and then evaluate fresh final-test lineages
   on the two paper regimes. Report residual and solve probability separately, including
   cases where the rare-event endpoint cannot be resolved.

```bash
# Validate historical evidence; these committed transfer files must fail a test claim.
python probes/constructor_evidence_audit.py --require-heldout-test results/transfer/m_*.log

# Display commands, without creating snapshots or starting training.
python probes/run_quality_study.py --config configs/constructor_quality_clean_f30.json \
  --phase pilot --out-dir runs/quality_clean_loo
python probes/run_quality_study.py --config configs/constructor_quality_credit_f30.json \
  --phase pilot --out-dir runs/quality_clean_loo_value

# With input corpora/checkpoints present, append --execute to each command.
```

These implementations and tests make the proposed experiment reproducible and interpretable.
They do not replace its large-corpus outcomes. Current measurement is classical simulated
annealing. Physical/QPU conclusions require physical measurements under a declared protocol.

## Local integration pilot and decision

See [`results/review/quality_credit_v3/README.md`](../../results/review/quality_credit_v3/README.md)
for the twelve completed runs and reproduction commands. The policy builds embeddings from
empty on four manifest-validation P2 instances with six logical variables. Three continuation
seeds share one fixed-epoch cloned initialization. Final residual means across seeds are
0.01449 frozen, 0.01709 continued feasibility, 0.00591 quality LOO and 0.00781 quality
`loo_value`; minorminer is 0.00532. Every selected output is valid. LOO improves over frozen
on two seeds and regresses on one. This is an implementation pilot on an easy, small corpus,
with a dirty-source record, not confirmation of generalization or superiority. Ordinary LOO
remains the reference: the additional critic has not earned replacing it.

## What would make the physics claim testable

Break frequency is a measured outcome, not a sufficient causal explanation. On the same
fresh validation problems, collect independent read blocks for each frozen embedding under
a preregistered strength grid, fixed sampler schedule, and explicit coefficient rescaling.
Keep strength-selection reads separate from assessment reads. Compare policy and baseline
at the same total measurement budget; do not grant only the policy a strength search.
Measure decoded residual, solve probability, break frequency and chain/contact statistics
jointly, retaining failures. A strength intervention can alter both breaks and effective
problem precision; lower breaks alone does not demonstrate better quality. Then isolate
local20 versus physics32 with identical initial logits and the same RL/reward protocol.
An auxiliary break predictor, if used later, must obtain labels from training lineages only
and must not replace the terminal quality objective. No such predictor is claimed by this patch.

For the main-track result, reserve fresh lineage-disjoint final tests after the developmental
reuse above. Test both established benchmark regimes, at matched time/read budgets, with
frozen, continued-feasibility and quality-trained policies plus strong solver baselines.
Report instance-level paired effects and training-seed variability separately. Demonstrating
that the quality objective changes learned construction is necessary; ordinary REINFORCE plus
features alone does not establish a new ML contribution or guarantee conference acceptance.
