# Empty-start RL for solution-quality-directed embedding

## Objective and current evidence

The intended method learns to construct the entire embedding from empty logical chains.
There is no minorminer completion step in the learned arm. A qubit cap limits admissible
embeddings; qubit count and chain length are not reward penalties. More qubits may be
beneficial, neutral or harmful, depending on where they are used and on the sampling pipeline.
The empirical claim to earn is better independently assessed solution quality under a
declared hardware, inference-time and measurement budget.

This change integrates the constructor branch at `78df086` with main at `2696498`.
Existing results do **not** establish a broadly superior learned embedder:

| Observation from committed logs | Supported interpretation |
|---|---|
| Full-host validation validity 1.00 / 0.95 | Four 24-variable validation instances, five episodes each, one seed per host; not large-logical-problem or quality superiority |
| Zephyr size-rung residual vs grown baseline -0.008324, interval [-0.014587,-0.002695] | Preliminary quality advantage on 12 validation instances and one seed; equal **average** qubit count, not paired resource equality |
| Same Zephyr run, final-minus-initial residual -0.004671 | Preliminary training progress; continued-feasibility control is needed to attribute it specifically to quality reward |
| Pegasus counterpart -0.004064, interval [-0.010776,+0.003095] | Inconclusive conditional quality; policy coverage 11/12 versus comparator 12/12 must remain visible |
| 20-channel versus historical 16/230-channel tables | Confounded by support, STOP bias and evaluation settings; cannot isolate the feature contribution |
| Test-list transfer rerun after inspecting its budget effect | Development diagnostic; the already opened test is no longer an untouched confirmatory set |

Sources: `results/curriculum/{P,Z}q_lin230_s1.log`, `results/inkdrop/ink24_*`,
`results/curriculum/{F,G}_local_s*.log`, `results/transfer/t3_*.log`.
Reward-channel measurements in `results/quality/reward_*` demonstrate an informative
measurement channel on router/growth pools, not a learned-policy gain or guaranteed
alignment between expected residual and solve probability.

## RL contract

State consists of the logical Ising coefficients, hardware graph, partial chains, archive,
legal materialized candidate actions, and remaining construction budget. Compact feature
summaries are observations, not claimed sufficient statistics or a proof of Markov sufficiency.
The policy samples one legal action, including placement, routing/refinement, restart and
the final COMMIT. An episode succeeds only through an on-time policy-selected valid COMMIT.
All assessment starts empty; prefix-assisted training remains optional and explicitly recorded.

For a logical instance let S = sum|h| + sum|J|, and let r be mean decoded excess energy
normalized by max(|E_ground|, 1e-9). The existing valid-episode utility is

    U = 1 - 0.5 r / B,    B = 2S / max(|E_ground|, 1e-9).

Invalid episodes have U=0. For S>0 this is exactly

    U = 1 - (mean(E) - E_ground)/(4S).

It orders valid embeddings by expected normalized energy, with no resource penalty.
The S=0 case has utility one for valid embeddings. This utility is a **soft** combination
of validity and quality; it does not prove a lexicographic validity guarantee and does not
guarantee that lower expected energy raises ground-state success probability.

For K independent episodes from one unchanged instance and starting state, REINFORCE uses

    A_i = U_i - mean_{j != i} U_j,
    L_actor = -(c/K) sum_i A_i sum_t log pi_theta(a_it | observation_it).

The optional positive constant c is declared before the batch. This preserves the expected
policy-gradient direction for the same utility. It is not per-batch standard-deviation
normalization, does not discard failed episodes, and does not selectively amplify quality
relative to feasibility. The implementation separately logs both sources of advantage.
Potential shaping with gamma=1 and terminal potential zero cancels exactly in this LOO
advantage, so enabling it does not manufacture dense credit. A learned value baseline is
available but not needed for the paired reference experiment.

## Representation change isolated from the optimizer

`--features physics` contains the unchanged local 20-channel prefix plus twelve before/after
channels: coefficient-weighted contact coverage and redundancy, load concentration, a
bridge-load bottleneck proxy, cycle redundancy, and signed realized coupling. They use
only public coefficients and topology. They are structural proxies, **not** estimates or
certificates of chain-break probability. COMMIT describes the actual selected archive entry.

`--expand-features` zero-pads a linear tiny/local checkpoint into the larger compatible
schema. Thus added features start with zero weights and preserve the old action logits.
Do not change support, STOP bias, data, prefix assistance or budgets in a feature ablation.
Compare the same initialized local20 policy to physics32; benefits and runtime overhead
remain experimental questions. Construction230 stays available as a separate reference.

## Selection, assessment and baseline strength

The deployed selector uses `(mean(E)+S)/max(S,1e-9)`, an affine public-energy score.
It does not read a certified optimum. Training labels and independent scientific assessment
may use certificates; the actor, constructor and deployed selector do not.

Two evaluation modes are explicit:

1. `--proposal-fraction 0`: equal measured-candidate cap, useful for comparing proposal quality.
2. A fraction in (0,1): use that portion of the common deadline to build an outcome-blind,
   bounded-memory structural-diversity shortlist; measure its candidates with the remaining
   time. Fast baselines can explore beyond K initial draws. Failed and duplicate attempts,
   feature extraction, compilation and late calls are charged. The last reporting-only
   assessment is outside deployment time for every arm.

The default study uses eight 256-read selection blocks at most and a fresh 4096-read
assessment block. Residual, solve probability and chain breaks are available from one
assessment block; adding a reported endpoint does not add a hidden sampler call.
All compared proposers receive the same protocol, physical cap and CPU allocation.
The shortlist is a declared heuristic, not an optimal anytime selector. Tune its fraction
and baseline router settings on validation before freezing the final comparison.

Controls include plain minorminer, random growth, selected zero-to-four-qubit growth
(including the unchanged draw), and coefficient-guided growth. No baseline is permitted
to complete the learned constructor. Report both failure-inclusive utility/success and
conditional residual, plus coverage and actual resource use. Equal caps do not imply
equal realized qubit expenditure.

## Paired experiment and commands

`configs/constructor_quality_study_v2.json` declares paired runs. For each host, feature
schema and seed, compare the **same initial policy** frozen, continued with feasibility
reward, and continued with quality reward. Training runs share datasets, updates, episode
counts and optimizer settings. Both training arms are evaluated and checkpoint-selected
with the same quality protocol. Quality labels add training cost, which must be reported
separately; equal updates are not equal training wall time.

First edit only the input paths if the corpus/checkpoints are stored elsewhere. The supplied
paths refer to the existing quality-cell corpora and their local20 feasibility pilot outputs.
An initial checkpoint must have been chosen without looking at the final test.

```bash
# Print exact commands; starts no training and opens no test data.
python probes/run_quality_study.py --phase pilot --hosts pegasus3 --features local

# Execute three controls sequentially, refusing missing inputs or overwritten artifacts.
python probes/run_quality_study.py --phase pilot --hosts pegasus3 --features local --execute

# Isolate the representation change, after the local reference works.
python probes/run_quality_study.py --phase pilot --hosts pegasus3 --features physics --execute

# Three training seeds on each selected host; use a separate output directory.
python probes/run_quality_study.py --phase replication --features local \
  --out-dir runs/quality_study_v2_replication --execute

python probes/quality_study_report.py \
  --quality-log runs/quality_study_v2/pegasus3_local_quality_s0.log \
  --control-log runs/quality_study_v2/pegasus3_local_continued_feasibility_s0.log
```

Validation-selected weights are saved as `NAME.best.pt`; `NAME.pt` remains the last weights.
The log distinguishes their scores. Do not silently present a best validation score as test
performance. Evaluation-only runs execute once, rather than duplicate init/final test passes.
Training and held-out lineage overlap is rejected, and only one representative per lineage
is sampled. Fixed `--data-seed` separates data randomness from optimizer randomness.
Run contracts record code revision, dirty status, graph/coefficient hashes, input checkpoint
content hash, settings, and split role. Legacy or mismatched contracts remain diagnostic.

The attribution reporter bootstraps lineages within a paired seed; it never treats reads or
repeated rollouts as independent graphs, pools seeds as independent tasks, or selects the
best observed iteration. Report the spread across at least three training seeds separately.
Missing assessment is an experiment error, not a poor-quality observation.

## Decision rules and remaining empirical work

Advance after the policy reliably constructs valid pools and quality training improves
independent validation quality relative to both its frozen checkpoint and matched continued
feasibility training. Replicate on fresh lineages and a second topology. If residual improves
but solve probability does not, state the expected-energy contribution and do not substitute
it for a solve-probability claim. Calibration now reports these cross-channel rankings directly.

After development, reserve genuinely new test lineages and freeze checkpoint, selector,
endpoint, baseline tuning, inference budget and practical effect margin before opening them.
Report outcomes on every instance, including failures, and a size/fill curve rather than
only favorable cells. Test at another declared annealing depth/strength policy and ultimately
on a QPU if claiming physical hardware gains. Current evaluator evidence is classical
simulated annealing, not quantum advantage.

These changes repair implementation and attribution gaps and supply a testable learned-method
design. They do not establish the remaining gains. Large training jobs, independent test
replication and hardware evaluations require the external corpora/checkpoints and compute;
their completion cannot be inferred from passing unit tests or a small local smoke run.
