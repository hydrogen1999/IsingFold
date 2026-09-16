# Results

## Index

One line per directory. The section below each name, where it exists, says how the number was
produced and what would make it an artefact. A directory listed here and described nowhere
else is described by its log's own header line.

| directory | what it holds | status |
|---|---|---|
| `v1/`, `v3/` | the registered-bar policy runs on the Chimera v1 corpus | history; auto schedule, pre-ADR-001 encoding |
| `quality/`, `quality_es/`, `quality_v2/`, `quality_cap/` | the quality surrogate on Chimera, four variants | history; auto schedule |
| `r1/`, `r2/`, `r2b/` | successor scorer labels and runs on Chimera | history; encoding mismatch (D1), auto schedule (D2) |
| `premise/` | the frontier premise checks | history |
| `hard/`, `hard_abl/` | the clique-16 hard corpus and its four-arm representation ablation | history; within seed noise |
| `diverse/`, `diverse_abl/` | seven-family corpus labels and the family-held-out five-arm ablation | history; within 0.02 of each other |
| `curve/` | learning curve 25/50/100/200 lineages, two seeds | history; flat, lexicographic subsets |
| `audit/` | sampler scale controls, registered schedule, and implementation regression gates | ADR-002 controls; `independent_constructor_unit.log` verifies ADR-005 contracts only, not training performance |
| `modern/` | Pegasus 6 and Zephyr 4 corpora, the power-law budget sweeps, three pool-ceiling versions | current hardware; auto schedule |
| `feasibility/` | minorminer validity by size and family, budget check, deterministic clique embedder | current |
| `fill/` | fill-planted corpora, minorminer at ten and fifty tries, constructor floor, witness quality, exact and long-chain sweeps, anytime baseline | current; `*_witness_registered.log` is Task 9 |
| `relabel/` | labels under the registered schedule with the corrected compiler, both modern hosts | current; Task 6 |
| `corrected/` | old versus corrected encoding on identical labels, and label reliability | current; Tasks 7 and 8 |
| `large/` | corrected successor scorer, 1,680 training lineages, three optimization seeds | positive validation gain over random; no paired resource-baseline CI or final-test claim |
| `adaptive/` | uniform, successive halving and UCB selection on frozen pools | classical allocation baseline; existing logs do not measure a learned prior |
| `hybrid/` | imitation-root deployment and early hybrid RL snapshots | learned roots tied the plain router in aggregate; new jobs need completed logs |
| `rl/`, `imitation/`, `search/`, `ours/` | constructive training, replay, deployment and router diagnostics | snapshots and diagnostics, not a demonstrated learned end-to-end win |
| `seeded/` | witness, random and partial-hint completion controls | witness is privileged information, not a learned result |

`modern/*/sweep_registered.log` supersedes the auto-schedule sweep for the registered objective.
Its contact-versus-start intervals include zero. The old log footer incorrectly describes
independently grown arms as nested; the raw observations remain unchanged for provenance.
See `docs/review/2026-09-16-quality-contract-fixes.md` for the corrected evaluation contracts.
Source fixes do not change results already recorded here; rerun and version the corrected protocol.

Logs from runs executed on this tree, and the checks that decide whether to believe them.

Nothing here is a headline. Two of the three arms below are development diagnostics measured on
lineages that were looked at repeatedly, and they are labelled as such. The audit checks are the
more useful part of this directory: they are the reason several earlier numbers from this project
were withdrawn.

## v1, the first run from the published tree

`runs/v1` on apollo, corpus digest `25570ef89d385f2c` from seed 20260914, Chimera 4, 16 nominal
variables, 600 instances, 420/90/90. Three model families, sixty rounds of elite imitation from a
minorminer floor, roughly 12,000 seconds each.

| file | what it holds |
|---|---|
| `v1/train_if-*.log` | per-round loss, rollout statistics and the development curve |
| `v1/registered_bar.log` | best-of-8 against a random controller at matched episodes, test split |
| `v1/frontier2.log` | the cost-utility frontier against minorminer, selection and assessment separated |
| `v1/corpus_manifest.json` | the corpus identity those numbers belong to |

The registered comparison, sixty test lineages, every arm on the same minorminer floor:

| family | best-of-8 | against random best-of-8 | assessed on independent reads |
|---|---|---|---|
| IF-Core | 0.7210 | +0.0242 [-0.0033, +0.0523] | +0.0280 [-0.0013, +0.0577] |
| IF-Dual | 0.7228 | +0.0260 [-0.0098, +0.0629] | +0.0311 [-0.0047, +0.0676] |
| IF-MLP | 0.7254 | +0.0286 [-0.0012, +0.0607] | +0.0327 [-0.0003, +0.0674] |

Three positives of about the same size, every interval touching zero. By the registered rule this
does not pass. The interval half-width is the same size as the effect, so it is a power question
and the test split holds ninety lineages rather than the sixty used here.

The development curve in the training logs reads +0.02 to +0.04 above the protected initializer,
but it was measured on validation and test pooled, watched every five rounds. It is a diagnostic.

## v3, the first run with both audits' P0 findings fixed

`runs/v3/seed0` on apollo, same corpus `25570ef89d385f2c`, sixty rounds, one seed per family.
`v3/run_identity.json` records the corpus digest, the seed, the round count and a sha256 of the
RL source tree, because a launcher that skips on round count alone cannot tell one seed or one
code version from another.

What differs from v1 and v2: the teacher accepts a trajectory on its re-measured gain rather than
on the block that made it the winner; RESTART reaches the real initializer during training, as it
does at deployment; the in-training curve reads validation only; the advantage baseline leaves
the episode out of its own baseline.

Two quantities the earlier runs could not produce.

The inflation a maximum over eight noisy blocks invents, measured rather than assumed:
`selected_gain` minus `remeasured_gain`, averaged over all sixty rounds, is 0.0151, 0.0131 and
0.0150 across the three families.

An earlier version of this file gave 0.0303, 0.0236 and 0.0270 for the same quantity. Those are
the first two rounds, read while the run was still starting and then written down as if they
described the run. A fourth external audit recomputed the sixty-round means and found the
discrepancy. The inflation is real and it is about half what this document previously claimed.

Between 17.4 and 18.8 percent of lineage-rounds produced no accepted teacher, 251, 270 and 256 of
1,440 attempts. The aggregate log does not separate a rollout rejected by the gate from a round
that collected no episodes, so this is a count of non-accepted attempts and not, as this file
previously said, a count of false teachers the old gate would have admitted. Separating the two
needs its own counter.

The single-episode policy against the protected initializer, on a fixed forty-instance validation
subset:

| family | rounds 0-25 | rounds 30-59 | final CE loss |
|---|---|---|---|
| IF-Core | +0.0150 | +0.0237 | 0.425 |
| IF-Dual | +0.0069 | +0.0373 | 0.328 |
| IF-MLP | +0.0075 | +0.0203 | 0.469 |

This is the first time in this project that a single policy episode beats the initializer on a
fixed subset and the margin grows with training. It is also not the bar: the best of forty-eight
random rollouts beats the initializer by +0.26 without learning anything. The registered
comparison against a random controller at matched episodes is what decides, and it is reported
separately when it finishes.

The curve is noisy enough that any three consecutive points mislead. IF-Core reads +0.0709 at
round 35 and -0.0373 at round 50. Read the block means, not the points; this document's author
read three points as a trend at round 13 and had to withdraw it.

### v3 at the registered bar, all ninety test lineages

Every arm protected by the same minorminer floor. Reference: initializer 0.4529 assessed, random
one episode 0.4985, random best-of-8 0.7055.

| family | one episode | best-of-8 | against random best-of-8, registered | assessed |
|---|---|---|---|---|
| IF-Core | 0.5118 | 0.7251 | +0.0157 [-0.0093, +0.0420] | +0.0196 [-0.0080, +0.0478] |
| IF-Dual | 0.5449 | 0.7363 | +0.0262 [-0.0041, +0.0563] | +0.0308 [-0.0013, +0.0622] |
| IF-MLP | 0.5211 | 0.7260 | +0.0135 [-0.0167, +0.0442] | +0.0204 [-0.0123, +0.0535] |

Three positives, every interval containing zero, on the full test split with every fix from two
audits applied. By the registered rule this does not pass, and the point estimates are no larger
than v1's were on sixty lineages, so the earlier reading that more lineages would settle it was
wrong.

The per-episode arms are the part that moved. A single policy episode reaches 0.5118, 0.5449 and
0.5211 against a random controller's 0.4985 and the initializer's 0.4529, so the policy is better
than random at producing one embedding. That edge does not survive the maximum: random best-of-8
reaches 0.7055 and the policies 0.7251 to 0.7363. The search is doing most of the work and an
improvement in the body of the distribution is worth little when the deployment rule takes the
tail.

## Learned quality, step one of the V4 plan

The question a third audit isolated: the signal within a state is real and a network fits it on
the states it was fitted on, but does the fit transfer to lineages it never saw? That is what
decides whether a learned scorer can stand in for evaluator budget.

`probes/train_quality.py` labels candidate pools with the real evaluator, supervises the
per-action quality head directly with a binomial likelihood on the counts plus a within-state
ranking term, and then measures the argmax pick on independent reads at states from lineages that
took no part in training. 200 training lineages gave 389 states, 60 held-out lineages gave 118,
3,834 labelled candidates in all.

| | fitted states | held-out lineages |
|---|---|---|
| oracle | 0.6239 | 0.7019 |
| head | 0.5991 | 0.5744 |
| random | 0.4669 | 0.5956 |
| resource | 0.4826 | 0.5698 |
| incumbent | 0.4858 | 0.6018 |
| head minus random | **+0.1322 [+0.0976, +0.1669]** | **-0.0212 [-0.0619, +0.0183]** |
| head minus oracle | -0.0247 | -0.1275 |
| rank correlation | +0.381 | +0.048 |

That is IF-MLP; IF-Dual gives -0.0344 [-0.0724, +0.0029] held out with correlation +0.095. On the
states it was fitted on the head captures 84 percent of a +0.157 ceiling. On new lineages it
captures nothing, while the ceiling there is still +0.1063 [+0.0809, +0.1346].

That run has no early stopping and no weight decay, so the gap was equally consistent with plain
memorisation. The control that separates the two is in `results/quality_es/`: a quarter of the
training lineages held out of the fit to stop on, weight decay, and the held-out lineages touched
by neither the gradient nor the stopping rule.

It settles it. Both families fit 291 states, stopped on 98 held out of the training lineages,
both stopped at epoch 45, and the best rank correlation the stopping rule could find on that
inner split was 0.048 and 0.071. The model never has a transferable ordering at any epoch; it is
not a case of learning one and then losing it to memorisation.

| held-out lineages, 118 states | IF-MLP | IF-Dual |
|---|---|---|
| oracle | 0.7037 | 0.7037 |
| head | 0.5764 | 0.5844 |
| random | 0.5945 | 0.5970 |
| resource | 0.5727 | 0.5738 |
| head minus random | -0.0181 [-0.0561, +0.0190] | -0.0126 [-0.0475, +0.0222] |
| ceiling | +0.1092 [+0.0833, +0.1380] | +0.1067 [+0.0808, +0.1352] |
| rank correlation | +0.029 | +0.029 |

On the fitted states early stopping takes the head from +0.1322 to +0.0358 and the correlation
from +0.381 to +0.098, which is what confirms the larger figure was memorisation.

**Withdrawn.** A fourth external audit found two label bugs in the code that produced the table
above, and both are in the direction that manufactures a negative result.

COMMIT was resolved as workspace plus new_chains. COMMIT carries no new_chains at all: the
environment returns the archive entry the candidate names. At the root state the workspace and
the archived entry coincide, so the error is invisible there and appears at every state after a
change, which is most of the dataset. Four fixtures out of four reproduce it.

The control named "incumbent" was the current workspace rather than the protected archive entry,
so the baseline the head was compared against changed identity partway through an episode. Eight
of sixteen probe states had the two differing.

Four statistical faults accompany them: the inner split cut a list of states instead of splitting
lineages, so a lineage could sit on both sides of it; arms were zipped to a common length instead
of joined on the state each number belongs to; the bootstrap treated two states of one lineage as
two independent draws; and the fresh-read seeds came from Python's per-process salted string
hash, the trap this repository wrote a hand-test for in the same week.

All are fixed and the experiment is re-running. On a fourteen-lineage probe of the corrected code
the held-out difference is +0.0343 rather than negative, so the sign of the earlier result was a
consequence of the labels. Nothing in the paragraph below should be relied on until the full
re-run is in.

**What this establishes and what it does not.** Within one state, which candidate is better is not
predicted by features that transfer between instances, for this architecture, this observation,
this corpus and a label budget of 256 reads. It was tested with direct supervision, a loss suited
to ranking, and a stopping rule able to catch memorisation, so the earlier failures were not
caused by the teacher or the loss. It does not show that no model could do this. It shows that
the model the specification names, trained properly, does not, and that three unrelated methods
now fail at the same place while the ceiling there is +0.11 and reproduces on independent reads.

### The same experiment after the label bugs were fixed

`results/quality_v2/`, same 200 training and 60 held-out lineages, early stopping on a
lineage-disjoint inner split, bootstrap resampled by lineage.

| held-out lineages, 118 states in 60 lineages | IF-MLP | IF-Dual |
|---|---|---|
| oracle | 0.7083 | 0.7083 |
| head | 0.5973 | 0.5895 |
| random | 0.5947 | 0.5947 |
| resource | 0.5739 | 0.5739 |
| head minus random | +0.0028 [-0.0281, +0.0329] | -0.0049 [-0.0378, +0.0266] |
| ceiling | +0.1123 [+0.0832, +0.1440] | +0.1123 [+0.0832, +0.1440] |
| rank correlation | +0.084 | +0.048 |

The held-out conclusion survives the correction: the head sits at zero where the ceiling is
+0.11. The +0.0343 seen on a fourteen-lineage probe of the fixed code was noise at that size.

What does not survive is the claim of capacity. On the states it was fitted on the head is now
+0.0152 and +0.0113, not the +0.1322 reported before, because that figure came from a run with
no stopping rule and the wrong labels. This pair of runs therefore cannot separate "cannot fit
correct labels" from "fits them but does not transfer", and `scripts/apollo/run_quality_cap.sh`
runs the arm that can: correct labels, no stopping rule, no weight decay.

How much the label bug mattered, on the corpus rather than on fixtures
(`results/audit/commit_label_check.log`, reproducible with `probes/check_commit_labels.py`):
of 178 legal COMMIT rows across 120 states, 60 named the wrong embedding. None of the 60 root
rows were wrong and 60 of the 118 later rows were. The error cannot occur at the first state
anyone would check by hand and affects half of everything after it.

### Capacity, settled

`results/quality_cap/`: correct labels, no stopping rule, no weight decay, 400 optimizer steps.
This is the arm that separates a model which cannot fit the labels from one that fits them and
does not transfer.

| | IF-MLP | IF-Dual |
|---|---|---|
| fitted states, head minus oracle | **-0.0000 [-0.0049, +0.0050]** | **+0.0031 [-0.0018, +0.0080]** |
| fitted states, head minus random | +0.1513 [+0.1214, +0.1853] | +0.1544 [+0.1245, +0.1880] |
| fitted states, rank correlation | +0.714 | +0.833 |
| held-out, head minus random | -0.0287 [-0.0610, +0.0018] | -0.0197 [-0.0530, +0.0133] |
| held-out, rank correlation | +0.139 | +0.060 |
| held-out ceiling | +0.1123 [+0.0832, +0.1440] | +0.1123 [+0.0832, +0.1440] |

The network reaches the label oracle exactly on the states it was fitted on, in both families.
Capacity, the gradient path, the teacher, the loss and the labels are therefore all ruled out.
What is left is generalisation: it learns a map from those particular states to their answers and
carries nothing to a new lineage.

That is a different diagnosis from the one this file carried two revisions ago, which said the
network could not learn the signal. It can, completely. It cannot transfer it.

### Changing what the model is shown

The fifth audit's reading is that the head has to reconstruct quality from the current state and
the factors that changed, so there is nothing for it to generalise over. `successor_scorer.py`
scores an embedding from the physical program it compiles to at the strength it will be run at:
qubit messages over the physical graph, pooling into the chain each qubit belongs to, messages
between chains along realised logical couplings, one number. It cannot see the state, the history
or the other candidates, which is the property being tested: two searches reaching the same
program must score it the same.

`train_successor.py` fits it on the same cached labels, the same states and the same references,
so the comparison is a representation ablation and not a new experiment.

| held-out lineages, 118 states in 60 lineages | head reading the state | scorer reading the program |
|---|---|---|
| head minus random | -0.0287 [-0.0610, +0.0018] | **+0.0362 [-0.0020, +0.0737]** |
| head minus the protected incumbent | -0.0343 [-0.0722, +0.0054] | +0.0148 [-0.0213, +0.0538] |
| rank correlation | +0.139 | +0.143 |
| ceiling | +0.1123 | +0.1297 |
| fitted states, head minus oracle | -0.0000 | -0.0002 |

Both reach the label oracle on the states they were fitted on, so capacity is matched and the
only difference is what the model is shown. Changing that moves the held-out difference by about
+0.065, from below a random pick to above it.

It is not a pass, and the number itself does not replicate. Three seeds at a looser gradient clip
give +0.0093 [-0.0264, +0.0451], +0.0109 [-0.0147, +0.0381] and +0.0157 [-0.0161, +0.0474], a
mean of +0.0120 rather than +0.0362. The single-seed figure was a lucky draw, which was said when
it was reported and then used as the headline anyway. Loosening the clip did not help either, so
the idea that clipping was holding the estimate down is also wrong; seed variance dominates.

What survives is the direction and roughly a third of the size. Against the action head's -0.0287,
-0.0197 and -0.0140 on the same task, the representation is worth about +0.025 to +0.030, with
every individual interval still containing zero.

The loss is no longer a suspect either: `results/r1/bce.log` fits the action head with a
likelihood term alone, no ranking term, and lands at -0.0140 [-0.0405, +0.0119] held out.

One detail worth keeping: the rank correlation barely moved, +0.143 against +0.139, while the
quality of the top pick moved a great deal. The new scorer does not order the whole list better;
it gets the first one right more often. For choosing a single candidate that is the property that
matters, and it suggests rank correlation was the wrong headline statistic throughout. Top-choice
regret is the quantity to report.

## A corpus where the objective can show itself

The measurement above says the qubit budget explains under a tenth of the variance in quality,
with the wrong sign. The corpus it was measured on explains why: nominally sixteen variables, it
holds 4.48 independent components whose largest piece has 11 of them and three spins coupled to
nothing, chains one or two qubits long, and minorminer finishes in two milliseconds. Chain
integrity is almost never the binding constraint there, so "more qubits" only ever means a longer
chain the problem did not need.

`probes/gen_hard_corpus.py` inverts the construction. Rather than dropping chains onto the
hardware and keeping whatever contact graph falls out, it starts from a dense logical graph,
plants a frustrated-loop Ising for a certified ground energy, and asks minorminer whether the
result can be embedded within the cap at all. The witness is minorminer's embedding: it proves
feasibility and is never a quality label.

| | ink-drop corpus | clique-16 corpus |
|---|---|---|
| logical edges | 21.6 | 45.3 |
| isolated spins | 3.00, in 41 of 48 | 0.42, in 16 of 48 |
| independent components | 4.48, more than one in 45 of 48 | 1.42, more than one in 16 of 48 |
| largest component | 11.06, at most 12 in 30 of 48 | 15.58, at most 12 in 0 of 48 |
| qubits used | about 22 | 43.2 |
| longest chain | about 2 | 4.36 |
| headroom over six embeddings | 0.358 | 0.339 |

600 instances, digest `e5382da2349ebafb`, none lost to planting or to the cap. It is one
connected problem over 15.6 of its 16 variables, with twice the edges, twice the qubits and
chains twice as long, and the spread in quality between independent embeddings of one instance
survives the change. Forty-three of the hundred and twenty allowed qubits are used, so there is
room to spend more if spending more helps.

Whether it helps is being measured, not assumed. minorminer still embeds these in eight
milliseconds, so this is not a corpus that is hard to embed at all; it is one where chain
integrity is on the critical path, which is the condition under which choosing an embedding by
the right objective can differ from choosing it by resource.

## Audit checks

An external audit of commit 41fcfac found five things worth fixing in the measurement and the
environment. These are the reproductions after the fixes.

| file | what it shows |
|---|---|
| `audit/frontier2_noop_control.log` | a policy that returns its input unchanged now reports +0.0000 [+0.0000, +0.0000]; the first version of the probe reported about -0.10 |
| `audit/sampler_scale_auto.log` | with the sampler choosing its own schedule, shrinking every coefficient to a sixteenth changes utility by exactly +0.0000 |
| `audit/sampler_scale_registered.log` | with a registered schedule the same shrink costs -0.4195, which is the effect a fixed-temperature argument is about |
| `audit/rank_actions.log` | ranking the actions at one state, measured on Q rather than V: oracle 0.7165, quality head 0.5687, random 0.5530, resource 0.5410 |
| `audit/corpus_report.log` | what the corpus contains: 3.00 isolated spins per instance, 4.48 independent components, largest component 11.06 of a nominal 16 |
| `rules/` | selection rules over the same eight router draws (first, random, fewest qubits, shortest chain, measured, oracle), Pegasus 6 and Zephyr 4, registered schedule; `probes/selection_rules.py` | current; the platform's central number |

The sampler pair is the one to read first. It says that every claim this project has made about
energy-scale compression was measured by an instrument that cancels it.

The corpus report is the second. A corpus described as sixteen variables is, on average, four
independent problems whose largest piece has eleven variables, and minorminer embeds it in two
milliseconds. It also has real headroom between different embeddings of the same instance, 0.358
mean spread over six draws, which is why selection matters here even though the problems are
small.

## What is not here

No checkpoints: they are large and are regenerated by `scripts/apollo/run_v1.sh`. No corpus: it
is regenerated from the seed in that script and the digest is checkable against
`v1/corpus_manifest.json`. No v2 results yet; that run uses the corrected teacher and is still
training.

## `curve/`, `hard_abl/`, `modern/*/pool*.log`

The learning curve of the successor scorer at 25, 50, 100 and 200 training lineages, two seeds
each, on the v1 cache; the four-arm representation ablation on the hard corpus; and the pool
ceiling probe on Pegasus 6 and Zephyr 4 in three versions: `pool_2k_biased.log` is the first
run whose mixed pool held sixteen candidates and is kept as the record of a withdrawn number,
`pool_equalk.log` is the corrected three-pool run, `pool.log` adds the diversity-chosen pool.
