# Results

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

This run has no early stopping and no weight decay: 389 states, 200 epochs, full batch, a model
with millions of parameters. The gap is therefore also consistent with plain memorisation, and a
failure to transfer under those conditions does not establish that the signal is untransferable.
`scripts/apollo/run_quality_es.sh` runs the control that separates the two, holding a quarter of
the training lineages out of the fit to stop on and adding weight decay, with the held-out
lineages touched by neither the gradient nor the stopping rule. Read that before drawing a
conclusion from this table.

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
