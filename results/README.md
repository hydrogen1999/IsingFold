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
`selected_gain` minus `remeasured_gain` is 0.0303, 0.0236 and 0.0270 across the three families.
About one lineage in five per round produced a rollout that won its own block and then failed its
independent re-measurement; under the old gate all of those were taught on.

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
