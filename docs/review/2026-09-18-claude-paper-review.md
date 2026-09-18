# Paper review, 2026-09-18: claim chain, attacks, and what to run next

Written as a reviewer of an A* submission aiming for an oral, against `STATUS.md` on `main`
at 2026-09-18 and the constructor code on `feat/constructor-curriculum`. Companion to the
GPT-6 astra half of the same review.

## 0. The finding that should change the plan

The fill regime has no objective. `STATUS.md` records it plainly: at 280 to 440 variables of
frustrated loops on Zephyr 4, the witness and minorminer's best of four both read p_solve
0.0000 in every cell, and 512 reads never reach the planted ground state whatever the
embedding. The project already noted the consequence in its own words: "if every one is near
zero, a validity win is a win on problems the annealer cannot solve."

So bottleneck 1, the experiment being treated as the result that lifts the paper's tier, is
about to produce validity numbers on instances where the paper's declared objective is
constant. Every reviewer who reads the appendix finds this. The mean energy residual is the
stand-in and it does discriminate (Task 9: -0.021 to -0.032), but residual above a planted
ground state on a 400-variable frustrated-loop instance is not what the introduction promises,
and a reviewer will say so.

This is not a reason to stop the fill runs. It is a reason to run experiment R1 below before
the paper is written, because the answer decides whether the headline regime is "80 to 95
percent fill at 300 variables" or "the largest instance where fill is high and the annealer
still solves something". Nobody has measured where that crossover is.

## 1. The strongest claim chain, with its evidence

**C1. Valid embeddings of one instance on one host differ substantially in downstream solve
probability.** [SUPPORTED] Oracle selection over eight router draws gains +0.163 p_solve over
the first draw on both Pegasus 6 and Zephyr 4, 30 instances each (`results/rules/`). Needs
Table 1.

**C2. Resource count is a weak proxy for that quality.** [SUPPORTED] Fewest qubits gains
+0.042 and +0.041 with intervals touching zero against measured selection at +0.147 and
+0.153; within-instance Spearman of qubits against p_solve is -0.27 and -0.21; the resource
rule stays at +0.04 at K = 4, 8 and 16 while the measured rule grows. Same table.

**C3. Measuring is cheap enough to deploy.** [SUPPORTED] 64 reads a draw already take 91 to
93 percent of the oracle gain at K = 8; the whole cost of best-of-8 is eight router draws and
512 reads. Needs Figure 2.

**C4. The standard tool has a regime where it returns nothing although a valid embedding
exists.** [SUPPORTED] Planted instances at 80 to 95 percent occupancy with short chains:
minorminer 0 of 6 at 20 tries, at 200 tries, and anytime to 600 s, on both hosts
(`results/fill/*_anytime*.log`, `*_budget.log`). Needs Table 2, the regime map.

**C5. What it lacks there is the roots, not search budget.** [SUPPORTED, and the most
under-sold result in the project] Seeded with witness roots minorminer completes every cell to
85 percent in seconds and most of 90 percent in one to two minutes; random roots add nothing;
half the roots right suffices at 80 percent and the requirement tightens with fill
(`results/seeded/`). This is a clean, surprising, well-controlled result and it currently sits
in the middle of the status file as a diagnostic.

**C6. A learned policy can construct valid embeddings from empty and generalise to unseen
instances.** [PARTIAL] Held-out validity rises on every rung to 20 variables on 128-qubit
fragments, three seeds, every interval above zero, and reaches 0.25 and 0.60 from empty at 24
variables on the full Pegasus 16 and Zephyr 15. Unsupported at 263 to 533 variables and high
fill; that is what is running. Needs Figure 3.

**C7. The learned constructor produces better embeddings than the standard tool on the
objective.** [CURRENTLY CONTRADICTED] At the one scale where quality moves (modern corpora, 16
to 20 variables on full hosts) the fragment-trained constructor is worse than both minorminer
and the growth control, on Zephyr with intervals above zero. At fragment scale the advantage
over minorminer is real (-0.03 to -0.14) but the growth control reproduces almost all of it,
leaving -0.004 to -0.008. Do not write C7 as a claim. Write it as the measured limit of what
the learned part contributes today.

**C8. Learning is necessary rather than convenient.** [UNSUPPORTED] Nothing yet separates the
learned constructor from "minorminer plus growth plus measured selection" except in a regime
where the router produces no draw to grow from. That regime is exactly the fill regime, which
is why R1 matters: the argument for C8 lives or dies on whether that regime has an objective.

## 2. The three attacks that will decide the review

**Attack A: "your target regime has no measurable objective."** Stated above. The evidence that
kills it does not exist yet. What kills it: a curve of witness p_solve against variable count at
fixed fill, showing a band where fill is at or above 80 percent and p_solve is materially above
zero, plus the paper committing to residual as a secondary metric with an explicit statement of
what residual means. If no such band exists, the honest paper reports validity at fill as a
feasibility result and keeps the quality claims at the scale where they hold, with the two
scales clearly separated. That is still publishable. Pretending one regime is the other is not.

**Attack B: "your baseline is a weak minorminer configuration."** Partly answered already: 200
tries moves exactly one cell, 600 s of anytime restarts moves one cell, and the deterministic
clique embedder stops at K60 on Pegasus 6 where the random search reaches K61 to K62, so the
standard tool is not merely under-budgeted. Two gaps remain and a reviewer will find both. The
paper never reports `minorminer.layout` or any layout-aware initialisation, which is the
obvious thing an expert does when a plain call fails on a large sparse host, and it never
sweeps `chainlength_patience` or `max_no_improvement`. Both are hours of compute. Without them
the 20-tries deployed budget is defensible only as "the budget the tool's own documentation
suggests", which is a weaker sentence than the data can support.

**Attack C: "planted instances are easy by construction, and you kept only the ones minorminer
fails on."** The second half is answerable today and the paper must say so in one sentence:
`gen_fill_corpus.py` plants the partition first, asks minorminer afterwards, and reports its
validity rate per cell; no instance is filtered on minorminer failing. The first half is
subtler. The witness guarantees embeddability at exactly the planted fill, and the logical
graph is the quotient of a host partition, so its structure is inherited from the hardware.
A reviewer can reasonably ask whether a quotient-of-Pegasus graph is a problem anyone wants to
embed. The answer the paper needs is a second family: application-derived logical graphs at
comparable minimal fill, even a small set, showing minorminer fails there too. Until that
exists, the regime claim is about planted instances and should be worded that way.

Two further attacks need pre-empting. "Your learned method is a heuristic that spends qubits"
is answered by the growth control and the matched-qubit comparison (-0.008 at 13.6 qubits
against 13.6); present that control as a first-class baseline, not an ablation, because it is
the strongest non-learned competitor. "No QPU" is answerable by declaration if the objective is
stated as a fixed simulated schedule with a fixed strength rule, and is far more convincing
with R2 below, which shows the decision rule survives a change of schedule.

## 3. Experiments not yet running, ranked by information per compute-hour

**R1. Quality against scale at fixed fill.** Take the planted corpora and, for variable counts
spanning roughly 30, 60, 120, 240 and 400 at fill 80, measure the witness embedding's p_solve
and residual under the registered schedule, 512 reads, six instances a point, both hosts.
Cost: an hour or two. Metric: p_solve of the witness. Success: a variable count where fill is
at or above 80 percent and mean p_solve is at or above 0.1. That band becomes the paper's
headline regime and every later comparison moves there. Failure is equally valuable: it fixes
the paper's framing before the fill runs report.

**R2. Schedule robustness of the selection rule.** The same eight draws and 30 instances from
`results/rules/`, re-assessed under two further schedules (a wider and a narrower beta range,
or 100 and 400 sweeps). Cost: three times a run that already takes a few hours, and it is pure
evaluation. Metric: the gain of measured selection and of fewest qubits under each schedule.
Success: measured selection stays at least three times the resource rule under all three. This
converts C1 to C3 from "true under our schedule" to "true of the decision rule", which is what
an oral needs when there is no QPU.

**R3. Cross-topology transfer, evaluation only.** Take the Pegasus-trained checkpoints and
evaluate them on the Zephyr held-out sets and the reverse, at the size and corpus-size rungs.
The checkpoints and the sets exist; this is minutes of compute. Metric: held-out validity.
Success: within 0.1 of the same-topology number. A single policy that transfers across two
hardware families is a headline sentence and it is nearly free.

**R4. Stronger baselines at the fill cells.** `minorminer.layout`, the clique embedder on the
clique cells, and a small sweep of chainlength patience and no-improvement patience. Cost:
hours. Metric: validity per cell. Success for the paper is that they also read 0 on the
short-chain cells; if one of them does not, the regime claim needs restating and it is far
better to find that now.

**R5. The ablation table on one rung, one seed set.** Support (64 against 512), features (16
against 230), actor (linear against contextual), baseline (leave-one-out against value),
curriculum (prefix against none). Most cells exist scattered across rungs with different seeds
and iteration counts, which is not a table. Run them on the corpus-size rung with three seeds.
Cost: the largest item here, perhaps a day of the cluster. Reviewers will demand it and its
absence alone can cost an oral.

**R6. Deployed best-of-K feasibility within a wall clock.** No compute. The evaluation samples
a fixed number of episodes and reports a per-episode rate; the deployed system would sample
until the deadline and return the first valid embedding. Report both, as the router's own
anytime numbers are reported. A per-episode 0.25 becomes something far larger at K = 10 and the
comparison to minorminer's restarts is then like for like.

**R7. Application-derived instances.** Highest value against Attack C, highest cost, and it
needs a corpus that does not exist. Schedule it only after R1 tells you which scale to generate
at.

On whether the selection result deserves its own paper: it is complete, controlled, two hosts,
K and read sweeps, and it stands without any learned component. If the constructor does not
reach a positive result at fill within the next fortnight, publishing C1 to C5 as a measurement
paper with the regime map is a better outcome than a paper whose learned contribution is
-0.008 residual at fragment scale. Keep the option open by writing C1 to C5 so they do not
depend on C6.

## 4. Three figures

**Figure 1, the regime map.** x axis minimal fill or occupancy from 70 to 95 percent, y axis
validity rate, one panel per host. Curves: minorminer at 20 tries, at 200 tries, anytime 600 s,
minorminer seeded with witness roots, and the learned constructor. Data exists for every curve
but the last: `results/fill/*_anytime*.log`, `*_budget.log`, `results/seeded/pegasus6.log` and
`zephyr4.log`. This figure is the paper's argument in one picture: the standard tool falls off
a cliff, the roots put it back, and the question is who supplies the roots.

**Figure 2, measure against count.** x axis K on a log scale at 4, 8, 16, y axis p_solve gain
over a single draw, lines for fewest qubits, measured at 64 reads, measured at 256 reads, and
the oracle, two panels for the two hosts, bootstrap bands. All data is in `results/rules/`
including the sweep files. The message is the separation between the flat resource line and the
rising measured line.

**Figure 3, the ladder.** x axis logical variables at 3, 4 to 8, 8 to 14, 12 to 20, 24, y axis
held-out valid rate, paired before and after training, markers by host, host qubit count as an
annotation on each point. Data in `results/curriculum/`. Add the fill points when they report.
This is the generalisation claim and it is the only figure that shows the learned component
doing something.

## 5. What to cut

Cut the successor scorer, the learning-curve flatness, the representation ablations and
direction 5. They are a retired method and its post mortem, and including them invites the
reviewer to evaluate the retired method. One appendix line, that predicting quality from
structure transfers at about +0.02, is enough and it motivates measuring.

Cut hybrid v3, the contact policy, the pool ceiling and the adaptive-allocation prior. Each is
a clean null and together they are most of the project's history, but a paper is not a lab
notebook. Successive halving matching uniform at half the reads is one line in the method
because it is what the platform deploys. Cut the eleven withdrawn numbers; that discipline
produced this record and does not belong in a submission.

Keep the negative that matters: the growth control reproducing the constructor's quality
advantage. Reporting it makes the remaining -0.008 credible, and a reviewer who finds an
author disclosing their own strongest baseline reads the rest of the paper differently.
