# Status

A running record of what has been done, what each result is worth, and what is still open. Kept
because this project has withdrawn eleven published numbers, and a list of what currently
stands is the only way to tell a finding from a leftover. Every number cites its log under
`results/`; every experiment in flight is on the board below with its log path, so a check
never depends on memory.

## Board, 2026-09-15

| task | what it decides | host, log | state |
|---|---|---|---|
| Task 7 | old vs corrected encoding on identical labels, 3 seeds each, both hosts | `results/corrected/*_seed*.log` | **done, Checkpoint 2 failed**: corrected +0.003 Pegasus, +0.035 Zephyr; legacy +0.012, +0.041; no corrected interval above zero; the improvement surrogate is retired as a method |
| Task 8 | label reliability, learnable ceiling | `results/corrected/*_reliability.log` | **done**: ceiling +0.109 / +0.122 |
| Task 9 | quality endpoint at the fill regime under the registered schedule, paired with intervals. Decides feasibility-only or not | apollo `runs/fill/*_witness_registered.log` -> `results/fill/` | **done** both hosts: residual discriminates, -0.021 [-0.030, -0.010] Zephyr and -0.032 [-0.047, -0.017] Pegasus at 80 percent |
| Task 10 | anytime minorminer with a wall-time deadline, 30 to 300 s | `results/fill/*_anytime.log` | **done** both hosts: Zephyr at 300 s medium chains 1.00 / 0.83 / 0.33 / 0 at 80 / 85 / 90 / 95 percent, short chains 0; Pegasus only 80 percent medium reaches 1.00, every other cell 0 at every deadline. ADR-003's three gates are passed on both hosts |
| budget x20 | minorminer at 200 tries on the fill corpora | `results/fill/*_budget.log` | **done** both hosts: Pegasus 80 percent medium 0.33 -> 0.83 -> 1.00 at 10, 50, 200 tries, 85 percent 0 -> 0.17 -> 0.17, everything else 0 at 300 to 480 s a draw; Zephyr done: 85 percent medium chains 0.50 -> 0.67 -> 0.83 at 10, 50, 200 tries; 90 percent stays 0.33; short chains stay 0 at 260 to 350 s a draw |
| Task 12 | witness replay through the construction API | `tests/unit/test_witness_replay.py`, apollo `runs/fill/*_replay_{nohint,hint}.log` | grammar sufficient at scale with the witness as prioritiser (first Zephyr instance, 366 variables: valid in 445 decisions, 267 s); unhinted heuristic order covers almost nothing; full corpus replay running |
| direction 3 | adaptive allocation of reads vs uniform, real evaluator, disjoint assessment | `results/adaptive/*.log` | **done, replicated**: successive halving matches uniform at half the reads on both hosts, -0.002 [-0.008, +0.004] Zephyr and +0.002 [-0.004, +0.009] Pegasus; UCB slightly worse; random -0.11 to -0.12 |
| curriculum baseline | anytime minorminer on Pegasus 3 and Zephyr 2 fill corpora, 12 a cell, deadlines 10 to 120 s | `results/small/*_anytime.log` | **done**: short chains from 80 percent 0 to 0.17 at 120 s with 20 to 40 attempts; medium chains 0.42 / 0.17 (Pegasus 3) and 0.75 / 0.08 (Zephyr 2) at 90 / 95 percent |
| direction 4, RL | REINFORCE on the constructive policy (policy = prioritiser), from scratch, Pegasus 3 and Zephyr 2 fill corpora, deadline 60 s an episode; anytime baseline on the same corpora at 10 to 120 s | apollo `runs/rl/*_scratch.log`, goose `runs/small/*_anytime.log` | running |
| direction 5 | labels on the 2400-instance Pegasus 6 corpus for the data-scale test | `results/relabel/pegasus6_large_labels.log`; apollo `runs/large/train_seed*.log` | labels **done**: 3298 training states, 696 held-out, 24,123 candidates, 3 h; oracle minus random +0.108 and +0.099 on 696 held-out states; three trainings running |

Checkpoints and gates are in `docs/plans/2026-09-15-plan.md`; decisions in `docs/decisions/`;
the two reviews in `docs/review/`. When a row above finishes, its log is copied to `results/`
and its number is written into the section of this file that it belongs to, in the same commit.

## Where it stands in one paragraph

The reinforcement embedder does not beat a random controller at matched episodes by enough to
claim it: +0.0157, +0.0262 and +0.0135 on ninety test lineages, every interval containing zero.
What the work has established instead is where the difficulty lies. Choosing among the candidates
at a single state is worth about +0.11 to +0.13 in solve probability. The specified network can
fit that choice exactly on states it has seen, and transfers almost none of it. Reading the
compiled program rather than the environment state recovers about +0.012 of it. And the number of
qubits an embedding uses carries no information about its quality at all.

## Claims and their standing

| Claim | Number | Standing |
|---|---|---|
| Objective selection beats resource selection | +0.0714 at N=2 to +0.2564 at N=40 | holds |
| Resource selection is flat in the number of candidates | 0.483 to 0.510 over a twentyfold increase | holds |
| A structural scorer does not beat resource selection | -0.022 to -0.031 | holds |
| Ceiling for choosing among candidates at one state | +0.1123 [+0.0832, +0.1440] | holds |
| The network fits that choice exactly on seen states | regret -0.0000 and +0.0031 | holds |
| It does not transfer | -0.029, -0.020, -0.014 | holds |
| Reading the compiled program instead of the state helps | +0.0093, +0.0109, +0.0157 | holds, three seeds, intervals contain zero |
| Qubit count predicts quality among sampled candidates | -0.002, 0.0% of variance on the hard corpus | holds, but answers the wrong question |
| A larger budget buys a better best embedding | -0.0241 easy, +0.0141 [-0.0064, +0.0340] hard | untested: the candidates span 5 qubits, not a budget range |
| Registered bar, RL best-of-8 against random best-of-8 | +0.0157, +0.0262, +0.0135 | holds: does not pass |
| The improvement operator degrades a selected embedding | -0.057 at one round, -0.085 at two | holds |
| Capacity, teacher, loss or labels explain the failure | | ruled out, each separately |

## Work done

**Measurement repairs, from five external audits.** The comparator that measured a policy against
a different embedding than the one it was given. A `.gitignore` line that silently excluded the
five logs the README cited. The teacher that gated on the block which made a rollout the winner
rather than on an independent re-measurement. Pinning the starting embedding, which also pinned
what RESTART could reach, so training met an action deployment would not give it. COMMIT resolved
as workspace plus new_chains when the environment returns the archive entry it names, wrong on
60 of 178 COMMIT rows and on none of the 60 at a root state. A within-lineage baseline that
included the episode's own return and shrank the gradient by (n-1)/n. The in-training curve
reading validation and test pooled. Best-of-n counting successes rather than attempts. Wall clock
counted twice on one arm and at zero on another. Ranks that broke ties by position. Fresh-read
seeds drawn from Python's salted string hash, the trap this repository had written a hand-test
for the same week.

**Environment repairs.** RESTART was unreachable after a single rewrite because restores that
rebuild the same assignment held every slot; identical successors are collapsed and the escape
family keeps a reserved share. `compile_program` and the evaluator sort every label before
writing a coefficient, because a frozenset of qubit labels iterates in per-process hash order and
one embedding measured in two processes returned two utilities with the seed pinned.
`Context.beta_range` registers the annealing schedule, because left to itself the sampler derives
beta from the programmed coefficients and cancels exactly the energy compression the theory is
about: shrinking every coefficient to a sixteenth changes utility by +0.0000 with auto beta and
by -0.4195 with a registered range.

**Experiments.** The registered comparison on all ninety test lineages. The cost-utility frontier
against minorminer with selection and assessment separated. Supervised quality with capacity,
early-stopping and loss arms. A representation ablation between reading the state and reading the
compiled program. A measurement of whether a resource-quality trade-off exists at all.

**Corpora.** The ink-drop corpus is 4.48 independent components per instance, largest holding 11
of a nominal 16, three spins coupled to nothing, chains of one or two qubits, minorminer done in
two milliseconds. `gen_hard_corpus.py` starts from a dense logical graph instead, plants a
frustrated-loop Ising for a certified ground energy and asks minorminer whether it embeds within
the cap. Clique-16, digest `e5382da2349ebafb`: 45.3 edges, 0.42 isolated spins, 1.42 components,
43.2 qubits, longest chain 4.36. Then, because 600 draws of one shape is not a benchmark, seven
families across three sizes filled round-robin, digest `b1167177611d43a4`: 630 instances, 30.6
edges, headroom 0.4098, and the family recorded in each lineage so a split can hold out whole
structures rather than only other coefficient draws.

## Running now

On apollo: eight representation arms on the clique-16 corpus, F0 against Fpos (native hardware
coordinates), Fphys (the energy margin of each chain's cheapest cut) and Fall, two seeds each,
one shared label cache. And a learning curve, fitting on 25, 50, 100 and 200 lineages with the
held-out set fixed and the training subsets nested, two seeds each.

On goose: label collection for the diverse corpus, 220 training and 70 held-out lineages across
seven structural families.

The learning curve is the one that matters. Capacity, the loss, the labels, the teacher and the
representation have each been ruled out or bounded, one at a time. The quantity nobody has varied
is how much data the thing is fitted on: 389 states from 200 lineages is about three thousand
labelled candidates for a function over program graphs. If the curve is still climbing at 200,
the bottleneck is data and the other effects are noise around it.

## Open, in the order the evidence points

1. Does reading the compiled program transfer better on the hard corpus, where chains are long
   enough for chain integrity to bind? The eight arms answer this.
2. Do the hardware coordinates or the chain-robustness margin add anything on top?
3. Does any of it transfer across structures rather than across coefficient draws? The diverse
   corpus makes this askable for the first time; nothing in this repository has answered it.
   Sections 7.2 to 7.4 of the meeting design are implemented and unrun: residual reachability
   after occupancy and faults, free volume at three radii, and first-edge directional capacity.
4. Only then: whether a learned scorer can stand in for evaluator budget at deployment, which is
   the claim that would be worth a paper.

## Spending qubits: the one place a positive result appeared and did not survive

Growing a valid embedding's chains to lengths drawn from a truncated power law gives several
embeddings of one problem, nested in resource, all realising the same couplings. Three ways to
spend the extra qubits: lengthen a chain, close cycles inside it, or grow toward the chains of
coupled variables so a logical coupling is carried by more contacts.

Which topology this runs on decides whether two of the three exist at all. On Chimera 4, mean
degree 5.5, growing for redundancy moves the bridge fraction from 1.000 to 0.975, which is
nothing. On Pegasus 6 it reaches 0.806 and on Zephyr 4 it reaches 0.731, and contacts per logical
edge rise by 55 and 76 percent rather than by a tenth. The same clique-16 costs 21 to 23 qubits
there against 41 on Chimera. Every result in this repository before this point was measured on
Chimera, which is neither the current hardware nor a typical one.

At two repeats per arm, contact growth read +0.0996 on Pegasus and +0.0636 on Zephyr against the
starting embedding, with intervals excluding zero. That was the maximum over two noisy blocks
while the start was measured once. At one repeat, on ninety lineages and a fresh seed, the same
arms read -0.0010 [-0.0265, +0.0258] and -0.0081 [-0.0379, +0.0222]. The effect was the
inflation, and the inflation is about +0.10, which is larger than any real effect measured in
this project.

What holds is the ordering. At the same cost, contact-seeking growth beats redundancy-seeking by
+0.073 on Pegasus and +0.094 on Zephyr, and beats lengthening by +0.012 and +0.048. How a qubit is
spent matters a great deal; spending more of them does not improve quality on any topology tested.
The claim that unused hardware is an opportunity is not supported by this test.

## The learning curve is flat, and the list of suspects is empty

Held-out gain of the successor scorer against random selection, two seeds per point, one cache:

    lineages fitted     25        50        100       200
    held-out gain    +0.0235   +0.0088   +0.0096   +0.0199   (three earlier seeds at 200: +0.0120)

Eight times the data changes nothing. Every point sits in [+0.009, +0.024] with intervals that
contain zero. Data joins capacity, loss, labels, teacher and representation as a ruled-out
cause. On the hard corpus the four representation arms read +0.0074, -0.0045, +0.0071 and
+0.0018 held-out, all within seed noise. Logs in `results/curve/` and `results/hard_abl/`.

## Learn to propose, not to predict: the pool ceiling

If quality cannot be predicted, a learned embedder can only help by proposing a better set of
candidates for measurement to choose from. `probes/pool_ceiling.py` measures the ceiling of
four pools of eight measured candidates each, Pegasus 6 and Zephyr 4, ninety lineages, the
chosen candidate re-measured on independent reads:

    pool                                      Pegasus 6                 Zephyr 4
    eight minorminer draws                    0.8232                    0.8500
    eight draws grown toward coupled chains   -0.0086 [-0.026, +0.010]  -0.0084 [-0.027, +0.011]
    four draws and the grown copy of each     -0.0635 [-0.091, -0.038]  -0.0404 [-0.066, -0.015]
    eight of twenty-four, chosen for distance +0.0040 [-0.026, +0.035]  +0.0140 [-0.015, +0.043]

Eight independent draws is the best pool measured. The mixed pool is the informative row: its
candidates are no worse one by one, but a grown copy is correlated with its parent, so the pool
has four independent seeds where the others have eight. Pool diversity is worth more than any
way of spending qubits, and choosing draws for structural distance does not add to it.

The first run of this probe built the mixed pool from the best half of each other pool and
read +0.011 and +0.021 with intervals excluding zero. That was a maximum over sixteen measured
candidates against eight. It is the eleventh withdrawn number and it was withdrawn before it
left the log. All three versions are in `results/modern/*/pool*.log`.

## Where the learned embedder stands

Against minorminer with measured selection at matched budget, no learned component tested here
has a lever: not predicting quality at any data scale, not spending qubits in any of three
ways, not proposing for diversity. The finding that stands is the one every measurement
agrees on: quality is not a function of structure the model can see, it is a function the
sampler has to be asked, and asking scales. That finding is the paper's claim. A learned
embedder is not, on this evidence.

Two regimes are untested and are where a learned embedder could still be measured to win:
instances near the embeddability threshold, where minorminer's draws fail often and validity
rate is the score; and allocation of measurement budget across candidates, which is a bandit
over the pool rather than a change to it.

## Representation, third corpus, family held out

Five arms on the diverse corpus with the modular family held out entirely, one seed, 44
lineages: F0 +0.0433, Fpos +0.0457, Fphys +0.0544, Fspace +0.0295, Fall +0.0137, every
interval about 0.09 wide and every pair within 0.02. No arm separates from the bare program.
That is three corpora and three splits on which the representation does not matter. Logs in
`results/diverse_abl/`.

## Planted fill is not hardness; minimal fill is

`probes/gen_fill_corpus.py` plants a chain partition of the host at a chosen fill and takes the
quotient as the logical graph, so a valid embedding at that fill is known. With chains of one
or two qubits (alpha 3.0) minorminer at ten tries finds an embedding on none of the instances
from eighty percent up, on either host. With longer chains (alpha 1.5 and 1.0) it finds one on
all of them up to ninety-five percent, and its own embedding fills sixty to eighty-seven percent
of the host: a long-chain witness is a wasteful embedding, and the tool compresses it.

So the fraction of the host an embedding uses says nothing about the instance. What does is the
fraction the instance cannot do without, its minimal fill, and the advisor's scale (ninety
slightly hard, ninety-five hard, a hundred infeasible) has to be read on that quantity. Chains of
length exactly one make it exact: the logical graph is then an induced subgraph of the host on
n = f·|H| nodes, every embedding needs at least n qubits, and the witness uses exactly n. That
sweep is running. Logs in `results/fill/`.

## Relaunch, 2026-09-15: two reviewers, two defects, a spec before code

Two independent critical reviews (`docs/review/`) were run against the premise that the method
and the objective are right, so a failure to learn is a defect. GPT-6 astra found two:

- **D1.** The successor scorer's input was compiled with chain strength ratio times mean|J|
  (`probes/train_successor.py:compile_for`); the environment that produced every label uses
  ratio times the RMS coefficient scale (`src/isingfold/rl/program.py:strength_registry`). On
  the audit fixtures the shown strength was 8.6, 18.5, 1.0 and 43.9 percent above the evaluated
  one; on the unit fixture 12.6 percent. The model fitted labels of a program it never saw.
  Fixed in `cf62297`, pinned by `tests/unit/test_probe_compile_matches_env.py`. ADR-001.
- **D2.** `Context.beta_range` defaulted to None and no labelling or assessment probe set it, so
  every quality label was measured under the auto schedule that cancels energy compression,
  which the repository had already shown (`results/audit/`). The registered range (0.1, 2.0)
  is now the default in `probes/_context.py`, so every earlier quality number is a number about
  a different objective. Fixed in `4ed5771`, pinned by `tests/unit/test_registered_schedule.py`.
  ADR-002.

The reviews also narrow what the earlier results rule out. The pool-ceiling result constrains
one growth distribution, not every local policy. The flat learning curve covers lexicographic
subsets of one cache with a mismatched encoding. Label reliability was already measured in
audit 3: within-state correlation 0.979 to 0.998 across two 512-read blocks. So the record
supports "the tested models did not transfer", not "nothing is learnable". Spec, three ADRs
and the task plan are in `docs/specs/`, `docs/decisions/`, `docs/plans/`.

The test suite was testing the wrong tree on the remote hosts (an editable install of the old
workspace shadowed PYTHONPATH). `tests/conftest.py` pins it to this repository; baseline on
apollo is 1234 passed, 16 failed, 5 collection errors, every failure and error in tests that
depend on LAC_B or on runtime assets outside this tree. Noticed, not touched.

Fill regime, further facts: with chains of length one, where minimal fill is exact, minorminer
finds nothing from 70 percent up on either host at 60 to 155 seconds a draw
(`results/fill/*_exact.log`). Pegasus 6 at fifty tries: 80 percent with medium chains rises
from 0.33 to 0.83 and 85 percent from 0 to 0.17; everything else stays at zero.

## The anytime baseline names the regime (Task 10, Zephyr 4)

minorminer restarted with fresh seeds until a wall-time deadline, every attempt counted, six
instances a cell (`results/fill/zephyr4_anytime.log`):

    cell                 <=30 s   <=60 s   <=120 s   <=300 s   attempts
    80 percent, medium    0.83     0.83     1.00      1.00       1.2
    85 percent, medium    0.33     0.33     0.67      0.83       1.3
    90 percent, medium    0.00     0.00     0.17      0.33       2.5
    95 percent, medium    0.00     0.00     0.00      0.00       3.0
    short chains, any     0.00     0.00     0.00      0.00       2 to 5

At five minutes the standard tool is at zero on every short-chain fill and at a third or
less from 90 percent with medium chains. Pegasus 6 is harder (`results/fill/pegasus6_anytime.log`):
only 80 percent with medium chains reaches 1.00 at 300 s, and every other cell is at zero at
every deadline. That is the regime a learned constructor is measured
in, at the same deadline. With the quality endpoint (Task 9) and the encoding comparison
(Task 7) done, the three gates of ADR-003 are passed on Zephyr; Pegasus is four instances
from done.

## Adaptive allocation of reads (direction 3)

Zephyr 4, 63 held-out states in 32 lineages, every read a real evaluator call, the chosen
candidate assessed on 512 independent reads (`results/adaptive/zephyr4.log`):

    arm                     reads   selected quality   minus uniform, 95% over lineages
    uniform, 8 x 256         1544       0.5043
    successive halving        760       0.5025          -0.0018 [-0.0078, +0.0035]
    UCB, blocks of 32         772       0.4972          -0.0071 [-0.0148, -0.0002]
    random, no reads            0       0.3809          -0.1234 [-0.1617, -0.0856]

Pegasus 6, 61 states in 32 lineages (`results/adaptive/pegasus6.log`): uniform 1574 reads
0.4456; halving 775 reads 0.4477, +0.0021 [-0.0042, +0.0088]; UCB 787 reads -0.0048
[-0.0133, +0.0047]; random 0.3338, -0.1118.

Successive halving reaches the quality of uniform selection with half the measurement, on
both current topologies. This
is the first positive result of the relaunch, and it is the measured-selection finding made
into a mechanism: the reads that decide are the ones spent on the contenders. A learned
prior over candidates can only be judged against this, not against uniform.

## Checkpoint 2: the corrected surrogate does not transfer either (Task 7)

Six trainings per host on the registered-schedule labels, three seeds with the corrected
encoding and three with the legacy one, identical labels, picks and assessment seeds, 36
held-out lineages per host (`results/corrected/*_seed*.log`):

    held-out gain      corrected                      legacy
    Pegasus 6          +0.006  -0.002  +0.006         +0.039  -0.007  +0.002
    Zephyr 4           +0.029  +0.037  +0.040         +0.042  +0.037  +0.043

The two encodings differ by less than 0.01 where the seed spread is 0.04, and no corrected
interval lies above zero. The ceiling measured on the same labels is +0.11 to +0.12 (Task 8).
D1 was a real defect and not the cause. By the rule written in the spec before the run, the
improvement surrogate is retired as a method; what remains of the prediction direction is the
data-scale closure on the 2400-instance corpus, which decides whether it is closed for good.

## Label reliability under the registered schedule (Task 8)

Two independent 512-read blocks per candidate, 61 and 63 states in 32 lineages per host,
bootstrap over lineages (`results/corrected/*_reliability.log`):

    quantity                         Pegasus 6                 Zephyr 4
    rank correlation A vs B          +0.72 [+0.64, +0.79]      +0.77 [+0.72, +0.82]
    top choice agrees                 0.59 [0.45, 0.73]         0.64 [0.52, 0.77]
    select on A, assess on B         +0.109 [+0.086, +0.133]   +0.122 [+0.094, +0.152]
    oracle on B, inflated            +0.119                    +0.129
    spread within a state             0.24                      0.27

The select-on-A row is the ceiling for anything trained on these labels: a perfect predictor
of the 256-read label earns +0.11 to +0.12 on fresh reads. The winner's curse in the oracle
row is about 0.01. The labels are reliable; the gap between the learned head's +0.01 and this
ceiling is generalisation, now with an interval.

Labels under the registered schedule with the corrected compiler exist for both modern
corpora (`results/relabel/`, 2298 and 2329 labelled candidates, 17 and 19 minutes). A side
observation from the labelling run's own one-epoch head, preliminary at 69 and 71 held-out
states: on Pegasus 6 picking the cheapest candidate scores 0.459 against 0.408 for random
choice and 0.525 for the label oracle; on Zephyr 4 it scores 0.438 against 0.440 and 0.537.
Under the auto schedule resource selection sat at chance on every corpus. A fixed temperature
makes compression cost something, which is what ADR-002 says it should; measured selection
still beats resource selection by about twice its margin. The corrected-versus-old comparison
and the reliability probe are running on these labels.

Task 9, Pegasus 6, registered schedule (`results/fill/pegasus6_witness_registered.log`): at
80 percent with medium chains, the only cell where minorminer finds anything, witness minus
minorminer best on the residual is -0.0316 [-0.0471, -0.0166] over 5; solve probability zero
on both sides everywhere.

Task 9, Zephyr 4, registered schedule (`results/fill/zephyr4_witness_registered.log`): paired
witness minus minorminer best of four on the energy residual, lower is better: 80 percent
-0.0207 [-0.0300, -0.0095] over 6, 85 percent -0.0148 [-0.0296, +0.0035] over 5, 90 percent
-0.0161 [-0.0344, +0.0022] over 2. Solve probability is zero on both sides everywhere. The
quality endpoint at the fill regime exists and it is the residual under the registered
schedule; the cells are small and the thirty-per-cell corpus of Task 10 is where it gets its
final interval.

Task 12 at corpus scale: the first replay on Pegasus 6 stalled within four decisions, for two
reasons found by direct diagnosis, both invisible at 100 qubits. One is structural: a ROUTE may
only add qubits that realise a demand between placed chains, so a chain's further qubits
could never appear before the neighbours that need to touch them, and those neighbours could
not be placed first. REWRITE_ONE accepts a superset chain, so construction gained a "grow"
family. The other is coverage: a witness root is one qubit among about fifty adjacent free
qubits, and a budget of a few roots per variable in name order covers it about half the time
per step. The registered cap of 64 candidates applies to a decision, not to the generator's
internal ranking, so the generator gained a preference hook consulted before truncation. With
the witness as the preference, the replay reaches a valid COMMIT on a 366-variable Zephyr 4
instance in 445 decisions; without it, the heuristic order stalls at once
(`results/fill/*_replay_nohint.log`). That is the design for the constructive policy: the
learned scorer is the prioritiser of its own shortlist.

Task 12 done in the environment (`src/isingfold/rl/proposal.py`, `probes/_context.py`): PLACE
follows placed logical neighbours and offers a spread of roots for the first placement; the
horizon scales with the instance; ROUTE offers, after the router's path, one-qubit bridges to
either owner and two-owner meeting routes, all charged by the neighbour scans that found them.
A planted witness now replays to a valid COMMIT on 16 and on 100 qubits with 70 to 76
variables, three seeds, chains inside the witness's. The suite shows no new failure.

Construction API, measured before any policy: on a 16-qubit toy host the environment in
construction mode reaches COMMIT with exactly the planted witness in 12 decisions when each
step picks a witness-consistent candidate (`tests/unit/test_witness_replay.py`). On a 100-qubit
host with 76 variables it fails at step 0: PLACE offers 24 roots, the lexicographically first
ones, for the first unplaced variable only, and the witness root is not among them. That, and
the 32-decision horizon, is what Task 12 has to change before imitation is possible.

The energy residual discriminates where solve probability reads zero: on Zephyr 4, under the
auto schedule, the witness sits at 0.122 to 0.132 and minorminer's best of four at 0.142 to
0.147 in the three cells where both exist (`results/fill/zephyr4_witness_res.log`); six
instances a cell, no interval. The registered-schedule run with paired intervals is Task 9.

## Three facts about the fill regime, measured before anything is built for it

- **The in-tree constructor is at zero.** `router_initializer`, the greedy degree-order
  placer behind the environment's PLACE and ROUTE proposals, produces no valid embedding on
  any fill-planted instance at 80 to 95 percent on either host, 24 draws per cell, under a
  second each (`results/fill/*_construct.log`). A learned constructor starts from that floor.
- **Solve probability is zero for every embedding at this scale.** On Zephyr 4 the witness and
  minorminer's best of four both read 0.0000 in every cell (`results/fill/zephyr4_witness.log`).
  At 280 to 440 variables of frustrated loops, 512 reads never reach the planted ground state,
  whatever the embedding. The registered objective does not discriminate here; the mean energy
  residual above the planted ground energy, which the evaluator already computes, is being
  measured in its place.
- **Five times the budget moves minorminer one cell.** Zephyr 4 at fifty tries: 85 percent fill
  with medium chains rises from 0.50 to 0.67 and 90 percent stays at 0.33, at 35 to 80 seconds
  a draw; short chains stay at zero (`results/fill/zephyr4_budget.log`).

With chains of length exactly one, where minimal fill equals planted fill, minorminer finds
nothing from 70 percent up on either host at 60 to 100 seconds a draw
(`results/fill/*_exact.log`, running).

## The embeddability threshold on Pegasus 6 and Zephyr 4

Validity rate of minorminer at ten tries, six instances by four draws per cell, in
`results/feasibility/`:

    family      Pegasus 6                          Zephyr 4
    clique      59: 1.00  60: 0.75  61: 0.38  62: 0.08   57: 1.00  58: 0.83  59: 0.58  60: 0.21
    dense       128: 1.00  136: 0.25  144: 0.00           136: 0.92  144: 0.00
    scalefree   176: 1.00  184: 0.62  200: 0.04           184: 0.96  200: 0.00

At fifty tries the clique threshold moves by about one variable (Pegasus 61: 0.88, 62: 0.25;
Zephyr 59: 0.88, 60: 0.25) at three to five times the wall time. The deterministic clique
embedder stops at K60 on Pegasus 6 and K56 on Zephyr 4, below the random search, so the
threshold is not already solved by the standard tool. At the threshold the host is about
88 percent full and a draw costs eight to fifty seconds.

Two facts before anything is built for this regime. The current learned embedder cannot
enter it: it starts from a minorminer embedding, and here there is none. And whether a valid
embedding at the threshold has any solve probability is unmeasured; if every one is near
zero, a validity win is a win on problems the annealer cannot solve.

## Withdrawn, and why

Seven numbers have been published here and then withdrawn. Four were caught by external audit
rather than by me.

- Two improvement-target figures, -0.1139 and -0.3107, from a comparator that measured against
  the wrong embedding.
- The critic being anti-correlated at -0.19, measured with V(s) against a target V cannot
  represent; on Q(s,a) the same checkpoints are weakly positive.
- An inflation figure of 0.0303 in the README, read from the first two rounds of a sixty-round
  run and written down as if it described the run; the true mean is 0.0151.
- A reading of three validation points as a downward trend at round 13, when the full curve rose.
- "The head captures 84 percent of the ceiling on fitted states", produced by a run with no
  stopping rule and wrong labels.
- "Three unrelated methods fail in the same place", which stopped being true when one of the
  three turned out to be measuring wrongly.
- The successor scorer at +0.0362, a single seed that three seeds put at +0.0120.
- "The resource-quality premise fails." It was measured by correlating qubit count with quality
  across the candidates at a state. Those are local perturbations of one embedding, and the
  monotonicity the design claims is a property of the optimum at each budget, which that
  measurement never touches. The design says so explicitly and it was read and then contradicted
  anyway. Asked within instances instead, the sign flips on the hard corpus to +0.0141 with the
  interval containing zero. Neither measurement tests the premise: the cheaper and dearer halves
  are 43.5 and 48.9 qubits apart, a twelve percent range rather than a budget sweep. The honest
  statement is that nobody has tested it.
- A mixed proposal pool at +0.011 and +0.021, which was sixteen measured candidates against
  eight. Withdrawn from the log before it was reported.
- Contact growth at +0.0996 and +0.0636, which was a maximum over two draws against a start
  measured once. Caught before publication this time, by running the control first.
- Reading the mean column of a frontier table and calling the frontier flat, when the best column,
  which is what a frontier is, rose from 0.8242 to 0.9727 before falling.

The pattern is the same every time: report the first result, state its limits correctly, then let
it become the headline anyway. Nothing goes into the results README now before three seeds.
