# Status

A running record of what has been done, what each result is worth, and what is still open. Kept
because this project has withdrawn seven published numbers, and a list of what currently stands
is the only way to tell a finding from a leftover.

Last updated against commit `ad43c21` plus the work described under "Running now".

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
- Contact growth at +0.0996 and +0.0636, which was a maximum over two draws against a start
  measured once. Caught before publication this time, by running the control first.
- Reading the mean column of a frontier table and calling the frontier flat, when the best column,
  which is what a frontier is, rose from 0.8242 to 0.9727 before falling.

The pattern is the same every time: report the first result, state its limits correctly, then let
it become the headline anyway. Nothing goes into the results README now before three seeds.
