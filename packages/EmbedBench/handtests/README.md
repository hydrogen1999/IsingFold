# Hand-tests for data generation

A hand-test is a case small enough that a person can check the answer by hand, run against the
real generator rather than a mock. Each one states what a human would do on paper, then does it
in code, and fails loudly when the generator disagrees.

These are not unit tests of the implementation. They are checks that the *claims the corpus
makes about itself* are true: that a planted embedding really is an embedding, that a stated
ground energy really is the minimum, that the split really is disjoint, that the instance a seed
names is the instance you get. Every claim below has been wrong at least once in some generator,
which is why it is checked here rather than assumed.

Run them all:

```
python3 handtests/run_all.py
```

Run one, with its explanation printed first:

```
python3 handtests/ht01_planted_is_an_embedding.py --explain
```

| # | what it checks | what a failure means |
|---|---|---|
| 01 | the planted chains really form a minor embedding | the witness is not a witness; every quality label built on it is void |
| 02 | the contact graph equals the logical graph the corpus ships | the problem is defined on edges the embedding does not realise |
| 03 | the stated ground energy is the true minimum, by enumeration | every solve probability is measured against the wrong target |
| 04 | the planted spin assignment attains that energy | the planting is inconsistent with its own witness |
| 05 | a large host does not silently produce an edgeless problem | the corpus looks fine and contains nothing to solve |
| 06 | lineage splits are disjoint, and instances follow their lineage | development leaks into confirmation |
| 07 | one seed gives one instance, and different seeds differ | results cannot be reproduced or are accidentally duplicated |
| 08 | the strength grid brackets the optimum it is used to find | the reported p_solve is an artefact of the grid, not of the embedding |

## What the tests print

Each one prints a line per case, so a pass is evidence rather than a green tick. Two of them are
worth reading even when they pass:

`ht05` reproduces the failure it guards against, on a real host. With confinement off, three
plantings of 16 variables on Pegasus 16 give logical graphs with 0, 0 and 1 edges, and none of
them survives to become an Ising problem. With confinement on, the same three give 21, 32 and 37
edges. That is the difference between a corpus and an empty one.

`ht08` prints the coarse and fine optima side by side, so the slack in the shipped five-point
strength grid is visible: on the six cases it runs, the fine grid buys at most 0.007 of solve
probability, and every fine optimum lies inside the coarse range. One case in six puts its coarse
optimum on an endpoint, which is exactly what `EmbeddingScore.strength_is_interior` reports, and
why a run that reports many boundary optima should widen the grid.

## Adding one

A hand-test is a file named `htNN_what_it_checks.py` in this folder. `run_all.py` picks it up by
name. Follow the shape of the existing ones: a docstring that says what a person would do on
paper and what a failure means, `explain_and_exit_if_asked(__doc__)` first, then `check(...)` for
every claim and a printed line per case. Keep it small enough to finish in seconds, and make it
run against the real generator, never a mock: the point is to catch the generator being wrong,
and a mock agrees with whatever you assumed.
