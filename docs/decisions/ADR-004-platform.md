# ADR-004: The platform is propose, complete, select; the learned part sits inside the search

## Status
Accepted, 2026-09-17.

## Context
The objective is measured solution quality under the registered schedule; the standard tools
optimise resource. Every learned component tried as a replacement for search lost: a policy
that predicts quality and commits (retired, Task 7), a constructive policy that builds
without backtracking (places well, cannot finish), a discrepancy search inside the
environment (zero valid at 120 s). Every measured gain came from search plus measurement:
best of K by measured quality (+0.11 to +0.13), successive halving at half the reads, and
minorminer's router turned from zero to one by the right roots.

## Decision
The platform has three stages and the learned policy is a component of the second:

1. Propose. A layout policy samples roots (and, later, chain shapes) quickly, many per
   deadline; a router completes each layout under the qubit budget. The policy is trained
   against the measured objective of the completed embedding, validity as the gate, with the
   router alone as the baseline on the same instance.
2. Search. Layouts are sampled until the deadline; completions that fail are discarded;
   the router is whichever is strongest (the standard router today, ours when it matches it
   from witness roots, both reported).
3. Select. The valid candidates are measured under the registered schedule and the best is
   returned, with reads allocated by successive halving.

The comparison that decides the paper: this platform against the strongest resource-first
baseline given the same wall time and the same reads, minorminer restarted until the deadline
followed by the same measured selection, on Pegasus 6 and Zephyr 4, planted and application
instances, with the measured objective as the score and validity reported beside it.

## Alternatives considered
- RL replaces the search. Rejected by measurement: no formulation reached a valid
  embedding at high fill without a router, and no predictor transferred beyond +0.02.
- Search without learning. This is the baseline. The learned layout has to beat it at
  matched time, or the paper reports search and measurement alone.

## Consequences
The premise that spending qubits on contacts raises the measured objective is re-measured
under the registered schedule (`scripts/apollo/run_spend_registered.sh`); the earlier spend
tables were measured under the auto schedule and say nothing about the objective. The
hybrid runs report validity and paired residual against the router alone. Application-derived
instances are added to the benchmark before the paper's tables are final.
