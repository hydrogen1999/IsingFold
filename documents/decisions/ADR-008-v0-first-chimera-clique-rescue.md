# ADR-008: V0-first defect-aware Chimera clique rescue

## Status

Accepted.

## Date

2026-09-13

## Context

The sealed v4 initializer-bank run exposed a support failure rather than a corrupt artifact. Of
6,400 conditional training episodes, 180 exhausted all 64 sealed draws. These failures represented
29 unique public training instances in one stratum:

- portfolio application instances;
- complete logical graphs of order 12;
- faulted Chimera hosts with 126 active qubits;
- embedding-hard, sampling-hard, and decision-hard labels.

Every affected prepared record already contained a valid planted embedding using 47 to 56 qubits,
so the instances were structurally feasible. The in-repository `v0` and `lns_v1` profiles failed
across larger transition and candidate limits, while stock minorminer succeeded 20 of 20 times on
one inspected instance. No evaluator targets, Ising outcomes, or hidden test records were opened in
this diagnosis. The failure therefore identified a search-support gap in the initializer, not a
quality-label result.

Silently removing the stratum would change the registered population and make the broad
generalization claim indefensible. Raising the 64-draw cap would also be misleading because every
draw failed for the same 29 instances. A versioned support repair is required before representation
or reinforcement-learning training can use a sealed initializer bank.

## Decision

Register `hybrid_chimera_clique_v1` as the production search profile and
`lac-minorminer-hybrid-chimera-clique-v1-initializer-v4` as its initializer method identity.

The profile is strictly `v0` first. It runs the unchanged `v0` search with the registered seed,
limits, timeout, and prospective work cap. If `v0` succeeds, the profile returns exactly the same
embedding, transition trace, incumbent records, and work ledger. The structural method is invoked
only after a nonfatal `v0` failure. A timeout or work-budget termination never receives extra work.
This activation order prevents successful cases from being replaced by a hand-engineered
initializer and keeps the learned-policy attribution testable.

The rescue is applicable only when all of the following hold:

1. The logical source is a complete graph with order from 3 through 256.
2. Target labels are either canonical nonnegative Chimera integers or coordinate tuples
   `(row, column, shore, lane)`.
3. The inferred host has at least two rows and two columns and tile size at most 16.
4. Every observed target edge is a legal Chimera edge.
5. At least 75 percent of nominal nodes remain active.
6. At least 75 percent of nominal edges induced by the active nodes remain present.

For an applicable host, the method constructs defect-aware L-shaped candidates. Each candidate is
the union of the connected vertical lane component and connected horizontal lane component joined
by one surviving intra-cell coupler. Two candidates are compatible only if their chains are
vertex-disjoint and at least one host coupler joins them. A deterministic, seed-tie-broken,
fixed-size clique search selects as many compatible chains as there are logical variables. The
independent public embedding validator remains mandatory before any embedding is returned.

The algorithm is hard bounded:

- at most 4,096 unique structural candidates;
- at most 100,000 clique-search states;
- clique order at most 256;
- the caller's existing wall-clock deadline;
- the caller's original nine-coordinate prospective work cap.

The work-counter schema is raised from version 2 to version 3. The nine coordinates do not change,
but the following structural meanings are now registered:

- `route_expansions`: lane-component vertex settlements;
- `materializations`: complete L-chain candidates constructed before deduplication;
- `feature_work`: topology checks, compatibility predicates, neighbor-mask updates, and bounded
  clique-search rank or intersection operations.

Every counted operation is charged before it executes. Structural work is added to the exact `v0`
prefix and checked against the original cap. Compiler calls, cut-edge visits, evaluator reads, and
internal restart work remain zero. The C++ backend and Python wrapper must both advertise schema
version 3 before the production adapter can construct a bank plan.

The experiment is versioned forward rather than repinned in place. The frozen v4 configuration,
grid, and downstream registries remain byte-identical. The hybrid experiment uses
`complete_system_lac_hybrid_cache_v1.json`, `rl_grid_hybrid_v1.json`, and separately named hybrid
external-tuning, quality-signal, warm-control, and capacity-control registries. Its staged-grid
identity is `if-core-v2-profile-i-hybrid-chimera-registered`. No artifact created under the old
grid may be relabeled as an artifact of this grid.

Diagnostics bind `search_profile`, `structural_fallback_invoked`, and a status-consistent
`profile_detail`. The complete-system adapter and serialized receipt validator reject a stale
profile, unknown detail, or detail that disagrees with the activation flag, success, timeout, or
work-budget status.

## Target-free support evidence

Before accepting this decision, the final v0-first implementation was evaluated on all 29 affected
public training instances at 64 independent initializer seeds each. It produced 1,856 valid
embeddings in 1,856 runs. Every run recorded `v0_failed_chimera_clique_success`. Under the exact
production residual cap, the observed maxima were:

- 111,576 of 191,808 feature-work units;
- 1,224 of 4,088 materializations;
- 19,530 of 200,000 route expansions.

Returned embeddings used 88 to 96 qubits, with median 93. This result establishes initializer
support only. It does not establish improved `P_solve`, connectivity utility, decoded energy, or
the claim that additional qubits generally improve solution quality. Those claims require the
registered end-to-end learned-policy evaluation.

## Alternatives considered

### Drop or relabel the hard stratum

This would condition training and evaluation on initializer success and remove exactly the case
that exposed the implementation weakness. It was rejected.

### Increase retries or transition limits

The failures persisted across 64 draws and broader local search settings. More retries would spend
additional work without adding observed support. It was rejected.

### Use stock minorminer as the production initializer

This could recover feasibility, but it would make the learned arm depend on an external baseline,
remove exact internal work accounting, and weaken the replaceable C++ framework. Stock minorminer
remains an external comparison, not a training-label authority or hidden initializer. This option
was rejected.

### Run the structural construction before v0

This succeeded on the hard stratum but changed initial embeddings for cases where `v0` already
succeeded. It would confound structural engineering with neural-policy gains. It is permitted only
as a named nonlearned ablation, not as the production activation order. This option was rejected.

### Treat the structural embedding as quality optimized

The construction targets valid clique support and fault tolerance. Its larger qubit count is a
consequence of the construction, not evidence of superior end-to-end quality. This interpretation
was rejected.

## Consequences

- The complete and quality initializer banks must be regenerated from fresh plans under the new
  method, source digest, runtime, and work schema.
- Quality labels, preflight receipts, validation bootstrap banks, or checkpoints that bind an old
  initializer bank cannot be relabeled or reused.
- Selector artifacts that do not bind initializer identity remain separate and may be reused only
  if their own frozen provenance checks pass.
- The hard training stratum remains in the registered population.
- Successful `v0` cases retain their old initializer behavior, which supports clean structural
  rescue and learned-policy ablations.
- The final paper must report initializer failures as utility zero and must report structural
  rescue separately from neural improvement.
