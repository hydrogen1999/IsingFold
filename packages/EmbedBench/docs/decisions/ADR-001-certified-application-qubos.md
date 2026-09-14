# ADR-001: Use certified application QUBOs and exact Ising conversion

## Status

Accepted

## Date

2026-09-12

## Context

The IsingFold corpus needs application-shaped logical Ising problems whose identities,
ground states, and constraints are scientifically defensible. The embedding method is
trained and evaluated against connectivity and downstream solution quality, so a malformed
logical objective would invalidate every later label even if the embedding itself were a
valid graph minor.

The previous scheduling generator wrote binary-QUBO-like coefficients directly into an
Ising model over spins in `{-1,+1}`. It penalized selecting two start times but did not
require selecting one, checked only equal-time machine conflicts instead of interval
overlap, and could generate an infeasible 2-job, 2-machine, horizon-2 instance. Exhaustive
audit found no guarantee that a ground state represented a feasible schedule.

The previous portfolio generator sampled a floating covariance matrix, discarded
correlations below a quantile, and added a maximum spanning tree to reconnect the graph.
That graph repair changed the mean-variance problem and did not enforce the stated asset
cardinality. It also lacked a proof that a ground state satisfied the application
constraint.

A separate corpus audit also found that the old Jobshop seed affected only a few discrete
processing times. Distinct planned lineages could therefore produce identical logical
problems after gauge transformation. Adding a lineage nonce to the problem identity would
hide repeated training units rather than correct the generator.

## Decision

### Common binary QUBO to Ising rule

Application constraints are first defined over binary variables `x_i in {0,1}`. For

```text
E_Q(x) = c + sum_i a_i x_i + sum_{i<j} b_ij x_i x_j,
```

the generator substitutes `x_i=(1+s_i)/2` exactly. It records

```text
h_i      = a_i/2 + sum_{j != i} b_ij/4,
J_ij     = b_ij/4,
offset   = c + sum_i a_i/2 + sum_{i<j} b_ij/4.
```

All source coefficients are integer numerators over a fixed power-of-two denominator. The
Ising coefficients and offset therefore remain exact dyadic rationals. Each application
record contains the QUBO numerators, the QUBO and Ising denominators, the substitution, and
the offset. Exhaustive tests compare integer QUBO energy with Ising energy plus offset for
every state at every production size up to 16 variables.

### Time-indexed Jobshop scheduling

There is one binary variable `x_(o,t)` for each operation and candidate start time. The
nonnegative application cost is seeded priority-weighted completion plus weighted
tardiness:

```text
c_(o,t) = priority_o (t+p_o)
        + tardiness_weight_o max(0, t+p_o-due_o).
```

Let `U=sum_o max_valid_t c_(o,t)` and `P=U+1`. The QUBO adds:

1. `P(sum_t x_(o,t)-1)^2` for exactly one start per operation;
2. `P x_(o,t)` for every start that finishes after the horizon;
3. `P x_(o,t)x_(o',t')` for every precedence-violating pair;
4. `P x_(o,t)x_(o',t')` for every true half-open interval overlap on one machine.

Every feasible schedule has energy at most `U`. Every infeasible assignment incurs at least
one penalty and has nonnegative application cost, so its energy is at least `P>U`. Thus
every ground minimizer is feasible.

The production wrapper uses the following constructive families:

| Logical variables | Jobs | Machines | Horizon | Processing-time rule |
|---:|---:|---:|---:|---|
| 6 | 2 | 1 | 3 | seeded positive composition with total 3 |
| 8 | 2 | 1 | 4 | seeded positive composition with total 4 |
| 10 | 2 | 1 | 5 | seeded positive composition with total 5 |
| 12 | 2 | 2 | 3 | unit duration with a constructive pipeline witness |
| 14 | 2 | 1 | 7 | seeded positive composition with total 7 |
| 16 | 2 | 2 | 4 | unit duration with a constructive pipeline witness |

These are synthetic time-indexed scheduling instances. The 2-machine subset deliberately
uses unit durations. The release does not claim broad industrial Jobshop coverage. Seeded
priority, tardiness, due-date, processing-order, and route information is part of the
application objective or feasible construction, not identity salt.

### Cardinality-constrained binary portfolio

The portfolio generator creates an integer positive-definite covariance matrix

```text
Sigma = A A^T + diag(D),   D_i > 0,
```

and retains every covariance entry. For a recorded cardinality `K`, the application
objective is

```text
f(x) = risk x^T Sigma x - mu^T x.
```

Because `risk>0` and `Sigma` is positive semidefinite,
`L=-sum_i max(mu_i,0)` is a global lower bound on `f` over all binary states. The generator
constructs and records one `K`-asset witness with objective `F`, then sets `P=F-L+1` and
adds `P(sum_i x_i-K)^2`.

Any wrong-cardinality state has energy at least `L+P=F+1`, while the witness has energy
`F`. Every ground minimizer is therefore cardinality-valid. Exhaustive tests also verify
that the QUBO ground energy equals the exact constrained argmin of the unpenalized
mean-variance objective.

### Contrast-sensitive Graphcut MRF

For observed intensity `y_i` and binary label `x_i=(1+s_i)/2`, the Graphcut unary
field is `h_i=1-2y_i`. It therefore represents twice the squared data term,

```text
2 (x_i-y_i)^2 = (1-2y_i) s_i + constant_i.
```

For contrast weight `w_ij=beta exp(-(y_i-y_j)^2/0.5)`, the coupling
`J_ij=-w_ij` represents twice the Potts disagreement cost because
`2 w_ij [x_i != x_j] = w_ij - w_ij s_i s_j`. Exhaustive tests check the full
energy equality, including the state-independent offset, for both registered
neighbourhoods. The latent disc only generates a noisy observation. It is not a
planted optimum or a reference certificate, and the release makes no such claim.

### Corpus identity and release contract

The Jobshop and portfolio generator identifiers are version 2. The complete corpus
generator protocol and release identity are version 4. The generation-plan schema is
version 2 and freezes a pre-generation policy requiring global uniqueness of both
`split_unit_id` and exact logical `problem_sha256`.

Preflight version 2 records the canonical ordered map
`(lineage_id, prospective_slot, split_unit_id, problem_sha256)`, its count and digest, the
plan commitment, staged source commitment, and generation-provenance commitment. It fails
before shard generation on any repeated identity. Apollo and Goose must replay this
preflight under the pinned runtime before canary or production generation, and each shard
command must authenticate the same commitments.

The version 4 OOD claim concerns registered topology/host-scale, fault, and
distribution-regime shifts. Graphcut, portfolio, and Jobshop are all present in train,
validation, and test. The unseen-family matrix in `docs/HARD_OOD_CORPUS_SPEC.md` is a
separate future or legacy benchmark profile and is outside this release.

## Alternatives considered

### Write binary penalties directly as Ising coefficients

Rejected. Binary and spin variables have different algebra. Omitting the exact variable
substitution changes energies and can make infeasible states optimal.

### Use pairwise at-most-one scheduling terms only

Rejected. At-most-one allows the all-zero assignment. Equal-start machine checks also miss
overlap whenever processing duration exceeds one time unit.

### Use a heuristic fixed constraint penalty

Rejected. A fixed number has no guarantee under changing instance scale or seeded costs.
The chosen penalties follow explicit lower and upper bounds and are checked exhaustively.

### Threshold portfolio covariance and reconnect it with an MST

Rejected. Thresholding and graph repair solve a different optimization problem and destroy
the direct interpretation of `Sigma`. The exact cardinality penalty already yields a dense
logical graph without altering covariance.

### Salt logical identities with lineage IDs

Rejected. A salt would label byte-identical problems as independent examples. Generator
entropy must enter the application objective or structure, and preflight must still reject
any residual collision.

### Use Minorminer outputs as application labels

Rejected. Minorminer is a resource-oriented proposal baseline, not the authority for the
connectivity and downstream solution-quality objective learned by IsingFold.

## Consequences

- Application feasibility and QUBO-to-Ising equivalence are mechanically testable with
  exact integer arithmetic.
- Portfolio logical graphs are complete and scheduling graphs can be dense. Embedding them
  is harder, but the difficulty now comes from the declared application problem rather than
  a graph-repair heuristic.
- Ising offsets must be retained whenever energies are compared with the original binary
  objective. Offsets do not affect embedding or the minimizing spin assignment.
- Existing plan, preflight, shard, canary, and runtime commitments from earlier protocols
  cannot be reused. New artifacts must be published to new paths.
- The installed third-party wheel inventory is unchanged by this decision. Source,
  generation-provenance, runtime-lock, plan, preflight, canary, and release digests change.
- The Jobshop corpus remains a controlled synthetic benchmark. Broader routes, more jobs,
  and non-unit multi-machine processing durations require a new generator version and a new
  feasibility proof.
