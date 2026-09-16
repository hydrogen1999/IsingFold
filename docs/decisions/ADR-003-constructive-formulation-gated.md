# ADR-003: A constructive formulation for the fill regime, gated on three checks

## Status
Proposed, 2026-09-15.

## Context
Instances planted with a valid embedding at 80 to 95 percent of Pegasus 6 or Zephyr 4 with
short chains are found by minorminer on 0 to 33 percent of draws at ten tries, and five times
the tries moves that by one cell (`results/fill/`). The in-tree greedy constructor finds none.
The improvement environment cannot enter this regime because it starts from a minorminer
embedding. Solve probability at this scale reads zero for every embedding, so quality there has
no measurable endpoint yet.

Both reviews agree a constructive formulation is where a learned embedder can be measured to
win; the codex review adds that the existing construction scaffolding is not enough (horizon 32,
24-root lexicographic placement shortlist, evaluator hardcoded to improvement mode) and that a
witness trajectory must be representable by the action API before imitation can start.

## Decision
Build the constructive formulation only after three checks pass: (1) the corrected surrogate
comparison of ADR-001 and ADR-002 is done, whichever way it comes out; (2) a quality endpoint
at the fill regime is either established or the regime is declared feasibility-only; (3) the
anytime minorminer baseline with a time deadline and tuned tries is measured on a certified
set. Then: replay witnesses through the action API, train a completion policy from partial
witnesses, and only if it beats heuristic completion on unseen instances proceed to full
construction with recovery learning.

## Alternatives considered
- Train the constructive policy now. Rejected: without the checks a favourable number could be
  a starved baseline or an objective that does not discriminate, both already seen here.
- Give up on a learned embedder and submit the measured-selection finding alone. Deferred: it
  remains the fallback, and the checks are cheap.

## Consequences
The plan's first phase is checks, not training. The paper has two candidate shapes and the
checks decide between them.
