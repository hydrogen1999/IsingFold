# ADR-002: The objective is measured under the registered annealing schedule

## Status
Accepted, 2026-09-15.

## Context
`Context.beta_range` defaults to None, under which the surrogate annealer derives its schedule
from the programmed coefficients. That cancels uniform energy compression: shrinking every
coefficient to a sixteenth changes utility by +0.0000 under the auto schedule and by -0.4195
under the registered range (0.1, 2.0) (`results/audit/`). Hardware anneals at a fixed effective
temperature, so compression is real there. Every quality label in `results/quality*`,
`results/r1`, `results/r2*`, `results/curve`, `results/hard_abl`, `results/diverse_abl` and
every pool ceiling was collected under the auto schedule.

## Decision
The registered schedule is part of the objective. Probes that produce labels or assess quality
construct their Context through `probes/_context.py`, which sets the registered range, and the
environment refuses to evaluate training reward under a Context without one.

## Alternatives considered
- Keep auto and report it as a limitation. Rejected: the resource-quality thesis is about
  compression, and the labels would measure a quantity that cannot show it.
- Derive the range per instance. Rejected for now: it reintroduces scale invariance by another
  route; the open question in the spec covers a hardware-derived fixed range.

## Consequences
Every earlier quality number is a number about a different objective. They stay in STATUS.md
as history. New labels are collected under the registered schedule; the corrected-scorer
comparison (ADR-001) is run on labels collected this way.
