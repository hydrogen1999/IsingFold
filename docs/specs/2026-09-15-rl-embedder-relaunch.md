# Spec: RL embedder relaunch after the two-reviewer audit

Status: proposed 2026-09-15. Written after two independent critical reviews
(`docs/review/2026-09-15-claude-review.md`, `docs/review/2026-09-15-codex-gpt6-astra-review.md`)
and before any code. The author's premise, taken as given: the method is right and the objective
is right, so a failure to learn is a defect to be found, not a result to be reported.

## Assumptions made without the author in the room

1. The registered annealing schedule for every quality measurement is `beta_range=(0.1, 2.0)`,
   the range that made energy compression visible in `results/audit/sampler_scale_registered.log`.
   Every label collected under the auto schedule is a label of a different objective.
2. The target hardware is Pegasus 6 (680 qubits) and Zephyr 4 (576 qubits). Chimera results are
   history, not evidence about the target.
3. The A* oral bar is one sentence: a learned embedder that beats tuned minorminer at matched
   wall time on untouched instances, on both target topologies, with quality measured under the
   registered schedule and feasibility counted over every attempt. Nothing below that is the
   paper's method contribution; the measured-selection finding stands on its own either way.
4. The improvement MDP is kept alive only through the corrected surrogate. If the corrected
   surrogate transfers below +0.03 on frozen pools with a tight interval, that MDP is retired
   as a method and the constructive formulation carries the paper.
5. The constructive formulation is gated on the cheap checks codex ranked 2, 3 and 5, in that
   order, before any policy is trained on it.

Correct any of these by editing this file; the plan follows it.

## Objective

### Layout-v4 implementation assumptions (cf31efc audit)

The opt-in v4 work targets the existing empty-start hybrid root-policy probe, not the
entire relaxed-overlap PPO system. The objective, registered annealing schedule, decoder,
strength ratios and resource-as-budget contract stay fixed. Support, representation,
contextual scoring, baseline and teacher are separate ablations. Training witnesses are
allowed only after the lineage split and never in inference. Toy unit learning is a
correctness check; large numerical experiments remain on the experiment hosts. Full
design, limitations and executable recipes: `docs/review/2026-09-16-layout-v4.md`.

### Relaunch objective

Find and fix what makes a correct objective look unlearnable, then either restore transfer of
the quality surrogate or move the learned part to construction, where the standard tool fails
with a witness in hand. Success is a number that survives an independent assessment block, a
tuned baseline, and a fresh lineage split.

Two defects are already identified and are the first work:

- **D1, identical compiled program.** `probes/train_successor.py:compile_for` builds the
  program the scorer sees with strength `ratio * mean|J|`; the environment that produced every
  label uses `strength_registry`, which is `ratio * RMS(h, J)` (`src/isingfold/rl/program.py:28`).
  On the four fixtures the shown strength exceeds the evaluated one by 8.6, 18.5, 1.0 and 43.9
  percent. Fix: one compiler, one registry, digest equality asserted.
- **D2, the objective is the registered schedule.** `Context.beta_range` defaults to None and
  no quality, successor or pool probe overrides it, so their labels were measured under a
  schedule that cancels energy compression. Fix: probes construct their Context with the
  registered range, and the Context refuses to be used for labels without one.

## Tech stack

Python 3.12, `src/isingfold/rl` (environment, compiler, evaluator, tensorization), PyTorch for
the scorer, `dwave.samplers.SimulatedAnnealingSampler` as the surrogate annealer, minorminer as
the baseline, networkx, numpy. Remote hosts apollo (many cores, CUDA) and goose (4 cores).
Nothing numerical runs on the laptop; unit tests on toy fixtures do.

## Commands

```
# unit tests, laptop, seconds
python3 -m pytest tests/unit -q
# one probe on apollo, from the committed tree
ssh apollo 'cd ~/prj_IsingFold && source ~/isingfold/.venv/bin/activate && \
  ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src python -u probes/<probe>.py ...'
# launcher pattern (never pkill a pattern that matches the ssh command line)
(setsid nohup bash scripts/apollo/<script>.sh > runs/<name>.out 2>&1 < /dev/null &)
# sync the committed tree to a host
git ls-files probes src scripts | rsync -a --files-from=- . <host>:~/prj_IsingFold/
```

## Project structure

```
src/isingfold/rl/      environment, compiler, evaluator, tensorization, PPO
probes/                one experiment per file, each with a docstring stating the question
probes/_context.py     Context for a whole modern host; will also carry the registered schedule
scripts/apollo/        launchers, one per experiment, idempotent (skip finished logs)
tests/unit/            fast tests on toy fixtures; tests of probes live here too
results/<experiment>/  every log that a number in STATUS.md cites
docs/review/           the two reviews
docs/decisions/        ADRs
docs/plans/            the task list for this spec
STATUS.md              the record, including withdrawn numbers
```

## Code style

Plain functions, explicit arguments, one probe per question. A probe's docstring says what it
measures and what would make its number an artefact. Example of the guard every probe carries:

```python
sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
```

No em-dashes anywhere. Numbers in tables, not prose.

## Testing strategy

pytest, `tests/unit`. Every defect gets a reproduction test that fails before the fix. Probes
that carry a measurement invariant (equal pool sizes, assessment seeds disjoint from selection
seeds, identical program digests between label and observation) get a unit test of that
invariant on a toy fixture, so the invariant cannot drift silently again. Integration runs are
the probes themselves on apollo; their logs are the evidence and are committed.

## Boundaries

- Always: run `pytest tests/unit -q` before a commit that touches `src/` or `probes/`; commit
  logs with the number they support; verify with `git ls-files` before pushing; state
  assumptions in this file before acting on them.
- Ask first: changing `Context` defaults that other code relies on; deleting a probe; changing
  the registered strength ratios; any change to `packages/EmbedBench`.
- Never: run training or sampling on the laptop; write the GitHub token to a file; modify
  `LAC_GPT/`; report a selection-block number as a result; publish a number without its log.

## Success criteria

- S1: a unit test asserts the scorer's compiled program equals the environment's program for
  the same embedding and strength index, by digest, and it passes.
- S2: a unit test asserts that a Context built for labelling carries the registered schedule,
  and that the labelling probes use it.
- S3: on frozen pools with identical labels, picks and assessment seeds, the corrected scorer
  is compared with the old one. Either it transfers at +0.03 or better, or the interval
  excludes that and the improvement surrogate is retired with that log as the reason.
- S4: at the fill regime, quality has a measurable endpoint under the registered schedule
  (energy residual or a longer schedule) that discriminates between the witness and
  minorminer's output on the same instances, or the regime is declared feasibility-only.
- S5: the anytime minorminer baseline exists: a time deadline, tuned tries, every attempt
  counted, on a certified set of at least 30 instances per host and fill.
- S6: a witness trajectory replays through the construction action API on a fill instance, so
  imitation is possible at all, before any constructive policy is trained.

## Open questions for the author

- Is `(0.1, 2.0)` the schedule to register, or should it be derived from the hardware's
  effective temperature for Advantage-class devices?
- Is a feasibility-only contribution at 85 to 95 percent fill acceptable if quality cannot be
  resolved there, or must every claim carry the objective?


## Amendment 2026-09-17: what fill means, and the two test cases

Fill is the occupancy of the state, rho_t = |union of chains| / |V_H|, and it bounds the
action space: the free set F_t is not equally reachable from every chain, so the decision
is which chain to extend and in which direction, given what that opens or closes for the
others. The resource is a budget for downstream quality, not a quantity to minimise.

Two test cases, kept separate:

1. Completion in tight space: a partial embedding at about 90 percent occupancy with
   demands still unmet; the policy and the search finish it within the budget. Measured by
   validity against the router alone at the same deadline, with failures counted.
2. Allocation for quality: a valid embedding at about 90 percent occupancy (the planted
   witness on the fill corpora); the policy spends the remaining space to raise the measured
   objective (energy residual at this scale). Measured by the paired objective against
   random spending of the same budget and against the start, with episodes that exceed the
   budget or cannot be measured counted at the start's value minus a penalty.

The state for both carries each chain's frontiers, the free space reachable from each
within two and three steps, the unmet demands, the other chains bordering that space, and
the contact and coefficient context (`probes/candidate_features.py:frontier_features`).
Occupancy is traced along every trajectory. A mask removes overlap and disconnection; it
does not remove dead ends, which the policy has to learn to avoid.

### Deployment boundary

The cases above are controlled learning/continuation tests. The deployed embedder starts from
empty chains given the logical problem, host and declared budgets. A supplied witness is training
supervision or a diagnostic start only, never a selectable deployment fallback. COMMIT requires
a valid embedding generated during the run. The contact-growth probe reports its supplied-start
scope explicitly; its gain does not establish empty-start construction performance. Full-deployment
evaluation must include the construction cost and every failed attempt.
