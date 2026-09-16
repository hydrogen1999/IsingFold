# ADR-001: Labels and observations refer to one compiled program

## Status
Accepted, 2026-09-15.

## Context
The successor scorer learns to rank candidate embeddings by a measured solve probability. The
label is produced by the environment, which compiles the embedding into a physical program with
chain strength `ratio * RMS(h, J)` (`src/isingfold/rl/program.py`). The scorer's input was
compiled by a probe-local function with strength `ratio * mean|J|`
(`probes/train_successor.py:compile_for`). Both reviews found the model fitting labels of a
program it did not see; on fixtures the strengths differ by up to 44 percent, which changes
the physical couplings and every derived robustness feature.

## Decision
There is one compiler and one strength registry, in `src/isingfold/rl/program.py`. Anything that
produces an observation of an embedding calls them with the same context the labeller used, and
asserts digest equality with the program the evaluator ran. Probe-local compilers are removed.

## Alternatives considered
- Keep the probe compiler and add the registry strength as a feature. Rejected: the model would
  still see wrong physical couplings, and two compilers drift again.
- Trust that raw h and J let the network reconstruct the right strength. Rejected: that is the
  claim under test, not a reason to keep the defect.

## Consequences
Every cached label set built before this decision has observations that do not match its
labels and must be rebuilt or re-encoded from the stored embeddings. The comparison old versus
corrected on frozen pools is the first experiment of the relaunch.
