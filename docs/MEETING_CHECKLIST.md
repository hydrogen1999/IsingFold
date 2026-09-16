# Meeting checklist

The advisor's direction, kept as a checklist so progress against it can be read rather than
recalled. Status is what the code and the logs say, checked with grep rather than from memory.
Where an item is partly done, what is missing is written out instead of being rounded up.

Legend: **done** means implemented and measured; **built, unrun** means the code exists and no
result exists; **partial** means some of the item; **not started** means absent from the tree.

## The core message

> The bottleneck is not PPO. It is the state representation, the stress tests, and the
> formulation of the trade-off.

Everything measured since agrees. Capacity was ruled out: the network fits the labels to zero
regret on states it has seen. Three losses transfer identically badly. The teacher and the labels
were wrong, were fixed, and the transfer gap did not move. Nothing in the results points at the
learning algorithm.

## 1. Why the current version is not convincing

| Point | Status |
|---|---|
| Feasibility near 99 percent proves nothing if the test is easy | **done, and quantified.** The ink-drop corpus is 4.48 independent components per instance, largest holding 11 of a nominal 16, three spins coupled to nothing, chains of one or two qubits, minorminer finished in two milliseconds |
| The model knows which qubits are occupied but not how much room is left | **done.** `probes/space_features.py` |
| ... nor how much expansion each direction allows | **done.** first-edge directional capacity, same file |
| ... nor which directions are blocked | **done.** dead-direction and blocked-neighbour counts |
| ... nor the physical position of a qubit | **done.** `native_coordinates` in `probes/successor_scorer.py` |
| ... nor the structure specific to Chimera, Pegasus and Zephyr | **partial.** Coordinates for all three families; the edge-relation encoding (internal, external, odd) is not implemented |

## 2. Features to add first

| Feature | Status |
|---|---|
| occupied / free state of a hardware node | already present in `tensorize.py` |
| remaining capacity around a position | **built, unrun.** `free_volume`, radii 1, 2, 3, plus the free component the anchor touches |
| availability per direction | **built, unrun.** `directional_capacity`, first-edge definition, never summed across directions |
| positional encoding from native topology coordinates | **done and running.** Chimera (i,j,u,k), Pegasus (u,w,k,z), Zephyr (u,w,k,j,z), from the generator's converter |
| connectivity and local structure derived from those coordinates | **not started** |
| a separate version with a clean before/after ablation | **done.** Six arms F0, Fpos, Fphys, Fspace, Fdir, Fall; F0/Fpos/Fphys/Fall running on the clique corpus |

## 3. Bottom-up, test-driven development

| Step | Status |
|---|---|
| a small setting the model can deliberately overfit, to confirm it learns the signal | **done.** With correct labels and no stopping rule it reaches the label oracle exactly: regret -0.0000 and +0.0031 across two families, rank correlation +0.714 and +0.833 |
| push instances to the feasibility threshold to find what information is missing | **not started.** minorminer never fails on any corpus here, so the threshold has not been approached |
| stress tests: a direction that looks good and closes; resource spent the wrong way; good and bad initial placement; cases needing restart; scale until failure | **not started** |
| only then move to solution quality | not applicable yet |
| state clearly what the teacher is teaching | **partial.** The teacher now accepts on an independent re-measurement and teaches the shortest prefix that produced the output, but the completion-existence and completion-policy heads of the design are absent |

## 4. Raising the paper's story

| Item | Status |
|---|---|
| current contribution: optimise downstream quality rather than a proxy | **supported.** Objective selection beats resource selection by +0.0714 at N=2 rising to +0.2564 at N=40, and resource selection is flat in the number of candidates |
| new perspective: embedding as a resource-quality trade-off | **formulated, not implemented** |
| a small experiment showing shorter chains do not mean better solutions | **done.** Within a state, the cheaper embedding scores strictly better 54.8 percent of the time on the easy corpus; on the hard corpus qubit count explains 0.0 percent of the variance in quality |

## 5. The gap: no mechanism for the trade-off

| Item | Status |
|---|---|
| a resource budget or trade-off weight as a control variable | **not started.** `qubit_budget` is absent from the tree |
| a mechanism balancing chain expansion against connectivity | **not started** |
| a conditioning parameter that can be turned up or down | **not started** |
| a rule for when an extra qubit is worth spending, and where | **not started** |
| output as an embedding for a budget, or a set on the frontier | **not started** |
| physics-informed: chain-break behaviour per chain | **partial.** The structural side is in: the energy margin of each chain's cheapest cut, from the bound dE >= 2*cut(A) - 2*L(A). The measured break-probability label is absent |
| chain connectivity, not only chain strength | **done.** bridge fraction and cut margins per chain |

## 6. Presenting the challenge

| Item | Status |
|---|---|
| quantify difficulty rather than saying the space is large | **not started** in the paper; the pieces exist as measurements |
| methodology: graph-generative model, RL, hardware-aware state, physics-informed signals, Pareto mechanism | partial, per the rows above |
| state the ML novelty concretely | **not started** |

## 7. Assigned next

| Task | Status |
|---|---|
| 1. add hardware positional, directional and available-space features | **done in code**, Fpos running, Fspace and Fdir unrun |
| 2. a separate version and a before/after ablation | **running** |
| 3. build the benchmark and stress tests before growing the model | **partial.** Two corpora built: clique-16 (`e5382da2349ebafb`) and seven families across three sizes (`b1167177611d43a4`). No stress tests |
| 4. test feasibility near the threshold, not on easy cases | **not started** |
| 5. design reward, teacher, shaping and the trade-off mechanism | **partial**, per section 5 |
| 6. write an outline: Title, Abstract, Introduction, Design, Code, Tests | **not started** |
| 7. check self-consistency between claim, design and experiment before coding it all | **not started** |

## The correction: power-law chain lengths

| Item | Status |
|---|---|
| generate embeddings with chain lengths drawn from a power law | **not started.** No file mentions it |
| decide the exponent, the bounds, whether it is truncated, and whether test carries a distribution shift | **not started.** Stated as a data-design decision needing its own ablation, not a free choice |

The current generators produce whatever chain lengths the placement happens to give: about 2 on
the ink-drop corpus and 4.36 on the clique corpus. Neither is a controlled distribution, so the
model has never been shown a deliberate mix of short and long chains.

## The correction: two mistakes in the literature

| Claim | Status |
|---|---|
| minimising resource can reduce solution quality | **supported here.** Resource ordering predicts quality ordering 41.6 percent of the time on the easy corpus and 47.7 percent on the hard one, both at or below chance |
| unused qubits are an opportunity, not waste | **not tested.** Asked within instances, the dearer half of the candidates wins in 149 of 260 instances on the hard corpus, +0.0141 [-0.0064, +0.0340]. But the two halves sit 5 qubits apart, a twelve percent range, which is a local perturbation and not a budget sweep. No generator here can build an embedding deliberately at a larger budget, so the claim is untested rather than supported or refuted |

## Joint formulation and the two challenges

| Item | Status |
|---|---|
| joint optimisation of resource and quality, output as a Pareto frontier | **not started** |
| learn when, where and how many extra qubits are worth using | **not started** |
| challenge 1, the search space, motivating sequential RL | formulated, not written up |
| challenge 2, hardware evaluation is noisy and expensive, motivating a surrogate | **partial.** `SuccessorScorer` is a quality surrogate over the compiled program. It reaches the oracle on states it has seen and transfers +0.0120 across three seeds where the ceiling is +0.1123, so it is not yet cheap reward anyone should train against |

## What the evidence currently says about the bottleneck

Ruled out one at a time: capacity, the loss, the labels, the teacher. Bounded: the representation
is worth about +0.008, which is below the spread between seeds. Never varied: how much data the
model is fitted on. A learning curve at 25, 50, 100 and 200 lineages is running, with the held-out
set fixed and the training subsets nested.

## Update 2026-09-15

- Power-law chain lengths: built (`probes/powerlaw_embeddings.py`), three spends separated,
  measured on Pegasus 6 and Zephyr 4 with one draw per arm. No spend beats the start; contact
  beats redundancy by +0.07 to +0.09 at equal cost.
- Learning curve: flat, 25 to 200 lineages. Data ruled out.
- Representation ablation on the hard corpus: four arms within seed noise.
- Pool ceiling: eight independent minorminer draws beat every grown, mixed or
  diversity-chosen pool of eight at matched measurements. Diversity is the resource and a
  variant of a draw is not a new seed.
- All earlier conclusions were Chimera-only; Pegasus and Zephyr are now the default hosts.
- Still open: feasibility-threshold regime, measurement-budget allocation, section 5 budget
  conditioning, section 9 completion heads, coordinate-permutation control, paper outline.

## Update 2026-09-15, relaunch

- Two independent reviews against the premise that the method is right. Two defects found and
  fixed with reproduction tests: the scorer saw a program compiled with the wrong strength
  (ADR-001), and every label was measured under the auto schedule (ADR-002).
- Section 3, bottom-up: label reliability now measured with intervals; ceiling +0.11 to +0.12.
- Section 3, feasibility threshold: located on Pegasus 6 and Zephyr 4 by size (`results/feasibility/`)
  and by planted fill (`results/fill/`); minimal fill defined exactly with singleton chains.
- Section 5, budget conditioning: still not started; the fill corpus is the instrument for it.
- Section 7, item 4: done. Item 6 (outline) and 7 (self-consistency): the spec in `docs/specs/`
  is the first version of both.
- Power-law chain lengths: exponent and bounds are a corpus parameter (`--alphas`, `--lmin`,
  `--lmax`); the test for distribution shift is not yet run.
- The joint formulation: a constructive MDP on the fill corpus is proposed in ADR-003 and gated.
