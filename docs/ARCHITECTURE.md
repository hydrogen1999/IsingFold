# Architecture

What the method is, as the code implements it today. Every claim here cites the file and line it
comes from. Where the architecture meets a measurement that constrains it, the measurement is
given with it, because several of the design choices below are the direct cause of results in
`docs/paper/RESULTS.md`.

---

## 1. What it is

**An independent constructive embedder.** The policy builds a minor embedding of a logical Ising
problem into a hardware graph, starting from an empty host, choosing every action itself, and
finishing when it chooses to commit. No external router completes, repairs or seeds its output,
at training time or at deployment. `docs/decisions/ADR-005-independent-constructor.md`.

minorminer appears in exactly two places and neither of them touches the learned arm: as a
certificate when an instance is generated, and as a comparison arm evaluated under the same
deadline (`probes/constructor_baseline.py`). A run with `--comparison none` invokes it not at all.

This is worth stating plainly because the project also contains a selection platform that ranks
embeddings a router produced. That is a different experiment. The word "candidate" means an
**action** here and an **embedding** there, and conflating them has caused confusion before.

---

## 2. The decision loop

One episode:

1. The environment is constructed in `Mode.CONSTRUCTION` with `initializer=None`, which is an
   empty host (`probes/constructor_rollout.py:130`).
2. At each step the environment enumerates the legal actions and presents them as a set of
   **candidate rows**, one row an action.
3. The policy scores every row, divides by a temperature, and takes a softmax over the set. In
   training it samples; in evaluation it takes the argmax (`probes/constructor_rollout.py:172-182`).
4. The chosen action is applied and the loop repeats.
5. The episode succeeds only if the last opcode is `COMMIT` and the returned embedding validates
   (`probes/constructor_rollout.py:209-210`). Anything else, including running out of decisions or
   choosing `STOP`, is a failure worth zero.

The action set has no fixed size, so the policy cannot have one output per action. It is
parameterised as a **per-action scorer**: the same weights are applied to every row and the
softmax is over however many rows the environment offered.

---

## 3. The action space

Eight opcodes, from `isingfold.rl.contracts.Opcode`:

| opcode | what it does |
|---|---|
| `PLACE` | give an unplaced variable a first qubit |
| `ROUTE` | realise a logical edge between two placed chains |
| `REWRITE_ONE` | add one adjacent free qubit to a chain, or remove one from it |
| `REWRITE_GROUP` | replace several chains at once |
| `REPAIR_GROUP` | restore archived chains |
| `RESTART` | abandon the workspace and begin again |
| `COMMIT` | return an archived embedding as the answer |
| `STOP` | end with nothing |

**Growth and shrinkage share the `REWRITE_ONE` opcode** (`src/isingfold/rl/proposal.py:415` grows,
`:492` shrinks). An opcode-level diagnostic therefore cannot separate them, which is how an
earlier reading of the action mass nearly went wrong.

### Support: how many actions are offered

The generator fills quotas per family. Two registrations exist (`probes/_context.py:36,80`):

| registration | place | route | grow | shrink | rewrite | repair | restart | total state-changing |
|---|---|---|---|---|---|---|---|---|
| registered | 32 | 16 | 12 | 4 | | | | 64 |
| **wide** | 288 | 128 | 48 | 16 | 16 | 12 | 4 | **512** |

The wide registration carries its own context version suffix so a run cannot silently mix the two.

**The wide support is required and this is measured, not assumed.** An unhinted witness replay,
which is the order a policy actually meets, reaches a valid commit on 0 of 6 instances at fill
0.50 and 1 of 4 at fill 0.90 under the registered support, and on 6 of 6 and 4 of 4 under the wide
one. A hinted replay succeeds under both and cannot settle it.

---

## 4. The observation

Each action is described by one row. Four schemas, selected by `--features`
(`probes/constructor_curriculum.py:542-549`):

| schema | channels | contents |
|---|---|---|
| `tiny` | 16 | 8 opcode indicators, 5 successor deltas, 3 terminal indicators |
| `local` | 20 | tiny plus 4 local-capacity channels |
| `physics` | 32 | local plus 6 before and 6 after structural channels |
| `construction` | 230 | a full description |

**The tiny row** (`probes/constructor_tiny_gate.py:200-219`) is a one-hot over the opcode, the
change the action would make to five state summaries, and three terminal indicators. The five
summaries are the fraction of variables placed, the fraction of logical edges realised, the
fraction of host qubits used, membership beyond that, and the fraction of chains disconnected
(`:116-120`). So the row describes **the action's effect on global counters**, not its geometry.

**The four local channels** (`:156-161`) are free host neighbours of the affected chains before
and after, unplaced logical neighbours of the affected variables, and the free neighbours those
neighbours can still reach. These are the only spatial information in the 20-channel schema.

**The twelve physics channels** (`probes/constructor_physics_features.py:38-50`) are six
quantities computed before and after the action: weighted realised contacts, weighted contact
redundancy, coefficient load concentration, a bridge-load bottleneck proxy, **cycle redundancy**,
and signed realised coupling. They are pooled over the edited chains and their logical
neighbourhood with fixed public weights.

Cycle redundancy is there because it is the measured predictor of whether a chain holds: at a
fixed chain length, internal redundancy predicts a chain's break rate at -0.439 [-0.589, -0.289]
over 496 chains, better than any other structural quantity measured.

---

## 5. The policy

Three classes, one interface (`probes/constructor_curriculum.py:525-539`):

| kind | structure | parameters |
|---|---|---|
| `linear` | `Linear(in_dim, 1, bias=False)`, **zero-initialised** | 16 to 32 |
| `mlp` | `Linear(in_dim, w)` then `SiLU` then `Linear(w, 1, bias=False)`, last layer zero | a few thousand |
| `contextual` | a candidate-set actor-critic with Deep Sets pooling and a state-value head | more |

Every headline result uses `linear`, which is **sixteen to thirty-two weights**. Zero
initialisation is deliberate: the starting policy is uniform over the legal actions, so a run
begins with no preference to unlearn.

`--expand-features` zero-pads a narrower checkpoint into a wider schema, so added channels begin
with no influence on any action logit. The weights are preserved exactly; the logits agree to
float32 rounding, because a dot product of length 32 does not accumulate in the same order as one
of length 20.

---

## 6. The learning rule

**Monte Carlo REINFORCE with a leave-one-out baseline**, no critic in the headline configuration,
no PPO, no Q-learning (`probes/constructor_learning.py:108-124`).

For `K` episodes drawn from one unchanged instance and starting state:

```
A_i = U_i - mean_{j != i} U_j
L   = -(c/K) * sum_i A_i * sum_t log pi(a_it | row_it)
```

The actor terms are summed along a trajectory and averaged over **all** episodes, including empty
ones. `c` is a fixed positive scalar declared before the batch, not a per-batch normalisation.

**This is the architecture's central limitation and it is measured.** Every decision in a
trajectory receives the same advantage, so a good final embedding never identifies the decision
that produced it. Over 518 branch points, the residual spread across four legal actions at a
single decision is 0.0491 [0.0467, 0.0515], against the 0.0268 that separates the policy from the
router, and the policy picks the better action 0.230 of the time against a chance rate of 0.250.

---

## 7. The reward

One scalar an episode, at the end.

**Feasibility**: one for a valid commit, zero otherwise.

**Quality** (`probes/constructor_objective.py:83-97`): `U = 1 - 0.5 * r / B`, where `r` is the
mean decoded excess energy normalised by the ground energy and `B` bounds it. A valid episode
scores between 0.5 and 1, an invalid one scores 0. There is **no resource term anywhere**: qubit
count and chain length are constraints through the host's capacity, never penalties.

The consequence is arithmetic. The residual gap the policy is being asked to close is worth about
0.008 on this scale and a failed episode is worth 0.99, so validity dominates the advantage
whenever training episodes fail.

**Conditional quality** (`--conditional-quality K`) draws until `K` valid episodes are in hand and
centres only those, which estimates the gradient of quality **given** validity rather than of the
mixed utility. That is a different objective and it is declared as one: it drops the term that
teaches validity, so coverage is reported at every evaluation, and the episodes drawn and rejected
appear in every iteration record.

---

## 8. The curriculum

Training episodes may start from a partial embedding; **evaluation always starts from empty**.
Three kinds of prefix (`probes/constructor_curriculum.py:401-440`):

| unit | what it hands over |
|---|---|
| `qubits` | a random subset of the witness's chains covering a fraction of its qubits |
| `variables` | a random fraction of the witness's variables |
| `trajectory` | **the state a successful forward walk actually occupied**, at that fraction along it |

The first two are valid partial embeddings the environment's own actions need never have
produced, and they teach nothing about the order a construction follows. The third is a real
state, so starting at the k-th from last asks the policy to finish a real construction with k
decisions left.

A mastery schedule lowers the prefix only after an iteration whose **assisted** episodes pass a
threshold, so failures of the empty-start mix do not hold the curriculum back. The step must match
the cell: at fill 0.90 an untrained policy finishes from 0.99 of a walk and fails from 0.97, so
the default step of 0.1, eleven decisions on a 110-state walk, moves far too fast.

---

## 9. Evaluation

Held-out instances never seen in training, and held-out instances deliberately carry **no
witness**, so a prefix cannot leak the answer.

- **Feasibility**: episodes from empty, the fraction reaching a valid commit.
- **Quality, deployment**: every arm proposes until a shared wall-clock deadline, selects among
  its proposals with 256-read blocks using a **public energy score that never reads the certified
  optimum**, and the selected embedding is assessed on a fresh 4096-read block. The router arm
  restarts under the same deadline.
- Sampling is classical simulated annealing under the registered schedule, beta range 0.1 to 2.0
  and 200 sweeps. Selection and assessment never share a block.

---

## 10. Where the architecture meets the measurements

Four design choices are directly implicated by results already on the record.

| choice | consequence, measured |
|---|---|
| per-action scoring, rows independent | an action cannot see the others, so "this move is better than that one here" is not directly expressible |
| leave-one-out advantage, one scalar an episode | every decision gets the same credit; a decision worth 0.0491 is invisible |
| the observation describes effects on global counters | a model fitted directly to measured branch orderings reaches 0.290 against 0.250 by chance |
| mixed validity-and-quality utility | a failed episode is worth 0.99 and the quality range is 0.5 |

The expressiveness audit rules out two alternatives. Capacity is not the constraint: the small
network overfits eight teacher paths and, given thirty-nine, gains 0.03 over the linear model. The
physics channels are not carrying the decision either: physics32 and local20 rank witness
decisions alike, 0.267 against 0.261.

So the open architectural question is the **action representation**: what a row would have to
contain for a good branch to be distinguishable from a bad one. No reward schedule substitutes
for it.
