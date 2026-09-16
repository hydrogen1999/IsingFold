# Critical review of the RL embedder, Claude Fable 5.1, 2026-09-15

Premise supplied by the author: the method is right and the objective is right, so if it does
not learn, something is wrong somewhere. This review takes that premise seriously and looks for
the something. It is written against the code and the logs, with paths, and it is paired with an
independent review by GPT-6 astra in the same directory.

## 1. What the evidence rules out, and what it does not

Ruled out with measurements that survive re-reading:

- Capacity. The scorer reaches the label oracle on states it has seen: regret -0.0000 and
  +0.0031, rank correlation +0.714 and +0.833 (STATUS.md, section on the ceiling).
- Loss. bce, consistent and joint transfer -0.014, -0.027, -0.028 (STATUS.md).
- Labels and teacher. Both were wrong, both were fixed (audits 2 and 4), the gap did not move.
- Representation. Four arms on the hard corpus and five arms with a family held out land within
  seed noise (results/hard_abl/, results/diverse_abl/).
- Label reliability. This one was never named but it has been measured all along: the "oracle"
  row in every table picks by the cached 256-read label and is re-measured on 512 independent
  reads, and it gains about +0.10 over random. A label block predicts a fresh block. The head
  matches that oracle on training states and gets +0.01 on new lineages. The gap is
  generalisation, not noise.
- The Hamiltonian is visible. probes/successor_scorer.py:249-270 feeds h per variable, J and |J|
  per logical edge, contacts, chain length, and the global scale and strength after autoscale.

Not ruled out, and this is where the premise can still be right:

- **Data scale.** The learning curve is flat from 25 to 200 lineages (results/curve/). That is
  strong evidence at those sizes and no evidence at all about 5,000 or 50,000. A physics
  surrogate that must learn how chain breaks interact with a frustrated landscape is exactly the
  kind of function that is flat at 400 states and rises at 40,000. Labels cost 256 sampler reads
  each, which is seconds; the slow part was environment overhead in probes/train_quality.py,
  and that is engineering, not physics.
- **The action space of the MDP.** This is the finding that I think answers the premise.

## 2. The MDP is built on the wrong action space

The environment (src/isingfold/rl/env.py, proposal.py) takes a minorminer embedding and improves
it by local moves: reroute a chain, grow it, shrink it, restart. The policy chooses among those.
Three measurements bound what any policy in that environment can be worth:

1. Policy best-of-K against random best-of-K in the same environment, matched episodes: +0.01,
   interval contains zero (STATUS.md, registered bar).
2. Environment best-of-K against minorminer best-of-K at matched measurements: contains zero.
3. Pool ceiling (results/modern/*/pool.log): eight independent minorminer draws beat eight grown
   variants, and beat four draws plus their four variants by -0.05 and -0.04. A variant of a
   draw is correlated with its parent and is worth less to the pool than a fresh draw.

Together: the value of an embedding lies in the tail of independent draws, and local edits of a
draw stay near their parent. A policy that chooses among local edits is choosing among options
that are each worth less than the option of drawing again. That holds for a perfect policy. It is
not a learning failure and cannot be trained away. It is the formulation.

The formulation in which a policy's decisions are not dominated by a restart is construction:
the policy builds the embedding, chain by chain, and its decisions determine whether a valid
embedding exists at all and how the host is filled. There, a decision is not a perturbation of a
heuristic's output, and diversity across episodes comes from the policy's own stochasticity.

## 3. The regime where construction has something to win

results/fill/: instances planted with a valid embedding at 80 to 95 percent of Pegasus 6 or
Zephyr 4, short chains. minorminer at its deployed budget finds an embedding on 0 to 33 percent
of them; with chains of one or two qubits, on none. The deterministic clique embedder is below
the random one (results/feasibility/busclique.log). Wall time at that regime is 10 to 50 seconds
per draw.

The advisor's scale is the fraction of the host the instance needs: ninety slightly hard,
ninety-five hard, a hundred infeasible. The planted corpus makes that exact when chains have
length one (results/fill/*_exact.log, running). No published embedder is measured there.

That is a claim an oral can carry: the first learned embedder that works where the standard tool
does not, on current hardware, at matched wall time, with exact supervision available for free
from the generator.

## 4. What a constructive embedder needs, concretely

- State: the host with occupancy, the logical graph, the partial embedding, the set of
  unplaced variables, native coordinates, and the space features already in
  probes/space_features.py (free volume, directional capacity). All of section 7 of the design.
- Action: place the next variable's chain root, or extend a chain by one qubit toward an
  unsatisfied coupling. Two heads: which variable next, which qubit.
- Reward: terminal validity, then a shaped term for couplings satisfied per qubit spent. Quality
  by measurement only at the end, and only among valid outputs, exactly as measured selection
  does now.
- Teacher: the planted witness gives a full trajectory per instance. Imitation first, then
  fine-tune with the terminal reward. The AlphaGo order the project was asked to follow.
- Curriculum: fill 0.70 to 0.95, short chains. Validity of minorminer along that axis is the
  difficulty label and it is already measured.
- Baseline at matched wall time: minorminer with tries chosen to match the policy's wall time,
  measured on the same instances, plus minorminer seeded with the policy's output.

## 5. Where a favourable number could still be an artefact

- Selection and assessment on the same reads. Guarded in pool_ceiling.py, witness_quality.py,
  train_quality.py; not guarded by construction anywhere, so every new probe must re-derive it.
- Maximum over repeats with the start measured once (budget_sweep.py, repeats=2). Withdrawn once.
- Pool size mismatch (pool_ceiling.py first version). Withdrawn once.
- The training-set evaluation block prints first and looks like a result
  (probes/train_successor.py:301). Twice misread this week by the monitor, never by a person,
  but the log format invites it.
- Cache key does not include the corpus content hash, only the path
  (probes/train_quality.py:232). Two corpora at the same path would share a cache silently.
- Any probe without the meta_path guard imports the old tree (fixed in c1a6b6b).

## 6. Ranked changes, with the experiment that kills each

1. **Constructive MDP on the fill-planted corpus, imitation from witnesses.** Prediction: after
   imitation alone, validity at 85 percent fill on held-out instances is above minorminer's at
   ten tries. Kills it: validity below minorminer's on the same instances at matched wall time.
2. **Data scale for the quality surrogate: 5,000 lineages on Pegasus 6.** Prediction: if the
   held-out gain rises above +0.03 the surrogate is real and worth a critic. Kills it: flat at
   +0.01, which closes prediction for good rather than for now.
3. **Witness quality at the threshold** (running). If valid embeddings at 90 percent fill have
   solve probability near zero for every embedding, item 1 is a feasibility paper and the
   objective thesis does not apply there; the two contributions would then be separate.
4. **Baseline at matched wall time** (running: tries 50 and 200). If minorminer at twenty times
   the budget reaches the fill regime, item 1 becomes a speed claim.
5. **Retire the improvement MDP as a method.** Keep it as the negative result that motivates
   construction: a policy over local edits is bounded by the pool-ceiling result.
