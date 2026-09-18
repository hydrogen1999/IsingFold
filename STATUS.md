# Status

A running record of what has been done, what each result is worth, and what is still open. Kept
because this project has withdrawn eleven published numbers, and a list of what currently
stands is the only way to tell a finding from a leftover. Every number cites its log under
`results/`; every experiment in flight is on the board below with its log path, so a check
never depends on memory.

## Board, 2026-09-15

| task | what it decides | host, log | state |
|---|---|---|---|
| Task 7 | old vs corrected encoding on identical labels, 3 seeds each, both hosts | `results/corrected/*_seed*.log` | **done, Checkpoint 2 failed**: corrected +0.003 Pegasus, +0.035 Zephyr; legacy +0.012, +0.041; no corrected interval above zero; the improvement surrogate is retired as a method |
| Task 8 | label reliability, learnable ceiling | `results/corrected/*_reliability.log` | **done**: ceiling +0.109 / +0.122 |
| Task 9 | quality endpoint at the fill regime under the registered schedule, paired with intervals. Decides feasibility-only or not | apollo `runs/fill/*_witness_registered.log` -> `results/fill/` | **done** both hosts: residual discriminates, -0.021 [-0.030, -0.010] Zephyr and -0.032 [-0.047, -0.017] Pegasus at 80 percent |
| Task 10 | anytime minorminer with a wall-time deadline, 30 to 300 s | `results/fill/*_anytime.log` | **done** both hosts: Zephyr at 300 s medium chains 1.00 / 0.83 / 0.33 / 0 at 80 / 85 / 90 / 95 percent, short chains 0; Pegasus only 80 percent medium reaches 1.00, every other cell 0 at every deadline. ADR-003's three gates are passed on both hosts |
| budget x20 | minorminer at 200 tries on the fill corpora | `results/fill/*_budget.log` | **done** both hosts: Pegasus 80 percent medium 0.33 -> 0.83 -> 1.00 at 10, 50, 200 tries, 85 percent 0 -> 0.17 -> 0.17, everything else 0 at 300 to 480 s a draw; Zephyr done: 85 percent medium chains 0.50 -> 0.67 -> 0.83 at 10, 50, 200 tries; 90 percent stays 0.33; short chains stay 0 at 260 to 350 s a draw |
| Task 12 | witness replay through the construction API | `results/fill/*_replay_{hint,nohint}.log` | **done**: with the witness as the generator's preference, a valid COMMIT on 48/48 instances per host, every cell, 400 to 620 decisions, 200 to 530 s; with the unhinted heuristic order, none. Imitation records collected; prioritiser training |
| direction 3 | adaptive allocation of reads vs uniform, real evaluator, disjoint assessment | `results/adaptive/*.log` | **done, replicated**: successive halving matches uniform at half the reads on both hosts, -0.002 [-0.008, +0.004] Zephyr and +0.002 [-0.004, +0.009] Pegasus; UCB slightly worse; random -0.11 to -0.12. Learned prior (`results/adaptive/pegasus6_prior.log`, direction 5's surrogate as the prior): prior alone at zero reads 0.388, +0.058 over random and -0.063 [-0.094, -0.035] under uniform; halving seeded by the prior's top half -0.020 [-0.038, -0.006] under plain halving. The prior carries signal and costs quality as a filter; measurement dominates it. Null as a learned component |
| curriculum baseline | anytime minorminer on Pegasus 3 and Zephyr 2 fill corpora, 12 a cell, deadlines 10 to 120 s | `results/small/*_anytime.log` | **done**: short chains from 80 percent 0 to 0.17 at 120 s with 20 to 40 attempts; medium chains 0.42 / 0.17 (Pegasus 3) and 0.75 / 0.08 (Zephyr 2) at 90 / 95 percent |
| hybrid | policy roots then minorminer; seeded-minorminer tolerance arms | apollo `runs/seeded/*.log`, `runs/hybrid/*.log` | witness roots turn 0 into 12/12 at 80 percent short chains on both small hosts and 2/2 so far on both big hosts; policy roots and tolerance running; big-host seeded tables done: witness roots 1.00 to 85 percent on Pegasus 6 and Zephyr 4, plain router 0 on short chains from 80 percent |
| hybrid v3, validity | fair protocol: both arms search until 60 s and select by measurement, full-host budget, curriculum hosts | apollo `runs/hybrid/v3_valid_*.log`, copies in `results/hybrid/` | 60 iterations: policy+search 0.50 vs router+search 0.47 at every checkpoint (Pegasus 3, init and scratch), 0.37 to 0.40 vs 0.40 (Zephyr 2); valid candidates equal in both arms at every checkpoint, so the layouts change no instance's completability. Null, closed |
| hybrid v3, quality | same, paired residual on a fresh block where both valid | apollo `runs/hybrid/v3_quality_*.log`, copies in `results/hybrid/` | iter 39: +0.0081 [-0.0129, +0.0277] over 14 (Pegasus 3), +0.0065 [-0.0064, +0.0176] over 12 (Zephyr 2), positive is worse; null, closed |
| selection rules | the same 8 router draws per instance chosen by: first draw, random, fewest qubits, shortest chain, measurement (256 reads), oracle (reads the assessment); assessed on a fresh 512-read block, registered schedule | apollo `runs/rules/pegasus6.log`, `zephyr4.log`; copies in `results/rules/` | p_solve minus first draw: fewest qubits +0.042 / +0.041 (intervals touch 0), measured +0.147 / +0.153, oracle +0.163 / +0.163 (Pegasus 6 / Zephyr 4, 30 instances each, every draw valid). Measurement takes 90 percent of the best-of-8 ceiling; the resource rule takes a quarter. Sweep: 64 reads a draw already take 91 to 93 percent of the oracle at K = 8; measured +0.105 / +0.120 at K = 4 and +0.169 / +0.161 at K = 16; fewest qubits stays at +0.04 at every K |
| layout v4 (PR #2, branch `7b465f3`, not merged) | all-free root support, capacity features, contextual actor-critic, witness-root warm start; arms legacy, support, capacity, warm, seed 0, Pegasus 3 and Zephyr 2 | apollo `~/prj_IsingFold_pr2/runs/hybrid_v4/{legacy,support,capacity,warm}_{pegasus3,zephyr2}_s0.log` (separate checkout of the PR) | launched and stopped 2026-09-16 at the user's request before the first training evaluation, with every other policy-then-router arm; 111 of its unit tests pass locally; not merged |
| independent constructor (PR #2 head `d06bf43`, ADR-005, not merged) | RL builds the whole embedding from empty, no router completion; contextual actor-critic over the environment's macro-actions, 230-channel observations; stage 1 feasibility curriculum, stage 2 quality from the stage-1 checkpoint; minorminer as a separate comparison arm only | apollo `~/prj_IsingFold_pr2/runs/constructor/feas_{pegasus3,zephyr2}_s{0,1,2}.log` | launched 2026-09-16; 203 of its unit tests pass locally; gate before any quality run: nonzero valid-COMMIT coverage on held-out lineages. Init and iteration 9, 30 held-out instances a run: policy valid 0.00 in all six runs (6 to 8 attempts in 60 s, none constructed), minorminer arm 0.37 to 0.60; training demand fraction 0.08 to 0.18, normalised entropy 1.0. Steps cost 0.5 to 0.8 s so a 30 s episode cannot finish a 20-variable instance; two runs with 120 s episodes added (`feas_long_*_s0.log`) to separate time from learnability. Iteration 19: still 0.00 valid in training and held-out; the actor's normalised entropy is 1.00000 at every iteration in every run (logit spread 0.007 across candidates at init, Adam at lr 3e-4 moves the logits by about 1e-2 in 19 updates, reward mean 0.04 with advantage std 0.02 to 0.03, no valid COMMIT in 320 training episodes to reward). Two runs at lr 3e-3 with 120 s episodes added (`feas_lr3_*_s0.log`), and two at lr 3e-3, 120 s, leave-one-out baseline, entropy 0 (`feas_loo_*_s0.log`, the clean ablation named in `docs/review/2026-09-17-constructor-learnability.md` of PR 2). Raw logs of every constructor run are in `results/constructor/` (runs of PR 2 code at `d06bf43`). The six 30 s-episode runs were stopped at iterations 30 to 40 after their fourth held-out evaluation (init, 9, 19, 29) read 0.00 with the actor still exactly uniform; their final logs are the committed ones. The six variants (120 s episodes, 240 s deadline; lr 3e-3; LOO with entropy 0) at their first trained evaluation: policy 0.00 held-out on both hosts with 14 to 24 attempts an instance, minorminer 0.53 / 0.43; the actor now moves (entropy 0.989 to 0.999 at lr 3e-3) and demand fraction is 0.15 to 0.26. Twenty variables from an empty start is still past the learnable regime; the ladder is the path. Stopped 2026-09-17 after that evaluation to free cores for the ladder; logs in `results/constructor/feas_l*.log` |
| constructor ladder, gate 2 (branch `feat/constructor-curriculum` off PR 2, `37d43a0`) | the tiny gate's 16-weight linear REINFORCE actor trained on a fixed set of 12 certified-embeddable small instances with dead ends and evaluated on 12 unseen non-isomorphic ones; stage a: 2 to 4 variables on cycles with pendant dead ends and a chord; stage b: 4 to 8 variables on grids with holes and dead ends; 3 seeds each, plus an MLP actor on stage b; no minorminer, witness or quality label after generation | apollo `~/prj_IsingFold_pr2/runs/curriculum/{a,b}_linear_s{0,1,2}.log`, `b_mlp_s0.log` | **stage a passes**: held-out valid rate 0.36 to 0.97 (+0.61 [+0.53, +0.69]) and 0.42 to 0.96 (+0.54 [+0.43, +0.65]) and 0.41 to 0.99 (+0.58 [+0.50, +0.65]) on seeds 0, 2 and 1, train 0.97 / 0.96 / 0.95; `results/curriculum/a_linear_s{0,1,2}.log`. **Stage b passes**: held-out 0.15 to 0.80 (+0.65), 0.14 to 0.74 (+0.60), 0.09 to 0.67 (+0.58) linear, 0.15 to 0.70 (+0.55) MLP, every interval above zero; `results/curriculum/b_*.log`. **Stages p and z pass**: Pegasus 2 fragments held-out 0.18 to 0.92 (+0.74), 0.22 to 0.79 (+0.58), 0.23 to 0.78 (+0.55); Zephyr 1 fragments 0.27 to 0.78 (+0.51), 0.33 to 0.90 (+0.58), 0.32 to 0.95 (+0.63); `results/curriculum/{p,z}_*.log` |
| constructor ladder, gate 3 (branch `feat/constructor-curriculum`, `559c7de`) | on the stage-b sets of gate 2: the PR's 230-channel contextual actor-critic with the value baseline, the same actor with leave-one-out, and a linear actor on the 230 channels, three seeds each, against the 16-weight linear actor's +0.58 to +0.65 held-out | apollo `~/prj_IsingFold_pr2/runs/curriculum/b_ctx_value_s*.log`, `b_ctx_loo_s*.log`, `b_lin230_s*.log` | **done**: linear on the 230 channels 0.97 / 0.85 / 0.85 held-out (+0.82 / +0.70 / +0.76) beats linear-16 on every seed; contextual actor-critic 0.87 / 0.68 / 0.00 (LOO) and 0.46 / 0.70 / 0.00 (critic), two collapses; features help; the contextual actor collapses at lr 0.03 and at lr 3e-3 matches linear-230 (held-out 0.94, 0.85 on seeds 0, 2); `results/curriculum/b_{lin230,ctx_loo,ctx_value}_*.log`, `b_ctx_loo_lr3e3_s{0,2}.log` |
| constructor ladder, size rung (branch `feat/constructor-curriculum`, `d765812`) | stages P and Z: 8 to 14 variables on 32 to 64 qubit fragments of Pegasus 3 and Zephyr 2, horizon 160 steps, 120 s episodes; 16-weight linear actor three seeds each, contextual actor-critic one seed each | apollo `~/prj_IsingFold_pr2/runs/curriculum/{P,Z}_linear_s*.log`, `{P,Z}_ctx_value_s0.log` | **passes on both hosts**: 16-channel linear held-out Pegasus 3 fragments 0.09 to 0.84 (+0.75 [+0.60, +0.86]) and 0.05 to 0.72 (+0.66 [+0.47, +0.83]), 0.08 to 0.63 (+0.55 [+0.35, +0.73]); Zephyr 2 fragments 0.10 to 0.84 (+0.73), 0.06 to 0.88 (+0.82), 0.07 to 0.77 (+0.70); contextual actor Zephyr 0.10 to 0.86 (+0.75); 230-channel linear at 0.90 / 0.76 / 0.69 (Pegasus) and 0.94 / 0.88 (Zephyr) held-out at iteration 100 to 150; `results/curriculum/{P,Z}_*.log` |
| constructor ladder, corpus-size rung (branch `feat/constructor-curriculum`, `a64092f`) | stages F and G: 12 to 20 variables on 64 to 128 qubit fragments of Pegasus 3 and Zephyr 2, the size of the small corpora where the old constructor read 0.00; horizon 240, 300 s episodes, 100 iterations, 6 episodes an instance, 10 evaluation episodes; 16-channel linear three seeds, 230-channel linear one seed, contextual at lr 3e-3 one seed, per host | apollo `~/prj_IsingFold_pr2/runs/curriculum/{F,G}_linear_s*.log`, `{F,G}_lin230_s0.log`, `{F,G}_ctx_lr3e3_s0.log` | **passes**: 16-channel linear seed 0 held-out 0.08 to 0.80 (+0.72 [+0.50, +0.90]) on Pegasus 3 fragments and 0.17 to 0.88 (+0.71 [+0.53, +0.85]) on Zephyr 2 fragments (`results/curriculum/{F,G}_linear_s0.log`); seeds 1 and 2 at 0.81 / 0.75 and 0.78 / 0.63 when stopped at iteration 60 to 100 to give the fill rung the cores; 230-channel linear Zephyr 0.17 to 0.94 (+0.78 [+0.62, +0.90]) and contextual at lr 3e-3 Zephyr 0.15 to 0.93 (+0.78 [+0.61, +0.92]) done; Pegasus 230-channel and contextual at 0.82 to 0.83 mid-run. The size where the old constructor read 0.00 through 40 iterations |
| constructor ladder, fill rung (branch `feat/constructor-curriculum`, `c1d29fb`) | the planted fill corpora themselves as a stage (`--stage corpus`, split by lineage, real coefficients, witness unreachable), with `--init` to carry a lower rung's checkpoint up; pilot at fill 80 on Pegasus 6 (428 variables on 680 qubits) to time an iteration | apollo `~/prj_IsingFold_pr2/runs/curriculum/pilot_fill80_pegasus6.log` | from-scratch controls on the registered 64-candidate support (`results/curriculum/fill{80,90}_{pegasus6,zephyr4}_scratch_s0.log`): 0/24 valid in training and 0.00 held-out through iteration 24 on both hosts, reward 0.013 to 0.036 (5 to 14 percent progress, the cold STOP hazard), 3 hours per 10 iterations; stopped 2026-09-17 after the unhinted replay showed that support cannot contain the witness at these fills, so the runs could not inform learning. The fill corpora hold 6 instances a cell, 8 cells a host, 263 to 533 variables. Step cost cut 2.5x first (`results/constructor/step_cost_pegasus6.txt`) |
| constructor ladder, quality rung (branch `feat/constructor-curriculum`, `15bf8f9`) | the measured-residual reward on the rungs already passed for feasibility, warm-started from their feasibility checkpoints; evaluation is the paper's protocol: proposals until a 20 s deadline, selection by a 256-read block, assessment on a fresh 512-read block, minorminer under the same deadline; stage b on the 230 channels (three seeds), stages p and z on the 16 channels (one seed) | apollo `~/prj_IsingFold_pr2/runs/curriculum/bq_lin230_s*.log`, goose `pq_linear_s0.log`, `zq_linear_s0.log` | first evaluations, held-out, paired residual policy minus minorminer (negative is better for the constructor), both arms 6 measured candidates: stage b seeds 0 / 1 / 2 at the feasibility checkpoint -0.077 [-0.143, -0.022] / -0.033 [-0.084, +0.008] / +0.000 [-0.013, +0.012]; after 100 quality iterations -0.066 [-0.134, -0.009] / -0.021 (it 99, both>=6) / -0.002; Pegasus 2 fragments -0.137 [-0.210, -0.067], Zephyr 1 fragments -0.084 [-0.156, -0.014] at the checkpoint. Reconciled against pool size: minorminer had 6.0 unique candidates an instance, the policy 4.9 to 6.0, and the paired difference is unchanged on instances where both had six. Shapes: the constructor spends 1 to 2 more qubits (chains of 2 to 3) where minorminer returns singletons. **Control**: minorminer plus 2 random growth qubits reproduces the whole gain on the hardware fragments (+0.000, -0.001), and leaves -0.019 [-0.029, -0.009] on stage b with the 230-channel actor; quality-reward training added nothing (finals -0.068 / -0.033 / +0.005 vs minorminer). Learned part: zero on p and z and for the 16-channel actor; with the 230-channel actor on b a small consistent component survives the control on three seeds (-0.018, -0.004, -0.010; two intervals below zero) with fewer qubits than the control; at the size rung with the 230-channel checkpoints it is -0.004 [-0.012, +0.004] on Zephyr and -0.006 [-0.013, -0.001] on Pegasus against the grown control (-0.064 and -0.033 against minorminer): small, borderline, consistent in sign across rungs, not a headline |
| quality reward at the size rung (branch `feat/constructor-curriculum`, `cfde791`) | 100 iterations of the measured-residual reward from the 0.96 held-out 230-channel checkpoints on Pegasus 3 and Zephyr 2 fragments; evaluation every 50 with minorminer and minorminer-plus-growth under a 30 s deadline; decides whether quality training moves the learned part past the -0.004 to -0.006 it shows at the checkpoint | apollo `~/prj_IsingFold_pr2/runs/curriculum/{P,Z}q_lin230_s1.log` | **final** (`results/curriculum/{P,Z}q_lin230_s1.log`): Zephyr 2 fragments, policy minus grown control -0.008 [-0.015, -0.003] at equal qubits (13.6 / 13.6), -0.069 [-0.126, -0.016] against minorminer, validity 1.00: the first learned quality component with its interval below zero at matched spend, after 100 iterations of the measured-residual reward (from -0.004 at the checkpoint). Pegasus 3 fragments: -0.004 [-0.011, +0.003] with validity 0.92 and fewer qubits (12.1 / 13.7). Small, one seed per host; bottleneck 2 has a foothold |
| fill rung, partial-start curriculum (branch `feat/constructor-curriculum`, `c8e2984`) | training episodes start from a random prefix of the train task's planted witness, scheduled from 90 percent of the variables pre-placed down to none over 100 iterations; held-out tasks never carry a witness and every evaluation starts from empty; fill 80 and 90 on Pegasus 6 and Zephyr 4, 16-channel actor | apollo `~/prj_IsingFold_pr2/runs/curriculum/fill{80,90}_{pegasus6,zephyr4}_prefix_s0.log` | relaunched with a lighter evaluation (2 episodes an instance, 600 s episodes) after the first launch spent hours in its initial evaluation at a third of a core each; the bet for bottleneck 1 |
| fill rung, partial-start curriculum warm-started (branch `feat/constructor-curriculum`) | the same prefix curriculum at fill 80 and 90 on both hosts, initialised from the corpus-size rung's checkpoints (`F_linear_s0.pt` for Pegasus, `G_linear_s0.pt` for Zephyr): the ladder's intended path, a policy that already avoids STOP and places and routes at 12 to 20 variables | apollo `~/prj_IsingFold_pr2/runs/curriculum/fill{80,90}_{pegasus6,zephyr4}_prefixinit_s0.log` | launched 2026-09-17 beside the cold prefix runs and the from-scratch controls; three arms per cell decide bottleneck 1 |
| fill rung, wide support (branch `feat/constructor-curriculum`, `2ef3676`) | the reviews' fixes together: 512-candidate support with every frontier placement, STOP/RESTART bias -6, occupancy prefixes shared per group with a mastery schedule and an empty-start mix, warm start from the corpus-size rung, horizon 2,000, 1,200 s episodes, the proven fast path; fill 80 seed 0 on apollo, fill 85 / 90 / 95 seed 0 and fill 80 / 90 seeds 1 and 2 on goose (one fourteen-core allocation) | apollo `~/prj_IsingFold_pr2/runs/curriculum/fill80_{pegasus6,zephyr4}_wide_s0.log`, goose `runs/curriculum/fill*_wide_s*.log` | relaunched 2026-09-17 with held-out-only evaluation and a 300 s training deadline (600 s for evaluation): the training-set evaluation cost as much as the held-out one and an evaluation pass at fill takes hours. The registered-support prefix runs and the from-scratch controls were stopped, the support being structurally unable to reach the witness at these fills |
| fill rung, short chains on the locked corpus (branch, `041acca`) | the regime the paper claims: short-chain cells only (alpha 3.0, 358 to 541 variables, where minorminer is 0 at every fill and the deployment teacher completes in about 510 decisions), on the 12-instance corpora with locked splits, train and held-out from the manifest's own lists, horizon 3,000, 400 s training episodes, 900 s evaluation, wide support, STOP bias, occupancy prefixes under the mastery schedule; fills 80, 85, 90, 95 on both hosts, two seeds at 80 and 90 | goose `~/prj_IsingFold_pr2/runs/curriculum/sfill{80,85,90,95}_{pegasus6,zephyr4}_s*.log` (one twelve-core allocation) | launched 2026-09-18. The earlier wide runs mixed in the long-chain cells, whose witness needs more than 2,000 decisions, so half their instances could not finish inside the horizon; they were stopped |
| local-capacity features (branch `feat/constructor-curriculum`, `095d940`) | both reviews name the same missing information: two placements that realise the same edge but leave different room have identical rows in the 16 channels. `--features local` adds four: free qubits within two hops of the affected chains before and after, their change, and the room per unplaced neighbour still to serve, all bounded so the cost is local. Ladder rerun on stage b and both fragment rungs, two seeds each, against the 16-channel numbers | apollo `~/prj_IsingFold_pr2/runs/curriculum/{b,F,G}_local_s{0,1}.log` | launched 2026-09-18; if it holds at the rungs it replaces the 16 channels at fill, where leaving room is the decision |
| quality reward on the modern corpora (branch `feat/constructor-curriculum`, `deee7ba`) | the experiment both reviews rank first for bottleneck 2: 100 iterations of the measured-residual reward on the regime where quality moves (16 to 20 variables on the full hosts, p_solve 0.4 to 0.6), wide support, 16 train and 12 held-out lineages, evaluated against minorminer and the growth control under a 60 s deadline; 16-channel and 230-channel actors, both hosts | apollo `~/prj_IsingFold_pr2/runs/curriculum/modq_train_{pegasus6,zephyr4}_lin{16,230}.log` | launched 2026-09-18; the fragment-trained constructor lost here by +0.008 to +0.044, so this decides whether training in the regime fixes it |
| behaviour cloning of the witness (branch, `2a839ad`) | now that the wide support contains the witness's path: a set-valued teacher over every witness-consistent offered candidate, train lineages only, 30 epochs, evaluation from empty on unseen instances; fill 80 on both hosts | apollo `~/prj_IsingFold_pr2/runs/clone/fill80_{pegasus6,zephyr4}_clone_s0.log` first teacher pass (`results/clone/*_v1.log`, the deployment configuration: no seeded chain, full-host cap, satisfied growth on) found two more things. The short-chain cells complete in 444 to 519 decisions, but the long-chain cells (alpha 2.0, chains of about 2.2) hit the 2,000-step horizon at progress 0.96 to 0.999, so the horizon and not the support is now the binding constraint there; and one instance stuck at step 0, because from empty there is no frontier and the first PLACE offered a single variable a coarse spread of roots. Fixed (`041acca`: several high-degree variables, spreads at different offsets, union covering the free host) and relaunched at a 6,000-step horizon |
| cross-topology transfer | each host's corpus-size checkpoint evaluated on both hosts' held-out fragment sets, evaluation only, 12 instances and 5 episodes each | apollo `~/prj_IsingFold_pr2/runs/transfer/{pegasus,zephyr}_on_{pegasus,zephyr}.log` **done, and the transfer is complete**: on the Pegasus 3 fragment held-out set the Pegasus-trained actor reaches 0.82 and the Zephyr-trained one also 0.82; on the Zephyr 2 fragment set both reach 0.90 (12 instances, 5 episodes each, `results/transfer/*.log`). The per-instance vectors differ slightly (the two actors' 16 weights differ by 0.1 to 0.4) but the topology the actor was trained on does not change its held-out rate. What the ladder learns is a topology-independent construction rule |
| deployment coverage (branch `feat/constructor-curriculum`, `82d80d7`) | what a practitioner gets rather than a per-episode rate: every arm proposes until a shared wall-clock deadline and the first valid embedding wins; minorminer restarts under the same deadline at 20 tries a draw; 12 held-out lineages of the modern corpora, deadlines 60 and 300 s, both hosts | apollo `~/prj_IsingFold_pr2/runs/deploy/modern_{pegasus6,zephyr4}_d{60,300}.log` **done on the modern corpora** (`results/deploy/modern_*.log`, 12 held-out lineages, 16 to 20 variables on the full hosts): coverage at a 60 s deadline, policy 0.42 (Pegasus 6) and 0.75 (Zephyr 4) against minorminer 1.00; at 300 s, 0.92 and 1.00 against 1.00. The router answers in 0.1 s, the constructor in 35 to 90 s. In the regime where minorminer works, it wins on coverage and by three orders of magnitude on time; the constructor catches up on coverage only with five times the deadline. The paper must report this table, and the learned method's case has to be made where the router returns nothing |
| the objective's scale curve | `probes/objective_scale.py`: the witness's p_solve and energy residual, and minorminer's, on planted instances from 16 to 440 variables at fill 80, registered schedule, 512 reads | apollo `~/prj_IsingFold_pr2/runs/scale/objective_{pegasus,zephyr}.log` **done** (`results/scale/objective_*.log`): witness p_solve 0.022 at 50 variables (Pegasus 3), 0.006 and 0.003 at 50 and 75 (Zephyr 2), exactly 0 from 250 on, on both hosts; the residual stays informative at 0.10 to 0.15 throughout. So p_solve is the objective up to about 100 variables and the residual above it, which is what the record does |
| datasets for the locked test split and the hardware-sized hosts | fill corpora regenerated at 12 instances a cell on Pegasus 6 and Zephyr 4 (`runs/fill_v2`, seed 20260917) so lineages can be split train / validation / test; ink-drop corpora on Pegasus 16 and Zephyr 15 (24, 48, 100 variables, seven families, `runs/inkdrop`) with the cap-scaled context fix (`2bb23a6`) | goose `~/prj_IsingFold/runs/fill_v2/*_generate.log`, `runs/inkdrop/*_generate.log` | ink-drop corpora **done**: 96 instances each on Pegasus 16 and Zephyr 15 (24 / 48 / 100 variables, seven families), minorminer embeds all of them at the 400-qubit cap (mean 102 and 90 qubits, longest chain 3.2 and 2.7), train / validation / test 67 / 14 / 15 by lineage, `goose:~/prj_IsingFold/runs/inkdrop/{pegasus16,zephyr15}/corpus`; fill_v2 on Zephyr 4 **done**: 96 instances, 12 a cell, train / validation / test 67 / 14 / 15 by lineage (`goose:~/prj_IsingFold/runs/fill_v2/zephyr4`); its generation log repeats the regime (minorminer at 20 tries: 1.00 / 0.50 / 0.08 / 0 on medium chains at 80 / 85 / 90 / 95 percent, 0 on short chains everywhere); Pegasus 6 fill_v2 **done** too (96 instances, 67 / 14 / 15); bottleneck 4's data is in place on both axes |
| hardware-sized rung (branch `feat/constructor-curriculum`, `7d1c989`) | 24-variable ink-drop instances on the full Pegasus 16 (5,640 qubits) and Zephyr 15 (7,440 qubits), train from the corpus' locked train list (16 tasks, witness prefix curriculum 90 percent down to none), held-out from its validation list (4 tasks, no witness), the test list untouched; 16-channel actor, horizon 400, 300 s episodes, 100 iterations | goose `~/prj_IsingFold_pr2/runs/curriculum/ink24_{pegasus16,zephyr15}_prefix_s0.log` | relaunched 2026-09-17 warm-started from the corpus-size checkpoints (`ink24_*_prefixinit_s0.log`), through Slurm as a two-core allocation (`scripts/goose/submit_curriculum_batch.sh`; the account allows two jobs in the queue, the partition had 247 idle CPUs): goose caps each ssh login session at one CPU (`cpu.max 100000 100000` on the session scope), so the first launch shared one core between two jobs and spent three hours in its initial evaluation. **Held-out validity at hardware size, from empty** (`results/hardware/ink24_*.log`, 4 validation-list tasks, 5 episodes each): Pegasus 16 (5,640 qubits) 0.10 at the corpus-size checkpoint, 0.25 at iteration 24, **0.85 at iteration 49** with assistance down to 0.36; Zephyr 15 (7,440 qubits) 0.35, 0.60, **0.90** with assistance down to 0.45. Evaluation from empty, minorminer forbidden, test list untouched |
| step cost at hardware size | `probes/constructor_step_cost.py` on Pegasus 16 fragments of 1024 and 2048 qubits and the full 5,640-qubit host, 24 variables | apollo `~/prj_IsingFold_pr2/runs/curriculum/step_cost_p16.log` | **measured** (`results/constructor/step_cost_pegasus16.txt`): on the full 5,640-qubit Pegasus 16 with 24 variables the environment step is 0.076 s and the 16 tiny channels 0.011 s (0.087 s a step, so a 100-variable instance is about 35 s an episode: trainable); the 230 construction channels are 0.74 s a step there (0.14 at 1024, 0.30 at 2048), too slow to train at hardware size until they are local. The hour-long first attempt was the quadratic fragment shaping on a stale copy of the probe, not the environment. Bottleneck 3 is closed for the actor that carries the feasibility ladder and open for the 230-channel actor |
| contact policy, low fill | contact growth vs random growth vs the start vs four fresh router draws, all with measured selection | goose `runs/contact/pegasus6.log`, `zephyr4.log` | Pegasus 6: restart minus start +0.141; random growth -0.075, policy -0.104 below it. Zephyr 4: restart +0.139; random -0.060, policy -0.063 below it. Both hosts: growing one draw loses to drawing again |
| contact policy, high fill | same, starting from the planted witness at 80 to 95 percent occupancy, budget = the space left, objective = energy residual, frontier features, occupancy traced | goose `runs/contact/*_fill.log`; Zephyr init copy `results/contact/zephyr4_fill_init.log` | Zephyr 4, init, 16 instances at 89 percent occupancy, residual (higher is better): random growth +0.0066 [+0.0047, +0.0085] over the start, policy +0.0077, policy minus random +0.0012 [-0.0017, +0.0046]; router redraws -0.0052 [-0.0103, -0.0010] under the start, growth +0.0117 over the redraws. The regime flips: at high fill growth beats redrawing and the start, and the learned part is still zero. Pegasus 6 (`results/contact/pegasus6_fill_init.log`) agrees: random growth +0.0061 [+0.0042, +0.0082], policy +0.0057 over the start, policy minus random -0.0005 [-0.0022, +0.0013]; router redraws +0.0003 [-0.0005, +0.0010], i.e. the router never redrew at 90 percent and that arm is the start. After 25 training iterations (`results/contact/*_fill_iter24.log`): policy minus random -0.0022 [-0.0042, -0.0003] Pegasus 6, +0.0007 [-0.0016, +0.0033] Zephyr 4; growth over the start unchanged at +0.004 to +0.007. At iteration 50: -0.0005 [-0.0024, +0.0014] and +0.0007 [-0.0021, +0.0036]. **Closed, null for the learned part**: three validations, the policy never differs from random; growth plus measurement is the gain (+0.006 over the start, both hosts). Stopped 2026-09-17 at iteration 50 of 100; final logs `results/contact/{pegasus6,zephyr4}_fill.log` |
| direction 4, RL | REINFORCE on the constructive policy (policy = prioritiser), dense progress reward, STOP and RESTART masked, evaluation samples until the deadline; from scratch and initialised from the imitation prioritiser; Pegasus 3 and Zephyr 2 | `results/rl/pegasus3_scratch.log`, `pegasus3_init.log`, `zephyr2_init.log` | **closed, null**: valid 0.00 on 30 held-out instances at every checkpoint through iterations 119 to 129 (demand fraction 0.51 to 0.64); stopped 2026-09-16 |
| anytime 600 s | minorminer restarted until 300 and 600 s on the big fill corpora | `results/fill/*_anytime600.log` | **done**: at 600 s nothing changes but Pegasus 85 percent medium chains 0 -> 0.17; short chains stay 0 in every cell on both hosts |
| imitation, deployed | the imitation prioritiser as a policy on the full fill corpora, sample until 600 s | `results/rl/pegasus6_imitation_deploy.log`, `zephyr4_imitation_deploy.log`; baseline `results/fill/*_anytime600.log` | **closed, null**: valid 0.00 in every cell, 48 instances per host, demand fraction 0.44 / 0.39 |
| placement completion | imitation PLACE-only roots (greedy and witness-hinted) completed by our construction search, 600 s an instance, big fill corpora | `results/fill/pegasus6_placement.log`, `zephyr4_placement.log` (33 of 48 instances each when stopped) | **closed, null**: no instance completed on either host, greedy or hinted, at demand fractions 0.90 to 0.99; stopped 2026-09-16 with the other policy-then-completion arms |
| direction 5 | quality surrogate at ten times the lineages, registered schedule, corrected compiler | `results/large/train_seed*.log` | **done**: held-out +0.021, +0.027, +0.023 over 360 lineages, every interval above zero, against +0.003 at 160 lineages; the ceiling is +0.11. Learnable, slowly |

Checkpoints and gates are in `docs/plans/2026-09-15-plan.md`; decisions in `docs/decisions/`;
the two reviews in `docs/review/`. When a row above finishes, its log is copied to `results/`
and its number is written into the section of this file that it belongs to, in the same commit.

## Where it stands in one paragraph

The objective is downstream solution quality under a declared hardware and measurement budget.
Fill means actual state occupancy; the witness-fill label alone does not make an empty-start
episode a tight-space completion test. Contact growth from a valid witness tests allocation of
the remaining space, with the witness supplied equally to every arm. It does not demonstrate
that the policy can construct that starting embedding. The large corrected scorer has a positive
validation gain over random (+0.0205, +0.0273, +0.0230), while learned hybrid roots have not shown
a completion advantage in the committed results. Plain successive halving is a useful selection
baseline, not a learned result. Registered growth sweeps show that indiscriminate growth can harm
quality; the contact-versus-start intervals still include zero. New contact-policy and hybrid
training jobs are not results until their completed logs and independent assessment are committed.

Implementation corrections and the verification scope are recorded in
`docs/review/2026-09-16-quality-contract-fixes.md`. They do not retroactively change historical
numbers or establish a new training gain. Checkpoint selection uses validation; an unopened final
test is still required for a paper result.

## Historical claims and their standing at the time

This table records earlier configurations, including the auto-schedule experiments. Its broad
negative statements are not conclusions about the current model family. In particular, the
large corrected scorer supersedes "does not transfer" as a universal claim. A noisy measured
selection reference is not a theoretical upper bound on what a learned selector can achieve.

| Claim | Number | Standing |
|---|---|---|
| Objective selection beats resource selection | +0.0714 at N=2 to +0.2564 at N=40 | holds |
| Resource selection is flat in the number of candidates | 0.483 to 0.510 over a twentyfold increase | holds |
| A structural scorer does not beat resource selection | -0.022 to -0.031 | holds |
| Ceiling for choosing among candidates at one state | +0.1123 [+0.0832, +0.1440] | holds |
| The network fits that choice exactly on seen states | regret -0.0000 and +0.0031 | holds |
| It does not transfer | -0.029, -0.020, -0.014 | holds |
| Reading the compiled program instead of the state helps | +0.0093, +0.0109, +0.0157 | holds, three seeds, intervals contain zero |
| Qubit count predicts quality among sampled candidates | -0.002, 0.0% of variance on the hard corpus | holds, but answers the wrong question |
| A larger budget buys a better best embedding | -0.0241 easy, +0.0141 [-0.0064, +0.0340] hard | untested: the candidates span 5 qubits, not a budget range |
| Registered bar, RL best-of-8 against random best-of-8 | +0.0157, +0.0262, +0.0135 | holds: does not pass |
| The improvement operator degrades a selected embedding | -0.057 at one round, -0.085 at two | holds |
| Capacity, teacher, loss or labels explain the failure | | ruled out, each separately |

## Work done

**Measurement repairs, from five external audits.** The comparator that measured a policy against
a different embedding than the one it was given. A `.gitignore` line that silently excluded the
five logs the README cited. The teacher that gated on the block which made a rollout the winner
rather than on an independent re-measurement. Pinning the starting embedding, which also pinned
what RESTART could reach, so training met an action deployment would not give it. COMMIT resolved
as workspace plus new_chains when the environment returns the archive entry it names, wrong on
60 of 178 COMMIT rows and on none of the 60 at a root state. A within-lineage baseline that
included the episode's own return and shrank the gradient by (n-1)/n. The in-training curve
reading validation and test pooled. Best-of-n counting successes rather than attempts. Wall clock
counted twice on one arm and at zero on another. Ranks that broke ties by position. Fresh-read
seeds drawn from Python's salted string hash, the trap this repository had written a hand-test
for the same week.

**Environment repairs.** RESTART was unreachable after a single rewrite because restores that
rebuild the same assignment held every slot; identical successors are collapsed and the escape
family keeps a reserved share. `compile_program` and the evaluator sort every label before
writing a coefficient, because a frozenset of qubit labels iterates in per-process hash order and
one embedding measured in two processes returned two utilities with the seed pinned.
`Context.beta_range` registers the annealing schedule, because left to itself the sampler derives
beta from the programmed coefficients and cancels exactly the energy compression the theory is
about: shrinking every coefficient to a sixteenth changes utility by +0.0000 with auto beta and
by -0.4195 with a registered range.

**Experiments.** The registered comparison on all ninety test lineages. The cost-utility frontier
against minorminer with selection and assessment separated. Supervised quality with capacity,
early-stopping and loss arms. A representation ablation between reading the state and reading the
compiled program. A measurement of whether a resource-quality trade-off exists at all.

**Corpora.** The ink-drop corpus is 4.48 independent components per instance, largest holding 11
of a nominal 16, three spins coupled to nothing, chains of one or two qubits, minorminer done in
two milliseconds. `gen_hard_corpus.py` starts from a dense logical graph instead, plants a
frustrated-loop Ising for a certified ground energy and asks minorminer whether it embeds within
the cap. Clique-16, digest `e5382da2349ebafb`: 45.3 edges, 0.42 isolated spins, 1.42 components,
43.2 qubits, longest chain 4.36. Then, because 600 draws of one shape is not a benchmark, seven
families across three sizes filled round-robin, digest `b1167177611d43a4`: 630 instances, 30.6
edges, headroom 0.4098, and the family recorded in each lineage so a split can hold out whole
structures rather than only other coefficient draws.

## Running now

On apollo: eight representation arms on the clique-16 corpus, F0 against Fpos (native hardware
coordinates), Fphys (the energy margin of each chain's cheapest cut) and Fall, two seeds each,
one shared label cache. And a learning curve, fitting on 25, 50, 100 and 200 lineages with the
held-out set fixed and the training subsets nested, two seeds each.

On goose: label collection for the diverse corpus, 220 training and 70 held-out lineages across
seven structural families.

The learning curve is the one that matters. Capacity, the loss, the labels, the teacher and the
representation have each been ruled out or bounded, one at a time. The quantity nobody has varied
is how much data the thing is fitted on: 389 states from 200 lineages is about three thousand
labelled candidates for a function over program graphs. If the curve is still climbing at 200,
the bottleneck is data and the other effects are noise around it.

## Open, in the order the evidence points

1. Does reading the compiled program transfer better on the hard corpus, where chains are long
   enough for chain integrity to bind? The eight arms answer this.
2. Do the hardware coordinates or the chain-robustness margin add anything on top?
3. Does any of it transfer across structures rather than across coefficient draws? The diverse
   corpus makes this askable for the first time; nothing in this repository has answered it.
   Sections 7.2 to 7.4 of the meeting design are implemented and unrun: residual reachability
   after occupancy and faults, free volume at three radii, and first-edge directional capacity.
4. Only then: whether a learned scorer can stand in for evaluator budget at deployment, which is
   the claim that would be worth a paper.

## Spending qubits: the one place a positive result appeared and did not survive

Growing a valid embedding's chains to lengths drawn from a truncated power law gives several
embeddings of one problem, nested in resource, all realising the same couplings. Three ways to
spend the extra qubits: lengthen a chain, close cycles inside it, or grow toward the chains of
coupled variables so a logical coupling is carried by more contacts.

Which topology this runs on decides whether two of the three exist at all. On Chimera 4, mean
degree 5.5, growing for redundancy moves the bridge fraction from 1.000 to 0.975, which is
nothing. On Pegasus 6 it reaches 0.806 and on Zephyr 4 it reaches 0.731, and contacts per logical
edge rise by 55 and 76 percent rather than by a tenth. The same clique-16 costs 21 to 23 qubits
there against 41 on Chimera. Every result in this repository before this point was measured on
Chimera, which is neither the current hardware nor a typical one.

At two repeats per arm, contact growth read +0.0996 on Pegasus and +0.0636 on Zephyr against the
starting embedding, with intervals excluding zero. That was the maximum over two noisy blocks
while the start was measured once. At one repeat, on ninety lineages and a fresh seed, the same
arms read -0.0010 [-0.0265, +0.0258] and -0.0081 [-0.0379, +0.0222]. The effect was the
inflation, and the inflation is about +0.10, which is larger than any real effect measured in
this project.

What holds is the ordering. At the same cost, contact-seeking growth beats redundancy-seeking by
+0.073 on Pegasus and +0.094 on Zephyr, and beats lengthening by +0.012 and +0.048. How a qubit is
spent matters a great deal; spending more of them does not improve quality on any topology tested.
The claim that unused hardware is an opportunity is not supported by this test.

## The learning curve is flat, and the list of suspects is empty

Held-out gain of the successor scorer against random selection, two seeds per point, one cache:

    lineages fitted     25        50        100       200
    held-out gain    +0.0235   +0.0088   +0.0096   +0.0199   (three earlier seeds at 200: +0.0120)

Eight times the data changes nothing. Every point sits in [+0.009, +0.024] with intervals that
contain zero. Data joins capacity, loss, labels, teacher and representation as a ruled-out
cause. On the hard corpus the four representation arms read +0.0074, -0.0045, +0.0071 and
+0.0018 held-out, all within seed noise. Logs in `results/curve/` and `results/hard_abl/`.

## Learn to propose, not to predict: the pool ceiling

If quality cannot be predicted, a learned embedder can only help by proposing a better set of
candidates for measurement to choose from. `probes/pool_ceiling.py` measures the ceiling of
four pools of eight measured candidates each, Pegasus 6 and Zephyr 4, ninety lineages, the
chosen candidate re-measured on independent reads:

    pool                                      Pegasus 6                 Zephyr 4
    eight minorminer draws                    0.8232                    0.8500
    eight draws grown toward coupled chains   -0.0086 [-0.026, +0.010]  -0.0084 [-0.027, +0.011]
    four draws and the grown copy of each     -0.0635 [-0.091, -0.038]  -0.0404 [-0.066, -0.015]
    eight of twenty-four, chosen for distance +0.0040 [-0.026, +0.035]  +0.0140 [-0.015, +0.043]

Eight independent draws is the best pool measured. The mixed pool is the informative row: its
candidates are no worse one by one, but a grown copy is correlated with its parent, so the pool
has four independent seeds where the others have eight. Pool diversity is worth more than any
way of spending qubits, and choosing draws for structural distance does not add to it.

The first run of this probe built the mixed pool from the best half of each other pool and
read +0.011 and +0.021 with intervals excluding zero. That was a maximum over sixteen measured
candidates against eight. It is the eleventh withdrawn number and it was withdrawn before it
left the log. All three versions are in `results/modern/*/pool*.log`.

## Where the learned embedder stands

Against minorminer with measured selection at matched budget, no learned component tested here
has a lever: not predicting quality at any data scale, not spending qubits in any of three
ways, not proposing for diversity. The finding that stands is the one every measurement
agrees on: quality is not a function of structure the model can see, it is a function the
sampler has to be asked, and asking scales. That finding is the paper's claim. A learned
embedder is not, on this evidence.

Two regimes are untested and are where a learned embedder could still be measured to win:
instances near the embeddability threshold, where minorminer's draws fail often and validity
rate is the score; and allocation of measurement budget across candidates, which is a bandit
over the pool rather than a change to it.

## Representation, third corpus, family held out

Five arms on the diverse corpus with the modular family held out entirely, one seed, 44
lineages: F0 +0.0433, Fpos +0.0457, Fphys +0.0544, Fspace +0.0295, Fall +0.0137, every
interval about 0.09 wide and every pair within 0.02. No arm separates from the bare program.
That is three corpora and three splits on which the representation does not matter. Logs in
`results/diverse_abl/`.

## Historical construction diagnostic: witness fill does not determine difficulty

`probes/gen_fill_corpus.py` plants a chain partition of the host at a chosen fill and takes the
quotient as the logical graph, so a valid embedding at that fill is known. With chains of one
or two qubits (alpha 3.0) minorminer at ten tries finds an embedding on none of the instances
from eighty percent up, on either host. With longer chains (alpha 1.5 and 1.0) it finds one on
all of them up to ninety-five percent, and its own embedding fills sixty to eighty-seven percent
of the host: a long-chain witness is a wasteful embedding, and the tool compresses it.

So the fraction of the host an embedding uses says nothing about the instance. What does is the
fraction the instance cannot do without, its minimal fill, and the advisor's scale (ninety
slightly hard, ninety-five hard, a hundred infeasible) has to be read on that quantity. Chains of
length exactly one make it exact: the logical graph is then an induced subgraph of the host on
n = f·|H| nodes, every embedding needs at least n qubits, and the witness uses exactly n. That
sweep is running. Logs in `results/fill/`.

## Relaunch, 2026-09-15: two reviewers, two defects, a spec before code

Two independent critical reviews (`docs/review/`) were run against the premise that the method
and the objective are right, so a failure to learn is a defect. GPT-6 astra found two:

- **D1.** The successor scorer's input was compiled with chain strength ratio times mean|J|
  (`probes/train_successor.py:compile_for`); the environment that produced every label uses
  ratio times the RMS coefficient scale (`src/isingfold/rl/program.py:strength_registry`). On
  the audit fixtures the shown strength was 8.6, 18.5, 1.0 and 43.9 percent above the evaluated
  one; on the unit fixture 12.6 percent. The model fitted labels of a program it never saw.
  Fixed in `cf62297`, pinned by `tests/unit/test_probe_compile_matches_env.py`. ADR-001.
- **D2.** `Context.beta_range` defaulted to None and no labelling or assessment probe set it, so
  every quality label was measured under the auto schedule that cancels energy compression,
  which the repository had already shown (`results/audit/`). The registered range (0.1, 2.0)
  is now the default in `probes/_context.py`, so every earlier quality number is a number about
  a different objective. Fixed in `4ed5771`, pinned by `tests/unit/test_registered_schedule.py`.
  ADR-002.

The reviews also narrow what the earlier results rule out. The pool-ceiling result constrains
one growth distribution, not every local policy. The flat learning curve covers lexicographic
subsets of one cache with a mismatched encoding. Label reliability was already measured in
audit 3: within-state correlation 0.979 to 0.998 across two 512-read blocks. So the record
supports "the tested models did not transfer", not "nothing is learnable". Spec, three ADRs
and the task plan are in `docs/specs/`, `docs/decisions/`, `docs/plans/`.

The test suite was testing the wrong tree on the remote hosts (an editable install of the old
workspace shadowed PYTHONPATH). `tests/conftest.py` pins it to this repository; baseline on
apollo is 1234 passed, 16 failed, 5 collection errors, every failure and error in tests that
depend on LAC_B or on runtime assets outside this tree. Noticed, not touched.

Fill regime, further facts: with chains of length one, where minimal fill is exact, minorminer
finds nothing from 70 percent up on either host at 60 to 155 seconds a draw
(`results/fill/*_exact.log`). Pegasus 6 at fifty tries: 80 percent with medium chains rises
from 0.33 to 0.83 and 85 percent from 0 to 0.17; everything else stays at zero.

## The anytime baseline names the regime (Task 10, Zephyr 4)

minorminer restarted with fresh seeds until a wall-time deadline, every attempt counted, six
instances a cell (`results/fill/zephyr4_anytime.log`):

    cell                 <=30 s   <=60 s   <=120 s   <=300 s   attempts
    80 percent, medium    0.83     0.83     1.00      1.00       1.2
    85 percent, medium    0.33     0.33     0.67      0.83       1.3
    90 percent, medium    0.00     0.00     0.17      0.33       2.5
    95 percent, medium    0.00     0.00     0.00      0.00       3.0
    short chains, any     0.00     0.00     0.00      0.00       2 to 5

At five minutes the standard tool is at zero on every short-chain fill and at a third or
less from 90 percent with medium chains. Pegasus 6 is harder (`results/fill/pegasus6_anytime.log`):
only 80 percent with medium chains reaches 1.00 at 300 s, and every other cell is at zero at
every deadline. That is the regime a learned constructor is measured
in, at the same deadline. With the quality endpoint (Task 9) and the encoding comparison
(Task 7) done, the three gates of ADR-003 are passed on Zephyr; Pegasus is four instances
from done.

## The imitation prioritiser (Task 12b)

Trained on the hinted-replay records of one host, held out by instance
(`results/imitation/*_train.log`): teacher agreement top-1 0.333 on Pegasus 6 and 0.314 on
Zephyr 4 among up to 64 candidates a decision, the teacher's pick at median rank 2, mean
rank 5.4. As the generator's preference on the other host with the witness-consistency
check still applied, the replay stalls within a few steps
(`results/fill/*_replay_scored_by_*.log`): that measures whether the model's shortlist
contains the exact witness root at every step, which at a third per step it cannot. The
deployment question is different, whether the policy sampled until a deadline reaches any
valid embedding, and that is measured by the RL evaluation protocol: running on the
curriculum hosts with the prioritiser as initialisation, and on the full fill corpora at
300 s (`runs/rl/*_imitation_deploy.log`).

## Search inside the environment does not reach a valid embedding (negative)

Limited discrepancy search over the construction environment, 60 s a instance on the
curriculum hosts, learned prioritiser and the generator's own order
(`results/search/*.log`): zero valid embeddings in every cell, one to three passes within
the deadline, demand fraction 0.55 to 0.74. A pass costs about twenty seconds at 144 qubits
whatever the quotas, because proposal generation and validation of every candidate is the
cost, not the model; stubbing the observation saves a third. The prioritiser at argmax
orders candidates worse than the generator's own order. Backtracking search has to run
outside this environment, on plain chain dictionaries.

Our own completion search, run that way (`probes/route_search.py`, `results/ours/`): a
chronological-backtracking router fails from witness roots at corpus scale; a negotiated-
congestion router with per-demand rip-up, randomised restarts and chain-connectivity
invariants reaches 9 of 24 hard small-host instances from witness roots in 13 s where
minorminer's router reaches 24 of 24 in under a second. A maximum matching of bridge demands to free qubits before negotiation makes it worse, 0 of
24: the matching takes exactly the qubits the remaining demands need and, being fixed, cannot
be ripped up. Matching the standard router is an
engineering project on its own; until it does, the learned layout is measured with the
standard router as the completion and the standard router alone as the baseline.

## Review of `7044446` (external, 2026-09-17) and what changed because of it

The review found the hybrid evaluator unfair to the baseline (the policy arm searched and
selected by measurement, the router arm stopped at its first valid embedding), the qubit
budget absent from the completion, the quality reward losing its signal where the router
alone fails and able to rank a valid embedding below an invalid one, and the actor blind to
the coefficients. All four are fixed in the commit that carries this note: both arms restart
until the deadline, keep up to six valid candidates, select by a measurement block and are
assessed on a fresh block, the paired residual with an interval; `attempt()` refuses an
embedding beyond the instance budget for either arm; a valid completion scores at least 0.6
plus a clipped quality term against the planted witness's residual, invalid at most 0.5; the
candidate features carry field and coupling magnitudes. The review's ordering is adopted:
the learned allocation prior (directions 3 plus 5) against plain halving on frozen pools is
the first learned-win test, hybrid validity then quality is the embedder's line, and
constructive RL alone is parked.

## The qubit-spend sweep under the registered schedule: how a qubit is spent decides the sign

Same corpora, lineages, exponents and seed as the unbiased confirmation, one draw per arm,
now measured under the objective (`results/modern/*/sweep_registered.log`), utility of the
grown embedding minus the start, 90 lineages:

    spend             alpha   Pegasus 6                     Zephyr 4
    lengthen          2.2     -0.057 [-0.070, -0.045]       -0.052 [-0.068, -0.036]
    lengthen          1.6     -0.092 [-0.117, -0.069]       -0.110 [-0.130, -0.090]
    redundancy        3.0     -0.024 [-0.039, -0.010]       -0.010 [-0.024, +0.004]
    redundancy        1.6     -0.118 [-0.143, -0.093]       -0.117 [-0.147, -0.089]
    contacts          3.0     +0.016 [-0.002, +0.036]       -0.000 [-0.014, +0.014]
    contacts          2.2     -0.012 [-0.037, +0.012]       +0.006 [-0.021, +0.033]
    contacts          1.6     -0.025 [-0.052, +0.001]       +0.011 [-0.024, +0.044]

Under this registered schedule, several lengthening and redundancy arms reduce measured
quality. Contact-growth effects versus the start have intervals containing zero on both hosts;
these experiments neither establish a positive gain nor prove that contact growth cannot hurt.
Each growth arm starts independently from the same base, so the rows are not a nested trajectory
and do not isolate qubit count from geometry. The result motivates learning where to spend rather
than assuming any extra qubit helps. The supplied start is a training/diagnostic reference;
end-to-end deployment starts empty and must construct its own valid output.

## Merged: `fix/quality-contracts-20260916` (audit of `7c5d225`)

An implementation audit of the five probes the relaunch depends on, merged fast-forward as
`426f55c` with 76 passing contract tests: router time counts against the deadline and late
results do not count; both arms consume one declared deployment budget; the qubit cap is a
declared constraint and a witness-derived cap must be requested by name; a witness is never a
deployment output; failed measurements are counted under a declared convention; the policy
gradient uses a leave-one-out baseline over independent episodes; selection and assessment
use separate sampler streams; a learned prior's checkpoint is checked for architecture and
training-lineage provenance. The audit's evidence boundaries stand: registered contact-growth
intervals include zero; the large scorer is validation evidence against random, not a gain
over halving; the selection reference is finite-read. Every run in flight was restarted on
the merged code; numbers from before the merge are history.

## The constructor ladder: the simple actor generalises at small scale (gate 2, stage a)

Gate 1 was the other agent's tiny gate (K3 into C5, one instance, 16-weight linear
REINFORCE, 31 to 93 percent, three seeds; reproduced here). Gate 2
(`probes/constructor_curriculum.py` on branch `feat/constructor-curriculum`) trains the same
actor on a fixed set of 12 certified-embeddable instances and evaluates on 12 unseen,
non-isomorphic ones, 20 stochastic episodes per instance, feasibility only, minorminer
forbidden after generation, no witness or quality label reachable. Stage a: 2 to 4 logical
variables on cycles with pendant dead ends and a chord (`results/curriculum/a_linear_s*.log`):

    seed    train before -> after    held-out before -> after    held-out gain, paired over 12
    0       0.42 -> 0.97             0.36 -> 0.97                +0.61 [+0.53, +0.69]
    2       0.38 -> 0.96             0.42 -> 0.96                +0.54 [+0.43, +0.65]
    1       0.35 -> 0.95             0.41 -> 0.99                +0.58 [+0.50, +0.65]

The held-out rate matches the training rate, so what is learned is a rule about placing,
growing and committing, not the instances. This is the first learned component in the
project with a positive held-out number; it is feasibility, not annealing quality.

Stage b, 4 to 8 variables on 3x3 to 4x4 grids with up to two holes and one to three dead
ends (`results/curriculum/b_*.log`, 200 iterations, same protocol):

    actor, seed        train before -> after    held-out before -> after    held-out gain
    linear, 0          0.07 -> 0.72             0.15 -> 0.80                +0.65 [+0.51, +0.75]
    linear, 1          0.11 -> 0.71             0.14 -> 0.74                +0.60 [+0.48, +0.70]
    linear, 2          0.11 -> 0.58             0.09 -> 0.67                +0.58 [+0.40, +0.74]
    mlp 32, 0          0.07 -> 0.52             0.15 -> 0.70                +0.55 [+0.40, +0.68]

Harder, slower (an hour to an hour and a half a run against fifteen minutes), still rising
at iteration 200, and again held-out at or above train. The small MLP is sharper (entropy
0.46 against 0.85) and not better: capacity is not what is missing at this rung.

Stages p and z, the same variable counts on connected 12 to 24 qubit fragments of Pegasus 2
and Zephyr 1, cut by a breadth-first ball and random connectivity-preserving removals so
they carry the hardware's own degree pattern and dead ends (`results/curriculum/p_*.log`,
`z_*.log`):

    host, seed          train before -> after    held-out before -> after    held-out gain
    Pegasus 2, 1        0.17 -> 0.85             0.18 -> 0.92                +0.74 [+0.67, +0.82]
    Pegasus 2, 2        0.29 -> 0.95             0.22 -> 0.79                +0.58 [+0.51, +0.65]
    Pegasus 2, 0        0.16 -> 0.80             0.23 -> 0.78                +0.55 [+0.43, +0.67]
    Zephyr 1, 0         0.22 -> 0.88             0.27 -> 0.78                +0.51 [+0.38, +0.63]
    Zephyr 1, 1         0.34 -> 0.89             0.33 -> 0.90                +0.58 [+0.50, +0.67]
    Zephyr 1, 2         0.33 -> 0.97             0.32 -> 0.95                +0.63 [+0.53, +0.74]

The rule the 16-weight actor learns transfers to the target topologies at this scale.

Size rung, 8 to 14 variables on 32 to 64 qubit fragments of Pegasus 3 and Zephyr 2
(`results/curriculum/{P,Z}_linear_s*.log`, 200 iterations, horizon 160, 120 s episodes):

    host, seed          train before -> after    held-out before -> after    held-out gain
    Pegasus 3, 1        0.08 -> 0.83             0.09 -> 0.84                +0.75 [+0.60, +0.86]
    Pegasus 3, 2        0.11 -> 0.86             0.05 -> 0.72                +0.66 [+0.47, +0.83]
    Pegasus 3, 0        0.04 -> 0.78             0.08 -> 0.63                +0.55 [+0.35, +0.73]
    Zephyr 2, 0         0.07 -> 0.88             0.10 -> 0.84                +0.73 [+0.61, +0.84]
    Zephyr 2, 1         0.12 -> 0.88             0.06 -> 0.88                +0.82 [+0.68, +0.93]
    Zephyr 2, 2         0.08 -> 0.92             0.07 -> 0.77                +0.70 [+0.53, +0.85]
    Zephyr 2, ctx, 0    0.07 -> 0.98             0.10 -> 0.86                +0.75 [+0.59, +0.90]

Corpus-size rung, 12 to 20 variables on 64 to 128 qubit fragments of Pegasus 3 and Zephyr 2
(`results/curriculum/{F,G}_linear_s0.log`, 100 iterations, horizon 240, 300 s episodes, the
size where the previous constructor read 0.00 through 40 iterations):

    host, seed, actor              train before -> after    held-out before -> after    held-out gain
    Pegasus 3, 0, 16 channels      0.04 -> 0.83             0.08 -> 0.80                +0.72 [+0.50, +0.90]
    Zephyr 2, 0, 16 channels       0.06 -> 0.74             0.17 -> 0.88                +0.71 [+0.53, +0.85]
    Zephyr 2, 0, 230 channels      0.06 -> 0.83             0.17 -> 0.94                +0.78 [+0.62, +0.90]
    Zephyr 2, 0, contextual 3e-3   0.06 -> 0.83             0.15 -> 0.93                +0.78 [+0.61, +0.92]

Every interval above zero (`{F,G}_*_s0.log`); the 230-channel linear actor sits higher again
at the size rung too: Zephyr 2 fragments, seed 1, held-out 0.06 -> 0.96 (+0.90 [+0.81, +0.97], `Z_lin230_s1.log`) and
Pegasus 3 fragments, seed 1, 0.09 -> 0.96 (+0.87 [+0.80, +0.94], `P_lin230_s1.log`); seeds
0 and 2 of the 230-channel runs were stopped at iteration 120 to 180 to give the fill rung
the cores (held-out 0.70 / 0.75 Pegasus, 0.93 / 0.88 Zephyr when stopped).

Quality at this rung with the 230-channel checkpoint (`Zq_grown2_s1.log`, Zephyr 2
fragments, 12 held-out, 30 s deadline, six measured candidates an arm): valid 1.00 on every
arm; residual policy 0.027, minorminer 0.091, minorminer plus two random growth qubits
0.031; paired policy minus minorminer -0.064 [-0.121, -0.012], policy minus grown -0.004
[-0.012, +0.004]. Pegasus 3 fragments (`Pq_grown2_s1.log`): residual policy 0.024,
minorminer 0.057, grown 0.031; policy minus minorminer -0.033 [-0.080, +0.001], policy
minus grown -0.006 [-0.013, -0.001], at 14.1 against 13.7 qubits and equal longest chains.
So at this rung the learned part beyond spend-and-measure is -0.004 to -0.006 residual,
one host's interval just below zero: the same small, borderline component as on stage b
(-0.004 to -0.018), present but not a headline. After 100 iterations of the quality reward
from those checkpoints (`{P,Z}q_lin230_s1.log`) the Zephyr component grows to -0.008
[-0.015, -0.003] at equal qubits with validity 1.00, the first interval below zero at matched
spend; Pegasus stays at -0.004 [-0.011, +0.003] and loses validity (0.92). Reward
signal-to-noise was 11 at this rung, so this is what the reward can teach on a linear actor
over these channels: real, small. The quality story stays with the platform until it
reappears at the modern-corpus scale, where the fragment-trained constructor currently
loses (next section).

Gate 3, the same stage-b sets, 200 iterations, lr 0.03 (`results/curriculum/b_lin230_*.log`,
`b_ctx_loo_*.log`, `b_ctx_value_*.log`), held-out after training by seed 0 / 1 / 2:

    actor                                   held-out after         held-out gain
    linear, 16 tiny channels (gate 2)       0.80 / 0.74 / 0.67     +0.65 / +0.60 / +0.58
    linear, 230 construction channels       0.97 / 0.85 / 0.85     +0.82 / +0.70 / +0.76
    contextual actor-critic, LOO baseline   0.87 / 0.68 / 0.00     +0.72 / +0.54 / -0.09
    contextual actor-critic, value baseline 0.46 / 0.70 / 0.00     +0.31 / +0.55 / -0.09

The PR's 230 channels are worth +0.10 to +0.17 held-out over the 16 tiny channels on every
seed with the same linear actor: the features carry information the small summaries do
not. The contextual actor-critic at this learning rate is unstable: two of six runs collapse
to 0.00 on train and held-out (entropy at 0.1 by iteration 50) and the others finish below
the linear actor on the same channels; the value baseline is not better than leave-one-out.
At lr 3e-3 the contextual actor with leave-one-out does not collapse and matches the linear
actor on the same channels: held-out 0.15 -> 0.94 (+0.79 [+0.70, +0.86]) on seed 0 and
0.09 -> 0.85 (+0.76 [+0.59, +0.90]) on seed 2, the seed that collapsed at lr 0.03
(`results/curriculum/b_ctx_loo_lr3e3_s{0,2}.log`). The collapse was the learning rate
inherited from the 16-weight gate, not the architecture. Two configurations stand at this
rung: linear on the 230 channels at lr 0.03, contextual on the 230 channels at lr 3e-3.
Next rungs in order: size (running: 8 to 14 variables on 32 to 64 qubit fragments of
Pegasus 3 and Zephyr 2, with both the 16-channel and the 230-channel linear actors), then
the full small hosts, then wall-clock, then quality.

## Quality on the modern corpora: the fragment-trained constructor over-spends and loses

The reviews' B2: the quality protocol on the regime where embedding quality moves the
objective (`runs/modern/{pegasus6,zephyr4}/corpus`, 16 to 20 variables on the full hosts,
p_solve 0.4 to 0.6, where best-of-8 selection gains +0.15). Evaluation only, 12 held-out
lineages, 60 s deadline, six measured candidates an arm, the corpus-size checkpoints
(`results/curriculum/modq_*.log`, residual, paired, positive is worse for the constructor):

    host, actor              valid pol/mm/grown   residual pol/mm/grown   pol minus mm              pol minus grown           qubits pol/mm/grown
    Pegasus 6, 16 channels   0.92/1.00/1.00       0.084/0.079/0.065       +0.017 [-0.018, +0.049]   +0.026 [-0.003, +0.056]   42.0/22.6/24.5
    Pegasus 6, 230 channels  0.67/1.00/1.00       0.113/0.079/0.070       +0.036 [-0.009, +0.077]   +0.044 [+0.004, +0.083]   40.9/22.6/24.8
    Zephyr 4, 16 channels    1.00/1.00/1.00       0.052/0.043/0.038       +0.008 [+0.004, +0.012]   +0.014 [+0.004, +0.027]   40.7/20.4/22.6
    Zephyr 4, 230 channels   0.75/1.00/1.00       0.084/0.043/0.042       +0.035 [+0.017, +0.055]   +0.035 [+0.014, +0.057]   33.4/20.4/22.6

The constructor trained on fragments builds chains of about two on every variable (40 qubits
against minorminer's 20 to 23) and is worse than both controls here, on Zephyr with the
interval above zero; the 230-channel actor also loses validity within the 60 s deadline on
the full hosts. What the small rungs taught (spend a qubit, realise an edge) is the wrong
rule at 16 to 20 variables on a 600-qubit host, where the router's near-singleton draw plus
two grown qubits is the best of the three. This is the table the paper needs either way; it
says the learned quality component is negative until the constructor is trained on this
regime with the quality reward (the experiment that follows the feasibility fixes).

## Where the objective stops discriminating: the regime map

A reviewer will ask why the record reports solve probability on the modern corpora and the
energy residual at fill. `probes/objective_scale.py` measures both for the planted witness
and for minorminer's draw, at fill 80, on a fresh 512-read block under the registered
schedule (`results/scale/objective_*.log`):

    host family    variables    witness p_solve    witness residual    minorminer p_solve    minorminer valid
    Pegasus 3      50           0.0215             0.126               0.0143                3 / 6
    Pegasus 6      300          0.0000             0.146               0.0000                4 / 4
    Pegasus 6      325          0.0000             0.144               0.0000                1 / 2
    Zephyr 2       50           0.0059             0.127               0.0000                1 / 1
    Zephyr 2       75           0.0027             0.098               0.0000                5 / 5
    Zephyr 4       250 to 300   0.0000             0.128 to 0.150      0.0000                5 / 6

Solve probability is already at 0.02 and below at 50 to 75 variables of frustrated loops and
scores zero hits in a 512-read block from 250 on. Zero hits bounds the per-read rate below
about 0.006 at 95 percent confidence; it does not establish that the rate is zero, and the
earlier wording here said it did. What the bound supports is that 512 reads cannot separate
embeddings at that size, not that no embedding ever solves. The energy residual
does, and it is the objective the record uses above 100 variables (Task 9 measured the
witness against minorminer's draw with paired intervals at fill 80 and found -0.021 to
-0.032). The paper therefore reports p_solve on the modern corpora (16 to 20 variables,
p_solve 0.4 to 0.6, where the selection result lives) and residual on the fill corpora, and
this table is why. It also bounds the fill claim: at 300 variables the benchmark measures
feasibility and residual, not solve probability.

## The audit through the deployment configuration, and what it cost

The second GPT-6 astra review was right that the witness replay was not the rollout: it
seeded the highest-degree variable's whole witness chain, capped qubits by the witness and
left satisfied growth off. `--deployment` removes all three, so the audit runs the
configuration a policy actually faces (`results/diag/replay_*_deploy.log`, short-chain
cells, three instances a cell):

    cell                  valid    decisions    seconds    place candidates a step
    Pegasus 6, fill 90    3 / 3    579 to 589   1342 to 3018    205 to 215
    Zephyr 4, fill 90     3 / 3    500 to 502   1028 to 1662    204 to 223
    Pegasus 6, fill 80    stuck at 1004        4656            112
    Zephyr 4, fill 80     stuck at 1136        5383             85

At fill 90 the deployment support contains the witness's path on every instance. At fill 80
the walk places every variable and then sticks: with a fifth of the host free, satisfied
growth fills the candidate batch with off-witness options and the witness-consistent route
is crowded out. That is a property of the teacher, not of the policy, which may use any
valid embedding; but it means the cloning teacher is reliable at 90 and not at 80.

The audit also exposed the real step cost: 2.3 to 4.6 s a step at those sizes, not the 0.16
measured on small hosts, because two parts of a decision were quadratic in (candidates times
chains). The legality dry run re-validated all five hundred chains for each of five hundred
candidates, and each candidate's payload key re-hashed the whole state. Both are now
incremental with equality tests (`tests/unit/test_incremental_identity.py`), and a wide step
at fill 90 from a 90 percent prefix costs 0.64 s against 1.36 s before. What remains is the
feature pass over the candidate batch (1.7 s of 6.4 s over ten steps) and the state
fingerprint, both linear in the batch.

## Deployment coverage where minorminer works: the router wins, and by a lot

The number a practitioner gets, not a per-episode rate: each arm proposes until a shared
wall-clock deadline and the first valid embedding wins; minorminer restarts at 20 tries a
draw under the same deadline. Modern corpora, 12 held-out lineages, 16 to 20 variables on
the full Pegasus 6 and Zephyr 4 (`results/deploy/modern_*.log`):

    deadline    host         policy coverage    minorminer coverage    policy seconds to first    minorminer seconds
    60 s        Pegasus 6    0.42                1.00                   36                         0.1
    60 s        Zephyr 4     0.75                1.00                   36                         0.1
    300 s       Pegasus 6    0.92                1.00                   88                         0.1
    300 s       Zephyr 4     1.00                1.00                   49                         0.1

On instances the standard router embeds in a tenth of a second, a learned constructor that
takes a minute is not a contribution, and this table says so plainly. It also sets the only
place the learned method can win: the congested regime, where the router at 20 to 200 tries
and 300 to 600 s returns nothing at all. That is the experiment running now. Reporting this
table is what makes the fill result, if it comes, believable rather than selective.

## The learned rule is topology-independent (cross-topology transfer)

Each corpus-size checkpoint evaluated on both hosts' held-out fragment sets, evaluation only,
12 unseen instances and 5 episodes each, from empty (`results/transfer/*.log`):

    held-out set                  Pegasus-trained actor    Zephyr-trained actor
    Pegasus 3 fragments, 12 to 20 vars    0.82                     0.82
    Zephyr 2 fragments, 12 to 20 vars     0.90                     0.90

The two actors are not the same function (their sixteen weights differ by 0.1 to 0.4 and the
per-instance rates differ on two instances of twelve), but which topology trained them does
not change the held-out rate at all. Pegasus and Zephyr differ in degree (15 against 20) and
in the shape of their unit cells, so this says the rule the ladder learns is about the
construction process, not about a hardware geometry: place where the frontier is tightest,
realise edges, avoid STOP, commit when complete. It also explains the earlier transfer to
the full Pegasus 16 and Zephyr 15 from fragment training.

## The hardware-sized rung: held-out validity from empty on Pegasus 16 and Zephyr 15

24-variable ink-drop instances on the full Pegasus 16 (5,640 qubits) and Zephyr 15 (7,440
qubits), the corpora with locked splits; 16 training tasks from the train list with the
occupancy prefix curriculum (a random prefix of the planted witness, scheduled down),
4 held-out tasks from the validation list, never the test list, evaluation from empty with
no witness reachable; the 16-channel linear actor warm-started from the corpus-size rung
(`goose:~/prj_IsingFold_pr2/runs/curriculum/ink24_{pegasus16,zephyr15}_prefixinit_s0.log`):

    host          held-out: checkpoint / iter 24 / iter 49    training validity (prefix)
    Pegasus 16    0.10  /  0.25  /  0.85                     20 / 24 at prefix 0.36 (iter 59)
    Zephyr 15     0.35  /  0.60  /  0.90                     23 / 24 at prefix 0.45

The corpus-size checkpoint already transfers to the full hosts (0.10 and 0.35 from empty on
hosts 40 times larger than its training fragments, because the 16 channels are host-size
independent), and the prefix curriculum takes Pegasus 16 from 0.10 to 0.85 held-out in 50
iterations and Zephyr 15 from 0.35 to 0.90, while the assistance falls from 0.9 to 0.36 and
0.45 of the witness (`results/hardware/`).
A 24-variable instance on 5,640 qubits is small in variables and large in host, so this is
the size axis, not the congestion axis; the fill axis (263 to 533 variables at 80 to 95
percent) is the one that decides the paper and is running on the wide support.

## Two critical reviews and the diagnostics they asked for (2026-09-17)

The two reviews (`docs/review/2026-09-17-claude-critical-review.md`,
`docs/review/2026-09-17-codex-gpt6-astra-review-bottlenecks.md`) converge on the same first
item for bottleneck 1: measure whether the environment's 64-candidate support can contain the
witness at fill before reading any training run. It cannot. The unhinted witness replay
(`probes/witness_replay.py`, now reporting the offer) at fill 80 on Zephyr 4 blocks at step
14 to 17 with 46 to 48 frontier variables and PLACE offered for 8 to 14 of them; at every
quota mix (place 32 / 48 / 56 of 64) the walk sticks within 0 to 89 steps on both hosts.
The shortlist offers eight variables with a few roots each (`_place`, `budget // 8`), so at
scale the action that continues the witness is simply not in the batch. The learned rungs
succeed because a 20-variable frontier fits in 64. Fix on the branch (`feat/constructor-curriculum`):
a wide construction registration (512 candidates, its own context version, every frontier
variable with up to twelve adjacent roots) and work caps sized by the horizon, since the
registered per-step feature charge exhausted the 200k-per-32-decision budget after a few
hundred decisions at 400 variables.

**The wide support contains the witness's path.** The same unhinted replay under the wide
registration (`results/diag/replay_*_wide.log`, short-chain cells, three instances each):

    cell                     valid    decisions    place candidates offered a step
    Pegasus 6, fill 80       3 / 3    509 to 523   227 to 236
    Pegasus 6, fill 90       2 / 2    578 to 584   229 (third instance running)
    Zephyr 4, fill 80        3 / 3    441 to 444   236 to 249
    Zephyr 4, fill 90        3 / 3    495 to 498   228 to 245

Under the 64-candidate registration the same walk blocked within 20 to 90 steps on every
instance; under the wide one it reaches a valid COMMIT on every instance at 80 and 90
percent fill, in 440 to 584 decisions (about 1.2 decisions a variable, inside the training
horizon). On the fast path a wide step costs 0.19 s from empty and 0.30 s at 81 percent
occupancy (434 to 497 candidates), so a 500-decision construction is two to three minutes.
Bottleneck 1 is therefore a learning problem with a reachable target, which it was not
before today; the training runs on the wide support are the first that can be read.

Support diagnostic (`results/diag/support_fill80_*.log`, fill 80 Pegasus 6, two instances,
registered 64-candidate support): the warm-started 16-channel actor puts 0.00 to 0.02 of its
mass on STOP and 0.77 to 0.82 on PLACE from empty (a zero actor: 0.02 on STOP, 0.33 to 0.42
on REWRITE, STOP within 10 to 108 steps as the reviews predicted); from empty it runs to the
400 s deadline at 0.44 progress; from a 90 percent witness prefix it reaches 0.989 progress
in 106 steps and then samples STOP. Steps cost 0.7 to 1.4 s from empty and 2.9 to 3.4 s
from the 90 percent prefix at 400 chains, not the 0.1 to 0.5 s assumed: a 1,100-decision
construction is 15 to 60 minutes an episode, which makes the step cost at hundreds of placed
chains the binding constraint for training at fill (profiling now).

Step cost at 400 placed chains, after the profile (`results/constructor/step_cost_pegasus6.txt`
records the small-instance curve; this is the fill regime): 4.2 s a step at the start of the
day; the tiny summary recomputed over every chain for every candidate (44 percent) is now
incremental; chain connectivity and chain-pair contacts in the validator are memoised
(26 percent); legality dry runs no longer count demands they never read; complete-embedding
checks run only for successors without an empty chain; chain identity rows are memoised;
and the environment's timing fast path (`ISINGFOLD_FAST_INTERNAL_ASSERTS=1`, which skips
the defensive invariant re-checks, the integrity seal it never verifies and the payload
re-check) is proven to give identical episodes (same decisions, supports and final chains on
two seeds at fill 80). Result: 0.52 s a step checked, 0.16 s on the fast path, a 26x cut.
Every equality is a unit test on random states and walks (`tests/unit/test_validate_memo.py`,
`test_tiny_features_incremental.py`, `test_wide_support.py`).

Launched on the new code: fill 80 on both hosts on apollo and fill 85 / 90 / 95 plus more
seeds on goose (`fill*_wide_s*.log`), all with the wide support, the STOP and RESTART start
bias of -6, occupancy-based witness prefixes shared per leave-one-out group under a
mastery schedule with a quarter of the slots from empty, warm-started from the corpus-size
checkpoints, horizon 2,000 decisions and 1,200 s episodes, evaluation from empty every ten
iterations. The unhinted wide replay of the witness at fill 80 and 90 runs beside them.

Reward signal-to-noise (`results/diag/snr_*.log`, six valid embeddings of the policy per
instance measured six times each on 256 reads): the between-embedding standard deviation of
the true residual is 0.016 to 0.017 against a within-embedding measurement standard
deviation of 0.005 to 0.008, signal-to-noise 2.9 (stage b) and 11 (size rung) per single
measurement; 23 to 128 reads suffice for signal-to-noise one. The quality reward is not
noise-dominated at 256 reads; the quality null is not a measurement null.

## What blocks the ladder above 128 qubits: the step cost grows with the host

Datasets on the two axes the paper needs: EmbedBench has hardware-sized hosts (Pegasus 16
with 5,640 qubits and Zephyr 15 with 7,440, 24 to 100 variables), and the fill corpora on
Pegasus 6 and Zephyr 4 hold 263 to 334 logical variables at 80 to 95 percent occupancy. No
Pegasus 16 or Zephyr 15 corpus exists in this repo; `probes/gen_fill_corpus.py --host-size`
can plant one.

Seconds per constructor step against host size, 16 variables, uniform linear policy
(`results/constructor/step_cost_pegasus6.txt`):

    host qubits     tiny 16 channels    construction 230 channels    environment part
    32              0.013               0.020                        0.012
    128             0.062               0.112                        0.054
    512             0.109               0.317                        0.099

Both the environment (candidate generation) and the 230-channel observation scale with the
host even though the logical problem is fixed at 16 variables. Profiling the environment
part on the 512-qubit fragment: 2.7 of 4.5 s of an episode were the environment's tensor
observation, which the constructor never reads. Skipping it (branch commit `4d9a0c0`, flag
off only in the constructor rollout, a test proves the trajectory is identical) gives

    host qubits     tiny 16 channels    construction 230 channels    environment part
    128             0.035               0.077                        0.026
    512             0.045               0.249                        0.035

On the full Pegasus 16 (5,640 qubits, 24 variables) the environment step is 0.076 s and
the 16 tiny channels 0.011 s (`results/constructor/step_cost_pegasus16.txt`), so the cheap
actor trains at hardware size; the 230 construction channels cost 0.74 s a step there and
need to become local before that actor does. The environment part is now nearly flat in
the host size. The 230-channel observation's hot spot was networkx subgraph views (free-host components and the contact walk over every
host edge, per candidate); adjacency-list walks with an element-for-element equality test
(branch commit `c64373c`) bring it to 0.032 s a step at 128 qubits and 0.092 at 512, so the
whole step is 0.058 and 0.127 s against 0.112 and 0.317 at the start. Remaining growth with
the host is in the local layout features. The corpus-size rung was restarted on the fixed
environment; at these costs a 300-variable fill instance is one to two minutes an episode.

## Quality rung, first evaluations: the constructor's embeddings measure better than minorminer's under the same protocol, at small scale

Protocol (`probes/constructor_curriculum.py --objective quality`): each arm proposes until a
20 s deadline, up to six distinct candidates are measured on a 256-read selection block, the
best is assessed on a fresh 512-read block; the minorminer arm proposes with its own restarts
under the same deadline, reads and assessment; residual under the registered schedule,
paired over the 12 held-out instances, 95 percent bootstrap. Negative favours the
constructor. At the feasibility checkpoints, before any quality training:

    stage                       held-out valid pol / mm    paired residual, policy minus minorminer
    b, seed 0 (230 channels)    1.00 / 1.00                -0.077 [-0.143, -0.022]
    b, seed 1                   1.00 / 1.00                -0.033 [-0.084, +0.008]
    b, seed 2                   1.00 / 1.00                +0.000 [-0.013, +0.012]
    p, Pegasus 2 fragments      0.92 / 1.00                -0.137 [-0.210, -0.067]
    z, Zephyr 1 fragments       1.00 / 1.00                -0.084 [-0.156, -0.014]

After 100 iterations of quality reward on stage b: -0.066 [-0.134, -0.009], -0.021, -0.002
(seeds 0, 1, 2): the reward does not move it much; the difference is already in what the
feasibility-trained constructor builds. Reconciliation before believing it: the first suspect
was pool size (a stochastic proposer against a deterministic router), and it is not that:
minorminer produced 6.0 unique candidates an instance on every set, the policy 4.9 to 6.0,
and restricting the pairing to instances where both arms had six leaves the numbers as they
are (-0.118 on p and z, -0.077 / -0.043 / +0.000 on b). The remaining suspect is shape: the
constructor may build longer chains, which under the fixed strength rule can lower the
residual on these small problems. The shapes (`*_shape_s0.log`, held-out, chosen embeddings):

    stage      arm          valid   qubits   longest chain   residual   corr(extra qubits, residual gain)
    b, s0      policy       1.00    7.2      2.58            0.029      -0.70 over 12
               minorminer   1.00    6.2      1.50            0.107
    p          policy       0.92    6.6      2.00            0.020      -0.60 over 11
               minorminer   1.00    5.8      1.08            0.144
    z          policy       0.92    7.3      2.27            0.023      -0.01 over 11
               minorminer   1.00    5.7      1.17            0.110

minorminer, resource-first, returns near-singleton embeddings; the constructor spends one
to two more qubits on chains of length two or three, and on two of the three sets the
per-instance gain tracks the extra qubits. Under a fixed schedule a strongly coupled chain
makes the logical spin heavier and its excited states rarer, so this is the objective's
own preference for spent qubits over minimal ones (the thesis), not necessarily anything
learned about placement. The control that separates the two is running: a third arm that
takes minorminer's draw and adds one or two random contact-growth qubits before the same
compile, selection and assessment (`--comparison minorminer_grown`, `*_grown*_s0.log`).
The control (`results/curriculum/*grown*_s0.log`, held-out, paired residual, policy minus the arm):

    stage, actor                    minus minorminer            minus minorminer + random growth
    b, 230 channels, +1 qubit       -0.077 [-0.143, -0.022]     -0.019 [-0.029, -0.009]
    b, 230 channels, +2 qubits      -0.077                      -0.018 [-0.039, -0.003]
    p, 16 channels, +2 qubits       -0.137 [-0.210, -0.067]     +0.000 [-0.003, +0.004]
    z, 16 channels, +2 qubits       -0.094 [-0.171, -0.020]     -0.001 [-0.010, +0.009]

    b, 16 channels, +2 qubits       -0.066 [-0.132, -0.008]     -0.007 [-0.017, +0.004]

On the hardware fragments, and on stage b with the 16-channel actor, minorminer's draw plus
two random contact-growth qubits reproduces the constructor's whole advantage: the learned
part is zero there, and the gain is the platform's spend-and-measure, which needs no
learning. On stage b with the 230-channel actor a quarter of the raw gap survives the
control on all three seeds: -0.018 [-0.039, -0.003], -0.004 [-0.018, +0.010], -0.010
[-0.020, -0.001] (`bq_grown2_s{0,1,2}.log`), two of three intervals below zero, all with
fewer qubits than the grown control (7.2 / 8.2 / 7.2 against 8.2 / 9.1 / 8.6). A small,
consistent learned-placement component that is not spending; it exists only with the 230
channels (the 16-channel actor is fully reproduced by the control). The quality-reward finals on stage b (`bq_lin230_s*.log`): -0.068 [-0.133,
-0.010], -0.033 [-0.088, +0.011], +0.005 [-0.009, +0.021] against minorminer, so the
quality reward did not add to what feasibility training built. Scale is small (4 to 8
variables, hosts of 9 to 24 qubits), so this is a rung, not the paper's number.

## High fill flips the regime: growth beats redrawing, and the policy still equals random

Contact policy from the planted witness on the Zephyr 4 fill corpus
(`results/contact/zephyr4_fill_init.log`, initial evaluation, 16 held-out instances, actual
start occupancy 0.89, spend 10 percent of the start, three candidates an arm selected by a
256-read block and assessed on a fresh 512-read block, objective the energy residual with the
sign flipped so that higher is better, failures counted at the declared penalty; none occurred):

    random contact growth minus the start     +0.0066 [+0.0047, +0.0085]
    policy minus the start                    +0.0077 [+0.0049, +0.0107]
    policy minus random                       +0.0012 [-0.0017, +0.0046]
    router redraws minus the start            -0.0052 [-0.0103, -0.0010]
    random growth minus router redraws        +0.0117 [+0.0070, +0.0171]

At low fill four router redraws with the same selection beat everything (+0.14 p_solve) and
growing one draw lost to them. At 89 percent occupancy the router's redraws are worse than
the planted start (this code still substitutes the start when a redraw fails, so the arm is
the start where the router failed and a worse embedding where it succeeded), and growing
contacts into the space that is left improves the residual over the start with the interval
above zero, on every arm that grows. So the platform's territory is the regime where the
router cannot redraw: complete or improve what exists, then measure. The learned part is
still zero here: the policy's proposals are worth exactly random proposals.

Pegasus 6 (`results/contact/pegasus6_fill_init.log`, same protocol, start occupancy 0.90):

    random contact growth minus the start     +0.0061 [+0.0042, +0.0082]
    policy minus the start                    +0.0057 [+0.0034, +0.0078]
    policy minus random                       -0.0005 [-0.0022, +0.0013]
    router redraws minus the start            +0.0003 [-0.0005, +0.0010]
    random growth minus router redraws        +0.0059 [+0.0042, +0.0078]

Here the router did not redraw a single instance at 90 percent within its 20 tries, so the
redraw arm is the start itself, and growth into the remaining space is the only thing that
moves the objective. Both hosts, both intervals above zero for growth, both policies equal
to random at initialisation. After 25 training iterations (`results/contact/*_fill_iter24.log`)
the policy is -0.0022 [-0.0042, -0.0003] below random on Pegasus 6 and +0.0007 [-0.0016,
+0.0033] on Zephyr 4, with growth over the start unchanged. Training on the measured
residual with REINFORCE has not moved the contact policy past random proposals at high
fill either; the platform's gain in this regime is growth plus measurement, not the policy.
At iteration 50 the same: -0.0005 [-0.0024, +0.0014] and +0.0007 [-0.0021, +0.0036]
(`results/contact/{pegasus6,zephyr4}_fill.log`, stopped there). Closed as a null for the
learned part, with the growth-plus-measurement gain standing on both hosts.

## The learned prior on allocation: signal at zero reads, a loss as a filter (B minus A)

The allocation probe with direction 5's surrogate as a prior (`results/adaptive/pegasus6_prior.log`,
Pegasus 6, 61 states in 32 lineages, every arm assessed on 512 independent reads, differences
paired over lineages):

    arm                  reads    p_solve    vs uniform                  vs halving
    uniform              1574     0.450
    halving               787     0.451      +0.001 [-0.005, +0.006]
    prior alone             0     0.388      -0.063 [-0.094, -0.035]     -0.063 [-0.096, -0.034]
    random pick             0     0.330      -0.121 [-0.162, -0.083]
    halving + prior       787     0.430      -0.020 [-0.036, -0.007]     -0.020 [-0.038, -0.006]

The prior is not empty: at zero reads it is +0.058 over a random pick, the same order as its
held-out +0.02 to +0.03 on ten times the lineages. But 787 reads of measurement are +0.063
above it, and using it to pre-select the half that halving then measures loses -0.020 with
the interval below zero: the candidates it ranks low include the ones measurement would
have kept. This was arm B minus arm A of the plan, the last place a learned component could
have entered the low-fill platform, and it is negative. Not tried: a soft prior (initial
allocation weights instead of a hard filter); the sign here and the +0.02 ceiling of the
surrogate say it could at best recover the -0.020, not add to halving.

## Resource count is a weak quality signal; measurement takes the ceiling (selection rules)

The paper's claim had no direct test in the repo, so `probes/selection_rules.py` makes one.
Thirty held-out instances per host (the contact runs' validation split of the modern
corpora), eight independent router draws each (20 tries a draw), every draw valid on both
hosts. Every rule picks among the same eight draws; every draw is assessed once on a fresh
512-read block under the registered schedule, so the rules differ only in the pick
(`results/rules/pegasus6.log`, `zephyr4.log`, p_solve, paired against the first draw):

    rule                       Pegasus 6                     Zephyr 4
    first draw (single)        0.405                         0.451
    random draw                +0.031 [-0.035, +0.106]       -0.034 [-0.101, +0.029]
    fewest qubits              +0.042 [-0.004, +0.101]       +0.041 [-0.011, +0.092]
    shortest longest chain     +0.040 [-0.007, +0.099]       +0.041 [-0.011, +0.092]
    measured, 256 reads        +0.147 [+0.087, +0.211]       +0.153 [+0.106, +0.201]
    oracle on the assessment   +0.163 [+0.109, +0.222]       +0.163 [+0.117, +0.209]

Within an instance the rank correlation between total qubits and assessed p_solve is -0.27
and -0.21 (fewer qubits slightly better, the direction the strength rule predicts), and
the fewest-qubit draw lands in the top half of its instance's draws 67 and 73 percent of the
time. So the resource count is a weak signal, not none: it takes a quarter of the best-of-8
gain with an interval that touches zero, while a 256-read measurement takes 90 percent of
the oracle's gain on both hosts. This is the platform's central number, on the target
hardware, with no failure accounting to explain: the objective is measurable and the
resource proxy is not a substitute for measuring it. It also agrees with the contact
control (+0.141 and +0.139 for four draws with the same selection) within noise.

How many draws and how many reads (`results/rules/*_k{4,8,16}_r{64,128,256}.log`, same
instances, same draws for a given K, p_solve minus the first draw):

    draws K, reads per draw     fewest qubits    measured           oracle       Pegasus 6
    K = 4,  256                  +0.037           +0.105             +0.113
    K = 8,   64                  +0.042           +0.151             +0.163
    K = 8,  128                  +0.042           +0.149             +0.163
    K = 8,  256                  +0.042           +0.147             +0.163
    K = 16, 256                  +0.048           +0.169             +0.187
                                                                                  Zephyr 4
    K = 4,  256                  +0.012           +0.120             +0.131
    K = 8,   64                  +0.041           +0.148             +0.163
    K = 8,  128                  +0.041           +0.150             +0.163
    K = 8,  256                  +0.041           +0.153             +0.163
    K = 16, 256                  +0.043           +0.161             +0.173

Sixty-four reads a draw already take 91 to 93 percent of the oracle's gain at K = 8, so the
measurement that separates draws is cheap; the gain grows with K roughly like the best of K
independent draws should, and the resource rule stays at +0.04 at every K. The full
platform cost of best-of-8 on these hosts is eight router draws of about a second each and
512 reads in total, against a single draw with no reads.

## Hybrid v3 is a null: policy layouts do not change what the router can complete

Five runs on the merged code with the fair protocol (both arms search until the 60 s deadline,
up to six valid candidates each, selection by measurement, assessment on a fresh block, full
host as the budget; `results/hybrid/v3_*.log`). Validity, held out over 30 instances, at
iterations 0, 19, 39, 59: policy+search 0.50, 0.53, 0.50, 0.50 against router+search 0.47 at
every checkpoint on Pegasus 3 (init from the imitation prioritiser and from scratch alike);
0.40, 0.40, 0.37 against 0.40 on Zephyr 2. Quality, paired residual where both arms are
valid: +0.008 [-0.013, +0.028] and +0.007 [-0.006, +0.018] at iteration 39, positive is
worse. The diagnostic is the candidate count: both arms find the same number of valid
candidates per instance at every checkpoint (0.5/0.5, 2.7/2.7, 2.3/2.3), so a policy layout
never turns an instance the router cannot complete in 60 s into one it can, and never adds
a candidate the measurement could prefer. With the roots result (witness roots turn 0 into
1.00; half-right roots suffice at 80 percent) the gap is the policy's roots, which after 60
iterations of reward from completion and measured quality are still worth nothing more than
the router's own restarts. Closed as a null; the learned-roots direction needs a different
learning signal than completion reward, if it is to be pursued at all.

## Contact growth with measured selection: the gain was the selection, withdrawn by its control

Contact policy, low fill, post-merge code (`results/contact/pegasus6.log`, `zephyr4.log`),
p_solve under the registered schedule, 30 validation instances, best of four samples chosen
by a measurement block, assessed on a fresh block, the start assessed the same way:

    arm minus start                    Pegasus 6                   Zephyr 4
    random contact growth, best of 4   +0.066 [+0.031, +0.114]     +0.079 [+0.043, +0.120]
    policy contact growth, best of 4   +0.038 [+0.016, +0.063]     +0.079 [+0.042, +0.122]
    policy minus random                -0.029 [-0.080, +0.013]     +0.000 [-0.032, +0.030]

High fill, from the planted witness at 96 percent occupancy, energy residual (sign flipped,
higher is better), 16 instances: policy minus start +0.0057 [+0.0034, +0.0078] and random
minus start +0.0061 on Pegasus 6; +0.0077 and +0.0066 on Zephyr 4.

The control decides it (`results/contact/pegasus6_control.log`, Pegasus 6, same protocol):
four fresh router draws selected by measurement beat the single start by +0.141 [+0.068,
+0.213]; random contact growth is -0.075 [-0.139, -0.009] below that control and the policy
-0.104 [-0.181, -0.026] below it. Zephyr 4 (`results/contact/zephyr4_control.log`) says the
same: restart minus start +0.139 [+0.084, +0.192], random -0.060 [-0.102, -0.019] and policy
-0.063 [-0.111, -0.016] below the restart control. The apparent gain of contact growth was the measured
selection over four candidates, and growing one draw is worth less than drawing again,
which the pool-ceiling result had already said. This is the twelfth withdrawn number, caught
by the control before it was reported as a claim. The learned policy adds nothing over
random proposals and both are below the router with restarts.

## The objective is quality; validity is the gate

minorminer optimises resource first, and its hundred percent from witness roots is
validity, not the objective. Under the registered schedule the witness embeddings sit below
minorminer's best of four on the energy residual in every fill cell where both exist
(Task 9: -0.021 to -0.032), with fewer qubits (544 against 560 at 80 percent on Pegasus 6):
at high fill the resource-first router completes with long chains, which the objective
punishes. The learned embedder is therefore trained and judged on measured quality among
valid embeddings, with the router alone as the baseline on the same instances; the search
over layouts continues until the deadline and selects by a measurement block.

Our router, final tables from witness roots (`results/ours/*.log`, negotiated congestion
with restarts, 30 s on the small hosts, 120 s on the big ones): Pegasus 3 from 1.00 at 70
percent down to 0.17 at 95 percent short chains, Zephyr 2 from 1.00 to 0.08; Pegasus 6 at
most 0.17, Zephyr 4 at most 0.83 at 80 percent medium chains and 0 from 85 percent short.
The standard router from the same roots: 1.00 in every small-host cell; on the big hosts
1.00 to 80 percent, then 0.67 / 0.83 / 0.00 / 0.50 (Pegasus 6, 90 medium, 90 short, 95
medium, 95 short) and 1.00 / 1.00 / 0.17 / 0.00 (Zephyr 4). Our router is not the completion
the platform deploys; it is reported beside the standard one.

## The roots are the missing information: minorminer seeded with witness roots

minorminer accepts initial chains. Seeded with one qubit per variable taken from the
witness, restarted with fresh seeds until the deadline (`results/seeded/*.log`):

    cell (12 a cell, 60 s)             no hint    random roots    witness roots   secs, witness
    Pegasus 3, 80 percent short         0.25         0.25            1.00           0
    Pegasus 3, 85 / 90 / 95 short       0 / 0 / 0    0 / 0 / 0       1 / 1 / 1      0
    Pegasus 3, 90 / 95 medium           0.42 / 0.25  0.42 / 0.25     1 / 1          0
    Zephyr 2, 80 percent short          0.08         0.00            1.00           0
    Zephyr 2, 85 / 90 / 95 short        0 / 0 / 0    0 / 0 / 0       1 / 1 / 1      0 to 1
    Zephyr 2, 90 / 95 medium            0.67 / 0.17  0.75 / 0.17     1 / 1          0

Target hardware, 6 a cell, 300 s budget (`results/seeded/pegasus6.log`, `zephyr4.log`):

    cell                     no hint    witness roots   random roots   secs, none   secs, witness
    Pegasus 6, 80 medium      1.00        1.00            0.33            21            0
    Pegasus 6, 80 short       0.00        1.00            0.00           312            2
    Pegasus 6, 85 med/short   0.17/0.00   1.00/1.00       0/0            261/315        1/1
    Pegasus 6, 90 med/short   0/0         0.83/0.83       0/0            309/313      124/89
    Pegasus 6, 95 med/short   0/0         0.17/0.50       0/0            309/307      270/188
    Zephyr 4, 80 medium       1.00        1.00            0.83             2            0
    Zephyr 4, 80 short        0.00        1.00            0.00           313            0
    Zephyr 4, 85 med/short    0.83/0.00   1.00/1.00       0.83/0         56/311         0/1
    Zephyr 4, 90 med/short    0.33/0.00   1.00/1.00       0.33/0        207/308        17/41
    Zephyr 4, 95 med/short    0/0         0.50/0.33       0/0           309/309      166/235

Same regime on the hardware the paper targets: the plain router is 0 on short chains from 80
percent and times out at 300 s; witness roots complete every cell to 85 percent in seconds,
most of 90 percent in one to two minutes, and a third to a half of 95 percent; random roots
add nothing. Full tables in `results/seeded/pegasus3.log`, `zephyr2.log`, `pegasus6.log`,
`zephyr4.log`.

Random roots do nothing, so the effect is the information in the roots, not the act of
seeding; and with the right roots the completion takes under a second at 680 qubits. The
constructive policy places well and cannot finish; minorminer finishes and cannot place. The
learned embedder is therefore the hybrid: a policy that predicts roots, minorminer's search as
the completion, measured against unseeded minorminer at the same deadline. Two runs decide
the shape of the learning problem: how accurate the roots must be (`*_tolerance.log`: half,
quarter, one-step-noisy witness roots) and what the current PLACE-only policy's roots are
worth (`runs/hybrid/*.log`); its first lines show it placing every variable in seconds and
agreeing with the witness on almost none.

Tolerance of the completion (`results/seeded/*_tolerance.log`, short chains, 12 a cell, 60 s):

    fill        none    half   quarter   noisy      none    half   quarter   noisy
                ---- Pegasus 3 ----                        ---- Zephyr 2 ----
    80 percent  0.25    1.00    0.50     0.58       0.08    0.92    0.25     0.17
    85 percent  0.00    0.67    0.08     0.00       0.00    0.17    0.08     0.08
    90 percent  0.00    0.25    0.08     0.00       0.00    0.00    0.00     0.00
    95 percent  0.00    0.00    0.00     0.00       0.00    0.00    0.00     0.00

Half of the roots right is enough at 80 percent and the requirement tightens with fill: at
90 to 95 percent nearly all roots must be right. The imitation prioritiser's PLACE-only roots, one greedy layout, are
worth nothing (`results/hybrid/pegasus3.log`, `zephyr2.log`: per cell equal to the unseeded router, with the witness roots at 1.00 in every cell), and agreement with one
witness is the wrong measure since the witness is one of many symmetric layouts: minorminer
needs a globally consistent layout, which a locally trained scorer does not give. So the
layout is learned against the deployment signal itself: PLACE-only episodes, minorminer
completion under a short deadline, reward one if it succeeds (`probes/train_hybrid_rl.py`).

## The constructive policy so far: it places, it does not finish

With the budget features the prioritiser's teacher agreement is 0.342 (Pegasus 6) and 0.336
(Zephyr 4), the teacher's pick at median rank 1 (`results/imitation/*_budget_train.log`).
Deployed on the curriculum hosts with episodes sampled until a 30 s deadline, the imitation
policy reaches 0.71 (Pegasus 3) and 0.64 (Zephyr 2) of demands and no valid embedding in
any cell, from 70 to 95 percent fill; from scratch 0.59 (`runs/rl/*_init.log`, `*_scratch.log`).
Before the budget it ended every episode with the host full at 83 percent of demands
(`results/rl/*_v3.log`, `*_v4.log`); under the budget it runs to the deadline without
finishing. The failure is in the last third of the demands, the constrained completion,
not in placing variables. probes/placement_completion.py measures whether a fixed
completion rule finishes from witness roots, which decides whether the learned part can be
placement alone.

## Data scale (direction 5): the surrogate transfers a little, and more data helps a little

Three seeds on 1680 training lineages of Pegasus 6 under the registered schedule with the
corrected compiler, 696 held-out states in 360 lineages (`results/large/train_seed*.log`):
+0.0205 [+0.0094, +0.0318], +0.0273 [+0.0147, +0.0387], +0.0230 [+0.0110, +0.0354]. At 160
lineages the same encoding gave +0.003. The prediction direction is not closed. The small and
large runs use different evaluation sets, so they do not establish a controlled scaling law
or a limit to data scaling. The measured-selection reference is not a theoretical ceiling.
The practical next question is whether this ranking signal helps adaptive allocation
(direction 3) against plain successive halving at the same cost. These are validation results,
not an unopened final-test result.

## Adaptive allocation of reads (direction 3)

Zephyr 4, 63 held-out states in 32 lineages, every read a real evaluator call, the chosen
candidate assessed on 512 independent reads (`results/adaptive/zephyr4.log`):

    arm                     reads   selected quality   minus uniform, 95% over lineages
    uniform, 8 x 256         1544       0.5043
    successive halving        760       0.5025          -0.0018 [-0.0078, +0.0035]
    UCB, blocks of 32         772       0.4972          -0.0071 [-0.0148, -0.0002]
    random, no reads            0       0.3809          -0.1234 [-0.1617, -0.0856]

Pegasus 6, 61 states in 32 lineages (`results/adaptive/pegasus6.log`): uniform 1574 reads
0.4456; halving 775 reads 0.4477, +0.0021 [-0.0042, +0.0088]; UCB 787 reads -0.0048
[-0.0133, +0.0047]; random 0.3338, -0.1118.

Successive halving reaches the quality of uniform selection with half the measurement, on
both current topologies. This
is the first positive result of the relaunch, and it is the measured-selection finding made
into a mechanism: the reads that decide are the ones spent on the contenders. A learned
prior over candidates can only be judged against this, not against uniform.

## Checkpoint 2: the corrected surrogate does not transfer either (Task 7)

Six trainings per host on the registered-schedule labels, three seeds with the corrected
encoding and three with the legacy one, identical labels, picks and assessment seeds, 36
held-out lineages per host (`results/corrected/*_seed*.log`):

    held-out gain      corrected                      legacy
    Pegasus 6          +0.006  -0.002  +0.006         +0.039  -0.007  +0.002
    Zephyr 4           +0.029  +0.037  +0.040         +0.042  +0.037  +0.043

The two encodings differ by less than 0.01 where the seed spread is 0.04, and no corrected
interval lies above zero. The ceiling measured on the same labels is +0.11 to +0.12 (Task 8).
D1 was a real defect and not the cause. By the rule written in the spec before the run, the
improvement surrogate is retired as a method; what remains of the prediction direction is the
data-scale closure on the 2400-instance corpus, which decides whether it is closed for good.

## Label reliability under the registered schedule (Task 8)

Two independent 512-read blocks per candidate, 61 and 63 states in 32 lineages per host,
bootstrap over lineages (`results/corrected/*_reliability.log`):

    quantity                         Pegasus 6                 Zephyr 4
    rank correlation A vs B          +0.72 [+0.64, +0.79]      +0.77 [+0.72, +0.82]
    top choice agrees                 0.59 [0.45, 0.73]         0.64 [0.52, 0.77]
    select on A, assess on B         +0.109 [+0.086, +0.133]   +0.122 [+0.094, +0.152]
    oracle on B, inflated            +0.119                    +0.129
    spread within a state             0.24                      0.27

The select-on-A row is the finite-read selection reference for anything trained on these
labels: a perfect predictor of the 256-read label earns +0.11 to +0.12 on fresh reads at
this read count. It is a reference, not a theoretical bound. The winner's curse in the oracle
row is about 0.01. The labels are reliable; the gap between the learned head's +0.01 and this
ceiling is generalisation, now with an interval.

Labels under the registered schedule with the corrected compiler exist for both modern
corpora (`results/relabel/`, 2298 and 2329 labelled candidates, 17 and 19 minutes). A side
observation from the labelling run's own one-epoch head, preliminary at 69 and 71 held-out
states: on Pegasus 6 picking the cheapest candidate scores 0.459 against 0.408 for random
choice and 0.525 for the label oracle; on Zephyr 4 it scores 0.438 against 0.440 and 0.537.
Under the auto schedule resource selection sat at chance on every corpus. A fixed temperature
makes compression cost something, which is what ADR-002 says it should; measured selection
still beats resource selection by about twice its margin. The corrected-versus-old comparison
and the reliability probe are running on these labels.

Task 9, Pegasus 6, registered schedule (`results/fill/pegasus6_witness_registered.log`): at
80 percent with medium chains, the only cell where minorminer finds anything, witness minus
minorminer best on the residual is -0.0316 [-0.0471, -0.0166] over 5; solve probability zero
on both sides everywhere.

Task 9, Zephyr 4, registered schedule (`results/fill/zephyr4_witness_registered.log`): paired
witness minus minorminer best of four on the energy residual, lower is better: 80 percent
-0.0207 [-0.0300, -0.0095] over 6, 85 percent -0.0148 [-0.0296, +0.0035] over 5, 90 percent
-0.0161 [-0.0344, +0.0022] over 2. Solve probability is zero on both sides everywhere. The
quality endpoint at the fill regime exists and it is the residual under the registered
schedule; the cells are small and the thirty-per-cell corpus of Task 10 is where it gets its
final interval.

Task 12 at corpus scale, final: with the witness as the generator's preference the
construction environment reaches a valid COMMIT on every instance of both fill corpora,
48 of 48 per host, every cell from 80 to 95 percent with short and medium chains, in 400 to
620 decisions and 200 to 530 seconds (`results/fill/*_replay_hint.log`). With the unhinted
heuristic order it reaches none (`*_replay_nohint.log`). The grammar is sufficient; the
prioritiser is the whole question, and its imitation records are 125 and 94 MB.

Task 12 at corpus scale, how it got there: the first replay on Pegasus 6 stalled within four decisions, for two
reasons found by direct diagnosis, both invisible at 100 qubits. One is structural: a ROUTE may
only add qubits that realise a demand between placed chains, so a chain's further qubits
could never appear before the neighbours that need to touch them, and those neighbours could
not be placed first. REWRITE_ONE accepts a superset chain, so construction gained a "grow"
family. The other is coverage: a witness root is one qubit among about fifty adjacent free
qubits, and a budget of a few roots per variable in name order covers it about half the time
per step. The registered cap of 64 candidates applies to a decision, not to the generator's
internal ranking, so the generator gained a preference hook consulted before truncation. With
the witness as the preference, the replay reaches a valid COMMIT on a 366-variable Zephyr 4
instance in 445 decisions; without it, the heuristic order stalls at once
(`results/fill/*_replay_nohint.log`). That is the design for the constructive policy: the
learned scorer is the prioritiser of its own shortlist.

Task 12 done in the environment (`src/isingfold/rl/proposal.py`, `probes/_context.py`): PLACE
follows placed logical neighbours and offers a spread of roots for the first placement; the
horizon scales with the instance; ROUTE offers, after the router's path, one-qubit bridges to
either owner and two-owner meeting routes, all charged by the neighbour scans that found them.
A planted witness now replays to a valid COMMIT on 16 and on 100 qubits with 70 to 76
variables, three seeds, chains inside the witness's. The suite shows no new failure.

Construction API, measured before any policy: on a 16-qubit toy host the environment in
construction mode reaches COMMIT with exactly the planted witness in 12 decisions when each
step picks a witness-consistent candidate (`tests/unit/test_witness_replay.py`). On a 100-qubit
host with 76 variables it fails at step 0: PLACE offers 24 roots, the lexicographically first
ones, for the first unplaced variable only, and the witness root is not among them. That, and
the 32-decision horizon, is what Task 12 has to change before imitation is possible.

The energy residual discriminates where solve probability reads zero: on Zephyr 4, under the
auto schedule, the witness sits at 0.122 to 0.132 and minorminer's best of four at 0.142 to
0.147 in the three cells where both exist (`results/fill/zephyr4_witness_res.log`); six
instances a cell, no interval. The registered-schedule run with paired intervals is Task 9.

## Three facts about the fill regime, measured before anything is built for it

- **The in-tree constructor is at zero.** `router_initializer`, the greedy degree-order
  placer behind the environment's PLACE and ROUTE proposals, produces no valid embedding on
  any fill-planted instance at 80 to 95 percent on either host, 24 draws per cell, under a
  second each (`results/fill/*_construct.log`). A learned constructor starts from that floor.
- **Solve probability is zero for every embedding at this scale.** On Zephyr 4 the witness and
  minorminer's best of four both read 0.0000 in every cell (`results/fill/zephyr4_witness.log`).
  At 280 to 440 variables of frustrated loops, 512 reads never reach the planted ground state,
  whatever the embedding. The registered objective does not discriminate here; the mean energy
  residual above the planted ground energy, which the evaluator already computes, is being
  measured in its place.
- **Five times the budget moves minorminer one cell.** Zephyr 4 at fifty tries: 85 percent fill
  with medium chains rises from 0.50 to 0.67 and 90 percent stays at 0.33, at 35 to 80 seconds
  a draw; short chains stay at zero (`results/fill/zephyr4_budget.log`).

With chains of length exactly one, where minimal fill equals planted fill, minorminer finds
nothing from 70 percent up on either host at 60 to 100 seconds a draw
(`results/fill/*_exact.log`, running).

## The embeddability threshold on Pegasus 6 and Zephyr 4

Validity rate of minorminer at ten tries, six instances by four draws per cell, in
`results/feasibility/`:

    family      Pegasus 6                          Zephyr 4
    clique      59: 1.00  60: 0.75  61: 0.38  62: 0.08   57: 1.00  58: 0.83  59: 0.58  60: 0.21
    dense       128: 1.00  136: 0.25  144: 0.00           136: 0.92  144: 0.00
    scalefree   176: 1.00  184: 0.62  200: 0.04           184: 0.96  200: 0.00

At fifty tries the clique threshold moves by about one variable (Pegasus 61: 0.88, 62: 0.25;
Zephyr 59: 0.88, 60: 0.25) at three to five times the wall time. The deterministic clique
embedder stops at K60 on Pegasus 6 and K56 on Zephyr 4, below the random search, so the
threshold is not already solved by the standard tool. At the threshold the host is about
88 percent full and a draw costs eight to fifty seconds.

Two facts before anything is built for this regime. The current learned embedder cannot
enter it: it starts from a minorminer embedding, and here there is none. And whether a valid
embedding at the threshold has any solve probability is unmeasured; if every one is near
zero, a validity win is a win on problems the annealer cannot solve.

## Withdrawn, and why

Seven numbers have been published here and then withdrawn. Four were caught by external audit
rather than by me.

- Two improvement-target figures, -0.1139 and -0.3107, from a comparator that measured against
  the wrong embedding.
- The critic being anti-correlated at -0.19, measured with V(s) against a target V cannot
  represent; on Q(s,a) the same checkpoints are weakly positive.
- An inflation figure of 0.0303 in the README, read from the first two rounds of a sixty-round
  run and written down as if it described the run; the true mean is 0.0151.
- A reading of three validation points as a downward trend at round 13, when the full curve rose.
- "The head captures 84 percent of the ceiling on fitted states", produced by a run with no
  stopping rule and wrong labels.
- "Three unrelated methods fail in the same place", which stopped being true when one of the
  three turned out to be measuring wrongly.
- The successor scorer at +0.0362, a single seed that three seeds put at +0.0120.
- "The resource-quality premise fails." It was measured by correlating qubit count with quality
  across the candidates at a state. Those are local perturbations of one embedding, and the
  monotonicity the design claims is a property of the optimum at each budget, which that
  measurement never touches. The design says so explicitly and it was read and then contradicted
  anyway. Asked within instances instead, the sign flips on the hard corpus to +0.0141 with the
  interval containing zero. Neither measurement tests the premise: the cheaper and dearer halves
  are 43.5 and 48.9 qubits apart, a twelve percent range rather than a budget sweep. The honest
  statement is that nobody has tested it.
- Contact growth with measured selection at +0.066 and +0.079 over the start, which was the
  selection over four candidates: four fresh router draws with the same selection give +0.141,
  and contact growth sits 0.08 to 0.10 below that. Withdrawn by its own control.
- A mixed proposal pool at +0.011 and +0.021, which was sixteen measured candidates against
  eight. Withdrawn from the log before it was reported.
- Contact growth at +0.0996 and +0.0636, which was a maximum over two draws against a start
  measured once. Caught before publication this time, by running the control first.
- Reading the mean column of a frontier table and calling the frontier flat, when the best column,
  which is what a frontier is, rose from 0.8242 to 0.9727 before falling.

The pattern is the same every time: report the first result, state its limits correctly, then let
it become the headline anyway. Nothing goes into the results README now before three seeds.

## The solvability frontier, 2026-09-18

Asked: restore solve probability on the congestion axis so the paper reports one objective
instead of two. Answer: it cannot be restored above about 100 variables, the reason is a
measured property of the sampler rather than a choice, and two declared regimes are the
stronger design. Probes `probes/solvability_frontier.py` and `probes/witness_prune.py`, logs
in `results/frontier/`, adversarial read in
`docs/review/2026-09-18-codex-gpt6-astra-review-three-solvability.md`.

**The diagnosis.** Two independent difficulties were being read as one number. Fill sets how
hard an instance is to embed. Frustrated-loop density sets how hard it is to solve once
embedded. Loop density is not usable as a knob: an edge survives into the logical graph only
where its summed coupling is nonzero (`planting.py`, the `active` filter), so a sparser problem
is a sparser embedding problem and the congestion claim moves with it.

**What was ruled out, by measurement, not argument.**

| knob | what it leaves alone | result | log |
|---|---|---|---|
| local field on a fraction of nodes | every edge, so the embedding problem exactly | dead: p_solve stays 0.000 at 195 to 475 variables at field rates 0.10, 0.25 and 0.50 | `results/frontier/*_field.log` |
| anneal depth, 200 to 20000 sweeps | the instance entirely | buys about 15 variables per decade of sweeps, no change in decay rate | `results/frontier/*_f90.log` |
| host size at fixed fill | one ratio, not the graph | restores p_solve, but builds a different instance and does not preserve congestion | `results/frontier/*_f90.log` |

**The law.** Decoded solve probability decays exponentially in the variable count at fixed
congestion, and the decay rate is invariant to anneal depth over two decades:

| sweeps | slope of log p_solve per variable | variables at p_solve 0.05 |
|---|---|---|
| 200 | -0.0499 | 66 |
| 2000 | -0.0383 | 83 |
| 20000 | -0.0457 | 80 |

Fitted on cells with a resolvable rate, so the zeros are censored and the true slope is at
least this steep. A hundredfold increase in sweeps moves the intercept, not the slope, so
reaching 434 variables at p_solve 0.05 needs roughly twenty-four further decades of sampling.
That is the quantitative reason the congested regime reports residual, and it replaces the
earlier hand-wave.

**The named fill overstates the congestion.** The witness certifies a sufficient occupancy,
not a necessary one. Greedy pruning, keeping every chain connected and every logical contact
realised:

| host, named fill | witness qubits | pruned | certified fill | lower bound | minorminer valid at 200 tries |
|---|---|---|---|---|---|
| Pegasus 2, 0.90 | 36.0 | 33.5 | 0.84 | 0.75 | 0.58 |
| Pegasus 2, 0.95 | 38.0 | 34.8 | 0.87 | 0.78 | 0.58 |
| Zephyr 1, 0.90 | 43.0 | 41.2 | 0.86 | 0.74 | 0.25 |
| Zephyr 1, 0.95 | 46.0 | 44.2 | 0.92 | 0.79 | 0.17 |

Named fill runs 5 to 7 points above what is certified. Where minorminer succeeds it uses about
as many qubits as the pruned witness (33.6 against 33.5, 42.3 against 41.2), so its failures
are failures to find anything, not failures to pack tightly. Corpus cells should be reported by
pruned fill from here. `results/frontier/prune_*.log`.

**All three channels discriminate, given a large enough dose.** Same instance, same host,
chains lengthened by absorbing free qubits, Pegasus 3 at fill 0.50, 2000 sweeps:

| absorbed | mean chain | p_solve | residual | broken fraction |
|---|---|---|---|---|
| 16 | 1.28 to 1.60 | 0.234 to 0.203 | 0.0484 to 0.0514 | 0.008 to 0.009 |
| 32 | 1.23 to 1.86 | 0.128 to 0.087 | 0.0449 to 0.0528 | 0.005 to 0.012 |
| 64 | 1.23 to 2.46 | 0.212 to 0.110 | 0.0335 to 0.0572 | 0.002 to 0.014 |

The earlier null at 8 absorbed qubits was an underpowered dose, not an insensitive channel.
`results/frontier/mech_*.log`, `results/frontier/discrim_*.log`.

**The one congested cell where both conditions hold.** Zephyr 1 at named fill 0.95, 38
variables, 12 instances, 200-try minorminer: p_solve 0.225 at the registered strength and 0.368
at the best of four, residual 0.0325, minorminer valid on 0.08 to 0.17 of instances. Pegasus 2
does not qualify: minorminer solves 0.55 to 0.58 of its fill-0.95 instances, so the earlier
"0 of 3 at 20 tries" was small-sample and small-budget. `results/frontier/cell_zephyr1_f95_*.log`,
`results/frontier/confirm_*.log`.

**Verdict.** Keep two declared regimes. Solve probability is primary below about 80 variables,
where it is measurable and now shown to discriminate; residual and feasibility are the
endpoints above it, with zero-hit upper bounds reported rather than zeros. The regime boundary
is itself a result, with the depth and field arms as the evidence that it is not a choice.

**Corrections this work forced.** Four claims were wrong and are retracted here: p_solve is not
"exactly zero" at scale, only bounded below 0.006; holding fill fixed does not hold the logical
graph fixed; named fill is not certified congestion; and the earlier table compared the
witness at the registered strength against minorminer at its best strength, which is two
different strength policies and not a like-for-like comparison.
