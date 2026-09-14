# `isingfold.rl`: IF-Core and the exact embedding-search environment

Implementation of `documents/ISINGFOLD_MODEL_SPEC.md` (model, environment contract, masked
PPO) and `documents/IsingFold_Architecture_Rev2` (objective IF-Q3-S0, data generation,
evaluation protocol). The environment owns validity, candidate materialisation and work
accounting; the model only reorders legal actions and can never grant validity.

## What each module owns

| Module | Responsibility | Must not depend on |
|---|---|---|
| `contracts` | context registry, work ledger, opcodes, decision/terminal records | any learned score |
| `program` | four-strength compilation, shrink-only autoscale, exact faithfulness | search caches |
| `validate` | `p_search`, `p_embed`, `p_return`, rebuilt from the serialised assignment | a cached Boolean |
| `router` | occupancy-weighted routing; overlap is priced, not forbidden | outcomes |
| `proposal` | bound, outcome-blind candidate batches with quotas and charged work | witness, optimum, evaluator |
| `env` | prepare → decide → apply → prepare once; archive, reserve, terminals | learned validity |
| `tensorize` | Appendix A schema: 120 scalar slots as value/knownness pairs | witnesses, ground energies, keys |
| `model` | IF-Core actor and utility/failure critics | environment mutation |
| `strength` | the frozen IF-Q3-S0 program-feature selector | `E_0`, planted spins, counts |
| `selector_audit` | post-freeze selected/F2/random/oracle diagnostics and immutable receipts | fitting, calibration, actor updates |
| `rollout`, `ppo` | complete-episode storage, GAE, masked PPO, transactional KL rollback, normalized entropy floor | candidate regeneration |
| `data/` | authenticated EmbedBench import, target-blind prepared tasks, publisher attestation, selector and replayable quality labels | policy access to evaluator targets |
| `initializer_bank` | target-free per-seed initializer plans, deterministic draw ledgers, immutable snapshots and authenticated training access | validation/test targets |
| `complete_system`, `complete_system_aggregate` | fresh initializer, pre-initialization denominator, raw evidence and three-seed endpoint | conditional success-only filtering |
| `capacity_control` | isolated near-parameter-matched IF-Core versus deep IF-Dual diagnostic | main-grid reselection |
| `evaluation_shards`, `evaluation_workflow_cli` | resumable whole-lineage plans, workers and complete-census merge for learned, stock and tuning workflows | shard-local sampling decisions |
| `final_strength_audit`, `final_strength_workflow` | outcome-blind final-strength plan, sealed source evidence, paired audit reads and merge | model or baseline reselection |
| `evaluator`, `evaluate`, `external`, `external_pairing`, `gates` | fresh reads, paired endpoints, diagnostic and authenticated complete-system external baselines, the four release gates | the trained checkpoint |

## The objective

IF-Q3-S0: maximise majority-decoded ground-state solve probability at a strength chosen by a
**frozen program-feature predictor**, subject to exact validity, a qubit cap and a work
budget. Qubits are a cap and a reported cost, never a reward. The four-strength maximum is an
oracle diagnostic, not a deployable result.

Transactional KL rollback and normalized entropy control training stability only. They do not
change IF-Q3-S0, terminal reward, data partitions, target labels, or the deployment action rule.

## Production pipeline

The production boundary is an authenticated CandidateBank export from the independently
installable `packages/EmbedBench` package in this monorepo. IsingFold consumes that export; it
does not import or fork the data generator. The ordered stages are:

1. `prepare`: verify provenance-complete CandidateBank-v2 records against an independently pinned
   IF-Core corpus-design manifest v2 and publish partition-sealed scientific prepared schema v4. The importer checks
   exact unique-base-lineage quotas, realized condition census, host/fault/problem-origin diversity,
   sealed-test OOD coverage, bound outcome-blind difficulty calibration, and explicit
   null-boundary-separation power/precision arithmetic. The primary noninferiority and paired-
   utility precision filters must cover every axis value realized in sealed test; hard/OOD cells
   remain quota-controlled subgroups. Registered independent-lineage floors are at least 1,024
   train, 512 validation, and 1,546 test bases, with test also covering the largest confirmatory
   target. Validation baseline tuning additionally requires the named valid-return noninferiority
   and paired-utility precision targets to cover the exact full realized validation population.
   Their prospective registry minimum is 128 independent lineages: the power assumptions
   (one-sided alpha 0.05, power 0.8, discordance 0.1, separation 0.07) calculate to 127, while the
   bounded paired-difference 95% half-width 0.2 calculation gives 97. These are denominator-
   adequacy guards for typed complete-population construction, not post-selection inference. The
   importer then
   separates policy-visible instances, protected initializers, and evaluator-only targets.
   Prepared v1/v2/v3 require an explicit diagnostic opt-out. Every target-opening command additionally
   requires an out-of-band publisher ID and attestation digest, then verifies exact evidence-manifest
   and certificate-artifact coverage. `verify-ground-certificates` executes the pinned standalone
   checker over the full target census and publishes ground-certificate protocol v2: one target-free
   root plus physically separate train, validation and test partition receipts. A target-opening
   consumer must authenticate the root and obtain only the receipt for its selected partition.
2. `label-selector-data`: sample the complete four-strength registry once and publish immutable
   train/calibration labels.
3. `fit-selector`: train and freeze the independent graph strength selector from those labels.
4. `label-quality`: create replayable `Q^mu(s,a)` warm-start labels under a named frozen
   continuation policy. Sampling is lineage-first, and every v7 row binds its prepared task,
   exact state, complete support, observation, continuation seeds and outcomes. Production
   planning and replay use one externally pinned, target-free persistent K=2 initializer bank;
   every row binds its bank episode and bootstrap snapshot. Long jobs may use deterministic
   whole-lineage shards across Apollo and Goose.
5. `merge-quality-labels`: require a complete nonoverlapping shard census, reauthenticate and
   exact-replay every row, recompute all denominators, and atomically publish a canonically ordered
   corpus. Individual shards cannot enter preflight, warm-start training, or the scientific grid.
6. `quality-preflight`: authenticate and exactly replay every row, recompute simultaneous
   plausible-best sets, and enforce preregistered resolved-row and independent-lineage minima.
   Warm start uses admitted train rows for both subset-normalized actor ranking and a detached,
   count-aware utility-critic target for the registered uniform-action-then-`mu` policy. Evaluator
   outcomes remain loss targets only and never become policy features.
7. `grid-cell`, `evaluate-representation-cell`, and `select-representation`: run and freeze the
   exact 9-cell representation screen from `configs/rl_grid_hybrid_v1.json`. Its authenticated
   payload is staged-grid schema v2 named
   `if-core-v2-profile-i-hybrid-chimera-registered`; the filename's `v1` identifies the first
   hybrid protocol revision, not the schema version. Aggregate equally over base lineages and all
   three registered seeds, apply
   the frozen feasibility noninferiority gate against `return_initial`, then select by unconditional
   IF-Q3-S0 utility and lower online cost. Evaluation uses a dedicated target-free persistent K=2
   bank under the `representation-validation` preset and seed 33049. Selection authenticates all
   five raw arms, all five complete-system receipt layers, and byte-identical bootstrap clones
   across arms. The RL-value `validation` bank uses seed 44021 and cannot be substituted.
8. `grid-cell`, `evaluate-rl-value-cell`, and `select-rl-value`: run all 18 supervised/PPO cells,
   authenticate their validation reports, and freeze one training configuration across all three
   seeds. A single best seed is never selected.
9. `audit-selector`: after the registered RL-value freeze, evaluate complete authenticated
   `audit_val` or `audit_test` selector partitions. Test access requires the exact RL-value receipt
   and its caller-supplied SHA-256. Audit rows are never accepted by selector fitting and cannot
   revise either model-selection receipt.
10. `plan-initializer-bank`, `generate-initializer-bank-shard`, and `seal-initializer-bank`: before
    target-bearing training, create one target-free conditional schedule per registered seed,
    preserve every failed draw before the accepted initializer, and seal exactly the immutable
    snapshots consumed by PPO. An exhausted conditional episode fails the bank instead of being
    silently replaced.
11. `complete-system-train-cell`, `evaluate-complete-system-cell`, and
    `aggregate-complete-system`: freshly retrain all three frozen seeds without checkpoint resume.
    Each PPO run consumes its pinned initializer bank. Evaluation still starts from the full sealed
    population before initialization, keeps initializer failures as zero utility, authenticates raw
    receipts and terminal evidence, and computes a crossed seed-by-lineage interval.
12. `evaluate`: assess locked checkpoints against same-support controls using independent reads.
13. `evaluate-external-tuning-cell` and `select-external-tuning`: run the fixed seven-candidate by
   three-seed stock-minorminer grid on the full validation population, then freeze one strategy
   without selecting a seed. All seven candidates within one seed share one machine identity.
14. `plan-evaluation-shards`, `run-evaluation-shard`, and `merge-evaluation-shards`: execute learned
    confirmation, tuned-stock confirmation, or external validation tuning as resumable whole-lineage
    shards. The plan authenticates the complete public population before target access; the merge
    rejects missing, duplicate or overlapping shards and recomputes canonical report inputs.
15. `evaluate-external-complete-system`: authenticate that frozen selection before opening any test
   target, then run one of three row-matched stock-minorminer evaluations on the same sealed census,
   seed schedule, selector, work cap, total online envelope, and compute identity as its learned
   counterpart. Each row includes replayable terminal evidence.
16. `aggregate-learned-vs-stock`: authenticate three pinned learned reports and three pinned stock
   reports, replay all six evidence artifacts against the pinned corpus and quality authority, then
   compute paired crossed-bootstrap utility, feasibility, and rowwise total wall-clock inference.
17. `plan-final-strength-audit`, `seal-final-strength-audit-execution`,
    `run-final-strength-audit-shard`, and `merge-final-strength-audit`: after the primary learned versus
    stock comparison is frozen, perform the preregistered independent A/B selected-strength audit.
    This diagnostic reports selector regret and strength sensitivity but cannot change the selected
    model, baseline or primary endpoint.

`evaluate-external` remains available for diagnostics with explicit seed and repetition controls.
It is not the publication-facing comparison command.

The external arm is not a proposal-source ablation. It has its own raw restart receipts, terminal
evidence, and wall-clock protocol. Its hard-deadline adapter starts one isolated worker per
task/repetition and reuses it across registered restarts, so no equal-internal-work claim is made
and the single adapter startup remains inside measured online time. Validation chooses among the
documented one-call ten-try default and time-saturating one-try restart variants. The latter use
either resource ordering or frozen-selector predicted `p_solve` ordering without evaluator
feedback. A
missing or version-mismatched optional `minorminer` package produces an
explicit unavailable report with no fabricated outcome rows. OCT and ATOM remain unimplemented
backend slots until native, versioned adapters are provided.

The exact commands, artifact contracts, Apollo direct launcher, and Goose Slurm launcher are in
[`documents/TRAINING_OPERATIONS.md`](../../../documents/TRAINING_OPERATIONS.md). Install the RL
dependencies with `python3 -m pip install -e '.[rl]'`.

`dev-generate` exists only for unit and end-to-end smoke tests. Its artifacts are marked
development-only and production commands reject them.

## Gates before architecture selection

`gates` reports the four release gates and refuses to summarise them as a single number:

1. **Exact conformance:** witnesses pass the independent validator, all four programs satisfy
   the coefficient-sum identity, and exact structural labels agree with enumeration.
2. **Support headroom:** the spread of independently scored quality inside the candidate
   batch. Below the meaningful effect size, stop before PPO instead of enlarging the model.
3. **Selector discrimination:** resolvable spread across the four strengths, and whether the
   frozen selector beats the fixed `F_2` control and a random choice.
4. **Valid returns:** rollouts return valid embeddings often enough, with a utility column
   that is neither all zero nor all one.

## Deliberate limits

* Difficulty is two axes, not one score: `--variables`, `--chain-size` and `--fault-rate`
  press on embedding, `--alpha`, `--clause-length` and `--weights` press on sampling. An easy
  cell produces a saturated utility column and gate 4 says so.
* A simulator result is a simulator claim. Nothing here establishes a QPU improvement.
* `forward` batches logical, hardware, conflict, and ownership graphs as a disjoint union. It
  slices that trunk before the observation-local factor, route, archive, action, and categorical
  reductions. Environment collection remains sequential because each support depends on the
  selected predecessor action.
