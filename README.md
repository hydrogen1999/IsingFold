# IsingFold RL

The learned embedder for quantum annealers: the reinforcement-learning runtime, the data
generation it consumes, and the evaluation that judges it. Everything here is what the model
needs and nothing else; the sibling analysis workspaces the research grew out of are not
included.

## What the problem is

The primary learned method is the **independent constructor** in
`probes/train_constructor_rl.py`: every episode starts empty and the policy chooses the
placement, route/rewrite, refinement and COMMIT actions. Minorminer is an optional separate
baseline, never a completion fallback. Public qubit budgets constrain the search; the
quality reward has no per-qubit penalty. Architecture, loss, limits and commands are in
[ADR-005](docs/decisions/ADR-005-independent-constructor.md). Historical hybrid probes learn
root layouts for a router and must be reported separately.

A logical Ising problem has to be mapped onto a fixed hardware graph, Chimera or Pegasus, by
giving every logical variable a connected chain of qubits so that adjacent variables' chains
touch. The usual objective is to use few qubits and short chains. This project measures a
different one: the fraction of annealer reads that decode to the ground state, at the chain
strength the method selects. That number is what a user of the machine actually gets, and it is
not a function of qubit count.

## Layout

| Path | What it holds |
|---|---|
| `src/isingfold/rl/` | The environment, the action grammar, the IF-Core actor-critic, masked PPO, checkpointing, the quality objective, the data trust boundary, and evaluation |
| `src/isingfold/` | Ising programming, embedding and surrogate primitives the runtime is built on |
| `src/lac_minorminer/` | The Python orchestration seams around the native search core |
| `cpp/` | Native candidate generation, routing, validity, accounting, and the Python bindings |
| `packages/EmbedBench/` | Data generation, certification and baselines, installable on its own |
| `configs/` | Registered experiment, objective, capacity and quality protocols |
| `probes/` | Training and measurement entry points, described below |
| `scripts/` | Launchers, runtime freezing, publication workflow |
| `tests/` | Unit, integration and trust-boundary checks |
| `documents/` | The model and architecture contracts the code implements |

EmbedBench and the learned runtime share this repository but keep a one-way boundary. EmbedBench
publishes authenticated records; the runtime consumes them through `src/isingfold/rl/data/` and
must not import generator or evaluator-target code.

## The probes

`probes/` is the working layer: each file is one question, stated in its docstring, with the
answer written to a log rather than to the repository.

Training:

- `train_ladder.py` runs masked PPO with a KL-target learning rate, within-lineage advantage
  baselines and a fixed development subset for the learning curve.
- `train_bestof.py` trains on the policy's own best rollouts instead, an AlphaGo-Zero style loop
  with a replay window.
- `train_value.py` fits the critic supervised, with an option to centre the target per lineage,
  which is what separates ranking candidates for one problem from guessing which problem is easy.

Both trainers take `--initializer minorminer` to protect the episode with the standard tool's
embedding rather than the self-contained router, so the policy learns to improve on minorminer
instead of to replace it.

Measurement:

- `external_baseline.py` puts stock minorminer into the environment as an initializer, so it is
  scored by the identical evaluator path as the policy, at a matched number of attempts and with
  wall clock reported separately.
- `frontier2.py` measures the cost-utility frontier with selection and assessment separated: each
  arm spends its budget of measurement blocks to choose an embedding, and the chosen embedding is
  then frozen and re-measured on independent reads that took no part in choosing it. It also
  reports a noise control, which is how large a difference the instrument can invent on its own.
- `scorer_baseline.py` asks whether a plain supervised model, given the same candidates, can pick
  a good embedding without measuring it.
- `check_fast_path.py` proves that the optional `ISINGFOLD_FAST_INTERNAL_ASSERTS` flag changes
  only the clock, by fingerprinting every episode with and without it.
- `gen_corpus.py` generates a corpus with the source tree pinned.

`frontier.py` is the earlier version of `frontier2.py` and is kept only so the difference between
them can be read: it reported the largest of the same scores it used to select, which is a
maximum over noise and cannot lose.

## Running

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e . -e packages/EmbedBench
export ISINGFOLD_SRC=$PWD/src PYTHONPATH=$PWD/src
python -u probes/gen_corpus.py --out runs/corpus --host chimera --host-size 4 \
  --variables 16 --chain-size 3 --qubit-cap 120 --instances 600 --seed 1
python -u probes/train_bestof.py --corpus runs/corpus --family if-core --out runs/bestof \
  --initializer minorminer --rounds 60 --k 8 --eval-every 5
python -u probes/frontier2.py --corpus runs/corpus --split test --lineages 60 \
  --checkpoints if-core=runs/bestof/policy.pt
```

Every probe pins `ISINGFOLD_SRC` and strips the editable-install finder from `sys.meta_path`
before importing anything, because an editable install elsewhere on the machine will otherwise
win and the run will silently execute different code.

## Measurement rules this code enforces

Three of them exist because breaking them produced a published-looking number that measured
something else.

**Select with one budget, assess with another.** An arm that reports the largest of the scores it
used to choose cannot lose, however bad its candidates are. `frontier2.py` freezes the chosen
embedding and measures it again on reads that took no part in the choice.

**Report the denominator.** An arm that fails to return an embedding has failed on that lineage.
Both the solved-only mean and the mean with failures scored zero are printed, with the count.

**Never let a maximum hide a direction.** Reporting the better of a state and where it started
turns a monotone degradation into a monotone improvement; the same policy and the same rounds
gave +0.024 under the clipped statistic and -0.085 under the honest one.

`compile_program` sorts every label before it writes a coefficient, for a related reason: a
frozenset of qubit labels iterates in per-process hash order, and that order became the sampler's
variable order, so one embedding measured in two processes returned two different utilities with
the seed pinned and the program digest identical.
