"""Train an independent constructive RL embedder, always from empty chains.

The policy chooses placement, route/rewrite, restart and COMMIT macro-actions.
No completion solver, supplied embedding, witness budget or per-qubit penalty is
used in learned episodes. Minorminer is an optional, isolated comparison arm.
Quality is the default objective; feasibility is an explicitly separate curriculum.
"""
# Runtime source pinning intentionally precedes imports from the checked-out tree.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]
import numpy as np
import torch
from isingfold.rl.data.generate import load_instances

from candidate_features import FeatureContext, WIDTH as LEGACY_WIDTH
from constructor_features import ConstructorFeatureContext, WIDTH, FEATURE_VERSION
from constructor_learning import constructor_loss
from constructor_protocol import checkpoint_key, evaluate_search, experiment_seed, split_by_lineage
# These helpers remain importable for existing probes. The hybrid is not imported here.
from constructor_rollout import candidate_tuple as candidate_tuple
from constructor_rollout import demands_realised as demands_realised
from constructor_rollout import episode
from layout_policy import LayoutActorCritic
from train_prioritiser import Prioritiser


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--init", default="")
    ap.add_argument("--allow-unverified-init", action="store_true",
                    help="explicit legacy import; marks cumulative training provenance unverified")
    ap.add_argument("--objective", choices=("quality", "feasibility"), default="quality")
    ap.add_argument("--actor", choices=("contextual", "local"), default="contextual")
    ap.add_argument("--features", choices=("construction", "legacy"), default="construction")
    ap.add_argument("--value-baseline", choices=("value", "loo"), default="value")
    ap.add_argument("--iterations", type=int, default=200)
    ap.add_argument("--instances-per-iteration", type=int, default=4)
    ap.add_argument("--episodes-per-instance", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=3e-4)
    ap.add_argument("--entropy-coef", type=float, default=0.01)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--shaping-coef", type=float, default=0.0,
                    help="gamma=1 potential shaping with terminal potential zero")
    ap.add_argument("--max-steps", type=int, default=3000)
    ap.add_argument("--deadline", type=float, default=300.0,
                    help="validation deployment deadline including proposals and selection")
    ap.add_argument("--episode-seconds", type=float, default=30.0)
    ap.add_argument("--qubit-cap", type=int, default=0, help="0 means full active host; never from witness")
    ap.add_argument("--holdout-fraction", type=float, default=0.25)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reward-reads", type=int, default=256)
    ap.add_argument("--selection-reads", type=int, default=256)
    ap.add_argument("--assessment-reads", type=int, default=256)
    ap.add_argument("--select-cap", type=int, default=6)
    ap.add_argument("--comparison", choices=("minorminer", "none"), default="minorminer")
    ap.add_argument("--baseline-router-seconds", type=float, default=2.0)
    ap.add_argument("--baseline-tries", type=int, default=2)
    a = ap.parse_args()
    positive = (a.instances_per_iteration, a.episodes_per_instance, a.temperature,
                a.learning_rate, a.max_steps, a.deadline, a.episode_seconds, a.eval_every,
                a.width, a.reward_reads, a.selection_reads, a.assessment_reads, a.select_cap,
                a.baseline_router_seconds, a.baseline_tries)
    nonnegative = (a.iterations, a.qubit_cap, a.entropy_coef, a.value_coef, a.shaping_coef)
    if not all(np.isfinite(x) and x > 0 for x in positive):
        ap.error("sizes, temperatures, learning rate, deadlines and read caps must be positive and finite")
    if not all(np.isfinite(x) and x >= 0 for x in nonnegative):
        ap.error("iterations, budgets and loss coefficients must be finite and nonnegative")
    if a.value_baseline == "value" and a.actor != "contextual":
        ap.error("a value baseline requires the contextual actor")
    if a.value_baseline == "loo" and a.episodes_per_instance < 2:
        ap.error("leave-one-out requires at least two episodes per instance")
    tasks = load_instances(a.corpus)
    try:
        train_tasks, eval_tasks = split_by_lineage(tasks, a.holdout_fraction, a.seed)
    except ValueError as exc:
        ap.error(str(exc))
    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    in_dim = WIDTH if a.features == "construction" else LEGACY_WIDTH
    spec = {"actor": a.actor, "feature_version": FEATURE_VERSION if a.features == "construction"
            else "candidate-v1", "width": a.width, "in_dim": in_dim}
    model = (LayoutActorCritic(a.width, in_dim=in_dim) if a.actor == "contextual"
             else Prioritiser(a.width, in_dim=in_dim))
    current_train_lineages = {t.lineage or t.name for t in train_tasks}
    validation_lineages = {t.lineage or t.name for t in eval_tasks}
    prior_train_lineages = set()
    init_provenance_verified = True
    if a.init:
        blob = torch.load(a.init, map_location="cpu", weights_only=True)
        old_spec = blob.get("model_spec", {"actor": "local", "feature_version": "candidate-v1",
                                          "width": blob["width"], "in_dim": LEGACY_WIDTH})
        if old_spec != spec:
            ap.error("checkpoint model/feature schema differs; match actor, features and width")
        # An initialization can already have trained on this run's held-out tasks.
        # Retain all known ancestors' training lineages, not just the latest stage.
        known_lineages = blob.get("training_lineages", blob.get("train_lineages"))
        if known_lineages is not None:
            if (not isinstance(known_lineages, (list, tuple))
                    or any(not isinstance(x, str) or not x for x in known_lineages)):
                ap.error("checkpoint training lineage provenance is malformed")
            prior_train_lineages = set(known_lineages)
        init_provenance_verified = (known_lineages is not None
                                    and blob.get("lineage_provenance_verified") is True)
        overlap = prior_train_lineages & validation_lineages
        if overlap:
            ap.error("initial checkpoint trained on current validation lineages: "
                     + ", ".join(sorted(overlap)))
        if not init_provenance_verified and not a.allow_unverified_init:
            ap.error("initial checkpoint has unverified training lineage provenance; "
                     "use --allow-unverified-init only for a declared legacy import")
        model.load_state_dict(blob["state"])
    lineage_metadata = {
        "train_lineages": sorted(current_train_lineages),
        "training_lineages": sorted(current_train_lineages | prior_train_lineages),
        "validation_lineages": sorted(validation_lineages),
        "current_split_verified": True,
        "init_lineage_provenance_verified": init_provenance_verified,
        "lineage_provenance_verified": init_provenance_verified,
    }
    opt = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    contexts = {}

    def budget(task):
        cap = task.host.number_of_nodes()
        return min(cap, a.qubit_cap) if a.qubit_cap else cap

    if any(budget(t) < t.logical.number_of_nodes() for t in tasks):
        ap.error("declared budget cannot place even one qubit per logical variable")

    def fc_for(task):
        if task.name not in contexts:
            cls = ConstructorFeatureContext if a.features == "construction" else FeatureContext
            contexts[task.name] = cls(task, budget(task))
        return contexts[task.name]

    def rollout(task, seed, seconds, evaluate_reward):
        # Every call creates a fresh construction environment. No witness or initializer.
        return episode(task, model, fc_for(task), a.temperature, a.max_steps,
                       np.random.default_rng(seed), seconds, train=True,
                       qubit_cap=budget(task), objective=a.objective,
                       reward_reads=a.reward_reads, shaping_coef=a.shaping_coef,
                       evaluate_reward=evaluate_reward)

    print(json.dumps({"protocol": "independent-constructor-v1", "config": vars(a),
                      "model_spec": spec, "train": len(train_tasks), "validation": len(eval_tasks),
                      **lineage_metadata,
                      "inference_initial_state": "empty", "completion_solver": None,
                      "allow_satisfied_growth": True,
                      "context_version_suffix": "-independent-constructor-v1",
                      "qubit_objective_penalty": 0, "split_unit": "lineage",
                      "comparison_only_solver": a.comparison,
                      "learning_algorithm": "single-update-Monte-Carlo-actor-critic-gamma1"}), flush=True)

    def evaluate(tag):
        model.eval()
        records = []
        for task in eval_tasks:
            def learned(seed, seconds_left):
                with torch.no_grad(), torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    result = rollout(task, seed, min(seconds_left, a.episode_seconds), False)
                return result["terminal"] if result["valid"] else None

            def baseline(seed, seconds_left):
                from constructor_baseline import baseline_proposal
                return baseline_proposal(task, budget(task), seed, seconds_left,
                                         a.baseline_tries, a.baseline_router_seconds)

            # Alternate arm order to reduce systematic first/second timing effects.
            arms = [("policy", learned)]
            if a.comparison == "minorminer":
                arms.append(("minorminer", baseline))
            if experiment_seed(a.seed, "arm-order", task.name) % 2:
                arms.reverse()
            for name, proposer in arms:
                r = evaluate_search(task, proposer, deadline=a.deadline, select_cap=a.select_cap,
                                    selection_reads=a.selection_reads, assessment_reads=a.assessment_reads,
                                    objective=a.objective,
                                    seed=experiment_seed(a.seed, "validation", task.name))
                records.append({"instance": task.name, "lineage": task.lineage or task.name,
                                "arm": name, **r})
        model.train()
        score = checkpoint_key([r for r in records if r["arm"] == "policy"], a.objective)
        print(json.dumps({"evaluation": tag, "role": "validation", "arms": records,
                          "checkpoint_key": score}), flush=True)
        return score

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    def save(path, score=None):
        torch.save({"state": model.state_dict(), "model_spec": spec, "width": a.width,
                    "config": vars(a), "validation_score": score,
                    **lineage_metadata,
                    "allow_satisfied_growth": True,
                    "context_version_suffix": "-independent-constructor-v1",
                    "protocol": "independent-constructor-v1"}, path)

    best = evaluate("init")
    save(a.out, best)
    for iteration in range(a.iterations):
        indices = rng.choice(len(train_tasks), min(a.instances_per_iteration, len(train_tasks)), replace=False)
        opt.zero_grad()
        metrics, episodes_all, updates = [], [], 0
        for index in indices:
            task = train_tasks[int(index)]
            episodes = [rollout(task, experiment_seed(a.seed, "train", task.name, iteration, e),
                                a.episode_seconds, True) for e in range(a.episodes_per_instance)]
            loss, metric = constructor_loss(episodes, baseline=a.value_baseline,
                                            entropy_coef=a.entropy_coef, value_coef=a.value_coef)
            if loss is not None:
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite constructor loss")
                (loss / len(indices)).backward()
                updates += 1
            metrics.append(metric)
            episodes_all.extend(episodes)
        norm = 0.0
        if updates:
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True))
            opt.step()
        merged = {k: float(np.mean([m[k] for m in metrics if m[k] is not None]))
                  if any(m[k] is not None for m in metrics) else None for k in metrics[0]}
        print(json.dumps({"training_iteration": iteration, "objective": a.objective,
                          "has_measured_quality_signal": any(e.get("residual") is not None for e in episodes_all),
                          "validity": float(np.mean([e["valid"] for e in episodes_all])),
                          "measurement_coverage": float(np.mean([e.get("residual") is not None for e in episodes_all])),
                          "mean_demands": float(np.mean([e["frac"] for e in episodes_all])),
                          "mean_steps": float(np.mean([e["steps"] for e in episodes_all])),
                          "gradient_norm_before_clip": norm, **merged}), flush=True)
        if (iteration + 1) % a.eval_every == 0:
            score = evaluate(f"iter {iteration}")
            if score > best:
                best = score
                save(a.out, best)
    save(a.out + ".last")
    print("CONSTRUCTOR RL DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
