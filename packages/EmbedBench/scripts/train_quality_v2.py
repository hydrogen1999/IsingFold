#!/usr/bin/env python3
"""Train the additive Quality Value V2 model on fixed, leakage-safe splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from training_artifacts import checkpoint_path, cpu_state_dict, runtime_provenance
from training_device import resolve_device, seed_device
from training_splits import (
    data_provenance,
    load_split_records,
    quality_problem_digest,
    quality_problem_id,
)


@dataclass(frozen=True)
class _Example:
    encoded: object
    model_input: object
    exact: object


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deployment_host_contract(files: list[str]) -> dict[str, object]:
    """Prove that pristine host reconstruction matches each generator configuration."""

    source_manifests = []
    for raw_file in files:
        corpus = Path(raw_file)
        manifest = Path(f"{corpus}.manifest.json")
        if not manifest.is_file():
            raise ValueError(
                f"--deploy-view requires generator manifest {manifest}; host defects "
                "cannot be reconstructed safely without it"
            )
        try:
            document = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid generator manifest {manifest}") from error
        if not isinstance(document, dict) or not isinstance(document.get("config"), dict):
            raise ValueError(f"generator manifest {manifest} has no config object")
        corpus_sha256 = _sha256(corpus)
        if document.get("sha256") != corpus_sha256:
            raise ValueError(f"generator manifest {manifest} does not match its corpus")
        config = document["config"]
        defect_values = {}
        for field in ("defect_qubits", "defect_couplers"):
            value = config.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"generator manifest {manifest} has invalid {field}")
            defect_values[field] = float(value)
        if any(value != 0.0 for value in defect_values.values()):
            raise ValueError(
                "--deploy-view pristine host reconstruction requires a defect-free "
                f"generator manifest; {manifest} declares {defect_values}"
            )
        source_manifests.append(
            {
                "file": corpus.name,
                "corpus_sha256": corpus_sha256,
                "manifest_sha256": _sha256(manifest),
                **defect_values,
            }
        )
    return {
        "mode": "pristine_topology_reconstruction",
        "source_manifests": source_manifests,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train quality-first candidate value heads; the fixed test partition remains "
            "locked and is never evaluated by this development entry point."
        )
    )
    parser.add_argument("files", nargs="+")
    parser.add_argument(
        "--arch",
        default="mpnn",
        choices=("mpnn", "gin", "gatv2", "gps", "hetero"),
    )
    parser.add_argument(
        "--objective-variant",
        default="full",
        choices=("p_only", "p_connectivity", "p_robustness", "full"),
    )
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tol", type=float, default=0.05)
    parser.add_argument(
        "--rank-margin",
        type=float,
        default=0.05,
        help="minimum p_solve gap for a pair whose two labels are stage 2",
    )
    parser.add_argument(
        "--stage1-rank-margin",
        type=float,
        default=0.10,
        help="minimum gap when either p_solve label is noisier stage 1",
    )
    parser.add_argument(
        "--lcb-z",
        type=float,
        default=1.0,
        help="lower-confidence-bound width; reported as secondary until calibrated",
    )
    parser.add_argument("--stage1-weight", type=float, default=0.25)
    parser.add_argument("--lambda-connectivity", type=float, default=0.1)
    parser.add_argument("--lambda-robustness", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--neighbour-feats", action="store_true")
    parser.add_argument(
        "--deploy-view",
        action="store_true",
        help="rebuild train/validation windows exactly as the deployment seam does",
    )
    parser.add_argument(
        "--deploy-max-free",
        type=int,
        default=8,
        help="maximum deterministic two-hop free qubits added by deploy-view",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--splits",
        required=True,
        help="schema-v2 fixed split manifest; test membership is loaded but kept locked",
    )
    parser.add_argument(
        "--evaluation-support",
        choices=("full",),
        default="full",
        help="V2 architecture selection always evaluates every presented candidate",
    )
    parser.add_argument("--save", default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.hidden <= 0 or args.layers <= 0 or args.heads <= 0:
        parser.error("--hidden, --layers, and --heads must be positive")
    if args.arch in {"gatv2", "gps", "hetero"} and args.hidden % args.heads:
        parser.error("--hidden must be divisible by --heads for attention backbones")
    if args.arch == "hetero" and args.neighbour_feats:
        parser.error("--neighbour-feats is not applicable to --arch hetero")
    if args.deploy_max_free < 0:
        parser.error("--deploy-max-free must be non-negative")
    if args.lr <= 0.0:
        parser.error("--lr must be positive")
    if not math.isfinite(args.lcb_z) or args.lcb_z < 0.0:
        parser.error("--lcb-z must be finite and non-negative")
    if (
        not math.isfinite(args.rank_margin)
        or args.rank_margin < 0.0
        or not math.isfinite(args.stage1_rank_margin)
        or args.stage1_rank_margin < args.rank_margin
    ):
        parser.error("ranking margins must be finite and stage-1 margin must be at least stage-2")
    seeds = args.seeds.split(",")
    try:
        args.seed_values = [int(seed) for seed in seeds]
    except ValueError:
        parser.error("--seeds must be a comma-separated list of integers")
    if not seeds or len(args.seed_values) != len(set(args.seed_values)):
        parser.error("--seeds must be a non-empty list without duplicates")
    return args


def _variant_weights(args: argparse.Namespace) -> tuple[float, float]:
    connectivity = (
        args.lambda_connectivity if args.objective_variant in {"p_connectivity", "full"} else 0.0
    )
    robustness = (
        args.lambda_robustness if args.objective_variant in {"p_robustness", "full"} else 0.0
    )
    return connectivity, robustness


def main() -> None:
    args = _parse_args()

    import torch
    from embedbench.hamiltonian_context import hamiltonian_context_contract
    from embedbench.models_chain import (
        deployment_view,
        encode_chain,
        evaluate_quality_support,
    )
    from embedbench.models_hetero import encode_hetero, encode_hetero_input
    from embedbench.models_quality_v2 import (
        build_quality_v2_model,
        derive_exact_candidate_metrics,
        evaluate_quality_v2_budgets,
        quality_v2_loss,
        save_quality_v2_model,
        tensors_from_chain,
    )
    from embedbench.structural import host_graph

    try:
        device = resolve_device(args.device)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    provenance = data_provenance(args.files, args.splits)
    runtime = runtime_provenance(device)
    manifest_provenance = provenance["split_manifest"]
    split_sha256 = manifest_provenance["sha256"]
    deployment_host_contract = (
        _deployment_host_contract(args.files)
        if args.deploy_view
        else {"mode": "not_applied", "source_manifests": []}
    )
    records = load_split_records(
        args.files,
        args.splits,
        group_key=quality_problem_id,
        record_group_key=quality_problem_digest,
        limit=args.limit,
    )
    if not all(records):
        raise ValueError("fixed split produced an empty train, validation, or test partition")

    hosts = {}

    def prepare(partition: list[dict]) -> list[_Example]:
        examples = []
        for record in partition:
            prepared = record
            if args.deploy_view:
                key = (record["topology"], record["size"])
                if key not in hosts:
                    hosts[key] = host_graph(*key)
                prepared = deployment_view(
                    record,
                    hosts[key],
                    max_free=args.deploy_max_free,
                )
            exact = derive_exact_candidate_metrics(prepared)
            if not bool(exact.feasible.all()):
                raise ValueError(
                    f"quality record {record.get('instance_id')!r} contains an exactly "
                    "infeasible candidate"
                )
            if args.arch == "hetero":
                encoded = encode_hetero(prepared)
                model_input = encode_hetero_input(
                    prepared,
                    require_hamiltonian_context=True,
                )
            else:
                encoded = encode_chain(
                    prepared,
                    use_neighbour_feats=args.neighbour_feats,
                    require_hamiltonian_context=True,
                )
                model_input = encoded
            examples.append(_Example(encoded=encoded, model_input=model_input, exact=exact))
        return examples

    train_examples = prepare(records.train)
    validation_examples = prepare(records.val)
    test_record_count = len(records.test)
    connectivity_weight, robustness_weight = _variant_weights(args)
    preprocessing = {
        "deploy_view": args.deploy_view,
        "deploy_max_free": args.deploy_max_free,
        "neighbour_feats": args.neighbour_feats,
        "encoder": "heterogeneous" if args.arch == "hetero" else "chain",
        "hamiltonian_context": hamiltonian_context_contract(),
    }
    training_scope = {
        "limit": args.limit,
        "full_fixed_train_validation": args.limit is None,
    }

    def forward(model, example: _Example):
        if args.arch == "hetero":
            return model(example.model_input)
        return model(*tensors_from_chain(example.model_input, device))

    def evaluate(model, examples: list[_Example]) -> dict[str, object]:
        model.eval()
        with torch.no_grad():
            outputs = [forward(model, example) for example in examples]
        output_by_identity = {
            id(example.encoded): output for example, output in zip(examples, outputs, strict=True)
        }

        def score(encoded) -> list[float]:
            return output_by_identity[id(encoded)].quality_logit.detach().cpu().tolist()

        full_support = evaluate_quality_support(
            score,
            [example.encoded for example in examples],
            support=args.evaluation_support,
            mode="development",
            tol=args.tol,
            random_seed=0,
        )
        full_support["budget_sweep"] = evaluate_quality_v2_budgets(
            outputs,
            [example.encoded for example in examples],
            [example.exact for example in examples],
            lcb_z=args.lcb_z,
            random_seed=0,
        )
        return full_support

    results = []
    for seed in args.seed_values:
        training_run_id = uuid.uuid4().hex
        seed_device(seed, device)
        rng = random.Random(seed)
        model = build_quality_v2_model(
            arch=args.arch,
            hidden=args.hidden,
            layers=args.layers,
            heads=args.heads,
            neighbour_feats=args.neighbour_feats,
        ).to(device)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        best = None
        started = time.time()
        for epoch in range(args.epochs):
            model.train()
            order = list(range(len(train_examples)))
            rng.shuffle(order)
            epoch_components: dict[str, float] = {}
            update_count = 0
            for start in range(0, len(order), 16):
                optimizer.zero_grad()
                batch_total = None
                batch_components: dict[str, float] = {}
                used = 0
                for index in order[start : start + 16]:
                    example = train_examples[index]
                    encoded = example.encoded
                    exact = example.exact
                    output = forward(model, example)
                    losses = quality_v2_loss(
                        output,
                        p_solve=torch.as_tensor(encoded.p, dtype=torch.float32, device=device),
                        stage=torch.as_tensor(encoded.stage, dtype=torch.long, device=device),
                        future_capacity=torch.as_tensor(
                            exact.residual_connectivity_proxy,
                            dtype=torch.float32,
                            device=device,
                        ),
                        robustness=torch.as_tensor(
                            exact.robustness, dtype=torch.float32, device=device
                        ),
                        rank_margin=args.rank_margin,
                        stage1_rank_margin=args.stage1_rank_margin,
                        lambda_capacity=connectivity_weight,
                        lambda_robustness=robustness_weight,
                        stage1_weight=args.stage1_weight,
                    )
                    batch_total = (
                        losses["loss"] if batch_total is None else batch_total + losses["loss"]
                    )
                    for name, value in losses.items():
                        batch_components[name] = batch_components.get(name, 0.0) + float(
                            value.detach()
                        )
                    used += 1
                if used:
                    (batch_total / used).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    update_count += 1
                    for name, value in batch_components.items():
                        epoch_components[name] = epoch_components.get(name, 0.0) + value / used

            validation = evaluate(model, validation_examples)
            primary_metric = validation["budget_sweep"]["primary_metric"]
            if primary_metric is None:
                coverage = validation["budget_sweep"]["primary_record_coverage"]
                raise RuntimeError(
                    "validation has no valid paired stock minorminer reference for "
                    f"B/Q_MM checkpoint selection: {coverage}"
                )
            print(
                f"[{args.arch}/{args.objective_variant} s{seed}] epoch {epoch + 1} "
                f"finite-budget regret "
                f"{primary_metric:.4f} "
                f"uncapped model {validation['model_regret']:.4f} "
                f"resource {validation['resource_regret']:.4f} "
                f"pair {validation['pair_acc']:.3f}",
                flush=True,
            )
            if best is None or primary_metric < best[0]:
                best = (
                    primary_metric,
                    epoch + 1,
                    cpu_state_dict(model),
                    {
                        name: value / max(1, update_count)
                        for name, value in epoch_components.items()
                    },
                )

        if best is None:  # guarded by positive epochs; retained as a fail-closed invariant
            raise RuntimeError("training completed without a validation checkpoint")
        model.load_state_dict(best[2])
        validation = evaluate(model, validation_examples)
        row = {
            "arch": args.arch,
            "objective_variant": args.objective_variant,
            "seed": seed,
            "training_run_id": training_run_id,
            "device": str(device),
            "hidden": args.hidden,
            "layers": args.layers,
            "heads": args.heads,
            "params": parameter_count,
            "best_epoch": best[1],
            "n_train": len(train_examples),
            "n_val": len(validation_examples),
            "n_test": test_record_count,
            "preprocessing": preprocessing,
            "deployment_host_contract": deployment_host_contract,
            "training_scope": training_scope,
            "evaluation_support": args.evaluation_support,
            "registered_budget_ratios": [1.0, 1.1, 1.25, 1.5, None],
            "primary_validation_metric": "mean_finite_budget_regret",
            "lcb_z": args.lcb_z,
            "lcb_is_secondary_until_calibrated": True,
            "ranking_thresholds": {
                "stage2_pair": args.rank_margin,
                "stage1_involved_pair": args.stage1_rank_margin,
            },
            "selection_contract": (
                "quality_mean_or_lcb_subject_to_exact_feasibility_and_budget-v1"
            ),
            "auxiliary_heads_used_for_selection": False,
            "validation": validation,
            "test_evaluated": False,
            "test_partition_encoded": False,
            "test": None,
            "seconds": round(time.time() - started, 1),
            "loss_weights": {
                "quality_probability": 1.0,
                "within_state_rank": 0.5,
                "residual_connectivity_proxy": connectivity_weight,
                "robustness": robustness_weight,
                "terminal_qubits": 0.0,
            },
            "best_epoch_training_losses": best[3],
        }
        results.append(row)
        print(json.dumps(row), flush=True)

        if args.save:
            save_quality_v2_model(
                model,
                checkpoint_path(args.save, args.arch, seed),
                metadata={
                    "objective_variant": args.objective_variant,
                    "seed": seed,
                    "training_run_id": training_run_id,
                    "best_epoch": best[1],
                    "selected_validation_metric": validation["budget_sweep"]["primary_metric"],
                    "device": str(device),
                    "splits": args.splits,
                    "split_sha256": split_sha256,
                    "data_provenance": provenance,
                    "runtime_provenance": runtime,
                    "evaluation_support": args.evaluation_support,
                    "registered_budget_ratios": [1.0, 1.1, 1.25, 1.5, None],
                    "primary_validation_metric": "mean_finite_budget_regret",
                    "lcb_z": args.lcb_z,
                    "lcb_is_secondary_until_calibrated": True,
                    "ranking_thresholds": row["ranking_thresholds"],
                    "selection_contract": (
                        "quality_mean_or_lcb_subject_to_exact_feasibility_and_budget-v1"
                    ),
                    "auxiliary_heads_used_for_selection": False,
                    "test_evaluated": False,
                    "test_partition_encoded": False,
                    "preprocessing": preprocessing,
                    "deployment_host_contract": deployment_host_contract,
                    "training_scope": training_scope,
                    "neighbour_feats": args.neighbour_feats,
                    "loss_weights": row["loss_weights"],
                },
            )

    destination = Path(args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(
            {
                "artifact_schema": "embedbench.quality-v2-training-results",
                "artifact_schema_version": 1,
                "args": {key: value for key, value in vars(args).items() if key != "seed_values"},
                "device": str(device),
                "split_sha256": split_sha256,
                "data_provenance": provenance,
                "runtime_provenance": runtime,
                "preprocessing": preprocessing,
                "deployment_host_contract": deployment_host_contract,
                "training_scope": training_scope,
                "test_locked": True,
                "test_partition_encoded": False,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
