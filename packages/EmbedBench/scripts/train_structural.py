#!/usr/bin/env python3
"""Architecture comparison on the structural corpus.
python3 scripts/train_structural.py FILES --splits SPLITS --arch gin \
    --seeds 0,1,2 --epochs 15 --out runs/arch/gin.json
"""

import argparse
import json
import time
from pathlib import Path

from embedbench.models_structural import (
    build_arch,
    encode,
    evaluate,
    load_records,
    split_by_instance,
    train,
)
from training_artifacts import checkpoint_path, cpu_state_dict, runtime_provenance
from training_device import resolve_device, seed_device
from training_splits import data_provenance, load_split_records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--arch", default="mpnn")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--holdout-topology", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--splits",
        default=None,
        help="release splits.json; fixes train/val/test independently of the training seed",
    )
    ap.add_argument(
        "--evaluate-test",
        action="store_true",
        help="evaluate the fixed test split after the architecture and hyperparameters are locked",
    )
    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="training device; auto uses CUDA when the allocated runtime exposes it",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--save",
        default=None,
        help="checkpoint prefix; writes PREFIX_ARCH_sSEED.pt for every seed",
    )
    a = ap.parse_args()
    if a.splits and a.holdout_topology:
        ap.error("--splits and --holdout-topology are mutually exclusive")
    try:
        device = resolve_device(a.device)
    except ValueError as error:
        ap.error(str(error))
    resolved_device = str(device)
    runtime = runtime_provenance(device)
    provenance = data_provenance(a.files, a.splits)
    manifest_provenance = provenance["split_manifest"]
    split_sha256 = (
        manifest_provenance["sha256"] if manifest_provenance is not None else None
    )
    fixed = None
    if a.splits:
        split_records = load_split_records(a.files, a.splits, limit=a.limit)
        fixed = tuple([encode(r) for r in records] for records in split_records)
        if not all(fixed):
            raise ValueError("fixed release split produced an empty train, val, or test partition")
        split_counts = tuple(map(len, fixed))
        print(f"{sum(split_counts)} samples; arch={a.arch}; fixed split counts={split_counts}")
    else:
        recs = load_records(a.files)[: a.limit]
        encs = [encode(r) for r in recs]
        print(f"{len(encs)} samples; arch={a.arch}")
    results = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        if fixed is not None:
            tr, va, te = fixed
        elif a.holdout_topology:
            tr = [e for e in encs if e.topology != a.holdout_topology]
            te = [e for e in encs if e.topology == a.holdout_topology]
        else:
            tr, te = split_by_instance(encs, a.test_frac, seed)
        if fixed is None:
            tr, va = split_by_instance(tr, 0.1, seed)
        seed_device(seed, device)
        t0 = time.time()
        model = build_arch(a.arch, a.hidden, a.layers, a.heads)
        n_params = sum(p.numel() for p in model.parameters())
        ep = train(
            model,
            tr,
            va,
            epochs=a.epochs,
            seed=seed,
            device=device,
            log=lambda s, run_seed=seed: print(f"[{a.arch} s{run_seed}] {s}"),
        )
        validation = evaluate(model, va, device=device)
        test_evaluated = fixed is None or a.evaluate_test
        ev = evaluate(model, te, device=device) if test_evaluated else None
        by = (
            {
                t: evaluate(model, [e for e in te if e.topology == t], device=device)
                for t in sorted({e.topology for e in te})
            }
            if test_evaluated
            else {}
        )
        row = {
            "arch": a.arch,
            "seed": seed,
            "device": resolved_device,
            "params": n_params,
            "best_epoch": ep,
            "n_train": len(tr),
            "n_val": len(va),
            "n_test": len(te),
            "validation": validation,
            "test_evaluated": test_evaluated,
            "test": ev,
            "by_topology": by,
            "seconds": round(time.time() - t0, 1),
        }
        print(json.dumps(row))
        results.append(row)
        if a.save:
            import torch

            torch.save(
                {
                    "state": cpu_state_dict(model),
                    "meta": {
                        "artifact_schema": "embedbench.structural-scorer",
                        "artifact_schema_version": 1,
                        "arch": a.arch,
                        "hidden": a.hidden,
                        "layers": a.layers,
                        "heads": a.heads,
                        "seed": seed,
                        "device": resolved_device,
                        "splits": a.splits,
                        "split_sha256": split_sha256,
                        "data_provenance": provenance,
                        "runtime_provenance": runtime,
                    },
                },
                checkpoint_path(a.save, a.arch, seed),
            )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        json.dumps(
            {
                "args": vars(a),
                "device": resolved_device,
                "split_sha256": split_sha256,
                "data_provenance": provenance,
                "runtime_provenance": runtime,
                "results": results,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
