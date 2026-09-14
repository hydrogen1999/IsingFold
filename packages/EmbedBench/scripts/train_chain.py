#!/usr/bin/env python3
"""Chain-scorer architecture comparison: mpnn / gin / gatv2 / gps / hetero, pairwise or listwise.
python3 scripts/train_chain.py FILES --splits SPLITS --arch hetero --loss listwise \
    --seeds 0,1,2 --out runs/arch_chain/hetero.json
"""

import argparse
import json
import random
import time
from functools import partial
from pathlib import Path

import numpy as np
from training_artifacts import checkpoint_path, cpu_state_dict, runtime_provenance
from training_device import resolve_device, seed_device
from training_splits import (
    SplitRecords,
    data_provenance,
    file_sha256,
    load_split_records,
    quality_problem_digest,
    quality_problem_id,
)


def _load_records_with_corpus_file(paths, load_records):
    records = []
    for raw_path in paths:
        corpus_file = Path(raw_path).name
        for record in load_records([raw_path]):
            records.append({**record, "_file": corpus_file})
    return records


def _attach_split_corpus_files(records, paths, splits_path, limit):
    """Restore corpus identity discarded by the generic split loader.

    ``load_split_records`` validates the manifest and emits each partition in input-file
    order.  Replaying only that stable ordering lets independent audit labels retain the
    same ``(file, instance_id, focus)`` identity used by ``rescore_quality.py``.
    """
    document = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    split_tables = document["splits"]
    sources = {name: [] for name in ("train", "val", "test")}
    loaded = 0
    for raw_path in paths:
        path = Path(raw_path)
        table = split_tables[path.name]
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                split = table[str(record["instance_id"])]
                if limit is None or loaded < limit:
                    sources[split].append(path.name)
                    loaded += 1

    attached = []
    for split, partition in zip(("train", "val", "test"), records, strict=True):
        if len(partition) != len(sources[split]):
            raise ValueError("corpus provenance does not align with the fixed split")
        attached.append(
            [
                {**record, "_file": corpus_file}
                for record, corpus_file in zip(
                    partition,
                    sources[split],
                    strict=True,
                )
            ]
        )
    return SplitRecords(*attached)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--arch", default="mpnn")
    ap.add_argument("--loss", default="pairwise", choices=["pairwise", "listwise"])
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument(
        "--evaluation-support",
        choices=("full", "legacy-reliable"),
        default="full",
        help=(
            "candidate support used by every selector; full is label-independent, "
            "legacy-reliable reproduces the V1 stage-2 shortlist diagnostic"
        ),
    )
    ap.add_argument(
        "--evaluation-mode",
        choices=("development", "audit", "paper"),
        default="development",
        help=(
            "locked-test claim mode; validation/model selection always uses provisional "
            "development labels, while audit and paper require complete independent "
            "labels on the evaluated test support"
        ),
    )
    ap.add_argument(
        "--audit-labels",
        nargs="+",
        default=[],
        metavar="JSONL",
        help="independent candidate-aligned high-read JSONL files",
    )
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--holdout-source", default=None)
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
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--save", default=None)
    ap.add_argument(
        "--deploy-view",
        action="store_true",
        help=(
            "rebuild every record's window the way the deployment seam does "
            "(train/deploy parity, L-115 parity check)"
        ),
    )
    ap.add_argument(
        "--neighbour-feats",
        action="store_true",
        help=(
            "append the two neighbour features (mean degree, mean chain size) "
            "to the candidate features"
        ),
    )
    ap.add_argument("--length-balance", action="store_true")
    ap.add_argument("--stage2-only", action="store_true")
    ap.add_argument("--no-length-feats", action="store_true")
    ap.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="training device; auto uses CUDA when the allocated runtime exposes it",
    )
    a = ap.parse_args()
    if a.splits and a.holdout_source:
        ap.error("--splits and --holdout-source are mutually exclusive")
    if a.evaluation_mode in {"audit", "paper"} and not a.splits:
        ap.error(
            f"--evaluation-mode {a.evaluation_mode} requires --splits and "
            "--evaluate-test; claim evaluation must use the registered fixed test split"
        )
    if a.evaluation_mode in {"audit", "paper"} and not a.evaluate_test:
        ap.error(
            f"--evaluation-mode {a.evaluation_mode} requires --evaluate-test with "
            "a fixed split; validation is reserved for development model selection"
        )
    if (
        a.evaluation_mode in {"audit", "paper"}
        and a.evaluation_support != "full"
    ):
        ap.error(
            f"--evaluation-mode {a.evaluation_mode} requires full candidate support; "
            "legacy-reliable is label-derived"
        )
    import torch
    from embedbench.models_chain import (
        build_chain_arch,
        encode_chain,
        evaluate_quality_support,
        load_chain_records,
        load_independent_audit_scores,
        pair_loss,
        quality_audit_context,
        quality_audit_key,
        split_by_instance,
        verify_independent_audit_scores,
    )
    from embedbench.models_hetero import build_hetero, encode_hetero, listwise_loss

    try:
        device = resolve_device(a.device)
    except ValueError as error:
        ap.error(str(error))
    resolved_device = str(device)
    runtime = runtime_provenance(device)
    provenance = data_provenance(a.files, a.splits)
    corpus_sha256 = {
        entry["file"]: entry["sha256"] for entry in provenance["corpus_inputs"]
    }
    manifest_provenance = provenance["split_manifest"]
    split_sha256 = (
        manifest_provenance["sha256"] if manifest_provenance is not None else None
    )
    audit_lookup = load_independent_audit_scores(a.audit_labels)
    audit_provenance = [
        {"file": Path(path).name, "sha256": file_sha256(path)}
        for path in a.audit_labels
    ]
    claim_evaluation_enabled = not bool(a.splits) or a.evaluate_test
    evaluation_protocol = {
        "support": a.evaluation_support,
        "selection_split": "validation",
        "selection_mode": "development",
        "claim_split": "test",
        "claim_mode": a.evaluation_mode,
        "claim_evaluation_enabled": claim_evaluation_enabled,
        "development_label_fidelity": "mixed-stage-release",
        "development_metrics_provisional": True,
        "audit_labels": audit_provenance,
        "paper_mode_requires_complete_audit_support": True,
        "independent_audit_binding": [
            "corpus_basename",
            "corpus_sha256",
            "instance_id",
            "focus",
            "candidate_signature",
        ],
        "independent_audit_provenance_required": [
            "reads_per_strength",
            "num_sweeps",
            "base_seed",
            "seed_schedule",
            "strength_schedule",
            "aggregation",
        ],
    }

    fixed_records = None
    if a.splits:
        fixed_records = load_split_records(
            a.files,
            a.splits,
            group_key=quality_problem_id,
            record_group_key=quality_problem_digest,
            limit=a.limit,
        )
        fixed_records = _attach_split_corpus_files(
            fixed_records,
            a.files,
            a.splits,
            a.limit,
        )
    else:
        recs = _load_records_with_corpus_file(a.files, load_chain_records)[: a.limit]
    if a.deploy_view:
        from embedbench.models_chain import deployment_view
        from embedbench.structural import host_graph

        _hosts = {}

        def _h(r):
            k = (r["topology"], r["size"])
            if k not in _hosts:
                _hosts[k] = host_graph(*k)
            return _hosts[k]

        if fixed_records is not None:
            fixed_records = type(fixed_records)(
                *([deployment_view(r, _h(r)) for r in records] for records in fixed_records)
            )
        else:
            recs = [deployment_view(r, _h(r)) for r in recs]
    hetero = a.arch == "hetero"
    if hetero and a.neighbour_feats:
        ap.error("--neighbour-feats is not applicable to --arch hetero")
    if hetero and a.no_length_feats:
        ap.error("--no-length-feats is not applicable to --arch hetero")
    encoder = (
        encode_hetero
        if hetero
        else partial(
            encode_chain,
            use_neighbour_feats=a.neighbour_feats,
            drop_length_feats=a.no_length_feats,
        )
    )
    audit_by_example = {}

    def encode_record(record):
        example = encoder(record)
        strict_claim = a.evaluation_mode in {"audit", "paper"}
        record_key = quality_audit_key(record) if audit_lookup or strict_claim else None
        evidence = audit_lookup.get(record_key) if record_key is not None else None
        if strict_claim:
            corpus_file = record_key[0]
            example.audit_context = quality_audit_context(
                record,
                corpus_sha256[corpus_file],
            )
        if evidence is not None and strict_claim:
            evidence = verify_independent_audit_scores(
                record,
                evidence,
                corpus_sha256[corpus_file],
            )
        audit_by_example[id(example)] = evidence
        return example

    fixed = None
    if fixed_records is not None:
        fixed = tuple([encode_record(r) for r in records] for records in fixed_records)
        if not all(fixed):
            raise ValueError("fixed release split produced an empty train, val, or test partition")
        split_counts = tuple(map(len, fixed))
        print(
            f"{sum(split_counts)} samples arch={a.arch} loss={a.loss} "
            f"fixed split counts={split_counts}"
        )
    else:
        encs = [encode_record(r) for r in recs]
        print(f"{len(encs)} samples arch={a.arch} loss={a.loss}")

    def fwd(model, e):
        if hetero:
            return model(e)
        return model(
            torch.from_numpy(e.x).to(device),
            torch.from_numpy(e.adj).to(device),
            torch.from_numpy(e.cand_masks).to(device),
            torch.from_numpy(e.cand_feats).to(device),
        )

    def evaluate(model, encs_, *, mode):
        model.eval()
        with torch.no_grad():
            return evaluate_quality_support(
                lambda example: fwd(model, example).detach().cpu().numpy(),
                encs_,
                support=a.evaluation_support,
                mode=mode,
                audit_scores=[audit_by_example[id(example)] for example in encs_],
                tol=a.tol,
            )

    results = []
    for seed in [int(s) for s in a.seeds.split(",")]:
        if fixed is not None:
            tr, va, te = fixed
        elif a.holdout_source:
            tr = [e for e in encs if e.source != a.holdout_source]
            te = [e for e in encs if e.source == a.holdout_source]
        else:
            tr, te = split_by_instance(encs, a.test_frac, seed)
        if fixed is None:
            tr, va = split_by_instance(tr, 0.1, seed)
        seed_device(seed, device)
        rng = random.Random(seed)
        model = (
            build_hetero(a.hidden, a.layers, a.heads)
            if hetero
            else build_chain_arch(
                a.arch,
                a.hidden,
                a.layers,
                a.heads,
                neighbour_feats=a.neighbour_feats,
            )
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
        best = None
        t0 = time.time()
        for ep in range(a.epochs):
            model.train()
            order = list(range(len(tr)))
            rng.shuffle(order)
            for b0 in range(0, len(order), 16):
                opt.zero_grad()
                loss = 0.0
                m = 0
                for k in order[b0 : b0 + 16]:
                    e = tr[k]
                    s = fwd(model, e)
                    batch_loss = (
                        listwise_loss(s, e.p, e.stage)
                        if a.loss == "listwise"
                        else pair_loss(
                            s,
                            e.p,
                            a.tol,
                            e.stage,
                            lengths=getattr(e, "lengths", None),
                            length_balance=a.length_balance,
                            stage2_only=a.stage2_only,
                        )
                    )
                    if batch_loss is not None:
                        loss = loss + batch_loss
                        m += 1
                if m:
                    (loss / m).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            evv = evaluate(model, va, mode="development")
            print(
                f"[{a.arch}/{a.loss} s{seed}] epoch {ep + 1} val regret "
                f"model {evv['model_regret']:.4f} "
                f"resource {evv['resource_regret']:.4f} "
                f"random {evv['random_regret']:.4f} pair {evv['pair_acc']:.3f}",
                flush=True,
            )
            if best is None or evv["model_regret"] < best[0]:
                best = (
                    evv["model_regret"],
                    ep + 1,
                    cpu_state_dict(model),
                )
        model.load_state_dict(best[2])
        validation = evaluate(model, va, mode="development")
        test_evaluated = fixed is None or a.evaluate_test
        evt = evaluate(model, te, mode=a.evaluation_mode) if test_evaluated else None
        by = {}
        if test_evaluated:
            for key in ("topology", "source"):
                for val in sorted({getattr(e, key) for e in te}):
                    by[f"{key}={val}"] = evaluate(
                        model,
                        [e for e in te if getattr(e, key) == val],
                        mode=a.evaluation_mode,
                    )
        from scipy.stats import spearmanr

        rhos = []
        rho_p = []
        with torch.no_grad():
            for e in te if test_evaluated else ():
                L = getattr(e, "lengths", None)
                if L and len(set(L)) > 1:
                    sc_ = fwd(model, e).cpu().numpy()
                    rhos.append(spearmanr(sc_, L).correlation)
                    rho_p.append(spearmanr(e.p, L).correlation)
        row = {
            "arch": a.arch,
            "loss": a.loss,
            "seed": seed,
            "device": resolved_device,
            "hidden": a.hidden,
            "layers": a.layers,
            "heads": a.heads,
            "params": n_params,
            "best_epoch": best[1],
            "n_train": len(tr),
            "n_val": len(va),
            "n_test": len(te),
            "validation": validation,
            "test_evaluated": test_evaluated,
            "test": evt,
            "by": by,
            "seconds": round(time.time() - t0, 1),
            "length_balance": a.length_balance,
            "stage2_only": a.stage2_only,
            "no_length_feats": a.no_length_feats,
            "deploy_view": a.deploy_view,
            "evaluation_support": a.evaluation_support,
            "evaluation_mode": a.evaluation_mode,
            "selection_evaluation_mode": "development",
            "label_fidelity": (
                evt["label_fidelity"] if evt is not None else validation["label_fidelity"]
            ),
            "audit_coverage": (
                evt["audit_coverage"] if evt is not None else validation["audit_coverage"]
            ),
            "spearman_score_length": float(np.nanmedian(rhos)) if rhos else None,
            "spearman_p_length": float(np.nanmedian(rho_p)) if rho_p else None,
        }
        print(json.dumps(row), flush=True)
        results.append(row)
        if a.save:
            preprocessing = {
                "neighbour_feats": a.neighbour_feats,
                "no_length_feats": a.no_length_feats,
                "deploy_view": a.deploy_view,
            }
            torch.save(
                {
                    "state": cpu_state_dict(model),
                    "meta": {
                        "artifact_schema": "embedbench.chain-scorer",
                        "artifact_schema_version": 1,
                        "arch": a.arch,
                        "hidden": a.hidden,
                        "layers": a.layers,
                        "heads": a.heads,
                        "seed": seed,
                        "l_cap": 4,
                        "no_length_feats": a.no_length_feats,
                        "deploy_view": a.deploy_view,
                        "stage2_only": a.stage2_only,
                        "length_balance": a.length_balance,
                        "neighbour_feats": a.neighbour_feats,
                        "preprocessing": preprocessing,
                        "device": resolved_device,
                        "splits": a.splits,
                        "split_sha256": split_sha256,
                        "data_provenance": provenance,
                        "runtime_provenance": runtime,
                        "evaluation_protocol": evaluation_protocol,
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
                "evaluation_protocol": evaluation_protocol,
                "results": results,
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
