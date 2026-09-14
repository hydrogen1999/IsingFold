from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_quality_v2.py"
_EVALUATOR_TEST_DRIVER = r"""
import json
import sys

sys.path.insert(0, sys.argv[1])
from evaluate_quality_v2 import _parse_args, _write_output, evaluate

args = _parse_args(sys.argv[2:])
result = evaluate(args)
_write_output(args, result)
print(json.dumps(result, allow_nan=False))
"""
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_quality_v2 import (  # noqa: E402
    PAPER_AUDIT_ROW_FIELDS,
    _captured_audit_inputs,
    _load_audit_paper_bindings,
    _sanitized_model_record,
    _secondary_strength_robustness,
)
from quality_v2_paper_contract import (  # noqa: E402
    AUDIT_SOURCE_HASH_ALGORITHM,
    PAPER_PIPELINE_SCRIPTS,
    PAPER_VERIFIER_PACKAGE_FILES,
    build_paper_preregistration,
    load_paper_audit_contract,
    selection_audit_binding,
)


def _record(instance_id: str, focus_h: float) -> dict:
    return {
        "instance_id": instance_id,
        "focus": 0,
        "n_vars": 2,
        "window_nodes": [0, 1, 4],
        "window_edges": [[0, 4], [1, 4]],
        "frozen": {"2": [5]},
        "all_chains": {"2": [5]},
        "frozen_adjacency": {"0": [5], "1": [5]},
        "neighbours": [2],
        "edge_J": [-1.0],
        "neighbour_degree": [1],
        "neighbour_chain_size": [1],
        "neighbour_h": [0.0],
        "candidates": [[0], [1], [0, 4]],
        "Q": [1, 1, 2],
        # Release labels are deliberately unusable. Paper evaluation must consume only
        # independently verified high-read audit labels.
        "p_solve": ["development-only", "development-only", "development-only"],
        "stage": [1, 1, 1],
        "best_index": 2,
        "resource_index": 0,
        "original_index": 1,
        "source": "minorminer",
        "topology": "chimera",
        "size": 1,
        "difficulty": "test",
        "l_cap": 4,
        "focus_h": focus_h,
        "problem": {
            "h": {"0": focus_h, "2": 0.0},
            "J": [[0, 2, -1.0]],
            "e0": -1.0 - focus_h,
        },
    }


def test_model_input_projection_removes_release_labels_policy_indices_and_ground_energy() -> None:
    record = _record("label-firewall", 0.25)
    record["unregistered_future_quality_label"] = 0.99

    sanitized = _sanitized_model_record(record)

    assert not {
        "p_solve",
        "stage",
        "best_index",
        "resource_index",
        "original_index",
        "unregistered_future_quality_label",
    } & set(sanitized)
    assert sanitized["problem"] == {
        "h": record["problem"]["h"],
        "J": record["problem"]["J"],
    }
    legacy_current = (
        set(sanitized["candidates"][sanitized["original_index"]])
        if sanitized.get("original_index", -1) >= 0
        else set()
    )
    assert legacy_current == set()


def test_captured_audit_row_identity_must_match_its_manifest_entry(tmp_path: Path) -> None:
    row = {"file": "quality.jsonl", "instance_id": "actual", "focus": 0}
    payload = (json.dumps(row, separators=(",", ":")) + "\n").encode()
    path = tmp_path / "audit.jsonl"
    path.write_bytes(payload)
    claimed = b'["quality.jsonl","claimed",0]'
    actual = b'["quality.jsonl","actual",0]'
    shard_index = int.from_bytes(hashlib.sha256(actual).digest(), "big") % 64
    manifest = {
        "shard_count": 64,
        "shard_index": shard_index,
        "jsonl": {
            "file": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        },
        "records": [
            {
                "identity": claimed.decode(),
                "relative_path": (
                    f"{path.name}.records/{hashlib.sha256(claimed).hexdigest()}.json"
                ),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        ],
    }

    with (
        pytest.raises(ValueError, match="identity disagrees with manifest"),
        _captured_audit_inputs(
            [path],
            {path.name: manifest},
            {path.name: {"file": "manifest.json", "sha256": "a" * 64}},
        ),
    ):
        raise AssertionError("mismatched audit input must not be yielded")


def test_captured_audit_row_schema_rejects_unknown_fields(tmp_path: Path) -> None:
    row = {field: None for field in PAPER_AUDIT_ROW_FIELDS}
    row.update(
        {
            "file": "quality.jsonl",
            "instance_id": "instance",
            "focus": 0,
            "unregistered_label": 1.0,
        }
    )
    payload = (json.dumps(row, separators=(",", ":")) + "\n").encode()
    path = tmp_path / "audit.jsonl"
    path.write_bytes(payload)
    identity = b'["quality.jsonl","instance",0]'
    shard_index = int.from_bytes(hashlib.sha256(identity).digest(), "big") % 64
    manifest = {
        "shard_count": 64,
        "shard_index": shard_index,
        "jsonl": {
            "file": path.name,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        },
        "records": [
            {
                "identity": identity.decode(),
                "relative_path": (
                    f"{path.name}.records/{hashlib.sha256(identity).hexdigest()}.json"
                ),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        ],
    }

    with (
        pytest.raises(ValueError, match="missing or unexpected fields"),
        _captured_audit_inputs(
            [path],
            {path.name: manifest},
            {path.name: {"file": "manifest.json", "sha256": "a" * 64}},
        ),
    ):
        raise AssertionError("unknown audit fields must not be yielded")


def test_captured_audit_inputs_accept_rescorer_record_directory(tmp_path: Path) -> None:
    """The evaluator must accept the exact relative path emitted by the rescorer."""

    from rescore_quality import finalize_paper_audit_shard

    row = {field: None for field in PAPER_AUDIT_ROW_FIELDS}
    row.update({"file": "quality.jsonl", "instance_id": "instance", "focus": 0})
    payload = (json.dumps(row, separators=(",", ":")) + "\n").encode()
    path = tmp_path / "audit-shard-00.jsonl"
    identity = b'["quality.jsonl","instance",0]'
    shard_index = int.from_bytes(hashlib.sha256(identity).digest(), "big") % 64
    record_path = path.with_name(f"{path.name}.records") / (
        f"{hashlib.sha256(identity).hexdigest()}.json"
    )
    record_path.parent.mkdir()
    record_path.write_bytes(payload)
    manifest = finalize_paper_audit_shard(
        path,
        [(identity, record_path, payload)],
        audit_contract_binding={"sha256": "1" * 64},
        preregistration_binding={"sha256": "4" * 64},
        selection_binding={"sha256": "2" * 64},
        audit_source_binding={"sha256": "3" * 64},
        shard_count=64,
        shard_index=shard_index,
        complete_fixed_test_record_count=1,
    )

    with _captured_audit_inputs(
        [path],
        {path.name: manifest},
        {path.name: {"file": "manifest.json", "sha256": "a" * 64}},
    ) as (captured, artifacts):
        assert [item.read_bytes() for item in captured] == [payload]
        assert artifacts[0]["sha256"] == hashlib.sha256(payload).hexdigest()


def test_paper_audit_requires_integer_success_counts_for_every_strength(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text(
        json.dumps(
            {
                "file": "quality.jsonl",
                "instance_id": "instance",
                "focus": 0,
                "resource_index": 0,
                "original_index": 1,
                "high_read_scores": [0.25, 0.5],
                "high_read_best": 1,
                "strength_p_solve": [[0.25] * 4, [0.5] * 4],
                "reads": 4000,
                "paper_binding": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="integer success-count evidence"):
        _load_audit_paper_bindings(
            [path],
            lambda row: (row["file"], row["instance_id"], row["focus"]),
            {("quality.jsonl", "instance", 0): (0, 1)},
            expected_candidate_counts={("quality.jsonl", "instance", 0): 2},
            registered_strength_count=4,
            registered_reads_per_strength=4000,
        )


def test_paper_audit_rejects_probability_not_derived_from_success_count(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text(
        json.dumps(
            {
                "file": "quality.jsonl",
                "instance_id": "instance",
                "focus": 0,
                "resource_index": 0,
                "original_index": 1,
                "high_read_scores": [0.25, 0.5],
                "high_read_best": 1,
                "strength_p_solve": [[0.25] * 4, [0.5] * 4],
                "strength_success_counts": [[999] * 4, [2000] * 4],
                "reads": 4000,
                "paper_binding": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="probabilities disagree with integer success-count"):
        _load_audit_paper_bindings(
            [path],
            lambda row: (row["file"], row["instance_id"], row["focus"]),
            {("quality.jsonl", "instance", 0): (0, 1)},
            expected_candidate_counts={("quality.jsonl", "instance", 0): 2},
            registered_strength_count=4,
            registered_reads_per_strength=4000,
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_signature(record: dict) -> str:
    payload = json.dumps(
        [sorted(int(qubit) for qubit in candidate) for candidate in record["candidates"]],
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _audit_row(
    corpus: Path,
    record: dict,
    scores: list[float],
    selection: Path,
    selection_document: dict,
    selection_sha256: str,
    audit_contract,
) -> dict:
    from embedbench.embedding import LogicalProblem
    from embedbench.surrogate import default_strength_grid
    from training_artifacts import runtime_provenance

    torch = pytest.importorskip("torch")

    raw_problem = record["problem"]
    problem = LogicalProblem.from_dicts(
        {int(key): value for key, value in raw_problem["h"].items()},
        {(u, v): weight for u, v, weight in raw_problem["J"]},
    )
    strengths = default_strength_grid(problem, 4)
    realized_seeds = [
        [(7919 * candidate_index + 31 * strength_index) % (2**31) for strength_index in range(4)]
        for candidate_index in range(len(record["candidates"]))
    ]
    strength_p_solve = [[float(score)] * 4 for score in scores]
    strength_success_counts = [[round(float(score) * 4000)] * 4 for score in scores]
    return {
        "audit_schema": "embedbench.independent-quality-audit",
        "audit_schema_version": 2,
        "file": corpus.name,
        "corpus_sha256": _sha256(corpus),
        "instance_id": record["instance_id"],
        "focus": record["focus"],
        "candidate_signature": _candidate_signature(record),
        "resource_index": record["resource_index"],
        "original_index": record.get("original_index", -1),
        "high_read_best": max(range(len(scores)), key=lambda index: scores[index]),
        "high_read_scores": scores,
        "strength_p_solve": strength_p_solve,
        "strength_success_counts": strength_success_counts,
        "reads": 4000,
        "provenance": {
            "generator": "EmbedBench/scripts/rescore_quality.py",
            "reads_per_strength": 4000,
            "num_sweeps": 200,
            "base_seed": 0,
            "registered_strength_count": 4,
            "seed_schedule": ("(base_seed + 7919*candidate_index + 31*strength_index) % 2**31"),
            "strength_schedule": "default_strength_grid(problem, 4)",
            "aggregation": "max_p_solve_over_strength_schedule",
            "score_evidence": "integer_success_counts_per_candidate_strength",
            "probability_derivation": ("success_count_divided_by_reads_per_strength_binary64"),
        },
        "paper_binding": {
            "audit_contract": audit_contract.public_binding(),
            "selection": selection_audit_binding(
                selection_document,
                selection_file=selection.name,
                selection_sha256=selection_sha256,
            ),
            "audit_source": {
                "algorithm": AUDIT_SOURCE_HASH_ALGORITHM,
                "sha256": selection_document["audit_source_sha256"],
            },
            "runtime_provenance": runtime_provenance(torch.device("cpu")),
            "ground_reference": {
                "status": "certified_exact",
                "method": "exhaustive_enumeration",
                "energy": raw_problem["e0"],
                "n_variables": len(raw_problem["h"]),
            },
            "realized_strengths": strengths,
            "realized_seeds": realized_seeds,
        },
    }


def _write_audit_shards(
    root: Path,
    *,
    row: dict,
    selection: Path,
    selection_document: dict,
    selection_sha256: str,
    preregistration_binding: dict,
    audit_contract,
) -> tuple[list[Path], Path, Path, str]:
    from rescore_quality import build_audit_release_commitment

    identity_bytes = json.dumps(
        [row["file"], row["instance_id"], row["focus"]],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    shard_count = 64
    occupied = int.from_bytes(hashlib.sha256(identity_bytes).digest(), "big") % shard_count
    selection_binding = selection_audit_binding(
        selection_document,
        selection_file=selection.name,
        selection_sha256=selection_sha256,
    )
    row["paper_binding"]["preregistration"] = preregistration_binding
    paths: list[Path] = []
    occupied_path: Path | None = None
    for shard_index in range(shard_count):
        path = root / f"audit-shard-{shard_index:02d}.jsonl"
        payload = (json.dumps(row) + "\n").encode() if shard_index == occupied else b""
        path.write_bytes(payload)
        records = []
        if payload:
            records.append(
                {
                    "identity": identity_bytes.decode(),
                    "relative_path": (
                        f"{path.name}.records/{hashlib.sha256(identity_bytes).hexdigest()}.json"
                    ),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                }
            )
            occupied_path = path
        manifest = {
            "schema": "embedbench.quality-v2-audit-shard",
            "schema_version": 2,
            "audit_contract": audit_contract.public_binding(),
            "preregistration": preregistration_binding,
            "selection": selection_binding,
            "audit_source": {
                "algorithm": AUDIT_SOURCE_HASH_ALGORITHM,
                "sha256": selection_document["audit_source_sha256"],
            },
            "shard_count": shard_count,
            "shard_index": shard_index,
            "assignment": "sha256(canonical_audit_identity) mod shard_count",
            "canonical_audit_identity": "json_compact_utf8_array[file,instance_id,focus]",
            "merge_order": "canonical_audit_identity_lexicographic",
            "complete_fixed_test_record_count": 1,
            "record_count": len(records),
            "records": records,
            "jsonl": {
                "file": path.name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            },
        }
        path.with_name(f"{path.name}.manifest.json").write_text(
            json.dumps(manifest),
            encoding="utf-8",
        )
        paths.append(path)
    assert occupied_path is not None
    audit_source = {
        "algorithm": AUDIT_SOURCE_HASH_ALGORITHM,
        "sha256": selection_document["audit_source_sha256"],
    }
    _, release_payload = build_audit_release_commitment(
        paths,
        audit_contract_binding=audit_contract.public_binding(),
        preregistration_binding=preregistration_binding,
        selection_binding=selection_binding,
        audit_source_binding=audit_source,
        expected_shard_count=shard_count,
    )
    release_manifest = root / "audit-release-manifest.json"
    release_manifest.write_bytes(release_payload)
    return paths, occupied_path, release_manifest, hashlib.sha256(release_payload).hexdigest()


def _refresh_audit_manifest(path: Path) -> None:
    payload = path.read_bytes()
    manifest_path = path.with_name(f"{path.name}.manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = []
    for line in payload.splitlines(keepends=True):
        row = json.loads(line)
        identity = json.dumps(
            [row["file"], row["instance_id"], row["focus"]],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        records.append(
            {
                "identity": identity,
                "relative_path": (
                    f"{path.name}.records/{hashlib.sha256(identity.encode()).hexdigest()}.json"
                ),
                "sha256": hashlib.sha256(line).hexdigest(),
                "bytes": len(line),
            }
        )
    manifest["record_count"] = len(records)
    manifest["records"] = records
    manifest["jsonl"] = {
        "file": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_fixture(
    tmp_path: Path,
    *,
    arch: str = "mpnn",
    deploy_view: bool = False,
    poison_development_partitions: bool = False,
    include_omitted_split_corpus: bool = False,
) -> dict[str, object]:
    pytest.importorskip("torch")

    tmp_path.mkdir(parents=True, exist_ok=True)
    records = [
        _record("problem-train-m", 0.25),
        _record("problem-val-m", 1.25),
        _record("problem-test-m", 2.25),
    ]
    for record in records[:2]:
        record["p_solve"] = [0.50, 0.30, 0.90]
        record["stage"] = [2, 2, 2]
    if poison_development_partitions:
        # These rows remain sufficient for split leakage validation but cannot be encoded,
        # window-rebuilt, or evaluated. A paper evaluator must never touch them downstream.
        for record in records[:1]:
            for field in (
                "all_chains",
                "candidates",
                "window_nodes",
                "window_edges",
                "topology",
                "size",
            ):
                record.pop(field)
    corpus = tmp_path / "quality_chimera1.jsonl"
    corpus.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    corpus_sha = _sha256(corpus)
    generator_manifest = Path(f"{corpus}.manifest.json")
    generator_manifest.write_text(
        json.dumps(
            {
                "config": {
                    "topology": "chimera",
                    "size": 1,
                    "defect_qubits": 0.0,
                    "defect_couplers": 0.0,
                },
                "sha256": corpus_sha,
            }
        ),
        encoding="utf-8",
    )
    corpus_inputs = [corpus]
    split_inputs = [{"file": corpus.name, "sha256": corpus_sha}]
    split_assignments = {
        corpus.name: {
            records[0]["instance_id"]: "train",
            records[1]["instance_id"]: "val",
            records[2]["instance_id"]: "test",
        }
    }
    if include_omitted_split_corpus:
        omitted = tmp_path / "quality_omitted.jsonl"
        omitted_record = _record("problem-omitted-test-m", 3.25)
        omitted.write_text(json.dumps(omitted_record) + "\n", encoding="utf-8")
        corpus_inputs.append(omitted)
        split_inputs.append({"file": omitted.name, "sha256": _sha256(omitted)})
        split_assignments[omitted.name] = {omitted_record["instance_id"]: "test"}
    deployment_host_contract: dict[str, object]
    if deploy_view:
        deployment_host_contract = {
            "mode": "pristine_topology_reconstruction",
            "source_manifests": [
                {
                    "file": corpus.name,
                    "corpus_sha256": corpus_sha,
                    "manifest_sha256": _sha256(generator_manifest),
                    "defect_qubits": 0.0,
                    "defect_couplers": 0.0,
                }
            ],
        }
    else:
        deployment_host_contract = {"mode": "not_applied", "source_manifests": []}
    splits = tmp_path / "splits.json"
    splits.write_text(
        json.dumps(
            {
                "schema": "embedbench.split-manifest",
                "schema_version": 2,
                "provenance": {"inputs": split_inputs},
                "splits": split_assignments,
            }
        ),
        encoding="utf-8",
    )
    from tests.test_select_training_grid import (
        REGISTERED_EXTRA_ARGS,
        _load_modules,
        _write_cell,
    )

    runner, selector = _load_modules()
    shutil.copytree(ROOT / "src" / "embedbench", tmp_path / "src" / "embedbench")
    (tmp_path / "scripts").mkdir()
    for script_name in PAPER_PIPELINE_SCRIPTS:
        shutil.copy2(ROOT / "scripts" / script_name, tmp_path / "scripts" / script_name)
    for relative in PAPER_VERIFIER_PACKAGE_FILES:
        destination = tmp_path / "scripts" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "scripts" / relative, destination)
    objective_variants = ["full", *(f"registered_{index:02d}" for index in range(1, 20))]
    extra_args = list(REGISTERED_EXTRA_ARGS)
    if deploy_view:
        extra_args.append("--deploy-view")
    grid_document = {
        "schema": "embedbench.training-grid",
        "schema_version": 1,
        "grid_id": "isingfold-quality-value-v2-screen",
        "selection": {
            "primary_metric": "mean_finite_budget_regret",
            "seeds": [0, 1, 2, 3],
        },
        "stages": {
            "quality_value_v2_screen": {
                "trainer": "scripts/train_quality_v2.py",
                "inputs": "quality_*.jsonl",
                "splits": splits.name,
                "output_dir": "runs/grid/screen",
                "checkpoint_dir": "runs/grid/checkpoints",
                "epochs": 1,
                "axis_order": ["objective_variant", "arch", "seed"],
                "axes": {
                    "objective_variant": objective_variants,
                    "arch": [arch],
                    "seed": [0, 1, 2, 3],
                },
                "extra_args": extra_args,
                "execution": {
                    "assignment": "cell_index_modulo",
                    "modulus": 1,
                    "remainder_to_site": {"0": "test-site"},
                    "required_device_type": "cpu",
                    "slurm_required_sites": [],
                    "slurm_forbidden_sites": ["test-site"],
                    "require_each_configuration_all_sites": True,
                    "exact_balance_axes": ["arch", "objective_variant"],
                },
            }
        },
    }
    grid = tmp_path / "configs" / "training_grid_quality_v2.json"
    grid.parent.mkdir()
    grid.write_text(json.dumps(grid_document), encoding="utf-8")
    cells = runner.expand_stage(grid_document, "quality_value_v2_screen")
    data_provenance = runner._data_provenance(tuple(corpus_inputs), splits)
    grid_sha = runner._sha256(grid)
    source_sha = runner._source_sha256(tmp_path)
    for cell in cells:
        _write_cell(
            runner,
            tmp_path,
            cell,
            0.1 if cell.params["objective_variant"] == "full" else 0.2,
            data_provenance,
            grid_sha,
            source_sha,
            grid_path=grid,
            deploy_view=deploy_view,
            deployment_host_contract_override=deployment_host_contract,
        )
    contract_path = ROOT / "configs" / "quality_v2_paper_audit_v2.json"
    contract = load_paper_audit_contract(contract_path, _sha256(contract_path))
    test_contract_document = copy.deepcopy(contract.document)
    test_contract_document["training_registration"] = {
        "grid_sha256": grid_sha,
        "source_sha256": source_sha,
        "selection": copy.deepcopy(grid_document["selection"]),
        "stage": copy.deepcopy(grid_document["stages"]["quality_value_v2_screen"]),
        "data_provenance": copy.deepcopy(data_provenance),
        "validation_replay": copy.deepcopy(contract.training_registration["validation_replay"]),
    }
    test_contract_path = tmp_path / "quality_v2_paper_audit_test.json"
    test_contract_path.write_text(json.dumps(test_contract_document), encoding="utf-8")
    test_contract_sha = _sha256(test_contract_path)
    test_contract_path.with_suffix(".sha256").write_text(
        f"{test_contract_sha}  {test_contract_path.name}\n",
        encoding="utf-8",
    )
    test_contract = load_paper_audit_contract(test_contract_path, test_contract_sha)
    selection_document = selector.select_grid(
        grid_document,
        grid_path=grid,
        stage_name="quality_value_v2_screen",
        root=tmp_path,
        audit_contract=test_contract,
    )
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps(selection_document), encoding="utf-8")
    selection_sha = _sha256(selection)
    checkpoint = tmp_path / selection_document["winner"]["checkpoint"]
    checkpoint_sha = _sha256(checkpoint)
    policy_freeze = tmp_path / "policy-freeze.json"
    freeze_fixture = {
        "corpus": corpus,
        "splits": splits,
        "checkpoint": checkpoint,
        "checkpoint_sha": checkpoint_sha,
        "selection": selection,
        "selection_sha": selection_sha,
        "audit_contract": test_contract_path,
        "audit_contract_sha": test_contract_sha,
        "policy_freeze": policy_freeze,
    }
    frozen = subprocess.run(
        _freeze_command(freeze_fixture),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    if frozen.returncode != 0:
        raise RuntimeError(f"test policy-freeze setup failed: {frozen.stderr}")
    policy_freeze_sha = _sha256(policy_freeze)
    selected_seed = selection_document["winner"]["selected_seed"]
    selected_freeze_document = json.loads(policy_freeze.read_text(encoding="utf-8"))
    policy_freezes: list[Path] = []
    for registered_checkpoint in selection_document["winner"]["paper_checkpoints"]:
        seed = registered_checkpoint["seed"]
        if seed == selected_seed:
            path = policy_freeze
        else:
            path = tmp_path / f"policy-freeze-seed-{seed}.json"
            document = copy.deepcopy(selected_freeze_document)
            document["selection"]["paper_checkpoint"] = registered_checkpoint
            document["checkpoint"].update(
                seed=seed,
                file=Path(registered_checkpoint["checkpoint"]).name,
                sha256=registered_checkpoint["checkpoint_sha256"],
            )
            path.write_bytes(
                (
                    json.dumps(
                        document,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode()
            )
        policy_freezes.append(path)
    preregistration_document, preregistration_payload = build_paper_preregistration(
        audit_contract=test_contract,
        selection_document=selection_document,
        selection_file=selection.name,
        selection_sha256=selection_sha,
        audit_source_root=ROOT,
        policy_freezes=policy_freezes,
    )
    preregistration = tmp_path / "paper-preregistration.json"
    preregistration.write_bytes(preregistration_payload)
    preregistration_sha = hashlib.sha256(preregistration_payload).hexdigest()
    preregistration_binding = {
        "file": preregistration.name,
        "sha256": preregistration_sha,
        "schema": preregistration_document["schema"],
        "schema_version": preregistration_document["schema_version"],
        "protocol_id": preregistration_document["protocol_id"],
        "trust_model": preregistration_document["trust_model"],
        "commitment_scope": preregistration_document["commitment_scope"],
    }
    audits, audit, audit_release_manifest, audit_release_manifest_sha = _write_audit_shards(
        tmp_path,
        row=_audit_row(
            corpus,
            records[2],
            [0.20, 0.80, 0.70],
            selection,
            selection_document,
            selection_sha,
            test_contract,
        ),
        selection=selection,
        selection_document=selection_document,
        selection_sha256=selection_sha,
        preregistration_binding=preregistration_binding,
        audit_contract=test_contract,
    )
    return {
        "corpus": corpus,
        "splits": splits,
        "audit": audit,
        "audits": audits,
        "audit_release_manifest": audit_release_manifest,
        "audit_release_manifest_sha": audit_release_manifest_sha,
        "checkpoint": checkpoint,
        "checkpoint_sha": checkpoint_sha,
        "selection": selection,
        "selection_sha": selection_sha,
        "audit_contract": test_contract_path,
        "audit_contract_sha": test_contract_sha,
        "policy_freeze": policy_freeze,
        "policy_freeze_sha": policy_freeze_sha,
        "policy_freezes": policy_freezes,
        "preregistration": preregistration,
        "preregistration_sha": preregistration_sha,
        "preregistration_binding": preregistration_binding,
        "output": tmp_path / "paper.json",
    }


def _driver_prefix(fixture: dict[str, object]) -> list[str]:
    return [
        sys.executable,
        "-c",
        _EVALUATOR_TEST_DRIVER,
        str(ROOT / "scripts"),
    ]


def _freeze_command(fixture: dict[str, object]) -> list[str]:
    return [
        *_driver_prefix(fixture),
        "--phase",
        "freeze",
        str(fixture["corpus"]),
        "--splits",
        str(fixture["splits"]),
        "--checkpoint",
        str(fixture["checkpoint"]),
        "--checkpoint-sha256",
        str(fixture["checkpoint_sha"]),
        "--selection",
        str(fixture["selection"]),
        "--selection-sha256",
        str(fixture["selection_sha"]),
        "--selection-root",
        str(Path(fixture["selection"]).parent),
        "--audit-contract",
        str(fixture["audit_contract"]),
        "--audit-contract-sha256",
        str(fixture["audit_contract_sha"]),
        "--out",
        str(fixture["policy_freeze"]),
    ]


def _command(fixture: dict[str, object]) -> list[str]:
    return [
        *_driver_prefix(fixture),
        "--phase",
        "score",
        str(fixture["corpus"]),
        "--splits",
        str(fixture["splits"]),
        "--checkpoint",
        str(fixture["checkpoint"]),
        "--checkpoint-sha256",
        str(fixture["checkpoint_sha"]),
        "--selection",
        str(fixture["selection"]),
        "--selection-sha256",
        str(fixture["selection_sha"]),
        "--selection-root",
        str(Path(fixture["selection"]).parent),
        "--policy-freeze",
        str(fixture["policy_freeze"]),
        "--preregistration-manifest",
        str(fixture["preregistration"]),
        "--preregistration-manifest-sha256",
        str(fixture["preregistration_sha"]),
        "--audit-labels",
        *(str(path) for path in fixture["audits"]),
        "--audit-release-manifest",
        str(fixture["audit_release_manifest"]),
        "--audit-release-manifest-sha256",
        str(fixture["audit_release_manifest_sha"]),
        "--audit-contract",
        str(fixture["audit_contract"]),
        "--audit-contract-sha256",
        str(fixture["audit_contract_sha"]),
        "--out",
        str(fixture["output"]),
    ]


def _refresh_audit_release(fixture: dict[str, object]) -> None:
    from rescore_quality import build_audit_release_commitment

    selection = Path(fixture["selection"])
    selection_document = json.loads(selection.read_text(encoding="utf-8"))
    contract_path = Path(fixture["audit_contract"])
    contract = load_paper_audit_contract(contract_path, str(fixture["audit_contract_sha"]))
    selection_binding = selection_audit_binding(
        selection_document,
        selection_file=selection.name,
        selection_sha256=str(fixture["selection_sha"]),
    )
    _, payload = build_audit_release_commitment(
        fixture["audits"],
        audit_contract_binding=contract.public_binding(),
        preregistration_binding=fixture["preregistration_binding"],
        selection_binding=selection_binding,
        audit_source_binding={
            "algorithm": AUDIT_SOURCE_HASH_ALGORITHM,
            "sha256": selection_document["audit_source_sha256"],
        },
        expected_shard_count=64,
    )
    Path(fixture["audit_release_manifest"]).write_bytes(payload)
    fixture["audit_release_manifest_sha"] = hashlib.sha256(payload).hexdigest()


def _refresh_preregistration(fixture: dict[str, object]) -> None:
    selection = Path(fixture["selection"])
    selection_document = json.loads(selection.read_text(encoding="utf-8"))
    audit_contract = load_paper_audit_contract(
        Path(fixture["audit_contract"]),
        str(fixture["audit_contract_sha"]),
    )
    document, payload = build_paper_preregistration(
        audit_contract=audit_contract,
        selection_document=selection_document,
        selection_file=selection.name,
        selection_sha256=str(fixture["selection_sha"]),
        audit_source_root=ROOT,
        policy_freezes=fixture["policy_freezes"],
    )
    preregistration = Path(fixture["preregistration"])
    preregistration.write_bytes(payload)
    binding = {
        "file": preregistration.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "schema": document["schema"],
        "schema_version": document["schema_version"],
        "protocol_id": document["protocol_id"],
        "trust_model": document["trust_model"],
        "commitment_scope": document["commitment_scope"],
    }
    fixture["preregistration_sha"] = binding["sha256"]
    fixture["preregistration_binding"] = binding
    fixture["policy_freeze_sha"] = _sha256(Path(fixture["policy_freeze"]))

    audit = Path(fixture["audit"])
    row = json.loads(audit.read_text(encoding="utf-8"))
    row["paper_binding"]["preregistration"] = binding
    audit.write_text(json.dumps(row) + "\n", encoding="utf-8")
    for path in fixture["audits"]:
        manifest_path = Path(path).with_name(f"{Path(path).name}.manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["preregistration"] = binding
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _refresh_audit_manifest(audit)
    _refresh_audit_release(fixture)


def _rewrite_audit(fixture: dict[str, object], mutate, *, recommit_release: bool = True) -> None:
    audit = Path(fixture["audit"])
    row = json.loads(audit.read_text(encoding="utf-8"))
    mutate(row)
    audit.write_text(json.dumps(row) + "\n", encoding="utf-8")
    _refresh_audit_manifest(audit)
    if recommit_release:
        _refresh_audit_release(fixture)


def _rewrite_selection(fixture: dict[str, object], mutate) -> None:
    selection = Path(fixture["selection"])
    document = json.loads(selection.read_text(encoding="utf-8"))
    mutate(document)
    selection.write_text(json.dumps(document), encoding="utf-8")
    fixture["selection_sha"] = _sha256(selection)


def _rewrite_checkpoint(fixture: dict[str, object], mutate) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = Path(fixture["checkpoint"])
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    mutate(payload)
    torch.save(payload, checkpoint)
    fixture["checkpoint_sha"] = _sha256(checkpoint)
    current_selection = json.loads(Path(fixture["selection"]).read_text(encoding="utf-8"))
    selected_cell_id = current_selection["winner"]["selected_cell_id"]
    receipt = (
        Path(fixture["selection"]).parent
        / "runs"
        / "grid"
        / "screen"
        / f"{selected_cell_id}.receipt.json"
    )
    receipt_document = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_document["checkpoint_sha256"] = fixture["checkpoint_sha"]
    receipt.write_text(json.dumps(receipt_document), encoding="utf-8")

    def update_selection(document: dict) -> None:
        selected_seed = document["winner"]["selected_seed"]
        for row in document["winner"]["paper_checkpoints"]:
            if row["seed"] == selected_seed:
                row["checkpoint_sha256"] = fixture["checkpoint_sha"]
        document["winner"]["checkpoint_sha256"] = fixture["checkpoint_sha"]

    _rewrite_selection(fixture, update_selection)


def test_frozen_checkpoint_evaluator_opens_only_fixed_test_and_emits_paper_metrics(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)

    completed = subprocess.run(
        _command(fixture),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(Path(fixture["output"]).read_text(encoding="utf-8"))
    assert result["artifact_schema"] == "embedbench.quality-v2-paper-evaluation"
    assert result["artifact_schema_version"] == 2
    assert result["evaluation_mode"] == "paper"
    assert result["partition"] == "test"
    assert result["provisional"] is True
    assert result["evaluation_design"] == "exploratory_legacy_iid"
    assert result["label_fidelity"] == "independent-audit"
    assert result["release_labels_consumed"] is False
    assert result["exact_host_contract"]["mode"] == "verified_pristine_manifests"
    assert result["runtime_provenance"]["schema"] == "embedbench.runtime-environment"
    assert result["runtime_provenance"]["device"]["selected"] == "cpu"
    assert result["candidate_support"] == "full"
    assert result["record_support"] == "complete_fixed_test_partition"
    assert result["support_uses_labels"] is False
    from embedbench.hamiltonian_context import hamiltonian_context_contract

    assert result["checkpoint"]["artifact_schema_version"] == 2
    assert result["preprocessing"]["hamiltonian_context"] == hamiltonian_context_contract()
    assert result["selection"]["sha256"] == fixture["selection_sha"]
    contract = load_paper_audit_contract(
        Path(fixture["audit_contract"]),
        str(fixture["audit_contract_sha"]),
    )
    assert result["audit_contract"] == contract.public_binding()
    assert result["selection"]["registered_seeds"] == [0, 1, 2, 3]
    selection_document = json.loads(Path(fixture["selection"]).read_text(encoding="utf-8"))
    expected_checkpoint = next(
        checkpoint
        for checkpoint in selection_document["winner"]["paper_checkpoints"]
        if checkpoint["seed"] == selection_document["winner"]["selected_seed"]
    )
    assert result["selection"]["paper_checkpoint"] == expected_checkpoint
    assert result["n_test"] == 1
    assert result["n_train_encoded"] == 0
    assert result["n_val_encoded"] == 0
    assert result["audit_coverage"]["record_fraction_complete"] == 1.0
    assert result["ground_reference_coverage"]["by_format"] == {
        "legacy_exhaustive_enumeration_v1_1_v1_2": 1,
        "legacy_ferromagnetic_s_t_min_cut_v1_1_v1_2": 0,
        "wp6_exact_enumeration_v1": 0,
        "wp6_planted_proof_v1": 0,
    }
    audit_binding = json.loads(Path(fixture["audit"]).read_text(encoding="utf-8"))["paper_binding"]
    assert len(result["evaluated_records"]) == 1
    evaluated = result["evaluated_records"][0]
    assert evaluated["record_index"] == 0
    assert evaluated["file"] == "quality_chimera1.jsonl"
    assert evaluated["line_number"] == 3
    assert evaluated["instance_id"] == "problem-test-m"
    assert evaluated["focus"] == 0
    assert evaluated["corpus_sha256"] == _sha256(Path(fixture["corpus"]))
    assert evaluated["candidate_signature"] == _candidate_signature(_record("problem-test-m", 2.25))
    assert evaluated["candidate_count"] == 3
    assert evaluated["ground_reference"] == audit_binding["ground_reference"]
    assert evaluated["realized_strengths"] == audit_binding["realized_strengths"]
    assert evaluated["realized_seeds"] == audit_binding["realized_seeds"]
    assert len(result["audit_artifacts"]) == 64
    freeze_evidence = result["label_free_policy_freeze"]
    assert "phase_1_registered_before_audit" not in freeze_evidence
    assert freeze_evidence["preregistration_manifest_caller_supplied_sha256_verified"] is True
    assert freeze_evidence["policy_freeze_listed_in_preregistration"] is True
    assert freeze_evidence["policy_freeze_bytes_match_preregistration"] is True
    assert freeze_evidence["byte_exact_phase_2_replay_verified"] is True
    assert freeze_evidence["audit_inputs_opened_after_preregistration_and_replay"] is True
    sweep = result["budget_sweep"]
    assert sweep["budget_contract"] == "B/Q_MM"
    assert sweep["budget_reference"] == "stock_minorminer_original_total_qubits"
    assert sweep["primary_record_coverage"]["included_records"] == 1
    assert sweep["budgets"]["1.00x"]["no_survivor"] == 0
    assert sweep["budgets"]["1.00x"]["selectors"]["mean"]["n_selected"] == 1
    secondary = result["secondary_endpoints"]
    assert secondary["selection_role"] == "report_only_after_label_free_policy_freeze"
    assert secondary["exact_feasibility_budget_survival"]["status"] == "available"
    assert (
        secondary["exact_feasibility_budget_survival"]["full_support"]["candidate_feasibility_rate"]
        == 1.0
    )
    assert secondary["terminal_resources"]["status"] == "available"
    assert secondary["residual_connectivity"]["status"] == "available"
    residual_original = secondary["residual_connectivity"]["budgets"]["1.00x"]["selectors"][
        "original"
    ]
    assert residual_original["mean_remaining_free_qubits"] is not None
    assert residual_original["mean_largest_remaining_component"] is not None
    assert residual_original["mean_largest_component_fraction"] is not None
    assert residual_original["mean_remaining_component_count"] is not None
    assert residual_original["paired_records"] == 1
    assert (
        residual_original["per_record"][0]["remaining_free_qubits_delta_vs_stock_original"] == 0.0
    )
    assert (
        residual_original["per_record"][0]["largest_remaining_component_delta_vs_stock_original"]
        == 0.0
    )
    assert secondary["strength_robustness"]["status"] == "available"
    assert secondary["strength_robustness"]["coverage"] == {
        "records": 1,
        "authenticated_records": 1,
        "coverage_rate": 1.0,
    }


def test_authenticated_per_strength_outcomes_enable_registered_robustness_endpoints(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    strength_p_solve = [
        [0.05, 0.10, 0.15, 0.20],
        [0.50, 0.60, 0.70, 0.80],
        [0.20, 0.40, 0.60, 0.70],
    ]
    _rewrite_audit(
        fixture,
        lambda row: row.update(
            strength_p_solve=strength_p_solve,
            strength_success_counts=[
                [round(score * 4000) for score in candidate_scores]
                for candidate_scores in strength_p_solve
            ],
        ),
    )

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    result = json.loads(Path(fixture["output"]).read_text(encoding="utf-8"))
    endpoint = result["secondary_endpoints"]["strength_robustness"]
    assert endpoint["status"] == "available"
    assert endpoint["coverage"] == {
        "records": 1,
        "authenticated_records": 1,
        "coverage_rate": 1.0,
    }
    original = endpoint["budgets"]["1.00x"]["selectors"]["original"]
    assert original["mean_worst_case_p_solve"] == 0.5
    assert original["mean_p_solve_spread"] == pytest.approx(0.3)
    assert original["mean_worst_case_p_solve_delta_vs_stock_original"] == 0.0
    assert original["mean_p_solve_spread_delta_vs_stock_original"] == 0.0
    assert original["paired_records"] == 1
    assert original["per_record"][0]["worst_case_p_solve_delta_vs_stock_original"] == 0.0
    full_original = endpoint["full_support"]["selectors"]["original"]
    assert full_original["mean_worst_case_p_solve"] == 0.5
    assert full_original["mean_p_solve_spread"] == pytest.approx(0.3)
    assert endpoint["support"]["full_support_records"] == 1
    assert endpoint["support"]["budget_conditioned_records"] == 1
    assert endpoint["support"]["budget_excluded_records"] == 0


def test_strength_robustness_full_support_does_not_inherit_stock_budget_exclusions() -> None:
    selectors = ["mean", "lcb", "resource", "original", "random"]
    rows = [
        SimpleNamespace(
            input_index=0,
            problem_digest="1" * 64,
            record={"source": "minorminer"},
            original_index=0,
            audit_strength_p_solve=((0.2, 0.4, 0.6, 0.8),),
        ),
        SimpleNamespace(
            input_index=1,
            problem_digest="2" * 64,
            record={"source": "inkdrop"},
            original_index=0,
            audit_strength_p_solve=((0.1, 0.3, 0.5, 0.7),),
        ),
    ]
    full = {
        "eligible_indices": [[0], [0]],
        "selection_indices": {selector: [0, 0] for selector in selectors},
    }
    budget_row = {
        "input_record_indices": [0],
        "selection_indices": {selector: [0] for selector in selectors},
    }
    budget = {
        "primary_record_coverage": {
            "input_records": 2,
            "included_records": 1,
            "excluded_records": 1,
            "coverage_rate": 0.5,
            "exclusion_reasons": {
                "non_minorminer_source": 1,
                "missing_original_reference": 0,
                "infeasible_original_reference": 0,
            },
        },
        "budgets": {
            key: copy.deepcopy(budget_row)
            for key in ("1.00x", "1.10x", "1.25x", "1.50x", "uncapped")
        },
    }
    contract_path = ROOT / "configs" / "quality_v2_paper_audit_v2.json"
    paper_contract = load_paper_audit_contract(contract_path, _sha256(contract_path))
    contract = dict(paper_contract.secondary_endpoints["strength_robustness"])
    contract["_selectors"] = selectors

    endpoint = _secondary_strength_robustness(
        rows,
        full,
        budget,
        contract=contract,
    )

    full_mean = endpoint["full_support"]["selectors"]["mean"]
    budget_mean = endpoint["budgets"]["1.00x"]["selectors"]["mean"]
    assert full_mean["support_records"] == 2
    assert full_mean["mean_worst_case_p_solve"] == pytest.approx(0.15)
    assert full_mean["paired_records"] == 1
    assert full_mean["pairing_excluded_records"] == 1
    assert budget_mean["support_records"] == 1
    assert budget_mean["mean_worst_case_p_solve"] == pytest.approx(0.2)
    assert endpoint["support"]["budget_excluded_records"] == 1


def test_per_strength_outcomes_must_reproduce_registered_max_aggregation(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    _rewrite_audit(
        fixture,
        lambda row: row.update(
            strength_p_solve=[
                [0.05, 0.10, 0.15, 0.19],
                [0.50, 0.60, 0.70, 0.80],
                [0.20, 0.40, 0.60, 0.70],
            ],
            strength_success_counts=[
                [200, 400, 600, 760],
                [2000, 2400, 2800, 3200],
                [800, 1600, 2400, 2800],
            ],
        ),
    )

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "strength_p_solve disagrees with high_read_scores" in completed.stderr


def test_phase_one_emits_atomic_label_free_freeze_and_refuses_audit_paths(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    freeze = Path(fixture["policy_freeze"])
    document = json.loads(freeze.read_text(encoding="utf-8"))
    sidecar_fields = freeze.with_suffix(".sha256").read_text(encoding="utf-8").split()

    assert document["schema"] == "embedbench.quality-v2-label-free-policy-freeze"
    assert document["schema_version"] == 2
    assert document["audit_paths_opened"] is False
    assert document["release_quality_labels_consumed"] is False
    assert sidecar_fields == [_sha256(freeze), freeze.name]

    completed = subprocess.run(
        [*_freeze_command(fixture), "--audit-labels", str(tmp_path / "must-not-open.jsonl")],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "phase freeze refuses --audit-labels" in completed.stderr

    preregistration = subprocess.run(
        [
            *_freeze_command(fixture),
            "--preregistration-manifest",
            str(fixture["preregistration"]),
            "--preregistration-manifest-sha256",
            str(fixture["preregistration_sha"]),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert preregistration.returncode != 0
    assert "phase freeze refuses a paper preregistration manifest" in preregistration.stderr


def test_phase_two_requires_an_external_audit_release_commitment(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    command = _command(fixture)
    start = command.index("--audit-release-manifest")
    del command[start : start + 4]

    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "phase score requires --audit-release-manifest" in completed.stderr


def test_phase_two_requires_the_out_of_band_preregistration_digest(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    command = _command(fixture)
    start = command.index("--preregistration-manifest")
    del command[start : start + 4]

    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "phase score requires --preregistration-manifest" in completed.stderr


def test_phase_two_rejects_self_rehashed_preregistration_before_audit_open(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    preregistration = Path(fixture["preregistration"])
    document = json.loads(preregistration.read_text(encoding="utf-8"))
    document["policy_freezes"][0]["sha256"] = "f" * 64
    tampered = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    preregistration.write_bytes(tampered)
    preregistration.with_suffix(".sha256").write_text(
        f"{hashlib.sha256(tampered).hexdigest()}  {preregistration.name}\n",
        encoding="utf-8",
    )
    forbidden = tmp_path / "must-not-open-audit.jsonl"
    fixture["audits"] = [forbidden] * 64

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "independently supplied SHA-256" in completed.stderr
    assert "must-not-open-audit" not in completed.stderr


def test_phase_two_rejects_nonidentical_registered_freeze_before_audit_open(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    freeze = Path(fixture["policy_freeze"])
    document = json.loads(freeze.read_text(encoding="utf-8"))
    document["records"][0]["exact_feasible"][0] = False
    payload = (
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    freeze.write_bytes(payload)
    fixture["policy_freeze_sha"] = hashlib.sha256(payload).hexdigest()
    freeze.with_suffix(".sha256").write_text(
        f"{fixture['policy_freeze_sha']}  {freeze.name}\n",
        encoding="utf-8",
    )
    forbidden = tmp_path / "must-not-open-audit.jsonl"
    fixture["audits"] = [forbidden] * 64

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "policy-freeze SHA-256 does not match the registered bytes" in completed.stderr
    assert "must-not-open-audit" not in completed.stderr


def test_train_and_validation_records_are_never_preprocessed_or_encoded(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path, poison_development_partitions=True)

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    result = json.loads(Path(fixture["output"]).read_text(encoding="utf-8"))
    assert result["partition_record_counts"] == {"train": 1, "val": 1, "test": 1}
    assert result["n_train_encoded"] == result["n_val_encoded"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update(audit_schema_version=1), "schema version 2"),
        (lambda row: row.update(corpus_sha256="0" * 64), "corpus SHA-256"),
        (lambda row: row.update(candidate_signature="0" * 64), "candidate signature"),
        (lambda row: row.update(high_read_scores=[0.2, 0.8]), "candidate support"),
        (lambda row: row.update(high_read_best=99), "high_read_best is inconsistent"),
        (lambda row: row.update(resource_index=1), "policy indices disagree"),
        (lambda row: row.update(original_index=0), "policy indices disagree"),
        (
            lambda row: row["provenance"].update(reads_per_strength=100),
            "at least 1000 reads_per_strength",
        ),
        (
            lambda row: row["provenance"].update(registered_strength_count=5),
            "registered_strength_count",
        ),
        (
            lambda row: row["provenance"].pop("registered_strength_count"),
            "requires registered_strength_count",
        ),
        (
            lambda row: row["provenance"].update(
                strength_schedule="default_strength_grid(problem, 5)"
            ),
            "strength_schedule",
        ),
    ],
)
def test_paper_evaluation_rejects_unverified_or_incomplete_audit_labels(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    fixture = _write_fixture(tmp_path)
    _rewrite_audit(fixture, mutation)

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert message in completed.stderr


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reads_per_strength", 8000),
        ("num_sweeps", 201),
        ("base_seed", 17),
    ],
)
def test_paper_evaluation_rejects_audit_provenance_outside_frozen_contract(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    fixture = _write_fixture(tmp_path)

    def mutate(row: dict) -> None:
        row["provenance"][field] = value
        if field == "reads_per_strength":
            row["reads"] = value

    _rewrite_audit(fixture, mutate)

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert f"audit_labels.{field}" in completed.stderr


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda row: row["paper_binding"]["audit_contract"].update(schema_version=True),
            "contract binding",
        ),
        (
            lambda row: row["paper_binding"]["selection"].update(sha256="0" * 64),
            "selection binding",
        ),
        (
            lambda row: row["paper_binding"]["audit_source"].update(sha256="0" * 64),
            "audit source",
        ),
        (
            lambda row: row["paper_binding"]["ground_reference"].update(
                status="uncertified_best_known"
            ),
            "certified exact ground reference",
        ),
        (
            lambda row: row["paper_binding"].update(realized_seeds=[[0], [1], [2]]),
            "realized seed schedule",
        ),
    ],
)
def test_paper_evaluation_rejects_stale_or_uncertified_paper_audit_binding(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    fixture = _write_fixture(tmp_path)
    _rewrite_audit(fixture, mutation)

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert message in completed.stderr


def test_paper_evaluation_validates_contract_before_opening_locked_audit_labels(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    contract = json.loads(Path(fixture["audit_contract"]).read_text(encoding="utf-8"))
    contract["evaluation"]["random_seed"] = 1
    drifted = tmp_path / "drifted-contract.json"
    drifted.write_text(json.dumps(contract), encoding="utf-8")
    drifted.with_suffix(".sha256").write_text(
        f"{_sha256(drifted)}  {drifted.name}\n",
        encoding="utf-8",
    )
    fixture["audit_contract"] = drifted
    fixture["audit_contract_sha"] = _sha256(drifted)
    fixture["audit"] = tmp_path / "must-not-be-opened.jsonl"
    fixture["audits"] = [tmp_path / "must-not-be-opened.jsonl"] * 64

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "evaluation.random_seed" in completed.stderr
    assert "must-not-be-opened" not in completed.stderr


def test_paper_evaluation_requires_every_test_row_to_have_full_audit_support(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    Path(fixture["audit"]).write_text("", encoding="utf-8")
    _refresh_audit_manifest(Path(fixture["audit"]))
    _refresh_audit_release(fixture)

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "exactly cover the fixed test identities" in completed.stderr


def test_paper_evaluation_rejects_unused_non_test_audit_rows(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    train_record = _record("problem-train-m", 0.25)
    identity = json.dumps(
        [Path(fixture["corpus"]).name, train_record["instance_id"], train_record["focus"]],
        separators=(",", ":"),
    ).encode()
    shard_index = int.from_bytes(hashlib.sha256(identity).digest(), "big") % 64
    audit = Path(fixture["audits"][shard_index])
    with audit.open("a", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                _audit_row(
                    Path(fixture["corpus"]),
                    train_record,
                    [0.10, 0.20, 0.30],
                    Path(fixture["selection"]),
                    json.loads(Path(fixture["selection"]).read_text(encoding="utf-8")),
                    str(fixture["selection_sha"]),
                    load_paper_audit_contract(
                        Path(fixture["audit_contract"]),
                        str(fixture["audit_contract_sha"]),
                    ),
                )
            )
            + "\n"
        )
    _refresh_audit_manifest(audit)
    _refresh_audit_release(fixture)

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "exactly cover the fixed test identities" in completed.stderr


def test_frozen_checkpoint_digest_and_data_binding_fail_closed(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    fixture["checkpoint_sha"] = "0" * 64

    wrong_bytes = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)
    assert wrong_bytes.returncode != 0
    assert "frozen checkpoint SHA-256" in wrong_bytes.stderr

    fixture = _write_fixture(tmp_path / "replay")
    _rewrite_checkpoint(
        fixture,
        lambda payload: payload["meta"].update(split_sha256="0" * 64),
    )
    replayed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)
    assert replayed.returncode != 0
    assert "checkpoint metadata mismatch" in replayed.stderr


def test_paper_evaluation_rejects_checkpoint_outside_winner_paper_set(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    arbitrary = tmp_path / "arbitrary-copy.pt"
    arbitrary.write_bytes(Path(fixture["checkpoint"]).read_bytes())
    fixture["checkpoint"] = arbitrary

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "winner.paper_checkpoints" in completed.stderr


def test_paper_evaluation_binds_selection_digest_and_data_provenance(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path / "digest")
    from evaluate_quality_v2 import load_selection_artifact

    selection = Path(fixture["selection"])
    original_selection = selection.read_bytes()
    legacy_selection = json.loads(original_selection)
    legacy_selection["schema_version"] = 2
    selection.write_text(json.dumps(legacy_selection), encoding="utf-8")
    legacy_sha = _sha256(selection)
    audit_contract = load_paper_audit_contract(
        fixture["audit_contract"],
        fixture["audit_contract_sha"],
    )
    with pytest.raises(ValueError, match="selection schema version 3"):
        load_selection_artifact(
            selection,
            legacy_sha,
            tmp_path / "digest",
            audit_contract,
        )
    selection.write_bytes(original_selection)
    fixture["selection_sha"] = "0" * 64

    wrong_digest = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)
    assert wrong_digest.returncode != 0
    assert "selection SHA-256" in wrong_digest.stderr

    fixture = _write_fixture(tmp_path / "provenance")
    _rewrite_selection(
        fixture,
        lambda document: document["data_provenance"]["corpus_inputs"][0].update(sha256="0" * 64),
    )

    stale = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)
    assert stale.returncode != 0
    assert "receipt-revalidated frozen grid" in stale.stderr


def test_paper_evaluation_binds_checkpoint_seed_to_registered_entry(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    selection = json.loads(Path(fixture["selection"]).read_text(encoding="utf-8"))
    selected_seed = selection["winner"]["selected_seed"]
    forged_seed = next(seed for seed in selection["registered_seeds"] if seed != selected_seed)
    _rewrite_checkpoint(fixture, lambda payload: payload["meta"].update(seed=forged_seed))

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "checkpoint metadata mismatch" in completed.stderr


def test_paper_evaluation_requires_four_registered_model_seeds(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)

    def keep_three_seeds(document: dict) -> None:
        document["registered_seeds"] = document["registered_seeds"][:3]
        document["winner"]["paper_checkpoints"] = document["winner"]["paper_checkpoints"][:3]

    _rewrite_selection(fixture, keep_three_seeds)

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "receipt-revalidated frozen grid" in completed.stderr


def test_all_80_receipts_are_revalidated_before_release_or_audit_inputs_open(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)

    def keep_three_seeds(document: dict) -> None:
        document["registered_seeds"] = document["registered_seeds"][:3]
        document["winner"]["paper_checkpoints"] = document["winner"]["paper_checkpoints"][:3]

    _rewrite_selection(fixture, keep_three_seeds)
    fixture["corpus"] = tmp_path / "release-input-must-not-open.jsonl"
    fixture["audits"] = [
        tmp_path / f"audit-input-must-not-open-{index:02d}.jsonl" for index in range(64)
    ]

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "receipt-revalidated frozen grid" in completed.stderr
    assert "release-input-must-not-open" not in completed.stderr
    assert "audit-input-must-not-open" not in completed.stderr


def test_record_keyed_random_baseline_is_independent_of_record_order() -> None:
    from evaluate_quality_v2 import _record_keyed_random_choice

    rows = [SimpleNamespace(record_key=key) for key in ("a" * 64, "b" * 64, "c" * 64)]

    forward = {
        row.record_key: _record_keyed_random_choice(
            row,
            [0, 2, 5],
            namespace="budget:1.50x",
            random_seed=260906,
        )
        for row in rows
    }
    reversed_order = {
        row.record_key: _record_keyed_random_choice(
            row,
            [0, 2, 5],
            namespace="budget:1.50x",
            random_seed=260906,
        )
        for row in reversed(rows)
    }

    assert forward == reversed_order


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--partition", "val"),
        ("--evaluation-mode", "development"),
        ("--arch", "gin"),
        ("--top-tolerance", "0.03"),
        ("--random-seed", "1"),
        ("--device", "cuda"),
    ],
)
def test_paper_entry_point_has_no_unlocked_or_architecture_selection_mode(
    tmp_path: Path,
    flag: str,
    value: str,
) -> None:
    fixture = _write_fixture(tmp_path)

    completed = subprocess.run(
        [*_command(fixture), flag, value],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0


def test_paper_selection_is_independent_of_audit_labels_until_scoring(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    first = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    first_result = json.loads(Path(fixture["output"]).read_text(encoding="utf-8"))

    def replace_quality_labels(row: dict) -> None:
        row["high_read_scores"] = [0.95, 0.10, 0.05]
        row["high_read_best"] = 0
        row["strength_p_solve"] = [[0.95] * 4, [0.10] * 4, [0.05] * 4]
        row["strength_success_counts"] = [[3800] * 4, [400] * 4, [200] * 4]

    _rewrite_audit(fixture, replace_quality_labels)
    second = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    second_result = json.loads(Path(fixture["output"]).read_text(encoding="utf-8"))

    first_indices = first_result["full_support_metrics"]["selectors"]["mean"]["selection_indices"]
    second_indices = second_result["full_support_metrics"]["selectors"]["mean"]["selection_indices"]
    assert first_indices == second_indices
    assert (
        first_result["full_support_metrics"]["selectors"]["mean"]["mean_p_solve"]
        != second_result["full_support_metrics"]["selectors"]["mean"]["mean_p_solve"]
    )


def test_self_rehashed_audit_cannot_cross_external_release_commitment(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)

    def forge_quality_labels(row: dict) -> None:
        row["high_read_scores"] = [0.99, 0.0, 0.01]
        row["high_read_best"] = 0

    _rewrite_audit(fixture, forge_quality_labels, recommit_release=False)
    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "differs from the external release commitment" in completed.stderr


def test_all_64_shard_manifests_are_validated_before_any_audit_payload_opens(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    first_audit = Path(fixture["audits"][0])
    first_audit.write_bytes(b"this is deliberately not JSON\n")
    first_audit.with_name(f"{first_audit.name}.manifest.json").unlink()

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "invalid paper audit shard manifest" in completed.stderr
    assert "JSONDecodeError" not in completed.stderr


def test_paper_evaluator_accepts_registered_ferromagnetic_min_cut_certificate(
    tmp_path: Path,
) -> None:
    from rescore_quality import _certified_ground_reference

    fixture = _write_fixture(tmp_path)
    record = _record("problem-test-m", 2.25)
    ground = _certified_ground_reference(record, {}, exhaustive_max_n=0)
    assert ground["method"] == "ferromagnetic_s_t_min_cut"
    _rewrite_audit(
        fixture,
        lambda row: row["paper_binding"].update(ground_reference=ground),
    )

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    result = json.loads(Path(fixture["output"]).read_text(encoding="utf-8"))
    assert result["evaluated_records"][0]["ground_reference"] == ground
    assert result["ground_reference_coverage"]["by_format"] == {
        "legacy_exhaustive_enumeration_v1_1_v1_2": 0,
        "legacy_ferromagnetic_s_t_min_cut_v1_1_v1_2": 1,
        "wp6_exact_enumeration_v1": 0,
        "wp6_planted_proof_v1": 0,
    }


def test_paper_evaluator_accepts_inline_replayable_wp6_ground_certificate(
    tmp_path: Path,
) -> None:
    from embedbench.ground_certificate import (
        CheckerRun,
        IsingProblem,
        SolverRun,
        build_exact_enumeration_certificate,
    )

    fixture = _write_fixture(tmp_path)
    problem = IsingProblem(
        variables=(0, 2),
        linear=((0, 2.25), (2, 0.0)),
        quadratic=((0, 2, -1.0),),
    )
    bundle = build_exact_enumeration_certificate(
        problem,
        solver=SolverRun(
            name="embedbench-gray-code-enumerator",
            version="1",
            command=("python3", "certify_ground_state.py", "problem.json"),
            seed=None,
            deterministic_work_limit=4,
            safety_timeout_seconds=None,
        ),
        checker=CheckerRun(
            source_sha256="1" * 64,
            environment_sha256="2" * 64,
            command=("python3", "verify_ground_state.py", "proof.json"),
            exit_code=0,
            log_sha256="3" * 64,
        ),
    )
    ground = {
        "certificate": bundle.certificate.to_dict(),
        "proof": bundle.proof.to_dict(),
        "certificate_sha256": bundle.certificate.digest,
        "proof_artifact_sha256": bundle.proof.digest,
    }
    _rewrite_audit(
        fixture,
        lambda row: row["paper_binding"].update(ground_reference=ground),
    )

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    result = json.loads(Path(fixture["output"]).read_text(encoding="utf-8"))
    assert result["evaluated_records"][0]["ground_reference"] == ground


def test_paper_evaluator_rejects_tampered_min_cut_certificate(tmp_path: Path) -> None:
    from rescore_quality import _certified_ground_reference

    fixture = _write_fixture(tmp_path)
    record = _record("problem-test-m", 2.25)
    ground = _certified_ground_reference(record, {}, exhaustive_max_n=0)
    ground["certificate"]["cut_value_scaled"] += 1
    _rewrite_audit(
        fixture,
        lambda row: row["paper_binding"].update(ground_reference=ground),
    )

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "not globally minimal" in completed.stderr


def test_registered_budget_uses_complete_embedding_qmm_not_release_focus_q(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    result = json.loads(Path(fixture["output"]).read_text(encoding="utf-8"))
    budgets = result["budget_sweep"]["budgets"]
    # Frozen chain size is one and stock minorminer focus chain size is one: Q_MM=2.
    assert budgets["1.00x"]["reference_total_qubits"] == [2]
    assert budgets["1.00x"]["budgets"] == [2]
    assert budgets["1.00x"]["maximum_eligible_candidates"] == 2
    assert budgets["1.50x"]["budgets"] == [3]
    assert budgets["1.50x"]["maximum_eligible_candidates"] == 3


@pytest.mark.parametrize("arch", ["mpnn", "hetero"])
def test_paper_evaluator_reconstructs_checkpoint_encoder_without_arch_cli(
    tmp_path: Path,
    arch: str,
) -> None:
    fixture = _write_fixture(tmp_path, arch=arch)

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    result = json.loads(Path(fixture["output"]).read_text(encoding="utf-8"))
    assert result["checkpoint"]["architecture"] == arch
    expected_encoder = "heterogeneous" if arch == "hetero" else "chain"
    assert result["preprocessing"]["encoder"] == expected_encoder


def test_paper_evaluator_applies_signed_deploy_view_contract_to_test_only(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(
        tmp_path,
        deploy_view=True,
        poison_development_partitions=True,
    )

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    result = json.loads(Path(fixture["output"]).read_text(encoding="utf-8"))
    assert result["preprocessing"]["deploy_view"] is True
    assert result["deployment_host_contract"]["mode"] == ("pristine_topology_reconstruction")
    assert result["n_train_encoded"] == result["n_val_encoded"] == 0


def test_paper_evaluator_rejects_v1_checkpoint_loader_misuse(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)

    def replace_with_v1(payload: dict) -> None:
        payload.clear()
        payload.update({"state": {}, "meta": {"arch": "mpnn"}})

    _rewrite_checkpoint(fixture, replace_with_v1)

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "quality V2 checkpoint" in completed.stderr


def test_paper_evaluator_rejects_legacy_split_and_unlocked_checkpoint(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path / "legacy")
    split_path = Path(fixture["splits"])
    document = json.loads(split_path.read_text(encoding="utf-8"))
    document.pop("schema")
    document.pop("schema_version")
    split_path.write_text(json.dumps(document), encoding="utf-8")
    legacy = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)
    assert legacy.returncode != 0
    assert "training data provenance" in legacy.stderr

    fixture = _write_fixture(tmp_path / "unlocked")
    _rewrite_checkpoint(
        fixture,
        lambda payload: payload["meta"].update(test_evaluated=True),
    )
    unlocked = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)
    assert unlocked.returncode != 0
    assert "checkpoint metadata mismatch" in unlocked.stderr


@pytest.mark.parametrize("failure", ["missing", "defective"])
def test_exact_paper_masks_require_verified_pristine_generator_manifests(
    tmp_path: Path,
    failure: str,
) -> None:
    fixture = _write_fixture(tmp_path)
    manifest = Path(f"{fixture['corpus']}.manifest.json")
    if failure == "missing":
        manifest.unlink()
    else:
        document = json.loads(manifest.read_text(encoding="utf-8"))
        document["config"]["defect_couplers"] = 0.01
        manifest.write_text(json.dumps(document), encoding="utf-8")

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert "exact host" in completed.stderr


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda meta: meta.update(registered_budget_ratios=[1.0, None]),
            "registered_budget_ratios",
        ),
        (
            lambda meta: meta.update(auxiliary_heads_used_for_selection=True),
            "auxiliary_heads_used_for_selection",
        ),
        (lambda meta: meta.pop("lcb_z"), "lcb_z"),
        (
            lambda meta: meta["preprocessing"].update(neighbour_feats=True),
            "preprocessing",
        ),
        (
            lambda meta: meta.update(
                training_scope={"limit": 128, "full_fixed_train_validation": False}
            ),
            "training_scope",
        ),
        (lambda meta: meta.pop("training_run_id"), "training_run_id"),
        (lambda meta: meta.update(best_epoch=0), "best_epoch"),
        (
            lambda meta: meta.update(selected_validation_metric=float("nan")),
            "selected validation metric",
        ),
    ],
)
def test_checkpoint_policy_and_preprocessing_contracts_fail_closed(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    fixture = _write_fixture(tmp_path)
    _rewrite_checkpoint(fixture, lambda payload: mutation(payload["meta"]))

    completed = subprocess.run(_command(fixture), cwd=ROOT, capture_output=True, text=True)

    assert completed.returncode != 0
    assert message in completed.stderr


def test_paper_evaluator_requires_every_corpus_committed_by_fixed_split(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="every corpus input"):
        _write_fixture(tmp_path, include_omitted_split_corpus=True)
