from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RESCORE_SCRIPT = ROOT / "scripts" / "rescore_quality.py"
sys.path.insert(0, str(ROOT / "scripts"))

from quality_v2_paper_contract import (  # noqa: E402
    AUDIT_SOURCE_HASH_ALGORITHM,
    CORE_RUNTIME_DISTRIBUTIONS,
    EXPECTED_PAPER_AUDIT_CONTRACT,
    PAPER_AUDIT_ENTRYPOINTS,
    PAPER_VERIFIER_PACKAGE_FILES,
    audit_source_files,
    audit_source_sha256,
    build_paper_preregistration,
    load_paper_audit_contract,
    load_paper_preregistration,
    strict_json_loads,
    validate_runtime_provenance,
    verify_preregistered_policy_freezes,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_minimal_paper_verifier(root: Path) -> None:
    for relative in PAPER_VERIFIER_PACKAGE_FILES:
        path = root / "scripts" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def _preregistration_fixture(tmp_path: Path):
    contract_path = ROOT / "configs" / "quality_v2_paper_audit_v2.json"
    contract = load_paper_audit_contract(contract_path, _sha256(contract_path))
    seeds = [0, 1, 2, 3]
    checkpoints = [
        {
            "seed": seed,
            "cell_id": f"cell-{seed}",
            "checkpoint": f"runs/cell-{seed}.pt",
            "checkpoint_sha256": f"{seed + 1:x}" * 64,
        }
        for seed in seeds
    ]
    selection = {
        "schema": "embedbench.training-grid-selection",
        "schema_version": 3,
        "grid_id": "isingfold-quality-value-v2-screen",
        "grid_path": "configs/training_grid_quality_v2.json",
        "grid_sha256": contract.training_registration["grid_sha256"],
        "source_sha256": contract.training_registration["source_sha256"],
        "selector_sha256": "a" * 64,
        "stage": "quality_value_v2_screen",
        "registered_cell_count": 80,
        "registered_seeds": seeds,
        "audit_source_hash_algorithm": AUDIT_SOURCE_HASH_ALGORITHM,
        "audit_source_sha256": audit_source_sha256(ROOT),
        "winner": {
            "params": {"arch": "mpnn", "objective_variant": "full"},
            "paper_checkpoints": checkpoints,
        },
    }
    selection_path = tmp_path / "selection.json"
    selection_payload = (json.dumps(selection, sort_keys=True) + "\n").encode()
    selection_path.write_bytes(selection_payload)
    selection_sha256 = hashlib.sha256(selection_payload).hexdigest()
    freezes = []
    for checkpoint in checkpoints:
        freeze = {
            "schema": "embedbench.quality-v2-label-free-policy-freeze",
            "schema_version": 2,
            "phase": "label_free_policy_freeze",
            "audit_contract": contract.public_binding(),
            "selection": {
                "file": selection_path.name,
                "sha256": selection_sha256,
                "artifact_schema": selection["schema"],
                "artifact_schema_version": selection["schema_version"],
                "grid_id": selection["grid_id"],
                "grid_sha256": selection["grid_sha256"],
                "source_sha256": selection["source_sha256"],
                "selector_sha256": selection["selector_sha256"],
                "stage": selection["stage"],
                "registered_seeds": seeds,
                "winner_params": selection["winner"]["params"],
                "paper_checkpoint": checkpoint,
            },
            "checkpoint": {
                "seed": checkpoint["seed"],
                "file": Path(checkpoint["checkpoint"]).name,
                "sha256": checkpoint["checkpoint_sha256"],
            },
            "audit_paths_opened": False,
            "release_quality_labels_consumed": False,
        }
        path = tmp_path / f"policy-freeze-{checkpoint['seed']}.json"
        path.write_bytes((json.dumps(freeze, sort_keys=True) + "\n").encode())
        freezes.append(path)
    document, payload = build_paper_preregistration(
        audit_contract=contract,
        selection_document=selection,
        selection_file=selection_path.name,
        selection_sha256=selection_sha256,
        audit_source_root=ROOT,
        policy_freezes=freezes,
    )
    preregistration_path = tmp_path / "paper-preregistration.json"
    preregistration_path.write_bytes(payload)
    return (
        contract,
        selection,
        selection_path,
        selection_sha256,
        freezes,
        preregistration_path,
        document,
        payload,
    )


def test_preregistration_manifest_binds_all_four_freezes_and_loads_from_external_sha(
    tmp_path: Path,
) -> None:
    (
        contract,
        selection,
        selection_path,
        selection_sha256,
        freezes,
        preregistration_path,
        document,
        payload,
    ) = _preregistration_fixture(tmp_path)
    expected_sha256 = hashlib.sha256(payload).hexdigest()

    loaded = load_paper_preregistration(
        preregistration_path,
        expected_sha256,
        audit_contract=contract,
        selection_document=selection,
        selection_file=selection_path.name,
        selection_sha256=selection_sha256,
        audit_source_root=ROOT,
    )
    verified = verify_preregistered_policy_freezes(loaded, freezes)

    assert loaded.document == document
    assert [entry["seed"] for entry in loaded.document["policy_freezes"]] == [0, 1, 2, 3]
    assert set(verified) == {0, 1, 2, 3}
    assert loaded.public_binding()["trust_model"] == (
        "sha256_supplied_out_of_band_before_audit_label_generation"
    )


def test_preregistration_manifest_rejects_self_rehash_and_colocated_sidecar_tampering(
    tmp_path: Path,
) -> None:
    (
        contract,
        selection,
        selection_path,
        selection_sha256,
        _,
        preregistration_path,
        document,
        payload,
    ) = _preregistration_fixture(tmp_path)
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    document["policy_freezes"][0]["sha256"] = "f" * 64
    tampered = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
    preregistration_path.write_bytes(tampered)
    preregistration_path.with_suffix(".sha256").write_text(
        f"{hashlib.sha256(tampered).hexdigest()}  {preregistration_path.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="independently supplied SHA-256"):
        load_paper_preregistration(
            preregistration_path,
            expected_sha256,
            audit_contract=contract,
            selection_document=selection,
            selection_file=selection_path.name,
            selection_sha256=selection_sha256,
            audit_source_root=ROOT,
        )


def test_preregistration_rejects_self_rehashed_policy_freeze(
    tmp_path: Path,
) -> None:
    (
        contract,
        selection,
        selection_path,
        selection_sha256,
        freezes,
        preregistration_path,
        _,
        payload,
    ) = _preregistration_fixture(tmp_path)
    loaded = load_paper_preregistration(
        preregistration_path,
        hashlib.sha256(payload).hexdigest(),
        audit_contract=contract,
        selection_document=selection,
        selection_file=selection_path.name,
        selection_sha256=selection_sha256,
        audit_source_root=ROOT,
    )
    freeze = freezes[0]
    document = json.loads(freeze.read_text(encoding="utf-8"))
    document["audit_paths_opened"] = True
    tampered = (json.dumps(document, sort_keys=True) + "\n").encode()
    freeze.write_bytes(tampered)
    freeze.with_suffix(".sha256").write_text(
        f"{hashlib.sha256(tampered).hexdigest()}  {freeze.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="do not match the paper preregistration"):
        verify_preregistered_policy_freezes(loaded, freezes)


def test_verifier_source_digest_is_separate_from_frozen_training_source(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in PAPER_AUDIT_ENTRYPOINTS:
        (scripts / name).write_text(f"# {name}\n", encoding="utf-8")
    _write_minimal_paper_verifier(tmp_path)
    training_source = tmp_path / "src" / "embedbench" / "model.py"
    training_source.parent.mkdir(parents=True)
    training_source.write_text("TRAINING = 1\n", encoding="utf-8")

    initial = audit_source_sha256(tmp_path)
    training_source.write_text("TRAINING = 2\n", encoding="utf-8")
    assert audit_source_sha256(tmp_path) == initial

    (scripts / "evaluate_quality_v2.py").write_text("# changed verifier\n", encoding="utf-8")
    assert audit_source_sha256(tmp_path) != initial


def test_audit_source_requires_the_complete_verifier_only_package(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in PAPER_AUDIT_ENTRYPOINTS:
        (scripts / name).write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="verifier package is incomplete"):
        audit_source_sha256(tmp_path)


def test_audit_source_closure_hashes_every_critical_transitive_dependency(
    tmp_path: Path,
) -> None:
    staged = tmp_path / "staged"
    for source in audit_source_files(ROOT):
        relative = source.relative_to(ROOT)
        destination = staged / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    critical = (
        "scripts/evaluate_quality_v2.py",
        "scripts/training_splits.py",
        "scripts/training_artifacts.py",
        "src/embedbench/models_chain.py",
        "src/embedbench/models_quality_v2.py",
        "src/embedbench/surrogate.py",
        "src/embedbench/embedding.py",
    )
    relative_closure = {path.relative_to(staged).as_posix() for path in audit_source_files(staged)}
    assert set(critical) <= relative_closure
    baseline = audit_source_sha256(staged)
    for relative in critical:
        path = staged / relative
        original = path.read_bytes()
        path.write_bytes(original + b"\n# dependency mutation\n")
        assert audit_source_sha256(staged) != baseline, relative
        path.write_bytes(original)


def test_newly_imported_local_dependency_is_discovered_and_hashed(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in PAPER_AUDIT_ENTRYPOINTS:
        payload = "from verifier_helper import VALUE\n" if name == "evaluate_quality_v2.py" else ""
        (scripts / name).write_text(payload, encoding="utf-8")
    _write_minimal_paper_verifier(tmp_path)
    helper = scripts / "verifier_helper.py"
    helper.write_text("from nested_helper import VALUE\n", encoding="utf-8")
    nested = scripts / "nested_helper.py"
    nested.write_text("VALUE = 1\n", encoding="utf-8")

    initial = audit_source_sha256(tmp_path)
    assert helper in audit_source_files(tmp_path)
    assert nested in audit_source_files(tmp_path)
    nested.write_text("VALUE = 2\n", encoding="utf-8")
    assert audit_source_sha256(tmp_path) != initial


def test_audit_source_closure_fails_on_unresolved_local_import(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in PAPER_AUDIT_ENTRYPOINTS:
        payload = "import embedbench.missing\n" if name == "evaluate_quality_v2.py" else ""
        (scripts / name).write_text(payload, encoding="utf-8")
    _write_minimal_paper_verifier(tmp_path)
    package = tmp_path / "src" / "embedbench"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="unresolved local import"):
        audit_source_sha256(tmp_path)


def test_audit_source_closure_fails_on_ambiguous_local_import(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in PAPER_AUDIT_ENTRYPOINTS:
        payload = "import helper\n" if name == "evaluate_quality_v2.py" else ""
        (scripts / name).write_text(payload, encoding="utf-8")
    _write_minimal_paper_verifier(tmp_path)
    (scripts / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    source_helper = tmp_path / "src" / "helper.py"
    source_helper.parent.mkdir()
    source_helper.write_text("VALUE = 2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="ambiguous local import"):
        audit_source_sha256(tmp_path)


@pytest.mark.parametrize(
    "payload",
    [
        'import importlib\nVALUE = importlib.import_module("helper").VALUE\n',
        ('import importlib\nload = importlib.import_module\nVALUE = load("helper").VALUE\n'),
        (
            "import importlib\n"
            "loaders = (importlib.import_module,)\n"
            'VALUE = loaders[0]("helper").VALUE\n'
        ),
        (
            "import importlib\n"
            'load = importlib.__dict__["import_module"]\n'
            'VALUE = load("helper").VALUE\n'
        ),
        (
            "import importlib\n"
            'load = importlib.__dict__.get("import_module")\n'
            'VALUE = load("helper").VALUE\n'
        ),
        (
            "import importlib\n"
            'load = vars(importlib)["import_module"]\n'
            'VALUE = load("helper").VALUE\n'
        ),
        (
            "import sys\n"
            'load = sys.modules["importlib"].import_module\n'
            'VALUE = load("helper").VALUE\n'
        ),
        ('import builtins\nrun = builtins.__dict__["exec"]\nrun("VALUE = 1")\n'),
        ("def pick(module):\n    return module.import_module\nVALUE = pick(None)\n"),
        (
            "def pick(module):\n"
            '    return type(module).__getattribute__(module, "import_module")\n'
            "VALUE = pick(None)\n"
        ),
        ('from operator import attrgetter\nVALUE = attrgetter("import_module")(None)\n'),
        'VALUE = __builtins__["__import__"]("helper").VALUE\n',
        'from importlib import import_module as load\nVALUE = load("helper").VALUE\n',
        ('from importlib.metadata import import_module as load\nVALUE = load("helper").VALUE\n'),
        "from importlib import metadata\nVALUE = metadata.entry_points()\n",
        (
            "from importlib import metadata\n"
            "def pick(module):\n"
            "    return module.entry_points\n"
            "VALUE = pick(metadata)\n"
        ),
        'VALUE = __import__("helper").VALUE\n',
        'import runpy\nVALUE = runpy.run_path("scripts/helper.py")["VALUE"]\n',
    ],
)
def test_audit_source_closure_rejects_dynamic_code_loading(
    tmp_path: Path,
    payload: str,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in PAPER_AUDIT_ENTRYPOINTS:
        source = payload if name == "evaluate_quality_v2.py" else ""
        (scripts / name).write_text(source, encoding="utf-8")
    _write_minimal_paper_verifier(tmp_path)
    (scripts / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="dynamic (import|code loading)"):
        audit_source_sha256(tmp_path)


def test_audit_source_closure_allows_static_importlib_resource_apis(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in PAPER_AUDIT_ENTRYPOINTS:
        payload = (
            "from importlib import metadata, resources\n"
            'VERSION = metadata.version("embedbench")\n'
            "MISSING = metadata.PackageNotFoundError\n"
            'FILES = resources.files("embedbench")\n'
            "AS_FILE = resources.as_file\n"
            if name == "evaluate_quality_v2.py"
            else ""
        )
        (scripts / name).write_text(payload, encoding="utf-8")
    _write_minimal_paper_verifier(tmp_path)

    assert len(audit_source_sha256(tmp_path)) == 64


def test_audit_source_closure_allows_only_named_importlib_resource_members(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in PAPER_AUDIT_ENTRYPOINTS:
        payload = (
            "from importlib.metadata import PackageNotFoundError, version\n"
            "from importlib.resources import as_file, files\n"
            if name == "evaluate_quality_v2.py"
            else ""
        )
        (scripts / name).write_text(payload, encoding="utf-8")
    _write_minimal_paper_verifier(tmp_path)

    assert len(audit_source_sha256(tmp_path)) == 64


@pytest.mark.parametrize(
    ("payload", "namespace"),
    [
        ("import embedbench.missing\n", "embedbench"),
        ("import training_artifacts\n", "training_artifacts"),
    ],
)
def test_audit_source_requires_registered_roots_for_protected_imports(
    tmp_path: Path, payload: str, namespace: str
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in PAPER_AUDIT_ENTRYPOINTS:
        source = payload if name == "evaluate_quality_v2.py" else ""
        (scripts / name).write_text(source, encoding="utf-8")
    _write_minimal_paper_verifier(tmp_path)

    with pytest.raises(ValueError, match=rf"registered local namespace root.*{namespace}"):
        audit_source_sha256(tmp_path)


def test_audit_source_closure_rejects_symlinked_source_file_escape(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in PAPER_AUDIT_ENTRYPOINTS:
        payload = "import helper\n" if name == "evaluate_quality_v2.py" else ""
        (scripts / name).write_text(payload, encoding="utf-8")
    _write_minimal_paper_verifier(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")
    try:
        (scripts / "helper.py").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("source symlinks are not supported on this platform")

    with pytest.raises(ValueError, match="symlink"):
        audit_source_sha256(tmp_path)


def test_audit_source_closure_rejects_a_symlinked_source_root_ancestor(
    tmp_path: Path,
) -> None:
    real_parent = tmp_path / "real-parent"
    source_root = real_parent / "source-root"
    scripts = source_root / "scripts"
    scripts.mkdir(parents=True)
    for name in PAPER_AUDIT_ENTRYPOINTS:
        (scripts / name).write_text("", encoding="utf-8")
    _write_minimal_paper_verifier(source_root)
    alias_parent = tmp_path / "alias-parent"
    try:
        alias_parent.symlink_to(real_parent, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are not supported on this platform")

    with pytest.raises(ValueError, match="symlink.*ancestor|ancestor.*symlink"):
        audit_source_sha256(alias_parent / "source-root")


def test_audit_source_closure_rejects_symlinked_entrypoint(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")
    for name in PAPER_AUDIT_ENTRYPOINTS:
        path = scripts / name
        if name == "evaluate_quality_v2.py":
            try:
                path.symlink_to(outside)
            except (OSError, NotImplementedError):
                pytest.skip("source symlinks are not supported on this platform")
        else:
            path.write_text("", encoding="utf-8")
    _write_minimal_paper_verifier(tmp_path)

    with pytest.raises(ValueError, match="symlink"):
        audit_source_sha256(tmp_path)


def test_audit_source_closure_rejects_symlinked_import_directory(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in PAPER_AUDIT_ENTRYPOINTS:
        (scripts / name).write_text("", encoding="utf-8")
    _write_minimal_paper_verifier(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    try:
        (scripts / "linked_package").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are not supported on this platform")

    with pytest.raises(ValueError, match="symlink"):
        audit_source_sha256(tmp_path)


def test_audit_source_closure_requires_nofollow_directory_descriptors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in PAPER_AUDIT_ENTRYPOINTS:
        (scripts / name).write_text("", encoding="utf-8")
    _write_minimal_paper_verifier(tmp_path)
    import quality_v2_paper_contract as paper_contract

    monkeypatch.setattr(paper_contract, "_SECURE_NOFOLLOW_AVAILABLE", False)
    with pytest.raises(ValueError, match="O_NOFOLLOW"):
        audit_source_sha256(tmp_path)


def test_audit_source_digest_rejects_inventory_drift_during_ast_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in PAPER_AUDIT_ENTRYPOINTS:
        payload = "import helper\n" if name == "evaluate_quality_v2.py" else ""
        (scripts / name).write_text(payload, encoding="utf-8")
    _write_minimal_paper_verifier(tmp_path)
    helper = scripts / "helper.py"
    helper.write_text("VALUE = 1\n", encoding="utf-8")

    import quality_v2_paper_contract as paper_contract

    real_parse = paper_contract.ast.parse
    mutated = False

    def mutate_after_snapshot(*args, **kwargs):
        nonlocal mutated
        result = real_parse(*args, **kwargs)
        if not mutated:
            mutated = True
            helper.write_text("import hidden\nVALUE = 2\n", encoding="utf-8")
            (scripts / "hidden.py").write_text("VALUE = 3\n", encoding="utf-8")
        return result

    monkeypatch.setattr(paper_contract.ast, "parse", mutate_after_snapshot)
    with pytest.raises(ValueError, match="source inventory changed"):
        audit_source_sha256(tmp_path)


def test_audit_source_digest_parses_and_hashes_one_retained_byte_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in PAPER_AUDIT_ENTRYPOINTS:
        payload = "import helper\n" if name == "evaluate_quality_v2.py" else ""
        (scripts / name).write_text(payload, encoding="utf-8")
    _write_minimal_paper_verifier(tmp_path)
    (scripts / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")

    def forbidden_second_path_read(self: Path) -> bytes:
        raise AssertionError(f"audit source was re-read through Path.read_bytes: {self}")

    monkeypatch.setattr(Path, "read_bytes", forbidden_second_path_read)
    digest = audit_source_sha256(tmp_path)

    assert len(digest) == 64


def test_paper_verifier_package_is_in_the_audit_closure() -> None:
    relative_closure = {path.relative_to(ROOT).as_posix() for path in audit_source_files(ROOT)}

    assert {f"scripts/{relative}" for relative in PAPER_VERIFIER_PACKAGE_FILES} <= relative_closure
    assert "src/embedbench/ground_certificate.py" not in relative_closure
    assert "src/embedbench/hard_ood_schema.py" not in relative_closure
    assert "src/embedbench/objective_embedder.py" in relative_closure
    assert all(not relative.endswith(".joblib") for relative in relative_closure)


def test_paper_entrypoints_do_not_expose_objective_embedder_resource_path() -> None:
    for name in PAPER_AUDIT_ENTRYPOINTS:
        if name == "quality_v2_paper_contract.py":
            continue
        payload = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "ObjectiveEmbedder" not in payload
        assert "embedbench.objective_embedder" not in payload
        assert "shortlist_v1.joblib" not in payload
        assert "screen_v1.joblib" not in payload


def test_audit_closure_rejects_a_transitive_objective_embedder_call_path(
    tmp_path: Path,
) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in PAPER_AUDIT_ENTRYPOINTS:
        payload = "import helper\n" if name == "evaluate_quality_v2.py" else ""
        (scripts / name).write_text(payload, encoding="utf-8")
    _write_minimal_paper_verifier(tmp_path)
    (scripts / "helper.py").write_text(
        "from embedbench.objective_embedder import ObjectiveEmbedder\nVALUE = ObjectiveEmbedder\n",
        encoding="utf-8",
    )
    package = tmp_path / "src" / "embedbench"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "objective_embedder.py").write_text(
        "class ObjectiveEmbedder:\n    pass\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="packaged learned-model resources"):
        audit_source_sha256(tmp_path)


def test_importing_paper_entrypoints_does_not_load_objective_embedder_resources() -> None:
    program = """
import importlib
import sys
for module in (
    "quality_v2_paper_contract",
    "select_training_grid",
    "evaluate_quality_v2",
    "rescore_quality",
    "aggregate_quality_v2_paper",
):
    importlib.import_module(module)
assert "embedbench.objective_embedder" not in sys.modules
assert "joblib" not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env={"PYTHONPATH": f"{ROOT / 'src'}:{ROOT / 'scripts'}"},
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "payload",
    [
        b'{"field":1,"field":2}',
        b'{"field":NaN}',
        b'{"field":1e999}',
        b'{"field":9223372036854775808}',
    ],
)
def test_strict_json_rejects_duplicate_nonfinite_and_overflow_values(payload: bytes) -> None:
    with pytest.raises(ValueError, match="strict JSON"):
        strict_json_loads(payload, location="adversarial fixture")


def test_registered_paper_audit_contract_freezes_every_stochastic_choice() -> None:
    path = ROOT / "configs" / "quality_v2_paper_audit_v2.json"
    checksum_path = ROOT / "configs" / "quality_v2_paper_audit_v2.sha256"
    registered_sha256, registered_file = checksum_path.read_text(encoding="utf-8").split()

    assert registered_file == path.name
    assert registered_sha256 == _sha256(path)
    contract = load_paper_audit_contract(path, registered_sha256)

    assert contract.document == EXPECTED_PAPER_AUDIT_CONTRACT
    assert contract.document["schema_version"] == 2
    assert contract.document["protocol_id"] == "quality-v2-paper-audit-v2"
    assert contract.secondary_endpoints["selection_role"] == (
        "report_only_after_label_free_policy_freeze"
    )
    assert contract.secondary_endpoints["strength_robustness"]["registered_strength_count"] == 4
    assert contract.document["audit_labels"] == {
        "schema": "embedbench.independent-quality-audit",
        "schema_version": 2,
        "generator": "EmbedBench/scripts/rescore_quality.py",
        "reads_per_strength": 4000,
        "num_sweeps": 200,
        "base_seed": 0,
        "seed_schedule": ("(base_seed + 7919*candidate_index + 31*strength_index) % 2**31"),
        "registered_strength_count": 4,
        "strength_schedule": "default_strength_grid(problem, 4)",
        "aggregation": "max_p_solve_over_strength_schedule",
        "score_evidence": "integer_success_counts_per_candidate_strength",
        "probability_derivation": "success_count_divided_by_reads_per_strength_binary64",
    }
    assert contract.document["audit_execution"] == {
        "shard_count": 64,
        "assignment": "sha256(canonical_audit_identity) mod shard_count",
        "canonical_audit_identity": "json_compact_utf8_array[file,instance_id,focus]",
        "record_output": "atomic_recomputed_one_record_per_audit_identity_no_partial_reuse",
        "merge_coverage": "exactly_once_complete_fixed_test_partition",
        "merge_order": "canonical_audit_identity_lexicographic",
        "release_commitment": "exact_jsonl_and_shard_manifest_bytes",
        "release_trust_anchor": "sha256_registered_out_of_band_before_phase_2",
    }
    assert contract.document["evaluation"] == {
        "provisional": True,
        "evaluation_design": "exploratory_legacy_iid",
        "legacy_test_exposure": ("membership_problem_sizes_and_e0_inspected_without_audit_p_solve"),
        "record_support": "complete_fixed_test_partition",
        "candidate_support": "full",
        "support_uses_labels": False,
        "ground_reference_policy": "certified_exact_only",
        "uncertified_ground_reference_action": "reject",
        "top_tolerance": 0.02,
        "random_seed": 0,
        "random_baseline_policy": "sha256_record_key_modulo_eligible_v1",
        "lcb_z": 1.0,
        "device": "cpu",
        "runtime_environment_schema": "embedbench.runtime-environment/v1-complete",
        "runtime_consistency": "exact_across_model_seed_evaluations",
        "selection_statistic": "mean",
        "reported_selection_statistics": ["mean", "lcb"],
        "budget_contract": "B/Q_MM",
        "budget_reference": "stock_minorminer_original_total_qubits",
        "primary_record_policy": "valid_stock_minorminer_original_reference_only",
        "exact_mask": "minor_embedding_feasible_and_full_embedding_total_qubits",
        "registered_budget_ratios": [1.0, 1.1, 1.25, 1.5, None],
        "primary_metric": "mean_finite_budget_regret",
        "accepted_ground_references": {
            "embedded_ground_state_certificate_v1": {
                "format_version": 1,
                "envelope_fields": [
                    "certificate",
                    "proof",
                    "certificate_sha256",
                    "proof_artifact_sha256",
                ],
                "certificate_schema": "embedbench.ground-state-certificate",
                "certificate_schema_version": 1,
                "accepted_statuses": ["planted_proof", "exact_enumeration"],
                "certified_optimal_action": "reject_until_executable_proof_checker",
                "artifact_storage": "embedded_closed_json_envelope",
            },
            "exhaustive_enumeration": {
                "format_version": "legacy_v1_1_v1_2",
                "certificate": "not_required",
                "max_variables": 22,
                "batch_states": 131072,
                "enumeration": "all_spin_assignments_vectorized_float64_v1",
            },
            "ferromagnetic_s_t_min_cut": {
                "format_version": "legacy_v1_1_v1_2",
                "certificate": "required",
                "algorithm": "networkx_preflow_push_integer_capacities_v1",
            },
        },
    }
    assert contract.document["selection"] == {
        "artifact_schema": "embedbench.training-grid-selection",
        "artifact_schema_version": 3,
        "grid_id": "isingfold-quality-value-v2-screen",
        "grid_path": "configs/training_grid_quality_v2.json",
        "stage": "quality_value_v2_screen",
        "registered_cell_count": 80,
        "registered_seeds": [0, 1, 2, 3],
        "validation_only": True,
    }
    assert contract.training_registration["validation_replay"] == {
        "partition": "val",
        "cell_coverage": "exactly_all_80_registered_frozen_checkpoints",
        "metric": "mean_finite_budget_regret",
        "evidence": ("per_record_problem_digest_exact_masks_total_qubits_qmm_budgets_and_choices"),
        "authoritative_cross_cell_selection_source": ("fresh_canonical_cpu_checkpoint_replay"),
        "stored_training_validation_role": (
            "required_training_provenance_and_best_epoch_binding_only"
        ),
        "cross_device_model_dependent_comparison": (
            "record_only_never_reject_rank_exclude_or_break_ties"
        ),
        "model_independent_replay_action": "reject_unless_exact_match",
        "test_access": "forbidden_until_all_80_replays_and_selection_are_frozen",
        "canonical_runtime": {
            "schema": "embedbench.canonical-validation-replay-policy",
            "schema_version": 1,
            "device": "cpu",
            "model_parameter_dtype": "float32",
            "intraop_threads": 1,
            "interop_threads": 1,
            "deterministic_algorithms": True,
            "float32_matmul_precision": "highest",
            "selection_metric_source": "canonical_cpu_checkpoint_replay",
            "training_device_metric_role": (
                "authenticated_training_time_best_epoch_provenance_only_never_cross_cell_selection"
            ),
            "cross_device_sweep_equality_required": False,
        },
    }
    assert contract.document["model_seed_aggregation"] == {
        "registered_seed_count": 4,
        "seed_source": "validation_selection.registered_seeds",
        "coverage": "exactly_every_registered_seed_checkpoint_once",
        "aggregation": "arithmetic_mean",
        "dispersion": "sample_standard_deviation_ddof_1",
    }
    assert contract.document["secondary_endpoint_inference"] == {
        "selection_role": "report_only_never_selection_or_primary_pass_fail",
        "selector": "mean",
        "reference": "stock_minorminer_original",
        "cluster_unit": "quality_problem_digest",
        "model_seed_reduction": "arithmetic_mean_within_record",
        "record_reduction": "arithmetic_mean_within_problem_digest",
        "missing_pair_action": "unavailable_or_partial_with_explicit_coverage",
        "bootstrap": {
            "method": "percentile_problem_cluster_bootstrap",
            "resamples": 10000,
            "confidence_level": 0.95,
            "seed": 260912,
            "quantile_method": "linear",
            "namespace_seed": "sha256_first_8_bytes_big_endian_added_to_seed",
            "minimum_problem_clusters": 20,
            "interval_role": "pointwise_descriptive_not_hypothesis_test",
            "inference_scope": "conditional_on_four_registered_model_checkpoints",
        },
        "endpoints": {
            "residual_connectivity": {
                "supports": ["1.00x", "1.10x", "1.25x", "1.50x", "uncapped"],
                "metrics": [
                    "remaining_free_qubits_delta_vs_stock_original",
                    "largest_remaining_component_delta_vs_stock_original",
                    "largest_component_fraction_delta_vs_stock_original",
                    "remaining_component_count_delta_vs_stock_original",
                ],
            },
            "strength_robustness": {
                "supports": [
                    "full_support",
                    "1.00x",
                    "1.10x",
                    "1.25x",
                    "1.50x",
                    "uncapped",
                ],
                "metrics": [
                    "worst_case_p_solve_delta_vs_stock_original",
                    "p_solve_spread_delta_vs_stock_original",
                ],
            },
        },
    }
    assert contract.document["secondary_inference"] == {
        "status": "secondary_distribution_free_bounded_mean_inference",
        "estimand": ("problem_equal_weighted_p_solve_delta_learned_mean_minus_stock_original"),
        "cluster_unit": "quality_problem_digest",
        "model_seed_reduction": "arithmetic_mean_within_record",
        "record_reduction": "arithmetic_mean_within_problem_digest",
        "inference_scope": "conditional_on_four_registered_model_checkpoints",
        "sampling_assumption": "independent_problem_clusters",
        "mean_inference": {
            "method": "two_sided_hoeffding_confidence_interval",
            "bounded_support": [-1.0, 1.0],
            "confidence_level": 0.95,
            "minimum_problem_clusters": 20,
        },
        "paired_test": {
            "method": "one_sided_hoeffding_bounded_mean_test",
            "null": "population_mean_p_solve_delta_less_than_or_equal_to_zero",
            "alternative": "learned_mean_p_solve_greater_than_stock_original",
        },
        "per_budget_multiplicity": {
            "family": "four_finite_registered_budgets",
            "method": "holm_bonferroni",
            "familywise_alpha": 0.05,
        },
    }


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("audit_labels", "reads_per_strength", 8000),
        ("audit_labels", "num_sweeps", 201),
        ("audit_labels", "base_seed", 17),
        ("audit_labels", "registered_strength_count", 5),
        ("evaluation", "top_tolerance", 0.03),
        ("evaluation", "random_seed", 1),
        ("evaluation", "candidate_support", "proposal"),
        ("evaluation", "registered_budget_ratios", [1.0, None]),
        ("evaluation", "selection_statistic", "lcb"),
        ("evaluation", "lcb_z", 2.0),
        ("evaluation", "provisional", False),
        ("model_seed_aggregation", "registered_seed_count", 3),
        ("preregistration", "required_policy_freeze_count", 3),
        ("preregistration", "colocated_checksum_sidecar_authoritative", True),
        ("secondary_endpoint_inference", "selector", "lcb"),
        ("selection", "registered_cell_count", 79),
        ("selection", "registered_seeds", [0, 1, 2]),
    ],
)
def test_paper_audit_contract_rejects_any_protocol_drift(
    tmp_path: Path,
    section: str,
    field: str,
    value: object,
) -> None:
    document = json.loads(json.dumps(EXPECTED_PAPER_AUDIT_CONTRACT))
    document[section][field] = value
    path = tmp_path / "quality_v2_paper_audit_v2.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    path.with_suffix(".sha256").write_text(
        f"{_sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=f"{section}.{field}"):
        load_paper_audit_contract(path, _sha256(path))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("authoritative_cross_cell_selection_source",), "stored_training_validation"),
        (("stored_training_validation_role",), "cross_cell_selection"),
        (("cross_device_model_dependent_comparison",), "reject_on_difference"),
        (("model_independent_replay_action",), "record_only"),
        (("canonical_runtime", "device"), "cuda"),
        (("canonical_runtime", "interop_threads"), 2),
        (
            ("canonical_runtime", "training_device_metric_role"),
            "authenticated_checkpoint_generation_diagnostic_only",
        ),
    ],
)
def test_paper_audit_contract_rejects_canonical_replay_protocol_drift(
    tmp_path: Path,
    path: tuple[str, ...],
    value: object,
) -> None:
    document = json.loads(json.dumps(EXPECTED_PAPER_AUDIT_CONTRACT))
    target = document["training_registration"]["validation_replay"]
    for component in path[:-1]:
        target = target[component]
    target[path[-1]] = value
    contract_path = tmp_path / "quality_v2_paper_audit_v2.json"
    contract_path.write_text(json.dumps(document), encoding="utf-8")
    contract_path.with_suffix(".sha256").write_text(
        f"{_sha256(contract_path)}  {contract_path.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="training_registration.validation_replay"):
        load_paper_audit_contract(contract_path, _sha256(contract_path))


def test_paper_audit_contract_rejects_digest_mismatch_before_use() -> None:
    path = ROOT / "configs" / "quality_v2_paper_audit_v2.json"

    with pytest.raises(ValueError, match="SHA-256"):
        load_paper_audit_contract(path, "0" * 64)


def test_data_specific_contract_is_loaded_only_from_authenticated_temporary_bytes(
    tmp_path: Path,
) -> None:
    document = json.loads(json.dumps(EXPECTED_PAPER_AUDIT_CONTRACT))
    document["training_registration"]["grid_sha256"] = "a" * 64
    document["training_registration"]["source_sha256"] = "b" * 64
    path = tmp_path / "quality_v2_paper_audit_test.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    digest = _sha256(path)
    path.with_suffix(".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )

    contract = load_paper_audit_contract(path, digest)

    assert contract.path == path.resolve()
    assert contract.sha256 == digest
    assert contract.training_registration["grid_sha256"] == "a" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        load_paper_audit_contract(path, "c" * 64)


def test_data_specific_contract_rejects_malformed_training_registration(
    tmp_path: Path,
) -> None:
    document = json.loads(json.dumps(EXPECTED_PAPER_AUDIT_CONTRACT))
    document["training_registration"]["source_sha256"] = "not-a-digest"
    path = tmp_path / "quality_v2_paper_audit_test.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    digest = _sha256(path)
    path.with_suffix(".sha256").write_text(
        f"{digest}  {path.name}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="training_registration.source_sha256"):
        load_paper_audit_contract(path, digest)


def test_paper_audit_contract_requires_its_registered_checksum_sidecar(
    tmp_path: Path,
) -> None:
    source = ROOT / "configs" / "quality_v2_paper_audit_v2.json"
    path = tmp_path / source.name
    path.write_bytes(source.read_bytes())

    with pytest.raises(ValueError, match="checksum sidecar"):
        load_paper_audit_contract(path, _sha256(path))

    sidecar = path.with_suffix(".sha256")
    sidecar.write_text(f"{'0' * 64}  {path.name}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum sidecar"):
        load_paper_audit_contract(path, _sha256(path))


def test_runtime_provenance_requires_the_complete_registered_shape() -> None:
    runtime = {
        "schema": "embedbench.runtime-environment",
        "schema_version": 1,
        "python": {
            "implementation": "CPython",
            "version": "3.12.0",
            "executable": "/usr/bin/python3",
        },
        "torch": {
            "version": "2.7.0",
            "cuda_available": False,
            "cuda_runtime_version": None,
            "cudnn_version": None,
        },
        "device": {"selected": "cpu", "type": "cpu", "index": None, "gpu": None},
        "packages": {name: None for name in CORE_RUNTIME_DISTRIBUTIONS},
    }

    assert validate_runtime_provenance(runtime, location="test", expected_device="cpu") == runtime
    incomplete = json.loads(json.dumps(runtime))
    incomplete["python"].pop("executable")
    with pytest.raises(ValueError, match="complete runtime provenance"):
        validate_runtime_provenance(incomplete, location="test", expected_device="cpu")
    wrong_type = json.loads(json.dumps(runtime))
    wrong_type["schema_version"] = True
    with pytest.raises(ValueError, match="complete runtime provenance"):
        validate_runtime_provenance(wrong_type, location="test", expected_device="cpu")


def test_locked_test_rescore_requires_contract_before_opening_inputs(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(RESCORE_SCRIPT),
            str(tmp_path / "must-not-be-opened.jsonl"),
            "--splits",
            str(tmp_path / "must-not-be-opened-splits.json"),
            "--split",
            "test",
            "--out",
            str(tmp_path / "audit.jsonl"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "locked test rescoring requires --audit-contract" in completed.stderr
    assert "No such file" not in completed.stderr


@pytest.mark.parametrize("omitted", ["splits", "split"])
def test_rescore_requires_explicit_split_contract_before_opening_inputs(
    tmp_path: Path,
    omitted: str,
) -> None:
    command = [
        sys.executable,
        str(RESCORE_SCRIPT),
        str(tmp_path / "must-not-be-opened.jsonl"),
        "--splits",
        str(tmp_path / "must-not-be-opened-splits.json"),
        "--split",
        "val",
        "--out",
        str(tmp_path / "audit.jsonl"),
    ]
    option = "--splits" if omitted == "splits" else "--split"
    index = command.index(option)
    del command[index : index + 2]

    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert f"the following arguments are required: {option}" in completed.stderr
    assert "No such file" not in completed.stderr


def test_locked_test_rescore_requires_selection_before_opening_inputs(
    tmp_path: Path,
) -> None:
    contract = ROOT / "configs" / "quality_v2_paper_audit_v2.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(RESCORE_SCRIPT),
            str(tmp_path / "must-not-be-opened.jsonl"),
            "--splits",
            str(tmp_path / "must-not-be-opened-splits.json"),
            "--split",
            "test",
            "--audit-contract",
            str(contract),
            "--audit-contract-sha256",
            _sha256(contract),
            "--out",
            str(tmp_path / "audit.jsonl"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "locked test rescoring requires --selection" in completed.stderr
    assert "No such file" not in completed.stderr


def test_locked_test_rescore_revalidates_selection_before_opening_inputs(
    tmp_path: Path,
) -> None:
    contract = ROOT / "configs" / "quality_v2_paper_audit_v2.json"
    selection = tmp_path / "selection.json"
    selection.write_text("{}", encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(RESCORE_SCRIPT),
            str(tmp_path / "must-not-be-opened.jsonl"),
            "--splits",
            str(tmp_path / "must-not-be-opened-splits.json"),
            "--split",
            "test",
            "--audit-contract",
            str(contract),
            "--audit-contract-sha256",
            _sha256(contract),
            "--selection",
            str(selection),
            "--selection-sha256",
            _sha256(selection),
            "--selection-root",
            str(tmp_path),
            "--out",
            str(tmp_path / "audit.jsonl"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "selection schema version 3" in completed.stderr
    assert "must-not-be-opened" not in completed.stderr


def test_locked_test_rescore_rejects_protocol_override_before_opening_inputs(
    tmp_path: Path,
) -> None:
    contract = ROOT / "configs" / "quality_v2_paper_audit_v2.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(RESCORE_SCRIPT),
            str(tmp_path / "must-not-be-opened.jsonl"),
            "--splits",
            str(tmp_path / "must-not-be-opened-splits.json"),
            "--split",
            "test",
            "--reads",
            "8000",
            "--audit-contract",
            str(contract),
            "--audit-contract-sha256",
            _sha256(contract),
            "--out",
            str(tmp_path / "audit.jsonl"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "audit_labels.reads_per_strength" in completed.stderr
    assert "No such file" not in completed.stderr


def test_locked_test_rescore_authenticates_preregistration_before_opening_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import rescore_quality
    import select_training_grid

    monkeypatch.setattr(
        select_training_grid,
        "revalidate_selection_artifact",
        lambda *args, **kwargs: {"winner": {"paper_checkpoints": []}},
    )
    monkeypatch.setattr(
        rescore_quality,
        "load_paper_preregistration",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("paper preregistration does not match the independently supplied SHA-256")
        ),
    )
    monkeypatch.setattr(
        rescore_quality,
        "_load_records",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("corpus input opened before preregistration authentication")
        ),
    )
    contract = ROOT / "configs" / "quality_v2_paper_audit_v2.json"

    with pytest.raises(SystemExit):
        rescore_quality.main(
            [
                str(tmp_path / "must-not-be-opened.jsonl"),
                "--splits",
                str(tmp_path / "must-not-be-opened-splits.json"),
                "--split",
                "test",
                "--audit-contract",
                str(contract),
                "--audit-contract-sha256",
                _sha256(contract),
                "--selection",
                str(tmp_path / "selection.json"),
                "--selection-sha256",
                "a" * 64,
                "--selection-root",
                str(tmp_path),
                "--preregistration-manifest",
                str(tmp_path / "self-rehashed-preregistration.json"),
                "--preregistration-manifest-sha256",
                "b" * 64,
                "--policy-freezes",
                *(str(tmp_path / f"freeze-{seed}.json") for seed in range(4)),
                "--shard-count",
                "64",
                "--out",
                str(tmp_path / "audit.jsonl"),
            ]
        )

    error = capsys.readouterr().err
    assert "independently supplied SHA-256" in error
    assert "must-not-be-opened" not in error
