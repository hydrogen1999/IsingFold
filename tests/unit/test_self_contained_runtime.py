from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path

import networkx as nx


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO_ROOT / "src"


def test_production_sources_do_not_import_sibling_workspaces() -> None:
    forbidden = ("isingfold_lac", "embedbench")
    violations: list[str] = []

    for path in sorted(SOURCE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if any(name == prefix or name.startswith(f"{prefix}.") for prefix in forbidden):
                    violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}: {name}")

    assert violations == []


def test_all_production_modules_import_from_this_workspace() -> None:
    code = """
from pathlib import Path
import importlib

root = Path.cwd().resolve()
modules = []
for path in sorted((root / "src").rglob("*.py")):
    relative = path.relative_to(root / "src").with_suffix("")
    parts = list(relative.parts)
    if parts[-1] == "__init__":
        parts.pop()
    name = ".".join(parts)
    if name and name not in modules:
        modules.append(name)
for name in modules:
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    if root not in path.parents:
        raise RuntimeError(f"{name} resolved outside this workspace: {path}")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_local_objective_core_and_native_embedder_form_a_tiny_end_to_end_path() -> None:
    embedding_module = importlib.import_module("isingfold.embedding")
    pipeline = importlib.import_module("isingfold_lac_b.pipeline")

    problem = embedding_module.LogicalProblem.from_dicts(
        h={0: 0.25, 1: -0.5},
        j={(0, 1): -1.0},
    )
    host = nx.path_graph(3)
    result = pipeline.embed(
        problem,
        host,
        method="lac_b",
        seed=7,
        timeout=1.0,
        tries=2,
        max_transitions=100,
        search_profile="v0",
    )

    assert result.valid
    assert result.chains is not None
    assert set(result.chains) == {0, 1}
    assert pipeline.t0_score(problem, host, result.chains)["p_solve"] is not None
