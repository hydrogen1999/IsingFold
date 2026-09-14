"""The benchmark publisher must not depend on the learned IsingFold runtime."""

from __future__ import annotations

import ast
from pathlib import Path


PACKAGE_SOURCE = Path(__file__).resolve().parents[1] / "src" / "embedbench"


def _imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append((node.lineno, node.module))
    return imports


def test_no_learned_runtime_imports() -> None:
    forbidden = ("isingfold", "isingfold_lac", "isingfold_l2")
    violations: list[str] = []
    for path in sorted(PACKAGE_SOURCE.glob("*.py")):
        for line, name in _imports(path):
            if any(name == prefix or name.startswith(f"{prefix}.") for prefix in forbidden):
                violations.append(f"{path.name}:{line}: {name}")
    assert violations == []


def test_lac_minorminer_is_limited_to_explicit_legacy_baseline_adapter() -> None:
    allowed = {"objective_embedder.py"}
    violations: list[str] = []
    observed: set[str] = set()
    for path in sorted(PACKAGE_SOURCE.glob("*.py")):
        for line, name in _imports(path):
            if name == "lac_minorminer" or name.startswith("lac_minorminer."):
                observed.add(path.name)
                if path.name not in allowed:
                    violations.append(f"{path.name}:{line}: {name}")
    assert violations == []
    assert observed == allowed
