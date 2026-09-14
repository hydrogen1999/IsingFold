from __future__ import annotations

import re
import tomllib
from pathlib import Path

from lac_minorminer import backend_info


ROOT = Path(__file__).resolve().parents[2]


def test_stock_minorminer_is_optional_and_absent_from_production_imports() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    runtime = metadata["project"].get("dependencies", [])
    build = metadata["build-system"]["requires"]

    assert all("minorminer" not in dependency.lower() for dependency in runtime + build)
    assert metadata["project"]["optional-dependencies"]["baseline"] == ["minorminer==0.2.22"]

    forbidden_import = re.compile(r"(?:from|import)\s+minorminer\b")
    production_python = (ROOT / "src" / "lac_minorminer").glob("*.py")
    assert not {
        path.relative_to(ROOT): forbidden_import.findall(path.read_text())
        for path in production_python
        if forbidden_import.search(path.read_text())
    }
    assert backend_info()["stock_minorminer_linked"] is False
