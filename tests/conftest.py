"""Make the test suite test this repository.

The venvs on the remote hosts carry an editable install of an older tree, and its finder sits
on sys.meta_path ahead of PYTHONPATH, so without this file `import isingfold` inside a test
resolves to that tree and the suite reports on code that is not here. The probes carry the
same guard; this puts it in one place for the tests.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PROBES = ROOT / "probes"

sys.meta_path[:] = [
    f for f in sys.meta_path
    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())
]
for name in list(sys.modules):
    if name == "isingfold" or name.startswith("isingfold."):
        mod = sys.modules[name]
        file = getattr(mod, "__file__", "") or ""
        if file and not file.startswith(str(SRC)):
            del sys.modules[name]
for path in (str(PROBES), str(SRC)):
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)
os.environ.setdefault("ISINGFOLD_SRC", str(SRC))
