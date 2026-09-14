"""Generate a corpus with the source tree pinned, and on a host big enough to be hard.

`python -m isingfold.rl.cli` imports whatever the editable install points at, which is a
different checkout. Every probe here pins `ISINGFOLD_SRC` first for exactly that reason, so the
generator gets the same treatment before its arguments are parsed.

On the difficulty: the registered contract allows a qubit cap up to 248, because the terminal
reserve must carry 32 feature-work units per capped qubit and the registered reserve holds 8192.
That is enough room for forty variables of chain size five on a Pegasus host, which is the
regime the Chimera 4 corpus does not reach: there minorminer never fails and a restart costs
0.17 seconds, so more restarts are always the best use of time.
"""
import os, sys
from pathlib import Path

sys.path.insert(0, os.environ["ISINGFOLD_SRC"])
sys.meta_path[:] = [f for f in sys.meta_path
                    if not ("editable" in (getattr(type(f), "__module__", "") or "").lower()
                            and "isingfold" in (getattr(type(f), "__module__", "") or "").lower())]

from isingfold.rl import cli

if __name__ == "__main__":
    import isingfold.rl.contracts as contracts
    print("pinned to", Path(contracts.__file__).resolve(), flush=True)
    sys.argv = ["isingfold-rl", "dev-generate"] + sys.argv[1:]
    raise SystemExit(cli.main())
