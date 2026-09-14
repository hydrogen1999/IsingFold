#!/usr/bin/env python3
"""Run every hand-test in order and report which of the corpus's claims survive.

Each test is a separate process, so one crash does not hide the rest, and a test that leaves the
interpreter in a strange state cannot affect its neighbours.

    python3 handtests/run_all.py            run everything
    python3 handtests/run_all.py 03 05      run only those
    python3 handtests/run_all.py --explain  print what each one checks, run nothing
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def tests() -> list:
    return sorted(p for p in HERE.glob("ht*.py"))


def main(argv: list) -> int:
    wanted = [a for a in argv if not a.startswith("-")]
    explain = "--explain" in argv
    chosen = [p for p in tests() if not wanted or any(p.name.startswith(f"ht{w.zfill(2)}") for w in wanted)]
    if not chosen:
        print(f"no hand-test matches {wanted}; have {[p.name for p in tests()]}")
        return 2

    failed = []
    for path in chosen:
        print(f"\n=== {path.name}")
        cmd = [sys.executable, str(path)] + (["--explain"] if explain else [])
        started = time.time()
        out = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(out.stdout)
        if out.returncode != 0:
            failed.append(path.name)
            sys.stdout.write(out.stderr)
            print(f"--- FAILED in {time.time()-started:.1f}s")
        elif not explain:
            print(f"--- passed in {time.time()-started:.1f}s")

    if explain:
        return 0
    print(f"\n{len(chosen)-len(failed)} of {len(chosen)} hand-tests passed")
    for name in failed:
        print(f"  failed: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
