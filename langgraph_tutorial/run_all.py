"""Run every offline tutorial in order and report pass/fail.

File 10 needs a Gemini API key, so it is skipped by default. Run it manually:

    python langgraph_tutorial/10_tools_and_prebuilt_agents.py

Usage:
    python langgraph_tutorial/run_all.py            # offline files
    python langgraph_tutorial/run_all.py --all      # include file 10
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SKIP = {"_shared.py", "run_all.py"}


def main() -> int:
    include_api = "--all" in sys.argv
    files = sorted(p for p in HERE.glob("[0-9][0-9]_*.py"))
    if not include_api:
        files = [p for p in files if not p.name.startswith("10_")]

    failures: list[tuple[str, str]] = []
    for path in files:
        print(f"\n{'=' * 74}\nRUN {path.name}\n{'=' * 74}")
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=HERE,
            capture_output=True,
            text=True,
        )
        tail = (proc.stdout or "").strip().splitlines()
        if tail:
            print("\n".join(tail[-3:]))
        if proc.returncode != 0:
            failures.append((path.name, (proc.stderr or "").strip().splitlines()[-1] if proc.stderr else "?"))
            print(f"  FAILED ({proc.returncode})")
        else:
            print("  ok")

    print(f"\n{'-' * 74}")
    if failures:
        print(f"{len(failures)} file(s) failed:")
        for name, error in failures:
            print(f"  - {name}: {error}")
        return 1

    print(f"All {len(files)} tutorial file(s) passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
