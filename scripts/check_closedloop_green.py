"""Local deterministic ClosedLoop quality gate without legacy heavyweight dependencies."""
from __future__ import annotations

import compileall
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    if not compileall.compile_dir(ROOT / "closedloop", quiet=1):
        return 1
    run([sys.executable, "-m", "pytest", "-q"])
    run([sys.executable, "-m", "closedloop", "equilibrium-verify", "--nr", "25", "--nz", "25", "--max-error", "1e-6"])
    ruff = shutil.which("ruff")
    if ruff:
        run([ruff, "check", "closedloop", "tests", "scripts/check_closedloop_green.py"])
    else:
        print("INFO: ruff not installed locally; CI installs and enforces it.")
    print(json.dumps({"closedloop_quality_gate": "GREEN", "experimental_validation": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
