"""Canonical evidence bundles with SHA-256 manifests and reproducibility metadata."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from math import isfinite
from pathlib import Path
from typing import Any

import numpy as np


def canonicalize(value: Any) -> Any:
    if is_dataclass(value):
        return canonicalize(asdict(value))
    if isinstance(value, np.ndarray):
        return canonicalize(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return float(format(v, ".12g")) if isfinite(v) else str(v)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, dict):
        return {str(k): canonicalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [canonicalize(v) for v in value]
    return value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo_root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(canonicalize(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_evidence_bundle(
    output_dir: str | Path,
    *,
    config: dict[str, Any],
    summary: dict[str, Any],
    timeseries: list[dict[str, Any]],
    gates: dict[str, Any],
    repo_root: str | Path | None = None,
) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    environment = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "git_commit": git_commit(root),
    }
    write_json(out / "config.json", config)
    write_json(out / "summary.json", summary)
    write_json(out / "timeseries.json", timeseries)
    write_json(out / "claim_gates.json", gates)
    write_json(out / "environment.json", environment)
    files = [p for p in out.iterdir() if p.is_file() and p.name != "SHA256SUMS"]
    manifest = "".join(f"{sha256_file(path)}  {path.name}\n" for path in sorted(files))
    (out / "SHA256SUMS").write_text(manifest, encoding="utf-8")
    return out


def verify_evidence_bundle(output_dir: str | Path) -> tuple[bool, list[str]]:
    out = Path(output_dir)
    manifest = out / "SHA256SUMS"
    if not manifest.exists():
        return False, ["missing SHA256SUMS"]
    errors: list[str] = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split("  ", 1)
        path = out / name
        if not path.exists():
            errors.append(f"missing {name}")
        elif sha256_file(path) != expected:
            errors.append(f"hash mismatch {name}")
    return not errors, errors
