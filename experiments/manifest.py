"""Reproducibility manifests without credentials or ambient environment data."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists():
        return None
    return hashlib.sha256(resolved.read_bytes()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def write_manifest(path: str | Path, payload: dict) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    safe_payload = {"git_commit": git_commit(), **payload}
    output.write_text(
        json.dumps(safe_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = ["file_sha256", "git_commit", "write_manifest"]
