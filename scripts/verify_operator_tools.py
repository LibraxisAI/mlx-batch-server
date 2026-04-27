#!/usr/bin/env python3
"""Verify bundled operator tool binaries."""

from __future__ import annotations

import hashlib
import json
import subprocess  # nosec B404 - fixed local spctl invocation for binary verification.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "tools" / "manifest" / "operator-tools.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _spctl_status(path: Path) -> str:
    try:
        result = subprocess.run(  # nosec B603 - arguments are fixed; path is local.
            ["spctl", "-a", "-vv", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "spctl unavailable"
    return "accepted" if result.returncode == 0 else "rejected"


def main() -> int:
    try:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        tools = manifest["tools"]
    except FileNotFoundError:
        print(f"ERROR: Manifest file not found: {MANIFEST}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as exc:
        print(f"ERROR: Invalid JSON in manifest {MANIFEST}: {exc}", file=sys.stderr)
        return 1
    except KeyError:
        print(
            f"ERROR: Manifest {MANIFEST} is missing required 'tools' key",
            file=sys.stderr,
        )
        return 1

    failed = False
    for tool in tools:
        path = ROOT / tool["path"]
        exists = path.exists()
        checksum = _sha256(path) if exists else ""
        ok = exists and checksum == tool["sha256"]
        failed = failed or not ok
        status = _spctl_status(path) if exists else "missing"
        marker = "OK" if ok else "FAIL"
        print(f"{marker} {tool['name']}: {status} {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
