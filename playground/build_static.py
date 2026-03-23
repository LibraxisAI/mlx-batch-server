#!/usr/bin/env python3
"""Build static assets for the API tester UI."""

from __future__ import annotations

import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT.parent / "lbrx-services" / "tools" / "api-tester" / "index.html"
TARGET_DIR = Path(__file__).parent / "static"
TARGET = TARGET_DIR / "api-tester.html"


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"Source not found: {SOURCE}")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE, TARGET)
    print(f"Built: {TARGET}")


if __name__ == "__main__":
    main()
