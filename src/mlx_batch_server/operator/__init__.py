"""Standalone operator backend for MLX Batch Server."""

from __future__ import annotations

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"

__all__ = ["PACKAGE_DIR", "STATIC_DIR", "TEMPLATES_DIR"]
