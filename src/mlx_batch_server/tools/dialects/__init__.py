"""Concrete model-output dialects for the canonical tool parser."""

from .fallback import PlainTextDialect
from .harmony import HarmonyToolDialect
from .qwen import QwenToolDialect

__all__ = ["HarmonyToolDialect", "PlainTextDialect", "QwenToolDialect"]
