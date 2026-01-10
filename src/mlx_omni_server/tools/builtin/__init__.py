"""
Built-in hosted tools for MLX Omni Server.

These tools are executed server-side when the model generates tool calls.

Created by M&K (c)2026 The LibraxisAI Team
"""

from .code_interpreter import execute_code
from .web_search import execute_web_search

__all__ = [
    "execute_code",
    "execute_web_search",
]
