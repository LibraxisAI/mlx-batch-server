"""
Mlx batch Server - Tool Registry and Execution.

Provides hosted tool support for the Responses API, similar to OpenAI's
web_search, code_interpreter, etc.

Vibecrafted. with AI Agents by VetCoders (c)2024-2026 The LibraxisAI Team
"""

from .registry import (
    BUILTIN_TOOLS,
    ToolRegistry,
    execute_tool,
    extract_tool_calls,
    format_tool_result,
    get_tool_definitions,
    is_hosted_tool,
)

__all__ = [
    "BUILTIN_TOOLS",
    "ToolRegistry",
    "execute_tool",
    "extract_tool_calls",
    "format_tool_result",
    "get_tool_definitions",
    "is_hosted_tool",
]
