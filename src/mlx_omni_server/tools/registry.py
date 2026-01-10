"""
Tool Registry - Registration and execution of hosted tools.

Supports:
- Built-in tools (web_search, calculator)
- Custom tool registration via decorators
- Async tool execution
- Result formatting for LLM context

Created by M&K (c)2026 The LibraxisAI Team
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any, ClassVar

from .builtin.code_interpreter import execute_code
from .builtin.web_search import execute_web_search

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Registry for hosted tools that can be executed server-side.

    Tools are registered with their implementation functions and
    can be executed when the model generates tool calls.
    """

    _tools: ClassVar[dict[str, dict[str, Any]]] = {}
    _implementations: ClassVar[dict[str, Callable]] = {}

    @classmethod
    def register(
        cls,
        name: str,
        description: str,
        parameters: dict[str, Any],
    ) -> Callable:
        """
        Decorator to register a tool implementation.

        Args:
            name: Tool name (e.g., "web_search")
            description: Human-readable description
            parameters: JSON Schema for parameters

        Returns:
            Decorator function
        """

        def decorator(func: Callable) -> Callable:
            cls._tools[name] = {
                "type": "function",
                "name": name,
                "description": description,
                "parameters": parameters,
            }
            cls._implementations[name] = func
            logger.info(f"Registered tool: {name}")
            return func

        return decorator

    @classmethod
    def get_tool(cls, name: str) -> dict[str, Any] | None:
        """Get tool definition by name."""
        return cls._tools.get(name)

    @classmethod
    def get_implementation(cls, name: str) -> Callable | None:
        """Get tool implementation function."""
        return cls._implementations.get(name)

    @classmethod
    def list_tools(cls) -> list[dict[str, Any]]:
        """List all registered tools."""
        return list(cls._tools.values())

    @classmethod
    def has_tool(cls, name: str) -> bool:
        """Check if a tool is registered."""
        return name in cls._tools


# Built-in tool definitions (OpenAI Responses API format)
BUILTIN_TOOLS: dict[str, dict[str, Any]] = {
    "web_search": {
        "type": "web_search",
        "description": "Search the web for current information",
    },
    "code_interpreter": {
        "type": "code_interpreter",
        "description": "Execute Python code in a sandboxed environment",
    },
}


def get_tool_definitions(
    requested_tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """
    Get tool definitions for a request.

    Expands hosted tool types (web_search, code_interpreter) into
    their full function definitions.

    Args:
        requested_tools: Tools from request body

    Returns:
        List of tool definitions in OpenAI function format
    """
    if not requested_tools:
        return []

    definitions = []

    for tool in requested_tools:
        tool_type = tool.get("type", "function")

        if tool_type == "function":
            # Pass through function definitions
            definitions.append(tool)

        elif tool_type == "web_search":
            # Expand to function definition
            definitions.append(
                {
                    "type": "function",
                    "name": "web_search",
                    "description": "Search the web for current information. "
                    "Use this when you need up-to-date information or facts you're unsure about.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query",
                            },
                        },
                        "required": ["query"],
                    },
                }
            )

        elif tool_type == "code_interpreter":
            definitions.append(
                {
                    "type": "function",
                    "name": "code_interpreter",
                    "description": "Execute Python code to perform calculations, "
                    "data analysis, or other programmatic tasks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "code": {
                                "type": "string",
                                "description": "Python code to execute",
                            },
                        },
                        "required": ["code"],
                    },
                }
            )

    # Add any registered custom tools
    definitions.extend(ToolRegistry.list_tools())

    return definitions


def is_hosted_tool(tool_name: str) -> bool:
    """
    Check if a tool is a hosted tool (executed server-side).

    Hosted tools include:
    - Built-in tools (web_search, code_interpreter)
    - Registered custom tools
    """
    if tool_name in BUILTIN_TOOLS:
        return True
    return ToolRegistry.has_tool(tool_name)


async def execute_tool(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """
    Execute a hosted tool and return the result.

    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments

    Returns:
        Tool result dict with 'success' and 'data' or 'error'
    """
    try:
        # Check built-in tools first
        if tool_name == "web_search":
            result = await execute_web_search(**arguments)
            return {"success": True, "data": result}

        if tool_name == "code_interpreter":
            result = await execute_code(**arguments)
            return {"success": True, "data": result}

        # Check registered tools
        impl = ToolRegistry.get_implementation(tool_name)
        if impl is not None:
            # Support both sync and async implementations
            if asyncio.iscoroutinefunction(impl):
                result = await impl(**arguments)
            else:
                result = impl(**arguments)
            return {"success": True, "data": result}

        return {
            "error": {
                "code": "unknown_tool",
                "message": f"Unknown tool: {tool_name}",
            }
        }

    except Exception as e:
        logger.error(f"Tool '{tool_name}' execution failed: {e}")
        return {
            "error": {
                "code": "execution_error",
                "message": str(e),
            }
        }


def extract_tool_calls(response_output: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Extract function_call items from Responses API output.

    Args:
        response_output: The 'output' list from a response

    Returns:
        List of function_call items with name, arguments, call_id
    """
    return [item for item in response_output if item.get("type") == "function_call"]


def format_tool_result(
    tool_name: str,
    call_id: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    """
    Format a tool result as a function_call_output item.

    Args:
        tool_name: Name of the tool
        call_id: The call_id from the function_call
        result: Tool execution result

    Returns:
        function_call_output item for Responses API
    """
    if "error" in result:
        output = f"Error: {result['error'].get('message', 'Unknown error')}"
    else:
        data = result.get("data", {})
        if isinstance(data, str):
            output = data
        elif isinstance(data, dict):
            # Format based on tool type
            if tool_name == "web_search":
                output = _format_search_results(data)
            elif tool_name == "code_interpreter":
                output = _format_code_result(data)
            else:
                output = json.dumps(data, indent=2)
        else:
            output = str(data)

    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
    }


def _format_search_results(data: dict[str, Any]) -> str:
    """Format web search results for LLM context."""
    lines = ["Web Search Results:"]

    results = data.get("results", [])
    for i, result in enumerate(results[:5], 1):
        title = result.get("title", "")
        url = result.get("url", "")
        snippet = result.get("snippet", result.get("description", ""))
        lines.append(f"\n[{i}] {title}")
        lines.append(f"    URL: {url}")
        if snippet:
            lines.append(f"    {snippet[:300]}")

    return "\n".join(lines)


def _format_code_result(data: dict[str, Any]) -> str:
    """Format code execution results for LLM context."""
    lines = ["Code Execution Result:"]

    if "output" in data:
        lines.append(f"\nOutput:\n{data['output']}")
    if "result" in data:
        lines.append(f"\nReturn value: {data['result']}")
    if "error" in data:
        lines.append(f"\nError: {data['error']}")

    return "\n".join(lines)
