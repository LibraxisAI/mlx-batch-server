"""Adapter from canonical agent-loop calls to the target hosted-tool registry."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeAlias

from .agent_loop import ToolExecutionResult
from .registry import execute_tool, is_hosted_tool

if TYPE_CHECKING:
    from .parser import ParsedToolCall

RegistryResult: TypeAlias = Mapping[str, Any]
RegistryExecute: TypeAlias = Callable[[str, dict[str, Any]], Awaitable[RegistryResult]]
RegistryPredicate: TypeAlias = Callable[[str], bool]


class RegistryToolExecutor:
    """Execute only explicitly admitted target-hosted tools.

    Exactly-once claim ownership remains in ``AgentLoop``. This adapter owns
    argument validation and conversion of the legacy registry result into one
    stable function-call output receipt.
    """

    def __init__(
        self,
        *,
        allowed_tools: frozenset[str] | None = None,
        max_arguments_bytes: int = 1_048_576,
        execute: RegistryExecute = execute_tool,
        is_hosted: RegistryPredicate = is_hosted_tool,
    ) -> None:
        if max_arguments_bytes < 2:
            raise ValueError("max_arguments_bytes must be at least 2")
        self._allowed_tools = allowed_tools
        self._max_arguments_bytes = max_arguments_bytes
        self._execute = execute
        self._is_hosted = is_hosted

    async def execute(self, call: ParsedToolCall) -> ToolExecutionResult:
        if self._allowed_tools is not None and call.name not in self._allowed_tools:
            return self._failure(call, "tool_not_allowed", "tool is not allowed")
        if not self._is_hosted(call.name):
            return self._failure(
                call,
                "client_tool_not_executable",
                "tool must be executed by the client",
            )

        encoded_arguments = call.arguments.encode("utf-8")
        if len(encoded_arguments) > self._max_arguments_bytes:
            return self._failure(
                call,
                "tool_arguments_too_large",
                "tool arguments exceed the configured byte limit",
            )
        try:
            arguments = json.loads(call.arguments, object_pairs_hook=_unique_object)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            return self._failure(call, "invalid_tool_arguments", str(error))
        if not isinstance(arguments, dict):
            return self._failure(
                call,
                "invalid_tool_arguments",
                "tool arguments must decode to a JSON object",
            )

        try:
            result = await self._execute(call.name, arguments)
        except Exception as error:
            return self._failure(
                call,
                "tool_execution_failed",
                str(error) or type(error).__name__,
            )
        if not isinstance(result, Mapping):
            return self._failure(
                call,
                "invalid_tool_result",
                "tool registry returned a non-object result",
            )
        error = result.get("error")
        if error is not None:
            error_object = dict(error) if isinstance(error, Mapping) else {}
            code = str(error_object.get("code") or "tool_execution_failed")
            message = str(error_object.get("message") or "tool execution failed")
            return ToolExecutionResult(
                call_id=call.call_id,
                output=_encode_json({"error": {"code": code, "message": message}}),
                metadata={"tool_name": call.name, "error_code": code},
                error=message,
            )

        if result.get("success") is not True:
            return self._failure(
                call,
                "invalid_tool_result",
                "tool registry result lacks an explicit success receipt",
            )
        try:
            output = _encode_json(result.get("data"))
        except (TypeError, ValueError):
            return self._failure(
                call,
                "invalid_tool_result",
                "tool registry data is not JSON-compatible",
            )
        return ToolExecutionResult(
            call_id=call.call_id,
            output=output,
            metadata={"tool_name": call.name},
        )

    @staticmethod
    def _failure(
        call: ParsedToolCall,
        code: str,
        message: str,
    ) -> ToolExecutionResult:
        return ToolExecutionResult(
            call_id=call.call_id,
            output=_encode_json({"error": {"code": code, "message": message}}),
            metadata={"tool_name": call.name, "error_code": code},
            error=message,
        )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _encode_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


__all__ = ["RegistryToolExecutor"]
