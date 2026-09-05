"""Target-owned hosted tool catalog, typed failure registry, and executor.

This module mints every hosted error code (design HOSTED_FAILURE_CONTINUATION
§3.3). ``SafePublicFetchError`` codes enter the registry namespaced with the
``fetch_`` prefix at the ``hosted_web`` boundary. Projectors map these codes to
wire shapes; they never invent codes, and no second registry may exist.

A hosted tool failure is never a raised exception at the executor boundary:
every outcome is one immutable ``ToolExecutionResult`` whose metadata carries
the typed receipt (§3.4). Receipts are structurally secret-free: the payload
schema has no field for provider keys, resolved addresses, or request bodies.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .agent_loop import ToolExecutionResult

if TYPE_CHECKING:
    from .parser import ParsedToolCall

# Every fail-closed code SafePublicFetch can raise, namespaced at the boundary.
_FETCH_CODES = frozenset(
    {
        "invalid_url",
        "invalid_url_scheme",
        "url_credentials_forbidden",
        "url_target_blocked",
        "dns_resolution_failed",
        "redirect_limit_exceeded",
        "redirect_not_allowed",
        "invalid_redirect",
        "url_not_allowed",
        "url_fetch_status",
        "url_fetch_timeout",
        "url_fetch_failed",
        "url_fetch_cancelled",
        "unsupported_media_type",
        "missing_media_type",
        "invalid_media_type",
        "invalid_content_length",
        "source_bytes_exceeded",
        "empty_source",
        "invalid_fetch_budget",
        "invalid_fetch_media_types",
        "token_budget",
    }
)

FETCH_CODE_PREFIX = "fetch_"

# The one frozen registry covering F4-F10 (plus the loop-owned bounds).
HOSTED_ERROR_CODES: frozenset[str] = frozenset(
    {
        "provider_unavailable",  # F4
        "provider_auth_failed",  # F5
        "tool_timeout",  # F8
        "tool_execution_failed",  # F9
        "invalid_tool_result",  # F9
        "invalid_tool_arguments",  # F10
        "tool_arguments_too_large",  # F10
        "tool_not_allowed",  # F10 (unknown/unadmitted hosted name)
        "continuation_exhausted",  # I8: hosted call inside the terminal continuation
        "tool_round_limit",  # AgentLoopLimitExceeded on a new round
    }
    | {f"{FETCH_CODE_PREFIX}{code}" for code in _FETCH_CODES}  # F6-F7
)


class HostedToolError(Exception):
    """One typed hosted tool failure carrying a registered error code."""

    def __init__(self, code: str, message: str) -> None:
        if code not in HOSTED_ERROR_CODES:
            raise ValueError(f"unregistered hosted error code {code!r}")
        if not isinstance(message, str) or not message.strip():
            raise ValueError("hosted error message must not be empty")
        super().__init__(message)
        self.code = code
        self.message = message


@runtime_checkable
class HostedTool(Protocol):
    """One target-executed tool; failures are raised as ``HostedToolError``."""

    @property
    def name(self) -> str: ...

    def describe(self) -> Mapping[str, Any]: ...

    async def invoke(self, arguments: Mapping[str, Any]) -> HostedToolSuccess: ...


class HostedToolSuccess:
    """Model-visible payload plus the audit-safe receipt fields of one success."""

    __slots__ = ("payload", "receipt_fields")

    def __init__(
        self,
        *,
        payload: Any,
        receipt_fields: Mapping[str, Any] | None = None,
    ) -> None:
        self.payload = payload
        self.receipt_fields: Mapping[str, Any] = MappingProxyType(
            dict(receipt_fields or {})
        )


class HostedToolCatalog:
    """Immutable name-to-tool catalog constructed exactly once at composition."""

    __slots__ = ("_tools",)

    def __init__(self, tools: Sequence[HostedTool] = ()) -> None:
        catalog: dict[str, HostedTool] = {}
        for tool in tools:
            name = tool.name
            if not isinstance(name, str) or not name.strip():
                raise ValueError("hosted tool name must not be empty")
            if name in catalog:
                raise ValueError(f"duplicate hosted tool {name!r}")
            catalog[name] = tool
        self._tools: Mapping[str, HostedTool] = MappingProxyType(catalog)

    @property
    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def get(self, name: str) -> HostedTool | None:
        return self._tools.get(name)

    def __bool__(self) -> bool:
        return bool(self._tools)


class HostedToolExecutor:
    """Execute admitted hosted calls; type every failure into one receipt.

    Exactly-once claims stay in ``AgentLoop``; this executor owns argument
    validation (F10), the per-call timeout slice (F8), and conversion of every
    outcome into one immutable typed receipt. The absolute outer deadline is
    owned by the runtime starter, never here.
    """

    def __init__(
        self,
        catalog: HostedToolCatalog,
        *,
        per_call_timeout_s: float = 30.0,
        max_arguments_bytes: int = 1_048_576,
    ) -> None:
        if not isinstance(catalog, HostedToolCatalog):
            raise TypeError("catalog must be a HostedToolCatalog")
        if per_call_timeout_s <= 0:
            raise ValueError("per_call_timeout_s must be positive")
        if max_arguments_bytes < 2:
            raise ValueError("max_arguments_bytes must be at least 2")
        self._catalog = catalog
        self._per_call_timeout_s = per_call_timeout_s
        self._max_arguments_bytes = max_arguments_bytes

    @property
    def catalog(self) -> HostedToolCatalog:
        return self._catalog

    async def execute(self, call: ParsedToolCall) -> ToolExecutionResult:
        started = time.monotonic()
        tool = self._catalog.get(call.name)
        if tool is None:
            return self._failure(
                call,
                "tool_not_allowed",
                "tool is not an admitted hosted tool",
                started,
            )
        arguments = self._decode_arguments(call, started)
        if isinstance(arguments, ToolExecutionResult):
            return arguments
        return await self._invoke(tool, call, arguments, started)

    def _decode_arguments(
        self,
        call: ParsedToolCall,
        started: float,
    ) -> dict[str, Any] | ToolExecutionResult:
        if len(call.arguments.encode("utf-8")) > self._max_arguments_bytes:
            return self._failure(
                call,
                "tool_arguments_too_large",
                "tool arguments exceed the configured byte limit",
                started,
            )
        try:
            arguments = json.loads(call.arguments, object_pairs_hook=_unique_object)
        except (TypeError, ValueError) as error:
            return self._failure(call, "invalid_tool_arguments", str(error), started)
        if not isinstance(arguments, dict):
            return self._failure(
                call,
                "invalid_tool_arguments",
                "tool arguments must decode to a JSON object",
                started,
            )
        return arguments

    async def _invoke(
        self,
        tool: HostedTool,
        call: ParsedToolCall,
        arguments: dict[str, Any],
        started: float,
    ) -> ToolExecutionResult:
        try:
            async with asyncio.timeout(self._per_call_timeout_s):
                success = await tool.invoke(arguments)
        except HostedToolError as error:
            return self._failure(call, error.code, error.message, started)
        except TimeoutError:
            return self._failure(
                call,
                "tool_timeout",
                "hosted tool call exceeded its per-call time budget",
                started,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            return self._failure(
                call,
                "tool_execution_failed",
                str(error) or type(error).__name__,
                started,
            )
        return self._success(call, success, started)

    def _success(
        self,
        call: ParsedToolCall,
        success: HostedToolSuccess,
        started: float,
    ) -> ToolExecutionResult:
        if not isinstance(success, HostedToolSuccess):
            return self._failure(
                call,
                "invalid_tool_result",
                "hosted tool returned no explicit success receipt",
                started,
            )
        try:
            output = _encode_json(success.payload)
        except (TypeError, ValueError):
            return self._failure(
                call,
                "invalid_tool_result",
                "hosted tool payload is not JSON-compatible",
                started,
            )
        receipt = build_receipt(
            call_id=call.call_id,
            tool_name=call.name,
            status="completed",
            duration_ms=_duration_ms(started),
            extra=success.receipt_fields,
        )
        return ToolExecutionResult(
            call_id=call.call_id,
            output=output,
            metadata={"tool_name": call.name, "receipt": receipt},
        )

    def _failure(
        self,
        call: ParsedToolCall,
        code: str,
        message: str,
        started: float,
    ) -> ToolExecutionResult:
        return failure_result(
            call_id=call.call_id,
            tool_name=call.name,
            code=code,
            message=message,
            duration_ms=_duration_ms(started),
        )


def build_receipt(
    *,
    call_id: str,
    tool_name: str,
    status: str,
    duration_ms: int,
    error: Mapping[str, str] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the §3.4 receipt payload; ``attempt`` is constitutionally 1."""

    receipt: dict[str, Any] = {
        "call_id": call_id,
        "tool_name": tool_name,
        "status": status,
        "duration_ms": duration_ms,
        "attempt": 1,
    }
    if error is not None:
        receipt["error"] = dict(error)
    for key, value in dict(extra or {}).items():
        if key in receipt:
            raise ValueError(f"receipt field {key!r} may not be overridden")
        receipt[key] = value
    return receipt


def failure_result(
    *,
    call_id: str,
    tool_name: str,
    code: str,
    message: str,
    duration_ms: int = 0,
) -> ToolExecutionResult:
    """Mint one typed error receipt result for a hosted call."""

    if code not in HOSTED_ERROR_CODES:
        raise ValueError(f"unregistered hosted error code {code!r}")
    receipt = build_receipt(
        call_id=call_id,
        tool_name=tool_name,
        status="failed",
        duration_ms=duration_ms,
        error={"code": code, "message": message},
    )
    return ToolExecutionResult(
        call_id=call_id,
        output=_encode_json(
            {
                "error": {"code": code, "message": message},
                "tool_name": tool_name,
                "call_id": call_id,
            }
        ),
        metadata={"tool_name": tool_name, "error_code": code, "receipt": receipt},
        error=message,
    )


def _duration_ms(started: float) -> int:
    return max(0, int((time.monotonic() - started) * 1000))


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _encode_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


__all__ = [
    "FETCH_CODE_PREFIX",
    "HOSTED_ERROR_CODES",
    "HostedTool",
    "HostedToolCatalog",
    "HostedToolError",
    "HostedToolExecutor",
    "HostedToolSuccess",
    "build_receipt",
    "failure_result",
]
