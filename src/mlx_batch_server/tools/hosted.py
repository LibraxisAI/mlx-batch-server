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
import contextvars
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
        "result_budget_exceeded",  # canonical result payload over its per-call bound
    }
    | {f"{FETCH_CODE_PREFIX}{code}" for code in _FETCH_CODES}  # F6-F7
)


# The one fixed model/audit-visible sentence for an unexpected (untyped)
# executor/provider crash. Raw exception text may carry secrets and never
# crosses this boundary (design §3.4 "structurally secret-free").
UNEXPECTED_EXECUTION_FAILURE_MESSAGE = "hosted tool execution failed unexpectedly"

# Closed success-receipt extra schema (§3.4): only explicitly designed,
# scalar, JSON-serializable audit fields. ``bool`` is rejected everywhere.
RECEIPT_EXTRA_FIELDS: Mapping[str, type] = MappingProxyType(
    {
        "final_url": str,
        "mime": str,
        "http_status": int,
        "redirect_count": int,
        "result_digest": str,
        "result_count": int,
    }
)

# Canonical decoded result channel (HR2-2). ``metadata["result"]`` is the sole
# producer boundary for the closed, bounded, model-agreeing success payload;
# the receipt stays a separate closed audit surface and the two must agree
# mechanically on digest/provenance. Failure, cancel, and deadline outcomes
# never carry a result payload.
RESULT_KIND_FOR_TOOL: Mapping[str, str] = MappingProxyType(
    {
        "web_fetch": "document",
        "web_search": "search_results",
    }
)
ACTION_KIND_FOR_TOOL: Mapping[str, str] = MappingProxyType(
    {
        "web_fetch": "fetch",
        "web_search": "search",
    }
)
MAX_RESULT_TEXT_CHARS = 262_144
MAX_RESULT_BYTES = 1_048_576

_DOCUMENT_RESULT_KEYS = frozenset(
    {"kind", "url", "media_type", "content", "digest", "retrieved_at"}
)
_SEARCH_RESULT_KEYS = frozenset({"kind", "query", "results", "digest"})
_SEARCH_ENTRY_KEYS = frozenset({"title", "url", "snippet"})
_SEARCH_ACTION_KEYS = frozenset({"kind", "query", "sources"})
_FETCH_ACTION_KEYS = frozenset({"kind", "url"})
_DIGEST_PATTERN = re.compile(r"\Asha256:[0-9a-f]{64}\Z")


class ResultBudgetExceeded(ValueError):
    """A structurally valid result payload exceeds its per-call bound."""


def canonical_json(value: Any) -> str:
    """The one canonical compact/sorted JSON representation used for digests."""

    return _encode_json(value)


def _require_result_identity(name: str, value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, str) or not value.strip():
        raise ValueError(f"result field {name!r} must be a non-empty string")
    return value


def _require_result_digest(value: Any) -> str:
    if not isinstance(value, str) or _DIGEST_PATTERN.match(value) is None:
        raise ValueError("result digest must be 'sha256:' plus 64 lowercase hex")
    return value


def validate_result_payload(tool_name: str, result: Any) -> dict[str, Any]:
    """Totally validate one closed canonical result payload for ``tool_name``.

    Unknown keys, wrong types, empty identities, invalid digests, a kind that
    does not belong to the tool, and nested foreign/debug/secret fields all
    fail closed with ``ValueError``. A structurally valid payload above its
    per-call bound raises ``ResultBudgetExceeded``. Returns a plain decoded
    ``dict`` copy; validation never touches any transport.
    """

    expected_kind = RESULT_KIND_FOR_TOOL.get(tool_name)
    if expected_kind is None:
        raise ValueError(f"tool {tool_name!r} has no canonical result schema")
    if not isinstance(result, Mapping):
        raise ValueError("result payload must be a mapping")
    if result.get("kind") != expected_kind:
        raise ValueError(f"result kind must be {expected_kind!r} for {tool_name!r}")
    if expected_kind == "document":
        validated = _validate_document_result(result)
    else:
        validated = _validate_search_result(result)
    if len(canonical_json(validated).encode("utf-8")) > MAX_RESULT_BYTES:
        raise ResultBudgetExceeded("result payload exceeds the per-call byte bound")
    return validated


def _validate_document_result(result: Mapping[str, Any]) -> dict[str, Any]:
    if set(result) != _DOCUMENT_RESULT_KEYS:
        raise ValueError("document result carries exactly its closed key set")
    content = result["content"]
    if isinstance(content, bool) or not isinstance(content, str):
        raise ValueError("document content must be a string")
    retrieved_at = result["retrieved_at"]
    if (
        isinstance(retrieved_at, bool)
        or not isinstance(retrieved_at, int)
        or retrieved_at < 0
    ):
        raise ValueError("retrieved_at must be a non-negative UTC integer")
    validated: dict[str, Any] = {
        "kind": "document",
        "url": _require_result_identity("url", result["url"]),
        "media_type": _require_result_identity("media_type", result["media_type"]),
        "content": content,
        "digest": _require_result_digest(result["digest"]),
        "retrieved_at": retrieved_at,
    }
    if len(content) > MAX_RESULT_TEXT_CHARS:
        raise ResultBudgetExceeded("document content exceeds the per-call bound")
    return validated


def _validate_search_result(result: Mapping[str, Any]) -> dict[str, Any]:
    if set(result) != _SEARCH_RESULT_KEYS:
        raise ValueError("search result carries exactly its closed key set")
    entries = result["results"]
    if isinstance(entries, str | bytes) or not isinstance(entries, Sequence):
        raise ValueError("search results must be a sequence of entries")
    sanitized_entries: list[dict[str, str]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("search result entries must be mappings")
        if set(entry) != _SEARCH_ENTRY_KEYS:
            raise ValueError(
                "search result entry carries exactly title, url and snippet"
            )
        validated_entry = {
            key: _require_result_identity(key, entry[key])
            for key in ("title", "url", "snippet")
        }
        sanitized_entries.append(validated_entry)
    return {
        "kind": "search_results",
        "query": _require_result_identity("query", result["query"]),
        "results": sanitized_entries,
        "digest": _require_result_digest(result["digest"]),
    }


def result_identities(result: Mapping[str, Any]) -> tuple[str, ...]:
    """The URL identities a validated result proves for later sealed actions."""

    if result["kind"] == "document":
        return (result["url"],)
    return tuple(entry["url"] for entry in result["results"])


def validate_sealed_action(
    tool_name: str,
    action: Any,
    *,
    result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Totally validate one closed sealed action against its tool and result.

    When ``result`` is given, a search action's sources must be a subset of
    the result-proven identities. A fetch action carries the model-REQUESTED
    URL and returns it unchanged: the redirect-resolved final URL is
    result/receipt provenance, so the two URLs are deliberately never
    compared here — their agreement law lives solely in
    ``_verify_result_receipt_agreement``. A supplied fetch result is still
    fully validated closed.
    """

    expected_kind = ACTION_KIND_FOR_TOOL.get(tool_name)
    if expected_kind is None:
        raise ValueError(f"tool {tool_name!r} has no sealed action schema")
    if not isinstance(action, Mapping):
        raise ValueError("sealed action must be a mapping")
    if action.get("kind") != expected_kind:
        raise ValueError(f"sealed action kind must be {expected_kind!r}")
    keys = set(action)
    if expected_kind == "search":
        if keys != _SEARCH_ACTION_KEYS:
            raise ValueError("search action carries exactly kind, query and sources")
        query = _require_result_identity("query", action["query"])
        sources = action["sources"]
        if isinstance(sources, str | bytes) or not isinstance(sources, Sequence):
            raise ValueError("search action sources must be a sequence")
        validated_sources = tuple(
            _require_result_identity("source", source) for source in sources
        )
        if len(set(validated_sources)) != len(validated_sources):
            raise ValueError("search action sources must be unique")
        if result is not None:
            proven = set(result_identities(validate_result_payload(tool_name, result)))
            if not set(validated_sources) <= proven:
                raise ValueError("search action sources are not proven by the result")
        return {"kind": "search", "query": query, "sources": list(validated_sources)}
    if keys != _FETCH_ACTION_KEYS:
        raise ValueError("fetch action carries exactly kind and url")
    url = _require_result_identity("url", action["url"])
    if result is not None:
        validate_result_payload(tool_name, result)
    return {"kind": "fetch", "url": url}


def _verify_result_receipt_agreement(
    result: Mapping[str, Any],
    receipt_fields: Mapping[str, Any],
) -> None:
    """Fail closed unless receipt provenance mechanically matches the result."""

    if receipt_fields.get("result_digest") != result["digest"]:
        raise ValueError("receipt result_digest does not match the result digest")
    if result["kind"] == "document":
        if receipt_fields.get("final_url") != result["url"]:
            raise ValueError("receipt final_url does not match the result url")
        if receipt_fields.get("mime") != result["media_type"]:
            raise ValueError("receipt mime does not match the result media_type")


@runtime_checkable
class ExecutionCancelCheck(Protocol):
    """Cooperative cancellation surface shared with the fetch transport."""

    @property
    def cancelled(self) -> bool: ...

    @property
    def reason(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class HostedExecutionScope:
    """Request-scoped immutable deadline/cancel context for hosted work.

    ``deadline`` is an absolute instant on the running event loop's clock
    (``loop.time()``), set once by the runtime starter per request. The scope
    travels via a context variable, so concurrent requests each observe their
    own snapshot and no request can inherit another's budget or cancel token.
    """

    deadline: float | None = None
    cancel: ExecutionCancelCheck | None = None

    def remaining_s(self) -> float | None:
        if self.deadline is None:
            return None
        return self.deadline - asyncio.get_running_loop().time()


_NO_SCOPE = HostedExecutionScope()
_EXECUTION_SCOPE: contextvars.ContextVar[HostedExecutionScope] = contextvars.ContextVar(
    "hosted_execution_scope",
    default=_NO_SCOPE,
)


def current_execution_scope() -> HostedExecutionScope:
    return _EXECUTION_SCOPE.get()


def set_execution_scope(
    scope: HostedExecutionScope,
) -> contextvars.Token[HostedExecutionScope]:
    if not isinstance(scope, HostedExecutionScope):
        raise TypeError("scope must be a HostedExecutionScope")
    return _EXECUTION_SCOPE.set(scope)


def reset_execution_scope(token: contextvars.Token[HostedExecutionScope]) -> None:
    _EXECUTION_SCOPE.reset(token)


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
    """Model-visible payload plus the audit-safe receipt fields of one success.

    ``result`` is the tool's canonical decoded payload for the
    ``metadata["result"]`` channel; the executor validates it closed and
    bounded before it reaches any consumer. ``payload`` stays the byte-source
    of the model continuation and is never reinterpreted downstream.
    """

    __slots__ = ("payload", "receipt_fields", "result")

    def __init__(
        self,
        *,
        payload: Any,
        receipt_fields: Mapping[str, Any] | None = None,
        result: Mapping[str, Any] | None = None,
    ) -> None:
        self.payload = payload
        self.receipt_fields: Mapping[str, Any] = MappingProxyType(
            dict(receipt_fields or {})
        )
        self.result = result


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
        scope = current_execution_scope()
        if scope.cancel is not None and scope.cancel.cancelled:
            raise asyncio.CancelledError(scope.cancel.reason or "cancelled")
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
        except Exception:
            # An untyped crash may carry provider internals or secrets in its
            # text; only the fixed audit-safe sentence crosses this boundary.
            return self._failure(
                call,
                "tool_execution_failed",
                UNEXPECTED_EXECUTION_FAILURE_MESSAGE,
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
        validated_result: dict[str, Any] | None = None
        if success.result is not None:
            try:
                validated_result = validate_result_payload(call.name, success.result)
                _verify_result_receipt_agreement(
                    validated_result, success.receipt_fields
                )
            except ResultBudgetExceeded:
                return self._failure(
                    call,
                    "result_budget_exceeded",
                    "hosted tool result exceeds its per-call budget",
                    started,
                )
            except (TypeError, ValueError):
                # Foreign fields, broken identities, or receipt disagreement
                # fail closed; the offending values never reach any surface.
                return self._failure(
                    call,
                    "invalid_tool_result",
                    "hosted tool produced an invalid result payload",
                    started,
                )
        try:
            receipt = build_receipt(
                call_id=call.call_id,
                tool_name=call.name,
                status="completed",
                duration_ms=_duration_ms(started),
                extra=success.receipt_fields,
            )
        except ValueError:
            # Unknown keys, overrides, nested payloads, or wrong types fail
            # closed; the offending field values never enter any surface.
            return self._failure(
                call,
                "invalid_tool_result",
                "hosted tool produced an invalid success receipt",
                started,
            )
        metadata: dict[str, Any] = {"tool_name": call.name, "receipt": receipt}
        if validated_result is not None:
            metadata["result"] = validated_result
        return ToolExecutionResult(
            call_id=call.call_id,
            output=output,
            metadata=metadata,
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
    extras = dict(extra or {})
    if error is not None:
        if extras:
            # Error receipts carry no URL/citation authority (§3.4).
            raise ValueError("error receipts may not carry extra receipt fields")
        receipt["error"] = dict(error)
        return receipt
    for key, value in extras.items():
        if key in receipt:
            raise ValueError(f"receipt field {key!r} may not be overridden")
        expected = RECEIPT_EXTRA_FIELDS.get(key)
        if expected is None:
            raise ValueError(f"receipt field {key!r} is outside the closed schema")
        if isinstance(value, bool) or not isinstance(value, expected):
            raise ValueError(f"receipt field {key!r} has an invalid type")
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
    "ACTION_KIND_FOR_TOOL",
    "FETCH_CODE_PREFIX",
    "HOSTED_ERROR_CODES",
    "MAX_RESULT_BYTES",
    "MAX_RESULT_TEXT_CHARS",
    "RECEIPT_EXTRA_FIELDS",
    "RESULT_KIND_FOR_TOOL",
    "UNEXPECTED_EXECUTION_FAILURE_MESSAGE",
    "ExecutionCancelCheck",
    "HostedExecutionScope",
    "HostedTool",
    "HostedToolCatalog",
    "HostedToolError",
    "HostedToolExecutor",
    "HostedToolSuccess",
    "ResultBudgetExceeded",
    "build_receipt",
    "canonical_json",
    "current_execution_scope",
    "failure_result",
    "reset_execution_scope",
    "result_identities",
    "set_execution_scope",
    "validate_result_payload",
    "validate_sealed_action",
]
