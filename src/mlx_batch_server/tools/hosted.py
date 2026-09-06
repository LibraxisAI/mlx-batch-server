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
import base64
import contextvars
import hashlib
import json
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

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
        "rate_limited",
        "connected_peer_mismatch",
        "connected_peer_unverified",
        "unsupported_content_encoding",
        "invalid_content_encoding",
        "decoded_bytes_exceeded",
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
        "url_not_in_prior_context",
        "fetch_url_too_long",
        "fetch_invalid_pdf",
        "admission_evidence_failed",
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
        "source_bytes": int,
        "raw_source_bytes": int,
        "decoded_source_bytes": int,
        "content_encoding": str,
        "content_tokenizer": str,
        "content_tokens": int,
        "content_token_limit": int,
        "content_truncation": str,
        "transport_receipt": Mapping,
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
        "open_page": "document",
        "find_in_page": "find_matches",
        "web_search": "search_results",
    }
)
ACTION_KIND_FOR_TOOL: Mapping[str, str] = MappingProxyType(
    {
        "web_fetch": "fetch",
        "open_page": "open_page",
        "find_in_page": "find_in_page",
        "web_search": "search",
    }
)
MAX_RESULT_TEXT_CHARS = 262_144
MAX_RESULT_BYTES = 1_048_576

_TEXT_DOCUMENT_RESULT_KEYS = frozenset(
    {
        "kind",
        "representation",
        "url",
        "media_type",
        "content",
        "digest",
        "retrieved_at",
    }
)
_PDF_DOCUMENT_RESULT_KEYS = frozenset(
    {
        "kind",
        "representation",
        "url",
        "media_type",
        "content",
        "extracted_text",
        "digest",
        "retrieved_at",
    }
)
_SEARCH_RESULT_KEYS = frozenset({"kind", "query", "results", "digest"})
_SEARCH_ENTRY_KEYS = frozenset({"title", "url", "snippet"})
_FIND_RESULT_KEYS = frozenset(
    {"kind", "url", "pattern", "matches", "digest", "retrieved_at"}
)
_FIND_MATCH_KEYS = frozenset({"start", "end", "excerpt"})
_SEARCH_ACTION_KEYS = frozenset({"kind", "query", "sources"})
_FETCH_ACTION_KEYS = frozenset({"kind", "url"})
_FIND_ACTION_KEYS = frozenset({"kind", "url", "pattern"})
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
    elif expected_kind == "search_results":
        validated = _validate_search_result(result)
    else:
        validated = _validate_find_result(result)
    if len(canonical_json(validated).encode("utf-8")) > MAX_RESULT_BYTES:
        raise ResultBudgetExceeded("result payload exceeds the per-call byte bound")
    return validated


def _validate_document_result(result: Mapping[str, Any]) -> dict[str, Any]:
    keys = set(result)
    representation = result.get("representation")
    if representation == "text" and keys != _TEXT_DOCUMENT_RESULT_KEYS:
        raise ValueError("text document result carries exactly its closed key set")
    if representation == "base64" and keys != _PDF_DOCUMENT_RESULT_KEYS:
        raise ValueError("PDF document result carries exactly its closed key set")
    if representation not in {"text", "base64"}:
        raise ValueError("document representation must be text or base64")
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
        "representation": representation,
        "url": _require_result_identity("url", result["url"]),
        "media_type": _require_result_identity("media_type", result["media_type"]),
        "content": content,
        "digest": _require_result_digest(result["digest"]),
        "retrieved_at": retrieved_at,
    }
    if representation == "base64":
        if validated["media_type"] != "application/pdf":
            raise ValueError("base64 document result must be application/pdf")
        extracted_text = result["extracted_text"]
        if not isinstance(extracted_text, str):
            raise ValueError("PDF extracted_text must be a string")
        try:
            raw = base64.b64decode(content, validate=True)
        except ValueError as error:
            raise ValueError("PDF content must be strict base64") from error
        expected = f"sha256:{hashlib.sha256(raw).hexdigest()}"
        if expected != validated["digest"]:
            raise ValueError("PDF content digest does not match raw bytes")
        if len(raw) > 262_144 or len(extracted_text) > MAX_RESULT_TEXT_CHARS:
            raise ResultBudgetExceeded("PDF result exceeds its per-call bound")
        validated["extracted_text"] = extracted_text
    if representation == "text" and len(content) > MAX_RESULT_TEXT_CHARS:
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


def _validate_find_result(result: Mapping[str, Any]) -> dict[str, Any]:
    if set(result) != _FIND_RESULT_KEYS:
        raise ValueError("find result carries exactly its closed key set")
    matches = result["matches"]
    if isinstance(matches, str | bytes) or not isinstance(matches, Sequence):
        raise ValueError("find matches must be a sequence")
    validated_matches: list[dict[str, Any]] = []
    for match in matches:
        if not isinstance(match, Mapping) or set(match) != _FIND_MATCH_KEYS:
            raise ValueError("find match carries exactly start, end and excerpt")
        start = match["start"]
        end = match["end"]
        excerpt = match["excerpt"]
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            raise ValueError("find match offsets are invalid")
        validated_matches.append(
            {
                "start": start,
                "end": end,
                "excerpt": _require_result_identity("excerpt", excerpt),
            }
        )
    retrieved_at = result["retrieved_at"]
    if (
        isinstance(retrieved_at, bool)
        or not isinstance(retrieved_at, int)
        or retrieved_at < 0
    ):
        raise ValueError("retrieved_at must be a non-negative UTC integer")
    return {
        "kind": "find_matches",
        "url": _require_result_identity("url", result["url"]),
        "pattern": _require_result_identity("pattern", result["pattern"]),
        "matches": validated_matches,
        "digest": _require_result_digest(result["digest"]),
        "retrieved_at": retrieved_at,
    }


def result_identities(result: Mapping[str, Any]) -> tuple[str, ...]:
    """The URL identities a validated result proves for later sealed actions."""

    if result["kind"] in {"document", "find_matches"}:
        return (result["url"],)
    return tuple(entry["url"] for entry in result["results"])


def _validate_search_action(
    action: Mapping[str, Any],
    result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if set(action) != _SEARCH_ACTION_KEYS:
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
        proven = set(result_identities(validate_result_payload("web_search", result)))
        if not set(validated_sources) <= proven:
            raise ValueError("search action sources are not proven by the result")
    return {"kind": "search", "query": query, "sources": list(validated_sources)}


def _validate_find_action(
    action: Mapping[str, Any],
    result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if set(action) != _FIND_ACTION_KEYS:
        raise ValueError("find action carries exactly kind, url and pattern")
    validated = {
        "kind": "find_in_page",
        "url": _require_result_identity("url", action["url"]),
        "pattern": _require_result_identity("pattern", action["pattern"]),
    }
    if result is not None:
        validate_result_payload("find_in_page", result)
    return validated


def _validate_fetch_action(
    tool_name: str,
    expected_kind: str,
    action: Mapping[str, Any],
    result: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if set(action) != _FETCH_ACTION_KEYS:
        raise ValueError("fetch action carries exactly kind and url")
    url = _require_result_identity("url", action["url"])
    if result is not None:
        validate_result_payload(tool_name, result)
    return {"kind": expected_kind, "url": url}


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
    if expected_kind == "search":
        return _validate_search_action(action, result)
    if expected_kind == "find_in_page":
        return _validate_find_action(action, result)
    return _validate_fetch_action(tool_name, expected_kind, action, result)


def _verify_result_receipt_agreement(
    result: Mapping[str, Any],
    receipt_fields: Mapping[str, Any],
) -> None:
    """Fail closed unless receipt provenance mechanically matches the result."""

    if receipt_fields.get("result_digest") != result["digest"]:
        raise ValueError("receipt result_digest does not match the result digest")
    if (
        result["kind"] in {"document", "find_matches"}
        and receipt_fields.get("final_url") != result["url"]
    ):
        raise ValueError("receipt final_url does not match the result url")
    if (
        result["kind"] == "document"
        and receipt_fields.get("mime") != result["media_type"]
    ):
        raise ValueError("receipt mime does not match the result media_type")


class HostedRoundMode(StrEnum):
    """The only three child-round capability states."""

    ACTION_SELECTION = "action_selection"
    SUCCESS_FOLLOWUP = "success_followup"
    FAILURE_CONTINUATION = "failure_continuation"


@dataclass(frozen=True, slots=True)
class HostedExecutionPolicy:
    """One immutable request/round policy snapshot consumed by hosted tools."""

    protocol: Literal["openai_responses", "anthropic_messages"]
    max_uses: int = 8
    max_content_tokens: int | None = None
    max_url_chars: int | None = None
    allowed_domains: tuple[str, ...] = ()
    blocked_domains: tuple[str, ...] = ()
    prior_urls: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.protocol not in {"openai_responses", "anthropic_messages"}:
            raise ValueError("hosted policy protocol is unknown")
        if isinstance(self.max_uses, bool) or self.max_uses < 1:
            raise ValueError("hosted policy max_uses must be positive")
        if self.max_content_tokens is not None and (
            isinstance(self.max_content_tokens, bool) or self.max_content_tokens < 1
        ):
            raise ValueError("hosted policy max_content_tokens must be positive")
        if self.max_url_chars is not None and (
            isinstance(self.max_url_chars, bool) or self.max_url_chars < 1
        ):
            raise ValueError("hosted policy max_url_chars must be positive")
        if self.allowed_domains and self.blocked_domains:
            raise ValueError(
                "allowed_domains and blocked_domains are mutually exclusive"
            )
        for field_name, values in (
            ("allowed_domains", self.allowed_domains),
            ("blocked_domains", self.blocked_domains),
        ):
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{field_name} entries must be non-empty strings")
        if any(not isinstance(value, str) or not value for value in self.prior_urls):
            raise ValueError("prior_urls entries must be non-empty strings")

    def with_prior_urls(self, values: Sequence[str]) -> HostedExecutionPolicy:
        return HostedExecutionPolicy(
            protocol=self.protocol,
            max_uses=self.max_uses,
            max_content_tokens=self.max_content_tokens,
            max_url_chars=self.max_url_chars,
            allowed_domains=self.allowed_domains,
            blocked_domains=self.blocked_domains,
            prior_urls=frozenset(values),
        )


@dataclass(frozen=True, slots=True)
class HostedToolPlan:
    """Closed translation from one protocol declaration to model tools."""

    protocol: Literal["openai_responses", "anthropic_messages"]
    public_names: tuple[str, ...]
    executable_names: frozenset[str]
    model_tools: tuple[Mapping[str, Any], ...]
    policy: HostedExecutionPolicy


class _FrozenJSONDict(dict[str, Any]):
    """JSON-compatible mapping that cannot be mutated after construction."""

    @staticmethod
    def _immutable(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("hosted model tool descriptors are immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    __ior__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _function_descriptor(
    name: str,
    description: str,
    properties: Mapping[str, Mapping[str, Any]],
    required: Sequence[str],
) -> Mapping[str, Any]:
    return _FrozenJSONDict(
        {
            "type": "function",
            "name": name,
            "description": description,
            "parameters": _FrozenJSONDict(
                {
                    "type": "object",
                    "properties": _FrozenJSONDict(
                        {
                            key: _FrozenJSONDict(dict(value))
                            for key, value in properties.items()
                        }
                    ),
                    "required": tuple(required),
                    "additionalProperties": False,
                }
            ),
        }
    )


_MODEL_TOOL_DESCRIPTORS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "web_search": _function_descriptor(
            "web_search",
            "Search the public web for current information.",
            {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Search query",
                }
            },
            ("query",),
        ),
        "open_page": _function_descriptor(
            "open_page",
            "Open a URL previously supplied by the user or returned by web_search.",
            {
                "url": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Previously known URL",
                }
            },
            ("url",),
        ),
        "find_in_page": _function_descriptor(
            "find_in_page",
            "Find literal text in a previously supplied or searched page.",
            {
                "url": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Previously known URL",
                },
                "pattern": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Literal text to find",
                },
            },
            ("url", "pattern"),
        ),
        "web_fetch": _function_descriptor(
            "web_fetch",
            "Fetch one public URL explicitly present in the conversation.",
            {
                "url": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Previously known URL",
                }
            },
            ("url",),
        ),
    }
)


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
    policy: HostedExecutionPolicy | None = None

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

    async def invoke(
        self,
        arguments: Mapping[str, Any],
        *,
        policy: HostedExecutionPolicy,
    ) -> HostedToolSuccess: ...


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
        self.receipt_fields = _freeze_json_mapping(receipt_fields or {})
        self.result = result


def _freeze_json_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    def freeze(item: Any) -> Any:
        if isinstance(item, Mapping):
            if any(not isinstance(key, str) for key in item):
                raise TypeError("receipt field keys must be strings")
            return MappingProxyType({key: freeze(val) for key, val in item.items()})
        if isinstance(item, list | tuple):
            return tuple(freeze(val) for val in item)
        return item

    frozen = freeze(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - type guard
        raise TypeError("receipt fields must be a mapping")
    return frozen


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

    def plan_for(self, request: Any) -> HostedToolPlan | None:
        """Translate one immutable GenerationRequest without mutating it.

        The two public protocols deliberately use disjoint hosted declarations:
        Responses says ``web_search`` and gains the closed search/page/find
        model arsenal; Anthropic says ``web_fetch`` and gains only that server
        tool. A mixed hosted/client declaration has no executable meaning.
        """

        declarations = tuple(
            tool for tool in getattr(request, "tools", ()) if isinstance(tool, Mapping)
        )
        hosted: list[Mapping[str, Any]] = []
        client: list[Mapping[str, Any]] = []
        for tool in declarations:
            kind = tool.get("type")
            name = tool.get("name")
            if kind == "web_search" or (kind == "web_fetch" and name == "web_fetch"):
                hosted.append(tool)
            else:
                if kind == "function" and name in {"open_page", "find_in_page"}:
                    raise ValueError("client tools cannot claim internal hosted names")
                client.append(tool)
        if hosted and client:
            raise ValueError("hosted tools cannot be mixed with client-owned tools")
        if not hosted:
            return None
        if len(hosted) != 1:
            raise ValueError("exactly one public hosted tool declaration is supported")
        declaration = hosted[0]
        public_name = str(declaration.get("name") or declaration.get("type"))
        if public_name == "web_search":
            protocol: Literal["openai_responses", "anthropic_messages"] = (
                "openai_responses"
            )
            desired = ("web_search", "open_page", "find_in_page")
            policy = HostedExecutionPolicy(protocol=protocol)
        elif public_name == "web_fetch":
            protocol = "anthropic_messages"
            desired = ("web_fetch",)
            policy = HostedExecutionPolicy(
                protocol=protocol,
                max_uses=_positive_int(declaration.get("max_uses"), default=8),
                max_content_tokens=_optional_positive_int(
                    declaration.get("max_content_tokens")
                ),
                max_url_chars=250,
                allowed_domains=_string_tuple(declaration.get("allowed_domains")),
                blocked_domains=_string_tuple(declaration.get("blocked_domains")),
            )
        else:  # pragma: no cover - guarded by hosted classification
            raise ValueError("unknown public hosted tool")
        executable = frozenset(name for name in desired if name in self._tools)
        return HostedToolPlan(
            protocol=protocol,
            public_names=(public_name,),
            executable_names=executable,
            model_tools=tuple(
                _MODEL_TOOL_DESCRIPTORS[name] for name in desired if name in executable
            ),
            policy=policy,
        )

    def __bool__(self) -> bool:
        return bool(self._tools)


def _positive_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("hosted policy integer must be positive")
    return value


def _optional_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    return _positive_int(value, default=1)


def _string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError("hosted domain policy must be a sequence")
    result = tuple(str(item).strip().lower() for item in value)
    if any(not item for item in result):
        raise ValueError("hosted domain policy entries must not be empty")
    return result


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
        if scope.policy is None:
            return self._failure(
                call,
                "tool_not_allowed",
                "hosted execution policy is missing",
                started,
            )
        try:
            async with asyncio.timeout(self._per_call_timeout_s):
                success = await tool.invoke(arguments, policy=scope.policy)
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
    "HostedExecutionPolicy",
    "HostedExecutionScope",
    "HostedRoundMode",
    "HostedTool",
    "HostedToolCatalog",
    "HostedToolError",
    "HostedToolExecutor",
    "HostedToolPlan",
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
