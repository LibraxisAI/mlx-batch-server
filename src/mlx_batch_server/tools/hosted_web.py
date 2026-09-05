"""Hosted web tools bound to the one SafePublicFetch transport boundary.

``SafePublicFetch`` remains the sole URL transport: every DNS answer and
redirect hop is public-policy checked and connect-time pinned there, with
``trust_env=False`` and no credential or cookie surface. This module only
namespaces its fail-closed codes into ``HOSTED_ERROR_CODES`` (``fetch_``
prefix) and types provider absence/auth failures (F4/F5). Provider secrets
never enter a message: auth failures are reported with one fixed sentence.
"""

from __future__ import annotations

import base64
import hashlib
import importlib
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from ..utils.safe_public_fetch import SafePublicFetch, SafePublicFetchError
from .hosted import (
    FETCH_CODE_PREFIX,
    HostedExecutionPolicy,
    HostedToolError,
    HostedToolSuccess,
    canonical_json,
    current_execution_scope,
)

SearchProvider = Callable[[str], Awaitable[Sequence[Mapping[str, Any]]]]

_DEFAULT_FETCH_MEDIA_TYPES = (
    "text/html",
    "text/plain",
    "text/markdown",
    "application/json",
    "application/pdf",
)

ANTHROPIC_APPROX_TOKENIZER = "anthropic_utf8_approx_v1"
_PDF_MAX_BYTES = 262_144


class ProviderAuthError(Exception):
    """Raised by a search provider client on credential rejection (401/403)."""


class HostedWebSearchTool:
    """Hosted ``web_search`` backed by an injected provider client.

    A missing provider is F4: the tool stays admitted so the failure becomes a
    typed error receipt plus one continuation at execution time, never an
    apparent success (the legacy adapter bug this design kills).
    """

    def __init__(self, *, provider: SearchProvider | None = None) -> None:
        self._provider = provider

    @property
    def name(self) -> str:
        return "web_search"

    def describe(self) -> Mapping[str, Any]:
        return {"type": "web_search", "name": self.name}

    async def invoke(
        self,
        arguments: Mapping[str, Any],
        *,
        policy: HostedExecutionPolicy,
    ) -> HostedToolSuccess:
        _require_protocol(policy, "openai_responses")
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise HostedToolError(
                "invalid_tool_arguments",
                "web_search requires a non-empty string 'query'",
            )
        if self._provider is None:
            raise HostedToolError(
                "provider_unavailable",
                "web search provider is not configured",
            )
        try:
            results = await self._provider(query)
        except ProviderAuthError:
            raise HostedToolError(
                "provider_auth_failed",
                "web search provider rejected the configured credentials",
            ) from None
        except HostedToolError:
            raise
        sanitized = _sanitized_results(results)
        # One digest over the existing canonical compact/sorted JSON bytes of
        # {query, results}; the canonical result and the audit receipt carry
        # the identical value so provenance agrees mechanically downstream.
        digest = (
            "sha256:"
            + hashlib.sha256(
                canonical_json({"query": query, "results": sanitized}).encode("utf-8")
            ).hexdigest()
        )
        return HostedToolSuccess(
            payload={"query": query, "results": sanitized},
            receipt_fields={"result_count": len(sanitized), "result_digest": digest},
            result={
                "kind": "search_results",
                "query": query,
                "results": [dict(entry) for entry in sanitized],
                "digest": digest,
            },
        )


class HostedWebFetchTool:
    """Hosted ``web_fetch``; SafePublicFetch is its only transport."""

    def __init__(
        self,
        *,
        fetch: SafePublicFetch | None = None,
        accepted_media_types: Sequence[str] = _DEFAULT_FETCH_MEDIA_TYPES,
        max_bytes: int | None = None,
        max_text_chars: int = 262_144,
    ) -> None:
        if max_text_chars < 1:
            raise ValueError("max_text_chars must be positive")
        self._fetch = fetch
        self._accepted_media_types = tuple(accepted_media_types)
        self._max_bytes = max_bytes
        self._max_text_chars = max_text_chars

    @property
    def name(self) -> str:
        return "web_fetch"

    def describe(self) -> Mapping[str, Any]:
        return {"type": "web_fetch", "name": self.name}

    async def invoke(
        self,
        arguments: Mapping[str, Any],
        *,
        policy: HostedExecutionPolicy,
    ) -> HostedToolSuccess:
        _require_protocol(policy, "anthropic_messages")
        url = arguments.get("url")
        if not isinstance(url, str) or not url.strip():
            raise HostedToolError(
                "invalid_tool_arguments",
                "web_fetch requires a non-empty string 'url'",
            )
        _require_url_policy(url, policy)
        return await self._fetch_document(url, policy=policy)

    async def _fetch_document(
        self,
        url: str,
        *,
        policy: HostedExecutionPolicy,
    ) -> HostedToolSuccess:
        if self._fetch is None:
            raise HostedToolError(
                "provider_unavailable",
                "web fetch is not enabled for this runtime",
            )
        # The request-scoped scope carries the one absolute deadline and the
        # request's own cancel token into the sole transport; SafePublicFetch
        # fails closed (url_fetch_timeout) on an exhausted budget.
        scope = current_execution_scope()
        try:
            resource = await self._fetch.fetch(
                url,
                accepted_media_types=self._accepted_media_types,
                max_bytes=self._max_bytes,
                cancel=scope.cancel,
                deadline_s=scope.remaining_s(),
            )
        except SafePublicFetchError as error:
            raise HostedToolError(
                f"{FETCH_CODE_PREFIX}{error.code}",
                str(error),
            ) from None
        digest = f"sha256:{hashlib.sha256(resource.content).hexdigest()}"
        retrieved_at = int(time.time())
        common_receipt: dict[str, Any] = {
            "final_url": resource.final_url,
            "mime": resource.media_type,
            "http_status": resource.http_status,
            "redirect_count": resource.redirect_count,
            "result_digest": digest,
            "source_bytes": len(resource.content),
        }
        if resource.media_type == "application/pdf":
            return _pdf_success(
                resource.content,
                final_url=resource.final_url,
                digest=digest,
                retrieved_at=retrieved_at,
                receipt=common_receipt,
            )
        text = resource.content.decode("utf-8", errors="replace")
        if len(text) > self._max_text_chars:
            raise HostedToolError(
                f"{FETCH_CODE_PREFIX}token_budget",
                "fetched content exceeds the hosted result character budget",
            )
        tokenizer = AnthropicApproxTextTokenizer()
        text, token_count, truncated = tokenizer.truncate(
            text, policy.max_content_tokens
        )
        # The digest is over the raw fetched bytes (pre-decode); the canonical
        # result and the receipt share the one value, and no consumer may
        # re-fetch or reinterpret this payload downstream.
        receipt = {
            **common_receipt,
            "content_encoding": "utf8",
            "content_tokenizer": ANTHROPIC_APPROX_TOKENIZER,
            "content_tokens": token_count,
            "content_truncation": (
                "max_content_tokens" if truncated else "none"
            ),
        }
        if policy.max_content_tokens is not None:
            receipt["content_token_limit"] = policy.max_content_tokens
        return HostedToolSuccess(
            payload={
                "url": resource.final_url,
                "media_type": resource.media_type,
                "content": text,
            },
            receipt_fields=receipt,
            result={
                "kind": "document",
                "representation": "text",
                "url": resource.final_url,
                "media_type": resource.media_type,
                "content": text,
                "digest": digest,
                "retrieved_at": retrieved_at,
            },
        )


class HostedOpenPageTool(HostedWebFetchTool):
    """OpenAI-internal page action over the same public fetch boundary."""

    @property
    def name(self) -> str:
        return "open_page"

    def describe(self) -> Mapping[str, Any]:
        return {"type": "open_page", "name": self.name}

    async def invoke(
        self,
        arguments: Mapping[str, Any],
        *,
        policy: HostedExecutionPolicy,
    ) -> HostedToolSuccess:
        _require_protocol(policy, "openai_responses")
        url = arguments.get("url")
        if not isinstance(url, str) or not url.strip():
            raise HostedToolError(
                "invalid_tool_arguments",
                "open_page requires a non-empty string 'url'",
            )
        _require_url_policy(url, policy)
        return await self._fetch_document(url, policy=policy)


class HostedFindInPageTool(HostedWebFetchTool):
    """OpenAI-internal literal find action over a prior URL."""

    @property
    def name(self) -> str:
        return "find_in_page"

    def describe(self) -> Mapping[str, Any]:
        return {"type": "find_in_page", "name": self.name}

    async def invoke(
        self,
        arguments: Mapping[str, Any],
        *,
        policy: HostedExecutionPolicy,
    ) -> HostedToolSuccess:
        _require_protocol(policy, "openai_responses")
        url = arguments.get("url")
        pattern = arguments.get("pattern")
        if not isinstance(url, str) or not url.strip():
            raise HostedToolError(
                "invalid_tool_arguments",
                "find_in_page requires a non-empty string 'url'",
            )
        if not isinstance(pattern, str) or not pattern:
            raise HostedToolError(
                "invalid_tool_arguments",
                "find_in_page requires a non-empty string 'pattern'",
            )
        _require_url_policy(url, policy)
        fetched = await self._fetch_document(url, policy=policy)
        result = fetched.result
        if result is None or result.get("representation") != "text":
            raise HostedToolError(
                f"{FETCH_CODE_PREFIX}unsupported_media_type",
                "find_in_page requires a text-family page",
            )
        content = str(result["content"])
        matches: list[dict[str, Any]] = []
        cursor = 0
        while len(matches) < 64:
            start = content.find(pattern, cursor)
            if start < 0:
                break
            end = start + len(pattern)
            matches.append(
                {
                    "start": start,
                    "end": end,
                    "excerpt": _find_excerpt(content, start=start, end=end),
                }
            )
            cursor = end
        find_result = {
            "kind": "find_matches",
            "url": result["url"],
            "pattern": pattern,
            "matches": matches,
            "digest": result["digest"],
            "retrieved_at": result["retrieved_at"],
        }
        return HostedToolSuccess(
            payload=find_result,
            receipt_fields=fetched.receipt_fields,
            result=find_result,
        )


class AnthropicApproxTextTokenizer:
    """Dependency-free frozen approximation used only for fetch truncation."""

    name = ANTHROPIC_APPROX_TOKENIZER

    def truncate(self, text: str, limit: int | None) -> tuple[str, int, bool]:
        spans = tuple(_token_spans(text))
        total = sum(units for _, _, units in spans)
        if limit is None or total <= limit:
            return text, total, False
        used = 0
        end = 0
        for start, stop, units in spans:
            if used + units > limit:
                end = start
                break
            used += units
            end = stop
        while end < len(text) and text[end].isspace():
            end += 1
        return text[:end], used, True


def _token_spans(text: str) -> Sequence[tuple[int, int, int]]:
    spans: list[tuple[int, int, int]] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        start = index
        if text[index].isalnum():
            index += 1
            while index < len(text) and text[index].isalnum():
                index += 1
            units = math.ceil(len(text[start:index].encode("utf-8")) / 4)
        else:
            index += 1
            units = 1
        spans.append((start, index, units))
    return spans


def _find_excerpt(content: str, *, start: int, end: int) -> str:
    """Return a deterministic at-most-240-character excerpt at the match."""

    width = min(240, len(content))
    left = max(0, start - max(0, (width - (end - start)) // 2))
    right = min(len(content), left + width)
    left = max(0, right - width)
    return content[left:right]


def _require_protocol(policy: HostedExecutionPolicy, expected: str) -> None:
    if not isinstance(policy, HostedExecutionPolicy) or policy.protocol != expected:
        raise HostedToolError(
            "tool_not_allowed",
            "hosted tool is outside this request protocol",
        )


def _require_url_policy(url: str, policy: HostedExecutionPolicy) -> None:
    if policy.max_url_chars is not None and len(url) > policy.max_url_chars:
        raise HostedToolError(
            "fetch_url_too_long",
            "web fetch URL exceeds the protocol limit",
        )
    if url not in policy.prior_urls:
        raise HostedToolError(
            "url_not_in_prior_context",
            "URL was not supplied by the user or a prior hosted result",
        )
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    if policy.allowed_domains and not any(
        host == domain or host.endswith(f".{domain}")
        for domain in policy.allowed_domains
    ):
        raise HostedToolError("fetch_url_not_allowed", "URL domain is not allowed")
    if policy.blocked_domains and any(
        host == domain or host.endswith(f".{domain}")
        for domain in policy.blocked_domains
    ):
        raise HostedToolError("fetch_url_not_allowed", "URL domain is blocked")


def _pdf_success(
    raw: bytes,
    *,
    final_url: str,
    digest: str,
    retrieved_at: int,
    receipt: Mapping[str, Any],
) -> HostedToolSuccess:
    if len(raw) > _PDF_MAX_BYTES or not raw.startswith(b"%PDF-"):
        raise HostedToolError("fetch_invalid_pdf", "fetched PDF is invalid or too large")
    try:
        pymupdf = importlib.import_module("pymupdf")
        document = pymupdf.open(stream=raw, filetype="pdf")
        try:
            extracted = "\n".join(page.get_text() for page in document)
        finally:
            document.close()
    except Exception as error:
        raise HostedToolError("fetch_invalid_pdf", "fetched PDF could not be parsed") from error
    extracted = extracted[:262_144]
    encoded = base64.b64encode(raw).decode("ascii")
    result = {
        "kind": "document",
        "representation": "base64",
        "url": final_url,
        "media_type": "application/pdf",
        "content": encoded,
        "extracted_text": extracted,
        "digest": digest,
        "retrieved_at": retrieved_at,
    }
    return HostedToolSuccess(
        payload={
            "url": final_url,
            "media_type": "application/pdf",
            "content": encoded,
            "extracted_text": extracted,
        },
        receipt_fields={
            **dict(receipt),
            "content_encoding": "base64",
            "content_tokenizer": "not_applicable_pdf",
            "content_truncation": "not_applicable_pdf",
        },
        result=result,
    )


def _sanitized_results(
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    sanitized: list[dict[str, str]] = []
    for item in results:
        if not isinstance(item, Mapping):
            continue
        entry: dict[str, str] = {}
        for key in ("title", "url", "snippet"):
            value = item.get(key)
            if not isinstance(value, str) or not value.strip():
                break
            entry[key] = value
        # A canonical result is exactly {title, url, snippet}; a row missing
        # any of them proves no admissible result and is dropped in place —
        # never widened, subset-admitted, or fabricated (original order kept).
        if len(entry) == 3:
            sanitized.append(entry)
    return sanitized


__all__ = [
    "ANTHROPIC_APPROX_TOKENIZER",
    "AnthropicApproxTextTokenizer",
    "HostedFindInPageTool",
    "HostedOpenPageTool",
    "HostedWebFetchTool",
    "HostedWebSearchTool",
    "ProviderAuthError",
    "SearchProvider",
]
