"""Hosted web tools bound to the one SafePublicFetch transport boundary.

``SafePublicFetch`` remains the sole URL transport: every DNS answer and
redirect hop is public-policy checked and connect-time pinned there, with
``trust_env=False`` and no credential or cookie surface. This module only
namespaces its fail-closed codes into ``HOSTED_ERROR_CODES`` (``fetch_``
prefix) and types provider absence/auth failures (F4/F5). Provider secrets
never enter a message: auth failures are reported with one fixed sentence.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from ..utils.safe_public_fetch import SafePublicFetch, SafePublicFetchError
from .hosted import (
    FETCH_CODE_PREFIX,
    HostedToolError,
    HostedToolSuccess,
    current_execution_scope,
)

SearchProvider = Callable[[str], Awaitable[Sequence[Mapping[str, Any]]]]

_DEFAULT_FETCH_MEDIA_TYPES = (
    "text/html",
    "text/plain",
    "text/markdown",
    "application/json",
)


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

    async def invoke(self, arguments: Mapping[str, Any]) -> HostedToolSuccess:
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
        return HostedToolSuccess(
            payload={"query": query, "results": sanitized},
            receipt_fields={"result_count": len(sanitized)},
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

    async def invoke(self, arguments: Mapping[str, Any]) -> HostedToolSuccess:
        url = arguments.get("url")
        if not isinstance(url, str) or not url.strip():
            raise HostedToolError(
                "invalid_tool_arguments",
                "web_fetch requires a non-empty string 'url'",
            )
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
        text = resource.content.decode("utf-8", errors="replace")
        if len(text) > self._max_text_chars:
            raise HostedToolError(
                f"{FETCH_CODE_PREFIX}token_budget",
                "fetched content exceeds the hosted token budget",
            )
        digest = hashlib.sha256(resource.content).hexdigest()
        return HostedToolSuccess(
            payload={
                "url": resource.final_url,
                "media_type": resource.media_type,
                "content": text,
            },
            receipt_fields={
                "final_url": resource.final_url,
                "mime": resource.media_type,
                "result_digest": f"sha256:{digest}",
            },
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
            if isinstance(value, str) and value:
                entry[key] = value
        if entry:
            sanitized.append(entry)
    return sanitized


__all__ = [
    "HostedWebFetchTool",
    "HostedWebSearchTool",
    "ProviderAuthError",
    "SearchProvider",
]
