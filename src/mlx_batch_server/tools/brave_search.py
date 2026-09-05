"""Closed Brave Search provider for the canonical hosted web-search tool."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import httpx

from .hosted import HostedToolError
from .hosted_web import ProviderAuthError

BRAVE_WEB_SEARCH_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_MAX_QUERY_CHARACTERS = 400
_MAX_QUERY_WORDS = 50
_RESULT_COUNT = 5
_INVALID_ARGUMENTS_MESSAGE = "web search query exceeds provider limits"
_TIMEOUT_MESSAGE = "web search provider timed out"
_EXECUTION_FAILURE_MESSAGE = "web search provider request failed"
_INVALID_RESULT_MESSAGE = "web search provider returned an invalid response"


class BraveSearchProvider:
    """Perform one bounded Brave Web Search request per admitted tool call."""

    __slots__ = ("_api_key", "_transport")

    def __init__(
        self,
        api_key: str,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("api_key must be a non-empty string")
        self._api_key = api_key.strip()
        self._transport = transport

    async def __call__(self, query: str) -> Sequence[Mapping[str, Any]]:
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > _MAX_QUERY_CHARACTERS
            or len(query.split()) > _MAX_QUERY_WORDS
        ):
            raise HostedToolError(
                "invalid_tool_arguments",
                _INVALID_ARGUMENTS_MESSAGE,
            )

        timeout = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
        try:
            async with httpx.AsyncClient(
                timeout=timeout,
                transport=self._transport,
                trust_env=False,
                follow_redirects=False,
            ) as client:
                response = await client.get(
                    BRAVE_WEB_SEARCH_ENDPOINT,
                    params={"q": query, "count": _RESULT_COUNT},
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": self._api_key,
                    },
                )
        except httpx.TimeoutException:
            raise HostedToolError("tool_timeout", _TIMEOUT_MESSAGE) from None
        except httpx.HTTPError:
            raise HostedToolError(
                "tool_execution_failed",
                _EXECUTION_FAILURE_MESSAGE,
            ) from None

        if response.status_code in {401, 403}:
            raise ProviderAuthError
        if not 200 <= response.status_code < 300:
            raise HostedToolError(
                "tool_execution_failed",
                _EXECUTION_FAILURE_MESSAGE,
            )

        try:
            payload = response.json()
        except ValueError:
            raise HostedToolError(
                "invalid_tool_result",
                _INVALID_RESULT_MESSAGE,
            ) from None
        return _normalize_results(payload)


def _normalize_results(payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, Mapping) or "web" not in payload:
        _raise_invalid_result()
    web = payload["web"]
    if web is None:
        return []
    if not isinstance(web, Mapping) or "results" not in web:
        _raise_invalid_result()
    results = web["results"]
    if not isinstance(results, Sequence) or isinstance(results, str | bytes):
        _raise_invalid_result()

    normalized: list[dict[str, str]] = []
    for row in results[:_RESULT_COUNT]:
        if not isinstance(row, Mapping):
            _raise_invalid_result()
        title = row.get("title")
        url = row.get("url")
        description = row.get("description")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (title, url, description)
        ):
            _raise_invalid_result()
        normalized.append(
            {
                "title": title,
                "url": url,
                "snippet": description,
            }
        )
    return normalized


def _raise_invalid_result() -> None:
    raise HostedToolError("invalid_tool_result", _INVALID_RESULT_MESSAGE)


__all__ = ["BRAVE_WEB_SEARCH_ENDPOINT", "BraveSearchProvider"]
