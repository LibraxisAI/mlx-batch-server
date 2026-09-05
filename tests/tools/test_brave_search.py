"""Contracts for the closed production Brave Search provider adapter."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from mlx_batch_server.core.config import Settings
from mlx_batch_server.tools.brave_search import (
    BRAVE_WEB_SEARCH_ENDPOINT,
    BraveSearchProvider,
)
from mlx_batch_server.tools.hosted import HostedToolError
from mlx_batch_server.tools.hosted_web import ProviderAuthError

_SECRET = "brave-test-secret-never-emit"
_INVALID_ARGUMENTS_MESSAGE = "web search query exceeds provider limits"
_TIMEOUT_MESSAGE = "web search provider timed out"
_EXECUTION_FAILURE_MESSAGE = "web search provider request failed"
_INVALID_RESULT_MESSAGE = "web search provider returned an invalid response"


def _provider(
    handler: Callable[[httpx.Request], httpx.Response],
) -> BraveSearchProvider:
    return BraveSearchProvider(_SECRET, transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_one_exact_request_returns_at_most_five_closed_results() -> None:
    requests: list[httpx.Request] = []
    query = "  exact admitted query  "

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "web": {
                    "results": [
                        {
                            "title": f"Result {index}",
                            "url": f"https://result-{index}.example/page",
                            "description": f"Description {index}",
                            "provider_only": "discarded",
                        }
                        for index in range(6)
                    ],
                    "more_results_available": True,
                },
                "provider_debug": {"ignored": True},
            },
            request=request,
        )

    results = await _provider(handler)(query)

    assert len(requests) == 1
    request = requests[0]
    assert request.method == "GET"
    assert request.url.scheme == "https"
    assert request.url.host == "api.search.brave.com"
    assert request.url.path == "/res/v1/web/search"
    assert (
        f"{request.url.scheme}://{request.url.host}{request.url.path}"
    ) == BRAVE_WEB_SEARCH_ENDPOINT
    assert request.url.params.get("q") == query
    assert request.url.params.get("count") == "5"
    assert request.headers["Accept"] == "application/json"
    assert request.headers["X-Subscription-Token"] == _SECRET
    assert request.extensions["timeout"] == {
        "connect": 5.0,
        "read": 10.0,
        "write": 5.0,
        "pool": 5.0,
    }
    assert results == [
        {
            "title": f"Result {index}",
            "url": f"https://result-{index}.example/page",
            "snippet": f"Description {index}",
        }
        for index in range(5)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    ["", "   ", "x" * 401, " ".join(["word"] * 51)],
)
async def test_invalid_query_fails_before_transport(query: str) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"web": None}, request=request)

    with pytest.raises(HostedToolError) as error:
        await _provider(handler)(query)

    assert error.value.code == "invalid_tool_arguments"
    assert error.value.message == _INVALID_ARGUMENTS_MESSAGE
    assert calls == 0


@pytest.mark.asyncio
async def test_documented_query_boundaries_are_admitted_without_rewriting() -> None:
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        queries.append(request.url.params["q"])
        return httpx.Response(200, json={"web": None}, request=request)

    provider = _provider(handler)
    four_hundred_characters = "x" * 400
    fifty_words = " ".join(["word"] * 50)

    assert await provider(four_hundred_characters) == []
    assert await provider(fifty_words) == []
    assert queries == [four_hundred_characters, fifty_words]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403])
async def test_auth_status_is_typed_without_body_or_secret(status: int) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            status,
            text=f"provider rejected {_SECRET}",
            request=request,
        )

    with pytest.raises(ProviderAuthError) as error:
        await _provider(handler)("query")

    assert calls == 1
    assert _SECRET not in str(error.value)


@pytest.mark.asyncio
async def test_timeout_maps_to_fixed_secret_free_failure() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ReadTimeout(f"timeout with {_SECRET}", request=request)

    with pytest.raises(HostedToolError) as error:
        await _provider(handler)("query")

    assert calls == 1
    assert error.value.code == "tool_timeout"
    assert error.value.message == _TIMEOUT_MESSAGE
    assert _SECRET not in str(error.value)


@pytest.mark.asyncio
async def test_network_failure_maps_to_fixed_secret_free_failure() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError(f"network with {_SECRET}", request=request)

    with pytest.raises(HostedToolError) as error:
        await _provider(handler)("query")

    assert calls == 1
    assert error.value.code == "tool_execution_failed"
    assert error.value.message == _EXECUTION_FAILURE_MESSAGE
    assert _SECRET not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [302, 400, 404, 429, 500, 503])
async def test_non_auth_status_fails_once_without_redirect_or_retry(
    status: int,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            status,
            headers={"location": "https://redirect.example/result"},
            text=f"provider body {_SECRET}",
            request=request,
        )

    with pytest.raises(HostedToolError) as error:
        await _provider(handler)("query")

    assert len(requests) == 1
    assert error.value.code == "tool_execution_failed"
    assert error.value.message == _EXECUTION_FAILURE_MESSAGE
    assert _SECRET not in str(error.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        [],
        {},
        {"web": "invalid"},
        {"web": {}},
        {"web": {"results": "invalid"}},
        {"web": {"results": ["invalid"]}},
        {"web": {"results": [{"title": "T", "url": "https://example"}]}},
        {
            "web": {
                "results": [
                    {
                        "title": "T",
                        "url": "https://example",
                        "description": "   ",
                    }
                ]
            }
        },
    ],
)
async def test_malformed_provider_schema_is_not_empty_success(payload: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    with pytest.raises(HostedToolError) as error:
        await _provider(handler)("query")

    assert error.value.code == "invalid_tool_result"
    assert error.value.message == _INVALID_RESULT_MESSAGE


@pytest.mark.asyncio
async def test_malformed_json_is_not_empty_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{", request=request)

    with pytest.raises(HostedToolError) as error:
        await _provider(handler)("query")

    assert error.value.code == "invalid_tool_result"
    assert error.value.message == _INVALID_RESULT_MESSAGE


@pytest.mark.asyncio
@pytest.mark.parametrize("payload", [{"web": None}, {"web": {"results": []}}])
async def test_genuine_zero_results_remain_success(payload: Any) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload, request=request)

    assert await _provider(handler)("no matches") == []


@pytest.mark.asyncio
async def test_cancellation_propagates_without_failure_conversion() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise asyncio.CancelledError

    provider = BraveSearchProvider(
        _SECRET,
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(asyncio.CancelledError):
        await provider("query")


def test_provider_repr_and_settings_export_never_reveal_the_secret() -> None:
    provider = BraveSearchProvider(_SECRET)
    settings = Settings(_env_file=None, brave_api_key=_SECRET)

    assert _SECRET not in repr(provider)
    assert settings.brave_api_key is not None
    assert settings.brave_api_key.get_secret_value() == _SECRET
    assert settings.to_dict()["brave_api_key"] == "***"


def test_missing_brave_setting_exports_none() -> None:
    settings = Settings(_env_file=None, brave_api_key=None)

    assert settings.brave_api_key is None
    assert settings.to_dict()["brave_api_key"] is None
