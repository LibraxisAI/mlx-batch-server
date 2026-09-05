"""RED contracts for hosted web tools and the one hosted error registry."""

from __future__ import annotations

import json
import socket

import httpx
import pytest

from mlx_batch_server.tools.hosted import (
    HOSTED_ERROR_CODES,
    HostedToolCatalog,
    HostedToolError,
    HostedToolExecutor,
    HostedToolSuccess,
)
from mlx_batch_server.tools.hosted_web import (
    HostedWebFetchTool,
    HostedWebSearchTool,
    ProviderAuthError,
)
from mlx_batch_server.tools.parser import ParsedToolCall
from mlx_batch_server.utils.safe_public_fetch import (
    SafePublicFetch,
    SafePublicFetchLimits,
)

_SECRET = "sk-super-secret-brave-key-12345"


def _addrinfo(ip: str = "1.1.1.1"):
    def resolver(host: str, port: int, *args: object, **kwargs: object):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    return resolver


def _fetch(handler, *, max_bytes: int = 4096) -> SafePublicFetch:
    return SafePublicFetch(
        limits=SafePublicFetchLimits(max_bytes=max_bytes, timeout=2.0),
        transport=httpx.MockTransport(handler),
        getaddrinfo=_addrinfo(),
    )


def _text_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/plain"},
        content=b"hosted fetch body",
        request=request,
    )


def _call(name: str, arguments: str, call_id: str = "call_1") -> ParsedToolCall:
    return ParsedToolCall(index=0, call_id=call_id, name=name, arguments=arguments)


def test_hosted_error_codes_registry_is_frozen_and_covers_f4_to_f10() -> None:
    assert isinstance(HOSTED_ERROR_CODES, frozenset)
    for code in (
        "provider_unavailable",
        "provider_auth_failed",
        "tool_timeout",
        "tool_execution_failed",
        "invalid_tool_result",
        "invalid_tool_arguments",
        "tool_arguments_too_large",
        "continuation_exhausted",
        "tool_round_limit",
        "fetch_url_target_blocked",
        "fetch_dns_resolution_failed",
        "fetch_source_bytes_exceeded",
        "fetch_redirect_limit_exceeded",
        "fetch_unsupported_media_type",
        "fetch_token_budget",
    ):
        assert code in HOSTED_ERROR_CODES, code


def test_hosted_tool_error_rejects_unregistered_codes() -> None:
    with pytest.raises(ValueError):
        HostedToolError("made_up_code", "nope")


@pytest.mark.asyncio
async def test_missing_provider_is_a_typed_error_receipt_not_success() -> None:
    """F4: the legacy adapter returned apparent success; this kills that bug."""

    catalog = HostedToolCatalog((HostedWebSearchTool(provider=None),))
    executor = HostedToolExecutor(catalog)
    result = await executor.execute(_call("web_search", '{"query":"loctree"}'))

    assert not result.ok
    assert result.metadata is not None
    assert result.metadata["error_code"] == "provider_unavailable"
    receipt = result.metadata["receipt"]
    assert receipt["status"] == "failed"
    assert receipt["attempt"] == 1
    assert receipt["error"]["code"] == "provider_unavailable"
    payload = json.loads(result.output)
    assert payload["error"]["code"] == "provider_unavailable"


@pytest.mark.asyncio
async def test_provider_auth_failure_is_typed_and_secret_free() -> None:
    async def provider(query: str):
        raise ProviderAuthError(f"401 unauthorized for key {_SECRET}")

    catalog = HostedToolCatalog((HostedWebSearchTool(provider=provider),))
    executor = HostedToolExecutor(catalog)
    result = await executor.execute(_call("web_search", '{"query":"q"}'))

    assert not result.ok
    assert result.metadata is not None
    assert result.metadata["error_code"] == "provider_auth_failed"
    assert _SECRET not in result.output
    assert _SECRET not in (result.error or "")
    assert _SECRET not in json.dumps(dict(result.metadata["receipt"]))


@pytest.mark.asyncio
async def test_ssrf_refusal_is_namespaced_into_the_registry() -> None:
    tool = HostedWebFetchTool(fetch=_fetch(_text_handler))
    with pytest.raises(HostedToolError) as error:
        await tool.invoke({"url": "http://169.254.169.254/latest/meta-data"})
    assert error.value.code == "fetch_url_target_blocked"
    assert error.value.code in HOSTED_ERROR_CODES
    assert "169.254" not in error.value.message


@pytest.mark.asyncio
async def test_byte_limit_is_a_typed_fetch_error() -> None:
    tool = HostedWebFetchTool(fetch=_fetch(_text_handler, max_bytes=4))
    with pytest.raises(HostedToolError) as error:
        await tool.invoke({"url": "https://cdn.example/page"})
    assert error.value.code == "fetch_source_bytes_exceeded"


@pytest.mark.asyncio
async def test_token_budget_overflow_is_a_typed_fetch_error() -> None:
    tool = HostedWebFetchTool(fetch=_fetch(_text_handler), max_text_chars=4)
    with pytest.raises(HostedToolError) as error:
        await tool.invoke({"url": "https://cdn.example/page"})
    assert error.value.code == "fetch_token_budget"


@pytest.mark.asyncio
async def test_disabled_fetch_is_provider_unavailable() -> None:
    tool = HostedWebFetchTool(fetch=None)
    with pytest.raises(HostedToolError) as error:
        await tool.invoke({"url": "https://cdn.example/page"})
    assert error.value.code == "provider_unavailable"


@pytest.mark.asyncio
async def test_successful_fetch_produces_digest_provenance() -> None:
    tool = HostedWebFetchTool(fetch=_fetch(_text_handler))
    success = await tool.invoke({"url": "https://cdn.example/page"})

    assert isinstance(success, HostedToolSuccess)
    assert success.payload["content"] == "hosted fetch body"
    fields = success.receipt_fields
    assert fields["final_url"] == "https://cdn.example/page"
    assert fields["mime"] == "text/plain"
    assert fields["result_digest"].startswith("sha256:")
    result = success.result
    assert result is not None
    assert result["kind"] == "document"
    assert result["url"] == fields["final_url"]
    assert result["media_type"] == fields["mime"]
    assert result["content"] == "hosted fetch body"
    assert result["digest"] == fields["result_digest"]
    assert isinstance(result["retrieved_at"], int) and result["retrieved_at"] > 0
    with pytest.raises(TypeError):
        fields["final_url"] = "https://attacker.example"  # type: ignore[index]


@pytest.mark.asyncio
async def test_search_success_sanitizes_results_to_known_fields() -> None:
    async def provider(query: str):
        return [
            {
                "title": "Loctree",
                "url": "https://loctree.dev",
                "snippet": "structural sight",
                "internal_debug": {"socket": "10.0.0.1:8100"},
            },
            "not-a-mapping",
        ]

    tool = HostedWebSearchTool(provider=provider)
    success = await tool.invoke({"query": "loctree"})
    assert success.payload["results"] == [
        {
            "title": "Loctree",
            "url": "https://loctree.dev",
            "snippet": "structural sight",
        }
    ]
    result = success.result
    assert result is not None
    assert result["kind"] == "search_results"
    assert result["query"] == "loctree"
    assert result["results"] == success.payload["results"]
    assert result["digest"] == success.receipt_fields["result_digest"]


@pytest.mark.asyncio
async def test_executor_types_timeout_invalid_args_and_crash() -> None:
    class _SlowTool:
        name = "slow"

        def describe(self):
            return {"name": "slow"}

        async def invoke(self, arguments):
            import asyncio

            await asyncio.sleep(30.0)

    class _CrashTool:
        name = "crash"

        def describe(self):
            return {"name": "crash"}

        async def invoke(self, arguments):
            raise RuntimeError("backend exploded")

    catalog = HostedToolCatalog((_SlowTool(), _CrashTool()))
    executor = HostedToolExecutor(catalog, per_call_timeout_s=0.05)

    timeout = await executor.execute(_call("slow", "{}"))
    assert timeout.metadata is not None
    assert timeout.metadata["error_code"] == "tool_timeout"

    crash = await executor.execute(_call("crash", "{}"))
    assert crash.metadata is not None
    assert crash.metadata["error_code"] == "tool_execution_failed"

    bad_json = await executor.execute(_call("crash", "{not json"))
    assert bad_json.metadata is not None
    assert bad_json.metadata["error_code"] == "invalid_tool_arguments"

    dupes = await executor.execute(_call("crash", '{"a":1,"a":2}'))
    assert dupes.metadata is not None
    assert dupes.metadata["error_code"] == "invalid_tool_arguments"

    unknown = await executor.execute(_call("nope", "{}"))
    assert unknown.metadata is not None
    assert unknown.metadata["error_code"] == "tool_not_allowed"


def test_catalog_is_immutable_and_rejects_duplicates() -> None:
    tool = HostedWebSearchTool()
    with pytest.raises(ValueError):
        HostedToolCatalog((tool, HostedWebSearchTool()))
    catalog = HostedToolCatalog((tool,))
    assert catalog.names == frozenset({"web_search"})
    assert not HostedToolCatalog()
