"""RED contracts for hosted web tools and the one hosted error registry."""

from __future__ import annotations

import asyncio
import base64
import json
import socket

import httpx
import pytest

from mlx_batch_server.tools.hosted import (
    HOSTED_ERROR_CODES,
    HostedExecutionPolicy,
    HostedExecutionScope,
    HostedToolCatalog,
    HostedToolError,
    HostedToolExecutor,
    HostedToolSuccess,
    reset_execution_scope,
    set_execution_scope,
    validate_result_payload,
)
from mlx_batch_server.tools.hosted_web import (
    AnthropicApproxTextTokenizer,
    HostedFindInPageTool,
    HostedOpenPageTool,
    HostedWebFetchTool,
    HostedWebSearchTool,
    ProviderAuthError,
)
from mlx_batch_server.tools.parser import ParsedToolCall
from mlx_batch_server.utils.safe_public_fetch import (
    FetchHopReceipt,
    FetchTransportReceipt,
    FetchedResource,
    SafePublicFetch,
    SafePublicFetchLimits,
)

_SECRET = "sk-super-secret-brave-key-12345"


def _addrinfo(ip: str = "1.1.1.1"):
    def resolver(host: str, port: int, *args: object, **kwargs: object):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    return resolver


def _fetch(handler, *, max_bytes: int = 4096) -> SafePublicFetch:
    def attested_handler(request: httpx.Request) -> httpx.Response:
        response = handler(request)
        response.extensions["mlx_batch_server.connected_peer"] = request.url.host
        return response

    return SafePublicFetch(
        limits=SafePublicFetchLimits(max_bytes=max_bytes, timeout=2.0),
        transport=httpx.MockTransport(attested_handler),
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


def _policy(
    protocol: str = "openai_responses",
    *,
    prior_urls: tuple[str, ...] = (),
) -> HostedExecutionPolicy:
    return HostedExecutionPolicy(  # type: ignore[arg-type]
        protocol=protocol,
        max_url_chars=250 if protocol == "anthropic_messages" else None,
        prior_urls=frozenset(prior_urls),
    )


async def _execute(
    executor: HostedToolExecutor,
    call: ParsedToolCall,
    *,
    policy: HostedExecutionPolicy | None = None,
):
    token = set_execution_scope(
        HostedExecutionScope(policy=policy or _policy())
    )
    try:
        return await executor.execute(call)
    finally:
        reset_execution_scope(token)


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
    result = await _execute(executor, _call("web_search", '{"query":"loctree"}'))

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
    result = await _execute(executor, _call("web_search", '{"query":"q"}'))

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
        await tool.invoke(
            {"url": "http://169.254.169.254/latest/meta-data"},
            policy=_policy(
                "anthropic_messages",
                prior_urls=("http://169.254.169.254/latest/meta-data",),
            ),
        )
    assert error.value.code == "fetch_url_target_blocked"
    assert error.value.code in HOSTED_ERROR_CODES
    assert "169.254" not in error.value.message


@pytest.mark.asyncio
async def test_byte_limit_is_a_typed_fetch_error() -> None:
    tool = HostedWebFetchTool(fetch=_fetch(_text_handler, max_bytes=4))
    with pytest.raises(HostedToolError) as error:
        await tool.invoke(
            {"url": "https://cdn.example/page"},
            policy=_policy(
                "anthropic_messages", prior_urls=("https://cdn.example/page",)
            ),
        )
    assert error.value.code == "fetch_source_bytes_exceeded"


@pytest.mark.asyncio
async def test_token_budget_overflow_is_a_typed_fetch_error() -> None:
    tool = HostedWebFetchTool(fetch=_fetch(_text_handler), max_text_chars=4)
    with pytest.raises(HostedToolError) as error:
        await tool.invoke(
            {"url": "https://cdn.example/page"},
            policy=_policy(
                "anthropic_messages", prior_urls=("https://cdn.example/page",)
            ),
        )
    assert error.value.code == "fetch_token_budget"


@pytest.mark.asyncio
async def test_disabled_fetch_is_provider_unavailable() -> None:
    tool = HostedWebFetchTool(fetch=None)
    with pytest.raises(HostedToolError) as error:
        await tool.invoke(
            {"url": "https://cdn.example/page"},
            policy=_policy(
                "anthropic_messages", prior_urls=("https://cdn.example/page",)
            ),
        )
    assert error.value.code == "provider_unavailable"


@pytest.mark.asyncio
async def test_successful_fetch_produces_digest_provenance() -> None:
    tool = HostedWebFetchTool(fetch=_fetch(_text_handler))
    success = await tool.invoke(
        {"url": "https://cdn.example/page"},
        policy=_policy(
            "anthropic_messages", prior_urls=("https://cdn.example/page",)
        ),
    )

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
    transport = fields["transport_receipt"]
    assert transport["requested_url"] == "https://cdn.example/page"
    assert transport["hops"][0]["connected_peer"] == "1.1.1.1"
    with pytest.raises(TypeError):
        fields["final_url"] = "https://attacker.example"  # type: ignore[index]
    with pytest.raises(TypeError):
        transport["hops"][0]["connected_peer"] = "127.0.0.1"


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
    success = await tool.invoke({"query": "loctree"}, policy=_policy())
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
async def test_search_producer_admits_only_exact_closed_entries_in_order() -> None:
    """Rows missing any canonical field are dropped, never widened or faked."""

    async def provider(query: str):
        return [
            {"title": "A", "url": "https://a.example", "snippet": "sa"},
            {"title": "no url", "snippet": "s"},
            {"url": "https://only-url.example"},
            {"title": "B", "url": "https://b.example", "snippet": "sb", "rank": 2},
            {"title": "  ", "url": "https://blank-title.example", "snippet": "s"},
            {"title": "C", "url": "https://c.example", "snippet": 3},
            "not-a-mapping",
            {"title": "D", "url": "https://d.example", "snippet": "sd"},
        ]

    tool = HostedWebSearchTool(provider=provider)
    success = await tool.invoke({"query": "q"}, policy=_policy())
    result = success.result
    assert result is not None
    assert result["results"] == [
        {"title": "A", "url": "https://a.example", "snippet": "sa"},
        {"title": "B", "url": "https://b.example", "snippet": "sb"},
        {"title": "D", "url": "https://d.example", "snippet": "sd"},
    ]
    for entry in result["results"]:
        assert set(entry) == {"title", "url", "snippet"}
    # The produced canonical payload passes the closed validator unchanged.
    assert validate_result_payload("web_search", result) == dict(result)


@pytest.mark.asyncio
async def test_executor_types_timeout_invalid_args_and_crash() -> None:
    class _SlowTool:
        name = "slow"

        def describe(self):
            return {"name": "slow"}

        async def invoke(self, arguments, *, policy):
            import asyncio

            await asyncio.sleep(30.0)

    class _CrashTool:
        name = "crash"

        def describe(self):
            return {"name": "crash"}

        async def invoke(self, arguments, *, policy):
            raise RuntimeError("backend exploded")

    catalog = HostedToolCatalog((_SlowTool(), _CrashTool()))
    executor = HostedToolExecutor(catalog, per_call_timeout_s=0.05)

    timeout = await _execute(executor, _call("slow", "{}"))
    assert timeout.metadata is not None
    assert timeout.metadata["error_code"] == "tool_timeout"

    crash = await _execute(executor, _call("crash", "{}"))
    assert crash.metadata is not None
    assert crash.metadata["error_code"] == "tool_execution_failed"

    bad_json = await _execute(executor, _call("crash", "{not json"))
    assert bad_json.metadata is not None
    assert bad_json.metadata["error_code"] == "invalid_tool_arguments"

    dupes = await _execute(executor, _call("crash", '{"a":1,"a":2}'))
    assert dupes.metadata is not None
    assert dupes.metadata["error_code"] == "invalid_tool_arguments"

    unknown = await _execute(executor, _call("nope", "{}"))
    assert unknown.metadata is not None
    assert unknown.metadata["error_code"] == "tool_not_allowed"


def test_catalog_is_immutable_and_rejects_duplicates() -> None:
    tool = HostedWebSearchTool()
    with pytest.raises(ValueError):
        HostedToolCatalog((tool, HostedWebSearchTool()))
    catalog = HostedToolCatalog((tool,))
    assert catalog.names == frozenset({"web_search"})
    assert not HostedToolCatalog()


class _ResourceFetch:
    def __init__(self, content: bytes, media_type: str = "text/plain") -> None:
        self.content = content
        self.media_type = media_type
        self.calls = 0

    async def fetch(self, url: str, **kwargs):
        self.calls += 1
        hop = FetchHopReceipt(
            url, ("1.1.1.1",), "1.1.1.1", "1.1.1.1", 200, None,
            "identity", len(self.content), len(self.content),
        )
        return FetchedResource(
            self.content,
            self.media_type,
            url,
            transport_receipt=FetchTransportReceipt(url, url, (hop,)),
        )


@pytest.mark.asyncio
async def test_anthropic_url_250_reaches_transport_and_251_does_not() -> None:
    base = "https://example.com/"
    url_250 = base + "a" * (250 - len(base))
    url_251 = url_250 + "b"
    fetch = _ResourceFetch(b"ok")
    tool = HostedWebFetchTool(fetch=fetch)  # type: ignore[arg-type]

    await tool.invoke(
        {"url": url_250},
        policy=_policy("anthropic_messages", prior_urls=(url_250,)),
    )
    with pytest.raises(HostedToolError) as error:
        await tool.invoke(
            {"url": url_251},
            policy=_policy("anthropic_messages", prior_urls=(url_251,)),
        )

    assert error.value.code == "fetch_url_too_long"
    assert fetch.calls == 1


def test_anthropic_approx_tokenizer_freezes_unicode_and_punctuation_vectors() -> None:
    tokenizer = AnthropicApproxTextTokenizer()
    assert tokenizer.truncate("abcd efgh!", 3) == ("abcd efgh!", 3, False)
    assert tokenizer.truncate("abcd efgh!", 2) == ("abcd efgh", 2, True)
    assert tokenizer.truncate("żółć", 1) == ("", 0, True)


@pytest.mark.asyncio
async def test_request_scoped_token_policies_do_not_leak_between_concurrent_calls() -> None:
    fetch = _ResourceFetch(b"abcd efgh!")
    tool = HostedWebFetchTool(fetch=fetch)  # type: ignore[arg-type]
    url = "https://example.com/page"
    short, full = await asyncio.gather(
        tool.invoke(
            {"url": url},
            policy=HostedExecutionPolicy(
                protocol="anthropic_messages",
                max_content_tokens=1,
                max_url_chars=250,
                prior_urls=frozenset((url,)),
            ),
        ),
        tool.invoke(
            {"url": url},
            policy=HostedExecutionPolicy(
                protocol="anthropic_messages",
                max_content_tokens=3,
                max_url_chars=250,
                prior_urls=frozenset((url,)),
            ),
        ),
    )
    assert short.result is not None and short.result["content"] == "abcd "
    assert full.result is not None and full.result["content"] == "abcd efgh!"


@pytest.mark.asyncio
async def test_open_and_find_require_prior_url_and_find_is_bounded() -> None:
    fetch = _ResourceFetch(("needle " * 80).encode())
    url = "https://example.com/page"
    policy = _policy("openai_responses", prior_urls=(url,))
    opened = await HostedOpenPageTool(fetch=fetch).invoke(  # type: ignore[arg-type]
        {"url": url}, policy=policy
    )
    found = await HostedFindInPageTool(fetch=fetch).invoke(  # type: ignore[arg-type]
        {"url": url, "pattern": "needle"}, policy=policy
    )
    assert opened.result is not None and opened.result["representation"] == "text"
    assert found.result is not None
    assert found.result["kind"] == "find_matches"
    assert len(found.result["matches"]) == 64
    assert all(len(match["excerpt"]) <= 240 for match in found.result["matches"])
    with pytest.raises(HostedToolError) as error:
        await HostedOpenPageTool(fetch=fetch).invoke(  # type: ignore[arg-type]
            {"url": "https://attacker.example"}, policy=policy
        )
    assert error.value.code == "url_not_in_prior_context"


@pytest.mark.asyncio
async def test_pdf_roundtrip_keeps_raw_bytes_and_honors_text_token_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = b"%PDF-1.7\nclosed fixture"
    fetch = _ResourceFetch(raw, "application/pdf")

    class _Page:
        def get_text(self) -> str:
            return "extracted fixture"

    class _Document:
        def __iter__(self):
            return iter((_Page(),))

        def close(self) -> None:
            return None

    class _PyMuPDF:
        @staticmethod
        def open(*, stream: bytes, filetype: str):
            assert stream == raw and filetype == "pdf"
            return _Document()

    monkeypatch.setattr(
        "mlx_batch_server.tools.hosted_web.importlib.import_module",
        lambda name: _PyMuPDF,
    )
    url = "https://example.com/doc.pdf"
    tool = HostedWebFetchTool(fetch=fetch)  # type: ignore[arg-type]
    result = await tool.invoke(
        {"url": url},
        policy=HostedExecutionPolicy(
            protocol="anthropic_messages",
            max_content_tokens=1,
            max_url_chars=250,
            prior_urls=frozenset((url,)),
        ),
    )
    assert result.result is not None
    assert base64.b64decode(result.result["content"], validate=True) == raw
    assert result.receipt_fields["content_token_limit"] == 1
    assert result.receipt_fields["content_tokens"] <= 1
    assert result.receipt_fields["content_truncation"] == "max_content_tokens"
    assert len(result.result["extracted_text"]) < len("extracted fixture")
    transport = result.receipt_fields["transport_receipt"]
    assert transport["requested_url"] == url
    assert transport["hops"][0]["connected_peer"] == "1.1.1.1"
