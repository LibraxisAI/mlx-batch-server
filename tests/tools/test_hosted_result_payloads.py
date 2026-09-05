"""Delivery verifier for the canonical hosted result producer (W3-HR2-2).

Golden digest fixtures pin the raw fetch bytes, the canonical search order,
every identity, and the receipt digest agreement. ``metadata["result"]`` is
the sole decoded producer channel; failure, cancel, and deadline outcomes
carry no payload; SafePublicFetch is invoked exactly once per fetch and
validation/projection paths touch no network.
"""

from __future__ import annotations

import asyncio
import json
import socket

import httpx
import pytest

from mlx_batch_server.tools.hosted import (
    HOSTED_ERROR_CODES,
    MAX_RESULT_TEXT_CHARS,
    HostedExecutionScope,
    HostedToolCatalog,
    HostedToolError,
    HostedToolExecutor,
    HostedToolSuccess,
    ResultBudgetExceeded,
    reset_execution_scope,
    set_execution_scope,
    validate_result_payload,
    validate_sealed_action,
)
from mlx_batch_server.tools.hosted_web import HostedWebFetchTool, HostedWebSearchTool
from mlx_batch_server.tools.parser import ParsedToolCall
from mlx_batch_server.utils.safe_public_fetch import (
    SafePublicFetch,
    SafePublicFetchLimits,
)

# Golden fixtures: change the raw fetch bytes, the canonical search order,
# any identity, or the digest algorithm and these assertions fail.
_FETCH_BODY = b"hosted fetch body"
_FETCH_DIGEST = (
    "sha256:56db332b5259c058a9c49ee237c7c3e8e3da56bcf1716ab2c2ef7e7ce8654150"
)
_FETCH_OUTPUT = (
    '{"content":"hosted fetch body","media_type":"text/plain",'
    '"url":"https://cdn.example/page"}'
)
_SEARCH_DIGEST = (
    "sha256:142d84459fbe755c7b9ff88404fafe9db9f7ac33da08fe27ec9a07b007e39492"
)


def _addrinfo(ip: str = "1.1.1.1"):
    def resolver(host: str, port: int, *args: object, **kwargs: object):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))]

    return resolver


class _CountingHandler:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            content=_FETCH_BODY,
            request=request,
        )


def _fetch(handler) -> SafePublicFetch:
    return SafePublicFetch(
        limits=SafePublicFetchLimits(max_bytes=4096, timeout=2.0),
        transport=httpx.MockTransport(handler),
        getaddrinfo=_addrinfo(),
    )


def _call(name: str, arguments: str, call_id: str = "call_1") -> ParsedToolCall:
    return ParsedToolCall(index=0, call_id=call_id, name=name, arguments=arguments)


def _document(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "kind": "document",
        "url": "https://cdn.example/page",
        "media_type": "text/plain",
        "content": "hosted fetch body",
        "digest": _FETCH_DIGEST,
        "retrieved_at": 1_757_000_000,
    }
    result.update(overrides)
    return result


def _search_result(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "kind": "search_results",
        "query": "loctree",
        "results": [
            {
                "title": "Loctree",
                "url": "https://loctree.dev",
                "snippet": "structural sight",
            }
        ],
        "digest": _SEARCH_DIGEST,
    }
    result.update(overrides)
    return result


def _entry_with(**overrides: object) -> dict[str, object]:
    entry: dict[str, object] = {
        "title": "Loctree",
        "url": "https://loctree.dev",
        "snippet": "structural sight",
    }
    entry.update(overrides)
    return entry


def _entry_without(key: str) -> dict[str, object]:
    entry = _entry_with()
    del entry[key]
    return entry


def test_result_budget_code_is_registered() -> None:
    assert "result_budget_exceeded" in HOSTED_ERROR_CODES


@pytest.mark.asyncio
async def test_fetch_result_is_canonical_golden_and_fetched_exactly_once() -> None:
    handler = _CountingHandler()
    catalog = HostedToolCatalog((HostedWebFetchTool(fetch=_fetch(handler)),))
    executor = HostedToolExecutor(catalog)

    result = await executor.execute(
        _call("web_fetch", '{"url":"https://cdn.example/page"}')
    )

    assert result.ok
    assert handler.calls == 1
    # The model continuation stays byte-identical to the admitted format.
    assert result.output == _FETCH_OUTPUT
    assert result.metadata is not None
    payload = result.metadata["result"]
    assert payload["kind"] == "document"
    assert payload["url"] == "https://cdn.example/page"
    assert payload["media_type"] == "text/plain"
    assert payload["content"] == "hosted fetch body"
    assert payload["digest"] == _FETCH_DIGEST
    assert isinstance(payload["retrieved_at"], int)
    assert payload["retrieved_at"] > 0
    receipt = result.metadata["receipt"]
    assert receipt["final_url"] == payload["url"]
    assert receipt["mime"] == payload["media_type"]
    assert receipt["result_digest"] == payload["digest"]
    # Validation/projection over the produced payload touches no transport.
    validate_result_payload("web_fetch", payload)
    validate_sealed_action(
        "web_fetch", {"kind": "fetch", "url": payload["url"]}, result=payload
    )
    assert handler.calls == 1


@pytest.mark.asyncio
async def test_search_result_digest_is_golden_and_receipt_agrees() -> None:
    async def provider(query: str):
        return [
            {
                "title": "Loctree",
                "url": "https://loctree.dev",
                "snippet": "structural sight",
                "internal_debug": {"socket": "10.0.0.1:8100"},
            },
            {"title": "no-url entries prove no identity"},
        ]

    catalog = HostedToolCatalog((HostedWebSearchTool(provider=provider),))
    executor = HostedToolExecutor(catalog)
    result = await executor.execute(_call("web_search", '{"query":"loctree"}'))

    assert result.ok
    assert result.metadata is not None
    payload = result.metadata["result"]
    assert payload["kind"] == "search_results"
    assert payload["query"] == "loctree"
    assert payload["results"] == [
        {
            "title": "Loctree",
            "url": "https://loctree.dev",
            "snippet": "structural sight",
        }
    ]
    assert payload["digest"] == _SEARCH_DIGEST
    receipt = result.metadata["receipt"]
    assert receipt["result_digest"] == _SEARCH_DIGEST
    assert receipt["result_count"] == 1
    # The continuation payload and the canonical result agree mechanically.
    assert json.loads(result.output) == {
        "query": payload["query"],
        "results": payload["results"],
    }


@pytest.mark.parametrize(
    ("tool_name", "mutation"),
    [
        ("web_fetch", _document(extra="field")),
        ("web_fetch", {k: v for k, v in _document().items() if k != "retrieved_at"}),
        ("web_fetch", _document(url="")),
        ("web_fetch", _document(url="   ")),
        ("web_fetch", _document(media_type=7)),
        ("web_fetch", _document(content=b"bytes")),
        ("web_fetch", _document(retrieved_at=True)),
        ("web_fetch", _document(retrieved_at=-1)),
        ("web_fetch", _document(digest="md5:abc")),
        ("web_fetch", _document(digest="sha256:" + "A" * 64)),
        ("web_fetch", _document(digest="sha256:" + "a" * 63)),
        ("web_fetch", _search_result()),
        ("web_search", _document()),
        ("web_search", _search_result(extra="field")),
        ("web_search", _search_result(query="")),
        ("web_search", _search_result(results="not-a-sequence")),
        ("web_search", _search_result(results=[{"title": "no identity"}])),
        ("web_search", _search_result(results=[{"url": ""}])),
        (
            "web_search",
            _search_result(
                results=[
                    {
                        "url": "https://loctree.dev",
                        "internal_debug": "secret",
                    }
                ]
            ),
        ),
        (
            "web_search",
            _search_result(results=[{"url": "https://loctree.dev", "snippet": 3}]),
        ),
        # The admitted entry shape is exactly {title, url, snippet}: each
        # missing key, extra key, empty string, and wrong type fails closed.
        (
            "web_search",
            _search_result(results=[_entry_without("title")]),
        ),
        (
            "web_search",
            _search_result(results=[_entry_without("url")]),
        ),
        (
            "web_search",
            _search_result(results=[_entry_without("snippet")]),
        ),
        (
            "web_search",
            _search_result(results=[_entry_with(rank=1)]),
        ),
        (
            "web_search",
            _search_result(results=[_entry_with(title="")]),
        ),
        (
            "web_search",
            _search_result(results=[_entry_with(url="   ")]),
        ),
        (
            "web_search",
            _search_result(results=[_entry_with(snippet="")]),
        ),
        (
            "web_search",
            _search_result(results=[_entry_with(title=7)]),
        ),
        (
            "web_search",
            _search_result(results=[_entry_with(url=True)]),
        ),
        (
            "web_search",
            _search_result(results=[_entry_with(snippet=["s"])]),
        ),
        ("reasoner", _document()),
        ("web_fetch", "not-a-mapping"),
    ],
)
def test_result_payload_validator_fails_closed(tool_name: str, mutation) -> None:
    with pytest.raises(ValueError):
        validate_result_payload(tool_name, mutation)


def test_result_payload_validator_accepts_golden_payloads() -> None:
    assert validate_result_payload("web_fetch", _document()) == _document()
    assert validate_result_payload("web_search", _search_result()) == _search_result()


def test_oversize_content_is_a_budget_error_not_a_partial_success() -> None:
    oversized = _document(content="x" * (MAX_RESULT_TEXT_CHARS + 1))
    with pytest.raises(ResultBudgetExceeded):
        validate_result_payload("web_fetch", oversized)


@pytest.mark.asyncio
async def test_executor_types_budget_overflow_without_leaking_content() -> None:
    class _OversizedTool:
        name = "web_fetch"

        def describe(self):
            return {"name": self.name}

        async def invoke(self, arguments):
            return HostedToolSuccess(
                payload={"ok": True},
                result=_document(content="x" * (MAX_RESULT_TEXT_CHARS + 1)),
            )

    executor = HostedToolExecutor(HostedToolCatalog((_OversizedTool(),)))
    result = await executor.execute(_call("web_fetch", "{}"))

    assert not result.ok
    assert result.metadata is not None
    assert result.metadata["error_code"] == "result_budget_exceeded"
    assert "result" not in result.metadata
    assert "xxxx" not in result.output


@pytest.mark.asyncio
async def test_receipt_disagreement_fails_closed() -> None:
    class _LyingTool:
        name = "web_fetch"

        def describe(self):
            return {"name": self.name}

        async def invoke(self, arguments):
            return HostedToolSuccess(
                payload={"ok": True},
                receipt_fields={
                    "final_url": "https://cdn.example/page",
                    "mime": "text/plain",
                    "result_digest": "sha256:" + "0" * 64,
                },
                result=_document(),
            )

    executor = HostedToolExecutor(HostedToolCatalog((_LyingTool(),)))
    result = await executor.execute(_call("web_fetch", "{}"))

    assert not result.ok
    assert result.metadata is not None
    assert result.metadata["error_code"] == "invalid_tool_result"
    assert "result" not in result.metadata


@pytest.mark.asyncio
async def test_failure_paths_carry_no_result_payload() -> None:
    class _CrashTool:
        name = "crash"

        def describe(self):
            return {"name": "crash"}

        async def invoke(self, arguments):
            raise RuntimeError("backend exploded")

    class _SlowTool:
        name = "slow"

        def describe(self):
            return {"name": "slow"}

        async def invoke(self, arguments):
            await asyncio.sleep(30.0)

    catalog = HostedToolCatalog(
        (_CrashTool(), _SlowTool(), HostedWebSearchTool(provider=None))
    )
    executor = HostedToolExecutor(catalog, per_call_timeout_s=0.05)

    for name, arguments in (
        ("crash", "{}"),
        ("slow", "{}"),
        ("web_search", '{"query":"q"}'),
    ):
        result = await executor.execute(_call(name, arguments))
        assert not result.ok
        assert result.metadata is not None
        assert "result" not in result.metadata
        assert result.metadata["receipt"]["status"] == "failed"


@pytest.mark.asyncio
async def test_cancelled_scope_produces_no_result() -> None:
    class _Cancel:
        cancelled = True
        reason = "client disconnected"

    handler = _CountingHandler()
    catalog = HostedToolCatalog((HostedWebFetchTool(fetch=_fetch(handler)),))
    executor = HostedToolExecutor(catalog)
    token = set_execution_scope(HostedExecutionScope(cancel=_Cancel()))
    try:
        with pytest.raises(asyncio.CancelledError):
            await executor.execute(
                _call("web_fetch", '{"url":"https://cdn.example/page"}')
            )
    finally:
        reset_execution_scope(token)
    assert handler.calls == 0


@pytest.mark.asyncio
async def test_exhausted_deadline_is_typed_and_carries_no_result() -> None:
    handler = _CountingHandler()
    catalog = HostedToolCatalog((HostedWebFetchTool(fetch=_fetch(handler)),))
    executor = HostedToolExecutor(catalog)
    loop = asyncio.get_running_loop()
    token = set_execution_scope(HostedExecutionScope(deadline=loop.time() - 1.0))
    try:
        result = await executor.execute(
            _call("web_fetch", '{"url":"https://cdn.example/page"}')
        )
    finally:
        reset_execution_scope(token)

    assert not result.ok
    assert result.metadata is not None
    assert result.metadata["error_code"] == "fetch_url_fetch_timeout"
    assert "result" not in result.metadata
    assert handler.calls == 0


@pytest.mark.asyncio
async def test_redirect_keeps_requested_action_and_final_provenance() -> None:
    """HRPD2 §2.1-2.2 falsifier: one fetch across a redirect keeps two truths.

    Goes red if the sealed action URL is compared with (or rewritten to) the
    final result URL, if result/receipt final URLs diverge, or if the
    transport runs a second time (the hop list is asserted exactly).
    """

    class _RedirectHandler:
        def __init__(self) -> None:
            self.paths: list[str] = []

        def __call__(self, request: httpx.Request) -> httpx.Response:
            self.paths.append(request.url.path)
            if request.url.path == "/start":
                return httpx.Response(
                    302,
                    headers={"location": "https://example.com/final"},
                    request=request,
                )
            return httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                content=_FETCH_BODY,
                request=request,
            )

    handler = _RedirectHandler()
    catalog = HostedToolCatalog((HostedWebFetchTool(fetch=_fetch(handler)),))
    executor = HostedToolExecutor(catalog)

    result = await executor.execute(
        _call("web_fetch", '{"url":"https://example.com/start"}')
    )

    assert result.ok
    # Exactly one transport pass: the redirect hop plus the final document.
    assert handler.paths == ["/start", "/final"]
    assert result.metadata is not None
    payload = result.metadata["result"]
    receipt = result.metadata["receipt"]
    # Result and receipt both carry the redirect-resolved final URL...
    assert payload["url"] == "https://example.com/final"
    assert receipt["final_url"] == "https://example.com/final"
    assert receipt["result_digest"] == payload["digest"] == _FETCH_DIGEST
    # ...while the sealed action retains the model-requested URL unchanged.
    assert validate_sealed_action(
        "web_fetch",
        {"kind": "fetch", "url": "https://example.com/start"},
        result=payload,
    ) == {"kind": "fetch", "url": "https://example.com/start"}
    assert handler.paths == ["/start", "/final"]


def test_sealed_action_validator_is_closed_and_total() -> None:
    document = _document()
    search = _search_result()

    assert validate_sealed_action(
        "web_fetch", {"kind": "fetch", "url": document["url"]}, result=document
    ) == {"kind": "fetch", "url": "https://cdn.example/page"}
    # HRPD2 §2.1: the sealed fetch action carries the model-REQUESTED URL and
    # is returned unchanged even when redirects made the result's final URL
    # differ; final-URL agreement is solely _verify_result_receipt_agreement.
    assert validate_sealed_action(
        "web_fetch",
        {"kind": "fetch", "url": "https://example.com/start"},
        result=document,
    ) == {"kind": "fetch", "url": "https://example.com/start"}
    assert validate_sealed_action(
        "web_search",
        {"kind": "search", "query": "loctree", "sources": ["https://loctree.dev"]},
        result=search,
    ) == {"kind": "search", "query": "loctree", "sources": ["https://loctree.dev"]}

    for tool_name, action, result in (
        ("web_fetch", {"kind": "search", "query": "q", "sources": []}, None),
        ("web_fetch", {"kind": "fetch", "url": "https://a", "extra": 1}, None),
        ("web_fetch", {"kind": "fetch", "url": ""}, None),
        # A supplied fetch result is still fully validated closed even though
        # its URL is never compared against the requested action URL.
        ("web_fetch", {"kind": "fetch", "url": "https://a"}, _document(extra=1)),
        ("web_fetch", {"kind": "fetch", "url": "https://a"}, _search_result()),
        ("web_search", {"kind": "fetch", "url": "https://a"}, None),
        ("web_search", {"kind": "search", "query": ""}, None),
        (
            "web_search",
            {"kind": "search", "query": "q", "sources": ["https://a", "https://a"]},
            None,
        ),
        (
            "web_search",
            {"kind": "search", "query": "q", "sources": ["https://unproven.example"]},
            search,
        ),
        ("reasoner", {"kind": "fetch", "url": "https://a"}, None),
        ("web_fetch", "not-a-mapping", None),
    ):
        with pytest.raises(ValueError):
            validate_sealed_action(tool_name, action, result=result)


@pytest.mark.asyncio
async def test_hosted_error_taxonomy_stays_closed_for_the_budget_code() -> None:
    error = HostedToolError("result_budget_exceeded", "over budget")
    assert error.code == "result_budget_exceeded"
    with pytest.raises(ValueError):
        HostedToolError("result_budget", "not a registered code")
