"""Focused SafePublicFetch tests for SSRF, redirects, and DNS rebinding.

These tests do not open a real network socket. DNS and HTTP are injected so
connect-time IP pinning and fail-closed classification can be proven.
"""

from __future__ import annotations

import socket

import httpx
import pytest

from mlx_batch_server.utils.safe_public_fetch import (
    SafePublicFetch,
    SafePublicFetchError,
    SafePublicFetchLimits,
)

_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00"
    b"\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N\x00"
    b"\x00\x00\x00IEND\xaeB`\x82"
)
_PUBLIC_IP = "1.1.1.1"


def _addrinfo(*ips: str):
    def resolver(host: str, port: int, *args: object, **kwargs: object):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0)) for ip in ips
        ]

    return resolver


class _RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        self._inner = inner
        self.requests: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return await self._inner.handle_async_request(request)


def _png_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "image/png"},
        content=_PNG,
        request=request,
    )


def _fetcher(
    handler,
    *,
    getaddrinfo=None,
    allowed_origins: frozenset[str] = frozenset(),
    record: bool = False,
    max_bytes: int = 1024,
):
    inner = httpx.MockTransport(handler)
    transport: httpx.AsyncBaseTransport = (
        _RecordingTransport(inner) if record else inner
    )
    fetch = SafePublicFetch(
        limits=SafePublicFetchLimits(max_bytes=max_bytes, timeout=2.0),
        allowed_origins=allowed_origins,
        transport=transport,
        getaddrinfo=getaddrinfo or _addrinfo(_PUBLIC_IP),
    )
    return fetch, transport


@pytest.mark.asyncio
async def test_public_https_url_succeeds_without_allowlist() -> None:
    fetch, _ = _fetcher(_png_handler)

    resource = await fetch.fetch(
        "https://cdn.example/pixel.png",
        accepted_media_types=("image/png",),
    )

    assert resource.content == _PNG
    assert resource.media_type == "image/png"
    assert resource.final_url == "https://cdn.example/pixel.png"


@pytest.mark.asyncio
async def test_connect_pins_validated_ip_and_keeps_logical_host() -> None:
    fetch, transport = _fetcher(_png_handler, record=True)
    assert isinstance(transport, _RecordingTransport)

    resource = await fetch.fetch(
        "https://cdn.example/pixel.png",
        accepted_media_types=("image/png",),
    )

    request = transport.requests[0]
    assert request.url.host == _PUBLIC_IP
    assert request.headers["host"] == "cdn.example"
    assert request.extensions.get("sni_hostname") == "cdn.example"
    assert resource.final_url == "https://cdn.example/pixel.png"


@pytest.mark.asyncio
async def test_credentials_and_invalid_schemes_fail_before_http() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _png_handler(request)

    fetch, _ = _fetcher(handler)
    cases = (
        ("https://user:pass@cdn.example/x.png", "url_credentials_forbidden"),
        ("https://user@cdn.example/x.png", "url_credentials_forbidden"),
        ("file:///etc/passwd", "invalid_url_scheme"),
        ("gopher://cdn.example/1", "invalid_url_scheme"),
        ("javascript:alert(1)", "invalid_url_scheme"),
    )
    for url, code in cases:
        with pytest.raises(SafePublicFetchError) as error:
            await fetch.fetch(url, accepted_media_types=("image/png",))
        assert error.value.code == code
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    (
        "http://localhost/secret",
        "http://127.0.0.1/secret",
        "http://10.0.0.8/secret",
        "http://192.168.1.4/secret",
        "http://169.254.169.254/latest/meta-data",
        "http://100.64.1.1/tailscale",
        "http://224.0.0.1/multicast",
        "http://[::1]/loopback",
        "http://[fd7a:115c:a1e0::1]/tailscale",
    ),
)
async def test_blocked_targets_fail_before_body_consumption(url: str) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _png_handler(request)

    fetch, _ = _fetcher(handler)
    with pytest.raises(SafePublicFetchError) as error:
        await fetch.fetch(url, accepted_media_types=("image/png",))
    assert error.value.code in {"url_target_blocked", "invalid_url"}
    assert "127." not in str(error.value)
    assert "10.0." not in str(error.value)
    assert "100.64." not in str(error.value)
    assert calls == []


@pytest.mark.asyncio
async def test_redirect_to_private_fails_closed_before_private_body() -> None:
    consumed_private = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal consumed_private
        if request.url.host in {"127.0.0.1", "localhost"}:
            consumed_private = True
            return _png_handler(request)
        return httpx.Response(
            302,
            headers={"location": "http://127.0.0.1/secret.png"},
            request=request,
        )

    fetch, _ = _fetcher(handler)
    with pytest.raises(SafePublicFetchError) as error:
        await fetch.fetch(
            "https://cdn.example/start",
            accepted_media_types=("image/png",),
        )

    assert error.value.code == "url_target_blocked"
    assert consumed_private is False
    assert "127.0.0.1" not in str(error.value)


@pytest.mark.asyncio
async def test_mixed_dns_answers_fail_closed_before_http() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _png_handler(request)

    fetch, _ = _fetcher(
        handler,
        getaddrinfo=_addrinfo("1.1.1.1", "127.0.0.1"),
    )
    with pytest.raises(SafePublicFetchError) as error:
        await fetch.fetch(
            "https://rebind.example/pixel.png",
            accepted_media_types=("image/png",),
        )

    assert error.value.code == "url_target_blocked"
    assert calls == []
    assert "127.0.0.1" not in str(error.value)


@pytest.mark.asyncio
async def test_optional_origin_lockdown_still_blocks_foreign_hosts() -> None:
    fetch, _ = _fetcher(
        _png_handler,
        allowed_origins=frozenset({"https://media.example"}),
    )
    with pytest.raises(SafePublicFetchError) as error:
        await fetch.fetch(
            "https://cdn.example/pixel.png",
            accepted_media_types=("image/png",),
        )
    assert error.value.code == "url_not_allowed"


@pytest.mark.asyncio
async def test_byte_and_mime_limits_are_enforced() -> None:
    fetch, _ = _fetcher(_png_handler, max_bytes=4)
    with pytest.raises(SafePublicFetchError) as too_large:
        await fetch.fetch(
            "https://cdn.example/pixel.png",
            accepted_media_types=("image/png",),
            max_bytes=4,
        )
    assert too_large.value.code == "source_bytes_exceeded"

    fetch, _ = _fetcher(_png_handler)
    with pytest.raises(SafePublicFetchError) as mime:
        await fetch.fetch(
            "https://cdn.example/pixel.png",
            accepted_media_types=("image/jpeg",),
        )
    assert mime.value.code == "unsupported_media_type"
