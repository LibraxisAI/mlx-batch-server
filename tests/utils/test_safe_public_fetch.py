"""Focused SafePublicFetch tests for SSRF, redirects, and DNS rebinding.

These tests do not open a real network socket. DNS and HTTP are injected so
connect-time IP pinning and fail-closed classification can be proven.
"""

from __future__ import annotations

import gzip
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
        stream=httpx.ByteStream(_PNG),
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
    def attested_handler(request: httpx.Request) -> httpx.Response:
        response = handler(request)
        response.extensions["mlx_batch_server.connected_peer"] = request.url.host
        return response

    inner = httpx.MockTransport(attested_handler)
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
async def test_mock_transport_success_body_is_consumed_through_raw_stream() -> None:
    responses: list[httpx.Response] = []

    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            200,
            headers={"content-type": "image/png"},
            stream=httpx.ByteStream(_PNG),
            request=request,
        )
        assert response.is_stream_consumed is False
        responses.append(response)
        return response

    fetch, _ = _fetcher(handler)
    resource = await fetch.fetch(
        "https://cdn.example/raw.png",
        accepted_media_types=("image/png",),
    )

    assert resource.content == _PNG
    assert responses[0].is_stream_consumed is True


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
    assert resource.transport_receipt is not None
    assert resource.transport_receipt.requested_url == "https://cdn.example/pixel.png"
    assert resource.transport_receipt.hops[0].connected_peer == _PUBLIC_IP


@pytest.mark.asyncio
async def test_connected_peer_drift_fails_before_body_acceptance() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "image/png"},
            stream=httpx.ByteStream(_PNG),
            request=request,
            extensions={"mlx_batch_server.connected_peer": "8.8.8.8"},
        )

    fetch = SafePublicFetch(
        transport=httpx.MockTransport(handler),
        getaddrinfo=_addrinfo(_PUBLIC_IP),
    )
    with pytest.raises(SafePublicFetchError, match="connected peer") as caught:
        await fetch.fetch("https://cdn.example/x", accepted_media_types=("image/png",))
    assert caught.value.code == "connected_peer_mismatch"


@pytest.mark.asyncio
async def test_compressed_bomb_is_rejected_by_decoded_budget() -> None:
    bomb = gzip.compress(b"x" * 4096)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain", "content-encoding": "gzip"},
            stream=httpx.ByteStream(bomb),
            request=request,
        )

    fetch, _ = _fetcher(handler, max_bytes=1024)
    with pytest.raises(SafePublicFetchError) as caught:
        await fetch.fetch(
            "https://cdn.example/bomb", accepted_media_types=("text/plain",)
        )
    assert caught.value.code == "decoded_bytes_exceeded"


@pytest.mark.asyncio
async def test_http_429_survives_as_typed_rate_limit_without_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            stream=httpx.ByteStream(b"provider secret"),
            request=request,
        )

    fetch, _ = _fetcher(handler)
    with pytest.raises(SafePublicFetchError) as caught:
        await fetch.fetch(
            "https://cdn.example/limited", accepted_media_types=("text/plain",)
        )
    assert caught.value.code == "rate_limited"
    assert caught.value.http_status == 429
    assert "secret" not in str(caught.value)


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
async def test_redirect_receipt_preserves_ordered_dns_and_peer_attestation() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["host"] == "start.example":
            return httpx.Response(
                302,
                headers={"location": "https://final.example/page"},
                request=request,
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            stream=httpx.ByteStream(b"final"),
            request=request,
        )

    fetch, _ = _fetcher(handler, getaddrinfo=_addrinfo("1.1.1.1", "8.8.8.8"))
    resource = await fetch.fetch(
        "https://start.example/root", accepted_media_types=("text/plain",)
    )

    receipt = resource.transport_receipt
    assert receipt is not None
    assert receipt.requested_url == "https://start.example/root"
    assert receipt.final_url == "https://final.example/page"
    assert [hop.requested_url for hop in receipt.hops] == [
        "https://start.example/root",
        "https://final.example/page",
    ]
    assert receipt.hops[0].dns_answers == ("1.1.1.1", "8.8.8.8")
    assert all(hop.connected_peer == hop.selected_address for hop in receipt.hops)


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


class _Cancelled:
    """Minimal FetchCancelCheck flipping to cancelled after N observations."""

    def __init__(self, *, after: int = 0) -> None:
        self._after = after
        self.checks = 0

    @property
    def cancelled(self) -> bool:
        self.checks += 1
        return self.checks > self._after

    @property
    def reason(self) -> str | None:
        return "client_cancelled"


@pytest.mark.asyncio
async def test_cancel_check_stops_the_fetch_before_the_first_hop() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _png_handler(request)

    fetch, _ = _fetcher(handler)
    import asyncio

    with pytest.raises(asyncio.CancelledError):
        await fetch.fetch(
            "https://cdn.example/pixel.png",
            accepted_media_types=("image/png",),
            cancel=_Cancelled(after=0),
        )
    assert calls == []


@pytest.mark.asyncio
async def test_cancel_check_stops_body_consumption_mid_stream() -> None:
    fetch, _ = _fetcher(_png_handler)
    import asyncio

    with pytest.raises(asyncio.CancelledError):
        await fetch.fetch(
            "https://cdn.example/pixel.png",
            accepted_media_types=("image/png",),
            cancel=_Cancelled(after=1),
        )


@pytest.mark.asyncio
async def test_non_positive_deadline_fails_closed_as_timeout() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _png_handler(request)

    fetch, _ = _fetcher(handler)
    with pytest.raises(SafePublicFetchError) as error:
        await fetch.fetch(
            "https://cdn.example/pixel.png",
            accepted_media_types=("image/png",),
            deadline_s=0.0,
        )
    assert error.value.code == "url_fetch_timeout"
    assert calls == []


@pytest.mark.asyncio
async def test_deadline_bounds_a_stalled_transport() -> None:
    import asyncio

    class _StallingTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(30.0)
            raise AssertionError("unreachable")

    fetch = SafePublicFetch(
        limits=SafePublicFetchLimits(max_bytes=1024, timeout=60.0),
        transport=_StallingTransport(),
        getaddrinfo=_addrinfo(_PUBLIC_IP),
    )
    with pytest.raises(SafePublicFetchError) as error:
        await fetch.fetch(
            "https://cdn.example/pixel.png",
            accepted_media_types=("image/png",),
            deadline_s=0.05,
        )
    assert error.value.code == "url_fetch_timeout"
