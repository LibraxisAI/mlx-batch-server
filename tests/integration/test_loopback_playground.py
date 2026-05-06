"""Loopback playground proxy works across operator + inference boundary.

Mocks the inference HTTP layer so we can exercise the SSE proxy without a
real model in the loop. Validates that the operator forwards prompts and that
SSE chunks make it back through the playground endpoint.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx


class _FakeStreamResponse:
    headers = {"content-type": "text/event-stream"}

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self) -> AsyncIterator[str]:
        yield 'data: {"response_id":"resp_test","model":"mlx-test"}'
        yield 'data: {"event":"text.delta","text":"hello"}'
        yield "data: [DONE]"


class _FakeStreamContext:
    async def __aenter__(self):
        return _FakeStreamResponse()

    async def __aexit__(self, exc_type, exc, tb):
        return None


class _FakeAsyncClient:
    """Minimal httpx.AsyncClient stand-in used as drop-in monkeypatch.

    Implements the surface area both the operator lifespan probe (``.get``)
    and the playground SSE proxy (``.stream``) need. ``.get`` raises an
    ``httpx.ConnectError`` so the lifespan probe records the inference server
    as unavailable without crashing the test app.
    """

    def __init__(self, *args, **kwargs):
        self.calls: list[tuple[str, str, dict | None, dict | None]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return None

    async def get(self, url, **kwargs):
        raise httpx.ConnectError(f"fake operator client refuses to connect to {url}")

    def stream(self, method, url, headers=None, json=None):
        self.calls.append((method, url, headers, json))
        return _FakeStreamContext()


def test_playground_proxies_to_inference_in_open_mode(monkeypatch):
    """Open mode: playground forwards to /v1/responses on inference port."""
    from fastapi.testclient import TestClient

    from mlx_batch_server.core.config import get_settings as core_settings_cache
    from mlx_batch_server.operator.config import (
        get_settings as op_settings_cache,
    )
    from mlx_batch_server.operator.main import create_app
    from mlx_batch_server.operator.routers import playground

    monkeypatch.delenv("SECURITY_LEVEL", raising=False)
    core_settings_cache.cache_clear()
    op_settings_cache.cache_clear()
    monkeypatch.setattr(playground.httpx, "AsyncClient", _FakeAsyncClient)

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/playground/responses",
            json={
                "model": "mlx-test",
                "input": [{"role": "user", "content": "hello"}],
                "stream": True,
                "session_id": "default",
            },
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "resp_test" in response.text
        assert "hello" in response.text


def test_playground_requires_auth_when_gated(monkeypatch):
    """Auth-gated mode: same proxy requires the key on operator boundary."""
    from fastapi.testclient import TestClient

    from mlx_batch_server.core.config import get_settings as core_settings_cache
    from mlx_batch_server.operator.config import (
        get_settings as op_settings_cache,
    )
    from mlx_batch_server.operator.main import create_app
    from mlx_batch_server.operator.routers import playground

    monkeypatch.setenv("SECURITY_LEVEL", "2")
    monkeypatch.setenv("API_KEY", "loopback-key")
    core_settings_cache.cache_clear()
    op_settings_cache.cache_clear()
    monkeypatch.setattr(playground.httpx, "AsyncClient", _FakeAsyncClient)

    with TestClient(create_app()) as client:
        unauthorized = client.post(
            "/api/playground/responses",
            json={
                "model": "mlx-test",
                "input": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
        assert unauthorized.status_code == 401

        ok = client.post(
            "/api/playground/responses",
            headers={"x-api-key": "loopback-key"},
            json={
                "model": "mlx-test",
                "input": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )
        assert ok.status_code == 200
        assert "resp_test" in ok.text


def test_playground_forwards_internal_api_key(monkeypatch):
    """Operator should forward MLX_BATCH_INTERNAL_API_KEY to inference."""
    from fastapi.testclient import TestClient

    from mlx_batch_server.core.config import get_settings as core_settings_cache
    from mlx_batch_server.operator.config import (
        get_settings as op_settings_cache,
    )
    from mlx_batch_server.operator.main import create_app
    from mlx_batch_server.operator.routers import playground

    monkeypatch.delenv("SECURITY_LEVEL", raising=False)
    monkeypatch.setenv("MLX_BATCH_INTERNAL_API_KEY", "internal-secret")
    core_settings_cache.cache_clear()
    op_settings_cache.cache_clear()

    captured: dict = {}

    class _CapturingClient(_FakeAsyncClient):
        def stream(self, method, url, headers=None, json=None):
            captured["headers"] = headers
            captured["url"] = url
            return _FakeStreamContext()

    monkeypatch.setattr(playground.httpx, "AsyncClient", _CapturingClient)
    # The operator lifespan probe also touches httpx.AsyncClient — patch the
    # probe module too so we don't reach real network during startup.
    from mlx_batch_server.operator.services import inference_probe

    monkeypatch.setattr(inference_probe.httpx, "AsyncClient", _CapturingClient)

    with TestClient(create_app()) as client:
        response = client.post(
            "/api/playground/responses",
            json={
                "model": "mlx-test",
                "input": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert response.status_code == 200

    assert captured["headers"]["x-api-key"] == "internal-secret"
    assert captured["url"].endswith("/v1/responses")
