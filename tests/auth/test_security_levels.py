"""Security level dial: 0 / 2 / 3 must gate consistently."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.security import HTTPAuthorizationCredentials
from starlette.requests import Request
from starlette.websockets import WebSocket

import mlx_batch_server.auth.dependency as auth_dependency
from mlx_batch_server.auth.dependency import (
    _normalized_security_level,
    is_auth_required,
    verify_auth,
    verify_websocket_auth,
)
from mlx_batch_server.auth.response_owner import (
    build_api_key_response_owner,
    build_hmac_response_owner,
    build_open_response_owner,
    build_session_response_owner,
)
from mlx_batch_server.core.config import get_settings


def _scope(headers: dict[str, str] | None = None) -> Request:
    return Request(
        scope={
            "type": "http",
            "method": "GET",
            "path": "/v1/test",
            "headers": [
                (k.lower().encode(), v.encode()) for k, v in (headers or {}).items()
            ],
            "query_string": b"",
            "client": ("127.0.0.1", 12345),
        }
    )


def _websocket(headers: dict[str, str] | None = None) -> WebSocket:
    async def receive():
        return {"type": "websocket.disconnect"}

    async def send(_message):
        return None

    return WebSocket(
        scope={
            "type": "websocket",
            "scheme": "ws",
            "path": "/v1/responses",
            "headers": [
                (key.lower().encode(), value.encode())
                for key, value in (headers or {}).items()
            ],
            "query_string": b"",
            "client": ("127.0.0.1", 12345),
            "server": ("127.0.0.1", 10240),
        },
        receive=receive,
        send=send,
    )


def test_level_1_is_promoted_to_2():
    assert _normalized_security_level(1) == 2


def test_level_0_is_default(monkeypatch):
    monkeypatch.delenv("SECURITY_LEVEL", raising=False)
    get_settings.cache_clear()
    assert get_settings().security_level == 0
    assert is_auth_required() is False


def test_level_0_bypasses_with_pseudo_owner(monkeypatch):
    monkeypatch.setenv("SECURITY_LEVEL", "0")
    get_settings.cache_clear()
    info = asyncio.run(verify_auth(_scope(), api_key=None, bearer_creds=None))
    assert info["auth_method"] == "bypass"
    assert info["user_id"].startswith("bypass:")
    assert info["resolved_api_key"].startswith("open-")
    assert info["response_owner_id"] == build_open_response_owner("127.0.0.1")


def test_level_2_without_credentials_allows_when_unconfigured(monkeypatch):
    """Level 2 with no API_KEY and no session enabled = anonymous (back-compat)."""
    monkeypatch.setenv("SECURITY_LEVEL", "2")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("SESSION_AUTH_ENABLED", raising=False)
    get_settings.cache_clear()
    info = asyncio.run(verify_auth(_scope(), api_key=None, bearer_creds=None))
    assert info["auth_method"] == "none"
    assert info["response_owner_id"] == build_open_response_owner("127.0.0.1")


def test_level_2_with_static_api_key_requires_match(monkeypatch):
    monkeypatch.setenv("SECURITY_LEVEL", "2")
    monkeypatch.setenv("API_KEY", "secret-test-key")
    get_settings.cache_clear()

    info = asyncio.run(
        verify_auth(_scope(), api_key="secret-test-key", bearer_creds=None)
    )
    assert info["auth_method"] == "api_key_static"
    assert info["response_owner_id"] == build_api_key_response_owner("secret-test-key")
    assert "secret-test-key" not in info["response_owner_id"]

    with pytest.raises(Exception):
        asyncio.run(verify_auth(_scope(), api_key="wrong", bearer_creds=None))


def test_level_3_demands_session(monkeypatch):
    monkeypatch.setenv("SECURITY_LEVEL", "3")
    get_settings.cache_clear()
    with pytest.raises(Exception):
        asyncio.run(verify_auth(_scope(), api_key=None, bearer_creds=None))


def test_verified_session_owner_uses_session_id_not_claimed_user(monkeypatch):
    async def resolve_session(session_id: str):
        assert session_id == "verified-session-token"
        return {
            "user_id": "client-controlled-label",
            "session_id": session_id,
        }

    monkeypatch.setenv("SECURITY_LEVEL", "3")
    monkeypatch.setenv("SESSION_AUTH_ENABLED", "1")
    get_settings.cache_clear()
    monkeypatch.setattr(auth_dependency, "_resolve_session_auth", resolve_session)

    info = asyncio.run(
        verify_auth(
            _scope(),
            api_key=None,
            bearer_creds=HTTPAuthorizationCredentials(
                scheme="Bearer",
                credentials="verified-session-token",
            ),
        )
    )

    assert info["response_owner_id"] == build_session_response_owner(
        "verified-session-token"
    )


def test_verified_hmac_owner_uses_client_id(monkeypatch):
    async def verify_hmac(_request: Request):
        return {
            "client_id": "stable-client",
            "auth_method": "hmac",
            "user_id": "hmac:stable-client",
            "session_id": None,
        }

    monkeypatch.setenv("SECURITY_LEVEL", "2")
    monkeypatch.delenv("SESSION_AUTH_ENABLED", raising=False)
    get_settings.cache_clear()
    monkeypatch.setattr(auth_dependency, "verify_hmac_request", verify_hmac)
    request = _scope(
        {
            "X-Client-ID": "stable-client",
            "X-Timestamp": "1",
            "X-Signature": "verified-elsewhere",
        }
    )

    info = asyncio.run(verify_auth(request, api_key=None, bearer_creds=None))

    assert info["response_owner_id"] == build_hmac_response_owner("stable-client")


def test_websocket_auth_uses_the_same_verified_api_key_owner(monkeypatch):
    monkeypatch.setenv("SECURITY_LEVEL", "2")
    monkeypatch.setenv("API_KEY", "websocket-secret")
    get_settings.cache_clear()
    websocket = _websocket({"Authorization": "Bearer websocket-secret"})

    info = asyncio.run(verify_websocket_auth(websocket))

    assert info["auth_method"] == "api_key_static"
    assert info["response_owner_id"] == build_api_key_response_owner("websocket-secret")
