"""Security level dial: 0 / 2 / 3 must gate consistently."""

from __future__ import annotations

import asyncio

import pytest
from starlette.requests import Request

from mlx_batch_server.auth.dependency import (
    _normalized_security_level,
    is_auth_required,
    verify_auth,
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


def test_level_2_without_credentials_allows_when_unconfigured(monkeypatch):
    """Level 2 with no API_KEY and no session enabled = anonymous (back-compat)."""
    monkeypatch.setenv("SECURITY_LEVEL", "2")
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("SESSION_AUTH_ENABLED", raising=False)
    get_settings.cache_clear()
    info = asyncio.run(verify_auth(_scope(), api_key=None, bearer_creds=None))
    assert info["auth_method"] == "none"


def test_level_2_with_static_api_key_requires_match(monkeypatch):
    monkeypatch.setenv("SECURITY_LEVEL", "2")
    monkeypatch.setenv("API_KEY", "secret-test-key")
    get_settings.cache_clear()

    info = asyncio.run(
        verify_auth(_scope(), api_key="secret-test-key", bearer_creds=None)
    )
    assert info["auth_method"] == "api_key_static"

    with pytest.raises(Exception):
        asyncio.run(verify_auth(_scope(), api_key="wrong", bearer_creds=None))


def test_level_3_demands_session(monkeypatch):
    monkeypatch.setenv("SECURITY_LEVEL", "3")
    get_settings.cache_clear()
    with pytest.raises(Exception):
        asyncio.run(verify_auth(_scope(), api_key=None, bearer_creds=None))
