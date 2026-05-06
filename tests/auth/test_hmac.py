"""HMAC client lifecycle + signature verification (file backend)."""

from __future__ import annotations

import asyncio
import hashlib
import hmac as _hmac
import time

import pytest
from starlette.requests import Request

from mlx_batch_server.auth import hmac as hmac_mod


def _scope(headers: dict[str, str], body: bytes = b"") -> Request:
    request = Request(
        scope={
            "type": "http",
            "method": "GET",
            "path": "/some/path",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "query_string": b"",
            "client": ("127.0.0.1", 5555),
        }
    )

    async def _receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request._receive = _receive  # type: ignore[attr-defined]
    return request


def _sign(secret: str, ts: int, method: str, path: str, body: bytes = b"") -> str:
    body_hash = hashlib.sha256(body).hexdigest() if body else ""
    msg = f"{ts}:{method.upper()}:{path}:{body_hash}"
    return _hmac.new(secret.encode(), msg.encode(), hashlib.sha256).hexdigest()


def test_register_and_revoke_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_BATCH_HMAC_SECRETS_FILE", str(tmp_path / "h.json"))
    secret = asyncio.run(hmac_mod.register_hmac_client("device-1"))
    assert len(secret) == 64

    clients = asyncio.run(hmac_mod.list_hmac_clients())
    assert "device-1" in clients

    assert asyncio.run(hmac_mod.revoke_hmac_client("device-1")) is True
    assert "device-1" not in asyncio.run(hmac_mod.list_hmac_clients())


def test_verify_request_accepts_valid_signature(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_BATCH_HMAC_SECRETS_FILE", str(tmp_path / "h.json"))
    secret = asyncio.run(hmac_mod.register_hmac_client("device-2"))
    ts = int(time.time())
    sig = _sign(secret, ts, "GET", "/some/path")

    info = asyncio.run(
        hmac_mod.verify_hmac_request(
            _scope(
                {
                    "X-Client-ID": "device-2",
                    "X-Timestamp": str(ts),
                    "X-Signature": sig,
                }
            )
        )
    )
    assert info["auth_method"] == "hmac"
    assert info["client_id"] == "device-2"


def test_verify_request_rejects_stale_timestamp(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_BATCH_HMAC_SECRETS_FILE", str(tmp_path / "h.json"))
    monkeypatch.setenv("HMAC_TIMESTAMP_TOLERANCE", "60")
    from mlx_batch_server.core.config import get_settings

    get_settings.cache_clear()
    secret = asyncio.run(hmac_mod.register_hmac_client("device-3"))
    ts = int(time.time()) - 600  # 10 minutes old
    sig = _sign(secret, ts, "GET", "/some/path")

    with pytest.raises(Exception):
        asyncio.run(
            hmac_mod.verify_hmac_request(
                _scope(
                    {
                        "X-Client-ID": "device-3",
                        "X-Timestamp": str(ts),
                        "X-Signature": sig,
                    }
                )
            )
        )


def test_verify_request_rejects_bad_signature(tmp_path, monkeypatch):
    monkeypatch.setenv("MLX_BATCH_HMAC_SECRETS_FILE", str(tmp_path / "h.json"))
    asyncio.run(hmac_mod.register_hmac_client("device-4"))
    ts = int(time.time())
    bad_sig = "0" * 64

    with pytest.raises(Exception):
        asyncio.run(
            hmac_mod.verify_hmac_request(
                _scope(
                    {
                        "X-Client-ID": "device-4",
                        "X-Timestamp": str(ts),
                        "X-Signature": bad_sig,
                    }
                )
            )
        )
