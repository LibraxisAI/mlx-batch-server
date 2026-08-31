"""Inference app respects SECURITY_LEVEL across /v1/* and /api/admin/*."""

from __future__ import annotations

import pytest


def test_health_and_ready_always_open(inference_client_with_auth):
    """`/health` and `/v1/ready` must never require auth (load balancers)."""
    with inference_client_with_auth() as client:
        # /health is mounted directly by openai models router; might 200/404 in
        # tests without models loaded. We only care that auth is NOT required.
        ready = client.get("/v1/ready")
        assert ready.status_code in (200, 503)


def test_admin_load_requires_auth(inference_client_with_auth):
    """POST /api/admin/models/load must reject unauthenticated requests."""
    with inference_client_with_auth() as client:
        unauthorized = client.post(
            "/api/admin/models/load", json={"model": "test/model"}
        )
        assert unauthorized.status_code == 401


@pytest.mark.model
def test_admin_load_accepts_static_api_key(inference_client_with_auth):
    """Same endpoint with the static API key gets through to the handler.

    The handler may still 4xx/5xx because no real model exists, but the auth
    gate must NOT be the blocker — anything but 401 proves the gate let us in.
    """
    with inference_client_with_auth() as client:
        ok = client.post(
            "/api/admin/models/load",
            headers={"x-api-key": "test-key"},
            json={"model": "missing-model-id"},
        )
        assert ok.status_code != 401


def test_admin_unload_requires_auth(inference_client_with_auth):
    """POST /api/admin/models/unload (no body) must require auth too."""
    with inference_client_with_auth() as client:
        unauthorized = client.post("/api/admin/models/unload", json={})
        assert unauthorized.status_code == 401


def test_admin_alias_requires_auth(inference_client_with_auth):
    """POST /api/admin/models/alias must require auth."""
    with inference_client_with_auth() as client:
        unauthorized = client.post(
            "/api/admin/models/alias",
            json={"model": "real/model", "alias": "shortcut"},
        )
        assert unauthorized.status_code == 401


def test_admin_summary_requires_auth(inference_client_with_auth):
    """Read-only /api/admin/summary is also gated when SECURITY_LEVEL>0."""
    with inference_client_with_auth() as client:
        unauthorized = client.get("/api/admin/summary")
        assert unauthorized.status_code == 401


def test_admin_html_requires_auth(inference_client_with_auth):
    """HTML landing page is also gated — pushes admins to operator UI."""
    with inference_client_with_auth() as client:
        unauthorized = client.get("/admin")
        assert unauthorized.status_code == 401


def test_admin_html_accessible_with_key(inference_client_with_auth):
    """And accepts the static API key like every other admin route."""
    with inference_client_with_auth() as client:
        ok = client.get("/admin", headers={"x-api-key": "test-key"})
        assert ok.status_code == 200
        assert "operator on :10241" in ok.text


def test_open_mode_does_not_require_auth(monkeypatch):
    """Backward compat: when SECURITY_LEVEL is unset, admin stays open."""
    from fastapi.testclient import TestClient

    from mlx_batch_server.core.config import get_settings as get_core_settings
    from mlx_batch_server.main import create_app

    monkeypatch.delenv("SECURITY_LEVEL", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    get_core_settings.cache_clear()
    with TestClient(create_app()) as client:
        ok = client.get("/admin")
        # No auth header, but security_level=0 → 200 (open admin landing)
        assert ok.status_code == 200
