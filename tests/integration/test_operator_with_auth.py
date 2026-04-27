"""Operator app respects SECURITY_LEVEL inherited from inference."""

from __future__ import annotations


def test_health_endpoints_always_open(operator_client_with_auth):
    """`/health` and `/api/health` must never require auth on operator side."""
    with operator_client_with_auth() as client:
        for path in ("/health", "/api/health"):
            response = client.get(path)
            assert response.status_code == 200, path
            assert response.json() == {"status": "ok"}


def test_lifecycle_status_requires_auth(operator_client_with_auth):
    with operator_client_with_auth() as client:
        unauthorized = client.get("/api/lifecycle/status")
        assert unauthorized.status_code == 401


def test_lifecycle_status_accepts_api_key(operator_client_with_auth):
    with operator_client_with_auth() as client:
        ok = client.get("/api/lifecycle/status", headers={"x-api-key": "test-key"})
        assert ok.status_code == 200
        body = ok.json()
        assert "pid" in body
        assert "uptime_seconds" in body


def test_admin_root_requires_auth(operator_client_with_auth):
    """The htmx admin entry page is gated."""
    with operator_client_with_auth() as client:
        unauthorized = client.get("/admin/")
        assert unauthorized.status_code == 401


def test_admin_root_accepts_api_key(operator_client_with_auth):
    with operator_client_with_auth() as client:
        ok = client.get("/admin/", headers={"x-api-key": "test-key"})
        assert ok.status_code == 200


def test_models_endpoints_require_auth(operator_client_with_auth):
    with operator_client_with_auth() as client:
        for path in ("/api/models/cache", "/api/models/registry"):
            assert client.get(path).status_code == 401, path


def test_sessions_endpoints_require_auth(operator_client_with_auth):
    with operator_client_with_auth() as client:
        assert client.post("/api/sessions/new").status_code == 401
        assert client.get("/api/sessions/recent").status_code == 401


def test_logs_tail_requires_auth(operator_client_with_auth):
    with operator_client_with_auth() as client:
        unauthorized = client.get("/api/logs/tail")
        assert unauthorized.status_code == 401


def test_open_mode_keeps_operator_unguarded(monkeypatch):
    """Backward compat: operator stays open when SECURITY_LEVEL unset."""
    from fastapi.testclient import TestClient

    from mlx_batch_server.core.config import get_settings as core_settings_cache
    from mlx_batch_server.operator.config import (
        get_settings as op_settings_cache,
    )
    from mlx_batch_server.operator.main import create_app

    monkeypatch.delenv("SECURITY_LEVEL", raising=False)
    monkeypatch.delenv("MLX_BATCH_OPERATOR_SECURITY_LEVEL", raising=False)
    monkeypatch.delenv("MLX_BATCH_OPERATOR_REQUIRE_AUTH", raising=False)
    core_settings_cache.cache_clear()
    op_settings_cache.cache_clear()

    with TestClient(create_app()) as client:
        for path in ("/api/lifecycle/status", "/admin/", "/api/models/cache"):
            assert client.get(path).status_code == 200, path


def test_require_auth_override_forces_gate(monkeypatch):
    """Operator can override and force auth even when inference is open."""
    from fastapi.testclient import TestClient

    from mlx_batch_server.core.config import get_settings as core_settings_cache
    from mlx_batch_server.operator.config import (
        get_settings as op_settings_cache,
    )
    from mlx_batch_server.operator.main import create_app

    monkeypatch.delenv("SECURITY_LEVEL", raising=False)
    monkeypatch.setenv("MLX_BATCH_OPERATOR_REQUIRE_AUTH", "true")
    monkeypatch.setenv("API_KEY", "operator-only-key")
    core_settings_cache.cache_clear()
    op_settings_cache.cache_clear()

    with TestClient(create_app()) as client:
        # Without key: 401 even though inference SECURITY_LEVEL=0
        assert client.get("/api/lifecycle/status").status_code == 401
        # With key: 200
        assert (
            client.get(
                "/api/lifecycle/status",
                headers={"x-api-key": "operator-only-key"},
            ).status_code
            == 200
        )


def test_operator_inherits_inference_security_level(monkeypatch):
    """Default precedence: SECURITY_LEVEL env value bleeds into operator."""
    from mlx_batch_server.core.config import get_settings as core_settings_cache
    from mlx_batch_server.operator.config import (
        get_settings as op_settings_cache,
    )

    monkeypatch.setenv("SECURITY_LEVEL", "3")
    monkeypatch.delenv("MLX_BATCH_OPERATOR_SECURITY_LEVEL", raising=False)
    core_settings_cache.cache_clear()
    op_settings_cache.cache_clear()

    op_settings = op_settings_cache()
    assert op_settings.security_level == 3


def test_operator_explicit_override_wins(monkeypatch):
    """Explicit MLX_BATCH_OPERATOR_SECURITY_LEVEL overrides inheritance."""
    from mlx_batch_server.core.config import get_settings as core_settings_cache
    from mlx_batch_server.operator.config import (
        get_settings as op_settings_cache,
    )

    monkeypatch.setenv("SECURITY_LEVEL", "2")
    monkeypatch.setenv("MLX_BATCH_OPERATOR_SECURITY_LEVEL", "0")
    core_settings_cache.cache_clear()
    op_settings_cache.cache_clear()

    op_settings = op_settings_cache()
    assert op_settings.security_level == 0
