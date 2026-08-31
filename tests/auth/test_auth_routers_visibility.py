"""Auth routers must be opt-in: NOT mounted on the default app."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_auth_routes_absent_by_default(monkeypatch, fresh_app):
    monkeypatch.delenv("SECURITY_LEVEL", raising=False)
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("SESSION_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("ACCESS_REGISTRATION_SECRET", raising=False)
    app = fresh_app()

    routes = set(app.openapi()["paths"])
    assert "/auth/login" not in routes
    assert "/hmac/register" not in routes
    assert "/access" not in routes


def test_auth_routes_appear_when_security_level_set(monkeypatch, fresh_app):
    monkeypatch.setenv("SECURITY_LEVEL", "2")
    app = fresh_app()

    routes = set(app.openapi()["paths"])
    assert "/auth/login" in routes
    assert "/hmac/register" in routes
    assert "/access" in routes


def test_auth_login_returns_400_when_session_disabled(monkeypatch, fresh_app):
    monkeypatch.setenv("SECURITY_LEVEL", "2")
    monkeypatch.setenv("API_KEY", "test-static")
    app = fresh_app()

    with TestClient(app) as client:
        response = client.post(
            "/auth/login",
            headers={"x-api-key": "test-static"},
            json={"user_id": "alice"},
        )
    # session_auth_enabled defaults to False -> 400
    assert response.status_code == 400


def test_static_api_key_protects_hmac_register(monkeypatch, fresh_app):
    monkeypatch.setenv("SECURITY_LEVEL", "2")
    monkeypatch.setenv("API_KEY", "test-static-key")
    app = fresh_app()

    with TestClient(app) as client:
        # Without the key -> 401
        unauthorized = client.post(
            "/hmac/register", json={"client_id": "device-x", "description": ""}
        )
        assert unauthorized.status_code == 401

        # With the key -> 200, secret comes back once.
        ok = client.post(
            "/hmac/register",
            headers={"x-api-key": "test-static-key"},
            json={"client_id": "device-x", "description": ""},
        )
        assert ok.status_code == 200
        body = ok.json()
        assert body["client_id"] == "device-x"
        assert len(body["secret_key"]) == 64
