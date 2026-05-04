"""``/v1/ready`` shape + status code semantics."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_ready_returns_503_when_no_models_loaded(monkeypatch, fresh_app):
    monkeypatch.delenv("SECURITY_LEVEL", raising=False)
    app = fresh_app()

    with TestClient(app) as client:
        response = client.get("/v1/ready")
    # No models loaded in the test process -> models_loaded is False.
    assert response.status_code == 503
    payload = response.json()
    assert payload["ready"] is False
    assert "checks" in payload
    assert "process" in payload["checks"]
    assert "models_loaded" in payload["checks"]
    assert "version" in payload


def test_ready_includes_auth_check_when_security_enabled(monkeypatch, fresh_app):
    monkeypatch.setenv("SECURITY_LEVEL", "2")
    app = fresh_app()

    with TestClient(app) as client:
        response = client.get("/v1/ready")
    payload = response.json()
    assert "auth_backends" in payload["checks"]
