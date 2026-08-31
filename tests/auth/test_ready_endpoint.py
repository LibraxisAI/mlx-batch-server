"""``/v1/ready`` shape + status code semantics."""

from __future__ import annotations

from fastapi.testclient import TestClient

from mlx_batch_server.chat.openai.models import models as models_module
from mlx_batch_server.health import ready as ready_module


def test_ready_returns_200_when_healthy_process_is_cold(monkeypatch, fresh_app):
    monkeypatch.delenv("SECURITY_LEVEL", raising=False)
    monkeypatch.setattr(ready_module, "_check_models_loaded", lambda: False)
    app = fresh_app()

    with TestClient(app) as client:
        response = client.get("/v1/ready")
    # Models load on demand, so correct cold residency must remain ready.
    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is True
    assert "checks" in payload
    assert "process" in payload["checks"]
    assert "models_loaded" in payload["checks"]
    assert payload["checks"]["models_loaded"] is False
    assert "version" in payload


def test_ready_includes_auth_check_when_security_enabled(monkeypatch, fresh_app):
    monkeypatch.setenv("SECURITY_LEVEL", "2")
    app = fresh_app()

    with TestClient(app) as client:
        response = client.get("/v1/ready")
    payload = response.json()
    assert "auth_backends" in payload["checks"]


def test_models_loaded_check_includes_non_llm_residency(monkeypatch):
    monkeypatch.setattr(models_module, "_snapshot_llm_runtime", lambda: {})
    monkeypatch.setattr(
        models_module,
        "_snapshot_process_residency",
        lambda _runtime: {"loaded_models_count": 1},
    )

    assert ready_module._check_models_loaded() is True
