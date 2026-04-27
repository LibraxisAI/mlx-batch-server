"""Shared fixtures for integration tests covering inference + operator + auth."""

from __future__ import annotations

import pytest

from mlx_batch_server.auth import api_keys as api_keys_mod
from mlx_batch_server.auth import hmac as hmac_mod
from mlx_batch_server.auth import router_access as access_mod
from mlx_batch_server.auth import session as session_mod
from mlx_batch_server.core.config import get_settings as get_core_settings
from mlx_batch_server.operator.config import get_settings as get_operator_settings


@pytest.fixture(autouse=True)
def reset_state(monkeypatch, tmp_path):
    """Hard-reset every singleton + redirect HMAC secrets file to tmp_path."""
    monkeypatch.setenv(
        "MLX_BATCH_HMAC_SECRETS_FILE", str(tmp_path / "hmac_secrets.json")
    )

    # Operator log path / inference URL stub (operator probe will fail closed,
    # but that just emits a warning and does not block tests).
    log_path = tmp_path / "logs" / "server.log"
    log_path.parent.mkdir()
    log_path.write_text("test\n", encoding="utf-8")
    monkeypatch.setenv("MLX_BATCH_OPERATOR_LOG_PATH", str(log_path))
    monkeypatch.setenv("MLX_BATCH_OPERATOR_INFERENCE_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("MLX_BATCH_OPERATOR_REQUEST_TIMEOUT_SECONDS", "0.05")

    get_core_settings.cache_clear()
    get_operator_settings.cache_clear()
    api_keys_mod._reset_for_tests()
    hmac_mod._reset_for_tests()
    session_mod._reset_for_tests()
    access_mod._reset_for_tests()
    yield
    api_keys_mod._reset_for_tests()
    hmac_mod._reset_for_tests()
    session_mod._reset_for_tests()
    access_mod._reset_for_tests()
    get_core_settings.cache_clear()
    get_operator_settings.cache_clear()


@pytest.fixture
def inference_client_with_auth(monkeypatch):
    """Build an inference TestClient with SECURITY_LEVEL=2 + static API key."""

    def _build():
        from fastapi.testclient import TestClient

        from mlx_batch_server.main import create_app

        get_core_settings.cache_clear()
        return TestClient(create_app())

    monkeypatch.setenv("SECURITY_LEVEL", "2")
    monkeypatch.setenv("API_KEY", "test-key")
    return _build


@pytest.fixture
def operator_client_with_auth(monkeypatch):
    """Build an operator TestClient inheriting SECURITY_LEVEL=2 from env."""

    def _build():
        from fastapi.testclient import TestClient

        from mlx_batch_server.operator.main import create_app

        get_operator_settings.cache_clear()
        return TestClient(create_app())

    monkeypatch.setenv("SECURITY_LEVEL", "2")
    monkeypatch.setenv("API_KEY", "test-key")
    return _build
