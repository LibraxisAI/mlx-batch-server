from __future__ import annotations

import pytest

from mlx_batch_server.operator.config import get_settings
from mlx_batch_server.operator.services.session_store import session_store


@pytest.fixture(autouse=True)
def reset_operator_state(monkeypatch: pytest.MonkeyPatch, tmp_path):
    log_path = tmp_path / "logs" / "server.log"
    log_path.parent.mkdir()
    log_path.write_text("line one\nline two\n", encoding="utf-8")
    monkeypatch.setenv("MLX_BATCH_OPERATOR_LOG_PATH", str(log_path))
    monkeypatch.setenv("MLX_BATCH_OPERATOR_INFERENCE_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("MLX_BATCH_OPERATOR_REQUEST_TIMEOUT_SECONDS", "0.05")
    get_settings.cache_clear()
    session_store.clear()
    from mlx_batch_server.operator.routers import playground

    playground._SESSION_HISTORY.clear()
    yield log_path
    get_settings.cache_clear()
    session_store.clear()
    playground._SESSION_HISTORY.clear()


@pytest.fixture
def operator_client():
    from fastapi.testclient import TestClient

    from mlx_batch_server.operator.main import create_app

    with TestClient(create_app()) as client:
        yield client
