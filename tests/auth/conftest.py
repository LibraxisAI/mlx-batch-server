"""Shared fixtures for auth tests.

`build_api_router` re-evaluates settings at every app build, so tests can
flip env vars freely between calls without poking at sys.modules.
"""

from __future__ import annotations

import pytest

from mlx_batch_server.auth import api_keys as api_keys_mod
from mlx_batch_server.auth import hmac as hmac_mod
from mlx_batch_server.auth import router_access as access_mod
from mlx_batch_server.auth import session as session_mod
from mlx_batch_server.core.config import get_settings


@pytest.fixture(autouse=True)
def reset_auth_state(tmp_path, monkeypatch):
    """Reset every singleton + redirect HMAC secrets file to tmp."""
    monkeypatch.setenv(
        "MLX_BATCH_HMAC_SECRETS_FILE", str(tmp_path / "hmac_secrets.json")
    )
    get_settings.cache_clear()
    api_keys_mod._reset_for_tests()
    hmac_mod._reset_for_tests()
    session_mod._reset_for_tests()
    access_mod._reset_for_tests()
    yield
    api_keys_mod._reset_for_tests()
    hmac_mod._reset_for_tests()
    session_mod._reset_for_tests()
    access_mod._reset_for_tests()
    get_settings.cache_clear()


@pytest.fixture
def fresh_app():
    """Return a callable that builds a new FastAPI app with current settings."""

    def _build():
        get_settings.cache_clear()
        from mlx_batch_server.main import create_app

        return create_app()

    return _build
