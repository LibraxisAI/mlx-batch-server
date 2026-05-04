"""Configuration for the standalone operator backend."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _inherit_security_level() -> int:
    """Default operator ``security_level`` to inference ``SECURITY_LEVEL``.

    Operator runs as a sibling app, so its auth posture follows the inference
    server unless explicitly overridden via ``MLX_BATCH_OPERATOR_SECURITY_LEVEL``.
    """
    raw = os.environ.get("SECURITY_LEVEL", "0").strip()
    try:
        return int(raw)
    except ValueError:
        return 0


class Settings(BaseSettings):
    """Operator settings loaded from ``MLX_BATCH_OPERATOR_*`` env vars."""

    model_config = SettingsConfigDict(
        env_prefix="MLX_BATCH_OPERATOR_",
        env_file=".env",
        extra="ignore",
    )

    host: str = Field(default="127.0.0.1")
    port: int = Field(default=10241)
    log_path: Path = Field(
        default_factory=lambda: (
            Path.home() / ".mlx-batch-server" / "logs" / "server.log"
        )
    )
    inference_base_url: str = Field(default="http://127.0.0.1:10240")
    request_timeout_seconds: float = Field(default=30.0)
    redis_url: str | None = Field(default=None)

    # === Auth (mirrors inference SECURITY_LEVEL unless overridden) ===
    security_level: int = Field(
        default_factory=_inherit_security_level,
        description=(
            "Auth gate level. Defaults to inference SECURITY_LEVEL env, can be "
            "overridden with MLX_BATCH_OPERATOR_SECURITY_LEVEL. 0=open, "
            "2=hmac/session/api_key, 3=session-only."
        ),
    )
    require_auth: bool = Field(
        default=False,
        description=(
            "Force auth on every operator route even when security_level=0. "
            "Useful when running operator behind a public reverse proxy with "
            "the inference server kept on a private network."
        ),
    )

    @property
    def normalized_inference_base_url(self) -> str:
        return self.inference_base_url.rstrip("/")

    @property
    def auth_enforced(self) -> bool:
        """True when operator should require auth on protected routes."""
        return bool(self.security_level > 0 or self.require_auth)


@lru_cache
def get_settings() -> Settings:
    return Settings()
