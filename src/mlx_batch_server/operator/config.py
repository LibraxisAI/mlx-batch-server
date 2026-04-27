"""Configuration for the standalone operator backend."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    @property
    def normalized_inference_base_url(self) -> str:
        return self.inference_base_url.rstrip("/")


@lru_cache
def get_settings() -> Settings:
    return Settings()
