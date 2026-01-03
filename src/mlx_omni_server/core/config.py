"""
Configuration management for MLX Omni Server.

Loads settings from environment variables with sensible defaults.
Supports both local MLX inference and cloud provider fallback.

"""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Cloud Provider Configuration:
    - OPENAI_API_KEY: OpenAI API key for GPT models
    - ANTHROPIC_API_KEY: Anthropic API key for Claude models
    - DEEPINFRA_API_KEY: DeepInfra API key for various models

    Fallback Behavior:
    - ENABLE_CLOUD_FALLBACK: Enable cloud providers as fallback (default: True)
    - CLOUD_FALLBACK_ORDER: Comma-separated provider order (default: openai,anthropic,deepinfra)
    - LOCAL_FIRST: Try local MLX before cloud (default: True)

    Local Provider Configuration:
    - LLM_BASE_URL: Primary local LLM endpoint
    - LLM_ALT_BASE_URL: Secondary local LLM endpoint
    - OLLAMA_API_URL: Ollama server URL

    Model Mapping:
    - DEFAULT_LOCAL_MODEL: Default model for local inference
    - MODEL_ALIASES: JSON mapping of model aliases to actual model names
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # === Cloud Provider API Keys ===
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key")
    deepinfra_api_key: str | None = Field(default=None, description="DeepInfra API key")

    # === Cloud Provider URLs ===
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL",
    )
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com/v1",
        description="Anthropic API base URL",
    )
    deepinfra_base_url: str = Field(
        default="https://api.deepinfra.com/v1/openai",
        description="DeepInfra API base URL (OpenAI-compatible)",
    )

    # === Fallback Behavior ===
    enable_cloud_fallback: bool = Field(
        default=True,
        description="Enable cloud providers as fallback when local fails",
    )
    cloud_fallback_order: str = Field(
        default="openai,anthropic,deepinfra",
        description="Comma-separated cloud provider fallback order",
    )
    local_first: bool = Field(
        default=True,
        description="Try local MLX models before cloud providers",
    )

    # === Local Provider Configuration ===
    llm_base_url: str | None = Field(
        default=None,
        description="Primary local LLM endpoint (e.g., http://localhost:1234)",
    )
    llm_alt_base_url: str | None = Field(
        default=None,
        description="Secondary local LLM endpoint",
    )
    llm_base_urls: str | None = Field(
        default=None,
        description="JSON array or comma-separated list of LLM URLs",
    )
    ollama_api_url: str | None = Field(
        default=None,
        description="Ollama server URL (e.g., http://localhost:11434)",
    )

    # === Model Configuration ===
    default_local_model: str | None = Field(
        default=None,
        description="Default model for local inference",
    )
    model_aliases: str | None = Field(
        default=None,
        description="JSON mapping of model aliases",
    )

    # === Timeouts and Limits ===
    cloud_timeout: int = Field(
        default=120,
        description="Timeout in seconds for cloud provider requests",
    )
    local_timeout: int = Field(
        default=300,
        description="Timeout in seconds for local model requests",
    )
    max_retries: int = Field(
        default=3,
        description="Maximum retries per provider",
    )

    # === Circuit Breaker ===
    circuit_breaker_failure_threshold: int = Field(
        default=5,
        description="Failures before opening circuit",
    )
    circuit_breaker_timeout: int = Field(
        default=60,
        description="Seconds before attempting recovery",
    )
    circuit_breaker_success_threshold: int = Field(
        default=2,
        description="Successes in half-open to close circuit",
    )

    # === Health Check ===
    health_check_interval: int = Field(
        default=30,
        description="Seconds between provider health checks",
    )
    health_check_enabled: bool = Field(
        default=True,
        description="Enable periodic health checks",
    )

    @field_validator("cloud_fallback_order")
    @classmethod
    def validate_fallback_order(cls, v: str) -> str:
        """Validate fallback order contains valid provider names."""
        valid_providers = {"openai", "anthropic", "deepinfra", "ollama", "local"}
        providers = [p.strip().lower() for p in v.split(",") if p.strip()]
        for p in providers:
            if p not in valid_providers:
                raise ValueError(
                    f"Invalid provider '{p}'. Valid options: {valid_providers}"
                )
        return ",".join(providers)

    def get_cloud_fallback_order(self) -> list[str]:
        """Get cloud fallback order as list."""
        return [p.strip() for p in self.cloud_fallback_order.split(",") if p.strip()]

    def get_available_cloud_providers(self) -> list[str]:
        """Get list of configured cloud providers (those with API keys)."""
        providers = []
        if self.openai_api_key:
            providers.append("openai")
        if self.anthropic_api_key:
            providers.append("anthropic")
        if self.deepinfra_api_key:
            providers.append("deepinfra")
        return providers

    def get_provider_for_model(self, model: str) -> str | None:
        """
        Determine which provider to use based on model name.

        Model routing rules:
        - gpt-* -> openai
        - claude-* -> anthropic
        - o1-*, o3-* -> openai (reasoning models)
        - llama-*, mistral-*, qwen-* -> local or deepinfra
        - Local path or HuggingFace ID -> local

        Returns:
            Provider name or None if should try local first
        """
        model_lower = model.lower()

        # OpenAI models
        if model_lower.startswith(
            ("gpt-", "o1-", "o3-", "text-embedding-", "whisper-")
        ):
            return "openai"

        # Anthropic models
        if model_lower.startswith(("claude-",)):
            return "anthropic"

        # DeepInfra models (various open models)
        deepinfra_prefixes = (
            "meta-llama/",
            "mistralai/",
            "deepinfra/",
            "nvidia/",
            "microsoft/",
        )
        if model_lower.startswith(deepinfra_prefixes):
            return "deepinfra"

        # Local path or HuggingFace ID patterns
        if "/" in model or model.startswith(".") or model.startswith("~"):
            return "local"

        # Default: try local first, then fallback chain
        return None

    def get_model_alias(self, model: str) -> str:
        """
        Resolve model alias to actual model name.

        Supports JSON-encoded MODEL_ALIASES env var.
        """
        if not self.model_aliases:
            return model

        try:
            aliases = json.loads(self.model_aliases)
            return aliases.get(model, model)
        except (json.JSONDecodeError, TypeError):
            return model

    def to_dict(self) -> dict[str, Any]:
        """Export settings as dict (hiding sensitive keys)."""
        return {
            "enable_cloud_fallback": self.enable_cloud_fallback,
            "cloud_fallback_order": self.get_cloud_fallback_order(),
            "local_first": self.local_first,
            "available_cloud_providers": self.get_available_cloud_providers(),
            "llm_base_url": self.llm_base_url,
            "llm_alt_base_url": self.llm_alt_base_url,
            "ollama_api_url": self.ollama_api_url,
            "default_local_model": self.default_local_model,
            "cloud_timeout": self.cloud_timeout,
            "local_timeout": self.local_timeout,
            "max_retries": self.max_retries,
            "health_check_enabled": self.health_check_enabled,
            # Mask API keys
            "openai_api_key": "***" if self.openai_api_key else None,
            "anthropic_api_key": "***" if self.anthropic_api_key else None,
            "deepinfra_api_key": "***" if self.deepinfra_api_key else None,
        }


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Settings are loaded once and cached for the lifetime of the process.
    To reload settings, clear the cache: get_settings.cache_clear()
    """
    return Settings()
