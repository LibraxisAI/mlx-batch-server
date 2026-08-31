"""
Configuration management for Mlx batch Server.

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

    # === Batch Processing ===
    enable_batch_inference: bool = Field(
        default=True,
        description="Enable batch processing for concurrent requests",
    )
    batch_window_ms: int = Field(
        default=50,
        description="Time window in ms to collect requests before processing batch",
    )
    max_batch_size: int = Field(
        default=10,
        description="Maximum number of requests per batch",
    )
    batch_completion_size: int = Field(
        default=32,
        description="Number of sequences to process per batch step (MLX parameter)",
    )
    batch_prefill_size: int = Field(
        default=8,
        description="Number of sequences to prefill together",
    )
    batch_prefill_step_size: int = Field(
        default=2048,
        description="Number of tokens to prefill per step",
    )

    # === VLM Batch Processing ===
    vlm_batch_enabled: bool = Field(
        default=True,
        description="Enable VLM micro-batch coordinator for eligible vision requests",
    )
    vlm_batch_window_ms: int = Field(
        default=50,
        description="Time window in ms to collect VLM requests before batching",
    )
    vlm_max_batch_size: int = Field(
        default=4,
        description="Maximum VLM requests per batch",
    )
    vlm_batch_group_by_shape: bool = Field(
        default=True,
        description="Group VLM requests by image shape for efficient batching",
    )
    vlm_batch_pad_to_uniform_size: bool = Field(
        default=True,
        description="Pad images to uniform size within a VLM batch",
    )
    vlm_batch_resize_shape: str | None = Field(
        default=None,
        description="Force resize images to this shape (WxH) before VLM batching",
    )
    vlm_stream_batch_enabled: bool = Field(
        default=True,
        description="Enable streaming VLM batch coordinator for eligible vision requests",
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

    # === Model Cache Configuration ===
    model_cache_max_size: int = Field(
        default=1,
        description="Max extra models in cache (beyond pinned). Set 0 for pinned-only.",
    )
    model_cache_ttl: int = Field(
        default=600,
        description="TTL in seconds for non-pinned models (default: 10 minutes)",
    )
    image_model_idle_ttl_seconds: float = Field(
        default=600,
        ge=0,
        description="Idle seconds before the heavyweight image worker is retired",
    )
    pinned_models: str = Field(
        default="",
        description="Comma-separated model IDs to keep always loaded (never evict)",
    )

    # === Auth Core ===
    # 0 = open (no auth, default), 2 = HMAC/session/api_key, 3 = session-only.
    # Level 1 is deprecated and silently treated as 2 by the dependency.
    security_level: int = Field(
        default=0,
        description="Auth gate level. 0=open, 2=hmac/session/api_key, 3=session-only",
    )
    session_auth_enabled: bool = Field(
        default=False,
        description="Enable /auth/* session lifecycle and bearer-token validation",
    )
    session_provider: str = Field(
        default="memory",
        description='Session backend: "memory" (default) or "redis"',
    )
    session_ttl_hours: int = Field(
        default=24,
        description="Default session TTL in hours",
    )
    api_key: str | None = Field(
        default=None,
        description="Static admin API key (None = disabled)",
    )
    api_key_header: str = Field(
        default="x-api-key",
        description="Header name carrying the API key",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis URL used by api_keys/hmac/sessions/rate-limit when enabled",
    )

    # === Rate Limiting ===
    rate_limit_enabled: bool = Field(
        default=False,
        description="Enable global RateLimitMiddleware (opt-in)",
    )
    rate_limit_per_minute: int = Field(
        default=60,
        description="Max requests per window per client",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        description="Rate limit window in seconds",
    )
    rate_limit_concurrent: int = Field(
        default=10,
        description="Max concurrent in-flight requests per client",
    )
    rate_limit_exempt_paths: str = Field(
        default="/,/health,/v1/ready,/metrics",
        description="Comma-separated paths exempt from rate limiting",
    )

    # === Access Registration ===
    access_registration_secret: str | None = Field(
        default=None,
        description="HMAC secret for /access registration tokens (None = /access disabled)",
    )
    access_registration_hashes: str = Field(
        default="",
        description="JSON array or CSV of pre-authorized registration token hashes",
    )
    access_rate_limit_per_minute: int = Field(
        default=5,
        description="Max /access issuance attempts per minute per IP",
    )
    access_max_ttl_hours: int = Field(
        default=336,
        description="Max TTL (hours) for issued API keys (default: 14 days)",
    )

    # === HMAC ===
    mlx_batch_hmac_secrets_file: str | None = Field(
        default=None,
        description="Override path for HMAC secrets file (None = XDG default)",
    )
    hmac_timestamp_tolerance: int = Field(
        default=300,
        description="HMAC request timestamp tolerance window in seconds",
    )

    debug: bool = Field(
        default=False,
        description="Debug mode flag (downgrades a few error logs to warnings)",
    )

    def get_rate_limit_exempt_paths(self) -> list[str]:
        """Parse rate_limit_exempt_paths into a list."""
        return [
            path.strip()
            for path in self.rate_limit_exempt_paths.split(",")
            if path.strip()
        ]

    def get_pinned_models(self) -> list[str]:
        """Get list of pinned model IDs that should never be evicted."""
        if not self.pinned_models:
            return []
        return [m.strip() for m in self.pinned_models.split(",") if m.strip()]

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
            # Batch processing
            "enable_batch_inference": self.enable_batch_inference,
            "batch_window_ms": self.batch_window_ms,
            "max_batch_size": self.max_batch_size,
            "batch_completion_size": self.batch_completion_size,
            "batch_prefill_size": self.batch_prefill_size,
            "vlm_batch_enabled": self.vlm_batch_enabled,
            "vlm_batch_window_ms": self.vlm_batch_window_ms,
            "vlm_max_batch_size": self.vlm_max_batch_size,
            "vlm_batch_group_by_shape": self.vlm_batch_group_by_shape,
            "vlm_batch_resize_shape": self.vlm_batch_resize_shape,
            "vlm_batch_pad_to_uniform_size": self.vlm_batch_pad_to_uniform_size,
            "vlm_stream_batch_enabled": self.vlm_stream_batch_enabled,
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
