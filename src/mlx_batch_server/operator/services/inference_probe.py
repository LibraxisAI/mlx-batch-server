"""Single-server inference status probe for the operator fleet tab."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from mlx_batch_server.operator.config import Settings, get_settings


@dataclass(slots=True)
class InferenceStatus:
    healthy: bool
    base_url: str
    health: dict[str, Any] | None = None
    ready: dict[str, Any] | None = None
    loaded_models: list[str] = field(default_factory=list)
    loaded_payload: dict[str, Any] | None = None
    version: str | None = None
    process_rss_gb: float | None = None
    mlx_active_memory_gb: float | None = None
    error: str | None = None

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def _loaded_ids(payload: Any) -> list[str]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            return sorted(
                str(item.get("id"))
                for item in data
                if isinstance(item, dict) and item.get("id")
            )
        loaded = payload.get("loaded_models")
        if isinstance(loaded, list):
            return sorted(str(item) for item in loaded)
    return []


async def probe_inference(settings: Settings | None = None) -> InferenceStatus:
    """Probe the colocated inference server without failing the operator."""

    settings = settings or get_settings()
    base_url = settings.normalized_inference_base_url
    timeout = httpx.Timeout(settings.request_timeout_seconds)
    status = InferenceStatus(healthy=False, base_url=base_url)

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            health_response = await client.get(f"{base_url}/health")
            status.healthy = health_response.is_success
            if health_response.headers.get("content-type", "").startswith(
                "application/json"
            ):
                status.health = health_response.json()
                version = status.health.get("version")
                status.version = version if isinstance(version, str) else None

            try:
                loaded_response = await client.get(f"{base_url}/v1/models/loaded")
                if loaded_response.is_success:
                    loaded_payload = loaded_response.json()
                    status.loaded_payload = loaded_payload
                    status.loaded_models = _loaded_ids(loaded_payload)
            except (httpx.HTTPError, ValueError):
                pass

            try:
                ready_response = await client.get(f"{base_url}/v1/ready")
                if ready_response.is_success:
                    status.ready = ready_response.json()
            except (httpx.HTTPError, ValueError):
                pass
    except httpx.HTTPError as exc:
        status.error = str(exc)

    return status
