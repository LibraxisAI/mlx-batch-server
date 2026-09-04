"""``/v1/ready`` — rich readiness signal beyond ``/health``.

Response shape::

    {
        "ready": bool,
        "checks": {
            "process": True,
            "models_loaded": <bool>,
            "batch_coordinators": <bool>,
            "config_valid": <bool>,
            "auth_backends": <bool>,        # only when SECURITY_LEVEL > 0
        },
        "version": "0.6.0-dev"
    }

Cold model residency is healthy: models load on demand. The residency check is
reported for operators but does not gate readiness. Returns 200 when the
process/config/coordinator/auth checks pass, 503 otherwise.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from ..core.config import get_settings

router = APIRouter(tags=["health"])


def _package_version() -> str:
    try:
        return version("mlx-batch-server")
    except PackageNotFoundError:
        return "0.0.0+unknown"


def _check_models_loaded() -> bool:
    try:
        from ..chat.openai.models.models import (
            _snapshot_llm_runtime,
            _snapshot_process_residency,
        )
    except Exception:
        return False
    try:
        residency = _snapshot_process_residency(_snapshot_llm_runtime())
    except Exception:
        return False
    return bool(residency["loaded_models_count"])


def _check_batch_coordinators() -> bool:
    """Batch coordinators are spun up on-demand; we treat the module as healthy
    as long as the registry is importable."""
    try:
        from ..batch.coordinator import _coordinators  # noqa: F401
    except Exception:
        return False
    return True


def _check_config_valid() -> bool:
    try:
        get_settings()
    except Exception:
        return False
    return True


async def _check_auth_backends() -> bool:
    """Only invoked when security_level > 0."""
    settings = get_settings()
    if settings.session_provider == "redis" or settings.rate_limit_enabled:
        try:
            import redis.asyncio as redis
        except ImportError:
            return False
        try:
            client = redis.from_url(settings.redis_url, decode_responses=True)
            await client.ping()
            await client.close()
            return True
        except Exception:
            return False
    return True


@router.get("/v1/ready")
async def ready(request: Request) -> JSONResponse:
    role_control = getattr(request.app.state, "role_control_service", None)
    if role_control is not None:
        status_code, payload = role_control.ready_payload()
        payload["version"] = _package_version()
        return JSONResponse(status_code=status_code, content=payload)

    settings = get_settings()
    checks: dict[str, Any] = {
        "process": True,
        "models_loaded": _check_models_loaded(),
        "batch_coordinators": _check_batch_coordinators(),
        "config_valid": _check_config_valid(),
    }
    if (settings.security_level or 0) > 0:
        checks["auth_backends"] = await _check_auth_backends()

    required_checks = {
        key: value for key, value in checks.items() if key != "models_loaded"
    }
    all_ready = all(required_checks.values())
    return JSONResponse(
        status_code=200 if all_ready else 503,
        content={
            "ready": all_ready,
            "checks": checks,
            "version": _package_version(),
        },
    )
