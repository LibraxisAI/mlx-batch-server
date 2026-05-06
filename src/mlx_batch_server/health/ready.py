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

Returns 200 when all checks pass, 503 otherwise.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from fastapi import APIRouter
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
        from ..chat.mlx.wrapper_cache import wrapper_cache
    except Exception:
        return False
    try:
        loaded = wrapper_cache.get_loaded_models()
    except Exception:
        return False
    return bool(loaded)


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
async def ready() -> JSONResponse:
    settings = get_settings()
    checks: dict[str, Any] = {
        "process": True,
        "models_loaded": _check_models_loaded(),
        "batch_coordinators": _check_batch_coordinators(),
        "config_valid": _check_config_valid(),
    }
    if (settings.security_level or 0) > 0:
        checks["auth_backends"] = await _check_auth_backends()

    all_ready = all(checks.values())
    return JSONResponse(
        status_code=200 if all_ready else 503,
        content={
            "ready": all_ready,
            "checks": checks,
            "version": _package_version(),
        },
    )
