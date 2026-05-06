"""Lifecycle management endpoints for the operator backend."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from datetime import UTC, datetime

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from mlx_batch_server.operator.auth import operator_auth

router = APIRouter(prefix="/api/lifecycle", tags=["lifecycle"])

_STARTED_AT = datetime.now(UTC)


def _build_status() -> dict[str, int | str]:
    now = datetime.now(UTC)
    uptime = (now - _STARTED_AT).total_seconds()
    return {
        "pid": os.getpid(),
        "started_at": _STARTED_AT.isoformat(),
        "uptime_seconds": int(uptime),
        "version": "0.6.0-dev",
    }


@router.get("/status")
async def lifecycle_status(
    _auth: dict | None = Depends(operator_auth),
) -> dict[str, int | str]:
    return _build_status()


@router.post("/restart-backend", response_model=None)
async def restart_backend(
    _auth: dict | None = Depends(operator_auth),
) -> dict[str, str] | JSONResponse:
    if os.environ.get("MLX_BATCH_UNDER_SUPERVISOR") != "1":
        return JSONResponse(
            status_code=501,
            content={"detail": "Running in dev mode - use Ctrl+C and relaunch"},
        )

    loop = asyncio.get_running_loop()
    loop.call_later(2.0, os.kill, os.getpid(), signal.SIGHUP)
    return {"status": "restarting", "message": "Backend will restart in ~2 seconds"}


@router.post("/stop-backend")
async def stop_backend(
    _auth: dict | None = Depends(operator_auth),
) -> dict[str, str]:
    logger = logging.getLogger(__name__)
    logger.info("Graceful stop requested via /api/lifecycle/stop-backend")

    loop = asyncio.get_running_loop()
    loop.call_later(2.0, os.kill, os.getpid(), signal.SIGTERM)
    return {"status": "stopping"}
