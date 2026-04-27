"""Lifecycle management endpoints for the operator backend."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from datetime import UTC, datetime

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/api/lifecycle", tags=["lifecycle"])

_STARTED_AT = datetime.now(UTC)


@router.get("/status")
async def lifecycle_status() -> dict[str, int | str]:
    now = datetime.now(UTC)
    uptime = (now - _STARTED_AT).total_seconds()
    return {
        "pid": os.getpid(),
        "started_at": _STARTED_AT.isoformat(),
        "uptime_seconds": int(uptime),
        "version": "0.6.0-dev",
    }


@router.post("/restart-backend", response_model=None)
async def restart_backend() -> dict[str, str] | JSONResponse:
    if os.environ.get("MLX_BATCH_UNDER_SUPERVISOR") != "1":
        return JSONResponse(
            status_code=501,
            content={"detail": "Running in dev mode - use Ctrl+C and relaunch"},
        )

    loop = asyncio.get_running_loop()
    loop.call_later(2.0, os.kill, os.getpid(), signal.SIGHUP)
    return {"status": "restarting", "message": "Backend will restart in ~2 seconds"}


@router.post("/stop-backend")
async def stop_backend() -> dict[str, str]:
    logger = logging.getLogger(__name__)
    logger.info("Graceful stop requested via /api/lifecycle/stop-backend")

    loop = asyncio.get_running_loop()
    loop.call_later(2.0, os.kill, os.getpid(), signal.SIGTERM)
    return {"status": "stopping"}
