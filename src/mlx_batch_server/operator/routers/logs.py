"""Log tail and follow endpoints."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from mlx_batch_server.operator.config import Settings, get_settings

router = APIRouter(prefix="/api/logs", tags=["logs"])


def resolve_log_path(settings: Settings, service: str | None = None) -> Path:
    if service in {None, "", "server", "mlx-batch-server"}:
        return settings.log_path
    return settings.log_path.parent / f"{service}.log"


def available_log_services(settings: Settings) -> list[str]:
    logs_dir = settings.log_path.parent
    if not logs_dir.exists():
        return []
    services = [
        path.stem
        for path in logs_dir.glob("*.log")
        if path.is_file() and not path.stem.startswith(".")
    ]
    return sorted(set(services))


@router.get("/tail")
async def tail_logs(
    service: str | None = None,
    lines: int = Query(default=200, ge=1, le=5000),
    settings: Settings = Depends(get_settings),
) -> dict[str, str | list[str]]:
    path = resolve_log_path(settings, service)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Log file not found: {path}")
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {
        "service": service or "server",
        "path": str(path),
        "lines": content[-lines:],
    }


@router.get("/follow")
async def follow_logs(
    request: Request,
    service: str | None = None,
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    path = resolve_log_path(settings, service)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Log file not found: {path}")

    async def stream() -> AsyncIterator[bytes]:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(0, 2)
            while True:
                if await request.is_disconnected():
                    return
                line = await asyncio.to_thread(handle.readline)
                if line:
                    payload = {"line": line.rstrip(), "service": service or "server"}
                    yield f"data: {json.dumps(payload)}\n\n".encode()
                else:
                    await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")
