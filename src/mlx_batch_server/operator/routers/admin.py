"""Server-rendered htmx admin dashboard at /admin."""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse

from mlx_batch_server.operator.config import Settings, get_settings
from mlx_batch_server.operator.model_registry import registry_rows
from mlx_batch_server.operator.routers.lifecycle import lifecycle_status
from mlx_batch_server.operator.routers.logs import (
    available_log_services,
    resolve_log_path,
)
from mlx_batch_server.operator.routers.playground import (
    PlaygroundRequest,
    proxy_responses,
)
from mlx_batch_server.operator.services.inference_probe import (
    InferenceStatus,
    probe_inference,
)
from mlx_batch_server.operator.services.session_store import session_store

if TYPE_CHECKING:
    from fastapi.templating import Jinja2Templates

router = APIRouter(prefix="/admin", tags=["admin"])

_BASE = "/admin"
_STATIC = "/admin/static"


def _format_bytes(value: int | float | None) -> str:
    if value is None:
        return "-"
    try:
        size = float(value)
    except (TypeError, ValueError):
        return "-"
    if size <= 0:
        return "-"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    index = 0
    while size >= 1024 and index < len(units) - 1:
        size /= 1024.0
        index += 1
    return f"{size:.1f} {units[index]}"


def _templates(request: Request) -> Jinja2Templates:
    templates = getattr(request.app.state, "admin_templates", None)
    if templates is None:
        templates = getattr(request.app.state, "templates", None)
    if templates is None:
        raise RuntimeError("operator templates not configured on app.state")
    return templates


def _ctx(
    settings: Settings,
    *,
    active: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ctx: dict[str, Any] = {
        "active": active,
        "base": _BASE,
        "static_url": _STATIC,
        "port": settings.port,
        "inference_base_url": settings.normalized_inference_base_url,
        "format_bytes": _format_bytes,
    }
    if extra:
        ctx.update(extra)
    return ctx


def _render(
    request: Request,
    settings: Settings,
    name: str,
    *,
    active: str,
    extra: dict[str, Any] | None = None,
    status_code: int = 200,
) -> HTMLResponse:
    return _templates(request).TemplateResponse(
        request,
        name,
        _ctx(settings, active=active, extra=extra),
        status_code=status_code,
    )


def _status_entry(status: InferenceStatus) -> dict[str, Any]:
    health = status.health or {}
    mlx_gb = status.mlx_active_memory_gb
    rss_gb = status.process_rss_gb
    return {
        "service": "mlx-batch-server",
        "port": status.base_url.rsplit(":", 1)[-1],
        "healthy": status.healthy,
        "model_id": ", ".join(status.loaded_models) or None,
        "pinned_model": None,
        "rss_bytes": int(rss_gb * 1024**3) if rss_gb is not None else None,
        "mlx_bytes": int(mlx_gb * 1024**3) if mlx_gb is not None else None,
        "requests_total": health.get("requests_total"),
        "uptime_since": health.get("started_at") or health.get("uptime_since"),
        "tokens_per_sec": health.get("tokens_per_sec"),
        "error": status.error,
    }


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def admin_root(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    return _render(request, settings, "fleet.html", active="fleet")


@router.get("/_partials/fleet", response_class=HTMLResponse)
async def fleet_partial(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    status = await probe_inference(settings)
    return _render(
        request,
        settings,
        "_fleet_cards.html",
        active="fleet",
        extra={"ports": [_status_entry(status)], "inference": status.asdict()},
    )


@router.get("/sessions", response_class=HTMLResponse)
async def sessions_page(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    return _render(request, settings, "sessions.html", active="sessions")


@router.get("/_partials/sessions", response_class=HTMLResponse)
async def sessions_partial(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    sessions = session_store.list_recent(limit=50)
    return _render(
        request,
        settings,
        "_sessions_list.html",
        active="sessions",
        extra={"sessions": sessions},
    )


@router.get("/_partials/session/{session_id}", response_class=HTMLResponse)
async def session_detail_partial(
    session_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    session = session_store.get(session_id)
    return _render(
        request,
        settings,
        "_session_detail.html",
        active="sessions",
        extra={"session": session},
    )


@router.delete("/sessions/{session_id}", response_class=HTMLResponse)
async def session_delete(
    session_id: str,
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    deleted = session_store.delete(session_id)
    sessions = session_store.list_recent(limit=50)
    return _render(
        request,
        settings,
        "_sessions_list.html",
        active="sessions",
        extra={
            "sessions": sessions,
            "message": (
                f"Session {session_id} deleted." if deleted else "Session not found."
            ),
            "level": "ok" if deleted else "warn",
        },
        status_code=200 if deleted else 404,
    )


@router.get("/logs", response_class=HTMLResponse)
async def logs_page(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    services = available_log_services(settings) or ["server"]
    return _render(
        request,
        settings,
        "logs.html",
        active="logs",
        extra={"services": services},
    )


@router.get("/logs/stream")
async def logs_stream(
    request: Request,
    service: str = Query(default="server", min_length=1, max_length=64),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    if not service.replace("-", "").replace("_", "").replace(".", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid service name")
    path = resolve_log_path(settings, service)
    if not path.exists():
        raise HTTPException(
            status_code=404, detail=f"No log file for service '{service}'"
        )

    async def stream() -> AsyncIterator[bytes]:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(0, 2)
            while True:
                if await request.is_disconnected():
                    return
                line = await asyncio.to_thread(handle.readline)
                if line:
                    payload = {"service": service, "line": line.rstrip()}
                    yield f"data: {json.dumps(payload)}\n\n".encode()
                else:
                    await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/lifecycle", response_class=HTMLResponse)
async def lifecycle_page(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    status = await lifecycle_status()
    return _render(
        request,
        settings,
        "lifecycle.html",
        active="lifecycle",
        extra={"status": status},
    )


@router.post("/lifecycle/restart", response_class=HTMLResponse)
async def lifecycle_restart(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    if os.environ.get("MLX_BATCH_UNDER_SUPERVISOR") != "1":
        return _render(
            request,
            settings,
            "_alert.html",
            active="lifecycle",
            extra={
                "message": "Dev mode - restart not available. Stop the process and relaunch.",
                "level": "warn",
            },
            status_code=501,
        )

    loop = asyncio.get_running_loop()
    loop.call_later(2.0, os.kill, os.getpid(), signal.SIGHUP)
    return _render(
        request,
        settings,
        "_alert.html",
        active="lifecycle",
        extra={
            "message": "Backend restart requested. Reconnect in ~5s.",
            "level": "ok",
        },
    )


@router.post("/lifecycle/stop", response_class=HTMLResponse)
async def lifecycle_stop(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    loop = asyncio.get_running_loop()
    loop.call_later(2.0, os.kill, os.getpid(), signal.SIGTERM)
    return _render(
        request,
        settings,
        "_alert.html",
        active="lifecycle",
        extra={
            "message": "Backend stopping. This admin session will end in ~2s.",
            "level": "warn",
        },
    )


@router.get("/playground", response_class=HTMLResponse)
async def playground_page(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> HTMLResponse:
    return _render(
        request,
        settings,
        "playground.html",
        active="playground",
        extra={"models": registry_rows()},
    )


@router.post("/playground/responses")
async def playground_responses(
    model: str = Form(...),
    prompt: str = Form(...),
    session_id: str = Form(default="default"),
    max_output_tokens: int | None = Form(default=None),
    settings: Settings = Depends(get_settings),
) -> StreamingResponse:
    payload = PlaygroundRequest(
        model=model,
        input=[{"role": "user", "content": prompt}],
        stream=True,
        session_id=session_id,
        max_output_tokens=max_output_tokens,
    )
    return await proxy_responses(payload, settings=settings, accept="text/event-stream")
