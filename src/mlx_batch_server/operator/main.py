"""FastAPI entrypoint for the standalone MLX Batch Server operator backend."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from mlx_batch_server.operator import STATIC_DIR, TEMPLATES_DIR
from mlx_batch_server.operator.config import get_settings
from mlx_batch_server.operator.routers import (
    admin,
    health,
    lifecycle,
    logs,
    models,
    playground,
    sessions,
)
from mlx_batch_server.operator.services.inference_probe import probe_inference

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        status = await probe_inference(settings)
        if status.healthy:
            logger.info(
                "operator: linked to inference at %s",
                settings.normalized_inference_base_url,
            )
        else:
            logger.warning(
                "operator: inference unavailable at %s: %s",
                settings.normalized_inference_base_url,
                status.error or "health check failed",
            )
        yield

    app = FastAPI(
        title="MLX Batch Server Operator",
        version="0.6.0-dev",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[],
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.templates = templates
    app.state.admin_templates = templates
    if STATIC_DIR.exists():
        app.mount(
            "/admin/static",
            StaticFiles(directory=str(STATIC_DIR)),
            name="admin_static",
        )

    for router in [
        admin.router,
        health.router,
        lifecycle.router,
        logs.router,
        models.router,
        playground.router,
        sessions.router,
    ]:
        app.include_router(router)

    return app


app = create_app()
