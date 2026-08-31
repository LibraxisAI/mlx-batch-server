"""
MLX Batch Server - Entry point.

Provides OpenAI-compatible APIs using Apple's MLX framework.
"""

# Patch deprecated pkg_resources usage in jieba (via f5-tts-mlx)
# MUST be first import - before any module that transitively imports jieba
from mlx_batch_server.utils import compat as _compat  # noqa: F401

import argparse
import asyncio
import os
import re

from .provenance import get_runtime_provenance, stamp_runtime_environment

DEFAULT_CORS_ALLOW_ORIGINS = (
    "http://localhost:*,http://127.0.0.1:*,http://100.*:*,https://100.*:*"
)
# NOTE: The `100.*` wildcard covers Tailscale/CGNAT address ranges.
# In a tightly controlled environment this is acceptable for local/tailnet use;
# override via the MLX_BATCH_CORS env variable to restrict to localhost-only
# (e.g. "http://localhost:*,http://127.0.0.1:*") in any internet-exposed deployment.


def _build_cors_config(cors_origins: str) -> tuple[list[str], str | None]:
    """Split exact origins from wildcard origins and compile a regex for the latter."""
    origins = [origin.strip() for origin in cors_origins.split(",") if origin.strip()]
    if not origins:
        return [], None

    if "*" in origins:
        return ["*"], None

    exact_origins: list[str] = []
    wildcard_patterns: list[str] = []

    for origin in origins:
        if "*" not in origin:
            exact_origins.append(origin)
            continue

        # Turn user-friendly origin globs such as https://*.tail.ts.net into a
        # strict origin regex accepted by Starlette's CORSMiddleware.
        wildcard_patterns.append(re.escape(origin).replace(r"\*", r"[^/]+"))

    allow_origin_regex = None
    if wildcard_patterns:
        allow_origin_regex = rf"^(?:{'|'.join(wildcard_patterns)})$"

    return exact_origins, allow_origin_regex


def build_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the server."""
    parser = argparse.ArgumentParser(
        description="MLX Batch Server - OpenAI-compatible APIs on Apple Silicon"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=10240,
        help="Port to bind the server to (default: 10240)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of workers to use (default: 1)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Set the logging level (default: info)",
    )
    parser.add_argument(
        "--cors-allow-origins",
        type=str,
        default=DEFAULT_CORS_ALLOW_ORIGINS,
        help='CORS origins, comma-separated (e.g., "*" or "http://localhost:3000")',
    )
    return parser


def create_app():  # noqa: PLR0915
    """Create and configure the FastAPI application.

    This is called lazily to avoid slow imports when just showing --help.
    """
    # Direct ``uvicorn module:app`` launches bypass ``start()``. Freeze source
    # identity during app construction so no launch path shells out per request.
    get_runtime_provenance()

    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from .core.config import get_settings
    from .middleware.logging import RequestResponseLoggingMiddleware
    from .routers import build_api_router

    settings = get_settings()
    api_router = build_api_router(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager for startup/shutdown hooks."""
        from .runtime_recycle import (
            start_idle_process_recycler,
            stop_idle_process_recycler,
        )

        await start_idle_process_recycler()
        # Startup: opt-in session auth manager
        if settings.session_auth_enabled:
            os.environ.setdefault("SESSION_PROVIDER", settings.session_provider)
            os.environ.setdefault("SESSION_REDIS_URL", settings.redis_url)
            os.environ.setdefault(
                "SESSION_DEFAULT_TTL_HOURS", str(settings.session_ttl_hours)
            )
            from .auth.session import session_auth

            await session_auth.start()
        try:
            yield
        finally:
            await stop_idle_process_recycler()
            # Shutdown - cleanup batch coordinators
            if settings.session_auth_enabled:
                try:
                    from .auth.session import session_auth

                    await session_auth.stop()
                except Exception:
                    pass
            from .images.image_runtime import shutdown_image_runtime_pool

            try:
                from .batch import shutdown_all_coordinators
                from .vision.vlm_batch import shutdown_all_vlm_coordinators

                await shutdown_all_coordinators()
                await shutdown_all_vlm_coordinators()
            except ImportError:
                pass  # Batch module may not be available
            finally:
                await shutdown_image_runtime_pool()
                from .aux_runtime import shutdown_aux_runtime_manager

                await asyncio.to_thread(shutdown_aux_runtime_manager)

    application = FastAPI(title="MLX Batch Server", lifespan=lifespan)

    # Add request/response logging middleware
    application.add_middleware(RequestResponseLoggingMiddleware)

    # Opt-in rate limiting (added BEFORE logging so it fires earlier in LIFO chain)
    if settings.rate_limit_enabled:
        from .auth.rate_limit import RateLimitMiddleware

        application.add_middleware(
            RateLimitMiddleware,
            requests_per_minute=settings.rate_limit_per_minute,
            window_size=settings.rate_limit_window_seconds,
            concurrent_limit=settings.rate_limit_concurrent,
            exempt_paths=settings.get_rate_limit_exempt_paths(),
        )

    # Include all API routes
    application.include_router(api_router)

    # Configure CORS from environment
    cors_origins = os.environ.get("MLX_BATCH_CORS", DEFAULT_CORS_ALLOW_ORIGINS)
    if cors_origins:
        origins, allow_origin_regex = _build_cors_config(cors_origins)
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_origin_regex=allow_origin_regex,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    return application


# Lazy app instance for uvicorn
# This is only created when uvicorn imports the module, not during CLI --help
_app_instance = None


def _get_app():
    """Get or create the FastAPI app instance (for uvicorn)."""
    global _app_instance
    if _app_instance is None:
        _app_instance = create_app()
    return _app_instance


# Module-level __getattr__ for lazy loading
# When uvicorn accesses main.app, this creates the app lazily
def __getattr__(name):
    if name == "app":
        return _get_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def start():
    """Start the MLX Batch Server."""
    # Parse args FIRST - before any heavy imports
    # This makes --help instant
    parser = build_parser()
    args = parser.parse_args()

    # Set environment variables for app configuration
    os.environ["MLX_BATCH_LOG_LEVEL"] = args.log_level
    os.environ["MLX_BATCH_CORS"] = args.cors_allow_origins
    # Capture the source checkout before Uvicorn imports the app or forks
    # workers. The inherited environment is the provenance contract.
    stamp_runtime_environment()

    # NOW import uvicorn and start (lazy import)
    import uvicorn

    from .utils.logger import logger, set_logger_level

    set_logger_level(logger, args.log_level)

    # Start server - uvicorn will import the app via __getattr__
    uvicorn.run(
        "mlx_batch_server.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        use_colors=True,
        workers=args.workers,
    )


if __name__ == "__main__":
    start()
