"""
MLX Batch Server - Entry point.

Provides OpenAI-compatible APIs using Apple's MLX framework.
"""

# Patch deprecated pkg_resources usage in jieba (via f5-tts-mlx)
# MUST be first import - before any module that transitively imports jieba
from mlx_batch_server.utils import compat as _compat  # noqa: F401

import argparse
import os


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
        default="",
        help='CORS origins, comma-separated (e.g., "*" or "http://localhost:3000")',
    )
    return parser


def create_app():
    """Create and configure the FastAPI application.

    This is called lazily to avoid slow imports when just showing --help.
    """
    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from .middleware.logging import RequestResponseLoggingMiddleware
    from .routers import api_router

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager for startup/shutdown hooks."""
        # Startup
        yield
        # Shutdown - cleanup batch coordinators
        try:
            from .batch import shutdown_all_coordinators

            await shutdown_all_coordinators()
        except ImportError:
            pass  # Batch module may not be available

    application = FastAPI(title="MLX Batch Server", lifespan=lifespan)

    # Add request/response logging middleware
    application.add_middleware(RequestResponseLoggingMiddleware)

    # Include all API routes
    application.include_router(api_router)

    # Configure CORS from environment
    cors_origins = os.environ.get("MLX_BATCH_CORS", "")
    if cors_origins:
        origins = [origin.strip() for origin in cors_origins.split(",")]
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
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
