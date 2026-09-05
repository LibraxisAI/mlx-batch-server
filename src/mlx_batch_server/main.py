"""
MLX Batch Server - Entry point.

Provides OpenAI-compatible APIs using Apple's MLX framework.
"""

# Patch deprecated pkg_resources usage in jieba (via f5-tts-mlx)
# MUST be first import - before any module that transitively imports jieba
from mlx_batch_server.utils import compat as _compat  # noqa: F401

import argparse
import asyncio
import json
import os
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from .provenance import get_runtime_provenance, stamp_runtime_environment

if TYPE_CHECKING:
    from .responses.errors import OpenAIError

DEFAULT_CORS_ALLOW_ORIGINS = (
    "http://localhost:*,http://127.0.0.1:*,http://100.*:*,https://100.*:*"
)
PRODUCTION_ROLE_PORTS = frozenset({8100, 8101, 8102})
_RESPONSES_REQUEST_BODY_PATHS = frozenset(
    {
        "/v1/responses",
        "/v1/responses/compact",
        "/v1/responses/input_tokens",
    }
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


def _is_responses_request_validation(request: Any) -> bool:
    if getattr(request, "method", None) != "POST":
        return False
    scope = getattr(request, "scope", {})
    route = scope.get("route") if isinstance(scope, Mapping) else None
    route_path = getattr(route, "path", None)
    request_path = getattr(getattr(request, "url", None), "path", None)
    resolved_path = route_path if isinstance(route_path, str) else request_path
    return resolved_path in _RESPONSES_REQUEST_BODY_PATHS


def _validation_param(error: Mapping[str, Any]) -> str | None:
    location = error.get("loc", ())
    if not isinstance(location, Sequence) or isinstance(location, str | bytes):
        return None
    transport_parts = {"body", "query", "path", "header", "cookie"}
    parts = [str(part) for part in location if part not in transport_parts]
    if parts:
        return ".".join(parts)
    return "body" if "body" in location else None


def _openai_request_validation_error(exception: Any) -> "OpenAIError":
    from .responses.errors import OpenAIError

    raw_errors = exception.errors()
    errors = raw_errors if isinstance(raw_errors, list) else []
    first = errors[0] if errors and isinstance(errors[0], Mapping) else {}
    error_kind = first.get("type")
    param = _validation_param(first)

    if error_kind == "json_invalid":
        message = "Malformed JSON request body."
        code = "invalid_json"
        param = "body"
    elif error_kind in {"dict_type", "model_attributes_type", "model_type"}:
        message = "Request body must be a JSON object."
        code = "invalid_request"
        param = "body"
    elif error_kind == "missing" and param == "body":
        message = "Request body is required."
        code = "invalid_request"
    elif error_kind == "missing" and param is not None:
        message = f"Missing required parameter: '{param}'."
        code = "invalid_request"
    elif param is not None and param != "body":
        message = f"Invalid value for '{param}'."
        code = "invalid_request"
    else:
        message = "Invalid request body."
        code = "invalid_request"

    return OpenAIError(
        message=message,
        type="invalid_request_error",
        code=code,
        param=param,
        status_code=400,
    )


def _openai_raw_body_error(body: bytes) -> "OpenAIError | None":
    from .responses.errors import OpenAIError

    if not isinstance(body, bytes):
        raise TypeError("body must be bytes")
    if not body.strip():
        return OpenAIError(
            message="Request body is required.",
            type="invalid_request_error",
            code="invalid_request",
            param="body",
            status_code=400,
        )
    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return OpenAIError(
            message="Malformed JSON request body.",
            type="invalid_request_error",
            code="invalid_json",
            param="body",
            status_code=400,
        )
    if isinstance(payload, Mapping):
        return None
    return OpenAIError(
        message="Request body must be a JSON object.",
        type="invalid_request_error",
        code="invalid_request",
        param="body",
        status_code=400,
    )


async def _responses_request_validation_exception_handler(
    request: Any,
    exception: Any,
) -> Any:
    """Normalize validation only after routing to a Responses body endpoint."""

    if _is_responses_request_validation(request):
        from .responses.errors import render_http_error

        return render_http_error(_openai_request_validation_error(exception))

    from fastapi.exception_handlers import request_validation_exception_handler

    return await request_validation_exception_handler(request, exception)


async def _responses_body_exception_boundary(
    request: Any,
    call_next: Any,
) -> Any:
    """Reject invalid Responses JSON before generic middleware can inspect it."""

    if _is_responses_request_validation(request):
        from .responses.errors import render_http_error

        error = _openai_raw_body_error(await request.body())
        if error is not None:
            return render_http_error(error)
    return await call_next(request)


def build_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for the server."""
    parser = argparse.ArgumentParser(
        description="MLX Batch Server - OpenAI-compatible APIs on Apple Silicon"
    )
    parser.add_argument(
        "--host",
        type=str,
        # Intentional, configurable server bind; the local default port is 10240.
        default="0.0.0.0",  # nosec B104
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
    parser.add_argument(
        "--runtime-role",
        choices=("main", "canary", "vision"),
        default=None,
        help="Enable the signed process-local runtime role",
    )
    parser.add_argument(
        "--media-url-origin",
        action="append",
        default=[],
        help="Exact HTTP(S) origin allowed for canonical media fetches",
    )
    return parser


def create_app(  # noqa: PLR0915
    *,
    responses_runtime=None,
    responses_shutdown_timeout_s: float = 30.0,
    worker_count: int | None = None,
):
    """Create and configure the FastAPI application.

    This is called lazily to avoid slow imports when just showing --help.
    """
    # Direct ``uvicorn module:app`` launches bypass ``start()``. Freeze source
    # identity during app construction so no launch path shells out per request.
    get_runtime_provenance()

    from contextlib import asynccontextmanager

    from fastapi import FastAPI
    from fastapi.exceptions import RequestValidationError
    from fastapi.middleware.cors import CORSMiddleware

    from .core.config import get_settings
    from .middleware.logging import RequestResponseLoggingMiddleware
    from .routers import build_api_router

    if responses_shutdown_timeout_s <= 0:
        raise ValueError("responses_shutdown_timeout_s must be positive")

    settings = get_settings()
    configured_worker_count = _configured_worker_count(worker_count)
    responses_route_runtime = None
    runtime_responses_router = None
    role_control_service = None
    runtime_control_router = None
    if responses_runtime is not None:
        from .responses.runtime_control import (
            RoleControlService,
            build_role_control_router,
        )
        from .responses.runtime_router import build_runtime_responses_router

        responses_route_runtime = _resolve_responses_route_runtime(responses_runtime)
        if (
            getattr(responses_route_runtime, "requires_single_worker", False)
            and configured_worker_count != 1
        ):
            raise RuntimeError(
                "canonical Responses runtime requires exactly one worker"
            )
        runtime_responses_router = build_runtime_responses_router(
            responses_route_runtime
        )
        role_control_service = RoleControlService(responses_route_runtime)
        runtime_control_router = build_role_control_router(role_control_service)
    api_router = build_api_router(
        settings,
        responses_router=runtime_responses_router,
        runtime_control_router=runtime_control_router,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager for startup/shutdown hooks."""
        legacy_runtime = responses_runtime is None
        session_started = False
        if legacy_runtime:
            from .runtime_recycle import start_idle_process_recycler

            await start_idle_process_recycler()
        try:
            # Startup: opt-in session auth manager
            if settings.session_auth_enabled:
                os.environ.setdefault("SESSION_PROVIDER", settings.session_provider)
                os.environ.setdefault("SESSION_REDIS_URL", settings.redis_url)
                os.environ.setdefault(
                    "SESSION_DEFAULT_TTL_HOURS", str(settings.session_ttl_hours)
                )
                from .auth.session import session_auth

                await session_auth.start()
                session_started = True
            if role_control_service is not None:
                await role_control_service.start_pinned_role()
            yield
        finally:
            try:
                if responses_runtime is not None:
                    await responses_runtime.shutdown(
                        deadline_s=responses_shutdown_timeout_s
                    )
            finally:
                if session_started:
                    try:
                        from .auth.session import session_auth

                        await session_auth.stop()
                    # Session shutdown is best-effort while the app is exiting.
                    except Exception:  # nosec B110
                        pass
                if legacy_runtime:
                    from .runtime_recycle import stop_idle_process_recycler

                    await stop_idle_process_recycler()
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
                        from .videos.video_runtime import shutdown_video_runtime

                        await shutdown_video_runtime()
                        from .aux_runtime import shutdown_aux_runtime_manager

                        await asyncio.to_thread(shutdown_aux_runtime_manager)

    application = FastAPI(title="MLX Batch Server", lifespan=lifespan)
    application.state.responses_runtime = responses_runtime
    application.state.responses_route_runtime = responses_route_runtime
    application.state.role_control_service = role_control_service
    application.state.worker_count = configured_worker_count
    application.add_exception_handler(
        RequestValidationError,
        _responses_request_validation_exception_handler,
    )

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

    # This boundary is intentionally added last so Starlette places it outside
    # generic logging/rate-limit middleware that may inspect the JSON body.
    application.middleware("http")(_responses_body_exception_boundary)

    return application


def _resolve_responses_route_runtime(responses_runtime):
    """Return the one route graph owned by a direct or fused receipt."""

    from .responses.runtime_router import ResponsesRouteRuntime

    route_runtime = getattr(responses_runtime, "responses", responses_runtime)
    if not isinstance(route_runtime, ResponsesRouteRuntime):
        raise TypeError(
            "responses_runtime must expose one controller/registry route graph"
        )
    if not callable(getattr(responses_runtime, "shutdown", None)):
        raise TypeError("responses_runtime must expose shutdown(deadline_s=...)")
    return route_runtime


def _configured_worker_count(worker_count: int | None) -> int:
    raw_count = (
        os.environ.get("MLX_BATCH_WORKERS", "1")
        if worker_count is None
        else worker_count
    )
    if isinstance(raw_count, bool):
        raise TypeError("worker_count must be an integer")
    try:
        value = int(raw_count)
    except (TypeError, ValueError) as exc:
        raise ValueError("worker_count must be an integer") from exc
    if value < 1:
        raise ValueError("worker_count must be positive")
    return value


def _compose_process_runtime(
    *,
    runtime_role: str | None,
    port: int,
    media_url_origins: tuple[str, ...] = (),
):
    """Build the explicit role graph or reject ambiguous production ports."""

    if runtime_role is None:
        if port in PRODUCTION_ROLE_PORTS:
            raise RuntimeError(
                f"production port {port} requires an explicit runtime role"
            )
        return None

    from .provenance import compose_source_build_receipt
    from .responses.runtime_bootstrap import compose_role_responses_runtime
    from .runtime.role_manifest import load_role_manifest, packaged_role_manifest_path

    manifest = load_role_manifest(packaged_role_manifest_path())
    spec = manifest.role_directory().resolve(runtime_role)
    if port != spec.port:
        raise RuntimeError(
            f"runtime role {spec.name.value!r} owns port {spec.port}, not {port}"
        )
    build_receipt = compose_source_build_receipt(
        role_manifest_sha256=manifest.role_manifest_sha256
    )
    return compose_role_responses_runtime(
        process_role=spec.name,
        role_manifest_path=manifest.source.path,
        allowed_url_origins=media_url_origins,
        build_receipt=build_receipt,
    )


# Lazy app instance for uvicorn
# This is only created when uvicorn imports the module, not during CLI --help
_app_instance = None


def _get_app():
    """Get or create the FastAPI app instance (for uvicorn)."""
    global _app_instance
    if _app_instance is None:
        runtime_role = os.environ.get("MLX_BATCH_RUNTIME_ROLE") or None
        raw_port = os.environ.get("MLX_BATCH_PORT")
        if runtime_role is not None and raw_port is None:
            raise RuntimeError("MLX_BATCH_PORT is required with MLX_BATCH_RUNTIME_ROLE")
        port = 10240 if raw_port is None else int(raw_port)
        origins = tuple(
            origin.strip()
            for origin in os.environ.get("MLX_BATCH_MEDIA_URL_ORIGINS", "").split(",")
            if origin.strip()
        )
        runtime = _compose_process_runtime(
            runtime_role=runtime_role,
            port=port,
            media_url_origins=origins,
        )
        _app_instance = create_app(responses_runtime=runtime)
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
    os.environ["MLX_BATCH_WORKERS"] = str(args.workers)
    os.environ["MLX_BATCH_PORT"] = str(args.port)
    if args.runtime_role is None:
        os.environ.pop("MLX_BATCH_RUNTIME_ROLE", None)
    else:
        os.environ["MLX_BATCH_RUNTIME_ROLE"] = args.runtime_role
    os.environ["MLX_BATCH_MEDIA_URL_ORIGINS"] = ",".join(args.media_url_origin)
    # Capture the source checkout before Uvicorn imports the app or forks
    # workers. The inherited environment is the provenance contract.
    stamp_runtime_environment()

    if args.runtime_role is not None and args.workers != 1:
        parser.error("canonical Responses runtime requires exactly one worker")
    responses_runtime = _compose_process_runtime(
        runtime_role=args.runtime_role,
        port=args.port,
        media_url_origins=tuple(args.media_url_origin),
    )
    application = (
        "mlx_batch_server.main:app"
        if responses_runtime is None
        else create_app(responses_runtime=responses_runtime, worker_count=args.workers)
    )

    # NOW import uvicorn and start (lazy import)
    import uvicorn

    from .utils.logger import logger, set_logger_level

    set_logger_level(logger, args.log_level)

    # Start server - uvicorn will import the app via __getattr__
    uvicorn.run(
        application,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        use_colors=True,
        workers=args.workers,
    )


if __name__ == "__main__":
    start()
