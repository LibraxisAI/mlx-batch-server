from fastapi import APIRouter

from .core.config import Settings, get_settings


def _auth_required(settings: Settings) -> bool:
    """True when any auth-related feature is opt-in."""
    return bool(
        int(settings.security_level or 0) > 0
        or settings.session_auth_enabled
        or settings.api_key
        or settings.access_registration_secret
    )


def build_api_router(
    settings: Settings | None = None,
    *,
    responses_router: APIRouter | None = None,
    runtime_control_router: APIRouter | None = None,
) -> APIRouter:
    """Compose the application API router.

    Auth, /access and /hmac surfaces are mounted only when at least one
    auth-related env var is configured. Called from ``main.create_app`` so the
    decision is taken against *current* settings at app build time.
    """
    settings = settings or get_settings()
    router = APIRouter()
    from .chat.anthropic import router as anthropic_router

    canonical_runtime = runtime_control_router is not None
    if canonical_runtime != (responses_router is not None):
        raise ValueError(
            "canonical Responses and runtime-control routers must be mounted together"
        )

    if not canonical_runtime:
        from .batch import router as batch_router
        from .chat.openai import router as chat_router
        from .chat.openai.models import models
        from .embeddings import router as embeddings_router
        from .embeddings import visual_router as visual_embeddings_router
        from .images import images
        from .stt import stt as stt_router
        from .tts import tts as tts_router
        from .videos import router as videos_router

        router.include_router(stt_router.router)
        router.include_router(tts_router.router)
        router.include_router(models.router)
        router.include_router(images.router)
        router.include_router(videos_router)
        router.include_router(chat_router.router)
        router.include_router(embeddings_router.router)
        router.include_router(visual_embeddings_router.router)
        from .responses.router import router as legacy_responses_router

        router.include_router(legacy_responses_router)
        router.include_router(batch_router.router)
    else:
        assert runtime_control_router is not None
        assert responses_router is not None
        router.include_router(runtime_control_router)
        router.include_router(responses_router)

    router.include_router(anthropic_router.router, prefix="/anthropic")

    from .admin.router import router as admin_router
    from .health import ready_router

    router.include_router(admin_router)
    router.include_router(ready_router)

    if _auth_required(settings):
        from .auth.router_access import router as access_router
        from .auth.router_hmac import router as hmac_router
        from .auth.router_session import router as auth_session_router

        router.include_router(auth_session_router)
        router.include_router(hmac_router)
        router.include_router(access_router)
    return router


def __getattr__(name: str) -> APIRouter:
    """Build the backwards-compatible router only when explicitly requested."""

    if name != "api_router":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    router = build_api_router()
    globals()[name] = router
    return router


__all__ = ["api_router", "build_api_router"]  # noqa: F822
