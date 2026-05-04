"""Session lifecycle router (`/auth/*`)."""

from __future__ import annotations

import hmac as _hmac
import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from ..core.config import get_settings
from .api_keys import validate_api_key as _validate_dynamic_key
from .hmac import verify_hmac_request
from .session import get_current_user, session_auth, verify_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    user_id: str = Field(
        ...,
        min_length=1,
        max_length=64,
        pattern=r"^[a-zA-Z0-9._:@-]+$",
    )
    api_key: str | None = Field(
        None,
        max_length=256,
        description="Deprecated. Prefer headers (x-api-key or Authorization: Bearer ...).",
    )
    user_tier: Literal["default", "premium", "unlimited"] = Field("default")
    metadata: dict[str, Any] | None = Field(None)


class LoginResponse(BaseModel):
    session_id: str
    user_id: str
    expires_at: str
    user_tier: str


class SessionInfoResponse(BaseModel):
    session_id: str
    user_id: str
    sandbox_id: str
    created_at: str
    expires_at: str
    last_accessed: str
    status: str
    custom_metadata: dict[str, Any]


async def _bootstrap_auth(request: Request, body_api_key: str | None) -> str:
    """Allow API key or HMAC bootstrap before any session exists."""
    if all(
        [
            request.headers.get("X-Client-ID"),
            request.headers.get("X-Timestamp"),
            request.headers.get("X-Signature"),
        ]
    ):
        await verify_hmac_request(request)
        return "hmac"

    settings = get_settings()
    candidates: list[str] = []
    for header_name in (
        settings.api_key_header,
        "X-MLX-API-KEY",
        "x-mlx-api-key",
        "x-api-key",
    ):
        val = request.headers.get(header_name)
        if val and val not in candidates:
            candidates.append(val)

    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        token = auth.split(" ", 1)[1].strip()
        if token and token not in candidates:
            candidates.append(token)

    if body_api_key and body_api_key not in candidates:
        candidates.append(body_api_key)

    for candidate in candidates:
        try:
            if await _validate_dynamic_key(candidate):
                return "api_key_dynamic"
        except Exception:
            pass
        if settings.api_key and _hmac.compare_digest(candidate, settings.api_key):
            return "api_key_static"

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
    )


@router.post("/login", response_model=LoginResponse)
async def login(payload: LoginRequest, http_request: Request) -> LoginResponse:
    settings = get_settings()
    if not settings.session_auth_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session auth is disabled",
        )
    await _bootstrap_auth(http_request, payload.api_key)
    try:
        session_id = await session_auth.create_session(
            user_id=payload.user_id,
            user_tier=payload.user_tier,
            metadata=payload.metadata,
        )
        info = await session_auth.get_session_info(session_id)
        if not info:
            raise RuntimeError("session_info_missing")
        return LoginResponse(
            session_id=session_id,
            user_id=payload.user_id,
            expires_at=info["expires_at"],
            user_tier=payload.user_tier,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Login failed for user %s: %s", payload.user_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create session",
        ) from e


@router.post("/logout")
async def logout(
    session_info: dict[str, Any] = Depends(verify_session),
) -> dict[str, str]:
    session_id = session_info["session_id"]
    deleted = await session_auth.delete_session(session_id)
    if deleted:
        return {"message": "Session successfully terminated"}
    return {"message": "Session already expired or not found"}


@router.get("/session", response_model=SessionInfoResponse)
async def get_session(
    session_info: dict[str, Any] = Depends(verify_session),
) -> SessionInfoResponse:
    return SessionInfoResponse(**session_info)


@router.post("/session/extend")
async def extend_session(
    hours: int = Query(24, ge=1, le=168),
    session_info: dict[str, Any] = Depends(verify_session),
) -> dict[str, str]:
    session_id = session_info["session_id"]
    extended = await session_auth.extend_session(session_id, hours)
    if not extended:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to extend session",
        )
    new_info = await session_auth.get_session_info(session_id)
    return {
        "message": "Session extended successfully",
        "new_expires_at": new_info["expires_at"] if new_info else "",
    }


@router.get("/rate-limit")
async def get_rate_limit_status(
    user_id: str = Depends(get_current_user),
    session_info: dict[str, Any] = Depends(verify_session),
) -> dict[str, Any]:
    user_tier = session_info.get("custom_metadata", {}).get("user_tier", "default")
    config = session_auth.rate_limit_configs.get(
        user_tier, session_auth.rate_limit_configs["default"]
    )
    limits = session_auth.rate_limits.get(user_id)
    if limits is None:
        return {
            "user_id": user_id,
            "user_tier": user_tier,
            "limits": {
                "requests_per_minute": config.requests_per_minute,
                "requests_per_hour": config.requests_per_hour,
                "burst_size": config.burst_size,
            },
            "current_usage": {
                "minute_counter": 0,
                "hour_counter": 0,
                "minute_remaining": config.requests_per_minute,
                "hour_remaining": config.requests_per_hour,
            },
        }
    return {
        "user_id": user_id,
        "user_tier": user_tier,
        "limits": {
            "requests_per_minute": config.requests_per_minute,
            "requests_per_hour": config.requests_per_hour,
            "burst_size": config.burst_size,
        },
        "current_usage": {
            "minute_counter": limits.minute_counter,
            "hour_counter": limits.hour_counter,
            "minute_remaining": max(
                0, config.requests_per_minute - limits.minute_counter
            ),
            "hour_remaining": max(0, config.requests_per_hour - limits.hour_counter),
        },
    }
