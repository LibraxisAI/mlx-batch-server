"""Unified ``verify_auth`` FastAPI dependency.

Implements the four security levels:

- 0 (default): bypass — every request maps to a stable pseudo-owner
- 1: deprecated, internally promoted to level 2 with a warning log
- 2: HMAC, session, or API key (any one of them)
- 3: session token only

Returns a dict shaped as ``{user_id, session_id, auth_method, resolved_api_key?}``
to give downstream code a single, consistent surface.
"""

from __future__ import annotations

import hmac as _hmac
import logging
from typing import Any

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from ..core.config import get_settings
from .api_keys import validate_api_key as _validate_dynamic_key
from .hmac import verify_hmac_request

logger = logging.getLogger(__name__)


def _api_key_security() -> APIKeyHeader:
    return APIKeyHeader(name=get_settings().api_key_header, auto_error=False)


api_key_header = _api_key_security()
session_scheme = HTTPBearer(auto_error=False)


def _normalized_security_level(level: int | None = None) -> int:
    """Levels 0/2/3 are valid; level 1 is silently promoted to 2."""
    settings = get_settings()
    resolved = int(level if level is not None else (settings.security_level or 0))
    return 2 if resolved == 1 else resolved


def is_auth_required(level: int | None = None) -> bool:
    """True when the configured level requires credentials at all."""
    settings = get_settings()
    normalized = _normalized_security_level(level)
    if normalized == 0:
        return False
    if normalized == 3:
        return True
    return bool(settings.api_key or settings.session_auth_enabled)


def build_open_auth_owner(client_host: str | None) -> str:
    """Stable pseudo-owner string used in level 0 mode for response storage."""
    host = (client_host or "127.0.0.1").strip()
    safe = (
        "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in host)
        or "127.0.0.1"
    )
    return f"open-{safe}"


async def _resolve_api_key_auth(candidate: str | None) -> tuple[str, str] | None:
    if not candidate:
        return None
    settings = get_settings()
    try:
        if await _validate_dynamic_key(candidate):
            return ("api_key_dynamic", candidate)
    except Exception:
        pass
    if settings.api_key and _hmac.compare_digest(candidate, settings.api_key):
        return ("api_key_static", candidate)
    return None


async def _resolve_session_auth(
    session_id: str | None, *, enforce_rate_limit: bool = True
) -> dict[str, Any] | None:
    settings = get_settings()
    if not settings.session_auth_enabled or not session_id:
        return None
    from .session import session_auth

    info = await session_auth.validate_session(session_id)
    if not info:
        return None
    if not enforce_rate_limit:
        return info
    user_id = info.get("user_id", "unknown")
    tier = info.get("custom_metadata", {}).get("user_tier", "default")
    if not await session_auth.check_rate_limit(user_id, tier):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={"Retry-After": "60"},
        )
    return info


async def verify_auth(
    request: Request,
    api_key: str | None = Security(api_key_header),
    bearer_creds: HTTPAuthorizationCredentials | None = Security(session_scheme),
) -> dict[str, Any]:
    """Unified auth dependency: HMAC > session > API key."""
    settings = get_settings()
    raw_level = int(settings.security_level or 0)
    level = _normalized_security_level(raw_level)

    if level == 0:
        client_host = (
            getattr(getattr(request, "client", None), "host", "127.0.0.1")
            if request
            else "127.0.0.1"
        )
        return {
            "user_id": f"bypass:{client_host}",
            "session_id": None,
            "auth_method": "bypass",
            "resolved_api_key": build_open_auth_owner(client_host),
        }

    if raw_level == 1:
        logger.warning(
            "security_level=1 is deprecated; enforcing validated API keys only"
        )

    # 1) HMAC (disabled on level 3)
    if (
        all(
            [
                request.headers.get("X-Client-ID"),
                request.headers.get("X-Timestamp"),
                request.headers.get("X-Signature"),
            ]
        )
        and level != 3
    ):
        try:
            return await verify_hmac_request(request)
        except HTTPException as e:
            logger.warning("HMAC auth failed: %s", e.detail)
            raise

    # 2) Session
    if settings.session_auth_enabled and bearer_creds:
        try:
            session_id = bearer_creds.credentials
            info = await _resolve_session_auth(session_id)
            if info is not None:
                info.setdefault("auth_method", "session")
                return info
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session",
            )
        except HTTPException:
            if level == 3:
                raise

    if level == 3:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Valid session token missing or expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 3) API key (header alias-friendly + Bearer fallback)
    candidates: list[str] = []
    if api_key:
        candidates.append(api_key)
    try:
        if request is not None:
            for header_name in (
                settings.api_key_header,
                "X-MLX-API-KEY",
                "x-mlx-api-key",
                "x-api-key",
            ):
                val = request.headers.get(header_name)
                if val and val not in candidates:
                    candidates.append(val)
    except Exception:
        pass
    if bearer_creds and bearer_creds.credentials not in candidates:
        candidates.append(bearer_creds.credentials)

    for candidate in candidates:
        resolved = await _resolve_api_key_auth(candidate)
        if resolved:
            method, key = resolved
            return {
                "user_id": "api_key_user",
                "session_id": None,
                "auth_method": method,
                "resolved_api_key": key,
            }

    if candidates:
        logger.warning("Invalid API key attempted (multiple candidates tested)")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer, ApiKey"},
        )

    if settings.api_key or settings.session_auth_enabled:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Provide either API key or session token.",
            headers={"WWW-Authenticate": "Bearer, ApiKey"},
        )

    return {"user_id": "anonymous", "session_id": None, "auth_method": "none"}


async def verify_api_key(
    request: Request | None = None,
    api_key: str | None = Security(api_key_header),
    bearer_creds: HTTPAuthorizationCredentials | None = Security(session_scheme),
) -> str | None:
    """Backward-compatible wrapper: returns a key/owner string for downstream."""
    auth_info = await verify_auth(request, api_key, bearer_creds)  # type: ignore[arg-type]
    if resolved := auth_info.get("resolved_api_key"):
        return resolved
    if session_id := auth_info.get("session_id"):
        return session_id
    if auth_info.get("auth_method") == "none":
        client_host = (
            getattr(getattr(request, "client", None), "host", None)
            if request is not None
            else None
        )
        return build_open_auth_owner(client_host)
    return api_key
