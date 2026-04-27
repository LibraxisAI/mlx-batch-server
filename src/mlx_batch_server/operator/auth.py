"""Conditional auth dependency for the standalone operator backend.

The operator runs as a sibling FastAPI app on a separate port (default 10241).
It reuses the inference server's auth primitives (HMAC, session token, API
key) but gates them behind the operator's own ``security_level`` /
``require_auth`` settings so the operator can stay open in dev while
inference is locked down — or vice versa.

When auth is *not* enforced the dependency returns ``None`` and acts as a
no-op, keeping the operator's open-by-default ergonomics intact.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

from mlx_batch_server.auth.dependency import (
    _resolve_api_key_auth,
    _resolve_session_auth,
)
from mlx_batch_server.auth.hmac import verify_hmac_request
from mlx_batch_server.core.config import get_settings as get_core_settings
from mlx_batch_server.operator.config import get_settings as get_operator_settings

logger = logging.getLogger(__name__)

# Re-export the same security schemes used by the core dependency so the
# OpenAPI docs render properly and the headers/bearer tokens are extracted
# automatically by FastAPI when the operator app is mounted standalone.
_api_key_scheme = APIKeyHeader(name="x-api-key", auto_error=False)
_bearer_scheme = HTTPBearer(auto_error=False)


def _operator_normalized_level() -> int:
    """Effective operator auth level, with the documented level-1 promotion."""
    op = get_operator_settings()
    if op.security_level == 0 and op.require_auth:
        return 2
    return 2 if op.security_level == 1 else int(op.security_level)


async def operator_auth(
    request: Request,
    api_key: str | None = Security(_api_key_scheme),
    bearer_creds: HTTPAuthorizationCredentials | None = Security(_bearer_scheme),
) -> dict[str, Any] | None:
    """Conditional auth wrapper used by every protected operator route.

    Returns ``None`` when auth is not enforced (open ergonomics preserved).
    Otherwise validates HMAC > session > API key using the inference auth
    primitives and returns the resolved auth context dict.
    """
    op_settings = get_operator_settings()
    if not op_settings.auth_enforced:
        return None

    level = _operator_normalized_level()
    core_settings = get_core_settings()

    # 1) HMAC (disabled on level 3)
    if level != 3 and all(
        request.headers.get(name)
        for name in ("X-Client-ID", "X-Timestamp", "X-Signature")
    ):
        try:
            return await verify_hmac_request(request)
        except HTTPException as e:
            logger.warning("operator HMAC auth failed: %s", e.detail)
            raise

    # 2) Session bearer token
    if core_settings.session_auth_enabled and bearer_creds:
        info = await _resolve_session_auth(bearer_creds.credentials)
        if info is not None:
            info.setdefault("auth_method", "session")
            return info
        if level == 3:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired session",
                headers={"WWW-Authenticate": "Bearer"},
            )

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
    for header_name in (
        core_settings.api_key_header,
        "X-MLX-API-KEY",
        "x-mlx-api-key",
        "x-api-key",
    ):
        val = request.headers.get(header_name)
        if val and val not in candidates:
            candidates.append(val)
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
        logger.warning("operator: invalid API key attempted")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer, ApiKey"},
        )

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required. Provide either API key or session token.",
        headers={"WWW-Authenticate": "Bearer, ApiKey"},
    )
