"""Session-based authentication for mlx-batch-server.

Self-contained session manager: no chuk_sessions dependency. Two backends
selected by `SESSION_PROVIDER`:

- "memory" (default): process-local dict, ideal for single-process / dev
- "redis": shared session store across processes/hosts

Sessions carry rate-limit tier metadata. Per-user rate limiting is enforced
in-process to avoid extra round-trips during the auth path.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ..core.config import get_settings

logger = logging.getLogger(__name__)

session_scheme = HTTPBearer(auto_error=False)


@dataclass
class RateLimitConfig:
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    burst_size: int = 10


@dataclass
class UserRateLimit:
    minute_counter: int = 0
    hour_counter: int = 0
    minute_reset: float = field(default_factory=time.time)
    hour_reset: float = field(default_factory=time.time)
    last_request: float = field(default_factory=time.time)


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def _now() -> datetime:
    return datetime.now(UTC)


class _MemorySessionStore:
    """Process-local session store keyed by session_id."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def put(self, session_id: str, data: dict[str, Any]) -> None:
        async with self._lock:
            self._sessions[session_id] = data

    async def get(self, session_id: str) -> dict[str, Any] | None:
        async with self._lock:
            return self._sessions.get(session_id)

    async def delete(self, session_id: str) -> bool:
        async with self._lock:
            return self._sessions.pop(session_id, None) is not None

    async def cleanup_expired(self) -> int:
        now = _now()
        async with self._lock:
            stale = [
                sid
                for sid, data in self._sessions.items()
                if datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
                < now
            ]
            for sid in stale:
                self._sessions.pop(sid, None)
        return len(stale)


class _RedisSessionStore:
    """Redis-backed session store. Keys: ``mlx_batch:session:{id}`` with TTL."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @staticmethod
    def _key(session_id: str) -> str:
        return f"mlx_batch:session:{session_id}"

    async def put(self, session_id: str, data: dict[str, Any]) -> None:
        try:
            exp = datetime.fromisoformat(data["expires_at"].replace("Z", "+00:00"))
            ttl = max(int((exp - _now()).total_seconds()), 1)
        except Exception:
            ttl = int(get_settings().session_ttl_hours * 3600)
        await self._client.setex(self._key(session_id), ttl, json.dumps(data))

    async def get(self, session_id: str) -> dict[str, Any] | None:
        raw = await self._client.get(self._key(session_id))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    async def delete(self, session_id: str) -> bool:
        return bool(await self._client.delete(self._key(session_id)))

    async def cleanup_expired(self) -> int:
        # Redis handles TTL eviction natively.
        return 0


class SessionAuthManager:
    """Owns the session store and per-user rate-limit counters."""

    def __init__(self) -> None:
        self._store: _MemorySessionStore | _RedisSessionStore | None = None
        self.rate_limits: dict[str, UserRateLimit] = defaultdict(UserRateLimit)
        self.rate_limit_configs: dict[str, RateLimitConfig] = {
            "default": RateLimitConfig(),
            "premium": RateLimitConfig(
                requests_per_minute=120, requests_per_hour=5000, burst_size=20
            ),
            "unlimited": RateLimitConfig(
                requests_per_minute=999_999,
                requests_per_hour=999_999,
                burst_size=999_999,
            ),
        }
        self._cleanup_task: asyncio.Task[Any] | None = None

    async def _ensure_store(self) -> None:
        if self._store is not None:
            return
        settings = get_settings()
        if settings.session_provider == "redis":
            try:
                import redis.asyncio as redis
            except ImportError as exc:
                raise RuntimeError(
                    "session_provider=redis but the 'redis' package is not installed"
                ) from exc
            client = redis.from_url(settings.redis_url, decode_responses=True)
            await client.ping()
            self._store = _RedisSessionStore(client)
        else:
            self._store = _MemorySessionStore()

    async def start(self) -> None:
        await self._ensure_store()
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self) -> None:
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(300)
                if self._store is not None:
                    await self._store.cleanup_expired()
                now = time.time()
                stale_users = [
                    user_id
                    for user_id, limits in self.rate_limits.items()
                    if now - limits.last_request > 3600
                ]
                for user_id in stale_users:
                    del self.rate_limits[user_id]
                if stale_users:
                    logger.info(
                        "Cleaned up rate limits for %d inactive users",
                        len(stale_users),
                    )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Session cleanup loop error: %s", e)

    async def create_session(
        self,
        user_id: str,
        user_tier: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        await self._ensure_store()
        assert self._store is not None
        ttl_hours = int(get_settings().session_ttl_hours or 24)
        session_id = secrets.token_urlsafe(32)
        now = _now()
        custom_metadata = dict(metadata or {})
        custom_metadata.setdefault("user_tier", user_tier)
        custom_metadata.setdefault("created_via", "mlx-batch-server")
        data = {
            "session_id": session_id,
            "user_id": user_id,
            "sandbox_id": "default",
            "status": "active",
            "created_at": _iso(now),
            "expires_at": _iso(now + timedelta(hours=ttl_hours)),
            "last_accessed": _iso(now),
            "custom_metadata": custom_metadata,
        }
        await self._store.put(session_id, data)
        logger.info(
            "Created session %s for user %s (tier=%s)",
            session_id[:12],
            user_id,
            user_tier,
        )
        return session_id

    async def validate_session(self, session_id: str) -> dict[str, Any] | None:
        await self._ensure_store()
        assert self._store is not None
        info = await self._store.get(session_id)
        if not info:
            return None
        try:
            exp = datetime.fromisoformat(info["expires_at"].replace("Z", "+00:00"))
        except Exception:
            return None
        if exp < _now():
            await self._store.delete(session_id)
            return None
        info["last_accessed"] = _iso(_now())
        await self._store.put(session_id, info)
        return info

    async def get_session_info(self, session_id: str) -> dict[str, Any] | None:
        await self._ensure_store()
        assert self._store is not None
        return await self._store.get(session_id)

    async def extend_session(self, session_id: str, hours: int = 24) -> bool:
        await self._ensure_store()
        assert self._store is not None
        info = await self._store.get(session_id)
        if not info:
            return False
        new_expiry = _now() + timedelta(hours=hours)
        info["expires_at"] = _iso(new_expiry)
        info["last_accessed"] = _iso(_now())
        await self._store.put(session_id, info)
        return True

    async def delete_session(self, session_id: str) -> bool:
        await self._ensure_store()
        assert self._store is not None
        return await self._store.delete(session_id)

    async def check_rate_limit(self, user_id: str, user_tier: str = "default") -> bool:
        config = self.rate_limit_configs.get(
            user_tier, self.rate_limit_configs["default"]
        )
        limits = self.rate_limits[user_id]
        now = time.time()
        if now - limits.minute_reset > 60:
            limits.minute_counter = 0
            limits.minute_reset = now
        if now - limits.hour_reset > 3600:
            limits.hour_counter = 0
            limits.hour_reset = now
        if limits.minute_counter >= config.requests_per_minute:
            logger.warning("User %s exceeded minute rate limit", user_id)
            return False
        if limits.hour_counter >= config.requests_per_hour:
            logger.warning("User %s exceeded hour rate limit", user_id)
            return False
        limits.minute_counter += 1
        limits.hour_counter += 1
        limits.last_request = now
        return True


# Module-level singleton; the store is created lazily on first start()/use
session_auth = SessionAuthManager()


async def verify_session(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Security(session_scheme),
) -> dict[str, Any]:
    """Verify the session token, enforce per-user rate limits."""
    settings = get_settings()
    if not settings.session_auth_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Session auth disabled",
        )
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    info = await session_auth.validate_session(credentials.credentials)
    if not info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_id = info.get("user_id", "unknown")
    tier = info.get("custom_metadata", {}).get("user_tier", "default")
    if not await session_auth.check_rate_limit(user_id, tier):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers={
                "Retry-After": "60",
                "X-RateLimit-Limit": str(
                    session_auth.rate_limit_configs[tier].requests_per_minute
                ),
                "X-RateLimit-Remaining": "0",
            },
        )
    return info


async def get_current_user(
    session_info: dict[str, Any] = Depends(verify_session),
) -> str:
    return session_info.get("user_id", "unknown")


def _reset_for_tests() -> None:
    """Tear down singleton state for use in test fixtures."""
    global session_auth
    session_auth = SessionAuthManager()
