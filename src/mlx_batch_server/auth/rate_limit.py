"""Global rate-limit middleware (Redis-backed with in-process fallback).

Stripped down from the api-router version: no Vista-product carve-outs, the
internal-trust header is renamed to ``x-mlx-internal``. Falls back to an
OrderedDict bucket if Redis is unreachable.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any

from fastapi import status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from ..core.config import get_settings

if TYPE_CHECKING:
    from starlette.requests import Request

logger = logging.getLogger(__name__)

_INTERNAL_REQUEST_HEADER = "x-mlx-internal"
_INTERNAL_OWNER_HEADER = "x-mlx-internal-owner-key"
_TRUSTED_INTERNAL_HOSTS = {"127.0.0.1", "::1", "localhost"}
_FALLBACK_MAX_CLIENTS = 50_000

_LUA_INCR_EXPIRE = (
    "local current = redis.call('INCR', KEYS[1]);"
    "if current == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]); end;"
    "return current;"
)


def hash_client_fingerprint(credential: str | None) -> str:
    normalized = (credential or "anonymous").strip() or "anonymous"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-credential token-bucket-ish rate limit middleware."""

    def __init__(
        self,
        app: Any,
        requests_per_minute: int = 60,
        window_size: int = 60,
        concurrent_limit: int = 10,
        exempt_paths: list[str] | None = None,
    ) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.window_size = window_size
        self.concurrent_limit = concurrent_limit
        self._fallback_counts: OrderedDict[str, list[float]] = OrderedDict()
        self._fallback_concurrent: dict[str, int] = {}
        self._warned_no_redis = False
        self._exempt_paths = set(exempt_paths or [])
        self._redis_client: Any = None

    async def _get_redis(self) -> Any:
        if self._redis_client is not None:
            return self._redis_client
        try:
            import redis.asyncio as redis
        except ImportError:
            return None
        try:
            client = redis.from_url(get_settings().redis_url, decode_responses=True)
            await client.ping()
        except Exception:
            return None
        self._redis_client = client
        return self._redis_client

    async def _increment(self, key: str) -> tuple[int, int]:
        client = await self._get_redis()
        if client is None:
            return self._increment_fallback(key)
        try:
            try:
                count = await client.eval(_LUA_INCR_EXPIRE, 1, key, self.window_size)
            except Exception:
                count = await client.incr(key)
                if count == 1:
                    await client.expire(key, self.window_size)
            try:
                ttl = await client.ttl(key)
            except Exception:
                ttl = self.window_size
        except Exception:
            return self._increment_fallback(key)
        try:
            count_int = int(count)
        except Exception:
            return self._increment_fallback(key)
        try:
            ttl_int = int(ttl)
        except Exception:
            ttl_int = self.window_size
        if ttl_int <= 0:
            ttl_int = self.window_size
        return count_int, ttl_int

    def _increment_fallback(self, key: str) -> tuple[int, int]:
        if not self._warned_no_redis:
            logger.warning(
                "RateLimitMiddleware running without Redis; using process-local limiter"
            )
            self._warned_no_redis = True
        now = time.time()
        bucket = self._fallback_counts.pop(key, [])
        bucket[:] = [ts for ts in bucket if now - ts < self.window_size]
        bucket.append(now)
        self._fallback_counts[key] = bucket
        self._prune_fallback_counts(now)
        ttl = int(self.window_size - (now - bucket[0])) if bucket else self.window_size
        return len(bucket), max(ttl, 1)

    def _prune_fallback_counts(self, now: float) -> None:
        if len(self._fallback_counts) <= _FALLBACK_MAX_CLIENTS:
            return
        stale = [
            k
            for k, b in self._fallback_counts.items()
            if not b or now - b[-1] >= self.window_size
        ]
        for k in stale:
            self._fallback_counts.pop(k, None)
        while len(self._fallback_counts) > _FALLBACK_MAX_CLIENTS:
            self._fallback_counts.popitem(last=False)

    def _is_internal_trust(self, request: Request) -> bool:
        if request.headers.get(_INTERNAL_REQUEST_HEADER, "").strip().lower() not in {
            "1",
            "true",
        }:
            return False
        owner = request.headers.get(_INTERNAL_OWNER_HEADER, "").strip()
        if not owner:
            return False
        host = getattr(getattr(request, "client", None), "host", None)
        return host in _TRUSTED_INTERNAL_HOSTS

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        if request.method == "OPTIONS":
            return await call_next(request)
        if request.url.path in self._exempt_paths:
            return await call_next(request)
        if self._is_internal_trust(request):
            return await call_next(request)

        settings = get_settings()
        token = request.headers.get(settings.api_key_header.lower())
        if not token:
            auth = request.headers.get("Authorization")
            if auth and auth.lower().startswith("bearer "):
                token = auth[7:]

        if token:
            tag = hash_client_fingerprint(token)
            key = f"mlx_batch:ratelimit:key:{tag}"
            concurrent_key = f"mlx_batch:concurrent:key:{tag}"
        else:
            client_host = getattr(getattr(request, "client", None), "host", "anon")
            key = f"mlx_batch:ratelimit:ip:{client_host}"
            concurrent_key = f"mlx_batch:concurrent:ip:{client_host}"

        # Concurrent guard
        client = await self._get_redis()
        is_redis_concurrent = False
        if client is not None:
            try:
                concurrent = await client.incr(concurrent_key)
                await client.expire(concurrent_key, 300)
                is_redis_concurrent = True
            except Exception:
                self._fallback_concurrent[concurrent_key] = (
                    self._fallback_concurrent.get(concurrent_key, 0) + 1
                )
                concurrent = self._fallback_concurrent[concurrent_key]
        else:
            self._fallback_concurrent[concurrent_key] = (
                self._fallback_concurrent.get(concurrent_key, 0) + 1
            )
            concurrent = self._fallback_concurrent[concurrent_key]

        if concurrent > self.concurrent_limit:
            if is_redis_concurrent:
                with contextlib.suppress(Exception):
                    await client.decr(concurrent_key)
            else:
                self._fallback_concurrent[concurrent_key] = max(
                    0, self._fallback_concurrent[concurrent_key] - 1
                )
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Too Many Requests",
                    "message": (
                        f"Concurrent request limit exceeded. "
                        f"Maximum {self.concurrent_limit} concurrent requests."
                    ),
                },
                headers={"Retry-After": "5"},
            )

        try:
            count, ttl = await self._increment(key)
            remaining = max(self.requests_per_minute - count, 0)
            reset = int(time.time() + ttl)
            if count > self.requests_per_minute:
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Too Many Requests",
                        "message": (
                            f"Rate limit exceeded. Maximum "
                            f"{self.requests_per_minute} requests per minute."
                        ),
                        "retry_after": ttl,
                    },
                    headers={
                        "Retry-After": str(ttl),
                        "X-RateLimit-Limit": str(self.requests_per_minute),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(reset),
                    },
                )
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(reset)
            return response
        finally:
            if is_redis_concurrent:
                with contextlib.suppress(Exception):
                    await client.decr(concurrent_key)
            else:
                self._fallback_concurrent[concurrent_key] = max(
                    0, self._fallback_concurrent.get(concurrent_key, 1) - 1
                )
