"""API key issuance and validation.

Keys are stored as SHA-256 hashes (never plaintext). Backed by Redis when
configured and reachable, with an in-memory fallback for dev/single-process
deployments.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from ..core.config import get_settings

_redis_client: Any = None
_memory_store: dict[str, dict[str, Any]] = {}


async def _get_redis() -> Any:
    """Return a connected redis.asyncio client, or None if unavailable."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis.asyncio as redis
    except ImportError:
        return None
    settings = get_settings()
    try:
        client = redis.from_url(settings.redis_url, decode_responses=True)
        await client.ping()
    except Exception:
        return None
    _redis_client = client
    return _redis_client


def _now_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _storage_key(key_hash: str) -> str:
    return f"mlx_batch:api_keys:{key_hash}"


async def issue_api_key(
    subject: str,
    ttl_hours: int = 168,
    scopes: list[str] | None = None,
) -> dict[str, Any]:
    """Mint a new API key and persist its hash with TTL.

    Returns the full record (including the plaintext key — only time it is
    exposed). Storage holds only the hash + metadata.
    """
    api_key = f"mlx-{secrets.token_urlsafe(32)}"
    created_at = _now_iso()
    expires_at = (
        (datetime.now(UTC) + timedelta(hours=ttl_hours))
        .isoformat()
        .replace("+00:00", "Z")
    )

    record: dict[str, Any] = {
        "api_key": api_key,
        "subject": subject,
        "scopes": scopes or ["default"],
        "created_at": created_at,
        "expires_at": expires_at,
        "ttl_hours": ttl_hours,
    }
    storage_record = record.copy()
    storage_record.pop("api_key", None)
    storage_record["key_hash"] = _hash_api_key(api_key)

    client = await _get_redis()
    if client:
        await client.setex(
            _storage_key(storage_record["key_hash"]),
            int(ttl_hours * 3600),
            json.dumps(storage_record),
        )
    else:
        _memory_store[storage_record["key_hash"]] = storage_record
    return record


async def validate_api_key(api_key: str | None) -> bool:
    """Return True iff the supplied key exists and is not expired."""
    if not api_key:
        return False
    key_hash = _hash_api_key(api_key)
    client = await _get_redis()
    if client:
        data = await client.get(_storage_key(key_hash))
        if data is None:
            return False
        try:
            record = json.loads(data)
            stored_hash = record.get("key_hash")
            if stored_hash and not hmac.compare_digest(stored_hash, key_hash):
                return False
        except Exception:
            return False
        return True

    record = _memory_store.get(key_hash)
    if not record:
        return False
    stored_hash = record.get("key_hash")
    if stored_hash and not hmac.compare_digest(stored_hash, key_hash):
        return False
    try:
        exp = datetime.fromisoformat(record["expires_at"].replace("Z", "+00:00"))
        return exp > datetime.now(UTC)
    except Exception:
        return False


async def revoke_api_key(api_key: str) -> bool:
    """Revoke the key by removing it from both backends."""
    client = await _get_redis()
    key_hash = _hash_api_key(api_key)
    removed = False
    if client:
        removed = bool(await client.delete(_storage_key(key_hash)))
    removed_local = _memory_store.pop(key_hash, None) is not None
    return removed or removed_local


def _reset_for_tests() -> None:
    """Clear in-memory state and the cached redis client (test helper)."""
    global _redis_client
    _redis_client = None
    _memory_store.clear()
