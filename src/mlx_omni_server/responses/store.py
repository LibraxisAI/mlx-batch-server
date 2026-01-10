"""
Persistent store for Responses API payloads and streaming events.

Provides in-memory storage with optional Redis backend support.
For production deployments, configure Redis via REDIS_URL environment variable.

Created by M&K (c)2026 The LibraxisAI Team
Co-Authored-By: Maciej (void@div0.space) & Klaudiusz (the1st@whoai.am)
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..utils.logger import logger

_RESPONSES_NAMESPACE = "responses"
_DEFAULT_RETENTION = 30 * 24 * 60 * 60  # 30 days
_RETENTION_SECONDS = int(os.getenv("RESPONSES_RETENTION_SECONDS", _DEFAULT_RETENTION))

# In-memory storage (used when Redis is not configured)
_memory_store: dict[str, dict[str, Any]] = {}
_event_memory: dict[str, dict[str, Any]] = {}

# Optional Redis support
_redis_client = None
try:
    import redis.asyncio as aioredis

    _REDIS_URL = os.getenv("REDIS_URL")
    if _REDIS_URL:
        _redis_client = aioredis.from_url(_REDIS_URL, decode_responses=True)
        logger.info(f"Responses store using Redis: {_REDIS_URL[:30]}...")
except ImportError:
    pass

if _redis_client is None:
    logger.info("Responses store using in-memory storage")


@dataclass(slots=True)
class StoredResponse:
    """Stored response record with metadata."""

    response_id: str
    api_key_hash: str
    created_at: str
    status: str
    model: str
    request: dict[str, Any]
    response: dict[str, Any]
    background: bool = False
    store_enabled: bool = True

    def to_json(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "response_id": self.response_id,
            "api_key_hash": self.api_key_hash,
            "created_at": self.created_at,
            "status": self.status,
            "model": self.model,
            "request": self.request,
            "response": self.response,
            "background": self.background,
            "store_enabled": self.store_enabled,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> StoredResponse:
        """Create from JSON dict."""
        return cls(
            response_id=payload.get("response_id", ""),
            api_key_hash=payload.get("api_key_hash", ""),
            created_at=payload.get("created_at", datetime.now(UTC).isoformat()),
            status=payload.get("status", "completed"),
            model=payload.get("model", "unknown"),
            request=payload.get("request", {}),
            response=payload.get("response", {}),
            background=payload.get("background", False),
            store_enabled=payload.get("store_enabled", True),
        )


def _hash_api_key(api_key: str | None) -> str:
    """Hash API key for storage partitioning."""
    if not api_key:
        return "anonymous"
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


async def store_response(
    *,
    response_id: str,
    api_key: str | None,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
    background: bool = False,
    store_enabled: bool = True,
    retention_seconds: int | None = None,
) -> None:
    """
    Store a response for later retrieval.

    Args:
        response_id: Unique response identifier
        api_key: API key for access control (hashed before storage)
        request_payload: Original request data
        response_payload: Response data
        background: Whether this is a background response
        store_enabled: Whether storage is enabled (can be disabled per-request)
        retention_seconds: TTL override (default: 30 days)
    """
    if not store_enabled:
        return

    now = datetime.now(UTC).isoformat()
    record = StoredResponse(
        response_id=response_id,
        api_key_hash=_hash_api_key(api_key),
        created_at=now,
        status=str(response_payload.get("status", "completed")),
        model=str(
            response_payload.get("model", request_payload.get("model", "unknown"))
        ),
        request=request_payload,
        response=response_payload,
        background=background,
        store_enabled=True,
    )

    ttl = retention_seconds or _RETENTION_SECONDS
    payload = record.to_json()

    if _redis_client is not None:
        try:
            key = f"{_RESPONSES_NAMESPACE}:{response_id}"
            await _redis_client.set(key, json.dumps(payload), ex=ttl)
            return
        except Exception as e:
            logger.warning(f"Redis store failed, falling back to memory: {e}")

    # In-memory fallback
    _memory_store[response_id] = payload


async def get_response(
    *,
    response_id: str,
    api_key: str | None,
) -> StoredResponse | None:
    """
    Retrieve a stored response.

    Args:
        response_id: Response identifier to look up
        api_key: API key for access control verification

    Returns:
        StoredResponse if found and authorized, None otherwise
    """
    payload = None

    if _redis_client is not None:
        try:
            key = f"{_RESPONSES_NAMESPACE}:{response_id}"
            raw = await _redis_client.get(key)
            if raw:
                payload = json.loads(raw)
        except Exception as e:
            logger.warning(f"Redis get failed: {e}")

    if payload is None:
        payload = _memory_store.get(response_id)

    if payload is None:
        return None

    record = StoredResponse.from_json(payload)

    # In development mode (no API key), skip partition check
    # Production deployments should enforce API key partitioning
    if (
        api_key
        and record.api_key_hash != _hash_api_key(api_key)
        and record.api_key_hash != "anonymous"
    ):
        return None  # Access denied - different API key

    return record


async def delete_response(*, response_id: str, api_key: str | None) -> bool:
    """
    Delete a stored response.

    Args:
        response_id: Response identifier to delete
        api_key: API key for access control verification

    Returns:
        True if deleted, False if not found

    Raises:
        PermissionError: If API key doesn't match
    """
    # First verify access
    stored = await get_response(response_id=response_id, api_key=api_key)
    if stored is None:
        return False

    removed = False

    if _redis_client is not None:
        try:
            key = f"{_RESPONSES_NAMESPACE}:{response_id}"
            removed = bool(await _redis_client.delete(key))
        except Exception as e:
            logger.warning(f"Redis delete failed: {e}")

    if response_id in _memory_store:
        del _memory_store[response_id]
        removed = True

    # Clean up event memory
    _event_memory.pop(response_id, None)

    return removed


async def init_event_log(
    response_id: str,
    api_key: str | None,
    retention_seconds: int | None = None,
) -> None:
    """
    Initialize event log for streaming response.

    Used to track SSE events for a response in progress.
    """
    owner_hash = _hash_api_key(api_key)
    _event_memory[response_id] = {"owner": owner_hash, "events": []}

    if _redis_client is not None:
        try:
            ttl = retention_seconds or _RETENTION_SECONDS
            owner_key = f"responses_owner:{response_id}"
            events_key = f"responses_events:{response_id}"
            await _redis_client.set(owner_key, owner_hash, ex=ttl)
            await _redis_client.delete(events_key)
        except Exception as e:
            logger.warning(f"Redis init_event_log failed: {e}")


async def append_event(
    response_id: str,
    api_key: str | None,
    event: dict[str, Any],
    retention_seconds: int | None = None,
) -> None:
    """
    Append an event to the response's event log.

    Used for tracking streaming events that can be replayed.
    """
    owner_hash = _hash_api_key(api_key)
    ttl = retention_seconds or _RETENTION_SECONDS

    # Verify ownership
    memory_entry = _event_memory.get(response_id)
    if memory_entry and memory_entry.get("owner") != owner_hash:
        raise PermissionError("API key does not own this response")

    if _redis_client is not None:
        try:
            owner_key = f"responses_owner:{response_id}"
            existing_owner = await _redis_client.get(owner_key)
            if existing_owner and existing_owner != owner_hash:
                raise PermissionError("API key does not own this response")

            events_key = f"responses_events:{response_id}"
            await _redis_client.rpush(events_key, json.dumps(event))
            await _redis_client.expire(events_key, ttl)
        except PermissionError:
            raise
        except Exception as e:
            logger.warning(f"Redis append_event failed: {e}")

    # Always update in-memory as well
    if response_id not in _event_memory:
        _event_memory[response_id] = {"owner": owner_hash, "events": []}
    _event_memory[response_id].setdefault("events", []).append(event)


async def get_events_since(
    response_id: str,
    api_key: str | None,
    starting_after: int | None = None,
) -> list[dict[str, Any]]:
    """
    Get events for a response, optionally filtered by sequence number.

    Args:
        response_id: Response identifier
        api_key: API key for access control
        starting_after: Only return events after this sequence number

    Returns:
        List of events
    """
    owner_hash = _hash_api_key(api_key)
    events: list[dict[str, Any]] = []

    if _redis_client is not None:
        try:
            owner_key = f"responses_owner:{response_id}"
            existing_owner = await _redis_client.get(owner_key)
            if existing_owner and existing_owner != owner_hash:
                raise PermissionError("API key does not own this response")

            events_key = f"responses_events:{response_id}"
            raw_events = await _redis_client.lrange(events_key, 0, -1)
            for raw in raw_events or []:
                try:
                    events.append(json.loads(raw))
                except (json.JSONDecodeError, TypeError):
                    # Skip malformed events - robustness over strictness
                    continue
        except PermissionError:
            raise
        except Exception as e:
            logger.warning(f"Redis get_events_since failed: {e}")

    if not events:
        memory_entry = _event_memory.get(response_id)
        if memory_entry:
            if memory_entry.get("owner") != owner_hash and owner_hash != "anonymous":
                raise PermissionError("API key does not own this response")
            events = list(memory_entry.get("events", []))

    if starting_after is not None:
        events = [
            evt for evt in events if evt.get("sequence_number", -1) > starting_after
        ]

    return events


def get_stored_count() -> int:
    """Get count of stored responses (for monitoring)."""
    return len(_memory_store)


def clear_memory_store() -> None:
    """Clear in-memory store (for testing)."""
    _memory_store.clear()
    _event_memory.clear()
