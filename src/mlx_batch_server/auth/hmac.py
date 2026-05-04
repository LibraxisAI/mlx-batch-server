"""HMAC-SHA256 request signing for trusted mobile/headless clients.

Secrets are persisted in Redis when reachable AND mirrored to a JSON file
under XDG-compliant data home. The file is the resilient fallback for local
deployments and single-host installs.

Canonical message:
    HMAC-SHA256(secret, f"{timestamp}:{METHOD}:{path}:{body_sha256}")
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac as _hmac
import json
import logging
import os
import secrets
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request, status

from ..core.config import get_settings

logger = logging.getLogger(__name__)

_secrets_lock = threading.Lock()
_redis_client: Any = None


def _resolve_secrets_file() -> Path:
    """Resolve the on-disk HMAC secrets file path (XDG-compliant)."""
    settings = get_settings()
    explicit = settings.mlx_batch_hmac_secrets_file or os.environ.get(
        "MLX_BATCH_HMAC_SECRETS_FILE"
    )
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(xdg) / "mlx-batch-server" / "hmac_secrets.json"


async def _get_redis() -> Any:
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


def _redis_key(client_id: str) -> str:
    return f"mlx_batch:hmac:{client_id}"


def _load_file_secrets() -> dict[str, str]:
    secrets_file = _resolve_secrets_file()
    with _secrets_lock:
        if not secrets_file.exists():
            return {}
        try:
            with secrets_file.open() as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load HMAC secrets file %s: %s", secrets_file, e)
            return {}


def _save_file_secrets(payload: dict[str, str]) -> None:
    secrets_file = _resolve_secrets_file()
    with _secrets_lock:
        try:
            secrets_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = secrets_file.with_suffix(".tmp")
            with tmp.open("w") as f:
                json.dump(payload, f, indent=2)
            tmp.replace(secrets_file)
        except OSError as e:
            logger.error("Failed to save HMAC secrets file %s: %s", secrets_file, e)
            raise


async def _read_secret(client_id: str) -> str | None:
    """Read secret with Redis-first fallback to file."""
    client = await _get_redis()
    if client:
        try:
            value = await client.get(_redis_key(client_id))
            if value:
                return value
        except Exception:
            pass
    return _load_file_secrets().get(client_id)


async def _write_secret(client_id: str, secret_key: str) -> None:
    """Write-through: Redis (best effort) AND file (always)."""
    client = await _get_redis()
    if client:
        with contextlib.suppress(Exception):
            await client.set(_redis_key(client_id), secret_key)
    file_secrets = _load_file_secrets()
    file_secrets[client_id] = secret_key
    _save_file_secrets(file_secrets)


async def _delete_secret(client_id: str) -> bool:
    """Delete from both backends. True if at least one removed an entry."""
    removed = False
    client = await _get_redis()
    if client:
        with contextlib.suppress(Exception):
            removed = bool(await client.delete(_redis_key(client_id)))
    file_secrets = _load_file_secrets()
    if client_id in file_secrets:
        del file_secrets[client_id]
        _save_file_secrets(file_secrets)
        removed = True
    return removed


async def _list_secret_ids() -> list[str]:
    ids = set(_load_file_secrets().keys())
    client = await _get_redis()
    if client:
        try:
            cursor = 0
            prefix = "mlx_batch:hmac:"
            while True:
                cursor, keys = await client.scan(cursor=cursor, match=f"{prefix}*")
                for k in keys:
                    ids.add(k.removeprefix(prefix))
                if cursor == 0:
                    break
        except Exception:
            pass
    return sorted(ids)


def generate_hmac_secret() -> str:
    """Generate a 256-bit hex secret."""
    return secrets.token_hex(32)


async def register_hmac_client(client_id: str, secret_key: str | None = None) -> str:
    """Register or rotate a client's HMAC secret. Returns the secret."""
    if not secret_key:
        secret_key = generate_hmac_secret()
    await _write_secret(client_id, secret_key)
    logger.info("Registered HMAC client: %s", client_id)
    return secret_key


async def revoke_hmac_client(client_id: str) -> bool:
    removed = await _delete_secret(client_id)
    if removed:
        logger.info("Revoked HMAC client: %s", client_id)
    return removed


def compute_signature(
    secret_key: str, timestamp: int, method: str, path: str, body_hash: str
) -> str:
    message = f"{timestamp}:{method.upper()}:{path}:{body_hash}"
    return _hmac.new(secret_key.encode(), message.encode(), hashlib.sha256).hexdigest()


async def verify_hmac_signature(
    client_id: str,
    timestamp: int,
    signature: str,
    method: str,
    path: str,
    body: bytes = b"",
) -> bool:
    """Verify a request signature. Raises HTTPException on failure."""
    secret_key = await _read_secret(client_id)
    if not secret_key:
        logger.warning("Unknown HMAC client: %s", client_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Unknown client ID"
        )

    tolerance = int(get_settings().hmac_timestamp_tolerance or 300)
    time_diff = abs(int(time.time()) - timestamp)
    if time_diff > tolerance:
        logger.warning(
            "HMAC timestamp out of window: client=%s diff=%ss", client_id, time_diff
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Request timestamp outside {tolerance}s window",
        )

    body_hash = hashlib.sha256(body).hexdigest() if body else ""
    expected = compute_signature(secret_key, timestamp, method, path, body_hash)
    if not _hmac.compare_digest(signature, expected):
        logger.warning("Invalid HMAC signature for client: %s", client_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid signature"
        )
    return True


async def verify_hmac_request(request: Request) -> dict[str, Any]:
    """FastAPI dependency: verify an HMAC-signed request."""
    client_id = request.headers.get("X-Client-ID")
    timestamp_str = request.headers.get("X-Timestamp")
    signature = request.headers.get("X-Signature")

    missing = []
    if not client_id:
        missing.append("X-Client-ID")
    if not timestamp_str:
        missing.append("X-Timestamp")
    if not signature:
        missing.append("X-Signature")
    if missing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required HMAC headers: {', '.join(missing)}",
        )

    try:
        timestamp = int(timestamp_str)  # type: ignore[arg-type]
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Timestamp must be Unix timestamp integer",
        ) from exc

    body = await request.body()
    await verify_hmac_signature(
        client_id=client_id,  # type: ignore[arg-type]
        timestamp=timestamp,
        signature=signature,  # type: ignore[arg-type]
        method=request.method,
        path=request.url.path,
        body=body,
    )
    return {
        "client_id": client_id,
        "auth_method": "hmac",
        "timestamp": timestamp,
        "user_id": f"hmac:{client_id}",
        "session_id": None,
    }


async def list_hmac_clients() -> dict[str, bool]:
    return dict.fromkeys(await _list_secret_ids(), True)


def _reset_for_tests() -> None:
    """Wipe Redis client cache (file remains as-is)."""
    global _redis_client
    _redis_client = None
