"""Public API key issuance via signed registration tokens (`/access`).

Strips Vista-specific user upsert. The HTML registration page is preserved
for parity with api-router but uses a CSP-friendly external font stack.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac as _hmac
import json
import logging
import os
import secrets
import time
from base64 import urlsafe_b64decode
from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from ..core.config import get_settings
from .api_keys import issue_api_key, validate_api_key

router = APIRouter(tags=["Access"])
token_scheme = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)

_NONCE_TTL_SECONDS = 900
_RATE_LIMIT_TTL_SECONDS = 60
_RATE_LIMIT_LUA = (
    "local current = redis.call('INCR', KEYS[1]);"
    "if current == 1 then redis.call('EXPIRE', KEYS[1], ARGV[1]); end;"
    "return current;"
)

_RATE_BUCKET: dict[str, dict[str, Any]] = defaultdict(dict)
_redis_client: Any = None


async def _get_redis() -> Any:
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis.asyncio as redis
    except ImportError:
        return None
    try:
        client = redis.from_url(get_settings().redis_url, decode_responses=True)
        await client.ping()
    except Exception:
        return None
    _redis_client = client
    return _redis_client


async def _check_nonce_replay(nonce: str) -> bool:
    """Returns True if nonce is fresh (NOT replayed)."""
    nonce_key = f"mlx_batch:nonce:replay:{nonce}"
    client = await _get_redis()
    if client is not None:
        try:
            return bool(
                await client.set(nonce_key, b"1", ex=_NONCE_TTL_SECONDS, nx=True)
            )
        except Exception as e:
            logger.warning("Redis nonce check failed, falling back: %s", e)
    if "__nonces__" not in _RATE_BUCKET:
        _RATE_BUCKET["__nonces__"] = {}
    expiry = _RATE_BUCKET["__nonces__"].get(nonce)
    now = time.time()
    if expiry and expiry > now:
        return False
    _RATE_BUCKET["__nonces__"][nonce] = now + _NONCE_TTL_SECONDS
    return True


def _load_registration_hashes() -> set[str]:
    settings = get_settings()
    raw = settings.access_registration_hashes
    if not raw:
        raw = os.getenv("ACCESS_REGISTRATION_HASHES") or ""
    if not raw:
        return set()
    raw = raw.strip()
    try:
        decoded = json.loads(raw)
        if isinstance(decoded, list):
            return {str(item).strip() for item in decoded if str(item).strip()}
    except json.JSONDecodeError:
        pass
    return {item.strip() for item in raw.split(",") if item.strip()}


def _registration_configured() -> bool:
    return bool(get_settings().access_registration_secret)


def _hash_token(token: str) -> str | None:
    secret = get_settings().access_registration_secret
    if not secret:
        return None
    return _hmac.new(
        secret.encode("utf-8"), token.encode("utf-8"), hashlib.sha256
    ).hexdigest()


async def _validate_registration_token(token: str) -> tuple[bool, str]:
    if not token or not _registration_configured():
        return False, ""
    secret = get_settings().access_registration_secret or ""
    if token.startswith("v1:"):
        try:
            raw = token[3:]
            parts = raw.split(".")
            if len(parts) != 2:
                return False, ""
            payload_b64, sig_b64 = parts
            payload_bytes = urlsafe_b64decode(payload_b64 + "==")
            sig = urlsafe_b64decode(sig_b64 + "==")
            calc = _hmac.new(
                secret.encode("utf-8"), payload_bytes, hashlib.sha256
            ).digest()
            if not _hmac.compare_digest(sig, calc):
                return False, ""
            payload = json.loads(payload_bytes.decode("utf-8"))
            iat = int(payload.get("iat", 0))
            exp = int(payload.get("exp", 0))
            now = int(time.time())
            if iat <= 0 or exp <= now or exp - iat > 3600:
                return False, ""
            nonce = str(payload.get("nonce", ""))
            if len(nonce) < 8:
                return False, ""
            device = str(payload.get("device_id", "device"))
            subject = f"mlx/{device}"
            if not await _check_nonce_replay(nonce):
                logger.warning("Replay attack detected: nonce %s...", nonce[:8])
                return False, ""
            return True, subject
        except Exception:
            return False, ""

    allowed_hashes = _load_registration_hashes()
    if not allowed_hashes:
        return False, ""
    hashed = _hash_token(token)
    if not hashed:
        return False, ""
    return (hashed in allowed_hashes), token


async def _throttle(client_ip: str) -> bool:
    settings = get_settings()
    limit = int(settings.access_rate_limit_per_minute or 0)
    if limit <= 0:
        return True
    client = await _get_redis()
    if client is not None:
        key = f"mlx_batch:access:ratelimit:{client_ip}"
        try:
            count = await client.eval(_RATE_LIMIT_LUA, 1, key, _RATE_LIMIT_TTL_SECONDS)
        except Exception:
            count = await client.incr(key)
            if count == 1:
                await client.expire(key, _RATE_LIMIT_TTL_SECONDS)
        return int(count) <= limit
    bucket = _RATE_BUCKET[client_ip]
    now = time.time()
    if now >= bucket.get("reset", 0.0):
        bucket["reset"] = now + _RATE_LIMIT_TTL_SECONDS
        bucket["count"] = 1
        return True
    if bucket.get("count", 0) >= limit:
        return False
    bucket["count"] = bucket.get("count", 0) + 1
    return True


class AccessIssueRequest(BaseModel):
    label: str | None = Field(None)
    ttl_hours: int = Field(72, ge=1)
    scopes: list[str] | None = Field(default_factory=lambda: [])
    email: str | None = Field(
        None, description="Requester contact email (recorded for audit)"
    )


class AccessIssueResponse(BaseModel):
    api_key: str
    created_at: str
    expires_at: str
    ttl_hours: int
    subject: str
    label: str | None = None
    scopes: list[str] = []


@router.post("/access", response_model=AccessIssueResponse)
async def issue_key(
    request: Request,
    body: AccessIssueRequest,
    creds: HTTPAuthorizationCredentials | None = Depends(token_scheme),
) -> JSONResponse:
    if not creds or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer registration token required",
        )
    if not _registration_configured():
        logger.error("/access invoked while registration is disabled")
        raise HTTPException(
            status_code=503, detail="Registration is temporarily disabled"
        )

    client_ip = (
        getattr(request.client, "host", "unknown") if request.client else "unknown"
    )
    if not await _throttle(client_ip):
        raise HTTPException(status_code=429, detail="Too many requests, slow down")

    token = creds.credentials.strip()
    ok, subject = await _validate_registration_token(token)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid registration token")
    if not body.email:
        raise HTTPException(status_code=400, detail="email is required")

    settings = get_settings()
    max_ttl = int(settings.access_max_ttl_hours or 24 * 14)
    if body.ttl_hours > max_ttl:
        raise HTTPException(status_code=400, detail=f"ttl_hours exceeds max {max_ttl}")

    record = await issue_api_key(
        subject=subject, ttl_hours=body.ttl_hours, scopes=body.scopes or ["default"]
    )
    record["label"] = body.label

    logger.info(
        "API key issued",
        extra={
            "event": "access.issue",
            "label": body.label,
            "ttl_hours": body.ttl_hours,
            "client_ip": client_ip,
            "subject_hash": _hash_token(subject),
            "email": body.email,
        },
    )

    response = AccessIssueResponse(**record)
    payload = JSONResponse(content=response.model_dump())
    payload.set_cookie(
        key="mlx_api_key",
        value=record["api_key"],
        httponly=True,
        secure=True,
        samesite="lax",
        max_age=body.ttl_hours * 3600,
        path="/",
    )
    return payload


class AccessValidateResponse(BaseModel):
    valid: bool


@router.get("/access/validate", response_model=AccessValidateResponse)
async def validate_key(
    request: Request, api_key: str | None = None
) -> AccessValidateResponse:
    if not api_key:
        raise HTTPException(status_code=400, detail="api_key query parameter required")
    client_ip = (
        getattr(request.client, "host", "unknown") if request.client else "unknown"
    )
    if not await _throttle(client_ip):
        raise HTTPException(status_code=429, detail="Too many validation requests")
    await asyncio.sleep(0.05 + secrets.randbelow(50) / 1000)
    return AccessValidateResponse(valid=await validate_api_key(api_key))


@router.get("/access", response_class=HTMLResponse)
async def access_page() -> str:
    """Minimal registration page for self-service token exchange."""
    return _ACCESS_HTML


_ACCESS_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <meta name="robots" content="noindex,nofollow" />
  <title>MLX Batch Server Access</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
      background:#0b0b0c; color:#e8e8ea; display:flex; align-items:center;
      justify-content:center; min-height:100vh; margin:0; }
    .card { width: min(680px, 92vw); background:#141416;
      border:1px solid #2a2a2e; border-radius:12px;
      padding:24px 24px 18px; box-shadow: 0 8px 30px rgba(0,0,0,.45); }
    h1 { margin: 0 0 8px; font-size: 22px; }
    p { margin: 6px 0 16px; color:#b9bac0; }
    label { display:block; font-size:13px; color:#cfd0d7; margin-bottom:6px; }
    input { width:100%; padding:12px 14px; background:#0f1012; color:#e8e8ea;
      border:1px solid #2a2a2e; border-radius:10px; outline:none; }
    button { margin-top:14px; padding:10px 14px; background:#5b8cff;
      color:#fff; border:0; border-radius:10px; cursor:pointer; font-weight:600; }
    button:disabled { opacity:.6; cursor:not-allowed; }
    .out { margin-top:16px; padding:12px; border:1px dashed #2a2a2e;
      border-radius:10px; background:#0f1012; word-break: break-all; }
    .ok { color:#99d48e } .err { color:#ff7a7a }
  </style>
</head>
<body>
  <div class="card">
    <h1>MLX Batch Server API Access</h1>
    <p>Exchange a registration token for a short-lived API key. Tokens are
       signed and rate limited.</p>
    <label for="email">Email</label>
    <input id="email" type="email" placeholder="you@example.com" autocomplete="email" />
    <label for="token" style="margin-top:10px">Registration token</label>
    <input id="token" type="text" placeholder="Paste your registration token" autocomplete="off" />
    <button id="go">Generate API key</button>
    <div id="out" class="out" aria-live="polite"></div>
  </div>
<script>
const btn = document.getElementById('go');
const out = document.getElementById('out');
function show(msg, cls='') { out.className = 'out ' + cls; out.textContent = msg; }
btn.addEventListener('click', async () => {
  const t = (document.getElementById('token').value || '').trim();
  const email = (document.getElementById('email').value || '').trim();
  if (!email) { show('Please provide an email.', 'err'); return; }
  if (!t) { show('Registration token is required.', 'err'); return; }
  btn.disabled = true; show('Issuing key…');
  try {
    const headers = { 'Content-Type': 'application/json',
                      'Authorization': 'Bearer ' + t };
    const res = await fetch('/access', {
      method: 'POST', headers,
      body: JSON.stringify({ ttl_hours: 72, scopes: ['default'], email })
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
    show(data.api_key, 'ok');
  } catch (e) {
    show('Error: ' + e.message, 'err');
  } finally { btn.disabled = false; }
});
</script>
</body>
</html>
"""


def _reset_for_tests() -> None:
    global _redis_client
    _redis_client = None
    _RATE_BUCKET.clear()
