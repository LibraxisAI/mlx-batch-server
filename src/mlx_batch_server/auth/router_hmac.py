"""HMAC client management endpoints (`/hmac/*`)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from .dependency import verify_auth
from .hmac import list_hmac_clients, register_hmac_client, revoke_hmac_client

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hmac", tags=["HMAC Management"])


class RegisterClientRequest(BaseModel):
    client_id: str = Field(..., description="Unique client identifier")
    description: str = Field("", description="Optional description (admin-only)")


class RegisterClientResponse(BaseModel):
    client_id: str
    secret_key: str = Field(..., description="HMAC secret. Shown only ONCE.")
    message: str = Field(default="Client registered. Store secret securely!")


class ListClientsResponse(BaseModel):
    clients: list[str]
    count: int


class RevokeClientRequest(BaseModel):
    client_id: str


class RevokeClientResponse(BaseModel):
    client_id: str
    revoked: bool
    message: str


@router.post("/register", response_model=RegisterClientResponse)
async def register_client(
    payload: RegisterClientRequest,
    auth_info: dict = Depends(verify_auth),
) -> RegisterClientResponse:
    try:
        secret_key = await register_hmac_client(payload.client_id)
        logger.info(
            "HMAC client registered: %s (by %s)",
            payload.client_id,
            auth_info.get("auth_method", "unknown"),
        )
        return RegisterClientResponse(
            client_id=payload.client_id, secret_key=secret_key
        )
    except Exception as e:
        logger.error("Failed to register HMAC client %s: %s", payload.client_id, e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register client",
        ) from e


@router.get("/clients", response_model=ListClientsResponse)
async def get_clients(auth_info: dict = Depends(verify_auth)) -> ListClientsResponse:
    clients = await list_hmac_clients()
    ids = list(clients.keys())
    return ListClientsResponse(clients=ids, count=len(ids))


@router.post("/revoke", response_model=RevokeClientResponse)
async def revoke_client(
    payload: RevokeClientRequest,
    auth_info: dict = Depends(verify_auth),
) -> RevokeClientResponse:
    revoked = await revoke_hmac_client(payload.client_id)
    if revoked:
        logger.info(
            "HMAC client revoked: %s (by %s)",
            payload.client_id,
            auth_info.get("auth_method", "unknown"),
        )
        message = f"Client {payload.client_id} revoked successfully"
    else:
        message = f"Client {payload.client_id} not found"
    return RevokeClientResponse(
        client_id=payload.client_id, revoked=revoked, message=message
    )


@router.get("/")
async def hmac_info() -> dict:
    """Public summary of how HMAC auth works for this server."""
    return {
        "method": "HMAC-SHA256",
        "description": (
            "Hash-based message authentication for trusted clients. "
            "Secret never traverses the network."
        ),
        "required_headers": {
            "X-Client-ID": "Unique client identifier",
            "X-Timestamp": "Unix timestamp (seconds since epoch)",
            "X-Signature": "HMAC-SHA256 hex signature",
        },
        "signature_format": (
            "HMAC-SHA256(secret, '{timestamp}:{METHOD}:{path}:{body_sha256}')"
        ),
        "body_hash_format": "SHA-256 hex of request body (empty string for GET)",
        "timestamp_tolerance_seconds": 300,
        "security_features": [
            "Shared secret never transmitted over network",
            "Replay attack protection via timestamp validation",
            "Request integrity verification via body hash",
            "Constant-time signature comparison",
        ],
    }
