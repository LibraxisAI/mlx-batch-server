"""Sessions management endpoints for the operator backend."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException

from mlx_batch_server.operator.auth import operator_auth
from mlx_batch_server.operator.services.session_store import session_store

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.post("/new")
async def create_session(
    _auth: dict | None = Depends(operator_auth),
) -> dict[str, str]:
    return session_store.create()


@router.get("/recent")
async def list_recent_sessions(
    limit: int = 50,
    _auth: dict | None = Depends(operator_auth),
) -> list[dict[str, Any]]:
    return session_store.list_recent(limit=limit)


@router.get("/{session_id}")
async def get_session(
    session_id: str,
    _auth: dict | None = Depends(operator_auth),
) -> dict[str, Any]:
    entry = session_store.get(session_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return entry


@router.delete("/{session_id}")
async def delete_session(
    session_id: str,
    x_confirm_delete: str | None = Header(None, alias="X-Confirm-Delete"),
    _auth: dict | None = Depends(operator_auth),
) -> dict[str, bool | str]:
    if x_confirm_delete != "true":
        raise HTTPException(
            status_code=400,
            detail="Missing X-Confirm-Delete: true header",
        )
    if not session_store.delete(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True, "session_id": session_id}
