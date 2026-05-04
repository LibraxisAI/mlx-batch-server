"""Health endpoints for the operator backend."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
@router.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
