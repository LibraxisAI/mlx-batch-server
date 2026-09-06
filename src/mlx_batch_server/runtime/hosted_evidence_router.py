"""Authenticated localhost-only admission evidence projection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException, Request

from ..auth.dependency import verify_auth

if TYPE_CHECKING:
    from .hosted_evidence import HostedEvidenceRegistry, HostedRequestEvidence


def build_hosted_evidence_router(
    registry: HostedEvidenceRegistry,
    *,
    config: dict[str, Any],
) -> APIRouter:
    router = APIRouter(prefix="/internal/v1/hosted-evidence", tags=["internal"])

    def local(request: Request) -> None:
        host = None if request.client is None else request.client.host
        if host not in {"127.0.0.1", "::1"}:
            raise HTTPException(
                status_code=403,
                detail="hosted evidence is localhost-only",
            )

    @router.get("/config")
    async def get_config(
        request: Request,
        _auth: dict[str, Any] = Depends(verify_auth),
    ) -> dict[str, Any]:
        local(request)
        return dict(config)

    @router.get("/by-response/{response_id}")
    async def by_response(
        response_id: str,
        request: Request,
        _auth: dict[str, Any] = Depends(verify_auth),
    ) -> dict[str, Any]:
        local(request)
        return _render(registry.by_response(response_id))

    @router.get("/by-trace/{trace_id}")
    async def by_trace(
        trace_id: str,
        request: Request,
        _auth: dict[str, Any] = Depends(verify_auth),
    ) -> dict[str, Any]:
        local(request)
        return _render(registry.by_trace(trace_id))

    return router


def _render(value: HostedRequestEvidence | None) -> dict[str, Any]:
    if value is None:
        raise HTTPException(status_code=404, detail="hosted evidence not found")
    rendered = _plain(value)
    if not isinstance(rendered, dict):  # pragma: no cover - dataclass contract
        raise TypeError("hosted evidence did not render as an object")
    return rendered


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field.name: _plain(getattr(value, field.name)) for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


__all__ = ["build_hosted_evidence_router"]
