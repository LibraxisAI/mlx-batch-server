"""SSE proxy from the operator playground to the local inference server."""

from __future__ import annotations

import json
import logging
import os
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any

import httpx
from fastapi import APIRouter, Depends, Header
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mlx_batch_server.operator.config import Settings, get_settings
from mlx_batch_server.operator.services.session_store import session_store

router = APIRouter(prefix="/api/playground", tags=["playground"])
logger = logging.getLogger(__name__)

_SESSION_HISTORY: dict[str, list[dict[str, Any]]] = defaultdict(list)


def _remember_response(session_id: str, response: dict[str, Any]) -> None:
    history = _SESSION_HISTORY[session_id]
    history.append(response)
    del history[:-20]

    try:
        session_store.remember_response(session_id, response)
    except Exception:
        logger.exception("Failed to sync playground response into session store")


class PlaygroundRequest(BaseModel):
    model: str
    input: list[dict[str, Any]]
    stream: bool = True
    previous_response_id: str | None = None
    session_id: str = "default"
    max_output_tokens: int | None = None


def _inference_headers(accept: str | None = None) -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("MLX_BATCH_INTERNAL_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    if accept:
        headers["Accept"] = accept
    return headers


@router.get("/history")
async def playground_history(session_id: str = "default") -> dict[str, Any]:
    return {
        "session_id": session_id,
        "items": _SESSION_HISTORY.get(session_id, [])[-20:],
    }


@router.post("/responses")
async def proxy_responses(
    payload: PlaygroundRequest,
    settings: Settings = Depends(get_settings),
    accept: str | None = Header(default=None),
) -> StreamingResponse:
    headers = _inference_headers(accept)
    request_body = payload.model_dump(exclude_none=True)
    url = f"{settings.normalized_inference_base_url}/v1/responses"
    timeout = httpx.Timeout(settings.request_timeout_seconds, read=None)

    async def stream() -> AsyncIterator[bytes]:
        async with (
            httpx.AsyncClient(timeout=timeout) as client,
            client.stream(
                "POST",
                url,
                headers=headers,
                json=request_body,
            ) as response,
        ):
            response.raise_for_status()
            last_response: dict[str, Any] | None = None
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        data = json.loads(line[6:])
                        response_obj = data.get("response") or {}
                        if isinstance(data, dict) and (
                            data.get("response_id") or response_obj.get("id")
                        ):
                            response_id = data.get("response_id") or response_obj.get(
                                "id"
                            )
                            model = data.get("model") or response_obj.get("model")
                            last_response = {
                                "response_id": response_id,
                                "model": model,
                            }
                    except json.JSONDecodeError:
                        pass
                yield (line + "\n").encode("utf-8")
            if last_response:
                _remember_response(payload.session_id, last_response)

    media_type = "text/event-stream" if payload.stream else "application/json"
    return StreamingResponse(stream(), media_type=media_type)
