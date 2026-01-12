"""
FastAPI router for /v1/responses endpoint.

Provides OpenAI Responses API compatible endpoints for
multi-turn conversations with extended capabilities.

Features:
- Response storage and retrieval
- Chain walking via previous_response_id
- SSE streaming with event logging
- Background response processing

Created by M&K (c)2026 The LibraxisAI Team
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..utils.logger import logger
from .adapter import ResponsesAdapter
from .context_builder import build_context_from_previous_response
from .schema import ResponseRequest, ResponseResponse, build_error_response
from .store import (
    delete_response as store_delete_response,
)
from .store import (
    get_response as store_get_response,
)
from .store import (
    store_response,
)

router = APIRouter(tags=["responses"])

# Shared adapter instance
_adapter: ResponsesAdapter | None = None


def get_adapter() -> ResponsesAdapter:
    """Get or create the shared ResponsesAdapter."""
    global _adapter
    if _adapter is None:
        _adapter = ResponsesAdapter()
    return _adapter


def _extract_api_key(request: Request, authorization: str | None = None) -> str | None:
    """Extract API key from request headers."""
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:]
    return request.headers.get("x-api-key")


@router.post("/v1/responses", response_model=ResponseResponse)
async def create_response(
    request: Request,
    body: ResponseRequest,
    authorization: str | None = Header(default=None),
) -> JSONResponse | StreamingResponse:
    """
    Create a response using the Responses API.

    This endpoint provides compatibility with OpenAI's Responses API,
    supporting multi-turn conversations with extended capabilities:

    - Text generation with reasoning
    - Multimodal input (images, audio, video)
    - Tool/function calling
    - Structured output (JSON schema)
    - Streaming responses
    - Response chaining via previous_response_id

    The endpoint automatically routes requests to the appropriate
    backend based on content type and model configuration.
    """
    adapter = get_adapter()
    api_key = _extract_api_key(request, authorization)

    # Handle previous_response_id - build context from chain
    if body.previous_response_id:
        try:
            # Get normalized input from body
            current_input = body.input
            if isinstance(current_input, str):
                current_input = [{"role": "user", "content": current_input}]

            # Build context including previous response
            enriched_input = await build_context_from_previous_response(
                previous_response_id=body.previous_response_id,
                api_key=api_key,
                current_input=current_input,
            )

            # Update body with enriched input
            body_dict = body.model_dump()
            body_dict["input"] = enriched_input
            body = ResponseRequest(**body_dict)

        except Exception as e:
            logger.warning(f"Failed to build context from previous response: {e}")
            # Continue without context enrichment

    if body.stream:
        # Streaming response
        async def event_generator():
            response_id = None
            try:
                async for event in adapter.generate_stream(body):
                    event_type = event.get("type", "")

                    # Capture response_id from response.created event
                    if event_type == "response.created":
                        resp_data = event.get("response", {})
                        response_id = resp_data.get("id")

                    yield f"event: {event_type}\n"
                    yield f"data: {json.dumps(event)}\n\n"

                yield "data: [DONE]\n\n"

                # Store completed response if storage enabled
                if body.store and response_id:
                    try:
                        # Get final response from event
                        final_response = event.get("response", {})
                        await store_response(
                            response_id=response_id,
                            api_key=api_key,
                            request_payload=body.model_dump(),
                            response_payload=final_response,
                            store_enabled=body.store,
                        )
                    except Exception as e:
                        logger.warning(f"Failed to store response: {e}")

            except Exception as e:
                logger.error(f"Streaming error: {e}", exc_info=True)
                error_event = {
                    "type": "error",
                    "error": {
                        "message": str(e),
                        "code": "internal_error",
                    },
                }
                yield "event: error\n"
                yield f"data: {json.dumps(error_event)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        # Non-streaming response
        try:
            response = await adapter.generate(body)

            # Store response if enabled
            if body.store:
                try:
                    response_dict = response.model_dump(exclude_none=True)
                    await store_response(
                        response_id=response.id,
                        api_key=api_key,
                        request_payload=body.model_dump(),
                        response_payload=response_dict,
                        store_enabled=body.store,
                    )
                except Exception as e:
                    logger.warning(f"Failed to store response: {e}")

            return JSONResponse(content=response.model_dump(exclude_none=True))
        except Exception as e:
            logger.error(f"Response generation error: {e}", exc_info=True)
            error_response = build_error_response(
                str(e),
                error_code="internal_error",
                model=body.model,
            )
            return JSONResponse(
                content=error_response.model_dump(exclude_none=True),
                status_code=500,
            )


@router.get("/v1/responses/{response_id}")
async def get_response(
    request: Request,
    response_id: str,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """
    Retrieve a stored response by ID.

    Returns the full response object including:
    - Request parameters
    - Generated output
    - Token usage
    - Metadata
    """
    api_key = _extract_api_key(request, authorization)

    stored = await store_get_response(response_id=response_id, api_key=api_key)

    if stored is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Response '{response_id}' not found",
                    "type": "invalid_request_error",
                    "code": "response_not_found",
                }
            },
        )

    return JSONResponse(content=stored.response)


@router.delete("/v1/responses/{response_id}")
async def delete_response(
    request: Request,
    response_id: str,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """
    Delete a stored response.

    Permanently removes the response and associated events.
    Returns success even if the response was already deleted.
    """
    api_key = _extract_api_key(request, authorization)

    try:
        deleted = await store_delete_response(response_id=response_id, api_key=api_key)
    except PermissionError as err:
        raise HTTPException(
            status_code=403,
            detail={
                "error": {
                    "message": "Access denied - API key does not own this response",
                    "type": "permission_error",
                    "code": "access_denied",
                }
            },
        ) from err

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Response '{response_id}' not found",
                    "type": "invalid_request_error",
                    "code": "response_not_found",
                }
            },
        )

    return JSONResponse(
        content={
            "id": response_id,
            "object": "response.deleted",
            "deleted": True,
        }
    )


@router.post("/v1/responses/{response_id}/cancel")
async def cancel_response(
    request: Request,
    response_id: str,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """
    Cancel an in-progress response.

    Only applicable for background responses that are still processing.
    Streaming responses should be cancelled by closing the connection.
    """
    api_key = _extract_api_key(request, authorization)

    stored = await store_get_response(response_id=response_id, api_key=api_key)

    if stored is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Response '{response_id}' not found",
                    "type": "invalid_request_error",
                    "code": "response_not_found",
                }
            },
        )

    # Check if response is still in progress
    if stored.status != "in_progress":
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"Response '{response_id}' is not in progress (status: {stored.status})",
                    "type": "invalid_request_error",
                    "code": "response_not_cancellable",
                }
            },
        )

    # TODO: Implement actual cancellation for background tasks
    # For now, return the response as-is
    return JSONResponse(content=stored.response)


@router.get("/v1/responses/{response_id}/input_items")
async def list_input_items(
    request: Request,
    response_id: str,
    authorization: str | None = Header(default=None),
) -> JSONResponse:
    """
    List input items for a response.

    Returns the normalized input from the original request,
    useful for debugging and auditing.
    """
    api_key = _extract_api_key(request, authorization)

    stored = await store_get_response(response_id=response_id, api_key=api_key)

    if stored is None:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Response '{response_id}' not found",
                    "type": "invalid_request_error",
                    "code": "response_not_found",
                }
            },
        )

    # Extract input items from stored request
    input_items = stored.request.get("input", [])

    # Build response format
    items: list[dict[str, Any]] = []
    for idx, item in enumerate(input_items):
        if isinstance(item, dict):
            items.append(
                {
                    "id": f"input_{idx}",
                    "type": "message",
                    "role": item.get("role", "user"),
                    "content": item.get("content", []),
                }
            )
        elif isinstance(item, str):
            items.append(
                {
                    "id": f"input_{idx}",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": item}],
                }
            )

    return JSONResponse(
        content={
            "object": "list",
            "data": items,
            "first_id": items[0]["id"] if items else None,
            "last_id": items[-1]["id"] if items else None,
            "has_more": False,
        }
    )
