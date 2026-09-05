"""FastAPI transport for one lifespan-owned Responses runtime graph."""

from __future__ import annotations

import asyncio
import inspect
import json
import time
import uuid
from collections.abc import AsyncIterator, Mapping
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.websockets import WebSocketDisconnect

from ..auth.dependency import verify_auth, verify_websocket_auth
from ..runtime.events import TERMINAL_EVENT_TYPES
from .compaction import CompactionError
from .errors import OpenAIError, render_http_error, render_sse_error
from .input_items import (
    DEFAULT_INPUT_ITEMS_LIMIT,
    InputItemsPaginationError,
    paginate_input_items,
)
from .operations import ResponsesOperationError
from .projector import project_event
from .registry import ResponseRegistry, ResponseRegistryError
from .request_contract import ResponsesMappingError, capability_profile, local_setting
from .transport import (
    PublishedResponseEvent,
    ResponseEventSource,
    TransportEnvelope,
    TransportProtocolError,
    TransportSession,
)
from .websocket import ResponsesWebSocketSession, render_protocol_error

if TYPE_CHECKING:
    from .controller import ResponsesController


@runtime_checkable
class ResponsesRouteRuntime(Protocol):
    """The only runtime references a protocol router may consume."""

    responses_controller: ResponsesController
    response_registry: ResponseRegistry


def build_runtime_responses_router(  # noqa: PLR0915 - protocol route factory
    runtime: ResponsesRouteRuntime,
    *,
    cancel_wait_timeout_s: float = 30.0,
) -> APIRouter:
    """Bind HTTP, SSE and WSS to one controller/registry pair."""

    if not isinstance(runtime, ResponsesRouteRuntime):
        raise TypeError("runtime must satisfy ResponsesRouteRuntime")
    if cancel_wait_timeout_s <= 0:
        raise ValueError("cancel_wait_timeout_s must be positive")

    controller = runtime.responses_controller
    registry = runtime.response_registry
    operations = getattr(runtime, "responses_operations", None)
    router = APIRouter()

    @router.post("/v1/responses", response_model=None)
    async def create_response(
        request: Request,
        auth_info: dict[str, Any] = Depends(verify_auth),
    ) -> JSONResponse | StreamingResponse:
        owner_id = _response_owner_id(auth_info)
        try:
            body = await _json_object_body(request)
            stream = local_setting("stream", body.get("stream", False))
            background = local_setting(
                "background",
                body.get("background", False),
            )
            if background and not stream:
                snapshot = await controller.create_background(
                    body,
                    owner_id=owner_id,
                )
                return JSONResponse(content=dict(snapshot))
            source = await controller.create(body, owner_id=owner_id)
        except ResponseRegistryError as error:
            return _registry_error(error)
        except (ResponsesMappingError, TypeError, ValueError) as error:
            return _request_exception(error)
        except RuntimeError as error:
            return _server_exception(
                error,
                status_code=503,
                code="responses_runtime_unavailable",
            )

        if stream:
            return StreamingResponse(
                _sse_events(source),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )
        try:
            terminal = await _drain_to_terminal(source)
        except Exception as error:
            return _server_exception(error)
        return JSONResponse(content=dict(terminal))

    @router.post("/v1/responses/compact", response_model=None)
    async def compact_response(
        request: Request,
        auth_info: dict[str, Any] = Depends(verify_auth),
    ) -> JSONResponse:
        if operations is None:
            return _server_error(
                "local Responses compaction is not configured",
                status_code=503,
                code="responses_compaction_unavailable",
            )
        try:
            body = await _json_object_body(request)
            compacted = await operations.compact(
                body,
                owner_id=_response_owner_id(auth_info),
            )
        except ResponseRegistryError as error:
            return _registry_error(error)
        except (CompactionError, ResponsesOperationError) as error:
            return _operation_error(error)
        except (ResponsesMappingError, TypeError, ValueError) as error:
            return _request_exception(error)
        except RuntimeError as error:
            return _server_exception(
                error,
                status_code=503,
                code="responses_compaction_unavailable",
            )
        return JSONResponse(content=dict(compacted))

    @router.post("/v1/responses/input_tokens", response_model=None)
    async def count_response_input_tokens(
        request: Request,
        auth_info: dict[str, Any] = Depends(verify_auth),
    ) -> JSONResponse:
        if operations is None:
            return _server_error(
                "local Responses input-token counting is not configured",
                status_code=503,
                code="responses_input_tokens_unavailable",
            )
        try:
            body = await _json_object_body(request)
            result = await operations.count_input_tokens(
                body,
                owner_id=_response_owner_id(auth_info),
            )
        except ResponseRegistryError as error:
            return _registry_error(error)
        except (CompactionError, ResponsesOperationError) as error:
            return _operation_error(error)
        except (ResponsesMappingError, TypeError, ValueError) as error:
            return _request_exception(error)
        except RuntimeError as error:
            return _server_exception(
                error,
                status_code=503,
                code="responses_input_tokens_unavailable",
            )
        return JSONResponse(content=dict(result))

    @router.get("/v1/responses/capabilities")
    async def get_response_capabilities(
        _auth_info: dict[str, Any] = Depends(verify_auth),
    ) -> JSONResponse:
        return JSONResponse(content=dict(capability_profile()))

    @router.get("/v1/responses/{response_id}")
    async def get_response(
        response_id: str,
        request: Request,
        auth_info: dict[str, Any] = Depends(verify_auth),
    ) -> JSONResponse:
        try:
            _validate_retrieve_query(request)
            response = registry.get(
                response_id,
                owner_id=_response_owner_id(auth_info),
            )
        except ResponseRegistryError as error:
            return _registry_error(error)
        except (ResponsesMappingError, TypeError, ValueError) as error:
            return _request_exception(error)
        return JSONResponse(content=response)

    @router.delete("/v1/responses/{response_id}")
    async def delete_response(
        response_id: str,
        auth_info: dict[str, Any] = Depends(verify_auth),
    ) -> JSONResponse:
        try:
            response = registry.delete(
                response_id,
                owner_id=_response_owner_id(auth_info),
            )
        except ResponseRegistryError as error:
            return _registry_error(error)
        return JSONResponse(content=response)

    @router.post("/v1/responses/{response_id}/cancel")
    async def cancel_response(
        response_id: str,
        auth_info: dict[str, Any] = Depends(verify_auth),
    ) -> JSONResponse:
        owner_id = _response_owner_id(auth_info)
        try:
            snapshot = registry.get(response_id, owner_id=owner_id)
            if snapshot.get("background") is not True:
                raise ResponseRegistryError(
                    f"response {response_id!r} was not created in background mode",
                    code="response_not_cancellable",
                    status_code=409,
                )
            controller.cancel(
                response_id,
                owner_id=owner_id,
                reason="http_cancel_requested",
            )
            terminal = await asyncio.to_thread(
                registry.wait_terminal,
                response_id,
                cancel_wait_timeout_s,
                owner_id=owner_id,
            )
        except ResponseRegistryError as error:
            return _registry_error(error)
        if terminal is None:
            return _server_error(
                "response cancellation did not reach a terminal state in time",
                status_code=504,
                code="response_cancel_timeout",
            )
        return JSONResponse(content=terminal)

    @router.get("/v1/responses/{response_id}/input_items")
    async def list_input_items(
        response_id: str,
        request: Request,
        auth_info: dict[str, Any] = Depends(verify_auth),
    ) -> JSONResponse:
        try:
            options = _input_items_query(request)
            items = registry.input_items(
                response_id,
                owner_id=_response_owner_id(auth_info),
            )
            page = paginate_input_items(items, **options)
        except ResponseRegistryError as error:
            return _registry_error(error)
        except (InputItemsPaginationError, ResponsesMappingError) as error:
            return _request_exception(error)
        return JSONResponse(content=page)

    @router.websocket("/v1/responses")
    async def responses_websocket(
        websocket: WebSocket,
        auth_info: dict[str, Any] = Depends(verify_websocket_auth),
    ) -> None:
        owner_id = _response_owner_id(auth_info)
        await websocket.accept()
        session = ResponsesWebSocketSession(
            TransportSession(
                connection_id=f"ws_{uuid.uuid4().hex}",
                principal=owner_id,
                opened_at=time.time(),
            ),
            start_response=lambda command: controller.create(
                command.response,
                owner_id=owner_id,
            ),
        )
        await _serve_websocket(websocket, session)

    return router


async def _drain_to_terminal(
    source: ResponseEventSource,
) -> Mapping[str, Any]:
    terminal_seen = False
    async for item in source.events:
        if not isinstance(item, PublishedResponseEvent):
            raise TypeError("controller source must yield published response events")
        if isinstance(item.event, TERMINAL_EVENT_TYPES):
            terminal_seen = True
    if not terminal_seen:
        raise RuntimeError("response source ended without a terminal event")
    if source.terminal_response is None:
        raise RuntimeError("response source is missing its terminal receipt")
    terminal = await source.terminal_response
    if not isinstance(terminal, Mapping):
        raise TypeError("terminal response must be a mapping")
    return terminal


async def _sse_events(source: ResponseEventSource) -> AsyncIterator[bytes]:
    terminal_seen = False
    next_sequence_number = 0
    try:
        async for item in source.events:
            if not isinstance(item, PublishedResponseEvent):
                raise TypeError(
                    "controller source must yield published response events"
                )
            next_sequence_number = item.sequence_number + 1
            terminal = isinstance(item.event, TERMINAL_EVENT_TYPES)
            envelope = TransportEnvelope(
                stream_id=None,
                sequence_number=item.sequence_number,
                event=item.event,
                terminal_response=(source.terminal_response if terminal else None),
                snapshot=item.snapshot,
            )
            projected = project_event(envelope)
            if terminal:
                terminal_seen = True
                if source.terminal_response is None:
                    raise RuntimeError("terminal event is missing its response receipt")
                terminal_response = await source.terminal_response
                if not isinstance(terminal_response, Mapping):
                    raise TypeError("terminal response must be a mapping")
                projected["response"] = dict(terminal_response)
            event_type = str(projected["type"])
            payload = json.dumps(
                projected,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            yield f"event: {event_type}\ndata: {payload}\n\n".encode()
        if not terminal_seen:
            raise RuntimeError("response source ended without a terminal event")
        yield b"data: [DONE]\n\n"
    except asyncio.CancelledError:
        raise
    except Exception as error:
        payload = json.dumps(
            render_sse_error(
                OpenAIError.from_exception(
                    error,
                    code="responses_transport_failed",
                    sequence_number=next_sequence_number,
                )
            ),
            separators=(",", ":"),
            ensure_ascii=False,
        )
        yield f"event: error\ndata: {payload}\n\n".encode()
    finally:
        if not terminal_seen and source.cancel_on_disconnect:
            await _cancel_source(source, "sse_transport_closed")


async def _serve_websocket(
    websocket: WebSocket,
    session: ResponsesWebSocketSession,
) -> None:
    send_lock = asyncio.Lock()

    async def send(payload: Mapping[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(dict(payload))

    async def receive_commands() -> None:
        try:
            while not session.closed:
                payload = await websocket.receive_json()
                if not isinstance(payload, Mapping):
                    await send(
                        render_protocol_error(
                            TransportProtocolError(
                                "WebSocket event must be a JSON object",
                                param="type",
                            )
                        )
                    )
                    continue
                outcome = await session.handle_payload(payload)
                if outcome is not None:
                    await send(outcome)
                if session.closed:
                    return
        except WebSocketDisconnect:
            return

    async def send_events() -> None:
        async for payload in session:
            await send(payload)

    receive_task = asyncio.create_task(receive_commands())
    send_task = asyncio.create_task(send_events())
    tasks = {receive_task, send_task}
    try:
        done, pending = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        error = next(
            (
                task.exception()
                for task in done
                if not task.cancelled() and task.exception() is not None
            ),
            None,
        )
        await session.close("transport_disconnected")
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        if error is not None:
            raise error
    finally:
        with suppress(Exception):
            await session.close("transport_disconnected")
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _cancel_source(source: ResponseEventSource, reason: str) -> None:
    if source.cancel is None:
        return
    with suppress(Exception):
        result = source.cancel(reason)
        if inspect.isawaitable(result):
            await result


async def _json_object_body(request: Request) -> dict[str, Any]:
    """Parse a Responses body without allowing FastAPI to synthesize a 422."""

    try:
        decoded = json.loads(await request.body())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ResponsesMappingError(
            "request body must contain valid JSON",
            code="invalid_json",
            param=None,
        ) from error
    if not isinstance(decoded, Mapping):
        raise ResponsesMappingError(
            "request body must be a JSON object",
            code="invalid_responses_request",
            param=None,
        )
    return dict(decoded)


def _query_values(request: Request) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for raw_name, value in request.query_params.multi_items():
        name = raw_name.removesuffix("[]")
        values.setdefault(name, []).append(value)
    return values


def _single_query_value(values: Mapping[str, list[str]], field: str) -> str | None:
    items = values.get(field)
    if items is None:
        return None
    if len(items) != 1:
        raise ResponsesMappingError(
            f"{field} must be supplied exactly once",
            code="invalid_responses_request",
            param=field,
        )
    return items[0]


def _parse_query_bool(value: str, field: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    raise ResponsesMappingError(
        f"{field} must be a boolean",
        code="invalid_responses_request",
        param=field,
    )


def _unsupported_query(field: str) -> ResponsesMappingError:
    return ResponsesMappingError(
        f"unsupported Responses query parameter: {field}",
        code="unsupported_parameter",
        param=field,
    )


def _validate_retrieve_query(request: Request) -> None:
    values = _query_values(request)
    for field in values:
        if field not in {"include", "stream"}:
            raise _unsupported_query(field)
    include = values.get("include")
    if include is not None:
        local_setting("include", include)
    stream_value = _single_query_value(values, "stream")
    if stream_value is not None:
        stream = local_setting("stream", _parse_query_bool(stream_value, "stream"))
        if stream:
            raise _unsupported_query("stream")


def _input_items_query(request: Request) -> dict[str, Any]:
    values = _query_values(request)
    for field in values:
        if field not in {"after", "include", "limit", "order"}:
            raise _unsupported_query(field)

    include = values.get("include")
    if include is not None:
        local_setting("include", include)
    after = _single_query_value(values, "after")
    order = _single_query_value(values, "order") or "desc"
    raw_limit = _single_query_value(values, "limit")
    limit = DEFAULT_INPUT_ITEMS_LIMIT
    if raw_limit is not None:
        try:
            limit = int(raw_limit)
        except ValueError as error:
            raise InputItemsPaginationError(
                "limit must be an integer between 1 and 100",
                code="invalid_limit",
                param="limit",
            ) from error
    return {"after": after, "limit": limit, "order": order}


def _response_owner_id(auth_info: Mapping[str, Any]) -> str:
    owner_id = auth_info.get("response_owner_id")
    if not isinstance(owner_id, str) or not owner_id.startswith("resp-owner:v1:"):
        raise HTTPException(
            status_code=500,
            detail="verified authentication is missing response ownership",
        )
    return owner_id


def _registry_error(error: ResponseRegistryError) -> JSONResponse:
    return render_http_error(
        OpenAIError(
            message=error.message,
            type=(
                "server_error" if error.status_code >= 500 else "invalid_request_error"
            ),
            code=error.code,
            param=error.param,
            status_code=error.status_code,
        )
    )


def _request_error(
    message: str,
    *,
    code: str = "invalid_request",
    param: str | None = None,
) -> JSONResponse:
    return render_http_error(
        OpenAIError(
            message=message,
            type="invalid_request_error",
            code=code,
            param=param,
            status_code=400,
        )
    )


def _request_exception(error: BaseException) -> JSONResponse:
    return _request_error(
        str(error) or type(error).__name__,
        code=str(getattr(error, "code", "invalid_request")),
        param=getattr(error, "param", None),
    )


def _operation_error(
    error: CompactionError | ResponsesOperationError,
) -> JSONResponse:
    status_code = int(getattr(error, "status_code", 400))
    error_type = "server_error" if status_code >= 500 else "invalid_request_error"
    return render_http_error(
        OpenAIError(
            message=str(error) or type(error).__name__,
            type=error_type,
            param=getattr(error, "param", None),
            code=str(getattr(error, "code", "invalid_request")),
            status_code=status_code,
        )
    )


def _server_error(
    message: str,
    *,
    status_code: int = 500,
    code: str = "internal_error",
) -> JSONResponse:
    return render_http_error(
        OpenAIError(
            message=message,
            type="server_error",
            param=None,
            code=code,
            status_code=status_code,
        )
    )


def _server_exception(
    error: BaseException,
    *,
    status_code: int = 500,
    code: str = "internal_error",
) -> JSONResponse:
    return render_http_error(
        OpenAIError.from_exception(
            error,
            status_code=status_code,
            code=code,
        )
    )


__all__ = ["ResponsesRouteRuntime", "build_runtime_responses_router"]
