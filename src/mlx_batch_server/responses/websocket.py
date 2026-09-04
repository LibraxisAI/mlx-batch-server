"""Framework-free WebSocket command adapter for ``/v1/responses``."""

from __future__ import annotations

import inspect
import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, TypeAlias, cast

from .projector import project_event
from .transport import (
    MultiplexedTransportSession,
    ResponseCreateCommand,
    ResponseEventSource,
    ResponseInjectCommand,
    ResponseSteerCommand,
    SessionClosedError,
    StreamId,
    TransportErrorOutcome,
    TransportOutcome,
    TransportProtocolError,
    TransportSession,
    UnsupportedClientEventError,
    UnsupportedInjectError,
    WebSocketCommand,
    parse_websocket_command,
)

ResponseStarter: TypeAlias = Callable[
    [ResponseCreateCommand],
    ResponseEventSource | Awaitable[ResponseEventSource],
]
ResponseSteerer: TypeAlias = Callable[
    [ResponseSteerCommand],
    Any | Awaitable[Any],
]


class ResponsesWebSocketSession:
    """Adapt WebSocket commands to the transport-neutral multiplex core.

    The class deliberately knows nothing about FastAPI or a concrete socket.
    Route wiring supplies a response starter and serializes values returned by
    ``receive_payload`` or ``receive_encoded``.
    """

    def __init__(
        self,
        session: TransportSession,
        start_response: ResponseStarter | None = None,
        steer_response: ResponseSteerer | None = None,
    ) -> None:
        self._core = MultiplexedTransportSession(session)
        self._start_response = start_response
        self._steer_response = steer_response

    @property
    def session(self) -> TransportSession:
        return self._core.session

    @property
    def closed(self) -> bool:
        return self._core.closed

    @property
    def active_stream_ids(self) -> tuple[StreamId, ...]:
        return self._core.active_stream_ids

    async def create(self, command: ResponseCreateCommand) -> None:
        if self._start_response is None:
            raise TransportProtocolError(
                "response starter is not configured",
                stream_id=command.stream_id,
            )
        await self._core.open(
            command.stream_id,
            lambda: self._resolve_source(command),
        )

    async def _resolve_source(
        self,
        command: ResponseCreateCommand,
    ) -> ResponseEventSource:
        if self._start_response is None:
            raise TransportProtocolError(
                "response starter is not configured",
                stream_id=command.stream_id,
            )
        source = self._start_response(command)
        if inspect.isawaitable(source):
            source = await cast("Awaitable[ResponseEventSource]", source)
        if not isinstance(source, ResponseEventSource):
            raise TypeError("response starter must return ResponseEventSource")
        return source

    async def cancel(self, stream_id: StreamId | None, reason: str) -> None:
        """Internal cancellation seam for the separate HTTP cancel surface."""

        await self._core.cancel(stream_id, reason)

    async def steer(self, command: ResponseSteerCommand) -> None:
        if self._steer_response is None:
            raise UnsupportedClientEventError(
                "response.steer is not configured for this transport",
                param="type",
            )
        result = self._steer_response(command)
        if inspect.isawaitable(result):
            await result

    async def handle(self, command: WebSocketCommand) -> Mapping[str, Any] | None:
        if isinstance(command, ResponseCreateCommand):
            await self.create(command)
            return None
        if isinstance(command, ResponseSteerCommand):
            await self.steer(command)
            return None
        if isinstance(command, ResponseInjectCommand):
            error = UnsupportedInjectError(
                "Beta Multi-agent response.inject is unsupported without an "
                "atomic active-response injection seam",
                param="type",
            )
            await self.close("protocol_error")
            raise error
        raise TransportProtocolError(
            f"unsupported WebSocket command: {type(command).__name__}"
        )

    async def handle_payload(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any] | None:
        """Parse one client event and render protocol failures as WS errors."""

        try:
            command = parse_websocket_command(payload)
            return await self.handle(command)
        except TransportProtocolError as error:
            rendered = render_protocol_error(error)
            if error.close_connection:
                await self.close("protocol_error")
            return rendered

    async def receive(self) -> TransportOutcome:
        return await self._core.receive()

    async def receive_payload(self) -> dict[str, Any]:
        outcome = await self.receive()
        if isinstance(outcome, TransportErrorOutcome):
            return render_protocol_error(outcome.error)
        projected = project_event(outcome)
        if outcome.terminal_response is not None:
            terminal_response = await outcome.terminal_response
            if not isinstance(terminal_response, Mapping):
                raise TypeError("terminal_response must resolve to a response mapping")
            projected["response"] = dict(terminal_response)
        return projected

    async def receive_encoded(self) -> str:
        return json.dumps(
            await self.receive_payload(),
            separators=(",", ":"),
            ensure_ascii=False,
        )

    async def close(self, reason: str = "transport_disconnected") -> None:
        await self._core.close(reason)

    def __aiter__(self) -> ResponsesWebSocketSession:
        return self

    async def __anext__(self) -> dict[str, Any]:
        try:
            return await self.receive_payload()
        except SessionClosedError as error:
            if not self._core.active_stream_ids:
                raise StopAsyncIteration from error
            raise


def render_protocol_error(error: TransportProtocolError) -> dict[str, Any]:
    """Render the official Responses WebSocket error event shape."""

    payload: dict[str, Any] = {
        "type": "error",
        "status": error.status_code,
        "error": {
            "type": error.error_type,
            "code": error.code,
            "message": str(error),
            "param": error.param,
        },
    }
    if error.stream_id is not None:
        payload["stream_id"] = error.stream_id.value
    return payload
