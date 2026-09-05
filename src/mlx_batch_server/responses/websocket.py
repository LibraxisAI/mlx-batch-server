"""Framework-free WebSocket command adapter for ``/v1/responses``."""

from __future__ import annotations

import inspect
import json
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

from .projector import NoWireResponseEvent, OpenAIProjectionState, project_event
from .transport import (
    MultiplexedTransportSession,
    ResponseCreateCommand,
    ResponseEventSource,
    ResponseInjectCommand,
    ResponseSteerCommand,
    ResponseSteerRejectedError,
    SessionClosedError,
    StreamId,
    TransportControlOutcome,
    TransportEnvelope,
    TransportErrorOutcome,
    TransportOutcome,
    TransportProtocolError,
    TransportSession,
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


@dataclass(frozen=True, slots=True)
class _ClientOwnedStop:
    command: ResponseCreateCommand
    stream_id: StreamId | None
    required_call_ids: frozenset[str]
    next_sequence_number: int


@dataclass(frozen=True, slots=True)
class _PendingSteer:
    stop: _ClientOwnedStop
    command: ResponseSteerCommand
    steer_id: str


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
        self._commands_by_response_id: dict[str, ResponseCreateCommand] = {}
        self._client_owned_stops: dict[str, _ClientOwnedStop] = {}
        self._pending_steers: dict[str, _PendingSteer] = {}
        self._projection_states: dict[StreamId | None, OpenAIProjectionState] = {}

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
        pending = self._pending_steer_for(command)
        successor = (
            command if pending is None else _merge_pending_steer(command, pending)
        )
        if pending is not None:
            self._pending_steers.pop(pending.command.previous_response_id, None)
            self._client_owned_stops.pop(pending.command.previous_response_id, None)
        try:
            await self._core.open(
                successor.stream_id,
                lambda: self._resolve_source(successor),
            )
        except Exception:
            if pending is not None:
                response_id = pending.command.previous_response_id
                self._pending_steers[response_id] = pending
                self._client_owned_stops[response_id] = pending.stop
            raise

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
        if source.response_id is not None:
            self._commands_by_response_id[source.response_id] = command
        return source

    async def cancel(self, stream_id: StreamId | None, reason: str) -> None:
        """Internal cancellation seam for the separate HTTP cancel surface."""

        await self._core.cancel(stream_id, reason)

    async def steer(self, command: ResponseSteerCommand) -> Mapping[str, Any] | None:
        if self._steer_response is not None:
            result = self._steer_response(command)
            if inspect.isawaitable(result):
                await result
            return None

        original = self._commands_by_response_id.get(command.previous_response_id)
        if original is None:
            return render_steer_failed(
                command,
                code="response_not_found",
                message="the target response is not active on this connection",
            )
        stopped = self._client_owned_stops.get(command.previous_response_id)
        if stopped is not None:
            if command.previous_response_id in self._pending_steers:
                return render_steer_failed(
                    command,
                    code="too_many_pending_steers",
                    message="the target response already has pending steering input",
                )
            steer_id = f"steer_{uuid.uuid4().hex}"
            pending = _PendingSteer(
                stop=stopped,
                command=command,
                steer_id=steer_id,
            )
            self._pending_steers[command.previous_response_id] = pending
            accepted = _render_pending_steer_control(
                pending,
                event_type="response.steer.accepted",
                sequence_number=stopped.next_sequence_number,
            )
            waiting = _render_pending_steer_control(
                pending,
                event_type="response.steer.pending",
                sequence_number=stopped.next_sequence_number + 1,
            )
            waiting = {**waiting, "type": "response.steer.pending"}
            try:
                await self._core.publish_controls(
                    stopped.stream_id,
                    (accepted, waiting),
                )
            except Exception:
                self._pending_steers.pop(command.previous_response_id, None)
                raise
            return None
        successor_body = dict(original.response)
        successor_body["previous_response_id"] = command.previous_response_id
        successor_body["input"] = command.input
        successor = ResponseCreateCommand(
            response=successor_body,
            stream_id=original.stream_id,
        )
        steer_id = f"steer_{uuid.uuid4().hex}"
        try:
            await self._core.steer(
                command.previous_response_id,
                steer_id,
                lambda: self._resolve_source(successor),
            )
        except ResponseSteerRejectedError as error:
            return render_steer_failed(
                command,
                code=error.code,
                message=str(error),
                steer_id=steer_id,
            )
        return None

    async def handle(self, command: WebSocketCommand) -> Mapping[str, Any] | None:
        if isinstance(command, ResponseCreateCommand):
            await self.create(command)
            return None
        if isinstance(command, ResponseSteerCommand):
            return await self.steer(command)
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
        while True:
            outcome = await self.receive()
            if isinstance(outcome, TransportControlOutcome):
                return dict(outcome.payload)
            if isinstance(outcome, TransportErrorOutcome):
                self._projection_states.pop(outcome.stream_id, None)
                return render_protocol_error(outcome.error)
            state = self._projection_states.setdefault(
                outcome.stream_id,
                OpenAIProjectionState(),
            )
            projected = project_event(outcome, state=state)
            if isinstance(projected, NoWireResponseEvent):
                continue
            if outcome.terminal_response is not None:
                terminal_response = await outcome.terminal_response
                if not isinstance(terminal_response, Mapping):
                    raise TypeError(
                        "terminal_response must resolve to a response mapping"
                    )
                projected["response"] = dict(terminal_response)
                self._remember_client_owned_stop(outcome, terminal_response)
                self._projection_states.pop(outcome.stream_id, None)
            return projected

    async def receive_encoded(self) -> str:
        return json.dumps(
            await self.receive_payload(),
            separators=(",", ":"),
            ensure_ascii=False,
        )

    async def close(self, reason: str = "transport_disconnected") -> None:
        self._client_owned_stops.clear()
        self._pending_steers.clear()
        self._projection_states.clear()
        await self._core.close(reason)

    def _remember_client_owned_stop(
        self,
        outcome: TransportEnvelope,
        terminal_response: Mapping[str, Any],
    ) -> None:
        response_id = terminal_response.get("id")
        if not isinstance(response_id, str):
            return
        original = self._commands_by_response_id.get(response_id)
        if original is None:
            return
        required = frozenset(
            call_id
            for item in terminal_response.get("output", ())
            if isinstance(item, Mapping) and item.get("type") == "function_call"
            if isinstance((call_id := item.get("call_id")), str) and call_id
        )
        if not required:
            self._client_owned_stops.pop(response_id, None)
            return
        self._client_owned_stops[response_id] = _ClientOwnedStop(
            command=original,
            stream_id=outcome.stream_id,
            required_call_ids=required,
            next_sequence_number=outcome.sequence_number + 1,
        )

    def _pending_steer_for(
        self,
        command: ResponseCreateCommand,
    ) -> _PendingSteer | None:
        previous_response_id = command.response.get("previous_response_id")
        if not isinstance(previous_response_id, str):
            return None
        return self._pending_steers.get(previous_response_id)

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


def render_steer_failed(
    command: ResponseSteerCommand,
    *,
    code: str,
    message: str,
    steer_id: str | None = None,
) -> dict[str, Any]:
    """Render rejected steering input so the client can safely retry it."""

    steer: dict[str, Any] = {
        "previous_response_id": command.previous_response_id,
    }
    if steer_id is not None:
        steer["id"] = steer_id
    return {
        "type": "response.steer.failed",
        "sequence_number": 0,
        "steer": steer,
        "input": command.input,
        "error": {"code": code, "message": message},
    }


def _render_pending_steer_control(
    pending: _PendingSteer,
    *,
    event_type: str,
    sequence_number: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": event_type,
        "sequence_number": sequence_number,
        "steer": {
            "id": pending.steer_id,
            "previous_response_id": pending.command.previous_response_id,
        },
        "input": pending.command.input,
    }
    if pending.stop.stream_id is not None:
        payload["stream_id"] = pending.stop.stream_id.value
    return payload


def _merge_pending_steer(
    command: ResponseCreateCommand,
    pending: _PendingSteer,
) -> ResponseCreateCommand:
    if command.stream_id != pending.stop.stream_id:
        raise TransportProtocolError(
            "tool output must resume the stream that owns pending steering",
            stream_id=command.stream_id,
            param="stream_id",
        )
    current = command.response.get("input")
    if isinstance(current, Mapping):
        current_items: list[Any] = [dict(current)]
    elif isinstance(current, list | tuple):
        current_items = [
            dict(item) if isinstance(item, Mapping) else item for item in current
        ]
    else:
        raise TransportProtocolError(
            "pending steering requires client-owned tool output input",
            stream_id=command.stream_id,
            param="input",
        )
    supplied = {
        call_id
        for item in current_items
        if isinstance(item, Mapping) and item.get("type") == "function_call_output"
        if isinstance((call_id := item.get("call_id")), str)
    }
    missing = pending.stop.required_call_ids - supplied
    if missing:
        raise TransportProtocolError(
            "pending steering requires output for every client-owned function call",
            stream_id=command.stream_id,
            param="input",
        )
    steer_input = pending.command.input
    if isinstance(steer_input, str):
        current_items.append(steer_input)
    else:
        current_items.extend(dict(item) for item in steer_input)
    body = dict(command.response)
    body["input"] = current_items
    return ResponseCreateCommand(response=body, stream_id=command.stream_id)
