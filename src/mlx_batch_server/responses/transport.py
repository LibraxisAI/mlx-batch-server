"""Transport-neutral contracts for multiplexed Responses WebSockets."""

from __future__ import annotations

import asyncio
import inspect
import re
from collections import deque
from collections.abc import AsyncIterable, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias, cast

from ..runtime.events import (
    TERMINAL_EVENT_TYPES,
    SequencedTurnEvent,
    TurnEvent,
)

_STREAM_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{1,256}$")
_DEFAULT_LANE = None


class TransportProtocolError(RuntimeError):
    """A client-visible violation of the Responses transport contract."""

    code = "transport_protocol_error"
    error_type = "invalid_request_error"
    status_code = 400
    close_connection = False

    def __init__(
        self,
        message: str,
        *,
        stream_id: StreamId | None = None,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stream_id = stream_id
        self.param = param


class SessionClosedError(TransportProtocolError):
    code = "session_closed"


class StreamCapacityError(TransportProtocolError):
    code = "websocket_stream_limit_reached"


class QueueCapacityError(TransportProtocolError):
    code = "websocket_queue_limit_reached"


class UnknownStreamError(TransportProtocolError):
    code = "unknown_stream_id"


class StreamAlreadyTerminalError(TransportProtocolError):
    code = "stream_already_terminal"


class UnsupportedClientEventError(TransportProtocolError):
    code = "unsupported_websocket_event"


class UnsupportedInjectError(UnsupportedClientEventError):
    code = "response_inject_not_supported"
    close_connection = True


class ResponseSteerRejectedError(TransportProtocolError):
    """A steer command could not be accepted for an active response."""

    code = "response_not_found"

    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message, param="previous_response_id")
        self.code = code


class TransportSequenceError(TransportProtocolError):
    code = "transport_sequence_error"
    error_type = "server_error"
    status_code = 500


class TransportBackpressureError(TransportProtocolError):
    code = "transport_backpressure"
    error_type = "rate_limit_error"
    status_code = 429


class TransportSourceError(TransportProtocolError):
    code = "transport_source_failed"
    error_type = "server_error"
    status_code = 500


class MissingTerminalEventError(TransportProtocolError):
    code = "missing_terminal_event"
    error_type = "server_error"
    status_code = 500


@dataclass(frozen=True, slots=True)
class StreamId:
    value: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.value, str)
            or _STREAM_ID_PATTERN.fullmatch(self.value) is None
        ):
            raise ValueError(
                "stream_id must be 1-256 characters containing only letters, "
                "numbers, underscores, hyphens, and periods"
            )


@dataclass(frozen=True, slots=True)
class TransportSession:
    connection_id: str
    principal: str
    opened_at: float
    max_active_responses: int = 16
    max_named_streams: int = 32
    max_queued_responses: int = 128
    max_pending_events_per_stream: int = 4096

    def __post_init__(self) -> None:
        connection_id = self.connection_id.strip()
        principal = self.principal.strip()
        if not connection_id:
            raise ValueError("connection_id must not be blank")
        if not principal:
            raise ValueError("principal must not be blank")
        if not 1 <= self.max_active_responses <= 16:
            raise ValueError("max_active_responses must be between 1 and 16")
        if not 1 <= self.max_named_streams <= 32:
            raise ValueError("max_named_streams must be between 1 and 32")
        if self.max_queued_responses < 1:
            raise ValueError("max_queued_responses must be positive")
        if self.max_pending_events_per_stream < 2:
            raise ValueError("max_pending_events_per_stream must be at least 2")
        object.__setattr__(self, "connection_id", connection_id)
        object.__setattr__(self, "principal", principal)


@dataclass(frozen=True, slots=True)
class ResponseCreateCommand:
    response: Mapping[str, Any]
    stream_id: StreamId | None = None
    type: str = "response.create"

    def __post_init__(self) -> None:
        body = dict(self.response)
        forbidden = next(
            (field for field in ("stream", "background") if field in body),
            None,
        )
        if forbidden is not None:
            raise ValueError(
                f"response.create does not accept transport field {forbidden!r}"
            )
        object.__setattr__(self, "response", body)


@dataclass(frozen=True, slots=True)
class ResponseSteerCommand:
    previous_response_id: str
    input: str | tuple[Mapping[str, Any], ...]
    type: str = "response.steer"


@dataclass(frozen=True, slots=True)
class ResponseInjectCommand:
    """Beta Multi-agent event parsed only for explicit fail-closed handling."""

    response_id: str
    input: tuple[Mapping[str, Any], ...]
    type: str = "response.inject"


WebSocketCommand: TypeAlias = (
    ResponseCreateCommand | ResponseSteerCommand | ResponseInjectCommand
)


@dataclass(frozen=True, slots=True)
class TransportEnvelope:
    stream_id: StreamId | None
    sequence_number: int
    event: TurnEvent
    terminal_response: Awaitable[Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.sequence_number < 0:
            raise ValueError("sequence_number must be non-negative")


@dataclass(frozen=True, slots=True)
class TransportErrorOutcome:
    """Lane-local transport failure outside the canonical response stream."""

    stream_id: StreamId | None
    error: TransportProtocolError
    terminal_response: Awaitable[Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        if self.error.stream_id != self.stream_id:
            raise ValueError("transport error stream_id must match its outcome")


@dataclass(frozen=True, slots=True)
class TransportControlOutcome:
    """A transport-authored event ordered inside one response lane."""

    payload: Mapping[str, Any]


TransportOutcome: TypeAlias = (
    TransportEnvelope | TransportErrorOutcome | TransportControlOutcome
)


CancelCallback: TypeAlias = Callable[[str], Any | Awaitable[Any]]
ResponseEvents: TypeAlias = AsyncIterable[SequencedTurnEvent | TurnEvent]


@dataclass(frozen=True, slots=True)
class ResponseEventSource:
    """One GenerationTurn subscription and its owned cancellation hook."""

    events: ResponseEvents
    cancel: CancelCallback | None = None
    cancel_on_disconnect: bool = True
    terminal_response: Awaitable[Mapping[str, Any]] | None = None
    response_id: str | None = None


ResponseSourceFactory: TypeAlias = Callable[
    [], ResponseEventSource | Awaitable[ResponseEventSource]
]


@dataclass(frozen=True, slots=True)
class _QueuedEvent:
    sequence_number: int
    event: TurnEvent
    terminal_response: Awaitable[Mapping[str, Any]] | None = None


@dataclass(frozen=True, slots=True)
class _QueuedTransportError:
    error: TransportProtocolError
    terminal_response: Awaitable[Mapping[str, Any]] | None


@dataclass(frozen=True, slots=True)
class _QueuedControl:
    payload: Mapping[str, Any]


@dataclass(slots=True)
class _PendingResponse:
    source_factory: ResponseSourceFactory


@dataclass(slots=True)
class _ActiveResponse:
    source: ResponseEventSource | None = None
    response_id: str | None = None
    next_source_sequence_number: int = 0
    next_wire_sequence_number: int = 0
    sequence_mode: str | None = None
    terminal_enqueued: bool = False
    transport_error_enqueued: bool = False
    slot_released: bool = False
    cancel_invoked: bool = False
    cancel_requested_reason: str | None = None
    steer_pending: bool = False
    startup_complete: asyncio.Event = field(default_factory=asyncio.Event)
    cancellation_complete: asyncio.Event = field(default_factory=asyncio.Event)
    pump: asyncio.Task[None] | None = None


@dataclass(slots=True)
class _Lane:
    stream_id: StreamId | None
    queue: asyncio.Queue[_QueuedEvent | _QueuedTransportError | _QueuedControl]
    pending: deque[_PendingResponse] = field(default_factory=deque)
    active: _ActiveResponse | None = None
    ready: bool = False


class MultiplexedTransportSession:
    """Run bounded, FIFO response lanes over one transport connection.

    One response may execute per lane. Different lanes can execute concurrently,
    up to the connection-wide active-response limit. Additional accepted creates
    remain in a bounded queue and are started only after capacity is released.
    """

    def __init__(self, session: TransportSession) -> None:
        self.session = session
        self._lanes: dict[StreamId | None, _Lane] = {}
        self._named_stream_ids: set[StreamId] = set()
        self._ready_lanes: deque[StreamId | None] = deque()
        self._active_responses = 0
        self._queued_responses = 0
        self._lock = asyncio.Lock()
        self._events_ready = asyncio.Event()
        self._cursor = 0
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def active_stream_ids(self) -> tuple[StreamId, ...]:
        return tuple(
            stream_id
            for stream_id, lane in self._lanes.items()
            if stream_id is not None and (lane.active is not None or lane.pending)
        )

    @property
    def named_stream_ids(self) -> tuple[StreamId, ...]:
        return tuple(self._named_stream_ids)

    @property
    def active_response_count(self) -> int:
        return self._active_responses

    @property
    def queued_response_count(self) -> int:
        return self._queued_responses

    async def open(
        self,
        stream_id: StreamId | None,
        source_factory: ResponseSourceFactory,
    ) -> None:
        """Accept a response create and queue it on its ordered lane."""

        if not callable(source_factory):
            raise TypeError("response source must be provided as a lazy factory")

        async with self._lock:
            if self._closed:
                raise SessionClosedError("transport session is closed")
            if (
                stream_id is not None
                and stream_id not in self._named_stream_ids
                and len(self._named_stream_ids) >= self.session.max_named_streams
            ):
                raise StreamCapacityError(
                    "this WebSocket connection has reached its maximum number "
                    f"of distinct stream IDs ({self.session.max_named_streams})",
                    stream_id=stream_id,
                    param="stream_id",
                )

            lane = self._lanes.get(stream_id)
            can_start_immediately = (
                lane is None or lane.active is None
            ) and self._active_responses < self.session.max_active_responses
            if (
                not can_start_immediately
                and self._queued_responses >= self.session.max_queued_responses
            ):
                raise QueueCapacityError(
                    "transport session response queue is full",
                    stream_id=stream_id,
                )
            if lane is None:
                lane = _Lane(
                    stream_id=stream_id,
                    queue=asyncio.Queue(
                        maxsize=self.session.max_pending_events_per_stream + 1
                    ),
                )
                self._lanes[stream_id] = lane
            if stream_id is not None:
                self._named_stream_ids.add(stream_id)
            lane.pending.append(_PendingResponse(source_factory))
            self._queued_responses += 1
            self._mark_lane_ready_locked(lane)
            self._schedule_locked()

    async def cancel(self, stream_id: StreamId | None, reason: str) -> None:
        """Forward cancellation intent without authoring a terminal event.

        The response runtime remains the sole terminal writer. Its source must
        eventually yield the canonical cancelled, failed, or completed event.
        """

        async with self._lock:
            lane = self._lanes.get(stream_id)
            active = None if lane is None else lane.active
            if active is None:
                raise UnknownStreamError(
                    "stream_id "
                    f"{_display_stream_id(stream_id)!r} has no active response",
                    stream_id=stream_id,
                )
            if active.terminal_enqueued:
                raise StreamAlreadyTerminalError(
                    "stream_id "
                    f"{_display_stream_id(stream_id)!r} already has a terminal event",
                    stream_id=stream_id,
                )
            if active.transport_error_enqueued:
                raise StreamAlreadyTerminalError(
                    "stream_id "
                    f"{_display_stream_id(stream_id)!r} already has a transport "
                    "error outcome",
                    stream_id=stream_id,
                )
            if active.cancel_requested_reason is not None:
                raise StreamAlreadyTerminalError(
                    "stream_id "
                    f"{_display_stream_id(stream_id)!r} is already cancelling",
                    stream_id=stream_id,
                )
            active.cancel_requested_reason = reason

        await active.startup_complete.wait()
        try:
            await self._invoke_cancel(active, reason)
        finally:
            active.cancellation_complete.set()

    async def steer(
        self,
        previous_response_id: str,
        steer_id: str,
        source_factory: ResponseSourceFactory,
    ) -> None:
        """Queue one automatic successor and stop its active parent safely."""

        async with self._lock:
            match = next(
                (
                    (lane, lane.active)
                    for lane in self._lanes.values()
                    if lane.active is not None
                    and lane.active.response_id == previous_response_id
                ),
                None,
            )
            if match is None:
                raise ResponseSteerRejectedError(
                    "the target response is not active on this connection",
                    code="response_not_found",
                )
            lane, active = match
            if self._has_final_outcome(active):
                raise ResponseSteerRejectedError(
                    "the target response is already complete",
                    code="response_already_completed",
                )
            if active.steer_pending or active.cancel_requested_reason is not None:
                raise ResponseSteerRejectedError(
                    "the target response already has pending steering input",
                    code="too_many_pending_steers",
                )
            if self._queued_responses >= self.session.max_queued_responses:
                raise ResponseSteerRejectedError(
                    "the transport response queue cannot accept a successor",
                    code="too_many_pending_steers",
                )

            payload: dict[str, Any] = {
                "type": "response.steer.accepted",
                "sequence_number": active.next_wire_sequence_number,
                "steer": {
                    "id": steer_id,
                    "previous_response_id": previous_response_id,
                },
            }
            if lane.stream_id is not None:
                payload["stream_id"] = lane.stream_id.value
            lane.queue.put_nowait(_QueuedControl(payload))
            active.next_wire_sequence_number += 1
            lane.pending.appendleft(_PendingResponse(source_factory))
            self._queued_responses += 1
            active.steer_pending = True
            active.cancel_requested_reason = "steered"
            self._events_ready.set()

        await active.startup_complete.wait()
        try:
            await self._invoke_cancel(active, "steered")
        finally:
            active.cancellation_complete.set()

    async def publish_controls(
        self,
        stream_id: StreamId | None,
        payloads: Sequence[Mapping[str, Any]],
    ) -> None:
        """Publish ordered connection-owned controls after a terminal stop."""

        controls = tuple(dict(payload) for payload in payloads)
        if not controls:
            raise ValueError("at least one control payload is required")
        async with self._lock:
            if self._closed:
                raise SessionClosedError("transport session is closed")
            lane = self._lanes.get(stream_id)
            if lane is None:
                lane = _Lane(
                    stream_id=stream_id,
                    queue=asyncio.Queue(
                        maxsize=self.session.max_pending_events_per_stream + 1
                    ),
                )
                self._lanes[stream_id] = lane
            if lane.queue.qsize() + len(controls) > lane.queue.maxsize:
                raise QueueCapacityError(
                    "transport stream cannot accept control events",
                    stream_id=stream_id,
                )
            if stream_id is not None:
                self._named_stream_ids.add(stream_id)
            for payload in controls:
                lane.queue.put_nowait(_QueuedControl(payload))
            self._events_ready.set()

    async def receive(self) -> TransportOutcome:
        """Return the next available event with round-robin lane fairness."""

        while True:
            self._events_ready.clear()
            async with self._lock:
                lane_keys = tuple(self._lanes)
                if not lane_keys:
                    if self._closed:
                        raise SessionClosedError("transport session is closed")
                else:
                    total = len(lane_keys)
                    start = self._cursor % total
                    for offset in range(total):
                        index = (start + offset) % total
                        lane_key = lane_keys[index]
                        lane = self._lanes.get(lane_key)
                        if lane is None:
                            continue
                        try:
                            queued = lane.queue.get_nowait()
                        except asyncio.QueueEmpty:
                            continue

                        self._cursor = index + 1
                        if isinstance(queued, _QueuedTransportError):
                            self._release_final_outcome_locked(lane)
                            if any(
                                not item.queue.empty() for item in self._lanes.values()
                            ):
                                self._events_ready.set()
                            return TransportErrorOutcome(
                                stream_id=lane.stream_id,
                                error=queued.error,
                                terminal_response=queued.terminal_response,
                            )
                        if isinstance(queued, _QueuedControl):
                            if any(
                                not item.queue.empty() for item in self._lanes.values()
                            ):
                                self._events_ready.set()
                            return TransportControlOutcome(payload=queued.payload)
                        if isinstance(queued.event, TERMINAL_EVENT_TYPES):
                            self._release_final_outcome_locked(lane)
                        if any(not item.queue.empty() for item in self._lanes.values()):
                            self._events_ready.set()
                        return TransportEnvelope(
                            stream_id=lane.stream_id,
                            sequence_number=queued.sequence_number,
                            event=queued.event,
                            terminal_response=queued.terminal_response,
                        )
            await self._events_ready.wait()

    async def close(self, reason: str = "transport_disconnected") -> None:
        """Detach this connection and cancel only its attached active responses."""

        async with self._lock:
            if self._closed:
                return
            self._closed = True
            active = tuple(
                lane.active for lane in self._lanes.values() if lane.active is not None
            )
            self._lanes.clear()
            self._ready_lanes.clear()
            self._active_responses = 0
            self._queued_responses = 0

        await asyncio.gather(
            *(item.startup_complete.wait() for item in active),
            return_exceptions=True,
        )
        cancellable = tuple(
            item
            for item in active
            if not item.terminal_enqueued
            and item.source is not None
            and item.source.cancel_on_disconnect
        )
        await asyncio.gather(
            *(self._invoke_cancel(item, reason) for item in cancellable),
            return_exceptions=True,
        )
        for item in cancellable:
            if item.pump is not None and not item.pump.done():
                item.pump.cancel()
        if cancellable:
            await asyncio.gather(
                *(item.pump for item in cancellable if item.pump is not None),
                return_exceptions=True,
            )
        self._events_ready.set()

    def __aiter__(self) -> MultiplexedTransportSession:
        return self

    async def __anext__(self) -> TransportOutcome:
        try:
            return await self.receive()
        except SessionClosedError as error:
            raise StopAsyncIteration from error

    def _mark_lane_ready_locked(self, lane: _Lane) -> None:
        if lane.active is None and lane.pending and not lane.ready:
            lane.ready = True
            self._ready_lanes.append(lane.stream_id)

    def _schedule_locked(self) -> None:
        while (
            self._active_responses < self.session.max_active_responses
            and self._ready_lanes
        ):
            lane_key = self._ready_lanes.popleft()
            lane = self._lanes.get(lane_key)
            if lane is None:
                continue
            lane.ready = False
            if lane.active is not None or not lane.pending:
                continue
            pending = lane.pending.popleft()
            self._queued_responses -= 1
            active = _ActiveResponse()
            lane.active = active
            self._active_responses += 1
            active.pump = asyncio.create_task(
                self._start_and_pump(lane, active, pending.source_factory),
                name=(
                    f"responses-{self.session.connection_id}-"
                    f"{_display_stream_id(lane.stream_id)}"
                ),
            )

    async def _start_and_pump(
        self,
        lane: _Lane,
        active: _ActiveResponse,
        source_factory: ResponseSourceFactory,
    ) -> None:
        try:
            source_result = source_factory()
            if inspect.isawaitable(source_result):
                source_result = await cast(
                    "Awaitable[ResponseEventSource]", source_result
                )
            source = source_result
            if not isinstance(source, ResponseEventSource):
                raise TypeError("response starter must return ResponseEventSource")
            active.source = source
            active.response_id = source.response_id
            active.startup_complete.set()

            async with self._lock:
                disconnected = self._closed or lane.active is not active
                cancellation_pending = active.cancel_requested_reason is not None
            if disconnected:
                return
            if cancellation_pending:
                await active.cancellation_complete.wait()

            async for item in source.events:
                async with self._lock:
                    if self._closed or lane.active is not active:
                        return
                    if active.terminal_enqueued:
                        return
                    if isinstance(item, SequencedTurnEvent):
                        item_mode = "sequenced"
                        sequence_number = item.sequence_number
                        event = item.event
                    else:
                        item_mode = "raw"
                        sequence_number = active.next_source_sequence_number
                        event = item

                    sequence_error = (
                        active.sequence_mode is not None
                        and active.sequence_mode != item_mode
                    ) or (
                        item_mode == "sequenced"
                        and sequence_number != active.next_source_sequence_number
                    )
                    if sequence_error:
                        self._enqueue_transport_error_locked(
                            lane,
                            active,
                            TransportSequenceError(
                                "response event source violated contiguous "
                                "sequence semantics",
                                stream_id=lane.stream_id,
                            ),
                        )
                        cancel_reason = "transport_sequence_error"
                    elif not self._enqueue_locked(
                        lane,
                        active,
                        _QueuedEvent(
                            active.next_wire_sequence_number,
                            event,
                            terminal_response=(
                                source.terminal_response
                                if isinstance(event, TERMINAL_EVENT_TYPES)
                                else None
                            ),
                        ),
                    ):
                        self._enqueue_transport_error_locked(
                            lane,
                            active,
                            TransportBackpressureError(
                                "client stream exceeded its pending event limit",
                                stream_id=lane.stream_id,
                            ),
                        )
                        cancel_reason = "transport_backpressure"
                    else:
                        active.sequence_mode = item_mode
                        active.next_source_sequence_number = sequence_number + 1
                        active.next_wire_sequence_number += 1
                        cancel_reason = None
                if cancel_reason is not None:
                    await self._request_source_cancel(active, cancel_reason)
                    return
                if isinstance(event, TERMINAL_EVENT_TYPES):
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            async with self._lock:
                fault_enqueued = (
                    not self._closed
                    and lane.active is active
                    and not self._has_final_outcome(active)
                    and self._enqueue_transport_error_locked(
                        lane,
                        active,
                        TransportSourceError(
                            "response event source failed",
                            stream_id=lane.stream_id,
                        ),
                    )
                )
            if fault_enqueued:
                await self._request_source_cancel(active, "transport_source_failed")
        else:
            async with self._lock:
                fault_enqueued = (
                    not self._closed
                    and lane.active is active
                    and not self._has_final_outcome(active)
                    and self._enqueue_transport_error_locked(
                        lane,
                        active,
                        MissingTerminalEventError(
                            "response event source ended without a terminal event",
                            stream_id=lane.stream_id,
                        ),
                    )
                )
            if fault_enqueued:
                await self._request_source_cancel(active, "missing_terminal_event")
        finally:
            active.startup_complete.set()

    def _enqueue_locked(
        self,
        lane: _Lane,
        active: _ActiveResponse,
        queued: _QueuedEvent,
    ) -> bool:
        if self._has_final_outcome(active):
            return False
        is_terminal = isinstance(queued.event, TERMINAL_EVENT_TYPES)
        if (
            not is_terminal
            and lane.queue.qsize() >= self.session.max_pending_events_per_stream
        ):
            return False
        try:
            lane.queue.put_nowait(queued)
        except asyncio.QueueFull:
            return False
        if is_terminal:
            active.terminal_enqueued = True
            self._release_capacity_locked(active)
            self._schedule_locked()
        self._events_ready.set()
        return True

    def _enqueue_transport_error_locked(
        self,
        lane: _Lane,
        active: _ActiveResponse,
        error: TransportProtocolError,
    ) -> bool:
        if self._has_final_outcome(active):
            return False
        source = active.source
        lane.queue.put_nowait(
            _QueuedTransportError(
                error=error,
                terminal_response=(
                    None if source is None else source.terminal_response
                ),
            )
        )
        active.transport_error_enqueued = True
        if active.cancel_requested_reason is None:
            active.cancel_requested_reason = error.code
        self._release_capacity_locked(active)
        self._schedule_locked()
        self._events_ready.set()
        return True

    @staticmethod
    def _has_final_outcome(active: _ActiveResponse) -> bool:
        return active.terminal_enqueued or active.transport_error_enqueued

    def _release_final_outcome_locked(self, lane: _Lane) -> None:
        if lane.active is not None:
            self._release_capacity_locked(lane.active)
            lane.active = None
        self._mark_lane_ready_locked(lane)
        if lane.active is None and not lane.pending and lane.queue.empty():
            self._lanes.pop(lane.stream_id, None)
        self._schedule_locked()

    def _release_capacity_locked(self, active: _ActiveResponse) -> None:
        if active.slot_released:
            return
        active.slot_released = True
        self._active_responses -= 1

    async def _invoke_cancel(
        self,
        active: _ActiveResponse,
        reason: str,
    ) -> None:
        source = active.source
        if active.cancel_invoked or source is None or source.cancel is None:
            return
        active.cancel_invoked = True
        result = source.cancel(reason)
        if inspect.isawaitable(result):
            await result

    async def _request_source_cancel(
        self,
        active: _ActiveResponse,
        reason: str,
    ) -> None:
        try:
            await self._invoke_cancel(active, reason)
        except Exception:
            pass
        finally:
            active.cancellation_complete.set()


def parse_websocket_command(payload: Mapping[str, Any]) -> WebSocketCommand:
    """Parse standard create/steer and isolate the Beta inject envelope."""

    command_type = payload.get("type")
    if command_type == "response.create":
        stream_id = _parse_stream_id(payload)
        body = dict(payload)
        body.pop("type", None)
        body.pop("stream_id", None)
        forbidden = next(
            (field for field in ("stream", "background") if field in body),
            None,
        )
        if forbidden is not None:
            raise TransportProtocolError(
                f"response.create does not accept transport field {forbidden!r}",
                stream_id=stream_id,
                param=forbidden,
            )
        return ResponseCreateCommand(response=body, stream_id=stream_id)

    if command_type == "response.steer":
        return _parse_response_steer(payload)

    if command_type == "response.inject":
        # Beta Multi-agent only. The adapter rejects this parsed command and
        # closes the connection until an atomic injection seam exists.
        return _parse_response_inject(payload)

    stream_id = _parse_stream_id(payload)
    raise TransportProtocolError(
        f"unsupported WebSocket command type: {command_type!r}",
        stream_id=stream_id,
        param="type",
    )


def _parse_response_steer(payload: Mapping[str, Any]) -> ResponseSteerCommand:
    _reject_extra_fields(
        payload,
        allowed=frozenset({"type", "previous_response_id", "input"}),
        event_type="response.steer",
    )
    previous_response_id = payload.get("previous_response_id")
    if not isinstance(previous_response_id, str):
        raise TransportProtocolError(
            "response.steer previous_response_id must be a string",
            param="previous_response_id",
        )

    steer_input = payload.get("input")
    if isinstance(steer_input, str):
        parsed_input: str | tuple[Mapping[str, Any], ...] = steer_input
    elif isinstance(steer_input, list | tuple) and steer_input:
        parsed_input = tuple(
            _parse_steer_message(item, index) for index, item in enumerate(steer_input)
        )
    else:
        raise TransportProtocolError(
            "response.steer input must be a string or non-empty message list",
            param="input",
        )
    return ResponseSteerCommand(
        previous_response_id=previous_response_id,
        input=parsed_input,
    )


def _parse_steer_message(item: Any, index: int) -> Mapping[str, Any]:
    param = f"input[{index}]"
    if not isinstance(item, Mapping):
        raise TransportProtocolError(
            "response.steer input items must be message objects",
            param=param,
        )
    _reject_extra_fields(
        item,
        allowed=frozenset({"type", "role", "content"}),
        event_type="response.steer message",
        param_prefix=param,
    )
    if item.get("type", "message") != "message":
        raise TransportProtocolError(
            "response.steer accepts only message input items",
            param=f"{param}.type",
        )
    if item.get("role") != "user":
        raise TransportProtocolError(
            "response.steer accepts only user messages",
            param=f"{param}.role",
        )
    content = item.get("content")
    if isinstance(content, list | tuple) and content:
        for part_index, part in enumerate(content):
            _parse_steer_content_part(part, f"{param}.content[{part_index}]")
    elif not isinstance(content, str):
        raise TransportProtocolError(
            "response.steer message content must be a string or non-empty part list",
            param=f"{param}.content",
        )
    return dict(item)


def _parse_steer_content_part(part: Any, param: str) -> None:
    if not isinstance(part, Mapping):
        raise TransportProtocolError(
            "response.steer content parts must be objects",
            param=param,
        )
    part_type = part.get("type")
    allowed_fields = {
        "input_text": frozenset({"type", "text", "prompt_cache_breakpoint"}),
        "input_image": frozenset({"type", "image_url", "file_id", "detail"}),
        "input_file": frozenset(
            {"type", "file_id", "file_data", "file_url", "filename", "detail"}
        ),
    }
    allowed = allowed_fields.get(part_type)
    if allowed is None:
        raise TransportProtocolError(
            "response.steer supports only input_text, input_image, and "
            "input_file parts",
            param=f"{param}.type",
        )
    _reject_extra_fields(
        part,
        allowed=allowed,
        event_type=f"response.steer {part_type}",
        param_prefix=param,
    )
    if part_type == "input_text":
        if not isinstance(part.get("text"), str):
            raise TransportProtocolError(
                "response.steer input_text requires string text",
                param=f"{param}.text",
            )
        return

    locator_fields = (
        ("image_url", "file_id")
        if part_type == "input_image"
        else ("file_id", "file_data", "file_url")
    )
    populated = [
        field
        for field in locator_fields
        if isinstance(part.get(field), str) and part[field]
    ]
    if len(populated) != 1:
        raise TransportProtocolError(
            f"response.steer {part_type} requires exactly one source field",
            param=param,
        )
    if part.get("detail", "auto") not in {
        "auto",
        "low",
        "high",
        "original",
    }:
        raise TransportProtocolError(
            f"response.steer {part_type} detail must be auto, low, high, or original",
            param=f"{param}.detail",
        )
    filename = part.get("filename")
    if filename is not None and (not isinstance(filename, str) or not filename):
        raise TransportProtocolError(
            "response.steer input_file filename must be a non-empty string",
            param=f"{param}.filename",
        )


def _parse_response_inject(payload: Mapping[str, Any]) -> ResponseInjectCommand:
    """Validate the Beta envelope before the unsupported path closes the socket."""

    try:
        _reject_extra_fields(
            payload,
            allowed=frozenset({"type", "response_id", "input"}),
            event_type="response.inject",
        )
        response_id = payload.get("response_id")
        if not isinstance(response_id, str) or not response_id.strip():
            raise TransportProtocolError(
                "response.inject response_id must be a non-blank string",
                param="response_id",
            )
        inject_input = payload.get("input")
        if not isinstance(inject_input, list | tuple) or not inject_input:
            raise TransportProtocolError(
                "response.inject input must be a non-empty list",
                param="input",
            )
        if not all(isinstance(item, Mapping) for item in inject_input):
            raise TransportProtocolError(
                "response.inject input items must be objects",
                param="input",
            )
    except TransportProtocolError as error:
        error.close_connection = True
        raise
    return ResponseInjectCommand(
        response_id=response_id,
        input=tuple(dict(item) for item in inject_input),
    )


def _reject_extra_fields(
    payload: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    event_type: str,
    param_prefix: str | None = None,
) -> None:
    extra = next((field for field in payload if field not in allowed), None)
    if extra is None:
        return
    param = f"{param_prefix}.{extra}" if param_prefix else extra
    raise TransportProtocolError(
        f"{event_type} does not accept field {extra!r}",
        param=param,
    )


def _parse_stream_id(payload: Mapping[str, Any]) -> StreamId | None:
    if "stream_id" not in payload:
        return _DEFAULT_LANE
    raw_stream_id = payload["stream_id"]
    if not isinstance(raw_stream_id, str):
        raise TransportProtocolError(
            "stream_id must be a string",
            param="stream_id",
        )
    try:
        return StreamId(raw_stream_id)
    except ValueError as error:
        raise TransportProtocolError(str(error), param="stream_id") from error


def _display_stream_id(stream_id: StreamId | None) -> str:
    return "default" if stream_id is None else stream_id.value
