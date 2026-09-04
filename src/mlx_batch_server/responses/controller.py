"""Target-owned orchestration between Responses state and the runtime turn."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Mapping,
    Sequence,
)
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from ..runtime.events import (
    TERMINAL_EVENT_TYPES,
    SequencedTurnEvent,
    TurnCancelled,
    TurnFailed,
    TurnStarted,
)
from ..runtime.service import FirstWriterCancelToken
from ..runtime.turn import GenerationTurn, TurnState
from .registry import ResponseRegistry, ResponseRegistryError
from .transport import ResponseEventSource

if TYPE_CHECKING:
    from ..runtime.contracts import BackendTurn, GenerationRequest, TurnSink


@dataclass(frozen=True, slots=True)
class PreparedResponse:
    """Protocol-neutral runtime input and the lineage state committed with it."""

    request: GenerationRequest
    materialized_messages: Sequence[Mapping[str, Any]]
    store: bool = True
    cancel_on_disconnect: bool = True


@runtime_checkable
class ResponseProjection(Protocol):
    """Per-response accumulator driven only by canonical turn events."""

    def observe(self, event: SequencedTurnEvent) -> None: ...

    def terminal_envelope(self) -> Mapping[str, Any]: ...


@runtime_checkable
class ResponseMapper(Protocol):
    """Translate a wire request and create its isolated response projection."""

    def prepare(
        self,
        payload: Mapping[str, Any],
        *,
        response_id: str,
        owner_id: str,
        parent_messages: Sequence[Mapping[str, Any]],
    ) -> PreparedResponse: ...

    def start_projection(
        self,
        prepared: PreparedResponse,
    ) -> ResponseProjection: ...


@runtime_checkable
class RuntimeStarter(Protocol):
    """Start one runtime turn against the canonical turn-event sink."""

    def start(
        self,
        request: GenerationRequest,
        sink: TurnSink,
        *,
        cancel: FirstWriterCancelToken,
    ) -> Awaitable[BackendTurn]: ...


_END = object()
_MailboxItem = SequencedTurnEvent | BaseException | object


class _TerminalResponseAwaitable:
    """Keep consumer cancellation from cancelling controller-owned finality."""

    def __init__(self, future: asyncio.Future[Mapping[str, Any]]) -> None:
        self._future = future

    def __await__(self):
        return asyncio.shield(self._future).__await__()


class _EventMailbox:
    """Single-consumer bounded relay preserving canonical event objects."""

    def __init__(self, max_pending_events: int) -> None:
        self._event_limit = max_pending_events
        self._queue: asyncio.Queue[_MailboxItem] = asyncio.Queue(
            maxsize=max_pending_events + 2
        )
        self._producer_closed = False
        self._consumer_closed = False
        self._claimed = False

    @property
    def producer_closed(self) -> bool:
        return self._producer_closed

    def publish(self, item: SequencedTurnEvent) -> bool:
        if self._producer_closed:
            return False
        terminal = isinstance(item.event, TERMINAL_EVENT_TYPES)
        if not terminal and self._queue.qsize() >= self._event_limit:
            return False
        self._queue.put_nowait(item)
        if terminal:
            self._producer_closed = True
            self._queue.put_nowait(_END)
        return True

    def fail(self, error: BaseException) -> None:
        if self._producer_closed:
            return
        self._producer_closed = True
        self._queue.put_nowait(error)
        self._queue.put_nowait(_END)

    def __aiter__(self) -> AsyncIterator[SequencedTurnEvent]:
        if self._claimed:
            raise RuntimeError("response event source may only be consumed once")
        self._claimed = True
        return self

    async def __anext__(self) -> SequencedTurnEvent:
        if self._consumer_closed:
            raise StopAsyncIteration
        item = await self._queue.get()
        if item is _END:
            self._consumer_closed = True
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            self._consumer_closed = True
            raise item
        if not isinstance(item, SequencedTurnEvent):  # pragma: no cover - invariant
            raise RuntimeError("invalid controller mailbox item")
        return item


@dataclass(slots=True)
class _ResponseLifecycle:
    response_id: str
    owner_id: str
    prepared: PreparedResponse
    projection: ResponseProjection
    turn: GenerationTurn
    mailbox: _EventMailbox
    terminal_response: asyncio.Future[Mapping[str, Any]]
    terminal_lock: threading.Lock = field(default_factory=threading.Lock)
    terminal_committed: bool = False
    overflow_cancel_requested: bool = False
    cancel_token: FirstWriterCancelToken = field(default_factory=FirstWriterCancelToken)


class ResponsesController:
    """Own one create/cancel lifecycle without owning HTTP or WebSocket policy."""

    def __init__(
        self,
        *,
        registry: ResponseRegistry,
        mapper: ResponseMapper,
        starter: RuntimeStarter,
        max_pending_events: int = 4096,
        turn_factory: Callable[[], GenerationTurn] | None = None,
    ) -> None:
        if max_pending_events < 2:
            raise ValueError("max_pending_events must be at least 2")
        self._registry = registry
        self._mapper = mapper
        self._starter = starter
        self._max_pending_events = int(max_pending_events)
        self._turn_factory = turn_factory
        self._tasks: set[asyncio.Task[None]] = set()
        self._shutdown_lock = asyncio.Lock()
        self._closing = False
        self._closed = False

    async def create(
        self,
        payload: Mapping[str, Any],
        *,
        owner_id: str,
    ) -> ResponseEventSource:
        """Register and start one response, returning the shared SSE/WSS source."""

        if self._closing or self._closed:
            raise RuntimeError("responses controller is shutting down")
        owner = self._require_owner_id(owner_id)
        if not isinstance(payload, Mapping):
            raise TypeError("response payload must be a mapping")
        request_payload = dict(payload)
        parent_messages = self._parent_messages(request_payload, owner)
        response_id = self._registry.allocate_id()
        prepared = self._normalize_prepared(
            self._mapper.prepare(
                request_payload,
                response_id=response_id,
                owner_id=owner,
                parent_messages=parent_messages,
            ),
            response_id,
        )
        projection = self._mapper.start_projection(prepared)
        turn = (
            self._turn_factory()
            if self._turn_factory is not None
            else GenerationTurn(max_pending_events=self._max_pending_events)
        )
        subscription = turn.subscribe(max_pending_events=self._max_pending_events)
        lifecycle = _ResponseLifecycle(
            response_id=response_id,
            owner_id=owner,
            prepared=prepared,
            projection=projection,
            turn=turn,
            mailbox=_EventMailbox(self._max_pending_events),
            terminal_response=asyncio.get_running_loop().create_future(),
        )

        self._registry.begin(
            response_id,
            owner_id=owner,
            store=prepared.store,
            materialized_messages=prepared.materialized_messages,
            cancel=lifecycle.cancel_token.cancel,
        )
        self._track(self._relay(lifecycle, subscription), "relay", response_id)
        self._track(self._drive_runtime(lifecycle), "runtime", response_id)

        def cancel(reason: str) -> None:
            self.cancel(response_id, owner_id=owner, reason=reason)

        return ResponseEventSource(
            events=lifecycle.mailbox,
            cancel=cancel,
            cancel_on_disconnect=prepared.cancel_on_disconnect,
            terminal_response=_TerminalResponseAwaitable(lifecycle.terminal_response),
        )

    def cancel(self, response_id: str, *, owner_id: str, reason: str) -> None:
        """Record one owned cancel intent; registry binding handles early races."""

        self._registry.request_cancel(
            response_id,
            reason,
            owner_id=self._require_owner_id(owner_id),
        )

    async def shutdown(self, *, timeout_s: float = 1.0) -> None:
        """Cancel and drain all controller-owned responses without detaching work."""

        if timeout_s < 0:
            raise ValueError("timeout_s must be non-negative")
        async with self._shutdown_lock:
            if self._closed:
                return
            self._closing = True
            loop = asyncio.get_running_loop()
            deadline_at = loop.time() + timeout_s
            waiters = self._registry.request_shutdown()
            remaining = max(0.0, deadline_at - loop.time())
            settled = await asyncio.to_thread(
                self._registry.wait_for_shutdown,
                waiters,
                remaining,
            )
            if not settled:
                raise TimeoutError("responses controller shutdown timed out")

            tasks = tuple(self._tasks)
            if tasks:
                remaining = max(0.0, deadline_at - loop.time())
                _, pending = await asyncio.wait(tasks, timeout=remaining)
                if pending:
                    raise TimeoutError("responses controller task drain timed out")
            self._closed = True

    async def _relay(
        self,
        lifecycle: _ResponseLifecycle,
        subscription: AsyncIterator[SequencedTurnEvent],
    ) -> None:
        try:
            async for item in subscription:
                lifecycle.projection.observe(item)
                if isinstance(item.event, TERMINAL_EVENT_TYPES):
                    self._commit_terminal(lifecycle)
                if lifecycle.mailbox.publish(item):
                    continue
                if not lifecycle.overflow_cancel_requested:
                    lifecycle.overflow_cancel_requested = True
                    self.cancel(
                        lifecycle.response_id,
                        owner_id=lifecycle.owner_id,
                        reason="controller_event_backpressure",
                    )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            with contextlib.suppress(ResponseRegistryError):
                self.cancel(
                    lifecycle.response_id,
                    owner_id=lifecycle.owner_id,
                    reason="controller_relay_failed",
                )
            if not lifecycle.terminal_response.done():
                lifecycle.terminal_response.set_exception(error)
            lifecycle.mailbox.fail(error)
        else:
            if not lifecycle.mailbox.producer_closed:
                error = RuntimeError("generation turn ended without a terminal event")
                if not lifecycle.terminal_response.done():
                    lifecycle.terminal_response.set_exception(error)
                lifecycle.mailbox.fail(error)

    async def _drive_runtime(self, lifecycle: _ResponseLifecycle) -> None:
        try:
            handle = await self._starter.start(
                lifecycle.prepared.request,
                lifecycle.turn,
                cancel=lifecycle.cancel_token,
            )
            if handle.response_id != lifecycle.response_id:
                handle.cancel("runtime_response_id_mismatch")
                raise RuntimeError(
                    "runtime turn returned a response id that does not match its request"
                )
            if lifecycle.turn.state is not TurnState.TERMINAL:
                bound = self._registry.bind_cancel(
                    lifecycle.response_id,
                    handle.cancel,
                    owner_id=lifecycle.owner_id,
                )
                if not bound:
                    handle.cancel("response_registry_unavailable")
            await handle.wait_closed()
            await asyncio.sleep(0)
            if lifecycle.turn.state is not TurnState.TERMINAL:
                lifecycle.turn.fail(
                    TurnFailed(
                        error="runtime turn closed without a terminal event",
                        code="runtime_missing_terminal",
                        status_code=500,
                    )
                )
        except asyncio.CancelledError:
            if lifecycle.cancel_token.cancelled:
                if lifecycle.turn.state is not TurnState.TERMINAL:
                    if lifecycle.turn.state is TurnState.IDLE:
                        lifecycle.turn.emit(
                            TurnStarted(
                                response_id=lifecycle.response_id,
                                model=lifecycle.prepared.request.runtime.model_id,
                                created_at=int(time.time()),
                            )
                        )
                    lifecycle.turn.cancel(
                        TurnCancelled(
                            lifecycle.cancel_token.reason or "client_cancelled"
                        )
                    )
                return
            if lifecycle.turn.state is not TurnState.TERMINAL:
                lifecycle.turn.fail(
                    TurnFailed(
                        error="response controller stopped before runtime completion",
                        code="controller_stopped",
                        status_code=503,
                    )
                )
            raise
        except Exception as error:
            if lifecycle.turn.state is not TurnState.TERMINAL:
                lifecycle.turn.fail(
                    TurnFailed(
                        error=str(error) or type(error).__name__,
                        code="runtime_start_failed",
                        status_code=500,
                    )
                )

    def _commit_terminal(self, lifecycle: _ResponseLifecycle) -> None:
        with lifecycle.terminal_lock:
            if lifecycle.terminal_committed:
                return
            envelope = lifecycle.projection.terminal_envelope()
            if envelope.get("id") != lifecycle.response_id:
                raise ValueError("terminal response envelope must preserve response_id")
            status = envelope.get("status")
            if status not in {"completed", "incomplete", "failed", "cancelled"}:
                raise ValueError("terminal response envelope must have terminal status")
            self._registry.commit(
                lifecycle.response_id,
                envelope,
                owner_id=lifecycle.owner_id,
                materialized_messages=lifecycle.prepared.materialized_messages,
            )
            lifecycle.terminal_committed = True
            lifecycle.terminal_response.set_result(envelope)

    def _parent_messages(
        self,
        payload: Mapping[str, Any],
        owner_id: str,
    ) -> Sequence[Mapping[str, Any]]:
        previous_response_id = payload.get("previous_response_id")
        if previous_response_id is None:
            return ()
        if (
            not isinstance(previous_response_id, str)
            or not previous_response_id.strip()
        ):
            raise ResponseRegistryError(
                "previous_response_id must be a non-empty string",
                code="invalid_previous_response_id",
                status_code=400,
                param="previous_response_id",
            )
        return self._registry.parent_messages(
            previous_response_id.strip(),
            owner_id=owner_id,
        )

    @staticmethod
    def _normalize_prepared(
        prepared: PreparedResponse,
        response_id: str,
    ) -> PreparedResponse:
        if not isinstance(prepared, PreparedResponse):
            raise TypeError("response mapper must return PreparedResponse")
        if prepared.request.response_id != response_id:
            raise ValueError("mapped generation request must preserve response_id")
        messages: list[Mapping[str, Any]] = []
        for message in prepared.materialized_messages:
            if not isinstance(message, Mapping):
                raise TypeError("materialized messages must be mappings")
            messages.append(dict(message))
        return PreparedResponse(
            request=prepared.request,
            materialized_messages=tuple(messages),
            store=bool(prepared.store),
            cancel_on_disconnect=bool(prepared.cancel_on_disconnect),
        )

    @staticmethod
    def _require_owner_id(owner_id: str) -> str:
        if not isinstance(owner_id, str) or not owner_id.strip():
            raise ResponseRegistryError(
                "owner_id must be a non-empty string",
                code="invalid_owner_id",
                status_code=400,
                param="owner_id",
            )
        return owner_id.strip()

    def _track(
        self,
        coroutine: Coroutine[Any, Any, None],
        role: str,
        response_id: str,
    ) -> None:
        task = asyncio.create_task(
            coroutine,
            name=f"responses-controller-{role}-{response_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task: asyncio.Task[None]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        task.exception()
