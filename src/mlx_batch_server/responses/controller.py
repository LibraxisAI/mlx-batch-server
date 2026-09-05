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
    TurnEvent,
    TurnFailed,
    TurnStarted,
)
from ..runtime.service import FirstWriterCancelToken
from ..runtime.turn import GenerationTurn, TurnState
from .registry import ResponseRegistry, ResponseRegistryError
from .request_contract import LOCAL_FIELD_NAMES
from .transport import (
    SNAPSHOT_EMBEDDING_EVENT_TYPES,
    PublishedResponseEvent,
    ResponseEventSource,
    ResponseSnapshotBuilder,
    ResponseSnapshotIdentity,
    build_response_snapshot,
    request_settings_from,
)

if TYPE_CHECKING:
    from ..runtime.contracts import BackendTurn, GenerationRequest, TurnSink


@dataclass(frozen=True, slots=True)
class PreparedResponse:
    """Protocol-neutral runtime input and the lineage state committed with it."""

    request: GenerationRequest
    materialized_messages: Sequence[Mapping[str, Any]]
    lineage_messages: Sequence[Mapping[str, Any]] | None = None
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
_MailboxItem = PublishedResponseEvent | BaseException | object


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

    def publish(self, item: PublishedResponseEvent) -> bool:
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

    def __aiter__(self) -> AsyncIterator[PublishedResponseEvent]:
        if self._claimed:
            raise RuntimeError("response event source may only be consumed once")
        self._claimed = True
        return self

    async def __anext__(self) -> PublishedResponseEvent:
        if self._consumer_closed:
            raise StopAsyncIteration
        item = await self._queue.get()
        if item is _END:
            self._consumer_closed = True
            raise StopAsyncIteration
        if isinstance(item, BaseException):
            self._consumer_closed = True
            raise item
        if not isinstance(item, PublishedResponseEvent):  # pragma: no cover
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
    background: bool
    created_at: int
    public_model: str
    request_settings: Mapping[str, Any]
    snapshot_builder: ResponseSnapshotBuilder
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

        source, _ = self._start_response(payload, owner_id=owner_id)
        return source

    async def create_background(
        self,
        payload: Mapping[str, Any],
        *,
        owner_id: str,
    ) -> Mapping[str, Any]:
        """Start one deferred response and return before its runtime completes."""

        source, lifecycle = self._start_response(
            payload,
            owner_id=owner_id,
            require_background=True,
        )
        self._track(
            self._drain_background(source),
            "background",
            lifecycle.response_id,
        )
        return self._registry.get(
            lifecycle.response_id,
            owner_id=lifecycle.owner_id,
        )

    def _start_response(
        self,
        payload: Mapping[str, Any],
        *,
        owner_id: str,
        require_background: bool = False,
    ) -> tuple[ResponseEventSource, _ResponseLifecycle]:
        """Create the single controller-owned lifecycle used by every transport."""

        if self._closing or self._closed:
            raise RuntimeError("responses controller is shutting down")
        owner = self._require_owner_id(owner_id)
        prepared = self.inspect(payload, owner_id=owner)
        response_id = prepared.request.response_id
        projection = self._mapper.start_projection(prepared)
        turn = (
            self._turn_factory()
            if self._turn_factory is not None
            else GenerationTurn(max_pending_events=self._max_pending_events)
        )
        subscription = turn.subscribe(max_pending_events=self._max_pending_events)
        background = self._background_setting(prepared)
        if require_background and not background:
            raise ValueError("create_background requires background=true")
        created_at = int(time.time())
        public_model = self._public_model(prepared)
        request_settings = self._public_request_settings(prepared)
        snapshot_builder = ResponseSnapshotBuilder()
        snapshot_builder.observe(
            TurnStarted(
                response_id=response_id,
                model=prepared.request.runtime.model_id,
                created_at=created_at,
                requested_model=public_model,
                request_settings=request_settings,
            )
        )
        initial_snapshot = build_response_snapshot(
            identity=ResponseSnapshotIdentity(
                response_id=response_id,
                created_at=created_at,
                public_model=public_model,
                physical_model=prepared.request.runtime.model_id,
            ),
            status="queued" if background else "in_progress",
            request_settings=request_settings,
        )
        if background:
            initial_snapshot["background"] = True
        lifecycle = _ResponseLifecycle(
            response_id=response_id,
            owner_id=owner,
            prepared=prepared,
            projection=projection,
            turn=turn,
            mailbox=_EventMailbox(self._max_pending_events),
            terminal_response=asyncio.get_running_loop().create_future(),
            background=background,
            created_at=created_at,
            public_model=public_model,
            request_settings=request_settings,
            snapshot_builder=snapshot_builder,
        )

        self._registry.begin(
            response_id,
            owner_id=owner,
            store=prepared.store,
            materialized_messages=(
                prepared.lineage_messages
                if prepared.lineage_messages is not None
                else prepared.materialized_messages
            ),
            cancel=lifecycle.cancel_token.cancel,
            background=True if background else None,
            public_snapshot=initial_snapshot,
        )
        self._track(self._relay(lifecycle, subscription), "relay", response_id)
        self._track(self._drive_runtime(lifecycle), "runtime", response_id)

        def cancel(reason: str) -> None:
            self.cancel(response_id, owner_id=owner, reason=reason)

        source = ResponseEventSource(
            events=lifecycle.mailbox,
            response_id=response_id,
            cancel=cancel,
            cancel_on_disconnect=prepared.cancel_on_disconnect,
            terminal_response=_TerminalResponseAwaitable(lifecycle.terminal_response),
        )
        return source, lifecycle

    def inspect(
        self,
        payload: Mapping[str, Any],
        *,
        owner_id: str,
    ) -> PreparedResponse:
        """Map an owned request without registering or starting a generation turn."""

        if self._closing or self._closed:
            raise RuntimeError("responses controller is shutting down")
        owner = self._require_owner_id(owner_id)
        if not isinstance(payload, Mapping):
            raise TypeError("response payload must be a mapping")
        request_payload = dict(payload)
        parent_messages = self._parent_messages(request_payload, owner)
        response_id = self._registry.allocate_id()
        return self._normalize_prepared(
            self._mapper.prepare(
                request_payload,
                response_id=response_id,
                owner_id=owner,
                parent_messages=parent_messages,
            ),
            response_id,
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
                snapshot = self._fold_snapshot(lifecycle, item.event)
                if isinstance(item.event, TERMINAL_EVENT_TYPES):
                    # Registry commit and terminal_response resolve before the
                    # terminal is published: first-writer truth stays intact.
                    self._commit_terminal(lifecycle)
                elif snapshot is not None:
                    self._registry.update(
                        lifecycle.response_id,
                        snapshot,
                        owner_id=lifecycle.owner_id,
                    )
                published = PublishedResponseEvent(
                    sequence_number=item.sequence_number,
                    event=item.event,
                    snapshot=snapshot,
                )
                if lifecycle.mailbox.publish(published):
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
            envelope = self._normalize_terminal_snapshot(
                lifecycle,
                lifecycle.projection.terminal_envelope(),
            )
            if envelope.get("id") != lifecycle.response_id:
                raise ValueError("terminal response envelope must preserve response_id")
            status = envelope.get("status")
            if status not in {"completed", "incomplete", "failed", "cancelled"}:
                raise ValueError("terminal response envelope must have terminal status")
            self._registry.commit(
                lifecycle.response_id,
                envelope,
                owner_id=lifecycle.owner_id,
                materialized_messages=(
                    lifecycle.prepared.lineage_messages
                    if lifecycle.prepared.lineage_messages is not None
                    else lifecycle.prepared.materialized_messages
                ),
            )
            lifecycle.terminal_committed = True
            lifecycle.terminal_response.set_result(envelope)

    async def _drain_background(self, source: ResponseEventSource) -> None:
        """Consume an unstreamed background mailbox under controller ownership."""

        terminal_seen = False
        async for item in source.events:
            if not isinstance(item, PublishedResponseEvent):
                raise TypeError(
                    "controller source must yield published response events"
                )
            if isinstance(item.event, TERMINAL_EVENT_TYPES):
                terminal_seen = True
        if not terminal_seen:
            raise RuntimeError("background response ended without a terminal event")
        if source.terminal_response is None:
            raise RuntimeError("background response is missing its terminal receipt")
        await source.terminal_response

    def _fold_snapshot(
        self,
        lifecycle: _ResponseLifecycle,
        event: TurnEvent,
    ) -> Mapping[str, Any] | None:
        """Fold one canonical event into the single per-response builder.

        Returns the complete snapshot for response-embedding events and None
        otherwise. Registry and stream both consume this one fold, so they can
        never disagree about the live response truth.
        """

        if isinstance(event, TurnStarted):
            lifecycle.snapshot_builder.observe(
                TurnStarted(
                    response_id=lifecycle.response_id,
                    model=lifecycle.prepared.request.runtime.model_id,
                    created_at=lifecycle.created_at,
                    requested_model=lifecycle.public_model,
                    request_settings=lifecycle.request_settings,
                )
            )
        else:
            lifecycle.snapshot_builder.observe(event)
        if not isinstance(event, SNAPSHOT_EMBEDDING_EVENT_TYPES):
            return None
        snapshot = lifecycle.snapshot_builder.snapshot(event)
        if snapshot is None:  # pragma: no cover - builder is seeded at start
            raise RuntimeError("response snapshot requested before its identity")
        if lifecycle.background:
            snapshot["background"] = True
        return snapshot

    @staticmethod
    def _background_setting(prepared: PreparedResponse) -> bool:
        value = prepared.request.metadata.get("background", False)
        if not isinstance(value, bool):
            raise TypeError("prepared background setting must be a boolean")
        return value

    @staticmethod
    def _public_model(prepared: PreparedResponse) -> str:
        value = prepared.request.metadata.get("requested_model")
        if value is None:
            return prepared.request.runtime.model_id
        if not isinstance(value, str) or not value.strip():
            raise ValueError("prepared requested_model must not be blank")
        return value

    @staticmethod
    def _public_request_settings(prepared: PreparedResponse) -> Mapping[str, Any]:
        metadata = {
            key: value
            for key, value in prepared.request.metadata.items()
            if key not in LOCAL_FIELD_NAMES
        }
        return request_settings_from(
            tools=prepared.request.tools,
            sampling=prepared.request.sampling,
            reasoning=prepared.request.reasoning,
            metadata=metadata,
        )

    @staticmethod
    def _normalize_terminal_snapshot(
        lifecycle: _ResponseLifecycle,
        envelope: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        normalized = dict(envelope)
        normalized["created_at"] = lifecycle.created_at
        normalized["model"] = lifecycle.public_model
        normalized["background"] = lifecycle.background
        metadata = normalized.get("metadata")
        if isinstance(metadata, Mapping):
            public_metadata = {
                key: value
                for key, value in metadata.items()
                if key not in LOCAL_FIELD_NAMES
            }
            if public_metadata:
                normalized["metadata"] = public_metadata
            else:
                normalized.pop("metadata", None)
        return normalized

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
        lineage_messages: list[Mapping[str, Any]] | None = None
        if prepared.lineage_messages is not None:
            lineage_messages = []
            for message in prepared.lineage_messages:
                if not isinstance(message, Mapping):
                    raise TypeError("lineage messages must be mappings")
                lineage_messages.append(dict(message))
        return PreparedResponse(
            request=prepared.request,
            materialized_messages=tuple(messages),
            lineage_messages=(
                tuple(lineage_messages) if lineage_messages is not None else None
            ),
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
