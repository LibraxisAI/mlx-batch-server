"""RED contracts for the source-only Responses controller seam.

These tests are authored under Compile Embargo and must not be executed until
the integrator releases HOLD.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import pytest

from mlx_batch_server.responses.controller import (
    PreparedResponse,
    ResponsesController,
)
from mlx_batch_server.responses.registry import (
    ResponseRegistry,
    ResponseRegistryError,
)
from mlx_batch_server.runtime.contracts import (
    GenerationRequest,
    RuntimeKey,
    TurnSink,
)
from mlx_batch_server.runtime.events import (
    ContentPartCompleted,
    ContentPartStarted,
    OutputItemCompleted,
    OutputItemStarted,
    SequencedTurnEvent,
    TextCompleted,
    TextDelta,
    TurnCancelled,
    TurnCompleted,
    TurnFailed,
    TurnStarted,
)
from mlx_batch_server.runtime.turn import GenerationTurn

if TYPE_CHECKING:
    from mlx_batch_server.responses.transport import ResponseEventSource
    from mlx_batch_server.runtime.service import FirstWriterCancelToken

OWNER_A = "principal:a"
OWNER_B = "principal:b"


class _CountingRegistry(ResponseRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.commit_calls = 0

    def commit(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.commit_calls += 1
        return super().commit(*args, **kwargs)


class _Projection:
    def __init__(self, response_id: str) -> None:
        self.response_id = response_id
        self.events: list[SequencedTurnEvent] = []

    def observe(self, event: SequencedTurnEvent) -> None:
        self.events.append(event)

    def terminal_envelope(self) -> Mapping[str, Any]:
        terminal = self.events[-1].event
        text = "".join(
            item.event.delta
            for item in self.events
            if isinstance(item.event, TextDelta)
        )
        if isinstance(terminal, TurnCompleted):
            status = "incomplete" if terminal.finish_reason == "length" else "completed"
            error = None
        elif isinstance(terminal, TurnCancelled):
            status = "cancelled"
            error = {"code": "request_cancelled", "message": terminal.reason}
        elif isinstance(terminal, TurnFailed):
            status = "failed"
            error = {"code": terminal.code, "message": terminal.error}
        else:  # pragma: no cover - contract assertion
            raise AssertionError("terminal projection requested before terminal event")
        return {
            "id": self.response_id,
            "object": "response",
            "status": status,
            "output": [{"type": "message", "content": text}],
            "error": error,
        }


class _Mapper:
    def __init__(self) -> None:
        self.owner_ids: list[str] = []
        self.parent_messages: list[list[Mapping[str, Any]]] = []
        self.response_ids: list[str] = []
        self.projections: list[_Projection] = []

    def prepare(
        self,
        payload: Mapping[str, Any],
        *,
        response_id: str,
        owner_id: str,
        parent_messages: Sequence[Mapping[str, Any]],
    ) -> PreparedResponse:
        self.owner_ids.append(owner_id)
        self.parent_messages.append(list(parent_messages))
        self.response_ids.append(response_id)
        messages = [*parent_messages, {"role": "user", "content": payload["input"]}]
        return PreparedResponse(
            request=GenerationRequest(
                response_id=response_id,
                runtime=RuntimeKey(model_id=str(payload.get("model", "buddy"))),
                messages=messages,
            ),
            materialized_messages=messages,
            store=bool(payload.get("store", True)),
        )

    def start_projection(self, prepared: PreparedResponse) -> _Projection:
        projection = _Projection(prepared.request.response_id)
        self.projections.append(projection)
        return projection


class _Handle:
    def __init__(self, response_id: str, sink: TurnSink) -> None:
        self._response_id = response_id
        self.sink = sink
        self.closed = asyncio.Event()
        self.cancel_reasons: list[str] = []

    @property
    def response_id(self) -> str:
        return self._response_id

    def cancel(self, reason: str) -> None:
        self.cancel_reasons.append(reason)
        self.sink.emit(TurnCancelled(reason))
        self.closed.set()

    async def wait_closed(self) -> None:
        await self.closed.wait()


class _Starter:
    def __init__(self, *, blocked: bool = False, fail: bool = False) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        if not blocked:
            self.release.set()
        self.fail = fail
        self.handle: _Handle | None = None

    async def start(
        self,
        request: GenerationRequest,
        sink: TurnSink,
        *,
        cancel: FirstWriterCancelToken,
    ) -> _Handle:
        self.entered.set()
        await self.release.wait()
        if cancel.cancelled:
            raise asyncio.CancelledError(cancel.reason)
        if self.fail:
            raise RuntimeError("backend refused request")
        sink.emit(
            TurnStarted(
                response_id=request.response_id,
                model=request.runtime.model_id,
                created_at=7,
            )
        )
        self.handle = _Handle(request.response_id, sink)
        return self.handle

    def complete(self, text: str = "ok", *, finish_reason: str = "stop") -> None:
        assert self.handle is not None
        item_id = f"msg_{self.handle.response_id}"
        self.handle.sink.emit(OutputItemStarted("message", 0, item_id))
        self.handle.sink.emit(ContentPartStarted("output_text", 0, 0, item_id))
        self.handle.sink.emit(TextDelta(text, item_id, 0, 0))
        self.handle.sink.emit(TextCompleted(text, item_id, 0, 0))
        self.handle.sink.emit(ContentPartCompleted("output_text", 0, 0, item_id, text))
        self.handle.sink.emit(
            OutputItemCompleted(
                "message",
                0,
                item_id,
                text=text,
                status="incomplete" if finish_reason == "length" else "completed",
            )
        )
        self.handle.sink.emit(TurnCompleted(finish_reason))
        self.handle.closed.set()


class _CountingTurn(GenerationTurn):
    def __init__(self) -> None:
        super().__init__(max_pending_events=16)
        self.subscribe_calls = 0

    def subscribe(
        self,
        *,
        max_pending_events: int | None = None,
    ) -> AsyncIterator[SequencedTurnEvent]:
        self.subscribe_calls += 1
        return super().subscribe(max_pending_events=max_pending_events)


async def _collect(source: ResponseEventSource) -> list[SequencedTurnEvent]:
    async def drain() -> list[SequencedTurnEvent]:
        observed: list[SequencedTurnEvent] = []
        async for event in source.events:
            assert isinstance(event, SequencedTurnEvent)
            observed.append(event)
        return observed

    return await asyncio.wait_for(drain(), timeout=1.0)


@pytest.mark.asyncio
async def test_create_uses_stable_owner_one_turn_subscription_and_one_commit() -> None:
    registry = _CountingRegistry()
    mapper = _Mapper()
    starter = _Starter()
    turns: list[_CountingTurn] = []

    def turn_factory() -> _CountingTurn:
        turn = _CountingTurn()
        turns.append(turn)
        return turn

    controller = ResponsesController(
        registry=registry,
        mapper=mapper,
        starter=starter,
        max_pending_events=16,
        turn_factory=turn_factory,
    )
    source = await controller.create(
        {"model": "buddy", "input": "hello", "owner_id": OWNER_B},
        owner_id=OWNER_A,
    )
    await starter.entered.wait()
    await asyncio.sleep(0)
    starter.complete("beautiful output")
    observed = await _collect(source)
    assert source.terminal_response is not None
    terminal_response = await source.terminal_response

    response_id = mapper.response_ids[0]
    assert mapper.owner_ids == [OWNER_A]
    assert turns[0].subscribe_calls == 1
    assert [type(item.event) for item in observed] == [
        TurnStarted,
        OutputItemStarted,
        ContentPartStarted,
        TextDelta,
        TextCompleted,
        ContentPartCompleted,
        OutputItemCompleted,
        TurnCompleted,
    ]
    assert registry.get(response_id, owner_id=OWNER_A)["status"] == "completed"
    assert terminal_response == registry.get(response_id, owner_id=OWNER_A)
    assert registry.commit_calls == 1
    with pytest.raises(ResponseRegistryError) as foreign:
        registry.get(response_id, owner_id=OWNER_B)
    assert foreign.value.code == "response_not_found"


@pytest.mark.asyncio
async def test_cancel_during_runtime_start_reaches_bridge_and_commits_terminal() -> (
    None
):
    registry = _CountingRegistry()
    mapper = _Mapper()
    starter = _Starter(blocked=True)
    controller = ResponsesController(
        registry=registry,
        mapper=mapper,
        starter=starter,
        max_pending_events=16,
    )
    source = await controller.create(
        {"model": "buddy", "input": "stop"},
        owner_id=OWNER_A,
    )
    await starter.entered.wait()

    assert source.cancel is not None
    source.cancel("transport_disconnected")
    source.cancel("later_reason")
    with pytest.raises(ResponseRegistryError) as foreign:
        controller.cancel(
            mapper.response_ids[0],
            owner_id=OWNER_B,
            reason="foreign_cancel",
        )
    assert foreign.value.code == "response_not_found"

    starter.release.set()
    observed = await _collect(source)
    assert starter.handle is None
    assert isinstance(observed[-1].event, TurnCancelled)
    stored = registry.get(mapper.response_ids[0], owner_id=OWNER_A)
    assert stored["status"] == "cancelled"
    assert stored["error"]["message"] == "transport_disconnected"
    assert registry.commit_calls == 1


@pytest.mark.asyncio
async def test_terminal_commit_does_not_depend_on_transport_consuming_source() -> None:
    registry = _CountingRegistry()
    mapper = _Mapper()
    starter = _Starter()
    controller = ResponsesController(
        registry=registry,
        mapper=mapper,
        starter=starter,
        max_pending_events=16,
    )
    source = await controller.create(
        {"model": "buddy", "input": "hello"},
        owner_id=OWNER_A,
    )
    await starter.entered.wait()
    await asyncio.sleep(0)
    starter.complete()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    response_id = mapper.response_ids[0]
    assert registry.get(response_id, owner_id=OWNER_A)["status"] == "completed"
    assert registry.commit_calls == 1
    assert isinstance((await _collect(source))[-1].event, TurnCompleted)


@pytest.mark.asyncio
async def test_incomplete_terminal_is_committed_and_exposed() -> None:
    registry = _CountingRegistry()
    mapper = _Mapper()
    starter = _Starter()
    controller = ResponsesController(
        registry=registry,
        mapper=mapper,
        starter=starter,
        max_pending_events=16,
    )
    source = await controller.create(
        {"model": "buddy", "input": "bounded"},
        owner_id=OWNER_A,
    )
    await starter.entered.wait()
    await asyncio.sleep(0)
    starter.complete("cut off", finish_reason="length")

    terminal = await source.terminal_response
    response_id = mapper.response_ids[0]
    assert terminal["status"] == "incomplete"
    assert registry.get(response_id, owner_id=OWNER_A)["status"] == "incomplete"
    assert registry.commit_calls == 1
    assert isinstance((await _collect(source))[-1].event, TurnCompleted)


@pytest.mark.asyncio
async def test_store_false_still_exposes_the_complete_terminal_response() -> None:
    registry = _CountingRegistry()
    mapper = _Mapper()
    starter = _Starter()
    controller = ResponsesController(
        registry=registry,
        mapper=mapper,
        starter=starter,
        max_pending_events=16,
    )
    source = await controller.create(
        {"model": "buddy", "input": "hello", "store": False},
        owner_id=OWNER_A,
    )
    await starter.entered.wait()
    await asyncio.sleep(0)
    starter.complete("ephemeral but complete")

    assert source.terminal_response is not None
    terminal = await source.terminal_response
    assert terminal["status"] == "completed"
    assert terminal["output"] == [
        {"type": "message", "content": "ephemeral but complete"}
    ]
    with pytest.raises(ResponseRegistryError):
        registry.get(mapper.response_ids[-1], owner_id=OWNER_A)


@pytest.mark.asyncio
async def test_runtime_start_failure_uses_the_same_turn_and_terminal_commit_path() -> (
    None
):
    registry = _CountingRegistry()
    mapper = _Mapper()
    starter = _Starter(fail=True)
    controller = ResponsesController(
        registry=registry,
        mapper=mapper,
        starter=starter,
        max_pending_events=16,
    )
    source = await controller.create(
        {"model": "buddy", "input": "hello"},
        owner_id=OWNER_A,
    )
    observed = await _collect(source)

    assert [type(item.event) for item in observed] == [TurnFailed]
    assert observed[0].event.code == "runtime_start_failed"
    stored = registry.get(mapper.response_ids[0], owner_id=OWNER_A)
    assert stored["status"] == "failed"
    assert registry.commit_calls == 1


@pytest.mark.asyncio
async def test_shutdown_cancels_and_drains_active_response_before_closing() -> None:
    registry = _CountingRegistry()
    mapper = _Mapper()
    starter = _Starter()
    controller = ResponsesController(
        registry=registry,
        mapper=mapper,
        starter=starter,
        max_pending_events=16,
    )
    source = await controller.create(
        {"model": "buddy", "input": "long response"},
        owner_id=OWNER_A,
    )
    await starter.entered.wait()
    await asyncio.sleep(0)

    await controller.shutdown(timeout_s=1.0)

    assert starter.handle is not None
    assert starter.handle.cancel_reasons == ["registry_shutdown"]
    assert registry.stats()["in_flight"] == 0
    assert isinstance((await _collect(source))[-1].event, TurnCancelled)
    with pytest.raises(RuntimeError, match="shutting down"):
        await controller.create(
            {"model": "buddy", "input": "too late"},
            owner_id=OWNER_A,
        )
    await controller.shutdown(timeout_s=0.0)


@pytest.mark.asyncio
async def test_shutdown_timeout_is_retryable_without_detaching_startup() -> None:
    registry = _CountingRegistry()
    mapper = _Mapper()
    starter = _Starter(blocked=True)
    controller = ResponsesController(
        registry=registry,
        mapper=mapper,
        starter=starter,
        max_pending_events=16,
    )
    source = await controller.create(
        {"model": "buddy", "input": "slow preparation"},
        owner_id=OWNER_A,
    )
    await starter.entered.wait()

    with pytest.raises(TimeoutError, match="shutdown timed out"):
        await controller.shutdown(timeout_s=0.0)

    starter.release.set()
    assert isinstance((await _collect(source))[-1].event, TurnCancelled)
    await controller.shutdown(timeout_s=1.0)
