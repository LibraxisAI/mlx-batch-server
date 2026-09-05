"""RED contracts for the explicit legacy MLX fallback boundary."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import pytest

from mlx_batch_server.runtime.backends.legacy_mlx import (
    LegacyBackendSelectionError,
    LegacyCapability,
    LegacyMlxBackend,
    LegacyPortContractError,
)
from mlx_batch_server.runtime.contracts import (
    BackendKind,
    CancelToken,
    GenerationRequest,
    LoadConfig,
    ModelSpec,
    RuntimeKey,
)
from mlx_batch_server.runtime.events import (
    REASONING_CONTENT_KIND,
    TEXT_CONTENT_KIND,
    ContentPartCompleted,
    ContentPartStarted,
    OutputItemCompleted,
    OutputItemStarted,
    ProgressUpdate,
    ReasoningCompleted,
    ReasoningDelta,
    TextCompleted,
    TextDelta,
    ToolCompleted,
    ToolDelta,
    TurnCancelled,
    TurnCompleted,
    TurnEvent,
    TurnFailed,
    UsageUpdate,
)
from mlx_batch_server.runtime.turn import GenerationTurn


class _CancelToken:
    def __init__(self) -> None:
        self._reason: str | None = None

    @property
    def cancelled(self) -> bool:
        return self._reason is not None

    @property
    def reason(self) -> str | None:
        return self._reason

    def cancel(self, reason: str) -> bool:
        if self._reason is not None:
            return False
        self._reason = reason
        return True


class _Port:
    def __init__(
        self,
        runtime_key: RuntimeKey,
        events: Sequence[TurnEvent] = (),
        *,
        block: bool = False,
        cancel_accepted: bool = True,
        close_failures: int = 0,
        block_close: bool = False,
        cancel_delay_s: float = 0.0,
    ) -> None:
        self._runtime_key = runtime_key
        self._events = tuple(events)
        self._block = block
        self._cancel_accepted = cancel_accepted
        self._close_failures = close_failures
        self._cancel_delay_s = cancel_delay_s
        self._release = asyncio.Event()
        self._close_release = asyncio.Event()
        if not block_close:
            self._close_release.set()
        self.cancel_calls: list[tuple[str, str]] = []
        self.close_calls: list[float] = []

    @property
    def runtime_key(self) -> RuntimeKey:
        return self._runtime_key

    async def events(
        self,
        request: GenerationRequest,
        cancel: CancelToken,
    ) -> AsyncIterator[TurnEvent]:
        del request, cancel
        if self._block:
            await self._release.wait()
        for event in self._events:
            yield event

    def cancel(self, response_id: str, reason: str) -> bool:
        if self._cancel_delay_s:
            time.sleep(self._cancel_delay_s)
        self.cancel_calls.append((response_id, reason))
        if self._cancel_accepted:
            self._release.set()
        return self._cancel_accepted

    def stats(self) -> Mapping[str, Any]:
        return {"port": "legacy-test"}

    async def close(self, deadline_s: float) -> None:
        self.close_calls.append(deadline_s)
        if self._close_failures:
            self._close_failures -= 1
            raise RuntimeError("port close failed")
        await self._close_release.wait()

    def release(self) -> None:
        self._release.set()

    def release_close(self) -> None:
        self._close_release.set()

    def accept_cancellation(self) -> None:
        self._cancel_accepted = True


class _Provider:
    def __init__(
        self,
        port: _Port,
        capability: LegacyCapability | None = None,
    ) -> None:
        self.port = port
        self.capability = capability or LegacyCapability(supported=True)
        self.acquire_calls: list[tuple[RuntimeKey, LoadConfig]] = []

    def probe(self, model: ModelSpec) -> LegacyCapability:
        del model
        return self.capability

    async def acquire(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
    ) -> _Port:
        self.acquire_calls.append((runtime, config))
        return self.port


class _RejectTerminalSink:
    def __init__(self) -> None:
        self.events: list[TurnEvent] = []
        self.terminal_attempts = 0

    def emit(self, event: TurnEvent) -> None:
        if isinstance(event, TurnCancelled | TurnCompleted | TurnFailed):
            self.terminal_attempts += 1
            raise RuntimeError("terminal rejected")
        self.events.append(event)


def _complete_output_lifecycle(response_id: str) -> tuple[TurnEvent, ...]:
    reasoning_id = f"{response_id}:reasoning:0"
    message_id = f"{response_id}:message:1"
    tool_id = f"{response_id}:function_call:2"
    call_id = f"{response_id}:call:2"
    arguments = '{"query":"legacy"}'
    return (
        OutputItemStarted(kind="reasoning", index=0, item_id=reasoning_id),
        ContentPartStarted(
            kind=REASONING_CONTENT_KIND,
            output_index=0,
            content_index=0,
            item_id=reasoning_id,
        ),
        ReasoningDelta(
            delta="considering",
            item_id=reasoning_id,
            output_index=0,
            content_index=0,
        ),
        ReasoningCompleted(
            text="considering",
            item_id=reasoning_id,
            output_index=0,
            content_index=0,
        ),
        ContentPartCompleted(
            kind=REASONING_CONTENT_KIND,
            output_index=0,
            content_index=0,
            item_id=reasoning_id,
            text="considering",
        ),
        OutputItemCompleted(
            kind="reasoning", index=0, item_id=reasoning_id, text="considering"
        ),
        OutputItemStarted(kind="message", index=1, item_id=message_id),
        ContentPartStarted(
            kind=TEXT_CONTENT_KIND,
            output_index=1,
            content_index=0,
            item_id=message_id,
        ),
        TextDelta(
            delta="legacy ready",
            item_id=message_id,
            output_index=1,
            content_index=0,
        ),
        TextCompleted(
            text="legacy ready",
            item_id=message_id,
            output_index=1,
            content_index=0,
        ),
        ContentPartCompleted(
            kind=TEXT_CONTENT_KIND,
            output_index=1,
            content_index=0,
            item_id=message_id,
            text="legacy ready",
        ),
        OutputItemCompleted(
            kind="message", index=1, item_id=message_id, text="legacy ready"
        ),
        OutputItemStarted(
            kind="function_call",
            index=2,
            item_id=tool_id,
            call_id=call_id,
            name="lookup",
        ),
        ToolDelta(
            index=2,
            call_id=call_id,
            item_id=tool_id,
            name="lookup",
            arguments_delta=arguments,
        ),
        ToolCompleted(
            index=2,
            call_id=call_id,
            item_id=tool_id,
            name="lookup",
            arguments=arguments,
        ),
        OutputItemCompleted(
            kind="function_call",
            index=2,
            item_id=tool_id,
            call_id=call_id,
            name="lookup",
            arguments=arguments,
        ),
        UsageUpdate(input_tokens=7, output_tokens=0, total_tokens=7),
        UsageUpdate(input_tokens=7, output_tokens=2, total_tokens=9),
    )


def _runtime(backend: BackendKind = BackendKind.LEGACY_MLX) -> RuntimeKey:
    return RuntimeKey(model_id="legacy/model", revision="frozen", backend=backend)


def _request(
    runtime: RuntimeKey,
    response_id: str = "resp_legacy",
) -> GenerationRequest:
    return GenerationRequest(
        response_id=response_id,
        runtime=runtime,
        messages=({"role": "user", "content": "hello"},),
    )


def test_probe_labels_supported_fused_reject_as_legacy_never_mtp() -> None:
    runtime = _runtime()
    provider = _Provider(
        _Port(runtime),
        LegacyCapability(
            supported=True,
            text=True,
            vision=True,
            tools=True,
            continuous_batching=True,
            facts={"mtp": True, "source": "legacy-port"},
        ),
    )

    report = LegacyMlxBackend(provider).probe(
        ModelSpec(
            model_id=runtime.model_id,
            architecture="UnsupportedByFusionForConditionalGeneration",
        )
    )

    assert report.supported is True
    assert report.backend is BackendKind.LEGACY_MLX
    assert report.text is True
    assert report.vision is True
    assert report.tools is True
    assert report.mtp is False
    assert report.facts["mtp"] is False
    assert report.facts["execution_mode"] == "legacy_mlx"


@pytest.mark.asyncio
async def test_load_requires_explicit_legacy_runtime_without_silent_fallback() -> None:
    fused = _runtime(BackendKind.FUSED_MTP_MLX)
    provider = _Provider(_Port(fused))

    with pytest.raises(LegacyBackendSelectionError, match="not automatic"):
        await LegacyMlxBackend(provider).load(fused, LoadConfig())

    assert provider.acquire_calls == []


@pytest.mark.asyncio
async def test_mismatched_acquired_port_is_closed_before_load_fails() -> None:
    requested = _runtime()
    mismatched = RuntimeKey(
        model_id="legacy/other-model",
        revision="frozen",
        backend=BackendKind.LEGACY_MLX,
    )
    port = _Port(mismatched)

    with pytest.raises(LegacyPortContractError, match="different RuntimeKey"):
        await LegacyMlxBackend(_Provider(port)).load(requested, LoadConfig())

    assert port.close_calls == [0.0]


@pytest.mark.asyncio
async def test_adapter_completes_nonterminal_port_stream_exactly_once() -> None:
    runtime = _runtime()
    port = _Port(
        runtime,
        (ProgressUpdate("decode"), *_complete_output_lifecycle("resp_legacy")),
    )
    handle = await LegacyMlxBackend(_Provider(port)).load(runtime, LoadConfig())
    sink = GenerationTurn()
    stream = sink.subscribe()

    backend_turn = await handle.start_turn(_request(runtime), sink, _CancelToken())
    await backend_turn.wait_closed()
    observed = [item.event async for item in stream]

    assert [type(event).__name__ for event in observed] == [
        "TurnStarted",
        "ProgressUpdate",
        "OutputItemStarted",
        "ContentPartStarted",
        "ReasoningDelta",
        "ReasoningCompleted",
        "ContentPartCompleted",
        "OutputItemCompleted",
        "OutputItemStarted",
        "ContentPartStarted",
        "TextDelta",
        "TextCompleted",
        "ContentPartCompleted",
        "OutputItemCompleted",
        "OutputItemStarted",
        "ToolDelta",
        "ToolCompleted",
        "OutputItemCompleted",
        "UsageUpdate",
        "UsageUpdate",
        "TurnCompleted",
    ]
    assert sum(isinstance(event, TurnCompleted) for event in observed) == 1
    completed = observed[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.usage == UsageUpdate(7, 2, 9)
    assert observed[2].item_id == "resp_legacy:reasoning:0"
    assert observed[8].item_id == "resp_legacy:message:1"
    assert observed[14].item_id == "resp_legacy:function_call:2"
    text_delta = observed[10]
    assert isinstance(text_delta, TextDelta)
    assert (text_delta.output_index, text_delta.content_index) == (1, 0)
    tool_delta = observed[15]
    assert isinstance(tool_delta, ToolDelta)
    assert (tool_delta.index, tool_delta.call_id) == (2, "resp_legacy:call:2")
    assert handle.stats()["mtp"] is False


@pytest.mark.asyncio
async def test_empty_port_stream_is_a_successful_adapter_owned_completion() -> None:
    runtime = _runtime()
    handle = await LegacyMlxBackend(_Provider(_Port(runtime))).load(
        runtime,
        LoadConfig(),
    )
    sink = GenerationTurn()
    stream = sink.subscribe()

    backend_turn = await handle.start_turn(_request(runtime), sink, _CancelToken())
    await backend_turn.wait_closed()
    observed = [item.event async for item in stream]

    assert [type(event).__name__ for event in observed] == [
        "TurnStarted",
        "TurnCompleted",
    ]


@pytest.mark.asyncio
async def test_provider_terminal_is_rejected_and_replaced_by_adapter_failure() -> None:
    runtime = _runtime()
    port = _Port(runtime, (TurnCompleted("provider-owned"),))
    handle = await LegacyMlxBackend(_Provider(port)).load(runtime, LoadConfig())
    sink = GenerationTurn()
    stream = sink.subscribe()

    backend_turn = await handle.start_turn(_request(runtime), sink, _CancelToken())
    await backend_turn.wait_closed()
    observed = [item.event async for item in stream]

    assert [type(event).__name__ for event in observed] == [
        "TurnStarted",
        "TurnFailed",
    ]
    assert observed[-1].code == "legacy_provider_event_contract"


@pytest.mark.asyncio
async def test_terminal_is_not_marked_delivered_when_sink_rejects_it() -> None:
    runtime = _runtime()
    handle = await LegacyMlxBackend(_Provider(_Port(runtime))).load(
        runtime,
        LoadConfig(),
    )
    sink = _RejectTerminalSink()

    backend_turn = await handle.start_turn(_request(runtime), sink, _CancelToken())
    with pytest.raises(LegacyPortContractError, match="rejected by the turn sink"):
        await backend_turn.wait_closed()

    assert sink.terminal_attempts == 1
    assert handle.stats()["active_turns"] == 0


@pytest.mark.asyncio
async def test_cancel_ack_forwards_reason_and_emits_cancelled_once() -> None:
    runtime = _runtime()
    port = _Port(runtime, block=True, cancel_accepted=True)
    handle = await LegacyMlxBackend(_Provider(port)).load(runtime, LoadConfig())
    sink = GenerationTurn()
    stream = sink.subscribe()
    cancel = _CancelToken()
    backend_turn = await handle.start_turn(_request(runtime), sink, cancel)
    await asyncio.sleep(0)

    backend_turn.cancel("client_disconnected")
    backend_turn.cancel("later_reason_must_not_win")
    await backend_turn.wait_closed()
    observed = [item.event async for item in stream]

    assert port.cancel_calls == [("resp_legacy", "client_disconnected")]
    assert cancel.reason == "client_disconnected"
    assert isinstance(observed[-1], TurnCancelled)
    assert observed[-1].reason == "client_disconnected"
    assert sum(isinstance(event, TurnCancelled) for event in observed) == 1


@pytest.mark.asyncio
async def test_blocking_cancel_ack_does_not_block_owner_event_loop() -> None:
    runtime = _runtime()
    port = _Port(
        runtime,
        block=True,
        cancel_accepted=True,
        cancel_delay_s=0.1,
    )
    handle = await LegacyMlxBackend(_Provider(port)).load(runtime, LoadConfig())
    backend_turn = await handle.start_turn(
        _request(runtime),
        GenerationTurn(),
        _CancelToken(),
    )
    await asyncio.sleep(0)

    backend_turn.cancel("client_disconnected")
    started = asyncio.get_running_loop().time()
    await asyncio.sleep(0.01)

    assert asyncio.get_running_loop().time() - started < 0.05
    await backend_turn.wait_closed()


@pytest.mark.asyncio
async def test_cancel_nack_keeps_provider_execution_attached_until_it_ends() -> None:
    runtime = _runtime()
    port = _Port(runtime, block=True, cancel_accepted=False)
    handle = await LegacyMlxBackend(_Provider(port)).load(runtime, LoadConfig())
    sink = GenerationTurn()
    stream = sink.subscribe()
    backend_turn = await handle.start_turn(_request(runtime), sink, _CancelToken())
    await asyncio.sleep(0)

    backend_turn.cancel("deadline")
    for _ in range(100):
        if port.cancel_calls:
            break
        await asyncio.sleep(0.001)

    assert handle.stats()["active_turns"] == 1
    assert port.cancel_calls == [("resp_legacy", "deadline")]

    port.release()
    await backend_turn.wait_closed()
    observed = [item.event async for item in stream]

    assert isinstance(observed[-1], TurnFailed)
    assert observed[-1].code == "legacy_cancel_rejected"
    assert observed[-1].status_code == 409


@pytest.mark.asyncio
async def test_cancel_uses_cancel_tokens_canonical_first_reason() -> None:
    runtime = _runtime()
    port = _Port(runtime, block=True)
    handle = await LegacyMlxBackend(_Provider(port)).load(runtime, LoadConfig())
    sink = GenerationTurn()
    stream = sink.subscribe()
    cancel = _CancelToken()
    backend_turn = await handle.start_turn(_request(runtime), sink, cancel)
    await asyncio.sleep(0)

    assert cancel.cancel("deadline") is True
    backend_turn.cancel("client_disconnected")
    await backend_turn.wait_closed()
    observed = [item.event async for item in stream]

    assert port.cancel_calls == [("resp_legacy", "deadline")]
    assert isinstance(observed[-1], TurnCancelled)
    assert observed[-1].reason == "deadline"


@pytest.mark.asyncio
async def test_active_response_id_is_unique_but_reusable_after_terminal() -> None:
    runtime = _runtime()
    port = _Port(runtime, block=True)
    handle = await LegacyMlxBackend(_Provider(port)).load(runtime, LoadConfig())
    first_sink = GenerationTurn()
    first_stream = first_sink.subscribe()
    first = await handle.start_turn(_request(runtime), first_sink, _CancelToken())
    await asyncio.sleep(0)

    with pytest.raises(LegacyPortContractError, match="already active"):
        await handle.start_turn(
            _request(runtime),
            GenerationTurn(),
            _CancelToken(),
        )

    port.release()
    await first.wait_closed()
    assert isinstance([item.event async for item in first_stream][-1], TurnCompleted)

    second_sink = GenerationTurn()
    second_stream = second_sink.subscribe()
    second = await handle.start_turn(_request(runtime), second_sink, _CancelToken())
    await second.wait_closed()
    assert isinstance([item.event async for item in second_stream][-1], TurnCompleted)


@pytest.mark.asyncio
async def test_close_timeout_keeps_live_turn_tracked_and_retryable() -> None:
    runtime = _runtime()
    port = _Port(runtime, block=True, cancel_accepted=False)
    handle = await LegacyMlxBackend(_Provider(port)).load(runtime, LoadConfig())
    turn = await handle.start_turn(
        _request(runtime),
        GenerationTurn(),
        _CancelToken(),
    )
    await asyncio.sleep(0)

    with pytest.raises(LegacyPortContractError, match="total deadline"):
        await handle.close(0.05)

    assert handle.stats()["active_turns"] == 1
    assert handle.stats()["retiring"] is True
    assert port.close_calls == []

    port.accept_cancellation()
    await handle.close(0.1)
    await turn.wait_closed()
    assert port.cancel_calls == [
        ("resp_legacy", "legacy_backend_closed"),
        ("resp_legacy", "legacy_backend_closed"),
    ]
    assert len(port.close_calls) == 1
    assert 0.0 <= port.close_calls[0] < 0.1


@pytest.mark.asyncio
async def test_failed_port_close_is_retryable_without_reopening_admission() -> None:
    runtime = _runtime()
    port = _Port(runtime, close_failures=1)
    handle = await LegacyMlxBackend(_Provider(port)).load(runtime, LoadConfig())

    with pytest.raises(LegacyPortContractError, match="port close failed"):
        await handle.close(0.1)

    with pytest.raises(LegacyPortContractError, match="closed"):
        await handle.start_turn(
            _request(runtime, "resp_after_retire"),
            GenerationTurn(),
            _CancelToken(),
        )

    await handle.close(0.1)
    assert len(port.close_calls) == 2


@pytest.mark.asyncio
async def test_port_close_timeout_uses_total_budget_and_can_be_retried() -> None:
    runtime = _runtime()
    port = _Port(runtime, block_close=True)
    handle = await LegacyMlxBackend(_Provider(port)).load(runtime, LoadConfig())

    with pytest.raises(LegacyPortContractError, match="total deadline"):
        await handle.close(0.05)

    assert len(port.close_calls) == 1
    assert 0.0 <= port.close_calls[0] <= 0.05
    port.release_close()
    await handle.close(0.1)
    assert len(port.close_calls) == 2


def test_legacy_output_path_honours_the_shared_item_identity_invariant() -> None:
    """The legacy adapter answers to the same contract as the fused one."""

    from tests.output_item_identity import assert_output_item_identity_contract

    events = _complete_output_lifecycle("resp_legacy")
    assert_output_item_identity_contract(events)

    tool_start = next(
        event
        for event in events
        if isinstance(event, OutputItemStarted) and event.kind == "function_call"
    )
    tool_done = next(
        event
        for event in events
        if isinstance(event, OutputItemCompleted) and event.kind == "function_call"
    )
    assert tool_start.call_id == tool_done.call_id == "resp_legacy:call:2"
    assert tool_start.name == tool_done.name == "lookup"
