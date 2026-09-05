"""RED contracts for source-only fused backend orchestration.

These tests must not run while the Compile Embargo is HOLD.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from mlx_batch_server.runtime.backends.fused_mtp_mlx import (
    FusedStepResult,
    MtpMlxBackend,
)
from mlx_batch_server.runtime.contracts import (
    BackendKind,
    GenerationRequest,
    LoadConfig,
    ModelSpec,
    PreparedGenerationRequest,
    RequestModality,
    RuntimeKey,
)
from mlx_batch_server.runtime.events import (
    REASONING_CONTENT_KIND,
    TEXT_CONTENT_KIND,
    ContentPartCompleted,
    ContentPartStarted,
    OutputItemCompleted,
    OutputItemStarted,
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
    TurnStarted,
    UsageUpdate,
)
from mlx_batch_server.runtime.fusion.cache import (
    CacheCleanupReceipt,
    CacheReleaseReason,
    CacheTier,
)
from mlx_batch_server.runtime.fusion.mtp import MtpDisableReason
from mlx_batch_server.runtime.fusion.scheduler import (
    DecodeResult,
    PrefillResult,
    SchedulerConfig,
)
from mlx_batch_server.runtime.turn import GenerationTurn

RUNTIME = RuntimeKey(
    model_id="grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit",
    revision="000544f8cddcbde27c1bc302deac2b5b4d45a5b1",
)
MODEL = ModelSpec(
    model_id=RUNTIME.model_id,
    revision=RUNTIME.revision,
    architecture="Qwen4ExpForConditionalGeneration",
    model_type="qwen4_exp",
    quantization="4bit",
)


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


class _Sink:
    def __init__(self, order: list[str] | None = None) -> None:
        self.turn = GenerationTurn()
        self.events: list[TurnEvent] = []
        self._order = order

    def emit(self, event: TurnEvent) -> None:
        self.turn.emit(event)
        self.events.append(event)
        if self._order is not None:
            self._order.append(f"event:{type(event).__name__}")


def _reasoning_start_events(response_id: str) -> tuple[TurnEvent, ...]:
    item_id = f"{response_id}:reasoning:0"
    return (
        OutputItemStarted(kind="reasoning", index=0, item_id=item_id),
        ContentPartStarted(
            kind=REASONING_CONTENT_KIND,
            output_index=0,
            content_index=0,
            item_id=item_id,
        ),
    )


def _completed_output_events(response_id: str) -> tuple[TurnEvent, ...]:
    reasoning_id = f"{response_id}:reasoning:0"
    message_id = f"{response_id}:message:1"
    tool_id = f"{response_id}:function_call:2"
    call_id = f"{response_id}:call:2"
    arguments = '{"query":"ready"}'
    return (
        ReasoningDelta(
            delta="thinking",
            item_id=reasoning_id,
            output_index=0,
            content_index=0,
        ),
        ReasoningCompleted(
            text="thinking",
            item_id=reasoning_id,
            output_index=0,
            content_index=0,
        ),
        ContentPartCompleted(
            kind=REASONING_CONTENT_KIND,
            output_index=0,
            content_index=0,
            item_id=reasoning_id,
            text="thinking",
        ),
        OutputItemCompleted(
            kind="reasoning", index=0, item_id=reasoning_id, text="thinking"
        ),
        OutputItemStarted(kind="message", index=1, item_id=message_id),
        ContentPartStarted(
            kind=TEXT_CONTENT_KIND,
            output_index=1,
            content_index=0,
            item_id=message_id,
        ),
        TextDelta(
            delta="ready",
            item_id=message_id,
            output_index=1,
            content_index=0,
        ),
        TextCompleted(
            text="ready",
            item_id=message_id,
            output_index=1,
            content_index=0,
        ),
        ContentPartCompleted(
            kind=TEXT_CONTENT_KIND,
            output_index=1,
            content_index=0,
            item_id=message_id,
            text="ready",
        ),
        OutputItemCompleted(kind="message", index=1, item_id=message_id, text="ready"),
        OutputItemStarted(kind="function_call", index=2, item_id=tool_id),
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
        UsageUpdate(input_tokens=32, output_tokens=0, total_tokens=32),
        UsageUpdate(input_tokens=32, output_tokens=1, total_tokens=33),
    )


@dataclass
class _CacheLease:
    request_id: str
    order: list[str]

    async def cleanup(self, reason: CacheReleaseReason) -> CacheCleanupReceipt:
        self.order.append(f"cache_cleanup:{reason.value}")
        return CacheCleanupReceipt(
            request_id=self.request_id,
            lease_id=f"lease:{self.request_id}",
            reason=reason,
            released_tiers=(CacheTier.PAGED, CacheTier.PREFIX, CacheTier.SSD),
            released_references=1,
            pending_writes_quiesced=True,
            retained_reusable_blocks=reason is CacheReleaseReason.COMPLETED,
        )


class _Cache:
    def __init__(self, order: list[str]) -> None:
        self.order = order

    async def acquire(self, request: PreparedGenerationRequest) -> _CacheLease:
        response_id = request.request.response_id
        self.order.append(f"cache_acquire:{response_id}")
        return _CacheLease(response_id, self.order)

    def stats(self) -> dict[str, Any]:
        return {"active_tensor_leases": 0}

    async def close(self, deadline_s: float) -> None:
        del deadline_s
        self.order.append("cache_close")


class _CacheFactory:
    def __init__(self, cache: _Cache) -> None:
        self.cache = cache

    async def load(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
        model: ModelSpec,
    ) -> _Cache:
        del runtime, config
        assert model is MODEL
        return self.cache


class _Executor:
    def __init__(self, order: list[str]) -> None:
        self.order = order
        self.calls = 0

    @property
    def model_spec(self) -> ModelSpec:
        return MODEL

    async def prepare_request(self, request, cancel) -> PreparedGenerationRequest:
        del cancel
        modality = RequestModality.VISION if request.media else RequestModality.TEXT
        backend_payload = object() if modality is RequestModality.VISION else None
        return PreparedGenerationRequest(
            request=request,
            modality=modality,
            backend_payload=backend_payload,
        )

    async def execute(self, plan, requests, mtp_policy) -> FusedStepResult:
        del requests
        self.calls += 1
        if plan.prefill_rows:
            return FusedStepResult(
                prefill_results=tuple(
                    PrefillResult(row.request_id, 32, complete=True)
                    for row in plan.prefill_rows
                ),
                events={
                    row.request_id: _reasoning_start_events(row.request_id)
                    for row in plan.prefill_rows
                },
            )
        assert mtp_policy.allow_proven_multirow is False
        return FusedStepResult(
            decode_results=tuple(
                DecodeResult(row.request_id, 33, finished=True, finish_reason="stop")
                for row in plan.decode_rows
            ),
            events={
                row.request_id: _completed_output_events(row.request_id)
                for row in plan.decode_rows
            },
            ar_decode_steps=1,
            ar_decode_tokens=1,
            mtp_rounds=1,
            mtp_drafted_tokens=3,
            mtp_accepted_tokens=2,
            mtp_rejected_tokens=1,
            mtp_fallbacks=(MtpDisableReason.MULTIROW_NOT_PROVEN,),
        )

    async def cleanup_cancelled(self, request_id: str, reason: str) -> None:
        self.order.append(f"executor_cleanup:{request_id}:{reason}")

    def stats(self) -> dict[str, Any]:
        return {"loaded": True}

    async def close(self, deadline_s: float) -> None:
        del deadline_s
        self.order.append("executor_close")


class _BlockingExecutor(_Executor):
    def __init__(self, order: list[str]) -> None:
        super().__init__(order)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def execute(self, plan, requests, mtp_policy) -> FusedStepResult:
        del requests, mtp_policy
        self.entered.set()
        await self.release.wait()
        return FusedStepResult(
            prefill_results=tuple(
                PrefillResult(row.request_id, row.position, complete=False)
                for row in plan.prefill_rows
            ),
            decode_results=tuple(
                DecodeResult(row.request_id, row.position) for row in plan.decode_rows
            ),
        )


class _StopSequenceExecutor(_Executor):
    async def execute(self, plan, requests, mtp_policy) -> FusedStepResult:
        del requests, mtp_policy
        if plan.prefill_rows:
            return FusedStepResult(
                prefill_results=tuple(
                    PrefillResult(row.request_id, 32, complete=True)
                    for row in plan.prefill_rows
                )
            )
        return FusedStepResult(
            decode_results=tuple(
                DecodeResult(
                    row.request_id,
                    33,
                    finished=True,
                    finish_reason="stop_sequence",
                    stop_sequence="Exact END",
                )
                for row in plan.decode_rows
            )
        )


class _TerminalWritingExecutor(_Executor):
    async def execute(self, plan, requests, mtp_policy) -> FusedStepResult:
        del requests, mtp_policy
        return FusedStepResult(
            prefill_results=tuple(
                PrefillResult(row.request_id, row.position, complete=True)
                for row in plan.prefill_rows
            ),
            decode_results=tuple(
                DecodeResult(row.request_id, row.position, finished=True)
                for row in plan.decode_rows
            ),
            events={
                row.request_id: (TurnCompleted("executor-owned-terminal"),)
                for row in plan.prefill_rows + plan.decode_rows
            },
        )


class _ExecutorFactory:
    def __init__(self, executor: _Executor) -> None:
        self.executor = executor
        self.scheduler_config: SchedulerConfig | None = None

    async def load(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
        scheduler_config: SchedulerConfig,
    ) -> _Executor:
        del runtime, config
        self.scheduler_config = scheduler_config
        return self.executor


def _request(response_id: str, *, vision: bool = False) -> GenerationRequest:
    return GenerationRequest(
        response_id=response_id,
        runtime=RUNTIME,
        messages=({"role": "user", "content": "describe"},),
        media=({"type": "input_image", "image_url": "data:image/png;base64,eA=="},)
        if vision
        else (),
    )


def _backend(
    executor: _Executor, cache: _Cache
) -> tuple[MtpMlxBackend, _ExecutorFactory]:
    executor_factory = _ExecutorFactory(executor)
    return (
        MtpMlxBackend(
            executor_factory=executor_factory,
            cache_factory=_CacheFactory(cache),
        ),
        executor_factory,
    )


def test_flash_probe_does_not_claim_unproven_fused_runtime_capabilities() -> None:
    report = MtpMlxBackend().probe(MODEL)

    assert report.supported is True
    assert report.backend is BackendKind.FUSED_MTP_MLX
    assert report.text is True
    assert report.vision is True
    assert report.tools is True
    assert report.mtp is False
    assert report.continuous_batching is False
    assert report.facts["mtp_contract_available"] is True
    assert report.facts["mtp_runtime_proven"] is False


def test_unknown_architecture_is_rejected_without_false_mtp_label() -> None:
    report = MtpMlxBackend().probe(
        ModelSpec(model_id="unknown/model", architecture="UnknownForCausalLM")
    )

    assert report.supported is False
    assert report.mtp is False
    assert report.rejection_reasons


@pytest.mark.asyncio
async def test_load_maps_public_limits_and_reports_only_observed_mtp_work() -> None:
    order: list[str] = []
    executor = _Executor(order)
    backend, executor_factory = _backend(executor, _Cache(order))
    handle = await backend.load(
        RUNTIME,
        LoadConfig(
            max_admitted_requests=8,
            max_decode_rows=4,
            max_vision_prefills=2,
            options={"max_prefill_rows": 2, "decode_fair_share": 0.75},
        ),
    )

    assert executor_factory.scheduler_config == SchedulerConfig(
        max_admitted_requests=8,
        max_decode_rows=4,
        max_prefill_rows=2,
        max_vision_prefills=2,
        decode_fair_share=0.75,
    )
    assert handle.stats()["mtp"] == {
        "draft_depth": 3,
        "rounds": 0,
        "drafted_tokens": 0,
        "accepted_tokens": 0,
        "rejected_tokens": 0,
        "acceptance_rate": 0.0,
        "observed": False,
        "fallback_counts": {},
    }
    await handle.close(1.0)


@pytest.mark.asyncio
async def test_injected_generation_turn_owns_complete_backend_lifecycle() -> None:
    order: list[str] = []
    executor = _Executor(order)
    backend, _ = _backend(executor, _Cache(order))
    handle = await backend.load(RUNTIME, LoadConfig())
    sink = _Sink(order)

    turn = await handle.start_turn(
        _request("resp_complete", vision=True),
        sink,
        _CancelToken(),
    )
    await asyncio.wait_for(turn.wait_closed(), timeout=1.0)

    assert [type(event) for event in sink.events] == [
        TurnStarted,
        OutputItemStarted,
        ContentPartStarted,
        ReasoningDelta,
        ReasoningCompleted,
        ContentPartCompleted,
        OutputItemCompleted,
        OutputItemStarted,
        ContentPartStarted,
        TextDelta,
        TextCompleted,
        ContentPartCompleted,
        OutputItemCompleted,
        OutputItemStarted,
        ToolDelta,
        ToolCompleted,
        OutputItemCompleted,
        UsageUpdate,
        UsageUpdate,
        TurnCompleted,
    ]
    completed = sink.events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.usage == UsageUpdate(32, 1, 33)
    assert sink.events[1].item_id == "resp_complete:reasoning:0"
    assert sink.events[8].item_id == "resp_complete:message:1"
    assert sink.events[13].item_id == "resp_complete:function_call:2"
    text_delta = sink.events[9]
    assert isinstance(text_delta, TextDelta)
    assert (text_delta.output_index, text_delta.content_index) == (1, 0)
    tool_delta = sink.events[14]
    assert isinstance(tool_delta, ToolDelta)
    assert (tool_delta.index, tool_delta.call_id) == (2, "resp_complete:call:2")
    stats = handle.stats()
    assert stats["autoregressive"] == {"decode_steps": 1, "decode_tokens": 1}
    assert stats["mtp"]["observed"] is True
    assert stats["mtp"]["acceptance_rate"] == pytest.approx(2 / 3)
    assert stats["mtp"]["fallback_counts"] == {"multirow_not_proven": 1}
    assert handle.capabilities.mtp is True
    assert handle.capabilities.continuous_batching is False
    assert handle.capabilities.facts["mtp_runtime_proven"] is True
    assert handle.capabilities.facts["mtp_policy_enabled"] is True
    assert handle.capabilities.facts["mtp_rounds"] == 1
    assert handle.capabilities.facts["mtp_acceptance_rate"] == pytest.approx(2 / 3)
    assert "cache_cleanup:completed" in order
    await handle.close(1.0)


@pytest.mark.asyncio
async def test_exact_stop_sequence_reaches_turn_completed_typed_state() -> None:
    order: list[str] = []
    backend, _ = _backend(_StopSequenceExecutor(order), _Cache(order))
    handle = await backend.load(RUNTIME, LoadConfig())
    sink = _Sink(order)

    turn = await handle.start_turn(
        _request("resp_stop_sequence"),
        sink,
        _CancelToken(),
    )
    await asyncio.wait_for(turn.wait_closed(), timeout=1.0)

    completed = sink.events[-1]
    assert isinstance(completed, TurnCompleted)
    assert completed.finish_reason == "stop_sequence"
    assert completed.stop_sequence == "Exact END"
    await handle.close(1.0)


@pytest.mark.asyncio
async def test_executor_terminal_is_rejected_by_generation_turn_boundary() -> None:
    order: list[str] = []
    backend, _ = _backend(_TerminalWritingExecutor(order), _Cache(order))
    handle = await backend.load(RUNTIME, LoadConfig())
    sink = _Sink(order)

    turn = await handle.start_turn(
        _request("resp_foreign_terminal"),
        sink,
        _CancelToken(),
    )
    await asyncio.wait_for(turn.wait_closed(), timeout=1.0)

    assert isinstance(sink.events[-1], TurnFailed)
    assert sink.events[-1].code == "backend_step_failed"
    assert "cannot write start or terminal" in sink.events[-1].error
    await handle.close(1.0)


@pytest.mark.asyncio
async def test_cancel_drains_scheduler_executor_and_cache_before_terminal_and_close() -> (
    None
):
    order: list[str] = []
    executor = _BlockingExecutor(order)
    backend, _ = _backend(executor, _Cache(order))
    handle = await backend.load(RUNTIME, LoadConfig())
    sink = _Sink(order)
    backend_turn = await handle.start_turn(
        _request("resp_cancel"),
        sink,
        _CancelToken(),
    )
    await executor.entered.wait()

    backend_turn.cancel("client disconnected")
    executor.release.set()
    await asyncio.wait_for(backend_turn.wait_closed(), timeout=1.0)
    await handle.close(1.0)

    assert isinstance(sink.events[-1], TurnCancelled)
    executor_cleanup = order.index("executor_cleanup:resp_cancel:client disconnected")
    cache_cleanup = order.index("cache_cleanup:cancelled")
    terminal = order.index("event:TurnCancelled")
    executor_close = order.index("executor_close")
    cache_close = order.index("cache_close")
    assert executor_cleanup < cache_cleanup < terminal < executor_close < cache_close
