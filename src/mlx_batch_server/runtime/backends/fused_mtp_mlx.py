"""Target-owned orchestration for the fused oMLX/MTPLX runtime.

The injected executor owns tensors and model stepping. The injected cache port
owns only cache modes proven by its concrete runtime; the first Qwen4Exp cut is
whole-boundary hot-prefix only. This module owns lifecycle, scheduler
boundaries, accounting, and the single ``GenerationTurn`` event path.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from ..contracts import (
    BackendHandle,
    BackendKind,
    BackendTurn,
    CancelToken,
    CapabilityReport,
    GenerationRequest,
    LoadConfig,
    ModelSpec,
    PreparedGenerationRequest,
    RequestModality,
    RuntimeKey,
    TurnSink,
)
from ..events import (
    TERMINAL_EVENT_TYPES,
    TurnCancelled,
    TurnCompleted,
    TurnEvent,
    TurnFailed,
    TurnStarted,
    UsageUpdate,
)
from ..fusion.cache import CacheCleanupReceipt, CacheReleaseReason
from ..fusion.mtp import MtpDisableReason, MtpPolicy, MtpStats
from ..fusion.qwen4_exp import probe_qwen4_exp
from ..fusion.scheduler import (
    CancelledRequest,
    DecodeResult,
    PrefillResult,
    RequestPhase,
    SchedulerChassis,
    SchedulerConfig,
    SchedulerPlan,
    SchedulerRequest,
    SchedulerUpdate,
    SubmitDisposition,
    WorkKind,
)


class FusedBackendError(RuntimeError):
    """Base error for target-owned fused backend orchestration."""


class FusedBackendConfigurationError(FusedBackendError):
    """Required concrete tensor or cache adapters were not supplied."""


class FusedBackendCapacityError(FusedBackendError):
    """The bounded fused scheduler cannot admit another request."""


@dataclass(frozen=True, slots=True)
class FusedStepResult:
    """One executor result, with accounting derived from work actually done."""

    prefill_results: tuple[PrefillResult, ...] = ()
    decode_results: tuple[DecodeResult, ...] = ()
    events: Mapping[str, tuple[TurnEvent, ...]] = field(default_factory=dict)
    prefill_elapsed_s: float = 0.0
    decode_elapsed_s: float = 0.0
    ar_decode_steps: int = 0
    ar_decode_tokens: int = 0
    mtp_rounds: int = 0
    mtp_drafted_tokens: int = 0
    mtp_accepted_tokens: int = 0
    mtp_rejected_tokens: int = 0
    mtp_fallbacks: tuple[MtpDisableReason, ...] = ()

    def __post_init__(self) -> None:
        counters = {
            "prefill_elapsed_s": self.prefill_elapsed_s,
            "decode_elapsed_s": self.decode_elapsed_s,
            "ar_decode_steps": self.ar_decode_steps,
            "ar_decode_tokens": self.ar_decode_tokens,
            "mtp_rounds": self.mtp_rounds,
            "mtp_drafted_tokens": self.mtp_drafted_tokens,
            "mtp_accepted_tokens": self.mtp_accepted_tokens,
            "mtp_rejected_tokens": self.mtp_rejected_tokens,
        }
        if any(value < 0 for value in counters.values()):
            raise ValueError("executor accounting values must be non-negative")
        if self.mtp_accepted_tokens > self.mtp_drafted_tokens:
            raise ValueError("accepted MTP tokens cannot exceed drafted tokens")


@runtime_checkable
class FusedExecutorPort(Protocol):
    """Concrete inference adapter returning canonical non-terminal events.

    Event batches must preserve stable item/call identities and the complete
    item, content, text/reasoning, and tool lifecycle across scheduler steps.
    The injected ``TurnSink`` remains the sole ordering and finality owner.
    """

    @property
    def model_spec(self) -> ModelSpec: ...

    async def prepare_request(
        self,
        request: GenerationRequest,
        cancel: CancelToken,
    ) -> PreparedGenerationRequest: ...

    async def execute(
        self,
        plan: SchedulerPlan,
        requests: Mapping[str, GenerationRequest],
        mtp_policy: MtpPolicy,
    ) -> FusedStepResult: ...

    async def cleanup_cancelled(self, request_id: str, reason: str) -> None: ...

    def stats(self) -> Mapping[str, Any]: ...

    async def close(self, deadline_s: float) -> None: ...


@runtime_checkable
class FusedExecutorFactoryPort(Protocol):
    async def load(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
        scheduler_config: SchedulerConfig,
    ) -> FusedExecutorPort: ...


@runtime_checkable
class FusedCacheLeasePort(Protocol):
    @property
    def request_id(self) -> str: ...

    async def cleanup(self, reason: CacheReleaseReason) -> CacheCleanupReceipt: ...


@runtime_checkable
class FusedCachePort(Protocol):
    async def acquire(
        self,
        request: PreparedGenerationRequest,
    ) -> FusedCacheLeasePort: ...

    def stats(self) -> Mapping[str, Any]: ...

    async def close(self, deadline_s: float) -> None: ...


@runtime_checkable
class FusedCacheFactoryPort(Protocol):
    async def load(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
        model: ModelSpec,
    ) -> FusedCachePort: ...


@dataclass(slots=True)
class _TurnContext:
    request: GenerationRequest
    prepared: PreparedGenerationRequest
    sink: TurnSink
    cancel_token: CancelToken
    cache_lease: FusedCacheLeasePort
    closed: asyncio.Event
    usage: UsageUpdate | None = None


class _FusedBackendTurn:
    def __init__(self, owner: _FusedBackendHandle, context: _TurnContext) -> None:
        self._owner = owner
        self._context = context

    @property
    def response_id(self) -> str:
        return self._context.request.response_id

    def cancel(self, reason: str) -> None:
        if not reason:
            raise ValueError("cancel reason must not be empty")
        if self._context.closed.is_set():
            return
        self._context.cancel_token.cancel(reason)
        effective_reason = self._context.cancel_token.reason or reason
        self._owner._request_cancel(self.response_id, effective_reason)

    async def wait_closed(self) -> None:
        await self._context.closed.wait()


class _FusedBackendHandle:
    def __init__(
        self,
        *,
        runtime: RuntimeKey,
        capabilities: CapabilityReport,
        scheduler_config: SchedulerConfig,
        executor: FusedExecutorPort,
        cache: FusedCachePort,
        mtp_policy: MtpPolicy,
    ) -> None:
        self._runtime = runtime
        self._capabilities = capabilities
        self._scheduler = SchedulerChassis(scheduler_config)
        self._executor = executor
        self._cache = cache
        self._mtp_policy = mtp_policy
        self._contexts: dict[str, _TurnContext] = {}
        self._start_lock = asyncio.Lock()
        self._pump_task: asyncio.Task[None] | None = None
        self._pump_lock = asyncio.Lock()
        self._closing = False
        self._closed = False
        self._ar_decode_steps = 0
        self._ar_decode_tokens = 0
        self._mtp_stats = MtpStats()

    @property
    def runtime_key(self) -> RuntimeKey:
        return self._runtime

    @property
    def capabilities(self) -> CapabilityReport:
        mtp = self._mtp_stats
        observed_mtp = mtp.rounds > 0
        facts = dict(self._capabilities.facts)
        facts.update(
            {
                "mtp_runtime_proven": observed_mtp,
                "mtp_policy_enabled": self._mtp_policy.enabled,
                "mtp_draft_depth": self._mtp_policy.draft_depth,
                "mtp_rounds": mtp.rounds,
                "mtp_drafted_tokens": mtp.drafted_tokens,
                "mtp_accepted_tokens": mtp.accepted_tokens,
                "mtp_rejected_tokens": mtp.rejected_tokens,
                "mtp_acceptance_rate": mtp.acceptance_rate,
            }
        )
        return replace(
            self._capabilities,
            mtp=observed_mtp,
            facts=MappingProxyType(facts),
        )

    @property
    def scheduler_config(self) -> SchedulerConfig:
        return self._scheduler.config

    async def start_turn(
        self,
        request: GenerationRequest,
        sink: TurnSink,
        cancel: CancelToken,
    ) -> BackendTurn:
        prepared = await self.prepare_request(request, cancel)
        return await self.start_prepared_turn(prepared, sink, cancel)

    async def prepare_request(
        self,
        request: GenerationRequest,
        cancel: CancelToken,
    ) -> PreparedGenerationRequest:
        if request.runtime != self._runtime:
            raise ValueError("request runtime does not match loaded backend")
        prepare = getattr(self._executor, "prepare_request", None)
        if callable(prepare):
            prepared = await prepare(request, cancel)
        elif request.media:
            raise FusedBackendConfigurationError(
                "media request requires a fused executor request preparer"
            )
        else:
            prepared = PreparedGenerationRequest(
                request=request,
                modality=RequestModality.TEXT,
            )
        if not isinstance(prepared, PreparedGenerationRequest):
            raise FusedBackendError("executor returned an invalid prepared request")
        if prepared.request is not request:
            raise FusedBackendError("executor replaced the canonical request")
        if prepared.request.runtime != self._runtime:
            raise FusedBackendError("prepared request changed runtime identity")
        return prepared

    async def start_prepared_turn(
        self,
        prepared: PreparedGenerationRequest,
        sink: TurnSink,
        cancel: CancelToken,
    ) -> BackendTurn:
        async with self._start_lock:
            return await self._start_turn(prepared, sink, cancel)

    async def _start_turn(
        self,
        prepared: PreparedGenerationRequest,
        sink: TurnSink,
        cancel: CancelToken,
    ) -> _FusedBackendTurn:
        request = prepared.request
        if self._closing or self._closed:
            raise FusedBackendError("backend is closing")
        if request.runtime != self._runtime:
            raise ValueError("request runtime does not match loaded backend")
        if request.response_id in self._contexts:
            raise ValueError(f"duplicate active response_id {request.response_id!r}")

        cache_lease = await self._cache.acquire(prepared)
        if self._closing or self._closed:
            await cache_lease.cleanup(CacheReleaseReason.SHUTDOWN)
            raise FusedBackendError("backend began closing during cache acquisition")
        if cache_lease.request_id != request.response_id:
            await cache_lease.cleanup(CacheReleaseReason.REJECTED)
            raise FusedBackendError("cache lease belongs to a different request")

        try:
            sink.emit(
                TurnStarted(
                    response_id=request.response_id,
                    model=self._runtime.model_id,
                    created_at=int(time.time()),
                )
            )
        except Exception:
            await cache_lease.cleanup(CacheReleaseReason.REJECTED)
            raise

        work_kind = (
            WorkKind.VISION
            if prepared.modality is RequestModality.VISION
            else WorkKind.TEXT
        )
        submitted = self._scheduler.submit(
            SchedulerRequest(request.response_id, kind=work_kind)
        )
        if submitted.disposition is not SubmitDisposition.ACCEPTED:
            await cache_lease.cleanup(CacheReleaseReason.REJECTED)
            raise FusedBackendCapacityError(submitted.reason)

        context = _TurnContext(
            request=request,
            prepared=prepared,
            sink=sink,
            cancel_token=cancel,
            cache_lease=cache_lease,
            closed=asyncio.Event(),
        )
        self._contexts[request.response_id] = context
        if cancel.cancelled:
            self._request_cancel(
                request.response_id,
                cancel.reason or "cancelled before backend admission",
            )
        self._ensure_pump()
        return _FusedBackendTurn(self, context)

    def stats(self) -> Mapping[str, Any]:
        snapshot = self._scheduler.snapshot()
        mtp = self._mtp_stats
        return {
            "runtime": {
                "model_id": self._runtime.model_id,
                "revision": self._runtime.revision,
                "backend": self._runtime.backend.value,
            },
            "scheduler": {
                "admitted_requests": snapshot.admitted_requests,
                "waiting_requests": snapshot.waiting_requests,
                "prefilling_requests": snapshot.prefilling_requests,
                "decoding_requests": snapshot.decoding_requests,
                "max_admitted_requests": self._scheduler.config.max_admitted_requests,
                "max_decode_rows": self._scheduler.config.max_decode_rows,
                "max_vision_prefills": self._scheduler.config.max_vision_prefills,
            },
            "autoregressive": {
                "decode_steps": self._ar_decode_steps,
                "decode_tokens": self._ar_decode_tokens,
            },
            "mtp": {
                "draft_depth": self._mtp_policy.draft_depth,
                "rounds": mtp.rounds,
                "drafted_tokens": mtp.drafted_tokens,
                "accepted_tokens": mtp.accepted_tokens,
                "rejected_tokens": mtp.rejected_tokens,
                "acceptance_rate": mtp.acceptance_rate,
                "observed": mtp.rounds > 0,
                "fallback_counts": {
                    reason.value: count
                    for reason, count in sorted(
                        mtp.fallback_counts.items(), key=lambda item: item[0].value
                    )
                },
            },
            "executor": dict(self._executor.stats()),
            "cache": dict(self._cache.stats()),
        }

    async def close(self, deadline_s: float) -> None:
        if deadline_s < 0:
            raise ValueError("deadline_s must be non-negative")
        if self._closed:
            return
        self._closing = True
        loop = asyncio.get_running_loop()
        deadline = loop.time() + deadline_s
        turns = tuple(self._contexts.values())
        for context in turns:
            self._request_cancel(context.request.response_id, "backend shutdown")
        self._ensure_pump()

        if turns:
            await asyncio.wait_for(
                asyncio.gather(*(context.closed.wait() for context in turns)),
                timeout=max(0.0, deadline - loop.time()),
            )
        if self._pump_task is not None:
            await asyncio.wait_for(
                asyncio.shield(self._pump_task),
                timeout=max(0.0, deadline - loop.time()),
            )
        await self._executor.close(max(0.0, deadline - loop.time()))
        await self._cache.close(max(0.0, deadline - loop.time()))
        self._closed = True

    def _request_cancel(self, request_id: str, reason: str) -> None:
        self._scheduler.request_cancel(request_id, reason)
        self._ensure_pump()

    def _ensure_pump(self) -> None:
        if self._pump_task is None or self._pump_task.done():
            self._pump_task = asyncio.create_task(
                self._drain_scheduler(),
                name=f"fused-scheduler:{self._runtime.model_id}",
            )

    async def _drain_scheduler(self) -> None:
        async with self._pump_lock:
            while True:
                self._observe_cancel_tokens()
                if not self._scheduler.has_work():
                    return
                plan = self._scheduler.next_plan()
                await self._cleanup_cancelled(plan.cancelled_requests)
                if not plan.requires_completion:
                    if not self._scheduler.has_work():
                        return
                    await asyncio.sleep(0)
                    continue
                await self._execute_plan(plan)

    def _observe_cancel_tokens(self) -> None:
        for request_id, context in tuple(self._contexts.items()):
            if context.closed.is_set() or not context.cancel_token.cancelled:
                continue
            self._scheduler.request_cancel(
                request_id,
                context.cancel_token.reason or "request cancelled",
            )

    async def _execute_plan(self, plan: SchedulerPlan) -> None:
        requests = {
            request_id: self._contexts[request_id].request
            for request_id in plan.prefill_request_ids + plan.decode_request_ids
        }
        try:
            result = await self._executor.execute(plan, requests, self._mtp_policy)
            self._validate_step_result(plan, result)
            self._emit_step_events(plan, result)
            update = self._scheduler.complete_step(
                plan,
                prefill_results=result.prefill_results,
                decode_results=result.decode_results,
                prefill_elapsed_s=result.prefill_elapsed_s,
                decode_elapsed_s=result.decode_elapsed_s,
            )
            self._record_accounting(result)
        except Exception as error:
            update = self._fail_plan(plan, error)
        await self._finish_scheduler_update(update)

    @staticmethod
    def _validate_step_result(plan: SchedulerPlan, result: FusedStepResult) -> None:
        active = set(plan.prefill_request_ids + plan.decode_request_ids)
        foreign = set(result.events) - active
        if foreign:
            raise FusedBackendError(
                f"executor emitted events for unleased requests: {sorted(foreign)!r}"
            )
        for events in result.events.values():
            for event in events:
                if isinstance(event, (TurnStarted, *TERMINAL_EVENT_TYPES)):
                    raise FusedBackendError(
                        "executor cannot write start or terminal turn events"
                    )

    def _emit_step_events(
        self,
        plan: SchedulerPlan,
        result: FusedStepResult,
    ) -> None:
        ordered_ids = plan.prefill_request_ids + plan.decode_request_ids
        for request_id in ordered_ids:
            context = self._contexts[request_id]
            for event in result.events.get(request_id, ()):
                context.sink.emit(event)
                if isinstance(event, UsageUpdate):
                    context.usage = event

    def _fail_plan(self, plan: SchedulerPlan, error: Exception) -> SchedulerUpdate:
        reason = f"{type(error).__name__}: {error}"
        prefill_results = tuple(
            PrefillResult(row.request_id, row.position, failed_reason=reason)
            for row in plan.prefill_rows
        )
        decode_results = tuple(
            DecodeResult(row.request_id, row.position, failed_reason=reason)
            for row in plan.decode_rows
        )
        return self._scheduler.complete_step(
            plan,
            prefill_results=prefill_results,
            decode_results=decode_results,
        )

    async def _cleanup_cancelled(
        self,
        cancelled: tuple[CancelledRequest, ...],
    ) -> None:
        for item in cancelled:
            context = self._contexts.get(item.request_id)
            if context is None:
                continue
            errors: list[str] = []
            try:
                await self._executor.cleanup_cancelled(item.request_id, item.reason)
            except Exception as error:
                errors.append(f"executor cleanup: {type(error).__name__}: {error}")
            try:
                receipt = await context.cache_lease.cleanup(
                    CacheReleaseReason.CANCELLED
                )
                if not receipt.succeeded:
                    errors.extend(receipt.errors)
            except Exception as error:
                errors.append(f"cache cleanup: {type(error).__name__}: {error}")
            terminal: TurnCancelled | TurnFailed
            if errors:
                terminal = TurnFailed(
                    error="; ".join(errors),
                    code="cancel_cleanup_failed",
                )
            else:
                terminal = TurnCancelled(item.reason)
            await self._close_context(context, terminal)

    async def _finish_scheduler_update(self, update: SchedulerUpdate) -> None:
        for terminal in update.terminal_requests:
            context = self._contexts.get(terminal.request_id)
            if context is None:
                continue
            reason = (
                CacheReleaseReason.COMPLETED
                if terminal.phase is RequestPhase.COMPLETED
                else CacheReleaseReason.FAILED
            )
            try:
                receipt = await context.cache_lease.cleanup(reason)
                if not receipt.succeeded:
                    raise FusedBackendError("; ".join(receipt.errors))
            except Exception as error:
                event: TurnCompleted | TurnFailed = TurnFailed(
                    error=f"cache cleanup: {type(error).__name__}: {error}",
                    code="cache_cleanup_failed",
                )
            else:
                if terminal.phase is RequestPhase.COMPLETED:
                    event = TurnCompleted(
                        finish_reason=terminal.reason,
                        usage=context.usage,
                        backend_stats=dict(self.stats()),
                    )
                else:
                    event = TurnFailed(
                        error=terminal.reason,
                        code="backend_step_failed",
                    )
            await self._close_context(context, event)

    def _record_accounting(self, result: FusedStepResult) -> None:
        self._ar_decode_steps += result.ar_decode_steps
        self._ar_decode_tokens += result.ar_decode_tokens
        self._mtp_stats.rounds += result.mtp_rounds
        self._mtp_stats.drafted_tokens += result.mtp_drafted_tokens
        self._mtp_stats.accepted_tokens += result.mtp_accepted_tokens
        self._mtp_stats.rejected_tokens += result.mtp_rejected_tokens
        for reason in result.mtp_fallbacks:
            self._mtp_stats.record_fallback(reason)

    async def _close_context(
        self,
        context: _TurnContext,
        terminal: TurnCancelled | TurnCompleted | TurnFailed,
    ) -> None:
        if context.closed.is_set():
            return
        try:
            context.sink.emit(terminal)
        finally:
            context.closed.set()
            self._contexts.pop(context.request.response_id, None)


class MtpMlxBackend:
    """Factory for one fused scheduler/cache/MTP backend handle."""

    def __init__(
        self,
        *,
        executor_factory: FusedExecutorFactoryPort | None = None,
        cache_factory: FusedCacheFactoryPort | None = None,
    ) -> None:
        self._executor_factory = executor_factory
        self._cache_factory = cache_factory

    def probe(self, model: ModelSpec) -> CapabilityReport:
        return probe_qwen4_exp(model)

    async def load(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
    ) -> BackendHandle:
        if runtime.backend is not BackendKind.FUSED_MTP_MLX:
            raise ValueError("runtime key selects a different backend")
        if self._executor_factory is None or self._cache_factory is None:
            raise FusedBackendConfigurationError(
                "fused executor and cache factories must be injected"
            )

        scheduler_config = _scheduler_config(config)
        mtp_policy = _mtp_policy(config)
        executor = await self._executor_factory.load(
            runtime,
            config,
            scheduler_config,
        )
        model = executor.model_spec
        report = self.probe(model)
        if model.model_id != runtime.model_id or (
            runtime.revision is not None and model.revision != runtime.revision
        ):
            await executor.close(0.0)
            raise FusedBackendError("executor loaded a different model identity")
        if not report.supported:
            await executor.close(0.0)
            raise FusedBackendError(
                f"unsupported fused model: {', '.join(report.rejection_reasons)}"
            )

        try:
            cache = await self._cache_factory.load(runtime, config, model)
        except Exception:
            await executor.close(0.0)
            raise
        return _FusedBackendHandle(
            runtime=runtime,
            capabilities=report,
            scheduler_config=scheduler_config,
            executor=executor,
            cache=cache,
            mtp_policy=mtp_policy,
        )


def _scheduler_config(config: LoadConfig) -> SchedulerConfig:
    options = config.options
    max_prefill_rows = _int_option(
        options,
        "max_prefill_rows",
        min(config.max_decode_rows, config.max_vision_prefills),
    )
    return SchedulerConfig(
        max_admitted_requests=config.max_admitted_requests,
        max_decode_rows=config.max_decode_rows,
        max_prefill_rows=max_prefill_rows,
        max_vision_prefills=config.max_vision_prefills,
        decode_fair_share=_float_option(options, "decode_fair_share", 0.5),
        terminal_history_size=_int_option(options, "terminal_history_size", 128),
    )


def _mtp_policy(config: LoadConfig) -> MtpPolicy:
    options = config.options
    live_proven = options.get("mtp_multirow_live_proven", False)
    if not isinstance(live_proven, bool):
        raise ValueError("mtp_multirow_live_proven must be a bool")
    enabled = options.get("mtp_enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("mtp_enabled must be a bool")
    draft_depth = _int_option(options, "mtp_draft_depth", 3)
    if not 1 <= draft_depth <= 8:
        raise ValueError("mtp_draft_depth must be between 1 and 8")
    return MtpPolicy(
        enabled=enabled,
        draft_depth=draft_depth,
        allow_proven_multirow=live_proven,
        max_proven_rows=_int_option(options, "mtp_max_proven_rows", 1),
    )


def _int_option(options: Mapping[str, Any], name: str, default: int) -> int:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    return value


def _float_option(options: Mapping[str, Any], name: str, default: float) -> float:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    return float(value)


__all__ = [
    "FusedBackendCapacityError",
    "FusedBackendConfigurationError",
    "FusedBackendError",
    "FusedCacheFactoryPort",
    "FusedCacheLeasePort",
    "FusedCachePort",
    "FusedExecutorFactoryPort",
    "FusedExecutorPort",
    "FusedStepResult",
    "MtpMlxBackend",
]
