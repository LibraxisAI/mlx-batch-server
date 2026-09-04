"""Explicit adapter for target-owned legacy MLX execution surfaces.

This module does not load models or import an MLX implementation. A caller must
select ``LEGACY_MLX`` and inject a provider which binds an already-owned legacy
runtime. The adapter keeps protocol sequencing in the supplied ``TurnSink``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
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

__all__ = [
    "LegacyBackendError",
    "LegacyBackendSelectionError",
    "LegacyCapability",
    "LegacyExecutionPort",
    "LegacyMlxBackend",
    "LegacyPortContractError",
    "LegacyPortProvider",
]


class LegacyBackendError(RuntimeError):
    """Base error for explicit legacy backend selection and execution."""


class LegacyBackendSelectionError(LegacyBackendError):
    """Raised when a caller attempts an implicit fused-to-legacy fallback."""


class LegacyPortContractError(LegacyBackendError):
    """Raised when an injected port violates the target runtime contract."""


@dataclass(frozen=True, slots=True)
class LegacyCapability:
    """Capabilities asserted by the injected legacy implementation.

    MTP is intentionally absent: a legacy port can never advertise MTP.
    """

    supported: bool
    text: bool = True
    vision: bool = False
    tools: bool = False
    continuous_batching: bool = False
    cache_modes: tuple[str, ...] = ()
    rejection_reasons: tuple[str, ...] = ()
    facts: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class LegacyExecutionPort(Protocol):
    """A bound legacy runtime owned and loaded outside this module.

    ``events`` yields only canonical non-terminal events with stable IDs and
    complete output-item/content/tool boundaries. Usage updates are cumulative.
    Normal iterator EOF is successful completion. The adapter alone publishes
    the terminal event using the latest accepted usage authority.
    """

    @property
    def runtime_key(self) -> RuntimeKey: ...

    def events(
        self,
        request: GenerationRequest,
        cancel: CancelToken,
    ) -> AsyncIterator[TurnEvent]: ...

    def cancel(self, response_id: str, reason: str) -> bool:
        """Return true only after the provider execution is stopped and cleaned up."""

        ...

    def stats(self) -> Mapping[str, Any]: ...

    def close(self, deadline_s: float) -> Awaitable[None]: ...


@runtime_checkable
class LegacyPortProvider(Protocol):
    """External owner which probes and binds concrete legacy runtimes."""

    def probe(self, model: ModelSpec) -> LegacyCapability: ...

    def acquire(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
    ) -> Awaitable[LegacyExecutionPort]: ...


class LegacyMlxBackend:
    """Backend factory for an explicitly selected legacy execution port."""

    def __init__(self, provider: LegacyPortProvider) -> None:
        self._provider = provider

    def probe(self, model: ModelSpec) -> CapabilityReport:
        capability = self._provider.probe(model)
        facts = dict(capability.facts)
        facts.update({"execution_mode": BackendKind.LEGACY_MLX.value, "mtp": False})
        return CapabilityReport(
            supported=capability.supported,
            backend=BackendKind.LEGACY_MLX,
            architecture=model.architecture,
            text=capability.text,
            vision=capability.vision,
            tools=capability.tools,
            mtp=False,
            continuous_batching=capability.continuous_batching,
            cache_modes=capability.cache_modes,
            rejection_reasons=capability.rejection_reasons,
            facts=facts,
        )

    async def load(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
    ) -> BackendHandle:
        if runtime.backend is not BackendKind.LEGACY_MLX:
            raise LegacyBackendSelectionError(
                "legacy backend requires an explicit RuntimeKey with "
                "backend='legacy_mlx'; fused fallback is not automatic"
            )
        port = await self._provider.acquire(runtime, config)
        if port.runtime_key != runtime:
            try:
                await port.close(0.0)
            except Exception as exc:
                raise LegacyPortContractError(
                    "legacy provider returned a port for a different RuntimeKey "
                    "and cleanup failed"
                ) from exc
            raise LegacyPortContractError(
                "legacy provider returned a port for a different RuntimeKey"
            )
        return _LegacyBackendHandle(port)


class _LegacyBackendHandle:
    """One bound port with unique active IDs and post-terminal ID reuse."""

    def __init__(self, port: LegacyExecutionPort) -> None:
        self._port = port
        self._active: dict[str, _LegacyBackendTurn] = {}
        self._close_lock = asyncio.Lock()
        self._retiring = False
        self._closed = False

    @property
    def runtime_key(self) -> RuntimeKey:
        return self._port.runtime_key

    async def start_turn(
        self,
        request: GenerationRequest,
        sink: TurnSink,
        cancel: CancelToken,
    ) -> BackendTurn:
        if self._closed or self._retiring:
            raise LegacyPortContractError("legacy backend handle is closed")
        if request.runtime != self.runtime_key:
            raise LegacyPortContractError(
                "generation request RuntimeKey does not match the bound legacy port"
            )
        if not request.response_id:
            raise ValueError("response_id must not be empty")
        if request.response_id in self._active:
            raise LegacyPortContractError(
                f"legacy response {request.response_id!r} is already active"
            )

        turn = _LegacyBackendTurn(
            port=self._port,
            request=request,
            sink=sink,
            cancel_token=cancel,
            on_closed=self._forget_turn,
        )
        self._active[request.response_id] = turn
        try:
            turn.start()
        except Exception:
            self._active.pop(request.response_id, None)
            raise
        return turn

    def stats(self) -> Mapping[str, Any]:
        stats = dict(self._port.stats())
        stats.update(
            {
                "backend": BackendKind.LEGACY_MLX.value,
                "mtp": False,
                "active_turns": len(self._active),
                "retiring": self._retiring,
            }
        )
        return stats

    async def close(self, deadline_s: float) -> None:
        if deadline_s < 0:
            raise ValueError("deadline_s must be non-negative")
        if self._closed:
            return

        loop = asyncio.get_running_loop()
        deadline_at = loop.time() + deadline_s
        try:
            async with asyncio.timeout_at(deadline_at):
                async with self._close_lock:
                    if self._closed:
                        return
                    self._retiring = True
                    active = tuple(self._active.values())
                    for turn in active:
                        turn.cancel("legacy_backend_closed")
                    if active:
                        await asyncio.gather(*(turn.wait_closed() for turn in active))
                    remaining_s = max(0.0, deadline_at - loop.time())
                    await self._port.close(remaining_s)
                    self._closed = True
        except TimeoutError as exc:
            raise LegacyPortContractError(
                "legacy backend did not close before the total deadline"
            ) from exc
        except LegacyPortContractError:
            raise
        except Exception as exc:
            raise LegacyPortContractError("legacy port close failed") from exc

    def _forget_turn(self, response_id: str, turn: _LegacyBackendTurn) -> None:
        if self._active.get(response_id) is turn:
            self._active.pop(response_id, None)


class _LegacyBackendTurn:
    def __init__(
        self,
        *,
        port: LegacyExecutionPort,
        request: GenerationRequest,
        sink: TurnSink,
        cancel_token: CancelToken,
        on_closed: Callable[[str, _LegacyBackendTurn], None],
    ) -> None:
        self._port = port
        self._request = request
        self._sink = sink
        self._cancel_token = cancel_token
        self._on_closed = on_closed
        self._loop = asyncio.get_running_loop()
        self._closed = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._terminal_attempted = False
        self._terminal_delivered = False
        self._usage: UsageUpdate | None = None
        self._delivery_error: LegacyPortContractError | None = None
        self._cancel_lock = threading.Lock()
        self._cancel_reason: str | None = None
        self._cancel_scheduled = False
        self._cancel_task: asyncio.Task[None] | None = None
        self._cancel_acknowledged = False
        self._cancel_rejected_reason: str | None = None
        self._cancel_error: Exception | None = None

    @property
    def response_id(self) -> str:
        return self._request.response_id

    def start(self) -> None:
        if self._task is not None:
            raise LegacyPortContractError("legacy turn already started")
        self._sink.emit(
            TurnStarted(
                response_id=self.response_id,
                model=self._request.runtime.model_id,
                created_at=int(time.time()),
            )
        )
        self._task = self._loop.create_task(
            self._run(),
            name=f"legacy-turn:{self.response_id}",
        )

    def cancel(self, reason: str) -> None:
        reason = reason.strip()
        if not reason:
            raise ValueError("cancellation reason must not be empty")
        self._cancel_token.cancel(reason)
        canonical_reason = (self._cancel_token.reason or reason).strip()
        if not canonical_reason:
            raise LegacyPortContractError(
                "cancel token returned an empty canonical cancellation reason"
            )
        with self._cancel_lock:
            if (
                self._cancel_acknowledged
                or self._cancel_scheduled
                or self._closed.is_set()
                or self._terminal_delivered
            ):
                return
            self._cancel_reason = canonical_reason
            self._cancel_scheduled = True
        self._loop.call_soon_threadsafe(self._cancel_on_loop)

    async def wait_closed(self) -> None:
        await self._closed.wait()
        if self._delivery_error is not None:
            raise self._delivery_error

    async def _run(self) -> None:
        try:
            if self._cancel_token.cancelled:
                self._publish_terminal(
                    TurnCancelled(self._cancel_token.reason or "legacy_cancelled")
                )
                return
            async for event in self._port.events(self._request, self._cancel_token):
                if isinstance(event, TurnStarted):
                    raise LegacyPortContractError(
                        "legacy port must not emit TurnStarted"
                    )
                if isinstance(event, TERMINAL_EVENT_TYPES):
                    raise LegacyPortContractError(
                        "legacy port must not emit terminal events"
                    )
                self._sink.emit(event)
                if isinstance(event, UsageUpdate):
                    self._usage = event

            cancel_task = self._cancel_task
            if cancel_task is not None and cancel_task is not asyncio.current_task():
                await asyncio.shield(cancel_task)

            with self._cancel_lock:
                cancel_error = self._cancel_error
                rejected_reason = self._cancel_rejected_reason
                acknowledged = self._cancel_acknowledged
                cancel_reason = self._cancel_reason
            if acknowledged:
                self._publish_terminal(
                    TurnCancelled(cancel_reason or "legacy_cancelled")
                )
            elif cancel_error is not None:
                self._publish_terminal(
                    TurnFailed(str(cancel_error), code="legacy_cancel_failed")
                )
            elif rejected_reason is not None:
                self._publish_terminal(
                    TurnFailed(
                        "legacy execution port rejected cancellation",
                        code="legacy_cancel_rejected",
                        status_code=409,
                    )
                )
            else:
                self._publish_terminal(
                    TurnCompleted(
                        "stop",
                        usage=self._usage,
                        backend_stats=dict(self._port.stats()),
                    )
                )
        except asyncio.CancelledError:
            if not self._terminal_attempted:
                try:
                    self._publish_terminal(
                        TurnCancelled(self._cancel_reason or "legacy_cancelled")
                    )
                except Exception as exc:
                    self._record_delivery_error(exc)
        except Exception as exc:
            if self._terminal_attempted:
                self._record_delivery_error(exc)
            else:
                code = (
                    "legacy_provider_event_contract"
                    if isinstance(exc, LegacyPortContractError)
                    else "legacy_execution_error"
                )
                try:
                    self._publish_terminal(TurnFailed(str(exc), code=code))
                except Exception as delivery_exc:
                    self._record_delivery_error(delivery_exc)
        finally:
            self._on_closed(self.response_id, self)
            self._closed.set()

    def _cancel_on_loop(self) -> None:
        with self._cancel_lock:
            self._cancel_scheduled = False
            reason = self._cancel_reason or "legacy_cancelled"
        if self._closed.is_set() or self._terminal_delivered:
            return
        if self._cancel_task is not None:
            return
        self._cancel_task = self._loop.create_task(
            self._cancel_provider(reason),
            name=f"legacy-cancel:{self.response_id}",
        )

    async def _cancel_provider(self, reason: str) -> None:
        accepted = False
        try:
            accepted = await asyncio.to_thread(
                self._port.cancel,
                self.response_id,
                reason,
            )
        except Exception as exc:
            with self._cancel_lock:
                self._cancel_error = exc
        else:
            with self._cancel_lock:
                if accepted:
                    self._cancel_acknowledged = True
                    self._cancel_rejected_reason = None
                    self._cancel_error = None
                else:
                    self._cancel_rejected_reason = reason
                    self._cancel_error = None
        if accepted:
            try:
                self._publish_terminal(TurnCancelled(reason))
            except Exception as exc:
                self._record_delivery_error(exc)
        current_task = asyncio.current_task()
        with self._cancel_lock:
            if self._cancel_task is current_task:
                self._cancel_task = None
        if accepted and self._task is not None and not self._task.done():
            self._task.cancel()

    def _publish_terminal(
        self,
        event: TurnCompleted | TurnCancelled | TurnFailed,
    ) -> None:
        if self._terminal_attempted:
            return
        self._terminal_attempted = True
        self._sink.emit(event)
        self._terminal_delivered = True

    def _record_delivery_error(self, exc: Exception) -> None:
        if self._delivery_error is None:
            self._delivery_error = LegacyPortContractError(
                "legacy terminal event was rejected by the turn sink"
            )
            self._delivery_error.__cause__ = exc
