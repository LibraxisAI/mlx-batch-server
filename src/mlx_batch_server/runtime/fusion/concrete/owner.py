# SPDX-License-Identifier: Apache-2.0
"""Single-thread tensor ownership for the fused Qwen4Exp runtime.

The lifecycle is a LibraxisAI target rewrite informed by these frozen
Apache-2.0 donor objects:

* oMLX ``omlx/engine_core.py``, ``omlx/request.py`` and ``omlx/scheduler.py``
  at ``e467261edc786efd33b1e9023d5c4a827f8aa1c1``: one inference thread,
  request-local generator state, deferred abort, and cleanup on that thread.
* MTPLX ``mtplx/model_scheduler.py``, ``mtplx/cache_state.py`` and
  ``mtplx/models/qwen4_exp.py`` at
  ``6d0ddf0575faa9acf77e63c57e48ea1602a7e4ab``: explicit model-owner
  serialization and exact Qwen4Exp MTP rollback ownership.

No donor server, scheduler, pool, store, admin, or tensor package is imported.
The unresolved MLX ABI is isolated behind injected synchronous driver ports;
every driver call, including load and shutdown, runs on the same owner thread.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from dataclasses import dataclass, replace
from enum import Enum, StrEnum
from queue import Queue
from typing import TYPE_CHECKING, Any, TypeVar, cast

from ...contracts import (
    BackendKind,
    CancelToken,
    GenerationRequest,
    LoadConfig,
    ModelSpec,
    PreparedGenerationRequest,
    RequestModality,
    RuntimeKey,
)
from ..cache import CacheCleanupReceipt, CacheReleaseReason
from ..qwen4_exp import (
    Qwen4ExpExecutionBinding,
    Qwen4ExpExecutionFactoryPort,
    Qwen4ExpExecutionPort,
    Qwen4ExpRequestPreparerPort,
    probe_qwen4_exp,
)
from ..scheduler import (
    SchedulerConfig,
    SchedulerPlan,
    WorkKind,
)
from .provider import FusedTensorOwnerBinding

if TYPE_CHECKING:
    from ...backends.fused_mtp_mlx import (
        FusedCacheLeasePort,
        FusedCachePort,
        FusedExecutorPort,
        FusedStepResult,
    )
    from ..mtp import MtpPolicy
    from ..qwen4_exp.execution import Qwen4ExpPreparedExecutionFactoryPort


class Qwen4ExpTensorOwnerError(RuntimeError):
    """Base failure for the target-owned Qwen4Exp tensor owner."""


class Qwen4ExpTensorIdentityError(Qwen4ExpTensorOwnerError):
    """Runtime, model, request, plan, or lease identity did not match."""


class Qwen4ExpTensorOwnerClosedError(Qwen4ExpTensorOwnerError):
    """The tensor owner cannot accept more work."""


class Qwen4ExpDriverContractError(Qwen4ExpTensorOwnerError):
    """The injected low-level driver violated its target contract."""


class Qwen4ExpTensorShutdownError(Qwen4ExpTensorOwnerError):
    """The owner could not drain all request and driver state."""


class Qwen4ExpRequestPhase(StrEnum):
    RESERVED = "reserved"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"
    FAILED = "failed"
    ABORTING = "aborting"
    ABORTED = "aborted"
    CLEANUP_FAILED = "cleanup_failed"
    CLEANED = "cleaned"


@dataclass(frozen=True, slots=True)
class Qwen4ExpRequestSnapshot:
    request_id: str
    lease_id: str
    phase: Qwen4ExpRequestPhase
    position: int
    abort_reason: str | None
    cleanup_reason: CacheReleaseReason | None
    last_error: str | None


_T = TypeVar("_T")
_STOP = object()
_IdentityAtom = tuple[Any, ...]


@dataclass(slots=True)
class _ThreadCommand:
    action: Callable[[], Any]
    future: Future[Any]


class _InferenceMailbox:
    """FIFO command mailbox with exactly one long-lived execution thread."""

    def __init__(self, name: str) -> None:
        self._queue: Queue[_ThreadCommand | object] = Queue()
        self._guard = threading.Lock()
        self._accepting = True
        self._stop_sent = False
        self._thread_id: int | None = None
        self._thread = threading.Thread(
            target=self._run,
            name=name,
            daemon=True,
        )
        self._thread.start()

    @property
    def thread_id(self) -> int | None:
        return self._thread_id

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def submit(self, action: Callable[[], _T], *, closing: bool = False) -> Future[_T]:
        with self._guard:
            if self._stop_sent or (not self._accepting and not closing):
                raise Qwen4ExpTensorOwnerClosedError("tensor owner is closing")
            future: Future[_T] = Future()
            self._queue.put(_ThreadCommand(action=action, future=future))
            return future

    def reject_new_work(self) -> None:
        with self._guard:
            self._accepting = False

    def stop_after_pending(self) -> None:
        with self._guard:
            self._accepting = False
            if self._stop_sent:
                return
            self._stop_sent = True
            self._queue.put(_STOP)

    async def join(self, timeout_s: float | None) -> None:
        await asyncio.to_thread(self._thread.join, timeout_s)
        if self._thread.is_alive():
            raise Qwen4ExpTensorShutdownError(
                "tensor owner thread did not stop before its deadline"
            )

    def _run(self) -> None:
        self._thread_id = threading.get_ident()
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            assert isinstance(item, _ThreadCommand)
            if not item.future.set_running_or_notify_cancel():
                continue
            try:
                result = item.action()
            except BaseException as error:
                item.future.set_exception(error)
            else:
                item.future.set_result(result)


@dataclass(slots=True)
class _RequestState:
    prepared: PreparedGenerationRequest
    request_identity: _IdentityAtom
    lease_id: str
    reservation: object
    phase: Qwen4ExpRequestPhase = Qwen4ExpRequestPhase.RESERVED
    position: int = 0
    abort_reason: str | None = None
    abort_acknowledged: bool = False
    cleanup_reason: CacheReleaseReason | None = None
    cleanup_origin_phase: Qwen4ExpRequestPhase | None = None
    last_error: str | None = None

    @property
    def request(self) -> GenerationRequest:
        return self.prepared.request

    def snapshot(self) -> Qwen4ExpRequestSnapshot:
        return Qwen4ExpRequestSnapshot(
            request_id=self.request.response_id,
            lease_id=self.lease_id,
            phase=self.phase,
            position=self.position,
            abort_reason=self.abort_reason,
            cleanup_reason=self.cleanup_reason,
            last_error=self.last_error,
        )


@dataclass(frozen=True, slots=True)
class _Tombstone:
    receipt: CacheCleanupReceipt
    snapshot: Qwen4ExpRequestSnapshot


class Qwen4ExpTensorOwner:
    """Own one driver and all of its mutable state on one inference thread."""

    def __init__(
        self,
        *,
        prepared_execution_factory: Qwen4ExpPreparedExecutionFactoryPort,
        runtime: RuntimeKey,
        config: LoadConfig,
        scheduler_config: SchedulerConfig,
        request_preparer: Qwen4ExpRequestPreparerPort | None = None,
    ) -> None:
        if runtime.backend is not BackendKind.FUSED_MTP_MLX:
            raise Qwen4ExpTensorIdentityError(
                "Qwen4Exp tensor owner requires the fused_mtp_mlx backend"
            )
        self._prepared_execution_factory = prepared_execution_factory
        self._runtime = runtime
        self._config = config
        self._config_identity = _freeze_value(config)
        self._scheduler_config = scheduler_config
        self._request_preparer = request_preparer
        self._mailbox = _InferenceMailbox(
            f"qwen4-exp-owner:{_thread_name(runtime.model_id)}"
        )
        self._execution: Qwen4ExpExecutionPort | None = None
        self._model: ModelSpec | None = None
        self._owner_identity: _IdentityAtom | None = None
        self._requests: dict[str, _RequestState] = {}
        self._tombstones: OrderedDict[str, _Tombstone] = OrderedDict()
        self._max_tombstones = _positive_int_option(
            config.options,
            "tensor_owner_tombstones",
            256,
        )
        self._state_guard = threading.Lock()
        self._stats_snapshot: dict[str, Any] = {
            "state": "opening",
            "owner_thread_id": None,
            "active_requests": 0,
            "tombstones": 0,
            "requests": (),
            "driver": {},
        }
        self._shutdown_guard = asyncio.Lock()
        self._shutdown_future: Future[None] | None = None
        self._last_step_id = -1
        self._closed = False

    @classmethod
    async def open(
        cls,
        *,
        execution_factory: Qwen4ExpExecutionFactoryPort,
        runtime: RuntimeKey,
        config: LoadConfig,
        scheduler_config: SchedulerConfig,
        request_preparer: Qwen4ExpRequestPreparerPort | None = None,
    ) -> Qwen4ExpTensorOwner:
        prepared_factory = execution_factory.prepare(
            runtime,
            config,
            scheduler_config,
        )
        _validate_prepared_execution_factory(
            prepared_factory,
            runtime=runtime,
            config=config,
            scheduler_config=scheduler_config,
        )
        owner = cls(
            prepared_execution_factory=prepared_factory,
            runtime=runtime,
            config=config,
            scheduler_config=scheduler_config,
            request_preparer=request_preparer,
        )
        future = owner._mailbox.submit(owner._open_on_thread)
        try:
            await _await_future(future)
        except BaseException as opening_error:
            load_completed = False
            try:
                await _await_future(future)
            except BaseException:
                load_completed = False
            else:
                load_completed = True

            close_error: BaseException | None = None
            if load_completed:
                close_future = owner._mailbox.submit(
                    lambda: owner._shutdown_on_thread(None),
                    closing=True,
                )
                try:
                    await _await_future(close_future)
                except BaseException as error:
                    close_error = error
            try:
                owner._mailbox.stop_after_pending()
                await owner._mailbox.join(None)
            finally:
                owner._closed = True
            if close_error is not None:
                raise Qwen4ExpTensorShutdownError(
                    "cancelled tensor owner open also failed defensive shutdown"
                ) from close_error
            raise opening_error
        return owner

    @property
    def runtime_key(self) -> RuntimeKey:
        return self._runtime

    @property
    def model_spec(self) -> ModelSpec:
        model = self._model
        if model is None:
            raise Qwen4ExpTensorOwnerClosedError("tensor owner is not open")
        return model

    @property
    def owner_thread_id(self) -> int | None:
        return self._mailbox.thread_id

    @property
    def closed(self) -> bool:
        return self._closed

    async def prepare_request(
        self,
        request: GenerationRequest,
        cancel: CancelToken,
    ) -> PreparedGenerationRequest:
        if request.runtime != self._runtime:
            raise Qwen4ExpTensorIdentityError(
                "request runtime does not match the tensor owner"
            )
        if self._closed:
            raise Qwen4ExpTensorOwnerClosedError("tensor owner is closed")
        preparer = self._request_preparer
        if preparer is None:
            if request.media:
                raise Qwen4ExpTensorIdentityError(
                    "media request requires the loaded backend request preparer"
                )
            return PreparedGenerationRequest(
                request=request,
                modality=RequestModality.TEXT,
            )
        prepared = await preparer.prepare(request, cancel)
        if not isinstance(prepared, PreparedGenerationRequest):
            raise Qwen4ExpDriverContractError(
                "request preparer returned an invalid envelope"
            )
        if prepared.request is not request:
            raise Qwen4ExpTensorIdentityError(
                "request preparer replaced the canonical request"
            )
        return prepared

    async def reserve(
        self,
        prepared: PreparedGenerationRequest,
    ) -> FusedCacheLeasePort:
        request = prepared.request
        lease_id = f"qwen4-cache-{uuid.uuid4().hex}"
        future = self._mailbox.submit(
            lambda: self._reserve_on_thread(prepared, lease_id)
        )
        try:
            await _await_future(future)
        except asyncio.CancelledError:
            await _await_future(future)
            await self.cleanup(
                request.response_id,
                lease_id,
                CacheReleaseReason.REJECTED,
            )
            raise
        return _Qwen4ExpCacheLease(self, request.response_id, lease_id)

    async def execute(
        self,
        plan: SchedulerPlan,
        requests: Mapping[str, GenerationRequest],
        mtp_policy: MtpPolicy,
    ) -> FusedStepResult:
        request_map = dict(requests)
        future = self._mailbox.submit(
            lambda: self._execute_on_thread(plan, request_map, mtp_policy)
        )
        return await _await_future(future)

    async def abort(self, request_id: str, reason: str) -> None:
        if not request_id:
            raise ValueError("request_id must not be empty")
        if not reason:
            raise ValueError("abort reason must not be empty")
        future = self._mailbox.submit(lambda: self._abort_on_thread(request_id, reason))
        await _await_future(future)

    async def cleanup(
        self,
        request_id: str,
        lease_id: str,
        reason: CacheReleaseReason,
    ) -> CacheCleanupReceipt:
        future = self._mailbox.submit(
            lambda: self._cleanup_on_thread(request_id, lease_id, reason)
        )
        return await _await_future(future)

    async def request_snapshot(self, request_id: str) -> Qwen4ExpRequestSnapshot:
        future = self._mailbox.submit(
            lambda: self._request_snapshot_on_thread(request_id)
        )
        return await _await_future(future)

    def stats(self) -> Mapping[str, Any]:
        with self._state_guard:
            return _copy_value(self._stats_snapshot)

    async def shutdown(self, deadline_s: float) -> None:
        _validate_deadline(deadline_s)
        async with self._shutdown_guard:
            if self._closed:
                return
            deadline_at = time.monotonic() + deadline_s if deadline_s > 0 else None
            self._mailbox.reject_new_work()
            if self._shutdown_future is None or (
                self._shutdown_future.done()
                and self._shutdown_future.exception() is not None
            ):
                self._shutdown_future = self._mailbox.submit(
                    lambda: self._shutdown_on_thread(deadline_at),
                    closing=True,
                )
            future = self._shutdown_future
            try:
                await _await_future(
                    future,
                    timeout_s=_remaining(deadline_at),
                )
            except TimeoutError as error:
                raise Qwen4ExpTensorShutdownError(
                    "tensor owner shutdown exceeded its deadline"
                ) from error
            self._mailbox.stop_after_pending()
            await self._mailbox.join(_remaining(deadline_at))
            self._closed = True

    def _open_on_thread(self) -> None:
        binding = self._prepared_execution_factory.load()
        try:
            self._validate_driver_binding(binding)
        except BaseException:
            binding.execution.shutdown(0.0)
            raise
        self._execution = binding.execution
        self._model = binding.model
        self._owner_identity = self._current_owner_identity()
        self._publish_stats("open")

    def _validate_driver_binding(self, binding: Qwen4ExpExecutionBinding) -> None:
        if binding.runtime != self._runtime:
            raise Qwen4ExpTensorIdentityError(
                "driver returned a different runtime identity"
            )
        if _freeze_value(binding.config) != self._config_identity:
            raise Qwen4ExpTensorIdentityError(
                "driver returned a different load config identity"
            )
        if binding.scheduler_config != self._scheduler_config:
            raise Qwen4ExpTensorIdentityError(
                "driver returned a different scheduler config identity"
            )
        if binding.model.model_id != self._runtime.model_id:
            raise Qwen4ExpTensorIdentityError("driver returned a different model id")
        if (
            self._runtime.revision is not None
            and binding.model.revision != self._runtime.revision
        ):
            raise Qwen4ExpTensorIdentityError(
                "driver returned a different model revision"
            )
        if not probe_qwen4_exp(binding.model).supported:
            raise Qwen4ExpTensorIdentityError(
                "driver model is not in the Qwen4Exp family"
            )
        if binding.execution.model_spec is not binding.model:
            raise Qwen4ExpTensorIdentityError(
                "driver must expose the canonical binding model object"
            )

    def _reserve_on_thread(
        self,
        prepared: PreparedGenerationRequest,
        lease_id: str,
    ) -> None:
        self._require_owner_identity()
        execution = self._require_driver()
        request = prepared.request
        if request.runtime != self._runtime:
            raise Qwen4ExpTensorIdentityError(
                "request runtime does not match the tensor owner"
            )
        request_id = request.response_id
        if not request_id:
            raise Qwen4ExpTensorIdentityError("response_id must not be empty")
        if request_id in self._requests or request_id in self._tombstones:
            raise Qwen4ExpTensorIdentityError(
                f"duplicate tensor request identity {request_id!r}"
            )
        identity = _request_identity(request)
        if request.media and prepared.backend_payload is None:
            raise Qwen4ExpTensorIdentityError(
                "raw media request reached the tensor owner without a sealed payload"
            )
        reservation = execution.reserve(prepared, lease_id)
        if reservation is None:
            raise Qwen4ExpDriverContractError(
                "driver reserve must return an opaque request state"
            )
        self._requests[request_id] = _RequestState(
            prepared=prepared,
            request_identity=identity,
            lease_id=lease_id,
            reservation=reservation,
        )
        self._publish_stats("open")

    def _execute_on_thread(
        self,
        plan: SchedulerPlan,
        requests: Mapping[str, GenerationRequest],
        mtp_policy: MtpPolicy,
    ) -> FusedStepResult:
        self._require_owner_identity()
        execution = self._require_driver()
        planned_ids = plan.prefill_request_ids + plan.decode_request_ids
        if not planned_ids or len(set(planned_ids)) != len(planned_ids):
            raise Qwen4ExpTensorIdentityError(
                "execution plan must contain unique request rows"
            )
        if set(requests) != set(planned_ids):
            raise Qwen4ExpTensorIdentityError(
                "execution request map must exactly match the scheduler plan"
            )
        if plan.step_id <= self._last_step_id:
            raise Qwen4ExpTensorIdentityError(
                "scheduler step ids must be strictly increasing"
            )

        states: dict[str, _RequestState] = {}
        for row in plan.prefill_rows:
            state = self._validated_request(row.request_id, requests[row.request_id])
            if state.phase not in {
                Qwen4ExpRequestPhase.RESERVED,
                Qwen4ExpRequestPhase.PREFILL,
            }:
                raise Qwen4ExpTensorIdentityError(
                    f"request {row.request_id!r} is not eligible for prefill"
                )
            self._validate_scheduled_row(row.kind, row.position, state)
            state.phase = Qwen4ExpRequestPhase.PREFILL
            states[row.request_id] = state
        for row in plan.decode_rows:
            state = self._validated_request(row.request_id, requests[row.request_id])
            if state.phase is not Qwen4ExpRequestPhase.DECODE:
                raise Qwen4ExpTensorIdentityError(
                    f"request {row.request_id!r} is not eligible for decode"
                )
            self._validate_scheduled_row(row.kind, row.position, state)
            states[row.request_id] = state

        reservations = {
            request_id: states[request_id].reservation for request_id in planned_ids
        }
        self._last_step_id = plan.step_id
        try:
            result = execution.execute(plan, reservations, requests, mtp_policy)
            self._validate_step_result(plan, result)
        except BaseException as error:
            detail = f"{type(error).__name__}: {error}"
            for state in states.values():
                state.phase = Qwen4ExpRequestPhase.FAILED
                state.last_error = detail
            self._publish_stats("open")
            raise

        for row, prefill_item in zip(
            plan.prefill_rows, result.prefill_results, strict=True
        ):
            state = states[row.request_id]
            state.position = prefill_item.position
            if prefill_item.failed_reason is not None:
                state.phase = Qwen4ExpRequestPhase.FAILED
                state.last_error = prefill_item.failed_reason
            elif prefill_item.complete:
                state.phase = Qwen4ExpRequestPhase.DECODE
        for row, decode_item in zip(
            plan.decode_rows, result.decode_results, strict=True
        ):
            state = states[row.request_id]
            state.position = decode_item.position
            if decode_item.failed_reason is not None:
                state.phase = Qwen4ExpRequestPhase.FAILED
                state.last_error = decode_item.failed_reason
            elif decode_item.finished:
                state.phase = Qwen4ExpRequestPhase.FINISHED
        self._publish_stats("open")
        return result

    @staticmethod
    def _validate_scheduled_row(
        kind: WorkKind,
        position: int,
        state: _RequestState,
    ) -> None:
        expected_kind = (
            WorkKind.VISION
            if state.prepared.modality is RequestModality.VISION
            else WorkKind.TEXT
        )
        if kind is not expected_kind:
            raise Qwen4ExpTensorIdentityError(
                f"scheduler work kind changed for {state.request.response_id!r}"
            )
        if position != state.position:
            raise Qwen4ExpTensorIdentityError(
                f"scheduler position changed for {state.request.response_id!r}"
            )

    @staticmethod
    def _validate_step_result(plan: SchedulerPlan, result: FusedStepResult) -> None:
        prefill_ids = tuple(item.request_id for item in result.prefill_results)
        decode_ids = tuple(item.request_id for item in result.decode_results)
        if prefill_ids != plan.prefill_request_ids:
            raise Qwen4ExpDriverContractError(
                "driver prefill results do not match the scheduler plan"
            )
        if decode_ids != plan.decode_request_ids:
            raise Qwen4ExpDriverContractError(
                "driver decode results do not match the scheduler plan"
            )
        for row, item in zip(plan.prefill_rows, result.prefill_results, strict=True):
            if item.position < row.position:
                raise Qwen4ExpDriverContractError(
                    f"prefill position moved backwards for {row.request_id!r}"
                )
        for row, item in zip(plan.decode_rows, result.decode_results, strict=True):
            if item.position < row.position:
                raise Qwen4ExpDriverContractError(
                    f"decode position moved backwards for {row.request_id!r}"
                )
        foreign_events = set(result.events) - set(
            plan.prefill_request_ids + plan.decode_request_ids
        )
        if foreign_events:
            detail = sorted(foreign_events)
            raise Qwen4ExpDriverContractError(
                f"driver emitted events for foreign requests: {detail!r}"
            )

    def _validated_request(
        self,
        request_id: str,
        request: GenerationRequest,
    ) -> _RequestState:
        state = self._requests.get(request_id)
        if state is None:
            raise Qwen4ExpTensorIdentityError(
                f"request {request_id!r} has no tensor reservation"
            )
        if _request_identity(request) != state.request_identity:
            raise Qwen4ExpTensorIdentityError(
                f"request {request_id!r} changed after reservation"
            )
        return state

    def _abort_on_thread(self, request_id: str, reason: str) -> None:
        self._require_owner_identity()
        state = self._requests.get(request_id)
        if state is None:
            if request_id in self._tombstones:
                return
            raise Qwen4ExpTensorIdentityError(
                f"request {request_id!r} has no tensor reservation"
            )
        if state.abort_acknowledged:
            return
        effective_reason = state.abort_reason or reason
        state.abort_reason = effective_reason
        previous_phase = state.phase
        state.phase = Qwen4ExpRequestPhase.ABORTING
        try:
            self._require_driver().abort(state.reservation, effective_reason)
        except BaseException as error:
            state.last_error = f"{type(error).__name__}: {error}"
            state.phase = previous_phase
            self._publish_stats("open")
            raise
        state.abort_acknowledged = True
        state.phase = Qwen4ExpRequestPhase.ABORTED
        state.last_error = None
        self._publish_stats("open")

    def _cleanup_on_thread(
        self,
        request_id: str,
        lease_id: str,
        reason: CacheReleaseReason,
    ) -> CacheCleanupReceipt:
        self._require_owner_identity()
        state = self._requests.get(request_id)
        if state is None:
            tombstone = self._tombstones.get(request_id)
            if tombstone is None or tombstone.receipt.lease_id != lease_id:
                raise Qwen4ExpTensorIdentityError(
                    f"request {request_id!r} has no matching cache lease"
                )
            if tombstone.receipt.reason is not reason:
                raise Qwen4ExpTensorIdentityError(
                    "cache cleanup reason conflicts with the first release"
                )
            return replace(tombstone.receipt, already_released=True)
        if state.lease_id != lease_id:
            raise Qwen4ExpTensorIdentityError(
                f"stale cache lease for request {request_id!r}"
            )
        if state.cleanup_reason is not None and state.cleanup_reason is not reason:
            raise Qwen4ExpTensorIdentityError(
                "cache cleanup reason conflicts with the first release"
            )
        if state.cleanup_reason is None:
            state.cleanup_origin_phase = state.phase
            state.cleanup_reason = reason

        if (
            reason
            in {
                CacheReleaseReason.CANCELLED,
                CacheReleaseReason.ABORTED,
                CacheReleaseReason.SHUTDOWN,
            }
            and not state.abort_acknowledged
        ):
            self._abort_on_thread(request_id, state.abort_reason or reason.value)

        try:
            receipt = self._require_driver().cleanup(state.reservation, reason)
            self._validate_cleanup_receipt(state, reason, receipt)
        except BaseException as error:
            state.phase = Qwen4ExpRequestPhase.CLEANUP_FAILED
            state.last_error = f"{type(error).__name__}: {error}"
            self._publish_stats("open")
            raise

        if not receipt.succeeded:
            state.phase = Qwen4ExpRequestPhase.CLEANUP_FAILED
            state.last_error = "; ".join(receipt.errors)
            self._publish_stats("open")
            return receipt

        state.phase = Qwen4ExpRequestPhase.CLEANED
        state.last_error = None
        snapshot = state.snapshot()
        self._requests.pop(request_id)
        self._tombstones[request_id] = _Tombstone(receipt, snapshot)
        self._tombstones.move_to_end(request_id)
        while len(self._tombstones) > self._max_tombstones:
            self._tombstones.popitem(last=False)
        self._publish_stats("open")
        return receipt

    @staticmethod
    def _validate_cleanup_receipt(
        state: _RequestState,
        reason: CacheReleaseReason,
        receipt: CacheCleanupReceipt,
    ) -> None:
        if receipt.request_id != state.request.response_id:
            raise Qwen4ExpDriverContractError(
                "driver cleanup returned a different request id"
            )
        if receipt.lease_id != state.lease_id:
            raise Qwen4ExpDriverContractError(
                "driver cleanup returned a different lease id"
            )
        if receipt.reason is not reason:
            raise Qwen4ExpDriverContractError(
                "driver cleanup returned a different release reason"
            )
        if receipt.already_released:
            raise Qwen4ExpDriverContractError(
                "first owner cleanup cannot already be released"
            )

    def _request_snapshot_on_thread(
        self,
        request_id: str,
    ) -> Qwen4ExpRequestSnapshot:
        state = self._requests.get(request_id)
        if state is not None:
            return state.snapshot()
        tombstone = self._tombstones.get(request_id)
        if tombstone is not None:
            return tombstone.snapshot
        raise Qwen4ExpTensorIdentityError(f"unknown request {request_id!r}")

    def _shutdown_on_thread(self, deadline_at: float | None) -> None:
        self._require_owner_identity()
        failures: list[str] = []
        for request_id, state in tuple(self._requests.items()):
            execution_phase = state.cleanup_origin_phase or state.phase
            if (
                execution_phase
                not in {
                    Qwen4ExpRequestPhase.FINISHED,
                    Qwen4ExpRequestPhase.FAILED,
                }
                and not state.abort_acknowledged
            ):
                try:
                    self._abort_on_thread(
                        request_id,
                        state.abort_reason or "tensor owner shutdown",
                    )
                except BaseException as error:
                    failures.append(
                        f"abort {request_id}: {type(error).__name__}: {error}"
                    )
                    continue
            cleanup_reason = state.cleanup_reason or CacheReleaseReason.SHUTDOWN
            try:
                receipt = self._cleanup_on_thread(
                    request_id,
                    state.lease_id,
                    cleanup_reason,
                )
            except BaseException as error:
                failures.append(
                    f"cleanup {request_id}: {type(error).__name__}: {error}"
                )
            else:
                if not receipt.succeeded:
                    failures.append(
                        f"cleanup {request_id}: {'; '.join(receipt.errors)}"
                    )
        if failures:
            self._publish_stats("closing")
            raise Qwen4ExpTensorShutdownError("; ".join(failures))

        execution = self._require_driver()
        execution.shutdown(_remaining(deadline_at) or 0.0)
        self._execution = None
        self._publish_stats("closed")

    def _require_driver(self) -> Qwen4ExpExecutionPort:
        execution = self._execution
        if execution is None:
            raise Qwen4ExpTensorOwnerClosedError("tensor driver is not available")
        return execution

    def _current_owner_identity(self) -> _IdentityAtom:
        model = self._model
        if model is None:
            raise Qwen4ExpTensorOwnerClosedError("tensor owner has no model")
        return (
            "qwen4-exp-owner-v1",
            _runtime_identity(self._runtime),
            _freeze_value(self._config),
            _freeze_value(self._scheduler_config),
            _freeze_value(model),
        )

    def _require_owner_identity(self) -> None:
        expected = self._owner_identity
        if expected is None or self._current_owner_identity() != expected:
            raise Qwen4ExpTensorIdentityError(
                "tensor owner identity changed after load"
            )

    def _publish_stats(self, state: str) -> None:
        driver_stats: Mapping[str, Any] = {}
        driver_error: str | None = None
        if self._execution is not None:
            try:
                driver_stats = self._execution.stats()
            except BaseException as error:
                driver_error = f"{type(error).__name__}: {error}"
        snapshots = tuple(
            item.snapshot()
            for item in sorted(
                self._requests.values(),
                key=lambda value: value.request.response_id,
            )
        )
        phases: dict[str, int] = {}
        for item in snapshots:
            phases[item.phase.value] = phases.get(item.phase.value, 0) + 1
        snapshot: dict[str, Any] = {
            "state": state,
            "owner_thread_id": self._mailbox.thread_id,
            "runtime": {
                "model_id": self._runtime.model_id,
                "revision": self._runtime.revision,
                "backend": self._runtime.backend.value,
            },
            "active_requests": len(self._requests),
            "tombstones": len(self._tombstones),
            "phases": phases,
            "requests": tuple(
                {
                    "request_id": item.request_id,
                    "lease_id": item.lease_id,
                    "phase": item.phase.value,
                    "position": item.position,
                    "abort_reason": item.abort_reason,
                    "cleanup_reason": (
                        item.cleanup_reason.value if item.cleanup_reason else None
                    ),
                    "last_error": item.last_error,
                }
                for item in snapshots
            ),
            "driver": _copy_value(driver_stats),
        }
        if driver_error is not None:
            snapshot["driver_stats_error"] = driver_error
        with self._state_guard:
            self._stats_snapshot = snapshot


class _Qwen4ExpExecutorFacade:
    def __init__(self, owner: Qwen4ExpTensorOwner) -> None:
        self._owner = owner

    @property
    def model_spec(self) -> ModelSpec:
        return self._owner.model_spec

    async def prepare_request(
        self,
        request: GenerationRequest,
        cancel: CancelToken,
    ) -> PreparedGenerationRequest:
        return await self._owner.prepare_request(request, cancel)

    async def execute(
        self,
        plan: SchedulerPlan,
        requests: Mapping[str, GenerationRequest],
        mtp_policy: MtpPolicy,
    ) -> FusedStepResult:
        return await self._owner.execute(plan, requests, mtp_policy)

    async def cleanup_cancelled(self, request_id: str, reason: str) -> None:
        await self._owner.abort(request_id, reason)

    def stats(self) -> Mapping[str, Any]:
        return self._owner.stats()

    async def close(self, deadline_s: float) -> None:
        """Validate direct closes; the provider registry owns shared release."""
        _validate_deadline(deadline_s)


class _Qwen4ExpCacheFacade:
    def __init__(self, owner: Qwen4ExpTensorOwner) -> None:
        self._owner = owner

    async def acquire(
        self,
        request: PreparedGenerationRequest,
    ) -> FusedCacheLeasePort:
        return await self._owner.reserve(request)

    def stats(self) -> Mapping[str, Any]:
        return self._owner.stats()

    async def close(self, deadline_s: float) -> None:
        """Validate direct closes; the provider registry owns shared release."""
        _validate_deadline(deadline_s)


class _Qwen4ExpCacheLease:
    def __init__(
        self,
        owner: Qwen4ExpTensorOwner,
        request_id: str,
        lease_id: str,
    ) -> None:
        self._owner = owner
        self._request_id = request_id
        self._lease_id = lease_id

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def lease_id(self) -> str:
        return self._lease_id

    async def cleanup(self, reason: CacheReleaseReason) -> CacheCleanupReceipt:
        return await self._owner.cleanup(
            self._request_id,
            self._lease_id,
            reason,
        )


class Qwen4ExpTensorOwnerLoader:
    """Existing provider-port adapter for a concrete Qwen4Exp owner."""

    def __init__(
        self,
        execution_factory: Qwen4ExpExecutionFactoryPort,
        *,
        request_preparer: Qwen4ExpRequestPreparerPort | None = None,
    ) -> None:
        self._execution_factory = execution_factory
        self._request_preparer = request_preparer

    async def load(
        self,
        runtime: RuntimeKey,
        config: LoadConfig,
        scheduler_config: SchedulerConfig,
    ) -> FusedTensorOwnerBinding:
        owner = await Qwen4ExpTensorOwner.open(
            execution_factory=self._execution_factory,
            runtime=runtime,
            config=config,
            scheduler_config=scheduler_config,
            request_preparer=self._request_preparer,
        )
        model = owner.model_spec
        executor: FusedExecutorPort = _Qwen4ExpExecutorFacade(owner)
        cache: FusedCachePort = _Qwen4ExpCacheFacade(owner)
        return FusedTensorOwnerBinding(
            owner=owner,
            runtime=runtime,
            config=config,
            scheduler_config=scheduler_config,
            model=model,
            executor=executor,
            cache=cache,
        )

    async def close(self, owner: object, deadline_s: float) -> None:
        if not isinstance(owner, Qwen4ExpTensorOwner):
            raise Qwen4ExpTensorIdentityError(
                "owner loader received a foreign tensor owner"
            )
        await owner.shutdown(deadline_s)


def _validate_prepared_execution_factory(
    prepared: object,
    *,
    runtime: RuntimeKey,
    config: LoadConfig,
    scheduler_config: SchedulerConfig,
) -> None:
    try:
        prepared_factory = cast("Qwen4ExpPreparedExecutionFactoryPort", prepared)
        prepared_runtime = prepared_factory.runtime
        prepared_config = prepared_factory.config
        prepared_scheduler = prepared_factory.scheduler_config
        prepared_plan = prepared_factory.model_plan
        prepared_load = prepared_factory.load
    except AttributeError as error:
        raise Qwen4ExpDriverContractError(
            "execution factory returned an invalid prepared factory"
        ) from error
    if not callable(prepared_load) or prepared_plan is None:
        raise Qwen4ExpDriverContractError(
            "execution factory returned an invalid prepared factory"
        )
    if prepared_runtime != runtime:
        raise Qwen4ExpTensorIdentityError(
            "prepared factory returned a different runtime identity"
        )
    if _freeze_value(prepared_config) != _freeze_value(config):
        raise Qwen4ExpTensorIdentityError(
            "prepared factory returned a different load config identity"
        )
    if prepared_scheduler != scheduler_config:
        raise Qwen4ExpTensorIdentityError(
            "prepared factory returned a different scheduler config identity"
        )


async def _await_future(
    future: Future[_T],
    *,
    timeout_s: float | None = None,
) -> _T:
    wrapped = asyncio.wrap_future(future)
    if timeout_s is None:
        return await asyncio.shield(wrapped)
    return await asyncio.wait_for(asyncio.shield(wrapped), timeout=timeout_s)


def _request_identity(request: GenerationRequest) -> _IdentityAtom:
    return (
        "generation-request-v1",
        request.response_id,
        _runtime_identity(request.runtime),
        _freeze_value(request.messages),
        _freeze_value(request.media),
        _freeze_value(request.tools),
        _freeze_value(request.sampling),
        _freeze_value(request.reasoning),
        _freeze_value(request.lineage),
        _freeze_value(request.metadata),
    )


def _runtime_identity(runtime: RuntimeKey) -> _IdentityAtom:
    return (
        "runtime-key",
        runtime.model_id,
        runtime.revision,
        runtime.adapter_path,
        runtime.draft_model_id,
        runtime.backend.value,
    )


def _freeze_value(value: Any) -> _IdentityAtom:
    if value is None:
        return ("none",)
    if isinstance(value, Enum):
        return ("enum", type(value).__module__, type(value).__qualname__, value.value)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise Qwen4ExpTensorIdentityError("identity values require finite floats")
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    if isinstance(value, RuntimeKey):
        return _runtime_identity(value)
    if isinstance(value, LoadConfig):
        return (
            "load-config",
            value.max_admitted_requests,
            value.max_decode_rows,
            value.max_vision_prefills,
            value.memory_budget_bytes,
            value.cache_directory,
            _freeze_value(value.options),
        )
    if isinstance(value, SchedulerConfig):
        return (
            "scheduler-config",
            value.max_admitted_requests,
            value.max_decode_rows,
            value.max_prefill_rows,
            value.max_vision_prefills,
            value.decode_fair_share.hex(),
            value.terminal_history_size,
        )
    if isinstance(value, ModelSpec):
        return (
            "model-spec",
            value.model_id,
            value.revision,
            value.architecture,
            value.model_type,
            value.quantization,
            value.local_path,
            _freeze_value(value.metadata),
        )
    if isinstance(value, Mapping):
        items: list[tuple[str, _IdentityAtom]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise Qwen4ExpTensorIdentityError(
                    "identity mappings require string keys"
                )
            items.append((key, _freeze_value(item)))
        return ("mapping", tuple(sorted(items, key=lambda item: item[0])))
    if isinstance(value, Sequence):
        return ("sequence", tuple(_freeze_value(item) for item in value))
    raise Qwen4ExpTensorIdentityError(
        f"unsupported identity value: {type(value).__qualname__}"
    )


def _copy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_copy_value(item) for item in value)
    if isinstance(value, list):
        return [_copy_value(item) for item in value]
    return value


def _positive_int_option(
    options: Mapping[str, Any],
    name: str,
    default: int,
) -> int:
    value = options.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive int")
    return value


def _thread_name(model_id: str) -> str:
    safe = "".join(character if character.isalnum() else "-" for character in model_id)
    return safe[-48:] or "model"


def _remaining(deadline_at: float | None) -> float | None:
    if deadline_at is None:
        return None
    return max(0.0, deadline_at - time.monotonic())


def _validate_deadline(deadline_s: float) -> None:
    if not math.isfinite(deadline_s) or deadline_s < 0:
        raise ValueError("deadline_s must be finite and non-negative")


__all__ = [
    "Qwen4ExpDriverContractError",
    "Qwen4ExpRequestPhase",
    "Qwen4ExpRequestSnapshot",
    "Qwen4ExpTensorIdentityError",
    "Qwen4ExpTensorOwner",
    "Qwen4ExpTensorOwnerClosedError",
    "Qwen4ExpTensorOwnerError",
    "Qwen4ExpTensorOwnerLoader",
    "Qwen4ExpTensorShutdownError",
]
