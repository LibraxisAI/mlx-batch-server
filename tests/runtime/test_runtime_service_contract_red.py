"""RED contracts for the source-only target runtime start service.

Compile Embargo was released at the source ownership checkpoint.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Mapping
from typing import Any

import pytest

from mlx_batch_server.runtime.admission import AdmissionController
from mlx_batch_server.runtime.contracts import (
    BackendKind,
    CancelToken,
    CapabilityReport,
    GenerationRequest,
    LoadConfig,
    ModelSpec,
    PreparedGenerationRequest,
    RequestModality,
    RoleName,
    RoleSpec,
    RuntimeKey,
    TurnSink,
)
from mlx_batch_server.runtime.manager import RuntimeManager, RuntimeManagerError
from mlx_batch_server.runtime.readiness import ReadinessService
from mlx_batch_server.runtime.roles import RoleDirectory
from mlx_batch_server.runtime.service import (
    FirstWriterCancelToken,
    RuntimeStartService,
)

FLASH_MODEL = "grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit"
VISION_MODEL = "mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit"


class _Sink:
    def __init__(self) -> None:
        self.events: list[object] = []

    def emit(self, event: object) -> None:
        self.events.append(event)


class _BackendTurn:
    def __init__(self, response_id: str) -> None:
        self._response_id = response_id
        self.closed = asyncio.Event()
        self.cancel_reasons: list[str] = []
        self.cancel_outcomes: list[object] = []
        self.cancel_entered: threading.Event | None = None
        self.cancel_release: threading.Event | None = None

    @property
    def response_id(self) -> str:
        return self._response_id

    def cancel(self, reason: str) -> bool | None:
        self.cancel_reasons.append(reason)
        if self.cancel_entered is not None:
            self.cancel_entered.set()
        if self.cancel_release is not None:
            assert self.cancel_release.wait(timeout=1)
        if not self.cancel_outcomes:
            return None
        outcome = self.cancel_outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        assert outcome is None or isinstance(outcome, bool)
        return outcome

    async def wait_closed(self) -> None:
        await self.closed.wait()


class _Handle:
    def __init__(self, runtime_key: RuntimeKey) -> None:
        self._runtime_key = runtime_key
        self.start_calls: list[tuple[GenerationRequest, TurnSink, CancelToken]] = []
        self.turns: list[_BackendTurn] = []
        self.start_failure: BaseException | None = None

    @property
    def runtime_key(self) -> RuntimeKey:
        return self._runtime_key

    async def start_turn(
        self,
        request: GenerationRequest,
        sink: TurnSink,
        cancel: CancelToken,
    ) -> _BackendTurn:
        self.start_calls.append((request, sink, cancel))
        if self.start_failure is not None:
            raise self.start_failure
        turn = _BackendTurn(request.response_id)
        self.turns.append(turn)
        return turn

    def stats(self) -> Mapping[str, Any]:
        return {}

    async def close(self, deadline_s: float) -> None:
        assert deadline_s >= 0


class _ControlledFactory:
    def __init__(self) -> None:
        self.load_calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.handle: _Handle | None = None

    def probe(self, model: ModelSpec) -> CapabilityReport:
        return CapabilityReport(
            supported=model.model_id == FLASH_MODEL,
            backend=BackendKind.FUSED_MTP_MLX,
        )

    async def load(self, runtime: RuntimeKey, config: LoadConfig) -> _Handle:
        assert config.max_admitted_requests > 0
        self.load_calls += 1
        self.started.set()
        await self.release.wait()
        self.handle = _Handle(runtime)
        return self.handle


class _Lease:
    def __init__(self) -> None:
        self.release_calls = 0

    @property
    def decision(self) -> object:
        return object()

    def release(self) -> None:
        self.release_calls += 1


class _SpyManager:
    def __init__(self, handle: _Handle) -> None:
        self.handle = handle
        self.lease = _Lease()
        self.role_calls: list[tuple[RoleName, RuntimeKey]] = []
        self.direct_calls: list[RuntimeKey] = []
        self.admit_calls: list[tuple[RoleName, float | None]] = []
        self.role_failure: BaseException | None = None

    async def acquire_role(
        self,
        role: RoleName,
        *,
        runtime: RuntimeKey,
    ) -> _Handle:
        self.role_calls.append((role, runtime))
        if self.role_failure is not None:
            raise self.role_failure
        return self.handle

    async def acquire(self, runtime: RuntimeKey) -> _Handle:
        self.direct_calls.append(runtime)
        return self.handle

    async def admit(
        self,
        role: RoleName,
        *,
        timeout_s: float | None = None,
    ) -> _Lease:
        self.admit_calls.append((role, timeout_s))
        return self.lease


def _roles() -> RoleDirectory:
    return RoleDirectory(
        [
            RoleSpec(
                name=RoleName.MAIN,
                port=8100,
                requested_model=FLASH_MODEL,
                backend=BackendKind.FUSED_MTP_MLX,
            ),
            RoleSpec(
                name=RoleName.VISION,
                port=8102,
                requested_model=VISION_MODEL,
                backend=BackendKind.LEGACY_MLX,
            ),
        ]
    )


def _request(
    response_id: str,
    runtime: RuntimeKey,
    *,
    role: RoleName | str | None = RoleName.MAIN,
) -> GenerationRequest:
    metadata: dict[str, Any] = {}
    if role is not None:
        metadata["runtime_role"] = role
    return GenerationRequest(
        response_id=response_id,
        runtime=runtime,
        messages=({"role": "user", "content": "hello"},),
        metadata=metadata,
    )


@pytest.mark.asyncio
async def test_concurrent_starts_join_one_manager_load_and_start_one_turn_each() -> (
    None
):
    runtime = RuntimeKey(model_id=FLASH_MODEL)
    factory = _ControlledFactory()
    roles = _roles()
    admission = AdmissionController(max_active_requests=2)
    manager = RuntimeManager(
        {BackendKind.FUSED_MTP_MLX: factory},
        roles=roles,
        readiness=ReadinessService(roles),
        admission=admission,
    )
    service = RuntimeStartService(manager)
    first_request = _request("resp_first", runtime)
    second_request = _request("resp_second", runtime)

    first_task = asyncio.create_task(service.start(first_request, _Sink()))
    second_task = asyncio.create_task(service.start(second_request, _Sink()))
    await factory.started.wait()
    await asyncio.sleep(0)

    assert factory.load_calls == 1

    factory.release.set()
    first, second = await asyncio.gather(first_task, second_task)
    assert factory.handle is not None
    assert sorted(call[0].response_id for call in factory.handle.start_calls) == [
        first_request.response_id,
        second_request.response_id,
    ]
    assert admission.snapshot(RoleName.MAIN)["active"] == 2

    factory.handle.turns[0].closed.set()
    factory.handle.turns[1].closed.set()
    await asyncio.gather(first.wait_closed(), second.wait_closed())
    assert admission.snapshot(RoleName.MAIN)["active"] == 0


@pytest.mark.asyncio
async def test_backend_start_failure_releases_the_single_admission_lease_once() -> None:
    runtime = RuntimeKey(model_id=FLASH_MODEL)
    handle = _Handle(runtime)
    handle.start_failure = RuntimeError("backend refused start")
    manager = _SpyManager(handle)
    service = RuntimeStartService(manager)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="backend refused start"):
        await service.start(_request("resp_failed", runtime), _Sink())

    assert manager.role_calls == [(RoleName.MAIN, runtime)]
    assert manager.admit_calls == [(RoleName.MAIN, None)]
    assert len(handle.start_calls) == 1
    assert manager.lease.release_calls == 1


@pytest.mark.asyncio
async def test_cancel_race_preserves_and_forwards_only_the_first_reason() -> None:
    runtime = RuntimeKey(model_id=FLASH_MODEL)
    handle = _Handle(runtime)
    manager = _SpyManager(handle)
    service = RuntimeStartService(manager)  # type: ignore[arg-type]
    turn = await service.start(_request("resp_cancel", runtime), _Sink())

    turn.cancel("client disconnected")
    turn.cancel("registry delete")

    token = handle.start_calls[0][2]
    assert token.cancelled is True
    assert token.reason == "client disconnected"
    assert handle.turns[0].cancel_reasons == ["client disconnected"]
    assert manager.lease.release_calls == 0

    handle.turns[0].closed.set()
    await turn.wait_closed()
    assert manager.lease.release_calls == 1


@pytest.mark.asyncio
async def test_cancel_retries_rejection_without_committing_token_early() -> None:
    runtime = RuntimeKey(model_id=FLASH_MODEL)
    handle = _Handle(runtime)
    manager = _SpyManager(handle)
    service = RuntimeStartService(manager)  # type: ignore[arg-type]
    turn = await service.start(_request("resp_cancel_retry", runtime), _Sink())
    backend_turn = handle.turns[0]
    backend_turn.cancel_outcomes.extend([RuntimeError("backend busy"), False, True])
    token = handle.start_calls[0][2]

    with pytest.raises(RuntimeError, match="backend busy"):
        turn.cancel("first reason")
    assert token.cancelled is False
    assert token.reason is None

    assert turn.cancel("second reason") is False
    assert token.cancelled is False
    assert token.reason is None

    assert turn.cancel("third reason") is True
    assert token.cancelled is True
    assert token.reason == "first reason"
    assert turn.cancel("fourth reason") is True
    assert backend_turn.cancel_reasons == [
        "first reason",
        "first reason",
        "first reason",
    ]

    backend_turn.closed.set()
    await turn.wait_closed()


@pytest.mark.asyncio
async def test_concurrent_cancel_callers_do_not_duplicate_backend_delivery() -> None:
    runtime = RuntimeKey(model_id=FLASH_MODEL)
    handle = _Handle(runtime)
    manager = _SpyManager(handle)
    service = RuntimeStartService(manager)  # type: ignore[arg-type]
    turn = await service.start(_request("resp_cancel_concurrent", runtime), _Sink())
    backend_turn = handle.turns[0]
    backend_turn.cancel_entered = threading.Event()
    backend_turn.cancel_release = threading.Event()
    results: list[bool] = []
    failures: list[BaseException] = []

    def cancel(reason: str) -> None:
        try:
            results.append(turn.cancel(reason))
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=cancel, args=("first reason",))
    second = threading.Thread(target=cancel, args=("second reason",))
    first.start()
    assert backend_turn.cancel_entered.wait(timeout=1)
    second.start()

    assert backend_turn.cancel_reasons == ["first reason"]
    backend_turn.cancel_release.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert results == [True, True]
    assert backend_turn.cancel_reasons == ["first reason"]
    token = handle.start_calls[0][2]
    assert token.cancelled is True
    assert token.reason == "first reason"

    backend_turn.closed.set()
    await turn.wait_closed()


@pytest.mark.asyncio
async def test_waiting_cancel_caller_retries_after_backend_nack() -> None:
    runtime = RuntimeKey(model_id=FLASH_MODEL)
    handle = _Handle(runtime)
    manager = _SpyManager(handle)
    service = RuntimeStartService(manager)  # type: ignore[arg-type]
    turn = await service.start(_request("resp_cancel_nack_race", runtime), _Sink())
    backend_turn = handle.turns[0]
    backend_turn.cancel_outcomes.extend([RuntimeError("first delivery rejected"), True])
    backend_turn.cancel_entered = threading.Event()
    backend_turn.cancel_release = threading.Event()
    results: list[bool] = []
    failures: list[BaseException] = []

    def cancel(reason: str) -> None:
        try:
            results.append(turn.cancel(reason))
        except BaseException as exc:
            failures.append(exc)

    first = threading.Thread(target=cancel, args=("first reason",))
    second = threading.Thread(target=cancel, args=("second reason",))
    first.start()
    assert backend_turn.cancel_entered.wait(timeout=1)
    second.start()
    assert backend_turn.cancel_reasons == ["first reason"]

    backend_turn.cancel_release.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert not first.is_alive()
    assert not second.is_alive()
    assert len(failures) == 1
    assert str(failures[0]) == "first delivery rejected"
    assert results == [True]
    assert backend_turn.cancel_reasons == ["first reason", "first reason"]
    token = handle.start_calls[0][2]
    assert token.cancelled is True
    assert token.reason == "first reason"

    backend_turn.closed.set()
    await turn.wait_closed()


@pytest.mark.asyncio
async def test_admission_is_held_until_wait_closed_even_if_waiter_is_cancelled() -> (
    None
):
    runtime = RuntimeKey(model_id=FLASH_MODEL)
    handle = _Handle(runtime)
    manager = _SpyManager(handle)
    service = RuntimeStartService(manager)  # type: ignore[arg-type]
    turn = await service.start(_request("resp_wait", runtime), _Sink())

    waiter = asyncio.ensure_future(turn.wait_closed())
    await asyncio.sleep(0)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert manager.lease.release_calls == 0
    handle.turns[0].closed.set()
    await turn.wait_closed()
    await turn.wait_closed()
    assert manager.lease.release_calls == 1


@pytest.mark.asyncio
async def test_role_selection_never_rewrites_model_or_backend() -> None:
    requested = RuntimeKey(
        model_id=FLASH_MODEL,
        backend=BackendKind.FUSED_MTP_MLX,
    )
    fused_factory = _ControlledFactory()
    fused_factory.release.set()
    roles = _roles()
    admission = AdmissionController()
    manager = RuntimeManager(
        {BackendKind.FUSED_MTP_MLX: fused_factory},
        roles=roles,
        readiness=ReadinessService(roles),
        admission=admission,
    )
    service = RuntimeStartService(manager)

    with pytest.raises(ValueError, match="requested"):
        await service.start(
            _request("resp_wrong_role", requested, role=RoleName.VISION),
            _Sink(),
        )

    assert fused_factory.load_calls == 0
    assert admission.snapshot(RoleName.VISION)["active"] == 0


@pytest.mark.asyncio
async def test_direct_acquire_requires_explicit_role_services_configuration() -> None:
    runtime = RuntimeKey(model_id=FLASH_MODEL)
    handle = _Handle(runtime)
    manager = _SpyManager(handle)
    manager.role_failure = RuntimeManagerError(
        "role acquisition requires roles and readiness"
    )
    request = _request("resp_direct", runtime, role=None)

    role_service = RuntimeStartService(manager)  # type: ignore[arg-type]
    with pytest.raises(RuntimeManagerError, match="requires roles"):
        await role_service.start(request, _Sink())
    assert manager.direct_calls == []
    assert manager.admit_calls == []

    direct_service = RuntimeStartService(
        manager,  # type: ignore[arg-type]
        direct_acquire_without_role_services=True,
    )
    turn = await direct_service.start(request, _Sink())
    assert manager.direct_calls == [runtime]
    assert manager.admit_calls == [(RoleName.MAIN, None)]

    handle.turns[0].closed.set()
    await turn.wait_closed()
    assert manager.lease.release_calls == 1


@pytest.mark.asyncio
async def test_prepared_backend_orders_prepare_before_admission_and_start() -> None:
    runtime = RuntimeKey(model_id=FLASH_MODEL)
    order: list[str] = []

    class _PreparedHandle(_Handle):
        async def prepare_request(self, request, cancel):
            order.append("prepare")
            assert cancel.cancelled is False
            return PreparedGenerationRequest(request, RequestModality.TEXT)

        async def start_prepared_turn(self, prepared, sink, cancel):
            order.append("start")
            assert prepared.request.response_id == "resp_prepared"
            return await super().start_turn(prepared.request, sink, cancel)

    class _PreparedManager(_SpyManager):
        async def acquire_role(self, role, *, runtime):
            order.append("acquire")
            return await super().acquire_role(role, runtime=runtime)

        async def admit(self, role, *, timeout_s=None):
            order.append("admit")
            return await super().admit(role, timeout_s=timeout_s)

    handle = _PreparedHandle(runtime)
    manager = _PreparedManager(handle)
    service = RuntimeStartService(manager)  # type: ignore[arg-type]

    turn = await service.start(_request("resp_prepared", runtime), _Sink())

    assert order == ["acquire", "prepare", "admit", "start"]
    handle.turns[0].closed.set()
    await turn.wait_closed()


@pytest.mark.asyncio
async def test_preparation_failure_never_acquires_admission() -> None:
    runtime = RuntimeKey(model_id=FLASH_MODEL)

    class _FailingPreparedHandle(_Handle):
        async def prepare_request(self, request, cancel):
            raise RuntimeError("media resolution failed")

        async def start_prepared_turn(self, prepared, sink, cancel):
            raise AssertionError("prepared start must not be reached")

    handle = _FailingPreparedHandle(runtime)
    manager = _SpyManager(handle)
    service = RuntimeStartService(manager)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="media resolution failed"):
        await service.start(_request("resp_prepare_failed", runtime), _Sink())

    assert manager.admit_calls == []
    assert manager.lease.release_calls == 0


@pytest.mark.asyncio
async def test_cancelled_preparation_never_acquires_admission() -> None:
    runtime = RuntimeKey(model_id=FLASH_MODEL)
    entered = asyncio.Event()
    release = asyncio.Event()

    class _BlockingPreparedHandle(_Handle):
        async def prepare_request(self, request, cancel):
            entered.set()
            await release.wait()
            if cancel.cancelled:
                raise asyncio.CancelledError(cancel.reason)
            return PreparedGenerationRequest(request, RequestModality.TEXT)

        async def start_prepared_turn(self, prepared, sink, cancel):
            raise AssertionError("prepared start must not be reached")

    handle = _BlockingPreparedHandle(runtime)
    manager = _SpyManager(handle)
    service = RuntimeStartService(manager)  # type: ignore[arg-type]
    cancel = FirstWriterCancelToken()
    task = asyncio.create_task(
        service.start(
            _request("resp_prepare_cancelled", runtime),
            _Sink(),
            cancel=cancel,
        )
    )
    await entered.wait()

    assert cancel.cancel("client_cancelled") is True
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert manager.admit_calls == []
    assert manager.lease.release_calls == 0
