"""RED contracts for bounded role admission and single-flight lifecycle."""

from __future__ import annotations

import asyncio

import pytest

from mlx_batch_server.runtime.admission import (
    AdmissionController,
    AdmissionRejected,
)
from mlx_batch_server.runtime.contracts import (
    AdmissionDisposition,
    BackendKind,
    CapabilityReport,
    LoadConfig,
    ModelSpec,
    ModelState,
    ProcessState,
    RoleName,
    RoleSpec,
    RuntimeKey,
)
from mlx_batch_server.runtime.manager import RuntimeManager
from mlx_batch_server.runtime.readiness import ReadinessService
from mlx_batch_server.runtime.roles import RoleDirectory

FLASH_MODEL = "grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit"


class FakeHandle:
    def __init__(self, runtime_key: RuntimeKey) -> None:
        self._runtime_key = runtime_key
        self.capabilities = CapabilityReport(
            supported=True,
            backend=runtime_key.backend,
            text=True,
            vision=True,
            tools=True,
        )
        self.close_calls = 0
        self.close_failure: Exception | None = None

    @property
    def runtime_key(self) -> RuntimeKey:
        return self._runtime_key

    def stats(self) -> dict[str, object]:
        return {}

    async def close(self, deadline_s: float) -> None:
        assert deadline_s >= 0
        self.close_calls += 1
        if self.close_failure is not None:
            raise self.close_failure


class ControlledFactory:
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.failure: Exception | None = None
        self.returned_runtime: RuntimeKey | None = None
        self.handles: list[FakeHandle] = []
        self.configs: list[LoadConfig] = []

    def probe(self, model: ModelSpec) -> CapabilityReport:
        return CapabilityReport(
            supported=model.model_id == FLASH_MODEL,
            backend=BackendKind.FUSED_MTP_MLX,
        )

    async def load(self, runtime: RuntimeKey, config: LoadConfig) -> FakeHandle:
        self.calls += 1
        self.configs.append(config)
        self.started.set()
        await self.release.wait()
        if self.failure is not None:
            raise self.failure
        handle = FakeHandle(self.returned_runtime or runtime)
        self.handles.append(handle)
        return handle


def _main_role() -> RoleSpec:
    return RoleSpec(
        name=RoleName.MAIN,
        port=8100,
        requested_model=FLASH_MODEL,
        backend=BackendKind.FUSED_MTP_MLX,
        revision="flash-revision",
        model_dir="/models/flash-revision",
        pinned=True,
        capabilities=("text", "vision", "tools", "mtp"),
    )


def _manager(
    factory: ControlledFactory,
) -> tuple[RuntimeManager, ReadinessService]:
    roles = RoleDirectory([_main_role()])
    readiness = ReadinessService(roles, receipt={"target_sha": "32cafd2"})
    manager = RuntimeManager(
        {BackendKind.FUSED_MTP_MLX: factory},
        roles=roles,
        readiness=readiness,
    )
    return manager, readiness


def test_cold_role_is_alive_and_loadable_not_dead() -> None:
    roles = RoleDirectory([_main_role()])
    readiness = ReadinessService(roles)

    snapshot = readiness.snapshot(RoleName.MAIN)

    assert snapshot.process_state is ProcessState.ALIVE
    assert snapshot.model_state is ModelState.COLD
    assert snapshot.loaded_model is None
    assert readiness.is_loadable(RoleName.MAIN) is True
    assert readiness.is_available(RoleName.MAIN) is True
    assert readiness.is_ready(RoleName.MAIN) is False

    dead = readiness.mark_dead(RoleName.MAIN, "process exited")
    assert dead.process_state is ProcessState.DEAD
    assert dead.model_state is ModelState.COLD
    assert dead.error == "process exited"
    assert readiness.is_loadable(RoleName.MAIN) is False
    assert readiness.is_available(RoleName.MAIN) is False


def test_role_directory_rejects_competing_port_owners() -> None:
    with pytest.raises(ValueError, match="belongs to both"):
        RoleDirectory(
            [
                _main_role(),
                RoleSpec(
                    name=RoleName.CANARY,
                    port=8100,
                    requested_model="canary/model",
                    backend=BackendKind.LEGACY_MLX,
                ),
            ]
        )


@pytest.mark.asyncio
async def test_concurrent_role_acquisition_joins_one_shielded_load() -> None:
    factory = ControlledFactory()
    manager, readiness = _manager(factory)

    cancelled_waiter = asyncio.create_task(manager.acquire_role(RoleName.MAIN))
    surviving_waiter = asyncio.create_task(manager.acquire_role(RoleName.MAIN))
    await factory.started.wait()
    await asyncio.sleep(0)

    assert factory.calls == 1
    assert factory.configs[0].options["model_dir"] == "/models/flash-revision"
    assert readiness.snapshot(RoleName.MAIN).model_state is ModelState.LOADING

    cancelled_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled_waiter

    factory.release.set()
    handle = await surviving_waiter

    assert handle is factory.handles[0]
    assert factory.calls == 1
    snapshot = readiness.snapshot(RoleName.MAIN)
    assert snapshot.model_state is ModelState.READY
    assert snapshot.loaded_model == FLASH_MODEL
    assert snapshot.capabilities is handle.capabilities
    assert manager.role_stats(RoleName.MAIN) == {}
    assert snapshot.error is None


@pytest.mark.asyncio
async def test_load_failure_is_truthful_and_successful_retry_clears_it() -> None:
    factory = ControlledFactory()
    factory.failure = RuntimeError("load exploded")
    factory.release.set()
    manager, readiness = _manager(factory)

    with pytest.raises(RuntimeError, match="load exploded"):
        await manager.acquire_role(RoleName.MAIN)

    failed = readiness.snapshot(RoleName.MAIN)
    assert failed.model_state is ModelState.DEGRADED
    assert failed.loaded_model is None
    assert failed.error == "RuntimeError: load exploded"
    assert manager.status(RoleDirectory([_main_role()]).runtime_key("main")) == {
        "state": "degraded",
        "loaded": False,
        "loading": False,
        "unloading": False,
        "error": "RuntimeError: load exploded",
    }

    factory.failure = None
    handle = await manager.acquire_role(RoleName.MAIN)

    assert handle is factory.handles[0]
    assert factory.calls == 2
    recovered = readiness.snapshot(RoleName.MAIN)
    assert recovered.model_state is ModelState.READY
    assert recovered.error is None


@pytest.mark.asyncio
async def test_mismatched_factory_handle_is_closed_and_never_published() -> None:
    factory = ControlledFactory()
    factory.release.set()
    requested = RuntimeKey(model_id=FLASH_MODEL, revision="flash-revision")
    factory.returned_runtime = RuntimeKey(
        model_id="foreign/model",
        revision="flash-revision",
    )
    manager, readiness = _manager(factory)

    with pytest.raises(RuntimeError, match="different runtime key"):
        await manager.acquire_role(RoleName.MAIN)

    assert len(factory.handles) == 1
    assert factory.handles[0].close_calls == 1
    assert manager.status(requested)["loaded"] is False
    snapshot = readiness.snapshot(RoleName.MAIN)
    assert snapshot.model_state is ModelState.DEGRADED
    assert snapshot.loaded_model is None


@pytest.mark.asyncio
async def test_unload_is_single_flight_and_returns_role_to_cold() -> None:
    factory = ControlledFactory()
    factory.release.set()
    manager, readiness = _manager(factory)
    handle = await manager.acquire_role(RoleName.MAIN)
    runtime = RuntimeKey(model_id=FLASH_MODEL, revision="flash-revision")

    first, second = await asyncio.gather(
        manager.unload(runtime, deadline_s=5.0),
        manager.unload(runtime, deadline_s=5.0),
    )

    assert first is True
    assert second is True
    assert handle.close_calls == 1
    assert manager.status(runtime)["state"] == ModelState.COLD.value
    assert readiness.is_loadable(RoleName.MAIN) is True


@pytest.mark.asyncio
async def test_unload_failure_retains_handle_and_reports_exact_transition() -> None:
    factory = ControlledFactory()
    factory.release.set()
    manager, readiness = _manager(factory)
    handle = await manager.acquire_role(RoleName.MAIN)
    handle.close_failure = RuntimeError("close exploded")
    runtime = RuntimeKey(model_id=FLASH_MODEL, revision="flash-revision")

    with pytest.raises(RuntimeError, match="close exploded"):
        await manager.unload(runtime, deadline_s=5.0)

    assert manager.status(runtime) == {
        "state": "degraded",
        "loaded": True,
        "loading": False,
        "unloading": False,
        "error": "RuntimeError: close exploded",
    }
    snapshot = readiness.snapshot(RoleName.MAIN)
    assert snapshot.model_state is ModelState.DEGRADED
    assert snapshot.loaded_model == FLASH_MODEL
    assert snapshot.transition == "unload_failed"


@pytest.mark.asyncio
async def test_manager_shutdown_is_single_flight_and_rejects_new_work() -> None:
    factory = ControlledFactory()
    factory.release.set()
    manager, readiness = _manager(factory)
    handle = await manager.acquire_role(RoleName.MAIN)

    await asyncio.gather(
        manager.shutdown(deadline_s=5.0),
        manager.shutdown(deadline_s=5.0),
    )

    assert handle.close_calls == 1
    assert readiness.snapshot(RoleName.MAIN).model_state is ModelState.COLD
    with pytest.raises(RuntimeError, match="shutting down"):
        await manager.acquire_role(RoleName.MAIN)
    with pytest.raises(RuntimeError, match="shutting down"):
        await manager.admit(RoleName.MAIN)
    await manager.shutdown(deadline_s=0.0)


@pytest.mark.asyncio
async def test_manager_shutdown_failure_keeps_retryable_handle() -> None:
    factory = ControlledFactory()
    factory.release.set()
    manager, _ = _manager(factory)
    handle = await manager.acquire_role(RoleName.MAIN)
    handle.close_failure = RuntimeError("close exploded")

    with pytest.raises(RuntimeError, match="shutdown incomplete"):
        await manager.shutdown(deadline_s=5.0)

    handle.close_failure = None
    await manager.shutdown(deadline_s=5.0)
    assert handle.close_calls == 2


@pytest.mark.asyncio
async def test_admission_is_fifo_bounded_and_release_is_idempotent() -> None:
    controller = AdmissionController(max_active_requests=1, max_waiters=1)
    first = await controller.acquire(RoleName.MAIN)
    queued = asyncio.create_task(controller.acquire(RoleName.MAIN))
    await asyncio.sleep(0)

    assert controller.decide(RoleName.MAIN).disposition is AdmissionDisposition.REJECT
    assert controller.snapshot(RoleName.MAIN) == {
        "active": 1,
        "waiting": 1,
        "limit": 1,
        "max_waiters": 1,
    }
    with pytest.raises(AdmissionRejected) as rejected:
        await controller.acquire(RoleName.MAIN)
    assert rejected.value.decision.disposition is AdmissionDisposition.REJECT

    first.release()
    first.release()
    second = await queued
    assert second.decision.disposition is AdmissionDisposition.ADMIT
    assert controller.snapshot(RoleName.MAIN)["active"] == 1

    second.release()
    assert controller.snapshot(RoleName.MAIN)["active"] == 0


@pytest.mark.asyncio
async def test_admission_deadline_reports_retry_without_leaking_a_slot() -> None:
    controller = AdmissionController(max_active_requests=1, max_waiters=1)
    first = await controller.acquire(RoleName.MAIN)

    with pytest.raises(AdmissionRejected) as retry:
        await controller.acquire(RoleName.MAIN, timeout_s=0.0)

    assert retry.value.decision.disposition is AdmissionDisposition.RETRY
    assert retry.value.decision.reason == "admission_deadline_exceeded"
    assert controller.snapshot(RoleName.MAIN)["active"] == 1
    assert controller.snapshot(RoleName.MAIN)["waiting"] == 0

    first.release()
    assert controller.snapshot(RoleName.MAIN)["active"] == 0
