"""RED contracts for fused physical materialization before READY.

These tests are authored but deliberately not executed while Compile Embargo
W2 is active.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from mlx_batch_server.runtime.contracts import (
    BackendKind,
    CapabilityReport,
    LoadConfig,
    ModelSpec,
    ModelState,
    RoleName,
    RoleSnapshot,
    RoleSpec,
    RuntimeKey,
    TensorMaterializationReceipt,
)
from mlx_batch_server.runtime.factory_authority import (
    _bind_trusted_fused_backend_factory,
)
from mlx_batch_server.runtime.manager import RuntimeManager, RuntimeManagerError
from mlx_batch_server.runtime.readiness import ReadinessService
from mlx_batch_server.runtime.roles import RoleDirectory

MODEL = "grant-ai/Qwen3.8-Flash-Next-Abliterated-MLX-4bit"
RUNTIME = RuntimeKey(
    model_id=MODEL,
    revision="exact-revision",
    backend=BackendKind.FUSED_MTP_MLX,
)
_TEST_MATERIALIZATION_AUTHORITY = object()


def _receipt(
    runtime: RuntimeKey = RUNTIME,
    *,
    authority: object = _TEST_MATERIALIZATION_AUTHORITY,
) -> TensorMaterializationReceipt:
    return TensorMaterializationReceipt(
        schema="mlx-tensor-materialization.v1",
        load_id="load-test",
        runtime=runtime,
        qwen4_exp_plan_sha256="1" * 64,
        artifact_inventory_sha256="2" * 64,
        parameter_manifest_sha256="3" * 64,
        parameter_path_count=3,
        evaluated_leaf_count=3,
        evaluated_logical_bytes=1024,
        owner_thread_id=1,
        completed_at_monotonic_ns=1,
        checkpoint_content_sha256="4" * 64,
        _issuer_authority=authority,
    )


_MISSING = object()


class _Handle:
    def __init__(
        self,
        runtime: RuntimeKey = RUNTIME,
        receipt: object = _receipt(),
    ) -> None:
        self._runtime = runtime
        self._receipt = receipt
        self.receipt_reads = 0
        self.capabilities = CapabilityReport(
            supported=True,
            backend=runtime.backend,
        )
        self.close_calls = 0

    @property
    def runtime_key(self) -> RuntimeKey:
        return self._runtime

    @property
    def materialization_receipt(self) -> object:
        self.receipt_reads += 1
        if self._receipt is _MISSING:
            raise AttributeError("materialization_receipt")
        return self._receipt

    def stats(self) -> dict[str, object]:
        return {}

    async def close(self, deadline_s: float) -> None:
        assert deadline_s >= 0
        self.close_calls += 1


class _ChangingReceiptHandle(_Handle):
    def __init__(self) -> None:
        self.initial_receipt = _receipt()
        super().__init__(receipt=self.initial_receipt)

    @property
    def materialization_receipt(self) -> TensorMaterializationReceipt:
        receipt = super().materialization_receipt
        assert isinstance(receipt, TensorMaterializationReceipt)
        if self.receipt_reads == 1:
            return receipt
        return replace(receipt, load_id=f"load-read-{self.receipt_reads}")


class _BlockingCloseHandle(_Handle):
    def __init__(self) -> None:
        super().__init__()
        self.close_entered = asyncio.Event()
        self.close_release = asyncio.Event()
        self.close_deadlines: list[float] = []

    async def close(self, deadline_s: float) -> None:
        self.close_calls += 1
        self.close_deadlines.append(deadline_s)
        self.close_entered.set()
        await self.close_release.wait()


class _Factory:
    def __init__(self, handle: _Handle) -> None:
        self.handle = handle
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls = 0

    def probe(self, model: ModelSpec) -> CapabilityReport:
        return CapabilityReport(
            supported=model.model_id == MODEL,
            backend=BackendKind.FUSED_MTP_MLX,
        )

    async def load(self, runtime: RuntimeKey, config: LoadConfig) -> _Handle:
        del config
        assert runtime == RUNTIME
        self.calls += 1
        self.started.set()
        await self.release.wait()
        return self.handle


def _services(
    factory: _Factory,
    *,
    trusted: bool = True,
) -> tuple[RuntimeManager, ReadinessService]:
    roles = RoleDirectory(
        (
            RoleSpec(
                name=RoleName.MAIN,
                port=8100,
                requested_model=MODEL,
                backend=BackendKind.FUSED_MTP_MLX,
                revision=RUNTIME.revision,
                model_dir="/models/exact-revision",
                pinned=True,
            ),
        )
    )
    readiness = ReadinessService(
        roles,
        receipt={"role_manifest_sha256": "build-receipt"},
    )
    registered_factory = (
        _bind_trusted_fused_backend_factory(
            factory,
            _TEST_MATERIALIZATION_AUTHORITY,
        )
        if trusted
        else factory
    )
    manager = RuntimeManager(
        {BackendKind.FUSED_MTP_MLX: registered_factory},
        roles=roles,
        readiness=readiness,
    )
    return manager, readiness


def test_directory_modelspec_and_ready_enum_are_not_materialization() -> None:
    factory = _Factory(_Handle(receipt=_MISSING))
    _, readiness = _services(factory)

    with pytest.raises(ValueError, match="materialization receipt"):
        readiness.mark_ready(
            RoleName.MAIN,
            loaded_model=MODEL,
            backend=BackendKind.FUSED_MTP_MLX,
        )
    assert readiness.is_ready(RoleName.MAIN) is False


@pytest.mark.asyncio
async def test_manager_rejects_and_closes_fused_handle_without_receipt() -> None:
    handle = _Handle(receipt=_MISSING)
    factory = _Factory(handle)
    factory.release.set()
    manager, readiness = _services(factory)

    with pytest.raises(RuntimeManagerError, match="materialization receipt"):
        await manager.acquire_role(RoleName.MAIN)

    assert handle.close_calls == 1
    assert readiness.snapshot(RoleName.MAIN).model_state is ModelState.DEGRADED
    assert readiness.snapshot(RoleName.MAIN).materialization is None


@pytest.mark.asyncio
async def test_plain_public_factory_cannot_publish_fabricated_receipt() -> None:
    handle = _Handle()
    factory = _Factory(handle)
    factory.release.set()
    manager, readiness = _services(factory, trusted=False)

    with pytest.raises(RuntimeManagerError, match="trusted factory authority"):
        await manager.acquire_role(RoleName.MAIN)

    assert handle.close_calls == 1
    assert manager.status(RUNTIME)["loaded"] is False
    assert readiness.is_ready(RoleName.MAIN) is False


@pytest.mark.asyncio
async def test_trusted_factory_rejects_receipt_fabricated_under_foreign_seal() -> None:
    handle = _Handle(receipt=_receipt(authority=object()))
    factory = _Factory(handle)
    factory.release.set()
    manager, readiness = _services(factory)

    with pytest.raises(RuntimeManagerError, match="trusted factory authority"):
        await manager.acquire_role(RoleName.MAIN)

    assert handle.close_calls == 1
    assert manager.status(RUNTIME)["loaded"] is False
    assert readiness.is_ready(RoleName.MAIN) is False


@pytest.mark.asyncio
async def test_runtime_mismatch_is_never_published_ready() -> None:
    foreign = replace(RUNTIME, revision="foreign")
    handle = _Handle(receipt=_receipt(foreign))
    factory = _Factory(handle)
    factory.release.set()
    manager, readiness = _services(factory)

    with pytest.raises(RuntimeManagerError, match="different runtime"):
        await manager.acquire_role(RoleName.MAIN)

    assert handle.close_calls == 1
    assert readiness.is_ready(RoleName.MAIN) is False


@pytest.mark.asyncio
async def test_cancelled_waiter_cannot_cancel_or_publish_shared_load_early() -> None:
    handle = _Handle()
    factory = _Factory(handle)
    manager, readiness = _services(factory)
    cancelled = asyncio.create_task(manager.acquire_role(RoleName.MAIN))
    survivor = asyncio.create_task(manager.acquire_role(RoleName.MAIN))
    await factory.started.wait()

    assert readiness.snapshot(RoleName.MAIN).model_state is ModelState.LOADING
    assert readiness.snapshot(RoleName.MAIN).materialization is None
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert factory.calls == 1
    assert readiness.is_ready(RoleName.MAIN) is False

    factory.release.set()
    assert await survivor is handle
    snapshot = readiness.snapshot(RoleName.MAIN)
    assert snapshot.model_state is ModelState.READY
    assert snapshot.materialization is handle.materialization_receipt
    assert readiness.is_ready(RoleName.MAIN) is True


@pytest.mark.asyncio
async def test_receipt_is_captured_once_and_same_object_is_published() -> None:
    handle = _ChangingReceiptHandle()
    factory = _Factory(handle)
    factory.release.set()
    manager, readiness = _services(factory)

    assert await manager.acquire_role(RoleName.MAIN) is handle

    snapshot = readiness.snapshot(RoleName.MAIN)
    assert handle.receipt_reads == 1
    assert snapshot.materialization is handle.initial_receipt
    assert manager._records[RUNTIME].materialization is handle.initial_receipt


@pytest.mark.asyncio
async def test_ready_publication_occurs_inside_manager_lifecycle_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handle = _Handle()
    factory = _Factory(handle)
    factory.release.set()
    manager, readiness = _services(factory)
    observed_lock_state: list[bool] = []
    original_mark_ready = readiness.mark_ready

    def mark_ready(*args: object, **kwargs: object) -> RoleSnapshot:
        observed_lock_state.append(manager._lock.locked())
        record = manager._records[RUNTIME]
        assert record.handle is handle
        assert record.state is ModelState.READY
        assert record.materialization is handle._receipt
        return original_mark_ready(*args, **kwargs)

    monkeypatch.setattr(readiness, "mark_ready", mark_ready)
    await manager.acquire_role(RoleName.MAIN)

    assert observed_lock_state == [True]
    assert manager.status(RUNTIME)["state"] == ModelState.READY.value
    assert readiness.is_ready(RoleName.MAIN) is True


@pytest.mark.asyncio
async def test_shutdown_bounds_an_already_running_unload_by_original_deadline() -> None:
    handle = _BlockingCloseHandle()
    factory = _Factory(handle)
    factory.release.set()
    manager, _ = _services(factory)
    await manager.acquire_role(RoleName.MAIN)

    unload = asyncio.create_task(manager.unload(RUNTIME, deadline_s=60.0))
    await handle.close_entered.wait()
    with pytest.raises(RuntimeManagerError, match="shutdown incomplete"):
        await asyncio.wait_for(manager.shutdown(deadline_s=0.01), timeout=0.2)

    assert unload.done() is False
    assert handle.close_deadlines == [60.0]
    handle.close_release.set()
    assert await unload is True
    await manager.shutdown(deadline_s=60.0)


@pytest.mark.asyncio
async def test_shutdown_during_load_suppresses_ready_and_closes_eventual_handle(
) -> None:
    handle = _Handle()
    factory = _Factory(handle)
    manager, readiness = _services(factory)
    load = asyncio.create_task(manager.acquire_role(RoleName.MAIN))
    await factory.started.wait()
    shutdown = asyncio.create_task(manager.shutdown(deadline_s=2.0))
    await asyncio.sleep(0)

    factory.release.set()
    with pytest.raises(RuntimeError, match="shutdown"):
        await load
    await shutdown

    assert handle.close_calls == 1
    snapshot = readiness.snapshot(RoleName.MAIN)
    assert snapshot.model_state is ModelState.COLD
    assert snapshot.transition == "shutdown_during_load"
    assert snapshot.materialization is None
    assert readiness.is_ready(RoleName.MAIN) is False


@pytest.mark.asyncio
async def test_timed_out_shutdown_tracks_late_cleanup_without_deadline_extension(
) -> None:
    handle = _BlockingCloseHandle()
    factory = _Factory(handle)
    manager, readiness = _services(factory)
    load = asyncio.create_task(manager.acquire_role(RoleName.MAIN))
    await factory.started.wait()

    with pytest.raises(RuntimeManagerError, match="shutdown incomplete"):
        await manager.shutdown(deadline_s=0.0)

    factory.release.set()
    await handle.close_entered.wait()
    with pytest.raises(RuntimeManagerError, match="shutdown incomplete"):
        await manager.shutdown(deadline_s=60.0)

    status = manager.status(RUNTIME)
    assert status["state"] == ModelState.UNLOADING.value
    assert status["loaded"] is False
    assert status["loading"] is False
    assert status["cleaning"] is True
    assert manager._closed is False
    assert readiness.is_ready(RoleName.MAIN) is False
    assert handle.close_deadlines == [0.0]

    handle.close_release.set()
    with pytest.raises(RuntimeError, match="shutdown"):
        await load
    await manager.shutdown(deadline_s=60.0)

    assert handle.close_calls == 1
    assert manager._closed is True
    assert manager.status(RUNTIME)["cleaning"] is False
    assert readiness.snapshot(RoleName.MAIN).model_state is ModelState.COLD


def test_every_non_ready_transition_clears_materialization() -> None:
    transitions = (
        "unloading",
        "cold",
        "loading",
        "degraded",
        "dead",
    )
    for transition in transitions:
        _, readiness = _services(_Factory(_Handle()))
        readiness.mark_ready(
            RoleName.MAIN,
            loaded_model=MODEL,
            backend=BackendKind.FUSED_MTP_MLX,
            materialization=_receipt(),
        )
        if transition == "unloading":
            snapshot = readiness.mark_unloading(RoleName.MAIN)
        elif transition == "cold":
            snapshot = readiness.mark_cold(RoleName.MAIN)
        elif transition == "loading":
            snapshot = readiness.mark_loading(RoleName.MAIN)
        elif transition == "degraded":
            snapshot = readiness.mark_degraded(RoleName.MAIN, "failed")
        else:
            snapshot = readiness.mark_dead(RoleName.MAIN, "dead")
        assert snapshot.model_state is not ModelState.READY
        assert snapshot.materialization is None

    _, readiness = _services(_Factory(_Handle()))
    readiness.mark_ready(
        RoleName.MAIN,
        loaded_model=MODEL,
        backend=BackendKind.FUSED_MTP_MLX,
        materialization=_receipt(),
    )
    readiness.mark_dead(RoleName.MAIN, "dead")
    snapshot = readiness.mark_alive(RoleName.MAIN)
    assert snapshot.model_state is ModelState.COLD
    assert snapshot.materialization is None
