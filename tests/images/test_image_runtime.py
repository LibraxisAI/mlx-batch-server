import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mlx_batch_server.chat.openai.models import models as model_routes
from mlx_batch_server.chat.openai.models.schema import (
    ModelLoadRequest,
    ModelUnloadRequest,
)
from mlx_batch_server.images import image_runtime as image_runtime_module
from mlx_batch_server.images.image_runtime import ImageRuntimePool
from mlx_batch_server.images.schema import ImageGenerationRequest
from mlx_batch_server.main import create_app


class TrackingExecutor(ThreadPoolExecutor):
    def __init__(self) -> None:
        super().__init__(max_workers=1)
        self.shutdown_calls = 0

    def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
        self.shutdown_calls += 1
        super().shutdown(wait=wait, cancel_futures=cancel_futures)


def image_result(operation: str, worker_pid: int) -> dict[str, Any]:
    if operation == "load":
        return {"already_loaded": False, "worker_pid": worker_pid}
    if operation == "unload":
        return {"unloaded": True, "worker_pid": worker_pid}
    if operation == "clear":
        return {"unloaded_models": ["image-model"], "worker_pid": worker_pid}
    return {
        "data": [{"b64_json": "ZmFrZQ=="}],
        "worker_pid": worker_pid,
    }


@pytest.mark.asyncio
async def test_real_process_pool_control_path_runs_in_child_process():
    runtime = ImageRuntimePool(idle_ttl_seconds=60)
    result = await runtime._execute("clear", {})
    assert result is not None
    assert result["worker_pid"] != os.getpid()
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_prewarm_and_generation_share_one_worker_queue():
    executor = TrackingExecutor()
    calls: list[tuple[str, int]] = []
    prewarm_started = threading.Event()
    release_prewarm = threading.Event()

    def worker(operation: str, _payload: dict[str, object]) -> dict[str, Any]:
        worker_id = threading.get_ident()
        calls.append((operation, worker_id))
        if operation == "load":
            prewarm_started.set()
            release_prewarm.wait(timeout=2)
        return image_result(operation, worker_id)

    runtime = ImageRuntimePool(
        idle_ttl_seconds=60,
        executor_factory=lambda: executor,
        worker_operation=worker,
    )
    prewarm = asyncio.create_task(runtime.prewarm("image-model"))
    await asyncio.to_thread(prewarm_started.wait, 2)
    generation = asyncio.create_task(
        runtime.generate(ImageGenerationRequest(model="image-model", prompt="test"))
    )
    await asyncio.sleep(0.02)

    assert [operation for operation, _ in calls] == ["load"]
    release_prewarm.set()
    await prewarm
    images = await generation

    assert images[0].b64_json == "ZmFrZQ=="
    assert [operation for operation, _ in calls] == ["load", "generate"]
    assert len({worker_id for _, worker_id in calls}) == 1
    await runtime.shutdown()


@pytest.mark.asyncio
async def test_specific_unload_and_clear_retire_the_worker_that_handled_them():
    executors: list[TrackingExecutor] = []

    def executor_factory() -> TrackingExecutor:
        executor = TrackingExecutor()
        executors.append(executor)
        return executor

    def worker(operation: str, _payload: dict[str, object]) -> dict[str, Any]:
        return image_result(operation, threading.get_ident())

    runtime = ImageRuntimePool(
        idle_ttl_seconds=60,
        executor_factory=executor_factory,
        worker_operation=worker,
    )

    await runtime.prewarm("image-model")
    assert runtime.snapshot()["resident_models"] == ["image-model"]
    assert await runtime.unload("image-model") is True
    assert executors[0].shutdown_calls == 1
    assert runtime.snapshot()["running"] is False
    assert runtime.snapshot()["resident_models"] == []

    await runtime.prewarm("image-model")
    assert await runtime.clear() == ["image-model"]
    assert executors[1].shutdown_calls == 1
    assert runtime.snapshot()["running"] is False


@pytest.mark.asyncio
async def test_cold_unload_and_clear_do_not_spawn_an_empty_worker():
    executor_factory_calls = 0

    def executor_factory() -> TrackingExecutor:
        nonlocal executor_factory_calls
        executor_factory_calls += 1
        return TrackingExecutor()

    runtime = ImageRuntimePool(
        idle_ttl_seconds=60,
        executor_factory=executor_factory,
    )

    assert await runtime.unload("image-model") is False
    assert await runtime.clear() == []
    assert executor_factory_calls == 0


@pytest.mark.asyncio
async def test_idle_retirement_waits_until_active_operation_finishes():
    executor = TrackingExecutor()
    started = threading.Event()
    release = threading.Event()

    def worker(operation: str, _payload: dict[str, object]) -> dict[str, Any]:
        started.set()
        release.wait(timeout=2)
        return image_result(operation, threading.get_ident())

    runtime = ImageRuntimePool(
        idle_ttl_seconds=0.03,
        executor_factory=lambda: executor,
        worker_operation=worker,
    )
    operation = asyncio.create_task(runtime.prewarm("image-model"))
    await asyncio.to_thread(started.wait, 2)
    await asyncio.sleep(0.06)

    assert executor.shutdown_calls == 0
    assert runtime.snapshot()["active_operations"] == 1

    release.set()
    await operation
    await asyncio.sleep(0.08)
    assert executor.shutdown_calls == 1
    assert runtime.snapshot()["running"] is False


@pytest.mark.asyncio
async def test_snapshot_does_not_renew_idle_ttl_and_shutdown_is_idempotent():
    executor = TrackingExecutor()

    def worker(operation: str, _payload: dict[str, object]) -> dict[str, Any]:
        return image_result(operation, threading.get_ident())

    runtime = ImageRuntimePool(
        idle_ttl_seconds=0.04,
        executor_factory=lambda: executor,
        worker_operation=worker,
    )
    await runtime.prewarm("image-model")
    await asyncio.sleep(0.025)
    assert runtime.snapshot()["running"] is True
    await asyncio.sleep(0.04)

    assert executor.shutdown_calls == 1
    assert runtime.snapshot()["running"] is False
    await runtime.shutdown()
    assert executor.shutdown_calls == 1


@pytest.mark.asyncio
async def test_shutdown_waits_for_queued_active_work_before_retiring_worker():
    executor = TrackingExecutor()
    started = threading.Event()
    release = threading.Event()

    def worker(operation: str, _payload: dict[str, object]) -> dict[str, Any]:
        started.set()
        release.wait(timeout=2)
        return image_result(operation, threading.get_ident())

    runtime = ImageRuntimePool(
        idle_ttl_seconds=60,
        executor_factory=lambda: executor,
        worker_operation=worker,
    )
    operation = asyncio.create_task(runtime.prewarm("image-model"))
    await asyncio.to_thread(started.wait, 2)
    shutdown = asyncio.create_task(runtime.shutdown())
    await asyncio.sleep(0.02)

    assert shutdown.done() is False
    assert executor.shutdown_calls == 0

    release.set()
    await operation
    await shutdown
    assert executor.shutdown_calls == 1


@pytest.mark.asyncio
async def test_image_model_control_routes_use_the_shared_runtime_owner(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    class FakeRuntime:
        async def prewarm(self, model: str) -> bool:
            calls.append(("load", model))
            return False

        async def unload(self, model: str) -> bool:
            calls.append(("unload", model))
            return True

        async def clear(self) -> list[str]:
            calls.append(("clear", None))
            return ["image-model"]

    runtime = FakeRuntime()
    monkeypatch.setattr(
        image_runtime_module,
        "get_image_runtime_pool",
        lambda: runtime,
    )

    loaded = await model_routes.load_model(
        ModelLoadRequest(model="image-model", task="images"),
        _auth={},
    )
    unloaded = await model_routes.unload_model(
        ModelUnloadRequest(model="image-model", task="images"),
        _auth={},
    )
    cleared = await model_routes.unload_model(
        ModelUnloadRequest(task="images"),
        _auth={},
    )

    assert loaded.status == "loaded"
    assert unloaded.status == "unloaded"
    assert cleared.status == "cleared"
    assert calls == [
        ("load", "image-model"),
        ("unload", "image-model"),
        ("clear", None),
    ]


def test_application_shutdown_closes_image_runtime(monkeypatch):
    shutdown_calls = 0

    async def fake_shutdown() -> None:
        nonlocal shutdown_calls
        shutdown_calls += 1

    monkeypatch.setattr(
        image_runtime_module,
        "shutdown_image_runtime_pool",
        fake_shutdown,
    )

    with TestClient(create_app()):
        pass

    assert shutdown_calls == 1
