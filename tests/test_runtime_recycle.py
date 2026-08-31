from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from mlx_batch_server.chat.mlx import runtime_leases
from mlx_batch_server.chat.mlx.wrapper_cache import wrapper_cache
from mlx_batch_server.images import image_runtime as image_runtime_module
from mlx_batch_server.images.image_runtime import ImageRuntimePool
from mlx_batch_server.runtime_recycle import (
    IdleProcessRecycler,
    RuntimeRecycleInProgress,
    _cancel_heavy_runtime_drain,
)


def _image_result(operation: str) -> dict[str, Any]:
    if operation == "load":
        return {"already_loaded": False, "worker_pid": 1}
    return {"data": [{"b64_json": "ZmFrZQ=="}], "worker_pid": 1}


@pytest.fixture(autouse=True)
def _reset_runtime_state(monkeypatch):
    runtime_leases.clear_runtime_leases()
    _cancel_heavy_runtime_drain()
    monkeypatch.setattr(wrapper_cache, "get_runtime_keys", lambda: [])
    monkeypatch.setattr(image_runtime_module, "_image_runtime_pool", None)
    yield
    runtime_leases.clear_runtime_leases()
    _cancel_heavy_runtime_drain()


@pytest.mark.asyncio
async def test_llm_lease_blocks_recycle_until_inference_releases_it():
    terminations: list[bool] = []
    recycler = IdleProcessRecycler(
        enabled=True,
        terminate_process=lambda: terminations.append(True),
    )

    runtime_leases.acquire_runtime_lease("model-27b")
    assert await recycler.attempt_recycle() is False
    assert terminations == []

    runtime_leases.release_runtime_lease("model-27b")
    assert await recycler.attempt_recycle() is True
    assert terminations == [True]


@pytest.mark.asyncio
async def test_image_active_and_queued_work_block_parent_recycle():
    executor = ThreadPoolExecutor(max_workers=1)
    started = threading.Event()
    release = threading.Event()

    def worker(operation: str, _payload: dict[str, object]) -> dict[str, Any]:
        started.set()
        release.wait(timeout=2)
        return _image_result(operation)

    runtime = ImageRuntimePool(
        idle_ttl_seconds=60,
        executor_factory=lambda: executor,
        worker_operation=worker,
    )
    image_runtime_module._image_runtime_pool = runtime
    recycler = IdleProcessRecycler(enabled=True, terminate_process=lambda: None)

    active = asyncio.create_task(runtime.prewarm("image-model"))
    await asyncio.to_thread(started.wait, 2)
    queued = asyncio.create_task(runtime.prewarm("image-model"))
    await asyncio.sleep(0.02)

    assert runtime.snapshot()["active_operations"] == 2
    assert await recycler.attempt_recycle() is False

    release.set()
    await asyncio.gather(active, queued)
    assert await recycler.attempt_recycle() is False

    await runtime.shutdown()
    assert await recycler.attempt_recycle() is True


@pytest.mark.asyncio
async def test_final_drain_rejects_racing_heavy_admission_before_model_start():
    drain_committed = threading.Event()
    admission_finished = threading.Event()
    admission_errors: list[Exception] = []

    def racing_admission() -> None:
        drain_committed.wait(timeout=2)
        try:
            runtime_leases.acquire_runtime_lease("racing-model")
        except Exception as exc:
            admission_errors.append(exc)
        finally:
            admission_finished.set()

    thread = threading.Thread(target=racing_admission)
    thread.start()

    def terminate() -> None:
        drain_committed.set()
        admission_finished.wait(timeout=2)

    recycler = IdleProcessRecycler(enabled=True, terminate_process=terminate)
    assert await recycler.attempt_recycle() is True
    thread.join(timeout=2)

    assert len(admission_errors) == 1
    assert isinstance(admission_errors[0], RuntimeRecycleInProgress)
    assert runtime_leases.list_runtime_leases() == []
