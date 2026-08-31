"""Single-owner lifecycle for the heavyweight image process worker."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from concurrent.futures import Executor, ProcessPoolExecutor
from contextlib import asynccontextmanager
from multiprocessing import get_context
from typing import Any

from ..core.config import get_settings
from ..runtime_recycle import admit_heavy_runtime_work, notify_idle_process_recycler
from .images_service import run_image_worker_operation
from .schema import ImageEditRequest, ImageGenerationRequest, ImageObject

WorkerOperation = Callable[[str, dict[str, object]], dict[str, Any]]
ExecutorFactory = Callable[[], Executor]


class ImageRuntimePool:
    """Serialize image work in one process and retire it after idle TTL.

    All operations, including prewarm and explicit unload, traverse the same
    single-worker executor. Queued work counts as active, so idle retirement
    cannot race a model load or generation already waiting for the worker.
    """

    def __init__(
        self,
        idle_ttl_seconds: float = 600,
        *,
        executor_factory: ExecutorFactory | None = None,
        worker_operation: WorkerOperation = run_image_worker_operation,
    ) -> None:
        self._idle_ttl_seconds = max(0.0, float(idle_ttl_seconds))
        self._executor_factory = executor_factory or self._new_process_pool
        self._worker_operation = worker_operation
        self._executor: Executor | None = None
        self._lock = asyncio.Lock()
        self._active_operations = 0
        self._idle = asyncio.Event()
        self._idle.set()
        self._idle_retirement: asyncio.Task[None] | None = None
        self._worker_pid: int | None = None
        self._resident_models: set[str] = set()

    @staticmethod
    def _new_process_pool() -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=1,
            mp_context=get_context("spawn"),
        )

    async def _execute(
        self,
        operation: str,
        payload: dict[str, object],
        *,
        start_if_missing: bool = True,
    ) -> dict[str, Any] | None:
        async with self._lock:
            executor: Executor | None = None

            def admit() -> None:
                nonlocal executor
                self._cancel_idle_retirement_locked()
                if self._executor is None:
                    if not start_if_missing:
                        return
                    self._executor = self._executor_factory()
                executor = self._executor
                self._active_operations += 1
                self._idle.clear()

            admit_heavy_runtime_work(admit)
            if executor is None:
                return None

        try:
            result = await asyncio.get_running_loop().run_in_executor(
                executor,
                self._worker_operation,
                operation,
                payload,
            )
            worker_pid = result.get("worker_pid")
            async with self._lock:
                if self._executor is executor:
                    if isinstance(worker_pid, int):
                        self._worker_pid = worker_pid
                    model_id = payload.get("model")
                    if operation in {"load", "generate", "edit"} and isinstance(
                        model_id, str
                    ):
                        self._resident_models.add(model_id)
                    elif operation == "unload" and isinstance(model_id, str):
                        self._resident_models.discard(model_id)
                    elif operation == "clear":
                        self._resident_models.clear()
            return result
        finally:
            async with self._lock:
                self._active_operations -= 1
                if self._active_operations == 0 and self._executor is executor:
                    self._idle.set()
                    self._schedule_idle_retirement_locked(executor)

    def _cancel_idle_retirement_locked(self) -> None:
        task = self._idle_retirement
        if task is not None and task is not asyncio.current_task():
            task.cancel()
        self._idle_retirement = None

    def _schedule_idle_retirement_locked(self, executor: Executor) -> None:
        self._cancel_idle_retirement_locked()
        self._idle_retirement = asyncio.create_task(
            self._retire_after_idle(executor),
            name="image-runtime-idle-retirement",
        )

    async def _retire_after_idle(self, executor: Executor) -> None:
        try:
            await asyncio.sleep(self._idle_ttl_seconds)
            async with self._lock:
                if self._executor is not executor or self._active_operations:
                    return
                self._executor = None
                self._worker_pid = None
                self._resident_models.clear()
                self._idle_retirement = None
            await asyncio.to_thread(
                executor.shutdown,
                wait=True,
                cancel_futures=False,
            )
            notify_idle_process_recycler()
        except asyncio.CancelledError:
            return

    async def prewarm(self, model_name: str) -> bool:
        result = await self._execute("load", {"model": model_name})
        assert result is not None
        return bool(result["already_loaded"])

    async def generate(
        self,
        request: ImageGenerationRequest,
    ) -> list[ImageObject]:
        result = await self._execute("generate", request.model_dump(mode="json"))
        assert result is not None
        return [ImageObject.model_validate(item) for item in result["data"]]

    async def edit(self, request: ImageEditRequest) -> list[ImageObject]:
        result = await self._execute("edit", request.model_dump(mode="json"))
        assert result is not None
        return [ImageObject.model_validate(item) for item in result["data"]]

    async def unload(self, model_name: str) -> bool:
        result = await self._execute(
            "unload",
            {"model": model_name},
            start_if_missing=False,
        )
        if result is None:
            return False
        unloaded = bool(result["unloaded"])
        if unloaded:
            await self.shutdown()
        return unloaded

    async def clear(self) -> list[str]:
        result = await self._execute("clear", {}, start_if_missing=False)
        if result is None:
            return []
        unloaded = [str(model_id) for model_id in result["unloaded_models"]]
        await self.shutdown()
        return unloaded

    def snapshot(self) -> dict[str, object]:
        """Return observer-only state without touching the idle deadline."""
        return {
            "running": self._executor is not None,
            "active_operations": self._active_operations,
            "idle_ttl_seconds": self._idle_ttl_seconds,
            "worker_pid": self._worker_pid,
            "resident_models": sorted(self._resident_models),
        }

    @asynccontextmanager
    async def process_recycle_guard(self) -> AsyncIterator[bool]:
        """Freeze image admission during the final parent recycle decision."""
        async with self._lock:
            yield self._active_operations == 0 and self._executor is None

    async def shutdown(self) -> None:
        while True:
            await self._idle.wait()
            async with self._lock:
                if self._active_operations:
                    continue
                self._cancel_idle_retirement_locked()
                executor = self._executor
                self._executor = None
                self._worker_pid = None
                self._resident_models.clear()
                break
        if executor is not None:
            await asyncio.to_thread(
                executor.shutdown,
                wait=True,
                cancel_futures=False,
            )
        notify_idle_process_recycler()


_image_runtime_pool: ImageRuntimePool | None = None


def get_image_runtime_pool() -> ImageRuntimePool:
    global _image_runtime_pool
    if _image_runtime_pool is None:
        settings = get_settings()
        _image_runtime_pool = ImageRuntimePool(settings.image_model_idle_ttl_seconds)
    return _image_runtime_pool


async def shutdown_image_runtime_pool() -> None:
    if _image_runtime_pool is not None:
        await _image_runtime_pool.shutdown()


@asynccontextmanager
async def image_runtime_recycle_guard() -> AsyncIterator[bool]:
    """Return image-lane idleness without creating a cold runtime owner."""
    runtime = _image_runtime_pool
    if runtime is None:
        yield True
        return
    async with runtime.process_recycle_guard() as idle:
        yield idle


def image_runtime_recycle_ready() -> bool:
    """Cheap preliminary check; the async guard remains final authority."""
    runtime = _image_runtime_pool
    if runtime is None:
        return True
    state = runtime.snapshot()
    return not state["running"] and state["active_operations"] == 0


def get_image_runtime_snapshot() -> dict[str, object]:
    """Return image residency without creating a cold process-pool owner."""
    runtime = _image_runtime_pool
    if runtime is None:
        return {
            "running": False,
            "active_operations": 0,
            "idle_ttl_seconds": get_settings().image_model_idle_ttl_seconds,
            "worker_pid": None,
            "resident_models": [],
        }
    return runtime.snapshot()
