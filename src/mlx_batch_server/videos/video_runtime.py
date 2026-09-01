"""Serialized runtime owner for MLX video subprocess jobs."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from ..runtime_recycle import admit_heavy_runtime_work, notify_idle_process_recycler
from .schema import VideoArtifact, VideoGenerationRequest
from .video_service import run_video_worker_operation

WorkerOperation = Callable[[str, dict[str, Any]], dict[str, Any]]


class VideoRuntime:
    def __init__(
        self,
        *,
        executor: Executor | None = None,
        worker_operation: WorkerOperation = run_video_worker_operation,
    ) -> None:
        self._executor = executor or ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="mlx-video"
        )
        self._owns_executor = executor is None
        self._worker_operation = worker_operation
        self._lock = asyncio.Lock()
        self._active_operations = 0

    async def generate(self, request: VideoGenerationRequest) -> VideoArtifact:
        async with self._lock:
            admit_heavy_runtime_work(self._admit)
            try:
                result = await asyncio.get_running_loop().run_in_executor(
                    self._executor,
                    self._worker_operation,
                    "generate",
                    request.model_dump(mode="json"),
                )
                return VideoArtifact.model_validate(result["artifact"])
            finally:
                self._active_operations -= 1
                notify_idle_process_recycler()

    def _admit(self) -> None:
        self._active_operations += 1

    def snapshot(self) -> dict[str, object]:
        return {
            "running": self._active_operations > 0,
            "active_operations": self._active_operations,
            "resident_models": [],
            "worker_mode": "isolated-subprocess",
        }

    @asynccontextmanager
    async def process_recycle_guard(self):
        async with self._lock:
            yield self._active_operations == 0

    async def shutdown(self) -> None:
        if self._owns_executor:
            executor = self._executor
            self._owns_executor = False
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=False)


_video_runtime: VideoRuntime | None = None


def get_video_runtime() -> VideoRuntime:
    global _video_runtime
    if _video_runtime is None:
        _video_runtime = VideoRuntime()
    return _video_runtime


async def shutdown_video_runtime() -> None:
    global _video_runtime
    runtime = _video_runtime
    _video_runtime = None
    if runtime is not None:
        await runtime.shutdown()


def get_video_runtime_snapshot() -> dict[str, object]:
    if _video_runtime is None:
        return {
            "running": False,
            "active_operations": 0,
            "resident_models": [],
            "worker_mode": "isolated-subprocess",
        }
    return _video_runtime.snapshot()


def video_runtime_recycle_ready() -> bool:
    return _video_runtime is None or _video_runtime.snapshot()["active_operations"] == 0


@asynccontextmanager
async def video_runtime_recycle_guard():
    if _video_runtime is None:
        yield True
        return
    async with _video_runtime.process_recycle_guard() as idle:
        yield idle
