"""Supervisor-backed process recycle after heavyweight runtimes become idle."""

from __future__ import annotations

import asyncio
import os
import signal
import threading
from collections.abc import Callable
from contextlib import suppress

from .utils.logger import logger

TerminateProcess = Callable[[], None]


class RuntimeRecycleInProgress(RuntimeError):
    """Raised before heavy work is admitted into a draining process."""


_heavy_admission_lock = threading.RLock()
_heavy_runtime_draining = False


def admit_heavy_runtime_work(admit: Callable[[], None]) -> None:
    """Atomically admit heavy work or reject it before model loading starts."""
    with _heavy_admission_lock:
        if _heavy_runtime_draining:
            raise RuntimeRecycleInProgress(
                "MLX runtime is recycling; retry after the supervisor restart"
            )
        admit()


def _begin_heavy_runtime_drain() -> bool:
    global _heavy_runtime_draining
    with _heavy_admission_lock:
        if _heavy_runtime_draining:
            return False
        _heavy_runtime_draining = True
        return True


def _cancel_heavy_runtime_drain() -> None:
    global _heavy_runtime_draining
    with _heavy_admission_lock:
        _heavy_runtime_draining = False


def _supervised_recycle_enabled() -> bool:
    return (
        os.environ.get("MLX_BATCH_UNDER_SUPERVISOR") == "1"
        and os.environ.get("MLX_BATCH_IDLE_PROCESS_RECYCLE") == "1"
    )


def _terminate_process() -> None:
    os.kill(os.getpid(), signal.SIGTERM)


class IdleProcessRecycler:
    """Recycle allocator-heavy parent state only when every MLX lane is idle."""

    def __init__(
        self,
        *,
        enabled: bool | None = None,
        retry_seconds: float = 0.25,
        terminate_process: TerminateProcess = _terminate_process,
    ) -> None:
        self._enabled = _supervised_recycle_enabled() if enabled is None else enabled
        self._retry_seconds = max(0.01, retry_seconds)
        self._terminate_process = terminate_process
        self._state_lock = threading.Lock()
        self._pending_reason: str | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._wake: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if not self._enabled or self._task is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._wake = asyncio.Event()
        self._task = asyncio.create_task(
            self._run(),
            name="idle-process-recycler",
        )
        with self._state_lock:
            pending = self._pending_reason is not None
        if pending:
            self._wake.set()

    async def stop(self) -> None:
        task = self._task
        self._task = None
        self._loop = None
        self._wake = None
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    def request(self, reason: str) -> None:
        """Request a recycle without blocking a cache-cleanup thread."""
        if not self._enabled:
            return
        with self._state_lock:
            self._pending_reason = reason
            loop = self._loop
            wake = self._wake
        if loop is not None and wake is not None:
            loop.call_soon_threadsafe(wake.set)

    def notify_state_change(self) -> None:
        """Wake a pending recycle after an image worker or lease becomes idle."""
        with self._state_lock:
            pending = self._pending_reason is not None
            loop = self._loop
            wake = self._wake
        if pending and loop is not None and wake is not None:
            loop.call_soon_threadsafe(wake.set)

    async def attempt_recycle(self) -> bool:
        """Perform one atomic cross-runtime idle check and terminate if safe."""
        if not self._enabled:
            return False

        from .aux_runtime import (  # noqa: PLC0415
            aux_runtime_recycle_guard,
            aux_runtime_recycle_ready,
        )
        from .chat.mlx.runtime_leases import (  # noqa: PLC0415
            all_runtime_retirement_guard,
            list_runtime_leases,
        )
        from .chat.mlx.wrapper_cache import wrapper_cache  # noqa: PLC0415
        from .images.image_runtime import (  # noqa: PLC0415
            image_runtime_recycle_guard,
            image_runtime_recycle_ready,
        )

        # Avoid repeatedly closing the admission gate while another runtime is
        # visibly busy. These are observer checks only; final authority is below.
        if (
            list_runtime_leases()
            or wrapper_cache.get_runtime_keys()
            or not image_runtime_recycle_ready()
            or not aux_runtime_recycle_ready()
        ):
            return False
        if not _begin_heavy_runtime_drain():
            return False

        try:
            # Canonical cross-runtime lock order: wrapper -> leases -> images -> aux.
            # TTL/LRU cleanup already holds the wrapper lock before consulting
            # leases, so the recycler must never acquire these in reverse.
            with (
                wrapper_cache.process_recycle_guard() as wrappers_idle,
                all_runtime_retirement_guard() as llm_idle,
            ):
                async with image_runtime_recycle_guard() as images_idle:
                    with aux_runtime_recycle_guard() as aux_idle:
                        if not (
                            wrappers_idle and llm_idle and images_idle and aux_idle
                        ):
                            _cancel_heavy_runtime_drain()
                            return False
                        with self._state_lock:
                            reason = self._pending_reason or "manual-attempt"
                            self._pending_reason = None
                        logger.warning(
                            "Recycling idle supervised process to release allocator RSS: %s",
                            reason,
                        )
                        self._terminate_process()
                        return True
        except BaseException:
            _cancel_heavy_runtime_drain()
            raise

    async def _run(self) -> None:
        wake = self._wake
        if wake is None:
            return
        while True:
            await wake.wait()
            wake.clear()
            while True:
                with self._state_lock:
                    pending = self._pending_reason is not None
                if not pending:
                    break
                await asyncio.sleep(self._retry_seconds)
                if await self.attempt_recycle():
                    break


_idle_process_recycler = IdleProcessRecycler()


async def start_idle_process_recycler() -> None:
    await _idle_process_recycler.start()


async def stop_idle_process_recycler() -> None:
    await _idle_process_recycler.stop()


def request_idle_process_recycle(reason: str) -> None:
    _idle_process_recycler.request(reason)


def notify_idle_process_recycler() -> None:
    _idle_process_recycler.notify_state_change()
