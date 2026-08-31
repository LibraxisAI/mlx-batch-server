"""One lifecycle owner for native embeddings, TTS, and STT runtimes."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .utils.logger import logger

RuntimeKey = tuple[str, str]
UnloadCallback = Callable[[], object]


@dataclass
class _ResidentRuntime:
    unload: UnloadCallback
    deadline: float
    retiring: bool = False


class AuxiliaryRuntimeManager:
    """Coordinate admission, residency, and one idle timer for auxiliary MLX lanes."""

    def __init__(
        self,
        idle_ttl_seconds: float = 600,
        *,
        clock: Callable[[], float] = time.monotonic,
        start_cleanup_thread: bool = True,
    ) -> None:
        self._idle_ttl_seconds = max(0.0, float(idle_ttl_seconds))
        self._clock = clock
        self._start_cleanup_thread = start_cleanup_thread
        self._condition = threading.Condition(threading.RLock())
        self._resident: dict[RuntimeKey, _ResidentRuntime] = {}
        self._active: dict[RuntimeKey, int] = {}
        self._stop = False
        self._cleanup_thread: threading.Thread | None = None

    @staticmethod
    def _key(lane: str, model_id: str) -> RuntimeKey:
        return (lane.strip().lower(), model_id)

    def _ensure_cleanup_thread_locked(self) -> None:
        if not self._start_cleanup_thread or self._cleanup_thread is not None:
            return
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            name="aux-runtime-idle-retirement",
            daemon=True,
        )
        self._cleanup_thread.start()

    @contextmanager
    def operation(self, lane: str, model_id: str) -> Iterator[None]:
        """Admit one queued/active operation before model loading can begin."""
        key = self._key(lane, model_id)
        from .runtime_recycle import admit_heavy_runtime_work  # noqa: PLC0415

        def admit() -> None:
            with self._condition:
                while (entry := self._resident.get(key)) is not None and entry.retiring:
                    self._condition.wait()
                self._active[key] = self._active.get(key, 0) + 1

        admit_heavy_runtime_work(admit)
        try:
            yield
        finally:
            with self._condition:
                remaining = self._active.get(key, 0) - 1
                if remaining > 0:
                    self._active[key] = remaining
                else:
                    self._active.pop(key, None)
                entry = self._resident.get(key)
                if entry is not None:
                    entry.deadline = self._clock() + self._idle_ttl_seconds
                self._condition.notify_all()
            from .runtime_recycle import notify_idle_process_recycler  # noqa: PLC0415

            notify_idle_process_recycler()

    def register(
        self,
        lane: str,
        model_id: str,
        unload: UnloadCallback,
    ) -> None:
        """Register actual resident weights without renewing observer reads."""
        key = self._key(lane, model_id)
        with self._condition:
            while (entry := self._resident.get(key)) is not None and entry.retiring:
                self._condition.wait()
            self._resident[key] = _ResidentRuntime(
                unload=unload,
                deadline=self._clock() + self._idle_ttl_seconds,
            )
            self._ensure_cleanup_thread_locked()
            self._condition.notify_all()

    def forget(self, lane: str, model_id: str) -> bool:
        """Remove lifecycle metadata before an explicit owner-driven unload."""
        key = self._key(lane, model_id)
        with self._condition:
            while (entry := self._resident.get(key)) is not None and entry.retiring:
                self._condition.wait()
            removed = self._resident.pop(key, None) is not None
            self._condition.notify_all()
            return removed

    def forget_lane(self, lane: str) -> list[str]:
        normalized_lane = lane.strip().lower()
        with self._condition:
            while any(
                key[0] == normalized_lane and entry.retiring
                for key, entry in self._resident.items()
            ):
                self._condition.wait()
            keys = [key for key in self._resident if key[0] == normalized_lane]
            for key in keys:
                self._resident.pop(key, None)
            self._condition.notify_all()
        return [key[1] for key in keys]

    def replace_lane_runtime(
        self,
        lane: str,
        model_id: str,
        unload: UnloadCallback,
    ) -> None:
        """Track a backend such as Whisper that can own only one model."""
        self.forget_lane(lane)
        self.register(lane, model_id, unload)

    def retire_expired(self, *, now: float | None = None) -> list[RuntimeKey]:
        """Synchronously retire due idle runtimes; exposed for deterministic tests."""
        current_time = self._clock() if now is None else now
        with self._condition:
            due = [
                (key, entry)
                for key, entry in self._resident.items()
                if not entry.retiring
                and self._active.get(key, 0) == 0
                and entry.deadline <= current_time
            ]
            for _, entry in due:
                entry.retiring = True

        retired: list[RuntimeKey] = []
        for key, entry in due:
            failed = False
            try:
                entry.unload()
            except Exception:
                failed = True
                logger.exception("Failed to retire auxiliary runtime %s:%s", *key)
            with self._condition:
                if self._resident.get(key) is entry:
                    if failed:
                        entry.retiring = False
                        entry.deadline = self._clock() + self._idle_ttl_seconds
                    else:
                        self._resident.pop(key, None)
                        retired.append(key)
                self._condition.notify_all()

        if retired:
            from .runtime_recycle import request_idle_process_recycle  # noqa: PLC0415

            request_idle_process_recycle("aux-runtime-idle-ttl")
        return retired

    def _cleanup_loop(self) -> None:
        while True:
            with self._condition:
                if self._stop:
                    return
                deadlines = [
                    entry.deadline
                    for key, entry in self._resident.items()
                    if not entry.retiring and self._active.get(key, 0) == 0
                ]
                timeout = None
                if deadlines:
                    timeout = max(0.0, min(deadlines) - self._clock())
                self._condition.wait(timeout=timeout)
                if self._stop:
                    return
            self.retire_expired()

    def snapshot(self) -> dict[str, Any]:
        """Observer-only residency; never renews idle deadlines."""
        with self._condition:
            resident_by_lane: dict[str, list[str]] = {}
            active_by_lane: dict[str, int] = {}
            for lane, model_id in self._resident:
                resident_by_lane.setdefault(lane, []).append(model_id)
            for (lane, _), count in self._active.items():
                active_by_lane[lane] = active_by_lane.get(lane, 0) + count
            return {
                "resident_by_lane": {
                    lane: sorted(models) for lane, models in resident_by_lane.items()
                },
                "active_by_lane": dict(sorted(active_by_lane.items())),
                "resident_count": len(self._resident),
                "active_operations": sum(self._active.values()),
                "idle_ttl_seconds": self._idle_ttl_seconds,
            }

    @contextmanager
    def process_recycle_guard(self) -> Iterator[bool]:
        """Freeze auxiliary admission state for the final recycle decision."""
        with self._condition:
            yield not self._resident and not self._active

    def recycle_ready(self) -> bool:
        with self._condition:
            return not self._resident and not self._active

    def shutdown(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
            cleanup_thread = self._cleanup_thread
            self._cleanup_thread = None
        if (
            cleanup_thread is not None
            and cleanup_thread is not threading.current_thread()
        ):
            cleanup_thread.join(timeout=2)

        with self._condition:
            entries = list(self._resident.values())
            self._resident.clear()
        for entry in entries:
            try:
                entry.unload()
            except Exception:
                logger.exception("Failed to unload auxiliary runtime during shutdown")


_aux_runtime_manager: AuxiliaryRuntimeManager | None = None


def get_aux_runtime_manager() -> AuxiliaryRuntimeManager:
    global _aux_runtime_manager
    if _aux_runtime_manager is None:
        from .core.config import get_settings  # noqa: PLC0415

        _aux_runtime_manager = AuxiliaryRuntimeManager(
            get_settings().aux_model_idle_ttl_seconds
        )
    return _aux_runtime_manager


def auxiliary_runtime_operation(lane: str, model_id: str):
    return get_aux_runtime_manager().operation(lane, model_id)


def register_aux_runtime(
    lane: str,
    model_id: str,
    unload: UnloadCallback,
) -> None:
    get_aux_runtime_manager().register(lane, model_id, unload)


def forget_aux_runtime(lane: str, model_id: str) -> bool:
    runtime = _aux_runtime_manager
    return False if runtime is None else runtime.forget(lane, model_id)


def forget_aux_runtime_lane(lane: str) -> list[str]:
    runtime = _aux_runtime_manager
    return [] if runtime is None else runtime.forget_lane(lane)


def replace_aux_runtime(
    lane: str,
    model_id: str,
    unload: UnloadCallback,
) -> None:
    get_aux_runtime_manager().replace_lane_runtime(lane, model_id, unload)


def get_aux_runtime_snapshot() -> dict[str, Any]:
    runtime = _aux_runtime_manager
    if runtime is None:
        from .core.config import get_settings  # noqa: PLC0415

        return {
            "resident_by_lane": {},
            "active_by_lane": {},
            "resident_count": 0,
            "active_operations": 0,
            "idle_ttl_seconds": get_settings().aux_model_idle_ttl_seconds,
        }
    return runtime.snapshot()


@contextmanager
def aux_runtime_recycle_guard() -> Iterator[bool]:
    runtime = _aux_runtime_manager
    if runtime is None:
        yield True
        return
    with runtime.process_recycle_guard() as idle:
        yield idle


def aux_runtime_recycle_ready() -> bool:
    runtime = _aux_runtime_manager
    return True if runtime is None else runtime.recycle_ready()


def shutdown_aux_runtime_manager() -> None:
    global _aux_runtime_manager
    runtime = _aux_runtime_manager
    _aux_runtime_manager = None
    if runtime is not None:
        runtime.shutdown()
