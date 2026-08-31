from __future__ import annotations

from contextlib import contextmanager

import pytest

from mlx_batch_server import aux_runtime as aux_runtime_module
from mlx_batch_server.aux_runtime import AuxiliaryRuntimeManager
from mlx_batch_server.chat.mlx import runtime_leases
from mlx_batch_server.chat.mlx.wrapper_cache import wrapper_cache
from mlx_batch_server.images import image_runtime as image_runtime_module
from mlx_batch_server.runtime_recycle import (
    IdleProcessRecycler,
    _cancel_heavy_runtime_drain,
)


@pytest.fixture(autouse=True)
def _reset_runtime_state(monkeypatch):
    @contextmanager
    def empty_wrapper_guard():
        yield True

    runtime_leases.clear_runtime_leases()
    _cancel_heavy_runtime_drain()
    monkeypatch.setattr(wrapper_cache, "get_runtime_keys", lambda: [])
    monkeypatch.setattr(wrapper_cache, "process_recycle_guard", empty_wrapper_guard)
    monkeypatch.setattr(image_runtime_module, "_image_runtime_pool", None)
    monkeypatch.setattr(aux_runtime_module, "_aux_runtime_manager", None)
    yield
    runtime = aux_runtime_module._aux_runtime_manager
    aux_runtime_module._aux_runtime_manager = None
    if runtime is not None:
        runtime.shutdown()
    runtime_leases.clear_runtime_leases()
    _cancel_heavy_runtime_drain()


@pytest.mark.parametrize("lane", ["embeddings", "tts", "stt"])
def test_auxiliary_lane_retires_only_after_600_idle_seconds(lane):
    now = [100.0]
    unloaded: list[str] = []
    runtime = AuxiliaryRuntimeManager(
        idle_ttl_seconds=600,
        clock=lambda: now[0],
        start_cleanup_thread=False,
    )

    with runtime.operation(lane, "model-a"):
        runtime.register(lane, "model-a", lambda: unloaded.append("model-a"))
        now[0] = 10_000.0
        assert runtime.retire_expired() == []

    assert runtime.snapshot()["active_operations"] == 0
    now[0] = 10_599.99
    assert runtime.retire_expired() == []
    now[0] = 10_600.0
    assert runtime.retire_expired() == [(lane, "model-a")]
    assert unloaded == ["model-a"]
    assert runtime.snapshot()["resident_count"] == 0


@pytest.mark.asyncio
async def test_auxiliary_active_and_resident_work_block_process_recycle(monkeypatch):
    runtime = AuxiliaryRuntimeManager(
        idle_ttl_seconds=600,
        start_cleanup_thread=False,
    )
    monkeypatch.setattr(aux_runtime_module, "_aux_runtime_manager", runtime)
    terminations: list[bool] = []
    recycler = IdleProcessRecycler(
        enabled=True,
        terminate_process=lambda: terminations.append(True),
    )

    with runtime.operation("tts", "voice-model"):
        runtime.register("tts", "voice-model", lambda: None)
        assert await recycler.attempt_recycle() is False

    assert await recycler.attempt_recycle() is False
    runtime.forget("tts", "voice-model")
    assert await recycler.attempt_recycle() is True
    assert terminations == [True]
