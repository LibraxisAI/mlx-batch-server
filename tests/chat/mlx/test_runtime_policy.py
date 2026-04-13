from __future__ import annotations

import asyncio

import pytest

from mlx_batch_server.chat.mlx.runtime_policy import (
    endpoint_runtime_session,
    ensure_single_endpoint_llm_runtime,
)
from mlx_batch_server.chat.mlx.wrapper_cache import WrapperCacheKey


class TestSingleEndpointRuntimePolicy:
    @pytest.mark.asyncio
    async def test_no_switch_when_target_runtime_is_already_only_resident(
        self,
        monkeypatch,
    ):
        state = {"shutdown_called": False, "unloaded": []}

        monkeypatch.setattr(
            "mlx_batch_server.chat.mlx.runtime_policy.wrapper_cache.get_runtime_keys",
            lambda: [WrapperCacheKey("demo-model", None, None)],
        )
        monkeypatch.setattr(
            "mlx_batch_server.chat.mlx.runtime_policy.get_loaded_batch_models",
            lambda: ["demo-model"],
        )
        monkeypatch.setattr(
            "mlx_batch_server.chat.mlx.runtime_policy.wrapper_cache.get_loaded_models",
            lambda: ["demo-model"],
        )
        monkeypatch.setattr(
            "mlx_batch_server.chat.mlx.runtime_policy.wrapper_cache.unload_model",
            lambda model_id: state["unloaded"].append(model_id) or True,
        )

        async def fake_shutdown():
            state["shutdown_called"] = True

        monkeypatch.setattr(
            "mlx_batch_server.chat.mlx.runtime_policy.shutdown_all_coordinators",
            fake_shutdown,
        )

        result = await ensure_single_endpoint_llm_runtime("demo-model")

        assert result["switched"] is False
        assert state["shutdown_called"] is False
        assert state["unloaded"] == []

    @pytest.mark.asyncio
    async def test_switches_when_another_model_is_loaded(self, monkeypatch):
        state = {"shutdown_called": False, "unloaded": []}

        monkeypatch.setattr(
            "mlx_batch_server.chat.mlx.runtime_policy.wrapper_cache.get_runtime_keys",
            lambda: [WrapperCacheKey("model-a", None, None)],
        )
        monkeypatch.setattr(
            "mlx_batch_server.chat.mlx.runtime_policy.get_loaded_batch_models",
            lambda: ["model-a"],
        )
        monkeypatch.setattr(
            "mlx_batch_server.chat.mlx.runtime_policy.wrapper_cache.get_loaded_models",
            lambda: ["model-a"],
        )
        monkeypatch.setattr(
            "mlx_batch_server.chat.mlx.runtime_policy.wrapper_cache.unload_model",
            lambda model_id: state["unloaded"].append(model_id) or True,
        )

        async def fake_shutdown():
            state["shutdown_called"] = True

        monkeypatch.setattr(
            "mlx_batch_server.chat.mlx.runtime_policy.shutdown_all_coordinators",
            fake_shutdown,
        )

        result = await ensure_single_endpoint_llm_runtime("model-b")

        assert result["switched"] is True
        assert result["evicted_models"] == ["model-a"]
        assert state["shutdown_called"] is True
        assert state["unloaded"] == ["model-a"]

    @pytest.mark.asyncio
    async def test_switches_when_same_model_has_different_runtime_key(
        self,
        monkeypatch,
    ):
        state = {"shutdown_called": False, "unloaded": []}

        monkeypatch.setattr(
            "mlx_batch_server.chat.mlx.runtime_policy.wrapper_cache.get_runtime_keys",
            lambda: [WrapperCacheKey("demo-model", "/old-adapter", None)],
        )
        monkeypatch.setattr(
            "mlx_batch_server.chat.mlx.runtime_policy.get_loaded_batch_models",
            lambda: ["demo-model"],
        )
        monkeypatch.setattr(
            "mlx_batch_server.chat.mlx.runtime_policy.wrapper_cache.get_loaded_models",
            lambda: ["demo-model"],
        )
        monkeypatch.setattr(
            "mlx_batch_server.chat.mlx.runtime_policy.wrapper_cache.unload_model",
            lambda model_id: state["unloaded"].append(model_id) or True,
        )

        async def fake_shutdown():
            state["shutdown_called"] = True

        monkeypatch.setattr(
            "mlx_batch_server.chat.mlx.runtime_policy.shutdown_all_coordinators",
            fake_shutdown,
        )

        result = await ensure_single_endpoint_llm_runtime(
            "demo-model",
            adapter_path="/new-adapter",
        )

        assert result["switched"] is True
        assert state["shutdown_called"] is True
        assert state["unloaded"] == ["demo-model"]

    @pytest.mark.asyncio
    async def test_runtime_session_blocks_different_model_until_active_runtime_drains(
        self,
        monkeypatch,
    ):
        events: list[str] = []

        async def fake_ensure(model_id, adapter_path=None, draft_model_id=None):
            del adapter_path, draft_model_id
            events.append(f"ensure:{model_id}")
            return {
                "switched": model_id == "model-b",
                "evicted_models": ["model-a"] if model_id == "model-b" else [],
                "target_model_id": model_id,
                "previous_runtime_keys": [],
                "previous_batch_models": [],
            }

        monkeypatch.setattr(
            "mlx_batch_server.chat.mlx.runtime_policy.ensure_single_endpoint_llm_runtime",
            fake_ensure,
        )

        gate = asyncio.Event()
        release = asyncio.Event()

        async def run_model_a():
            async with endpoint_runtime_session("model-a"):
                events.append("enter:model-a")
                gate.set()
                await release.wait()
                events.append("exit:model-a")

        async def run_model_b():
            await gate.wait()
            async with endpoint_runtime_session("model-b"):
                events.append("enter:model-b")

        task_a = asyncio.create_task(run_model_a())
        task_b = asyncio.create_task(run_model_b())

        await gate.wait()
        await asyncio.sleep(0)
        assert events == ["ensure:model-a", "enter:model-a"]

        release.set()
        await asyncio.gather(task_a, task_b)

        assert events == [
            "ensure:model-a",
            "enter:model-a",
            "exit:model-a",
            "ensure:model-b",
            "enter:model-b",
        ]
