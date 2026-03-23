from __future__ import annotations

import pytest

from mlx_batch_server.batch import coordinator as batch_coordinator_module
from mlx_batch_server.chat.mlx import wrapper_cache as wrapper_cache_module
from mlx_batch_server.chat.openai.models import models as models_module
from mlx_batch_server.chat.openai.models.schema import ModelUnloadRequest
from mlx_batch_server.responses import adapter as responses_adapter_module


class TestLoadedModelsRuntime:
    @pytest.mark.asyncio
    async def test_list_loaded_models_reports_batch_coordinators(self, monkeypatch):
        """Loaded-models endpoint should surface batch coordinator residency."""
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_models",
            lambda: ["model-a"],
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_cache_info",
            lambda: {"cache_size": 1},
        )
        monkeypatch.setattr(
            batch_coordinator_module,
            "get_loaded_batch_models",
            lambda: ["model-a"],
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_vlm_models",
            lambda: ["model-b"],
        )

        payload = await models_module.list_loaded_models()

        assert [item["id"] for item in payload["data"]] == ["model-a", "model-b"]
        assert payload["data"][0]["backends"] == ["wrapper"]
        assert payload["data"][0]["runtime"]["active_lanes"] == ["text"]
        assert payload["data"][0]["runtime"]["text"]["batch_resident"] is True
        assert payload["data"][1]["backends"] == ["vlm"]
        assert payload["data"][1]["runtime"]["active_lanes"] == ["multimodal"]
        assert payload["coordinators"] == {"llm_batch": ["model-a"]}
        assert payload["caches"] == {"wrapper": ["model-a"], "vlm": ["model-b"]}
        assert payload["cache_info"] == {"cache_size": 1}
        assert payload["runtime_contract"]["text"]["tool_capable"] is True
        assert payload["runtime_contract"]["multimodal"]["execution"] == "single_flight"

    @pytest.mark.asyncio
    async def test_list_loaded_models_merges_wrapper_and_vlm_residency(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_models",
            lambda: ["model-a"],
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_cache_info",
            lambda: {"cache_size": 1},
        )
        monkeypatch.setattr(
            batch_coordinator_module,
            "get_loaded_batch_models",
            lambda: [],
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_vlm_models",
            lambda: ["model-a"],
        )

        payload = await models_module.list_loaded_models()

        assert len(payload["data"]) == 1
        runtime_entry = payload["data"][0]
        assert runtime_entry["id"] == "model-a"
        assert runtime_entry["backends"] == ["wrapper", "vlm"]
        assert runtime_entry["runtime"]["product_residency"] == "single_model"
        assert runtime_entry["runtime"]["active_lanes"] == ["text", "multimodal"]
        assert runtime_entry["runtime"]["text"]["resident"] is True
        assert runtime_entry["runtime"]["multimodal"]["resident"] is True

    @pytest.mark.asyncio
    async def test_health_check_keeps_one_model_story_across_text_and_multimodal(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_models",
            lambda: ["model-a"],
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_cache_info",
            lambda: {"cache_size": 1, "max_size": 2, "ttl_seconds": 600},
        )
        monkeypatch.setattr(
            batch_coordinator_module,
            "get_loaded_batch_models",
            lambda: ["model-a"],
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_vlm_models",
            lambda: ["model-a"],
        )

        payload = await models_module.health_check()

        assert payload["loaded_models_count"] == 1
        assert payload["loaded_models"] == ["model-a"]
        assert payload["loaded_models_by_backend"] == {
            "wrapper": ["model-a"],
            "vlm": ["model-a"],
            "batch": ["model-a"],
        }
        assert payload["runtime_contract"]["multimodal"]["execution"] == "single_flight"
        runtime_entry = payload["loaded_models_runtime"][0]
        assert runtime_entry["runtime"]["text"]["tool_capable"] is True
        assert runtime_entry["runtime"]["text"]["batch_resident"] is True
        assert runtime_entry["runtime"]["multimodal"]["resident"] is True


class TestUnloadRuntime:
    @pytest.mark.asyncio
    async def test_unload_llm_shuts_down_batch_coordinator_first(self, monkeypatch):
        """LLM unload should tear down the batch coordinator before cache unload."""
        state = {"shutdown_called": False}

        async def fake_shutdown(model_id: str) -> int:
            assert model_id == "model-a"
            state["shutdown_called"] = True
            return 1

        class FakeModelsService:
            def unload_model(self, model_id):
                assert state["shutdown_called"] is True
                assert model_id == "model-a"
                return {
                    "status": "unloaded",
                    "message": "ok",
                    "unloaded_models": ["model-a"],
                    "cache_info": {"cache_size": 0},
                }

        monkeypatch.setattr(
            batch_coordinator_module,
            "shutdown_batch_coordinator",
            fake_shutdown,
        )
        monkeypatch.setattr(
            responses_adapter_module,
            "unload_vlm_model",
            lambda model_id=None: [],
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_models",
            lambda: [],
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_cache_info",
            lambda: {"cache_size": 0},
        )
        monkeypatch.setattr(
            batch_coordinator_module,
            "get_loaded_batch_models",
            lambda: [],
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_vlm_models",
            lambda: [],
        )

        def fake_get_models_service() -> FakeModelsService:
            return FakeModelsService()

        monkeypatch.setattr(
            models_module,
            "get_models_service",
            fake_get_models_service,
        )

        response = await models_module.unload_model(
            ModelUnloadRequest(model="model-a", task="llm")
        )

        assert response.status == "unloaded"
        assert response.unloaded_models == ["model-a"]
        assert response.cache_info["loaded_models_count"] == 0

    @pytest.mark.asyncio
    async def test_unload_llm_can_succeed_from_vlm_cache_only(self, monkeypatch):
        """LLM unload should clear hidden VLM residency even without wrapper cache hit."""
        state = {"shutdown_called": False}

        async def fake_shutdown(model_id: str) -> int:
            assert model_id == "model-a"
            state["shutdown_called"] = True
            return 0

        class FakeModelsService:
            def unload_model(self, model_id):
                assert state["shutdown_called"] is True
                assert model_id == "model-a"
                return {
                    "status": "not_found",
                    "message": "Model model-a was not loaded",
                    "unloaded_models": [],
                    "cache_info": {"cache_size": 0},
                }

        monkeypatch.setattr(
            batch_coordinator_module,
            "shutdown_batch_coordinator",
            fake_shutdown,
        )
        monkeypatch.setattr(
            responses_adapter_module,
            "unload_vlm_model",
            lambda model_id=None: ["model-a"],
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_models",
            lambda: [],
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_cache_info",
            lambda: {"cache_size": 0},
        )
        monkeypatch.setattr(
            batch_coordinator_module,
            "get_loaded_batch_models",
            lambda: [],
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_vlm_models",
            lambda: [],
        )
        monkeypatch.setattr(
            models_module,
            "get_models_service",
            FakeModelsService,
        )

        response = await models_module.unload_model(
            ModelUnloadRequest(model="model-a", task="llm")
        )

        assert response.status == "unloaded"
        assert response.message == "Model model-a unloaded successfully"
        assert response.unloaded_models == ["model-a"]
        assert response.cache_info["loaded_models_by_backend"] == {
            "wrapper": [],
            "vlm": [],
            "batch": [],
        }

    @pytest.mark.asyncio
    async def test_unload_all_llm_models_shuts_down_all_coordinators(
        self,
        monkeypatch,
    ):
        """Clearing all LLM models should also clear all batch coordinators."""
        state = {"shutdown_all_called": False}

        async def fake_shutdown_all() -> None:
            state["shutdown_all_called"] = True

        class FakeModelsService:
            def unload_model(self, model_id):
                assert state["shutdown_all_called"] is True
                assert model_id is None
                return {
                    "status": "cleared",
                    "message": "ok",
                    "unloaded_models": [],
                    "cache_info": {"cache_size": 0},
                }

        monkeypatch.setattr(
            batch_coordinator_module,
            "shutdown_all_coordinators",
            fake_shutdown_all,
        )
        monkeypatch.setattr(
            responses_adapter_module,
            "unload_vlm_model",
            lambda model_id=None: ["model-b"],
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_models",
            lambda: [],
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_cache_info",
            lambda: {"cache_size": 0},
        )
        monkeypatch.setattr(
            batch_coordinator_module,
            "get_loaded_batch_models",
            lambda: [],
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_vlm_models",
            lambda: [],
        )

        def fake_get_models_service() -> FakeModelsService:
            return FakeModelsService()

        monkeypatch.setattr(
            models_module,
            "get_models_service",
            fake_get_models_service,
        )

        response = await models_module.unload_model(ModelUnloadRequest(task="llm"))

        assert response.status == "cleared"
        assert response.unloaded_models == ["model-b"]
        assert response.cache_info["runtime_contract"]["multimodal"]["execution"] == (
            "single_flight"
        )
