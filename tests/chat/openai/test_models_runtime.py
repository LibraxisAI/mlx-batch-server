from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from mlx_batch_server import aux_runtime as aux_runtime_module
from mlx_batch_server.batch import coordinator as batch_coordinator_module
from mlx_batch_server.chat.mlx import runtime_aliases as runtime_aliases_module
from mlx_batch_server.chat.mlx import runtime_attachments as runtime_attachments_module
from mlx_batch_server.chat.mlx import wrapper_cache as wrapper_cache_module
from mlx_batch_server.chat.openai.models import models as models_module
from mlx_batch_server.chat.openai.models import models_service as models_service_module
from mlx_batch_server.chat.openai.models.schema import (
    ModelLoadRequest,
    ModelUnloadRequest,
)
from mlx_batch_server.embeddings import embeddings_service as embeddings_service_module
from mlx_batch_server.embeddings import visual_router as visual_router_module
from mlx_batch_server.images import image_runtime as image_runtime_module
from mlx_batch_server.stt import whisper_model as whisper_model_module
from mlx_batch_server.tts import tts_service as tts_service_module
from mlx_batch_server.vision import vlm_batch as vlm_batch_module
from mlx_batch_server.vision import vlm_cache as vlm_cache_module


@pytest.fixture(autouse=True)
def clear_runtime_aliases():
    runtime_aliases_module.clear_runtime_aliases()
    runtime_attachments_module.clear_runtime_surface_attachments()
    yield
    runtime_aliases_module.clear_runtime_aliases()
    runtime_attachments_module.clear_runtime_surface_attachments()


class TestLoadedModelsRuntime:
    def test_process_residency_includes_every_heavy_backend(self, monkeypatch):
        class FakeEmbeddingsService:
            @staticmethod
            def get_loaded_native_models():
                return ["embed-model"]

        monkeypatch.setattr(
            embeddings_service_module,
            "get_embeddings_service",
            FakeEmbeddingsService,
        )
        monkeypatch.setattr(
            image_runtime_module,
            "get_image_runtime_snapshot",
            lambda: {
                "running": True,
                "active_operations": 0,
                "idle_ttl_seconds": 600,
                "worker_pid": 42,
                "resident_models": ["image-model"],
            },
        )
        monkeypatch.setattr(
            whisper_model_module,
            "get_loaded_whisper_models",
            lambda: ["stt-model"],
        )
        monkeypatch.setattr(
            tts_service_module.TTSService,
            "get_loaded_models",
            lambda: ["tts-model"],
        )
        monkeypatch.setattr(
            aux_runtime_module,
            "get_aux_runtime_snapshot",
            lambda: {
                "resident_by_lane": {
                    "embeddings": ["embed-model"],
                    "tts": ["tts-model"],
                    "stt": ["stt-model"],
                },
                "active_by_lane": {},
                "resident_count": 3,
                "active_operations": 0,
                "idle_ttl_seconds": 600,
            },
        )
        llm_runtime = {
            "caches": {"wrapper": ["llm-model"]},
            "coordinators": {"llm_batch": ["llm-model"], "vlm_batch": []},
        }

        residency = models_module._snapshot_process_residency(llm_runtime)

        assert residency["loaded_models"] == [
            "embed-model",
            "image-model",
            "llm-model",
            "stt-model",
            "tts-model",
        ]
        assert residency["loaded_models_count"] == 5
        assert residency["loaded_models_by_backend"] == {
            "wrapper": ["llm-model"],
            "batch": ["llm-model"],
            "vlm_batch": [],
            "image": ["image-model"],
            "embeddings": ["embed-model"],
            "tts": ["tts-model"],
            "stt": ["stt-model"],
        }

    @pytest.mark.asyncio
    async def test_list_loaded_models_reports_batch_coordinators(self, monkeypatch):
        """Loaded-models endpoint should surface batch coordinator residency."""
        monkeypatch.setattr(
            models_module,
            "_get_runtime_memory_snapshot",
            lambda: {
                "process_rss_gb": 12.34,
                "rss_gb": 12.34,
                "mlx_active_memory_gb": 5.67,
                "mlx_active_gb": 5.67,
                "mlx_cache_memory_gb": 1.23,
                "mlx_cache_gb": 1.23,
                "pid": 4321,
            },
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_models",
            lambda: ["model-a", "model-b"],
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
        monkeypatch.setattr(
            vlm_batch_module,
            "get_loaded_vlm_batch_models",
            lambda: ["model-b"],
        )

        payload = await models_module.list_loaded_models()

        assert [item["id"] for item in payload["data"]] == ["model-a", "model-b"]
        assert payload["data"][0]["backends"] == ["wrapper"]
        assert payload["data"][0]["runtime"]["active_lanes"] == ["text"]
        assert payload["data"][0]["runtime"]["text"]["batch_resident"] is True
        assert payload["data"][1]["backends"] == ["wrapper"]
        assert payload["data"][1]["runtime"]["active_lanes"] == [
            "text",
            "multimodal",
        ]
        assert payload["data"][1]["runtime"]["text"]["batch_resident"] is False
        assert payload["data"][1]["runtime"]["multimodal"]["batch_resident"] is True
        assert payload["coordinators"] == {
            "llm_batch": ["model-a"],
            "vlm_batch": ["model-b"],
        }
        assert payload["caches"] == {"wrapper": ["model-a", "model-b"]}
        assert payload["loaded_models"] == ["model-a", "model-b"]
        assert payload["loaded_models_count"] == 2
        assert payload["loaded_models_by_backend"]["wrapper"] == [
            "model-a",
            "model-b",
        ]
        assert payload["cache_info"] == {"cache_size": 1, "runtime_keys": []}
        assert payload["runtime_contract"]["text"]["tool_capable"] is True
        assert payload["runtime_contract"]["multimodal"]["execution"] == "single_flight"
        assert payload["runtime_contract"]["multimodal"]["batch_capable"] is True
        assert payload["runtime"] == {
            "process_rss_gb": 12.34,
            "rss_gb": 12.34,
            "mlx_active_memory_gb": 5.67,
            "mlx_active_gb": 5.67,
            "mlx_cache_memory_gb": 1.23,
            "mlx_cache_gb": 1.23,
            "pid": 4321,
        }

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
        assert runtime_entry["backends"] == ["wrapper"]
        assert runtime_entry["runtime"]["product_residency"] == "single_model"
        assert runtime_entry["runtime"]["active_lanes"] == ["text", "multimodal"]
        assert runtime_entry["runtime"]["text"]["resident"] is True
        assert runtime_entry["runtime"]["multimodal"]["resident"] is True

    @pytest.mark.asyncio
    async def test_list_loaded_models_collapses_case_mismatch_across_backends(
        self, monkeypatch
    ):
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_models",
            lambda: ["libraxisai/gpt-oss-120b-mlx-mxfp4"],
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_cache_info",
            lambda: {"cache_size": 1},
        )
        monkeypatch.setattr(
            batch_coordinator_module,
            "get_loaded_batch_models",
            lambda: ["LibraxisAI/GPT-OSS-120B-mlx-mxfp4"],
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_vlm_models",
            lambda: ["libraxisai/gpt-oss-120b-mlx-mxfp4"],
        )

        payload = await models_module.list_loaded_models()

        assert len(payload["data"]) == 1
        entry = payload["data"][0]
        assert entry["id"] == "libraxisai/gpt-oss-120b-mlx-mxfp4"
        assert entry["backends"] == ["wrapper"]
        assert entry["runtime"]["text"]["batch_resident"] is True

    @pytest.mark.asyncio
    async def test_list_loaded_models_exposes_structured_runtime_keys(
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
            lambda: {"cache_size": 2},
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_runtime_keys",
            lambda: [
                wrapper_cache_module.WrapperCacheKey("model-a", None, None),
                wrapper_cache_module.WrapperCacheKey(
                    "model-a",
                    "/adapter/frontier",
                    "draft-a",
                ),
            ],
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

        payload = await models_module.list_loaded_models()

        assert payload["runtime_keys"] == [
            {
                "model_id": "model-a",
                "adapter_path": None,
                "draft_model_id": None,
            },
            {
                "model_id": "model-a",
                "adapter_path": "/adapter/frontier",
                "draft_model_id": "draft-a",
            },
        ]
        assert payload["cache_info"]["runtime_keys"] == payload["runtime_keys"]

    @pytest.mark.asyncio
    async def test_list_loaded_models_exposes_surface_ownership_for_shared_vlm(
        self,
        monkeypatch,
    ):
        runtime_attachments_module.attach_runtime_surface("model-vlm", "llm")
        runtime_attachments_module.attach_runtime_surface("model-vlm", "visual")

        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_models",
            lambda: ["model-vlm"],
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
            lambda: ["model-vlm"],
        )
        monkeypatch.setattr(
            vlm_batch_module,
            "get_loaded_vlm_batch_models",
            lambda: ["model-vlm"],
        )

        payload = await models_module.list_loaded_models()

        assert payload["surface_attachments"] == {"model-vlm": ["llm", "visual"]}
        assert payload["data"][0]["attached_tasks"] == ["llm", "visual"]
        assert payload["data"][0]["runtime"]["active_lanes"] == [
            "text",
            "multimodal",
        ]
        assert payload["data"][0]["runtime"]["multimodal"]["batch_resident"] is True

    @pytest.mark.asyncio
    async def test_health_check_keeps_one_model_story_across_text_and_multimodal(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            models_module,
            "_get_runtime_memory_snapshot",
            lambda: {
                "process_rss_gb": 10.5,
                "rss_gb": 10.5,
                "mlx_active_memory_gb": 4.25,
                "mlx_active_gb": 4.25,
                "mlx_cache_memory_gb": 0.75,
                "mlx_cache_gb": 0.75,
                "pid": 2468,
            },
        )
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
        monkeypatch.setattr(
            vlm_batch_module,
            "get_loaded_vlm_batch_models",
            lambda: [],
        )

        payload = await models_module.health_check()

        assert payload["loaded_models_count"] == 1
        assert payload["loaded_models"] == ["model-a"]
        assert payload["loaded_models_by_backend"] == {
            "wrapper": ["model-a"],
            "batch": ["model-a"],
            "vlm_batch": [],
            "image": [],
            "embeddings": [],
            "tts": [],
            "stt": [],
        }
        assert payload["runtime_contract"]["multimodal"]["execution"] == "single_flight"
        assert payload["runtime_contract"]["multimodal"]["batch_capable"] is True
        runtime_entry = payload["loaded_models_runtime"][0]
        assert runtime_entry["runtime"]["text"]["tool_capable"] is True
        assert runtime_entry["runtime"]["text"]["batch_resident"] is True
        assert runtime_entry["runtime"]["multimodal"]["resident"] is True
        assert payload["memory"] == {
            "process_rss_gb": 10.5,
            "rss_gb": 10.5,
            "mlx_active_memory_gb": 4.25,
            "mlx_active_gb": 4.25,
            "mlx_cache_memory_gb": 0.75,
            "mlx_cache_gb": 0.75,
            "pid": 2468,
        }


class TestUnloadRuntime:
    @pytest.mark.asyncio
    async def test_unload_llm_shuts_down_batch_coordinator_first(self, monkeypatch):
        """LLM unload should tear down the batch coordinator before cache unload."""
        state = {"shutdown_called": False}

        async def fake_shutdown(
            model_id: str, *, adapter_path: str | None = None
        ) -> int:
            assert model_id == "model-a"
            assert adapter_path is None
            state["shutdown_called"] = True
            return 1

        class FakeModelsService:
            def unload_model(
                self,
                model_id,
                *,
                adapter_path=None,
                draft_model_id=None,
                release_runtime=True,
            ):
                assert state["shutdown_called"] is True
                assert model_id == "model-a"
                assert adapter_path is None
                assert draft_model_id is None
                assert release_runtime is True
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
            vlm_cache_module,
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
    async def test_unload_llm_detaches_when_visual_surface_still_holds_runtime(
        self,
        monkeypatch,
    ):
        state = {"shutdown_called": False, "vlm_shutdown_called": False}
        runtime_attachments_module.attach_runtime_surface("model-a", "llm")
        runtime_attachments_module.attach_runtime_surface("model-a", "visual")

        async def fake_shutdown(
            model_id: str, *, adapter_path: str | None = None
        ) -> int:
            assert model_id == "model-a"
            assert adapter_path is None
            state["shutdown_called"] = True
            return 0

        async def fake_shutdown_vlm(model_id: str) -> int:
            state["vlm_shutdown_called"] = True
            return 0

        class FakeModelsService:
            def unload_model(
                self,
                model_id,
                *,
                adapter_path=None,
                draft_model_id=None,
                release_runtime=True,
            ):
                assert state["shutdown_called"] is True
                assert model_id == "model-a"
                assert adapter_path is None
                assert draft_model_id is None
                assert release_runtime is False
                return {
                    "status": "detached",
                    "message": "Model model-a detached while shared runtime stayed hot",
                    "unloaded_models": ["model-a"],
                    "cache_info": {"cache_size": 0},
                }

        monkeypatch.setattr(
            batch_coordinator_module,
            "shutdown_batch_coordinator",
            fake_shutdown,
        )
        monkeypatch.setattr(
            vlm_batch_module,
            "shutdown_vlm_coordinator",
            fake_shutdown_vlm,
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_models",
            lambda: ["model-a"],
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
            vlm_batch_module,
            "get_loaded_vlm_batch_models",
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

        assert response.status == "detached"
        assert "retained by visual" in response.message
        assert response.unloaded_models == ["model-a"]
        assert response.cache_info["loaded_models_by_backend"] == {
            "wrapper": ["model-a"],
            "batch": [],
            "vlm_batch": [],
        }
        assert state["vlm_shutdown_called"] is False
        assert runtime_attachments_module.get_runtime_surface_attachments(
            "model-a"
        ) == ["visual"]

    @pytest.mark.asyncio
    async def test_unload_llm_shuts_down_vlm_batch_when_only_embeddings_surface_remains(
        self,
        monkeypatch,
    ):
        state = {"shutdown_called": False, "vlm_shutdown_called": False}
        runtime_attachments_module.attach_runtime_surface("model-a", "llm")
        runtime_attachments_module.attach_runtime_surface("model-a", "embeddings")

        async def fake_shutdown(
            model_id: str, *, adapter_path: str | None = None
        ) -> int:
            assert model_id == "model-a"
            assert adapter_path is None
            state["shutdown_called"] = True
            return 0

        async def fake_shutdown_vlm(model_id: str) -> int:
            assert model_id == "model-a"
            state["vlm_shutdown_called"] = True
            return 1

        class FakeModelsService:
            def unload_model(
                self,
                model_id,
                *,
                adapter_path=None,
                draft_model_id=None,
                release_runtime=True,
            ):
                assert state["shutdown_called"] is True
                assert model_id == "model-a"
                assert adapter_path is None
                assert draft_model_id is None
                assert release_runtime is False
                return {
                    "status": "detached",
                    "message": "Model model-a detached while shared runtime stayed hot",
                    "unloaded_models": ["model-a"],
                    "cache_info": {"cache_size": 0},
                }

        monkeypatch.setattr(
            batch_coordinator_module,
            "shutdown_batch_coordinator",
            fake_shutdown,
        )
        monkeypatch.setattr(
            vlm_batch_module,
            "shutdown_vlm_coordinator",
            fake_shutdown_vlm,
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_models",
            lambda: ["model-a"],
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
            vlm_batch_module,
            "get_loaded_vlm_batch_models",
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

        assert response.status == "detached"
        assert "retained by embeddings" in response.message
        assert state["vlm_shutdown_called"] is True
        assert runtime_attachments_module.get_runtime_surface_attachments(
            "model-a"
        ) == ["embeddings"]

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
            def unload_model(
                self,
                model_id,
                *,
                adapter_path=None,
                draft_model_id=None,
                release_runtime=True,
            ):
                assert state["shutdown_all_called"] is True
                assert model_id == "model-b"
                assert adapter_path is None
                assert draft_model_id is None
                assert release_runtime is True
                return {
                    "status": "unloaded",
                    "message": "ok",
                    "unloaded_models": ["model-b"],
                    "cache_info": {"cache_size": 0},
                }

        runtime_attachments_module.attach_runtime_surface("model-b", "llm")
        monkeypatch.setattr(
            batch_coordinator_module,
            "shutdown_all_coordinators",
            fake_shutdown_all,
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
        assert (
            runtime_attachments_module.get_runtime_surface_attachments("model-b") == []
        )
        assert response.cache_info["runtime_contract"]["multimodal"]["execution"] == (
            "single_flight"
        )

    @pytest.mark.asyncio
    async def test_unload_all_llm_falls_back_when_surface_registry_is_empty(
        self,
        monkeypatch,
    ):
        """Legacy llm clear should still work before attachment bookkeeping exists."""
        state = {"shutdown_all_called": False}

        async def fake_shutdown_all() -> None:
            state["shutdown_all_called"] = True

        class FakeModelsService:
            def unload_model(
                self,
                model_id,
                *,
                adapter_path=None,
                draft_model_id=None,
                release_runtime=True,
            ):
                assert state["shutdown_all_called"] is True
                assert model_id is None
                assert adapter_path is None
                assert draft_model_id is None
                assert release_runtime is True
                return {
                    "status": "cleared",
                    "message": "ok",
                    "unloaded_models": ["legacy-llm"],
                    "cache_info": {"cache_size": 0},
                }

        monkeypatch.setattr(
            batch_coordinator_module,
            "shutdown_all_coordinators",
            fake_shutdown_all,
        )
        monkeypatch.setattr(
            vlm_cache_module,
            "unload_vlm_model",
            lambda model_id=None: ["legacy-vlm"] if model_id is None else [],
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

        response = await models_module.unload_model(ModelUnloadRequest(task="llm"))

        assert response.status == "cleared"
        assert response.unloaded_models == ["legacy-llm", "legacy-vlm"]
        assert "legacy runtime state had no surface attachments" in response.message
        assert response.cache_info["loaded_models_count"] == 0

    @pytest.mark.asyncio
    async def test_unload_all_llm_preserves_runtime_for_visual_surfaces(
        self,
        monkeypatch,
    ):
        state = {"shutdown_all_called": False}
        calls: list[tuple[str, bool]] = []

        async def fake_shutdown_all() -> None:
            state["shutdown_all_called"] = True

        class FakeModelsService:
            def unload_model(
                self,
                model_id,
                *,
                adapter_path=None,
                draft_model_id=None,
                release_runtime=True,
            ):
                assert state["shutdown_all_called"] is True
                assert adapter_path is None
                assert draft_model_id is None
                calls.append((model_id, release_runtime))
                return {
                    "status": "detached",
                    "message": "ok",
                    "unloaded_models": [model_id],
                    "cache_info": {"cache_size": 1},
                }

        runtime_attachments_module.attach_runtime_surface("model-b", "llm")
        runtime_attachments_module.attach_runtime_surface("model-b", "visual")
        monkeypatch.setattr(
            batch_coordinator_module,
            "shutdown_all_coordinators",
            fake_shutdown_all,
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_models",
            lambda: ["model-b"],
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
            lambda: ["model-b"],
        )
        monkeypatch.setattr(
            models_module,
            "get_models_service",
            FakeModelsService,
        )

        response = await models_module.unload_model(ModelUnloadRequest(task="llm"))

        assert response.status == "cleared"
        assert response.unloaded_models == ["model-b"]
        assert calls == [("model-b", False)]
        assert runtime_attachments_module.get_runtime_surface_attachments(
            "model-b"
        ) == ["visual"]

    @pytest.mark.asyncio
    async def test_unload_all_llm_shuts_down_vlm_batch_when_only_embeddings_surface_remains(
        self,
        monkeypatch,
    ):
        state = {"shutdown_all_called": False, "vlm_shutdown_models": []}
        calls: list[tuple[str, bool]] = []

        async def fake_shutdown_all() -> None:
            state["shutdown_all_called"] = True

        async def fake_shutdown_vlm(model_id: str) -> int:
            state["vlm_shutdown_models"].append(model_id)
            return 1

        class FakeModelsService:
            def unload_model(
                self,
                model_id,
                *,
                adapter_path=None,
                draft_model_id=None,
                release_runtime=True,
            ):
                assert state["shutdown_all_called"] is True
                assert adapter_path is None
                assert draft_model_id is None
                calls.append((model_id, release_runtime))
                return {
                    "status": "detached",
                    "message": "ok",
                    "unloaded_models": [model_id],
                    "cache_info": {"cache_size": 1},
                }

        runtime_attachments_module.attach_runtime_surface("model-b", "llm")
        runtime_attachments_module.attach_runtime_surface("model-b", "embeddings")
        monkeypatch.setattr(
            batch_coordinator_module,
            "shutdown_all_coordinators",
            fake_shutdown_all,
        )
        monkeypatch.setattr(
            vlm_batch_module,
            "shutdown_vlm_coordinator",
            fake_shutdown_vlm,
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_models",
            lambda: ["model-b"],
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
            lambda: ["model-b"],
        )
        monkeypatch.setattr(
            vlm_batch_module,
            "get_loaded_vlm_batch_models",
            lambda: [],
        )
        monkeypatch.setattr(
            models_module,
            "get_models_service",
            FakeModelsService,
        )

        response = await models_module.unload_model(ModelUnloadRequest(task="llm"))

        assert response.status == "cleared"
        assert response.unloaded_models == ["model-b"]
        assert calls == [("model-b", False)]
        assert state["vlm_shutdown_models"] == ["model-b"]
        assert runtime_attachments_module.get_runtime_surface_attachments(
            "model-b"
        ) == ["embeddings"]

    @pytest.mark.asyncio
    async def test_unload_llm_exact_runtime_target_preserves_sibling_variant(
        self,
        monkeypatch,
    ):
        calls: list[tuple[str, str | None, str | None, bool]] = []
        batch_shutdowns: list[tuple[str, str | None]] = []
        vlm_shutdowns: list[str] = []

        runtime_attachments_module.attach_runtime_surface(
            "model-a",
            "llm",
            adapter_path="/adapter-a",
        )
        runtime_attachments_module.attach_runtime_surface(
            "model-a",
            "llm",
            adapter_path="/adapter-b",
        )

        async def fake_shutdown(
            model_id: str, *, adapter_path: str | None = None
        ) -> int:
            batch_shutdowns.append((model_id, adapter_path))
            return 1

        async def fake_shutdown_vlm(model_id: str) -> int:
            vlm_shutdowns.append(model_id)
            return 1

        class FakeModelsService:
            def unload_model(
                self,
                model_id,
                *,
                adapter_path=None,
                draft_model_id=None,
                release_runtime=True,
            ):
                calls.append((model_id, adapter_path, draft_model_id, release_runtime))
                return {
                    "status": "unloaded",
                    "message": f"Model {model_id} unloaded successfully",
                    "unloaded_models": [model_id],
                    "cache_info": {"cache_size": 1},
                }

        monkeypatch.setattr(
            batch_coordinator_module,
            "shutdown_batch_coordinator",
            fake_shutdown,
        )
        monkeypatch.setattr(
            vlm_batch_module,
            "shutdown_vlm_coordinator",
            fake_shutdown_vlm,
        )
        monkeypatch.setattr(
            models_module,
            "get_models_service",
            FakeModelsService,
        )
        monkeypatch.setattr(
            models_module,
            "_build_llm_cache_info",
            lambda: {"loaded_models_count": 1},
        )

        response = await models_module.unload_model(
            ModelUnloadRequest(
                model="model-a",
                task="llm",
                adapter_path="/adapter-a",
            )
        )

        assert response.status == "unloaded"
        assert response.unloaded_models == ["model-a"]
        assert calls == [("model-a", "/adapter-a", None, True)]
        assert batch_shutdowns == [("model-a", "/adapter-a")]
        assert vlm_shutdowns == ["model-a"]
        assert (
            runtime_attachments_module.get_runtime_surface_attachments(
                "model-a",
                adapter_path="/adapter-a",
            )
            == []
        )
        assert runtime_attachments_module.get_runtime_surface_attachments(
            "model-a",
            adapter_path="/adapter-b",
        ) == ["llm"]


class TestLoadRuntime:
    @pytest.mark.asyncio
    async def test_native_control_loads_use_auxiliary_admission(self, monkeypatch):
        admissions: list[tuple[str, str]] = []

        @asynccontextmanager
        async def unused_runtime_session(*args, **kwargs):
            yield {"switched": False, "evicted_models": []}

        from contextlib import contextmanager

        @contextmanager
        def admission(lane: str, model_id: str):
            admissions.append((lane, model_id))
            yield

        class FakeEmbeddingsService:
            @staticmethod
            def uses_shared_vlm_runtime(model_id: str) -> bool:
                return False

            @staticmethod
            def canonicalize_model_id(model_id: str) -> str:
                return model_id

            @staticmethod
            def load_model(model_id: str) -> bool:
                return False

        monkeypatch.setattr(
            aux_runtime_module,
            "auxiliary_runtime_operation",
            admission,
        )
        monkeypatch.setattr(
            embeddings_service_module,
            "get_embeddings_service",
            FakeEmbeddingsService,
        )
        monkeypatch.setattr(
            whisper_model_module,
            "preload_whisper_model",
            lambda model_id: False,
        )
        monkeypatch.setattr(
            tts_service_module.TTSService,
            "preload_model",
            lambda model_id: False,
        )
        monkeypatch.setattr(
            models_module,
            "endpoint_runtime_session",
            unused_runtime_session,
        )

        for task, model_id in (
            ("embeddings", "embed-model"),
            ("stt", "stt-model"),
            ("tts", "tts-model"),
        ):
            response = await models_module.load_model(
                ModelLoadRequest(model=model_id, task=task),
                _auth={},
            )
            assert response.status == "loaded"

        assert admissions == [
            ("embeddings", "embed-model"),
            ("stt", "stt-model"),
            ("tts", "tts-model"),
        ]

    @pytest.mark.asyncio
    async def test_load_llm_hard_switches_before_loading(self, monkeypatch):
        state = {"switch_called": False}

        @asynccontextmanager
        async def fake_runtime_session(
            model_id, adapter_path=None, draft_model_id=None
        ):
            assert model_id == "model-new"
            assert adapter_path == "/adapter"
            assert draft_model_id == "draft-model"
            state["switch_called"] = True
            yield {
                "switched": True,
                "evicted_models": ["model-old"],
            }

        class FakeModelsService:
            def load_model(self, model_id, adapter_path=None, draft_model_id=None):
                assert state["switch_called"] is True
                assert model_id == "model-new"
                assert adapter_path == "/adapter"
                assert draft_model_id == "draft-model"
                return {
                    "id": model_id,
                    "status": "loaded",
                    "message": f"Model {model_id} loaded successfully",
                    "cache_info": {"cache_size": 1},
                }

        monkeypatch.setattr(
            models_module,
            "endpoint_runtime_session",
            fake_runtime_session,
        )
        monkeypatch.setattr(
            models_module,
            "get_models_service",
            FakeModelsService,
        )
        monkeypatch.setattr(
            models_module,
            "_build_llm_cache_info",
            lambda: {"loaded_models_count": 1},
        )

        response = await models_module.load_model(
            ModelLoadRequest(
                model="model-new",
                task="llm",
                adapter_path="/adapter",
                draft_model_id="draft-model",
            )
        )

        assert response.status == "loaded"
        assert "after evicting 1 prior model(s): model-old" in response.message
        assert response.cache_info["loaded_models_count"] == 1

    @pytest.mark.asyncio
    async def test_load_llm_registers_alias_and_loads_canonical_runtime(
        self, monkeypatch
    ):
        state = {"switch_called": False, "loaded": False}

        @asynccontextmanager
        async def fake_runtime_session(
            model_id, adapter_path=None, draft_model_id=None
        ):
            assert model_id == "libraxisai/gpt-oss-120b-mlx-mxfp4"
            assert adapter_path is None
            assert draft_model_id is None
            state["switch_called"] = True
            yield {
                "switched": False,
                "evicted_models": [],
            }

        class FakeModelsService:
            def load_model(self, model_id, adapter_path=None, draft_model_id=None):
                assert state["switch_called"] is True
                assert model_id == "libraxisai/gpt-oss-120b-mlx-mxfp4"
                state["loaded"] = True
                return {
                    "id": model_id,
                    "status": "already_loaded",
                    "message": f"Model {model_id} was already loaded",
                    "cache_info": {"cache_size": 1},
                }

        monkeypatch.setattr(
            models_module,
            "endpoint_runtime_session",
            fake_runtime_session,
        )
        monkeypatch.setattr(
            models_module,
            "get_models_service",
            FakeModelsService,
        )
        monkeypatch.setattr(
            models_module,
            "_build_llm_cache_info",
            lambda: {"loaded_models_count": 1},
        )

        response = await models_module.load_model(
            ModelLoadRequest(
                model="libraxisai/gpt-oss-120b-mlx-mxfp4",
                task="llm",
                alias="qwen3.5-vl-crack",
            )
        )

        assert state["loaded"] is True
        assert response.id == "libraxisai/gpt-oss-120b-mlx-mxfp4"
        assert (
            runtime_aliases_module.resolve_runtime_model_id("qwen3.5-vl-crack")
            == "libraxisai/gpt-oss-120b-mlx-mxfp4"
        )
        assert (
            "runtime alias registered: qwen3.5-vl-crack -> "
            "libraxisai/gpt-oss-120b-mlx-mxfp4" in response.message
        )
        assert response.cache_info["loaded_models_count"] == 1

    @pytest.mark.asyncio
    async def test_load_llm_registers_alias_with_full_runtime_target(self, monkeypatch):
        state = {"switch_called": False, "loaded": False}
        expanded_adapter_path = str(Path("~/adapters/frontier-lora").expanduser())

        @asynccontextmanager
        async def fake_runtime_session(
            model_id, adapter_path=None, draft_model_id=None
        ):
            assert model_id == "libraxisai/qwen3-vl-30b"
            assert adapter_path == expanded_adapter_path
            assert draft_model_id == "mlx-community/qwen3-1.7b-4bit"
            state["switch_called"] = True
            yield {
                "switched": False,
                "evicted_models": [],
            }

        class FakeModelsService:
            def load_model(self, model_id, adapter_path=None, draft_model_id=None):
                assert state["switch_called"] is True
                assert model_id == "libraxisai/qwen3-vl-30b"
                assert adapter_path == expanded_adapter_path
                assert draft_model_id == "mlx-community/qwen3-1.7b-4bit"
                state["loaded"] = True
                return {
                    "id": model_id,
                    "status": "loaded",
                    "message": f"Model {model_id} loaded successfully",
                    "cache_info": {"cache_size": 1},
                }

        monkeypatch.setattr(
            models_module,
            "endpoint_runtime_session",
            fake_runtime_session,
        )
        monkeypatch.setattr(
            models_module,
            "get_models_service",
            FakeModelsService,
        )
        monkeypatch.setattr(
            models_module,
            "_build_llm_cache_info",
            lambda: {"loaded_models_count": 1},
        )

        response = await models_module.load_model(
            ModelLoadRequest(
                model="LibraxisAI/Qwen3-VL-30B",
                task="llm",
                adapter_path="~/adapters/frontier-lora",
                draft_model_id="MLX-Community/Qwen3-1.7B-4bit",
                alias="frontier-vlm",
            )
        )

        target = runtime_aliases_module.resolve_runtime_target("frontier-vlm")

        assert state["loaded"] is True
        assert response.id == "libraxisai/qwen3-vl-30b"
        assert target.model_id == "libraxisai/qwen3-vl-30b"
        assert target.adapter_path == expanded_adapter_path
        assert target.draft_model_id == "mlx-community/qwen3-1.7b-4bit"

    def test_create_model_alias_endpoint_registers_without_loading(self):
        async def run_alias_registration():
            return await models_module.create_model_alias(
                models_module.ModelAliasRequest(
                    alias="operator-chat",
                    model="LibraxisAI/Qwen3-VL-30B",
                    adapter_path="~/adapters/frontier-lora",
                )
            )

        response = asyncio.run(run_alias_registration())

        target = runtime_aliases_module.resolve_runtime_target("operator-chat")

        assert response.alias == "operator-chat"
        assert response.model == "libraxisai/qwen3-vl-30b"
        assert target.model_id == "libraxisai/qwen3-vl-30b"
        assert target.adapter_path == str(Path("~/adapters/frontier-lora").expanduser())

    @pytest.mark.asyncio
    async def test_load_visual_uses_shared_runtime_session(self, monkeypatch):
        state = {"switch_called": False, "loaded_model": None}

        @asynccontextmanager
        async def fake_runtime_session(
            model_id, adapter_path=None, draft_model_id=None
        ):
            assert model_id == "libraxisai/qwen3-vl-30b"
            assert adapter_path is None
            assert draft_model_id is None
            state["switch_called"] = True
            yield {
                "switched": True,
                "evicted_models": ["model-old"],
            }

        def fake_get_visual_embedder(
            model_id,
            projection_path=None,
            processor_id=None,
            *,
            adapter_path=None,
            draft_model_id=None,
        ):
            assert state["switch_called"] is True
            assert projection_path is None
            assert processor_id is None
            assert adapter_path is None
            assert draft_model_id is None
            state["loaded_model"] = model_id
            return object()

        monkeypatch.setattr(
            models_module,
            "endpoint_runtime_session",
            fake_runtime_session,
        )
        monkeypatch.setattr(
            visual_router_module,
            "get_visual_embedder",
            fake_get_visual_embedder,
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_loaded_vlm_models",
            lambda: [],
        )
        monkeypatch.setattr(
            models_module,
            "_build_llm_cache_info",
            lambda: {"loaded_models_count": 1},
        )

        response = await models_module.load_model(
            ModelLoadRequest(model="LibraxisAI/Qwen3-VL-30B", task="visual")
        )

        assert response.id == "libraxisai/qwen3-vl-30b"
        assert response.task == "visual"
        assert response.status == "loaded"
        assert state["loaded_model"] == "libraxisai/qwen3-vl-30b"
        assert "after evicting 1 prior model(s): model-old" in response.message
        assert response.cache_info["loaded_models_count"] == 1

    @pytest.mark.asyncio
    async def test_load_embeddings_qwen3_vl_uses_shared_runtime_session(
        self, monkeypatch
    ):
        state = {"switch_called": False, "loaded_model": None}

        @asynccontextmanager
        async def fake_runtime_session(
            model_id, adapter_path=None, draft_model_id=None
        ):
            assert model_id == "libraxisai/qwen3-vl-30b"
            assert adapter_path is None
            assert draft_model_id is None
            state["switch_called"] = True
            yield {
                "switched": True,
                "evicted_models": ["model-old"],
            }

        class FakeEmbeddingsService:
            def uses_shared_vlm_runtime(self, model_id):
                return True

            def canonicalize_model_id(self, model_id):
                return "libraxisai/qwen3-vl-30b"

            def load_model(
                self,
                model_id,
                adapter_path=None,
                draft_model_id=None,
            ):
                assert state["switch_called"] is True
                assert adapter_path is None
                assert draft_model_id is None
                state["loaded_model"] = model_id
                return False

        monkeypatch.setattr(
            models_module,
            "endpoint_runtime_session",
            fake_runtime_session,
        )
        monkeypatch.setattr(
            embeddings_service_module,
            "get_embeddings_service",
            FakeEmbeddingsService,
        )
        monkeypatch.setattr(
            models_module,
            "_build_llm_cache_info",
            lambda: {"loaded_models_count": 1},
        )

        response = await models_module.load_model(
            ModelLoadRequest(model="LibraxisAI/Qwen3-VL-30B", task="embeddings")
        )

        assert response.id == "libraxisai/qwen3-vl-30b"
        assert response.task == "embeddings"
        assert response.status == "loaded"
        assert state["loaded_model"] == "libraxisai/qwen3-vl-30b"
        assert "after evicting 1 prior model(s): model-old" in response.message
        assert response.cache_info["loaded_models_count"] == 1

    @pytest.mark.asyncio
    async def test_unload_visual_returns_shared_runtime_cache_info(self, monkeypatch):
        monkeypatch.setattr(
            visual_router_module,
            "unload_visual_embedder",
            lambda model_id=None, adapter_path=None, draft_model_id=None, release_runtime=True: (
                ["model-vlm"] if model_id == "model-vlm" else []
            ),
        )
        monkeypatch.setattr(
            vlm_batch_module,
            "shutdown_vlm_coordinator",
            AsyncMock(return_value=1),
        )
        monkeypatch.setattr(
            models_module,
            "_build_llm_cache_info",
            lambda: {"loaded_models_count": 0},
        )

        response = await models_module.unload_model(
            ModelUnloadRequest(model="model-vlm", task="visual")
        )

        assert response.task == "visual"
        assert response.status == "unloaded"
        assert response.unloaded_models == ["model-vlm"]
        assert response.cache_info["loaded_models_count"] == 0

    @pytest.mark.asyncio
    async def test_unload_visual_detaches_surface_when_llm_still_holds_runtime(
        self,
        monkeypatch,
    ):
        runtime_attachments_module.attach_runtime_surface("model-vlm", "llm")
        runtime_attachments_module.attach_runtime_surface("model-vlm", "visual")

        release_runtime_flags: list[bool] = []
        monkeypatch.setattr(
            visual_router_module,
            "unload_visual_embedder",
            lambda model_id=None, adapter_path=None, draft_model_id=None, release_runtime=True: (
                release_runtime_flags.append(release_runtime) or ["model-vlm"]
            ),
        )
        monkeypatch.setattr(
            models_module,
            "_build_llm_cache_info",
            lambda: {
                "loaded_models_count": 1,
                "surface_attachments": {"model-vlm": ["llm"]},
            },
        )

        response = await models_module.unload_model(
            ModelUnloadRequest(model="model-vlm", task="visual")
        )

        assert response.task == "visual"
        assert response.status == "detached"
        assert response.unloaded_models == ["model-vlm"]
        assert "retained by llm" in response.message
        assert release_runtime_flags == [False]
        assert runtime_attachments_module.get_runtime_surface_attachments(
            "model-vlm"
        ) == ["llm"]

    @pytest.mark.asyncio
    async def test_unload_embeddings_qwen3_vl_returns_shared_runtime_cache_info(
        self, monkeypatch
    ):
        class FakeEmbeddingsService:
            def uses_shared_vlm_runtime(self, model_id):
                return True

            def canonicalize_model_id(self, model_id):
                return "libraxisai/qwen3-vl-30b"

            def unload_model(
                self,
                model_id,
                *,
                adapter_path=None,
                draft_model_id=None,
                release_runtime=True,
            ):
                assert adapter_path is None
                assert draft_model_id is None
                assert release_runtime is True
                return True

        monkeypatch.setattr(
            embeddings_service_module,
            "get_embeddings_service",
            FakeEmbeddingsService,
        )
        monkeypatch.setattr(
            models_module,
            "_build_llm_cache_info",
            lambda: {"loaded_models_count": 0},
        )

        response = await models_module.unload_model(
            ModelUnloadRequest(model="LibraxisAI/Qwen3-VL-30B", task="embeddings")
        )

        assert response.task == "embeddings"
        assert response.status == "unloaded"
        assert response.unloaded_models == ["libraxisai/qwen3-vl-30b"]
        assert response.cache_info["loaded_models_count"] == 0

    @pytest.mark.asyncio
    async def test_unload_embeddings_detaches_surface_when_llm_still_holds_runtime(
        self,
        monkeypatch,
    ):
        runtime_attachments_module.attach_runtime_surface(
            "libraxisai/qwen3-vl-30b", "llm"
        )
        runtime_attachments_module.attach_runtime_surface(
            "libraxisai/qwen3-vl-30b", "embeddings"
        )

        class FakeEmbeddingsService:
            def uses_shared_vlm_runtime(self, model_id):
                return True

            def canonicalize_model_id(self, model_id):
                return "libraxisai/qwen3-vl-30b"

            def unload_model(
                self,
                model_id,
                *,
                adapter_path=None,
                draft_model_id=None,
                release_runtime=True,
            ):
                assert adapter_path is None
                assert draft_model_id is None
                assert release_runtime is False
                return True

        monkeypatch.setattr(
            embeddings_service_module,
            "get_embeddings_service",
            FakeEmbeddingsService,
        )
        monkeypatch.setattr(
            models_module,
            "_build_llm_cache_info",
            lambda: {
                "loaded_models_count": 1,
                "surface_attachments": {"libraxisai/qwen3-vl-30b": ["llm"]},
            },
        )

        response = await models_module.unload_model(
            ModelUnloadRequest(model="LibraxisAI/Qwen3-VL-30B", task="embeddings")
        )

        assert response.task == "embeddings"
        assert response.status == "detached"
        assert response.unloaded_models == ["libraxisai/qwen3-vl-30b"]
        assert "retained by llm" in response.message
        assert runtime_attachments_module.get_runtime_surface_attachments(
            "libraxisai/qwen3-vl-30b"
        ) == ["llm"]

    @pytest.mark.asyncio
    async def test_unload_embeddings_shuts_down_vlm_batch_when_visual_surface_is_gone(
        self,
        monkeypatch,
    ):
        state = {"vlm_shutdown_called": False}
        runtime_attachments_module.attach_runtime_surface(
            "libraxisai/qwen3-vl-30b", "llm"
        )
        runtime_attachments_module.attach_runtime_surface(
            "libraxisai/qwen3-vl-30b", "embeddings"
        )

        class FakeEmbeddingsService:
            def uses_shared_vlm_runtime(self, model_id):
                return True

            def canonicalize_model_id(self, model_id):
                return "libraxisai/qwen3-vl-30b"

            def unload_model(
                self,
                model_id,
                *,
                adapter_path=None,
                draft_model_id=None,
                release_runtime=True,
            ):
                assert adapter_path is None
                assert draft_model_id is None
                assert release_runtime is False
                return True

        async def fake_shutdown_vlm(model_id: str) -> int:
            assert model_id == "libraxisai/qwen3-vl-30b"
            state["vlm_shutdown_called"] = True
            return 1

        monkeypatch.setattr(
            embeddings_service_module,
            "get_embeddings_service",
            FakeEmbeddingsService,
        )
        monkeypatch.setattr(
            vlm_batch_module,
            "shutdown_vlm_coordinator",
            fake_shutdown_vlm,
        )
        monkeypatch.setattr(
            models_module,
            "_build_llm_cache_info",
            lambda: {
                "loaded_models_count": 1,
                "surface_attachments": {"libraxisai/qwen3-vl-30b": ["llm"]},
            },
        )

        response = await models_module.unload_model(
            ModelUnloadRequest(model="LibraxisAI/Qwen3-VL-30B", task="embeddings")
        )

        assert response.task == "embeddings"
        assert response.status == "detached"
        assert state["vlm_shutdown_called"] is True
        assert runtime_attachments_module.get_runtime_surface_attachments(
            "libraxisai/qwen3-vl-30b"
        ) == ["llm"]

    @pytest.mark.asyncio
    async def test_unload_embeddings_fans_out_exact_runtime_keys_to_service(
        self,
        monkeypatch,
    ):
        calls: list[tuple[str, str | None, str | None, bool]] = []
        shutdowns: list[str] = []

        runtime_attachments_module.attach_runtime_surface(
            "libraxisai/qwen3-vl-30b",
            "embeddings",
            adapter_path="/adapter-a",
        )
        runtime_attachments_module.attach_runtime_surface(
            "libraxisai/qwen3-vl-30b",
            "embeddings",
            adapter_path="/adapter-b",
        )

        class FakeEmbeddingsService:
            def uses_shared_vlm_runtime(self, model_id):
                return True

            def canonicalize_model_id(self, model_id):
                return "libraxisai/qwen3-vl-30b"

            def unload_model(
                self,
                model_id,
                *,
                adapter_path=None,
                draft_model_id=None,
                release_runtime=True,
            ):
                calls.append((model_id, adapter_path, draft_model_id, release_runtime))
                return True

        async def fake_shutdown_vlm(model_id: str) -> int:
            shutdowns.append(model_id)
            return 1

        monkeypatch.setattr(
            embeddings_service_module,
            "get_embeddings_service",
            FakeEmbeddingsService,
        )
        monkeypatch.setattr(
            vlm_batch_module,
            "shutdown_vlm_coordinator",
            fake_shutdown_vlm,
        )
        monkeypatch.setattr(
            models_module,
            "_build_llm_cache_info",
            lambda: {"loaded_models_count": 0, "surface_attachments": {}},
        )

        response = await models_module.unload_model(
            ModelUnloadRequest(model="LibraxisAI/Qwen3-VL-30B", task="embeddings")
        )

        assert response.task == "embeddings"
        assert response.status == "unloaded"
        assert response.unloaded_models == ["libraxisai/qwen3-vl-30b"]
        assert calls == [
            ("libraxisai/qwen3-vl-30b", "/adapter-a", None, True),
            ("libraxisai/qwen3-vl-30b", "/adapter-b", None, True),
        ]
        assert shutdowns == ["libraxisai/qwen3-vl-30b"]
        assert (
            runtime_attachments_module.get_runtime_surface_attachments(
                "libraxisai/qwen3-vl-30b",
                adapter_path="/adapter-a",
            )
            == []
        )
        assert (
            runtime_attachments_module.get_runtime_surface_attachments(
                "libraxisai/qwen3-vl-30b",
                adapter_path="/adapter-b",
            )
            == []
        )

    @pytest.mark.asyncio
    async def test_unload_visual_exact_runtime_preserves_sibling_visual_variant(
        self,
        monkeypatch,
    ):
        calls: list[tuple[str, str | None, str | None, bool]] = []
        shutdowns: list[str] = []

        runtime_attachments_module.attach_runtime_surface(
            "libraxisai/qwen3-vl-30b",
            "visual",
            adapter_path="/adapter-a",
        )
        runtime_attachments_module.attach_runtime_surface(
            "libraxisai/qwen3-vl-30b",
            "visual",
            adapter_path="/adapter-b",
        )

        monkeypatch.setattr(
            visual_router_module,
            "unload_visual_embedder",
            lambda model_id, **kwargs: (
                calls.append(
                    (
                        model_id,
                        kwargs.get("adapter_path"),
                        kwargs.get("draft_model_id"),
                        kwargs.get("release_runtime"),
                    )
                )
                or [model_id]
            ),
        )

        async def fake_shutdown_vlm(model_id: str) -> int:
            shutdowns.append(model_id)
            return 1

        monkeypatch.setattr(
            vlm_batch_module,
            "shutdown_vlm_coordinator",
            fake_shutdown_vlm,
        )
        monkeypatch.setattr(
            models_module,
            "_build_llm_cache_info",
            lambda: {"loaded_models_count": 1},
        )

        response = await models_module.unload_model(
            ModelUnloadRequest(
                model="LibraxisAI/Qwen3-VL-30B",
                task="visual",
                adapter_path="/adapter-a",
            )
        )

        assert response.task == "visual"
        assert response.status == "unloaded"
        assert response.unloaded_models == ["libraxisai/qwen3-vl-30b"]
        assert calls == [("libraxisai/qwen3-vl-30b", "/adapter-a", None, True)]
        assert shutdowns == []
        assert (
            runtime_attachments_module.get_runtime_surface_attachments(
                "libraxisai/qwen3-vl-30b",
                adapter_path="/adapter-a",
            )
            == []
        )
        assert runtime_attachments_module.get_runtime_surface_attachments(
            "libraxisai/qwen3-vl-30b",
            adapter_path="/adapter-b",
        ) == ["visual"]

    @pytest.mark.asyncio
    async def test_unload_embeddings_exact_runtime_targets_one_service_variant(
        self,
        monkeypatch,
    ):
        calls: list[tuple[str, str | None, str | None, bool]] = []
        shutdowns: list[str] = []

        runtime_attachments_module.attach_runtime_surface(
            "libraxisai/qwen3-vl-30b",
            "embeddings",
            adapter_path="/adapter-a",
        )
        runtime_attachments_module.attach_runtime_surface(
            "libraxisai/qwen3-vl-30b",
            "embeddings",
            adapter_path="/adapter-b",
        )
        runtime_attachments_module.attach_runtime_surface(
            "libraxisai/qwen3-vl-30b",
            "visual",
            adapter_path="/adapter-b",
        )

        class FakeEmbeddingsService:
            def uses_shared_vlm_runtime(self, model_id):
                return True

            def canonicalize_model_id(self, model_id):
                return "libraxisai/qwen3-vl-30b"

            def unload_model(
                self,
                model_id,
                *,
                adapter_path=None,
                draft_model_id=None,
                release_runtime=True,
            ):
                calls.append((model_id, adapter_path, draft_model_id, release_runtime))
                return True

        async def fake_shutdown_vlm(model_id: str) -> int:
            shutdowns.append(model_id)
            return 1

        monkeypatch.setattr(
            embeddings_service_module,
            "get_embeddings_service",
            FakeEmbeddingsService,
        )
        monkeypatch.setattr(
            vlm_batch_module,
            "shutdown_vlm_coordinator",
            fake_shutdown_vlm,
        )
        monkeypatch.setattr(
            models_module,
            "_build_llm_cache_info",
            lambda: {"loaded_models_count": 1},
        )

        response = await models_module.unload_model(
            ModelUnloadRequest(
                model="LibraxisAI/Qwen3-VL-30B",
                task="embeddings",
                adapter_path="/adapter-a",
            )
        )

        assert response.task == "embeddings"
        assert response.status == "unloaded"
        assert response.unloaded_models == ["libraxisai/qwen3-vl-30b"]
        assert calls == [("libraxisai/qwen3-vl-30b", "/adapter-a", None, True)]
        assert shutdowns == []
        assert (
            runtime_attachments_module.get_runtime_surface_attachments(
                "libraxisai/qwen3-vl-30b",
                adapter_path="/adapter-a",
            )
            == []
        )
        assert runtime_attachments_module.get_runtime_surface_attachments(
            "libraxisai/qwen3-vl-30b",
            adapter_path="/adapter-b",
        ) == ["embeddings", "visual"]


class TestModelsServiceRuntimeKeys:
    def test_load_model_checks_exact_runtime_key_not_only_base_model(
        self,
        monkeypatch,
    ):
        monkeypatch.setattr(
            models_service_module.ModelsService,
            "_scan_models",
            lambda self: [],
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "is_model_loaded",
            lambda model_id: True,
        )
        seen_runtime_checks: list[tuple[str, str | None, str | None]] = []
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "is_runtime_loaded",
            lambda model_id, adapter_path=None, draft_model_id=None: (
                seen_runtime_checks.append((model_id, adapter_path, draft_model_id))
                or False
            ),
        )
        seen_loads: list[tuple[str, str | None, str | None]] = []
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_wrapper",
            lambda model_id, adapter_path=None, draft_model_id=None: (
                seen_loads.append((model_id, adapter_path, draft_model_id)) or object()
            ),
        )
        monkeypatch.setattr(
            wrapper_cache_module.wrapper_cache,
            "get_cache_info",
            lambda: {"cache_size": 2},
        )

        service = models_service_module.ModelsService()
        result = service.load_model(
            "model-a",
            adapter_path="/adapter/frontier",
            draft_model_id="draft-a",
        )

        assert result["status"] == "loaded"
        assert seen_runtime_checks == [("model-a", "/adapter/frontier", "draft-a")]
        assert seen_loads == [("model-a", "/adapter/frontier", "draft-a")]
