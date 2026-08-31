from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_batch_server.chat.mlx import runtime_aliases as runtime_aliases_module
from mlx_batch_server.chat.mlx import runtime_attachments as runtime_attachments_module
from mlx_batch_server.embeddings import embeddings_service as embeddings_service_module
from mlx_batch_server.embeddings import router as embeddings_router_module
from mlx_batch_server.embeddings.embeddings_service import EmbeddingsService
from mlx_batch_server.embeddings.schema import EmbeddingRequest


def _clear_runtime_aliases() -> None:
    runtime_aliases_module.clear_runtime_aliases()
    runtime_attachments_module.clear_runtime_surface_attachments()


def test_embeddings_service_reuses_canonical_runtime_alias_for_shared_vlm(
    monkeypatch,
):
    _clear_runtime_aliases()
    service = EmbeddingsService()
    created_ids: list[str] = []
    monkeypatch.setattr(
        embeddings_service_module,
        "resolves_to_multimodal_runtime",
        lambda model_id: True,
    )

    monkeypatch.setattr(
        service,
        "_get_shared_vlm_embedder",
        lambda model_id: created_ids.append(model_id) or object(),
    )

    runtime_aliases_module.register_runtime_alias(
        "frontier-vlm",
        "LibraxisAI/Qwen3-VL-30B",
    )

    alias_loaded = service.load_model("frontier-vlm")
    canonical_loaded = service.load_model("libraxisai/qwen3-vl-30b")

    assert alias_loaded is False
    assert canonical_loaded is True
    assert created_ids == ["libraxisai/qwen3-vl-30b"]
    assert service.has_shared_vlm_runtime_models() is True

    _clear_runtime_aliases()


def test_embeddings_service_uses_pooled_shared_vlm_sentence_embeddings(monkeypatch):
    _clear_runtime_aliases()
    service = EmbeddingsService()
    embed_calls: list[str] = []
    monkeypatch.setattr(
        embeddings_service_module,
        "resolves_to_multimodal_runtime",
        lambda model_id: True,
    )

    fake_embedder = SimpleNamespace(
        embed_text_pooled=lambda text: (
            embed_calls.append(text)
            or SimpleNamespace(
                embeddings=mx.array([1.0, 2.0, 3.0], dtype=mx.float32),
                num_tokens=3,
            )
        )
    )
    monkeypatch.setattr(
        service,
        "_get_shared_vlm_embedder",
        lambda model_id: fake_embedder,
    )

    response = service.generate_embeddings(
        EmbeddingRequest(model="LibraxisAI/Qwen3-VL-30B", input=["hello", "world"])
    )

    assert embed_calls == ["hello", "world"]
    assert response.model == "LibraxisAI/Qwen3-VL-30B"
    assert response.data[0].embedding == [1.0, 2.0, 3.0]
    assert response.data[1].embedding == [1.0, 2.0, 3.0]
    assert response.usage.prompt_tokens == 6

    _clear_runtime_aliases()


def test_embeddings_service_unload_clears_shared_vlm_runtime(monkeypatch):
    _clear_runtime_aliases()
    service = EmbeddingsService()
    service._shared_vlm_models.add("libraxisai/qwen3-vl-30b")
    unloaded_calls: list[str] = []
    monkeypatch.setattr(
        embeddings_service_module,
        "resolves_to_multimodal_runtime",
        lambda model_id: True,
    )

    monkeypatch.setattr(
        service,
        "_unload_shared_vlm_embedder",
        lambda model_id, **kwargs: (
            unloaded_calls.append(
                (model_id, kwargs.get("adapter_path"), kwargs.get("draft_model_id"))
            )
            or [model_id]
        ),
    )

    runtime_aliases_module.register_runtime_alias(
        "frontier-vlm",
        "LibraxisAI/Qwen3-VL-30B",
    )

    assert service.unload_model("frontier-vlm") is True
    assert unloaded_calls == [("libraxisai/qwen3-vl-30b", None, None)]
    assert service.has_shared_vlm_runtime_models() is False

    _clear_runtime_aliases()


def test_embeddings_service_unload_reports_runtime_only_shared_vlm(monkeypatch):
    _clear_runtime_aliases()
    service = EmbeddingsService()
    unloaded_calls: list[str] = []
    monkeypatch.setattr(
        embeddings_service_module,
        "resolves_to_multimodal_runtime",
        lambda model_id: True,
    )

    monkeypatch.setattr(
        service,
        "_unload_shared_vlm_embedder",
        lambda model_id, **kwargs: (
            unloaded_calls.append(
                (model_id, kwargs.get("adapter_path"), kwargs.get("draft_model_id"))
            )
            or [model_id]
        ),
    )

    assert service.unload_model("LibraxisAI/Qwen3-VL-30B") is True
    assert unloaded_calls == [("libraxisai/qwen3-vl-30b", None, None)]

    _clear_runtime_aliases()


def test_embeddings_service_unload_exact_runtime_preserves_sibling_variant(
    monkeypatch,
):
    _clear_runtime_aliases()
    service = EmbeddingsService()
    unloaded_calls: list[tuple[str, str | None, str | None]] = []

    monkeypatch.setattr(
        embeddings_service_module,
        "resolves_to_multimodal_runtime",
        lambda model_id: True,
    )

    def fake_get_shared_vlm_embedder(model_id: str, **kwargs):
        runtime_attachments_module.attach_runtime_surface(
            model_id,
            "embeddings",
            adapter_path=kwargs.get("adapter_path"),
            draft_model_id=kwargs.get("draft_model_id"),
        )
        return object()

    monkeypatch.setattr(
        service,
        "_get_shared_vlm_embedder",
        fake_get_shared_vlm_embedder,
    )
    monkeypatch.setattr(
        service,
        "_unload_shared_vlm_embedder",
        lambda model_id, **kwargs: (
            unloaded_calls.append(
                (model_id, kwargs.get("adapter_path"), kwargs.get("draft_model_id"))
            )
            or [model_id]
        ),
    )

    assert (
        service.load_model("LibraxisAI/Qwen3-VL-30B", adapter_path="/adapter-a")
        is False
    )
    assert (
        service.load_model("LibraxisAI/Qwen3-VL-30B", adapter_path="/adapter-b")
        is False
    )

    assert (
        service.unload_model("LibraxisAI/Qwen3-VL-30B", adapter_path="/adapter-a")
        is True
    )
    assert unloaded_calls == [("libraxisai/qwen3-vl-30b", "/adapter-a", None)]
    assert service.get_shared_vlm_runtime_keys() == [
        ("libraxisai/qwen3-vl-30b", "/adapter-b", None)
    ]
    assert service.get_shared_vlm_models() == ["libraxisai/qwen3-vl-30b"]
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
    ) == ["embeddings"]

    _clear_runtime_aliases()


def test_embeddings_service_routes_non_qwen_vlm_alias_to_shared_runtime(monkeypatch):
    _clear_runtime_aliases()
    service = EmbeddingsService()
    created_ids: list[str] = []
    monkeypatch.setattr(
        embeddings_service_module,
        "resolves_to_multimodal_runtime",
        lambda model_id: (
            "pixtral"
            in runtime_aliases_module.resolve_runtime_model_id(model_id).lower()
        ),
    )
    monkeypatch.setattr(
        service,
        "_get_shared_vlm_embedder",
        lambda model_id: created_ids.append(model_id) or object(),
    )

    runtime_aliases_module.register_runtime_alias(
        "frontier-vlm",
        "mlx-community/pixtral-12b-4bit",
    )

    alias_loaded = service.load_model("frontier-vlm")
    canonical_loaded = service.load_model("MLX-Community/Pixtral-12B-4bit")

    assert alias_loaded is False
    assert canonical_loaded is True
    assert created_ids == ["mlx-community/pixtral-12b-4bit"]
    assert service.has_shared_vlm_runtime_models() is True

    _clear_runtime_aliases()


@pytest.mark.asyncio
async def test_embeddings_router_reserves_endpoint_runtime_for_shared_vlm(monkeypatch):
    calls: list[tuple[str, str]] = []

    class FakeService:
        def uses_shared_vlm_runtime(self, model_id: str) -> bool:
            calls.append(("shared", model_id))
            return True

        def canonicalize_model_id(self, model_id: str) -> str:
            calls.append(("canonicalize", model_id))
            return "libraxisai/qwen3-vl-30b"

        def generate_embeddings(self, request: EmbeddingRequest):
            calls.append(("generate", request.model))
            return "ok"

    @asynccontextmanager
    async def fake_endpoint_runtime_session(
        model_id: str,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ):
        assert adapter_path is None
        assert draft_model_id is None
        calls.append(("session", model_id))
        yield {"switched": False}

    monkeypatch.setattr(
        embeddings_router_module,
        "embeddings_service",
        FakeService(),
    )
    monkeypatch.setattr(
        embeddings_router_module,
        "endpoint_runtime_session",
        fake_endpoint_runtime_session,
    )

    result = await embeddings_router_module.create_embeddings(
        EmbeddingRequest(model="frontier-vlm", input="hello")
    )

    assert result == "ok"
    assert calls == [
        ("shared", "frontier-vlm"),
        ("canonicalize", "frontier-vlm"),
        ("session", "libraxisai/qwen3-vl-30b"),
        ("generate", "frontier-vlm"),
    ]


@pytest.mark.asyncio
async def test_embeddings_router_skips_endpoint_runtime_for_native_embeddings(
    monkeypatch,
):
    calls: list[tuple[str, str]] = []

    class FakeService:
        def uses_shared_vlm_runtime(self, model_id: str) -> bool:
            calls.append(("shared", model_id))
            return False

        def canonicalize_model_id(self, model_id: str) -> str:
            raise AssertionError("canonicalize_model_id should not be used")

        def generate_embeddings(self, request: EmbeddingRequest):
            calls.append(("generate", request.model))
            return "native"

    @asynccontextmanager
    async def fake_endpoint_runtime_session(*args, **kwargs):
        raise AssertionError("endpoint_runtime_session should not run")
        yield  # pragma: no cover

    monkeypatch.setattr(
        embeddings_router_module,
        "embeddings_service",
        FakeService(),
    )
    monkeypatch.setattr(
        embeddings_router_module,
        "endpoint_runtime_session",
        fake_endpoint_runtime_session,
    )

    result = await embeddings_router_module.create_embeddings(
        EmbeddingRequest(model="mlx-community/all-MiniLM-L6-v2-4bit", input="hello")
    )

    assert result == "native"
    assert calls == [
        ("shared", "mlx-community/all-MiniLM-L6-v2-4bit"),
        ("generate", "mlx-community/all-MiniLM-L6-v2-4bit"),
    ]
