from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx

from mlx_batch_server.chat.mlx import runtime_aliases as runtime_aliases_module
from mlx_batch_server.embeddings import embeddings_service as embeddings_service_module
from mlx_batch_server.embeddings.embeddings_service import EmbeddingsService
from mlx_batch_server.embeddings.schema import EmbeddingRequest


def _clear_runtime_aliases() -> None:
    runtime_aliases_module.clear_runtime_aliases()


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
        embed_text_pooled=lambda text: embed_calls.append(text)
        or SimpleNamespace(
            embeddings=mx.array([1.0, 2.0, 3.0], dtype=mx.float32),
            num_tokens=3,
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
        lambda model_id: unloaded_calls.append(model_id) or [model_id],
    )

    runtime_aliases_module.register_runtime_alias(
        "frontier-vlm",
        "LibraxisAI/Qwen3-VL-30B",
    )

    assert service.unload_model("frontier-vlm") is True
    assert unloaded_calls == ["libraxisai/qwen3-vl-30b"]
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
        lambda model_id: unloaded_calls.append(model_id) or [model_id],
    )

    assert service.unload_model("LibraxisAI/Qwen3-VL-30B") is True
    assert unloaded_calls == ["libraxisai/qwen3-vl-30b"]

    _clear_runtime_aliases()


def test_embeddings_service_routes_non_qwen_vlm_alias_to_shared_runtime(monkeypatch):
    _clear_runtime_aliases()
    service = EmbeddingsService()
    created_ids: list[str] = []
    monkeypatch.setattr(
        embeddings_service_module,
        "resolves_to_multimodal_runtime",
        lambda model_id: "pixtral"
        in runtime_aliases_module.resolve_runtime_model_id(model_id).lower(),
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
