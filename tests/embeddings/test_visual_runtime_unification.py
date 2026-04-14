from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_batch_server.chat.mlx import runtime_aliases as runtime_aliases_module
from mlx_batch_server.embeddings import embeddings_service as embeddings_service_module
from mlx_batch_server.embeddings import qwen3_vl_embedder as embedder_module
from mlx_batch_server.embeddings import visual_router as visual_router_module
from mlx_batch_server.embeddings.embeddings_service import EmbeddingsService
from mlx_batch_server.embeddings.schema import EmbeddingRequest


def _clear_visual_state() -> None:
    runtime_aliases_module.clear_runtime_aliases()
    visual_router_module._embedder_cache.clear()


def test_visual_router_reuses_canonical_runtime_alias(monkeypatch):
    _clear_visual_state()

    created_ids: list[str] = []
    monkeypatch.setattr(
        embedder_module.Qwen3VLEmbedder,
        "load",
        lambda self: created_ids.append(self.model_id),
    )
    monkeypatch.setattr(
        embedder_module.Qwen3VLEmbedder,
        "log_summary",
        lambda self: None,
    )

    runtime_aliases_module.register_runtime_alias(
        "frontier-vlm",
        "LibraxisAI/Qwen3-VL-30B",
    )

    alias_embedder = visual_router_module.get_visual_embedder("frontier-vlm")
    canonical_embedder = visual_router_module.get_visual_embedder(
        "libraxisai/qwen3-vl-30b"
    )

    assert alias_embedder is canonical_embedder
    assert created_ids == ["libraxisai/qwen3-vl-30b"]
    assert len(visual_router_module._embedder_cache) == 1

    _clear_visual_state()


def test_unload_visual_embedder_clears_shared_vlm_runtime(monkeypatch):
    _clear_visual_state()

    monkeypatch.setattr(embedder_module.Qwen3VLEmbedder, "load", lambda self: None)
    monkeypatch.setattr(
        embedder_module.Qwen3VLEmbedder,
        "log_summary",
        lambda self: None,
    )

    visual_router_module.get_visual_embedder("LibraxisAI/Qwen3-VL-30B")

    unloaded_calls: list[str] = []
    monkeypatch.setattr(
        visual_router_module.wrapper_cache,
        "unload_vlm_model",
        lambda model_id=None: unloaded_calls.append(model_id) or [model_id],
    )

    unloaded = visual_router_module.unload_visual_embedder("libraxisai/qwen3-vl-30b")

    assert unloaded == ["libraxisai/qwen3-vl-30b"]
    assert unloaded_calls == ["libraxisai/qwen3-vl-30b"]
    assert visual_router_module._embedder_cache == {}

    _clear_visual_state()


def test_qwen3_vl_embedder_loads_from_shared_runtime(monkeypatch, tmp_path):
    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    fake_model = SimpleNamespace(config=SimpleNamespace(image_token_id=777))
    fake_processor = SimpleNamespace(tokenizer=SimpleNamespace(image_token_id=888))
    backend_calls: list[str] = []

    monkeypatch.setattr(embedder_module, "get_model_path", lambda model_id: tmp_path)
    monkeypatch.setattr(
        embedder_module.wrapper_cache,
        "get_vlm_backend",
        lambda model_id: backend_calls.append(model_id) or (fake_model, fake_processor),
    )
    monkeypatch.setattr(
        embedder_module.AutoProcessor,
        "from_pretrained",
        staticmethod(lambda model_id, trust_remote_code=True: SimpleNamespace()),
    )

    embedder = embedder_module.Qwen3VLEmbedder("LibraxisAI/Qwen3-VL-30B")
    embedder.load()

    assert backend_calls == ["libraxisai/qwen3-vl-30b"]
    assert embedder.model is None
    assert embedder.processor is None
    assert embedder._image_token_id == 777


def test_qwen3_vl_embedder_serializes_shared_runtime_on_text_embed(monkeypatch):
    embedder = embedder_module.Qwen3VLEmbedder("model-vlm")
    embedder._loaded = True
    embedder.tomoro_processor = SimpleNamespace(
        tokenizer=lambda text, return_tensors=None: {
            "input_ids": [[1, 2]],
        }
    )

    events: list[tuple[str, str]] = []

    @contextmanager
    def fake_vlm_execution(model_id: str):
        events.append(("enter", model_id))
        try:
            yield
        finally:
            events.append(("exit", model_id))

    monkeypatch.setattr(
        embedder_module.wrapper_cache,
        "vlm_execution",
        fake_vlm_execution,
    )
    monkeypatch.setattr(
        embedder,
        "_get_backend",
        lambda: (
            SimpleNamespace(),
            SimpleNamespace(tokenizer=embedder.tomoro_processor.tokenizer),
        ),
    )
    monkeypatch.setattr(
        embedder,
        "_get_language_model",
        lambda model: SimpleNamespace(
            embed_tokens=lambda input_ids: mx.ones((1, input_ids.shape[1], 4))
        ),
    )
    monkeypatch.setattr(
        embedder,
        "_build_position_ids",
        lambda batch_size, seq_len: mx.zeros((3, batch_size, seq_len), dtype=mx.int32),
    )
    monkeypatch.setattr(
        embedder,
        "_run_language_layers",
        lambda inner_model, inputs_embeds, position_ids: inputs_embeds,
    )
    monkeypatch.setattr(
        embedder,
        "_project_and_normalize",
        lambda hidden_states: hidden_states,
    )

    result = embedder.embed_text("hello")

    assert result.num_tokens == 2
    assert events == [("enter", "model-vlm"), ("exit", "model-vlm")]


def test_qwen3_vl_embedder_pools_sentence_embedding_from_shared_runtime(monkeypatch):
    embedder = embedder_module.Qwen3VLEmbedder("model-vlm")
    embedder._loaded = True

    hidden_states = mx.array(
        [[[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]],
        dtype=mx.float32,
    )
    attention_mask = mx.array([[1, 1, 0]], dtype=mx.int32)

    monkeypatch.setattr(
        embedder,
        "_embed_text_hidden_states",
        lambda text: (
            hidden_states,
            mx.array([[1, 2, 0]], dtype=mx.int64),
            attention_mask,
            2,
        ),
    )
    monkeypatch.setattr(
        embedder,
        "_project_and_normalize",
        lambda pooled: pooled,
    )

    result = embedder.embed_text_pooled("hello")

    assert result.num_tokens == 2
    assert result.source_type == "text"
    assert embedder_module.Qwen3VLEmbedder.to_numpy(result).tolist() == [0.0, 2.0]


def test_embeddings_service_routes_qwen3_vl_text_to_shared_runtime(monkeypatch):
    service = EmbeddingsService()
    events: list[tuple[str, str]] = []

    class FakeEmbedder:
        def embed_text_pooled(self, text: str):
            events.append(("embed_text", text))
            return SimpleNamespace(
                num_tokens=7,
                embeddings=np.array([0.70710677, 0.70710677], dtype=np.float32),
            )

    monkeypatch.setattr(
        service,
        "_get_shared_vlm_embedder",
        lambda model_id: events.append(("get_embedder", model_id)) or FakeEmbedder(),
    )
    monkeypatch.setattr(
        service,
        "canonicalize_model_id",
        lambda model_id: "libraxisai/qwen3-vl-30b",
    )

    response = service.generate_embeddings(
        EmbeddingRequest(
            model="LibraxisAI/Qwen3-VL-30B",
            input=["alpha", "beta"],
        )
    )

    assert events == [
        ("get_embedder", "libraxisai/qwen3-vl-30b"),
        ("embed_text", "alpha"),
        ("embed_text", "beta"),
    ]
    assert service._shared_vlm_models == {"libraxisai/qwen3-vl-30b"}
    assert response.usage.prompt_tokens == 14
    assert response.usage.total_tokens == 14
    assert response.data[0].embedding == pytest.approx([0.70710677, 0.70710677])
    assert response.data[1].embedding == pytest.approx([0.70710677, 0.70710677])


def test_embeddings_service_tracks_shared_vlm_load_and_unload(monkeypatch):
    service = EmbeddingsService()
    loaded: list[str] = []
    unloaded: list[str] = []
    state = {"runtime_loaded": False}

    monkeypatch.setattr(
        service,
        "canonicalize_model_id",
        lambda model_id: "libraxisai/qwen3-vl-30b",
    )
    monkeypatch.setattr(
        service,
        "_get_shared_vlm_embedder",
        lambda model_id: loaded.append(model_id) or state.__setitem__("runtime_loaded", True) or object(),
    )
    monkeypatch.setattr(
        service,
        "_unload_shared_vlm_embedder",
        lambda model_id: unloaded.append(model_id)
        or (
            state.__setitem__("runtime_loaded", False) or [model_id]
            if state["runtime_loaded"]
            else []
        ),
    )

    assert service.load_model("LibraxisAI/Qwen3-VL-30B") is False
    assert service.load_model("LibraxisAI/Qwen3-VL-30B") is True
    assert service.unload_model("LibraxisAI/Qwen3-VL-30B") is True
    assert service.unload_model("LibraxisAI/Qwen3-VL-30B") is False

    assert loaded == ["libraxisai/qwen3-vl-30b"]
    assert unloaded == ["libraxisai/qwen3-vl-30b"]


def test_embeddings_service_clears_native_mlx_cache_after_request(monkeypatch):
    service = EmbeddingsService()
    cleared: list[str] = []

    monkeypatch.setattr(
        service,
        "_get_model",
        lambda model_id: (object(), object()),
    )
    monkeypatch.setattr(
        service,
        "_get_bert_embeddings",
        lambda model, processor, text, model_id: np.array([[1.0, 2.0]]),
    )
    monkeypatch.setattr(
        embeddings_service_module.mx,
        "clear_cache",
        lambda: cleared.append("clear"),
    )

    response = service.generate_embeddings(
        EmbeddingRequest(
            model="mlx-community/all-MiniLM-L6-v2-4bit",
            input="cache hygiene",
        )
    )

    assert len(response.data) == 1
    assert cleared == ["clear"]
