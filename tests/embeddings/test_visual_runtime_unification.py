from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from mlx_batch_server.chat.mlx import runtime_aliases as runtime_aliases_module
from mlx_batch_server.chat.mlx import runtime_attachments as runtime_attachments_module
from mlx_batch_server.embeddings import embeddings_service as embeddings_service_module
from mlx_batch_server.embeddings import qwen3_vl_embedder as embedder_module
from mlx_batch_server.embeddings import visual_router as visual_router_module
from mlx_batch_server.embeddings.embeddings_service import EmbeddingsService
from mlx_batch_server.embeddings.schema import EmbeddingRequest


def _clear_visual_state() -> None:
    runtime_aliases_module.clear_runtime_aliases()
    runtime_attachments_module.clear_runtime_surface_attachments()
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
    assert runtime_attachments_module.get_runtime_surface_attachments(
        "frontier-vlm"
    ) == ["visual"]

    _clear_visual_state()


def test_visual_router_canonicalizes_projection_and_processor_identity(monkeypatch):
    _clear_visual_state()

    created_configs: list[tuple[str, str | None, str | None]] = []
    projection_path = "~/weights/frontier-projection.safetensors"
    expanded_projection_path = str(Path(projection_path).expanduser())

    monkeypatch.setattr(
        embedder_module.Qwen3VLEmbedder,
        "load",
        lambda self: created_configs.append(
            (self.model_id, self.projection_path, self.processor_id)
        ),
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

    alias_embedder = visual_router_module.get_visual_embedder(
        "frontier-vlm",
        projection_path=projection_path,
        processor_id="LibraxisAI/Qwen3-VL-30B",
    )
    canonical_embedder = visual_router_module.get_visual_embedder(
        "libraxisai/qwen3-vl-30b",
        projection_path=expanded_projection_path,
        processor_id="libraxisai/qwen3-vl-30b",
    )

    assert alias_embedder is canonical_embedder
    assert created_configs == [
        (
            "libraxisai/qwen3-vl-30b",
            expanded_projection_path,
            "libraxisai/qwen3-vl-30b",
        )
    ]
    assert len(visual_router_module._embedder_cache) == 1
    assert runtime_attachments_module.get_runtime_surface_attachments(
        "LibraxisAI/Qwen3-VL-30B"
    ) == ["visual"]

    _clear_visual_state()


@pytest.mark.asyncio
async def test_visual_router_reserves_endpoint_runtime_for_canonical_shared_vlm(
    monkeypatch,
):
    _clear_visual_state()
    runtime_aliases_module.register_runtime_alias(
        "frontier-vlm",
        "LibraxisAI/Qwen3-VL-30B",
    )

    calls: list[tuple[str, str]] = []

    class FakeEmbedder:
        embedding_dim = 2

        def embed_text(self, text: str):
            calls.append(("embed_text", text))
            return SimpleNamespace(num_tokens=3, source_type="text")

        @staticmethod
        def to_numpy(result):
            return np.array([1.0, 2.0], dtype=np.float32)

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
        visual_router_module,
        "endpoint_runtime_session",
        fake_endpoint_runtime_session,
    )
    monkeypatch.setattr(
        visual_router_module,
        "_get_embedder",
        lambda model_id, projection_path, processor_id: (
            calls.append(("get_embedder", model_id)),
            FakeEmbedder(),
        )[1],
    )

    response = await visual_router_module.create_visual_embeddings(
        visual_router_module.VisualEmbeddingRequest(
            model="frontier-vlm",
            texts=["hello"],
        )
    )

    assert calls == [
        ("session", "libraxisai/qwen3-vl-30b"),
        ("get_embedder", "frontier-vlm"),
        ("embed_text", "hello"),
    ]
    assert response["model"] == "frontier-vlm"
    assert response["dim"] == 2
    assert response["text_embeddings"][0]["embedding"] == [1.0, 2.0]

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


def test_unload_visual_embedder_preserves_runtime_when_llm_surface_remains(
    monkeypatch,
):
    _clear_visual_state()

    monkeypatch.setattr(embedder_module.Qwen3VLEmbedder, "load", lambda self: None)
    monkeypatch.setattr(
        embedder_module.Qwen3VLEmbedder,
        "log_summary",
        lambda self: None,
    )

    visual_router_module.get_visual_embedder("LibraxisAI/Qwen3-VL-30B")
    runtime_attachments_module.attach_runtime_surface(
        "libraxisai/qwen3-vl-30b",
        "llm",
    )

    unloaded_calls: list[str] = []
    monkeypatch.setattr(
        visual_router_module.wrapper_cache,
        "unload_vlm_model",
        lambda model_id=None: unloaded_calls.append(model_id) or [model_id],
    )

    unloaded = visual_router_module.unload_visual_embedder("libraxisai/qwen3-vl-30b")

    assert unloaded == ["libraxisai/qwen3-vl-30b"]
    assert unloaded_calls == []
    assert runtime_attachments_module.get_runtime_surface_attachments(
        "libraxisai/qwen3-vl-30b"
    ) == ["llm"]

    _clear_visual_state()


def test_unload_visual_embedder_exact_runtime_preserves_sibling_variant(monkeypatch):
    _clear_visual_state()

    monkeypatch.setattr(embedder_module.Qwen3VLEmbedder, "load", lambda self: None)
    monkeypatch.setattr(
        embedder_module.Qwen3VLEmbedder,
        "log_summary",
        lambda self: None,
    )

    visual_router_module.get_visual_embedder(
        "LibraxisAI/Qwen3-VL-30B",
        adapter_path="/adapter-a",
    )
    visual_router_module.get_visual_embedder(
        "LibraxisAI/Qwen3-VL-30B",
        adapter_path="/adapter-b",
    )

    unloaded_calls: list[tuple[str | None, str | None, str | None]] = []
    monkeypatch.setattr(
        visual_router_module.wrapper_cache,
        "unload_vlm_model",
        lambda model_id=None, **kwargs: unloaded_calls.append(
            (
                model_id,
                kwargs.get("adapter_path"),
                kwargs.get("draft_model_id"),
            )
        )
        or [model_id],
    )

    unloaded = visual_router_module.unload_visual_embedder(
        "libraxisai/qwen3-vl-30b",
        adapter_path="/adapter-a",
    )

    assert unloaded == ["libraxisai/qwen3-vl-30b"]
    assert unloaded_calls == [("libraxisai/qwen3-vl-30b", "/adapter-a", None)]
    assert (
        visual_router_module.has_visual_embedder(
            "libraxisai/qwen3-vl-30b",
            adapter_path="/adapter-a",
        )
        is False
    )
    assert (
        visual_router_module.has_visual_embedder(
            "libraxisai/qwen3-vl-30b",
            adapter_path="/adapter-b",
        )
        is True
    )
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

    _clear_visual_state()


def test_qwen3_vl_embedder_loads_from_shared_runtime(monkeypatch, tmp_path):
    (tmp_path / "preprocessor_config.json").write_text("{}", encoding="utf-8")

    fake_model = SimpleNamespace(config=SimpleNamespace(image_token_id=777))
    fake_processor = SimpleNamespace(tokenizer=SimpleNamespace(image_token_id=888))
    backend_calls: list[tuple[str, str | None]] = []

    monkeypatch.setattr(embedder_module, "get_model_path", lambda model_id: tmp_path)
    monkeypatch.setattr(
        embedder_module.wrapper_cache,
        "get_vlm_backend",
        lambda model_id, **kwargs: backend_calls.append(
            (model_id, kwargs.get("surface"))
        )
        or (fake_model, fake_processor),
    )
    monkeypatch.setattr(
        embedder_module.AutoProcessor,
        "from_pretrained",
        staticmethod(lambda model_id, trust_remote_code=True: SimpleNamespace()),
    )

    embedder = embedder_module.Qwen3VLEmbedder("LibraxisAI/Qwen3-VL-30B")
    embedder.load()

    assert backend_calls == [("libraxisai/qwen3-vl-30b", "visual")]
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


def test_embeddings_service_attaches_shared_runtime_surface_on_lazy_load(monkeypatch):
    _clear_visual_state()

    monkeypatch.setattr(
        embeddings_service_module,
        "resolves_to_multimodal_runtime",
        lambda model_id: True,
    )
    monkeypatch.setattr(
        embeddings_service_module.SharedVLMTextEmbedder,
        "load",
        lambda self: None,
    )

    service = EmbeddingsService()
    embedder = service._get_shared_vlm_embedder("LibraxisAI/Qwen3-VL-30B")

    assert embedder.model_id == "libraxisai/qwen3-vl-30b"
    assert (
        runtime_attachments_module.get_runtime_surface_attachments("frontier-vlm") == []
    )
    assert runtime_attachments_module.get_runtime_surface_attachments(
        "LibraxisAI/Qwen3-VL-30B"
    ) == ["embeddings"]


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


def test_qwen3_vl_embedder_resets_nested_language_runtime_state_before_text_embed(
    monkeypatch,
):
    embedder = embedder_module.Qwen3VLEmbedder("model-vlm")
    embedder._loaded = True
    embedder.tomoro_processor = SimpleNamespace(
        tokenizer=lambda text, return_tensors=None: {
            "input_ids": [[1, 2]],
        }
    )

    inner_model = SimpleNamespace(
        _position_ids="stale-inner",
        _rope_deltas="stale-inner",
    )

    def fake_embed_tokens(input_ids):
        assert language_wrapper._position_ids is None
        assert inner_model._position_ids is None
        assert inner_model._rope_deltas is None
        return mx.ones((1, input_ids.shape[1], 4))

    inner_model.embed_tokens = fake_embed_tokens
    language_wrapper = SimpleNamespace(
        model=inner_model,
        _position_ids="stale-outer",
    )
    runtime_model = SimpleNamespace(language_model=language_wrapper)

    monkeypatch.setattr(
        embedder,
        "_get_backend",
        lambda: (
            runtime_model,
            SimpleNamespace(tokenizer=embedder.tomoro_processor.tokenizer),
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
    assert language_wrapper._position_ids is None
    assert inner_model._position_ids is None
    assert inner_model._rope_deltas is None


def test_embeddings_service_routes_qwen3_vl_text_to_shared_runtime(monkeypatch):
    service = EmbeddingsService()
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(
        embeddings_service_module,
        "resolves_to_multimodal_runtime",
        lambda model_id: True,
    )

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
        embeddings_service_module,
        "resolves_to_multimodal_runtime",
        lambda model_id: True,
    )

    monkeypatch.setattr(
        service,
        "canonicalize_model_id",
        lambda model_id: "libraxisai/qwen3-vl-30b",
    )
    monkeypatch.setattr(
        service,
        "_get_shared_vlm_embedder",
        lambda model_id, **kwargs: loaded.append(
            (model_id, kwargs.get("adapter_path"), kwargs.get("draft_model_id"))
        )
        or state.__setitem__("runtime_loaded", True)
        or object(),
    )
    monkeypatch.setattr(
        service,
        "_unload_shared_vlm_embedder",
        lambda model_id, **kwargs: unloaded.append(
            (model_id, kwargs.get("adapter_path"), kwargs.get("draft_model_id"))
        )
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

    assert loaded == [("libraxisai/qwen3-vl-30b", None, None)]
    assert unloaded == [
        ("libraxisai/qwen3-vl-30b", None, None),
        ("libraxisai/qwen3-vl-30b", None, None),
    ]


def test_embeddings_service_distinguishes_shared_vlm_runtime_variants(monkeypatch):
    _clear_visual_state()
    service = EmbeddingsService()
    loaded: list[tuple[str, str | None, str | None]] = []

    monkeypatch.setattr(
        embeddings_service_module,
        "resolves_to_multimodal_runtime",
        lambda model_id: True,
    )
    monkeypatch.setattr(
        service,
        "_get_shared_vlm_embedder",
        lambda model_id, **kwargs: loaded.append(
            (model_id, kwargs.get("adapter_path"), kwargs.get("draft_model_id"))
        )
        or object(),
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
        service.load_model("LibraxisAI/Qwen3-VL-30B", adapter_path="/adapter-a") is True
    )

    assert loaded == [
        ("libraxisai/qwen3-vl-30b", "/adapter-a", None),
        ("libraxisai/qwen3-vl-30b", "/adapter-b", None),
    ]

    _clear_visual_state()


def test_embeddings_service_unload_preserves_runtime_when_llm_surface_remains(
    monkeypatch,
):
    _clear_visual_state()
    service = EmbeddingsService()
    unloaded: list[str] = []

    monkeypatch.setattr(
        embeddings_service_module,
        "resolves_to_multimodal_runtime",
        lambda model_id: True,
    )
    monkeypatch.setattr(
        service,
        "canonicalize_model_id",
        lambda model_id: "libraxisai/qwen3-vl-30b",
    )
    monkeypatch.setattr(
        service,
        "_get_shared_vlm_embedder",
        lambda model_id, **kwargs: object(),
    )
    monkeypatch.setattr(
        service,
        "_unload_shared_vlm_embedder",
        lambda model_id, **kwargs: unloaded.append(
            (model_id, kwargs.get("adapter_path"), kwargs.get("draft_model_id"))
        )
        or [model_id],
    )

    assert service.load_model("LibraxisAI/Qwen3-VL-30B") is False
    runtime_attachments_module.attach_runtime_surface(
        "libraxisai/qwen3-vl-30b",
        "llm",
    )

    assert service.unload_model("LibraxisAI/Qwen3-VL-30B") is True
    assert unloaded == []
    assert runtime_attachments_module.get_runtime_surface_attachments(
        "libraxisai/qwen3-vl-30b"
    ) == ["llm"]

    _clear_visual_state()


def test_embeddings_service_clear_models_releases_shared_vlm_alias(monkeypatch):
    _clear_visual_state()
    runtime_aliases_module.register_runtime_alias(
        "frontier-vlm",
        "LibraxisAI/Qwen3-VL-30B",
    )

    service = EmbeddingsService()
    unloaded: list[str] = []
    monkeypatch.setattr(
        embeddings_service_module,
        "resolves_to_multimodal_runtime",
        lambda model_id: True,
    )

    monkeypatch.setattr(
        service,
        "_get_shared_vlm_embedder",
        lambda model_id, **kwargs: object(),
    )
    monkeypatch.setattr(
        service,
        "_unload_shared_vlm_embedder",
        lambda model_id, **kwargs: unloaded.append(
            (model_id, kwargs.get("adapter_path"), kwargs.get("draft_model_id"))
        )
        or [model_id],
    )

    assert service.load_model("frontier-vlm") is False
    assert service.has_shared_vlm_runtime_models() is True
    assert service.clear_models() == ["libraxisai/qwen3-vl-30b"]
    assert service.has_shared_vlm_runtime_models() is False
    assert unloaded == [("libraxisai/qwen3-vl-30b", None, None)]

    _clear_visual_state()


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
