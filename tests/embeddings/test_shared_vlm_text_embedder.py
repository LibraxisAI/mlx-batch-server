from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mlx_batch_server.chat.mlx import runtime_aliases as runtime_aliases_module
from mlx_batch_server.embeddings import shared_vlm_text_embedder as embedder_module


def test_shared_vlm_text_embedder_pools_last_non_padding_token(monkeypatch):
    events: list[tuple[str, str]] = []

    @contextmanager
    def fake_vlm_execution(model_id: str):
        events.append(("enter", model_id))
        try:
            yield
        finally:
            events.append(("exit", model_id))

    fake_language_model = SimpleNamespace(
        model=SimpleNamespace(
            embed_tokens=lambda input_ids: mx.array(
                [[[1.0, 0.0], [0.0, 2.0], [3.0, 4.0]]],
                dtype=mx.float32,
            ),
            layers=[
                lambda hidden, position_ids=None: hidden,
            ],
            norm=lambda hidden: hidden,
        )
    )
    fake_model = SimpleNamespace(language_model=fake_language_model)
    fake_processor = SimpleNamespace(
        tokenizer=lambda text, return_tensors=None: {
            "input_ids": [[1, 2, 0]],
            "attention_mask": [[1, 1, 0]],
        }
    )

    monkeypatch.setattr(
        embedder_module.wrapper_cache,
        "get_vlm_backend",
        lambda model_id, **kwargs: (fake_model, fake_processor),
    )
    monkeypatch.setattr(
        embedder_module.wrapper_cache,
        "vlm_execution",
        fake_vlm_execution,
    )

    embedder = embedder_module.SharedVLMTextEmbedder("MLX-Community/Pixtral-12B-4bit")
    result = embedder.embed_text_pooled("hello")

    assert result.num_tokens == 2
    assert result.source_type == "text"
    assert result.embeddings.tolist() == pytest.approx([0.0, 1.0])
    assert embedder.embedding_dim == 2
    assert events == [
        ("enter", "mlx-community/pixtral-12b-4bit"),
        ("exit", "mlx-community/pixtral-12b-4bit"),
    ]


def test_shared_vlm_text_embedder_resets_nested_language_runtime_state(monkeypatch):
    inner_model = SimpleNamespace(
        _position_ids="stale-inner",
        _rope_deltas="stale-inner",
    )

    def fake_embed_tokens(input_ids):
        assert language_wrapper._position_ids is None
        assert inner_model._position_ids is None
        assert inner_model._rope_deltas is None
        return mx.array(
            [[[1.0, 0.0], [0.0, 2.0]]],
            dtype=mx.float32,
        )

    inner_model.embed_tokens = fake_embed_tokens
    inner_model.layers = [lambda hidden, position_ids=None: hidden]
    inner_model.norm = lambda hidden: hidden

    language_wrapper = SimpleNamespace(
        model=inner_model,
        _position_ids="stale-outer",
    )
    fake_model = SimpleNamespace(language_model=language_wrapper)
    fake_processor = SimpleNamespace(
        tokenizer=lambda text, return_tensors=None: {
            "input_ids": [[1, 2]],
            "attention_mask": [[1, 1]],
        }
    )

    monkeypatch.setattr(
        embedder_module.wrapper_cache,
        "get_vlm_backend",
        lambda model_id, **kwargs: (fake_model, fake_processor),
    )

    embedder = embedder_module.SharedVLMTextEmbedder("LibraxisAI/Qwen3-VL-30B")
    result = embedder.embed_text_pooled("hello")

    assert result.num_tokens == 2


def test_shared_vlm_text_embedder_resolves_alias_scoped_adapter(monkeypatch):
    expanded_adapter_path = str(Path("~/adapters/frontier-lora").expanduser())
    fake_model = SimpleNamespace(language_model=SimpleNamespace())
    fake_processor = SimpleNamespace(
        tokenizer=lambda text, return_tensors=None: {
            "input_ids": [[1]],
            "attention_mask": [[1]],
        }
    )
    seen: list[tuple[str, str | None, str | None, str | None]] = []

    runtime_aliases_module.clear_runtime_aliases()
    runtime_aliases_module.register_runtime_alias(
        "frontier-vlm",
        "LibraxisAI/Qwen3-VL-30B",
        adapter_path="~/adapters/frontier-lora",
    )

    monkeypatch.setattr(
        embedder_module.wrapper_cache,
        "get_vlm_backend",
        lambda model_id, **kwargs: (
            seen.append(
                (
                    model_id,
                    kwargs.get("adapter_path"),
                    kwargs.get("draft_model_id"),
                    kwargs.get("surface"),
                )
            )
            or (fake_model, fake_processor)
        ),
    )
    monkeypatch.setattr(
        embedder_module.SharedVLMTextEmbedder,
        "_get_language_model",
        lambda self, model: SimpleNamespace(
            embed_tokens=lambda input_ids: mx.ones((1, input_ids.shape[1], 2)),
            layers=[lambda hidden, position_ids=None: hidden],
            norm=lambda hidden: hidden,
        ),
    )

    embedder = embedder_module.SharedVLMTextEmbedder("frontier-vlm")
    embedder.embed_text_pooled("hello")

    assert seen == [
        ("libraxisai/qwen3-vl-30b", expanded_adapter_path, None, "embeddings"),
    ]
    runtime_aliases_module.clear_runtime_aliases()
