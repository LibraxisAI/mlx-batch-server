from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import mlx.core as mx
import pytest

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
        lambda model_id: (fake_model, fake_processor),
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
