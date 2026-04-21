"""Regression tests for the VLM cache compatibility shim."""

from __future__ import annotations

import asyncio
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from mlx_batch_server.chat.mlx import runtime_aliases as runtime_aliases_module
from mlx_batch_server.chat.mlx import runtime_attachments as runtime_attachments_module
from mlx_batch_server.chat.mlx.wrapper_cache import WrapperCacheKey
from mlx_batch_server.core.config import get_settings


@pytest.fixture(autouse=True)
def _reset_runtime_state() -> None:
    runtime_aliases_module.clear_runtime_aliases()
    runtime_attachments_module.clear_runtime_surface_attachments()
    get_settings.cache_clear()
    yield
    runtime_aliases_module.clear_runtime_aliases()
    runtime_attachments_module.clear_runtime_surface_attachments()
    get_settings.cache_clear()


def test_resolve_vlm_model_id_rejects_non_pinned_in_pinned_only_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlx_batch_server.vision.vlm_cache import resolve_vlm_model_id

    monkeypatch.setenv("PINNED_MODELS", "mlx-community/Qwen3-VL-30B-A3B-Instruct-8bit")
    monkeypatch.setenv("MODEL_CACHE_MAX_SIZE", "0")
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="not allowed in pinned-only mode"):
        resolve_vlm_model_id("Qwen/Qwen3-VL-30B-A3B-Instruct")


def test_load_vlm_model_reuses_exact_runtime_key_in_pinned_only_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlx_batch_server.vision import vlm_cache

    monkeypatch.setenv("PINNED_MODELS", "MLX-Community/Qwen3-VL-30B-A3B-Instruct-8bit")
    monkeypatch.setenv("MODEL_CACHE_MAX_SIZE", "0")
    get_settings.cache_clear()

    runtime_keys: list[WrapperCacheKey] = []
    loaded_models: list[str] = []
    backend_calls: list[tuple[str, str | None]] = []

    monkeypatch.setattr(
        vlm_cache.wrapper_cache,
        "get_runtime_keys",
        lambda: list(runtime_keys),
    )
    monkeypatch.setattr(
        vlm_cache.wrapper_cache,
        "get_loaded_vlm_models",
        lambda: list(loaded_models),
    )

    def fake_get_vlm_backend(model_id: str, **kwargs):
        target = runtime_aliases_module.resolve_runtime_target(
            model_id,
            adapter_path=kwargs.get("adapter_path"),
            draft_model_id=kwargs.get("draft_model_id"),
        )
        key = WrapperCacheKey(
            target.model_id,
            target.adapter_path,
            target.draft_model_id,
        )
        if key not in runtime_keys:
            runtime_keys.append(key)
        if target.model_id not in loaded_models:
            loaded_models.append(target.model_id)
        backend_calls.append((target.model_id, kwargs.get("surface")))
        return SimpleNamespace(name="model"), SimpleNamespace(name="processor")

    monkeypatch.setattr(
        vlm_cache.wrapper_cache,
        "get_vlm_backend",
        fake_get_vlm_backend,
    )

    _, _, already_loaded_first = vlm_cache.load_vlm_model(
        "mlx-community/Qwen3-VL-30B-A3B-Instruct-8bit",
        surface="visual",
    )
    _, _, already_loaded_second = vlm_cache.load_vlm_model(
        "MLX-Community/Qwen3-VL-30B-A3B-Instruct-8bit",
        surface="visual",
    )

    assert already_loaded_first is False
    assert already_loaded_second is True
    assert backend_calls == [
        ("mlx-community/qwen3-vl-30b-a3b-instruct-8bit", "visual"),
        ("mlx-community/qwen3-vl-30b-a3b-instruct-8bit", "visual"),
    ]
    assert vlm_cache.get_cache_info()["cached_keys"] == [
        "mlx-community/qwen3-vl-30b-a3b-instruct-8bit"
    ]


def test_load_vlm_model_blocks_non_pinned_without_invoking_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlx_batch_server.vision import vlm_cache

    monkeypatch.setenv("PINNED_MODELS", "mlx-community/Qwen3-VL-30B-A3B-Instruct-8bit")
    monkeypatch.setenv("MODEL_CACHE_MAX_SIZE", "0")
    get_settings.cache_clear()

    called = {"backend": 0}

    def fake_get_vlm_backend(*args, **kwargs):
        called["backend"] += 1
        return SimpleNamespace(), SimpleNamespace()

    monkeypatch.setattr(
        vlm_cache.wrapper_cache,
        "get_vlm_backend",
        fake_get_vlm_backend,
    )

    with pytest.raises(ValueError, match="not allowed in pinned-only mode"):
        vlm_cache.load_vlm_model("Qwen/Qwen3-VL-30B-A3B-Instruct")

    assert called["backend"] == 0


def test_batch_coordinators_reject_non_pinned_models_in_pinned_only_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mlx_batch_server.vision.vlm_batch import (
        get_vlm_batch_coordinator,
        get_vlm_stream_coordinator,
    )

    monkeypatch.setenv("PINNED_MODELS", "mlx-community/Qwen3-VL-30B-A3B-Instruct-8bit")
    monkeypatch.setenv("MODEL_CACHE_MAX_SIZE", "0")
    get_settings.cache_clear()

    with pytest.raises(ValueError, match="not allowed in pinned-only mode"):
        get_vlm_batch_coordinator(
            model_id="Qwen/Qwen3-VL-30B-A3B-Instruct",
            batch_window_ms=50,
            max_batch_size=4,
            group_by_shape=True,
        )

    with pytest.raises(ValueError, match="not allowed in pinned-only mode"):
        get_vlm_stream_coordinator(
            model_id="Qwen/Qwen3-VL-30B-A3B-Instruct",
            batch_window_ms=50,
            max_batch_size=4,
        )


def test_vlm_batch_coordinator_attaches_llm_surface(monkeypatch) -> None:
    from mlx_batch_server.vision import vlm_batch

    backend_calls: list[tuple[str, str | None, str | None, str | None]] = []

    fake_model = SimpleNamespace(config=SimpleNamespace())
    fake_processor = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda text: [1, 2]),
    )

    monkeypatch.setattr(
        vlm_batch,
        "get_vlm_backend",
        lambda model_id, **kwargs: backend_calls.append(
            (
                model_id,
                kwargs.get("adapter_path"),
                kwargs.get("draft_model_id"),
                kwargs.get("surface"),
            )
        )
        or (fake_model, fake_processor),
    )
    monkeypatch.setattr(
        vlm_batch,
        "_vlm_apply_chat_template",
        lambda processor, config, messages, **kwargs: "formatted prompt",
    )
    monkeypatch.setattr(
        vlm_batch,
        "_vlm_batch_generate",
        lambda model,
        processor,
        *,
        images=None,
        prompts=None,
        max_tokens=128,
        **kwargs: SimpleNamespace(
            texts=["batched vision"],
            prompt_tokens=3,
            generation_tokens=2,
            total_tokens=5,
        ),
    )

    async def _run() -> None:
        coordinator = vlm_batch.VlmBatchCoordinator(
            "LibraxisAI/Qwen3-VL-30B",
            adapter_path="/adapter-a",
            draft_model_id="MLX-Community/Qwen3-1.7B-4bit",
            batch_window_ms=10,
            max_batch_size=4,
        )
        try:
            result = await coordinator.submit_request(
                messages=[{"role": "user", "content": "Describe this image"}],
                images=["https://example.com/cat.png"],
                max_tokens=12,
                temperature=None,
                top_p=None,
            )
            assert result.text == "batched vision"
        finally:
            await coordinator.shutdown()

    asyncio.run(_run())

    assert backend_calls == [
        (
            "libraxisai/qwen3-vl-30b",
            "/adapter-a",
            "mlx-community/qwen3-1.7b-4bit",
            "llm",
        )
    ]


def test_vlm_stream_state_attaches_llm_surface(monkeypatch) -> None:
    from mlx_batch_server.vision import vlm_batch

    backend_calls: list[tuple[str, str | None, str | None, str | None]] = []
    generator_calls: list[tuple[object, object, int, int]] = []

    class FakeBatchGenerator:
        def __init__(
            self,
            language_model,
            processor,
            *,
            prefill_batch_size: int,
            completion_batch_size: int,
            sampler,
        ) -> None:
            generator_calls.append(
                (
                    language_model,
                    processor,
                    prefill_batch_size,
                    completion_batch_size,
                )
            )

        def insert(self, input_ids_list, max_tokens):
            return [f"uid-{idx}" for idx, _ in enumerate(input_ids_list)]

    fake_model = SimpleNamespace(
        config=SimpleNamespace(),
        language_model=SimpleNamespace(name="tower"),
    )
    fake_processor = SimpleNamespace(
        tokenizer=SimpleNamespace(name="tokenizer"),
        detokenizer=SimpleNamespace(reset=lambda: None),
    )

    monkeypatch.setattr(
        vlm_batch,
        "get_settings",
        lambda: SimpleNamespace(
            vlm_batch_resize_shape=None,
            vlm_batch_pad_to_uniform_size=True,
        ),
    )
    monkeypatch.setattr(
        vlm_batch,
        "get_vlm_backend",
        lambda model_id, **kwargs: backend_calls.append(
            (
                model_id,
                kwargs.get("adapter_path"),
                kwargs.get("draft_model_id"),
                kwargs.get("surface"),
            )
        )
        or (fake_model, fake_processor),
    )
    monkeypatch.setattr(
        vlm_batch,
        "_require_vlm_stream_support",
        lambda: ("apply", FakeBatchGenerator, "prepare"),
    )
    monkeypatch.setattr(
        vlm_batch,
        "_prepare_stream_batch_inputs",
        lambda **kwargs: {
            "input_ids": [[1, 2]],
            "input_ids_list": [[1, 2]],
            "pixel_values": None,
            "data_kwargs": {},
            "max_tokens": 9,
        },
    )
    monkeypatch.setattr(
        vlm_batch,
        "_build_stream_gen_kwargs",
        lambda model, input_ids, pixel_values, data_kwargs: {"prepared": True},
    )

    state = vlm_batch._init_stream_batch_state(
        model_id="LibraxisAI/Qwen3-VL-30B",
        adapter_path="/adapter-a",
        draft_model_id="MLX-Community/Qwen3-1.7B-4bit",
        batch=[
            vlm_batch.PendingVlmStreamRequest(
                request_id="req-1",
                messages=[{"role": "user", "content": "Describe this image"}],
                images=["https://example.com/cat.png"],
                max_tokens=9,
                temperature=None,
                top_p=None,
                response_queue=None,
                created_at=0.0,
            )
        ],
    )

    assert backend_calls == [
        (
            "LibraxisAI/Qwen3-VL-30B",
            "/adapter-a",
            "MLX-Community/Qwen3-1.7B-4bit",
            "llm",
        )
    ]
    assert generator_calls == [(fake_model.language_model, fake_processor, 1, 1)]
    assert state["uids"] == ["uid-0"]


def test_vlm_stream_coordinator_groups_requests_by_image_shape() -> None:
    from mlx_batch_server.vision import vlm_batch

    coordinator = vlm_batch.VlmStreamBatchCoordinator(
        "LibraxisAI/Qwen3-VL-30B",
        batch_window_ms=10,
        max_batch_size=4,
        group_by_shape=True,
    )

    same_shape = Image.new("RGB", (32, 24), "red")
    different_shape = Image.new("RGB", (48, 48), "blue")

    grouped = coordinator._group_stream_requests(
        [
            vlm_batch.PendingVlmStreamRequest(
                request_id="req-a",
                messages=[{"role": "user", "content": "a"}],
                images=[same_shape],
                max_tokens=8,
                temperature=None,
                top_p=None,
                response_queue=asyncio.Queue(),
                created_at=0.0,
            ),
            vlm_batch.PendingVlmStreamRequest(
                request_id="req-b",
                messages=[{"role": "user", "content": "b"}],
                images=[different_shape],
                max_tokens=8,
                temperature=None,
                top_p=None,
                response_queue=asyncio.Queue(),
                created_at=0.0,
            ),
        ]
    )

    assert len(grouped) == 2
    assert {tuple(req.images[0].size for req in group) for group in grouped} == {
        ((32, 24),),
        ((48, 48),),
    }


def test_get_cache_info_filters_to_multimodal_runtime_surface(monkeypatch) -> None:
    from mlx_batch_server.vision import vlm_cache

    monkeypatch.setattr(
        vlm_cache.wrapper_cache,
        "get_cache_info",
        lambda: {
            "cache_size": 3,
            "max_size": 2,
            "ttl_seconds": 600,
            "runtime_keys": [
                {
                    "model_id": "mlx-community/qwen3-vl-30b",
                    "adapter_path": None,
                    "draft_model_id": None,
                },
                {
                    "model_id": "mlx-community/pixtral-12b-4bit",
                    "adapter_path": "/adapter-a",
                    "draft_model_id": None,
                },
                {
                    "model_id": "mlx-community/gpt-oss-20b",
                    "adapter_path": None,
                    "draft_model_id": None,
                },
            ],
            "vlm_cached_keys": [
                "mlx-community/qwen3-vl-30b",
                "mlx-community/pixtral-12b-4bit",
            ],
            "surface_runtime_attachments": [
                {
                    "model_id": "mlx-community/qwen3-vl-30b",
                    "adapter_path": None,
                    "draft_model_id": None,
                    "surfaces": ["visual"],
                },
                {
                    "model_id": "mlx-community/gpt-oss-20b",
                    "adapter_path": None,
                    "draft_model_id": None,
                    "surfaces": ["llm"],
                },
            ],
            "lru_order": [
                "WrapperCacheKey(model_id='mlx-community/qwen3-vl-30b', adapter_path=None, draft_model_id=None)",
                "WrapperCacheKey(model_id='mlx-community/gpt-oss-20b', adapter_path=None, draft_model_id=None)",
            ],
            "ttl_info": [
                {
                    "key": "WrapperCacheKey(model_id='mlx-community/qwen3-vl-30b', adapter_path=None, draft_model_id=None)",
                    "remaining_ttl_seconds": 300,
                    "expires_at": 1234,
                },
                {
                    "key": "WrapperCacheKey(model_id='mlx-community/gpt-oss-20b', adapter_path=None, draft_model_id=None)",
                    "remaining_ttl_seconds": 250,
                    "expires_at": 1200,
                },
            ],
        },
    )

    info = vlm_cache.get_cache_info()

    assert info["cache_size"] == 2
    assert info["runtime_cache_size"] == 2
    assert info["cached_keys"] == [
        "mlx-community/pixtral-12b-4bit",
        "mlx-community/qwen3-vl-30b",
    ]
    assert info["runtime_keys"] == [
        {
            "model_id": "mlx-community/qwen3-vl-30b",
            "adapter_path": None,
            "draft_model_id": None,
        },
        {
            "model_id": "mlx-community/pixtral-12b-4bit",
            "adapter_path": "/adapter-a",
            "draft_model_id": None,
        },
    ]
    assert info["surface_runtime_attachments"] == [
        {
            "model_id": "mlx-community/qwen3-vl-30b",
            "adapter_path": None,
            "draft_model_id": None,
            "surfaces": ["visual"],
        }
    ]
    assert info["lru_order"] == [
        "WrapperCacheKey(model_id='mlx-community/qwen3-vl-30b', adapter_path=None, draft_model_id=None)"
    ]


def test_get_loaded_vlm_models_is_explicit_alias(monkeypatch) -> None:
    from mlx_batch_server.vision import vlm_cache

    monkeypatch.setattr(
        vlm_cache.wrapper_cache,
        "get_loaded_vlm_models",
        lambda: ["mlx-community/qwen3-vl-30b", "mlx-community/pixtral-12b-4bit"],
    )

    assert vlm_cache.get_loaded_vlm_models() == [
        "mlx-community/pixtral-12b-4bit",
        "mlx-community/qwen3-vl-30b",
    ]


def test_vlm_execution_resolves_alias_scoped_runtime_key(monkeypatch) -> None:
    from mlx_batch_server.vision import vlm_cache

    expanded_adapter_path = str(Path("~/adapters/frontier-lora").expanduser())
    runtime_aliases_module.register_runtime_alias(
        "frontier-vlm",
        "LibraxisAI/Qwen3-VL-30B",
        adapter_path="~/adapters/frontier-lora",
        draft_model_id="LibraxisAI/Qwen3-1.7B-draft",
    )

    seen: list[tuple[str, str | None, str | None]] = []

    @contextmanager
    def fake_execution(
        model_id: str,
        *,
        adapter_path: str | None = None,
        draft_model_id: str | None = None,
    ):
        seen.append((model_id, adapter_path, draft_model_id))
        yield

    monkeypatch.setattr(
        vlm_cache.wrapper_cache,
        "vlm_execution",
        fake_execution,
    )

    with vlm_cache.vlm_execution("frontier-vlm"):
        pass

    assert seen == [
        (
            "libraxisai/qwen3-vl-30b",
            expanded_adapter_path,
            "libraxisai/qwen3-1.7b-draft",
        )
    ]
