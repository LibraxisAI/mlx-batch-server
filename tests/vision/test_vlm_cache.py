"""Regression tests for the VLM cache compatibility shim."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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
