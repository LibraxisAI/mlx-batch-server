"""Compatibility VLM cache surface backed by the unified runtime registry.

`mlx-batch-server` historically exposed a dedicated `vision.vlm_cache` module.
`mlx-batch-runner` moved the real residency ownership into `wrapper_cache` so
text, vision, and embeddings all share one runtime. This module restores the
VLM-facing API as a thin shim over the unified owner so both repositories can
converge on one contract without reintroducing split-brain caches.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, cast

from ..chat.mlx.runtime_aliases import (
    normalize_runtime_model_id,
    resolve_runtime_target,
)
from ..chat.mlx.wrapper_cache import (
    normalize_runtime_key,
    serialize_runtime_key,
    wrapper_cache,
)
from ..core.config import get_settings


def _normalized_pinned_models() -> set[str]:
    settings = get_settings()
    return {
        normalize_runtime_model_id(candidate)
        for candidate in settings.get_pinned_models()
    }


def _is_pinned_only_mode() -> bool:
    settings = get_settings()
    return settings.model_cache_max_size <= 0 and bool(_normalized_pinned_models())


def _enforce_pinned_only_vlm_guard(model_id: str) -> None:
    pinned = _normalized_pinned_models()
    if not _is_pinned_only_mode() or not pinned:
        return

    normalized = normalize_runtime_model_id(model_id)
    if normalized in pinned:
        return

    allowed = ", ".join(sorted(pinned))
    raise ValueError(
        f"VLM model '{model_id}' is not allowed in pinned-only mode. Allowed: {allowed}"
    )


def normalize_vlm_model_id(model_id: str) -> str:
    """Public canonicalizer for VLM model identifiers."""
    return normalize_runtime_model_id(model_id)


def resolve_vlm_model_id(model_id: str) -> str:
    """Resolve aliases and enforce pinned-only mode before VLM work begins."""
    target = resolve_runtime_target(model_id)
    _enforce_pinned_only_vlm_guard(target.model_id)
    return target.model_id


def is_vlm_loaded(
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> bool:
    """Return True when a shared multimodal runtime is already resident."""
    target = resolve_runtime_target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    loaded_models = set(wrapper_cache.get_loaded_vlm_models())
    if target.model_id not in loaded_models:
        return False

    if adapter_path is None and draft_model_id is None:
        return True

    runtime_key = normalize_runtime_key(
        target.model_id,
        adapter_path=target.adapter_path,
        draft_model_id=target.draft_model_id,
    )
    return runtime_key in wrapper_cache.get_runtime_keys()


def get_loaded_models() -> list[str]:
    """Return canonical multimodal model ids currently resident."""
    return sorted(wrapper_cache.get_loaded_vlm_models())


def get_loaded_vlm_models() -> list[str]:
    """Compatibility alias for callers expecting the explicit VLM naming."""
    return get_loaded_models()


def load_vlm_model(
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
    surface: str | None = None,
) -> tuple[Any, Any, bool]:
    """Load or reuse a shared VLM runtime through the unified registry."""
    target = resolve_runtime_target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    _enforce_pinned_only_vlm_guard(target.model_id)
    already_loaded = is_vlm_loaded(
        target.model_id,
        adapter_path=target.adapter_path,
        draft_model_id=target.draft_model_id,
    )
    model, processor = wrapper_cache.get_vlm_backend(
        target.model_id,
        adapter_path=target.adapter_path,
        draft_model_id=target.draft_model_id,
        surface=surface,
    )
    return model, processor, already_loaded


def get_vlm_backend(
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
    surface: str | None = None,
) -> tuple[Any, Any]:
    """Return the resident VLM backend without exposing wrapper_cache directly."""
    model, processor, _ = load_vlm_model(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
        surface=surface,
    )
    return model, processor


def unload_vlm_model(
    model_id: str | None = None,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> list[str]:
    """Unload one exact runtime key or all shared multimodal runtimes."""
    if model_id is None:
        return wrapper_cache.unload_vlm_model()

    target = resolve_runtime_target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    return wrapper_cache.unload_vlm_model(
        target.model_id,
        adapter_path=target.adapter_path,
        draft_model_id=target.draft_model_id,
    )


def clear_vlm_models() -> list[str]:
    """Compatibility helper mirroring the old dedicated VLM cache API."""
    return wrapper_cache.unload_vlm_model()


@contextmanager
def vlm_execution(
    model_id: str,
    *,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
):
    """Serialize VLM execution on the canonical shared runtime identity."""
    target = resolve_runtime_target(
        model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    with wrapper_cache.vlm_execution(
        target.model_id,
        adapter_path=target.adapter_path,
        draft_model_id=target.draft_model_id,
    ):
        yield


def get_cache_info() -> dict[str, Any]:
    """Return a VLM-filtered view of unified runtime cache state."""
    info = wrapper_cache.get_cache_info()
    cached_models: list[str] = sorted(
        cast("list[str]", info.get("vlm_cached_keys", []))
        or list(wrapper_cache.get_loaded_vlm_models())
    )
    surface_runtime_attachments = cast(
        "list[dict[str, Any]]",
        info.get("surface_runtime_attachments", []),
    )
    runtime_key_items = cast("list[dict[str, Any]]", info.get("runtime_keys", [])) or [
        serialize_runtime_key(key)
        for key in wrapper_cache.get_runtime_keys()
        if key.model_id in cached_models
    ]
    lru_entries = cast("list[str]", info.get("lru_order", []))
    ttl_entries = cast("list[dict[str, Any]]", info.get("ttl_info", []))
    surface_attachments = [
        item
        for item in surface_runtime_attachments
        if item.get("model_id") in cached_models
    ]
    runtime_keys = [
        item for item in runtime_key_items if item.get("model_id") in cached_models
    ]
    lru_order = [
        item
        for item in lru_entries
        if any(f"model_id='{model_id}'" in item for model_id in cached_models)
    ]
    ttl_info = [
        item
        for item in ttl_entries
        if any(
            f"model_id='{model_id}'" in item.get("key", "")
            for model_id in cached_models
        )
    ]

    return {
        "cache_size": len(cached_models),
        "runtime_cache_size": len(runtime_keys),
        "max_size": info.get("max_size", 0),
        "ttl_seconds": info.get("ttl_seconds", 0),
        "cached_keys": cached_models,
        "runtime_keys": runtime_keys,
        "surface_runtime_attachments": surface_attachments,
        "lru_order": lru_order,
        "ttl_info": ttl_info,
    }


__all__ = [
    "clear_vlm_models",
    "get_cache_info",
    "get_loaded_models",
    "get_loaded_vlm_models",
    "get_vlm_backend",
    "is_vlm_loaded",
    "load_vlm_model",
    "normalize_vlm_model_id",
    "resolve_vlm_model_id",
    "unload_vlm_model",
    "vlm_execution",
]
