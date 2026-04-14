"""Endpoint runtime policy helpers for MLX-backed LLM/VLM endpoints."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from ...batch.coordinator import get_loaded_batch_models, shutdown_all_coordinators
from ...utils.logger import logger

try:
    from ...vision.vlm_batch import (
        get_loaded_vlm_batch_models,
        shutdown_all_vlm_coordinators,
    )
except ImportError:
    # `mlx-batch-runner` does not always ship the sibling repo's vision batch lane.
    # Keep runtime policy usable here while automatically picking up the richer path
    # when `vision.vlm_batch` is present in repos that provide it.
    def get_loaded_vlm_batch_models() -> list[str]:
        return []

    async def shutdown_all_vlm_coordinators() -> None:
        return None


from .wrapper_cache import (
    WrapperCacheKey,
    normalize_model_id,
    normalize_runtime_key,
    wrapper_cache,
)

_endpoint_runtime_condition = asyncio.Condition()
_active_runtime_key: WrapperCacheKey | None = None
_active_runtime_count = 0


def _normalized_batch_models() -> list[str]:
    return sorted(
        {
            normalize_model_id(model_id)
            for model_id in [
                *get_loaded_batch_models(),
                *get_loaded_vlm_batch_models(),
            ]
        }
    )


def _requires_hard_switch(
    target_key: WrapperCacheKey,
    runtime_keys: list[WrapperCacheKey],
    batch_models: list[str],
) -> bool:
    if not runtime_keys and not batch_models:
        return False

    if len(runtime_keys) == 1 and runtime_keys[0] == target_key:
        return set(batch_models).difference({target_key.model_id}) != set()

    return True


async def ensure_single_endpoint_llm_runtime(
    model_id: str,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
) -> dict[str, Any]:
    """Hard-enforce single-model residency for LLM/VLM endpoints."""
    target_key = normalize_runtime_key(
        model_id=model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )
    runtime_keys = wrapper_cache.get_runtime_keys()
    batch_models = _normalized_batch_models()

    if not _requires_hard_switch(target_key, runtime_keys, batch_models):
        return {
            "switched": False,
            "target_model_id": target_key.model_id,
            "previous_runtime_keys": runtime_keys,
            "previous_batch_models": batch_models,
            "evicted_models": [],
        }

    previous_runtime_keys = runtime_keys
    previous_batch_models = batch_models
    evicted_models = sorted({key.model_id for key in runtime_keys})

    logger.warning(
        "Hard-switching endpoint runtime to %s (adapter=%s, draft=%s); "
        "evicting prior llm runtimes=%s batch=%s",
        target_key.model_id,
        target_key.adapter_path,
        target_key.draft_model_id,
        [str(key) for key in runtime_keys],
        batch_models,
    )

    await shutdown_all_coordinators()
    await shutdown_all_vlm_coordinators()

    for loaded_model_id in list(wrapper_cache.get_loaded_models()):
        wrapper_cache.unload_model(loaded_model_id)

    return {
        "switched": True,
        "target_model_id": target_key.model_id,
        "previous_runtime_keys": previous_runtime_keys,
        "previous_batch_models": previous_batch_models,
        "evicted_models": evicted_models,
    }


@asynccontextmanager
async def endpoint_runtime_session(
    model_id: str,
    adapter_path: str | None = None,
    draft_model_id: str | None = None,
):
    """Reserve the endpoint for one runtime key while a request is in flight."""
    global _active_runtime_key, _active_runtime_count

    target_key = normalize_runtime_key(
        model_id=model_id,
        adapter_path=adapter_path,
        draft_model_id=draft_model_id,
    )

    async with _endpoint_runtime_condition:
        while _active_runtime_key is not None and _active_runtime_key != target_key:
            await _endpoint_runtime_condition.wait()

        switch_result = {
            "switched": False,
            "target_model_id": target_key.model_id,
            "previous_runtime_keys": [],
            "previous_batch_models": [],
            "evicted_models": [],
        }
        if _active_runtime_key is None:
            switch_result = await ensure_single_endpoint_llm_runtime(
                model_id=model_id,
                adapter_path=adapter_path,
                draft_model_id=draft_model_id,
            )
            _active_runtime_key = target_key
            _active_runtime_count = 1
        else:
            _active_runtime_count += 1

    try:
        yield switch_result
    finally:
        async with _endpoint_runtime_condition:
            if _active_runtime_key == target_key:
                _active_runtime_count -= 1
                if _active_runtime_count == 0:
                    _active_runtime_key = None
                    _endpoint_runtime_condition.notify_all()
